"""
HDR_model_hybrid_Teacher.py — HDR RAW denoisers for BGGR packed Bayer input
=============================================================================

All models expect a (B, 4, H, W) packed Bayer tensor in [0, 1] with channel
order B, G1, G2, R (one channel per position of the 2x2 BGGR cell) and H, W
divisible by 8 (three PixelUnshuffle(2) stages).

Contents
────────
  TransUNet_Teacher_HDR   Heavy CNN-Transformer U-Net (single denoiser).
                          `se_reduction` adds Squeeze-Excitation to the
                          residual blocks (None = legacy architecture, so
                          old checkpoints still load).

  MoEDenoiser             Semi-lightweight Mixture-of-Experts:
                            * ONE shared trunk (encoder + Restormer latent
                              + decoder) — this is where ~95% of compute is.
                            * K lightweight expert heads, each predicting a
                              full-res residual. Experts specialise on
                              different noise levels.
                            * A tiny per-pixel gating CNN conditioned on the
                              noisy input + local SNR map produces softmax
                              weights over the K experts.
                          vs. DualSNRDenoiser (two full teachers):
                            ~2x fewer parameters and ~2x fewer FLOPs,
                            while supporting K>=2 noise-level experts.

  DualSNRDenoiser         Legacy 2-expert variant: two full teachers blended
                          by the SNR map (kept for old checkpoints).
  SingleDenoiser          Ablation baseline (one teacher).

  All three wrappers share one forward signature:
      blended, expert_outs, gates = model(x, snr_map)
        blended:     [B, 4, H, W]   final output
        expert_outs: [B, K, 4, H, W] per-expert outputs (for aux losses)
        gates:       [B, K, H, W]   per-pixel routing weights (sum to 1)

  build_denoiser(mode, ...)  factory: mode in {"moe", "dual", "single"}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the heavy attention blocks from the existing repository
from blocks_Restormer import RestormerBlock

# Lowest representable value: keeps the µ-law tonemap log argument positive.
_CLAMP_EPS = 1.0 / (2 ** 20 - 1)


# ---------------------------------------------------------
# 0. Shared utilities
# ---------------------------------------------------------

def estimate_local_snr_map(x, window_size=5, eps=1e-5):
    """
    Per-pixel SNR estimate from a noisy image: local mean / local std in a
    window, averaged over channels, normalised to [0, 1] per image.
    Canonical implementation — train and test scripts import this so the
    routing signal can never diverge between them.

    x: [B, C, H, W] or [C, H, W]   ->   [B, 1, H, W] or [1, H, W]
    """
    unbatched = x.dim() == 3
    if unbatched:
        x = x.unsqueeze(0)
    pad           = window_size // 2
    local_mean    = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    local_sq_mean = F.avg_pool2d(x * x, window_size, stride=1, padding=pad)
    local_var     = torch.clamp(local_sq_mean - local_mean ** 2, min=0.0)
    local_std     = torch.sqrt(local_var + eps)
    spatial_snr   = (local_mean / local_std).mean(dim=1, keepdim=True)
    image_max     = spatial_snr.amax(dim=(2, 3), keepdim=True)
    snr_norm      = spatial_snr / (image_max + eps)
    if unbatched:
        snr_norm = snr_norm.squeeze(0)
    return snr_norm


# ---------------------------------------------------------
# 1. CNN blocks
# ---------------------------------------------------------

class SqueezeExcite(nn.Module):
    """Channel recalibration — cheap global adaptation to the noise level."""
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ResidualConvBlock(nn.Module):
    """
    Standard residual block used throughout the CNN paths.
    se_reduction=None reproduces the legacy block exactly (old checkpoints
    load unchanged); an int enables Squeeze-Excitation.
    """
    def __init__(self, dim, se_reduction=None):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.act2 = nn.GELU()
        self.se = SqueezeExcite(dim, se_reduction) if se_reduction else nn.Identity()

    def forward(self, x):
        res = x
        x = self.act1(self.conv1(x))
        x = self.conv2(x)
        x = self.se(x)
        return self.act2(res + x)


# Name referenced by the train/test configs ("required by SEResidualBlock").
SEResidualBlock = ResidualConvBlock


class HeavyExposhare(nn.Module):
    """
    Dense conv stack that aligns features across the channel dimension
    (exposure sharing between the four Bayer planes).
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        z = self.act(self.conv1(x))
        z = self.act(self.conv2(z))
        z = self.conv3(z)
        return res + z


# ---------------------------------------------------------
# 2. The TransUNet Teacher Model
# ---------------------------------------------------------

class TransUNet_Teacher_HDR(nn.Module):
    def __init__(self, out_channels=4, dim=32, num_blocks=(4, 4, 4, 6),
                 num_refinement_blocks=4, heads=(1, 2, 4, 8), se_reduction=None):
        """
        dim=32, num_blocks=[4,4,4,6], heads=[1,2,4,8] -> ~19.5M parameters.
        Expects (B, 4, H, W) packed Bayer input (B, G1, G2, R channels).
        se_reduction: None = legacy blocks (old checkpoints load);
                      int  = enable Squeeze-Excitation in residual blocks.
        """
        super().__init__()
        self.dim = dim

        # ======== First Layer (Patch Embedding) ========
        # PixelUnshuffle(2) on 4-ch packed input -> 16 channels at H/2, W/2
        self.bayer_unshuffle = nn.PixelUnshuffle(2)
        self.patch_embed = nn.Conv2d(16, dim, kernel_size=3, stride=1, padding=1)

        # ======== CNN Encoder ========
        self.encoder_level_1 = nn.Sequential(*[
            ResidualConvBlock(dim, se_reduction) for _ in range(num_blocks[0])
        ])
        self.x_expo_1 = HeavyExposhare(dim)

        self.down_unshuffle_1_2 = nn.PixelUnshuffle(2)
        self.encoder_level_2 = nn.Sequential(*[
            ResidualConvBlock(dim * 4, se_reduction) for _ in range(num_blocks[1])
        ])
        self.x_expo_2 = HeavyExposhare(dim * 4)

        # ======== Transformer Bottleneck (Latent) ========
        self.down_unshuffle_2_3 = nn.PixelUnshuffle(2)
        self.latent = nn.Sequential(*[
            RestormerBlock(dim=dim * 16, num_heads=heads[3])
            for _ in range(num_blocks[3])
        ])
        self.latent_fusion = nn.Conv2d(dim * 16, dim * 16, kernel_size=1)

        # ======== CNN Decoder ========
        self.up_shuffle_3_2 = nn.PixelShuffle(2)
        self.decoder_level_2 = nn.Sequential(*[
            ResidualConvBlock(dim * 8, se_reduction) for _ in range(num_blocks[2])
        ])
        self.reduce_chan_level_2 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)

        self.up_shuffle_2_1 = nn.PixelShuffle(2)
        self.decoder_level_1 = nn.Sequential(*[
            ResidualConvBlock(dim * 2, se_reduction) for _ in range(num_blocks[0])
        ])

        # Level 0 Refinement: Full Resolution
        self.decoder_level_0 = nn.Sequential(*[
            ResidualConvBlock(dim * 2, se_reduction) for _ in range(num_refinement_blocks)
        ])
        self.up_shuffle_1_0 = nn.PixelShuffle(2)

        # Final Refinement
        self.refinement_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)
        self.act_final = nn.GELU()
        self.refinement_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)
        self.reduce_chan_final = nn.Conv2d(dim // 2, out_channels, kernel_size=1)

    def forward(self, x, return_maps=False):
        # --- Encoder ---
        x_shuffled = self.bayer_unshuffle(x)          # (B, 16, H/2, W/2)
        x_level1 = self.patch_embed(x_shuffled)

        x_level1 = self.encoder_level_1(x_level1)
        x_level1 = self.x_expo_1(x_level1)

        x_level2 = self.down_unshuffle_1_2(x_level1)
        x_level2 = self.encoder_level_2(x_level2)
        x_level2 = self.x_expo_2(x_level2)

        # --- Transformer Latent ---
        x_latent = self.down_unshuffle_2_3(x_level2)
        x_latent = self.latent(x_latent)
        x_latent = self.latent_fusion(x_latent)

        # --- Decoder ---
        w_level2 = self.up_shuffle_3_2(x_latent)
        w_level2 = torch.cat([w_level2, x_level2], dim=1)
        w_level2 = self.decoder_level_2(w_level2)
        w_level2 = self.reduce_chan_level_2(w_level2)

        w_level1 = self.up_shuffle_2_1(w_level2)
        w_level1 = torch.cat([w_level1, x_level1], dim=1)
        w_level1 = self.decoder_level_1(w_level1)

        w_level0 = self.decoder_level_0(w_level1)
        w_level0 = self.up_shuffle_1_0(w_level0)

        # --- Refinement ---
        w_level0 = self.act_final(self.refinement_conv1(w_level0))
        w_level0 = self.refinement_conv2(w_level0)

        # Save map for KD loss before final channel reduction
        w_level0_refined_map = w_level0

        w_out = self.reduce_chan_final(w_level0) + x
        # Floor keeps the µ-law log argument positive. The ceiling is applied
        # only in eval mode: clamping during training zeroes the gradient on
        # over-shot highlight pixels, which stalls learning in saturated
        # regions (the µ-law loss handles values > 1 without it).
        output = w_out.clamp(min=_CLAMP_EPS)
        if not self.training:
            output = output.clamp(max=1.0)

        if return_maps:
            return output, {
                "x_level1": x_level1,
                "x_level2": x_level2,
                "x_latent": x_latent,
                "w_level2": w_level2,
                "w_level1": w_level1,
                "w_level0_refined": w_level0_refined_map
            }

        return output


# ---------------------------------------------------------
# 3. Mixture-of-Experts denoiser (shared trunk + light expert heads)
# ---------------------------------------------------------

class NoiseGate(nn.Module):
    """
    Tiny per-pixel router. Sees the noisy input (absolute intensity — shot
    noise scales with signal) and the local SNR map (relative noisiness) and
    outputs softmax weights over the K experts at full resolution.
    The final conv is zero-initialised so training starts from uniform
    routing (1/K everywhere) instead of an arbitrary expert winning early.
    """
    def __init__(self, num_experts, in_channels=5, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, num_experts, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, snr_map):
        logits = self.net(torch.cat([x, snr_map], dim=1))
        return torch.softmax(logits, dim=1)          # [B, K, H, W]


class ExpertHead(nn.Module):
    """
    Lightweight per-expert decoder head (~150K params at dim=32):
    a couple of residual blocks at H/2, pixel-shuffle to full resolution,
    small refinement, and a zero-initialised 1x1 residual projection
    (each expert starts as the identity — important under heavy noise).
    """
    def __init__(self, in_dim, out_channels=4, num_blocks=2, se_reduction=8):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualConvBlock(in_dim, se_reduction) for _ in range(num_blocks)
        ])
        self.up = nn.PixelShuffle(2)                 # in_dim -> in_dim/4, full res
        r = in_dim // 4
        self.refine1 = nn.Conv2d(r, r, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.refine2 = nn.Conv2d(r, r, kernel_size=3, padding=1)
        self.proj_out = nn.Conv2d(r, out_channels, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, feat):
        z = self.blocks(feat)
        z = self.up(z)
        z = self.act(self.refine1(z))
        z = self.refine2(z)
        return self.proj_out(z)                      # full-res residual


class MoEDenoiser(nn.Module):
    """
    Semi-lightweight Mixture-of-Experts HDR RAW denoiser.

    One shared trunk (encoder -> Restormer latent -> decoder, identical
    topology to TransUNet_Teacher_HDR through decoder_level_1) feeds K
    lightweight expert heads. A per-pixel gate conditioned on the noisy
    input + SNR map blends the expert outputs, so different experts can
    own different noise levels (dark/high-noise vs bright/low-noise pixels)
    without duplicating the expensive trunk per expert.

    At the default size (dim=32, num_blocks=[4,4,4,4], K=3) this is ~21M
    parameters and ~1.05x the FLOPs of a single teacher — roughly half of
    DualSNRDenoiser (two full teachers, ~41M, 2x FLOPs) with one extra
    expert.

    forward(x, snr_map) -> (blended, expert_outs, gates)
        x:        [B, 4, H, W] packed BGGR Bayer in [0, 1], H, W % 8 == 0
        snr_map:  [B, 1, H, W] normalised local SNR in [0, 1]
    """
    def __init__(self, out_channels=4, dim=32, num_blocks=(4, 4, 4, 4),
                 num_refinement_blocks=4, heads=(1, 2, 4, 8), se_reduction=8,
                 num_experts=3, expert_blocks=2, gate_hidden=16):
        super().__init__()
        # num_refinement_blocks is accepted for config compatibility with the
        # other wrappers; in the MoE the refinement stage lives inside the
        # expert heads (expert_blocks controls their depth).
        del num_refinement_blocks
        self.dim = dim
        self.num_experts = num_experts

        # ======== Shared trunk (same topology as the teacher) ========
        self.bayer_unshuffle = nn.PixelUnshuffle(2)
        self.patch_embed = nn.Conv2d(16, dim, kernel_size=3, stride=1, padding=1)

        self.encoder_level_1 = nn.Sequential(*[
            ResidualConvBlock(dim, se_reduction) for _ in range(num_blocks[0])
        ])
        self.x_expo_1 = HeavyExposhare(dim)

        self.down_unshuffle_1_2 = nn.PixelUnshuffle(2)
        self.encoder_level_2 = nn.Sequential(*[
            ResidualConvBlock(dim * 4, se_reduction) for _ in range(num_blocks[1])
        ])
        self.x_expo_2 = HeavyExposhare(dim * 4)

        self.down_unshuffle_2_3 = nn.PixelUnshuffle(2)
        self.latent = nn.Sequential(*[
            RestormerBlock(dim=dim * 16, num_heads=heads[3])
            for _ in range(num_blocks[3])
        ])
        self.latent_fusion = nn.Conv2d(dim * 16, dim * 16, kernel_size=1)

        self.up_shuffle_3_2 = nn.PixelShuffle(2)
        self.decoder_level_2 = nn.Sequential(*[
            ResidualConvBlock(dim * 8, se_reduction) for _ in range(num_blocks[2])
        ])
        self.reduce_chan_level_2 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)

        self.up_shuffle_2_1 = nn.PixelShuffle(2)
        self.decoder_level_1 = nn.Sequential(*[
            ResidualConvBlock(dim * 2, se_reduction) for _ in range(num_blocks[0])
        ])

        # ======== Experts + gate ========
        self.experts = nn.ModuleList([
            ExpertHead(dim * 2, out_channels, expert_blocks, se_reduction)
            for _ in range(num_experts)
        ])
        self.gate = NoiseGate(num_experts, in_channels=out_channels + 1,
                              hidden=gate_hidden)

    def _trunk(self, x):
        x_level1 = self.patch_embed(self.bayer_unshuffle(x))
        x_level1 = self.x_expo_1(self.encoder_level_1(x_level1))

        x_level2 = self.down_unshuffle_1_2(x_level1)
        x_level2 = self.x_expo_2(self.encoder_level_2(x_level2))

        x_latent = self.down_unshuffle_2_3(x_level2)
        x_latent = self.latent_fusion(self.latent(x_latent))

        w_level2 = torch.cat([self.up_shuffle_3_2(x_latent), x_level2], dim=1)
        w_level2 = self.reduce_chan_level_2(self.decoder_level_2(w_level2))

        w_level1 = torch.cat([self.up_shuffle_2_1(w_level2), x_level1], dim=1)
        return self.decoder_level_1(w_level1)         # [B, 2*dim, H/2, W/2]

    def forward(self, x, snr_map):
        feat = self._trunk(x)

        # Per-expert full-res outputs: identity + learned residual
        expert_outs = torch.stack(
            [(x + head(feat)).clamp(min=_CLAMP_EPS) for head in self.experts],
            dim=1)                                    # [B, K, 4, H, W]

        gates = self.gate(x, snr_map)                 # [B, K, H, W]
        blended = (gates.unsqueeze(2) * expert_outs).sum(dim=1)

        if not self.training:
            blended = blended.clamp(max=1.0)
            expert_outs = expert_outs.clamp(max=1.0)

        return blended, expert_outs, gates


# ---------------------------------------------------------
# 4. Legacy wrappers (unified return signature)
# ---------------------------------------------------------

class DualSNRDenoiser(nn.Module):
    """
    Two independent full teachers blended by the pixel-wise SNR map:
        out = (1 - snr) * out_low + snr * out_high
    Heavy (2x teacher params and FLOPs) — kept for existing checkpoints.
    Returns the unified (blended, expert_outs, gates) tuple where
    gates = [1 - snr, snr].
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.num_experts = 2
        self.denoiser_low_snr = TransUNet_Teacher_HDR(**kwargs)
        self.denoiser_high_snr = TransUNet_Teacher_HDR(**kwargs)

    def forward(self, x, snr_map):
        out_low = self.denoiser_low_snr(x)
        out_high = self.denoiser_high_snr(x)
        blended = (1.0 - snr_map) * out_low + snr_map * out_high
        expert_outs = torch.stack([out_low, out_high], dim=1)
        gates = torch.cat([1.0 - snr_map, snr_map], dim=1)
        return blended, expert_outs, gates


# Backward-compatible alias (older scripts referenced this name).
DualSNRTeacher = DualSNRDenoiser


class SingleDenoiser(nn.Module):
    """Baseline: one denoiser, no routing. Same return signature."""
    def __init__(self, **kwargs):
        super().__init__()
        self.num_experts = 1
        self.denoiser = TransUNet_Teacher_HDR(**kwargs)

    def forward(self, x, snr_map):
        out = self.denoiser(x)
        gates = torch.ones_like(snr_map)
        return out, out.unsqueeze(1), gates


def build_denoiser(mode, num_experts=3, expert_blocks=2, gate_hidden=16, **kwargs):
    """
    Factory shared by the train and test scripts.
    mode: "moe" (shared trunk + K light experts)  |  "dual"  |  "single"
    kwargs are forwarded to the underlying model(s):
        out_channels, dim, num_blocks, num_refinement_blocks, heads,
        se_reduction (None for legacy pre-SE checkpoints).
    """
    mode = mode.lower()
    if mode == "moe":
        return MoEDenoiser(num_experts=num_experts, expert_blocks=expert_blocks,
                           gate_hidden=gate_hidden, **kwargs)
    if mode == "dual":
        return DualSNRDenoiser(**kwargs)
    if mode == "single":
        return SingleDenoiser(**kwargs)
    raise ValueError(f"Unknown denoiser mode '{mode}' (use moe | dual | single)")
