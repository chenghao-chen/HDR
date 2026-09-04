"""
HDR_model_hybrid_Teacher.py — Joint denoising + demosaicing for BGGR packed Bayer
===================================================================================

Input / Output
──────────────
  Input  (B, 4, H, W)   packed BGGR Bayer in [0, 1], channel order B, G1, G2, R.
                         H, W are the PACKED dimensions (= sensor_H/2, sensor_W/2)
                         and must each be divisible by 8 (three PixelUnshuffle(2)
                         stages in the encoder).

  Output (B, 3, 2H, 2W) clean RGB at full sensor resolution.
                         The final PixelShuffle(2) stage performs the 2× spatial
                         upsampling (demosaicing) inside the network, so no
                         external GBTF step is needed for the model output.

Contents
────────
  TransUNet_Teacher_HDR   Heavy CNN-Transformer U-Net (single joint-DD network).
                          `se_reduction` adds Squeeze-Excitation to residual
                          blocks (None = legacy architecture; old checkpoints
                          still load).

  MoEDenoiser             Semi-lightweight Mixture-of-Experts:
                            * ONE shared trunk (encoder → Restormer latent →
                              decoder) — ~95% of compute.
                            * K lightweight expert heads, each producing a
                              full-sensor-resolution RGB image.
                            * A tiny per-pixel gate CNN conditioned on the
                              noisy BGGR input + local SNR map routes softmax
                              weights over the K experts.

  DualSNRDenoiser         Legacy 2-expert variant: two full teachers blended
                          by the SNR map (kept for old checkpoints).
  SingleDenoiser          Ablation baseline (one teacher).

  All three wrappers share one forward signature:
      blended, expert_outs, gates = model(x, snr_map)
        blended:     [B, 3, 2H, 2W]    final RGB output at sensor resolution
        expert_outs: [B, K, 3, 2H, 2W] per-expert RGB outputs
        gates:       [B, K, 2H, 2W]    per-pixel routing weights (sum to 1)

  build_denoiser(mode, ...)  factory: mode in {"moe", "dual", "single"}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks_Restormer import RestormerBlock

_CLAMP_EPS = 1.0 / (2 ** 20 - 1)

# Initial output level of a fresh ExpertHead. Two constraints:
#
#   * strictly above _CLAMP_EPS, with margin. The expert heads have no
#     residual path around proj_out, so whatever proj_out emits IS the
#     prediction, and a prediction at or below the clamp floor has exactly
#     zero gradient. This value is ~5000x the floor.
#   * near the data's own scale, so training does not spend its first epochs
#     just correcting a global brightness offset. 0.005 is the measured mean
#     linear intensity of the Mobile-HDR training split (HDR linear data is
#     mostly dark; the mu-law mean is a much larger 0.25).
#
# The exact value is not critical — anything well clear of the floor and
# roughly at the data scale behaves the same.
_INIT_OUT_LEVEL = 0.005

# Multiplicative half-range spread applied across a MoE's K expert heads at
# construction (see MoEDenoiser.__init__). Two heads drawn from the same
# init distribution have no reason to specialise: the router's only signal
# to prefer one over the other is how differently they already perform,
# which starts at "not at all", so training spends its early steps doing
# nothing useful with K experts instead of one. Verified empirically (120
# training steps on real Mobile-HDR crops): with identical init, gate
# weights across the image stayed within a 0.001-wide band regardless of
# the SNR map they were conditioned on; staggering each head's output
# level by its index widened that to a 0.01-0.04-wide band correlated with
# SNR. The staggering is multiplicative, not additive, so bias stays
# positive (and clear of _CLAMP_EPS) for every expert at any K.
_EXPERT_INIT_SPREAD = 0.4


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
    def __init__(self, out_channels=3, dim=32, num_blocks=(4, 4, 4, 6),
                 num_refinement_blocks=4, heads=(1, 2, 4, 8), se_reduction=None):
        """
        dim=32, num_blocks=[4,4,4,6], heads=[1,2,4,8] -> ~19.5M parameters.
        Input:  (B, 4, H, W) packed BGGR in [0, 1], H and W divisible by 8.
        Output: (B, out_channels=3, 2H, 2W) clean RGB at sensor resolution.

        se_reduction: None = legacy blocks (old checkpoints load unchanged);
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

        # Level 0 Refinement at H/2, W/2 (half of packed Bayer resolution)
        self.decoder_level_0 = nn.Sequential(*[
            ResidualConvBlock(dim * 2, se_reduction) for _ in range(num_refinement_blocks)
        ])
        # Upsample back to packed Bayer resolution H, W
        self.up_shuffle_1_0 = nn.PixelShuffle(2)     # dim*2 -> dim//2, ×2 spatial

        # Refinement at packed Bayer resolution (H, W)
        self.refinement_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)
        self.act_final = nn.GELU()
        self.refinement_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1)

        # Upsample to full sensor resolution (2H, 2W) and project to RGB:
        # Conv2d maps dim//2 -> out_channels*4, then PixelShuffle(2) gives
        # out_channels channels at 2H, 2W (demosaicing upsampling).
        self.to_rgb = nn.Conv2d(dim // 2, out_channels * 4, kernel_size=1)
        self.upshuffle_rgb = nn.PixelShuffle(2)

    def forward(self, x, return_maps=False):
        # --- Encoder ---
        x_shuffled = self.bayer_unshuffle(x)          # (B, 16, H/2, W/2)
        x_level1 = self.patch_embed(x_shuffled)       # (B, dim, H/2, W/2)

        x_level1 = self.encoder_level_1(x_level1)
        x_level1 = self.x_expo_1(x_level1)

        x_level2 = self.down_unshuffle_1_2(x_level1)  # (B, dim*4, H/4, W/4)
        x_level2 = self.encoder_level_2(x_level2)
        x_level2 = self.x_expo_2(x_level2)

        # --- Transformer Latent ---
        x_latent = self.down_unshuffle_2_3(x_level2)  # (B, dim*16, H/8, W/8)
        x_latent = self.latent(x_latent)
        x_latent = self.latent_fusion(x_latent)

        # --- Decoder ---
        w_level2 = self.up_shuffle_3_2(x_latent)      # (B, dim*4, H/4, W/4)
        w_level2 = torch.cat([w_level2, x_level2], dim=1)
        w_level2 = self.decoder_level_2(w_level2)
        w_level2 = self.reduce_chan_level_2(w_level2)  # (B, dim*4, H/4, W/4)

        w_level1 = self.up_shuffle_2_1(w_level2)      # (B, dim, H/2, W/2)
        w_level1 = torch.cat([w_level1, x_level1], dim=1)
        w_level1 = self.decoder_level_1(w_level1)     # (B, dim*2, H/2, W/2)

        w_level0 = self.decoder_level_0(w_level1)     # (B, dim*2, H/2, W/2)
        w_level0 = self.up_shuffle_1_0(w_level0)      # (B, dim//2, H, W)

        # --- Refinement at packed Bayer resolution ---
        w_level0 = self.act_final(self.refinement_conv1(w_level0))
        w_level0 = self.refinement_conv2(w_level0)    # (B, dim//2, H, W)

        # --- Upsample to sensor resolution + project to RGB ---
        # to_rgb: dim//2 -> out_channels*4, upshuffle_rgb: ×2 spatial -> (B, 3, 2H, 2W)
        output = self.upshuffle_rgb(self.to_rgb(w_level0))
        output = output.clamp(min=_CLAMP_EPS)
        if not self.training:
            output = output.clamp(max=1.0)

        if return_maps:
            return output, {
                "x_level1": x_level1,
                "x_level2": x_level2,
                "x_latent": x_latent,
                "w_level2": w_level2,
                "w_level1": w_level1,
                "w_level0_refined": w_level0,
            }

        return output   # (B, 3, 2H, 2W)  — sensor-resolution RGB


# ---------------------------------------------------------
# 3. Mixture-of-Experts denoiser (shared trunk + light expert heads)
# ---------------------------------------------------------

class NoiseGate(nn.Module):
    """
    Tiny per-pixel router. Sees the noisy BGGR input (4 channels, absolute
    intensity — shot noise scales with signal) and the local SNR map (1 channel,
    relative noisiness) and outputs softmax weights over the K experts at the
    input (packed Bayer) spatial resolution.

    The gate output is upsampled 2× in MoEDenoiser.forward before being applied
    to the expert outputs which are at full sensor resolution.

    in_channels is always 5: 4 BGGR Bayer channels + 1 SNR channel.

    The final conv is initialised small (std 1e-3) rather than exactly zero.
    Exact zeros give uniform routing at step 0, which is what we want, but they
    also make d(logits)/d(hidden) identically zero, so the two hidden convs sit
    frozen until the final layer drifts off zero. A small non-zero scale starts
    routing effectively uniform (logit spread ~1e-3 => gates within 0.1% of 1/K)
    while keeping the whole gate trainable from the first step.
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
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, snr_map):
        logits = self.net(torch.cat([x, snr_map], dim=1))
        return torch.softmax(logits, dim=1)   # [B, K, H, W]  at Bayer resolution


class ExpertHead(nn.Module):
    """
    Lightweight per-expert decoder head (~150K params at dim=32).

    Takes trunk features at half packed-Bayer resolution (H/2, W/2), applies
    residual blocks, then two PixelShuffle(2) stages:
      1st: H/2 -> H  (back to packed Bayer resolution)
      2nd: H   -> 2H (sensor resolution — the demosaicing upsampling)
    The final 1×1 conv starts as a small-variance perturbation around a dim
    positive constant (_INIT_OUT_LEVEL), so a fresh expert predicts a nearly
    uniform dark image rather than noise.

    That keeps the original "each expert starts from a neutral prediction"
    intent, but strictly ABOVE the output floor. proj_out used to be
    zero-initialised, which put the raw head output at exactly 0.0 — and
    MoEDenoiser.forward clamps to min=_CLAMP_EPS, where clamp's backward is
    zero. Every output element sat on the floor, so no gradient reached any
    parameter in the model, proj_out included, and it could not bootstrap out
    at any learning rate. Plain default init fixes the gradient but is
    zero-mean, which leaves ~50% of output pixels on the floor at step 0; the
    positive bias puts essentially all of them above it.
    See tests/test_model_moe.py::test_fresh_model_produces_nonzero_gradients.
    """
    def __init__(self, in_dim, out_channels=3, num_blocks=2, se_reduction=8):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualConvBlock(in_dim, se_reduction) for _ in range(num_blocks)
        ])
        self.up = nn.PixelShuffle(2)                  # in_dim -> in_dim//4, ×2 spatial
        r = in_dim // 4                               # = dim//2 = 16 at default dim=32
        self.refine1 = nn.Conv2d(r, r, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.refine2 = nn.Conv2d(r, r, kernel_size=3, padding=1)
        # Maps r channels -> out_channels*4 for the second PixelShuffle(2)
        self.proj_out = nn.Conv2d(r, out_channels * 4, kernel_size=1)
        self.up2 = nn.PixelShuffle(2)                 # -> (out_channels, 2H, 2W)
        nn.init.normal_(self.proj_out.weight, std=1e-3)
        nn.init.constant_(self.proj_out.bias, _INIT_OUT_LEVEL)

    def forward(self, feat):
        z = self.blocks(feat)
        z = self.up(z)                                # (B, r, H, W)  — packed Bayer res
        z = self.act(self.refine1(z))
        z = self.refine2(z)
        return self.up2(self.proj_out(z))             # (B, out_channels, 2H, 2W)  — sensor res


class MoEDenoiser(nn.Module):
    """
    Semi-lightweight Mixture-of-Experts HDR joint denoising + demosaicing model.

    One shared trunk (encoder -> Restormer latent -> decoder) feeds K lightweight
    expert heads. Each head outputs a full-sensor-resolution RGB prediction.
    A per-pixel gate conditioned on the noisy BGGR input + SNR map (both at
    packed Bayer resolution) produces softmax weights, which are bilinearly
    upsampled 2× to sensor resolution before blending the expert RGB outputs.

    At the default size (dim=32, num_blocks=[4,4,4,4], K=3) this is ~21M
    parameters and ~1.05x the FLOPs of a single teacher.

    forward(x, snr_map) -> (blended, expert_outs, gates)
        x:        [B, 4, H, W]  packed BGGR in [0, 1], H, W % 8 == 0
        snr_map:  [B, 1, H, W]  normalised local SNR in [0, 1]
    Returns:
        blended:     [B, 3, 2H, 2W]    sensor-resolution RGB
        expert_outs: [B, K, 3, 2H, 2W] per-expert RGB
        gates:       [B, K, 2H, 2W]    routing weights at sensor resolution
    """
    def __init__(self, out_channels=3, dim=32, num_blocks=(4, 4, 4, 4),
                 num_refinement_blocks=4, heads=(1, 2, 4, 8), se_reduction=8,
                 num_experts=3, expert_blocks=2, gate_hidden=16):
        super().__init__()
        del num_refinement_blocks   # lives inside the expert heads
        self.dim = dim
        self.num_experts = num_experts

        # ======== Shared trunk (same topology as the teacher through decoder_level_1) ========
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
        # Trunk output: [B, dim*2, H/2, W/2]

        # ======== Experts + gate ========
        self.experts = nn.ModuleList([
            ExpertHead(dim * 2, out_channels, expert_blocks, se_reduction)
            for _ in range(num_experts)
        ])
        # Break the symmetry between otherwise-identical expert heads (see
        # _EXPERT_INIT_SPREAD). No-op for a single expert.
        if num_experts > 1:
            for k, head in enumerate(self.experts):
                scale = 1.0 + (k - (num_experts - 1) / 2.0) * (
                    2.0 * _EXPERT_INIT_SPREAD / max(num_experts - 1, 1))
                with torch.no_grad():
                    head.proj_out.bias.mul_(scale)
                nn.init.normal_(head.proj_out.weight, std=1e-3 * (1 + k))
        # Gate always sees 4 BGGR channels + 1 SNR channel = 5 inputs,
        # regardless of the RGB output channel count.
        self.gate = NoiseGate(num_experts, in_channels=5, hidden=gate_hidden)

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
        return self.decoder_level_1(w_level1)          # [B, dim*2, H/2, W/2]

    def forward(self, x, snr_map):
        feat = self._trunk(x)                          # [B, dim*2, H/2, W/2]

        # Each expert head outputs sensor-resolution RGB: [B, 3, 2H, 2W]
        expert_outs = torch.stack(
            [head(feat).clamp(min=_CLAMP_EPS) for head in self.experts],
            dim=1)                                     # [B, K, 3, 2H, 2W]

        # Gate at packed Bayer resolution, then bilinearly upsample to sensor res
        gates_lr = self.gate(x, snr_map)              # [B, K, H, W]
        gates = F.interpolate(gates_lr, scale_factor=2.0,
                              mode='bilinear', align_corners=False)  # [B, K, 2H, 2W]

        blended = (gates.unsqueeze(2) * expert_outs).sum(dim=1)    # [B, 3, 2H, 2W]

        if not self.training:
            blended     = blended.clamp(max=1.0)
            expert_outs = expert_outs.clamp(max=1.0)

        return blended, expert_outs, gates


# ---------------------------------------------------------
# 4. Legacy wrappers (unified return signature)
# ---------------------------------------------------------

class DualSNRDenoiser(nn.Module):
    """
    Two independent full teachers blended by the pixel-wise SNR map.
    The SNR map is at packed Bayer resolution; it is bilinearly upsampled to
    sensor resolution before blending the RGB teacher outputs:
        out = (1 - snr_up) * out_low + snr_up * out_high
    Heavy (2x teacher params and FLOPs) — kept for existing checkpoints.
    Returns the unified (blended, expert_outs, gates) tuple where gates are at
    sensor resolution to match expert_outs.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.num_experts = 2
        self.denoiser_low_snr  = TransUNet_Teacher_HDR(**kwargs)
        self.denoiser_high_snr = TransUNet_Teacher_HDR(**kwargs)

    def forward(self, x, snr_map):
        out_low  = self.denoiser_low_snr(x)            # [B, 3, 2H, 2W]
        out_high = self.denoiser_high_snr(x)           # [B, 3, 2H, 2W]
        # Upsample SNR map from Bayer res to sensor res for blending
        snr_up = F.interpolate(snr_map, size=out_low.shape[-2:],
                               mode='bilinear', align_corners=False)  # [B, 1, 2H, 2W]
        blended = (1.0 - snr_up) * out_low + snr_up * out_high
        expert_outs = torch.stack([out_low, out_high], dim=1)         # [B, 2, 3, 2H, 2W]
        gates = torch.cat([1.0 - snr_up, snr_up], dim=1)             # [B, 2, 2H, 2W]
        return blended, expert_outs, gates


DualSNRTeacher = DualSNRDenoiser


class SingleDenoiser(nn.Module):
    """Baseline: one denoiser, no routing. Same return signature."""
    def __init__(self, **kwargs):
        super().__init__()
        self.num_experts = 1
        self.denoiser = TransUNet_Teacher_HDR(**kwargs)

    def forward(self, x, snr_map):
        out = self.denoiser(x)                         # [B, 3, 2H, 2W]
        # Gate is uniform 1.0 at sensor resolution
        gates = F.interpolate(torch.ones_like(snr_map), size=out.shape[-2:],
                              mode='nearest')          # [B, 1, 2H, 2W]
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
