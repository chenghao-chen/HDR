"""
End-to-end integration tests: the real modules wired together the way the
training and evaluation scripts wire them.

Everything here is CPU-only and deliberately tiny (dim=8, packed 16-32 px,
batch 2) so the whole file stays well under a minute. What these tests pin
down is the *seams* between modules — the places unit tests never look:

  dataset -> collate -> estimate_local_snr_map -> GBTF ground truth ->
  model -> composite loss -> backward -> clip -> optimizer.step()

and the checkpoint format that has to survive the trip from
train_A100_MoE_two_phase.py to test_dual_MoE_two_phase.py.
"""

import math
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import test_dual_MoE_two_phase as evalscript
import train_A100_MoE_two_phase as trainscript
from DifferentiableGBTF_BGGR import DifferentiableGBTF_BGGR
from HDR_Mobile_dataset import MobileHDRDataset
from HDR_model_hybrid_Teacher import (_CLAMP_EPS, build_denoiser,
                                      estimate_local_snr_map)
from helpers import (assert_finite, assert_shape, make_dataset,
                     packed_bayer)

hdr_tonemap = trainscript.hdr_tonemap
collate_xy = trainscript.collate_xy
collate_pad_to_max = trainscript.collate_pad_to_max

# Loss weights: exactly the Phase-1 values from train_A100_MoE_two_phase.py.
MU = 5000
GAMMA = 0.1
AUX_WEIGHT = 0.5
BALANCE_WEIGHT = 0.01
GRAD_CLIP = 1.0
LPIPS_CROP = 256


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file — conftest/helpers are shared and read-only)
# ─────────────────────────────────────────────────────────────────────────────

def _vgg16_weights_cached():
    """True only if torchvision's VGG16 weights are already on disk.

    lpips.LPIPS(net='vgg') builds a torchvision VGG16 with pretrained=True,
    which downloads from download.pytorch.org on a cache miss. The test suite
    must never touch the network, so we look before we leap.
    """
    hub = torch.hub.get_dir()
    return os.path.isfile(os.path.join(hub, "checkpoints", "vgg16-397923af.pth"))


class _StandInPerceptual(nn.Module):
    """
    Offline stand-in for lpips.LPIPS(net='vgg').

    The real LPIPS needs pretrained VGG16 weights that are not in this
    machine's torch hub cache, and downloading them would hit the network
    (forbidden). This module has the same call contract — takes two
    [B, 3, h, w] tensors in [-1, 1], returns a per-sample tensor whose
    .mean() is the scalar perceptual loss — and the same structural role:
    a frozen, non-trainable feature extractor whose L1 feature distance is
    differentiable w.r.t. the prediction. That is everything the training
    step needs from it; it is NOT a perceptual-quality claim.
    """

    def __init__(self, seed=7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)
        with torch.no_grad():
            for c in (self.conv1, self.conv2):
                c.weight.copy_(torch.randn(c.weight.shape, generator=g) * 0.2)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def _feats(self, t):
        f1 = F.gelu(self.conv1(t))
        f2 = self.conv2(f1)
        return f1, f2

    def forward(self, a, b):
        fa, fb = self._feats(a), self._feats(b)
        per_sample = sum((x - y).abs().mean(dim=(1, 2, 3))
                         for x, y in zip(fa, fb))
        return per_sample.view(-1, 1, 1, 1)


def _perceptual_loss():
    """Real LPIPS when its weights are cached, else the offline stand-in."""
    if _vgg16_weights_cached():
        import lpips
        net = lpips.LPIPS(net="vgg")
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net, "lpips-vgg"
    return _StandInPerceptual(), "stand-in"


def _gbtf():
    g = DifferentiableGBTF_BGGR()
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g


def _valid_mask_like_training_loop(sample, x):
    """
    Byte-for-byte reproduction of the valid_mask construction in
    train_A100_MoE_two_phase.py's inner loop.

    orig_h / orig_w are in PACKED Bayer space; the mask lives at sensor
    resolution, hence the factor 2.
    """
    B, _, H, W = x.shape
    Hs, Ws = H * 2, W * 2
    if "orig_h" in sample:
        orig_h, orig_w = sample["orig_h"], sample["orig_w"]
        valid_mask = torch.zeros(B, 1, Hs, Ws)
        for b in range(B):
            valid_mask[b, 0, :orig_h[b] * 2, :orig_w[b] * 2] = 1.0
    else:
        orig_h = torch.full((B,), H)
        orig_w = torch.full((B,), W)
        valid_mask = torch.ones(B, 1, Hs, Ws)
    return valid_mask, orig_h, orig_w


def _composite_loss(model, x, snr_map, y_rgb, valid_mask, orig_h, orig_w,
                    percep, K, crop_origin=(0, 0),
                    gamma=GAMMA, aux_weight=AUX_WEIGHT,
                    balance_weight=BALANCE_WEIGHT):
    """
    The exact Phase-1 composite loss of train_A100_MoE_two_phase.py:

        L = L1-mu(valid pixels)
            + gamma * perceptual(256-crop)
            + aux_weight * sum_k  gate_k-weighted L1-mu(expert_k)
            + balance_weight * (K * sum_k mean_gate_k^2 - 1)

    Returns (total, components dict, forward outputs). The only deviation
    from the script is that the LPIPS crop origin is passed in instead of
    drawn from random.randint, so tests can be exactly reproducible.
    """
    y_pred, expert_outs, gates = model(x, snr_map)
    C = y_pred.shape[1]

    tm_pred = hdr_tonemap(y_pred, mu=MU)
    tm_gt = hdr_tonemap(y_rgb, mu=MU)
    valid_px = valid_mask.sum()
    loss_l1_mu = (((tm_pred - tm_gt).abs() * valid_mask).sum()
                  / (valid_px * C + 1e-6))

    if aux_weight > 0:
        tm_experts = hdr_tonemap(expert_outs, mu=MU)
        err = (tm_experts - tm_gt.unsqueeze(1)).abs()
        w = gates.detach().unsqueeze(2) * valid_mask.unsqueeze(1)
        loss_aux = ((w * err).sum(dim=(0, 2, 3, 4))
                    / (w.sum(dim=(0, 2, 3, 4)) * C + 1e-6)).sum()
    else:
        loss_aux = x.new_zeros(())

    gate_usage = ((gates * valid_mask).sum(dim=(0, 2, 3))
                  / valid_px.clamp(min=1.0))
    if balance_weight > 0:
        loss_balance = K * (gate_usage ** 2).sum() - 1.0
    else:
        loss_balance = x.new_zeros(())

    min_h = int(orig_h.min().item()) * 2
    min_w = int(orig_w.min().item()) * 2
    crop_h = min(LPIPS_CROP, min_h)
    crop_w = min(LPIPS_CROP, min_w)
    top, left = crop_origin

    def _crop(t):
        c = t[:, :, top:top + crop_h, left:left + crop_w].float()
        return hdr_tonemap(c.clamp(0, 1), mu=MU) * 2.0 - 1.0

    loss_perceptual = percep(_crop(y_pred), _crop(y_rgb)).mean()

    total = (loss_l1_mu
             + gamma * loss_perceptual
             + aux_weight * loss_aux
             + balance_weight * loss_balance)
    comps = {
        "l1_mu": loss_l1_mu,
        "percep": loss_perceptual,
        "aux": loss_aux,
        "balance": loss_balance,
        "gate_usage": gate_usage,
    }
    return total, comps, (y_pred, expert_outs, gates)


def _break_the_zero_init(model, seed=3):
    """
    Give every zero-initialised output layer a small non-zero value.

    ExpertHead.proj_out and NoiseGate.net[-1] are zero-initialised on
    purpose, which makes every expert output exactly 0 at step 0. Several
    tests need a model whose predictions are non-degenerate (a checkpoint
    round trip against an all-eps output would be vacuous, and a gradient
    can only be observed where the graph is alive). This does not touch any
    source module — it perturbs an instantiated model in place.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("proj_out.weight") or name.endswith("net.4.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            elif name.endswith("proj_out.bias"):
                p.fill_(0.4)
            elif name.endswith("net.4.bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return model


def _tiny_batch(tiny_kwargs, batch=2, h=16, w=16, num_experts=2, seed=0):
    """A ready-to-train (model, x, snr, y_rgb, valid_mask, orig_h, orig_w, K)."""
    model = build_denoiser("moe", num_experts=num_experts, **tiny_kwargs)
    model.train()
    x = packed_bayer(batch, h, w, seed=seed)
    y = packed_bayer(batch, h, w, seed=seed + 100) * 0.8 + 0.1
    sample = {"x": x, "y": y}
    valid_mask, orig_h, orig_w = _valid_mask_like_training_loop(sample, x)
    with torch.no_grad():
        snr_map = estimate_local_snr_map(x, window_size=5)
        y_rgb = _gbtf()(F.pixel_shuffle(y.float(), 2))
    return model, x, snr_map, y_rgb, valid_mask, orig_h, orig_w, model.num_experts


# ═════════════════════════════════════════════════════════════════════════════
# 1. Full Phase-1 training step
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_phase1_training_step_end_to_end(tmp_path, tiny_kwargs):
    """
    The real Phase-1 inner loop, from on-disk dataset to optimizer.step().

    dataset -> DataLoader(collate_fn=collate_xy) -> estimate_local_snr_map ->
    gbtf(pixel_shuffle(y, 2)) -> model(x, snr) -> composite loss -> backward
    -> clip_grad_norm_ -> step.

    Pins: the loss is finite and positive, every requires_grad parameter
    receives a non-None finite gradient (a None gradient means a parameter is
    disconnected from the graph — a real architectural defect), and no
    parameter turns into NaN after the step. Perceptual term uses the offline
    stand-in (see _StandInPerceptual) because LPIPS' VGG16 weights are not
    cached and downloading them would need the network.
    """
    root = make_dataset(tmp_path / "ds", n_train=2, n_test=1, h=48, w=48)
    ds = MobileHDRDataset(base_dir=root, split="train", transform=None,
                          num_patch=1, crop_size=16)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                         num_workers=0, collate_fn=collate_xy)
    sample = next(iter(loader))
    assert set(sample) == {"x", "y"}, "collate_xy must drop the duplicate 'xm'"

    x, y = sample["x"], sample["y"]
    assert_shape(x, (2, 4, 16, 16), "collated x")

    model = build_denoiser("moe", num_experts=2, **tiny_kwargs)
    model.train()
    K = model.num_experts
    percep, percep_name = _perceptual_loss()
    gbtf = _gbtf()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    valid_mask, orig_h, orig_w = _valid_mask_like_training_loop(sample, x)
    assert float(valid_mask.sum()) == 2 * 32 * 32, \
        "Phase-1 patches are uniform, so the mask must be all ones at sensor res"

    with torch.no_grad():
        snr_map = estimate_local_snr_map(x, window_size=5)
        y_rgb = gbtf(F.pixel_shuffle(y.float(), 2))
    assert_shape(snr_map, (2, 1, 16, 16), "snr_map")
    assert_shape(y_rgb, (2, 3, 32, 32), "gbtf ground truth")
    assert_finite(y_rgb, "gbtf ground truth")

    optimizer.zero_grad(set_to_none=True)
    total, comps, (y_pred, expert_outs, gates) = _composite_loss(
        model, x, snr_map, y_rgb, valid_mask, orig_h, orig_w, percep, K)

    assert_shape(y_pred, (2, 3, 32, 32), "y_pred")
    assert_shape(expert_outs, (2, K, 3, 32, 32), "expert_outs")
    assert_shape(gates, (2, K, 32, 32), "gates")

    assert torch.isfinite(total), f"total loss not finite ({percep_name})"
    assert float(total) > 0.0, f"total loss must be positive, got {float(total)}"
    for name, v in comps.items():
        if name == "gate_usage":
            continue
        assert torch.isfinite(v).all(), f"loss component {name!r} not finite"
    assert float(comps["l1_mu"]) > 0.0
    assert float(comps["aux"]) > 0.0
    assert float(comps["balance"]) >= -1e-5, "load-balance term must be >= 0"

    total.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"parameters disconnected from the loss graph: {missing}"
    nonfinite = [n for n, p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not nonfinite, f"non-finite gradients on: {nonfinite}"

    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    assert torch.isfinite(gnorm)
    optimizer.step()

    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not bad, f"parameters became NaN/Inf after optimizer.step(): {bad}"


@pytest.mark.xfail(reason="BUG: zero-init proj_out + clamp(min=_CLAMP_EPS) "
                          "gives every parameter exactly zero gradient",
                   strict=False)
def test_phase1_training_step_delivers_nonzero_gradient(tiny_kwargs):
    """
    A freshly built MoE model must be trainable: at least the expert heads
    and the shared trunk have to receive a NON-ZERO gradient from the
    composite loss on step 0.

    Why it matters: ExpertHead.proj_out is zero-initialised, so every expert
    output is exactly 0.0 before clamping. MoEDenoiser.forward then applies
    .clamp(min=_CLAMP_EPS); clamp's derivative is 0 below the bound, so the
    whole trunk+expert subgraph is cut off. With zero gradients Adam takes a
    zero-sized step, proj_out stays zero, and the model never leaves its
    initialisation. Non-None-but-zero gradients look healthy to every
    smoke test, which is exactly why this needs pinning.
    """
    model, x, snr, y_rgb, mask, oh, ow, K = _tiny_batch(tiny_kwargs)
    percep, _ = _perceptual_loss()
    total, _, _ = _composite_loss(model, x, snr, y_rgb, mask, oh, ow, percep, K)
    total.backward()

    grad_sums = {n: float(p.grad.abs().sum()) for n, p in model.named_parameters()
                 if p.grad is not None}
    live = {n: v for n, v in grad_sums.items() if v > 0.0}
    assert live, ("every one of %d parameters received an exactly-zero "
                  "gradient — the model cannot train" % len(grad_sums))
    trunk_live = [n for n in live if n.startswith(("patch_embed", "encoder",
                                                  "latent", "decoder"))]
    assert trunk_live, "shared trunk received no gradient"


@pytest.mark.slow
@pytest.mark.xfail(reason="BUG: dead gradient at init (see "
                          "test_phase1_training_step_delivers_nonzero_gradient) "
                          "freezes the loss for every step",
                   strict=False)
def test_phase1_overfits_one_batch_from_default_init(tiny_kwargs):
    """
    Five Phase-1 steps on a single fixed batch must reduce the loss: a model
    with ~100K parameters can trivially overfit two 16x16 patches, so a flat
    or rising curve means the optimisation is not connected to the objective.

    This is the end-to-end consequence of the dead-gradient defect: from the
    shipped initialisation the loss does not move at all.
    """
    model, x, snr, y_rgb, mask, oh, ow, K = _tiny_batch(tiny_kwargs)
    percep, _ = _perceptual_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = _composite_loss(model, x, snr, y_rgb, mask, oh, ow,
                                      percep, K)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(float(total))

    assert all(math.isfinite(v) for v in losses), losses
    assert losses[-1] < losses[0], \
        f"loss did not decrease over 5 steps on one batch: {losses}"


@pytest.mark.slow
def test_phase1_overfits_one_batch_with_live_gradients(tiny_kwargs):
    """
    Control for the two xfails above: the loss, optimizer and data plumbing
    reproduced here are correct — only the zero initialisation is fatal.

    With ExpertHead.proj_out nudged off exactly-zero (so clamp(min=eps) is no
    longer clipping), the very same five-step loop drives the loss down and
    delivers non-zero gradients to the shared trunk. That localises the bug
    to the init/clamp interaction rather than to this test's loss.
    """
    model, x, snr, y_rgb, mask, oh, ow, K = _tiny_batch(tiny_kwargs)
    _break_the_zero_init(model)
    percep, _ = _perceptual_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses, trunk_grads = [], []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = _composite_loss(model, x, snr, y_rgb, mask, oh, ow,
                                      percep, K)
        total.backward()
        trunk_grads.append(float(model.patch_embed.weight.grad.abs().sum()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        losses.append(float(total))

    assert all(math.isfinite(v) for v in losses), losses
    assert trunk_grads[0] > 0.0, "shared trunk still starved of gradient"
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    assert not bad, f"parameters became NaN/Inf: {bad}"


# ═════════════════════════════════════════════════════════════════════════════
# 2. Load-balance loss algebra
# ═════════════════════════════════════════════════════════════════════════════

def _balance(gates, valid_mask):
    """The training loop's load-balance term, verbatim."""
    K = gates.shape[1]
    valid_px = valid_mask.sum()
    usage = ((gates * valid_mask).sum(dim=(0, 2, 3)) / valid_px.clamp(min=1.0))
    return float(K * (usage ** 2).sum() - 1.0), usage


@pytest.mark.parametrize("K", [2, 3, 5])
def test_load_balance_is_zero_for_uniform_and_K_minus_1_on_collapse(K):
    """
    K * sum_k(mean gate_k)^2 - 1 must be EXACTLY 0 for perfectly uniform
    routing and exactly K-1 when all traffic collapses onto one expert.

    This is the anti-collapse term: multiplied by balance_weight it is the
    only pressure keeping experts in use. A sign error or a misplaced -1
    would silently invert the incentive and reward collapse, which no shape
    test would ever catch, so the two analytic endpoints get pinned here.
    """
    B, H, W = 2, 8, 8
    mask = torch.ones(B, 1, H, W)

    uniform = torch.full((B, K, H, W), 1.0 / K)
    val, usage = _balance(uniform, mask)
    assert usage.sum() == pytest.approx(1.0, abs=1e-6)
    assert val == pytest.approx(0.0, abs=1e-6), \
        f"uniform routing must cost 0, got {val}"

    collapsed = torch.zeros(B, K, H, W)
    collapsed[:, 0] = 1.0
    val_c, usage_c = _balance(collapsed, mask)
    assert usage_c.sum() == pytest.approx(1.0, abs=1e-6)
    assert val_c == pytest.approx(K - 1.0, abs=1e-5), \
        f"total collapse must cost K-1={K - 1}, got {val_c}"


def test_load_balance_grows_monotonically_toward_collapse():
    """
    Interpolating routing from uniform to collapsed must make the penalty
    increase monotonically and never go negative.

    The two endpoints alone do not fix the direction of the incentive in
    between; this walks the whole path so a term that dipped (rewarding
    partial collapse) would fail.
    """
    K, B, H, W = 3, 1, 8, 8
    mask = torch.ones(B, 1, H, W)
    uniform = torch.full((B, K, H, W), 1.0 / K)
    collapsed = torch.zeros(B, K, H, W)
    collapsed[:, 0] = 1.0

    vals = []
    for t in torch.linspace(0.0, 1.0, 9):
        gates = (1 - t) * uniform + t * collapsed
        assert gates.sum(dim=1).allclose(torch.ones(B, H, W)), "gates must sum to 1"
        v, _ = _balance(gates, mask)
        assert v >= -1e-6, f"penalty went negative ({v}) at t={float(t)}"
        vals.append(v)

    assert vals[0] == pytest.approx(0.0, abs=1e-6)
    assert vals[-1] == pytest.approx(K - 1.0, abs=1e-5)
    for a, b in zip(vals, vals[1:]):
        assert b > a - 1e-9, f"penalty not monotonic: {vals}"


def test_load_balance_is_nonnegative_for_random_softmax_routing():
    """
    For ANY per-pixel softmax routing the penalty is >= 0, with 0 only at
    uniform usage. That lower bound is what makes it safe to add to the
    total loss; a term that could go negative would let the optimiser buy
    unbounded loss reduction by manipulating the gate alone.
    """
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        K = 2 + seed % 3
        logits = torch.randn((3, K, 8, 8), generator=g) * 3.0
        gates = torch.softmax(logits, dim=1)
        v, usage = _balance(gates, torch.ones(3, 1, 8, 8))
        assert usage.sum() == pytest.approx(1.0, abs=1e-5)
        assert v >= -1e-6, f"seed {seed}: penalty {v} < 0"
        assert v <= K - 1.0 + 1e-5, f"seed {seed}: penalty {v} above K-1"


# ═════════════════════════════════════════════════════════════════════════════
# 3. Auxiliary per-expert loss algebra
# ═════════════════════════════════════════════════════════════════════════════

def _aux(expert_outs, gt, gates, valid_mask):
    """The training loop's auxiliary term, verbatim (tonemapping already applied)."""
    C = gt.shape[1]
    err = (expert_outs - gt.unsqueeze(1)).abs()
    w = gates.detach().unsqueeze(2) * valid_mask.unsqueeze(1)
    return ((w * err).sum(dim=(0, 2, 3, 4))
            / (w.sum(dim=(0, 2, 3, 4)) * C + 1e-6)).sum()


def test_aux_loss_equals_that_experts_own_l1_when_one_expert_takes_everything():
    """
    With hand-built one-hot gates, the gate-weighted per-expert term must
    collapse to exactly the winning expert's plain L1 — and the starved
    experts must contribute exactly 0, not a division blow-up.

    This is the normalisation contract: the denominator is the gate mass
    times the channel count, so each expert's term is a true mean over the
    pixels routed to it. If the denominator were wrong the aux term would
    scale with the gate mass and dominate (or vanish from) the total loss.
    """
    B, K, C, H, W = 2, 3, 3, 8, 8
    g = torch.Generator().manual_seed(11)
    gt = torch.rand((B, C, H, W), generator=g)
    experts = torch.rand((B, K, C, H, W), generator=g)
    mask = torch.ones(B, 1, H, W)

    for winner in range(K):
        gates = torch.zeros(B, K, H, W)
        gates[:, winner] = 1.0
        got = float(_aux(experts, gt, gates, mask))
        expected = float((experts[:, winner] - gt).abs().mean())
        assert got == pytest.approx(expected, rel=1e-5), \
            f"winner={winner}: aux {got} != its own L1 {expected}"

        # the starved experts contributed nothing at all
        other = float((experts[:, (winner + 1) % K] - gt).abs().mean())
        assert abs(got - (expected + other)) > 1e-6, \
            "starved experts must not contribute to the aux term"


def test_aux_loss_with_uniform_gates_is_the_sum_of_per_expert_l1s():
    """
    Uniform routing must give sum_k L1(expert_k) — the gate mass cancels
    out of numerator and denominator, so each expert contributes its full
    mean error regardless of how many experts share the pixel.

    That is what makes aux_weight's effective magnitude scale with K, which
    is exactly the behaviour the docstring advertises ("aux . sum_k gate_k
    weighted L1"). A per-expert *average* instead of a sum would silently
    shrink the aux pressure as experts are added.
    """
    B, K, C, H, W = 2, 3, 3, 8, 8
    g = torch.Generator().manual_seed(12)
    gt = torch.rand((B, C, H, W), generator=g)
    experts = torch.rand((B, K, C, H, W), generator=g)
    gates = torch.full((B, K, H, W), 1.0 / K)
    mask = torch.ones(B, 1, H, W)

    got = float(_aux(experts, gt, gates, mask))
    expected = float(sum((experts[:, k] - gt).abs().mean() for k in range(K)))
    assert got == pytest.approx(expected, rel=1e-5)


def test_aux_loss_gates_are_detached_so_routing_gets_no_gradient_from_it():
    """
    The aux term must not push gradient into the gate. It is detached on
    purpose: otherwise the cheapest way to shrink aux is to route every
    pixel to whichever expert is momentarily best, which is precisely the
    collapse the balance term exists to prevent.
    """
    B, K, C, H, W = 1, 2, 3, 8, 8
    g = torch.Generator().manual_seed(13)
    gt = torch.rand((B, C, H, W), generator=g)
    experts = torch.rand((B, K, C, H, W), generator=g, requires_grad=True)
    logits = torch.randn((B, K, H, W), generator=g, requires_grad=True)
    gates = torch.softmax(logits, dim=1)
    mask = torch.ones(B, 1, H, W)

    _aux(experts, gt, gates, mask).backward()
    assert logits.grad is None, \
        "gate logits received gradient from the aux term (missing .detach())"
    assert experts.grad is not None and float(experts.grad.abs().sum()) > 0, \
        "aux term must still train the expert outputs"


# ═════════════════════════════════════════════════════════════════════════════
# 4. Phase-2 padding path
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_phase2_valid_mask_makes_the_loss_ignore_padded_pixels(tiny_kwargs):
    """
    Phase 2 stacks differently sized full frames via collate_pad_to_max and
    then excludes the padding with a valid_mask built from orig_h/orig_w.
    The whole point of that machinery is this property: the loss must be
    completely insensitive to whatever sits in the padded region.

    Also pins the packed-vs-sensor factor 2 (orig dims are packed-Bayer, the
    mask is at sensor resolution) and that gate usage still sums to 1 over
    the valid region — if the mask were sized or scaled wrong, that sum
    would drift and the balance term would stop meaning anything.
    """
    g = torch.Generator().manual_seed(21)
    sizes = [(20, 28), (24, 17)]          # packed dims, deliberately unequal
    batch = []
    for i, (h, w) in enumerate(sizes):
        batch.append({
            "x": torch.rand((4, h, w), generator=g),
            "y": torch.rand((4, h, w), generator=g) * 0.8 + 0.1,
        })
    sample = collate_pad_to_max(batch)

    assert_shape(sample["x"], (2, 4, 24, 32), "padded x")
    assert sample["x"].shape[2] % 8 == 0 and sample["x"].shape[3] % 8 == 0, \
        "padded packed dims must stay divisible by 8 for the three unshuffles"
    assert sample["orig_h"].tolist() == [20, 24]
    assert sample["orig_w"].tolist() == [28, 17]

    x, y = sample["x"], sample["y"]
    valid_mask, orig_h, orig_w = _valid_mask_like_training_loop(sample, x)
    expected_valid = sum(4 * h * w for h, w in sizes)   # (2h)*(2w) per frame
    assert float(valid_mask.sum()) == float(expected_valid), \
        "valid_mask must cover exactly 2*orig_h x 2*orig_w sensor pixels"

    model = build_denoiser("moe", num_experts=2, **tiny_kwargs)
    _break_the_zero_init(model)
    model.train()
    K = model.num_experts
    percep, _ = _perceptual_loss()
    with torch.no_grad():
        snr_map = estimate_local_snr_map(x, window_size=5)
        y_rgb = _gbtf()(F.pixel_shuffle(y.float(), 2))
    assert_shape(y_rgb, (2, 3, 48, 64), "gbtf ground truth at sensor res")

    with torch.no_grad():
        base, comps, _ = _composite_loss(model, x, snr_map, y_rgb, valid_mask,
                                         orig_h, orig_w, percep, K)
        assert comps["gate_usage"].sum() == pytest.approx(1.0, abs=1e-4), \
            "gate usage over the valid region must still sum to 1"

        # Replace the padded region of the ground truth with garbage.
        junk = torch.full_like(y_rgb, 7.5)
        y_rgb_junk = y_rgb * valid_mask + junk * (1.0 - valid_mask)
        assert not torch.allclose(y_rgb_junk, y_rgb), \
            "fixture bug: there is no padded region to corrupt"
        polluted, _, _ = _composite_loss(model, x, snr_map, y_rgb_junk,
                                         valid_mask, orig_h, orig_w, percep, K)
        assert float(polluted) == pytest.approx(float(base), rel=1e-6), \
            "padded pixels leaked into the masked loss"

        # Sanity: the loss is not simply constant — a valid pixel matters.
        y_rgb_valid_edit = y_rgb.clone()
        y_rgb_valid_edit[0, :, 0, 0] += 0.5
        moved, _, _ = _composite_loss(model, x, snr_map, y_rgb_valid_edit,
                                      valid_mask, orig_h, orig_w, percep, K)
        assert abs(float(moved) - float(base)) > 1e-9, \
            "changing a VALID pixel did not change the loss (mask is vacuous)"


def test_phase2_lpips_crop_window_stays_inside_every_frames_valid_region():
    """
    The perceptual crop is taken at the same (top, left) for the whole batch,
    sized from min(orig_h)*2 x min(orig_w)*2. That is the only reason the
    unmasked LPIPS term is safe on a padded batch, so it needs pinning: for
    every frame in the batch the crop must fit inside that frame's real
    pixels.

    If the crop were sized from the *padded* dims instead, LPIPS would be
    computed partly on reflection padding and would quietly bias Phase 2.
    """
    sizes = [(20, 28), (24, 17), (23, 31)]
    orig_h = torch.tensor([h for h, _ in sizes])
    orig_w = torch.tensor([w for _, w in sizes])
    min_h, min_w = int(orig_h.min()) * 2, int(orig_w.min()) * 2
    crop_h, crop_w = min(LPIPS_CROP, min_h), min(LPIPS_CROP, min_w)

    assert crop_h > 0 and crop_w > 0
    # randint(0, min_h - crop_h) is degenerate here, but the invariant must
    # hold for every legal origin.
    for top in range(0, min_h - crop_h + 1):
        for left in range(0, min_w - crop_w + 1):
            for h, w in sizes:
                assert top + crop_h <= 2 * h, \
                    f"crop rows {top}:{top + crop_h} exceed frame height {2 * h}"
                assert left + crop_w <= 2 * w, \
                    f"crop cols {left}:{left + crop_w} exceed frame width {2 * w}"


# ═════════════════════════════════════════════════════════════════════════════
# 5. Checkpoint round trip across the two scripts
# ═════════════════════════════════════════════════════════════════════════════

def _train_style_checkpoint(path, model, mode, num_experts, model_kwargs,
                            prefix=""):
    """Exactly the save_dict train_A100_MoE_two_phase.py writes."""
    state = {prefix + k: v for k, v in model.state_dict().items()}
    torch.save({
        "epoch": 3,
        "phase": 1,
        "mode": mode,
        "num_experts": num_experts,
        "model_kwargs": model_kwargs,
        "model_state_dict": state,
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "loss": 0.42,
        "best_psnr_mu": 27.5,
    }, str(path))
    return str(path)


def _eval_inputs(batch=1, h=16, w=16, seed=5):
    x = packed_bayer(batch, h, w, seed=seed)
    return x, estimate_local_snr_map(x, window_size=5)


def _assert_bit_identical(model_a, model_b, x, snr):
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        a = model_a(x, snr)
        b = model_b(x, snr)
    for i, name in enumerate(("blended", "expert_outs", "gates")):
        assert torch.equal(a[i], b[i]), (
            f"{name} differs after checkpoint round trip (max abs diff "
            f"{float((a[i] - b[i]).abs().max())})")


def test_checkpoint_round_trip_two_experts(tmp_path, tiny_kwargs, device):
    """
    A checkpoint written in the training script's format must reload through
    test_dual_MoE_two_phase.load_model_from_checkpoint and reproduce the
    original model's output bit for bit.

    This is the seam between the two scripts: mode / model_kwargs /
    num_experts are stored precisely so the benchmark can rebuild the
    architecture without manual sync. The weights are perturbed off the
    zero init first, otherwise every expert emits the same constant and the
    comparison would pass for any architecture at all.
    """
    model = build_denoiser("moe", num_experts=2, **tiny_kwargs)
    _break_the_zero_init(model, seed=41)
    ckpt = _train_style_checkpoint(tmp_path / "p1.pth", model, "moe", 2,
                                   dict(tiny_kwargs))

    loaded, mode = evalscript.load_model_from_checkpoint(
        ckpt, fallback_kwargs={"dim": 999}, device=device)
    assert mode == "moe", "mode must come from the checkpoint, not the fallback"
    assert loaded.num_experts == 2
    assert loaded.dim == tiny_kwargs["dim"], \
        "model_kwargs in the checkpoint must win over fallback_kwargs"
    assert not loaded.training, "benchmark loader must return an eval-mode model"

    x, snr = _eval_inputs()
    _assert_bit_identical(model, loaded, x, snr)

    # And the comparison is not vacuous: a different init disagrees.
    other = build_denoiser("moe", num_experts=2, **tiny_kwargs)
    _break_the_zero_init(other, seed=999)
    other.eval()
    with torch.no_grad():
        assert not torch.equal(other(x, snr)[0], loaded(x, snr)[0]), \
            "two different inits produced identical output — test is vacuous"


@pytest.mark.xfail(reason="BUG: load_model_from_checkpoint hardcodes "
                          "num_experts=2, ignoring the checkpoint's value",
                   strict=False)
def test_checkpoint_round_trip_three_experts(tmp_path, tiny_kwargs, device):
    """
    The same round trip with num_experts=3. The checkpoint records
    num_experts, and load_model_from_checkpoint documents that it rebuilds
    the architecture from that metadata, so a K=3 run must reload and
    reproduce its outputs exactly.

    It does not: line 86 of test_dual_MoE_two_phase.py is
        num_experts  = 2 #ckpt.get("num_experts", fallback_num_experts)
    so every K != 2 checkpoint either dies on load_state_dict(strict=True)
    or (worse, if strict were relaxed) silently drops experts. This breaks
    the train/test contract for the default MoE configuration, whose
    build_denoiser signature defaults to num_experts=3.
    """
    model = build_denoiser("moe", num_experts=3, **tiny_kwargs)
    _break_the_zero_init(model, seed=42)
    ckpt = _train_style_checkpoint(tmp_path / "p1_k3.pth", model, "moe", 3,
                                   dict(tiny_kwargs))

    loaded, mode = evalscript.load_model_from_checkpoint(
        ckpt, fallback_kwargs=dict(tiny_kwargs), device=device,
        fallback_num_experts=3)
    assert loaded.num_experts == 3, \
        f"rebuilt with {loaded.num_experts} experts, checkpoint says 3"

    x, snr = _eval_inputs()
    _assert_bit_identical(model, loaded, x, snr)


def test_checkpoint_loader_strips_the_torch_compile_prefix(tmp_path,
                                                           tiny_kwargs, device):
    """
    A model saved while wrapped by torch.compile has every key prefixed with
    '_orig_mod.'. The loader must strip that and still load with
    strict=True, otherwise every checkpoint from a USE_COMPILE=True run is
    unreadable by the benchmark script.
    """
    model = build_denoiser("moe", num_experts=2, **tiny_kwargs)
    _break_the_zero_init(model, seed=43)
    ckpt = _train_style_checkpoint(tmp_path / "compiled.pth", model, "moe", 2,
                                   dict(tiny_kwargs), prefix="_orig_mod.")

    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    assert all(k.startswith("_orig_mod.") for k in raw["model_state_dict"]), \
        "fixture bug: the saved keys are not prefixed"

    loaded, _ = evalscript.load_model_from_checkpoint(
        ckpt, fallback_kwargs=dict(tiny_kwargs), device=device)
    assert not any(k.startswith("_orig_mod.") for k in loaded.state_dict()), \
        "prefix survived into the live module"

    x, snr = _eval_inputs()
    _assert_bit_identical(model, loaded, x, snr)


@pytest.mark.parametrize("mode,expected_k", [("dual", 2), ("single", 1)])
def test_checkpoint_round_trip_legacy_modes(tmp_path, tiny_kwargs, device,
                                            mode, expected_k):
    """
    The 'dual' and 'single' legacy modes must survive the same round trip.
    Their num_experts is fixed by the class (2 and 1), so the loader's
    hardcoded num_experts=2 happens to be harmless here — which is exactly
    why the K=3 failure above went unnoticed.
    """
    model = build_denoiser(mode, **tiny_kwargs)
    assert model.num_experts == expected_k
    ckpt = _train_style_checkpoint(tmp_path / f"{mode}.pth", model, mode,
                                   expected_k, dict(tiny_kwargs))

    loaded, got_mode = evalscript.load_model_from_checkpoint(
        ckpt, fallback_kwargs=dict(tiny_kwargs), device=device)
    assert got_mode == mode
    assert loaded.num_experts == expected_k

    x, snr = _eval_inputs()
    _assert_bit_identical(model, loaded, x, snr)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Dataset -> model shape compatibility
# ═════════════════════════════════════════════════════════════════════════════

def test_dataset_sample_flows_through_the_model_at_sensor_resolution(tmp_path,
                                                                     tiny_kwargs):
    """
    A real MobileHDRDataset sample, cropped to a multiple of 8, must go
    straight into the model and come back as sensor-resolution RGB: exactly
    2x the packed dims in H and W, 3 channels, gates matching that
    resolution and expert_outs carrying the K axis.

    This is the contract that ties the dataset's [4, h, w] packed BGGR output
    to the model's [B, 3, 2h, 2w] RGB output; a mismatch here breaks every
    run before the first backward pass.
    """
    root = make_dataset(tmp_path / "ds", n_train=2, n_test=1, h=40, w=40)
    ds = MobileHDRDataset(base_dir=root, split="train", transform=None,
                          num_patch=1, crop_size=24)
    s = ds[0]
    assert_shape(s["x"], (4, 24, 24), "dataset x")
    assert_shape(s["y"], (4, 24, 24), "dataset y")
    assert s["x"].shape[1] % 8 == 0 and s["x"].shape[2] % 8 == 0

    x = s["x"].unsqueeze(0)
    snr = estimate_local_snr_map(x, window_size=5)
    assert_shape(snr, (1, 1, 24, 24), "snr_map")

    model = build_denoiser("moe", num_experts=2, **tiny_kwargs)
    model.eval()
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)

    assert_shape(blended, (1, 3, 48, 48), "blended")
    assert_shape(expert_outs, (1, 2, 3, 48, 48), "expert_outs")
    assert_shape(gates, (1, 2, 48, 48), "gates")
    assert_finite(blended, "blended")

    # Eval mode clamps to [_CLAMP_EPS, 1.0]; gates are a softmax over K.
    assert float(blended.min()) >= _CLAMP_EPS - 1e-12
    assert float(blended.max()) <= 1.0 + 1e-6
    assert torch.allclose(gates.sum(dim=1), torch.ones(1, 48, 48), atol=1e-5), \
        "upsampled gates must still sum to 1 per pixel"

    # The GBTF ground truth for the same sample lines up with the prediction.
    y_rgb = _gbtf()(F.pixel_shuffle(s["y"].unsqueeze(0).float(), 2))
    assert y_rgb.shape == blended.shape, \
        f"GT {tuple(y_rgb.shape)} != prediction {tuple(blended.shape)}"


def test_dataset_full_frame_batch_flows_through_padding_collate(tmp_path,
                                                                tiny_kwargs):
    """
    The Phase-2 wiring on real dataset samples: crop_size=None full frames
    -> collate_pad_to_max -> model. Pins that the collate's round-up to a
    multiple of 8 is what makes the three PixelUnshuffle(2) stages legal,
    and that the model output is 2x the PADDED packed dims (not the
    original ones) — the reason the loss needs a mask at all.
    """
    root = make_dataset(tmp_path / "ds", n_train=2, n_test=1, h=20, w=28)
    ds = MobileHDRDataset(base_dir=root, split="train", transform=None,
                          num_patch=1, crop_size=None)
    batch = [ds[0], ds[1]]
    assert batch[0]["x"].shape == (4, 20, 28), "fixture bug: unexpected frame size"

    sample = collate_pad_to_max(batch)
    x = sample["x"]
    assert_shape(x, (2, 4, 24, 32), "padded x")
    assert sample["orig_h"].tolist() == [20, 20]
    assert sample["orig_w"].tolist() == [28, 28]

    model = build_denoiser("moe", num_experts=2, **tiny_kwargs)
    model.eval()
    with torch.no_grad():
        blended, _, gates = model(x, estimate_local_snr_map(x, window_size=5))
    assert_shape(blended, (2, 3, 48, 64), "blended at padded sensor res")
    assert_shape(gates, (2, 2, 48, 64), "gates at padded sensor res")

    valid_mask, _, _ = _valid_mask_like_training_loop(sample, x)
    assert float(valid_mask.sum()) == 2 * 40 * 56
    assert float(valid_mask.numel() - valid_mask.sum()) > 0, \
        "there must be padded pixels for the mask to exclude"
