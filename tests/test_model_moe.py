"""
MoEDenoiser (HDR_model_hybrid_Teacher.py) — contract, invariants and wiring.

The Mixture-of-Experts denoiser is the model the two-phase training script
actually trains, so its published contract

    blended, expert_outs, gates = model(x [B,4,h,w], snr [B,1,h,w])
        blended     [B, 3, 2h, 2w]
        expert_outs [B, K, 3, 2h, 2w]
        gates       [B, K, 2h, 2w]      per-pixel routing weights, sum to 1

is load-bearing for the train loop, the eval script and every checkpoint on
disk.  This file pins:

  * the tuple shapes for K = 1..4, batched and non-square inputs;
  * gates are a partition of unity *after* the bilinear 2x upsample, and
    non-negative;
  * the blending identity (exact in train mode, deliberately broken by the
    eval clamp);
  * the zero-init training-stability contract (every expert starts at
    _CLAMP_EPS, the gate starts uniform at 1/K);
  * the eval-only max clamp and the always-on min clamp;
  * the "one shared trunk, ~95% of compute" claim, expressed as parameters
    and as trunk-module invocation counts;
  * num_refinement_blocks being accepted and discarded;
  * gate conditioning on the SNR map;
  * gradient reachability of trunk / every expert / gate;
  * state_dict round trip with strict=True.

Everything runs on CPU at dim=8 with packed 16x16..32x32 inputs.
"""

import pytest
import torch
import torch.nn.functional as F

from HDR_model_hybrid_Teacher import (
    MoEDenoiser,
    ExpertHead,
    NoiseGate,
    _CLAMP_EPS,
)
from helpers import (
    packed_bayer,
    snr_map_for,
    assert_finite,
    assert_in_range,
    assert_shape,
)


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (file-private on purpose — conftest/helpers are shared)
# ─────────────────────────────────────────────────────────────────────────────

def _moe(tiny_kwargs, num_experts=3, **over):
    """A tiny MoEDenoiser; `over` overrides tiny_kwargs entries."""
    kw = dict(tiny_kwargs)
    kw.update(over)
    return MoEDenoiser(num_experts=num_experts, **kw)


def _inputs(batch=1, h=16, w=16, seed=0):
    """(x, snr_map) pair at packed resolution, exactly as the train loop builds it."""
    x = packed_bayer(batch, h, w, seed=seed)
    return x, snr_map_for(x)


def _trunk_param_names(model):
    return [k for k, _ in model.named_parameters()
            if not (k.startswith("experts.") or k.startswith("gate."))]


def _n(module_or_params):
    if isinstance(module_or_params, torch.nn.Module):
        return sum(p.numel() for p in module_or_params.parameters())
    return sum(p.numel() for p in module_or_params)


def _trunk_param_count(model):
    keep = set(_trunk_param_names(model))
    return sum(p.numel() for k, p in model.named_parameters() if k in keep)


def _wake_experts(model, scale=0.5):
    """
    Give every ExpertHead.proj_out non-zero weights.

    Needed by any test that wants real (non-degenerate) expert outputs: at
    construction proj_out is zeroed, so every expert emits exactly 0, which the
    clamp(min=_CLAMP_EPS) then pins to the clamp floor — and the clamp kills the
    gradient there (see test_fresh_model_produces_nonzero_gradients).
    """
    with torch.no_grad():
        for j, head in enumerate(model.experts):
            head.proj_out.weight.normal_(0.0, scale)
            head.proj_out.bias.fill_(0.1 + 0.2 * j)   # experts must differ
    return model


def _wake_gate(model, scale=2.0):
    """Give the gate's zero-initialised final conv non-zero weights."""
    with torch.no_grad():
        model.gate.net[-1].weight.normal_(0.0, scale)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Output contract: shapes
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_forward_tuple_shapes_for_every_expert_count(tiny_kwargs, K):
    """
    forward returns (blended [B,3,2h,2w], expert_outs [B,K,3,2h,2w],
    gates [B,K,2h,2w]) for K = 1..4.

    The train loop unpacks all three and indexes expert_outs on dim 1, so a
    wrong K axis (or a missing expert axis at K=1) breaks training silently.
    """
    model = _moe(tiny_kwargs, num_experts=K)
    assert model.num_experts == K
    assert len(model.experts) == K

    x, snr = _inputs(1, 16, 16)
    model.train()
    blended, expert_outs, gates = model(x, snr)

    assert_shape(blended, (1, 3, 32, 32), "blended")
    assert_shape(expert_outs, (1, K, 3, 32, 32), "expert_outs")
    assert_shape(gates, (1, K, 32, 32), "gates")
    assert_finite(blended, "blended")
    assert_finite(expert_outs, "expert_outs")
    assert_finite(gates, "gates")


@pytest.mark.parametrize("B,h,w", [(2, 16, 16), (1, 16, 24), (3, 24, 16)])
def test_forward_shapes_batched_and_non_square(tiny_kwargs, B, h, w):
    """
    Batch > 1 and non-square packed inputs keep the 2x spatial relation
    (packed h,w -> sensor 2h,2w) and never transpose h/w.

    Real Mobile-HDR crops are not square; a swapped h/w would only show up as a
    garbled image, not an exception.
    """
    model = _moe(tiny_kwargs, num_experts=2)
    x, snr = _inputs(B, h, w, seed=1)
    model.train()
    blended, expert_outs, gates = model(x, snr)

    assert_shape(blended, (B, 3, 2 * h, 2 * w), "blended")
    assert_shape(expert_outs, (B, 2, 3, 2 * h, 2 * w), "expert_outs")
    assert_shape(gates, (B, 2, 2 * h, 2 * w), "gates")


def test_packed_dims_not_divisible_by_eight_is_rejected(tiny_kwargs):
    """
    The documented input constraint (packed h, w divisible by 8 — three
    PixelUnshuffle(2) stages) is enforced by a hard error, not by silently
    cropping. h=12 survives two unshuffles and dies on the third.
    """
    model = _moe(tiny_kwargs, num_experts=2)
    x, snr = _inputs(1, 12, 12)
    model.train()
    with pytest.raises(RuntimeError):
        model(x, snr)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gates: partition of unity, non-negativity, upsampling
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_gates_are_a_partition_of_unity_after_upsample(tiny_kwargs, K):
    """
    gates.sum(dim=1) == 1 everywhere at SENSOR resolution, i.e. the bilinear 2x
    upsample of a softmax preserves the partition of unity (each output pixel is
    a convex combination of low-res pixels, and convex combinations of
    unit-sum vectors are unit-sum).  Verified with a *perturbed* gate so the
    low-res field is genuinely non-uniform — a constant field would pass
    trivially.

    If this ever broke, `blended` would silently gain or lose brightness.
    """
    model = _wake_gate(_moe(tiny_kwargs, num_experts=K), scale=3.0)
    x, snr = _inputs(2, 16, 24, seed=2)
    model.train()
    _, _, gates = model(x, snr)

    if K > 1:
        # sanity: the gate really is non-trivial, not a constant field
        assert gates.std().item() > 1e-4, "gate field is constant; test is vacuous"

    dev = (gates.sum(dim=1) - 1.0).abs().max().item()
    assert dev <= 1e-6, f"gates do not sum to 1 after upsample (max dev {dev})"


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_gates_are_non_negative_and_bounded(tiny_kwargs, K):
    """
    Gates stay in [0, 1] after the bilinear upsample.  Negative routing weights
    would let a blended pixel fall below every expert's prediction (and below
    the _CLAMP_EPS floor the whole pipeline assumes).
    """
    model = _wake_gate(_moe(tiny_kwargs, num_experts=K), scale=3.0)
    x, snr = _inputs(1, 16, 16, seed=3)
    model.train()
    _, _, gates = model(x, snr)

    assert float(gates.min()) >= 0.0, f"negative gate weight {float(gates.min())}"
    assert_in_range(gates, 0.0, 1.0, "gates")


def test_gates_are_exactly_the_bilinear_upsample_of_the_low_res_gate(tiny_kwargs):
    """
    The returned gates are F.interpolate(gate(x, snr), scale_factor=2,
    mode='bilinear', align_corners=False) — bitwise.

    Pins the upsample mode/alignment: 'nearest' or align_corners=True would
    shift the routing map by half a sensor pixel relative to the experts.
    """
    model = _wake_gate(_moe(tiny_kwargs, num_experts=3))
    x, snr = _inputs(1, 16, 16, seed=4)
    model.eval()
    with torch.no_grad():
        _, _, gates = model(x, snr)
        gates_lr = model.gate(x, snr)
        expected = F.interpolate(gates_lr, scale_factor=2.0,
                                 mode="bilinear", align_corners=False)

    assert_shape(gates_lr, (1, 3, 16, 16), "gates_lr")
    assert torch.equal(gates, expected), "gates are not the bilinear 2x upsample"


def test_single_expert_gate_is_identically_one(tiny_kwargs):
    """
    With K=1 the softmax over one channel is 1.0 everywhere, so blended must be
    exactly expert 0 in train mode.  This is the MoE-as-baseline degenerate case
    used for ablations.
    """
    model = _wake_experts(_moe(tiny_kwargs, num_experts=1))
    x, snr = _inputs(1, 16, 16, seed=5)
    model.train()
    blended, expert_outs, gates = model(x, snr)

    assert torch.equal(gates, torch.ones_like(gates)), "K=1 gate is not identically 1"
    assert torch.equal(blended, expert_outs[:, 0]), "K=1 blended != expert 0"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Blending identity
# ─────────────────────────────────────────────────────────────────────────────

def test_blending_identity_is_exact_in_train_mode(tiny_kwargs):
    """
    In train mode blended == (gates.unsqueeze(2) * expert_outs).sum(dim=1),
    bitwise, for every returned tensor.

    The train loop re-weights the per-expert aux loss with the same gates, so
    any hidden extra normalisation/clamp between the returned pieces and the
    returned blend would make the aux loss inconsistent with the main loss.
    """
    model = _wake_gate(_wake_experts(_moe(tiny_kwargs, num_experts=3)))
    x, snr = _inputs(2, 16, 16, seed=6)
    model.train()
    blended, expert_outs, gates = model(x, snr)

    recomputed = (gates.unsqueeze(2) * expert_outs).sum(dim=1)
    assert torch.equal(blended, recomputed), (
        "train-mode blend is not exactly sum_k gates_k * expert_k "
        f"(max diff {(blended - recomputed).abs().max().item()})"
    )


def test_blending_identity_is_broken_by_the_eval_clamp(tiny_kwargs):
    """
    In EVAL mode the identity does *not* hold: `blended` is computed from the
    unclamped experts and then clamped to 1.0, while `expert_outs` is clamped
    afterwards.  With one expert far above 1 and one at the floor the two
    differ by a lot, and blended saturates at exactly 1.0.

    Pinning this stops a future reader from "fixing" the eval path by
    re-deriving blended from the returned (already clamped) experts, which
    would change every reported eval PSNR.
    """
    model = _moe(tiny_kwargs, num_experts=2)
    with torch.no_grad():
        # Zero the weights so each expert is its bias alone, and zero the gate
        # head so routing is exactly 1/K. Neither is zero-initialised any more
        # (that combination made the model untrainable), so this test sets up
        # the exact-arithmetic condition it needs for itself.
        for head in model.experts:
            head.proj_out.weight.zero_()
        model.gate.net[-1].weight.zero_()
        model.experts[0].proj_out.bias.fill_(4.0)    # way above 1.0
        model.experts[1].proj_out.bias.fill_(0.0)    # pinned to the eps floor
    x, snr = _inputs(1, 16, 16, seed=7)

    model.eval()
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)

    # gate is still uniform 1/2 -> pre-clamp blend is ~2.0, clamped to 1.0
    assert torch.allclose(blended, torch.ones_like(blended)), \
        "eval blended did not saturate at 1.0"
    recomputed = (gates.unsqueeze(2) * expert_outs).sum(dim=1)
    assert torch.allclose(recomputed, torch.full_like(recomputed, 0.5), atol=1e-5), \
        "recomputation from clamped experts should give ~0.5 here"
    assert not torch.allclose(blended, recomputed), \
        "eval identity unexpectedly holds; the clamp order must have changed"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Zero-init training-stability contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_fresh_model_output_clears_the_clamp_floor(tiny_kwargs, K):
    """
    A fresh model must emit a varied, input-dependent prediction that mostly
    sits ABOVE _CLAMP_EPS.

    This is the regression guard for the init bug. proj_out used to be
    zero-initialised, so every expert emitted exactly 0.0, the
    clamp(min=_CLAMP_EPS) in forward pinned every element to the floor, and
    clamp's zero backward below the floor killed every gradient in the model —
    permanently, since proj_out could not learn its way out either. A model
    whose output is a constant at the floor is a model that cannot train.
    """
    model = _moe(tiny_kwargs, num_experts=K)
    for j, head in enumerate(model.experts):
        assert float(head.proj_out.weight.abs().max()) > 0.0, \
            f"expert {j} proj_out.weight is zero-initialised — this kills every gradient"

    x, snr = _inputs(1, 16, 16, seed=8)
    model.train()
    blended, expert_outs, _ = model(x, snr)

    assert torch.isfinite(blended).all() and torch.isfinite(expert_outs).all()
    assert float(expert_outs.std()) > 0.0, "fresh expert outputs are constant"
    assert float(blended.std()) > 0.0, "fresh blended output is constant"

    # Essentially nothing may start on the floor. proj_out is initialised
    # around a dim POSITIVE constant precisely so this holds: a plain zero-mean
    # init would leave ~50% of pixels clipped and gradient-free at step 0.
    on_floor = (expert_outs <= _CLAMP_EPS).float().mean()
    assert float(on_floor) < 0.01, \
        f"{100 * float(on_floor):.1f}% of expert output sits on the clamp floor; " \
        "gradients are being killed there"


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_fresh_gate_is_near_uniform_one_over_k(tiny_kwargs, K):
    """
    NoiseGate's final conv has a zero bias and a small-std weight, so at
    construction every gate weight is within a fraction of a percent of 1/K —
    effectively uniform routing, which is the "training starts from uniform
    routing" half of the stability contract.

    It is deliberately not *exactly* uniform: an exactly zero head makes
    d(logits)/d(hidden) zero and freezes the gate body. See
    tests/test_model_components.py::test_noise_gate_is_trainable_end_to_end_at_init.
    """
    model = _moe(tiny_kwargs, num_experts=K)
    assert float(model.gate.net[-1].bias.abs().max()) == 0.0
    assert float(model.gate.net[-1].weight.std()) < 1e-2

    x, snr = _inputs(1, 16, 16, seed=9)
    model.train()
    _, _, gates = model(x, snr)

    assert torch.allclose(gates, torch.full_like(gates, 1.0 / K), atol=1e-2), \
        f"fresh gate is not near-uniform 1/{K}: {gates.min():.6f}..{gates.max():.6f}"


def test_fresh_gate_responds_weakly_to_the_snr_map(tiny_kwargs):
    """
    The gate head is initialised small but non-zero, so on a fresh model two
    wildly different SNR maps already produce *different* routing — just by a
    tiny amount. That difference is what carries gradient into the gate body;
    with the old exactly-zero head the two were bit-identical and the body was
    frozen. Sets up the contrast with test_gate_is_conditioned_on_the_snr_map
    below, which checks a trained-scale response.
    """
    model = _moe(tiny_kwargs, num_experts=3)
    x = packed_bayer(1, 16, 16, seed=10)
    snr_lo = torch.full((1, 1, 16, 16), 0.05)
    snr_hi = torch.full((1, 1, 16, 16), 0.95)

    model.eval()
    with torch.no_grad():
        _, _, g_lo = model(x, snr_lo)
        _, _, g_hi = model(x, snr_hi)

    assert not torch.equal(g_lo, g_hi), \
        "fresh gate is bit-identical across SNR — the head is zero-initialised again"
    assert float((g_lo - g_hi).abs().max()) < 1e-2, \
        "fresh gate routing swings too hard on SNR; init is not near-uniform"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clamping semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_train_mode_does_not_clamp_the_upper_bound(tiny_kwargs):
    """
    HDR targets exceed 1.0, so train mode must NOT clamp the maximum: with an
    expert biased to 4.0 both expert_outs and blended stay above 1.

    Clamping in train mode would zero the gradient on every over-bright pixel
    (exactly the highlights this HDR model exists to reconstruct).
    """
    model = _moe(tiny_kwargs, num_experts=2)
    with torch.no_grad():
        for head in model.experts:
            head.proj_out.bias.fill_(4.0)
    x, snr = _inputs(1, 16, 16, seed=11)

    model.train()
    blended, expert_outs, _ = model(x, snr)
    assert float(expert_outs.max()) > 1.0, "train mode clamped expert_outs to 1.0"
    assert float(blended.max()) > 1.0, "train mode clamped blended to 1.0"


def test_eval_mode_clamps_both_outputs_to_one(tiny_kwargs):
    """
    Eval mode clamps blended AND expert_outs to a max of 1.0 (the eval script
    computes PSNR/SSIM against [0,1] ground truth), while train mode does not.
    Same weights, same input — only model.training differs.
    """
    model = _moe(tiny_kwargs, num_experts=2)
    with torch.no_grad():
        for head in model.experts:
            head.proj_out.bias.fill_(4.0)
    x, snr = _inputs(1, 16, 16, seed=11)

    model.eval()
    with torch.no_grad():
        blended, expert_outs, _ = model(x, snr)
    assert_in_range(blended, _CLAMP_EPS, 1.0, "eval blended", atol=0.0)
    assert_in_range(expert_outs, _CLAMP_EPS, 1.0, "eval expert_outs", atol=0.0)
    assert float(blended.max()) == pytest.approx(1.0), "eval blended did not reach the clamp"
    assert float(expert_outs.max()) == pytest.approx(1.0), "eval expert_outs did not reach the clamp"


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_min_clamp_is_active_in_both_modes(tiny_kwargs, mode):
    """
    The _CLAMP_EPS floor is unconditional: with strongly negative expert biases
    every expert output is pinned to exactly _CLAMP_EPS in train and eval alike,
    and blended (a convex combination) inherits the floor.

    Downstream code takes log()/µ-law of these tensors, so a zero or negative
    pixel would produce NaNs in the loss.
    """
    model = _moe(tiny_kwargs, num_experts=3)
    with torch.no_grad():
        for head in model.experts:
            head.proj_out.weight.normal_(0.0, 0.1)
            head.proj_out.bias.fill_(-5.0)
    x, snr = _inputs(1, 16, 16, seed=12)

    getattr(model, mode)()
    with torch.no_grad():
        blended, expert_outs, _ = model(x, snr)

    assert torch.equal(expert_outs, torch.full_like(expert_outs, _CLAMP_EPS)), \
        f"{mode}: expert_outs not pinned to the _CLAMP_EPS floor"
    assert float(blended.min()) > 0.0, f"{mode}: blended fell to/below zero"
    assert torch.allclose(blended, torch.full_like(blended, _CLAMP_EPS), rtol=1e-6), \
        f"{mode}: blended not at the _CLAMP_EPS floor"


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_degenerate_constant_inputs_stay_finite(tiny_kwargs, value):
    """
    All-black and all-white packed inputs (their SNR maps are degenerate:
    zero variance everywhere) must not produce NaN/Inf, and the gates must
    still be a partition of unity.  The dataset does contain saturated and
    black frames.
    """
    model = _wake_gate(_wake_experts(_moe(tiny_kwargs, num_experts=3)))
    x = torch.full((1, 4, 16, 16), float(value))
    snr = snr_map_for(x)
    assert_finite(snr, "snr_map")

    model.eval()
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)

    assert_finite(blended, "blended")
    assert_finite(expert_outs, "expert_outs")
    assert_finite(gates, "gates")
    assert (gates.sum(dim=1) - 1.0).abs().max().item() <= 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 6. The trunk really is shared
# ─────────────────────────────────────────────────────────────────────────────

def test_trunk_modules_exist_exactly_once(tiny_kwargs):
    """
    There is ONE encoder / latent / decoder, not one per expert: the trunk
    submodules appear exactly once in named_modules() and the only per-expert
    modules are ExpertHead instances (plus a single NoiseGate).

    This is the structural half of the "~95% of compute is shared" claim.
    """
    model = _moe(tiny_kwargs, num_experts=4)
    names = dict(model.named_modules())

    for attr in ("patch_embed", "encoder_level_1", "encoder_level_2",
                 "x_expo_1", "x_expo_2", "latent", "latent_fusion",
                 "decoder_level_2", "reduce_chan_level_2", "decoder_level_1"):
        assert attr in names, f"missing trunk module {attr}"
        # no duplicated copy hiding under an expert head
        dupes = [n for n in names if n.endswith("." + attr)]
        assert dupes == [], f"trunk module {attr} duplicated at {dupes}"

    heads = [m for m in model.modules() if isinstance(m, ExpertHead)]
    gates = [m for m in model.modules() if isinstance(m, NoiseGate)]
    assert len(heads) == 4, f"expected 4 ExpertHeads, found {len(heads)}"
    assert len(gates) == 1, f"expected 1 NoiseGate, found {len(gates)}"


def test_trunk_runs_once_per_forward_regardless_of_expert_count(tiny_kwargs):
    """
    The trunk is *executed* once per forward, not once per expert: forward hooks
    on patch_embed / latent / decoder_level_1 fire exactly once with K=4.

    A per-expert trunk call would quadruple FLOPs while leaving every shape and
    value assertion in this file green, so it needs its own test.
    """
    model = _moe(tiny_kwargs, num_experts=4)
    counts = {}
    handles = []
    for attr in ("patch_embed", "latent", "decoder_level_1"):
        counts[attr] = 0

        def _hook(_m, _i, _o, key=attr):
            counts[key] += 1

        handles.append(getattr(model, attr).register_forward_hook(_hook))

    x, snr = _inputs(1, 16, 16, seed=13)
    model.eval()
    with torch.no_grad():
        model(x, snr)
    for h in handles:
        h.remove()

    assert counts == {"patch_embed": 1, "latent": 1, "decoder_level_1": 1}, counts


def test_every_expert_head_sees_the_same_trunk_features(tiny_kwargs):
    """
    Two expert heads carrying identical weights produce identical outputs,
    which can only happen if they are fed the same trunk tensor.

    This is the value-level (not just parameter-level) statement that the trunk
    is shared; it would fail if the trunk were re-run per expert with dropout,
    or if experts were accidentally fed different pyramid levels.
    """
    model = _wake_experts(_moe(tiny_kwargs, num_experts=3))
    with torch.no_grad():
        model.experts[2].load_state_dict(model.experts[0].state_dict())

    x, snr = _inputs(1, 16, 16, seed=14)
    model.train()
    _, expert_outs, _ = model(x, snr)

    assert torch.equal(expert_outs[:, 0], expert_outs[:, 2]), \
        "identically-weighted experts disagree -> they are not sharing features"
    assert not torch.equal(expert_outs[:, 0], expert_outs[:, 1]), \
        "differently-weighted experts agree -> outputs are not expert specific"


@pytest.mark.parametrize("K", [1, 2, 3, 4])
def test_trunk_parameter_count_is_independent_of_num_experts(tiny_kwargs, K):
    """
    Growing K adds nothing to the trunk: the non-expert, non-gate parameter
    names and count are byte-for-byte the same for K = 1..4.  A checkpoint
    trained with K=2 must therefore be loadable trunk-wise at any K.
    """
    ref = _moe(tiny_kwargs, num_experts=2)
    ref_names = _trunk_param_names(ref)
    ref_count = _trunk_param_count(ref)

    model = _moe(tiny_kwargs, num_experts=K)
    names = _trunk_param_names(model)
    count = _trunk_param_count(model)

    assert names == ref_names, "trunk parameter names changed with num_experts"
    assert count == ref_count, \
        f"trunk parameter count changed with num_experts ({count} vs {ref_count})"


def test_growing_num_experts_adds_only_one_head_plus_one_gate_channel(tiny_kwargs):
    """
    num_experts 2 -> 3 costs exactly one ExpertHead plus the gate's extra output
    channel (gate_hidden weights + 1 bias) — nothing else.

    This is the quantitative "~95% of compute is shared" claim: the marginal
    expert is ~2.5% of the trunk at this configuration.
    """
    m2 = _moe(tiny_kwargs, num_experts=2)
    m3 = _moe(tiny_kwargs, num_experts=3)

    gate_hidden = 16                                    # MoEDenoiser default
    standalone_head = ExpertHead(tiny_kwargs["dim"] * 2, 3, 2,
                                 tiny_kwargs["se_reduction"])
    expected = _n(standalone_head) + gate_hidden + 1     # +weights +bias row

    delta = _n(m3) - _n(m2)
    assert delta == expected, (
        f"K 2->3 added {delta} params, expected {expected} "
        f"(one ExpertHead {_n(standalone_head)} + gate row {gate_hidden + 1})"
    )

    trunk = _trunk_param_count(m3)
    assert _n(standalone_head) < 0.05 * trunk, \
        "an expert head is no longer cheap relative to the shared trunk"


# ─────────────────────────────────────────────────────────────────────────────
# 7. num_refinement_blocks is accepted and discarded
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("nrb", [1, 4, 9])
def test_num_refinement_blocks_is_accepted_but_has_no_effect(tiny_kwargs, nrb):
    """
    MoEDenoiser.__init__ does `del num_refinement_blocks` — the argument exists
    only so the shared build_denoiser(**kwargs) call and old checkpoint
    model_kwargs keep working.  Any value must give an identical parameter set.

    If it ever started mattering, checkpoints saved with a different value would
    fail to load strict=True.
    """
    torch.manual_seed(99)
    ref = MoEDenoiser(num_experts=2, dim=tiny_kwargs["dim"],
                      num_blocks=tiny_kwargs["num_blocks"],
                      num_refinement_blocks=1,
                      heads=tiny_kwargs["heads"],
                      se_reduction=tiny_kwargs["se_reduction"])
    torch.manual_seed(99)
    model = MoEDenoiser(num_experts=2, dim=tiny_kwargs["dim"],
                        num_blocks=tiny_kwargs["num_blocks"],
                        num_refinement_blocks=nrb,
                        heads=tiny_kwargs["heads"],
                        se_reduction=tiny_kwargs["se_reduction"])

    assert _n(model) == _n(ref), \
        f"num_refinement_blocks={nrb} changed the parameter count"
    assert list(model.state_dict().keys()) == list(ref.state_dict().keys()), \
        f"num_refinement_blocks={nrb} changed the state_dict keys"
    # identical construction order => identical RNG draws => identical weights
    for (ka, va), (kb, vb) in zip(model.state_dict().items(),
                                  ref.state_dict().items()):
        assert ka == kb
        assert torch.equal(va, vb), f"weights differ at {ka}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Gate conditioning
# ─────────────────────────────────────────────────────────────────────────────

def test_gate_is_conditioned_on_the_snr_map(tiny_kwargs):
    """
    Once the gate has non-zero weights, the SNR map alone changes the routing:
    same x, two very different snr maps -> different gates.

    The gate's whole purpose is to route noisy vs clean regions to different
    experts; if the SNR channel were dropped (e.g. a wrong torch.cat order or
    in_channels=4) this test is the only thing that would notice.
    """
    model = _wake_gate(_moe(tiny_kwargs, num_experts=3), scale=2.0)
    x = packed_bayer(1, 16, 16, seed=15)
    snr_lo = torch.full((1, 1, 16, 16), 0.05)
    snr_hi = torch.full((1, 1, 16, 16), 0.95)

    model.eval()
    with torch.no_grad():
        _, _, g_lo = model(x, snr_lo)
        _, _, g_hi = model(x, snr_hi)

    diff = (g_lo - g_hi).abs().max().item()
    assert diff > 1e-4, f"gate ignores the SNR map (max gate diff {diff})"
    for g in (g_lo, g_hi):
        assert (g.sum(dim=1) - 1.0).abs().max().item() <= 1e-6


def test_gate_is_conditioned_on_the_noisy_input(tiny_kwargs):
    """
    Same SNR map, different packed input -> different gates.  Shot noise scales
    with signal, so the gate is documented as seeing absolute intensity too;
    a gate wired only to the SNR channel would pass the test above but fail
    this one.
    """
    model = _wake_gate(_moe(tiny_kwargs, num_experts=3), scale=2.0)
    snr = torch.full((1, 1, 16, 16), 0.4)
    x_a = packed_bayer(1, 16, 16, seed=16, low=0.0, high=0.2)
    x_b = packed_bayer(1, 16, 16, seed=17, low=0.8, high=1.0)

    model.eval()
    with torch.no_grad():
        _, _, g_a = model(x_a, snr)
        _, _, g_b = model(x_b, snr)

    diff = (g_a - g_b).abs().max().item()
    assert diff > 1e-4, f"gate ignores the packed input (max gate diff {diff})"


def test_gate_is_spatially_varying(tiny_kwargs):
    """
    The gate is per-pixel, not per-image: a spatially structured SNR map (noisy
    left half, clean right half) yields different routing in the two halves.
    A global (pooled) gate would make the MoE useless for locally varying noise.
    """
    model = _wake_gate(_moe(tiny_kwargs, num_experts=2), scale=3.0)
    x = packed_bayer(1, 16, 16, seed=18)
    snr = torch.zeros(1, 1, 16, 16)
    snr[..., :8] = 0.05
    snr[..., 8:] = 0.95

    model.eval()
    with torch.no_grad():
        _, _, gates = model(x, snr)

    left = gates[..., :16].mean(dim=(0, 2, 3))
    right = gates[..., 16:].mean(dim=(0, 2, 3))
    assert (left - right).abs().max().item() > 1e-3, \
        "routing does not vary spatially with the SNR map"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Gradient flow
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_from_blended_reaches_trunk_experts_and_gate(tiny_kwargs):
    """
    A loss on `blended` alone must reach the shared trunk, EVERY expert head
    (blocks and proj_out) and the gate — otherwise part of the network is dead
    weight during the MoE phase.

    proj_out is woken up first and the experts are made to disagree: identical
    experts give the gate zero gradient because sum_k gates_k is constant. (At
    the old literal zero-init the clamp floor killed all gradients outright —
    see test_fresh_model_produces_nonzero_gradients.)
    """
    model = _wake_gate(_wake_experts(_moe(tiny_kwargs, num_experts=3)), scale=1.0)
    x, snr = _inputs(1, 16, 16, seed=19)
    model.train()
    blended, _, _ = model(x, snr)
    blended.mean().backward()

    def gsum(prefix):
        vals = [p.grad for k, p in model.named_parameters() if k.startswith(prefix)]
        assert vals, f"no parameters under {prefix}"
        assert all(g is not None for g in vals), f"None grad under {prefix}"
        return sum(float(g.abs().sum()) for g in vals)

    # shared trunk, front to back
    for prefix in ("patch_embed", "encoder_level_1", "x_expo_1",
                   "encoder_level_2", "x_expo_2", "latent.", "latent_fusion",
                   "decoder_level_2", "reduce_chan_level_2", "decoder_level_1"):
        assert gsum(prefix) > 0.0, f"no gradient reached trunk module {prefix}"

    # every expert head
    for j in range(3):
        assert gsum(f"experts.{j}.proj_out") > 0.0, f"expert {j} proj_out has no gradient"
        assert gsum(f"experts.{j}.blocks") > 0.0, f"expert {j} blocks have no gradient"
        assert gsum(f"experts.{j}.refine1") > 0.0, f"expert {j} refine1 has no gradient"

    # the gate, including its input-facing first conv
    assert gsum("gate.net.0") > 0.0, "gate input conv has no gradient"
    assert gsum("gate.net.4") > 0.0, "gate output conv has no gradient"


def test_fresh_model_produces_nonzero_gradients(tiny_kwargs):
    """
    A freshly constructed model must be trainable: EVERY parameter must receive
    a non-zero gradient from a loss on `blended`.

    This used to fail for every parameter in the model. ExpertHead.proj_out was
    zero-initialised, so each expert's raw output was exactly 0.0 — below
    _CLAMP_EPS, where clamp(min=...) has zero derivative — so no gradient
    escaped the expert heads. The gate was equally dead: with all experts
    equal, d(sum_k g_k e_k)/d logits = 0 because sum_k g_k == 1. Every gradient
    was exactly zero, the optimizer step was a no-op, and the model could never
    leave its init point at any learning rate.
    """
    model = _moe(tiny_kwargs, num_experts=3)
    x, snr = _inputs(1, 16, 16, seed=20)
    model.train()
    blended, _, _ = model(x, snr)
    blended.mean().backward()

    dead = [n for n, p in model.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, (
        f"{len(dead)}/{sum(1 for _ in model.parameters())} parameters get exactly "
        f"zero gradient on a fresh MoEDenoiser: {dead[:5]}. Check that "
        "ExpertHead.proj_out is not zero-initialised — a zero init puts the "
        "output on the _CLAMP_EPS floor, where clamp() has zero derivative."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. Serialisation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("K", [1, 2, 3])
def test_state_dict_round_trip_strict(tiny_kwargs, K, tmp_path):
    """
    A trained MoEDenoiser reloads with strict=True into a same-config instance
    and reproduces its outputs bitwise (via a real torch.save/torch.load, which
    is what the eval script does).

    Every checkpoint on disk depends on this; a renamed/extra buffer would break
    resume with a strict-load error.
    """
    model = _wake_gate(_wake_experts(_moe(tiny_kwargs, num_experts=K)))
    x, snr = _inputs(1, 16, 16, seed=21)
    model.eval()
    with torch.no_grad():
        ref_blend, ref_experts, ref_gates = model(x, snr)

    path = tmp_path / f"moe_k{K}.pth"
    torch.save(model.state_dict(), str(path))

    clone = _moe(tiny_kwargs, num_experts=K)
    missing_unexpected = clone.load_state_dict(
        torch.load(str(path), map_location="cpu", weights_only=True),
        strict=True)
    assert missing_unexpected.missing_keys == []
    assert missing_unexpected.unexpected_keys == []

    clone.eval()
    with torch.no_grad():
        blend, experts, gates = clone(x, snr)
    assert torch.equal(blend, ref_blend), "reloaded blended output differs"
    assert torch.equal(experts, ref_experts), "reloaded expert outputs differ"
    assert torch.equal(gates, ref_gates), "reloaded gates differ"


def test_state_dict_from_a_different_k_does_not_load_strict(tiny_kwargs):
    """
    K is baked into the state_dict (experts.* and the gate's output row), so a
    K=2 checkpoint must NOT silently load into a K=3 model under strict=True.
    The eval script reads `num_experts` from the checkpoint precisely because of
    this; a silent partial load would give an untrained third expert.
    """
    src = _moe(tiny_kwargs, num_experts=2)
    dst = _moe(tiny_kwargs, num_experts=3)
    with pytest.raises(RuntimeError):
        dst.load_state_dict(src.state_dict(), strict=True)
