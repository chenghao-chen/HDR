"""
tests/test_model_factory.py — build_denoiser + the legacy wrappers.
====================================================================

`build_denoiser` is the single seam between the two entry-point scripts
(train_A100_MoE_two_phase.py, test_dual_MoE_two_phase.py) and the three model
families.  Both scripts do the same thing with whatever comes back:

    model = build_denoiser(mode, num_experts=K, **model_kwargs)
    K = model.num_experts
    blended, expert_outs, gates = model(x, snr_map)

So the factory has to guarantee three separate things, and this file tests each
of them:

  1. dispatch    — the right class for the right (case-insensitive) mode name,
                   and a loud ValueError for anything else;
  2. uniformity  — every mode exposes `.num_experts` and the identical
                   forward signature / 3-tuple with mutually consistent shapes,
                   so one downstream loss can consume all three;
  3. isolation   — the MoE-only knobs (num_experts, expert_blocks, gate_hidden)
                   are absorbed by the factory and never reach the legacy
                   constructors, which would raise TypeError on them.

Plus the value-level semantics of the two legacy wrappers (the SNR blend
direction of DualSNRDenoiser and the degenerate gate of SingleDenoiser), which
are easy to get backwards and impossible to notice from shapes alone.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import assert_finite, assert_shape, packed_bayer, snr_map_for

from HDR_model_hybrid_Teacher import (
    _CLAMP_EPS,
    DualSNRDenoiser,
    DualSNRTeacher,
    MoEDenoiser,
    SingleDenoiser,
    build_denoiser,
)

MODES = ["moe", "dual", "single"]

EXPECTED_CLASS = {
    "moe": MoEDenoiser,
    "dual": DualSNRDenoiser,
    "single": SingleDenoiser,
}

# .num_experts each mode reports when the factory is called with its default
# num_experts=3.  The legacy wrappers hardcode their own K.
DEFAULT_K = {"moe": 3, "dual": 2, "single": 1}


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — see the harness rules)
# ─────────────────────────────────────────────────────────────────────────────

def _inputs(batch=1, h=16, w=16, seed=0):
    """A (packed BGGR, canonical SNR map) pair at packed resolution h x w."""
    x = packed_bayer(batch, h, w, seed=seed)
    return x, snr_map_for(x)


def _gate_weighted_loss(blended, expert_outs, gates, target):
    """
    A *downstream consumer* of the unified 3-tuple, mimicking the shape of the
    training objective: an L1 term on the blended output plus a per-expert L1
    term weighted by that expert's gate.  It only ever uses the contract
    (blended [B,3,S,S], expert_outs [B,K,3,S,S], gates [B,K,S,S]) — never the
    mode.  If any mode's tuple deviates, this either broadcasts wrongly or
    raises, which is exactly what we want to detect.
    """
    main = (blended - target).abs().mean()
    per_expert = (gates.unsqueeze(2) * (expert_outs - target.unsqueeze(1)).abs())
    return main + per_expert.mean()


# _CLAMP_EPS is a Python double; the tensors are float32, and the nearest
# float32 to it is *below* the double, so a naive `>= _CLAMP_EPS` comparison
# fails on a correctly clamped tensor.  Compare against the float32 value, with
# a whisker of relative slack for `blended`, which in moe/dual mode is a
# gate-weighted sum of already-clamped experts (mathematically >= the floor,
# but the weighted sum can land one ulp below it).
_EPS32 = float(torch.tensor(_CLAMP_EPS, dtype=torch.float32))


def _assert_min_clamped(t, name):
    """The models promise a hard floor of _CLAMP_EPS in every mode."""
    floor = _EPS32 * (1.0 - 1e-5)
    assert float(t.min()) >= floor, \
        f"{name} min {float(t.min())!r} < _CLAMP_EPS {_EPS32!r}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dispatch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", MODES)
def test_factory_dispatches_to_the_expected_class(mode, tiny_kwargs):
    """
    "moe"/"dual"/"single" must map to MoEDenoiser/DualSNRDenoiser/
    SingleDenoiser exactly (`type is`, not merely isinstance — the wrappers are
    unrelated classes and a silent fallback to the wrong one would still pass
    every shape test while training a completely different network).
    """
    model = build_denoiser(mode, **tiny_kwargs)
    assert type(model) is EXPECTED_CLASS[mode], \
        f"mode {mode!r} built {type(model).__name__}"
    assert isinstance(model, nn.Module)
    assert model.training, "factory must hand back a module in train mode"


@pytest.mark.parametrize("spelling,mode", [
    ("MOE", "moe"), ("MoE", "moe"), ("mOe", "moe"),
    ("DUAL", "dual"), ("Dual", "dual"),
    ("SINGLE", "single"), ("Single", "single"),
])
def test_factory_mode_is_case_insensitive(spelling, mode, tiny_kwargs):
    """
    build_denoiser lowercases `mode`, so a checkpoint or config that stored
    "MoE" must rebuild the same architecture as "moe".  A case-sensitive
    factory would silently fall through to the ValueError branch and break
    checkpoint reloading.
    """
    model = build_denoiser(spelling, **tiny_kwargs)
    assert type(model) is EXPECTED_CLASS[mode]
    assert model.num_experts == DEFAULT_K[mode]


def test_factory_unknown_mode_raises_valueerror_naming_the_valid_modes():
    """
    An unknown mode must fail fast with a ValueError that (a) echoes the
    offending name and (b) lists all three valid modes — this message is the
    only feedback a user typo'ing MODE in the train script ever gets.  It must
    not be a KeyError, and it must not silently default to a mode.
    """
    with pytest.raises(ValueError) as excinfo:
        build_denoiser("Trible")
    msg = str(excinfo.value)
    # the normalised (lowercased) name is echoed back
    assert "trible" in msg
    for valid in MODES:
        assert valid in msg, f"error message does not mention {valid!r}: {msg}"


def test_factory_rejects_unknown_mode_before_building_anything(tiny_kwargs):
    """
    The mode check must happen before any expensive construction: passing a bad
    mode together with garbage architecture kwargs still yields ValueError (not
    a TypeError from a half-built submodule).
    """
    with pytest.raises(ValueError):
        build_denoiser("quad", dim=7, not_a_real_kwarg=object())


# ─────────────────────────────────────────────────────────────────────────────
# 2. num_experts + the uniform forward contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,requested,expected", [
    ("moe", 1, 1),
    ("moe", 2, 2),
    ("moe", 3, 3),
    ("moe", 4, 4),
    ("dual", 5, 2),     # legacy wrapper hardcodes 2 — request is ignored
    ("single", 7, 1),   # legacy wrapper hardcodes 1 — request is ignored
])
def test_num_experts_attribute_matches_the_architecture(mode, requested, expected,
                                                        tiny_kwargs):
    """
    Both scripts read `model.num_experts` to size their expert/gate
    accumulators (test_dual_MoE_two_phase.infer_patches allocates
    [1, K, 3, H, W] buffers from it).  It must equal the real number of experts
    the forward pass produces, for every mode — including the two legacy
    wrappers, whose K is fixed by the architecture and NOT by the caller's
    num_experts.
    """
    model = build_denoiser(mode, num_experts=requested, **tiny_kwargs)
    assert model.num_experts == expected

    x, snr = _inputs()
    model.eval()
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)
    assert expert_outs.shape[1] == expected
    assert gates.shape[1] == expected


@pytest.mark.parametrize("mode", MODES)
def test_uniform_forward_contract(mode, tiny_kwargs):
    """
    THE point of the factory: one signature, one 3-tuple, mutually consistent
    shapes.  Identical assertions for all three modes, on a batched, non-square
    input:

      blended     [B, 3, 2h, 2w]
      expert_outs [B, K, 3, 2h, 2w]
      gates       [B, K, 2h, 2w]      with K == model.num_experts

    plus the invariants downstream code relies on: gates are non-negative and
    sum to 1 across experts at every pixel, everything is finite float32 on the
    input's device, the output floor is _CLAMP_EPS, and `blended` really is the
    gate-weighted sum of `expert_outs` (train mode, where no post-hoc clamp can
    hide a mismatch).  If that identity fails, a gate-weighted loss is
    optimising something other than the model's own output.
    """
    b, h, w = 2, 16, 24
    model = build_denoiser(mode, num_experts=3, **tiny_kwargs)
    model.train()
    K = model.num_experts
    x, snr = _inputs(b, h, w)

    out = model(x, snr)
    assert isinstance(out, tuple) and len(out) == 3
    blended, expert_outs, gates = out

    assert_shape(blended, (b, 3, 2 * h, 2 * w), "blended")
    assert_shape(expert_outs, (b, K, 3, 2 * h, 2 * w), "expert_outs")
    assert_shape(gates, (b, K, 2 * h, 2 * w), "gates")

    for name, t in (("blended", blended), ("expert_outs", expert_outs), ("gates", gates)):
        assert_finite(t, name)
        assert t.dtype == torch.float32, f"{name} dtype {t.dtype}"
        assert t.device == x.device

    assert float(gates.min()) >= 0.0, "gates must be non-negative routing weights"
    assert torch.allclose(gates.sum(dim=1), torch.ones(b, 2 * h, 2 * w), atol=1e-6), \
        "gates must sum to 1 over the expert axis at every pixel"

    _assert_min_clamped(blended, "blended")
    _assert_min_clamped(expert_outs, "expert_outs")

    recon = (gates.unsqueeze(2) * expert_outs).sum(dim=1)
    assert torch.allclose(blended, recon, atol=1e-6), \
        "blended is not the gate-weighted sum of expert_outs"


@pytest.mark.parametrize("mode", MODES)
def test_forward_accepts_snr_map_as_keyword(mode, tiny_kwargs):
    """
    Every mode names its second argument `snr_map`, so generic code may call
    model(x, snr_map=...).  Renaming it in one wrapper would break only that
    mode — a trap worth a cheap guard.
    """
    model = build_denoiser(mode, **tiny_kwargs).eval()
    x, snr = _inputs()
    with torch.no_grad():
        by_pos = model(x, snr)
        by_kw = model(x, snr_map=snr)
    for a, bb in zip(by_pos, by_kw):
        assert torch.equal(a, bb)


@pytest.mark.parametrize("mode", MODES)
def test_eval_mode_clamps_every_mode_into_the_unit_range(mode, tiny_kwargs):
    """
    Contract: min-clamp to _CLAMP_EPS always, max-clamp to 1.0 only in eval.
    The eval clamp must apply to `blended` AND `expert_outs` in all three
    modes, otherwise the PSNR/SSIM numbers reported by the eval script are
    computed on out-of-gamut pixels for some modes but not others.
    """
    model = build_denoiser(mode, num_experts=2, **tiny_kwargs).eval()
    x, snr = _inputs()
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)

    for name, t in (("blended", blended), ("expert_outs", expert_outs)):
        _assert_min_clamped(t, name)
        assert float(t.max()) <= 1.0, f"{name} max {float(t.max())} > 1.0 in eval mode"

    # Gates are routing weights, not pixels: they are never clamped, only
    # required to be a valid distribution.
    assert torch.allclose(gates.sum(dim=1), torch.ones_like(gates[:, 0]), atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_cross_mode_downstream_consumer(mode, tiny_kwargs):
    """
    The whole justification for the shared signature: ONE loss function must be
    able to consume all three tuples untouched, and gradients must reach the
    model.  _gate_weighted_loss broadcasts gates [B,K,S,S] against
    expert_outs [B,K,3,S,S]; a mode whose gates lack the expert axis (or carry
    it at Bayer resolution) would either raise here or silently broadcast into
    a wrong-shaped tensor.
    """
    model = build_denoiser(mode, num_experts=2, **tiny_kwargs)
    model.train()
    x, snr = _inputs(1, 16, 16)
    target = torch.rand(1, 3, 32, 32)

    blended, expert_outs, gates = model(x, snr)
    loss = _gate_weighted_loss(blended, expert_outs, gates, target)

    assert loss.shape == (), "loss must reduce to a scalar for every mode"
    assert torch.isfinite(loss), f"{mode}: non-finite loss"
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads, "model exposes no trainable parameters"
    assert all(g is not None for g in grads), \
        f"{mode}: some parameters received no gradient from the unified loss"
    assert all(torch.isfinite(g).all() for g in grads), \
        f"{mode}: non-finite gradient from the unified loss"


@pytest.mark.parametrize("mode", ["moe", "dual", "single"])
def test_gradients_actually_reach_the_model_in_every_mode(mode, tiny_kwargs):
    """
    A freshly built model must be trainable: the unified gate-weighted loss has
    to produce a non-zero gradient somewhere, in every mode.  Phase-1 training
    starts from a random init, so a mode whose gradient is identically zero at
    initialisation can never learn anything — Adam on an all-zero gradient is a
    no-op, the weights never move, and the gradient stays zero forever.

    It used to hold for dual/single but fail for moe: ExpertHead.proj_out was
    zero-initialised so every head emitted exactly 0.0, and MoEDenoiser.forward
    clamps that with `.clamp(min=_CLAMP_EPS)`. clamp's gradient below the floor
    is 0, so `expert_outs` — and therefore `blended` — was a *constant* with
    respect to every trunk and expert parameter. Nor did the gate escape: with
    zero-init gate logits the softmax is exactly uniform and all experts are
    identical, so dL/dlogits is a constant vector, which the softmax Jacobian
    maps to exactly zero. Measured at the time: 30 Adam steps moved 0 of 113
    parameters and the output stayed pinned at _CLAMP_EPS. proj_out now starts
    around a dim positive constant and the gate head small-but-non-zero, so
    every mode is trainable from step 0.
    """
    model = build_denoiser(mode, num_experts=2, **tiny_kwargs)
    model.train()
    x, snr = _inputs(1, 16, 16)
    target = torch.rand(1, 3, 32, 32)

    blended, expert_outs, gates = model(x, snr)
    _gate_weighted_loss(blended, expert_outs, gates, target).backward()

    total = sum(float(p.grad.abs().sum())
                for p in model.parameters() if p.grad is not None)
    assert total > 0.0, f"{mode}: the unified loss produced an all-zero gradient"


def _teachers(mode, model):
    """The TransUNet_Teacher_HDR-shaped sub-network(s) inside a built model."""
    if mode == "moe":
        return [model]              # the MoE trunk carries the teacher topology
    if mode == "dual":
        return [model.denoiser_low_snr, model.denoiser_high_snr]
    return [model.denoiser]


@pytest.mark.parametrize("mode", MODES)
def test_factory_forwards_architecture_kwargs(mode, tiny_kwargs):
    """
    `dim` and `se_reduction` must actually reach the underlying network(s):
    se_reduction=None is the legacy pre-SE architecture whose checkpoints must
    keep loading, so it has to turn the SE block into nn.Identity everywhere,
    not just in the MoE trunk.
    """
    kwargs = dict(tiny_kwargs)
    kwargs["se_reduction"] = None
    model = build_denoiser(mode, **kwargs)

    for t in _teachers(mode, model):
        assert t.dim == tiny_kwargs["dim"]
        assert isinstance(t.encoder_level_1[0].se, nn.Identity), \
            "se_reduction=None must disable Squeeze-Excitation (legacy checkpoints)"

    # ... and an int enables it again.
    model_se = build_denoiser(mode, **tiny_kwargs)
    for t in _teachers(mode, model_se):
        assert not isinstance(t.encoder_level_1[0].se, nn.Identity)


# ─────────────────────────────────────────────────────────────────────────────
# 3. MoE-only kwargs must not leak into the legacy constructors
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,expected_k", [("dual", 2), ("single", 1)])
def test_moe_only_kwargs_are_absorbed_by_the_factory(mode, expected_k, tiny_kwargs):
    """
    num_experts / expert_blocks / gate_hidden are *named* parameters of
    build_denoiser, so for "dual"/"single" they are swallowed by the signature
    and never forwarded.  Documented behaviour, precisely:
      * the call does NOT raise;
      * the request is silently IGNORED — K stays 2 (dual) / 1 (single);
      * no gate / expert-head modules are created.
    The train script always passes num_experts=NUM_EXPERTS regardless of MODE,
    so this absorption is load-bearing.
    """
    model = build_denoiser(mode, num_experts=5, expert_blocks=9, gate_hidden=77,
                           **tiny_kwargs)
    assert type(model) is EXPECTED_CLASS[mode]
    assert model.num_experts == expected_k
    assert not hasattr(model, "gate"), "legacy wrapper must not build a NoiseGate"
    assert not hasattr(model, "experts"), "legacy wrapper must not build ExpertHeads"

    x, snr = _inputs()
    model.eval()
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)
    assert expert_outs.shape[1] == expected_k
    assert gates.shape[1] == expected_k


@pytest.mark.parametrize("cls", [DualSNRDenoiser, SingleDenoiser])
@pytest.mark.parametrize("bad_kwarg", ["num_experts", "expert_blocks", "gate_hidden"])
def test_legacy_wrappers_reject_moe_kwargs_when_called_directly(cls, bad_kwarg,
                                                                tiny_kwargs):
    """
    The mirror image of the test above: the legacy wrappers forward **kwargs
    straight into TransUNet_Teacher_HDR, which has no such parameters, so a
    direct call explodes with TypeError.  This pins WHY the factory must keep
    those three as named parameters — if someone ever moves them into **kwargs,
    every dual/single run started from the train script dies at construction.
    """
    with pytest.raises(TypeError) as excinfo:
        cls(**{bad_kwarg: 3}, **tiny_kwargs)
    assert bad_kwarg in str(excinfo.value)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DualSNRDenoiser semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_dual_zero_snr_selects_the_low_snr_expert(tiny_kwargs):
    """
    Blend direction, half one:  out = (1-snr)*low + snr*high.
    With an all-zero SNR map (maximally noisy) the output must be EXACTLY the
    low-SNR teacher and the gates exactly [1, 0].  Getting this backwards would
    route noisy pixels to the clean-image expert and be nearly invisible in
    aggregate PSNR — hence the exact, hand-built check.
    """
    model = build_denoiser("dual", **tiny_kwargs).eval()
    x = packed_bayer(1, 16, 16, seed=3)
    snr = torch.zeros(1, 1, 16, 16)
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)

    assert torch.equal(blended, expert_outs[:, 0]), \
        "snr=0 must reproduce the low-SNR expert exactly"
    assert torch.equal(gates[:, 0], torch.ones_like(gates[:, 0]))
    assert torch.equal(gates[:, 1], torch.zeros_like(gates[:, 1]))


def test_dual_unit_snr_selects_the_high_snr_expert(tiny_kwargs):
    """
    Blend direction, half two: an all-one SNR map (clean) must reproduce the
    high-SNR teacher exactly, gates [0, 1].  Together with the previous test
    this pins the sign of the blend beyond ambiguity.
    """
    model = build_denoiser("dual", **tiny_kwargs).eval()
    x = packed_bayer(1, 16, 16, seed=3)
    snr = torch.ones(1, 1, 16, 16)
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)

    assert torch.equal(blended, expert_outs[:, 1]), \
        "snr=1 must reproduce the high-SNR expert exactly"
    assert torch.equal(gates[:, 0], torch.zeros_like(gates[:, 0]))
    assert torch.equal(gates[:, 1], torch.ones_like(gates[:, 1]))


def test_dual_expert_outs_stacks_low_then_high(tiny_kwargs):
    """
    expert_outs[:, 0] is the LOW-SNR teacher and [:, 1] the HIGH-SNR one.
    The eval script visualises / scores individual experts by index, so a
    swapped stack order silently mislabels every per-expert metric.
    """
    model = build_denoiser("dual", **tiny_kwargs).eval()
    x, snr = _inputs(1, 16, 16, seed=5)
    with torch.no_grad():
        _, expert_outs, _ = model(x, snr)
        low = model.denoiser_low_snr(x)
        high = model.denoiser_high_snr(x)

    assert torch.equal(expert_outs[:, 0], low)
    assert torch.equal(expert_outs[:, 1], high)
    assert not torch.allclose(low, high), \
        "the two teachers are independently initialised and must differ"


def test_dual_gates_are_the_bilinearly_upsampled_snr_map(tiny_kwargs):
    """
    The SNR map arrives at packed-Bayer resolution but the teachers output at
    sensor resolution, so it must be upsampled 2x BILINEARLY before blending.
    A checkerboard SNR map makes the choice observable: bilinear invents
    intermediate weights (>2 distinct values), nearest would not.  Also pins
    gates == [1 - snr_up, snr_up], hence sum-to-1 exactly.
    """
    model = build_denoiser("dual", **tiny_kwargs).eval()
    x = packed_bayer(1, 16, 16, seed=7)
    snr = torch.zeros(1, 1, 16, 16)
    snr[..., ::2, ::2] = 1.0
    with torch.no_grad():
        blended, expert_outs, gates = model(x, snr)

    expected_up = F.interpolate(snr, size=(32, 32), mode="bilinear",
                                align_corners=False)
    assert torch.equal(gates[:, 1:2], expected_up), \
        "high-SNR gate must be the bilinearly upsampled SNR map"
    assert torch.equal(gates[:, 0:1], 1.0 - expected_up)
    assert torch.equal(gates.sum(dim=1), torch.ones(1, 32, 32))

    nearest_up = F.interpolate(snr, size=(32, 32), mode="nearest")
    assert not torch.allclose(gates[:, 1:2], nearest_up), \
        "gates look like a nearest-neighbour upsample, not bilinear"
    assert gates[:, 1].unique().numel() > snr.unique().numel(), \
        "bilinear upsampling must produce intermediate gate values"

    # And the blend really uses those upsampled weights, pixel for pixel.
    manual = (1.0 - expected_up) * expert_outs[:, 0] + expected_up * expert_outs[:, 1]
    assert torch.allclose(blended, manual, atol=1e-6)


def test_dual_is_two_independent_full_teachers(tiny_kwargs):
    """
    "Two independent full teachers" is the defining (and expensive) property of
    the legacy dual mode: exactly twice a SingleDenoiser's parameters, with no
    tensor shared between the two halves.  Any accidental weight sharing would
    halve capacity while still loading old checkpoints only partially.
    Also pins the state_dict prefixes old checkpoints were written with.
    """
    dual = build_denoiser("dual", **tiny_kwargs)
    single = build_denoiser("single", **tiny_kwargs)

    n_dual = sum(p.numel() for p in dual.parameters())
    n_single = sum(p.numel() for p in single.parameters())
    assert n_dual == 2 * n_single, f"dual has {n_dual} params, expected 2x{n_single}"

    low_ids = {id(p) for p in dual.denoiser_low_snr.parameters()}
    high_ids = {id(p) for p in dual.denoiser_high_snr.parameters()}
    assert low_ids and high_ids
    assert low_ids.isdisjoint(high_ids), "the two teachers share parameter tensors"

    prefixes = {k.split(".")[0] for k in dual.state_dict()}
    assert prefixes == {"denoiser_low_snr", "denoiser_high_snr"}, prefixes
    assert {k.split(".")[0] for k in single.state_dict()} == {"denoiser"}


def test_moe_shares_one_trunk_and_is_cheaper_than_dual(tiny_kwargs):
    """
    At the same expert count, MoE (one shared trunk + light heads) must have
    strictly fewer parameters than dual (two full teachers) — that is the
    entire reason the MoE variant exists, and a regression that duplicated the
    trunk per expert would show up here before it showed up on the GPU.
    """
    moe = build_denoiser("moe", num_experts=2, **tiny_kwargs)
    dual = build_denoiser("dual", num_experts=2, **tiny_kwargs)
    n_moe = sum(p.numel() for p in moe.parameters())
    n_dual = sum(p.numel() for p in dual.parameters())
    assert moe.num_experts == dual.num_experts == 2
    assert n_moe < n_dual, f"moe {n_moe} params >= dual {n_dual}"
    assert len(moe.experts) == 2


def test_dual_snr_teacher_is_an_alias_of_dual_snr_denoiser():
    """
    DualSNRTeacher is the old name kept for import-compatibility with scripts
    and pickles written before the rename.  It must be the SAME object, not a
    subclass or a copy, or `isinstance` checks and unpickling diverge.
    """
    assert DualSNRTeacher is DualSNRDenoiser


# ─────────────────────────────────────────────────────────────────────────────
# 5. SingleDenoiser semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_single_gates_are_all_ones_and_ignore_the_snr_map(tiny_kwargs):
    """
    The baseline has no routing: gates must be exactly ones of shape
    [B, 1, 2h, 2w] (a nearest upsample of ones_like(snr_map)), independent of
    the SNR map's contents — even a wild, out-of-range map.  Downstream code
    averages `gates` for load-balancing diagnostics, so a leaked SNR signal
    here would be reported as expert routing that does not exist.
    """
    model = build_denoiser("single", **tiny_kwargs).eval()
    x = packed_bayer(2, 16, 16, seed=11)
    wild = torch.randn(2, 1, 16, 16) * 5.0
    with torch.no_grad():
        _, _, gates = model(x, wild)

    assert_shape(gates, (2, 1, 32, 32), "gates")
    assert torch.equal(gates, torch.ones_like(gates)), \
        "single-mode gates must be identically 1.0"


def test_single_blended_is_exactly_the_bare_teacher_output(tiny_kwargs):
    """
    In single mode `blended` IS the one teacher's output — no gate arithmetic,
    no rescaling.  Multiplying by a gate of 1.0 would be a no-op numerically
    but any other post-processing would silently change the ablation baseline
    everything else is compared against.
    """
    model = build_denoiser("single", **tiny_kwargs).eval()
    x, snr = _inputs(1, 16, 16, seed=13)
    with torch.no_grad():
        blended, _, _ = model(x, snr)
        direct = model.denoiser(x)
    assert torch.equal(blended, direct)


def test_single_expert_outs_is_the_output_unsqueezed(tiny_kwargs):
    """
    expert_outs must be out.unsqueeze(1): shape [B, 1, 3, 2h, 2w] and bitwise
    identical to `blended`, so per-expert code paths (K-loops in the eval
    script) work unchanged at K=1.
    """
    model = build_denoiser("single", **tiny_kwargs).eval()
    x, snr = _inputs(2, 16, 16, seed=17)
    with torch.no_grad():
        blended, expert_outs, _ = model(x, snr)

    assert_shape(expert_outs, (2, 1, 3, 32, 32), "expert_outs")
    assert torch.equal(expert_outs[:, 0], blended)
    # unsqueeze is a view: same storage, so it is genuinely the same tensor
    # (this is why in-place edits of expert_outs would corrupt blended).
    assert expert_outs.data_ptr() == blended.data_ptr()
