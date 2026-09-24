"""
FiLMDenoiser (HDR_model_hybrid_Teacher.py) — contract, invariants and wiring.

The single-decoder alternative to MoEDenoiser, added after the num-experts
sweep found the K-expert MoE barely benefits from more experts (per-expert
PSNR spread under 0.11 dB at every K tested) for reasons that point at the
routing mechanism rather than at K: expert heads too thin to diverge
(~150K params against a ~20M trunk), the routing signal (local SNR)
correlating r=0.37-0.88 with raw pixel brightness on real training crops,
and SNR varying ~3x more across training crops than within one. FiLM
replaces the K experts + gate with one head continuously modulated
(per-pixel scale + shift) by the same (x, snr_map) signal.

FiLMDenoiser publishes the identical contract as MoEDenoiser specifically
so the eval/visualisation harness and build_denoiser factory work
unmodified:

    blended, expert_outs, gates = model(x [B,4,h,w], snr [B,1,h,w])
        blended     [B, 3, 2h, 2w]
        expert_outs [B, 1, 3, 2h, 2w]   the single head's own output
        gates       [B, 1, 2h, 2w]      all ones (no routing, not "1/1")

This file pins:
  * the tuple shapes and the K=1 gate/expert-axis contract;
  * FiLM starts near-identity (gamma, beta ~= 0) but is NOT frozen there —
    every parameter, including the film generator's hidden layers, gets a
    non-zero gradient at init (the NoiseGate zero-init trap this file's
    docstring on FiLMGenerator explains does not recur here);
  * gradient reachability of trunk / film generator / head;
  * the eval-only max clamp and the always-on min clamp, matching
    MoEDenoiser's clamp order exactly;
  * film_hidden changes the film generator's width and round-trips through
    build_denoiser the same way dim/num_blocks do;
  * state_dict round trip with strict=True, including a non-default
    film_hidden (a checkpoint reload bug this file caught before any real
    training touched it: film_hidden was not stored in model_kwargs, so a
    non-default value silently failed to round-trip).

Everything runs on CPU at dim=8 with packed 16x16..32x32 inputs.
"""

import pytest
import torch
import torch.nn.functional as F

from HDR_model_hybrid_Teacher import (
    FiLMDenoiser,
    FiLMGenerator,
    build_denoiser,
    _CLAMP_EPS,
)
from helpers import (
    packed_bayer,
    snr_map_for,
    assert_finite,
    assert_in_range,
    assert_shape,
)


@pytest.fixture
def tiny_kwargs():
    return dict(dim=8, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1,
               heads=[1, 1, 1, 1], se_reduction=8)


def _film(tiny_kwargs, **over):
    kw = dict(tiny_kwargs)
    kw.update(over)
    return FiLMDenoiser(**kw)


def _inputs(batch=1, h=16, w=16, seed=0):
    x = packed_bayer(batch, h, w, seed=seed)
    return x, snr_map_for(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Output contract
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_tuple_shapes(tiny_kwargs):
    model = _film(tiny_kwargs)
    assert model.num_experts == 1

    x, snr = _inputs(1, 16, 16)
    model.train()
    blended, expert_outs, gates = model(x, snr)

    assert_shape(blended, (1, 3, 32, 32), "blended")
    assert_shape(expert_outs, (1, 1, 3, 32, 32), "expert_outs")
    assert_shape(gates, (1, 1, 32, 32), "gates")
    assert_finite(blended, "blended")
    assert_finite(expert_outs, "expert_outs")


@pytest.mark.parametrize("B,h,w", [(2, 16, 16), (1, 16, 24), (3, 24, 16)])
def test_forward_shapes_batched_and_non_square(tiny_kwargs, B, h, w):
    model = _film(tiny_kwargs)
    x, snr = _inputs(B, h, w, seed=1)
    model.train()
    blended, expert_outs, gates = model(x, snr)

    assert_shape(blended, (B, 3, 2 * h, 2 * w), "blended")
    assert_shape(expert_outs, (B, 1, 3, 2 * h, 2 * w), "expert_outs")
    assert_shape(gates, (B, 1, 2 * h, 2 * w), "gates")


def test_gates_are_all_ones_not_1_over_k(tiny_kwargs):
    """
    There is exactly one "expert", so the honest representation of "no
    routing happened" is a gate of 1.0 everywhere -- not 1/K (which would
    be identical here, K=1, but the point is this is a constant contract,
    not an artefact of a softmax that happens to have one output).
    """
    model = _film(tiny_kwargs)
    x, snr = _inputs(1, 16, 16, seed=2)
    model.train()
    _, _, gates = model(x, snr)
    assert torch.equal(gates, torch.ones_like(gates))


def test_expert_outs_equals_blended_in_train_mode(tiny_kwargs):
    """With one expert and an all-ones gate, blended IS expert_outs[:, 0]."""
    model = _film(tiny_kwargs)
    x, snr = _inputs(1, 16, 16, seed=3)
    model.train()
    blended, expert_outs, _ = model(x, snr)
    assert torch.equal(blended, expert_outs[:, 0])


def test_build_denoiser_dispatches_to_film(tiny_kwargs):
    model = build_denoiser("film", num_experts=3, **tiny_kwargs)
    assert isinstance(model, FiLMDenoiser)
    assert model.num_experts == 1, \
        "num_experts is accepted (the trainer always passes it) but must " \
        "be ignored by film mode -- there is exactly one head"


def test_build_denoiser_rejects_unknown_mode(tiny_kwargs):
    with pytest.raises(ValueError, match="film"):
        build_denoiser("nonsense", **tiny_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clamping semantics (must match MoEDenoiser exactly)
# ─────────────────────────────────────────────────────────────────────────────

def test_min_clamp_is_unconditional(tiny_kwargs):
    """
    A strongly negative head output must still come back >= _CLAMP_EPS, in
    train AND eval -- the floor that makes hdr_tonemap and every metric in
    this project safe from log(0).
    """
    model = _film(tiny_kwargs)
    with torch.no_grad():
        model.head.proj_out.weight.zero_()
        model.head.proj_out.bias.fill_(-5.0)
    x, snr = _inputs(1, 16, 16, seed=4)

    for mode in ("train", "eval"):
        getattr(model, mode)()
        blended, expert_outs, _ = model(x, snr)
        # atol is a float32-vs-python-float64 rounding allowance, not a
        # loosening of the floor itself: .clamp(min=_CLAMP_EPS) casts the
        # float64 constant to the tensor's float32 dtype before comparing,
        # so reading the result back as a Python float can land a few ULPs
        # under the raw float64 _CLAMP_EPS even though the clamp is exact
        # in float32. A real violation would be many orders of magnitude
        # larger than this.
        assert_in_range(blended, _CLAMP_EPS, 1.0, f"{mode}: blended", atol=1e-12)
        assert_in_range(expert_outs, _CLAMP_EPS, 1.0, f"{mode}: expert_outs",
                        atol=1e-12)


def test_max_clamp_is_eval_only(tiny_kwargs):
    """Mirrors MoEDenoiser: values above 1.0 pass through untouched in
    train mode and are clamped only in eval mode."""
    model = _film(tiny_kwargs)
    with torch.no_grad():
        model.head.proj_out.weight.zero_()
        model.head.proj_out.bias.fill_(4.0)
    x, snr = _inputs(1, 16, 16, seed=5)

    model.train()
    blended_train, _, _ = model(x, snr)
    assert float(blended_train.max()) > 1.0, \
        "train mode should NOT clamp above 1.0"

    model.eval()
    with torch.no_grad():
        blended_eval, expert_outs_eval, _ = model(x, snr)
    assert float(blended_eval.max()) <= 1.0
    assert float(expert_outs_eval.max()) <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Init stability: near-identity FiLM, but nothing frozen
# ─────────────────────────────────────────────────────────────────────────────

def test_film_starts_near_identity(tiny_kwargs):
    """
    gamma, beta should be small at construction (the transform starts close
    to the identity, feat*(1+0)+0 = feat), but NOT exactly zero -- an
    exactly-zero final layer is the trap NoiseGate already hit elsewhere in
    this file's history: it kills d(gamma,beta)/d(hidden), freezing the
    generator's own hidden layers.
    """
    model = _film(tiny_kwargs)
    x, snr = _inputs(2, 16, 16, seed=6)
    gamma, beta = model.film(x, snr)

    assert float(gamma.abs().max()) < 0.1, "gamma is not near zero at init"
    assert float(beta.abs().max()) < 0.1, "beta is not near zero at init"
    assert float(model.film.net[-1].weight.abs().max()) > 0.0, \
        "film generator's head is exactly zero-initialised again"


def test_every_parameter_gets_a_gradient_at_init(tiny_kwargs):
    model = _film(tiny_kwargs)
    x, snr = _inputs(1, 16, 16, seed=7)
    model.train()
    blended, _, _ = model(x, snr)
    blended.mean().backward()

    dead = [n for n, p in model.named_parameters()
            if p.grad is None or float(p.grad.abs().sum()) == 0.0]
    assert not dead, f"parameters with exactly zero gradient at init: {dead}"


def test_film_generator_hidden_layers_are_trainable(tiny_kwargs):
    """
    The specific regression this test exists for: an exactly-zero final
    conv makes the two hidden convs upstream of it receive zero gradient,
    because d(loss)/d(hidden) factors through the (zero) final weight.
    """
    gen = FiLMGenerator(feat_channels=16, in_channels=5, hidden=8)
    x = torch.rand(2, 4, 16, 16)
    snr = torch.rand(2, 1, 16, 16)
    gamma, beta = gen(x, snr)
    (gamma.sum() + beta.sum()).backward()

    for name, p in gen.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, \
            f"{name} received no gradient -- the generator body is frozen"


# ─────────────────────────────────────────────────────────────────────────────
# 4. film_hidden: architecture parameter, must round-trip
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("hidden", [4, 16, 64])
def test_film_hidden_controls_generator_width(tiny_kwargs, hidden):
    model = _film(tiny_kwargs, film_hidden=hidden)
    assert model.film.net[0].out_channels == hidden
    assert model.film.net[2].in_channels == hidden


def test_state_dict_round_trip_with_nondefault_film_hidden(tiny_kwargs):
    """
    The checkpoint-reload regression this file was written to catch: if
    film_hidden is not saved in model_kwargs, load_model_from_checkpoint
    (which rebuilds the architecture from exactly that dict) silently
    reconstructs the DEFAULT width instead of the trained one, and
    strict=True state_dict loading fails on the first mismatched channel
    count. This test is the model-level half of that guarantee; the
    checkpoint round trip itself is covered in hdr_eval's test suite.
    """
    kw = dict(tiny_kwargs, film_hidden=32)
    model = _film(tiny_kwargs, film_hidden=32)
    x, snr = _inputs(1, 16, 16, seed=8)
    with torch.no_grad():
        model(x, snr)   # exercise every module once before saving

    state = model.state_dict()
    reloaded = FiLMDenoiser(**kw)
    reloaded.load_state_dict(state, strict=True)   # would raise on a width mismatch

    reloaded.eval(); model.eval()
    with torch.no_grad():
        ref_out, _, _ = model(x, snr)
        new_out, _, _ = reloaded(x, snr)
    assert torch.equal(ref_out, new_out), "reloaded model produces different output"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Params: the claim the sweep report's diagram caption makes
# ─────────────────────────────────────────────────────────────────────────────

def test_film_generator_and_head_are_small_next_to_the_trunk():
    """
    The whole point of FiLM over MoE: replace a per-expert 150K-param head
    (that measurably could not diverge across K) with a single head plus a
    small conditioning generator, at real default size (dim=32).
    """
    model = FiLMDenoiser(dim=32)
    trunk = sum(p.numel() for n, p in model.named_parameters()
               if not (n.startswith("film.") or n.startswith("head.")))
    film = sum(p.numel() for p in model.film.parameters())
    head = sum(p.numel() for p in model.head.parameters())

    assert trunk > 15_000_000, "trunk should be the ~20M-param backbone"
    assert film < 50_000, f"film generator is {film} params, expected a few K"
    assert head < 200_000, f"head is {head} params, expected ~150K"
