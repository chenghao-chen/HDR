"""
test_eval_inference.py — inference + checkpoint-loading paths of
test_dual_MoE_two_phase.py
=============================================================================
Covered here:

  * load_model_from_checkpoint — architecture rebuild from the checkpoint
    metadata ('mode', 'model_kwargs', 'num_experts'), the legacy fallback
    path, the `_orig_mod.` (torch.compile) prefix strip, bit-exact weight
    restoration and the eval()-mode contract that the max-1.0 output clamp
    depends on.
  * infer_full   — reflect-pad packed h,w up to a multiple of 8, run, crop the
    RGB result back to exactly (2*h_orig, 2*w_orig).
  * infer_patches — the overlapping-tile arithmetic: padded extent is an exact
    number of strides, every output pixel is covered by >= 1 patch, the
    accumulate/divide yields (2H, 2W) and reduces to infer_full for a model
    whose output is a purely local function of its input.
  * estimate_flops — positive float with torchinfo, graceful None without it,
    and no mutation of the model.
  * OUTPUT_DIR derivation from CHECKPOINT (evaluated straight out of the
    module's own AST, since it lives behind the __main__ guard).

Everything runs on CPU. `torch.amp.autocast(device_type='cuda')` — which both
inference helpers call unconditionally — is *not* fatal on a CUDA-less box in
torch 2.5.1: it emits a UserWarning ("CUDA is not available. Disabling") and
becomes a no-op, so the real code path is exercised here rather than skipped.
A monkeypatched-autocast twin of each test proves the pad/crop/tile logic is
independent of autocast, and pins the hardcoded device_type as a portability
observation.
"""

import ast
import contextlib
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import test_dual_MoE_two_phase as ev
from HDR_model_hybrid_Teacher import _CLAMP_EPS, build_denoiser, estimate_local_snr_map

from helpers import (
    assert_finite,
    assert_in_range,
    assert_shape,
    make_checkpoint,
    packed_bayer,
    snr_map_for,
)

MODULE_PATH = Path(ev.__file__).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (deliberately NOT in tests/helpers.py — only this file needs
# them, and other agents are editing files alongside this one)
# ─────────────────────────────────────────────────────────────────────────────

def _randomize(model, scale=0.05, seed=0):
    """
    Give every parameter a small non-zero value.

    A freshly built MoEDenoiser starts near a dim constant (ExpertHead.proj_out
    is initialised around _INIT_OUT_LEVEL with a small-variance weight), so
    "the reloaded model reproduces the original" would be close to vacuous on
    an untouched model. Randomising first makes those comparisons meaningful.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.empty(p.shape).uniform_(-scale, scale, generator=g))
    return model


def _bias_experts(model, value):
    """Push every expert's output up by ~`value` (proj_out is a 1x1 conv)."""
    with torch.no_grad():
        for head in model.experts:
            head.proj_out.bias.fill_(float(value))
    return model


def _tiny_moe(tiny_kwargs, num_experts=2, seed=0, randomize=True):
    m = build_denoiser("moe", num_experts=num_experts, **tiny_kwargs)
    if randomize:
        _randomize(m, seed=seed)
    return m


class LocalStub(nn.Module):
    """
    Stand-in model whose output is a *purely local* function of the packed
    input: out[..., 2i+dy, 2j+dx] depends only on x[..., i, j].

    Locality is what makes exact comparisons possible: reflect padding and
    tiling cannot change any pixel inside the valid region, so
    infer_patches == infer_full == direct call, bit for bit. `const` instead
    returns a constant field, which turns infer_patches' output into a direct
    read-out of its internal coverage map (see the coverage test).

    Ignores snr_map on purpose: infer_patches normalises the SNR map over the
    *padded* image, so a snr-dependent stub would legitimately differ between
    the two inference paths and the comparison would prove nothing.
    """

    def __init__(self, num_experts=2, const=None):
        super().__init__()
        self.num_experts = num_experts
        self.const = const
        self.calls = []          # spatial shapes of every patch received

    def forward(self, x, snr_map):
        B, _, H, W = x.shape
        self.calls.append((H, W))
        if self.const is not None:
            base = torch.full((B, 3, H, W), float(self.const))
        else:
            # B, G1, R -> a deterministic pointwise RGB-ish triple
            base = torch.stack([x[:, 0], x[:, 1], x[:, 3]], dim=1)
        up = F.interpolate(base, scale_factor=2.0, mode="nearest")
        experts = torch.stack(
            [up * (1.0 + 0.1 * k) for k in range(self.num_experts)], dim=1)
        gates = torch.full((B, self.num_experts, 2 * H, 2 * W),
                           1.0 / self.num_experts)
        return up, experts, gates


@contextlib.contextmanager
def _no_autocast(monkeypatch):
    """
    Replace torch.amp.autocast with a recording no-op.

    Two jobs: (1) prove the pad/crop/tile logic does not depend on autocast,
    so these tests would still cover it if a future torch made
    device_type='cuda' on a CPU-only box fatal instead of merely noisy;
    (2) capture the device_type the module actually asks for.
    """
    seen = []

    def fake_autocast(*args, **kwargs):
        seen.append(kwargs.get("device_type", args[0] if args else None))
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)
    yield seen


class _FakeAccelerator:
    """
    Reports a GPU backend while the tensors stay on the CPU.

    Lets a login node verify that the *right* autocast backend is requested
    for a device it does not have. Only the two members the inference path
    touches are implemented.
    """

    def __init__(self, kind):
        self.kind = kind
        self.device = torch.device("cpu")

    def autocast(self, dtype=torch.bfloat16):
        # Deliberately the real call, so the recording patch in
        # _no_autocast sees the device_type that production code would send.
        return torch.amp.autocast(device_type=self.kind, dtype=dtype)


def _expected_tiling(n_pixels, patch_size, overlap):
    """
    Reference tiling arithmetic, derived from what overlapping-patch inference
    *means* (cover n_pixels with patch_size windows advancing by stride), not
    copied from the implementation:
        n_patches = ceil((n_pixels - patch_size) / stride) + 1, at least 1
        padded    = (n_patches - 1) * stride + patch_size
    """
    stride = patch_size - overlap
    n = max(1, math.ceil((n_pixels - patch_size) / stride) + 1)
    return n, (n - 1) * stride + patch_size, stride


def _output_dir_expr():
    """
    Compile the module's own `OUTPUT_DIR = f"test_results/{...}"` expression
    out of its AST. The assignment lives behind the __main__ guard, so it
    cannot be imported — but it can still be evaluated exactly as written.
    """
    tree = ast.parse(MODULE_PATH.read_text())
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "OUTPUT_DIR"
                     for t in n.targets)]
    assert len(nodes) == 1, f"expected one OUTPUT_DIR assignment, found {len(nodes)}"
    return compile(ast.Expression(nodes[0].value), str(MODULE_PATH), "eval")


def _eval_output_dir(checkpoint):
    # `os` is in scope: the expression derives the run directory with
    # os.path.basename/dirname rather than splitting the string, so that an
    # absolute HDR_CHECKPOINT does not collapse to a shared 'test_results/'.
    import os
    return eval(_output_dir_expr(), {"CHECKPOINT": checkpoint, "os": os})


# ═════════════════════════════════════════════════════════════════════════════
# load_model_from_checkpoint
# ═════════════════════════════════════════════════════════════════════════════

def test_load_returns_model_and_mode_tuple(tmp_path, tiny_kwargs, device):
    """
    The function's contract is `(model, mode)` — the eval loop unpacks exactly
    two values and uses `mode` as the run's mode_tag. A silent change to a
    single return value would break the caller at line 1 of main().
    """
    model = _tiny_moe(tiny_kwargs)
    ck = make_checkpoint(tmp_path / "ck.pth", model, mode="moe",
                         num_experts=2, model_kwargs=tiny_kwargs)

    out = ev.load_model_from_checkpoint(ck, tiny_kwargs, device, 2)

    assert isinstance(out, tuple) and len(out) == 2
    loaded, mode = out
    assert isinstance(loaded, nn.Module)
    assert mode == "moe", "the mode stored in the checkpoint must be returned"


def test_load_rebuilds_moe_and_reproduces_original_outputs(tmp_path, tiny_kwargs, device):
    """
    The whole point of storing 'mode' + 'model_kwargs': the benchmark must
    rebuild the *same* architecture and restore the *same* weights, so the
    reloaded model is numerically identical to the trained one. Any mismatch
    (wrong dim, wrong block count, partially loaded weights) invalidates every
    number the benchmark prints.
    """
    original = _tiny_moe(tiny_kwargs, num_experts=2, seed=1).eval()
    ck = make_checkpoint(tmp_path / "ck.pth", original, mode="moe",
                         num_experts=2, model_kwargs=tiny_kwargs)

    loaded, mode = ev.load_model_from_checkpoint(ck, tiny_kwargs, device, 2)

    x = packed_bayer(1, 16, 16, seed=5)
    snr = snr_map_for(x)
    with torch.no_grad():
        ref = original(x, snr)
        got = loaded(x, snr)

    assert type(loaded) is type(original)
    assert loaded.num_experts == original.num_experts
    for name, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a, b), f"return element {name} differs after reload"
    # Guard against the vacuous case: a zero-init model emits constant eps.
    assert float(ref[0].std()) > 0, "reference output is constant — test is vacuous"


def test_load_puts_model_in_eval_mode_which_enables_the_max_clamp(
        tmp_path, tiny_kwargs, device):
    """
    The function documents `model.eval()` as "enables the [.., 1] clamp":
    MoEDenoiser only clamps its output to max 1.0 when `not self.training`.
    A model returned in train mode would emit >1 radiance into PSNR/SSIM and
    into the saved JPEGs. Pinned with experts biased well above 1 so the clamp
    is actually load-bearing, not a no-op.
    """
    model = _bias_experts(_tiny_moe(tiny_kwargs, seed=2), value=3.0)
    ck = make_checkpoint(tmp_path / "ck.pth", model, mode="moe",
                         num_experts=2, model_kwargs=tiny_kwargs)

    loaded, _ = ev.load_model_from_checkpoint(ck, tiny_kwargs, device, 2)
    assert loaded.training is False, "returned model must be in eval mode"

    x = packed_bayer(1, 16, 16, seed=6)
    snr = snr_map_for(x)
    with torch.no_grad():
        blended_eval, experts_eval, _ = loaded(x, snr)
        loaded.train()
        blended_train, experts_train, _ = loaded(x, snr)

    assert float(blended_train.max()) > 1.0, \
        "fixture failed to exceed 1.0 — the clamp assertion would be vacuous"
    assert float(blended_eval.max()) <= 1.0 + 1e-6
    assert torch.equal(blended_eval, blended_train.clamp(max=1.0))
    assert torch.equal(experts_eval, experts_train.clamp(max=1.0))
    assert float(experts_eval.min()) >= _CLAMP_EPS - 1e-12


def test_load_prefers_stored_kwargs_over_fallback(tmp_path, tiny_kwargs, device):
    """
    Stored metadata must win over the script's MODEL_KWARGS constant, which is
    only a legacy fallback. If the fallback were used instead, a checkpoint
    trained at dim=8 would be rebuilt at the fallback width and the strict
    load would explode (or, worse, silently mismatch).
    """
    model = _tiny_moe(tiny_kwargs, seed=3)
    ck = make_checkpoint(tmp_path / "ck.pth", model, mode="moe",
                         num_experts=2, model_kwargs=tiny_kwargs)

    wrong_fallback = dict(tiny_kwargs, dim=16, heads=[2, 2, 2, 2])
    loaded, _ = ev.load_model_from_checkpoint(ck, wrong_fallback, device, 2)

    assert loaded.dim == tiny_kwargs["dim"] == 8
    assert loaded.patch_embed.out_channels == 8


@pytest.mark.parametrize("mode,expected_cls,expected_k", [
    ("moe", "MoEDenoiser", 2),
    ("dual", "DualSNRDenoiser", 2),
    ("single", "SingleDenoiser", 1),
])
def test_load_honours_stored_mode(tmp_path, tiny_kwargs, device,
                                  mode, expected_cls, expected_k):
    """
    'mode' selects which of three structurally different models is built.
    Reading it from the checkpoint is what lets one benchmark script evaluate
    moe / dual / single runs without a manual edit; getting it wrong is an
    immediate strict-load failure.
    """
    model = build_denoiser(mode, num_experts=expected_k, **tiny_kwargs)
    ck = make_checkpoint(tmp_path / f"{mode}.pth", model, mode=mode,
                         num_experts=expected_k, model_kwargs=tiny_kwargs)

    loaded, returned_mode = ev.load_model_from_checkpoint(ck, tiny_kwargs, device, 2)

    assert returned_mode == mode
    assert type(loaded).__name__ == expected_cls
    assert loaded.num_experts == expected_k
    assert loaded.training is False


def test_legacy_checkpoint_falls_back_to_kwargs_and_default_dual_mode(
        tmp_path, tiny_kwargs, device):
    """
    Docstring: legacy checkpoints "that predate the metadata" must fall back to
    the passed-in kwargs and the default mode ('dual'). Old runs saved before
    mode/model_kwargs existed were DualSNR teachers, so the default has to be
    'dual' and the architecture has to come from fallback_kwargs.
    """
    legacy_model = build_denoiser("dual", **tiny_kwargs)
    _randomize(legacy_model, seed=4)
    path = tmp_path / "legacy.pth"
    torch.save({"epoch": 7, "loss": 0.5,
                "model_state_dict": legacy_model.state_dict()}, path)

    loaded, mode = ev.load_model_from_checkpoint(str(path), tiny_kwargs, device, 2)

    assert mode == "dual", "missing 'mode' must default to 'dual'"
    assert type(loaded).__name__ == "DualSNRDenoiser"
    assert loaded.denoiser_low_snr.patch_embed.out_channels == tiny_kwargs["dim"]
    # weights really came across
    for k, v in legacy_model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), f"{k} not restored"


def test_bare_state_dict_checkpoint_is_accepted(tmp_path, tiny_kwargs, device):
    """
    `state = ckpt.get("model_state_dict", ckpt)` also supports the oldest
    format: the file *is* the state dict. Pinning it stops that fallback from
    being refactored away, which would break every pre-metadata checkpoint.
    """
    legacy_model = _randomize(build_denoiser("dual", **tiny_kwargs), seed=5)
    path = tmp_path / "raw_sd.pth"
    torch.save(legacy_model.state_dict(), path)

    loaded, mode = ev.load_model_from_checkpoint(str(path), tiny_kwargs, device, 2)

    assert mode == "dual"
    x = packed_bayer(1, 16, 16, seed=7)
    snr = snr_map_for(x)
    legacy_model.eval()
    with torch.no_grad():
        assert torch.equal(legacy_model(x, snr)[0], loaded(x, snr)[0])


def test_load_strips_torch_compile_orig_mod_prefix(tmp_path, tiny_kwargs, device):
    """
    A checkpoint saved from a torch.compile'd model has every key prefixed
    with `_orig_mod.`. The loader strips it; without the strip, strict=True
    would reject the whole state dict, so a compiled training run would be
    unevaluatable.
    """
    model = _tiny_moe(tiny_kwargs, seed=6).eval()
    compiled_state = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    path = tmp_path / "compiled.pth"
    torch.save({"mode": "moe", "num_experts": 2, "model_kwargs": tiny_kwargs,
                "model_state_dict": compiled_state}, path)

    loaded, mode = ev.load_model_from_checkpoint(str(path), tiny_kwargs, device, 2)

    assert mode == "moe"
    for k, v in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v), f"{k} not restored"


def test_load_rebuilds_stored_num_experts_three(tmp_path, tiny_kwargs, device):
    """
    CRITICAL contract: the docstring promises the architecture is rebuilt from
    the checkpoint's 'num_experts'. A K=3 MoE run (the model default, and the
    script's own FALLBACK_NUM_EXPERTS) must therefore load and expose 3
    experts. Today line 87 pins num_experts = 2, so the rebuilt trunk has two
    expert heads and load_state_dict(strict=True) dies on the unexpected
    `experts.2.*` keys — every non-2-expert checkpoint is unloadable.
    """
    original = _tiny_moe(tiny_kwargs, num_experts=3, seed=8).eval()
    ck = make_checkpoint(tmp_path / "k3.pth", original, mode="moe",
                         num_experts=3, model_kwargs=tiny_kwargs)

    loaded, mode = ev.load_model_from_checkpoint(ck, tiny_kwargs, device, 3)

    assert mode == "moe"
    assert loaded.num_experts == 3
    assert len(loaded.experts) == 3
    x = packed_bayer(1, 16, 16, seed=9)
    snr = snr_map_for(x)
    with torch.no_grad():
        assert torch.equal(original(x, snr)[0], loaded(x, snr)[0])


def test_fallback_num_experts_argument_is_used(tmp_path, tiny_kwargs, device):
    """
    Second half of the same defect: for a checkpoint with no stored
    'num_experts', the `fallback_num_experts` parameter is supposed to decide
    the expert count (main() passes FALLBACK_NUM_EXPERTS = 3). The parameter is
    never read, so passing 3 still builds 2 experts and the load fails.
    """
    original = _tiny_moe(tiny_kwargs, num_experts=3, seed=10).eval()
    path = tmp_path / "legacy_moe.pth"
    torch.save({"mode": "moe", "model_kwargs": tiny_kwargs,
                "model_state_dict": original.state_dict()}, path)

    loaded, _ = ev.load_model_from_checkpoint(str(path), tiny_kwargs, device,
                                             fallback_num_experts=3)

    assert loaded.num_experts == 3


# ═════════════════════════════════════════════════════════════════════════════
# infer_full — reflect-pad to a multiple of 8, crop back to 2x the true size
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("h,w", [(30, 34), (25, 33), (17, 9), (32, 24)])
def test_infer_full_crops_back_to_exact_sensor_size(tiny_kwargs, h, w):
    """
    THE contract of infer_full: an arbitrary packed size is reflect-padded up
    to a multiple of 8 (the three PixelUnshuffle(2) stages need it) and the RGB
    result is cropped back to exactly (2h, 2w). If the crop were forgotten the
    benchmark would compare a padded prediction against an unpadded GT and
    every metric would be garbage (or crash on the shape mismatch).
    """
    model = _tiny_moe(tiny_kwargs, seed=11).eval()
    x = packed_bayer(1, h, w, seed=12)

    with torch.no_grad():
        blended, experts, gates = ev.infer_full(model, x, device="cpu")

    assert_shape(blended, (1, 3, 2 * h, 2 * w), "blended")
    assert_shape(experts, (1, 2, 3, 2 * h, 2 * w), "expert_outs")
    assert_shape(gates, (1, 2, 2 * h, 2 * w), "gates")
    assert_finite(blended, "blended")
    assert_in_range(blended, _CLAMP_EPS, 1.0, "blended")
    # eval-mode model + softmax gate: routing weights are a partition of unity
    assert torch.allclose(gates.sum(dim=1), torch.ones_like(gates[:, 0]), atol=1e-5)


def test_infer_full_no_pad_path_is_identical_to_a_direct_call(tiny_kwargs):
    """
    When h, w are already multiples of 8 the function must hand the tensor to
    the model untouched (`if pad_h or pad_w` guard). Pinning bit-exact equality
    with a manual call proves no stray padding, no shape juggling and that the
    SNR map is built with the documented window_size=5.
    """
    model = _tiny_moe(tiny_kwargs, seed=13).eval()
    x = packed_bayer(1, 16, 24, seed=14)

    with torch.no_grad():
        got = ev.infer_full(model, x, device="cpu")
        ref = model(x, estimate_local_snr_map(x, window_size=5))

    for i, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a, b), f"return element {i} differs from a direct call"


def test_infer_full_padding_does_not_disturb_the_valid_region(tiny_kwargs):
    """
    Padding is an implementation detail: for a model whose output is a purely
    local function of its input, the cropped result must equal the model run on
    the unpadded image. This is what rules out an off-by-one crop (e.g.
    cropping from the wrong edge, or cropping at Bayer instead of sensor
    resolution).
    """
    stub = LocalStub(num_experts=2)
    x = packed_bayer(1, 30, 34, seed=15)

    with torch.no_grad():
        blended, experts, gates = ev.infer_full(stub, x, device="cpu")
        ref_blended, ref_experts, ref_gates = stub(x, snr_map_for(x))

    assert torch.equal(blended, ref_blended)
    assert torch.equal(experts, ref_experts)
    assert torch.equal(gates, ref_gates)


def test_infer_full_skips_autocast_on_the_cpu(tiny_kwargs, monkeypatch):
    """
    autocast used to be hardcoded to device_type='cuda', which raises on an
    Intel GPU and merely warns on a CPU-only host. It is now taken from the
    device, so a CPU run requests no autocast at all.

    The second half is what makes the rest of this file meaningful: with
    autocast neutralised the pad/crop/crop-shape behaviour is bit-identical,
    so the CPU coverage here is exercising the same logic that runs on a GPU.
    """
    model = _tiny_moe(tiny_kwargs, seed=16).eval()
    x = packed_bayer(1, 30, 34, seed=17)

    with torch.no_grad():
        real = ev.infer_full(model, x, device="cpu")[0]

    with _no_autocast(monkeypatch) as seen:
        with torch.no_grad():
            patched = ev.infer_full(model, x, device="cpu")[0]

    assert seen == [], f"cpu inference should request no autocast, got {seen}"
    assert_shape(patched, (1, 3, 60, 68), "blended")
    assert torch.equal(real, patched)


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_infer_full_asks_for_the_devices_own_autocast_backend(
        tiny_kwargs, monkeypatch, kind):
    """
    The device_type passed to autocast must be the backend the tensors are
    actually on — 'cuda' on Polaris, 'xpu' on Aurora. Getting this wrong is
    not a slow path but a hard error on Aurora, and it cannot be caught on a
    login node without faking the accelerator, which is what this does: the
    model still runs on the CPU, but the Accelerator reports a GPU kind.
    """
    model = _tiny_moe(tiny_kwargs, seed=16).eval()
    x = packed_bayer(1, 30, 34, seed=17)

    monkeypatch.setattr(ev, "get_accelerator",
                        lambda *_a, **_k: _FakeAccelerator(kind))

    with _no_autocast(monkeypatch) as seen:
        with torch.no_grad():
            out = ev.infer_full(model, x, device="cpu")[0]

    assert seen == [kind], f"expected one {kind} autocast request, got {seen}"
    assert_shape(out, (1, 3, 60, 68), "blended")


@pytest.mark.gpu
def test_infer_full_under_real_gpu_autocast(tiny_kwargs, gpu_device):
    """
    On a real GPU — A100 on Polaris, Max 1550 on Aurora — the bf16 autocast
    region is live; the pad/crop contract and the eval-mode clamp must
    survive reduced precision. Auto-skips on a login node.
    """
    dev = gpu_device
    model = _tiny_moe(tiny_kwargs, seed=18).eval().to(dev)
    x = packed_bayer(1, 30, 34, seed=19).to(dev)

    with torch.no_grad():
        blended, experts, gates = ev.infer_full(model, x, device=dev)

    assert_shape(blended, (1, 3, 60, 68), "blended")
    assert_shape(experts, (1, 2, 3, 60, 68), "expert_outs")
    assert_shape(gates, (1, 2, 60, 68), "gates")
    assert_finite(blended.float(), "blended")
    assert float(blended.float().max()) <= 1.0 + 1e-2


# ═════════════════════════════════════════════════════════════════════════════
# infer_patches — tiling arithmetic
# ═════════════════════════════════════════════════════════════════════════════

TILINGS = [
    (64, 64, 32, 8),
    (64, 64, 32, 16),
    (100, 60, 32, 8),
    (40, 40, 32, 8),
    (24, 64, 32, 8),      # H below patch_size but pad_h < H, so still legal
    (30, 34, 16, 4),
]


@pytest.mark.parametrize("H,W,patch,overlap", TILINGS)
def test_infer_patches_pads_to_an_exact_number_of_strides(
        H, W, patch, overlap, monkeypatch):
    """
    The padded extent must satisfy Hp = n*stride + overlap exactly, i.e. the
    last window ends flush with the padded edge. Anything else leaves either a
    sliver of the image outside every window (uncovered -> divide by ~0) or a
    window running off the end. Verified by intercepting the module's own
    F.pad call, and cross-checked against an independently derived tiling.
    """
    records = []
    real_pad = F.pad

    def recording_pad(t, pad, *a, **kw):
        out = real_pad(t, pad, *a, **kw)
        records.append((tuple(t.shape), tuple(pad), tuple(out.shape)))
        return out

    monkeypatch.setattr(ev.F, "pad", recording_pad)

    stub = LocalStub(num_experts=2, const=1.0)
    with torch.no_grad():
        ev.infer_patches(stub, packed_bayer(1, H, W, seed=20), 2,
                         patch_size=patch, overlap=overlap, device="cpu")

    assert records, "infer_patches never called F.pad"
    _, pad_spec, padded_shape = records[0]
    pad_w, pad_h = pad_spec[1], pad_spec[3]
    Hp, Wp = padded_shape[2], padded_shape[3]

    n_h, exp_Hp, stride = _expected_tiling(H, patch, overlap)
    n_w, exp_Wp, _ = _expected_tiling(W, patch, overlap)

    assert pad_h >= 0 and pad_w >= 0, \
        f"negative pad {pad_spec} would make F.pad CROP the image"
    assert (Hp - overlap) % stride == 0 and (Wp - overlap) % stride == 0
    assert (Hp, Wp) == (exp_Hp, exp_Wp)
    assert Hp >= H and Wp >= W
    assert Hp - H < stride and Wp - W < stride, "padding is not minimal"
    # one model call per tile
    assert len(stub.calls) == n_h * n_w
    assert set(stub.calls) == {(patch, patch)}, "every tile must be patch_size²"


@pytest.mark.parametrize("H,W,patch,overlap", TILINGS)
def test_infer_patches_covers_every_output_pixel(H, W, patch, overlap):
    """
    weight_sum must be >= 1 at every output pixel before the division. A stub
    that predicts a constant 1.0 turns the returned tensor into a direct
    read-out of the coverage map: coverage/(coverage+1e-8) is ~1 where covered
    and exactly 0 where not. So "output == 1 everywhere" == "no gaps", and any
    uncovered pixel shows up as a black hole in the benchmark image.
    """
    stub = LocalStub(num_experts=2, const=1.0)
    with torch.no_grad():
        blended, experts, gates = ev.infer_patches(
            stub, packed_bayer(1, H, W, seed=21), 2,
            patch_size=patch, overlap=overlap, device="cpu")

    assert_shape(blended, (1, 3, 2 * H, 2 * W), "blended")
    assert_shape(experts, (1, 2, 3, 2 * H, 2 * W), "expert_outs")
    assert_shape(gates, (1, 2, 2 * H, 2 * W), "gates")
    assert float(blended.min()) > 0.999, \
        f"uncovered output pixels: min={float(blended.min())}"
    assert torch.allclose(blended, torch.ones_like(blended), atol=1e-6)
    # per-expert channel keeps its own scale (1.0 and 1.1) through the average
    assert torch.allclose(experts[:, 0], torch.ones_like(experts[:, 0]), atol=1e-6)
    assert torch.allclose(experts[:, 1], 1.1 * torch.ones_like(experts[:, 1]),
                          atol=1e-6)
    # gates were 0.5 each in every tile -> still 0.5 after normalisation
    assert torch.allclose(gates, 0.5 * torch.ones_like(gates), atol=1e-6)
    assert torch.allclose(gates.sum(dim=1), torch.ones_like(gates[:, 0]), atol=1e-6)


def test_infer_patches_equals_infer_full_for_a_local_model():
    """
    Tiling must be a pure implementation detail. For a model with no spatial
    receptive field, overlap-averaging is mathematically an identity, so
    infer_patches must reproduce infer_full bit for bit. This is the strongest
    available check that the sensor-resolution offsets (ys = 2*yi) and the
    normalisation are right — a factor-of-two slip in the offsets would still
    produce the correct *shape* but scrambled content.
    """
    stub = LocalStub(num_experts=2)
    x = packed_bayer(1, 64, 48, seed=22)

    with torch.no_grad():
        full = ev.infer_full(stub, x, device="cpu")
        patched = ev.infer_patches(stub, x, 2, patch_size=32, overlap=8,
                                   device="cpu")

    for i, (a, b) in enumerate(zip(full, patched)):
        assert a.shape == b.shape
        assert torch.allclose(a, b, atol=1e-6), \
            f"element {i}: max |diff| = {float((a - b).abs().max())}"


def test_infer_patches_with_real_model_is_finite_and_close_to_full(tiny_kwargs):
    """
    End-to-end sanity with the real MoE trunk: the tiled path must produce the
    documented shapes, stay finite, keep the eval-mode [eps, 1] range, and land
    in the same ballpark as full-image inference (they differ only through the
    convolutional receptive field at tile seams).
    """
    model = _tiny_moe(tiny_kwargs, seed=23).eval()
    x = packed_bayer(1, 32, 32, seed=24)

    with torch.no_grad():
        p_blend, p_experts, p_gates = ev.infer_patches(
            model, x, model.num_experts, patch_size=16, overlap=8, device="cpu")
        f_blend = ev.infer_full(model, x, device="cpu")[0]

    assert_shape(p_blend, (1, 3, 64, 64), "blended")
    assert_shape(p_experts, (1, 2, 3, 64, 64), "expert_outs")
    assert_shape(p_gates, (1, 2, 64, 64), "gates")
    assert_finite(p_blend, "blended")
    assert_in_range(p_blend, 0.0, 1.0, "blended")
    assert torch.allclose(p_gates.sum(dim=1), torch.ones_like(p_gates[:, 0]),
                          atol=1e-5), "normalised gates must still sum to 1"
    assert float((p_blend - f_blend).abs().mean()) < 0.15, \
        "tiled inference diverges wildly from full-image inference"


def test_infer_patches_skips_autocast_on_the_cpu(monkeypatch):
    """
    Same as infer_full, per tile: no autocast is requested on the CPU, and
    the tiling maths is unaffected when autocast is neutralised — so the CPU
    coverage above is testing the real code path.
    """
    stub = LocalStub(num_experts=2, const=1.0)
    x = packed_bayer(1, 64, 64, seed=25)

    with torch.no_grad():
        real = ev.infer_patches(stub, x, 2, patch_size=32, overlap=8,
                                device="cpu")[0]

    stub.calls.clear()
    with _no_autocast(monkeypatch) as seen:
        with torch.no_grad():
            patched = ev.infer_patches(stub, x, 2, patch_size=32, overlap=8,
                                       device="cpu")[0]

    assert seen == [] and len(stub.calls) == 9
    assert torch.equal(real, patched)


@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_infer_patches_asks_for_the_devices_own_autocast_backend(
        monkeypatch, kind):
    """
    One autocast request per tile, each naming the backend the tensors live
    on. The count matters as much as the name: an accelerator looked up once
    per tile instead of once per call would still pass the name check while
    re-probing the device 9 times.
    """
    stub = LocalStub(num_experts=2, const=1.0)
    x = packed_bayer(1, 64, 64, seed=25)

    monkeypatch.setattr(ev, "get_accelerator",
                        lambda *_a, **_k: _FakeAccelerator(kind))

    with _no_autocast(monkeypatch) as seen:
        with torch.no_grad():
            ev.infer_patches(stub, x, 2, patch_size=32, overlap=8,
                             device="cpu")

    assert seen == [kind] * 9, f"expected 9 {kind} requests, got {seen}"


def test_infer_patches_handles_image_smaller_than_the_patch():
    """
    A 16x16 packed image with patch_size=32 needs pad_h = 16 == H, and reflect
    padding requires pad < dim, so F.pad raises. The condition is
    H <= patch_size/2 in general (with the script's PATCH_SIZE=1024 that is any
    test image under ~512 packed pixels), and infer_full handles such images
    fine — so the patch path crashes on input the full path accepts. Correct
    behaviour: pad (replicate, or tile-clamped) and return (2H, 2W).
    """
    stub = LocalStub(num_experts=2, const=1.0)
    x = packed_bayer(1, 16, 16, seed=26)

    with torch.no_grad():
        blended, _, _ = ev.infer_patches(stub, x, 2, patch_size=32, overlap=8,
                                         device="cpu")

    assert_shape(blended, (1, 3, 32, 32), "blended")
    assert torch.allclose(blended, torch.ones_like(blended), atol=1e-6)


def test_infer_patches_never_returns_a_silently_empty_prediction():
    """
    For H <= overlap, ceil((H-overlap)/stride) == 0, so Hp < patch_size and
    `range(0, Hp - patch_size + 1, stride)` is empty: no tile is ever run, the
    model is never called, weight_sum stays 0 and every pixel comes out as
    0/1e-8 = 0. A correctly shaped, entirely black prediction is silent data
    corruption — worse than an exception, because the benchmark would happily
    report a PSNR for it.
    """
    stub = LocalStub(num_experts=2, const=1.0)
    x = packed_bayer(1, 8, 64, seed=27)

    with torch.no_grad():
        blended, _, _ = ev.infer_patches(stub, x, 2, patch_size=32, overlap=8,
                                         device="cpu")

    assert_shape(blended, (1, 3, 16, 128), "blended")
    assert stub.calls, "the model was never run — no tile covered the image"
    assert float(blended.max()) > 0.0, "prediction is entirely zero"


def test_infer_patches_pad_is_provably_non_negative():
    """
    Companion to the two xfails above: F.pad silently *crops* on a negative
    pad, which would delete image rows. Sweeping the arithmetic shows pad is
    never negative (ceil(a/s)*s >= a for s > 0), so the failure mode of small
    images is the reflect-pad exception / empty tile loop, not silent
    truncation. Documents the boundary precisely.
    """
    for patch in (16, 32, 256):
        for overlap in (0, 4, patch // 8, patch // 4):
            stride = patch - overlap
            for H in range(1, 3 * patch):
                pad = (math.ceil((H - overlap) / stride) * stride + overlap) - H
                assert pad >= 0, (H, patch, overlap, pad)
                Hp = H + pad
                n_tiles = len(range(0, Hp - patch + 1, stride))
                # zero tiles happen exactly when the image fits inside `overlap`
                assert (n_tiles == 0) == (H <= overlap), (H, patch, overlap)
                # reflect padding is only legal while pad < H
                assert (pad < H) or (H <= patch // 2), (H, patch, overlap, pad)


# ═════════════════════════════════════════════════════════════════════════════
# estimate_flops
# ═════════════════════════════════════════════════════════════════════════════

def test_estimate_flops_returns_a_positive_float(tiny_kwargs, capsys):
    """
    With torchinfo installed the function must return GFLOPs as a positive
    float (the caller prints it into the results log). A bigger model must
    also cost more, which pins that the number really tracks the traced graph
    rather than being a constant.
    """
    small = _tiny_moe(tiny_kwargs, seed=28)
    g_small = ev.estimate_flops(small, patch_size=16, device="cpu")

    assert isinstance(g_small, float)
    assert g_small > 0.0
    assert math.isfinite(g_small)

    g_big = ev.estimate_flops(small, patch_size=32, device="cpu")
    assert g_big > g_small, "4x the pixels must cost more FLOPs"

    out = capsys.readouterr().out
    assert "Parameters" in out and "GFLOPs" in out


def test_estimate_flops_returns_none_when_torchinfo_is_missing(
        tiny_kwargs, monkeypatch, capsys):
    """
    Documented as "gracefully skipped if torchinfo is not installed" — the
    benchmark must not die on an optional dependency. Simulated by poisoning
    sys.modules so `from torchinfo import summary` raises ImportError, which is
    what the except clause claims to handle.
    """
    model = _tiny_moe(tiny_kwargs, seed=29)
    monkeypatch.setitem(sys.modules, "torchinfo", None)

    result = ev.estimate_flops(model, patch_size=16, device="cpu")

    assert result is None
    assert "torchinfo not installed" in capsys.readouterr().out


def test_estimate_flops_returns_none_on_any_internal_failure(capsys):
    """
    The broad `except Exception` is the second half of the graceful-degradation
    contract: a model torchinfo cannot trace must not abort the benchmark
    before a single image is evaluated.
    """
    class Exploding(nn.Module):
        num_experts = 1

        def forward(self, x, snr_map):
            raise ValueError("cannot trace me")

    assert ev.estimate_flops(Exploding(), patch_size=8, device="cpu") is None
    assert "FLOPs estimation failed" in capsys.readouterr().out


@pytest.mark.parametrize("train_mode", [True, False])
def test_estimate_flops_does_not_mutate_the_model(tiny_kwargs, train_mode):
    """
    estimate_flops runs before the real evaluation (and before torch.compile),
    so it must leave the model untouched: identical parameters/buffers and the
    same training flag. torchinfo forces mode='eval' internally — if that
    leaked, a subsequent training-mode caller would silently lose the
    train/eval distinction the output clamp depends on.
    """
    model = _tiny_moe(tiny_kwargs, seed=30)
    model.train(train_mode)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    ev.estimate_flops(model, patch_size=16, device="cpu")

    after = model.state_dict()
    assert set(after) == set(before)
    for k in before:
        assert torch.equal(before[k], after[k]), f"{k} was mutated"
    assert model.training is train_mode


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT_DIR derivation (inside the __main__ guard — evaluated via the AST)
# ═════════════════════════════════════════════════════════════════════════════

def test_output_dir_uses_the_run_directory_of_a_relative_checkpoint():
    """
    OUTPUT_DIR is meant to name the results folder after the training run that
    produced the checkpoint, so results from different runs never overwrite
    each other's log.txt / results.csv. Works for the documented relative
    "models_p{PHASE}_{mode}_.../phase{PHASE}_best.pth" form.
    """
    got = _eval_output_dir("models_p1_moe_Teacher_MobileHDR_20260619_0059/phase1_best.pth")
    assert got == "test_results/models_p1_moe_Teacher_MobileHDR_20260619_0059"


def test_output_dir_survives_an_absolute_checkpoint_path():
    """
    CHECKPOINT is overridable via $HDR_CHECKPOINT, where an absolute path is
    the natural thing to pass. split('/')[0] is then the empty string, so
    OUTPUT_DIR degenerates to 'test_results/' and *every* absolute-path run
    writes its log.txt, results.csv and rgb/*.jpg over the previous one.
    Correct behaviour: derive the run directory from the checkpoint's parent
    (e.g. os.path.basename(os.path.dirname(CHECKPOINT))).
    """
    got = _eval_output_dir("/scratch/runs/models_p1_moe_20260619_0059/phase1_best.pth")

    tail = got[len("test_results/"):]
    assert tail, "OUTPUT_DIR has no run-specific component"
    assert tail == "models_p1_moe_20260619_0059"


def test_output_dir_is_run_specific_for_every_distinct_checkpoint():
    """
    Two checkpoints from two different runs must not map to the same output
    directory, or the second benchmark silently overwrites the first one's
    results. Relative paths satisfy this; kept as a regression guard next to
    the absolute-path xfail above.
    """
    a = _eval_output_dir("models_p1_moe_A/phase1_best.pth")
    b = _eval_output_dir("models_p2_moe_B/phase2_best.pth")
    assert a != b
    assert a.startswith("test_results/") and b.startswith("test_results/")
