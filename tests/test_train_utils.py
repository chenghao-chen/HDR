"""
Module-level utilities of train_A100_MoE_two_phase.py.

Everything tested here runs on the CPU in microseconds, but the invariants are
the ones that silently corrupt a multi-day training run when they break:

  * hdr_tonemap defines the metric space every logged number lives in — if it
    ever diverges from the identically-named function in
    test_dual_MoE_two_phase.py, the W&B training curves and the benchmark
    numbers stop being comparable and nobody notices.
  * batch_psnr_gpu is the reported PSNR; averaging per-image PSNR and taking
    the PSNR of the pooled MSE differ by several dB on a mixed batch.
  * the collate functions define the data contract between the Dataset and the
    training loop (packed-Bayer layout, orig_h/orig_w -> valid_mask).
  * build_scheduler is stepped once per epoch for the whole run; an off-by-one
    at the warmup boundary is invisible except as a slightly wrong LR curve.
"""

import math
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import train_A100_MoE_two_phase as T
import test_dual_MoE_two_phase as E

from helpers import (
    assert_finite,
    assert_shape,
    packed_bayer,
    packed_bayer_3d,
)

REPO_ROOT = os.path.dirname(os.path.abspath(T.__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (kept in this file on purpose — see harness rules)
# ─────────────────────────────────────────────────────────────────────────────

def _channel_tagged_packed(h, w, seed=0):
    """
    Packed BGGR [4, h, w] whose channel c holds values in [c, c+1).

    floor(value) therefore recovers which Bayer colour plane a sample came
    from, which lets a test check the Bayer phase of padded / shuffled output
    without caring about the actual pixel values.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.rand((4, h, w), generator=g) * 0.9
    return base + torch.arange(4, dtype=torch.float32).view(4, 1, 1)


def _sample(h, w, seed=0):
    """One Dataset-shaped sample dict: x / xm (alias) / y, all [4, h, w]."""
    x = packed_bayer_3d(h, w, seed=seed)
    y = packed_bayer_3d(h, w, seed=seed + 1000)
    return {"x": x, "xm": x, "y": y}


def _lr_trajectory(scheduler, optimizer, n_steps):
    """LR seen at each of the n_steps+1 epoch boundaries, stepping after read."""
    lrs = []
    for _ in range(n_steps + 1):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    return lrs


def _fresh_optimizer(lr):
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.Adam([p], lr=lr)


# ─────────────────────────────────────────────────────────────────────────────
# Import-time side effects
# ─────────────────────────────────────────────────────────────────────────────

def test_import_does_not_start_training(tmp_path):
    """
    Importing the training script must be inert: the whole run lives behind
    `if __name__ == "__main__"`. This is what makes the module importable from
    a test (or from the eval script) at all, and it also protects against a
    stray top-level statement launching a 50-epoch job inside pytest.

    Checked in a subprocess so the assertions see a virgin interpreter:
      * exit status 0, no output on stdout,
      * WANDB_* defaults applied with setdefault (a pre-set value survives —
        job scripts rely on that to redirect W&B to cluster scratch),
      * no wandb run created,
      * nothing written into the process CWD (the run folder is created by
        main(), never at import).
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, json, sys\n"
        "before = sorted(os.listdir('.'))\n"
        "import train_A100_MoE_two_phase as T\n"
        "import wandb\n"
        "info = {\n"
        "    'wandb_dir': os.environ.get('WANDB_DIR'),\n"
        "    'wandb_cache': os.environ.get('WANDB_CACHE_DIR'),\n"
        "    'wandb_data': os.environ.get('WANDB_DATA_DIR'),\n"
        "    'run_is_none': wandb.run is None,\n"
        "    'cwd_before': before,\n"
        "    'cwd_after': sorted(os.listdir('.')),\n"
        "    'has_main_guard': hasattr(T, 'hdr_tonemap'),\n"
        "}\n"
        "sys.stderr.write(json.dumps(info))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT
    env["WANDB_DIR"] = "/sentinel/wandb/dir"      # pre-set: must survive
    env.pop("WANDB_CACHE_DIR", None)              # unset: must get the default
    env.pop("WANDB_DATA_DIR", None)
    env["WANDB_MODE"] = "offline"

    r = subprocess.run([sys.executable, str(probe)], cwd=str(tmp_path), env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"import failed:\n{r.stderr}"
    assert r.stdout == "", f"import printed to stdout: {r.stdout!r}"

    import json
    info = json.loads(r.stderr[r.stderr.index("{"):])
    assert info["has_main_guard"] is True
    assert info["run_is_none"] is True, "importing created a W&B run"
    # setdefault, not assignment
    assert info["wandb_dir"] == "/sentinel/wandb/dir"
    # The defaults follow the machine (eagle on Polaris, flare on Aurora), not
    # the original Gilbreth scratch paths, which exist on neither system.
    from hdr_platform import resolve_project_root
    wandb_root = resolve_project_root()
    assert info["wandb_cache"] == os.path.join(wandb_root, "wandb_cache")
    assert info["wandb_data"] == os.path.join(wandb_root, "wandb_data")
    # no run folder / log file created at import time
    assert info["cwd_after"] == info["cwd_before"], \
        f"import wrote to CWD: {info['cwd_after']} vs {info['cwd_before']}"


def test_public_utilities_exist():
    """
    The eval script and the submit scripts reach into this module by name;
    a rename would break them at the first epoch, not at import.
    """
    for name in ("hdr_tonemap", "batch_psnr_gpu", "collate_xy",
                 "collate_pad_to_max", "build_scheduler", "make_d4_transform",
                 "write_log", "create_folder", "print_running_loss",
                 "print_epoch_loss"):
        assert callable(getattr(T, name)), f"{name} missing or not callable"


# ─────────────────────────────────────────────────────────────────────────────
# hdr_tonemap
# ─────────────────────────────────────────────────────────────────────────────

def test_tonemap_endpoints_are_exact():
    """
    tonemap must map the [0, 1] linear range onto exactly [0, 1]: 0 -> 0 and
    1 -> 1 bit-exactly, otherwise every PSNR-µ / LPIPS number is computed on a
    slightly rescaled signal and the `*2-1` LPIPS remap leaves [-1, 1].
    """
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([0.0, 1.0], dtype=dtype)
        out = T.hdr_tonemap(x)
        assert float(out[0]) == 0.0
        assert float(out[1]) == 1.0
        assert out.dtype == dtype


def test_tonemap_matches_closed_form():
    """
    Pins the formula log1p(mu*x)/log1p(mu) elementwise against a scalar
    reference for several mu, so a "small" refactor of the tonemap cannot
    silently change the metric space.
    """
    xs = [0.0, 1e-6, 1e-3, 0.017, 0.25, 0.5, 0.9, 1.0]
    for mu in (1.0, 100, 5000, 20000):
        got = T.hdr_tonemap(torch.tensor(xs, dtype=torch.float64), mu=mu)
        want = torch.tensor([math.log1p(mu * v) / math.log1p(mu) for v in xs],
                            dtype=torch.float64)
        assert torch.allclose(got, want, rtol=0, atol=1e-12), f"mu={mu}"


def test_tonemap_default_mu_is_5000():
    """mu defaults to the Kalantari-2017 standard; training and eval both rely
    on the default when they call hdr_tonemap without an explicit mu."""
    x = torch.linspace(0, 1, 17)
    assert torch.equal(T.hdr_tonemap(x), T.hdr_tonemap(x, mu=5000))
    assert not torch.equal(T.hdr_tonemap(x), T.hdr_tonemap(x, mu=500))


def test_tonemap_strictly_monotonic():
    """
    Strict monotonicity is what makes the tonemapped domain a valid space for
    an L1/PSNR metric: no two different linear intensities may collapse onto
    the same tonemapped value, or the loss stops distinguishing them.
    """
    x = torch.linspace(0.0, 1.0, 2001, dtype=torch.float64)
    y = T.hdr_tonemap(x)
    d = y[1:] - y[:-1]
    assert bool((d > 0).all()), f"non-increasing step at {int(d.argmin())}"


def test_tonemap_is_concave_and_lifts_shadows():
    """
    Concavity is the whole point: highlights get compressed, shadows get
    lifted, so the L1-µ loss spends its budget on dark detail instead of on
    the few bright pixels. Verified three ways — midpoint above chord,
    monotonically decreasing slope, and f(x) > x on (0, 1).
    """
    x = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    y = T.hdr_tonemap(x)

    # 1. midpoint of any chord lies below the curve
    a, b = x[:-2], x[2:]
    mid = (a + b) / 2
    assert bool((T.hdr_tonemap(mid) > (T.hdr_tonemap(a) + T.hdr_tonemap(b)) / 2).all())

    # 2. discrete second difference strictly negative
    slope = y[1:] - y[:-1]
    assert bool((slope[1:] - slope[:-1] < 0).all())

    # 3. shadows lifted everywhere strictly inside (0, 1)
    inner = x[1:-1]
    assert bool((T.hdr_tonemap(inner) > inner).all())


def test_tonemap_agrees_bitwise_with_eval_script():
    """
    train_A100_MoE_two_phase.hdr_tonemap and
    test_dual_MoE_two_phase.hdr_tonemap are two copies of the same code. If
    they ever drift, PSNR-µ logged during training and PSNR-µ reported by the
    benchmark are measured in different spaces and the comparison the eval
    script's docstring promises ("mu=5000 matches training") is a lie.
    Bit-exact equality, not allclose.
    """
    for dtype in (torch.float32, torch.float64):
        x = torch.linspace(0.0, 1.0, 1024, dtype=dtype)
        for mu in (5000, 100, 1.0):
            assert torch.equal(T.hdr_tonemap(x, mu=mu), E.hdr_tonemap(x, mu=mu)), \
                f"tonemap drift at dtype={dtype} mu={mu}"
    # and the defaults agree too (the value actually used by both scripts)
    x = torch.rand(64, dtype=torch.float64)
    assert torch.equal(T.hdr_tonemap(x), E.hdr_tonemap(x))


def test_tonemap_is_shape_and_layout_agnostic():
    """
    The training loop applies it to [B,3,H,W] predictions and to the
    [B,K,3,H,W] expert stack in the same call; it must be a pure elementwise
    map with no shape assumptions.
    """
    t5 = torch.rand(2, 3, 3, 8, 8)
    out = T.hdr_tonemap(t5)
    assert_shape(out, t5.shape, "tonemapped expert stack")
    assert_finite(out, "tonemapped expert stack")
    # elementwise: flattening commutes with the map
    assert torch.equal(out.reshape(-1), T.hdr_tonemap(t5.reshape(-1)))


# ─────────────────────────────────────────────────────────────────────────────
# batch_psnr_gpu
# ─────────────────────────────────────────────────────────────────────────────

def test_psnr_identical_images_is_exactly_100():
    """
    The mse == 0 branch is a sentinel, not infinity: a perfect reconstruction
    must report exactly 100.0 so the running average stays finite (an inf
    would poison every subsequent logged mean).
    """
    img = torch.rand(3, 3, 8, 8)
    v = T.batch_psnr_gpu(img, img.clone())
    assert isinstance(v, float)
    assert v == 100.0


def test_psnr_known_mse_matches_analytic():
    """A constant offset gives an exactly known MSE; PSNR must equal
    10*log10(data_range^2 / mse) — pins both the formula and the log base."""
    gt = torch.zeros(1, 3, 16, 16)
    for offset in (0.1, 0.01, 0.5):
        img = torch.full_like(gt, offset)
        want = 10.0 * math.log10(1.0 / offset ** 2)
        assert T.batch_psnr_gpu(img, gt) == pytest.approx(want, abs=1e-4)


def test_psnr_respects_data_range():
    """data_range enters as data_range**2 in the numerator; passing 2.0 must
    shift PSNR by exactly 10*log10(4) dB, not 20*log10(2)/2 or similar."""
    gt = torch.zeros(1, 3, 8, 8)
    img = torch.full_like(gt, 0.1)
    base = T.batch_psnr_gpu(img, gt, data_range=1.0)
    wide = T.batch_psnr_gpu(img, gt, data_range=2.0)
    assert wide - base == pytest.approx(10.0 * math.log10(4.0), abs=1e-4)


def test_psnr_averages_per_image_not_pooled_mse():
    """
    THE metric-semantics test. A batch with per-image MSEs 1e-4 and 1e-2 has
    per-image PSNRs 40 dB and 20 dB -> mean 30 dB, while the PSNR of the
    pooled MSE (5.05e-3) is ~22.97 dB. Reporting the pooled number would make
    every logged PSNR pessimistic and incomparable with the eval script, which
    also averages per image.
    """
    gt = torch.zeros(2, 3, 20, 20)
    img = gt.clone()
    img[0] = 0.01     # mse 1e-4 -> 40 dB
    img[1] = 0.1      # mse 1e-2 -> 20 dB

    got = T.batch_psnr_gpu(img, gt)
    pooled = 10.0 * math.log10(1.0 / float(((img - gt) ** 2).mean()))

    assert got == pytest.approx(30.0, abs=1e-3)
    assert abs(got - pooled) > 5.0, "cannot distinguish per-image from pooled"


def test_psnr_mixed_perfect_and_imperfect_image_stays_finite():
    """
    torch.where evaluates both branches, so a zero-MSE image produces an inf
    in the discarded branch. The selected result must still be finite —
    an inf leaking through would NaN the epoch average.
    """
    gt = torch.zeros(2, 3, 8, 8)
    img = gt.clone()
    img[1] = 0.1                      # image 0 perfect, image 1 at 20 dB
    got = T.batch_psnr_gpu(img, gt)
    assert math.isfinite(got)
    assert got == pytest.approx((100.0 + 20.0) / 2.0, abs=1e-3)


def test_psnr_agrees_with_eval_script_psnr():
    """
    The training metric and the benchmark metric must be the same function, or
    the numbers in the W&B run and in the results CSV cannot be compared.
    """
    gt = torch.rand(4, 3, 12, 12)
    img = (gt + 0.05 * torch.randn_like(gt)).clamp(0, 1)
    assert T.batch_psnr_gpu(img, gt) == pytest.approx(E.psnr(img, gt), abs=1e-5)
    assert T.batch_psnr_gpu(gt, gt) == E.psnr(gt, gt) == 100.0


def test_psnr_builds_no_autograd_graph():
    """
    batch_psnr_gpu is called on the live prediction inside the training step;
    if it built a graph (or forgot the .detach()) it would keep the whole
    forward activation set alive for the metric alone. Detected by counting
    saved-for-backward packs: an equivalent grad-enabled computation saves
    tensors, this one must save none.
    """
    leaf = torch.rand(2, 3, 8, 8, requires_grad=True)
    gt = torch.rand(2, 3, 8, 8)
    img = leaf * 2.0                      # non-leaf, has grad_fn

    packs = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (packs.append(1), t)[1], lambda t: t):
        val = T.batch_psnr_gpu(img, gt)
        during = len(packs)
        control = torch.mean((leaf * 2.0 - gt) ** 2)   # sanity: this DOES save
        after = len(packs)

    assert during == 0, f"batch_psnr_gpu saved {during} tensors for backward"
    assert after > during, "saved_tensors_hooks detector is not working"
    assert control.grad_fn is not None
    assert isinstance(val, float)
    assert leaf.grad is None


# ─────────────────────────────────────────────────────────────────────────────
# collate_xy
# ─────────────────────────────────────────────────────────────────────────────

def test_collate_xy_stacks_and_drops_xm_alias():
    """
    The Dataset returns 'xm' as a backward-compat alias of 'x'. collate_xy
    deliberately drops it: with pin_memory=True a duplicated batch would be
    copied into pinned host memory every step for nothing. Assert the alias is
    really gone (and that x/y survive intact, in batch order).
    """
    batch = [_sample(16, 16, seed=s) for s in range(3)]
    out = T.collate_xy(batch)

    assert set(out.keys()) == {"x", "y"}, f"unexpected keys: {sorted(out)}"
    assert "xm" not in out
    assert_shape(out["x"], (3, 4, 16, 16), "collated x")
    assert_shape(out["y"], (3, 4, 16, 16), "collated y")
    for i, s in enumerate(batch):
        assert torch.equal(out["x"][i], s["x"]), f"x reordered at {i}"
        assert torch.equal(out["y"][i], s["y"]), f"y reordered at {i}"


def test_collate_xy_keeps_x_and_y_distinct():
    """A copy/paste slip ("y": stack of s['x']) would make the loss zero and
    training look miraculous; pin that x and y come from different keys."""
    batch = [_sample(8, 8, seed=1)]
    out = T.collate_xy(batch)
    assert not torch.equal(out["x"], out["y"])
    assert torch.equal(out["y"][0], batch[0]["y"])


def test_collate_xy_requires_uniform_sizes():
    """collate_xy is the Phase 1 (fixed-crop) path — it is torch.stack, so a
    ragged batch must fail loudly rather than silently truncate. This is why
    Phase 2 needs collate_pad_to_max at all."""
    batch = [_sample(16, 16, seed=0), _sample(16, 24, seed=1)]
    with pytest.raises(RuntimeError):
        T.collate_xy(batch)


# ─────────────────────────────────────────────────────────────────────────────
# collate_pad_to_max
# ─────────────────────────────────────────────────────────────────────────────

def test_collate_pad_no_op_when_uniform_and_multiple_of_8():
    """
    Packed dims already divisible by 8 (the three PixelUnshuffle(2) stages
    require it) and all equal -> the collate must not touch the pixels at all,
    and orig_h/orig_w must equal the real dims.
    """
    batch = [_sample(32, 24, seed=s) for s in range(2)]
    out = T.collate_pad_to_max(batch)

    assert_shape(out["x"], (2, 4, 32, 24), "collated x")
    assert torch.equal(out["orig_h"], torch.tensor([32, 32]))
    assert torch.equal(out["orig_w"], torch.tensor([24, 24]))
    for i, s in enumerate(batch):
        assert torch.equal(out["x"][i], s["x"])
        assert torch.equal(out["y"][i], s["y"])


@pytest.mark.parametrize(
    "sizes, exp_h, exp_w",
    [
        ([(16, 16), (24, 32)], 24, 32),        # already multiples of 8
        ([(17, 20), (23, 13)], 24, 24),        # neither is; max 23,20 -> 24,24
        ([(9, 9)], 16, 16),                    # single ragged image
        ([(32, 33), (30, 25)], 32, 40),        # w rounds up past the max
    ],
)
def test_collate_pad_rounds_max_up_to_multiple_of_8(sizes, exp_h, exp_w):
    """
    The padded batch must be (ceil8(max H), ceil8(max W)): the encoder halves
    the resolution three times, so a packed dim that is not a multiple of 8
    makes the skip connections mismatch and the forward pass crash.
    """
    batch = [_sample(h, w, seed=i) for i, (h, w) in enumerate(sizes)]
    out = T.collate_pad_to_max(batch)

    assert_shape(out["x"], (len(sizes), 4, exp_h, exp_w), "padded x")
    assert_shape(out["y"], (len(sizes), 4, exp_h, exp_w), "padded y")
    assert exp_h % 8 == 0 and exp_w % 8 == 0
    assert exp_h >= max(h for h, _ in sizes)
    assert exp_w >= max(w for _, w in sizes)


def test_collate_pad_preserves_real_pixels_top_left():
    """
    Padding must be bottom/right only: the loop rebuilds valid_mask as
    [:orig_h*2, :orig_w*2], so any content shifted by the pad would be scored
    against the wrong GT pixels for the whole of Phase 2.
    """
    sizes = [(17, 20), (23, 13)]
    batch = [_sample(h, w, seed=i) for i, (h, w) in enumerate(sizes)]
    out = T.collate_pad_to_max(batch)

    for i, (h, w) in enumerate(sizes):
        assert torch.equal(out["x"][i, :, :h, :w], batch[i]["x"])
        assert torch.equal(out["y"][i, :, :h, :w], batch[i]["y"])


def test_collate_pad_orig_sizes_are_true_pre_pad_dims():
    """orig_h/orig_w are the *packed* pre-pad dims as index-able integer
    tensors; the loop does `valid_mask[b, 0, :orig_h[b]*2, :orig_w[b]*2] = 1`,
    which needs integer dtype and per-sample ordering."""
    sizes = [(17, 20), (23, 13), (8, 40)]
    batch = [_sample(h, w, seed=i) for i, (h, w) in enumerate(sizes)]
    out = T.collate_pad_to_max(batch)

    assert out["orig_h"].dtype in (torch.int64, torch.int32)
    assert out["orig_w"].dtype in (torch.int64, torch.int32)
    assert out["orig_h"].tolist() == [h for h, _ in sizes]
    assert out["orig_w"].tolist() == [w for _, w in sizes]
    assert set(out.keys()) == {"x", "y", "orig_h", "orig_w"}


def test_collate_pad_valid_mask_reconstruction_covers_exactly_real_pixels():
    """
    End-to-end contract with the training loop: rebuilding valid_mask the way
    the loop does must mark exactly the sensor pixels that came from real
    data (2x the packed dims), no more and no less. A units slip (packed vs
    sensor) here silently averages the loss over reflected padding.
    """
    sizes = [(17, 20), (23, 13)]
    batch = [_sample(h, w, seed=i) for i, (h, w) in enumerate(sizes)]
    out = T.collate_pad_to_max(batch)

    B, _, H, W = out["x"].shape
    Hs, Ws = H * 2, W * 2
    valid = torch.zeros(B, 1, Hs, Ws)
    for b in range(B):
        valid[b, 0, :out["orig_h"][b] * 2, :out["orig_w"][b] * 2] = 1.0

    for b, (h, w) in enumerate(sizes):
        assert float(valid[b].sum()) == (2 * h) * (2 * w)
        assert bool((valid[b, 0, :2 * h, :2 * w] == 1).all())
        assert float(valid[b, 0, 2 * h:, :].sum()) == 0.0
        assert float(valid[b, 0, :, 2 * w:].sum()) == 0.0


def test_collate_pad_preserves_bayer_phase_in_padded_region():
    """
    Reflect-padding is applied in PACKED space, where each of the 4 channels is
    a single-colour plane. Padding therefore mirrors whole 2x2 Bayer cells and
    every padded sample stays in its own colour plane, so after
    pixel_shuffle(...,2) the sensor mosaic still has B at (even,even),
    G1 at (even,odd), G2 at (odd,even), R at (odd,odd) everywhere — padding
    included. (Reflect-padding the *mosaic* instead would flip the phase for
    odd pad widths and corrupt the demosaic.)

    Verified with channel-tagged data: channel c holds values in [c, c+1), so
    floor() of every padded sensor pixel must equal the channel index its
    position implies.
    """
    xs = [_channel_tagged_packed(17, 21, seed=0), _channel_tagged_packed(23, 13, seed=1)]
    batch = [{"x": x, "xm": x, "y": x.clone()} for x in xs]
    out = T.collate_pad_to_max(batch)

    mosaic = F.pixel_shuffle(out["x"], 2)          # [B, 1, 2H, 2W]
    tag = torch.floor(mosaic[:, 0])
    for c in range(4):
        r, col = c // 2, c % 2
        sub = tag[:, r::2, col::2]
        assert bool((sub == c).all()), (
            f"Bayer phase broken for channel {c}: found tags "
            f"{sorted(set(sub.unique().tolist()))}")


def test_collate_pad_padded_region_replicates_the_edge():
    """
    The padded values are not garbage/zeros: replicate mode repeats the last
    real row/column. Zeros would look like black pixels to the SNR estimator
    and to the encoder's receptive field near the border, so pin that the
    padding is a genuine continuation of the interior.

    This was `mode="reflect"` until reflect turned out to reject any pad that
    is not strictly smaller than the source dimension — see
    test_collate_pad_handles_a_pad_larger_than_the_image.
    """
    x = packed_bayer_3d(17, 16, seed=3)
    out = T.collate_pad_to_max([{"x": x, "xm": x, "y": x.clone()}])
    padded = out["x"][0]
    assert padded.shape[1] == 24            # 17 -> 24
    # rows 17..23 all repeat row 16, the last real one
    for r in range(17, 24):
        assert torch.equal(padded[:, r, :16], x[:, 16, :16]), \
            f"row {r} is not a replication of the last real row"


def test_collate_pad_handles_a_pad_larger_than_the_image():
    """
    A batch whose largest image is more than ~2x the smallest must still
    collate. `mode="reflect"` could not do this — PyTorch requires a reflect
    pad to be strictly smaller than the dimension, so this raised
    "Padding size should be less than the corresponding input dimension"
    instead of padding. It was latent only because Phase 2 runs at batch_sz=1
    (max == the image itself, so the pad is at most 7); it fired the moment
    anyone raised the Phase 2 batch size on a dataset with mixed orientations.
    """
    small = _sample(9, 9, seed=0)
    big = _sample(24, 24, seed=1)
    out = T.collate_pad_to_max([small, big])

    assert_shape(out["x"], (2, 4, 24, 24), "padded x")
    assert_shape(out["y"], (2, 4, 24, 24), "padded y")
    assert torch.isfinite(out["x"]).all()
    # The small image's real content survives in the top-left corner.
    assert torch.equal(out["x"][0, :, :9, :9], small["x"])
    assert torch.equal(out["orig_h"], torch.tensor([9, 24]))
    assert torch.equal(out["orig_w"], torch.tensor([9, 24]))


def test_collate_pad_batch_is_stackable_and_finite():
    """The whole reason this collate exists is that DataLoader must return one
    stacked tensor per key; also guard against NaNs sneaking in from the pad."""
    batch = [_sample(h, w, seed=i)
             for i, (h, w) in enumerate([(17, 20), (23, 13), (20, 19)])]
    out = T.collate_pad_to_max(batch)
    assert out["x"].shape == out["y"].shape
    assert out["x"].dtype == torch.float32
    assert_finite(out["x"], "padded x")
    assert_finite(out["y"], "padded y")


# ─────────────────────────────────────────────────────────────────────────────
# build_scheduler
# ─────────────────────────────────────────────────────────────────────────────

def test_build_scheduler_no_warmup_is_plain_cosine():
    """
    warmup_epochs == 0 (Phase 2: the model is already trained, no ramp) must
    return a bare CosineAnnealingLR starting at the full base LR — not a
    SequentialLR wrapper with an empty warmup, whose state_dict would not
    reload into a plain cosine on restart.
    """
    lr, eta_min, total = 5e-6, 1e-7, 30
    opt = _fresh_optimizer(lr)
    sched = T.build_scheduler(opt, 0, total, eta_min=eta_min)

    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert not isinstance(sched, torch.optim.lr_scheduler.SequentialLR)

    lrs = _lr_trajectory(sched, opt, total)
    assert lrs[0] == pytest.approx(lr, rel=1e-9)
    assert lrs[total] == pytest.approx(eta_min, rel=1e-6)
    diffs = [b - a for a, b in zip(lrs, lrs[1:])]
    assert all(d <= 1e-15 for d in diffs), "cosine LR increased somewhere"


def test_build_scheduler_warmup_returns_sequential():
    """warmup_epochs > 0 (Phase 1) must produce the SequentialLR
    warmup->cosine pair; the run's checkpoint stores this scheduler's
    state_dict, so the type is part of the resume contract."""
    opt = _fresh_optimizer(1e-4)
    sched = T.build_scheduler(opt, 10, 50)
    assert isinstance(sched, torch.optim.lr_scheduler.SequentialLR)


def test_build_scheduler_full_warmup_cosine_trajectory():
    """
    Steps the Phase 1 schedule (lr=1e-4, warmup=10, total=50, eta_min=1e-6)
    through every epoch and pins the shape of the whole curve:
      (a) epoch 0 starts at 1% of base lr (the "prevents early instability"
          claim in the source comment),
      (b) the base lr is reached exactly at epoch == warmup_epochs — one step
          per warmup epoch, no off-by-one,
      (c) the warmup segment is strictly increasing and never overshoots lr,
      (d) after the boundary the LR is monotonically non-increasing, and
      (e) the final epoch lands on eta_min.
    A break in any of these is invisible in the loss for several epochs.
    """
    lr, eta_min, warm, total = 1e-4, 1e-6, 10, 50
    opt = _fresh_optimizer(lr)
    sched = T.build_scheduler(opt, warm, total, eta_min=eta_min)
    lrs = _lr_trajectory(sched, opt, total)

    assert len(lrs) == total + 1
    # (a)
    assert lrs[0] == pytest.approx(0.01 * lr, rel=1e-6)
    # (b) no off-by-one at the milestone
    assert lrs[warm] == pytest.approx(lr, rel=1e-6), \
        f"warmup boundary LR {lrs[warm]} != base lr {lr}"
    assert lrs[warm - 1] < lr, "warmup reached base lr one epoch early"
    # (c)
    for i in range(warm):
        assert lrs[i + 1] > lrs[i], f"warmup not increasing at epoch {i}"
        assert lrs[i] <= lr + 1e-18
    # (d)
    for i in range(warm, total):
        assert lrs[i + 1] <= lrs[i] + 1e-18, \
            f"post-warmup LR increased at epoch {i}: {lrs[i]} -> {lrs[i+1]}"
    # (e)
    assert lrs[total] == pytest.approx(eta_min, rel=1e-6)
    assert min(lrs) == lrs[total]


def test_build_scheduler_warmup_is_linear_in_epoch():
    """
    LinearLR(start_factor=0.01, end_factor=1.0, total_iters=warmup) means the
    factor must be affine in the epoch index: 0.01 + 0.99*e/warmup. Pins that
    the ramp is linear (not exponential / cosine), which is what makes the
    "1% of lr" starting point meaningful.
    """
    lr, warm, total = 1e-4, 5, 20
    opt = _fresh_optimizer(lr)
    sched = T.build_scheduler(opt, warm, total)
    lrs = _lr_trajectory(sched, opt, total)
    for e in range(warm + 1):
        want = lr * (0.01 + 0.99 * e / warm)
        assert lrs[e] == pytest.approx(want, rel=1e-6), f"epoch {e}"


def test_build_scheduler_warmup_of_one_epoch():
    """
    Edge case warmup_epochs == 1: milestone and total_iters both 1. The
    SequentialLR milestone logic (`scheduler.step(0)` exactly on the
    milestone) is the fiddly part; a one-epoch warmup must still be
    0.01*lr then lr, and still anneal to eta_min at the end.
    """
    lr, eta_min, total = 1e-3, 1e-6, 8
    opt = _fresh_optimizer(lr)
    sched = T.build_scheduler(opt, 1, total, eta_min=eta_min)
    lrs = _lr_trajectory(sched, opt, total)

    assert lrs[0] == pytest.approx(0.01 * lr, rel=1e-6)
    assert lrs[1] == pytest.approx(lr, rel=1e-6)
    assert lrs[total] == pytest.approx(eta_min, rel=1e-6)
    for i in range(1, total):
        assert lrs[i + 1] <= lrs[i] + 1e-18


def test_build_scheduler_cosine_spans_remaining_epochs():
    """
    The cosine leg gets T_max = total - warmup, i.e. it finishes exactly at
    total_epochs rather than early (which would flat-line at eta_min) or late
    (which would stop mid-curve, wasting the anneal). Checked via the
    half-way LR of a cosine: (lr + eta_min)/2 at the midpoint of the leg.
    """
    lr, eta_min, warm, total = 1e-4, 1e-6, 10, 50
    opt = _fresh_optimizer(lr)
    sched = T.build_scheduler(opt, warm, total, eta_min=eta_min)
    lrs = _lr_trajectory(sched, opt, total)

    mid = warm + (total - warm) // 2
    assert lrs[mid] == pytest.approx((lr + eta_min) / 2, rel=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# write_log / create_folder
# ─────────────────────────────────────────────────────────────────────────────

def test_write_log_newfile_truncates_then_appends(tmp_path):
    """
    The run log is opened with newfile=True once at start-up and appended to
    for the rest of the run. If newfile did not truncate, a restart on a
    preemptable queue would interleave two runs in one file; if append did not
    append, only the last line would survive.
    """
    log = tmp_path / "train.log"
    T.write_log(str(log), "first run line", newfile=True)
    T.write_log(str(log), "second line")
    assert log.read_text() == "first run line\nsecond line\n"

    T.write_log(str(log), "restart", newfile=True)
    assert log.read_text() == "restart\n"


def test_write_log_creates_missing_file_and_honours_end(tmp_path):
    """newfile=True on a non-existent path must create it (the save folder is
    created first, the log is not pre-touched), and `end` must be respected so
    progress lines can be written without a newline."""
    log = tmp_path / "fresh.log"
    assert not log.exists()
    T.write_log(str(log), "hdr", newfile=True, end="")
    T.write_log(str(log), "|tail", end="")
    assert log.read_text() == "hdr|tail"


def test_write_log_stringifies_non_str(tmp_path):
    """The loop passes the string returned by print_epoch_loss, but "%s" also
    has to survive being handed a number — a TypeError here kills a run that
    was otherwise fine."""
    log = tmp_path / "n.log"
    T.write_log(str(log), 3.5, newfile=True)
    T.write_log(str(log), {"a": 1})
    assert log.read_text().splitlines() == ["3.5", "{'a': 1}"]


def test_create_folder_is_idempotent(tmp_path):
    """
    create_folder runs on every start-up, including a resume into an existing
    run folder (HDR_SAVE_FOLDER pins the name on preemptable queues), so it
    must be exist_ok and must not disturb existing contents.
    """
    d = tmp_path / "models_p1_moe"
    T.create_folder(str(d))
    assert d.is_dir()
    keep = d / "latest.pth"
    keep.write_text("checkpoint")

    T.create_folder(str(d))          # second call must not raise or wipe
    assert d.is_dir()
    assert keep.read_text() == "checkpoint"


def test_create_folder_makes_nested_path(tmp_path):
    """save_folder can be a nested relative path from the submit script;
    makedirs must build the whole chain, not just the leaf."""
    d = tmp_path / "a" / "b" / "c"
    T.create_folder(str(d))
    assert d.is_dir()


# ─────────────────────────────────────────────────────────────────────────────
# print_running_loss / print_epoch_loss
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("print_every", [1, 5, 20])
def test_print_running_loss_cadence(capsys, print_every):
    """
    The step line is printed only every print_every steps: at batch 8 and
    thousands of steps per epoch, printing every step floods the job's stdout
    file (megabytes of log per epoch on a shared filesystem). Pins that it
    fires on i % print_every == print_every - 1, i.e. on the 1-based multiples.
    """
    n = 40
    for i in range(n):
        T.print_running_loss(1.0 * (i + 1), 0.0, i, print_every=print_every)
    out = capsys.readouterr().out

    steps = [int(chunk.split()[1]) for chunk in out.split("\r") if chunk.strip()]
    assert steps == [s for s in range(1, n + 1) if s % print_every == 0]


def test_print_running_loss_is_silent_off_cadence(capsys):
    """Off-cadence calls must print absolutely nothing (not even a bare \\r),
    otherwise the progress line is cleared on every step."""
    for i in (0, 1, 5, 18):
        T.print_running_loss(10.0, 100.0, i, print_every=20)
    assert capsys.readouterr().out == ""


def test_print_running_loss_reports_running_averages(capsys):
    """
    Both numbers are sums accumulated over the epoch and must be divided by
    the number of steps so far (i+1) — printing the raw sums would make the
    logged loss grow linearly with the step index and look like divergence.
    """
    T.print_running_loss(50.0, 600.0, 19, print_every=20)   # 20 steps so far
    out = capsys.readouterr().out
    assert out == "\r  step    20  loss =     2.5000  PSNR-µ = 30.00"


def test_print_running_loss_omits_psnr_when_not_tracked(capsys):
    """
    A falsy running_psnr_mu (0.0 — the "not accumulated" sentinel) suppresses
    the PSNR field entirely, so the line must contain no PSNR text and no
    stray separator.
    """
    T.print_running_loss(40.0, 0.0, 19, print_every=20)
    out = capsys.readouterr().out
    assert out == "\r  step    20  loss =     2.0000"
    assert "PSNR" not in out


def test_print_epoch_loss_format_and_return(capsys):
    """
    print_epoch_loss returns the exact string it printed; write_log stores that
    return value, so the terminal log and the on-disk log must not be allowed
    to drift. Also pins the phase tag / field widths that make the log
    greppable.
    """
    msg = T.print_epoch_loss(3, 1.5, 30.0, 28.25, 1, time_spent=12.34)
    out = capsys.readouterr().out
    assert msg == "\r[P1] Epoch   3  loss=    1.5000  PSNR-µ= 30.00  PSNR= 28.25  (12.3s)"
    assert out == msg + "\n"


def test_print_epoch_loss_unknown_time(capsys):
    """time_spent=None must render as "?" rather than crashing on the %.1f
    format — the first epoch of a resumed run has no timing."""
    msg = T.print_epoch_loss(7, 0.25, 31.0, 29.0, 2, time_spent=None)
    assert msg.endswith("(?)")
    assert msg.startswith("\r[P2] Epoch   7")
    assert capsys.readouterr().out == msg + "\n"


def test_print_epoch_loss_prints_once_per_call(capsys):
    """Unlike the step line there is no cadence here: exactly one line per
    epoch, so the epoch count in a log file equals the epochs actually run."""
    for e in range(3):
        T.print_epoch_loss(e, 1.0, 20.0, 19.0, 1, time_spent=1.0)
    out = capsys.readouterr().out
    assert out.count("\n") == 3
    assert out.count("[P1] Epoch") == 3
