"""
tests/test_dataset_loader.py — MobileHDRDataset (HDR_Mobile_dataset.py)
=======================================================================

Every test here builds its own synthetic on-disk tree under `tmp_path`
(via `helpers.make_dataset` or the local `_write_pt`).  The real Mobile-HDR
dataset is covered by a separate file; nothing here touches it.

What this file pins down, and why each one matters for a training run:

* **Split semantics / file discovery.**  A silently-truncated or silently-
  empty file list turns a 40-hour training run into garbage without ever
  raising.  So the recursive train glob, the flat `with_gt` test glob, the
  sort, and both error paths are asserted explicitly.
* **The `{x, xm, y}` payload contract** that `collate_xy` and
  `collate_pad_to_max` in `train_A100_MoE_two_phase.py` consume: `[4, h, w]`,
  values in [0, 1], finite, `xm` an alias of `x`.
* **The `crop_size` efficiency claim.**  The crop must happen BEFORE noise
  synthesis — that is the entire point ("~12x less dataloader CPU").  A
  regression here is invisible in the loss curve and only shows up as a
  starved GPU.
* **The per-file range cache.**  Crops are normalised by the FULL image
  range, so a dark crop stays dark.  If this regressed to per-crop
  normalisation, every dark patch would be silently contrast-stretched to
  full scale and the Poisson-Gaussian model would be applied at the wrong
  signal level — a training-data corruption that never raises.
* **Test-split determinism.**  `test_dual_MoE_two_phase.py` claims
  reproducible benchmark numbers; that claim lives entirely in the
  per-index `torch.Generator` here, and it must survive a different global
  RNG state and a different DataLoader worker count.
* **Crop geometry.**  `top` must index H and `left` must index W; a swapped
  pair would still produce correctly-shaped square patches and would sail
  through any shape-only test.
"""

import os
import random

import pytest
import torch
from torch.utils.data import DataLoader

import HDR_Mobile_dataset as ds_mod
from HDR_Mobile_dataset import MobileHDRDataset

from helpers import make_dataset, assert_finite, assert_in_range, assert_shape


PIX_MAX_10BIT = 1023.0


# ─────────────────────────────────────────────────────────────────────────────
# Local helpers (deliberately NOT in helpers.py — other agents share that file)
# ─────────────────────────────────────────────────────────────────────────────

def _write_pt(root, rel, tensor):
    """Save `tensor` (as contiguous float32) at root/rel. Returns the abs path."""
    path = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(tensor.contiguous().float(), path)
    return path


def _unit_range_image(c=4, h=40, w=64, seed=7):
    """
    A [c, h, w] tensor whose exact min is 0.0 and exact max is 1.0, with all
    other values random and distinct.

    Because the dataset normalises by (min, max) -> [0, pix_max] and then
    divides by pix_max again, an image with range exactly [0, 1] round-trips
    to itself: `y` comes back bit-comparable to the (cropped) source.  That
    turns fuzzy "looks plausible" checks into exact equality assertions.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.rand((c, h, w), generator=g)
    t[0, 0, 0] = 0.0
    t[0, -1, -1] = 1.0
    return t


def _two_zone_image(h=64, w=64, dark_lo=0.20, dark_hi=0.24):
    """
    Top half is a full-scale ramp (contains the global min 0.0 and max 1.0);
    bottom half is a narrow DARK band in [dark_lo, dark_hi].

    A crop taken from the bottom half is dark in absolute terms but has a
    tiny local range, so full-image vs. per-crop normalisation give wildly
    different answers (0.2 vs 1.0).  This is the fixture that can tell them
    apart.
    """
    img = torch.empty((4, h, w))
    half = h // 2
    bright = torch.linspace(0.0, 1.0, half).view(1, half, 1).expand(4, half, w)
    dark = torch.linspace(dark_lo, dark_hi, h - half).view(1, h - half, 1) \
                .expand(4, h - half, w)
    img[:, :half] = bright
    img[:, half:] = dark
    return img.contiguous()


class _ScriptedRandom:
    """
    Stand-in for the `random` module as seen by HDR_Mobile_dataset, so crop
    offsets become deterministic without patching the stdlib for everyone.

    Records every (a, b) it was called with, which lets a test assert the
    offset is drawn from the full legal range and that H and W bounds are
    not swapped.
    """

    def __init__(self, values):
        self._values = list(values)
        self.calls = []

    def randint(self, a, b):
        self.calls.append((a, b))
        i = len(self.calls) - 1
        v = self._values[i] if i < len(self._values) else self._values[-1]
        return max(a, min(b, v))


def _spy_on_noise(monkeypatch):
    """
    Wrap HDR_Mobile_dataset.add_photon_noise, recording (input_shape, kwargs)
    for each call and delegating to the real implementation.
    """
    real = ds_mod.add_photon_noise
    calls = []

    def spy(image, **kwargs):
        calls.append((tuple(image.shape), dict(kwargs)))
        return real(image, **kwargs)

    monkeypatch.setattr(ds_mod, "add_photon_noise", spy)
    return calls


def _spy_on_loads(dataset):
    """Record the path of every _load_clean call on this dataset instance."""
    real = dataset._load_clean
    seen = []

    def spy(path):
        seen.append(path)
        return real(path)

    dataset._load_clean = spy
    return seen


def _identity_transform(noisy, gt):
    """Module-level (picklable) no-op transform for worker-process tests."""
    return noisy, gt


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tree(tmp_path):
    """Standard synthetic tree: 3 train tensors across 2 subdirs, 2 test."""
    return make_dataset(tmp_path / "ds", n_train=3, n_test=2, h=64, w=64)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Construction, split semantics, file discovery
# ═════════════════════════════════════════════════════════════════════════════

def test_train_split_reads_tensors_recursively(tree):
    """
    train reads base/train/tensors/**/*.pt recursively.

    The real dataset is organised into `static/` and `dynamic/` subdirs; a
    non-recursive glob would find zero files there and the dataset would
    raise instead of training. Also asserts nothing from the test split
    leaked in.
    """
    ds = MobileHDRDataset(tree, split="train")
    assert ds.tensor_dir == os.path.join(tree, "train", "tensors")
    assert len(ds.file_list) == 3
    assert {os.path.basename(p) for p in ds.file_list} == \
        {"train_000.pt", "train_001.pt", "train_002.pt"}
    # Files really do live one level down, so recursion was required.
    assert all(os.path.dirname(p) != ds.tensor_dir for p in ds.file_list)
    assert not any(os.sep + "test" + os.sep in p for p in ds.file_list)


def test_train_glob_finds_files_at_any_depth(tmp_path):
    """
    The recursive `**` pattern must match tensors directly in `tensors/` and
    tensors nested arbitrarily deep.

    `tensors/**/*.pt` matching zero directory levels is a real (and easy to
    get wrong) property of Python's recursive glob; a flat-only or
    one-level-only match would silently drop part of the corpus.
    """
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/flat.pt", torch.rand(4, 16, 16))
    _write_pt(root, "train/tensors/a/one.pt", torch.rand(4, 16, 16))
    _write_pt(root, "train/tensors/a/b/two.pt", torch.rand(4, 16, 16))
    _write_pt(root, "test/tensors/with_gt/t.pt", torch.rand(4, 16, 16))

    ds = MobileHDRDataset(str(root), split="train")
    assert {os.path.basename(p) for p in ds.file_list} == \
        {"flat.pt", "one.pt", "two.pt"}


def test_test_split_reads_with_gt_non_recursively(tmp_path):
    """
    test reads base/test/tensors/with_gt/*.pt and nothing else.

    Benchmark numbers are only comparable if the evaluation set is exactly
    the `with_gt` frames: neither the no-GT frames sitting beside it nor
    anything nested below it may be swept in.
    """
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/x.pt", torch.rand(4, 16, 16))
    _write_pt(root, "test/tensors/with_gt/keep_a.pt", torch.rand(4, 16, 16))
    _write_pt(root, "test/tensors/with_gt/keep_b.pt", torch.rand(4, 16, 16))
    _write_pt(root, "test/tensors/with_gt/nested/skip.pt", torch.rand(4, 16, 16))
    _write_pt(root, "test/tensors/without_gt/skip2.pt", torch.rand(4, 16, 16))
    _write_pt(root, "test/tensors/skip3.pt", torch.rand(4, 16, 16))

    ds = MobileHDRDataset(str(root), split="test")
    assert ds.tensor_dir == os.path.join(str(root), "test", "tensors", "with_gt")
    assert [os.path.basename(p) for p in ds.file_list] == ["keep_a.pt", "keep_b.pt"]


def test_unknown_split_raises_value_error(tree):
    """
    An unrecognised split is a ValueError naming the legal values.

    Typos like "val" or "Train" must fail loudly at construction, not fall
    through to an empty file list or (worse) silently train on the test set.
    """
    for bad in ("val", "Train", "TEST", "", "validation"):
        with pytest.raises(ValueError, match="Unknown split"):
            MobileHDRDataset(tree, split=bad)
    # The message tells the caller what the legal values are.
    with pytest.raises(ValueError, match=r"train \| test"):
        MobileHDRDataset(tree, split="val")


@pytest.mark.parametrize("split", ["train", "test"])
def test_missing_tensor_dir_raises_runtime_error(tmp_path, split):
    """
    Pointing at a base_dir with no tensor directory raises RuntimeError
    naming the directory it looked in.

    A wrong --dataset-dir is the single most common launch mistake; it must
    fail in seconds with the path it tried, not after the scheduler has
    burned a node-hour.
    """
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No .pt tensors found"):
        MobileHDRDataset(str(empty), split=split)

    expected_dir = (os.path.join(str(empty), "train", "tensors") if split == "train"
                    else os.path.join(str(empty), "test", "tensors", "with_gt"))
    with pytest.raises(RuntimeError, match=None) as exc:
        MobileHDRDataset(str(empty), split=split)
    assert expected_dir in str(exc.value)


@pytest.mark.parametrize("split,rel", [
    ("train", "train/tensors"),
    ("test", "test/tensors/with_gt"),
])
def test_existing_but_empty_tensor_dir_raises_runtime_error(tmp_path, split, rel):
    """
    A tensor directory that exists but holds no .pt raises RuntimeError.

    This is the "conversion script never ran" case. An empty file_list would
    otherwise give len(dataset) == 0 and a training loop that completes every
    epoch instantly while learning nothing.
    """
    d = tmp_path / "ds" / rel
    d.mkdir(parents=True)
    (d / "README.txt").write_text("no tensors yet")
    with pytest.raises(RuntimeError, match="No .pt tensors found"):
        MobileHDRDataset(str(tmp_path / "ds"), split=split)


def test_non_pt_files_are_ignored(tmp_path):
    """
    Only *.pt is picked up; sidecar files must not enter the file list.

    Conversion pipelines leave .npz/.json/.png droppings next to the tensors;
    torch.load on those would crash a worker mid-epoch.
    """
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/s/real.pt", torch.rand(4, 16, 16))
    for junk in ("meta.json", "orig.npz", "preview.png", "notes.pt.bak"):
        (root / "train" / "tensors" / "s" / junk).write_text("junk")

    ds = MobileHDRDataset(str(root), split="train")
    assert [os.path.basename(p) for p in ds.file_list] == ["real.pt"]


def test_base_dir_may_be_a_pathlib_path(tmp_path):
    """
    A pathlib.Path base_dir works as well as a str.

    Callers routinely pass Path objects; os.path.join accepts them, and this
    pins that no str-only operation sneaks into the construction path.
    """
    root = make_dataset(tmp_path / "ds", n_train=2, n_test=1, h=16, w=16)
    ds = MobileHDRDataset(tmp_path / "ds", split="train")
    assert len(ds.file_list) == 2
    assert ds.tensor_dir == os.path.join(root, "train", "tensors")


def test_constructor_stores_noise_config_verbatim(tmp_path):
    """
    Every noise/augmentation knob is stored on the instance as given, and
    read_noise_range is normalised to a tuple.

    The training script sets these once at launch; if the constructor
    dropped or rewrote one, the run would silently use the defaults.
    """
    root = make_dataset(tmp_path / "ds", n_train=1, n_test=1, h=16, w=16)
    ds = MobileHDRDataset(root, split="train", nbits=12, random_alpha=False,
                          num_patch=5, crop_size=8, do_expand=True,
                          shot_gain=3.5, read_noise_range=[1.0, 2.0],
                          test_noise_seed=99)
    assert (ds.nbits, ds.random_alpha, ds.num_patch, ds.crop_size) == (12, False, 5, 8)
    assert (ds.do_expand, ds.shot_gain, ds.test_noise_seed) == (True, 3.5, 99)
    assert ds.read_noise_range == (1.0, 2.0)
    assert isinstance(ds.read_noise_range, tuple), \
        "read_noise_range must be a tuple so it cannot be mutated per-worker"


# ═════════════════════════════════════════════════════════════════════════════
# 2. File list ordering
# ═════════════════════════════════════════════════════════════════════════════

def test_file_list_is_sorted_not_filesystem_ordered(tmp_path):
    """
    file_list is lexicographically sorted regardless of creation order.

    glob order is filesystem-dependent; without the sort, index i would map
    to a different frame on a different machine and "deterministic test noise
    per index" would be meaningless across hosts.
    """
    root = tmp_path / "ds"
    # Created in deliberately reverse-lexicographic order, across subdirs.
    creation_order = [
        "train/tensors/zebra/z_9.pt",
        "train/tensors/zebra/z_1.pt",
        "train/tensors/alpha/m_5.pt",
        "train/tensors/alpha/a_0.pt",
    ]
    for rel in creation_order:
        _write_pt(root, rel, torch.rand(4, 16, 16))

    ds = MobileHDRDataset(str(root), split="train")
    assert ds.file_list == sorted(ds.file_list)
    assert [os.path.relpath(p, ds.tensor_dir) for p in ds.file_list] == [
        os.path.join("alpha", "a_0.pt"),
        os.path.join("alpha", "m_5.pt"),
        os.path.join("zebra", "z_1.pt"),
        os.path.join("zebra", "z_9.pt"),
    ]


def test_test_file_list_is_sorted(tmp_path):
    """
    The test split is sorted too, so benchmark index i is a stable frame.

    The eval script prints per-index metrics; unsorted order would shuffle
    which image "sample 3" refers to between runs.
    """
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/a.pt", torch.rand(4, 16, 16))
    for name in ("t_10.pt", "t_02.pt", "t_01.pt", "t_20.pt"):
        _write_pt(root, f"test/tensors/with_gt/{name}", torch.rand(4, 16, 16))

    ds = MobileHDRDataset(str(root), split="test")
    assert [os.path.basename(p) for p in ds.file_list] == \
        ["t_01.pt", "t_02.pt", "t_10.pt", "t_20.pt"]


@pytest.mark.parametrize("split", ["train", "test"])
def test_two_instances_agree_on_file_order(tree, split):
    """
    Two datasets over the same tree produce identical file lists.

    Train/eval scripts build the dataset separately (and per worker); index
    stability across instances is what makes checkpoint-time and eval-time
    sample identities line up.
    """
    a = MobileHDRDataset(tree, split=split)
    b = MobileHDRDataset(tree, split=split)
    assert a.file_list == b.file_list


# ═════════════════════════════════════════════════════════════════════════════
# 3. __len__
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("num_patch", [1, 3, 8, 16])
def test_len_train_is_files_times_num_patch(tree, num_patch):
    """
    train: len == len(file_list) * num_patch (the virtual epoch).

    The training script prints "N virtual samples -> M batches/epoch" and the
    LR scheduler is stepped per batch, so a wrong length rescales the whole
    schedule.
    """
    ds = MobileHDRDataset(tree, split="train", num_patch=num_patch)
    assert len(ds.file_list) == 3
    assert len(ds) == 3 * num_patch


@pytest.mark.parametrize("num_patch", [1, 8, 16])
def test_len_test_is_one_per_file_and_ignores_num_patch(tree, num_patch):
    """
    test: len == len(file_list); num_patch must not inflate it.

    Evaluation must visit each frame exactly once, or the reported mean PSNR
    is an average over duplicated frames.
    """
    ds = MobileHDRDataset(tree, split="test", num_patch=num_patch)
    assert len(ds) == len(ds.file_list) == 2


# ═════════════════════════════════════════════════════════════════════════════
# 4. __getitem__ payload contract
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("split", ["train", "test"])
def test_getitem_payload_keys_shape_dtype_range_finite(tree, split):
    """
    __getitem__ returns exactly {x, xm, y}, each [4, h, w] float32, finite,
    and inside [0, 1].

    This is the contract collate_xy / collate_pad_to_max and the model's
    input assumptions are built on: 4 packed BGGR channels, already
    normalised, no NaNs to poison the first backward pass.
    """
    ds = MobileHDRDataset(tree, split=split, num_patch=2)
    sample = ds[0]
    assert set(sample.keys()) == {"x", "xm", "y"}
    for key in ("x", "xm", "y"):
        t = sample[key]
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32, f"{key} dtype {t.dtype}"
        assert_shape(t, (4, 64, 64), name=key)
        assert_finite(t, name=key)
        assert_in_range(t, 0.0, 1.0, name=key)


@pytest.mark.parametrize("split", ["train", "test"])
def test_xm_is_an_alias_of_x(tree, split):
    """
    'xm' is the documented back-compat duplicate of 'x' — equal values and
    the same underlying storage.

    collate_xy skips 'xm' precisely because it is a duplicate ("stacking and
    pinning it would waste CPU"). If xm ever became a genuinely different
    tensor (e.g. a mosaic), that optimisation would silently drop data.
    """
    ds = MobileHDRDataset(tree, split=split)
    s = ds[0]
    assert torch.equal(s["x"], s["xm"])
    assert s["x"].data_ptr() == s["xm"].data_ptr(), "xm should alias x, not copy it"


def test_test_split_gt_is_exactly_the_range_normalised_image(tmp_path):
    """
    On the test split, y == (img - min) / (max - min) exactly: alpha is
    forced to 1.0 and no expansion offset is applied.

    y is the regression target. If the test split applied the low-light
    alpha (as train does), every reported PSNR would be measured against a
    randomly dimmed target and would be uncomparable to the train-time loss.
    """
    img = _unit_range_image(h=32, w=48)
    root = tmp_path / "ds"
    _write_pt(root, "test/tensors/with_gt/t.pt", img)
    ds = MobileHDRDataset(str(root), split="test", random_alpha=True,
                          do_expand=True)

    y = ds[0]["y"]
    # min == 0, max == 1 for this fixture, so the normalisation is identity.
    assert torch.allclose(y, img, atol=1e-5)


def test_noisy_input_is_quantised_to_the_nbits_grid(tmp_path):
    """
    x lands on the 2**nbits - 1 quantisation grid; y does not.

    x models a sensor readout (integral DN / pix_max), while y is the
    unquantised clean signal. Losing the quantisation would make the
    denoiser's job easier than reality; quantising y would inject
    quantisation noise into the target.
    """
    img = _unit_range_image(h=32, w=32)
    root = tmp_path / "ds"
    _write_pt(root, "test/tensors/with_gt/t.pt", img)

    for nbits in (8, 10, 12):
        pix_max = float(2 ** nbits - 1)
        ds = MobileHDRDataset(str(root), split="test", nbits=nbits)
        s = ds[0]
        grid = s["x"] * pix_max
        assert torch.allclose(grid, grid.round(), atol=1e-3), \
            f"x is not on the {nbits}-bit grid"
        ygrid = s["y"] * pix_max
        assert not torch.allclose(ygrid, ygrid.round(), atol=1e-3), \
            "y should keep sub-DN precision"


def test_noisy_input_is_noisy_but_still_tracks_the_target(tmp_path):
    """
    x differs from y everywhere-ish, yet correlates strongly with it.

    Guards both failure directions at once: x == y would mean the noise
    synthesis silently no-op'd (nothing to denoise), while an uncorrelated x
    would mean the signal was destroyed (nothing recoverable).
    """
    img = _unit_range_image(h=48, w=48)
    root = tmp_path / "ds"
    _write_pt(root, "test/tensors/with_gt/t.pt", img)
    s = MobileHDRDataset(str(root), split="test")[0]

    assert not torch.equal(s["x"], s["y"])
    assert (s["x"] != s["y"]).float().mean() > 0.5, "noise barely perturbed anything"
    corr = torch.corrcoef(torch.stack([s["x"].flatten(), s["y"].flatten()]))[0, 1]
    assert float(corr) > 0.5, f"x lost the signal (corr={float(corr):.3f})"


def test_getitem_does_not_mutate_the_source_tensor_on_disk(tmp_path):
    """
    Reading a sample leaves the .pt file byte-identical.

    Tensors are loaded with mmap=True; any in-place op inside the noise
    pipeline would write through to the dataset on disk and corrupt it
    permanently. Nothing recovers from that.
    """
    img = _unit_range_image(h=32, w=32)
    root = tmp_path / "ds"
    path = _write_pt(root, "test/tensors/with_gt/t.pt", img)
    _write_pt(root, "train/tensors/s/t.pt", img)
    before = torch.load(path, weights_only=True).clone()

    ds = MobileHDRDataset(str(root), split="test")
    for _ in range(3):
        ds[0]
    train_ds = MobileHDRDataset(str(root), split="train", crop_size=16, num_patch=2)
    for i in range(len(train_ds)):
        train_ds[i]

    assert torch.equal(torch.load(path, weights_only=True), before), \
        "the on-disk tensor was modified in place"


@pytest.mark.parametrize("bad_shape", [(3, 16, 16), (5, 16, 16), (16, 16),
                                       (1, 4, 16, 16)])
def test_non_bayer_tensor_shape_is_rejected(tmp_path, bad_shape):
    """
    A stored tensor that is not [4, H, W] fails loudly on access.

    The whole pipeline assumes 4 packed BGGR channels. A [3, H, W] RGB
    tensor left in the corpus would otherwise flow into the model and be
    reinterpreted as Bayer — garbage that trains without complaint.
    """
    root = tmp_path / "ds"
    _write_pt(root, "test/tensors/with_gt/bad.pt", torch.rand(*bad_shape))
    ds = MobileHDRDataset(str(root), split="test")
    with pytest.raises(AssertionError, match=r"Expected \[4, H, W\]"):
        ds[0]


# ═════════════════════════════════════════════════════════════════════════════
# 5. crop_size
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("crop", [8, 16, 32, 64])
def test_train_crop_size_yields_square_crop(tmp_path, crop):
    """
    train + crop_size=k returns [4, k, k], including k == the full frame.

    Phase 1 trains on uniform square patches so collate_xy can stack them;
    an off-by-one here breaks batching for the entire run.
    """
    root = make_dataset(tmp_path / "ds", n_train=2, n_test=1, h=64, w=64)
    ds = MobileHDRDataset(root, split="train", crop_size=crop, num_patch=2)
    for i in (0, len(ds) - 1):
        s = ds[i]
        for key in ("x", "xm", "y"):
            assert_shape(s[key], (4, crop, crop), name=key)
        assert_in_range(s["x"], 0.0, 1.0, name="x")
        assert_finite(s["y"], name="y")


def test_crop_is_taken_before_noise_synthesis(tmp_path, monkeypatch):
    """
    add_photon_noise sees the CROPPED tensor, and is handed the FULL-image
    (norm_min, norm_max).

    This is the documented efficiency claim — noise for crop_size^2 pixels
    instead of a ~2K x 1.5K frame (~12x less dataloader CPU). It is also the
    only place the two requirements meet: crop first (cheap) but normalise by
    the full range (correct brightness). Shape-only tests cannot see a
    regression to "noise first, then crop".
    """
    img = _unit_range_image(h=64, w=64)
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/s/a.pt", img)
    calls = _spy_on_noise(monkeypatch)

    ds = MobileHDRDataset(str(root), split="train", crop_size=16)
    ds[0]

    assert len(calls) == 1
    shape, kwargs = calls[0]
    assert shape == (4, 16, 16), \
        f"noise was synthesised for {shape}, so the crop happened after it"
    assert kwargs["norm_min"] == pytest.approx(0.0, abs=1e-6)
    assert kwargs["norm_max"] == pytest.approx(1.0, abs=1e-6)


def test_crop_offsets_index_h_and_w_in_that_order(tmp_path, monkeypatch):
    """
    top is drawn from [0, H - crop] and left from [0, W - crop], and the
    returned patch is exactly img[:, top:top+k, left:left+k].

    Deliberately uses a NON-square frame: with H == W a swapped (top, left)
    pair still yields a correctly-shaped square patch and passes every
    shape assertion while quietly reading the wrong pixels.
    """
    img = _unit_range_image(h=40, w=64)      # H != W on purpose
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/s/a.pt", img)

    scripted = _ScriptedRandom([5, 7])       # top=5, left=7
    monkeypatch.setattr(ds_mod, "random", scripted)

    ds = MobileHDRDataset(str(root), split="train", crop_size=16,
                          random_alpha=False)
    y = ds[0]["y"]

    assert scripted.calls == [(0, 40 - 16), (0, 64 - 16)], \
        f"offset bounds wrong or H/W swapped: {scripted.calls}"
    # Range is exactly [0, 1], so y round-trips to the raw crop values.
    assert torch.allclose(y, img[:, 5:21, 7:23], atol=1e-5)


def test_crop_offsets_vary_across_calls(tmp_path):
    """
    Repeated reads of the same index land on different crop offsets.

    num_patch virtual repeats only add value if each visit sees a different
    region; a frozen offset would turn 8 repeats into 8 copies of one patch.
    """
    img = _unit_range_image(h=64, w=64)
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/s/a.pt", img)
    ds = MobileHDRDataset(str(root), split="train", crop_size=16,
                          random_alpha=False)

    # y round-trips to raw crop values, so distinct crops => distinct digests.
    digests = {tuple(ds[0]["y"].flatten()[:8].tolist()) for _ in range(12)}
    assert len(digests) > 1, "crop offset never changed across 12 reads"


@pytest.mark.parametrize("h,w", [(24, 64), (64, 24), (24, 24), (31, 31)])
def test_crop_larger_than_image_raises_helpful_runtime_error(tmp_path, h, w):
    """
    crop_size bigger than the frame in EITHER dimension raises RuntimeError
    naming the file, its size and the crop_size.

    Without the guard, random.randint(0, negative) raises a bare "empty
    range" ValueError from deep inside a worker with no clue which frame was
    at fault. Both dimensions are covered because a guard that only checked
    h would pass the (24, 64) case and then crash confusingly on (64, 24).
    """
    root = tmp_path / "ds"
    small = _write_pt(root, "train/tensors/s/small.pt", torch.rand(4, h, w))
    ds = MobileHDRDataset(str(root), split="train", crop_size=32)
    with pytest.raises(RuntimeError) as exc:
        ds[0]
    msg = str(exc.value)
    assert "smaller than crop_size" in msg
    assert small in msg
    assert f"{h}x{w}" in msg, f"size not reported as HxW: {msg}"
    assert "crop_size=32" in msg


def test_crop_size_is_ignored_on_the_test_split(tmp_path):
    """
    split="test" returns FULL frames even when crop_size is passed.

    crop_size is documented train-only. Cropping at eval time would report
    PSNR over random patches instead of whole images and would make the
    numbers depend on the RNG — silently, since the shapes stay valid.
    """
    root = tmp_path / "ds"
    _write_pt(root, "test/tensors/with_gt/t.pt", _unit_range_image(h=40, w=56))
    ds = MobileHDRDataset(str(root), split="test", crop_size=8, num_patch=4)
    s = ds[0]
    for key in ("x", "xm", "y"):
        assert_shape(s[key], (4, 40, 56), name=key)


def test_test_split_below_crop_size_still_works(tmp_path):
    """
    A test frame smaller than crop_size does NOT raise.

    Corollary of crop_size being train-only: the size guard must not fire on
    the eval path, or a small benchmark frame would abort the whole eval run.
    """
    root = tmp_path / "ds"
    _write_pt(root, "test/tensors/with_gt/t.pt", _unit_range_image(h=8, w=8))
    ds = MobileHDRDataset(str(root), split="test", crop_size=64)
    assert_shape(ds[0]["x"], (4, 8, 8), name="x")


def test_crop_size_none_keeps_the_full_frame(tmp_path, monkeypatch):
    """
    crop_size=None (legacy / Phase 2) adds noise to the full frame and never
    consults the RNG for offsets.

    Phase 2 trains at full resolution with collate_pad_to_max; a sneaky
    default crop would silently shrink the Phase 2 receptive field.
    """
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/s/a.pt", _unit_range_image(h=40, w=64))
    scripted = _ScriptedRandom([0])
    monkeypatch.setattr(ds_mod, "random", scripted)
    calls = _spy_on_noise(monkeypatch)

    ds = MobileHDRDataset(str(root), split="train", crop_size=None)
    s = ds[0]
    assert_shape(s["x"], (4, 40, 64), name="x")
    assert calls[0][0] == (4, 40, 64)
    assert scripted.calls == [], "crop RNG was consulted with crop_size=None"


def test_crop_preserves_packed_bggr_channel_identity(tmp_path):
    """
    Cropping slices only H and W: channel c of the patch is still channel c
    (B, G1, G2, R) of the source.

    Every packed pixel is a whole BGGR cell, so any (top, left) is legal —
    but only if the crop never touches dim 0. A channel roll would flip the
    colour interpretation of the entire training set while keeping shapes,
    ranges and finiteness perfectly valid.
    """
    root = tmp_path / "ds"
    # Per-channel plateaus, strictly increasing B < G1 < G2 < R, plus a ramp
    # so the image range is non-degenerate.
    base = torch.tensor([0.10, 0.35, 0.60, 0.85]).view(4, 1, 1)
    ramp = torch.linspace(0.0, 0.10, 64).view(1, 1, 64)
    img = (base + ramp).expand(4, 64, 64).contiguous()
    img[0, 0, 0] = 0.0
    img[3, -1, -1] = 1.0
    _write_pt(root, "train/tensors/s/a.pt", img)

    ds = MobileHDRDataset(str(root), split="train", crop_size=16,
                          random_alpha=False)
    for _ in range(6):
        y = ds[0]["y"]
        means = [float(y[c].mean()) for c in range(4)]
        assert means == sorted(means), f"channel order scrambled by crop: {means}"
        # And the absolute level of each channel survives: the plateau value
        # plus whatever slice of the 0..0.10 ramp the crop happened to cover.
        for c in range(4):
            lo = float(base[c, 0, 0]) - 0.01
            hi = float(base[c, 0, 0]) + 0.11
            assert lo <= means[c] <= hi, \
                f"channel {c} level drifted to {means[c]:.4f}"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Determinism
# ═════════════════════════════════════════════════════════════════════════════

def test_test_split_is_deterministic_across_repeated_calls(tree):
    """
    ds[i] twice returns identical x and y on the test split.

    The whole reproducible-benchmark claim: "every run benchmarks the exact
    same noisy inputs". Without it, two evaluations of the same checkpoint
    disagree and no A/B comparison is meaningful.
    """
    ds = MobileHDRDataset(tree, split="test")
    for i in range(len(ds)):
        a, b = ds[i], ds[i]
        assert torch.equal(a["x"], b["x"]), f"x not reproducible at index {i}"
        assert torch.equal(a["y"], b["y"]), f"y not reproducible at index {i}"


def test_test_split_ignores_the_global_rng_state(tree):
    """
    Test-split noise is unaffected by the global torch/python RNG.

    Noise must come from the per-index generator only. If it leaked into the
    global stream, eval results would depend on how much randomness the
    training loop happened to consume first — the classic irreproducible
    benchmark.
    """
    ds = MobileHDRDataset(tree, split="test")
    torch.manual_seed(0)
    first = ds[0]["x"].clone()

    torch.manual_seed(999)
    torch.randn(10_000)
    random.seed(7)
    random.random()
    second = ds[0]["x"]

    assert torch.equal(first, second)


def test_test_split_is_identical_across_instances(tree):
    """
    Two separately constructed test datasets yield identical samples.

    DataLoader workers each hold their own dataset copy; per-instance state
    (a lazily filled range cache, say) must not perturb the noise.
    """
    a = MobileHDRDataset(tree, split="test")
    b = MobileHDRDataset(tree, split="test")
    for i in range(len(a)):
        assert torch.equal(a[i]["x"], b[i]["x"])


def test_different_test_noise_seed_changes_noise_but_not_target(tree):
    """
    A different test_noise_seed gives different x but the same y.

    test_noise_seed is the knob for "same frames, different noise draw" —
    it must move the noise realisation only. If it moved y too, the target
    would depend on a benchmark nuisance parameter.
    """
    a = MobileHDRDataset(tree, split="test", test_noise_seed=2025)
    b = MobileHDRDataset(tree, split="test", test_noise_seed=4242)
    for i in range(len(a)):
        sa, sb = a[i], b[i]
        assert not torch.equal(sa["x"], sb["x"]), f"seed had no effect at {i}"
        assert torch.equal(sa["y"], sb["y"]), f"seed changed the target at {i}"


def test_test_seed_is_the_base_seed_plus_the_index(tmp_path):
    """
    The per-index seed is exactly test_noise_seed + index.

    Pinned with two byte-identical frames: seed=S at index 1 must reproduce
    seed=S+1 at index 0. Without a per-index offset, every test frame would
    get the SAME noise realisation — an artificially easy, correlated
    benchmark that still looks deterministic.
    """
    img = _unit_range_image(h=32, w=32)
    root = tmp_path / "ds"
    _write_pt(root, "test/tensors/with_gt/a.pt", img)
    _write_pt(root, "test/tensors/with_gt/b.pt", img)   # identical content

    ds = MobileHDRDataset(str(root), split="test", test_noise_seed=1000)
    shifted = MobileHDRDataset(str(root), split="test", test_noise_seed=1001)

    # Identical frames + different indices => different noise.
    assert not torch.equal(ds[0]["x"], ds[1]["x"])
    # ...and the offset is exactly +1 per index.
    assert torch.equal(ds[1]["x"], shifted[0]["x"])


def test_train_split_is_not_deterministic_across_calls(tree):
    """
    The train split draws fresh noise (and crops) on every access.

    Virtual repeats and multi-epoch training rely on this; a deterministic
    train split would collapse num_patch=8 into 8 identical samples and
    remove the augmentation entirely.
    """
    ds = MobileHDRDataset(tree, split="train", crop_size=None, num_patch=1)
    xs = [ds[0]["x"] for _ in range(4)]
    assert not any(torch.equal(xs[0], other) for other in xs[1:]), \
        "train noise is frozen across calls"

    cropped = MobileHDRDataset(tree, split="train", crop_size=16, num_patch=1)
    cs = [cropped[0]["x"] for _ in range(4)]
    assert not any(torch.equal(cs[0], other) for other in cs[1:])


# ═════════════════════════════════════════════════════════════════════════════
# 7. num_patch virtual repeats
# ═════════════════════════════════════════════════════════════════════════════

def test_index_maps_to_file_modulo_file_count(tmp_path):
    """
    Index i reads file_list[i % len(file_list)], for every i in range(len).

    This is the virtual-epoch mapping. If it were i // num_patch (or
    anything else), a shuffled DataLoader would still look fine while
    visiting some frames far more often than others.
    """
    root = make_dataset(tmp_path / "ds", n_train=3, n_test=1, h=16, w=16)
    ds = MobileHDRDataset(root, split="train", num_patch=4, crop_size=None)
    seen = _spy_on_loads(ds)

    n = len(ds.file_list)
    assert len(ds) == n * 4
    for i in range(len(ds)):
        ds[i]
    assert seen == [ds.file_list[i % n] for i in range(len(ds))]


def test_repeat_indices_share_a_file_but_differ_in_content(tmp_path):
    """
    i and i + len(file_list) read the same frame yet return different data.

    That is the point of num_patch: same source, fresh crop and fresh noise.
    Identical content would mean the virtual epoch is just duplication.
    """
    root = make_dataset(tmp_path / "ds", n_train=2, n_test=1, h=64, w=64)
    ds = MobileHDRDataset(root, split="train", num_patch=3, crop_size=16)
    seen = _spy_on_loads(ds)
    n = len(ds.file_list)

    a = ds[0]
    b = ds[0 + n]
    c = ds[0 + 2 * n]
    assert seen == [ds.file_list[0]] * 3, "repeats did not hit the same file"
    assert not torch.equal(a["x"], b["x"])
    assert not torch.equal(b["x"], c["x"])
    for s in (a, b, c):
        assert_shape(s["x"], (4, 16, 16), name="x")


def test_indices_past_the_file_count_do_not_raise(tmp_path):
    """
    Every index in range(len(ds)) is readable, including the last one.

    len() and __getitem__ must agree, or a DataLoader crashes on the final
    batch of every epoch.
    """
    root = make_dataset(tmp_path / "ds", n_train=3, n_test=1, h=16, w=16)
    ds = MobileHDRDataset(root, split="train", num_patch=5, crop_size=8)
    assert len(ds) == 15
    last = ds[len(ds) - 1]
    assert_shape(last["y"], (4, 8, 8), name="y")


def test_negative_index_wraps_to_the_last_file(tmp_path):
    """
    ds[-1] reads the same frame as ds[len(ds) - 1].

    len(ds) is always a multiple of len(file_list), so Python's non-negative
    modulo makes negative indexing land correctly. Pinned because callers
    (and pytest-style ad hoc checks) do reach for ds[-1], and a silent
    off-by-one there would sample the wrong frame.
    """
    root = make_dataset(tmp_path / "ds", n_train=3, n_test=1, h=16, w=16)
    ds = MobileHDRDataset(root, split="train", num_patch=4, crop_size=8)
    seen = _spy_on_loads(ds)
    ds[-1]
    ds[len(ds) - 1]
    assert seen[0] == seen[1] == ds.file_list[-1]


# ═════════════════════════════════════════════════════════════════════════════
# 8. Per-file range cache
# ═════════════════════════════════════════════════════════════════════════════

def test_range_cache_starts_empty_and_fills_lazily(tmp_path):
    """
    _range_cache is empty at construction and gains one entry per file read.

    "Filled lazily per worker" — construction must not read every tensor,
    or spinning up 8 workers would scan the whole corpus 8 times before the
    first batch.
    """
    root = make_dataset(tmp_path / "ds", n_train=3, n_test=1, h=32, w=32)
    ds = MobileHDRDataset(root, split="train", num_patch=2, crop_size=8)
    assert ds._range_cache == {}

    ds[0]
    assert list(ds._range_cache) == [ds.file_list[0]]
    ds[1]
    assert set(ds._range_cache) == set(ds.file_list[:2])


def test_range_cache_stores_the_full_image_min_max_not_the_crop(tmp_path):
    """
    The cached (min, max) is the FULL image range even when a small dark
    crop was taken.

    Cache the crop's range instead and every patch gets contrast-stretched
    to full scale: absolute brightness — the one thing an HDR pipeline must
    preserve — is destroyed, silently.
    """
    img = _two_zone_image(h=64, w=64)
    root = tmp_path / "ds"
    path = _write_pt(root, "train/tensors/s/a.pt", img)

    ds = MobileHDRDataset(str(root), split="train", crop_size=16,
                          random_alpha=False)
    for _ in range(4):
        ds[0]

    assert list(ds._range_cache) == [path]
    lo, hi = ds._range_cache[path]
    assert lo == pytest.approx(float(img.min()), abs=1e-6)
    assert hi == pytest.approx(float(img.max()), abs=1e-6)
    assert (lo, hi) == pytest.approx((0.0, 1.0), abs=1e-6)


def test_one_cache_entry_per_file_across_many_reads(tmp_path):
    """
    Re-reading the same frame adds no new cache entries.

    The cache is keyed by path; a key that varied per access (index, say)
    would grow without bound in a persistent worker and re-scan the file
    every time, defeating the mmap story.
    """
    root = make_dataset(tmp_path / "ds", n_train=2, n_test=1, h=32, w=32)
    ds = MobileHDRDataset(root, split="train", num_patch=6, crop_size=8)
    for i in range(len(ds)):
        ds[i]
    assert len(ds._range_cache) == 2 == len(ds.file_list)


def test_cached_range_is_what_actually_drives_normalisation(tmp_path):
    """
    Mutating the cache entry changes the output proportionally, proving the
    cached value is consumed rather than recomputed per access.

    Doubling/quadrupling the cached max must scale y down by the same
    factor. If normalisation recomputed min/max from the (cropped) tensor,
    the injected range would be ignored and this test would not budge.
    """
    img = _unit_range_image(h=32, w=32)
    root = tmp_path / "ds"
    path = _write_pt(root, "test/tensors/with_gt/t.pt", img)
    ds = MobileHDRDataset(str(root), split="test")

    y_real = ds[0]["y"].clone()
    assert ds._range_cache[path] == pytest.approx((0.0, 1.0), abs=1e-6)

    ds._range_cache[path] = (0.0, 4.0)
    y_wide = ds[0]["y"]
    assert torch.allclose(y_wide, y_real / 4.0, atol=1e-4), \
        "the cached range was not used for normalisation"


def test_dark_crop_of_a_bright_image_keeps_its_absolute_brightness(tmp_path,
                                                                     monkeypatch):
    """
    A dark patch cut from a bright frame comes back DARK, not stretched.

    The headline claim: "crops keep their absolute brightness (a dark crop
    stays dark)". Here the source spans [0, 1] but the bottom half only
    spans [0.20, 0.24]. Full-image normalisation returns ~0.2; per-crop
    normalisation would return a patch spanning the whole [0, 1] — a 5x
    brightness error and a 25x contrast error, with the noise model then
    applied at completely the wrong signal level.
    """
    img = _two_zone_image(h=64, w=64, dark_lo=0.20, dark_hi=0.24)
    root = tmp_path / "ds"
    _write_pt(root, "train/tensors/s/a.pt", img)

    # Force the crop into the dark bottom-left region: top = 32, left = 0.
    monkeypatch.setattr(ds_mod, "random", _ScriptedRandom([32, 0]))

    ds = MobileHDRDataset(str(root), split="train", crop_size=32,
                          random_alpha=False)
    y = ds[0]["y"]

    expected = img[:, 32:64, 0:32]
    assert torch.allclose(y, expected, atol=1e-5), \
        "dark crop was not normalised by the full-image range"
    assert float(y.max()) < 0.30, f"dark crop was brightened to {float(y.max()):.3f}"
    assert float(y.min()) > 0.15
    assert float(y.max() - y.min()) < 0.10, "dark crop was contrast-stretched"


def test_degenerate_constant_image_yields_zero_target_without_nan(tmp_path):
    """
    A constant frame (zero range) yields an all-zero y and a finite, in-range
    x — the divide-by-zero guard, not a NaN.

    A fully saturated or fully black frame in the corpus must not put NaNs
    into the loss and kill a multi-hour run. Documenting the behaviour also
    makes it visible that such frames contribute a degenerate all-zero
    target rather than being skipped.
    """
    root = tmp_path / "ds"
    _write_pt(root, "test/tensors/with_gt/flat.pt", torch.full((4, 32, 32), 0.7))
    s = MobileHDRDataset(str(root), split="test")[0]

    assert_finite(s["x"], name="x")
    assert_finite(s["y"], name="y")
    assert torch.count_nonzero(s["y"]) == 0
    assert_in_range(s["x"], 0.0, 1.0, name="x")
    # Pure read noise on a zero signal: small but non-zero.
    assert float(s["x"].max()) > 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 9. Noise-parameter plumbing
# ═════════════════════════════════════════════════════════════════════════════

def test_noise_parameters_are_forwarded_to_add_photon_noise(tmp_path, monkeypatch):
    """
    nbits, shot_gain, read_noise_range, random_alpha and do_expand reach
    add_photon_noise unchanged on the train split.

    The module docstring's whole point is that the noise is strong
    (shot_gain=14, read_var ~U(135,160)); a dropped kwarg would silently
    restore the "~100x too weak" behaviour it warns about.
    """
    root = make_dataset(tmp_path / "ds", n_train=1, n_test=1, h=32, w=32)
    calls = _spy_on_noise(monkeypatch)

    ds = MobileHDRDataset(root, split="train", nbits=12, random_alpha=True,
                          do_expand=True, shot_gain=3.25,
                          read_noise_range=(11.0, 12.0))
    ds[0]
    _, kwargs = calls[0]
    assert kwargs["nbits"] == 12
    assert kwargs["shot_gain"] == 3.25
    assert kwargs["read_noise_range"] == (11.0, 12.0)
    assert kwargs["random_alpha"] is True
    assert kwargs["do_expand"] is True
    assert kwargs["generator"] is None, "train must use the global RNG stream"


def test_augmentation_is_disabled_on_the_test_split(tmp_path, monkeypatch):
    """
    On the test split random_alpha and do_expand are forced False and a
    seeded generator is supplied — even when the constructor was told True.

    Random dimming or a random saturation offset at eval time would make
    every reported metric a lottery. This pins the override, not just the
    resulting determinism.
    """
    root = make_dataset(tmp_path / "ds", n_train=1, n_test=2, h=32, w=32)
    calls = _spy_on_noise(monkeypatch)

    ds = MobileHDRDataset(root, split="test", random_alpha=True,
                          do_expand=True, test_noise_seed=555)
    ds[1]
    _, kwargs = calls[0]
    assert kwargs["random_alpha"] is False
    assert kwargs["do_expand"] is False
    assert isinstance(kwargs["generator"], torch.Generator)
    assert kwargs["generator"].initial_seed() == 555 + 1


# ═════════════════════════════════════════════════════════════════════════════
# 10. transform hook
# ═════════════════════════════════════════════════════════════════════════════

def test_transform_receives_noisy_and_gt_and_its_output_is_returned(tmp_path):
    """
    transform is called as transform(noisy, gt) and its two return values
    become x/xm and y.

    make_d4_transform returns a paired flip/transpose; if the arguments were
    swapped or the return ignored, the input and target would be augmented
    inconsistently and the model would learn to predict a mirrored image.
    """
    root = make_dataset(tmp_path / "ds", n_train=1, n_test=1, h=32, w=32)
    seen = {}

    def tf(noisy, gt):
        seen["noisy"] = noisy.clone()
        seen["gt"] = gt.clone()
        return torch.full_like(noisy, 0.25), torch.full_like(gt, 0.75)

    ref = MobileHDRDataset(root, split="test")[0]        # deterministic
    ds = MobileHDRDataset(root, split="test", transform=tf)
    s = ds[0]

    # The hook saw exactly the pre-transform pair.
    assert torch.equal(seen["noisy"], ref["x"])
    assert torch.equal(seen["gt"], ref["y"])
    # ...and its output is what came back, with xm still aliasing x.
    assert torch.allclose(s["x"], torch.full((4, 32, 32), 0.25))
    assert torch.allclose(s["y"], torch.full((4, 32, 32), 0.75))
    assert s["x"].data_ptr() == s["xm"].data_ptr()


def test_transform_may_change_the_spatial_size(tmp_path):
    """
    A transform that crops (the documented legacy path) is honoured.

    With crop_size=None, "transform may crop (legacy behaviour)" — the
    dataset must not re-assert the original shape afterwards.
    """
    root = make_dataset(tmp_path / "ds", n_train=1, n_test=1, h=32, w=32)

    def crop8(noisy, gt):
        return noisy[:, :8, :8], gt[:, :8, :8]

    s = MobileHDRDataset(root, split="train", crop_size=None,
                         transform=crop8)[0]
    for key in ("x", "xm", "y"):
        assert_shape(s[key], (4, 8, 8), name=key)


def test_no_transform_by_default(tmp_path):
    """
    transform defaults to None and the raw noise output is returned.

    The eval script passes transform=None explicitly ("no augmentation at
    test time"); a default augmentation would silently corrupt benchmarks.
    """
    root = make_dataset(tmp_path / "ds", n_train=1, n_test=1, h=32, w=32)
    ds = MobileHDRDataset(root, split="test")
    assert ds.transform is None
    assert_shape(ds[0]["x"], (4, 32, 32), name="x")


# ═════════════════════════════════════════════════════════════════════════════
# 11. Back-compat no-op API
# ═════════════════════════════════════════════════════════════════════════════

def test_regen_crops_and_regen_noise_are_callable_no_ops(tree):
    """
    regen_crops()/regen_noise() exist, return None, and change no state.

    Older training loops call them between epochs. They must stay present
    (an AttributeError would crash such a loop) and must stay inert: the
    deterministic test split has to survive a regen_noise() call unchanged.
    """
    ds = MobileHDRDataset(tree, split="test")
    before = ds[0]["x"].clone()
    keys_before = set(ds.__dict__)
    files_before = list(ds.file_list)

    assert ds.regen_crops() is None
    assert ds.regen_noise() is None

    assert set(ds.__dict__) == keys_before
    assert ds.file_list == files_before
    assert torch.equal(ds[0]["x"], before), "regen_noise() was not a no-op"


# ═════════════════════════════════════════════════════════════════════════════
# 12. DataLoader integration
# ═════════════════════════════════════════════════════════════════════════════

def test_works_in_a_dataloader_single_process(tmp_path):
    """
    Default collation over {x, xm, y} batches correctly with num_workers=0.

    Pins that the payload is a plain dict of same-shaped tensors, which is
    what lets the training script swap in its own collate_fn.
    """
    root = make_dataset(tmp_path / "ds", n_train=3, n_test=2, h=32, w=32)
    ds = MobileHDRDataset(root, split="train", num_patch=2, crop_size=16)
    loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0,
                        drop_last=False)

    total = 0
    for batch in loader:
        assert set(batch) == {"x", "xm", "y"}
        b = batch["x"].shape[0]
        for key in ("x", "xm", "y"):
            assert_shape(batch[key], (b, 4, 16, 16), name=key)
            assert_finite(batch[key], name=key)
            assert_in_range(batch[key], 0.0, 1.0, name=key)
        assert torch.equal(batch["x"], batch["xm"])
        total += b
    assert total == len(ds) == 6


def test_works_in_a_dataloader_with_two_workers(tmp_path):
    """
    The dataset survives multiprocessing: 2 workers, batch_size 2.

    The lazily-filled _range_cache and the mmap'd tensors both have to
    behave after a fork, and every virtual sample must be delivered exactly
    once. Training uses num_workers=8 with persistent_workers=True.
    """
    root = make_dataset(tmp_path / "ds", n_train=4, n_test=2, h=32, w=32)
    ds = MobileHDRDataset(root, split="train", num_patch=2, crop_size=16,
                          transform=_identity_transform)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)

    seen = 0
    for batch in loader:
        assert_shape(batch["x"], (2, 4, 16, 16), name="x")
        assert_finite(batch["y"], name="y")
        assert_in_range(batch["x"], 0.0, 1.0, name="x")
        seen += batch["x"].shape[0]
    assert seen == len(ds) == 8


def test_test_split_batches_are_worker_count_independent(tmp_path):
    """
    The test split yields byte-identical batches with 0 and 2 workers.

    Per-index seeding is what makes this true; a dataset that seeded per
    worker or from the global RNG would give different eval numbers
    depending on num_workers — reproducibility that quietly depends on a
    performance knob.
    """
    root = make_dataset(tmp_path / "ds", n_train=1, n_test=4, h=32, w=32)

    def collect(workers):
        ds = MobileHDRDataset(root, split="test")
        loader = DataLoader(ds, batch_size=2, shuffle=False,
                            num_workers=workers)
        return [(b["x"].clone(), b["y"].clone()) for b in loader]

    serial, parallel = collect(0), collect(2)
    assert len(serial) == len(parallel) == 2
    for (xa, ya), (xb, yb) in zip(serial, parallel):
        assert torch.equal(xa, xb)
        assert torch.equal(ya, yb)


def test_crops_differ_across_dataloader_workers(tmp_path):
    """
    Two workers reading the SAME frame produce different crops.

    Every file here is byte-identical, so any difference can only come from
    the crop offset. torch seeds each worker with base_seed + worker_id; if
    the dataset had cached an offset at construction (or drawn it from a
    fixed seed), all workers would emit the same patch sequence and
    num_patch would multiply the data volume without adding any diversity —
    invisible in the loss curve, fatal to generalisation.
    """
    root = tmp_path / "ds"
    img = _unit_range_image(h=64, w=64)
    for i in range(4):
        _write_pt(root, f"train/tensors/s/f{i}.pt", img)   # identical content
    ds = MobileHDRDataset(str(root), split="train", crop_size=16,
                          num_patch=2, random_alpha=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # y is noise-free here, so a digest of it identifies the crop offset.
    digests = [tuple(b["y"].flatten()[:6].tolist()) for b in loader]
    assert len(digests) == 8
    # Consecutive batches go to different workers (round-robin, batch_size=1).
    assert digests[0] != digests[1], "both workers drew the identical crop"
    assert digests[2] != digests[3]
    # Shared worker seeds would collapse 8 crops into 4 distinct ones.
    assert len(set(digests)) >= 6, \
        f"only {len(set(digests))} distinct crops across 8 reads"
