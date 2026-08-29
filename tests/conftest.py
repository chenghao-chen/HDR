"""
Shared pytest configuration for the HDR test suite.

IMPORTANT — thread caps are set BEFORE torch/numpy are imported.
Polaris login nodes report 256 cores; OpenBLAS then tries to spawn one thread
per core and dies at import time with
    "blas_thread_init: pthread_create failed ... RLIMIT_NPROC"
which is a hard crash, not a warning. The submit scripts export the same caps.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "4")

# The project modules live in the repo root, one level up from tests/.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

torch.set_num_threads(4)


# ─────────────────────────────────────────────────────────────────────────────
# Markers
# ─────────────────────────────────────────────────────────────────────────────
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "gpu: requires a GPU (CUDA on Polaris, XPU on Aurora)")
    config.addinivalue_line("markers", "dataset: requires the real Mobile-HDR dataset on disk")
    config.addinivalue_line("markers", "cuda: requires CUDA specifically, not just any GPU")


def _gpu_device():
    """
    The accelerator this host exposes, or None.

    Deliberately routed through hdr_platform rather than
    ``torch.cuda.is_available()`` — that call is False on an Aurora compute
    node with six working Intel GPUs, which would silently skip the entire
    GPU suite there and report a green run.
    """
    try:
        from hdr_platform import get_accelerator
    except Exception:
        return None
    try:
        acc = get_accelerator()
    except Exception:
        return None
    return acc if acc.is_gpu else None


def pytest_collection_modifyitems(config, items):
    """
    Auto-skip gpu/cuda/dataset tests when the resource is unavailable.

    Marker membership is read with get_closest_marker, not ``in
    item.keywords``. Keywords also contain parametrize ids, so a perfectly
    portable ``@pytest.mark.parametrize("kind", ["cuda", "xpu"])`` case would
    otherwise be skipped as "needs CUDA" — silently dropping exactly the
    tests that cover the Aurora port.
    """
    acc = _gpu_device()

    if acc is None:
        skip_gpu = pytest.mark.skip(reason="no GPU visible (login node)")
        for item in items:
            if item.get_closest_marker("gpu"):
                item.add_marker(skip_gpu)

    if acc is None or acc.kind != "cuda":
        found = "none" if acc is None else acc.kind
        skip_cuda = pytest.mark.skip(
            reason=f"test needs CUDA specifically; this host has {found}")
        for item in items:
            if item.get_closest_marker("cuda"):
                item.add_marker(skip_cuda)

    if not _real_dataset_dir():
        skip_ds = pytest.mark.skip(reason="real Mobile-HDR dataset not found")
        for item in items:
            if item.get_closest_marker("dataset"):
                item.add_marker(skip_ds)


def _real_dataset_dir():
    """
    Path to the real Mobile-HDR dataset, or None when it is not present.

    The site-resolved path covers Polaris (eagle) and Aurora (flare) without
    hardcoding either; the literal below stays as a last resort so the suite
    still finds the data if hdr_platform is broken.
    """
    candidates = [os.environ.get("HDR_DATASET_DIR")]
    try:
        from hdr_platform import resolve_dataset_dir
        candidates.append(resolve_dataset_dir("Mobile-HDR"))
    except Exception:
        pass
    candidates.append(
        "/lus/eagle/projects/lighthouse-purdue/ryanchen/datasets/Mobile-HDR")

    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "train", "tensors")):
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def deterministic_seed():
    """Every test starts from the same RNG state (torch + python random)."""
    import random
    torch.manual_seed(1234)
    random.seed(1234)
    yield


@pytest.fixture
def device():
    """CPU device — the default for tests, which must run on a login node."""
    return torch.device("cpu")


@pytest.fixture
def gpu_device():
    """
    This host's accelerator. Skips when there is none, so a test using it
    does not also need the ``gpu`` marker to be safe (though it should carry
    the marker anyway, for ``-m 'not gpu'`` selection).
    """
    acc = _gpu_device()
    if acc is None:
        pytest.skip("no GPU visible")
    return acc.device


@pytest.fixture
def accelerator():
    """The full :class:`hdr_platform.Accelerator` for this host, GPU or CPU."""
    from hdr_platform import get_accelerator
    return get_accelerator()


@pytest.fixture
def tiny_kwargs():
    """
    Smallest architecture that still exercises every code path:
      dim=8  -> latent 8*16=128 channels, dim//2=4, ExpertHead r=4.
      Three PixelUnshuffle(2) stages need packed H, W divisible by 8.
    """
    return {
        "dim": 8,
        "num_blocks": [1, 1, 1, 1],
        "num_refinement_blocks": 1,
        "heads": [1, 1, 1, 1],
        "se_reduction": 8,
    }


@pytest.fixture
def real_dataset_dir():
    """The on-disk Mobile-HDR root; tests using it must be marked `dataset`."""
    d = _real_dataset_dir()
    if d is None:
        pytest.skip("real Mobile-HDR dataset not found")
    return d
