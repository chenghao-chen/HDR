"""
Process setup that both machines need and neither does by default.

Two things bite on ALCF hardware and are easy to forget:

*   **Thread explosions.** A Polaris login node reports 256 hardware threads
    and an Aurora node 208. OpenBLAS spawns one thread per reported core at
    import time, hits ``RLIMIT_NPROC`` and dies with
    ``blas_thread_init: pthread_create failed`` — a crash on ``import numpy``,
    not a warning. DataLoader workers inherit the setting, so the cap has to
    account for ``workers x threads`` staying under the core count.

*   **Reproducibility across backends.** Seeding CUDA and seeding XPU are
    different calls, and the DataLoader worker seed has to be derived rather
    than shared or every worker draws the same augmentation.

:func:`configure` does both and hands back the accelerator, so a script's
preamble is one call instead of fifteen lines.
"""

from __future__ import annotations

import os
import random
from typing import Optional

__all__ = [
    "cap_blas_threads", "seed_everything", "worker_init_fn",
    "make_generator", "configure", "RuntimeInfo",
]

# Every library that reads a thread count from the environment. NUMEXPR and
# VECLIB are included because pandas and Accelerate pull them in indirectly.
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def cap_blas_threads(threads: int = 4, force: bool = False) -> int:
    """
    Bound every BLAS backend's thread pool.

    Call this **before importing numpy or torch** for it to take effect —
    OpenBLAS reads the environment once, at load time. ``tests/conftest.py``
    and the submit scripts both do so; this function exists for scripts that
    have their own entry point.

    ``force`` overwrites a value the caller already set. The default only
    fills in what is missing, so ``OMP_NUM_THREADS=8 ./run_tests.sh`` still
    means eight.

    Returns the thread count now in effect.
    """
    if threads < 1:
        raise ValueError(f"threads must be >= 1, got {threads}")
    value = str(threads)
    for var in _THREAD_VARS:
        if force:
            os.environ[var] = value
        else:
            os.environ.setdefault(var, value)
    return int(os.environ["OMP_NUM_THREADS"])


def seed_everything(seed: int, accelerator=None) -> int:
    """
    Seed python, numpy and torch — including the accelerator's RNG.

    Returns the seed, so it can be logged in one line:
    ``print("seed", seed_everything(21))``.
    """
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass

    if accelerator is None:
        from .accelerator import get_accelerator
        accelerator = get_accelerator()
    accelerator.seed_all(seed)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """
    Per-worker seeding for ``DataLoader(worker_init_fn=...)``.

    torch already gives each worker a distinct ``initial_seed()``; what it
    does *not* do is propagate that to ``random`` or numpy, which is what the
    augmentation pipeline actually draws from. Without this, every worker
    applies the same flips to different images.
    """
    import torch

    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def make_generator(seed: int):
    """A seeded ``torch.Generator`` for ``DataLoader(generator=...)``."""
    import torch

    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


class RuntimeInfo:
    """What :func:`configure` set up, for logging."""

    def __init__(self, site, accelerator, seed: int, threads: int):
        self.site = site
        self.accelerator = accelerator
        self.seed = seed
        self.threads = threads

    @property
    def device(self):
        return self.accelerator.device

    def banner(self) -> str:
        """A four-line header worth having at the top of every job log."""
        from . import __version__
        lines = [
            f"hdr_platform {__version__}",
            f"site       : {self.site.name} — {self.site.description}",
            f"device     : {self.accelerator.summary()}",
            f"seed       : {self.seed}   blas threads: {self.threads}",
        ]
        return "\n".join(lines)


def configure(seed: Optional[int] = 21,
              device: Optional[str] = None,
              threads: Optional[int] = None,
              fast_matmul: bool = True) -> RuntimeInfo:
    """
    Set up a process to run this project on whichever machine it landed on.

    Picks the device, seeds every RNG, caps BLAS threads and enables the
    reduced-precision matmul path. Safe to call more than once.

    ``threads`` defaults to whatever the environment already carries (the
    submit scripts set it), falling back to 4 — the value that keeps
    8 workers x threads inside a Polaris node's core budget.
    """
    import torch

    from .accelerator import get_accelerator
    from .site import detect_site, get_site

    n_threads = cap_blas_threads(
        int(os.environ.get("OMP_NUM_THREADS", threads or 4))
        if threads is None else threads,
        force=threads is not None)
    torch.set_num_threads(n_threads)

    acc = get_accelerator(device)
    if fast_matmul:
        acc.enable_fast_matmul()

    used_seed = seed_everything(seed, acc) if seed is not None else -1
    return RuntimeInfo(get_site(detect_site()), acc, used_seed, n_threads)
