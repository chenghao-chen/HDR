"""
One interface over CUDA (Polaris), XPU (Aurora) and CPU.

The training and evaluation code was written against CUDA and says so in a
few dozen places: ``torch.device("cuda:0")``, ``torch.cuda.synchronize()``,
``torch.amp.autocast(device_type="cuda")``, ``fused=True`` on Adam,
``torch.backends.cuda.matmul.allow_tf32``. Each of those is either a hard
error or a silent no-op on Aurora's Intel GPUs. :class:`Accelerator` is the
single place that knows the difference.

    from hdr_platform import get_accelerator

    acc = get_accelerator()             # cuda:0, xpu:0 or cpu
    model = model.to(acc.device)
    with acc.autocast(torch.bfloat16):
        pred = model(x, snr)
    acc.synchronize()
    print(acc.peak_memory_gb())

Backend notes that cost time to rediscover:

*   **Importing IPEX.** Older ``frameworks`` modules on Aurora only wire up
    ``torch.xpu`` once ``intel_extension_for_pytorch`` has been imported;
    newer PyTorch has XPU support built in and needs no import. Asking for
    XPU therefore tries the import first and ignores its absence.

*   **``torch.xpu`` exists everywhere.** From PyTorch 2.5 on, the namespace
    is present even in a CUDA-only build and ``torch.xpu.is_available()``
    correctly returns False. Probing it is safe on Polaris.

*   **Pinned memory.** ``DataLoader(pin_memory=True)`` pins for CUDA unless
    told otherwise. XPU needs ``pin_memory_device="xpu"``; both are handled
    by :meth:`dataloader_kwargs`.

*   **Fused Adam** is a CUDA-only fast path. Passing ``fused=True`` on XPU
    raises, so :meth:`optimizer_kwargs` drops it.
"""

from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, Dict, Optional, Sequence

import torch

__all__ = [
    "Accelerator", "get_accelerator", "available_backends",
    "resolve_device", "IPEX_AVAILABLE",
]


# ─────────────────────────────────────────────────────────────────────────────
# Backend probing
# ─────────────────────────────────────────────────────────────────────────────
IPEX_AVAILABLE: Optional[bool] = None   # None until first probed


def _try_import_ipex() -> bool:
    """
    Import Intel Extension for PyTorch if it is installed.

    Returns True when IPEX is importable. Caches, because the import is slow
    (it loads the oneAPI runtime) and because failing twice is pointless.
    Never raises: on Polaris IPEX simply is not installed.
    """
    global IPEX_AVAILABLE
    if IPEX_AVAILABLE is not None:
        return IPEX_AVAILABLE
    try:
        import intel_extension_for_pytorch  # noqa: F401
        IPEX_AVAILABLE = True
    except Exception:
        # ImportError is the normal case; a broken oneAPI install can raise
        # OSError from the shared-library load, and that is not fatal either.
        IPEX_AVAILABLE = False
    return IPEX_AVAILABLE


def _cuda_ready() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _xpu_ready() -> bool:
    """
    True when an Intel GPU is usable from this interpreter.

    Checks before *and* after importing IPEX: native-XPU builds need no
    import, and paying for it would slow every Polaris job down.
    """
    if getattr(torch, "xpu", None) is None:
        return False
    try:
        if torch.xpu.is_available():
            return True
    except Exception:
        return False
    if _try_import_ipex():
        try:
            return bool(torch.xpu.is_available())
        except Exception:
            return False
    return False


def available_backends() -> Sequence[str]:
    """Backends this interpreter can actually reach, best first."""
    found = []
    if _cuda_ready():
        found.append("cuda")
    if _xpu_ready():
        found.append("xpu")
    found.append("cpu")
    return tuple(found)


def resolve_device(spec: Optional[str] = None) -> torch.device:
    """
    Turn a device string — or nothing at all — into a concrete device.

    ``spec`` may be ``"cuda:0"``, ``"xpu"``, ``"cpu"``, ``"auto"`` or None.
    ``auto``/None picks the first working accelerator, then CPU. The
    ``HDR_DEVICE`` environment variable is consulted before falling back to
    auto-detection, which is how a submit script pins a job to one tile.
    """
    if spec is None:
        spec = os.environ.get("HDR_DEVICE") or "auto"
    spec = str(spec).strip()

    if spec.lower() in ("", "auto"):
        best = available_backends()[0]
        return torch.device(f"{best}:0" if best != "cpu" else "cpu")

    device = torch.device(spec)
    # An explicit request for an absent backend is a mistake worth catching
    # here rather than 40 lines later inside a .to() call.
    if device.type == "cuda" and not _cuda_ready():
        raise RuntimeError(
            f"device {spec!r} was requested but no CUDA device is visible. "
            "On a Polaris login node this is expected — submit a job, or "
            "pass --device cpu.")
    if device.type == "xpu" and not _xpu_ready():
        raise RuntimeError(
            f"device {spec!r} was requested but no XPU device is visible. "
            "On Aurora, load the frameworks module first: "
            "`module use /soft/modulefiles && module load frameworks`.")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# The abstraction
# ─────────────────────────────────────────────────────────────────────────────
class Accelerator:
    """
    A device plus the backend-specific operations the project performs on it.

    Construct through :func:`get_accelerator` rather than directly; it caches
    per device string, and repeated construction re-probes IPEX.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device: torch.device = (device if isinstance(device, torch.device)
                                     else resolve_device(device))
        self.kind: str = self.device.type
        self._backend = self._backend_module(self.kind)

    # ── construction helpers ─────────────────────────────────────────────
    @staticmethod
    def _backend_module(kind: str):
        """``torch.cuda`` / ``torch.xpu`` / None, matching ``kind``."""
        if kind == "cuda":
            return torch.cuda
        if kind == "xpu":
            return getattr(torch, "xpu", None)
        return None

    def __repr__(self) -> str:
        return f"<Accelerator {self.device} ({self.name})>"

    def __str__(self) -> str:
        return str(self.device)

    # ── identity ─────────────────────────────────────────────────────────
    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "xpu")

    @property
    def index(self) -> int:
        """Ordinal of the selected device; 0 when unspecified."""
        return self.device.index if self.device.index is not None else 0

    @property
    def name(self) -> str:
        """Marketing name of the device, e.g. ``NVIDIA A100-SXM4-40GB``."""
        if self._backend is None:
            return "cpu"
        try:
            props = self._backend.get_device_properties(self.index)
            return getattr(props, "name", str(props))
        except Exception:
            try:
                return str(self._backend.get_device_name(self.index))
            except Exception:
                return self.kind

    def device_count(self) -> int:
        """Accelerators visible to this process (respects CUDA/ZE masks)."""
        if self._backend is None:
            return 0
        try:
            return int(self._backend.device_count())
        except Exception:
            return 0

    def total_memory_gb(self) -> float:
        """HBM on the selected device, in GB. 0.0 on CPU."""
        if self._backend is None:
            return 0.0
        try:
            props = self._backend.get_device_properties(self.index)
            return float(props.total_memory) / 1e9
        except Exception:
            return 0.0

    # ── the operations the training loop performs ────────────────────────
    def autocast(self, dtype: Optional[torch.dtype] = torch.bfloat16):
        """
        Mixed-precision context for this device, or a no-op.

        Autocast is skipped entirely on CPU: bf16 autocast there is legal
        but slow enough to distort a smoke test, and float16 on CPU is worse.
        Pass ``dtype=None`` to force full precision anywhere.
        """
        if dtype is None or not self.is_gpu:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=self.kind, dtype=dtype)

    def synchronize(self) -> None:
        """Block until queued work finishes. Required before any timing."""
        if self._backend is None:
            return
        try:
            self._backend.synchronize()
        except Exception:
            pass

    def empty_cache(self) -> None:
        """Return cached blocks to the driver. Use between phases, not in a loop."""
        if self._backend is None:
            return
        try:
            self._backend.empty_cache()
        except Exception:
            pass

    def reset_peak_memory(self) -> None:
        if self._backend is None:
            return
        try:
            self._backend.reset_peak_memory_stats()
        except Exception:
            pass

    def memory_gb(self) -> float:
        """Currently allocated tensor memory, in GB."""
        return self._memory_stat("memory_allocated")

    def peak_memory_gb(self) -> float:
        """High-water mark since the last :meth:`reset_peak_memory`, in GB."""
        return self._memory_stat("max_memory_allocated")

    def _memory_stat(self, attr: str) -> float:
        if self._backend is None:
            return 0.0
        fn = getattr(self._backend, attr, None)
        if fn is None:
            return 0.0
        try:
            return float(fn(self.index)) / 1e9
        except Exception:
            return 0.0

    def seed_all(self, seed: int) -> None:
        """Seed torch's host RNG and every device RNG on this backend."""
        torch.manual_seed(seed)
        if self._backend is None:
            return
        for attr in ("manual_seed_all", "manual_seed"):
            fn = getattr(self._backend, attr, None)
            if fn is None:
                continue
            try:
                fn(seed)
                return
            except Exception:
                continue

    # ── configuration the call sites used to hardcode ────────────────────
    def enable_fast_matmul(self) -> None:
        """
        Trade a little matmul precision for throughput.

        On CUDA that is TF32 on the tensor cores. Intel GPUs reach the same
        place through ``set_float32_matmul_precision("high")``, which maps
        onto TF32-equivalent hardware paths; there is no ``allow_tf32`` flag
        to set. Guarded because ``torch.backends.cuda`` is missing entirely
        in some CPU-only builds.
        """
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        if self.kind != "cuda":
            return
        for module, flag in ((getattr(torch.backends, "cuda", None), "matmul"),
                             (getattr(torch.backends, "cudnn", None), None)):
            if module is None:
                continue
            try:
                if flag == "matmul":
                    module.matmul.allow_tf32 = True
                else:
                    module.allow_tf32 = True
            except Exception:
                pass

    @property
    def supports_fused_adam(self) -> bool:
        """
        Whether ``torch.optim.Adam(..., fused=True)`` works here.

        The fused path is CUDA-only. On XPU it raises at construction, which
        would kill a job several minutes in, after the dataset has loaded.
        """
        return self.kind == "cuda"

    def optimizer_kwargs(self, **overrides: Any) -> Dict[str, Any]:
        """
        Optimizer keyword arguments that are legal on this backend.

        >>> Accelerator(torch.device("cpu")).optimizer_kwargs(lr=1e-4)
        {'lr': 0.0001, 'fused': False}
        """
        kwargs: Dict[str, Any] = {"fused": self.supports_fused_adam}
        kwargs.update(overrides)
        if kwargs.get("fused") and not self.supports_fused_adam:
            kwargs["fused"] = False
        return kwargs

    def dataloader_kwargs(self, num_workers: int = 0,
                          pin_memory: Optional[bool] = None,
                          **overrides: Any) -> Dict[str, Any]:
        """
        DataLoader arguments wired for this backend.

        Pinned host memory only helps when there is a device to copy to, and
        XPU needs to be *told* which device to pin for — the default target
        is CUDA, so a bare ``pin_memory=True`` on Aurora either warns or
        silently gives up the speedup.
        """
        pin = self.is_gpu if pin_memory is None else bool(pin_memory)
        kwargs: Dict[str, Any] = {
            "num_workers": num_workers,
            "pin_memory": pin,
        }
        if pin and self.kind == "xpu":
            kwargs["pin_memory_device"] = "xpu"
        if num_workers > 0:
            kwargs["persistent_workers"] = True
        kwargs.update(overrides)
        return kwargs

    def optimize_model(self, model, optimizer=None, dtype=None):
        """
        Apply IPEX graph optimisations on Aurora; a pass-through elsewhere.

        ``ipex.optimize`` fuses conv+bn, reorders weights into the layout the
        Xe cores want, and is where a good part of Intel GPU performance
        comes from. It is strictly optional: when IPEX is absent, or when it
        refuses a model it cannot handle, the unmodified model is returned
        and training proceeds a little slower.

        Returns ``model`` when ``optimizer`` is None, else ``(model, optimizer)``
        — matching ``ipex.optimize``'s own contract.
        """
        if self.kind != "xpu" or not _try_import_ipex():
            return model if optimizer is None else (model, optimizer)

        import intel_extension_for_pytorch as ipex
        try:
            if optimizer is None:
                return ipex.optimize(model, dtype=dtype)
            return ipex.optimize(model, optimizer=optimizer, dtype=dtype)
        except Exception as exc:                       # pragma: no cover
            warnings.warn(f"ipex.optimize failed ({exc!r}); "
                          "continuing with the unoptimised model.")
            return model if optimizer is None else (model, optimizer)

    # ── reporting ────────────────────────────────────────────────────────
    def summary(self) -> str:
        """One-line description for a job log header."""
        if not self.is_gpu:
            return f"cpu ({torch.get_num_threads()} torch threads)"
        return (f"{self.device} — {self.name}, "
                f"{self.total_memory_gb():.0f} GB, "
                f"{self.device_count()} visible")


# ─────────────────────────────────────────────────────────────────────────────
# Cached accessor
# ─────────────────────────────────────────────────────────────────────────────
_CACHE: Dict[str, Accelerator] = {}


def get_accelerator(device: Optional[str] = None) -> Accelerator:
    """
    The :class:`Accelerator` for ``device`` (default: auto-detect).

    Cached per resolved device string so that probing IPEX, which loads the
    oneAPI runtime, happens once per process.
    """
    resolved = resolve_device(device)
    key = str(resolved)
    if key not in _CACHE:
        _CACHE[key] = Accelerator(resolved)
    return _CACHE[key]
