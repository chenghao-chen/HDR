"""
Machine abstraction for running this project on Polaris and Aurora.

The rest of the codebase targets CUDA. Polaris is CUDA; Aurora is Intel XPU,
where every ``torch.cuda`` call is either an exception or a silent no-op.
This package is the seam between the two, plus the site knowledge (paths,
queues, filesystems, proxy) that the submit scripts and the Python code have
to agree on.

    from hdr_platform import configure

    rt = configure(seed=21)             # device, seeds, thread caps
    print(rt.banner())
    model = model.to(rt.device)

Everything is imported lazily. ``import hdr_platform`` pulls in neither
torch nor the site tables, which matters because the shell helpers call
``python -m hdr_platform.site`` in the preflight of every job and a torch
import costs several seconds there.

Modules
-------
``site``          which machine, and where things live on it (no torch)
``accelerator``   CUDA / XPU / CPU behind one interface
``runtime``       thread caps, seeding, process setup
``doctor``        ``python -m hdr_platform.doctor`` — is this node usable?
"""

from __future__ import annotations

__version__ = "1.0.0"

# name -> submodule it lives in. Lazy so that `python -m hdr_platform.site`
# does not import the package eagerly and then re-execute the module under
# runpy, which emits a RuntimeWarning onto every job log's stderr.
_EXPORTS = {
    "SiteSpec": "site",
    "POLARIS": "site",
    "AURORA": "site",
    "LOCAL": "site",
    "SITES": "site",
    "detect_site": "site",
    "get_site": "site",
    "resolve_project_root": "site",
    "resolve_dataset_dir": "site",

    "Accelerator": "accelerator",
    "get_accelerator": "accelerator",
    "available_backends": "accelerator",
    "resolve_device": "accelerator",

    "configure": "runtime",
    "seed_everything": "runtime",
    "worker_init_fn": "runtime",
    "make_generator": "runtime",
    "cap_blas_threads": "runtime",
    "RuntimeInfo": "runtime",
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str):
    """PEP 562 lazy attribute access — see ``_EXPORTS``."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return __all__
