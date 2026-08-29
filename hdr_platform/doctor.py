"""
Is this node actually able to run the job?

    python -m hdr_platform.doctor            # human-readable
    python -m hdr_platform.doctor --json     # for a log parser
    python -m hdr_platform.doctor --strict   # non-zero exit on any warning

Every submit script runs this before starting real work. The point is to
fail in ten seconds with a sentence naming the problem, rather than twenty
minutes in — after the dataset has loaded and the queue time has been spent
— with a stack trace from inside a ``.to(device)`` call.

Each check returns ``ok`` / ``warn`` / ``fail``:

    ok     usable
    warn   usable, but something will be slower or offline than intended
    fail   the job cannot run — the script should exit here
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional

# Keep the import cheap; torch is loaded inside the checks that need it.
from .site import (detect_site, get_site, resolve_project_root,
                   resolve_dataset_dir)

__all__ = ["Check", "run_checks", "main"]

OK, WARN, FAIL = "ok", "warn", "fail"

_SYMBOL = {OK: "PASS", WARN: "WARN", FAIL: "FAIL"}


@dataclass
class Check:
    """One diagnostic and its outcome."""
    name: str
    status: str
    detail: str
    hint: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────
def check_site() -> Check:
    name = detect_site()
    spec = get_site(name)
    if name == "local":
        return Check("site", WARN, "unrecognised host — treating as local",
                     "Set HDR_SITE=polaris or HDR_SITE=aurora if this is wrong.")
    return Check("site", OK, f"{spec.name} — {spec.description}")


def check_python() -> Check:
    v = sys.version_info
    detail = f"{platform.python_version()} at {sys.executable}"
    if v < (3, 9):
        return Check("python", FAIL, detail,
                     "This project uses 3.9+ syntax. Load a newer interpreter.")
    return Check("python", OK, detail)


def check_torch() -> Check:
    try:
        import torch
    except Exception as exc:
        return Check("torch", FAIL, f"import failed: {exc!r}",
                     "On Aurora: module use /soft/modulefiles && module load frameworks. "
                     "On Polaris: use the project's miniforge env.")
    return Check("torch", OK, f"{torch.__version__} ({torch.__file__})")


def check_accelerator() -> Check:
    """The one that decides whether this is a GPU job or an eight-hour CPU job."""
    try:
        from .accelerator import get_accelerator, available_backends
    except Exception as exc:
        return Check("accelerator", FAIL, f"probe failed: {exc!r}")

    backends = available_backends()
    try:
        acc = get_accelerator()
    except Exception as exc:
        return Check("accelerator", FAIL, f"resolve failed: {exc!r}")

    site = get_site(detect_site())
    if not acc.is_gpu:
        expected = site.accelerator
        if expected == "cpu":
            return Check("accelerator", OK, "cpu (no GPU expected here)")
        hint = ("This is a login node — GPU work belongs in a job. "
                if site.is_alcf else "")
        if site.name == "aurora":
            hint += ("Inside a job, check that the frameworks module is "
                     "loaded and that ZE_AFFINITY_MASK is not empty.")
        elif site.name == "polaris":
            hint += "Inside a job, check nvidia-smi and the CUDA driver."
        return Check("accelerator", WARN,
                     f"cpu only; {site.name} expects {expected}", hint)

    if acc.kind != site.accelerator and site.accelerator != "cpu":
        return Check("accelerator", WARN,
                     f"found {acc.kind}, but {site.name} normally uses "
                     f"{site.accelerator}",
                     "Check HDR_SITE / HDR_DEVICE.")

    return Check("accelerator", OK,
                 f"{acc.summary()}  [backends: {', '.join(backends)}]")


def check_device_math() -> Check:
    """
    Prove the device computes, not merely that it enumerates.

    A visible-but-broken GPU is a real failure mode: a driver mismatch on
    CUDA, or an unset ``ZE_AFFINITY_MASK`` on XPU, both let
    ``device_count()`` succeed and the first real kernel fail.
    """
    try:
        import torch
        from .accelerator import get_accelerator
        acc = get_accelerator()
    except Exception as exc:
        return Check("device math", FAIL, f"setup failed: {exc!r}")

    if not acc.is_gpu:
        return Check("device math", OK, "skipped (cpu)")

    try:
        a = torch.randn(256, 256, device=acc.device)
        b = torch.randn(256, 256, device=acc.device)
        with acc.autocast(torch.bfloat16):
            c = a @ b
        acc.synchronize()
        if not torch.isfinite(c.float()).all():
            return Check("device math", FAIL,
                         "bf16 matmul produced non-finite values",
                         "Suspect a driver or runtime mismatch.")
        return Check("device math", OK,
                     f"256x256 bf16 matmul ok; "
                     f"{acc.peak_memory_gb() * 1e3:.0f} MB peak")
    except Exception as exc:
        return Check("device math", FAIL, f"kernel launch failed: {exc!r}",
                     "The device enumerates but cannot run work.")


def check_packages() -> Check:
    """Third-party imports the training and eval paths need."""
    required = ["numpy", "torchvision"]
    optional = ["lpips", "wandb", "cv2", "torchinfo", "PIL"]

    missing_required, missing_optional = [], []
    for mod in required:
        try:
            importlib.import_module(mod)
        except Exception:
            missing_required.append(mod)
    for mod in optional:
        try:
            importlib.import_module(mod)
        except Exception:
            missing_optional.append(mod)

    if missing_required:
        return Check("packages", FAIL,
                     f"missing: {', '.join(missing_required)}",
                     "pip install -r requirements.txt")
    if missing_optional:
        return Check("packages", WARN,
                     f"optional missing: {', '.join(missing_optional)}",
                     "lpips is needed for the perceptual loss term; "
                     "wandb for logging; cv2 for visual output.")
    return Check("packages", OK, "all present")


def check_project_modules() -> Check:
    """The project's own packages, which need the repo root on sys.path."""
    mods = ["hdr_data", "hdr_eval", "hdr_baselines"]
    missing = []
    for mod in mods:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            missing.append(f"{mod} ({type(exc).__name__})")
    if missing:
        return Check("project modules", FAIL, f"import failed: {', '.join(missing)}",
                     "Run from the repository root, or set PYTHONPATH to it.")
    return Check("project modules", OK, ", ".join(mods))


def check_dataset() -> Check:
    """The dataset the job is about to read, at the path it will read it from."""
    path = resolve_dataset_dir()
    if not os.path.isdir(path):
        return Check("dataset", FAIL, f"not a directory: {path}",
                     "Set HDR_DATASET_DIR, or stage the data with "
                     "scripts/stage_data.sh.")

    needed = [os.path.join("train", "tensors"),
              os.path.join("test", "tensors", "with_gt")]
    missing = [d for d in needed if not os.path.isdir(os.path.join(path, d))]
    if missing:
        return Check("dataset", FAIL,
                     f"{path} is missing {', '.join(missing)}",
                     "The copy is incomplete — re-run scripts/stage_data.sh.")

    n_train = len(os.listdir(os.path.join(path, "train", "tensors")))
    return Check("dataset", OK, f"{path} ({n_train} train entries)")


def check_writable() -> Check:
    """Somewhere to put checkpoints, which are large and must not go to /home."""
    root = resolve_project_root()
    target = os.environ.get("HDR_SAVE_FOLDER") or os.getcwd()
    if not os.path.isdir(target):
        parent = os.path.dirname(os.path.abspath(target)) or "."
        target = parent
    if not os.access(target, os.W_OK):
        return Check("writable", FAIL, f"cannot write to {target}",
                     "Checkpoints have nowhere to go.")

    free_gb = shutil.disk_usage(target).free / 1e9
    if free_gb < 20:
        return Check("writable", WARN,
                     f"{target} writable, only {free_gb:.0f} GB free",
                     "Checkpoints are hundreds of MB each; "
                     f"consider a run directory under {root}.")
    return Check("writable", OK, f"{target} ({free_gb:.0f} GB free)")


def check_network() -> Check:
    """
    Whether W&B will reach the internet from here.

    Compute nodes on both machines route outbound traffic through the ALCF
    proxy and have no direct path. Missing proxy settings do not fail a job;
    they make W&B block on a connect timeout and then fall back to offline,
    which is worth a warning at the top of the log rather than a surprise
    twenty minutes in.
    """
    site = get_site(detect_site())
    if site.proxy is None:
        return Check("network", OK, "no proxy expected here")

    proxied = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    mode = os.environ.get("WANDB_MODE", "")

    if mode == "offline" or os.environ.get("WANDB_DISABLED") == "true":
        return Check("network", OK, "W&B offline — proxy not needed")
    if not proxied:
        return Check("network", WARN, "no HTTPS_PROXY set",
                     f"Compute nodes need {site.proxy}; "
                     "W&B will otherwise hang, then go offline.")
    return Check("network", OK, f"proxy {proxied}, WANDB_MODE={mode or 'unset'}")


def check_threads() -> Check:
    """
    The BLAS thread cap. Its absence is a crash on ``import numpy``, not a
    slowdown: OpenBLAS spawns one thread per reported core and hits
    RLIMIT_NPROC on a 256-thread login node.
    """
    omp = os.environ.get("OMP_NUM_THREADS")
    cores = os.cpu_count() or 1
    if omp is None:
        return Check("threads", WARN,
                     f"OMP_NUM_THREADS unset on a {cores}-thread host",
                     "Export OMP_NUM_THREADS/OPENBLAS_NUM_THREADS (4-8). "
                     "Without it OpenBLAS can die at import.")
    return Check("threads", OK, f"OMP_NUM_THREADS={omp} of {cores} threads")


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
_CHECKS: List[Callable[[], Check]] = [
    check_site,
    check_python,
    check_torch,
    check_threads,
    check_accelerator,
    check_device_math,
    check_packages,
    check_project_modules,
    check_dataset,
    check_writable,
    check_network,
]


def run_checks(only: Optional[List[str]] = None,
               skip: Optional[List[str]] = None) -> List[Check]:
    """
    Run the diagnostics and collect their results.

    A check that raises is reported as a failure rather than aborting the
    run — the remaining checks usually explain why it raised.
    """
    results = []
    for fn in _CHECKS:
        label = fn.__name__.replace("check_", "").replace("_", " ")
        if only and label not in only:
            continue
        if skip and label in skip:
            continue
        try:
            results.append(fn())
        except Exception as exc:                       # pragma: no cover
            results.append(Check(label, FAIL, f"check itself raised: {exc!r}"))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m hdr_platform.doctor",
        description="Check that this node can run the HDR project.")
    p.add_argument("--json", action="store_true", help="Machine-readable output.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on warnings too, not just failures.")
    p.add_argument("--skip", default="",
                   help="Comma-separated check names to skip, e.g. dataset,network.")
    p.add_argument("--only", default="",
                   help="Comma-separated check names to run exclusively.")
    args = p.parse_args(argv)

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    results = run_checks(only=split(args.only), skip=split(args.skip))

    if args.json:
        print(json.dumps([asdict(c) for c in results], indent=2))
    else:
        width = max((len(c.name) for c in results), default=10)
        print("=" * 70)
        print("HDR environment check")
        print("=" * 70)
        for c in results:
            print(f"[{_SYMBOL[c.status]}] {c.name:<{width}}  {c.detail}")
            if c.hint and c.status != OK:
                print(f"       {'':<{width}}  -> {c.hint}")
        print("-" * 70)

    n_fail = sum(c.status == FAIL for c in results)
    n_warn = sum(c.status == WARN for c in results)
    if not args.json:
        print(f"{len(results)} checks: {len(results) - n_fail - n_warn} ok, "
              f"{n_warn} warning(s), {n_fail} failure(s)")

    if n_fail:
        return 1
    if n_warn and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
