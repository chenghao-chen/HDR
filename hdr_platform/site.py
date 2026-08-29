"""
Which machine are we on, and where does everything live on it.

The project runs on two ALCF systems that agree on the scheduler (PBS Pro)
and disagree on nearly everything else:

    Polaris   4x NVIDIA A100 40 GB per node, CUDA, /lus/eagle
    Aurora    6x Intel Data Center GPU Max 1550 per node (2 tiles each),
              XPU via oneAPI, /lus/flare

Nothing here imports torch, so it stays cheap and is safe to call from
argument parsing, from shell helpers (``python -m hdr_platform.site``) and
from tests on a login node.

Detection order, first hit wins:

    1. ``HDR_SITE``            explicit override, always respected
    2. hostname                login nodes carry the system name
    3. filesystem markers      /lus/flare vs /lus/eagle
    4. ``PBS_O_HOST``          compute nodes inherit the submitting host
    5. "local"                 anything else — a laptop, a CI container

Compute-node hostnames on both machines look like ``x4204c3s3b0n0``, so
step 2 deliberately does not try to read the system out of them; the
filesystem check in step 3 is what actually resolves a compute node.
"""

from __future__ import annotations

import os
import platform
import re
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional

__all__ = [
    "SiteSpec", "POLARIS", "AURORA", "LOCAL", "SITES",
    "detect_site", "get_site", "resolve_project_root", "resolve_dataset_dir",
]


# ─────────────────────────────────────────────────────────────────────────────
# Site descriptions
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SiteSpec:
    """
    Static facts about a machine.

    These are the numbers the submit scripts and the runtime need to agree
    on. They are deliberately *static*: the live query ("how many GPUs does
    this node actually expose") belongs to :mod:`hdr_platform.accelerator`,
    which needs torch. This is what we know before the job starts.

    Attributes
    ----------
    accelerator
        Backend the site's GPUs speak: ``cuda``, ``xpu`` or ``cpu``.
    gpus_per_node
        Physical accelerators visible to one node with the default device
        hierarchy. On Aurora this is 6 GPUs; each has 2 tiles, so a FLAT
        hierarchy (``ZE_FLAT_DEVICE_HIERARCHY=FLAT``) exposes 12 devices
        instead — see ``tiles_per_gpu``.
    gpu_memory_gb
        HBM per *device* as counted by ``gpus_per_node``. Aurora's Max 1550
        carries 128 GB per GPU, which is 64 GB per tile.
    cpus_per_node
        Physical cores, not hardware threads. Both machines run SMT, so the
        thread count is double this.
    filesystems
        The value for ``#PBS -l filesystems=``. A job that touches a
        filesystem it did not request can be killed mid-run.
    queue_*
        Queue names, and the smallest node count the production queue will
        accept. On Polaris ``prod`` requires >= 10 nodes, which is why the
        long single-node training job has to be preemptable.
    proxy
        Compute nodes have no direct outbound route on either machine. This
        is the HTTP proxy W&B and any package download must go through.
    """

    name: str
    description: str
    accelerator: str
    gpus_per_node: int
    tiles_per_gpu: int
    gpu_memory_gb: float
    cpus_per_node: int
    scheduler: str
    filesystems: str
    scratch_root: str
    default_project: str
    queue_debug: str
    queue_prod: str
    queue_long: str
    prod_min_nodes: int
    debug_max_walltime: str
    proxy: Optional[str]
    select_extra: str = ""
    module_setup: str = ""
    smi_command: str = ""
    notes: str = ""

    # ── derived ──────────────────────────────────────────────────────────
    @property
    def devices_per_node(self) -> int:
        """Accelerators a job sees when every tile is its own device."""
        return self.gpus_per_node * self.tiles_per_gpu

    @property
    def memory_per_device_gb(self) -> float:
        """HBM per device once tiles are counted separately."""
        return self.gpu_memory_gb / self.tiles_per_gpu

    @property
    def is_alcf(self) -> bool:
        return self.scheduler == "pbs"

    def select_line(self, nodes: int = 1) -> str:
        """The ``#PBS -l select=`` value for an N-node job here."""
        base = f"select={nodes}"
        return f"{base}:{self.select_extra}" if self.select_extra else base

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["devices_per_node"] = self.devices_per_node
        d["memory_per_device_gb"] = self.memory_per_device_gb
        return d


POLARIS = SiteSpec(
    name="polaris",
    description="ALCF Polaris — 4x NVIDIA A100 40 GB per node",
    accelerator="cuda",
    gpus_per_node=4,
    tiles_per_gpu=1,
    gpu_memory_gb=40.0,
    cpus_per_node=32,
    scheduler="pbs",
    filesystems="home:eagle",
    scratch_root="/lus/eagle/projects",
    default_project="lighthouse-purdue",
    queue_debug="debug",
    queue_prod="prod",
    # prod/small want >= 10 nodes, so a long 1-node job has to be preemptable.
    queue_long="preemptable",
    prod_min_nodes=10,
    debug_max_walltime="01:00:00",
    proxy="http://proxy.alcf.anl.gov:3128",
    select_extra="system=polaris",
    module_setup="module use /soft/modulefiles",
    smi_command="nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
    notes=("A100 40 GB, not the 80 GB the training script was written for: "
           "full-resolution Phase 2 is the part that runs out of memory."),
)

AURORA = SiteSpec(
    name="aurora",
    description="ALCF Aurora — 6x Intel Data Center GPU Max 1550 (2 tiles each) per node",
    accelerator="xpu",
    gpus_per_node=6,
    tiles_per_gpu=2,
    gpu_memory_gb=128.0,
    cpus_per_node=104,
    scheduler="pbs",
    filesystems="home:flare",
    scratch_root="/lus/flare/projects",
    default_project="lighthouse-purdue",
    queue_debug="debug",
    queue_prod="prod",
    queue_long="prod",
    prod_min_nodes=1,
    debug_max_walltime="01:00:00",
    proxy="http://proxy.alcf.anl.gov:3128",
    # Aurora's PBS does not take a system= selector the way Polaris does.
    select_extra="",
    module_setup="module use /soft/modulefiles",
    smi_command="xpu-smi discovery",
    notes=("64 GB of HBM per tile is more than a Polaris A100 has, so the "
           "full-resolution Phase 2 that OOMs on Polaris is expected to fit."),
)

LOCAL = SiteSpec(
    name="local",
    description="unrecognised host — laptop, workstation or container",
    accelerator="cpu",
    gpus_per_node=0,
    tiles_per_gpu=1,
    gpu_memory_gb=0.0,
    cpus_per_node=os.cpu_count() or 1,
    scheduler="none",
    filesystems="",
    scratch_root="",
    default_project="",
    queue_debug="",
    queue_prod="",
    queue_long="",
    prod_min_nodes=0,
    debug_max_walltime="",
    proxy=None,
    notes="No scheduler. Everything runs in the foreground.",
)

SITES: Dict[str, SiteSpec] = {s.name: s for s in (POLARIS, AURORA, LOCAL)}


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────
_HOSTNAME_PATTERNS = (
    (re.compile(r"^polaris", re.I), "polaris"),
    (re.compile(r"^aurora", re.I), "aurora"),
    # Aurora's user-access nodes are aurora-uan-XXXX; its older name shows up
    # in some module files and job output as "sunspot", which is a different
    # machine but the same software stack.
    (re.compile(r"^uan-", re.I), "aurora"),
)

# A filesystem that exists on exactly one of the two machines.
_FILESYSTEM_MARKERS = (
    ("/lus/flare", "aurora"),
    ("/lus/eagle", "polaris"),
)


def detect_site(hostname: Optional[str] = None,
                environ: Optional[Dict[str, str]] = None) -> str:
    """
    Name of the machine we are running on.

    Parameters are injectable so the tests can drive every branch without a
    real Polaris or Aurora node.

    Returns one of the keys of :data:`SITES`; never raises.
    """
    env = os.environ if environ is None else environ

    # 1. Explicit override. Anything unrecognised is an error worth surfacing,
    #    because silently falling back to "local" would run a training job on
    #    the CPU for eight hours.
    override = (env.get("HDR_SITE") or "").strip().lower()
    if override:
        if override not in SITES:
            raise ValueError(
                f"HDR_SITE={override!r} is not a known site. "
                f"Choose one of: {', '.join(sorted(SITES))}.")
        return override

    host = hostname if hostname is not None else platform.node()
    host = (host or "").strip().lower()

    # 2. Login nodes name their machine; compute nodes (x4204c3s3b0n0) do not.
    for pattern, name in _HOSTNAME_PATTERNS:
        if pattern.search(host):
            return name

    # 3. Filesystem markers — this is what identifies a compute node.
    for path, name in _FILESYSTEM_MARKERS:
        if os.path.isdir(path):
            return name

    # 4. A PBS job inherits the submitting host, which *is* a login node.
    submit_host = (env.get("PBS_O_HOST") or "").strip().lower()
    for pattern, name in _HOSTNAME_PATTERNS:
        if pattern.search(submit_host):
            return name

    return "local"


def get_site(name: Optional[str] = None) -> SiteSpec:
    """
    The :class:`SiteSpec` for ``name``, or for the detected machine.

    >>> get_site("aurora").accelerator
    'xpu'
    """
    if name is None:
        name = detect_site()
    key = name.strip().lower()
    if key not in SITES:
        raise ValueError(f"unknown site {name!r}; "
                         f"choose one of: {', '.join(sorted(SITES))}.")
    return SITES[key]


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
def resolve_project_root(site: Optional[SiteSpec] = None,
                         environ: Optional[Dict[str, str]] = None) -> str:
    """
    The per-user project directory holding ``datasets/``, ``envs/`` and
    ``projects/``.

    ``HDR_PROJ_ROOT`` overrides everything. Otherwise it is built from the
    site's scratch filesystem, the allocation name and ``$USER``:

        Polaris   /lus/eagle/projects/lighthouse-purdue/<user>
        Aurora    /lus/flare/projects/lighthouse-purdue/<user>

    The allocation is very often named differently on the two machines — a
    Polaris/eagle award and an Aurora award are separate things. Set
    ``HDR_PROJECT`` (or the full ``HDR_PROJ_ROOT``) when they differ.
    """
    env = os.environ if environ is None else environ

    explicit = env.get("HDR_PROJ_ROOT")
    if explicit:
        return explicit.rstrip("/")

    spec = site if site is not None else get_site(detect_site(environ=env))
    if not spec.scratch_root:
        # No shared filesystem to speak of: fall back to the checkout.
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    project = env.get("HDR_PROJECT") or spec.default_project
    user = env.get("USER") or env.get("LOGNAME") or "unknown"
    return f"{spec.scratch_root}/{project}/{user}"


def resolve_dataset_dir(name: str = "Mobile-HDR",
                        site: Optional[SiteSpec] = None,
                        environ: Optional[Dict[str, str]] = None) -> str:
    """
    Where dataset ``name`` lives on this machine.

    ``HDR_DATASET_DIR`` points straight at one dataset and wins outright —
    that is the variable every existing submit script already sets.
    ``HDR_DATASET_ROOT`` names the directory that *contains* the datasets,
    which is the useful override once there is more than one.
    """
    env = os.environ if environ is None else environ

    explicit = env.get("HDR_DATASET_DIR")
    if explicit:
        return explicit.rstrip("/")

    root = env.get("HDR_DATASET_ROOT")
    if not root:
        root = os.path.join(resolve_project_root(site=site, environ=env),
                            "datasets")
    return os.path.join(root.rstrip("/"), name)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — shell scripts read these to stay in step with the Python side
# ─────────────────────────────────────────────────────────────────────────────
def _main(argv=None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        prog="python -m hdr_platform.site",
        description="Report the detected ALCF site and its paths.")
    p.add_argument("--site", default=None,
                   help="Describe this site instead of detecting one.")
    p.add_argument("--field", default=None,
                   help="Print one field bare (for shell $(...) capture).")
    p.add_argument("--json", action="store_true",
                   help="Dump the whole spec as JSON.")
    args = p.parse_args(argv)

    spec = get_site(args.site)
    info = spec.to_dict()
    info["project_root"] = resolve_project_root(site=spec)
    info["dataset_dir"] = resolve_dataset_dir(site=spec)

    if args.field:
        if args.field not in info:
            p.error(f"no such field {args.field!r}; "
                    f"available: {', '.join(sorted(info))}")
        print(info[args.field])
        return 0

    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    width = max(len(k) for k in info)
    for key in sorted(info):
        print(f"{key:<{width}} : {info[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
