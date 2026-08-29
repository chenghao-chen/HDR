# Running on Polaris and Aurora

Both machines are ALCF, both run PBS Pro, and almost nothing else about them
is the same. This is what differs and what to type.

| | **Polaris** | **Aurora** |
|---|---|---|
| GPU | 4x NVIDIA A100 | 6x Intel Data Center GPU Max 1550 |
| Backend | CUDA (`cuda:0`) | oneAPI / Level Zero (`xpu:0`) |
| Memory per device | 40 GB | 64 GB per tile (128 GB per GPU, 2 tiles) |
| Devices per node | 4 | 12 with `ZE_FLAT_DEVICE_HIERARCHY=FLAT` |
| CPU cores | 32 | 104 |
| Filesystem | `/lus/eagle` | `/lus/flare` |
| Torch from | miniforge env we build | the `frameworks` module |
| Long single-node job | `preemptable` only | `prod` |

Two consequences drive everything below:

1. **`torch.cuda` does not exist on Aurora.** `torch.cuda.is_available()`
   returns False on a node with six working GPUs, and
   `torch.amp.autocast(device_type="cuda")` raises. All of this is handled
   in `hdr_platform/`; no training or eval code names a backend any more.
2. **The two filesystems do not see each other.** Data staged on eagle is
   not readable from Aurora, at all. `scripts/stage_data.sh` moves it.

---

## First time on a machine

```bash
# Polaris
./scripts/setup/polaris_env.sh      # miniforge + CUDA torch, ~15 min
./run_tests.sh                      # CPU suite on the login node
./scripts/submit.sh smoke           # GPU smoke test, debug queue

# Aurora
./scripts/setup/aurora_env.sh       # venv on top of the frameworks module
./scripts/stage_data.sh             # eagle -> flare, ~12 GB
./run_tests.sh
./scripts/submit.sh smoke
```

Run the smoke job before anything long. It executes the full pytest suite
with the GPU tests active, then a real training step and a full-resolution
inference pass on the real data. Twenty minutes in the debug queue against
an eight-hour job that dies on its first kernel launch.

## Day to day

```bash
./scripts/submit.sh train                                  # 8 h
./scripts/submit.sh test                                   # evaluate
./scripts/submit.sh train -v HDR_SAVE_FOLDER=models_run2/   # named run
./scripts/submit.sh test -v HDR_CHECKPOINT=models_run2/phase1_best.pth
./scripts/submit.sh train --dry-run                         # show the qsub
```

`submit.sh` detects the machine and picks `scripts/<site>/<job>.pbs`.
Everything after the job name goes to `qsub` untouched, so `-q`, `-l
walltime=`, `-A` and `-v` all work as usual.

## Checking a node before trusting it

```bash
python -m hdr_platform.doctor          # 11 checks, human-readable
python -m hdr_platform.doctor --json   # for a log parser
python -m hdr_platform.site            # what machine is this, and where is everything
```

The submit scripts run the doctor as a preflight. It checks the interpreter,
the accelerator, that the accelerator can *compute* (a visible-but-broken GPU
is a real failure mode), the packages, the dataset, writable space and the
proxy — and it fails in ten seconds with a sentence rather than twenty
minutes in with a stack trace.

---

## Environment variables

| Variable | Effect |
|---|---|
| `HDR_SITE` | Force `polaris` / `aurora` / `local`. Overrides detection. |
| `HDR_PROJECT` | Allocation name. **Often different on the two machines.** |
| `HDR_PROJ_ROOT` | Full project root, overriding the name-based default. |
| `HDR_DATASET_DIR` | Path to one dataset. Wins over everything. |
| `HDR_PYTHON` | Interpreter, overriding the site default. |
| `HDR_DEVICE` | Pin to a device, e.g. `xpu:1`. Default: auto. |
| `HDR_SAVE_FOLDER` | Run directory. Pin it — see *preemption* below. |
| `HDR_CHECKPOINT` | Checkpoint for the eval job. |
| `HDR_THREADS` | BLAS threads per process. Default 4 (Polaris) / 8 (Aurora). |
| `WANDB_MODE` | `online` / `offline`. Auto-detected from credentials. |

The allocation name is the one most likely to bite: an Aurora award is a
separate thing from a Polaris/eagle award and frequently has a different
name. If `qsub` rejects the account or the paths come out wrong:

```bash
export HDR_PROJECT=your-aurora-allocation
# or, more bluntly:
export HDR_PROJ_ROOT=/lus/flare/projects/your-alloc/$USER
```

Then update `#PBS -A` in `scripts/aurora/*.pbs`.

---

## Preemption, and why the save folder is pinned

Polaris' `prod` and `small` queues both require **10 or more nodes**. A
single-node job can therefore only use `debug` (≤ 1 h), `debug-scaling`
(≤ 1 h), or `preemptable` (≤ 72 h). An eight-hour training run has no
option but `preemptable`, and preemptable jobs get killed and requeued.

So `scripts/polaris/train.pbs` sets:

* `#PBS -r y` — mark the job rerunnable so PBS requeues it,
* `HDR_SAVE_FOLDER` to a **fixed** name, not a timestamped one.

The training script resumes from `<folder>/latest.pth`. A fresh timestamp on
each attempt would mean the resume never finds a checkpoint and the run
silently restarts from epoch 0 — burning the allocation and never finishing.

Aurora's `prod` queue accepts single-node jobs, so none of this is forced
there. `scripts/aurora/train.pbs` keeps `-r y` and the pinned folder anyway,
because a node failure requeues a job on any queue.

---

## Moving data

```bash
./scripts/stage_data.sh --verify           # what is where
./scripts/stage_data.sh --dry-run          # show the commands
./scripts/stage_data.sh                    # stage onto this machine
./scripts/stage_data.sh --all              # every dataset, ~17 GB
```

Three methods, tried in this order:

* **local** — both paths visible here. A plain rsync.
* **globus** — the supported way between ALCF filesystems. Runs on the
  transfer nodes, restarts itself, checksums, survives your logout. Needs
  the collection IDs once:

  ```bash
  globus endpoint search 'ALCF Eagle'
  globus endpoint search 'ALCF Flare'
  export HDR_GLOBUS_EAGLE=<uuid> HDR_GLOBUS_FLARE=<uuid>
  ```

* **ssh** — rsync over ssh to the other login node. Works, holds your
  terminal for the duration.

Mobile-HDR is 246 files and 12 GB.

---

## Memory: the one real capability difference

`train_A100_MoE_two_phase.py` notes that "batch_sz=1 is the limit for full
frames on A100-80 GB". Polaris A100s have **40 GB**.

* **Phase 1** (512x512 patches, batch 8) fits on both machines.
* **Phase 2** (full-resolution fine-tune) will very likely OOM on Polaris
  even at batch_sz=1. An Aurora tile has 64 GB, so Phase 2 has a real chance
  there — **Aurora is the machine to try it on**.

Before committing to an eight-hour Phase 2, read the peak-memory line the
smoke job prints for the full-resolution pass:

```
full-res   : in=(1, 4, 1500, 2000) -> out=(1, 3, 3000, 4000) in 4.21s, peak memory 38.4 GB
```

If that is close to the device size, Phase 2 will not fit. The alternatives
are gradient checkpointing or tiled fine-tuning, both code changes.

---

## What is portable and what is not

**Checkpoints are portable.** They hold CPU-side tensors and the eval script
maps them onto whichever device is present. Train on Polaris, evaluate on
Aurora, or the reverse.

**Numbers will differ in the last decimal.** bf16 rounds differently on
tensor cores and on Xe cores. A PSNR that moves by 0.01 dB between machines
is the hardware, not a bug. A PSNR that moves by 1 dB is a bug.

**Nothing is multi-GPU.** Both training scripts use one device; the other 3
A100s (or 11 tiles) sit idle. Changing that means DDP — NCCL on Polaris,
oneCCL on Aurora — which is a code change, not a flag.

---

## Layout

```
hdr_platform/            the machine abstraction
  site.py                which machine, and where things are (no torch)
  accelerator.py         CUDA / XPU / CPU behind one interface
  runtime.py             thread caps, seeding, process setup
  doctor.py              python -m hdr_platform.doctor

scripts/
  submit.sh              site-detecting front end for qsub
  stage_data.sh          eagle <-> flare
  gpu_smoke.py           real train step + full-res eval, either backend
  lib/hdr_env.sh         shared shell bootstrap
  lib/site_polaris.sh    Polaris specifics
  lib/site_aurora.sh     Aurora specifics
  polaris/*.pbs          train / test / smoke
  aurora/*.pbs           train / test / smoke
  setup/polaris_env.sh   miniforge + CUDA torch
  setup/aurora_env.sh    venv over the frameworks module
```

The three `submit_*_polaris.sh` scripts in the repository root are the
originals this work generalised. They still run. The maintained versions are
under `scripts/polaris/`.

---

## Troubleshooting

**`torch.xpu.is_available()` is False on an Aurora compute node.** The
`frameworks` module did not load. Check `module avail frameworks` and set
`HDR_FRAMEWORKS_MODULE` to the exact name.

**A CPU-only torch shadows the module's.** Something pip-installed torch
into the venv. `scripts/setup/aurora_env.sh` checks for this and refuses to
finish; the fix is `$HDR_PYTHON -m pip uninstall -y torch torchvision`.

**`blas_thread_init: pthread_create failed`** at `import numpy`. The BLAS
thread cap is missing. `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`, or
go through `run_tests.sh` / the submit scripts, which set it.

**W&B hangs for minutes, then goes offline.** No proxy. Compute nodes reach
the internet only through `http://proxy.alcf.anl.gov:3128`; `hdr::init` sets
it. Authenticate on a login node with `wandb login`, or sync afterwards:
`$HDR_PYTHON -m wandb sync $WANDB_DIR/wandb/offline-run-*`.

**The job says the dataset is missing on Aurora.** eagle is not mounted
there. `./scripts/stage_data.sh`.

**A job resumed and started from epoch 0.** `HDR_SAVE_FOLDER` was
timestamped rather than pinned. See *preemption* above.
