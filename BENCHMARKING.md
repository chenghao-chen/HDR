# Evaluation harness, baselines and data pipeline

Three packages sit beside the existing training and model code. They are
additive: nothing in `train_A100_MoE_two_phase.py`,
`HDR_model_hybrid_Teacher.py`, `HDR_Mobile_dataset.py`,
`blocks_Restormer.py` or `DifferentiableGBTF_BGGR.py` was changed.

| Package | What it is |
|---|---|
| `hdr_data/` | CFA algebra, a configurable sensor noise model, and three datasets on one base class |
| `hdr_baselines/` | Classical and learned comparison models, all with the project's `(x, snr) -> (blended, experts, gates)` signature |
| `hdr_eval/` | Metrics, region stratification, tiled inference, the benchmark loop, reports, figures, and a CLI |

## Quick start

```bash
# what is available
python -m hdr_eval.cli --list

# a checkpoint against the classical lineup, five test images
python -m hdr_eval.cli \
    --dataset mobile_hdr --split test \
    --checkpoint models_p1_moe_.../phase1_best.pth \
    --baselines gbtf,wavelet+gbtf,malvar \
    --limit 5 --out test_results/compare

# how a model degrades as the noise gets worse
for n in low medium high extreme; do
  python -m hdr_eval.cli --dataset mobile_hdr --noise $n \
      --checkpoint best.pth --baselines gbtf --out results/$n
done
```

Each run writes `<model>.csv` (one row per image), `<model>.json`
(aggregate + records + run configuration), `report.md` (summary, deltas
against a reference, per-image win/loss, worst frames) and, with
`--save-visuals`, a `visuals/` directory.

From Python:

```python
from hdr_data import build_dataset, D4Transform
from hdr_baselines import build_baseline
from hdr_eval import BenchmarkRunner, RunConfig, build_report

ds = build_dataset("mobile_hdr", split="test", noise="high", crop_size=512)
res = BenchmarkRunner(build_baseline("wavelet+gbtf"), ds,
                      RunConfig(limit=4), name="wavelet+gbtf").run()
print(build_report([res], markdown=False))
```

## `hdr_data`

**`bayer`** — packing, mosaicking, phase-aware symmetries and RGB↔CFA
conversion for all four Bayer phases, replacing the BGGR assumptions
scattered through the original code.

**`noise`** — `NoiseModel` is **bit-exact with
`HDR_Mobile_dataset.add_photon_noise`** on its legacy preset (asserted in
`tests/test_hdr_data_noise.py`), so existing checkpoints are still
evaluated on the data they were trained for. On top of that it adds
per-channel quantum efficiency, PRNU, row/column readout banding and hot
pixels, all off by default, plus graded presets (`low` … `extreme`) and
two `realistic` ones. `calibrate_from_pairs` runs the model backwards,
recovering `(shot_gain, read_var)` from clean/noisy pairs by least
squares on binned variance-versus-signal — it recovers the parameters it
was given to within 0.3% and 9% respectively.

**Datasets** — `MobileHDRPacked` (bit-identical to the original class on
the test split), `I2VideoDataset` (the mp4 corpus through a virtual
sensor) and `KalantariDataset` (bracketed multi-exposure scenes), sharing
one base class that handles crop-before-noise, full-frame range
normalisation, deterministic test noise and augmentation.

**Augmentation** — `D4Transform` offers three modes. `make_d4_transform_compat`
is bit-identical to the training script's transform; `mode="cell"` is the
new default and preserves the CFA phase; `mode="pixel"` performs a true
mosaic symmetry and *reports* the resulting phase. See "The CFA phase
question" below.

## `hdr_baselines`

Classical demosaicing (`nearest`, `bilinear`, `malvar`, `gbtf`), six
denoisers (`gaussian`, `median`, `bilateral`, `guided`, `nlm`, `wavelet`),
either order of composition, an `vst_` prefix for generalised-Anscombe
variance stabilisation, and four trainable reference architectures
(`dncnn`, `unet`, `demosaicnet`, `restormer_lite`).

Names compose: `wavelet+gbtf`, `wavelet+gbtf_post` (denoise after
demosaicing), `vst_guided+malvar`.

Every denoiser exposes `for_noise_sigma()` and every pipeline defaults to
`auto_sigma=True`, re-parameterising per image from
`estimate_noise_sigma` (robust MAD on the finest Haar HH band). This
matters for fairness: with one constant tuned at one noise level the
classical baselines look far worse than they are, which would flatter the
learned model for no good reason.

## `hdr_eval`

`metrics` — PSNR and SSIM **agreeing with `test_dual_MoE_two_phase.py` to
floating point**, plus MS-SSIM, a tiled LPIPS wrapper, and CIEDE2000.
The colour metric earns its place: a red-to-green swap and a small
luminance change score the same PSNR (7.78 dB) and dE00 of 77.8 versus
2.8.

`regions` — per-luminance-band, per-SNR-band, edge and saturation
stratification. A whole-frame PSNR on HDR is a highlight metric; these
are what show a model getting worse in the shadows while gaining overall.
Empty bands report NaN rather than a fabricated number, and the
aggregation skips NaN.

`inference` — full-frame or tiled. The tiled path uses a tapered
(Tukey) blend window, disabled on sides touching the image boundary, and
computes tile positions so the last tile lands flush with the far edge.
It reproduces full-frame inference to ~1e-7 for a local model, at every
tile size and overlap tested. Autocast is CUDA-only, so the same code
runs on a login node.

`runner`, `report`, `visualize`, `cli`, `checkpoints` — the loop, the
tables, the figures without matplotlib, the command line, and a
checkpoint loader that reads the stored `num_experts`.

## Two things to know about the numbers

**Ground truth is demosaiced.** The datasets emit clean *CFA*, so the RGB
reference is produced by demosaicing it — GBTF by default, matching the
existing test script. No model can beat that reference by more than the
reference demosaicer's own accuracy on noise-free input, and a model that
demosaics differently but equally well is penalised. `--gt-demosaic`
changes it, and the choice is recorded in every run's metadata. It also
explains a result you will see immediately: the `gbtf` baseline reports a
gain of exactly +0.00 dB, because it is the reference algorithm applied to
the noisy input.

**The CFA phase question.** `make_d4_transform` performs a true mosaic
flip — which is what its tests assert, and they are right. But a true
horizontal flip of a BGGR mosaic produces a **GBRG** one. The transform
does not track that, so on roughly half of augmented training samples the
model receives GBRG data labelled BGGR, and the matching ground truth is
demosaiced by the BGGR-only GBTF, swapping the red and blue assignments.
`hdr_data.bayer.pattern_after` makes the phase change explicit and
`tests/test_hdr_data_augment.py` demonstrates that pixel mode reaches all
four phases. `mode="cell"` gives all eight D4 symmetries with the phase
preserved and is the recommended default for new training runs. Nothing
in the existing training script was changed — this is a finding, not a
fix.

## Tests

```bash
./run_tests.sh                              # everything
./run_tests.sh tests/test_hdr_data_noise.py # one file
./run_tests.sh -k hdr_eval                  # the eval harness
./run_tests.sh -m "not slow"                # skip the slow ones
```

Markers: `slow`, `gpu` (auto-skipped without an accelerator — CUDA on
Polaris, XPU on Aurora), `cuda` (CUDA specifically), `dataset`
(auto-skipped without the real Mobile-HDR corpus on disk).

Running on either machine, and the Polaris/Aurora differences, are covered
in [RUNNING.md](RUNNING.md).

## Environment notes

* **Kalantari imagery is not present.** `datasets/kalantari2017` holds 74
  train and 15 test scene directories, all empty. The loader is written to
  the layout `convert_to_tfrecord.py` reads and is tested against
  synthetic scenes; it has never been run against the real files.
* **Video decoding is broken on the Polaris login nodes.** OpenCV's
  ffmpeg cannot initialise its scaling context, and every `read()` then
  returns a frame-shaped buffer of garbage while reporting success — a
  write-then-read round trip of six known grey levels returns six
  identical frames. `hdr_data.video_i2.probe_decoder()` detects exactly
  this, and `I2VideoDataset` refuses to index mp4 clips until it passes.
  Run it on a compute node, or pass `frames_root=` a directory of
  pre-extracted frames.

## Looking at the output

```bash
python scripts/save_test_outputs.py                  # best checkpoint, all 28 frames
python scripts/save_test_outputs.py --limit 6        # a quick look
python scripts/save_test_outputs.py --crop-size 256  # small enough for a login node
./scripts/submit.sh test                             # benchmark + visuals, on a GPU
```

Writes `test_visuals/<run>/frame_NNNN_panel.jpg` — noisy | predicted |
reference on the top row, each expert's own output plus the tone-mapped
absolute error on the bottom, every panel labelled with its own PSNR-µ —
along with `frame_NNNN_gates.jpg` (per-pixel routing), `results.csv` and
`summary.md`.

The per-expert panels are the point. A mixture can report a healthy blended
PSNR while both experts have converged to the same function, in which case
the second expert costs parameters and latency and buys nothing. `summary.md`
calls this out directly: it flags a collapsed router (any expert with a mean
gate weight below 0.05) and near-duplicate experts (solo PSNRs within
0.15 dB).
