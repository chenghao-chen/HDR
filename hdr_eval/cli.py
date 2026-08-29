"""
hdr_eval/cli.py — run the benchmark from the command line
==========================================================

    # one checkpoint against the classical lineup, on 5 test images
    python -m hdr_eval.cli --dataset mobile_hdr --split test \
        --checkpoint models_p1_moe_.../phase1_best.pth \
        --baselines gbtf,wavelet+gbtf,malvar \
        --limit 5 --out test_results/compare

    # how the same model degrades as the noise gets worse
    for n in low medium high extreme; do
      python -m hdr_eval.cli --dataset mobile_hdr --noise $n \
          --checkpoint best.pth --out results/$n
    done

    # what is available
    python -m hdr_eval.cli --list

Every run writes, under ``--out``: one ``<model>.csv`` of per-image rows
and ``<model>.json`` per model, a ``report.md`` comparing them, and
``visuals/`` when ``--save-visuals`` is given.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default="mobile_hdr",
                      help="Registered dataset name (default: mobile_hdr).")
    data.add_argument("--data-root", default=None,
                      help="Override the dataset root; otherwise resolved "
                           "from the dataset's environment variables.")
    data.add_argument("--split", default="test", choices=("train", "test"))
    data.add_argument("--noise", default=None,
                      help="Noise preset (default: the dataset's own).")
    data.add_argument("--crop", type=int, default=None,
                      help="Centre-crop this many packed pixels per image.")
    data.add_argument("--limit", type=int, default=None,
                      help="Stop after this many images.")
    data.add_argument("--pattern", default="BGGR",
                      help="CFA phase of the data (default: BGGR).")

    models = parser.add_argument_group("models")
    models.add_argument("--checkpoint", action="append", default=[],
                        metavar="[NAME=]PATH",
                        help="A trained checkpoint; repeatable. Prefix with "
                             "'name=' to label it in the report.")
    models.add_argument("--baselines", default="",
                        help="Comma-separated baseline names, e.g. "
                             "'gbtf,wavelet+gbtf,malvar'.")
    models.add_argument("--default-baselines", action="store_true",
                        help="Add the standard classical lineup.")
    models.add_argument("--reference", default=None,
                        help="Model name to compute deltas against "
                             "(default: the first baseline, if any).")

    infer = parser.add_argument_group("inference")
    infer.add_argument("--inference", default="full", choices=("full", "tiled"))
    infer.add_argument("--tile", type=int, default=512,
                       help="Tile size in packed pixels (tiled mode).")
    infer.add_argument("--overlap", type=int, default=None,
                       help="Tile overlap; defaults to a quarter of --tile.")
    infer.add_argument("--device", default=None,
                       help="cuda:0, xpu:0, cpu, ... "
                            "(default: this machine's accelerator, else cpu).")
    infer.add_argument("--no-amp", action="store_true",
                       help="Disable bfloat16 autocast on CUDA.")
    infer.add_argument("--workers", type=int, default=2,
                       help="DataLoader workers (default: 2).")

    metrics = parser.add_argument_group("metrics")
    metrics.add_argument("--mu", type=float, default=5000.0,
                         help="Tone-curve mu. Must match training (5000) for "
                              "PSNR-mu to be comparable with the W&B curves.")
    metrics.add_argument("--lpips", action="store_true",
                         help="Also compute LPIPS (slow; needs the package).")
    metrics.add_argument("--ms-ssim", action="store_true",
                         help="Also compute MS-SSIM (needs >=161px images).")
    metrics.add_argument("--no-delta-e", action="store_true",
                         help="Skip the CIEDE2000 colour metric.")
    metrics.add_argument("--stratify", action="store_true",
                         help="Add per-luminance-band and per-SNR-band columns.")
    metrics.add_argument("--edges", action="store_true",
                         help="Add edge-region and saturation columns.")
    metrics.add_argument("--gt-demosaic", default="gbtf",
                         choices=("nearest", "bilinear", "malvar", "gbtf"),
                         help="Demosaicer that builds the RGB reference.")

    out = parser.add_argument_group("output")
    out.add_argument("--out", default="test_results/benchmark",
                     help="Output directory.")
    out.add_argument("--save-visuals", action="store_true")
    out.add_argument("--visual-stride", type=int, default=1)
    out.add_argument("--quiet", action="store_true",
                     help="Suppress the per-image progress lines.")
    out.add_argument("--list", action="store_true",
                     help="List datasets, baselines and noise presets, then exit.")


def _print_catalogue() -> None:
    from hdr_baselines.registry import describe_baselines
    from hdr_data.noise import list_presets
    from hdr_data.registry import DATASETS

    print("Datasets:")
    for name, spec in sorted(DATASETS.items()):
        print(f"  {name:<12} {spec.description}")
    print("\nNoise presets:")
    print("  " + ", ".join(list_presets()))
    print("\nBaselines:")
    for line in describe_baselines().splitlines():
        print("  " + line)


def _parse_checkpoint_arg(value: str) -> Tuple[str, str]:
    """``name=path`` or ``path`` -> (name, path)."""
    if "=" in value:
        name, path = value.split("=", 1)
        name, path = name.strip(), path.strip()
        if name and path:
            return name, path
    # Name it after the run directory, which is what distinguishes two
    # checkpoints far more often than the file name (both are usually
    # 'phase1_best.pth').
    parent = os.path.basename(os.path.dirname(value.rstrip("/")))
    stem = os.path.splitext(os.path.basename(value))[0]
    return (f"{parent}/{stem}" if parent else stem), value


def build_models(args: argparse.Namespace, device: torch.device
                 ) -> List[Tuple[str, object]]:
    """Construct every model named on the command line, in order."""
    from hdr_baselines.registry import DEFAULT_LINEUP, build_baseline

    from .checkpoints import load_model_from_checkpoint

    models: List[Tuple[str, object]] = []

    names: List[str] = []
    if args.default_baselines:
        names.extend(DEFAULT_LINEUP)
    if args.baselines:
        names.extend(n.strip() for n in args.baselines.split(",") if n.strip())
    for name in names:
        if any(existing == name for existing, _ in models):
            continue
        models.append((name, build_baseline(name, pattern=args.pattern)))

    for entry in args.checkpoint:
        name, path = _parse_checkpoint_arg(entry)
        model, info = load_model_from_checkpoint(path, device=device)
        print(f"  loaded {name}: {info}")
        models.append((name, model))

    if not models:
        raise SystemExit(
            "Nothing to evaluate. Pass --checkpoint and/or --baselines "
            "(or --default-baselines). See --list.")
    return models


def build_dataset(args: argparse.Namespace):
    """Construct the dataset described by the command line."""
    from hdr_data.registry import build_dataset as _build

    kwargs: Dict[str, object] = {"split": args.split, "pattern": args.pattern}
    if args.noise:
        kwargs["noise"] = args.noise
    if args.crop:
        kwargs["crop_size"] = args.crop
    kwargs["return_meta"] = True
    return _build(args.dataset, root=args.data_root, **kwargs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hdr_eval",
        description="Benchmark HDR joint denoise+demosaic models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    _add_arguments(parser)
    args = parser.parse_args(argv)

    if args.list:
        _print_catalogue()
        return 0

    from .inference import InferenceConfig
    from .metrics import MetricConfig
    from .report import write_report
    from .runner import BenchmarkRunner, RunConfig

    # Resolves to cuda:0 on Polaris, xpu:0 on Aurora, cpu on a login node,
    # and raises a readable error if --device names a backend that is not
    # present rather than failing later inside a .to() call.
    from hdr_platform import get_accelerator
    accelerator = get_accelerator(args.device)
    device = accelerator.device
    # Same reduced-precision matmul settings the training and test scripts
    # use, so timings are comparable with them.
    accelerator.enable_fast_matmul()

    print(f"Device: {accelerator.summary()}")
    dataset = build_dataset(args)
    print(f"Dataset: {dataset}")
    models = build_models(args, device)

    overlap = args.overlap if args.overlap is not None else args.tile // 4
    run_config = RunConfig(
        device=device,
        inference=InferenceConfig(
            mode=args.inference, tile=args.tile, overlap=overlap,
            amp_dtype=None if args.no_amp else torch.bfloat16),
        metrics=MetricConfig(
            mu=args.mu, lpips=args.lpips, ms_ssim=args.ms_ssim,
            delta_e=not args.no_delta_e),
        stratify_luminance=args.stratify,
        stratify_snr=args.stratify,
        report_edges=args.edges,
        gt_demosaic=args.gt_demosaic,
        pattern=args.pattern,
        limit=args.limit,
        num_workers=args.workers,
        save_visuals=args.save_visuals,
        visual_stride=args.visual_stride,
        output_dir=args.out,
        progress=not args.quiet,
    )

    os.makedirs(args.out, exist_ok=True)
    results = []
    for name, model in models:
        print(f"\n=== {name} ===")
        runner = BenchmarkRunner(model, dataset, run_config, name=name,
                                 dataset_name=args.dataset)
        result = runner.run()
        result.to_csv(os.path.join(args.out, f"{_safe(name)}.csv"))
        result.to_json(os.path.join(args.out, f"{_safe(name)}.json"))
        results.append(result)

    reference = args.reference
    if reference is None and len(results) > 1:
        reference = results[0].model_name

    report_path = write_report(
        os.path.join(args.out, "report.md"), results,
        reference=reference, markdown=True)

    print()
    from .report import summary_table
    print(summary_table(results))
    print(f"\nWrote {report_path}")
    print(f"Per-image CSV/JSON in {args.out}")
    return 0


def _safe(name: str) -> str:
    """Filesystem-safe version of a model name."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


if __name__ == "__main__":
    sys.exit(main())
