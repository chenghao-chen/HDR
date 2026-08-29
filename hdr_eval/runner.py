"""
hdr_eval/runner.py — the benchmark loop
========================================

One model, one dataset, one row of numbers per image, plus the aggregate.
The loop itself is unremarkable; the choices around it are the part worth
reading.

**Ground truth is demosaiced, not measured.** The datasets hand back a
clean *CFA* frame, so the RGB reference has to be produced by demosaicing
it — this project's GBTF by default, matching what
test_dual_MoE_two_phase.py does, so numbers stay comparable with the
existing results. The consequence is that no model can score better
against this reference than the reference demosaicer's own accuracy on
noise-free input, and a model that demosaics *differently but equally
well* is penalised. ``gt_demosaic`` is therefore recorded in the run
metadata, and can be changed.

**The noisy input is scored too.** Every run reports the metrics of the
demosaiced noisy input, so every improvement figure has a denominator. A
model that gains 8 dB over the noisy input on one noise preset and 2 dB
on another is telling you something a bare PSNR column is not.

**Per-expert and per-gate columns are always present.** For a model with
one expert they are trivially its own score and 1.0; for the MoE they are
the routing diagnostics. Keeping the columns uniform means one CSV schema
across every model in a comparison.

**Timing excludes metrics.** ``time_sec`` is the inference call alone,
CUDA-synchronised, so it is comparable across models. Metric computation
(LPIPS especially) can cost more than inference and would swamp it.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hdr_platform import get_accelerator

from .inference import InferenceConfig, ModelOutput, infer
from .metrics import MetricConfig, MetricSuite, psnr_per_image
from .regions import (
    DEFAULT_LUMINANCE_EDGES,
    DEFAULT_SNR_EDGES,
    edge_mask,
    luminance_bands,
    masked_psnr,
    saturation_mask,
    snr_bands,
    stratify,
)
from .tonemap import mu_law

__all__ = ["RunConfig", "BenchmarkResult", "BenchmarkRunner", "run_benchmark"]


def _default_snr_fn(x: torch.Tensor) -> torch.Tensor:
    """The canonical SNR map both train and test scripts use."""
    from HDR_model_hybrid_Teacher import estimate_local_snr_map
    return estimate_local_snr_map(x, window_size=5)


@dataclass
class RunConfig:
    """
    Everything about *how* a benchmark runs, separate from what it runs.

    Attributes
    ──────────
    device
        Where inference happens. Defaults to CUDA when available.
    inference
        Full-frame or tiled, tile size, autocast dtype.
    metrics
        Which metrics to compute.
    stratify_luminance / stratify_snr / report_edges
        Extra per-region columns. Off by default: they roughly triple the
        column count, which is useful for analysis and noisy for a
        headline table.
    gt_demosaic
        Which demosaicer builds the RGB reference from the clean CFA.
    limit
        Stop after this many images — for smoke tests.
    save_visuals / visual_stride
        Write comparison images every Nth sample.
    progress
        Print a line per image.
    """

    device: Optional[torch.device] = None
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    stratify_luminance: bool = False
    stratify_snr: bool = False
    report_edges: bool = False
    gt_demosaic: str = "gbtf"
    pattern: str = "BGGR"
    limit: Optional[int] = None
    num_workers: int = 2
    save_visuals: bool = False
    visual_stride: int = 1
    output_dir: Optional[str] = None
    progress: bool = True

    def resolved_device(self) -> torch.device:
        """
        The device to run on, defaulting to whatever accelerator this
        machine has — CUDA on Polaris, XPU on Aurora, CPU anywhere else.
        """
        if self.device is not None:
            return torch.device(self.device)
        return get_accelerator().device

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly view, stored alongside the results."""
        d = asdict(self)
        d["device"] = str(self.resolved_device())
        d["inference"]["amp_dtype"] = str(self.inference.amp_dtype)
        return d


@dataclass
class BenchmarkResult:
    """
    The outcome of one (model, dataset) run.

    ``records`` holds one dict per image; ``aggregate`` is the mean of
    every numeric column. Both are written out, because a mean alone
    hides the two catastrophic frames that produced it.
    """

    model_name: str
    dataset_name: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_images(self) -> int:
        return len(self.records)

    def columns(self) -> List[str]:
        """Union of every record's keys, in first-seen order."""
        seen: List[str] = []
        for rec in self.records:
            for key in rec:
                if key not in seen:
                    seen.append(key)
        return seen

    def aggregate(self) -> Dict[str, float]:
        """
        Mean of each numeric column, skipping NaN.

        NaN is what an empty stratification band reports, and a band that
        is empty on some images should not poison the average of the ones
        where it is not.
        """
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for rec in self.records:
            for key, value in rec.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if value != value:                      # NaN
                    continue
                sums[key] = sums.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        return {k: sums[k] / counts[k] for k in sums if counts[k]}

    def to_csv(self, path: str) -> str:
        """Write the per-image records. Returns the path."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cols = self.columns()
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for rec in self.records:
                writer.writerow(rec)
        return path

    def to_json(self, path: str) -> str:
        """Write aggregate + metadata + records. Returns the path."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": self.model_name,
            "dataset": self.dataset_name,
            "num_images": self.num_images,
            "aggregate": self.aggregate(),
            "meta": self.meta,
            "records": self.records,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    @classmethod
    def from_json(cls, path: str) -> "BenchmarkResult":
        """Reload a previously written result, for later comparison."""
        with open(path) as fh:
            payload = json.load(fh)
        return cls(model_name=payload["model"],
                   dataset_name=payload.get("dataset", "?"),
                   records=payload.get("records", []),
                   meta=payload.get("meta", {}))


class BenchmarkRunner:
    """
    Runs one model over one dataset.

        runner = BenchmarkRunner(model, dataset, RunConfig(), name="moe")
        result = runner.run()
        print(result.aggregate()["psnr_mu"])
    """

    def __init__(self, model: Callable, dataset: Dataset,
                 config: Optional[RunConfig] = None,
                 name: str = "model", dataset_name: str = "dataset",
                 snr_fn: Optional[Callable] = None):
        self.model = model
        self.dataset = dataset
        self.config = config or RunConfig()
        self.name = name
        self.dataset_name = dataset_name
        self.snr_fn = snr_fn or _default_snr_fn
        self.device = self.config.resolved_device()
        # Same device, but with the backend-specific operations attached:
        # synchronise, memory stats, pinned-memory support.
        self.accelerator = get_accelerator(str(self.device))
        self._suite = MetricSuite(self.config.metrics, device=self.device)
        self._gt_demosaic = self._build_gt_demosaic()

    # ── setup ────────────────────────────────────────────────────────────
    def _build_gt_demosaic(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """The function that turns clean packed CFA into the RGB reference."""
        from hdr_baselines.pipelines import DEMOSAIC_FUNCTIONS
        from hdr_data.bayer import unpack

        name = self.config.gt_demosaic
        if name not in DEMOSAIC_FUNCTIONS:
            raise ValueError(
                f"Unknown gt_demosaic '{name}'. Available: "
                f"{sorted(DEMOSAIC_FUNCTIONS)}")
        fn = DEMOSAIC_FUNCTIONS[name]
        pattern = self.config.pattern

        def demosaic(packed: torch.Tensor) -> torch.Tensor:
            return fn(unpack(packed.clamp(0, 1).float()), pattern).clamp(0, 1)

        return demosaic

    def _prepare_model(self):
        """Move to device and switch to eval, when the model supports it."""
        model = self.model
        if isinstance(model, torch.nn.Module):
            model = model.to(self.device)
            model.eval()
        return model

    # ── per-image work ───────────────────────────────────────────────────
    def _expert_columns(self, out: ModelOutput, gt_rgb: torch.Tensor,
                        mu: float) -> Dict[str, float]:
        """Per-expert PSNR-mu and mean gate usage."""
        row: Dict[str, float] = {}
        if out.experts is None or out.gates is None:
            return row
        tm_gt = mu_law(gt_rgb, mu)
        k = out.experts.shape[1]
        for i in range(k):
            expert = out.experts[:, i].clamp(0, 1).float()
            row[f"psnr_expert{i}_mu"] = float(
                psnr_per_image(mu_law(expert, mu), tm_gt).mean())
            row[f"gate{i}_usage"] = float(out.gates[:, i].mean())
        return row

    def _region_columns(self, pred: torch.Tensor, gt: torch.Tensor,
                        snr_map: torch.Tensor, mu: float) -> Dict[str, float]:
        """Stratified PSNR-mu columns, per the config's switches."""
        cfg = self.config
        row: Dict[str, float] = {}
        if not (cfg.stratify_luminance or cfg.stratify_snr or cfg.report_edges):
            return row

        tm_pred, tm_gt = mu_law(pred, mu), mu_law(gt, mu)
        if cfg.stratify_luminance:
            row.update(stratify(tm_pred, tm_gt,
                                luminance_bands(gt, DEFAULT_LUMINANCE_EDGES)))
        if cfg.stratify_snr:
            # The SNR map is at packed resolution; the images are at sensor
            # resolution, so it has to be upsampled to index the same pixels.
            snr_up = F.interpolate(snr_map, size=gt.shape[-2:],
                                   mode="bilinear", align_corners=False)
            row.update(stratify(tm_pred, tm_gt,
                                snr_bands(snr_up, DEFAULT_SNR_EDGES)))
        if cfg.report_edges:
            mask = edge_mask(gt)
            row["psnr_mu.edges"] = float(masked_psnr(tm_pred, tm_gt, mask).nanmean())
            row["psnr_mu.edges.coverage"] = float(mask.float().mean())
            sat = saturation_mask(gt)
            row["saturated_fraction"] = float(sat.float().mean())
        return row

    def _time_inference(self, model, x: torch.Tensor) -> tuple:
        """
        Run inference, returning (output, wall-clock seconds).

        Both syncs are load-bearing: GPU work is queued asynchronously, so
        without them this measures how long it takes to *submit* the kernels
        — microseconds — rather than to run them.
        """
        self.accelerator.synchronize()
        t0 = time.perf_counter()
        out = infer(model, x, self.snr_fn, self.config.inference)
        self.accelerator.synchronize()
        return out, time.perf_counter() - t0

    # ── the loop ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def run(self) -> BenchmarkResult:
        cfg = self.config
        model = self._prepare_model()
        mu = cfg.metrics.mu

        # pin_memory needs a device to pin *for*: the DataLoader default is
        # CUDA, so on Aurora it has to be told "xpu" explicitly or the copy
        # stays pageable and the speedup is silently lost.
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False,
                            **self.accelerator.dataloader_kwargs(
                                num_workers=cfg.num_workers))

        result = BenchmarkResult(
            model_name=self.name, dataset_name=self.dataset_name,
            meta={
                "config": cfg.to_dict(),
                "num_available": len(self.dataset),
                "trainable_untrained": bool(
                    getattr(model, "trainable", False)
                    and not getattr(model, "loaded_checkpoint", False)),
                "dataset_repr": repr(self.dataset),
            })

        if cfg.save_visuals and cfg.output_dir:
            os.makedirs(os.path.join(cfg.output_dir, "visuals"), exist_ok=True)

        for i, sample in enumerate(loader):
            if cfg.limit is not None and i >= cfg.limit:
                break

            x = sample["x"].to(self.device, non_blocking=True)
            y = sample["y"].to(self.device, non_blocking=True)

            out, elapsed = self._time_inference(model, x)
            pred = out.blended.clamp(0, 1).float()

            gt_rgb = self._gt_demosaic(y)
            noisy_rgb = self._gt_demosaic(x)
            snr_map = self.snr_fn(x)

            row: Dict[str, Any] = {
                "index": i,
                "source_id": self._source_id(sample, i),
                "height": int(pred.shape[-2]),
                "width": int(pred.shape[-1]),
                "time_sec": elapsed,
            }
            row.update(self._suite(pred, gt_rgb))
            row.update({f"noisy_{k}": v
                        for k, v in self._suite(noisy_rgb, gt_rgb).items()})
            row["psnr_mu_gain"] = row["psnr_mu"] - row["noisy_psnr_mu"]
            row["pct_low_snr_pixels"] = float((snr_map < 0.5).float().mean()) * 100.0
            row.update(self._expert_columns(out, gt_rgb, mu))
            row.update(self._region_columns(pred, gt_rgb, snr_map, mu))
            result.records.append(row)

            if cfg.progress:
                print(f"  [{self.name}] {i + 1:>4}/{len(self.dataset)}  "
                      f"PSNR-mu {row['psnr_mu']:6.2f} dB  "
                      f"(noisy {row['noisy_psnr_mu']:6.2f}, "
                      f"gain {row['psnr_mu_gain']:+5.2f})  "
                      f"{elapsed:6.3f}s", flush=True)

            if (cfg.save_visuals and cfg.output_dir
                    and i % max(cfg.visual_stride, 1) == 0):
                self._save_visual(i, noisy_rgb, pred, gt_rgb, out)

        return result

    @staticmethod
    def _source_id(sample: Dict[str, Any], index: int) -> str:
        """The dataset's own identifier when it provides one."""
        sid = sample.get("source_id")
        if isinstance(sid, (list, tuple)) and sid:
            return str(sid[0])
        if isinstance(sid, str):
            return sid
        return f"sample_{index:04d}"

    def _save_visual(self, index: int, noisy: torch.Tensor,
                     pred: torch.Tensor, gt: torch.Tensor,
                     out: ModelOutput) -> None:
        from .visualize import save_comparison, save_gate_map
        directory = os.path.join(self.config.output_dir, "visuals")
        stem = f"{self.name}_{index:04d}"
        save_comparison(
            os.path.join(directory, f"{stem}_compare.jpg"),
            {"noisy": noisy, "predicted": pred, "reference": gt},
            mu=self.config.metrics.mu)
        if out.gates is not None and out.gates.shape[1] > 1:
            save_gate_map(os.path.join(directory, f"{stem}_gates.jpg"),
                          out.gates)


def run_benchmark(model: Callable, dataset: Dataset,
                  config: Optional[RunConfig] = None, name: str = "model",
                  dataset_name: str = "dataset") -> BenchmarkResult:
    """Convenience wrapper around :class:`BenchmarkRunner`."""
    return BenchmarkRunner(model, dataset, config, name, dataset_name).run()
