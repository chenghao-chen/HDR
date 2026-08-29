"""
hdr_eval — evaluation harness for the HDR joint denoise+demosaic models
=======================================================================

    from hdr_data import build_dataset
    from hdr_baselines import build_baseline
    from hdr_eval import BenchmarkRunner, RunConfig, build_report

    ds = build_dataset("mobile_hdr", split="test", noise="high")
    res = BenchmarkRunner(build_baseline("wavelet+gbtf"), ds,
                          RunConfig(limit=4), name="wavelet+gbtf").run()
    print(build_report([res], markdown=False))

or from the command line::

    python -m hdr_eval.cli --checkpoint best.pth --default-baselines --limit 5

Modules: ``tonemap`` (curves), ``metrics`` (PSNR/SSIM/MS-SSIM/LPIPS/dE00),
``regions`` (per-band stratification), ``inference`` (full and tiled),
``runner`` (the loop), ``report`` (tables), ``visualize`` (figures),
``checkpoints`` (rebuilding a trained model), ``cli``.
"""

from .checkpoints import CheckpointInfo, load_checkpoint, load_model_from_checkpoint
from .inference import (
    InferenceConfig,
    ModelOutput,
    infer,
    infer_full,
    infer_tiled,
)
from .metrics import (
    LPIPSMetric,
    MetricConfig,
    MetricSuite,
    delta_e_2000,
    ms_ssim,
    psnr,
    psnr_per_image,
    ssim,
)
from .regions import (
    edge_mask,
    luminance_bands,
    masked_psnr,
    saturation_mask,
    snr_bands,
    stratify,
)
from .report import build_report, summary_table, write_report
from .runner import BenchmarkResult, BenchmarkRunner, RunConfig, run_benchmark
# NB: the `tonemap` *function* is deliberately not re-exported here —
# it would shadow the `hdr_eval.tonemap` submodule on the package,
# so `from hdr_eval import tonemap` would hand back a function.
# Import it as `from hdr_eval.tonemap import tonemap`.
from .tonemap import MU_DEFAULT, mu_law, mu_law_inverse
from .visualize import save_comparison, save_error_heatmap, save_gate_map

__all__ = [
    "MU_DEFAULT", "mu_law", "mu_law_inverse",
    "psnr", "psnr_per_image", "ssim", "ms_ssim", "delta_e_2000",
    "MetricConfig", "MetricSuite", "LPIPSMetric",
    "luminance_bands", "snr_bands", "edge_mask", "saturation_mask",
    "masked_psnr", "stratify",
    "InferenceConfig", "ModelOutput", "infer", "infer_full", "infer_tiled",
    "RunConfig", "BenchmarkRunner", "BenchmarkResult", "run_benchmark",
    "build_report", "summary_table", "write_report",
    "save_comparison", "save_error_heatmap", "save_gate_map",
    "load_model_from_checkpoint", "load_checkpoint", "CheckpointInfo",
]
