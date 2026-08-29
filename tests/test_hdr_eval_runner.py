"""
The benchmark loop — `hdr_eval.runner`.

Why this file matters
─────────────────────
The runner decides what a "result" is, and three of its choices are the
sort that quietly invalidate a comparison if they break:

* **The noisy input is scored alongside the model**, so every gain figure
  has a denominator. Without it, "+8 dB" is unreadable across noise levels.
* **Expert and gate columns are uniform** across every model, so a
  single-output baseline and a K-expert MoE share one CSV schema and can
  sit in one table.
* **NaN-safe aggregation.** An empty stratification band reports NaN; if
  the mean did not skip those, one empty band on one image would blank out
  the whole column.

Timing is asserted to be present and positive but never asserted to be
fast — that would be a flaky test of the machine, not the code.
"""

import json
import os

import pytest
import torch

from hdr_eval.inference import InferenceConfig
from hdr_eval.metrics import MetricConfig
from hdr_eval.runner import (
    BenchmarkResult,
    BenchmarkRunner,
    RunConfig,
    run_benchmark,
)


class TinyDataset(torch.utils.data.Dataset):
    """Four small packed CFA pairs, deterministic."""

    def __init__(self, n=4, h=32, w=32, with_ids=True):
        g = torch.Generator().manual_seed(0)
        self.samples = []
        for i in range(n):
            clean = (torch.rand((4, h, w), generator=g) * 0.7 + 0.15)
            noisy = (clean + 0.03 * torch.randn((4, h, w), generator=g)).clamp(0, 1)
            item = {"x": noisy, "xm": noisy, "y": clean}
            if with_ids:
                item["source_id"] = f"img{i:02d}"
            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def identity_model(x, snr):
    """Demosaic-free passthrough: upsample the CFA channels to RGB."""
    import torch.nn.functional as F
    return F.interpolate(x[:, :3], scale_factor=2, mode="nearest")


def moe_like_model(x, snr):
    """Two experts and a real gate, as the project's models return."""
    import torch.nn.functional as F
    rgb = F.interpolate(x[:, :3], scale_factor=2, mode="nearest")
    experts = torch.stack([rgb, rgb * 0.5], dim=1)
    gates = torch.stack([torch.full_like(rgb[:, 0], 0.7),
                         torch.full_like(rgb[:, 0], 0.3)], dim=1)
    blended = (gates.unsqueeze(2) * experts).sum(dim=1)
    return blended, experts, gates


@pytest.fixture
def cpu_config(tmp_path):
    return RunConfig(
        device=torch.device("cpu"),
        inference=InferenceConfig(mode="full", amp_dtype=None),
        metrics=MetricConfig(delta_e=False),
        num_workers=0, progress=False, output_dir=str(tmp_path))


class TestRunConfig:
    def test_it_defaults_to_cuda_when_present(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert RunConfig().resolved_device().type == expected

    def test_an_explicit_device_wins(self):
        assert RunConfig(device=torch.device("cpu")).resolved_device().type == "cpu"

    def test_to_dict_is_json_serialisable(self, cpu_config):
        json.dumps(cpu_config.to_dict())

    def test_an_unknown_gt_demosaic_is_rejected(self, cpu_config):
        cpu_config.gt_demosaic = "guesswork"
        with pytest.raises(ValueError, match="gt_demosaic"):
            BenchmarkRunner(identity_model, TinyDataset(), cpu_config)


class TestRun:
    def test_it_produces_one_record_per_image(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config,
                                 name="id").run()
        assert result.num_images == 4

    def test_limit_stops_early(self, cpu_config):
        cpu_config.limit = 2
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert result.num_images == 2

    def test_the_record_carries_the_expected_columns(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        row = result.records[0]
        for key in ("index", "source_id", "height", "width", "time_sec",
                    "psnr_mu", "noisy_psnr_mu", "psnr_mu_gain",
                    "pct_low_snr_pixels"):
            assert key in row, key

    def test_the_noisy_baseline_is_scored_too(self, cpu_config):
        """Every gain figure needs its denominator recorded beside it."""
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        row = result.records[0]
        assert row["psnr_mu_gain"] == pytest.approx(
            row["psnr_mu"] - row["noisy_psnr_mu"])

    def test_the_source_id_comes_from_the_dataset(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert [r["source_id"] for r in result.records] == [
            "img00", "img01", "img02", "img03"]

    def test_it_falls_back_to_a_positional_id(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(with_ids=False),
                                 cpu_config).run()
        assert result.records[0]["source_id"] == "sample_0000"

    def test_timing_is_recorded_and_positive(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert all(r["time_sec"] > 0 for r in result.records)

    def test_output_is_at_sensor_resolution(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert result.records[0]["height"] == 64

    def test_tiled_mode_runs(self, cpu_config):
        cpu_config.inference = InferenceConfig(mode="tiled", tile=16, overlap=4,
                                               amp_dtype=None)
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert result.num_images == 4


class TestExpertColumns:
    def test_a_routed_model_reports_per_expert_scores_and_usage(self, cpu_config):
        result = BenchmarkRunner(moe_like_model, TinyDataset(), cpu_config,
                                 name="moe").run()
        row = result.records[0]
        assert "psnr_expert0_mu" in row and "psnr_expert1_mu" in row
        assert row["gate0_usage"] == pytest.approx(0.7, abs=1e-5)
        assert row["gate1_usage"] == pytest.approx(0.3, abs=1e-5)

    def test_gate_usage_sums_to_one(self, cpu_config):
        result = BenchmarkRunner(moe_like_model, TinyDataset(), cpu_config).run()
        row = result.records[0]
        assert row["gate0_usage"] + row["gate1_usage"] == pytest.approx(1.0, abs=1e-5)

    def test_an_unrouted_model_simply_omits_them(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert "gate0_usage" not in result.records[0]


class TestStratifiedColumns:
    def test_they_are_off_by_default(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert not any(k.startswith("lum[") for k in result.records[0])

    def test_luminance_bands_can_be_enabled(self, cpu_config):
        cpu_config.stratify_luminance = True
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert any(k.startswith("lum[") for k in result.records[0])

    def test_snr_bands_can_be_enabled(self, cpu_config):
        cpu_config.stratify_snr = True
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert any(k.startswith("snr[") for k in result.records[0])

    def test_edge_and_saturation_columns_can_be_enabled(self, cpu_config):
        cpu_config.report_edges = True
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        row = result.records[0]
        assert "psnr_mu.edges" in row and "saturated_fraction" in row


class TestResultObject:
    def test_aggregate_averages_the_numeric_columns(self):
        result = BenchmarkResult("m", "d", records=[
            {"psnr_mu": 10.0, "source_id": "a"},
            {"psnr_mu": 20.0, "source_id": "b"},
        ])
        assert result.aggregate()["psnr_mu"] == 15.0

    def test_aggregate_skips_nan(self):
        """One empty band on one image must not blank out the column."""
        result = BenchmarkResult("m", "d", records=[
            {"band": float("nan")}, {"band": 4.0}, {"band": 6.0}])
        assert result.aggregate()["band"] == 5.0

    def test_aggregate_ignores_strings_and_bools(self):
        result = BenchmarkResult("m", "d", records=[
            {"source_id": "x", "ok": True, "psnr_mu": 3.0}])
        assert set(result.aggregate()) == {"psnr_mu"}

    def test_aggregate_of_nothing_is_empty(self):
        assert BenchmarkResult("m", "d").aggregate() == {}

    def test_columns_are_in_first_seen_order(self):
        result = BenchmarkResult("m", "d", records=[
            {"a": 1, "b": 2}, {"b": 2, "c": 3}])
        assert result.columns() == ["a", "b", "c"]

    def test_csv_round_trips_through_the_filesystem(self, tmp_path, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        path = result.to_csv(str(tmp_path / "out" / "r.csv"))
        assert os.path.exists(path)
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        assert len(lines) == 5                     # header + four images

    def test_json_carries_aggregate_and_records(self, tmp_path, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config,
                                 name="id").run()
        path = result.to_json(str(tmp_path / "r.json"))
        payload = json.loads(open(path).read())
        assert payload["model"] == "id" and payload["num_images"] == 4
        assert "psnr_mu" in payload["aggregate"]

    def test_json_reloads_into_a_result(self, tmp_path, cpu_config):
        original = BenchmarkRunner(identity_model, TinyDataset(), cpu_config,
                                   name="id").run()
        path = original.to_json(str(tmp_path / "r.json"))
        reloaded = BenchmarkResult.from_json(path)
        assert reloaded.model_name == "id"
        assert reloaded.num_images == original.num_images
        assert reloaded.aggregate()["psnr_mu"] == pytest.approx(
            original.aggregate()["psnr_mu"])


class TestVisuals:
    def test_they_are_written_when_asked(self, tmp_path, cpu_config):
        cpu_config.save_visuals = True
        cpu_config.limit = 2
        BenchmarkRunner(identity_model, TinyDataset(), cpu_config,
                        name="id").run()
        files = os.listdir(tmp_path / "visuals")
        assert len([f for f in files if f.endswith("_compare.jpg")]) == 2

    def test_the_stride_thins_them_out(self, tmp_path, cpu_config):
        cpu_config.save_visuals = True
        cpu_config.visual_stride = 2
        BenchmarkRunner(identity_model, TinyDataset(), cpu_config,
                        name="id").run()
        assert len(os.listdir(tmp_path / "visuals")) == 2

    def test_a_routed_model_also_writes_gate_maps(self, tmp_path, cpu_config):
        cpu_config.save_visuals = True
        cpu_config.limit = 1
        BenchmarkRunner(moe_like_model, TinyDataset(), cpu_config,
                        name="moe").run()
        files = os.listdir(tmp_path / "visuals")
        assert any(f.endswith("_gates.jpg") for f in files)


class TestMetadata:
    def test_an_untrained_learned_baseline_is_flagged(self, cpu_config):
        """Otherwise a random-init smoke test looks like a real result."""
        from hdr_baselines import build_baseline
        model = build_baseline("dncnn")
        result = BenchmarkRunner(model, TinyDataset(), cpu_config,
                                 name="dncnn").run()
        assert result.meta["trainable_untrained"] is True

    def test_a_classical_baseline_is_not_flagged(self, cpu_config):
        from hdr_baselines import build_baseline
        result = BenchmarkRunner(build_baseline("malvar"), TinyDataset(),
                                 cpu_config, name="malvar").run()
        assert result.meta["trainable_untrained"] is False

    def test_the_run_configuration_is_recorded(self, cpu_config):
        result = BenchmarkRunner(identity_model, TinyDataset(), cpu_config).run()
        assert result.meta["config"]["gt_demosaic"] == "gbtf"
        assert result.meta["num_available"] == 4


class TestConvenienceWrapper:
    def test_run_benchmark_matches_the_class(self, cpu_config):
        a = run_benchmark(identity_model, TinyDataset(), cpu_config, name="id")
        b = BenchmarkRunner(identity_model, TinyDataset(), cpu_config,
                            name="id").run()
        assert a.aggregate()["psnr_mu"] == pytest.approx(
            b.aggregate()["psnr_mu"])
