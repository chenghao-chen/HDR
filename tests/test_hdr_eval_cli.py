"""
The command line — `hdr_eval.cli`.

Why this file matters
─────────────────────
The CLI is how the harness is actually used, so its contract is the one
that matters in practice: given a dataset name, some baselines and an
output directory, it must produce per-image CSV, per-model JSON and a
comparison report — and it must fail with a readable message rather than
a traceback when something is missing.

Runs here use a synthetic on-disk corpus, tiny crops and a hard limit, so
the whole file stays fast enough to be part of the normal suite.
"""

import json
import os

import pytest
import torch

from helpers import make_dataset
from hdr_eval.cli import _parse_checkpoint_arg, _safe, main


@pytest.fixture
def corpus(tmp_path):
    return make_dataset(tmp_path / "data", n_train=2, n_test=2, h=64, w=64)


def run_cli(corpus, out, *extra):
    argv = ["--dataset", "mobile_hdr", "--data-root", corpus,
            "--split", "test", "--crop", "16", "--limit", "2",
            "--device", "cpu", "--no-amp", "--workers", "0",
            "--quiet", "--out", str(out), *extra]
    return main(argv)


class TestCatalogue:
    def test_list_exits_cleanly(self, capsys):
        assert main(["--list"]) == 0
        text = capsys.readouterr().out
        assert "mobile_hdr" in text and "wavelet+gbtf" in text

    def test_list_names_the_noise_presets(self, capsys):
        main(["--list"])
        assert "realistic" in capsys.readouterr().out


class TestRuns:
    def test_a_baseline_run_writes_every_artefact(self, corpus, tmp_path):
        out = tmp_path / "run"
        assert run_cli(corpus, out, "--baselines", "bilinear") == 0
        assert os.path.exists(out / "bilinear.csv")
        assert os.path.exists(out / "bilinear.json")
        assert os.path.exists(out / "report.md")

    def test_the_csv_has_one_row_per_image(self, corpus, tmp_path):
        out = tmp_path / "run"
        run_cli(corpus, out, "--baselines", "bilinear")
        lines = open(out / "bilinear.csv").read().strip().splitlines()
        assert len(lines) == 3               # header + two images

    def test_the_json_carries_the_aggregate(self, corpus, tmp_path):
        out = tmp_path / "run"
        run_cli(corpus, out, "--baselines", "malvar")
        payload = json.loads(open(out / "malvar.json").read())
        assert payload["model"] == "malvar" and "psnr_mu" in payload["aggregate"]

    def test_several_baselines_land_in_one_report(self, corpus, tmp_path):
        out = tmp_path / "run"
        run_cli(corpus, out, "--baselines", "bilinear,malvar,gaussian+gbtf")
        report = open(out / "report.md").read()
        for name in ("bilinear", "malvar", "gaussian+gbtf"):
            assert name in report
        assert "Change vs" in report

    def test_the_default_lineup_runs(self, corpus, tmp_path):
        out = tmp_path / "run"
        assert run_cli(corpus, out, "--default-baselines") == 0
        assert os.path.exists(out / "report.md")

    def test_a_noise_preset_is_honoured(self, corpus, tmp_path):
        low = tmp_path / "low"
        high = tmp_path / "high"
        run_cli(corpus, low, "--baselines", "bilinear", "--noise", "low")
        run_cli(corpus, high, "--baselines", "bilinear", "--noise", "extreme")
        low_psnr = json.loads(open(low / "bilinear.json").read())["aggregate"]["psnr_mu"]
        high_psnr = json.loads(open(high / "bilinear.json").read())["aggregate"]["psnr_mu"]
        assert low_psnr > high_psnr

    def test_tiled_inference_is_selectable(self, corpus, tmp_path):
        out = tmp_path / "run"
        assert run_cli(corpus, out, "--baselines", "bilinear",
                       "--inference", "tiled", "--tile", "8",
                       "--overlap", "4") == 0

    def test_stratified_columns_appear_when_requested(self, corpus, tmp_path):
        out = tmp_path / "run"
        run_cli(corpus, out, "--baselines", "bilinear", "--stratify")
        header = open(out / "bilinear.csv").readline()
        assert "lum[" in header and "snr[" in header

    def test_visuals_are_written_when_requested(self, corpus, tmp_path):
        out = tmp_path / "run"
        run_cli(corpus, out, "--baselines", "bilinear", "--save-visuals")
        assert len(os.listdir(out / "visuals")) >= 1

    def test_a_checkpoint_is_evaluated_alongside_baselines(self, corpus, tmp_path):
        from HDR_model_hybrid_Teacher import build_denoiser
        from helpers import make_checkpoint

        kwargs = {"dim": 8, "num_blocks": [1, 1, 1, 1],
                  "num_refinement_blocks": 1, "heads": [1, 1, 1, 1],
                  "se_reduction": 8}
        model = build_denoiser("moe", num_experts=2, **kwargs)
        ckpt = make_checkpoint(tmp_path / "run_dir" / "phase1_best.pth", model,
                               mode="moe", num_experts=2, model_kwargs=kwargs)
        out = tmp_path / "run"
        assert run_cli(corpus, out, "--baselines", "bilinear",
                       "--checkpoint", f"mine={ckpt}") == 0
        assert os.path.exists(out / "mine.json")
        header = open(out / "mine.csv").readline()
        assert "gate0_usage" in header

    def test_the_reference_model_can_be_chosen(self, corpus, tmp_path):
        out = tmp_path / "run"
        run_cli(corpus, out, "--baselines", "bilinear,malvar",
                "--reference", "malvar")
        assert "Change vs malvar" in open(out / "report.md").read()


class TestErrors:
    def test_no_models_is_a_readable_error(self, corpus, tmp_path):
        with pytest.raises(SystemExit, match="Nothing to evaluate"):
            run_cli(corpus, tmp_path / "run")

    def test_an_unknown_baseline_names_the_alternatives(self, corpus, tmp_path):
        with pytest.raises(KeyError, match="Demosaic only"):
            run_cli(corpus, tmp_path / "run", "--baselines", "magic")

    def test_an_unknown_dataset_names_the_alternatives(self, tmp_path):
        with pytest.raises(KeyError, match="mobile_hdr"):
            main(["--dataset", "nope", "--baselines", "bilinear",
                  "--out", str(tmp_path / "r")])

    def test_a_missing_dataset_root_explains_itself(self, tmp_path):
        with pytest.raises(RuntimeError, match="no samples found"):
            main(["--dataset", "mobile_hdr", "--data-root", str(tmp_path),
                  "--baselines", "bilinear", "--out", str(tmp_path / "r"),
                  "--device", "cpu", "--workers", "0"])


class TestArgumentHelpers:
    def test_a_named_checkpoint_is_split(self):
        assert _parse_checkpoint_arg("mine=/a/b.pth") == ("mine", "/a/b.pth")

    def test_an_unnamed_checkpoint_is_named_after_its_run_directory(self):
        """Two runs usually both end in phase1_best.pth; the directory differs."""
        name, path = _parse_checkpoint_arg("models_p1_moe_2026/phase1_best.pth")
        assert name == "models_p1_moe_2026/phase1_best"
        assert path == "models_p1_moe_2026/phase1_best.pth"

    def test_a_bare_filename_still_gets_a_name(self):
        name, path = _parse_checkpoint_arg("best.pth")
        assert name == "best" and path == "best.pth"

    def test_model_names_become_safe_filenames(self):
        assert _safe("wavelet+gbtf_post") == "wavelet_gbtf_post"
        assert _safe("run/phase1_best") == "run_phase1_best"
