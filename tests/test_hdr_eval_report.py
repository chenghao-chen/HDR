"""
Result presentation — `hdr_eval.report`.

Why this file matters
─────────────────────
Reports are where a benchmark either answers the question or buries it.
The failure modes are specific and each is pinned here:

* **Sorting the wrong way for lower-is-better metrics.** Rank a table by
  LPIPS or dE00 with the PSNR rule and the worst model appears first,
  presented as the winner. `is_lower_better` drives both sorting and
  delta signs so "+" always means better.
* **Averages that hide their distribution.** A model can win on the mean
  and lose on most images; `win_loss` counts that explicitly.
* **Ragged columns.** An MoE has gate columns a bilinear baseline does
  not. The table must render both rather than raise.
"""

import os

import pytest

from hdr_eval.report import (
    DEFAULT_COLUMNS,
    build_report,
    delta_table,
    format_table,
    is_lower_better,
    per_image_table,
    summary_rows,
    summary_table,
    win_loss,
    write_report,
)
from hdr_eval.runner import BenchmarkResult


def make_result(name, scores, extra=None, dataset="mobile_hdr", meta=None):
    records = []
    for i, value in enumerate(scores):
        row = {"source_id": f"img{i:02d}", "psnr_mu": value,
               "noisy_psnr_mu": 10.0, "psnr_mu_gain": value - 10.0,
               "delta_e2000": 20.0 - value / 2.0, "time_sec": 0.1}
        if extra:
            row.update(extra)
        records.append(row)
    return BenchmarkResult(name, dataset, records=records, meta=meta or {})


@pytest.fixture
def results():
    return [
        make_result("bilinear", [20.0, 21.0, 22.0]),
        make_result("moe", [25.0, 19.0, 30.0]),
        make_result("wavelet+gbtf", [23.0, 23.0, 23.0]),
    ]


class TestDirectionOfImprovement:
    @pytest.mark.parametrize("column", ["lpips", "delta_e2000", "mae_mu",
                                        "mse", "time_sec"])
    def test_these_are_better_when_smaller(self, column):
        assert is_lower_better(column)

    @pytest.mark.parametrize("column", ["psnr_mu", "ssim_linear",
                                        "psnr_mu_gain", "msssim_mu"])
    def test_these_are_better_when_larger(self, column):
        assert not is_lower_better(column)


class TestFormatTable:
    def test_it_renders_a_header_and_a_row_per_entry(self):
        text = format_table([{"a": 1.0}, {"a": 2.0}], ["a"])
        assert len(text.splitlines()) == 4          # header, rule, two rows

    def test_markdown_mode_emits_pipes_and_a_separator(self):
        text = format_table([{"a": 1.0}], ["a"], markdown=True)
        lines = text.splitlines()
        assert lines[0].startswith("|") and set(lines[1]) <= set("|-: ")

    def test_a_missing_column_becomes_a_dash_not_an_error(self):
        """Models legitimately differ in which columns they produce."""
        text = format_table([{"a": 1.0}, {"b": 2.0}], ["a", "b"])
        assert "—" in text

    def test_nan_is_rendered_as_not_available(self):
        text = format_table([{"a": float("nan")}], ["a"])
        assert "n/a" in text

    def test_decimals_are_configurable(self):
        text = format_table([{"a": 1.23456}], ["a"], decimals=2)
        assert "1.23" in text and "1.2345" not in text

    def test_no_columns_is_rejected(self):
        with pytest.raises(ValueError, match="at least one column"):
            format_table([{"a": 1}], [])

    def test_an_empty_row_list_still_renders_the_header(self):
        text = format_table([], ["psnr_mu"])
        assert "psnr_mu" in text


class TestSummary:
    def test_it_produces_one_row_per_model(self, results):
        rows = summary_rows(results)
        assert len(rows) == 3

    def test_it_sorts_best_first_for_a_higher_is_better_column(self, results):
        rows = summary_rows(results, sort_by="psnr_mu")
        assert rows[0]["model"] == "moe"          # mean 24.67

    def test_it_sorts_the_other_way_for_lower_is_better(self, results):
        rows = summary_rows(results, sort_by="delta_e2000")
        assert rows[0]["model"] == "moe"          # lowest dE00 by construction

    def test_sorting_can_be_disabled(self, results):
        rows = summary_rows(results, sort_by=None)
        assert [r["model"] for r in rows] == [r.model_name for r in results]

    def test_untrained_models_are_labelled_in_the_table(self):
        result = make_result("unet", [15.0], meta={"trainable_untrained": True})
        rows = summary_rows([result])
        assert rows[0]["model"] == "unet (untrained)"

    def test_the_image_count_is_carried(self, results):
        assert summary_rows(results)[0]["images"] == 3

    def test_the_default_columns_are_the_headline_ones(self):
        assert "psnr_mu" in DEFAULT_COLUMNS and "delta_e2000" in DEFAULT_COLUMNS

    def test_the_table_names_every_model(self, results):
        text = summary_table(results)
        for r in results:
            assert r.model_name in text


class TestDeltas:
    def test_the_reference_is_zero_against_itself(self, results):
        text = delta_table(results, "bilinear")
        line = [l for l in text.splitlines() if l.startswith("bilinear")][0]
        assert "0.0000" in line

    def test_a_better_model_shows_a_positive_delta(self, results):
        text = delta_table(results, "bilinear", columns=["psnr_mu"])
        line = [l for l in text.splitlines() if l.startswith("moe")][0]
        assert "-" not in line.split("moe")[1]

    def test_lower_is_better_columns_flip_sign_so_plus_means_better(self, results):
        """dE00 falls as quality rises; the delta must still read positive."""
        text = delta_table(results, "bilinear", columns=["delta_e2000"])
        line = [l for l in text.splitlines() if l.startswith("moe")][0]
        assert "-" not in line.split("moe")[1]

    def test_an_unknown_reference_is_reported(self, results):
        with pytest.raises(KeyError, match="not among the results"):
            delta_table(results, "nonexistent")


class TestWinLoss:
    def test_it_counts_per_image_outcomes(self, results):
        counts = win_loss(results, "bilinear")
        # moe: 25>20 win, 19<21 loss, 30>22 win
        assert counts["moe"] == {"wins": 2, "losses": 1, "ties": 0}

    def test_the_mean_can_disagree_with_the_per_image_record(self, results):
        """The exact case this exists to surface."""
        agg = {r.model_name: r.aggregate()["psnr_mu"] for r in results}
        assert agg["moe"] > agg["bilinear"]
        assert win_loss(results, "bilinear")["moe"]["losses"] > 0

    def test_the_reference_is_not_compared_with_itself(self, results):
        assert "bilinear" not in win_loss(results, "bilinear")

    def test_lower_is_better_columns_count_correctly(self, results):
        counts = win_loss(results, "bilinear", column="delta_e2000")
        assert counts["moe"]["wins"] == 2

    def test_ties_are_counted(self):
        a = make_result("a", [10.0, 10.0])
        b = make_result("b", [10.0, 10.0])
        assert win_loss([a, b], "a")["b"]["ties"] == 2

    def test_an_unknown_reference_is_reported(self, results):
        with pytest.raises(KeyError, match="not among"):
            win_loss(results, "nope")


class TestPerImage:
    def test_it_lists_every_image(self, results):
        text = per_image_table(results[0])
        assert "img00" in text and "img02" in text

    def test_sorting_ascending_surfaces_the_worst_frames(self, results):
        text = per_image_table(results[1], sort_by="psnr_mu", limit=1)
        assert "img01" in text                 # the 19.0 dB frame

    def test_the_limit_truncates(self, results):
        text = per_image_table(results[0], limit=2)
        assert len(text.splitlines()) == 4


class TestBuildReport:
    def test_it_contains_the_summary(self, results):
        text = build_report(results)
        assert "Summary" in text and "moe" in text

    def test_deltas_and_wins_appear_when_a_reference_is_given(self, results):
        text = build_report(results, reference="bilinear")
        assert "Change vs bilinear" in text
        assert "Per-image wins" in text

    def test_they_are_omitted_without_a_reference(self, results):
        text = build_report(results, reference=None)
        assert "Change vs" not in text

    def test_the_worst_frames_section_can_be_switched_off(self, results):
        assert "worst" not in build_report(results, worst_n=0)

    def test_the_run_configuration_is_included_when_recorded(self):
        result = make_result("m", [10.0], meta={"config": {
            "device": "cpu", "gt_demosaic": "gbtf", "pattern": "BGGR",
            "inference": {"mode": "full", "tile": 512, "overlap": 128,
                          "amp_dtype": "torch.bfloat16"},
            "metrics": {"mu": 5000.0}}})
        text = build_report([result])
        assert "gt_demosaic: gbtf" in text and "mu=5000.0" in text

    def test_the_dataset_is_named_in_the_title(self, results):
        assert "mobile_hdr" in build_report(results)

    def test_an_empty_result_list_is_rejected(self):
        with pytest.raises(ValueError, match="No results"):
            build_report([])

    def test_plain_text_mode_has_no_markdown_syntax(self, results):
        text = build_report(results, markdown=False)
        assert "|" not in text and "##" not in text


class TestWriteReport:
    def test_it_writes_and_creates_directories(self, tmp_path, results):
        path = write_report(str(tmp_path / "nested" / "report.md"), results)
        assert os.path.exists(path)
        assert "Summary" in open(path).read()

    def test_the_file_ends_with_a_newline(self, tmp_path, results):
        path = write_report(str(tmp_path / "r.md"), results)
        assert open(path).read().endswith("\n")
