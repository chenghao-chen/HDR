"""
hdr_eval/report.py — turning results into something readable
=============================================================

A benchmark that produces a CSV per model and stops has not finished the
job: the question is always "which is better, by how much, and where".
This module renders one or many :class:`~hdr_eval.runner.BenchmarkResult`
objects as console tables and Markdown, with the comparison made explicit:

* a **summary table**, one row per model, sorted by a chosen column;
* a **delta table** against a named reference, so "+1.8 dB over GBTF" is
  read off rather than computed by the reader;
* a **per-image table** for one model, which is where the outliers that
  move an average become visible;
* a **win/loss count**, because a model that is better on average and
  worse on two thirds of images is a different result from one that is
  better on all of them.

Higher-is-better and lower-is-better columns are tracked by name, so
sorting and delta signs come out right for LPIPS and dE00 as well as PSNR.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .runner import BenchmarkResult

__all__ = [
    "LOWER_IS_BETTER",
    "is_lower_better",
    "format_table",
    "summary_rows",
    "summary_table",
    "delta_table",
    "per_image_table",
    "win_loss",
    "build_report",
    "write_report",
    "DEFAULT_COLUMNS",
]

#: Column-name fragments whose metrics improve as they get smaller.
LOWER_IS_BETTER = ("lpips", "delta_e", "mae", "mse", "time_sec", "loss")

#: The columns a summary shows unless told otherwise.
DEFAULT_COLUMNS = ("psnr_mu", "psnr_linear", "ssim_mu", "delta_e2000",
                   "psnr_mu_gain", "time_sec")


def is_lower_better(column: str) -> bool:
    """Whether smaller values of `column` are better."""
    name = column.lower()
    return any(frag in name for frag in LOWER_IS_BETTER)


def _fmt(value: Any, decimals: int = 4) -> str:
    """Render one cell: numbers to fixed precision, None as an em dash."""
    if value is None:
        return "—"
    if isinstance(value, float):
        if value != value:
            return "n/a"
        return f"{value:.{decimals}f}"
    return str(value)


def format_table(rows: Sequence[Dict[str, Any]],
                 columns: Sequence[str], *, markdown: bool = False,
                 decimals: int = 4, align_right: bool = True) -> str:
    """
    Render a list of dicts as a fixed-width or Markdown table.

    Missing keys become an em dash rather than raising, so results from
    models with different column sets (an MoE has gate columns, a
    bilinear baseline does not) can share a table.
    """
    if not columns:
        raise ValueError("Need at least one column.")
    header = [str(c) for c in columns]
    body = [[_fmt(row.get(c), decimals) for c in columns] for row in rows]
    widths = [max(len(header[i]), *(len(r[i]) for r in body)) if body
              else len(header[i]) for i in range(len(columns))]

    def line(cells: Sequence[str], pad: str = " ") -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(widths[i], pad) if align_right and i
                       else cell.ljust(widths[i], pad))
        return ("| " + " | ".join(out) + " |") if markdown else "  ".join(out)

    lines = [line(header)]
    if markdown:
        lines.append("|" + "|".join(
            ("-" * (w + 2)) if i == 0 else (":" + "-" * (w + 1))
            for i, w in enumerate(widths)) + "|")
    else:
        lines.append("-" * len(lines[0]))
    lines.extend(line(r) for r in body)
    return "\n".join(lines)


def summary_rows(results: Sequence[BenchmarkResult],
                 columns: Sequence[str] = DEFAULT_COLUMNS,
                 sort_by: Optional[str] = "psnr_mu") -> List[Dict[str, Any]]:
    """One aggregated row per result, optionally sorted."""
    rows: List[Dict[str, Any]] = []
    for res in results:
        agg = res.aggregate()
        row: Dict[str, Any] = {"model": res.model_name, "images": res.num_images}
        for col in columns:
            row[col] = agg.get(col)
        if res.meta.get("trainable_untrained"):
            row["model"] = f"{res.model_name} (untrained)"
        rows.append(row)

    if sort_by:
        def key(r):
            v = r.get(sort_by)
            if v is None or v != v:
                return float("inf")
            return v if is_lower_better(sort_by) else -v
        rows.sort(key=key)
    return rows


def summary_table(results: Sequence[BenchmarkResult],
                  columns: Sequence[str] = DEFAULT_COLUMNS,
                  sort_by: Optional[str] = "psnr_mu",
                  markdown: bool = False) -> str:
    """One row per model, best first."""
    rows = summary_rows(results, columns, sort_by)
    return format_table(rows, ["model", "images", *columns], markdown=markdown)


def delta_table(results: Sequence[BenchmarkResult], reference: str,
                columns: Sequence[str] = DEFAULT_COLUMNS,
                markdown: bool = False) -> str:
    """
    Every model's aggregate minus the reference model's.

    Signs are normalised so that a positive delta always means "better",
    including for the lower-is-better columns.
    """
    by_name = {r.model_name: r.aggregate() for r in results}
    if reference not in by_name:
        raise KeyError(
            f"Reference model '{reference}' is not among the results: "
            f"{sorted(by_name)}")
    ref = by_name[reference]

    rows = []
    for res in results:
        agg = by_name[res.model_name]
        row: Dict[str, Any] = {"model": res.model_name}
        for col in columns:
            a, b = agg.get(col), ref.get(col)
            if a is None or b is None or a != a or b != b:
                row[col] = None
                continue
            diff = a - b
            row[col] = -diff if is_lower_better(col) else diff
        rows.append(row)
    return format_table(rows, ["model", *columns], markdown=markdown)


def per_image_table(result: BenchmarkResult,
                    columns: Sequence[str] = ("source_id", "psnr_mu",
                                              "noisy_psnr_mu",
                                              "psnr_mu_gain", "time_sec"),
                    limit: Optional[int] = None, sort_by: Optional[str] = None,
                    markdown: bool = False) -> str:
    """
    The per-image rows, optionally sorted and truncated.

    Sorting by ``psnr_mu`` ascending and limiting to ten is the fastest
    way to find the frames a model is failing on.
    """
    rows = list(result.records)
    if sort_by:
        rows.sort(key=lambda r: (r.get(sort_by) is None, r.get(sort_by)))
    if limit is not None:
        rows = rows[:limit]
    return format_table(rows, list(columns), markdown=markdown)


def win_loss(results: Sequence[BenchmarkResult], reference: str,
             column: str = "psnr_mu") -> Dict[str, Dict[str, int]]:
    """
    Per-image wins, losses and ties against the reference, by `column`.

    Images are matched on ``source_id`` when present and on position
    otherwise, so this is only meaningful for runs over the same dataset
    in the same order — which is what the CLI produces.
    """
    def keyed(res: BenchmarkResult) -> Dict[Any, float]:
        out: Dict[Any, float] = {}
        for i, rec in enumerate(res.records):
            key = rec.get("source_id", i)
            value = rec.get(column)
            if value is not None and value == value:
                out[key] = float(value)
        return out

    by_name = {r.model_name: keyed(r) for r in results}
    if reference not in by_name:
        raise KeyError(
            f"Reference model '{reference}' is not among the results: "
            f"{sorted(by_name)}")
    ref = by_name[reference]
    lower = is_lower_better(column)

    out: Dict[str, Dict[str, int]] = {}
    for name, values in by_name.items():
        if name == reference:
            continue
        wins = losses = ties = 0
        for key, value in values.items():
            if key not in ref:
                continue
            better = value < ref[key] if lower else value > ref[key]
            worse = value > ref[key] if lower else value < ref[key]
            wins += int(better)
            losses += int(worse)
            ties += int(not better and not worse)
        out[name] = {"wins": wins, "losses": losses, "ties": ties}
    return out


def build_report(results: Sequence[BenchmarkResult],
                 columns: Sequence[str] = DEFAULT_COLUMNS,
                 reference: Optional[str] = None,
                 sort_by: str = "psnr_mu",
                 worst_n: int = 5, markdown: bool = True) -> str:
    """
    The whole report as one string: summary, deltas, win/loss, worst frames.
    """
    if not results:
        raise ValueError("No results to report.")
    dataset = results[0].dataset_name
    parts: List[str] = []
    head = "# " if markdown else ""
    parts.append(f"{head}HDR benchmark — {dataset}")
    parts.append("")
    parts.append(f"{len(results)} model(s), "
                 f"{results[0].num_images} image(s) each, "
                 f"sorted by {sort_by}.")
    parts.append("")
    parts.append(("## " if markdown else "") + "Summary")
    parts.append("")
    parts.append(summary_table(results, columns, sort_by, markdown))
    parts.append("")

    if reference and len(results) > 1:
        parts.append(("## " if markdown else "")
                     + f"Change vs {reference} (positive = better)")
        parts.append("")
        parts.append(delta_table(results, reference, columns, markdown))
        parts.append("")

        parts.append(("## " if markdown else "")
                     + f"Per-image wins vs {reference} ({sort_by})")
        parts.append("")
        wl = win_loss(results, reference, sort_by)
        rows = [{"model": k, **v} for k, v in wl.items()]
        parts.append(format_table(rows, ["model", "wins", "losses", "ties"],
                                  markdown=markdown))
        parts.append("")

    if worst_n:
        for res in results:
            parts.append(("### " if markdown else "")
                         + f"{res.model_name}: worst {worst_n} frames by {sort_by}")
            parts.append("")
            parts.append(per_image_table(
                res, limit=worst_n, sort_by=sort_by, markdown=markdown))
            parts.append("")

    meta = results[0].meta.get("config", {})
    if meta:
        parts.append(("## " if markdown else "") + "Run configuration")
        parts.append("")
        for key in ("device", "gt_demosaic", "pattern", "limit"):
            if key in meta:
                parts.append(f"- {key}: {meta[key]}")
        inf = meta.get("inference", {})
        if inf:
            parts.append(f"- inference: {inf.get('mode')} "
                         f"(tile={inf.get('tile')}, overlap={inf.get('overlap')}, "
                         f"amp={inf.get('amp_dtype')})")
        mets = meta.get("metrics", {})
        if mets:
            parts.append(f"- tone curve: mu-law, mu={mets.get('mu')}")
        parts.append("")
    return "\n".join(parts)


def write_report(path: str, results: Sequence[BenchmarkResult], **kwargs) -> str:
    """Render :func:`build_report` to a file. Returns the path."""
    text = build_report(results, **kwargs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
        if not text.endswith("\n"):
            fh.write("\n")
    return path
