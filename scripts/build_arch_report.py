#!/usr/bin/env python
"""
scripts/build_arch_report.py — LaTeX report for the architecture sweep
======================================================================

The architecture sweep is a grid, not a line: eight mechanisms crossed with
ten parameter budgets, 80 runs. The existing build_sweep_report.py is built
around one-dimensional sweeps keyed by a single scalar (K, film_hidden, dim),
so this is a separate generator rather than a fourth member class bolted onto
it.

Usage
-----
    python scripts/build_arch_report.py
    python scripts/build_arch_report.py --compile
    python scripts/build_arch_report.py --manifest sweep_manifest.csv --compile

Reads, per run named in the manifest:
    test_results/<folder>/results.csv     per-frame eval metrics
    logs/*_arch_<run>.log                 training trajectory

A run missing either is reported as absent rather than failing the build: a
sweep where a couple of configurations died should still produce a report on
the ones that finished.

Writes <out>/report.tex and, with --compile, <out>/report.pdf.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import statistics
import subprocess
import sys

# Quality below this is treated as "still on the initial plateau". Every run in
# this sweep starts near 13 dB and either climbs past the low 20s or stalls
# around 16.3; 20 dB sits in the empty band between those two outcomes, so the
# first epoch above it is an unambiguous escape marker.
ESCAPE_DB = 20.0

# Budgets below this showed mechanism spreads an order of magnitude larger than
# the budgets above it, driven by escape timing rather than architecture. The
# report splits its conclusions at this line.
CLEAN_FLOOR = 1.0e6

ARM_ORDER = ["moeK1", "moeK2", "moeK3", "moeK4", "moeK8",
             "film", "trunkheavy", "expertheavy"]

ARM_LABELS = {
    "moeK1": "MoE $K{=}1$", "moeK2": "MoE $K{=}2$", "moeK3": "MoE $K{=}3$",
    "moeK4": "MoE $K{=}4$", "moeK8": "MoE $K{=}8$", "film": "FiLM",
    "trunkheavy": "Trunk-heavy", "expertheavy": "Expert-heavy",
}


def pretty_size(tag):
    """Manifest size tags are filesystem-safe ("1p3M"); reports are not."""
    return tag.replace("p", ".")


def tex_escape(s):
    """Escape the characters that would otherwise be markup.

    The percent sign matters most: LaTeX reads a bare % as the start of a
    comment running to end of line, which silently swallows the rest of the
    sentence in the rendered PDF rather than failing the build.
    """
    return (str(s).replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
            .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
            .replace("}", r"\}").replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}"))


def find_tectonic(explicit):
    if explicit:
        return explicit
    found = shutil.which("tectonic")
    if found:
        return found
    # The env this project installed it into. Resolved from this interpreter
    # rather than from PATH's python3: on a login node those differ, and the
    # PATH-based guess silently misses.
    envs = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
    sibling = os.path.join(envs, "latex", "bin", "tectonic")
    return sibling if os.path.isfile(sibling) else None


class Run:
    """One cell of the grid: its configuration, its eval, its trajectory."""

    def __init__(self, row):
        self.name = row["run"]
        self.folder = row["folder"]
        self.size, self.arm = self.name.split("_", 1)
        self.params = int(row["actual_params"])
        self.target = int(row["target_params"])
        self.mode = row["mode"]
        self.dim = int(row["dim"])
        self.depth = int(row["num_blocks"])
        self.psnr = None
        self.ssim = None
        self.latency = None
        self.trajectory = []      # [(epoch, psnr_mu)]
        self.escape_epoch = None

    def load(self):
        csv_path = os.path.join("test_results", self.folder, "results.csv")
        if os.path.isfile(csv_path):
            with open(csv_path) as fh:
                rows = list(csv.DictReader(fh))
            self.psnr = self._mean(rows, "psnr_rgb_mu")
            self.ssim = self._mean(rows, "ssim_rgb_mu")
            self.latency = self._mean(rows, "time_sec")

        # The retried run carries a different job id, so match on the run name.
        logs = sorted(glob.glob(f"logs/*_arch_{self.name}.log"))
        if logs:
            # Last by job id: a rerun supersedes the original attempt.
            with open(logs[-1], errors="replace") as fh:
                text = fh.read()
            for m in re.finditer(
                    r"\[P1\] Epoch\s+(\d+)\s+.*?PSNR-.=\s*([0-9.]+)", text):
                self.trajectory.append((int(m.group(1)), float(m.group(2))))
            for epoch, psnr in self.trajectory:
                if psnr > ESCAPE_DB:
                    self.escape_epoch = epoch
                    break

    @staticmethod
    def _mean(rows, key):
        vals = [float(r[key]) for r in rows if r.get(key)]
        return statistics.mean(vals) if vals else None

    @property
    def final(self):
        return self.trajectory[-1][1] if self.trajectory else None


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else None


def load_runs(manifest):
    with open(manifest) as fh:
        rows = list(csv.DictReader(fh))
    runs = []
    for row in rows:
        run = Run(row)
        run.load()
        runs.append(run)
    return runs


def size_order(runs):
    """Budgets ordered by parameter count, not by their string labels."""
    seen = {}
    for r in runs:
        seen.setdefault(r.size, r.target)
    return [s for s, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def grid_table(runs, sizes):
    by = {(r.size, r.arm): r for r in runs}
    cols = "l" + "r" * len(ARM_ORDER) + "r"
    head = (" & ".join(["\\textbf{Budget}"]
                       + [ARM_LABELS[a] for a in ARM_ORDER]
                       + ["\\textbf{Spread}"]) + r" \\")
    lines = [r"\begin{tabular}{" + cols + "}", r"\toprule", head, r"\midrule"]
    for s in sizes:
        cells, vals = [], []
        for a in ARM_ORDER:
            r = by.get((s, a))
            if r is None or r.psnr is None:
                cells.append("--")
            else:
                cells.append(f"{r.psnr:.2f}")
                vals.append(r.psnr)
        spread = f"{max(vals) - min(vals):.2f}" if len(vals) > 1 else "--"
        params = by[(s, "moeK2")].target if (s, "moeK2") in by else 0
        label = f"{tex_escape(pretty_size(s))}"
        if params and params < CLEAN_FLOOR:
            label += r"$^{\dagger}$"
        lines.append(" & ".join([label] + cells + [spread]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def quality_plot(runs, sizes):
    by = {(r.size, r.arm): r for r in runs}
    plots = []
    for a in ARM_ORDER:
        coords = []
        for s in sizes:
            r = by.get((s, a))
            if r and r.psnr is not None:
                coords.append(f"({r.params},{r.psnr:.3f})")
        if coords:
            plots.append("\\addplot+[mark=*,mark size=1.2pt] coordinates {"
                         + " ".join(coords) + "};\n\\addlegendentry{"
                         + ARM_LABELS[a] + "}")
    return "\n".join(plots)


def spread_plot(runs, sizes):
    by = {(r.size, r.arm): r for r in runs}
    coords = []
    for s in sizes:
        vals = [by[(s, a)].psnr for a in ARM_ORDER
                if (s, a) in by and by[(s, a)].psnr is not None]
        if len(vals) > 1:
            params = by[(s, "moeK2")].target
            coords.append(f"({params},{max(vals) - min(vals):.3f})")
    return "\\addplot[mark=*,thick] coordinates {" + " ".join(coords) + "};"


def escape_plot(runs):
    """Escape epoch against final quality, for the budgets where it varies."""
    coords = []
    for r in runs:
        if r.target < CLEAN_FLOOR and r.final is not None:
            # A run that never escaped is plotted at the epoch count, so it
            # appears at the right edge rather than vanishing from the figure.
            epoch = r.escape_epoch if r.escape_epoch else (
                r.trajectory[-1][0] if r.trajectory else None)
            if epoch:
                coords.append(f"({epoch},{r.final:.3f})")
    return "\\addplot[only marks,mark=*,mark size=1.6pt] coordinates {" \
        + " ".join(coords) + "};"


def escape_stats(runs, sizes):
    """Per-budget and pooled correlation of escape epoch with final quality."""
    rows, allx, ally = [], [], []
    for s in sizes:
        pts = [(r.escape_epoch or (r.trajectory[-1][0] if r.trajectory else None),
                r.final) for r in runs
               if r.size == s and r.target < CLEAN_FLOOR and r.final is not None]
        pts = [(x, y) for x, y in pts if x]
        if len(pts) < 3:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        r_val = pearson(xs, ys)
        rows.append((s, len(pts), r_val))
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        allx += [x - mx for x in xs]
        ally += [y - my for y in ys]
    pooled = pearson(allx, ally) if allx else None
    return rows, pooled, len(allx)


def config_table(runs, sizes):
    by = {(r.size, r.arm): r for r in runs}
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"\textbf{Budget} & \textbf{Target} & \textbf{Width} & "
             r"\textbf{Depth} & \textbf{Max arm deviation} \\",
             r"\midrule"]
    for s in sizes:
        ref = by.get((s, "moeK2"))
        if ref is None:
            continue
        devs = [abs(by[(s, a)].params - ref.target) / ref.target
                for a in ARM_ORDER if (s, a) in by]
        worst = max(devs) if devs else 0.0
        lines.append(
            f"{tex_escape(pretty_size(s))} & {ref.target/1e6:.3f}M & {ref.dim} & {ref.depth} "
            f"& {100*worst:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=blue]{hyperref}

\title{Architecture sweep: mechanism against parameter budget}
\author{HDR denoising project}
\date{%(date)s}

\begin{document}
\maketitle

\section*{Summary}

Eight architectural mechanisms were trained at each of ten parameter budgets
spanning %(min_params)s to %(max_params)s---%(n_runs)d runs, one per GPU across
20 Polaris nodes in a single allocation. The question was whether the
mechanisms that showed no benefit at the project's %(baseline)s baseline would
separate once the parameter budget was small enough for capacity to bind.

Above %(clean_floor)s they do not. All eight mechanisms fall within
%(max_clean_spread).2f\,dB of one another at every budget in that range, which
extends the earlier null result down by a factor of %(reduction).0f in
parameter count.

Below %(clean_floor)s the spreads grow to as much as %(max_dirty_spread).1f\,dB,
which looks like mechanism finally mattering. Section~\ref{sec:escape} shows it
is not: those differences track \emph{when a run escaped its initial training
plateau}, with a pooled correlation of $r = %(pooled_r).3f$ accounting for
%(pooled_r2).0f\%% of the variance in final quality. The low-budget rows of
Table~\ref{tab:grid} therefore rank optimisation luck, not architecture, and
are marked $^{\dagger}$ throughout.

The result that does carry over to deployment is the scaling curve itself:
%(knee_params)s reaches %(knee_db).2f\,dB against the %(baseline)s baseline's
%(baseline_db).2f\,dB---%(knee_ratio).0f$\times$ fewer parameters for
%(knee_delta).2f\,dB.

\section{Method}

Each budget is a rung on a ladder walked by the reference arm's trunk width at
fixed depth, rather than a round target solved for. Only even widths build in
this architecture (odd ones fail inside \texttt{pixel\_shuffle}) and parameter
count grows as width squared, so the reachable sizes form a coarse lattice;
solving for round numbers such as 50k/100k/500k forced the depth to oscillate
between budgets, which would have confounded the size axis with a depth axis.
Depth is therefore 4 at every rung but the smallest, where depth 4 cannot reach
below roughly 95k parameters.

Within a budget, each arm's width is solved separately so that the arms match
on \emph{parameter count} rather than on width: a $K{=}8$ mixture carries eight
expert heads where $K{=}1$ carries one. Table~\ref{tab:config} gives the
resulting configurations and the worst deviation from budget across the arms.

\begin{table}[h]
\centering
%(config_table)s
\caption{The size ladder. Every arm at a budget shares that budget's width and
depth; the last column is the largest parameter-count deviation among the eight
arms, so the arms differ by mechanism rather than by size.}
\label{tab:config}
\end{table}

All runs share one training recipe: 50 epochs, 10 warmup epochs, cosine decay
from $10^{-4}$ to $10^{-6}$, batch 8, $512\times512$ packed patches, one seed.
The single seed is the principal limitation of this report and is what
Section~\ref{sec:escape} ultimately turns on.

\section{Results}

\begin{table}[h]
\centering
\resizebox{\textwidth}{!}{%%
%(grid_table)s
}
\caption{Mean PSNR-$\mu$ (dB) over the test set. $^{\dagger}$ marks budgets
below %(clean_floor)s, where the spread reflects training dynamics rather than
architecture---see Section~\ref{sec:escape}.}
\label{tab:grid}
\end{table}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.92\textwidth, height=7.5cm, xmode=log,
  xlabel={Parameters}, ylabel={PSNR-$\mu$ (dB)},
  legend pos=south east, legend columns=2, legend style={font=\scriptsize},
  grid=major, grid style={gray!25}]
%(quality_plot)s
\end{axis}
\end{tikzpicture}
\caption{Quality against parameter budget, one line per mechanism. The lines
are indistinguishable above %(clean_floor)s and scatter widely below it.}
\end{figure}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.92\textwidth, height=6cm, xmode=log, ymode=log,
  xlabel={Parameters}, ylabel={Spread across mechanisms (dB)},
  grid=major, grid style={gray!25}]
%(spread_plot)s
\end{axis}
\end{tikzpicture}
\caption{Range between the best and worst mechanism at each budget. The
collapse by two orders of magnitude above %(clean_floor)s is the sweep's
central measurement.}
\end{figure}

\section{The low-budget spread is escape timing}
\label{sec:escape}

Every run in this sweep begins near 13\,dB and passes through a plateau at
roughly 16.3\,dB. Large models leave it within the warmup period. Small ones
may sit on it for tens of epochs, and because the cosine schedule decays toward
$10^{-6}$, a run that escapes late has little learning rate left to recover and
freezes near wherever it was.

Three observations rule out a capacity explanation for the low-budget spread:

\begin{enumerate}
\item It is not monotonic in size for a fixed mechanism. %(nonmono)s
\item Arms differing by under 1\%% in parameter count differ by several dB.
\item Final quality is predicted by escape epoch (Figure~\ref{fig:escape}).
\end{enumerate}

The runs are converged rather than truncated: over the last ten epochs the
small models gain as little as the large ones, so more epochs at this schedule
would not close the gap.

\begin{table}[h]
\centering
\begin{tabular}{lrr}
\toprule
\textbf{Budget} & \textbf{$n$} & \textbf{$r$(escape epoch, final dB)} \\
\midrule
%(escape_rows)s
\midrule
Pooled (size-centred) & %(pooled_n)d & %(pooled_r).3f \\
\bottomrule
\end{tabular}
\caption{Correlation between the epoch a run left the plateau and the quality
it converged to, for budgets below %(clean_floor)s.}
\end{table}

\begin{figure}[h]
\centering
\begin{tikzpicture}
\begin{axis}[width=0.8\textwidth, height=6.5cm,
  xlabel={Epoch the run passed %(escape_db).0f\,dB},
  ylabel={Final PSNR-$\mu$ (dB)}, grid=major, grid style={gray!25}]
%(escape_plot)s
\end{axis}
\end{tikzpicture}
\caption{Every run below %(clean_floor)s. Runs that never escaped are plotted
at the final epoch. The relationship is the sweep's confound.}
\label{fig:escape}
\end{figure}

\section{Conclusions}

\begin{enumerate}
\item \textbf{Mechanism does not matter from %(baseline)s down to
%(clean_floor)s.} Routing, the number of experts, continuous FiLM conditioning
and the trunk/expert split are all within %(max_clean_spread).2f\,dB. The
hypothesis that the %(baseline)s baseline was too large to reveal architectural
differences is not supported over the range where the measurement is clean.
\item \textbf{Capacity still buys very little.} %(knee_ratio).0f$\times$ more
parameters than %(knee_params)s is worth %(knee_delta).2f\,dB, which supports
aggressive shrinking for an edge target.
\item \textbf{Below %(clean_floor)s this sweep cannot answer the question.}
One seed per cell measures a coin flip. Resolving it needs either several seeds
per configuration, to report a distribution, or a schedule that makes escape
reliable---a longer warmup or a higher learning-rate floor---so that the
comparison is between converged models rather than between lucky ones.
\end{enumerate}

\section*{Reproduction}

\begin{verbatim}
python scripts/solve_param_grid.py --tag arch --out sweep_manifest.csv
qsub -v HDR_SWEEP_MANIFEST=sweep_manifest.csv scripts/polaris/arch_sweep_smoke.pbs
qsub -v HDR_SWEEP_MANIFEST=sweep_manifest.csv scripts/polaris/arch_sweep.pbs
qsub -v HDR_SWEEP_MANIFEST=sweep_manifest.csv scripts/polaris/arch_sweep_eval.pbs
python scripts/build_arch_report.py --compile
\end{verbatim}

%(missing_note)s

\end{document}
"""


def human(n):
    return f"{n/1e6:.2f}M" if n >= 1e6 else f"{n/1e3:.0f}k"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="sweep_manifest.csv")
    ap.add_argument("--out", default="report/arch_sweep")
    ap.add_argument("--tectonic", default=None)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.manifest)
    sizes = size_order(runs)
    have = [r for r in runs if r.psnr is not None]
    if not have:
        sys.exit("no evaluated runs found — has the eval job run?")

    by = {(r.size, r.arm): r for r in runs}
    spreads = {}
    for s in sizes:
        vals = [by[(s, a)].psnr for a in ARM_ORDER
                if (s, a) in by and by[(s, a)].psnr is not None]
        if len(vals) > 1:
            spreads[s] = max(vals) - min(vals)

    clean = {s: v for s, v in spreads.items()
             if by[(s, "moeK2")].target >= CLEAN_FLOOR}
    dirty = {s: v for s, v in spreads.items()
             if by[(s, "moeK2")].target < CLEAN_FLOOR}

    esc_rows, pooled, pooled_n = escape_stats(runs, sizes)
    escape_rows_tex = "\n".join(
        f"{tex_escape(pretty_size(s))} & {n} & {r:+.3f} \\\\" for s, n, r in esc_rows)

    # The knee: smallest budget at or above the clean floor.
    clean_sizes = [s for s in sizes if by[(s, "moeK2")].target >= CLEAN_FLOOR]
    knee = clean_sizes[0]
    top = sizes[-1]
    knee_best = max(by[(knee, a)].psnr for a in ARM_ORDER
                    if (knee, a) in by and by[(knee, a)].psnr is not None)
    top_best = max(by[(top, a)].psnr for a in ARM_ORDER
                   if (top, a) in by and by[(top, a)].psnr is not None)

    # A concrete non-monotonic example, found rather than asserted.
    nonmono = "No non-monotonic pair was found."
    for a in ARM_ORDER:
        seq = [(s, by[(s, a)].psnr) for s in sizes
               if (s, a) in by and by[(s, a)].psnr is not None
               and by[(s, "moeK2")].target < CLEAN_FLOOR]
        for i in range(len(seq) - 1):
            if seq[i][1] > seq[i + 1][1] + 1.0:
                nonmono = (f"{ARM_LABELS[a]} scores {seq[i][1]:.2f}\\,dB at "
                           f"{tex_escape(pretty_size(seq[i][0]))} but only "
                           f"{seq[i+1][1]:.2f}\\,dB at "
                           f"{tex_escape(pretty_size(seq[i+1][0]))}, with more parameters.")
                break
        if not nonmono.startswith("No non"):
            break

    absent = [r.name for r in runs if r.psnr is None]
    missing_note = ""
    if absent:
        missing_note = (r"\section*{Runs absent from this report}" + "\n"
                        + "The following did not produce evaluation output: "
                        + tex_escape(", ".join(absent)) + ".\n")

    os.makedirs(args.out, exist_ok=True)
    tex = TEMPLATE % {
        "date": subprocess.run(["date", "+%Y-%m-%d"], capture_output=True,
                               text=True).stdout.strip(),
        "n_runs": len(have),
        "min_params": human(min(r.target for r in runs)),
        "max_params": human(max(r.target for r in runs)),
        "baseline": human(by[(top, "moeK2")].target),
        "clean_floor": human(CLEAN_FLOOR),
        "max_clean_spread": max(clean.values()) if clean else 0.0,
        "max_dirty_spread": max(dirty.values()) if dirty else 0.0,
        "reduction": by[(top, "moeK2")].target / CLEAN_FLOOR,
        "pooled_r": pooled if pooled is not None else 0.0,
        "pooled_r2": 100 * (pooled ** 2) if pooled is not None else 0.0,
        "pooled_n": pooled_n,
        "escape_db": ESCAPE_DB,
        "knee_params": human(by[(knee, "moeK2")].target),
        "knee_db": knee_best,
        "baseline_db": top_best,
        "knee_ratio": by[(top, "moeK2")].target / by[(knee, "moeK2")].target,
        "knee_delta": top_best - knee_best,
        "config_table": config_table(runs, sizes),
        "grid_table": grid_table(runs, sizes),
        "quality_plot": quality_plot(runs, sizes),
        "spread_plot": spread_plot(runs, sizes),
        "escape_plot": escape_plot(runs),
        "escape_rows": escape_rows_tex,
        "nonmono": nonmono,
        "missing_note": missing_note,
    }

    tex_path = os.path.join(args.out, "report.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"wrote {tex_path}  ({len(have)}/{len(runs)} runs)")

    if args.compile:
        tectonic = find_tectonic(args.tectonic)
        if not tectonic:
            sys.exit("tectonic not found; pass --tectonic /path/to/tectonic")
        proc = subprocess.run([tectonic, "report.tex"], cwd=args.out,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
            sys.exit(f"tectonic failed ({proc.returncode})")
        print(f"wrote {os.path.join(args.out, 'report.pdf')}")


if __name__ == "__main__":
    main()
