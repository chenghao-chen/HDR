#!/usr/bin/env python
"""
scripts/build_sweep_report.py — LaTeX report for the num-experts sweep
========================================================================

Reads what scripts/save_test_outputs.py and test_dual_MoE_two_phase.py
already wrote for each K in a sweep, and assembles one LaTeX report
comparing them: params, FLOPs, quality, colour fidelity, expert
specialisation and latency, plus a side-by-side figure of the same test
frame denoised by every K.

Usage
-----
    python scripts/build_sweep_report.py --tag colorfix
    python scripts/build_sweep_report.py --tag colorfix --compile
    python scripts/build_sweep_report.py --tag colorfix --experts 1 2 4

Per K, looks for:
    models_p1_moe_K{K}_<tag>_polaris/phase1_best.pth        (params, epoch)
    test_results/models_p1_moe_K{K}_<tag>_polaris/log.txt   (GFLOPs)
    test_visuals/models_p1_moe_K{K}_<tag>_polaris/results.csv, summary.md
    test_visuals/models_p1_moe_K{K}_<tag>_polaris/frame_{FRAME:04d}_panel.jpg

A missing K is skipped with a note in the report rather than failing the
whole build — a sweep member that OOM'd or ran out of walltime should not
block reporting on the ones that finished.

Writes <out>/report.tex, copies the frame figures into <out>/assets/, and
(with --compile) runs tectonic to produce <out>/report.pdf.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from datetime import date

import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.insert(0, REPO_ROOT)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Build a LaTeX report comparing a num-experts sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--tag", required=True,
                   help="HDR_SWEEP_TAG the sweep was submitted with, e.g. "
                        "'colorfix'. Matches models_p1_moe_K{K}_<tag>_polaris/.")
    p.add_argument("--experts", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--frame", type=int, default=21,
                   help="Test-frame index to embed side by side across K "
                        "(same scene -> a fair visual comparison).")
    p.add_argument("--out", default="report",
                   help="Output directory for report.tex / report.pdf / assets/.")
    p.add_argument("--tectonic", default=None,
                   help="Path to the tectonic binary. Default: look on PATH, "
                        "then the sibling 'latex' conda env used to install it.")
    p.add_argument("--compile", action="store_true",
                   help="Also run tectonic to produce report.pdf.")
    return p.parse_args(argv)


def tex_escape(s: str) -> str:
    """Escape the handful of characters LaTeX treats specially in text mode."""
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def find_tectonic(explicit):
    if explicit:
        return explicit
    found = shutil.which("tectonic")
    if found:
        return found
    # The env this project installed it into for exactly this purpose.
    sibling = os.path.join(os.path.dirname(os.path.dirname(
        shutil.which("python3") or shutil.which("python") or "/usr/bin/python3")),
        "envs", "latex", "bin", "tectonic")
    if os.path.isfile(sibling):
        return sibling
    return None


class SweepMember:
    """Everything gathered about one K in the sweep. Fields are None if missing."""

    def __init__(self, k, tag):
        self.k = k
        self.checkpoint = f"models_p1_moe_K{k}_{tag}_polaris/phase1_best.pth"
        self.bench_log = f"test_results/models_p1_moe_K{k}_{tag}_polaris/log.txt"
        self.visuals_dir = f"test_visuals/models_p1_moe_K{k}_{tag}_polaris"
        self.results_csv = os.path.join(self.visuals_dir, "results.csv")
        self.panel = os.path.join(self.visuals_dir, f"frame_{0:04d}_panel.jpg")

        self.present = os.path.isfile(self.checkpoint)
        self.params_m = None
        self.epoch = None
        self.gflops = None
        self.agg = {}          # column -> mean, from results.csv
        self.n_frames = 0

    def load(self, frame_idx):
        if not self.present:
            return
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("model_state_dict", payload)
        self.params_m = sum(t.numel() for t in state.values()
                            if hasattr(t, "numel")) / 1e6
        self.epoch = payload.get("epoch")

        if os.path.isfile(self.bench_log):
            text = open(self.bench_log).read()
            m = re.search(r"GFLOPs\s*/\s*patch:\s*([\d.]+)", text)
            if m:
                self.gflops = float(m.group(1))

        if os.path.isfile(self.results_csv):
            rows = list(csv.DictReader(open(self.results_csv)))
            self.n_frames = len(rows)
            if rows:
                numeric_cols = [c for c in rows[0] if c != "frame"]
                for c in numeric_cols:
                    vals = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
                    if vals:
                        self.agg[c] = sum(vals) / len(vals)

        self.panel = os.path.join(self.visuals_dir, f"frame_{frame_idx:04d}_panel.jpg")
        if not os.path.isfile(self.panel):
            self.panel = None


def build_comparison_table(members):
    present = [m for m in members if m.present]
    if not present:
        return "No sweep member checkpoints were found.\n"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Num-experts sweep, held-out Mobile-HDR test split "
        r"(full resolution, 28 frames unless noted).}",
        r"\label{tab:sweep}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{r r r r r r r r r}",
        r"\toprule",
        r"$K$ & Params (M) & GFLOPs/patch & PSNR-$\mu$ (dB) & SSIM & "
        r"Gain (dB) & Chroma (\%) & Expert spread (dB) & $s$/frame \\",
        r"\midrule",
    ]
    for m in present:
        gflops = f"{m.gflops:.2f}" if m.gflops is not None else "--"
        psnr = f"{m.agg.get('psnr_mu', float('nan')):.2f}"
        ssim = f"{m.agg.get('ssim', float('nan')):.4f}"
        gain = f"{m.agg.get('gain_db', float('nan')):+.2f}"
        chroma = f"{100 * m.agg.get('chroma_ratio', float('nan')):.1f}"
        sec = f"{m.agg.get('seconds', float('nan')):.3f}"

        if m.k > 1:
            expert_psnrs = [m.agg[f"expert{j}_psnr_mu"] for j in range(m.k)
                           if f"expert{j}_psnr_mu" in m.agg]
            spread = f"{max(expert_psnrs) - min(expert_psnrs):.2f}" if expert_psnrs else "--"
        else:
            spread = "--"

        lines.append(f"{m.k} & {m.params_m:.2f} & {gflops} & {psnr} & {ssim} & "
                     f"{gain} & {chroma} & {spread} & {sec} \\\\")
    missing = [m.k for m in members if not m.present]
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    if missing:
        lines.append(r"\par\noindent\textit{Note: K = %s "
                     r"had no checkpoint at report time (still training, "
                     r"walltime-limited, or failed) and is omitted above.}"
                     % ", ".join(str(k) for k in missing))
    return "\n".join(lines) + "\n"


def build_gate_table(members):
    with_gate = [m for m in members if m.present and m.k > 1]
    if not with_gate:
        return ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-expert routing usage. A collapsed router sends one "
        r"expert's mean gate weight near 0; a fully engaged $K$-way router "
        r"spreads weight near $1/K$ each with real per-pixel variation.}",
        r"\label{tab:gates}",
        r"\small",
        r"\resizebox{0.6\textwidth}{!}{%",
        r"\begin{tabular}{r " + "r " * 4 + r"}",
        r"\toprule",
        r"$K$ & \multicolumn{4}{c}{mean gate weight, expert 0..3} \\",
        r"\midrule",
    ]
    for m in with_gate:
        cells = []
        for j in range(4):
            v = m.agg.get(f"expert{j}_gate_mean")
            cells.append(f"{v:.3f}" if v is not None and j < m.k else "--")
        lines.append(f"{m.k} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def build_pgfplot(members):
    present = [m for m in members if m.present and "psnr_mu" in m.agg]
    if len(present) < 2:
        return ""
    psnr_coords = " ".join(f"({m.k},{m.agg['psnr_mu']:.3f})" for m in present)
    lat_coords = " ".join(
        f"({m.k},{1000 * m.agg.get('seconds', 0):.2f})" for m in present)
    return r"""
\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{axis}[
    width=0.46\textwidth, height=4.2cm,
    xlabel={$K$ (experts)}, ylabel={PSNR-$\mu$ (dB)},
    xtick=data, grid=major, title={Quality vs.\ $K$},
]
\addplot[mark=*, thick] coordinates {%s};
\end{axis}
\end{tikzpicture}
\hfill
\begin{tikzpicture}
\begin{axis}[
    width=0.46\textwidth, height=4.2cm,
    xlabel={$K$ (experts)}, ylabel={ms / frame},
    xtick=data, grid=major, title={Latency vs.\ $K$},
]
\addplot[mark=square*, thick, color=orange!80!black] coordinates {%s};
\end{axis}
\end{tikzpicture}
\caption{Quality and full-resolution inference latency as a function of
expert count. Latency includes the trunk (shared across every $K$) plus
$K$ expert heads evaluated at every pixel regardless of routing.}
\label{fig:k-tradeoff}
\end{figure}
""" % (psnr_coords, lat_coords)


def build_frame_figure(members, out_dir, frame_idx):
    present = [m for m in members if m.present and m.panel]
    if not present:
        return ""
    assets_dir = os.path.join(out_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # One standalone \begin{figure} per K, not one figure crammed with N
    # subfigures: the real panels are wide (noisy | predicted | reference on
    # top, per-expert + error below) but not short, and N of them stacked in
    # a single float can overflow the page silently rather than erroring.
    # Separate floats let LaTeX's normal placement algorithm break across
    # pages wherever it needs to.
    blocks = []
    for m in present:
        dest = f"K{m.k}_frame{frame_idx:04d}_panel.jpg"
        shutil.copyfile(m.panel, os.path.join(assets_dir, dest))
        blocks.append(r"""
\begin{figure}[p]
\centering
\includegraphics[width=0.92\textwidth]{assets/%s}
\caption{Test frame %d, $K=%d$ (%.2fM params, PSNR-$\mu$=%.2f~dB,
chroma=%.0f\%%). Top row: noisy $\vert$ predicted $\vert$ reference.
Bottom row: each expert's own output $\vert$ tone-mapped $|$error$|$.}
\label{fig:frame-k%d}
\end{figure}""" % (dest, frame_idx, m.k, m.params_m,
                    m.agg.get("psnr_mu", float("nan")),
                    100 * m.agg.get("chroma_ratio", float("nan")), m.k))

    return "\n".join(blocks) + "\n"


def build_architecture_diagram():
    """
    Box-and-arrow TikZ diagram of MoEDenoiser, matching
    HDR_model_hybrid_Teacher.py's actual module wiring (patch_embed ->
    encoder_level_{1,2} -> latent (Restormer) -> decoder_level_{2,1}, two
    skip connections, then K expert heads blended by a gate conditioned on
    the raw input and the SNR map) rather than a generic encoder-decoder
    sketch. Static content: nothing here depends on the sweep's data, only
    on the architecture, so it does not vary by K.
    """
    return r"""
\begin{figure}[htbp]
\centering
\resizebox{0.95\textwidth}{!}{%
\begin{tikzpicture}[
    node distance=3mm and 8mm,
    box/.style={draw, rounded corners, minimum width=32mm, minimum height=7mm,
                align=center, font=\scriptsize, inner sep=1pt, fill=blue!4},
    small/.style={draw, rounded corners, minimum width=24mm, minimum height=5mm,
                align=center, font=\scriptsize, inner sep=1pt, fill=gray!8},
    op/.style={draw, circle, minimum size=5mm, font=\small, inner sep=0pt},
    io/.style={draw, rounded corners, minimum width=32mm, minimum height=6mm,
               align=center, font=\scriptsize\bfseries, inner sep=1pt, fill=orange!12},
    arr/.style={-{Latex[length=2mm]}, thick},
]

% ---- Trunk: encoder -> Restormer latent -> decoder ----
\node[io]  (input)    {Packed BGGR\\ $[4,H,W]$};
\node[box, below=of input]   (embed)   {PixelUnshuffle(2) + Conv\\ patch embed $\to \ndim{}$};
\node[box, below=of embed]   (enc1)    {Encoder L1\\ ResidualBlocks + SE\\ $\ndim{},\ H/2$};
\node[small, below=of enc1]  (down1)   {PixelUnshuffle(2)};
\node[box, below=of down1]   (enc2)    {Encoder L2\\ ResidualBlocks + SE\\ $4\ndim{},\ H/4$};
\node[small, below=of enc2]  (down2)   {PixelUnshuffle(2)};
\node[box, below=of down2, fill=violet!8] (latent) {Restormer Latent\\ transformer blocks\\ $16\ndim{},\ H/8$};
\node[small, below=of latent] (up2)    {PixelShuffle(2)};
\node[box, below=of up2]     (dec2)    {Decoder L2\\ ResidualBlocks\\ $\to 4\ndim{},\ H/4$};
\node[small, below=of dec2]  (up1)     {PixelShuffle(2)};
\node[box, below=of up1, fill=green!8] (dec1) {Decoder L1 (trunk output)\\ ResidualBlocks\\ $2\ndim{},\ H/2$};

\draw[arr] (input) -- (embed);
\draw[arr] (embed) -- (enc1);
\draw[arr] (enc1)  -- (down1);
\draw[arr] (down1) -- (enc2);
\draw[arr] (enc2)  -- (down2);
\draw[arr] (down2) -- (latent);
\draw[arr] (latent) -- (up2);
\draw[arr] (up2)   -- (dec2);
\draw[arr] (dec2)  -- (up1);
\draw[arr] (up1)   -- (dec1);

% Skip connections (concatenated, not summed) -- enc2 bows in tighter than
% enc1, which has to clear it, matching a standard U-Net skip drawing.
\draw[arr, dashed] (enc2.east) to[bend left=35] node[right, font=\tiny, xshift=1mm]
    {skip (concat)} (dec2.east);
\draw[arr, dashed] (enc1.east) to[bend left=55] node[right, font=\tiny, xshift=6mm]
    {skip (concat)} (dec1.east);

% ---- K expert heads, fed from the trunk output ----
\node[box, below=8mm of dec1, xshift=-38mm, fill=yellow!10] (e0) {Expert head 0\\ ResBlocks + $2\times$PixelShuffle\\ $\to$ RGB $[3,2H,2W]$};
\node[box, below=8mm of dec1, fill=yellow!10] (e1) {Expert head 1\\ (same topology)};
\node[font=\scriptsize, align=center, below=8mm of dec1, xshift=38mm] (edots) {$\cdots$\\ Expert head $K{-}1$};

\draw[arr] (dec1) -- ++(0,-6mm) -| (e0);
\draw[arr] (dec1) -- (e1);
\draw[arr] (dec1) -- ++(0,-6mm) -| (edots);

% ---- Gate: a separate small path, NOT fed through the trunk ----
\node[io, right=22mm of input, yshift=-3mm] (noisyin) {Noisy BGGR\\ $[4,H,W]$};
\node[io, below=4mm of noisyin] (snrin) {Local SNR map\\ $[1,H,W]$};
\node[box, below=5mm of snrin, fill=red!8] (gate) {NoiseGate\\ 2 conv layers $\to$ softmax\\ $[K,H,W]$};
\node[small, below=4mm of gate] (upgate) {Bilinear $\times 2$\\ $[K,2H,2W]$};

\draw[arr] (noisyin) -- (snrin);
\draw[arr] (noisyin) |- (gate);
\draw[arr] (snrin)  -- (gate);
\draw[arr] (gate)   -- (upgate);

% ---- Blend ----
\node[op, below=13mm of e1] (sum) {$\sum$};
\draw[arr] (e0) -- ++(0,-8mm) -| (sum);
\draw[arr] (e1) -- (sum);
\draw[arr] (edots) -- ++(0,-8mm) -| (sum);
\draw[arr] (upgate.south) |- node[below, font=\tiny, pos=0.85] {per-pixel weights} (sum.east);

\node[io, below=6mm of sum] (out) {Blended RGB\\ $[3,2H,2W]$};
\draw[arr] (sum) -- (out);

\end{tikzpicture}}
\caption{\texttt{MoEDenoiser} architecture. One shared trunk (patch embed
$\to$ two ResidualBlock/SE encoder levels $\to$ a Restormer transformer
latent $\to$ two ResidualBlock decoder levels with concatenated skip
connections) produces one feature map, consumed identically by $K$
lightweight expert heads. A separate NoiseGate path — seeing the raw noisy
Bayer input and a local SNR estimate directly, not the trunk features —
produces a per-pixel softmax over the $K$ experts, upsampled $2\times$ and
used to blend the experts' sensor-resolution RGB outputs. Every expert
head shares the trunk's cost; only the lightweight heads (each
$\approx$150K parameters at $\ndim{}=32$) scale with $K$, which is the
architectural basis for the params/FLOPs-vs-$K$ figures in
Table~\ref{tab:sweep}.}
\label{fig:architecture}
\end{figure}
"""


def build_recommendation(members):
    present = [m for m in members if m.present and "psnr_mu" in m.agg]
    if len(present) < 2:
        return "Not enough completed sweep members to compare."
    best_psnr = max(present, key=lambda m: m.agg["psnr_mu"])
    best_efficiency = min(
        present, key=lambda m: m.params_m / max(m.agg["psnr_mu"], 1e-6))
    psnr_span = max(m.agg["psnr_mu"] for m in present) - \
        min(m.agg["psnr_mu"] for m in present)

    spreads = {}
    for m in present:
        if m.k > 1:
            vals = [m.agg[f"expert{j}_psnr_mu"] for j in range(m.k)
                   if f"expert{j}_psnr_mu" in m.agg]
            if vals:
                spreads[m.k] = max(vals) - min(vals)

    lines = [
        f"Across $K \\in \\{{{', '.join(str(m.k) for m in present)}\\}}$, "
        f"PSNR-$\\mu$ spans {psnr_span:.2f}~dB "
        f"(best: $K={best_psnr.k}$ at {best_psnr.agg['psnr_mu']:.2f}~dB).",
    ]
    if spreads:
        widest_k = max(spreads, key=spreads.get)
        lines.append(
            f"Per-expert specialisation (the spread between each expert's "
            f"own solo PSNR) is widest at $K={widest_k}$ "
            f"({spreads[widest_k]:.2f}~dB) — the clearest sign of any "
            f"expert count actually differentiating its experts rather than "
            f"learning near-duplicate functions.")
    lines.append(
        f"By params-per-dB-of-quality, $K={best_efficiency.k}$ is the most "
        f"parameter-efficient point tested "
        f"({best_efficiency.params_m:.2f}M / {best_efficiency.agg['psnr_mu']:.2f}~dB).")
    if psnr_span < 0.5:
        lines.append(
            "The quality spread across $K$ is small enough "
            "(under 0.5~dB) that expert count is not, on this run, "
            "a strong quality lever — the efficiency argument for a "
            "smaller $K$ is not paying much of a quality tax.")
    return " ".join(lines)


TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{microtype}
\newcommand{\ndim}{d}

\title{HDR Denoising: Bug-Fix and Mixture-of-Experts Sweep Report}
\author{HDR Denoising Project \\ \small ALCF Polaris (lighthouse-purdue allocation)}
\date{%(date)s}

\begin{document}
\maketitle

\begin{abstract}
This report covers a debugging pass on the joint denoising/demosaicing
Mixture-of-Experts (MoE) pipeline, followed by a controlled sweep over the
number of experts $K \in \{%(k_list)s\}$, trained in parallel across the four
GPUs of one Polaris node and evaluated on the held-out Mobile-HDR test
split. Five defects were found and fixed prior to this sweep: the model
could not learn at all (a zero-initialisation bug gave every parameter
exactly zero gradient); the MoE router had collapsed to uniform routing
regardless of input; the classical GBTF demosaicer used to build ground
truth wrapped incorrectly at image borders; the evaluation harness crashed
on GPU; and the training-time data augmentation silently corrupted the
color-filter-array phase on roughly three quarters of samples, driving the
model to predict near-grayscale output. The last of these is verified
fixed in this report (Section~\ref{sec:color}); the $K$ sweep in
Section~\ref{sec:sweep} is trained entirely with the corrected pipeline.
\end{abstract}

\tableofcontents

\section{Background}
The project's stated goal is an HDR denoising model that is fast and
lightweight at inference time while remaining competitive in quality. The
architecture under test is a shared convolutional/Restormer trunk feeding
$K$ lightweight expert heads, blended by a per-pixel gate conditioned on
noisy input and a local SNR estimate — a mixture-of-experts (MoE) design
intended to let different experts specialise on different noise regimes
without each carrying the full trunk's cost.

Before this pass, the model could not train: every parameter received
exactly zero gradient at initialisation (a zero-initialised output layer
sat on a clamp floor with zero backward gradient below it). Once that was
fixed, the router was found to have collapsed to near-uniform routing
regardless of the input, and the ground-truth reference used for every
metric in this document — produced by demosaicing clean sensor data with a
differentiable GBTF implementation — was found to wrap incorrectly at
image borders and to carry a small systematic bias from an imprecise
kernel constant. All of these were fixed prior to the work in this
report; see the project's \texttt{BENCHMARKING.md} and git history for
the full technical detail on each.

\subsection{Model architecture}
Figure~\ref{fig:architecture} diagrams \texttt{MoEDenoiser} directly from
its module wiring: one shared trunk, $K$ identical-topology expert heads
reading that trunk's output, and a gate that reads the raw noisy input and
SNR map \emph{directly} rather than through the trunk. The practical
consequence for this report is in the caption: only the expert heads
scale with $K$, so params and FLOPs grow slowly with expert count while
the trunk's cost is fixed.

%(architecture_diagram)s

\section{The color/augmentation bug}
\label{sec:color}
The most consequential remaining defect, diagnosed from the per-frame test
figures rather than any aggregate metric: trained checkpoints produced
visibly near-grayscale output. Measured directly, a fully-trained (50
epoch) checkpoint retained only 16\%% of the reference image's chroma
(mean $|{\rm channel} - {\rm luma}|$, prediction vs.\ ground truth), and
predicted red and blue channels that were numerically almost identical.

The root cause was in training-time data augmentation. The project's D4
(dihedral) augmentation flips and rotates training patches for
generalisation, and must relabel each 2$\times$2 Bayer cell's channel
assignment to match — a horizontal flip of a BGGR sensor pattern produces
GBRG, not BGGR, for instance. The augmentation in use did not correctly
track this: only 27\%% of its draws round-tripped back to the true BGGR
phase (verified directly against all eight dihedral symmetries), while the
ground-truth RGB reference was, on every sample, demosaiced assuming BGGR
regardless. On the mislabelled 73\%% of samples, the network was
therefore trained against a target where red and blue were arbitrarily
swapped or scrambled relative to its actual input. The loss-minimising
response to that contradiction is to predict red $\approx$ blue
$\approx$ their average — exactly the desaturation observed.

The fix swaps in an existing, already-tested augmentation
(\texttt{hdr\_data.D4Transform(mode="cell")}) that moves each Bayer cell as
a spatial unit rather than permuting channels, which keeps the CFA phase
exactly BGGR on every draw (verified: 200/200). It required no change to
the model architecture. Checked directly on a mid-training checkpoint
(epoch 6 of 50) from this sweep: chroma retention rose from 16\%% to
100.8\%%, and the red/blue channel means — identical before the fix — became
genuinely distinct. The per-$K$ frame figures near the end of this report
(one per expert count, same test scene throughout) show this qualitatively.

\section{Num-experts sweep}
\label{sec:sweep}
$K \in \{%(k_list)s\}$ trained simultaneously, one process per GPU
(\texttt{CUDA\_VISIBLE\_DEVICES} pinned), all with the corrected
augmentation, 50 epochs of Phase~1 patch training (512$\times$512 packed
patches, batch 8). Evaluated on the full held-out test split at full
sensor resolution.

%(comparison_table)s

%(gate_table)s

%(pgfplot)s

\subsection{Reading the table}
%(recommendation)s

Two caveats on comparing $K$ directly to the pre-sweep single-$K=2$ runs
in this project's earlier history: those were trained {\it before} the
augmentation fix, so their absolute numbers are not comparable to this
table, and the per-expert-PSNR / gate-usage columns should be read
alongside \texttt{summary.md}'s automatic near-duplicate-expert and
collapsed-router warnings in each run's own
\texttt{test\_visuals/models\_p1\_moe\_K\{K\}\_%(tag_escaped)s\_polaris/}
directory, which flag exactly this failure mode per run.

%(frame_figure)s

\section{Reproducing this report}
\begin{verbatim}
qsub -v HDR_SWEEP_TAG=%(tag)s scripts/polaris/sweep_experts.pbs
qsub -v HDR_SWEEP_TAG=%(tag)s scripts/polaris/sweep_experts_eval.pbs
python scripts/build_sweep_report.py --tag %(tag)s --compile
\end{verbatim}
Per-frame figures, per-image metrics (\texttt{results.csv}) and a
plain-language summary (\texttt{summary.md}, including automatic
collapsed-router / near-duplicate-expert / desaturation warnings) for
every $K$ are under \texttt{test\_visuals/models\_p1\_moe\_K\{K\}\_%(tag_escaped)s\_polaris/}.

\section{Limitations and next steps}
\begin{itemize}
\item This sweep is Phase~1 only (512$\times$512 patches); Phase~2
full-resolution fine-tuning was not run for any $K$ here and is where the
project's own notes flag likely out-of-memory risk on Polaris' 40~GB A100s.
\item Expert specialisation (Table~\ref{tab:sweep}'s spread column,
Table~\ref{tab:gates}'s per-expert gate usage) should be checked against
each run's own \texttt{summary.md} warning before concluding a given $K$ is
``worth'' its extra parameters — a wide PSNR spread with near-uniform gate
usage indicates the router is not exploiting the specialisation the experts
did learn, which is a different problem from experts that never
differentiated in the first place.
\item Two smaller, previously-documented bugs remain unfixed and out of
scope here: an even \texttt{window\_size} in the SNR-map estimator
under-pads by one pixel (unreachable in production, since every call site
uses \texttt{window\_size=5}), and a \texttt{do\_expand} noise-augmentation
option can saturate an entire synthetic frame (dormant, since it defaults
off and is never enabled by the training script).
\end{itemize}

\end{document}
"""


def main(argv=None):
    args = parse_args(argv)
    os.chdir(REPO_ROOT)

    members = [SweepMember(k, args.tag) for k in args.experts]
    for m in members:
        m.load(args.frame)

    n_present = sum(m.present for m in members)
    print(f"sweep tag '{args.tag}': {n_present}/{len(members)} checkpoints found")
    for m in members:
        status = f"epoch {m.epoch}, {m.params_m:.2f}M params" if m.present else "MISSING"
        print(f"  K={m.k}: {status}")

    if n_present == 0:
        print("Nothing to report — no checkpoints found for this tag.")
        return 1

    os.makedirs(args.out, exist_ok=True)

    doc = TEMPLATE % {
        "date": date.today().isoformat(),
        "k_list": ", ".join(str(m.k) for m in members),
        "comparison_table": build_comparison_table(members),
        "architecture_diagram": build_architecture_diagram(),
        "gate_table": build_gate_table(members),
        "pgfplot": build_pgfplot(members),
        "recommendation": build_recommendation(members),
        "frame_figure": build_frame_figure(members, args.out, args.frame),
        "tag": args.tag,
        "tag_escaped": tex_escape(args.tag),
    }

    tex_path = os.path.join(args.out, "report.tex")
    with open(tex_path, "w") as fh:
        fh.write(doc)
    print(f"\nwrote {tex_path}")

    if args.compile:
        tectonic = find_tectonic(args.tectonic)
        if not tectonic:
            print("--compile requested but no tectonic binary found "
                 "(checked PATH and the sibling 'latex' conda env).")
            return 1
        print(f"compiling with {tectonic} ...")
        result = subprocess.run(
            [tectonic, "report.tex"], cwd=args.out,
            capture_output=True, text=True, timeout=300)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            print(f"tectonic failed (exit {result.returncode})")
            return 1
        pdf = os.path.join(args.out, "report.pdf")
        if os.path.isfile(pdf):
            print(f"wrote {pdf}")
        else:
            print("tectonic exited 0 but report.pdf is missing — inspect the "
                 "output above.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
