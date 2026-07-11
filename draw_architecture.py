"""
draw_architecture.py — Block diagram of the MoEDenoiser architecture.

Usage:
    python draw_architecture.py              # saves architecture.pdf + architecture.png
    python draw_architecture.py --show       # also opens the figure interactively
"""

import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "input":    "#4A90D9",   # blue
    "trunk":    "#5BA85A",   # green
    "expert":   "#E8A838",   # amber
    "gate":     "#C0392B",   # red
    "blend":    "#8E44AD",   # purple
    "output":   "#2C3E50",   # dark navy
    "arrow":    "#555555",
    "bg":       "#F8F8F8",
    "white":    "#FFFFFF",
}

FONT = "DejaVu Sans"


def box(ax, x, y, w, h, label, sublabel=None, color="#5BA85A",
        fontsize=9, sublabel_fs=7.5, radius=0.015):
    """Draw a rounded rectangle with a centred label (and optional sub-label)."""
    rect = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle=f"round,pad=0.01,rounding_size={radius}",
                          linewidth=1.2, edgecolor="white",
                          facecolor=color, zorder=3, clip_on=False)
    ax.add_patch(rect)
    dy = 0.012 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="white", family=FONT, zorder=4, clip_on=False)
    if sublabel:
        ax.text(x, y - dy - 0.005, sublabel, ha="center", va="center",
                fontsize=sublabel_fs, color="white", alpha=0.88,
                family=FONT, zorder=4, clip_on=False)


def arrow(ax, x0, y0, x1, y1, label=None, color="#555555", lw=1.5,
          label_fs=7.2, label_dx=0.0, label_dy=0.012):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=10),
                zorder=2)
    if label:
        mx, my = (x0 + x1) / 2 + label_dx, (y0 + y1) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=label_fs, color=color, family=FONT, zorder=5,
                bbox=dict(fc=C["bg"], ec="none", pad=1.0))


def line(ax, xs, ys, color="#555555", lw=1.5, ls="-"):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")


def draw():
    fig, ax = plt.subplots(figsize=(13, 9.5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── title ────────────────────────────────────────────────────────────────
    ax.text(0.5, 0.975, "MoEDenoiser — HDR Bayer Denoising Architecture",
            ha="center", va="top", fontsize=13, fontweight="bold",
            color=C["output"], family=FONT)

    # ──────────────────────────────────────────────────────────────────────────
    # Column x-positions
    # ──────────────────────────────────────────────────────────────────────────
    xC = 0.50   # centre spine
    xE0 = 0.28  # expert 0
    xE1 = 0.72  # expert 1
    xG  = 0.50  # gate (centre)

    BW  = 0.18  # standard box width
    BH  = 0.065 # standard box height
    BWS = 0.14  # small box width
    BHS = 0.052 # small box height

    # ── 1. INPUT ─────────────────────────────────────────────────────────────
    y_in = 0.895
    box(ax, xC, y_in, BW, BH,
        "Noisy Bayer Input", "(B, 4, H, W)  ∈ [0,1]",
        color=C["input"])

    # ── 2. SNR MAP branch ────────────────────────────────────────────────────
    y_snr = 0.810
    box(ax, xC, y_snr, BW, BH,
        "SNR Map", "estimate_local_snr_map(x)\n(B, 1, H, W)",
        color=C["input"], sublabel_fs=6.8)
    arrow(ax, xC, y_in - BH/2, xC, y_snr + BH/2)

    # ── 3. SHARED TRUNK ──────────────────────────────────────────────────────
    y_trunk_top = 0.720
    trunk_height = 0.235
    trunk_mid = y_trunk_top - trunk_height / 2

    trunk_rect = FancyBboxPatch((xC - 0.22, trunk_mid - trunk_height/2),
                                0.44, trunk_height,
                                boxstyle="round,pad=0.012,rounding_size=0.015",
                                linewidth=1.5, edgecolor=C["trunk"],
                                facecolor="#EBF5EB", zorder=2, clip_on=False)
    ax.add_patch(trunk_rect)
    ax.text(xC, trunk_mid + trunk_height/2 - 0.018, "Shared Trunk",
            ha="center", va="top", fontsize=10, fontweight="bold",
            color=C["trunk"], family=FONT, zorder=3)

    # sub-blocks inside trunk
    sub_w, sub_h = 0.34, 0.042
    sub_gap = 0.050
    sub_x = xC
    ys = [y_trunk_top - 0.052,
          y_trunk_top - 0.052 - sub_gap,
          y_trunk_top - 0.052 - 2 * sub_gap,
          y_trunk_top - 0.052 - 3 * sub_gap,
          y_trunk_top - 0.052 - 4 * sub_gap - 0.004]

    sub_labels = [
        ("PixelUnshuffle(2)  →  4-ch → 16-ch @ H/2, W/2",   C["trunk"]),
        ("CNN Encoder  ×2  (ResidualConvBlock + PixelUnshuffle)",  C["trunk"]),
        ("Restormer Bottleneck  (MDTA + GDFN)  @  H/8, W/8",      "#3A6B3A"),
        ("CNN Decoder  ×2  (PixelShuffle + skip connections)",     C["trunk"]),
        ("Trunk Features  (B, 64, H/2, W/2)",                     "#2980B9"),
    ]
    for i, (lbl, col) in enumerate(sub_labels):
        box(ax, sub_x, ys[i], sub_w, sub_h, lbl, color=col, fontsize=7.5)
        if i < len(sub_labels) - 1:
            arrow(ax, sub_x, ys[i] - sub_h/2, sub_x, ys[i+1] + sub_h/2)

    arrow(ax, xC, y_snr - BH/2, xC, y_trunk_top)

    # ── 4. SPLIT: trunk features → experts and gate ──────────────────────────
    y_split = trunk_mid - trunk_height/2
    y_heads = y_split - 0.085

    # horizontal split line
    split_y_mid = (y_split + y_heads + BHS/2) / 2 + 0.015
    line(ax, [xE0, xG, xE1], [split_y_mid, split_y_mid, split_y_mid])
    arrow(ax, xC, y_split, xC, split_y_mid + 0.002)
    arrow(ax, xE0, split_y_mid, xE0, y_heads + BHS/2)
    arrow(ax, xG,  split_y_mid, xG,  y_heads + BHS/2)
    arrow(ax, xE1, split_y_mid, xE1, y_heads + BHS/2)

    # ── 5. EXPERT HEADS ──────────────────────────────────────────────────────
    box(ax, xE0, y_heads, BWS, BHS, "Expert Head 0",
        "ResConvBlocks → PixelShuffle(2)\n(B, 4, H, W)",
        color=C["expert"], sublabel_fs=6.5)
    box(ax, xE1, y_heads, BWS, BHS, "Expert Head 1",
        "ResConvBlocks → PixelShuffle(2)\n(B, 4, H, W)",
        color=C["expert"], sublabel_fs=6.5)

    # ── 6. GATE ──────────────────────────────────────────────────────────────
    box(ax, xG, y_heads, BWS, BHS, "NoiseGate",
        "Conv(x ∥ SNR) → Softmax\n(B, K, H, W)",
        color=C["gate"], sublabel_fs=6.5)

    # residual: add noisy input to expert outputs
    y_res = y_heads - 0.075
    box(ax, xE0, y_res, BWS, BHS, "Residual  +  noisy x",
        "expert_out + noisy_input\nclamp(0, 1)",
        color=C["expert"], fontsize=7.5, sublabel_fs=6.3)
    box(ax, xE1, y_res, BWS, BHS, "Residual  +  noisy x",
        "expert_out + noisy_input\nclamp(0, 1)",
        color=C["expert"], fontsize=7.5, sublabel_fs=6.3)
    arrow(ax, xE0, y_heads - BHS/2, xE0, y_res + BHS/2)
    arrow(ax, xE1, y_heads - BHS/2, xE1, y_res + BHS/2)

    # gate spatial blur
    y_blur = y_heads - 0.075
    box(ax, xG, y_blur, BWS, BHS, "Spatial Gate Blur",
        "avg_pool2d(k=33)  +  renorm\nsmooth routing boundaries",
        color=C["gate"], fontsize=7.5, sublabel_fs=6.3)
    arrow(ax, xG, y_heads - BHS/2, xG, y_blur + BHS/2)

    # ── 7. WEIGHTED BLEND ────────────────────────────────────────────────────
    y_blend = y_res - 0.090
    box(ax, xC, y_blend, BW, BH, "Weighted Blend",
        "Σ  gate_k · expert_out_k",
        color=C["blend"])

    arrow(ax, xE0, y_res - BHS/2, xC - BW/4, y_blend + BH/2,
          label="gate₀", label_dx=-0.04, label_dy=0.008, color=C["blend"])
    arrow(ax, xE1, y_res - BHS/2, xC + BW/4, y_blend + BH/2,
          label="gate₁", label_dx=0.04, label_dy=0.008, color=C["blend"])
    arrow(ax, xG,  y_blur - BHS/2, xC, y_blend + BH/2)

    # ── 8. DENOISED BAYER OUTPUT ─────────────────────────────────────────────
    y_bayer_out = y_blend - 0.090
    box(ax, xC, y_bayer_out, BW, BH,
        "Denoised Bayer", "(B, 4, H, W)  ∈ [0,1]",
        color=C["output"])
    arrow(ax, xC, y_blend - BH/2, xC, y_bayer_out + BH/2)

    # ── 9. TEST-TIME ONLY: GBTF → RGB ────────────────────────────────────────
    y_gbtf = y_bayer_out - 0.085
    box(ax, xC, y_gbtf, BW, BH,
        "GBTF Demosaicing", "PixelShuffle(2) → DifferentiableGBTF_BGGR\n(test only)",
        color="#7F8C8D", sublabel_fs=6.5)
    arrow(ax, xC, y_bayer_out - BH/2, xC, y_gbtf + BH/2,
          color="#7F8C8D")

    y_rgb = y_gbtf - 0.085
    box(ax, xC, y_rgb, BW, BH,
        "RGB Output", "(B, 3, 2H, 2W)\nfor metrics & visualisation (test only)",
        color="#7F8C8D", sublabel_fs=6.5)
    arrow(ax, xC, y_gbtf - BH/2, xC, y_rgb + BH/2, color="#7F8C8D")

    # ── LEGEND ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=C["input"],  label="Input / SNR"),
        mpatches.Patch(facecolor=C["trunk"],  label="Shared Trunk"),
        mpatches.Patch(facecolor=C["expert"], label="Expert Heads"),
        mpatches.Patch(facecolor=C["gate"],   label="Routing Gate"),
        mpatches.Patch(facecolor=C["blend"],  label="Weighted Blend"),
        mpatches.Patch(facecolor="#7F8C8D",   label="Test-time only"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              bbox_to_anchor=(0.01, 0.01), fontsize=8,
              framealpha=0.9, edgecolor="#CCCCCC", ncol=3)

    # ── PARAMETER ANNOTATION ─────────────────────────────────────────────────
    ax.text(0.99, 0.01,
            "~20.66 M parameters  |  216 GFLOPs / 512×512 patch  |  K=2 experts",
            ha="right", va="bottom", fontsize=7.5,
            color=C["output"], alpha=0.7, family=FONT)

    plt.tight_layout(pad=0.3)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true",
                        help="Show figure interactively after saving")
    parser.add_argument("--out", default="architecture",
                        help="Output file stem (default: architecture → .pdf + .png)")
    args = parser.parse_args()

    fig = draw()
    for ext in ("pdf", "png"):
        path = f"{args.out}.{ext}"
        dpi = 300 if ext == "png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
