"""
EDA/plots/p2_occlusion_eda.py
==============================
Occlusion EDA — ECCV Publication Figures

  Figure 2A:  Occlusion ratio histogram (per dataset) with baseline overlay
  Figure 2B:  Spatial occlusion heatmap (mean occlusion mask aggregated)

Usage:
    python EDA/plots/p2_occlusion_eda.py \
        --features eda_cache/viton_features.npz eda_cache/dresscode_features.npz \
        --labels VITON DressCode --out_dir figures/occlusion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import (
    apply_paper_style, save_fig, add_stat_box, 
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes, get_cmap_for_heatmap,
)

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# 2A — Occlusion ratio histogram (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_occlusion_histogram(
    datasets: Dict[str, np.ndarray],   # {name: occ_ratios (N,)}
    out_dir: str = "figures/occlusion",
    bins: int = 40,
):
    """
    ECCV-style overlaid histogram + KDE of occlusion ratios.
    - Maximum visibility for overlapping distributions
    - Step histograms with light fills
    - Strong KDE lines with subtle fills
    - Distinct colors and line styles per dataset
    """
    from plot_style import (
        FILL_ALPHA, LINE_ALPHA, DATASET_LINESTYLES,
        plot_overlapping_kde, plot_overlapping_histogram
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.8))
    ax_hist, ax_kde = axes

    legend_handles = []
    
    for i, (name, occ) in enumerate(datasets.items()):
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        linestyle = DATASET_LINESTYLES.get(name, "-")
        occ = occ[np.isfinite(occ)]
        
        # ── Histogram panel (step style for visibility) ──────────────────
        # Step outline (always visible)
        ax_hist.hist(
            occ, bins=bins, histtype="step",
            color=color, density=True,
            linewidth=1.5, linestyle=linestyle,
            alpha=LINE_ALPHA,
        )
        # Light fill for context
        ax_hist.hist(
            occ, bins=bins, histtype="stepfilled",
            alpha=FILL_ALPHA * 0.5, color=color, density=True,
            linewidth=0,
        )
        ax_hist.axvline(
            occ.mean(), color=color, linestyle="--", 
            linewidth=1.3, alpha=0.9, zorder=10
        )

        # ── KDE panel (strong lines, subtle fill) ────────────────────────
        sns.kdeplot(
            occ, ax=ax_kde, fill=True, 
            alpha=FILL_ALPHA,  # Low alpha for overlap visibility
            color=color, 
            linewidth=1.8,     # Strong border line
            linestyle=linestyle,
        )
        ax_kde.axvline(
            occ.mean(), color=color, linestyle="--", 
            linewidth=1.3, alpha=0.9, zorder=10
        )
        
        # Build legend handle with marker + line style
        legend_handles.append(
            Line2D([0], [0], marker=marker, color=color,
                   markerfacecolor=color, markersize=7,
                   linewidth=2.0, linestyle=linestyle,
                   label=f"{name} (μ={occ.mean():.2f})")
        )

    # Style histogram panel
    ax_hist.set_xlabel("Occlusion Ratio $O_i$", fontsize=9)
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_hist.set_title("Histogram", fontsize=10, fontweight="bold", pad=6)
    ax_hist.set_xlim(-0.02, 1.02)
    despine_axes(ax_hist)
    ax_hist.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax_hist.set_axisbelow(True)
    add_subplot_label(ax_hist, "(a)", x=-0.12, y=1.08, fontsize=10)

    # Style KDE panel
    ax_kde.set_xlabel("Occlusion Ratio $O_i$", fontsize=9)
    ax_kde.set_ylabel("Density", fontsize=9)
    ax_kde.set_title("Kernel Density Estimate", fontsize=10, fontweight="bold", pad=6)
    ax_kde.set_xlim(-0.02, 1.02)
    despine_axes(ax_kde)
    ax_kde.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax_kde.set_axisbelow(True)
    add_subplot_label(ax_kde, "(b)", x=-0.12, y=1.08, fontsize=10)

    # Shared legend
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(datasets), 5),
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
    )

    fig.suptitle("Garment Occlusion Distribution", fontsize=11,
                 fontweight="bold", y=1.12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, Path(out_dir), "occlusion_combined")

    # ══════════════════════════════════════════════════════════════════════════
    # Individual plots per dataset (for detailed analysis)
    # ══════════════════════════════════════════════════════════════════════════
    for i, (name, occ) in enumerate(datasets.items()):
        color = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        occ = occ[np.isfinite(occ)]
        
        fig_ind, ax_ind = plt.subplots(figsize=(4.0, 3.0))
        
        # Histogram with KDE overlay
        ax_ind.hist(
            occ, bins=bins, histtype="stepfilled",
            alpha=0.3, color=color, density=True,
            edgecolor=color, linewidth=1.0,
        )
        sns.kdeplot(
            occ, ax=ax_ind, fill=False,
            color=color, linewidth=2.0,
        )
        ax_ind.axvline(
            occ.mean(), color=color, linestyle="--",
            linewidth=1.5, alpha=0.9, label=f"μ={occ.mean():.3f}"
        )
        ax_ind.axvline(
            np.median(occ), color=color, linestyle=":",
            linewidth=1.5, alpha=0.9, label=f"med={np.median(occ):.3f}"
        )
        
        ax_ind.set_xlabel("Occlusion Ratio $O_i$", fontsize=10)
        ax_ind.set_ylabel("Density", fontsize=10)
        ax_ind.set_title(f"{name} — Occlusion Distribution", fontsize=11, fontweight="bold")
        ax_ind.set_xlim(-0.02, 1.02)
        ax_ind.legend(loc="upper right", fontsize=8)
        despine_axes(ax_ind)
        ax_ind.yaxis.grid(True, linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        save_fig(fig_ind, Path(out_dir), f"occlusion_{name.lower().replace(' ', '_')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2B — Spatial occlusion heatmap (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_occlusion_heatmap(
    datasets: Dict[str, np.ndarray],   # {name: occ_maps (N, H, W)}
    out_dir: str = "figures/occlusion",
):
    """
    ECCV-style spatial occlusion heatmaps.
    - Consistent colorbar scaling
    - Clean subplot titles
    - Peak location markers
    """
    n = len(datasets)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(
        rows, cols, 
        figsize=(2.5 * cols, 2.8 * rows),
        squeeze=False
    )
    axes = axes.flatten()

    # Use perceptually uniform colormap (better for print)
    cmap = "YlOrRd"  # Yellow-Orange-Red, prints well in grayscale
    
    # Find global max for consistent scaling
    global_max = max(d.mean(axis=0).max() for d in datasets.values())
    vmax = min(global_max, 0.6)

    for i, (name, maps) in enumerate(datasets.items()):
        mean_map = maps.mean(axis=0)    # (H, W)
        ax = axes[i]
        
        im = ax.imshow(
            mean_map, cmap=cmap, vmin=0.0, vmax=vmax,
            aspect="equal", origin="upper",
            interpolation="bilinear",
        )
        
        # Add colorbar
        cbar = fig.colorbar(
            im, ax=ax, fraction=0.046, pad=0.04,
            format="%.2f"
        )
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Freq.", fontsize=7, rotation=270, labelpad=10)
        
        ax.set_title(name, fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])

        # Mark peak occlusion location
        peak_y, peak_x = np.unravel_index(mean_map.argmax(), mean_map.shape)
        ax.plot(
            peak_x, peak_y, 
            marker="+", color="cyan", 
            markersize=10, markeredgewidth=2,
            zorder=10
        )
        
        # Add panel label
        if i == 0:
            add_subplot_label(ax, f"(c)", x=-0.05, y=1.05, fontsize=10)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Spatial Occlusion Heatmap ($\\bar{M}_{occ}$)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "occlusion_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/occlusion")
    args = p.parse_args()

    assert len(args.features) == len(args.labels)
    occ_ratios = {}; occ_maps = {}
    for f, lbl in zip(args.features, args.labels):
        d = dict(np.load(f, allow_pickle=True))
        occ_ratios[lbl] = d["occ_ratios"]
        occ_maps[lbl]   = d["occ_maps"]

    plot_occlusion_histogram(occ_ratios, args.out_dir)
    plot_occlusion_heatmap(occ_maps,   args.out_dir)


if __name__ == "__main__":
    _cli()
