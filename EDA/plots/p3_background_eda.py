"""
EDA/plots/p3_background_eda.py
================================
Background Complexity EDA — ECCV Publication Figures

  Figure 3A:  Background entropy histogram (vs baseline overlay)
  Figure 3B:  Entropy vs Object-count scatter (clean-studio vs complex)

Usage:
    python EDA/plots/p3_background_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/background
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import (
    apply_paper_style, save_fig, add_stat_box, 
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes,
    FILL_ALPHA, LINE_ALPHA, DATASET_LINESTYLES,
)

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# 3A — Background entropy histogram (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_bg_entropy_histogram(
    datasets: Dict[str, np.ndarray],   # {name: bg_entropy (N,)}
    out_dir: str = "figures/background",
    bins: int = 40,
):
    """
    ECCV-style overlaid entropy histogram + KDE.
    Clean dual-panel layout with mean lines.
    Generates both combined and individual plots.
    """
    # ══════════════════════════════════════════════════════════════════════════
    # Combined overlay plot
    # ══════════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.8))

    legend_handles = []
    
    for i, (name, ent) in enumerate(datasets.items()):
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        linestyle = DATASET_LINESTYLES.get(name, "-")
        ent = ent[np.isfinite(ent)]
        
        # Histogram - step style for overlap visibility
        ax1.hist(
            ent, bins=bins, density=True, histtype="step",
            alpha=LINE_ALPHA, color=c, linewidth=1.5
        )
        ax1.hist(
            ent, bins=bins, density=True, histtype="stepfilled",
            alpha=FILL_ALPHA, color=c, edgecolor="none"
        )
        ax1.axvline(ent.mean(), color=c, linestyle="--", linewidth=1.3, alpha=0.9)
        
        # KDE - strong line, subtle fill
        sns.kdeplot(ent, ax=ax2, fill=True, alpha=FILL_ALPHA, 
                    color=c, linewidth=1.8, linestyle=linestyle)
        ax2.axvline(ent.mean(), color=c, linestyle="--", linewidth=1.3, alpha=0.9)
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color=c,
                   markerfacecolor=c, markersize=7,
                   linewidth=1.8, linestyle=linestyle,
                   label=f"{name} (μ={ent.mean():.2f})")
        )

    # Style panels
    for ax, title, label in [
        (ax1, "Histogram", "(a)"),
        (ax2, "Kernel Density Estimate", "(b)"),
    ]:
        ax.set_xlabel("Background Entropy $H_{bg}$ (bits)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        despine_axes(ax)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        add_subplot_label(ax, label, x=-0.12, y=1.08, fontsize=10)

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

    fig.suptitle("Background Complexity — Texture Entropy",
                 fontsize=11, fontweight="bold", y=1.12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, Path(out_dir), "bg_entropy_combined")

    # ══════════════════════════════════════════════════════════════════════════
    # Individual per-dataset plots
    # ══════════════════════════════════════════════════════════════════════════
    for i, (name, ent) in enumerate(datasets.items()):
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        ent = ent[np.isfinite(ent)]
        
        fig_ind, ax_ind = plt.subplots(figsize=(4.0, 3.0))
        
        # Histogram with KDE overlay
        ax_ind.hist(
            ent, bins=bins, density=True, histtype="stepfilled",
            alpha=0.3, color=c, edgecolor=c, linewidth=1.0
        )
        sns.kdeplot(ent, ax=ax_ind, fill=False, color=c, linewidth=2.0)
        ax_ind.axvline(ent.mean(), color=c, linestyle="--", linewidth=1.5, 
                       alpha=0.9, label=f"μ={ent.mean():.3f}")
        ax_ind.axvline(np.median(ent), color=c, linestyle=":", linewidth=1.5,
                       alpha=0.9, label=f"med={np.median(ent):.3f}")
        
        ax_ind.set_xlabel("Background Entropy $H_{bg}$ (bits)", fontsize=10)
        ax_ind.set_ylabel("Density", fontsize=10)
        ax_ind.set_title(f"{name} — Background Entropy", fontsize=11, fontweight="bold")
        ax_ind.legend(loc="upper right", fontsize=8)
        despine_axes(ax_ind)
        ax_ind.yaxis.grid(True, linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        save_fig(fig_ind, Path(out_dir), f"bg_entropy_{name.lower().replace(' ', '_')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3B — Entropy vs Object density scatter (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_entropy_vs_objects(
    datasets_ent: Dict[str, np.ndarray],   # {name: bg_entropy (N,)}
    datasets_obj: Dict[str, np.ndarray],   # {name: bg_obj_count (N,)}
    out_dir: str = "figures/background",
):
    """
    ECCV-style 2D scatter: x = #objects, y = entropy.
    Quadrant annotations for interpretation.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    all_ents = []; all_objs = []
    legend_handles = []
    
    for i, (name, ent) in enumerate(datasets_ent.items()):
        obj = datasets_obj.get(name, np.zeros(len(ent)))
        mask = np.isfinite(ent) & np.isfinite(obj)
        ent_m = ent[mask]; obj_m = obj[mask]
        
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        
        ax.scatter(
            obj_m, ent_m, s=10, alpha=0.45, 
            color=c, marker=marker,
            linewidths=0.3, edgecolors="white",
            rasterized=True,
        )
        
        all_ents.extend(ent_m.tolist())
        all_objs.extend(obj_m.tolist())
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=c, markeredgecolor=c,
                   markersize=7, linewidth=0, label=name)
        )

    # Quadrant annotations (subtle)
    ent_med = np.median(all_ents)
    obj_med = np.median(all_objs)
    
    quadrant_labels = [
        (0.03, 0.97, "Simple", "left", "top"),
        (0.97, 0.97, "Complex", "right", "top"),
        (0.03, 0.03, "Flat", "left", "bottom"),
        (0.97, 0.03, "Cluttered", "right", "bottom"),
    ]
    for (tx, ty, label, ha, va) in quadrant_labels:
        ax.text(tx, ty, label, transform=ax.transAxes,
                ha=ha, va=va, fontsize=7, color="#888888", 
                style="italic", alpha=0.7)

    # Median crosshairs
    ax.axhline(ent_med, color="#cccccc", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axvline(obj_med, color="#cccccc", linestyle=":", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Background Object Count", fontsize=9)
    ax.set_ylabel("Background Entropy $H_{bg}$ (bits)", fontsize=9)
    ax.set_title("Background Complexity", fontsize=10, fontweight="bold", pad=8)
    
    legend = ax.legend(
        handles=legend_handles,
        title="Dataset",
        markerscale=1.2,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
        loc="upper right",
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_fontweight("bold")
    
    despine_axes(ax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)
    
    add_subplot_label(ax, "(c)", x=-0.10, y=1.08, fontsize=10)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "bg_entropy_vs_objects")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/background")
    args = p.parse_args()

    assert len(args.features) == len(args.labels)
    ent_data = {}; obj_data = {}
    for fpath, lbl in zip(args.features, args.labels):
        d = dict(np.load(fpath, allow_pickle=True))
        ent_data[lbl] = d.get("bg_entropy", np.array([]))
        obj_data[lbl] = d.get("bg_obj_count", np.zeros(len(ent_data[lbl])))

    plot_bg_entropy_histogram(ent_data, args.out_dir)
    plot_entropy_vs_objects(ent_data, obj_data, args.out_dir)


if __name__ == "__main__":
    _cli()
