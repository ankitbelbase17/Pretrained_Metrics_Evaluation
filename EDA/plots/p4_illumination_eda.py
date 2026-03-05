"""
EDA/plots/p4_illumination_eda.py
=================================
Illumination EDA — ECCV Publication Figures

  Figure 4A:  Luminance spectrum (distribution of mean-L per image)
  Figure 4B:  PCA of illumination maps → 2D scatter revealing lighting modes

Usage:
    python EDA/plots/p4_illumination_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/illumination
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import (
    apply_paper_style, save_fig, add_stat_box, 
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes,
    FILL_ALPHA, LINE_ALPHA, DATASET_LINESTYLES,
)

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# 4A — Luminance spectrum (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_luminance_spectrum(
    datasets: Dict[str, np.ndarray],   # {name: lum_mean (N,)}
    datasets_grad: Dict[str, np.ndarray],  # {name: lum_grad_var (N,)}
    out_dir: str = "figures/illumination",
):
    """
    ECCV-style luminance distribution with regime bands.
    Panel 1: Mean luminance KDE
    Panel 2: Gradient variance distribution
    Generates both combined and individual plots.
    """
    # ══════════════════════════════════════════════════════════════════════════
    # Combined overlay plot
    # ══════════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.8))

    legend_handles = []
    
    for i, (name, lum) in enumerate(datasets.items()):
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        linestyle = DATASET_LINESTYLES.get(name, "-")
        lum = lum[np.isfinite(lum)]
        
        # Strong line, subtle fill for overlap visibility
        sns.kdeplot(lum, ax=ax1, fill=True, alpha=FILL_ALPHA, 
                    color=c, linewidth=1.8, linestyle=linestyle)
        ax1.axvline(lum.mean(), color=c, linestyle="--", linewidth=1.3, alpha=0.9)

        gv = datasets_grad.get(name, np.array([]))
        gv = gv[np.isfinite(gv)]
        if len(gv) > 2:
            sns.kdeplot(gv, ax=ax2, fill=True, alpha=FILL_ALPHA,
                        color=c, linewidth=1.8, linestyle=linestyle)
            ax2.axvline(gv.mean(), color=c, linestyle="--", linewidth=1.3, alpha=0.9)
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color=c,
                   markerfacecolor=c, markersize=7,
                   linewidth=1.8, linestyle=linestyle, label=name)
        )

    # Regime bands for luminance (subtle background shading)
    regime_bands = [
        (0.0, 0.3, "Dark", "#E8E8E8", 0.5),
        (0.3, 0.7, "Mid",  "#F5F5F5", 0.3),
        (0.7, 1.0, "Bright", "#FFFDE7", 0.5),
    ]
    for (xl, xr, label, color, alpha) in regime_bands:
        ax1.axvspan(xl, xr, alpha=alpha, color=color, zorder=0)
        # Label at top
        ax1.text(
            (xl + xr) / 2, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 0.95,
            label, ha="center", fontsize=7, color="#666666", 
            style="italic", alpha=0.8,
            transform=ax1.get_xaxis_transform()
        )

    ax1.set_xlabel("Mean Luminance $L_i$ (normalized)", fontsize=9)
    ax1.set_ylabel("Density", fontsize=9)
    ax1.set_title("Luminance Distribution", fontsize=10, fontweight="bold", pad=6)
    ax1.set_xlim(0, 1)
    despine_axes(ax1)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax1.set_axisbelow(True)
    add_subplot_label(ax1, "(a)", x=-0.12, y=1.08, fontsize=10)

    ax2.set_xlabel("Gradient Variance $\\mathrm{Var}(||\\nabla I||)$", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("Illumination Gradient Variance", fontsize=10, fontweight="bold", pad=6)
    despine_axes(ax2)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax2.set_axisbelow(True)
    add_subplot_label(ax2, "(b)", x=-0.12, y=1.08, fontsize=10)

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

    fig.suptitle("Illumination Complexity Analysis",
                 fontsize=11, fontweight="bold", y=1.12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, Path(out_dir), "luminance_combined")

    # ══════════════════════════════════════════════════════════════════════════
    # Individual per-dataset plots
    # ══════════════════════════════════════════════════════════════════════════
    for i, (name, lum) in enumerate(datasets.items()):
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        lum = lum[np.isfinite(lum)]
        gv = datasets_grad.get(name, np.array([]))
        gv = gv[np.isfinite(gv)]
        
        # Individual dual-panel figure for this dataset
        fig_ind, (ax_l, ax_g) = plt.subplots(1, 2, figsize=(6.0, 2.5))
        
        # Luminance KDE
        sns.kdeplot(lum, ax=ax_l, fill=True, alpha=0.35, color=c, linewidth=2.0)
        ax_l.axvline(lum.mean(), color=c, linestyle="--", linewidth=1.5, 
                     alpha=0.9, label=f"μ={lum.mean():.3f}")
        ax_l.axvline(np.median(lum), color=c, linestyle=":", linewidth=1.5,
                     alpha=0.9, label=f"med={np.median(lum):.3f}")
        ax_l.set_xlabel("Mean Luminance $L_i$", fontsize=9)
        ax_l.set_ylabel("Density", fontsize=9)
        ax_l.set_title("Luminance", fontsize=10, fontweight="bold")
        ax_l.set_xlim(0, 1)
        ax_l.legend(fontsize=7, loc="upper right")
        despine_axes(ax_l)
        ax_l.yaxis.grid(True, linestyle="--", alpha=0.3)
        
        # Gradient variance KDE
        if len(gv) > 2:
            sns.kdeplot(gv, ax=ax_g, fill=True, alpha=0.35, color=c, linewidth=2.0)
            ax_g.axvline(gv.mean(), color=c, linestyle="--", linewidth=1.5,
                         alpha=0.9, label=f"μ={gv.mean():.3f}")
            ax_g.axvline(np.median(gv), color=c, linestyle=":", linewidth=1.5,
                         alpha=0.9, label=f"med={np.median(gv):.3f}")
        ax_g.set_xlabel("Gradient Variance", fontsize=9)
        ax_g.set_ylabel("Density", fontsize=9)
        ax_g.set_title("Illumination Gradient", fontsize=10, fontweight="bold")
        ax_g.legend(fontsize=7, loc="upper right")
        despine_axes(ax_g)
        ax_g.yaxis.grid(True, linestyle="--", alpha=0.3)
        
        fig_ind.suptitle(f"{name} — Illumination Analysis", fontsize=11, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_fig(fig_ind, Path(out_dir), f"illumination_{name.lower().replace(' ', '_')}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4B — PCA of illumination maps (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_illumination_pca(
    datasets: Dict[str, np.ndarray],   # {name: lum_maps (N, H, W)}
    out_dir: str = "figures/illumination",
    n_components: int = 2,
):
    """
    ECCV-style PCA scatter of illumination maps.
    Reveals distinct lighting modes.
    """
    labels_all, mats = [], []
    for name, maps in datasets.items():
        N = len(maps)
        flat = maps.reshape(N, -1)   # (N, H*W)
        labels_all.extend([name] * N)
        mats.append(flat)

    if not mats:
        return

    X = np.concatenate(mats, axis=0).astype(np.float32)
    X = np.nan_to_num(X, nan=0.5, posinf=1.0, neginf=0.0)

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X)
    ev  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    legend_handles = []
    for i, name in enumerate(datasets.keys()):
        mask = np.array(labels_all) == name
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        
        ax.scatter(
            Z[mask, 0], Z[mask, 1], 
            s=10, alpha=0.45,
            linewidths=0.3, edgecolors="white",
            color=c, marker=marker,
            rasterized=True,
        )
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=c, markeredgecolor=c,
                   markersize=7, linewidth=0, label=name)
        )

    ax.set_xlabel(f"PC-1 ({ev[0]:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC-2 ({ev[1]:.1f}% var)", fontsize=9)
    ax.set_title("PCA of Illumination Maps", fontsize=10, fontweight="bold", pad=8)
    
    legend = ax.legend(
        handles=legend_handles,
        title="Dataset",
        markerscale=1.2,
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
        ncol=2 if len(datasets) > 4 else 1,
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_fontweight("bold")

    # Eigenvalue spectrum inset
    n_show = min(8, pca.n_components_)
    inset = ax.inset_axes([0.68, 0.04, 0.30, 0.30])
    ev_full = pca.explained_variance_ratio_[:n_show] * 100
    inset.bar(
        range(1, n_show + 1), ev_full, 
        color="#0077BB", edgecolor="white", linewidth=0.5,
        alpha=0.8
    )
    inset.set_title("Var %", fontsize=7, fontweight="bold")
    inset.set_xlabel("PC", fontsize=6)
    inset.tick_params(labelsize=6)
    inset.set_ylim(0, max(ev_full) * 1.1)
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)

    despine_axes(ax, keep_left=False, keep_bottom=False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")
    
    add_subplot_label(ax, "(c)", x=-0.08, y=1.08, fontsize=10)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "illumination_pca")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/illumination")
    args = p.parse_args()

    assert len(args.features) == len(args.labels)
    lum_data = {}; grad_data = {}; map_data = {}
    for fpath, lbl in zip(args.features, args.labels):
        d = dict(np.load(fpath, allow_pickle=True))
        lum_data[lbl]  = d.get("lum_mean", np.array([]))
        grad_data[lbl] = d.get("lum_grad_var", np.array([]))
        map_data[lbl]  = d.get("lum_maps", np.array([]))

    plot_luminance_spectrum(lum_data, grad_data, args.out_dir)
    if any(len(m) > 0 for m in map_data.values()):
        plot_illumination_pca(map_data, args.out_dir)


if __name__ == "__main__":
    _cli()
