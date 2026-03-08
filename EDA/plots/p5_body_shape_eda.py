"""
EDA/plots/p5_body_shape_eda.py
================================
Body Shape EDA — ECCV Publication Figures

  Figure 5A:  PCA of shape parameters → 2D scatter (Z_shape)
  Figure 5B:  Per-coefficient histogram grid (10 subplots, one per β_j)

Usage:
    python EDA/plots/p5_body_shape_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/body_shape
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import (
    apply_paper_style, save_fig, add_stat_box, 
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes,
    FILL_ALPHA, LINE_ALPHA, DATASET_LINESTYLES,
)

apply_paper_style()

BETA_LABELS = [f"$\\beta_{{{j}}}$" for j in range(10)]


# ═══════════════════════════════════════════════════════════════════════════════
# 5A — PCA of β (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_shape_pca(
    datasets: Dict[str, np.ndarray],   # {name: betas (N,10)}
    out_dir: str = "figures/body_shape",
):
    """
    ECCV-style PCA scatter of SMPL shape parameters.
    Includes 1σ confidence ellipses per dataset.
    """
    labels_all, mats = [], []
    for name, B in datasets.items():
        labels_all.extend([name] * len(B))
        mats.append(B)

    if not mats:
        return

    X = np.concatenate(mats, axis=0).astype(np.float32)
    X = np.nan_to_num(X)

    # Scale the features so extreme outliers don't dominate the PCA
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    Z   = pca.fit_transform(X_scaled)
    ev  = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    legend_handles = []
    for i, name in enumerate(datasets.keys()):
        mask = np.array(labels_all) == name
        Zm   = Z[mask]
        c    = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        
        ax.scatter(
            Zm[:, 0], Zm[:, 1], 
            s=6, alpha=0.4,
            linewidths=0,
            color=c, marker=marker,
            rasterized=True,
        )

        # 1-σ confidence ellipse
        if len(Zm) >= 3:
            mu  = Zm.mean(0)
            cov = np.cov(Zm.T)
            ev2, evec = np.linalg.eigh(cov)
            order = ev2.argsort()[::-1]
            ev2 = ev2[order]; evec = evec[:, order]
            angle = np.degrees(np.arctan2(*evec[:, 0][::-1]))
            w = 2 * np.sqrt(ev2[0]) * 1.0
            h = 2 * np.sqrt(ev2[1]) * 1.0
            ell = Ellipse(
                xy=mu, width=w, height=h, angle=angle,
                edgecolor=c, facecolor="none",
                linewidth=1.2, linestyle="-", zorder=3
            )
            ax.add_patch(ell)
            ax.scatter(*mu, color=c, s=80, marker="x", linewidths=2.5, zorder=4)
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=c, markeredgecolor=c,
                   markersize=7, linewidth=0, label=name)
        )

    ax.set_xlabel(f"PC-1 ({ev[0]:.1f}% var)", fontsize=9)
    ax.set_ylabel(f"PC-2 ({ev[1]:.1f}% var)", fontsize=9)
    ax.set_title("Body Shape Diversity — PCA($\\beta$)", fontsize=10, fontweight="bold", pad=8)
    
    legend = ax.legend(
        handles=legend_handles,
        title="Dataset",
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
        ncol=2 if len(datasets) > 4 else 1,
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_fontweight("bold")
    
    ax.autoscale_view()
    despine_axes(ax, keep_left=False, keep_bottom=False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")
    
    add_subplot_label(ax, "(a)", x=-0.08, y=1.08, fontsize=10)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "shape_pca")


# ═══════════════════════════════════════════════════════════════════════════════
# 5B — Per-coefficient histogram (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_shape_coefficient_histograms(
    datasets: Dict[str, np.ndarray],   # {name: betas (N,10)}
    out_dir: str = "figures/body_shape",
):
    """
    ECCV-style 2×5 grid of KDE plots for each SMPL shape coefficient.
    Includes N(0,1) prior reference.
    Generates both combined and individual plots.
    """
    # ══════════════════════════════════════════════════════════════════════════
    # Combined overlay plot
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(7.0, 4.2))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.45, wspace=0.35)

    legend_handles = []
    
    for j in range(10):
        row, col = divmod(j, 5)
        ax = fig.add_subplot(gs[row, col])

        for i, (name, B) in enumerate(datasets.items()):
            vals = B[:, j]
            vals = vals[np.isfinite(vals)]
            if len(vals) < 3:
                continue
            c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
            linestyle = DATASET_LINESTYLES.get(name, "-")
            
            # Strong line, subtle fill for overlap visibility
            sns.kdeplot(
                vals, ax=ax, fill=True, alpha=FILL_ALPHA, 
                color=c, linewidth=1.5, linestyle=linestyle,
            )
            ax.axvline(vals.mean(), color=c, linestyle="--", linewidth=1.0, alpha=0.9)
            
            if j == 0:
                legend_handles.append(
                    Line2D([0], [0], color=c, linewidth=2, linestyle=linestyle, label=name)
                )

        ax.set_title(BETA_LABELS[j], fontsize=9, fontweight="bold", pad=3)
        ax.set_xlabel("", fontsize=7)
        ax.set_ylabel("" if col > 0 else "Density", fontsize=7)
        ax.tick_params(labelsize=6)
        
        # SMPL prior ≈ N(0,1) reference
        x_rng = np.linspace(-3, 3, 100)
        prior_line, = ax.plot(
            x_rng, np.exp(-0.5 * x_rng ** 2) / np.sqrt(2 * np.pi),
            "k:", linewidth=0.8, alpha=0.6
        )
        
        if j == 0:
            legend_handles.append(
                Line2D([0], [0], color="k", linestyle=":", linewidth=1, 
                       label="$\\mathcal{N}(0,1)$ prior")
            )
        
        despine_axes(ax)
        ax.set_xlim(-3.5, 3.5)

    # Shared legend
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(datasets) + 1, 6),
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=7,
    )
    
    fig.suptitle("SMPL Shape Coefficient Distributions ($\\beta_0 \\ldots \\beta_9$)",
                 fontsize=11, fontweight="bold", y=1.08)
    save_fig(fig, Path(out_dir), "shape_coefficient_combined")

    # ══════════════════════════════════════════════════════════════════════════
    # Individual per-dataset plots
    # ══════════════════════════════════════════════════════════════════════════
    for i, (name, B) in enumerate(datasets.items()):
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        
        fig_ind = plt.figure(figsize=(6.5, 3.8))
        gs_ind = gridspec.GridSpec(2, 5, figure=fig_ind, hspace=0.45, wspace=0.35)
        
        for j in range(10):
            row, col = divmod(j, 5)
            ax = fig_ind.add_subplot(gs_ind[row, col])
            
            vals = B[:, j]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 3:
                sns.kdeplot(vals, ax=ax, fill=True, alpha=0.35, color=c, linewidth=1.8)
                ax.axvline(vals.mean(), color=c, linestyle="--", linewidth=1.2, alpha=0.9)
            
            # N(0,1) prior reference
            x_rng = np.linspace(-3, 3, 100)
            ax.plot(x_rng, np.exp(-0.5 * x_rng ** 2) / np.sqrt(2 * np.pi),
                    "k:", linewidth=0.8, alpha=0.6)
            
            ax.set_title(BETA_LABELS[j], fontsize=9, fontweight="bold", pad=3)
            ax.set_xlabel("", fontsize=7)
            ax.set_ylabel("" if col > 0 else "Density", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlim(-3.5, 3.5)
            despine_axes(ax)
        
        fig_ind.suptitle(f"{name} — Shape Coefficients ($\\beta_0 \\ldots \\beta_9$)",
                         fontsize=11, fontweight="bold", y=1.02)
        save_fig(fig_ind, Path(out_dir), f"shape_coeff_{name.lower().replace(' ', '_')}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/body_shape")
    args = p.parse_args()

    assert len(args.features) == len(args.labels)
    beta_data = {}
    for fpath, lbl in zip(args.features, args.labels):
        d = dict(np.load(fpath, allow_pickle=True))
        beta_data[lbl] = d.get("betas", np.array([]))

    plot_shape_pca(beta_data, args.out_dir)
    plot_shape_coefficient_histograms(beta_data, args.out_dir)


if __name__ == "__main__":
    _cli()
