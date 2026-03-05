"""
EDA/plots/p6_appearance_eda.py
================================
Appearance / Face Diversity EDA — ECCV Publication Figures

  Figure 6A:  UMAP of face embeddings (coloured by dataset)
  Figure 6B:  Pairwise cosine distance histogram (higher mean → more diversity)

Usage:
    python EDA/plots/p6_appearance_eda.py \
        --features eda_cache/*.npz --labels ... --out_dir figures/appearance
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import (
    apply_paper_style, save_fig, add_stat_box, 
    PALETTE, DATASET_COLORS, DATASET_MARKERS,
    add_subplot_label, despine_axes,
    FILL_ALPHA, LINE_ALPHA, DATASET_LINESTYLES,
)

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# 6A — UMAP of face embeddings (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_face_umap(
    datasets: Dict[str, np.ndarray],   # {name: face_embs (N,D)}
    out_dir: str = "figures/appearance",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_per_ds: int = 2000,
):
    """
    ECCV-style UMAP of face embeddings showing appearance diversity.
    """
    labels_all, embs = [], []
    for name, E in datasets.items():
        idx = np.random.default_rng(42).choice(len(E), min(max_per_ds, len(E)), replace=False)
        labels_all.extend([name] * len(idx))
        embs.append(E[idx])

    if not embs:
        return

    E_all = np.concatenate(embs, axis=0).astype(np.float32)
    E_all = np.nan_to_num(E_all)
    E_all = E_all / (np.linalg.norm(E_all, axis=1, keepdims=True) + 1e-12)

    print(f"  [AppearanceEDA] Embedding {len(E_all)} face vectors …")

    try:
        import umap
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=42, n_jobs=1, metric="cosine")
        Z = reducer.fit_transform(E_all)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, random_state=42,
                 metric="cosine",
                 perplexity=min(30, len(E_all) // 4)).fit_transform(E_all)
        method = "t-SNE"

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    
    legend_handles = []
    for i, name in enumerate(datasets.keys()):
        mask = np.array(labels_all) == name
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        
        ax.scatter(
            Z[mask, 0], Z[mask, 1], 
            s=6, alpha=0.5,
            linewidths=0,
            color=c, marker=marker,
            rasterized=True,
        )
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w",
                   markerfacecolor=c, markeredgecolor=c,
                   markersize=7, linewidth=0, label=name)
        )

    ax.set_title(f"Face Embedding {method} — Appearance Diversity",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(f"{method}-1", fontsize=9)
    ax.set_ylabel(f"{method}-2", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    
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
    
    despine_axes(ax, keep_left=False, keep_bottom=False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")
    
    add_subplot_label(ax, "(a)", x=-0.05, y=1.05, fontsize=10)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "appearance_umap")


# ═══════════════════════════════════════════════════════════════════════════════
# 6B — Pairwise cosine distance distribution (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pairwise_distance_distribution(
    datasets: Dict[str, np.ndarray],   # {name: face_embs (N,D)}
    out_dir: str = "figures/appearance",
    max_pairs: int = 5000,
):
    """
    ECCV-style KDE of pairwise cosine distances.
    Higher mean = more diverse faces.
    Generates both combined and individual plots.
    """
    # ══════════════════════════════════════════════════════════════════════════
    # Combined overlay plot
    # ══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    diversity_scores = {}
    legend_handles = []

    for i, (name, E) in enumerate(datasets.items()):
        E = E.astype(np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        N = len(E)

        # Sample pairs
        rng = np.random.default_rng(42)
        n_pairs = min(max_pairs, N * (N - 1) // 2)
        idxA = rng.integers(0, N, n_pairs)
        idxB = rng.integers(0, N, n_pairs)
        same = idxA == idxB
        idxB[same] = (idxA[same] + 1) % N

        cos_sim = (E[idxA] * E[idxB]).sum(axis=1)
        cos_dist = 1.0 - cos_sim

        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        marker = DATASET_MARKERS.get(name, "o")
        linestyle = DATASET_LINESTYLES.get(name, "-")
        
        # Strong line, subtle fill for overlap visibility
        sns.kdeplot(
            cos_dist, ax=ax, fill=True, alpha=FILL_ALPHA, 
            color=c, linewidth=1.8, linestyle=linestyle
        )
        ax.axvline(cos_dist.mean(), color=c, linestyle="--", linewidth=1.3, alpha=0.9)
        
        diversity_scores[name] = float(cos_dist.mean())
        
        legend_handles.append(
            Line2D([0], [0], marker=marker, color=c,
                   markerfacecolor=c, markersize=7,
                   linewidth=1.8, linestyle=linestyle,
                   label=f"{name} ($D$={cos_dist.mean():.3f})")
        )

    ax.set_xlabel("Pairwise Cosine Distance $(1 - \\cos\\theta)$", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Face Diversity Distribution", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlim(0, 2.0)
    
    legend = ax.legend(
        handles=legend_handles,
        title="Dataset ($D_{face}$)",
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=7,
        loc="upper right",
    )
    legend.get_title().set_fontsize(8)
    legend.get_title().set_fontweight("bold")
    
    despine_axes(ax)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
    ax.set_axisbelow(True)

    # Diversity score bar chart inset
    if len(diversity_scores) > 1:
        names  = list(diversity_scores.keys())
        scores = [diversity_scores[n] for n in names]
        colors = [DATASET_COLORS.get(n, PALETTE[i % len(PALETTE)]) for i, n in enumerate(names)]
        
        inset = ax.inset_axes([0.58, 0.55, 0.38, 0.40])
        bars = inset.bar(
            range(len(names)), scores, 
            color=colors, edgecolor="white", linewidth=0.5, alpha=0.85
        )
        inset.set_xticks(range(len(names)))
        inset.set_xticklabels([n[:5] for n in names], rotation=45, fontsize=6, ha="right")
        inset.set_ylabel("$D_{face}$", fontsize=7)
        inset.set_title("Diversity Score", fontsize=7, fontweight="bold")
        inset.tick_params(axis="y", labelsize=6)
        for spine in inset.spines.values():
            spine.set_linewidth(0.5)
    
    add_subplot_label(ax, "(b)", x=-0.10, y=1.08, fontsize=10)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "appearance_pairwise_combined")

    # ══════════════════════════════════════════════════════════════════════════
    # Individual per-dataset plots
    # ══════════════════════════════════════════════════════════════════════════
    for i, (name, E) in enumerate(datasets.items()):
        E = E.astype(np.float32)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        N = len(E)
        
        rng = np.random.default_rng(42)
        n_pairs = min(max_pairs, N * (N - 1) // 2)
        idxA = rng.integers(0, N, n_pairs)
        idxB = rng.integers(0, N, n_pairs)
        same = idxA == idxB
        idxB[same] = (idxA[same] + 1) % N
        
        cos_sim = (E[idxA] * E[idxB]).sum(axis=1)
        cos_dist = 1.0 - cos_sim
        
        c = DATASET_COLORS.get(name, PALETTE[i % len(PALETTE)])
        
        fig_ind, ax_ind = plt.subplots(figsize=(4.0, 3.0))
        
        sns.kdeplot(cos_dist, ax=ax_ind, fill=True, alpha=0.35, 
                    color=c, linewidth=2.0)
        ax_ind.axvline(cos_dist.mean(), color=c, linestyle="--", linewidth=1.5,
                       alpha=0.9, label=f"μ={cos_dist.mean():.3f}")
        ax_ind.axvline(np.median(cos_dist), color=c, linestyle=":", linewidth=1.5,
                       alpha=0.9, label=f"med={np.median(cos_dist):.3f}")
        
        ax_ind.set_xlabel("Pairwise Cosine Distance", fontsize=10)
        ax_ind.set_ylabel("Density", fontsize=10)
        ax_ind.set_title(f"{name} — Face Diversity", fontsize=11, fontweight="bold")
        ax_ind.set_xlim(0, 2.0)
        ax_ind.legend(loc="upper right", fontsize=8)
        despine_axes(ax_ind)
        ax_ind.yaxis.grid(True, linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        save_fig(fig_ind, Path(out_dir), f"appearance_{name.lower().replace(' ', '_')}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--labels",   nargs="+", required=True)
    p.add_argument("--out_dir",  default="figures/appearance")
    args = p.parse_args()

    assert len(args.features) == len(args.labels)
    face_data = {}
    for fpath, lbl in zip(args.features, args.labels):
        d = dict(np.load(fpath, allow_pickle=True))
        face_data[lbl] = d.get("face_embs", np.array([]))

    plot_face_umap(face_data, args.out_dir)
    plot_pairwise_distance_distribution(face_data, args.out_dir)


if __name__ == "__main__":
    _cli()
