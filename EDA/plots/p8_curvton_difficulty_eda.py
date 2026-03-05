"""
EDA/plots/p8_curvton_difficulty_eda.py
========================================
CURVTON-Specific EDA Plots — ECCV Publication Figures

Specialized plots for comparing Easy vs Medium vs Hard difficulty splits:
- Difficulty-aware color scheme (Green → Orange → Red)
- Side-by-side comparisons
- Radar/spider charts for metric comparison
- Difficulty progression analysis

Usage:
    python EDA/plots/p8_curvton_difficulty_eda.py \
        --features eda_cache/curvton/*.npz \
        --out_dir figures/curvton
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import (
    apply_paper_style, save_fig, add_stat_box, 
    PALETTE, add_subplot_label, despine_axes,
    FILL_ALPHA, LINE_ALPHA,
)

apply_paper_style()


# ═══════════════════════════════════════════════════════════════════════════════
# CURVTON Color Scheme — ColorBrewer Dark2 (Maximally Distinct)
# Uses hue separation to avoid occlusion: Teal vs Orange vs Purple
# NOT traffic-light (green/yellow/red) — those colors overlap perceptually
# ═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY_COLORS = {
    "Easy":   "#1B9E77",  # Dark teal-green (cool tone)
    "Medium": "#D95F02",  # Dark orange (warm tone)
    "Hard":   "#7570B3",  # Muted purple-blue (distinct from both)
}

DIFFICULTY_MARKERS = {
    "Easy":   "o",   # circle (round = easy)
    "Medium": "s",   # square (angular = moderate)
    "Hard":   "^",   # triangle (sharp = difficult)
}

DIFFICULTY_LINESTYLES = {
    "Easy":   "-",    # solid (continuous = smooth)
    "Medium": "--",   # dashed (interrupted = moderate)
    "Hard":   "-.",   # dash-dot (complex = difficult)
}

# Line widths for emphasis (Hard slightly thicker for visibility)
DIFFICULTY_LINEWIDTHS = {
    "Easy":   2.0,
    "Medium": 2.2,
    "Hard":   2.4,
}

DIFFICULTY_ORDER = ["Easy", "Medium", "Hard"]

# Fill alphas for overlapping KDEs (lower for Easy so others show through)
DIFFICULTY_FILL_ALPHA = {
    "Easy":   0.18,   # Most transparent (background)
    "Medium": 0.22,   # Medium transparency
    "Hard":   0.25,   # Least transparent (foreground)
}


# ═══════════════════════════════════════════════════════════════════════════════
# 8A — Difficulty Distribution Overview (ECCV Style)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_difficulty_overview(
    datasets: Dict[str, Dict[str, np.ndarray]],  # {difficulty: {metric: values}}
    out_dir: str = "figures/curvton",
):
    """
    ECCV-style overview panel comparing key metrics across difficulty levels.
    Creates a 2×3 grid showing distribution overlays for main metrics.
    """
    metrics_to_plot = [
        ("pose_error", "Pose Error", "°"),
        ("occlusion", "Occlusion Ratio", ""),
        ("bg_entropy", "Background Entropy", "bits"),
        ("lum_mean", "Mean Luminance", ""),
        ("betas_pc1", "Body Shape (PC1)", ""),
        ("garment_div", "Garment Diversity", ""),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.5))
    axes = axes.flatten()
    
    legend_handles = []
    
    for idx, (metric_key, metric_name, unit) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Plot in reverse order so Easy is drawn first (background)
        for diff in reversed(DIFFICULTY_ORDER):
            if diff not in datasets:
                continue
            
            values = datasets[diff].get(metric_key, np.array([]))
            if len(values) == 0:
                continue
            
            values = values[np.isfinite(values)]
            if len(values) < 5:
                continue
            
            color = DIFFICULTY_COLORS[diff]
            linestyle = DIFFICULTY_LINESTYLES[diff]
            linewidth = DIFFICULTY_LINEWIDTHS[diff]
            fill_alpha = DIFFICULTY_FILL_ALPHA[diff]
            
            # KDE with variable fill alpha, strong distinct lines
            sns.kdeplot(
                values, ax=ax,
                fill=True, alpha=fill_alpha,
                linewidth=linewidth, color=color,
                linestyle=linestyle,
                label=diff if idx == 0 else None,
            )
            
            # Mean line (matching color, dashed)
            ax.axvline(
                values.mean(), color=color,
                linestyle=":", linewidth=1.5, alpha=0.85,
            )
        
        # Style axes
        xlabel = f"{metric_name}"
        if unit:
            xlabel += f" ({unit})"
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Density" if idx % 3 == 0 else "", fontsize=8)
        ax.set_title(metric_name, fontsize=9, fontweight="bold", pad=4)
        ax.tick_params(labelsize=7)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, linewidth=0.4)
        despine_axes(ax)
        
        # Panel label
        panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        add_subplot_label(ax, panel_labels[idx], x=-0.15, y=1.08, fontsize=9)
    
    # Create legend handles with marker + line
    for diff in DIFFICULTY_ORDER:
        legend_handles.append(
            Line2D([0], [0], 
                   color=DIFFICULTY_COLORS[diff],
                   marker=DIFFICULTY_MARKERS[diff],
                   markerfacecolor=DIFFICULTY_COLORS[diff],
                   markersize=7,
                   linewidth=DIFFICULTY_LINEWIDTHS[diff], 
                   linestyle=DIFFICULTY_LINESTYLES[diff],
                   label=diff)
        )
    
    # Shared legend with white background
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        framealpha=0.98,
        edgecolor="#999999",
        fontsize=9,
        fancybox=True,
        shadow=False,
    )
    
    fig.suptitle("CURVTON Difficulty Level Comparison",
                 fontsize=12, fontweight="bold", y=1.10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, Path(out_dir), "curvton_difficulty_overview")


# ═══════════════════════════════════════════════════════════════════════════════
# 8B — Radar/Spider Chart for Metric Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_difficulty_radar(
    metrics: Dict[str, Dict[str, float]],  # {difficulty: {metric: mean_value}}
    out_dir: str = "figures/curvton",
):
    """
    ECCV-style radar chart comparing normalized metrics across difficulty levels.
    """
    metric_names = [
        "Pose\nError",
        "Occlusion",
        "Background\nComplexity",
        "Illumination\nVariance",
        "Shape\nDiversity",
        "Garment\nDiversity",
    ]
    metric_keys = [
        "pose_error", "occlusion_ratio", "bg_complexity",
        "illumination_consistency", "body_shape_preservation", "garment_texture_fidelity"
    ]
    
    n_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    
    # Normalize metrics to [0, 1] range for visualization
    all_values = {}
    for key in metric_keys:
        values = [metrics[d].get(key, 0) for d in DIFFICULTY_ORDER if d in metrics]
        if values:
            all_values[key] = (min(values), max(values))
    
    legend_handles = []
    
    for diff in DIFFICULTY_ORDER:
        if diff not in metrics:
            continue
        
        values = []
        for key in metric_keys:
            val = metrics[diff].get(key, 0)
            if key in all_values and all_values[key][1] != all_values[key][0]:
                # Normalize to [0, 1]
                val_norm = (val - all_values[key][0]) / (all_values[key][1] - all_values[key][0])
            else:
                val_norm = 0.5
            values.append(val_norm)
        
        values += values[:1]  # Close polygon
        
        color = DIFFICULTY_COLORS[diff]
        
        ax.plot(angles, values, "o-", linewidth=2.0, color=color,
                markersize=6, label=diff)
        ax.fill(angles, values, alpha=0.15, color=color)
        
        legend_handles.append(
            Line2D([0], [0], color=color, marker="o", linewidth=2,
                   markersize=6, label=diff)
        )
    
    # Style radar
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Legend
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.25, 1.1),
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9,
    )
    
    ax.set_title("CURVTON Difficulty — Metric Comparison",
                 fontsize=11, fontweight="bold", pad=20)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "curvton_difficulty_radar")


# ═══════════════════════════════════════════════════════════════════════════════
# 8C — Difficulty Progression Bar Chart
# ═══════════════════════════════════════════════════════════════════════════════

def plot_difficulty_bars(
    metrics: Dict[str, Dict[str, float]],  # {difficulty: {metric: mean_value}}
    out_dir: str = "figures/curvton",
):
    """
    ECCV-style grouped bar chart showing metric values across difficulty levels.
    """
    metric_display = {
        "pose_error": "Pose Error",
        "occlusion_ratio": "Occlusion",
        "bg_complexity": "BG Complexity",
        "illumination_consistency": "Illumination",
        "body_shape_preservation": "Shape Pres.",
        "garment_texture_fidelity": "Texture Fid.",
        "unified_index": "Unified Index",
    }
    
    metric_keys = list(metric_display.keys())
    n_metrics = len(metric_keys)
    n_difficulties = len([d for d in DIFFICULTY_ORDER if d in metrics])
    
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    
    x = np.arange(n_metrics)
    width = 0.25
    
    for i, diff in enumerate(DIFFICULTY_ORDER):
        if diff not in metrics:
            continue
        
        values = [metrics[diff].get(k, 0) or 0 for k in metric_keys]
        offset = (i - n_difficulties / 2 + 0.5) * width
        
        bars = ax.bar(
            x + offset, values, width,
            label=diff, color=DIFFICULTY_COLORS[diff],
            edgecolor="white", linewidth=0.8,
            alpha=0.85,
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6, rotation=90,
                )
    
    ax.set_xlabel("Metric", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("CURVTON Metrics by Difficulty Level",
                 fontsize=11, fontweight="bold", pad=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels([metric_display[k] for k in metric_keys],
                       rotation=45, ha="right", fontsize=8)
    
    ax.legend(
        title="Difficulty",
        loc="upper right",
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=8,
    )
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    despine_axes(ax)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "curvton_difficulty_bars")


# ═══════════════════════════════════════════════════════════════════════════════
# 8D — Sample Count Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sample_statistics(
    sample_counts: Dict[str, int],  # {difficulty: count}
    out_dir: str = "figures/curvton",
):
    """
    ECCV-style pie chart / donut showing sample distribution.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 3.0))
    
    # Pie chart
    labels = []
    sizes = []
    colors = []
    explode = []
    
    for diff in DIFFICULTY_ORDER:
        if diff in sample_counts:
            labels.append(diff)
            sizes.append(sample_counts[diff])
            colors.append(DIFFICULTY_COLORS[diff])
            explode.append(0.02)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors,
        explode=explode, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=9),
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight("bold")
    
    ax1.set_title("Sample Distribution", fontsize=10, fontweight="bold")
    
    # Bar chart with counts
    x = np.arange(len(labels))
    bars = ax2.bar(x, sizes, color=colors, edgecolor="white", linewidth=1.5, alpha=0.85)
    
    # Add count labels
    for bar, count in zip(bars, sizes):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{count:,}", ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("Number of Samples", fontsize=9)
    ax2.set_title("Samples per Difficulty", fontsize=10, fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
    despine_axes(ax2)
    
    total = sum(sizes)
    fig.suptitle(f"CURVTON Dataset — {total:,} Total Samples",
                 fontsize=11, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, Path(out_dir), "curvton_sample_statistics")


# ═══════════════════════════════════════════════════════════════════════════════
# 8E — Scaling Analysis (Multi-Ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_scaling_analysis(
    multi_ratio_metrics: Dict[str, Dict[str, Dict[str, float]]],
    # {ratio: {difficulty: {metric: value}}}
    out_dir: str = "figures/curvton",
):
    """
    ECCV-style line plot showing how metrics change with dataset size.
    """
    metric_keys = ["pose_error", "occlusion_ratio", "unified_index"]
    metric_names = ["Pose Error", "Occlusion Ratio", "Unified Index"]
    
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    
    ratios = sorted([int(r.replace("%", "")) for r in multi_ratio_metrics.keys()])
    ratio_labels = [f"{r}%" for r in ratios]
    
    for idx, (metric_key, metric_name) in enumerate(zip(metric_keys, metric_names)):
        ax = axes[idx]
        
        for diff in DIFFICULTY_ORDER:
            values = []
            for r in ratios:
                ratio_key = f"{r}%"
                if ratio_key in multi_ratio_metrics and diff in multi_ratio_metrics[ratio_key]:
                    val = multi_ratio_metrics[ratio_key][diff].get(metric_key, None)
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)
            
            color = DIFFICULTY_COLORS[diff]
            marker = DIFFICULTY_MARKERS[diff]
            linestyle = DIFFICULTY_LINESTYLES[diff]
            
            ax.plot(
                ratios, values, marker=marker,
                color=color, linewidth=1.8,
                linestyle=linestyle, markersize=6,
                label=diff if idx == 0 else None,
            )
        
        ax.set_xlabel("Dataset Size (%)", fontsize=9)
        ax.set_ylabel(metric_name if idx == 0 else "", fontsize=9)
        ax.set_title(metric_name, fontsize=10, fontweight="bold")
        ax.set_xticks(ratios)
        ax.tick_params(labelsize=8)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        despine_axes(ax)
    
    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        framealpha=0.95,
        fontsize=8,
    )
    
    fig.suptitle("CURVTON Scaling Analysis",
                 fontsize=11, fontweight="bold", y=1.12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, Path(out_dir), "curvton_scaling_analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out_dir", default="figures/curvton")
    args = parser.parse_args()
    
    assert len(args.features) == len(args.labels)
    
    # Load features
    datasets = {}
    for fpath, label in zip(args.features, args.labels):
        datasets[label] = dict(np.load(fpath, allow_pickle=True))
    
    # Generate plots
    plot_difficulty_overview(datasets, args.out_dir)
    
    # If we have metric summaries, generate comparison plots
    # This would be called separately with metrics JSON


if __name__ == "__main__":
    _cli()
