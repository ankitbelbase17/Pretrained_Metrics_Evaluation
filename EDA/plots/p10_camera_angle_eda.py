"""
EDA/plots/p10_camera_angle_eda.py
=================================
Camera Angle Diversity EDA Visualization

Creates polar/radial plots showing the distribution of camera viewing angles
(azimuth and elevation) across the dataset.

ECCV-ready styling with professional visualizations.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection

from ..plot_style import apply_eccv_style, ECCV_COLORS, get_color_palette


def plot_camera_angle_distribution(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    dataset_name: str = "Dataset",
    output_dir: str = "./eda_outputs",
    n_azimuth_bins: int = 12,
    n_elevation_bins: int = 6,
    figsize: tuple = (14, 5),
) -> Dict[str, str]:
    """
    Create camera angle distribution visualizations.

    Args:
        azimuths: Array of azimuth angles in degrees [-180, 180]
        elevations: Array of elevation angles in degrees [-90, 90]
        dataset_name: Name for plot titles
        output_dir: Directory to save plots
        n_azimuth_bins: Number of azimuth bins (default 12 = 30° each)
        n_elevation_bins: Number of elevation bins
        figsize: Figure size

    Returns:
        Dict mapping plot names to file paths
    """
    apply_eccv_style()
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}

    colors = get_color_palette(5)

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 1: Polar histogram of azimuth angles
    # ═══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Polar plot (azimuth) ──────────────────────────────────────────────────
    ax_polar = plt.subplot(131, projection='polar')

    # Convert to radians and shift so 0° is at top
    az_rad = np.deg2rad(azimuths)

    # Create histogram
    bin_edges = np.linspace(-np.pi, np.pi, n_azimuth_bins + 1)
    hist, _ = np.histogram(az_rad, bins=bin_edges)

    # Bar width
    width = 2 * np.pi / n_azimuth_bins

    # Center of each bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Rotate so 0° (frontal) is at top
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)

    # Plot bars
    bars = ax_polar.bar(
        bin_centers, hist,
        width=width * 0.9,
        color=colors[0],
        alpha=0.7,
        edgecolor='white',
        linewidth=1
    )

    # Add labels
    ax_polar.set_title(f'Azimuth Distribution\n({dataset_name})', pad=15, fontsize=11)
    ax_polar.set_xticks(np.deg2rad([0, 45, 90, 135, 180, -135, -90, -45]))
    ax_polar.set_xticklabels(['Front', '45°R', 'Right', '135°R', 
                              'Back', '135°L', 'Left', '45°L'], fontsize=8)

    # ── Elevation histogram ───────────────────────────────────────────────────
    ax_elev = axes[1]

    el_bins = np.linspace(-45, 45, n_elevation_bins + 1)
    ax_elev.hist(
        np.clip(elevations, -45, 45),
        bins=el_bins,
        color=colors[1],
        alpha=0.7,
        edgecolor='white',
        linewidth=1
    )
    ax_elev.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Eye level')
    ax_elev.set_xlabel('Elevation Angle (°)', fontsize=10)
    ax_elev.set_ylabel('Count', fontsize=10)
    ax_elev.set_title(f'Elevation Distribution\n({dataset_name})', fontsize=11)
    ax_elev.legend(fontsize=8)

    # ── View category pie chart ───────────────────────────────────────────────
    ax_pie = axes[2]

    # Categorize views
    frontal = np.sum(np.abs(azimuths) < 30)
    quarter = np.sum((np.abs(azimuths) >= 30) & (np.abs(azimuths) < 60))
    side = np.sum((np.abs(azimuths) >= 60) & (np.abs(azimuths) < 120))
    three_quarter = np.sum((np.abs(azimuths) >= 120) & (np.abs(azimuths) < 150))
    back = np.sum(np.abs(azimuths) >= 150)

    categories = ['Frontal\n(±30°)', '3/4 Front\n(30-60°)', 'Side\n(60-120°)', 
                  '3/4 Back\n(120-150°)', 'Back\n(>150°)']
    counts = [frontal, quarter, side, three_quarter, back]

    # Filter out zero categories
    nonzero_mask = np.array(counts) > 0
    categories = [c for c, m in zip(categories, nonzero_mask) if m]
    counts = [c for c, m in zip(counts, nonzero_mask) if m]
    pie_colors = [colors[i] for i, m in enumerate(nonzero_mask) if m]

    if counts:
        wedges, texts, autotexts = ax_pie.pie(
            counts,
            labels=categories,
            colors=pie_colors,
            autopct='%1.1f%%',
            pctdistance=0.75,
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor='white'),
        )
        for autotext in autotexts:
            autotext.set_fontsize(8)
        for text in texts:
            text.set_fontsize(8)

    ax_pie.set_title(f'View Category Distribution\n({dataset_name})', fontsize=11)

    plt.tight_layout()
    path = os.path.join(output_dir, f"camera_angle_distribution_{dataset_name.lower().replace(' ', '_')}.pdf")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_paths['distribution'] = path

    # ═══════════════════════════════════════════════════════════════════════════
    # Plot 2: 2D heatmap of azimuth vs elevation
    # ═══════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create 2D histogram
    az_bins = np.linspace(-180, 180, n_azimuth_bins + 1)
    el_bins = np.linspace(-45, 45, n_elevation_bins + 1)

    H, xedges, yedges = np.histogram2d(
        azimuths, np.clip(elevations, -45, 45),
        bins=[az_bins, el_bins]
    )

    # Plot heatmap
    im = ax.imshow(
        H.T,
        origin='lower',
        extent=[-180, 180, -45, 45],
        aspect='auto',
        cmap='YlOrRd',
        interpolation='bilinear'
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Count', shrink=0.8)

    # Mark view regions
    for x in [-150, -120, -60, -30, 30, 60, 120, 150]:
        ax.axvline(x=x, color='gray', linestyle=':', alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Azimuth (°)', fontsize=10)
    ax.set_ylabel('Elevation (°)', fontsize=10)
    ax.set_title(f'Camera Angle Heatmap — {dataset_name}', fontsize=12)

    # Add region labels
    ax.text(0, -40, 'Frontal', ha='center', fontsize=8, alpha=0.7)
    ax.text(-90, -40, 'Left', ha='center', fontsize=8, alpha=0.7)
    ax.text(90, -40, 'Right', ha='center', fontsize=8, alpha=0.7)
    ax.text(180, -40, 'Back', ha='center', fontsize=8, alpha=0.7)

    plt.tight_layout()
    path = os.path.join(output_dir, f"camera_angle_heatmap_{dataset_name.lower().replace(' ', '_')}.pdf")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_paths['heatmap'] = path

    return saved_paths


def plot_camera_angle_comparison(
    datasets: Dict[str, Dict[str, np.ndarray]],
    output_dir: str = "./eda_outputs",
    figsize: tuple = (12, 8),
) -> str:
    """
    Compare camera angle distributions across multiple datasets.

    Args:
        datasets: Dict mapping dataset names to dicts with 'azimuths' and 'elevations'
        output_dir: Directory to save plot
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    apply_eccv_style()
    os.makedirs(output_dir, exist_ok=True)

    n_datasets = len(datasets)
    colors = get_color_palette(n_datasets)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ── Azimuth KDE comparison ────────────────────────────────────────────────
    ax = axes[0, 0]
    for i, (name, data) in enumerate(datasets.items()):
        az = data['azimuths']
        # Kernel density estimate
        from scipy import stats
        kde = stats.gaussian_kde(az, bw_method=0.2)
        x = np.linspace(-180, 180, 200)
        ax.plot(x, kde(x), label=name, color=colors[i], linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.2, color=colors[i])

    ax.set_xlabel('Azimuth (°)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Azimuth Distribution Comparison', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(-180, 180)

    # ── Elevation KDE comparison ──────────────────────────────────────────────
    ax = axes[0, 1]
    for i, (name, data) in enumerate(datasets.items()):
        el = np.clip(data['elevations'], -45, 45)
        kde = stats.gaussian_kde(el, bw_method=0.3)
        x = np.linspace(-45, 45, 100)
        ax.plot(x, kde(x), label=name, color=colors[i], linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.2, color=colors[i])

    ax.set_xlabel('Elevation (°)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Elevation Distribution Comparison', fontsize=11)
    ax.legend(fontsize=8)

    # ── View category bar comparison ──────────────────────────────────────────
    ax = axes[1, 0]

    categories = ['Frontal', '3/4 Front', 'Side', '3/4 Back', 'Back']
    x = np.arange(len(categories))
    width = 0.8 / n_datasets

    for i, (name, data) in enumerate(datasets.items()):
        az = data['azimuths']
        counts = [
            np.sum(np.abs(az) < 30),
            np.sum((np.abs(az) >= 30) & (np.abs(az) < 60)),
            np.sum((np.abs(az) >= 60) & (np.abs(az) < 120)),
            np.sum((np.abs(az) >= 120) & (np.abs(az) < 150)),
            np.sum(np.abs(az) >= 150),
        ]
        ratios = np.array(counts) / len(az) * 100

        offset = (i - n_datasets / 2 + 0.5) * width
        ax.bar(x + offset, ratios, width, label=name, color=colors[i], alpha=0.8)

    ax.set_xlabel('View Category', fontsize=10)
    ax.set_ylabel('Percentage (%)', fontsize=10)
    ax.set_title('View Category Comparison', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, rotation=15)
    ax.legend(fontsize=8)

    # ── Diversity metrics comparison ──────────────────────────────────────────
    ax = axes[1, 1]

    metrics_names = ['Az. Std', 'Az. Entropy', 'El. Std', 'Diversity']
    x = np.arange(len(metrics_names))

    for i, (name, data) in enumerate(datasets.items()):
        az = data['azimuths']
        el = data['elevations']

        # Compute metrics
        az_std = np.std(az) / 90  # Normalize
        el_std = np.std(el) / 30  # Normalize

        # Entropy
        hist, _ = np.histogram(az, bins=12, range=(-180, 180), density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        az_entropy = -np.sum(hist * np.log2(hist)) / np.log2(12)

        # Diversity score
        diversity = (az_std * 0.4 + az_entropy * 0.4 + el_std * 0.2)

        values = [az_std, az_entropy, el_std, diversity]
        offset = (i - n_datasets / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=colors[i], alpha=0.8)

    ax.set_xlabel('Metric', fontsize=10)
    ax.set_ylabel('Normalized Score', fontsize=10)
    ax.set_title('Camera Angle Diversity Metrics', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(output_dir, "camera_angle_comparison.pdf")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    return path


def run_camera_angle_eda(
    features: Dict[str, np.ndarray],
    dataset_name: str = "Dataset",
    output_dir: str = "./eda_outputs",
) -> Dict[str, str]:
    """
    Run complete camera angle EDA.

    Args:
        features: Dict containing 'azimuths' and 'elevations' arrays
        dataset_name: Name for plot titles
        output_dir: Directory to save plots

    Returns:
        Dict mapping plot names to file paths
    """
    azimuths = features.get('azimuths', features.get('azimuth', np.array([])))
    elevations = features.get('elevations', features.get('elevation', np.array([])))

    if len(azimuths) == 0:
        print(f"[CameraAngleEDA] No azimuth data found for {dataset_name}")
        return {}

    return plot_camera_angle_distribution(
        azimuths=azimuths,
        elevations=elevations,
        dataset_name=dataset_name,
        output_dir=output_dir,
    )
