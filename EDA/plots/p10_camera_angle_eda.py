"""
EDA/plots/p10_camera_angle_eda.py
=================================
Camera-angle EDA plots.

Uses explicit camera-angle outputs (azimuth/elevation) only.
No pose-based proxy fallback is used.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from plot_style import apply_paper_style, PALETTE, save_fig
except ImportError:
    from ..plot_style import apply_paper_style, PALETTE, save_fig


def _get_color_palette(n: int):
    if n <= 0:
        return ["#0077BB"]
    reps = (n // len(PALETTE)) + 1
    return (PALETTE * reps)[:n]


def _estimate_camera_proxy_from_pose(pose_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate camera-view proxies from pose keypoints when explicit camera angles
    are unavailable.

    pose_vecs: (N, 34), flattened COCO-17 normalized keypoints.
    Returns:
      azimuths   in [-180, 180]
      elevations in [-45, 45]
    """
    if pose_vecs.ndim != 2 or pose_vecs.shape[1] != 34:
        return np.array([]), np.array([])

    kps = pose_vecs.reshape(-1, 17, 2)

    # COCO keypoint indices
    NOSE = 0
    LS, RS = 5, 6
    LH, RH = 11, 12

    nose = kps[:, NOSE, :]
    ls = kps[:, LS, :]
    rs = kps[:, RS, :]
    lh = kps[:, LH, :]
    rh = kps[:, RH, :]

    shoulder_mid = 0.5 * (ls + rs)
    hip_mid = 0.5 * (lh + rh)
    shoulder_w = np.abs(rs[:, 0] - ls[:, 0]) + 1e-6
    torso_h = np.abs(hip_mid[:, 1] - shoulder_mid[:, 1]) + 1e-6

    # Horizontal asymmetry proxy (nose offset from shoulder center)
    azimuth = np.degrees(np.arctan2((nose[:, 0] - shoulder_mid[:, 0]), 0.5 * shoulder_w))
    azimuth = np.clip(azimuth, -180, 180).astype(np.float32)

    # Vertical framing proxy (nose above/below shoulder center)
    elevation = ((shoulder_mid[:, 1] - nose[:, 1]) / torso_h * 30.0).astype(np.float32)
    elevation = np.clip(elevation, -45, 45)

    valid = np.isfinite(azimuth) & np.isfinite(elevation)
    return azimuth[valid], elevation[valid]


def extract_camera_angles(features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Get (azimuths, elevations) from explicit camera-angle features only."""
    azimuths = features.get("azimuths", features.get("azimuth", np.array([])))
    elevations = features.get("elevations", features.get("elevation", np.array([])))

    if len(azimuths) > 0:
        az = np.asarray(azimuths, dtype=np.float32)
        el = np.asarray(elevations if len(elevations) > 0 else np.zeros_like(az), dtype=np.float32)
        valid = np.isfinite(az) & np.isfinite(el)
        return az[valid], el[valid]

    return np.array([]), np.array([])


def plot_camera_angle_distribution(
    azimuths: np.ndarray,
    elevations: np.ndarray,
    dataset_name: str = "Dataset",
    output_dir: str = "./eda_outputs",
    n_azimuth_bins: int = 12,
    n_elevation_bins: int = 6,
    figsize: tuple = (14, 5),
) -> Dict[str, str]:
    """Create per-dataset camera-angle distribution plots."""
    apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: Dict[str, str] = {}

    colors = _get_color_palette(5)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Polar azimuth histogram
    ax_polar = plt.subplot(131, projection="polar")
    az_rad = np.deg2rad(azimuths)
    bin_edges = np.linspace(-np.pi, np.pi, n_azimuth_bins + 1)
    hist, _ = np.histogram(az_rad, bins=bin_edges)
    width = 2 * np.pi / n_azimuth_bins
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax_polar.set_theta_zero_location("N")
    ax_polar.set_theta_direction(-1)
    ax_polar.bar(
        centers,
        hist,
        width=width * 0.9,
        color=colors[0],
        alpha=0.75,
        edgecolor="white",
        linewidth=1,
    )
    ax_polar.set_title(f"Azimuth Distribution\n({dataset_name})", pad=15, fontsize=11)
    ax_polar.set_xticks(np.deg2rad([0, 45, 90, 135, 180, -135, -90, -45]))
    ax_polar.set_xticklabels(["Front", "45R", "Right", "135R", "Back", "135L", "Left", "45L"], fontsize=8)

    # Elevation histogram
    ax_elev = axes[1]
    el_bins = np.linspace(-45, 45, n_elevation_bins + 1)
    ax_elev.hist(np.clip(elevations, -45, 45), bins=el_bins, color=colors[1], alpha=0.75, edgecolor="white", linewidth=1)
    ax_elev.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="Eye level")
    ax_elev.set_xlabel("Elevation (deg)", fontsize=10)
    ax_elev.set_ylabel("Count", fontsize=10)
    ax_elev.set_title(f"Elevation Distribution\n({dataset_name})", fontsize=11)
    ax_elev.legend(fontsize=8)

    # View-category donut
    ax_pie = axes[2]
    frontal = np.sum(np.abs(azimuths) < 30)
    quarter = np.sum((np.abs(azimuths) >= 30) & (np.abs(azimuths) < 60))
    side = np.sum((np.abs(azimuths) >= 60) & (np.abs(azimuths) < 120))
    three_quarter = np.sum((np.abs(azimuths) >= 120) & (np.abs(azimuths) < 150))
    back = np.sum(np.abs(azimuths) >= 150)

    categories = ["Frontal", "3/4 Front", "Side", "3/4 Back", "Back"]
    counts = [frontal, quarter, side, three_quarter, back]
    mask = np.array(counts) > 0
    categories = [c for c, m in zip(categories, mask) if m]
    counts = [c for c, m in zip(counts, mask) if m]
    pie_colors = [colors[i] for i, m in enumerate(mask) if m]

    if counts:
        wedges, texts, autotexts = ax_pie.pie(
            counts,
            labels=categories,
            colors=pie_colors,
            autopct="%1.1f%%",
            pctdistance=0.75,
            startangle=90,
            wedgeprops=dict(width=0.6, edgecolor="white"),
        )
        for t in texts:
            t.set_fontsize(8)
        for t in autotexts:
            t.set_fontsize(8)
    ax_pie.set_title(f"View Category Distribution\n({dataset_name})", fontsize=11)

    plt.tight_layout()
    stem = f"camera_angle_distribution_{dataset_name.lower().replace(' ', '_')}"
    save_fig(fig, output_dir, stem, formats=("pdf", "png"), dpi=600)
    saved_paths["distribution_pdf"] = os.path.join(output_dir, f"{stem}.pdf")
    saved_paths["distribution_png"] = os.path.join(output_dir, f"{stem}.png")

    # 2D heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    az_bins = np.linspace(-180, 180, n_azimuth_bins + 1)
    el_bins = np.linspace(-45, 45, n_elevation_bins + 1)
    H, _, _ = np.histogram2d(azimuths, np.clip(elevations, -45, 45), bins=[az_bins, el_bins])

    im = ax.imshow(H.T, origin="lower", extent=[-180, 180, -45, 45], aspect="auto", cmap="YlOrRd", interpolation="bilinear")
    plt.colorbar(im, ax=ax, label="Count", shrink=0.8)

    for x in [-150, -120, -60, -30, 30, 60, 120, 150]:
        ax.axvline(x=x, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Azimuth (deg)", fontsize=10)
    ax.set_ylabel("Elevation (deg)", fontsize=10)
    ax.set_title(f"Camera Angle Heatmap - {dataset_name}", fontsize=12)

    plt.tight_layout()
    stem = f"camera_angle_heatmap_{dataset_name.lower().replace(' ', '_')}"
    save_fig(fig, output_dir, stem, formats=("pdf", "png"), dpi=600)
    saved_paths["heatmap_pdf"] = os.path.join(output_dir, f"{stem}.pdf")
    saved_paths["heatmap_png"] = os.path.join(output_dir, f"{stem}.png")

    return saved_paths


def plot_camera_angle_comparison(
    datasets: Dict[str, Dict[str, np.ndarray]],
    output_dir: str = "./eda_outputs",
    figsize: tuple = (12, 8),
) -> str:
    """Compare camera-angle distributions across datasets."""
    apply_paper_style()
    os.makedirs(output_dir, exist_ok=True)

    n_datasets = len(datasets)
    colors = _get_color_palette(n_datasets)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Azimuth density curves (histogram density, no scipy dependency)
    ax = axes[0, 0]
    for i, (name, data) in enumerate(datasets.items()):
        az = data["azimuths"]
        dens, edges = np.histogram(az, bins=48, range=(-180, 180), density=True)
        x = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(x, dens, label=name, color=colors[i], linewidth=2)
        ax.fill_between(x, dens, alpha=0.2, color=colors[i])
    ax.set_xlabel("Azimuth (deg)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Azimuth Distribution Comparison", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(-180, 180)

    # Elevation density curves
    ax = axes[0, 1]
    for i, (name, data) in enumerate(datasets.items()):
        el = np.clip(data["elevations"], -45, 45)
        dens, edges = np.histogram(el, bins=24, range=(-45, 45), density=True)
        x = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(x, dens, label=name, color=colors[i], linewidth=2)
        ax.fill_between(x, dens, alpha=0.2, color=colors[i])
    ax.set_xlabel("Elevation (deg)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Elevation Distribution Comparison", fontsize=11)
    ax.legend(fontsize=8)

    # Category bars
    ax = axes[1, 0]
    categories = ["Frontal", "3/4 Front", "Side", "3/4 Back", "Back"]
    x = np.arange(len(categories))
    width = 0.8 / max(n_datasets, 1)

    for i, (name, data) in enumerate(datasets.items()):
        az = data["azimuths"]
        counts = [
            np.sum(np.abs(az) < 30),
            np.sum((np.abs(az) >= 30) & (np.abs(az) < 60)),
            np.sum((np.abs(az) >= 60) & (np.abs(az) < 120)),
            np.sum((np.abs(az) >= 120) & (np.abs(az) < 150)),
            np.sum(np.abs(az) >= 150),
        ]
        ratios = np.array(counts) / max(len(az), 1) * 100
        offset = (i - n_datasets / 2 + 0.5) * width
        ax.bar(x + offset, ratios, width, label=name, color=colors[i], alpha=0.8)

    ax.set_xlabel("View Category", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=10)
    ax.set_title("View Category Comparison", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8, rotation=15)
    ax.legend(fontsize=8)

    # Diversity bars
    ax = axes[1, 1]
    metrics_names = ["Az Std", "Az Entropy", "El Std", "Diversity"]
    x = np.arange(len(metrics_names))

    for i, (name, data) in enumerate(datasets.items()):
        az = data["azimuths"]
        el = data["elevations"]

        az_std = np.std(az) / 90.0
        el_std = np.std(el) / 30.0

        hist, _ = np.histogram(az, bins=12, range=(-180, 180), density=True)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        az_entropy = -np.sum(hist * np.log2(hist)) / np.log2(12)

        diversity = (az_std * 0.4 + az_entropy * 0.4 + el_std * 0.2)
        values = [az_std, az_entropy, el_std, diversity]

        offset = (i - n_datasets / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=colors[i], alpha=0.8)

    ax.set_xlabel("Metric", fontsize=10)
    ax.set_ylabel("Normalized Score", fontsize=10)
    ax.set_title("Camera Angle Diversity Metrics", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    stem = "camera_angle_comparison"
    save_fig(fig, output_dir, stem, formats=("pdf", "png"), dpi=600)
    return os.path.join(output_dir, f"{stem}.png")


def run_camera_angle_eda(
    features: Dict[str, np.ndarray],
    dataset_name: str = "Dataset",
    output_dir: str = "./eda_outputs",
) -> Dict[str, str]:
    """Run camera-angle EDA for a single dataset."""
    azimuths, elevations = extract_camera_angles(features)

    if len(azimuths) == 0:
        print(
            f"[CameraAngleEDA] No explicit camera-angle features found for {dataset_name}. "
            "Re-extract with camera backend enabled."
        )
        return {}

    return plot_camera_angle_distribution(
        azimuths=azimuths,
        elevations=elevations,
        dataset_name=dataset_name,
        output_dir=output_dir,
    )
