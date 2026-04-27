from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


REQUIRED_SAMPLE_RATIO = 0.2


def _apply_eccv_style() -> None:
    try:
        from EDA.plot_style import apply_paper_style

        apply_paper_style()
    except Exception:
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )


def _difficulty_colors() -> Dict[str, str]:
    try:
        from EDA.plot_style import CURVTON_COLORS

        return {
            "Easy": CURVTON_COLORS.get("Easy", "#1B9E77"),
            "Medium": CURVTON_COLORS.get("Medium", "#D95F02"),
            "Hard": CURVTON_COLORS.get("Hard", "#7570B3"),
        }
    except Exception:
        return {
            "Easy": "#1B9E77",
            "Medium": "#D95F02",
            "Hard": "#7570B3",
        }


def _load_pose_arrays(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "pose_vecs" not in data:
        raise KeyError(f"Missing pose_vecs in {npz_path}")
    if "angles" not in data:
        raise KeyError(f"Missing angles in {npz_path}")
    pose_vecs = data["pose_vecs"].astype(np.float32)
    angles = data["angles"].astype(np.float32)
    return pose_vecs, angles


def _subsample_pair(
    pose_vecs: np.ndarray,
    angles: np.ndarray,
    sample_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_ratio >= 1.0:
        return pose_vecs, angles
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = pose_vecs.shape[0]
    if n <= 1:
        return pose_vecs, angles
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return pose_vecs[idx], angles[idx]


def _pose_diversity_score(pose_vecs: np.ndarray) -> float:
    if pose_vecs.ndim != 2:
        raise ValueError(f"pose_vecs expected shape (N, D), got {pose_vecs.shape}")
    if pose_vecs.shape[0] < 2:
        return 0.0
    # Normalize per-dimension to focus on diversity, not scale.
    mu = pose_vecs.mean(axis=0, keepdims=True)
    sig = pose_vecs.std(axis=0, keepdims=True) + 1e-8
    normed = (pose_vecs - mu) / sig
    return float(np.mean(np.var(normed, axis=0)))


def _angle_diversity_score(angles: np.ndarray) -> float:
    if angles.ndim != 2:
        raise ValueError(f"angles expected shape (N, K), got {angles.shape}")
    if angles.shape[0] < 2:
        return 0.0
    angles_deg = angles * (180.0 / np.pi)
    return float(np.mean(np.var(angles_deg, axis=0)))


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from EDA.plot_style import save_fig

        save_fig(fig, out_dir, stem)
    except Exception:
        fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_pose_diversity(
    metrics: Dict[str, Tuple[float, float]],
    out_dir: Path,
    stem: str,
    show_legend: bool,
) -> None:
    _apply_eccv_style()
    colors = _difficulty_colors()

    labels = ["Easy", "Medium", "Hard"]
    x = np.arange(len(labels))
    width = 0.34

    pose_vals = [metrics[lbl][0] for lbl in labels]
    angle_vals = [metrics[lbl][1] for lbl in labels]

    fig, ax = plt.subplots(figsize=(5.4, 3.2))

    ax.bar(x - width / 2, pose_vals, width, color=[colors[l] for l in labels], alpha=0.85, label="Pose variance")
    ax.bar(x + width / 2, angle_vals, width, color=[colors[l] for l in labels], alpha=0.45, label="Angle variance")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")

    # Keep the figure clean: no axis labels, minimal ticks.
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.tick_params(axis="x", bottom=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")

    if show_legend:
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    _save_fig(fig, out_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style pose diversity comparison across easy/medium/hard splits."
    )
    parser.add_argument("--easy", type=Path, default=None, help="NPZ with pose_vecs and angles for Easy.")
    parser.add_argument("--medium", type=Path, default=None, help="NPZ with pose_vecs and angles for Medium.")
    parser.add_argument("--hard", type=Path, default=None, help="NPZ with pose_vecs and angles for Hard.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/pose_diversity"))
    parser.add_argument("--stem", type=str, default="pose_diversity_easy_medium_hard")
    parser.add_argument("--no-legend", action="store_true", help="Disable legend for minimal text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    forced_ratio = REQUIRED_SAMPLE_RATIO

    auto_defaults = False
    if args.easy is None and args.medium is None and args.hard is None:
        pct = int(round(forced_ratio * 100))
        args.easy = args.cache_dir / f"curvton_easy_{pct}pct.npz"
        args.medium = args.cache_dir / f"curvton_medium_{pct}pct.npz"
        args.hard = args.cache_dir / f"curvton_hard_{pct}pct.npz"
        auto_defaults = True

    for p in [args.easy, args.medium, args.hard]:
        if p is None or not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

    metrics: Dict[str, Tuple[float, float]] = {}
    for label, path in [
        ("Easy", args.easy),
        ("Medium", args.medium),
        ("Hard", args.hard),
    ]:
        pose_vecs, angles = _load_pose_arrays(path)
        if not auto_defaults:
            pose_vecs, angles = _subsample_pair(pose_vecs, angles, forced_ratio, args.seed)
        pose_score = _pose_diversity_score(pose_vecs)
        angle_score = _angle_diversity_score(angles)
        metrics[label] = (pose_score, angle_score)

    plot_pose_diversity(
        metrics=metrics,
        out_dir=args.out_dir,
        stem=args.stem,
        show_legend=not args.no_legend,
    )

    print("Pose diversity scores:")
    for label, (pose_score, angle_score) in metrics.items():
        print(f"  {label}: pose_var={pose_score:.4f}, angle_var={angle_score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
