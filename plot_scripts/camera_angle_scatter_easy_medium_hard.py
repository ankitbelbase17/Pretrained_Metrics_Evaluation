from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from curvton_cache_autogen import (
    add_autogen_args,
    caches_requiring_generation,
    ensure_curvton_caches,
)


REQUIRED_SAMPLE_RATIO = 0.25


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


def _load_angles(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = dict(np.load(npz_path, allow_pickle=True))
    az = data.get("azimuths", data.get("azimuth", np.array([])))
    el = data.get("elevations", data.get("elevation", np.array([])))
    az = np.asarray(az, dtype=np.float32)
    el = np.asarray(el, dtype=np.float32)
    if az.size == 0:
        raise KeyError(f"Missing azimuths in {npz_path}")
    if el.size == 0:
        el = np.zeros_like(az)
    mask = np.isfinite(az) & np.isfinite(el)
    return az[mask], el[mask]


def _subsample_pair(az: np.ndarray, el: np.ndarray, sample_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if sample_ratio >= 1.0:
        return az, el
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = az.shape[0]
    if n <= 1:
        return az, el
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return az[idx], el[idx]


def _wrap_azimuth(az: np.ndarray) -> np.ndarray:
    az = np.remainder(az, 360.0)
    return np.where(az < 0.0, az + 360.0, az)


def _iqr_bounds(values: np.ndarray) -> Tuple[float, float]:
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = max(1e-6, q3 - q1)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def _compute_global_bounds(datasets: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    az_all = np.concatenate([_wrap_azimuth(v[0]) for v in datasets.values()])
    el_all = np.concatenate([v[1] for v in datasets.values()])

    az_low, az_high = _iqr_bounds(az_all)
    el_low, el_high = _iqr_bounds(el_all)

    az_low = max(0.0, az_low)
    az_high = min(360.0, az_high)
    el_low = max(-45.0, el_low)
    el_high = min(45.0, el_high)

    if az_high <= az_low:
        az_low, az_high = 0.0, 360.0
    if el_high <= el_low:
        el_low, el_high = -45.0, 45.0

    return (az_low, az_high), (el_low, el_high)


def _smooth2d(H: np.ndarray, iters: int = 2) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    kernel = kernel / kernel.sum()
    out = H.astype(np.float32)
    for _ in range(max(1, iters)):
        out = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = (
            out[:, :-2] * kernel[0]
            + out[:, 1:-1] * kernel[1]
            + out[:, 2:] * kernel[2]
        )
        out = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = (
            out[:-2, :] * kernel[0]
            + out[1:-1, :] * kernel[1]
            + out[2:, :] * kernel[2]
        )
    return out


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from EDA.plot_style import save_fig

        save_fig(fig, out_dir, stem)
    except Exception:
        fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_camera_scatter(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    stem: str,
    bins_az: int,
    bins_el: int,
    show_contours: bool,
) -> None:
    _apply_eccv_style()
    colors = _difficulty_colors()

    (az_low, az_high), (el_low, el_high) = _compute_global_bounds(datasets)
    az_span = max(1e-6, az_high - az_low)
    el_span = max(1e-6, el_high - el_low)

    fig, ax = plt.subplots(figsize=(6.8, 4.8), dpi=150)

    for label in ["Easy", "Medium", "Hard"]:
        if label not in datasets:
            continue
        az, el = datasets[label]
        az = _wrap_azimuth(az)
        az = np.clip(az, az_low, az_high)
        el = np.clip(el, el_low, el_high)
        az_plot = (az - az_low) / az_span * 360.0
        el_plot = (el - el_low) / el_span * 90.0

        ax.scatter(
            az_plot,
            el_plot,
            s=12,
            color=colors[label],
            alpha=0.35,
            edgecolors="white",
            linewidths=0.2,
        )

        if show_contours:
            H, xedges, yedges = np.histogram2d(
                az_plot,
                el_plot,
                bins=[bins_az, bins_el],
                range=[[0, 360], [0, 90]],
            )
            H = _smooth2d(H, iters=2)
            X = 0.5 * (xedges[:-1] + xedges[1:])
            Y = 0.5 * (yedges[:-1] + yedges[1:])
            Xg, Yg = np.meshgrid(X, Y, indexing="xy")
            levels = np.percentile(H[H > 0], [70, 85, 95]) if np.any(H > 0) else [1, 2, 3]
            ax.contour(
                Xg,
                Yg,
                H.T,
                levels=levels,
                colors=[colors[label]],
                linewidths=1.2,
                alpha=0.9,
            )

    ax.set_xlabel("Azimuth (deg)")
    ax.set_ylabel("Elevation (deg)")
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 90)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#bdbdbd")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", color=colors["Easy"], markersize=6, label="Easy"),
        plt.Line2D([0], [0], marker="o", linestyle="None", color=colors["Medium"], markersize=6, label="Medium"),
        plt.Line2D([0], [0], marker="o", linestyle="None", color=colors["Hard"], markersize=6, label="Hard"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=8)

    _save_fig(fig, out_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style camera angle scatter plot with easy/medium/hard overlays."
    )
    parser.add_argument("--easy", type=Path, default=None, help="NPZ with azimuths/elevations for Easy.")
    parser.add_argument("--medium", type=Path, default=None, help="NPZ with azimuths/elevations for Medium.")
    parser.add_argument("--hard", type=Path, default=None, help="NPZ with azimuths/elevations for Hard.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/camera_angle_scatter"))
    parser.add_argument("--stem", type=str, default="camera_angle_scatter_easy_medium_hard")
    parser.add_argument("--bins-az", type=int, default=36)
    parser.add_argument("--bins-el", type=int, default=18)
    parser.add_argument("--no-contours", action="store_true", help="Disable density contours.")
    add_autogen_args(parser)
    return parser.parse_args()


def _require_cache_files(paths: Tuple[Path, ...]) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Missing cache files: "
            f"{missing_list}. Provide --curvton-root or pre-generate the caches."
        )


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

    if auto_defaults:
        missing_diffs = caches_requiring_generation(
            cache_dir=args.cache_dir,
            sample_ratio=forced_ratio,
            difficulties=["easy", "medium", "hard"],
            required_keys=["azimuths", "elevations"],
        )
    else:
        missing_diffs = []
    if missing_diffs:
        ensure_curvton_caches(
            missing_diffs,
            args,
            forced_ratio,
            required_keys=["azimuths", "elevations"],
        )
    _require_cache_files((args.easy, args.medium, args.hard))

    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for label, path in [
        ("Easy", args.easy),
        ("Medium", args.medium),
        ("Hard", args.hard),
    ]:
        az, el = _load_angles(path)
        if not auto_defaults:
            az, el = _subsample_pair(az, el, forced_ratio, args.seed)
        datasets[label] = (az, el)

    plot_camera_scatter(
        datasets=datasets,
        out_dir=args.out_dir,
        stem=args.stem,
        bins_az=args.bins_az,
        bins_el=args.bins_el,
        show_contours=not args.no_contours,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
