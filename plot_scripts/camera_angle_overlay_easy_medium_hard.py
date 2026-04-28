from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


REQUIRED_SAMPLE_RATIO = 0.2


_ROOT = Path(__file__).resolve().parents[1]


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
    """Map azimuth to [-180, 180] for centered plotting."""
    az = np.remainder(az, 360.0)
    az = np.where(az > 180.0, az - 360.0, az)
    return az


def _smooth2d(H: np.ndarray, iters: int = 2) -> np.ndarray:
    kernel = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    kernel = kernel / kernel.sum()
    out = H.astype(np.float32)
    for _ in range(max(1, iters)):
        # Convolve rows
        out = np.pad(out, ((0, 0), (1, 1)), mode="edge")
        out = (
            out[:, :-2] * kernel[0]
            + out[:, 1:-1] * kernel[1]
            + out[:, 2:] * kernel[2]
        )
        # Convolve cols
        out = np.pad(out, ((1, 1), (0, 0)), mode="edge")
        out = (
            out[:-2, :] * kernel[0]
            + out[1:-1, :] * kernel[1]
            + out[2:, :] * kernel[2]
        )
    return out


def plot_camera_overlay(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    stem: str,
    bins_az: int,
    bins_el: int,
) -> None:
    _apply_eccv_style()
    colors = _difficulty_colors()

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150)

    for label in ["Easy", "Medium", "Hard"]:
        if label not in datasets:
            continue
        az, el = datasets[label]
        az = _wrap_azimuth(az)
        el = np.clip(el, -90.0, 90.0)
        H, xedges, yedges = np.histogram2d(az, el, bins=[bins_az, bins_el], range=[[-180, 180], [-90, 90]])
        H = _smooth2d(H, iters=2)
        X = 0.5 * (xedges[:-1] + xedges[1:])
        Y = 0.5 * (yedges[:-1] + yedges[1:])
        Xg, Yg = np.meshgrid(X, Y, indexing="xy")

        # Use percentile-based contour levels for stable overlays.
        levels = np.percentile(H[H > 0], [60, 80, 92]) if np.any(H > 0) else [1, 2, 3]
        ax.contour(
            Xg,
            Yg,
            H.T,
            levels=levels,
            colors=[colors[label]],
            linewidths=1.4,
            alpha=0.9,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#cccccc")

    legend_handles = [
        plt.Line2D([0], [0], color=colors["Easy"], linewidth=2.0, label="Easy"),
        plt.Line2D([0], [0], color=colors["Medium"], linewidth=2.0, label="Medium"),
        plt.Line2D([0], [0], color=colors["Hard"], linewidth=2.0, label="Hard"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=8)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style overlay of camera angle distributions for easy/medium/hard splits."
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
    parser.add_argument(
        "--curvton-root",
        type=Path,
        default=None,
        help="CurvTON base path (overrides CURVTON_ROOT).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/camera_angle_overlay"))
    parser.add_argument("--stem", type=str, default="camera_angle_overlay_easy_medium_hard")
    parser.add_argument("--bins-az", type=int, default=36)
    parser.add_argument("--bins-el", type=int, default=18)
    parser.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Disable auto-generation of missing CurvTON caches.",
    )
    return parser.parse_args()


def _resolve_curvton_root(curvton_root: Path | None) -> Path:
    if curvton_root is not None:
        return curvton_root
    env_root = os.getenv("CURVTON_ROOT") or os.getenv("CURVTON_BASE_PATH")
    if env_root:
        return Path(env_root)
    raise RuntimeError(
        "CurvTON base path not set. Provide --curvton-root or set CURVTON_ROOT."
    )


def _generate_curvton_caches(cache_dir: Path, sample_ratio: float, curvton_root: Path | None) -> None:
    base_path = _resolve_curvton_root(curvton_root)
    if not base_path.exists():
        raise FileNotFoundError(f"CurvTON base path not found: {base_path}")

    out_dir = Path("./outputs/curvton_eda_autogen")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = _ROOT / "EDA" / "run_curvton_eda.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--base_path",
        str(base_path),
        "--out_dir",
        str(out_dir),
        "--cache_dir",
        str(cache_dir),
        "--sample_ratio",
        str(sample_ratio),
        "--difficulties",
        "easy",
        "medium",
        "hard",
    ]
    print("[auto] Generating CurvTON caches via:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_ROOT))


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

    missing = [p for p in [args.easy, args.medium, args.hard] if p is None or not p.exists()]
    if missing and not args.no_auto_generate:
        if auto_defaults:
            _generate_curvton_caches(args.cache_dir, forced_ratio, args.curvton_root)
        else:
            print("[warn] Missing cache files detected; auto-generation is only enabled for defaults.")

    missing = [p for p in [args.easy, args.medium, args.hard] if p is None or not p.exists()]
    if missing:
        raise FileNotFoundError(f"File not found: {', '.join(str(p) for p in missing)}")

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

    plot_camera_overlay(
        datasets=datasets,
        out_dir=args.out_dir,
        stem=args.stem,
        bins_az=args.bins_az,
        bins_el=args.bins_el,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
