from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from curvton_cache_autogen import add_autogen_args, default_cache_paths, ensure_curvton_caches


REQUIRED_SAMPLE_RATIO = 0.2


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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


def _get_palette(label: str, idx: int) -> str:
    try:
        from EDA.plot_style import DATASET_COLORS, PALETTE

        return DATASET_COLORS.get(label, PALETTE[idx % len(PALETTE)])
    except Exception:
        fallback = ["#1B6CA8", "#D14D41", "#1B9E77", "#D95F02", "#7570B3"]
        return fallback[idx % len(fallback)]


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from EDA.plot_style import save_fig

        save_fig(fig, out_dir, stem)
    except Exception:
        fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _load_bg_features(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "bg_entropy" not in data:
        raise KeyError(f"Missing bg_entropy in {npz_path}")
    ent = data["bg_entropy"].astype(np.float32)
    obj = data.get("bg_obj_count", np.zeros(len(ent))).astype(np.float32)
    return ent, obj


def _subsample_pair(
    ent: np.ndarray,
    obj: np.ndarray,
    sample_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if sample_ratio >= 1.0:
        return ent, obj
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = ent.shape[0]
    if n <= 1:
        return ent, obj
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return ent[idx], obj[idx]


def _clean(ent: np.ndarray, obj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(ent) & np.isfinite(obj)
    return ent[mask], obj[mask]


def _print_stats(label: str, ent: np.ndarray, obj: np.ndarray) -> None:
    if len(ent) == 0:
        print(f"{label}: no valid samples")
        return
    print(
        f"{label}: n={len(ent)} | entropy mean={ent.mean():.3f} var={ent.var():.3f} "
        f"| object mean={obj.mean():.3f} var={obj.var():.3f}"
    )


def plot_entropy_histogram(
    datasets: Dict[str, np.ndarray],
    out_dir: Path,
    bins: int,
    show_legend: bool,
) -> None:
    _apply_eccv_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.6))
    legend_handles: List[Line2D] = []

    for i, (label, ent) in enumerate(datasets.items()):
        ent = ent[np.isfinite(ent)]
        if len(ent) == 0:
            continue
        color = _get_palette(label, i)
        ax1.hist(ent, bins=bins, density=True, histtype="step", color=color, linewidth=1.4, alpha=0.95)
        ax1.hist(ent, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.20)
        ax1.axvline(ent.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        sns.kdeplot(ent, ax=ax2, fill=True, alpha=0.20, color=color, linewidth=1.6)
        ax2.axvline(ent.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.6, label=label)
        )

    for ax in (ax1, ax2):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.4)

    if show_legend and legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)))

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_entropy_histogram")
    plt.close(fig)


def plot_object_histogram(
    datasets: Dict[str, np.ndarray],
    out_dir: Path,
    bins: int,
    show_legend: bool,
) -> None:
    _apply_eccv_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.875, 2.6))
    legend_handles: List[Line2D] = []

    for i, (label, obj) in enumerate(datasets.items()):
        obj = obj[np.isfinite(obj)]
        if len(obj) == 0:
            continue
        color = _get_palette(label, i)
        ax1.hist(obj, bins=bins, density=True, histtype="step", color=color, linewidth=1.4, alpha=0.95)
        ax1.hist(obj, bins=bins, density=True, histtype="stepfilled", color=color, alpha=0.20)
        ax1.axvline(obj.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        sns.kdeplot(obj, ax=ax2, fill=True, alpha=0.20, color=color, linewidth=1.6)
        ax2.axvline(obj.mean(), color=color, linestyle="--", linewidth=1.1, alpha=0.9)

        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=1.6, label=label)
        )

    for ax in (ax1, ax2):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.4)

    if show_legend and legend_handles:
        fig.legend(handles=legend_handles, loc="upper center", ncol=min(4, len(legend_handles)))

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_object_density_histogram")
    plt.close(fig)


def plot_entropy_vs_objects(
    datasets_ent: Dict[str, np.ndarray],
    datasets_obj: Dict[str, np.ndarray],
    out_dir: Path,
    show_legend: bool,
) -> None:
    _apply_eccv_style()

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    legend_handles: List[Line2D] = []

    for i, (label, ent) in enumerate(datasets_ent.items()):
        obj = datasets_obj.get(label, np.zeros(len(ent)))
        ent, obj = _clean(ent, obj)
        if len(ent) == 0:
            continue
        color = _get_palette(label, i)
        ax.scatter(
            obj,
            ent,
            s=10,
            alpha=0.5,
            color=color,
            linewidths=0.2,
            edgecolors="white",
            rasterized=True,
        )
        legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, label=label))

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(True, linestyle="--", alpha=0.25, linewidth=0.4)

    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="best", framealpha=0.9)

    fig.tight_layout()
    _save_fig(fig, out_dir, "bg_entropy_vs_objects")
    plt.close(fig)


def plot_density_heatmap(
    ent: np.ndarray,
    obj: np.ndarray,
    out_dir: Path,
    stem: str,
    bins: int,
    vmin: float,
    vmax: float,
) -> None:
    _apply_eccv_style()

    ent, obj = _clean(ent, obj)
    if len(ent) == 0:
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    h = ax.hist2d(obj, ent, bins=bins, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(False)

    cbar = fig.colorbar(h[3], ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("", fontsize=7)

    fig.tight_layout()
    _save_fig(fig, out_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style EDA plots for background entropy and object density."
    )
    parser.add_argument("--features", nargs="+", default=None, help="NPZ files with bg_entropy/bg_obj_count.")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each NPZ file.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./eda_cache/curvton"),
        help="Default cache directory to resolve easy/medium/hard NPZs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/background_eda"))
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--heatmap-bins", type=int, default=40)
    parser.add_argument("--heatmap-vmin", type=float, default=0.0)
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=-1.0,
        help="Max value for heatmap. If negative, uses data max.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Disable legends for cleaner visuals.",
    )
    add_autogen_args(parser)
    return parser.parse_args()


def _require_cache_files(paths: List[Path]) -> None:
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
    if args.features is None and args.labels is None:
        pct = int(round(forced_ratio * 100))
        args.features = [
            args.cache_dir / f"curvton_easy_{pct}pct.npz",
            args.cache_dir / f"curvton_medium_{pct}pct.npz",
            args.cache_dir / f"curvton_hard_{pct}pct.npz",
        ]
        args.labels = ["Easy", "Medium", "Hard"]
        auto_defaults = True

    if not args.features or not args.labels:
        raise ValueError("Provide --features and --labels, or omit both to use defaults.")
    if len(args.features) != len(args.labels):
        raise ValueError("--features and --labels must have the same length")

    missing = [Path(p) for p in args.features if not Path(p).exists()]
    if missing and auto_defaults:
        defaults = default_cache_paths(args.cache_dir, forced_ratio)
        missing_diffs = [diff for diff, path in defaults.items() if path in missing]
        ensure_curvton_caches(
            missing_diffs,
            args,
            forced_ratio,
            required_keys=["bg_entropy", "bg_obj_count"],
        )
        missing = [Path(p) for p in args.features if not Path(p).exists()]
    _require_cache_files([Path(p) for p in args.features])

    ent_data: Dict[str, np.ndarray] = {}
    obj_data: Dict[str, np.ndarray] = {}

    for path, label in zip(args.features, args.labels):
        ent, obj = _load_bg_features(Path(path))
        if not auto_defaults:
            ent, obj = _subsample_pair(ent, obj, forced_ratio, args.seed)
        ent, obj = _clean(ent, obj)
        ent_data[label] = ent
        obj_data[label] = obj
        _print_stats(label, ent, obj)

    show_legend = not args.no_legend

    plot_entropy_histogram(ent_data, args.out_dir, bins=args.bins, show_legend=show_legend)
    plot_object_histogram(obj_data, args.out_dir, bins=args.bins, show_legend=show_legend)
    plot_entropy_vs_objects(ent_data, obj_data, args.out_dir, show_legend=show_legend)

    # Combined density heatmap
    all_ent = np.concatenate(list(ent_data.values())) if ent_data else np.array([])
    all_obj = np.concatenate(list(obj_data.values())) if obj_data else np.array([])
    vmax = args.heatmap_vmax
    if vmax < 0 and len(all_ent) > 0:
        h, _, _ = np.histogram2d(all_obj, all_ent, bins=args.heatmap_bins)
        vmax = float(h.max()) if h.size else 1.0
    plot_density_heatmap(
        all_ent,
        all_obj,
        args.out_dir,
        stem="bg_entropy_object_density_heatmap",
        bins=args.heatmap_bins,
        vmin=args.heatmap_vmin,
        vmax=vmax,
    )

    # Per-dataset density heatmaps
    for label in ent_data:
        plot_density_heatmap(
            ent_data[label],
            obj_data[label],
            args.out_dir,
            stem=f"bg_entropy_object_density_heatmap_{label.lower().replace(' ', '_')}",
            bins=args.heatmap_bins,
            vmin=args.heatmap_vmin,
            vmax=vmax,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
