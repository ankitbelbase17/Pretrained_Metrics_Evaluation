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


def _load_betas(npz_path: Path) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "betas" not in data:
        raise KeyError(f"Missing betas in {npz_path}")
    betas = data["betas"].astype(np.float32)
    if betas.ndim != 2:
        raise ValueError(f"betas expected shape (N, D), got {betas.shape}")
    return betas


def _subsample(betas: np.ndarray, sample_ratio: float, seed: int) -> np.ndarray:
    if sample_ratio >= 1.0:
        return betas
    if sample_ratio <= 0.0:
        raise ValueError("sample_ratio must be in (0, 1]")
    n = betas.shape[0]
    if n <= 1:
        return betas
    rng = np.random.default_rng(seed)
    n_keep = max(1, int(round(n * sample_ratio)))
    n_keep = min(n_keep, n)
    idx = rng.choice(n, size=n_keep, replace=False)
    idx = np.sort(idx)
    return betas[idx]


def _standardize(betas: np.ndarray) -> np.ndarray:
    mu = betas.mean(axis=0, keepdims=True)
    sig = betas.std(axis=0, keepdims=True) + 1e-8
    return (betas - mu) / sig


def _variation_score(betas: np.ndarray) -> float:
    if betas.shape[0] < 2:
        return 0.0
    z = _standardize(betas)
    return float(np.mean(np.var(z, axis=0)))


def _entropy_score(betas: np.ndarray, bins: int = 40) -> float:
    if betas.shape[0] < 2:
        return 0.0
    z = _standardize(betas)
    entropies = []
    for j in range(z.shape[1]):
        vals = z[:, j]
        counts, _ = np.histogram(vals, bins=bins, density=True)
        p = counts[counts > 0]
        if p.size == 0:
            continue
        p = p / p.sum()
        ent = -np.sum(p * np.log(p))
        entropies.append(float(ent))
    return float(np.mean(entropies)) if entropies else 0.0


def _save_fig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from EDA.plot_style import save_fig

        save_fig(fig, out_dir, stem)
    except Exception:
        fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def plot_body_shape_variation(
    metrics: Dict[str, Tuple[float, float]],
    betas_by_label: Dict[str, np.ndarray],
    out_dir: Path,
    stem: str,
    show_legend: bool,
) -> None:
    _apply_eccv_style()
    colors = _difficulty_colors()

    labels = ["Easy", "Medium", "Hard"]
    x = np.arange(len(labels))
    width = 0.34

    var_vals = [metrics[lbl][0] for lbl in labels]
    ent_vals = [metrics[lbl][1] for lbl in labels]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), dpi=150)
    ax_bar, ax_pdf, ax_cdf = axes

    # Bar plot: variance + entropy summary
    ax_bar.bar(x - width / 2, var_vals, width, color=[colors[l] for l in labels], alpha=0.85, label="Variance")
    ax_bar.bar(x + width / 2, ent_vals, width, color=[colors[l] for l in labels], alpha=0.45, label="Entropy")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax_bar.set_xlabel("")
    ax_bar.set_ylabel("")
    ax_bar.tick_params(axis="y", left=False, labelleft=False)
    ax_bar.tick_params(axis="x", bottom=False)

    # PDF/CDF overlays for standardized betas
    for label in labels:
        betas = betas_by_label.get(label)
        if betas is None or betas.size == 0:
            continue
        z = _standardize(betas).reshape(-1)
        z = z[np.isfinite(z)]
        if z.size < 5:
            continue
        color = colors[label]
        # PDF via KDE (seaborn available in repo stack)
        try:
            import seaborn as sns

            sns.kdeplot(z, ax=ax_pdf, color=color, linewidth=1.6)
        except Exception:
            hist, edges = np.histogram(z, bins=60, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax_pdf.plot(centers, hist, color=color, linewidth=1.4)

        # CDF via empirical distribution
        zs = np.sort(z)
        ys = np.linspace(0.0, 1.0, zs.size)
        ax_cdf.plot(zs, ys, color=color, linewidth=1.5)

    for ax in (ax_pdf, ax_cdf):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.grid(True, linestyle="--", alpha=0.2, linewidth=0.4)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#cccccc")

    if show_legend:
        ax_bar.legend(loc="upper right", framealpha=0.9, fontsize=8)

    fig.tight_layout()
    _save_fig(fig, out_dir, stem)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECCV-style body shape variation comparison across easy/medium/hard splits."
    )
    parser.add_argument("--easy", type=Path, default=None, help="NPZ with betas for Easy.")
    parser.add_argument("--medium", type=Path, default=None, help="NPZ with betas for Medium.")
    parser.add_argument("--hard", type=Path, default=None, help="NPZ with betas for Hard.")
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
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs/body_shape_variation"))
    parser.add_argument("--stem", type=str, default="body_shape_variation_easy_medium_hard")
    parser.add_argument("--no-legend", action="store_true", help="Disable legend for minimal text.")
    parser.add_argument("--entropy-bins", type=int, default=40)
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

    metrics: Dict[str, Tuple[float, float]] = {}
    betas_by_label: Dict[str, np.ndarray] = {}
    for label, path in [
        ("Easy", args.easy),
        ("Medium", args.medium),
        ("Hard", args.hard),
    ]:
        betas = _load_betas(path)
        if not auto_defaults:
            betas = _subsample(betas, forced_ratio, args.seed)
        var_score = _variation_score(betas)
        ent_score = _entropy_score(betas, bins=args.entropy_bins)
        metrics[label] = (var_score, ent_score)
        betas_by_label[label] = betas

    plot_body_shape_variation(
        metrics=metrics,
        betas_by_label=betas_by_label,
        out_dir=args.out_dir,
        stem=args.stem,
        show_legend=not args.no_legend,
    )

    print("Body shape variation scores:")
    for label, (var_score, ent_score) in metrics.items():
        print(f"  {label}: variance={var_score:.4f}, entropy={ent_score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
