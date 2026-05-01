from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _apply_eccv_style() -> None:
    try:
        from EDA.plot_style import apply_paper_style

        apply_paper_style()
    except Exception:
        plt.style.use("seaborn-v0_8-white")
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
                "font.size": 10,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        )


def _load_occ_maps(npz_path: Path) -> np.ndarray:
    data = dict(np.load(npz_path, allow_pickle=True))
    if "occ_maps" not in data:
        raise KeyError(f"Missing 'occ_maps' in {npz_path}")
    occ_maps = np.asarray(data["occ_maps"], dtype=np.float32)
    if occ_maps.ndim != 3:
        raise ValueError(f"Expected occ_maps shape (N,H,W), got {occ_maps.shape}")
    return occ_maps


def _mean_map(occ_maps: np.ndarray) -> np.ndarray:
    return occ_maps.mean(axis=0)


def plot_occ_maps_eccv(
    maps_by_label: Dict[str, np.ndarray],
    out_dir: Path,
    stem: str,
    cmap: str,
    show_colorbar: bool,
) -> None:
    _apply_eccv_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: List[str] = list(maps_by_label.keys())
    mean_maps = [_mean_map(maps_by_label[k]) for k in labels]
    vmax = max(float(m.max()) for m in mean_maps) if mean_maps else 1.0
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(1, len(labels), figsize=(7.0, 2.8), dpi=180, constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]

    im = None
    for ax, label, mm in zip(axes, labels, mean_maps):
        im = ax.imshow(
            mm,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            origin="upper",
            aspect="equal",
        )
        ax.set_title(label, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#D0D5DD")

    if show_colorbar and im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Occlusion Intensity", fontsize=8)

    fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot ECCV-style occlusion heatmaps from separate occ_maps caches."
    )
    p.add_argument("--occ-maps-dir", type=Path, required=True, help="Directory containing *_occ_maps.npz files.")
    p.add_argument("--sample-ratio", type=float, default=0.05, help="Sample ratio used in filename suffix.")
    p.add_argument("--out-dir", type=Path, default=Path("./outputs/occlusion_from_occ_maps"))
    p.add_argument("--stem", type=str, default="occlusion_heatmap_easy_medium_hard")
    p.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap name (e.g., magma, inferno).")
    p.add_argument("--no-colorbar", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pct = int(round(args.sample_ratio * 100))

    files = {
        "Easy": args.occ_maps_dir / f"curvton_easy_{pct}pct_occ_maps.npz",
        "Medium": args.occ_maps_dir / f"curvton_medium_{pct}pct_occ_maps.npz",
        "Hard": args.occ_maps_dir / f"curvton_hard_{pct}pct_occ_maps.npz",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing occ_maps cache files: " + ", ".join(missing))

    maps_by_label = {k: _load_occ_maps(v) for k, v in files.items()}
    plot_occ_maps_eccv(
        maps_by_label=maps_by_label,
        out_dir=args.out_dir,
        stem=args.stem,
        cmap=args.cmap,
        show_colorbar=not args.no_colorbar,
    )
    print(f"Saved: {args.out_dir / (args.stem + '.png')}")
    print(f"Saved: {args.out_dir / (args.stem + '.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

