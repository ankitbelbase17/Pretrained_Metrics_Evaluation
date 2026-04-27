from __future__ import annotations

import argparse
from pathlib import Path
import sys

from EDA.run_eda import run_all_plots
from EDA.plot_entry_common import (
    find_dataset_cache,
    load_curvton_split_caches,
    load_npz_dict,
    log_plot_error
)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot overlapped comparison EDA figures from cached datasets")
    p.add_argument("--cache_root", type=str, default="cache_root")
    p.add_argument("--curvton_cache_dir", type=str, default="cache_root/curvton")
    p.add_argument("--curvton_ratio_pct", type=int, default=100)
    p.add_argument("--include_curvton_splits", action="store_true", help="Include CurvTON easy/medium/hard/all instead of pooled only")
    p.add_argument("--out_dir", type=str, default="assets")
    p.add_argument("--error_log", type=str, default="logs/plot_overlapped_comparison_errors.log")
    p.add_argument("--no_pairplot", action="store_true")
    args = p.parse_args()

    try:
        cache_root = Path(args.cache_root)
        all_data = {}

        if args.include_curvton_splits:
            all_data.update(
                load_curvton_split_caches(Path(args.curvton_cache_dir), ratio_pct=args.curvton_ratio_pct)
            )
        else:
            cp = Path(args.curvton_cache_dir) / f"curvton_all_{args.curvton_ratio_pct}pct.npz"
            if cp.exists():
                all_data["CurvTON"] = load_npz_dict(cp)

        for label, key in [
            ("VITON-HD", "vitonhd"),
            ("DressCode", "dresscode"),
            ("StreetTryOn", "street_tryon"),
        ]:
            pth = find_dataset_cache(cache_root, key)
            if pth is not None:
                all_data[label] = load_npz_dict(pth)

        if len(all_data) < 2:
            raise RuntimeError(
                "Need at least 2 datasets for overlapped comparison plots. "
                "Check cache paths and availability."
            )

        run_all_plots(all_data, out_root=args.out_dir, no_pairplot=args.no_pairplot)
    except Exception as e:
        log_plot_error("plot_overlapped_comparison", args.error_log, e)
        print(f"[ERROR] Plotting failed. Details logged to: {args.error_log}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
