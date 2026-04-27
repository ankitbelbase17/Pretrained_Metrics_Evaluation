from __future__ import annotations

import argparse
from pathlib import Path
import sys

from EDA.run_eda import run_all_plots
from EDA.run_eric_plots import run_eric_6
from EDA.plot_entry_common import load_curvton_split_caches, log_plot_error


def main() -> None:
    p = argparse.ArgumentParser(description="Plot CurvTON split and pooled EDA figures from cache")
    p.add_argument("--curvton_cache_dir", type=str, default="cache_root/curvton")
    p.add_argument("--ratio_pct", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="assets")
    p.add_argument("--error_log", type=str, default="logs/plot_curvton_and_splits_errors.log")
    p.add_argument("--no_pairplot", action="store_true")
    p.add_argument("--skip_eric_plots", action="store_true", help="Skip Eric-specific plot generation")
    args = p.parse_args()

    try:
        curvton_cache_dir = Path(args.curvton_cache_dir)
        all_data = load_curvton_split_caches(curvton_cache_dir, ratio_pct=args.ratio_pct)
        if not all_data:
            raise FileNotFoundError(
                f"No CurvTON caches found under {curvton_cache_dir} for ratio={args.ratio_pct}pct"
            )

        out_root = Path(args.out_dir)

        # 1) Standard EDA plots for CurvTON splits/all.
        standard_out = out_root / "curvton_and_splits"
        run_all_plots(all_data, out_root=str(standard_out), no_pairplot=args.no_pairplot)

        # 2) Eric-specific plots in corresponding directories.
        if not args.skip_eric_plots:
            eric_root = out_root / "eric"

            # CurvTON splits + all (overlapped)
            run_eric_6(
                all_data,
                out_dir=str(eric_root / "curvton_and_splits"),
                tag="curvton_splits",
            )

            # CurvTON pooled/all only (single dataset)
            pooled = {}
            if "CurvTON-All" in all_data:
                pooled = {"CurvTON": all_data["CurvTON-All"]}
            elif "CurvTON" in all_data:
                pooled = {"CurvTON": all_data["CurvTON"]}

            if pooled:
                run_eric_6(
                    pooled,
                    out_dir=str(eric_root / "curvton"),
                    tag="curvton",
                )

        print(f"[OK] CurvTON plots saved under: {out_root.resolve()}")
    except Exception as e:
        log_plot_error("plot_curvton_and_splits", args.error_log, e)
        print(f"[ERROR] Plotting failed. Details logged to: {args.error_log}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
