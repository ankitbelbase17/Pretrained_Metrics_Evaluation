from __future__ import annotations

import argparse
from pathlib import Path
import sys

from EDA.run_eda import run_all_plots
from EDA.plot_entry_common import find_dataset_cache, load_npz_dict, log_plot_error


def main() -> None:
    p = argparse.ArgumentParser(description="Plot VITON-HD EDA figures from cache")
    p.add_argument("--cache_root", type=str, default="cache_root")
    p.add_argument("--out_dir", type=str, default="assets")
    p.add_argument("--error_log", type=str, default="logs/plot_vitonhd_errors.log")
    p.add_argument("--no_pairplot", action="store_true")
    args = p.parse_args()

    try:
        cache_root = Path(args.cache_root)
        cache_path = find_dataset_cache(cache_root, "vitonhd")
        if cache_path is None:
            raise FileNotFoundError(f"Could not find VITON-HD cache under {cache_root}")

        all_data = {"VITON-HD": load_npz_dict(cache_path)}
        run_all_plots(all_data, out_root=args.out_dir, no_pairplot=args.no_pairplot)
    except Exception as e:
        log_plot_error("plot_vitonhd", args.error_log, e)
        print(f"[ERROR] Plotting failed. Details logged to: {args.error_log}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
