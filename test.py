"""
Multi-dataset sequential metric test runner.

This script loads dataset entries from a YAML config and runs the pretrained
metric suite sequentially for each dataset/split entry. At the end, it prints
summary tables for status and key metric values.

Default config:
  configs/pretrained_metrics_datasets.yaml

Usage:
  python test.py
  python test.py --config configs/pretrained_metrics_datasets.yaml
  python test.py --only_datasets curvton viton_hd
  python test.py --max_datasets 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pretrained_metrics.cache_setup import configure_model_caches, DEFAULT_MODEL_BASE
from pretrained_metrics.compute_pretrained_metrics import evaluate_one_dataset


def _fmt(v: object) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        if math.isnan(v):
            return "NA"
        return f"{v:.4f}"
    return str(v)


def _table(title: str, headers: List[str], rows: List[List[object]]):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    print(title)
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential multi-dataset test runner for pretrained metrics")
    p.add_argument("--config", type=str, default="configs/pretrained_metrics_datasets.yaml")
    p.add_argument("--download_base", type=str, default=DEFAULT_MODEL_BASE)
    p.add_argument("--only_datasets", nargs="*", default=None,
                   help="Optional dataset name filter, e.g. curvton viton_hd dresscode")
    p.add_argument("--max_datasets", type=int, default=0,
                   help="Run at most N entries from config (0 = all)")
    p.add_argument(
        "--continue_on_error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue all entries even if one entry fails (default: enabled).",
    )
    p.add_argument("--device", type=str, default=None,
                   help="Override device from config defaults")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size from config defaults")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Override num_workers from config defaults")
    p.add_argument("--split", type=str, default=None,
                   help="Override split for all entries")
    return p.parse_args()


def main() -> int:
    args = _parse()

    cache_info = configure_model_caches(args.download_base, set_home_for_hmr2=True)
    print("=" * 110)
    print("Sequential Test Runner: M1-M9 across configured datasets/splits")
    print(f"config={args.config}")
    print(f"download_base={cache_info['base_path']}")
    print("=" * 110)

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"[FATAL] Config file not found: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = dict(raw.get("defaults", {}))
    if args.device is not None:
        defaults["device"] = args.device
    if args.batch_size is not None:
        defaults["batch_size"] = args.batch_size
    if args.num_workers is not None:
        defaults["num_workers"] = args.num_workers
    if args.split is not None:
        defaults["split"] = args.split

    entries = list(raw.get("datasets", []))
    if not entries:
        print("[FATAL] No dataset entries found in config.")
        return 1

    if args.only_datasets:
        filt = {x.lower() for x in args.only_datasets}
        entries = [e for e in entries if str(e.get("name", "")).lower() in filt]

    if args.max_datasets > 0:
        entries = entries[: args.max_datasets]

    if not entries:
        print("[INFO] No dataset entries selected after filtering.")
        return 0

    status_rows: List[List[object]] = []
    metric_rows: List[List[object]] = []

    n_fail = 0
    started = time.time()

    for idx, entry in enumerate(entries, start=1):
        cfg = {**defaults, **dict(entry)}
        name = str(cfg.pop("name"))
        root = str(cfg.pop("root"))
        cfg.pop("pred_dir", None)

        label_split = str(cfg.get("split", "test"))
        label_cat = str(cfg.get("dresscode_category", "-"))
        tag = f"[{idx}/{len(entries)}] dataset={name} split={label_split} category={label_cat}"

        print("\n" + "-" * 110)
        print(tag)
        print("-" * 110)

        t0 = time.time()
        try:
            result = evaluate_one_dataset(name, root, cfg)
            elapsed = time.time() - t0

            if not result:
                n_fail += 1
                status_rows.append([name, label_split, label_cat, "FAILED", "no result", f"{elapsed:.2f}s"])
                if not args.continue_on_error:
                    break
                continue

            status_rows.append([name, label_split, label_cat, "OK", int(result.get("n_samples", 0)), f"{elapsed:.2f}s"])

            metric_rows.append([
                name,
                label_split,
                label_cat,
                _fmt(result.get("pose_diversity")),
                _fmt(result.get("occlusion_complexity")),
                _fmt(result.get("bg_entropy_mean")),
                _fmt(result.get("illumination_complexity")),
                _fmt(result.get("shape_diversity_logdet")),
                _fmt(result.get("appearance_diversity_mean")),
                _fmt(result.get("garment_diversity_logdet")),
                _fmt(result.get("vae_diversity_logdet")),
                _fmt(result.get("camera_diversity_score")),
                _fmt(result.get("category_complexity_mean_0_1")),
            ])

        except Exception as e:
            elapsed = time.time() - t0
            n_fail += 1
            status_rows.append([name, label_split, label_cat, "FAILED", f"{type(e).__name__}: {e}", f"{elapsed:.2f}s"])
            if not args.continue_on_error:
                break

    print("\n" + "=" * 110)
    print("Final Report")
    print("=" * 110)

    _table(
        "Dataset Execution Status",
        ["Dataset", "Split", "Category", "Status", "Samples/Error", "Time"],
        status_rows,
    )

    if metric_rows:
        _table(
            "Per-Dataset Metrics (by split)",
            [
                "Dataset",
                "Split",
                "Category",
                "pose_div",
                "occ_cmp",
                "bg_ent",
                "illum_cmp",
                "shape_logdet",
                "appear_mean",
                "garment_logdet",
                "vae_logdet",
                "camera_score",
                "overall_0_1",
            ],
            metric_rows,
        )

    total_elapsed = time.time() - started
    print(f"\nCompleted {len(status_rows)} entries in {total_elapsed:.2f}s | failures={n_fail}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
