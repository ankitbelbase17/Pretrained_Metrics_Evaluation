"""
test_EDA.py
===========
End-to-end EDA validation runner for all datasets in config.

What it does
------------
1) Reads dataset entries from YAML config.
2) Runs `EDA/run_eda.py` for each dataset entry (full dataset, no sampling).
3) Verifies plot outputs were generated.
4) Optionally runs `EDA/run_curvton_eda.py` with sample_ratio=1.0.
5) Prints a final status table.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


ROOT = Path(__file__).parent


def _fmt_table(headers: List[str], rows: List[List[object]], title: str) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    print("\n" + title)
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)


def _load_entries(config_path: Path, split_override: str | None = None) -> List[Dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults = dict(raw.get("defaults", {}))
    entries = [dict(e) for e in raw.get("datasets", [])]
    merged: List[Dict] = []
    for entry in entries:
        cfg = {**defaults, **entry}
        if split_override:
            cfg["split"] = split_override
        merged.append(cfg)

    expanded: List[Dict] = []
    for cfg in merged:
        name = str(cfg.get("name", "")).lower()
        if name == "dresscode" and str(cfg.get("dresscode_category", "")).lower() == "all":
            for cat in ("upper_body", "lower_body", "dresses"):
                c = dict(cfg)
                c["dresscode_category"] = cat
                expanded.append(c)
        else:
            expanded.append(cfg)

    seen = set()
    unique: List[Dict] = []
    for cfg in expanded:
        key = (
            str(cfg.get("name", "")),
            str(cfg.get("root", "")),
            str(cfg.get("split", "test")),
            str(cfg.get("dresscode_category", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(cfg)
    return unique


def _count_plot_files(path: Path) -> int:
    if not path.exists():
        return 0
    exts = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".webp"}
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _label_for(cfg: Dict) -> str:
    name = str(cfg["name"])
    split = str(cfg.get("split", "test"))
    cat = str(cfg.get("dresscode_category", ""))
    root = str(cfg.get("root", "")).replace("\\", "/")
    tail = root.rstrip("/").split("/")[-1]
    bits = [name, split]
    if cat:
        bits.append(cat)
    if tail:
        bits.append(tail)
    return "_".join(bits)


def _run_one_eda(cfg: Dict, args: argparse.Namespace) -> Tuple[bool, str, float, int, str]:
    dataset = str(cfg["name"])
    root = str(cfg["root"])
    split = str(cfg.get("split", "test"))
    category = str(cfg.get("dresscode_category", ""))
    label = _label_for(cfg)
    out_dir = Path(args.out_root) / label
    cache_dir = Path(args.cache_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "EDA/run_eda.py",
        "--dataset",
        dataset,
        "--root",
        root,
        "--split",
        split,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--cache_dir",
        str(cache_dir),
        "--out_dir",
        str(out_dir),
    ]
    if category:
        cmd += ["--dresscode_category", category]
    if args.no_pairplot:
        cmd += ["--no_pairplot"]
    if args.no_resume:
        cmd += ["--no_resume"]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    n_plots = _count_plot_files(out_dir)
    ok = (proc.returncode == 0) and (n_plots > 0)
    msg = "-" if ok else f"exit={proc.returncode}, plots={n_plots}"
    return ok, str(out_dir), dt, n_plots, msg


def _find_curvton_bases(entries: List[Dict]) -> List[str]:
    bases = []
    for cfg in entries:
        name = str(cfg.get("name", "")).lower()
        root = str(cfg.get("root", "")).replace("\\", "/").rstrip("/")
        if name != "curvton":
            continue
        tail = root.split("/")[-1].lower() if root else ""
        if tail in {"easy", "medium", "hard"}:
            continue
        bases.append(root)
    return sorted(set(bases))


def _run_curvton_special(base_path: str, args: argparse.Namespace) -> Tuple[bool, str, float, int, str]:
    label = f"curvton_special_{Path(base_path).name}"
    out_dir = Path(args.out_root) / label
    cache_dir = Path(args.cache_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "EDA/run_curvton_eda.py",
        "--base_path",
        base_path,
        "--out_dir",
        str(out_dir),
        "--cache_dir",
        str(cache_dir),
        "--sample_ratio",
        "1.0",
        "--difficulties",
        "easy",
        "medium",
        "hard",
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    n_plots = _count_plot_files(out_dir)
    ok = (proc.returncode == 0) and (n_plots > 0)
    msg = "-" if ok else f"exit={proc.returncode}, plots={n_plots}"
    return ok, str(out_dir), dt, n_plots, msg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test EDA pipeline on all datasets and all datapoints.")
    p.add_argument("--config", type=str, default="configs/pretrained_metrics_datasets.yaml")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--split", type=str, default=None, help="Optional split override for all datasets.")
    p.add_argument("--max_datasets", type=int, default=0, help="Debug cap (0 = all).")
    p.add_argument("--no_pairplot", action="store_true", help="Disable heavy pairplot.")
    p.add_argument("--no_resume", action="store_true", help="Re-extract features from scratch.")
    p.add_argument("--continue_on_error", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--run_curvton_special", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out_root", type=str, default="assets/eda_test/plots")
    p.add_argument("--cache_root", type=str, default="assets/eda_test/cache")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[FATAL] Config not found: {config_path}")
        return 1

    entries = _load_entries(config_path, split_override=args.split)
    if args.max_datasets > 0:
        entries = entries[: args.max_datasets]
    if not entries:
        print("[FATAL] No dataset entries selected.")
        return 1

    print("=" * 110)
    print("EDA Test Runner - Full Dataset Coverage")
    print(f"config={config_path}")
    print(f"entries={len(entries)} | batch_size={args.batch_size} | num_workers={args.num_workers}")
    print("=" * 110)

    rows: List[List[object]] = []
    n_fail = 0

    for cfg in entries:
        dataset = str(cfg["name"])
        split = str(cfg.get("split", "test"))
        category = str(cfg.get("dresscode_category", "-")) or "-"
        root = str(cfg["root"])
        label = _label_for(cfg)
        print("\n" + "-" * 110)
        print(f"[EDA] {dataset} | split={split} | category={category}")
        print(f"root={root}")
        print("-" * 110)

        ok, out_dir, dt, n_plots, msg = _run_one_eda(cfg, args)
        rows.append([dataset, split, category, label, "PASS" if ok else "FAIL", n_plots, f"{dt:.1f}s", out_dir, msg])
        if not ok:
            n_fail += 1
            if not args.continue_on_error:
                break

    if args.run_curvton_special:
        for base in _find_curvton_bases(entries):
            print("\n" + "-" * 110)
            print(f"[CURVTON_SPECIAL] base={base} | sample_ratio=100%")
            print("-" * 110)
            ok, out_dir, dt, n_plots, msg = _run_curvton_special(base, args)
            rows.append(["curvton_special", "test", "-", Path(base).name, "PASS" if ok else "FAIL", n_plots, f"{dt:.1f}s", out_dir, msg])
            if not ok:
                n_fail += 1
                if not args.continue_on_error:
                    break

    _fmt_table(
        ["Dataset", "Split", "Category", "Label", "Status", "Plots", "Time", "OutDir", "Error"],
        rows,
        "EDA Execution Summary",
    )

    print(f"\nCompleted entries={len(rows)} | failures={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

