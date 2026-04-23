"""
test_appearance.py
==================
Standalone smoke-test for the M6 Appearance (face-embedding diversity) metric.

What this script does:
1. Builds a dataloader from CurvTON hard split.
2. Loads the AppearanceMetrics (M6) module.
3. Runs update + compute over the loaded batches.
4. Prints a detailed audit report with backend info, computed values,
   and per-chain model status.

Usage
-----
# Quick smoke-test (2 batches)
python test_appearance.py --max_batches 2

# Full split
python test_appearance.py --max_batches 0

# CPU-only
python test_appearance.py --device cpu --max_batches 4
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pretrained_metrics.cache_setup import configure_model_caches, DEFAULT_MODEL_BASE

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_metrics"))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _cyan(s: str) -> str:
    return f"\033[96m{s}\033[0m"


def _fmt_value(v: object) -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "NA"
        return f"{v:.6g}"
    if isinstance(v, (np.floating,)):
        fv = float(v)
        if np.isnan(fv):
            return "NA"
        return f"{fv:.6g}"
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if v is None:
        return "NA"
    return str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChainModel:
    label: str
    token: str


@dataclass
class MetricAudit:
    key: str
    metric: str
    status: str
    selected_backend: Optional[str]
    fallback_used: Optional[bool]
    chains: Dict[str, List[ChainModel]]
    selected_by_chain: Dict[str, Optional[str]]
    notes: List[str]
    error: Optional[str] = None
    computed_values: Optional[Dict[str, object]] = None
    elapsed_s: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Dataloader
# ─────────────────────────────────────────────────────────────────────────────

def _to_bchw(x: torch.Tensor, name: str) -> torch.Tensor:
    """Ensure image tensor is B,C,H,W with C=3."""
    if x.ndim != 4:
        raise RuntimeError(f"[{name}] Expected 4-D tensor, got shape {tuple(x.shape)}")
    if x.shape[1] != 3 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    if x.shape[1] != 3:
        raise RuntimeError(f"[{name}] Expected channel dimension C=3, got shape {tuple(x.shape)}")
    return x


def _collect_batches(args) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    from pretrained_metrics.dataloader import get_dataloader

    loader = get_dataloader(
        dataset_name=args.dataset_name,
        root=args.curvton_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size),
    )

    batches: List[Dict[str, torch.Tensor]] = []
    n_images = 0
    for i, batch in enumerate(loader):
        person = _to_bchw(batch["person"].float(), "person")
        cloth = _to_bchw(batch["cloth"].float(), "cloth")
        batches.append({"person": person, "cloth": cloth})
        n_images += int(person.shape[0])
        if args.max_batches > 0 and (i + 1) >= args.max_batches:
            break

    if not batches:
        raise RuntimeError("No batches were loaded from the dataloader.")
    return batches, len(batches), n_images


# ─────────────────────────────────────────────────────────────────────────────
# M6 Appearance probe
# ─────────────────────────────────────────────────────────────────────────────

def _run_updates(metric_obj, batches: List[Dict[str, torch.Tensor]]):
    """Feed person images through the metric and return compute() dict."""
    for b in batches:
        metric_obj.update(b["person"])
    return metric_obj.compute()


def _probe_appearance(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    """Load and evaluate the M6 Appearance metric."""
    chains = {
        "face_embedder": [
            ChainModel("InsightFace ArcFace", "arcface"),
            ChainModel("open_clip ViT-B/32", "open_clip"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m6_appearance import AppearanceMetrics

        obj = AppearanceMetrics(device=device)
        result = _run_updates(obj, batches)
        backend = getattr(obj._embedder, "_backend", None)
        return MetricAudit(
            key="m6",
            metric="M6 Appearance (face diversity)",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "arcface"),
            chains=chains,
            selected_by_chain={"face_embedder": backend},
            notes=[f"compute_keys={sorted(result.keys())}", f"backend={backend}"],
            computed_values=result,
        )
    except Exception as e:
        return MetricAudit(
            key="m6",
            metric="M6 Appearance (face diversity)",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"face_embedder": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Chain status helper
# ─────────────────────────────────────────────────────────────────────────────

def _chain_statuses(
    chain: List[ChainModel],
    selected_token: Optional[str],
) -> List[Tuple[str, str]]:
    if not chain:
        return []
    tokens = [c.token for c in chain]
    if selected_token in tokens:
        sel_idx = tokens.index(selected_token)
        out: List[Tuple[str, str]] = []
        for i, c in enumerate(chain):
            if i == sel_idx:
                out.append((c.label, "LOADED"))
            elif i < sel_idx:
                out.append((c.label, "NOT LOADED"))
            else:
                out.append((c.label, "NOT ATTEMPTED"))
        return out
    return [(c.label, "NOT LOADED") for c in chain]


# ─────────────────────────────────────────────────────────────────────────────
# Printing / reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_metric_audit(a: MetricAudit):
    header = f"[{a.key.upper()}] {a.metric}"
    st = _green("LOADED") if a.status == "LOADED" else _red("NOT LOADED")
    print(f"  {st:<20} {header}")
    if a.selected_backend is not None:
        print(f"      selected_backend : {a.selected_backend}")
    if a.fallback_used is not None:
        print(f"      fallback_used    : {a.fallback_used}")
    for n in a.notes:
        print(f"      note             : {n}")
    if a.status == "LOADED" and a.computed_values:
        print("      computed_values  :")
        for k in sorted(a.computed_values.keys()):
            print(f"        - {k} = {_fmt_value(a.computed_values[k])}")
    else:
        print("      computed_values  : NA")
    for chain_name, chain in a.chains.items():
        selected_token = a.selected_by_chain.get(chain_name)
        statuses = _chain_statuses(chain, selected_token)
        if not statuses:
            print(f"      {chain_name}: no pretrained model required")
            continue
        print(f"      {chain_name}:")
        for label, status in statuses:
            color_status = status
            if status == "LOADED":
                color_status = _green(status)
            elif status == "NOT LOADED":
                color_status = _red(status)
            elif status == "NOT ATTEMPTED":
                color_status = _yellow(status)
            print(f"        - {label} -> {color_status}")
    if a.error:
        print(f"      error            : {_red(a.error)}")


def _compact_values(values: Optional[Dict[str, object]]) -> str:
    if not values:
        return "NA"
    parts: List[str] = []
    for k in sorted(values.keys()):
        parts.append(f"{k}={_fmt_value(values[k])}")
    return ", ".join(parts)


def _fmt_cell(v: object) -> str:
    return str(v) if v is not None else "-"


def _print_table(title: str, headers: List[str], rows: List[List[object]]):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(_fmt_cell(c)))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    print(title)
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(_fmt_cell(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_appearance_test(args) -> int:
    cache_info = configure_model_caches(args.download_base, set_home_for_hmr2=False)

    print("\n" + "=" * 90)
    print("  M6 Appearance Metric — Standalone Smoke-Test")
    print(f"  device={args.device} | root={args.curvton_root}")
    print(f"  batch_size={args.batch_size} | max_batches={args.max_batches} | split={args.split}")
    print(f"  download_base={cache_info['base_path']}")
    print("=" * 90)

    # ── Load data ─────────────────────────────────────────────────────────
    try:
        batches, n_batches, n_images = _collect_batches(args)
        print(f"  Loaded batches: {n_batches} | images: {n_images}")
    except Exception as e:
        print(f"  {_red('FAILED')} dataloader setup: {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

    # ── Run M6 ────────────────────────────────────────────────────────────
    print("\n" + _cyan("M6 APPEARANCE METRIC AUDIT"))
    print("-" * 90)

    t0 = time.time()
    audit = None
    try:
        audit = _probe_appearance(args.device, batches)
    except Exception as e:
        print(f"  {_red('NOT LOADED'):<20} [M6] internal checker failure")
        print(f"      error            : {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1
    dt = time.time() - t0

    if audit is not None:
        audit.elapsed_s = dt
        _print_metric_audit(audit)
    print(f"      elapsed          : {dt:.1f}s\n")

    # ── Summary tables ────────────────────────────────────────────────────
    print("=" * 90)
    print("  Summary")
    print("=" * 90)

    if audit and audit.status == "LOADED":
        print(f"  {_green('M6 Appearance')}: LOADED ({audit.selected_backend})")
    else:
        print(f"  {_red('M6 Appearance')}: NOT LOADED")
        if audit and audit.error:
            print(f"  Error: {audit.error}")

    # Metric overview table
    if audit:
        _print_table(
            "\n  Metric Overview",
            ["Key", "Metric", "Status", "Backend", "Fallback", "Values", "Time"],
            [[
                audit.key.upper(),
                audit.metric,
                audit.status,
                audit.selected_backend or "-",
                audit.fallback_used if audit.fallback_used is not None else "-",
                _compact_values(audit.computed_values) if audit.status == "LOADED" else "NA",
                f"{audit.elapsed_s:.2f}s" if audit.elapsed_s is not None else "NA",
            ]],
        )

    # Detailed values table
    if audit and audit.status == "LOADED" and audit.computed_values:
        value_rows = []
        for mk in sorted(audit.computed_values.keys()):
            value_rows.append([audit.key.upper(), mk, _fmt_value(audit.computed_values[mk]), "OK"])
        _print_table(
            "\n  Detailed Metric Values",
            ["Metric", "Value Key", "Value", "Note"],
            value_rows,
        )
    print()

    failed = 0 if (audit and audit.status == "LOADED") else 1
    status_str = _green("PASSED") if failed == 0 else _red("FAILED")
    print(f"  Test result: {status_str}")
    print("=" * 90 + "\n")
    return failed


def _parse():
    p = argparse.ArgumentParser(
        description="Standalone smoke-test for the M6 Appearance metric (face diversity)."
    )
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--download_base", type=str, default=DEFAULT_MODEL_BASE)
    p.add_argument(
        "--curvton_root",
        type=str,
        default="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test/hard",
        help="Absolute path to CurvTON hard split root",
    )
    p.add_argument("--dataset_name", type=str, default="curvton",
                   help="Dataset registry name")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_size", type=int, nargs=2, default=[512, 384],
                   metavar=("H", "W"))
    p.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Number of batches to process; 0 = full split",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.batch_size < 4:
        print(f"[Config] Requested batch_size={args.batch_size} is below minimum; using 4.")
        args.batch_size = 4
    n_failed = run_appearance_test(args)
    sys.exit(0 if n_failed == 0 else 1)
