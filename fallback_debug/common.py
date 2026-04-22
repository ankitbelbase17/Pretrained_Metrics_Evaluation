from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback
from typing import Callable, List, Optional, Tuple

import torch


def parse_common_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to initialize model backends on",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print full Python traceback for failed attempts",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Try all backends even after first successful one",
    )
    return p.parse_args()


def print_env(device: str):
    print("=" * 90)
    print("Fallback Debug Environment")
    print("=" * 90)
    print(f"python           : {sys.executable}")
    print(f"python_version   : {platform.python_version()}")
    print(f"platform         : {platform.platform()}")
    print(f"cwd              : {os.getcwd()}")
    print(f"torch            : {torch.__version__}")
    print(f"cuda_available   : {torch.cuda.is_available()}")
    print(f"requested_device : {device}")
    print("=" * 90 + "\n")


def run_attempt(
    name: str,
    fn: Callable[[], str],
    verbose: bool,
) -> Tuple[bool, str, float]:
    t0 = time.time()
    try:
        info = fn()
        return True, info, time.time() - t0
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        if verbose:
            err = f"{err}\n{traceback.format_exc()}"
        return False, err, time.time() - t0


def print_attempt(name: str, ok: bool, detail: str, elapsed: float):
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name} ({elapsed:.1f}s)")
    print(f"       {detail}")


def print_summary(rows: List[Tuple[str, bool, float]]):
    print("\n" + "-" * 90)
    print("Summary")
    print("-" * 90)
    for name, ok, elapsed in rows:
        print(f"{name:40} | {'LOADED' if ok else 'NOT LOADED':10} | {elapsed:6.1f}s")
    print("-" * 90 + "\n")


def first_success_index(rows: List[Tuple[str, bool, float]]) -> Optional[int]:
    for i, (_, ok, _) in enumerate(rows):
        if ok:
            return i
    return None

