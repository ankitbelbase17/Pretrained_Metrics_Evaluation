from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_metrics"))

from fallback_debug.common import (
    first_success_index,
    parse_common_args,
    print_attempt,
    print_env,
    print_summary,
    run_attempt,
)


def try_hmr2(device: str) -> str:
    from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2

    import torch.serialization as _ts
    from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig

    _ts.add_safe_globals([_DictConfig, _ListConfig])
    download_models()
    model, _cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device).eval()
    return f"shape_backend=hmr2, checkpoint={DEFAULT_CHECKPOINT}"


def try_vit_proxy(device: str) -> str:
    import timm

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(device).eval()
    return f"shape_backend=vit_proxy, model={model.__class__.__name__}"


def main():
    args = parse_common_args("M5 fallback debug (HMR2 -> ViT proxy)")
    print_env(args.device)

    attempts = [
        ("HMR2.0 (primary)", lambda: try_hmr2(args.device)),
        ("ViT-B/16 proxy (fallback)", lambda: try_vit_proxy(args.device)),
    ]

    rows: List[Tuple[str, bool, float]] = []
    selected = None
    for i, (name, fn) in enumerate(attempts):
        ok, detail, elapsed = run_attempt(name, fn, args.verbose)
        rows.append((name, ok, elapsed))
        print_attempt(name, ok, detail, elapsed)
        if ok and selected is None:
            selected = i
            if not args.all:
                break

    print_summary(rows)
    idx = first_success_index(rows)
    if idx is not None:
        print(f"Selected backend chain step: {attempts[idx][0]}")
    else:
        print("No backend loaded. This explains full metric failure.")


if __name__ == "__main__":
    main()

