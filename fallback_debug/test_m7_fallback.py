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
    setup_caches,
)


def try_openai_clip(device: str) -> str:
    import clip as _oa_clip

    if not hasattr(_oa_clip, "load"):
        raise ImportError(
            "'clip' package is not OpenAI CLIP; expected pip package openai-clip"
        )
    model, _pre = _oa_clip.load("ViT-B/32", device=device)
    model.eval()
    return "garment_backend=openai_clip"


def try_open_clip(device: str) -> str:
    import open_clip

    model, _, _pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    return "garment_backend=open_clip"


def try_hf_clip(device: str) -> str:
    from transformers import CLIPModel, CLIPProcessor

    _ = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    return f"garment_backend=hf_clip, model={model.__class__.__name__}"


def try_vit(device: str) -> str:
    import timm

    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(device).eval()
    return f"garment_backend=vit_proxy, model={model.__class__.__name__}"


def main():
    args = parse_common_args("M7 fallback debug (OpenAI CLIP -> open_clip -> HF CLIP -> ViT)")
    print_env(args.device)
    setup_caches(args.download_base)

    attempts = [
        ("OpenAI CLIP ViT-B/32 (primary)", lambda: try_openai_clip(args.device)),
        ("open_clip ViT-B/32 (fallback #1)", lambda: try_open_clip(args.device)),
        ("HF CLIP (fallback #2)", lambda: try_hf_clip(args.device)),
        ("ViT-B/16 proxy (fallback #3)", lambda: try_vit(args.device)),
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
