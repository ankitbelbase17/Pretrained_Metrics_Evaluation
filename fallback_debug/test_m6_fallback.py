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


def try_arcface() -> str:
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in providers
    chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    ctx_id = 0 if use_cuda else -1
    app = FaceAnalysis(providers=chosen)
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return f"appearance_backend=arcface, provider={chosen[0]}"


def try_openai_clip(device: str) -> str:
    import clip as _oa_clip

    if not hasattr(_oa_clip, "load"):
        raise ImportError(
            "'clip' package is not OpenAI CLIP; expected pip package openai-clip"
        )
    model, _pre = _oa_clip.load("ViT-B/32", device=device)
    model.eval()
    return "appearance_backend=clip (openai)"


def try_open_clip(device: str) -> str:
    import open_clip

    model, _, _pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    return "appearance_backend=open_clip"


def main():
    args = parse_common_args("M6 fallback debug (ArcFace -> OpenAI CLIP -> open_clip)")
    print_env(args.device)

    attempts = [
        ("InsightFace ArcFace (primary)", lambda: try_arcface()),
        ("OpenAI CLIP ViT-B/32 (fallback #1)", lambda: try_openai_clip(args.device)),
        ("open_clip ViT-B/32 (fallback #2)", lambda: try_open_clip(args.device)),
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

