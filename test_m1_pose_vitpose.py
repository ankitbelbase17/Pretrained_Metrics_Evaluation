from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch

from metric_test_common import add_common_args, run_metric_on_all_datasets


def _probe_m1_vitpose(device: str, batches: List[Dict[str, torch.Tensor]], _args) -> Tuple[Dict, str, float]:
    from pretrained_metrics.metrics.m1_pose import PoseMetrics

    obj = PoseMetrics(device=device)
    backend = getattr(obj.extractor, "_backend", "unknown")
    if backend not in {"mmpose_vitpose", "vitpose_hf_fallback"}:
        raise RuntimeError(
            f"Expected ViTPose backend, got '{backend}'. "
            "Install MMPose ViTPose or HF ViTPose dependencies."
        )

    t0 = time.time()
    for batch in batches:
        obj.update(batch["person"])
    result = obj.compute()
    return result, backend, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 Pose metric test (ViTPose-only) across configured datasets")
    add_common_args(parser)
    parser.set_defaults(continue_on_error=False, error_log="logs/test_m1_pose_vitpose_errors.log")
    args = parser.parse_args()

    return run_metric_on_all_datasets(
        args=args,
        metric_title="M1 Pose (ViTPose-only)",
        probe_fn=_probe_m1_vitpose,
        paper_score_key="pose_diversity",
        paper_score_label="Pose Diversity",
        set_home_for_hmr2=False,
    )


if __name__ == "__main__":
    sys.exit(main())
