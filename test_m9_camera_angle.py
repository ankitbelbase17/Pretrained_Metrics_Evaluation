from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

import torch

from metric_test_common import add_common_args, run_metric_on_all_datasets


def _probe_m9(device: str, batches: List[Dict[str, torch.Tensor]], _args) -> Tuple[Dict, str, float]:
    from pretrained_metrics.metrics.m9_camera_angle import CameraAngleMetrics

    obj = CameraAngleMetrics(device=device)
    backend = getattr(getattr(obj, "_backend", None), "_backend", "unknown")
    t0 = time.time()
    for batch in batches:
        obj.update(batch["person"])
    result = obj.compute()
    return result, backend, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="M9 Camera Angle metric test across all configured datasets")
    add_common_args(parser)
    args = parser.parse_args()
    return run_metric_on_all_datasets(
        args=args,
        metric_title="M9 Camera Angle",
        probe_fn=_probe_m9,
        paper_score_key="camera_diversity_score",
        paper_score_label="Camera Angle Diversity Score",
        set_home_for_hmr2=True,
    )


if __name__ == "__main__":
    sys.exit(main())
