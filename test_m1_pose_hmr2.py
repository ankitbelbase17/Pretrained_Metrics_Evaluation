from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from metric_test_common import add_common_args, run_metric_on_all_datasets


def _find_2d_keypoints_tensor(obj) -> Optional[torch.Tensor]:
    """Recursively search HMR2 output dict for a 2D keypoints/joints tensor."""
    if torch.is_tensor(obj):
        if obj.ndim >= 3 and obj.shape[-1] >= 2:
            return obj
        return None

    if isinstance(obj, dict):
        # Prefer explicit 2D keypoint names first.
        priority_keys = [
            "pred_keypoints_2d",
            "pred_joints_2d",
            "keypoints_2d",
            "joints_2d",
            "smpl_joints2d",
        ]
        for k in priority_keys:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]

        for k, v in obj.items():
            lk = str(k).lower()
            if ("2d" in lk and ("keypoint" in lk or "joint" in lk)) and torch.is_tensor(v):
                return v

        for v in obj.values():
            t = _find_2d_keypoints_tensor(v)
            if t is not None:
                return t

    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _find_2d_keypoints_tensor(v)
            if t is not None:
                return t

    return None


class _HMR2KeypointExtractor:
    """Extract COCO-17-like 2D keypoints from HMR2 outputs."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend = "hmr2"
        self._model = None
        self._load()

    def _load(self):
        from pretrained_metrics.cache_setup import configure_model_caches, ensure_hmr2_smpl_model

        cache_info = configure_model_caches(set_home_for_hmr2=True)
        _ = ensure_hmr2_smpl_model(cache_info["base_path"])

        from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT
        import torch.serialization as _ts
        from omegaconf import DictConfig as _DictConfig, ListConfig as _ListConfig

        _ts.add_safe_globals([_DictConfig, _ListConfig])
        download_models()
        self._model, _ = load_hmr2(DEFAULT_CHECKPOINT)
        self._model = self._model.to(self.device).eval()

        print(f"[PoseMetric-HMR2] Using HMR2 backend (cache: {cache_info['fourdhumans_cache']}).")

    @torch.no_grad()
    def __call__(self, imgs: torch.Tensor) -> np.ndarray:
        if imgs.ndim != 4:
            raise RuntimeError(f"[PoseMetric-HMR2] Expected 4D tensor, got {tuple(imgs.shape)}")
        if imgs.shape[1] != 3 and imgs.shape[-1] == 3:
            imgs = imgs.permute(0, 3, 1, 2).contiguous()
        if imgs.shape[1] != 3:
            raise RuntimeError(f"[PoseMetric-HMR2] Expected C=3, got {tuple(imgs.shape)}")

        b = imgs.shape[0]
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (imgs.to(self.device) - mean) / std
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        out = self._model({"img": x})
        kps = _find_2d_keypoints_tensor(out)
        if kps is None:
            raise RuntimeError(
                "[PoseMetric-HMR2] Could not find 2D keypoints tensor in HMR2 output."
            )

        if kps.ndim == 4 and kps.shape[1] == 1:
            kps = kps[:, 0]
        if kps.ndim != 3:
            raise RuntimeError(f"[PoseMetric-HMR2] Unexpected keypoint tensor shape: {tuple(kps.shape)}")

        kps = kps[..., :2].detach().cpu().numpy().astype(np.float32)

        # Convert to fixed (B, 17, 2) expected by PoseMetrics.
        out_kps = np.zeros((b, 17, 2), dtype=np.float32)
        n = min(17, kps.shape[1])
        out_kps[:, :n, :] = kps[:, :n, :]
        return out_kps


def _probe_m1_hmr2(device: str, batches: List[Dict[str, torch.Tensor]], _args) -> Tuple[Dict, str, float]:
    from pretrained_metrics.metrics.m1_pose import PoseMetrics

    obj = PoseMetrics(device=device)
    # Force HMR2 extractor for this dedicated test.
    obj.extractor = _HMR2KeypointExtractor(device=device)
    backend = getattr(obj.extractor, "_backend", "unknown")

    t0 = time.time()
    for batch in batches:
        obj.update(batch["person"])
    result = obj.compute()
    return result, backend, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 Pose metric test (HMR2-backed keypoints) across configured datasets")
    add_common_args(parser)
    parser.set_defaults(continue_on_error=False, error_log="logs/test_m1_pose_hmr2_errors.log")
    args = parser.parse_args()

    return run_metric_on_all_datasets(
        args=args,
        metric_title="M1 Pose (HMR2-backed)",
        probe_fn=_probe_m1_hmr2,
        paper_score_key="pose_diversity",
        paper_score_label="Pose Diversity",
        set_home_for_hmr2=True,
    )


if __name__ == "__main__":
    sys.exit(main())
