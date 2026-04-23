"""
test.py
=======
Compute smoke-test for pretrained metrics (M1-M9) and EDA plots using CurvTON-hard.

What this script does:
1. Builds a dataloader from CurvTON hard split (full set by default).
2. Runs real metric computation (update + compute) for M1-M9.
3. Extracts EDA feature tensors from the same mini-batches.
4. Runs EDA plotting functions (P1-P10) and reports exact errors.
5. Prints summarized tables compatible with run.py parsing.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from pretrained_metrics.cache_setup import configure_model_caches, DEFAULT_MODEL_BASE

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_metrics"))
sys.path.insert(0, str(ROOT / "EDA"))


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _cyan(s: str) -> str:
    return f"\033[96m{s}\033[0m"


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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


@dataclass
class EDASummary:
    key: str
    plot: str
    status: str
    required_metrics: List[str]
    selected_mode: str
    fallback_used: bool
    notes: List[str]
    error: Optional[str] = None


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


def _compact_values(values: Optional[Dict[str, object]]) -> str:
    if not values:
        return "NA"
    parts: List[str] = []
    for k in sorted(values.keys()):
        parts.append(f"{k}={_fmt_value(values[k])}")
    return ", ".join(parts)


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


def _collect_curvton_easy_batches(args) -> Tuple[List[Dict[str, torch.Tensor]], int, int]:
    from pretrained_metrics.dataloader import get_dataloader

    def _to_bchw(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"[{name}] Expected 4-D tensor, got shape {tuple(x.shape)}")
        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[1] != 3:
            raise RuntimeError(f"[{name}] Expected channel dimension C=3, got shape {tuple(x.shape)}")
        return x

    loader = get_dataloader(
        dataset_name=args.dataset_name,
        root=args.curvton_easy_root,
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
        raise RuntimeError("No batches were loaded from CurvTON hard dataloader.")
    return batches, len(batches), n_images


def _run_updates(metric_obj, batches: List[Dict[str, torch.Tensor]], use_cloth: bool = False):
    for b in batches:
        x = b["cloth"] if use_cloth else b["person"]
        metric_obj.update(x)
    return metric_obj.compute()


def _probe_m1(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    chains = {"pose_extractor": [ChainModel("KeypointRCNN", "keypointrcnn")]}
    try:
        from pretrained_metrics.metrics.m1_pose import PoseMetrics

        obj = PoseMetrics(device=device)
        result = _run_updates(obj, batches)
        backend = getattr(obj.extractor, "_backend", None)
        return MetricAudit(
            key="m1",
            metric="M1 Pose",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "keypointrcnn"),
            chains=chains,
            selected_by_chain={"pose_extractor": backend},
            notes=[f"compute_keys={sorted(result.keys())}", f"backend={backend}"],
            computed_values=result,
        )
    except Exception as e:
        return MetricAudit(
            key="m1",
            metric="M1 Pose",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"pose_extractor": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


def _probe_m2(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    seg_chain = [
        ChainModel("Mask2Former", "mask2former"),
        ChainModel("SegFormer-B2 clothes", "segformer"),
        ChainModel("DeepLabV3-ResNet101", "deeplabv3_skin"),
    ]
    det_chain = [
        ChainModel("DETR", "detr"),
        ChainModel("YOLOv8", "yolo"),
    ]
    chains = {"segmentation": seg_chain, "object_detector": det_chain}
    try:
        from pretrained_metrics.metrics.m2_occlusion import OcclusionMetrics

        obj = OcclusionMetrics(device=device)
        result = _run_updates(obj, batches)
        seg_backend = getattr(obj._seg, "_backend", None)
        det = getattr(obj._seg, "_object_detector", None)
        det_backend = det.get("type") if isinstance(det, dict) else None
        det_required = seg_backend in {"segformer", "deeplabv3_skin"}
        notes = [f"compute_keys={sorted(result.keys())}", f"segmentation_backend={seg_backend}"]
        if det_required:
            notes.append(f"object_detector_backend={det_backend}")
        else:
            notes.append("object_detector_backend=N/A")
        return MetricAudit(
            key="m2",
            metric="M2 Occlusion",
            status="LOADED",
            selected_backend=seg_backend,
            fallback_used=(seg_backend != "mask2former"),
            chains=chains,
            selected_by_chain={
                "segmentation": seg_backend,
                "object_detector": det_backend if det_required else "N/A",
            },
            notes=notes,
            computed_values=result,
        )
    except Exception as e:
        return MetricAudit(
            key="m2",
            metric="M2 Occlusion",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"segmentation": None, "object_detector": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


def _probe_m3(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    chains = {
        "person_segmenter": [ChainModel("DeepLabV3-ResNet101", "deeplabv3")],
        "object_detector": [ChainModel("DETR", "detr")],
    }
    try:
        from pretrained_metrics.metrics.m3_background import BackgroundMetrics

        obj = BackgroundMetrics(device=device)
        result = _run_updates(obj, batches)
        seg_loaded = getattr(obj._segmenter, "_model", None) is not None
        det_backend = getattr(obj._detector, "_backend", None)
        return MetricAudit(
            key="m3",
            metric="M3 Background",
            status="LOADED",
            selected_backend="deeplabv3+detr",
            fallback_used=False,
            chains=chains,
            selected_by_chain={
                "person_segmenter": "deeplabv3" if seg_loaded else None,
                "object_detector": det_backend,
            },
            notes=[f"compute_keys={sorted(result.keys())}"],
            computed_values=result,
        )
    except Exception as e:
        return MetricAudit(
            key="m3",
            metric="M3 Background",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"person_segmenter": None, "object_detector": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


def _probe_m4(_: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    chains = {"signal_processing": []}
    try:
        from pretrained_metrics.metrics.m4_illumination import IlluminationMetrics

        obj = IlluminationMetrics()
        result = _run_updates(obj, batches)
        return MetricAudit(
            key="m4",
            metric="M4 Illumination",
            status="LOADED",
            selected_backend="signal_processing",
            fallback_used=False,
            chains=chains,
            selected_by_chain={"signal_processing": "signal_processing"},
            notes=[f"compute_keys={sorted(result.keys())}"],
            computed_values=result,
        )
    except Exception as e:
        return MetricAudit(
            key="m4",
            metric="M4 Illumination",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"signal_processing": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )

def _probe_m5(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    chains = {"shape_extractor": [ChainModel("HMR2.0", "hmr2"), ChainModel("ViT-B/16 proxy", "vit_proxy")]}
    try:
        from pretrained_metrics.metrics.m5_body_shape import BodyShapeMetrics

        obj = BodyShapeMetrics(device=device)
        result = _run_updates(obj, batches)
        backend = getattr(obj._extractor, "_backend", None)
        return MetricAudit(
            key="m5",
            metric="M5 Body Shape",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "hmr2"),
            chains=chains,
            selected_by_chain={"shape_extractor": backend},
            notes=[f"compute_keys={sorted(result.keys())}", f"backend={backend}"],
            computed_values=result,
        )
    except Exception as e:
        return MetricAudit(
            key="m5",
            metric="M5 Body Shape",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"shape_extractor": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


def _probe_m6(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    chains = {"face_embedder": [ChainModel("InsightFace ArcFace", "arcface"), ChainModel("open_clip ViT-B/32", "open_clip")]}
    try:
        from pretrained_metrics.metrics.m6_appearance import AppearanceMetrics

        obj = AppearanceMetrics(device=device)
        result = _run_updates(obj, batches)
        backend = getattr(obj._embedder, "_backend", None)
        return MetricAudit(
            key="m6",
            metric="M6 Appearance",
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
            metric="M6 Appearance",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"face_embedder": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


def _probe_m7(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    chains = {"garment_encoder": [ChainModel("open_clip ViT-B/32", "open_clip"), ChainModel("ViT-B/16 proxy", "vit")]}
    try:
        from pretrained_metrics.metrics.m7_garment_texture import GarmentTextureMetrics

        obj = GarmentTextureMetrics(device=device)
        _ = _run_updates(obj, batches, use_cloth=True)
        backend = getattr(obj._encoder, "_backend", None)
        result = obj.compute()
        return MetricAudit(
            key="m7",
            metric="M7 Garment Texture",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "open_clip"),
            chains=chains,
            selected_by_chain={"garment_encoder": backend},
            notes=[f"compute_keys={sorted(result.keys())}", f"backend={backend}"],
            computed_values=result,
        )
    except Exception as e:
        return MetricAudit(
            key="m7",
            metric="M7 Garment Texture",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"garment_encoder": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


def _probe_m8(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    chains = {
        "vae_encoder": [
            ChainModel("stabilityai/sd-vae-ft-mse", "sd_vae_mse"),
            ChainModel("CompVis/stable-diffusion-v1-4", "sd_v14_vae"),
            ChainModel("runwayml/stable-diffusion-v1-5", "sd_v15_vae"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m8_vae_latent import VAELatentMetric

        obj = VAELatentMetric(device=device)
        result = _run_updates(obj, batches)
        backend = getattr(obj._encoder, "backend_name", None)
        if backend is not None and not isinstance(backend, str):
            backend = str(backend)
        return MetricAudit(
            key="m8",
            metric="M8 VAE Latent",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "sd_vae_mse"),
            chains=chains,
            selected_by_chain={"vae_encoder": backend},
            notes=[f"compute_keys={sorted(result.keys())}", f"backend={backend}"],
            computed_values=result,
        )
    except Exception as e:
        return MetricAudit(
            key="m8",
            metric="M8 VAE Latent",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"vae_encoder": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


def _probe_m9(device: str, batches: List[Dict[str, torch.Tensor]]) -> MetricAudit:
    chains = {
        "camera_backend": [
            ChainModel("HMR2.0", "hmr2"),
            ChainModel("ViTPose", "vitpose"),
            ChainModel("KeypointRCNN", "keypointrcnn"),
            ChainModel("DINOv2", "dino"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m9_camera_angle import CameraAngleMetrics

        obj = CameraAngleMetrics(device=device)
        result = _run_updates(obj, batches)
        backend = getattr(obj._backend, "_backend", None)
        return MetricAudit(
            key="m9",
            metric="M9 Camera Angle",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "hmr2"),
            chains=chains,
            selected_by_chain={"camera_backend": backend},
            notes=[f"compute_keys={sorted(result.keys())}", f"backend={backend}"],
        )
    except Exception as e:
        return MetricAudit(
            key="m9",
            metric="M9 Camera Angle",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"camera_backend": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
        )


def _probe_vlm(device: str) -> MetricAudit:
    chains = {
        "vlm_backend": [
            ChainModel("Qwen3-VL-32B-Instruct", "qwen3vl"),
            ChainModel("Qwen2-VL-7B-Instruct", "qwen2vl"),
            ChainModel("Stub (0.5)", "stub"),
        ]
    }
    try:
        from metrics.vlm_score import VLMScoreMetric

        obj = VLMScoreMetric(device=device)
        backend = getattr(obj, "_backend", None)
        return MetricAudit(
            key="vlm",
            metric="VLM Plausibility",
            status="LOADED" if backend is not None else "NOT LOADED",
            selected_backend=backend,
            fallback_used=(backend != "qwen3vl") if backend is not None else None,
            chains=chains,
            selected_by_chain={"vlm_backend": backend},
            notes=["load-only check (not part of M1-M9 dataloader compute)"],
            computed_values=None,
        )
    except Exception as e:
        return MetricAudit(
            key="vlm",
            metric="VLM Plausibility",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain={"vlm_backend": None},
            notes=[],
            error=f"{type(e).__name__}: {e}",
            computed_values=None,
        )


def _extract_eda_features_from_batches(device: str, batches: List[Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
    from pretrained_metrics.metrics.m1_pose import _KeypointExtractor, _normalise_pose, _joint_angle, TRIPLET_IDX
    from pretrained_metrics.metrics.m2_occlusion import _SegBackend
    from pretrained_metrics.metrics.m3_background import _PersonSegmenter, _texture_entropy, _ObjectDetector
    from pretrained_metrics.metrics.m4_illumination import _rgb_to_lab_l, _sobel_gradient_variance
    from pretrained_metrics.metrics.m5_body_shape import _ShapeExtractor
    from pretrained_metrics.metrics.m6_appearance import _FaceEmbedder
    from pretrained_metrics.metrics.m7_garment_texture import _GarmentEncoder
    from pretrained_metrics.metrics.m8_vae_latent import _VAEEncoder
    from pretrained_metrics.metrics.m9_camera_angle import _CameraAngleBackend

    kp_ext = _KeypointExtractor(device)
    seg = _SegBackend(device)
    per_seg = _PersonSegmenter(device)
    obj_det = _ObjectDetector(device)
    shape_ex = _ShapeExtractor(device)
    face_ex = _FaceEmbedder(device)
    garment_ex = _GarmentEncoder(device)
    vae_ex = _VAEEncoder(device)
    camera_ex = None
    try:
        camera_ex = _CameraAngleBackend(device)
    except Exception:
        camera_ex = None

    pose_vecs: List[np.ndarray] = []
    angles: List[np.ndarray] = []
    occ_ratios: List[float] = []
    occ_maps: List[np.ndarray] = []
    bg_entropy: List[float] = []
    bg_obj_count: List[int] = []
    lum_mean: List[float] = []
    lum_grad_var: List[float] = []
    lum_maps: List[np.ndarray] = []
    betas: List[np.ndarray] = []
    face_embs: List[np.ndarray] = []
    garment_embs: List[np.ndarray] = []
    vae_embs: List[np.ndarray] = []
    azimuths: List[float] = []
    elevations: List[float] = []
    camera_confidence: List[float] = []

    mask_ds = (64, 48)

    def _to_bchw(x: torch.Tensor, name: str) -> torch.Tensor:
        """Ensure image tensor is B,C,H,W with C=3."""
        if x.ndim != 4:
            raise RuntimeError(f"[{name}] Expected 4-D tensor, got shape {tuple(x.shape)}")
        # If input is B,H,W,C convert to B,C,H,W.
        if x.shape[1] != 3 and x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.shape[1] != 3:
            raise RuntimeError(f"[{name}] Expected channel dimension C=3, got shape {tuple(x.shape)}")
        return x

    for b in batches:
        person = _to_bchw(b["person"].float(), "person")
        cloth = _to_bchw(b["cloth"].float(), "cloth")
        B = person.shape[0]

        kps_raw = kp_ext(person)
        kps_norm, valid = _normalise_pose(kps_raw)
        for i in range(B):
            if valid[i]:
                pn = kps_norm[i]
                pose_vecs.append(pn.flatten().astype(np.float32))
                ang = [_joint_angle(pn[ia], pn[ib], pn[ic]) for ia, ib, ic in TRIPLET_IDX]
                angles.append(np.array([a if not np.isnan(a) else 0.0 for a in ang], dtype=np.float32))
            else:
                pose_vecs.append(np.zeros(34, dtype=np.float32))
                angles.append(np.zeros(len(TRIPLET_IDX), dtype=np.float32))

        seg_masks = seg.segment(person)
        G = seg_masks["garment"].float()
        occluder = ((seg_masks["arms"].float() + seg_masks["hair"].float() + seg_masks["other"].float()) > 0).float()
        overlap = G * occluder
        for i in range(B):
            g_area = G[i].sum().item()
            ratio = overlap[i].sum().item() / max(g_area, 1.0)
            occ_ratios.append(float(min(ratio, 1.0)))
            full_map = overlap[i].unsqueeze(0).unsqueeze(0)
            ds_map = F.interpolate(full_map, size=mask_ds, mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            occ_maps.append(ds_map.astype(np.float32))

        person_masks = per_seg(person)
        obj_counts = obj_det.count_objects(person, person_masks)
        for i in range(B):
            ent = _texture_entropy(person[i], person_masks[i])
            bg_entropy.append(float(ent) if not np.isnan(ent) else 0.0)
            bg_obj_count.append(int(obj_counts[i]))

        mean_L, L_maps = _rgb_to_lab_l(person.cpu())
        for i in range(B):
            lum_mean.append(float(mean_L[i]))
            lum_grad_var.append(float(_sobel_gradient_variance(L_maps[i])))
            l_map = torch.from_numpy(L_maps[i]).float().unsqueeze(0).unsqueeze(0)
            ds_l = F.interpolate(l_map, size=mask_ds, mode="bilinear", align_corners=False).squeeze().cpu().numpy()
            lum_maps.append(ds_l.astype(np.float32))

        for bi in shape_ex(person):
            betas.append(bi.astype(np.float32))
        for fi in face_ex(person):
            face_embs.append(fi.astype(np.float32))
        for gi in garment_ex(cloth):
            garment_embs.append(gi.astype(np.float32))
        for vi in vae_ex(person):
            vae_embs.append(vi.astype(np.float32))

        if camera_ex is not None:
            try:
                cam = camera_ex.estimate_angles(person)
                az = cam["azimuth"].detach().cpu().numpy().astype(np.float32)
                el = cam["elevation"].detach().cpu().numpy().astype(np.float32)
                cf = cam["confidence"].detach().cpu().numpy().astype(np.float32)
                for i in range(B):
                    azimuths.append(float(az[i]))
                    elevations.append(float(el[i]))
                    camera_confidence.append(float(cf[i]))
            except Exception:
                camera_ex = None

    data = {
        "pose_vecs": np.stack(pose_vecs),
        "angles": np.stack(angles),
        "occ_ratios": np.array(occ_ratios, dtype=np.float32),
        "occ_maps": np.stack(occ_maps),
        "bg_entropy": np.array(bg_entropy, dtype=np.float32),
        "bg_obj_count": np.array(bg_obj_count, dtype=np.int32),
        "lum_mean": np.array(lum_mean, dtype=np.float32),
        "lum_grad_var": np.array(lum_grad_var, dtype=np.float32),
        "lum_maps": np.stack(lum_maps),
        "betas": np.stack(betas),
        "face_embs": np.stack(face_embs),
        "garment_embs": np.stack(garment_embs),
        "vae_embs": np.stack(vae_embs),
    }
    if azimuths:
        data["azimuths"] = np.array(azimuths, dtype=np.float32)
        data["elevations"] = np.array(elevations, dtype=np.float32)
        data["camera_confidence"] = np.array(camera_confidence, dtype=np.float32)
    return data

def _choose_reducer_umap_tsne_pca() -> Tuple[str, bool]:
    if _has_module("umap"):
        return "umap", False
    if _has_module("sklearn"):
        return "sklearn_tsne", True
    return "numpy_pca", True


def _choose_pose_embedder_mode() -> Tuple[str, bool]:
    if _has_module("umap"):
        return "umap", False
    if _has_module("sklearn"):
        return "sklearn_pca", True
    return "numpy_pca", True


def _run_eda_compute_tests(
    device: str,
    batches: List[Dict[str, torch.Tensor]],
    out_dir: str,
) -> List[EDASummary]:
    from EDA.plots.p1_pose_eda import plot_pose_umap, plot_joint_angle_distributions
    from EDA.plots.p2_occlusion_eda import plot_occlusion_histogram, plot_occlusion_heatmap
    from EDA.plots.p3_background_eda import plot_bg_entropy_histogram, plot_entropy_vs_objects
    from EDA.plots.p4_illumination_eda import plot_luminance_spectrum, plot_illumination_pca
    from EDA.plots.p5_body_shape_eda import plot_shape_pca, plot_shape_coefficient_histograms
    from EDA.plots.p6_appearance_eda import plot_face_umap, plot_pairwise_distance_distribution
    from EDA.plots.p7_garment_eda import plot_garment_umap, plot_eigenvalue_spectrum
    from EDA.plots.p8_meta_correlation import plot_correlation_matrix, _build_feature_matrix
    from EDA.plots.p9_vae_eda import plot_vae_pca, plot_vae_pca_combined, plot_vae_explained_variance, plot_vae_tsne
    from EDA.plots.p10_camera_angle_eda import run_camera_angle_eda

    features = _extract_eda_features_from_batches(device, batches)
    ds_name = "curvton_easy_smoke"
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    p1_mode, p1_fallback = _choose_pose_embedder_mode()
    p6_mode, p6_fallback = _choose_reducer_umap_tsne_pca()
    p7_mode, p7_fallback = _choose_reducer_umap_tsne_pca()
    has_sklearn = _has_module("sklearn")
    p10_mode = "explicit_camera_angles_from_M9" if "azimuths" in features else "pose_proxy_from_M1"
    p10_fallback = "azimuths" not in features

    checks: List[Tuple[str, str, List[str], str, bool, callable]] = [
        ("p1", "P1 Pose EDA", ["M1"], p1_mode, p1_fallback,
         lambda: (
             plot_pose_umap({ds_name: features["pose_vecs"]}, out_dir=str(out_root / "pose")),
             plot_joint_angle_distributions({ds_name: features["angles"]}, out_dir=str(out_root / "pose")),
         )),
        ("p2", "P2 Occlusion EDA", ["M2"], "feature_only", False,
         lambda: (
             plot_occlusion_histogram({ds_name: features["occ_ratios"]}, out_dir=str(out_root / "occlusion")),
             plot_occlusion_heatmap({ds_name: features["occ_maps"]}, out_dir=str(out_root / "occlusion")),
         )),
        ("p3", "P3 Background EDA", ["M3"], "feature_only", False,
         lambda: (
             plot_bg_entropy_histogram({ds_name: features["bg_entropy"]}, out_dir=str(out_root / "background")),
             plot_entropy_vs_objects({ds_name: features["bg_entropy"]}, {ds_name: features["bg_obj_count"].astype(float)}, out_dir=str(out_root / "background")),
         )),
        ("p4", "P4 Illumination EDA", ["M4"], "sklearn_pca+standardscaler" if has_sklearn else "numpy_standardize+pca", not has_sklearn,
         lambda: (
             plot_luminance_spectrum({ds_name: features["lum_mean"]}, {ds_name: features["lum_grad_var"]}, out_dir=str(out_root / "illumination")),
             plot_illumination_pca({ds_name: features["lum_maps"]}, out_dir=str(out_root / "illumination")),
         )),
        ("p5", "P5 Body Shape EDA", ["M5"], "sklearn_pca+standardscaler" if has_sklearn else "numpy_standardize+pca", not has_sklearn,
         lambda: (
             plot_shape_pca({ds_name: features["betas"]}, out_dir=str(out_root / "body_shape")),
             plot_shape_coefficient_histograms({ds_name: features["betas"]}, out_dir=str(out_root / "body_shape")),
         )),
        ("p6", "P6 Appearance EDA", ["M6"], p6_mode, p6_fallback,
         lambda: (
             plot_face_umap({ds_name: features["face_embs"]}, out_dir=str(out_root / "appearance")),
             plot_pairwise_distance_distribution({ds_name: features["face_embs"]}, out_dir=str(out_root / "appearance")),
         )),
        ("p7", "P7 Garment EDA", ["M7"], p7_mode, p7_fallback,
         lambda: (
             plot_garment_umap({ds_name: features["garment_embs"]}, out_dir=str(out_root / "garment")),
             plot_eigenvalue_spectrum({ds_name: features["garment_embs"]}, out_dir=str(out_root / "garment")),
         )),
        ("p8", "P8 Meta Correlation EDA", ["M1", "M2", "M3", "M4", "M5", "M6", "M7"], "feature_only", False,
         lambda: (
             plot_correlation_matrix({ds_name: _build_feature_matrix(features)}, out_dir=str(out_root / "meta")),
         )),
        ("p9", "P9 VAE EDA", ["M8"], "sklearn_required", False,
         lambda: (
             plot_vae_pca({ds_name: features["vae_embs"]}, out_dir=str(out_root / "vae")),
             plot_vae_pca_combined({ds_name: features["vae_embs"]}, out_dir=str(out_root / "vae")),
             plot_vae_explained_variance({ds_name: features["vae_embs"]}, out_dir=str(out_root / "vae")),
             plot_vae_tsne({ds_name: features["vae_embs"]}, out_dir=str(out_root / "vae")),
         )),
        ("p10", "P10 Camera Angle EDA", ["M9 (preferred)", "M1 (fallback pose proxy)"], p10_mode, p10_fallback,
         lambda: run_camera_angle_eda(features, dataset_name=ds_name, output_dir=str(out_root / "camera_angle"))),
    ]

    out: List[EDASummary] = []
    for key, name, req, mode, fallback, fn in checks:
        try:
            fn()
            out.append(
                EDASummary(
                    key=key,
                    plot=name,
                    status="READY",
                    required_metrics=req,
                    selected_mode=mode,
                    fallback_used=fallback,
                    notes=[f"output_dir={out_root.resolve()}"],
                )
            )
        except Exception as e:
            out.append(
                EDASummary(
                    key=key,
                    plot=name,
                    status="FAILED",
                    required_metrics=req,
                    selected_mode=mode,
                    fallback_used=fallback,
                    notes=[],
                    error=f"{type(e).__name__}: {e}",
                )
            )
    return out


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
        if selected_token == "N/A":
            print(f"      {chain_name}: N/A")
            continue
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


def _print_eda_summary(e: EDASummary):
    if e.status == "READY":
        st = _green(e.status)
    elif e.status == "FAILED":
        st = _red(e.status)
    else:
        st = _yellow(e.status)
    print(f"  {st:<20} [{e.key.upper()}] {e.plot}")
    print(f"      required_metrics : {', '.join(e.required_metrics)}")
    print(f"      selected_mode    : {e.selected_mode}")
    print(f"      fallback_used    : {e.fallback_used}")
    print(f"      plot_value       : {'GENERATED' if e.status == 'READY' else 'NA'}")
    for n in e.notes:
        print(f"      note             : {n}")
    if e.error:
        print(f"      error            : {_red(e.error)}")

def _fmt_cell(v: object) -> str:
    return str(v) if v is not None else "-"


def _models_used_from_audit(a: MetricAudit) -> str:
    if a.status != "LOADED":
        return "NA"
    models: List[str] = []
    for chain_name, selected_token in a.selected_by_chain.items():
        if selected_token in (None, "N/A"):
            continue
        chain = a.chains.get(chain_name, [])
        label = next((c.label for c in chain if c.token == selected_token), selected_token)
        models.append(f"{chain_name}:{label}")
    return "; ".join(models) if models else "NA"


def _remarks_for_metric(a: MetricAudit) -> str:
    if a.status == "LOADED":
        return "Success"
    return a.error or "Failed"


def _remarks_for_eda(e: EDASummary) -> str:
    if e.status == "READY":
        return "Success"
    return e.error or "Failed"


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


def run_checks(args) -> int:
    cache_info = configure_model_caches(args.download_base, set_home_for_hmr2=True)
    print("\n" + "=" * 90)
    print("  Pretrained Metrics + EDA Compute Smoke-Test (CurvTON hard)")
    print(f"  device={args.device} | root={args.curvton_easy_root}")
    print(f"  batch_size={args.batch_size} | max_batches={args.max_batches} | split={args.split}")
    print(f"  download_base={cache_info['base_path']}")
    print("=" * 90)

    try:
        batches, n_batches, n_images = _collect_curvton_easy_batches(args)
        print(f"  Loaded smoke batches: {n_batches} | images: {n_images}")
    except Exception as e:
        print(f"  {_red('FAILED')} dataloader setup: {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

    metric_results: Dict[str, MetricAudit] = {}
    failed_count = 0
    skipped_count = 0

    probes = [
        ("m1", lambda: _probe_m1(args.device, batches)),
        ("m2", lambda: _probe_m2(args.device, batches)),
        ("m3", lambda: _probe_m3(args.device, batches)),
        ("m4", lambda: _probe_m4(args.device, batches)),
        ("m5", lambda: _probe_m5(args.device, batches)),
        ("m6", lambda: _probe_m6(args.device, batches)),
        ("m7", lambda: _probe_m7(args.device, batches)),
        ("m8", lambda: _probe_m8(args.device, batches)),
        ("m9", lambda: _probe_m9(args.device, batches)),
        ("vlm", lambda: _probe_vlm(args.device)),
    ]

    print("\n" + _cyan("METRIC COMPUTE AUDIT"))
    print("-" * 90)
    for key, fn in probes:
        if any(s.lower() in key.lower() for s in args.skip):
            print(f"  {_yellow('SKIP'):<20} [{key.upper()}] metric check skipped")
            skipped_count += 1
            continue
        t0 = time.time()
        audit = None
        try:
            audit = fn()
        except Exception as e:
            failed_count += 1
            print(f"  {_red('NOT LOADED'):<20} [{key.upper()}] internal checker failure")
            print(f"      error            : {type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()
        dt = time.time() - t0
        if audit is not None:
            audit.elapsed_s = dt
            metric_results[audit.key] = audit
            _print_metric_audit(audit)
            if audit.status != "LOADED":
                failed_count += 1
        print(f"      elapsed          : {dt:.1f}s\n")

    print(_cyan("EDA COMPUTE AUDIT"))
    print("-" * 90)
    eda_rows: List[EDASummary] = []
    try:
        eda_rows = _run_eda_compute_tests(args.device, batches, args.eda_out_dir)
    except Exception as e:
        print(f"  {_red('FAILED'):<20} [EDA] feature extraction/plot setup failed")
        print(f"      error            : {type(e).__name__}: {e}")
        if args.verbose:
            traceback.print_exc()
        failed_count += 1
        eda_rows = [
            EDASummary(
                key="eda",
                plot="EDA feature extraction/plot setup",
                status="FAILED",
                required_metrics=["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"],
                selected_mode="setup",
                fallback_used=False,
                notes=[],
                error=f"{type(e).__name__}: {e}",
            )
        ]

    for row in eda_rows:
        if any(s.lower() in row.key.lower() for s in args.skip):
            print(f"  {_yellow('SKIP'):<20} [{row.key.upper()}] EDA check skipped")
            skipped_count += 1
            continue
        _print_eda_summary(row)
        if row.status != "READY":
            failed_count += 1
        print()

    print("=" * 90)
    print("  Final Summary")
    print("=" * 90)
    loaded_metrics = sum(1 for x in metric_results.values() if x.status == "LOADED")
    total_metrics = len(metric_results)
    print(f"  {_green('Loaded metrics')}     : {loaded_metrics}/{total_metrics}")
    print(f"  {_red('Failed checks')}      : {failed_count}")
    print(f"  {_yellow('Skipped checks')}    : {skipped_count}")
    print("=" * 90 + "\n")

    metric_rows: List[List[object]] = []
    for k in sorted(metric_results.keys()):
        a = metric_results[k]
        metric_rows.append([
            a.key.upper(),
            a.metric,
            a.status,
            a.selected_backend or "-",
            a.fallback_used if a.fallback_used is not None else "-",
            _compact_values(a.computed_values) if a.status == "LOADED" else "NA",
            f"{a.elapsed_s:.2f}s" if a.elapsed_s is not None else "NA",
        ])
    if metric_rows:
        _print_table(
            "  Summarized Table: Metrics",
            ["Key", "Metric", "Status", "Loaded Backend", "Fallback", "Values", "Time"],
            metric_rows,
        )
        print()

    metric_value_rows: List[List[object]] = []
    for k in sorted(metric_results.keys()):
        a = metric_results[k]
        if a.status == "LOADED" and a.computed_values:
            for mk in sorted(a.computed_values.keys()):
                metric_value_rows.append([a.key.upper(), mk, _fmt_value(a.computed_values[mk]), "OK"])
        else:
            metric_value_rows.append([a.key.upper(), "ALL", "NA", a.error or "Not loaded"])
    if metric_value_rows:
        _print_table(
            "  Detailed Metric Values (NA where errors occurred)",
            ["Metric", "Value Key", "Value", "Note"],
            metric_value_rows,
        )
        print()

    eda_rows_table: List[List[object]] = []
    for row in eda_rows:
        if any(s.lower() in row.key.lower() for s in args.skip):
            continue
        eda_rows_table.append([
            row.key.upper(),
            row.plot,
            row.status,
            row.selected_mode,
            row.fallback_used,
            "GENERATED" if row.status == "READY" else "NA",
        ])
    if eda_rows_table:
        _print_table(
            "  Summarized Table: EDA",
            ["Key", "Plot", "Status", "Selected Mode", "Fallback", "Value"],
            eda_rows_table,
        )
        print()

    final_metric_rows: List[List[object]] = []
    for k in sorted(metric_results.keys()):
        a = metric_results[k]
        final_metric_rows.append([
            a.key.upper(),
            a.metric,
            "YES" if a.status == "LOADED" else "NO",
            _compact_values(a.computed_values) if a.status == "LOADED" else "NA",
            _models_used_from_audit(a),
            _remarks_for_metric(a),
            f"{a.elapsed_s:.2f}s" if a.elapsed_s is not None else "NA",
        ])

    if final_metric_rows:
        _print_table(
            "  Final Audit Table: Metrics",
            ["Key", "Metric", "Computed", "Metric Values", "Pretrained Model Used", "Remarks", "Time"],
            final_metric_rows,
        )
        print()

    final_plot_rows: List[List[object]] = []
    for e in eda_rows:
        if any(s.lower() in e.key.lower() for s in args.skip):
            continue
        final_plot_rows.append([
            e.key.upper(),
            e.plot,
            "YES" if e.status == "READY" else "NO",
            "GENERATED" if e.status == "READY" else "NA",
            ", ".join(e.required_metrics),
            _remarks_for_eda(e),
        ])

    if final_plot_rows:
        _print_table(
            "  Final Audit Table: EDA Plots",
            ["Key", "Plot", "Computed", "Value", "Driven By", "Remarks"],
            final_plot_rows,
        )
        print()

    return failed_count


def _parse():
    p = argparse.ArgumentParser(
        description="Compute smoke-test for M1-M9 and EDA using CurvTON hard dataloader."
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip", nargs="*", default=[], help="Skip keys by substring, e.g. m8 p9 vlm")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--cache_dir", type=str, default="./eda_cache", help="Compatibility arg (unused in compute mode)")
    p.add_argument("--download_base", type=str, default=DEFAULT_MODEL_BASE)
    p.add_argument(
        "--curvton_easy_root",
        type=str,
        default="/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test/hard",
        help="Absolute path to CurvTON hard split root",
    )
    p.add_argument("--dataset_name", type=str, default="curvton", help="Dataset registry name")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--img_size", type=int, nargs=2, default=[512, 384], metavar=("H", "W"))
    p.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Number of batches for compute; use 0 or negative to process full split",
    )
    p.add_argument("--eda_out_dir", type=str, default="assets/plots/test_compute_audit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.batch_size < 16:
        print(f"[Config] Requested batch_size={args.batch_size} is below minimum; using 16.")
        args.batch_size = 16
    n_failed = run_checks(args)
    sys.exit(0 if n_failed == 0 else 1)
