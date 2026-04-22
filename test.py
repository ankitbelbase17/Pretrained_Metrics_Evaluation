"""
test.py
=======
Pretrained metrics + EDA dependency audit.

What this script does:
1. Checks which pretrained model backend each metric actually loaded.
2. Shows fallback usage (primary vs fallback backend).
3. Shows which candidate models in each fallback chain did not load.
4. Summarizes EDA plot requirements and whether they run full backend or fallback.

It does NOT run metric forward passes over datasets.

Run:
    python test.py
    python test.py --device cuda
    python test.py --skip m2 p9
    python test.py --verbose
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from pretrained_metrics.cache_setup import configure_model_caches, DEFAULT_MODEL_BASE

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pretrained_metrics"))


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


@dataclass
class EDASummary:
    key: str
    plot: str
    status: str
    required_metrics: List[str]
    selected_mode: str
    fallback_used: bool
    notes: List[str]


def _chain_statuses(
    chain: List[ChainModel],
    selected_token: Optional[str],
) -> List[Tuple[str, str]]:
    """
    Returns list of (model_label, status_text):
      - LOADED
      - NOT LOADED (failed before fallback)
      - NOT ATTEMPTED (fallback chain stopped earlier)
      - N/A (not required)
    """
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

    # Unknown selected token or total failure
    return [(c.label, "NOT LOADED") for c in chain]


# -----------------------------------------------------------------------------
# Metric probes
# -----------------------------------------------------------------------------


def _probe_m1(device: str) -> MetricAudit:
    chains = {
        "pose_extractor": [
            ChainModel("HRNet-W32 (timm)", "hrnet"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m1_pose import _KeypointExtractor

        obj = _KeypointExtractor(device)
        return MetricAudit(
            key="m1",
            metric="M1 Pose",
            status="LOADED",
            selected_backend=obj._backend,
            fallback_used=False,
            chains=chains,
            selected_by_chain={"pose_extractor": obj._backend},
            notes=[f"backend={obj._backend}"],
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
        )


def _probe_m2(device: str) -> MetricAudit:
    seg_chain = [
        ChainModel("Mask2Former (facebook/mask2former-swin-large-coco-panoptic)", "mask2former"),
        ChainModel("SegFormer-B2 clothes (mattmdjaga/segformer_b2_clothes)", "segformer"),
        ChainModel("DeepLabV3-ResNet101 (torchvision)", "deeplabv3_skin"),
    ]
    det_chain = [
        ChainModel("DETR (facebook/detr-resnet-50)", "detr"),
        ChainModel("YOLOv8 (ultralytics)", "yolo"),
    ]
    chains = {"segmentation": seg_chain, "object_detector": det_chain}
    try:
        from pretrained_metrics.metrics.m2_occlusion import _SegBackend

        obj = _SegBackend(device)
        seg_backend = getattr(obj, "_backend", None)
        det_backend = None
        det_required = seg_backend in {"segformer", "deeplabv3_skin"}
        if getattr(obj, "_object_detector", None):
            det_backend = obj._object_detector.get("type")

        notes = [f"segmentation_backend={seg_backend}"]
        if det_required:
            notes.append(f"object_detector_backend={det_backend}")
        else:
            notes.append("object_detector_backend=N/A (not required for mask2former)")

        # Fallback is based on segmentation backend choice
        fallback = seg_backend != "mask2former"
        selected_by_chain = {
            "segmentation": seg_backend,
            "object_detector": det_backend if det_required else "N/A",
        }
        return MetricAudit(
            key="m2",
            metric="M2 Occlusion",
            status="LOADED",
            selected_backend=seg_backend,
            fallback_used=fallback,
            chains=chains,
            selected_by_chain=selected_by_chain,
            notes=notes,
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
        )


def _probe_m3(device: str) -> MetricAudit:
    chains = {
        "person_segmenter": [
            ChainModel("DeepLabV3-ResNet101 (torchvision)", "deeplabv3"),
        ],
        "object_detector": [
            ChainModel("DETR (facebook/detr-resnet-50)", "detr"),
        ],
    }
    selected_by_chain: Dict[str, Optional[str]] = {"person_segmenter": None, "object_detector": None}
    notes: List[str] = []
    try:
        from pretrained_metrics.metrics.m3_background import _ObjectDetector, _PersonSegmenter

        seg = _PersonSegmenter(device)
        selected_by_chain["person_segmenter"] = "deeplabv3" if getattr(seg, "_model", None) is not None else None
        notes.append(f"person_segmenter_loaded={getattr(seg, '_model', None) is not None}")

        det = _ObjectDetector(device)
        selected_by_chain["object_detector"] = getattr(det, "_backend", None)
        notes.append(f"object_detector_backend={getattr(det, '_backend', None)}")

        return MetricAudit(
            key="m3",
            metric="M3 Background",
            status="LOADED",
            selected_backend="deeplabv3+detr",
            fallback_used=False,
            chains=chains,
            selected_by_chain=selected_by_chain,
            notes=notes,
        )
    except Exception as e:
        return MetricAudit(
            key="m3",
            metric="M3 Background",
            status="NOT LOADED",
            selected_backend=None,
            fallback_used=None,
            chains=chains,
            selected_by_chain=selected_by_chain,
            notes=notes,
            error=f"{type(e).__name__}: {e}",
        )


def _probe_m4(_: str) -> MetricAudit:
    # M4 has no pretrained model backend.
    chains = {"signal_processing": []}
    try:
        from pretrained_metrics.metrics.m4_illumination import IlluminationMetrics

        _ = IlluminationMetrics()
        return MetricAudit(
            key="m4",
            metric="M4 Illumination",
            status="LOADED",
            selected_backend="signal_processing",
            fallback_used=False,
            chains=chains,
            selected_by_chain={"signal_processing": "signal_processing"},
            notes=["No pretrained model required."],
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
        )


def _probe_m5(device: str) -> MetricAudit:
    chains = {
        "shape_extractor": [
            ChainModel("HMR2.0 (4D-Humans)", "hmr2"),
            ChainModel("ViT-B/16 proxy (timm)", "vit_proxy"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m5_body_shape import _ShapeExtractor

        obj = _ShapeExtractor(device)
        backend = getattr(obj, "_backend", None)
        return MetricAudit(
            key="m5",
            metric="M5 Body Shape",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "hmr2"),
            chains=chains,
            selected_by_chain={"shape_extractor": backend},
            notes=[f"backend={backend}"],
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
        )


def _probe_m6(device: str) -> MetricAudit:
    chains = {
        "face_embedder": [
            ChainModel("InsightFace ArcFace", "arcface"),
            ChainModel("open_clip ViT-B/32", "open_clip"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m6_appearance import _FaceEmbedder

        obj = _FaceEmbedder(device)
        backend = getattr(obj, "_backend", None)
        return MetricAudit(
            key="m6",
            metric="M6 Appearance",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "arcface"),
            chains=chains,
            selected_by_chain={"face_embedder": backend},
            notes=[f"backend={backend}"],
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
        )


def _probe_m7(device: str) -> MetricAudit:
    chains = {
        "garment_encoder": [
            ChainModel("open_clip ViT-B/32", "open_clip"),
            ChainModel("ViT-B/16 proxy (timm)", "vit"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m7_garment_texture import _GarmentEncoder

        obj = _GarmentEncoder(device)
        backend = getattr(obj, "_backend", None)
        return MetricAudit(
            key="m7",
            metric="M7 Garment Texture",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "open_clip"),
            chains=chains,
            selected_by_chain={"garment_encoder": backend},
            notes=[f"backend={backend}"],
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
        )


def _probe_m8(device: str) -> MetricAudit:
    chains = {
        "vae_encoder": [
            ChainModel("stabilityai/sd-vae-ft-mse", "sd_vae_mse"),
            ChainModel("CompVis/stable-diffusion-v1-4 (VAE)", "sd_v14_vae"),
            ChainModel("runwayml/stable-diffusion-v1-5 (VAE)", "sd_v15_vae"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m8_vae_latent import _VAEEncoder

        obj = _VAEEncoder(device)
        backend = getattr(obj, "_backend", None)
        return MetricAudit(
            key="m8",
            metric="M8 VAE Latent",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "sd_vae_mse"),
            chains=chains,
            selected_by_chain={"vae_encoder": backend},
            notes=[f"backend={backend}"],
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
        )


def _probe_m9(device: str) -> MetricAudit:
    chains = {
        "camera_backend": [
            ChainModel("HMR2.0 (4D-Humans)", "hmr2"),
            ChainModel("ViTPose (usyd-community/vitpose-base-simple)", "vitpose"),
            ChainModel("KeypointRCNN (torchvision)", "keypointrcnn"),
            ChainModel("DINOv2 (facebook/dinov2-base)", "dino"),
        ]
    }
    try:
        from pretrained_metrics.metrics.m9_camera_angle import _CameraAngleBackend

        obj = _CameraAngleBackend(device)
        backend = getattr(obj, "_backend", None)
        return MetricAudit(
            key="m9",
            metric="M9 Camera Angle",
            status="LOADED",
            selected_backend=backend,
            fallback_used=(backend != "hmr2"),
            chains=chains,
            selected_by_chain={"camera_backend": backend},
            notes=[f"backend={backend}"],
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
            ChainModel("Stub (neutral score=0.5)", "stub"),
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
            fallback_used=(backend != "qwen3vl"),
            chains=chains,
            selected_by_chain={"vlm_backend": backend},
            notes=[f"backend={backend}"],
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
        )


METRIC_PROBES: List[Callable[[str], MetricAudit]] = [
    _probe_m1,
    _probe_m2,
    _probe_m3,
    _probe_m4,
    _probe_m5,
    _probe_m6,
    _probe_m7,
    _probe_m8,
    _probe_m9,
    _probe_vlm,
]


# -----------------------------------------------------------------------------
# EDA summary (requirements + fallback logic)
# -----------------------------------------------------------------------------


def _choose_reducer_umap_tsne_pca() -> Tuple[str, bool]:
    """
    For P6/P7:
      umap -> sklearn TSNE -> numpy PCA fallback.
    """
    if _has_module("umap"):
        return "umap", False
    if _has_module("sklearn"):
        return "sklearn_tsne", True
    return "numpy_pca", True


def _choose_pose_embedder_mode() -> Tuple[str, bool]:
    """
    P1 in run_eda defaults to use_tsne=False:
      umap -> sklearn PCA -> numpy PCA fallback.
    """
    if _has_module("umap"):
        return "umap", False
    if _has_module("sklearn"):
        return "sklearn_pca", True
    return "numpy_pca", True


def _eda_summaries(metric_results: Dict[str, MetricAudit]) -> List[EDASummary]:
    has_sklearn = _has_module("sklearn")

    p1_mode, p1_fallback = _choose_pose_embedder_mode()
    p6_mode, p6_fallback = _choose_reducer_umap_tsne_pca()
    p7_mode, p7_fallback = _choose_reducer_umap_tsne_pca()

    m9_loaded = metric_results.get("m9") and metric_results["m9"].status == "LOADED"
    m1_loaded = metric_results.get("m1") and metric_results["m1"].status == "LOADED"
    if m9_loaded:
        p10_mode = "explicit_camera_angles_from_M9"
        p10_fallback = False
    elif m1_loaded:
        p10_mode = "pose_proxy_from_M1"
        p10_fallback = True
    else:
        p10_mode = "unavailable (needs M9 camera or M1 pose)"
        p10_fallback = True

    summaries = [
        EDASummary(
            key="p1",
            plot="P1 Pose EDA",
            status="READY",
            required_metrics=["M1"],
            selected_mode=p1_mode,
            fallback_used=p1_fallback,
            notes=["Dimensionality reduction backend selected for pose scatter."],
        ),
        EDASummary(
            key="p2",
            plot="P2 Occlusion EDA",
            status="READY",
            required_metrics=["M2"],
            selected_mode="feature_only",
            fallback_used=False,
            notes=["No extra pretrained model used in plotting stage."],
        ),
        EDASummary(
            key="p3",
            plot="P3 Background EDA",
            status="READY",
            required_metrics=["M3"],
            selected_mode="feature_only",
            fallback_used=False,
            notes=["No extra pretrained model used in plotting stage."],
        ),
        EDASummary(
            key="p4",
            plot="P4 Illumination EDA",
            status="READY",
            required_metrics=["M4"],
            selected_mode="sklearn_pca+standardscaler" if has_sklearn else "numpy_standardize+pca",
            fallback_used=not has_sklearn,
            notes=["Sklearn is optional here; numpy fallback exists."],
        ),
        EDASummary(
            key="p5",
            plot="P5 Body Shape EDA",
            status="READY",
            required_metrics=["M5"],
            selected_mode="sklearn_pca+standardscaler" if has_sklearn else "numpy_standardize+pca",
            fallback_used=not has_sklearn,
            notes=["Sklearn is optional here; numpy fallback exists."],
        ),
        EDASummary(
            key="p6",
            plot="P6 Appearance EDA",
            status="READY",
            required_metrics=["M6"],
            selected_mode=p6_mode,
            fallback_used=p6_fallback,
            notes=["UMAP preferred; TSNE/PCA fallback depending on installed packages."],
        ),
        EDASummary(
            key="p7",
            plot="P7 Garment EDA",
            status="READY",
            required_metrics=["M7"],
            selected_mode=p7_mode,
            fallback_used=p7_fallback,
            notes=["UMAP preferred; TSNE/PCA fallback depending on installed packages."],
        ),
        EDASummary(
            key="p8",
            plot="P8 Meta Correlation EDA",
            status="READY",
            required_metrics=["M1", "M2", "M3", "M4", "M5", "M6", "M7"],
            selected_mode="feature_only",
            fallback_used=False,
            notes=["Uses extracted arrays from P1-P7 features."],
        ),
        EDASummary(
            key="p9",
            plot="P9 VAE EDA",
            status="READY" if has_sklearn else "SKIPPED_IF_RUN",
            required_metrics=["M8"],
            selected_mode="sklearn_required" if has_sklearn else "unavailable_without_sklearn",
            fallback_used=not has_sklearn,
            notes=["No numpy fallback in p9_vae_eda.py; run_eda skips P9 when sklearn is missing."],
        ),
        EDASummary(
            key="p10",
            plot="P10 Camera Angle EDA",
            status="READY" if (m9_loaded or m1_loaded) else "NOT READY",
            required_metrics=["M9 (preferred)", "M1 (fallback pose proxy)"],
            selected_mode=p10_mode,
            fallback_used=p10_fallback,
            notes=["Uses camera angles if available, otherwise derives camera proxy from pose vectors."],
        ),
    ]
    return summaries


def _print_metric_audit(a: MetricAudit):
    header = f"[{a.key.upper()}] {a.metric}"
    if a.status == "LOADED":
        st = _green("LOADED")
    else:
        st = _red("NOT LOADED")
    print(f"  {st:<20} {header}")
    if a.selected_backend is not None:
        print(f"      selected_backend : {a.selected_backend}")
    if a.fallback_used is not None:
        print(f"      fallback_used    : {a.fallback_used}")
    for n in a.notes:
        print(f"      note             : {n}")

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
    elif e.status == "NOT READY":
        st = _red(e.status)
    else:
        st = _yellow(e.status)
    print(f"  {st:<20} [{e.key.upper()}] {e.plot}")
    print(f"      required_metrics : {', '.join(e.required_metrics)}")
    print(f"      selected_mode    : {e.selected_mode}")
    print(f"      fallback_used    : {e.fallback_used}")
    for n in e.notes:
        print(f"      note             : {n}")


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


def run_checks(device: str, skip: List[str], verbose: bool) -> int:
    cache_info = configure_model_caches(DEFAULT_MODEL_BASE, set_home_for_hmr2=True)
    print("\n" + "=" * 90)
    print("  Pretrained Metrics + EDA Model/Fallback Audit")
    print(f"  device={device}  |  skip={skip}")
    print(f"  download_base={cache_info['base_path']}")
    print("=" * 90)

    metric_results: Dict[str, MetricAudit] = {}
    failed_count = 0
    skipped_count = 0

    print("\n" + _cyan("METRIC LOAD AUDIT"))
    print("-" * 90)
    for probe in METRIC_PROBES:
        key = probe.__name__.replace("_probe_", "")
        if any(s.lower() in key.lower() for s in skip):
            print(f"  {_yellow('SKIP'):<20} [{key.upper()}] metric check skipped")
            skipped_count += 1
            continue
        t0 = time.time()
        try:
            audit = probe(device)
            metric_results[audit.key] = audit
            _print_metric_audit(audit)
            if audit.status != "LOADED":
                failed_count += 1
        except Exception as e:
            failed_count += 1
            print(f"  {_red('NOT LOADED'):<20} [{key.upper()}] internal checker failure")
            print(f"      error            : {type(e).__name__}: {e}")
            if verbose:
                traceback.print_exc()
        dt = time.time() - t0
        print(f"      elapsed          : {dt:.1f}s\n")

    # EDA summary tied to current metric statuses and local dependency availability
    print(_cyan("EDA REQUIREMENT AUDIT"))
    print("-" * 90)
    eda_rows = _eda_summaries(metric_results)
    for row in eda_rows:
        if any(s.lower() in row.key.lower() for s in skip):
            print(f"  {_yellow('SKIP'):<20} [{row.key.upper()}] EDA check skipped")
            skipped_count += 1
            continue
        _print_eda_summary(row)
        print()

    print("=" * 90)
    print("  Final Summary")
    print("=" * 90)
    loaded_metrics = sum(1 for x in metric_results.values() if x.status == "LOADED")
    total_metrics = len(metric_results)
    print(f"  {_green('Loaded metrics')}     : {loaded_metrics}/{total_metrics}")
    print(f"  {_red('Not loaded metrics')} : {failed_count}")
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
        ])
    if metric_rows:
        _print_table(
            "  Summarized Table: Metrics",
            ["Key", "Metric", "Status", "Loaded Backend", "Fallback"],
            metric_rows,
        )
        print()

    eda_rows_table: List[List[object]] = []
    for row in eda_rows:
        if any(s.lower() in row.key.lower() for s in skip):
            continue
        eda_rows_table.append([
            row.key.upper(),
            row.plot,
            row.status,
            row.selected_mode,
            row.fallback_used,
        ])
    if eda_rows_table:
        _print_table(
            "  Summarized Table: EDA",
            ["Key", "Plot", "Status", "Selected Mode", "Fallback"],
            eda_rows_table,
        )
        print()

    return failed_count


def _parse():
    p = argparse.ArgumentParser(
        description="Audit pretrained metric backends and EDA fallback requirements"
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for backend initialization",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="Substring keywords to skip checks (case-insensitive), e.g. m8 p9 vlm",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print full traceback for failed checks",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    n_failed = run_checks(args.device, args.skip, args.verbose)
    sys.exit(0 if n_failed == 0 else 1)
