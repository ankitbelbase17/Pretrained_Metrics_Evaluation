from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.curvton_dataloader import CURVTONDataloader
from pretrained_metrics.metrics.m1_pose import (
    TRIPLET_IDX,
    _KeypointExtractor,
    _joint_angle,
    _normalise_pose,
)
from pretrained_metrics.metrics.m2_occlusion import _SegBackend
from pretrained_metrics.metrics.m3_background import (
    _ObjectDetector,
    _PersonSegmenter,
    _texture_entropy,
)
from pretrained_metrics.metrics.m4_illumination import (
    _rgb_to_lab_l,
    _sobel_gradient_variance,
)
from pretrained_metrics.metrics.m5_body_shape import _ShapeExtractor
from pretrained_metrics.metrics.m7_garment_texture import _GarmentEncoder
from pretrained_metrics.metrics.m9_camera_angle import _CameraAngleBackend


MASK_DS = (64, 48)

ALL_FEATURE_KEYS = {
    "pose_vecs",
    "angles",
    "occ_ratios",
    "occ_maps",
    "bg_entropy",
    "bg_obj_count",
    "lum_mean",
    "lum_grad_var",
    "lum_maps",
    "betas",
    "garment_embs",
    "azimuths",
    "elevations",
    "camera_confidence",
}


def _m2_occ_ratio_and_map(
    seg_masks: Dict[str, torch.Tensor],
    idx: int,
) -> Tuple[float, np.ndarray]:
    """Match m2_occlusion person-centric logic for one sample."""
    G = seg_masks["garment"].float()
    body = seg_masks.get("body_parts", seg_masks.get("arms", torch.zeros_like(G))).float()
    carried = seg_masks.get("carried_obj", torch.zeros_like(G)).float()
    accessory = seg_masks.get("accessories", torch.zeros_like(G)).float()
    env = seg_masks.get("environment", torch.zeros_like(G)).float()
    people = seg_masks.get("other_people", torch.zeros_like(G)).float()
    other = seg_masks.get("other", torch.zeros_like(G)).float()

    if body.sum() == 0 and "arms" in seg_masks:
        arms = seg_masks["arms"].float()
        hair = seg_masks.get("hair", torch.zeros_like(G)).float()
        body = ((arms + hair) > 0).float()

    H, W = G.shape[-2], G.shape[-1]
    person_core = ((G[idx] + body[idx]) > 0)
    if person_core.sum() < 10:
        return 0.0, np.zeros(MASK_DS, dtype=np.float32)

    rows = person_core.any(dim=1).nonzero(as_tuple=True)[0]
    cols = person_core.any(dim=0).nonzero(as_tuple=True)[0]
    r0, r1 = rows[0].item(), rows[-1].item()
    c0, c1 = cols[0].item(), cols[-1].item()

    pr = max(1, int((r1 - r0) * 0.10))
    pc = max(1, int((c1 - c0) * 0.10))
    r0 = max(0, r0 - pr)
    r1 = min(H - 1, r1 + pr)
    c0 = max(0, c0 - pc)
    c1 = min(W - 1, c1 + pc)

    g_in = G[idx][r0 : r1 + 1, c0 : c1 + 1].sum().item()
    b_in = body[idx][r0 : r1 + 1, c0 : c1 + 1].sum().item()
    ca_in = carried[idx][r0 : r1 + 1, c0 : c1 + 1].sum().item()
    ac_in = accessory[idx][r0 : r1 + 1, c0 : c1 + 1].sum().item()
    en_in = env[idx][r0 : r1 + 1, c0 : c1 + 1].sum().item()
    pe_in = people[idx][r0 : r1 + 1, c0 : c1 + 1].sum().item()
    ot_in = other[idx][r0 : r1 + 1, c0 : c1 + 1].sum().item()

    person_px = g_in + b_in
    occluder_px = ca_in + ac_in + en_in + pe_in + ot_in
    base_area = float(person_px + occluder_px)
    if base_area < 1.0:
        base_area = 1.0
    occ_ratio = float(occluder_px / base_area)

    # Spatial map: category-aware non-person occluders constrained to person bbox.
    occ_map_full = torch.zeros_like(G[idx], dtype=torch.float32)
    occ_map_bbox = (
        carried[idx][r0 : r1 + 1, c0 : c1 + 1]
        + accessory[idx][r0 : r1 + 1, c0 : c1 + 1]
        + env[idx][r0 : r1 + 1, c0 : c1 + 1]
        + people[idx][r0 : r1 + 1, c0 : c1 + 1]
        + other[idx][r0 : r1 + 1, c0 : c1 + 1]
    )
    occ_map_bbox = (occ_map_bbox > 0).float()
    occ_map_full[r0 : r1 + 1, c0 : c1 + 1] = occ_map_bbox

    ds_map = F.interpolate(
        occ_map_full.unsqueeze(0).unsqueeze(0),
        size=MASK_DS,
        mode="bilinear",
        align_corners=False,
    )
    return occ_ratio, ds_map.squeeze().cpu().numpy().astype(np.float32)


def _free_gpu() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_tensor_loader(
    base_path: str,
    difficulty: str,
    gender: str,
    sample_ratio: float,
    seed: int,
    max_samples: int | None,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    from PIL import Image

    tf = T.Compose([T.Resize(img_size), T.ToTensor()])

    class _CurvtonTensorDataset(Dataset):
        def __init__(self, loader: CURVTONDataloader) -> None:
            self.samples = loader.samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            sample = self.samples[idx]
            try:
                person = tf(Image.open(sample["person_path"]).convert("RGB"))
                cloth = tf(Image.open(sample["cloth_path"]).convert("RGB"))
            except Exception:
                person = torch.zeros(3, img_size[0], img_size[1])
                cloth = torch.zeros(3, img_size[0], img_size[1])
            return person, cloth

    loader = CURVTONDataloader(
        base_path=base_path,
        difficulty=difficulty,
        gender=gender,
        sample_ratio=sample_ratio,
        seed=seed,
        return_paths=True,
        max_samples=max_samples,
    )

    dataset = _CurvtonTensorDataset(loader)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _select_garment_embeddings(garment_out: object, preferred: str | None = None) -> np.ndarray:
    if isinstance(garment_out, dict):
        if preferred in garment_out:
            return garment_out[preferred]
        if "fashion_clip" in garment_out:
            return garment_out["fashion_clip"]
        if "dinov2" in garment_out:
            return garment_out["dinov2"]
        return next(iter(garment_out.values()))
    return garment_out


class _SDCLIPGarmentEncoder:
    def __init__(self, model_id: str, device: str) -> None:
        from transformers import AutoProcessor, CLIPModel

        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def encode(self, cloth_imgs: torch.Tensor) -> np.ndarray:
        import torchvision.transforms.functional as TF

        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in cloth_imgs]
        inputs = self.processor(images=pils, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model.get_image_features(**inputs)
        if torch.is_tensor(out):
            feats = out
        elif hasattr(out, "image_embeds") and torch.is_tensor(out.image_embeds):
            feats = out.image_embeds
        elif hasattr(out, "pooler_output") and torch.is_tensor(out.pooler_output):
            feats = out.pooler_output
        elif hasattr(out, "last_hidden_state") and torch.is_tensor(out.last_hidden_state):
            feats = out.last_hidden_state.mean(dim=1)
        else:
            vision_out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            feats = vision_out.pooler_output
            if hasattr(self.model, "visual_projection"):
                feats = self.model.visual_projection(feats)
        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()


def _verify_models(
    device: str,
    garment_backend: str,
    sd_clip_model_id: str,
    required_keys: Iterable[str] | None,
) -> None:
    required = set(required_keys) if required_keys else set(ALL_FEATURE_KEYS)
    try:
        if {"pose_vecs", "angles"} & required:
            _KeypointExtractor(device)
        if {"occ_ratios", "occ_maps"} & required:
            _SegBackend(device)
        if {"bg_entropy", "bg_obj_count"} & required:
            _PersonSegmenter(device)
            _ObjectDetector(device)
        if {"lum_mean", "lum_grad_var", "lum_maps"} & required:
            pass
        if "betas" in required:
            _ShapeExtractor(device)
        if "garment_embs" in required:
            if garment_backend == "sd_clip":
                _SDCLIPGarmentEncoder(sd_clip_model_id, device)
            else:
                _GarmentEncoder(device)
        if {"azimuths", "elevations", "camera_confidence"} & required:
            _CameraAngleBackend(device)
    except Exception as exc:
        raise RuntimeError(f"Model initialization failed: {exc}") from exc
    finally:
        _free_gpu()


def _extract_features(
    dl: torch.utils.data.DataLoader,
    device: str,
    img_size: Tuple[int, int],
    garment_backend: str,
    sd_clip_model_id: str,
    required_keys: Iterable[str] | None,
) -> Dict[str, np.ndarray]:
    required = set(required_keys) if required_keys else set(ALL_FEATURE_KEYS)

    pose_backend = _KeypointExtractor(device) if {"pose_vecs", "angles"} & required else None
    seg_backend = _SegBackend(device) if {"occ_ratios", "occ_maps"} & required else None
    person_seg = _PersonSegmenter(device) if {"bg_entropy", "bg_obj_count"} & required else None
    obj_det = _ObjectDetector(device) if {"bg_entropy", "bg_obj_count"} & required else None
    shape_backend = _ShapeExtractor(device) if "betas" in required else None

    garment_encoder = None
    sd_clip_encoder = None
    if "garment_embs" in required:
        if garment_backend == "sd_clip":
            sd_clip_encoder = _SDCLIPGarmentEncoder(sd_clip_model_id, device)
        else:
            garment_encoder = _GarmentEncoder(device)

    camera_backend = None
    if {"azimuths", "elevations", "camera_confidence"} & required:
        try:
            camera_backend = _CameraAngleBackend(device)
        except Exception:
            camera_backend = None

    pose_vecs: List[np.ndarray] = []
    angles_list: List[np.ndarray] = []
    occ_ratios: List[float] = []
    occ_maps: List[np.ndarray] = []
    bg_ents: List[float] = []
    bg_objs: List[int] = []
    lum_means: List[float] = []
    lum_vars: List[float] = []
    lum_maps_acc: List[np.ndarray] = []
    betas: List[np.ndarray] = []
    garment_embs: List[np.ndarray] = []
    azimuths: List[float] = []
    elevations: List[float] = []
    cam_conf: List[float] = []

    for person, cloth in dl:
        if pose_backend is not None:
            # M1: Pose
            kps_raw = pose_backend(person)
            kps_norm, valid = _normalise_pose(kps_raw)
            for i in range(person.shape[0]):
                if np.any(valid[i]):
                    pn = kps_norm[i]
                    pose_vecs.append(pn.flatten().astype(np.float32))
                    ang = [_joint_angle(pn[ia], pn[ib], pn[ic]) for ia, ib, ic in TRIPLET_IDX]
                    angles_list.append(
                        np.array([a if not np.isnan(a) else 0.0 for a in ang], dtype=np.float32)
                    )
                else:
                    pose_vecs.append(np.zeros(34, dtype=np.float32))
                    angles_list.append(np.zeros(len(TRIPLET_IDX), dtype=np.float32))

        if seg_backend is not None:
            # M2: Occlusion
            seg_masks = seg_backend.segment(person)
            for i in range(person.shape[0]):
                ratio, ds_map = _m2_occ_ratio_and_map(seg_masks, i)
                occ_ratios.append(ratio)
                occ_maps.append(ds_map)

        if person_seg is not None and obj_det is not None:
            # M3: Background
            person_masks = person_seg(person)
            obj_counts = obj_det.count_objects(person, person_masks)
            for i in range(person.shape[0]):
                ent = _texture_entropy(person[i], person_masks[i])
                bg_ents.append(float(ent) if not np.isnan(ent) else 0.0)
                bg_objs.append(int(obj_counts[i]))

        if {"lum_mean", "lum_grad_var", "lum_maps"} & required:
            # M4: Illumination
            mean_L, L_maps = _rgb_to_lab_l(person.cpu())
            for i in range(person.shape[0]):
                lum_means.append(float(mean_L[i]))
                lum_vars.append(_sobel_gradient_variance(L_maps[i]))
                ds_L = cv2.resize(L_maps[i], (MASK_DS[1], MASK_DS[0]), interpolation=cv2.INTER_AREA)
                lum_maps_acc.append(ds_L.astype(np.float32))

        if shape_backend is not None:
            # M5: Body shape
            b = shape_backend(person)
            for bi in b:
                betas.append(bi.astype(np.float32))

        if "garment_embs" in required:
            # M7: Garment texture
            if sd_clip_encoder is not None:
                g = sd_clip_encoder.encode(cloth)
            else:
                preferred = None
                if garment_backend in {"fashion_clip", "dinov2"}:
                    preferred = garment_backend
                g = _select_garment_embeddings(garment_encoder(cloth), preferred=preferred)
            for gi in g:
                garment_embs.append(np.asarray(gi, dtype=np.float32))

        if camera_backend is not None:
            # M9: Camera angle (optional)
            try:
                cam = camera_backend.estimate_angles(person)
                az = cam["azimuth"].detach().cpu().numpy().astype(np.float32)
                el = cam["elevation"].detach().cpu().numpy().astype(np.float32)
                cf = cam["confidence"].detach().cpu().numpy().astype(np.float32)
                for i in range(person.shape[0]):
                    azimuths.append(float(az[i]))
                    elevations.append(float(el[i]))
                    cam_conf.append(float(cf[i]))
            except Exception:
                camera_backend = None

    data: Dict[str, np.ndarray] = {}
    if "pose_vecs" in required:
        data["pose_vecs"] = np.stack(pose_vecs) if pose_vecs else np.array([])
    if "angles" in required:
        data["angles"] = np.stack(angles_list) if angles_list else np.array([])
    if "occ_ratios" in required:
        data["occ_ratios"] = np.array(occ_ratios, dtype=np.float32)
    if "occ_maps" in required:
        data["occ_maps"] = np.stack(occ_maps) if occ_maps else np.array([])
    if "bg_entropy" in required:
        data["bg_entropy"] = np.array(bg_ents, dtype=np.float32)
    if "bg_obj_count" in required:
        data["bg_obj_count"] = np.array(bg_objs, dtype=np.int32)
    if "lum_mean" in required:
        data["lum_mean"] = np.array(lum_means, dtype=np.float32)
    if "lum_grad_var" in required:
        data["lum_grad_var"] = np.array(lum_vars, dtype=np.float32)
    if "lum_maps" in required:
        data["lum_maps"] = np.stack(lum_maps_acc) if lum_maps_acc else np.array([])
    if "betas" in required:
        data["betas"] = np.stack(betas) if betas else np.array([])
    if "garment_embs" in required:
        data["garment_embs"] = np.stack(garment_embs) if garment_embs else np.array([])
    if "azimuths" in required and azimuths:
        data["azimuths"] = np.array(azimuths, dtype=np.float32)
    if "elevations" in required and elevations:
        data["elevations"] = np.array(elevations, dtype=np.float32)
    if "camera_confidence" in required and cam_conf:
        data["camera_confidence"] = np.array(cam_conf, dtype=np.float32)

    return data


def _save_cache(cache_path: Path, data: Dict[str, np.ndarray], force: bool) -> None:
    if cache_path.exists() and not force:
        print(f"[skip] Cache exists: {cache_path}")
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f"[save] {cache_path}")


def _save_occ_maps_only(
    occ_maps_path: Path,
    data: Dict[str, np.ndarray],
    force: bool,
) -> None:
    if "occ_maps" not in data:
        return
    if occ_maps_path.exists() and not force:
        print(f"[skip] Occ maps cache exists: {occ_maps_path}")
        return
    occ_maps_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(occ_maps_path, occ_maps=data["occ_maps"])
    print(f"[save] {occ_maps_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CurvTON feature caches for plot_scripts (easy/medium/hard/all)."
    )
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("./eda_cache/curvton"))
    parser.add_argument(
        "--occ-maps-dir",
        type=Path,
        default=None,
        help="Optional separate directory to save occ_maps-only NPZ per difficulty.",
    )
    parser.add_argument("--sample-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gender", type=str, default="all", choices=["all", "male", "female"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=(512, 384))
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "medium", "hard", "all"],
        choices=["easy", "medium", "hard", "all"],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--garment-backend",
        type=str,
        default="ensemble",
        choices=["ensemble", "fashion_clip", "dinov2", "sd_clip"],
        help="Garment embedding backend (sd_clip uses OpenAI CLIP like Stable Diffusion).",
    )
    parser.add_argument(
        "--sd-clip-model-id",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="Model id to use when --garment-backend=sd_clip.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--required-keys",
        nargs="+",
        default=None,
        help="Limit features to these keys (defaults to all).",
    )
    return parser.parse_args()


def generate_curvton_caches(
    *,
    base_path: str,
    cache_dir: Path,
    occ_maps_dir: Path | None,
    sample_ratio: float,
    difficulties: List[str],
    seed: int,
    batch_size: int,
    num_workers: int,
    gender: str,
    max_samples: int | None,
    img_size: Tuple[int, int],
    device: str,
    garment_backend: str,
    sd_clip_model_id: str,
    force: bool,
    required_keys: Iterable[str] | None = None,
) -> None:
    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1]")

    _verify_models(device, garment_backend, sd_clip_model_id, required_keys)

    pct = int(round(sample_ratio * 100))
    for diff in difficulties:
        cache_path = cache_dir / f"curvton_{diff}_{pct}pct.npz"
        dl = _build_tensor_loader(
            base_path=base_path,
            difficulty=diff,
            gender=gender,
            sample_ratio=sample_ratio,
            seed=seed,
            max_samples=max_samples,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        data = _extract_features(
            dl,
            device=device,
            img_size=img_size,
            garment_backend=garment_backend,
            sd_clip_model_id=sd_clip_model_id,
            required_keys=required_keys,
        )
        _save_cache(cache_path, data, force=force)
        if occ_maps_dir is not None and "occ_maps" in data:
            occ_maps_path = occ_maps_dir / f"curvton_{diff}_{pct}pct_occ_maps.npz"
            _save_occ_maps_only(occ_maps_path, data, force=force)


def main() -> int:
    args = parse_args()

    generate_curvton_caches(
        base_path=args.base_path,
        cache_dir=args.cache_dir,
        occ_maps_dir=args.occ_maps_dir,
        sample_ratio=args.sample_ratio,
        difficulties=list(args.difficulties),
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gender=args.gender,
        max_samples=args.max_samples,
        img_size=tuple(args.img_size),
        device=args.device,
        garment_backend=args.garment_backend,
        sd_clip_model_id=args.sd_clip_model_id,
        force=args.force,
        required_keys=args.required_keys,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
