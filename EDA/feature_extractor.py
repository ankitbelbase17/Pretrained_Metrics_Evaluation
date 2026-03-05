"""
EDA/feature_extractor.py
=========================
Centralised feature extraction pipeline.

Runs ONCE per dataset; results are cached to a ``.npz`` file so all
plotting scripts load instantly without re-running heavy models.

**Dataloader source**
All datasets are loaded via ``datasets/loaders.py`` and
``datasets/base_dataset.py`` — the single canonical source of truth for
folder structures, split logic, and image transforms.  No dataset classes
are re-implemented here.

Per-image features stored
--------------------------
  v_i       : (34,)    normalised pose vector                (m1)
  angles_i  : (8,)    joint angles for 8 limb triplets     (m1)
  occ_i     : float   garment-occlusion ratio              (m2)
  occ_mask_i: (H,W)   spatial occlusion mask (downsampled) (m2)
  bg_ent_i  : float   background texture entropy           (m3)
  bg_obj_i  : int     #objects in background               (m3)
  lum_i     : float   mean LAB-L luminance                 (m4)
  grad_var_i: float   Var(Sobel magnitude) of L channel    (m4)
  lum_map_i : (h,w)   downsampled L channel map            (m4)
  beta_i    : (10,)   body shape coefficients (proxy)      (m5)
  face_i    : (D,)    ArcFace / CLIP face embedding        (m6)
  garment_i : (D,)    CLIP garment embedding               (m7)
  vae_i     : (D,)    Stable Diffusion VAE latent          (m8)

Everything is stored as a single ``.npz`` file per dataset in ``cache_dir``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── workspace root on path ────────────────────────────────────────────────────
_HERE      = Path(__file__).parent          # EDA/
_WORKSPACE = _HERE.parent                   # pretrained_metrics_evals/
sys.path.insert(0, str(_WORKSPACE))

# ── Canonical dataloader — uses datasets/ exclusively ────────────────────────
from pretrained_metrics.dataloader import get_dataloader   # thin shim → datasets/

# ── Metric backends (extractors only — we don't call .compute()) ─────────────
from pretrained_metrics.metrics.m1_pose            import _KeypointExtractor, _normalise_pose, _joint_angle, TRIPLET_IDX
from pretrained_metrics.metrics.m2_occlusion       import _SegBackend
from pretrained_metrics.metrics.m3_background      import _PersonSegmenter, _texture_entropy, _ObjectDetector
from pretrained_metrics.metrics.m4_illumination    import _rgb_to_lab_l, _sobel_gradient_variance
from pretrained_metrics.metrics.m5_body_shape      import _ShapeExtractor
from pretrained_metrics.metrics.m6_appearance      import _FaceEmbedder
from pretrained_metrics.metrics.m7_garment_texture import _GarmentEncoder
from pretrained_metrics.metrics.m8_vae_latent      import _VAEEncoder


# Import standalone dataloaders (from dataloaders/ package)
try:
    from dataloaders.dresscode_dataloader import Dresscode as StandaloneDresscode, custom_collate_fn as dresscode_collate
    from dataloaders.vitonhd_dataloader import VITONHDDataset as StandaloneVITONHD, custom_collate_fn as vitonhd_collate
    from dataloaders.laion_rvs_fashion_dataloader import LAIONRVSFashionDataset as StandaloneLAION
    from dataloaders import (
        canonical_collate_dresscode,
        canonical_collate_vitonhd,
        canonical_collate_laion,
    )
except ImportError:
    StandaloneDresscode = StandaloneVITONHD = StandaloneLAION = None
    canonical_collate_dresscode = canonical_collate_vitonhd = canonical_collate_laion = None


# Downsampled spatial-map resolution stored per image (keeps cache small)
MASK_DS = (64, 48)   # (H, W)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts and caches all per-image features for EDA.

    Parameters
    ----------
    device    : torch device string (``"cpu"`` or ``"cuda"``)
    cache_dir : directory where ``.npz`` cache files are written
    """

    def __init__(self, device: str = "cpu", cache_dir: str = "./eda_cache"):
        self.device    = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print("[FeatureExtractor] Loading pretrained backends …")
        self._kp_ext     = _KeypointExtractor(device)
        self._seg        = _SegBackend(device)
        self._per_seg    = _PersonSegmenter(device)
        self._obj_det    = _ObjectDetector(device)
        self._shape_ex   = _ShapeExtractor(device)
        self._face_ex    = _FaceEmbedder(device)
        self._garment_ex = _GarmentEncoder(device)
        self._vae_ex     = _VAEEncoder(device)
        print("[FeatureExtractor] All backends ready.")

    # ── single-image extraction (used by run_curvton_eda) ─────────────── #
    _img_transform = None

    @staticmethod
    def _get_transform(img_size: Tuple[int, int] = (512, 384)):
        import torchvision.transforms as T
        return T.Compose([T.Resize(img_size), T.ToTensor()])

    def extract_all(
        self,
        person_path: str,
        cloth_path: str,
        img_size: Tuple[int, int] = (512, 384),
    ) -> Dict[str, np.ndarray]:
        """
        Extract **all** per-image features from a single (person, cloth) pair.

        This is the single-image counterpart of :meth:`extract` (which
        processes a full dataset via DataLoader).  Called by
        ``run_curvton_eda.extract_features_for_difficulty``.

        Parameters
        ----------
        person_path : str – path to the person / initial image
        cloth_path  : str – path to the cloth image

        Returns
        -------
        dict with keys that match the batch-level feature names:
            pose_vecs, angles, occlusion, bg_entropy, bg_obj_count,
            lum_mean, lum_grad_var, betas, face_embs, garment_embs
        """
        from PIL import Image

        if self._img_transform is None:
            FeatureExtractor._img_transform = self._get_transform(img_size)
        tf = self._img_transform

        person_t = tf(Image.open(person_path).convert("RGB")).unsqueeze(0)  # (1,3,H,W)
        cloth_t  = tf(Image.open(cloth_path).convert("RGB")).unsqueeze(0)

        out: Dict[str, np.ndarray] = {}

        # M1 – Pose
        kps_raw = self._kp_ext(person_t)                 # (1,17,2)
        kps_norm, valid = _normalise_pose(kps_raw)
        if valid[0]:
            pn = kps_norm[0]
            out["pose_vecs"] = pn.flatten().astype(np.float32)
            ang = [
                _joint_angle(pn[ia], pn[ib], pn[ic])
                for ia, ib, ic in TRIPLET_IDX
            ]
            out["angles"] = np.array(
                [a if not math.isnan(a) else 0.0 for a in ang], dtype=np.float32,
            )
        else:
            out["pose_vecs"] = np.zeros(34, dtype=np.float32)
            out["angles"]    = np.zeros(len(TRIPLET_IDX), dtype=np.float32)

        # M2 – Occlusion
        seg = self._seg.segment(person_t)
        G  = seg["garment"].float()
        occ = ((seg["arms"].float() + seg["hair"].float() + seg["other"].float()) > 0).float()
        overlap = G * occ
        g_area = G[0].sum().item()
        out["occlusion"] = float(min(overlap[0].sum().item() / max(g_area, 1.0), 1.0))

        # M3 – Background
        pmask = self._per_seg(person_t)
        obj_c = self._obj_det.count_objects(person_t, pmask)
        ent   = _texture_entropy(person_t[0], pmask[0])
        out["bg_entropy"]   = float(ent) if not math.isnan(ent) else 0.0
        out["bg_obj_count"] = int(obj_c[0])

        # M4 – Illumination
        mean_L, L_maps = _rgb_to_lab_l(person_t.cpu())
        out["lum_mean"]     = float(mean_L[0])
        out["lum_grad_var"] = _sobel_gradient_variance(L_maps[0])

        # M5 – Body shape
        b = self._shape_ex(person_t)
        out["betas"] = b[0].astype(np.float32)

        # M6 – Appearance
        f = self._face_ex(person_t)
        out["face_embs"] = f[0].astype(np.float32)

        # M7 – Garment
        g = self._garment_ex(cloth_t)
        out["garment_embs"] = g[0].astype(np.float32)

        return out

    # ── cache management ──────────────────────────────────────────────── #
    def cache_path(self, dataset_name: str) -> Path:
        return self.cache_dir / f"{dataset_name}_features.npz"

    def is_cached(self, dataset_name: str) -> bool:
        return self.cache_path(dataset_name).exists()

    # ── main extraction ───────────────────────────────────────────────── #
    def extract(
        self,
        dataset_name: str,
        root: str = None,
        split: str = "test",
        batch_size: int = 8,
        num_workers: int = 4,
        img_size: Tuple[int, int] = (512, 384),
        force: bool = False,
        use_anish: bool = False,
        cache_label: str = None,
        **dl_kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features or load from cache.

        The DataLoader is obtained from ``pretrained_metrics.dataloader.get_dataloader``
        which in turn delegates entirely to ``datasets/loaders.py``.
        All split percentages and folder structures are governed by that module.

        Parameters
        ----------
        dataset_name : str    — one of ALL_DATASETS
        root         : str    — dataset root directory
        split        : str    — ``"train"`` | ``"test"`` | ``"val"``
        batch_size   : int
        num_workers  : int
        img_size     : (H, W)
        force        : bool   — re-extract even if cache exists
        use_anish    : bool   — use dedicated Anish dataloaders
        cache_label  : str    — override the cache filename stem
                                (useful for dresscode categories:
                                 ``"dresscode_upper_body"`` etc.)
        **dl_kwargs  : forwarded to get_dataloader (e.g. category for DressCode)

        Returns
        -------
        dict of numpy arrays (per-image features)
        """
        cp = self.cache_path(cache_label or dataset_name)
        if cp.exists() and not force:
            print(f"[FeatureExtractor] Loading cached features → {cp}")
            return dict(np.load(cp, allow_pickle=True))

        # Resolve root from config if available
        if not root:
            try:
                import config
                root = config.get_root(dataset_name)
            except ImportError:
                pass

        # Adjust dataset name if using anish specialized loaders
        # NOTE: LAION is excluded from EDA - only dresscode, vitonhd, street_tryon supported
        if use_anish and not dataset_name.endswith("_anish"):
            if dataset_name.lower() in ["dresscode", "vitonhd", "street_tryon"]:
                dataset_name = f"{dataset_name.lower()}_anish"

        label_for_print = cache_label or dataset_name
        print(f"\n[FeatureExtractor] Extracting '{label_for_print}' (split={split}) …")

        # ── DataLoader ────────────────────────────────────────────────── #
        loader = get_dataloader(
            dataset_name, root,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            **dl_kwargs,
        )
        N = len(loader.dataset)
        print(f"[FeatureExtractor]  {N} samples | batch_size={batch_size}")

        # ── Accumulators ──────────────────────────────────────────────── #
        pose_vecs, angles_list = [], []
        occ_ratios, occ_maps   = [], []
        bg_ents, bg_objs       = [], []
        lum_means, lum_vars, lum_maps_acc = [], [], []
        betas, face_embs, garment_embs    = [], [], []
        vae_embs = []

        for batch in tqdm(loader, desc=f"  {dataset_name}", unit="batch"):
            person = batch["person"].float()   # (B, 3, H, W)  [0,1]
            cloth  = batch["cloth"].float()    # (B, 3, H, W)  [0,1]
            B = person.shape[0]

            # ── M1: Pose ─────────────────────────────────────────────── #
            kps_raw   = self._kp_ext(person)          # (B, 17, 2)
            kps_norm, valid = _normalise_pose(kps_raw)

            for i in range(B):
                if valid[i]:
                    pn = kps_norm[i]
                    pose_vecs.append(pn.flatten().astype(np.float32))
                    ang = [
                        _joint_angle(pn[ia], pn[ib], pn[ic])
                        for ia, ib, ic in TRIPLET_IDX
                    ]
                    angles_list.append(np.array(
                        [a if not math.isnan(a) else 0.0 for a in ang],
                        dtype=np.float32,
                    ))
                else:
                    pose_vecs.append(np.zeros(34, dtype=np.float32))
                    angles_list.append(np.zeros(len(TRIPLET_IDX), dtype=np.float32))

            # ── M2: Occlusion ─────────────────────────────────────────── #
            seg_masks = self._seg.segment(person)
            G  = seg_masks["garment"].float()
            A  = seg_masks["arms"].float()
            Ha = seg_masks["hair"].float()
            Ot = seg_masks["other"].float()
            occluder = ((A + Ha + Ot) > 0).float()
            overlap  = G * occluder

            for i in range(B):
                g_area = G[i].sum().item()
                ratio  = overlap[i].sum().item() / max(g_area, 1.0)
                occ_ratios.append(float(min(ratio, 1.0)))

                full_map = overlap[i].unsqueeze(0).unsqueeze(0)
                ds_map   = torch.nn.functional.interpolate(
                    full_map, size=MASK_DS, mode="bilinear", align_corners=False
                ).squeeze().numpy()
                occ_maps.append(ds_map.astype(np.float32))

            # ── M3: Background ─────────────────────────────────────────── #
            person_masks = self._per_seg(person)         # (B, H, W) bool
            obj_counts   = self._obj_det.count_objects(person, person_masks)
            for i in range(B):
                ent = _texture_entropy(person[i], person_masks[i])
                bg_ents.append(float(ent) if not math.isnan(ent) else 0.0)
                bg_objs.append(int(obj_counts[i]))

            # ── M4: Illumination ───────────────────────────────────────── #
            mean_L, L_maps = _rgb_to_lab_l(person.cpu())
            for i in range(B):
                lum_means.append(float(mean_L[i]))
                lum_vars.append(_sobel_gradient_variance(L_maps[i]))
                ds_L = cv2.resize(
                    L_maps[i], (MASK_DS[1], MASK_DS[0]),
                    interpolation=cv2.INTER_AREA,
                )
                lum_maps_acc.append(ds_L.astype(np.float32))

            # ── M5: Body Shape ──────────────────────────────────────────── #
            b = self._shape_ex(person)                   # (B, 10)
            for bi in b:
                betas.append(bi.astype(np.float32))

            # ── M6: Appearance ──────────────────────────────────────────── #
            f = self._face_ex(person)                    # (B, D)
            for fi in f:
                face_embs.append(fi.astype(np.float32))

            # ── M7: Garment ─────────────────────────────────────────────── #
            g = self._garment_ex(cloth)                  # (B, D)
            for gi in g:
                garment_embs.append(gi.astype(np.float32))

            # ── M8: VAE Latent ──────────────────────────────────────────── #
            v = self._vae_ex(person)                     # (B, D)
            for vi in v:
                vae_embs.append(vi.astype(np.float32))

        # ── Pack and save ───────────────────────────────────────────────── #
        data = dict(
            pose_vecs    = np.stack(pose_vecs),
            angles       = np.stack(angles_list),
            occ_ratios   = np.array(occ_ratios,   dtype=np.float32),
            occ_maps     = np.stack(occ_maps),
            bg_entropy   = np.array(bg_ents,       dtype=np.float32),
            bg_obj_count = np.array(bg_objs,       dtype=np.int32),
            lum_mean     = np.array(lum_means,     dtype=np.float32),
            lum_grad_var = np.array(lum_vars,      dtype=np.float32),
            lum_maps     = np.stack(lum_maps_acc),
            betas        = np.stack(betas),
            face_embs    = np.stack(face_embs),
            garment_embs = np.stack(garment_embs),
            vae_embs     = np.stack(vae_embs),
        )
        np.savez_compressed(cp, **data)
        print(f"[FeatureExtractor] Cached → {cp}")
        return data
