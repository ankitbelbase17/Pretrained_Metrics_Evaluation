"""
metrics/m2_occlusion.py
========================
Metric 2 — Generalized Occlusion Complexity
--------------------------------------------
Measures how much the garment region is occluded by ANY occluding objects
and how much that occlusion varies across the dataset.

Occlusion Sources (Comprehensive)
----------------------------------
  1. Body Parts:
     - Arms (crossing torso, folded)
     - Hands (holding objects, gestures)
     - Hair (long hair covering shoulders/chest)
     - Face/Head (tilted, covering neckline)
     - Legs (crossed, covering lower garments)

  2. Carried Objects:
     - Handbags, purses, clutches
     - Bag straps (crossbody, shoulder straps)
     - Backpacks
     - Shopping bags
     - Phone, tablet held in front

  3. Worn Accessories:
     - Scarves, shawls
     - Jewelry (necklaces covering neckline)
     - Belts (if covering garment details)
     - Sunglasses (on chest/neckline)
     - Hats (wide brims casting shadows)

  4. Environmental Occlusion:
     - Other people (person behind, crowd)
     - Furniture (chairs, tables)
     - Railings, poles
     - Plants, foliage
     - Vehicles (partial visibility)

  5. Self-Occlusion:
     - Garment folds/draping
     - Layered clothing
     - Open jackets/cardigans

Formula
--------
    C_occ = E[O_i] + Var(O_i)

where O_i = |G_i ∩ (Union of all occluders)| / |G_i|

Per-category breakdown also provided for detailed analysis.

Pretrained Models
------------------
Primary:  Mask2Former (facebook/mask2former-swin-large-coco-panoptic)
          - 133 COCO panoptic classes for comprehensive object detection
Fallback: Segformer-B2-clothes (mattmdjaga/segformer_b2_clothes)
          - 18-class human parsing
Fallback: DeepLabV3-ResNet101 + object detection
Fallback: Gradient-based saliency proxy

Input
------
person_imgs : torch.Tensor  (B, 3, H, W)  float32  [0, 1]

Returns (via compute())
------------------------
dict with:
    occlusion_mean           : E[O_i]  (total occlusion)
    occlusion_var            : Var(O_i)
    occlusion_complexity     : E[O_i] + Var(O_i)   (C_occ)
    occlusion_by_body_parts  : E[O_body]  (arms, hands, hair)
    occlusion_by_objects     : E[O_obj]   (bags, accessories)
    occlusion_by_environment : E[O_env]   (other people, furniture)
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation backend
# ─────────────────────────────────────────────────────────────────────────────

class _SegBackend:
    """
    Abstracts the segmentation model; returns per-pixel class maps.

    Backend priority:
      1. Mask2Former (facebook/mask2former-swin-large-coco-panoptic)
         - 133 COCO panoptic classes for comprehensive object detection
         - Detects people, bags, furniture, vehicles, etc.
      
      2. mattmdjaga/segformer_b2_clothes  — 18-class human parsing (HuggingFace)
         Classes: 0=BG,1=Hat,2=Hair,3=Sunglasses,4=Upper-clothes,5=Skirt,
                  6=Pants,7=Dress,8=Belt,9=Left-shoe,10=Right-shoe,11=Face,
                  12=Left-leg,13=Right-leg,14=Left-arm,15=Right-arm,16=Bag,17=Scarf
         → garment={4,5,6,7}, arms={14,15}, hair={2}, accessories={1,3,8,16,17}

      3. DeepLabV3-ResNet101 + skin-colour proxy  — torchvision (Pascal VOC 21).
         class 15 = person. Within the person region, skin-coloured pixels
         (detected via YCbCr thresholds) serve as the arm/face occluder proxy,
         and non-skin person pixels approximate the garment region.

      4. Sobel-edge stub  — no model.
    
    Occlusion Categories Detected:
      - body_parts: arms, hands, hair, face, legs
      - carried_objects: bags, handbags, backpacks, phones
      - accessories: scarves, belts, sunglasses, hats, jewelry
      - environment: other people, furniture, plants, vehicles
    """

    # ═══════════════════════════════════════════════════════════════════════
    # Segformer class IDs (18-class human parsing)
    # ═══════════════════════════════════════════════════════════════════════
    _SF_GARMENT     = {4, 5, 6, 7}      # upper-clothes, skirt, pants, dress
    _SF_ARMS        = {14, 15}          # left-arm, right-arm
    _SF_LEGS        = {12, 13}          # left-leg, right-leg
    _SF_HAIR        = {2}               # hair
    _SF_FACE        = {11}              # face
    _SF_ACCESSORIES = {1, 3, 8, 16, 17} # hat, sunglasses, belt, bag, scarf
    _SF_SHOES       = {9, 10}           # left-shoe, right-shoe
    
    # ═══════════════════════════════════════════════════════════════════════
    # COCO Panoptic class IDs (for Mask2Former)
    # ═══════════════════════════════════════════════════════════════════════
    # Person and body parts
    _COCO_PERSON = {0}  # person class
    
    # Carried objects that can occlude garments
    _COCO_BAGS = {
        26,   # handbag
        27,   # tie (can cover shirt)
        31,   # backpack
        32,   # umbrella (when held)
        73,   # laptop (held)
        74,   # mouse (unlikely but included)
        76,   # keyboard (unlikely)
        77,   # cell phone (held in front)
        78,   # microwave (unlikely)
        84,   # book (held)
        85,   # clock (unlikely)
        86,   # vase (held)
        87,   # scissors (held)
    }
    
    # Accessories that can occlude
    _COCO_ACCESSORIES = {
        27,   # tie
        32,   # umbrella
    }
    
    # Other people (environmental occlusion)
    _COCO_OTHER_PERSON = {0}  # same class, but different instance
    
    # Furniture and environmental objects
    _COCO_FURNITURE = {
        56,   # chair
        57,   # couch
        59,   # bed
        60,   # dining table
        61,   # toilet
        62,   # tv
        69,   # oven
        71,   # sink
        72,   # refrigerator
    }
    
    # Vehicles (partial occlusion in street scenes)
    _COCO_VEHICLES = {
        1,    # bicycle
        2,    # car
        3,    # motorcycle
        4,    # airplane
        5,    # bus
        6,    # train
        7,    # truck
        8,    # boat
    }
    
    # Plants and natural elements
    _COCO_PLANTS = {
        58,   # potted plant
    }
    
    # Sports equipment (can occlude)
    _COCO_SPORTS = {
        33,   # skis
        34,   # snowboard
        35,   # sports ball
        36,   # kite
        37,   # baseball bat
        38,   # baseball glove
        39,   # skateboard
        40,   # surfboard
        41,   # tennis racket
    }

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._backend   = "stub"
        self._model     = None
        self._processor = None
        self._dl_model  = None
        self._mask2former_model = None
        self._mask2former_processor = None
        self._object_detector = None
        self._load()

    # --------------------------------------------------------------------- #
    def _load(self):
        # 1) Try Mask2Former (COCO panoptic - 133 classes, best for general objects)
        try:
            from transformers import (Mask2FormerForUniversalSegmentation,
                                       Mask2FormerImageProcessor)
            self._mask2former_processor = Mask2FormerImageProcessor.from_pretrained(
                "facebook/mask2former-swin-large-coco-panoptic"
            )
            self._mask2former_model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-coco-panoptic"
            ).to(self.device).eval()
            self._backend = "mask2former"
            print("[OcclusionMetric] Using Mask2Former (COCO panoptic) "
                  "for comprehensive object segmentation.")
            return
        except Exception as e:
            print(f"[OcclusionMetric] Mask2Former unavailable ({e}). "
                  "Trying Segformer...")
        
        # 2) Try HuggingFace Segformer (18-class human parsing)
        try:
            from transformers import (SegformerImageProcessor,
                                       SegformerForSemanticSegmentation)
            self._processor = SegformerImageProcessor.from_pretrained(
                "mattmdjaga/segformer_b2_clothes"
            )
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                "mattmdjaga/segformer_b2_clothes"
            ).to(self.device).eval()
            
            # Also try to load object detector for bags/accessories
            self._try_load_object_detector()
            
            self._backend = "segformer"
            print("[OcclusionMetric] Using Segformer-B2 (human parsing) "
                  "for segmentation.")
            return
        except Exception as e:
            print(f"[OcclusionMetric] Segformer unavailable ({e}). "
                  "Falling back to DeepLabV3 + skin-colour proxy.")

        # 3) Try DeepLabV3 (torchvision) + skin-colour proxy + object detection
        try:
            import torchvision.models.segmentation as seg_models
            self._dl_model = seg_models.deeplabv3_resnet101(
                weights=seg_models.DeepLabV3_ResNet101_Weights.DEFAULT
            ).to(self.device).eval()
            
            # Also try to load object detector
            self._try_load_object_detector()
            
            self._backend = "deeplabv3_skin"
            print("[OcclusionMetric] Using DeepLabV3 + skin-colour proxy "
                  "for segmentation.")
            return
        except Exception as e:
            print(f"[OcclusionMetric] DeepLabV3 unavailable ({e}). "
                  "Falling back to saliency proxy.")

        self._backend = "stub"
    
    # --------------------------------------------------------------------- #
    def _try_load_object_detector(self):
        """Try to load an object detector for carried objects, other people, etc."""
        # Try DETR for object detection
        try:
            from transformers import DetrForObjectDetection, DetrImageProcessor
            self._object_detector = {
                "model": DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50"
                ).to(self.device).eval(),
                "processor": DetrImageProcessor.from_pretrained(
                    "facebook/detr-resnet-50"
                ),
                "type": "detr"
            }
            print("[OcclusionMetric] + DETR object detector for bags/people/objects.")
            return
        except Exception:
            pass
        
        # Try YOLOv5 via ultralytics
        try:
            from ultralytics import YOLO
            self._object_detector = {
                "model": YOLO("yolov8n.pt"),
                "type": "yolo"
            }
            print("[OcclusionMetric] + YOLOv8 object detector for bags/people/objects.")
            return
        except Exception:
            pass
        
        print("[OcclusionMetric] No object detector available. "
              "Environmental occlusion detection limited.")

    # --------------------------------------------------------------------- #
    def _dilate_mask(self, mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        Dilate a mask to simulate overlap with adjacent regions.
        mask: (B, H, W) float32 or bool
        Returns: (B, H, W) float32
        """
        if mask.dtype == torch.bool:
            mask = mask.float()
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        
        # Create circular dilation kernel
        k = kernel_size
        kernel = torch.zeros(1, 1, k, k, device=mask.device)
        center = k // 2
        for i in range(k):
            for j in range(k):
                if (i - center) ** 2 + (j - center) ** 2 <= center ** 2:
                    kernel[0, 0, i, j] = 1.0
        
        # Apply dilation via max pooling approximation
        pad = k // 2
        dilated = F.conv2d(mask, kernel, padding=pad)
        dilated = (dilated > 0).float()
        
        return dilated.squeeze(1)  # (B, H, W)

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def segment(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        imgs : (B, 3, H, W)  float32  [0,1]
        Returns dict of boolean masks (B, H, W):
            "garment"      - target garment region
            "body_parts"   - arms, hands, hair, face, legs
            "accessories"  - bags, scarves, belts, sunglasses, hats
            "carried_obj"  - handbags, backpacks, phones, etc.
            "environment"  - other people, furniture, plants
            "other"        - anything else not background
            
            Legacy keys (for backward compatibility):
            "arms", "hair", "other"
        """
        B, C, H, W = imgs.shape

        if self._backend == "mask2former":
            return self._mask2former_masks(imgs, H, W)
        if self._backend == "segformer":
            return self._segformer_masks(imgs, H, W)
        if self._backend == "deeplabv3_skin":
            return self._deeplabv3_skin_masks(imgs, H, W)
        return self._stub_masks(imgs, H, W)
    
    # --------------------------------------------------------------------- #
    def _mask2former_masks(self, imgs: torch.Tensor, H: int, W: int):
        """
        Run Mask2Former (COCO panoptic) for comprehensive object segmentation.
        Detects 133 object classes including people, bags, furniture, vehicles.
        """
        B = imgs.shape[0]
        
        # Initialize output masks
        garment = torch.zeros((B, H, W), dtype=torch.bool)
        body_parts = torch.zeros((B, H, W), dtype=torch.bool)
        accessories = torch.zeros((B, H, W), dtype=torch.bool)
        carried_obj = torch.zeros((B, H, W), dtype=torch.bool)
        environment = torch.zeros((B, H, W), dtype=torch.bool)
        other_mask = torch.zeros((B, H, W), dtype=torch.bool)
        
        # Process each image
        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()).convert("RGB") for img in imgs]
        
        for i, pil in enumerate(pils):
            inputs = self._mask2former_processor(images=pil, return_tensors="pt").to(self.device)
            outputs = self._mask2former_model(**inputs)
            
            # Post-process to get panoptic segmentation
            result = self._mask2former_processor.post_process_panoptic_segmentation(
                outputs, target_sizes=[(H, W)]
            )[0]
            
            seg_map = result["segmentation"].cpu()  # (H, W) with segment IDs
            segments = result["segments_info"]
            
            # Map segments to occlusion categories
            for seg_info in segments:
                seg_id = seg_info["id"]
                label_id = seg_info["label_id"]
                mask = (seg_map == seg_id)
                
                # Classify by COCO category
                if label_id in self._COCO_BAGS:
                    carried_obj[i] |= mask
                elif label_id in self._COCO_FURNITURE:
                    environment[i] |= mask
                elif label_id in self._COCO_VEHICLES:
                    environment[i] |= mask
                elif label_id in self._COCO_PLANTS:
                    environment[i] |= mask
                elif label_id in self._COCO_SPORTS:
                    carried_obj[i] |= mask
                elif label_id == 0:  # person
                    # This could be another person or the main subject
                    # For now, count as potential environment occlusion
                    # (main person detection is handled separately)
                    pass
                else:
                    other_mask[i] |= mask
        
        # For body parts and garment, we still need human parsing
        # Run Segformer if available for detailed body part segmentation
        if self._model is not None:
            sf_masks = self._segformer_masks_internal(imgs, H, W)
            garment = sf_masks["garment"]
            body_parts = sf_masks["body_parts"]
            accessories |= sf_masks["accessories"]
        
        # Legacy compatibility
        arms = body_parts.clone()
        hair = torch.zeros((B, H, W), dtype=torch.bool)
        
        # Combine all non-garment as "other" for legacy
        other_combined = body_parts | accessories | carried_obj | environment | other_mask
        
        return {
            "garment": garment,
            "body_parts": body_parts,
            "accessories": accessories,
            "carried_obj": carried_obj,
            "environment": environment,
            "other": other_combined,
            # Legacy keys
            "arms": arms,
            "hair": hair,
        }

    # --------------------------------------------------------------------- #
    def _segformer_masks_internal(self, imgs: torch.Tensor, H: int, W: int):
        """Internal helper for Segformer segmentation (used by Mask2Former too)."""
        pils = []
        for img in imgs:
            pil = TF.to_pil_image(img.clamp(0, 1).cpu()).convert("RGB")
            if pil.width < 32 or pil.height < 32:
                pil = pil.resize((224, 224))
            pils.append(pil)
        
        inputs = self._processor(
            images=pils,
            return_tensors="pt",
            input_data_format="channels_last",
        ).to(self.device)
        logits = self._model(**inputs).logits
        logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred = logits_up.argmax(dim=1).cpu()
        
        B = pred.shape[0]
        garment = torch.zeros(pred.shape, dtype=torch.bool)
        body_parts = torch.zeros(pred.shape, dtype=torch.bool)
        accessories = torch.zeros(pred.shape, dtype=torch.bool)
        
        for cls in self._SF_GARMENT:
            garment |= (pred == cls)
        for cls in self._SF_ARMS | self._SF_LEGS | self._SF_HAIR | self._SF_FACE:
            body_parts |= (pred == cls)
        for cls in self._SF_ACCESSORIES:
            accessories |= (pred == cls)
        
        # Dilate body parts
        k = max(3, int(H * 0.02))
        k = k if k % 2 == 1 else k + 1
        bp_f = body_parts.float().unsqueeze(1)
        body_parts = (F.max_pool2d(bp_f, kernel_size=k, stride=1, padding=k // 2) > 0.5).squeeze(1)
        
        return {"garment": garment, "body_parts": body_parts, "accessories": accessories}

    # --------------------------------------------------------------------- #
    def _segformer_masks(self, imgs: torch.Tensor, H: int, W: int):
        """
        Run Segformer-B2-clothes for comprehensive human parsing.
        
        Segformer classes:
          0=BG, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes, 5=Skirt,
          6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe, 11=Face,
          12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm, 16=Bag, 17=Scarf
        
        Returns comprehensive occlusion categories + legacy keys.
        """
        pils = []
        for img in imgs:
            pil = TF.to_pil_image(img.clamp(0, 1).cpu()).convert("RGB")
            if pil.width < 32 or pil.height < 32:
                pil = pil.resize((224, 224))
            pils.append(pil)
        inputs = self._processor(
            images=pils,
            return_tensors="pt",
            input_data_format="channels_last",
        ).to(self.device)
        logits = self._model(**inputs).logits        # (B, C, h', w')

        # Upsample to original size
        logits_up = F.interpolate(logits, size=(H, W),
                                  mode="bilinear", align_corners=False)
        pred = logits_up.argmax(dim=1).cpu()         # (B, H, W)
        B = pred.shape[0]

        # Initialize all masks
        garment = torch.zeros(pred.shape, dtype=torch.bool)
        arms = torch.zeros(pred.shape, dtype=torch.bool)
        legs = torch.zeros(pred.shape, dtype=torch.bool)
        hair = torch.zeros(pred.shape, dtype=torch.bool)
        face = torch.zeros(pred.shape, dtype=torch.bool)
        accessories = torch.zeros(pred.shape, dtype=torch.bool)  # bag, scarf, belt, hat, sunglasses

        # Map classes to masks
        for cls in self._SF_GARMENT:
            garment |= (pred == cls)
        for cls in self._SF_ARMS:
            arms |= (pred == cls)
        for cls in self._SF_LEGS:
            legs |= (pred == cls)
        for cls in self._SF_HAIR:
            hair |= (pred == cls)
        for cls in self._SF_FACE:
            face |= (pred == cls)
        for cls in self._SF_ACCESSORIES:
            accessories |= (pred == cls)

        # Dilate body parts by ~2% of image height for overlap
        k = max(3, int(H * 0.02))
        k = k if k % 2 == 1 else k + 1
        
        def dilate_mask(mask):
            m_f = mask.float().unsqueeze(1)
            return (F.max_pool2d(m_f, kernel_size=k, stride=1, padding=k // 2) > 0.5).squeeze(1)
        
        arms_dilated = dilate_mask(arms)
        legs_dilated = dilate_mask(legs)
        hair_dilated = dilate_mask(hair)
        face_dilated = dilate_mask(face)
        accessories_dilated = dilate_mask(accessories)
        
        # Combine body parts
        body_parts = arms_dilated | legs_dilated | hair_dilated | face_dilated
        
        # Detect objects via object detector if available
        carried_obj = torch.zeros(pred.shape, dtype=torch.bool)
        environment = torch.zeros(pred.shape, dtype=torch.bool)
        
        if self._object_detector is not None:
            obj_masks = self._detect_objects(imgs, H, W)
            carried_obj = obj_masks.get("carried_obj", carried_obj)
            environment = obj_masks.get("environment", environment)

        # "Other" = anything not background, garment, or identified occluders
        other = (~(garment | body_parts | accessories_dilated | carried_obj | environment |
                   (pred == 0)))

        return {
            # Comprehensive categories
            "garment": garment,
            "body_parts": body_parts,
            "accessories": accessories_dilated,
            "carried_obj": carried_obj,
            "environment": environment,
            "other": other,
            # Legacy keys for backward compatibility
            "arms": arms_dilated,
            "hair": hair_dilated,
        }
    
    # --------------------------------------------------------------------- #
    def _detect_objects(self, imgs: torch.Tensor, H: int, W: int) -> Dict[str, torch.Tensor]:
        """
        Use object detector to find carried objects, other people, furniture.
        """
        B = imgs.shape[0]
        carried_obj = torch.zeros((B, H, W), dtype=torch.bool)
        environment = torch.zeros((B, H, W), dtype=torch.bool)
        
        if self._object_detector is None:
            return {"carried_obj": carried_obj, "environment": environment}
        
        pils = [TF.to_pil_image(img.clamp(0, 1).cpu()).convert("RGB") for img in imgs]
        
        if self._object_detector["type"] == "detr":
            processor = self._object_detector["processor"]
            model = self._object_detector["model"]
            
            # COCO classes for carried objects
            # 27=handbag, 31=backpack, 77=cell phone, 73=laptop
            carried_classes = {26, 27, 31, 73, 77, 84}  # adjusted for 0-indexed
            # Furniture and environment
            env_classes = {0, 56, 57, 59, 60, 62}  # person, chair, couch, bed, table, tv
            
            for i, pil in enumerate(pils):
                inputs = processor(images=pil, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Post-process
                target_size = torch.tensor([[H, W]])
                results = processor.post_process_object_detection(
                    outputs, target_sizes=target_size, threshold=0.7
                )[0]
                
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    label_id = label.item()
                    x1, y1, x2, y2 = box.int().tolist()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    
                    if label_id in carried_classes:
                        carried_obj[i, y1:y2, x1:x2] = True
                    elif label_id in env_classes and label_id != 0:
                        # Don't count main person as environment
                        environment[i, y1:y2, x1:x2] = True
        
        elif self._object_detector["type"] == "yolo":
            model = self._object_detector["model"]
            
            # YOLO COCO classes
            carried_classes = {24, 26, 28, 63, 67, 73}  # backpack, handbag, suitcase, laptop, phone, book
            env_classes = {0, 56, 57, 59, 60, 62}  # person, chair, couch, bed, table, tv
            
            for i, pil in enumerate(pils):
                results = model(pil, verbose=False)[0]
                
                for box in results.boxes:
                    cls = int(box.cls.item())
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    
                    if cls in carried_classes:
                        carried_obj[i, y1:y2, x1:x2] = True
                    elif cls in env_classes and cls != 0:
                        environment[i, y1:y2, x1:x2] = True
        
        return {"carried_obj": carried_obj, "environment": environment}

    # --------------------------------------------------------------------- #
    def _deeplabv3_skin_masks(self, imgs: torch.Tensor, H: int, W: int):
        """
        DeepLabV3 (Pascal VOC classes) + YCbCr skin detection + optional
        object detection for comprehensive occlusion coverage.

        Pascal VOC class mapping:
            0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle,
            6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=dining table, 12=dog,
            13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep,
            18=sofa, 19=train, 20=tv/monitor

        YCbCr skin thresholds (standard Chai & Ngan 1999):
            77 ≤ Y ≤ 235,  133 ≤ Cb ≤ 173,  77 ≤ Cr ≤ 127   (0-255 range)
        """
        norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        x   = torch.stack([norm(im) for im in imgs]).to(self.device)
        out = self._dl_model(x)["out"]               # (B, 21, H, W)
        pred = out.argmax(1)                          # (B, H, W)
        pred_cpu = pred.cpu()

        B = imgs.shape[0]

        # ── Person mask ───────────────────────────────────────────────────────
        person_mask = (pred_cpu == 15)                # (B, H, W) bool

        # ── YCbCr skin colour detection ───────────────────────────────────────
        imgs_cpu = imgs.cpu()
        R = imgs_cpu[:, 0]
        G = imgs_cpu[:, 1]
        B_ch = imgs_cpu[:, 2]

        # RGB → YCbCr  (BT.601, output in [0,255] range)
        Y_lum = 16 + 65.481 * R + 128.553 * G + 24.966 * B_ch
        Cb = 128 - 37.797 * R - 74.203 * G + 112.000 * B_ch
        Cr = 128 + 112.000 * R - 93.786 * G - 18.214 * B_ch

        skin = (
            (Y_lum >= 77)  & (Y_lum <= 235) &
            (Cb >= 133) & (Cb <= 173) &
            (Cr >= 77)  & (Cr <= 127)
        )                                             # (B, H, W) bool

        # ── Basic body parts ──────────────────────────────────────────────────
        # Garment = person & NOT skin  (clothing is not skin-coloured)
        # Arms    = person & skin      (exposed arms/face are skin-coloured)
        garment = person_mask & ~skin
        arms    = person_mask & skin

        # Hair approximation: top 15% of person region with darker pixels
        hair = torch.zeros(B, H, W, dtype=torch.bool)
        for i in range(B):
            pm = person_mask[i]
            if pm.any():
                rows = pm.any(dim=1).nonzero(as_tuple=True)[0]
                if len(rows) > 0:
                    top_row = rows.min().item()
                    hair_height = int(H * 0.15)
                    hair_region = torch.zeros_like(pm)
                    hair_region[top_row:min(top_row + hair_height, H), :] = True
                    # Darker pixels in top region (not skin)
                    luminance = (R[i] * 0.299 + G[i] * 0.587 + B_ch[i] * 0.114)
                    dark_pixels = luminance < 0.4
                    hair[i] = pm & hair_region & dark_pixels & ~skin[i]

        body_parts = arms | hair  # Union of arms + hair

        # Dilate body parts for potential overlap with garment
        body_dilated = self._dilate_mask(body_parts.float(), kernel_size=7).bool()

        # ── Pascal VOC environment classes ────────────────────────────────────
        VOC_FURNITURE = {9, 11, 18, 20}  # chair, dining table, sofa, tv/monitor
        VOC_VEHICLES  = {1, 2, 4, 6, 7, 14, 19}  # aeroplane, bicycle, boat, bus, car, motorbike, train
        VOC_PLANTS    = {16}  # potted plant
        VOC_ENV = VOC_FURNITURE | VOC_VEHICLES | VOC_PLANTS

        environment = torch.zeros(B, H, W, dtype=torch.bool)
        for cls_id in VOC_ENV:
            environment = environment | (pred_cpu == cls_id)

        # Other people (separate person instances are hard to distinguish in semantic seg)
        # We can only detect if there's person mask far from the center
        other_people = torch.zeros(B, H, W, dtype=torch.bool)

        # ── Carried objects & accessories via object detector ─────────────────
        carried_obj = torch.zeros(B, H, W, dtype=torch.bool)
        accessories = torch.zeros(B, H, W, dtype=torch.bool)

        if self._object_detector is not None:
            det_masks = self._detect_objects(imgs, H, W)
            carried_obj = det_masks.get("carried_obj", carried_obj)
            environment = environment | det_masks.get("environment", torch.zeros_like(environment))
            other_people = det_masks.get("other_people", other_people)

        # ── Other: anything segmented but not person or known environment ─────
        other = (~person_mask) & (pred_cpu != 0)
        for cls_id in VOC_ENV:
            other = other & (pred_cpu != cls_id)

        return {
            # New comprehensive keys
            "garment":      garment,
            "body_parts":   body_dilated,
            "accessories":  accessories,
            "carried_obj":  carried_obj,
            "environment":  environment,
            "other_people": other_people,
            "other":        other,
            # Legacy keys for backward compatibility
            "arms":         arms,
            "hair":         hair,
        }

    # --------------------------------------------------------------------- #
    def _stub_masks(self, imgs: torch.Tensor, H: int, W: int):
        """
        Saliency-based proxy fallback: high-gradient regions ≈ garment boundary.
        Returns comprehensive categories but with limited accuracy (no real segmentation).
        """
        gray = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]
        B = gray.shape[0]
        gray4 = gray.unsqueeze(1)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=imgs.device).view(1, 1, 3, 3)
        sy = sx.transpose(-2, -1)
        gx = F.conv2d(gray4, sx, padding=1)
        gy = F.conv2d(gray4, sy, padding=1)
        edge = (gx ** 2 + gy ** 2).sqrt().squeeze(1)

        thr     = edge.flatten(1).median(1).values[:, None, None] * 1.5
        garment = (edge > thr)

        # Heuristic split: top half = body parts, bottom half = garment
        h_split = H // 2
        arms    = torch.zeros_like(garment)
        arms[:, :h_split, :] = garment[:, :h_split, :]
        garment_lower = garment.clone()
        garment_lower[:, :h_split, :] = False

        # Empty placeholders for comprehensive categories
        zeros = torch.zeros(B, H, W, dtype=torch.bool)

        return {
            # New comprehensive keys
            "garment":      garment_lower.cpu(),
            "body_parts":   arms.cpu(),
            "accessories":  zeros.clone(),
            "carried_obj":  zeros.clone(),
            "environment":  zeros.clone(),
            "other_people": zeros.clone(),
            "other":        zeros.clone(),
            # Legacy keys for backward compatibility
            "arms":         arms.cpu(),
            "hair":         zeros.clone(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# OcclusionMetrics
# ─────────────────────────────────────────────────────────────────────────────

class OcclusionMetrics:
    """
    Accumulates per-image occlusion ratios across multiple occlusion sources,
    then computes comprehensive C_occ with per-category breakdown.

    Occlusion Categories:
    ─────────────────────
    1. Body parts (arms, legs, face, hair) - self-occlusion from pose
    2. Carried objects (handbags, backpacks, phones, bags)
    3. Accessories (sunglasses, jewelry, scarves, hats)
    4. Environmental (furniture, vehicles, plants, architecture)
    5. Other people (background persons, crowds)
    6. Other (unknown occluding objects)
    """

    def __init__(self, device: str = "cpu"):
        self._seg = _SegBackend(device=device)
        # Per-category ratio accumulators
        self._ratios_body:     List[float] = []
        self._ratios_carried:  List[float] = []
        self._ratios_accessory:List[float] = []
        self._ratios_env:      List[float] = []
        self._ratios_people:   List[float] = []
        self._ratios_other:    List[float] = []
        self._ratios_total:    List[float] = []  # union of all

    # ------------------------------------------------------------------ #
    def update(self, person_imgs: torch.Tensor):
        """
        person_imgs : (B, 3, H, W)  float32  [0,1]

        Computes occlusion ratio per category for each image.
        """
        masks = self._seg.segment(person_imgs)

        G = masks["garment"].float()  # (B, H, W)

        # Comprehensive occlusion sources
        body      = masks.get("body_parts",   masks.get("arms", torch.zeros_like(G))).float()
        carried   = masks.get("carried_obj",  torch.zeros_like(G)).float()
        accessory = masks.get("accessories",  torch.zeros_like(G)).float()
        env       = masks.get("environment",  torch.zeros_like(G)).float()
        people    = masks.get("other_people", torch.zeros_like(G)).float()
        other     = masks.get("other",        torch.zeros_like(G)).float()

        # Legacy support: if new keys missing, fall back to old keys
        if body.sum() == 0 and "arms" in masks:
            arms = masks["arms"].float()
            hair = masks.get("hair", torch.zeros_like(G)).float()
            body = ((arms + hair) > 0).float()

        # Total occluder: union of all categories
        all_occluders = ((body + carried + accessory + env + people + other) > 0).float()

        B = G.shape[0]
        for i in range(B):
            g_area = G[i].sum().item()
            if g_area < 1:
                # No garment detected
                self._ratios_body.append(0.0)
                self._ratios_carried.append(0.0)
                self._ratios_accessory.append(0.0)
                self._ratios_env.append(0.0)
                self._ratios_people.append(0.0)
                self._ratios_other.append(0.0)
                self._ratios_total.append(0.0)
                continue

            # Per-category overlap with garment
            self._ratios_body.append(float(min((G[i] * body[i]).sum().item() / g_area, 1.0)))
            self._ratios_carried.append(float(min((G[i] * carried[i]).sum().item() / g_area, 1.0)))
            self._ratios_accessory.append(float(min((G[i] * accessory[i]).sum().item() / g_area, 1.0)))
            self._ratios_env.append(float(min((G[i] * env[i]).sum().item() / g_area, 1.0)))
            self._ratios_people.append(float(min((G[i] * people[i]).sum().item() / g_area, 1.0)))
            self._ratios_other.append(float(min((G[i] * other[i]).sum().item() / g_area, 1.0)))

            # Total occlusion (union, not sum)
            total_overlap = (G[i] * all_occluders[i]).sum().item()
            self._ratios_total.append(float(min(total_overlap / g_area, 1.0)))

    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        """
        Returns comprehensive occlusion statistics:
        - Per-category mean occlusion ratios
        - Total occlusion mean/variance/complexity (C_occ)
        """
        if not self._ratios_total:
            return {
                # Legacy keys
                "occlusion_mean":              float("nan"),
                "occlusion_var":               float("nan"),
                "occlusion_complexity":        float("nan"),
                # Per-category breakdown
                "occlusion_body_parts":        float("nan"),
                "occlusion_carried_objects":   float("nan"),
                "occlusion_accessories":       float("nan"),
                "occlusion_environment":       float("nan"),
                "occlusion_other_people":      float("nan"),
                "occlusion_other":             float("nan"),
            }

        # Per-category means
        body_mean      = float(np.array(self._ratios_body).mean())
        carried_mean   = float(np.array(self._ratios_carried).mean())
        accessory_mean = float(np.array(self._ratios_accessory).mean())
        env_mean       = float(np.array(self._ratios_env).mean())
        people_mean    = float(np.array(self._ratios_people).mean())
        other_mean     = float(np.array(self._ratios_other).mean())

        # Total statistics (backward compatible)
        arr_total = np.array(self._ratios_total)
        total_mean = float(arr_total.mean())
        total_var  = float(arr_total.var())

        return {
            # Legacy keys (backward compatible)
            "occlusion_mean":              total_mean,
            "occlusion_var":               total_var,
            "occlusion_complexity":        total_mean + total_var,
            # Per-category breakdown
            "occlusion_body_parts":        body_mean,
            "occlusion_carried_objects":   carried_mean,
            "occlusion_accessories":       accessory_mean,
            "occlusion_environment":       env_mean,
            "occlusion_other_people":      people_mean,
            "occlusion_other":             other_mean,
        }

    def reset(self):
        self._ratios_body.clear()
        self._ratios_carried.clear()
        self._ratios_accessory.clear()
        self._ratios_env.clear()
        self._ratios_people.clear()
        self._ratios_other.clear()
        self._ratios_total.clear()