"""
metrics/vlm_score.py
=====================
VLM Plausibility Score for Virtual Try-On (Continuous 0-1 Scale).

Evaluates virtual try-on images using a Vision-Language Model with a 
comprehensive prompt that assesses multiple aspects of realism:

  1. Photorealism - Fabric texture, wrinkles, shading naturalness
  2. Lighting Consistency - Matching scene lighting, shadows, highlights
  3. Color/Intensity Matching - Exposure and brightness consistency
  4. Seamless Blending - No cut-and-paste artifacts, halos, sharp edges
  5. Body Alignment - Clothing follows body pose and geometry
  6. Occlusion Handling - Correct interaction with arms, hair, objects
  7. Global Scene Consistency - Result looks captured in same photograph

Output
------
  vlm_score : float ∈ [0, 1]  (continuous, higher = better)
  reason    : str             (brief explanation of score)

Score Interpretation:
  1.0       : Perfect photorealistic try-on
  0.8-0.99  : Very realistic with minor imperfections
  0.6-0.79  : Generally believable but noticeable artifacts
  0.4-0.59  : Clearly synthetic in some areas
  0.2-0.39  : Strong artifacts, incorrect lighting
  0.0-0.19  : Completely unrealistic

Backend
-------
Primary  : BLIP-2 (Salesforce/blip2-opt-2.7b via HuggingFace Transformers)
Fallback : InstructBLIP if BLIP-2 fails to load
Stub     : neutral 0.5 if no VLM can be loaded

Usage
-----
    from metrics.vlm_score import VLMScoreMetric

    metric = VLMScoreMetric(device="cuda")

    # evaluate.py per-batch call:
    detailed = metric.compute_batch(pred_tensor)
    # Returns list[dict]:
    # [{'vlm_score': 0.85, 'reason': 'Good lighting match...'}, ...]

    # Back-compat: scalar list
    scalars = metric.compute_batch_scalar(pred_tensor)
    # Returns list[float]: [0.85, ...]
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

import torch
from PIL import Image
import torchvision.transforms.functional as TF


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt for VLM Evaluation
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert evaluator for virtual try-on systems. Your task is to judge whether a generated try-on image looks realistic and naturally integrated with the original in-the-wild photograph.

Inputs you will receive:
1. An in-the-wild image of a person.
2. A generated virtual try-on image where a garment has been applied to the person.

Your task is to evaluate how realistic and well-integrated the try-on result is.

Focus on the following aspects:

1. Photorealism
- The clothing should look like a real garment, not cartoon-like, synthetic, or painted.
- Fabric texture, wrinkles, and shading should appear natural.

2. Lighting Consistency
- The garment lighting should match the scene lighting.
- Shadows, highlights, and shading should align with the direction and intensity of light in the image.

3. Color and Intensity Matching
- The garment colors and brightness should match the overall exposure of the image.
- No unnatural brightness, saturation, or glow.

4. Seamless Blending
- The garment should integrate naturally with the body.
- No visible cut-and-paste artifacts, halos, or sharp edges.

5. Body Alignment and Shape
- The clothing should follow the body pose and geometry.
- No unrealistic stretching, floating cloth, or broken structure.

6. Occlusion Handling
- Correct interaction with arms, hair, or objects.
- Cloth should appear partially hidden where appropriate.

7. Global Scene Consistency
- The try-on should look like it was captured in the same photograph.
- The result should not look like an overlay or edited region.

Scoring:

Assign a continuous score between 0.0 and 1.0.

1.0
Perfect photorealistic try-on. Seamlessly integrated with correct lighting, shading, and body alignment.

0.8 – 0.99
Very realistic with only minor blending or lighting imperfections.

0.6 – 0.79
Generally believable but noticeable artifacts or slight lighting mismatch.

0.4 – 0.59
Clearly synthetic in some areas or poorly blended.

0.2 – 0.39
Strong artifacts, incorrect lighting, or unrealistic garment behavior.

0.0 – 0.19
Completely unrealistic, cartoon-like, or obvious overlay.

Important rules:
- Focus only on visual realism and integration of the try-on garment.
- Do not judge clothing style or aesthetics.
- Be consistent across different poses, backgrounds, ethnicities, and clothing types.

Output format:
Return ONLY a JSON object:

{
  "score": <float between 0 and 1>,
  "reason": "<short explanation mentioning lighting, blending, realism, or artifacts>"
}"""

_EVAL_PROMPT = "Evaluate this virtual try-on image for realism. Return JSON with score (0-1) and reason."


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Parse JSON response from VLM
# ─────────────────────────────────────────────────────────────────────────────

def _parse_vlm_response(text: str, fallback_score: float = 0.5) -> Dict[str, any]:
    """
    Parse VLM JSON response to extract score and reason.
    
    Args:
        text: Raw VLM output text
        fallback_score: Default score if parsing fails
    
    Returns:
        Dict with 'vlm_score' and 'reason' keys
    """
    # Try to extract JSON from response
    try:
        # Look for JSON pattern in text
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            score = float(data.get("score", fallback_score))
            reason = str(data.get("reason", ""))
            # Clamp score to [0, 1]
            score = max(0.0, min(1.0, score))
            return {"vlm_score": score, "reason": reason}
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    
    # Fallback: try to extract any float from text
    nums = re.findall(r"0?\.\d+|\d+\.?\d*", text)
    for num_str in nums:
        try:
            num = float(num_str)
            if 0 <= num <= 1:
                return {"vlm_score": num, "reason": "Score extracted from text"}
            elif 1 < num <= 10:
                # Convert 1-10 scale to 0-1
                return {"vlm_score": (num - 1) / 9, "reason": "Score converted from 1-10 scale"}
        except ValueError:
            continue
    
    return {"vlm_score": fallback_score, "reason": "Failed to parse VLM response"}


# ─────────────────────────────────────────────────────────────────────────────
# VLMScoreMetric
# ─────────────────────────────────────────────────────────────────────────────

class VLMScoreMetric:
    """
    VLM-based plausibility scorer for virtual try-on.
    
    Outputs a continuous score between 0 and 1 based on comprehensive
    evaluation of photorealism, lighting, blending, and scene consistency.

    Parameters
    ----------
    model_name  : HuggingFace model ID (default: BLIP-2 opt-2.7b).
    device      : 'cpu' | 'cuda' | 'cuda:N'.
    vlm_batch   : Max images per VLM forward pass (reduce if OOM).
    neutral     : Fallback score when VLM unavailable (default 0.5).
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cpu",
        vlm_batch: int = 2,
        neutral: float = 0.5,
        **kwargs,  # Accept legacy kwargs for backward compatibility
    ):
        self.device = device
        self.vlm_batch = vlm_batch
        self.neutral = neutral

        self._model = None
        self._processor = None
        self._backend = "stub"
        self._load_model(model_name)

    # ── Model loading ────────────────────────────────────────────────────── #

    def _load_model(self, model_name: str):
        # ── Try BLIP-2 ────────────────────────────────────────────────────
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            print(f"[VLMScore] Loading BLIP-2: {model_name} …")
            self._processor = Blip2Processor.from_pretrained(model_name)
            self._model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map=self.device,
            )
            self._model.eval()
            self._backend = "blip2"
            print("[VLMScore] BLIP-2 loaded.")
            return
        except Exception as e:
            print(f"[VLMScore] BLIP-2 unavailable ({e}). Trying InstructBLIP …")

        # ── Try InstructBLIP fallback ────────────────────────────────────
        try:
            from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
            fb = "Salesforce/instructblip-vicuna-7b"
            print(f"[VLMScore] Loading InstructBLIP: {fb} …")
            self._processor = InstructBlipProcessor.from_pretrained(fb)
            self._model = InstructBlipForConditionalGeneration.from_pretrained(
                fb,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map=self.device,
            )
            self._model.eval()
            self._backend = "instructblip"
            print("[VLMScore] InstructBLIP loaded.")
            return
        except Exception as e:
            print(f"[VLMScore] InstructBLIP unavailable ({e}). Using stub.")

        self._backend = "stub"

    # ── Internal VLM evaluation ──────────────────────────────────────────── #

    @torch.no_grad()
    def _evaluate_images(self, pil_images: List[Image.Image]) -> List[Dict[str, any]]:
        """
        Run the comprehensive evaluation prompt on a batch of PIL images.
        
        Returns:
            List of dicts with 'vlm_score' (float 0-1) and 'reason' (str)
        """
        if self._backend == "stub":
            return [{"vlm_score": self.neutral, "reason": "VLM stub mode"} for _ in pil_images]

        results = []
        
        try:
            # Construct the full prompt with system context
            full_prompt = f"{_SYSTEM_PROMPT}\n\n{_EVAL_PROMPT}"
            
            inputs = self._processor(
                images=pil_images,
                text=[full_prompt] * len(pil_images),
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            generated = self._model.generate(
                **inputs,
                max_new_tokens=150,  # Allow enough tokens for JSON response
                do_sample=False,
            )
            texts = self._processor.batch_decode(generated, skip_special_tokens=True)
            
            for text in texts:
                result = _parse_vlm_response(text, self.neutral)
                results.append(result)

        except Exception as e:
            print(f"[VLMScore] Inference error ({e}). Using neutral score.")
            results = [{"vlm_score": self.neutral, "reason": f"Inference error: {e}"} for _ in pil_images]

        return results

    # ── Public batch scoring ─────────────────────────────────────────────── #

    def compute_batch(
        self, 
        pred: torch.Tensor,
        person_images: Optional[torch.Tensor] = None,
        cloth_images: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, any]]:
        """
        Score a batch of try-on result images on realism and integration.

        Parameters
        ----------
        pred : torch.Tensor  (B, 3, H, W) float32 in [0, 1]
            The generated try-on images to evaluate.
        person_images : torch.Tensor, optional  (B, 3, H, W)
            Original person images (for context, not directly used in prompt).
        cloth_images : torch.Tensor, optional  (B, 3, H, W)
            Original cloth images (for context, not directly used in prompt).

        Returns
        -------
        list[dict] — one dict per image::

            {
              'vlm_score' : float  continuous score ∈ [0, 1]
              'reason'    : str    brief explanation of score
            }
        """
        B = pred.shape[0]
        pil_list = [TF.to_pil_image(img.clamp(0, 1).cpu()) for img in pred]

        # Process in batches
        results: List[Dict[str, any]] = []
        for start in range(0, B, self.vlm_batch):
            chunk = pil_list[start: start + self.vlm_batch]
            chunk_results = self._evaluate_images(chunk)
            results.extend(chunk_results)

        return results

    def compute_batch_scalar(self, pred: torch.Tensor) -> List[float]:
        """
        Convenience wrapper — returns only the VLM score per image.

        Back-compat alias so evaluate.py can call this with no changes.
        """
        return [r["vlm_score"] for r in self.compute_batch(pred)]

    # ── Describe the scoring ─────────────────────────────────────────────── #

    def describe(self) -> str:
        lines = [
            f"VLMScoreMetric  backend={self._backend}",
            f"  Evaluation Criteria:",
            f"    1. Photorealism (texture, wrinkles, shading)",
            f"    2. Lighting Consistency (shadows, highlights)",
            f"    3. Color/Intensity Matching (exposure, brightness)",
            f"    4. Seamless Blending (no artifacts, halos)",
            f"    5. Body Alignment (pose, geometry)",
            f"    6. Occlusion Handling (arms, hair, objects)",
            f"    7. Global Scene Consistency",
            f"  Output: vlm_score ∈ [0, 1]  (continuous, higher = better)",
        ]
        return "\n".join(lines)
