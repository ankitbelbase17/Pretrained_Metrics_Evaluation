"""
metrics/vlm_score.py
=====================
VLM Plausibility Score for Virtual Try-On (Continuous 0-1 Scale).

Uses Qwen3-VL-32B-Instruct for multi-image evaluation of virtual try-on results.

Evaluates virtual try-on images by comparing:
  - Image A: Original person image (identity ground truth)
  - Image B: Target garment image (garment ground truth)  
  - Image C: Generated try-on result (what we evaluate)

Evaluation Criteria:
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
Primary  : Qwen3-VL-32B-Instruct (with flash_attention_2)
Fallback : Qwen2-VL-7B-Instruct
Stub     : neutral 0.5 if no VLM can be loaded

Usage
-----
    from metrics.vlm_score import VLMScoreMetric

    metric = VLMScoreMetric(device="cuda")

    # With all three images (recommended):
    results = metric.compute_batch(
        tryon_images,
        person_images=person_imgs,
        cloth_images=cloth_imgs,
    )
    # Returns list[dict]: [{'vlm_score': 0.85, 'reason': '...'}, ...]

    # Back-compat: scalar list
    scalars = metric.compute_batch_scalar(tryon_images)
    # Returns list[float]: [0.85, ...]
"""

from __future__ import annotations

import gc
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
1. An in-the-wild image of a person (Image A - Person Ground Truth).
2. A garment image (Image B - Garment Ground Truth).
3. A generated virtual try-on image where the garment has been applied to the person (Image C - Try-On Result).

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
    
    # Try to find a more complex JSON structure
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(text[start:end + 1])
            score = float(data.get("score", fallback_score))
            reason = str(data.get("reason", ""))
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
    Qwen3-VL-based plausibility scorer for virtual try-on.
    
    Uses multi-image input to evaluate try-on quality by comparing:
      - Person image (identity preservation)
      - Garment image (garment fidelity)
      - Try-on result (what we evaluate)
    
    Outputs a continuous score between 0 and 1.

    Parameters
    ----------
    model_name  : HuggingFace model ID (default: Qwen3-VL-32B-Instruct).
    device      : 'cpu' | 'cuda' | 'cuda:N'.
    vlm_batch   : Max images per VLM forward pass (reduce if OOM).
    neutral     : Fallback score when VLM unavailable (default 0.5).
    max_new_tokens : Max tokens for generation (default 300).
    image_size  : Size to resize images to (default 512).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
        device: str = "cuda",
        vlm_batch: int = 1,
        neutral: float = 0.5,
        max_new_tokens: int = 300,
        image_size: int = 512,
        **kwargs,  # Accept legacy kwargs for backward compatibility
    ):
        self.device = device
        self.vlm_batch = vlm_batch
        self.neutral = neutral
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size

        self._model = None
        self._processor = None
        self._backend = "stub"
        self._load_model(model_name)

    # ── Model loading ────────────────────────────────────────────────────── #

    def _load_model(self, model_name: str):
        """Load Qwen3-VL model with fallback chain."""
        
        # ── Try Qwen3-VL-32B-Instruct ─────────────────────────────────────
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            print(f"[VLMScore] Loading Qwen3-VL: {model_name} …")
            
            self._processor = AutoProcessor.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            self._processor.tokenizer.padding_side = "left"
            
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=None,
                trust_remote_code=True,
            )
            self._model.to(self.device)
            self._model.eval()
            self._backend = "qwen3vl"
            print(f"[VLMScore] Qwen3-VL loaded ✓ (flash_attention_2)")
            return
        except Exception as e:
            print(f"[VLMScore] Qwen3-VL unavailable ({e}). Trying Qwen2-VL …")

        # ── Try Qwen2-VL-7B-Instruct fallback ─────────────────────────────
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            fb_model = "Qwen/Qwen2-VL-7B-Instruct"
            print(f"[VLMScore] Loading Qwen2-VL: {fb_model} …")
            
            self._processor = AutoProcessor.from_pretrained(
                fb_model, 
                trust_remote_code=True
            )
            self._processor.tokenizer.padding_side = "left"
            
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                fb_model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=None,
                trust_remote_code=True,
            )
            self._model.to(self.device)
            self._model.eval()
            self._backend = "qwen2vl"
            print(f"[VLMScore] Qwen2-VL loaded ✓ (flash_attention_2)")
            return
        except Exception as e:
            print(f"[VLMScore] Qwen2-VL unavailable ({e}). Using stub.")

        self._backend = "stub"
        print("[VLMScore] WARNING: No VLM loaded. Using stub mode (neutral scores).")

    # ── Image preprocessing ──────────────────────────────────────────────── #

    def _prepare_image(self, img: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image and resize."""
        if img.dim() == 4:
            img = img[0]  # Remove batch dimension
        pil_img = TF.to_pil_image(img.clamp(0, 1).cpu())
        return pil_img.resize((self.image_size, self.image_size), Image.LANCZOS)

    # ── Build conversation for Qwen VL ───────────────────────────────────── #

    def _build_conversation(
        self,
        person_img: Image.Image,
        cloth_img: Image.Image,
        tryon_img: Image.Image,
    ) -> List[Dict]:
        """
        Build the conversation format for Qwen3-VL.
        
        Format follows the reference implementation with system prompt
        and user message containing 3 images.
        """
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image A (Person Ground Truth):"},
                    {"type": "image", "image": person_img},
                    {"type": "text", "text": "\nImage B (Garment Ground Truth):"},
                    {"type": "image", "image": cloth_img},
                    {"type": "text", "text": "\nImage C (Try-On Result):"},
                    {"type": "image", "image": tryon_img},
                    {"type": "text", "text": "\nEvaluate this try-on result against the garment ground truth and identity ground truth. Return JSON with score and reason."},
                ],
            },
        ]
        return messages

    # ── Single evaluation ────────────────────────────────────────────────── #

    @torch.no_grad()
    def _evaluate_single(
        self,
        person_img: Image.Image,
        cloth_img: Image.Image,
        tryon_img: Image.Image,
    ) -> Dict[str, any]:
        """
        Evaluate a single triplet of images.
        
        Returns:
            Dict with 'vlm_score' (float 0-1) and 'reason' (str)
        """
        if self._backend == "stub":
            return {"vlm_score": self.neutral, "reason": "VLM stub mode"}

        try:
            # Build conversation
            messages = self._build_conversation(person_img, cloth_img, tryon_img)
            
            # Apply chat template
            encoded = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            # Move to device
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            pixel_values = encoded["pixel_values"].to(self.device)
            image_grid_thw = encoded["image_grid_thw"].to(self.device)
            
            # Generate
            gen_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                do_sample=False,
            )
            
            # Decode response (only new tokens)
            input_len = input_ids.shape[1]
            response = self._processor.tokenizer.decode(
                gen_ids[0, input_len:], 
                skip_special_tokens=True
            )
            
            # Parse JSON response
            result = _parse_vlm_response(response.strip(), self.neutral)
            
            # Print score to console
            print(f"[VLMScore] score={result['vlm_score']:.3f} | reason={result['reason'][:80]}...")
            
            return result

        except torch.cuda.OutOfMemoryError:
            print("[VLMScore] CUDA OOM — clearing cache and returning neutral score")
            torch.cuda.empty_cache()
            gc.collect()
            return {"vlm_score": self.neutral, "reason": "CUDA OOM"}
        
        except Exception as e:
            print(f"[VLMScore] Inference error: {e}")
            return {"vlm_score": self.neutral, "reason": f"Inference error: {str(e)[:50]}"}

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
            Original person images (identity ground truth).
        cloth_images : torch.Tensor, optional  (B, 3, H, W)
            Original cloth images (garment ground truth).

        Returns
        -------
        list[dict] — one dict per image::

            {
              'vlm_score' : float  continuous score ∈ [0, 1]
              'reason'    : str    brief explanation of score
            }
        """
        B = pred.shape[0]
        results: List[Dict[str, any]] = []
        
        for i in range(B):
            # Prepare images
            tryon_pil = self._prepare_image(pred[i])
            
            # Use provided images or create dummy black images
            if person_images is not None:
                person_pil = self._prepare_image(person_images[i])
            else:
                person_pil = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            
            if cloth_images is not None:
                cloth_pil = self._prepare_image(cloth_images[i])
            else:
                cloth_pil = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            
            # Evaluate
            result = self._evaluate_single(person_pil, cloth_pil, tryon_pil)
            results.append(result)
            
            # Clear cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
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
            f"  Model: Qwen3-VL (multi-image evaluation)",
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
