"""
on_the_wild_evaluation/run_evaluation.py
=========================================
Unified In-the-Wild Evaluation Runner with Triplet Dataloader

Runs all in-the-wild metrics on triplets of (person, cloth, tryon) images.

Metric → Input Mapping & Output Range:
──────────────────────────────────────
  - VLM Score           : person + cloth + tryon (ALL THREE) → [0, 1] continuous
  - Pose Consistency    : person + tryon (pose preservation)
  - NIQE                : tryon ONLY (no-reference quality) → lower is better
  - MUSIQ               : tryon ONLY (no-reference quality) → higher is better
  - JEPA                : tryon ONLY (self-consistency)
  - CLIP Garment        : cloth + tryon (garment fidelity)

Usage:
    # With triplet directories:
    python on_the_wild_evaluation/run_evaluation.py \\
        --person_dir path/to/persons \\
        --cloth_dir path/to/clothes \\
        --tryon_dir path/to/tryons \\
        --output results.json

    # With manifest file:
    python on_the_wild_evaluation/run_evaluation.py \\
        --manifest path/to/manifest.json \\
        --output results.json

    # Single triplet evaluation:
    python on_the_wild_evaluation/run_evaluation.py \\
        --person path/to/person.jpg \\
        --cloth path/to/cloth.jpg \\
        --tryon path/to/tryon.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from on_the_wild_evaluation.vlm_evaluator import VLMEvaluator
from on_the_wild_evaluation.jepa_evaluator import JEPAEvaluator
from on_the_wild_evaluation.pose_evaluator import PoseConsistencyEvaluator
from on_the_wild_evaluation.clip_garment_evaluator import CLIPGarmentEvaluator
from on_the_wild_evaluation.niqe_evaluator import NIQEEvaluator
from on_the_wild_evaluation.musiq_evaluator import MUSIQEvaluator
from on_the_wild_evaluation.dataloaders import TripletDataLoader


def load_image(path: str, size: Tuple[int, int] = (512, 384)) -> torch.Tensor:
    """
    Load and preprocess an image.
    
    Args:
        path: Path to image file
        size: Target size (H, W)
    
    Returns:
        (3, H, W) float32 tensor in [0, 1]
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((size[1], size[0]), Image.LANCZOS)  # PIL uses (W, H)
    return TF.to_tensor(img)


def compute_batch_mean_scores(per_sample_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute mean scores from a batch of per-sample results.
    
    All scores in on_the_wild_evaluation are computed as:
        batch_score = mean(individual_sample_scores)
    
    Args:
        per_sample_results: List of dicts, each containing metric scores for one sample
    
    Returns:
        Dict with mean of each score across all samples
    """
    if not per_sample_results:
        return {}
    
    # Collect all keys
    all_keys = set()
    for r in per_sample_results:
        all_keys.update(r.keys())
    
    # Filter to numeric keys only
    mean_scores = {}
    for key in all_keys:
        values = [r.get(key) for r in per_sample_results if key in r]
        numeric_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
        if numeric_values:
            mean_scores[f"{key}_mean"] = float(np.mean(numeric_values))
    
    return mean_scores


class WildEvaluationSuite:
    """
    Complete in-the-wild evaluation suite for virtual try-on.
    
    Handles triplets: (person_image, cloth_image, tryon_image)
    
    Score Computation:
        All scores are computed per-sample, then aggregated as:
            batch_score = mean(individual_sample_scores)
    
    Metric → Input Mapping:
        VLM Score       : person + cloth + tryon (all three)
        Pose Consistency: person + tryon
        NIQE            : tryon only
        MUSIQ           : tryon only
        JEPA            : tryon only
        CLIP Garment    : cloth + tryon
    """
    
    def __init__(
        self,
        device: str = "cuda",
        metrics: Optional[List[str]] = None,
    ):
        """
        Args:
            device: torch device
            metrics: List of metrics to run. Default: all.
                     Options: ["vlm", "jepa", "pose", "clip", "niqe", "musiq"]
        """
        self.device = device
        self.metrics = metrics or ["vlm", "jepa", "pose", "clip", "niqe", "musiq"]
        
        print("\n" + "=" * 70)
        print("  In-the-Wild Virtual Try-On Evaluation Suite")
        print("=" * 70)
        print(f"  Device: {device}")
        print(f"  Metrics: {', '.join(self.metrics)}")
        print()
        print("  Metric → Input Mapping & Output Range:")
        print("  ─────────────────────────────────────────────────────────────")
        if "vlm" in self.metrics:
            print("    VLM Score        : person + cloth + tryon → [0, 1] continuous")
        if "pose" in self.metrics:
            print("    Pose Consistency : person + tryon")
        if "clip" in self.metrics:
            print("    CLIP Garment     : cloth + tryon")
        if "jepa" in self.metrics:
            print("    JEPA             : tryon ONLY")
        if "niqe" in self.metrics:
            print("    NIQE             : tryon ONLY (lower = better)")
        if "musiq" in self.metrics:
            print("    MUSIQ            : tryon ONLY (higher = better)")
        print("=" * 70 + "\n")
        
        # Initialize evaluators
        self._evaluators = {}
        
        if "vlm" in self.metrics:
            print("[1/6] Loading VLM evaluator (BLIP-2)...")
            self._evaluators["vlm"] = VLMEvaluator(device=device)
        
        if "jepa" in self.metrics:
            print("[2/6] Loading JEPA evaluator...")
            self._evaluators["jepa"] = JEPAEvaluator(device=device)
        
        if "pose" in self.metrics:
            print("[3/6] Loading Pose evaluator (ViTPose)...")
            self._evaluators["pose"] = PoseConsistencyEvaluator(device=device)
        
        if "clip" in self.metrics:
            print("[4/6] Loading CLIP Garment evaluator...")
            self._evaluators["clip"] = CLIPGarmentEvaluator(device=device)
        
        if "niqe" in self.metrics:
            print("[5/6] Loading NIQE evaluator...")
            self._evaluators["niqe"] = NIQEEvaluator(device=device)
        
        if "musiq" in self.metrics:
            print("[6/6] Loading MUSIQ evaluator...")
            self._evaluators["musiq"] = MUSIQEvaluator(device=device)
        
        print("\n✓ All evaluators loaded.\n")
    
    def evaluate_triplet(
        self,
        person_image: torch.Tensor,
        cloth_image: torch.Tensor,
        tryon_image: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate a single triplet.
        
        Args:
            person_image: (3, H, W) original person image
            cloth_image: (3, H, W) target garment image
            tryon_image: (3, H, W) generated try-on result
        
        Returns:
            Dict with all metric scores
        """
        results = {}
        
        # Ensure batch dimension
        person = person_image.unsqueeze(0) if person_image.dim() == 3 else person_image
        cloth = cloth_image.unsqueeze(0) if cloth_image.dim() == 3 else cloth_image
        tryon = tryon_image.unsqueeze(0) if tryon_image.dim() == 3 else tryon_image
        
        # ═══════════════════════════════════════════════════════════════════════
        # VLM Score: person + cloth + tryon (ALL THREE)
        # ═══════════════════════════════════════════════════════════════════════
        if "vlm" in self._evaluators:
            vlm_result = self._evaluators["vlm"].evaluate_single(
                tryon_image=tryon.squeeze(0),
                person_image=person.squeeze(0),
                cloth_image=cloth.squeeze(0),
            )
            results.update({f"vlm_{k}": v for k, v in vlm_result.items()})
        
        # ═══════════════════════════════════════════════════════════════════════
        # Pose Consistency: person + tryon
        # ═══════════════════════════════════════════════════════════════════════
        if "pose" in self._evaluators:
            pose_result = self._evaluators["pose"].evaluate_single(
                person_image=person.squeeze(0),
                tryon_image=tryon.squeeze(0),
            )
            results.update({f"pose_{k}": v for k, v in pose_result.items()})
        
        # ═══════════════════════════════════════════════════════════════════════
        # CLIP Garment: cloth + tryon
        # ═══════════════════════════════════════════════════════════════════════
        if "clip" in self._evaluators:
            clip_result = self._evaluators["clip"].evaluate_single(
                garment_image=cloth.squeeze(0),
                tryon_image=tryon.squeeze(0),
            )
            results.update({f"clip_{k}": v for k, v in clip_result.items()})
        
        # ═══════════════════════════════════════════════════════════════════════
        # JEPA: tryon ONLY
        # ═══════════════════════════════════════════════════════════════════════
        if "jepa" in self._evaluators:
            jepa_result = self._evaluators["jepa"].evaluate_single(
                tryon_image=tryon.squeeze(0),
            )
            results["jepa_epe"] = jepa_result["epe"]
            results["jepa_log_epe"] = jepa_result["log_epe"]
        
        # ═══════════════════════════════════════════════════════════════════════
        # NIQE: tryon ONLY (no reference)
        # ═══════════════════════════════════════════════════════════════════════
        if "niqe" in self._evaluators:
            results["niqe"] = self._evaluators["niqe"].evaluate_single(
                tryon.squeeze(0)
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # MUSIQ: tryon ONLY (no reference)
        # ═══════════════════════════════════════════════════════════════════════
        if "musiq" in self._evaluators:
            results["musiq"] = self._evaluators["musiq"].evaluate_single(
                tryon.squeeze(0)
            )
        
        return results
    
    def evaluate_batch(
        self,
        person_images: torch.Tensor,
        cloth_images: torch.Tensor,
        tryon_images: torch.Tensor,
    ) -> List[Dict[str, float]]:
        """
        Evaluate a batch of triplets.
        
        Args:
            person_images: (B, 3, H, W) original person images
            cloth_images: (B, 3, H, W) target garment images
            tryon_images: (B, 3, H, W) generated try-on results
        """
        B = tryon_images.shape[0]
        all_results = [{} for _ in range(B)]
        
        # ═══════════════════════════════════════════════════════════════════════
        # VLM Score: person + cloth + tryon (ALL THREE)
        # ═══════════════════════════════════════════════════════════════════════
        if "vlm" in self._evaluators:
            vlm_results = self._evaluators["vlm"].evaluate_batch(
                tryon_images=tryon_images,
                person_images=person_images,
                cloth_images=cloth_images,
            )
            for i, r in enumerate(vlm_results):
                all_results[i].update({f"vlm_{k}": v for k, v in r.items()})
        
        # ═══════════════════════════════════════════════════════════════════════
        # Pose Consistency: person + tryon
        # ═══════════════════════════════════════════════════════════════════════
        if "pose" in self._evaluators:
            pose_results = self._evaluators["pose"].evaluate_batch(
                person_images=person_images,
                tryon_images=tryon_images,
            )
            for i in range(B):
                for k in pose_results:
                    all_results[i][f"pose_{k}"] = pose_results[k][i]
        
        # ═══════════════════════════════════════════════════════════════════════
        # CLIP Garment: cloth + tryon
        # ═══════════════════════════════════════════════════════════════════════
        if "clip" in self._evaluators:
            clip_results = self._evaluators["clip"].evaluate_batch(
                garment_images=cloth_images,
                tryon_images=tryon_images,
            )
            for i in range(B):
                for k in clip_results:
                    all_results[i][f"clip_{k}"] = clip_results[k][i]
        
        # ═══════════════════════════════════════════════════════════════════════
        # JEPA: tryon ONLY
        # ═══════════════════════════════════════════════════════════════════════
        if "jepa" in self._evaluators:
            jepa_results = self._evaluators["jepa"].evaluate_batch(
                tryon_images=tryon_images,
            )
            for i in range(B):
                all_results[i]["jepa_epe"] = jepa_results["epe"][i]
                all_results[i]["jepa_log_epe"] = jepa_results["log_epe"][i]
        
        # ═══════════════════════════════════════════════════════════════════════
        # NIQE: tryon ONLY
        # ═══════════════════════════════════════════════════════════════════════
        if "niqe" in self._evaluators:
            niqe_scores = self._evaluators["niqe"].evaluate_batch(tryon_images)
            for i, s in enumerate(niqe_scores):
                all_results[i]["niqe"] = s
        
        # ═══════════════════════════════════════════════════════════════════════
        # MUSIQ: tryon ONLY
        # ═══════════════════════════════════════════════════════════════════════
        if "musiq" in self._evaluators:
            musiq_scores = self._evaluators["musiq"].evaluate_batch(tryon_images)
            for i, s in enumerate(musiq_scores):
                all_results[i]["musiq"] = s
        
        return all_results
    
    def evaluate_batch_with_mean(
        self,
        person_images: torch.Tensor,
        cloth_images: torch.Tensor,
        tryon_images: torch.Tensor,
    ) -> Dict[str, any]:
        """
        Evaluate a batch and return both per-sample scores and batch means.
        
        All scores are computed as:
            batch_mean_score = mean(individual_sample_scores)
        
        Args:
            person_images: (B, 3, H, W) original person images
            cloth_images: (B, 3, H, W) target garment images
            tryon_images: (B, 3, H, W) generated try-on results
        
        Returns:
            Dict with:
                "per_sample": List of per-sample score dicts
                "batch_mean": Dict of mean scores across all samples
        """
        per_sample = self.evaluate_batch(person_images, cloth_images, tryon_images)
        batch_mean = compute_batch_mean_scores(per_sample)
        
        return {
            "per_sample": per_sample,
            "batch_mean": batch_mean,
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.
        
        Returns mean of all individual sample scores accumulated so far.
        """
        summary = {}
        for name, evaluator in self._evaluators.items():
            summary[name] = evaluator.get_summary()
        return summary
    
    def reset(self):
        """Reset all evaluators."""
        for evaluator in self._evaluators.values():
            evaluator.reset()


def run_evaluation_from_triplet_loader(
    loader: TripletDataLoader,
    suite: WildEvaluationSuite,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Run evaluation using a TripletDataLoader.
    
    Args:
        loader: TripletDataLoader instance
        suite: WildEvaluationSuite instance
        output_path: Path to save JSON results
    
    Returns:
        Dict with per-image results and summary
    """
    all_results = []
    
    print(f"\nEvaluating {len(loader)} triplets...")
    
    for batch in tqdm(loader, desc="Evaluating"):
        person_imgs = batch["person"]
        cloth_imgs = batch["cloth"]
        tryon_imgs = batch["tryon"]
        paths = batch["paths"]
        
        # Evaluate batch
        batch_results = suite.evaluate_batch(person_imgs, cloth_imgs, tryon_imgs)
        
        # Add paths to results
        for i, r in enumerate(batch_results):
            r["person_path"] = paths[i][0]
            r["cloth_path"] = paths[i][1]
            r["tryon_path"] = paths[i][2]
            all_results.append(r)
    
    # Get summary
    summary = suite.get_summary()
    
    # Compile output
    output = {
        "per_image": all_results,
        "summary": summary,
        "config": {
            "n_triplets": len(loader),
            "metrics": suite.metrics,
        }
    }
    
    # Save results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    _print_summary(summary)
    
    return output


def run_wild_evaluation(
    person_dir: Optional[str] = None,
    cloth_dir: Optional[str] = None,
    tryon_dir: Optional[str] = None,
    manifest_path: Optional[str] = None,
    output_path: Optional[str] = None,
    device: str = "cuda",
    batch_size: int = 4,
    metrics: Optional[List[str]] = None,
    img_size: Tuple[int, int] = (512, 384),
    match_by: str = "filename",
) -> Dict:
    """
    Run complete in-the-wild evaluation.
    
    Args:
        person_dir: Directory containing person images
        cloth_dir: Directory containing cloth images
        tryon_dir: Directory containing try-on results
        manifest_path: Path to JSON/CSV manifest file
        output_path: Path to save JSON results
        device: torch device
        batch_size: Batch size for evaluation
        metrics: List of metrics to run
        img_size: Image size (H, W)
        match_by: How to match triplets ("filename", "order", "manifest")
    
    Returns:
        Dict with per-image results and summary statistics
    """
    # Create dataloader
    loader = TripletDataLoader(
        person_dir=person_dir,
        cloth_dir=cloth_dir,
        tryon_dir=tryon_dir,
        manifest_path=manifest_path,
        batch_size=batch_size,
        img_size=img_size,
        match_by=match_by,
    )
    
    # Initialize suite
    suite = WildEvaluationSuite(device=device, metrics=metrics)
    
    # Run evaluation
    return run_evaluation_from_triplet_loader(loader, suite, output_path)


def _print_summary(summary: Dict[str, Dict[str, float]]):
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    
    for metric_name, stats in summary.items():
        print(f"\n  {metric_name.upper()}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="In-the-Wild Virtual Try-On Evaluation with Triplet Dataloader"
    )
    
    # Directory mode
    parser.add_argument("--person_dir", type=str, 
                       help="Directory with person images")
    parser.add_argument("--cloth_dir", type=str, 
                       help="Directory with cloth/garment images")
    parser.add_argument("--tryon_dir", type=str, 
                       help="Directory with try-on results")
    
    # Manifest mode
    parser.add_argument("--manifest", type=str,
                       help="Path to JSON/CSV manifest file")
    
    # Single triplet mode
    parser.add_argument("--person", type=str, 
                       help="Single person image path")
    parser.add_argument("--cloth", type=str, 
                       help="Single cloth image path")
    parser.add_argument("--tryon", type=str, 
                       help="Single try-on image path")
    
    # Options
    parser.add_argument("--output", type=str, default="wild_eval_results.json",
                       help="Output JSON path")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--metrics", nargs="+", default=None,
                       help="Metrics to run: vlm jepa pose clip niqe musiq")
    parser.add_argument("--img_size", type=int, nargs=2, default=[512, 384],
                       help="Image size (H W)")
    parser.add_argument("--match_by", type=str, default="filename",
                       choices=["filename", "order"],
                       help="How to match triplets from directories")
    
    args = parser.parse_args()
    
    # Single triplet mode
    if args.person and args.cloth and args.tryon:
        suite = WildEvaluationSuite(device=args.device, metrics=args.metrics)
        
        person = load_image(args.person, tuple(args.img_size))
        cloth = load_image(args.cloth, tuple(args.img_size))
        tryon = load_image(args.tryon, tuple(args.img_size))
        
        results = suite.evaluate_triplet(person, cloth, tryon)
        
        print("\n" + "=" * 70)
        print("  SINGLE TRIPLET EVALUATION RESULTS")
        print("=" * 70)
        print(f"\n  Person: {args.person}")
        print(f"  Cloth:  {args.cloth}")
        print(f"  TryOn:  {args.tryon}")
        print("\n  Scores:")
        for k, v in sorted(results.items()):
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        print("=" * 70 + "\n")
        
        return
    
    # Directory or manifest mode
    run_wild_evaluation(
        person_dir=args.person_dir,
        cloth_dir=args.cloth_dir,
        tryon_dir=args.tryon_dir,
        manifest_path=args.manifest,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        metrics=args.metrics,
        img_size=tuple(args.img_size),
        match_by=args.match_by,
    )


if __name__ == "__main__":
    main()
