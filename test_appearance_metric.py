"""
Test script for AppearanceMetrics (face diversity metric only).
"""
import torch
from pretrained_metrics.metrics.m6_appearance import AppearanceMetrics

def main():
    # Small batch, random images (B=4, 3x320x320)
    B, C, H, W = 4, 3, 320, 320
    imgs = torch.rand(B, C, H, W)
    metric = AppearanceMetrics(device="cpu")
    metric.update(imgs)
    result = metric.compute()
    print("Appearance metric result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
