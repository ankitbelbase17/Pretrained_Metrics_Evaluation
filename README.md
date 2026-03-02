# Virtual Try-On Evaluation Pipeline

## Overview

A modular evaluation suite computing **11 metrics** across **10 virtual try-on datasets** from a single command.

---

## Metrics

| Metric | Direction | Implementation |
|---|---|---|
| **PSNR** | ↑ | `skimage.metrics.peak_signal_noise_ratio` |
| **SSIM** | ↑ | `skimage.metrics.structural_similarity` |
| **Masked SSIM** | ↑ | SSIM restricted to garment mask region |
| **LPIPS** | ↓ | `lpips` library (AlexNet backbone) |
| **FID** | ↓ | `torch-fidelity` |
| **IS (mean ± std)** | ↑ | `torch-fidelity` |
| **KID (mean ± std)** | ↓ | `torch-fidelity` |
| **Pose Error (PE)** | ↓ | HRNet-W32 keypoints via `timm`; pixel-proxy fallback |
| **VLM Score** | ↑ | BLIP-2 plausibility 1-10; graceful stub fallback |
| **JEPA EPE** | ↓ | Embedding Prediction Error via ViT-B/16 + linear predictor |
| **JEPA Trace Σ** | ↑ | Tr(Cov) over target embeddings |

---

## Datasets

| # | Name | Flag | Paper |
|---|---|---|---|
| 1 | VITON | `viton` | Han et al. 2018 |
| 2 | VITON-HD | `viton_hd` | Choi et al. 2021 |
| 3 | DressCode | `dresscode` | Morelli et al. 2022 |
| 4 | MPV | `mpv` | Dong et al. 2019 |
| 5 | DeepFashion-TryOn | `deepfashion_tryon` | Ge et al. 2021 |
| 6 | ACGPN | `acgpn` | Yang et al. 2020 |
| 7 | CP-VTON | `cp_vton` | Wang et al. 2018 |
| 8 | HR-VTON | `hr_vton` | Lee et al. 2022 |
| 9 | LaDI-VTON | `ladi_vton` | Morelli et al. 2023 |
| 10 | OVNet | `ovnet` | — |

Each dataset returns `(cloth, person, gt, mask)` via a unified DataLoader.

---

## Project Structure

```
pretrained_metrics_evals/
│
├── evaluate.py               ← Main evaluation script
├── demo_synthetic.py         ← Smoke-test (no dataset needed)
├── requirements.txt
│
├── datasets/
│   ├── __init__.py
│   ├── base_dataset.py       ← Abstract base class
│   └── loaders.py            ← 10 concrete dataset loaders + registry
│
├── metrics/
│   ├── __init__.py
│   ├── image_metrics.py      ← PSNR, SSIM, Masked-SSIM, LPIPS
│   ├── distribution_metrics.py ← FID, IS, KID
│   ├── pose_error.py         ← Pose Error (HRNet / proxy)
│   ├── vlm_score.py          ← VLM Plausibility Score (BLIP-2)
│   └── jepa_metrics.py       ← JEPA EPE + Trace
│
├── configs/
│   └── all_datasets.yaml     ← Multi-dataset config template
│
└── results/                  ← Auto-created; JSON + CSV outputs
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Optional heavy dependencies** (install only what you need):
> - `pip install lpips` — LPIPS
> - `pip install torch-fidelity` — FID / IS / KID
> - `pip install timm` — HRNet (Pose Error) + ViT (JEPA)
> - `pip install transformers accelerate` — VLM Score (BLIP-2)

---

## Quick Start

### 1. Smoke-test (no dataset)
```bash
python demo_synthetic.py
```

### 2. Single dataset
```bash
python evaluate.py \
    --dataset viton \
    --root /path/to/VITON \
    --pred_dir /path/to/your_model_outputs \
    --output_dir ./results \
    --batch_size 8 \
    --device cuda
```

### 3. All 10 datasets at once
```bash
# 1. Edit configs/all_datasets.yaml — fill in root / pred_dir for each dataset
# 2. Run:
python evaluate.py --config configs/all_datasets.yaml
```

### 4. Skip heavy metrics
```bash
python evaluate.py --config configs/all_datasets.yaml \
    --no_vlm    # skip BLIP-2
    --no_jepa   # skip JEPA
    --no_pose   # skip HRNet pose
```

---

## Integrating Your Model

By default, `evaluate.py` loads predictions from `--pred_dir` (PNG/JPGs named by sample ID).

To run your model **inline**, edit the `_run_model()` function in `evaluate.py`:

```python
# evaluate.py  →  _run_model()
def _run_model(cloth: torch.Tensor, person: torch.Tensor) -> torch.Tensor:
    # Replace this with your model forward pass:
    return my_tryon_model(person, cloth)
```

---

## Output

Results are written to `--output_dir` (default: `./results/`):
- `metrics_YYYYMMDD_HHMMSS.json` — full results per dataset
- `metrics_YYYYMMDD_HHMMSS.csv`  — tabular summary

---

## Dataset Folder Structures

### VITON / CP-VTON / ACGPN
```
<root>/
  test/
    image/          ← person images
    cloth/          ← garment images
    image-parse/    ← segmentation masks (optional)
  test_pairs.txt    ← "person_img.jpg cloth_img.jpg"
```

### VITON-HD / HR-VTON
```
<root>/
  test/
    image/
    cloth/
    agnostic-mask/  ← binary body mask
  test_pairs.txt
```

### DressCode
```
<root>/
  upper_body/       ← (or lower_body / dresses)
    images/         ← person (id_0.jpg) + gt (id_1.jpg)
    clothes/        ← garment (id_1.jpg)
    label_maps/     ← optional segmentation
  upper_body_pairs_test.txt
```

### DeepFashion-TryOn
```
<root>/
  test/
    image/  cloth/  gt/  mask/
  test_pairs.txt  "person cloth gt"
```

### LaDI-VTON
```
<root>/
  test/  images/  clothes/  gt/  masks/
  test_pairs.txt
```

---

## Notes

- **Masked SSIM**: uses the per-sample `mask` from the dataloader. If no mask file exists, the full image is used (white mask).
- **Pose Error fallback**: if `timm` HRNet inference fails, a mean-pixel-displacement proxy is used and labelled accordingly.  
- **VLM Score fallback**: if BLIP-2 fails to load (GPU memory / missing weights), each image gets a neutral score of 5.0.
- **JEPA**: uses ViT-B/16 as a surrogate context/target encoder with a random linear predictor. Swap in official I-JEPA weights for research-grade results.
