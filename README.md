# Pretrained Metrics Evaluation

A comprehensive evaluation framework for Virtual Try-On systems using pretrained metrics. Includes EDA (Exploratory Data Analysis), pretrained metrics computation, and in-the-wild evaluation capabilities.

---

## 📁 Project Structure

```
Pretrained_Metrics_Evaluation/
├── configs/                    # YAML configuration files
├── dataloaders/                # Dataset-specific dataloaders
│   ├── curvton_dataloader.py   # CURVTON dataset (Easy/Medium/Hard)
│   ├── dresscode_dataloader.py # DressCode dataset
│   ├── vitonhd_dataloader.py   # VITON-HD dataset
│   └── ...
├── datasets/                   # Base dataset classes
├── EDA/                        # Exploratory Data Analysis
│   ├── feature_extractor.py    # Feature extraction pipeline
│   ├── plot_style.py           # ECCV publication-quality styling
│   ├── run_curvton_eda.py      # CURVTON EDA pipeline
│   ├── run_eda.py              # General EDA pipeline
│   └── plots/                  # Individual EDA plot modules
│       ├── p1_pose_eda.py
│       ├── p2_occlusion_eda.py
│       ├── p3_background_eda.py
│       ├── p4_illumination_eda.py
│       ├── p5_body_shape_eda.py
│       ├── p6_appearance_eda.py
│       ├── p7_garment_eda.py
│       ├── p11_clip_embedding_eda.py
│       └── ...
├── metrics/                    # Core evaluation metrics
│   ├── distribution_metrics.py # FID, KID, etc.
│   ├── image_metrics.py        # SSIM, LPIPS, etc.
│   ├── vlm_score.py            # VLM-based plausibility (Qwen3-VL)
│   └── ...
├── pretrained_metrics/         # Pretrained metric computation
│   ├── metrics/                # Individual metric implementations
│   │   ├── m1_pose.py
│   │   ├── m2_occlusion.py
│   │   ├── m3_background.py
│   │   ├── m4_illumination.py
│   │   ├── m5_body_shape.py
│   │   ├── m6_appearance.py
│   │   ├── m7_garment_texture.py
│   │   ├── m8_vae_latent.py
│   │   └── m9_camera_angle.py
│   └── compute_pretrained_metrics.py
├── on_the_wild_evaluation/     # In-the-wild evaluation
│   ├── vlm_evaluator.py        # VLM-based evaluation
│   ├── clip_garment_evaluator.py
│   ├── pose_evaluator.py
│   └── ...
├── evaluate.py                 # Main evaluation script
├── config.py                   # Global configuration
└── requirements.txt            # Python dependencies
```

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/ankitbelbase17/Pretrained_Metrics_Evaluation.git
cd Pretrained_Metrics_Evaluation
```

### 2. Create virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install additional packages (optional)
```bash
# For CLIP embeddings
pip install openai-clip
# or
pip install open-clip-torch

# For flash attention (recommended for VLM)
pip install flash-attn --no-build-isolation

# For distribution metrics
pip install torch-fidelity

# For pose estimation
pip install timm
```

---

## 📊 Metrics

| Metric | Direction | Implementation |
|---|---|---|
| **PSNR** | ↑ | `skimage.metrics.peak_signal_noise_ratio` |
| **SSIM** | ↑ | `skimage.metrics.structural_similarity` |
| **Masked SSIM** | ↑ | SSIM restricted to garment mask region |
| **LPIPS** | ↓ | `lpips` library (AlexNet backbone) |
| **FID** | ↓ | `torch-fidelity` |
| **IS (mean ± std)** | ↑ | `torch-fidelity` |
| **KID (mean ± std)** | ↓ | `torch-fidelity` |
| **Pose Error (PE)** | ↓ | HRNet-W32 keypoints via `timm` |
| **VLM Score** | ↑ | Qwen3-VL-32B plausibility 0-1 |
| **JEPA EPE** | ↓ | Embedding Prediction Error via ViT-B/16 |

---

## 📊 Running EDA (Exploratory Data Analysis)

### CURVTON Dataset EDA

```bash
# Full EDA pipeline (all difficulty levels)
python EDA/run_curvton_eda.py \
    --base_path /path/to/curvton/dataset_ultimate \
    --out_dir figures/curvton \
    --sample_ratio 1.0

# 20% sample for quick analysis
python EDA/run_curvton_eda.py \
    --base_path /path/to/curvton/dataset_ultimate \
    --out_dir figures/curvton_20pct \
    --sample_ratio 0.2

# Specific difficulty levels
python EDA/run_curvton_eda.py \
    --base_path /path/to/curvton/dataset_ultimate \
    --difficulties easy medium \
    --sample_ratio 0.5

# Force recompute cached features
python EDA/run_curvton_eda.py \
    --base_path /path/to/curvton/dataset_ultimate \
    --force_recompute
```

### CLIP Embedding EDA (PCA/t-SNE)

```bash
python EDA/plots/p11_clip_embedding_eda.py \
    --base_path /path/to/curvton/dataset_ultimate \
    --out_dir figures/clip_embeddings \
    --sample_ratio 0.2 \
    --device cuda
```

### General EDA (VITON-HD, DressCode, etc.)

```bash
python EDA/run_eda.py \
    --config configs/eda_datasets.yaml \
    --out_dir figures/eda
```

---

## 📈 Computing Pretrained Metrics

### CURVTON Dataset

```bash
python pretrained_metrics/compute_curvton_metrics.py \
    --base_path /path/to/curvton/dataset_ultimate \
    --out_dir results/curvton_metrics \
    --sample_ratio 1.0
```

### Other Datasets

```bash
# VITON-HD
bash pretrained_metrics/sh/compute_vitonhd.sh

# DressCode
bash pretrained_metrics/sh/compute_dresscode.sh

# LAION-RVS-Fashion
bash pretrained_metrics/sh/compute_laion.sh
```

---

## 🎯 Evaluation

### Standard Evaluation (with ground truth)

```bash
python evaluate.py \
    --dataset viton_hd \
    --root /path/to/vitonhd \
    --pred_dir /path/to/predictions \
    --output_dir ./results \
    --batch_size 8 \
    --device cuda
```

### Multi-dataset Evaluation

```bash
# Edit configs/all_datasets.yaml with your paths
python evaluate.py --config configs/all_datasets.yaml
```

### Skip Heavy Metrics

```bash
python evaluate.py --config configs/all_datasets.yaml \
    --no_vlm    # skip VLM score
    --no_jepa   # skip JEPA
    --no_pose   # skip pose error
```

### In-the-Wild Evaluation (no ground truth)

```bash
python on_the_wild_evaluation/run_evaluation.py \
    --config configs/model2street.yaml \
    --tryon_dir /path/to/tryon_results \
    --person_dir /path/to/person_images \
    --cloth_dir /path/to/cloth_images \
    --out_dir results/wild_evaluation
```

---

## 🔍 VLM-based Evaluation

The VLM evaluator uses **Qwen3-VL-32B-Instruct** for multi-image evaluation:

```python
from metrics.vlm_score import VLMScoreMetric

metric = VLMScoreMetric(device="cuda")

# Evaluate with all three images (recommended)
results = metric.compute_batch(
    tryon_images,           # Generated try-on results
    person_images=person,   # Original person images
    cloth_images=cloth,     # Original garment images
)
# Returns: [{'vlm_score': 0.85, 'reason': '...'}, ...]
```

### VLM Evaluation Criteria
1. **Photorealism** - Fabric texture, wrinkles, shading
2. **Lighting Consistency** - Shadows, highlights alignment
3. **Color/Intensity Matching** - Exposure consistency
4. **Seamless Blending** - No artifacts, halos
5. **Body Alignment** - Pose and geometry
6. **Occlusion Handling** - Arms, hair, objects
7. **Global Scene Consistency** - Natural integration

---

## 📊 EDA Output

The EDA pipeline generates publication-quality figures:

| Plot | Description | Image Used |
|------|-------------|------------|
| `pose/` | Pose distribution (UMAP, joint angles) | initial_person_image |
| `occlusion/` | Garment occlusion analysis | initial_person_image |
| `background/` | Background complexity (entropy, objects) | initial_person_image |
| `illumination/` | Lighting analysis (luminance, gradients) | initial_person_image |
| `body_shape/` | Body shape distribution (SMPL betas) | initial_person_image |
| `appearance/` | Face/identity diversity | initial_person_image |
| `garment/` | Garment texture diversity (CLIP) | cloth_image |
| `clip_embeddings/` | CLIP image/text PCA/t-SNE | cloth_image |

---

## 🎨 Publication-Quality Figures

All figures are styled for ECCV/CVPR/ICCV/NeurIPS:
- **PDF**: 600 DPI, Type 42 fonts (editable in Illustrator)
- **PNG**: 150 DPI preview
- **Colorblind-friendly** palette
- **Single-column**: 3.25" width
- **Double-column**: 6.875" width

---

## 📦 Dataset Structures

### CURVTON Dataset
```
dataset_ultimate/
├── easy/
│   ├── female/
│   │   ├── cloth_image/
│   │   ├── initial_person_image/
│   │   └── tryon_image/
│   └── male/
├── medium/
└── hard/
```

### VITON-HD
```
vitonhd/
├── train/
│   ├── image/
│   ├── cloth/
│   └── ...
└── test/
    ├── image/
    ├── cloth/
    └── agnostic-mask/
```

### DressCode
```
dresscode/
├── upper_body/
│   ├── images/
│   ├── clothes/
│   └── label_maps/
├── lower_body/
└── dresses/
```

---

## 📋 Configuration Files

| Config | Description |
|--------|-------------|
| `all_datasets.yaml` | All supported datasets |
| `benchmark_test.yaml` | Standard benchmark evaluation |
| `curvton_eda.yaml` | CURVTON EDA settings |
| `model2model.yaml` | Model-to-model comparison |
| `model2street.yaml` | Model-to-street evaluation |
| `shop2model.yaml` | Shop-to-model evaluation |
| `training_eval.yaml` | Training-time evaluation |

---

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python EDA/run_curvton_eda.py --sample_ratio 0.1

# Use CPU for VLM
python evaluate.py --device cpu
```

### Missing CLIP
```bash
pip install openai-clip
# or
pip install open-clip-torch
```

### Flash Attention Issues
```bash
# The code will fall back to standard attention automatically
pip install transformers --upgrade
```

---

## 📄 License

This project is for research purposes.

## 📧 Contact

For questions, please open an issue on GitHub.
