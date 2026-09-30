# SphereVAD: Training-Free Video Anomaly Detection via Geodesic Inference on the Unit Hypersphere

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7.0](https://img.shields.io/badge/PyTorch-2.7.0%2Bcu126-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.08003-b31b1b.svg)](https://arxiv.org/abs/2605.08003)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **SphereVAD: Training-Free Video Anomaly Detection via Geodesic Inference on the Unit Hypersphere** (arXiv:2605.08003).

SphereVAD is a training-free, calibration-driven framework that performs video anomaly detection directly on the unit hypersphere $\mathbb{S}^{d-1}$. By reformulating feature normalization and anomaly boundary modeling in non-Euclidean spherical space, SphereVAD effectively separates anomalies without requiring real abnormal training videos.

---

## 🌟 Key Highlights

- **Riemannian Spherical Centering**: Projects features onto the tangent space $T_{\mu}\mathbb{S}^{d-1}$ via Riemannian Log/Exp maps centered at the Fréchet (Karcher) mean, eliminating representation bias from foundation models.
- **Hyperspherical vMF Scoring**: Models normal and abnormal manifolds using spherical prototype clustering and calculates posterior anomaly probabilities via von Mises–Fisher (vMF) likelihood-ratio scoring.
- **Holistic Scene Attention (HSA)**: Harnesses visual tokens to aggregate semantic context across global corpus clips and local intra-video clips.
- **Spherical Geodesic Pulling (SGP)**: Employs an adaptive ambiguity band (MAD) to detect uncertain clips and shifts them along great-circle geodesics via Spherical Linear Interpolation (SLERP).
- **Synthetic Calibration Storyboards**: Utilizes paired normal/abnormal synthetic scenarios to construct decision anchors without exposing the model to test domains.

---

## 📂 Repository Structure

```text
SphereVAD/
├── SphereVAD-pipeline/
│   ├── main.py                     # Main evaluation script for UCF-Crime, XD-Violence, UBnormal
│   ├── Feature_exactor/
│   │   ├── Get_freature.py         # Multi-GPU token extraction (feat_last_token, feat_vis_last)
│   │   └── DSLP-compute.py         # Multi-layer hidden-state feature extractor
│   ├── Inference/                  # Core SphereVAD mathematical implementation
│   │   ├── __init__.py
│   │   ├── frechet_mean.py         # Karcher / Fréchet mean iterative gradient descent on S^{d-1}
│   │   ├── spherical_centering.py  # Riemannian tangent-space projection & re-normalization
│   │   ├── vmf_prototypes.py       # Spherical K-Means & Euclidean prototype learning
│   │   ├── vmf_scoring.py          # Two-class vMF mixture posterior likelihood scoring
│   │   ├── hsa.py                  # Holistic Scene Attention (intra- & cross-video enhancement)
│   │   ├── sgp.py                  # Spherical Geodesic Pulling & adaptive ambiguity bands
│   │   └── utils.py                # Geodesic metrics, SLERP, temporal Gaussian smoothing
│   └── Preprocess/
│       ├── generate_labels.py      # Standardized ground-truth parser for UCF, XD, UBnormal
│       └── extract_frames.py       # Multi-process video sampler (uniform 4-frame clips)
└── Sysneticdata-pipeline/
    ├── generate_json.py            # LLM prompt pipeline for paired storyboard scenarios
    └── generate_img.py             # Generative API call to produce paired 2x2 storyboard grids
```

---

## 📦 Synthetic Calibration Dataset (`syndata.zip`)

The synthetic paired calibration dataset is available on Baidu Netdisk:

- **Link**: [Baidu Netdisk (百度网盘)](https://pan.baidu.com/s/1UxPtd6ib53Ov-YwKj-lh4Q?pwd=a7ks)
- **Extraction Code (提取码)**: `a7ks`

Unpack the archive to your target data directory:
```bash
unzip syndata.zip -d ./data/
```

---

## 🛠️ Installation

### 1. Environment Setup
We recommend using **Python 3.12** inside Conda:

```bash
git clone https://github.com/your-username/SphereVAD.git
cd SphereVAD

conda create -n spherevad python=3.12 -y
conda activate spherevad
```

### 2. Install PyTorch & Dependencies
Install dependencies according to `requirements.txt`:

```bash
# Install PyTorch 2.7.0 with CUDA 12.6 support
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126

# Install remaining libraries
pip install -r requirements.txt
```

> **Note on optional dependencies**: For faster API calls and video decoding, you may also optionally install `decord` or `flash-attn` depending on your GPU architecture.

---

## 🚀 Step-by-Step Workflow

### Step 1: Benchmark Preprocessing (UCF-Crime / XD-Violence / UBnormal)

1. **Generate ground truth label CSVs:**
   Configure dataset paths in `SphereVAD-pipeline/Preprocess/generate_labels.py` and run:
   ```bash
   python SphereVAD-pipeline/Preprocess/generate_labels.py
   ```
2. **Sample video clips:**
   Extract uniform 4-frame clips at your target resolution (default $336 \times 336$):
   ```bash
   python SphereVAD-pipeline/Preprocess/extract_frames.py
   ```

---

### Step 2: Feature Extraction via Vision-Language Models (VLM)

Extract sequence token representations (`feat_last_token`) and holistic visual tokens (`feat_vis_last`):

- **Target Benchmarks (UCF-Crime / XD-Violence / UBnormal):**
  Set `DATASET = "UBnormal"` (or `"UCF"` / `"XD"`), configure `MODEL_PATH`, and assign your GPU workers in `SphereVAD-pipeline/Feature_exactor/Get_freature.py`, then run:
  ```bash
  python SphereVAD-pipeline/Feature_exactor/Get_freature.py
  ```

- **Synthetic Calibration Data:**
  Set `DATASET = "SYN"` in `Get_freature.py` (or use `DSLP-compute.py` for full intermediate layer analysis):
  ```bash
  python SphereVAD-pipeline/Feature_exactor/Get_freature.py
  ```

---

### Step 3: Run SphereVAD Inference & Evaluation

Configure dataset paths and hyperparameter weights in `SphereVAD-pipeline/main.py`:

```python
DATASET = "UBnormal"  # Options: "UCF", "XD", "UBnormal"

# Set calibration feature locations
TRAIN_FEATURE_DIR = "./data/syndata_features/"
TRAIN_LABEL_CSV   = "./data/syndata_train_label.csv"

# Core Hyperparameters
KAPPA          = 1.0 / 0.3  # vMF concentration parameter
K_NORM         = 3          # Normal prototype count
K_ABN          = 25         # Anomaly prototype count
HOLISTIC_ALPHA = 0.35       # Cross-video HSA weight
SLERP_BETA     = 0.1        # SGP geodesic pulling step
```

Run evaluation:
```bash
python SphereVAD-pipeline/main.py
```

The pipeline outputs comparative performance metrics across three progressive stages:
- **`M1: vMF-Baseline`**: Riemannian Spherical Centering + vMF Likelihood Ratio Scoring.
- **`M2: +Holistic(Global)`**: Cross-Video Holistic Scene Attention (HSA).
- **`M3: +vMF-Guided(Single)`**: Spherical Geodesic Pulling (SGP) on ambiguous clips.

Evaluation metrics reported: **Frame-level AP**, **Frame-level AUC**, and **Video-level AUC**.

---

### Step 4: Synthetic Data Generation Pipeline (Optional)

To synthesize customized paired storyboard datasets using generative models:

1. **Generate prompt paired scenarios:**
   Configure your LLM endpoint in `Sysneticdata-pipeline/generate_json.py` and run:
   ```bash
   python Sysneticdata-pipeline/generate_json.py
   ```

2. **Render 2x2 storyboard image grids:**
   Configure your image generation API credentials in `Sysneticdata-pipeline/generate_img.py` and run:
   ```bash
   python Sysneticdata-pipeline/generate_img.py
   ```

---

## 📜 Citation

If you find this work or the dataset helpful in your research, please cite our paper:

```bibtex
@article{huang2026spherevad,
  title={SphereVAD: Training-Free Video Anomaly Detection via Geodesic Inference on the Unit Hypersphere},
  author={Huang, Chao and Wei, Penfei and Wang, Wei and Wen, Jie and Wang, Zhihua and Shen, Li and Ren, Wenqi and Cao, Xiaochun},
  journal={arXiv preprint arXiv:2605.08003},
  year={2026}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

