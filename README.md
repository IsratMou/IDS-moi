# MOI-Lite + E-SATF: Lightweight Hierarchical Intrusion Detection System for IoT Networks

> **CSE 414 Project** | TON-IoT Dataset | TensorFlow · SHAP · HuggingFace Spaces

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/MOUcat/iot-intrusion-detection-dashboard)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)](https://www.kaggle.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)

---

## 📌 Overview

We propose **MOI-Lite**, a lightweight neural architecture with only **37K parameters**, designed specifically for IoT intrusion detection under tight memory constraints. Combined with **E-SATF** (Explanation-Stability-Aware Training Framework), the system delivers:

- ✅ **99.79% binary F1** (Normal vs. Attack classification)
- ✅ **94.37% multiclass macro F1** (Attack type identification)
- ✅ **65 KB after INT8 quantization** — fits Arduino-class IoT microcontrollers
- ✅ Statistically equivalent to DNN/CNN baselines (McNemar *p* > 0.05) using **36.5% fewer parameters**
- ✅ Up to **26% relative gain** in FGSM adversarial robustness with E-SATF

---

## 🏗️ Architecture

### Hierarchical Pipeline

```
IoT Network Traffic
        │
   ┌────▼────┐
   │ Stage 1 │  Binary Classification: Normal vs. Attack
   │(MOI-Lite)│  → 99.79% F1
   └────┬────┘
        │ Attack detected
   ┌────▼────┐
   │ Stage 2 │  Multiclass Classification: Attack Type
   │(MOI-Lite)│  → 94.37% macro F1
   └────┬────┘
        │
  Attack Label
(DoS, DDoS, Backdoor,
 Injection, MITM, etc.)
```

### MOI-Lite Architecture (37K Parameters)

- Multi-scale dilated 1D-CNN blocks (dilation rates: 1, 2, 4)
- Lightweight self-attention (2 heads, key dim 16)
- Spectral Normalization + DropPath (Stochastic Depth)
- INT8 quantization → **65 KB** on-device footprint

### E-SATF (Explanation-Stability-Aware Training Framework)

E-SATF augments standard training with a **consistency regularization term** that penalizes instability in SHAP explanations under input perturbation, improving both interpretability and adversarial robustness.

---

## 📊 Results Summary

### Stage-1 Binary Classification

| Model         | E-SATF | F1 Score | Params  |
|---------------|--------|----------|---------|
| DNN           | ✗      | ~99.7%   | ~58K    |
| DNN           | ✓      | ~99.7%   | ~58K    |
| CNN           | ✗      | ~99.7%   | ~55K    |
| CNN           | ✓      | ~99.7%   | ~55K    |
| **MOI-Lite**  | ✗      | ~99.7%   | **37K** |
| **MOI-Lite**  | ✓      | **99.79%** | **37K** |

### Stage-2 Multiclass Classification (Attack-Only)

| Model         | E-SATF | Macro F1 |
|---------------|--------|----------|
| DNN           | ✓      | ~94%     |
| CNN           | ✓      | ~94%     |
| **MOI-Lite**  | ✓      | **94.37%** |

### Adversarial Robustness (FGSM, ε=0.10)

E-SATF improves FGSM robustness by up to **26% relative gain** across all three architectures.

### Quantization Footprint

| Model       | Full (float32) | INT8 Quantized |
|-------------|---------------|----------------|
| DNN         | ~220 KB       | ~55 KB         |
| CNN         | ~270 KB       | ~68 KB         |
| **MOI-Lite**| ~148 KB       | **65 KB** ✅   |

---

## 📁 Repository Structure

```
IDS-moi/
├── ton-iot-moi-cse-414.ipynb   # Main Kaggle notebook (full pipeline)
├── README.md                    # This file
└── ...
```

### Notebook Cells Overview

| Phase | Cells | Description |
|-------|-------|-------------|
| 0 | 0.0–0.7 | Setup, config, data loading, preprocessing, feature engineering |
| 1 | 1.1–1.6 | Loss functions (Focal + E-SATF), model architectures (DNN/CNN/MOI-Lite) |
| 2 | 2.1–2.2 | Custom training loop with E-SATF, LR scheduling |
| 3 | 3.1–3.3 | Train all 6 Stage-1 & 6 Stage-2 models; hierarchical pipeline assembly |
| 4 | 4.1–4.2 | SHAP stability analysis (multi-seed, Top-K Jaccard) |
| 5 | 5.1–5.2 | FGSM adversarial evaluation (4 epsilons × 6 models) |
| 6 | 6.1–6.2 | INT8 quantization + latency benchmark |
| 7 | 7.1–7.2 | Master results tables, paper-ready figures, confusion matrices |

---

## 🗂️ Dataset

**TON-IoT** (Network Traffic Dataset)

- Source: Kaggle — [`arnobbhowmik/ton-iot-network-dataset`](https://www.kaggle.com/datasets/arnobbhowmik/ton-iot-network-dataset)
- ~500K samples (subsampled from full dataset with majority-class cap of 50K)
- Train / Validation / Test split: **70 / 15 / 15**
- Attack classes: DoS, DDoS, Backdoor, Injection, MITM, Password, Ransomware, Scanning, XSS, Normal

### Preprocessing Pipeline

1. Sparse feature removal
2. Feature engineering (log-scale + ratio features for high-range columns)
3. One-hot encoding of categorical features
4. StandardScaler normalization
5. Class-weighted sampling for imbalanced classes

---

## 🖥️ Live Demo

The interactive dashboard is hosted on Hugging Face Spaces:

🔗 **[https://huggingface.co/spaces/MOUcat/iot-intrusion-detection-dashboard](https://huggingface.co/spaces/MOUcat/iot-intrusion-detection-dashboard)**

Features:
- Input IoT network traffic features (or auto-fill with sample data)
- Choose model: DNN, CNN, or MOI-Lite (with/without E-SATF)
- Get binary classification (Normal / Attack) + attack type prediction
- View SHAP feature importance explanations

---

## 🚀 Running the Notebook

### On Kaggle (Recommended)

1. Open the notebook on Kaggle
2. Add the dataset: `arnobbhowmik/ton-iot-network-dataset`
3. Set accelerator: **GPU T4 ×2** (Settings → Accelerator)
4. Run all cells in order (Cell 0.0 **must** run first for determinism)

### Environment Requirements

```
tensorflow >= 2.12
numpy
pandas
scikit-learn
shap
matplotlib
seaborn
```

> ⚠️ **Note:** Cell 0.0 must be the very first cell executed. It sets critical environment variables (`PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`) **before** TensorFlow is imported.

---

## 🔬 Key Contributions

1. **MOI-Lite Architecture** — A novel 37K-parameter model combining dilated CNNs with lightweight self-attention, specifically designed to fit IoT memory budgets (≤ 65 KB after quantization).

2. **E-SATF** — An Explanation-Stability-Aware Training Framework that adds a consistency regularization loss term, improving both SHAP explanation stability and adversarial robustness without sacrificing accuracy.

3. **Hierarchical IDS Pipeline** — A two-stage detection pipeline that first identifies whether traffic is malicious, then classifies the specific attack type, achieving high macro F1 even on minority attack classes.

4. **Statistical Validation** — All comparisons validated with McNemar's test (*p* > 0.05), confirming MOI-Lite matches larger baselines without statistically significant accuracy loss.

---

## 📈 Figures

The notebook generates 6 paper-ready figures (Cell 7.2) including:

- Binary & multiclass F1 comparison across all 6 models
- SHAP Top-K Jaccard stability comparison (with vs. without E-SATF)
- FGSM adversarial accuracy decay curves
- Quantization size vs. accuracy trade-off
- Confusion matrices for all model-stage combinations

---

## 👥 Authors

**Israt Mou** — [GitHub](https://github.com/IsratMou)

Course: CSE 414 | Dataset: TON-IoT | Framework: TensorFlow + SHAP

---

## 📄 License

This project is for academic/research use. Dataset license follows Kaggle TON-IoT terms.
