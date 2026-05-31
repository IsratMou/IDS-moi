---
title: Multi-Model Hierarchical IoT IDS Demo
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
python_version: 3.12
app_file: app.py
pinned: true
license: mit
---

# 🛡️ Multi-Model Hierarchical IoT IDS Dashboard (DNN vs CNN vs MOI-Lite)

This repository hosts the live production demo of a **Two-Stage Hierarchical Intrusion Detection System (IDS)** designed for IoT networks. It features a comparative framework allowing users to evaluate standard deep architectures (**DNN**, **CNN**) against our proposed **MOI-Lite** architecture, reinforced with **E-SATF** (Explanation-Stability-Aware Training Framework).

---

## 📊 Core Performance Summary (TON-IoT Dataset)

The underlying models are trained and validated on the comprehensive **TON-IoT network dataset** (spanning 62 processed feature signatures and 10 network event classes).

### Stage-1 Binary Performance Comparison

| Pipeline Architecture  | Parameters | Best Epoch | Binary F1-Score | Validation Accuracy | PR-AUC |
| :--------------------- | :--------: | :--------: | :-------------: | :-----------------: | :----: |
| **DNN (NoSATF)**       |   58,689   |     71     |   **0.9976**    |       0.9962        | 1.0000 |
| **DNN (SATF)**         |   58,689   |     62     |     0.9960      |       0.9933        | 0.9999 |
| **CNN (NoSATF)**       |   71,169   |     68     |     0.9972      |       0.9955        | 0.9997 |
| **CNN (SATF)**         |   71,169   |     48     |     0.9931      |       0.9892        | 0.9998 |
| **MOI-Lite (NoSATF)**  |   61,473   |     40     |     0.9967      |       0.9938        | 0.9999 |
| **MOI-Lite (SATF)** 🚀 | **61,473** |     50     |   **0.9957**    |       0.9926        | 0.9998 |

> 💡 **Statistical Rigor**: McNemar tests confirm that the lightweight **MOI-Lite** variants maintain statistical parity ($p > 0.05$) with the heavy DNN/CNN baselines while significantly lowering operational and memory overheads.

---

## ⚙️ Two-Stage Hierarchical Routing Engine
