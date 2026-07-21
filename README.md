# MDEP: Microglial-Driven Evidential Pruning

[![Paper](https://img.shields.io/badge/AAAI--2026-Accepted-success.svg)](https://aaai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository is the official implementation of **MDEP (Microglial-Driven Evidential Pruning)**, a structured, uncertainty-guided 2:4 pruning framework designed for high-stakes medical tasks (e.g., melanoma classification on the ISIC 2024 challenge dataset) under extreme class imbalance.

---

## 📖 Overview

MDEP mimics biological processes of glial cells in the brain to dynamically adapt neural network topology:
*   **Microglia (Pruning)**: Signed first-order Taylor criteria prune connections expected to reduce epistemic-aleatoric uncertainty risk.
*   **Astrocytes (Regrowth)**: Evidence-driven local gradient saliency regrows connections in critical, under-represented features.
*   **Dynamic Exploration**: Anti-crystallization noise prevents structural gridlock during optimization.

Recent model updates integrate **Class-Balanced EDL (CB-EDL)** for extreme imbalance, **Detached OOD Projection (v2)** with Batch Normalization protection and dual-group gradient clipping, and strict **NVIDIA TensorRT 2:4 structured sparsity alignment** for verified hardware speedups.

For full mathematical descriptions, latest model updates, experiment protocols, and running guides, please see the sub-directory documentation:

👉 **[Experiments Suite Documentation (experiments/README.md)](file:///d:/MDEP/experiments/README.md)**

---

## 🛠️ Kaggle Quick Start

MDEP runs out-of-the-box on Kaggle. Clone and run:

```bash
# Clone the repository
%cd /kaggle/working
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
%cd MDEP-Microglial-Driven-Evidential-Pruning

# Smoke test on synthetic dummy data
!python experiments/run_kaggle_paper_suite.py --smoke

# Run the complete ISIC 2024 suite (all baselines & proposed models)
!python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

---

## 📊 Core Theoretical & Architectural Framework

### 1. Dirichlet Evidential Formulation
Instead of standard Softmax probabilities, MDEP outputs non-negative evidence vector $\mathbf{e} \ge 0$ parameterizing a Dirichlet distribution $\operatorname{Dir}(\bm{\alpha})$:
$$\alpha_c = e_c + 1, \quad S = \sum_{c=1}^K \alpha_c$$
*   **Epistemic Uncertainty (Vacuity)**: $u_e = \frac{K}{S}$
*   **Aleatoric Uncertainty (Ambiguity)**: $u_a = \sum_{c=1}^K \frac{\alpha_c}{S} \left[ \psi(S+1) - \psi(\alpha_c+1) \right]$

### 2. Signed First-Order Pruning Criterion
The pruning score $C_{ij}$ isolates connections whose removal is mathematically expected to reduce risk:
$$C_{ij} = \left[ w_{ij} \frac{\partial R}{\partial w_{ij}} \right]_+$$
where $R = \frac{u_a}{u_e + \epsilon}$.

---

## 🔄 Latest Model & Codebase Updates

### 💎 1. Class-Balanced EDL (CB-EDL)
Handles extreme class imbalance (malignant rate $<1\%$) using a learnable prior parameter $\beta$ and a class-pooling loss computed dynamically across active categories in each batch.

### 🛡️ 2. Detached OOD Projection (v2)
Out-of-Distribution (OOD) evaluation on datasets like PAD-UFES-20:
*   **RNG Isolation & BatchNorm Protection**: Feature extraction for OOD inputs is run strictly in `eval` mode under `torch.no_grad()`. This ensures that Outlier Exposure (OE) batches do not contaminate the running statistics (mean and variance) of the ID Batch Normalization layers.
*   **Independent Gradient Clipping Groups**: ID task gradients and auxiliary OOD projection head gradients are clipped independently in the trainer. This keeps the ID update trajectory completely invariant to the auxiliary domain classifier.

### ⚡ 3. NVIDIA TensorRT 2:4 Structured Sparsity Alignment
Pruning matrices are grouped strictly along the input channel axis ($C$) in $KCRS$ convolutional layouts. Every block of 4 weights contains exactly 2 zeros and 2 non-zeros along the reduction axis, allowing Level-3 TensorRT FP16 compiler optimization and verified hardware speedups.

---

## 📂 Active Runners

*   [`isic_paper_experiments.py`](file:///d:/MDEP/experiments/isic_paper_experiments.py): Main paper experiment runner.
*   [`run_group_kfold.py`](file:///d:/MDEP/experiments/run_group_kfold.py): Patient-grouped cross-validation.
*   [`backbone_generalization_runner.py`](file:///d:/MDEP/experiments/backbone_generalization_runner.py): Backbone sweeps (Swin-T, ConvNeXt, ResNet-18).
*   [`run_external_validation.py`](file:///d:/MDEP/experiments/run_external_validation.py): Validation on Fitzpatrick17k & PAD-UFES-20.
*   [`run_pad_adaptation.py`](file:///d:/MDEP/experiments/run_pad_adaptation.py): Leakage-safe adaptation on PAD-UFES-20.
*   [`run_calibration_study.py`](file:///d:/MDEP/experiments/run_calibration_study.py): Ablations of regrowth/calibration modes.
*   [`compare_rigl_guds.py`](file:///d:/MDEP/experiments/compare_rigl_guds.py): Regrowth comparison (RigL vs GUD).
*   [`export_sparse_acceleration.py`](file:///d:/MDEP/experiments/export_sparse_acceleration.py): ONNX and TensorRT preflight.
*   [`nvidia_sparse_benchmark.py`](file:///d:/MDEP/experiments/nvidia_sparse_benchmark.py): local TensorRT GPU sparse-kernel benchmarking.
*   [`run_complete_paper_suite.ipynb`](file:///d:/MDEP/run_complete_paper_suite.ipynb): Master Jupyter Notebook.

---

## 📜 Citation

If you find this work useful in your research, please cite:

```latex
@inproceedings{mdep2026,
  title={Microglial-Driven Evidential Pruning: Structured Sparsity for Imbalanced and High-Stakes Melanoma Classification},
  author={Rizvi, M. and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
