# MDEP: Microglial-Driven Evidential Pruning (Experiments Suite)

This directory contains the active runners, validation scripts, and optimization pipelines for the **MDEP (Microglial-Driven Evidential Pruning)** framework. MDEP is a structured, uncertainty-guided 2:4 pruning framework designed for high-stakes clinical tasks like melanoma classification (ISIC 2024) under extreme class imbalance.

---

## 🚀 Key Mathematical & Architectural Foundations

MDEP integrates **Evidential Deep Learning (EDL)** with dynamic structural sparsity modeled after glial cell activities in the brain (Microglia for pruning, Astrocytes for regrowth).

### 1. Dirichlet Evidential Formulation
Instead of standard Softmax probabilities, the model outputs non-negative evidence vector $\mathbf{e} \ge 0$ via a Softplus activation. This evidence parameterizes a Dirichlet distribution $\operatorname{Dir}(\bm{\alpha})$:
$$\alpha_c = e_c + 1, \quad S = \sum_{c=1}^K \alpha_c$$
*   **Epistemic Uncertainty (Vacuity)**: $u_e = \frac{K}{S}$ (represents overall lack of evidence).
*   **Aleatoric Uncertainty (Ambiguity)**: $u_a = \sum_{c=1}^K \frac{\alpha_c}{S} \left[ \psi(S+1) - \psi(\alpha_c+1) \right]$ (represents conflicting evidence, where $\psi$ is the digamma function).

### 2. Microglial-Guided Pruning
Pruning is formulated as a signed first-order Taylor criterion on the evidential risk ratio $R = \frac{u_a}{u_e + \epsilon}$. The pruning score $C_{ij}$ isolates connections whose removal is mathematically expected to reduce risk:
$$C_{ij} = \left[ w_{ij} \frac{\partial R}{\partial w_{ij}} \right]_+$$
*   **Positive Gradient ($w_{ij} \frac{\partial R}{\partial w_{ij}} > 0$)**: Pruning this connection is expected to decrease risk. These are targeted for pruning.
*   **Negative Gradient ($w_{ij} \frac{\partial R}{\partial w_{ij}} < 0$)**: Pruning this connection is expected to increase risk. These are preserved.

### 3. Astrocyte-Guided Regrowth & Anti-Crystallization
Dormant weights are dynamically regrown using a local first-order saliency score derived from a regrowth objective $L_{\text{grow}}$:
$$G_{ij} = \left| \frac{\partial L_{\text{grow}}}{\partial w_{ij}} \right|$$
To prevent structural crystallization (where topology gets stuck in a local minimum and growth signals freeze), an **Anti-Crystallization stochastic noise** term is introduced when structural gradients decay:
$$\tilde{G}_{ij} = G_{ij} + \xi_{ij} \cdot \sigma(\mathbf{V}^{(l)}), \quad \xi_{ij} \sim \mathcal{N}(0, 1)$$

---

## 🔄 Latest Model & Codebase Updates

The repository has recently been updated with the following features to improve classification under extreme class imbalance, out-of-distribution (OOD) safety, and hardware compatibility:

### 💎 1. Class-Balanced EDL (CB-EDL)
Designed specifically to handle extreme class imbalance (such as in ISIC 2024, where the positive malignant rate is $<1\%$).
*   **Pooling Loss & Learnable Prior**: CB-EDL replaces static uniform priors with a learnable prior parameter $\beta$ and introduces a class-pooling loss computed dynamically across active categories in each batch to prevent rare-class gradients from being overwhelmed by majority classes.
*   **Evidential Focal Loss (EFL) warmup**: Modulates expected cross-entropy without distorting the Dirichlet structure, using a cosine-decayed gamma factor that is scheduled post-warmup.

### 🛡️ 2. Detached OOD Projection (v2)
Out-of-Distribution (OOD) evaluation on datasets like PAD-UFES-20 has been upgraded to prevent data contamination and training instability:
*   **RNG Isolation & BatchNorm Protection**: Feature extraction for OOD inputs is run strictly in `eval` mode (`self.model.eval()`) under `torch.no_grad()`. This ensures that Outlier Exposure (OE) batches do not contaminate the running statistics (mean and variance) of the In-Distribution (ID) Batch Normalization layers.
*   **Independent Gradient Clipping Groups**: ID task gradients and auxiliary OOD projection head gradients are clipped independently in the trainer. This prevents large auxiliary domain-classification gradients from rescaling and distorting the optimization trajectory of the primary task.

### ⚡ 3. NVIDIA TensorRT 2:4 Structured Sparsity Alignment
*   **Layout Correction**: Custom pruning matrices are now grouped strictly along the input channel axis ($C$) in $KCRS$ convolutional layouts (using the `nvidia_kcrs` layout and `nvidia_v3` profile).
*   **TensorRT Eligibility**: Every block of 4 weights contains exactly 2 zeros and 2 non-zeros along the reduction axis. This satisfies Ampere/Ada Lovelace/Hopper GPU hardware sparse Tensor Core constraints, allowing Level-3 TensorRT FP16 compiler optimization and verified speedups.

---

## 📂 Active Runners and Scripts

The `experiments/` directory contains the following runners and validation utilities:

| Script / Notebook | Purpose & Description |
| :--- | :--- |
| [`isic_paper_experiments.py`](file:///d:/MDEP/experiments/isic_paper_experiments.py) | Main runner for ISIC 2024 baselines, evidential models, proposed GUDS-EDL, and ablations. |
| [`run_group_kfold.py`](file:///d:/MDEP/experiments/run_group_kfold.py) | Evaluates MDEP under patient-grouped nested cross-validation to prevent data leakage. |
| [`backbone_generalization_runner.py`](file:///d:/MDEP/experiments/backbone_generalization_runner.py) | Tests structural sparsity on diverse backbones (ResNet-18, Swin Transformer, ConvNeXt). |
| [`run_external_validation.py`](file:///d:/MDEP/experiments/run_external_validation.py) | Evaluates zero-shot validation, fairness, and OOD performance on Fitzpatrick17k & PAD-UFES-20. |
| [`run_pad_adaptation.py`](file:///d:/MDEP/experiments/run_pad_adaptation.py) | Performs leakage-safe domain adaptation on PAD-UFES-20 using frozen features. |
| [`run_calibration_study.py`](file:///d:/MDEP/experiments/run_calibration_study.py) | Ablation runner evaluating regrowth modes (KL, Vacuity, Ambiguity, Ratio) and ECE. |
| [`compare_rigl_guds.py`](file:///d:/MDEP/experiments/compare_rigl_guds.py) | Directly compares evidential regrowth versus standard gradient-based regrowth (RigL). |
| [`export_sparse_acceleration.py`](file:///d:/MDEP/experiments/export_sparse_acceleration.py) | Exports sparse MDEP check-pointed models to ONNX and runs hardware acceleration preflights. |
| [`nvidia_sparse_benchmark.py`](file:///d:/MDEP/experiments/nvidia_sparse_benchmark.py) | Level-3 TensorRT FP16 benchmarking suite on local hardware (RTX A2000). |
| [`run_complete_paper_suite.ipynb`](file:///d:/MDEP/run_complete_paper_suite.ipynb) | Master notebook in repo root that runs the entire training and baseline pipeline. |

---

## 🛠️ Kaggle Quick Start

MDEP is optimized for running on Kaggle with single-click notebook cells.

```bash
# Clone the repository
%cd /kaggle/working
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
%cd MDEP-Microglial-Driven-Evidential-Pruning

# Run a quick smoke test on synthetic/dummy data
!python experiments/run_kaggle_paper_suite.py --smoke

# Run the complete ISIC 2024 experiment suite (all models, 3 seeds)
!python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

---

## 📈 Running Aligned Evaluation Protocols

> [!IMPORTANT]
> To reproduce paper-facing benchmarks, execute the commands below. Ensure you use the exact seed parameters to match the paper-readiness gates.

### 1. Main Table Baselines (ISIC 2024)
```bash
MDEP_DETERMINISTIC=1 python -u experiments/isic_paper_experiments.py \
  --suite main_tables \
  --epochs 40 \
  --batch_size 32 \
  --lr 4e-5 \
  --seeds 42 123 456 \
  --split_seed 42 \
  --subsample_scope train \
  --subsample_ratio 20 \
  --structural_proxy_batches 4 \
  --checkpoint_selection last \
  --run_suffix _fair_v3_nvidia24
```

### 2. Held-Out External OOD Evaluation (PAD-UFES-20)
Evaluate the checkpoints generated above on the unseen skin-cancer domain:
```bash
for seed in 42 123 456; do
  python -u experiments/run_external_validation.py \
    --model_path "/kaggle/working/paper_experiment_outputs/isic/full_guds_fair_v3_nvidia24/seed_${seed}/model_state.pth" \
    --seed "${seed}" \
    --split_seed 42 \
    --custom_image_folder /kaggle/input/datasets/mahdavi1202/skin-cancer \
    --pad_ufes_csv /kaggle/input/datasets/mahdavi1202/skin-cancer/metadata.csv \
    --pad_ufes_partition imgs_part_3 \
    --knn_primary_layer layer3 \
    --primary_ood_score knn_layer3
done
```

### 3. CIFAR-100-LT Generalization benchmarks
```bash
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --seeds 42 43 44
```

---

## 💾 Outputs & Artifacts

All training runs write their configurations, execution logs, JSON metrics, and model checkpoints to:
`paper_experiment_outputs/isic/`

You can aggregate results across seeds at any point using:
```bash
python experiments/summarize_results.py
```
