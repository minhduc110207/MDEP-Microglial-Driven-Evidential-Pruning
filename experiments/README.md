# GUDS-EDL Experiments Setup Guide

This folder contains the experimental pipeline for evaluating the **GUDS-EDL** (Generalized Uncertainty-Guided Dynamic Sparsification for Evidential Long-Tailed Learning) framework across different extreme imbalance and long-tailed benchmarks.

## 1. Prerequisites
Ensure your environment has the following installed:
- Python 3.8+
- PyTorch (with CUDA support)
- torchvision, numpy, pandas, scikit-learn, matplotlib, jupyter
- wandb (Optional, for online experiment tracking)

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib jupyter wandb
```

## 2. Supported Benchmarks & Runners

We provide targeted runners to evaluate GUDS-EDL on different benchmarks. Each runner handles dataset loading, model initialization, dynamic sparse training, bias-corrected temperature calibration, and final evaluation.

### Group A: Controlled Long-Tailed Recognition
- **`cifar_lt_runner.py`**: Runs CIFAR-100 with exponential class imbalance (Ratios 1:10, 1:50, 1:100). Automatically downloads the dataset if not present.

### Group B: Industrial Defect / Anomaly Detection
- **`mvtec_ad_runner.py`**: Simulates MVTec AD for binary rare-defect classification. Includes a fallback dummy data generator for dry-runs if the real MVTec dataset is not present.

### Group C: High-Stakes Rare-Event Case Study
- **ISIC 2024**: Evaluated via the main core file `../guds_edl_core.py`. Requires the ISIC dataset downloaded from Kaggle.

## 3. How to Run Experiments

### Kaggle From GitHub
After this repository has been pushed to GitHub, you do not need to upload the
whole codebase as a Kaggle Dataset. In a Kaggle notebook, clone the repository
into `/kaggle/working` and run the suite from there:

```bash
cd /kaggle/working
git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
cd MDEP-Microglial-Driven-Evidential-Pruning
python experiments/run_kaggle_paper_suite.py --smoke
```

For the full paper-facing run:

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

If the GitHub repository is private, add a Kaggle secret named `GITHUB_TOKEN`
with read access to the repository, then clone with the token inside the
notebook without printing it:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
```

See `experiments/KAGGLE_GITHUB_SETUP.md` for the complete Kaggle setup order.

### Kaggle Paper Suite (Recommended)
After cloning or copying the repository to `/kaggle/working`, run:

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

Before spending GPU hours, run a one-epoch smoke test:

```bash
python experiments/run_kaggle_paper_suite.py --smoke
```

The suite writes logs, checkpoints, and metrics under:

```text
paper_experiment_outputs/
```

By default, a full run uses seeds `42 43 44` for reproducibility. A smoke test
uses only seed `42`. To run a cheaper first full pass, explicitly pass one seed:

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --seeds 42 --no_save_model --keep_going
```

For ISIC-only table reproduction:

```bash
python experiments/isic_paper_experiments.py --suite main_tables
```

For all ISIC baselines and ablations:

```bash
python experiments/isic_paper_experiments.py --suite all
```

The complete planned CIFAR/MVTec baseline suites are run through:

```bash
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark mvtec --category hazelnut --epochs 20 --seeds 42 43 44
```

Hardware profiling and summary aggregation:

```bash
python experiments/hardware_profile.py
python experiments/summarize_results.py
```

### Automated Batch Mode (Recommended)
You can automatically execute the full suite of runners across datasets using the provided batch script:
1. Open PowerShell or Command Prompt.
2. Execute: `.\run_benchmarks.bat`

### Component Ablation Study
To investigate the contribution of individual GUDS-EDL components (Microglia Pruner, Astrocyte Regrower, Structural Baselines), we provide a dedicated Jupyter Notebook:
1. Open terminal in this folder.
2. Run `jupyter notebook ablation_experiments.ipynb`.
3. Follow the cells to train and evaluate the model under different ablation settings.

### Manual CLI Execution
You can also run any of the benchmarks manually from the terminal. The core and runners support argparse flags to toggle ablations.
```bash
# Full GUDS-EDL on CIFAR-100-LT
python cifar_lt_runner.py --imbalance_ratio 100 --epochs 100

# Ablation: Disable Astrocyte Regrowing (Topology Freezing)
python cifar_lt_runner.py --imbalance_ratio 100 --disable_regrower

# Ablation: Magnitude Pruning & Random Growth Baseline
python cifar_lt_runner.py --imbalance_ratio 100 --pruner_type magnitude --regrower_type random
```

## 4. Expected Outputs
During training and evaluation, the scripts will generate:
- `mdep_model.pth`: The raw trained dense-sparse weights.
- `resnet_calibrated_adaptive.pth`: The final model with calibrated temperature and adaptive thresholds (Balanced Utility, High-Recall, Quality-Gated).
- WandB Logs: If logged in, training metrics, dynamic $\gamma$ schedules, and evaluation metrics will sync to your Weights & Biases dashboard.
