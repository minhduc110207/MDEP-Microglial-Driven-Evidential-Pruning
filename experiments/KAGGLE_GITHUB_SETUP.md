# Kaggle Setup From GitHub

This guide is the recommended way to run the paper experiments on Kaggle once
the repository is available on GitHub. The code should come from GitHub; the
datasets should be attached through Kaggle inputs.

## 1. Create the Kaggle Notebook

1. Create a new Kaggle notebook.
2. Set accelerator to GPU.
3. Add the required datasets under **Add Input**:
   - ISIC 2024 training metadata and images for the main paper experiments.
   - MVTec AD only if you want to run the planned industrial anomaly protocol.
   - CIFAR-100 is downloaded automatically by `torchvision`, so it usually does
     not need a Kaggle input dataset.

## 2. Clone the Repository

For a public GitHub repository, run this in the first notebook cell:

```bash
%cd /kaggle/working
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
%cd MDEP-Microglial-Driven-Evidential-Pruning
```

For a private repository, create a Kaggle secret named `GITHUB_TOKEN` with read
access to the repository, then run:

```python
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
%cd MDEP-Microglial-Driven-Evidential-Pruning
```

Do not print the token in notebook output.

## 3. Install Runtime Dependencies

Kaggle normally includes PyTorch, torchvision, numpy, pandas, scikit-learn, and
matplotlib. If a package is missing, install only the missing package:

```bash
!pip install -q scikit-learn matplotlib pandas
```

## 4. Verify the Code Path

Run a smoke test before launching the full experiment suite:

```bash
!python experiments/run_kaggle_paper_suite.py --smoke
```

The smoke test is the fastest check that imports, dataset discovery, output
folders, and the core training loop are wired correctly.

## 5. Run the Paper-Facing Suite

Run all ISIC baselines and ablations described by the paper-facing experiment
map:

```bash
!python experiments/run_kaggle_paper_suite.py --isic_suite all
```

The outputs are written to:

```text
paper_experiment_outputs/
```

## 6. Optional Protocols

Run the planned CIFAR-100-LT checks:

```bash
!python experiments/cifar_lt_runner.py --imbalance_ratio 10 --epochs 100
!python experiments/cifar_lt_runner.py --imbalance_ratio 50 --epochs 100
!python experiments/cifar_lt_runner.py --imbalance_ratio 100 --epochs 100
```

Run MVTec AD after attaching a real MVTec dataset:

```bash
!python experiments/mvtec_ad_runner.py --category hazelnut --epochs 20
!python experiments/mvtec_ad_runner.py --category bottle --epochs 20
```

If no real MVTec category is found, the runner falls back to dummy tensors and
the result should be treated only as a pipeline smoke test.

## 7. Updating Code During a Kaggle Run

If you push fixes to GitHub while the Kaggle notebook is open, update the local
copy with:

```bash
%cd /kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning
!git pull
```

Then rerun the smoke test before the full run.
