# Kaggle Setup From GitHub

Use this when the code lives on GitHub and the experiments run inside a Kaggle
notebook.

## 1. Notebook Settings

- Accelerator: GPU P100 or T4.
- Internet: on.
- Persistence: on.

## 2. Clone And Install

```python
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git"
REPO_DIR = Path("/kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning")

def run(cmd, cwd=None):
    cmd = list(map(str, cmd))
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

if REPO_DIR.exists():
    run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
else:
    run(["git", "clone", REPO_URL, str(REPO_DIR)])

run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "matplotlib", "pandas", "h5py", "tqdm", "scipy"])

os.environ.setdefault("MDEP_NUM_WORKERS", "4")
os.environ.setdefault("MDEP_PREFETCH_FACTOR", "4")
os.chdir(REPO_DIR)
```

## 3. Dataset Setup

The runners first search `/kaggle/input`, then repo-local `data/` paths. The
fastest path is to attach the ISIC 2024 competition dataset as a Kaggle input.
If it is not attached, use the Kaggle API after accepting the competition rules:

```python
from pathlib import Path

DATA_DIR = Path("/kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning/data/isic-2024-challenge")
if not (DATA_DIR / "train-metadata.csv").exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "competitions", "download", "-c", "isic-2024-challenge", "-p", DATA_DIR])
    for archive in DATA_DIR.glob("*.zip"):
        run(["unzip", "-q", "-n", archive, "-d", DATA_DIR])
os.environ["ISIC_ROOT"] = str(DATA_DIR)
```

CIFAR-100 is downloaded automatically by `torchvision` into `./data`.

## 4. Smoke Test

```bash
!python experiments/run_kaggle_paper_suite.py --smoke --no_save_model
```

## 5. Full Runs

```bash
!python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

CIFAR-only ratios:

```bash
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --seeds 42 43 44
```

## 6. Outputs

Metrics and logs are written under `paper_experiment_outputs/`. Use:

```bash
!python experiments/summarize_results.py
!zip -r /kaggle/working/mdep_results.zip paper_experiment_outputs/
```
