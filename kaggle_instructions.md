# Kaggle Training Instructions For MDEP

This guide runs the active ISIC 2024 and CIFAR-100-LT experiments on Kaggle
without downloading datasets manually to your local machine.

## 1. Kaggle Notebook Settings

- Accelerator: GPU P100 preferred, T4 also works.
- Internet: On.
- Persistence: On.
- ISIC access: accept the ISIC 2024 Challenge rules once in your Kaggle account.

## 2. Setup Cell

Run this first in every Kaggle notebook. It clones or updates the repo, installs
lightweight dependencies, enables faster runtime defaults, and finds the ISIC
dataset automatically. If the dataset is mounted under `/kaggle/input`, it uses
that copy. Otherwise it downloads the competition data into Kaggle working
storage through the Kaggle API.

```python
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git"
REPO_DIR = Path("/kaggle/working/MDEP-Microglial-Driven-Evidential-Pruning")

def run(cmd, cwd=None):
    cmd = list(map(str, cmd))
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

if REPO_DIR.exists():
    run(["git", "pull", "--ff-only"], cwd=REPO_DIR)
else:
    run(["git", "clone", REPO_URL, str(REPO_DIR)])

run([
    sys.executable, "-m", "pip", "install", "-q",
    "scikit-learn", "matplotlib", "pandas", "h5py", "tqdm", "scipy"
])

os.environ.setdefault("MDEP_NUM_WORKERS", "4")
os.environ.setdefault("MDEP_PREFETCH_FACTOR", "4")
os.environ.setdefault("MDEP_CUDNN_BENCHMARK", "1")
os.environ.setdefault("MDEP_MATMUL_PRECISION", "high")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

def find_isic_root():
    candidates = [
        Path("/kaggle/input/isic-2024-challenge"),
        Path("/kaggle/input"),
        REPO_DIR / "data" / "isic-2024-challenge",
    ]
    for base in candidates:
        if not base.exists():
            continue
        if (base / "train-metadata.csv").exists():
            return base
        for path in base.rglob("train-metadata.csv"):
            return path.parent
    return None

isic_root = find_isic_root()
if isic_root is None:
    isic_root = REPO_DIR / "data" / "isic-2024-challenge"
    isic_root.mkdir(parents=True, exist_ok=True)
    run(["kaggle", "competitions", "download", "-c", "isic-2024-challenge", "-p", isic_root])
    for archive in isic_root.glob("*.zip"):
        run(["unzip", "-q", "-n", archive, "-d", isic_root])

os.environ["ISIC_ROOT"] = str(isic_root)
os.chdir(REPO_DIR)
print("Repo:", REPO_DIR)
print("ISIC_ROOT:", os.environ["ISIC_ROOT"])
run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR)
print("SETUP COMPLETE. Run the experiment command in a new cell.", flush=True)
```

## 3. Smoke And Syntax Checks

Run this before spending GPU hours:

```python
!python experiments/run_kaggle_paper_suite.py --smoke --no_save_model
```

For a syntax-only check in Kaggle:

```python
!python -m py_compile guds_edl_core.py experiments/*.py
```

## 4. Do Not Start With The Largest Run

The all-in-one launcher defaults to:

- ISIC `--isic_suite all`
- CIFAR ratios `10 50 100`
- seeds `42 43 44` unless `--smoke` is used
- CIFAR full planned suite when no `--experiment` is selected

That is a very large workload. Use focused commands first, then expand.

## 5. Recommended First Real Runs

### ISIC Main Table, One Seed

```python
!python experiments/isic_paper_experiments.py \
    --suite main_tables \
    --epochs 40 \
    --batch_size 32 \
    --seeds 42 \
    --no_save_model
```

### CIFAR One Experiment, One Seed

Use `standard_ce` for a fast pipeline sanity check:

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment standard_ce \
    --epochs 1 \
    --batch_size 128 \
    --seeds 42
```

Use `full_guds` for the slow sparse code path:

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --epochs 5 \
    --batch_size 128 \
    --seeds 42
```

If the notebook shows only clone/setup lines for several minutes, it has not
entered the CIFAR runner yet. Stop the cell, make sure the setup cell ends with
`SETUP COMPLETE`, then run the CIFAR command in a separate cell.

Optional CIFAR download preflight:

```python
!python -u -c "import torchvision; root='/kaggle/working/cifar_data'; print('CIFAR root:', root, flush=True); torchvision.datasets.CIFAR100(root=root, train=True, download=True); torchvision.datasets.CIFAR100(root=root, train=False, download=True); print('CIFAR download ready', flush=True)"
```

## 6. Split Full Training Across Notebooks

Run separate Kaggle notebooks/sessions for each group when wall-clock time
matters.

### ISIC Softmax Baselines

```python
!python experiments/run_isic_softmax_baselines.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope train \
    --seeds 42 43 44 \
    --no_save_model
```

### ISIC Evidential Baselines

```python
!python experiments/run_isic_evidential_baselines.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope train \
    --seeds 42 43 44 \
    --no_save_model
```

### ISIC GUDS-EDL And Ablations

```python
!python experiments/run_isic_guds_ablations.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope train \
    --seeds 42 43 44 \
    --no_save_model
```

### CIFAR-100-LT

Run one imbalance ratio per notebook. If you only need selected methods, repeat
`--experiment` and avoid the full 14-model suite.

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --experiment standard_ce \
    --experiment dense_edl \
    --epochs 100 \
    --batch_size 128 \
    --seeds 42 43 44
```

Full CIFAR sweeps are expensive because each command runs every planned model
when `--experiment` is omitted:

```python
!python -u experiments/run_cifar_suite.py --ratio 100 --epochs 100 --batch_size 128 --seeds 42 43 44
!python -u experiments/run_cifar_suite.py --ratio 50 --epochs 100 --batch_size 128 --seeds 42 43 44
!python -u experiments/run_cifar_suite.py --ratio 10 --epochs 100 --batch_size 128 --seeds 42 43 44
```

## 7. Why Seed 42 Can Look Slow

ISIC training is not finished when the final epoch prints. After epoch 40, the
runner still performs calibration, adaptive-mode evaluation, quality-gate
reporting, extended metrics, and result writing. This is especially visible for
seed 42 because the split includes large validation, calibration, and test
loaders. Wait for the final `[DONE]` line before assuming the run is stuck.

Use `--no_save_model` for broad ISIC sweeps. It avoids checkpoint writes and
keeps Kaggle storage smaller.

## 8. CIFAR Speed Notes

- `run_cifar_suite.py` without `--experiment` runs all 14 planned CIFAR models.
- A healthy CIFAR run prints `Running CIFAR-100-LT Generalization Suite`, then
  `[RUN]`, then `[CIFAR] Downloading/validating ...` before training.
- Sparse methods such as `full_guds`, `static_24_edl`, and `rigl_style_24` are
  slower than dense baselines because they update structural masks during
  training.
- Use `--experiment standard_ce --epochs 1` for a fast data/training sanity
  check.
- Use `--experiment full_guds --epochs 5` when checking the slow sparse path.

## 9. All-In-One Launcher

Use this only after the focused runs work:

```python
!python experiments/run_kaggle_paper_suite.py \
    --isic_suite all \
    --no_save_model \
    --keep_going
```

For a cheaper all-in-one pass:

```python
!python experiments/run_kaggle_paper_suite.py \
    --isic_suite main_tables \
    --cifar_ratios 100 \
    --seeds 42 \
    --no_save_model \
    --keep_going
```

To run ISIC only:

```python
!python experiments/run_kaggle_paper_suite.py \
    --skip_cifar \
    --skip_hardware \
    --isic_suite main_tables \
    --seeds 42 \
    --no_save_model
```

## 10. Speed And Storage Knobs

- Keep `MDEP_NUM_WORKERS=4` and `MDEP_PREFETCH_FACTOR=4` on Kaggle. Try `6`
  workers only if CPU/RAM usage is stable.
- Keep `MDEP_CUDNN_BENCHMARK=1` because image inputs are fixed-size.
- Use `--no_save_model` for ISIC broad sweeps unless checkpoints are required.
- For CIFAR, checkpoints are already off by default. Add `--save_model` only
  when needed.
- Use one seed first, then launch multi-seed runs.
- Split ISIC softmax, ISIC evidential, ISIC ablations, and each CIFAR ratio into
  separate notebooks for faster wall-clock completion.
- If data loading stalls, retry with `MDEP_NUM_WORKERS=2`.

## 11. Local Full Experiment Runner

Use the local runner when training from this Windows workspace. CIFAR-100-LT
downloads automatically through `torchvision`. Real ISIC training still needs a
local ISIC folder, either through `ISIC_ROOT` or `--isic_root`.

Install the local logging dependency once:

```powershell
pip install wandb
```

The runner uses W&B in offline mode by default. It saves local W&B files under
`paper_experiment_outputs/wandb/`, raw sub-run logs under
`paper_experiment_outputs/local_logs/<timestamp>/`, and a local parameter
manifest named `local_run_config.json`.

### Local Smoke Test With Real ISIC

```powershell
$env:ISIC_ROOT="D:\datasets\isic-2024-challenge"
python experiments/run_local_full_experiments.py `
    --smoke `
    --isic_root $env:ISIC_ROOT `
    --no_save_model `
    --wandb_mode offline `
    --wandb_project mdep-local-experiments
```

### Local CIFAR-Only Run

Use this when you do not have ISIC locally yet:

```powershell
python experiments/run_local_full_experiments.py `
    --skip_isic `
    --cifar_ratios 100 `
    --cifar_epochs 1 `
    --seeds 42 `
    --skip_hardware `
    --wandb_mode offline `
    --wandb_project mdep-local-experiments
```

### Local Full Run

```powershell
$env:ISIC_ROOT="D:\datasets\isic-2024-challenge"
python experiments/run_local_full_experiments.py `
    --isic_root $env:ISIC_ROOT `
    --isic_suite all `
    --cifar_ratios 10 50 100 `
    --epochs 40 `
    --cifar_epochs 100 `
    --batch_size 32 `
    --seeds 42 43 44 `
    --no_save_model `
    --keep_going `
    --wandb_mode offline `
    --wandb_project mdep-local-experiments
```

### Local Dry-Run Without ISIC

This checks wiring only. Do not use dummy data for paper results.

```powershell
python experiments/run_local_full_experiments.py `
    --smoke `
    --allow_dummy_data `
    --no_save_model `
    --wandb_mode offline
```

To disable W&B completely:

```powershell
python experiments/run_local_full_experiments.py --smoke --wandb_mode disabled
```

## 12. Outputs

All runners write under:

```text
paper_experiment_outputs/
```

Aggregate and zip results:

```python
!python experiments/summarize_results.py
!zip -r /kaggle/working/mdep_results.zip paper_experiment_outputs/ -x "*.pth"
```

Download `mdep_results.zip` from the Kaggle notebook Output panel.

## 13. Common Fixes

- If ISIC download fails, confirm the competition rules are accepted and
  Internet is enabled.
- If GPU memory runs out, reduce ISIC `--batch_size` to `16` or CIFAR
  `--batch_size` to `64`.
- If Kaggle shows only setup lines such as `git rev-parse --short HEAD`, stop
  the cell and rerun setup separately. The setup cell should end with
  `SETUP COMPLETE`.
- If a run appears frozen after the final epoch, check the logs for `[CAL]`,
  `[EVAL]`, adaptive-mode messages, and finally `[DONE]`.
- If local W&B is not installed, either run `pip install wandb` or add
  `--wandb_mode disabled`.
