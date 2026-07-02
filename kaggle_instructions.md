# Kaggle Training Instructions For MDEP

This guide runs the active ISIC 2024 and CIFAR-100-LT experiments on Kaggle
without downloading datasets manually to your local machine.

## 1. Kaggle Notebook Settings

- Accelerator: GPU P100 preferred, T4 also works.
- Internet: On.
- Persistence: On.
- ISIC access: accept the ISIC 2024 Challenge rules once in your Kaggle account.
- CIFAR-100 access: Add the `CIFAR-100 Python` dataset from Kaggle to your notebook via the "Add Data" button.

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

def link_cifar_dataset():
    candidates = [
        Path("/kaggle/input/cifar-100-python"),
        Path("/kaggle/input/cifar100"),
        Path("/kaggle/input/cifar-100"),
    ]
    cifar_source = None
    for base in candidates:
        if (base / "train").exists() and (base / "meta").exists():
            cifar_source = base
            break
            
    if cifar_source is not None:
        cifar_target_dir = REPO_DIR / "data"
        cifar_target = cifar_target_dir / "cifar-100-python"
        cifar_target_dir.mkdir(parents=True, exist_ok=True)
        if not cifar_target.exists():
            try:
                os.symlink(cifar_source, cifar_target)
                print("Linked CIFAR-100 from:", cifar_source, flush=True)
            except OSError as e:
                print("Warning: Could not symlink CIFAR-100:", e, flush=True)

link_cifar_dataset()

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
- seeds `42 123 456` unless `--smoke` is used
- ISIC `--split_seed 42`, so all model seeds share the same patient-level
  train/validation/calibration/test split
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
    --split_seed 42 \
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
    --seeds 42 123 456 \
    --split_seed 42 \
    --no_save_model
```

### ISIC Evidential Baselines

```python
!python experiments/run_isic_evidential_baselines.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope train \
    --seeds 42 123 456 \
    --split_seed 42 \
    --no_save_model
```

### ISIC GUDS-EDL And Ablations

```python
!python experiments/run_isic_guds_ablations.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope train \
    --seeds 42 123 456 \
    --split_seed 42 \
    --no_save_model
```

`run_isic_guds_ablations.py` lists `full_guds` first. Because the ISIC runner
now loops experiment-first and seed-second, the first three ablation jobs are
`full_guds` with model seeds `42`, `123`, and `456` on the fixed
`split_seed=42` split.

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
    --seeds 42 123 456
```

Full CIFAR sweeps are expensive because each command runs every planned model
when `--experiment` is omitted:

```python
!python -u experiments/run_cifar_suite.py --ratio 100 --epochs 100 --batch_size 128 --seeds 42 123 456
!python -u experiments/run_cifar_suite.py --ratio 50 --epochs 100 --batch_size 128 --seeds 42 123 456
!python -u experiments/run_cifar_suite.py --ratio 10 --epochs 100 --batch_size 128 --seeds 42 123 456
```

### CIFAR-100-LT Improvement Roadmap

Goal: improve GUDS-EDL on CIFAR-100-LT without turning the paper into a list of
many GUDS versions. Treat all commands below as selection diagnostics. Promote
only one final GUDS configuration after it wins on model seeds `42`, `123`, and
`456` under the fixed `--split_seed 42` protocol.

Use these decision metrics in this order:

1. `balanced_accuracy`
2. `macro_auroc`
3. `macro_pr_auc`
4. `aurc` lower is better
5. `ece_adaptive` lower is better

The CIFAR runner enables best validation checkpoint selection by default for
all methods. The shared selection rule is validation balanced accuracy, then
validation Macro AUROC, then lower validation AURC. The selected checkpoint is
restored before calibration and held-out test evaluation. This is fair only
when applied to every compared CIFAR method. To save memory, the runner keeps
only one CPU copy of the best `state_dict`; it does not write a checkpoint at
every epoch. Use `--no_best_checkpoint` only for debugging final-epoch behavior.
Use `--checkpoint_eval_every N` only if the same value is used for all compared
methods.

The default CIFAR `full_guds` profile is the strongest current diagnostic
configuration: `lr=5e-4`, `structural_proxy_batches=16`,
`structural_proxy_min_classes=40`, `efl_gamma_final=1.0`, and best validation
checkpoint selection. Passing any of these flags manually overrides the default.

#### Phase 0: one-seed sanity check

Run this first to confirm the CIFAR path, result writing, and fixed split:

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --epochs 5 \
    --batch_size 128 \
    --seeds 42 \
    --split_seed 42 \
    --run_suffix _smoke
```

#### Phase 1: rebuild the current comparison on a fixed CIFAR split

This is the minimum rerun before judging whether a new GUDS variant really
improves over the reported result.

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment standard_ce \
    --experiment dense_edl \
    --experiment static_24_edl \
    --experiment rigl_style_24 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --seeds 42 123 456 \
    --split_seed 42

!python experiments/summarize_results.py \
    --root /kaggle/working/paper_experiment_outputs \
    --output /kaggle/working/paper_experiment_outputs/cifar_ir100_current_summary.csv
```

#### Phase 2: high-probability GUDS structural tuning

The first likely bottleneck is noisy topology selection on a 100-class
long-tailed dataset. Increase the structural proxy batch and require more class
diversity before each structural update.

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --seeds 42 \
    --split_seed 42 \
    --structural_proxy_batches 16 \
    --structural_proxy_min_classes 40 \
    --run_suffix _proxy16_min40

!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --seeds 42 \
    --split_seed 42 \
    --structural_proxy_batches 24 \
    --structural_proxy_min_classes 50 \
    --run_suffix _proxy24_min50
```

Summarize after the one-seed tuning pass:

```python
!python experiments/summarize_results.py \
    --root /kaggle/working/paper_experiment_outputs \
    --output /kaggle/working/paper_experiment_outputs/cifar_ir100_tuning_seed42_summary.csv
```

Keep only the better of `_proxy16_min40` and `_proxy24_min50` for the next phase.

#### Phase 3: optimizer stability sweep for the best structural candidate

If seed 42 improves but AURC/ECE is unstable, try smaller learning rates. Replace
`_proxy16_min40` below with `_proxy24_min50` if Phase 2 picked that candidate.

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --lr 5e-4 \
    --seeds 42 \
    --split_seed 42 \
    --structural_proxy_batches 16 \
    --structural_proxy_min_classes 40 \
    --run_suffix _proxy16_min40_lr5e-4

!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --lr 3e-4 \
    --seeds 42 \
    --split_seed 42 \
    --structural_proxy_batches 16 \
    --structural_proxy_min_classes 40 \
    --run_suffix _proxy16_min40_lr3e-4
```

#### Phase 4: three-seed confirmation

Run the best candidate from Phases 2--3 on all three model seeds. This is the
only result eligible to replace the CIFAR table in the paper.

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --seeds 42 123 456 \
    --split_seed 42

!python experiments/summarize_results.py \
    --root /kaggle/working/paper_experiment_outputs \
    --output /kaggle/working/paper_experiment_outputs/cifar_ir100_candidate_final_summary.csv
```

Promotion rule: replace the paper's CIFAR GUDS row only if
the current `full_guds` improves over the previous reported `full_guds` on at least
`balanced_accuracy` and `macro_auroc`, while preserving exact 2:4 structural
validity. If it improves only ECE, keep CIFAR as a limitation/stress test.

#### Phase 4b: focal-floor ablation only if needed

The late-phase focal floor is now part of the default CIFAR `full_guds` profile.
Use this ablation only to verify that the floor is helping. It should not be
reported as a separate model.

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 100 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --efl_gamma_final 0.0 \
    --seeds 42 \
    --split_seed 42 \
    --run_suffix _no_gamma_floor
```

If this ablation performs better than the default strong profile, revisit the
default. Otherwise keep the strong profile as the single `full_guds` setting.

#### Phase 5: broaden only after IR100 improves

Do not spend budget on IR50/IR10 until IR100 improves. Once the candidate is
confirmed on IR100:

```python
!python -u experiments/run_cifar_suite.py \
    --ratio 50 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --lr 5e-4 \
    --seeds 42 123 456 \
    --split_seed 42 \
    --structural_proxy_batches 16 \
    --structural_proxy_min_classes 40 \
    --run_suffix _candidate_final

!python -u experiments/run_cifar_suite.py \
    --ratio 10 \
    --experiment full_guds \
    --epochs 100 \
    --batch_size 128 \
    --lr 5e-4 \
    --seeds 42 123 456 \
    --split_seed 42 \
    --structural_proxy_batches 16 \
    --structural_proxy_min_classes 40 \
    --run_suffix _candidate_final
```

#### Phase 6: next code task if tuning is not enough

If the best candidate still trails Standard CE, the next real improvement path
is a two-stage CIFAR recipe:

1. train a dense long-tail teacher (`standard_ce` or `balanced_softmax`);
2. initialize GUDS from that checkpoint;
3. keep a longer dense warmup;
4. activate 2:4 GUDS only for fine-tuning.

Do not report this as the same GUDS configuration until the code path is added,
rerun for all seeds, and compared against the same baselines.

## 7. Why ISIC Can Look Slow After Training

ISIC training is not finished when the final epoch prints. After epoch 40, the
runner still performs calibration, adaptive-mode evaluation, quality-gate
reporting, extended metrics, held-out prediction CSV export, and result writing. With the fixed
`--split_seed 42` protocol, model seeds `42`, `123`, and `456` use the same
validation, calibration, and test loaders; runtime differences mostly come from
training dynamics and sparse structural updates, not from different data splits.
Wait for the final `[DONE]` line before assuming the run is stuck.

Use `--no_save_model` for broad ISIC sweeps. It avoids checkpoint writes and
keeps Kaggle storage smaller.

After the three-seed ISIC main-table runs finish, compute paired bootstrap
diagnostics from the saved `test_predictions.csv` files:

```python
!python experiments/bootstrap_isic_predictions.py \
    --root /kaggle/working/paper_experiment_outputs \
    --target full_guds \
    --baseline dense_edl static_24_edl rigl_style_24 \
    --unit patient \
    --n_bootstrap 1000
```

If patient-level bootstrap is too slow, use `--unit image` and report it only as
an image-level diagnostic.

### High-Yield GUDS Tuning Queue

Do not mix these tuning outputs into the reported `full_guds` folder. Use
`--run_suffix` so each candidate is isolated, then promote a candidate only if
it wins consistently over seeds 42, 123, and 456.

```python
# Better structural signal estimate; higher cost, often the first thing to try.
!python experiments/isic_paper_experiments.py \
    --experiment full_guds \
    --run_suffix _proxy8 \
    --structural_proxy_batches 8 \
    --epochs 40 \
    --batch_size 32 \
    --lr 4e-5 \
    --subsample_scope train \
    --seeds 42 123 456 \
    --split_seed 42 \
    --no_save_model

# Slightly more conservative optimizer step; useful if pAUC varies by seed.
!python experiments/isic_paper_experiments.py \
    --experiment full_guds \
    --run_suffix _proxy8_lr3e-5 \
    --structural_proxy_batches 8 \
    --epochs 40 \
    --batch_size 32 \
    --lr 3e-5 \
    --subsample_scope train \
    --seeds 42 123 456 \
    --split_seed 42 \
    --no_save_model
```

After a tuning run, compare it with the reported baselines:

```python
!python experiments/bootstrap_isic_predictions.py \
    --target full_guds_proxy8 \
    --baseline dense_edl static_24_edl rigl_style_24 \
    --unit patient \
    --n_bootstrap 1000
```

Keep the manuscript as one reported GUDS-EDL model. Treat tuning runs as
selection diagnostics until a single replacement configuration is fully
rerun for all reported baselines and ablations.

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
    --split_seed 42 \
    --no_save_model \
    --keep_going
```

For a cheaper all-in-one pass:

```python
!python experiments/run_kaggle_paper_suite.py \
    --isic_suite main_tables \
    --cifar_ratios 100 \
    --seeds 42 \
    --split_seed 42 \
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
    --split_seed 42 \
    --no_save_model
```

For the minimum Level 1 and Level 2 hardware/sparsity evidence used in the
paper, run:

```python
!python experiments/hardware_profile.py --batch_size 32 --warmup 10 --iters 50
!python experiments/summarize_results.py
```

This records active density, valid 2:4 block fraction, theoretical wrapped-layer
sparse reduction, dense-execution GMACs/GFLOPs, theoretical active GMACs/GFLOPs,
per-image GMACs/GFLOPs, masked-PyTorch throughput, latency, and peak CUDA memory
for dense, static 2:4, and DST-EDL model constructors. Use the per-image fields
for paper tables because they are independent of batch size.

For a stronger diagnostic, repeat with a larger batch size if the GPU memory is
stable:

```python
!python experiments/hardware_profile.py --batch_size 64 --warmup 20 --iters 100
!python experiments/summarize_results.py
```

The hardware profile reports structural compatibility, theoretical MAC/FLOP
reduction, and masked-PyTorch throughput only. Do not describe it as realized
sparse Tensor Core speedup. Real speedup requires exporting frozen valid 2:4
weights to a sparse-kernel backend such as cuSPARSELt, TensorRT sparse
execution, or a supported PyTorch semi-structured sparse path. If that export is
not implemented, report the current profile as Level 1/2 evidence only.

## 10. Speed And Storage Knobs

- Keep `MDEP_NUM_WORKERS=4` and `MDEP_PREFETCH_FACTOR=4` on Kaggle. Try `6`
  workers only if CPU/RAM usage is stable.
- Keep `MDEP_CUDNN_BENCHMARK=1` because image inputs are fixed-size.
- Use `--no_save_model` for ISIC broad sweeps unless checkpoints are required.
- For CIFAR, checkpoints are already off by default. Add `--save_model` only
  when needed.
- Use one model seed first, then launch multi-seed runs with the same
  `--split_seed`.
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
    --seeds 42 123 456 `
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
