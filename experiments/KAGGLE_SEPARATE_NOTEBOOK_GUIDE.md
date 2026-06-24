# Kaggle Separate Notebook Guide

This guide is the current paper-facing Kaggle workflow. It assumes the latest
GUDS-EDL fixes are already pushed to GitHub:

- Full GUDS uses `regrower_type="class_conditioned"`.
- Class-conditioned regrowth is a scalar sample gate, not `alpha * class_weight`.
- EFL weights CE and KL separately to avoid double-weighting the KL term.
- Active links use `G - C`; dormant links use attenuated regrowth plus optional
  exploration.
- MVTec uses image-level anomaly metrics, not ISIC clinical metrics.

## 0. Files To Push To GitHub

Push these files before opening Kaggle notebooks. They are the minimum needed
for the current paper and experiment suite.

### Required Core Files

```text
main_text.tex
guds_edl_core.py
```

### Required Experiment Files

```text
experiments/KAGGLE_GITHUB_SETUP.md
experiments/KAGGLE_SEPARATE_NOTEBOOK_GUIDE.md
experiments/PAPER_EXPERIMENT_COVERAGE.md
experiments/README.md
experiments/BASELINE_SOTA_AUDIT_NOTES.md
experiments/cifar_lt_runner.py
experiments/generalization_paper_suite.py
experiments/isic_paper_experiments.py
experiments/metrics_ext.py
experiments/mvtec_ad_runner.py
experiments/mvtec_patchcore_reference.py
experiments/mvtec_simplenet_reference.py
experiments/run_kaggle_paper_suite.py
experiments/run_local_full_experiments.py
experiments/backbone_generalization_runner.py
experiments/hardware_profile.py
experiments/run_kaggle_all_isic.py
experiments/summarize_results.py
```

### Optional Notes

```text
THEORY_CHANGES.md
```

### Do Not Push Outputs Or Scratch Files

Do not push generated artifacts, temporary refactor scripts, local PDF/image
outputs, or Kaggle result folders:

```text
artifacts/
paper_experiment_outputs/
calibration_reliability_diagrams.png
model_complexity_comparison.png
selective_referral_curves.png
extracted_tables.txt
refactor_*.py
rewrite_main.py
trim.py
main_text_refactored_v2.tex
supplementary.tex
```

Recommended Git commands:

```bash
git add main_text.tex guds_edl_core.py
git add experiments/KAGGLE_GITHUB_SETUP.md
git add experiments/KAGGLE_SEPARATE_NOTEBOOK_GUIDE.md
git add experiments/PAPER_EXPERIMENT_COVERAGE.md
git add experiments/README.md
git add experiments/BASELINE_SOTA_AUDIT_NOTES.md
git add experiments/cifar_lt_runner.py
git add experiments/generalization_paper_suite.py
git add experiments/isic_paper_experiments.py
git add experiments/metrics_ext.py
git add experiments/mvtec_ad_runner.py
git add experiments/mvtec_patchcore_reference.py
git add experiments/mvtec_simplenet_reference.py
git add experiments/run_kaggle_paper_suite.py
git add experiments/run_local_full_experiments.py
git add experiments/backbone_generalization_runner.py
git add experiments/hardware_profile.py
git add experiments/run_kaggle_all_isic.py
git add experiments/summarize_results.py
git commit -m "Update GUDS-EDL theory and paper experiment suite"
git push
```

## 1. Kaggle Inputs

Attach inputs according to the notebook you run.

### ISIC 2024

Attach the official Kaggle competition input:

```text
ISIC 2024 - Skin Cancer Detection with 3D-TBP
```

The code searches for:

```text
/kaggle/input/competitions/isic-2024-challenge/train-metadata.csv
/kaggle/input/competitions/isic-2024-challenge/train-image.hdf5
```

### MVTec AD

Attach an MVTec AD dataset with category folders such as:

```text
bottle/
cable/
capsule/
carpet/
grid/
hazelnut/
leather/
metal_nut/
pill/
screw/
tile/
toothbrush/
transistor/
wood/
zipper/
```

Each category should contain the official structure:

```text
train/good/
test/good/
test/<defect_type>/
ground_truth/
```

The code searches under `/kaggle/input` automatically. If you run locally,
set `MVTEC_ROOT` or pass a local path to the local runner.

### CIFAR-100-LT

CIFAR-100-LT does not need a Kaggle input dataset. The runner downloads
CIFAR-100 and creates long-tailed splits for imbalance ratios 10, 50, and 100.
Kaggle internet must be enabled for this download.

## 2. Common Setup Cell

Run this cell at the top of every separate Kaggle notebook. Change `REPO_URL`
only if your GitHub repository URL is different.

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

run([
    sys.executable, "-m", "pip", "install", "-q",
    "scikit-learn", "matplotlib", "pandas", "h5py", "tqdm", "scipy"
])

os.chdir(REPO_DIR)
print("Repo:", REPO_DIR)
run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR)
```

Quick dataset check cell:

```python
!find /kaggle/input -maxdepth 4 -name "train-metadata.csv"
!find /kaggle/input -maxdepth 4 -name "train-image.hdf5"
!find /kaggle/input -maxdepth 3 -type d -iname "hazelnut" -o -type d -iname "bottle"
```

## 3. Recommended Notebook Order

Use separate Kaggle notebooks/jobs in this order:

1. Smoke test.
2. ISIC main tables.
3. ISIC full baseline suite.
4. ISIC GUDS ablations.
5. CIFAR-100-LT ratio 10.
6. CIFAR-100-LT ratio 50.
7. CIFAR-100-LT ratio 100.
8. MVTec supervised GUDS-compatible classifier, one category per notebook.
9. MVTec PatchCore-style normal-only reference, one category per notebook.
10. MVTec SimpleNet-style normal-only reference, one category per notebook.
11. Hardware profiling.
12. Optional backbone generalization.
13. Summary aggregation.

All-in-one alternative for a long Kaggle job:

```python
!python experiments/run_kaggle_paper_suite.py \
  --isic_suite all \
  --cifar_ratios 10 50 100 \
  --mvtec_categories hazelnut bottle \
  --seeds 42 43 44 \
  --no_save_model \
  --keep_going
```

This currently runs ISIC, CIFAR-100-LT, MVTec supervised classifier,
PatchCore-style MVTec reference, SimpleNet-style MVTec reference, hardware
profiling, and summary aggregation. Add `--include_backbones` only for the
heavier ResNet/ConvNeXt/Swin protocol.

For final paper tables, prefer 3 seeds:

```text
42 43 44
```

For a first debug pass, use only:

```text
42
```

## 4. Smoke Test Notebook

This confirms import paths, dataset discovery, and output writing. Smoke mode
uses 1 epoch, so seeing `Epoch [1/1]` is expected.

```python
!python experiments/run_kaggle_paper_suite.py \
  --smoke \
  --no_save_model \
  --keep_going
```

Zip outputs:

```python
%cd /kaggle/working
!zip -r paper_experiment_outputs_smoke.zip paper_experiment_outputs
```

## 5. ISIC Main Tables Notebook

This runs the main paper rows available in `SUITES["main_tables"]`:

```text
fisher_edl
flexible_edl
r_edl
full_guds
```

Command:

```python
!python experiments/isic_paper_experiments.py \
  --suite main_tables \
  --epochs 40 \
  --batch_size 32 \
  --seeds 42 43 44 \
  --no_save_model \
  --log_every 5
```

Output root:

```text
/kaggle/working/paper_experiment_outputs/isic/
```

## 6. ISIC Baselines Notebook

This runs the long-tailed, evidential, and sparse baselines:

```text
standard_ce
focal_loss
logit_adjustment
class_balanced_ce
balanced_softmax
ldam_drw
decoupled_crt
mislas
dense_edl
fisher_edl
flexible_edl
r_edl
static_24_edl
rigl_style_24
```

Command:

```python
!python experiments/isic_paper_experiments.py \
  --suite baselines \
  --epochs 40 \
  --batch_size 32 \
  --seeds 42 43 44 \
  --no_save_model \
  --log_every 5
```

CE-style baselines are evaluated with softmax probabilities. EDL/GUDS models
are evaluated with evidential probabilities after the evidential calibration
step.

## 7. ISIC GUDS Ablations Notebook

This is the key component-isolation suite for Appendix C:

```text
full_guds
guds_without_pruner
guds_without_regrower
guds_symmetric_kl
guds_without_efl
guds_without_anticryst
guds_absolute_pruner
guds_kl_uniform_regrower
guds_without_topology_cache
guds_temperature_only
guds_no_posthoc_calibration
```

Command:

```python
!python experiments/isic_paper_experiments.py \
  --suite ablations \
  --epochs 40 \
  --batch_size 32 \
  --seeds 42 43 44 \
  --no_save_model \
  --log_every 5
```

Full GUDS in this suite uses:

```text
pruner_type = signed_first_order
regrower_type = class_conditioned
kl_scaling = asymmetric
EFL enabled
anti-crystallization enabled
topology cache enabled
```

## 8. CIFAR-100-LT Notebooks

Run one imbalance ratio per notebook. CIFAR uses multiclass long-tailed metrics:
top-1/top-5 accuracy, balanced accuracy, macro-F1, many/medium/few-shot
accuracy, macro-AUROC, macro-PR-AUC, NLL, Brier score, ECE, AURC, and
failure-detection metrics.

Ratio 10:

```python
!python experiments/generalization_paper_suite.py \
  --benchmark cifar \
  --ratio 10 \
  --epochs 100 \
  --batch_size 128 \
  --seeds 42 43 44 \
  --log_every 10
```

Ratio 50:

```python
!python experiments/generalization_paper_suite.py \
  --benchmark cifar \
  --ratio 50 \
  --epochs 100 \
  --batch_size 128 \
  --seeds 42 43 44 \
  --log_every 10
```

Ratio 100:

```python
!python experiments/generalization_paper_suite.py \
  --benchmark cifar \
  --ratio 100 \
  --epochs 100 \
  --batch_size 128 \
  --seeds 42 43 44 \
  --log_every 10
```

Cheaper first pass:

```python
!python experiments/generalization_paper_suite.py \
  --benchmark cifar \
  --ratio 100 \
  --epochs 50 \
  --batch_size 128 \
  --seeds 42 \
  --log_every 10
```

## 9. MVTec Supervised Classifier Notebooks

Run one category per notebook if Kaggle time is limited. This is the
GUDS-compatible supervised few-shot image-level binary classifier protocol.
It uses image AUROC, image AP, F1-max, balanced accuracy, ECE, and
failure-detection metrics.

Example: bottle

```python
!python experiments/generalization_paper_suite.py \
  --benchmark mvtec \
  --category bottle \
  --epochs 20 \
  --batch_size 16 \
  --seeds 42 43 44 \
  --log_every 5
```

Example: hazelnut

```python
!python experiments/generalization_paper_suite.py \
  --benchmark mvtec \
  --category hazelnut \
  --epochs 20 \
  --batch_size 16 \
  --seeds 42 43 44 \
  --log_every 5
```

To run another category, replace `bottle` or `hazelnut` with the folder name.

## 10. MVTec PatchCore-Style Reference Notebooks

PatchCore-style runs use the normal-only anomaly-detection setting, separate
from the supervised GUDS-compatible classifier.

```python
!python experiments/mvtec_patchcore_reference.py \
  --category bottle \
  --batch_size 16 \
  --seeds 42 43 44
```

```python
!python experiments/mvtec_patchcore_reference.py \
  --category hazelnut \
  --batch_size 16 \
  --seeds 42 43 44
```

## 11. MVTec SimpleNet-Style Reference Notebooks

SimpleNet-style runs use pretrained patch features, synthetic feature anomalies,
and a discriminator-style anomaly score.

```python
!python experiments/mvtec_simplenet_reference.py \
  --category bottle \
  --batch_size 16 \
  --epochs 10 \
  --seeds 42 43 44
```

```python
!python experiments/mvtec_simplenet_reference.py \
  --category hazelnut \
  --batch_size 16 \
  --epochs 10 \
  --seeds 42 43 44
```

## 12. Hardware Profile Notebook

This profiles dense, static 2:4, and GUDS-style masked PyTorch execution.

```python
!python experiments/hardware_profile.py \
  --batch_size 32 \
  --iters 50 \
  --warmup 10
```

Important: this reports masked PyTorch throughput, active density, valid 2:4
block fraction, and structural overhead. It is not a true cuSPARSELt/TensorRT
sparse Tensor Core benchmark.

## 13. Optional Backbone Generalization Notebook

This is heavier and should be run only after the main ISIC/CIFAR/MVTec suites
are stable.

```python
!python experiments/backbone_generalization_runner.py \
  --backbones resnet18 convnext_tiny swin_t \
  --epochs 40 \
  --seeds 42 43 44
```

## 14. Summary Aggregation Notebook

After one or more notebooks have produced outputs, run:

```python
!python experiments/summarize_results.py \
  --root /kaggle/working/paper_experiment_outputs
```

Expected output:

```text
/kaggle/working/paper_experiment_outputs/summary_metrics.csv
```

If you ran experiments across separate notebooks, download or attach previous
output zips before aggregating everything together.

## 15. Zip Outputs For Kaggle Save Version

Run this at the end of every notebook:

```python
%cd /kaggle/working
!zip -r paper_experiment_outputs.zip paper_experiment_outputs
```

For separate jobs, rename the zip so it identifies the notebook:

```python
%cd /kaggle/working
!zip -r isic_ablations_seed42_43_44.zip paper_experiment_outputs
```

## 16. If Kaggle Times Out

Split by seed:

```python
!python experiments/isic_paper_experiments.py \
  --suite ablations \
  --epochs 40 \
  --batch_size 32 \
  --seeds 42 \
  --no_save_model \
  --log_every 5
```

Then create separate notebooks for:

```text
--seeds 43
--seeds 44
```

You can also split by experiment:

```python
!python experiments/isic_paper_experiments.py \
  --experiment full_guds \
  --epochs 40 \
  --batch_size 32 \
  --seeds 42 \
  --no_save_model \
  --log_every 5
```

For CIFAR, split by ratio and seed. For MVTec, split by category, method, and
seed.

## 17. Log Controls

Default logs are compact:

```text
--log_every 5
```

Print every epoch:

```text
--log_every 1
```

Debug structural sparsity internals:

```text
--verbose_structural_logs
```

Only use `--verbose_structural_logs` for debugging. It can make Kaggle output
very long.

## 18. Quick Sanity Checklist

Before starting a long run:

```text
[ ] GitHub has main_text.tex and guds_edl_core.py pushed.
[ ] GitHub has mvtec_simplenet_reference.py pushed.
[ ] Kaggle notebook setup cell prints the latest commit after git pull.
[ ] ISIC train-metadata.csv and train-image.hdf5 are visible under /kaggle/input.
[ ] MVTec category folders are visible under /kaggle/input.
[ ] Smoke test finishes with 1 epoch.
[ ] Full GUDS run_config.json shows regrower_type = class_conditioned.
[ ] Full GUDS run_config.json shows kl_scaling = asymmetric.
[ ] Output zip is created before Save Version.
```
