# GUDS-EDL Experiments

This folder contains the active ISIC 2024 and CIFAR-100-LT experiment runners.

## Notebook Separation

- `run_fair_v3_nvidia24_experiments.ipynb` (repository root) contains training,
  ablations, CIFAR-100-LT, OOD evaluation, and metric aggregation only. It does
  not execute hardware profiling or TensorRT.
- `run_rtx_a2000_hardware_experiments.ipynb` (repository root) is the standalone
  local RTX A2000 workflow. It consumes saved fair-v3 checkpoints and runs
  structural profiling, ONNX preflight, TensorRT Level 3, and paper-readiness
  checks without loading datasets or training models.

## Active Runners

- `isic_paper_experiments.py`: ISIC 2024 main-table baselines, evidential
  baselines, GUDS-EDL ablations, calibration variants, and quality-gated
  reports.
- `run_isic_softmax_baselines.py`: thin wrapper for ISIC softmax baselines.
- `run_isic_evidential_baselines.py`: thin wrapper for ISIC evidential
  baselines.
- `run_isic_guds_ablations.py`: thin wrapper for ISIC GUDS-EDL ablations.
- `generalization_paper_suite.py`: CIFAR-100-LT paper-facing baseline suite.
- `run_cifar_suite.py`: thin wrapper that injects `--benchmark cifar`.
- `run_kaggle_paper_suite.py`: Kaggle launcher for ISIC, CIFAR, summary, and
  optional backbone runs. Hardware profiling is intentionally excluded.
- `run_local_full_experiments.py`: local launcher with compact logging.
- `hardware_profile.py`: dense/static-2:4/GUDS structural profiling.
- `nvidia_sparse_benchmark.py`: Level-3 TensorRT FP16 benchmark for trained
  Dense EDL, Static 2:4, RigL-style 2:4, and DST-EDL checkpoints. It separates
  network comparison from same-model sparse-kernel ablation and emits a LaTeX
  table only when the RTX A2000, fair-v3 NVIDIA-layout checkpoint, graph-equivalence, repeat,
  and sparse-build evidence gates pass.
- `run_nvidia_hardware_rtx_a2000.ps1`: local Windows launcher for the TensorRT
  benchmark. It expects `trtexec.exe` on `PATH`, via `TENSORRT_ROOT`, or through
  the `-TrtExec` argument.
- `summarize_results.py`: aggregate JSON metric files across seeds.
- `run_external_validation.py`: unadapted PAD/domain-shift classification,
  validity-aware fairness, and OOD scores. Keep `knn_layer3` as the primary
  unseen-domain score.
- `run_pad_adaptation.py`: patient-grouped nested PAD-UFES adaptation using
  frozen features, binary or six-diagnosis heads, multi-seed ensembling,
  held-out calibration, fairness-adjusted thresholds, patient bootstrap
  intervals, and an optional detached supervised domain head.
- `run_pad_layer4_kd.py`: higher-risk second stage that opens only layer4
  weights, freezes sparse topology and layers through layer3, and replays ISIC
  calibration batches with knowledge distillation.

## Kaggle Quick Start

```bash
%cd /kaggle/working
!git clone https://github.com/minhduc110207/MDEP-Microglial-Driven-Evidential-Pruning.git
%cd MDEP-Microglial-Driven-Evidential-Pruning
!python experiments/run_kaggle_paper_suite.py --smoke
!python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

## ISIC Fair-v3 NVIDIA-layout Protocol

The manuscript-facing comparison must be regenerated with the aligned
`isic_fair_v3_nvidia24_2026_07_09` protocol:

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

Do not mix archived metrics or checkpoints with fair-v3 results. External OOD
evaluation is post-hoc and must consume the three new
`full_guds_fair_v3_nvidia24/seed_{42,123,456}` checkpoints; it must not select or tune
the ISIC model.

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

For CIFAR-only runs:

```bash
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --seeds 42 43 44
```

## PAD-UFES Adaptation

The zero-shot external-validation result and the adapted PAD result answer
different questions. Do not replace the former with the latter.

Train the fixed final-epoch fair-v3 ISIC checkpoints into a separate output
folder:

```bash
python experiments/isic_paper_experiments.py \
  --experiment full_guds \
  --epochs 40 \
  --seeds 42 123 456 \
  --split_seed 42 \
  --subsample_scope train \
  --subsample_ratio 20 \
  --checkpoint_selection last \
  --run_suffix _fair_v3_nvidia24
```

Then run the leakage-safe PAD adapter. The braces in `--model_path` are
expanded by the Python runner for each seed:

```bash
python experiments/run_pad_adaptation.py \
  --pad_root /kaggle/input/datasets/mahdavi1202/skin-cancer \
  --pad_csv /kaggle/input/datasets/mahdavi1202/skin-cancer/metadata.csv \
  --partition all \
  --model_path '/kaggle/working/paper_experiment_outputs/isic/full_guds_fair_v3_nvidia24/seed_{seed}/model_state.pth' \
  --seeds 42 123 456 \
  --target_mode diagnosis6 \
  --feature_layer auto \
  --head linear \
  --outer_folds 5 \
  --inner_folds 3 \
  --fairness_min_group_size 20 \
  --fairness_min_class_size 10 \
  --target_sensitivity 0.80 \
  --bootstrap_repeats 1000 \
  --train_domain_head
```

`pad_adaptation_summary.json` labels the adapted classification, fairness, and
domain-head claims explicitly. The supervised domain head is not an
unseen-domain OOD result; use `run_external_validation.py` with
`--primary_ood_score knn_layer3` for that claim.

Only if the frozen adapter is insufficient, run the constrained layer4 stage:

```bash
python experiments/run_pad_layer4_kd.py \
  --pad_root /kaggle/input/datasets/mahdavi1202/skin-cancer \
  --pad_csv /kaggle/input/datasets/mahdavi1202/skin-cancer/metadata.csv \
  --partition all \
  --model_path '/kaggle/working/paper_experiment_outputs/isic/full_guds_fair_v3_nvidia24/seed_{seed}/model_state.pth' \
  --seeds 42 123 456 \
  --target_mode diagnosis6 \
  --outer_folds 5 \
  --inner_folds 3 \
  --epochs 12 \
  --lr 1e-5 \
  --kd_weight 2.0 \
  --kd_temperature 2.0
```

The runner asserts after every fit that all sparse scores and masks are
bitwise unchanged. It also records ISIC validation pAUC/AP for each fold and
never uses the ISIC test set for optimization.

## Outputs

Runs write metrics, configs, logs, and optional checkpoints under
`paper_experiment_outputs/`. Pass `--no_save_model` for large multi-seed
sweeps to avoid filling Kaggle storage.

## Fairness Notes

The main evidential/sparse comparison shares the same split, ResNet-18
backbone, seeds, 40-epoch budget, learning-rate/loss-scale warmup, FP32
objective path, calibration/evaluation surface, and deterministic runtime.
Static 2:4 uses one fixed magnitude mask; RigL-style 2:4 uses magnitude pruning
and task-gradient regrowth. Proxy baselines such as Fisher EDL, Flexible EDL,
R-EDL, MiSLAS-style LAS+cRT, and RigL-style 2:4 must be described as controlled
in-repo implementations rather than official external-code reproductions.
