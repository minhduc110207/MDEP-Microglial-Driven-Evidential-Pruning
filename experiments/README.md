# GUDS-EDL Experiments

This folder contains the active ISIC 2024 and CIFAR-100-LT experiment runners.

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
- `run_kaggle_paper_suite.py`: Kaggle launcher for ISIC, CIFAR, hardware
  profiling, and optional backbone runs.
- `run_local_full_experiments.py`: local launcher with compact logging.
- `hardware_profile.py`: dense/static-2:4/GUDS structural profiling.
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

For CIFAR-only runs:

```bash
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --seeds 42 43 44
!python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --seeds 42 43 44
```

## PAD-UFES Adaptation

The zero-shot external-validation result and the adapted PAD result answer
different questions. Do not replace the former with the latter.

Optionally train validation-selected ISIC checkpoints into a separate output
folder:

```bash
python experiments/isic_paper_experiments.py \
  --experiment full_guds \
  --epochs 40 \
  --seeds 42 123 456 \
  --split_seed 42 \
  --subsample_scope train \
  --subsample_ratio 20 \
  --checkpoint_selection pauc_then_ap \
  --checkpoint_eval_every 5 \
  --checkpoint_start_epoch 10 \
  --run_suffix _bestckpt
```

Then run the leakage-safe PAD adapter. The braces in `--model_path` are
expanded by the Python runner for each seed:

```bash
python experiments/run_pad_adaptation.py \
  --pad_root /kaggle/input/datasets/mahdavi1202/skin-cancer \
  --pad_csv /kaggle/input/datasets/mahdavi1202/skin-cancer/metadata.csv \
  --partition all \
  --model_path '/kaggle/working/paper_experiment_outputs/isic/full_guds_bestckpt/seed_{seed}/model_state.pth' \
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
  --model_path '/kaggle/working/paper_experiment_outputs/isic/full_guds_bestckpt/seed_{seed}/model_state.pth' \
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

All in-repo baselines share the same splits, backbone family, seeds, epoch
budget, calibration/evaluation surface, and output format. Proxy baselines such
as Fisher EDL, Flexible EDL, R-EDL, MiSLAS-style LAS+cRT, and RigL-style 2:4
should be described as controlled in-repo implementations rather than official
external-code reproductions.
