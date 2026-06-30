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
