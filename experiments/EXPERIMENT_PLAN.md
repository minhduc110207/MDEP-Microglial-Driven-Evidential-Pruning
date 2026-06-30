# GUDS-EDL Experiment Plan

This plan covers the active experiment surface after pruning the removed
industrial-defect protocol from the repository.

## Goals

- Evaluate GUDS-EDL under extreme class imbalance.
- Keep ISIC 2024 as the high-stakes rare-event case study.
- Use CIFAR-100-LT as the controlled long-tailed recognition benchmark.
- Report calibration, selective-risk, ranking, and structural sparsity metrics
  under shared splits, seeds, backbone family, epoch budgets, and calibration.

## Active Benchmarks

### ISIC 2024

Run with `experiments/isic_paper_experiments.py`. This is the main clinical
case-study path and supports main-table baselines, evidential baselines,
GUDS-EDL ablations, calibration variants, and quality-gated reporting.

### CIFAR-100-LT

Run with `experiments/generalization_paper_suite.py --benchmark cifar` or the
thin wrapper `experiments/run_cifar_suite.py`. Ratios 10, 50, and 100 are
supported.

## Training Pipeline

1. Discover or download the dataset.
2. Build train, validation, calibration, and test loaders.
3. Train dense or 2:4 sparse models with AdamW and cosine scheduling.
4. Fit post-hoc temperature/bias calibration on the calibration split.
5. Evaluate on the held-out test split and write JSON metrics.

## Output

All paper-facing runs write to `paper_experiment_outputs/`. Use
`experiments/summarize_results.py` after multi-seed runs to aggregate metrics.
