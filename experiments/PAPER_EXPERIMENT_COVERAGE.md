# Paper Experiment Coverage

This file maps experiments described in `main_text.tex` to runnable Kaggle
commands in the `experiments/` folder.

## One-Command Kaggle Suite

Clone the GitHub repository into `/kaggle/working` first. The complete notebook
setup order is documented in `experiments/KAGGLE_GITHUB_SETUP.md`.

Run the whole paper-facing suite:

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all
```

For a quick smoke test before spending GPU hours:

```bash
python experiments/run_kaggle_paper_suite.py --smoke
```

All logs and metrics are written under:

```text
paper_experiment_outputs/
```

## ISIC 2024 Main Tables

The main ISIC tables in `main_text.tex` are covered by:

```bash
python experiments/isic_paper_experiments.py --suite main_tables
```

| Paper row | Runnable experiment |
| --- | --- |
| Fisher EDL | `fisher_edl` |
| Flexible EDL | `flexible_edl` |
| R-EDL | `r_edl` |
| GUDS-EDL (Ours) | `full_guds` |

The broader baseline suite can be run with:

```bash
python experiments/isic_paper_experiments.py --suite baselines
```

It includes Standard CE, Focal Loss, Logit Adjustment, Dense EDL, Fisher EDL,
Flexible EDL, R-EDL, Static 2:4 EDL, and a RigL-style 2:4 proxy.

## Appendix C Ablations

Run:

```bash
python experiments/isic_paper_experiments.py --suite ablations
```

| Paper target | Runnable experiment |
| --- | --- |
| Full GUDS-EDL | `full_guds` |
| w/o uncertainty pruner | `guds_without_pruner` |
| w/o evidence regrower | `guds_without_regrower` |
| w/o asymmetric KL | `guds_symmetric_kl` |
| w/o Evidential Focal Loss | `guds_without_efl` |
| w/o anti-crystallization | `guds_without_anticryst` |
| absolute-gradient pruning ablation | `guds_absolute_pruner` |
| KL-to-uniform regrowth ablation | `guds_kl_uniform_regrower` |

## Planned Generalization Protocols

The CIFAR-100-LT protocol can be run by the all-in-one suite, or manually:

```bash
python experiments/cifar_lt_runner.py --imbalance_ratio 10 --epochs 100
python experiments/cifar_lt_runner.py --imbalance_ratio 50 --epochs 100
python experiments/cifar_lt_runner.py --imbalance_ratio 100 --epochs 100
```

The MVTec AD image-level protocol can be run by:

```bash
python experiments/mvtec_ad_runner.py --category hazelnut --epochs 20
python experiments/mvtec_ad_runner.py --category bottle --epochs 20
```

`mvtec_ad_runner.py` now searches for a real MVTec category under `MVTEC_ROOT`,
`./data/mvtec_ad`, `./data/mvtec`, or `/kaggle/input`. If no real category is
found, it falls back to dummy tensors and should only be treated as a pipeline
smoke test.

## Remaining Caveats

- `Flexible EDL` and `R-EDL` are implemented as reproducible in-repo baseline
  variants so the paper table can be regenerated on the same split. They are
  not official external code releases.
- `RigL-style 2:4` is a proxy using the available structured sparse update
  surface. A fully faithful RigL implementation would require a separate update
  engine.
- Real speedups for 2:4 sparsity still require sparse Tensor Core kernels; these
  scripts report training/evaluation metrics and structural sparsity, not
  guaranteed hardware acceleration.
