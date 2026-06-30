# Paper Experiment Coverage

This file maps the active paper-facing experiment scripts to runnable commands.

## One-Command Kaggle Suite

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

For a quick import/data-path check:

```bash
python experiments/run_kaggle_paper_suite.py --smoke
```

All logs and metrics are written under `paper_experiment_outputs/`.

## ISIC 2024 Main Tables

```bash
python experiments/isic_paper_experiments.py --suite main_tables --seeds 42 43 44 --no_save_model
```

| Paper row | Runnable experiment |
| --- | --- |
| Fisher EDL | `fisher_edl` |
| Flexible EDL | `flexible_edl` |
| R-EDL | `r_edl` |
| GUDS-EDL | `full_guds` |

## ISIC Baselines

```bash
python experiments/isic_paper_experiments.py --suite baselines --epochs 40 --batch_size 32 --seeds 42 43 44 --no_save_model
```

This includes Standard CE, Focal Loss, Class-Balanced CE, Balanced Softmax,
LDAM-DRW, cRT-style classifier retraining, MiSLAS-style LAS+cRT, Dense EDL,
Fisher EDL, Flexible EDL, R-EDL, Static 2:4 EDL, and a RigL-style 2:4 proxy.

## ISIC Ablations

```bash
python experiments/isic_paper_experiments.py --suite ablations --epochs 40 --batch_size 32 --seeds 42 43 44 --no_save_model
```

The ablation suite covers the full model, pruner removal, regrower removal,
KL scaling, EFL removal, anti-crystallization removal, alternative pruner and
regrower variants, topology-cache removal, and calibration variants.

## CIFAR-100-LT

```bash
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --batch_size 128 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --batch_size 128 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --batch_size 128 --seeds 42 43 44
```

CIFAR metrics include top-1/top-5 accuracy, balanced accuracy, macro-F1,
macro-AUROC, macro-PR-AUC, NLL, multiclass Brier, calibration, selective-risk,
many/medium/few-shot accuracy, group ECE, classwise ECE, and worst-group
accuracy.

## Hardware And Backbones

```bash
python experiments/hardware_profile.py
python experiments/backbone_generalization_runner.py --backbones resnet18 convnext_tiny swin_t --epochs 40 --seeds 42 43 44
python experiments/summarize_results.py
```

Hardware metrics are structural/profiling numbers unless the model is exported
to a real sparse Tensor Core inference kernel.
