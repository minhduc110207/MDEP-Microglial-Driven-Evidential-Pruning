# Kaggle Separate Notebook Guide

Use separate notebooks when a full multi-seed suite is likely to exceed one
Kaggle session.

## Shared Setup Cell

Use the setup cell in `kaggle_instructions.md` or `KAGGLE_GITHUB_SETUP.md` at
the top of every notebook. Keep these environment variables for faster data
loading:

```python
import os
os.environ.setdefault("MDEP_NUM_WORKERS", "4")
os.environ.setdefault("MDEP_PREFETCH_FACTOR", "4")
```

## Recommended Notebook Split

1. ISIC softmax baselines.
2. ISIC evidential baselines.
3. ISIC GUDS-EDL and ablations.
4. CIFAR-100-LT ratio 100.
5. CIFAR-100-LT ratio 50.
6. CIFAR-100-LT ratio 10.
7. Hardware profiling and result summarization.
8. Optional backbone generalization.

## Commands

```bash
!python experiments/run_isic_softmax_baselines.py --epochs 40 --batch_size 32 --lr 4e-5 --subsample_scope train --seeds 42 123 456 --no_save_model
!python experiments/run_isic_evidential_baselines.py --epochs 40 --batch_size 32 --lr 4e-5 --subsample_scope train --seeds 42 123 456 --no_save_model
!python experiments/run_isic_guds_ablations.py --epochs 40 --batch_size 32 --lr 4e-5 --subsample_scope train --seeds 42 123 456 --no_save_model
!python experiments/run_cifar_suite.py --ratio 100 --epochs 100 --batch_size 128 --seeds 42 123 456
!python experiments/run_cifar_suite.py --ratio 50 --epochs 100 --batch_size 128 --seeds 42 123 456
!python experiments/run_cifar_suite.py --ratio 10 --epochs 100 --batch_size 128 --seeds 42 123 456
!python experiments/hardware_profile.py
!python experiments/summarize_results.py
```

## Speed Notes

- Prefer P100 for long ISIC runs when available.
- Keep model checkpoint saving off during broad sweeps with `--no_save_model`.
- Increase `MDEP_NUM_WORKERS` to `6` only if the notebook has spare CPU and RAM.
- Use one seed first for a quick full-pipeline check, then launch multi-seed
  notebooks.

## Output Handling

Zip only metrics/logs unless checkpoints are needed:

```bash
!zip -r /kaggle/working/mdep_results.zip paper_experiment_outputs/ -x "*.pth"
```
