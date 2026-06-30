# Baseline And SOTA Audit Notes

This note records the active controlled baseline suite.

## Positioning

- The baselines are suitable for controlled comparison with GUDS-EDL under
  shared splits, backbone family, seed protocol, epoch budget, calibration, and
  metrics.
- They are not all official external-code reproductions.
- Use wording such as `in-repo implementation`, `reference baseline`, `proxy`,
  or `SOTA-style baseline` when appropriate.
- Do not claim broad SOTA superiority unless official/tuned external baselines
  are also run.

## Long-Tailed / Imbalanced Baselines

- `standard_ce`: dense ResNet-18 with standard cross-entropy.
- `focal_loss`: dense ResNet-18 with focal modulation.
- `class_balanced_ce`: effective-number class-balanced weighting.
- `balanced_softmax`: train-prior logits inside the CE objective.
- `ldam_drw`: label-distribution-aware margins with deferred reweighting.
- `decoupled_crt`: dense CE representation learning followed by classifier
  retraining with a class-balanced sampler.
- `mislas`: MiSLAS-style LAS+cRT proxy under the shared protocol.

## Evidential Baselines

- `dense_edl`: dense EDL with symmetric KL and no focal modulation.
- `fisher_edl`: Fisher-regularized EDL proxy.
- `flexible_edl`: EDL with a learnable positive evidence scale.
- `r_edl`: relaxed EDL proxy with reduced KL pressure and no focal modulation.

## Sparse / Dynamic Sparse Baselines

- `static_24_edl`: static 2:4 sparse EDL with fixed magnitude-derived masks.
- `rigl_style_24`: 2:4 structured RigL-style proxy with task-gradient regrowth.

## GUDS-EDL Ablations

- `full_guds`
- `guds_without_pruner`
- `guds_without_regrower`
- `guds_asymmetric_kl`
- `guds_without_efl`
- `guds_without_anticryst`
- `guds_absolute_pruner`
- `guds_class_conditioned_regrower`
- `guds_without_topology_cache`
- `guds_temperature_only`
- `guds_no_posthoc_calibration`

## Metric Policy

ISIC keeps clinical/medical metrics: pAUC at high TPR, sensitivity/specificity
at operating thresholds, high-recall operating points, PPV/NPV under deployment
prevalence, quality-gated referral metrics, ECE, AURC, and uncertainty
separation.

CIFAR-100-LT uses long-tail multiclass metrics: top-1/top-5 accuracy, balanced
accuracy, macro-F1, macro-AUROC, macro-PR-AUC, NLL, multiclass Brier, ECE,
AURC/E-AURC, many/medium/few-shot accuracy, group ECE, classwise ECE, and
worst-group accuracy.

## Main Commands

```bash
python experiments/isic_paper_experiments.py --suite baselines --epochs 40 --batch_size 32 --seeds 42 43 44 --no_save_model
python experiments/isic_paper_experiments.py --suite ablations --epochs 40 --batch_size 32 --seeds 42 43 44 --no_save_model
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --batch_size 128 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --batch_size 128 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --batch_size 128 --seeds 42 43 44
```

## Safe Paper Wording

`We compare GUDS-EDL against controlled in-repo baselines under the same split,
backbone family, seed protocol, epoch budget, calibration, and metrics.`

Avoid:

`GUDS-EDL outperforms all SOTA methods.`
