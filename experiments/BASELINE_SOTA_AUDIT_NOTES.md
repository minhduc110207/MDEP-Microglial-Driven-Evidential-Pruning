# Baseline and SOTA Audit Notes

This note records the current state of the baseline suite after the SOTA/fairness
audit. It is meant to be a memory file for future paper writing, Kaggle runs,
and GitHub updates.

## Current Position

The experiment suite is now stronger than the original controlled baseline set,
but it should still be described carefully:

- The current baselines are suitable for controlled comparison with GUDS-EDL
  under shared splits, backbone family, seed protocol, epoch budget,
  calibration, and metrics.
- They are not all official external-code reproductions.
- For AAAI-style writing, use wording such as `in-repo implementation`,
  `reference baseline`, `proxy`, or `SOTA-style baseline` when appropriate.
- Do not claim that GUDS-EDL outperforms full SOTA unless official/tuned
  external baselines are also run.

## Long-Tailed / Imbalanced Classification Baselines

The following are implemented in `experiments/isic_paper_experiments.py` and are
also available through `experiments/generalization_paper_suite.py` where
applicable.

### Standard Controlled Baselines

- `standard_ce`
  - Dense ResNet-18 with standard cross-entropy.
  - Useful as the ordinary ERM reference.

- `focal_loss`
  - Dense ResNet-18 with focal modulation.
  - Kept unweighted so it is not accidentally mixed with GUDS class weighting.

- `logit_adjustment`
  - Implements train-prior/target-prior logit correction in the training loss.
  - Suitable for label-prior-shift style comparison.

- `class_balanced_ce`
  - Uses effective-number class-balanced weighting.
  - Matches the standard Class-Balanced Loss idea.

- `balanced_softmax`
  - Adds train-prior logits inside the CE objective.
  - Suitable for long-tailed label-prior correction.

- `ldam_drw`
  - Uses label-distribution-aware margins and deferred reweighting.
  - This is a compact in-repo implementation, not a full official training
    recipe reproduction.

- `decoupled_crt`
  - Dense CE representation learning followed by classifier retraining.
  - The classifier head is re-initialized and retrained with a class-balanced
    sampler.

### Added SOTA-Style Long-Tail Baseline

- `mislas`
  - Added as a MiSLAS-style baseline.
  - Implements the key practical pieces that fit this repo:
    decoupled classifier retraining, class-balanced sampling, and
    label-aware smoothing.
  - This is not the full official MiSLAS code. It does not include every detail
    of the official shifted batch normalization and paper-specific tuning grid.
  - Recommended wording in paper:
    `MiSLAS-style LAS+cRT baseline under the shared GUDS-EDL evaluation protocol`.

## Evidential Baselines

- `dense_edl`
  - Dense EDL with symmetric KL and no focal modulation.

- `fisher_edl`
  - Fisher-regularized EDL proxy.
  - Not the official Fisher EDL authors' code.
  - Use cautious wording: `Fisher-EDL-inspired in-repo proxy`.

- `flexible_edl`
  - Uses `FlexibleEvidenceLayer`, i.e. a learnable positive evidence scale.
  - Not the full official Flexible EDL code.

- `r_edl`
  - Relaxed EDL proxy with reduced KL pressure and no focal modulation.
  - Related to the Re-EDL/R-EDL idea, but not official external code.

Important paper-writing caveat:

- These baselines are good for same-codebase controlled comparison.
- They are not enough to claim state-of-the-art uncertainty quantification.
- EDL uncertainty should be described as a practical structural/adaptation
  signal, not as a guaranteed Bayesian epistemic uncertainty estimate.

## Sparse / Dynamic Sparse Baselines

- `static_24_edl`
  - Static 2:4 sparse EDL with fixed magnitude-derived masks.
  - Useful for separating the effect of valid-block 2:4 sparsity from dynamic
    topology adaptation.

- `rigl_style_24`
  - Corrected during the audit.
  - Now uses task-gradient regrowth through `grad_L_w`.
  - No longer mixes GUDS anti-crystallization/uncertainty noise into the
    RigL-style baseline.
  - Still a 2:4 structured proxy, not the official unstructured RigL or SRigL
    codebase.

Recommended wording:

`RigL-style 2:4 proxy with task-gradient regrowth under the same structured
sparsity interface as GUDS-EDL.`

## GUDS-EDL Ablations

The ablation suite remains:

- `full_guds`
- `guds_without_pruner`
- `guds_without_regrower`
- `guds_symmetric_kl`
- `guds_without_efl`
- `guds_without_anticryst`
- `guds_absolute_pruner`
- `guds_kl_uniform_regrower`
- `guds_without_topology_cache`
- `guds_temperature_only`
- `guds_no_posthoc_calibration`

These should be used to attribute gains to the pruner, regrower, asymmetric KL,
focal modulation, anti-crystallization, topology cache, and calibration choices.

## MVTec AD Baselines

The MVTec AD part now has two different protocol families.

### GUDS-Compatible Classifier Protocol

`experiments/generalization_paper_suite.py --benchmark mvtec`

- Runs the same classifier-style baseline family as CIFAR/ISIC.
- Uses MVTec as supervised few-shot image-level rare-defect classification.
- This is useful for testing whether GUDS-EDL transfers to industrial
  rare-event classification.
- It should not be presented as the canonical unsupervised MVTec AD protocol.

### Normal-Only Anomaly Detection References

`experiments/mvtec_patchcore_reference.py`

- PatchCore-style reference.
- Uses normal-only train/good images.
- Uses pretrained multi-layer ResNet-18 patch features.
- Scores images by nearest-neighbor distance to the memory bank.
- It is stronger and closer to PatchCore than the previous single-layer
  PatchCore-lite version.
- Still not the official PatchCore implementation with full coreset/tuning.

`experiments/mvtec_simplenet_reference.py`

- Newly added SimpleNet-style reference.
- Uses pretrained patch features, a shallow feature adapter, synthetic Gaussian
  feature anomalies, and a lightweight discriminator.
- Scores images by the maximum patch anomaly probability.
- Not the official SimpleNet authors' code.

Recommended wording:

`PatchCore-style and SimpleNet-style normal-only anomaly-detection references
are included to contextualize MVTec image-level performance.`

## Metric Policy After Audit

Metrics are now benchmark-specific.

### ISIC

ISIC keeps clinical/medical metrics:

- pAUC at high TPR
- sensitivity/specificity at operating thresholds
- high-recall fail-safe threshold
- PPV/NPV under deployment prevalence
- quality-gated referral metrics
- ECE, AURC, uncertainty separation

### CIFAR-100-LT

CIFAR uses long-tail multiclass metrics:

- top-1 accuracy
- top-5 accuracy
- balanced accuracy
- macro-F1
- macro-AUROC
- macro-PR-AUC
- NLL
- multiclass Brier
- ECE
- AURC / E-AURC
- many/medium/few-shot accuracy and ECE
- worst-group accuracy

No clinical metrics should be used for CIFAR.

### MVTec AD

MVTec uses image-level anomaly/classification metrics:

- image AUROC
- image AP
- F1-max
- balanced accuracy at default threshold
- precision / recall / F1 at default threshold
- Brier
- NLL
- ECE
- failure-detection AUROC/AUPR
- risk-at-coverage

No ISIC-style clinical PPV/NPV, Wilson intervals, decision-curve analysis, or
rule-out/rule-in reporting should be used for MVTec.

## Main Run Commands

### ISIC Baselines

```bash
python experiments/isic_paper_experiments.py --suite baselines --epochs 40 --batch_size 32 --seeds 42 43 44 --no_save_model
```

This includes `mislas`.

### ISIC Ablations

```bash
python experiments/isic_paper_experiments.py --suite ablations --epochs 40 --batch_size 32 --seeds 42 43 44 --no_save_model
```

### CIFAR-100-LT

```bash
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 10 --epochs 100 --batch_size 128 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 50 --epochs 100 --batch_size 128 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark cifar --ratio 100 --epochs 100 --batch_size 128 --seeds 42 43 44
```

### MVTec Classifier Protocol

```bash
python experiments/generalization_paper_suite.py --benchmark mvtec --category hazelnut --epochs 20 --batch_size 16 --seeds 42 43 44
python experiments/generalization_paper_suite.py --benchmark mvtec --category bottle --epochs 20 --batch_size 16 --seeds 42 43 44
```

### MVTec PatchCore-Style Reference

```bash
python experiments/mvtec_patchcore_reference.py --category hazelnut --batch_size 16 --seeds 42 43 44
python experiments/mvtec_patchcore_reference.py --category bottle --batch_size 16 --seeds 42 43 44
```

### MVTec SimpleNet-Style Reference

```bash
python experiments/mvtec_simplenet_reference.py --category hazelnut --batch_size 16 --epochs 10 --seeds 42 43 44
python experiments/mvtec_simplenet_reference.py --category bottle --batch_size 16 --epochs 10 --seeds 42 43 44
```

### All-In-One Kaggle Suite

```bash
python experiments/run_kaggle_paper_suite.py --isic_suite all --no_save_model --keep_going
```

Useful skip flags:

```bash
--skip_mvtec_reference
--skip_mvtec_simplenet
--skip_hardware
--skip_summary
```

## Files Added Or Modified For This Audit

Important files to push:

- `guds_edl_core.py`
- `experiments/isic_paper_experiments.py`
- `experiments/generalization_paper_suite.py`
- `experiments/metrics_ext.py`
- `experiments/mvtec_ad_runner.py`
- `experiments/mvtec_patchcore_reference.py`
- `experiments/mvtec_simplenet_reference.py`
- `experiments/run_kaggle_paper_suite.py`
- `experiments/README.md`
- `experiments/PAPER_EXPERIMENT_COVERAGE.md`
- `experiments/KAGGLE_GITHUB_SETUP.md`
- `experiments/KAGGLE_SEPARATE_NOTEBOOK_GUIDE.md`
- `experiments/BASELINE_SOTA_AUDIT_NOTES.md`

## Recommended Paper Wording

Safe wording:

`We compare GUDS-EDL against strong controlled baselines under the same split,
backbone family, seed protocol, epoch budget, calibration, and metrics. The
suite includes long-tailed classification baselines, evidential baselines,
structured sparse baselines, and MVTec anomaly-detection reference methods.`

Avoid:

`GUDS-EDL outperforms all SOTA methods.`

Use only if official/tuned external baselines are later run:

`GUDS-EDL is competitive with SOTA methods under the official benchmark
protocol.`
