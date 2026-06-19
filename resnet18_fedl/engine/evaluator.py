"""
============================================================================
  Evaluator — Metrics, plots, and uncertainty analysis for F-EDL.

  All metric computation is isolated here so you can call `evaluate()`
  from any training script, notebook, or ablation harness.
============================================================================
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_recall_curve, auc,
    fbeta_score, roc_curve,
)

import config as cfg
from models.fedl_head import compute_uncertainties_fd


# ── Scalar Metrics ─────────────────────────────────────────────────────

def compute_ece(confidences, accuracies, n_bins=None):
    """Expected Calibration Error with equal-width bins."""
    n_bins = n_bins or cfg.ECE_BINS
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
            continue
        b_acc  = accuracies[mask].mean()
        b_conf = confidences[mask].mean()
        b_size = mask.sum()
        ece += (b_size / len(confidences)) * abs(b_acc - b_conf)
        bin_accs.append(b_acc)
        bin_confs.append(b_conf)
        bin_sizes.append(b_size)
    return ece, bin_accs, bin_confs, bin_sizes


def compute_isic_pauc(y_true, y_prob, min_tpr=None):
    """Official ISIC 2024 metric: pAUC above a given TPR threshold."""
    min_tpr = min_tpr or cfg.PAUC_MIN_TPR
    v_gt = abs(np.asarray(y_true) - 1)
    v_pred = -1.0 * np.asarray(y_prob)
    max_fpr = abs(1.0 - min_tpr)

    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    if max_fpr is None or max_fpr == 1.0:
        return auc(fpr, tpr)

    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr_at_max_fpr = np.interp(max_fpr, x_interp, y_interp)

    fpr = np.append(fpr[:stop], max_fpr)
    tpr = np.append(tpr[:stop], tpr_at_max_fpr)
    return auc(fpr, tpr)


def compute_aurc(y_true, y_pred, confidences):
    """Area Under Risk-Coverage Curve."""
    sorted_idx = np.argsort(-confidences)
    sorted_true = y_true[sorted_idx]
    sorted_pred = y_pred[sorted_idx]

    n = len(y_true)
    errors = (sorted_true != sorted_pred).astype(float)
    cum_errors = np.cumsum(errors)
    coverages = np.arange(1, n + 1) / n
    risks = cum_errors / np.arange(1, n + 1)
    return auc(coverages, risks)


def compute_patient_level_se_top15(df, probs):
    """Patient-level sensitivity in top-15."""
    df = df.copy()
    df["prob"] = probs[:, 1]

    total = 0
    found = 0
    for _, group in df.groupby("patient_id"):
        patient_mal = group["target"].sum()
        if patient_mal == 0:
            continue
        total += patient_mal
        top_15 = group.sort_values(by="prob", ascending=False).head(15)
        found += top_15["target"].sum()

    return found / total if total > 0 else 1.0


# ── Plots ──────────────────────────────────────────────────────────────

def plot_reliability_diagram(bin_accs, bin_confs, bin_sizes, n_bins=None):
    n_bins = n_bins or cfg.ECE_BINS
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x = np.arange(n_bins)
    ax.bar(x, bin_accs, 0.8, label="Accuracy", color="#4e79a7", alpha=0.85)
    ax.bar(x, bin_confs, 0.8, label="Confidence", color="#e15759", alpha=0.4)
    ax.plot([-0.5, n_bins - 0.5], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Value")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_histogram(u_e_correct, u_e_incorrect):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if len(u_e_correct) > 0:
        ax.hist(u_e_correct, bins=40, alpha=0.6, label="Correct", color="#59a14f")
    if len(u_e_incorrect) > 0:
        ax.hist(u_e_incorrect, bins=40, alpha=0.6, label="Incorrect", color="#e15759")
    ax.set_xlabel("Epistemic Uncertainty (u_e)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Distribution")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_true, probs):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(recall, precision, color="#86bcB6", lw=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_risk_coverage_curve(y_true, y_pred, confidences):
    sorted_idx = np.argsort(-confidences)
    sorted_true = y_true[sorted_idx]
    sorted_pred = y_pred[sorted_idx]

    n = len(y_true)
    errors = (sorted_true != sorted_pred).astype(float)
    cum_errors = np.cumsum(errors)
    coverages = np.arange(1, n + 1) / n
    risks = cum_errors / np.arange(1, n + 1)
    aurc_val = auc(coverages, risks)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(coverages, risks, color="#f28e2b", lw=2,
            label=f"Risk-Coverage (AURC = {aurc_val:.4f})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (Error Rate)")
    ax.set_title("Risk-Coverage Curve")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Main Evaluation ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, test_loader, device, num_classes=None):
    """
    Full evaluation: classification metrics, calibration, uncertainty analysis, and plots.

    Args:
        model:       Trained ResNetFEDL model.
        test_loader: DataLoader for the test split.
        device:      torch.device.
        num_classes: Override for number of classes (default from config).

    Returns:
        dict of evaluation metrics.
    """
    num_classes = num_classes or cfg.NUM_CLASSES
    model.eval()

    all_targets, all_preds, all_confs = [], [], []
    all_probs, all_u_e, all_u_a = [], [], []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        alpha, pi, tau = model(inputs)
        unc = compute_uncertainties_fd(alpha, pi, tau)

        p_hat = unc["p_hat"].cpu().numpy()
        preds = p_hat.argmax(axis=1)
        confs = p_hat.max(axis=1)

        all_targets.append(targets.numpy())
        all_preds.append(preds)
        all_confs.append(confs)
        all_probs.append(p_hat)
        all_u_e.append(unc["epistemic"].cpu().numpy()[:, 0])
        all_u_a.append(unc["aleatoric"].cpu().numpy()[:, 0])

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    confs  = np.concatenate(all_confs)
    probs  = np.concatenate(all_probs, axis=0)
    u_e    = np.concatenate(all_u_e)
    u_a    = np.concatenate(all_u_a)
    correct = (y_pred == y_true).astype(float)

    # ── Threshold Optimisation ─────────────────────────────────────
    best_t_bal_acc, best_bal_acc = 0.5, 0.0
    best_t_clinical, best_spec_at_sens80 = 0.5, 0.0
    found_sens80 = False

    if num_classes == 2:
        thresholds = np.linspace(0.01, 0.99, 199)
        for t in thresholds:
            y_pred_t = (probs[:, 1] >= t).astype(int)
            bal_acc_t = balanced_accuracy_score(y_true, y_pred_t)
            if bal_acc_t > best_bal_acc:
                best_bal_acc = bal_acc_t
                best_t_bal_acc = t

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
            sens_t = tp / (tp + fn + 1e-8)
            spec_t = tn / (tn + fp + 1e-8)
            if sens_t >= 0.80:
                if spec_t > best_spec_at_sens80 or not found_sens80:
                    best_spec_at_sens80 = spec_t
                    best_t_clinical = t
                    found_sens80 = True

        if not found_sens80:
            best_sens = 0.0
            for t in thresholds:
                y_pred_t = (probs[:, 1] >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
                sens_t = tp / (tp + fn + 1e-8)
                if sens_t > best_sens:
                    best_sens = sens_t
                    best_t_clinical = t

    # Helper: threshold-dependent metrics
    def _metrics_at_threshold(t):
        if num_classes > 2:
            y_p = y_pred
        else:
            y_p = (probs[:, 1] >= t).astype(int)
        if num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
        else:
            sens, spec = 0.0, 0.0
        bal = balanced_accuracy_score(y_true, y_p)
        mf1 = f1_score(y_true, y_p, average="macro")
        f2  = fbeta_score(y_true, y_p, beta=2) if num_classes == 2 else 0.0
        return sens, spec, bal, mf1, f2

    sens_05, spec_05, bal_05, mf1_05, f2_05       = _metrics_at_threshold(0.50)
    sens_bal, spec_bal, bal_bal, mf1_bal, f2_bal    = _metrics_at_threshold(best_t_bal_acc)
    sens_cl, spec_cl, bal_cl, mf1_cl, f2_cl        = _metrics_at_threshold(best_t_clinical)

    # ── Threshold-independent Metrics ──
    ece_val, bin_accs, bin_confs, bin_sizes = compute_ece(confs, correct)

    minority_mask = y_true == 1
    m_ece = compute_ece(confs[minority_mask], correct[minority_mask])[0] if minority_mask.sum() > 0 else float("nan")

    if num_classes == 2:
        macro_auroc = roc_auc_score(y_true, probs[:, 1], average="macro")
        pr_auc = average_precision_score(y_true, probs[:, 1])
        try:
            pauc = compute_isic_pauc(y_true, probs[:, 1])
        except Exception:
            pauc = float("nan")
        aurc = compute_aurc(y_true, y_pred, confs)

        if hasattr(test_loader.dataset, "data_frame"):
            test_df = test_loader.dataset.data_frame
        else:
            test_df = pd.DataFrame({
                "target": [test_loader.dataset[i][1].item() for i in range(len(test_loader.dataset))],
                "patient_id": [f"patient_{i // 5}" for i in range(len(test_loader.dataset))],
            })
        se_top15 = compute_patient_level_se_top15(test_df, probs)
    else:
        macro_auroc = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        pr_auc = pauc = aurc = se_top15 = float("nan")

    # ── Print ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" [EVAL] CLASSIFICATION METRICS PER DECISION THRESHOLD:")
    print("-" * 80)
    print(f"{'Metric':25s} | {'Default (0.50)':16s} | {'Bal. Acc. Opt':16s} | {'Clinical Opt':16s}")
    print("-" * 80)
    print(f"{'Decision Threshold':25s} | {0.5:16.4f} | {best_t_bal_acc:16.4f} | {best_t_clinical:16.4f}")
    print(f"{'Sensitivity (Recall)':25s} | {sens_05:16.4f} | {sens_bal:16.4f} | {sens_cl:16.4f}")
    print(f"{'Specificity':25s} | {spec_05:16.4f} | {spec_bal:16.4f} | {spec_cl:16.4f}")
    print(f"{'Balanced Accuracy':25s} | {bal_05:16.4f} | {bal_bal:16.4f} | {bal_cl:16.4f}")
    print(f"{'F2-Score':25s} | {f2_05:16.4f} | {f2_bal:16.4f} | {f2_cl:16.4f}")
    print(f"{'Macro F1-Score':25s} | {mf1_05:16.4f} | {mf1_bal:16.4f} | {mf1_cl:16.4f}")
    print("-" * 80)
    print("\n [EVAL] CALIBRATION & RANKING METRICS (THRESHOLD-INDEPENDENT):")
    print("-" * 80)
    print(f"  pAUC 0.80 (ISIC 2024)      : {pauc:.4f}")
    print(f"  SE_top-15 (Patient-level)  : {se_top15:.4f}")
    print(f"  PR-AUC                     : {pr_auc:.4f}")
    print(f"  Macro-AUROC                : {macro_auroc:.4f}")
    print(f"  AURC (Risk-Coverage)       : {aurc:.4f}")
    print(f"  ECE (15 bins)              : {ece_val:.4f}")
    print(f"  Minority-ECE (Class 1)     : {m_ece:.4f}")
    print(f"  Mean Epistemic uncertainty : {u_e.mean():.4f}")
    print(f"  Mean Aleatoric uncertainty : {u_a.mean():.4f}")
    print("=" * 80 + "\n")

    # ── Plots ──────────────────────────────────────────────────────
    plot_reliability_diagram(bin_accs, bin_confs, bin_sizes)
    plot_uncertainty_histogram(u_e[correct.astype(bool)], u_e[~correct.astype(bool)])
    if num_classes == 2:
        plot_pr_curve(y_true, probs[:, 1])
        y_pred_clin = (probs[:, 1] >= best_t_clinical).astype(int)
        plot_risk_coverage_curve(y_true, y_pred_clin, confs)

    return {
        "balanced_accuracy": bal_05,
        "balanced_accuracy_opt": bal_bal,
        "sensitivity_opt": sens_cl,
        "specificity_opt": spec_cl,
        "macro_auroc": macro_auroc,
        "pauc": pauc,
        "aurc": aurc,
        "se_top15": se_top15,
        "ece": ece_val,
        "minority_ece": m_ece,
        "mean_u_e": float(u_e.mean()),
        "mean_u_a": float(u_a.mean()),
    }
