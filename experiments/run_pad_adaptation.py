"""
Leakage-safe PAD-UFES-20 adaptation for trained ISIC GUDS-EDL checkpoints.

The runner keeps the sparse backbone immutable and learns small post-hoc heads
inside patient-grouped nested folds. It reports classification, calibration,
fairness-adjusted operating points, a detached supervised domain head, and a
multi-seed ensemble without using an outer test fold for tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.generalization_paper_suite import EvidenceResNet
from experiments.isic_paper_experiments import PROTOCOL_VERSION, json_safe, seed_everything
from experiments.run_external_validation import (
    PADUFES20MetadataDataset,
    equalized_odds_report,
    extract_all_features_and_logits,
    load_state_with_optional_ood_head,
    validate_checkpoint_protocol,
)
from guds_edl_core import configure_training_runtime, get_imbalanced_dataloaders, replace_conv2d_with_mdep


DIAGNOSES = ("NEV", "BCC", "ACK", "SEK", "SCC", "MEL")
MALIGNANT_DIAGNOSES = {"BCC", "SCC", "MEL"}
FEATURE_LAYERS = ("layer3", "layer4", "penultimate")


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


def positive_ece(y_true, probability, n_bins=15):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(y_true), 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1.0 else probability <= high)
        if mask.any():
            value += mask.sum() / total * abs(y_true[mask].mean() - probability[mask].mean())
    return float(value)


def fit_temperature_bias(raw_logits, y_true):
    """Fit positive temperature plus scalar bias without changing ranking."""
    x = torch.as_tensor(np.asarray(raw_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y_true), dtype=torch.float64)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature, bias], lr=0.25, max_iter=80)

    def closure():
        optimizer.zero_grad()
        scaled = x / torch.exp(log_temperature).clamp_min(1e-3) + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach()), float(bias.detach())


def apply_temperature_bias(raw_logits, temperature, bias):
    return sigmoid(np.asarray(raw_logits) / max(float(temperature), 1e-6) + float(bias))


def choose_global_threshold(y_true, probability, target_sensitivity=None):
    candidates = np.unique(np.quantile(probability, np.linspace(0.0, 1.0, 201)))
    best_threshold, best_value = 0.5, -math.inf
    for threshold in candidates:
        pred = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        if target_sensitivity is None:
            value = 0.5 * (sensitivity + specificity)
        elif sensitivity >= target_sensitivity:
            value = specificity
        else:
            value = sensitivity - target_sensitivity - 1.0
        if value > best_value:
            best_value, best_threshold = value, float(threshold)
    return best_threshold


def fit_group_thresholds(y_true, probability, groups, global_threshold):
    """Match each calibration group's TPR/FPR to the global operating point."""
    y_true = np.asarray(y_true, dtype=int)
    groups = np.asarray(groups, dtype=object)
    global_pred = (probability >= global_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, global_pred, labels=[0, 1]).ravel()
    target_tpr = tp / max(tp + fn, 1)
    target_fpr = fp / max(fp + tn, 1)
    thresholds = {"__global__": float(global_threshold)}
    for group in np.unique(groups):
        mask = groups == group
        yt, score = y_true[mask], probability[mask]
        if str(group).lower() == "unknown" or len(np.unique(yt)) < 2 or min((yt == 0).sum(), (yt == 1).sum()) < 5:
            thresholds[str(group)] = float(global_threshold)
            continue
        best = (math.inf, float(global_threshold))
        for threshold in np.unique(np.quantile(score, np.linspace(0.0, 1.0, 101))):
            pred = (score >= threshold).astype(int)
            gtn, gfp, gfn, gtp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
            tpr = gtp / max(gtp + gfn, 1)
            fpr = gfp / max(gfp + gtn, 1)
            bal = 0.5 * (tpr + gtn / max(gtn + gfp, 1))
            objective = abs(tpr - target_tpr) + abs(fpr - target_fpr) + 0.05 * (1.0 - bal)
            if objective < best[0]:
                best = (objective, float(threshold))
        thresholds[str(group)] = best[1]
    return thresholds


def predict_with_group_thresholds(probability, groups, thresholds):
    global_threshold = thresholds["__global__"]
    return np.asarray([
        int(score >= thresholds.get(str(group), global_threshold))
        for score, group in zip(probability, groups)
    ])


def classification_metrics(y_true, probability, threshold):
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "nll": float(log_loss(y_true, np.column_stack([1.0 - probability, probability]), labels=[0, 1])),
        "brier": float(np.mean((probability - y_true) ** 2)),
        "ece_positive": positive_ece(y_true, probability),
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def classification_metrics_from_predictions(y_true, probability, prediction):
    """Aggregate outer-fold decisions without refitting a threshold on OOF labels."""
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "nll": float(log_loss(y_true, np.column_stack([1.0 - probability, probability]), labels=[0, 1])),
        "brier": float(np.mean((probability - y_true) ** 2)),
        "ece_positive": positive_ece(y_true, probability),
        "threshold": "fold-specific calibration threshold",
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def patient_bootstrap_ci(y_true, probability, groups, repeats, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    values = {"auroc": [], "average_precision": []}
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        yt, score = y_true[indices], probability[indices]
        if len(np.unique(yt)) < 2:
            continue
        values["auroc"].append(roc_auc_score(yt, score))
        values["average_precision"].append(average_precision_score(yt, score))
    return {
        key: {
            "low": float(np.quantile(metric_values, 0.025)),
            "high": float(np.quantile(metric_values, 0.975)),
        }
        for key, metric_values in values.items() if metric_values
    }


def make_adapter(head, c_value, seed):
    if head == "linear":
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            alpha=1.0 / max(c_value, 1e-8),
            batch_size=64,
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        )
    return make_pipeline(StandardScaler(), classifier)


def malignant_probability(model, features, target_mode):
    if target_mode == "binary":
        if hasattr(model[-1], "decision_function"):
            return sigmoid(model.decision_function(features))
        return model.predict_proba(features)[:, list(model[-1].classes_).index(1)]
    probabilities = model.predict_proba(features)
    classes = model[-1].classes_
    malignant_ids = {DIAGNOSES.index(name) for name in MALIGNANT_DIAGNOSES}
    columns = [i for i, cls in enumerate(classes) if int(cls) in malignant_ids]
    return probabilities[:, columns].sum(axis=1)


def select_layer_and_c(features_by_layer, binary_targets, adapter_targets, groups, indices, args, seed):
    layers = FEATURE_LAYERS if args.feature_layer == "auto" else (args.feature_layer,)
    split_count = min(args.inner_folds, len(np.unique(groups[indices])))
    if split_count < 2:
        return layers[-1], args.c_values[0], {}
    splitter = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=seed)
    scores = {}
    for layer in layers:
        for c_value in args.c_values:
            fold_scores = []
            for train_rel, val_rel in splitter.split(
                np.zeros(len(indices)), binary_targets[indices], groups[indices]
            ):
                train_idx, val_idx = indices[train_rel], indices[val_rel]
                model = make_adapter(args.head, c_value, seed)
                model.fit(features_by_layer[layer][train_idx], adapter_targets[train_idx])
                probability = malignant_probability(model, features_by_layer[layer][val_idx], args.target_mode)
                if len(np.unique(binary_targets[val_idx])) == 2:
                    fold_scores.append(roc_auc_score(binary_targets[val_idx], probability))
            scores[f"{layer}:C={c_value:g}"] = float(np.mean(fold_scores)) if fold_scores else float("nan")
    valid = {key: value for key, value in scores.items() if np.isfinite(value)}
    best_key = max(valid, key=valid.get) if valid else f"{layers[-1]}:C={args.c_values[0]:g}"
    layer, c_text = best_key.split(":C=")
    return layer, float(c_text), scores


def load_model_and_features(checkpoint, dataset, args, seed, device):
    model = EvidenceResNet(num_classes=2, dataset="isic", pretrained=False)
    replace_conv2d_with_mdep(model.backbone, learn_permutation=False)
    load_state_with_optional_ood_head(model, checkpoint, device)
    model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    features = extract_all_features_and_logits(model, loader, device)
    return model, {layer: features[layer] for layer in FEATURE_LAYERS}


def resolve_checkpoint(args, seed):
    if args.model_path:
        value = str(args.model_path).format(seed=seed)
        path = Path(value)
    else:
        root = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
        path = root / "paper_experiment_outputs" / "isic" / "full_guds_fair_v3_nvidia24" / f"seed_{seed}" / "model_state.pth"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found for seed {seed}: {path}")
    return path


def fit_domain_head(id_features, pad_features, seed):
    n = min(len(id_features), len(pad_features))
    rng = np.random.default_rng(seed)
    id_idx = rng.choice(len(id_features), n, replace=False)
    pad_idx = rng.choice(len(pad_features), n, replace=False)
    x = np.concatenate([id_features[id_idx], pad_features[pad_idx]])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=seed),
    )
    model.fit(x, y)
    return model


def main():
    configure_training_runtime()
    parser = argparse.ArgumentParser(description="Patient-grouped PAD adaptation and audit.")
    parser.add_argument("--pad_root", required=True)
    parser.add_argument("--pad_csv", required=True)
    parser.add_argument("--partition", default="all")
    parser.add_argument("--model_path", help="Checkpoint path; may contain {seed}.")
    parser.add_argument("--allow_legacy_checkpoint", action="store_true", help="Allow pre-fair-v3 checkpoints for diagnostic-only runs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--feature_layer", choices=["auto", *FEATURE_LAYERS], default="auto")
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--target_mode", choices=["binary", "diagnosis6"], default="diagnosis6")
    parser.add_argument("--c_values", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--fairness_groups", nargs="+", default=["gender", "age", "fitspatrick"])
    parser.add_argument("--fairness_min_group_size", type=int, default=20)
    parser.add_argument("--fairness_min_class_size", type=int, default=10)
    parser.add_argument("--target_sensitivity", type=float, default=0.80)
    parser.add_argument("--bootstrap_repeats", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_domain_head", action="store_true")
    parser.add_argument("--max_id_test_samples", type=int, default=20000)
    parser.add_argument("--output_dir")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    seed_everything(args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = PADUFES20MetadataDataset(
        args.pad_root,
        metadata_csv=args.pad_csv,
        partition=args.partition,
        transform=transform,
        subgroup_column="fitspatrick",
    )
    y = dataset.targets.astype(int)
    patient_ids = dataset.patient_ids.astype(str)
    diag_to_id = {name: index for index, name in enumerate(DIAGNOSES)}
    diagnosis_targets = np.asarray([diag_to_id[value] for value in dataset.diagnostics], dtype=int)
    adapter_targets = y if args.target_mode == "binary" else diagnosis_targets
    subgroup_values = {}
    for name in args.fairness_groups:
        try:
            subgroup_values[name] = dataset.subgroup_values(name)
        except ValueError as exc:
            print(f"[WARN] Skipping fairness group {name}: {exc}")

    seed_features = {}
    id_features = {}
    loaded_models = []
    for seed in args.seeds:
        checkpoint = resolve_checkpoint(args, seed)
        validate_checkpoint_protocol(checkpoint, args.allow_legacy_checkpoint)
        print(f"[FEATURES] seed={seed} checkpoint={checkpoint}")
        model, features = load_model_and_features(checkpoint, dataset, args, seed, device)
        seed_features[seed] = features
        if args.train_domain_head:
            loaders = get_imbalanced_dataloaders(
                batch_size=args.batch_size,
                seed=args.split_seed,
                allow_dummy_data=False,
            )
            _, _, cal_loader, test_loader, *_ = loaders
            cal = extract_all_features_and_logits(model, cal_loader, device)
            test = extract_all_features_and_logits(model, test_loader, device)
            id_features[seed] = {
                "cal": {layer: cal[layer] for layer in FEATURE_LAYERS},
                "test": {layer: test[layer][:args.max_id_test_samples] for layer in FEATURE_LAYERS},
            }
        loaded_models.append(model)
    del loaded_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    outer = StratifiedGroupKFold(
        n_splits=args.outer_folds,
        shuffle=True,
        random_state=args.split_seed,
    )
    all_predictions = np.full(len(dataset), np.nan, dtype=float)
    all_decisions = np.full(len(dataset), -1, dtype=int)
    all_adjusted_decisions = {
        name: np.full(len(dataset), -1, dtype=int)
        for name in subgroup_values
    }
    fold_rows = []
    model_artifacts = []
    domain_rows = []
    for fold, (outer_train, outer_test) in enumerate(
        outer.split(np.zeros(len(y)), y, patient_ids), start=1
    ):
        inner = StratifiedGroupKFold(n_splits=max(2, args.inner_folds), shuffle=True, random_state=args.split_seed + fold)
        fit_rel, cal_rel = next(inner.split(
            np.zeros(len(outer_train)), y[outer_train], patient_ids[outer_train]
        ))
        fit_idx, cal_idx = outer_train[fit_rel], outer_train[cal_rel]
        cal_logits, test_logits = [], []
        seed_records = []
        for seed in args.seeds:
            layer, c_value, selection = select_layer_and_c(
                seed_features[seed], y, adapter_targets, patient_ids, fit_idx, args, seed + fold
            )
            adapter = make_adapter(args.head, c_value, seed + fold)
            adapter.fit(seed_features[seed][layer][fit_idx], adapter_targets[fit_idx])
            cal_probability = malignant_probability(adapter, seed_features[seed][layer][cal_idx], args.target_mode)
            test_probability = malignant_probability(adapter, seed_features[seed][layer][outer_test], args.target_mode)
            cal_logits.append(logit(cal_probability))
            test_logits.append(logit(test_probability))
            seed_records.append({"seed": seed, "layer": layer, "C": c_value, "inner_scores": selection})
            model_artifacts.append((fold, seed, layer, c_value, adapter))

            if args.train_domain_head:
                domain = fit_domain_head(
                    id_features[seed]["cal"][layer],
                    seed_features[seed][layer][fit_idx],
                    seed + fold,
                )
                id_score = domain.predict_proba(id_features[seed]["test"][layer])[:, 1]
                pad_score = domain.predict_proba(seed_features[seed][layer][outer_test])[:, 1]
                rng = np.random.default_rng(seed + fold)
                balanced_n = min(len(id_score), len(pad_score))
                balanced_id = rng.choice(len(id_score), balanced_n, replace=False)
                balanced_pad = rng.choice(len(pad_score), balanced_n, replace=False)
                balanced_labels = np.concatenate([
                    np.zeros(balanced_n, dtype=int),
                    np.ones(balanced_n, dtype=int),
                ])
                balanced_scores = np.concatenate([
                    id_score[balanced_id],
                    pad_score[balanced_pad],
                ])
                domain_rows.append({
                    "fold": fold,
                    "seed": seed,
                    "layer": layer,
                    "auroc": float(roc_auc_score(
                        np.concatenate([np.zeros(len(id_score)), np.ones(len(pad_score))]),
                        np.concatenate([id_score, pad_score]),
                    )),
                    "average_precision_full": float(average_precision_score(
                        np.concatenate([np.zeros(len(id_score)), np.ones(len(pad_score))]),
                        np.concatenate([id_score, pad_score]),
                    )),
                    "auroc_balanced": float(roc_auc_score(balanced_labels, balanced_scores)),
                    "average_precision_balanced": float(
                        average_precision_score(balanced_labels, balanced_scores)
                    ),
                    "label": "PAD-adapted supervised domain head; not unseen-domain OOD",
                })

        ensemble_cal_logits = np.mean(cal_logits, axis=0)
        ensemble_test_logits = np.mean(test_logits, axis=0)
        temperature, bias = fit_temperature_bias(ensemble_cal_logits, y[cal_idx])
        cal_probability = apply_temperature_bias(ensemble_cal_logits, temperature, bias)
        test_probability = apply_temperature_bias(ensemble_test_logits, temperature, bias)
        threshold = choose_global_threshold(y[cal_idx], cal_probability, args.target_sensitivity)
        all_predictions[outer_test] = test_probability
        all_decisions[outer_test] = (test_probability >= threshold).astype(int)
        fold_result = {
            "fold": fold,
            "n_fit": len(fit_idx),
            "n_calibration": len(cal_idx),
            "n_test": len(outer_test),
            "temperature": temperature,
            "bias": bias,
            "global": classification_metrics(y[outer_test], test_probability, threshold),
            "models": seed_records,
            "fairness": {},
        }
        for group_name, groups in subgroup_values.items():
            group_thresholds = fit_group_thresholds(
                y[cal_idx], cal_probability, groups[cal_idx], threshold
            )
            global_pred = (test_probability >= threshold).astype(int)
            adjusted_pred = predict_with_group_thresholds(
                test_probability, groups[outer_test], group_thresholds
            )
            all_adjusted_decisions[group_name][outer_test] = adjusted_pred
            fold_result["fairness"][group_name] = {
                "global_threshold": equalized_odds_report(
                    y[outer_test],
                    global_pred,
                    groups[outer_test],
                    args.fairness_min_group_size,
                    args.fairness_min_class_size,
                ),
                "group_thresholds": group_thresholds,
                "adjusted": equalized_odds_report(
                    y[outer_test],
                    adjusted_pred,
                    groups[outer_test],
                    args.fairness_min_group_size,
                    args.fairness_min_class_size,
                ),
                "adjusted_balanced_accuracy": float(balanced_accuracy_score(y[outer_test], adjusted_pred)),
            }
        fold_rows.append(fold_result)
        print(
            f"[FOLD {fold}] AUROC={fold_result['global']['auroc']:.4f} "
            f"AP={fold_result['global']['average_precision']:.4f} "
            f"ECE={fold_result['global']['ece_positive']:.4f}"
        )

    valid = np.isfinite(all_predictions)
    overall = classification_metrics_from_predictions(
        y[valid],
        all_predictions[valid],
        all_decisions[valid],
    )
    overall["patient_bootstrap_95ci"] = patient_bootstrap_ci(
        y[valid],
        all_predictions[valid],
        patient_ids[valid],
        args.bootstrap_repeats,
        args.split_seed,
    )
    overall_fairness = {}
    for group_name, groups in subgroup_values.items():
        overall_fairness[group_name] = {
            "fold_specific_global_thresholds": equalized_odds_report(
                y[valid],
                all_decisions[valid],
                groups[valid],
                args.fairness_min_group_size,
                args.fairness_min_class_size,
            ),
            "fold_specific_group_thresholds": equalized_odds_report(
                y[valid],
                all_adjusted_decisions[group_name][valid],
                groups[valid],
                args.fairness_min_group_size,
                args.fairness_min_class_size,
            ),
            "adjusted_balanced_accuracy": float(
                balanced_accuracy_score(y[valid], all_adjusted_decisions[group_name][valid])
            ),
        }
    output_dir = Path(args.output_dir) if args.output_dir else (
        (Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT)
        / "paper_experiment_outputs" / "pad_adaptation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "path": dataset.sample_paths,
        "patient_id": patient_ids,
        "lesion_id": dataset.lesion_ids,
        "diagnostic": dataset.diagnostics,
        "target": y,
        "out_of_fold_probability": all_predictions,
    }).to_csv(output_dir / "pad_oof_predictions.csv", index=False)
    with (output_dir / "pad_adapter_models.pkl").open("wb") as handle:
        pickle.dump(model_artifacts, handle)
    result = {
        "checkpoint_protocol_version": PROTOCOL_VERSION,
        "protocol": "patient-grouped nested cross-validation",
        "target_mode": args.target_mode,
        "head": args.head,
        "seeds": args.seeds,
        "partition": args.partition,
        "n_samples": len(dataset),
        "n_patients": int(len(np.unique(patient_ids))),
        "overall_oof": overall,
        "overall_fairness": overall_fairness,
        "folds": fold_rows,
        "adapted_domain_head": domain_rows,
        "claim_boundaries": {
            "classification": "PAD-adapted external classification; not zero-shot ISIC transfer",
            "fairness": "post-hoc fairness-adjusted operating point",
            "domain_head": "supervised PAD-adapted domain detection; not unseen-domain OOD",
            "primary_unadapted_ood": "retain knn_layer3 from run_external_validation.py",
        },
    }
    (output_dir / "pad_adaptation_summary.json").write_text(
        json.dumps(json_safe(result), indent=2), encoding="utf-8"
    )
    print(f"[DONE] Overall OOF AUROC={overall['auroc']:.4f} AP={overall['average_precision']:.4f}")
    print(f"[DONE] Outputs: {output_dir}")


if __name__ == "__main__":
    main()
