"""
Bootstrap paired ISIC metric differences from saved test prediction CSV files.

Expected input layout:

    paper_experiment_outputs/isic/<experiment>/seed_<seed>/test_predictions.csv

The script compares a target experiment against one or more baselines using
paired resampling over patient-level units by default. It is a diagnostic
confidence-interval tool; it does not replace repeated-seed reporting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    auc,
)


METRICS = ("pauc_0.80", "macro_auroc", "pr_auc", "balanced_accuracy")


def resolve_isic_root(root: Path) -> Path:
    if (root / "isic").exists():
        return root / "isic"
    return root


def load_experiment_predictions(isic_root: Path, experiment: str) -> pd.DataFrame:
    files = sorted((isic_root / experiment).glob("seed_*/test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No test_predictions.csv files found for {experiment} under {isic_root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "seed" not in frame.columns:
            frame["seed"] = int(path.parent.name.replace("seed_", ""))
        if "sample_id" not in frame.columns:
            frame["sample_id"] = frame.get("row_id", pd.Series(range(len(frame)))).astype(str)
        if "patient_id" not in frame.columns:
            frame["patient_id"] = frame["sample_id"].astype(str)
        frame["seed"] = frame["seed"].astype(int)
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["patient_id"] = frame["patient_id"].astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def compute_isic_pauc(y_true: np.ndarray, y_prob: np.ndarray, min_tpr: float = 0.80) -> float:
    """Official-style ISIC pAUC above a TPR threshold."""
    v_gt = np.abs(np.asarray(y_true, dtype=int) - 1)
    v_pred = -1.0 * np.asarray(y_prob, dtype=float)
    max_fpr = abs(1.0 - min_tpr)
    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    stop = np.searchsorted(fpr, max_fpr, "right")
    if stop <= 0 or stop >= len(fpr):
        return float("nan")
    tpr_at_max_fpr = np.interp(max_fpr, [fpr[stop - 1], fpr[stop]], [tpr[stop - 1], tpr[stop]])
    fpr_clip = np.append(fpr[:stop], max_fpr)
    tpr_clip = np.append(tpr[:stop], tpr_at_max_fpr)
    return float(auc(fpr_clip, tpr_clip))


def safe_metric(fn) -> float:
    try:
        value = float(fn())
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def metric_values(frame: pd.DataFrame, suffix: str) -> dict[str, float]:
    y_true = frame[f"y_true_{suffix}"].to_numpy(dtype=int)
    prob = frame[f"prob_1_{suffix}"].to_numpy(dtype=float)
    pred_col = f"y_pred_balanced_{suffix}"
    if pred_col in frame.columns:
        y_pred = frame[pred_col].to_numpy(dtype=int)
    else:
        y_pred = (prob >= 0.5).astype(int)
    return {
        "pauc_0.80": safe_metric(lambda: compute_isic_pauc(y_true, prob, min_tpr=0.80)),
        "macro_auroc": safe_metric(lambda: roc_auc_score(y_true, prob)),
        "pr_auc": safe_metric(lambda: average_precision_score(y_true, prob)),
        "balanced_accuracy": safe_metric(lambda: balanced_accuracy_score(y_true, y_pred)),
    }


def paired_frame(target: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    keys = ["seed", "sample_id"]
    keep = [
        "seed",
        "sample_id",
        "patient_id",
        "y_true",
        "prob_1",
        "y_pred_balanced",
    ]
    target_keep = [c for c in keep if c in target.columns]
    baseline_keep = [c for c in keep if c in baseline.columns]
    merged = target[target_keep].merge(
        baseline[baseline_keep],
        on=keys,
        how="inner",
        suffixes=("_target", "_baseline"),
    )
    if "patient_id_target" not in merged.columns:
        merged["patient_id_target"] = merged["sample_id"]
    return merged


def bootstrap_difference(
    merged: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
    unit: str,
) -> dict[str, dict[str, float]]:
    if unit == "patient":
        merged["_unit"] = merged["seed"].astype(str) + "::" + merged["patient_id_target"].astype(str)
    elif unit == "image":
        merged["_unit"] = merged["seed"].astype(str) + "::" + merged["sample_id"].astype(str)
    else:
        raise ValueError(f"Unknown bootstrap unit: {unit}")

    unit_values = merged["_unit"].unique()
    group_indices = {
        key: np.flatnonzero(merged["_unit"].to_numpy() == key)
        for key in unit_values
    }

    full_target = metric_values(merged, "target")
    full_baseline = metric_values(merged, "baseline")
    deltas = {metric: [] for metric in METRICS}

    for _ in range(n_bootstrap):
        sampled_units = rng.choice(unit_values, size=len(unit_values), replace=True)
        indices = np.concatenate([group_indices[key] for key in sampled_units])
        sample = merged.iloc[indices]
        target_values = metric_values(sample, "target")
        baseline_values = metric_values(sample, "baseline")
        for metric in METRICS:
            delta = target_values[metric] - baseline_values[metric]
            if np.isfinite(delta):
                deltas[metric].append(float(delta))

    result: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        arr = np.asarray(deltas[metric], dtype=float)
        point = full_target[metric] - full_baseline[metric]
        if len(arr):
            low, high = np.percentile(arr, [2.5, 97.5])
        else:
            low, high = float("nan"), float("nan")
        result[metric] = {
            "target_metric": full_target[metric],
            "baseline_metric": full_baseline[metric],
            "delta": float(point),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "n_bootstrap_valid": int(len(arr)),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap CIs for ISIC prediction files.")
    parser.add_argument("--root", type=Path, default=Path("/kaggle/working/paper_experiment_outputs"))
    parser.add_argument("--target", default="full_guds")
    parser.add_argument("--baseline", nargs="+", default=["dense_edl", "static_24_edl", "rigl_style_24"])
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--unit", choices=["patient", "image"], default="patient")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    isic_root = resolve_isic_root(args.root)
    if not isic_root.exists():
        isic_root = resolve_isic_root(Path("paper_experiment_outputs"))
    target = load_experiment_predictions(isic_root, args.target)
    rng = np.random.default_rng(args.seed)
    rows = []
    for baseline_name in args.baseline:
        baseline = load_experiment_predictions(isic_root, baseline_name)
        merged = paired_frame(target, baseline)
        if merged.empty:
            raise RuntimeError(f"No paired rows for {args.target} vs {baseline_name}")
        stats = bootstrap_difference(merged, args.n_bootstrap, rng, args.unit)
        unit_id = merged["patient_id_target"] if args.unit == "patient" else merged["sample_id"]
        n_units = int(merged["seed"].astype(str).str.cat(unit_id.astype(str), sep="::").nunique())
        for metric, values in stats.items():
            row = {
                "comparison": f"{args.target} - {baseline_name}",
                "metric": metric,
                "paired_unit": args.unit,
                "n_paired_rows": int(len(merged)),
                "n_units": n_units,
            }
            row.update(values)
            rows.append(row)

    output = args.output or (isic_root / "bootstrap_paired_ci.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved paired bootstrap CIs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
