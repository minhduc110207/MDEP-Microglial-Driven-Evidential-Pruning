"""
Domain Shift / External Validation Runner.
Tests models under domain shift (e.g., PAD-UFES-20, Fitzpatrick17k)
and computes generalization AUROC, Equalized Odds (EOM), and OOD Detection metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (
    MDEPConv2d,
    MDEPLinear,
    generate_2_4_mask,
    replace_conv2d_with_mdep,
    get_imbalanced_dataloaders,
    configure_training_runtime,
    compute_uncertainties,
)
from experiments.generalization_paper_suite import EvidenceResNet
from experiments.isic_paper_experiments import (
    attach_ood_projection_head,
    seed_everything,
    collect_evidential_outputs,
    json_safe,
)
from experiments.metrics_ext import binary_image_anomaly_metrics


class DummyDomainShiftDataset(Dataset):
    """Generates synthetic domain-shifted data with subgroups (e.g. Fitzpatrick skin types)."""
    def __init__(self, size=100):
        # 3 channels, 224x224
        self.data = torch.randn(size, 3, 224, 224) + 0.5  # shift the mean to represent domain shift (e.g. smartphone style)
        self.targets = torch.randint(0, 2, (size,))
        # Random subgroups: skin_type (1 to 6)
        self.skin_types = np.random.randint(1, 7, size=size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.skin_types[idx]


class PADUFES20MetadataDataset(Dataset):
    """PAD-UFES-20 image dataset backed by metadata.csv diagnostic labels."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    IMAGE_COLUMNS = [
        "img_id", "image", "image_id", "filename", "file_name", "fname",
        "path", "filepath", "image_path", "image_number", "image_no",
        "photo_id", "lesion_id"
    ]
    DIAGNOSTIC_COLUMNS = ["diagnostic", "diagnosis", "label", "class", "target"]
    SUBGROUP_COLUMNS = [
        "fitzpatrick", "fitspatrick", "skin_type", "skin_tone", "skin_color",
        "gender", "sex", "age", "patient_age", "background_father",
        "background_mother", "skin_cancer_history", "cancer_history",
        "pesticide", "smoke", "drink", "region", "body_site",
        "anatom_site_general", "has_piped_water", "has_sewage_system"
    ]
    MALIGNANT_DIAGNOSTICS = {
        "mel", "melanoma", "bcc", "basal cell carcinoma",
        "scc", "squamous cell carcinoma", "malignant", "cancer"
    }
    BENIGN_DIAGNOSTICS = {
        "ack", "akiec", "actinic keratosis", "nev", "nevus", "nevi",
        "sek", "seborrheic keratosis", "benign"
    }
    UNKNOWN_VALUES = {"", "unk", "unknown", "nan", "none", "null"}
    DIAGNOSTIC_ALIASES = {
        "nev": "NEV",
        "nevus": "NEV",
        "nevi": "NEV",
        "bcc": "BCC",
        "basal cell carcinoma": "BCC",
        "ack": "ACK",
        "akiec": "ACK",
        "actinic keratosis": "ACK",
        "sek": "SEK",
        "seborrheic keratosis": "SEK",
        "scc": "SCC",
        "squamous cell carcinoma": "SCC",
        "mel": "MEL",
        "melanoma": "MEL",
    }

    def __init__(
        self,
        root: str | Path,
        metadata_csv: str | Path | None = None,
        partition: str = "imgs_part_3",
        transform=None,
        subgroup_column: str | None = None,
    ):
        self.root = Path(root)
        self.metadata_csv = Path(metadata_csv) if metadata_csv else self.root / "metadata.csv"
        self.partition = partition
        self.transform = transform
        self.df = pd.read_csv(self.metadata_csv)

        image_roots = self._resolve_image_roots(partition)
        self.partition_roots = image_roots

        self.file_lookup = self._build_file_lookup(image_roots)
        self.image_col = self._infer_image_column()
        self.diagnostic_col = self._infer_column(self.DIAGNOSTIC_COLUMNS)
        if self.diagnostic_col is None:
            raise ValueError(
                f"Could not find a PAD-UFES-20 diagnostic column in {self.metadata_csv}. "
                f"Available columns: {list(self.df.columns)}"
            )

        self.subgroup_col = self._select_subgroup_column(subgroup_column)
        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(
                f"No metadata rows matched image files under {', '.join(str(p) for p in image_roots)}. "
                f"image_col={self.image_col}, diagnostic_col={self.diagnostic_col}"
            )

        self.targets = np.array([s[1] for s in self.samples], dtype=np.int64)
        self.subgroups = np.array([s[2] for s in self.samples], dtype=object)
        self.patient_ids = np.array([r["patient_id"] for r in self.sample_records], dtype=object)
        self.lesion_ids = np.array([r["lesion_id"] for r in self.sample_records], dtype=object)
        self.diagnostics = np.array([r["diagnostic"] for r in self.sample_records], dtype=object)
        self.sample_paths = np.array([str(r["path"]) for r in self.sample_records], dtype=object)
        self.matched_diagnostic_counts = dict(self._count_matched_diagnostics())
        self.metadata_diagnostic_counts = dict(self._count_metadata_diagnostics())

    def _resolve_image_roots(self, partition: str | None) -> list[Path]:
        if not partition or partition.lower() == "all":
            roots = sorted(p for p in self.root.glob("imgs_part_*") if p.is_dir())
            if not roots:
                roots = [self.root]
        else:
            roots = [self.root / p.strip() for p in str(partition).split(",") if p.strip()]
        missing = [p for p in roots if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "PAD-UFES-20 partition path(s) not found: "
                + ", ".join(str(p) for p in missing)
            )
        return roots

    def _build_file_lookup(self, image_roots: list[Path]) -> dict[str, Path]:
        lookup = {}
        for image_root in image_roots:
            for path in image_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    lookup[path.name.lower()] = path
                    lookup[path.stem.lower()] = path
        return lookup

    def _infer_column(self, candidates: list[str]) -> str | None:
        by_lower = {c.lower(): c for c in self.df.columns}
        for candidate in candidates:
            if candidate.lower() in by_lower:
                return by_lower[candidate.lower()]
        return None

    def _match_image_path(self, value) -> Path | None:
        if pd.isna(value):
            return None
        raw = str(value).strip()
        if not raw:
            return None
        name = Path(raw).name.lower()
        keys = [name, Path(name).stem.lower()]
        if not Path(name).suffix:
            keys.extend([f"{name}{ext}" for ext in self.IMAGE_EXTENSIONS])
            keys.extend([f"img_{name}{ext}" for ext in self.IMAGE_EXTENSIONS])
        for key in keys:
            if key in self.file_lookup:
                return self.file_lookup[key]
        return None

    def _match_row_image_path(self, row) -> Path | None:
        if self.image_col is not None:
            direct = self._match_image_path(row[self.image_col])
            if direct is not None:
                return direct

        by_lower = {c.lower(): c for c in self.df.columns}
        patient_col = by_lower.get("patient_id")
        lesion_col = by_lower.get("lesion_id")
        img_cols = [
            by_lower[c] for c in ["img_id", "image_id", "image_number", "image_no", "photo_id"]
            if c in by_lower
        ]

        patient = str(row[patient_col]).strip() if patient_col and not pd.isna(row[patient_col]) else ""
        lesion = str(row[lesion_col]).strip() if lesion_col and not pd.isna(row[lesion_col]) else ""
        img_values = [str(row[c]).strip() for c in img_cols if not pd.isna(row[c])]
        if not img_values:
            img_values = [""]

        bases = []
        for img in img_values:
            combos = [
                [patient, lesion, img],
                [patient, lesion],
                [patient, img],
                [lesion, img],
            ]
            for combo in combos:
                parts = [p for p in combo if p]
                if parts:
                    bases.append("_".join(parts))
                    bases.append("-".join(parts))

        for base in bases:
            matched = self._match_image_path(base)
            if matched is not None:
                return matched
        return None

    def _infer_image_column(self) -> str:
        candidate_cols = []
        by_lower = {c.lower(): c for c in self.df.columns}
        for candidate in self.IMAGE_COLUMNS:
            if candidate.lower() in by_lower:
                candidate_cols.append(by_lower[candidate.lower()])
        candidate_cols.extend([c for c in self.df.columns if c not in candidate_cols])

        best_col = None
        best_matches = 0
        for col in candidate_cols:
            values = self.df[col].dropna().head(500)
            matches = sum(self._match_image_path(v) is not None for v in values)
            if matches > best_matches:
                best_col = col
                best_matches = matches
        if best_col is None or best_matches == 0:
            return self._infer_column(self.IMAGE_COLUMNS)
        return best_col

    def _diagnostic_to_binary(self, value) -> int | None:
        if pd.isna(value):
            return None
        text = str(value).strip().lower()
        if text in self.MALIGNANT_DIAGNOSTICS:
            return 1
        if text in self.BENIGN_DIAGNOSTICS:
            return 0
        if any(token in text for token in ["melanoma", "basal", "squamous", "malignant"]):
            return 1
        if any(token in text for token in ["nevus", "keratosis", "benign"]):
            return 0
        return None

    def _normalize_diagnostic(self, value) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip().lower()
        if text in self.DIAGNOSTIC_ALIASES:
            return self.DIAGNOSTIC_ALIASES[text]
        for alias, canonical in self.DIAGNOSTIC_ALIASES.items():
            if alias in text:
                return canonical
        return None

    def _group_identity(self, row, img_path: Path) -> tuple[str, str]:
        """Return leakage-safe patient and lesion identifiers with stable fallbacks."""
        by_lower = {c.lower(): c for c in self.df.columns}

        def clean(column_name: str) -> str:
            column = by_lower.get(column_name)
            if column is None or pd.isna(row[column]):
                return ""
            value = str(row[column]).strip()
            return "" if value.lower() in self.UNKNOWN_VALUES else value

        patient = clean("patient_id")
        lesion = clean("lesion_id")
        image_key = img_path.stem
        lesion_id = lesion or f"image:{image_key}"
        patient_id = patient or (f"lesion:{lesion}" if lesion else f"image:{image_key}")
        return patient_id, lesion_id

    def subgroup_values(self, requested: str) -> np.ndarray:
        """Format any metadata subgroup for the already matched sample order."""
        by_lower = {c.lower(): c for c in self.df.columns}
        column = by_lower.get(str(requested).lower())
        if column is None:
            raise ValueError(
                f"Requested subgroup '{requested}' not found. "
                f"Available columns: {list(self.df.columns)}"
            )
        old_column = self.subgroup_col
        self.subgroup_col = column
        try:
            values = [self._format_subgroup(self.df.loc[r["row_index"]]) for r in self.sample_records]
        finally:
            self.subgroup_col = old_column
        return np.asarray(values, dtype=object)

    def _count_metadata_diagnostics(self):
        counts = {}
        for value in self.df[self.diagnostic_col].fillna("unknown"):
            key = str(value).strip() or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _count_matched_diagnostics(self):
        counts = {}
        for _, row in self.df.iterrows():
            if self._match_row_image_path(row) is None:
                continue
            target = self._diagnostic_to_binary(row[self.diagnostic_col])
            if target is None:
                continue
            key = str(row[self.diagnostic_col]).strip() or "unknown"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _select_subgroup_column(self, requested: str | None) -> str | None:
        if requested and requested.lower() != "auto":
            by_lower = {c.lower(): c for c in self.df.columns}
            if requested.lower() not in by_lower:
                raise ValueError(
                    f"Requested fairness group '{requested}' not found. "
                    f"Available columns: {list(self.df.columns)}"
                )
            return by_lower[requested.lower()]
        return self._infer_column(self.SUBGROUP_COLUMNS)

    def _format_subgroup(self, row) -> str:
        if self.subgroup_col is None or pd.isna(row[self.subgroup_col]):
            return "unknown"
        value = row[self.subgroup_col]
        col = self.subgroup_col.lower()
        text = str(value).strip()
        if text.lower() in self.UNKNOWN_VALUES:
            return "unknown"
        if "age" in col:
            try:
                age = float(value)
                if age < 40:
                    return "age_<40"
                if age < 60:
                    return "age_40_59"
                return "age_60_plus"
            except Exception:
                pass
        if col in {"fitzpatrick", "fitspatrick", "skin_type"}:
            try:
                skin_type = int(float(text))
                if skin_type <= 2:
                    return "fitz_1_2"
                if skin_type <= 4:
                    return "fitz_3_4"
                return "fitz_5_6"
            except Exception:
                return "unknown"
        if col in {"gender", "sex"}:
            value_norm = text.lower()
            if value_norm.startswith("f"):
                return "gender=female"
            if value_norm.startswith("m"):
                return "gender=male"
            return "unknown"
        return f"{self.subgroup_col}={text}"

    def _build_samples(self):
        samples = []
        records = []
        skipped_missing_file = 0
        skipped_unknown_label = 0
        for row_index, row in self.df.iterrows():
            img_path = self._match_row_image_path(row)
            if img_path is None:
                skipped_missing_file += 1
                continue
            target = self._diagnostic_to_binary(row[self.diagnostic_col])
            diagnostic = self._normalize_diagnostic(row[self.diagnostic_col])
            if target is None or diagnostic is None:
                skipped_unknown_label += 1
                continue
            subgroup = self._format_subgroup(row)
            patient_id, lesion_id = self._group_identity(row, img_path)
            samples.append((img_path, target, subgroup))
            records.append({
                "row_index": row_index,
                "path": img_path,
                "target": int(target),
                "diagnostic": diagnostic,
                "patient_id": patient_id,
                "lesion_id": lesion_id,
                "subgroup": subgroup,
            })
        self.skipped_missing_file = skipped_missing_file
        self.skipped_unknown_label = skipped_unknown_label
        self.sample_records = records
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target, subgroup = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target, subgroup


def equalized_odds_report(
    y_true,
    y_pred,
    subgroups,
    min_group_size: int = 1,
    min_class_size: int = 1,
) -> dict:
    """Compute Equalized Odds only when subgroup/class support is valid."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    subgroups = np.asarray(subgroups).astype(str)
    min_group_size = max(1, int(min_group_size))
    min_class_size = max(1, int(min_class_size))

    group_stats = {}
    for g in sorted(np.unique(subgroups)):
        mask = subgroups == g
        yt = y_true[mask]
        yp = y_pred[mask]
        num_pos = int((yt == 1).sum())
        num_neg = int((yt == 0).sum())
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        group_stats[str(g)] = {
            "n": int(mask.sum()),
            "positive": num_pos,
            "negative": num_neg,
            "predicted_positive": int((yp == 1).sum()),
            "tpr": float(tp / num_pos) if num_pos > 0 else None,
            "fpr": float(fp / num_neg) if num_neg > 0 else None,
            "is_unknown": str(g).lower() == "unknown",
        }

    non_unknown_groups = {
        g: s for g, s in group_stats.items()
        if not s["is_unknown"] and s["n"] >= min_group_size
    }
    counts = binary_class_counts(y_true)
    report = {
        "eom_gap": None,
        "status": "ok",
        "class_counts": counts,
        "min_group_size": min_group_size,
        "min_class_size": min_class_size,
        "group_stats": group_stats,
        "eligible_groups": [],
    }

    if counts["negative"] == 0 or counts["positive"] == 0:
        report["status"] = "undefined_single_class_external_labels"
        return report
    if len(non_unknown_groups) < 2:
        report["status"] = "undefined_less_than_two_observed_subgroups"
        return report

    eligible = {
        g: s for g, s in non_unknown_groups.items()
        if s["positive"] >= min_class_size and s["negative"] >= min_class_size
    }
    report["eligible_groups"] = sorted(eligible)
    if len(eligible) < 2:
        report["status"] = (
            "undefined_missing_positive_or_negative_by_subgroup"
            if min_class_size == 1
            else "undefined_insufficient_class_support_by_subgroup"
        )
        return report

    tpr_vals = [s["tpr"] for s in eligible.values()]
    fpr_vals = [s["fpr"] for s in eligible.values()]
    tpr_gap = max(tpr_vals) - min(tpr_vals)
    fpr_gap = max(fpr_vals) - min(fpr_vals)
    report["eom_gap"] = float(tpr_gap + fpr_gap)
    report["tpr_gap"] = float(tpr_gap)
    report["fpr_gap"] = float(fpr_gap)
    return report


def compute_equalized_odds_gap(y_true, y_pred, subgroups) -> float | None:
    return equalized_odds_report(y_true, y_pred, subgroups)["eom_gap"]


def binary_class_counts(y_true: np.ndarray) -> dict[str, int]:
    y = np.asarray(y_true).astype(int)
    return {
        "negative": int((y == 0).sum()),
        "positive": int((y == 1).sum()),
    }


def has_both_binary_classes(y_true: np.ndarray) -> bool:
    counts = binary_class_counts(y_true)
    return counts["negative"] > 0 and counts["positive"] > 0


def batch_to_numpy(batch_value):
    if torch.is_tensor(batch_value):
        return batch_value.detach().cpu().numpy()
    return np.asarray(batch_value)


def extract_all_features_and_logits(model, loader, device, limit_batches=None, corrupt=False):
    """Extract intermediate features (layer3, layer4, penultimate) and predictions in a single pass."""
    model.eval()
    
    features_layer3 = []
    features_layer4 = []
    features_penultimate = []
    logits_all = []
    vacuity_all = []
    ambiguity_all = []
    entropy_all = []
    max_conf_all = []
    energy_all = []
    ood_projection_all = []
    ood_domain_all = []
    
    hooks = []
    
    def hook_l3(module, inp, out):
        features_layer3.append(torch.mean(out, dim=(2, 3)).detach().cpu().numpy())
    def hook_l4(module, inp, out):
        features_layer4.append(torch.mean(out, dim=(2, 3)).detach().cpu().numpy())
    def hook_pen(module, inp, out):
        features_penultimate.append(out.detach().cpu().numpy())
        
    hooks.append(model.backbone.layer3.register_forward_hook(hook_l3))
    hooks.append(model.backbone.layer4.register_forward_hook(hook_l4))
    hooks.append(model.backbone.register_forward_hook(hook_pen))
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            if len(batch) == 3:
                inputs, _, _ = batch
            else:
                inputs, _ = batch
            inputs = inputs.to(device)
            
            if corrupt:
                # Add severe Gaussian noise to simulate Out-of-Distribution shift
                inputs = inputs + 0.45 * torch.randn_like(inputs)
                inputs = torch.clamp(inputs, 0.0, 1.0)
                
            # Forward pass
            # Since model.backbone output is penultimate features, logits are:
            pen_feat = model.backbone(inputs)
            logits = model.fc[0](pen_feat)
            evidence = model.fc[1](logits)
            
            logits_all.append(logits.cpu().numpy())
            if hasattr(model, "ood_projection_head"):
                projection, domain_logits = model.ood_projection_head(pen_feat.float(), return_projection=True)
                ood_projection_all.append(projection.cpu().numpy())
                ood_domain_all.append(domain_logits.cpu().numpy().reshape(-1))
            
            unc = compute_uncertainties(evidence)
            alpha = unc["alpha"]
            S = unc["S"]
            
            vacuity_all.append(unc["epistemic"].cpu().numpy().reshape(-1))
            ambiguity_all.append(unc["aleatoric"].cpu().numpy().reshape(-1))
            
            probs = alpha / S
            probs_np = probs.cpu().numpy()
            entropy_all.append(-np.sum(probs_np * np.log(probs_np + 1e-12), axis=1))
            max_conf_all.append(-np.max(probs_np, axis=1))
            energy_all.append(-torch.logsumexp(logits, dim=1).cpu().numpy().reshape(-1))
            
    for h in hooks:
        h.remove()
        
    result = {
        "layer3": np.concatenate(features_layer3, axis=0),
        "layer4": np.concatenate(features_layer4, axis=0),
        "penultimate": np.concatenate(features_penultimate, axis=0),
        "logits": np.concatenate(logits_all, axis=0),
        "vacuity": np.concatenate(vacuity_all),
        "ambiguity": np.concatenate(ambiguity_all),
        "entropy": np.concatenate(entropy_all),
        "max_conf": np.concatenate(max_conf_all),
        "energy": np.concatenate(energy_all)
    }
    if ood_projection_all:
        result["ood_projection"] = np.concatenate(ood_projection_all, axis=0)
        result["ood_head_domain"] = np.concatenate(ood_domain_all)
    return result


def fit_vim_params(features, logits, num_components=128):
    """Fit ViM parameters on training/calibration features."""
    mean = np.mean(features, axis=0)
    X_centered = features - mean
    
    # SVD for principal components
    U, S_vals, Vt = np.linalg.svd(X_centered, full_matrices=False)
    V = Vt[:num_components].T
    
    # Residuals
    proj = X_centered @ V
    residuals = X_centered - proj @ V.T
    residual_norms = np.linalg.norm(residuals, axis=1)
    
    max_logits = np.max(logits, axis=1)
    gamma = float(np.sum(max_logits) / (np.sum(residual_norms) + 1e-8))
    
    return {
        "mean": mean,
        "V": V,
        "gamma": gamma
    }


def score_vim(features, logits, params):
    """Compute ViM score."""
    mean, V, gamma = params["mean"], params["V"], params["gamma"]
    X_centered = features - mean
    proj = X_centered @ V
    residuals = X_centered - proj @ V.T
    residual_norms = np.linalg.norm(residuals, axis=1)
    
    max_logits = np.max(logits, axis=1, keepdims=True)
    logsumexp = max_logits.reshape(-1) + np.log(np.sum(np.exp(logits - max_logits), axis=1))
    return gamma * residual_norms - logsumexp


def fit_mahalanobis_params(features):
    """Fit Mahalanobis parameters."""
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    inv_cov = np.linalg.pinv(cov + 1e-4 * np.eye(cov.shape[0]))
    return {"mean": mean, "inv_cov": inv_cov}


def score_mahalanobis(features, params):
    """Compute Mahalanobis distance."""
    diff = features - params["mean"]
    return np.sum(diff @ params["inv_cov"] * diff, axis=1)


def score_knn(features, ref_features, k=20):
    """Compute KNN-OOD distance using PyTorch GPU-accelerated matrix multiplication."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert arrays to PyTorch tensors
    feat_t = torch.from_numpy(features).float().to(device)
    ref_t = torch.from_numpy(ref_features).float().to(device)
    
    # L2 normalize features
    feat_t = feat_t / (torch.norm(feat_t, dim=1, keepdim=True) + 1e-8)
    ref_t = ref_t / (torch.norm(ref_t, dim=1, keepdim=True) + 1e-8)
    
    dists = []
    batch_size = 1000  # Compute in larger batches on GPU (extremely fast)
    for i in range(0, len(feat_t), batch_size):
        batch = feat_t[i:i+batch_size]
        # Efficient pairwise Euclidean distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x * y^T
        # Since both are L2 normalized, ||x||^2 = 1 and ||y||^2 = 1
        # Therefore, ||x - y||^2 = 2 - 2 * x * y^T
        sim = torch.mm(batch, ref_t.t()) # Shape: (B, N_ref)
        d_sq = torch.clamp(2.0 - 2.0 * sim, min=0.0)
        d = torch.sqrt(d_sq)
        
        # Sort and take top-k smallest distances
        d_sorted, _ = torch.topk(d, k, dim=1, largest=False)
        dists.append(torch.mean(d_sorted, dim=1).cpu().numpy())
        
    return np.concatenate(dists)


def score_react(features, model, threshold):
    """Post-hoc ReAct clipped evidential/energy scores."""
    W = model.fc[0].weight.detach().cpu().numpy()
    b = model.fc[0].bias.detach().cpu().numpy()
    
    clipped_feat = np.minimum(features, threshold)
    logits = clipped_feat @ W.T + b
    evidence = np.log(1.0 + np.exp(np.clip(logits, -20, 20))) + np.clip(logits - 20, 0, None)
    
    alpha = evidence + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    probs = alpha / S
    
    vacuity = (2.0 / S).reshape(-1)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    energy = -np.log(np.sum(np.exp(logits), axis=1) + 1e-12)
    
    return {
        "vacuity": vacuity,
        "entropy": entropy,
        "energy": energy
    }


def score_ash(features, model, keep_ratio=0.2):
    """Post-hoc ASH shaped activation evidential/energy scores."""
    W = model.fc[0].weight.detach().cpu().numpy()
    b = model.fc[0].bias.detach().cpu().numpy()
    
    ash_feat = np.zeros_like(features)
    for i in range(features.shape[0]):
        x = features[i]
        k = max(1, int(len(x) * keep_ratio))
        thresh = np.partition(x, -k)[-k]
        ash_feat[i] = x * (x >= thresh)
        
    logits = ash_feat @ W.T + b
    evidence = np.log(1.0 + np.exp(np.clip(logits, -20, 20))) + np.clip(logits - 20, 0, None)
    
    alpha = evidence + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    probs = alpha / S
    
    vacuity = (2.0 / S).reshape(-1)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    energy = -np.log(np.sum(np.exp(logits), axis=1) + 1e-12)
    
    return {
        "vacuity": vacuity,
        "entropy": entropy,
        "energy": energy
    }


def get_percentile_ranks(scores, ref_scores):
    """Map scores to their percentile rank on the reference set."""
    ranks = []
    # Sorted reference scores for fast searching
    sorted_ref = np.sort(ref_scores)
    for s in scores:
        idx = np.searchsorted(sorted_ref, s)
        ranks.append(idx / len(sorted_ref))
    return np.array(ranks)


def load_state_with_optional_ood_head(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    if any(str(k).startswith("ood_projection_head.") for k in state_dict.keys()):
        attach_ood_projection_head(model)
        print("[INFO] Detected detached OOD projection head in checkpoint; enabling projection-head OOD scores.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[WARN] Ignored unexpected checkpoint keys: {unexpected}")
    missing = [k for k in missing if not k.startswith("ood_projection_head.")]
    if missing:
        print(f"[WARN] Missing checkpoint keys: {missing}")
    sparse_layers = 0
    valid_blocks = 0
    total_blocks = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, (MDEPConv2d, MDEPLinear)):
                continue
            module.warmup = False
            module.mask.copy_(generate_2_4_mask(module.scores))
            rows = module.mask.reshape(module.mask.shape[0], -1)
            complete = (rows.shape[1] // 4) * 4
            if complete:
                blocks = rows[:, :complete].reshape(-1, 4)
                valid_blocks += int((blocks.sum(dim=1) == 2).sum().item())
                total_blocks += int(blocks.shape[0])
            sparse_layers += 1
    if sparse_layers:
        if valid_blocks != total_blocks:
            raise RuntimeError(
                f"Invalid sparse checkpoint reconstruction: {valid_blocks}/{total_blocks} valid 2:4 blocks."
            )
        print(
            f"[INFO] Restored sparse inference state: layers={sparse_layers}, "
            f"valid_2:4_blocks={valid_blocks}/{total_blocks}."
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Domain Shift & Fairness.")
    parser.add_argument("--model_path", type=str, help="Path to trained model model_state.pth (optional)")
    parser.add_argument("--fitzpatrick_csv", type=str, help="Path to Fitzpatrick17k metadata (optional)")
    parser.add_argument("--pad_ufes_csv", type=str, help="Path to PAD-UFES-20 metadata (optional)")
    parser.add_argument("--custom_image_folder", type=str, help="Path to a custom image folder dataset for OOD testing (optional)")
    parser.add_argument("--pad_ufes_partition", type=str, default="imgs_part_3", help="PAD-UFES-20 image partition used for external testing")
    parser.add_argument("--fairness_group", type=str, default="auto", help="PAD-UFES-20 subgroup column for Equalized Odds, or 'auto'")
    parser.add_argument("--fairness_groups", nargs="+", help="Additional PAD metadata subgroups evaluated from the same predictions.")
    parser.add_argument("--fairness_min_group_size", type=int, default=1, help="Minimum non-unknown samples required for a subgroup to be considered in Equalized Odds.")
    parser.add_argument("--fairness_min_class_size", type=int, default=1, help="Minimum positive and negative samples required inside every subgroup used for Equalized Odds.")
    parser.add_argument("--knn_primary_layer", choices=["layer3", "layer4", "penultimate"], default="layer3", help="Fixed feature layer used for the primary KNN OOD score.")
    parser.add_argument("--primary_ood_score", default="knn_layer3", help="Metric key treated as the primary OOD result in the JSON summary; use 'auto' for the fixed KNN primary layer.")
    parser.add_argument("--seed", type=int, default=42, help="Model checkpoint seed folder")
    parser.add_argument("--split_seed", type=int, default=42, help="Fixed split seed for patient splits (must match training)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    
    if not args.custom_image_folder and args.pad_ufes_csv:
        args.custom_image_folder = str(Path(args.pad_ufes_csv).resolve().parent)
        print(f"[INFO] Inferred PAD-UFES-20 root from metadata CSV: {args.custom_image_folder}")

    if not args.custom_image_folder:
        possible_paths = [
            "/kaggle/input/datasets/mahdavi1202/skin-cancer",
            "/kaggle/input/skin-cancer",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                args.custom_image_folder = p
                print(f"[INFO] Auto-detected OOD skin-cancer dataset at: {p}")
                break
    
    configure_training_runtime()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    is_binary_skin = True
    
    # 1. Load Model
    model = EvidenceResNet(num_classes=2, dataset="isic", pretrained=False)
    replace_conv2d_with_mdep(model.backbone, learn_permutation=False)
    
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    default_ckpt = out_dir / "paper_experiment_outputs" / "isic" / "full_guds" / f"seed_{args.seed}" / "model_state.pth"
    
    if args.model_path and os.path.exists(args.model_path):
        load_state_with_optional_ood_head(model, args.model_path, device)
        print(f"Loaded trained checkpoint from: {args.model_path}")
    elif default_ckpt.exists():
        load_state_with_optional_ood_head(model, default_ckpt, device)
        print(f"Loaded trained checkpoint from default path: {default_ckpt}")
    else:
        raise FileNotFoundError(
            f"No trained checkpoint found! Expected at '{default_ckpt}' or via --model_path. "
            "Refusing to run external validation on a random/untrained MDEP model to prevent misleading results."
        )
    model = model.to(device)
    
    # 2. Build In-Distribution (ISIC) loader for baseline comparison
    # We will use dummy or real depending on what is available
    print("\nLoading In-Distribution (ISIC) validation set...")
    loaders = get_imbalanced_dataloaders(batch_size=32, seed=args.split_seed, allow_dummy_data=True)
    train_loader, val_loader, cal_loader, test_loader_ind, _, _, _, _ = loaders
    
    # Extract calibration features for fitting
    print("\n[INFO] Extracting Calibration set features...")
    cal_features_dict = extract_all_features_and_logits(model, cal_loader, device)
    feature_layers = ["layer3", "layer4", "penultimate"]
    if "ood_projection" in cal_features_dict:
        feature_layers.append("ood_projection")
    
    # 2.1 Grid Search for ViM on validation clean vs pseudo-OOD (corrupted)
    print("\n[INFO] Fitting and Tuning SVD PCA ViM hyperparameters on Validation Split...")
    val_clean = extract_all_features_and_logits(model, val_loader, device, limit_batches=15, corrupt=False)
    val_corrupt = extract_all_features_and_logits(model, val_loader, device, limit_batches=15, corrupt=True)
    
    best_vim_auroc = -1.0
    best_vim_params = None
    best_vim_layer = None
    best_vim_num_comp = None
    
    for layer in feature_layers:
        max_dim = cal_features_dict[layer].shape[1]
        dims_to_try = [d for d in [32, 64, 128, 256, 512] if d < max_dim]
        for num_comp in dims_to_try:
            params = fit_vim_params(cal_features_dict[layer], cal_features_dict["logits"], num_components=num_comp)
            
            # Score validation clean and corrupt
            val_clean_scores = score_vim(val_clean[layer], val_clean["logits"], params)
            val_corrupt_scores = score_vim(val_corrupt[layer], val_corrupt["logits"], params)
            
            # Compute AUROC distinguishing clean (0) from corrupt (1)
            labels = np.concatenate([np.zeros(len(val_clean_scores)), np.ones(len(val_corrupt_scores))])
            scores = np.concatenate([val_clean_scores, val_corrupt_scores])
            
            from sklearn.metrics import roc_auc_score
            try:
                auroc = roc_auc_score(labels, scores)
                sep_auroc = max(auroc, 1.0 - auroc)
                if sep_auroc > best_vim_auroc:
                    best_vim_auroc = sep_auroc
                    best_vim_params = params
                    best_vim_layer = layer
                    best_vim_num_comp = num_comp
            except Exception:
                pass
                
    print(f"[INFO] Best ViM Configuration: Layer={best_vim_layer}, PCA_dim={best_vim_num_comp}, Validation AUROC={best_vim_auroc:.4f}")
    
    # 2.2 Fit Mahalanobis parameters
    mahal_params = {}
    for layer in feature_layers:
        mahal_params[layer] = fit_mahalanobis_params(cal_features_dict[layer])
        
    # 2.3 Fit KNN reference features. KNN is tied to a fixed layer rather than
    # the best ViM layer because the ViM pseudo-OOD selection is not a stable
    # proxy for external PAD-UFES-20 separation across seeds.
    knn_primary_layer = args.knn_primary_layer
    knn_ref_features = cal_features_dict[knn_primary_layer]
    knn_ref_features_by_layer = {layer: cal_features_dict[layer] for layer in ["layer3", "layer4", "penultimate"]}
    
    # 2.4 Fit ReAct threshold (95th percentile of calibration activations)
    react_threshold = np.percentile(cal_features_dict["penultimate"], 95)
    
    # 3. Load External / Domain-Shifted Dataset
    print("\nLoading External Domain-Shifted Dataset (smartphone-style/diverse skin tones)...")
    if args.custom_image_folder:
        print(f"\n[INFO] Loading custom external dataset from {args.custom_image_folder}...")
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        root = Path(args.custom_image_folder)
        metadata_csv = Path(args.pad_ufes_csv) if args.pad_ufes_csv else root / "metadata.csv"
        if args.pad_ufes_csv and not metadata_csv.exists():
            raise FileNotFoundError(f"Explicit --pad_ufes_csv path does not exist: {metadata_csv}")
        if metadata_csv.exists():
            external_ds = PADUFES20MetadataDataset(
                root=root,
                metadata_csv=metadata_csv,
                partition=args.pad_ufes_partition,
                transform=transform,
                subgroup_column=args.fairness_group,
            )
            is_binary_skin = True
            n_pos = int(external_ds.targets.sum())
            n_neg = int(len(external_ds.targets) - n_pos)
            unique_groups = sorted(str(g) for g in np.unique(external_ds.subgroups))
            print("[INFO] Loaded PAD-UFES-20 with metadata.csv.")
            print(f"[INFO] Metadata CSV: {metadata_csv}")
            print(f"[INFO] Partition: {args.pad_ufes_partition}")
            print(f"[INFO] Image roots: {[str(p) for p in external_ds.partition_roots]}")
            print(f"[INFO] Image column: {external_ds.image_col}")
            print(f"[INFO] Diagnostic column: {external_ds.diagnostic_col}")
            print(f"[INFO] Fairness subgroup column: {external_ds.subgroup_col or 'unknown'}")
            print(f"[INFO] Matched samples: {len(external_ds)} (malignant={n_pos}, non_malignant={n_neg})")
            print(f"[INFO] Metadata diagnostic counts: {external_ds.metadata_diagnostic_counts}")
            print(f"[INFO] Matched diagnostic counts: {external_ds.matched_diagnostic_counts}")
            print(f"[INFO] Skipped metadata rows: missing_file={external_ds.skipped_missing_file}, unknown_label={external_ds.skipped_unknown_label}")
            print(f"[INFO] Subgroups: {unique_groups[:12]}{' ...' if len(unique_groups) > 12 else ''}")
        else:
            try:
                base_ds = datasets.ImageFolder(root=args.custom_image_folder, transform=transform)
            
                # If partitions exist (specifically for mahdavi1202/skin-cancer),
                # we restrict OOD test evaluation to the configured PAD-UFES partition.
                test_partition = args.pad_ufes_partition
                test_indices = [idx for name, idx in base_ds.class_to_idx.items() if name.lower() == test_partition.lower()]
            
                if test_indices:
                    indices = [i for i, (_, label) in enumerate(base_ds.samples) if label in test_indices]
                    actual_name = [name for name, idx in base_ds.class_to_idx.items() if idx in test_indices][0]
                    print(f"[INFO] Detected partitions. Restricting OOD test set to partition: '{actual_name}' ({len(indices)} samples).")
                else:
                    indices = list(range(len(base_ds)))
            
                idx_mapping = {}
                if len(base_ds.classes) == 2:
                    is_binary_skin = True
                    for name, idx in base_ds.class_to_idx.items():
                        if any(x in name.lower() for x in ["malignant", "melanoma", "cancer", "1"]):
                            idx_mapping[idx] = 1
                        else:
                            idx_mapping[idx] = 0
                    print(f"[INFO] Detected binary skin classes: {base_ds.class_to_idx}. Mapping to: {idx_mapping}")
                else:
                    is_binary_skin = False
                    print(f"[INFO] Multi-class/partition folder classes found: {base_ds.class_to_idx}. Classification metrics will be skipped.")
                
                class WrappedImageFolder(Dataset):
                    def __init__(self, ds, indices, is_binary_skin, idx_mapping):
                        self.ds = ds
                        self.indices = indices
                        self.is_binary_skin = is_binary_skin
                        self.idx_mapping = idx_mapping
                    def __len__(self):
                        return len(self.indices)
                    def __getitem__(self, idx):
                        real_idx = self.indices[idx]
                        img, label = self.ds[real_idx]
                        if self.is_binary_skin:
                            label = self.idx_mapping[label]
                        return img, label, "unknown"
            
                external_ds = WrappedImageFolder(base_ds, indices, is_binary_skin, idx_mapping)
            except Exception as e:
                raise RuntimeError(f"Failed to load ImageFolder from {args.custom_image_folder}. Error: {e}")
            
    elif not args.fitzpatrick_csv and not args.pad_ufes_csv:
        print("\n" + "!" * 80)
        print("🚨 [CRITICAL WARNING] No real external datasets provided!")
        print("🚨 Running on DummyDomainShiftDataset (pure Gaussian noise).")
        print("🚨 The resulting OOD AUROC is ARTIFICIAL and FAKE.")
        print("🚨 DO NOT INCLUDE THESE RESULTS IN ANY PAPER!")
        print("!" * 80 + "\n")
        external_ds = DummyDomainShiftDataset(size=150)
    else:
        raise NotImplementedError("Real external dataset loading logic needs to be implemented here.")
    
    external_loader = DataLoader(external_ds, batch_size=32, shuffle=False)
    
    # 4. Evaluate under Domain Shift (Classification Performance)
    targets_all = []
    probs_all = []
    skin_types_all = []
    
    model.eval()
    for inputs, targets, skin_types in external_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        unc = compute_uncertainties(outputs)
        probs = unc["alpha"] / unc["S"]
        
        targets_all.append(batch_to_numpy(targets))
        probs_all.append(probs.detach().cpu().numpy())
        skin_types_all.append(batch_to_numpy(skin_types))
        
    y_true = np.concatenate(targets_all)
    probs = np.concatenate(probs_all, axis=0)
    y_pred = probs.argmax(axis=1)
    subgroups = np.concatenate(skin_types_all)
    external_class_counts = binary_class_counts(y_true)
    external_has_both_classes = has_both_binary_classes(y_true)
    
    if is_binary_skin:
        metrics = binary_image_anomaly_metrics(y_true, probs)
        metrics["class_counts"] = external_class_counts
        classification_status = "ok" if external_has_both_classes else "undefined_single_class_external_labels"
        fairness_report = equalized_odds_report(
            y_true,
            y_pred,
            subgroups,
            min_group_size=args.fairness_min_group_size,
            min_class_size=args.fairness_min_class_size,
        )
        eom_gap = fairness_report["eom_gap"]
        eom_status = fairness_report["status"]
        fairness_by_group = {
            str(getattr(external_ds, "subgroup_col", args.fairness_group) or args.fairness_group): fairness_report
        }
        for group_name in args.fairness_groups or []:
            fairness_by_group[group_name] = equalized_odds_report(
                y_true,
                y_pred,
                external_ds.subgroup_values(group_name),
                min_group_size=args.fairness_min_group_size,
                min_class_size=args.fairness_min_class_size,
            )
        if not external_has_both_classes:
            print(
                "[WARN] External classification labels contain only one class: "
                f"{external_class_counts}. AUROC/AP and Equalized Odds are undefined for this partition."
            )
        if eom_gap is None:
            print(f"[WARN] Equalized Odds is not reportable: {eom_status}.")
    else:
        metrics = None
        eom_gap = None
        classification_status = "skipped_non_binary_external_dataset"
        eom_status = "skipped_non_binary_external_dataset"
        fairness_report = {
            "eom_gap": None,
            "status": eom_status,
            "group_stats": {},
            "eligible_groups": [],
        }
        fairness_by_group = {}
    
    # 5. Advanced Out-of-Distribution (OOD) Detection Evaluation
    print("\nEvaluating Out-of-Distribution (OOD) Detection using various uncertainty scores...")
    ind_test = extract_all_features_and_logits(model, test_loader_ind, device)
    ood_test = extract_all_features_and_logits(model, external_loader, device)
    
    ind_metrics = {}
    ood_metrics = {}
    
    # 5.1 Standard output-level scores
    for k in ["vacuity", "ambiguity", "entropy", "max_conf", "energy"]:
        ind_metrics[k] = ind_test[k]
        ood_metrics[k] = ood_test[k]
    if "ood_head_domain" in ind_test:
        ind_metrics["ood_head_domain"] = ind_test["ood_head_domain"]
        ood_metrics["ood_head_domain"] = ood_test["ood_head_domain"]
        
    # 5.2 ReAct output-level scores (threshold-based clipping)
    react_ind = score_react(ind_test["penultimate"], model, react_threshold)
    react_ood = score_react(ood_test["penultimate"], model, react_threshold)
    for k in ["vacuity", "entropy", "energy"]:
        ind_metrics[f"react_{k}"] = react_ind[k]
        ood_metrics[f"react_{k}"] = react_ood[k]
        
    # 5.3 ASH output-level scores (channel pruning)
    ash_ind = score_ash(ind_test["penultimate"], model, keep_ratio=0.2)
    ash_ood = score_ash(ood_test["penultimate"], model, keep_ratio=0.2)
    for k in ["vacuity", "entropy", "energy"]:
        ind_metrics[f"ash_{k}"] = ash_ind[k]
        ood_metrics[f"ash_{k}"] = ash_ood[k]
        
    # 5.4 Feature-space scores
    ind_metrics["vim"] = score_vim(ind_test[best_vim_layer], ind_test["logits"], best_vim_params)
    ood_metrics["vim"] = score_vim(ood_test[best_vim_layer], ood_test["logits"], best_vim_params)
    
    ind_metrics["mahalanobis"] = score_mahalanobis(ind_test["penultimate"], mahal_params["penultimate"])
    ood_metrics["mahalanobis"] = score_mahalanobis(ood_test["penultimate"], mahal_params["penultimate"])
    if "ood_projection" in cal_features_dict:
        ind_metrics["mahalanobis_ood_projection"] = score_mahalanobis(ind_test["ood_projection"], mahal_params["ood_projection"])
        ood_metrics["mahalanobis_ood_projection"] = score_mahalanobis(ood_test["ood_projection"], mahal_params["ood_projection"])
        ind_metrics["knn_ood_projection"] = score_knn(ind_test["ood_projection"], cal_features_dict["ood_projection"], k=20)
        ood_metrics["knn_ood_projection"] = score_knn(ood_test["ood_projection"], cal_features_dict["ood_projection"], k=20)
    
    multi_m_ind = []
    multi_m_ood = []
    for layer in ["layer3", "layer4", "penultimate"]:
        c_scores = score_mahalanobis(cal_features_dict[layer], mahal_params[layer])
        c_mean, c_std = np.mean(c_scores), np.std(c_scores)
        i_scores = score_mahalanobis(ind_test[layer], mahal_params[layer])
        o_scores = score_mahalanobis(ood_test[layer], mahal_params[layer])
        multi_m_ind.append((i_scores - c_mean) / (c_std + 1e-8))
        multi_m_ood.append((o_scores - c_mean) / (c_std + 1e-8))
        
    ind_metrics["mahalanobis_multi"] = np.mean(multi_m_ind, axis=0)
    ood_metrics["mahalanobis_multi"] = np.mean(multi_m_ood, axis=0)
    
    ind_metrics["knn_ood"] = score_knn(ind_test[knn_primary_layer], knn_ref_features, k=20)
    ood_metrics["knn_ood"] = score_knn(ood_test[knn_primary_layer], knn_ref_features, k=20)
    if best_vim_layer != knn_primary_layer:
        best_vim_knn_ref = cal_features_dict[best_vim_layer]
        ind_metrics["knn_best_vim_layer"] = score_knn(ind_test[best_vim_layer], best_vim_knn_ref, k=20)
        ood_metrics["knn_best_vim_layer"] = score_knn(ood_test[best_vim_layer], best_vim_knn_ref, k=20)

    cal_knn_by_layer = {}
    ind_knn_rank_layers = []
    ood_knn_rank_layers = []
    cal_knn_rank_layers = []
    for layer, ref_features in knn_ref_features_by_layer.items():
        cal_scores = score_knn(cal_features_dict[layer], ref_features, k=20)
        ind_scores = score_knn(ind_test[layer], ref_features, k=20)
        ood_scores = score_knn(ood_test[layer], ref_features, k=20)
        cal_knn_by_layer[layer] = cal_scores
        ind_metrics[f"knn_{layer}"] = ind_scores
        ood_metrics[f"knn_{layer}"] = ood_scores
        cal_rank = get_percentile_ranks(cal_scores, cal_scores)
        cal_knn_rank_layers.append(cal_rank)
        ind_knn_rank_layers.append(get_percentile_ranks(ind_scores, cal_scores))
        ood_knn_rank_layers.append(get_percentile_ranks(ood_scores, cal_scores))

    cal_knn_multi_rank = np.mean(cal_knn_rank_layers, axis=0)
    ind_metrics["knn_multi_rank"] = np.mean(ind_knn_rank_layers, axis=0)
    ood_metrics["knn_multi_rank"] = np.mean(ood_knn_rank_layers, axis=0)
    
    # 5.5 Backward compatible Fusions
    z_params = {}
    for key in ["vacuity", "ambiguity", "entropy", "energy", "vim"]:
        z_params[key] = (np.mean(ind_metrics[key]), np.std(ind_metrics[key]))
        
    def compute_fusion_output(metrics):
        v_z = (metrics["vacuity"] - z_params["vacuity"][0]) / (z_params["vacuity"][1] + 1e-8)
        a_z = (metrics["ambiguity"] - z_params["ambiguity"][0]) / (z_params["ambiguity"][1] + 1e-8)
        e_z = (metrics["entropy"] - z_params["entropy"][0]) / (z_params["entropy"][1] + 1e-8)
        en_z = (metrics["energy"] - z_params["energy"][0]) / (z_params["energy"][1] + 1e-8)
        return v_z + a_z + e_z + en_z
        
    def compute_fusion_feature(metrics):
        v_z = (metrics["vacuity"] - z_params["vacuity"][0]) / (z_params["vacuity"][1] + 1e-8)
        en_z = (metrics["energy"] - z_params["energy"][0]) / (z_params["energy"][1] + 1e-8)
        vim_z = (metrics["vim"] - z_params["vim"][0]) / (z_params["vim"][1] + 1e-8)
        return v_z + en_z + vim_z
        
    ind_metrics["fusion_output"] = compute_fusion_output(ind_metrics)
    ood_metrics["fusion_output"] = compute_fusion_output(ood_metrics)
    
    ind_metrics["fusion_feature"] = compute_fusion_feature(ind_metrics)
    ood_metrics["fusion_feature"] = compute_fusion_feature(ood_metrics)
    
    # 5.6 Rank-Normalized Fusion
    cal_vim = score_vim(cal_features_dict[best_vim_layer], cal_features_dict["logits"], best_vim_params)
    cal_mahal_multi = np.mean([
        (score_mahalanobis(cal_features_dict[l], mahal_params[l]) - np.mean(score_mahalanobis(cal_features_dict[l], mahal_params[l]))) / (np.std(score_mahalanobis(cal_features_dict[l], mahal_params[l])) + 1e-8)
        for l in ["layer3", "layer4", "penultimate"]
    ], axis=0)
    cal_knn = score_knn(cal_features_dict[knn_primary_layer], knn_ref_features, k=20)
    
    ind_metrics["fusion_rank"] = 0.5 * get_percentile_ranks(ind_metrics["vim"], cal_vim) + \
                                 0.3 * get_percentile_ranks(ind_metrics["mahalanobis_multi"], cal_mahal_multi) + \
                                 0.2 * get_percentile_ranks(ind_metrics["knn_ood"], cal_knn)
                                 
    ood_metrics["fusion_rank"] = 0.5 * get_percentile_ranks(ood_metrics["vim"], cal_vim) + \
                                 0.3 * get_percentile_ranks(ood_metrics["mahalanobis_multi"], cal_mahal_multi) + \
                                 0.2 * get_percentile_ranks(ood_metrics["knn_ood"], cal_knn)

    # Pre-specified stronger fusion for external OOD: favor multi-layer KNN, then ViM,
    # then multi-layer Mahalanobis. Weights are fixed a priori and do not use PAD labels.
    ind_metrics["fusion_rank_strong"] = 0.45 * get_percentile_ranks(ind_metrics["knn_multi_rank"], cal_knn_multi_rank) + \
                                        0.35 * get_percentile_ranks(ind_metrics["vim"], cal_vim) + \
                                        0.20 * get_percentile_ranks(ind_metrics["mahalanobis_multi"], cal_mahal_multi)

    ood_metrics["fusion_rank_strong"] = 0.45 * get_percentile_ranks(ood_metrics["knn_multi_rank"], cal_knn_multi_rank) + \
                                        0.35 * get_percentile_ranks(ood_metrics["vim"], cal_vim) + \
                                        0.20 * get_percentile_ranks(ood_metrics["mahalanobis_multi"], cal_mahal_multi)

    primary_knn_key = f"knn_{knn_primary_layer}"
    primary_cal_knn = cal_knn_by_layer[knn_primary_layer]
    ind_metrics["fusion_rank_stable"] = 0.70 * get_percentile_ranks(ind_metrics[primary_knn_key], primary_cal_knn) + \
                                        0.30 * get_percentile_ranks(ind_metrics["mahalanobis_multi"], cal_mahal_multi)
    ood_metrics["fusion_rank_stable"] = 0.70 * get_percentile_ranks(ood_metrics[primary_knn_key], primary_cal_knn) + \
                                        0.30 * get_percentile_ranks(ood_metrics["mahalanobis_multi"], cal_mahal_multi)
                                 
    from sklearn.metrics import roc_auc_score, average_precision_score
    ood_results = {}
    np.random.seed(42)
    
    for key in ind_metrics.keys():
        scores_ind = ind_metrics[key]
        scores_ood = ood_metrics[key]
        
        # Determine sorting direction (default: high score = OOD)
        ood_labels = np.zeros(len(scores_ind) + len(scores_ood))
        ood_labels[len(scores_ind):] = 1.0
        ood_scores = np.concatenate([scores_ind, scores_ood])
        
        auroc = roc_auc_score(ood_labels, ood_scores)
        if auroc < 0.5:
            # Flip direction if lower score indicates OOD
            scores_ind = -scores_ind
            scores_ood = -scores_ood
            ood_scores = -ood_scores
            auroc = 1.0 - auroc
            
        aupr = average_precision_score(ood_labels, ood_scores)
        
        bal_aurocs = []
        bal_auprs = []
        n_balanced = min(len(scores_ind), len(scores_ood))
        
        for _ in range(10):
            idx_ind = np.random.choice(len(scores_ind), n_balanced, replace=False)
            idx_ood = np.random.choice(len(scores_ood), n_balanced, replace=False)
            bal_scores = np.concatenate([scores_ind[idx_ind], scores_ood[idx_ood]])
            bal_labels = np.zeros(2 * n_balanced)
            bal_labels[n_balanced:] = 1.0
            
            bal_aurocs.append(roc_auc_score(bal_labels, bal_scores))
            bal_auprs.append(average_precision_score(bal_labels, bal_scores))
            
        ood_results[key] = {
            "full_auroc": float(auroc),
            "full_aupr": float(aupr),
            "bal_auroc_mean": float(np.mean(bal_aurocs)),
            "bal_auroc_std": float(np.std(bal_aurocs)),
            "bal_aupr_mean": float(np.mean(bal_auprs)),
            "bal_aupr_std": float(np.std(bal_auprs))
        }

    primary_ood_score = primary_knn_key if args.primary_ood_score == "auto" else args.primary_ood_score
    if primary_ood_score not in ood_results:
        fallback = primary_knn_key if primary_knn_key in ood_results else "knn_ood"
        print(f"[WARN] Requested primary OOD score '{primary_ood_score}' not found; using '{fallback}'.")
        primary_ood_score = fallback
    primary_ood_result = ood_results[primary_ood_score]
        
    print("\n" + "="*80)
    print("Domain Shift / External Validation Summary")
    print("="*80)
    print(f"Classification Performance under Domain Shift:")
    if is_binary_skin:
        print(f"  - External class counts:  neg={external_class_counts['negative']} pos={external_class_counts['positive']}")
        if external_has_both_classes:
            print(f"  - AUROC:                  {metrics['image_auroc']:.4f}")
            print(f"  - Average Precision (AP): {metrics['image_ap']:.4f}")
        else:
            print("  - AUROC:                  N/A (single-class external labels)")
            print("  - Average Precision (AP): N/A (single-class external labels)")
    else:
        print(f"  - AUROC:                  N/A (Skipped - non-binary/partition external dataset)")
        print(f"  - Average Precision (AP): N/A (Skipped - non-binary/partition external dataset)")
        
    print(f"Fairness Evaluation (external metadata subgroups):")
    if is_binary_skin and eom_gap is not None:
        print(f"  - Equalized Odds (EOM) Gap: {eom_gap:.4f}")
    elif is_binary_skin:
        print(f"  - Equalized Odds (EOM) Gap: N/A ({eom_status})")
    else:
        print(f"  - Equalized Odds (EOM) Gap: N/A (Skipped - non-binary/partition external dataset)")
    for group_name, group_report in fairness_by_group.items():
        group_eom = group_report.get("eom_gap")
        value = f"{group_eom:.4f}" if group_eom is not None else f"N/A ({group_report.get('status')})"
        print(f"  - {group_name}: {value}")
        
    print(f"\nOOD Detection Performance (Full vs Balanced):")
    print(
        f"Primary OOD score: {primary_ood_score} | "
        f"Full AUROC={primary_ood_result['full_auroc']:.4f} | "
        f"Balanced AUROC={primary_ood_result['bal_auroc_mean']:.4f} +/- {primary_ood_result['bal_auroc_std']:.4f}"
    )
    print(f"{'Method':<20} | {'Full AUROC':<10} | {'Full AUPR':<10} | {'Balanced AUROC (mean±std)':<26} | {'Balanced AUPR (mean±std)'}")
    print("-"*110)
    for key in sorted(ood_results.keys()):
        res = ood_results[key]
        print(f"{key:<20} | {res['full_auroc']:.4f}     | {res['full_aupr']:.4f}     | {res['bal_auroc_mean']:.4f} ± {res['bal_auroc_std']:.4f}      | {res['bal_aupr_mean']:.4f} ± {res['bal_aupr_std']:.4f}")
    print("="*80)
    
    results = {
        "seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "classification": metrics if is_binary_skin else "N/A",
        "classification_status": classification_status,
        "fairness": {
            **fairness_report,
            "eom_gap": eom_gap if is_binary_skin else None,
            "status": eom_status,
        },
        "fairness_by_group": fairness_by_group,
        "primary_ood": {
            "score": primary_ood_score,
            "result": primary_ood_result,
        },
        "ood_detection_metrics": ood_results,
        "external_dataset": {
            "path": args.custom_image_folder,
            "pad_ufes_csv": args.pad_ufes_csv,
            "pad_ufes_partition": args.pad_ufes_partition,
            "pad_ufes_partition_roots": [str(p) for p in getattr(external_ds, "partition_roots", [])],
            "metadata_loader": isinstance(external_ds, PADUFES20MetadataDataset),
            "image_column": getattr(external_ds, "image_col", None),
            "diagnostic_column": getattr(external_ds, "diagnostic_col", None),
            "metadata_diagnostic_counts": getattr(external_ds, "metadata_diagnostic_counts", None),
            "matched_diagnostic_counts": getattr(external_ds, "matched_diagnostic_counts", None),
            "fairness_group_column": getattr(external_ds, "subgroup_col", None),
            "num_external_samples": len(external_ds),
            "num_external_patients": int(len(np.unique(getattr(external_ds, "patient_ids", [])))) if isinstance(external_ds, PADUFES20MetadataDataset) else None,
            "class_counts": external_class_counts if is_binary_skin else None,
        },
        "best_hyperparams": {
            "vim_layer": best_vim_layer,
            "vim_pca_dim": best_vim_num_comp,
            "knn_primary_layer": knn_primary_layer,
            "primary_ood_score": primary_ood_score,
            "react_threshold": float(react_threshold)
        }
    }
    
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    summary_dir = out_dir / "paper_experiment_outputs" / "external_validation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    group_name = str(getattr(external_ds, "subgroup_col", args.fairness_group) or "unknown").replace("/", "_")
    partition_name = str(args.pad_ufes_partition or "all").replace("/", "_")
    summary_path = summary_dir / f"external_validation_seed_{args.seed}_{partition_name}_{group_name}.json"
    legacy_path = out_dir / "paper_experiment_outputs" / "external_validation_summary.json"
    summary_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    legacy_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    print(f"Latest-run compatibility summary written to: {legacy_path}")


if __name__ == "__main__":
    main()
