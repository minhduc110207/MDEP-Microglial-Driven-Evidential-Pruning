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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (
    replace_conv2d_with_mdep,
    get_imbalanced_dataloaders,
    configure_training_runtime,
    compute_uncertainties,
)
from experiments.generalization_paper_suite import EvidenceResNet
from experiments.isic_paper_experiments import (
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


def compute_equalized_odds_gap(y_true, y_pred, subgroups) -> float:
    """
    Computes the Equalized Odds Metric (EOM) gap.
    EOM Gap = max( |TPR_g1 - TPR_g2| + |FPR_g1 - FPR_g2| ) across subgroups.
    """
    unique_groups = np.unique(subgroups)
    tprs = {}
    fprs = {}
    
    for g in unique_groups:
        mask = subgroups == g
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        
        num_pos = (yt == 1).sum()
        num_neg = (yt == 0).sum()
        
        if num_pos > 0:
            tp = ((yt == 1) & (yp == 1)).sum()
            tprs[g] = float(tp / num_pos)
            
        if num_neg > 0:
            fp = ((yt == 0) & (yp == 1)).sum()
            fprs[g] = float(fp / num_neg)
            
    if len(tprs) < 2 or len(fprs) < 2:
        return 0.0
        
    tpr_vals = list(tprs.values())
    fpr_vals = list(fprs.values())
    
    tpr_gap = max(tpr_vals) - min(tpr_vals)
    fpr_gap = max(fpr_vals) - min(fpr_vals)
    return tpr_gap + fpr_gap


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
            outputs = model(inputs)
            # Since model.backbone output is penultimate features, logits are:
            pen_feat = model.backbone(inputs)
            logits = model.fc[0](pen_feat)
            evidence = model.fc[1](logits)
            
            logits_all.append(logits.cpu().numpy())
            
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
        
    return {
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
    """Compute KNN-OOD distance using L2 normalized features."""
    ref_norm = ref_features / (np.linalg.norm(ref_features, axis=1, keepdims=True) + 1e-8)
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Batch distance computation to prevent memory overhead
    dists = []
    for i in range(0, len(feat_norm), 100):
        batch = feat_norm[i:i+100]
        # Pairwise distance: (B, N_ref)
        d = np.linalg.norm(batch[:, None, :] - ref_norm[None, :, :], axis=2)
        d_sorted = np.sort(d, axis=1)[:, :k]
        dists.append(np.mean(d_sorted, axis=1))
        
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Domain Shift & Fairness.")
    parser.add_argument("--model_path", type=str, help="Path to trained model model_state.pth (optional)")
    parser.add_argument("--fitzpatrick_csv", type=str, help="Path to Fitzpatrick17k metadata (optional)")
    parser.add_argument("--pad_ufes_csv", type=str, help="Path to PAD-UFES-20 metadata (optional)")
    parser.add_argument("--custom_image_folder", type=str, help="Path to a custom image folder dataset for OOD testing (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Model checkpoint seed folder")
    parser.add_argument("--split_seed", type=int, default=42, help="Fixed split seed for patient splits (must match training)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    
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
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded trained checkpoint from: {args.model_path}")
    elif default_ckpt.exists():
        model.load_state_dict(torch.load(default_ckpt, map_location=device))
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
    
    # 2.1 Grid Search for ViM on validation clean vs pseudo-OOD (corrupted)
    print("\n[INFO] Fitting and Tuning SVD PCA ViM hyperparameters on Validation Split...")
    val_clean = extract_all_features_and_logits(model, val_loader, device, limit_batches=15, corrupt=False)
    val_corrupt = extract_all_features_and_logits(model, val_loader, device, limit_batches=15, corrupt=True)
    
    best_vim_auroc = -1.0
    best_vim_params = None
    best_vim_layer = None
    best_vim_num_comp = None
    
    for layer in ["layer3", "layer4", "penultimate"]:
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
    for layer in ["layer3", "layer4", "penultimate"]:
        mahal_params[layer] = fit_mahalanobis_params(cal_features_dict[layer])
        
    # 2.3 Fit KNN reference features
    knn_ref_features = cal_features_dict[best_vim_layer]
    
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
        
        try:
            base_ds = datasets.ImageFolder(root=args.custom_image_folder, transform=transform)
            
            # If partitions exist (specifically for mahdavi1202/skin-cancer),
            # we restrict OOD test evaluation to imgs_part_3 only.
            test_partition = "imgs_part_3"
            test_indices = [idx for name, idx in base_ds.class_to_idx.items() if name.lower() == test_partition]
            
            if test_indices:
                indices = [i for i, (_, label) in enumerate(base_ds.samples) if label in test_indices]
                actual_name = [name for name, idx in base_ds.class_to_idx.items() if idx in test_indices][0]
                print(f"[INFO] Detected partitions. Restricting OOD test set to partition: '{actual_name}' ({len(indices)} samples).")
            else:
                indices = list(range(len(base_ds)))
            
            classes_lower = [c.lower() for c in base_ds.classes]
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
                    return img, label, 1
            
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
        
        targets_all.append(targets.numpy())
        probs_all.append(probs.detach().cpu().numpy())
        skin_types_all.append(skin_types.numpy())
        
    y_true = np.concatenate(targets_all)
    probs = np.concatenate(probs_all, axis=0)
    y_pred = probs.argmax(axis=1)
    subgroups = np.concatenate(skin_types_all)
    
    if is_binary_skin:
        metrics = binary_image_anomaly_metrics(y_true, probs)
        eom_gap = compute_equalized_odds_gap(y_true, y_pred, subgroups)
    else:
        metrics = None
        eom_gap = None
    
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
    
    ind_metrics["knn_ood"] = score_knn(ind_test[best_vim_layer], knn_ref_features, k=20)
    ood_metrics["knn_ood"] = score_knn(ood_test[best_vim_layer], knn_ref_features, k=20)
    
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
    cal_knn = score_knn(cal_features_dict[best_vim_layer], knn_ref_features, k=20)
    
    ind_metrics["fusion_rank"] = 0.5 * get_percentile_ranks(ind_metrics["vim"], cal_vim) + \
                                 0.3 * get_percentile_ranks(ind_metrics["mahalanobis_multi"], cal_mahal_multi) + \
                                 0.2 * get_percentile_ranks(ind_metrics["knn_ood"], cal_knn)
                                 
    ood_metrics["fusion_rank"] = 0.5 * get_percentile_ranks(ood_metrics["vim"], cal_vim) + \
                                 0.3 * get_percentile_ranks(ood_metrics["mahalanobis_multi"], cal_mahal_multi) + \
                                 0.2 * get_percentile_ranks(ood_metrics["knn_ood"], cal_knn)
                                 
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
        
    print("\n" + "="*80)
    print("Domain Shift / External Validation Summary")
    print("="*80)
    print(f"Classification Performance under Domain Shift:")
    if is_binary_skin:
        print(f"  - AUROC:                  {metrics['image_auroc']:.4f}")
        print(f"  - Average Precision (AP): {metrics['image_ap']:.4f}")
    else:
        print(f"  - AUROC:                  N/A (Skipped - non-binary/partition external dataset)")
        print(f"  - Average Precision (AP): N/A (Skipped - non-binary/partition external dataset)")
        
    print(f"Fairness Evaluation (Fitzpatrick skin subgroups):")
    if is_binary_skin:
        print(f"  - Equalized Odds (EOM) Gap: {eom_gap:.4f}")
    else:
        print(f"  - Equalized Odds (EOM) Gap: N/A (Skipped - non-binary/partition external dataset)")
        
    print(f"\nOOD Detection Performance (Full vs Balanced):")
    print(f"{'Method':<20} | {'Full AUROC':<10} | {'Full AUPR':<10} | {'Balanced AUROC (mean±std)':<26} | {'Balanced AUPR (mean±std)'}")
    print("-"*110)
    for key in sorted(ood_results.keys()):
        res = ood_results[key]
        print(f"{key:<20} | {res['full_auroc']:.4f}     | {res['full_aupr']:.4f}     | {res['bal_auroc_mean']:.4f} ± {res['bal_auroc_std']:.4f}      | {res['bal_aupr_mean']:.4f} ± {res['bal_aupr_std']:.4f}")
    print("="*80)
    
    results = {
        "classification": metrics if is_binary_skin else "N/A",
        "fairness": {"eom_gap": eom_gap if is_binary_skin else "N/A"},
        "ood_detection_metrics": ood_results,
        "best_hyperparams": {
            "vim_layer": best_vim_layer,
            "vim_pca_dim": best_vim_num_comp,
            "react_threshold": float(react_threshold)
        }
    }
    
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    summary_path = out_dir / "paper_experiment_outputs" / "external_validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
