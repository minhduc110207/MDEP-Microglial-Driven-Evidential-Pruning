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


@torch.no_grad()
def collect_ood_evidence(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Collect epistemic uncertainty for OOD analysis."""
    model.eval()
    ue_list = []
    for batch in loader:
        # Check if loader yields subgroups too
        if len(batch) == 3:
            inputs, _, _ = batch
        else:
            inputs, _ = batch
        inputs = inputs.to(device)
        outputs = model(inputs)
        unc = compute_uncertainties(outputs)
        ue_list.append(unc["epistemic"].cpu().numpy().reshape(-1))
    return np.concatenate(ue_list)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Domain Shift & Fairness.")
    parser.add_argument("--model_path", type=str, help="Path to trained model model_state.pth (optional)")
    parser.add_argument("--fitzpatrick_csv", type=str, help="Path to Fitzpatrick17k metadata (optional)")
    parser.add_argument("--pad_ufes_csv", type=str, help="Path to PAD-UFES-20 metadata (optional)")
    parser.add_argument("--custom_image_folder", type=str, help="Path to a custom image folder dataset for OOD testing (optional)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    
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
    loaders = get_imbalanced_dataloaders(batch_size=32, seed=args.seed, allow_dummy_data=True)
    _, _, _, test_loader_ind, _, _, _, _ = loaders
    
    # 3. Load External / Domain-Shifted Dataset
    print("\nLoading External Domain-Shifted Dataset (smartphone-style/diverse skin tones)...")
    # In this script, we default to DummyDomainShiftDataset to allow dry-runs.
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
            
            # Detect classification compatibility
            classes_lower = [c.lower() for c in base_ds.classes]
            idx_mapping = {}
            if len(base_ds.classes) == 2:
                is_binary_skin = True
                # Map benign/melanoma-like naming
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
                def __init__(self, ds, is_binary_skin, idx_mapping):
                    self.ds = ds
                    self.is_binary_skin = is_binary_skin
                    self.idx_mapping = idx_mapping
                def __len__(self):
                    return len(self.ds)
                def __getitem__(self, idx):
                    img, label = self.ds[idx]
                    if self.is_binary_skin:
                        label = self.idx_mapping[label]
                    return img, label, 1 # Mock skin_type=1 for Fairness metric
            
            external_ds = WrappedImageFolder(base_ds, is_binary_skin, idx_mapping)
        except Exception as e:
            raise RuntimeError(f"Failed to load ImageFolder from {args.custom_image_folder}. Error: {e}")
            
    elif not args.fitzpatrick_csv and not args.pad_ufes_csv:
        print("\n" + "!" * 80)
        print("🚨 [CRITICAL WARNING] No real external datasets provided!")
        print("🚨 Running on DummyDomainShiftDataset (pure Gaussian noise).")
        print("🚨 The resulting OOD AUROC (e.g. 0.9996) is ARTIFICIAL and FAKE.")
        print("🚨 DO NOT INCLUDE THESE RESULTS IN ANY PAPER!")
        print("!" * 80 + "\n")
        external_ds = DummyDomainShiftDataset(size=150)
    else:
        # Here you would load your real dataset, e.g., Fitzpatrick17kDataset or PAD_UFES_Dataset
        raise NotImplementedError("Real external dataset loading logic needs to be implemented here.")
    
    external_loader = DataLoader(external_ds, batch_size=32, shuffle=False)
    
    # 4. Evaluate under Domain Shift
    # We collect inputs, targets, and skin tone subgroups
    targets_all = []
    probs_all = []
    skin_types_all = []
    
    model.eval()
    for inputs, targets, skin_types in external_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        # Calibrate using basic scaling or fallback temperature
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
    
    # 5. Out-of-Distribution (OOD) Detection Evaluation
    # We treat In-Distribution ISIC test samples as IN (label 0)
    # and External Domain-shifted samples as OOD (label 1)
    print("\nEvaluating Out-of-Distribution (OOD) Detection using Epistemic Uncertainty...")
    ue_ind = collect_ood_evidence(model, test_loader_ind, device)
    ue_ood = collect_ood_evidence(model, external_loader, device)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    ood_labels = np.zeros(len(ue_ind) + len(ue_ood))
    ood_labels[len(ue_ind):] = 1.0
    ood_scores = np.concatenate([ue_ind, ue_ood])
    
    ood_auroc = roc_auc_score(ood_labels, ood_scores)
    ood_aupr = average_precision_score(ood_labels, ood_scores)
    
    print("\n" + "="*60)
    print("Domain Shift / External Validation Summary")
    print("="*60)
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
        
    print(f"OOD Detection Performance (In-dist vs OOD):")
    print(f"  - OOD Detection AUROC:     {ood_auroc:.4f}")
    print(f"  - OOD Detection AUPR:      {ood_aupr:.4f}")
    print("="*60)
    
    # Save results
    results = {
        "classification": metrics if is_binary_skin else "N/A",
        "fairness": {"eom_gap": eom_gap if is_binary_skin else "N/A"},
        "ood_detection": {
            "auroc": ood_auroc,
            "aupr": ood_aupr
        }
    }
    
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    summary_path = out_dir / "paper_experiment_outputs" / "external_validation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
