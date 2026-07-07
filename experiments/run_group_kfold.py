"""
Run GroupKFold cross-validation on ISIC 2024 grouped by patient_id.
Supports dry-runs with --allow_dummy_data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (
    replace_conv2d_with_mdep,
    get_imbalanced_dataloaders,
    configure_training_runtime,
    dataloader_runtime_kwargs,
    MDEPTrainer,
    EvidentialFocalLoss,
    move_batch_to_device,
    print_sparsity_report,
)
from experiments.generalization_paper_suite import EvidenceResNet
from experiments.isic_paper_experiments import (
    EXPERIMENTS,
    seed_everything,
    run_calibration,
    prior_logit_delta,
    json_safe,
    output_root,
)
from experiments.metrics_ext import binary_image_anomaly_metrics, collect_evidential_outputs


def get_group_kfold_loaders(df, image_dir, hdf5_path, train_tf, test_tf, train_idx, val_idx, cal_idx, test_idx, batch_size, subsample_ratio, seed):
    from guds_edl_core import LongTailedDataset
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    cal_df = df.iloc[cal_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # Calculate true prior
    class_counts_true = train_df['target'].value_counts().sort_index()
    total_true = len(train_df)
    p_true = [max(class_counts_true.get(c, 0), 1e-4) for c in range(2)]
    sum_true = sum(p_true)
    p_true = [p / sum_true for p in p_true]
    
    # Subsample benign class in train to match malignant * subsample_ratio
    if subsample_ratio is not None and subsample_ratio > 0:
        train_malignant = train_df[train_df['target'] == 1]
        train_benign = train_df[train_df['target'] == 0]
        num_train_malignant = len(train_malignant)
        if num_train_malignant > 0:
            num_benign_to_sample = min(len(train_benign), num_train_malignant * subsample_ratio)
            train_benign_sampled = train_benign.sample(n=num_benign_to_sample, random_state=seed)
            train_df = pd.concat([train_malignant, train_benign_sampled]).reset_index(drop=True)
            train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            
    class_counts = train_df['target'].value_counts().sort_index()
    total = len(train_df)
    p_train = [max(class_counts.get(c, 0), 1e-4) for c in range(2)]
    sum_train = sum(p_train)
    p_train = [p / sum_train for p in p_train]
    
    train_ds = LongTailedDataset(train_df, image_dir, transform=train_tf, hdf5_path=hdf5_path)
    val_ds = LongTailedDataset(val_df, image_dir, transform=test_tf, hdf5_path=hdf5_path)
    cal_ds = LongTailedDataset(cal_df, image_dir, transform=test_tf, hdf5_path=hdf5_path)
    test_ds = LongTailedDataset(test_df, image_dir, transform=test_tf, hdf5_path=hdf5_path)
    
    loader_kwargs = dataloader_runtime_kwargs()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    cw_raw = [math.sqrt(total / class_counts.get(c, 1)) for c in range(2)]
    cw = torch.tensor([w / cw_raw[0] for w in cw_raw], dtype=torch.float32)
    
    return train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train


def run_fold(fold_idx, loaders_data, args, seed, device):
    train_loader, val_loader, cal_loader, test_loader, class_weights, p_true, p_train = loaders_data
    spec = EXPERIMENTS["full_guds"]
    
    model = EvidenceResNet(num_classes=2, dataset="isic", pretrained=False)
    if spec.sparse:
        replace_conv2d_with_mdep(model.backbone, learn_permutation=False)
    model = model.to(device)
    
    # Train
    from experiments.isic_paper_experiments import train_guds
    print(f"\n--- [FOLD {fold_idx}] Starting training ---")
    train_guds(
        model,
        train_loader,
        device,
        spec,
        class_weights,
        args.epochs,
        args.lr,
        log_every=args.log_every
    )
    
    # Calibrate
    temperature, bias, thresholds = run_calibration(
        model, cal_loader, val_loader, device, spec.calibration_mode, p_true, p_train
    )
    
    # Evaluate
    prior_delta = prior_logit_delta(p_true, p_train, 2, device=device)
    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
        
    outputs = collect_evidential_outputs(model, test_loader, device, temperature, eval_bias)
    metrics = binary_image_anomaly_metrics(outputs["y_true"], outputs["probs"])
    print(f"[FOLD {fold_idx}] AUROC: {metrics['image_auroc']:.4f} | AP: {metrics['image_ap']:.4f} | ECE: {metrics['ece_default']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run patient-grouped K-fold cross-validation on ISIC.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--subsample_ratio", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_dummy_data", action="store_true")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    
    configure_training_runtime()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # 1. Search for dataset metadata CSV
    import torchvision.transforms as transforms
    from sklearn.model_selection import train_test_split
    
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Locate dataset
    csv_path = None
    search_dirs = [os.environ.get("ISIC_ROOT"), r'E:\Testing\mdep\isic-2024-challenge', './data/isic-2024-challenge', '/kaggle/input/competitions/isic-2024-challenge', '/kaggle/input/isic-2024-challenge', '/kaggle/input']
    search_dirs = [s for s in search_dirs if s is not None]
    for sdir in search_dirs:
        cand = Path(sdir) / 'train-metadata.csv'
        if cand.exists():
            csv_path = str(cand)
            break
            
    if csv_path is None or not os.path.exists(csv_path):
        if not args.allow_dummy_data:
            raise FileNotFoundError("ISIC dataset not found. Use --allow_dummy_data for dry-runs.")
        print("[WARNING] Running GroupKFold on DUMMY data.")
        # Create dummy dataframe and splits
        dummy_df = pd.DataFrame({
            "isic_id": [f"img_{i}" for i in range(200)],
            "target": [0] * 180 + [1] * 20,
            "patient_id": [f"patient_{i // 5}" for i in range(200)]
        })
        df = dummy_df
        image_dir = None
        hdf5_path = None
    else:
        df = pd.read_csv(csv_path).dropna(subset=['patient_id']).reset_index(drop=True)
        # Locate hdf5/image_dir
        dataset_root = os.path.dirname(csv_path)
        image_dir = os.path.join(dataset_root, 'train-image')
        hdf5_path = None
        for name in ['train-image.hdf5', 'train-image.h5']:
            cand = os.path.join(dataset_root, name)
            if os.path.exists(cand):
                hdf5_path = cand
                break
                
    # Define GroupKFold based on patients
    print(f"Defining {args.folds}-Fold GroupKFold based on patient_id...")
    gkf = GroupKFold(n_splits=args.folds)
    groups = df['patient_id'].values
    
    fold_metrics = []
    
    # Loop over folds
    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(df, df['target'], groups)):
        print(f"\n================ FOLD {fold+1} / {args.folds} ================")
        
        # Sub-split train_val_idx into train, val, and cal
        # To maintain group-wise separation, we perform GroupKFold again or split groups
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        tv_groups = train_val_df['patient_id'].values
        
        # Split 80% train, 10% val, 10% cal
        inner_gkf = GroupKFold(n_splits=10)
        inner_splits = list(inner_gkf.split(train_val_df, train_val_df['target'], tv_groups))
        
        # Fold 0 and Fold 1 from inner splits can act as val and cal
        val_sub_idx = inner_splits[0][1]
        cal_sub_idx = inner_splits[1][1]
        
        # The rest is training
        val_patients = train_val_df.iloc[val_sub_idx]['patient_id'].unique()
        cal_patients = train_val_df.iloc[cal_sub_idx]['patient_id'].unique()
        
        train_sub_df = train_val_df[~train_val_df['patient_id'].isin(np.concatenate([val_patients, cal_patients]))]
        val_sub_df = train_val_df[train_val_df['patient_id'].isin(val_patients)]
        cal_sub_df = train_val_df[train_val_df['patient_id'].isin(cal_patients)]
        
        # Map indices back to original dataframe df
        train_idx_final = df[df['isic_id'].isin(train_sub_df['isic_id'])].index.values
        val_idx_final = df[df['isic_id'].isin(val_sub_df['isic_id'])].index.values
        cal_idx_final = df[df['isic_id'].isin(cal_sub_df['isic_id'])].index.values
        
        # Build dataloaders
        if csv_path is None:  # Dummy path
            # Create dummy subsets
            X = torch.randn(200, 3, 224, 224)
            Y = torch.tensor(df['target'].values, dtype=torch.long)
            full_ds = TensorDataset(X, Y)
            train_loader = DataLoader(Subset(full_ds, train_idx_final), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(Subset(full_ds, val_idx_final), batch_size=args.batch_size)
            cal_loader = DataLoader(Subset(full_ds, cal_idx_final), batch_size=args.batch_size)
            test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=args.batch_size)
            class_weights = torch.ones(2)
            p_true = [0.5, 0.5]
            p_train = [0.5, 0.5]
        else:
            train_loader, val_loader, cal_loader, test_loader, class_weights, p_true, p_train = get_group_kfold_loaders(
                df, image_dir, hdf5_path, train_tf, test_tf, train_idx_final, val_idx_final, cal_idx_final, test_idx,
                args.batch_size, args.subsample_ratio, args.seed
            )
            
        fold_res = run_fold(
            fold, (train_loader, val_loader, cal_loader, test_loader, class_weights, p_true, p_train),
            args, args.seed + fold, device
        )
        fold_metrics.append(fold_res)
        
    # Aggregate and print results
    keys = ["image_auroc", "image_ap", "ece_default", "accuracy_default"]
    summary = {}
    print(f"\n{'='*70}\nGroupKFold Cross-Validation Summary ({args.folds} Folds)\n{'='*70}")
    for key in keys:
        vals = [m[key] for m in fold_metrics if not np.isnan(m[key])]
        if vals:
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            summary[key] = {"mean": mean_val, "std": std_val}
            print(f"{key:<20} : {mean_val:.4f} ± {std_val:.4f}")
            
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    summary_path = out_dir / "paper_experiment_outputs" / "group_kfold_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
