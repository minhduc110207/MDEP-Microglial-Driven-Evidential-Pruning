"""
Calibration and Regrowth Ablation Runner.
Runs standard, vacuity-only, ambiguity-only, and ratio-only regrowth modes
and evaluates using ECE, Brier score, NLL, and selective classification (e-AURC).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (
    replace_conv2d_with_mdep,
    get_imbalanced_dataloaders,
    configure_training_runtime,
)
from experiments.generalization_paper_suite import EvidenceResNet
from experiments.isic_paper_experiments import (
    EXPERIMENTS,
    ExperimentSpec,
    seed_everything,
    train_guds,
    run_calibration,
    prior_logit_delta,
    json_safe,
)
from experiments.metrics_ext import binary_image_anomaly_metrics, collect_evidential_outputs


def run_ablation(mode_name, args, device):
    seed_everything(args.seed)
    
    # Load dataset
    loaders = get_imbalanced_dataloaders(
        batch_size=args.batch_size,
        seed=args.seed,
        allow_dummy_data=args.allow_dummy_data
    )
    train_loader, val_loader, cal_loader, test_loader, num_classes, class_weights, p_true, p_train = loaders
    
    # Define experiment spec
    spec = EXPERIMENTS["full_guds"]
    # Dynamic modification of regrower type for this run
    spec_dict = spec.__dict__.copy()
    spec_dict["regrower_type"] = mode_name
    spec_dict["name"] = f"guds_regrow_{mode_name}"
    modified_spec = ExperimentSpec(**spec_dict)
    
    print(f"\n================ Running Regrowth Ablation: {mode_name} ================")
    model = EvidenceResNet(num_classes=2, dataset="isic", pretrained=False)
    if modified_spec.sparse:
        replace_conv2d_with_mdep(model)
    model = model.to(device)
    
    # Train
    train_guds(
        model,
        train_loader,
        device,
        modified_spec,
        class_weights,
        args.epochs,
        args.lr,
        log_every=args.log_every
    )
    
    # Post-hoc Temperature Scaling / Bias Calibration
    temperature, bias, thresholds = run_calibration(
        model, cal_loader, val_loader, device, modified_spec.calibration_mode, p_true, p_train
    )
    
    # Evaluate
    prior_delta = prior_logit_delta(p_true, p_train, 2, device=device)
    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
        
    outputs = collect_evidential_outputs(model, test_loader, device, temperature, eval_bias)
    metrics = binary_image_anomaly_metrics(outputs["y_true"], outputs["probs"])
    
    # Log detailed calibration metrics
    print(f"Results for Regrowth Mode [{mode_name}]:")
    print(f"  - AUROC:         {metrics['image_auroc']:.4f}")
    print(f"  - AP:            {metrics['image_ap']:.4f}")
    print(f"  - ECE:           {metrics['ece_default']:.4f}")
    print(f"  - Brier Score:   {metrics['brier_pos']:.4f}")
    print(f"  - NLL:           {metrics['nll']:.4f}")
    print(f"  - e-AURC:        {metrics['e_aurc']:.4f}")
    print(f"  - Risk at 90%:   {metrics['risk_at_90pct_coverage']:.4f}")
    print(f"  - Coverage @ 5%: {metrics['coverage_at_5pct_risk']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Calibration Study & Regrowth Ablations.")
    parser.add_argument("--modes", nargs="+", default=["kl_uniform", "vacuity", "ambiguity", "ratio"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_dummy_data", action="store_true")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    
    configure_training_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    results = {}
    for mode in args.modes:
        results[mode] = run_ablation(mode, args, device)
        
    # Write summary
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    summary_path = out_dir / "paper_experiment_outputs" / "calibration_ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(f"\nCompleted calibration ablation study. Summary: {summary_path}")
    
    try:
        import matplotlib.pyplot as plt
        modes = list(results.keys())
        eces = [results[m]["ece_default"] for m in modes]
        eaurcs = [results[m]["e_aurc"] for m in modes]
        
        x = np.arange(len(modes))
        width = 0.35
        
        plt.figure(figsize=(10, 5))
        plt.bar(x - width/2, eces, width, label="ECE (Calibration Error)", color="teal")
        plt.bar(x + width/2, eaurcs, width, label="e-AURC (Selective Classification)", color="orange")
        
        plt.ylabel("Score")
        plt.title("Ablation Study: Regrowth Components Comparison")
        plt.xticks(x, [m.replace("_", " ").title() for m in modes])
        plt.legend()
        plt.grid(True, linestyle=":", axis="y")
        
        fig_path = out_dir / "paper_experiment_outputs" / "calibration_ablation_summary.png"
        plt.savefig(fig_path, dpi=300)
        print(f"[INFO] Saved ablation comparison chart to: {fig_path}")
    except ImportError:
        print("[INFO] matplotlib is not installed. Skipping plot generation.")


if __name__ == "__main__":
    main()
