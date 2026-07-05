"""
Compares RigL-style 2:4 (magnitude/gradient regrowth) vs DST-EDL (evidentiary regrowth).
Loads the training metrics from paper_experiment_outputs and creates a comparison summary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    print("=== RigL-style 2:4 vs DST-EDL Evidential Growth Comparison ===")
    
    output_dir = Path("/kaggle/working/paper_experiment_outputs/isic")
    if not output_dir.exists():
        output_dir = REPO_ROOT / "paper_experiment_outputs" / "isic"
        
    guds_metrics_path = output_dir / "full_guds" / "seed_42" / "metrics.json"
    rigl_metrics_path = output_dir / "rigl_style_24" / "seed_42" / "metrics.json"
    
    if not guds_metrics_path.exists() or not rigl_metrics_path.exists():
        print("[WARNING] Could not find completed runs for full_guds and/or rigl_style_24.")
        print("Please train them first by running:")
        print("  python experiments/isic_paper_experiments.py --experiment full_guds --allow_dummy_data")
        print("  python experiments/isic_paper_experiments.py --experiment rigl_style_24 --allow_dummy_data")
        print("\nUsing mock metrics for demonstration purposes...")
        
        guds_metrics = {
            "metrics": {
                "image_auroc": 0.8845,
                "image_ap": 0.8123,
                "ece_default": 0.0543,
                "e_aurc": 0.0210
            },
            "history": {
                "train_loss": [0.85, 0.72, 0.61, 0.52, 0.45, 0.39, 0.35, 0.32, 0.30, 0.28],
                "val_auroc": [0.70, 0.75, 0.81, 0.83, 0.85, 0.86, 0.87, 0.88, 0.88, 0.88]
            }
        }
        rigl_metrics = {
            "metrics": {
                "image_auroc": 0.8412,
                "image_ap": 0.7431,
                "ece_default": 0.0912,
                "e_aurc": 0.0385
            },
            "history": {
                "train_loss": [0.92, 0.81, 0.73, 0.66, 0.60, 0.55, 0.51, 0.48, 0.46, 0.44],
                "val_auroc": [0.65, 0.68, 0.72, 0.76, 0.79, 0.81, 0.82, 0.83, 0.84, 0.84]
            }
        }
    else:
        guds_metrics = json.loads(guds_metrics_path.read_text(encoding="utf-8"))
        rigl_metrics = json.loads(rigl_metrics_path.read_text(encoding="utf-8"))
        print("Successfully loaded real training runs from disk.")
        
    g_met = guds_metrics["metrics"]
    r_met = rigl_metrics["metrics"]
    
    print("\n" + "="*70)
    print(f"{'Metric':<30} | {'RigL-style 2:4':<15} | {'DST-EDL (GUDS)':<15} | {'Gain':<10}")
    print("-"*70)
    
    for k in ["image_auroc", "image_ap", "ece_default", "e_aurc"]:
        g_val = g_met.get(k, 0.0)
        r_val = r_met.get(k, 0.0)
        
        if k in ["ece_default", "e_aurc"]:
            gain = r_val - g_val  # lower is better
            gain_str = f"+{gain:.4f} (Better)" if gain > 0 else f"{gain:.4f}"
        else:
            gain = g_val - r_val  # higher is better
            gain_str = f"+{gain:.4f} (Better)" if gain > 0 else f"{gain:.4f}"
            
        print(f"{k:<30} | {r_val:<15.4f} | {g_val:<15.4f} | {gain_str:<10}")
    print("="*70)
    
    # Analyze convergence
    g_history = guds_metrics.get("history", {})
    r_history = rigl_metrics.get("history", {})
    
    g_loss = g_history.get("train_loss", [])
    r_loss = r_history.get("train_loss", [])
    
    if g_loss and r_loss:
        print("\nConvergence Rate (Train Loss over epochs):")
        for epoch in range(min(len(g_loss), len(r_loss))):
            print(f"  Epoch {epoch+1:02d}: RigL Loss = {r_loss[epoch]:.4f} | GUDS Loss = {g_loss[epoch]:.4f}")


if __name__ == "__main__":
    main()
