"""
============================================================================
  F-EDL — Flexible Evidential Deep Learning with Frozen ResNet-18 Backbone
  
  Entry point.  Run:
      python main.py
  
  Structure:
      config.py           → all hyperparameters
      models/backbone.py  → frozen ResNet-18 feature extractor
      models/fedl_head.py → FD evidence head + uncertainty computation
      losses/fedl_loss.py → F-EDL loss (Expected MSE + Brier + dispersion)
      data/isic_dataset.py→ ISIC 2024 dataset + dataloaders
      engine/trainer.py   → training loop
      engine/evaluator.py → metrics, plots, uncertainty analysis
      utils/checkpoint.py → save / load checkpoints
============================================================================
"""

import os
import sys
import time

import torch
import torch.optim as optim

# Ensure the project root is on the path (for Kaggle single-cell execution)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from models import ResNetFEDL
from losses import FEDLLoss
from data import get_isic_dataloaders
from engine import FEDLTrainer, evaluate
from utils import save_checkpoint, load_checkpoint


def main():
    device = cfg.DEVICE
    print(f"🖥  Device: {device}")

    # ── 1. Data ────────────────────────────────────────────────────
    train_loader, test_loader, num_classes, class_weights = get_isic_dataloaders()
    print(f"📊 Classes: {num_classes}")
    print(f"   Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")

    # ── 2. Model ───────────────────────────────────────────────────
    model = ResNetFEDL(
        num_classes=num_classes,
        backbone_name=cfg.BACKBONE,
        pretrained=cfg.PRETRAINED,
        freeze_backbone=cfg.FREEZE_BACKBONE,
        alpha_offset=cfg.ALPHA_OFFSET,
        tau_offset=cfg.TAU_OFFSET,
        head_init_std=cfg.HEAD_INIT_STD,
    ).to(device)

    # Report parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Total params:     {total_params:,}")
    print(f"📐 Trainable params: {trainable_params:,}  ({trainable_params/total_params*100:.1f}%)")

    # ── 3. Loss ────────────────────────────────────────────────────
    criterion = FEDLLoss(
        num_classes=num_classes,
        reg_lambda=cfg.REG_LAMBDA,
        disp_reg_weight=cfg.DISP_REG_WEIGHT,
        annealing_epochs=cfg.ANNEALING_EPOCHS,
        class_weights=class_weights.to(device),
    )

    # ── 4. Optimiser + Scheduler ───────────────────────────────────
    optimizer = optim.Adam(
        model.trainable_parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    scheduler = None
    if cfg.SCHEDULER == "cosine":
        t_max = cfg.COSINE_T_MAX or cfg.TOTAL_EPOCHS
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif cfg.SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # ── 5. Trainer ─────────────────────────────────────────────────
    trainer = FEDLTrainer(model, optimizer, criterion, scheduler)

    # ── 6. Resume from checkpoint ──────────────────────────────────
    start_epoch, best_loss = load_checkpoint(
        model, optimizer, scaler=trainer.scaler, device=device,
    )

    # ── 7. WandB (optional) ────────────────────────────────────────
    has_wandb = False
    try:
        import wandb
        if os.environ.get("WANDB_API_KEY"):
            wandb.login()
        else:
            wandb.login(anonymous="allow")

        wandb.init(
            project=cfg.WANDB_PROJECT,
            name=cfg.WANDB_RUN_NAME,
            config={
                "learning_rate": cfg.LEARNING_RATE,
                "total_epochs": cfg.TOTAL_EPOCHS,
                "batch_size": cfg.BATCH_SIZE,
                "architecture": f"{cfg.BACKBONE}-F-EDL",
                "freeze_backbone": cfg.FREEZE_BACKBONE,
                "reg_lambda": cfg.REG_LAMBDA,
            },
        )
        has_wandb = True
    except Exception as e:
        print(f"⚠️ WandB skipped: {e}. Running without online logging.")

    # ── 8. Training loop ───────────────────────────────────────────
    print("\n🚀 Starting Training (F-EDL Framework)")
    print("=" * 60)
    start_time = time.time()

    for epoch in range(start_epoch, cfg.TOTAL_EPOCHS):
        # Kaggle time-out safety
        if time.time() - start_time > cfg.KAGGLE_TIME_LIMIT:
            print("⏳ Approaching Kaggle time limit. Stopping early.")
            break

        loss = trainer.train_epoch(epoch, train_loader, device)
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch [{epoch+1:>2}/{cfg.TOTAL_EPOCHS}]  "
            f"| Phase: F-EDL Training  "
            f"| LR: {lr:.2e}  "
            f"| Loss: {loss:.4f}"
        )

        # Save checkpoint
        is_best = loss < best_loss
        if is_best:
            best_loss = loss

        save_checkpoint(
            model, optimizer, epoch, loss, best_loss,
            scaler=trainer.scaler, is_best=is_best,
        )

        if has_wandb:
            try:
                wandb.log({
                    "epoch": epoch + 1,
                    "loss": loss,
                    "lr": lr,
                    "best_loss": best_loss,
                })
            except Exception as e:
                print(f"⚠️ WandB log failed: {e}")

    print("=" * 60)
    print("✅ Training complete.\n")

    # ── 9. Load best weights for evaluation ────────────────────────
    if os.path.exists(cfg.CHECKPOINT_BEST):
        print(f"Loading best checkpoint for final evaluation...")
        try:
            best_ckpt = torch.load(cfg.CHECKPOINT_BEST, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
        except Exception as e:
            print(f"⚠️ Error loading best checkpoint: {e}. Evaluating with final weights.")

    # ── 10. Evaluation ─────────────────────────────────────────────
    eval_metrics = evaluate(model, test_loader, device, num_classes)

    if has_wandb:
        try:
            wandb.log(eval_metrics)
            wandb.finish()
        except Exception as e:
            print(f"⚠️ WandB final log failed: {e}")

    # ── 11. Save final model ───────────────────────────────────────
    torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
    print("=" * 60)
    print(f"💾 Model weights saved to: {cfg.MODEL_SAVE_PATH}")
    print("   (Download from 'Output' tab on Kaggle)")


if __name__ == "__main__":
    main()
