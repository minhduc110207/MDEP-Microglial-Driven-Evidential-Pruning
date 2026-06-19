"""
============================================================================
  Checkpoint — Save / load training state.

  Supports:
    • Model weights
    • Optimizer state
    • AMP scaler state
    • Epoch + best loss tracking
    • Auto-search in Kaggle input datasets
============================================================================
"""

import os
import torch

import config as cfg


def save_checkpoint(
    model, optimizer, epoch, loss, best_loss,
    path=None, scaler=None, is_best=False,
):
    """
    Save a training checkpoint.

    Args:
        model:      The model (will save state_dict).
        optimizer:  The optimizer (will save state_dict).
        epoch:      Current epoch number.
        loss:       Current loss value.
        best_loss:  Best loss seen so far.
        path:       File path to save to (default: CHECKPOINT_LATEST from config).
        scaler:     Optional AMP GradScaler.
        is_best:    If True, also save to CHECKPOINT_BEST path.
    """
    path = path or cfg.CHECKPOINT_LATEST

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "loss": loss,
        "best_loss": best_loss,
    }
    torch.save(state, path)

    if is_best:
        best_path = cfg.CHECKPOINT_BEST
        state["best_loss"] = best_loss
        torch.save(state, best_path)
        print(f"⭐ New best loss: {best_loss:.4f}. Saved best checkpoint.")


def load_checkpoint(model, optimizer=None, scaler=None, device=None):
    """
    Load a training checkpoint if one exists.

    Search order:
        1. CHECKPOINT_LATEST in OUTPUT_DIR
        2. Kaggle input directories (latest_checkpoint.pth, then best_checkpoint.pth)

    Returns:
        (start_epoch, best_loss) — epoch to resume from and best loss seen.
        (0, inf) if no checkpoint found.
    """
    device = device or cfg.DEVICE
    resume_path = cfg.CHECKPOINT_LATEST

    # Search Kaggle input dirs if local checkpoint not found
    if not os.path.exists(resume_path) and os.path.isdir("/kaggle/input"):
        found = None
        for root, dirs, files in os.walk("/kaggle/input"):
            if "latest_checkpoint.pth" in files:
                found = os.path.join(root, "latest_checkpoint.pth")
                break
            elif "best_checkpoint.pth" in files and not found:
                found = os.path.join(root, "best_checkpoint.pth")
        if found:
            resume_path = found
            print(f"🔍 Found previous checkpoint in dataset: {resume_path}")

    if not os.path.exists(resume_path):
        return 0, float("inf")

    print(f"🔄 Loading checkpoint from {resume_path} ...")
    try:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if scaler is not None and ckpt.get("scaler_state_dict") is not None:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"⏩ Resuming from epoch {start_epoch + 1} (Best Loss: {best_loss:.4f})")
        return start_epoch, best_loss

    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}. Starting from scratch.")
        return 0, float("inf")
