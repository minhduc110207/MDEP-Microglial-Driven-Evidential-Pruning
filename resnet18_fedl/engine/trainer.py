"""
============================================================================
  Trainer — Training loop for F-EDL.

  Much simpler than the MDEP trainer because:
    • No sparsity engine (no masks, no scores, no multi-agent updates)
    • No amortised gradient computation
    • Backbone is frozen → only head gradients
============================================================================
"""

import time
import torch

import config as cfg
from models.fedl_head import ResNetFEDL


class FEDLTrainer:
    """
    Training engine for the F-EDL framework.

    Handles:
        • AMP (Automatic Mixed Precision) for speed on GPU
        • Gradient clipping
        • LR scheduling
        • Progress logging
    """

    def __init__(self, model, optimizer, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler("cuda")

    # ──────────────────────────────────────────────────────────────
    def train_epoch(self, epoch, dataloader, device, print_interval=None):
        """
        Run one training epoch.

        Returns:
            Exponential-moving-average of the batch losses.
        """
        print_interval = print_interval or cfg.PRINT_INTERVAL
        self.model.train()

        # Ensure backbone stays frozen (safety guard)
        if hasattr(self.model, "freeze_backbone") and self.model.freeze_backbone:
            self.model.backbone.eval()

        ema_loss = None
        num_batches = len(dataloader)
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs  = inputs.to(device)
            targets = targets.to(device)

            self.optimizer.zero_grad()

            # Forward (AMP)
            with torch.amp.autocast("cuda"):
                alpha, pi, tau = self.model(inputs)

            # Loss in FP32 to avoid digamma/log underflow
            with torch.amp.autocast("cuda", enabled=False):
                loss = self.criterion(
                    alpha.float(), pi.float(), tau.float(), targets, epoch,
                )

            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            trainable_params = [
                p for p in self.model.parameters() if p.requires_grad
            ]
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=cfg.GRAD_CLIP_NORM)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA loss tracking
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = 0.95 * ema_loss + 0.05 * loss.item()

            # Progress print
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - epoch_start
                avg_time = elapsed / (batch_idx + 1)
                eta = avg_time * (num_batches - batch_idx - 1)
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"    Batch [{batch_idx+1:>5}/{num_batches}]  "
                    f"| Loss: {ema_loss:.4f}  "
                    f"| LR: {lr:.2e}  "
                    f"| Elapsed: {elapsed/60:.1f}m  "
                    f"| ETA: {eta/60:.1f}m",
                    flush=True,
                )

        # Step scheduler at epoch end
        if self.scheduler is not None:
            self.scheduler.step()

        return ema_loss if ema_loss is not None else 0.0
