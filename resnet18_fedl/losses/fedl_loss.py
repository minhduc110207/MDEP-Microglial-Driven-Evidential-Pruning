"""
============================================================================
  F-EDL Loss — Flexible Evidential Deep Learning objective.

  Components:
    1. Expected MSE under the Flexible Dirichlet (data-fit)
    2. Brier-score regularisation (calibration)
    3. Dispersion regularisation (prevent τ collapse)

  Reference: "Flexible Evidential Deep Learning" (NeurIPS / OpenReview)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FEDLLoss(nn.Module):
    """
    Loss = E_FD[MSE]  +  λ · anneal(t) · Brier  −  μ · log(τ)

    Args:
        num_classes:      Number of target classes (K).
        reg_lambda:       Weight for Brier-score regularisation term.
        disp_reg_weight:  Weight for dispersion (τ) regularisation.
        annealing_epochs: Epochs over which to linearly ramp reg from 0→1.
        class_weights:    Optional (K,) tensor — higher for rare classes.
    """

    def __init__(
        self,
        num_classes: int,
        reg_lambda: float = 0.1,
        disp_reg_weight: float = 0.01,
        annealing_epochs: int = 10,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_lambda = reg_lambda
        self.disp_reg_weight = disp_reg_weight
        self.annealing_epochs = annealing_epochs

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    # ──────────────────────────────────────────────────────────────
    def forward(self, alpha, pi, tau, targets, epoch=None):
        """
        Args:
            alpha:   (B, K)  concentration parameters.
            pi:      (B, K)  allocation probabilities.
            tau:     (B, K)  dispersion parameters.
            targets: (B,) or (B, K)  class labels / one-hot.
            epoch:   Current epoch for annealing (optional).

        Returns:
            Scalar loss.
        """
        if targets.dim() == 1:
            targets = F.one_hot(targets, self.num_classes).float()

        S = alpha.sum(dim=1, keepdim=True)   # Dirichlet strength
        p_hat = alpha / S                    # expected probability (base Dir)

        # ── 1. Expected MSE under Flexible Dirichlet ──
        # E[(y - p)^2] = (y - p_hat)^2 + Var(p)
        err = (targets - p_hat) ** 2
        var = p_hat * (1.0 - p_hat) / (S + 1.0)
        # Weight by allocation π  (FD generalisation)
        loss_mse = torch.sum(pi * (err + var), dim=1, keepdim=True)

        # ── 2. Brier-score regularisation ──
        brier = torch.sum((p_hat - targets) ** 2, dim=1, keepdim=True)

        # ── 3. Dispersion regularisation (prevent τ → 0 collapse) ──
        disp_reg = -torch.mean(torch.log(tau + 1e-8), dim=1, keepdim=True)

        # ── Annealing coefficient ──
        if epoch is not None and self.annealing_epochs > 0:
            anneal = min(1.0, epoch / self.annealing_epochs)
        else:
            anneal = 1.0

        # ── Per-sample class weight ──
        if self.class_weights is not None:
            sample_w = torch.sum(
                targets * self.class_weights.unsqueeze(0), dim=1, keepdim=True
            )
        else:
            sample_w = 1.0

        # ── Total loss ──
        loss = sample_w * (
            loss_mse
            + self.reg_lambda * anneal * brier
            + self.disp_reg_weight * anneal * disp_reg
        )
        return loss.mean()
