"""
============================================================================
  F-EDL Head — Flexible Dirichlet Evidence Head + Uncertainty Computation.

  The head outputs three parameter groups for the Flexible Dirichlet (FD)
  distribution:
      α  (concentration)  — controls how peaked/flat each class belief is
      π  (allocation)     — how evidence mass is distributed across classes
      τ  (dispersion)     — spread/sharpness of the distribution

  Reference: "Flexible Evidential Deep Learning" (NeurIPS / OpenReview)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone


# ── Flexible Dirichlet Evidence Head ───────────────────────────────────

class FDEvidenceHead(nn.Module):
    """
    Predicts the three parameter groups of a Flexible Dirichlet distribution.

    Forward input:  feature vector (B, D)  from the backbone.
    Forward output: (alpha, pi, tau) — each of shape (B, K).
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        alpha_offset: float = 1.0,
        tau_offset: float = 1e-6,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha_offset = alpha_offset
        self.tau_offset = tau_offset

        # Three independent linear projections
        self.alpha_head = nn.Linear(in_features, num_classes)
        self.pi_head    = nn.Linear(in_features, num_classes)
        self.tau_head   = nn.Linear(in_features, num_classes)

        # Small init to prevent KL / evidence explosion at epoch 0
        for head in (self.alpha_head, self.pi_head, self.tau_head):
            nn.init.normal_(head.weight, mean=0, std=init_std)
            nn.init.constant_(head.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, D) feature vector.
        Returns:
            alpha: (B, K)  concentration,     α > alpha_offset
            pi:    (B, K)  allocation probs,   Σ_k π_k = 1
            tau:   (B, K)  dispersion,         τ > 0
        """
        alpha = F.softplus(self.alpha_head(x)) + self.alpha_offset
        pi    = F.softmax(self.pi_head(x), dim=-1)
        tau   = F.softplus(self.tau_head(x)) + self.tau_offset
        return alpha, pi, tau


# ── Full Model (Backbone + Head) ──────────────────────────────────────

class ResNetFEDL(nn.Module):
    """
    Complete model: frozen backbone  →  F-EDL evidence head.

    Example::

        model = ResNetFEDL(num_classes=2)
        alpha, pi, tau = model(images)          # images: (B, 3, 224, 224)
        unc = compute_uncertainties_fd(alpha, pi, tau)
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        alpha_offset: float = 1.0,
        tau_offset: float = 1e-6,
        head_init_std: float = 0.01,
    ):
        super().__init__()
        self.backbone, in_features = build_backbone(
            backbone_name, pretrained, freeze_backbone,
        )
        self.fd_head = FDEvidenceHead(
            in_features, num_classes,
            alpha_offset=alpha_offset,
            tau_offset=tau_offset,
            init_std=head_init_std,
        )
        self.freeze_backbone = freeze_backbone

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.fd_head(features)

    def trainable_parameters(self):
        """Return only the parameters that require gradients (the head)."""
        return [p for p in self.parameters() if p.requires_grad]


# ── Uncertainty Computation ────────────────────────────────────────────

def compute_uncertainties_fd(alpha, pi, tau):
    """
    Derive epistemic / aleatoric uncertainties from Flexible Dirichlet params.

    Args:
        alpha:  (B, K)  concentration parameters   (α > 1)
        pi:     (B, K)  allocation probabilities    (Σ π_k = 1)
        tau:    (B, K)  dispersion parameters       (τ > 0)

    Returns:
        dict with keys:
            p_hat       (B, K)  expected class probability
            epistemic   (B, 1)  epistemic uncertainty
            aleatoric   (B, 1)  aleatoric uncertainty
            alpha, pi, tau, S   original params + Dirichlet strength
    """
    S = alpha.sum(dim=1, keepdim=True)          # Dirichlet strength
    K = alpha.shape[1]
    p_base = alpha / S                          # base Dirichlet mean

    # Expected probability under FD: weighted by allocation π
    p_hat = pi * p_base

    # ── Epistemic Uncertainty ──
    # Generalisation of u_e = K/S, modulated by dispersion.
    # Higher τ → more dispersed → more uncertain.
    tau_mean = tau.mean(dim=1, keepdim=True)
    u_e = K / S * tau_mean / (tau_mean + 1.0)   # normalised to [0, K)

    # ── Aleatoric Uncertainty ──
    # Variance of the FD distribution:  Var(p_k) = π_k * α_k/S * (1 - α_k/S) / (S+1)
    var_per_class = pi * p_base * (1 - p_base) / (S + 1)
    u_a = var_per_class.sum(dim=1, keepdim=True)

    return {
        "p_hat":      p_hat,
        "epistemic":  u_e,
        "aleatoric":  u_a,
        "alpha":      alpha,
        "pi":         pi,
        "tau":        tau,
        "S":          S,
    }
