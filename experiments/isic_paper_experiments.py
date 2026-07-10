"""
Train/evaluate ISIC 2024 experiments referenced by the paper manuscript.

This runner is designed for Kaggle after the repo has been copied to
/kaggle/working. It reuses the dataset split, calibration, and metrics from
guds_edl_core.py, then adds paper-facing baseline variants that can be trained
from the same command-line surface.

Examples:

    python experiments/isic_paper_experiments.py --experiment full_guds
    python experiments/isic_paper_experiments.py --suite main_tables
    python experiments/isic_paper_experiments.py --suite all

Outputs:

    /kaggle/working/paper_experiment_outputs/isic/<experiment_name>/seed_<seed>/
        run_config.json
        metrics.json
        test_predictions.csv
        model_state.pth
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from multi_gpu_utils import TransparentDataParallel
except ImportError:
    class TransparentDataParallel(nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def __setattr__(self, name, value):
            if name in ["module", "device_ids", "output_device", "dim", "_is_replica"]:
                super().__setattr__(name, value)
            elif hasattr(self, "module") and hasattr(self.module, name):
                setattr(self.module, name, value)
            else:
                super().__setattr__(name, value)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "isic_fair_v3_nvidia24_2026_07_09"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (  # noqa: E402
    AdaptiveThresholdDecisionSupport,
    EvidenceLayer,
    EvidentialFocalLoss,
    MDEPTrainer,
    MDEPConv2d,
    MDEPLinear,
    configure_training_runtime,
    compute_adaptive_ece,
    compute_aurc,
    compute_class_conditional_ece,
    compute_ece,
    compute_isic_pauc,
    compute_patient_level_se_top15,
    compute_uncertainties,
    evaluate,
    evaluate_adaptive_modes,
    generate_2_4_mask,
    get_imbalanced_dataloaders,
    dataloader_runtime_kwargs,
    make_grad_scaler,
    move_batch_to_device,
    print_sparsity_report,
    replace_conv2d_with_mdep,
)
from experiments.metrics_ext import (  # noqa: E402
    binary_extended_metrics,
    collect_evidential_outputs,
    uncertainty_separation_metrics,
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    description: str
    sparse: bool = False
    static_sparse: bool = False
    use_mdep_trainer: bool = False
    loss_name: str = "edl"
    pruner_type: str = "signed_first_order"
    regrower_type: str = "kl_uniform"
    pruning_strength: float = 0.5
    disable_pruner: bool = False
    disable_regrower: bool = False
    kl_scaling: str = "symmetric"
    disable_efl: bool = False
    disable_anticryst: bool = False
    disable_topology_cache: bool = False
    calibration_mode: str = "bias_temperature"
    classifier_retrain: bool = False
    label_aware_smoothing: bool = False


EXPERIMENTS: dict[str, ExperimentSpec] = {
    # Main result rows in the paper's ISIC comparison tables.
    "standard_ce": ExperimentSpec(
        name="standard_ce",
        family="long_tailed_baseline",
        description="Dense ResNet-18 trained with standard cross-entropy.",
        loss_name="ce",
    ),
    "focal_loss": ExperimentSpec(
        name="focal_loss",
        family="long_tailed_baseline",
        description="Dense ResNet-18 trained with focal loss.",
        loss_name="focal",
    ),
    "class_balanced_ce": ExperimentSpec(
        name="class_balanced_ce",
        family="long_tailed_baseline",
        description="Class-Balanced Loss baseline using effective-number reweighting.",
        loss_name="class_balanced_ce",
    ),
    "balanced_softmax": ExperimentSpec(
        name="balanced_softmax",
        family="long_tailed_baseline",
        description="Balanced Softmax baseline using train-prior logits inside the CE objective.",
        loss_name="balanced_softmax",
    ),
    "ldam_drw": ExperimentSpec(
        name="ldam_drw",
        family="long_tailed_baseline",
        description="LDAM with deferred effective-number reweighting.",
        loss_name="ldam_drw",
    ),
    "decoupled_crt": ExperimentSpec(
        name="decoupled_crt",
        family="long_tailed_baseline",
        description="cRT-style baseline: dense CE representation learning followed by classifier retraining.",
        loss_name="ce",
        classifier_retrain=True,
    ),
    "mislas": ExperimentSpec(
        name="mislas",
        family="long_tailed_sota_baseline",
        description="MiSLAS-style baseline: decoupled classifier retraining with label-aware smoothing.",
        loss_name="ce",
        classifier_retrain=True,
        label_aware_smoothing=True,
    ),
    "dense_edl": ExperimentSpec(
        name="dense_edl",
        family="evidential_baseline",
        description="Dense EDL baseline with symmetric KL.",
        loss_name="edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "fisher_edl": ExperimentSpec(
        name="fisher_edl",
        family="evidential_baseline",
        description="Dense EDL with an additional Fisher-information penalty.",
        loss_name="fisher_edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "flexible_edl": ExperimentSpec(
        name="flexible_edl",
        family="evidential_baseline",
        description="Dense EDL with a learnable positive evidence scale.",
        loss_name="edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "r_edl": ExperimentSpec(
        name="r_edl",
        family="evidential_baseline",
        description="Relaxed EDL proxy: reduced KL pressure and no focal modulation.",
        loss_name="r_edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "static_24_edl": ExperimentSpec(
        name="static_24_edl",
        family="dynamic_sparse_baseline",
        description="Static 2:4 sparse EDL with fixed magnitude-derived masks.",
        sparse=True,
        static_sparse=True,
        loss_name="edl",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "rigl_style_24": ExperimentSpec(
        name="rigl_style_24",
        family="dynamic_sparse_baseline",
        description="RigL-style 2:4 baseline using magnitude pruning and task-gradient regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="magnitude",
        regrower_type="gradient",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "full_guds": ExperimentSpec(
        name="full_guds",
        family="proposed",
        description="Full GUDS-EDL with signed pruner, KL-uniform regrower, EFL, and symmetric KL.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
    ),
    # Appendix C ablations.
    "guds_without_pruner": ExperimentSpec(
        name="guds_without_pruner",
        family="ablation",
        description="GUDS-EDL without uncertainty-guided pruning.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
        disable_pruner=True,
    ),
    "guds_without_regrower": ExperimentSpec(
        name="guds_without_regrower",
        family="ablation",
        description="GUDS-EDL without evidence-seeking regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
        disable_regrower=True,
    ),
    "guds_asymmetric_kl": ExperimentSpec(
        name="guds_asymmetric_kl",
        family="ablation",
        description="GUDS-EDL with asymmetric KL (diagnostic) instead of symmetric KL.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="asymmetric",
    ),
    "guds_without_efl": ExperimentSpec(
        name="guds_without_efl",
        family="ablation",
        description="GUDS-EDL without Evidential Focal Loss modulation.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "guds_without_anticryst": ExperimentSpec(
        name="guds_without_anticryst",
        family="ablation",
        description="GUDS-EDL without anti-crystallization noise.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
        disable_anticryst=True,
    ),
    "guds_absolute_pruner": ExperimentSpec(
        name="guds_absolute_pruner",
        family="ablation",
        description="GUDS-EDL with absolute-gradient pruning instead of signed pruning.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="absolute_grad",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
    ),
    "guds_class_conditioned_regrower": ExperimentSpec(
        name="guds_class_conditioned_regrower",
        family="ablation",
        description="GUDS-EDL with class-conditioned regrowth (diagnostic) instead of KL-uniform regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="class_conditioned",
        kl_scaling="symmetric",
    ),
    "guds_without_topology_cache": ExperimentSpec(
        name="guds_without_topology_cache",
        family="ablation",
        description="GUDS-EDL without amortized topology caching; structural signals are recomputed per batch.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
        disable_topology_cache=True,
    ),
    "guds_temperature_only": ExperimentSpec(
        name="guds_temperature_only",
        family="ablation",
        description="GUDS-EDL calibrated with scalar temperature only, without bias correction.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
        calibration_mode="temperature_only",
    ),
    "guds_no_posthoc_calibration": ExperimentSpec(
        name="guds_no_posthoc_calibration",
        family="ablation",
        description="GUDS-EDL evaluated without post-hoc temperature or bias calibration.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="kl_uniform",
        kl_scaling="symmetric",
        calibration_mode="none",
    ),
}


SUITES: dict[str, list[str]] = {
    "main_tables": [
        "full_guds",
        "dense_edl",
        "fisher_edl",
        "flexible_edl",
        "r_edl",
        "static_24_edl",
        "rigl_style_24",
    ],
    "baselines": [
        "standard_ce",
        "focal_loss",
        "class_balanced_ce",
        "balanced_softmax",
        "ldam_drw",
        "decoupled_crt",
        "mislas",
        "dense_edl",
        "fisher_edl",
        "flexible_edl",
        "r_edl",
        "static_24_edl",
        "rigl_style_24",
    ],
    "ablations": [
        "full_guds",
        "guds_without_pruner",
        "guds_without_regrower",
        "guds_asymmetric_kl",
        "guds_without_efl",
        "guds_without_anticryst",
        "guds_absolute_pruner",
        "guds_class_conditioned_regrower",
        "guds_without_topology_cache",
        "guds_temperature_only",
        "guds_no_posthoc_calibration",
    ],
}
SUITES["all"] = list(dict.fromkeys(["full_guds"] + SUITES["baselines"] + SUITES["ablations"]))


DISCRIMINATIVE_LOSS_NAMES = {"ce", "focal", "class_balanced_ce", "balanced_softmax", "ldam_drw"}


def uses_softmax_evaluation(spec: ExperimentSpec) -> bool:
    return spec.loss_name in DISCRIMINATIVE_LOSS_NAMES or spec.classifier_retrain


class FlexibleEvidenceLayer(EvidenceLayer):
    """Softplus evidence with a learnable positive logit scale."""

    def __init__(self, max_evidence: float = 20.0):
        super().__init__(activation="softplus", max_evidence=max_evidence)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x * torch.exp(self.log_scale))


class ResNetEvidenceModel(nn.Module):
    """ResNet-18 backbone with a paper-compatible `fc=[linear,evidence]` head."""

    def __init__(self, num_classes: int, flexible: bool = False, pretrained: bool = True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        try:
            self.backbone = models.resnet18(weights=weights)
        except Exception as exc:
            print(f"[WARN] Could not load pretrained ResNet-18 weights ({exc}); using random init.")
            self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        evidence = FlexibleEvidenceLayer() if flexible else EvidenceLayer(activation="softplus")
        self.fc = nn.Sequential(nn.Linear(in_features, num_classes), evidence)
        nn.init.normal_(self.fc[0].weight, mean=0.0, std=0.001)
        nn.init.constant_(self.fc[0].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))


class OODProjectionHead(nn.Module):
    """Small detached-feature domain head for low-impact Outlier Exposure."""

    def __init__(self, in_features: int, projection_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        self.domain_classifier = nn.Linear(projection_dim, 1)

    def forward(self, features: torch.Tensor, return_projection: bool = False):
        projection = F.normalize(self.projector(features), dim=1)
        domain_logits = self.domain_classifier(projection).squeeze(1)
        if return_projection:
            return projection, domain_logits
        return domain_logits


def attach_ood_projection_head(model: nn.Module, projection_dim: int = 128) -> nn.Module:
    in_features = int(model.fc[0].in_features)
    if not hasattr(model, "ood_projection_head"):
        model.ood_projection_head = OODProjectionHead(in_features, projection_dim=projection_dim)
    return model


class FisherEDLLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, fisher_lambda: float = 1e-3):
        super().__init__()
        self.base_loss = base_loss
        self.fisher_lambda = fisher_lambda

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base = self.base_loss(evidence, targets, epoch=epoch)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=1, keepdim=True)
        fisher = torch.mean(torch.clamp(torch.polygamma(1, alpha) - torch.polygamma(1, strength), min=0.0))
        return base + self.fisher_lambda * fisher


class RelaxedEDLLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: torch.Tensor | None, total_epochs: int):
        super().__init__()
        self.loss = EvidentialFocalLoss(
            gamma=0.0,
            num_classes=num_classes,
            kl_lambda=0.01,
            class_weights=class_weights,
            warmup_epochs=max(1, int(0.30 * total_epochs)),
            total_epochs=total_epochs,
            disable_efl=True,
            kl_scaling="symmetric",
        )

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        return self.loss(evidence, targets, epoch=epoch)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def output_root() -> Path:
    root = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT
    return root / "paper_experiment_outputs" / "isic"


def set_static_sparse_mode(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            module.warmup = False
            module.gamma = 0.15
            module.static_24_baseline = True
            with torch.no_grad():
                module.mask.copy_(generate_2_4_mask(
                    module.scores, getattr(module, "mask_layout", "nvidia_kcrs")
                ))


def resolve_protocol_profile(args: argparse.Namespace) -> dict[str, object]:
    """Freeze performance-relevant semantics under a named protocol profile.

    The legacy profile is a reproducibility path for the manuscript's v2
    results, not a TensorRT sparse-convolution profile.  The default v3 path
    is the only one suitable for NVIDIA-layout structural or hardware claims.
    """
    if args.protocol_profile == "legacy_v2":
        return {
            "mask_layout": "legacy_flattened",
            "loader_profile": "legacy_v2",
            "eval_batch_size": args.batch_size,
            "hardware_compatible": False,
        }
    return {
        "mask_layout": "nvidia_kcrs",
        "loader_profile": "nvidia_v3",
        "eval_batch_size": 128 if torch.cuda.is_available() else args.batch_size,
        "hardware_compatible": True,
    }


def model_head(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    head = model.fc if hasattr(model, "fc") else model.head
    return head[0], head[1]


def prior_logit_delta(
    p_true: list[float],
    p_train: list[float],
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    values = [math.log(p_true[c] + 1e-8) - math.log(p_train[c] + 1e-8) for c in range(num_classes)]
    return torch.tensor(values, dtype=dtype, device=device)


@torch.no_grad()
def collect_logits_labels(model: nn.Module, loader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    linear, _ = model_head(model)
    logits_list = []
    labels_list = []
    for inputs, targets in loader:
        inputs, targets = move_batch_to_device(inputs, targets, device)
        if hasattr(model, "backbone"):
            features = model.backbone(inputs)
        else:
            original_head = model.fc if hasattr(model, "fc") else model.head
            if hasattr(model, "fc"):
                model.fc = nn.Identity()
                features = model(inputs)
                model.fc = original_head
            else:
                model.head = nn.Identity()
                features = model(inputs)
                model.head = original_head
        logits_list.append(linear(features).detach())
        labels_list.append(targets)
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def optimize_thresholds(
    model: nn.Module,
    val_loader,
    device: torch.device,
    temperature: float,
    bias: torch.Tensor | None,
    prior_delta: torch.Tensor | None = None,
) -> dict[str, float]:
    linear, evidence_layer = model_head(model)
    logits, labels = collect_logits_labels(model, val_loader, device)
    with torch.no_grad():
        if prior_delta is not None:
            logits = logits + prior_delta.to(device=logits.device, dtype=logits.dtype)
        scaled_logits = logits / temperature
        if bias is not None:
            scaled_logits = scaled_logits + bias
        evidence = evidence_layer(scaled_logits)
        unc = compute_uncertainties(evidence)
        probs = (unc["alpha"] / unc["S"]).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    best_t_bal_acc = 0.5
    best_bal_acc = 0.0
    best_t_clinical = 0.5
    best_spec_at_sens80 = 0.0
    found_sens80 = False
    p_min = float(probs[:, 1].min())
    p_max = float(probs[:, 1].max())
    percentiles = np.linspace(0, 100, 199)
    search_space = np.unique(np.percentile(probs[:, 1], percentiles))
        
    for threshold in search_space:
        y_pred = (probs[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        bal_acc = 0.5 * (sens + spec)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_t_bal_acc = float(threshold)
        if sens >= 0.80 and (spec > best_spec_at_sens80 or not found_sens80):
            best_spec_at_sens80 = spec
            best_t_clinical = float(threshold)
            found_sens80 = True

    return {
        "rule_out": best_t_clinical,
        "high_recall": best_t_clinical,
        "double_read": best_t_clinical,
        "balanced": best_t_bal_acc,
        "rule_in": 0.5,
    }


def run_calibration(
    model: nn.Module,
    cal_loader,
    val_loader,
    device: torch.device,
    mode: str,
    p_true: list[float],
    p_train: list[float],
) -> tuple[float, torch.Tensor | None, dict[str, float]]:
    linear, evidence_layer = model_head(model)
    logits, labels = collect_logits_labels(model, cal_loader, device)
    prior_delta = prior_logit_delta(
        p_true,
        p_train,
        linear.out_features,
        device=logits.device,
        dtype=logits.dtype,
    )
    if mode == "none":
        prior_delta = torch.zeros_like(prior_delta)
    logits_for_calibration = logits + prior_delta

    def evidential_nll(scaled_logits: torch.Tensor) -> torch.Tensor:
        evidence = evidence_layer(scaled_logits)
        unc = compute_uncertainties(evidence)
        probs = (unc["alpha"] / unc["S"]).clamp_min(1e-8)
        return F.nll_loss(torch.log(probs), labels)

    if mode == "none":
        temperature = 1.0
        bias = None
    elif mode == "temperature_only":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        optimizer = optim.LBFGS([temp_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
            loss = evidential_nll(logits_for_calibration / temp_param.clamp_min(0.1))
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = None
    elif mode == "bias_temperature":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        bias_param = nn.Parameter(torch.zeros(linear.out_features, device=device))
        optimizer = optim.LBFGS([temp_param, bias_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
            loss = evidential_nll(logits_for_calibration / temp_param.clamp_min(0.1) + bias_param)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = bias_param.detach()
    else:
        raise ValueError(f"Unknown calibration_mode: {mode}")

    thresholds = optimize_thresholds(model, val_loader, device, temperature, bias, prior_delta=prior_delta)
    print(
        f"[CAL] mode={mode} | T={temperature:.4f} | "
        f"prior_delta={prior_delta.detach().cpu().numpy()} | "
        f"bias={None if bias is None else bias.detach().cpu().numpy()} | thresholds={thresholds}"
    )
    return temperature, bias, thresholds


@torch.no_grad()
def collect_softmax_outputs(
    model: nn.Module,
    loader,
    device: torch.device,
    temperature: float,
    bias: torch.Tensor | None,
) -> dict[str, np.ndarray]:
    logits, labels = collect_logits_labels(model, loader, device)
    scaled_logits = logits / temperature
    if bias is not None:
        scaled_logits = scaled_logits + bias.to(device=scaled_logits.device, dtype=scaled_logits.dtype)
    probs = F.softmax(scaled_logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy().astype(int)
    return {
        "y_true": y_true,
        "probs": probs,
        "y_pred": probs.argmax(axis=1),
        "confidences": probs.max(axis=1),
    }


def thresholds_from_probabilities(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    best_t_bal_acc = 0.5
    best_bal_acc = 0.0
    best_t_clinical = 0.5
    best_spec_at_sens80 = 0.0
    found_sens80 = False
    p_min = float(probs[:, 1].min())
    p_max = float(probs[:, 1].max())
    percentiles = np.linspace(0, 100, 199)
    search_space = np.unique(np.percentile(probs[:, 1], percentiles))
        
    for threshold in search_space:
        y_pred = (probs[:, 1] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        bal_acc = 0.5 * (sens + spec)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_t_bal_acc = float(threshold)
        if sens >= 0.80 and (spec > best_spec_at_sens80 or not found_sens80):
            best_spec_at_sens80 = spec
            best_t_clinical = float(threshold)
            found_sens80 = True
    return {
        "rule_out": best_t_clinical,
        "high_recall": best_t_clinical,
        "double_read": best_t_clinical,
        "balanced": best_t_bal_acc,
        "rule_in": 0.5,
    }


def run_softmax_calibration(
    model: nn.Module,
    cal_loader,
    val_loader,
    device: torch.device,
    mode: str,
    p_true: list[float],
    p_train: list[float],
) -> tuple[float, torch.Tensor | None, dict[str, float], torch.Tensor]:
    logits, labels = collect_logits_labels(model, cal_loader, device)
    prior_delta = prior_logit_delta(p_true, p_train, logits.shape[1], device=logits.device, dtype=logits.dtype)
    if mode == "none":
        prior_delta = torch.zeros_like(prior_delta)
    logits_for_calibration = logits + prior_delta

    if mode == "none":
        temperature = 1.0
        bias = None
    elif mode == "temperature_only":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        optimizer = optim.LBFGS([temp_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_for_calibration / temp_param.clamp_min(0.1), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = None
    elif mode == "bias_temperature":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        bias_param = nn.Parameter(torch.zeros(logits.shape[1], device=device))
        optimizer = optim.LBFGS([temp_param, bias_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_for_calibration / temp_param.clamp_min(0.1) + bias_param, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = bias_param.detach()
    else:
        raise ValueError(f"Unknown calibration_mode: {mode}")

    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
    val_outputs = collect_softmax_outputs(model, val_loader, device, temperature, eval_bias)
    thresholds = thresholds_from_probabilities(val_outputs["y_true"], val_outputs["probs"])
    print(f"[CAL] mode={mode} | evaluator=softmax | T={temperature:.4f} | thresholds={thresholds}")
    return temperature, bias, thresholds, prior_delta


def test_frame_from_loader(test_loader) -> pd.DataFrame:
    if isinstance(test_loader.dataset, torch.utils.data.Subset) and hasattr(test_loader.dataset.dataset, "data_frame"):
        return test_loader.dataset.dataset.data_frame.iloc[test_loader.dataset.indices]
    if hasattr(test_loader.dataset, "data_frame"):
        return test_loader.dataset.data_frame
    return pd.DataFrame({
        "target": [test_loader.dataset[i][1].item() for i in range(len(test_loader.dataset))],
        "patient_id": [f"patient_{i // 5}" for i in range(len(test_loader.dataset))],
    })


def evaluate_softmax_baseline(
    model: nn.Module,
    test_loader,
    device: torch.device,
    temperature: float,
    eval_bias: torch.Tensor,
    thresholds: dict[str, float],
    deployment_prevalence: float,
) -> dict[str, float]:
    outputs = collect_softmax_outputs(model, test_loader, device, temperature, eval_bias)
    y_true = outputs["y_true"]
    probs = outputs["probs"]
    y_pred = (probs[:, 1] >= 0.5).astype(int)
    confidences = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    ece_adaptive, _, _, _ = compute_adaptive_ece(confidences, correct)
    ece_eq_width, _, _, _ = compute_ece(confidences, correct)
    class_eces = compute_class_conditional_ece(probs, y_true)
    threshold_report = {
        "balanced": float(thresholds.get("balanced", 0.5)),
        "high_recall": float(thresholds.get("high_recall", thresholds.get("rule_out", 0.5))),
    }
    metrics = {
        "pauc": compute_isic_pauc(y_true, probs[:, 1], min_tpr=0.80),
        "se_top15": compute_patient_level_se_top15(test_frame_from_loader(test_loader), probs),
        "pr_auc": average_precision_score(y_true, probs[:, 1]),
        "macro_auroc": roc_auc_score(y_true, probs[:, 1]),
        "aurc": compute_aurc(y_true, y_pred, confidences),
        "ece_adaptive": float(ece_adaptive),
        "ece_eq_width": float(ece_eq_width),
        "class_ece_0": float(class_eces[0]),
        "class_ece_1": float(class_eces[1]),
        "balanced_accuracy_default": float(balanced_accuracy_score(y_true, y_pred)),
    }
    metrics.update(binary_extended_metrics(
        y_true,
        probs,
        thresholds=threshold_report,
        deployment_prevalence=deployment_prevalence,
    ))
    return metrics


def save_test_predictions(
    run_dir: Path,
    spec: ExperimentSpec,
    seed: int,
    split_seed: int,
    test_loader,
    outputs: dict[str, np.ndarray],
    thresholds: dict[str, float],
    experiment_name: str | None = None,
) -> Path:
    """Write calibrated held-out test predictions for bootstrap CIs."""
    probs = np.asarray(outputs["probs"])
    y_true = np.asarray(outputs["y_true"]).astype(int)
    n = len(y_true)
    frame = test_frame_from_loader(test_loader).reset_index(drop=True)
    if len(frame) != n:
        frame = pd.DataFrame(index=np.arange(n))

    pred = pd.DataFrame({
        "row_id": np.arange(n, dtype=int),
        "experiment": experiment_name or spec.name,
        "seed": int(seed),
        "split_seed": int(split_seed),
        "y_true": y_true,
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "y_pred_argmax": np.asarray(outputs["y_pred"]).astype(int),
        "confidence": np.asarray(outputs["confidences"], dtype=float),
    })
    if "isic_id" in frame.columns:
        pred.insert(0, "sample_id", frame["isic_id"].astype(str).to_numpy())
    else:
        pred.insert(0, "sample_id", pred["row_id"].astype(str))
    if "patient_id" in frame.columns:
        pred.insert(1, "patient_id", frame["patient_id"].astype(str).to_numpy())
    else:
        pred.insert(1, "patient_id", [f"patient_{i // 5}" for i in range(n)])

    balanced_t = float(thresholds.get("balanced", 0.5))
    high_recall_t = float(thresholds.get("high_recall", thresholds.get("rule_out", 0.5)))
    pred["threshold_balanced"] = balanced_t
    pred["threshold_high_recall"] = high_recall_t
    pred["y_pred_balanced"] = (pred["prob_1"].to_numpy() >= balanced_t).astype(int)
    pred["y_pred_high_recall"] = (pred["prob_1"].to_numpy() >= high_recall_t).astype(int)
    if "u_e" in outputs:
        pred["u_e"] = np.asarray(outputs["u_e"], dtype=float)
    if "u_a" in outputs:
        pred["u_a"] = np.asarray(outputs["u_a"], dtype=float)

    path = run_dir / "test_predictions.csv"
    pred.to_csv(path, index=False)
    print(f"[DONE] Saved held-out test predictions to {path}")
    return path


@torch.no_grad()
def quality_gate_report(decision_support: AdaptiveThresholdDecisionSupport, test_loader, device: torch.device) -> dict[str, float]:
    targets_all = []
    decisions_all = []
    ua_all = []
    for inputs, targets in test_loader:
        inputs, _ = move_batch_to_device(inputs, targets, device)
        final_decision, _, _, u_a = decision_support(inputs, mode="balanced", quality_gated=True)
        targets_all.append(targets.numpy())
        decisions_all.append(final_decision.cpu().numpy())
        ua_all.append(u_a.squeeze(-1).cpu().numpy())

    y_true = np.concatenate(targets_all)
    decisions = np.concatenate(decisions_all)
    u_a = np.concatenate(ua_all)
    accepted = decisions != 3
    discarded = decisions == 3
    report = {
        "quality_gate_accepted_coverage": float(accepted.mean()) if len(accepted) else 0.0,
        "quality_gate_discard_rate": float(discarded.mean()) if len(discarded) else 0.0,
        "quality_gate_mean_ua_accepted": float(u_a[accepted].mean()) if accepted.any() else 0.0,
        "quality_gate_mean_ua_discarded": float(u_a[discarded].mean()) if discarded.any() else 0.0,
    }
    if accepted.any():
        valid_pred = decisions[accepted]
        valid_true = y_true[accepted]
        tn, fp, fn, tp = confusion_matrix(valid_true, valid_pred, labels=[0, 1]).ravel()
        report.update({
            "quality_gate_sensitivity": float(tp / (tp + fn + 1e-8)),
            "quality_gate_specificity": float(tn / (tn + fp + 1e-8)),
            "quality_gate_error_rate": float((valid_pred != valid_true).mean()),
        })
    return report


def make_loss(
    spec: ExperimentSpec,
    num_classes: int,
    class_weights: torch.Tensor,
    total_epochs: int,
    device: torch.device,
    efl_gamma_final: float = 0.0,
) -> nn.Module:
    if spec.loss_name == "r_edl":
        return RelaxedEDLLoss(num_classes, class_weights.to(device), total_epochs)

    base = EvidentialFocalLoss(
        gamma=5.0,
        num_classes=num_classes,
        kl_lambda=0.1,
        class_weights=class_weights.to(device),
        warmup_epochs=max(1, int(0.30 * total_epochs)),
        total_epochs=total_epochs,
        disable_efl=spec.disable_efl,
        kl_scaling=spec.kl_scaling,
        gamma_final=efl_gamma_final,
    )
    if spec.loss_name == "fisher_edl":
        return FisherEDLLoss(base)
    return base


def logits_from_model(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model.fc[0](model.backbone(inputs))


def effective_number_weights(p_train: list[float], total_samples: int = 10000, beta: float = 0.9999, device: torch.device | None = None) -> torch.Tensor:
    """Class-Balanced Loss weights from absolute training frequencies."""
    counts = torch.tensor(p_train, dtype=torch.float32, device=device) * total_samples
    counts = counts.clamp_min(1.0)
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, device=counts.device), counts).clamp_max(1.0 - 1e-8))
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights


def ldam_margins(p_train: list[float], total_samples: int = 10000, max_margin: float = 0.5, device: torch.device | None = None) -> torch.Tensor:
    counts = torch.tensor(p_train, dtype=torch.float32, device=device) * total_samples
    counts = counts.clamp_min(1.0)
    margins = 1.0 / torch.sqrt(torch.sqrt(counts.clamp_min(1.0)))
    margins = margins * (max_margin / margins.max().clamp_min(1e-8))
    return margins


def _extract_dataset_labels(dataset) -> list[int]:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_labels = _extract_dataset_labels(dataset.dataset)
        return [base_labels[int(idx)] for idx in dataset.indices]
    if hasattr(dataset, "data_frame") and "target" in dataset.data_frame:
        return [int(v) for v in dataset.data_frame["target"].tolist()]
    if hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        return [int(v) for v in dataset.tensors[1].detach().cpu().tolist()]
    labels = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        labels.append(int(item[1]))
    return labels


def make_class_balanced_loader(train_loader) -> DataLoader:
    labels = _extract_dataset_labels(train_loader.dataset)
    counts = np.bincount(labels)
    sample_weights = [1.0 / max(counts[label], 1) for label in labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        **dataloader_runtime_kwargs(
            num_workers=getattr(train_loader, "num_workers", 0),
            pin_memory=getattr(train_loader, "pin_memory", False),
        ),
        drop_last=False,
    )


def label_aware_smoothing_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    p_train: list[float],
    head_smoothing: float = 0.10,
    tail_smoothing: float = 0.00,
) -> torch.Tensor:
    """Label-aware smoothing used by the MiSLAS-style classifier retraining baseline."""
    num_classes = logits.shape[1]
    freqs = torch.tensor(p_train, dtype=logits.dtype, device=logits.device)
    if freqs.numel() != num_classes:
        freqs = torch.ones(num_classes, dtype=logits.dtype, device=logits.device) / num_classes
    denom = (freqs.max() - freqs.min()).clamp_min(1e-8)
    class_smoothing = tail_smoothing + (head_smoothing - tail_smoothing) * ((freqs - freqs.min()) / denom)
    eps = class_smoothing[targets].view(-1, 1)
    off_value = eps / max(num_classes - 1, 1)
    target_dist = torch.full_like(logits, 0.0)
    target_dist.scatter_(1, targets.view(-1, 1), 1.0)
    smooth_targets = target_dist * (1.0 - eps) + (1.0 - target_dist) * off_value
    log_probs = F.log_softmax(logits, dim=1)
    return torch.mean(torch.sum(-smooth_targets * log_probs, dim=1))


def ce_or_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    p_true: list[float],
    p_train: list[float],
    epoch: int,
    total_epochs: int,
    total_samples: int = 10000,
) -> torch.Tensor:
    adjusted = logits

    if spec.loss_name == "ce":
        return F.cross_entropy(adjusted, targets)

    if spec.loss_name == "class_balanced_ce":
        cb_weights = effective_number_weights(p_train, total_samples=total_samples, device=logits.device)
        return F.cross_entropy(adjusted, targets, weight=cb_weights)

    if spec.loss_name == "balanced_softmax":
        log_prior = torch.tensor(
            [math.log(max(p, 1e-8)) for p in p_train],
            dtype=logits.dtype,
            device=logits.device,
        )
        return F.cross_entropy(adjusted + log_prior, targets)

    if spec.loss_name == "ldam_drw":
        margins = ldam_margins(p_train, total_samples=total_samples, device=logits.device)
        one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(logits.dtype)
        logits_m = adjusted - one_hot * margins.unsqueeze(0)
        weight = None
        if epoch >= int(0.75 * total_epochs):
            weight = effective_number_weights(p_train, total_samples=total_samples, device=logits.device)
        return F.cross_entropy(logits_m, targets, weight=weight)

    probs = F.softmax(adjusted, dim=1)
    pt = probs.gather(1, targets.view(-1, 1)).clamp_min(1e-8)
    ce = F.cross_entropy(adjusted, targets, reduction="none").view(-1, 1)
    return torch.mean(((1.0 - pt) ** 2.0) * ce)


def retrain_classifier_crt(
    model: nn.Module,
    train_loader,
    device: torch.device,
    spec: ExperimentSpec,
    p_train: list[float],
    total_epochs: int,
    lr: float,
    validation_callback=None,
) -> list[dict[str, float]]:
    """Classifier re-training baseline inspired by cRT/decoupled classifiers."""
    nn.init.normal_(model.fc[0].weight, mean=0.0, std=0.001)
    nn.init.constant_(model.fc[0].bias, 0.0)
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.fc[0].parameters():
        param.requires_grad = True

    epochs = max(10, int(0.10 * total_epochs))
    optimizer = optim.AdamW(model.fc[0].parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history: list[dict[str, float]] = []
    crt_loader = make_class_balanced_loader(train_loader)
    for epoch in range(epochs):
        model.train()
        losses = []
        start = time.time()
        for inputs, targets in crt_loader:
            inputs, targets = move_batch_to_device(inputs, targets, device)
            optimizer.zero_grad(set_to_none=True)
            logits = logits_from_model(model, inputs)
            if spec.label_aware_smoothing:
                loss = label_aware_smoothing_loss(logits, targets, p_train)
            else:
                loss = F.cross_entropy(logits, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.fc[0].parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        avg_loss = float(np.mean(losses)) if losses else 0.0
        scheduler.step()
        elapsed = time.time() - start
        row = {"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed, "stage": "crt"}
        if validation_callback is not None:
            row.update(validation_callback(epoch + 1, stage="crt") or {})
        history.append(row)
        print(f"cRT [{epoch + 1:>2}/{epochs}] | loss={avg_loss:.4f} | {elapsed:.1f}s")

    for param in model.backbone.parameters():
        param.requires_grad = True
    return history


def train_standard(
    model: nn.Module,
    train_loader,
    device: torch.device,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    p_true: list[float],
    p_train: list[float],
    total_epochs: int,
    lr: float,
    log_every: int = 5,
    validation_callback=None,
) -> list[dict[str, float]]:
    params = [p for name, p in model.named_parameters() if "scores" not in name]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    scaler = make_grad_scaler(enabled=device.type == "cuda")
    history: list[dict[str, float]] = []

    if spec.static_sparse:
        set_static_sparse_mode(model)

    criterion = None
    if spec.loss_name not in DISCRIMINATIVE_LOSS_NAMES:
        criterion = make_loss(spec, model.fc[0].out_features, class_weights, total_epochs, device)

    total_samples = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 10000
    num_batches = max(1, len(train_loader))
    base_lr = float(lr)

    for epoch in range(total_epochs):
        model.train()
        if spec.static_sparse:
            set_static_sparse_mode(model)
        losses = []
        start = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if epoch == 0:
                warmup_fraction = batch_idx / num_batches
                current_lr = 1e-6 + (base_lr - 1e-6) * warmup_fraction
                current_loss_scale = 4.0 - 3.0 * warmup_fraction
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            else:
                current_loss_scale = 1.0

            inputs, targets = move_batch_to_device(inputs, targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                if spec.loss_name in DISCRIMINATIVE_LOSS_NAMES:
                    logits = logits_from_model(model, inputs)
                else:
                    evidence = model(inputs)

            # All objectives use FP32; AMP is restricted to the forward pass.
            with torch.amp.autocast('cuda', enabled=False):
                if spec.loss_name in DISCRIMINATIVE_LOSS_NAMES:
                    loss = ce_or_focal_loss(
                        logits.float(),
                        targets,
                        spec,
                        class_weights,
                        p_true,
                        p_train,
                        epoch,
                        total_epochs,
                        total_samples=total_samples,
                    )
                else:
                    loss = criterion(evidence.float(), targets, epoch=epoch)

            scaler.scale(loss * current_loss_scale).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        avg_loss = float(np.mean(losses)) if losses else 0.0
        elapsed = time.time() - start
        row = {"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed}
        if validation_callback is not None:
            row.update(validation_callback(epoch + 1, stage="train") or {})
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(log_every, 1) == 0 or (epoch + 1) == total_epochs:
            print(f"[TRAIN] epoch={epoch + 1:03d}/{total_epochs:03d} loss={avg_loss:.4f} time={elapsed:.1f}s")
    if spec.classifier_retrain:
        history.extend(
            retrain_classifier_crt(
                model,
                train_loader,
                device,
                spec,
                p_train,
                total_epochs,
                lr,
                validation_callback=validation_callback,
            )
        )
    return history


def train_guds(
    model: nn.Module,
    train_loader,
    device: torch.device,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    total_epochs: int,
    lr: float,
    log_every: int = 5,
    verbose_structural_logs: bool = False,
    structural_proxy_batches: int = 4,
    structural_proxy_min_classes: int = 2,
    efl_gamma_final: float = 0.0,
    validation_callback=None,
    ood_loader=None,
    lambda_ood=0.05,
    ood_start_epoch=None,
    ood_loss_target="direct",
) -> list[dict[str, float]]:
    warmup_epochs = max(1, int(0.30 * total_epochs))
    criterion = make_loss(
        spec,
        model.fc[0].out_features,
        class_weights,
        total_epochs,
        device,
        efl_gamma_final=efl_gamma_final,
    )
    params = [p for name, p in model.named_parameters() if "scores" not in name]
    # Sparse coordinates receive explicit active-only AdamW-style decay in
    # MDEPTrainer; setting optimizer decay to zero avoids decaying dormant links.
    optimizer = optim.AdamW(params, lr=lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    trainer_args = SimpleNamespace(
        disable_pruner=spec.disable_pruner,
        disable_regrower=spec.disable_regrower,
        pruner_type=spec.pruner_type,
        regrower_type=spec.regrower_type,
        kl_scaling=spec.kl_scaling,
        disable_efl=spec.disable_efl,
        disable_anticryst=spec.disable_anticryst,
        use_anticryst=not spec.disable_anticryst,
        disable_topology_cache=spec.disable_topology_cache,
        pruning_strength=spec.pruning_strength,
        verbose_structural_logs=verbose_structural_logs,
        structural_proxy_batches=structural_proxy_batches,
        structural_proxy_min_classes=structural_proxy_min_classes,
    )
    trainer = MDEPTrainer(model, optimizer, criterion, total_epochs, warmup_epochs, args=trainer_args, scheduler=scheduler)
    history = []
    for epoch in range(total_epochs):
        loss = trainer.train_epoch(
            epoch,
            train_loader,
            device,
            print_interval=200,
            ood_loader=ood_loader,
            lambda_ood=lambda_ood,
            ood_start_epoch=ood_start_epoch,
            ood_loss_target=ood_loss_target,
        )
        topo_gamma = float(trainer.step_gamma(epoch))
        efl_gamma = float(getattr(criterion, "gamma", 0.0))
        row = {
            "epoch": epoch + 1,
            "objective_loss": float(loss),
            "topology_gamma": topo_gamma,
            "efl_gamma": efl_gamma,
        }
        if validation_callback is not None:
            row.update(validation_callback(epoch + 1, stage="train") or {})
        history.append(row)
        if epoch == 0 or (epoch + 1) % max(log_every, 1) == 0 or (epoch + 1) == total_epochs:
            print(
                f"[TRAIN] epoch={epoch + 1:03d}/{total_epochs:03d} "
                f"objective_loss={loss:.4f} topo_gamma={topo_gamma:.4f} efl_gamma={efl_gamma:.4f}"
            )
    return history


def print_metrics_table(spec_name: str, metrics: dict[str, float]):
    print(f"\n{'='*70}")
    print(f"🏥 CLINICAL EVALUATION (ISIC) | {spec_name}")
    print(f"{'='*70}")
    print(f"{'Metric':<40} | {'Value':>10}")
    print(f"{'-'*40}-+-{'-'*10}")
    
    groups = {
        "Ranking & Detection": ["macro_auroc", "pr_auc", "pauc"],
        "Clinical Balance": ["balanced_accuracy_default", "sensitivity_at_balanced", "specificity_at_balanced"],
        "High Recall Operating Point": ["sensitivity_at_high_recall", "specificity_at_high_recall"],
        "Uncertainty & Calibration": ["ece_adaptive", "aurc", "nll", "brier"],
        "Sparsity": ["active_density", "valid_24_fraction", "masked_throughput_relative"]
    }
    
    printed_keys = set()
    for group_name, keys in groups.items():
        printed_any = False
        for k in keys:
            if k in metrics:
                if not printed_any:
                    print(f" {group_name.upper()}")
                    printed_any = True
                display_name = "  " + k.replace("_", " ").title()
                if k == "pauc":
                    display_name = "  pAUC (TPR > 0.8) 🌟"
                print(f"{display_name:<40} | {metrics[k]:>10.4f}")
                printed_keys.add(k)
        if printed_any:
            print(f"{'-'*40}-+-{'-'*10}")
            
    remaining = [k for k in sorted(metrics.keys()) if k not in printed_keys and isinstance(metrics[k], (int, float))]
    if remaining:
        print(" OTHER METRICS")
        for k in remaining:
            display_name = "  " + k.replace("_", " ").title()
            print(f"{display_name:<40} | {metrics[k]:>10.4f}")
        print(f"{'-'*40}-+-{'-'*10}")
            
    print(f"{'='*70}\n")


def make_ranking_checkpoint_callback(
    model: nn.Module,
    val_loader,
    device: torch.device,
    args: argparse.Namespace,
    spec: ExperimentSpec,
):
    """Track a validation-only ranking checkpoint without touching calibration/test data."""
    tracker = {
        "enabled": args.checkpoint_selection != "last",
        "metric": args.checkpoint_selection,
        "best_epoch": None,
        "best_score": -float("inf"),
        "best_pauc": None,
        "best_ap": None,
        "state_dict": None,
        "evaluations": [],
    }
    if not tracker["enabled"]:
        return None, tracker
    if spec.classifier_retrain:
        print("[CHECKPOINT] Disabled for classifier-retraining baselines because cRT has a separate epoch schedule.")
        tracker["enabled"] = False
        return None, tracker

    @torch.no_grad()
    def callback(epoch: int, stage: str = "train"):
        if stage != "train":
            return {}
        if epoch < args.checkpoint_start_epoch:
            return {}
        if epoch % args.checkpoint_eval_every != 0 and epoch != args.epochs:
            return {}
        model.eval()
        labels_all, scores_all = [], []
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            logits = logits_from_model(model, inputs)
            evidence = model_head(model)[1](logits)
            unc = compute_uncertainties(evidence)
            probability = (unc["alpha"] / unc["S"])[:, 1]
            labels_all.append(labels.detach().cpu().numpy())
            scores_all.append(probability.detach().cpu().numpy())
        y_true = np.concatenate(labels_all)
        y_score = np.concatenate(scores_all)
        if len(np.unique(y_true)) < 2:
            print("[CHECKPOINT] Validation split has one class; retaining final epoch.")
            tracker["enabled"] = False
            return {}
        pauc = float(compute_isic_pauc(y_true, y_score, min_tpr=0.80))
        ap = float(average_precision_score(y_true, y_score))
        if args.checkpoint_selection == "pauc":
            score = pauc
        elif args.checkpoint_selection == "ap":
            score = ap
        else:
            # pAUC remains primary; AP only resolves near-ties.
            score = pauc + 1e-3 * ap
        tracker["evaluations"].append({
            "epoch": int(epoch),
            "pauc": pauc,
            "average_precision": ap,
            "selection_score": score,
        })
        if score > tracker["best_score"]:
            tracker["best_score"] = score
            tracker["best_epoch"] = int(epoch)
            tracker["best_pauc"] = pauc
            tracker["best_ap"] = ap
            tracker["state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            print(
                f"[CHECKPOINT] new best epoch={epoch} metric={args.checkpoint_selection} "
                f"pAUC={pauc:.4f} AP={ap:.4f}"
            )
        return {"val_pauc": pauc, "val_average_precision": ap}

    return callback, tracker


def run_one(spec: ExperimentSpec, args: argparse.Namespace, seed: int) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    profile = resolve_protocol_profile(args)
    run_experiment_name = f"{spec.name}{args.run_suffix}" if args.run_suffix else spec.name
    run_dir = output_root() / run_experiment_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n[RUN] dataset=isic experiment={run_experiment_name} family={spec.family} "
        f"evaluator={'softmax' if uses_softmax_evaluation(spec) else 'evidential'} "
        f"seed={seed} split_seed={args.split_seed} epochs={args.epochs} profile={args.protocol_profile} "
        f"device={device} output={run_dir}"
    )

    loaders = get_imbalanced_dataloaders(
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        subsample_ratio=args.subsample_ratio,
        subsample_scope=args.subsample_scope,
        seed=args.split_seed,
        allow_dummy_data=args.allow_dummy_data,
        loader_profile=str(profile["loader_profile"]),
        eval_batch_size=int(profile["eval_batch_size"]),
    )
    train_loader, val_loader, cal_loader, test_loader, num_classes, class_weights, p_true, p_train = loaders

    model = ResNetEvidenceModel(
        num_classes=num_classes,
        flexible=(spec.name == "flexible_edl"),
        pretrained=not args.no_pretrained,
    )
    if spec.sparse:
        replace_conv2d_with_mdep(
            model.backbone,
            learn_permutation=False,
            mask_layout=str(profile["mask_layout"]),
        )
        print(
            "[INFO] GUDS-EDL sparse mode: 2:4 backbone convs only; "
            f"layout={profile['mask_layout']}; classifier head is unmasked by design; "
            "identity channel order is frozen."
        )
    model = model.to(device)
    if False:  # Forced to single GPU execution
        if spec.sparse:
            print(
                f"[INFO] Detected {torch.cuda.device_count()} GPUs; running sparse GUDS-EDL on single GPU "
                "so cached masks and effective-weight structural gradients stay on the original modules."
            )
        else:
            print(f"[INFO] Using {torch.cuda.device_count()} GPUs via DataParallel.")
            model = TransparentDataParallel(model)

    ood_loader = None
    if args.outlier_exposure:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, Subset
        
        transform_ood = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if args.outlier_exposure.lower() == "auto":
            possible_paths = [
                "/kaggle/input/datasets/mahdavi1202/skin-cancer",
                "/kaggle/input/skin-cancer",
            ]
            detected_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    detected_path = p
                    break
            if detected_path:
                args.outlier_exposure = detected_path
                print(f"[INFO] Auto-detected skin-cancer dataset for Outlier Exposure at: {detected_path}")
            else:
                args.outlier_exposure = "cifar10"
                print("[INFO] Skin-cancer dataset not found. Falling back to CIFAR-10 for Outlier Exposure.")

        if args.outlier_exposure.lower() == "cifar10":
            # 1. First check if a local python pickle dataset exists (e.g., pankrzysiu/cifar10-python)
            possible_pickle_paths = [
                "/kaggle/input/datasets/pankrzysiu/cifar10-python",
                "/kaggle/input/cifar10-python",
            ]
            local_pickle_path = None
            for p in possible_pickle_paths:
                if os.path.exists(p):
                    local_pickle_path = p
                    break
            
            loaded_pickle = False
            if local_pickle_path:
                print(f"\n[INFO] Auto-detected local python CIFAR-10 pickle dataset at: {local_pickle_path}")
                try:
                    os.makedirs("./data", exist_ok=True)
                    target_link = os.path.abspath("./data/cifar-10-batches-py")
                    if os.path.exists(target_link) or os.path.islink(target_link):
                        if os.path.islink(target_link):
                            os.unlink(target_link)
                        else:
                            shutil.rmtree(target_link)
                    
                    sub_dir = os.path.join(local_pickle_path, "cifar-10-batches-py")
                    src_dir = sub_dir if os.path.exists(sub_dir) else local_pickle_path
                    
                    os.symlink(src_dir, target_link)
                    print(f"[INFO] Created symlink for torchvision: {src_dir} -> {target_link}")
                    
                    ood_ds = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_ood)
                    ood_batch_size = max(1, int(args.batch_size * args.ood_batch_ratio))
                    ood_loader = DataLoader(ood_ds, batch_size=ood_batch_size, shuffle=True, num_workers=2, drop_last=True)
                    print(f"[INFO] Outlier Exposure active. OOD Batch size: {ood_batch_size}, Total OOD samples: {len(ood_ds)}")
                    loaded_pickle = True
                except Exception as e:
                    print(f"[WARNING] Failed to load local pickle dataset: {e}. Trying other methods.")
            
            # 2. If pickle is not loaded, check for local PNG dataset format (ImageFolder)
            if not loaded_pickle:
                local_paths = [
                    "/kaggle/input/cifar10-pngs-in-folders/cifar10/train",
                    "/kaggle/input/cifar10-pngs-in-folders/cifar10/cifar10/train",
                    "/kaggle/input/cifar10/train",
                    "/kaggle/input/cifar-10/train",
                ]
                local_cifar_path = None
                for p in local_paths:
                    if os.path.exists(p):
                        local_cifar_path = p
                        break
                
                if local_cifar_path:
                    print(f"\n[INFO] Auto-detected local PNG CIFAR-10 dataset at: {local_cifar_path}. Loading without download.")
                    try:
                        ood_ds = datasets.ImageFolder(root=local_cifar_path, transform=transform_ood)
                        ood_batch_size = max(1, int(args.batch_size * args.ood_batch_ratio))
                        ood_loader = DataLoader(ood_ds, batch_size=ood_batch_size, shuffle=True, num_workers=2, drop_last=True)
                        print(f"[INFO] Outlier Exposure active. OOD Batch size: {ood_batch_size}, Total OOD samples: {len(ood_ds)}")
                        loaded_pickle = True
                    except Exception as e:
                        print(f"[WARNING] Failed to load local PNG CIFAR-10: {e}. Falling back to default download.")
            
            # 3. Fallback to online PyTorch download
            if not loaded_pickle:
                print("\n[INFO] Loading CIFAR-10 for Outlier Exposure (OOD Regularization) via PyTorch download...")
                try:
                    ood_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_ood)
                    ood_batch_size = max(1, int(args.batch_size * args.ood_batch_ratio))
                    ood_loader = DataLoader(ood_ds, batch_size=ood_batch_size, shuffle=True, num_workers=2, drop_last=True)
                    print(f"[INFO] Outlier Exposure active. OOD Batch size: {ood_batch_size}, Total OOD samples: {len(ood_ds)}")
                except Exception as e:
                    print(f"[WARNING] Failed to load CIFAR-10: {e}. Outlier Exposure is disabled.")
                    ood_loader = None
        else:
            # Custom folder path passed
            print(f"\n[INFO] Loading custom folder for Outlier Exposure: {args.outlier_exposure}")
            try:
                base_ds = datasets.ImageFolder(root=args.outlier_exposure, transform=transform_ood)
                # Auto-partitioning: If it has classes like imgs_part_1, imgs_part_2, imgs_part_3,
                # we only use imgs_part_1 and imgs_part_2 for Outlier Exposure to prevent leakage with imgs_part_3
                allowed_classes = ["imgs_part_1", "imgs_part_2"]
                allowed_indices = [idx for name, idx in base_ds.class_to_idx.items() if name.lower() in allowed_classes]
                
                if allowed_indices:
                    indices = [i for i, (_, label) in enumerate(base_ds.samples) if label in allowed_indices]
                    ood_ds = Subset(base_ds, indices)
                    print(f"[INFO] Detected partitions. Using {[name for name, idx in base_ds.class_to_idx.items() if idx in allowed_indices]} for Outlier Exposure. Filtered samples: {len(ood_ds)} / {len(base_ds)}")
                else:
                    ood_ds = base_ds
                    print(f"[INFO] Using entire custom folder for Outlier Exposure. Total samples: {len(ood_ds)}")
                
                ood_batch_size = max(1, int(args.batch_size * args.ood_batch_ratio))
                ood_loader = DataLoader(ood_ds, batch_size=ood_batch_size, shuffle=True, num_workers=2, drop_last=True)
                print(f"[INFO] Outlier Exposure active. OOD Batch size: {ood_batch_size}")
            except Exception as e:
                print(f"[WARNING] Failed to load custom OE folder: {e}. Outlier Exposure is disabled.")
                ood_loader = None

    if ood_loader is not None and args.ood_loss_target == "projection":
        attach_ood_projection_head(model)
        model.ood_projection_head.to(device)
        print(
            "[INFO] Low-impact OE active: training detached OOD projection head only; "
            "backbone, classifier, and sparse topology are protected from OE gradients."
        )

    checkpoint_callback, checkpoint_tracker = make_ranking_checkpoint_callback(
        model,
        val_loader,
        device,
        args,
        spec,
    )
    if spec.use_mdep_trainer:
        history = train_guds(
            model,
            train_loader,
            device,
            spec,
            class_weights,
            args.epochs,
            args.lr,
            log_every=args.log_every,
            verbose_structural_logs=args.verbose_structural_logs,
            structural_proxy_batches=args.structural_proxy_batches,
            structural_proxy_min_classes=args.structural_proxy_min_classes,
            efl_gamma_final=args.efl_gamma_final,
            ood_loader=ood_loader,
            lambda_ood=args.lambda_ood,
            ood_start_epoch=args.ood_start_epoch,
            ood_loss_target=args.ood_loss_target,
            validation_callback=checkpoint_callback,
        )
    else:
        history = train_standard(
            model,
            train_loader,
            device,
            spec,
            class_weights,
            p_true,
            p_train,
            args.epochs,
            args.lr,
            log_every=args.log_every,
            validation_callback=checkpoint_callback,
        )

    if checkpoint_tracker["enabled"] and checkpoint_tracker["state_dict"] is not None:
        model.load_state_dict(checkpoint_tracker["state_dict"], strict=True)
        print(
            f"[CHECKPOINT] restored epoch={checkpoint_tracker['best_epoch']} "
            f"before calibration and held-out evaluation."
        )
    checkpoint_record = {
        key: value for key, value in checkpoint_tracker.items()
        if key != "state_dict"
    }

    quality_metrics = {}

    # Models that already compensate for class imbalance during training
    # (balanced_softmax, cRT, MiSLAS) produce prior-free
    # logits. Passing the skewed p_train to calibration would apply the
    # prior correction a second time (double-adjustment). We neutralise this
    # by telling the calibrator that training was uniform.
    prior_corrected = (
        spec.loss_name == "balanced_softmax"
        or spec.classifier_retrain
    )
    effective_p_train = p_train
    if prior_corrected:
        K = len(p_train)
        effective_p_train = [1.0 / K] * K

    if uses_softmax_evaluation(spec):
        temperature, bias, thresholds, prior_delta = run_softmax_calibration(
            model,
            cal_loader,
            val_loader,
            device,
            spec.calibration_mode,
            p_true,
            effective_p_train,
        )
        eval_bias = prior_delta / max(temperature, 1e-8)
        if bias is not None:
            eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
        metrics = evaluate_softmax_baseline(
            model,
            test_loader,
            device,
            temperature,
            eval_bias,
            thresholds,
            args.deployment_prevalence,
        )
        outputs = collect_softmax_outputs(model, test_loader, device, temperature, eval_bias)
    else:
        temperature, bias, thresholds = run_calibration(
            model,
            cal_loader,
            val_loader,
            device,
            spec.calibration_mode,
            p_true,
            p_train,
        )
        decision_support = AdaptiveThresholdDecisionSupport(
            model,
            is_resnet=True,
            thresholds=thresholds,
            temperature=temperature,
            bias=bias,
            true_class_prior=p_true,
            train_class_prior=p_train,
        )
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        evaluate_adaptive_modes(decision_support, test_loader, device)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        quality_metrics = quality_gate_report(decision_support, test_loader, device)
        decision_support.restore_model()

        prior_delta = prior_logit_delta(
            p_true,
            p_train,
            num_classes,
            device=device,
            dtype=torch.float32,
        )
        eval_bias = prior_delta / max(temperature, 1e-8)
        if bias is not None:
            eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)

        _, metrics = evaluate(
            model,
            val_loader,
            test_loader,
            device,
            num_classes=num_classes,
            temperature=temperature,
            bias=eval_bias,
            plot=False,
        )
        outputs = collect_evidential_outputs(model, test_loader, device, temperature=temperature, bias=eval_bias)
        threshold_report = {
            "balanced": float(thresholds.get("balanced", 0.5)),
            "high_recall": float(thresholds.get("high_recall", thresholds.get("rule_out", 0.5))),
        }
        metrics.update(binary_extended_metrics(
            outputs["y_true"],
            outputs["probs"],
            thresholds=threshold_report,
            deployment_prevalence=args.deployment_prevalence,
        ))
        metrics.update(uncertainty_separation_metrics(
            outputs["y_true"],
            outputs["y_pred"],
            outputs["u_e"],
            outputs["u_a"],
        ))
    if spec.sparse:
        print_sparsity_report(model)
    save_test_predictions(run_dir, spec, seed, args.split_seed, test_loader, outputs, thresholds, run_experiment_name)

    experiment_record = asdict(spec)
    experiment_record["name"] = run_experiment_name
    experiment_record["base_name"] = spec.name
    experiment_record["run_suffix"] = args.run_suffix

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": experiment_record,
        "seed": seed,
        "model_seed": seed,
        "split_seed": args.split_seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "outlier_exposure": args.outlier_exposure,
        "lambda_ood": args.lambda_ood,
        "ood_batch_ratio": args.ood_batch_ratio,
        "ood_start_epoch": args.ood_start_epoch,
        "ood_loss_target": args.ood_loss_target,
        "temperature": float(temperature),
        "bias": bias.tolist() if bias is not None else None,
        "calibration_bias": bias.tolist() if bias is not None else None,
        "evaluation_bias": eval_bias.tolist() if isinstance(eval_bias, torch.Tensor) else None,
        "prior_delta": prior_delta.tolist() if isinstance(prior_delta, torch.Tensor) else None,
        "thresholds": thresholds,
        "p_true": p_true,
        "p_train": p_train,
        "history": history,
        "checkpoint_selection": checkpoint_record,
        "metrics": metrics,
        "quality_gate": quality_metrics,
        "evaluator": "softmax" if uses_softmax_evaluation(spec) else "evidential",
        "runtime": {
            "deterministic": os.environ.get("MDEP_DETERMINISTIC", "1") != "0",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "protocol_profile": args.protocol_profile,
        "sparsity_layout": profile["mask_layout"],
        "loader_profile": profile["loader_profile"],
        "hardware_compatible_sparsity": profile["hardware_compatible"],
    }
    run_config = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment": experiment_record,
        "model_seed": seed,
        "split_seed": args.split_seed,
        "arguments": vars(args),
        "runtime": result["runtime"],
        "protocol_profile": args.protocol_profile,
        "sparsity_layout": profile["mask_layout"],
        "loader_profile": profile["loader_profile"],
        "hardware_compatible_sparsity": profile["hardware_compatible"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(json_safe(run_config), indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    if not args.no_save_model:
        torch.save(model.state_dict(), run_dir / "model_state.pth")
    print(f"[DONE] Saved metrics and model state to {run_dir}")
    print_metrics_table(spec.name, metrics)
    
    # Explicitly clear memory to prevent Kaggle RAM accumulation
    del model
    del train_loader, val_loader, cal_loader, test_loader
    if 'outputs' in locals():
        del outputs
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return result


def selected_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    if args.experiment:
        return [EXPERIMENTS[name] for name in args.experiment]
    names = SUITES[args.suite]
    return [EXPERIMENTS[name] for name in names]


def main() -> int:
    configure_training_runtime()
    parser = argparse.ArgumentParser(description="Run ISIC 2024 paper experiments.")
    parser.add_argument("--experiment", action="append", choices=sorted(EXPERIMENTS), help="Run one experiment; can be repeated.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="main_tables")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, help="Run one model seed when --seeds is not provided.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Run all selected experiments for these model seeds. Defaults to 42 123 456.")
    parser.add_argument("--split_seed", type=int, default=42, help="Fixed seed for patient splits and majority subsampling; keep constant across repeated model seeds.")
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--subsample_ratio", type=int, default=20)
    parser.add_argument("--subsample_scope", choices=["all", "train"], default="train", help="Apply ISIC majority-class subsampling before patient split or to train only.")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument("--allow_dummy_data", action="store_true", help="Permit synthetic dummy data for dry-runs only.")
    parser.add_argument("--deployment_prevalence", type=float, default=0.0015, help="Prevalence used for PPV/NPV/NNB reporting.")
    parser.add_argument("--log_every", type=int, default=5, help="Print training progress every N epochs.")
    parser.add_argument("--verbose_structural_logs", action="store_true", help="Print detailed per-layer structural update diagnostics.")
    parser.add_argument("--structural_proxy_batches", type=int, default=4, help="Maximum train mini-batches accumulated for one cached GUDS structural proxy batch.")
    parser.add_argument("--structural_proxy_min_classes", type=int, default=2, help="Minimum distinct target classes requested before caching a GUDS structural proxy batch.")
    parser.add_argument("--efl_gamma_final", type=float, default=0.0, help="Final EFL focal gamma after cosine decay; keep 0.0 for the reported clean default.")
    parser.add_argument(
        "--protocol_profile",
        choices=["nvidia_v3", "legacy_v2"],
        default="legacy_v2",
        help=(
            "legacy_v2 reproduces the fixed performance protocol used by the manuscript tables. "
            "nvidia_v3 uses TensorRT-compatible KCRS 2:4 masks and the memory-safe loader for a separate "
            "deployment diagnostic; legacy_v2 must not support TensorRT sparse-kernel claims."
        ),
    )
    parser.add_argument("--run_suffix", default="", help="Optional suffix for output experiment folders, useful for tuning runs without overwriting reported metrics.")
    parser.add_argument(
        "--checkpoint_selection",
        choices=["last", "pauc", "ap", "pauc_then_ap"],
        default="last",
        help="Validation-only checkpoint rule. 'last' preserves the reported baseline; other modes restore the best validation ranking epoch before calibration/test.",
    )
    parser.add_argument("--checkpoint_eval_every", type=int, default=5, help="Evaluate the validation ranking checkpoint every N training epochs.")
    parser.add_argument("--checkpoint_start_epoch", type=int, default=10, help="First epoch eligible for validation-only checkpoint selection.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--outlier_exposure", type=str, nargs="?", const="auto", default=None, help="Enable Outlier Exposure. Pass path to custom folder, or leave blank/'auto' to auto-detect skin-cancer folder.")
    parser.add_argument("--lambda_ood", type=float, default=0.003, help="OOD loss scaling weight when Outlier Exposure is enabled.")
    parser.add_argument("--ood_batch_ratio", type=float, default=0.125, help="OOD batch size ratio relative to ID batch size.")
    parser.add_argument("--ood_start_epoch", type=int, default=30, help="First zero-based epoch that receives Outlier Exposure loss.")
    parser.add_argument(
        "--ood_loss_target",
        choices=["projection", "direct"],
        default="projection",
        help="projection trains a detached OOD head only; direct applies OE to the main evidential model.",
    )
    args = parser.parse_args()

    output_root().mkdir(parents=True, exist_ok=True)
    all_results = []
    seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else [42, 123, 456])
    for spec in selected_experiments(args):
        for seed in seeds:
            all_results.append(run_one(spec, args, seed))

    summary_path = output_root() / "isic_summary.json"
    summary_path.write_text(json.dumps(json_safe(all_results), indent=2), encoding="utf-8")
    print(f"\nAll selected ISIC experiments completed. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
