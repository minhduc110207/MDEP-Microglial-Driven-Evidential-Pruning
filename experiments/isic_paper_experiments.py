"""
Train/evaluate ISIC 2024 experiments referenced by main_text.tex.

This runner is designed for Kaggle after the repo has been copied to
/kaggle/working. It reuses the dataset split, calibration, and metrics from
guds_edl_core.py, then adds paper-facing baseline variants that can be trained
from the same command-line surface.

Examples:

    python experiments/isic_paper_experiments.py --experiment full_guds
    python experiments/isic_paper_experiments.py --suite main_tables
    python experiments/isic_paper_experiments.py --suite all

Outputs:

    /kaggle/working/paper_experiment_outputs/isic/<experiment_name>/
        run_config.json
        metrics.json
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
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import (  # noqa: E402
    AdaptiveThresholdDecisionSupport,
    EvidenceLayer,
    EvidentialFocalLoss,
    MDEPTrainer,
    MDEPConv2d,
    MDEPLinear,
    calibrate_temperature,
    compute_uncertainties,
    evaluate,
    evaluate_adaptive_modes,
    get_imbalanced_dataloaders,
    print_sparsity_report,
    replace_conv2d_with_mdep,
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
    regrower_type: str = "class_conditioned"
    disable_pruner: bool = False
    disable_regrower: bool = False
    kl_scaling: str = "asymmetric"
    disable_efl: bool = False
    disable_anticryst: bool = False
    logit_adjustment_train: bool = False


EXPERIMENTS: dict[str, ExperimentSpec] = {
    # Main result rows in main_text.tex Tables 1--2.
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
    "logit_adjustment": ExperimentSpec(
        name="logit_adjustment",
        family="long_tailed_baseline",
        description="Dense ResNet-18 trained with logit-adjusted cross-entropy.",
        loss_name="ce",
        logit_adjustment_train=True,
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
        description="RigL-style 2:4 proxy using absolute-gradient pruning and gradient regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="absolute_grad",
        regrower_type="gradient",
        kl_scaling="symmetric",
        disable_efl=True,
    ),
    "full_guds": ExperimentSpec(
        name="full_guds",
        family="proposed",
        description="Full GUDS-EDL with signed pruner, class-conditioned regrower, EFL, and asymmetric KL.",
        sparse=True,
        use_mdep_trainer=True,
        loss_name="edl",
        pruner_type="signed_first_order",
        regrower_type="class_conditioned",
        kl_scaling="asymmetric",
    ),
    # Appendix C ablations.
    "guds_without_pruner": ExperimentSpec(
        name="guds_without_pruner",
        family="ablation",
        description="GUDS-EDL without uncertainty-guided pruning.",
        sparse=True,
        use_mdep_trainer=True,
        disable_pruner=True,
    ),
    "guds_without_regrower": ExperimentSpec(
        name="guds_without_regrower",
        family="ablation",
        description="GUDS-EDL without evidence-seeking regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        disable_regrower=True,
    ),
    "guds_symmetric_kl": ExperimentSpec(
        name="guds_symmetric_kl",
        family="ablation",
        description="GUDS-EDL with symmetric KL instead of asymmetric KL.",
        sparse=True,
        use_mdep_trainer=True,
        kl_scaling="symmetric",
    ),
    "guds_without_efl": ExperimentSpec(
        name="guds_without_efl",
        family="ablation",
        description="GUDS-EDL without Evidential Focal Loss modulation.",
        sparse=True,
        use_mdep_trainer=True,
        disable_efl=True,
    ),
    "guds_without_anticryst": ExperimentSpec(
        name="guds_without_anticryst",
        family="ablation",
        description="GUDS-EDL without anti-crystallization noise.",
        sparse=True,
        use_mdep_trainer=True,
        disable_anticryst=True,
    ),
    "guds_absolute_pruner": ExperimentSpec(
        name="guds_absolute_pruner",
        family="ablation",
        description="GUDS-EDL with absolute-gradient pruning instead of signed pruning.",
        sparse=True,
        use_mdep_trainer=True,
        pruner_type="absolute_grad",
    ),
    "guds_kl_uniform_regrower": ExperimentSpec(
        name="guds_kl_uniform_regrower",
        family="ablation",
        description="GUDS-EDL with KL-to-uniform regrowth instead of class-conditioned regrowth.",
        sparse=True,
        use_mdep_trainer=True,
        regrower_type="kl_uniform",
    ),
}


SUITES: dict[str, list[str]] = {
    "main_tables": [
        "fisher_edl",
        "flexible_edl",
        "r_edl",
        "full_guds",
    ],
    "baselines": [
        "standard_ce",
        "focal_loss",
        "logit_adjustment",
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
        "guds_symmetric_kl",
        "guds_without_efl",
        "guds_without_anticryst",
        "guds_absolute_pruner",
        "guds_kl_uniform_regrower",
    ],
}
SUITES["all"] = list(dict.fromkeys(SUITES["baselines"] + SUITES["ablations"]))


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


def make_loss(spec: ExperimentSpec, num_classes: int, class_weights: torch.Tensor, total_epochs: int, device: torch.device) -> nn.Module:
    if spec.loss_name == "r_edl":
        return RelaxedEDLLoss(num_classes, class_weights.to(device), total_epochs)

    base = EvidentialFocalLoss(
        gamma=1.2,
        num_classes=num_classes,
        kl_lambda=0.1,
        class_weights=class_weights.to(device),
        warmup_epochs=max(1, int(0.30 * total_epochs)),
        total_epochs=total_epochs,
        disable_efl=spec.disable_efl,
        kl_scaling=spec.kl_scaling,
    )
    if spec.loss_name == "fisher_edl":
        return FisherEDLLoss(base)
    return base


def logits_from_model(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model.fc[0](model.backbone(inputs))


def ce_or_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    p_true: list[float],
    p_train: list[float],
) -> torch.Tensor:
    adjusted = logits
    if spec.logit_adjustment_train:
        delta = torch.tensor(
            [math.log(p_true[c] + 1e-8) - math.log(p_train[c] + 1e-8) for c in range(logits.shape[1])],
            dtype=logits.dtype,
            device=logits.device,
        )
        adjusted = logits + delta

    if spec.loss_name == "ce":
        return F.cross_entropy(adjusted, targets, weight=class_weights.to(logits.device))

    probs = F.softmax(adjusted, dim=1)
    pt = probs.gather(1, targets.view(-1, 1)).clamp_min(1e-8)
    ce = F.cross_entropy(adjusted, targets, weight=class_weights.to(logits.device), reduction="none").view(-1, 1)
    return torch.mean(((1.0 - pt) ** 2.0) * ce)


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
) -> list[dict[str, float]]:
    params = [p for name, p in model.named_parameters() if "scores" not in name]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history: list[dict[str, float]] = []

    if spec.static_sparse:
        set_static_sparse_mode(model)

    criterion = None
    if spec.loss_name not in {"ce", "focal"}:
        criterion = make_loss(spec, model.fc[0].out_features, class_weights, total_epochs, device)

    for epoch in range(total_epochs):
        model.train()
        if spec.static_sparse:
            set_static_sparse_mode(model)
        losses = []
        start = time.time()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                if spec.loss_name in {"ce", "focal"}:
                    logits = logits_from_model(model, inputs)
                    loss = ce_or_focal_loss(logits, targets, spec, class_weights, p_true, p_train)
                else:
                    evidence = model(inputs)
                    loss = criterion(evidence, targets, epoch=epoch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        avg_loss = float(np.mean(losses)) if losses else 0.0
        elapsed = time.time() - start
        history.append({"epoch": epoch + 1, "loss": avg_loss, "seconds": elapsed})
        print(f"Epoch [{epoch + 1:>2}/{total_epochs}] | loss={avg_loss:.4f} | {elapsed:.1f}s")
    return history


def train_guds(
    model: nn.Module,
    train_loader,
    device: torch.device,
    spec: ExperimentSpec,
    class_weights: torch.Tensor,
    total_epochs: int,
    lr: float,
) -> list[dict[str, float]]:
    warmup_epochs = max(1, int(0.30 * total_epochs))
    criterion = make_loss(spec, model.fc[0].out_features, class_weights, total_epochs, device)
    params = [p for name, p in model.named_parameters() if "scores" not in name]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)
    trainer_args = SimpleNamespace(
        disable_pruner=spec.disable_pruner,
        disable_regrower=spec.disable_regrower,
        pruner_type=spec.pruner_type,
        regrower_type=spec.regrower_type,
        kl_scaling=spec.kl_scaling,
        disable_efl=spec.disable_efl,
        disable_anticryst=spec.disable_anticryst,
        use_anticryst=not spec.disable_anticryst,
    )
    trainer = MDEPTrainer(model, optimizer, criterion, total_epochs, warmup_epochs, args=trainer_args)
    history = []
    for epoch in range(total_epochs):
        loss = trainer.train_epoch(epoch, train_loader, device, print_interval=200)
        history.append({"epoch": epoch + 1, "loss": float(loss), "gamma": float(trainer.step_gamma(epoch))})
        print(f"Epoch [{epoch + 1:>2}/{total_epochs}] | loss={loss:.4f} | gamma={trainer.step_gamma(epoch):.4f}")
    return history


def run_one(spec: ExperimentSpec, args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    run_dir = output_root() / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 90}\nRunning {spec.name}: {spec.description}\nOutput: {run_dir}\nDevice: {device}\n{'=' * 90}")

    loaders = get_imbalanced_dataloaders(
        batch_size=args.batch_size,
        test_ratio=args.test_ratio,
        subsample_ratio=args.subsample_ratio,
    )
    train_loader, val_loader, cal_loader, test_loader, num_classes, class_weights, p_true, p_train = loaders

    model = ResNetEvidenceModel(
        num_classes=num_classes,
        flexible=(spec.name == "flexible_edl"),
        pretrained=not args.no_pretrained,
    )
    if spec.sparse:
        replace_conv2d_with_mdep(model)
    model = model.to(device)

    if spec.use_mdep_trainer:
        history = train_guds(model, train_loader, device, spec, class_weights, args.epochs, args.lr)
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
        )

    temperature, bias, thresholds = calibrate_temperature(model, cal_loader, device)
    decision_support = AdaptiveThresholdDecisionSupport(
        model,
        is_resnet=True,
        thresholds=thresholds,
        temperature=temperature,
        bias=bias,
        true_class_prior=p_true,
        train_class_prior=p_train,
    )
    evaluate_adaptive_modes(decision_support, test_loader, device)
    decision_support.restore_model()

    if hasattr(model.fc[1], "logit_adjustment"):
        adjustment = [math.log(p_true[c] + 1e-8) - math.log(p_train[c] + 1e-8) for c in range(num_classes)]
        model.fc[1].logit_adjustment = torch.tensor(adjustment, dtype=torch.float32, device=device)

    _, metrics = evaluate(
        model,
        val_loader,
        test_loader,
        device,
        num_classes=num_classes,
        temperature=temperature,
        bias=bias,
        plot=False,
    )
    if spec.sparse:
        print_sparsity_report(model)

    result = {
        "experiment": asdict(spec),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "temperature": temperature,
        "bias": bias,
        "thresholds": thresholds,
        "p_true": p_true,
        "p_train": p_train,
        "history": history,
        "metrics": metrics,
    }
    (run_dir / "run_config.json").write_text(json.dumps(json_safe(asdict(spec)), indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run_dir / "model_state.pth")
    print(f"[DONE] Saved metrics and model state to {run_dir}")
    return result


def selected_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    if args.experiment:
        return [EXPERIMENTS[name] for name in args.experiment]
    names = SUITES[args.suite]
    return [EXPERIMENTS[name] for name in names]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ISIC 2024 paper experiments.")
    parser.add_argument("--experiment", action="append", choices=sorted(EXPERIMENTS), help="Run one experiment; can be repeated.")
    parser.add_argument("--suite", choices=sorted(SUITES), default="main_tables")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.20)
    parser.add_argument("--subsample_ratio", type=int, default=20)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    output_root().mkdir(parents=True, exist_ok=True)
    all_results = []
    for spec in selected_experiments(args):
        all_results.append(run_one(spec, args))

    summary_path = output_root() / "isic_summary.json"
    summary_path.write_text(json.dumps(json_safe(all_results), indent=2), encoding="utf-8")
    print(f"\nAll selected ISIC experiments completed. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
