"""
Patient-grouped PAD layer4 adaptation with ISIC knowledge distillation.

This is the higher-risk stage after frozen-head adaptation. Sparse scores,
masks, channel order, layers through layer3, and the original ISIC head stay
frozen. Only layer4 weights and a PAD-specific head are optimized.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.generalization_paper_suite import EvidenceResNet
from experiments.isic_paper_experiments import json_safe, seed_everything
from experiments.run_external_validation import (
    PADUFES20MetadataDataset,
    load_state_with_optional_ood_head,
)
from experiments.run_pad_adaptation import (
    DIAGNOSES,
    MALIGNANT_DIAGNOSES,
    apply_temperature_bias,
    choose_global_threshold,
    classification_metrics_from_predictions,
    fit_temperature_bias,
    logit,
    patient_bootstrap_ci,
    resolve_checkpoint,
)
from guds_edl_core import (
    MDEPConv2d,
    MDEPLinear,
    compute_isic_pauc,
    configure_training_runtime,
    get_imbalanced_dataloaders,
    replace_conv2d_with_mdep,
)


class TargetSubset(Dataset):
    def __init__(self, dataset, indices, targets):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=int)
        self.targets = np.asarray(targets, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        image, _, _ = self.dataset[index]
        return image, int(self.targets[index])


class PADStudent(nn.Module):
    def __init__(self, base, num_outputs, dropout):
        super().__init__()
        self.base = base
        self.pad_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_outputs),
        )

    def pad_logits(self, images):
        return self.pad_head(self.base.backbone(images))

    def isic_logits(self, images):
        return self.base.fc[0](self.base.backbone(images))


def binary_probability(logits, target_mode):
    probability = torch.softmax(logits, dim=1)
    if target_mode == "binary":
        return probability[:, 1]
    malignant = [DIAGNOSES.index(name) for name in MALIGNANT_DIAGNOSES]
    return probability[:, malignant].sum(dim=1)


def sparse_state(model):
    return {
        name: (module.scores.detach().cpu().clone(), module.mask.detach().cpu().clone())
        for name, module in model.named_modules()
        if isinstance(module, (MDEPConv2d, MDEPLinear))
    }


def assert_sparse_state_unchanged(model, before):
    for name, module in model.named_modules():
        if name not in before:
            continue
        old_scores, old_mask = before[name]
        if not torch.equal(old_scores, module.scores.detach().cpu()):
            raise RuntimeError(f"Sparse topology scores changed in frozen module {name}.")
        if not torch.equal(old_mask, module.mask.detach().cpu()):
            raise RuntimeError(f"Sparse mask changed in frozen module {name}.")


def configure_trainable(student):
    for parameter in student.parameters():
        parameter.requires_grad = False
    for name, parameter in student.base.backbone.layer4.named_parameters():
        if "scores" not in name and "perm_logits" not in name:
            parameter.requires_grad = True
    for parameter in student.pad_head.parameters():
        parameter.requires_grad = True


def evaluate_pad(student, loader, device, target_mode):
    student.eval()
    scores = []
    with torch.no_grad():
        for images, _ in loader:
            logits = student.pad_logits(images.to(device, non_blocking=True))
            scores.append(binary_probability(logits, target_mode).cpu().numpy())
    return np.concatenate(scores)


def evaluate_isic_ranking(student, loader, device):
    student.eval()
    labels, scores = [], []
    with torch.no_grad():
        for images, target in loader:
            logits = student.isic_logits(images.to(device, non_blocking=True))
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            labels.append(target.numpy())
    y_true, probability = np.concatenate(labels), np.concatenate(scores)
    return {
        "pauc": float(compute_isic_pauc(y_true, probability, min_tpr=0.80)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "auroc": float(roc_auc_score(y_true, probability)),
    }


def train_fold(student, teacher, pad_loader, id_loader, class_weights, args, device):
    configure_trainable(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    id_iterator = iter(id_loader)
    history = []
    for epoch in range(args.epochs):
        student.eval()
        student.base.backbone.layer4.train()
        student.pad_head.train()
        losses = []
        for pad_images, pad_target in pad_loader:
            try:
                id_images, _ = next(id_iterator)
            except StopIteration:
                id_iterator = iter(id_loader)
                id_images, _ = next(id_iterator)
            pad_images = pad_images.to(device, non_blocking=True)
            pad_target = pad_target.to(device, non_blocking=True)
            id_images = id_images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pad_logits = student.pad_logits(pad_images)
            pad_loss = F.cross_entropy(pad_logits, pad_target, weight=class_weights)
            with torch.no_grad():
                teacher_logits = teacher.fc[0](teacher.backbone(id_images))
            student_logits = student.isic_logits(id_images)
            temperature = args.kd_temperature
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
            loss = pad_loss + args.kd_weight * kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
    return history


def main():
    configure_training_runtime()
    parser = argparse.ArgumentParser(description="Layer4+KD PAD adaptation.")
    parser.add_argument("--pad_root", required=True)
    parser.add_argument("--pad_csv", required=True)
    parser.add_argument("--partition", default="all")
    parser.add_argument("--model_path", help="Checkpoint path; may contain {seed}.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--outer_folds", type=int, default=5)
    parser.add_argument("--inner_folds", type=int, default=3)
    parser.add_argument("--target_mode", choices=["binary", "diagnosis6"], default="diagnosis6")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--kd_weight", type=float, default=2.0)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--target_sensitivity", type=float, default=0.80)
    parser.add_argument("--bootstrap_repeats", type=int, default=500)
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--output_dir")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    seed_everything(args.split_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = PADUFES20MetadataDataset(
        args.pad_root, args.pad_csv, args.partition, train_transform, "fitspatrick"
    )
    eval_dataset = PADUFES20MetadataDataset(
        args.pad_root, args.pad_csv, args.partition, eval_transform, "fitspatrick"
    )
    if not np.array_equal(train_dataset.sample_paths, eval_dataset.sample_paths):
        raise RuntimeError("Training and evaluation PAD sample orders differ.")
    y = eval_dataset.targets.astype(int)
    groups = eval_dataset.patient_ids.astype(str)
    diagnosis_map = {name: index for index, name in enumerate(DIAGNOSES)}
    diagnosis_targets = np.asarray([diagnosis_map[value] for value in eval_dataset.diagnostics])
    train_targets = y if args.target_mode == "binary" else diagnosis_targets
    num_outputs = 2 if args.target_mode == "binary" else len(DIAGNOSES)
    loaders = get_imbalanced_dataloaders(
        batch_size=args.batch_size, seed=args.split_seed, allow_dummy_data=False
    )
    _, val_loader, id_cal_loader, _, *_ = loaders
    outer = StratifiedGroupKFold(args.outer_folds, shuffle=True, random_state=args.split_seed)
    oof_probability = np.full(len(y), np.nan)
    oof_prediction = np.full(len(y), -1, dtype=int)
    fold_results = []
    output_dir = Path(args.output_dir) if args.output_dir else (
        (Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT)
        / "paper_experiment_outputs" / "pad_layer4_kd"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold, (outer_train, outer_test) in enumerate(outer.split(np.zeros(len(y)), y, groups), 1):
        inner = StratifiedGroupKFold(args.inner_folds, shuffle=True, random_state=args.split_seed + fold)
        fit_rel, cal_rel = next(inner.split(np.zeros(len(outer_train)), y[outer_train], groups[outer_train]))
        fit_idx, cal_idx = outer_train[fit_rel], outer_train[cal_rel]
        cal_logits, test_logits, seed_results = [], [], []
        for seed in args.seeds:
            checkpoint = resolve_checkpoint(args, seed)
            base = EvidenceResNet(2, "isic", pretrained=False)
            replace_conv2d_with_mdep(base.backbone, learn_permutation=False)
            load_state_with_optional_ood_head(base, checkpoint, device)
            before = sparse_state(base)
            teacher = copy.deepcopy(base).to(device).eval()
            student = PADStudent(base.to(device), num_outputs, args.dropout).to(device)
            pad_loader = DataLoader(
                TargetSubset(train_dataset, fit_idx, train_targets),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            cal_loader = DataLoader(
                TargetSubset(eval_dataset, cal_idx, train_targets),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            test_loader = DataLoader(
                TargetSubset(eval_dataset, outer_test, train_targets),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            counts = np.bincount(train_targets[fit_idx], minlength=num_outputs)
            weights = np.zeros(num_outputs, dtype=np.float32)
            observed = counts > 0
            weights[observed] = len(fit_idx) / (observed.sum() * counts[observed])
            class_weights = torch.as_tensor(weights, device=device)
            history = train_fold(
                student,
                teacher,
                pad_loader,
                id_cal_loader,
                class_weights,
                args,
                device,
            )
            assert_sparse_state_unchanged(student.base, before)
            cal_logits.append(logit(evaluate_pad(student, cal_loader, device, args.target_mode)))
            test_logits.append(logit(evaluate_pad(student, test_loader, device, args.target_mode)))
            isic_validation = evaluate_isic_ranking(student, val_loader, device)
            seed_results.append({
                "seed": seed,
                "history": history,
                "isic_validation": isic_validation,
                "sparse_topology_unchanged": True,
            })
            if args.save_models:
                torch.save(student.state_dict(), output_dir / f"fold_{fold}_seed_{seed}.pth")
            del student, teacher, base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ensemble_cal = np.mean(cal_logits, axis=0)
        ensemble_test = np.mean(test_logits, axis=0)
        temperature, bias = fit_temperature_bias(ensemble_cal, y[cal_idx])
        cal_probability = apply_temperature_bias(ensemble_cal, temperature, bias)
        test_probability = apply_temperature_bias(ensemble_test, temperature, bias)
        threshold = choose_global_threshold(y[cal_idx], cal_probability, args.target_sensitivity)
        oof_probability[outer_test] = test_probability
        oof_prediction[outer_test] = (test_probability >= threshold).astype(int)
        fold_results.append({
            "fold": fold,
            "temperature": temperature,
            "bias": bias,
            "threshold": threshold,
            "seeds": seed_results,
        })
        print(f"[FOLD {fold}] AUROC={roc_auc_score(y[outer_test], test_probability):.4f}")

    valid = np.isfinite(oof_probability)
    overall = classification_metrics_from_predictions(y[valid], oof_probability[valid], oof_prediction[valid])
    overall["patient_bootstrap_95ci"] = patient_bootstrap_ci(
        y[valid], oof_probability[valid], groups[valid], args.bootstrap_repeats, args.split_seed
    )
    result = {
        "protocol": "patient-grouped layer4 adaptation with ISIC calibration-replay KD",
        "target_mode": args.target_mode,
        "seeds": args.seeds,
        "overall_oof": overall,
        "folds": fold_results,
        "constraints": {
            "topology_scores_and_masks": "asserted unchanged after every model fit",
            "layers_through_layer3": "frozen",
            "primary_knn_layer3": "representation unchanged by optimizer",
            "isic_test": "never used for optimization or checkpoint selection",
        },
    }
    pd.DataFrame({
        "path": eval_dataset.sample_paths,
        "patient_id": groups,
        "target": y,
        "probability": oof_probability,
    }).to_csv(output_dir / "pad_layer4_kd_oof_predictions.csv", index=False)
    (output_dir / "pad_layer4_kd_summary.json").write_text(
        json.dumps(json_safe(result), indent=2), encoding="utf-8"
    )
    print(f"[DONE] OOF AUROC={overall['auroc']:.4f} AP={overall['average_precision']:.4f}")


if __name__ == "__main__":
    main()
