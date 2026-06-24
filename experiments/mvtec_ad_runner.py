import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torchvision import models, transforms
from PIL import Image

# Add parent directory to path to import core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guds_edl_core import (
    EvidenceLayer, replace_conv2d_with_mdep, EvidentialFocalLoss, 
    MDEPTrainer, compute_uncertainties
)
from experiments.metrics_ext import binary_image_anomaly_metrics, collect_evidential_outputs
from experiments.isic_paper_experiments import prior_logit_delta

class MVTecImageLevelDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def _find_mvtec_category_dir(category):
    candidates = []
    if os.environ.get("MVTEC_ROOT"):
        candidates.append(os.environ["MVTEC_ROOT"])
    candidates.extend([
        "./data/mvtec_ad",
        "./data/mvtec",
        "/kaggle/input",
    ])

    for base in candidates:
        if not os.path.isdir(base):
            continue
        direct = os.path.join(base, category)
        if os.path.isdir(direct) and os.path.isdir(os.path.join(direct, "test")):
            return direct
        for root, dirs, _ in os.walk(base):
            if os.path.basename(root).lower() == category.lower() and os.path.isdir(os.path.join(root, "test")):
                return root
            if root.replace(base, "").count(os.sep) > 4:
                dirs[:] = []
    return None


def _list_images(root_dir):
    samples = []
    if not os.path.isdir(root_dir):
        return samples
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                samples.append(os.path.join(root, file_name))
    return sorted(samples)


def _collect_mvtec_official_samples(category_dir):
    """Return official MVTec image-level pools without mixing train/test."""
    train_good = [(path, 0) for path in _list_images(os.path.join(category_dir, "train", "good"))]
    test_good = [(path, 0) for path in _list_images(os.path.join(category_dir, "test", "good"))]
    test_defect = []
    test_dir = os.path.join(category_dir, "test")
    if os.path.isdir(test_dir):
        for defect_type in sorted(os.listdir(test_dir)):
            if defect_type.lower() == "good":
                continue
            defect_dir = os.path.join(test_dir, defect_type)
            test_defect.extend((path, 1) for path in _list_images(defect_dir))
    return train_good, test_good, test_defect


def _split_eval_samples(samples, seed=42):
    rng = np.random.default_rng(seed)
    val_samples, cal_samples, test_samples = [], [], []
    for label in [0, 1]:
        class_samples = [sample for sample in samples if sample[1] == label]
        rng.shuffle(class_samples)
        n = len(class_samples)
        if n < 3:
            raise ValueError(f"Need at least 3 evaluation samples for class {label}, found {n}.")
        n_val = max(1, int(round(0.20 * n)))
        n_cal = max(1, int(round(0.20 * n)))
        if n_val + n_cal >= n:
            n_val, n_cal = 1, 1
        val_samples.extend(class_samples[:n_val])
        cal_samples.extend(class_samples[n_val:n_val + n_cal])
        test_samples.extend(class_samples[n_val + n_cal:])
    rng.shuffle(val_samples)
    rng.shuffle(cal_samples)
    rng.shuffle(test_samples)
    return val_samples, cal_samples, test_samples


def get_mvtec_ad_classification_dataloaders(
    category="hazelnut",
    batch_size=32,
    seed=42,
    allow_dummy_data=False,
    defect_train_fraction=0.20,
):
    """
    Load MVTec AD as a supervised few-shot image-level classifier.
    Normal images = Class 0 (majority), defective images = Class 1 (minority).
    Official train/good images are kept for training; a small labeled fraction
    of official test defects is used for supervised positives, and the remaining
    official test images are split into validation, calibration, and final test.
    Dummy tensors are available only for explicit dry-runs.
    """
    print(f"Loading MVTec AD ({category}) for Image-Level Classification...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    category_dir = _find_mvtec_category_dir(category)
    if category_dir is not None:
        train_good, test_good, test_defect = _collect_mvtec_official_samples(category_dir)
        if len(train_good) >= 4 and len(test_good) >= 4 and len(test_defect) >= 8:
            print(f"✅ Found real MVTec category at: {category_dir}")
            print(
                "📊 Official pools: "
                f"train_good={len(train_good)} | test_good={len(test_good)} | test_defect={len(test_defect)}"
            )
            rng = np.random.default_rng(seed)
            defect_indices = np.arange(len(test_defect))
            rng.shuffle(defect_indices)
            n_defect_train = max(1, int(round(len(test_defect) * defect_train_fraction)))
            n_defect_train = min(n_defect_train, max(1, len(test_defect) - 6))
            defect_train = [test_defect[i] for i in defect_indices[:n_defect_train]]
            defect_eval = [test_defect[i] for i in defect_indices[n_defect_train:]]
            eval_samples = test_good + defect_eval
            val_samples, cal_samples, test_samples = _split_eval_samples(eval_samples, seed=seed)
            train_samples = train_good + defect_train

            print(
                "📊 Supervised few-shot split: "
                f"train={len(train_samples)} (defect={len(defect_train)}) | "
                f"val={len(val_samples)} | cal={len(cal_samples)} | test={len(test_samples)}"
            )
            train_ds = MVTecImageLevelDataset(train_samples, transform=transform)
            val_ds = MVTecImageLevelDataset(val_samples, transform=transform)
            cal_ds = MVTecImageLevelDataset(cal_samples, transform=transform)
            test_ds = MVTecImageLevelDataset(test_samples, transform=transform)
            workers = 2 if os.name != 'nt' else 0
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
            cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
            train_labels = [label for _, label in train_samples]
            eval_labels = [label for _, label in eval_samples]
            counts = np.bincount(train_labels, minlength=2)
            cw = torch.tensor([1.0, max(1.0, np.sqrt(counts[0] / max(counts[1], 1)))], dtype=torch.float32)
            eval_counts = np.bincount(eval_labels, minlength=2)
            p_true = [eval_counts[0] / max(eval_counts.sum(), 1), eval_counts[1] / max(eval_counts.sum(), 1)]
            p_train = [counts[0] / max(counts.sum(), 1), counts[1] / max(counts.sum(), 1)]
            return train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train

    if not allow_dummy_data:
        raise FileNotFoundError(
            f"Real MVTec category '{category}' not found. Add the MVTec AD Kaggle "
            "dataset so category folders such as bottle/ and hazelnut/ are visible "
            "under /kaggle/input, or set MVTEC_ROOT. Use --allow_dummy_data only for dry-runs."
        )

    print("⚠ Real MVTec category not found. Falling back to dummy tensors because allow_dummy_data=True.")
    # Represents 500 normal samples and 20 anomalies (1:25 extreme imbalance)
    X_normal = torch.randn(500, 3, 224, 224)
    y_normal = torch.zeros(500, dtype=torch.long)
    X_anomaly = torch.randn(20, 3, 224, 224) * 1.5 + 0.5
    y_anomaly = torch.ones(20, dtype=torch.long)
    
    X = torch.cat([X_normal, X_anomaly])
    Y = torch.cat([y_normal, y_anomaly])
    dataset = TensorDataset(X, Y)
    
    train_len = 350
    val_len = 50
    cal_len = 50
    test_len = len(dataset) - train_len - val_len - cal_len
    
    train_ds, val_ds, cal_ds, test_ds = random_split(
        dataset, [train_len, val_len, cal_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    cw = torch.tensor([1.0, np.sqrt(500.0 / 20.0)], dtype=torch.float32)
    p_true = [0.5, 0.5] # Assume uninformative prior for test
    p_train = [500.0/520.0, 20.0/520.0]
    
    return train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train


@torch.no_grad()
def _collect_binary_logits_labels(model, loader, device):
    model.eval()
    logits_list = []
    labels_list = []
    linear = model.fc[0]
    original_fc = model.fc
    try:
        model.fc = nn.Identity()
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits_list.append(linear(model(inputs)).detach())
            labels_list.append(targets.to(device))
    finally:
        model.fc = original_fc
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def calibrate_mvtec_binary_image(model, cal_loader, device, mode, p_true, p_train):
    logits, labels = _collect_binary_logits_labels(model, cal_loader, device)
    prior_delta = prior_logit_delta(p_true, p_train, 2, device=logits.device, dtype=logits.dtype)
    logits_for_calibration = logits + prior_delta
    evidence_layer = model.fc[1]

    def calibration_loss(scaled_logits):
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
            loss = calibration_loss(logits_for_calibration / temp_param.clamp_min(0.1))
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = None
    elif mode == "bias_temperature":
        temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
        bias_param = nn.Parameter(torch.zeros(2, device=device))
        optimizer = optim.LBFGS([temp_param, bias_param], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            model.zero_grad(set_to_none=True)
            loss = calibration_loss(logits_for_calibration / temp_param.clamp_min(0.1) + bias_param)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = max(0.1, float(temp_param.detach().item()))
        bias = bias_param.detach()
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")

    print(
        f"[CAL] MVTec binary-image mode={mode} | T={temperature:.4f} | "
        f"prior_delta={prior_delta.detach().cpu().numpy()} | "
        f"bias={None if bias is None else bias.detach().cpu().numpy()}"
    )
    return temperature, bias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVTec AD Image-Level Benchmark for GUDS-EDL")
    parser.add_argument("--category", type=str, default="hazelnut")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_dummy_data", action="store_true", help="Permit synthetic dummy data for dry-runs only.")
    parser.add_argument("--defect_train_fraction", type=float, default=0.20, help="Fraction of official test defects used as labeled few-shot training positives.")
    
    # GUDS-EDL Ablation Flags
    parser.add_argument('--disable_pruner', action='store_true')
    parser.add_argument('--disable_regrower', action='store_true')
    parser.add_argument('--pruner_type', type=str, default='signed_first_order', choices=['signed_first_order', 'absolute_grad', 'magnitude', 'random'])
    parser.add_argument('--regrower_type', type=str, default='class_conditioned', choices=['kl_uniform', 'class_conditioned', 'gradient', 'random'])
    parser.add_argument('--kl_scaling', type=str, default='asymmetric', choices=['asymmetric', 'symmetric'])
    parser.add_argument('--disable_efl', action='store_true')
    parser.add_argument('--disable_anticryst', action='store_true')
    
    args = parser.parse_args()
    args.use_anticryst = not args.disable_anticryst

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")
    print(f"Starting GUDS-EDL on MVTec AD ({args.category})")
    print(f"⚙️ Ablations: {vars(args)}")
    
    train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train = get_mvtec_ad_classification_dataloaders(
        args.category,
        args.batch_size,
        seed=args.seed,
        allow_dummy_data=args.allow_dummy_data,
        defect_train_fraction=args.defect_train_fraction,
    )
    
    # 1. Initialize Binary Classification Model (ResNet-18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 2),
        EvidenceLayer(activation='softplus')
    )
    nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)
    nn.init.constant_(model.fc[0].bias, 0)
    
    # 2. Convert to Sparse
    replace_conv2d_with_mdep(model)
    model = model.to(device)
    
    # 3. Setup Loss and Trainer
    warmup_epochs = max(1, int(0.2 * args.epochs))
    criterion = EvidentialFocalLoss(
        gamma=1.2, num_classes=2, kl_lambda=0.1,
        class_weights=cw.to(device),
        warmup_epochs=warmup_epochs, total_epochs=args.epochs,
        disable_efl=args.disable_efl, kl_scaling=args.kl_scaling
    )
    
    trainable_params = [p for name, p in model.named_parameters() if 'scores' not in name]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    trainer = MDEPTrainer(model, optimizer, criterion, args.epochs, warmup_epochs, args=args)
    
    # 4. Train Loop
    start_time = time.time()
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(epoch, train_loader, device, print_interval=10)
        phase = "Warm-up" if epoch < warmup_epochs else "Dynamic 2:4"
        gamma = trainer.step_gamma(epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}] | {phase} | loss: {loss:.4f} | gamma: {gamma:.4f}")
        
    print(f"Training finished in {(time.time()-start_time)/60:.1f} minutes.")
    
    # 5. Calibration & Evaluation
    print("\n--- Running MVTec Binary-Image Calibration ---")
    temperature, bias = calibrate_mvtec_binary_image(
        model,
        cal_loader,
        device,
        "bias_temperature",
        p_true,
        p_train,
    )
    
    print("\n--- Final Test Evaluation ---")
    prior_delta = prior_logit_delta(p_true, p_train, 2, device=device, dtype=torch.float32)
    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
    model.fc[1].logit_adjustment = torch.zeros(1, dtype=torch.float32, device=device)
    
    outputs = collect_evidential_outputs(model, test_loader, device, temperature=temperature, bias=eval_bias)
    metrics = binary_image_anomaly_metrics(outputs["y_true"], outputs["probs"])
    
    print("\n✅ MVTec AD Summary Results:")
    print(f"  Image AUROC: {metrics.get('image_auroc', 0):.4f}")
    print(f"  Image AP:    {metrics.get('image_ap', 0):.4f}")
    print(f"  F1-max:      {metrics.get('f1_max', 0):.4f}")
    print(f"  Bal. Acc@.5: {metrics.get('balanced_accuracy_default', 0):.4f}")
    print(f"  ECE@.5:      {metrics.get('ece_default', 0):.4f}")
    
    print("\nRun completed. Update Table 2 in main_text.tex with the results.")
