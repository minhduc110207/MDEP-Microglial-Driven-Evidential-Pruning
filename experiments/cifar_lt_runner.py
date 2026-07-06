import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
from torchvision import transforms, models

# Add parent directory to path to import core framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guds_edl_core import (
    EvidenceLayer, replace_conv2d_with_mdep, EvidentialFocalLoss, 
    MDEPTrainer, configure_training_runtime, dataloader_runtime_kwargs
)


def cifar_data_root() -> str:
    if os.environ.get("CIFAR_ROOT"):
        return os.environ["CIFAR_ROOT"]
    if Path("/kaggle/working").exists():
        return "/kaggle/working/cifar_data"
    return "./data"


def get_cifar100_lt_dataloaders(imbalance_ratio=100, batch_size=128, seed=42):
    """
    Loads CIFAR-100 and applies exponential decay to class frequencies.
    """
    data_root = cifar_data_root()
    print(
        f"Loading CIFAR-100-LT with Imbalance Ratio 1:{imbalance_ratio} "
        f"from {data_root}...",
        flush=True,
    )
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    cifar_folder = Path(data_root) / "cifar-100-python"
    has_local_dataset = cifar_folder.exists() and (cifar_folder / "train").exists() and (cifar_folder / "test").exists()
    should_download = not has_local_dataset

    print(f"[CIFAR] Loading train split (download={should_download})...", flush=True)
    train_ds = torchvision.datasets.CIFAR100(root=data_root, train=True, download=should_download, transform=transform_train)
    print(f"[CIFAR] Loading test split (download={should_download})...", flush=True)
    test_ds = torchvision.datasets.CIFAR100(root=data_root, train=False, download=should_download, transform=transform_test)
    print(f"[CIFAR] Loaded train={len(train_ds)} test={len(test_ds)}", flush=True)
    
    num_classes = 100
    img_num_per_cls = []
    for cls_idx in range(num_classes):
        num = int(500 * (1.0 / imbalance_ratio) ** (cls_idx / (num_classes - 1.0)))
        img_num_per_cls.append(max(1, num))
        
    rng = np.random.default_rng(seed)
    train_targets = np.array(train_ds.targets)
    imbalanced_indices = []
    for cls_idx, num in enumerate(img_num_per_cls):
        idx = np.where(train_targets == cls_idx)[0]
        rng.shuffle(idx)
        imbalanced_indices.extend(idx[:num])
        
    train_ds = Subset(train_ds, imbalanced_indices)
    
    # Split test set (10,000) into val (2000), cal (2000), and test (6000)
    test_len = len(test_ds)
    val_len = int(test_len * 0.2)
    cal_len = int(test_len * 0.2)
    test_final_len = test_len - val_len - cal_len
    
    val_ds, cal_ds, test_final_ds = random_split(
        test_ds, [val_len, cal_len, test_final_len], 
        generator=torch.Generator().manual_seed(seed)
    )

    loader_kwargs = dataloader_runtime_kwargs()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_final_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    # Dampened inverse-frequency weights for GUDS/EDL. Standard CE-style
    # baselines decide separately whether to use class weights.
    cw = [np.sqrt(500.0 / n) for n in img_num_per_cls]
    cw = torch.tensor(cw, dtype=torch.float32)
    cw = cw / cw.min() # Normalize so majority class has weight 1.0
    
    p_train = [n / sum(img_num_per_cls) for n in img_num_per_cls]
    p_true = [1.0 / num_classes] * num_classes
    
    return train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train

if __name__ == "__main__":
    configure_training_runtime()
    parser = argparse.ArgumentParser(description="CIFAR-100-LT Benchmark Runner for GUDS-EDL")
    parser.add_argument("--imbalance_ratio", type=int, default=100, choices=[10, 50, 100])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    
    # GUDS-EDL Ablation Flags
    parser.add_argument('--disable_pruner', action='store_true')
    parser.add_argument('--disable_regrower', action='store_true')
    parser.add_argument('--pruner_type', type=str, default='signed_first_order', choices=['signed_first_order', 'absolute_grad', 'magnitude', 'random'])
    parser.add_argument('--regrower_type', type=str, default='class_conditioned', choices=['kl_uniform', 'class_conditioned', 'gradient', 'random'])
    parser.add_argument('--kl_scaling', type=str, default='asymmetric', choices=['asymmetric', 'symmetric'])
    parser.add_argument('--disable_efl', action='store_true')
    parser.add_argument('--disable_anticryst', action='store_true')
    parser.add_argument('--structural_proxy_batches', type=int, default=8)
    parser.add_argument('--structural_proxy_min_classes', type=int, default=20)
    
    args = parser.parse_args()
    args.use_anticryst = not args.disable_anticryst

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")
    print(f"Starting GUDS-EDL on CIFAR-100-LT (Ratio 1:{args.imbalance_ratio})")
    print(f"⚙️ Ablations: {vars(args)}")
    
    train_loader, val_loader, cal_loader, test_loader, cw, p_true, p_train = get_cifar100_lt_dataloaders(args.imbalance_ratio, args.batch_size, seed=args.seed)
    
    # 1. Initialize ResNet-18 adapted for 32x32 CIFAR images
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 100),
        EvidenceLayer(activation='softplus')
    )
    # Prevent KL explosion at initialization
    nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)
    nn.init.constant_(model.fc[0].bias, 0)
    
    # 2. Convert backbone convs to GUDS-EDL sparse layers. Keep the
    # 100-class classifier head unmasked so CIFAR follows the paper protocol.
    replace_conv2d_with_mdep(model.backbone, learn_permutation=False)
    model = model.to(device)
    
    # 3. Setup Loss and Trainer
    warmup_epochs = max(1, int(0.1 * args.epochs))
    criterion = EvidentialFocalLoss(
        gamma=1.2, num_classes=100, kl_lambda=0.1,
        class_weights=cw.to(device),
        warmup_epochs=warmup_epochs, total_epochs=args.epochs,
        disable_efl=args.disable_efl, kl_scaling=args.kl_scaling
    )
    
    trainable_params = [p for name, p in model.named_parameters() if 'scores' not in name]
    optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    trainer = MDEPTrainer(model, optimizer, criterion, args.epochs, warmup_epochs, args=args, scheduler=scheduler)
    
    # 4. Training Loop
    start_time = time.time()
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(epoch, train_loader, device, print_interval=100)
        phase = "Warm-up" if epoch < warmup_epochs else "Dynamic 2:4"
        gamma = trainer.step_gamma(epoch)
        print(f"Epoch [{epoch+1}/{args.epochs}] | {phase} | loss: {loss:.4f} | gamma: {gamma:.4f}")
        
    print(f"Training finished in {(time.time()-start_time)/60:.1f} minutes.")
    
    # 5. Calibration & Testing
    print("\n--- Running Bias-Corrected Temperature Calibration ---")
    from experiments.generalization_paper_suite import calibrate_multiclass, cifar_class_counts, evaluate_multiclass
    from experiments.isic_paper_experiments import prior_logit_delta

    temperature, bias, _ = calibrate_multiclass(
        model,
        cal_loader,
        device,
        "bias_temperature",
        p_true,
        p_train,
    )
    
    print("\n--- Final Test Evaluation ---")
    prior_delta = prior_logit_delta(p_true, p_train, 100, device=device, dtype=torch.float32)
    eval_bias = prior_delta / max(temperature, 1e-8)
    if bias is not None:
        eval_bias = eval_bias + bias.to(device=device, dtype=eval_bias.dtype)
    model.fc[1].logit_adjustment = torch.zeros(1, dtype=torch.float32, device=device)

    metrics = evaluate_multiclass(
        model,
        test_loader,
        device,
        num_classes=100,
        temperature=temperature,
        bias=eval_bias,
        class_counts=cifar_class_counts(args.imbalance_ratio),
    )

    print("\n✅ CIFAR-100-LT Summary Results:")
    print(f"  Macro-AUROC: {metrics.get('macro_auroc', 0):.4f}")
    print(f"  AURC:        {metrics.get('aurc', 0):.4f}")
    print(f"  ECE (Adp):   {metrics.get('ece_adaptive', 0):.4f}")
    
    print("\nRun completed. Update main_text.tex with the results.")
