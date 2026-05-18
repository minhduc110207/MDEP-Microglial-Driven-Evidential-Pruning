"""
============================================================================
  MDEP — Ablation Study: PRUNE-ONLY (Microglia Only)
  Single-file Kaggle Notebook version
  
  This ablation disables the Astrocyte (growing) agent entirely.
  Only the Microglia (pruning) agent drives sparsity decisions.
  Compare results with the full MDEP and grow-only ablation.
  
  HOW TO RUN ON KAGGLE:
    1. Create a new Notebook, set Accelerator to GPU (T4 or P100).
    2. Click "Add Data" → search "ISIC 2024" → add the challenge dataset.
    3. Copy-paste this entire file into a single code cell.
    4. Run the cell.
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from PIL import Image
import io
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, f1_score, precision_recall_curve, auc
)
# ============================================================================
#  SECTION 1 — EDL Core (Evidential Deep Learning foundations)
# ============================================================================

class EvidenceLayer(nn.Module):
    """
    Ensures the output of the network is non-negative evidence (e >= 0).
    Replaces the traditional Softmax layer for EDL.
    """
    def __init__(self, activation='softplus'):
        super(EvidenceLayer, self).__init__()
        if activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.activation(x)


def compute_uncertainties(evidence):
    """
    Computes epistemic and aleatoric uncertainties from the evidence.

    Args:
        evidence (torch.Tensor): Output evidence of shape (batch_size, num_classes)

    Returns:
        dict: Contains epistemic (u_e), aleatoric (u_a), alpha, and S.
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    K = evidence.shape[1]

    # Epistemic Uncertainty: u_e = K / S
    u_e = K / S

    # Aleatoric Uncertainty: u_a = - sum (alpha_c / S) * (psi(S+1) - psi(alpha_c+1))
    digamma_S = torch.digamma(S + 1.0)
    digamma_alpha = torch.digamma(alpha + 1.0)
    u_a_term = (alpha / S) * (digamma_S - digamma_alpha)
    u_a = -torch.sum(u_a_term, dim=1, keepdim=True)

    return {
        'epistemic': u_e,
        'aleatoric': u_a,
        'alpha': alpha,
        'S': S,
    }


# ============================================================================
#  SECTION 2 — Loss Functions (Evidential Focal Loss + KL regularization)
# ============================================================================

def kl_divergence(alpha, num_classes):
    """
    KL divergence between a Dirichlet(alpha) and a uniform Dirichlet(1,...,1).
    """
    beta = torch.ones(1, num_classes, dtype=torch.float32, device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


class EvidentialFocalLoss(nn.Module):
    """
    Evidential Focal Loss (EFL) with KL Divergence Regularization.
    The focal weight modulates the CE term — not the evidence space directly —
    so the Dirichlet structure stays valid even on highly imbalanced data.
    """
    def __init__(self, gamma=2.0, num_classes=10, kl_lambda=0.1, class_weights=None, annealing_epochs=10):
        super(EvidentialFocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.kl_lambda = kl_lambda
        self.annealing_epochs = annealing_epochs
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(self, evidence, targets, epoch=None):
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=self.num_classes).float()

        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)

        p_hat = alpha / S

        loss_ce = torch.sum(
            targets * (torch.digamma(S) - torch.digamma(alpha)),
            dim=1, keepdim=True,
        )

        p_target = torch.sum(targets * p_hat, dim=1, keepdim=True)
        focal_weight = (1.0 - p_target.detach()) ** self.gamma

        if self.class_weights is not None:
            sample_weight = torch.sum(targets * self.class_weights.unsqueeze(0), dim=1, keepdim=True)
        else:
            sample_weight = 1.0

        alpha_tilde = targets + (1 - targets) * alpha
        loss_kl = kl_divergence(alpha_tilde, self.num_classes)

        # KL Annealing
        if epoch is not None and self.annealing_epochs > 0:
            annealing_coef = min(1.0, epoch / self.annealing_epochs)
        else:
            annealing_coef = 1.0

        loss = sample_weight * focal_weight * loss_ce + self.kl_lambda * annealing_coef * loss_kl
        return torch.mean(loss)


# ============================================================================
#  SECTION 3 — MDEP Multi-Agent Sparsity Engine
# ============================================================================

class SmoothedSTE(torch.autograd.Function):
    """
    Smoothed Straight-Through Estimator.
    Forward: passes the hard binary mask unchanged.
    Backward: approximates dM/dS ≈ sigma'(S/gamma) so gradients flow to
              dormant connections for the Astrocyte agent.
    """
    @staticmethod
    def forward(ctx, scores, mask, gamma):
        ctx.save_for_backward(scores, torch.tensor(gamma))
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        scores, gamma = ctx.saved_tensors
        gamma_val = gamma.item()
        sig = torch.sigmoid(scores / gamma_val)
        grad_scores = grad_output * sig * (1.0 - sig) / gamma_val
        return grad_scores, None, None


def generate_2_4_mask(scores):
    """
    Generates an NVIDIA 2:4 structured sparsity mask.
    For every contiguous block of 4 elements the top-2 (by score) survive.
    This replaces a single global threshold tau with a dynamic, local one.
    """
    if scores.numel() % 4 != 0:
        return torch.ones_like(scores)

    shape = scores.shape
    scores_flat = scores.view(-1, 4)
    _, indices = torch.topk(scores_flat, 2, dim=-1)
    mask_flat = torch.zeros_like(scores_flat)
    mask_flat.scatter_(1, indices, 1.0)
    return mask_flat.view(shape)


class MDEPLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with MDEP dynamic sparsity."""
    def __init__(self, in_features, out_features, bias=True):
        super(MDEPLinear, self).__init__(in_features, out_features, bias)
        self.scores = nn.Parameter(torch.randn_like(self.weight))
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.gamma = 1.0
        self.warmup = True

    def forward(self, x):
        if self.warmup:
            effective_weight = self.weight
        else:
            raw_mask = generate_2_4_mask(self.scores)
            self.mask.copy_(raw_mask)
            differentiable_mask = SmoothedSTE.apply(self.scores, self.mask, self.gamma)
            effective_weight = self.weight * differentiable_mask
        return F.linear(x, effective_weight, self.bias)


class MDEPConv2d(nn.Conv2d):
    """Drop-in replacement for nn.Conv2d with MDEP dynamic sparsity."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super(MDEPConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias,
        )
        self.scores = nn.Parameter(torch.randn_like(self.weight))
        self.register_buffer('mask', torch.ones_like(self.weight))
        self.gamma = 1.0
        self.warmup = True

    def forward(self, x):
        if self.warmup:
            effective_weight = self.weight
        else:
            raw_mask = generate_2_4_mask(self.scores)
            self.mask.copy_(raw_mask)
            differentiable_mask = SmoothedSTE.apply(self.scores, self.mask, self.gamma)
            effective_weight = self.weight * differentiable_mask
        return F.conv2d(
            x, effective_weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )


def update_scores_agents(model, beta=1.0):
    """
    ABLATION: Prune-Only — Microglia agent only.
    The Astrocyte (growing) signal G_ij is zeroed out.

    Microglia (§5.2): C_ij = Norm(|w·∂L_EFL/∂w|) + β·Norm(|w·∂u_a/∂w|)
    """
    for module in model.modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            if not hasattr(module, 'grad_L_w'):
                continue

            w_val = module.weight.data

            # --- Microglia agent: pruning score (§5.2) ---
            c1 = torch.abs(w_val * module.grad_L_w)
            c1_norm = c1 / (c1.max() + 1e-8)

            c2 = torch.abs(w_val * getattr(module, 'grad_ua_w', torch.zeros_like(w_val)))
            c2_norm = c2 / (c2.max() + 1e-8)

            C_ij = c1_norm + beta * c2_norm

            # --- Astrocyte agent: DISABLED (zeroed out) ---
            G_ij = torch.zeros_like(C_ij)

            # Update latent scores (only pruning contributes)
            module.scores.data += (C_ij + G_ij) * 0.1


# ============================================================================
#  SECTION 4 — Trainer (warm-up, cosine schedules, amortized gradients)
# ============================================================================

class MDEPTrainer:
    def __init__(self, model, optimizer, criterion, total_epochs, warmup_epochs=15):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

        # Smoothed-STE temperature schedule
        self.gamma_initial = 5.0
        self.gamma_final = 0.05

    def step_gamma(self, epoch):
        """Cosine-annealed temperature for the Smoothed STE."""
        if epoch < self.warmup_epochs:
            return self.gamma_initial
        progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
        gamma = self.gamma_final + 0.5 * (self.gamma_initial - self.gamma_final) * (
            1 + math.cos(math.pi * progress)
        )
        return gamma

    def set_warmup_state(self, is_warmup, gamma):
        for module in self.model.modules():
            if isinstance(module, (MDEPLinear, MDEPConv2d)):
                module.warmup = is_warmup
                module.gamma = gamma

    def compute_amortized_gradients(self, inputs):
        """
        Amortized backward passes that compute:
          • ∂u_a / ∂w      → signal for the Microglia agent
          • ∂u_e / ∂a^(l)  → signal for the Astrocyte agent (per-neuron)
        Called only once per epoch to keep FLOPs low.
        """
        self.model.train()

        # Register forward hooks to capture layer activations for Astrocyte
        activations = {}
        hooks = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)):
                def _hook(module, inp, out, n=name):
                    activations[n] = out
                hooks.append(m.register_forward_hook(_hook))

        outputs = self.model(inputs)
        uncertainties = compute_uncertainties(outputs)

        u_a = torch.mean(uncertainties['aleatoric'])
        u_e = torch.mean(uncertainties['epistemic'])

        # 1. ∂u_a/∂w → Microglia agent (per-weight signal)
        self.optimizer.zero_grad()
        u_a.backward(retain_graph=True)
        for m in self.model.modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)) and m.weight.grad is not None:
                m.grad_ua_w = m.weight.grad.clone().detach()

        # 2. ∂u_e/∂a^(l) → Astrocyte agent (per-neuron signal)
        act_tensors = []
        act_modules = []
        for name, m in self.model.named_modules():
            if isinstance(m, (MDEPLinear, MDEPConv2d)) and name in activations:
                act_tensors.append(activations[name])
                act_modules.append(m)

        if act_tensors:
            self.optimizer.zero_grad()
            grads = torch.autograd.grad(u_e, act_tensors, allow_unused=True)
            for m, grad in zip(act_modules, grads):
                if grad is not None:
                    if isinstance(m, MDEPLinear):
                        m.u_e_node = torch.abs(grad).mean(dim=0).detach()
                    elif isinstance(m, MDEPConv2d):
                        m.u_e_node = torch.abs(grad).mean(dim=(0, 2, 3)).detach()
                else:
                    m.u_e_node = None

        for h in hooks:
            h.remove()
        self.optimizer.zero_grad()

    def train_epoch(self, epoch, dataloader, device, print_interval=200):
        self.model.train()

        is_warmup = epoch < self.warmup_epochs
        gamma = self.step_gamma(epoch)
        self.set_warmup_state(is_warmup, gamma)

        # Manual LR Warmup parameters
        warmup_period = 5
        base_lr = 1e-3

        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = len(dataloader)
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Smooth per-batch LR Warmup
            if epoch < warmup_period:
                current_step = epoch * num_batches + batch_idx
                total_warmup_steps = warmup_period * num_batches
                current_lr = 1e-6 + (base_lr - 1e-6) * (current_step / total_warmup_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                current_lr = base_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            inputs, targets = inputs.to(device), targets.to(device)

            # Amortized uncertainty-gradient pass on the first batch of the epoch
            if not is_warmup and batch_idx == 0:
                self.compute_amortized_gradients(inputs)

            self.optimizer.zero_grad()
            evidence = self.model(inputs)
            loss = self.criterion(evidence, targets, epoch)
            
            # Loss scaling to counteract Focal Loss shrinkage
            scaled_loss = loss * 4.0
            scaled_loss.backward()

            # Gradient clipping and norm tracking
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            total_grad_norm += grad_norm.item()

            # Cache primary gradient for structural updates
            if not is_warmup:
                for m in self.model.modules():
                    if isinstance(m, (MDEPLinear, MDEPConv2d)) and m.weight.grad is not None:
                        m.grad_L_w = m.weight.grad.clone().detach()

            self.optimizer.step()

            # Multi-agent structure optimization (once per epoch)
            if not is_warmup and batch_idx == 0:
                update_scores_agents(self.model)

            total_loss += loss.item()

            # Progress printing
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == num_batches:
                elapsed = time.time() - epoch_start
                avg_time = elapsed / (batch_idx + 1)
                eta = avg_time * (num_batches - batch_idx - 1)
                avg_loss = total_loss / (batch_idx + 1)
                avg_grad = total_grad_norm / (batch_idx + 1)
                print(
                    f"    Batch [{batch_idx+1:>5}/{num_batches}]  "
                    f"| Loss: {avg_loss:.4f}  "
                    f"| LR: {current_lr:.2e}  "
                    f"| GradNorm: {avg_grad:.4f}  "
                    f"| Elapsed: {elapsed/60:.1f}m  "
                    f"| ETA: {eta/60:.1f}m",
                    flush=True,
                )

        return total_loss / num_batches


# ============================================================================
#  SECTION 5 — ISIC 2024 Dataset + ResNet backbone + main()
# ============================================================================

class ISICDataset(Dataset):
    """PyTorch Dataset for the ISIC 2024 Skin Cancer challenge on Kaggle.
    Supports loading images from individual files OR from an HDF5 archive."""
    def __init__(self, dataframe, image_dir, transform=None, hdf5_path=None):
        self.data_frame = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.hdf5_path = hdf5_path
        self._hdf5_file = None

    def _get_hdf5(self):
        """Lazy-open HDF5 file (one handle per worker process)."""
        if self._hdf5_file is None and self.hdf5_path and HAS_H5PY:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        isic_id = self.data_frame.iloc[idx]['isic_id']
        image = None

        # Try 1: Load from individual image file
        img_path = os.path.join(self.image_dir, f"{isic_id}.jpg")
        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                image = None

        # Try 2: Load from HDF5 archive
        if image is None and self.hdf5_path and HAS_H5PY:
            try:
                hf = self._get_hdf5()
                if isic_id in hf:
                    img_bytes = hf[isic_id][()]
                    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception:
                image = None

        # Fallback: black placeholder
        if image is None:
            image = Image.new('RGB', (224, 224), color='black')

        target = self.data_frame.iloc[idx]['target']
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(target, dtype=torch.long)


def get_isic_dataloaders(batch_size=32, test_ratio=0.2):
    """
    Returns (train_loader, test_loader, num_classes).
    Uses stratified 80/20 split. Falls back to dummy data if not on Kaggle.
    """
    num_classes = 2

    # Auto-detect the ISIC dataset path under /kaggle/input/
    # Competition datasets are mounted under /kaggle/input/competitions/<slug>/
    # Regular datasets are mounted under /kaggle/input/<slug>/
    csv_path = None
    image_dir = None
    kaggle_input = '/kaggle/input'
    
    # Debug: show full tree under /kaggle/input/
    print(f"🔍 Checking Kaggle input dir: {kaggle_input}")
    print(f"   Exists? {os.path.isdir(kaggle_input)}")
    if os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            depth = root.replace(kaggle_input, '').count(os.sep)
            if depth < 3:  # Only show first 3 levels
                indent = '   ' + '  ' * depth
                print(f"{indent}📁 {os.path.basename(root)}/")
                for f in files[:5]:  # Show first 5 files per dir
                    print(f"{indent}  📄 {f}")
                if len(files) > 5:
                    print(f"{indent}  ... and {len(files)-5} more files")
    
    def _try_find_dataset(base_dir):
        """Search for train-metadata.csv in a directory and return (csv_path, image_dir) or (None, None)."""
        if not os.path.isdir(base_dir):
            return None, None
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            candidate_csv = os.path.join(folder_path, 'train-metadata.csv')
            if os.path.exists(candidate_csv):
                img_dir = None
                for img_sub in ['train-image/image', 'train-image', 'train-images/image', 'train-images']:
                    candidate_img = os.path.join(folder_path, img_sub)
                    if os.path.isdir(candidate_img):
                        img_dir = candidate_img
                        break
                if img_dir is None:
                    img_dir = os.path.join(folder_path, 'train-image')
                print(f"✅ Found ISIC dataset at: {folder_path}/")
                return candidate_csv, img_dir
        return None, None
    
    # Strategy 1: Check directly under /kaggle/input/<slug>/
    csv_path, image_dir = _try_find_dataset(kaggle_input)
    
    # Strategy 2: Check under /kaggle/input/competitions/<slug>/
    if csv_path is None:
        competitions_dir = os.path.join(kaggle_input, 'competitions')
        csv_path, image_dir = _try_find_dataset(competitions_dir)
    
    # Strategy 3: Recursive scan — check ALL subdirectories up to 2 levels deep
    if csv_path is None and os.path.isdir(kaggle_input):
        for root, dirs, files in os.walk(kaggle_input):
            depth = root.replace(kaggle_input, '').count(os.sep)
            if depth > 2:
                continue
            if 'train-metadata.csv' in files:
                csv_path = os.path.join(root, 'train-metadata.csv')
                for img_sub in ['train-image/image', 'train-image', 'train-images/image', 'train-images']:
                    candidate_img = os.path.join(root, img_sub)
                    if os.path.isdir(candidate_img):
                        image_dir = candidate_img
                        break
                if image_dir is None:
                    image_dir = os.path.join(root, 'train-image')
                print(f"✅ Found ISIC dataset via deep scan at: {root}/")
                break
    
    if csv_path:
        print(f"📂 CSV path:   {csv_path}")
        print(f"📂 Image dir:  {image_dir}")
    else:
        print(f"❌ train-metadata.csv not found anywhere under {kaggle_input}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if csv_path is None or not os.path.exists(csv_path):
        print("⚠ ISIC dataset not found. Falling back to dummy data.")
        X = torch.randn(200, 3, 224, 224)
        Y = torch.randint(0, 2, (200,))
        full = TensorDataset(X, Y)
        tr = Subset(full, range(160))
        te = Subset(full, range(160, 200))
        return (DataLoader(tr, batch_size=batch_size, shuffle=True),
                DataLoader(te, batch_size=batch_size),
                num_classes,
                torch.ones(num_classes))

    df = pd.read_csv(csv_path)
    print(f"📊 Loaded CSV with {len(df)} rows, columns: {list(df.columns[:5])}")
    
    # Detect HDF5 archive for images
    hdf5_path = None
    dataset_root = os.path.dirname(csv_path)
    for hdf5_name in ['train-image.hdf5', 'train-image.h5']:
        candidate = os.path.join(dataset_root, hdf5_name)
        if os.path.exists(candidate):
            hdf5_path = candidate
            print(f"📂 HDF5 archive: {hdf5_path}")
            break
    
    # Debug: list available files in dataset root
    if os.path.isdir(dataset_root):
        print(f"📂 Dataset contents: {os.listdir(dataset_root)}")
    
    # Subsample to keep training feasible within Kaggle session limits.
    # Set MAX_SAMPLES = None to use the full dataset.
    MAX_SAMPLES = None  # None = use full dataset (401K samples)
    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        print(f"📉 Subsampling: {len(df)} → {MAX_SAMPLES} samples (set MAX_SAMPLES=None for full dataset)")
        df = df.groupby('target', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), int(MAX_SAMPLES * len(x) / len(df))), random_state=42)
        ).reset_index(drop=True)
        print(f"   After stratified subsample: {len(df)} samples, target distribution:")
        print(f"   {df['target'].value_counts().to_dict()}")

    train_df, test_df = train_test_split(
        df, test_size=test_ratio, stratify=df['target'], random_state=42,
    )
    print(f"📊 Train: {len(train_df)} samples  |  Test: {len(test_df)} samples")
    train_ds = ISICDataset(train_df, image_dir, transform=train_tf, hdf5_path=hdf5_path)
    test_ds  = ISICDataset(test_df,  image_dir, transform=test_tf,  hdf5_path=hdf5_path)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    class_counts = train_df['target'].value_counts().sort_index()
    total = len(train_df)
    cw = torch.tensor([total / (num_classes * class_counts.get(c, 1)) for c in range(num_classes)],
                       dtype=torch.float32)
    print(f"⚖️  Class weights: {dict(enumerate(cw.tolist()))}")

    return train_loader, test_loader, num_classes, cw


def replace_conv2d_with_mdep(model):
    """Recursively swap nn.Conv2d / nn.Linear → MDEPConv2d / MDEPLinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            new = MDEPConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                bias=(module.bias is not None),
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        elif isinstance(module, nn.Linear):
            new = MDEPLinear(
                module.in_features, module.out_features,
                bias=(module.bias is not None),
            )
            new.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                new.bias.data.copy_(module.bias.data)
            setattr(model, name, new)
        else:
            replace_conv2d_with_mdep(module)


# ============================================================================
#  SECTION 6 — Evaluation, Metrics & Visualization
# ============================================================================

def compute_ece(confidences, accuracies, n_bins=15):
    """Expected Calibration Error with equal-width bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_sizes.append(0)
            continue
        b_acc  = accuracies[mask].mean()
        b_conf = confidences[mask].mean()
        b_size = mask.sum()
        ece += (b_size / len(confidences)) * abs(b_acc - b_conf)
        bin_accs.append(b_acc)
        bin_confs.append(b_conf)
        bin_sizes.append(b_size)
    return ece, bin_accs, bin_confs, bin_sizes


def plot_reliability_diagram(bin_accs, bin_confs, bin_sizes, n_bins=15):
    """Reliability diagram: accuracy vs confidence per bin."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    x = np.arange(n_bins)
    width = 0.8
    ax.bar(x, bin_accs, width, label='Accuracy', color='#4e79a7', alpha=0.85)
    ax.bar(x, bin_confs, width, label='Confidence', color='#e15759', alpha=0.4)
    ax.plot([-0.5, n_bins - 0.5], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax.set_xlabel('Bin')
    ax.set_ylabel('Value')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


def plot_uncertainty_histogram(u_e_correct, u_e_incorrect):
    """Overlaid histograms of epistemic uncertainty for correct vs wrong."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if len(u_e_correct) > 0:
        ax.hist(u_e_correct, bins=40, alpha=0.6, label='Correct', color='#59a14f')
    if len(u_e_incorrect) > 0:
        ax.hist(u_e_incorrect, bins=40, alpha=0.6, label='Incorrect', color='#e15759')
    ax.set_xlabel('Epistemic Uncertainty (u_e)')
    ax.set_ylabel('Count')
    ax.set_title('Uncertainty Distribution')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_true, probs):
    """Precision-Recall Curve with AUC."""
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(recall, precision, color='#86bcB6', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_risk_coverage_curve(y_true, y_pred, confidences):
    """Risk-Coverage curve and AURC (Area Under Risk-Coverage)."""
    # Sort instances by descending confidence
    sorted_indices = np.argsort(-confidences)
    sorted_true = y_true[sorted_indices]
    sorted_pred = y_pred[sorted_indices]
    
    coverages = []
    risks = []
    
    n_samples = len(y_true)
    errors = (sorted_true != sorted_pred).astype(float)
    cumulative_errors = np.cumsum(errors)
    
    for i in range(1, n_samples + 1):
        coverages.append(i / n_samples)
        risks.append(cumulative_errors[i-1] / i)
        
    aurc = auc(coverages, risks)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(coverages, risks, color='#f28e2b', lw=2, label=f'Risk-Coverage (AURC = {aurc:.4f})')
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk (Error Rate)')
    ax.set_title('Risk-Coverage Curve')
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_sparsity_report(model):
    """Per-layer and total sparsity stats + 2:4 pattern check + MACs estimation."""
    print("\n📐 Sparsity & Hardware Metrics Report")
    print("-" * 75)
    total_params = 0
    total_zeros  = 0
    total_macs_dense = 0
    total_macs_sparse = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (MDEPLinear, MDEPConv2d)):
            mask = module.mask
            n = mask.numel()
            z = (mask == 0).sum().item()
            total_params += n
            total_zeros  += z
            sparsity = z / n * 100 if n > 0 else 0.0
            
            # Simple MACs heuristic based on input/output size if possible
            # We assume MACs scales linearly with the number of non-zero parameters
            # for a given input size. We use parameter count as a proxy for MACs savings
            # on Tensor Cores which accelerate 2:4 structured sparsity by exactly 2x.
            macs_dense = n
            macs_sparse = n - z
            total_macs_dense += macs_dense
            total_macs_sparse += macs_sparse
            
            # Check 2:4 pattern
            if n % 4 == 0:
                blocks = mask.view(-1, 4)
                valid = (blocks.sum(dim=1) == 2).all().item()
                pattern = "✅ 2:4 (TensorCore Ready)" if valid else "❌ Not 2:4"
            else:
                pattern = "⚠ skip (size%4≠0)"
            print(f"  {name:30s} | {sparsity:5.1f}% sparse | {pattern}")
            
    overall = total_zeros / total_params * 100 if total_params > 0 else 0.0
    macs_saved = (total_macs_dense - total_macs_sparse) / total_macs_dense * 100 if total_macs_dense > 0 else 0.0
    print("-" * 75)
    print(f"  {'TOTAL PARAMS':30s} | {overall:5.1f}% sparse")
    print(f"  {'THEORETICAL MACs SAVED':30s} | {macs_saved:5.1f}% reduction in MDEP layers")
    print("  *(Note: Ampere GPU Tensor Cores provide 2x speedup for strict 2:4 sparsity)*")
    print()


@torch.no_grad()
def evaluate(model, test_loader, device, num_classes):
    """Full evaluation: metrics, plots, and uncertainty analysis."""
    model.eval()

    all_targets  = []
    all_preds    = []
    all_confs    = []
    all_probs    = []
    all_u_e      = []
    all_u_a      = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        evidence = model(inputs)
        unc = compute_uncertainties(evidence)

        alpha = unc['alpha']
        S     = unc['S']
        p_hat = (alpha / S).cpu().numpy()
        preds = p_hat.argmax(axis=1)
        confs = p_hat.max(axis=1)

        all_targets.append(targets.numpy())
        all_preds.append(preds)
        all_confs.append(confs)
        all_probs.append(p_hat)
        all_u_e.append(unc['epistemic'].cpu().numpy().squeeze())
        all_u_a.append(unc['aleatoric'].cpu().numpy().squeeze())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    confs  = np.concatenate(all_confs)
    probs  = np.concatenate(all_probs, axis=0)
    u_e    = np.concatenate(all_u_e)
    u_a    = np.concatenate(all_u_a)
    correct = (y_pred == y_true).astype(float)

    # ── Scalar Metrics ─────────────────────────────────────────────
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    if num_classes == 2:
        macro_auroc = roc_auc_score(y_true, probs[:, 1], average='macro')
        pr_auc = average_precision_score(y_true, probs[:, 1])
        brier = brier_score_loss(y_true, probs[:, 1])
        try:
            pauc = roc_auc_score(y_true, probs[:, 1], max_fpr=0.2)
        except ValueError:
            pauc = float('nan')
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
    else:
        macro_auroc = roc_auc_score(y_true, probs, multi_class='ovr', average='macro')
        pauc = float('nan')
        pr_auc = float('nan')
        brier = float('nan')
        sensitivity = float('nan')
        specificity = float('nan')

    ece_val, bin_accs, bin_confs, bin_sizes = compute_ece(confs, correct)

    # Minority-ECE (class 1 = malignant)
    minority_mask = (y_true == 1)
    if minority_mask.sum() > 0:
        m_ece, _, _, _ = compute_ece(confs[minority_mask], correct[minority_mask])
    else:
        m_ece = float('nan')

    # ── Print Results ──────────────────────────────────────────────
    print("\n📈 Evaluation Results")
    print("=" * 50)
    print(f"  Balanced Accuracy     : {bal_acc:.4f}")
    print(f"  Macro F1-Score        : {macro_f1:.4f}")
    if num_classes == 2:
        print(f"  Sensitivity (Recall)  : {sensitivity:.4f}")
        print(f"  Specificity           : {specificity:.4f}")
        print(f"  PR-AUC                : {pr_auc:.4f}")
        print(f"  Brier Score           : {brier:.4f}")
    print(f"  Macro-AUROC           : {macro_auroc:.4f}")
    print(f"  pAUC (@ 20% FPR)      : {pauc:.4f}")
    print(f"  ECE (15 bins)         : {ece_val:.4f}")
    print(f"  Minority-ECE (cls 1)  : {m_ece:.4f}")
    print(f"  Mean Epistemic u_e    : {u_e.mean():.4f}")
    print(f"  Mean Aleatoric u_a    : {u_a.mean():.4f}")
    print("=" * 50)

    # ── Plots ──────────────────────────────────────────────────────
    plot_reliability_diagram(bin_accs, bin_confs, bin_sizes)
    plot_uncertainty_histogram(
        u_e[correct.astype(bool)],
        u_e[~correct.astype(bool)],
    )
    if num_classes == 2:
        plot_pr_curve(y_true, probs[:, 1])
    plot_risk_coverage_curve(y_true, y_pred, confs)

    return {
        'balanced_accuracy': bal_acc,
        'macro_auroc': macro_auroc,
        'pauc': pauc,
        'ece': ece_val,
        'minority_ece': m_ece,
        'mean_u_e': float(u_e.mean()),
        'mean_u_a': float(u_a.mean()),
    }


# ============================================================================
#  SECTION 7 — main()
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥  Device: {device}")

    # ── Data (stratified train / test split) ────────────────────────
    train_loader, test_loader, num_classes, class_weights = get_isic_dataloaders(batch_size=32)
    print(f"📊 Classes: {num_classes}")
    print(f"   Train batches: {len(train_loader)}  |  Test batches: {len(test_loader)}")

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        EvidenceLayer(activation='softplus'),
    )
    # Initialize evidence output to be small to prevent KL explosion
    nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)
    nn.init.constant_(model.fc[0].bias, 0)
    replace_conv2d_with_mdep(model)
    model = model.to(device)

    criterion = EvidentialFocalLoss(
        gamma=2.0, num_classes=num_classes, kl_lambda=0.1,
        class_weights=class_weights.to(device),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    total_epochs  = 15
    warmup_epochs = 3

    trainer = MDEPTrainer(model, optimizer, criterion, total_epochs, warmup_epochs)

    # ── Training ───────────────────────────────────────────────────
    print("\n🚀 Starting Training (ABLATION: Prune-Only / Microglia Only)")
    print("=" * 60)
    for epoch in range(total_epochs):
        loss = trainer.train_epoch(epoch, train_loader, device)
        phase = "Warm-up (Dense)" if epoch < warmup_epochs else "Dynamic 2:4 Sparsity"
        gamma = trainer.step_gamma(epoch)
        print(
            f"  Epoch [{epoch+1:>2}/{total_epochs}]  "
            f"| Phase: {phase:<22} "
            f"| γ: {gamma:.4f}  "
            f"| Loss: {loss:.4f}"
        )
    print("=" * 60)
    print("✅ Training complete.\n")

    # ── Evaluation ─────────────────────────────────────────────────
    evaluate(model, test_loader, device, num_classes)
    print_sparsity_report(model)


# ── Run ────────────────────────────────────────────────────────────────
main()
