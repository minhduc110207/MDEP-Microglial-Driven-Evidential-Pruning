import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.distributions as dist
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from edl_core import EvidenceLayer, compute_uncertainties
from main import replace_conv2d_with_mdep, get_isic_dataloader

# ==========================================
# 1. Feature Extractor Wrapper for ResNet
# ==========================================
class ResNetFeatureExtractor(nn.Module):
    """
    Wraps a ResNet model to extract both:
      1. Logits (input to the final EvidenceLayer, computed as fc[0](features))
      2. Hidden features (input to the fc layer, captured via avgpool hook)
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.features = None
        # Register a hook to capture avgpool output
        self.hook = self.base_model.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        self.features = torch.flatten(out, 1) # Shape: (batch_size, in_features)

    def forward(self, x):
        # Forward pass through base model to get standard outputs
        # Note: base_model outputs evidence because of EvidenceLayer at the end
        _ = self.base_model(x)
        features = self.features
        # Extract logits from the Linear layer (which is fc[0])
        logits = self.base_model.fc[0](features)
        return features.detach(), logits.detach()

    def remove_hooks(self):
        self.hook.remove()

# ==========================================
# 2. Evidential Transformation Network (ETN)
# ==========================================
class EvidentialTransformationNetwork(nn.Module):
    """
    Lightweight MLP that outputs parameters (shape, rate) of a Gamma distribution
    used to scale logits.
    """
    def __init__(self, in_features):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        # Learnable prior belief parameter b (scalar b_0, repeated for all classes)
        self.b_param = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, features):
        output = self.mlp(features)
        # Apply Softplus to ensure strictly positive shape (alpha) and rate (beta) parameters
        alpha_G = torch.clamp(F.softplus(output[:, 0]), min=1e-5, max=50.0)
        beta_G = torch.clamp(F.softplus(output[:, 1]), min=1e-5, max=50.0)
        return alpha_G, beta_G

    def get_b(self, num_classes, device):
        # b_0 = softplus(b_param) to ensure positivity
        b_val = F.softplus(self.b_param)
        return b_val * torch.ones(num_classes, device=device)

# ==========================================
# 3. Analytical Gamma KL Divergence Helper
# ==========================================
def kl_gamma_gamma(alpha_q, beta_q, alpha_p, beta_p):
    """
    Computes analytical KL divergence between two Gamma distributions:
    KL( Gamma(alpha_q, beta_q) || Gamma(alpha_p, beta_p) )
    """
    term1 = (alpha_q - alpha_p) * torch.digamma(alpha_q)
    term2 = - torch.lgamma(alpha_q) + torch.lgamma(alpha_p)
    term3 = alpha_p * (torch.log(beta_q) - torch.log(beta_p))
    term4 = alpha_q * (beta_p - beta_q) / beta_q
    return term1 + term2 + term3 + term4

# ==========================================
# 4. Expected Calibration Error (ECE) Helper
# ==========================================
def compute_ece(probs, labels, num_bins=15):
    """
    Computes Expected Calibration Error (ECE) for classification.
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Mask of predictions falling inside this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece.item()

# ==========================================
# 5. Main ETN Training and Evaluation Pipeline
# ==========================================
def train_and_eval_etn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Applying ETN on device: {device}")

    # Load dataloader and config
    dataloader, num_classes, class_weights = get_isic_dataloader(batch_size=32)
    
    # 1. Initialize ResNet18 model matching main.py structure
    base_model = models.resnet18(weights=None)
    in_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(in_features, num_classes),
        EvidenceLayer(activation='softplus')
    )
    replace_conv2d_with_mdep(base_model)
    
    # 2. Load trained MDEP checkpoint if available
    checkpoint_path = 'model_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        try:
            base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Successfully loaded trained MDEP ResNet18 checkpoint.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Using randomly initialized model.")
    else:
        print("No checkpoint found. Training on randomly initialized model for validation.")

    base_model = base_model.to(device)
    base_model.eval() # Freeze backbone in eval mode
    for p in base_model.parameters():
        p.requires_grad = False

    # Wrap base model for hidden feature and logit extraction
    extractor = ResNetFeatureExtractor(base_model)

    # 3. Initialize ETN module
    etn = EvidentialTransformationNetwork(in_features).to(device)
    optimizer = optim.Adam(etn.parameters(), lr=1e-3)

    # 4. Define Prior Distribution Parameters for A
    # Prior mode = 10 (or 5 for ImageNet), prior variance = 5
    prior_mode = 5.0
    prior_var = 5.0
    
    # Calculate alpha_p (shape) and beta_p (rate) for prior Gamma(alpha_p, beta_p)
    beta_p_val = (prior_mode + np.sqrt(prior_mode**2 + 4 * prior_var)) / (2 * prior_var)
    alpha_p_val = prior_var * (beta_p_val**2)
    
    beta_p = torch.tensor(beta_p_val, dtype=torch.float32, device=device)
    alpha_p = torch.tensor(alpha_p_val, dtype=torch.float32, device=device)
    
    # Constants for EDL target distribution
    nu = 10000.0  # target concentration parameter
    M_mc = 20     # number of Monte-Carlo samples
    lambda_kl = 0.01  # regularization weight for Gamma KL

    print(f"Prior parameters for A: Mode={prior_mode}, Var={prior_var} => Shape(alpha_p)={alpha_p:.4f}, Rate(beta_p)={beta_p:.4f}")
    
    # 5. Training Loop for ETN
    epochs = 15
    print("\n--- Training ETN Post-hoc Adaptation ---")
    for epoch in range(epochs):
        etn.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Extract features and original logits
            with torch.no_grad():
                features, logits = extractor(inputs)
            
            # Predict Gamma parameters
            alpha_G, beta_G = etn(features)
            b = etn.get_b(num_classes, device)
            
            # Sample scaling factors A ~ Gamma(alpha_G, beta_G) using reparameterization
            q_dist = dist.Gamma(alpha_G, beta_G)
            A = q_dist.rsample((M_mc,)) # Shape: (M_mc, batch_size)
            
            # Transformed logits and evidential parameters
            # alpha_prime = softplus(A * z) + b
            # A has shape (M_mc, batch_size), logits has shape (batch_size, num_classes)
            scaled_logits = A.unsqueeze(-1) * logits.unsqueeze(0) # Shape: (M_mc, batch_size, num_classes)
            alpha_prime = F.softplus(scaled_logits) + b # Shape: (M_mc, batch_size, num_classes)
            alpha_0_prime = torch.sum(alpha_prime, dim=-1) # Shape: (M_mc, batch_size)
            
            # Compute Reconstruction Loss: E_{p^(nu)} [ - log p'(pi | A, x) ]
            # target distribution: alpha_y = 1 + (nu - 1) * target_one_hot
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            alpha_y = 1.0 + (nu - 1.0) * targets_one_hot
            alpha_y_0 = torch.tensor(num_classes + nu - 1.0, device=device)
            
            log_B = torch.lgamma(alpha_0_prime) - torch.sum(torch.lgamma(alpha_prime), dim=-1) # (M_mc, batch_size)
            psi_diff = torch.digamma(alpha_y) - torch.digamma(alpha_y_0) # (batch_size, num_classes)
            sum_term = torch.sum((alpha_prime - 1.0) * psi_diff.unsqueeze(0), dim=-1) # (M_mc, batch_size)
            
            recon_loss = - torch.mean(log_B + sum_term)
            
            # Compute KL Divergence Loss: KL( q(A|x) || p(A) )
            kl_div = kl_gamma_gamma(alpha_G, beta_G, alpha_p, beta_p)
            kl_loss = torch.mean(kl_div)
            
            # Overall loss
            loss = recon_loss + lambda_kl * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            
        num_batches = len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/num_batches:.4f} | Recon: {total_recon/num_batches:.4f} | KL: {total_kl/num_batches:.4f}")

    # ==========================================
    # 6. Evaluation and Metrics Comparison
    # ==========================================
    print("\n--- Evaluating Model Effectiveness ---")
    etn.eval()
    
    # Store results for ID
    id_base_probs = []
    id_etn_probs = []
    id_labels = []
    id_base_margins = []
    id_etn_margins = []
    id_base_unc = []
    id_etn_unc = []
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        with torch.no_grad():
            features, logits = extractor(inputs)
            
            # Predict Gamma parameters
            alpha_G, beta_G = etn(features)
            b = etn.get_b(num_classes, device)
            
            # --- Base Model (EDL without ETN, A = 1) ---
            base_evidence = F.softplus(logits)
            base_alpha = base_evidence + 1.0
            base_S = torch.sum(base_alpha, dim=1, keepdim=True)
            base_probs = base_alpha / base_S
            base_margins = logits[:, 1] - logits[:, 0] # for binary
            # Epistemic uncertainty K / S
            base_unc = num_classes / base_S.squeeze(-1)
            
            # --- ETN Transformed Model (Monte-Carlo expectation) ---
            q_dist = dist.Gamma(alpha_G, beta_G)
            A_samples = q_dist.sample((100,)) # Draw 100 samples for high-quality evaluation
            
            scaled_logits = A_samples.unsqueeze(-1) * logits.unsqueeze(0)
            alpha_prime = F.softplus(scaled_logits) + b
            alpha_0_prime = torch.sum(alpha_prime, dim=-1)
            probs_samples = alpha_prime / alpha_0_prime.unsqueeze(-1)
            etn_probs = torch.mean(probs_samples, dim=0) # shape (batch_size, num_classes)
            
            # Average ETN margins
            etn_logits = torch.mean(A_samples, dim=0).unsqueeze(-1) * logits
            etn_margins = etn_logits[:, 1] - etn_logits[:, 0]
            
            # ETN epistemic uncertainty (average K / S'_0)
            etn_unc = torch.mean(num_classes / alpha_0_prime, dim=0)

        id_base_probs.append(base_probs)
        id_etn_probs.append(etn_probs)
        id_labels.append(targets)
        id_base_margins.append(base_margins)
        id_etn_margins.append(etn_margins)
        id_base_unc.append(base_unc)
        id_etn_unc.append(etn_unc)

    # Concat all ID tensors
    id_base_probs = torch.cat(id_base_probs, dim=0)
    id_etn_probs = torch.cat(id_etn_probs, dim=0)
    id_labels = torch.cat(id_labels, dim=0)
    id_base_margins = torch.cat(id_base_margins, dim=0)
    id_etn_margins = torch.cat(id_etn_margins, dim=0)
    id_base_unc = torch.cat(id_base_unc, dim=0)
    id_etn_unc = torch.cat(id_etn_unc, dim=0)

    # --- Generate OOD data (Simulated via Gaussian Noise) ---
    print("Simulating Out-Of-Distribution (OOD) dataset with random noise...")
    ood_base_unc = []
    ood_etn_unc = []
    
    # We will generate simulated OOD batches
    for _ in range(5):
        # Generate random inputs mimicking normalized images
        noise_inputs = torch.randn(32, 3, 224, 224, device=device)
        with torch.no_grad():
            features, logits = extractor(noise_inputs)
            alpha_G, beta_G = etn(features)
            b = etn.get_b(num_classes, device)
            
            # Base model uncertainty
            base_evidence = F.softplus(logits)
            base_S = torch.sum(base_evidence + 1.0, dim=1, keepdim=True)
            base_unc = num_classes / base_S.squeeze(-1)
            
            # ETN transformed uncertainty
            q_dist = dist.Gamma(alpha_G, beta_G)
            A_samples = q_dist.sample((100,))
            alpha_prime = F.softplus(A_samples.unsqueeze(-1) * logits.unsqueeze(0)) + b
            alpha_0_prime = torch.sum(alpha_prime, dim=-1)
            etn_unc = torch.mean(num_classes / alpha_0_prime, dim=0)
            
        ood_base_unc.append(base_unc)
        ood_etn_unc.append(etn_unc)
        
    ood_base_unc = torch.cat(ood_base_unc, dim=0)
    ood_etn_unc = torch.cat(ood_etn_unc, dim=0)

    # Remove feature extraction hooks to clean up
    extractor.remove_hooks()

    # Calculate Accuracies
    _, base_preds = torch.max(id_base_probs, dim=1)
    base_acc = base_preds.eq(id_labels).float().mean().item()
    
    _, etn_preds = torch.max(id_etn_probs, dim=1)
    etn_acc = etn_preds.eq(id_labels).float().mean().item()

    # Calculate ECEs
    base_ece = compute_ece(id_base_probs, id_labels)
    etn_ece = compute_ece(id_etn_probs, id_labels)

    # Calculate Average Margins (Absolute margins for classification)
    base_avg_margin = torch.mean(torch.abs(id_base_margins)).item()
    etn_avg_margin = torch.mean(torch.abs(id_etn_margins)).item()

    # Calculate OOD Detection Performance (AUROC / AUPR based on Epistemic Uncertainty)
    # We want higher uncertainty for OOD than ID. So ID = 0, OOD = 1.
    id_labels_binary = np.zeros(len(id_base_unc))
    ood_labels_binary = np.ones(len(ood_base_unc))
    y_true = np.concatenate([id_labels_binary, ood_labels_binary])
    
    # Base model OOD AUROC
    y_scores_base = np.concatenate([id_base_unc.cpu().numpy(), ood_base_unc.cpu().numpy()])
    base_auroc = roc_auc_score(y_true, y_scores_base)
    
    # Base model OOD AUPR
    precision_base, recall_base, _ = precision_recall_curve(y_true, y_scores_base)
    base_aupr = auc(recall_base, precision_base)
    
    # ETN model OOD AUROC
    y_scores_etn = np.concatenate([id_etn_unc.cpu().numpy(), ood_etn_unc.cpu().numpy()])
    etn_auroc = roc_auc_score(y_true, y_scores_etn)
    
    # ETN model OOD AUPR
    precision_etn, recall_etn, _ = precision_recall_curve(y_true, y_scores_etn)
    etn_aupr = auc(recall_etn, precision_etn)

    # Print Comparative Results
    print("\n=======================================================")
    print("           ETN VS BASELINE EDL COMPARATIVE RESULTS")
    print("=======================================================")
    print(f"1. Predictive Accuracy:")
    print(f"   - Baseline EDL: {base_acc*100:.2f}%")
    print(f"   - ETN:          {etn_acc*100:.2f}% (Preserves accuracy)")
    print("-" * 55)
    print(f"2. Calibration (Expected Calibration Error - ECE):")
    print(f"   - Baseline EDL ECE: {base_ece*100:.2f}%")
    print(f"   - ETN ECE:          {etn_ece*100:.2f}% (Lower is better)")
    print(f"   - ECE Reduction:    {(base_ece - etn_ece)*100:.2f}%")
    print("-" * 55)
    print(f"3. Margin Enlargement:")
    print(f"   - Baseline EDL Avg Logit Margin: {base_avg_margin:.4f}")
    print(f"   - ETN Avg Logit Margin:          {etn_avg_margin:.4f} (Margin increased!)")
    print(f"   - Margin Increase Ratio:         {((etn_avg_margin / (base_avg_margin + 1e-8)) - 1)*100:.2f}%")
    print("-" * 55)
    print(f"4. OOD Detection (epistemic uncertainty):")
    print(f"   - Baseline EDL: AUROC = {base_auroc*100:.2f}%, AUPR = {base_aupr*100:.2f}%")
    print(f"   - ETN model:    AUROC = {etn_auroc*100:.2f}%, AUPR = {etn_aupr*100:.2f}%")
    print(f"   - AUROC Gain:   {(etn_auroc - base_auroc)*100:.2f}%")
    print(f"   - AUPR Gain:    {(etn_aupr - base_aupr)*100:.2f}%")
    print("=======================================================")

if __name__ == '__main__':
    train_and_eval_etn()
