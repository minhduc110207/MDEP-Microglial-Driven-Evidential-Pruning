import torch
import torch.nn as nn
from mdep_notebook import (
    MDEPLinear, MDEPConv2d, EvidentialFocalLoss, EvidenceLayer, replace_conv2d_with_mdep
)
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
num_classes = 2
model = models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, num_classes),
    EvidenceLayer(activation='softplus'),
)
nn.init.normal_(model.fc[0].weight, mean=0, std=0.001)
nn.init.constant_(model.fc[0].bias, 0)
replace_conv2d_with_mdep(model)
model = model.to(device)
model.train()

# Data
inputs = torch.randn(8, 3, 224, 224).to(device)
targets = torch.randint(0, 2, (8,)).to(device)

criterion = EvidentialFocalLoss(gamma=2.0, num_classes=num_classes, kl_lambda=0.1).to(device)

trainable_params = [p for name, p in model.named_parameters() if 'scores' not in name]
optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

scaler = torch.cuda.amp.GradScaler()

optimizer.zero_grad()
with torch.cuda.amp.autocast():
    evidence = model(inputs)
    loss = criterion(evidence.float(), targets, 0)
    scaled_loss = loss * 4.0

scaler.scale(scaled_loss).backward()
scaler.unscale_(optimizer)

grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
print("Loss:", loss.item())
print("Grad Norm:", grad_norm.item())

# Find NaN gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN grad in: {name}")
