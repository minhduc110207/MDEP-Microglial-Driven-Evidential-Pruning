import torch
import math
from mdep_agents import MDEPLinear, MDEPConv2d, update_scores_agents
from edl_core import compute_uncertainties

class MDEPTrainer:
    def __init__(self, model, optimizer, criterion, total_epochs, warmup_epochs=15):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
        # User requested parameters for gamma
        self.gamma_initial = 5.0
        self.gamma_final = 0.05
        
        # AMP Scaler for Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler()
        
    def step_gamma(self, epoch):
        if epoch < self.warmup_epochs:
            return self.gamma_initial
        # Cosine annealing for Smoothed STE temperature
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        gamma = self.gamma_final + 0.5 * (self.gamma_initial - self.gamma_final) * (1 + math.cos(math.pi * progress))
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
        num_batches = len(dataloader)
                
        total_loss = 0
        total_grad_norm = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Smooth per-batch LR Warmup
            if epoch < warmup_period:
                current_step = epoch * num_batches + batch_idx
                total_warmup_steps = warmup_period * num_batches
                current_lr = 1e-6 + (base_lr - 1e-6) * (current_step / total_warmup_steps)
                
                # Linear decay for Loss Scaling from 4.0 to 1.0 to prevent overshooting
                current_loss_scale = 4.0 - 3.0 * (current_step / total_warmup_steps)
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                current_lr = base_lr
                current_loss_scale = 1.0
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr

            inputs, targets = inputs.to(device), targets.to(device)
            
            # Amortized evaluation on first batch
            if not is_warmup and batch_idx == 0:
                self.compute_amortized_gradients(inputs)
                
            self.optimizer.zero_grad()
            
            # Use Automatic Mixed Precision for Forward Pass
            with torch.cuda.amp.autocast():
                evidence = self.model(inputs)
                # Ensure Evidential Loss runs in FP32 to avoid underflow
                loss = self.criterion(evidence.float(), targets, epoch)
            
            # Loss scaling to counteract Focal Loss shrinkage (decayed)
            scaled_loss = loss * current_loss_scale
            
            self.scaler.scale(scaled_loss).backward()
            
            # Gradient clipping and norm tracking
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            total_grad_norm += grad_norm.item()
            
            # Store primary gradient for structural updates
            if not is_warmup:
                for module in self.model.modules():
                    if isinstance(module, (MDEPLinear, MDEPConv2d)) and module.weight.grad is not None:
                        module.grad_L_w = module.weight.grad.clone().detach()
                        
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Perform multi-agent structure optimization
            if not is_warmup and batch_idx == 0:
                update_scores_agents(self.model)
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        avg_grad = total_grad_norm / len(dataloader)
        print(f"    Epoch {epoch} | LR: {current_lr:.2e} | GradNorm: {avg_grad:.4f}")
        return avg_loss
