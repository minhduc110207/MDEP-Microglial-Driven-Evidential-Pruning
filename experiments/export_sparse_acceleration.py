"""
Realized 2:4 Sparse Tensor Core Acceleration Profiler & Exporter.

This script demonstrates Level 3 of the Hardware Evidence Ladder: Realized Sparse Acceleration.
It provides two paths:
1. Native PyTorch Semi-Structured Sparsity (for nn.Linear layers).
2. ONNX Export with frozen 2:4 masks (for full-model TensorRT sparse execution).

Requirements:
- GPU: NVIDIA Ampere architecture or newer (e.g., RTX 30-series, A100, H100).
- Software: PyTorch 2.1+ (for native) or TensorRT 8.6+ (for ONNX path).
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import MDEPConv2d, MDEPLinear, generate_2_4_mask
from experiments.generalization_paper_suite import EvidenceResNet

def verify_ampere_gpu() -> bool:
    if not torch.cuda.is_available():
        return False
    # Compute Capability >= 8.0 corresponds to Ampere+
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8

def export_to_semi_structured(model: nn.Module, device: torch.device) -> tuple[nn.Module, dict[str, int]]:
    """
    Converts MDEP layers to standard PyTorch layers and applies
    torch.sparse.to_sparse_semi_structured to nn.Linear layers.
    Conv2d layers are kept dense (with mask applied) because PyTorch
    does not yet natively support to_sparse_semi_structured on 4D tensors.
    """
    from torch.sparse import to_sparse_semi_structured
    
    new_model = copy.deepcopy(model).to(device).half()
    
    stats = {
        "linear_converted": 0,
        "linear_failed": 0,
        "conv_dense_fallback": 0
    }

    def replace_module(parent: nn.Module):
        for name, child in parent.named_children():
            if isinstance(child, MDEPLinear):
                # 1. Update mask
                child.mask.copy_(generate_2_4_mask(child.scores.data))
                # 2. Compute effective deployed weight
                dense_weight = (child.weight * child.mask).to(device).half()
                
                # 3. Create standard nn.Linear replacement
                new_layer = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                new_layer = new_layer.to(device).half()
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data.to(device).half())
                    
                try:
                    # 4. Compress to 2:4 Sparse Tensor format
                    sparse_weight = to_sparse_semi_structured(dense_weight)
                    new_layer.weight = nn.Parameter(sparse_weight, requires_grad=False)
                    setattr(parent, name, new_layer)
                    stats["linear_converted"] += 1
                except Exception as e:
                    print(f"Failed to compress Linear layer {name}: {e}")
                    new_layer.weight.data.copy_(dense_weight)
                    setattr(parent, name, new_layer)
                    stats["linear_failed"] += 1
                    
            elif isinstance(child, MDEPConv2d):
                # Fallback for Conv2d: apply mask mathematically but keep dense format
                child.mask.copy_(generate_2_4_mask(child.scores.data))
                dense_weight = (child.weight * child.mask).to(device).half()
                
                new_layer = nn.Conv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding, dilation=child.dilation,
                    groups=child.groups, bias=child.bias is not None
                )
                new_layer = new_layer.to(device).half()
                new_layer.weight.data.copy_(dense_weight)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data.to(device).half())
                    
                setattr(parent, name, new_layer)
                stats["conv_dense_fallback"] += 1
            else:
                replace_module(child)

    replace_module(new_model)
    return new_model, stats


def convert_to_standard_dense_masked(model: nn.Module) -> nn.Module:
    """
    Converts MDEP layers to standard PyTorch nn.Linear and nn.Conv2d layers,
    freezing the 2:4 masks and applying them mathematically to the weights.
    This creates a standard, clean PyTorch model (no control flow, no custom layers)
    that can be exported to ONNX on any machine (even CPU).
    """
    new_model = copy.deepcopy(model).cpu()
    
    def replace_module(parent: nn.Module):
        for name, child in parent.named_children():
            if isinstance(child, MDEPLinear):
                # Ensure mask is updated
                child.mask.copy_(generate_2_4_mask(child.scores.data))
                dense_weight = (child.weight * child.mask).detach().clone()
                
                new_layer = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                new_layer.weight.data.copy_(dense_weight)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data.detach().clone())
                setattr(parent, name, new_layer)
                
            elif isinstance(child, MDEPConv2d):
                child.mask.copy_(generate_2_4_mask(child.scores.data))
                dense_weight = (child.weight * child.mask).detach().clone()
                
                new_layer = nn.Conv2d(
                    child.in_channels, child.out_channels, child.kernel_size,
                    stride=child.stride, padding=child.padding, dilation=child.dilation,
                    groups=child.groups, bias=child.bias is not None
                )
                new_layer.weight.data.copy_(dense_weight)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data.detach().clone())
                setattr(parent, name, new_layer)
            else:
                replace_module(child)

    replace_module(new_model)
    return new_model


def export_to_onnx(model: nn.Module, save_path: str, img_size: int = 224):
    """
    Exports a clean standard PyTorch model (no MDEP layers) to ONNX.
    This ONNX file can be fed to TensorRT's trtexec with --sparsity=force
    to accelerate both Conv2d and Linear layers.
    """
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path, 
        export_params=True,
        opset_version=14, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported to ONNX: {save_path}")
    print("To benchmark with TensorRT, run:")
    print(f"trtexec --onnx={save_path} --fp16 --sparsity=force --shapes=input:128x3x{img_size}x{img_size}")


def benchmark_throughput(model: nn.Module, device: torch.device, batch_size: int, img_size: int, warmup: int = 20, iters: int = 50) -> dict[str, float]:
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device).half()
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    end_event.record()
    torch.cuda.synchronize(device)
    
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / iters
    ips = (batch_size * iters) / (elapsed_ms / 1000.0)
    
    return {"ms_per_batch": avg_ms, "images_per_sec": ips}


def main():
    parser = argparse.ArgumentParser(description="Realized 2:4 Sparse Tensor Core Benchmark")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--onnx_path", type=str, default="dst_edl_resnet18_sparse.onnx", help="Path to save the ONNX model")
    args = parser.parse_args()
    
    print("=== Level 3 Hardware Evidence: Realized Sparse Acceleration ===")
    
    # 1. Build Original MDEP Model
    print("\nBuilding MDEP ResNet-18 model...")
    from guds_edl_core import replace_conv2d_with_mdep
    model_mdep = EvidenceResNet(num_classes=2, dataset="isic", pretrained=False)
    replace_conv2d_with_mdep(model_mdep)
    
    # 2. Convert to standard dense masked model for clean ONNX export (no custom modules/control flow)
    print("Converting MDEP model to clean standard model for ONNX export...")
    model_clean = convert_to_standard_dense_masked(model_mdep)
    
    # 3. ONNX Export for TensorRT
    export_to_onnx(model_clean, args.onnx_path, args.image_size)
    
    # 4. Check for PyTorch Native Support
    if not verify_ampere_gpu():
        print("\n[WARNING] No Ampere+ GPU detected. Skipping PyTorch semi-structured dispatch.")
        print("To test realized PyTorch speedup, please run this script on an RTX 3090, A100, H100, or similar.")
        return
        
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nRunning PyTorch benchmark on GPU: {gpu_name} (Ampere+ compliant)")
    
    model_mdep = model_mdep.to(device)
    
    # 5. Build Dense Baseline
    print("Building standard dense baseline...")
    model_dense = EvidenceResNet(num_classes=2, dataset="isic", pretrained=False).to(device).half()
    
    # 6. Export to PyTorch Semi-Structured
    print("Converting MDEP model to PyTorch Semi-Structured format (nn.Linear only)...")
    try:
        model_sparse, stats = export_to_semi_structured(model_mdep, device)
        print(f"Conversion stats: {stats}")
    except Exception as e:
        print(f"Export failed: {e}")
        return

    # 7. Benchmark
    print(f"\nBenchmarking Dense Baseline (batch_size={args.batch_size})...")
    dense_metrics = benchmark_throughput(model_dense, device, args.batch_size, args.image_size)
    print(f"Dense Baseline: {dense_metrics['ms_per_batch']:.2f} ms/batch | {dense_metrics['images_per_sec']:.2f} img/s")
    
    print(f"\nBenchmarking Sparse Model (batch_size={args.batch_size})...")
    sparse_metrics = benchmark_throughput(model_sparse, device, args.batch_size, args.image_size)
    print(f"Sparse Model  : {sparse_metrics['ms_per_batch']:.2f} ms/batch | {sparse_metrics['images_per_sec']:.2f} img/s")
    
    speedup = dense_metrics['ms_per_batch'] / max(sparse_metrics['ms_per_batch'], 1e-8)
    print(f"\nSpeedup: {speedup:.2f}x")
    if speedup < 1.05:
        print("\nNote: ResNet-18 is heavily Convolution-bound. PyTorch native semi-structured sparsity only accelerates the final Linear layer.")
        print("To observe whole-model acceleration (including Conv2d), use the generated ONNX file with TensorRT's trtexec --sparsity=force.")
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
