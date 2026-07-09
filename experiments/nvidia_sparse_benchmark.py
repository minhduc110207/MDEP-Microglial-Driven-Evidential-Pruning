"""
Reproducible NVIDIA TensorRT Level-3 benchmark for the ISIC fair-v2 models.

This runner is deliberately separate from ``hardware_profile.py``:

* ``hardware_profile.py`` reports structural feasibility and masked-PyTorch cost.
* this file freezes trained checkpoints, exports deployment graphs, invokes
  NVIDIA ``trtexec``, and reports realized TensorRT latency/throughput.

Two comparisons are kept separate:

1. Network comparison:
   Dense EDL (dense TensorRT) vs Static 2:4, RigL-style 2:4, and DST-EDL
   (TensorRT sparsity enabled).
2. Kernel ablation:
   the same frozen DST-EDL graph with TensorRT sparsity disabled vs enabled.

No paper-ready LaTeX table is emitted unless checkpoint, graph-equivalence,
RTX-A2000, TensorRT, and sparse-build evidence gates all pass.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guds_edl_core import MDEPConv2d, MDEPLinear, replace_conv2d_with_mdep  # noqa: E402
from experiments.hardware_profile import structural_stats  # noqa: E402
from experiments.isic_paper_experiments import (  # noqa: E402
    EXPERIMENTS,
    PROTOCOL_VERSION,
    ResNetEvidenceModel,
    json_safe,
)


MODEL_NAMES = ("dense_edl", "static_24_edl", "rigl_style_24", "full_guds")
SPARSE_MODEL_NAMES = frozenset({"static_24_edl", "rigl_style_24", "full_guds"})
DISPLAY_NAMES = {
    "dense_edl": "Dense EDL",
    "static_24_edl": "Static 2:4 EDL",
    "rigl_style_24": "RigL-style 2:4",
    "full_guds": "DST-EDL",
}


def run_command(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
    log_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("$", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    completed = subprocess.run(
        [str(x) for x in command],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    print(completed.stdout, flush=True)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}. "
            f"See {log_path if log_path else 'captured output'}."
        )
    return completed


def default_output_root() -> Path:
    return REPO_ROOT / "paper_experiment_outputs" / "hardware_nvidia_rtx_a2000"


def default_checkpoint(model_name: str, seed: int) -> Path:
    return (
        REPO_ROOT
        / "paper_experiment_outputs"
        / "isic"
        / f"{model_name}_fair_v2"
        / f"seed_{seed}"
        / "model_state.pth"
    )


def parse_checkpoint_overrides(values: list[str], seed: int) -> dict[str, Path]:
    result = {name: default_checkpoint(name, seed) for name in MODEL_NAMES}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--checkpoint expects MODEL=PATH, got: {value!r}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if name not in MODEL_NAMES:
            raise ValueError(f"Unknown checkpoint model {name!r}; expected one of {MODEL_NAMES}")
        result[name] = Path(raw_path.strip()).expanduser().resolve()
    return result


def checkpoint_protocol(checkpoint: Path, allow_legacy: bool) -> str | None:
    metrics_path = checkpoint.parent / "metrics.json"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}\n"
            "Hardware comparison requires saved trained checkpoints. Re-run the corresponding "
            "fair-v2 experiment without --no_save_model, or pass --checkpoint MODEL=PATH."
        )
    found = None
    if metrics_path.exists():
        try:
            found = json.loads(metrics_path.read_text(encoding="utf-8")).get("protocol_version")
        except Exception as exc:
            raise RuntimeError(f"Could not read checkpoint metadata {metrics_path}: {exc}") from exc
    if found != PROTOCOL_VERSION and not allow_legacy:
        raise RuntimeError(
            f"{checkpoint} has protocol_version={found!r}; expected {PROTOCOL_VERSION!r}. "
            "Use --allow_legacy_checkpoint only for diagnostic runs; such runs are not paper-ready."
        )
    return found


def clean_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict) and "model_state_dict" in raw:
        raw = raw["model_state_dict"]
    if not isinstance(raw, dict):
        raise TypeError("Checkpoint must contain a state_dict mapping.")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        key = str(key)
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_trained_model(model_name: str, checkpoint: Path) -> nn.Module:
    spec = EXPERIMENTS[model_name]
    model = ResNetEvidenceModel(num_classes=2, flexible=False, pretrained=False)
    if spec.sparse:
        # Match training exactly: sparse backbone only; RGB stem and classifier stay dense.
        replace_conv2d_with_mdep(model.backbone, learn_permutation=False)
    state = clean_state_dict(torch.load(checkpoint, map_location="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_unexpected = [key for key in unexpected if key.startswith("ood_projection_head.")]
    real_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if missing or real_unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {model_name}: missing={missing}, "
            f"unexpected={real_unexpected}"
        )
    if spec.sparse:
        # ``warmup`` is runtime state rather than a state_dict entry. A freshly
        # reconstructed MDEP module defaults to dense warmup=True, so explicitly
        # restore sparse inference before validating or freezing the checkpoint.
        for module in model.modules():
            if isinstance(module, (MDEPConv2d, MDEPLinear)):
                module.warmup = False
    model.eval()
    return model


def assert_identity_permutations(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if not isinstance(module, (MDEPConv2d, MDEPLinear)):
            continue
        expected = torch.arange(module.perm_indices.numel(), dtype=torch.long)
        actual = module.perm_indices.detach().cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(
                f"{name} uses a non-identity channel permutation. The paper fair-v2 deployment "
                "protocol requires frozen identity order; refusing to export a mismatched graph."
            )


def freeze_mdep_for_deployment(model: nn.Module) -> nn.Module:
    """Bake each learned mask once and remove all MDEP controller overhead."""
    frozen = copy.deepcopy(model).cpu().eval()
    assert_identity_permutations(frozen)

    def replace_children(parent: nn.Module) -> None:
        for child_name, child in list(parent.named_children()):
            if isinstance(child, MDEPConv2d):
                layer = nn.Conv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode,
                )
                layer.weight.data.copy_((child.weight.detach() * child.mask.detach()).cpu())
                if child.bias is not None:
                    layer.bias.data.copy_(child.bias.detach().cpu())
                setattr(parent, child_name, layer)
            elif isinstance(child, MDEPLinear):
                layer = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                layer.weight.data.copy_((child.weight.detach() * child.mask.detach()).cpu())
                if child.bias is not None:
                    layer.bias.data.copy_(child.bias.detach().cpu())
                setattr(parent, child_name, layer)
            else:
                replace_children(child)

    replace_children(frozen)
    return frozen.eval()


@torch.inference_mode()
def verify_frozen_equivalence(
    original: nn.Module,
    frozen: nn.Module,
    image_size: int,
    atol: float,
    rtol: float,
) -> dict[str, float | bool]:
    torch.manual_seed(20260709)
    x = torch.randn(2, 3, image_size, image_size)
    original = original.cpu().float().eval()
    frozen = frozen.cpu().float().eval()
    reference = original(x)
    candidate = frozen(x)
    difference = (reference - candidate).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    passed = bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol))
    return {
        "passed": passed,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def export_onnx(
    model: nn.Module,
    output_path: Path,
    image_size: int,
    opset: int,
) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX export requires `pip install onnx`.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().float().eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    graph = onnx.load(str(output_path))
    onnx.checker.check_model(graph)


def query_gpu_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_vram_mb": float(props.total_memory / (1024**2)),
            }
        )
    smi = shutil.which("nvidia-smi")
    if smi:
        query = [
            smi,
            "--query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,"
            "clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(query, text=True, capture_output=True, check=False)
        metadata["nvidia_smi_snapshot"] = completed.stdout.strip()
    return metadata


def verify_rtx_a2000(metadata: dict[str, Any], allow_other_gpu: bool) -> bool:
    name = str(metadata.get("gpu_name", ""))
    is_a2000 = "RTX A2000" in name.upper()
    if not is_a2000 and not allow_other_gpu:
        raise RuntimeError(
            f"Detected GPU {name!r}, not an NVIDIA RTX A2000. "
            "Use --allow_other_gpu only for diagnostic runs; those results are not the A2000 table."
        )
    capability = metadata.get("compute_capability", [0, 0])
    if capability[0] < 8:
        raise RuntimeError(f"Compute capability {capability} is not Ampere-or-newer.")
    return is_a2000


def find_trtexec(requested: str | None) -> Path:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested))
    found = shutil.which("trtexec")
    if found:
        candidates.append(Path(found))
    root = os.environ.get("TENSORRT_ROOT")
    if root:
        candidates.extend([Path(root) / "bin" / "trtexec.exe", Path(root) / "bin" / "trtexec"])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find NVIDIA trtexec. Pass --trtexec PATH or set TENSORRT_ROOT "
        "to the extracted TensorRT installation."
    )


def trtexec_version(trtexec: Path) -> str:
    completed = subprocess.run(
        [str(trtexec), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def parse_trtexec_output(text: str, batch_size: int) -> dict[str, float]:
    patterns = {
        "queries_per_second": r"Throughput:\s*([0-9.eE+-]+)\s*qps",
        "latency_min_ms": r"Latency:\s*min\s*=\s*([0-9.eE+-]+)\s*ms",
        "latency_max_ms": r"Latency:.*?max\s*=\s*([0-9.eE+-]+)\s*ms",
        "latency_mean_ms": r"Latency:.*?mean\s*=\s*([0-9.eE+-]+)\s*ms",
        "latency_median_ms": r"Latency:.*?median\s*=\s*([0-9.eE+-]+)\s*ms",
        "latency_p95_ms": r"Latency:.*?percentile\(95%\)\s*=\s*([0-9.eE+-]+)\s*ms",
        "gpu_compute_mean_ms": r"GPU Compute Time:.*?mean\s*=\s*([0-9.eE+-]+)\s*ms",
    }
    parsed: dict[str, float] = {}
    one_line = " ".join(text.splitlines())
    for key, pattern in patterns.items():
        match = re.search(pattern, one_line, flags=re.IGNORECASE)
        if match:
            parsed[key] = float(match.group(1))
    if "queries_per_second" not in parsed or "latency_mean_ms" not in parsed:
        raise RuntimeError("Could not parse TensorRT throughput/latency from trtexec output.")
    parsed["images_per_second"] = parsed["queries_per_second"] * batch_size
    return parsed


def sparse_build_evidence(text: str) -> dict[str, Any]:
    evidence_lines = []
    for line in text.splitlines():
        low = line.lower()
        if "spars" in low and any(token in low for token in ("weight", "tactic", "kernel", "eligible")):
            evidence_lines.append(line.strip())
    eligible_counts = [
        int(value)
        for value in re.findall(
            r"Found\s+(\d+)\s+layer\(s\)\s+eligible\s+to\s+use\s+sparse\s+tactics",
            text,
            flags=re.IGNORECASE,
        )
    ]
    chosen_counts = [
        int(value)
        for value in re.findall(
            r"Chose\s+(\d+)\s+layer\(s\)\s+using\s+sparse\s+tactics",
            text,
            flags=re.IGNORECASE,
        )
    ]
    older_picked = bool(
        re.search(
            r"(picked\s+sparse\s+implementation|selected.*sparse.*tactic)",
            text,
            flags=re.IGNORECASE,
        )
    )
    eligible_layers = max(eligible_counts, default=0)
    chosen_layers = max(chosen_counts, default=(1 if older_picked else 0))
    return {
        "verified": chosen_layers > 0,
        "eligible_layers": eligible_layers,
        "chosen_sparse_layers": chosen_layers,
        "lines": evidence_lines[:100],
        "note": (
            "Verified requires the verbose TensorRT build log to report at least one "
            "layer chosen with a sparse tactic. Eligibility and --sparsity=enable alone "
            "are not treated as proof."
        ),
    }


def build_engine(
    trtexec: Path,
    onnx_path: Path,
    engine_path: Path,
    *,
    batch_size: int,
    image_size: int,
    sparse: bool,
    workspace_mb: int,
    log_path: Path,
) -> dict[str, Any]:
    shape = f"input:{batch_size}x3x{image_size}x{image_size}"
    command = [
        str(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--sparsity={'enable' if sparse else 'disable'}",
        f"--minShapes={shape}",
        f"--optShapes={shape}",
        f"--maxShapes={shape}",
        f"--memPoolSize=workspace:{workspace_mb}",
        "--profilingVerbosity=detailed",
        "--verbose",
        "--skipInference",
    ]
    completed = run_command(command, log_path=log_path)
    evidence = sparse_build_evidence(completed.stdout) if sparse else {
        "verified": True,
        "lines": [],
        "note": "Dense build; sparse-tactic evidence is not applicable.",
    }
    if not engine_path.exists():
        raise RuntimeError(f"TensorRT reported success but did not create {engine_path}")
    return evidence


def benchmark_engine_once(
    trtexec: Path,
    engine_path: Path,
    *,
    batch_size: int,
    image_size: int,
    warmup_ms: int,
    duration_s: int,
    log_path: Path,
) -> dict[str, float]:
    shape = f"input:{batch_size}x3x{image_size}x{image_size}"
    command = [
        str(trtexec),
        f"--loadEngine={engine_path}",
        f"--shapes={shape}",
        f"--warmUp={warmup_ms}",
        f"--duration={duration_s}",
        "--useCudaGraph",
        "--noDataTransfers",
        "--useSpinWait",
    ]
    completed = run_command(command, log_path=log_path)
    return parse_trtexec_output(completed.stdout, batch_size)


def aggregate_repeats(repeats: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    keys = sorted(set().union(*(item.keys() for item in repeats)))
    for key in keys:
        values = [item[key] for item in repeats if key in item and math.isfinite(item[key])]
        if not values:
            continue
        result[f"{key}_mean"] = float(statistics.mean(values))
        result[f"{key}_median"] = float(statistics.median(values))
        result[f"{key}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return result


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex_table(rows: list[dict[str, Any]], path: Path, seed: int) -> None:
    network_rows = [row for row in rows if row["comparison"] == "network"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{TensorRT FP16 inference on an NVIDIA RTX A2000. Results use trained fair-v2 "
        r"checkpoints, CUDA Graph execution, disabled host--device transfers, and repeated timed runs.}",
        r"\label{tab:rtx_a2000_level3}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & TensorRT path & Batch & Median img/s & Mean latency (ms) & p95 latency (ms) \\",
        r"\midrule",
    ]
    for row in sorted(network_rows, key=lambda item: (int(item["batch_size"]), MODEL_NAMES.index(item["model"]))):
        lines.append(
            f"{latex_escape(DISPLAY_NAMES[row['model']])} & "
            f"{'Sparse FP16' if row['sparsity_enabled'] else 'Dense FP16'} & "
            f"{int(row['batch_size'])} & "
            f"{row['images_per_second_median']:.1f} & "
            f"{row['latency_mean_ms_median']:.3f} & "
            f"{row.get('latency_p95_ms_median', float('nan')):.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"% fair-v2 checkpoint seed: {seed}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA TensorRT Level-3 benchmark on RTX A2000.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--trtexec", help="Path to NVIDIA TensorRT trtexec(.exe).")
    parser.add_argument("--output_dir", type=Path, default=default_output_root())
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup_ms", type=int, default=2000)
    parser.add_argument("--duration_s", type=int, default=10)
    parser.add_argument("--workspace_mb", type=int, default=2048)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--export_only", action="store_true")
    parser.add_argument("--allow_other_gpu", action="store_true")
    parser.add_argument("--allow_legacy_checkpoint", action="store_true")
    parser.add_argument("--equivalence_atol", type=float, default=1e-5)
    parser.add_argument("--equivalence_rtol", type=float, default=1e-4)
    args = parser.parse_args()

    if args.repeats < 3:
        raise ValueError("Use at least 3 repeats; 5 or more is recommended for paper reporting.")
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("All batch sizes must be positive.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir.resolve() / f"seed_{args.seed}_{timestamp}"
    onnx_dir = run_dir / "onnx"
    engine_dir = run_dir / "engines"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=False)

    checkpoints = parse_checkpoint_overrides(args.checkpoint, args.seed)
    checkpoint_records: dict[str, Any] = {}
    frozen_models: dict[str, nn.Module] = {}
    for name in MODEL_NAMES:
        protocol = checkpoint_protocol(checkpoints[name], args.allow_legacy_checkpoint)
        trained = load_trained_model(name, checkpoints[name])
        stats = structural_stats(trained)
        if name in SPARSE_MODEL_NAMES and stats["valid_24_block_fraction"] != 1.0:
            raise RuntimeError(
                f"{name} checkpoint is not exact 2:4: "
                f"valid_fraction={stats['valid_24_block_fraction']}"
            )
        frozen = freeze_mdep_for_deployment(trained)
        equivalence = verify_frozen_equivalence(
            trained,
            frozen,
            args.image_size,
            args.equivalence_atol,
            args.equivalence_rtol,
        )
        if not equivalence["passed"]:
            raise RuntimeError(f"Frozen graph equivalence failed for {name}: {equivalence}")
        onnx_path = onnx_dir / f"{name}_seed{args.seed}.onnx"
        export_onnx(frozen, onnx_path, args.image_size, args.opset)
        frozen_models[name] = frozen
        checkpoint_records[name] = {
            "path": str(checkpoints[name]),
            "protocol_version": protocol,
            "structural_stats": stats,
            "frozen_equivalence": equivalence,
            "onnx_path": str(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
        }

    metadata = query_gpu_metadata()
    metadata["is_rtx_a2000"] = False
    metadata["trtexec_version"] = None
    if args.export_only:
        manifest = {
            "status": "export_only",
            "paper_ready": False,
            "protocol_version": PROTOCOL_VERSION,
            "seed": args.seed,
            "checkpoints": checkpoint_records,
            "runtime": metadata,
        }
        (run_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
        print(f"Export complete: {run_dir}")
        return 0

    metadata["is_rtx_a2000"] = verify_rtx_a2000(metadata, args.allow_other_gpu)
    trtexec = find_trtexec(args.trtexec)
    metadata["trtexec"] = str(trtexec)
    metadata["trtexec_version"] = trtexec_version(trtexec)

    # Build each unique engine once. Network comparison uses sparse TensorRT for
    # sparse models. DST-EDL is additionally built dense for the kernel ablation.
    engine_specs: list[dict[str, Any]] = []
    for batch_size in sorted(set(args.batch_sizes)):
        for model_name in MODEL_NAMES:
            engine_specs.append(
                {
                    "comparison": "network",
                    "model": model_name,
                    "batch_size": batch_size,
                    "sparsity_enabled": model_name in SPARSE_MODEL_NAMES,
                }
            )
        engine_specs.append(
            {
                "comparison": "kernel_ablation",
                "model": "full_guds",
                "batch_size": batch_size,
                "sparsity_enabled": False,
            }
        )
        engine_specs.append(
            {
                "comparison": "kernel_ablation",
                "model": "full_guds",
                "batch_size": batch_size,
                "sparsity_enabled": True,
            }
        )

    unique_engines: dict[tuple[str, int, bool], dict[str, Any]] = {}
    for spec in engine_specs:
        key = (spec["model"], spec["batch_size"], spec["sparsity_enabled"])
        if key in unique_engines:
            continue
        model_name, batch_size, sparse = key
        tag = f"{model_name}_b{batch_size}_{'sparse' if sparse else 'dense'}"
        engine_path = engine_dir / f"{tag}.engine"
        build_log = log_dir / f"build_{tag}.log"
        evidence = build_engine(
            trtexec,
            Path(checkpoint_records[model_name]["onnx_path"]),
            engine_path,
            batch_size=batch_size,
            image_size=args.image_size,
            sparse=sparse,
            workspace_mb=args.workspace_mb,
            log_path=build_log,
        )
        unique_engines[key] = {
            "engine_path": engine_path,
            "build_log": build_log,
            "sparse_build_evidence": evidence,
        }

    # Randomize execution order in every repeat to reduce order/thermal bias.
    raw_results: list[dict[str, Any]] = []
    rng = random.Random(20260709)
    for repeat in range(1, args.repeats + 1):
        order = list(unique_engines.items())
        rng.shuffle(order)
        for (model_name, batch_size, sparse), engine_record in order:
            tag = f"{model_name}_b{batch_size}_{'sparse' if sparse else 'dense'}"
            log_path = log_dir / f"run_{tag}_repeat{repeat}.log"
            metrics = benchmark_engine_once(
                trtexec,
                engine_record["engine_path"],
                batch_size=batch_size,
                image_size=args.image_size,
                warmup_ms=args.warmup_ms,
                duration_s=args.duration_s,
                log_path=log_path,
            )
            raw_results.append(
                {
                    "model": model_name,
                    "batch_size": batch_size,
                    "sparsity_enabled": sparse,
                    "repeat": repeat,
                    **metrics,
                }
            )

    rows: list[dict[str, Any]] = []
    for spec in engine_specs:
        model_name = spec["model"]
        batch_size = spec["batch_size"]
        sparse = spec["sparsity_enabled"]
        samples = [
            {
                key: value
                for key, value in item.items()
                if isinstance(value, (int, float)) and key not in {"batch_size", "repeat"}
            }
            for item in raw_results
            if item["model"] == model_name
            and item["batch_size"] == batch_size
            and item["sparsity_enabled"] == sparse
        ]
        aggregate = aggregate_repeats(samples)
        engine_record = unique_engines[(model_name, batch_size, sparse)]
        row = {
            **spec,
            **aggregate,
            "display_name": DISPLAY_NAMES[model_name],
            "sparse_tactic_verified": engine_record["sparse_build_evidence"]["verified"],
        }
        rows.append(row)

    # Deduplicate because DST sparse/dense engines appear in both comparisons.
    unique_rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (
            row["comparison"],
            row["model"],
            row["batch_size"],
            row["sparsity_enabled"],
        )
        if key not in seen:
            unique_rows.append(row)
            seen.add(key)
    rows = unique_rows

    all_sparse_verified = all(
        bool(record["sparse_build_evidence"]["verified"])
        for (model_name, _batch, sparse), record in unique_engines.items()
        if sparse and model_name in SPARSE_MODEL_NAMES
    )
    checkpoints_current = all(
        record["protocol_version"] == PROTOCOL_VERSION for record in checkpoint_records.values()
    )
    paper_ready = bool(
        metadata["is_rtx_a2000"]
        and checkpoints_current
        and all_sparse_verified
        and len(raw_results) == len(unique_engines) * args.repeats
    )

    manifest = {
        "status": "complete",
        "paper_ready": paper_ready,
        "quality_gates": {
            "rtx_a2000_verified": metadata["is_rtx_a2000"],
            "fair_v2_checkpoints": checkpoints_current,
            "frozen_graph_equivalence": all(
                record["frozen_equivalence"]["passed"] for record in checkpoint_records.values()
            ),
            "all_sparse_builds_have_log_evidence": all_sparse_verified,
            "all_repeats_complete": len(raw_results) == len(unique_engines) * args.repeats,
        },
        "protocol_version": PROTOCOL_VERSION,
        "seed": args.seed,
        "settings": {
            "batch_sizes": sorted(set(args.batch_sizes)),
            "image_size": args.image_size,
            "precision": "TensorRT FP16",
            "repeats": args.repeats,
            "warmup_ms": args.warmup_ms,
            "duration_s": args.duration_s,
            "workspace_mb": args.workspace_mb,
            "cuda_graph": True,
            "host_device_transfers": False,
            "spin_wait": True,
        },
        "runtime": metadata,
        "checkpoints": checkpoint_records,
        "engines": {
            f"{model}_b{batch}_{'sparse' if sparse else 'dense'}": {
                "engine_path": str(record["engine_path"]),
                "build_log": str(record["build_log"]),
                "sparse_build_evidence": record["sparse_build_evidence"],
            }
            for (model, batch, sparse), record in unique_engines.items()
        },
        "summary": rows,
        "raw_results": raw_results,
        "reporting_scope": {
            "network_comparison": (
                "Compares separately trained fair-v2 networks under their intended TensorRT path."
            ),
            "kernel_ablation": (
                "Compares the same frozen DST-EDL graph with TensorRT sparsity disabled/enabled; "
                "this is the clean evidence for realized sparse-kernel speedup."
            ),
        },
    }
    (run_dir / "results.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    write_csv(rows, run_dir / "summary.csv")
    write_csv(raw_results, run_dir / "raw_repeats.csv")
    if paper_ready:
        write_latex_table(rows, run_dir / "paper_table.tex", args.seed)
        print(f"[PAPER READY] Quality gates passed. LaTeX table: {run_dir / 'paper_table.tex'}")
    else:
        print(
            "[NOT PAPER READY] Results were saved, but at least one quality gate failed. "
            "Inspect results.json and TensorRT build logs; no LaTeX table was generated."
        )
    print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
