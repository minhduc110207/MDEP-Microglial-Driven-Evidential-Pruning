"""
Local full experiment launcher for the GUDS-EDL paper suite.

This runner is the local-machine counterpart of run_kaggle_paper_suite.py. It
does not require Kaggle paths. Point it to a local ISIC dataset with ISIC_ROOT
or pass --isic_root.

Typical smoke test:

    python experiments/run_local_full_experiments.py --smoke --isic_root D:\\datasets\\isic-2024-challenge

Typical full run:

    python experiments/run_local_full_experiments.py ^
      --isic_root D:\\datasets\\isic-2024-challenge ^
      --isic_suite all ^
      --seeds 42 43 44 ^
      --no_save_model ^
      --keep_going

The terminal prints compact progress. Full stdout/stderr for every sub-run is
saved under paper_experiment_outputs/local_logs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "paper_experiment_outputs"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: list[str]


@dataclass(frozen=True)
class CommandResult:
    name: str
    command: list[str]
    log_path: str
    exit_code: int
    elapsed_seconds: float
    status: str


def has_file_tree(root: Path, filename: str, max_depth: int = 3) -> bool:
    if not root.exists():
        return False
    root_depth = len(root.parts)
    for path in root.rglob(filename):
        if len(path.parts) - root_depth <= max_depth:
            return True
    return False


def detect_isic_root(path_arg: str | None) -> Path | None:
    candidates: list[Path] = []
    for value in [path_arg, os.environ.get("ISIC_ROOT"), "data/isic-2024-challenge", "data/isic2024"]:
        if value:
            candidates.append(Path(value).expanduser().resolve())
    for candidate in candidates:
        if (candidate / "train-metadata.csv").exists():
            return candidate
        if has_file_tree(candidate, "train-metadata.csv"):
            return candidate
    return None


def compact_line(line: str) -> bool:
    markers = [
        "[START]",
        "[END]",
        "[RUN]",
        "[TRAIN]",
        "[CAL]",
        "[DONE]",
        "[ERROR]",
        "[WARN]",
        "Traceback",
        "RuntimeError",
        "ValueError",
        "FileNotFoundError",
        "Completed ",
        "Saved summary",
        "Saved hardware profile",
        "All selected",
    ]
    return any(marker in line for marker in markers)


def run_command(spec: CommandSpec, env: dict[str, str], log_dir: Path, stream_mode: str) -> CommandResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec.name}.log"
    start = time.time()
    print("\n" + "=" * 100)
    print(f"[START] {spec.name}")
    print("Command:", " ".join(spec.command))
    print("Log:", log_path)
    print("=" * 100)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            spec.command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            if stream_mode == "full" or (stream_mode == "compact" and compact_line(line)):
                print(line, end="")
    code = process.wait()
    elapsed = time.time() - start
    status = "OK" if code == 0 else f"FAILED exit={code}"
    print("-" * 100)
    print(f"[END] {spec.name} | {status} | elapsed={elapsed / 60:.1f} min")
    print("-" * 100)
    return CommandResult(
        name=spec.name,
        command=spec.command,
        log_path=str(log_path),
        exit_code=code,
        elapsed_seconds=elapsed,
        status=status,
    )


def python_cmd(args: list[str]) -> list[str]:
    return [sys.executable, *args]


def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def git_pull_latest() -> None:
    try:
        before = run_git(["rev-parse", "--short", "HEAD"]).stdout.strip()
        branch = run_git(["branch", "--show-current"]).stdout.strip() or "detached"
    except Exception as exc:
        raise RuntimeError(f"Could not inspect git repository at {REPO_ROOT}: {exc}") from exc

    print("\nGit Update")
    print("-" * 80)
    print(f"Repo   : {REPO_ROOT}")
    print(f"Branch : {branch}")
    print(f"Before : {before}")

    try:
        result = run_git(["pull", "--ff-only"])
    except subprocess.CalledProcessError as exc:
        print(exc.stdout)
        raise RuntimeError(
            "git pull --ff-only failed. Commit/stash local changes first, or resolve "
            "the branch state manually before running experiments."
        ) from exc

    after = run_git(["rev-parse", "--short", "HEAD"]).stdout.strip()
    print(result.stdout.strip())
    print(f"After  : {after}")
    print("-" * 80)


def build_commands(args: argparse.Namespace) -> list[CommandSpec]:
    isic_epochs = 1 if args.smoke else args.epochs
    cifar_epochs = 1 if args.smoke else args.cifar_epochs
    batch_size = min(args.batch_size, 8) if args.smoke else args.batch_size
    seeds = [42] if args.smoke else args.seeds
    commands: list[CommandSpec] = []

    if not args.skip_isic:
        cmd = python_cmd([
            "experiments/isic_paper_experiments.py",
            "--suite",
            args.isic_suite,
            "--epochs",
            str(isic_epochs),
            "--batch_size",
            str(batch_size),
            "--seeds",
            *map(str, seeds),
            "--log_every",
            str(args.log_every),
        ])
        if args.no_save_model:
            cmd.append("--no_save_model")
        if args.allow_dummy_data:
            cmd.append("--allow_dummy_data")
        if args.cpu:
            cmd.append("--cpu")
        if args.verbose_structural_logs:
            cmd.append("--verbose_structural_logs")
        commands.append(CommandSpec(f"isic_{args.isic_suite}", cmd))

    if not args.skip_cifar:
        for ratio in args.cifar_ratios:
            cmd = python_cmd([
                "experiments/generalization_paper_suite.py",
                "--benchmark",
                "cifar",
                "--ratio",
                str(ratio),
                "--epochs",
                str(cifar_epochs),
                "--batch_size",
                str(batch_size),
                "--seeds",
                *map(str, seeds),
                "--log_every",
                str(args.log_every),
            ])
            if args.allow_dummy_data:
                cmd.append("--allow_dummy_data")
            if args.cpu:
                cmd.append("--cpu")
            if args.verbose_structural_logs:
                cmd.append("--verbose_structural_logs")
            commands.append(CommandSpec(f"cifar100lt_ir{ratio}", cmd))

    if not args.skip_hardware:
        cmd = python_cmd([
            "experiments/hardware_profile.py",
            "--batch_size",
            str(min(batch_size, 16)),
            "--iters",
            str(5 if args.smoke else args.hardware_iters),
            "--warmup",
            str(2 if args.smoke else args.hardware_warmup),
        ])
        if args.cpu:
            cmd.append("--cpu")
        commands.append(CommandSpec("hardware_profile", cmd))

    if args.include_backbones:
        cmd = python_cmd([
            "experiments/backbone_generalization_runner.py",
            "--epochs",
            str(1 if args.smoke else args.backbone_epochs),
            "--batch_size",
            str(min(batch_size, 8)),
            "--seeds",
            *map(str, seeds),
        ])
        if args.no_save_model:
            pass
        if args.allow_dummy_data:
            cmd.append("--allow_dummy_data")
        if args.cpu:
            cmd.append("--cpu")
        commands.append(CommandSpec("backbone_generalization", cmd))

    if not args.skip_summary:
        commands.append(CommandSpec("summarize_results", python_cmd([
            "experiments/summarize_results.py",
            "--root",
            str(OUTPUT_ROOT),
        ])))
    return commands


def print_environment_summary(args: argparse.Namespace, isic_root: Path | None) -> None:
    print("\nLocal Experiment Configuration")
    print("-" * 80)
    print(f"Repo root      : {REPO_ROOT}")
    print(f"Python         : {sys.executable}")
    print(f"Output root    : {OUTPUT_ROOT}")
    print(f"ISIC_ROOT      : {isic_root if isic_root else 'not found'}")
    print(f"Seeds          : {[42] if args.smoke else args.seeds}")
    print(f"Mode           : {'smoke' if args.smoke else 'full'}")
    print(f"Stream mode    : {args.stream}")
    print("-" * 80)


def args_to_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
    }


def write_local_run_manifest(
    args: argparse.Namespace,
    isic_root: Path | None,
    commands: list[CommandSpec],
    log_dir: Path,
) -> Path:
    manifest = {
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "output_root": str(OUTPUT_ROOT),
        "log_dir": str(log_dir),
        "isic_root": str(isic_root) if isic_root else None,
        "args": args_to_config(args),
        "commands": [{"name": spec.name, "command": spec.command} for spec in commands],
    }
    manifest_path = log_dir / "local_run_config.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def init_wandb(
    args: argparse.Namespace,
    isic_root: Path | None,
    commands: list[CommandSpec],
    log_dir: Path,
    manifest_path: Path,
):
    if args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb is not installed. Install it with `pip install wandb`, or use --wandb_mode disabled.")
        return None

    wandb_dir = Path(args.wandb_dir).expanduser().resolve() if args.wandb_dir else OUTPUT_ROOT / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = args.wandb_mode
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))

    config = {
        "repo_root": str(REPO_ROOT),
        "python": sys.executable,
        "output_root": str(OUTPUT_ROOT),
        "log_dir": str(log_dir),
        "isic_root": str(isic_root) if isic_root else None,
        "args": args_to_config(args),
        "commands": [{"name": spec.name, "command": " ".join(spec.command)} for spec in commands],
    }
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        dir=str(wandb_dir),
        mode=args.wandb_mode,
        config=config,
        job_type="local_full_experiment",
    )
    wandb.save(str(manifest_path), base_path=str(log_dir))
    print(f"W&B local run : {run.dir}")
    return wandb


def log_wandb_result(wandb_module, result: CommandResult, step: int) -> None:
    if wandb_module is None:
        return
    wandb_module.log(
        {
            "subrun/index": step,
            "subrun/exit_code": result.exit_code,
            "subrun/elapsed_minutes": result.elapsed_seconds / 60.0,
            "subrun/success": int(result.exit_code == 0),
        },
        step=step,
    )
    try:
        wandb_module.save(result.log_path)
    except Exception as exc:
        print(f"[WARN] Could not attach log to W&B: {exc}")


def finish_wandb(wandb_module, results: list[CommandResult], failed: list[tuple[str, int]], log_dir: Path) -> None:
    if wandb_module is None:
        return
    summary_path = log_dir / "local_run_results.json"
    summary = {
        "results": [
            {
                "name": result.name,
                "command": result.command,
                "log_path": result.log_path,
                "exit_code": result.exit_code,
                "elapsed_seconds": result.elapsed_seconds,
                "status": result.status,
            }
            for result in results
        ],
        "failed": [{"name": name, "exit_code": code} for name, code in failed],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    wandb_module.save(str(summary_path), base_path=str(log_dir))
    wandb_module.summary["subruns_total"] = len(results)
    wandb_module.summary["subruns_failed"] = len(failed)
    wandb_module.summary["suite_success"] = int(not failed)
    wandb_module.finish()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full GUDS-EDL paper suite locally.")
    parser.add_argument("--isic_root", type=str, help="Local ISIC 2024 root containing train-metadata.csv.")
    parser.add_argument("--isic_suite", choices=["main_tables", "baselines", "ablations", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs for ISIC experiments.")
    parser.add_argument("--cifar_epochs", type=int, default=100)
    parser.add_argument("--backbone_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cifar_ratios", type=int, nargs="+", default=[10, 50, 100], choices=[10, 50, 100])
    parser.add_argument("--hardware_iters", type=int, default=50)
    parser.add_argument("--hardware_warmup", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--stream", choices=["compact", "full", "none"], default="compact")
    parser.add_argument("--skip_isic", action="store_true")
    parser.add_argument("--skip_cifar", action="store_true")
    parser.add_argument("--skip_hardware", action="store_true")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--include_backbones", action="store_true")
    parser.add_argument("--no_save_model", action="store_true", help="Do not save model checkpoints for large sweeps.")
    parser.add_argument("--allow_dummy_data", action="store_true", help="Only for dry-runs; never use for paper results.")
    parser.add_argument("--pull_latest", action="store_true", help="Run git pull --ff-only before launching experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run 1 epoch and seed 42 only.")
    parser.add_argument("--keep_going", action="store_true", help="Continue after failed sub-runs.")
    parser.add_argument("--verbose_structural_logs", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--wandb_mode", choices=["offline", "online", "disabled"], default="offline", help="Track local launcher logs/config with W&B.")
    parser.add_argument("--wandb_project", default="mdep-local-experiments")
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_dir", default=None, help="Directory for local W&B files. Defaults to paper_experiment_outputs/wandb.")
    args = parser.parse_args()

    if shutil.which(sys.executable) is None and not Path(sys.executable).exists():
        print(f"[WARN] Could not verify Python executable: {sys.executable}")

    if args.pull_latest:
        git_pull_latest()

    isic_root = detect_isic_root(args.isic_root)
    if isic_root is not None:
        os.environ["ISIC_ROOT"] = str(isic_root)
    elif not args.skip_isic and not args.allow_dummy_data:
        print("[WARN] ISIC dataset not found. Pass --isic_root or set ISIC_ROOT, or use --skip_isic.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = OUTPUT_ROOT / "local_logs" / stamp
    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("WANDB_SILENT", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    print_environment_summary(args, isic_root)
    commands = build_commands(args)
    print("Planned sub-runs:")
    for idx, spec in enumerate(commands, start=1):
        print(f"  {idx:02d}. {spec.name}")

    manifest_path = write_local_run_manifest(args, isic_root, commands, log_dir)
    wandb_module = init_wandb(args, isic_root, commands, log_dir, manifest_path)

    failed: list[tuple[str, int]] = []
    results: list[CommandResult] = []
    for idx, spec in enumerate(commands, start=1):
        result = run_command(spec, env=env, log_dir=log_dir, stream_mode=args.stream)
        results.append(result)
        log_wandb_result(wandb_module, result, idx)
        if result.exit_code != 0:
            failed.append((spec.name, result.exit_code))
            if not args.keep_going:
                break
    finish_wandb(wandb_module, results, failed, log_dir)

    print("\nLocal suite finished.")
    print(f"Logs   : {log_dir}")
    print(f"Output : {OUTPUT_ROOT}")
    if failed:
        print("Failed sub-runs:")
        for name, code in failed:
            print(f"  - {name}: exit {code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
