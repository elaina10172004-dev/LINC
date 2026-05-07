from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = ROOT / "CVRPTW"
RESULT_DIR = ROOT / "results" / "ablation" / "cvrptw"

_DATA_SRC = (TASK_ROOT / "data" / "solomon56.pt").resolve()
_DATA_OUT = (TASK_ROOT / "data" / "solomon56_n50.pt").resolve()
_N = 50

VARIANTS = [
    "baseline",
    "full_linc",
    "no_local",
    "naive_mlp",
    "centered_mlp",
    "raw_linear",
    "no_step_summary",
    "full_mu_summary",
    "no_gateattn",
    "no_depth_mixer",
    "no_soft_top1",
    "group_mean",
]

# Dataset builder


def _slice_first_dim(value, limit: int):
    if torch.is_tensor(value) and value.ndim >= 2 and value.size(1) >= limit:
        if value.ndim == 2:
            return value[:, :limit].clone()
        if value.ndim == 3:
            return value[:, :limit, ...].clone()
    return value


def build_dataset(source: Path, output: Path, customers: int) -> dict:
    data = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict dataset at {source}, got {type(data)!r}")

    node_xy = data["node_xy"]
    if int(node_xy.size(1)) < int(customers):
        raise ValueError(
            f"{source} has only {node_xy.size(1)} customers, cannot truncate to {customers}"
        )

    out = {}
    for key, value in data.items():
        if key == "bks_cost":
            continue
        if key in {"node_xy", "node_demand", "node_tw"}:
            out[key] = _slice_first_dim(value, int(customers))
        elif torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value

    out["problem_size"] = int(customers)
    out["source_dataset"] = str(source)
    out["source_problem_size"] = int(node_xy.size(1))
    out["bks_note"] = ""

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)
    return out


def ensure_dataset():
    if _DATA_OUT.exists():
        return
    build_dataset(_DATA_SRC, _DATA_OUT, _N)


# Command builders


def _python():
    return sys.executable


def train_command(args, variant: str) -> list[str]:
    run_name = f"{args.run_prefix}_{variant}_seed{args.seed}"
    return [
        _python(),
        str(ROOT / "CVRPTW" / "exp" / "training" / "train_linc.py"),
        "--run-name", run_name,
        "--ablation-variant", variant,
        "--problem-size", str(_N),
        "--epochs", str(args.epochs),
        "--instances-per-epoch", str(args.instances_per_epoch),
        "--batch-size", str(args.batch_size),
        "--k", str(args.k),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--max-grad-norm", str(args.max_grad_norm),
        "--tau-start", str(args.tau_start),
        "--tau-end", str(args.tau_end),
        "--tau-anneal-ratio", str(args.tau_anneal_ratio),
        "--best-only-start-ratio", str(args.best_only_start_ratio),
        "--reward-scale", str(args.reward_scale),
        "--save-interval", str(args.save_interval),
        "--latest-save-interval", str(args.latest_save_interval),
        "--candidate-rollout-chunk-size", str(args.candidate_rollout_chunk_size),
        "--seed", str(args.seed),
        "--cuda-device", str(args.cuda_device),
    ]


def eval_command(args, variant: str) -> list[str]:
    run_name = f"{args.run_prefix}_{variant}_seed{args.seed}"
    return [
        _python(),
        str(ROOT / "CVRPTW" / "exp" / "sampling" / "eval_linc.py"),
        "--data", str(_DATA_OUT),
        "--checkpoint", f"result/run_{run_name}",
        "--epoch", str(args.epochs),
        "--episodes", "56",
        "--batch-size", str(args.eval_batch_size),
        "--z-samples", str(args.eval_z_samples),
        "--aug", str(args.eval_aug),
        "--eval-type", args.eval_type,
        "--seed", str(args.seed),
        "--cuda-device", str(args.cuda_device),
        "--output-json", str(RESULT_DIR / f"{variant}_eval.json"),
    ]


# Runner


def run_commands(commands: list[list[str]], max_parallel: int, cwd: Path):
    active: list[tuple[str, subprocess.Popen]] = []
    failures: list[tuple[str, int]] = []

    def poll_active(block: bool = False):
        while active and (block or len(active) >= max_parallel):
            name, proc = active[0]
            code = proc.poll()
            if code is None:
                if not block and len(active) < max_parallel:
                    return
                time.sleep(5)
                continue
            active.pop(0)
            if code != 0:
                failures.append((name, code))
                print(f"[fail] {name}: exit={code}", flush=True)
            else:
                print(f"[done] {name}", flush=True)

    for cmd in commands:
        poll_active(block=False)
        print("[run]", shlex.join(cmd), flush=True)
        active.append((" ".join(cmd), subprocess.Popen(cmd, cwd=str(cwd))))

    while active:
        poll_active(block=True)

    if failures:
        raise SystemExit(f"{len(failures)} command(s) failed: {failures}")


# Main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--stage", choices=("train", "eval", "both", "print"), default="print")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--run-prefix", default="ablation_cvrptw")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    # training
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--instances-per-epoch", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=56)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--tau-start", type=float, default=100.0)
    parser.add_argument("--tau-end", type=float, default=0.01)
    parser.add_argument("--tau-anneal-ratio", type=float, default=0.15)
    parser.add_argument("--best-only-start-ratio", type=float, default=0.15)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--latest-save-interval", type=int, default=50)
    parser.add_argument("--candidate-rollout-chunk-size", type=int, default=128)
    # eval
    parser.add_argument("--eval-batch-size", type=int, default=56)
    parser.add_argument("--eval-z-samples", type=int, default=128)
    parser.add_argument("--eval-aug", type=int, default=8)
    parser.add_argument("--eval-type", choices=("greedy", "sampling", "argmax", "softmax"), default="sampling")
    args = parser.parse_args()

    ensure_dataset()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Valid variants: {VARIANTS}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "variants": variants,
        "train": {variant: train_command(args, variant) for variant in variants},
        "eval": {variant: eval_command(args, variant) for variant in variants},
    }
    (RESULT_DIR / "commands.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_commands = [train_command(args, variant) for variant in variants]
    eval_commands = [eval_command(args, variant) for variant in variants]
    commands: list[list[str]] = []
    if args.stage in {"train", "both", "print"}:
        commands.extend(train_commands)
    if args.stage in {"eval", "both", "print"}:
        commands.extend(eval_commands)

    if args.stage == "print":
        for cmd in commands:
            print(shlex.join(cmd))
        return

    if args.stage == "both":
        run_commands(train_commands, max(1, int(args.max_parallel)), ROOT)
        run_commands(eval_commands, max(1, int(args.max_parallel)), ROOT)
    else:
        run_commands(commands, max(1, int(args.max_parallel)), ROOT)


if __name__ == "__main__":
    main()
