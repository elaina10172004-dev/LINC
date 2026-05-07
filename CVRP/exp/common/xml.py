"""CVRPLIB XML helpers shared by CVRP evaluation entrypoints."""

from __future__ import annotations

import importlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


TASK_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CANDIDATE_FEATURES = [
    "travel_dist_norm",
    "demand_ratio",
    "dist_to_depot_norm",
    "depot_angle_diff_norm",
]
DEFAULT_NODE_STATIC_FEATURES = [
    "knn_mean_dist_norm",
]

DEFAULT_POLY_PARAMS = {
    "embedding_dim": 128,
    "poly_embedding_dim": 256,
    "sqrt_embedding_dim": math.sqrt(128.0),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "softmax",
    "z_dim": 16,
    "use_fast_attention": True,
    "force_first_move": False,
    "use_depth_mixer": False,
    "use_gated_attention": False,
    "gated_attention_init_bias": 2.0,
    "alpha_attn_gate": 1.0,
    "gated_attention_scale_mode": "centered_sigmoid",
    "use_candidate_features": False,
    "selected_candidate_feature_names": [],
    "relative_candidate_feature_names": [],
    "selected_node_static_feature_names": [],
    "node_static_embedding_mode": "concat",
    "candidate_feature_hidden_dim": 0,
    "candidate_rollout_chunk_size": 32,
    "use_decoder_checkpointing": False,
    "candidate_scorer_type": "baseline_additive",
    "quotient_scorer_hidden_dim": 64,
    "quotient_lite_hidden_dim": 64,
    "quotient_scorer_activation": "gelu",
    "capture_candidate_aux": False,
    "qlite_force_alpha_one": False,
    "qlite_disable_summary_modulation": False,
    "zero_depot_relative_features": False,
}

POMO_MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": math.sqrt(128.0),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
}


def resolve_task_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = TASK_ROOT / path
    return path.resolve()


def resolve_checkpoint_path(checkpoint: str | Path, epoch: int | None = None) -> Path:
    path = resolve_task_path(checkpoint)
    if path.suffix == ".pt":
        return path
    if epoch is None:
        raise ValueError(f"checkpoint directory requires an epoch: {path}")
    return (path / f"checkpoint-{int(epoch)}.pt").resolve()


def configure_device(seed: int, cuda_device: int = 0, cpu: bool = False) -> torch.device:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and not cpu:
        torch.cuda.set_device(cuda_device)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.set_float32_matmul_precision("medium")
        return torch.device("cuda", cuda_device)
    torch.set_default_tensor_type("torch.FloatTensor")
    return torch.device("cpu")


def _split_header_value(line: str) -> str:
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.split()[-1].strip()


def read_vrplib_cvrp(path: Path, bks_by_name: dict[str, float] | None = None) -> dict:
    coords: list[tuple[float, float]] = []
    demands: list[float] = []
    depot = None
    capacity = None
    mode = ""

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("CAPACITY"):
                capacity = float(_split_header_value(line))
            elif line == "NODE_COORD_SECTION":
                mode = "coord"
            elif line == "DEMAND_SECTION":
                mode = "demand"
            elif line == "DEPOT_SECTION":
                mode = "depot"
            elif line == "EOF":
                break
            elif mode == "coord":
                node, x, y = line.split()[:3]
                if int(node) == 1:
                    depot = (float(x), float(y))
                else:
                    coords.append((float(x), float(y)))
            elif mode == "demand":
                _, demand = line.split()[:2]
                demands.append(float(demand))
            elif mode == "depot":
                continue

    if depot is None or capacity is None:
        raise ValueError(f"Failed to parse depot/capacity from {path}")
    if len(demands) != len(coords) + 1:
        raise ValueError(f"Unexpected demand count in {path}: {len(demands)} vs {len(coords) + 1}")

    grid_size = float(max(1.0, depot[0], depot[1], max(x for x, _ in coords), max(y for _, y in coords)))
    item = {
        "name": path.stem,
        "depot_xy": torch.tensor(depot, dtype=torch.float32) / grid_size,
        "node_xy": torch.tensor(coords, dtype=torch.float32) / grid_size,
        "node_demand": torch.tensor(demands[1:], dtype=torch.float32) / float(capacity),
        "capacity": float(capacity),
        "grid_size": grid_size,
    }
    if bks_by_name is not None:
        if path.stem not in bks_by_name:
            raise KeyError(f"Missing BKS for {path.stem}")
        item["bks_cost"] = float(bks_by_name[path.stem])
    return item


def load_xml_instances(
    instances_root: str | Path,
    *,
    bks_json: str | Path | None = None,
    manifest: str | Path | None = None,
    max_instances: int = 0,
) -> list[dict]:
    instances_root = resolve_task_path(instances_root)
    bks_by_name = None
    if bks_json:
        bks_by_name = json.loads(resolve_task_path(bks_json).read_text(encoding="utf-8"))
    if manifest:
        manifest_data = json.loads(resolve_task_path(manifest).read_text(encoding="utf-8"))
        paths = [instances_root / name for name in manifest_data["files"]]
    else:
        paths = sorted(instances_root.glob("*.vrp"))
    if max_instances > 0:
        paths = paths[: int(max_instances)]
    instances = [read_vrplib_cvrp(path, bks_by_name=bks_by_name) for path in paths]
    if not instances:
        raise ValueError(f"No XML CVRP instances found under {instances_root}")
    first_size = int(instances[0]["node_xy"].shape[0])
    bad = [item["name"] for item in instances if int(item["node_xy"].shape[0]) != first_size]
    if bad:
        raise ValueError(f"Mixed-size XML batch is not supported for CVRP XML100: first mismatch {bad[0]}")
    return instances


def build_saved_tensors(instances: Iterable[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    items = list(instances)
    depot_xy = torch.stack([item["depot_xy"] for item in items], dim=0)[:, None, :].to(device)
    node_xy = torch.stack([item["node_xy"] for item in items], dim=0).to(device)
    node_demand = torch.stack([item["node_demand"] for item in items], dim=0).to(device)
    return depot_xy, node_xy, node_demand


def summarize_rows(rows: list[dict]) -> dict:
    mean_cost = sum(row["cost"] for row in rows) / len(rows)
    payload = {
        "instance_count": len(rows),
        "mean_cost": float(mean_cost),
    }
    if rows and all("bks_cost" in row for row in rows):
        mean_bks = sum(row["bks_cost"] for row in rows) / len(rows)
        mean_gap = sum(row["gap_pct"] for row in rows) / len(rows)
        aggregate_gap = 100.0 * (
            sum(row["cost"] for row in rows) - sum(row["bks_cost"] for row in rows)
        ) / sum(row["bks_cost"] for row in rows)
        payload.update(
            {
                "mean_bks_cost": float(mean_bks),
                "mean_gap_pct": float(mean_gap),
                "aggregate_gap_pct": float(aggregate_gap),
            }
        )
    return payload


def add_bks_metrics(row: dict, inst: dict) -> dict:
    if "bks_cost" not in inst:
        return row
    bks = float(inst["bks_cost"])
    row["bks_cost"] = bks
    row["gap_pct"] = 100.0 * (float(row["cost"]) - bks) / bks
    return row


def format_bks_json(bks_json: str | Path | None) -> str | None:
    return str(resolve_task_path(bks_json)) if bks_json else None


def write_result(payload: dict, output_json: str | Path | None) -> None:
    if not output_json:
        return
    output_path = resolve_task_path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(payload: dict) -> None:
    keys = [
        "method",
        "instance_count",
        "mean_cost",
        "mean_bks_cost",
        "mean_gap_pct",
        "aggregate_gap_pct",
        "elapsed_sec",
        "peak_memory_mb",
    ]
    print(json.dumps({key: payload.get(key) for key in keys if key in payload}, indent=2))


def _reset_impl_modules() -> None:
    for module_name in ("CVRPEnv", "CVRPModel", "CVRProblemDef", "candidate_scorers"):
        sys.modules.pop(module_name, None)


def import_task_impl(impl: str):
    _reset_impl_modules()
    impl_root = TASK_ROOT / impl
    if not impl_root.exists():
        raise FileNotFoundError(impl_root)
    for path in (impl_root, TASK_ROOT):
        path_s = str(path)
        if path_s in sys.path:
            sys.path.remove(path_s)
        sys.path.insert(0, path_s)
    env_module = importlib.import_module("CVRPEnv")
    model_module = importlib.import_module("CVRPModel")
    return env_module.CVRPEnv, model_module.CVRPModel


def infer_poly_model_params(checkpoint: dict) -> dict:
    model_params = dict(DEFAULT_POLY_PARAMS)
    state_dict = checkpoint["model_state_dict"]
    model_params["z_dim"] = int(checkpoint.get("z_dim", model_params["z_dim"]))
    model_params["force_first_move"] = bool(checkpoint.get("force_first_move", model_params["force_first_move"]))

    embedding_node = state_dict.get("encoder.embedding_node.weight")
    if embedding_node is not None and int(embedding_node.shape[1]) == 4:
        model_params["selected_node_static_feature_names"] = list(DEFAULT_NODE_STATIC_FEATURES)
        model_params["node_static_embedding_mode"] = "concat"

    phi_proj = state_dict.get("decoder.full_phi_proj.weight")
    if phi_proj is not None:
        candidate_dim = int(phi_proj.shape[1])
        model_params["use_candidate_features"] = candidate_dim > 0
        model_params["selected_candidate_feature_names"] = list(DEFAULT_CANDIDATE_FEATURES[:candidate_dim])
        model_params["candidate_scorer_type"] = "quotient_lite"
        model_params["use_decoder_checkpointing"] = False

    model_params.update(dict(checkpoint.get("model_params", {})))
    if not model_params.get("use_candidate_features", False):
        model_params["candidate_scorer_type"] = "baseline_additive"
        model_params["selected_candidate_feature_names"] = []
        model_params["relative_candidate_feature_names"] = []
    return model_params


def load_poly_model(impl: str, checkpoint_path: Path, device: torch.device):
    env_cls, model_cls = import_task_impl(impl)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_params = infer_poly_model_params(checkpoint)
    model = model_cls(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return env_cls, model, model_params, checkpoint


def sample_seed_parallel_z(batch_size: int, z_samples: int, z_dim: int, seed: int, aug_factor: int, device: torch.device):
    pool_size = 2 ** int(z_dim)
    replacement = int(z_samples) > pool_size
    probs = torch.full(
        (int(batch_size) * int(aug_factor), pool_size),
        1.0 / pool_size,
        dtype=torch.float32,
        device="cpu",
    )
    binary_string_pool = torch.tensor(
        [[(i >> bit) & 1 for bit in range(int(z_dim) - 1, -1, -1)] for i in range(pool_size)],
        dtype=torch.float32,
        device="cpu",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    z_idx = torch.multinomial(probs, int(z_samples), replacement=replacement, generator=generator)
    return binary_string_pool[z_idx].to(device=device, dtype=torch.float32)


def _peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def evaluate_poly_xml_sampling(
    *,
    impl: str,
    method: str,
    checkpoint: str | Path,
    epoch: int | None,
    instances_root: str | Path,
    bks_json: str | Path | None,
    output_json: str | Path | None,
    episodes: int,
    batch_size: int,
    z_samples: int,
    aug_factor: int,
    seed: int,
    cuda_device: int = 0,
    cpu: bool = False,
    amp: bool = True,
    greedy: bool = False,
) -> dict:
    device = configure_device(seed, cuda_device=cuda_device, cpu=cpu)
    checkpoint_path = resolve_checkpoint_path(checkpoint, epoch)
    instances = load_xml_instances(instances_root, bks_json=bks_json, max_instances=episodes)
    problem_size = int(instances[0]["node_xy"].shape[0])
    env_cls, model, model_params, checkpoint_data = load_poly_model(impl, checkpoint_path, device)
    # Performance: use chunk_size 128 like CVRPTW (checkpoint may have stale small value)
    if model_params.get("use_candidate_features"):
        model.candidate_rollout_chunk_size = 128
    try:
        env = env_cls(
            problem_size=problem_size,
            enable_candidate_features=bool(model_params.get("use_candidate_features", False)),
        )
    except TypeError:
        env = env_cls(problem_size=problem_size)
    env.FLAG__use_saved_problems = True

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict] = []
    elapsed = 0.0

    with torch.inference_mode():
        for offset in range(0, len(instances), int(batch_size)):
            batch_instances = instances[offset : offset + int(batch_size)]
            current_batch = len(batch_instances)
            env.saved_depot_xy, env.saved_node_xy, env.saved_node_demand = build_saved_tensors(batch_instances, device)
            env.saved_index = 0

            start = time.perf_counter()
            env.load_problems(current_batch, int(z_samples), int(aug_factor))
            reset_state, _, _ = env.reset()
            z = sample_seed_parallel_z(
                current_batch,
                int(z_samples),
                int(model_params["z_dim"]),
                int(seed) + offset,
                int(aug_factor),
                device,
            )
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and amp)):
                model.pre_forward(reset_state, z)
                state, reward, done = env.pre_step()
                while not done:
                    selected, _ = model(state, greedy_construction=greedy)
                    state, reward, done = env.step(selected)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start

            aug_reward = reward.reshape(int(aug_factor), current_batch, int(z_samples))
            max_rollout_reward = aug_reward.max(dim=2).values
            no_aug_cost = -max_rollout_reward[0].float().cpu()
            aug_cost = -max_rollout_reward.max(dim=0).values.float().cpu()
            for local_idx, inst in enumerate(batch_instances):
                scale = float(inst["grid_size"])
                cost = float(aug_cost[local_idx].item()) * scale
                rows.append(
                    add_bks_metrics(
                        {
                            "name": inst["name"],
                            "no_aug_cost": float(no_aug_cost[local_idx].item()) * scale,
                            "cost": cost,
                        },
                        inst,
                    )
                )

    payload = summarize_rows(rows)
    payload.update(
        {
            "method": method,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint_data.get("epoch", -1)),
            "instances_root": str(resolve_task_path(instances_root)),
            "bks_json": format_bks_json(bks_json),
            "batch_size": int(batch_size),
            "z_samples": int(z_samples),
            "aug_factor": int(aug_factor),
            "seed": int(seed),
            "device": str(device),
            "elapsed_sec": float(elapsed),
            "peak_memory_mb": _peak_memory_mb(device),
            "rows": rows,
        }
    )
    write_result(payload, output_json)
    print_summary(payload)
    return payload


def load_pomo_model(checkpoint_path: Path, device: torch.device):
    env_cls, model_cls = import_task_impl("POMO")
    model = model_cls(**POMO_MODEL_PARAMS).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return env_cls, model, checkpoint


def evaluate_pomo_xml_sampling(
    *,
    checkpoint: str | Path,
    epoch: int | None,
    instances_root: str | Path,
    bks_json: str | Path | None,
    output_json: str | Path | None,
    episodes: int,
    batch_size: int,
    aug_factor: int,
    seed: int,
    cuda_device: int = 0,
    cpu: bool = False,
    amp: bool = True,
    pomo_size: int | None = None,
    mode: str = "sampling",
    eval_type: str = "greedy",
) -> dict:
    device = configure_device(seed, cuda_device=cuda_device, cpu=cpu)
    checkpoint_path = resolve_checkpoint_path(checkpoint, epoch)
    instances = load_xml_instances(instances_root, bks_json=bks_json, max_instances=episodes)
    problem_size = int(instances[0]["node_xy"].shape[0])
    pomo_size = int(problem_size if pomo_size is None else pomo_size)
    env_cls, model, checkpoint_data = load_pomo_model(checkpoint_path, device)
    env = env_cls(problem_size=problem_size, pomo_size=pomo_size)
    env.FLAG__use_saved_problems = True

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict] = []
    elapsed = 0.0

    with torch.inference_mode():
        for offset in range(0, len(instances), int(batch_size)):
            batch_instances = instances[offset : offset + int(batch_size)]
            current_batch = len(batch_instances)
            env.saved_depot_xy, env.saved_node_xy, env.saved_node_demand = build_saved_tensors(batch_instances, device)
            env.saved_index = 0

            start = time.perf_counter()
            env.load_problems(current_batch, int(aug_factor))
            reset_state, _, _ = env.reset()
            with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda" and amp)):
                model.pre_forward(reset_state)
                state, reward, done = env.pre_step()
                while not done:
                    selected, _ = model(state, eval_type=eval_type)
                    state, reward, done = env.step(selected)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - start

            aug_reward = reward.reshape(int(aug_factor), current_batch, pomo_size)
            max_pomo_reward = aug_reward.max(dim=2).values
            no_aug_cost = -max_pomo_reward[0].float().cpu()
            aug_cost = -max_pomo_reward.max(dim=0).values.float().cpu()
            for local_idx, inst in enumerate(batch_instances):
                scale = float(inst["grid_size"])
                cost = float(aug_cost[local_idx].item()) * scale
                rows.append(
                    add_bks_metrics(
                        {
                            "name": inst["name"],
                            "no_aug_cost": float(no_aug_cost[local_idx].item()) * scale,
                            "cost": cost,
                        },
                        inst,
                    )
                )

    payload = summarize_rows(rows)
    payload.update(
        {
            "method": f"pomo_{mode}",
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint_data.get("epoch", -1)),
            "instances_root": str(resolve_task_path(instances_root)),
            "bks_json": format_bks_json(bks_json),
            "batch_size": int(batch_size),
            "pomo_size": int(pomo_size),
            "aug_factor": int(aug_factor),
            "mode": mode,
            "rollouts_per_instance": int(pomo_size) * int(aug_factor),
            "eval_type": eval_type,
            "seed": int(seed),
            "device": str(device),
            "elapsed_sec": float(elapsed),
            "peak_memory_mb": _peak_memory_mb(device),
            "rows": rows,
        }
    )
    write_result(payload, output_json)
    print_summary(payload)
    return payload


def evaluate_poly_xml_sgbs(
    *,
    method: str,
    checkpoint: str | Path,
    instances_root: str | Path,
    bks_json: str | Path | None,
    output_json: str | Path | None,
    episodes: int,
    batch_size: int,
    z_samples: int,
    aug_factor: int,
    beam_width: int,
    expand_k: int,
    seed: int,
    cuda_device: int = 0,
    cpu: bool = False,
) -> dict:
    device = configure_device(seed, cuda_device=cuda_device, cpu=cpu)
    checkpoint_path = resolve_checkpoint_path(checkpoint, None)
    instances = load_xml_instances(instances_root, bks_json=bks_json, max_instances=episodes)
    problem_size = int(instances[0]["node_xy"].shape[0])

    if "linc" in method:
        from exp.sgbs import linc_sgbs_impl as sgbs
    else:
        from exp.sgbs import polynet_sgbs_impl as sgbs

    model, model_params = sgbs.load_model(checkpoint_path, device)
    if model_params.get("use_candidate_features"):
        model.candidate_rollout_chunk_size = 128
    try:
        env = sgbs.CVRPEnv(
            problem_size=problem_size,
            enable_candidate_features=bool(model_params.get("use_candidate_features", False)),
        )
    except TypeError:
        env = sgbs.CVRPEnv(problem_size=problem_size)
        env.enable_candidate_features = True
    env.FLAG__use_saved_problems = True

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict] = []
    start_all = time.perf_counter()

    with torch.inference_mode():
        for offset in range(0, len(instances), int(batch_size)):
            batch_instances = instances[offset : offset + int(batch_size)]
            current_batch = len(batch_instances)
            env.saved_depot_xy, env.saved_node_xy, env.saved_node_demand = build_saved_tensors(batch_instances, device)
            env.saved_index = 0
            env.load_problems(current_batch, int(z_samples), int(aug_factor))
            env._model_batch_index = torch.arange(env.batch_size, device=device, dtype=torch.long)
            reset_state, _, _ = env.reset()
            z = sgbs.sample_seed_parallel_z(
                current_batch,
                int(z_samples),
                int(model_params["z_dim"]),
                int(seed) + offset,
                int(aug_factor),
                device,
            )
            model.pre_forward(reset_state, z)
            batch_t0 = time.perf_counter()
            reward = sgbs.run_batch_sgbs(
                model,
                env,
                z,
                beam_width=int(beam_width),
                expand_k=int(expand_k),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_elapsed = time.perf_counter() - batch_t0
            batch_summary = sgbs.summarize_batch(
                reward,
                aug_factor=int(aug_factor),
                instance_batch_size=current_batch,
                beam_width=int(beam_width),
            )
            for local_idx, inst in enumerate(batch_instances):
                scale = float(inst["grid_size"])
                cost = float(batch_summary["costs"][local_idx].item()) * scale
                rows.append(
                    add_bks_metrics(
                        {
                            "name": inst["name"],
                            "cost": cost,
                            "elapsed_sec": batch_elapsed / current_batch,
                        },
                        inst,
                    )
                )

    payload = summarize_rows(rows)
    payload.update(
        {
            "method": method,
            "checkpoint": str(checkpoint_path),
            "instances_root": str(resolve_task_path(instances_root)),
            "bks_json": format_bks_json(bks_json),
            "batch_size": int(batch_size),
            "z_samples": int(z_samples),
            "aug_factor": int(aug_factor),
            "beam_width": int(beam_width),
            "expand_k": int(expand_k),
            "seed": int(seed),
            "device": str(device),
            "elapsed_sec": float(time.perf_counter() - start_all),
            "peak_memory_mb": _peak_memory_mb(device),
            "rows": rows,
        }
    )
    write_result(payload, output_json)
    print_summary(payload)
    return payload


def evaluate_pomo_xml_sgbs(
    *,
    checkpoint: str | Path,
    instances_root: str | Path,
    bks_json: str | Path | None,
    output_json: str | Path | None,
    episodes: int,
    batch_size: int,
    aug_factor: int,
    beam_width: int,
    expand_k: int,
    seed: int,
    cuda_device: int = 0,
    cpu: bool = False,
    amp: bool = True,
    start_mode: str = "topk",
) -> dict:
    device = configure_device(seed, cuda_device=cuda_device, cpu=cpu)
    checkpoint_path = resolve_checkpoint_path(checkpoint, None)
    instances = load_xml_instances(instances_root, bks_json=bks_json, max_instances=episodes)
    problem_size = int(instances[0]["node_xy"].shape[0])

    from exp.sgbs import pomo_sgbs_impl as pomo_sgbs

    model = pomo_sgbs.load_model(checkpoint_path, device)
    env = pomo_sgbs.CVRPEnv(problem_size=problem_size, pomo_size=problem_size)
    env.FLAG__use_saved_problems = True

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rows: list[dict] = []
    start_all = time.perf_counter()

    with torch.inference_mode():
        for offset in range(0, len(instances), int(batch_size)):
            batch_instances = instances[offset : offset + int(batch_size)]
            current_batch = len(batch_instances)
            env.saved_depot_xy, env.saved_node_xy, env.saved_node_demand = build_saved_tensors(batch_instances, device)
            env.saved_index = 0
            env.pomo_size = problem_size
            env.load_problems(current_batch, int(aug_factor))
            no_aug, aug = pomo_sgbs.run_batch_sgbs(
                model,
                env,
                problem_size,
                int(beam_width),
                int(expand_k),
                int(aug_factor),
                start_mode,
                device.type == "cuda" and amp,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for local_idx, inst in enumerate(batch_instances):
                scale = float(inst["grid_size"])
                cost = float(aug[local_idx].item()) * scale
                rows.append(
                    add_bks_metrics(
                        {
                            "name": inst["name"],
                            "no_aug_cost": float(no_aug[local_idx].item()) * scale,
                            "cost": cost,
                        },
                        inst,
                    )
                )

    payload = summarize_rows(rows)
    payload.update(
        {
            "method": "pomo_sgbs",
            "checkpoint": str(checkpoint_path),
            "instances_root": str(resolve_task_path(instances_root)),
            "bks_json": format_bks_json(bks_json),
            "batch_size": int(batch_size),
            "pomo_size": int(problem_size),
            "aug_factor": int(aug_factor),
            "beam_width": int(beam_width),
            "expand_k": int(expand_k),
            "start_mode": start_mode,
            "seed": int(seed),
            "device": str(device),
            "elapsed_sec": float(time.perf_counter() - start_all),
            "peak_memory_mb": _peak_memory_mb(device),
            "rows": rows,
        }
    )
    write_result(payload, output_json)
    print_summary(payload)
    return payload
