"""POMO CVRPTW sampling/greedy evaluation (requires rl4co)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


TASK_ROOT = Path(__file__).resolve().parents[2]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from exp.common.datasets import add_dataset_args, resolve_dataset  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = TASK_ROOT / path
    return path.resolve()


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_horizon(depot_due: float, customer_due: np.ndarray) -> float:
    if math.isfinite(depot_due) and depot_due > 1e-6:
        return float(depot_due)
    finite = customer_due[np.isfinite(customer_due)]
    if finite.size:
        return float(max(float(finite.max()), 1.0))
    return 1.0


def _convert_dataset(dataset_pt: str | Path, episodes: int, scale_mode: str, ignore_depot_due: bool):
    data = torch.load(dataset_pt, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise TypeError(f"expected dict dataset, got {type(data)!r}")

    total = int(_as_numpy(data["node_xy"]).shape[0])
    limit = min(int(episodes), total) if int(episodes) > 0 else total

    depot_xy = _as_numpy(data["depot_xy"]).astype(np.float32)[:limit, 0, :]
    node_xy = _as_numpy(data["node_xy"]).astype(np.float32)[:limit]
    node_demand = _as_numpy(data["node_demand"]).astype(np.float32)[:limit]
    node_tw = _as_numpy(data["node_tw"]).astype(np.float32)[:limit]
    capacity = _as_numpy(data["capacity"]).astype(np.float32).reshape(-1)[:limit]
    grid_size = _as_numpy(data.get("grid_size", data.get("scale", np.ones(limit)))).astype(np.float32).reshape(-1)[:limit]
    travel_time_scale = _as_numpy(data.get("travel_time_scale", np.ones(limit, dtype=np.float32))).astype(np.float32).reshape(-1)[:limit]

    service_key = "service_t" if "service_t" in data else "service_duration"
    service_t = _as_numpy(data[service_key]).astype(np.float32)
    service_t = service_t[:, 0] if service_t.ndim == 2 else service_t.reshape(-1)
    service_t = service_t[:limit]

    if "depot_tw" in data:
        depot_tw = _as_numpy(data["depot_tw"]).astype(np.float32)[:limit, 0, :]
    elif "depot_horizon" in data:
        horizon = _as_numpy(data["depot_horizon"]).astype(np.float32)[:limit]
        depot_tw = horizon if horizon.ndim == 2 else np.stack([np.zeros(limit, dtype=np.float32), horizon], axis=1)
    else:
        depot_tw = np.stack([np.zeros(limit, dtype=np.float32), np.full(limit, np.inf, dtype=np.float32)], axis=1)
    if ignore_depot_due:
        depot_tw = depot_tw.copy()
        depot_tw[:, 1] = np.inf

    n_nodes = int(node_xy.shape[1])
    depot = np.zeros_like(depot_xy)
    locs = np.zeros_like(node_xy)
    demand = np.zeros_like(node_demand)
    durations = np.zeros((limit, n_nodes + 1), dtype=np.float32)
    time_windows = np.zeros((limit, n_nodes + 1, 2), dtype=np.float32)
    distance_scale = np.ones((limit, 1), dtype=np.float32)

    for idx in range(limit):
        if scale_mode == "grid":
            scale = float(max(grid_size[idx], 1.0))
        else:
            scale = _safe_horizon(float(depot_tw[idx, 1]), node_tw[idx, :, 1])
        tscale = float(travel_time_scale[idx])
        depot[idx] = depot_xy[idx] * tscale / scale
        locs[idx] = node_xy[idx] * tscale / scale
        demand[idx] = node_demand[idx] / max(float(capacity[idx]), 1.0)
        durations[idx, 1:] = service_t[idx] / scale
        time_windows[idx, 0] = depot_tw[idx] / scale
        time_windows[idx, 1:] = node_tw[idx] / scale
        distance_scale[idx, 0] = scale

    return {
        "depot": depot.astype(np.float32),
        "locs": locs.astype(np.float32),
        "demand": demand.astype(np.float32),
        "capacity": capacity[:, None].astype(np.float32),
        "durations": durations.astype(np.float32),
        "time_windows": time_windows.astype(np.float32),
        "distance_scale": distance_scale.astype(np.float32),
        "names": data.get("names", None),
        "bks_cost": data.get("bks_cost", None),
    }


def _route_cost_from_actions(locs: torch.Tensor, actions: torch.Tensor, chunk_size: int):
    """Match RL4CO CVRP/CVRPTW reward layout after multistart decoding."""
    if actions.numel() == 0:
        return torch.empty(actions.shape[0], dtype=locs.dtype, device=locs.device)
    locs = locs.float()
    actions = actions.long()
    flat_count, route_len = actions.shape
    aug_batch = int(locs.shape[0])
    rows_all = torch.arange(flat_count, device=actions.device)
    costs = []
    for start in range(0, flat_count, max(1, int(chunk_size))):
        end = min(start + int(chunk_size), flat_count)
        rows = rows_all[start:end]
        base_idx = rows.remainder(aug_batch)
        locs_chunk = locs.index_select(0, base_idx)
        idx = actions[start:end]
        ordered = locs_chunk.gather(1, idx.unsqueeze(-1).expand(-1, -1, locs_chunk.size(-1)))
        depot = locs_chunk[:, 0, :]
        if route_len == 1:
            cost = (ordered[:, 0, :] - depot).norm(p=2, dim=-1) * 2.0
        else:
            first = (ordered[:, 0, :] - depot).norm(p=2, dim=-1)
            middle = (ordered[:, 1:, :] - ordered[:, :-1, :]).norm(p=2, dim=-1).sum(dim=1)
            last = (ordered[:, -1, :] - depot).norm(p=2, dim=-1)
            cost = first + middle + last
        costs.append(cost)
    return torch.cat(costs, dim=0)


def _reward_to_batch_aug_start(reward: torch.Tensor, batch_size: int, aug: int, starts: int):
    if reward.ndim == 3:
        if reward.shape[0] == batch_size:
            return reward
        if reward.shape[1] == batch_size:
            return reward.permute(1, 0, 2)
    return reward.reshape(batch_size, aug, starts)


def _load_model(checkpoint_path: Path, aug: int, starts: int, device: torch.device):
    from rl4co.envs import CVRPTWEnv
    from rl4co.models.zoo import POMO

    env = CVRPTWEnv()
    model = POMO(env, num_augment=int(aug), num_starts=int(starts))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="synthetic100")
    parser.add_argument("--checkpoint", default="models/POMO_env/n100/checkpoint-608871.ckpt")
    parser.add_argument("--problem-size", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=0, help="Number of POMO starts for sampling. Default 0 means n starts.")
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--mode", choices=("sampling", "greedy"), default="sampling")
    parser.add_argument("--scale-mode", choices=("horizon", "grid"), default="horizon")
    parser.add_argument("--reward-chunk-size", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--respect-depot-due", action="store_true")
    args = parser.parse_args()
    resolve_dataset(args)
    if args.mode == "greedy":
        args.k = 1
    elif int(args.k) <= 0:
        args.k = int(args.problem_size)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = torch.float16

    ckpt_path = _resolve(args.checkpoint)
    data_path = _resolve(args.data)
    model, checkpoint = _load_model(ckpt_path, args.aug, args.k, device)

    prep_t0 = time.perf_counter()
    arrays = _convert_dataset(data_path, args.episodes, args.scale_mode, ignore_depot_due=not args.respect_depot_due)
    names = arrays.pop("names")
    bks = arrays.pop("bks_cost")
    td_all = TensorDict(arrays, batch_size=len(arrays["depot"]))
    prep_time = time.perf_counter() - prep_t0
    total = int(td_all.batch_size[0])
    print(f"[data] vectorized prep: {prep_time:.2f}s")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    score_sum = 0.0
    aug_sum = 0.0
    rows = []
    decode_type = "multistart_greedy"
    start = time.perf_counter()

    from rl4co.utils.ops import unbatchify

    with torch.inference_mode():
        for offset in range(0, total, int(args.batch_size)):
            batch = td_all[offset : offset + int(args.batch_size)].to(device)
            bsz = int(batch.batch_size[0])
            td = model.env.reset(batch)
            if int(args.aug) > 1:
                td = model.augment(td)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model.policy(
                    td,
                    model.env,
                    phase="test",
                    calc_reward=False,
                    return_actions=True,
                    num_starts=int(args.k),
                    decode_type=decode_type,
                )

            reward_flat = -_route_cost_from_actions(td["locs"], out["actions"], int(args.reward_chunk_size))
            reward = unbatchify(reward_flat, (int(args.aug), int(args.k)))
            reward3 = _reward_to_batch_aug_start(reward, bsz, int(args.aug), int(args.k))
            scale = batch["distance_scale"].squeeze(-1).float()
            no_aug_cost = (-reward3[:, 0, :].max(dim=1).values.float()) * scale
            aug_cost = (-reward3.max(dim=2).values.max(dim=1).values.float()) * scale

            score_sum += float(no_aug_cost.sum().item())
            aug_sum += float(aug_cost.sum().item())
            for local_idx in range(bsz):
                global_idx = offset + local_idx
                row = {
                    "instance": str(names[global_idx]) if names is not None else f"inst_{global_idx:03d}",
                    "no_aug_distance": float(no_aug_cost[local_idx].item()),
                    "distance": float(aug_cost[local_idx].item()),
                }
                if bks is not None:
                    bks_value = float(torch.as_tensor(bks[global_idx]).item())
                    row["bks_cost"] = bks_value
                    row["gap_pct"] = 100.0 * (row["distance"] - bks_value) / bks_value
                rows.append(row)
            if offset % (int(args.batch_size) * 5) == 0:
                print(f"  [{min(offset + bsz, total)}/{total}] no_aug={score_sum/len(rows):.4f} aug={aug_sum/len(rows):.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_memory = None
    elapsed = time.perf_counter() - start
    print(f"\nPOMO CVRPTW {args.mode}: no_aug={score_sum/len(rows):.4f}  aug*k={aug_sum/len(rows):.4f}  time={elapsed:.1f}s")

    if args.output_json:
        result = {
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "dataset_pt": str(data_path),
            "seed": int(args.seed),
            "problem_size": int(args.problem_size),
            "episodes": len(rows),
            "batch_size": int(args.batch_size),
            "k": int(args.k),
            "aug_factor": int(args.aug),
            "mode": args.mode,
            "decode_type": decode_type,
            "rollouts_per_instance": int(args.k) * int(args.aug),
            "scale_mode": args.scale_mode,
            "ignore_depot_due": not args.respect_depot_due,
            "score_mean": score_sum / max(1, len(rows)),
            "aug_score_mean": aug_sum / max(1, len(rows)),
            "mean_distance": aug_sum / max(1, len(rows)),
            "no_aug_mean_distance": score_sum / max(1, len(rows)),
            "data_prep_sec": prep_time,
            "elapsed_sec": elapsed,
            "peak_memory_mb": peak_memory,
            "rows": rows,
        }
        if rows and "bks_cost" in rows[0]:
            result["mean_bks_cost"] = sum(row["bks_cost"] for row in rows) / len(rows)
            result["mean_gap_pct"] = sum(row["gap_pct"] for row in rows) / len(rows)
            result["aggregate_gap_pct"] = 100.0 * (
                sum(row["distance"] for row in rows) - sum(row["bks_cost"] for row in rows)
            ) / sum(row["bks_cost"] for row in rows)
        output = _resolve(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
