"""LINC CVRPTW SGBS implementation."""

import argparse
import copy
import json
import math
import pathlib
import sys
import time
from pathlib import Path

import torch


if sys.platform.startswith("win"):
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[3]
MODEL_CVRPTW = (ROOT / "CVRPTW" / "LINC").resolve()
for module_path in (ROOT, ROOT / "CVRPTW", MODEL_CVRPTW):
    module_path = str(module_path)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from CVRPTWModel import CVRPTWModel as Model, _get_encoding  # noqa: E402
from E_CVRPTWEnv import E_CVRPTWEnv as Env  # noqa: E402


def infer_model_params_from_checkpoint(checkpoint):
    if "model_params" in checkpoint:
        return dict(checkpoint["model_params"])
    raise KeyError("Checkpoint does not contain model_params; cannot infer LINC CVRPTW parameters.")


def parse_args():
    parser = argparse.ArgumentParser(description="SGBS CVRPTW benchmark for LINC checkpoints.")
    parser.add_argument("--checkpoint", "--ours-checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--dataset-pt", "--dataset-pkl", dest="dataset_pt", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--z-samples", type=int, default=128)
    parser.add_argument("--aug-factor", type=int, default=8)
    parser.add_argument("--batch-size", "--instance-batch-size", dest="batch_size", type=int, default=1500)
    parser.add_argument("--limit", "--num-instances", dest="limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-routes", action="store_true", help="Store the best decoded tour in each JSON row.")
    parser.add_argument("--amp", choices=("on", "off"), default="on")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--linc-force-alpha-one", "--qlite-force-alpha-one", dest="qlite_force_alpha_one", action="store_true")
    parser.add_argument(
        "--linc-disable-summary-modulation",
        "--qlite-disable-summary-modulation",
        dest="qlite_disable_summary_modulation",
        action="store_true",
    )
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_model(checkpoint_path, device, args):
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_params = infer_model_params_from_checkpoint(checkpoint)
    if args.qlite_force_alpha_one:
        model_params["qlite_force_alpha_one"] = True
    if args.qlite_disable_summary_modulation:
        model_params["qlite_disable_summary_modulation"] = True
    model = Model(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.decoder.capture_candidate_aux = False
    return model, model_params


def sample_bitwise_z(batch_size, z_samples, z_dim, seed, aug_factor, device, mode="random", k=None, global_start_idx=0):
    if mode == "polynet_binary_vectors":
        k = int(k or z_samples)
        pool = torch.tensor(
            [[(i >> bit) & 1 for bit in range(int(z_dim) - 1, -1, -1)] for i in range(k)],
            dtype=torch.float32,
            device=device,
        )
        repeat = math.ceil(int(z_samples) / max(1, k))
        z = pool.repeat(repeat, 1)[: int(z_samples)]
        return z[None, :, :].expand(int(aug_factor) * int(batch_size), -1, -1).contiguous()

    generator = torch.Generator(device="cpu")
    z = torch.empty((int(aug_factor) * int(batch_size), int(z_samples), int(z_dim)), dtype=torch.int64, device="cpu")
    for local_idx in range(int(batch_size)):
        generator.manual_seed(int(seed) + int(global_start_idx) + local_idx)
        z_block = torch.randint(
            low=0,
            high=2,
            size=(int(aug_factor), int(z_samples), int(z_dim)),
            generator=generator,
            dtype=torch.int64,
            device="cpu",
        )
        for aug_idx in range(int(aug_factor)):
            z[aug_idx * int(batch_size) + local_idx] = z_block[aug_idx]
    return z.to(device=device, dtype=torch.float32)


def prepare_env_batch(env, dataset, start_idx, batch_size, device):
    batch_slice = slice(start_idx, start_idx + batch_size)
    env.FLAG__use_saved_problems = True
    env.saved_depot_xy = dataset["depot_xy"][batch_slice]
    env.saved_node_xy = dataset["node_xy"][batch_slice]
    env.saved_node_demand = dataset["node_demand"][batch_slice]
    env.saved_node_tw = dataset["node_tw"][batch_slice]
    env.saved_depot_tw = None if dataset.get("depot_tw") is None else dataset["depot_tw"][batch_slice]
    env.capacity = dataset["capacity"][batch_slice]
    env.saved_grid_size = dataset["grid_size"][batch_slice]
    env.saved_service_t = dataset["service_duration"][batch_slice].unsqueeze(1)
    env.saved_travel_time_scale = dataset["travel_time_scale"][batch_slice]
    env.grid_size = float(dataset["grid_size"][batch_slice].float().mean().item())
    env.service_t = env.saved_service_t
    env.saved_index = 0
    env.load_problems(batch_size, env.rollout_size, device, aug_factor=env.aug_factor)


def load_saved_dataset(dataset_path):
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(dataset, dict):
        raise TypeError(f"expected dict dataset, got {type(dataset)!r}")

    node_xy = dataset["node_xy"].float()
    batch_size = int(node_xy.shape[0])

    def ensure_batch_vector(value, default):
        if value is None:
            return torch.full((batch_size,), float(default), dtype=torch.float32)
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim == 0:
            return tensor.repeat(batch_size)
        if tensor.ndim == 1:
            if tensor.numel() == 1:
                return tensor.repeat(batch_size)
            return tensor
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            return tensor[:, 0]
        raise ValueError(f"Expected scalar, vector, or (batch,1) tensor, got {tuple(tensor.shape)}")

    normalized = {
        "depot_xy": dataset["depot_xy"].float(),
        "node_xy": node_xy,
        "node_demand": dataset["node_demand"].float(),
        "capacity": ensure_batch_vector(dataset.get("capacity", None), 1.0),
        "grid_size": ensure_batch_vector(dataset.get("grid_size", dataset.get("scale", None)), 100.0),
        "service_duration": ensure_batch_vector(
            dataset.get("service_t", dataset.get("service_duration", None)),
            0.0,
        ),
        "travel_time_scale": ensure_batch_vector(dataset.get("travel_time_scale", None), 1.0),
    }
    if "node_tw" not in dataset:
        raise KeyError("Saved dataset is missing node_tw")
    normalized["node_tw"] = dataset["node_tw"].float()
    if "depot_tw" in dataset:
        depot_tw = dataset["depot_tw"].float()
        if depot_tw.ndim == 2:
            depot_tw = depot_tw.unsqueeze(1)
        normalized["depot_tw"] = depot_tw
    elif "depot_horizon" in dataset:
        depot_horizon = dataset["depot_horizon"].float()
        if depot_horizon.ndim == 1:
            depot_horizon = depot_horizon.unsqueeze(1)
        if depot_horizon.ndim == 2:
            depot_horizon = depot_horizon.unsqueeze(1)
        normalized["depot_tw"] = depot_horizon
    else:
        normalized["depot_tw"] = None
    if "bks_cost" in dataset:
        normalized["bks_cost"] = torch.as_tensor(dataset["bks_cost"], dtype=torch.float32)
    if "names" in dataset:
        normalized["names"] = list(dataset["names"])
    return normalized


def get_expand_score(model, state, return_logits=False):
    if state.selected_count == 0:
        batch_size = state.BATCH_IDX.size(0)
        rollout_size = state.BATCH_IDX.size(1)
        score = torch.full(
            (batch_size, rollout_size, model.encoded_nodes.size(1)),
            float("-inf") if return_logits else 0.0,
            dtype=model.encoded_nodes.dtype,
            device=model.encoded_nodes.device,
        )
        score[:, :, 0] = 0.0 if return_logits else 1.0
        return score

    encoded_last_node = _get_encoding(model.encoded_nodes, state.current_node)
    return model.decoder(
        encoded_last_node,
        state.load,
        state.time,
        ninf_mask=state.ninf_mask,
        candidate_features=getattr(state, "candidate_features", None),
        current_node=getattr(state, "current_node", None),
        visited_mask=getattr(state, "visited_mask", None),
        finished=getattr(state, "finished", None),
        return_logits=return_logits,
    )


def greedy_step(model, state):
    logits = get_expand_score(model, state, return_logits=True)
    return logits.argmax(dim=2)


def greedy_finish(model, env, z, use_amp, amp_dtype):
    model.decoder.set_z(z)
    state = env.step_state
    if state.current_node is None or state.selected_count != env.selected_count:
        state, _, done = env.pre_step()
    else:
        done = bool(env.finished.all())
    reward = None
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        while not done:
            selected = greedy_step(model, state)
            state, reward, done = env.step(selected)
    return reward.detach()


def clone_env(env):
    cloned = object.__new__(env.__class__)
    cloned.__dict__ = dict(env.__dict__)
    cloned.reset_state = type(env.reset_state)()
    cloned.reset_state.__dict__ = dict(env.reset_state.__dict__)
    cloned.step_state = type(env.step_state)()
    cloned.step_state.__dict__ = dict(env.step_state.__dict__)
    cloned.selected_count = int(env.selected_count)
    for attr in (
        "current_node",
        "selected_node_list",
        "at_the_depot",
        "load",
        "visited_ninf_flag",
        "ninf_mask",
        "finished",
        "time",
        "used_vehicles",
    ):
        value = getattr(env, attr, None)
        if torch.is_tensor(value):
            value = value.clone()
        setattr(cloned, attr, value)
    model_batch_index = getattr(env, "_model_batch_index", None)
    cloned._model_batch_index = model_batch_index.clone() if torch.is_tensor(model_batch_index) else model_batch_index
    if hasattr(cloned, "_sync_step_state"):
        cloned._sync_step_state()
    return cloned


def select_initial_beams(model, env, z, beam_width, use_amp, amp_dtype):
    device = env.depot_node_xy.device
    depot_action = torch.zeros((env.batch_size, env.rollout_size), dtype=torch.long, device=device)
    state, _, _ = env.step(depot_action)
    model.decoder.set_z(z)
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        first_action = greedy_step(model, state)
    state, _, done = env.step(first_action)

    initial_env = clone_env(env)
    scoring_env = clone_env(env)
    full_reward = greedy_finish(model, scoring_env, z, use_amp, amp_dtype)
    top_reward, top_index = full_reward.topk(k=min(int(beam_width), full_reward.size(1)), dim=1)

    beam_env = clone_env(env)
    beam_env.reset_by_gathering_rollout_env(initial_env, top_index)
    beam_z = _gather_rollout_tensor(z, top_index)
    model.decoder.set_z(beam_z)
    return beam_env, beam_z, top_reward, done


def run_batch_sgbs(model, env, z, beam_width, expand_k, use_amp, amp_dtype):
    expansion_size_minus1 = max(1, int(expand_k) - 1)
    rollout_width = int(beam_width) * expansion_size_minus1

    beam_env, beam_z, beam_reward, done = select_initial_beams(
        model, env, z, beam_width, use_amp, amp_dtype
    )
    rollout_env = clone_env(beam_env)
    rollout_env.modify_rollout_size(rollout_width)
    rollout_env_snapshot = clone_env(rollout_env)

    first_rollout_flag = True
    while not done:
        model.decoder.set_z(beam_z)
        state = beam_env.step_state
        state.candidate_features = beam_env._get_candidate_features() if beam_env.enable_candidate_features else None
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = get_expand_score(model, state, return_logits=True)
            select_k = expansion_size_minus1 if first_rollout_flag else expansion_size_minus1 + 1
            ordered_score, ordered_i = logits.topk(k=select_k, dim=2, largest=True, sorted=True)
            selected_score_all = ordered_score[:, :, :select_k]
            selected_i_all = ordered_i[:, :, :select_k]

        greedy_next_node = selected_i_all[:, :, 0]
        if first_rollout_flag:
            score_selected = selected_score_all[:, :, :expansion_size_minus1]
            idx_selected = selected_i_all[:, :, :expansion_size_minus1]
        else:
            score_selected = selected_score_all[:, :, 1:expansion_size_minus1 + 1]
            idx_selected = selected_i_all[:, :, 1:expansion_size_minus1 + 1]

        next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion_size_minus1)
        is_valid = torch.isfinite(score_selected)
        next_nodes[is_valid] = idx_selected[is_valid]

        rollout_env.reset_by_repeating_bs_env(beam_env, repeat=expansion_size_minus1)
        rollout_env_snapshot.copy_dynamic_from(rollout_env)
        rollout_z = beam_z.repeat_interleave(expansion_size_minus1, dim=1)
        next_nodes = next_nodes.reshape(beam_env.batch_size, rollout_width)

        model.decoder.set_z(rollout_z)
        rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            while not rollout_done:
                selected = greedy_step(model, rollout_state)
                rollout_state, rollout_reward, rollout_done = rollout_env.step(selected)

        rollout_reward[(~is_valid).reshape(beam_env.batch_size, rollout_width)] = float("-inf")

        if not first_rollout_flag:
            rollout_env_snapshot.merge(beam_env)
            rollout_reward = torch.cat((rollout_reward, beam_reward), dim=1)
            next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
            rollout_z = torch.cat((rollout_z, beam_z), dim=1)
        first_rollout_flag = False

        beam_reward, beam_index = rollout_reward.topk(k=beam_width, dim=1, largest=True, sorted=True)

        beam_env.reset_by_gathering_rollout_env(rollout_env_snapshot, gathering_index=beam_index)
        beam_z = _gather_rollout_tensor(rollout_z, beam_index)
        selected = next_nodes.gather(dim=1, index=beam_index)
        model.decoder.set_z(beam_z)
        _, reward, done = beam_env.step(selected)

    return reward, beam_env


def solve_batch(env, model, model_params, device, dataset, start_idx, batch_size, args, amp_dtype):
    env.rollout_size = int(args.z_samples)
    env.aug_factor = int(args.aug_factor)
    prepare_env_batch(env, dataset, start_idx, batch_size, device)
    reset_state, _, _ = env.reset()
    z = sample_bitwise_z(
        batch_size=batch_size,
        z_samples=int(args.z_samples),
        z_dim=int(model_params["z_dim"]),
        seed=int(args.seed),
        aug_factor=int(args.aug_factor),
        device=device,
        mode=model_params.get("z_sampling_mode", "random"),
        k=model_params.get("polynet_k", None),
        global_start_idx=int(start_idx),
    )
    model.pre_forward(reset_state, z)
    reward, final_env = run_batch_sgbs(
        model=model,
        env=env,
        z=z,
        beam_width=int(args.beam_width),
        expand_k=int(args.expand_k),
        use_amp=(args.amp == "on" and device.type == "cuda"),
        amp_dtype=amp_dtype,
    )
    reward_by_aug = reward.reshape(int(args.aug_factor), int(batch_size), int(args.beam_width))
    best_beam_reward, beam_argmax = reward_by_aug.max(dim=2)
    best_reward, aug_argmax = best_beam_reward.max(dim=0)
    best_beam = beam_argmax.gather(dim=0, index=aug_argmax.unsqueeze(0)).squeeze(0)

    selected_count = int(getattr(final_env, "selected_count", 0))
    selected_node_list = getattr(final_env, "selected_node_list", None)
    batch_rows = []
    for local_idx in range(int(batch_size)):
        aug_index = int(aug_argmax[local_idx].item())
        beam_index = int(best_beam[local_idx].item())
        row = {
            "distance": -float(best_reward[local_idx].item()),
            "aug_index": aug_index,
            "beam_index": beam_index,
        }
        if args.include_routes:
            if selected_node_list is None:
                raise RuntimeError("final_env does not expose selected_node_list; cannot include routes")
            env_row = aug_index * int(batch_size) + local_idx
            tour = selected_node_list[env_row, beam_index, :selected_count].detach().long().cpu().tolist()
            row["tour"] = tour
            row["selected_count"] = selected_count
        batch_rows.append(row)
    return batch_rows


def _gather_rollout_tensor(value, gathering_index):
    if value.ndim == 2:
        return value.gather(dim=1, index=gathering_index)
    expand_index = gathering_index
    for _ in range(2, value.ndim):
        expand_index = expand_index.unsqueeze(-1)
    expand_index = expand_index.expand(-1, -1, *value.shape[2:])
    return value.gather(dim=1, index=expand_index)


def main():
    args = parse_args()
    if torch.cuda.is_available():
        tf32_enabled = args.tf32 == "on"
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda:0")
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    checkpoint_path = resolve(args.checkpoint)
    dataset_path = resolve(args.dataset_pt)
    if args.output_json:
        output_json = resolve(args.output_json)
    else:
        output_json = ROOT / "results" / "cvrptw_linc_sgbs_eval.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_saved_dataset(dataset_path)
    total = int(dataset["node_xy"].shape[0])
    if args.limit > 0:
        total = min(total, int(args.limit))
        dataset = {
            key: value[:total] if torch.is_tensor(value) and value.shape[0] >= total else value
            for key, value in dataset.items()
        }
    problem_size = int(dataset["node_xy"].shape[1])
    names = dataset.get("names", [f"inst_{idx:03d}" for idx in range(total)])
    bks_cost = dataset.get("bks_cost", None)

    model, model_params = load_model(checkpoint_path, device, args)
    env = Env(problem_size=problem_size)
    env.input_scale_mode = str(model_params.get("cvrptw_input_scale_mode", "grid"))
    env.enforce_depot_return = True
    env.enable_candidate_features = bool(model_params.get("use_candidate_features", False))
    use_fused_candidate_features = bool(
        env.enable_candidate_features
        and model_params.get("candidate_scorer_type", "quotient_lite") == "quotient_lite"
    )
    env.use_fused_candidate_features = use_fused_candidate_features
    env.use_selected_candidate_features = bool(env.enable_candidate_features and not use_fused_candidate_features)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    rows = []
    with torch.inference_mode():
        for start_idx in range(0, total, int(args.batch_size)):
            batch_size = min(int(args.batch_size), total - start_idx)
            batch_start = time.perf_counter()
            batch_rows = solve_batch(env, model, model_params, device, dataset, start_idx, batch_size, args, amp_dtype)
            batch_elapsed = time.perf_counter() - batch_start
            for local_idx, batch_row in enumerate(batch_rows):
                global_idx = start_idx + local_idx
                row = {
                    "instance": str(names[global_idx]),
                    "distance": float(batch_row.pop("distance")),
                    "elapsed_sec": batch_elapsed / batch_size,
                }
                row.update(batch_row)
                if bks_cost is not None:
                    bks_value = float(torch.as_tensor(bks_cost[global_idx]).item())
                    row["bks_cost"] = bks_value
                    row["gap_pct"] = 100.0 * (row["distance"] - bks_value) / bks_value
                rows.append(
                    row
                )
            print(f"[progress] {start_idx + batch_size}/{total}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_memory_mb = None
    elapsed_sec = time.perf_counter() - start_time
    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset_pt": str(dataset_path),
        "instance_count": len(rows),
        "problem_size": problem_size,
        "mean_distance": sum(row["distance"] for row in rows) / len(rows),
        "beam_width": int(args.beam_width),
        "expand_k": int(args.expand_k),
        "z_samples": int(args.z_samples),
        "aug_factor": int(args.aug_factor),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "elapsed_sec": elapsed_sec,
        "peak_memory_mb": peak_memory_mb,
        "rows": rows,
    }
    if rows and "bks_cost" in rows[0]:
        payload["mean_bks_cost"] = sum(row["bks_cost"] for row in rows) / len(rows)
        payload["mean_gap_pct"] = sum(row["gap_pct"] for row in rows) / len(rows)
        payload["aggregate_gap_pct"] = 100.0 * (
            sum(row["distance"] for row in rows) - sum(row["bks_cost"] for row in rows)
        ) / sum(row["bks_cost"] for row in rows)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
