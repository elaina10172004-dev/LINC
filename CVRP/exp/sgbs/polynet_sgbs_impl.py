"""PolyNet CVRP SGBS implementation.

This module imports only ``CVRP/PolyNet`` code. LINC uses the sibling
``linc_sgbs_impl.py``.
"""

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
CVRP_ROOT = ROOT
CVRP_MODEL_ROOT = CVRP_ROOT / "PolyNet"

if str(CVRP_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(CVRP_MODEL_ROOT))
if str(CVRP_ROOT) not in sys.path:
    sys.path.insert(0, str(CVRP_ROOT))

from CVRPEnv import CVRPEnv  # noqa: E402
from CVRPModel import CVRPModel, _get_encoding  # noqa: E402


def infer_problem_size(dataset_pkl: Path, index_begin: int = 0) -> int:
    with dataset_pkl.open("rb") as f:
        data = pickle.load(f)
    sample = data[int(index_begin)]
    return int(len(sample[1]))


DEFAULT_OFFICIAL_MODEL_PARAMS = {
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
    "qlite_force_alpha_one": False,
    "qlite_disable_summary_modulation": False,
    "zero_depot_relative_features": False,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PolyNet CVRP checkpoints with SGBS."
    )
    parser.add_argument("--checkpoint", "--ours-checkpoint", dest="checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset-pkl",
        type=str,
        default=str(CVRP_ROOT / "data" / "vrp100_test_seed1234.pkl"),
    )
    parser.add_argument("--num-instances", type=int, default=100)
    parser.add_argument("--index-begin", type=int, default=0)
    parser.add_argument("--instance-batch-size", type=int, default=800)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--aug-factor", type=int, default=8)
    parser.add_argument("--z-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(CVRP_ROOT / "results" / "cvrp_polynet_sgbs.json"),
    )
    return parser.parse_args()


def infer_model_params_from_checkpoint(checkpoint):
    model_params = dict(DEFAULT_OFFICIAL_MODEL_PARAMS)
    checkpoint_model_params = dict(checkpoint.get("model_params", {}))
    model_params["z_dim"] = int(checkpoint.get("z_dim", model_params["z_dim"]))
    model_params["force_first_move"] = bool(
        checkpoint.get("force_first_move", model_params["force_first_move"])
    )
    model_params.update(checkpoint_model_params)
    if model_params.get("use_candidate_features", False):
        model_params.setdefault("selected_candidate_feature_names", [])
        model_params.setdefault("relative_candidate_feature_names", [])
    else:
        model_params["selected_candidate_feature_names"] = list(
            model_params.get("selected_candidate_feature_names", [])
        )
        model_params["relative_candidate_feature_names"] = list(
            model_params.get("relative_candidate_feature_names", [])
        )
        if not model_params["selected_candidate_feature_names"]:
            model_params["candidate_scorer_type"] = "baseline_additive"
    model_params.setdefault("selected_node_static_feature_names", [])
    model_params.setdefault("node_static_embedding_mode", "concat")
    return model_params


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_params = infer_model_params_from_checkpoint(checkpoint)
    model = CVRPModel(**model_params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, model_params


def sample_seed_parallel_z(batch_size, z_samples, z_dim, seed, aug_factor, device):
    pool_size = 2 ** int(z_dim)
    replacement = int(z_samples) > pool_size
    probs = torch.full(
        (int(batch_size) * int(aug_factor), pool_size),
        1.0 / pool_size,
        dtype=torch.float32,
        device="cpu",
    )
    binary_string_pool = torch.tensor(
        [[(i >> bit) & 1 for bit in range(z_dim - 1, -1, -1)] for i in range(pool_size)],
        dtype=torch.float32,
        device="cpu",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    z_idx = torch.multinomial(probs, int(z_samples), replacement=replacement, generator=generator)
    z = binary_string_pool[z_idx]
    return z.to(device=device, dtype=torch.float32)


def sync_rollout_indices(env, rollout_size: int):
    env.rollout_size = int(rollout_size)
    device = env.depot_node_xy.device
    env.BATCH_IDX = torch.arange(env.batch_size, device=device)[:, None].expand(env.batch_size, env.rollout_size)
    env.ROLLOUT_IDX = torch.arange(env.rollout_size, device=device)[None, :].expand(env.batch_size, env.rollout_size)
    if hasattr(env, "depot_node_demand"):
        env.demand_rollout = env.depot_node_demand[:, None, :].expand(env.batch_size, env.rollout_size, -1)
    if torch.is_tensor(getattr(env, "dist_to_depot_norm", None)):
        env.dist_to_depot_rollout = env.dist_to_depot_norm[:, None, :].expand(env.batch_size, env.rollout_size, -1)
    env.step_state.BATCH_IDX = env.BATCH_IDX
    env.step_state.ROLLOUT_IDX = env.ROLLOUT_IDX


def clone_env_shallow(env):
    cloned = object.__new__(env.__class__)
    cloned.__dict__ = dict(env.__dict__)
    cloned.step_state = type(env.step_state)()
    cloned.step_state.__dict__ = dict(env.step_state.__dict__)
    cloned.reset_state = type(env.reset_state)()
    cloned.reset_state.__dict__ = dict(env.reset_state.__dict__)
    return cloned


def copy_dynamic_state(env):
    cloned = clone_env_shallow(env)
    cloned.selected_count = int(env.selected_count)
    dynamic_attrs = [
        "current_node",
        "selected_node_list",
        "at_the_depot",
        "load",
        "visited_ninf_flag",
        "ninf_mask",
        "finished",
    ]
    for attr in dynamic_attrs:
        value = getattr(env, attr, None)
        if torch.is_tensor(value):
            value = value.clone()
        setattr(cloned, attr, value)
    model_batch_index = getattr(env, "_model_batch_index", None)
    if torch.is_tensor(model_batch_index):
        cloned._model_batch_index = model_batch_index.clone()
    else:
        cloned._model_batch_index = model_batch_index
    sync_rollout_indices(cloned, env.rollout_size)
    return cloned


def repeat_rollout_tensor(value, repeat: int):
    if value is None:
        return None
    return value.repeat_interleave(int(repeat), dim=1)


def gather_rollout_tensor(value, gathering_index: torch.Tensor):
    if value is None:
        return None
    if value.ndim == 2:
        return value.gather(dim=1, index=gathering_index)
    expand_index = gathering_index
    for _ in range(2, value.ndim):
        expand_index = expand_index.unsqueeze(-1)
    expand_index = expand_index.expand(-1, -1, *value.shape[2:])
    return value.gather(dim=1, index=expand_index)


def repeat_bs_env(bs_env, repeat: int):
    rollout_env = clone_env_shallow(bs_env)
    rollout_env.selected_count = int(bs_env.selected_count)
    for attr in [
        "current_node",
        "selected_node_list",
        "at_the_depot",
        "load",
        "visited_ninf_flag",
        "ninf_mask",
        "finished",
    ]:
        setattr(rollout_env, attr, repeat_rollout_tensor(getattr(bs_env, attr, None), repeat))
    rollout_env._model_batch_index = getattr(bs_env, "_model_batch_index", None)
    sync_rollout_indices(rollout_env, bs_env.rollout_size * int(repeat))
    return rollout_env


def gather_rollout_env(source_env, gathering_index: torch.Tensor):
    gathered_env = clone_env_shallow(source_env)
    gathered_env.selected_count = int(source_env.selected_count)
    for attr in [
        "current_node",
        "selected_node_list",
        "at_the_depot",
        "load",
        "visited_ninf_flag",
        "ninf_mask",
        "finished",
    ]:
        setattr(
            gathered_env,
            attr,
            gather_rollout_tensor(getattr(source_env, attr, None), gathering_index),
        )
    gathered_env._model_batch_index = getattr(source_env, "_model_batch_index", None)
    sync_rollout_indices(gathered_env, gathering_index.size(1))
    return gathered_env


def merge_envs(left_env, right_env):
    merged_env = clone_env_shallow(left_env)
    merged_env.selected_count = int(left_env.selected_count)
    for attr in [
        "current_node",
        "selected_node_list",
        "at_the_depot",
        "load",
        "visited_ninf_flag",
        "ninf_mask",
        "finished",
    ]:
        left_value = getattr(left_env, attr, None)
        right_value = getattr(right_env, attr, None)
        if left_value is None:
            merged_value = right_value
        elif right_value is None:
            merged_value = left_value
        else:
            merged_value = torch.cat((left_value, right_value), dim=1)
        setattr(merged_env, attr, merged_value)
    left_index = getattr(left_env, "_model_batch_index", None)
    right_index = getattr(right_env, "_model_batch_index", None)
    if torch.is_tensor(left_index) and torch.is_tensor(right_index):
        if not torch.equal(left_index, right_index):
            raise ValueError("Merged envs must share the same model batch rows")
        merged_env._model_batch_index = left_index.clone()
    else:
        merged_env._model_batch_index = left_index if left_index is not None else right_index
    sync_rollout_indices(merged_env, left_env.rollout_size + right_env.rollout_size)
    return merged_env


def activate_model_cache(model, env, z):
    batch_index = getattr(env, "_model_batch_index", None)
    cache = {
        "encoded_nodes": model.encoded_nodes,
        "k": model.decoder.k,
        "v": model.decoder.v,
        "single_head_key": model.decoder.single_head_key,
        "z": model.decoder.z,
    }
    if hasattr(model.decoder, "node_embeddings"):
        cache["node_embeddings"] = model.decoder.node_embeddings
    if torch.is_tensor(batch_index):
        model.encoded_nodes = cache["encoded_nodes"].index_select(0, batch_index)
        model.decoder.k = cache["k"].index_select(0, batch_index)
        model.decoder.v = cache["v"].index_select(0, batch_index)
        model.decoder.single_head_key = cache["single_head_key"].index_select(0, batch_index)
        if "node_embeddings" in cache:
            model.decoder.node_embeddings = cache["node_embeddings"].index_select(0, batch_index)
    model.decoder.z = z
    return cache


def restore_model_cache(model, cache):
    model.encoded_nodes = cache["encoded_nodes"]
    model.decoder.k = cache["k"]
    model.decoder.v = cache["v"]
    model.decoder.single_head_key = cache["single_head_key"]
    if "node_embeddings" in cache:
        model.decoder.node_embeddings = cache["node_embeddings"]
    model.decoder.z = cache["z"]


def _amp_enabled(tensor):
    return torch.is_tensor(tensor) and tensor.is_cuda


def decode_scores(model, env, z):
    cache = activate_model_cache(model, env, z)
    try:
        state, _, _ = env.pre_step()
        encoded_last_node = _get_encoding(model.encoded_nodes, state.current_node)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=_amp_enabled(encoded_last_node)):
            scores = model.decoder(
                encoded_last_node,
                state.load,
                ninf_mask=state.ninf_mask,
            )
        return state, scores
    finally:
        restore_model_cache(model, cache)


def greedy_step(model, state):
    encoded_last_node = _get_encoding(model.encoded_nodes, state.current_node)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=_amp_enabled(encoded_last_node)):
        scores = model.decoder(
            encoded_last_node,
            state.load,
            ninf_mask=state.ninf_mask,
        )
    return scores.argmax(dim=2)


def greedy_finish(model, env, z):
    cache = activate_model_cache(model, env, z)
    try:
        done = bool(env.finished.all())
        reward = None
        while not done:
            state, _, _ = env.pre_step()
            selected = greedy_step(model, state)
            _, reward, done = env.step(selected)
        return reward.detach()
    finally:
        restore_model_cache(model, cache)


def select_initial_beams(model, env, z, beam_width: int):
    device = env.depot_node_xy.device
    depot_action = torch.zeros((env.batch_size, env.rollout_size), dtype=torch.long, device=device)
    state, _, _ = env.step(depot_action)
    cache = activate_model_cache(model, env, z)
    try:
        first_action = greedy_step(model, state)
    finally:
        restore_model_cache(model, cache)

    initial_env = copy_dynamic_state(env)
    initial_env.step(first_action)

    scoring_env = copy_dynamic_state(initial_env)
    full_reward = greedy_finish(model, scoring_env, z)
    top_reward, top_index = full_reward.topk(k=min(int(beam_width), full_reward.size(1)), dim=1)
    beam_env = gather_rollout_env(initial_env, top_index)
    beam_z = gather_rollout_tensor(z, top_index)
    return beam_env, beam_z, top_reward


def run_batch_sgbs(model, env, z, *, beam_width: int, expand_k: int):
    expansion_size_minus1 = max(1, int(expand_k) - 1)
    beam_env, beam_z, beam_reward = select_initial_beams(model, env, z, beam_width)
    done = bool(beam_env.finished.all())
    reward = None
    first_rollout_flag = True

    while not done:
        state, scores = decode_scores(model, beam_env, beam_z)
        select_k = expansion_size_minus1 if first_rollout_flag else expansion_size_minus1 + 1
        selected_score_all, selected_i_all = scores.topk(k=select_k, dim=2, largest=True, sorted=True)
        selected_valid_all = torch.isfinite(state.ninf_mask.gather(dim=2, index=selected_i_all))
        greedy_next_node = selected_i_all[:, :, 0]

        rollout_width = int(beam_width) * expansion_size_minus1
        if first_rollout_flag:
            score_selected = selected_score_all[:, :, :expansion_size_minus1]
            idx_selected = selected_i_all[:, :, :expansion_size_minus1]
            valid_selected = selected_valid_all[:, :, :expansion_size_minus1]
        else:
            score_selected = selected_score_all[:, :, 1 : expansion_size_minus1 + 1]
            idx_selected = selected_i_all[:, :, 1 : expansion_size_minus1 + 1]
            valid_selected = selected_valid_all[:, :, 1 : expansion_size_minus1 + 1]

        next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion_size_minus1)
        is_valid = valid_selected
        next_nodes[is_valid] = idx_selected[is_valid]

        rollout_env_pre = repeat_bs_env(beam_env, expansion_size_minus1)
        rollout_z = beam_z.repeat_interleave(expansion_size_minus1, dim=1)
        rollout_env = copy_dynamic_state(rollout_env_pre)

        next_nodes = next_nodes.reshape(beam_env.batch_size, rollout_width)
        cache = activate_model_cache(model, rollout_env, rollout_z)
        try:
            rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes)
            while not rollout_done:
                selected = greedy_step(model, rollout_state)
                rollout_state, rollout_reward, rollout_done = rollout_env.step(selected)
        finally:
            restore_model_cache(model, cache)

        is_redundant = (~is_valid).reshape(beam_env.batch_size, rollout_width)
        rollout_reward[is_redundant] = float("-inf")

        merged_env = rollout_env_pre
        merged_next_nodes = next_nodes
        merged_reward = rollout_reward
        merged_z = rollout_z
        if not first_rollout_flag:
            merged_env = merge_envs(rollout_env_pre, beam_env)
            merged_next_nodes = torch.cat((next_nodes, greedy_next_node), dim=1)
            merged_reward = torch.cat((rollout_reward, beam_reward), dim=1)
            merged_z = torch.cat((rollout_z, beam_z), dim=1)

        beam_reward, beam_index = merged_reward.topk(k=beam_width, dim=1, largest=True, sorted=True)

        beam_env = gather_rollout_env(merged_env, beam_index)
        beam_z = gather_rollout_tensor(merged_z, beam_index)
        selected = merged_next_nodes.gather(dim=1, index=beam_index)
        _, reward, done = beam_env.step(selected)
        first_rollout_flag = False

    return reward


def summarize_batch(reward, aug_factor: int, instance_batch_size: int, beam_width: int):
    aug_reward = reward.reshape(int(aug_factor), int(instance_batch_size), int(beam_width))
    max_beam_reward, beam_argmax = aug_reward.max(dim=2)
    max_aug_reward, aug_argmax = max_beam_reward.max(dim=0)
    best_beam = beam_argmax.gather(dim=0, index=aug_argmax.unsqueeze(0)).squeeze(0)
    return {
        "costs": (-max_aug_reward).detach().cpu(),
        "aug_index": aug_argmax.detach().cpu(),
        "beam_index": best_beam.detach().cpu(),
    }


def run_variant(
    name: str,
    checkpoint_path: Path,
    *,
    dataset_pkl: Path,
    num_instances: int,
    index_begin: int,
    instance_batch_size: int,
    beam_width: int,
    expand_k: int,
    aug_factor: int,
    z_samples: int,
    seed: int,
    device: torch.device,
):
    model, model_params = load_model(checkpoint_path, device)
    env = CVRPEnv(problem_size=infer_problem_size(dataset_pkl, index_begin))

    rows = []
    start = time.perf_counter()
    peak_mem_mb = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for batch_start in range(0, int(num_instances), int(instance_batch_size)):
            current_batch = min(int(instance_batch_size), int(num_instances) - batch_start)
            current_index = int(index_begin) + batch_start
            env.use_pkl_saved_problems(str(dataset_pkl), current_batch, index_begin=current_index)
            env.saved_depot_xy = env.saved_depot_xy.to(device)
            env.saved_node_xy = env.saved_node_xy.to(device)
            env.saved_node_demand = env.saved_node_demand.to(device)
            env.load_problems(batch_size=current_batch, rollout_size=int(z_samples), aug_factor=int(aug_factor))
            env._model_batch_index = torch.arange(env.batch_size, device=device, dtype=torch.long)
            reset_state, _, _ = env.reset()

            batch_seed = int(seed) + current_index
            z = sample_seed_parallel_z(
                batch_size=current_batch,
                z_samples=int(z_samples),
                z_dim=int(model_params["z_dim"]),
                seed=batch_seed,
                aug_factor=int(aug_factor),
                device=device,
            )
            model.pre_forward(reset_state, z)

            batch_t0 = time.perf_counter()
            reward = run_batch_sgbs(
                model,
                env,
                z,
                beam_width=int(beam_width),
                expand_k=int(expand_k),
            )
            batch_elapsed = time.perf_counter() - batch_t0
            batch_summary = summarize_batch(
                reward,
                aug_factor=int(aug_factor),
                instance_batch_size=current_batch,
                beam_width=int(beam_width),
            )

            for local_idx in range(current_batch):
                rows.append(
                    {
                        "instance": int(current_index + local_idx),
                        "cost": float(batch_summary["costs"][local_idx].item()),
                        "aug_index": int(batch_summary["aug_index"][local_idx].item()),
                        "beam_index": int(batch_summary["beam_index"][local_idx].item()),
                        "elapsed_sec": batch_elapsed / current_batch,
                    }
                )

    elapsed_sec = time.perf_counter() - start
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    mean_cost = sum(row["cost"] for row in rows) / len(rows)
    payload = {
        "variant": name,
        "checkpoint": str(checkpoint_path),
        "num_instances": len(rows),
        "index_begin": int(index_begin),
        "beam_width": int(beam_width),
        "expand_k": int(expand_k),
        "aug_factor": int(aug_factor),
        "z_samples": int(z_samples),
        "instance_batch_size": int(instance_batch_size),
        "seed": int(seed),
        "mean_cost": float(mean_cost),
        "elapsed_sec": float(elapsed_sec),
        "peak_mem_alloc_mb": peak_mem_mb,
        "rows": rows,
    }
    return payload


def main():
    args = parse_args()
    dataset_pkl = Path(args.dataset_pkl).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)

    result = run_variant(
        "polynet_sgbs",
        Path(args.checkpoint).resolve(),
        dataset_pkl=dataset_pkl,
        num_instances=int(args.num_instances),
        index_begin=int(args.index_begin),
        instance_batch_size=int(args.instance_batch_size),
        beam_width=int(args.beam_width),
        expand_k=int(args.expand_k),
        aug_factor=int(args.aug_factor),
        z_samples=int(args.z_samples),
        seed=int(args.seed),
        device=device,
    )
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "variant": result["variant"],
                "num_instances": result["num_instances"],
                "beam_width": result["beam_width"],
                "expand_k": result["expand_k"],
                "aug_factor": result["aug_factor"],
                "z_samples": result["z_samples"],
                "instance_batch_size": result["instance_batch_size"],
                "mean_cost": result["mean_cost"],
                "elapsed_sec": result["elapsed_sec"],
                "peak_mem_alloc_mb": result["peak_mem_alloc_mb"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
