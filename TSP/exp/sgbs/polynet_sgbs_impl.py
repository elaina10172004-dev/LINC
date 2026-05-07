"""PolyNet TSP SGBS implementation."""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TSP_ROOT = ROOT
POLYNET_ROOT = TSP_ROOT / "PolyNet"

if str(POLYNET_ROOT) not in sys.path:
    sys.path.insert(0, str(POLYNET_ROOT))
if str(TSP_ROOT) not in sys.path:
    sys.path.insert(0, str(TSP_ROOT))

from TSPEnv import TSPEnv  # noqa: E402
from TSPModel import TSPModel, _get_encoding  # noqa: E402


DEFAULT_MODEL_PARAMS = {
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
    "use_candidate_features": False,
    "candidate_scorer_type": "baseline_additive",
}


def parse_args():
    parser = argparse.ArgumentParser(description="PolyNet TSP SGBS benchmark.")
    parser.add_argument("--ours-checkpoint", type=str, required=True)
    parser.add_argument("--dataset-pkl", type=str, default=str(TSP_ROOT / "data" / "tsp100_test_seed1234.pkl"))
    parser.add_argument("--num-instances", type=int, default=10000)
    parser.add_argument("--index-begin", type=int, default=0)
    parser.add_argument("--instance-batch-size", type=int, default=1000)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--aug-factor", type=int, default=8)
    parser.add_argument("--z-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-json", type=str, default=str(ROOT / "results" / "tsp_polynet_sgbs.json"))
    return parser.parse_args()


def infer_model_params_from_checkpoint(checkpoint):
    mp = dict(DEFAULT_MODEL_PARAMS)
    mp["z_dim"] = int(checkpoint.get("z_dim", mp["z_dim"]))
    mp["force_first_move"] = bool(checkpoint.get("force_first_move", mp["force_first_move"]))
    return mp


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mp = infer_model_params_from_checkpoint(checkpoint)
    model = TSPModel(**mp).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if hasattr(model, "set_module_slow_start_progress"):
        model.set_module_slow_start_progress(1.0)
    return model, mp


def infer_problem_size(dataset_pkl: Path, index_begin: int = 0) -> int:
    with dataset_pkl.open("rb") as f:
        data = pickle.load(f)
    sample = data[int(index_begin)]
    return int(torch.as_tensor(sample).shape[-2])


def prepare_tsp_batch(batch_data, device):
    raw = torch.tensor(batch_data, dtype=torch.float32, device=device)
    coord_scale = raw.amax(dim=(1, 2)).clamp_min(1.0)
    use_tsplib_metric = bool((coord_scale > 2.0).any().item())
    model_input = raw / coord_scale[:, None, None] if use_tsplib_metric else raw
    return raw, model_input, use_tsplib_metric


def tsplib_euc2d_cost(raw_coords, tours):
    batch_size, rollout_size, problem_size = tours.shape
    coords = raw_coords[:, None, :, :].expand(batch_size, rollout_size, problem_size, 2)
    ordered = coords.gather(dim=2, index=tours[:, :, :, None].expand(-1, -1, -1, 2))
    segment = ordered[:, :, 1:, :] - ordered[:, :, :-1, :]
    closing = ordered[:, :, :1, :] - ordered[:, :, -1:, :]
    edges = torch.cat((segment, closing), dim=2)
    return torch.floor(torch.linalg.vector_norm(edges, dim=-1) + 0.5).sum(dim=2)


def sample_seed_parallel_z(batch_size, z_samples, z_dim, seed, aug_factor, device):
    pool_size = 2 ** int(z_dim)
    replacement = int(z_samples) > pool_size
    probs = torch.full((int(batch_size) * int(aug_factor), pool_size), 1.0 / pool_size, dtype=torch.float32, device="cpu")
    binary_string_pool = torch.tensor(
        [[(i >> bit) & 1 for bit in range(z_dim - 1, -1, -1)] for i in range(pool_size)],
        dtype=torch.float32, device="cpu",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    z_idx = torch.multinomial(probs, int(z_samples), replacement=replacement, generator=generator)
    return binary_string_pool[z_idx].to(device=device, dtype=torch.float32)


# SGBS core. PolyNet uses the same search logic as LINC, without candidate features.

def sync_rollout_indices(env, rollout_size: int):
    env.rollout_size = int(rollout_size)
    device = env.problems.device
    env.BATCH_IDX = torch.arange(env.batch_size, device=device)[:, None].expand(env.batch_size, env.rollout_size)
    env.ROLLOUT_IDX = torch.arange(env.rollout_size, device=device)[None, :].expand(env.batch_size, env.rollout_size)
    env.step_state.BATCH_IDX = env.BATCH_IDX
    env.step_state.ROLLOUT_IDX = env.ROLLOUT_IDX
    if getattr(env, "dist_to_centroid_norm", None) is not None:
        env.dist_to_centroid_rollout = env.dist_to_centroid_norm[:, None, :].expand(env.batch_size, env.rollout_size, -1)
    if hasattr(env, "start_dist_norm"):
        env.start_dist_norm = None


def clone_env_shallow(env):
    cloned = object.__new__(env.__class__)
    cloned.__dict__ = dict(env.__dict__)
    cloned.step_state = object.__new__(env.step_state.__class__)
    cloned.step_state.__dict__.update(dict(env.step_state.__dict__))
    if hasattr(env, "reset_state") and env.reset_state is not None:
        cloned.reset_state = object.__new__(env.reset_state.__class__)
        cloned.reset_state.__dict__.update(dict(env.reset_state.__dict__))
    return cloned


def copy_dynamic_state(env):
    cloned = clone_env_shallow(env)
    cloned.selected_count = int(env.selected_count)
    for attr in ["current_node", "selected_node_list"]:
        value = getattr(env, attr, None)
        if torch.is_tensor(value):
            value = value.clone()
        setattr(cloned, attr, value)
    cloned.step_state.current_node = None if env.step_state.current_node is None else env.step_state.current_node.clone()
    set_ninf_mask(cloned, env.step_state.ninf_mask.clone())
    cloned.step_state.candidate_features = None
    model_batch_index = getattr(env, "_model_batch_index", None)
    cloned._model_batch_index = model_batch_index.clone() if torch.is_tensor(model_batch_index) else model_batch_index
    sync_rollout_indices(cloned, env.rollout_size)
    return cloned


def repeat_rollout_tensor(value, repeat: int):
    return None if value is None else value.repeat_interleave(int(repeat), dim=1)


def gather_rollout_tensor(value, gathering_index: torch.Tensor):
    if value is None:
        return None
    if value.ndim == 2:
        return value.gather(dim=1, index=gathering_index)
    expand_index = gathering_index
    for _ in range(2, value.ndim):
        expand_index = expand_index.unsqueeze(-1)
    return value.gather(dim=1, index=expand_index.expand(-1, -1, *value.shape[2:]))


def set_ninf_mask(env, mask: torch.Tensor):
    env.step_state.ninf_mask = mask
    if hasattr(env, "ninf_mask"):
        env.ninf_mask = mask


def repeat_bs_env(bs_env, repeat: int):
    rollout_env = clone_env_shallow(bs_env)
    rollout_env.selected_count = int(bs_env.selected_count)
    rollout_env.current_node = repeat_rollout_tensor(bs_env.current_node, repeat)
    rollout_env.selected_node_list = repeat_rollout_tensor(bs_env.selected_node_list, repeat)
    rollout_env.step_state.current_node = repeat_rollout_tensor(bs_env.step_state.current_node, repeat)
    set_ninf_mask(rollout_env, repeat_rollout_tensor(bs_env.step_state.ninf_mask, repeat))
    rollout_env.step_state.candidate_features = None
    rollout_env._model_batch_index = getattr(bs_env, "_model_batch_index", None)
    sync_rollout_indices(rollout_env, bs_env.rollout_size * int(repeat))
    return rollout_env


def gather_rollout_env(source_env, gathering_index: torch.Tensor):
    gathered_env = clone_env_shallow(source_env)
    gathered_env.selected_count = int(source_env.selected_count)
    gathered_env.current_node = gather_rollout_tensor(source_env.current_node, gathering_index)
    gathered_env.selected_node_list = gather_rollout_tensor(source_env.selected_node_list, gathering_index)
    gathered_env.step_state.current_node = gather_rollout_tensor(source_env.step_state.current_node, gathering_index)
    set_ninf_mask(gathered_env, gather_rollout_tensor(source_env.step_state.ninf_mask, gathering_index))
    gathered_env.step_state.candidate_features = None
    gathered_env._model_batch_index = getattr(source_env, "_model_batch_index", None)
    sync_rollout_indices(gathered_env, gathering_index.size(1))
    return gathered_env


def merge_envs(left_env, right_env):
    merged_env = clone_env_shallow(left_env)
    merged_env.selected_count = int(left_env.selected_count)
    merged_env.current_node = torch.cat((left_env.current_node, right_env.current_node), dim=1)
    merged_env.selected_node_list = torch.cat((left_env.selected_node_list, right_env.selected_node_list), dim=1)
    merged_env.step_state.current_node = merged_env.current_node
    set_ninf_mask(merged_env, torch.cat((left_env.step_state.ninf_mask, right_env.step_state.ninf_mask), dim=1))
    merged_env.step_state.candidate_features = None
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
        "k": model.decoder.k, "v": model.decoder.v,
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


def set_q1_from_env(model, env):
    first_node = env.selected_node_list[:, :, 0]
    encoded_first_node = _get_encoding(model.encoded_nodes, first_node)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=encoded_first_node.is_cuda):
        model.decoder.set_q1(encoded_first_node)


def decode_scores(model, env, z):
    cache = activate_model_cache(model, env, z)
    try:
        state, _, _ = env.pre_step()
        set_q1_from_env(model, env)
        encoded_last_node = _get_encoding(model.encoded_nodes, state.current_node)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=encoded_last_node.is_cuda):
            scores = model.decoder(encoded_last_node, ninf_mask=state.ninf_mask, return_logits=True)
        return state, scores
    finally:
        restore_model_cache(model, cache)


def greedy_finish(model, env, z):
    cache = activate_model_cache(model, env, z)
    try:
        done = env.selected_count == env.problem_size
        reward = None
        while not done:
            state, _, _ = env.pre_step()
            if env.current_node is None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=env.problems.is_cuda):
                    selected, _ = model(state, greedy_construction=True)
            else:
                set_q1_from_env(model, env)
                encoded_last_node = _get_encoding(model.encoded_nodes, state.current_node)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=encoded_last_node.is_cuda):
                    scores = model.decoder(encoded_last_node, ninf_mask=state.ninf_mask, return_logits=True)
                selected = scores.argmax(dim=2)
            _, reward, done = env.step(selected)
        return reward.detach()
    finally:
        restore_model_cache(model, cache)


def select_initial_beams(model, env, z, beam_width: int):
    state, _, _ = env.pre_step()
    cache = activate_model_cache(model, env, z)
    try:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=env.problems.is_cuda):
            first_action, _ = model(state, greedy_construction=True)
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
    done = beam_env.selected_count == beam_env.problem_size
    reward = None
    first_rollout_flag = True

    while not done:
        state, scores = decode_scores(model, beam_env, beam_z)
        select_k = expansion_size_minus1 if first_rollout_flag else expansion_size_minus1 + 1
        selected_score_all, selected_i_all = scores.topk(k=select_k, dim=2, largest=True, sorted=True)
        greedy_next_node = selected_i_all[:, :, 0]

        rollout_width = int(beam_width) * expansion_size_minus1
        if first_rollout_flag:
            score_selected = selected_score_all[:, :, :expansion_size_minus1]
            idx_selected = selected_i_all[:, :, :expansion_size_minus1]
        else:
            score_selected = selected_score_all[:, :, 1:expansion_size_minus1 + 1]
            idx_selected = selected_i_all[:, :, 1:expansion_size_minus1 + 1]

        next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion_size_minus1)
        is_valid = torch.isfinite(score_selected)
        next_nodes[is_valid] = idx_selected[is_valid]

        rollout_env_pre = repeat_bs_env(beam_env, expansion_size_minus1)
        rollout_z = beam_z.repeat_interleave(expansion_size_minus1, dim=1)
        rollout_env = copy_dynamic_state(rollout_env_pre)
        next_nodes = next_nodes.reshape(beam_env.batch_size, rollout_width)

        cache = activate_model_cache(model, rollout_env, rollout_z)
        try:
            rollout_state, rollout_reward, rollout_done = rollout_env.step(next_nodes)
            while not rollout_done:
                set_q1_from_env(model, rollout_env)
                encoded_last_node = _get_encoding(model.encoded_nodes, rollout_state.current_node)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=encoded_last_node.is_cuda):
                    scores = model.decoder(encoded_last_node, ninf_mask=rollout_state.ninf_mask, return_logits=True)
                selected = scores.argmax(dim=2)
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
        _, reward, _ = beam_env.step(selected)
        done = beam_env.selected_count == beam_env.problem_size
        first_rollout_flag = False

    return reward, beam_env


def summarize_batch(reward, aug_factor: int, instance_batch_size: int, beam_width: int, final_env=None, raw_batch=None, use_tsplib_metric=False):
    if use_tsplib_metric:
        if final_env is None or raw_batch is None:
            raise ValueError("TSPLIB metric requires final_env and raw_batch")
        tours = final_env.selected_node_list[:, :, :final_env.problem_size]
        raw_aug = raw_batch.repeat(int(aug_factor), 1, 1)
        cost = tsplib_euc2d_cost(raw_aug, tours).reshape(int(aug_factor), int(instance_batch_size), int(beam_width))
        no_aug_cost, no_aug_beam = cost[0].min(dim=1)
        best_beam_cost, beam_argmin = cost.min(dim=2)
        best_cost, aug_argmin = best_beam_cost.min(dim=0)
        best_beam = beam_argmin.gather(dim=0, index=aug_argmin.unsqueeze(0)).squeeze(0)
        return {
            "costs": best_cost.detach().cpu(),
            "no_aug_costs": no_aug_cost.detach().cpu(),
            "aug_index": aug_argmin.detach().cpu(),
            "beam_index": best_beam.detach().cpu(),
            "no_aug_beam_index": no_aug_beam.detach().cpu(),
        }
    aug_reward = reward.reshape(int(aug_factor), int(instance_batch_size), int(beam_width))
    max_beam_reward, beam_argmax = aug_reward.max(dim=2)
    max_aug_reward, aug_argmax = max_beam_reward.max(dim=0)
    best_beam = beam_argmax.gather(dim=0, index=aug_argmax.unsqueeze(0)).squeeze(0)
    no_aug_reward, no_aug_beam = aug_reward[0].max(dim=1)
    return {
        "costs": (-max_aug_reward).detach().cpu(),
        "no_aug_costs": (-no_aug_reward).detach().cpu(),
        "aug_index": aug_argmax.detach().cpu(),
        "beam_index": best_beam.detach().cpu(),
        "no_aug_beam_index": no_aug_beam.detach().cpu(),
    }


def run_variant(checkpoint_path, *, dataset_pkl, num_instances, index_begin, instance_batch_size,
                beam_width, expand_k, aug_factor, z_samples, seed, device):
    model, mp = load_model(checkpoint_path, device)
    env = TSPEnv(problem_size=infer_problem_size(dataset_pkl, index_begin))

    rows = []
    used_tsplib_metric = False
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with open(dataset_pkl, "rb") as f:
        full_data = pickle.load(f)

    with torch.no_grad():
        for batch_start in range(0, int(num_instances), int(instance_batch_size)):
            current_batch = min(int(instance_batch_size), int(num_instances) - batch_start)
            current_index = int(index_begin) + batch_start
            batch_data = full_data[current_index:current_index + current_batch]
            raw_batch, model_batch, use_tsplib_metric = prepare_tsp_batch(batch_data, device)
            used_tsplib_metric = used_tsplib_metric or use_tsplib_metric
            env.FLAG__use_saved_problems = True
            env.saved_problems = model_batch
            env.saved_index = 0
            env.load_problems(batch_size=current_batch, rollout_size=int(z_samples), aug_factor=int(aug_factor))
            env._model_batch_index = torch.arange(env.batch_size, device=device, dtype=torch.long)
            reset_state, _, _ = env.reset()

            z = sample_seed_parallel_z(current_batch, int(z_samples), mp["z_dim"], int(seed) + current_index, int(aug_factor), device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                model.pre_forward(reset_state, z)

            batch_t0 = time.perf_counter()
            reward, final_env = run_batch_sgbs(model, env, z, beam_width=int(beam_width), expand_k=int(expand_k))
            batch_elapsed = time.perf_counter() - batch_t0
            batch_summary = summarize_batch(
                reward,
                int(aug_factor),
                current_batch,
                int(beam_width),
                final_env=final_env,
                raw_batch=raw_batch,
                use_tsplib_metric=use_tsplib_metric,
            )

            for local_idx in range(current_batch):
                rows.append({
                    "instance": int(current_index + local_idx),
                    "cost": float(batch_summary["costs"][local_idx].item()),
                    "no_aug_cost": float(batch_summary["no_aug_costs"][local_idx].item()),
                    "aug_index": int(batch_summary["aug_index"][local_idx].item()),
                    "beam_index": int(batch_summary["beam_index"][local_idx].item()),
                    "elapsed_sec": batch_elapsed / current_batch,
                })

    elapsed_sec = time.perf_counter() - start
    peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else None
    mean_cost = sum(row["cost"] for row in rows) / len(rows)
    metric = "tsplib_euc2d" if used_tsplib_metric else "euclidean"
    return {
        "checkpoint": str(checkpoint_path), "num_instances": len(rows), "index_begin": int(index_begin),
        "beam_width": int(beam_width), "expand_k": int(expand_k), "aug_factor": int(aug_factor),
        "z_samples": int(z_samples), "instance_batch_size": int(instance_batch_size), "seed": int(seed),
        "metric": metric,
        "mean_cost": float(mean_cost), "elapsed_sec": float(elapsed_sec), "peak_mem_alloc_mb": peak_mem_mb, "rows": rows,
    }


def main():
    args = parse_args()
    dataset_pkl = Path(args.dataset_pkl).resolve()
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    result = run_variant(Path(args.ours_checkpoint).resolve(), dataset_pkl=dataset_pkl,
                         num_instances=int(args.num_instances), index_begin=int(args.index_begin),
                         instance_batch_size=int(args.instance_batch_size), beam_width=int(args.beam_width),
                         expand_k=int(args.expand_k), aug_factor=int(args.aug_factor),
                         z_samples=int(args.z_samples), seed=int(args.seed), device=device)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("num_instances", "beam_width", "expand_k", "aug_factor", "z_samples",
                                               "instance_batch_size", "mean_cost", "elapsed_sec", "peak_mem_alloc_mb")}, indent=2))


if __name__ == "__main__":
    main()
