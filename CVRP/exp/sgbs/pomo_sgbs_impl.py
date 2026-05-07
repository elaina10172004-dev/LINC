import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
POMO_ROOT = ROOT / "POMO"
if str(POMO_ROOT) not in sys.path:
    sys.path.insert(0, str(POMO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CVRPEnv import CVRPEnv  # noqa: E402
from CVRPModel import CVRPModel  # noqa: E402


MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128**0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Official-style POMO SGBS for CVRP.")
    parser.add_argument("--checkpoint", default="models/POMO/saved_CVRP100_model/checkpoint-30500.pt")
    parser.add_argument("--data", "--dataset-pkl", dest="data", default="data/vrp100_test_seed1234.pkl")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--episodes", "--num-instances", dest="episodes", type=int, default=10000)
    parser.add_argument("--index-begin", type=int, default=0)
    parser.add_argument("--batch-size", "--instance-batch-size", dest="batch_size", type=int, default=625)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--start-mode", choices=("topk", "all"), default="topk")
    parser.add_argument("--aug", "--aug-factor", dest="aug_factor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--amp", choices=("on", "off"), default="on")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def configure_device(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.cuda_device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        return torch.device("cuda", args.cuda_device)
    torch.set_default_tensor_type("torch.FloatTensor")
    return torch.device("cpu")


def load_model(checkpoint_path, device):
    model = CVRPModel(**MODEL_PARAMS).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def set_pomo_size(env, width):
    env.pomo_size = int(width)
    device = env.depot_node_xy.device
    env.BATCH_IDX = torch.arange(env.batch_size, device=device)[:, None].expand(env.batch_size, env.pomo_size)
    env.POMO_IDX = torch.arange(env.pomo_size, device=device)[None, :].expand(env.batch_size, env.pomo_size)
    env.demand_rollout = env.depot_node_demand[:, None, :].expand(env.batch_size, env.pomo_size, -1)
    env.step_state.BATCH_IDX = env.BATCH_IDX
    env.step_state.POMO_IDX = env.POMO_IDX


def _blank_like(env):
    cloned = object.__new__(env.__class__)
    cloned.__dict__ = dict(env.__dict__)
    cloned.step_state = type(env.step_state)()
    return cloned


def repeat_env(env, repeat):
    repeated = _blank_like(env)
    repeat = int(repeat)
    repeated.selected_count = int(env.selected_count)
    repeated.current_node = env.current_node.repeat_interleave(repeat, dim=1)
    repeated.selected_node_list = env.selected_node_list.repeat_interleave(repeat, dim=1)
    repeated.at_the_depot = env.at_the_depot.repeat_interleave(repeat, dim=1)
    repeated.load = env.load.repeat_interleave(repeat, dim=1)
    repeated.visited_ninf_flag = env.visited_ninf_flag.repeat_interleave(repeat, dim=1)
    repeated.ninf_mask = env.ninf_mask.repeat_interleave(repeat, dim=1)
    repeated.finished = env.finished.repeat_interleave(repeat, dim=1)
    repeated.step_state.selected_count = repeated.selected_count
    repeated.step_state.load = repeated.load
    repeated.step_state.current_node = repeated.current_node
    repeated.step_state.ninf_mask = repeated.ninf_mask
    repeated.step_state.finished = repeated.finished
    set_pomo_size(repeated, env.pomo_size * repeat)
    return repeated


def gather_env(env, gathering_index):
    gathered = _blank_like(env)
    width = int(gathering_index.size(1))
    gathered.selected_count = int(env.selected_count)
    gathered.current_node = env.current_node.gather(dim=1, index=gathering_index)
    list_index = gathering_index[:, :, None].expand(-1, -1, env.selected_node_list.size(2))
    gathered.selected_node_list = env.selected_node_list.gather(dim=1, index=list_index)
    gathered.at_the_depot = env.at_the_depot.gather(dim=1, index=gathering_index)
    gathered.load = env.load.gather(dim=1, index=gathering_index)
    mask_index = gathering_index[:, :, None].expand(-1, -1, env.problem_size + 1)
    gathered.visited_ninf_flag = env.visited_ninf_flag.gather(dim=1, index=mask_index)
    gathered.ninf_mask = env.ninf_mask.gather(dim=1, index=mask_index)
    gathered.finished = env.finished.gather(dim=1, index=gathering_index)
    gathered.step_state.selected_count = gathered.selected_count
    gathered.step_state.load = gathered.load
    gathered.step_state.current_node = gathered.current_node
    gathered.step_state.ninf_mask = gathered.ninf_mask
    gathered.step_state.finished = gathered.finished
    set_pomo_size(gathered, width)
    return gathered


def merge_envs(left, right):
    merged = _blank_like(left)
    merged.selected_count = int(left.selected_count)
    merged.current_node = torch.cat((left.current_node, right.current_node), dim=1)
    merged.selected_node_list = torch.cat((left.selected_node_list, right.selected_node_list), dim=1)
    merged.at_the_depot = torch.cat((left.at_the_depot, right.at_the_depot), dim=1)
    merged.load = torch.cat((left.load, right.load), dim=1)
    merged.visited_ninf_flag = torch.cat((left.visited_ninf_flag, right.visited_ninf_flag), dim=1)
    merged.ninf_mask = torch.cat((left.ninf_mask, right.ninf_mask), dim=1)
    merged.finished = torch.cat((left.finished, right.finished), dim=1)
    merged.step_state.selected_count = merged.selected_count
    merged.step_state.load = merged.load
    merged.step_state.current_node = merged.current_node
    merged.step_state.ninf_mask = merged.ninf_mask
    merged.step_state.finished = merged.finished
    set_pomo_size(merged, merged.current_node.size(1))
    return merged


def greedy_rollout(model, env, state, done, use_amp, reward=None):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        while not done:
            selected, _ = model(state)
            state, reward, done = env.step(selected)
    return reward


def get_pomo_starting_points(model, env, problem_size, beam_width, start_mode, use_amp):
    set_pomo_size(env, problem_size)
    env.reset()
    state, reward, done = env.pre_step()
    reward = greedy_rollout(model, env, state, done, use_amp)
    start_count = int(problem_size) if start_mode == "all" else int(beam_width)
    start_count = min(start_count, int(reward.size(1)))
    selected = reward.topk(k=start_count, dim=1, largest=True, sorted=True).indices
    return selected + 1


def run_batch_sgbs(model, env, problem_size, beam_width, expand_k, aug_factor, start_mode, use_amp):
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    starting_points = get_pomo_starting_points(model, env, problem_size, beam_width, start_mode, use_amp)
    active_width = int(starting_points.size(1))
    set_pomo_size(env, active_width)
    env.reset()
    selected = torch.zeros(size=(env.batch_size, env.pomo_size), dtype=torch.long, device=env.depot_node_xy.device)
    state, _, done = env.step(selected)
    state, reward, done = env.step(starting_points)

    expansion = max(1, int(expand_k) - 1)
    first_rollout = True
    beam_reward = None

    while not done:
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            probs = model.get_expand_prob(state)
        select_k = expansion if first_rollout else expansion + 1
        selected_prob_all, selected_i_all = probs.topk(k=select_k, dim=2, largest=True, sorted=True)
        greedy_next_node = selected_i_all[:, :, 0]
        if first_rollout:
            prob_selected = selected_prob_all[:, :, :expansion]
            idx_selected = selected_i_all[:, :, :expansion]
        else:
            prob_selected = selected_prob_all[:, :, 1 : expansion + 1]
            idx_selected = selected_i_all[:, :, 1 : expansion + 1]

        next_nodes = greedy_next_node[:, :, None].repeat(1, 1, expansion)
        is_valid = prob_selected > 0
        next_nodes[is_valid] = idx_selected[is_valid]

        rollout_env = repeat_env(env, expansion)
        rollout_env_pre = repeat_env(env, expansion)
        flat_next_nodes = next_nodes.reshape(env.batch_size, active_width * expansion)
        rollout_state, rollout_reward, rollout_done = rollout_env.step(flat_next_nodes)
        rollout_reward = greedy_rollout(model, rollout_env, rollout_state, rollout_done, use_amp, rollout_reward)
        rollout_reward[(~is_valid).reshape(env.batch_size, active_width * expansion)] = float("-inf")

        if first_rollout:
            merged_env = rollout_env_pre
            merged_reward = rollout_reward
            merged_next_nodes = flat_next_nodes
            first_rollout = False
        else:
            merged_env = merge_envs(rollout_env_pre, env)
            merged_reward = torch.cat((rollout_reward, beam_reward), dim=1)
            merged_next_nodes = torch.cat((flat_next_nodes, greedy_next_node), dim=1)

        beam_reward, beam_index = merged_reward.topk(k=beam_width, dim=1, largest=True, sorted=True)
        env = gather_env(merged_env, beam_index)
        selected = merged_next_nodes.gather(dim=1, index=beam_index)
        state, reward, done = env.step(selected)
        active_width = beam_width

    final_width = int(reward.size(1))
    aug_reward = reward.reshape(aug_factor, -1, final_width)
    no_aug_best = aug_reward[0].max(dim=1).values
    aug_best = aug_reward.max(dim=2).values.max(dim=0).values
    return -no_aug_best, -aug_best


def main():
    args = parse_args()
    device = configure_device(args)
    use_amp = device.type == "cuda" and args.amp == "on"
    checkpoint = resolve(args.checkpoint)
    data_path = resolve(args.data)
    model = load_model(checkpoint, device)
    env = CVRPEnv(problem_size=args.problem_size, pomo_size=args.problem_size)
    if data_path.exists():
        env.use_saved_problems(str(data_path), device)
        env.saved_index = int(args.index_begin)
        total = min(int(args.episodes), int(env.saved_node_xy.size(0)) - int(args.index_begin))
    else:
        print(f"[warning] dataset not found, using random generated problems: {data_path}", flush=True)
        total = int(args.episodes)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    rows = []
    score_sum = 0.0
    aug_sum = 0.0

    with torch.inference_mode():
        offset = 0
        while offset < total:
            bsz = min(int(args.batch_size), total - offset)
            env.pomo_size = args.problem_size
            env.load_problems(bsz, args.aug_factor)
            no_aug, aug = run_batch_sgbs(
                model,
                env,
                args.problem_size,
                int(args.beam_width),
                int(args.expand_k),
                int(args.aug_factor),
                args.start_mode,
                use_amp,
            )
            score_sum += float(no_aug.sum().item())
            aug_sum += float(aug.sum().item())
            for local_idx in range(bsz):
                rows.append(
                    {
                        "index": int(args.index_begin + offset + local_idx),
                        "cost": float(no_aug[local_idx].item()),
                        "aug_cost": float(aug[local_idx].item()),
                    }
                )
            offset += bsz
            print(f"[progress] {offset}/{total}", flush=True)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak = None
    elapsed = time.perf_counter() - start
    payload = {
        "method": "pomo_sgbs",
        "problem": "cvrp",
        "checkpoint": str(checkpoint),
        "dataset": str(data_path),
        "seed": int(args.seed),
        "problem_size": int(args.problem_size),
        "episodes": len(rows),
        "batch_size": int(args.batch_size),
        "beam_width": int(args.beam_width),
        "expand_k": int(args.expand_k),
        "start_mode": args.start_mode,
        "aug_factor": int(args.aug_factor),
        "score_mean": score_sum / max(1, len(rows)),
        "aug_score_mean": aug_sum / max(1, len(rows)),
        "elapsed_sec": elapsed,
        "peak_memory_mb": peak,
        "rows": rows,
    }
    if args.output_json:
        output = resolve(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("episodes", "score_mean", "aug_score_mean", "elapsed_sec", "peak_memory_mb")}, indent=2))


if __name__ == "__main__":
    main()
