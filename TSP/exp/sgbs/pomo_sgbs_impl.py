import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
POMO_ROOT = ROOT / "POMO"
if str(POMO_ROOT) not in sys.path:
    sys.path.insert(0, str(POMO_ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from TSPEnv import TSPEnv  # noqa: E402
from TSPModel import (  # noqa: E402
    TSPModel,
    TSP_Decoder,
    TSP_Encoder,
    _get_encoding,
    multi_head_attention,
    reshape_by_heads,
)


MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128**0.5,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
}


class SGBSTSPDecoder(TSP_Decoder):
    def forward(self, encoded_last_node, ninf_mask, first_node=None):
        head_num = self.model_params["head_num"]
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        if first_node is None:
            q_first = self.q_first
        else:
            qkv_dim = self.model_params["qkv_dim"]
            gather_index = first_node[:, None, :, None].expand(-1, head_num, -1, qkv_dim)
            q_first = self.q_first.gather(dim=2, index=gather_index)
        q = q_first + q_last
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)
        score = torch.matmul(mh_atten_out, self.single_head_key)
        score_scaled = score / self.model_params["sqrt_embedding_dim"]
        score_clipped = self.model_params["logit_clipping"] * torch.tanh(score_scaled)
        return F.softmax(score_clipped + ninf_mask, dim=2)


class SGBSTSPModel(TSPModel):
    def __init__(self, **model_params):
        nn.Module.__init__(self)
        self.model_params = model_params
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = SGBSTSPDecoder(**model_params)
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)
        self.decoder.set_kv(self.encoded_nodes)
        batch_size, problem_size = reset_state.problems.size(0), reset_state.problems.size(1)
        all_nodes = torch.arange(problem_size, device=reset_state.problems.device)[None, :].expand(batch_size, problem_size)
        encoded_first_node = _get_encoding(self.encoded_nodes, all_nodes)
        self.decoder.set_q1(encoded_first_node)

    def get_expand_prob(self, state):
        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        return self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, first_node=state.first_node)

    def forward(self, state, eval_type="greedy"):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        if state.current_node is None:
            selected = torch.arange(pomo_size, device=state.BATCH_IDX.device)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size), device=state.BATCH_IDX.device)
            return selected, prob

        encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
        probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, first_node=state.first_node)
        if self.training or eval_type == "softmax":
            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, pomo_size)
            prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
        else:
            selected = probs.argmax(dim=2)
            prob = None
        return selected, prob


def parse_args():
    parser = argparse.ArgumentParser(description="Official-style POMO SGBS for TSP.")
    parser.add_argument("--checkpoint", default="models/POMO/saved_tsp100_model2_longTrain/checkpoint-3100.pt")
    parser.add_argument("--data", "--dataset-pkl", dest="data", default="data/tsp100_test_seed1234.pkl")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--episodes", "--num-instances", dest="episodes", type=int, default=10000)
    parser.add_argument("--index-begin", type=int, default=0)
    parser.add_argument("--batch-size", "--instance-batch-size", dest="batch_size", type=int, default=400)
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


def prepare_saved_tsp_data(saved_problems):
    raw = saved_problems.to(dtype=torch.float32)
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


def summarize_tsplib_costs(final_env, raw_batch, aug_factor, beam_width):
    tours = final_env.selected_node_list[:, :, :final_env.problem_size]
    raw_aug = raw_batch.repeat(int(aug_factor), 1, 1)
    cost = tsplib_euc2d_cost(raw_aug, tours).reshape(int(aug_factor), raw_batch.size(0), int(beam_width))
    no_aug = cost[0].min(dim=1).values
    aug = cost.min(dim=2).values.min(dim=0).values
    return no_aug.detach().cpu(), aug.detach().cpu()


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
    model = SGBSTSPModel(**MODEL_PARAMS).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def env_step(env, selected):
    state, reward, done = env.step(selected)
    state.first_node = env.selected_node_list[:, :, 0]
    return state, reward, done


def set_pomo_size(env, width):
    env.pomo_size = int(width)
    device = env.problems.device
    env.BATCH_IDX = torch.arange(env.batch_size, device=device)[:, None].expand(env.batch_size, env.pomo_size)
    env.POMO_IDX = torch.arange(env.pomo_size, device=device)[None, :].expand(env.batch_size, env.pomo_size)
    if hasattr(env, "step_state") and env.step_state is not None:
        env.step_state.BATCH_IDX = env.BATCH_IDX
        env.step_state.POMO_IDX = env.POMO_IDX


def _blank_like(env):
    cloned = object.__new__(env.__class__)
    cloned.__dict__ = dict(env.__dict__)
    cloned.step_state = type(env.step_state)(BATCH_IDX=None, POMO_IDX=None)
    return cloned


def repeat_env(env, repeat):
    repeated = _blank_like(env)
    repeat = int(repeat)
    repeated.selected_count = int(env.selected_count)
    repeated.current_node = env.current_node.repeat_interleave(repeat, dim=1)
    repeated.selected_node_list = env.selected_node_list.repeat_interleave(repeat, dim=1)
    repeated.step_state.current_node = repeated.current_node
    repeated.step_state.ninf_mask = env.step_state.ninf_mask.repeat_interleave(repeat, dim=1)
    repeated.step_state.first_node = repeated.selected_node_list[:, :, 0]
    set_pomo_size(repeated, env.pomo_size * repeat)
    return repeated


def gather_env(env, gathering_index):
    gathered = _blank_like(env)
    width = int(gathering_index.size(1))
    gathered.selected_count = int(env.selected_count)
    gathered.current_node = env.current_node.gather(dim=1, index=gathering_index)
    list_index = gathering_index[:, :, None].expand(-1, -1, gathered.selected_count)
    gathered.selected_node_list = env.selected_node_list.gather(dim=1, index=list_index)
    mask_index = gathering_index[:, :, None].expand(-1, -1, env.problem_size)
    gathered.step_state.current_node = gathered.current_node
    gathered.step_state.ninf_mask = env.step_state.ninf_mask.gather(dim=1, index=mask_index)
    gathered.step_state.first_node = gathered.selected_node_list[:, :, 0]
    set_pomo_size(gathered, width)
    return gathered


def merge_envs(left, right):
    merged = _blank_like(left)
    merged.selected_count = int(left.selected_count)
    merged.current_node = torch.cat((left.current_node, right.current_node), dim=1)
    merged.selected_node_list = torch.cat((left.selected_node_list, right.selected_node_list), dim=1)
    merged.step_state.current_node = merged.current_node
    merged.step_state.ninf_mask = torch.cat((left.step_state.ninf_mask, right.step_state.ninf_mask), dim=1)
    merged.step_state.first_node = merged.selected_node_list[:, :, 0]
    set_pomo_size(merged, merged.current_node.size(1))
    return merged


def greedy_rollout(model, env, state, done, use_amp, reward=None):
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        while not done:
            selected, _ = model(state)
            state, reward, done = env_step(env, selected)
    return reward


def get_pomo_starting_points(model, env, problem_size, beam_width, start_mode, use_amp):
    set_pomo_size(env, problem_size)
    env.reset()
    state, reward, done = env.pre_step()
    reward = greedy_rollout(model, env, state, done, use_amp)
    start_count = int(problem_size) if start_mode == "all" else int(beam_width)
    start_count = min(start_count, int(reward.size(1)))
    return reward.topk(k=start_count, dim=1, largest=True, sorted=True).indices


def run_batch_sgbs(model, env, problem_size, beam_width, expand_k, aug_factor, start_mode, use_amp):
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    starting_points = get_pomo_starting_points(model, env, problem_size, beam_width, start_mode, use_amp)
    active_width = int(starting_points.size(1))
    set_pomo_size(env, active_width)
    env.reset()
    state, reward, done = env_step(env, starting_points)

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
        rollout_state, rollout_reward, rollout_done = env_step(rollout_env, flat_next_nodes)
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
        state, reward, done = env_step(env, selected)
        active_width = beam_width

    final_width = int(reward.size(1))
    aug_reward = reward.reshape(aug_factor, -1, final_width)
    no_aug_best = aug_reward[0].max(dim=1).values
    aug_best = aug_reward.max(dim=2).values.max(dim=0).values
    return -no_aug_best, -aug_best, env


def main():
    args = parse_args()
    device = configure_device(args)
    use_amp = device.type == "cuda" and args.amp == "on"
    checkpoint = resolve(args.checkpoint)
    data_path = resolve(args.data)
    model = load_model(checkpoint, device)
    env = TSPEnv(problem_size=args.problem_size, pomo_size=args.problem_size)
    raw_saved_problems = None
    use_tsplib_metric = False
    if data_path.exists():
        env.use_saved_problems(str(data_path))
        raw_saved_problems, env.saved_problems, use_tsplib_metric = prepare_saved_tsp_data(env.saved_problems)
        env.saved_index = int(args.index_begin)
        total = min(int(args.episodes), int(env.saved_problems.size(0)) - int(args.index_begin))
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
            set_pomo_size(env, args.problem_size) if env.problems is not None else None
            raw_batch = None
            if use_tsplib_metric and raw_saved_problems is not None:
                raw_batch = raw_saved_problems[env.saved_index:env.saved_index + bsz]
            env.load_problems(bsz, args.aug_factor)
            no_aug, aug, final_env = run_batch_sgbs(
                model,
                env,
                args.problem_size,
                int(args.beam_width),
                int(args.expand_k),
                int(args.aug_factor),
                args.start_mode,
                use_amp,
            )
            if use_tsplib_metric:
                no_aug, aug = summarize_tsplib_costs(final_env, raw_batch, int(args.aug_factor), int(args.beam_width))
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
        "problem": "tsp",
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
        "metric": "tsplib_euc2d" if use_tsplib_metric else "euclidean",
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
    summary_keys = ("episodes", "score_mean", "aug_score_mean", "elapsed_sec", "peak_memory_mb", "metric")
    print(json.dumps({k: payload[k] for k in summary_keys}, indent=2))


if __name__ == "__main__":
    main()
