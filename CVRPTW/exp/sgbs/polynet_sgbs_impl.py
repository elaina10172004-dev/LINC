"""PolyNet CVRPTW SGBS implementation."""

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
MODEL_CVRPTW = (ROOT / "CVRPTW" / "PolyNet").resolve()
for module_path in (ROOT, ROOT / "CVRPTW", MODEL_CVRPTW):
    module_path = str(module_path)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from CVRPTWModel import CVRPTWModel as Model, _get_encoding  # noqa: E402
from CVRPTWEnv import CVRPTWEnv  # noqa: E402


class Env(CVRPTWEnv):
    """SGBS state-management extension for the PolyNet CVRPTW env."""

    def modify_rollout_size(self, new_rollout_size):
        self.rollout_size = int(new_rollout_size)
        device = self.depot_node_xy.device
        self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(
            self.batch_size, self.rollout_size
        )
        self.ROLLOUT_IDX = torch.arange(self.rollout_size, device=device)[None, :].expand(
            self.batch_size, self.rollout_size
        )
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.ROLLOUT_IDX = self.ROLLOUT_IDX

    def reset_by_repeating_bs_env(self, bs_env, repeat):
        repeat = int(repeat)
        self.selected_count = int(bs_env.selected_count)
        self.current_node = _repeat_rollout_tensor(bs_env.current_node, repeat)
        self.selected_node_list = _repeat_rollout_tensor(bs_env.selected_node_list, repeat)
        self.at_the_depot = _repeat_rollout_tensor(bs_env.at_the_depot, repeat)
        self.load = _repeat_rollout_tensor(bs_env.load, repeat)
        self.visited_ninf_flag = _repeat_rollout_tensor(bs_env.visited_ninf_flag, repeat)
        self.ninf_mask = _repeat_rollout_tensor(bs_env.ninf_mask, repeat)
        self.finished = _repeat_rollout_tensor(bs_env.finished, repeat)
        self.time = _repeat_rollout_tensor(bs_env.time, repeat)
        self.used_vehicles = _repeat_rollout_tensor(bs_env.used_vehicles, repeat)
        self.modify_rollout_size(bs_env.rollout_size * repeat)
        self._sync_step_state()

    def reset_by_gathering_rollout_env(self, rollout_env, gathering_index):
        self.selected_count = int(rollout_env.selected_count)
        self.current_node = _gather_rollout_tensor(rollout_env.current_node, gathering_index)
        self.selected_node_list = _gather_rollout_tensor(rollout_env.selected_node_list, gathering_index)
        self.at_the_depot = _gather_rollout_tensor(rollout_env.at_the_depot, gathering_index)
        self.load = _gather_rollout_tensor(rollout_env.load, gathering_index)
        self.visited_ninf_flag = _gather_rollout_tensor(rollout_env.visited_ninf_flag, gathering_index)
        self.ninf_mask = _gather_rollout_tensor(rollout_env.ninf_mask, gathering_index)
        self.finished = _gather_rollout_tensor(rollout_env.finished, gathering_index)
        self.time = _gather_rollout_tensor(rollout_env.time, gathering_index)
        self.used_vehicles = _gather_rollout_tensor(rollout_env.used_vehicles, gathering_index)
        self.modify_rollout_size(gathering_index.size(1))
        self._sync_step_state()

    def merge(self, other_env):
        self.current_node = _cat_rollout_tensor(self.current_node, other_env.current_node)
        self.selected_node_list = _cat_rollout_tensor(self.selected_node_list, other_env.selected_node_list)
        self.at_the_depot = _cat_rollout_tensor(self.at_the_depot, other_env.at_the_depot)
        self.load = _cat_rollout_tensor(self.load, other_env.load)
        self.visited_ninf_flag = _cat_rollout_tensor(self.visited_ninf_flag, other_env.visited_ninf_flag)
        self.ninf_mask = _cat_rollout_tensor(self.ninf_mask, other_env.ninf_mask)
        self.finished = _cat_rollout_tensor(self.finished, other_env.finished)
        self.time = _cat_rollout_tensor(self.time, other_env.time)
        self.used_vehicles = _cat_rollout_tensor(self.used_vehicles, other_env.used_vehicles)
        self.modify_rollout_size(self.rollout_size + other_env.rollout_size)
        self._sync_step_state()

    def copy_dynamic_from(self, source_env):
        self.selected_count = int(source_env.selected_count)
        self.current_node = _copy_or_clone_tensor(self.current_node, source_env.current_node)
        self.selected_node_list = _copy_or_clone_tensor(self.selected_node_list, source_env.selected_node_list)
        self.at_the_depot = _copy_or_clone_tensor(self.at_the_depot, source_env.at_the_depot)
        self.load = _copy_or_clone_tensor(self.load, source_env.load)
        self.visited_ninf_flag = _copy_or_clone_tensor(self.visited_ninf_flag, source_env.visited_ninf_flag)
        self.ninf_mask = _copy_or_clone_tensor(self.ninf_mask, source_env.ninf_mask)
        self.finished = _copy_or_clone_tensor(self.finished, source_env.finished)
        self.time = _copy_or_clone_tensor(self.time, source_env.time)
        self.used_vehicles = _copy_or_clone_tensor(self.used_vehicles, source_env.used_vehicles)
        if self.rollout_size != source_env.rollout_size:
            self.modify_rollout_size(source_env.rollout_size)
        self._sync_step_state()

    def _sync_step_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.visited_mask = self.visited_ninf_flag == float("-inf")
        self.step_state.finished = self.finished
        self.step_state.time = self._get_model_time() if hasattr(self, "_get_model_time") else self.time / self.grid_size


def infer_model_params_from_checkpoint(checkpoint):
    if "model_params" in checkpoint:
        return dict(checkpoint["model_params"])
    return {
        "embedding_dim": 128,
        "poly_embedding_dim": 256,
        "sqrt_embedding_dim": 128**0.5,
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "eval_type": "argmax",
        "z_dim": int(checkpoint.get("z_dim", 16)),
        "use_fast_attention": True,
        "force_first_move": bool(checkpoint.get("force_first_move", False)),
        "use_candidate_features": False,
        "candidate_scorer_type": "baseline_additive",
        "use_depth_mixer": False,
        "use_gated_attention": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="SGBS CVRPTW benchmark for PolyNet checkpoints.")
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
    embedding_weight = checkpoint["model_state_dict"].get("encoder.embedding_node.weight")
    if torch.is_tensor(embedding_weight):
        model_params["node_feature_dim"] = int(embedding_weight.shape[1])
        if int(embedding_weight.shape[1]) >= 6:
            model_params["include_service_duration_in_node_embedding"] = True
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

    def slice_saved(value, default=None):
        if value is None:
            return default
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= start_idx + batch_size:
            return value[batch_slice]
        return value

    env.FLAG__use_saved_problems = True
    env.saved_depot_xy = dataset["depot_xy"][batch_slice]
    env.saved_node_xy = dataset["node_xy"][batch_slice]
    env.saved_node_demand = dataset["node_demand"][batch_slice]
    env.saved_node_tw = dataset["node_tw"][batch_slice]
    env.saved_depot_tw = slice_saved(dataset.get("depot_tw"))
    env.capacity = slice_saved(dataset["capacity"])
    env.saved_grid_size = slice_saved(dataset.get("grid_size", 1.0))
    service_value = slice_saved(dataset.get("service_t", dataset.get("service_duration", 0.0)), 0.0)
    env.saved_service_t = env._normalize_service_tensor(service_value, batch_size, device=torch.device("cpu"))
    travel_scale = slice_saved(dataset.get("travel_time_scale", 1.0), 1.0)
    env.saved_travel_time_scale = env._normalize_travel_time_scale(
        travel_scale, batch_size, device=torch.device("cpu")
    )
    env.grid_size = float(env._normalize_grid_size(env.saved_grid_size, batch_size, device=torch.device("cpu")).mean().item())
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
    probs = model.decoder(
        encoded_last_node,
        state.load,
        state.time,
        ninf_mask=state.ninf_mask,
    )
    if return_logits:
        return torch.where(torch.isfinite(state.ninf_mask), probs, torch.full_like(probs, float("-inf")))
    return probs


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


def _repeat_rollout_tensor(value, repeat):
    return None if value is None else value.repeat_interleave(int(repeat), dim=1)


def _cat_rollout_tensor(left_value, right_value):
    if left_value is None:
        return right_value
    if right_value is None:
        return left_value
    return torch.cat((left_value, right_value), dim=1)


def _copy_or_clone_tensor(current_value, source_value):
    if not torch.is_tensor(source_value):
        return source_value
    if torch.is_tensor(current_value) and current_value.shape == source_value.shape:
        current_value.copy_(source_value)
        return current_value
    return source_value.clone()


def _gather_rollout_tensor(value, gathering_index):
    if value is None:
        return None
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
        output_json = ROOT / "results" / "cvrptw_polynet_sgbs_eval.json"
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
    input_scale_mode = str(model_params.get("cvrptw_input_scale_mode", "grid"))
    env.input_scale_mode = input_scale_mode
    env.enforce_depot_return = True
    env.enable_candidate_features = bool(model_params.get("use_candidate_features", False))

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
        "input_scale_mode": input_scale_mode,
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
