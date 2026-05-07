"""CVRPTW POMO SGBS runner.

The RL4CO POMO checkpoint is loaded into the repository's shared CVRPTW model
so SGBS uses the same optimized environment/model path as PolyNet and LINC.
"""

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

import torch


if sys.platform.startswith("win"):
    pathlib.PosixPath = pathlib.WindowsPath

ROOT = Path(__file__).resolve().parents[3]
MODEL_CVRPTW = (ROOT / "CVRPTW" / "PolyNet").resolve()
for module_path in (ROOT, ROOT / "CVRPTW", MODEL_CVRPTW, Path(__file__).resolve().parent):
    module_path = str(module_path)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from CVRPTWModel import CVRPTWModel as Model  # noqa: E402
from polynet_sgbs_impl import (  # noqa: E402
    Env,
    load_saved_dataset,
    prepare_env_batch,
    run_batch_sgbs,
    sample_bitwise_z,
)


def parse_args():
    parser = argparse.ArgumentParser(description="CVRPTW POMO SGBS evaluation.")
    parser.add_argument("--checkpoint", default="models/POMO_env/n100/checkpoint-608871.ckpt")
    parser.add_argument("--dataset-pt", "--dataset-pkl", "--data", dest="dataset_pt", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--episodes", "--limit", "--num-instances", dest="limit", type=int, default=0)
    parser.add_argument("--batch-size", "--instance-batch-size", dest="batch_size", type=int, default=3000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--aug", "--aug-factor", dest="aug_factor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-routes", action="store_true")
    parser.add_argument("--amp", choices=("on", "off"), default="on")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / "CVRPTW" / path
    return path.resolve()


def native_pomo_model_params(problem_size=100):
    del problem_size
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
        "z_dim": 16,
        "use_fast_attention": True,
        "force_first_move": True,
        "use_candidate_features": False,
        "candidate_scorer_type": "baseline_additive",
        "use_depth_mixer": False,
        "use_gated_attention": False,
        "node_feature_dim": 6,
        "include_service_duration_in_node_embedding": True,
        "encoder_qkv_bias": True,
        "use_poly_residual": False,
        "use_projected_logit_key": True,
        "cvrptw_input_scale_mode": "horizon",
    }


def map_rl4co_pomo_state_dict(source_state, target_state):
    mapped = {}
    mapped["encoder.embedding_node.weight"] = source_state["policy.encoder.init_embedding.init_embed.weight"]
    mapped["encoder.embedding_node.bias"] = source_state["policy.encoder.init_embedding.init_embed.bias"]
    mapped["encoder.embedding_depot.weight"] = source_state["policy.encoder.init_embedding.init_embed_depot.weight"]
    mapped["encoder.embedding_depot.bias"] = source_state["policy.encoder.init_embedding.init_embed_depot.bias"]

    for layer in range(6):
        prefix = f"policy.encoder.net.layers.{layer}"
        qkv_weight = source_state[f"{prefix}.0.module.Wqkv.weight"]
        qkv_bias = source_state[f"{prefix}.0.module.Wqkv.bias"]
        mapped[f"encoder.layers.{layer}.Wq.weight"] = qkv_weight[0:128]
        mapped[f"encoder.layers.{layer}.Wk.weight"] = qkv_weight[128:256]
        mapped[f"encoder.layers.{layer}.Wv.weight"] = qkv_weight[256:384]
        mapped[f"encoder.layers.{layer}.Wq.bias"] = qkv_bias[0:128]
        mapped[f"encoder.layers.{layer}.Wk.bias"] = qkv_bias[128:256]
        mapped[f"encoder.layers.{layer}.Wv.bias"] = qkv_bias[256:384]
        mapped[f"encoder.layers.{layer}.multi_head_combine.weight"] = source_state[f"{prefix}.0.module.out_proj.weight"]
        mapped[f"encoder.layers.{layer}.multi_head_combine.bias"] = source_state[f"{prefix}.0.module.out_proj.bias"]
        mapped[f"encoder.layers.{layer}.add_n_normalization_1.norm.weight"] = source_state[f"{prefix}.1.normalizer.weight"]
        mapped[f"encoder.layers.{layer}.add_n_normalization_1.norm.bias"] = source_state[f"{prefix}.1.normalizer.bias"]
        mapped[f"encoder.layers.{layer}.feed_forward.W1.weight"] = source_state[f"{prefix}.2.module.lins.0.weight"]
        mapped[f"encoder.layers.{layer}.feed_forward.W1.bias"] = source_state[f"{prefix}.2.module.lins.0.bias"]
        mapped[f"encoder.layers.{layer}.feed_forward.W2.weight"] = source_state[f"{prefix}.2.module.lins.1.weight"]
        mapped[f"encoder.layers.{layer}.feed_forward.W2.bias"] = source_state[f"{prefix}.2.module.lins.1.bias"]
        mapped[f"encoder.layers.{layer}.add_n_normalization_2.norm.weight"] = source_state[f"{prefix}.3.normalizer.weight"]
        mapped[f"encoder.layers.{layer}.add_n_normalization_2.norm.bias"] = source_state[f"{prefix}.3.normalizer.bias"]

    projected_nodes = source_state["policy.decoder.project_node_embeddings.weight"]
    mapped["decoder.Wq_last.weight"] = source_state["policy.decoder.context_embedding.project_context.weight"]
    mapped["decoder.Wk.weight"] = projected_nodes[0:128]
    mapped["decoder.Wv.weight"] = projected_nodes[128:256]
    mapped["decoder.Wlogit.weight"] = projected_nodes[256:384]
    mapped["decoder.multi_head_combine.weight"] = source_state["policy.decoder.pointer.project_out.weight"]
    mapped["decoder.multi_head_combine.bias"] = torch.zeros_like(target_state["decoder.multi_head_combine.bias"])

    shape_errors = [
        (key, tuple(target_state[key].shape), tuple(value.shape))
        for key, value in mapped.items()
        if key not in target_state or tuple(target_state[key].shape) != tuple(value.shape)
    ]
    if shape_errors:
        raise RuntimeError(f"POMO/native state mapping shape mismatch: {shape_errors[:5]}")

    state = dict(target_state)
    state.update(mapped)
    return state


def load_model(checkpoint_path, device, args):
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source_state = checkpoint.get("state_dict", checkpoint)
    model_params = native_pomo_model_params(args.problem_size)
    model = Model(**model_params).to(device)
    state = map_rl4co_pomo_state_dict(source_state, model.state_dict())
    model.load_state_dict(state, strict=True)
    model.eval()
    model.decoder.capture_candidate_aux = False
    return model, model_params, checkpoint


def solve_batch(env, model, model_params, device, dataset, start_idx, batch_size, args, amp_dtype):
    env.rollout_size = int(args.k)
    env.aug_factor = int(args.aug_factor)
    prepare_env_batch(env, dataset, start_idx, batch_size, device)
    reset_state, _, _ = env.reset()
    z = sample_bitwise_z(
        batch_size=batch_size,
        z_samples=int(args.k),
        z_dim=int(model_params["z_dim"]),
        seed=int(args.seed),
        aug_factor=int(args.aug_factor),
        device=device,
        mode="random",
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
            row["tour"] = selected_node_list[env_row, beam_index, :selected_count].detach().long().cpu().tolist()
            row["selected_count"] = selected_count
        batch_rows.append(row)
    return batch_rows


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
    output_json = resolve(args.output_json) if args.output_json else ROOT / "CVRPTW" / "results" / "cvrptw_pomo_sgbs_eval.json"
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

    model, model_params, checkpoint = load_model(checkpoint_path, device, args)
    env = Env(problem_size=problem_size)
    env.input_scale_mode = str(model_params.get("cvrptw_input_scale_mode", "horizon"))
    env.enforce_depot_return = True
    env.enable_candidate_features = False

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
                rows.append(row)
            print(f"[progress] {start_idx + batch_size}/{total}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_memory_mb = None
    elapsed_sec = time.perf_counter() - start_time
    payload = {
        "method": "pomo_sgbs",
        "problem": "cvrptw",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)) if isinstance(checkpoint, dict) else -1,
        "dataset_pt": str(dataset_path),
        "instance_count": len(rows),
        "problem_size": problem_size,
        "mean_distance": sum(row["distance"] for row in rows) / len(rows),
        "beam_width": int(args.beam_width),
        "expand_k": int(args.expand_k),
        "k": int(args.k),
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
