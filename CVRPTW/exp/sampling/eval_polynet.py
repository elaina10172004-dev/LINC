"""PolyNet CVRPTW sampling evaluation."""
import argparse, json, os, sys, random
from pathlib import Path
import numpy as np
import torch

_task_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_impl_root = os.path.join(_task_root, "PolyNet")
for _path in (_impl_root, _task_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(_task_root)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from exp.common.datasets import add_dataset_args, resolve_dataset
from PolyNet.CVRPTWTester import CVRPTWTester as Tester


def _attach_dataset_rows(result, dataset_path):
    if not result.get("rows"):
        return result
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if not isinstance(dataset, dict):
        return result
    names = dataset.get("names")
    bks_cost = dataset.get("bks_cost")
    for row in result["rows"]:
        idx = int(row["instance"])
        if names is not None and 0 <= idx < len(names):
            row["instance"] = str(names[idx])
        if bks_cost is not None and 0 <= idx < len(bks_cost):
            bks_value = float(torch.as_tensor(bks_cost[idx]).item())
            row["bks_cost"] = bks_value
            row["gap_pct"] = 100.0 * (float(row["distance"]) - bks_value) / bks_value
    if result["rows"] and "bks_cost" in result["rows"][0]:
        result["mean_bks_cost"] = sum(row["bks_cost"] for row in result["rows"]) / len(result["rows"])
        result["mean_gap_pct"] = sum(row["gap_pct"] for row in result["rows"]) / len(result["rows"])
        result["aggregate_gap_pct"] = 100.0 * (
            sum(row["distance"] for row in result["rows"]) - sum(row["bks_cost"] for row in result["rows"])
        ) / sum(row["bks_cost"] for row in result["rows"])
    return result


def _resolve_checkpoint(checkpoint, epoch):
    ckpt_path = checkpoint
    ckpt_epoch = epoch
    if ckpt_path.endswith(".pt"):
        ckpt_epoch = int(os.path.basename(ckpt_path).split("-")[1].split(".")[0])
        ckpt_path = os.path.dirname(ckpt_path)
    full_name = os.path.join(ckpt_path, f"checkpoint-{ckpt_epoch}.pt")
    return ckpt_path, ckpt_epoch, full_name


def _checkpoint_model_params(full_name, defaults):
    params = dict(defaults)
    checkpoint = torch.load(full_name, map_location="cpu", weights_only=False)
    if isinstance(checkpoint.get("model_params"), dict):
        params.update(checkpoint["model_params"])
    if "z_dim" in checkpoint:
        params["z_dim"] = int(checkpoint["z_dim"])
    if "force_first_move" in checkpoint:
        params["force_first_move"] = bool(checkpoint["force_first_move"])
    return params

def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="synthetic100")
    parser.add_argument("--checkpoint", default="models/PolyNet_env/n100/checkpoint-300000.pt")
    parser.add_argument("--epoch", type=int, default=300000)
    parser.add_argument("--problem-size", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--z-samples", type=int, default=800)
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-type", choices=("greedy", "sampling", "argmax", "softmax"), default="")
    parser.add_argument("--input-scale-mode", choices=("checkpoint", "grid", "horizon"), default="checkpoint")
    args = parser.parse_args()
    resolve_dataset(args)

    _et = args.eval_type
    if _et in ("greedy", "argmax"): _et = "argmax"
    elif _et in ("sampling", "softmax"): _et = "softmax"
    else: _et = ""

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    ckpt_path, ckpt_epoch, ckpt_full_name = _resolve_checkpoint(args.checkpoint, args.epoch)
    model_params = _checkpoint_model_params(ckpt_full_name, {
        "embedding_dim": 128, "poly_embedding_dim": 256,
        "sqrt_embedding_dim": 128**0.5, "encoder_layer_num": 6,
        "qkv_dim": 16, "head_num": 8, "logit_clipping": 10,
        "ff_hidden_dim": 512, "eval_type": "softmax", "z_dim": 16,
        "use_fast_attention": True, "force_first_move": False,
        "use_candidate_features": False, "candidate_scorer_type": "baseline_additive",
        "use_depth_mixer": False, "use_gated_attention": False,
    })

    if _et:
        model_params["eval_type"] = _et

    input_scale_mode = str(model_params.get("cvrptw_input_scale_mode", "grid"))
    if args.input_scale_mode != "checkpoint":
        input_scale_mode = args.input_scale_mode

    tester = Tester(
        env_params={
            "problem_size": args.problem_size,
            "input_scale_mode": input_scale_mode,
            "enable_candidate_features": bool(model_params.get("use_candidate_features", False)),
            "enforce_depot_return": True,
        },
        model_params=model_params,
        tester_params={
            "use_cuda": torch.cuda.is_available(), "cuda_device_num": args.cuda_device,
            "amp_inference": not args.no_amp,
            "model_load": {"path": ckpt_path, "epoch": ckpt_epoch},
            "test_episodes": args.episodes,
            "test_batch_size": args.batch_size,
            "augmentation_enable": args.aug > 1, "aug_factor": args.aug,
            "aug_batch_size": args.batch_size,
            "test_z_sample_size": args.z_samples,
            "EAS_params": {"enable": False},
            "test_data_load": {"enable": True, "filename": args.data},
            "solution_max_length": None,
        },
    )
    result = tester.run()
    result.update(
        {
            "method": "polynet_sampling",
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": int(ckpt_epoch),
            "dataset_path": str(args.data),
            "problem_size": int(args.problem_size),
            "seed": int(args.seed),
            "input_scale_mode": input_scale_mode,
        }
    )
    _attach_dataset_rows(result, args.data)
    if args.output_json:
        output = Path(args.output_json)
        if not output.is_absolute():
            output = Path(_task_root) / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("episodes", "score_mean", "aug_score_mean", "peak_memory_mb", "z_samples", "aug_factor", "eval_type", "input_scale_mode")}, indent=2))
    return result

if __name__ == "__main__":
    main()
