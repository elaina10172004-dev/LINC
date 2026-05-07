"""PolyNet TSP sampling evaluation."""

import argparse
import json
import os
import random
import sys
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

from exp.common.datasets import add_dataset_args, resolve_dataset, run_variable_size_tsp, run_variable_size_parallel  # noqa: E402
from PolyNet.TSPTester import CVRPTester as Tester  # noqa: E402


def _checkpoint_parts(checkpoint, epoch):
    if checkpoint.endswith(".pt"):
        return os.path.dirname(checkpoint), int(os.path.basename(checkpoint).split("-")[1].split(".")[0])
    return checkpoint, int(epoch)


def run_once(args):
    ckpt_path, ckpt_epoch = _checkpoint_parts(args.checkpoint, args.epoch)
    eval_type = "argmax" if args.eval_type == "greedy" else "softmax"
    tester = Tester(
        env_params={"problem_size": args.problem_size},
        model_params={
            "embedding_dim": 128,
            "poly_embedding_dim": 256,
            "sqrt_embedding_dim": 128**0.5,
            "encoder_layer_num": 6,
            "qkv_dim": 16,
            "head_num": 8,
            "logit_clipping": 10,
            "ff_hidden_dim": 512,
            "eval_type": eval_type,
            "z_dim": 16,
            "use_fast_attention": True,
            "force_first_move": False,
            "use_candidate_features": False,
            "selected_candidate_feature_names": [],
            "relative_candidate_feature_names": [],
            "selected_node_static_feature_names": [],
            "node_static_embedding_mode": "concat",
            "candidate_feature_hidden_dim": 0,
            "candidate_rollout_chunk_size": 32,
            "use_decoder_checkpointing": False,
            "candidate_scorer_type": "baseline_additive",
            "use_depth_mixer": False,
            "use_gated_attention": False,
        },
        tester_params={
            "use_cuda": torch.cuda.is_available() and not args.cpu,
            "cuda_device_num": args.cuda_device,
            "amp_inference": not args.no_amp,
            "model_load": {"path": ckpt_path, "epoch": ckpt_epoch},
            "test_episodes": args.episodes,
            "test_batch_size": args.batch_size,
            "augmentation_enable": args.aug > 1,
            "aug_factor": args.aug,
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
        }
    )
    if args.output_json:
        output = Path(args.output_json)
        if not output.is_absolute():
            output = Path(_task_root) / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("episodes", "score_mean", "aug_score_mean", "z_samples", "aug_factor", "eval_type")}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="kool100")
    parser.add_argument("--checkpoint", default="models/PolyNet/n100")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--problem-size", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=600)
    parser.add_argument("--z-samples", type=int, default=800)
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-type", choices=("greedy", "sampling"), default="sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = resolve_dataset(args)
    if dataset.variable_size:
        args._argv = sys.argv
        run_variable_size_parallel(args, dataset)
    else:
        return run_once(args)


if __name__ == "__main__":
    main()
