"""Train LINC on CVRP100 with the default paper configuration."""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch


_task_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_impl_root = os.path.join(_task_root, "LINC")
for _path in (_impl_root, _task_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(_task_root)

from CVRPTrainer import CVRPTrainer as Trainer  # noqa: E402
from utils.utils import create_logger  # noqa: E402


def _parse_checkpoint(path, default_epoch):
    if not path:
        return {"enable": False}
    if path.endswith(".pt"):
        epoch = int(os.path.basename(path).split("-")[1].split(".")[0])
        return {"enable": True, "path": os.path.dirname(path), "epoch": epoch}
    return {"enable": True, "path": path, "epoch": int(default_epoch)}


def _parse_milestones(text):
    if not text:
        return []
    return [int(item) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LINC CVRP100 model.")
    parser.add_argument("--run-name", default="linc_cvrp_n100")
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1430)
    parser.add_argument("--instances-per-epoch", type=int, default=6400)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-milestones", default="")
    parser.add_argument("--lr-gamma", type=float, default=1.0)
    parser.add_argument("--slow-start-epochs", type=int, default=100)
    parser.add_argument("--warm-start-checkpoint", default="models/PolyNet/n100/checkpoint-300.pt")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--val-data", default="")
    parser.add_argument("--val-episodes", type=int, default=0)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--val-z-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    return parser.parse_args()


def build_params(args):
    model_params = {
        "embedding_dim": 128,
        "poly_embedding_dim": 256,
        "sqrt_embedding_dim": math.sqrt(128.0),
        "encoder_layer_num": 6,
        "qkv_dim": 16,
        "head_num": 8,
        "logit_clipping": 10,
        "ff_hidden_dim": 512,
        "z_dim": 16,
        "use_fast_attention": True,
        "force_first_move": False,
        "use_depth_mixer": True,
        "use_gated_attention": True,
        "gated_attention_scale_mode": "centered_sigmoid",
        "gated_attention_init_bias": 2.0,
        "alpha_attn_gate": 1.0,
        "node_static_embedding_mode": "residual",
        "node_static_slow_start_ratio": 1.0,
        "node_static_start_alpha": 0.0,
        "depth_mixer_slow_start_ratio": 1.0,
        "depth_mixer_start_alpha": 0.0,
        "gated_attention_slow_start_ratio": 1.0,
        "gated_attention_start_alpha": 0.0,
        "use_candidate_features": True,
        "selected_candidate_feature_names": [
            "travel_dist_norm", "demand_ratio", "dist_to_depot_norm", "depot_angle_diff_norm",
        ],
        "selected_node_static_feature_names": ["knn_mean_dist_norm"],
        "candidate_feature_hidden_dim": 0,
        "candidate_rollout_chunk_size": 32,
        "use_decoder_checkpointing": True,
        "candidate_full_gate_slow_start_ratio": 1.0,
        "candidate_full_gate_start_alpha": 0.0,
        "candidate_phi_bias_slow_start_ratio": 1.0,
        "candidate_phi_bias_start_alpha": 0.0,
        "candidate_feature_residual_slow_start_ratio": 1.0,
        "candidate_feature_residual_start_alpha": 0.0,
        "quotient_lite_slow_start_ratio": 1.0,
        "quotient_lite_start_alpha": 0.0,
        "candidate_scorer_type": "quotient_lite",
        "relative_candidate_feature_names": [
            "travel_dist_norm", "demand_ratio", "load_after_ratio", "dist_to_depot_norm",
        ],
        "quotient_scorer_hidden_dim": 64,
        "quotient_lite_hidden_dim": 64,
        "quotient_scorer_activation": "gelu",
        "qlite_force_alpha_one": False,
        "qlite_disable_summary_modulation": False,
        "zero_depot_relative_features": False,
    }
    trainer_params = {
        "use_cuda": torch.cuda.is_available() and not args.cpu,
        "cuda_device_num": args.cuda_device,
        "compile_model": args.compile_model,
        "epochs": args.epochs,
        "train_batch_size": args.batch_size,
        "train_num_rollouts": int(args.instances_per_epoch) * int(args.k),
        "K": args.k,
        "amp_training": not args.no_amp,
        "model_load": {"enable": False} if args.no_warm_start else _parse_checkpoint(args.warm_start_checkpoint, 300),
        "enable_epoch_validation": args.val_episodes > 0,
        "val_episodes": args.val_episodes,
        "val_batch_size": args.val_batch_size,
        "val_z_sample_size": args.val_z_samples,
        "validation_data_load": {"enable": bool(args.val_data), "filename": args.val_data},
        "module_slow_start_epochs": args.slow_start_epochs,
        "module_slow_start_by_batch": True,
        "advantage_schedule_by_batch": True,
        "advantage_params": {
            "mode": "group_mean",
            "tau": 1.0,
            "reward_scale": 1.0,
            "normalize_adv": False,
            "eps": 1e-6,
            "rollout_mask_mode": "best_only",
            "tau_start": 1.0,
            "tau_end": 1.0,
            "tau_anneal_ratio": 0.0,
            "best_only_start_ratio": 0.0,
        },
        "logging": {"model_save_interval": args.save_interval},
    }
    optimizer_params = {
        "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"milestones": _parse_milestones(args.lr_milestones), "gamma": args.lr_gamma},
    }
    return (
        {"name": args.run_name},
        {"problem_size": args.problem_size},
        model_params,
        optimizer_params,
        trainer_params,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    create_logger(log_file={"desc": args.run_name, "filename": "log.txt"}, run_name=args.run_name)
    trainer = Trainer(*build_params(args))
    trainer.run()


if __name__ == "__main__":
    main()
