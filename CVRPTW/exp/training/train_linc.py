"""Train LINC on CVRPTW100 with the default envelope-data configuration."""

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


_task_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_impl_root = os.path.join(_task_root, "LINC")
for _path in (_impl_root, _task_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(_task_root)

from CVRPTWTrainer import CVRPTWTrainer as Trainer  # noqa: E402
from utils.utils import create_logger  # noqa: E402


ABLATION_VARIANTS = {
    "baseline": {
        "use_candidate_features": False,
        "use_gated_attention": False,
        "use_depth_mixer": False,
        "candidate_scorer_type": "baseline_additive",
        "advantage_mode": "group_mean",
        "rollout_mask_mode": "dense",
        "best_only_start_ratio": 0.0,
    },
    "full_linc": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "quotient_lite",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
        "phi_proj_bias": True,
        "zero_depot_relative_features": False,
    },
    "no_phi_bias": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "quotient_lite",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
        "phi_proj_bias": False,
    },
    "no_local": {
        "use_candidate_features": False,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "baseline_additive",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
    },
    "naive_mlp": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "mlp_score_only",
        "mlp_score_feature_mode": "raw",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
    },
    "centered_mlp": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "mlp_score_only",
        "mlp_score_feature_mode": "centered",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
    },
    "raw_linear": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "quotient_lite",
        "qlite_feature_centering_mode": "raw",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
    },
    "no_step_summary": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "quotient_lite",
        "qlite_disable_summary_modulation": True,
        "qlite_force_alpha_one": True,
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
    },
    "full_mu_summary": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "quotient_lite",
        "qlite_summary_mode": "full_mu",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
    },
    "no_gateattn": {
        "use_candidate_features": True,
        "use_gated_attention": False,
        "use_depth_mixer": True,
        "candidate_scorer_type": "quotient_lite",
        "mlp_score_feature_mode": "raw",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
    },
    "no_depth_mixer": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": False,
        "candidate_scorer_type": "quotient_lite",
        "mlp_score_feature_mode": "raw",
        "advantage_mode": "matched_soft_top1",
        "rollout_mask_mode": "best_only",
    },
    "no_soft_top1": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "quotient_lite",
        "mlp_score_feature_mode": "raw",
        "advantage_mode": "group_mean",
        "rollout_mask_mode": "best_only",
        "best_only_start_ratio": 0.0,
    },
    "group_mean": {
        "use_candidate_features": True,
        "use_gated_attention": True,
        "use_depth_mixer": True,
        "candidate_scorer_type": "quotient_lite",
        "mlp_score_feature_mode": "raw",
        "advantage_mode": "group_mean",
        "rollout_mask_mode": "dense",
        "best_only_start_ratio": 0.0,
    },
}


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


def _resolve_existing_path(path):
    if not path:
        return ""
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    for base in (Path(_task_root), Path(_task_root).parent):
        resolved = base / candidate
        if resolved.exists():
            return str(resolved)
    return str(candidate)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the LINC CVRPTW100 model.")
    parser.add_argument("--run-name", default="linc_cvrptw_env_n100")
    parser.add_argument("--problem-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=12000)
    parser.add_argument("--instances-per-epoch", type=int, default=6400)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--lr-milestones", default="")
    parser.add_argument("--lr-gamma", type=float, default=1.0)
    parser.add_argument("--warm-start-checkpoint", default="")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--latest-save-interval", type=int, default=1)
    parser.add_argument("--distribution", choices=["envelope", "uniform"], default="envelope")
    parser.add_argument("--distribution-seed", type=int, default=1234)
    parser.add_argument("--tau-start", type=float, default=4.0)
    parser.add_argument("--tau-end", type=float, default=0.25)
    parser.add_argument("--tau-anneal-ratio", type=float, default=0.0033333333333333335)
    parser.add_argument("--best-only-start-ratio", type=float, default=0.25)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--train-shard-dir", default="")
    parser.add_argument("--train-data", default="", help="Fixed tensor dataset for repeated training batches. Auto-detected for ablation variants.")
    parser.add_argument("--val-data", default="")
    parser.add_argument("--val-episodes", type=int, default=0)
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--val-z-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--candidate-rollout-chunk-size", type=int, default=128)
    parser.add_argument("--enforce-depot-return", dest="enforce_depot_return", action="store_true", help="Enable depot-return feasibility mask during training.")
    parser.add_argument("--no-enforce-depot-return", dest="enforce_depot_return", action="store_false", help="Disable depot-return feasibility mask for legacy reproduction.")
    parser.set_defaults(enforce_depot_return=True)
    parser.add_argument("--ablation-variant", choices=sorted(ABLATION_VARIANTS), default="full_linc")
    parser.add_argument("--candidate-scorer-type", choices=("baseline_additive", "quotient_lite", "mlp_score_only"), default="")
    parser.add_argument("--advantage-mode", choices=("matched_soft_top1", "group_mean"), default="")
    parser.add_argument("--rollout-mask-mode", choices=("best_only", "dense"), default="")
    parser.add_argument("--use-candidate-features", dest="use_candidate_features", action="store_true")
    parser.add_argument("--no-candidate-features", dest="use_candidate_features", action="store_false")
    parser.set_defaults(use_candidate_features=None)
    parser.add_argument("--use-gate-attn", dest="use_gated_attention", action="store_true")
    parser.add_argument("--no-gate-attn", dest="use_gated_attention", action="store_false")
    parser.set_defaults(use_gated_attention=None)
    parser.add_argument("--use-depth-mixer", dest="use_depth_mixer", action="store_true")
    parser.add_argument("--no-depth-mixer", dest="use_depth_mixer", action="store_false")
    parser.set_defaults(use_depth_mixer=None)
    parser.add_argument("--zero-depot-relative-features", dest="zero_depot_relative_features", action="store_true")
    parser.add_argument("--no-zero-depot-relative-features", dest="zero_depot_relative_features", action="store_false")
    parser.set_defaults(zero_depot_relative_features=None)
    return parser.parse_args()


def build_params(args):
    if not args.train_data and args.problem_size == 50:
        default_data = (Path(__file__).resolve().parents[2] / "data" / "solomon56_n50.pt")
        if default_data.exists():
            args.train_data = str(default_data)
    variant_cfg = dict(ABLATION_VARIANTS[args.ablation_variant])
    if args.use_candidate_features is not None:
        variant_cfg["use_candidate_features"] = bool(args.use_candidate_features)
    if args.use_gated_attention is not None:
        variant_cfg["use_gated_attention"] = bool(args.use_gated_attention)
    if args.use_depth_mixer is not None:
        variant_cfg["use_depth_mixer"] = bool(args.use_depth_mixer)
    if args.candidate_scorer_type:
        variant_cfg["candidate_scorer_type"] = args.candidate_scorer_type
    if args.advantage_mode:
        variant_cfg["advantage_mode"] = args.advantage_mode
    if args.rollout_mask_mode:
        variant_cfg["rollout_mask_mode"] = args.rollout_mask_mode
    if args.zero_depot_relative_features is not None:
        variant_cfg["zero_depot_relative_features"] = bool(args.zero_depot_relative_features)

    selected_candidate_feature_names = [
        "travel_dist_norm", "wait_norm", "tw_slack_ratio",
        "arrival_time_norm", "departure_time_norm", "depot_angle_diff_norm",
    ]
    qlite_summary_mode = variant_cfg.get("qlite_summary_mode", "partial")
    qlite_summary_dim = 1 + len(selected_candidate_feature_names) if qlite_summary_mode == "full_mu" else 4
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
        "use_depth_mixer": bool(variant_cfg["use_depth_mixer"]),
        "use_gated_attention": bool(variant_cfg["use_gated_attention"]),
        "gated_attention_init_bias": 2.0,
        "alpha_attn_gate": 1.0,
        "gated_attention_scale_mode": "centered_sigmoid",
        "use_candidate_features": bool(variant_cfg["use_candidate_features"]),
        "selected_candidate_feature_names": selected_candidate_feature_names,
        "relative_candidate_feature_names": [
            "travel_dist_norm", "wait_norm", "tw_slack_ratio",
            "arrival_time_norm", "departure_time_norm",
        ],
        "selected_node_static_feature_names": [
            "knn_nearest_dist_norm", "knn_mean_dist_norm", "knn_min_dist_norm",
        ],
        "candidate_feature_hidden_dim": 0,
        "candidate_rollout_chunk_size": args.candidate_rollout_chunk_size,
        "use_decoder_checkpointing": True,
        "candidate_scorer_type": variant_cfg["candidate_scorer_type"],
        "mlp_score_feature_mode": variant_cfg.get("mlp_score_feature_mode", "raw"),
        "quotient_scorer_hidden_dim": 128,
        "quotient_lite_hidden_dim": 128,
        "quotient_scorer_activation": "gelu",
        "qlite_force_alpha_one": bool(variant_cfg.get("qlite_force_alpha_one", False)),
        "qlite_disable_summary_modulation": bool(variant_cfg.get("qlite_disable_summary_modulation", False)),
        "qlite_feature_centering_mode": variant_cfg.get("qlite_feature_centering_mode", "centered"),
        "qlite_summary_mode": qlite_summary_mode,
        "qlite_summary_dim": qlite_summary_dim,
        "zero_depot_relative_features": bool(variant_cfg.get("zero_depot_relative_features", False)),
        "use_learned_corrector": False,
        "phi_proj_bias": bool(variant_cfg.get("phi_proj_bias", True)),
        "ablation_variant": args.ablation_variant,
    }
    distribution = {"data_type": args.distribution, "seed": args.distribution_seed}
    train_shards_enable = bool(args.train_shard_dir)
    trainer_params = {
        "use_cuda": torch.cuda.is_available() and not args.cpu,
        "cuda_device_num": args.cuda_device,
        "compile_model": args.compile_model,
        "epochs": args.epochs,
        "train_batch_size": args.batch_size,
        "train_num_rollouts": int(args.instances_per_epoch) * int(args.k),
        "K": args.k,
        "amp_training": not args.no_amp,
        "model_load": _parse_checkpoint(args.warm_start_checkpoint, 0) if args.warm_start_checkpoint else {"enable": False},
        "train_data_load": {"enable": bool(args.train_data), "filename": _resolve_existing_path(args.train_data)},
        "train_shards_load": {
            "enable": train_shards_enable,
            "shard_dir": args.train_shard_dir,
            "shuffle_each_epoch": True,
        },
        "epoch_volume_rule": "",
        "batch_single_by_size": [args.instances_per_epoch],
        "train_subbatch_by_size": [args.batch_size],
        "train_batch_size_schedule": [],
        "train_subbatch_size_schedule": [],
        "early_full_grad_ratio": 0.0,
        "max_grad_norm": float(args.max_grad_norm),
        "enable_epoch_validation": args.val_episodes > 0,
        "val_episodes": args.val_episodes,
        "val_batch_size": args.val_batch_size,
        "val_z_sample_size": args.val_z_samples,
        "validation_data_load": {"enable": bool(args.val_data), "filename": _resolve_existing_path(args.val_data)},
        "advantage_schedule_by_batch": True,
        "advantage_params": {
            "mode": variant_cfg["advantage_mode"],
            "tau": args.tau_start,
            "reward_scale": args.reward_scale,
            "normalize_adv": True,
            "eps": 1e-6,
            "rollout_mask_mode": variant_cfg["rollout_mask_mode"],
            "tau_start": args.tau_start,
            "tau_end": args.tau_end,
            "tau_anneal_ratio": args.tau_anneal_ratio,
            "best_only_start_ratio": float(variant_cfg.get("best_only_start_ratio", args.best_only_start_ratio)),
        },
        "corrector_params": {"enable": False},
        "logging": {
            "model_save_interval": args.save_interval,
            "latest_save_interval": args.latest_save_interval,
            "save_latest_on_best": True,
            "save_best_checkpoint": True,
        },
    }
    optimizer_params = {
        "optimizer": {"lr": args.lr, "weight_decay": args.weight_decay},
        "scheduler": {"milestones": _parse_milestones(args.lr_milestones), "gamma": args.lr_gamma},
    }
    use_fused_candidate_features = bool(
        model_params["use_candidate_features"] and model_params["candidate_scorer_type"] == "quotient_lite"
    )
    use_selected_candidate_features = bool(
        model_params["use_candidate_features"] and not use_fused_candidate_features
    )
    return (
        {"name": args.run_name},
        {
            "problem_size": args.problem_size,
            "distribution": distribution,
            "enable_candidate_features": bool(model_params["use_candidate_features"]),
            "use_selected_candidate_features": use_selected_candidate_features,
            "use_fused_candidate_features": use_fused_candidate_features,
            "enforce_depot_return": bool(getattr(args, "enforce_depot_return", True)),
        },
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
