"""PolyNet CVRP sampling evaluation."""
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
from exp.common.xml import evaluate_poly_xml_sampling

def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="kool100")
    parser.add_argument("--checkpoint", default="models/PolyNet/n100")
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--problem-size", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--z-samples", type=int, default=800)
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--eval-type", choices=("greedy", "sampling"), default="sampling")
    args = parser.parse_args()
    dataset = resolve_dataset(args)
    if dataset.kind == "xml":
        evaluate_poly_xml_sampling(
            impl="PolyNet",
            method="polynet_sampling",
            checkpoint=args.checkpoint,
            epoch=args.epoch,
            instances_root=dataset.path,
            bks_json=dataset.bks_json,
            output_json=args.output_json,
            episodes=args.episodes,
            batch_size=args.batch_size,
            z_samples=args.z_samples,
            aug_factor=args.aug,
            seed=args.seed,
            cuda_device=args.cuda_device,
            cpu=args.cpu,
            amp=not args.no_amp,
            greedy=(args.eval_type == "greedy"),
        )
        return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    from PolyNet.CVRPTester import CVRPTester as Tester
    use_cuda = torch.cuda.is_available() and not args.cpu

    ckpt_path = args.checkpoint
    ckpt_epoch = args.epoch
    if ckpt_path.endswith(".pt"):
        ckpt_epoch = int(os.path.basename(ckpt_path).split("-")[1].split(".")[0])
        ckpt_path = os.path.dirname(ckpt_path)

    eval_type = "argmax" if args.eval_type == "greedy" else "softmax"
    tester = Tester(
        env_params={"problem_size": args.problem_size},
        model_params={
            "embedding_dim": 128, "poly_embedding_dim": 256,
            "sqrt_embedding_dim": 128**0.5, "encoder_layer_num": 6,
            "qkv_dim": 16, "head_num": 8, "logit_clipping": 10,
            "ff_hidden_dim": 512, "eval_type": eval_type, "z_dim": 16,
            "use_fast_attention": True, "force_first_move": False,
            "use_candidate_features": False, "candidate_scorer_type": "baseline_additive",
            "use_depth_mixer": False, "use_gated_attention": False,
        },
        tester_params={
            "use_cuda": use_cuda, "cuda_device_num": args.cuda_device,
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

if __name__ == "__main__":
    main()
