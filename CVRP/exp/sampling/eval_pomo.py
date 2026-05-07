"""POMO CVRP sampling evaluation."""
import argparse, json, os, sys, random
from pathlib import Path
import numpy as np
import torch

_task_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_impl_root = os.path.join(_task_root, "POMO")
for _path in (_impl_root, _task_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)
os.chdir(_task_root)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

from exp.common.datasets import add_dataset_args, resolve_dataset
from exp.common.xml import evaluate_pomo_xml_sampling

def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="kool100")
    parser.add_argument("--checkpoint", default="models/POMO/saved_CVRP100_model")
    parser.add_argument("--epoch", type=int, default=30500)
    parser.add_argument("--problem-size", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2500)
    parser.add_argument("--z-samples", type=int, default=0, help="Number of POMO starts for sampling. Default 0 means n starts.")
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mode", choices=("sampling", "greedy"), default="sampling")
    parser.add_argument("--eval-type", choices=("greedy", "softmax"), default="", help="Legacy override. Default uses greedy decoding for both modes.")
    args = parser.parse_args()
    dataset = resolve_dataset(args)
    if args.mode == "greedy":
        pomo_size = 1
    else:
        pomo_size = int(args.problem_size) if int(args.z_samples) <= 0 else min(int(args.z_samples), int(args.problem_size))
    eval_type = args.eval_type or "greedy"
    if dataset.kind == "xml":
        evaluate_pomo_xml_sampling(
            checkpoint=args.checkpoint,
            epoch=args.epoch,
            instances_root=dataset.path,
            bks_json=dataset.bks_json,
            output_json=args.output_json,
            episodes=args.episodes,
            batch_size=args.batch_size,
            aug_factor=args.aug,
            seed=args.seed,
            cuda_device=args.cuda_device,
            cpu=args.cpu,
            amp=not args.no_amp,
            pomo_size=pomo_size,
            mode=args.mode,
            eval_type=eval_type,
        )
        return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    from POMO.POMOTester import POMOTester as Tester
    use_cuda = torch.cuda.is_available() and not args.cpu

    ckpt_path = args.checkpoint
    ckpt_epoch = args.epoch
    if ckpt_path.endswith(".pt"):
        ckpt_epoch = int(os.path.basename(ckpt_path).split("-")[1].split(".")[0])
        ckpt_path = os.path.dirname(ckpt_path)

    tester = Tester(
        env_params={"problem_size": args.problem_size, "pomo_size": pomo_size},
        model_params={
            "embedding_dim": 128, "sqrt_embedding_dim": 128**0.5,
            "encoder_layer_num": 6, "qkv_dim": 16, "head_num": 8,
            "logit_clipping": 10, "ff_hidden_dim": 512,
        },
        tester_params={
            "use_cuda": use_cuda, "cuda_device_num": args.cuda_device,
            "amp_inference": not args.no_amp,
            "model_load": {
                "path": ckpt_path,
                "epoch": ckpt_epoch,
            },
            "test_episodes": args.episodes,
            "test_batch_size": args.batch_size,
            "augmentation_enable": args.aug > 1, "aug_factor": args.aug,
            "aug_batch_size": args.batch_size,
            "test_z_sample_size": pomo_size,
            "eval_type": eval_type,
            "test_data_load": {"enable": True, "filename": args.data},
        },
    )
    result = tester.run()
    result.update(
        {
            "method": f"pomo_{args.mode}",
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": int(ckpt_epoch),
            "dataset_path": str(args.data),
            "problem_size": int(args.problem_size),
            "mode": args.mode,
            "pomo_size": int(pomo_size),
            "aug_factor": int(args.aug),
            "rollouts_per_instance": int(pomo_size) * int(args.aug),
            "eval_type": eval_type,
            "seed": int(args.seed),
        }
    )
    if args.output_json:
        output = Path(args.output_json)
        if not output.is_absolute():
            output = Path(_task_root) / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_keys = ("episodes", "score_mean", "aug_score_mean", "elapsed_sec", "peak_memory_mb", "metric")
    print(json.dumps({k: result[k] for k in summary_keys if k in result}, indent=2))

if __name__ == "__main__":
    main()
