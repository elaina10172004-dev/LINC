"""Linc CVRPTW SGBS evaluation."""
import argparse, os, subprocess, sys
_task_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _task_root not in sys.path:
    sys.path.insert(0, _task_root)
from exp.common.datasets import add_dataset_args, resolve_dataset
_sgbs_impl = os.path.join(os.path.dirname(__file__), "linc_sgbs_impl.py")

def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="synthetic100")
    parser.add_argument("--checkpoint", default="models/LINC_env_scratch/n100/checkpoint-11885.pt")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1500)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--z-samples", type=int, default=128)
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--include-routes", action="store_true")
    args = parser.parse_args()
    resolve_dataset(args)
    cmd = [sys.executable, _sgbs_impl,
           "--ours-checkpoint", os.path.join(_task_root, args.checkpoint),
           "--dataset-pkl", args.data,
           "--num-instances", str(args.episodes),
           "--instance-batch-size", str(args.batch_size),
           "--beam-width", str(args.beam_width),
           "--expand-k", str(args.expand_k),
           "--z-samples", str(args.z_samples),
           "--aug-factor", str(args.aug),
           "--seed", str(args.seed)]
    if args.output_json:
        cmd += ["--output-json", os.path.join(_task_root, args.output_json)]
    if args.include_routes:
        cmd.append("--include-routes")
    if args.cpu: cmd.append("--cpu")
    subprocess.run(cmd, cwd=_task_root, check=True)

if __name__ == "__main__":
    main()
