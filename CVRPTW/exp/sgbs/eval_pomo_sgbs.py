"""POMO CVRPTW SGBS evaluation."""

import argparse
import os
import subprocess
import sys

_task_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _task_root not in sys.path:
    sys.path.insert(0, _task_root)
from exp.common.datasets import add_dataset_args, resolve_dataset  # noqa: E402


_sgbs_impl = os.path.join(os.path.dirname(__file__), "pomo_sgbs_impl.py")


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="synthetic100")
    parser.add_argument("--checkpoint", default="models/POMO_env/n100/checkpoint-608871.ckpt")
    parser.add_argument("--problem-size", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--k", type=int, default=0, help="Number of POMO starts. Default 0 means n starts.")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-routes", action="store_true")
    parser.add_argument("--amp", choices=("on", "off"), default="on")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    resolve_dataset(args)
    if int(args.k) <= 0:
        args.k = int(args.problem_size)
    cmd = [
        sys.executable,
        _sgbs_impl,
        "--checkpoint",
        os.path.join(_task_root, args.checkpoint),
        "--dataset-pt",
        args.data,
        "--problem-size",
        str(args.problem_size),
        "--episodes",
        str(args.episodes),
        "--batch-size",
        str(args.batch_size),
        "--k",
        str(args.k),
        "--beam-width",
        str(args.beam_width),
        "--expand-k",
        str(args.expand_k),
        "--aug",
        str(args.aug),
        "--seed",
        str(args.seed),
        "--amp",
        args.amp,
        "--amp-dtype",
        args.amp_dtype,
    ]
    if args.output_json:
        cmd += ["--output-json", os.path.join(_task_root, args.output_json)]
    if args.include_routes:
        cmd.append("--include-routes")
    if args.cpu:
        cmd.append("--cpu")
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(cmd, cwd=_task_root, check=True, env=env)

if __name__ == "__main__":
    main()
