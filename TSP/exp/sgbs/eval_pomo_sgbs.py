"""POMO TSP SGBS evaluation."""

import argparse
import os
import subprocess
import sys

_task_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _task_root not in sys.path:
    sys.path.insert(0, _task_root)

from exp.common.datasets import add_dataset_args, resolve_dataset, run_variable_size_tsp, run_variable_size_parallel  # noqa: E402

_sgbs_impl = os.path.join(os.path.dirname(__file__), "pomo_sgbs_impl.py")


def run_once(args):
    cmd = [
        sys.executable,
        _sgbs_impl,
        "--checkpoint",
        os.path.join(_task_root, args.checkpoint),
        "--data",
        args.data,
        "--problem-size",
        str(args.problem_size),
        "--episodes",
        str(args.episodes),
        "--batch-size",
        str(args.batch_size),
        "--beam-width",
        str(args.beam_width),
        "--expand-k",
        str(args.expand_k),
        "--start-mode",
        args.start_mode,
        "--aug",
        str(args.aug),
        "--seed",
        str(args.seed),
        "--cuda-device",
        str(args.cuda_device),
        "--amp",
        args.amp,
    ]
    if args.output_json:
        cmd += ["--output-json", os.path.join(_task_root, args.output_json)]
    if args.cpu:
        cmd.append("--cpu")
    subprocess.run(cmd, cwd=_task_root, check=True)


def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="kool100")
    parser.add_argument("--checkpoint", default="models/POMO/saved_tsp100_model2_longTrain/checkpoint-3100.pt")
    parser.add_argument("--problem-size", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--start-mode", choices=("topk", "all"), default="topk")
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--amp", choices=("on", "off"), default="on")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    dataset = resolve_dataset(args)
    if dataset.variable_size:
        args._argv = sys.argv
        run_variable_size_parallel(args, dataset)
    else:
        run_once(args)


if __name__ == "__main__":
    main()
