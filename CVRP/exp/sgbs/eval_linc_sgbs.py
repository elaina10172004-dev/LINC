"""Linc CVRP SGBS evaluation."""
import argparse, os, subprocess, sys
_task_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _task_root not in sys.path:
    sys.path.insert(0, _task_root)
from exp.common.datasets import add_dataset_args, resolve_dataset
from exp.common.xml import evaluate_poly_xml_sgbs
_sgbs_impl = os.path.join(os.path.dirname(__file__), "linc_sgbs_impl.py")

def main():
    parser = argparse.ArgumentParser()
    add_dataset_args(parser, default="kool100")
    parser.add_argument("--checkpoint", default="models/LINC_official_morph/n100/checkpoint-1430.pt")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=800)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--expand-k", type=int, default=4)
    parser.add_argument("--z-samples", type=int, default=128)
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    dataset = resolve_dataset(args)
    if dataset.kind == "xml":
        evaluate_poly_xml_sgbs(
            method="linc_sgbs",
            checkpoint=os.path.join(_task_root, args.checkpoint),
            instances_root=dataset.path,
            bks_json=dataset.bks_json,
            output_json=args.output_json,
            episodes=args.episodes,
            batch_size=args.batch_size,
            z_samples=args.z_samples,
            aug_factor=args.aug,
            beam_width=args.beam_width,
            expand_k=args.expand_k,
            seed=args.seed,
            cpu=args.cpu,
        )
        return
    cmd = [sys.executable, _sgbs_impl,
           "--checkpoint", os.path.join(_task_root, args.checkpoint),
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
    if args.cpu: cmd.append("--cpu")
    subprocess.run(cmd, cwd=_task_root, check=True)

if __name__ == "__main__":
    main()
