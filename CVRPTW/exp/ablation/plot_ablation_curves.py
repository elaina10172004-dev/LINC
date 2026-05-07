"""Plot Solomon50 ablation training curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "results" / "ablation" / "cvrptw" / "curves"


VARIANT_LABELS = {
    "baseline": "Baseline",
    "full_linc": "Full LINC",
    "no_local": "w/o local consequence interface",
    "naive_mlp": "Naive feature injection",
    "no_gateattn": "w/o GateAttn",
    "no_depth_mixer": "w/o Depth Mixer",
    "no_soft_top1": "Hard top-1",
    "group_mean": "Group mean",
}


def _load_curve(run_dir: Path, metric: str):
    ckpt = run_dir / "checkpoint_latest.pt"
    if not ckpt.exists():
        candidates = sorted(run_dir.glob("checkpoint-*.pt"))
        if not candidates:
            return None
        ckpt = candidates[-1]
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw = data.get("result_log")
    if not raw:
        return None
    _, series = raw
    if metric not in series:
        return None
    points = series[metric]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return xs, ys


def _plot(metric: str, curves: dict[str, tuple[list[float], list[float]]], output_dir: Path):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for variant, curve in curves.items():
        xs, ys = curve
        label = VARIANT_LABELS.get(variant, variant)
        linewidth = 2.6 if variant == "full_linc" else 1.4
        alpha = 1.0 if variant == "full_linc" else 0.82
        plt.plot(xs, ys, label=label, linewidth=linewidth, alpha=alpha)
    plt.xlabel("Training batch")
    plt.ylabel(metric.replace("_", " "))
    if metric == "train_score":
        plt.ylim(top=1000)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    all_path = output_dir / f"{metric}_all.png"
    plt.savefig(all_path, dpi=200)
    plt.close()

    full_curve = curves.get("full_linc")
    if full_curve is None:
        return
    for variant, curve in curves.items():
        if variant == "full_linc":
            continue
        plt.figure(figsize=(8, 4.5))
        xs, ys = full_curve
        plt.plot(xs, ys, label="Full LINC", linewidth=2.4)
        xs, ys = curve
        plt.plot(xs, ys, label=VARIANT_LABELS.get(variant, variant), linewidth=1.8)
        plt.xlabel("Training batch")
        plt.ylabel(metric.replace("_", " "))
        if metric == "train_score":
            plt.ylim(top=1000)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_{variant}_vs_full.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", default="ablation_cvrptw")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--variants", default=",".join(VARIANT_LABELS))
    parser.add_argument("--metric", choices=("train_score", "train_loss", "both"), default="train_score")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    metrics = ["train_score", "train_loss"] if args.metric == "both" else [args.metric]
    output_dir = Path(args.output_dir).resolve()

    for metric in metrics:
        curves = {}
        for variant in variants:
            run_dir = ROOT / "CVRPTW" / "result" / f"run_{args.run_prefix}_{variant}_seed{args.seed}"
            curve = _load_curve(run_dir, metric)
            if curve is not None:
                curves[variant] = curve
        if not curves:
            raise FileNotFoundError(f"No curves found for metric={metric}")
        _plot(metric, curves, output_dir)
    print({"output_dir": str(output_dir), "metrics": metrics})


if __name__ == "__main__":
    main()
