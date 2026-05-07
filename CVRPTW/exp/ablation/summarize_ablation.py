"""Summarize Solomon50 ablation eval JSON files into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "results" / "ablation" / "cvrptw"


ROWS = [
    ("baseline", "Baseline", "N", "N", "N", "original"),
    ("full_linc", "Full LINC", "Y", "Y", "Y", "soft top-1"),
    ("no_local", "w/o local consequence interface", "N", "Y", "Y", "soft top-1"),
    ("naive_mlp", "Naive feature injection", "MLP score only", "Y", "Y", "soft top-1"),
    ("no_gateattn", "w/o GateAttn", "Y", "N", "Y", "soft top-1"),
    ("no_depth_mixer", "w/o Depth Mixer", "Y", "Y", "N", "soft top-1"),
    ("no_soft_top1", "w/o soft top-1", "Y", "Y", "Y", "hard top-1"),
    ("group_mean", "group_mean", "Y", "Y", "Y", "group_mean"),
]


def _load_obj(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("aug_score_mean", "mean_distance", "score_mean"):
        if key in data:
            return float(data[key])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--reference", default="full_linc", help="Variant used as the gap denominator when no BKS exists.")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    values = {variant: _load_obj(results_dir / f"{variant}_eval.json") for variant, *_ in ROWS}
    ref = values.get(args.reference)
    if ref is None:
        available = {k: v for k, v in values.items() if v is not None}
        if not available:
            raise FileNotFoundError(f"No eval JSON files found under {results_dir}")
        ref = min(available.values())

    output_rows = []
    for variant, label, local, gate, depth, credit in ROWS:
        obj = values.get(variant)
        gap = None if obj is None else 100.0 * (obj - ref) / ref
        output_rows.append({
            "variant": variant,
            "Model variant": label,
            "Local consequence interface": local,
            "GateAttn": gate,
            "Depth Mixer": depth,
            "Credit assignment": credit,
            "Obj.": "" if obj is None else f"{obj:.4f}",
            "Gap": "" if gap is None else f"{gap:.2f}%",
        })

    csv_path = Path(args.output_csv).resolve() if args.output_csv else results_dir / "ablation_table.csv"
    md_path = Path(args.output_md).resolve() if args.output_md else results_dir / "ablation_table.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Model variant", "Local consequence interface", "GateAttn", "Depth Mixer", "Credit assignment", "Obj.", "Gap"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in output_rows:
        lines.append("| " + " | ".join(row[key] for key in fieldnames) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"csv": str(csv_path), "markdown": str(md_path), "reference_obj": ref})


if __name__ == "__main__":
    main()
