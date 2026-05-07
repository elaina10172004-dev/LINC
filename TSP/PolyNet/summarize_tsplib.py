"""Summarize TSPLIB50-200 JSON results against classic baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_MEAN = 1197100.0 / 36.0
DEFAULT_METHODS = [
    ("Concorde", REFERENCE_MEAN),
    ("LKH3", REFERENCE_MEAN),
    ("POMO greedy", 35013.5),
    ("POMO sampling", 34365.8),
    ("POMO SGBS", 33705.3),
    ("PolyNet SGBS", 33844.2),
]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write TSPLIB method comparison tables.")
    parser.add_argument("--result-json", default="results/tsplib_linc_sgbs.json")
    parser.add_argument("--out-dir", default="results/tsplib_tables")
    parser.add_argument("--reference-mean", type=float, default=REFERENCE_MEAN)
    parser.add_argument(
        "--extra-method",
        action="append",
        default=[],
        help="Additional comparison row as NAME=OBJ. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_mean_cost(path: Path) -> tuple[float, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mean_cost") is not None:
        return float(payload["mean_cost"]), payload
    rows = payload.get("rows", [])
    values = []
    for row in rows:
        if "cost" in row:
            values.append(float(row["cost"]))
        elif "distance" in row:
            values.append(float(row["distance"]))
    if not values:
        raise ValueError(f"Cannot infer mean cost from {path}")
    return sum(values) / len(values), payload


def parse_extra_methods(extra: list[str]) -> list[tuple[str, float]]:
    rows = []
    for item in extra:
        if "=" not in item:
            raise ValueError(f"--extra-method must be NAME=OBJ, got {item!r}")
        name, value = item.split("=", 1)
        rows.append((name.strip(), float(value)))
    return rows


def gap(obj: float, reference: float) -> float:
    return 100.0 * (float(obj) - float(reference)) / float(reference)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def latex_table(headers: list[str], rows: list[list[str]]) -> str:
    body = ["\\begin{tabular}{" + "l" * len(headers) + "}", "\\toprule"]
    body.append(" & ".join(headers) + r" \\")
    body.append("\\midrule")
    for row in rows:
        latex_row = [cell.replace("%", r"\%") for cell in row]
        body.append(" & ".join(latex_row) + r" \\")
    body.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(body)


def write_method_table(linc_obj: float, reference: float, extra: list[tuple[str, float]], out_dir: Path) -> None:
    method_rows = []
    for name, obj in DEFAULT_METHODS + extra + [("LINC SGBS", linc_obj)]:
        method_rows.append([name, f"{obj:.1f}", f"{gap(obj, reference):.2f}%"])
    headers = ["Method", "Obj.", "Gap vs Concorde"]
    (out_dir / "tsplib_method_comparison.md").write_text(markdown_table(headers, method_rows), encoding="utf-8")
    (out_dir / "tsplib_method_comparison.tex").write_text(latex_table(headers, method_rows), encoding="utf-8")


def write_group_table(payload: dict, out_dir: Path) -> None:
    groups = payload.get("groups", [])
    if not groups:
        return
    rows = []
    for group in sorted(groups, key=lambda item: int(item.get("problem_size", 0))):
        mean_cost = group.get("mean_cost", group.get("aug_score_mean"))
        rows.append(
            [
                str(group.get("problem_size", "")),
                str(group.get("instances", "")),
                f"{float(mean_cost):.1f}" if mean_cost is not None else "--",
                str(group.get("metric") or "--"),
            ]
        )
    headers = ["n", "Instances", "LINC SGBS Obj.", "Metric"]
    (out_dir / "tsplib_linc_sgbs_by_size.md").write_text(markdown_table(headers, rows), encoding="utf-8")
    (out_dir / "tsplib_linc_sgbs_by_size.tex").write_text(latex_table(headers, rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result_path = resolve(args.result_json)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    linc_obj, payload = load_mean_cost(result_path)
    extra = parse_extra_methods(args.extra_method)
    write_method_table(linc_obj, float(args.reference_mean), extra, out_dir)
    write_group_table(payload, out_dir)
    print(
        json.dumps(
            {
                "result_json": str(result_path),
                "out_dir": str(out_dir),
                "linc_sgbs_obj": linc_obj,
                "reference_mean": float(args.reference_mean),
                "gap_pct": gap(linc_obj, float(args.reference_mean)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
