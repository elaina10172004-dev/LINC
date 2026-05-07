"""Plot the six lowest-gap TSPLIB LINC SGBS tours."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot best-gap LINC SGBS tours on TSPLIB.")
    parser.add_argument("--result-json", default="TSP/results/tsplib_linc_sgbs_routes.json")
    parser.add_argument("--dataset-pkl", default="TSP/data/tsplib_50_200.pkl")
    parser.add_argument("--out-dir", default="results/tsplib_linc_sgbs_routes")
    parser.add_argument("--top-k", type=int, default=6)
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = [row.get("instance", idx) for idx, row in enumerate(rows) if "tour" not in row]
    if missing:
        sample = ", ".join(map(str, missing[:5]))
        raise ValueError(f"Rows are missing tours; rerun LINC SGBS with --include-routes. Examples: {sample}")
    return payload


def load_instances(path: Path) -> list:
    with path.open("rb") as f:
        return pickle.load(f)


def gap_pct(row: dict) -> float:
    ref = float(row["reference_cost"])
    return 100.0 * (float(row["cost"]) - ref) / ref


def select_rows(rows: list[dict], top_k: int) -> list[dict]:
    eligible = [row for row in rows if row.get("reference_cost") is not None]
    return sorted(eligible, key=gap_pct)[:top_k]


def plot_instance(ax, row: dict, coords) -> None:
    points = list(coords)
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    tour = [int(node) for node in row["tour"]]
    tour_xy = [points[node] for node in tour] + [points[tour[0]]]
    tx = [float(p[0]) for p in tour_xy]
    ty = [float(p[1]) for p in tour_xy]

    ax.plot(tx, ty, color="#2b6cb0", linewidth=1.25, alpha=0.92)
    ax.scatter(xs, ys, s=13, c="#4a5568", linewidths=0, zorder=3)
    ax.scatter([tx[0]], [ty[0]], s=45, c="#111111", marker="*", zorder=4)
    title = (
        f'{row.get("instance_name", row.get("instance"))} '
        f'({int(row.get("problem_size", len(points)))}) | '
        f'{float(row["cost"]):.0f} / {float(row["reference_cost"]):.0f} | '
        f'{gap_pct(row):.2f}%'
    )
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.06)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#c8ccd2")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def latex_table(headers: list[str], rows: list[list[str]]) -> str:
    body = ["\\begin{tabular}{" + "l" * len(headers) + "}", "\\toprule"]
    body.append(" & ".join(headers) + r" \\")
    body.append("\\midrule")
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(body)


def main() -> None:
    args = parse_args()
    result_path = resolve(args.result_json)
    dataset_path = resolve(args.dataset_pkl)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_payload(result_path)
    instances = load_instances(dataset_path)
    selected = select_rows(payload["rows"], int(args.top_k))

    table_rows = []
    records = []
    for row in selected:
        idx = int(row["instance"])
        record = {
            "instance": idx,
            "instance_name": row.get("instance_name", str(idx)),
            "problem_size": int(row.get("problem_size", len(instances[idx]))),
            "cost": float(row["cost"]),
            "reference_cost": float(row["reference_cost"]),
            "gap_pct": gap_pct(row),
            "aug_index": int(row.get("aug_index", -1)),
            "beam_index": int(row.get("beam_index", -1)),
            "tour": row["tour"],
        }
        records.append(record)
        table_rows.append(
            [
                record["instance_name"],
                str(record["problem_size"]),
                f'{record["reference_cost"]:.0f}',
                f'{record["cost"]:.0f}',
                f'{record["gap_pct"]:.2f}%',
            ]
        )

    (out_dir / "selected_routes.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (out_dir / "selected_routes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance", "instance_name", "problem_size", "cost", "reference_cost", "gap_pct", "aug_index", "beam_index"])
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in writer.fieldnames})

    headers = ["Instance", "N", "Opt.", "LINC SGBS", "Gap"]
    (out_dir / "tsplib_linc_sgbs_selected_routes.md").write_text(markdown_table(headers, table_rows), encoding="utf-8")
    (out_dir / "tsplib_linc_sgbs_selected_routes.tex").write_text(latex_table(headers, table_rows), encoding="utf-8")

    for record in records:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        plot_instance(ax, record, instances[int(record["instance"])])
        fig.savefig(out_dir / f'{record["instance_name"]}.png', dpi=240, bbox_inches="tight")
        fig.savefig(out_dir / f'{record["instance_name"]}.pdf', bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for ax, record in zip(axes.flat, records):
        plot_instance(ax, record, instances[int(record["instance"])])
    fig.tight_layout()
    fig.savefig(out_dir / "tsplib_linc_sgbs_selected_routes.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "tsplib_linc_sgbs_selected_routes.pdf", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"result_json": str(result_path), "out_dir": str(out_dir), "selected": [r["instance_name"] for r in records]}, indent=2))


if __name__ == "__main__":
    main()
