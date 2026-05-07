"""Build Solomon LINC SGBS route figures and summary tables.

The input result JSON must be produced with:

    python CVRPTW/exp/sgbs/eval_linc_sgbs.py --dataset solomon56 --include-routes ...

It selects the lowest-gap LINC SGBS instance from each Solomon class
(C1, C2, R1, R2, RC1, RC2), plots the decoded routes, and writes compact
Markdown/LaTeX tables for paper use.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]


CLASS_ORDER = ("C1", "C2", "R1", "R2", "RC1", "RC2")
DEFAULT_CLASSIC_CSV = ROOT / "paper" / "paper_tables" / "solomon_hgs_reference.csv"


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Solomon LINC SGBS best-class routes and tables.")
    parser.add_argument("--result-json", default="results/solomon56_linc_sgbs_routes.json")
    parser.add_argument("--dataset-pt", default="CVRPTW/data/solomon56.pt")
    parser.add_argument("--out-dir", default="results/solomon56_linc_sgbs_routes")
    parser.add_argument(
        "--classic-csv",
        default=str(DEFAULT_CLASSIC_CSV) if DEFAULT_CLASSIC_CSV.exists() else "",
        help="Optional Solomon per-instance CSV with vetted HGS/community reference values.",
    )
    parser.add_argument(
        "--classic-methods",
        default="HGS",
        help="Comma-separated columns to include from --classic-csv.",
    )
    return parser.parse_args()


def solomon_class(name: str) -> str:
    name = str(name).upper()
    for prefix in ("RC1", "RC2", "C1", "C2", "R1", "R2"):
        if name.startswith(prefix):
            return prefix
    return "other"


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if str(value).strip() == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = [row.get("instance", f"row_{idx}") for idx, row in enumerate(rows) if "tour" not in row]
    if missing:
        sample = ", ".join(map(str, missing[:5]))
        raise ValueError(f"Rows are missing tours; rerun SGBS with --include-routes. Examples: {sample}")
    return rows


def load_dataset(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    names = list(data.get("names", []))
    if not names:
        names = [f"inst_{idx:03d}" for idx in range(int(data["node_xy"].shape[0]))]
    return {
        "names": names,
        "depot_xy": data["depot_xy"].float(),
        "node_xy": data["node_xy"].float(),
        "bks_cost": data.get("bks_cost", None),
    }


def split_routes(tour: list[int], customer_count: int) -> list[list[int]]:
    routes: list[list[int]] = []
    current: list[int] = []
    for node in tour:
        node = int(node)
        if node == 0:
            if current:
                routes.append(current)
                current = []
            continue
        if 1 <= node <= customer_count:
            current.append(node)
    if current:
        routes.append(current)
    return routes


def route_stats(tour: list[int], customer_count: int) -> dict:
    visits = [int(node) for node in tour if 1 <= int(node) <= customer_count]
    unique = set(visits)
    return {
        "route_count": len(split_routes(tour, customer_count)),
        "visit_count": len(visits),
        "unique_visit_count": len(unique),
        "duplicate_visit_count": len(visits) - len(unique),
        "missing_count": customer_count - len(unique),
    }


def select_best_by_class(rows: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        cls = solomon_class(str(row.get("instance", "")))
        if cls in CLASS_ORDER:
            grouped[cls].append(row)
    selected = {}
    for cls in CLASS_ORDER:
        cls_rows = grouped.get(cls, [])
        if not cls_rows:
            continue
        selected[cls] = min(cls_rows, key=lambda row: float(row.get("gap_pct", float("inf"))))
    return selected


def aggregate(rows: list[dict]) -> dict:
    distance_sum = sum(float(row["distance"]) for row in rows)
    bks_rows = [row for row in rows if "bks_cost" in row]
    bks_sum = sum(float(row["bks_cost"]) for row in bks_rows)
    out = {
        "count": len(rows),
        "obj": distance_sum / len(rows),
        "mean_gap_pct": sum(float(row.get("gap_pct", 0.0)) for row in bks_rows) / len(bks_rows)
        if bks_rows
        else None,
        "bks": bks_sum / len(bks_rows) if bks_rows else None,
        "aggregate_gap_pct": 100.0 * (distance_sum - bks_sum) / bks_sum if bks_sum else None,
    }
    return out


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


def write_class_tables(rows: list[dict], selected: dict[str, dict], out_dir: Path) -> None:
    table_rows = []
    for cls in CLASS_ORDER:
        cls_rows = [row for row in rows if solomon_class(str(row.get("instance", ""))) == cls]
        if not cls_rows:
            continue
        agg = aggregate(cls_rows)
        chosen = selected.get(cls, {})
        table_rows.append(
            [
                cls,
                str(agg["count"]),
                f'{agg["bks"]:.1f}' if agg["bks"] is not None else "--",
                f'{agg["obj"]:.1f}',
                f'{agg["aggregate_gap_pct"]:.2f}%' if agg["aggregate_gap_pct"] is not None else "--",
                str(chosen.get("instance", "--")),
                f'{float(chosen.get("gap_pct", 0.0)):.2f}%' if "gap_pct" in chosen else "--",
            ]
        )
    all_agg = aggregate(rows)
    table_rows.append(
        [
            "All",
            str(all_agg["count"]),
            f'{all_agg["bks"]:.1f}' if all_agg["bks"] is not None else "--",
            f'{all_agg["obj"]:.1f}',
            f'{all_agg["aggregate_gap_pct"]:.2f}%' if all_agg["aggregate_gap_pct"] is not None else "--",
            "--",
            "--",
        ]
    )
    headers = ["Class", "N", "BKS", "LINC SGBS", "Gap", "Best-gap instance", "Best gap"]
    (out_dir / "solomon_linc_sgbs_by_class.md").write_text(markdown_table(headers, table_rows), encoding="utf-8")
    (out_dir / "solomon_linc_sgbs_by_class.tex").write_text(latex_table(headers, table_rows), encoding="utf-8")


def read_classic_rows(path: Path) -> list[dict]:
    if not str(path) or not path.exists() or path.is_dir():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_method_table(rows: list[dict], classic_rows: list[dict], methods: list[str], out_dir: Path) -> None:
    method_rows = []
    if classic_rows:
        result_names = {str(row.get("instance", "")) for row in rows}
        classic_rows = [row for row in classic_rows if str(row.get("Instance", "")) in result_names]
        bks_values = [safe_float(row.get("BKS", row.get("HGS", row.get("Optimal")))) for row in classic_rows]
        bks_values = [value for value in bks_values if value is not None]
        bks_mean = sum(bks_values) / len(bks_values) if bks_values else None
        for method in methods:
            values = [safe_float(row.get(method)) for row in classic_rows]
            values = [value for value in values if value is not None]
            if not values:
                continue
            obj = sum(values) / len(values)
            label = "HGS" if method in ("HGS", "Optimal", "BKS") else method
            gap = 100.0 * (obj - bks_mean) / bks_mean if bks_mean else None
            method_rows.append([label, str(len(values)), f"{obj:.1f}", f"{gap:.2f}%" if gap is not None else "--"])

    agg = aggregate(rows)
    method_rows.append(
        [
            "LINC SGBS",
            str(agg["count"]),
            f'{agg["obj"]:.1f}',
            f'{agg["aggregate_gap_pct"]:.2f}%' if agg["aggregate_gap_pct"] is not None else "--",
        ]
    )
    headers = ["Method", "N", "Obj.", "Gap vs BKS"]
    (out_dir / "solomon_method_comparison.md").write_text(markdown_table(headers, method_rows), encoding="utf-8")
    (out_dir / "solomon_method_comparison.tex").write_text(latex_table(headers, method_rows), encoding="utf-8")


def plot_instance(ax, row: dict, dataset: dict, out_path: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    names = dataset["names"]
    idx = names.index(str(row["instance"]))
    depot = dataset["depot_xy"][idx].reshape(-1, 2)[0].numpy()
    customers = dataset["node_xy"][idx].numpy()
    routes = split_routes([int(node) for node in row["tour"]], customers.shape[0])
    cmap = plt.get_cmap("tab20")

    ax.scatter(customers[:, 0], customers[:, 1], s=13, c="#5d6675", alpha=0.75, linewidths=0)
    ax.scatter([depot[0]], [depot[1]], s=95, marker="*", c="#111111", zorder=5)
    for route_idx, route in enumerate(routes):
        coords = [depot]
        coords.extend(customers[node - 1] for node in route)
        coords.append(depot)
        xs = [float(point[0]) for point in coords]
        ys = [float(point[1]) for point in coords]
        ax.plot(xs, ys, color=cmap(route_idx % 20), linewidth=1.35, alpha=0.9)

    gap = row.get("gap_pct", None)
    title = f'{row["instance"]} | obj {float(row["distance"]):.1f}'
    if gap is not None:
        title += f" | gap {float(gap):.2f}%"
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.08)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#c8ccd2")

    if out_path is not None:
        ax.figure.savefig(out_path, dpi=220, bbox_inches="tight")


def write_selected_outputs(selected: dict[str, dict], dataset: dict, out_dir: Path) -> None:
    customer_count = int(dataset["node_xy"].shape[1])
    records = []
    for cls, row in selected.items():
        stats = route_stats([int(node) for node in row["tour"]], customer_count)
        record = {
            "class": cls,
            "instance": row["instance"],
            "distance": float(row["distance"]),
            "bks_cost": float(row["bks_cost"]) if "bks_cost" in row else None,
            "gap_pct": float(row["gap_pct"]) if "gap_pct" in row else None,
            "aug_index": int(row.get("aug_index", -1)),
            "beam_index": int(row.get("beam_index", -1)),
            **stats,
            "tour": row["tour"],
        }
        records.append(record)
    (out_dir / "selected_routes.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    csv_fields = [
        "class",
        "instance",
        "distance",
        "bks_cost",
        "gap_pct",
        "aug_index",
        "beam_index",
        "route_count",
        "visit_count",
        "unique_visit_count",
        "duplicate_visit_count",
        "missing_count",
    ]
    with (out_dir / "selected_routes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in csv_fields})


def main() -> None:
    args = parse_args()
    result_path = resolve(args.result_json)
    dataset_path = resolve(args.dataset_pt)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(result_path)
    dataset = load_dataset(dataset_path)
    selected = select_best_by_class(rows)
    if set(selected) != set(CLASS_ORDER):
        missing = sorted(set(CLASS_ORDER) - set(selected))
        raise ValueError(f"Missing Solomon classes in result rows: {missing}")

    write_selected_outputs(selected, dataset, out_dir)
    write_class_tables(rows, selected, out_dir)

    classic_path = Path(args.classic_csv) if args.classic_csv else Path()
    classic_rows = read_classic_rows(classic_path)
    methods = [method.strip() for method in args.classic_methods.split(",") if method.strip()]
    write_method_table(rows, classic_rows, methods, out_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cls in CLASS_ORDER:
        fig, ax = plt.subplots(figsize=(5.0, 4.2))
        plot_instance(ax, selected[cls], dataset)
        fig.savefig(out_dir / f"{cls}_{selected[cls]['instance']}.png", dpi=240, bbox_inches="tight")
        fig.savefig(out_dir / f"{cls}_{selected[cls]['instance']}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for ax, cls in zip(axes.flat, CLASS_ORDER):
        plot_instance(ax, selected[cls], dataset)
        ax.text(0.02, 0.96, cls, transform=ax.transAxes, ha="left", va="top", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "solomon_linc_sgbs_selected_routes.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / "solomon_linc_sgbs_selected_routes.pdf", bbox_inches="tight")
    plt.close(fig)

    print(
        json.dumps(
            {
                "result_json": str(result_path),
                "dataset_pt": str(dataset_path),
                "out_dir": str(out_dir),
                "selected": {cls: selected[cls]["instance"] for cls in CLASS_ORDER},
                "classic_csv_used": str(classic_path) if classic_rows else "",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
