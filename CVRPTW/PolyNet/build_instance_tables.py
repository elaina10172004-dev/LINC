"""Generate per-instance appendix tables for Solomon56 and TSPLIB50-200."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "paper_tables"
SOURCE_DIR = OUT_DIR / "source_results"
SOLOMON_HGS_CSV = OUT_DIR / "solomon_hgs_reference.csv"


def load_json(path: str) -> dict:
    path_obj = Path(path)
    if path_obj.is_absolute():
        resolved = path_obj
    else:
        source_copy = SOURCE_DIR / path_obj.name
        resolved = source_copy if source_copy.exists() else ROOT / path_obj
    return json.loads(resolved.read_text(encoding="utf-8"))


def load_source_json(filename: str, required: bool = True) -> dict:
    resolved = SOURCE_DIR / filename
    if not resolved.exists():
        if required:
            raise FileNotFoundError(f"Missing paper source result: {resolved}")
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def safe_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def fmt(value, digits=1) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def table_env(caption: str, label: str, headers: list[str], rows: list[list[str]], tabcolsep=2) -> str:
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\scriptsize",
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}",
        r"\renewcommand{\arraystretch}{0.88}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}{colspec}@{{}}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def read_solomon_hgs() -> dict[str, dict]:
    with SOLOMON_HGS_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["Instance"]: row for row in csv.DictReader(f) if row.get("Instance")}


def rows_by_name(payload: dict, value_key="distance") -> dict[str, float]:
    return {str(row["instance"]): float(row[value_key]) for row in payload.get("rows", [])}


def rows_by_index(payload: dict, names: list[str], value_key="aug_cost") -> dict[str, float]:
    out = {}
    for row in payload.get("rows", []):
        index = int(row.get("index", row.get("instance", len(out))))
        if 0 <= index < len(names):
            out[names[index]] = float(row.get(value_key, row.get("cost")))
    return out


def build_solomon() -> str:
    hgs_reference = read_solomon_hgs()
    names = list(hgs_reference)
    pomo_greedy = rows_by_name(load_source_json("solomon56_pomo_greedy_rows.json"))
    pomo_sampling = rows_by_name(load_source_json("solomon56_pomo_sampling_rows.json"))
    polynet = rows_by_name(load_source_json("solomon56_polynet_sampling_rows.json", required=False))
    linc = rows_by_name(load_source_json("solomon56_linc_sampling_rows.json"))
    pomo_sgbs = rows_by_index(load_source_json("solomon56_pomo_sgbs_rows.json"), names)
    polynet_sgbs = rows_by_name(load_source_json("solomon56_polynet_sgbs_rows.json"))
    linc_sgbs = rows_by_name(load_source_json("solomon56_linc_sgbs_rows.json"))

    headers = [
        "Instance",
        "HGS",
        "POMO-g",
        "POMO-s",
        "PolyNet",
        "LINC",
        "POMO-SGBS",
        "PolyNet-SGBS",
        "LINC-SGBS",
    ]
    rows = []
    mean_columns = [[] for _ in headers[1:]]
    for name in names:
        c = hgs_reference[name]
        values = [
            safe_float(c.get("HGS", c.get("BKS"))),
            pomo_greedy.get(name),
            pomo_sampling.get(name),
            polynet.get(name),
            linc.get(name),
            pomo_sgbs.get(name),
            polynet_sgbs.get(name),
            linc_sgbs.get(name),
        ]
        for idx, value in enumerate(values):
            mean_columns[idx].append(value)
        rows.append([latex_escape(name)] + [fmt(value) for value in values])
    rows.append([r"\textbf{Mean}"] + [r"\textbf{" + fmt(mean(col)) + "}" for col in mean_columns])
    return table_env(
        "Per-instance Solomon56 results. HGS is the community best-known reference; other columns are evaluated in this repository. Values are route distances.",
        "tab:appendix_solomon_instance_methods",
        headers,
        rows,
        tabcolsep=2,
    )


def tsplib_rows(payload: dict, value_key="cost", fallback_meta=None) -> dict[str, dict]:
    out = {}
    fallback_meta = fallback_meta or {}
    for row in payload.get("rows", []):
        name = row.get("instance_name")
        meta = None
        if not name:
            key = (int(row["problem_size"]), int(row.get("instance", row.get("index", 0))))
            meta = fallback_meta.get(key)
            name = None if meta is None else meta["instance_name"]
        if not name:
            continue
        reference = row.get("reference_cost")
        if reference is None and meta is not None:
            reference = meta["reference_cost"]
        out[str(name)] = {
            "n": int(row["problem_size"]),
            "reference": float(reference),
            "value": float(row.get(value_key, row.get("cost"))),
        }
    return out


def build_tsplib() -> str:
    linc_payload = load_json("TSP/results/instance_tables/tsplib_linc_sgbs_rows.json")
    fallback_meta = {
        (int(row["problem_size"]), int(row.get("instance", row.get("index", 0)))): {
            "instance_name": row["instance_name"],
            "reference_cost": row["reference_cost"],
        }
        for row in linc_payload.get("rows", [])
    }
    pomo_greedy = tsplib_rows(load_json("TSP/results/instance_tables/tsplib_pomo_greedy_rows.json"), fallback_meta=fallback_meta)
    pomo_sampling = tsplib_rows(load_json("TSP/results/instance_tables/tsplib_pomo_sampling_rows.json"), fallback_meta=fallback_meta)
    pomo_sgbs = tsplib_rows(load_json("TSP/results/instance_tables/tsplib_pomo_sgbs_rows.json"), "aug_cost", fallback_meta)
    polynet_sgbs = tsplib_rows(load_json("TSP/results/instance_tables/tsplib_polynet_sgbs_rows.json"), fallback_meta=fallback_meta)
    linc_sgbs = tsplib_rows(linc_payload)
    names = sorted(linc_sgbs)

    headers = ["Instance", "$n$", "Concorde", "POMO-g", "POMO-s", "POMO-SGBS", "PolyNet-SGBS", "LINC-SGBS"]
    rows = []
    mean_columns = [[] for _ in headers[2:]]
    for name in names:
        values = [
            linc_sgbs[name]["reference"],
            pomo_greedy[name]["value"],
            pomo_sampling[name]["value"],
            pomo_sgbs[name]["value"],
            polynet_sgbs[name]["value"],
            linc_sgbs[name]["value"],
        ]
        for idx, value in enumerate(values):
            mean_columns[idx].append(value)
        rows.append([latex_escape(name), str(linc_sgbs[name]["n"])] + [fmt(value, 0) for value in values])
    rows.append([r"\textbf{Mean}", "--"] + [r"\textbf{" + fmt(mean(col), 1) + "}" for col in mean_columns])
    return table_env(
        "Per-instance TSPLIB50--200 results. Values use EUC\\_2D integer distances.",
        "tab:appendix_tsplib_instance_methods",
        headers,
        rows,
        tabcolsep=2,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    solomon = build_solomon()
    tsplib = build_tsplib()
    (OUT_DIR / "solomon_instance_methods.tex").write_text(solomon, encoding="utf-8")
    (OUT_DIR / "tsplib_instance_methods.tex").write_text(tsplib, encoding="utf-8")
    print(OUT_DIR / "solomon_instance_methods.tex")
    print(OUT_DIR / "tsplib_instance_methods.tex")


if __name__ == "__main__":
    main()
