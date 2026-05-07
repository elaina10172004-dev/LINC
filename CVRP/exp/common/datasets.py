"""Dataset resolution helpers for CVRP evaluation entrypoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


TASK_ROOT = Path(__file__).resolve().parents[2]


DATASETS = {
    "kool100": {
        "path": "data/vrp100_test_seed1234.pkl",
        "problem_size": 100,
        "episodes": 10000,
        "kind": "pkl",
    },
    "kool150": {
        "path": "data/vrp150_test_small_seed1235.pkl",
        "problem_size": 150,
        "episodes": 1000,
        "kind": "pkl",
    },
    "xml100": {
        "path": "data/xml100_full_download/instances/instances",
        "problem_size": 100,
        "episodes": 10000,
        "kind": "xml",
    },
}

ALIASES = {
    "cvrp100": "kool100",
    "cvrp150": "kool150",
    "cvrpxml": "xml100",
}


def add_dataset_args(parser, default="kool100"):
    parser.add_argument(
        "--dataset",
        default=default,
        choices=sorted(set(DATASETS) | set(ALIASES)),
        help="Named dataset. Use --data to override the path.",
    )
    parser.add_argument("--data", default="", help="Explicit dataset path overriding --dataset.")
    parser.add_argument("--bks-json", default="", help="BKS JSON path for CVRPLIB/XML datasets.")
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")


def _resolve(path):
    path = Path(path)
    if not path.is_absolute():
        path = TASK_ROOT / path
    return path


def resolve_dataset(args):
    dataset_name = ALIASES.get(args.dataset, args.dataset)
    spec = dict(DATASETS[dataset_name])
    path = _resolve(args.data or spec["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"CVRP dataset '{dataset_name}' not found at {path}. "
            f"Place the file/directory under {TASK_ROOT / 'data'} or pass --data."
        )
    bks_json = ""
    if args.bks_json:
        bks_json_path = _resolve(args.bks_json)
        if not bks_json_path.exists():
            raise FileNotFoundError(f"CVRP BKS JSON not found at {bks_json_path}")
        bks_json = str(bks_json_path)
    if not getattr(args, "problem_size", 0):
        args.problem_size = int(spec.get("problem_size", 0) or 0)
    elif spec.get("problem_size") and not args.data:
        args.problem_size = int(spec["problem_size"])
    if getattr(args, "episodes", 0) <= 0:
        args.episodes = int(spec.get("episodes", 0) or 0)
    args.data = str(path)
    args.bks_json = bks_json
    return SimpleNamespace(
        name=dataset_name,
        path=path,
        bks_json=Path(bks_json) if bks_json else None,
        kind=spec["kind"],
        problem_size=int(args.problem_size),
    )
