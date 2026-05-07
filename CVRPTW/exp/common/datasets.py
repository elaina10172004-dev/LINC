"""Dataset resolution helpers for CVRPTW evaluation entrypoints."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


TASK_ROOT = Path(__file__).resolve().parents[2]


DATASETS = {
    "synthetic100": {
        "path": "data/LINC_CVRPTW100_test_10k_seed1234.pt",
        "problem_size": 100,
        "episodes": 10000,
    },
    "solomon56": {
        "path": "data/solomon56.pt",
        "problem_size": 100,
        "episodes": 56,
    },
    "solomon56_n50": {
        "path": "data/solomon56_n50.pt",
        "problem_size": 50,
        "episodes": 56,
    },
    "homberger200": {
        "path": "data/homberger200_60.pt",
        "problem_size": 200,
        "episodes": 60,
    },
}

ALIASES = {
    "cvrptw100": "synthetic100",
    "synthetic": "synthetic100",
    "solomon": "solomon56",
    "solomon50": "solomon56_n50",
    "homberger": "homberger200",
}


def add_dataset_args(parser, default="synthetic100"):
    parser.add_argument(
        "--dataset",
        default=default,
        choices=sorted(set(DATASETS) | set(ALIASES)),
        help="Named dataset. Use --data to override the file path.",
    )
    parser.add_argument("--data", default="", help="Explicit dataset path overriding --dataset.")
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")


def resolve_dataset(args):
    dataset_name = ALIASES.get(args.dataset, args.dataset)
    spec = dict(DATASETS[dataset_name])
    path = Path(args.data or spec["path"])
    if not path.is_absolute():
        path = TASK_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"CVRPTW dataset '{dataset_name}' not found at {path}. "
            f"Place the file under {TASK_ROOT / 'data'} or pass --data."
        )
    if not getattr(args, "problem_size", 0):
        args.problem_size = int(spec.get("problem_size", 0) or 0)
    elif spec.get("problem_size") and not args.data:
        args.problem_size = int(spec["problem_size"])
    if getattr(args, "episodes", 0) <= 0:
        args.episodes = int(spec.get("episodes", 0) or 0)
    args.data = str(path)
    return SimpleNamespace(
        name=dataset_name,
        path=path,
        problem_size=int(args.problem_size),
    )
