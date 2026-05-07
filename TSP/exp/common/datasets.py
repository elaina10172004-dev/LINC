"""Dataset resolution helpers for TSP evaluation entrypoints."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace


TASK_ROOT = Path(__file__).resolve().parents[2]

TSPLIB_50_200_NAMES = [
    "berlin52",
    "bier127",
    "ch130",
    "ch150",
    "d198",
    "eil101",
    "eil51",
    "eil76",
    "kroA100",
    "kroA150",
    "kroA200",
    "kroB100",
    "kroB150",
    "kroB200",
    "kroC100",
    "kroD100",
    "kroE100",
    "lin105",
    "pr107",
    "pr124",
    "pr136",
    "pr144",
    "pr152",
    "pr76",
    "rat195",
    "rat99",
    "rd100",
    "st70",
    "u159",
]

TSPLIB_50_200_REFERENCE = [
    7542.0,
    118282.0,
    6110.0,
    6528.0,
    15780.0,
    629.0,
    426.0,
    538.0,
    21282.0,
    26524.0,
    29368.0,
    22141.0,
    26130.0,
    29437.0,
    20749.0,
    21294.0,
    22068.0,
    14379.0,
    44303.0,
    59030.0,
    96772.0,
    58537.0,
    73682.0,
    108159.0,
    2323.0,
    1211.0,
    7910.0,
    675.0,
    42080.0,
]


DATASETS = {
    "kool100": {
        "path": "data/tsp100_test_seed1234.pkl",
        "problem_size": 100,
        "episodes": 10000,
    },
    "kool150": {
        "path": "data/tsp150_test_small_seed1235.pkl",
        "problem_size": 150,
        "episodes": 1000,
    },
    "tsplib_50_200": {
        "path": "data/tsplib_50_200.pkl",
        "variable_size": True,
        "episodes": 0,
    },
}

ALIASES = {
    "tsp100": "kool100",
    "tsp150": "kool150",
    "tsplib": "tsplib_50_200",
}


def add_dataset_args(parser, default="kool100"):
    parser.add_argument(
        "--dataset",
        default=default,
        choices=sorted(set(DATASETS) | set(ALIASES)),
        help="Named dataset. Use --data to override the file path.",
    )
    parser.add_argument("--data", default="", help="Explicit dataset path overriding --dataset.")
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    parser.add_argument("--parallel", action="store_true", help="Parallel subprocess per size group for variable-size datasets.")


def resolve_dataset(args):
    dataset_name = ALIASES.get(args.dataset, args.dataset)
    spec = dict(DATASETS[dataset_name])
    raw_path = args.data or spec["path"]
    path = Path(raw_path)
    if not path.is_absolute():
        path = TASK_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"TSP dataset '{dataset_name}' not found at {path}. "
            f"Place the file under {TASK_ROOT / 'data'} or pass --data."
        )
    if not getattr(args, "problem_size", 0):
        args.problem_size = int(spec.get("problem_size", 0) or 0)
    elif spec.get("problem_size") and int(args.problem_size) != int(spec["problem_size"]):
        if not args.data:
            args.problem_size = int(spec["problem_size"])
    if getattr(args, "episodes", 0) <= 0:
        args.episodes = int(spec.get("episodes", 0) or 0)
    args.data = str(path)
    return SimpleNamespace(
        name=dataset_name,
        path=path,
        variable_size=bool(spec.get("variable_size", False)),
        problem_size=int(getattr(args, "problem_size", 0) or 0),
    )


def load_pickle_instances(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def grouped_tsp_instances(path: Path, max_instances=0):
    data = load_pickle_instances(path)
    if max_instances and max_instances > 0:
        data = data[: int(max_instances)]
    groups = OrderedDict()
    for inst in data:
        groups.setdefault(len(inst), []).append(inst)
    return groups


def indexed_tsp_groups(path: Path, max_instances=0):
    data = load_pickle_instances(path)
    if max_instances and max_instances > 0:
        data = data[: int(max_instances)]
    groups = OrderedDict()
    for index, inst in enumerate(data):
        groups.setdefault(len(inst), []).append((index, inst))
    return groups


def tsplib_metadata(dataset_name: str, index: int, problem_size: int) -> dict:
    if dataset_name != "tsplib_50_200" or index >= len(TSPLIB_50_200_NAMES):
        return {}
    return {
        "instance_index": int(index),
        "instance_name": TSPLIB_50_200_NAMES[index],
        "reference_cost": float(TSPLIB_50_200_REFERENCE[index]),
        "problem_size": int(problem_size),
    }


def _replace_arg(argv, flag, value):
    """Replace or append a flag in an argv list."""
    try:
        idx = argv.index(flag)
        argv = list(argv)
        argv[idx + 1] = value
        return argv
    except (ValueError, IndexError):
        return list(argv) + [flag, value]


def run_variable_size_parallel(args, dataset_info):
    """Launch one subprocess per size group, all in parallel. Aggregate scores from stdout."""
    groups = indexed_tsp_groups(dataset_info.path, getattr(args, "episodes", 0))
    total = sum(len(v) for v in groups.values())
    print(f"[dataset] {dataset_info.name}: {total} instances across {len(groups)} sizes (PARALLEL)")

    script_argv = list(getattr(args, "_argv", sys.argv))
    if script_argv:
        script_path = Path(script_argv[0])
        if not script_path.is_absolute():
            for candidate in (TASK_ROOT / script_path, TASK_ROOT.parent / script_path):
                if candidate.exists():
                    script_path = candidate
                    break
        script_argv[0] = str(script_path.resolve())
    base_argv = [a for a in script_argv if a not in ("--parallel",)]
    for flag in ("--dataset", "--data", "--problem-size", "--episodes", "--output-json"):
        try:
            idx = base_argv.index(flag)
            del base_argv[idx + 1]
            del base_argv[idx]
        except (ValueError, IndexError):
            pass

    with tempfile.TemporaryDirectory(prefix="linc_tsp_parallel_") as tmp:
        tmp_dir = Path(tmp)
        processes = []

        for problem_size, indexed_instances in groups.items():
            instances = [inst for _, inst in indexed_instances]
            group_path = tmp_dir / f"{dataset_info.name}_n{problem_size}.pkl"
            group_result_path = tmp_dir / f"{dataset_info.name}_n{problem_size}_result.json"
            with group_path.open("wb") as f:
                pickle.dump(instances, f)
            cmd = [sys.executable] + base_argv + [
                "--data", str(group_path),
                "--problem-size", str(problem_size),
                "--episodes", str(len(instances)),
                "--output-json", str(group_result_path),
            ]
            print(f"[dataset] launching n={problem_size}, instances={len(instances)}")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append((problem_size, indexed_instances, group_result_path, proc))

        failed = []
        group_results = []
        total_cost = 0.0
        total_inst = 0
        t0 = time.perf_counter()
        for problem_size, indexed_instances, group_result_path, proc in processes:
            count = len(indexed_instances)
            stdout, stderr = proc.communicate(timeout=600)
            if proc.returncode != 0:
                failed.append(problem_size)
                print(f"[dataset] FAIL n={problem_size}: {stderr[-200:]}")
                continue
            # Try to parse JSON block from output (may be multi-line)
            try:
                if group_result_path.exists():
                    result = json.loads(group_result_path.read_text(encoding="utf-8"))
                else:
                    # Find the outermost JSON object
                    start = stdout.find("{")
                    end = stdout.rfind("}")
                    if start >= 0 and end > start:
                        result = json.loads(stdout[start:end + 1])
                    else:
                        result = json.loads(stdout.strip().split("\n")[-1])
                if "mean_cost" in result:
                    total_cost += result["mean_cost"] * count
                    total_inst += count
                elif "aug_score_mean" in result:
                    total_cost += result["aug_score_mean"] * count
                    total_inst += count
                group_results.append((problem_size, indexed_instances, result))
            except (json.JSONDecodeError, KeyError):
                pass

        wall = time.perf_counter() - t0
        mean = total_cost / total_inst if total_inst > 0 else None
        if failed:
            print(f"[dataset] FAILED groups (n={failed})")
        if mean is not None:
            print(f"[dataset] TSPLIB mean: {mean:.1f}")
        print(f"[dataset] parallel wall time: {wall:.1f}s")
        output_json = getattr(args, "output_json", "")
        if output_json:
            output_path = Path(output_json)
            if not output_path.is_absolute():
                output_path = TASK_ROOT / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            groups_payload = []
            for problem_size, indexed_instances, result in group_results:
                count = len(indexed_instances)
                groups_payload.append(
                    {
                        "problem_size": int(problem_size),
                        "instances": int(count),
                        "mean_cost": result.get("mean_cost", result.get("aug_score_mean")),
                        "elapsed_sec": result.get("elapsed_sec"),
                        "metric": result.get("metric"),
                        "mode": result.get("mode"),
                        "pomo_size": result.get("pomo_size"),
                        "rollouts_per_instance": result.get("rollouts_per_instance"),
                        "eval_type": result.get("eval_type"),
                        "beam_width": result.get("beam_width"),
                        "expand_k": result.get("expand_k"),
                        "start_mode": result.get("start_mode"),
                    }
                )
                for row in result.get("rows", []):
                    row = dict(row)
                    local_index = int(row.get("instance", row.get("index", len(rows))))
                    if 0 <= local_index < len(indexed_instances):
                        global_index = int(indexed_instances[local_index][0])
                        row["instance"] = global_index
                        row.update(tsplib_metadata(dataset_info.name, global_index, int(problem_size)))
                    else:
                        row["problem_size"] = int(problem_size)
                    rows.append(row)
            payload = {
                "dataset": dataset_info.name,
                "variable_size": True,
                "num_instances": int(total_inst),
                "mean_cost": None if mean is None else float(mean),
                "elapsed_sec": float(wall),
                "failed_groups": [int(x) for x in failed],
                "groups": groups_payload,
                "rows": rows,
            }
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return wall


def run_variable_size_tsp(args, dataset_info, runner):
    """Run a fixed-size evaluator once per problem size group (sequential)."""

    groups = indexed_tsp_groups(dataset_info.path, getattr(args, "episodes", 0))
    total = sum(len(v) for v in groups.values())
    print(f"[dataset] {dataset_info.name}: {total} instances across {len(groups)} sizes")

    group_results = []
    total_cost = 0.0
    total_inst = 0
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="linc_tsp_groups_") as tmp:
        tmp_dir = Path(tmp)
        for problem_size, indexed_instances in groups.items():
            instances = [inst for _, inst in indexed_instances]
            group_path = tmp_dir / f"{dataset_info.name}_n{problem_size}.pkl"
            with group_path.open("wb") as f:
                pickle.dump(instances, f)
            group_args = SimpleNamespace(**vars(args))
            group_args.data = str(group_path)
            group_args.problem_size = int(problem_size)
            group_args.episodes = len(instances)
            group_args.batch_size = min(int(args.batch_size), len(instances))
            group_args.output_json = ""
            print(f"[dataset] running n={problem_size}, instances={len(instances)}")
            result = runner(group_args)
            if result is None:
                continue
            count = len(indexed_instances)
            mean_cost = result.get("mean_cost", result.get("aug_score_mean"))
            if mean_cost is not None:
                total_cost += float(mean_cost) * count
                total_inst += count
            group_results.append((problem_size, indexed_instances, result))

    wall = time.perf_counter() - t0
    mean = total_cost / total_inst if total_inst > 0 else None
    if mean is not None:
        print(f"[dataset] TSPLIB mean: {mean:.1f}")
    print(f"[dataset] sequential wall time: {wall:.1f}s")

    output_json = getattr(args, "output_json", "")
    if output_json:
        output_path = Path(output_json)
        if not output_path.is_absolute():
            output_path = TASK_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        groups_payload = []
        for problem_size, indexed_instances, result in group_results:
            count = len(indexed_instances)
            groups_payload.append(
                {
                    "problem_size": int(problem_size),
                    "instances": int(count),
                    "mean_cost": result.get("mean_cost", result.get("aug_score_mean")),
                    "elapsed_sec": result.get("elapsed_sec"),
                    "metric": result.get("metric"),
                    "mode": result.get("mode"),
                    "pomo_size": result.get("pomo_size"),
                    "rollouts_per_instance": result.get("rollouts_per_instance"),
                    "z_samples": result.get("z_samples"),
                    "eval_type": result.get("eval_type"),
                }
            )
            for row in result.get("rows", []):
                row = dict(row)
                local_index = int(row.get("instance", row.get("index", len(rows))))
                if 0 <= local_index < len(indexed_instances):
                    global_index = int(indexed_instances[local_index][0])
                    row["instance"] = global_index
                    row.update(tsplib_metadata(dataset_info.name, global_index, int(problem_size)))
                else:
                    row["problem_size"] = int(problem_size)
                rows.append(row)
        payload = {
            "dataset": dataset_info.name,
            "variable_size": True,
            "num_instances": int(total_inst),
            "mean_cost": None if mean is None else float(mean),
            "elapsed_sec": float(wall),
            "failed_groups": [],
            "groups": groups_payload,
            "rows": rows,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return wall
