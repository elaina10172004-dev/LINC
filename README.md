# LINC

LINC is a neural combinatorial optimization architecture for TSP, CVRP, and
CVRPTW. The repository keeps the same high-level layout for all three tasks:

```text
TSP|CVRP|CVRPTW/
|-- LINC/              # LINC model, env, trainer, tester
|-- PolyNet/           # PolyNet baseline
|-- POMO/              # POMO baseline, when applicable
|-- exp/
|   |-- common/        # Dataset resolution and task-local helpers
|   |-- sampling/      # eval_pomo.py, eval_polynet.py, eval_linc.py
|   |-- sgbs/          # eval_*_sgbs.py plus method-specific *_sgbs_impl.py
|   `-- training/      # train_linc.py
|-- models/            # Selected checkpoints
`-- data/              # Evaluation datasets or dataset README
```

## Setup

Use Python 3.10+ with PyTorch and NumPy. Full reproduction runs require a CUDA
GPU; CPU mode is intended only for small smoke tests.

```bash
pip install torch numpy
```

CVRPTW POMO sampling/greedy evaluation additionally uses RL4CO and TensorDict:

```bash
pip install rl4co tensordict
```

## Datasets

TSP:

- `kool100`: `TSP/data/tsp100_test_seed1234.pkl`
- `kool150`: `TSP/data/tsp150_test_small_seed1235.pkl`
- `tsplib_50_200`: `TSP/data/tsplib_50_200.pkl`

CVRP:

- `kool100`: `CVRP/data/vrp100_test_seed1234.pkl`
- `kool150`: `CVRP/data/vrp150_test_small_seed1235.pkl`
- `xml100`: CVRPLIB/XML `.vrp` files unpacked at
  `CVRP/data/xml100_full_download/instances/instances`

CVRPTW:

- `synthetic100`: `CVRPTW/data/LINC_CVRPTW100_test_10k_seed1234.pt`
- `solomon56`: `CVRPTW/data/solomon56.pt`
- `homberger200`: `CVRPTW/data/homberger200_60.pt`

## Evaluation Protocol

For reproduction runs on CUDA, set the PyTorch allocator configuration before
running the commands below:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This reduces CUDA memory fragmentation for the large `z * aug` decoding
batches used by LINC and PolyNet. Keep the command arguments unchanged unless
you are intentionally running a reduced-budget smoke test.

Dataset defaults:

- `kool100` / `synthetic100` / `xml100`: 10000 instances.
- `kool150`: 1000-instance small set.
- `solomon56`: 56 instances.
- `homberger200`: 60 instances.
- `tsplib_50_200`: 29 instances.

POMO table semantics:

- POMO greedy is one deterministic start with 8 augmentations: `1 * aug8`.
- POMO sampling is deterministic greedy from all starts with 8 augmentations:
  `n * aug8`. It is not stochastic action sampling.

PolyNet/LINC ordinary sampling follows the PolyNet protocol:
`--z-samples 800 --aug 8`. TSP uses `--eval-type sampling` for n100/n150.
Official PolyNet CVRP/CVRPTW uses `--eval-type sampling` for n100/n200 and
`--eval-type greedy` for n150.
For CVRPTW normalization, POMO and PolyNet use horizon scaling, while LINC uses
grid scaling. The CVRPTW sampling commands below set these modes explicitly;
the CVRPTW SGBS runners use the same method-specific defaults.

Every evaluation command accepts `--episodes` for reduced-budget smoke tests and
`--output-json` for structured outputs. JSON files include aggregate metrics and
per-instance rows when supported.

CVRPTW LINC training and the shared CVRPTW LINC/PolyNet/SGBS evaluation
environment use a depot-return feasibility mask to ensure decoded routes can
return to the depot within the depot time window.

For CVRP XML evaluation, pass `--bks-json path/to/xml100_full_bks.json` to
report gap metrics when BKS/reference costs are available; otherwise the XML
evaluators report solution costs only.

## Sampling Evaluation

TSP n100 checkpoints evaluated on the TSP150 small set:

```bash
python TSP/exp/sampling/eval_pomo.py --dataset kool150 --mode greedy --episodes 1000 --batch-size 2000 --aug 8 --seed 1234 --output-json results/tsp150_pomo_greedy.json
python TSP/exp/sampling/eval_pomo.py --dataset kool150 --mode sampling --episodes 1000 --batch-size 2000 --aug 8 --seed 1234 --output-json results/tsp150_pomo_sampling.json
python TSP/exp/sampling/eval_polynet.py --dataset kool150 --episodes 1000 --batch-size 500 --z-samples 800 --aug 8 --eval-type sampling --seed 1234 --output-json results/tsp150_polynet_sampling.json
python TSP/exp/sampling/eval_linc.py --dataset kool150 --episodes 1000 --batch-size 500 --z-samples 800 --aug 8 --eval-type sampling --seed 1234 --output-json results/tsp150_linc_sampling.json
```

CVRP n100 checkpoints evaluated on the CVRP150 small set:

```bash
python CVRP/exp/sampling/eval_pomo.py --dataset kool150 --mode greedy --episodes 1000 --batch-size 2500 --aug 8 --seed 1234 --output-json results/cvrp150_pomo_greedy.json
python CVRP/exp/sampling/eval_pomo.py --dataset kool150 --mode sampling --episodes 1000 --batch-size 2500 --aug 8 --seed 1234 --output-json results/cvrp150_pomo_sampling.json
python CVRP/exp/sampling/eval_polynet.py --dataset kool150 --episodes 1000 --batch-size 500 --z-samples 800 --aug 8 --eval-type greedy --seed 1234 --output-json results/cvrp150_polynet_greedy.json
python CVRP/exp/sampling/eval_linc.py --dataset kool150 --episodes 1000 --batch-size 500 --z-samples 800 --aug 8 --eval-type greedy --seed 1234 --output-json results/cvrp150_linc_greedy.json
```

CVRPTW synthetic100:

```bash
python CVRPTW/exp/sampling/eval_pomo.py --dataset synthetic100 --mode greedy --episodes 10000 --batch-size 256 --aug 8 --scale-mode horizon --seed 1234 --output-json results/cvrptw100_pomo_greedy.json
python CVRPTW/exp/sampling/eval_pomo.py --dataset synthetic100 --mode sampling --episodes 10000 --batch-size 256 --aug 8 --scale-mode horizon --seed 1234 --output-json results/cvrptw100_pomo_sampling.json
python CVRPTW/exp/sampling/eval_polynet.py --dataset synthetic100 --episodes 10000 --batch-size 500 --z-samples 800 --aug 8 --eval-type sampling --input-scale-mode horizon --seed 1234 --output-json results/cvrptw100_polynet_sampling.json
python CVRPTW/exp/sampling/eval_linc.py --dataset synthetic100 --episodes 10000 --batch-size 500 --z-samples 800 --aug 8 --eval-type sampling --input-scale-mode grid --seed 1234 --output-json results/cvrptw100_linc_sampling.json
```

CVRPTW Solomon and Homberger:

```bash
python CVRPTW/exp/sampling/eval_pomo.py --dataset solomon56 --mode greedy --episodes 56 --batch-size 56 --aug 8 --scale-mode horizon --seed 1234 --output-json results/solomon56_pomo_greedy.json
python CVRPTW/exp/sampling/eval_pomo.py --dataset solomon56 --mode sampling --episodes 56 --batch-size 56 --aug 8 --scale-mode horizon --seed 1234 --output-json results/solomon56_pomo_sampling.json
python CVRPTW/exp/sampling/eval_polynet.py --dataset solomon56 --episodes 56 --batch-size 56 --z-samples 800 --aug 8 --eval-type sampling --input-scale-mode horizon --seed 1234 --output-json results/solomon56_polynet_sampling.json
python CVRPTW/exp/sampling/eval_linc.py --dataset solomon56 --episodes 56 --batch-size 56 --z-samples 800 --aug 8 --eval-type sampling --input-scale-mode grid --seed 1234 --output-json results/solomon56_linc_sampling.json
python CVRPTW/exp/sampling/eval_pomo.py --dataset homberger200 --mode greedy --episodes 60 --batch-size 60 --aug 8 --scale-mode horizon --seed 1234 --output-json results/homberger200_pomo_greedy.json
python CVRPTW/exp/sampling/eval_pomo.py --dataset homberger200 --mode sampling --episodes 60 --batch-size 60 --aug 8 --scale-mode horizon --seed 1234 --output-json results/homberger200_pomo_sampling.json
python CVRPTW/exp/sampling/eval_polynet.py --dataset homberger200 --episodes 60 --batch-size 60 --z-samples 800 --aug 8 --eval-type sampling --input-scale-mode horizon --seed 1234 --output-json results/homberger200_polynet_sampling.json
python CVRPTW/exp/sampling/eval_linc.py --dataset homberger200 --episodes 60 --batch-size 60 --z-samples 800 --aug 8 --eval-type sampling --input-scale-mode grid --seed 1234 --output-json results/homberger200_linc_sampling.json
```

TSP TSPLIB is variable-size and grouped by node count. LINC runs groups
sequentially by default; add `--parallel` for full parallelism.

```bash
python TSP/exp/sampling/eval_pomo.py --dataset tsplib_50_200 --mode sampling --batch-size 2000 --aug 8 --seed 1234 --output-json results/tsplib_pomo_sampling.json
python TSP/exp/sampling/eval_linc.py --dataset tsplib_50_200 --batch-size 100 --z-samples 800 --aug 8 --eval-type sampling --seed 1234 --output-json results/tsplib_linc_sampling.json
```

## SGBS Evaluation

Default table protocol is `beam=4`, `expand=4`, `aug=8`. POMO SGBS greedily
completes all POMO starts once, keeps the best `--beam-width` starts as initial
beams, and later steps expand `--expand-k - 1` alternatives.

TSP150 SGBS:

```bash
python TSP/exp/sgbs/eval_pomo_sgbs.py --dataset kool150 --episodes 1000 --batch-size 800 --beam-width 4 --expand-k 4 --aug 8 --seed 1234 --output-json results/tsp150_pomo_sgbs.json
python TSP/exp/sgbs/eval_polynet_sgbs.py --dataset kool150 --episodes 1000 --batch-size 1000 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/tsp150_polynet_sgbs.json
python TSP/exp/sgbs/eval_linc_sgbs.py --dataset kool150 --episodes 1000 --batch-size 1000 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/tsp150_linc_sgbs.json
```

CVRP150 SGBS:

```bash
python CVRP/exp/sgbs/eval_pomo_sgbs.py --dataset kool150 --episodes 1000 --batch-size 800 --beam-width 4 --expand-k 4 --aug 8 --seed 1234 --output-json results/cvrp150_pomo_sgbs.json
python CVRP/exp/sgbs/eval_polynet_sgbs.py --dataset kool150 --episodes 1000 --batch-size 800 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/cvrp150_polynet_sgbs.json
python CVRP/exp/sgbs/eval_linc_sgbs.py --dataset kool150 --episodes 1000 --batch-size 800 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/cvrp150_linc_sgbs.json
```

CVRPTW SGBS:

```bash
python CVRPTW/exp/sgbs/eval_pomo_sgbs.py --dataset synthetic100 --episodes 10000 --batch-size 3000 --beam-width 4 --expand-k 4 --aug 8 --seed 1234 --output-json results/cvrptw100_pomo_sgbs.json
python CVRPTW/exp/sgbs/eval_polynet_sgbs.py --dataset synthetic100 --episodes 10000 --batch-size 1500 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/cvrptw100_polynet_sgbs.json
python CVRPTW/exp/sgbs/eval_linc_sgbs.py --dataset synthetic100 --episodes 10000 --batch-size 1500 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/cvrptw100_linc_sgbs.json
python CVRPTW/exp/sgbs/eval_pomo_sgbs.py --dataset solomon56 --episodes 56 --batch-size 56 --beam-width 4 --expand-k 4 --aug 8 --seed 1234 --output-json results/solomon56_pomo_sgbs.json
python CVRPTW/exp/sgbs/eval_polynet_sgbs.py --dataset solomon56 --episodes 56 --batch-size 56 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/solomon56_polynet_sgbs.json
python CVRPTW/exp/sgbs/eval_linc_sgbs.py --dataset solomon56 --episodes 56 --batch-size 56 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/solomon56_linc_sgbs.json
python CVRPTW/exp/sgbs/eval_pomo_sgbs.py --dataset homberger200 --episodes 60 --batch-size 60 --beam-width 4 --expand-k 4 --aug 8 --seed 1234 --output-json results/homberger200_pomo_sgbs.json
python CVRPTW/exp/sgbs/eval_polynet_sgbs.py --dataset homberger200 --episodes 60 --batch-size 60 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/homberger200_polynet_sgbs.json
python CVRPTW/exp/sgbs/eval_linc_sgbs.py --dataset homberger200 --episodes 60 --batch-size 60 --beam-width 4 --expand-k 4 --z-samples 128 --aug 8 --seed 1234 --output-json results/homberger200_linc_sgbs.json
```

## Training

```bash
python TSP/exp/training/train_linc.py
python CVRP/exp/training/train_linc.py
python CVRPTW/exp/training/train_linc.py
```

Default training settings match the reported n=100 checkpoints. TSP and CVRP
warm-start from PolyNet checkpoints; CVRPTW trains from scratch on the LINC
synthetic CVRPTW distribution. CVRPTW envelope generation can slow online
training; for large runs, pre-generate large datasets and cycle through them
offline.

## Ablation

```bash
python CVRPTW/exp/ablation/run_ablation.py --stage both --variants baseline,full_linc,no_local,naive_mlp,no_gateattn,no_depth_mixer,no_soft_top1,group_mean,centered_mlp,raw_linear,no_step_summary,full_mu_summary --max-parallel 12
```

## Checkpoints

- `TSP/models/LINC_official_morph/n100/checkpoint-325.pt`
- `CVRP/models/LINC_official_morph/n100/checkpoint-1430.pt`
- `CVRPTW/models/LINC_env_scratch/n100/checkpoint-11885.pt`

Baseline checkpoints are kept under each task's `models/POMO/` and
`models/PolyNet/` directories.

## License

The repository is released under the MIT license. Baseline components derived
from POMO, SGBS, and PolyNet retain their original notices; see
`THIRD_PARTY_NOTICES.md` and source-file headers.
