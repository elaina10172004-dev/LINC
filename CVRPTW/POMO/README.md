# CVRPTW POMO

The CVRPTW POMO evaluation entrypoints use the RL4CO checkpoint in
`CVRPTW/models/POMO_env/n100/checkpoint-608871.ckpt`.

Sampling and greedy evaluation:

```bash
python CVRPTW/exp/sampling/eval_pomo.py --dataset synthetic100 --mode sampling --episodes 10000 --batch-size 256 --aug 8 --scale-mode horizon --seed 1234 --output-json results/cvrptw100_pomo_sampling.json
```

SGBS evaluation:

```bash
python CVRPTW/exp/sgbs/eval_pomo_sgbs.py --dataset synthetic100 --episodes 10000 --batch-size 3000 --beam-width 4 --expand-k 4 --aug 8 --seed 1234 --output-json results/cvrptw100_pomo_sgbs.json
```

For SGBS, the checkpoint is mapped into the shared CVRPTW model path used by
PolyNet and LINC.
