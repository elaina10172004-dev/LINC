# TSP Data

The official evaluation pkl files used by the paper entrypoints live in this
directory.

Required for the default n100 evaluation entrypoints:

- `tsp100_test_seed1234.pkl`

Optional validation files, only needed when `--val-data` is set and
`--val-episodes > 0`:

- `tsp100_valid_seed1233.pkl`

Optional generalization sets:

- `tsp150_test_small_seed1235.pkl`
- `tsplib_50_200.pkl` for the variable-size TSPLIB subset used by
  `--dataset tsplib_50_200`.
