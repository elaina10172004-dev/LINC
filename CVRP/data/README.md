# CVRP Data

The random CVRP pkl files used by the paper entrypoints live in this directory.
CVRPXML should be unpacked under the path listed below.

Required for the default n100 evaluation entrypoints:

- `vrp100_test_seed1234.pkl`

Optional validation files, only needed when `--val-data` is set and
`--val-episodes > 0`:

- `vrp100_valid_seed1233.pkl`

Optional generalization sets:

- `vrp150_test_small_seed1235.pkl`

For CVRPLIB/XML100, unpack the `.vrp` files at:

- `xml100_full_download/instances/instances/*.vrp`

BKS/reference costs are also external. Pass them explicitly when needed:

```bash
--bks-json path/to/xml100_full_bks.json
```
