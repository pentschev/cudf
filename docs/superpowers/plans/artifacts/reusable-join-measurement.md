# Reusable Join Measurement

## Hardware

- GPU model:
- GPU count per node:
- Node count:
- Interconnect:
- CUDA version:
- RAPIDS branch:
- cudf-polars commit:

## Dataset

- Directory:
- Left rows:
- Right rows:
- Row group size:
- Left join key column: `k = row_id % (2 * right_rows)`
- Right join key column: `k = 0..right_rows-1`
- Expected semi result rows:
- Expected anti result rows:

## Commands

Semi example:

```bash
python python/cudf_polars/cudf_polars/streaming/benchmarks/reusable_join.py \
  --directory /tmp/reusable-join \
  --left-rows 100000000 \
  --right-rows 1000000 \
  --row-group-size 1000000 \
  --how semi \
  --iterations 5 \
  --max-rows-per-partition 1000000 \
  --target-partition-size 0 \
  --broadcast-limit 0
```

Anti example:

```bash
python python/cudf_polars/cudf_polars/streaming/benchmarks/reusable_join.py \
  --directory /tmp/reusable-join \
  --left-rows 100000000 \
  --right-rows 1000000 \
  --row-group-size 1000000 \
  --how anti \
  --iterations 5 \
  --max-rows-per-partition 1000000 \
  --target-partition-size 0 \
  --broadcast-limit 0
```

## Results Table/Template

| Join type | Iteration | Phase | Left rows | Right rows | Max rows per partition | Target partition size | Broadcast limit | Seconds | Result rows | Notes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| semi | 0 | first_iteration |  |  |  |  |  |  |  |  |
| semi | 1 | repeat |  |  |  |  |  |  |  |  |
| anti | 0 | first_iteration |  |  |  |  |  |  |  |  |
| anti | 1 | repeat |  |  |  |  |  |  |  |  |

The `repeat` phase is a repeated full query execution. It helps separate first
execution overheads, file/cache warmup, and steady-state timing, but it does not
represent reusable join state being carried across iterations. The reusable
`FilteredJoin` state is created once per broadcast join execution and reused
across left-side chunks within that execution.

## Inner/Left Reusable Hash Join Decision

- This benchmark covers the reusable `FilteredJoin` path for broadcast-right
  semi and anti joins only.
- Inner and left joins still need payload columns and row-pair materialization,
  so their reusable-state decision should be evaluated separately from this
  filtered probe result.
- Record whether repeated-probe savings here are large enough to justify a
  broader libcudf reusable hash-join API for joins that produce matched rows
  from both sides.

## Notes For Interpreting Whether To Pursue A Larger libcudf Reusable Hash-Join API

- Compare `first_iteration` to `repeat` for the same join type to account for
  first-execution overheads and file/cache warmup. Use the benchmark JSON
  `filtered_join_*` counters to confirm the reusable path was exercised.
- Keep `max_rows_per_partition` fixed while varying left rows to separate
  per-chunk probe cost from per-execution right-side build cost.
- Keep the right side comfortably below `broadcast_limit`; by default the
  benchmark raises if it does not observe the reusable FilteredJoin path.
- Validate small runs first, then disable `--validate` for timing runs so CPU
  collection does not affect wall-clock interpretation.
- Treat semi/anti gains as necessary but not sufficient evidence for inner/left
  reusable hash joins because output construction and duplicate-match handling
  can dominate those join types.
