# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Experimental reusable filtered-join benchmark.

WARNING: This is an experimental (and unofficial) benchmark script. It is not
intended for public use and may be modified or removed at any time.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

BENCHMARK_NAME = "reusable_filtered_join"
JoinType = Literal["semi", "anti"]


@dataclass(frozen=True)
class DatasetInfo:
    """Resolved benchmark input files and their actual metadata."""

    left_path: Path
    right_path: Path
    left_rows: int
    right_rows: int
    left_row_groups: int
    right_row_groups: int
    created_left: bool
    created_right: bool


@dataclass
class FilteredJoinObservation:
    """Observed reusable FilteredJoin construction/probe counts."""

    builds: int = 0
    semi_calls: int = 0
    anti_calls: int = 0

    @property
    def probe_calls(self) -> int:
        return self.semi_calls + self.anti_calls

    @property
    def observed(self) -> bool:
        return self.builds > 0 and self.probe_calls > 0


def require_single_rank_launch() -> None:
    """Reject multi-rank ``rrun`` launches until this benchmark is rank-aware."""
    from rapidsmpf import bootstrap as rapidsmpf_bootstrap  # noqa: PLC0415

    if not rapidsmpf_bootstrap.is_running_with_rrun():
        return
    nranks = rapidsmpf_bootstrap.get_nranks()
    if nranks != 1:
        raise RuntimeError(
            "reusable_join.py is currently a single-rank benchmark. "
            f"Run with one rank or use a single Python process; got {nranks} ranks."
        )


def iteration_phase(iteration: int) -> Literal["first_iteration", "repeat"]:
    """Return a cache-neutral label for a timing iteration."""
    return "first_iteration" if iteration == 0 else "repeat"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be greater than or equal to zero"
        )
    return parsed


def _join_types(selected: str) -> list[JoinType]:
    if selected == "both":
        return ["semi", "anti"]
    return [cast("JoinType", selected)]


def _make_left_batch(*, start: int, size: int, key_modulus: int) -> Any:
    import pyarrow as pa  # noqa: PLC0415

    stop = start + size
    return pa.table(
        {
            "row_id": pa.array(range(start, stop), type=pa.int64()),
            "k": pa.array((i % key_modulus for i in range(start, stop)), type=pa.int64()),
            "payload": pa.array(
                ((i * 17) % 10_000 for i in range(start, stop)), type=pa.int64()
            ),
        }
    )


def _make_right_batch(*, start: int, size: int) -> Any:
    import pyarrow as pa  # noqa: PLC0415

    stop = start + size
    return pa.table({"k": pa.array(range(start, stop), type=pa.int64())})


def _write_table_batches(
    path: Path,
    *,
    rows: int,
    row_group_size: int,
    make_batch: Any,
) -> None:
    import pyarrow.parquet as pq  # noqa: PLC0415

    first_size = min(rows, row_group_size)
    first = make_batch(start=0, size=first_size)
    with pq.ParquetWriter(path, first.schema) as writer:
        writer.write_table(first, row_group_size=first_size)
        for start in range(first_size, rows, row_group_size):
            size = min(row_group_size, rows - start)
            writer.write_table(
                make_batch(start=start, size=size), row_group_size=size
            )


def _parquet_row_count_and_groups(path: Path) -> tuple[int, int]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return metadata.num_rows, metadata.num_row_groups


def make_dataset(
    directory: Path,
    *,
    left_rows: int,
    right_rows: int,
    row_group_size: int,
) -> DatasetInfo:
    """Write benchmark Parquet inputs that are absent from *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    left_path = directory / "left.parquet"
    right_path = directory / "right.parquet"
    created_left = False
    created_right = False

    if not left_path.exists():
        _write_table_batches(
            left_path,
            rows=left_rows,
            row_group_size=row_group_size,
            make_batch=lambda start, size: _make_left_batch(
                start=start,
                size=size,
                key_modulus=right_rows * 2,
            ),
        )
        created_left = True

    if not right_path.exists():
        _write_table_batches(
            right_path,
            rows=right_rows,
            row_group_size=row_group_size,
            make_batch=_make_right_batch,
        )
        created_right = True

    actual_left_rows, left_row_groups = _parquet_row_count_and_groups(left_path)
    actual_right_rows, right_row_groups = _parquet_row_count_and_groups(right_path)
    return DatasetInfo(
        left_path=left_path,
        right_path=right_path,
        left_rows=actual_left_rows,
        right_rows=actual_right_rows,
        left_row_groups=left_row_groups,
        right_row_groups=right_row_groups,
        created_left=created_left,
        created_right=created_right,
    )


def build_query(left_path: Path, right_path: Path, how: JoinType) -> pl.LazyFrame:
    """Build a broadcast-right semi/anti join query."""
    import polars as pl  # noqa: PLC0415

    left = pl.scan_parquet(left_path)
    right = pl.scan_parquet(right_path).select("k")
    return left.join(right, on="k", how=how)


def build_count_query(left_path: Path, right_path: Path, how: JoinType) -> pl.LazyFrame:
    """Build the benchmark query and reduce it to a row-count output."""
    import polars as pl  # noqa: PLC0415

    return build_query(left_path, right_path, how).select(
        pl.len().alias("row_count")
    )


def _make_engine(
    *,
    max_rows_per_partition: int,
    target_partition_size: int,
    broadcast_limit: int,
) -> Any:
    from cudf_polars.engine.options import StreamingOptions  # noqa: PLC0415
    from cudf_polars.engine.spmd import SPMDEngine  # noqa: PLC0415

    options = StreamingOptions(
        statistics=True,
        max_rows_per_partition=max_rows_per_partition,
        target_partition_size=target_partition_size,
        broadcast_limit=broadcast_limit,
        raise_on_fail=True,
    )
    return SPMDEngine.from_options(options)


@contextlib.contextmanager
def observe_filtered_join() -> Any:
    """Wrap the real FilteredJoin to prove the reusable path was exercised."""
    from cudf_polars.streaming.actor_graph import join as join_mod  # noqa: PLC0415

    observation = FilteredJoinObservation()
    real_filtered_join = join_mod.plc.join.FilteredJoin

    class ObservedFilteredJoin:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = real_filtered_join(*args, **kwargs)
            observation.builds += 1

        def semi_join(self, *args: Any, **kwargs: Any) -> Any:
            observation.semi_calls += 1
            return self._inner.semi_join(*args, **kwargs)

        def anti_join(self, *args: Any, **kwargs: Any) -> Any:
            observation.anti_calls += 1
            return self._inner.anti_join(*args, **kwargs)

    join_mod.plc.join.FilteredJoin = ObservedFilteredJoin
    try:
        yield observation
    finally:
        join_mod.plc.join.FilteredJoin = real_filtered_join


def _row_count(result: pl.DataFrame) -> int:
    return int(result["row_count"][0])


def validate_results_equal(gpu_count: int, cpu_count: int, how: JoinType) -> None:
    """Raise if GPU and CPU benchmark row counts differ."""
    if gpu_count != cpu_count:
        raise ValueError(
            f"GPU and CPU {how!r} join results differ: "
            f"gpu_count={gpu_count}, cpu_count={cpu_count}"
        )


def run_once(
    left_path: Path,
    right_path: Path,
    *,
    how: JoinType,
    engine: Any,
    validate: bool,
) -> tuple[float, int, FilteredJoinObservation]:
    """Collect one benchmark query and return timing plus result row count."""
    query = build_count_query(left_path, right_path, how)
    with observe_filtered_join() as observation:
        start = time.perf_counter()
        result = query.collect(engine=engine)
        seconds = time.perf_counter() - start
    result_row_count = _row_count(result)

    if validate:
        expected = _row_count(build_count_query(left_path, right_path, how).collect())
        validate_results_equal(result_row_count, expected, how)

    return seconds, result_row_count, observation


def _print_json(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for this benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cudf-polars streaming reusable FilteredJoin for repeated "
            "large-left/small-right semi and anti probes."
        ),
    )
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("/tmp/reusable-join"),
        help=(
            "Directory containing left.parquet and right.parquet, creating missing "
            "files first."
        ),
    )
    parser.add_argument(
        "--left-rows",
        type=_positive_int,
        default=100_000_000,
        help="Rows to generate for left.parquet when it does not exist.",
    )
    parser.add_argument(
        "--right-rows",
        type=_positive_int,
        default=1_000_000,
        help="Rows to generate for right.parquet when it does not exist.",
    )
    parser.add_argument(
        "--row-group-size",
        type=_positive_int,
        default=1_000_000,
        help="Rows per generated Parquet row group.",
    )
    parser.add_argument(
        "--how",
        choices=["semi", "anti", "both"],
        default="both",
        help="Join type to benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=3,
        help="Timed query collections per selected join type.",
    )
    parser.add_argument(
        "--max-rows-per-partition",
        type=_positive_int,
        default=1_000_000,
        help="cudf-polars streaming max_rows_per_partition executor option.",
    )
    parser.add_argument(
        "--target-partition-size",
        type=_nonnegative_int,
        default=0,
        help=(
            "cudf-polars streaming target_partition_size executor option; "
            "0 means auto."
        ),
    )
    parser.add_argument(
        "--broadcast-limit",
        type=_nonnegative_int,
        default=0,
        help=(
            "cudf-polars streaming broadcast_limit executor option; 0 means auto."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate each GPU result against CPU Polars for the same query.",
    )
    parser.add_argument(
        "--allow-non-reusable",
        action="store_true",
        help=(
            "Emit records instead of raising when the reusable FilteredJoin path "
            "is not observed."
        ),
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.directory.exists() and not args.directory.is_dir():
        parser.error(f"--directory must be a directory, got file {args.directory}")
    if args.left_rows <= args.right_rows:
        parser.error(
            "--left-rows must be greater than --right-rows for the "
            "large-left/small-right reusable join benchmark"
        )


def _validate_dataset(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    dataset: DatasetInfo,
) -> None:
    if dataset.left_rows <= dataset.right_rows:
        parser.error(
            "actual left.parquet row count must be greater than actual "
            "right.parquet row count for this benchmark; got "
            f"left={dataset.left_rows}, right={dataset.right_rows}"
        )
    if dataset.left_rows <= args.max_rows_per_partition:
        parser.error(
            "--max-rows-per-partition must be smaller than the actual left "
            "row count so the benchmark performs repeated large-side probes; "
            f"got max_rows_per_partition={args.max_rows_per_partition}, "
            f"left_rows={dataset.left_rows}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reusable filtered-join benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    try:
        require_single_rank_launch()
    except RuntimeError as err:
        parser.error(str(err))

    dataset = make_dataset(
        args.directory,
        left_rows=args.left_rows,
        right_rows=args.right_rows,
        row_group_size=args.row_group_size,
    )
    _validate_dataset(parser, args, dataset)

    for how_order, how in enumerate(_join_types(args.how)):
        with _make_engine(
            max_rows_per_partition=args.max_rows_per_partition,
            target_partition_size=args.target_partition_size,
            broadcast_limit=args.broadcast_limit,
        ) as engine:
            for iteration in range(args.iterations):
                seconds, result_row_count, observation = run_once(
                    dataset.left_path,
                    dataset.right_path,
                    how=how,
                    engine=engine,
                    validate=args.validate,
                )
                if not observation.observed and not args.allow_non_reusable:
                    raise RuntimeError(
                        "Reusable FilteredJoin path was not observed. Check "
                        "--broadcast-limit, input sizes, and partition settings, "
                        "or pass --allow-non-reusable to record the run anyway."
                    )
                observed_strategy = (
                    "broadcast_right_reusable_filtered_join"
                    if observation.observed
                    else "unknown_or_non_reusable"
                )
                _print_json(
                    {
                        "benchmark": BENCHMARK_NAME,
                        "observed_strategy": observed_strategy,
                        "join_type": how,
                        "iteration": iteration,
                        "phase": iteration_phase(iteration),
                        "how_order": how_order,
                        "seconds": round(seconds, 6),
                        "result_row_count": result_row_count,
                        "left_rows": dataset.left_rows,
                        "right_rows": dataset.right_rows,
                        "left_row_groups": dataset.left_row_groups,
                        "right_row_groups": dataset.right_row_groups,
                        "requested_left_rows": args.left_rows,
                        "requested_right_rows": args.right_rows,
                        "requested_row_group_size": args.row_group_size,
                        "max_rows_per_partition": args.max_rows_per_partition,
                        "target_partition_size": args.target_partition_size,
                        "broadcast_limit": args.broadcast_limit,
                        "directory": str(args.directory),
                        "left_path": str(dataset.left_path),
                        "right_path": str(dataset.right_path),
                        "created_left": dataset.created_left,
                        "created_right": dataset.created_right,
                        "filtered_join_builds": observation.builds,
                        "filtered_join_probe_calls": observation.probe_calls,
                        "filtered_join_semi_calls": observation.semi_calls,
                        "filtered_join_anti_calls": observation.anti_calls,
                        "validated": args.validate,
                    }
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
