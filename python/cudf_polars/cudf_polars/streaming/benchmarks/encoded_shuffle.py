# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Experimental encoded-shuffle benchmark.

WARNING: This is an experimental (and unofficial) benchmark script. It is not
intended for public use and may be modified or removed at any time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from rapidsmpf import bootstrap as rapidsmpf_bootstrap

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

SOURCE_ROOT = Path(__file__).resolve().parents[3]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

ENCODED_SHUFFLE_ENV = "CUDF_POLARS__EXECUTOR__ENCODED_SHUFFLE"
LEGACY_ENCODED_SHUFFLE_ENV = "CUDF_POLARS__ENCODED_SHUFFLE"
ENCODED_SHUFFLE_VALUES = frozenset({"auto", "off"})
PARTITION_STAT_NAMES = (
    "cudf-partition-input-bytes",
    "cudf-partition-packed-bytes",
    "cudf-unpack-input-bytes",
    "cudf-unpack-output-bytes",
)


EncodedShuffleMode = Literal["auto", "off"]


def require_single_rank_launch() -> None:
    """Reject multi-rank ``rrun`` launches until this benchmark is rank-aware."""
    if not rapidsmpf_bootstrap.is_running_with_rrun():
        return
    nranks = rapidsmpf_bootstrap.get_nranks()
    if nranks != 1:
        raise RuntimeError(
            "encoded_shuffle.py is currently a single-rank benchmark. "
            f"Run with one rank or use a single Python process; got {nranks} ranks."
        )


def iteration_phase(iteration: int) -> Literal["first_iteration", "repeat"]:
    """Return a cache-neutral label for a timing iteration."""
    return "first_iteration" if iteration == 0 else "repeat"


def validate_results_equal(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """Raise if two benchmark result frames differ."""
    if not left.equals(right):
        raise ValueError(
            "encoded_shuffle=auto and encoded_shuffle=off results differ: "
            f"left_shape={left.shape}, right_shape={right.shape}"
        )


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


def _env_encoded_shuffle() -> tuple[EncodedShuffleMode | None, str | None]:
    for name in (ENCODED_SHUFFLE_ENV, LEGACY_ENCODED_SHUFFLE_ENV):
        value = os.environ.get(name)
        if value is None:
            continue
        if value not in ENCODED_SHUFFLE_VALUES:
            raise ValueError(
                f"{name} must be one of {sorted(ENCODED_SHUFFLE_VALUES)}, "
                f"got {value!r}"
            )
        return cast("EncodedShuffleMode", value), name
    return None, None


def _resolve_encoded_shuffle_modes(
    selected: str,
) -> tuple[list[EncodedShuffleMode], str | None]:
    if selected == "both":
        return ["auto", "off"], None
    if selected in ENCODED_SHUFFLE_VALUES:
        return [cast("EncodedShuffleMode", selected)], None

    env_mode, env_name = _env_encoded_shuffle()
    return [env_mode or "off"], env_name


def _make_arrow_batch(
    *,
    start: int,
    size: int,
    cardinality: int,
    dictionary: Any,
) -> Any:
    import pyarrow as pa  # noqa: PLC0415

    stop = start + size
    offsets = range(start, stop)
    indices = pa.array((i % cardinality for i in offsets), type=pa.int32())
    region = pa.DictionaryArray.from_arrays(indices, dictionary)
    return pa.table(
        {
            "k": pa.array(
                (i % (cardinality * 8) for i in range(start, stop)),
                type=pa.int32(),
            ),
            "region": region,
            "v": pa.array((i % 100 for i in range(start, stop)), type=pa.int64()),
        }
    )


def make_dataset(
    path: Path,
    *,
    rows: int,
    cardinality: int,
    batch_size: int,
) -> None:
    """Write a dictionary-encoded string-heavy Parquet dataset."""
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    dictionary = pa.array(
        [f"region_{i:06d}_encoded_shuffle_key" for i in range(cardinality)]
    )
    first_size = min(rows, batch_size)
    first = _make_arrow_batch(
        start=0,
        size=first_size,
        cardinality=cardinality,
        dictionary=dictionary,
    )

    with pq.ParquetWriter(path, first.schema, use_dictionary=["region"]) as writer:
        writer.write_table(first, row_group_size=first_size)
        for start in range(first_size, rows, batch_size):
            size = min(batch_size, rows - start)
            writer.write_table(
                _make_arrow_batch(
                    start=start,
                    size=size,
                    cardinality=cardinality,
                    dictionary=dictionary,
                ),
                row_group_size=size,
            )


def build_query(path: Path) -> pl.LazyFrame:
    """Build the encoded-shuffle benchmark query."""
    import polars as pl  # noqa: PLC0415

    return (
        pl.scan_parquet(path)
        .group_by("region")
        .agg(pl.col("v").sum().alias("total"))
        .sort("region")
    )


def _make_engine(
    *,
    encoded_shuffle: EncodedShuffleMode,
    max_rows_per_partition: int,
    target_partition_size: int,
    native_parquet: bool,
) -> Any:
    from cudf_polars.engine.options import StreamingOptions  # noqa: PLC0415
    from cudf_polars.engine.spmd import SPMDEngine  # noqa: PLC0415

    options = StreamingOptions(
        statistics=True,
        encoded_shuffle=encoded_shuffle,
        max_rows_per_partition=max_rows_per_partition,
        target_partition_size=target_partition_size,
        raise_on_fail=True,
        parquet_options={"use_rapidsmpf_native": native_parquet},
    )
    return SPMDEngine.from_options(options)


def _collect_partition_statistics(engine: Any) -> dict[str, dict[str, float | int]]:
    statistics = engine.global_statistics(clear=True)
    all_statistics = statistics.to_dict()
    return {
        name: all_statistics[name]
        for name in PARTITION_STAT_NAMES
        if name in all_statistics
    }


def run_once(path: Path, engine: Any) -> tuple[float, pl.DataFrame, dict[str, Any]]:
    """Collect the benchmark query once and return timing plus shuffle stats."""
    query = build_query(path)
    engine.global_statistics(clear=True)
    start = time.perf_counter()
    result = query.collect(engine=engine)
    seconds = time.perf_counter() - start
    statistics = _collect_partition_statistics(engine)
    return seconds, result, statistics


def _print_json(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for this benchmark."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cudf-polars streaming groupby shuffle on a "
            "dictionary/string-heavy Parquet dataset."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("/tmp/encoded-shuffle.parquet"),
        help="Parquet file to read, creating it first when it does not exist.",
    )
    parser.add_argument(
        "--rows",
        type=_positive_int,
        default=10_000_000,
        help="Rows to generate when --path does not exist.",
    )
    parser.add_argument(
        "--cardinality",
        type=_positive_int,
        default=256,
        help="Number of distinct string keys to generate.",
    )
    parser.add_argument(
        "--repeats",
        type=_positive_int,
        default=2,
        help="Timed query collections per encoded-shuffle mode.",
    )
    parser.add_argument(
        "--encoded-shuffle",
        choices=["auto", "off", "both", "env"],
        default="env",
        help=(
            "Encoded-shuffle policy to benchmark. 'both' runs auto then off; "
            "'env' reads CUDF_POLARS__EXECUTOR__ENCODED_SHUFFLE, then the "
            "legacy CUDF_POLARS__ENCODED_SHUFFLE, then defaults to off."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1_000_000,
        help="Rows per generated Parquet row group.",
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
        "--native-parquet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the RAPIDSMPF native Parquet reader in the streaming executor.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the encoded-shuffle benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        require_single_rank_launch()
    except RuntimeError as err:
        parser.error(str(err))

    try:
        encoded_shuffle_modes, env_source = _resolve_encoded_shuffle_modes(
            args.encoded_shuffle
        )
    except ValueError as err:
        parser.error(str(err))

    created = False
    if not args.path.exists():
        make_dataset(
            args.path,
            rows=args.rows,
            cardinality=args.cardinality,
            batch_size=args.batch_size,
        )
        created = True
    elif args.path.is_dir():
        parser.error(f"--path must be a Parquet file path, got directory {args.path}")

    _print_json(
        {
            "event": "dataset",
            "path": str(args.path),
            "created": created,
            "requested_rows": args.rows,
            "requested_cardinality": args.cardinality,
            "batch_size": args.batch_size,
            "encoded_shuffle_env_source": env_source,
        }
    )

    first_results: dict[EncodedShuffleMode, pl.DataFrame] = {}
    for mode_order, encoded_shuffle in enumerate(encoded_shuffle_modes):
        with _make_engine(
            encoded_shuffle=encoded_shuffle,
            max_rows_per_partition=args.max_rows_per_partition,
            target_partition_size=args.target_partition_size,
            native_parquet=args.native_parquet,
        ) as engine:
            for iteration in range(args.repeats):
                seconds, result, statistics = run_once(args.path, engine)
                if iteration == 0:
                    first_results[encoded_shuffle] = result
                _print_json(
                    {
                        "event": "timing",
                        "encoded_shuffle": encoded_shuffle,
                        "iteration": iteration,
                        "phase": iteration_phase(iteration),
                        "mode_order": mode_order,
                        "seconds": round(seconds, 6),
                        "path": str(args.path),
                        "requested_rows": args.rows,
                        "requested_cardinality": args.cardinality,
                        "result_rows": result.height,
                        "statistics": statistics,
                    }
                )
    if "auto" in first_results and "off" in first_results:
        validate_results_equal(first_results["auto"], first_results["off"])
        _print_json(
            {
                "event": "validation",
                "encoded_shuffle_modes": ["auto", "off"],
                "result_equal": True,
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
