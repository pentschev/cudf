# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Validation of the ``group_by_dynamic`` index.

Polars requires the index to have no nulls and to be sorted, within each group
when there are group keys. A stream is checked partition by partition, and
per-key summaries of the partitions, in input order, are then checked across
partition boundaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

import pylibcudf as plc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import Column

__all__ = [
    "check_summaries",
    "compact_summary",
    "index_error",
    "raise_index_error",
    "summarize",
]

# Error codes. The largest one of all partitions wins.
OK, UNSORTED, GROUPED_UNSORTED, GROUPED_NULLS = range(4)
_ERRORS = {
    UNSORTED: (
        pl.exceptions.InvalidOperationError,
        "argument in operation 'group_by_dynamic' is not sorted, "
        "please sort the 'expr/series/column' first",
    ),
    GROUPED_UNSORTED: (pl.exceptions.ComputeError, "input data is not sorted"),
    GROUPED_NULLS: (
        pl.exceptions.ComputeError,
        "null values in dynamic group_by not supported, fill nulls.",
    ),
}
_ASC = plc.types.Order.ASCENDING
_BEFORE, _AFTER = plc.types.NullOrder.BEFORE, plc.types.NullOrder.AFTER


def raise_index_error(code: int) -> None:
    """Raise the polars exception for a non-zero error code."""
    if code != OK:
        exception, message = _ERRORS[code]
        raise exception(message)


def _is_sorted(
    columns: Sequence[plc.Column],
    stream: Stream,
    null_order: plc.types.NullOrder = _BEFORE,
) -> bool:
    n = len(columns)
    return plc.sorting.is_sorted(
        plc.Table(list(columns)), [_ASC] * n, [null_order] * n, stream=stream
    )


def index_error(index: Column, keys: Sequence[Column], stream: Stream) -> int:
    """Return the error code of an index and its group keys (``OK`` if valid)."""
    if not keys:
        sorted_ = index.null_count == 0 and index.check_sorted(
            order=_ASC, null_order=_BEFORE, stream=stream
        )
        return OK if sorted_ else UNSORTED
    if index.null_count > 0:
        return GROUPED_NULLS
    # Common layouts are recognized in linear time: the index sorted overall,
    # or rows sorted by the keys, then the index. Otherwise sort by the keys.
    if index.check_sorted(order=_ASC, null_order=_BEFORE, stream=stream):
        return OK
    columns = [*(k.obj for k in keys), index.obj]
    if any(_is_sorted(columns, stream, nulls) for nulls in (_BEFORE, _AFTER)):
        return OK
    n = len(keys)
    grouped = plc.sorting.stable_sort_by_key(
        plc.Table(columns),
        plc.Table(columns[:n]),
        [_ASC] * n,
        [_BEFORE] * n,
        stream=stream,
    )
    return OK if _is_sorted(grouped.columns(), stream) else GROUPED_UNSORTED


def summarize(
    keys: Sequence[plc.Column],
    first: plc.Column,
    last: plc.Column,
    error: int,
    stream: Stream,
) -> plc.Table:
    """
    Summarize index bounds as ``[*keys, first, last, error]``, one row per key.

    Each key gets the smallest of its ``first`` and the largest of its
    ``last`` values. For a partition, pass its index as both. ``error`` is
    the error code of the rows summarized. The table owns its memory and is
    complete on return, so it outlives its inputs.
    """
    _min, _max = plc.aggregation.min(), plc.aggregation.max()
    if keys:
        grouped, results = plc.groupby.GroupBy(
            plc.Table(list(keys)), null_handling=plc.types.NullPolicy.INCLUDE
        ).aggregate(
            [
                plc.groupby.GroupByRequest(first, [_min]),
                plc.groupby.GroupByRequest(last, [_max]),
            ],
            stream=stream,
        )
        keys = grouped.columns()
        bounds = [result.columns()[0] for result in results]
    else:
        bounds = [
            plc.Column.from_scalar(
                plc.reduce.reduce(column, agg, column.type(), stream=stream),
                min(column.size(), 1),
                stream=stream,
            )
            for column, agg in ((first, _min), (last, _max))
        ]
    error_column = plc.Column.from_scalar(
        plc.Scalar.from_py(error, plc.DataType(plc.TypeId.INT8), stream=stream),
        bounds[0].size(),
        stream=stream,
    )
    summary = plc.Table([*keys, *bounds, error_column])
    # The inputs and temporary scalars are read asynchronously and may be
    # freed once this returns.
    stream.synchronize()
    return summary


def check_summaries(summaries: plc.Table, stream: Stream) -> int:
    """
    Return the error code of :func:`summarize` outputs concatenated in order.

    That is the worst error of any summary, or an error if some key's index
    decreases from one summary to the next.
    """
    if summaries.num_rows() == 0:
        return OK
    *keys, first, last, error = summaries.columns()
    worst = plc.reduce.reduce(error, plc.aggregation.max(), error.type(), stream=stream)
    if (code := int(worst.to_py(stream=stream) or OK)) != OK:
        return code
    if keys:
        # A stable sort keeps each key's summaries in input order.
        *keys, first, last = plc.sorting.stable_sort_by_key(
            plc.Table([*keys, first, last]),
            plc.Table(keys),
            [_ASC] * len(keys),
            [_BEFORE] * len(keys),
            stream=stream,
        ).columns()
    # Each key's first and last values, in order, must be ascending.
    interleaved = [
        plc.reshape.interleave_columns(plc.Table(list(pair)), stream=stream)
        for pair in (*((k, k) for k in keys), (first, last))
    ]
    if _is_sorted(interleaved, stream):
        return OK
    return GROUPED_UNSORTED if keys else UNSORTED


def compact_summary(
    running: plc.Table | None, pending: Sequence[plc.Table], stream: Stream
) -> plc.Table:
    """
    Fold summaries, in input order, into a single :func:`summarize` output.

    The result has one row per key, so it is bounded by the number of keys,
    and every row carries the worst error of all its inputs, including
    decreases between them. This reads all of ``running``, so callers should
    batch ``pending`` summaries rather than compact after each one.
    """
    combined = plc.concatenate.concatenate(
        [*([] if running is None else [running]), *pending], stream=stream
    )
    *keys, first, last, _ = combined.columns()
    return summarize(keys, first, last, check_summaries(combined, stream), stream)
