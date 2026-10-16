# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.utils.dynamic_index import (
    GROUPED_NULLS,
    GROUPED_UNSORTED,
    OK,
    UNSORTED,
    check_summaries,
    compact_summary,
    index_error,
    summarize,
)
from cudf_polars.utils.cuda_stream import get_cuda_stream


def _summary(index: list, keys: list | None) -> plc.Table:
    stream = get_cuda_stream()

    def column(values: list, dtype: pl.DataType) -> Column:
        series = pl.Series(values, dtype=dtype)
        return Column(plc.Column.from_arrow(series.to_arrow()), dtype=DataType(dtype))

    idx = column(index, pl.Int64())
    by = [] if keys is None else [column(keys, pl.String())]
    error = index_error(idx, by, stream)
    return summarize([k.obj for k in by], idx.obj, idx.obj, error, stream)


@pytest.mark.parametrize(
    "ranks, expected",
    [
        ([[([1, 2], None), ([2, 4], None)]], OK),
        ([[([3, 4], None), ([1, 2], None)]], UNSORTED),
        ([[([2, 1], None), ([3, 4], None)]], UNSORTED),
        ([[([], None), ([1, 2], None), ([], None)]], OK),
        ([[([1, 5], ["a", "b"]), ([2, 6], ["a", "b"])]], OK),
        ([[([5, 1], ["a", "b"]), ([2, 6], ["a", "b"])]], GROUPED_UNSORTED),
        ([[([1, 5], ["a", None]), ([3, 2], [None, "a"])]], GROUPED_UNSORTED),
        ([[([1, None], ["a", "b"]), ([3, 4], ["a", "b"])]], GROUPED_NULLS),
        ([[([1, 2], None)], [([3], None), ([], None), ([4, 5], None)]], OK),
        # Rank 1's first value must survive compacting its later partitions.
        ([[([0], None)], [([-5], None), ([], None), ([4], None)]], UNSORTED),
        ([[([1, 2], ["a", "a"]), ([3], ["b"])], [([4], ["a"]), ([5], ["b"])]], OK),
        ([[([5], ["a"])], [([1], ["a"]), ([9], ["a"])]], GROUPED_UNSORTED),
        # A partition's error must survive compacting a later, valid one.
        ([[([2, 1], ["a", "a"]), ([5], ["a"])], [([6], ["a"])]], GROUPED_UNSORTED),
    ],
    ids=[
        "ties",
        "boundary",
        "partition",
        "empty",
        "grouped",
        "grouped-boundary",
        "null-key-boundary",
        "null-index",
        "ranks",
        "ranks-first",
        "ranks-grouped",
        "ranks-grouped-first",
        "ranks-error",
    ],
)
def test_check_summaries(ranks: list, expected: int) -> None:
    # Checking every partition's summary in order agrees with checking each
    # rank's compacted state, whether compacted one partition at a time or
    # all together.
    stream = get_cuda_stream()
    summaries = [[_summary(*partition) for partition in rank] for rank in ranks]
    everything = plc.concatenate.concatenate(
        [s for rank in summaries for s in rank], stream=stream
    )
    assert check_summaries(everything, stream) == expected
    for batched in (False, True):
        states = []
        for rank in summaries:
            first, *rest = [rank] if batched else [[s] for s in rank]
            state = compact_summary(None, first, stream)
            for batch in rest:
                state = compact_summary(state, batch, stream)
            states.append(state)
        gathered = plc.concatenate.concatenate(states, stream=stream)
        assert check_summaries(gathered, stream) == expected
