# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation of the group_by_dynamic index across streaming partitions."""

from __future__ import annotations

import re

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import allgather_polars_dataframe
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.testing.asserts import assert_gpu_result_equal

ROWS_PER_PARTITION = 4


@pytest.fixture
def engine(streaming_engine_factory):
    # Small, known partitions, so that tests can place rows on either side of
    # a partition boundary.
    return streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=ROWS_PER_PARTITION,
            dynamic_planning=None,
            fallback_mode="raise",
        )
    )


def _query(t: list, keys: dict[str, list]) -> pl.LazyFrame:
    return (
        pl.LazyFrame({"t": t, **keys, "v": range(len(t))})
        .group_by_dynamic("t", every="2i", group_by=list(keys) or None)
        .agg(pl.col("v").sum())
    )


# 64 partitions, with each key in an early and a late partition, so the
# per-key state keeps growing and is compacted in batches.
_MANY = 64 * ROWS_PER_PARTITION
_MANY_KEYS = {"g": [i % (_MANY // 2) for i in range(_MANY)]}


@pytest.mark.parametrize(
    "t, keys",
    [
        (list(range(12)), {}),
        # Equal values on both sides of a boundary.
        ([1, 2, 3, 4, 4, 4, 5, 6], {}),
        # Each partition is sorted, but the index decreases across the boundary.
        ([5, 6, 7, 8, 1, 2, 3, 4], {}),
        # A later partition starts below the end of an earlier one.
        ([1, 2, 3, 9, 4, 5, 6, 7, 10, 11, 12, 13], {}),
        # Groups interleave across partitions, each sorted across them.
        ([5, 1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 6], {"g": ["a", "b"] * 6}),
        # Group "a" decreases across the boundary, but group "b" does not.
        ([5, 1, 6, 2, 1, 3, 2, 4], {"g": ["a", "b"] * 4}),
        # A group may decrease relative to another group's values.
        ([9, 1, 10, 2, 11, 3, 12, 4], {"g": [None, "b"] * 4}),
        # The null group decreases across the boundary, with two keys.
        ([5, 1, 6, 2, 1, 3, 2, 4], {"g": [None, "b"] * 4, "h": [1] * 8}),
        # Groups missing from some partitions.
        ([1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8], {"g": list("aaaabbbbaabb")}),
        (list(range(_MANY)), _MANY_KEYS),
        # The last row decreases against its key's earlier partition.
        ([*range(_MANY - 1), -1], _MANY_KEYS),
    ],
    ids=[
        "ungrouped",
        "ungrouped-ties",
        "ungrouped-unsorted",
        "ungrouped-unsorted-later",
        "grouped",
        "grouped-unsorted",
        "grouped-null",
        "grouped-null-unsorted-two-keys",
        "grouped-sparse",
        "many-keys",
        "many-keys-unsorted",
    ],
)
def test_across_partitions(engine: pl.GPUEngine, t: list, keys: dict) -> None:
    q = _query(t, keys)
    try:
        q.collect(engine="in-memory")
    except pl.exceptions.PolarsError as e:
        with pytest.RaisesGroup(pytest.RaisesExc(type(e), match=re.escape(str(e)))):
            q.collect(engine=engine)
    else:
        assert_gpu_result_equal(q, engine=engine)


def test_empty_partitions(engine: pl.GPUEngine) -> None:
    # Filtering leaves some partitions empty.
    q = (
        pl.LazyFrame({"t": list(range(16)), "v": range(16)})
        .filter((pl.col("t") < 3) | (pl.col("t") > 12))
        .group_by_dynamic("t", every="2i")
        .agg(pl.col("v").sum())
    )
    assert_gpu_result_equal(q, engine=engine)


def test_non_pointwise_keys_fall_back(engine: pl.GPUEngine) -> None:
    # Keys evaluated partition by partition would differ from the whole frame.
    q = (
        pl.LazyFrame({"t": list(range(8)), "g": [1, 1, 2, 2] * 2, "v": range(8)})
        .group_by_dynamic("t", every="2i", group_by=pl.col("g").cum_sum())
        .agg(pl.col("v").sum())
    )
    with pytest.raises(NotImplementedError, match="group_by_dynamic keys"):
        q.collect(engine=engine)


def _fragment(case: str, rank: int) -> dict[str, list]:
    """This rank's contiguous fragment: two partitions of four rows."""
    v = list(range(8 * rank, 8 * rank + 8))  # global row numbers
    if case == "ungrouped":
        return {"t": v, "v": v}
    if case == "ungrouped-rank-boundary":
        # Rank 0 ends above where rank 1 starts, though every rank is sorted.
        start = 1000 if rank == 0 else 8 * rank
        return {"t": list(range(start, start + 8)), "v": v}
    # Groups interleave within each rank.
    g = ["a", "b"] * 4
    if case == "grouped-rank-boundary" and rank == 0:
        # Group "a" decreases from rank 0 to rank 1, while group "b" increases.
        t = [5000 + i if k == "a" else i for i, k in enumerate(g)]
        return {"t": t, "g": g, "v": v}
    return {"t": [100 * rank + i // 2 for i in range(8)], "g": g, "v": v}


@pytest.mark.spmd
@pytest.mark.parametrize(
    "case",
    ["ungrouped", "ungrouped-rank-boundary", "grouped", "grouped-rank-boundary"],
)
def test_across_ranks(spmd_engine_factory, case: str) -> None:
    # Every rank holds a contiguous fragment of the input, in rank order.
    engine = spmd_engine_factory(
        StreamingOptions(
            max_rows_per_partition=ROWS_PER_PARTITION,
            dynamic_planning={},
            fallback_mode="raise",
        )
    )
    fragments = [_fragment(case, r) for r in range(engine.nranks)]
    whole = {k: [x for f in fragments for x in f[k]] for k in fragments[0]}
    by = ["g", "t"] if "g" in whole else ["t"]

    def query(frame: dict[str, list]) -> pl.LazyFrame:
        return (
            pl.LazyFrame(frame)
            .group_by_dynamic("t", every="2i", group_by=by[:-1] or None)
            .agg(pl.col("v").sum())
        )

    local = query(fragments[engine.rank])
    try:
        expected = query(whole).collect()
    except pl.exceptions.PolarsError as e:
        # Only an index that decreases across the rank boundary is invalid.
        assert engine.nranks > 1
        with pytest.RaisesGroup(pytest.RaisesExc(type(e)), flatten_subgroups=True):
            local.collect(engine=engine)
        return
    with reserve_op_id() as op_id:
        result = allgather_polars_dataframe(
            engine=engine, local_df=local.collect(engine=engine), op_id=op_id
        )
    assert_frame_equal(result.sort(by), expected.sort(by))
