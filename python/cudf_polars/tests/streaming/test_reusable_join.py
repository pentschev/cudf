# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl
import pytest

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.streaming.actor_graph import join as join_mod
from cudf_polars.testing.asserts import assert_gpu_result_equal


def _join_query(how: str, *, empty_right: bool = False) -> pl.LazyFrame:
    left = pl.LazyFrame(
        {
            "k": [0, 1, None, 2, 3, 4, None, 5, 6, 7, 8, 9],
            "payload": list(range(12)),
            "tag": ["a", "b", "c", "d"] * 3,
        }
    )
    right = (
        pl.LazyFrame(
            {
                "rk": pl.Series("rk", [], dtype=pl.Int64),
                "dim": pl.Series("dim", [], dtype=pl.String),
            }
        )
        if empty_right
        else pl.LazyFrame(
            {
                "rk": [1, 3, None, 8],
                "dim": ["one", "three", "null", "eight"],
            }
        )
    )
    return left.join(
        right,
        left_on=pl.col("k") + 1,
        right_on=pl.col("rk"),
        how=how,
        nulls_equal=True,
    )


def _streaming_options() -> StreamingOptions:
    return StreamingOptions(
        max_rows_per_partition=3,
        target_partition_size=64,
        broadcast_limit=1_000_000,
    )


def test_reusable_filtered_join_semi_correctness(streaming_engine_factory) -> None:
    streaming_engine = streaming_engine_factory(_streaming_options())
    assert_gpu_result_equal(
        _join_query("semi"),
        engine=streaming_engine,
        check_row_order=False,
    )


def test_reusable_filtered_join_anti_correctness(streaming_engine_factory) -> None:
    streaming_engine = streaming_engine_factory(_streaming_options())
    assert_gpu_result_equal(
        _join_query("anti"),
        engine=streaming_engine,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    "how, expected_call, other_call",
    [
        ("semi", "semi_calls", "anti_calls"),
        ("anti", "anti_calls", "semi_calls"),
    ],
)
def test_reusable_filtered_join_reuses_state_for_large_chunks(
    spmd_engine_factory,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
    expected_call: str,
    other_call: str,
) -> None:
    real_filtered_join = join_mod.plc.join.FilteredJoin
    records = []

    class SpyFilteredJoin:
        def __init__(self, *args, **kwargs) -> None:
            self._inner = real_filtered_join(*args, **kwargs)
            self.record = {"semi_calls": 0, "anti_calls": 0}
            records.append(self.record)

        def semi_join(self, *args, **kwargs):
            self.record["semi_calls"] += 1
            return self._inner.semi_join(*args, **kwargs)

        def anti_join(self, *args, **kwargs):
            self.record["anti_calls"] += 1
            return self._inner.anti_join(*args, **kwargs)

    monkeypatch.setattr(join_mod.plc.join, "FilteredJoin", SpyFilteredJoin)

    streaming_engine = spmd_engine_factory(_streaming_options())
    assert_gpu_result_equal(
        _join_query(how),
        engine=streaming_engine,
        check_row_order=False,
    )

    assert len(records) == 1
    assert records[0][expected_call] > 1
    assert records[0][other_call] == 0


@pytest.mark.parametrize("how", ["semi", "anti"])
def test_reusable_filtered_join_empty_right_falls_back(
    spmd_engine_factory,
    monkeypatch: pytest.MonkeyPatch,
    how: str,
) -> None:
    records = []

    class SpyFilteredJoin:
        def __init__(self, *args, **kwargs) -> None:
            records.append((args, kwargs))

    monkeypatch.setattr(join_mod.plc.join, "FilteredJoin", SpyFilteredJoin)

    streaming_engine = spmd_engine_factory(_streaming_options())
    assert_gpu_result_equal(
        _join_query(how, empty_right=True),
        engine=streaming_engine,
        check_row_order=False,
    )

    assert records == []
