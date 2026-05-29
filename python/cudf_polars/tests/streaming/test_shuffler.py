# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, cast

import pytest
from rapidsmpf.statistics import Formatter
from rapidsmpf.streaming.cudf.channel_metadata import (
    ChannelMetadata,
    HashScheme,
    Partitioning,
)
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

import polars as pl

import cudf_polars.streaming.actor_graph.collectives.shuffle as shuffle_mod
from cudf_polars.containers import DataFrame, DataType
from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import allgather_polars_dataframe
from cudf_polars.streaming.actor_graph.collectives.common import reserve_op_id
from cudf_polars.streaming.actor_graph.collectives.shuffle import (
    LocalRepartitioner,
    ShuffleManager,
)
from cudf_polars.streaming.actor_graph.utils import (
    _is_already_partitioned,
)
from cudf_polars.testing.asserts import assert_gpu_result_equal

if TYPE_CHECKING:
    from cudf_polars.dsl.ir import IRExecutionContext


@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=1, broadcast_limit=48),
        StreamingOptions(max_rows_per_partition=5, broadcast_limit=48),
    ],
)
def test_join_rapidsmpf(streaming_engine_factory, options) -> None:
    streaming_engine = streaming_engine_factory(options)
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how="inner")
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=1),
        StreamingOptions(max_rows_per_partition=5),
    ],
)
def test_sort_rapidsmpf(streaming_engine_factory, options) -> None:
    streaming_engine = streaming_engine_factory(options)
    df = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    q = df.sort(by=["y", "z"])
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=True)


def test_is_already_partitioned():
    # Unit test for _is_already_partitioned helper
    chunks = 4
    columns = (0, 1)
    modulus = 8
    nranks = 1

    # Exact match: should return True
    metadata_match = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_match, columns, modulus, nranks) is True

    # Different columns: should return False
    metadata_diff_cols = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme((0,), modulus),
            local="inherit",
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_cols, columns, modulus, nranks) is False
    )

    # Different local partitioning: should return False
    metadata_diff_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=None,
        ),
    )
    assert (
        _is_already_partitioned(metadata_diff_local, columns, modulus, nranks) is False
    )

    # Different modulus: should return False
    metadata_diff_mod = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, 16),
            local="inherit",
        ),
    )
    assert _is_already_partitioned(metadata_diff_mod, columns, modulus, nranks) is False

    # No partitioning: should return False
    metadata_none = ChannelMetadata(chunks)
    assert _is_already_partitioned(metadata_none, columns, modulus, nranks) is False

    # Local not "inherit": should return False
    metadata_local = ChannelMetadata(
        chunks,
        partitioning=Partitioning(
            inter_rank=HashScheme(columns, modulus),
            local=HashScheme((0,), 4),
        ),
    )
    assert _is_already_partitioned(metadata_local, columns, modulus, nranks) is False


class _FakeStatistics:
    def __init__(self) -> None:
        self.stats: dict[str, list[float]] = {}
        self.report_entries: dict[str, tuple[list[str], Any]] = {}

    def add_stat(self, name: str, value: float) -> None:
        self.stats.setdefault(name, []).append(value)

    def add_report_entry(
        self, name: str, stat_names: list[str], formatter: Any
    ) -> None:
        self.report_entries.setdefault(name, (stat_names, formatter))


class _FakeContext:
    def __init__(self) -> None:
        self._statistics = _FakeStatistics()
        self._br = object()

    def statistics(self) -> _FakeStatistics:
        return self._statistics

    def br(self) -> object:
        return self._br


class _FakeComm:
    rank = 0
    nranks = 1


class _FakeIRContext:
    def __init__(self) -> None:
        self.stream = object()

    def get_cuda_stream(self) -> object:
        return self.stream


class _FakeChannel:
    def __init__(self, messages: list[Any] | None = None) -> None:
        self._messages = list(messages or [])
        self.sent: list[Any] = []
        self.drained = False

    async def recv(self, context: _FakeContext) -> Any:
        del context
        if self._messages:
            return self._messages.pop(0)
        return None

    async def send(self, context: _FakeContext, msg: Any) -> None:
        del context
        self.sent.append(msg)

    async def drain(self, context: _FakeContext) -> None:
        del context
        self.drained = True


def _capture_nvtx(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []

    def fake_annotate(
        *, message: str, **kwargs: Any
    ) -> contextlib.AbstractContextManager[None]:
        del kwargs

        @contextlib.contextmanager
        def scope() -> Any:
            messages.append(message)
            yield

        return scope()

    monkeypatch.setattr(
        shuffle_mod, "nvtx_annotate_cudf_polars", fake_annotate, raising=False
    )
    return messages


def test_global_shuffle_metrics_skip_active_shuffle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = ChannelMetadata(
        local_count=1,
        partitioning=Partitioning(inter_rank=HashScheme((0,), 1), local="inherit"),
    )
    sent_metadata: list[ChannelMetadata] = []

    async def fake_recv_metadata(
        ch: _FakeChannel, context: _FakeContext
    ) -> ChannelMetadata:
        del ch, context
        return metadata

    async def fake_send_metadata(
        ch: _FakeChannel, context: _FakeContext, metadata: ChannelMetadata
    ) -> None:
        del ch, context
        sent_metadata.append(metadata)

    monkeypatch.setattr(shuffle_mod, "recv_metadata", fake_recv_metadata)
    monkeypatch.setattr(shuffle_mod, "send_metadata", fake_send_metadata)
    monkeypatch.setattr(shuffle_mod, "_is_already_partitioned", lambda *args: True)
    nvtx_messages = _capture_nvtx(monkeypatch)

    context = _FakeContext()
    ch_in = _FakeChannel([object()])
    ch_out = _FakeChannel()

    asyncio.run(
        shuffle_mod._global_shuffle(
            context,
            _FakeComm(),
            cast("IRExecutionContext", _FakeIRContext()),
            ch_out,
            ch_in,
            columns_to_hash=(0,),
            num_partitions=1,
            collective_id=1,
        )
    )

    assert sent_metadata == [metadata]
    assert len(ch_out.sent) == 1
    assert ch_out.drained
    assert "cudf-polars-shuffle-total-time" in context.statistics().stats
    assert "cudf-polars-shuffle-rapidsmpf-time" not in context.statistics().stats
    assert context.statistics().report_entries["cudf-polars-shuffle-total-time"] == (
        ["cudf-polars-shuffle-total-time"],
        Formatter.Duration,
    )
    assert "cudf-polars-shuffle-total-time" in nvtx_messages
    assert "cudf-polars-shuffle-rapidsmpf-time" not in nvtx_messages


def test_global_shuffle_metrics_active_shuffle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = ChannelMetadata(local_count=1)
    sent_metadata: list[ChannelMetadata] = []
    inserted_columns: list[tuple[int, ...]] = []
    extracted: list[tuple[int, object]] = []

    async def fake_recv_metadata(
        ch: _FakeChannel, context: _FakeContext
    ) -> ChannelMetadata:
        del ch, context
        return metadata

    async def fake_send_metadata(
        ch: _FakeChannel, context: _FakeContext, metadata: ChannelMetadata
    ) -> None:
        del ch, context
        sent_metadata.append(metadata)

    class FakeTableChunk:
        def make_available_and_spill(
            self, br: object, *, allow_overbooking: bool
        ) -> FakeTableChunk:
            del br, allow_overbooking
            return self

        @classmethod
        def from_message(cls, msg: object, br: object) -> FakeTableChunk:
            del msg, br
            return cls()

        @classmethod
        def from_pylibcudf_table(
            cls,
            table: object,
            stream: object,
            *,
            exclusive_view: bool,
            br: object,
        ) -> tuple[object, object, bool, object]:
            del cls
            return (table, stream, exclusive_view, br)

    class FakeMessage:
        def __init__(self, partition_id: int, payload: object) -> None:
            self.partition_id = partition_id
            self.payload = payload

    class FakeInserter:
        async def __aenter__(self) -> FakeInserter:
            return self

        async def __aexit__(self, *args: object) -> None:
            del args

        def insert_hash(
            self, chunk: FakeTableChunk, columns_to_hash: tuple[int, ...]
        ) -> None:
            del chunk
            inserted_columns.append(columns_to_hash)

    class FakeShuffleManager:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def inserting(self) -> FakeInserter:
            return FakeInserter()

        def local_partitions(self) -> list[int]:
            return [0]

        def extract_chunk(self, partition_id: int, stream: object) -> object:
            extracted.append((partition_id, stream))
            return object()

    monkeypatch.setattr(shuffle_mod, "recv_metadata", fake_recv_metadata)
    monkeypatch.setattr(shuffle_mod, "send_metadata", fake_send_metadata)
    monkeypatch.setattr(shuffle_mod, "_is_already_partitioned", lambda *args: False)
    monkeypatch.setattr(shuffle_mod, "TableChunk", FakeTableChunk)
    monkeypatch.setattr(shuffle_mod, "Message", FakeMessage)
    monkeypatch.setattr(shuffle_mod, "ShuffleManager", FakeShuffleManager)
    nvtx_messages = _capture_nvtx(monkeypatch)

    context = _FakeContext()
    fake_ir_context = _FakeIRContext()
    ir_context = cast("IRExecutionContext", fake_ir_context)
    ch_in = _FakeChannel([object()])
    ch_out = _FakeChannel()

    asyncio.run(
        shuffle_mod._global_shuffle(
            context,
            _FakeComm(),
            ir_context,
            ch_out,
            ch_in,
            columns_to_hash=(0,),
            num_partitions=1,
            collective_id=1,
        )
    )

    assert len(sent_metadata) == 1
    assert inserted_columns == [(0,)]
    assert extracted == [(0, fake_ir_context.stream)]
    assert len(ch_out.sent) == 1
    assert ch_out.drained
    assert "cudf-polars-shuffle-total-time" in context.statistics().stats
    assert "cudf-polars-shuffle-rapidsmpf-time" in context.statistics().stats
    assert context.statistics().report_entries["cudf-polars-shuffle-total-time"] == (
        ["cudf-polars-shuffle-total-time"],
        Formatter.Duration,
    )
    assert context.statistics().report_entries[
        "cudf-polars-shuffle-rapidsmpf-time"
    ] == (["cudf-polars-shuffle-rapidsmpf-time"], Formatter.Duration)
    assert "cudf-polars-shuffle-total-time" in nvtx_messages
    assert "cudf-polars-shuffle-rapidsmpf-time" in nvtx_messages


@pytest.mark.spmd
@pytest.mark.parametrize("local_count", [1, 2, 4])
def test_local_repartitioner_hash(spmd_engine, local_count) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm

    pl_df = pl.DataFrame({"key": list(range(4)) * 3, "val": list(range(12))})
    col_names = pl_df.columns
    dtypes = [DataType(dt) for dt in pl_df.dtypes]

    results: list[tuple[int, pl.DataFrame]] = []

    async def _run() -> None:
        stream = context.get_stream_from_pool()
        cudf_df = DataFrame.from_polars(pl_df, stream)
        with reserve_op_id() as op_id:
            shuffle = ShuffleManager(
                context, comm, num_partitions=comm.nranks, collective_id=op_id
            )
            async with shuffle.inserting() as inserter:
                inserter.insert_hash(
                    TableChunk.from_pylibcudf_table(
                        cudf_df.table, stream, exclusive_view=True, br=context.br()
                    ),
                    columns_to_hash=(0,),
                )

            local = LocalRepartitioner(shuffle, local_count=local_count)
            await local.repartition_by_hash(columns_to_hash=(0,), stream=stream)

            for pid in local.local_partitions():
                tbl = local.extract_chunk(pid, stream)
                results.append(
                    (
                        pid,
                        DataFrame.from_table(
                            tbl, col_names, dtypes, stream
                        ).to_polars(),
                    )
                )

    asyncio.run(_run())

    assert len(results) == local_count

    # Same key always lands in the same local partition.
    key_to_pid: dict[int, int] = {}
    for pid, df in results:
        for key_val in df["key"].to_list():
            assert key_to_pid.setdefault(key_val, pid) == pid

    # AllGather across ranks: every rank inserts 12 rows, all must survive.
    local_df = pl.concat([df for _, df in results])
    with reserve_op_id() as op_id:
        global_df = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local_df, op_id=op_id
        )
    assert global_df.height == 12 * comm.nranks


@pytest.mark.spmd
@pytest.mark.parametrize("local_count", [1, 2, 4])
def test_local_repartitioner_index(spmd_engine, local_count) -> None:
    context = spmd_engine.context
    comm = spmd_engine.comm

    pl_payload = pl.DataFrame(
        {
            "local_part": [i % local_count for i in range(12)],
            "val": list(range(12)),
        }
    )
    pl_rank_part = pl.DataFrame({"rank_part": [i % comm.nranks for i in range(12)]})
    out_col_names = ["val"]
    out_dtypes = [DataType(pl.Int32())]

    results: list[tuple[int, pl.DataFrame]] = []

    async def _run() -> None:
        stream = context.get_stream_from_pool()
        payload_df = DataFrame.from_polars(pl_payload, stream)
        rank_part_df = DataFrame.from_polars(pl_rank_part, stream)

        with reserve_op_id() as op_id:
            shuffle = ShuffleManager(
                context, comm, num_partitions=comm.nranks, collective_id=op_id
            )
            async with shuffle.inserting() as inserter:
                inserter.insert_index(
                    TableChunk.from_pylibcudf_table(
                        payload_df.table, stream, exclusive_view=True, br=context.br()
                    ),
                    TableChunk.from_pylibcudf_table(
                        rank_part_df.table, stream, exclusive_view=True, br=context.br()
                    ),
                )

            local = LocalRepartitioner(shuffle, local_count=local_count)
            await local.repartition_by_index(partition_col=0, stream=stream)

            for pid in local.local_partitions():
                tbl = local.extract_chunk(pid, stream)
                results.append(
                    (
                        pid,
                        DataFrame.from_table(
                            tbl, out_col_names, out_dtypes, stream
                        ).to_polars(),
                    )
                )

    asyncio.run(_run())

    assert len(results) == local_count

    # Routing: val=v must land in local partition v % local_count (== local_part).
    for pid, df in results:
        assert df.columns == ["val"]
        for val in df["val"].to_list():
            assert val % local_count == pid

    # Global: every inserted row survives.
    local_df = pl.concat([df for _, df in results])
    with reserve_op_id() as op_id:
        global_df = allgather_polars_dataframe(
            engine=spmd_engine, local_df=local_df, op_id=op_id
        )
    assert global_df.height == 12 * comm.nranks
