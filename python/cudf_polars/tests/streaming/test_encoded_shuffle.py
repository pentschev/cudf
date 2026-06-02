# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for encoded shuffle policy plumbing."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from cudf_polars.streaming.actor_graph import groupby, join, over
from cudf_polars.streaming.actor_graph.collectives import shuffle
from cudf_polars.streaming.actor_graph.collectives.shuffle import ShuffleManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _FakeContext:
    def br(self) -> object:
        return object()

    def create_channel(self) -> object:
        return object()


class _FakeComm:
    nranks = 1


class _FakeShuffler:
    def __init__(self) -> None:
        self.inserted: list[Any] = []

    def insert(self, packed: Any) -> None:
        self.inserted.append(packed)


class _FakeManager:
    def __init__(self) -> None:
        self.context = _FakeContext()
        self.num_partitions = 2
        self.shuffler = _FakeShuffler()


class _FakeChunk:
    stream = object()

    def table_view(self) -> object:
        return object()


def test_insert_hash_forwards_preserve_encoded_false(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_partition_and_pack(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(shuffle, "py_partition_and_pack", fake_partition_and_pack)
    monkeypatch.setattr(
        shuffle, "_partition_and_pack_supports_preserve_encoded", lambda: True
    )

    manager = _FakeManager()
    inserter = ShuffleManager.Inserter(manager)  # type: ignore[arg-type]
    inserter.insert_hash(
        _FakeChunk(),  # type: ignore[arg-type]
        columns_to_hash=(0,),
        preserve_encoded=False,
    )

    assert calls[0]["preserve_encoded"] is False
    assert len(manager.shuffler.inserted) == 1


def test_insert_hash_uses_legacy_signature_when_preserving_encoded(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_partition_and_pack(
        table: object,
        columns_to_hash: tuple[int, ...],
        num_partitions: int,
        stream: object,
        br: object,
    ) -> object:
        calls.append(
            {
                "table": table,
                "columns_to_hash": columns_to_hash,
                "num_partitions": num_partitions,
                "stream": stream,
                "br": br,
            }
        )
        return object()

    monkeypatch.setattr(shuffle, "py_partition_and_pack", fake_partition_and_pack)
    monkeypatch.setattr(
        shuffle, "_partition_and_pack_supports_preserve_encoded", lambda: False
    )

    manager = _FakeManager()
    inserter = ShuffleManager.Inserter(manager)  # type: ignore[arg-type]
    inserter.insert_hash(_FakeChunk(), columns_to_hash=(0,))

    assert calls[0]["columns_to_hash"] == (0,)
    assert len(manager.shuffler.inserted) == 1


def test_insert_hash_requires_new_rapidsmpf_for_materialization(
    monkeypatch,
) -> None:
    calls: list[object] = []

    def fake_partition_and_pack(*_: Any, **__: Any) -> object:
        calls.append(object())
        return object()

    monkeypatch.setattr(shuffle, "py_partition_and_pack", fake_partition_and_pack)
    monkeypatch.setattr(
        shuffle, "_partition_and_pack_supports_preserve_encoded", lambda: False
    )

    manager = _FakeManager()
    inserter = ShuffleManager.Inserter(manager)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="preserve_encoded"):
        inserter.insert_hash(
            _FakeChunk(),  # type: ignore[arg-type]
            columns_to_hash=(0,),
            preserve_encoded=False,
        )

    assert calls == []


def test_shuffle_join_forwards_preserve_encoded_to_both_global_shuffles(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    @asynccontextmanager
    async def fake_shutdown_on_error(
        *_: Any, **__: Any
    ) -> AsyncIterator[None]:
        yield None

    async def fake_send_metadata(*_: Any, **__: Any) -> None:
        return None

    async def fake_global_shuffle(
        *_: Any, preserve_encoded: bool = True, **__: Any
    ) -> None:
        calls.append(preserve_encoded)

    async def fake_join_chunks(*_: Any, **__: Any) -> None:
        return None

    monkeypatch.setattr(join, "shutdown_on_error", fake_shutdown_on_error)
    monkeypatch.setattr(join, "send_metadata", fake_send_metadata)
    monkeypatch.setattr(join, "use_bloom_filter", lambda *_: False)
    monkeypatch.setattr(join, "_global_shuffle", fake_global_shuffle)
    monkeypatch.setattr(join, "_join_chunks", fake_join_chunks)

    asyncio.run(
        join._shuffle_join(
            _FakeContext(),  # type: ignore[arg-type]
            _FakeComm(),  # type: ignore[arg-type]
            SimpleNamespace(options=("Inner",)),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            join.JoinStrategy(
                shuffle_modulus=2,
                output_indices=(0,),
                left_indices=(0,),
                right_indices=(0,),
            ),
            [10, 11, 12],
            row_counts=(1, 1),
            tracer=None,
            bloom_threshold=0.0,
            preserve_encoded=False,
        )
    )

    assert calls == [False, False]


def test_join_actor_uses_encoded_shuffle_policy_for_dynamic_shuffle(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    @asynccontextmanager
    async def fake_shutdown_on_error(
        *_: Any, **__: Any
    ) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(decision=None)

    async def fake_gather(*coroutines: Any) -> list[Any]:
        return [await coroutine for coroutine in coroutines]

    async def fake_recv_metadata(*_: Any, **__: Any) -> object:
        return object()

    async def fake_choose_strategy(*_: Any, **__: Any) -> tuple[Any, Any, Any]:
        sample = SimpleNamespace(chunks=[], total_rows=1)
        strategy = join.JoinStrategy(
            shuffle_modulus=2,
            output_indices=(0,),
            left_indices=(0,),
            right_indices=(0,),
        )
        return sample, sample, strategy

    async def fake_replay_buffered_channel(*_: Any, **__: Any) -> None:
        return None

    async def fake_shuffle_join(
        *_: Any, preserve_encoded: bool, **__: Any
    ) -> None:
        calls.append(preserve_encoded)

    monkeypatch.setattr(join, "shutdown_on_error", fake_shutdown_on_error)
    monkeypatch.setattr(join, "gather_in_task_group", fake_gather)
    monkeypatch.setattr(join, "recv_metadata", fake_recv_metadata)
    monkeypatch.setattr(join, "_choose_strategy", fake_choose_strategy)
    monkeypatch.setattr(join, "replay_buffered_channel", fake_replay_buffered_channel)
    monkeypatch.setattr(join, "_shuffle_join", fake_shuffle_join)

    asyncio.run(
        join.join_actor.__wrapped__(
            _FakeContext(),  # type: ignore[arg-type]
            _FakeComm(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            SimpleNamespace(
                encoded_shuffle="auto",
                dynamic_planning=SimpleNamespace(bloom_filter_threshold=0.0),
            ),  # type: ignore[arg-type]
            [10, 11, 12, 13],
        )
    )

    assert calls == [False]


def test_groupby_subnetwork_uses_encoded_shuffle_policy(monkeypatch) -> None:
    calls: list[bool] = []
    child = object()

    class FakeIR:
        children = (child,)

    class FakeChannelManager:
        def reserve_input_slot(self) -> object:
            return object()

        def reserve_output_slot(self) -> object:
            return object()

    def fake_groupby_actor(*_: Any, preserve_encoded: bool) -> object:
        calls.append(preserve_encoded)
        return object()

    ir = FakeIR()
    executor = SimpleNamespace(
        name="streaming",
        dynamic_planning=object(),
        encoded_shuffle="auto",
        target_partition_size=1024,
    )
    rec = SimpleNamespace(
        state={
            "config_options": SimpleNamespace(executor=executor),
            "context": object(),
            "comm": object(),
            "ir_context": object(),
            "collective_id_map": {ir: [10, 11]},
        }
    )

    monkeypatch.setattr(
        groupby,
        "process_children",
        lambda *_: ({}, {child: FakeChannelManager()}),
    )
    monkeypatch.setattr(groupby, "ChannelManager", lambda _: FakeChannelManager())
    monkeypatch.setattr(groupby, "groupby_actor", fake_groupby_actor)

    groupby._(ir, rec)  # type: ignore[arg-type]

    assert calls == [False]


def test_over_subnetwork_uses_encoded_shuffle_policy(monkeypatch) -> None:
    calls: list[bool] = []
    child = object()

    class FakeIR:
        children = (child,)
        is_scalar = False

    class FakeChannelManager:
        def reserve_input_slot(self) -> object:
            return object()

        def reserve_output_slot(self) -> object:
            return object()

    def fake_over_actor(*_: Any, preserve_encoded: bool) -> object:
        calls.append(preserve_encoded)
        return object()

    ir = FakeIR()
    executor = SimpleNamespace(
        dynamic_planning=SimpleNamespace(sample_chunk_count=3),
        encoded_shuffle="auto",
        target_partition_size=1024,
    )
    rec = SimpleNamespace(
        state={
            "config_options": SimpleNamespace(executor=executor),
            "context": object(),
            "comm": object(),
            "ir_context": object(),
            "collective_id_map": {ir: [10, 11]},
        }
    )

    monkeypatch.setattr(
        over,
        "process_children",
        lambda *_: ({}, {child: FakeChannelManager()}),
    )
    monkeypatch.setattr(over, "ChannelManager", lambda _: FakeChannelManager())
    monkeypatch.setattr(over, "over_actor", fake_over_actor)

    over._(ir, rec)  # type: ignore[arg-type]

    assert calls == [False]


def test_shuffle_reduce_forwards_preserve_encoded_to_groupby_hash_insert(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    @asynccontextmanager
    async def fake_inserting() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            insert_hash=lambda *_args, preserve_encoded=True: calls.append(
                preserve_encoded
            )
        )

    class FakeShuffleManager:
        def __init__(self, *_: Any, **__: Any) -> None:
            return None

        def inserting(self) -> Any:
            return fake_inserting()

        def local_partitions(self) -> list[int]:
            return []

    async def fake_send_metadata(*_: Any, **__: Any) -> None:
        return None

    class FakeOutputChannel:
        async def drain(self, *_: Any) -> None:
            return None

    monkeypatch.setattr(groupby, "ShuffleManager", FakeShuffleManager)
    monkeypatch.setattr(groupby, "send_metadata", fake_send_metadata)
    monkeypatch.setattr(groupby, "_make_hash_shuffle_metadata", lambda *_: object())
    monkeypatch.setattr(groupby, "_enforce_schema", lambda chunk, *_: chunk)

    asyncio.run(
        groupby._shuffle_reduce(
            _FakeContext(),  # type: ignore[arg-type]
            _FakeComm(),  # type: ignore[arg-type]
            SimpleNamespace(
                output_indices=(0,),
                shuffle_indices=(0,),
                reduction_ir=SimpleNamespace(schema={}),
                select_ir=None,
            ),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            FakeOutputChannel(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            2,
            10,
            1024,
            local=False,
            aggregated=object(),  # type: ignore[arg-type]
            input_drained=True,
            preserve_encoded=False,
        )
    )

    assert calls == [False]


def test_distribute_by_group_forwards_preserve_encoded_to_over_hash_insert(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    @asynccontextmanager
    async def fake_inserting() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            insert_hash=lambda *_args, preserve_encoded=True: calls.append(
                preserve_encoded
            )
        )

    class FakeChannel:
        def __init__(self) -> None:
            self._messages = [SimpleNamespace(sequence_number=7), None]

        async def shutdown_metadata(self, *_: Any) -> None:
            return None

        async def recv(self, *_: Any) -> Any:
            return self._messages.pop(0)

    class FakeChunk:
        def make_available_and_spill(self, *_: Any, **__: Any) -> object:
            return object()

    class FakeIRContext:
        def get_cuda_stream(self) -> object:
            return object()

        async def to_thread(self, *_: Any, **__: Any) -> object:
            return object()

    class FakeTableChunk:
        @staticmethod
        def from_message(*_: Any, **__: Any) -> FakeChunk:
            return FakeChunk()

    monkeypatch.setattr(over, "TableChunk", FakeTableChunk)

    sequence_numbers = asyncio.run(
        over._distribute_by_group(
            _FakeContext(),  # type: ignore[arg-type]
            SimpleNamespace(rank=0),  # type: ignore[arg-type]
            SimpleNamespace(inserting=fake_inserting),
            FakeChannel(),  # type: ignore[arg-type]
            (0,),
            FakeIRContext(),  # type: ignore[arg-type]
            skip_insert=False,
            preserve_encoded=False,
        )
    )

    assert sequence_numbers == [7]
    assert calls == [False]
