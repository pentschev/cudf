# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming actor for the ``group_by_dynamic`` index check."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cudf_streaming.table_chunk import (
    TableChunk,
    make_table_chunks_available_or_wait,
)
from rapidsmpf.streaming.core.actor import define_actor

from cudf_polars.dsl.ir import CheckDynamicIndex
from cudf_polars.dsl.utils.dynamic_index import (
    check_summaries,
    compact_summary,
    index_error,
    raise_index_error,
    summarize,
)
from cudf_polars.streaming.actor_graph.collectives.allgather import AllGatherManager
from cudf_polars.streaming.actor_graph.dispatch import (
    generate_ir_sub_network,
    ir_context_for_node,
)
from cudf_polars.streaming.actor_graph.tracing import send_chunk
from cudf_polars.streaming.actor_graph.utils import (
    ChannelManager,
    chunk_to_frame,
    process_children,
    recv_metadata,
    send_metadata,
    shutdown_on_error,
)

if TYPE_CHECKING:
    import pylibcudf as plc
    from rapidsmpf.communicator.communicator import Communicator
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context

    from cudf_polars.dsl.ir import IR, IRExecutionContext
    from cudf_polars.streaming.actor_graph.dispatch import SubNetGenerator


@define_actor()
async def check_dynamic_index_actor(
    context: Context,
    comm: Communicator,
    ir: CheckDynamicIndex,
    ir_context: IRExecutionContext,
    ch_out: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    *,
    collective_id: int,
) -> None:
    """
    Check the index of each chunk, and across chunks, forwarding them unchanged.

    Chunk summaries are folded into a per-key state, in batches at least as
    large as the state so the total work stays linear. After the last chunk,
    the states of all ranks are gathered (unless every rank holds the same
    data) and checked, so every rank raises the same error, if any. Like
    other order-sensitive streaming operations, this takes chunks to arrive in
    order and ranks to own contiguous ranges of the input.
    """
    async with shutdown_on_error(
        context, chs_in=(ch_in,), chs_out=(ch_out,), trace_ir=ir, ir_context=ir_context
    ) as actor_scope:
        tracer = actor_scope.tracer
        ir_context = actor_scope.require_ir_context()
        metadata = await recv_metadata(ch_in, context)
        await send_metadata(ch_out, context, metadata)
        state: plc.Table | None = None
        pending: list[plc.Table] = []
        pending_rows = 0
        while (msg := await ch_in.recv(context)) is not None:
            seq = msg.sequence_number
            chunk, _ = await make_table_chunks_available_or_wait(
                context,
                TableChunk.from_message(msg, br=context.br()),
                reserve_extra=0,
                net_memory_delta=0,
            )
            del msg
            df = chunk_to_frame(chunk, ir.children[0])
            index, keys = ir.index.evaluate(df), [k.evaluate(df) for k in ir.keys]
            error = index_error(index, keys, df.stream)
            summary = summarize(
                [k.obj for k in keys], index.obj, index.obj, error, df.stream
            )
            pending.append(summary)
            pending_rows += summary.num_rows()
            if state is None or pending_rows >= state.num_rows():
                state = compact_summary(state, pending, df.stream)
                pending, pending_rows = [], 0
            del df, index, keys, summary
            await send_chunk(context, ch_out, chunk, seq, tracer=tracer)

        stream = ir_context.get_cuda_stream()
        if pending:
            state = compact_summary(state, pending, stream)
        del pending
        if comm.nranks > 1 and not metadata.duplicated:
            allgather = AllGatherManager(context, comm, collective_id)
            with allgather.inserting() as inserter:
                if state is not None:
                    await inserter.insert(
                        0,
                        TableChunk.from_pylibcudf_table(
                            state, stream, exclusive_view=True, br=context.br()
                        ),
                    )
            del state
            state = await allgather.extract_concatenated(
                stream, ordered=True, ir_context=ir_context
            )
        if state is not None and state.num_columns() > 0:
            raise_index_error(check_summaries(state, stream))
        await ch_out.drain(context)


@generate_ir_sub_network.register(CheckDynamicIndex)
def _(
    ir: CheckDynamicIndex, rec: SubNetGenerator
) -> tuple[dict[IR, list[Any]], dict[IR, ChannelManager]]:
    actors, channels = process_children(ir, rec)
    channels[ir] = ChannelManager(rec.state["context"])
    (collective_id,) = rec.state["collective_id_map"][ir]
    actors[ir] = [
        check_dynamic_index_actor(
            rec.state["context"],
            rec.state["comm"],
            ir,
            ir_context_for_node(rec, ir),
            channels[ir].reserve_input_slot(),
            channels[ir.children[0]].reserve_output_slot(),
            collective_id=collective_id,
        )
    ]
    return actors, channels
