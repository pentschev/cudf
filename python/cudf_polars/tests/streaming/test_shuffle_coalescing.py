# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from cudf_polars.streaming.actor_graph.collectives.shuffle import (
    _coalesce_packed_pieces,
    _iter_packed_piece_batches,
)


@dataclass(frozen=True)
class Piece:
    size: int

    def data_size(self) -> int:
        return self.size


@dataclass(frozen=True)
class LegacyPiece:
    size: int


@dataclass(frozen=True)
class PieceWithMetadata:
    size: int
    metadata_size: int

    def data_size(self) -> int:
        return self.size


def _coalesced_sizes(sizes: list[int], target_size: int) -> list[list[int]]:
    pieces = [Piece(size) for size in sizes]
    return [
        [piece.data_size() for piece in batch]
        for batch in _coalesce_packed_pieces(pieces, target_size)
    ]


def test_coalesce_packed_pieces_combines_until_target() -> None:
    assert _coalesced_sizes([3, 4, 8, 1], target_size=10) == [[3, 4], [8, 1]]


def test_coalesce_packed_pieces_yields_initial_oversized_piece_alone() -> None:
    assert _coalesced_sizes([12, 2], target_size=10) == [[12], [2]]


def test_coalesce_packed_pieces_batches_by_payload_size_only() -> None:
    pieces = [PieceWithMetadata(4, 100), PieceWithMetadata(4, 100)]
    batches = list(_coalesce_packed_pieces(pieces, target_size=8))
    assert [[piece.size for piece in batch] for batch in batches] == [[4, 4]]


def test_iter_packed_piece_batches_preserves_legacy_one_piece_batches() -> None:
    pieces = [LegacyPiece(3), LegacyPiece(4)]
    batches = list(_iter_packed_piece_batches(pieces, target_size=10))
    assert [[piece.size for piece in batch] for batch in batches] == [[3], [4]]
