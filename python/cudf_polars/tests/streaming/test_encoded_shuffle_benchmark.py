# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the encoded-shuffle benchmark helpers."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.streaming.benchmarks import encoded_shuffle


def test_require_single_rank_launch_rejects_multirank_rrun(monkeypatch) -> None:
    monkeypatch.setattr(
        encoded_shuffle.rapidsmpf_bootstrap, "is_running_with_rrun", lambda: True
    )
    monkeypatch.setattr(encoded_shuffle.rapidsmpf_bootstrap, "get_nranks", lambda: 2)

    with pytest.raises(RuntimeError, match="single-rank"):
        encoded_shuffle.require_single_rank_launch()


def test_iteration_phase_label_does_not_claim_cold_cache() -> None:
    assert encoded_shuffle.iteration_phase(0) == "first_iteration"
    assert encoded_shuffle.iteration_phase(1) == "repeat"


def test_validate_results_equal_rejects_mismatched_totals() -> None:
    left = pl.DataFrame({"region": ["a"], "total": [1]})
    right = pl.DataFrame({"region": ["a"], "total": [2]})

    with pytest.raises(ValueError, match="differ"):
        encoded_shuffle.validate_results_equal(left, right)


def test_main_validates_both_mode_results(monkeypatch, tmp_path) -> None:
    class FakeEngine:
        def __init__(self, mode: str) -> None:
            self.mode = mode

        def __enter__(self) -> FakeEngine:
            return self

        def __exit__(self, *_args) -> None:
            return None

    def fake_make_engine(**kwargs) -> FakeEngine:
        return FakeEngine(kwargs["encoded_shuffle"])

    def fake_make_dataset(path, **_kwargs) -> None:
        path.touch()

    def fake_run_once(_path, engine) -> tuple[float, pl.DataFrame, dict]:
        total = 1 if engine.mode == "auto" else 2
        return 0.0, pl.DataFrame({"region": ["a"], "total": [total]}), {}

    monkeypatch.setattr(encoded_shuffle, "require_single_rank_launch", lambda: None)
    monkeypatch.setattr(encoded_shuffle, "make_dataset", fake_make_dataset)
    monkeypatch.setattr(encoded_shuffle, "_make_engine", fake_make_engine)
    monkeypatch.setattr(encoded_shuffle, "run_once", fake_run_once)
    monkeypatch.setattr(encoded_shuffle, "_print_json", lambda _record: None)

    with pytest.raises(ValueError, match="differ"):
        encoded_shuffle.main(
            [
                "--path",
                str(tmp_path / "encoded.parquet"),
                "--rows",
                "1",
                "--cardinality",
                "1",
                "--encoded-shuffle",
                "both",
                "--repeats",
                "1",
            ]
        )
