# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc

import pyarrow as pa

import pylibcudf as plc


def _indices(column):
    return sorted(column.to_arrow().to_pylist())


def _table(values):
    return plc.Table.from_arrow(
        pa.table({"key": pa.array(values, type=pa.int64())})
    )


def _mark_join_from_temporary_build():
    return plc.join.MarkJoin(_table([1, 2, 2, 3, 4]))


def test_mark_join_semi_join_matches_left_semi_join():
    build = _table([1, 2, 2, 3, 4])
    probe = _table([2, 4, 5])

    got = plc.join.MarkJoin(build).semi_join(probe)
    expect = plc.join.left_semi_join(
        build, probe, plc.types.NullEquality.EQUAL
    )

    assert _indices(got) == _indices(expect)


def test_mark_join_anti_join_matches_left_anti_join():
    build = _table([1, 2, 2, 3, 4])
    probe = _table([2, 4, 5])

    got = plc.join.MarkJoin(build).anti_join(probe)
    expect = plc.join.left_anti_join(
        build, probe, plc.types.NullEquality.EQUAL
    )

    assert _indices(got) == _indices(expect)


def test_mark_join_reuses_build_table_for_multiple_probes():
    build = _table([1, 2, 2, 3, 4])
    first_probe = _table([2, 4, 5])
    second_probe = _table([1, 3, 5])
    reusable = plc.join.MarkJoin(build, plc.types.NullEquality.EQUAL)

    first = reusable.semi_join(first_probe)
    second = reusable.semi_join(second_probe)
    third = reusable.anti_join(second_probe)

    assert _indices(first) == _indices(
        plc.join.left_semi_join(
            build, first_probe, plc.types.NullEquality.EQUAL
        )
    )
    assert _indices(second) == _indices(
        plc.join.left_semi_join(
            build, second_probe, plc.types.NullEquality.EQUAL
        )
    )
    assert _indices(third) == _indices(
        plc.join.left_anti_join(
            build, second_probe, plc.types.NullEquality.EQUAL
        )
    )


def test_mark_join_keeps_build_table_alive():
    build = _table([1, 2, 2, 3, 4])
    probe = _table([2, 4, 5])
    reusable = _mark_join_from_temporary_build()
    gc.collect()

    got = reusable.semi_join(probe)

    assert _indices(got) == _indices(
        plc.join.left_semi_join(build, probe, plc.types.NullEquality.EQUAL)
    )
