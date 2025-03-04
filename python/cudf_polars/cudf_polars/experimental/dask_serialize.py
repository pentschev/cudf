# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Dask serialization."""

from __future__ import annotations

from distributed.protocol import dask_deserialize, dask_serialize
from distributed.protocol.cuda import cuda_deserialize, cuda_serialize
from distributed.utils import log_errors

import pylibcudf as plc
import rmm

from cudf_polars.containers import Column, DataFrame

__all__ = ["register"]


def frames_to_gpumemoryview(frames):
    """
    Convert the elements of `frames` to gpumemoryview.

    UCX transfers produce `rmm.DeviceBuffer` objects instead of `gpumemoryview`.
    This function leverages CUDA array interface to convert the elements of
    `frames` to `gpumemoryview`, if necessary.

    Parameters
    ----------
    frames: list[Any]
        List of frames to convert to `gpumemoryview` if they implement CUDA
        array interface.

    Returns
    -------
    converted: list[Any]
        List of frames where all frames implementing CUDA array interface have
        been converted to `plc.gpumemoryview`.
    """
    return [
        plc.gpumemoryview(f) if hasattr(f, "__cuda_array_interface__") else f
        for f in frames
    ]


def register() -> None:
    """Register dask serialization routines for DataFrames."""

    @cuda_serialize.register((Column, DataFrame))
    def _(x: DataFrame | Column):
        with log_errors():
            header, frames = x.serialize()
            return header, list(frames)  # Dask expect a list of frames

    @cuda_deserialize.register(DataFrame)
    def _(header, frames):
        with log_errors():
            frames = frames_to_gpumemoryview(frames)
            assert len(frames) == 2
            return DataFrame.deserialize(header, tuple(frames))

    @cuda_deserialize.register(Column)
    def _(header, frames):
        with log_errors():
            frames = frames_to_gpumemoryview(frames)
            assert len(frames) == 2
            return Column.deserialize(header, tuple(frames))

    @dask_serialize.register((Column, DataFrame))
    def _(x: DataFrame | Column):
        with log_errors():
            header, (metadata, gpudata) = x.serialize()

            # For robustness, we check that the gpu data is contiguous
            cai = gpudata.__cuda_array_interface__
            assert len(cai["shape"]) == 1
            assert cai["strides"] is None or cai["strides"] == (1,)
            assert cai["typestr"] == "|u1"
            nbytes = cai["shape"][0]

            # Copy the gpudata to host memory
            gpudata_on_host = memoryview(
                rmm.DeviceBuffer(ptr=gpudata.ptr, size=nbytes).copy_to_host()
            )
            return header, (metadata, gpudata_on_host)

    @dask_deserialize.register(DataFrame)
    def _(header, frames) -> DataFrame:
        with log_errors():
            assert len(frames) == 2
            # Copy the second frame (the gpudata in host memory) back to the gpu
            frames = frames[0], plc.gpumemoryview(rmm.DeviceBuffer.to_device(frames[1]))
            return DataFrame.deserialize(header, frames)

    @dask_deserialize.register(Column)
    def _(header, frames) -> Column:
        with log_errors():
            assert len(frames) == 2
            # Copy the second frame (the gpudata in host memory) back to the gpu
            frames = frames[0], plc.gpumemoryview(rmm.DeviceBuffer.to_device(frames[1]))
            return Column.deserialize(header, frames)
