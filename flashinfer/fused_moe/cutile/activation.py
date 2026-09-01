# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable cuTile activation kernels for fused MoE pipelines."""

from __future__ import annotations

from typing import TypeAlias

import cuda.tile as ct
import torch

from ...tllm_enums import ActivationType
from ...utils import next_positive_power_of_2
from ..api import _CUTILE_SUPPORTED_ACTIVATIONS
from .indexing import needs_int64_indexing

ConstInt: TypeAlias = ct.Constant[int]

_SWIGLU = int(ActivationType.Swiglu)
_RELU2 = int(ActivationType.Relu2)


def _apply_activation(x, activation_type: ConstInt):
    """Apply activation math in FP32; ``activation_type`` is compile-time."""
    if activation_type == _SWIGLU:
        return x / (1.0 + ct.exp(-x))
    relu = ct.maximum(x, 0.0)
    return relu * relu


def _validate_activation(activation_type: ActivationType) -> ActivationType:
    activation_type = ActivationType(activation_type)
    if activation_type not in _CUTILE_SUPPORTED_ACTIVATIONS:
        raise NotImplementedError(
            f"cuTile MoE does not support activation {activation_type!r}."
        )
    return activation_type


@ct.function
def _gated_activation_impl(
    X,
    OUT,
    activation_type: ConstInt,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    row = ct.bid(0)
    offsets = ct.bid(1) * TILE_I + ct.arange(TILE_I, dtype=ct.int32)
    check_bounds = NUM_TILES * TILE_I != I
    gate = ct.astype(
        ct.gather(X, (row, offsets), check_bounds=check_bounds, latency=1),
        ct.float32,
    )
    up = ct.astype(
        ct.gather(X, (row, offsets + I), check_bounds=check_bounds, latency=1),
        ct.float32,
    )
    result = _apply_activation(gate, activation_type) * up
    ct.scatter(
        OUT,
        (row, offsets),
        ct.astype(result, OUT.dtype),
        check_bounds=check_bounds,
        latency=1,
    )


@ct.kernel
def _gated_activation(
    X,
    OUT,
    activation_type: ConstInt,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    _gated_activation_impl(X, OUT, activation_type, I, TILE_I, NUM_TILES)


@ct.kernel
def _gated_activation_i64(
    X: ct.IndexedWithInt64,
    OUT: ct.IndexedWithInt64,
    activation_type: ConstInt,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    _gated_activation_impl(X, OUT, activation_type, I, TILE_I, NUM_TILES)


@ct.function
def _ungated_activation_impl(
    X,
    OUT,
    activation_type: ConstInt,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    row = ct.bid(0)
    offsets = ct.bid(1) * TILE_I + ct.arange(TILE_I, dtype=ct.int32)
    check_bounds = NUM_TILES * TILE_I != I
    values = ct.astype(
        ct.gather(X, (row, offsets), check_bounds=check_bounds, latency=1),
        ct.float32,
    )
    result = _apply_activation(values, activation_type)
    ct.scatter(
        OUT,
        (row, offsets),
        ct.astype(result, OUT.dtype),
        check_bounds=check_bounds,
        latency=1,
    )


@ct.kernel
def _ungated_activation(
    X,
    OUT,
    activation_type: ConstInt,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    _ungated_activation_impl(X, OUT, activation_type, I, TILE_I, NUM_TILES)


@ct.kernel
def _ungated_activation_i64(
    X: ct.IndexedWithInt64,
    OUT: ct.IndexedWithInt64,
    activation_type: ConstInt,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    _ungated_activation_impl(X, OUT, activation_type, I, TILE_I, NUM_TILES)


def launch_activation(
    x: torch.Tensor,
    output: torch.Tensor,
    activation_type: ActivationType,
) -> None:
    """Launch a gated or plain MoE activation into caller-owned storage."""
    activation_type = _validate_activation(activation_type)
    if x.ndim != 2 or output.ndim != 2 or x.shape[0] != output.shape[0]:
        raise ValueError(
            "cuTile MoE activation expects 2D input/output with matching rows."
        )

    intermediate_size = output.shape[1]
    expected_input_size = intermediate_size * (2 if activation_type.is_gated else 1)
    if x.shape[1] != expected_input_size:
        raise ValueError(
            f"cuTile MoE activation input width {x.shape[1]} != expected "
            f"{expected_input_size} for {activation_type!r}."
        )

    tile_i = min(next_positive_power_of_2(intermediate_size), 1024)
    num_tiles = (intermediate_size + tile_i - 1) // tile_i
    use_int64 = needs_int64_indexing(x, output)
    if activation_type.is_gated:
        kernel = _gated_activation_i64 if use_int64 else _gated_activation
    else:
        kernel = _ungated_activation_i64 if use_int64 else _ungated_activation
    ct.launch(
        torch.cuda.current_stream(x.device),
        (x.shape[0], num_tiles),
        kernel,
        (
            x,
            output,
            int(activation_type),
            intermediate_size,
            tile_i,
            num_tiles,
        ),
    )


__all__ = ["launch_activation"]
