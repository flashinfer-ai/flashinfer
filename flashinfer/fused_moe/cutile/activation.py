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

from ...tllm_enums import ActivationType, DEFAULT_SWIGLU_LIMIT
from ...utils import next_positive_power_of_2
from ..api import (
    _CUTILE_SUPPORTED_ACTIVATIONS,
    ActivationConfig,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from .indexing import needs_int64_indexing

ConstFloat: TypeAlias = ct.Constant[float]
ConstInt: TypeAlias = ct.Constant[int]

_SWIGLU = int(ActivationType.Swiglu)
_SWIGLU_STEP = int(ActivationType.SwigluStep)
_GEGLU = int(ActivationType.Geglu)
_GEGLU_TANH = int(ActivationType.GegluTanh)
_SITU = int(ActivationType.Situ)
_RELU2 = int(ActivationType.Relu2)
_GELU = int(ActivationType.Gelu)
_RELU = int(ActivationType.Relu)
_SILU = int(ActivationType.Silu)


def _activation_kernel_args(
    activation: ActivationConfig,
) -> tuple[int, float, float, float]:
    """Lower a typed activation to the compact scalar cuTile kernel ABI."""
    activation = _validate_activation(activation)
    if isinstance(activation, SwiGLU):
        # Zero represents the API's unbounded float32-max default.
        clamp_limit = (
            0.0 if activation.limit == DEFAULT_SWIGLU_LIMIT else float(activation.limit)
        )
        return (
            int(activation.type),
            float(activation.alpha),
            float(activation.beta),
            clamp_limit,
        )
    if isinstance(activation, SwiGLUStep):
        return int(activation.type), float(activation.limit), 0.0, 0.0
    if isinstance(activation, SiTU):
        return (
            int(activation.type),
            float(activation.gate_scale),
            0.0 if activation.linear_scale is None else float(activation.linear_scale),
            0.0 if activation.clamp_limit is None else float(activation.clamp_limit),
        )
    return int(activation.type), 0.0, 0.0, 0.0


def _erf(x):
    """Approximate erf to ~1.5e-7 absolute error for exact GELU/GeGLU."""
    abs_x = ct.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * abs_x)
    polynomial = (
        (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t
    magnitude = 1.0 - polynomial * ct.exp(-(abs_x * abs_x))
    return ct.where(x < 0.0, -magnitude, magnitude)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x * 0.7071067811865476))


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + ct.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


def _silu(x):
    return x / (1.0 + ct.exp(-x))


def _tanh_via_sigmoid(x):
    # SiTU's linear scale can amplify tanh approximation error, so mirror the
    # CUTLASS implementation and derive tanh from the exponential sigmoid.
    return 2.0 / (1.0 + ct.exp(-2.0 * x)) - 1.0


def _apply_ungated_activation(
    x,
    activation_type: ConstInt,
    param1: ConstFloat,
    param2: ConstFloat,
    param3: ConstFloat,
):
    """Apply a non-gated activation in FP32; all scalars are compile-time."""
    if activation_type == _GELU:
        return _gelu(x)
    if activation_type == _RELU:
        return ct.maximum(x, 0.0)
    if activation_type == _SILU:
        return _silu(x)
    if activation_type == _RELU2:
        relu = ct.maximum(x, 0.0)
        return relu * relu
    # _IDENTITY
    return x


def _apply_gated_activation(
    gate,
    up,
    activation_type: ConstInt,
    param1: ConstFloat,
    param2: ConstFloat,
    param3: ConstFloat,
):
    """Apply a gated activation in FP32; all scalars are compile-time."""
    if activation_type == _SWIGLU:
        if param3 > 0.0:
            gate = ct.minimum(gate, param3)
            up = ct.maximum(ct.minimum(up, param3), -param3)
        return gate * (1.0 / (1.0 + ct.exp(-(param1 * gate)))) * (up + param2)
    if activation_type == _SWIGLU_STEP:
        return ct.minimum(_silu(gate), param1) * ct.maximum(
            ct.minimum(up, param1), -param1
        )
    if activation_type == _GEGLU:
        return _gelu(gate) * up
    if activation_type == _GEGLU_TANH:
        return _gelu_tanh(gate) * up

    if activation_type == _SITU:
        if param3 > 0.0:
            gate = ct.minimum(gate, param3)
            up = ct.maximum(ct.minimum(up, param3), -param3)
        gate = param1 * _tanh_via_sigmoid(gate / param1) * (1.0 / (1.0 + ct.exp(-gate)))
        if param2 > 0.0:
            up = param2 * _tanh_via_sigmoid(up / param2)
        return gate * up
    # Should not reach here.
    return gate * up


def _validate_activation(activation: ActivationConfig) -> ActivationConfig:
    if not isinstance(activation, ActivationConfig):
        raise TypeError(
            "cuTile MoE activation must be an ActivationConfig value, got "
            f"{type(activation).__name__}."
        )
    if activation.type not in _CUTILE_SUPPORTED_ACTIVATIONS:
        raise NotImplementedError(
            f"cuTile MoE does not support activation {activation.type!r}."
        )
    return activation


@ct.function
def _gated_activation_impl(
    X,
    OUT,
    activation_type: ConstInt,
    param1: ConstFloat,
    param2: ConstFloat,
    param3: ConstFloat,
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
    result = _apply_gated_activation(gate, up, activation_type, param1, param2, param3)
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
    param1: ConstFloat,
    param2: ConstFloat,
    param3: ConstFloat,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    _gated_activation_impl(
        X, OUT, activation_type, param1, param2, param3, I, TILE_I, NUM_TILES
    )


@ct.kernel
def _gated_activation_i64(
    X: ct.IndexedWithInt64,
    OUT: ct.IndexedWithInt64,
    activation_type: ConstInt,
    param1: ConstFloat,
    param2: ConstFloat,
    param3: ConstFloat,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    _gated_activation_impl(
        X, OUT, activation_type, param1, param2, param3, I, TILE_I, NUM_TILES
    )


@ct.function
def _ungated_activation_impl(
    X,
    OUT,
    activation_type: ConstInt,
    param1: ConstFloat,
    param2: ConstFloat,
    param3: ConstFloat,
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
    result = _apply_ungated_activation(values, activation_type, param1, param2, param3)
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
    param1: ConstFloat,
    param2: ConstFloat,
    param3: ConstFloat,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    _ungated_activation_impl(
        X, OUT, activation_type, param1, param2, param3, I, TILE_I, NUM_TILES
    )


@ct.kernel
def _ungated_activation_i64(
    X: ct.IndexedWithInt64,
    OUT: ct.IndexedWithInt64,
    activation_type: ConstInt,
    param1: ConstFloat,
    param2: ConstFloat,
    param3: ConstFloat,
    I: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
):
    _ungated_activation_impl(
        X, OUT, activation_type, param1, param2, param3, I, TILE_I, NUM_TILES
    )


def launch_activation(
    x: torch.Tensor,
    output: torch.Tensor,
    activation: ActivationConfig,
) -> None:
    """Launch a gated or plain MoE activation into caller-owned storage."""
    activation = _validate_activation(activation)
    if x.ndim != 2 or output.ndim != 2 or x.shape[0] != output.shape[0]:
        raise ValueError(
            "cuTile MoE activation expects 2D input/output with matching rows."
        )

    intermediate_size = output.shape[1]
    expected_input_size = intermediate_size * (2 if activation.is_gated else 1)
    if x.shape[1] != expected_input_size:
        raise ValueError(
            f"cuTile MoE activation input width {x.shape[1]} != expected "
            f"{expected_input_size} for {activation.type!r}."
        )

    tile_i = min(next_positive_power_of_2(intermediate_size), 1024)
    num_tiles = (intermediate_size + tile_i - 1) // tile_i
    use_int64 = needs_int64_indexing(x, output)
    if activation.is_gated:
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
            *_activation_kernel_args(activation),
            intermediate_size,
            tile_i,
            num_tiles,
        ),
    )


__all__ = ["launch_activation"]
