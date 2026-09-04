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

"""NVFP4 cuTile kernels for the unified MoE backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import cuda.tile as ct
import torch

from ...cutile.cutile_common import cached_replace_hints
from ...utils import next_positive_power_of_2
from ..api import ActivationConfig
from .activation import (
    _activation_kernel_args,
    _apply_gated_activation,
    _apply_ungated_activation,
    _validate_activation,
    launch_activation,
)
from .indexing import needs_int64_indexing
from .moe import (
    GemmConfig,
    Workspace as Bf16Workspace,
    _combine,
    _combine_i64,
    _combine_tile_h,
    _permute,
    allocate_workspace as allocate_bf16_workspace,
)

ConstFloat: TypeAlias = ct.Constant[float]
ConstInt: TypeAlias = ct.Constant[int]
ConstBool: TypeAlias = ct.Constant[bool]

_NVFP4_BLOCK_SIZE = 16
_E4M3_TINY = 2.0**-9
_SORT_INPUT_MIN_ASSIGNMENTS = 4096
_SORT_INPUT_SMALL_I_MIN_ASSIGNMENTS = 65536
_PERSISTENT_CTAS_PER_SM = 2
_PERSISTENT_MIN_ASSIGNMENTS = 65536
_PERSISTENT_MAX_K = 1024
_SEPARATE_ACTIVATION_MAX_ASSIGNMENTS = 64


def _use_row_major_scale_layout(num_assignments: int, intermediate_size: int) -> bool:
    return num_assignments >= _SORT_INPUT_MIN_ASSIGNMENTS and (
        intermediate_size >= 1024
        or num_assignments >= _SORT_INPUT_SMALL_I_MIN_ASSIGNMENTS
    )


def _persistent_grid_m(
    x: torch.Tensor,
    grid_m: int,
    num_n_blocks: int,
) -> int:
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    return min(
        grid_m,
        max(1, _PERSISTENT_CTAS_PER_SM * num_sms // num_n_blocks),
    )


@dataclass
class Workspace(Bf16Workspace):
    """Graph-stable workspace for the W4A4 pipeline."""

    input_q: torch.Tensor
    input_scale: torch.Tensor
    sorted_input_q: torch.Tensor | None
    sorted_input_scale: torch.Tensor | None
    activation_q: torch.Tensor
    activation_scale: torch.Tensor
    scale_row_major: bool


def allocate_workspace(
    *,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    is_gated: bool,
    block_sizes: tuple[int, ...],
    device: torch.device,
) -> Workspace:
    """Allocate graph-stable routing, GEMM, and quantization buffers."""
    num_assignments = num_tokens * top_k
    use_sorted_layout = _use_row_major_scale_layout(num_assignments, intermediate_size)
    max_assignment_rows = max(
        num_assignments + num_experts * (block_size - 1) for block_size in block_sizes
    )
    assignment_rows = (
        max_assignment_rows
        if use_sorted_layout
        else ((num_assignments + 127) // 128) * 128
    )
    base = allocate_bf16_workspace(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        is_gated=is_gated,
        block_sizes=block_sizes,
        # Hybrid buckets can be almost 2x the runtime token count. Keep the
        # small gated-activation buffer whenever this bucket can contain a
        # launch with fewer than 64 assignments.
        allocate_activation_output=(
            is_gated and num_tokens * top_k < 2 * _SEPARATE_ACTIVATION_MAX_ASSIGNMENTS
        ),
        allocate_gemm1_output=True,
        gemm1_output_rows=assignment_rows,
        device=device,
    )
    input_rows = ((num_tokens + 127) // 128) * 128
    scale_row_major = use_sorted_layout
    input_q = torch.empty(
        input_rows, hidden_size // 2, dtype=torch.uint8, device=device
    )
    input_scale_shape = (
        (input_rows, hidden_size // _NVFP4_BLOCK_SIZE)
        if scale_row_major
        else (hidden_size // _NVFP4_BLOCK_SIZE, input_rows)
    )
    input_scale = torch.empty(
        input_scale_shape,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    if use_sorted_layout:
        sorted_input_q = torch.empty(
            base.sorted_slots.shape[0],
            hidden_size // 2,
            dtype=torch.uint8,
            device=device,
        )
        sorted_input_scale = torch.empty(
            base.sorted_slots.shape[0],
            hidden_size // _NVFP4_BLOCK_SIZE,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
    else:
        sorted_input_q = sorted_input_scale = None
    activation_q = torch.empty(
        assignment_rows,
        intermediate_size // 2,
        dtype=torch.uint8,
        device=device,
    )
    activation_scale_shape = (
        (assignment_rows, intermediate_size // _NVFP4_BLOCK_SIZE)
        if scale_row_major
        else (intermediate_size // _NVFP4_BLOCK_SIZE, assignment_rows)
    )
    activation_scale = torch.empty(
        activation_scale_shape,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    return Workspace(
        **vars(base),
        input_q=input_q,
        input_scale=input_scale,
        sorted_input_q=sorted_input_q,
        sorted_input_scale=sorted_input_scale,
        activation_q=activation_q,
        activation_scale=activation_scale,
        scale_row_major=scale_row_major,
    )


def _encode_nvfp4_groups(
    groups,
    tile_m: ConstInt,
    tile_k: ConstInt,
):
    amax = ct.max(ct.abs(groups), axis=2)
    scale = ct.astype(amax / 6.0, ct.float8_e4m3fn)
    dequant_scale = ct.maximum(ct.astype(scale, ct.float32), _E4M3_TINY)
    values = groups / ct.expand_dims(dequant_scale, axis=2)
    magnitude = ct.abs(values)
    u8 = ct.uint8
    # These are the midpoints between positive E2M1 values.
    codes = (
        ct.astype(magnitude > 0.25, u8)
        + ct.astype(magnitude > 0.75, u8)
        + ct.astype(magnitude > 1.25, u8)
        + ct.astype(magnitude > 1.75, u8)
        + ct.astype(magnitude > 2.5, u8)
        + ct.astype(magnitude > 3.5, u8)
        + ct.astype(magnitude > 5.0, u8)
        + ct.astype(values < 0.0, u8) * 8
    )
    pairs = ct.reshape(codes, (tile_m, tile_k // 2, 2))
    pair_weights = ct.astype(ct.arange(2, dtype=ct.int32) * 15 + 1, u8)
    packed = ct.astype(ct.sum(pairs * pair_weights, axis=2), u8)
    return packed, scale


@ct.function
def _quantize_nvfp4_impl(
    X,
    Q,
    SCALE,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
):
    row_block = ct.bid(0)
    k_block = ct.bid(1)
    groups_per_tile = TILE_K // _NVFP4_BLOCK_SIZE
    x = ct.load(
        X,
        index=(row_block, k_block),
        shape=(TILE_M, TILE_K),
        padding_mode=ct.PaddingMode.ZERO,
    )
    groups = ct.reshape(
        ct.astype(x, ct.float32),
        (TILE_M, groups_per_tile, _NVFP4_BLOCK_SIZE),
    )
    packed, scale = _encode_nvfp4_groups(groups, TILE_M, TILE_K)
    ct.store(Q, index=(row_block, k_block), tile=packed)
    if SCALE_ROW_MAJOR:
        ct.store(SCALE, index=(row_block, k_block), tile=scale)
    else:
        ct.store(
            SCALE,
            index=(k_block, row_block),
            tile=ct.permute(scale, (1, 0)),
        )


@ct.kernel
def _quantize_nvfp4(
    X,
    Q,
    SCALE,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
):
    _quantize_nvfp4_impl(X, Q, SCALE, K, TILE_M, TILE_K, SCALE_ROW_MAJOR)


@ct.kernel
def _quantize_nvfp4_i64(
    X: ct.IndexedWithInt64,
    Q: ct.IndexedWithInt64,
    SCALE: ct.IndexedWithInt64,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
):
    _quantize_nvfp4_impl(X, Q, SCALE, K, TILE_M, TILE_K, SCALE_ROW_MAJOR)


@ct.function
def _activation_quantize_nvfp4_impl(
    X,
    Q,
    SCALE,
    activation_type: ConstInt,
    activation_param1: ConstFloat,
    activation_param2: ConstFloat,
    activation_param3: ConstFloat,
    IS_GATED: ConstBool,
    TILE_M: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
):
    row_block = ct.bid(0)
    i_block = ct.bid(1)
    groups_per_tile = TILE_I // _NVFP4_BLOCK_SIZE
    values = ct.astype(
        ct.load(
            X,
            index=(row_block, i_block),
            shape=(TILE_M, TILE_I),
            padding_mode=ct.PaddingMode.ZERO,
        ),
        ct.float32,
    )
    if IS_GATED:
        up = ct.astype(
            ct.load(
                X,
                index=(row_block, i_block + NUM_TILES),
                shape=(TILE_M, TILE_I),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            ct.float32,
        )
        values = _apply_gated_activation(
            values,
            up,
            activation_type,
            activation_param1,
            activation_param2,
            activation_param3,
        )
    else:
        values = _apply_ungated_activation(
            values,
            activation_type,
            activation_param1,
            activation_param2,
            activation_param3,
        )

    # Preserve the existing BF16 activation boundary while eliminating its
    # global-memory round trip.
    groups = ct.reshape(
        ct.astype(ct.astype(values, X.dtype), ct.float32),
        (TILE_M, groups_per_tile, _NVFP4_BLOCK_SIZE),
    )
    packed, scale = _encode_nvfp4_groups(groups, TILE_M, TILE_I)
    ct.store(Q, index=(row_block, i_block), tile=packed)
    if SCALE_ROW_MAJOR:
        ct.store(SCALE, index=(row_block, i_block), tile=scale)
    else:
        ct.store(
            SCALE,
            index=(i_block, row_block),
            tile=ct.permute(scale, (1, 0)),
        )


@ct.kernel
def _activation_quantize_nvfp4(
    X,
    Q,
    SCALE,
    activation_type: ConstInt,
    activation_param1: ConstFloat,
    activation_param2: ConstFloat,
    activation_param3: ConstFloat,
    IS_GATED: ConstBool,
    TILE_M: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
):
    _activation_quantize_nvfp4_impl(
        X,
        Q,
        SCALE,
        activation_type,
        activation_param1,
        activation_param2,
        activation_param3,
        IS_GATED,
        TILE_M,
        TILE_I,
        NUM_TILES,
        SCALE_ROW_MAJOR,
    )


@ct.kernel
def _activation_quantize_nvfp4_i64(
    X: ct.IndexedWithInt64,
    Q: ct.IndexedWithInt64,
    SCALE: ct.IndexedWithInt64,
    activation_type: ConstInt,
    activation_param1: ConstFloat,
    activation_param2: ConstFloat,
    activation_param3: ConstFloat,
    IS_GATED: ConstBool,
    TILE_M: ConstInt,
    TILE_I: ConstInt,
    NUM_TILES: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
):
    _activation_quantize_nvfp4_impl(
        X,
        Q,
        SCALE,
        activation_type,
        activation_param1,
        activation_param2,
        activation_param3,
        IS_GATED,
        TILE_M,
        TILE_I,
        NUM_TILES,
        SCALE_ROW_MAJOR,
    )


def _input_quantize_config(
    rows: int,
    k: int,
    num_sms: int,
    scale_row_major: bool,
) -> tuple[int, int, int]:
    """Select TILE_M, TILE_K, and occupancy for input quantization."""
    max_tile_k = next(tile_k for tile_k in (256, 128, 64) if k % tile_k == 0)
    if scale_row_major:
        if max_tile_k == 256:
            return 8, 256, 4
        if rows < 4 * num_sms:
            return 16, max_tile_k, 4
        return 64, 64, 4

    many_k_tiles = max_tile_k == 256 and k // 256 >= 16
    if rows <= 8:
        return 2, min(max_tile_k, 128), 0
    if rows <= 16:
        return 4, max_tile_k, 0
    if rows <= 32:
        return (2 if max_tile_k == 256 else 4), max_tile_k, (4 if many_k_tiles else 0)
    if rows <= 64:
        return (4, 256, 4) if max_tile_k == 256 else (16, 64, 4)
    if rows <= 128:
        if many_k_tiles:
            return 8, 128, 4
        return (4, 256, 4) if max_tile_k == 256 else (16, 64, 4)
    if rows <= 256:
        if many_k_tiles:
            return 8, 256, 4
        if max_tile_k >= 128:
            return 8, 128, 4
        return 16, 64, 4
    if many_k_tiles:
        return 32, 64, 4
    if max_tile_k == 256:
        return 4, 256, 4
    if max_tile_k >= 128:
        return 8, 128, 4
    return 16, 64, 4


def _quantize(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    *,
    scale_row_major: bool = False,
) -> None:
    rows, k = x.shape
    if k % 64 != 0:
        raise ValueError(f"W4A4 activation dimension must be divisible by 64, got {k}.")
    expected_scale_shape = (
        (q.shape[0], k // _NVFP4_BLOCK_SIZE)
        if scale_row_major
        else (k // _NVFP4_BLOCK_SIZE, q.shape[0])
    )
    if scale.shape != expected_scale_shape:
        raise ValueError(
            f"cuTile NVFP4 scale shape must be {expected_scale_shape}, got "
            f"{tuple(scale.shape)}."
        )
    num_sms = (
        torch.cuda.get_device_properties(x.device).multi_processor_count
        if scale_row_major
        else 1
    )
    tile_m, tile_k, occupancy = _input_quantize_config(
        rows, k, num_sms, scale_row_major
    )
    base_kernel = (
        _quantize_nvfp4_i64 if needs_int64_indexing(x, q, scale) else _quantize_nvfp4
    )
    kernel = (
        base_kernel
        if occupancy == 0
        else cached_replace_hints(base_kernel, occupancy=occupancy)
    )
    ct.launch(
        torch.cuda.current_stream(x.device),
        (((rows + tile_m - 1) // tile_m), k // tile_k),
        kernel,
        (x, q, scale, k, tile_m, tile_k, scale_row_major),
    )


@ct.function
def _pack_nvfp4_rows_impl(
    Q,
    SCALE,
    SORTED_SLOTS,
    OUT_Q,
    OUT_SCALE,
    top_k,
    input_rows,
    scale_stride,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
    USE_INT64: ConstBool,
):
    row_block = ct.bid(0)
    k_block = ct.bid(1)
    row_offsets = row_block * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    slots = ct.gather(SORTED_SLOTS, (row_offsets,), padding_value=input_rows * top_k)
    source_rows = slots // top_k
    if USE_INT64:
        source_rows = ct.astype(source_rows, ct.int64)

    packed_offsets = ct.reshape(
        k_block * (TILE_K // 2) + ct.arange(TILE_K // 2, dtype=ct.int32),
        (1, TILE_K // 2),
    )
    packed = ct.gather(
        Q,
        (ct.reshape(source_rows, (TILE_M, 1)) * (K // 2) + packed_offsets,),
        padding_value=0,
    )
    ct.store(OUT_Q, index=(row_block, k_block), tile=packed)

    groups_per_tile = TILE_K // _NVFP4_BLOCK_SIZE
    group_offsets = ct.reshape(
        k_block * groups_per_tile + ct.arange(groups_per_tile, dtype=ct.int32),
        (1, groups_per_tile),
    )
    if SCALE_ROW_MAJOR:
        scale_indices = (
            ct.reshape(source_rows, (TILE_M, 1)) * scale_stride + group_offsets
        )
    else:
        scale_indices = group_offsets * scale_stride + ct.reshape(
            source_rows, (TILE_M, 1)
        )
    packed_scale = ct.gather(SCALE, (scale_indices,), padding_value=0.0)
    ct.store(OUT_SCALE, index=(row_block, k_block), tile=packed_scale)


@ct.kernel
def _pack_nvfp4_rows(
    Q,
    SCALE,
    SORTED_SLOTS,
    OUT_Q,
    OUT_SCALE,
    top_k,
    input_rows,
    scale_stride,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
):
    _pack_nvfp4_rows_impl(
        Q,
        SCALE,
        SORTED_SLOTS,
        OUT_Q,
        OUT_SCALE,
        top_k,
        input_rows,
        scale_stride,
        K,
        TILE_M,
        TILE_K,
        SCALE_ROW_MAJOR,
        False,
    )


@ct.kernel
def _pack_nvfp4_rows_i64(
    Q: ct.IndexedWithInt64,
    SCALE: ct.IndexedWithInt64,
    SORTED_SLOTS,
    OUT_Q: ct.IndexedWithInt64,
    OUT_SCALE: ct.IndexedWithInt64,
    top_k,
    input_rows,
    scale_stride,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
):
    _pack_nvfp4_rows_impl(
        Q,
        SCALE,
        SORTED_SLOTS,
        OUT_Q,
        OUT_SCALE,
        top_k,
        input_rows,
        scale_stride,
        K,
        TILE_M,
        TILE_K,
        SCALE_ROW_MAJOR,
        True,
    )


def _pack_input(
    q: torch.Tensor,
    scale: torch.Tensor,
    sorted_slots: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    top_k: int,
    block_size: int,
    scale_row_major: bool,
) -> None:
    k = q.shape[1] * 2
    tile_k = 128 if k % 128 == 0 else 64
    scale_stride = scale.shape[1]
    ct.launch(
        torch.cuda.current_stream(q.device),
        (
            (sorted_slots.shape[0] + block_size - 1) // block_size,
            k // tile_k,
        ),
        (
            _pack_nvfp4_rows_i64
            if needs_int64_indexing(q, scale, output_q, output_scale)
            else _pack_nvfp4_rows
        ),
        (
            q.reshape(-1),
            scale.reshape(-1),
            sorted_slots,
            output_q,
            output_scale,
            top_k,
            q.shape[0],
            scale_stride,
            k,
            block_size,
            tile_k,
            scale_row_major,
        ),
    )


def _launch_activation_quantize(
    x: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
    activation: ActivationConfig,
    *,
    tile_i: int = 64,
    occupancy: int = 2,
    scale_row_major: bool = False,
) -> None:
    activation = _validate_activation(activation)
    intermediate_size = q.shape[1] * 2
    expected_input_size = intermediate_size * (2 if activation.is_gated else 1)
    if x.ndim != 2 or x.shape[1] != expected_input_size:
        raise ValueError(
            f"cuTile NVFP4 activation input shape {tuple(x.shape)} is incompatible "
            f"with intermediate_size={intermediate_size} and {activation.type!r}."
        )
    if intermediate_size % 64 != 0:
        raise ValueError(
            "cuTile NVFP4 activation width must be divisible by 64, got "
            f"{intermediate_size}."
        )
    if tile_i not in (64, 128) or intermediate_size % tile_i != 0:
        raise ValueError(
            f"cuTile NVFP4 activation tile {tile_i} must be 64 or 128 and "
            f"divide intermediate_size={intermediate_size}."
        )
    expected_scale_shape = (
        (q.shape[0], intermediate_size // _NVFP4_BLOCK_SIZE)
        if scale_row_major
        else (intermediate_size // _NVFP4_BLOCK_SIZE, q.shape[0])
    )
    if scale.shape != expected_scale_shape:
        raise ValueError(
            f"cuTile NVFP4 scale shape must be {expected_scale_shape}, got "
            f"{tuple(scale.shape)}."
        )
    num_tiles = intermediate_size // tile_i
    tile_m = min(128, next_positive_power_of_2(x.shape[0] + 1))
    base_kernel = (
        _activation_quantize_nvfp4_i64
        if needs_int64_indexing(x, q, scale)
        else _activation_quantize_nvfp4
    )
    kernel = cached_replace_hints(base_kernel, occupancy=occupancy)
    ct.launch(
        torch.cuda.current_stream(x.device),
        (((x.shape[0] + tile_m - 1) // tile_m), num_tiles),
        kernel,
        (
            x,
            q,
            scale,
            *_activation_kernel_args(activation),
            activation.is_gated,
            tile_m,
            tile_i,
            num_tiles,
            scale_row_major,
        ),
    )


def _activation_quantize_config(
    rows: int,
    intermediate_size: int,
    num_sms: int,
    scale_row_major: bool,
) -> tuple[int, int]:
    """Select TILE_I and occupancy for fused activation quantization."""
    work = rows * intermediate_size
    occupancy = 4 if scale_row_major or work >= num_sms * 4096 else 2
    # A 64-wide tile covers four NVFP4 scale groups and was robust in cold L2.
    return 64, occupancy


def _launch_unfused_activation_quantize(
    gemm1_out: torch.Tensor,
    workspace: Workspace,
    activation: ActivationConfig,
    *,
    num_assignments: int,
    num_sms: int,
) -> None:
    if activation.is_gated and num_assignments < _SEPARATE_ACTIVATION_MAX_ASSIGNMENTS:
        if workspace.activation_out.numel() == 0:
            raise RuntimeError("W4A4 workspace is missing the activation buffer.")
        activation_out = workspace.activation_out[:num_assignments]
        launch_activation(gemm1_out, activation_out, activation)
        _quantize(
            activation_out,
            workspace.activation_q,
            workspace.activation_scale,
            scale_row_major=workspace.scale_row_major,
        )
        return

    tile_i, occupancy = _activation_quantize_config(
        gemm1_out.shape[0],
        workspace.activation_q.shape[1] * 2,
        num_sms,
        workspace.scale_row_major,
    )
    _launch_activation_quantize(
        gemm1_out,
        workspace.activation_q,
        workspace.activation_scale,
        activation,
        tile_i=tile_i,
        occupancy=occupancy,
        scale_row_major=workspace.scale_row_major,
    )


def _unswizzle_32_4_4(scale):
    m0 = scale.shape[0]
    k0 = scale.shape[1]
    return ct.reshape(
        ct.permute(ct.reshape(scale, (m0, k0, 32, 4, 4)), (0, 3, 2, 1, 4)),
        (m0 * 128, k0 * 4),
    )


def _load_w4a4_weight_tile(
    weights,
    weight_scale,
    expert,
    n_block,
    k_tile,
    tile_n: ConstInt,
    tile_k: ConstInt,
):
    groups_per_tile = tile_k // _NVFP4_BLOCK_SIZE
    weight_bytes = ct.reshape(
        ct.load(
            weights,
            index=(expert, n_block, k_tile),
            shape=(1, tile_n, tile_k // 2),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
            allow_tma=True,
        ),
        (tile_n, tile_k // 2),
    )
    weight = ct.reshape(
        ct.unpack_from_bytes(ct.reshape(weight_bytes, (-1,)), ct.float4_e2m1fn),
        (tile_n, tile_k),
    )
    weight_scale_swizzled = ct.reshape(
        ct.load(
            weight_scale,
            index=(expert, n_block, k_tile, 0, 0),
            shape=(1, tile_n // 128, groups_per_tile // 4, 32, 16),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
            allow_tma=True,
        ),
        (tile_n // 128, groups_per_tile // 4, 32, 16),
    )
    decoded_scale = ct.permute(_unswizzle_32_4_4(weight_scale_swizzled), (1, 0))
    return weight, decoded_scale


def _load_w4a4_activation_tile(
    activation,
    activation_scale,
    rows,
    k_tile,
    activation_scale_stride,
    k_in: ConstInt,
    tile_m: ConstInt,
    tile_k: ConstInt,
    scale_row_major: ConstBool,
):
    groups_per_tile = tile_k // _NVFP4_BLOCK_SIZE
    packed_k_offsets = ct.reshape(
        k_tile * (tile_k // 2) + ct.arange(tile_k // 2, dtype=ct.int32),
        (1, tile_k // 2),
    )
    activation_indices = ct.reshape(rows, (tile_m, 1)) * (k_in // 2) + packed_k_offsets
    activation_bytes = ct.gather(
        activation,
        (activation_indices,),
        padding_value=0,
        check_bounds=True,
        latency=3,
    )
    activation_bytes = ct.where(
        packed_k_offsets < k_in // 2,
        activation_bytes,
        0,
    )
    decoded_activation = ct.reshape(
        ct.unpack_from_bytes(
            ct.reshape(activation_bytes, (-1,)),
            ct.float4_e2m1fn,
        ),
        (tile_m, tile_k),
    )
    group_offsets = ct.reshape(
        k_tile * groups_per_tile + ct.arange(groups_per_tile, dtype=ct.int32),
        (1, groups_per_tile),
    )
    if scale_row_major:
        activation_scale_indices = (
            group_offsets + ct.reshape(rows, (tile_m, 1)) * activation_scale_stride
        )
    else:
        activation_scale_indices = group_offsets * activation_scale_stride + ct.reshape(
            rows, (tile_m, 1)
        )
    decoded_scale = ct.gather(
        activation_scale,
        (activation_scale_indices,),
        padding_value=0.0,
        check_bounds=True,
        latency=3,
    )
    decoded_scale = ct.where(
        group_offsets < k_in // _NVFP4_BLOCK_SIZE,
        decoded_scale,
        0.0,
    )
    return decoded_activation, decoded_scale


def _load_w4a4_contiguous_activation_tile(
    activation,
    activation_scale,
    m_block,
    k_tile,
    tile_m: ConstInt,
    tile_k: ConstInt,
):
    activation_bytes = ct.reshape(
        ct.load(
            activation,
            index=(m_block, k_tile),
            shape=(tile_m, tile_k // 2),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
            allow_tma=True,
        ),
        (tile_m, tile_k // 2),
    )
    decoded_activation = ct.reshape(
        ct.unpack_from_bytes(
            ct.reshape(activation_bytes, (-1,)),
            ct.float4_e2m1fn,
        ),
        (tile_m, tile_k),
    )
    groups_per_tile = tile_k // _NVFP4_BLOCK_SIZE
    decoded_scale = ct.reshape(
        ct.load(
            activation_scale,
            index=(m_block, k_tile),
            shape=(tile_m, groups_per_tile),
            padding_mode=ct.PaddingMode.ZERO,
            latency=3,
            allow_tma=True,
        ),
        (tile_m, groups_per_tile),
    )
    return decoded_activation, decoded_scale


@ct.function
def _grouped_gemm_w4a4_impl(
    X,
    X_SCALE,
    SORTED_X,
    SORTED_X_SCALE,
    W,
    W_SCALE,
    W_GLOBAL_SCALE,
    SORTED_SLOTS,
    BLOCK_EXPERT,
    NUM_POST_PAD,
    OUT,
    top_k,
    grid_m,
    activation_scale_groups,
    global_scale_shards,
    global_scale_shard_width,
    K_IN: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
    SHARDED_GLOBAL_SCALE: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
    USE_INT64: ConstBool,
):
    initial_m_block = ct.bid(0)
    n_block = ct.bid(1)
    num_post_pad = ct.gather(
        NUM_POST_PAD, ct.zeros((1,), dtype=ct.int32), padding_value=0
    ).item()
    num_live_blocks = (num_post_pad + TILE_M - 1) // TILE_M
    num_iterations = (num_live_blocks - initial_m_block + grid_m - 1) // grid_m
    n_offsets = n_block * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    num_k_tiles = (K_IN + TILE_K - 1) // TILE_K
    for iteration in range(num_iterations):
        m_block = initial_m_block + iteration * grid_m
        if m_block * TILE_M < num_post_pad:
            m_offsets = m_block * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
            slots = ct.gather(SORTED_SLOTS, (m_offsets,), padding_value=0)
            expert = ct.gather(
                BLOCK_EXPERT,
                ct.full((1,), m_block, dtype=ct.int32),
                padding_value=0,
            ).item()
            if SHARDED_GLOBAL_SCALE:
                shards = n_offsets // global_scale_shard_width
                alpha = ct.reshape(
                    ct.gather(
                        W_GLOBAL_SCALE,
                        (expert * global_scale_shards + shards,),
                        padding_value=1.0,
                    ),
                    (1, TILE_N),
                )
            else:
                alpha = ct.gather(
                    W_GLOBAL_SCALE,
                    ct.full(
                        (1,),
                        expert * global_scale_shards,
                        dtype=ct.int32,
                    ),
                    padding_value=1.0,
                ).item()
            rows = slots // top_k
            if USE_INT64:
                rows = ct.astype(rows, ct.int64)
            accumulator = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
            for k_tile in range(num_k_tiles):
                if INPUT_SORTED:
                    activation, activation_scale = (
                        _load_w4a4_contiguous_activation_tile(
                            SORTED_X,
                            SORTED_X_SCALE,
                            m_block,
                            k_tile,
                            TILE_M,
                            TILE_K,
                        )
                    )
                else:
                    activation, activation_scale = _load_w4a4_activation_tile(
                        X,
                        X_SCALE,
                        rows,
                        k_tile,
                        activation_scale_groups,
                        K_IN,
                        TILE_M,
                        TILE_K,
                        SCALE_ROW_MAJOR,
                    )
                weight, weight_scale = _load_w4a4_weight_tile(
                    W,
                    W_SCALE,
                    expert,
                    n_block,
                    k_tile,
                    TILE_N,
                    TILE_K,
                )
                accumulator = ct.mma_scaled(
                    activation,
                    activation_scale,
                    ct.permute(weight, (1, 0)),
                    weight_scale,
                    accumulator,
                )
            output_rows = m_offsets if OUTPUT_SORTED else slots
            ct.scatter(
                OUT,
                (
                    ct.reshape(output_rows, (TILE_M, 1)),
                    ct.reshape(n_offsets, (1, TILE_N)),
                ),
                ct.astype(accumulator * alpha, OUT.dtype),
                check_bounds=True,
            )


@ct.kernel
def _grouped_gemm_w4a4(
    X,
    X_SCALE,
    SORTED_X,
    SORTED_X_SCALE,
    W,
    W_SCALE,
    W_GLOBAL_SCALE,
    SORTED_SLOTS,
    BLOCK_EXPERT,
    NUM_POST_PAD,
    OUT,
    top_k,
    grid_m,
    activation_scale_groups,
    global_scale_shards,
    global_scale_shard_width,
    K_IN: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
    SHARDED_GLOBAL_SCALE: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
):
    _grouped_gemm_w4a4_impl(
        X,
        X_SCALE,
        SORTED_X,
        SORTED_X_SCALE,
        W,
        W_SCALE,
        W_GLOBAL_SCALE,
        SORTED_SLOTS,
        BLOCK_EXPERT,
        NUM_POST_PAD,
        OUT,
        top_k,
        grid_m,
        activation_scale_groups,
        global_scale_shards,
        global_scale_shard_width,
        K_IN,
        TILE_M,
        TILE_N,
        TILE_K,
        SCALE_ROW_MAJOR,
        SHARDED_GLOBAL_SCALE,
        INPUT_SORTED,
        OUTPUT_SORTED,
        False,
    )


@ct.kernel
def _grouped_gemm_w4a4_i64(
    X: ct.IndexedWithInt64,
    X_SCALE: ct.IndexedWithInt64,
    SORTED_X: ct.IndexedWithInt64,
    SORTED_X_SCALE: ct.IndexedWithInt64,
    W: ct.IndexedWithInt64,
    W_SCALE: ct.IndexedWithInt64,
    W_GLOBAL_SCALE: ct.IndexedWithInt64,
    SORTED_SLOTS,
    BLOCK_EXPERT,
    NUM_POST_PAD,
    OUT: ct.IndexedWithInt64,
    top_k,
    grid_m,
    activation_scale_groups,
    global_scale_shards,
    global_scale_shard_width,
    K_IN: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
    SHARDED_GLOBAL_SCALE: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
):
    _grouped_gemm_w4a4_impl(
        X,
        X_SCALE,
        SORTED_X,
        SORTED_X_SCALE,
        W,
        W_SCALE,
        W_GLOBAL_SCALE,
        SORTED_SLOTS,
        BLOCK_EXPERT,
        NUM_POST_PAD,
        OUT,
        top_k,
        grid_m,
        activation_scale_groups,
        global_scale_shards,
        global_scale_shard_width,
        K_IN,
        TILE_M,
        TILE_N,
        TILE_K,
        SCALE_ROW_MAJOR,
        SHARDED_GLOBAL_SCALE,
        INPUT_SORTED,
        OUTPUT_SORTED,
        True,
    )


@ct.function
def _grouped_gemm1_w4a4_fused_impl(
    X,
    X_SCALE,
    SORTED_X,
    SORTED_X_SCALE,
    W,
    W_SCALE,
    W_GLOBAL_SCALE,
    SORTED_SLOTS,
    BLOCK_EXPERT,
    NUM_POST_PAD,
    OUT_Q,
    OUT_SCALE,
    top_k,
    grid_m,
    activation_scale_groups,
    global_scale_shards,
    global_scale_shard_width,
    K_IN: ConstInt,
    INTERMEDIATE_SIZE: ConstInt,
    ACTIVATION_TYPE: ConstInt,
    ACTIVATION_PARAM1: ConstFloat,
    ACTIVATION_PARAM2: ConstFloat,
    ACTIVATION_PARAM3: ConstFloat,
    IS_GATED: ConstInt,
    TILE_M: ConstInt,
    TILE_I: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
    USE_INT64: ConstBool,
):
    initial_m_block = ct.bid(0)
    i_block = ct.bid(1)
    num_post_pad = ct.gather(
        NUM_POST_PAD, ct.zeros((1,), dtype=ct.int32), padding_value=0
    ).item()
    num_live_blocks = (num_post_pad + TILE_M - 1) // TILE_M
    num_iterations = (num_live_blocks - initial_m_block + grid_m - 1) // grid_m
    num_k_tiles = (K_IN + TILE_K - 1) // TILE_K
    num_i_blocks = (INTERMEDIATE_SIZE + TILE_I - 1) // TILE_I
    groups_per_output_tile = TILE_I // _NVFP4_BLOCK_SIZE
    packed_offsets = i_block * (TILE_I // 2) + ct.arange(TILE_I // 2, dtype=ct.int32)
    scale_offsets = i_block * groups_per_output_tile + ct.arange(
        groups_per_output_tile, dtype=ct.int32
    )

    for iteration in range(num_iterations):
        m_block = initial_m_block + iteration * grid_m
        if m_block * TILE_M < num_post_pad:
            m_offsets = m_block * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
            slots = ct.gather(SORTED_SLOTS, (m_offsets,), padding_value=0)
            expert = ct.gather(
                BLOCK_EXPERT,
                ct.full((1,), m_block, dtype=ct.int32),
                padding_value=0,
            ).item()
            gate_weight_block = i_block
            gate_shard = gate_weight_block * TILE_I // global_scale_shard_width
            gate_alpha = ct.gather(
                W_GLOBAL_SCALE,
                ct.full(
                    (1,),
                    expert * global_scale_shards + gate_shard,
                    dtype=ct.int32,
                ),
                padding_value=1.0,
            ).item()
            if IS_GATED:
                up_weight_block = i_block + num_i_blocks
                up_shard = up_weight_block * TILE_I // global_scale_shard_width
                up_alpha = ct.gather(
                    W_GLOBAL_SCALE,
                    ct.full(
                        (1,),
                        expert * global_scale_shards + up_shard,
                        dtype=ct.int32,
                    ),
                    padding_value=1.0,
                ).item()

            rows = slots // top_k
            if USE_INT64:
                rows = ct.astype(rows, ct.int64)
            gate_accumulator = ct.zeros((TILE_M, TILE_I), dtype=ct.float32)
            if IS_GATED:
                up_accumulator = ct.zeros((TILE_M, TILE_I), dtype=ct.float32)
            for k_tile in range(num_k_tiles):
                if INPUT_SORTED:
                    activation, activation_scale = (
                        _load_w4a4_contiguous_activation_tile(
                            SORTED_X,
                            SORTED_X_SCALE,
                            m_block,
                            k_tile,
                            TILE_M,
                            TILE_K,
                        )
                    )
                else:
                    activation, activation_scale = _load_w4a4_activation_tile(
                        X,
                        X_SCALE,
                        rows,
                        k_tile,
                        activation_scale_groups,
                        K_IN,
                        TILE_M,
                        TILE_K,
                        SCALE_ROW_MAJOR,
                    )
                gate_weight, gate_weight_scale = _load_w4a4_weight_tile(
                    W,
                    W_SCALE,
                    expert,
                    gate_weight_block,
                    k_tile,
                    TILE_I,
                    TILE_K,
                )
                gate_accumulator = ct.mma_scaled(
                    activation,
                    activation_scale,
                    ct.permute(gate_weight, (1, 0)),
                    gate_weight_scale,
                    gate_accumulator,
                )
                if IS_GATED:
                    up_weight, up_weight_scale = _load_w4a4_weight_tile(
                        W,
                        W_SCALE,
                        expert,
                        up_weight_block,
                        k_tile,
                        TILE_I,
                        TILE_K,
                    )
                    up_accumulator = ct.mma_scaled(
                        activation,
                        activation_scale,
                        ct.permute(up_weight, (1, 0)),
                        up_weight_scale,
                        up_accumulator,
                    )

            # Match the standalone BF16 GEMM precision boundary so fused and
            # unfused activation paths remain bitwise equivalent.
            values = ct.astype(
                ct.astype(gate_accumulator * gate_alpha, ct.bfloat16),
                ct.float32,
            )
            if IS_GATED:
                up = ct.astype(
                    ct.astype(up_accumulator * up_alpha, ct.bfloat16),
                    ct.float32,
                )
                values = _apply_gated_activation(
                    values,
                    up,
                    ACTIVATION_TYPE,
                    ACTIVATION_PARAM1,
                    ACTIVATION_PARAM2,
                    ACTIVATION_PARAM3,
                )
            else:
                values = _apply_ungated_activation(
                    values,
                    ACTIVATION_TYPE,
                    ACTIVATION_PARAM1,
                    ACTIVATION_PARAM2,
                    ACTIVATION_PARAM3,
                )
            groups = ct.reshape(
                ct.astype(ct.astype(values, ct.bfloat16), ct.float32),
                (TILE_M, groups_per_output_tile, _NVFP4_BLOCK_SIZE),
            )
            packed, scale = _encode_nvfp4_groups(groups, TILE_M, TILE_I)
            output_rows = m_offsets if OUTPUT_SORTED else slots
            ct.scatter(
                OUT_Q,
                (
                    ct.reshape(output_rows, (TILE_M, 1)),
                    ct.reshape(packed_offsets, (1, TILE_I // 2)),
                ),
                packed,
                check_bounds=True,
            )
            if SCALE_ROW_MAJOR:
                ct.scatter(
                    OUT_SCALE,
                    (
                        ct.reshape(output_rows, (TILE_M, 1)),
                        ct.reshape(scale_offsets, (1, groups_per_output_tile)),
                    ),
                    scale,
                    check_bounds=True,
                )
            else:
                ct.scatter(
                    OUT_SCALE,
                    (
                        ct.reshape(scale_offsets, (1, groups_per_output_tile)),
                        ct.reshape(output_rows, (TILE_M, 1)),
                    ),
                    scale,
                    check_bounds=True,
                )


@ct.kernel
def _grouped_gemm1_w4a4_fused(
    X,
    X_SCALE,
    SORTED_X,
    SORTED_X_SCALE,
    W,
    W_SCALE,
    W_GLOBAL_SCALE,
    SORTED_SLOTS,
    BLOCK_EXPERT,
    NUM_POST_PAD,
    OUT_Q,
    OUT_SCALE,
    top_k,
    grid_m,
    activation_scale_groups,
    global_scale_shards,
    global_scale_shard_width,
    K_IN: ConstInt,
    INTERMEDIATE_SIZE: ConstInt,
    ACTIVATION_TYPE: ConstInt,
    ACTIVATION_PARAM1: ConstFloat,
    ACTIVATION_PARAM2: ConstFloat,
    ACTIVATION_PARAM3: ConstFloat,
    IS_GATED: ConstInt,
    TILE_M: ConstInt,
    TILE_I: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
):
    _grouped_gemm1_w4a4_fused_impl(
        X,
        X_SCALE,
        SORTED_X,
        SORTED_X_SCALE,
        W,
        W_SCALE,
        W_GLOBAL_SCALE,
        SORTED_SLOTS,
        BLOCK_EXPERT,
        NUM_POST_PAD,
        OUT_Q,
        OUT_SCALE,
        top_k,
        grid_m,
        activation_scale_groups,
        global_scale_shards,
        global_scale_shard_width,
        K_IN,
        INTERMEDIATE_SIZE,
        ACTIVATION_TYPE,
        ACTIVATION_PARAM1,
        ACTIVATION_PARAM2,
        ACTIVATION_PARAM3,
        IS_GATED,
        TILE_M,
        TILE_I,
        TILE_K,
        SCALE_ROW_MAJOR,
        INPUT_SORTED,
        OUTPUT_SORTED,
        False,
    )


@ct.kernel
def _grouped_gemm1_w4a4_fused_i64(
    X: ct.IndexedWithInt64,
    X_SCALE: ct.IndexedWithInt64,
    SORTED_X: ct.IndexedWithInt64,
    SORTED_X_SCALE: ct.IndexedWithInt64,
    W: ct.IndexedWithInt64,
    W_SCALE: ct.IndexedWithInt64,
    W_GLOBAL_SCALE: ct.IndexedWithInt64,
    SORTED_SLOTS,
    BLOCK_EXPERT,
    NUM_POST_PAD,
    OUT_Q: ct.IndexedWithInt64,
    OUT_SCALE: ct.IndexedWithInt64,
    top_k,
    grid_m,
    activation_scale_groups,
    global_scale_shards,
    global_scale_shard_width,
    K_IN: ConstInt,
    INTERMEDIATE_SIZE: ConstInt,
    ACTIVATION_TYPE: ConstInt,
    ACTIVATION_PARAM1: ConstFloat,
    ACTIVATION_PARAM2: ConstFloat,
    ACTIVATION_PARAM3: ConstFloat,
    IS_GATED: ConstInt,
    TILE_M: ConstInt,
    TILE_I: ConstInt,
    TILE_K: ConstInt,
    SCALE_ROW_MAJOR: ConstBool,
    INPUT_SORTED: ConstBool,
    OUTPUT_SORTED: ConstBool,
):
    _grouped_gemm1_w4a4_fused_impl(
        X,
        X_SCALE,
        SORTED_X,
        SORTED_X_SCALE,
        W,
        W_SCALE,
        W_GLOBAL_SCALE,
        SORTED_SLOTS,
        BLOCK_EXPERT,
        NUM_POST_PAD,
        OUT_Q,
        OUT_SCALE,
        top_k,
        grid_m,
        activation_scale_groups,
        global_scale_shards,
        global_scale_shard_width,
        K_IN,
        INTERMEDIATE_SIZE,
        ACTIVATION_TYPE,
        ACTIVATION_PARAM1,
        ACTIVATION_PARAM2,
        ACTIVATION_PARAM3,
        IS_GATED,
        TILE_M,
        TILE_I,
        TILE_K,
        SCALE_ROW_MAJOR,
        INPUT_SORTED,
        OUTPUT_SORTED,
        True,
    )


def _grouped_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weights: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    sorted_slots: torch.Tensor,
    block_expert: torch.Tensor,
    num_post_pad: torch.Tensor,
    output: torch.Tensor,
    *,
    top_k: int,
    block_size: int,
    config: GemmConfig,
    scale_row_major: bool = False,
    sorted_x: torch.Tensor | None = None,
    sorted_x_scale: torch.Tensor | None = None,
    output_sorted: bool = False,
) -> None:
    if config.tile_n % 128 != 0 or config.tile_k % 64 != 0:
        raise ValueError("W4A4 tiles require tile_n divisible by 128 and tile_k by 64.")
    n = output.shape[1]
    k = x.shape[1] * 2
    num_assignment_rows = output.shape[0]
    m_blocks = (sorted_slots.shape[0] + block_size - 1) // block_size
    grid_m = max(1, min(m_blocks, num_assignment_rows))
    global_scale_shards = (
        int(weight_global_scale.shape[1]) if weight_global_scale.ndim == 2 else 1
    )
    if n % global_scale_shards != 0:
        raise ValueError(
            f"GEMM output size {n} is not divisible by {global_scale_shards} global-scale shards."
        )
    if k <= _PERSISTENT_MAX_K and num_assignment_rows >= _PERSISTENT_MIN_ASSIGNMENTS:
        n_blocks = (n + config.tile_n - 1) // config.tile_n
        grid_m = _persistent_grid_m(x, grid_m, n_blocks)
    launch_args: tuple[object, ...] = (
        x.reshape(-1),
        x_scale.reshape(-1),
    )
    input_sorted = sorted_x is not None
    if input_sorted != (sorted_x_scale is not None):
        raise ValueError("sorted W4A4 values and scales must be provided together.")
    indexed_tensors = [
        x,
        x_scale,
        weights,
        weight_scale,
        weight_global_scale,
        output,
    ]
    if sorted_x is not None and sorted_x_scale is not None:
        indexed_tensors.extend((sorted_x, sorted_x_scale))
    base_kernel = (
        _grouped_gemm_w4a4_i64
        if needs_int64_indexing(*indexed_tensors)
        else _grouped_gemm_w4a4
    )
    kernel = cached_replace_hints(base_kernel, occupancy=config.occupancy)
    launch_args += (
        sorted_x if sorted_x is not None else x,
        sorted_x_scale if sorted_x_scale is not None else x_scale,
    )
    launch_args += (
        weights,
        weight_scale,
        weight_global_scale.reshape(-1),
        sorted_slots,
        block_expert,
        num_post_pad,
        output,
    )
    launch_args += (top_k,)
    launch_args += (
        grid_m,
        x_scale.shape[1],
        global_scale_shards,
        n // global_scale_shards,
        k,
        block_size,
        config.tile_n,
        config.tile_k,
    )
    launch_args += (
        scale_row_major,
        weight_global_scale.ndim == 2,
        input_sorted,
        output_sorted,
    )
    ct.launch(
        torch.cuda.current_stream(x.device),
        (grid_m, (n + config.tile_n - 1) // config.tile_n),
        kernel,
        launch_args,
    )


def _can_fuse_gemm1(
    intermediate_size: int,
    activation: ActivationConfig,
    config: GemmConfig,
) -> bool:
    activation = _validate_activation(activation)
    tile_i = config.tile_n // 2 if activation.is_gated else config.tile_n
    return (
        tile_i >= 128
        and intermediate_size % 64 == 0
        and (not activation.is_gated or intermediate_size % tile_i == 0)
    )


def _grouped_gemm1_fused(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weights: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    sorted_slots: torch.Tensor,
    block_expert: torch.Tensor,
    num_post_pad: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    *,
    num_assignments: int,
    top_k: int,
    intermediate_size: int,
    activation: ActivationConfig,
    block_size: int,
    config: GemmConfig,
    scale_row_major: bool,
    sorted_x: torch.Tensor | None = None,
    sorted_x_scale: torch.Tensor | None = None,
    output_sorted: bool = False,
) -> None:
    activation = _validate_activation(activation)
    tile_i = config.tile_n // 2 if activation.is_gated else config.tile_n
    if not _can_fuse_gemm1(intermediate_size, activation, config):
        raise ValueError(
            "unsupported fused W4A4 GEMM1 activation tile: "
            f"intermediate_size={intermediate_size}, activation={activation.type!r}, "
            f"tile_i={tile_i}."
        )
    k = x.shape[1] * 2
    m_blocks = (sorted_slots.shape[0] + block_size - 1) // block_size
    grid_m = max(1, min(m_blocks, num_assignments))
    num_i_blocks = (intermediate_size + tile_i - 1) // tile_i
    # Unlike the non-fused kernel, the fused activation/quantization epilogue
    # keeps this schedule profitable beyond _PERSISTENT_MAX_K.
    if num_assignments >= _PERSISTENT_MIN_ASSIGNMENTS:
        grid_m = _persistent_grid_m(x, grid_m, num_i_blocks)
    global_scale_shards = (
        int(weight_global_scale.shape[1]) if weight_global_scale.ndim == 2 else 1
    )
    physical_n = intermediate_size * (2 if activation.is_gated else 1)
    if physical_n % global_scale_shards != 0:
        raise ValueError(
            f"GEMM1 output size {physical_n} is not divisible by "
            f"{global_scale_shards} global-scale shards."
        )
    input_sorted = sorted_x is not None
    if input_sorted != (sorted_x_scale is not None):
        raise ValueError("sorted W4A4 values and scales must be provided together.")
    indexed_tensors = [
        x,
        x_scale,
        weights,
        weight_scale,
        weight_global_scale,
        output_q,
        output_scale,
    ]
    if sorted_x is not None and sorted_x_scale is not None:
        indexed_tensors.extend((sorted_x, sorted_x_scale))
    base_kernel = (
        _grouped_gemm1_w4a4_fused_i64
        if needs_int64_indexing(*indexed_tensors)
        else _grouped_gemm1_w4a4_fused
    )
    kernel = cached_replace_hints(base_kernel, occupancy=config.occupancy)
    ct.launch(
        torch.cuda.current_stream(x.device),
        (grid_m, (intermediate_size + tile_i - 1) // tile_i),
        kernel,
        (
            x.reshape(-1),
            x_scale.reshape(-1),
            sorted_x if sorted_x is not None else x,
            sorted_x_scale if sorted_x_scale is not None else x_scale,
            weights,
            weight_scale,
            weight_global_scale.reshape(-1),
            sorted_slots,
            block_expert,
            num_post_pad,
            output_q,
            output_scale,
            top_k,
            grid_m,
            x_scale.shape[1],
            global_scale_shards,
            physical_n // global_scale_shards,
            k,
            intermediate_size,
            *_activation_kernel_args(activation),
            int(activation.is_gated),
            block_size,
            tile_i,
            config.tile_k,
            scale_row_major,
            input_sorted,
            output_sorted,
        ),
    )


def run_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w1_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    output: torch.Tensor,
    workspace: Workspace,
    *,
    activation: ActivationConfig,
    fuse_gemm1: bool | None = None,
    num_sms: int,
    block_size: int,
    gemm1_config: GemmConfig,
    gemm2_config: GemmConfig,
) -> torch.Tensor:
    """Run the pre-routed W4A4 NVFP4 MoE pipeline."""
    activation = _validate_activation(activation)
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_assignments = num_tokens * top_k
    sorted_slots, block_expert, num_post_pad = _permute(
        topk_ids, w1.shape[0], block_size, workspace
    )

    _quantize(
        hidden_states,
        workspace.input_q,
        workspace.input_scale,
        scale_row_major=workspace.scale_row_major,
    )
    gemm1_input = workspace.input_q
    gemm1_input_scale = workspace.input_scale

    sorted_gemm1_input = None
    sorted_gemm1_input_scale = None
    use_sorted_io = _use_row_major_scale_layout(
        num_assignments, workspace.activation_q.shape[1] * 2
    )
    if use_sorted_io:
        if workspace.sorted_input_q is None or workspace.sorted_input_scale is None:
            raise RuntimeError("W4A4 workspace is missing sorted input buffers.")
        _pack_input(
            gemm1_input,
            gemm1_input_scale,
            sorted_slots,
            workspace.sorted_input_q,
            workspace.sorted_input_scale,
            top_k=top_k,
            block_size=block_size,
            scale_row_major=workspace.scale_row_major,
        )
        sorted_gemm1_input = workspace.sorted_input_q
        sorted_gemm1_input_scale = workspace.sorted_input_scale

    can_fuse_gemm1 = _can_fuse_gemm1(
        workspace.activation_q.shape[1] * 2,
        activation,
        gemm1_config,
    )
    auto_fuse_gemm1 = (
        not activation.is_gated
        or num_assignments >= _SEPARATE_ACTIVATION_MAX_ASSIGNMENTS
        or workspace.activation_q.shape[1] * 2 % gemm1_config.tile_n != 0
    ) and can_fuse_gemm1
    use_fused_gemm1 = auto_fuse_gemm1 if fuse_gemm1 is None else fuse_gemm1
    if use_fused_gemm1 and not can_fuse_gemm1:
        raise ValueError(
            "the selected tactic cannot fuse GEMM1 for "
            f"intermediate_size={workspace.activation_q.shape[1] * 2} and "
            f"tile_n={gemm1_config.tile_n}."
        )
    if use_fused_gemm1:
        intermediate_size = workspace.activation_q.shape[1] * 2
        _grouped_gemm1_fused(
            gemm1_input,
            gemm1_input_scale,
            w1,
            w1_scale,
            w1_global_scale,
            sorted_slots,
            block_expert,
            num_post_pad,
            workspace.activation_q,
            workspace.activation_scale,
            num_assignments=num_assignments,
            top_k=top_k,
            intermediate_size=intermediate_size,
            activation=activation,
            block_size=block_size,
            config=gemm1_config,
            scale_row_major=workspace.scale_row_major,
            sorted_x=sorted_gemm1_input,
            sorted_x_scale=sorted_gemm1_input_scale,
            output_sorted=use_sorted_io,
        )
    else:
        gemm1_rows = sorted_slots.shape[0] if use_sorted_io else num_assignments
        gemm1_out = workspace.gemm1_out[:gemm1_rows]
        _grouped_gemm(
            gemm1_input,
            gemm1_input_scale,
            w1,
            w1_scale,
            w1_global_scale,
            sorted_slots,
            block_expert,
            num_post_pad,
            gemm1_out,
            top_k=top_k,
            block_size=block_size,
            config=gemm1_config,
            scale_row_major=workspace.scale_row_major,
            sorted_x=sorted_gemm1_input,
            sorted_x_scale=sorted_gemm1_input_scale,
            output_sorted=use_sorted_io,
        )
    if not use_fused_gemm1:
        _launch_unfused_activation_quantize(
            gemm1_out,
            workspace,
            activation,
            num_assignments=num_assignments,
            num_sms=num_sms,
        )
    gemm2_input = workspace.activation_q
    gemm2_input_scale = workspace.activation_scale

    gemm2_out = workspace.gemm2_out[:num_assignments]
    _grouped_gemm(
        gemm2_input,
        gemm2_input_scale,
        w2,
        w2_scale,
        w2_global_scale,
        sorted_slots,
        block_expert,
        num_post_pad,
        gemm2_out,
        top_k=1,
        block_size=block_size,
        config=gemm2_config,
        scale_row_major=workspace.scale_row_major,
        sorted_x=gemm2_input if use_sorted_io else None,
        sorted_x_scale=gemm2_input_scale if use_sorted_io else None,
    )
    tile_h = _combine_tile_h(num_tokens, hidden_size)
    ct.launch(
        torch.cuda.current_stream(hidden_states.device),
        (num_tokens, (hidden_size + tile_h - 1) // tile_h),
        (_combine_i64 if needs_int64_indexing(gemm2_out, output) else _combine),
        (
            gemm2_out.reshape(-1),
            topk_weights.reshape(-1),
            output.view(-1),
            top_k,
            hidden_size,
            tile_h,
        ),
    )
    return output


__all__ = [
    "GemmConfig",
    "Workspace",
    "allocate_workspace",
    "run_moe",
]
