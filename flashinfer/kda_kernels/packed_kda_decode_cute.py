"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Experimental CuTe DSL backend for serving-native packed Kimi K3 decode.

This is the packed-input T=1 counterpart of the wide-vector GDN decode kernel.
It applies a distinct per-key-dimension decay to every state column.
"""

import functools
from pathlib import Path
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass.utils import SmemAllocator
import tvm_ffi  # noqa: F401 -- TVM FFI is required for kernel dispatch

from ..jit.cpp_ext import is_cuda_version_at_least
from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from ..utils import get_compute_capability

F32 = cutlass.Float32
BF16 = cutlass.BFloat16

_HEADS = 12
_HEAD_DIM = 128
_MIXED_WIDTH = 3 * _HEADS * _HEAD_DIM
_GATE_WIDTH = _HEADS * _HEAD_DIM
_NUM_THREADS = 128
_LANES_PER_ROW = 16
_NUM_GROUPS = _NUM_THREADS // _LANES_PER_ROW
_ELEMS_PER_LANE = _HEAD_DIM // _LANES_PER_ROW
_ILP_ROWS = 4
_Q_SCALE = _HEAD_DIM**-0.5
_L2_EPS = 1.0e-6
_LOWER_BOUND = -5.0
_SUPPORTED_TILE_V = (8, 16, 32, 64, 128)
_CUTE_DSL_MODULE = "packed_kda_decode"
_SOURCE_FILES = (str(Path(__file__).resolve()),)


def _sigmoid(value):
    return cute.rcp(cute.exp(-value, fastmath=True) + 1.0, approx=True, ftz=True)


def _select_tile_v(batch: int) -> int:
    """Select the one-warp tile matching the packed serving workload."""
    return 8 if batch < 26 else 16


@cute.kernel
def _packed_kda_decode_warp_kernel(
    mixed_qkv,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state,
    state_indices,
    output,
    tile_v: cutlass.Constexpr,
    num_v_tiles: cutlass.Constexpr,
    use_aligned_io: cutlass.Constexpr,
):
    """One warp owns a row tile without CTA barriers."""
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, row_idx, _ = cute.arch.block_idx()
    lane_idx = thread_idx % 32
    k_lane = lane_idx % _LANES_PER_ROW
    v_lane = lane_idx // _LANES_PER_ROW

    value_tile_idx = block_idx % num_v_tiles
    head_idx = block_idx // num_v_tiles
    value_tile_base = value_tile_idx * tile_v

    requested_slot = state_indices[row_idx]
    is_live = requested_slot >= 0
    if is_live:
        source_width: cutlass.Constexpr = _HEAD_DIM // 32
        source_start = lane_idx * source_width
        query_bf16 = cute.make_rmem_tensor((source_width,), BF16)
        key_bf16 = cute.make_rmem_tensor((source_width,), BF16)
        gate_bf16 = cute.make_rmem_tensor((source_width,), BF16)
        dt_bias_f32 = cute.make_rmem_tensor((source_width,), F32)
        query_source = cute.make_rmem_tensor((source_width,), F32)
        key_source = cute.make_rmem_tensor((source_width,), F32)
        decay_source = cute.make_rmem_tensor((source_width,), F32)

        query_head = mixed_qkv[(None, 0, head_idx, row_idx)]
        key_head = mixed_qkv[(None, 1, head_idx, row_idx)]
        gate_head = raw_gate[(None, head_idx, row_idx)]
        query_tile = cute.local_tile(query_head, (source_width,), (lane_idx,))
        key_tile = cute.local_tile(key_head, (source_width,), (lane_idx,))
        gate_tile = cute.local_tile(gate_head, (source_width,), (lane_idx,))
        if cutlass.const_expr(use_aligned_io):
            # Preserve the packed source layout and lane*4 channel ownership,
            # but express each lane's contiguous slice as one wide transaction.
            bf16x4_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BF16, num_bits_per_copy=64
            )
            cute.copy(bf16x4_copy, query_tile, query_bf16)
            cute.copy(bf16x4_copy, key_tile, key_bf16)
            cute.copy(bf16x4_copy, gate_tile, gate_bf16)
            if cutlass.const_expr(tile_v == 8):
                # At tile8 the lower instruction count wins; at tile16 this
                # wide dependency increases long-scoreboard stalls, so its
                # bias loads remain scalar and overlap the decay arithmetic.
                dt_bias_head = dt_bias[(None, head_idx)]
                dt_bias_tile = cute.local_tile(
                    dt_bias_head, (source_width,), (lane_idx,)
                )
                f32x4_copy = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), F32, num_bits_per_copy=128
                )
                cute.copy(f32x4_copy, dt_bias_tile, dt_bias_f32)
        else:
            for elem in cutlass.range_constexpr(source_width):
                query_bf16[elem] = query_tile[elem]
                key_bf16[elem] = key_tile[elem]
                gate_bf16[elem] = gate_tile[elem]

        value_loaded = F32(0.0)
        if cutlass.const_expr(tile_v <= 16):
            if lane_idx < tile_v:
                value_loaded = mixed_qkv[
                    (value_tile_base + lane_idx, 2, head_idx, row_idx)
                ].to(F32)

        rows_per_half: cutlass.Constexpr = 4
        row_blocks: cutlass.Constexpr = tile_v // (2 * rows_per_half)
        state_head = state[(cutlass.Int64(requested_slot), head_idx, None, None)]
        state_bf16 = cute.make_rmem_tensor(
            cute.make_layout(
                (rows_per_half, _ELEMS_PER_LANE),
                stride=(_ELEMS_PER_LANE, 1),
            ),
            BF16,
        )
        for local_row in cutlass.range_constexpr(rows_per_half):
            value_idx = value_tile_base + v_lane + 2 * local_row
            state_tile = cute.local_tile(
                state_head, (1, _ELEMS_PER_LANE), (value_idx, k_lane)
            )
            if cutlass.const_expr(use_aligned_io):
                cute.autovec_copy(state_tile, state_bf16[(local_row, None)])
            else:
                for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                    state_bf16[(local_row, elem)] = state_tile[(0, elem)]

        A = F32(0.0)
        beta = F32(0.0)
        if lane_idx == 0:
            A = cute.exp(A_log[head_idx], fastmath=True)
            beta = _sigmoid(raw_beta[(head_idx, row_idx)].to(F32))
        A = cute.arch.shuffle_sync(A, offset=0, mask=0xFFFFFFFF)
        beta = cute.arch.shuffle_sync(beta, offset=0, mask=0xFFFFFFFF)

        for elem in cutlass.range_constexpr(source_width):
            channel = source_start + elem
            query_source[elem] = query_bf16[elem].to(F32)
            key_source[elem] = key_bf16[elem].to(F32)
            if cutlass.const_expr(use_aligned_io and tile_v == 8):
                gate = gate_bf16[elem].to(F32) + dt_bias_f32[elem]
            else:
                gate = gate_bf16[elem].to(F32) + dt_bias[(channel, head_idx)]
            decay_source[elem] = cute.exp(
                _LOWER_BOUND * _sigmoid(A * gate), fastmath=True
            )
        query_norm_even = query_source[0] * query_source[0]
        query_norm_odd = query_source[1] * query_source[1]
        query_norm_even += query_source[2] * query_source[2]
        query_norm_odd += query_source[3] * query_source[3]
        query_norm = query_norm_even + query_norm_odd
        key_norm_even = key_source[0] * key_source[0]
        key_norm_odd = key_source[1] * key_source[1]
        key_norm_even += key_source[2] * key_source[2]
        key_norm_odd += key_source[3] * key_source[3]
        key_norm = key_norm_even + key_norm_odd
        for offset in (16, 8, 4, 2, 1):
            query_norm += cute.arch.shuffle_sync_bfly(
                query_norm, offset=offset, mask=0xFFFFFFFF
            )
            key_norm += cute.arch.shuffle_sync_bfly(
                key_norm, offset=offset, mask=0xFFFFFFFF
            )
        query_scale = cute.rsqrt(query_norm + _L2_EPS, fastmath=True) * _Q_SCALE
        key_scale = cute.rsqrt(key_norm + _L2_EPS, fastmath=True)

        query = cute.make_rmem_tensor((_ELEMS_PER_LANE,), F32)
        key = cute.make_rmem_tensor((_ELEMS_PER_LANE,), F32)
        decay = cute.make_rmem_tensor((_ELEMS_PER_LANE,), F32)
        for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
            source_lane = 2 * k_lane + elem // source_width
            source_elem = elem % source_width
            query[elem] = (
                cute.arch.shuffle_sync(
                    query_source[source_elem],
                    offset=source_lane,
                    mask=0xFFFFFFFF,
                )
                * query_scale
            )
            key[elem] = (
                cute.arch.shuffle_sync(
                    key_source[source_elem],
                    offset=source_lane,
                    mask=0xFFFFFFFF,
                )
                * key_scale
            )
            decay[elem] = cute.arch.shuffle_sync(
                decay_source[source_elem],
                offset=source_lane,
                mask=0xFFFFFFFF,
            )

        for row_block in cutlass.range_constexpr(row_blocks):
            if cutlass.const_expr(row_block > 0):
                for local_row in cutlass.range_constexpr(rows_per_half):
                    value_idx = (
                        value_tile_base
                        + v_lane
                        + 2 * (row_block * rows_per_half + local_row)
                    )
                    state_tile = cute.local_tile(
                        state_head, (1, _ELEMS_PER_LANE), (value_idx, k_lane)
                    )
                    if cutlass.const_expr(use_aligned_io):
                        cute.autovec_copy(state_tile, state_bf16[(local_row, None)])
                    else:
                        for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                            state_bf16[(local_row, elem)] = state_tile[(0, elem)]
            # Pair independent rows to hide BF16 conversion, FFMA, shuffle, and
            # state-memory latency behind a second dependency chain.
            for row_pair in cutlass.range_constexpr(rows_per_half // 2):
                local_row0 = 2 * row_pair
                local_row1 = local_row0 + 1
                value_idx0 = (
                    value_tile_base
                    + v_lane
                    + 2 * (row_block * rows_per_half + local_row0)
                )
                value_idx1 = (
                    value_tile_base
                    + v_lane
                    + 2 * (row_block * rows_per_half + local_row1)
                )
                prediction0_even = (
                    state_bf16[(local_row0, 0)].to(F32) * decay[0] * key[0]
                )
                prediction0_odd = (
                    state_bf16[(local_row0, 1)].to(F32) * decay[1] * key[1]
                )
                prediction1_even = (
                    state_bf16[(local_row1, 0)].to(F32) * decay[0] * key[0]
                )
                prediction1_odd = (
                    state_bf16[(local_row1, 1)].to(F32) * decay[1] * key[1]
                )
                for pair in cutlass.range_constexpr(1, 4):
                    elem = 2 * pair
                    prediction0_even += (
                        state_bf16[(local_row0, elem)].to(F32) * decay[elem] * key[elem]
                    )
                    prediction0_odd += (
                        state_bf16[(local_row0, elem + 1)].to(F32)
                        * decay[elem + 1]
                        * key[elem + 1]
                    )
                    prediction1_even += (
                        state_bf16[(local_row1, elem)].to(F32) * decay[elem] * key[elem]
                    )
                    prediction1_odd += (
                        state_bf16[(local_row1, elem + 1)].to(F32)
                        * decay[elem + 1]
                        * key[elem + 1]
                    )
                prediction0 = prediction0_even + prediction0_odd
                prediction1 = prediction1_even + prediction1_odd
                for offset in (8, 4, 2, 1):
                    prediction0 += cute.arch.shuffle_sync_bfly(
                        prediction0, offset=offset, mask=0xFFFFFFFF
                    )
                    prediction1 += cute.arch.shuffle_sync_bfly(
                        prediction1, offset=offset, mask=0xFFFFFFFF
                    )

                value0 = F32(0.0)
                value1 = F32(0.0)
                if cutlass.const_expr(tile_v <= 16):
                    value0 = cute.arch.shuffle_sync(
                        value_loaded,
                        offset=value_idx0 - value_tile_base,
                        mask=0xFFFFFFFF,
                    )
                    value1 = cute.arch.shuffle_sync(
                        value_loaded,
                        offset=value_idx1 - value_tile_base,
                        mask=0xFFFFFFFF,
                    )
                else:
                    if k_lane == 0:
                        value0 = mixed_qkv[(value_idx0, 2, head_idx, row_idx)].to(F32)
                        value1 = mixed_qkv[(value_idx1, 2, head_idx, row_idx)].to(F32)
                    value0 = cute.arch.shuffle_sync(
                        value0, offset=v_lane * _LANES_PER_ROW, mask=0xFFFFFFFF
                    )
                    value1 = cute.arch.shuffle_sync(
                        value1, offset=v_lane * _LANES_PER_ROW, mask=0xFFFFFFFF
                    )
                delta0 = (value0 - prediction0) * beta
                delta1 = (value1 - prediction1) * beta

                projected0_even = F32(0.0)
                projected0_odd = F32(0.0)
                projected1_even = F32(0.0)
                projected1_odd = F32(0.0)
                for pair in cutlass.range_constexpr(4):
                    elem = 2 * pair
                    updated0_even = (
                        state_bf16[(local_row0, elem)].to(F32) * decay[elem]
                        + delta0 * key[elem]
                    )
                    updated0_odd = (
                        state_bf16[(local_row0, elem + 1)].to(F32) * decay[elem + 1]
                        + delta0 * key[elem + 1]
                    )
                    updated1_even = (
                        state_bf16[(local_row1, elem)].to(F32) * decay[elem]
                        + delta1 * key[elem]
                    )
                    updated1_odd = (
                        state_bf16[(local_row1, elem + 1)].to(F32) * decay[elem + 1]
                        + delta1 * key[elem + 1]
                    )
                    state_bf16[(local_row0, elem)] = updated0_even.to(BF16)
                    state_bf16[(local_row0, elem + 1)] = updated0_odd.to(BF16)
                    state_bf16[(local_row1, elem)] = updated1_even.to(BF16)
                    state_bf16[(local_row1, elem + 1)] = updated1_odd.to(BF16)
                    projected0_even += updated0_even * query[elem]
                    projected0_odd += updated0_odd * query[elem + 1]
                    projected1_even += updated1_even * query[elem]
                    projected1_odd += updated1_odd * query[elem + 1]
                projected0 = projected0_even + projected0_odd
                projected1 = projected1_even + projected1_odd
                for offset in (8, 4, 2, 1):
                    projected0 += cute.arch.shuffle_sync_bfly(
                        projected0, offset=offset, mask=0xFFFFFFFF
                    )
                    projected1 += cute.arch.shuffle_sync_bfly(
                        projected1, offset=offset, mask=0xFFFFFFFF
                    )

                state_tile0 = cute.local_tile(
                    state_head, (1, _ELEMS_PER_LANE), (value_idx0, k_lane)
                )
                state_tile1 = cute.local_tile(
                    state_head, (1, _ELEMS_PER_LANE), (value_idx1, k_lane)
                )
                if cutlass.const_expr(use_aligned_io):
                    cute.autovec_copy(state_bf16[(local_row0, None)], state_tile0)
                    cute.autovec_copy(state_bf16[(local_row1, None)], state_tile1)
                else:
                    for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                        state_tile0[(0, elem)] = state_bf16[(local_row0, elem)]
                        state_tile1[(0, elem)] = state_bf16[(local_row1, elem)]
                if k_lane == 0:
                    output[(value_idx0, head_idx, row_idx)] = projected0.to(BF16)
                    output[(value_idx1, head_idx, row_idx)] = projected1.to(BF16)
    else:
        if k_lane == 0:
            inactive_rows_per_half: cutlass.Constexpr = 4
            inactive_row_blocks: cutlass.Constexpr = tile_v // (
                2 * inactive_rows_per_half
            )
            for row_block in cutlass.range_constexpr(inactive_row_blocks):
                for local_row in cutlass.range_constexpr(inactive_rows_per_half):
                    value_idx = (
                        value_tile_base
                        + v_lane
                        + 2 * (row_block * inactive_rows_per_half + local_row)
                    )
                    output[(value_idx, head_idx, row_idx)] = BF16(0.0)


@cute.kernel
def _packed_kda_decode_kernel(
    mixed_qkv,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state,
    state_indices,
    output,
    tile_v: cutlass.Constexpr,
    num_v_tiles: cutlass.Constexpr,
    use_aligned_io: cutlass.Constexpr,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    lane_in_warp = thread_idx % 32
    group_idx = thread_idx // _LANES_PER_ROW
    lane_in_group = thread_idx % _LANES_PER_ROW
    k_start = lane_in_group * _ELEMS_PER_LANE

    value_tile_idx = block_idx % num_v_tiles
    head_row = block_idx // num_v_tiles
    head_idx = head_row % _HEADS
    row_idx = head_row // _HEADS

    rows_per_group: cutlass.Constexpr = tile_v // _NUM_GROUPS
    iterations_per_group: cutlass.Constexpr = rows_per_group // _ILP_ROWS

    smem = SmemAllocator()
    query_smem = smem.allocate_tensor(
        F32, cute.make_layout((_HEAD_DIM,), stride=(1,)), byte_alignment=16
    )
    key_smem = smem.allocate_tensor(
        F32, cute.make_layout((_HEAD_DIM,), stride=(1,)), byte_alignment=16
    )
    decay_smem = smem.allocate_tensor(
        F32, cute.make_layout((_HEAD_DIM,), stride=(1,)), byte_alignment=16
    )
    beta_smem = smem.allocate_tensor(
        F32, cute.make_layout((4,), stride=(1,)), byte_alignment=16
    )

    requested_slot = state_indices[row_idx]
    is_live = requested_slot >= 0

    if is_live:
        # The two 16-lane halves of warp zero redundantly load the vectors.
        # Their writes are identical.  This follows the proven GDN T=1
        # subgroup reduction while leaving all eight groups available for V.
        if thread_idx < 32:
            pre_lane = lane_in_warp % _LANES_PER_ROW
            pre_start = pre_lane * _ELEMS_PER_LANE
            query_bf16 = cute.make_rmem_tensor((_ELEMS_PER_LANE,), BF16)
            key_bf16 = cute.make_rmem_tensor((_ELEMS_PER_LANE,), BF16)
            query = cute.make_rmem_tensor((_ELEMS_PER_LANE,), F32)
            key = cute.make_rmem_tensor((_ELEMS_PER_LANE,), F32)

            if cutlass.const_expr(use_aligned_io):
                cute.autovec_copy(
                    mixed_qkv[(None, pre_lane, 0, head_idx, row_idx)], query_bf16
                )
                cute.autovec_copy(
                    mixed_qkv[(None, pre_lane, 1, head_idx, row_idx)], key_bf16
                )
            else:
                for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                    query_bf16[elem] = mixed_qkv[(elem, pre_lane, 0, head_idx, row_idx)]
                    key_bf16[elem] = mixed_qkv[(elem, pre_lane, 1, head_idx, row_idx)]
            query_norm = F32(0.0)
            key_norm = F32(0.0)
            for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                query[elem] = query_bf16[elem].to(F32)
                key[elem] = key_bf16[elem].to(F32)
                query_norm += query[elem] * query[elem]
                key_norm += key[elem] * key[elem]
            for offset in (8, 4, 2, 1):
                query_norm += cute.arch.shuffle_sync_bfly(
                    query_norm, offset=offset, mask=-1, mask_and_clamp=31
                )
                key_norm += cute.arch.shuffle_sync_bfly(
                    key_norm, offset=offset, mask=-1, mask_and_clamp=31
                )
            query_scale = cute.rsqrt(query_norm + _L2_EPS, fastmath=True) * _Q_SCALE
            key_scale = cute.rsqrt(key_norm + _L2_EPS, fastmath=True)
            A = cute.exp(A_log[head_idx], fastmath=True)
            for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                channel = pre_start + elem
                query_smem[channel] = query[elem] * query_scale
                key_smem[channel] = key[elem] * key_scale
                gate = (
                    raw_gate[(elem, pre_lane, head_idx, row_idx)].to(F32)
                    + dt_bias[(elem, pre_lane, head_idx)]
                )
                decay_smem[channel] = cute.exp(
                    _LOWER_BOUND * _sigmoid(A * gate), fastmath=True
                )
            if lane_in_warp == 0:
                beta_smem[0] = _sigmoid(raw_beta[(head_idx, row_idx)].to(F32))

        cute.arch.barrier()

        query = cute.make_rmem_tensor((_ELEMS_PER_LANE,), F32)
        key = cute.make_rmem_tensor((_ELEMS_PER_LANE,), F32)
        decay = cute.make_rmem_tensor((_ELEMS_PER_LANE,), F32)
        for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
            channel = k_start + elem
            query[elem] = query_smem[channel]
            key[elem] = key_smem[channel]
            decay[elem] = decay_smem[channel]
        beta = beta_smem[0]

        state_head = state[(cutlass.Int64(requested_slot), head_idx, None, None)]
        state_rows = cute.make_rmem_tensor(
            cute.make_layout((_ILP_ROWS, _ELEMS_PER_LANE), stride=(_ELEMS_PER_LANE, 1)),
            F32,
        )
        state_bf16 = cute.make_rmem_tensor(
            cute.make_layout((_ILP_ROWS, _ELEMS_PER_LANE), stride=(_ELEMS_PER_LANE, 1)),
            BF16,
        )

        for iteration in cutlass.range_constexpr(iterations_per_group):
            value_base = (
                value_tile_idx * tile_v
                + group_idx * rows_per_group
                + iteration * _ILP_ROWS
            )
            for local_row in cutlass.range_constexpr(_ILP_ROWS):
                value_idx = value_base + local_row
                state_tile = cute.local_tile(
                    state_head, (1, _ELEMS_PER_LANE), (value_idx, lane_in_group)
                )
                if cutlass.const_expr(use_aligned_io):
                    cute.autovec_copy(state_tile, state_bf16[(local_row, None)])
                else:
                    for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                        state_bf16[(local_row, elem)] = state_tile[(0, elem)]
                for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                    state_rows[(local_row, elem)] = (
                        state_bf16[(local_row, elem)].to(F32) * decay[elem]
                    )

            prediction = cute.make_rmem_tensor((_ILP_ROWS,), F32)
            for local_row in cutlass.range_constexpr(_ILP_ROWS):
                value = F32(0.0)
                for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                    value += state_rows[(local_row, elem)] * key[elem]
                for offset in (8, 4, 2, 1):
                    value += cute.arch.shuffle_sync_bfly(
                        value, offset=offset, mask=-1, mask_and_clamp=31
                    )
                prediction[local_row] = value

            for local_row in cutlass.range_constexpr(_ILP_ROWS):
                value_idx = value_base + local_row
                value_lane = value_idx // _ELEMS_PER_LANE
                value_elem = value_idx % _ELEMS_PER_LANE
                value = mixed_qkv[(value_elem, value_lane, 2, head_idx, row_idx)].to(
                    F32
                )
                delta = (value - prediction[local_row]) * beta
                projected = F32(0.0)
                for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                    updated = state_rows[(local_row, elem)] + delta * key[elem]
                    state_rows[(local_row, elem)] = updated
                    state_bf16[(local_row, elem)] = updated.to(BF16)
                    projected += updated * query[elem]
                for offset in (8, 4, 2, 1):
                    projected += cute.arch.shuffle_sync_bfly(
                        projected, offset=offset, mask=-1, mask_and_clamp=31
                    )
                state_tile = cute.local_tile(
                    state_head, (1, _ELEMS_PER_LANE), (value_idx, lane_in_group)
                )
                if cutlass.const_expr(use_aligned_io):
                    cute.autovec_copy(state_bf16[(local_row, None)], state_tile)
                else:
                    for elem in cutlass.range_constexpr(_ELEMS_PER_LANE):
                        state_tile[(0, elem)] = state_bf16[(local_row, elem)]
                if lane_in_group == 0:
                    output[(value_idx, head_idx, row_idx)] = projected.to(BF16)
    else:
        # Inactive graph-padding rows have no state access and a zero output.
        for iteration in cutlass.range_constexpr(iterations_per_group):
            value_base = (
                value_tile_idx * tile_v
                + group_idx * rows_per_group
                + iteration * _ILP_ROWS
            )
            if lane_in_group == 0:
                for local_row in cutlass.range_constexpr(_ILP_ROWS):
                    output[(value_base + local_row, head_idx, row_idx)] = BF16(0.0)


@cute.jit
def _packed_kda_decode_warp_launch(
    mixed_qkv,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state,
    state_indices,
    output,
    stream: cuda.CUstream,
    tile_v: cutlass.Constexpr,
    use_aligned_io: cutlass.Constexpr,
):
    batch = state_indices.shape[0]
    num_v_tiles: cutlass.Constexpr = _HEAD_DIM // tile_v

    mixed_stride = mixed_qkv.stride[0]
    gate_stride = raw_gate.stride[0]
    if cutlass.const_expr(use_aligned_io):
        # The wrapper verifies the row strides.  Exposing that fact lets the
        # copy atoms lower lane*4 slices to 64-bit BF16 transactions.
        mixed_stride = cute.assume(mixed_stride, divby=_ELEMS_PER_LANE)
        gate_stride = cute.assume(gate_stride, divby=_ELEMS_PER_LANE)
    mixed_layout = cute.make_tensor(
        mixed_qkv.iterator,
        cute.make_layout(
            (_HEAD_DIM, 3, _HEADS, batch),
            stride=(1, _GATE_WIDTH, _HEAD_DIM, mixed_stride),
        ),
    )
    gate_layout = cute.make_tensor(
        raw_gate.iterator,
        cute.make_layout(
            (_HEAD_DIM, _HEADS, batch),
            stride=(1, _HEAD_DIM, gate_stride),
        ),
    )
    beta_layout = cute.make_tensor(
        raw_beta.iterator,
        cute.make_layout((_HEADS, batch), stride=(1, raw_beta.stride[0])),
    )
    dt_bias_layout = cute.make_tensor(
        dt_bias.iterator,
        cute.make_layout((_HEAD_DIM, _HEADS), stride=(1, _HEAD_DIM)),
    )
    output_layout = cute.make_tensor(
        output.iterator,
        cute.make_layout(
            (_HEAD_DIM, _HEADS, batch),
            stride=(1, _HEAD_DIM, output.stride[0]),
        ),
    )
    state_stride = state.stride[0]
    if cutlass.const_expr(use_aligned_io):
        # The Python wrapper verifies both base and slot-stride alignment.  Make
        # the dynamic stride fact visible to the vectorizer so a state row
        # slice lowers to a single 128-bit load/store instead of eight U16s.
        state_stride = cute.assume(state_stride, divby=_ELEMS_PER_LANE)
    state_layout = cute.make_tensor(
        state.iterator,
        cute.make_layout(
            state.shape,
            stride=(state_stride, _HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1),
        ),
    )

    _packed_kda_decode_warp_kernel(
        mixed_layout,
        gate_layout,
        beta_layout,
        A_log,
        dt_bias_layout,
        state_layout,
        state_indices,
        output_layout,
        tile_v,
        num_v_tiles,
        use_aligned_io,
    ).launch(
        grid=[_HEADS * num_v_tiles, batch, 1],
        block=[32, 1, 1],
        smem=0,
        stream=stream,
    )


@cute.jit
def _packed_kda_decode_launch(
    mixed_qkv,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    state,
    state_indices,
    output,
    stream: cuda.CUstream,
    tile_v: cutlass.Constexpr,
    use_aligned_io: cutlass.Constexpr,
):
    batch = state_indices.shape[0]
    num_v_tiles: cutlass.Constexpr = _HEAD_DIM // tile_v

    mixed_layout = cute.make_tensor(
        mixed_qkv.iterator,
        cute.make_layout(
            (_ELEMS_PER_LANE, _LANES_PER_ROW, 3, _HEADS, batch),
            stride=(1, _ELEMS_PER_LANE, _GATE_WIDTH, _HEAD_DIM, mixed_qkv.stride[0]),
        ),
    )
    gate_layout = cute.make_tensor(
        raw_gate.iterator,
        cute.make_layout(
            (_ELEMS_PER_LANE, _LANES_PER_ROW, _HEADS, batch),
            stride=(1, _ELEMS_PER_LANE, _HEAD_DIM, raw_gate.stride[0]),
        ),
    )
    beta_layout = cute.make_tensor(
        raw_beta.iterator,
        cute.make_layout((_HEADS, batch), stride=(1, raw_beta.stride[0])),
    )
    dt_bias_layout = cute.make_tensor(
        dt_bias.iterator,
        cute.make_layout(
            (_ELEMS_PER_LANE, _LANES_PER_ROW, _HEADS),
            stride=(1, _ELEMS_PER_LANE, _HEAD_DIM),
        ),
    )
    output_layout = cute.make_tensor(
        output.iterator,
        cute.make_layout(
            (_HEAD_DIM, _HEADS, batch),
            stride=(1, _HEAD_DIM, output.stride[0]),
        ),
    )
    state_stride = state.stride[0]
    if cutlass.const_expr(use_aligned_io):
        state_stride = cute.assume(state_stride, divby=_ELEMS_PER_LANE)
    state_layout = cute.make_tensor(
        state.iterator,
        cute.make_layout(
            state.shape,
            stride=(state_stride, _HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1),
        ),
    )

    _packed_kda_decode_kernel(
        mixed_layout,
        gate_layout,
        beta_layout,
        A_log,
        dt_bias_layout,
        state_layout,
        state_indices,
        output_layout,
        tile_v,
        num_v_tiles,
        use_aligned_io,
    ).launch(
        grid=[batch * _HEADS * num_v_tiles, 1, 1],
        block=[_NUM_THREADS, 1, 1],
        smem=4 * (3 * _HEAD_DIM + 4) + 64,
        stream=stream,
    )


def _make_compile_inputs(use_aligned_io: bool):
    batch = cute.sym_int()
    slots = cute.sym_int()

    mixed_qkv = cute.runtime.make_fake_tensor(
        BF16,
        shape=(batch, _MIXED_WIDTH),
        stride=(cute.sym_int64(), 1),
        assumed_align=16,
    )
    raw_gate = cute.runtime.make_fake_tensor(
        BF16,
        shape=(batch, _GATE_WIDTH),
        stride=(cute.sym_int64(), 1),
        assumed_align=16,
    )
    raw_beta = cute.runtime.make_fake_tensor(
        BF16,
        shape=(batch, _HEADS),
        stride=(cute.sym_int64(), 1),
        assumed_align=16,
    )
    state = cute.runtime.make_fake_tensor(
        BF16,
        shape=(slots, _HEADS, _HEAD_DIM, _HEAD_DIM),
        stride=(cute.sym_int64(), _HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1),
        assumed_align=16,
    )
    compact = cute.runtime.make_fake_compact_tensor
    return (
        mixed_qkv,
        raw_gate,
        raw_beta,
        compact(F32, (_HEADS,), assumed_align=16, stride_order=(0,)),
        compact(
            F32,
            (_GATE_WIDTH,),
            assumed_align=16 if use_aligned_io else 4,
            stride_order=(0,),
        ),
        state,
        compact(cutlass.Int32, (batch,), assumed_align=16, stride_order=(0,)),
        compact(
            BF16,
            (batch, 1, _HEADS, _HEAD_DIM),
            assumed_align=16,
            stride_order=(3, 2, 1, 0),
        ),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )


@functools.cache
def _get_compiled_kernel(tile_v: int, use_aligned_io: bool):
    io_name = "aligned" if use_aligned_io else "unaligned"
    launch = (
        _packed_kda_decode_warp_launch
        if tile_v in (8, 16, 32)
        else _packed_kda_decode_launch
    )
    return build_and_load_cute_dsl_kernel(
        _CUTE_DSL_MODULE,
        f"d128_h12_tile{tile_v}_bf16_{io_name}",
        lambda: cute.compile(
            launch,
            *_make_compile_inputs(use_aligned_io),
            tile_v,
            use_aligned_io,
            options="--enable-tvm-ffi --generate-line-info",
        ),
        extra_key_files=_SOURCE_FILES,
    )


def _check_cuda_tensor(name, tensor, dtype):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _check_b200(device: torch.device) -> None:
    capability = get_compute_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "experimental CuTe packed KDA requires exact compute capability "
            f"10.0 (B200), got {capability[0]}.{capability[1]}"
        )
    if not is_cuda_version_at_least("12.8"):
        raise RuntimeError(
            "experimental CuTe packed KDA on compute capability 10.0 requires "
            "CUDA 12.8 or newer"
        )


@torch.no_grad()
def run_packed_kda_decode_cute(
    mixed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    *,
    tile_v: Optional[int] = None,
) -> torch.Tensor:
    """Run the experimental B200 CuTe packed KDA T=1 kernel.

    The tensor and numerical contract is identical to
    :func:`flashinfer.packed_kda_decode`.  ``tile_v`` is an experimental
    benchmark override; production-style calls should leave it as ``None``.
    """
    _check_cuda_tensor("mixed_qkv", mixed_qkv, torch.bfloat16)
    _check_cuda_tensor("raw_gate", raw_gate, torch.bfloat16)
    _check_cuda_tensor("raw_beta", raw_beta, torch.bfloat16)
    _check_cuda_tensor("A_log", A_log, torch.float32)
    _check_cuda_tensor("dt_bias", dt_bias, torch.float32)
    _check_cuda_tensor("state", state, torch.bfloat16)
    _check_cuda_tensor("state_indices", state_indices, torch.int32)

    for name, tensor in (
        ("raw_gate", raw_gate),
        ("raw_beta", raw_beta),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("state", state),
        ("state_indices", state_indices),
    ):
        if tensor.device != mixed_qkv.device:
            raise ValueError(f"{name} must be on the same device as mixed_qkv")

    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != _MIXED_WIDTH:
        raise ValueError(f"mixed_qkv must have shape [B, {_MIXED_WIDTH}]")
    batch = int(mixed_qkv.shape[0])
    if batch <= 0 or batch > 65535:
        raise ValueError(f"packed KDA T=1 batch must be in [1, 65535], got {batch}")
    if mixed_qkv.stride(1) != 1 or mixed_qkv.stride(0) < _MIXED_WIDTH:
        raise ValueError("mixed_qkv must have contiguous, non-overlapping rows")
    if raw_gate.shape != (batch, _GATE_WIDTH) or raw_gate.stride(1) != 1:
        raise ValueError(f"raw_gate must have shape [B, {_GATE_WIDTH}] with stride 1")
    if raw_gate.stride(0) < _GATE_WIDTH:
        raise ValueError("raw_gate rows must not overlap")
    if raw_beta.shape != (batch, _HEADS) or raw_beta.stride(1) != 1:
        raise ValueError(f"raw_beta must have shape [B, {_HEADS}] with stride 1")
    if raw_beta.stride(0) < _HEADS:
        raise ValueError("raw_beta rows must not overlap")
    if A_log.shape != (_HEADS,) or not A_log.is_contiguous():
        raise ValueError(f"A_log must be contiguous with shape [{_HEADS}]")
    if dt_bias.shape != (_GATE_WIDTH,) or not dt_bias.is_contiguous():
        raise ValueError(f"dt_bias must be contiguous with shape [{_GATE_WIDTH}]")
    if state_indices.shape != (batch,) or not state_indices.is_contiguous():
        raise ValueError("state_indices must be contiguous with shape [B]")
    if (
        state.ndim != 4
        or state.shape[1:] != (_HEADS, _HEAD_DIM, _HEAD_DIM)
        or state.stride(0) < _HEADS * _HEAD_DIM * _HEAD_DIM
        or tuple(state.stride()[1:]) != (_HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1)
    ):
        raise ValueError(
            "state must have shape [N,12,128,128] with compact inner dimensions"
        )

    expected_output_shape = (batch, 1, _HEADS, _HEAD_DIM)
    if output is None:
        output = mixed_qkv.new_empty(expected_output_shape)
    else:
        _check_cuda_tensor("output", output, torch.bfloat16)
        if output.device != mixed_qkv.device:
            raise ValueError("output must be on the same device as mixed_qkv")
        if output.shape != expected_output_shape or not output.is_contiguous():
            raise ValueError("output must be contiguous with shape [B,1,12,128]")

    _check_b200(mixed_qkv.device)
    selected_tile_v = _select_tile_v(batch) if tile_v is None else tile_v
    if selected_tile_v not in _SUPPORTED_TILE_V:
        raise ValueError(f"tile_v must be one of {_SUPPORTED_TILE_V}")

    use_aligned_io = (
        mixed_qkv.data_ptr() % 16 == 0
        and mixed_qkv.stride(0) % _ELEMS_PER_LANE == 0
        and raw_gate.data_ptr() % 16 == 0
        and raw_gate.stride(0) % _ELEMS_PER_LANE == 0
        and dt_bias.data_ptr() % 16 == 0
        and state.data_ptr() % 16 == 0
        and state.stride(0) % _ELEMS_PER_LANE == 0
    )
    kernel = _get_compiled_kernel(selected_tile_v, use_aligned_io)
    kernel(
        mixed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
        output,
    )
    return output


__all__ = ["run_packed_kda_decode_cute"]
