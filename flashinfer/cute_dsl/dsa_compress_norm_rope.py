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

"""DeepSeek-style sparse KV cache compression using CuTe DSL.

This module provides two split kernels for the sparse MLA cache construction
path:

* ``dsa_compress_kv`` computes the softmax-weighted compressed KV vector from
  a paged state cache.
* ``dsa_norm_rope_store`` applies RMSNorm, RoPE, FP8 E4M3 quantization, and
  stores into a DeepSeek-style paged MLA KV cache layout.

The kernels are currently validated for DeepSeek-style shapes: head size 512,
RoPE head dim 64, quant block 64, and compression ratio 4 or 128.
"""

from __future__ import annotations

import functools
from typing import Callable

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .utils import torch_to_cutlass_dtype


# =============================================================================
# PTX helpers
# =============================================================================


@dsl_user_op
def _recast_val(x, dtype, *, loc=None, ip=None):
    return dtype(llvm.bitcast(dtype.mlir_type, x.ir_value(loc=loc, ip=ip)))


@dsl_user_op
def _fp32x2_to_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    out = llvm.inline_asm(
        T.i32(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint32(out)


@dsl_user_op
def _bf16x2_to_fp32(data: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [data.ir_value(loc=loc, ip=ip)],
        "shl.b32 $0, $2, 16;\n\tand.b32 $1, $2, 0xFFFF0000;\n",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _fp32x2_to_fp8e4m3x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint16:
    out = llvm.inline_asm(
        T.i16(),
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;",
        "=h,f,f",
        has_side_effects=False,
        is_align_stack=False,
    )
    return Uint16(out)


# =============================================================================
# CuTe DSL kernel classes
# =============================================================================


class DSACompressKernel:
    head_tile = 64
    rows_per_warp = 8
    row_pairs_per_warp = rows_per_warp // 2
    elems_per_lane = 4
    lanes_per_row = head_tile // elems_per_lane
    num_warps = 16
    stats_warp_stride = num_warps + 1
    tb_size = num_warps * 32
    rcp_ln2 = 1.4426950408889634

    def __init__(
        self,
        head_size: int,
        state_width: int,
        compress_ratio: int,
        overlap: bool,
    ):
        self.head_dim = head_size
        self.num_splits = head_size // self.head_tile
        self.state_width = state_width
        self.compress_ratio = compress_ratio
        self.overlap = overlap
        self.window = (1 + int(overlap)) * compress_ratio

    @cute.jit
    def __call__(
        self,
        state_cache: cute.Tensor,
        token_to_req_indices: cute.Tensor,
        positions: cute.Tensor,
        slot_mapping: cute.Tensor,
        block_table: cute.Tensor,
        block_size: Int64,
        compressed_kv: cute.Tensor,
        stream: CUstream,
    ):
        grid = (positions.shape[0] * self.num_splits, 1, 1)
        self.kernel(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_size,
            compressed_kv,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        state_cache: cute.Tensor,
        token_to_req_indices: cute.Tensor,
        positions: cute.Tensor,
        slot_mapping: cute.Tensor,
        block_table: cute.Tensor,
        block_size: Int64,
        compressed_kv: cute.Tensor,
    ):
        block_id, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        row_lane = lane_id // self.lanes_per_row
        col_group = lane_id % self.lanes_per_row

        token_idx = block_id // self.num_splits
        split_idx = block_id - token_idx * self.num_splits
        col_base = split_idx * self.head_tile + col_group * self.elems_per_lane

        slot_id = slot_mapping[token_idx]
        position = positions[token_idx]
        boundary = (position + Int64(1)) % Int64(self.compress_ratio) == Int64(0)
        active = slot_id >= Int64(0) and boundary

        if active:
            smem = cutlass.utils.SmemAllocator()
            s_max = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (
                        self.lanes_per_row,
                        self.elems_per_lane,
                        self.stats_warp_stride,
                    ),
                    stride=(
                        self.elems_per_lane * self.stats_warp_stride,
                        self.stats_warp_stride,
                        1,
                    ),
                ),
                byte_alignment=4,
            )
            s_sum = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (
                        self.lanes_per_row,
                        self.elems_per_lane,
                        self.stats_warp_stride,
                    ),
                    stride=(
                        self.elems_per_lane * self.stats_warp_stride,
                        self.stats_warp_stride,
                        1,
                    ),
                ),
                byte_alignment=4,
            )
            s_product = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (
                        self.lanes_per_row,
                        self.elems_per_lane,
                        self.stats_warp_stride,
                    ),
                    stride=(
                        self.elems_per_lane * self.stats_warp_stride,
                        self.stats_warp_stride,
                        1,
                    ),
                ),
                byte_alignment=4,
            )

            row_pair_layout = cute.make_layout(
                (self.row_pairs_per_warp, self.elems_per_lane),
                stride=(self.elems_per_lane, 1),
            )
            kv_vals = cute.make_rmem_tensor(row_pair_layout, Float32)
            score_vals = cute.make_rmem_tensor(row_pair_layout, Float32)
            local_max = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            local_sum = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            local_product = cute.make_rmem_tensor((self.elems_per_lane,), Float32)

            for e in cutlass.range_constexpr(self.elems_per_lane):
                local_max[e] = -Float32.inf
                local_sum[e] = Float32(0.0)
                local_product[e] = Float32(0.0)

            req_idx = token_to_req_indices[token_idx]
            start = position - Int64(self.window - 1)
            cp_f32x4 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            row_mask_and_clamp = const_expr(
                (cute.arch.WARP_SIZE - self.lanes_per_row) << 8
                | (cute.arch.WARP_SIZE - 1)
            )

            for i in cutlass.range_constexpr(self.row_pairs_per_warp):
                row = warp_id * self.rows_per_warp + i * 2 + row_lane
                pos = start + row.to(Int64)
                valid = row < self.window and pos >= Int64(0)
                head_offset = ((row // self.compress_ratio) * self.head_dim).to(Int64)

                for e in cutlass.range_constexpr(self.elems_per_lane):
                    kv = Float32(0.0)
                    score = -Float32.inf
                    kv_vals[i, e] = kv
                    score_vals[i, e] = score

                block_index = Int64(0)
                block_offset = Int64(0)
                block_number_i32 = Int32(0)
                if valid:
                    block_index = pos // block_size
                    block_offset = pos - block_index * block_size
                    if col_group == 0:
                        block_number_i32 = block_table[req_idx, block_index]
                block_number_i32 = cute.arch.shuffle_sync(
                    block_number_i32,
                    offset=0,
                    mask_and_clamp=row_mask_and_clamp,
                )

                if valid:
                    block_number = block_number_i32.to(Int64)
                    row_tensor = state_cache[block_number, block_offset, None]
                    col_tile = (head_offset + col_base.to(Int64)) // Int64(
                        self.elems_per_lane
                    )
                    kv_src = cute.local_tile(
                        row_tensor,
                        tiler=(self.elems_per_lane,),
                        coord=(col_tile,),
                    )
                    score_src = cute.local_tile(
                        row_tensor,
                        tiler=(self.elems_per_lane,),
                        coord=(
                            col_tile + Int64(self.state_width // self.elems_per_lane),
                        ),
                    )
                    cute.copy(cp_f32x4, kv_src, kv_vals[i, None])
                    cute.copy(cp_f32x4, score_src, score_vals[i, None])

                for e in cutlass.range_constexpr(self.elems_per_lane):
                    local_max[e] = cute.arch.fmax(local_max[e], score_vals[i, e])

            for e in cutlass.range_constexpr(self.elems_per_lane):
                if local_max[e] > -Float32.inf:
                    for i in cutlass.range_constexpr(self.row_pairs_per_warp):
                        exp_score = cute.math.exp2(
                            (score_vals[i, e] - local_max[e]) * Float32(self.rcp_ln2),
                            fastmath=True,
                        )
                        local_sum[e] += exp_score
                        local_product[e] += kv_vals[i, e] * exp_score

            for e in cutlass.range_constexpr(self.elems_per_lane):
                pair_max = cute.arch.shuffle_sync_bfly(local_max[e], offset=16)
                pair_sum = cute.arch.shuffle_sync_bfly(local_sum[e], offset=16)
                pair_product = cute.arch.shuffle_sync_bfly(local_product[e], offset=16)
                warp_max = cute.arch.fmax(local_max[e], pair_max)
                warp_sum = Float32(0.0)
                warp_product = Float32(0.0)
                if warp_max > -Float32.inf:
                    local_scale = cute.math.exp2(
                        (local_max[e] - warp_max) * Float32(self.rcp_ln2),
                        fastmath=True,
                    )
                    pair_scale = cute.math.exp2(
                        (pair_max - warp_max) * Float32(self.rcp_ln2),
                        fastmath=True,
                    )
                    warp_sum = local_sum[e] * local_scale + pair_sum * pair_scale
                    warp_product = (
                        local_product[e] * local_scale + pair_product * pair_scale
                    )
                if lane_id < self.lanes_per_row:
                    s_max[col_group, e, warp_id] = warp_max
                    s_sum[col_group, e, warp_id] = warp_sum
                    s_product[col_group, e, warp_id] = warp_product
            cute.arch.sync_threads()

            out_group = tid // Int32(self.num_warps)
            final_lane = tid % Int32(self.num_warps)
            final_groups_per_pass = const_expr(self.tb_size // self.num_warps)
            for pass_idx in cutlass.range_constexpr(
                self.head_tile // final_groups_per_pass
            ):
                out_idx = Int32(pass_idx * final_groups_per_pass) + out_group
                out_lane = out_idx // Int32(self.elems_per_lane)
                out_elem = out_idx % Int32(self.elems_per_lane)

                local_warp_max = s_max[out_lane, out_elem, final_lane]
                global_max = local_warp_max
                for step in cutlass.range_constexpr(4):
                    offset = const_expr(8 >> step)
                    global_max = cute.arch.fmax(
                        global_max,
                        cute.arch.shuffle_sync_bfly(
                            global_max,
                            offset=offset,
                            mask_and_clamp=row_mask_and_clamp,
                        ),
                    )

                scale = cute.math.exp2(
                    (local_warp_max - global_max) * Float32(self.rcp_ln2),
                    fastmath=True,
                )
                global_sum = s_sum[out_lane, out_elem, final_lane] * scale
                global_product = s_product[out_lane, out_elem, final_lane] * scale
                for step in cutlass.range_constexpr(4):
                    offset = const_expr(8 >> step)
                    global_sum += cute.arch.shuffle_sync_bfly(
                        global_sum,
                        offset=offset,
                        mask_and_clamp=row_mask_and_clamp,
                    )
                    global_product += cute.arch.shuffle_sync_bfly(
                        global_product,
                        offset=offset,
                        mask_and_clamp=row_mask_and_clamp,
                    )

                if final_lane == Int32(0):
                    out_col = split_idx * Int32(self.head_tile) + out_idx
                    compressed_kv[token_idx, out_col] = global_product / global_sum


class DSANormRopeStoreKernel:
    min_scale = 1.0e-4

    def __init__(
        self,
        head_size: int,
        rope_head_dim: int,
        fp8_max: float,
        quant_block: int,
        token_stride: int,
        scale_dim: int,
        kv_block_stride: int,
        compress_ratio: int,
    ):
        self.head_dim = head_size
        self.rope_dim = rope_head_dim
        self.nope_dim = head_size - rope_head_dim
        self.fp8_max = fp8_max
        self.quant_block = quant_block
        self.token_stride = token_stride
        self.scale_dim = scale_dim
        self.kv_block_stride = kv_block_stride
        self.num_warps = head_size // quant_block
        self.nope_blocks = self.nope_dim // quant_block
        self.tb_size = head_size // 2
        self.compress_ratio = compress_ratio

    @cute.jit
    def __call__(
        self,
        compressed_kv: cute.Tensor,
        positions: cute.Tensor,
        slot_mapping: cute.Tensor,
        rms_norm_weight: cute.Tensor,
        rms_norm_eps: Float32,
        cos_sin_cache: cute.Tensor,
        k_cache: cute.Tensor,
        kv_slot_mapping: cute.Tensor,
        kv_cache_block_size: Int64,
        stream: CUstream,
    ):
        grid = (positions.shape[0], 1, 1)
        self.kernel(
            compressed_kv,
            positions,
            slot_mapping,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
            kv_cache_block_size,
        ).launch(grid=grid, block=(self.tb_size, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        compressed_kv: cute.Tensor,
        positions: cute.Tensor,
        slot_mapping: cute.Tensor,
        rms_norm_weight: cute.Tensor,
        rms_norm_eps: Float32,
        cos_sin_cache: cute.Tensor,
        k_cache: cute.Tensor,
        kv_slot_mapping: cute.Tensor,
        kv_cache_block_size: Int64,
    ):
        token_idx, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        elem0 = tid * 2

        slot_id = slot_mapping[token_idx]
        position = positions[token_idx]
        boundary = (position + Int64(1)) % Int64(self.compress_ratio) == Int64(0)
        kv_slot_idx = kv_slot_mapping[token_idx]
        active = slot_id >= Int64(0) and boundary and kv_slot_idx >= Int64(0)

        if active:
            x0 = compressed_kv[token_idx, elem0]
            x1 = compressed_kv[token_idx, elem0 + 1]

            local_sumsq = x0 * x0 + x1 * x1
            warp_sum = local_sumsq
            for step in cutlass.range_constexpr(5):
                offset = const_expr(16 >> step)
                warp_sum += cute.arch.shuffle_sync_bfly(warp_sum, offset)

            smem = cutlass.utils.SmemAllocator()
            partial_sums = smem.allocate_tensor(
                Float32, cute.make_layout((self.num_warps,)), byte_alignment=4
            )
            rrms_shared = smem.allocate_tensor(
                Float32, cute.make_layout((1,)), byte_alignment=4
            )

            if lane_id == 0:
                partial_sums[warp_id] = warp_sum
            cute.arch.sync_threads()
            if tid == 0:
                total = Float32(0.0)
                for i in cutlass.range_constexpr(self.num_warps):
                    total += partial_sums[i]
                rrms_shared[0] = cute.math.rsqrt(
                    total / Float32(self.head_dim) + rms_norm_eps, fastmath=True
                )
            cute.arch.sync_threads()

            rrms = rrms_shared[0]
            x0 = x0 * rrms * rms_norm_weight[elem0].to(Float32)
            x1 = x1 * rrms * rms_norm_weight[elem0 + 1].to(Float32)

            k_cache_u16 = cute.recast_tensor(k_cache, Uint16)
            k_cache_u32 = cute.recast_tensor(k_cache, Uint32)
            k_cache_flat = cute.make_tensor(
                k_cache.iterator,
                cute.make_layout((cute.size(k_cache),), stride=(1,)),
            )
            k_cache_u16_flat = cute.make_tensor(
                k_cache_u16.iterator,
                cute.make_layout((cute.size(k_cache_u16),), stride=(1,)),
            )
            k_cache_u32_flat = cute.make_tensor(
                k_cache_u32.iterator,
                cute.make_layout((cute.size(k_cache_u32),), stride=(1,)),
            )
            page = kv_slot_idx // kv_cache_block_size
            kv_offset = kv_slot_idx - page * kv_cache_block_size
            value_base = page * Int64(self.kv_block_stride) + kv_offset * Int64(
                self.token_stride
            )
            scale_base = (
                page * Int64(self.kv_block_stride)
                + kv_cache_block_size * Int64(self.token_stride)
                + kv_offset * Int64(self.scale_dim)
            )

            if warp_id == self.nope_blocks:
                pair_idx = lane_id
                compressed_pos = (position // Int64(self.compress_ratio)) * Int64(
                    self.compress_ratio
                )
                cos_v = cos_sin_cache[compressed_pos, pair_idx]
                sin_v = cos_sin_cache[
                    compressed_pos, pair_idx + Int32(self.rope_dim // 2)
                ]
                real = x0 * cos_v - x1 * sin_v
                imag = x0 * sin_v + x1 * cos_v
                packed = _fp32x2_to_bf16x2(real, imag)
                out_base = value_base + Int64(self.nope_dim) + (lane_id * 4).to(Int64)
                k_cache_u32_flat[out_base // Int64(4)] = packed
            else:
                q_packed = _fp32x2_to_bf16x2(x0, x1)
                q0, q1 = _bf16x2_to_fp32(q_packed)
                abs0 = cute.arch.fmax(q0, -q0)
                abs1 = cute.arch.fmax(q1, -q1)
                local_absmax = cute.arch.fmax(abs0, abs1)
                absmax = local_absmax
                for step in cutlass.range_constexpr(5):
                    offset = const_expr(16 >> step)
                    absmax = cute.arch.fmax(
                        absmax, cute.arch.shuffle_sync_bfly(absmax, offset)
                    )
                scale_raw = cute.arch.fmax(
                    Float32(self.min_scale),
                    absmax / Float32(self.fp8_max),
                )
                bits = _recast_val(scale_raw, Uint32)
                ue8m0 = ((bits + Uint32(0x7FFFFF)) >> Uint32(23)) & Uint32(0xFF)
                inv_scale = _recast_val((Uint32(254) - ue8m0) << Uint32(23), Float32)
                y0 = cute.arch.fmax(q0 * inv_scale, Float32(-self.fp8_max))
                y0 = -cute.arch.fmax(-y0, Float32(-self.fp8_max))
                y1 = cute.arch.fmax(q1 * inv_scale, Float32(-self.fp8_max))
                y1 = -cute.arch.fmax(-y1, Float32(-self.fp8_max))
                packed_fp8 = _fp32x2_to_fp8e4m3x2(y0, y1)
                out_base = value_base + (warp_id * self.quant_block + lane_id * 2).to(
                    Int64
                )
                k_cache_u16_flat[out_base // Int64(2)] = packed_fp8
                if lane_id == 0:
                    k_cache_flat[scale_base + warp_id.to(Int64)] = ue8m0.to(Uint8)
                    if warp_id == 0:
                        k_cache_flat[scale_base + Int64(self.nope_blocks)] = Uint8(0)


# =============================================================================
# PyTorch API functions
# =============================================================================


@functools.cache
def _get_dsa_compress_kernel(
    head_size: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
) -> Callable:
    """Compile and cache the DSA compression kernel."""
    if head_size % DSACompressKernel.head_tile != 0:
        raise ValueError("head_size must be divisible by the 64-wide head tile.")

    num_tokens = cute.sym_int()
    num_blocks = cute.sym_int()
    state_cache_block_size = cute.sym_int()
    block_table_width = cute.sym_int()
    state_cache_width = state_width * 2

    state_cache_fake = cute.runtime.make_fake_tensor(
        Float32,
        (num_blocks, state_cache_block_size, state_cache_width),
        stride=(
            cute.sym_int64(divisibility=16),
            cute.sym_int64(divisibility=16),
            1,
        ),
        assumed_align=16,
    )
    token_to_req_indices_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (num_tokens,), assumed_align=4
    )
    positions_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (num_tokens,), assumed_align=8
    )
    slot_mapping_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (num_tokens,), assumed_align=8
    )
    block_table_fake = cute.runtime.make_fake_compact_tensor(
        Int32,
        (cute.sym_int(), block_table_width),
        stride_order=(1, 0),
        assumed_align=4,
    )
    compressed_kv_fake = cute.runtime.make_fake_tensor(
        Float32,
        (num_tokens, head_size),
        stride=(cute.sym_int64(divisibility=4), 1),
        assumed_align=4,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = DSACompressKernel(
        head_size=head_size,
        state_width=state_width,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )
    return cute.compile(
        kernel,
        state_cache_fake,
        token_to_req_indices_fake,
        positions_fake,
        slot_mapping_fake,
        block_table_fake,
        Int64(0),
        compressed_kv_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _get_dsa_norm_rope_store_kernel(
    head_size: int,
    rope_head_dim: int,
    fp8_max: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
    kv_block_stride: int,
    compress_ratio: int,
    norm_weight_dtype: type[cutlass.Numeric],
) -> Callable:
    """Compile and cache the DSA norm/RoPE/store kernel."""
    if quant_block != 64:
        raise ValueError("CuTe DSL DSA store currently requires quant_block=64.")
    if rope_head_dim != 64:
        raise ValueError("CuTe DSL DSA store currently requires rope_head_dim=64.")
    if head_size % quant_block != 0:
        raise ValueError("head_size must be divisible by quant_block.")
    if token_stride < head_size + rope_head_dim:
        raise ValueError("token_stride is too small for the packed FP8/BF16 row.")
    expected_scale_dim = (head_size - rope_head_dim) // quant_block + 1
    if scale_dim < expected_scale_dim:
        raise ValueError("scale_dim is too small for the UE8M0 scale row.")

    num_tokens = cute.sym_int()
    max_pos = cute.sym_int()
    num_blocks = cute.sym_int()

    compressed_kv_fake = cute.runtime.make_fake_tensor(
        Float32,
        (num_tokens, head_size),
        stride=(cute.sym_int64(divisibility=4), 1),
        assumed_align=4,
    )
    positions_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (num_tokens,), assumed_align=8
    )
    slot_mapping_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (num_tokens,), assumed_align=8
    )
    rms_norm_weight_fake = cute.runtime.make_fake_compact_tensor(
        norm_weight_dtype, (head_size,), assumed_align=4
    )
    cos_sin_cache_fake = cute.runtime.make_fake_tensor(
        Float32,
        (max_pos, rope_head_dim),
        stride=(cute.sym_int64(divisibility=4), 1),
        assumed_align=4,
    )
    k_cache_fake = cute.runtime.make_fake_tensor(
        Uint8,
        (num_blocks, cute.sym_int(), cute.sym_int()),
        stride=(
            cute.sym_int64(divisibility=16),
            cute.sym_int64(divisibility=8),
            1,
        ),
        assumed_align=16,
    )
    kv_slot_mapping_fake = cute.runtime.make_fake_compact_tensor(
        Int64, (num_tokens,), assumed_align=8
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = DSANormRopeStoreKernel(
        head_size=head_size,
        rope_head_dim=rope_head_dim,
        fp8_max=fp8_max,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
        kv_block_stride=kv_block_stride,
        compress_ratio=compress_ratio,
    )
    return cute.compile(
        kernel,
        compressed_kv_fake,
        positions_fake,
        slot_mapping_fake,
        rms_norm_weight_fake,
        Float32(0.0),
        cos_sin_cache_fake,
        k_cache_fake,
        kv_slot_mapping_fake,
        Int64(0),
        stream_fake,
        options="--enable-tvm-ffi",
    )


def dsa_compress_kv(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compressed_kv: torch.Tensor,
    head_size: int = 512,
    state_width: int = 512,
    compress_ratio: int = 128,
    overlap: bool = False,
) -> None:
    """Compute compressed KV vectors from a paged DSA state cache.

    ``compressed_kv`` is updated in place.  The state cache stores KV and score
    rows as float32 with shape ``[num_blocks, block_size, 2 * state_width]``.
    """
    if positions.numel() == 0:
        return
    compiled_kernel = _get_dsa_compress_kernel(
        head_size=head_size,
        state_width=state_width,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )
    compiled_kernel(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        compressed_kv,
    )


def dsa_norm_rope_store(
    compressed_kv: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    kv_block_stride: int,
    head_size: int = 512,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 64,
    token_stride: int = 576,
    scale_dim: int = 8,
    compress_ratio: int = 128,
) -> None:
    """Apply RMSNorm/RoPE/FP8 quantization and store into paged MLA KV cache.

    ``k_cache`` is updated in place using the packed DSA cache layout:
    quantized NoPE bytes first, BF16 RoPE bytes next, and UE8M0 scales in the
    per-page scale area.
    """
    if positions.numel() == 0:
        return
    try:
        norm_weight_dtype = torch_to_cutlass_dtype(rms_norm_weight.dtype)
    except TypeError as exc:
        raise ValueError(
            "CuTe DSL DSA store supports rms_norm_weight dtype bf16/fp32, "
            f"got {rms_norm_weight.dtype}."
        ) from exc
    if norm_weight_dtype not in (BFloat16, Float32):
        raise ValueError(
            "CuTe DSL DSA store supports rms_norm_weight dtype bf16/fp32, "
            f"got {rms_norm_weight.dtype}."
        )
    if k_cache.ndim != 3:
        raise ValueError(
            "CuTe DSL DSA store expects a 3D k_cache layout "
            f"[num_blocks, block_size, token_bytes], got ndim={k_cache.ndim}."
        )

    compiled_kernel = _get_dsa_norm_rope_store_kernel(
        head_size=head_size,
        rope_head_dim=rope_head_dim,
        fp8_max=fp8_max,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
        kv_block_stride=kv_block_stride,
        compress_ratio=compress_ratio,
        norm_weight_dtype=norm_weight_dtype,
    )
    compiled_kernel(
        compressed_kv,
        positions,
        slot_mapping,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
        kv_cache_block_size,
    )


__all__ = [
    "DSACompressKernel",
    "DSANormRopeStoreKernel",
    "dsa_compress_kv",
    "dsa_norm_rope_store",
]
