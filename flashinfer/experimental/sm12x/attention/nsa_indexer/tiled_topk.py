# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/indexer/tiled_topk.py @ c795c02a (2026-07-16) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Faithful CuTe DSL port of the sglang radix-select topk kernel.

Ported from: sgl-kernel/csrc/elementwise/topk.cu :: fast_topk_cuda_tl

Uses raw PTX shared memory loads/stores/atomics for the double-buffered
histogram prefix sum, since CuTe DSL does not support dynamic selection
between tensor objects. Variables in dynamic control flow require SSA-style
initialization before use.
"""

from __future__ import annotations

from functools import lru_cache
import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from flashinfer.experimental.sm12x._lib.compiler import (
    DimKey,
    KernelCompileSpec,
    launch as sm12x_launch,
    tensor_compile_fact,
)
from flashinfer.experimental.sm12x._lib.intrinsics import (
    atomic_add_shared_i32,
    ld_shared_i32,
    ld_shared_i32_relaxed,
    red_add_shared_i32,
    shared_ptr_to_u32,
    st_shared_i32,
)
from flashinfer.experimental.sm12x._lib.utils import current_cuda_stream

_THREADS_PER_CTA = 1024
_DEFAULT_TOPK = 2048
_SUPPORTED_TOPK = (512, 1024, 2048)
_RADIX = 256
_SMEM_CANDS = 4096
_SCAN_UNROLL = 4
_SUPERTILE_K_ENV = "FLASHINFER_EXP_SM12X_NSA_TOPK_SUPERTILE_K"
_SUPERTILE_K_DEFAULT = 32768


@dsl_user_op
def _cvt_rn_f16_f32(val: Float32, *, loc=None, ip=None) -> Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [Float32(val).ir_value(loc=loc, ip=ip)],
        "cvt.rn.f16.f32 $0, $1;",
        "=h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _float_as_uint32(val: Float32, *, loc=None, ip=None) -> Uint32:
    result = llvm.inline_asm(
        T.i32(),
        [Float32(val).ir_value(loc=loc, ip=ip)],
        "mov.b32 $0, $1;",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@cute.jit
def _smem_ld(base: Int32, idx: Int32) -> Int32:
    return ld_shared_i32(base + idx * Int32(4))


@cute.jit
def _smem_ld_relaxed(base: Int32, idx: Int32) -> Int32:
    return ld_shared_i32_relaxed(base + idx * Int32(4))


@cute.jit
def _smem_st(base: Int32, idx: Int32, val: Int32):
    st_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _smem_xadd(base: Int32, idx: Int32, val: Int32) -> Int32:
    return atomic_add_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _smem_red_add(base: Int32, idx: Int32, val: Int32):
    red_add_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _load_topk_input_from_row_base(
    input_tensor,
    row_base: Int32,
    logical_k: Int32,
    block_q: cutlass.Constexpr[int],
    block_k: cutlass.Constexpr[int],
    is_tiled: cutlass.Constexpr[bool],
) -> Float32:
    value = Float32(0.0)
    if cutlass.const_expr(is_tiled):
        tile_size = Int32(block_q * block_k)
        k_tile_idx = Int32(0)
        k_local = Int32(0)
        if cutlass.const_expr(block_k == 256):
            k_tile_idx = logical_k >> Int32(8)
            k_local = logical_k & Int32(255)
        else:
            k_tile_idx = logical_k >> Int32(9)
            k_local = logical_k & Int32(511)
        value = Float32(input_tensor[row_base + k_tile_idx * tile_size + k_local])
    else:
        value = Float32(input_tensor[row_base + logical_k])
    return value


@cute.jit
def _load_value_virtual(
    input_tensor,
    carry_values,
    row_base: Int32,
    row_start: Int32,
    carry_base: Int32,
    chunk_len: Int32,
    vidx: Int32,
    block_q: cutlass.Constexpr[int],
    block_k: cutlass.Constexpr[int],
    is_tiled: cutlass.Constexpr[bool],
    is_first: cutlass.Constexpr[bool],
) -> Float32:
    """Load a candidate's logit by virtual index.

    vidx in [0, chunk_len) reads this chunk's tiled logits; vidx >= chunk_len reads
    the carried running-topk value at slot (vidx - chunk_len). When is_first there is
    no carry range, so the carry branch is compiled away.
    """
    value = Float32(0.0)
    if cutlass.const_expr(is_first):
        value = _load_topk_input_from_row_base(
            input_tensor, row_base, row_start + vidx, block_q, block_k, is_tiled
        )
    else:
        if vidx < chunk_len:
            value = _load_topk_input_from_row_base(
                input_tensor, row_base, row_start + vidx, block_q, block_k, is_tiled
            )
        else:
            value = Float32(carry_values[carry_base + (vidx - chunk_len)])
    return value


@cute.jit
def _emit_global_index_virtual(
    carry_indices,
    output_page_table,
    row_start: Int32,
    output_index_offset: Int32,
    carry_base: Int32,
    chunk_len: Int32,
    vidx: Int32,
    row_idx: Int32,
    page_size: Int32,
    is_first: cutlass.Constexpr[bool],
    output_physical_slots: cutlass.Constexpr[bool],
) -> Int32:
    """Map a virtual index to its final logical or physical K-index.

    Local elements reconstruct as row_start + vidx + output_index_offset; carried
    elements pass their already-global logical index through verbatim (never
    re-offset). The final paged-indexer chunk optionally translates that logical
    request-relative position through the real page table in this same store.
    """
    gidx = Int32(0)
    if cutlass.const_expr(is_first):
        gidx = row_start + vidx + output_index_offset
    else:
        if vidx < chunk_len:
            gidx = row_start + vidx + output_index_offset
        else:
            gidx = Int32(carry_indices[carry_base + (vidx - chunk_len)])
    if cutlass.const_expr(output_physical_slots):
        physical_idx = Int32(-1)
        if gidx >= Int32(0):
            page_col = gidx // page_size
            page_offset = gidx - page_col * page_size
            page_id = Int32(output_page_table[row_idx, page_col])
            if page_id >= Int32(0):
                physical_idx = page_id * page_size + page_offset
        gidx = physical_idx
    return gidx


@cute.jit
def _convert_to_uint8(x: Float32) -> Uint32:
    h_bits = _cvt_rn_f16_f32(x)
    bits16 = h_bits & Uint32(0xFFFF)
    sign = bits16 & Uint32(0x8000)
    key16 = Uint32(0)
    if sign != Uint32(0):
        key16 = Uint32(0xFFFF) ^ bits16
    else:
        key16 = bits16 | Uint32(0x8000)
    return (key16 >> Uint32(8)) & Uint32(0xFF)


@cute.jit
def _convert_to_uint32(x: Float32) -> Uint32:
    bits = _float_as_uint32(x)
    sign = bits & Uint32(0x80000000)
    result = Uint32(0)
    if sign != Uint32(0):
        result = ~bits
    else:
        result = bits | Uint32(0x80000000)
    return result


def _to_kernel_tensor(tensor, dtype, *, assumed_align=16):
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = dtype
    if tensor.ndim >= 1:
        leading_dim = next(
            (idx for idx, stride in enumerate(tensor.stride()) if stride == 1), None
        )
        if leading_dim is not None:
            cute_tensor = cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return cute_tensor


def _tensor_compile_key(name, tensor, *, dynamic_dims=()):
    return tensor_compile_fact(name, tensor, dynamic_dims=dynamic_dims)


def _flat_tensor_compile_key(name, tensor, *, dynamic=False):
    flat = tensor.reshape(-1)
    dims = (DimKey.dynamic() if dynamic else DimKey.exact(int(flat.shape[0])),)
    return tensor_compile_fact(name, flat, dims=dims)


def _tensor_meta_key(tensor, *, dynamic_dims=()):
    dynamic_dim_set = set(dynamic_dims)
    return (
        tuple(
            DimKey.dynamic() if idx in dynamic_dim_set else int(dim)
            for idx, dim in enumerate(tensor.shape)
        ),
        tuple(tensor.stride()),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


def _flat_tensor_meta_key(tensor):
    return (
        (int(tensor.numel()),),
        (1,),
        str(tensor.dtype),
        (tensor.device.type, tensor.device.index),
    )


class SparseNSATiledTopkKernel:
    def __init__(
        self,
        *,
        is_tiled: bool = False,
        block_q: int = 1,
        block_k: int = 1,
        topk: int = _DEFAULT_TOPK,
        zero_row_start: bool = False,
        is_first: bool = True,
        output_physical_slots: bool = False,
        extent_splits: int = 1,
    ):
        self.extent_splits = int(extent_splits)
        self.is_tiled = is_tiled
        self.block_q = int(block_q)
        self.block_k = int(block_k)
        self.topk = int(topk)
        self.zero_row_start = bool(zero_row_start)
        # When is_first the carry buffers are unread (the running-topk fold has no
        # predecessor for the first supertile chunk). There is no `is_last`: the
        # kernel always emits (value, global index) topk into values/indices; the
        # orchestrator decides whether that tensor is the next chunk's carry buffer
        # or the user's final output.
        self.is_first = bool(is_first)
        self.output_physical_slots = bool(output_physical_slots)

    @cute.jit
    def __call__(
        self,
        input_tensor,
        row_starts,
        lengths,
        values,
        indices,
        carry_values,
        carry_indices,
        output_page_table,
        batch_size,
        input_stride,
        num_k_tiles,
        tile_k_offset,
        block_q,
        block_k,
        topk_val,
        input_index_offset,
        input_extent,
        output_index_offset,
        output_page_size,
        out_row_stride,
        out_row_base,
        stream,
    ):
        self.kernel(
            input_tensor,
            row_starts,
            lengths,
            values,
            indices,
            carry_values,
            carry_indices,
            output_page_table,
            batch_size,
            input_stride,
            num_k_tiles,
            tile_k_offset,
            block_q,
            block_k,
            topk_val,
            input_index_offset,
            input_extent,
            output_index_offset,
            output_page_size,
            out_row_stride,
            out_row_base,
        ).launch(
            grid=(batch_size, 1, 1),
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        row_starts: cute.Tensor,
        lengths: cute.Tensor,
        values: cute.Tensor,
        indices: cute.Tensor,
        carry_values: cute.Tensor,
        carry_indices: cute.Tensor,
        output_page_table: cute.Tensor,
        batch_size: Int32,
        input_stride: Int32,
        num_k_tiles: Int32,
        tile_k_offset: Int32,
        block_q: Int32,
        block_k: Int32,
        topk_val: Int32,
        input_index_offset: Int32,
        input_extent: Int32,
        output_index_offset: Int32,
        output_page_size: Int32,
        out_row_stride: Int32,
        out_row_base: Int32,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        bid = Int32(bid)

        # extent_splits > 1: pseudo-row decomposition. CTA `bid` selects the
        # top-k of ONE extent slice of real row bid//splits: slice s covers
        # [s*input_extent, (s+1)*input_extent) via the existing chunk-clip
        # logic, so the emitted global indices stay correct. Outputs land at
        # pseudo-row `bid`; a second (linear) pass folds the slices.
        data_bid = bid
        if cutlass.const_expr(self.extent_splits > 1):
            splits_i = Int32(self.extent_splits)
            data_bid = bid // splits_i
            split_id = bid - data_bid * splits_i
            input_index_offset = input_index_offset + split_id * input_extent
            output_index_offset = output_index_offset + split_id * input_extent
            tile_k_offset = tile_k_offset + split_id * (
                input_extent // Int32(self.block_k)
            )

        row_start = Int32(0)
        if not cutlass.const_expr(self.zero_row_start):
            row_start = Int32(row_starts[data_bid])
        length = Int32(lengths[data_bid])
        if input_extent > Int32(0):
            row_end = row_start + length
            clipped_start = row_start
            if clipped_start < input_index_offset:
                clipped_start = input_index_offset
            clipped_end = row_end
            chunk_end = input_index_offset + input_extent
            if cutlass.const_expr(self.is_tiled):
                # A balanced extent split can round its final slice up by one
                # tile. Clip that slice to the tiles actually present instead
                # of reading the next scratch region as logits.
                tiled_end = input_index_offset + (num_k_tiles - tile_k_offset) * block_k
                if chunk_end > tiled_end:
                    chunk_end = tiled_end
            if clipped_end > chunk_end:
                clipped_end = chunk_end
            if clipped_end > clipped_start:
                row_start = clipped_start - input_index_offset
                length = clipped_end - clipped_start
            else:
                row_start = Int32(0)
                length = Int32(0)
        topk_capacity = self.topk
        topk_static = Int32(self.topk)
        out_row = data_bid * out_row_stride + out_row_base
        if cutlass.const_expr(self.extent_splits > 1):
            out_row = out_row + split_id
        out_base = out_row * topk_static
        row_base = data_bid * input_stride
        if cutlass.const_expr(self.is_tiled):
            block_q_i = Int32(self.block_q)
            block_k_i = Int32(self.block_k)
            tile_size = Int32(self.block_q * self.block_k)
            q_tile_idx = data_bid // block_q_i
            q_local = data_bid - q_tile_idx * block_q_i
            row_base = (
                q_tile_idx * num_k_tiles * tile_size
                + tile_k_offset * tile_size
                + q_local * block_k_i
            )

        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            hist0: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 384], 128]
            hist1: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 384], 128]
            out_idx: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, topk_capacity], 128
            ]
            counter: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            thr_id: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            ni0: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            ni1: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            last_rem: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 1], 128]
            cand0: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _SMEM_CANDS], 128
            ]
            cand1: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _SMEM_CANDS], 128
            ]

        storage = smem_alloc.allocate(SharedStorage)

        s_hist0 = storage.hist0.get_tensor(cute.make_layout((384,), stride=(1,)))
        s_hist1 = storage.hist1.get_tensor(cute.make_layout((384,), stride=(1,)))
        s_out = storage.out_idx.get_tensor(
            cute.make_layout((topk_capacity,), stride=(1,))
        )
        s_cand0 = storage.cand0.get_tensor(
            cute.make_layout((_SMEM_CANDS,), stride=(1,))
        )
        s_cand1 = storage.cand1.get_tensor(
            cute.make_layout((_SMEM_CANDS,), stride=(1,))
        )

        h0 = shared_ptr_to_u32(storage.hist0.data_ptr())
        ctr = shared_ptr_to_u32(storage.counter.data_ptr())
        thr = shared_ptr_to_u32(storage.thr_id.data_ptr())
        ni0 = shared_ptr_to_u32(storage.ni0.data_ptr())
        ni1 = shared_ptr_to_u32(storage.ni1.data_ptr())
        lr = shared_ptr_to_u32(storage.last_rem.data_ptr())

        # Virtual candidate range: local logits in [0, length) plus, for fold chunks
        # (not is_first), the topk carried running-topk slots in [length, length+topk).
        total_len = length
        if not cutlass.const_expr(self.is_first):
            total_len = length + topk_static

        need_radix = total_len > topk_static

        if not need_radix:
            i = Int32(tx)
            while i < topk_static:
                is_valid = i < total_len
                values[out_base + i] = (
                    _load_value_virtual(
                        input_tensor,
                        carry_values,
                        row_base,
                        row_start,
                        out_base,
                        length,
                        i,
                        self.block_q,
                        self.block_k,
                        self.is_tiled,
                        self.is_first,
                    )
                    if is_valid
                    else Float32(float("-inf"))
                )
                indices[out_base + i] = (
                    _emit_global_index_virtual(
                        carry_indices,
                        output_page_table,
                        row_start,
                        output_index_offset,
                        out_base,
                        length,
                        i,
                        bid,
                        output_page_size,
                        self.is_first,
                        self.output_physical_slots,
                    )
                    if is_valid
                    else Int32(-1)
                )
                i = i + Int32(_THREADS_PER_CTA)

        if need_radix:
            topk = topk_static

            if tx < Int32(257):
                s_hist0[tx] = Int32(0)
            cute.arch.sync_threads()

            idx_base = Int32(tx)
            full_scan_limit = total_len - Int32((_SCAN_UNROLL - 1) * _THREADS_PER_CTA)
            while idx_base < full_scan_limit:
                for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
                    idx = idx_base + Int32(scan_u * _THREADS_PER_CTA)
                    val = _load_value_virtual(
                        input_tensor,
                        carry_values,
                        row_base,
                        row_start,
                        out_base,
                        length,
                        idx,
                        self.block_q,
                        self.block_k,
                        self.is_tiled,
                        self.is_first,
                    )
                    bin8 = _convert_to_uint8(val)
                    _smem_red_add(h0, Int32(bin8), Int32(1))
                idx_base = idx_base + Int32(_THREADS_PER_CTA * _SCAN_UNROLL)
            while idx_base < total_len:
                val = _load_value_virtual(
                    input_tensor,
                    carry_values,
                    row_base,
                    row_start,
                    out_base,
                    length,
                    idx_base,
                    self.block_q,
                    self.block_k,
                    self.is_tiled,
                    self.is_first,
                )
                bin8 = _convert_to_uint8(val)
                _smem_red_add(h0, Int32(bin8), Int32(1))
                idx_base = idx_base + Int32(_THREADS_PER_CTA)

            cute.arch.sync_threads()

            # Parallel prefix sum: 8 stages, double-buffer h0/h1
            for stage in cutlass.range_constexpr(8):
                j = Int32(1 << stage)
                if tx < Int32(256):
                    if (stage & 1) == 0:
                        value = Int32(s_hist0[tx])
                        if tx < Int32(256) - j:
                            value = value + Int32(s_hist0[tx + j])
                        s_hist1[tx] = value
                    else:
                        value = Int32(s_hist1[tx])
                        if tx < Int32(256) - j:
                            value = value + Int32(s_hist1[tx + j])
                        s_hist0[tx] = value
                cute.arch.sync_threads()

            # Find threshold bin
            if tx < Int32(256):
                val_tx = Int32(s_hist0[tx])
                val_tx1 = Int32(s_hist0[tx + Int32(1)])
                if val_tx > topk:
                    if val_tx1 <= topk:
                        _smem_st(thr, Int32(0), Int32(tx))
                        _smem_st(ni0, Int32(0), Int32(0))
                        _smem_st(ctr, Int32(0), Int32(0))

            cute.arch.sync_threads()
            threshold_bin = _smem_ld(thr, Int32(0))
            topk = topk - Int32(s_hist0[threshold_bin + Int32(1)])

            if topk == Int32(0):
                idx_base = Int32(tx)
                full_scan_limit = total_len - Int32(
                    (_SCAN_UNROLL - 1) * _THREADS_PER_CTA
                )
                while idx_base < full_scan_limit:
                    for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
                        idx = idx_base + Int32(scan_u * _THREADS_PER_CTA)
                        val = _load_value_virtual(
                            input_tensor,
                            carry_values,
                            row_base,
                            row_start,
                            out_base,
                            length,
                            idx,
                            self.block_q,
                            self.block_k,
                            self.is_tiled,
                            self.is_first,
                        )
                        bin8 = _convert_to_uint8(val)
                        if Int32(bin8) > threshold_bin:
                            pos = _smem_xadd(ctr, Int32(0), Int32(1))
                            s_out[pos] = idx
                    idx_base = idx_base + Int32(_THREADS_PER_CTA * _SCAN_UNROLL)
                while idx_base < total_len:
                    val = _load_value_virtual(
                        input_tensor,
                        carry_values,
                        row_base,
                        row_start,
                        out_base,
                        length,
                        idx_base,
                        self.block_q,
                        self.block_k,
                        self.is_tiled,
                        self.is_first,
                    )
                    bin8 = _convert_to_uint8(val)
                    if Int32(bin8) > threshold_bin:
                        pos = _smem_xadd(ctr, Int32(0), Int32(1))
                        s_out[pos] = idx_base
                    idx_base = idx_base + Int32(_THREADS_PER_CTA)

            if topk != Int32(0):
                cute.arch.sync_threads()

                if tx < Int32(257):
                    s_hist0[tx] = Int32(0)
                cute.arch.sync_threads()

                idx_base = Int32(tx)
                full_scan_limit = total_len - Int32(
                    (_SCAN_UNROLL - 1) * _THREADS_PER_CTA
                )
                while idx_base < full_scan_limit:
                    for scan_u in cutlass.range_constexpr(_SCAN_UNROLL):
                        idx = idx_base + Int32(scan_u * _THREADS_PER_CTA)
                        raw_input = _load_value_virtual(
                            input_tensor,
                            carry_values,
                            row_base,
                            row_start,
                            out_base,
                            length,
                            idx,
                            self.block_q,
                            self.block_k,
                            self.is_tiled,
                            self.is_first,
                        )
                        bin8 = _convert_to_uint8(raw_input)
                        if Int32(bin8) > threshold_bin:
                            pos = _smem_xadd(ctr, Int32(0), Int32(1))
                            s_out[pos] = idx
                        else:
                            if Int32(bin8) == threshold_bin:
                                cand_pos = _smem_xadd(ni0, Int32(0), Int32(1))
                                if cand_pos < Int32(_SMEM_CANDS):
                                    s_cand0[cand_pos] = idx
                                    key32 = _convert_to_uint32(raw_input)
                                    sub_bin = (key32 >> Uint32(24)) & Uint32(0xFF)
                                    _smem_red_add(h0, Int32(sub_bin), Int32(1))
                    idx_base = idx_base + Int32(_THREADS_PER_CTA * _SCAN_UNROLL)
                while idx_base < total_len:
                    raw_input = _load_value_virtual(
                        input_tensor,
                        carry_values,
                        row_base,
                        row_start,
                        out_base,
                        length,
                        idx_base,
                        self.block_q,
                        self.block_k,
                        self.is_tiled,
                        self.is_first,
                    )
                    bin8 = _convert_to_uint8(raw_input)
                    if Int32(bin8) > threshold_bin:
                        pos = _smem_xadd(ctr, Int32(0), Int32(1))
                        s_out[pos] = idx_base
                    else:
                        if Int32(bin8) == threshold_bin:
                            cand_pos = _smem_xadd(ni0, Int32(0), Int32(1))
                            if cand_pos < Int32(_SMEM_CANDS):
                                s_cand0[cand_pos] = idx_base
                                key32 = _convert_to_uint32(raw_input)
                                sub_bin = (key32 >> Uint32(24)) & Uint32(0xFF)
                                _smem_red_add(h0, Int32(sub_bin), Int32(1))
                    idx_base = idx_base + Int32(_THREADS_PER_CTA)

                cute.arch.sync_threads()

                # Stage 2: refine with 8-bit radix passes
                for round_idx in cutlass.range_constexpr(4):
                    if topk != Int32(-1):
                        r_idx_is_0 = (round_idx % 2) == 0
                        r_idx_next_is_0 = not r_idx_is_0

                        raw_num_input = (
                            _smem_ld(ni0, Int32(0))
                            if cutlass.const_expr(r_idx_is_0)
                            else _smem_ld(ni1, Int32(0))
                        )
                        num_input = (
                            raw_num_input
                            if raw_num_input < Int32(_SMEM_CANDS)
                            else Int32(_SMEM_CANDS)
                        )

                        # Prefix sum
                        for stage in cutlass.range_constexpr(8):
                            j = Int32(1 << stage)
                            if tx < Int32(256):
                                if (stage & 1) == 0:
                                    value = Int32(s_hist0[tx])
                                    if tx < Int32(256) - j:
                                        value = value + Int32(s_hist0[tx + j])
                                    s_hist1[tx] = value
                                else:
                                    value = Int32(s_hist1[tx])
                                    if tx < Int32(256) - j:
                                        value = value + Int32(s_hist1[tx + j])
                                    s_hist0[tx] = value
                            cute.arch.sync_threads()

                        if tx < Int32(256):
                            val_tx = Int32(s_hist0[tx])
                            val_tx1 = Int32(s_hist0[tx + Int32(1)])
                            if val_tx > topk:
                                if val_tx1 <= topk:
                                    _smem_st(thr, Int32(0), Int32(tx))
                                    if cutlass.const_expr(r_idx_next_is_0):
                                        _smem_st(ni0, Int32(0), Int32(0))
                                    else:
                                        _smem_st(ni1, Int32(0), Int32(0))
                                    _smem_st(lr, Int32(0), topk - val_tx1)

                        cute.arch.sync_threads()

                        sub_threshold = _smem_ld(thr, Int32(0))
                        topk = topk - Int32(s_hist0[sub_threshold + Int32(1)])

                        # Quick exit
                        if topk == Int32(0):
                            i = Int32(tx)
                            while i < num_input:
                                c_idx = (
                                    Int32(s_cand0[i])
                                    if cutlass.const_expr(r_idx_is_0)
                                    else Int32(s_cand1[i])
                                )
                                offset = Int32(24 - round_idx * 8)
                                raw_val = _load_value_virtual(
                                    input_tensor,
                                    carry_values,
                                    row_base,
                                    row_start,
                                    out_base,
                                    length,
                                    c_idx,
                                    self.block_q,
                                    self.block_k,
                                    self.is_tiled,
                                    self.is_first,
                                )
                                key32 = _convert_to_uint32(raw_val)
                                bin = (key32 >> Uint32(offset)) & Uint32(0xFF)
                                if Int32(bin) > sub_threshold:
                                    pos = _smem_xadd(ctr, Int32(0), Int32(1))
                                    s_out[pos] = c_idx
                                i = i + Int32(_THREADS_PER_CTA)
                            topk = Int32(-1)

                        # Continue refinement
                        if topk != Int32(-1):
                            cute.arch.sync_threads()

                            if tx < Int32(257):
                                s_hist0[tx] = Int32(0)
                            cute.arch.sync_threads()

                            i = Int32(tx)
                            while i < num_input:
                                c_idx = (
                                    Int32(s_cand0[i])
                                    if cutlass.const_expr(r_idx_is_0)
                                    else Int32(s_cand1[i])
                                )
                                raw_val = _load_value_virtual(
                                    input_tensor,
                                    carry_values,
                                    row_base,
                                    row_start,
                                    out_base,
                                    length,
                                    c_idx,
                                    self.block_q,
                                    self.block_k,
                                    self.is_tiled,
                                    self.is_first,
                                )
                                offset = Int32(24 - round_idx * 8)
                                key32 = _convert_to_uint32(raw_val)
                                bin = (key32 >> Uint32(offset)) & Uint32(0xFF)

                                if Int32(bin) > sub_threshold:
                                    pos = _smem_xadd(ctr, Int32(0), Int32(1))
                                    s_out[pos] = c_idx
                                else:
                                    if Int32(bin) == sub_threshold:
                                        if cutlass.const_expr(round_idx == 3):
                                            old_rem = _smem_xadd(
                                                lr, Int32(0), Int32(-1)
                                            )
                                            if old_rem > Int32(0):
                                                s_out[topk_static - old_rem] = c_idx
                                        else:
                                            cand_pos = (
                                                _smem_xadd(ni0, Int32(0), Int32(1))
                                                if cutlass.const_expr(r_idx_next_is_0)
                                                else _smem_xadd(ni1, Int32(0), Int32(1))
                                            )
                                            if cand_pos < Int32(_SMEM_CANDS):
                                                if cutlass.const_expr(r_idx_next_is_0):
                                                    s_cand0[cand_pos] = c_idx
                                                else:
                                                    s_cand1[cand_pos] = c_idx
                                                sub_bin = (
                                                    key32
                                                    >> Uint32(24 - (round_idx + 1) * 8)
                                                ) & Uint32(0xFF)
                                                _smem_red_add(
                                                    h0, Int32(sub_bin), Int32(1)
                                                )

                                i = i + Int32(_THREADS_PER_CTA)

                            cute.arch.sync_threads()

            cute.arch.sync_threads()
            idx0 = Int32(tx)
            if idx0 < topk_static:
                selected0 = Int32(s_out[idx0])
                values[out_base + idx0] = _load_value_virtual(
                    input_tensor,
                    carry_values,
                    row_base,
                    row_start,
                    out_base,
                    length,
                    selected0,
                    self.block_q,
                    self.block_k,
                    self.is_tiled,
                    self.is_first,
                )
                indices[out_base + idx0] = _emit_global_index_virtual(
                    carry_indices,
                    output_page_table,
                    row_start,
                    output_index_offset,
                    out_base,
                    length,
                    selected0,
                    bid,
                    output_page_size,
                    self.is_first,
                    self.output_physical_slots,
                )
            idx1 = idx0 + Int32(_THREADS_PER_CTA)
            if idx1 < topk_static:
                selected1 = Int32(s_out[idx1])
                values[out_base + idx1] = _load_value_virtual(
                    input_tensor,
                    carry_values,
                    row_base,
                    row_start,
                    out_base,
                    length,
                    selected1,
                    self.block_q,
                    self.block_k,
                    self.is_tiled,
                    self.is_first,
                )
                indices[out_base + idx1] = _emit_global_index_virtual(
                    carry_indices,
                    output_page_table,
                    row_start,
                    output_index_offset,
                    out_base,
                    length,
                    selected1,
                    bid,
                    output_page_size,
                    self.is_first,
                    self.output_physical_slots,
                )


@lru_cache(maxsize=64)
def _build_tiled_topk_kernel(
    block_q: int,
    block_k: int,
    topk: int,
    zero_row_start: bool = False,
    is_first: bool = True,
    output_physical_slots: bool = False,
    extent_splits: int = 1,
):
    return SparseNSATiledTopkKernel(
        is_tiled=True,
        block_q=block_q,
        block_k=block_k,
        topk=topk,
        zero_row_start=zero_row_start,
        is_first=is_first,
        output_physical_slots=output_physical_slots,
        extent_splits=extent_splits,
    )


@lru_cache(maxsize=8)
def _build_row_topk_kernel(topk: int, output_physical_slots: bool = False):
    return SparseNSATiledTopkKernel(
        is_tiled=False,
        block_q=1,
        block_k=1,
        topk=topk,
        zero_row_start=True,
        output_physical_slots=output_physical_slots,
    )


def clear_tiled_topk_kernel_cache() -> None:
    _build_tiled_topk_kernel.cache_clear()
    _build_row_topk_kernel.cache_clear()


def _validate_supported_topk(topk: int, *, caller: str) -> int:
    topk = int(topk)
    if topk not in _SUPPORTED_TOPK:
        raise ValueError(
            f"{caller} supports topk values {_SUPPORTED_TOPK}, got topk={topk}"
        )
    return topk


def run_tiled_topk(
    *,
    tile_logits: torch.Tensor,
    k_start: torch.Tensor | None,
    k_end: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    topk: int,
    block_q: int,
    block_k: int,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    num_k_tiles: int | None = None,
    tile_k_offset: int = 0,
    input_index_offset: int = 0,
    input_extent: int = 0,
    output_index_offset: int = 0,
    zero_row_start: bool = False,
    carry_values: torch.Tensor | None = None,
    carry_indices: torch.Tensor | None = None,
    is_first: bool = True,
    output_page_table: torch.Tensor | None = None,
    output_page_size: int = 64,
    extent_splits: int = 1,
    output_row_stride: int | None = None,
    output_row_base: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk = _validate_supported_topk(topk, caller="run_tiled_topk")
    extent_splits = int(extent_splits)
    if extent_splits > 1:
        # Level-1 of the two-level fold: each pseudo-row selects one extent
        # slice; slices are folded by a second linear pass. Slice windows ride
        # the chunk-clip logic, so a per-slice input_extent is required.
        if int(input_extent) <= 0:
            raise ValueError("extent_splits > 1 requires a per-slice input_extent")
        if not is_first or carry_values is not None or carry_indices is not None:
            raise ValueError("extent_splits > 1 does not fold carries")
        if output_page_table is not None:
            raise ValueError("extent_splits > 1 emits logical indices only")
    if k_end is None and lengths is None:
        raise ValueError("run_tiled_topk requires either k_end or lengths")
    if not tile_logits.is_contiguous():
        raise ValueError("tile_logits must be contiguous")
    if k_start is None:
        if not zero_row_start:
            raise ValueError(
                "run_tiled_topk requires k_start unless zero_row_start is true"
            )
        if lengths is None:
            raise ValueError("run_tiled_topk zero_row_start requires explicit lengths")
        k_start = lengths
    if not k_start.is_contiguous():
        raise ValueError("k_start must be contiguous")
    if lengths is None:
        if k_end is None:
            raise AssertionError("unreachable")
        lengths = k_end - k_start
    elif not lengths.is_contiguous():
        raise ValueError("lengths must be contiguous")

    num_q_rows = int(k_start.shape[0])
    if output_row_stride is None:
        output_row_stride = extent_splits
    output_row_stride = int(output_row_stride)
    output_row_base = int(output_row_base)
    if output_row_base + extent_splits > output_row_stride:
        raise ValueError(
            "output_row_base + extent_splits must fit within output_row_stride, "
            f"got {output_row_base} + {extent_splits} > {output_row_stride}"
        )
    out_rows = num_q_rows * output_row_stride
    output_physical_slots = output_page_table is not None
    if output_physical_slots:
        assert output_page_table is not None
        if output_page_table.ndim != 2:
            raise ValueError(
                "output_page_table must be rank-2, got "
                f"{tuple(output_page_table.shape)}"
            )
        if int(output_page_table.shape[0]) != num_q_rows:
            raise ValueError(
                "output_page_table rows must match top-k rows, got "
                f"{int(output_page_table.shape[0])} != {num_q_rows}"
            )
        if output_page_table.dtype != torch.int32:
            raise ValueError(
                "output_page_table must have dtype torch.int32, got "
                f"{output_page_table.dtype}"
            )
        if output_page_table.device != tile_logits.device:
            raise ValueError("output_page_table device must match tile_logits")
        if int(output_page_table.stride(1)) != 1 or not (
            output_page_table.is_contiguous() or int(output_page_table.stride(0)) == 0
        ):
            raise ValueError(
                "output_page_table must be contiguous or a row-shared stride-0 view"
            )
        if int(output_page_size) <= 0:
            raise ValueError(
                f"output_page_size must be positive, got {output_page_size}"
            )
        output_page_table_for_kernel = output_page_table
    else:
        # The physical mapping branch is constexpr-elided. Reuse row metadata as
        # its dummy tensor so ordinary tiled top-k remains allocation-free.
        output_page_table_for_kernel = lengths
        output_page_size = 1
    num_q_tiles = (num_q_rows + block_q - 1) // block_q
    tile_size = block_q * block_k
    total_elements = int(tile_logits.shape[0])
    if num_k_tiles is None:
        num_k_tiles = total_elements // (num_q_tiles * tile_size)
        if num_k_tiles == 0:
            num_k_tiles = getattr(tile_logits, "_sm12x_num_k_tiles", None)
            if num_k_tiles is None:
                raise ValueError("Cannot determine num_k_tiles")
    if int(num_k_tiles) <= 0:
        raise ValueError("Cannot determine num_k_tiles")

    if output_indices is None:
        topk_indices = torch.empty(
            (out_rows, topk),
            dtype=torch.int32,
            device=tile_logits.device,
        )
    else:
        if output_indices.shape != (out_rows, topk):
            raise ValueError(
                f"output_indices must have shape {(out_rows, topk)}, got {tuple(output_indices.shape)}"
            )
        if not output_indices.is_contiguous():
            raise ValueError("output_indices must be contiguous")
        topk_indices = output_indices
    if output_values is None:
        topk_values = torch.empty(
            (out_rows, topk),
            dtype=torch.float32,
            device=tile_logits.device,
        )
    else:
        if output_values.shape != (out_rows, topk):
            raise ValueError(
                f"output_values must have shape {(out_rows, topk)}, got {tuple(output_values.shape)}"
            )
        if not output_values.is_contiguous():
            raise ValueError("output_values must be contiguous")
        topk_values = output_values

    # Carry buffers hold the previous supertile chunk's running top-k (value, global
    # index) that this launch folds in. When is_first there is no predecessor, so the
    # kernel never reads them (the read path is constexpr-elided) — but a real tensor
    # of the right shape must still be passed, so allocate a throwaway if none given.
    if not is_first:
        if carry_values is None or carry_indices is None:
            raise ValueError(
                "run_tiled_topk requires carry_values and carry_indices when is_first is False"
            )
        if carry_values.shape != (num_q_rows, topk):
            raise ValueError(
                f"carry_values must have shape {(num_q_rows, topk)}, got {tuple(carry_values.shape)}"
            )
        if carry_indices.shape != (num_q_rows, topk):
            raise ValueError(
                f"carry_indices must have shape {(num_q_rows, topk)}, got {tuple(carry_indices.shape)}"
            )
        if carry_values.dtype != torch.float32:
            raise ValueError(f"carry_values must be float32, got {carry_values.dtype}")
        if carry_indices.dtype != torch.int32:
            raise ValueError(f"carry_indices must be int32, got {carry_indices.dtype}")
        if not carry_values.is_contiguous() or not carry_indices.is_contiguous():
            raise ValueError("carry_values and carry_indices must be contiguous")
        if (
            carry_values.device != tile_logits.device
            or carry_indices.device != tile_logits.device
        ):
            raise ValueError("carry buffers must be on the tile_logits device")
        # carry-IN is read throughout selection; the output (carry-OUT for non-final
        # chunks) is written. They must be physically distinct to avoid an
        # intra-launch read/write race — the orchestrator ping-pongs two buffers.
        if carry_values.data_ptr() == topk_values.data_ptr():
            raise ValueError("carry_values must not alias the output values buffer")
        if carry_indices.data_ptr() == topk_indices.data_ptr():
            raise ValueError("carry_indices must not alias the output indices buffer")
    else:
        # is_first: carry is never read (the read path is constexpr-elided). Reuse the
        # output buffers as the throwaway carry tensors so the hot single-chunk /
        # first-chunk path allocates nothing (safe under CUDA-graph capture). The
        # kernel only writes `values`/`indices`, so aliasing them as the unread carry
        # is harmless.
        if carry_values is None:
            carry_values = topk_values
        if carry_indices is None:
            carry_indices = topk_indices

    input_stride = Int32(0)
    flat_input = tile_logits.reshape(-1)
    flat_values = topk_values.reshape(-1).contiguous()
    flat_indices = topk_indices.reshape(-1).contiguous()
    flat_carry_values = carry_values.reshape(-1).contiguous()
    flat_carry_indices = carry_indices.reshape(-1).contiguous()

    kernel = _build_tiled_topk_kernel(
        block_q,
        block_k,
        topk,
        bool(zero_row_start),
        bool(is_first),
        bool(output_physical_slots),
        extent_splits,
    )
    input_key_tensor = tile_logits
    lengths_key_tensor = lengths
    row_start_key_tensor = k_start
    if zero_row_start:
        row_start_key_tensor = lengths_key_tensor
    values_key_tensor = topk_values
    indices_key_tensor = topk_indices
    carry_values_key_tensor = carry_values
    carry_indices_key_tensor = carry_indices
    output_page_table_key_tensor = output_page_table_for_kernel
    args = (
        _to_kernel_tensor(flat_input, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(k_start, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(flat_values, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(flat_indices, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(flat_carry_values, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(flat_carry_indices, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(output_page_table_for_kernel, cutlass.Int32, assumed_align=4),
        Int32(num_q_rows * extent_splits),
        input_stride,
        Int32(num_k_tiles),
        Int32(tile_k_offset),
        Int32(block_q),
        Int32(block_k),
        Int32(topk),
        Int32(input_index_offset),
        Int32(input_extent),
        Int32(output_index_offset),
        Int32(output_page_size),
        Int32(output_row_stride),
        Int32(output_row_base),
        current_cuda_stream(),
    )
    cache_key = (
        _flat_tensor_compile_key("input", input_key_tensor, dynamic=True),
        _tensor_compile_key("row_start", row_start_key_tensor, dynamic_dims=(0,)),
        _tensor_compile_key("lengths", lengths_key_tensor, dynamic_dims=(0,)),
        _flat_tensor_compile_key("topk_values", values_key_tensor, dynamic=True),
        _flat_tensor_compile_key("topk_indices", indices_key_tensor, dynamic=True),
        _flat_tensor_compile_key("carry_values", carry_values_key_tensor, dynamic=True),
        _flat_tensor_compile_key(
            "carry_indices", carry_indices_key_tensor, dynamic=True
        ),
        _tensor_compile_key(
            "output_page_table",
            output_page_table_key_tensor,
            dynamic_dims=(0,),
        ),
        (
            "tiled_topk_v21",
            topk,
            block_q,
            block_k,
            bool(zero_row_start),
            bool(is_first),
            bool(output_physical_slots),
            extent_splits,
        ),
    )
    compile_spec = KernelCompileSpec.from_key(
        "attention.indexer.tiled_topk",
        4,
        cache_key,
        labels=(
            "input",
            "row_start",
            "lengths",
            "topk_values",
            "topk_indices",
            "carry_values",
            "carry_indices",
            "output_page_table",
            "policy",
        ),
    )
    sm12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )
    return topk_values, topk_indices


def run_row_topk(
    *,
    row_logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    output_index_offset: int = 0,
    output_gather_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact row-wise topk over a dense row-major logits tile.

    output_gather_table: optional int32 (rows, width) table; selected logical
    positions are remapped through it (out = table[row, pos]), turning the
    pass into a fold over (value, index) candidate pairs. Padded candidates
    (-inf value, -1 index) propagate -1 naturally.
    """
    topk = _validate_supported_topk(topk, caller="run_row_topk")
    if row_logits.ndim != 2:
        raise ValueError(f"row_logits must be rank-2, got {tuple(row_logits.shape)}")
    if not row_logits.is_contiguous():
        raise ValueError("row_logits must be contiguous")
    if row_logits.dtype != torch.float32:
        raise ValueError(
            f"row_logits must have dtype torch.float32, got {row_logits.dtype}"
        )
    if lengths.ndim != 1:
        raise ValueError(f"lengths must be rank-1, got {tuple(lengths.shape)}")
    if lengths.dtype != torch.int32:
        raise ValueError(f"lengths must have dtype torch.int32, got {lengths.dtype}")
    if not lengths.is_contiguous():
        raise ValueError("lengths must be contiguous")
    num_q_rows = int(row_logits.shape[0])
    width = int(row_logits.shape[1])
    if lengths.shape[0] != num_q_rows:
        raise ValueError(
            f"lengths rows {lengths.shape[0]} do not match row_logits rows {num_q_rows}"
        )

    if output_indices is None:
        topk_indices = torch.empty(
            (num_q_rows, topk),
            dtype=torch.int32,
            device=row_logits.device,
        )
    else:
        if output_indices.shape != (num_q_rows, topk):
            raise ValueError(
                f"output_indices must have shape {(num_q_rows, topk)}, got {tuple(output_indices.shape)}"
            )
        if not output_indices.is_contiguous():
            raise ValueError("output_indices must be contiguous")
        topk_indices = output_indices
    if output_values is None:
        topk_values = torch.empty(
            (num_q_rows, topk),
            dtype=torch.float32,
            device=row_logits.device,
        )
    else:
        if output_values.shape != (num_q_rows, topk):
            raise ValueError(
                f"output_values must have shape {(num_q_rows, topk)}, got {tuple(output_values.shape)}"
            )
        if not output_values.is_contiguous():
            raise ValueError("output_values must be contiguous")
        topk_values = output_values

    flat_input = row_logits.reshape(-1)
    flat_values = topk_values.reshape(-1).contiguous()
    flat_indices = topk_indices.reshape(-1).contiguous()
    # Row top-k is always a single-chunk (is_first) selection. The carry path is
    # constexpr-elided, so alias the unread carry tensors to the outputs instead
    # of allocating throwaways inside graph-captured serving paths.
    if output_gather_table is not None:
        if output_gather_table.shape != (num_q_rows, width):
            raise ValueError(
                f"output_gather_table must have shape {(num_q_rows, width)}, "
                f"got {tuple(output_gather_table.shape)}"
            )
        if output_gather_table.dtype != torch.int32:
            raise ValueError("output_gather_table must be int32")
        if not output_gather_table.is_contiguous():
            raise ValueError("output_gather_table must be contiguous")
        if output_gather_table.device != row_logits.device:
            raise ValueError("output_gather_table must be on the logits device")
    carry_values = topk_values
    carry_indices = topk_indices
    flat_carry_values = carry_values.reshape(-1)
    flat_carry_indices = carry_indices.reshape(-1)
    kernel = _build_row_topk_kernel(topk, output_gather_table is not None)
    input_key_tensor = row_logits
    lengths_key_tensor = lengths
    values_key_tensor = topk_values
    indices_key_tensor = topk_indices
    carry_values_key_tensor = carry_values
    carry_indices_key_tensor = carry_indices
    args = (
        _to_kernel_tensor(flat_input, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(lengths, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(flat_values, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(flat_indices, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(flat_carry_values, cutlass.Float32, assumed_align=4),
        _to_kernel_tensor(flat_carry_indices, cutlass.Int32, assumed_align=4),
        _to_kernel_tensor(
            output_gather_table if output_gather_table is not None else lengths,
            cutlass.Int32,
            assumed_align=4,
        ),
        Int32(num_q_rows),
        Int32(width),
        Int32(0),
        Int32(0),
        Int32(1),
        Int32(1),
        Int32(topk),
        Int32(0),
        Int32(width),
        Int32(output_index_offset),
        Int32(1),
        Int32(1),
        Int32(0),
        current_cuda_stream(),
    )
    cache_key = (
        _flat_tensor_compile_key("logits", input_key_tensor, dynamic=True),
        _tensor_compile_key("lengths", lengths_key_tensor, dynamic_dims=(0,)),
        _flat_tensor_compile_key("topk_values", values_key_tensor, dynamic=True),
        _flat_tensor_compile_key("topk_indices", indices_key_tensor, dynamic=True),
        _flat_tensor_compile_key("carry_values", carry_values_key_tensor, dynamic=True),
        _flat_tensor_compile_key(
            "carry_indices", carry_indices_key_tensor, dynamic=True
        ),
        (
            "row_topk_v5",
            topk,
            output_gather_table is not None,
        ),
    )
    compile_spec = KernelCompileSpec.from_key(
        "attention.indexer.row_topk",
        4,
        cache_key,
        labels=(
            "logits",
            "lengths",
            "topk_values",
            "topk_indices",
            "carry_values",
            "carry_indices",
            "policy",
        ),
    )
    sm12x_launch(
        kernel,
        compile_spec=compile_spec,
        compile_args=args,
        runtime_args=args,
    )
    return topk_values, topk_indices


def _resolve_supertile_k(supertile_k: int | None, *, block_k: int) -> int:
    if supertile_k is None:
        raw = os.environ.get(_SUPERTILE_K_ENV)
        if raw is None:
            supertile_k = _SUPERTILE_K_DEFAULT
        else:
            try:
                supertile_k = int(raw)
            except ValueError as exc:
                raise ValueError(
                    f"{_SUPERTILE_K_ENV} must be an integer, got {raw!r}"
                ) from exc
    supertile_k = max(int(supertile_k), int(block_k))
    return ((supertile_k + block_k - 1) // block_k) * block_k


def merge_tiled_topk_candidates(
    *,
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    topk: int,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    merge_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge per-supertile exact topk candidates into one row-wise topk."""
    if candidate_values.shape != candidate_indices.shape:
        raise ValueError(
            "candidate_values and candidate_indices must have the same shape, got "
            f"{tuple(candidate_values.shape)} vs {tuple(candidate_indices.shape)}"
        )
    if candidate_values.ndim != 3:
        raise ValueError(
            f"candidates must have shape (chunks, rows, topk), got {tuple(candidate_values.shape)}"
        )
    num_chunks, num_q_rows, local_topk = candidate_values.shape
    if int(local_topk) != int(topk):
        raise ValueError(
            f"candidate local topk {local_topk} does not match requested topk {topk}"
        )
    candidate_cols = int(num_chunks) * int(topk)
    candidate_values_2d = candidate_values.permute(1, 0, 2).reshape(
        num_q_rows, candidate_cols
    )
    candidate_indices_2d = candidate_indices.permute(1, 0, 2).reshape(
        num_q_rows, candidate_cols
    )
    if output_values is None:
        topk_values = torch.empty(
            (num_q_rows, topk),
            dtype=candidate_values.dtype,
            device=candidate_values.device,
        )
    else:
        if (
            output_values.ndim != 2
            or output_values.shape[0] < num_q_rows
            or output_values.shape[1] < topk
        ):
            raise ValueError(
                "output_values must have shape at least "
                f"({num_q_rows}, {topk}), got {tuple(output_values.shape)}"
            )
        if output_values.dtype != candidate_values.dtype:
            raise ValueError(
                f"output_values must have dtype {candidate_values.dtype}, got {output_values.dtype}"
            )
        if output_values.device != candidate_values.device:
            raise ValueError("output_values device must match candidate_values")
        topk_values = output_values[:num_q_rows, :topk]

    if merge_positions is None:
        merge_pos = torch.empty(
            (num_q_rows, topk),
            dtype=torch.int64,
            device=candidate_values.device,
        )
    else:
        if (
            merge_positions.ndim != 2
            or merge_positions.shape[0] < num_q_rows
            or merge_positions.shape[1] < topk
        ):
            raise ValueError(
                "merge_positions must have shape at least "
                f"({num_q_rows}, {topk}), got {tuple(merge_positions.shape)}"
            )
        if merge_positions.dtype != torch.int64:
            raise ValueError(
                f"merge_positions must have dtype torch.int64, got {merge_positions.dtype}"
            )
        if merge_positions.device != candidate_values.device:
            raise ValueError("merge_positions device must match candidate_values")
        if not merge_positions.is_contiguous():
            raise ValueError("merge_positions must be contiguous")
        merge_pos = merge_positions[:num_q_rows, :topk]

    torch.topk(
        candidate_values_2d,
        k=topk,
        dim=1,
        largest=True,
        sorted=False,
        out=(topk_values, merge_pos),
    )

    if output_indices is None:
        topk_indices = torch.gather(candidate_indices_2d, 1, merge_pos).contiguous()
    else:
        if (
            output_indices.ndim != 2
            or output_indices.shape[0] < num_q_rows
            or output_indices.shape[1] < topk
        ):
            raise ValueError(
                "output_indices must have shape at least "
                f"({num_q_rows}, {topk}), got {tuple(output_indices.shape)}"
            )
        if output_indices.dtype != candidate_indices.dtype:
            raise ValueError(
                f"output_indices must have dtype {candidate_indices.dtype}, got {output_indices.dtype}"
            )
        if output_indices.device != candidate_indices.device:
            raise ValueError("output_indices device must match candidate_indices")
        topk_indices = output_indices[:num_q_rows, :topk]
        torch.gather(candidate_indices_2d, 1, merge_pos, out=topk_indices)
    return topk_values, topk_indices


def run_tiled_supertile_topk(
    *,
    tile_logits: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    topk: int,
    block_q: int,
    block_k: int,
    supertile_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact topk over tiled logits by local-selecting K supertiles then merging candidates."""
    topk = _validate_supported_topk(topk, caller="run_tiled_supertile_topk")
    if not tile_logits.is_contiguous():
        raise ValueError("tile_logits must be contiguous")
    if not k_start.is_contiguous() or not k_end.is_contiguous():
        raise ValueError("k_start and k_end must be contiguous")

    num_q_rows = int(k_start.shape[0])
    num_q_tiles = (num_q_rows + block_q - 1) // block_q
    tile_size = block_q * block_k
    total_elements = int(tile_logits.shape[0])
    num_k_tiles = total_elements // (num_q_tiles * tile_size)
    if num_k_tiles == 0:
        num_k_tiles = getattr(tile_logits, "_sm12x_num_k_tiles", None)
        if num_k_tiles is None:
            raise ValueError("Cannot determine num_k_tiles")
    resolved_supertile_k = _resolve_supertile_k(supertile_k, block_k=block_k)
    supertile_tiles = max(1, resolved_supertile_k // block_k)
    num_chunks = (int(num_k_tiles) + supertile_tiles - 1) // supertile_tiles
    if num_chunks <= 1:
        return run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            k_end=k_end,
            topk=topk,
            block_q=block_q,
            block_k=block_k,
        )

    # Streaming fold: each chunk folds the previous chunk's running top-k (carry) into
    # its own selection. Two (M, topk) carry halves ping-pong (read prev, write next);
    # the final chunk writes the user output. No (num_chunks, ...) slab, no merge.
    carry_buf_values = torch.empty(
        (2, num_q_rows, topk), dtype=torch.float32, device=tile_logits.device
    )
    carry_buf_indices = torch.empty(
        (2, num_q_rows, topk), dtype=torch.int32, device=tile_logits.device
    )
    out_values = torch.empty(
        (num_q_rows, topk), dtype=torch.float32, device=tile_logits.device
    )
    out_indices = torch.empty(
        (num_q_rows, topk), dtype=torch.int32, device=tile_logits.device
    )
    global_lengths = (k_end - k_start).contiguous()

    for chunk_idx in range(num_chunks):
        chunk_tile_begin = chunk_idx * supertile_tiles
        chunk_tile_end = min(chunk_tile_begin + supertile_tiles, int(num_k_tiles))
        chunk_start = chunk_tile_begin * block_k
        chunk_rows = (chunk_tile_end - chunk_tile_begin) * block_k
        is_first = chunk_idx == 0
        is_last = chunk_idx == num_chunks - 1
        carry_values = carry_buf_values[(chunk_idx - 1) % 2]
        carry_indices = carry_buf_indices[(chunk_idx - 1) % 2]
        out_v = out_values if is_last else carry_buf_values[chunk_idx % 2]
        out_i = out_indices if is_last else carry_buf_indices[chunk_idx % 2]
        run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            lengths=global_lengths,
            topk=topk,
            block_q=block_q,
            block_k=block_k,
            output_values=out_v,
            output_indices=out_i,
            num_k_tiles=int(num_k_tiles),
            tile_k_offset=chunk_tile_begin,
            input_index_offset=chunk_start,
            input_extent=chunk_rows,
            output_index_offset=chunk_start,
            carry_values=carry_values,
            carry_indices=carry_indices,
            is_first=is_first,
        )

    return out_values, out_indices
