"""Paged QSA representative scoring with device-side context bounds.

Inactive CTAs initialize their score range without loading query or cache data.
Bounded execution visits only the live prefix with an SM-sized grid. Indices-only
execution leaves scores unwritten when all eligible groups fit in the selection.
BF16 tensor cores score 64 groups per CTA for up to eight index heads and
16-aligned head dimensions. Other geometries use scalar FP32 reductions over
32 groups per CTA. No live extent enters compilation keys.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float8E4M3FN, Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import (
    compile_only_launches_enabled,
    program_keys,
    record_program,
)
from b12x._lib.program_cache import register_program_cache
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.intrinsics.packed import f32_to_raw_bits
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

_CACHE: dict[tuple, object] = {}
register_program_cache(_CACHE)
_THREADS = 128


class _RepresentativeScoreKernel:
    def __init__(
        self,
        heads,
        dim,
        ratio,
        page_size,
        max_groups,
        budget,
        dcp_size,
        dcp_rank,
        cp_interleave,
        weighted=False,
        fp8=False,
        position_offset=1,
        workers=0,
        indices_only=False,
    ):
        self.heads = heads
        self.dim = dim
        self.ratio = ratio
        self.page_size = page_size
        self.max_groups = max_groups
        self.budget = budget
        self.dcp_size = dcp_size
        self.dcp_rank = dcp_rank
        self.group_interleave = cp_interleave // ratio
        self.lane_values = (dim + 31) // 32
        self.scale = 1.0 / math.sqrt(dim)
        self.weighted = weighted
        self.fp8 = fp8
        self.position_offset = position_offset
        self.workers = workers
        self.indices_only = indices_only
        self.use_mma = heads <= 8 and dim % 16 == 0
        self.groups = 64 if self.use_mma else 32

    @cute.jit
    def load_pair(self, pointer, offset):
        value = Uint32(0)
        if cutlass.const_expr(self.fp8):
            lo = f32_to_raw_bits(pointer[offset].to(Float32)) >> 16
            hi = f32_to_raw_bits(pointer[offset + Int64(1)].to(Float32))
            value = (lo | hi).to(Uint32)
        else:
            address = pointer + offset
            if (address.toint().to(Int64) & Int64(3)) == Int64(0):
                value = cute.recast_ptr(address.align(4), dtype=Uint32)[0].to(Uint32)
            else:
                # Odd storage offsets and token strides remain supported.
                lo = f32_to_raw_bits(pointer[offset].to(Float32)) >> 16
                hi = f32_to_raw_bits(pointer[offset + Int64(1)].to(Float32))
                value = (lo | hi).to(Uint32)
        return value

    @cute.jit
    def score_mma(
        self,
        query,
        weights,
        cache,
        table,
        scores,
        request,
        row,
        page_stride,
        token_stride,
        table_stride,
        score_row,
        group_offset,
        group_count,
        local_start,
        eligible,
        carry,
        lane,
        warp,
    ):
        # m16n8k16: A rows are representatives; B columns are index heads.
        # Each four-lane subgroup owns two representative rows in C.
        matrix_row = lane // 4
        matrix_pair = lane % 4
        warp_start = local_start + warp * 16
        if group_offset + warp_start < eligible:
            base = cute.make_rmem_tensor((2,), Int64)
            valid = cute.make_rmem_tensor((2,), Int32)
            for part in cutlass.range_constexpr(2):
                local_group = warp_start + matrix_row + part * 8
                group = group_offset + local_group
                base[part] = Int64(0)
                valid[part] = Int32(0)
                if (local_group < group_count) & (group < eligible):
                    page = table[
                        request * table_stride + (group // self.page_size).to(Int64)
                    ].to(Int64)
                    if page >= Int64(0):
                        base[part] = (
                            page * page_stride
                            + (group % self.page_size).to(Int64) * token_stride
                        )
                        valid[part] = Int32(1)
            d0, d1, d2, d3 = Float32(0), Float32(0), Float32(0), Float32(0)
            for tile in cutlass.range_constexpr(self.dim // 16):
                mma_column = (tile * 16 + matrix_pair * 2).to(Int64)
                a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if valid[0] != 0:
                    a0 = self.load_pair(cache, base[0] + mma_column)
                    a2 = self.load_pair(cache, base[0] + mma_column + Int64(8))
                if valid[1] != 0:
                    a1 = self.load_pair(cache, base[1] + mma_column)
                    a3 = self.load_pair(cache, base[1] + mma_column + Int64(8))
                b0, b1 = Uint32(0), Uint32(0)
                if matrix_row < self.heads:
                    query_base = (
                        row.to(Int64) * self.heads * self.dim
                        + matrix_row.to(Int64) * self.dim
                        + mma_column
                    )
                    b0 = self.load_pair(query, query_base)
                    b1 = self.load_pair(query, query_base + Int64(8))
                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                    d0,
                    d1,
                    d2,
                    d3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                )
            for part in cutlass.range_constexpr(2):
                first = d0 if part == 0 else d2
                second = d1 if part == 0 else d3
                first = cutlass.max(first, Float32(0))
                second = cutlass.max(second, Float32(0))
                score = Float32(0)
                # Preserve head order, including zero-padded MMA columns.
                for head in cutlass.range_constexpr(self.heads):
                    value = first if head % 2 == 0 else second
                    value = cute.arch.shuffle_sync(
                        value,
                        (lane // 4) * 4 + head // 2,
                    )
                    if cutlass.const_expr(self.weighted):
                        value *= weights[row.to(Int64) * self.heads + head]
                    score = score + value
                if cutlass.const_expr(not self.weighted):
                    score = score * Float32(self.scale)
                if valid[part] == 0:
                    score = Float32(-float("inf"))
                local_group = warp_start + matrix_row + part * 8
                if (matrix_pair == 0) & (local_group < group_count):
                    scores[score_row + (carry + local_group).to(Int64)] = score
        else:
            local_group = warp_start + lane
            if (lane < 16) & (local_group < group_count):
                scores[score_row + (carry + local_group).to(Int64)] = Float32(
                    -float("inf")
                )

    @cute.jit
    def __call__(
        self,
        pointers: tuple,
        strides: tuple,
        rows: Int32,
        group_offset: Int32,
        group_count: Int32,
        stream: cuda.CUstream,
    ):
        blocks = (group_count + self.groups - 1) // self.groups
        if cutlass.const_expr(self.workers > 0):
            blocks = cutlass.min(
                blocks, cutlass.max(Int32(self.workers) // rows, Int32(1))
            )
        self.kernel(pointers, strides, group_offset, group_count).launch(
            grid=(blocks, rows, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        pointers: tuple,
        strides: tuple,
        group_offset: Int32,
        group_count: Int32,
    ):
        (
            query,
            positions,
            requests,
            lengths,
            cache,
            table,
            scores,
            counts,
            merges,
            weights,
        ) = pointers
        page_stride, token_stride, table_stride, score_stride = strides
        block, row, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane = thread % 32
        warp = thread // 32
        request = requests[row].to(Int64)
        eligible = Int32(0)
        if request >= Int64(0):
            position_groups = (
                cutlass.max(
                    positions[row].to(Int64) + Int64(self.position_offset), Int64(0)
                )
                // self.ratio
            )
            sequence_groups = lengths[request].to(Int64) // self.ratio
            global_eligible = cutlass.min(position_groups, sequence_groups).to(Int32)
            eligible = global_eligible
            if cutlass.const_expr(self.dcp_size > 1):
                round_width = Int32(self.dcp_size * self.group_interleave)
                complete_rounds = global_eligible // round_width
                remainder = global_eligible - complete_rounds * round_width
                rank_remainder = cutlass.min(
                    cutlass.max(
                        remainder - Int32(self.dcp_rank * self.group_interleave),
                        Int32(0),
                    ),
                    Int32(self.group_interleave),
                )
                eligible = (
                    complete_rounds * Int32(self.group_interleave) + rank_remainder
                )
            eligible = cutlass.min(eligible, Int32(self.max_groups))
        carry = cutlass.min(cutlass.min(eligible, group_offset), Int32(self.budget))
        if (block == 0) & (thread == 0):
            counts[row] = eligible
            if cutlass.const_expr(self.indices_only):
                if eligible <= self.budget:
                    counts[row] = Int32(0)
            merges[row] = carry + cutlass.min(
                cutlass.max(eligible - group_offset, Int32(0)), group_count
            )
        extent = group_count
        if cutlass.const_expr(self.workers > 0):
            extent = cutlass.min(
                group_count, cutlass.max(eligible - group_offset, Int32(0))
            )
        if cutlass.const_expr(self.indices_only):
            if eligible <= self.budget:
                extent = Int32(0)
        blocks, _, _ = cute.arch.grid_dim()
        for local_start in cutlass.range(
            block * self.groups, extent, blocks * self.groups
        ):
            self.score_tile(
                pointers,
                strides,
                request,
                row,
                group_offset,
                group_count,
                local_start,
                eligible,
                carry,
                lane,
                warp,
                thread,
            )

    @cute.jit
    def score_tile(
        self,
        pointers,
        strides,
        request,
        row,
        group_offset,
        group_count,
        local_start,
        eligible,
        carry,
        lane,
        warp,
        thread,
    ):
        (
            query,
            positions,
            requests,
            lengths,
            cache,
            table,
            scores,
            counts,
            merges,
            weights,
        ) = pointers
        page_stride, token_stride, table_stride, score_stride = strides
        score_row = row.to(Int64) * score_stride
        if group_offset + local_start >= eligible:
            local_group = local_start + thread
            if (thread < self.groups) & (local_group < group_count):
                scores[score_row + (carry + local_group).to(Int64)] = Float32(
                    -float("inf")
                )
        elif cutlass.const_expr(self.use_mma):
            self.score_mma(
                query,
                weights,
                cache,
                table,
                scores,
                request,
                row,
                page_stride,
                token_stride,
                table_stride,
                score_row,
                group_offset,
                group_count,
                local_start,
                eligible,
                carry,
                lane,
                warp,
            )
        else:
            prepared = cute.make_rmem_tensor((self.heads, self.lane_values), Float32)
            for head in cutlass.range_constexpr(self.heads):
                for part in cutlass.range_constexpr(self.lane_values):
                    dim = lane + part * 32
                    value = Float32(0)
                    if dim < self.dim:
                        query_offset = (
                            row.to(Int64) * self.heads * self.dim
                            + head * self.dim
                            + dim
                        )
                        value = query[query_offset].to(Float32)
                    prepared[head, part] = value
            for item in cutlass.range(self.groups // (_THREADS // 32), unroll=1):
                local_group = local_start + warp + item * (_THREADS // 32)
                group = group_offset + local_group
                score = Float32(-float("inf"))
                if (local_group < group_count) & (group < eligible):
                    logical_page = group // self.page_size
                    page = table[request * table_stride + logical_page.to(Int64)].to(
                        Int64
                    )
                    if page >= Int64(0):
                        key = cute.make_rmem_tensor((self.lane_values,), Float32)
                        key_base = (
                            page * page_stride
                            + (group % self.page_size).to(Int64) * token_stride
                        )
                        for part in cutlass.range_constexpr(self.lane_values):
                            dim = lane + part * 32
                            value = Float32(0)
                            if dim < self.dim:
                                value = cache[key_base + dim].to(Float32)
                            key[part] = value
                        score = Float32(0)
                        for head in cutlass.range_constexpr(self.heads):
                            dot = Float32(0)
                            for part in cutlass.range_constexpr(self.lane_values):
                                dot = dot + key[part] * prepared[head, part]
                            for shift in cutlass.range_constexpr(5):
                                dot = dot + cute.arch.shuffle_sync_bfly(
                                    dot, offset=16 >> shift
                                )
                            dot = cutlass.max(dot, Float32(0))
                            if cutlass.const_expr(self.weighted):
                                dot *= weights[row.to(Int64) * self.heads + head]
                            score = score + dot
                        if cutlass.const_expr(not self.weighted):
                            score = score * Float32(self.scale)
                if (lane == 0) & (local_group < group_count):
                    scores[score_row + (carry + local_group).to(Int64)] = score


def _pointer(tensor, dtype):
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=dtype.width // 8
    )


def compile_score_representatives(
    *,
    prepared_query,
    query_positions,
    request_ids,
    sequence_lengths,
    compressed_cache,
    compressed_block_table,
    scores,
    eligible_counts,
    merge_lengths,
    caps,
    weights=None,
    position_offset=1,
    bounded=False,
    indices_only=False,
):
    """Resolve the exact configured CuTe score executable for QSA preparation."""
    tensors = (
        prepared_query,
        query_positions,
        request_ids,
        sequence_lengths,
        compressed_cache,
        compressed_block_table,
        scores,
        eligible_counts,
        merge_lengths,
        scores if weights is None else weights,
    )
    types = tuple(
        {
            torch.bfloat16: BFloat16,
            torch.float8_e4m3fn: Float8E4M3FN,
            torch.float32: Float32,
            torch.int32: Int32,
            torch.int64: Int64,
        }[tensor.dtype]
        for tensor in tensors
    )
    if indices_only and not bounded:
        raise ValueError("indices-only scoring requires bounded live-prefix execution")
    device_index = prepared_query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    workers = (
        2 * torch.cuda.get_device_properties(device_index).multi_processor_count
        if bounded
        else 0
    )
    geometry = (
        int(caps.index_heads),
        int(caps.index_head_dim),
        int(caps.compress_ratio),
        int(caps.compressed_page_size),
        int(caps.max_groups),
        int(caps.group_budget),
        int(caps.dcp_size),
        int(caps.dcp_rank),
        int(caps.cp_kv_cache_interleave_size),
        weights is not None,
        prepared_query.dtype == torch.float8_e4m3fn,
        int(position_offset),
        workers,
        bool(indices_only),
    )
    device_index = prepared_query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.cuda.device(device_index):
        key = (device_index, geometry, tuple(t.dtype for t in tensors))
        raw = _CACHE.get(key)
        if raw is None:
            kernel = _RepresentativeScoreKernel(*geometry)
            raise_if_kernel_resolution_frozen(
                "cute.compile", target=kernel, cache_key=key
            )
            fake = tuple(
                make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8)
                for t in types
            )
            raw = b12x_compile(
                kernel,
                fake,
                (Int64(1),) * 4,
                Int32(1),
                Int32(0),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.qsa.representative_score", 4, key
                ),
            )
            _CACHE[key] = raw
    return raw


def launch_score_representatives(
    *,
    prepared_query,
    query_positions,
    request_ids,
    sequence_lengths,
    compressed_cache,
    compressed_block_table,
    scores,
    eligible_counts,
    merge_lengths,
    group_offset,
    group_count,
    caps,
    weights=None,
    position_offset=1,
    bounded=False,
    indices_only=False,
    _prepared=None,
):
    if indices_only and group_offset != 0:
        raise ValueError("indices-only scoring does not accept segmented scores")
    tensors = (
        prepared_query,
        query_positions,
        request_ids,
        sequence_lengths,
        compressed_cache,
        compressed_block_table,
        scores,
        eligible_counts,
        merge_lengths,
        scores if weights is None else weights,
    )
    types = tuple(
        {
            torch.bfloat16: BFloat16,
            torch.float8_e4m3fn: Float8E4M3FN,
            torch.float32: Float32,
            torch.int32: Int32,
            torch.int64: Int64,
        }[t.dtype]
        for t in tensors
    )
    raw = (
        _prepared
        if _prepared is not None
        else compile_score_representatives(
            prepared_query=prepared_query,
            query_positions=query_positions,
            request_ids=request_ids,
            sequence_lengths=sequence_lengths,
            compressed_cache=compressed_cache,
            compressed_block_table=compressed_block_table,
            scores=scores,
            eligible_counts=eligible_counts,
            merge_lengths=merge_lengths,
            caps=caps,
            weights=weights,
            position_offset=position_offset,
            bounded=bounded,
            indices_only=indices_only,
        )
    )
    if compile_only_launches_enabled():
        for program in program_keys(raw):
            record_program(program)
        return raw
    raw(
        tuple(_pointer(t, dt) for t, dt in zip(tensors, types, strict=True)),
        (
            Int64(compressed_cache.stride(0)),
            Int64(compressed_cache.stride(1)),
            Int64(compressed_block_table.stride(0)),
            Int64(scores.stride(0)),
        ),
        Int32(prepared_query.shape[0]),
        Int32(group_offset),
        Int32(group_count),
        current_cuda_stream(),
    )
