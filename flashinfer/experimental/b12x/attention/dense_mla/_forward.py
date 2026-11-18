"""Standalone warp-specialized dense K3 MLA forward kernel."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from b12x._lib.intrinsics import shared_ptr_to_u32

from ._io import issue_dense_page_gather
from ._layout import (
    CANDIDATES_PER_CHUNK,
    HEADS_PER_TILE,
    MATH_WARPS_PER_QUERY,
    SmemLayout,
    get_shared_storage_cls,
)
from ._math import (
    mask_and_scale,
    online_softmax,
    pv_bf16,
    pv_fp8,
    qk_bf16,
    qk_fp8,
    stage_absorbed_query,
    write_partial_or_final,
)


@dsl_user_op
def _exit_thread(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "exit;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class DenseMlaForwardKernel:
    """A K3-specialized dense page traversal and attention pipeline."""

    def __init__(
        self,
        *,
        layout: SmemLayout,
        page_size: int,
        num_heads: int,
        num_splits: int,
        chunks_per_split: int,
        query_tile: int,
        fp8: bool,
        qk_dim: int,
        value_dim: int,
        window_size: int | None,
    ):
        self.layout = layout
        self.page_size = int(page_size)
        self.num_heads = int(num_heads)
        self.head_tiles = (self.num_heads + HEADS_PER_TILE - 1) // HEADS_PER_TILE
        self.num_splits = int(num_splits)
        self.chunks_per_split = int(chunks_per_split)
        self.query_tile = int(query_tile)
        self.fp8 = bool(fp8)
        self.qk_dim = int(qk_dim)
        self.value_dim = int(value_dim)
        self.window_size = None if window_size is None else int(window_size)
        self.kv_stages = int(layout.kv_stages)
        self.math_warps = MATH_WARPS_PER_QUERY * self.query_tile
        self.math_threads = self.math_warps * 32
        self.block_threads = self.math_threads + 32

    @cute.jit
    def __call__(
        self,
        q_bytes: cute.Tensor,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        kv_scale: cute.Tensor,
        q_scale: cute.Tensor,
        sm_scale_log2: Float32,
        q_stride_row_bytes: Int64,
        q_stride_head_bytes: Int64,
        page_stride_bytes: Int64,
        cache_record_stride_bytes: Int64,
        page_table_stride: Int64,
        total_q: Int32,
        batch: Int32,
        active_splits: Int32,
        stream: cuda.CUstream,
    ):
        query_tiles = (total_q + Int32(self.query_tile - 1)) // Int32(self.query_tile)
        self.kernel(
            q_bytes,
            cache_bytes,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            output,
            final_lse,
            partial_output,
            partial_lse,
            kv_scale,
            q_scale,
            sm_scale_log2,
            q_stride_row_bytes,
            q_stride_head_bytes,
            page_stride_bytes,
            cache_record_stride_bytes,
            page_table_stride,
            total_q,
            batch,
            active_splits,
        ).launch(
            grid=(query_tiles, self.head_tiles, active_splits),
            block=[self.block_threads, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.jit
    def _run_math_group(
        self,
        q_smem_addr: Int32,
        kv_smem_addr: Int32,
        reduce_max_addr: Int32,
        reduce_sum_addr: Int32,
        weight_scale_addr: Int32,
        weight_addr: Int32,
        mbar_base,
        active_chunks: Int32,
        split_first_chunk: Int32,
        visible_begin: Int32,
        visible_end: Int32,
        query_row: Int32,
        head_base: Int32,
        split: Int32,
        active_splits: Int32,
        local_warp: Int32,
        lane: Int32,
        local_tid: Int32,
        query_valid: Int32,
        score_scale_log2: Float32,
        value_scale: Float32,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        *,
        barrier_id: cutlass.Constexpr,
    ):
        accumulator_fragment = cute.make_rmem_tensor(self.value_dim // 16, Float32)
        global_max_fragment = cute.make_rmem_tensor(2, Float32)
        global_sum_fragment = cute.make_rmem_tensor(2, Float32)
        for idx in cutlass.range_constexpr(self.value_dim // 16):
            accumulator_fragment[idx] = Float32(0.0)
        global_max_fragment[0] = Float32(-1.0e30)
        global_max_fragment[1] = Float32(-1.0e30)
        global_sum_fragment[0] = Float32(0.0)
        global_sum_fragment[1] = Float32(0.0)

        consumer_phase = Int32(0)
        consumer_index = Int32(0)
        kv_buffer_bytes = Int32(CANDIDATES_PER_CHUNK * self.layout.record_stride_bytes)

        for local_chunk in cutlass.range(active_chunks, unroll=1):
            chunk = split_first_chunk + Int32(local_chunk)
            chunk_begin = chunk * Int32(CANDIDATES_PER_CHUNK)
            buffer = Int32(0)
            if cutlass.const_expr(self.kv_stages == 2):
                buffer = Int32(local_chunk) & Int32(1)
            kv_buffer_addr = kv_smem_addr + buffer * kv_buffer_bytes
            cute.arch.mbarrier_wait(
                mbar_base + consumer_index,
                phase=consumer_phase,
            )
            cute.arch.barrier(
                barrier_id=barrier_id,
                number_of_threads=128,
            )

            if query_valid != Int32(0) and chunk_begin < visible_end:
                accumulator = [
                    [
                        accumulator_fragment[tile * 2],
                        accumulator_fragment[tile * 2 + 1],
                    ]
                    for tile in range(self.value_dim // 32)
                ]
                global_max = [
                    global_max_fragment[0],
                    global_max_fragment[1],
                ]
                global_sum = [
                    global_sum_fragment[0],
                    global_sum_fragment[1],
                ]
                qk = [
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                ]
                if cutlass.const_expr(self.fp8):
                    qk = qk_fp8(
                        qk,
                        q_smem_addr,
                        kv_buffer_addr,
                        local_warp,
                        lane,
                        record_stride_bytes=self.layout.record_stride_bytes,
                        qk_dim=self.qk_dim,
                    )
                else:
                    qk = qk_bf16(
                        qk,
                        q_smem_addr,
                        kv_buffer_addr,
                        local_warp,
                        lane,
                        record_stride_bytes=self.layout.record_stride_bytes,
                        qk_dim=self.qk_dim,
                    )
                qk = mask_and_scale(
                    qk,
                    chunk_begin,
                    visible_begin,
                    visible_end,
                    score_scale_log2,
                    local_warp,
                    lane,
                )
                probability = [
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                ]
                probability, warp_scale0, warp_scale1 = online_softmax(
                    qk,
                    probability,
                    accumulator,
                    global_max,
                    global_sum,
                    reduce_max_addr,
                    reduce_sum_addr,
                    local_warp,
                    lane,
                    local_tid,
                    barrier_id=barrier_id,
                    value_dim=self.value_dim,
                )
                weights = [
                    probability[0] * warp_scale0,
                    probability[1] * warp_scale1,
                    probability[2] * warp_scale0,
                    probability[3] * warp_scale1,
                ]
                if cutlass.const_expr(self.fp8):
                    accumulator = pv_fp8(
                        weights,
                        accumulator,
                        kv_buffer_addr,
                        weight_scale_addr,
                        weight_addr,
                        local_warp,
                        lane,
                        local_tid,
                        record_stride_bytes=self.layout.record_stride_bytes,
                        barrier_id=barrier_id,
                        value_dim=self.value_dim,
                    )
                else:
                    accumulator = pv_bf16(
                        weights,
                        accumulator,
                        kv_buffer_addr,
                        weight_addr,
                        local_warp,
                        lane,
                        record_stride_bytes=self.layout.record_stride_bytes,
                        barrier_id=barrier_id,
                        value_dim=self.value_dim,
                    )
                for tile in cutlass.range_constexpr(self.value_dim // 32):
                    accumulator_fragment[tile * 2] = accumulator[tile][0]
                    accumulator_fragment[tile * 2 + 1] = accumulator[tile][1]
                global_max_fragment[0] = global_max[0]
                global_max_fragment[1] = global_max[1]
                global_sum_fragment[0] = global_sum[0]
                global_sum_fragment[1] = global_sum[1]

            cute.arch.barrier(
                barrier_id=barrier_id,
                number_of_threads=128,
            )
            if local_tid == Int32(0):
                cute.arch.mbarrier_arrive(
                    mbar_base + Int32(self.kv_stages) + consumer_index
                )
            consumer_index += Int32(1)
            if consumer_index == Int32(self.kv_stages):
                consumer_index = Int32(0)
                consumer_phase ^= Int32(1)

        accumulator = [
            [
                accumulator_fragment[tile * 2],
                accumulator_fragment[tile * 2 + 1],
            ]
            for tile in range(self.value_dim // 32)
        ]
        global_max = [
            global_max_fragment[0],
            global_max_fragment[1],
        ]
        global_sum = [
            global_sum_fragment[0],
            global_sum_fragment[1],
        ]
        if cutlass.const_expr(self.num_splits > 1):
            if active_splits > Int32(1):
                write_partial_or_final(
                    accumulator,
                    global_max,
                    global_sum,
                    partial_output,
                    partial_lse,
                    query_row,
                    head_base,
                    split,
                    local_warp,
                    lane,
                    value_scale,
                    query_valid,
                    num_heads=self.num_heads,
                    has_splits=True,
                    fp8=self.fp8,
                    ln2=0.6931471805599453,
                    value_dim=self.value_dim,
                )
            else:
                write_partial_or_final(
                    accumulator,
                    global_max,
                    global_sum,
                    output,
                    final_lse,
                    query_row,
                    head_base,
                    split,
                    local_warp,
                    lane,
                    value_scale,
                    query_valid,
                    num_heads=self.num_heads,
                    has_splits=False,
                    fp8=self.fp8,
                    ln2=0.6931471805599453,
                    value_dim=self.value_dim,
                )
        else:
            write_partial_or_final(
                accumulator,
                global_max,
                global_sum,
                output,
                final_lse,
                query_row,
                head_base,
                split,
                local_warp,
                lane,
                value_scale,
                query_valid,
                num_heads=self.num_heads,
                has_splits=False,
                fp8=self.fp8,
                ln2=0.6931471805599453,
                value_dim=self.value_dim,
            )

    @cute.kernel
    def kernel(
        self,
        q_bytes: cute.Tensor,
        cache_bytes: cute.Tensor,
        page_table: cute.Tensor,
        cache_seqlens: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        output: cute.Tensor,
        final_lse: cute.Tensor,
        partial_output: cute.Tensor,
        partial_lse: cute.Tensor,
        kv_scale: cute.Tensor,
        q_scale: cute.Tensor,
        sm_scale_log2: Float32,
        q_stride_row_bytes: Int64,
        q_stride_head_bytes: Int64,
        page_stride_bytes: Int64,
        cache_record_stride_bytes: Int64,
        page_table_stride: Int64,
        total_q: Int32,
        batch: Int32,
        active_splits: Int32,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp = tid >> Int32(5)
        query_tile_index, head_tile, split = cute.arch.block_idx()
        query_tile_index = Int32(query_tile_index)
        head_tile = Int32(head_tile)
        split = Int32(split)
        query_start = query_tile_index * Int32(self.query_tile)
        head_base = head_tile * Int32(HEADS_PER_TILE)

        request = Int32(0)
        if cutlass.const_expr(self.query_tile == 1):
            lower = Int32(0)
            upper = batch
            while lower + Int32(1) < upper:
                middle = (lower + upper) // Int32(2)
                if query_start >= Int32(cu_seqlens_q[middle]):
                    lower = middle
                else:
                    upper = middle
            request = lower
        query_begin = Int32(cu_seqlens_q[request])
        query_end = Int32(cu_seqlens_q[request + Int32(1)])
        query_length = query_end - query_begin
        cache_length = Int32(cache_seqlens[request])

        first_valid_chunk = Int32(0)
        visible_chunks_end = cache_length
        if cutlass.const_expr(self.window_size is not None):
            tile_local_query = query_start - query_begin
            visible_chunks_end = (
                cache_length - query_length + tile_local_query + Int32(1)
            )
            if visible_chunks_end < Int32(0):
                visible_chunks_end = Int32(0)
            if visible_chunks_end > cache_length:
                visible_chunks_end = cache_length
            visible_chunks_begin = visible_chunks_end - Int32(self.window_size)
            if visible_chunks_begin < Int32(0):
                visible_chunks_begin = Int32(0)
            first_valid_chunk = visible_chunks_begin // Int32(CANDIDATES_PER_CHUNK)
        valid_chunks = (visible_chunks_end + Int32(CANDIDATES_PER_CHUNK - 1)) // Int32(
            CANDIDATES_PER_CHUNK
        )
        split_first_chunk = first_valid_chunk + split * Int32(self.chunks_per_split)
        split_last_chunk = split_first_chunk + Int32(self.chunks_per_split)
        if split_last_chunk > valid_chunks:
            split_last_chunk = valid_chunks
        active_chunks = split_last_chunk - split_first_chunk
        if active_chunks < Int32(0):
            active_chunks = Int32(0)

        if active_chunks == Int32(0):
            entry = tid
            while entry < Int32(self.query_tile * HEADS_PER_TILE):
                query_slot = entry // Int32(HEADS_PER_TILE)
                local_head = entry - query_slot * Int32(HEADS_PER_TILE)
                query_row = query_start + query_slot
                if query_row < total_q and head_base + local_head < Int32(
                    self.num_heads
                ):
                    if cutlass.const_expr(self.num_splits > 1):
                        if active_splits > Int32(1):
                            partial_lse[
                                query_row,
                                head_base + local_head,
                                split,
                            ] = Float32(-Float32.inf)
                        else:
                            final_lse[
                                query_row,
                                head_base + local_head,
                            ] = Float32(-Float32.inf)
                    else:
                        final_lse[
                            query_row,
                            head_base + local_head,
                        ] = Float32(-Float32.inf)
                entry += Int32(self.block_threads)
            _exit_thread()

        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(get_shared_storage_cls(self.layout))
        q_smem_addr = shared_ptr_to_u32(storage.q.data_ptr())
        kv_smem_addr = shared_ptr_to_u32(storage.kv.data_ptr())
        reduce_smem_addr = shared_ptr_to_u32(storage.reduce.data_ptr())
        weight_scale_smem_addr = shared_ptr_to_u32(storage.weight_scale.data_ptr())
        weight_smem_addr = shared_ptr_to_u32(storage.weight.data_ptr())
        mbar_base = storage.mbar.data_ptr()

        if tid == Int32(0):
            for stage in cutlass.range_constexpr(self.kv_stages):
                cute.arch.mbarrier_init(mbar_base + stage, Int32(1))
                cute.arch.mbarrier_init(
                    mbar_base + Int32(self.kv_stages + stage),
                    Int32(self.query_tile),
                )
        cute.arch.barrier()

        if warp == Int32(self.math_warps):
            io_lane = lane
            producer_phase = Int32(1)
            producer_index = Int32(0)
            kv_buffer_bytes = Int32(
                CANDIDATES_PER_CHUNK * self.layout.record_stride_bytes
            )
            for local_chunk in cutlass.range(active_chunks, unroll=1):
                chunk = split_first_chunk + Int32(local_chunk)
                token_begin = chunk * Int32(CANDIDATES_PER_CHUNK)
                buffer = Int32(0)
                if cutlass.const_expr(self.kv_stages == 2):
                    buffer = Int32(local_chunk) & Int32(1)
                cute.arch.mbarrier_wait(
                    mbar_base + Int32(self.kv_stages) + producer_index,
                    phase=producer_phase,
                )
                issue_dense_page_gather(
                    cache_bytes,
                    page_table,
                    kv_smem_addr + buffer * kv_buffer_bytes,
                    mbar_base + buffer,
                    request,
                    token_begin,
                    cache_length,
                    io_lane,
                    page_stride_bytes,
                    cache_record_stride_bytes,
                    page_table_stride,
                    page_size=self.page_size,
                    record_bytes=self.layout.record_bytes,
                    record_stride_bytes=self.layout.record_stride_bytes,
                )
                producer_index += Int32(1)
                if producer_index == Int32(self.kv_stages):
                    producer_index = Int32(0)
                    producer_phase ^= Int32(1)
        else:
            stage_absorbed_query(
                q_bytes,
                q_smem_addr,
                query_start,
                head_base,
                total_q,
                tid,
                q_stride_row_bytes,
                q_stride_head_bytes,
                num_heads=self.num_heads,
                query_tile=self.query_tile,
                record_bytes=self.layout.record_bytes,
                record_stride_bytes=self.layout.record_stride_bytes,
                math_threads=self.math_threads,
            )
            group = warp // Int32(MATH_WARPS_PER_QUERY)
            local_warp = warp - group * Int32(MATH_WARPS_PER_QUERY)
            local_tid = local_warp * Int32(32) + lane
            query_row = query_start + group
            query_valid = Int32(1)
            if (
                query_row >= total_q
                or query_row < query_begin
                or query_row >= query_end
            ):
                query_valid = Int32(0)
            local_query = query_row - query_begin
            visible_end = cache_length - query_length + local_query + Int32(1)
            if visible_end < Int32(0):
                visible_end = Int32(0)
            if visible_end > cache_length:
                visible_end = cache_length
            visible_begin = Int32(0)
            if cutlass.const_expr(self.window_size is not None):
                visible_begin = visible_end - Int32(self.window_size)
                if visible_begin < Int32(0):
                    visible_begin = Int32(0)

            score_scale = sm_scale_log2
            value_scale = Float32(1.0)
            if cutlass.const_expr(self.fp8):
                value_scale = Float32(kv_scale[Int32(0)])
                score_scale = score_scale * value_scale * Float32(q_scale[Int32(0)])

            group_q_addr = q_smem_addr + group * Int32(
                HEADS_PER_TILE * self.layout.record_stride_bytes
            )
            group_reduce_addr = reduce_smem_addr + group * Int32(
                self.layout.reduce_group_bytes
            )
            reduce_max_addr = group_reduce_addr
            reduce_sum_addr = group_reduce_addr + Int32(
                MATH_WARPS_PER_QUERY * HEADS_PER_TILE * 4
            )
            group_weight_scale_addr = weight_scale_smem_addr + group * Int32(
                self.layout.weight_scale_group_bytes
            )
            group_weight_addr = weight_smem_addr + group * Int32(
                self.layout.weight_group_bytes
            )
            if group == Int32(0):
                self._run_math_group(
                    group_q_addr,
                    kv_smem_addr,
                    reduce_max_addr,
                    reduce_sum_addr,
                    group_weight_scale_addr,
                    group_weight_addr,
                    mbar_base,
                    active_chunks,
                    split_first_chunk,
                    visible_begin,
                    visible_end,
                    query_row,
                    head_base,
                    split,
                    active_splits,
                    local_warp,
                    lane,
                    local_tid,
                    query_valid,
                    score_scale,
                    value_scale,
                    output,
                    final_lse,
                    partial_output,
                    partial_lse,
                    barrier_id=2,
                )
            elif group == Int32(1):
                self._run_math_group(
                    group_q_addr,
                    kv_smem_addr,
                    reduce_max_addr,
                    reduce_sum_addr,
                    group_weight_scale_addr,
                    group_weight_addr,
                    mbar_base,
                    active_chunks,
                    split_first_chunk,
                    visible_begin,
                    visible_end,
                    query_row,
                    head_base,
                    split,
                    active_splits,
                    local_warp,
                    lane,
                    local_tid,
                    query_valid,
                    score_scale,
                    value_scale,
                    output,
                    final_lse,
                    partial_output,
                    partial_lse,
                    barrier_id=3,
                )
            elif group == Int32(2):
                self._run_math_group(
                    group_q_addr,
                    kv_smem_addr,
                    reduce_max_addr,
                    reduce_sum_addr,
                    group_weight_scale_addr,
                    group_weight_addr,
                    mbar_base,
                    active_chunks,
                    split_first_chunk,
                    visible_begin,
                    visible_end,
                    query_row,
                    head_base,
                    split,
                    active_splits,
                    local_warp,
                    lane,
                    local_tid,
                    query_valid,
                    score_scale,
                    value_scale,
                    output,
                    final_lse,
                    partial_output,
                    partial_lse,
                    barrier_id=4,
                )
            else:
                self._run_math_group(
                    group_q_addr,
                    kv_smem_addr,
                    reduce_max_addr,
                    reduce_sum_addr,
                    group_weight_scale_addr,
                    group_weight_addr,
                    mbar_base,
                    active_chunks,
                    split_first_chunk,
                    visible_begin,
                    visible_end,
                    query_row,
                    head_base,
                    split,
                    active_splits,
                    local_warp,
                    lane,
                    local_tid,
                    query_valid,
                    score_scale,
                    value_scale,
                    output,
                    final_lse,
                    partial_output,
                    partial_lse,
                    barrier_id=5,
                )


__all__ = ["DenseMlaForwardKernel"]
