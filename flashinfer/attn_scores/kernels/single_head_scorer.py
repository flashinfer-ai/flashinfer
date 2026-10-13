# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""The single-head paged scorer: FP32 dot products of query rows with their
request's paged BF16 or NVFP4 keys, on SM100 tensor cores (tcgen05)."""

from functools import cache
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.experimental.primitives import exit as exit_thread

from ...cute_dsl.fp4_common import (
    cvt_e2m1x8_f32,
    cvt_f32_to_e4m3,
    fabs_f32,
    fdiv_rn,
    fmax_f32,
    fp8_e4m3_to_f32,
)

# The most key tiles a CTA runs over, staging their page indices (8 B a tile),
# so a run covers 32K keys.
TILES_PER_CTA = 256
# NVFP4 values (E2M1, two to a byte) come with one E4M3 scale per 16.
SCALE_TYPE = cutlass.Float8E4M3FN
SCALE_VECTOR = 16


@cute.jit
def partition_scales(shared: cute.Tensor, tensor: cute.Tensor):
    """The S2T scale copy of mainloop_s2t_copy_and_partition in FlashInfer's
    block-scaled SM100 GEMM (flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py).
    It puts the scales into TMEM in the layout that the block-scaled MMA reads."""
    atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), SCALE_TYPE)
    shared, tensor = (cute.filter_zeros(shared), cute.filter_zeros(tensor))
    tiled_copy = tcgen05.make_s2t_copy(atom, tensor)
    thread_copy = tiled_copy.get_slice(0)
    source = tcgen05.get_s2t_smem_desc_tensor(
        tiled_copy, thread_copy.partition_S(shared)
    )
    return (tiled_copy, source, thread_copy.partition_D(tensor))


class PagedScorer:
    """Scores of query rows against their request's paged keys, BF16 or NVFP4
    (``nvfp4``: E2M1 values with E4M3 scales).

    A job is ``rows`` (1 to 8) consecutive query rows that share block-table row
    ``tile_requests[j]`` and see keys ``< row_ends[j]``. Each job gets
    ``workers`` CTAs (the jobs are the grid's fastest-varying index), each
    owning one equal run of the job's 128-key tiles.

    A CTA streams its run through its key stages: a tile's two 64-key pages are
    TMA-loaded into a stage once the MMA of the stage's previous tile is done.
    Warps are specialized: warp 5 loads the keys, warp 4 issues each tile's MMA
    into one of two accumulators, and warps 0-3 store the scores. The keys are
    on the TMEM lanes (M = 128 keys, N = 8 query columns, ``rows`` of them real),
    so a warp stores 32 consecutive scores per row. With NVFP4, warps 0-3 also
    copy each tile's key scales by cp.async into its stage, counted in on the
    stage's full barrier.
    """

    def __init__(self, rows: int, nvfp4: bool):
        self.rows = rows  # query rows per job
        self.values = cutlass.Float4E2M1FN if nvfp4 else cutlass.BFloat16
        self.scaled = nvfp4
        # The MMA tile (M, N, K) of 128 keys, 8 query columns (the smallest MMA
        # N) and 128 values; 192 threads; two accumulators of 8 columns and
        # NVFP4's scale blocks in the 32 TMEM columns tcgen05.alloc takes at least.
        self.tile, self.threads = (128, 8, 128), 192
        self.accumulators, self.tmem_columns = 2, 32
        self.stages = 0  # key stages, set at compile time

    @cute.jit
    def __call__(
        self,
        stream: cuda.CUstream,
        query: cute.Tensor,
        query_scales: Optional[cute.Tensor],
        key_cache: cute.Tensor,
        key_scales: Optional[cute.Tensor],
        row_ends: cute.Tensor,
        tile_requests: cute.Tensor,
        block_table: cute.Tensor,
        output: cute.Tensor,
        key_rows: cutlass.Int32,
        workers: cutlass.Int32,
    ):
        jobs, pages = query.shape[0] // self.rows, key_cache.shape[0]
        packing = key_cache.element_type.width // self.values.width  # NVFP4: 2
        matrix_key = cute.make_tensor(
            cute.recast_ptr(key_cache.iterator, dtype=self.values),
            cute.make_layout(
                (64, 128, pages), stride=(128, 1, packing * key_cache.stride[0])
            ),
        )
        # A job's 8-row query box over its rows: TMA zero-fills the rest.
        matrix_query = cute.make_tensor(
            cute.recast_ptr(query.iterator, dtype=self.values),
            cute.make_layout((self.rows, 128, jobs), stride=(128, 1, self.rows * 128)),
        )
        k_major = cute.nvgpu.OperandMajorMode.K
        if cutlass.const_expr(self.scaled):
            tiled_mma = cute.make_tiled_mma(
                tcgen05.MmaMXF4NVF4Op(
                    SCALE_TYPE,
                    self.tile[:2] + (64,),  # the MXF4 MMA's K: two per 128-value tile
                    tcgen05.CtaGroup.ONE,
                    tcgen05.OperandSource.SMEM,
                )
            )
        else:  # BF16 with FP32 accumulation, K = 16: eight per 128-value tile
            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.values,
                self.values,
                k_major,
                k_major,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                self.tile[:2],
            )
        # The keys are the MMA's A operand, the query B.
        make_key = (
            sm100_utils.make_smem_layout_a,
            blockscaled_utils.make_smem_layout_sfa,
        )
        make_query = (
            sm100_utils.make_smem_layout_b,
            blockscaled_utils.make_smem_layout_sfb,
        )

        def operand(make: tuple, stages: int) -> tuple:
            """An operand's shared-memory layout, NVFP4's scale layout, and bytes."""
            layout = make[0](tiled_mma, self.tile, self.values, stages)
            scales, size = None, cute.size_in_bytes(self.values, layout)
            if cutlass.const_expr(self.scaled):
                scales = make[1](tiled_mma, self.tile, SCALE_VECTOR, stages)
                size += cute.size_in_bytes(SCALE_TYPE, scales)
            return layout, scales, size

        # Shared memory: the query, the barriers and TMEM address (padded to
        # 128 B), the page indices, then as many key stages as fit in half an
        # SM's 228 KB (a CTA's 227 KB plus the 1 KB reserved per CTA).
        capacity = utils.get_smem_capacity_in_bytes("sm_100") + 1024
        layout_query, scales_query, fixed = operand(make_query, 1)
        fixed += 8 * 2 * (1 + self.accumulators) + 4 + 127 + 8 * TILES_PER_CTA
        stage = 16 + operand(make_key, 1)[2]
        self.stages = (capacity // 2 - 1024 - fixed) // stage
        layout_key, scales_key, _ = operand(make_key, self.stages)
        shared_bytes = fixed + self.stages * stage
        # A key tile as TMA writes it, then the stages: the key operand's swizzled
        # layout of 128 keys, keys first (as CUTLASS tiles K-major operands), so a
        # 64-key page is one half.
        key_tile = cute.slice_(
            sm100_utils.make_smem_layout(k_major, (128, 128), self.values, 1),
            (None, None, 0),
        )
        layout_key_page = cute.get(cute.tiled_divide(key_tile, (64, 128)), mode=[0])
        key_pages = cute.select(cute.tiled_divide(key_tile.outer, (64, 128)), [0, 1])
        key_pages = cute.append(
            key_pages,
            cute.make_layout(
                self.stages,
                stride=cute.cosize(
                    cute.slice_(layout_key.outer, (None, None, None, 0))
                ),
            ),
        )
        tma_key, tensor_key = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            matrix_key,
            layout_key_page,
            (64, 128),
        )
        tma_query, tensor_query = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            matrix_query,
            cute.slice_(layout_query, (None, None, None, 0)),
            self.tile,
            tiled_mma,
        )
        self.kernel(
            tiled_mma,
            tma_key,
            tensor_key,
            tma_query,
            tensor_query,
            key_scales,
            query_scales,
            row_ends,
            tile_requests,
            block_table,
            output,
            layout_key,
            layout_query,
            scales_key,
            scales_query,
            key_pages,
            key_rows,
            workers,
        ).launch(
            grid=(jobs, workers, 1),
            block=(self.threads, 1, 1),
            smem=shared_bytes,
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_key: cute.CopyAtom,
        tensor_key: cute.Tensor,
        tma_query: cute.CopyAtom,
        tensor_query: cute.Tensor,
        key_scales: Optional[cute.Tensor],
        query_scales: Optional[cute.Tensor],
        row_ends: cute.Tensor,
        tile_requests: cute.Tensor,
        block_table: cute.Tensor,
        output: cute.Tensor,
        layout_key: cute.ComposedLayout,
        layout_query: cute.ComposedLayout,
        scales_key: Optional[cute.Layout],
        scales_query: Optional[cute.Layout],
        key_pages: cute.Layout,
        key_rows: cutlass.Int32,
        workers: cutlass.Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        job, worker, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        @cute.struct
        class SharedStorage:
            key_barriers: cute.struct.MemRange[cutlass.Int64, 2 * self.stages]
            query_barriers: cute.struct.MemRange[cutlass.Int64, 2]
            acc_barriers: cute.struct.MemRange[cutlass.Int64, 2 * self.accumulators]
            tmem_address: cutlass.Int32

        shared = utils.SmemAllocator()
        storage = shared.allocate(SharedStorage)
        # Named barriers: 0 is the CTA's, and 1 guards the TMEM address retrieval.
        # Until a CTA relinquishes its SM's TMEM allocation permit or exits, no
        # other CTA of this kernel launches on that SM. So allocate and relinquish
        # before anything waits on memory.
        tmem = utils.TmemAllocator(
            storage.tmem_address.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(1, self.threads),
        )
        tmem_pool = tmem.reserve(self.tmem_columns)
        tmem.relinquish_alloc_permit()
        base = tmem_pool.base_ptr
        local_thread = thread % 128  # the TMEM lane: a key
        # Flat views with 64-bit offsets: cache and output offsets can pass 2**31.
        flat_int64 = cute.make_layout(cutlass.Int64(2**31 - 1))
        pages = cute.make_tensor(block_table.iterator, flat_int64)
        # The job's request and last key; its rows share one length.
        request = cute.make_tensor(tile_requests.iterator, flat_int64)[job]
        job_end = cutlass.min(row_ends[job], key_rows)
        # This CTA's run of key tiles.
        job_tiles = (job_end + 127) // 128
        run = (job_tiles + workers - 1) // workers
        first_tile = worker * run
        tile_count = cutlass.min(run, job_tiles - first_tile)
        if tile_count <= 0:
            if warp == 0:
                # tmem.free() counts columns while tracing; the final one needs them.
                cute.arch.dealloc_tmem(base, self.tmem_columns)
            exit_thread()
        shared_key = shared.allocate_tensor(
            self.values, layout_key.outer, 128, layout_key.inner
        )
        shared_query = shared.allocate_tensor(
            self.values, layout_query.outer, 128, layout_query.inner
        )
        shared_key_scales = shared_query_scales = scale_key = scale_query = None
        if cutlass.const_expr(self.scaled):
            shared_key_scales = shared.allocate_tensor(SCALE_TYPE, scales_key, 128)
            shared_query_scales = shared.allocate_tensor(SCALE_TYPE, scales_query, 128)
        # Each 64-key half-tile's page.
        tile_pages = shared.allocate_tensor(
            cutlass.Int32, cute.make_layout(2 * TILES_PER_CTA), 16
        )
        accumulator = tmem_pool.allocate_tensor(
            tiled_mma.make_fragment_C(
                cute.append(
                    tiled_mma.partition_shape_C(self.tile[:2]), self.accumulators
                )
            ).layout,
            cutlass.Float32,
        )
        if cutlass.const_expr(self.scaled):  # the pool asserts they fit
            scale_key = tmem_pool.allocate_tensor(
                blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma,
                    self.tile,
                    SCALE_VECTOR,
                    cute.slice_(scales_key, (None, None, None, 0)),
                ),
                SCALE_TYPE,
            )
            scale_query = tmem_pool.allocate_tensor(
                blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma,
                    self.tile,
                    SCALE_VECTOR,
                    cute.slice_(scales_query, (None, None, None, 0)),
                ),
                SCALE_TYPE,
            )
        fragment_key = tiled_mma.make_fragment_A(shared_key)
        fragment_query = tiled_mma.make_fragment_B(shared_query)
        # The run's page indices: 0 past the job's end, whose entries go unread.
        request_row = request.to(cutlass.Int64) * block_table.shape[1]
        for slot in cutlass.range(thread, 2 * tile_count, self.threads):
            block, page = first_tile * 2 + slot, cutlass.Int32(0)
            if block * 64 < job_end:
                page = pages[request_row + block]
            tile_pages[slot] = page
        # NVFP4's query scales in the MMA's order: row r's scale words w = 0, 1
        # at w * 128 + (r % 32) * 4 + r // 32.
        if cutlass.const_expr(self.scaled):
            words = cute.make_tensor(
                cute.recast_ptr(shared_query_scales.iterator, dtype=cutlass.Uint32),
                cute.make_layout(256),
            )
            if thread < self.rows:
                packed = cute.make_tensor(
                    cute.recast_ptr(query_scales.iterator, dtype=cutlass.Uint64),
                    flat_int64,
                )[job * self.rows + thread]
                words[(thread % 32) * 4 + thread // 32] = cutlass.Uint32(packed)
                words[128 + (thread % 32) * 4 + thread // 32] = cutlass.Uint32(
                    packed >> 32
                )
            cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # A key stage is full once its TMA bytes and NVFP4's scale copies land.
        key_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.key_barriers.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 129 if self.scaled else 1
            ),
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(self.values, shared_key.layout) // self.stages,
        )
        key_producer, key_consumer = key_pipeline.make_participants()
        query_producer, query_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.query_barriers.data_ptr(),
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(self.values, shared_query.layout),
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_barriers.data_ptr(),
            num_stages=self.accumulators,
            producer_group=producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
        ).make_participants()
        mma_thread = tiled_mma.get_slice(0)
        global_key = cute.local_tile(tensor_key, (64, 128), (None, None, None))
        load_key_dst, load_key_src = cpasync.tma_partition(
            tma_key,
            0,
            cute.make_layout(1),
            cute.make_tensor(shared_key.iterator, key_pages),
            cute.group_modes(global_key, 0, 2),
        )
        query_tile = cute.local_tile(tensor_query, (8, 128), (None, None, None))
        load_query_dst, load_query_src = cpasync.tma_partition(
            tma_query,
            0,
            cute.make_layout(1),
            cute.group_modes(shared_query, 0, 3),
            cute.group_modes(mma_thread.partition_B(query_tile), 0, 3),
        )
        load_query_dst = cute.filter_zeros(load_query_dst)
        load_query_src = cute.filter_zeros(load_query_src)
        # Each thread reads the 8 query columns of its key from TMEM, where an
        # address is lane << 16 | column.
        atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
            cutlass.Float32,
        )
        resident_layout = cute.make_layout((128, 8), stride=(65536, 1))
        tiled_copy = tcgen05.make_tmem_copy(
            atom, cute.make_tensor(base, resident_layout)
        )
        thread_copy = tiled_copy.get_slice(local_thread)
        registers = cute.make_fragment_like(
            thread_copy.partition_D(cute.make_identity_tensor((128, 8))),
            cutlass.Float32,
        )
        copies = scale_loads = None
        if cutlass.const_expr(self.scaled):
            copies = (
                partition_scales(shared_query_scales, scale_query),
                partition_scales(shared_key_scales, scale_key),
            )
            scale_loads = (key_scales, shared_key_scales, tile_pages, key_pipeline)
        # The keys are the MMA's A operand, the query B.
        mma: tuple = (tiled_mma, accumulator, fragment_key, fragment_query)
        mma += (scale_key, scale_query, copies)
        key_loads = (tma_key, load_key_src, load_key_dst, tile_pages)
        # Tile t + stages - 1 loads into the stage of tile t - 1, whose MMA is done.
        ahead = self.stages - 1
        scale_stage = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stages
        )

        # Warp 5 loads the query once, then the keys; warp 4 issues tile t's MMA
        # into one accumulator while warps 0-3 store tile t - 1's scores from
        # the other.
        if warp == 5:
            query_empty = query_producer.acquire_and_advance()
            cute.copy(
                tma_query,
                load_query_src[None, 0, 0, job],
                load_query_dst[None, 0],
                tma_bar_ptr=query_empty.barrier,
            )
            for tile in cutlass.range(tile_count):
                self.load_keys(key_producer.acquire_and_advance(), key_loads, tile)
        if warp == 4:
            query_consumer.wait_and_advance()
            for tile in cutlass.range(tile_count):
                self.mma(
                    mma,
                    acc_producer.acquire_and_advance(),
                    key_consumer.wait_and_advance(),
                    tile == 0,
                )
        if warp < 4:
            if cutlass.const_expr(self.scaled):
                for slot in cutlass.range(cutlass.min(ahead, tile_count)):
                    self.load_scales(scale_loads, slot, scale_stage, thread)
                    scale_stage.advance()
            scores = cute.make_tensor(output.iterator, flat_int64)  # [rows, keys]
            for tile in cutlass.range(tile_count):
                if cutlass.const_expr(self.scaled):
                    if tile + ahead < tile_count:
                        self.load_scales(scale_loads, tile + ahead, scale_stage, thread)
                        scale_stage.advance()
                acc_full = acc_consumer.wait_and_advance()
                cute.copy(
                    tiled_copy,
                    thread_copy.partition_S(
                        cute.make_tensor(
                            accumulator[None, None, None, acc_full.index].iterator,
                            resident_layout,
                        )
                    ),
                    registers,
                )
                cute.arch.fence_view_async_tmem_load()
                acc_full.release()
                # Register r is the job's query row r; a warp holds 32 keys.
                key = (first_tile + tile) * 128 + local_thread
                if key < job_end:
                    for row in cutlass.range_constexpr(self.rows):
                        scores[
                            (job * self.rows + row).to(cutlass.Int64) * key_rows + key
                        ] = registers[row]
        cute.arch.barrier()
        tmem.free(base)

    @cute.jit
    def load_keys(self, key_empty, loads: tuple, tile: cutlass.Int32):
        """Warp 5: TMA the run's tile `tile` into the stage `key_empty`."""
        tma_key, sources, destinations, tile_pages = loads
        for half in cutlass.range(2, unroll_full=True):
            cute.copy(
                tma_key,
                sources[None, 0, 0, tile_pages[tile * 2 + half]],
                destinations[None, half, key_empty.index],
                tma_bar_ptr=key_empty.barrier,
            )

    @cute.jit
    def mma(self, operands: tuple, acc_empty, key_full, first):
        """Warp 4: NVFP4's scales to TMEM (the query's if `first`), then the MMA
        of stage `key_full` into accumulator `acc_empty`; the stage is released
        and the accumulator committed."""
        tiled_mma, accumulator, fragment_a, fragment_b, scale_a, scale_b, copies = (
            operands
        )
        if cutlass.const_expr(self.scaled):
            (copy_qsf, qsf_src, qsf_dst), (copy_ksf, ksf_src, ksf_dst) = copies
            if first:
                cute.copy(copy_qsf, qsf_src[None, None, None, None, 0], qsf_dst)
            # cp.async wrote the key scales; the TMEM copy reads them through the
            # async proxy.
            cute.arch.fence_view_async_shared()
            cute.copy(
                copy_ksf, ksf_src[None, None, None, None, key_full.index], ksf_dst
            )
        partial = accumulator[None, None, None, acc_empty.index]
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # MMAs per tile: NVFP4 2, BF16 8.
        for depth in cutlass.range(cute.size(fragment_a, mode=[2]), unroll_full=True):
            if cutlass.const_expr(self.scaled):
                tiled_mma.set(tcgen05.Field.SFA, scale_a[None, None, depth].iterator)
                tiled_mma.set(tcgen05.Field.SFB, scale_b[None, None, depth].iterator)
            cute.gemm(
                tiled_mma,
                partial,
                fragment_a[None, None, depth, key_full.index],
                fragment_b[None, None, depth, 0],
                partial,
            )
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        key_full.release()
        acc_empty.commit()

    @cute.jit
    def load_scales(self, loads: tuple, tile, stage, thread: cutlass.Int32):
        """NVFP4, warps 0-3: once the key stage is free, thread k copies key k's
        two scale words (8 scales) of the run's tile `tile` by cp.async, counted
        in on the stage's full barrier. As the query's, word w goes to
        w * 128 + (k % 32) * 4 + k // 32 of the stage's 256."""
        key_scales, shared_key_scales, tile_pages, keys = loads
        keys.sync_object_empty.wait(stage.index, stage.phase)
        page = tile_pages[tile * 2 + thread // 64]
        source = cute.recast_ptr(key_scales.iterator, dtype=cutlass.Uint32) + (
            page.to(cutlass.Int64) * (key_scales.stride[0] // 4) + thread % 64 * 2
        )
        place = cute.recast_ptr(shared_key_scales.iterator, dtype=cutlass.Uint32) + (
            stage.index * 256 + (thread % 32) * 4 + thread // 32
        )
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cutlass.Uint32, num_bits_per_copy=32
        )
        for half in cutlass.range_constexpr(2):
            cute.copy(
                atom,
                cute.make_tensor(source + half, cute.make_layout(1)),
                cute.make_tensor(place + half * 128, cute.make_layout(1)),
            )
        cute.arch.cp_async_mbarrier_arrive_noinc(keys.producer_get_barrier(stage))


@cute.kernel
def index_k_write_kernel(
    key: cute.Tensor,
    slots: cute.Tensor,
    key_cache: cute.Tensor,
    key_scales: cute.Tensor,
):
    """Quantize key i to NVFP4 (an E4M3 scale per 16 values, their amax over 6,
    and 64 packed E2M1 bytes as 16 words) and write it at slot slots[i];
    negative slots are not written. Lanes 16 h .. 16 h + 15 of warp w take key
    2 w + h, lane l quantizing values 8 l .. 8 l + 7: lane pairs share a scale.
    The E4M3 path of rmsnorm_fp4quant (flashinfer/cute_dsl), with a unit global
    scale, so the largest value maps to about 6."""
    lane = cute.arch.lane_idx()
    part = lane % 16
    i = (cute.arch.block_idx()[0] * 4 + cute.arch.warp_idx()) * 2 + lane // 16
    key_cache = cute.recast_tensor(key_cache, cutlass.Int32)  # packed values as words
    key_scales = cute.recast_tensor(key_scales, cutlass.Uint8)
    slot = cutlass.Int64(-1)
    values = cute.make_rmem_tensor((8,), cutlass.Float32)
    values.fill(0.0)
    if i < key.shape[0]:  # past the keys, a lane still joins its pair's shuffle
        slot = slots[i].to(cutlass.Int64)
        loaded = cute.make_rmem_tensor((8,), key.element_type)
        loaded.store(cute.local_tile(key[i, None], (8,), (part,)).load())
        values.store(loaded.load().to(cutlass.Float32))
    amax = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(8):
        amax = fmax_f32(amax, fabs_f32(values[j]))
    amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
    scale = cvt_f32_to_e4m3(fdiv_rn(amax, cutlass.Float32(6.0)))
    inverse = cutlass.Float32(0.0)
    if fp8_e4m3_to_f32(scale) > 0:
        inverse = fdiv_rn(cutlass.Float32(1.0), fp8_e4m3_to_f32(scale))
    packed = cvt_e2m1x8_f32(*(values[j] * inverse for j in range(8)))
    if slot >= 0:
        page, row = slot // 64, slot % 64
        key_cache[page, row, part] = packed.to(cutlass.Int32)
        if part % 2 == 0:
            key_scales[page, row, part // 2] = scale.to(cutlass.Uint8)


# One scorer per query rows per request and key format, so each compiles once.
scorers = cache(PagedScorer)
