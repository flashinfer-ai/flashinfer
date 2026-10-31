# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""The single-head paged scorer: FP32 dot products of query rows with their
request's paged BF16 or NVFP4 keys, on SM100 tensor cores (tcgen05), stored as
scores or selected against per-row thresholds."""

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

# Keys come in block-table pages of PAGE keys and are scored in tiles of TILE
# keys (PAGES_PER_TILE pages). A key has DIM values, the MMA's K. A decode job is
# up to MAX_ROWS query rows, the MMA's N (tcgen05's smallest). CTAS_PER_SM CTAs
# share an SM when they fit.
PAGE, TILE, DIM, MAX_ROWS, CTAS_PER_SM = 64, 128, 128, 8, 2
PAGES_PER_TILE = TILE // PAGE
# Decode's warps: STORE_WARPS store scores (32 TMEM lanes each), then one issues
# the MMAs and one loads the keys.
STORE_WARPS = 4
MMA_WARP, LOAD_WARP = STORE_WARPS, STORE_WARPS + 1
THREADS = 32 * (LOAD_WARP + 1)
ALIGN = 128  # bytes, of the shared-memory operands TMA writes
SMEM_RESERVED_PER_CTA = 1024  # bytes the driver keeps per CTA
# The most key tiles a CTA runs over, since it stages their page indices (8 bytes
# a tile): 256 in decode, so a run covers 32K keys, and 128 in select.
TILES_PER_CTA = 256
SELECT_TILES_PER_CTA = 128
# NVFP4 values (E2M1, two to a byte) come with one E4M3 scale per 16.
SCALE_TYPE = cutlass.Float8E4M3FN
SCALE_VECTOR = 16
# NVFP4 keys: DIM // 2 bytes of E2M1 values, SCALES_PER_KEY E4M3 scales
# (SCALE_WORDS 32-bit words). The MXF4 MMA's K is half a key.
PACKED_DIM, MXF4_K = DIM // 2, DIM // 2
SCALES_PER_KEY = DIM // SCALE_VECTOR
SCALE_WORDS = SCALES_PER_KEY // 4
E2M1_MAX = 6.0
# Select: a job is a TILE-row query tile on the TMEM lanes, scored by
# SELECT_THREADS threads; each warpgroup compares its rows' scores of KEY_GROUP
# keys of every tile. A selected (key, score) pair takes PAIR_BYTES.
SELECT_THREADS = 512
KEY_GROUP = TILE // (SELECT_THREADS // TILE)
PAIR_BYTES = 8
COUNTERS = 2  # a select row's candidate and tie counts
# index_k_write_kernel: KEY_LANES lanes quantize a key, LANE_VALUES values each,
# so LANES_PER_SCALE lanes share a scale; a CTA of TILE threads writes
# WRITE_KEYS_PER_CTA keys.
KEY_LANES = 16
LANE_VALUES = DIM // KEY_LANES
LANES_PER_SCALE = SCALE_VECTOR // LANE_VALUES
WRITE_KEYS_PER_CTA = TILE // KEY_LANES


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

    A job is ``rows`` consecutive query rows that share block-table row
    ``tile_requests[j]``. In "decode", a job is 1 to 8 rows that all see keys
    below ``row_ends[j]``. In "select", it is a 128-row query tile whose row r
    sees keys below ``row_ends[r]``. Each job gets ``workers`` CTAs, the first
    ``wide`` jobs one more (jobs vary fastest in the grid), and each CTA owns an
    equal run of the job's 128-key tiles below its longest row end.

    A CTA streams its run through its key stages: a tile's two 64-key pages are
    TMA-loaded into a stage once the MMA of the stage's previous tile is done.
    Warps are specialized: LOAD_WARP loads the keys, MMA_WARP issues each tile's
    MMA into one of two accumulators, and the others store the scores. The keys are
    on the TMEM lanes (M = 128 keys, N = 8 query columns, ``rows`` of them real),
    so a warp stores 32 consecutive scores per row. With NVFP4, threads 0-127
    also copy each tile's key scales by cp.async into its stage, counted in on
    the stage's full barrier.

    "select" puts the queries on the TMEM lanes (M = 128 rows, N = 128 keys)
    over 512 threads, each comparing its row's 32 scores of a tile (warpgroup g
    owns keys 32 g .. 32 g + 31) to the row's threshold.
    """

    def __init__(self, mode: str, rows: int, nvfp4: bool):
        self.decode = mode == "decode"
        self.rows = rows  # query rows per job
        self.values = cutlass.Float4E2M1FN if nvfp4 else cutlass.BFloat16
        self.scaled = nvfp4
        # The MMA tile (M, N, K) of TILE keys, MAX_ROWS query columns and DIM
        # values; two accumulators of MAX_ROWS columns and NVFP4's scale blocks in
        # the 32 TMEM columns tcgen05.alloc takes at least.
        self.tile, self.threads = (TILE, MAX_ROWS, DIM), THREADS
        self.accumulators, self.tmem_columns = 2, 32
        self.max_tiles = TILES_PER_CTA
        if not self.decode:  # one accumulator in 256 columns, N = 128 keys
            self.tile, self.threads = (TILE, TILE, DIM), SELECT_THREADS
            self.accumulators, self.tmem_columns = 1, 256
            self.max_tiles = SELECT_TILES_PER_CTA
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
        thresholds: Optional[cute.Tensor],
        counters: Optional[cute.Tensor],
        output: cute.Tensor,
        key_rows: cutlass.Int32,
        workers: cutlass.Int32,
        wide: cutlass.Int32,
        ties: cutlass.Int32,
    ):
        jobs, pages = query.shape[0] // self.rows, key_cache.shape[0]
        packing = key_cache.element_type.width // self.values.width  # NVFP4: 2
        matrix_key = cute.make_tensor(
            cute.recast_ptr(key_cache.iterator, dtype=self.values),
            cute.make_layout(
                (PAGE, DIM, pages), stride=(DIM, 1, packing * key_cache.stride[0])
            ),
        )
        # A job's MAX_ROWS-row query box over its rows: TMA zero-fills the rest.
        matrix_query = cute.make_tensor(
            cute.recast_ptr(query.iterator, dtype=self.values),
            cute.make_layout((self.rows, DIM, jobs), stride=(DIM, 1, self.rows * DIM)),
        )
        k_major = cute.nvgpu.OperandMajorMode.K
        if cutlass.const_expr(self.scaled):
            tiled_mma = cute.make_tiled_mma(
                tcgen05.MmaMXF4NVF4Op(
                    SCALE_TYPE,
                    self.tile[:2] + (MXF4_K,),
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
        # Decode's keys are the MMA's A operand, the query B; select swaps them.
        make_key = (
            sm100_utils.make_smem_layout_a,
            blockscaled_utils.make_smem_layout_sfa,
        )
        make_query = (
            sm100_utils.make_smem_layout_b,
            blockscaled_utils.make_smem_layout_sfb,
        )
        if cutlass.const_expr(not self.decode):
            make_key, make_query = make_query, make_key

        def operand(make: tuple, stages: int) -> tuple:
            """An operand's shared-memory layout, NVFP4's scale layout, and bytes."""
            layout = make[0](tiled_mma, self.tile, self.values, stages)
            scales, size = None, cute.size_in_bytes(self.values, layout)
            if cutlass.const_expr(self.scaled):
                scales = make[1](tiled_mma, self.tile, SCALE_VECTOR, stages)
                size += cute.size_in_bytes(SCALE_TYPE, scales)
            return layout, scales, size

        # Shared memory: the query, the barriers and TMEM address (padded to
        # 128 B), the page indices, select's 32 staged scores per thread, then as
        # many key stages as fit in half an SM's 228 KB (a CTA's 227 KB plus the
        # 1 KB reserved per CTA), or in all of it if two CTAs do not fit.
        capacity = utils.get_smem_capacity_in_bytes("sm_100") + SMEM_RESERVED_PER_CTA
        layout_query, scales_query, fixed = operand(make_query, 1)
        fixed += 8 * 2 * (1 + self.accumulators) + 4 + ALIGN - 1
        fixed += 4 * PAGES_PER_TILE * self.max_tiles
        fixed += 0 if self.decode else 4 * KEY_GROUP * self.threads
        stage = 2 * 8 + operand(make_key, 1)[2]
        ctas = min(CTAS_PER_SM, capacity // (fixed + stage + SMEM_RESERVED_PER_CTA))
        self.stages = (capacity // ctas - SMEM_RESERVED_PER_CTA - fixed) // stage
        layout_key, scales_key, _ = operand(make_key, self.stages)
        shared_bytes = fixed + self.stages * stage
        # A key tile as TMA writes it, then the stages: the key operand's swizzled
        # layout of TILE keys, keys first (as CUTLASS tiles K-major operands), so
        # a page is one PAGES_PER_TILE-th of it.
        key_tile = cute.slice_(
            sm100_utils.make_smem_layout(k_major, (TILE, DIM), self.values, 1),
            (None, None, 0),
        )
        layout_key_page = cute.get(cute.tiled_divide(key_tile, (PAGE, DIM)), mode=[0])
        key_pages = cute.select(cute.tiled_divide(key_tile.outer, (PAGE, DIM)), [0, 1])
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
            (PAGE, DIM),
        )
        make_tma_query = cute.nvgpu.make_tiled_tma_atom_B
        if cutlass.const_expr(not self.decode):
            make_tma_query = cute.nvgpu.make_tiled_tma_atom_A
        tma_query, tensor_query = make_tma_query(
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
            thresholds,
            counters,
            output,
            layout_key,
            layout_query,
            scales_key,
            scales_query,
            key_pages,
            key_rows,
            workers,
            wide,
            ties,
        ).launch(
            grid=(jobs, workers + cutlass.Int32(wide > 0), 1),
            block=(self.threads, 1, 1),
            smem=shared_bytes,
            min_blocks_per_mp=ctas,
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
        thresholds: Optional[cute.Tensor],
        counters: Optional[cute.Tensor],
        output: cute.Tensor,
        layout_key: cute.ComposedLayout,
        layout_query: cute.ComposedLayout,
        scales_key: Optional[cute.Layout],
        scales_query: Optional[cute.Layout],
        key_pages: cute.Layout,
        key_rows: cutlass.Int32,
        workers: cutlass.Int32,
        wide: cutlass.Int32,
        ties: cutlass.Int32,
    ):
        thread, _, _ = cute.arch.thread_idx()
        job, worker, _ = cute.arch.block_idx()
        # The grid has a CTA more per job than all but the first `wide` jobs use;
        # the others' extra CTA owns no tiles and exits below.
        workers = cutlass.Int32(cutlass.select_(job < wide, workers + 1, workers))
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
        local_thread = thread % TILE  # the TMEM lane: a key or a query row
        key_group = thread // TILE  # select: KEY_GROUP key columns of every tile
        # Flat views with 64-bit offsets: cache and output offsets can pass 2**31.
        flat_int64 = cute.make_layout(cutlass.Int64(2**31 - 1))
        pages = cute.make_tensor(block_table.iterator, flat_int64)
        # The job's request and last key; decode rows share one length.
        request = cute.make_tensor(tile_requests.iterator, flat_int64)[job]
        job_end = cutlass.Int32(0)
        if cutlass.const_expr(self.decode):
            job_end = row_ends[job]
        else:  # a select tile's longest row
            for row_group in cutlass.range(self.rows // 32, unroll_full=True):
                row = job * self.rows + row_group * 32 + thread % 32
                job_end = cutlass.max(job_end, row_ends[row])
            job_end = cute.arch.warp_redux_sync(job_end, "max")
        job_end = cutlass.min(job_end, key_rows)
        my_row = job * self.rows + local_thread
        my_end, my_threshold = job_end, cutlass.Float32(0)
        if cutlass.const_expr(not self.decode):
            my_end = cutlass.min(row_ends[my_row], key_rows)
            my_threshold = thresholds[my_row]
        # This CTA's run of key tiles.
        job_tiles = (job_end + TILE - 1) // TILE
        run = (job_tiles + workers - 1) // workers
        first_tile = worker * run
        tile_count = cutlass.min(run, job_tiles - first_tile)
        if tile_count <= 0:
            if warp == 0:
                # tmem.free() counts columns while tracing; the final one needs them.
                cute.arch.dealloc_tmem(base, self.tmem_columns)
            exit_thread()
        shared_key = shared.allocate_tensor(
            self.values, layout_key.outer, ALIGN, layout_key.inner
        )
        shared_query = shared.allocate_tensor(
            self.values, layout_query.outer, ALIGN, layout_query.inner
        )
        shared_key_scales = shared_query_scales = scale_key = scale_query = None
        if cutlass.const_expr(self.scaled):
            shared_key_scales = shared.allocate_tensor(SCALE_TYPE, scales_key, ALIGN)
            shared_query_scales = shared.allocate_tensor(
                SCALE_TYPE, scales_query, ALIGN
            )
        # The page of each of the run's pages.
        tile_pages = shared.allocate_tensor(
            cutlass.Int32, cute.make_layout(PAGES_PER_TILE * self.max_tiles), 16
        )
        staged = None
        if cutlass.const_expr(not self.decode):
            staged = shared.allocate_tensor(
                cutlass.Float32, cute.make_layout(KEY_GROUP * self.threads), 16
            )
        accumulator = tmem_pool.allocate_tensor(
            tiled_mma.make_fragment_C(
                cute.append(
                    tiled_mma.partition_shape_C(self.tile[:2]), self.accumulators
                )
            ).layout,
            cutlass.Float32,
        )
        # Decode's keys are the MMA's A operand, select's queries.
        make_key_sf = blockscaled_utils.make_tmem_layout_sfa
        make_query_sf = blockscaled_utils.make_tmem_layout_sfb
        if cutlass.const_expr(not self.decode):
            make_key_sf, make_query_sf = make_query_sf, make_key_sf
        if cutlass.const_expr(self.scaled):  # the pool asserts they fit
            scale_key = tmem_pool.allocate_tensor(
                make_key_sf(
                    tiled_mma,
                    self.tile,
                    SCALE_VECTOR,
                    cute.slice_(scales_key, (None, None, None, 0)),
                ),
                SCALE_TYPE,
            )
            scale_query = tmem_pool.allocate_tensor(
                make_query_sf(
                    tiled_mma,
                    self.tile,
                    SCALE_VECTOR,
                    cute.slice_(scales_query, (None, None, None, 0)),
                ),
                SCALE_TYPE,
            )
        if cutlass.const_expr(self.decode):
            fragment_key = tiled_mma.make_fragment_A(shared_key)
            fragment_query = tiled_mma.make_fragment_B(shared_query)
        else:
            fragment_query = tiled_mma.make_fragment_A(shared_query)
            fragment_key = tiled_mma.make_fragment_B(shared_key)
        # The run's page indices: 0 past the job's end, whose entries go unread.
        request_row = request.to(cutlass.Int64) * block_table.shape[1]
        for slot in cutlass.range(thread, PAGES_PER_TILE * tile_count, self.threads):
            block, page = first_tile * PAGES_PER_TILE + slot, cutlass.Int32(0)
            if block * PAGE < job_end:
                page = pages[request_row + block]
            tile_pages[slot] = page
        # NVFP4's query scales in the MMA's order: row r's scale words w = 0, 1
        # at w * 128 + (r % 32) * 4 + r // 32.
        if cutlass.const_expr(self.scaled):
            words = cute.make_tensor(
                cute.recast_ptr(shared_query_scales.iterator, dtype=cutlass.Uint32),
                cute.make_layout(SCALE_WORDS * TILE),
            )
            if thread < self.rows:
                packed = cute.make_tensor(
                    cute.recast_ptr(query_scales.iterator, dtype=cutlass.Uint64),
                    flat_int64,
                )[job * self.rows + thread]
                words[(thread % 32) * 4 + thread // 32] = cutlass.Uint32(packed)
                words[TILE + (thread % 32) * 4 + thread // 32] = cutlass.Uint32(
                    packed >> 32
                )
            cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        # Select: this row's candidate and tie counters, and the address of its
        # first (key index, score) pair; the candidates follow `ties` tie slots.
        if cutlass.const_expr(not self.decode):
            capacity = output.shape[1] - ties
            row_counters = counters.iterator + my_row * COUNTERS
            row_pairs = cutlass.Int64(output.iterator.toint()) + PAIR_BYTES * (
                my_row.to(cutlass.Int64) * output.shape[1]
            )
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # A key stage is full once its TMA bytes and NVFP4's scale copies land.
        key_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.key_barriers.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, TILE + 1 if self.scaled else 1
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
            consumer_group=pipeline.CooperativeGroup(  # decode: warps 0-3
                pipeline.Agent.Thread, 32 * STORE_WARPS if self.decode else self.threads
            ),
        ).make_participants()
        mma_thread = tiled_mma.get_slice(0)
        global_key = cute.local_tile(tensor_key, (PAGE, DIM), (None, None, None))
        load_key_dst, load_key_src = cpasync.tma_partition(
            tma_key,
            0,
            cute.make_layout(1),
            cute.make_tensor(shared_key.iterator, key_pages),
            cute.group_modes(global_key, 0, 2),
        )
        query_box = MAX_ROWS if self.decode else TILE  # the MMA's query N or M
        query_tile = cute.local_tile(tensor_query, (query_box, DIM), (None,) * 3)
        partition = mma_thread.partition_B if self.decode else mma_thread.partition_A
        load_query_dst, load_query_src = cpasync.tma_partition(
            tma_query,
            0,
            cute.make_layout(1),
            cute.group_modes(shared_query, 0, 3),
            cute.group_modes(partition(query_tile), 0, 3),
        )
        load_query_dst = cute.filter_zeros(load_query_dst)
        load_query_src = cute.filter_zeros(load_query_src)
        # Each thread reads its lane's scores from TMEM, where an address is
        # lane << 16 | column: decode, the 8 query columns of its key; select,
        # its row's 32 scores of its warpgroup's key group.
        columns = MAX_ROWS if self.decode else KEY_GROUP
        atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(columns), tcgen05.Pack.NONE),
            cutlass.Float32,
        )
        resident_layout = cute.make_layout((TILE, columns), stride=(1 << 16, 1))
        resident = cute.make_tensor(base + key_group * columns, resident_layout)
        tiled_copy = tcgen05.make_tmem_copy(atom, resident)
        thread_copy = tiled_copy.get_slice(local_thread)
        registers = cute.make_fragment_like(
            thread_copy.partition_D(cute.make_identity_tensor((TILE, columns))),
            cutlass.Float32,
        )
        copies = scale_loads = None
        if cutlass.const_expr(self.scaled):
            copies = (
                partition_scales(shared_query_scales, scale_query),
                partition_scales(shared_key_scales, scale_key),
            )
            scale_loads = (key_scales, shared_key_scales, tile_pages, key_pipeline)
        # Decode's keys are the MMA's A operand, select's queries.
        mma: tuple = (tiled_mma, accumulator, fragment_key, fragment_query)
        mma += (scale_key, scale_query, copies)
        if cutlass.const_expr(not self.decode):
            mma = (tiled_mma, accumulator, fragment_query, fragment_key)
            mma += (scale_query, scale_key, copies)
        key_loads = (tma_key, load_key_src, load_key_dst, tile_pages)
        # Tile t + stages - 1 loads into the stage of tile t - 1, whose MMA is done.
        ahead = self.stages - 1
        scale_stage = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.stages
        )

        # Select runs its tiles in lockstep. Step s reads tile s - 1's scores,
        # issues tile s's MMA (warp 0 also loads the query and keys), and then
        # selects from tile s - 1's scores. In decode, warp 5 loads the query once
        # and then the keys, and warp 4 issues tile t's MMA into one accumulator
        # while warps 0-3 store tile t - 1's scores from the other.
        if cutlass.const_expr(not self.decode):
            source = thread_copy.partition_S(resident)
            selection = (row_counters, row_pairs, ties, capacity, staged)
            if warp == 0:
                query_empty = query_producer.acquire_and_advance()
                cute.copy(
                    tma_query,
                    load_query_src[None, 0, 0, job],
                    load_query_dst[None, 0],
                    tma_bar_ptr=query_empty.barrier,
                )
            for step in cutlass.range(-ahead, tile_count + 1):
                if step > 0:
                    acc_full = acc_consumer.wait_and_advance()
                    cute.copy(tiled_copy, source, registers)
                    cute.arch.fence_view_async_tmem_load()
                    acc_full.release()
                if step + ahead >= 0 and step + ahead < tile_count:
                    if warp == 0:
                        key_empty = key_producer.acquire_and_advance()
                        self.load_keys(key_empty, key_loads, step + ahead)
                    if cutlass.const_expr(self.scaled):
                        if thread < TILE:
                            tile = step + ahead
                            self.load_scales(scale_loads, tile, scale_stage, thread)
                            scale_stage.advance()
                if warp == 0 and step >= 0 and step < tile_count:
                    if step == 0:
                        query_consumer.wait_and_advance()
                    self.mma(
                        mma,
                        acc_producer.acquire_and_advance(),
                        key_consumer.wait_and_advance(),
                        step == 0,
                    )
                if step > 0:
                    tile = first_tile + step - 1
                    key_base = tile * TILE + key_group * KEY_GROUP
                    limit = cutlass.min(cutlass.max(my_end - key_base, 0), KEY_GROUP)
                    # Keys at or past my_end are neither; only tiles that
                    # straddle a row end (of any row in this warp) check it.
                    if cute.arch.vote_all_sync(tile * TILE + TILE <= my_end):
                        self.select_keys(
                            registers,
                            my_threshold,
                            KEY_GROUP,
                            key_base,
                            selection,
                            thread,
                        )
                    else:
                        self.select_keys(
                            registers, my_threshold, limit, key_base, selection, thread
                        )
        elif warp == LOAD_WARP:
            query_empty = query_producer.acquire_and_advance()
            cute.copy(
                tma_query,
                load_query_src[None, 0, 0, job],
                load_query_dst[None, 0],
                tma_bar_ptr=query_empty.barrier,
            )
            for tile in cutlass.range(tile_count):
                self.load_keys(key_producer.acquire_and_advance(), key_loads, tile)
        elif warp == MMA_WARP:
            query_consumer.wait_and_advance()
            for tile in cutlass.range(tile_count):
                self.mma(
                    mma,
                    acc_producer.acquire_and_advance(),
                    key_consumer.wait_and_advance(),
                    tile == 0,
                )
        elif warp < STORE_WARPS:
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
                key = (first_tile + tile) * TILE + local_thread
                if key < job_end:
                    for row in cutlass.range_constexpr(self.rows):
                        scores[
                            (job * self.rows + row).to(cutlass.Int64) * key_rows + key
                        ] = registers[row]
        cute.arch.barrier()
        tmem.free(base)

    @cute.jit
    def load_keys(self, key_empty, loads: tuple, tile: cutlass.Int32):
        """Decode's warp 5, select's warp 0: TMA the run's tile `tile` into the
        stage `key_empty`."""
        tma_key, sources, destinations, tile_pages = loads
        for part in cutlass.range(PAGES_PER_TILE, unroll_full=True):
            cute.copy(
                tma_key,
                sources[None, 0, 0, tile_pages[tile * PAGES_PER_TILE + part]],
                destinations[None, part, key_empty.index],
                tma_bar_ptr=key_empty.barrier,
            )

    @cute.jit
    def mma(self, operands: tuple, acc_empty, key_full, first):
        """Decode's warp 4, select's warp 0: NVFP4's scales to TMEM (the query's
        if `first`), then the MMA of stage `key_full` into accumulator
        `acc_empty`; the stage is released and the accumulator committed."""
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
        stage_a, stage_b = (key_full.index, 0) if self.decode else (0, key_full.index)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        # MMAs per tile: NVFP4 2, BF16 8.
        for depth in cutlass.range(cute.size(fragment_a, mode=[2]), unroll_full=True):
            if cutlass.const_expr(self.scaled):
                tiled_mma.set(tcgen05.Field.SFA, scale_a[None, None, depth].iterator)
                tiled_mma.set(tcgen05.Field.SFB, scale_b[None, None, depth].iterator)
            cute.gemm(
                tiled_mma,
                partial,
                fragment_a[None, None, depth, stage_a],
                fragment_b[None, None, depth, stage_b],
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
            page.to(cutlass.Int64) * (key_scales.stride[0] // 4)
            + thread % PAGE * SCALE_WORDS
        )
        place = cute.recast_ptr(shared_key_scales.iterator, dtype=cutlass.Uint32) + (
            stage.index * SCALE_WORDS * TILE + (thread % 32) * 4 + thread // 32
        )
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cutlass.Uint32, num_bits_per_copy=32
        )
        for word in cutlass.range_constexpr(SCALE_WORDS):
            cute.copy(
                atom,
                cute.make_tensor(source + word, cute.make_layout(1)),
                cute.make_tensor(place + word * TILE, cute.make_layout(1)),
            )
        cute.arch.cp_async_mbarrier_arrive_noinc(keys.producer_get_barrier(stage))

    @cute.jit
    def select_keys(
        self,
        registers: cute.Tensor,
        threshold: cutlass.Float32,
        limit: cutlass.Int32,
        key_base: cutlass.Int32,
        selection: tuple,
        thread: cutlass.Int32,
    ):
        """A thread's 32 scores of its row, the first `limit` live, as candidates
        (above the threshold) and ties (neither above nor below): one atomic add
        per class reserves slots, and each written slot gets one 8-byte (key,
        score) pair; past the row's slots they are counted, not written. The
        scores are staged in the thread's 128 shared-memory bytes so the
        candidate loop runs once per set mask bit."""
        row_counters, row_pairs, ties, capacity, staged = selection
        mask = cutlass.Uint32(0)
        tie_seen = cutlass.Boolean(False)
        for j in cutlass.range_constexpr(KEY_GROUP):
            if j < limit:
                mask |= cutlass.Uint32(threshold < registers[j]) << j
                tie_seen |= self.is_tie(registers[j], threshold)
        # Score j at word j ^ (4 * (thread % 8)): the 8 threads of a quarter
        # warp store their 16-byte chunks to distinct banks.
        swizzle = (thread % 8) * 4
        mine = thread * KEY_GROUP
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
        )
        for chunk in cutlass.range_constexpr(8):
            cute.copy(
                atom,
                cute.make_tensor(registers.iterator + chunk * 4, cute.make_layout(4)),
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        (staged.iterator + mine + (chunk * 4 ^ swizzle)).toint(),
                        cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout(4),
                ),
            )
        if tie_seen:
            tied = cutlass.Int32(0)
            for j in cutlass.range_constexpr(KEY_GROUP):
                if j < limit:
                    tied += cutlass.Int32(self.is_tie(registers[j], threshold))
            slot = cute.arch.atomic_add(
                (row_counters + 1).llvm_ptr, tied, sem="relaxed", scope="gpu"
            )
            address = row_pairs + PAIR_BYTES * (ties - 1 - slot).to(cutlass.Int64)
            for j in cutlass.range_constexpr(KEY_GROUP):
                if j < limit:
                    if self.is_tie(registers[j], threshold):
                        if slot < ties:
                            self.store_pair(
                                address, key_base + j, registers[j], threshold
                            )
                        slot += 1
                        address -= PAIR_BYTES
        above = cute.arch.popc(mask)
        slot = cute.arch.atomic_add(
            row_counters.llvm_ptr, above, sem="relaxed", scope="gpu"
        )
        address = row_pairs + PAIR_BYTES * (ties + slot).to(cutlass.Int64)
        for _ in cutlass.range(cutlass.min(above, cutlass.max(capacity - slot, 0))):
            j = cutlass.Int32(cute.arch.bfind(mask))
            mask ^= cutlass.Uint32(1) << j
            self.store_pair(
                address, key_base + j, staged[mine + (j ^ swizzle)], threshold
            )
            address += PAIR_BYTES

    @cute.jit
    def is_tie(
        self, score: cutlass.Float32, threshold: cutlass.Float32
    ) -> cutlass.Boolean:
        """Neither above nor below the threshold: equal, or NaN."""
        return not (threshold < score) and not (score < threshold)

    @cute.jit
    def store_pair(
        self,
        address: cutlass.Int64,
        key: cutlass.Int32,
        score: cutlass.Float32,
        lower_bound: cutlass.Float32,
    ):
        # The score's distance above the lower bound where exact, up to twice it
        # (Sterbenz lemma); larger scores stay, above every shifted one, so the
        # order holds. A negative lower bound never passes the check.
        if score <= lower_bound + lower_bound:
            score -= lower_bound
        pair = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int64, address, cute.AddressSpace.gmem, assumed_align=8
            ),
            cute.make_layout(1),
        )
        bits = score.bitcast(cutlass.Uint32).to(cutlass.Int64)
        pair[0] = (bits << 32) | key.to(cutlass.Uint32).to(cutlass.Int64)


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
    part = lane % KEY_LANES
    i = (
        cute.arch.block_idx()[0] * WRITE_KEYS_PER_CTA
        + (cute.arch.warp_idx() * 32 + lane) // KEY_LANES
    )
    key_cache = cute.recast_tensor(key_cache, cutlass.Int32)  # packed values as words
    key_scales = cute.recast_tensor(key_scales, cutlass.Uint8)
    slot = cutlass.Int64(-1)
    values = cute.make_rmem_tensor((LANE_VALUES,), cutlass.Float32)
    values.fill(0.0)
    if i < key.shape[0]:  # past the keys, a lane still joins its pair's shuffle
        slot = slots[i].to(cutlass.Int64)
        loaded = cute.make_rmem_tensor((LANE_VALUES,), key.element_type)
        loaded.store(cute.local_tile(key[i, None], (LANE_VALUES,), (part,)).load())
        values.store(loaded.load().to(cutlass.Float32))
    amax = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(LANE_VALUES):
        amax = fmax_f32(amax, fabs_f32(values[j]))
    amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
    scale = cvt_f32_to_e4m3(fdiv_rn(amax, cutlass.Float32(E2M1_MAX)))
    inverse = cutlass.Float32(0.0)
    if fp8_e4m3_to_f32(scale) > 0:
        inverse = fdiv_rn(cutlass.Float32(1.0), fp8_e4m3_to_f32(scale))
    packed = cvt_e2m1x8_f32(*(values[j] * inverse for j in range(LANE_VALUES)))
    if slot >= 0:
        page, row = slot // PAGE, slot % PAGE
        key_cache[page, row, part] = packed.to(cutlass.Int32)
        if part % 2 == 0:
            key_scales[page, row, part // 2] = scale.to(cutlass.Uint8)


# One scorer per mode, query rows per job and key format, so each compiles once.
scorers = cache(PagedScorer)
