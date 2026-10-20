# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""The single-head paged scorer: FP32 dot products of query rows with their
request's paged keys, on SM100 tensor cores (tcgen05)."""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.experimental.primitives import exit as exit_thread

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
# The most key tiles a CTA runs over, staging their page indices (8 B a tile),
# so a run covers 32K keys.
TILES_PER_CTA = 256


class PagedScorer:
    """Scores of query rows against their request's paged BF16 keys.

    A job is ``rows`` (1 to 8) consecutive query rows that share block-table row
    ``tile_requests[j]`` and see keys ``< row_ends[j]``. Each job gets
    ``workers`` CTAs (the jobs are the grid's fastest-varying index), each
    owning one equal run of the job's 128-key tiles.

    A CTA streams its run through its key stages: a tile's two 64-key pages are
    TMA-loaded into a stage once the MMA of the stage's previous tile is done.
    Warps are specialized: LOAD_WARP loads the keys, MMA_WARP issues each tile's
    MMA into one of two accumulators, and the others store the scores. The keys are
    on the TMEM lanes (M = 128 keys, N = 8 query columns, ``rows`` of them real),
    so a warp stores 32 consecutive scores per row.
    """

    def __init__(self, rows: int):
        self.rows = rows  # query rows per job
        self.values = cutlass.BFloat16
        # The MMA tile (M, N, K) of TILE keys, MAX_ROWS query columns and DIM
        # values; two accumulators of MAX_ROWS columns in the 32 TMEM columns
        # tcgen05.alloc takes at least.
        self.tile, self.threads = (TILE, MAX_ROWS, DIM), THREADS
        self.accumulators, self.tmem_columns = 2, 32
        self.stages = 0  # key stages, set at compile time

    @cute.jit
    def __call__(
        self,
        stream: cuda.CUstream,
        query: cute.Tensor,
        key_cache: cute.Tensor,
        row_ends: cute.Tensor,
        tile_requests: cute.Tensor,
        block_table: cute.Tensor,
        output: cute.Tensor,
        key_rows: cutlass.Int32,
        workers: cutlass.Int32,
    ):
        jobs, pages = query.shape[0] // self.rows, key_cache.shape[0]
        matrix_key = cute.make_tensor(
            key_cache.iterator,
            cute.make_layout((PAGE, DIM, pages), stride=(DIM, 1, key_cache.stride[0])),
        )
        # A job's MAX_ROWS-row query box over its rows: TMA zero-fills the rest.
        matrix_query = cute.make_tensor(
            query.iterator,
            cute.make_layout((self.rows, DIM, jobs), stride=(DIM, 1, self.rows * DIM)),
        )
        k_major = cute.nvgpu.OperandMajorMode.K
        # BF16 with FP32 accumulation, K = 16: eight MMAs per 128-value tile.
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.values,
            self.values,
            k_major,
            k_major,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.tile[:2],
        )

        def operand(make, stages: int) -> tuple:
            """An operand's shared-memory layout and bytes: keys A, query B."""
            layout = make(tiled_mma, self.tile, self.values, stages)
            return layout, cute.size_in_bytes(self.values, layout)

        # Shared memory: the query, the barriers and TMEM address (padded to
        # 128 B), the page indices, then as many key stages as fit in half an
        # SM's 228 KB (a CTA's 227 KB plus the 1 KB reserved per CTA).
        capacity = utils.get_smem_capacity_in_bytes("sm_100") + SMEM_RESERVED_PER_CTA
        layout_query, fixed = operand(sm100_utils.make_smem_layout_b, 1)
        fixed += 8 * 2 * (1 + self.accumulators) + 4 + ALIGN - 1
        fixed += 4 * PAGES_PER_TILE * TILES_PER_CTA
        stage = 2 * 8 + operand(sm100_utils.make_smem_layout_a, 1)[1]
        self.stages = (capacity // CTAS_PER_SM - SMEM_RESERVED_PER_CTA - fixed) // stage
        layout_key, _ = operand(sm100_utils.make_smem_layout_a, self.stages)
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
            row_ends,
            tile_requests,
            block_table,
            output,
            layout_key,
            layout_query,
            key_pages,
            key_rows,
            workers,
        ).launch(
            grid=(jobs, workers, 1),
            block=(self.threads, 1, 1),
            smem=shared_bytes,
            min_blocks_per_mp=CTAS_PER_SM,
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
        row_ends: cute.Tensor,
        tile_requests: cute.Tensor,
        block_table: cute.Tensor,
        output: cute.Tensor,
        layout_key: cute.ComposedLayout,
        layout_query: cute.ComposedLayout,
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
        local_thread = thread % TILE  # the TMEM lane: a key
        # Flat views with 64-bit offsets: cache and output offsets can pass 2**31.
        flat_int64 = cute.make_layout(cutlass.Int64(2**31 - 1))
        pages = cute.make_tensor(block_table.iterator, flat_int64)
        # The job's request and last key; its rows share one length.
        request = cute.make_tensor(tile_requests.iterator, flat_int64)[job]
        job_end = cutlass.min(row_ends[job], key_rows)
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
        # The page of each of the run's pages.
        tile_pages = shared.allocate_tensor(
            cutlass.Int32, cute.make_layout(PAGES_PER_TILE * TILES_PER_CTA), 16
        )
        accumulator = tmem_pool.allocate_tensor(
            tiled_mma.make_fragment_C(
                cute.append(
                    tiled_mma.partition_shape_C(self.tile[:2]), self.accumulators
                )
            ).layout,
            cutlass.Float32,
        )
        fragment_key = tiled_mma.make_fragment_A(shared_key)
        fragment_query = tiled_mma.make_fragment_B(shared_query)
        # The run's page indices: 0 past the job's end, whose entries go unread.
        request_row = request.to(cutlass.Int64) * block_table.shape[1]
        for slot in cutlass.range(thread, PAGES_PER_TILE * tile_count, self.threads):
            block, page = first_tile * PAGES_PER_TILE + slot, cutlass.Int32(0)
            if block * PAGE < job_end:
                page = pages[request_row + block]
            tile_pages[slot] = page
        cute.arch.barrier()
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        key_producer, key_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.key_barriers.data_ptr(),
            num_stages=self.stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=cute.size_in_bytes(self.values, shared_key.layout) // self.stages,
        ).make_participants()
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
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * STORE_WARPS
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
        query_tile = cute.local_tile(tensor_query, (MAX_ROWS, DIM), (None, None, None))
        load_query_dst, load_query_src = cpasync.tma_partition(
            tma_query,
            0,
            cute.make_layout(1),
            cute.group_modes(shared_query, 0, 3),
            cute.group_modes(mma_thread.partition_B(query_tile), 0, 3),
        )
        load_query_dst = cute.filter_zeros(load_query_dst)
        load_query_src = cute.filter_zeros(load_query_src)
        # Each thread reads the MAX_ROWS query columns of its key from TMEM, where
        # an address is lane << 16 | column.
        atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition(MAX_ROWS), tcgen05.Pack.NONE),
            cutlass.Float32,
        )
        resident_layout = cute.make_layout((TILE, MAX_ROWS), stride=(1 << 16, 1))
        tiled_copy = tcgen05.make_tmem_copy(
            atom, cute.make_tensor(base, resident_layout)
        )
        thread_copy = tiled_copy.get_slice(local_thread)
        registers = cute.make_fragment_like(
            thread_copy.partition_D(cute.make_identity_tensor((TILE, MAX_ROWS))),
            cutlass.Float32,
        )

        # The load warp loads the query once, then the keys; the MMA warp issues
        # tile t's MMA into one accumulator while the store warps store tile
        # t - 1's scores from the other.
        if warp == LOAD_WARP:
            query_empty = query_producer.acquire_and_advance()
            cute.copy(
                tma_query,
                load_query_src[None, 0, 0, job],
                load_query_dst[None, 0],
                tma_bar_ptr=query_empty.barrier,
            )
            for tile in cutlass.range(tile_count):
                key_empty = key_producer.acquire_and_advance()
                for part in cutlass.range(PAGES_PER_TILE, unroll_full=True):
                    cute.copy(
                        tma_key,
                        load_key_src[
                            None, 0, 0, tile_pages[tile * PAGES_PER_TILE + part]
                        ],
                        load_key_dst[None, part, key_empty.index],
                        tma_bar_ptr=key_empty.barrier,
                    )
        if warp == MMA_WARP:
            query_consumer.wait_and_advance()
            for _ in cutlass.range(tile_count):
                acc_empty = acc_producer.acquire_and_advance()
                key_full = key_consumer.wait_and_advance()
                partial = accumulator[None, None, None, acc_empty.index]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for depth in cutlass.range(
                    cute.size(fragment_key, mode=[2]), unroll_full=True
                ):
                    cute.gemm(
                        tiled_mma,
                        partial,
                        fragment_key[None, None, depth, key_full.index],
                        fragment_query[None, None, depth, 0],
                        partial,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                key_full.release()
                acc_empty.commit()
        if warp < STORE_WARPS:
            scores = cute.make_tensor(output.iterator, flat_int64)  # [rows, keys]
            for tile in cutlass.range(tile_count):
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


# One scorer per query rows per request, so each compiles once.
scorers = cache(PagedScorer)
