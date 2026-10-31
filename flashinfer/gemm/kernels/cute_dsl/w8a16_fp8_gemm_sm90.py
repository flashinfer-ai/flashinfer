# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Structure after the CuTe-DSL Hopper persistent dense GEMM example
# (examples/python/CuTeDSL/hopper/dense_gemm_persistent.py, BSD-3-Clause) and
# CUTLASS's sm90_mma_tma_gmma_rs_warpspecialized_mixed_input mainloop.

"""Hopper (sm_90a) W8A16 GEMM: ``out[m, n] = act[m, K] @ (e4m3 weight[n, K] * scale).T
(+ bias[n])`` with the dequantization in the mainloop.

Operand mapping as the Blackwell kernel: the weight is MMA operand A (kernel M = weight
rows), the activation operand B (kernel N = activation rows), C[n, m] column-major is
``out`` row-major.

Persistent, warp-specialized: a producer warpgroup whose warp 0 TMA-loads the fp8
weight tile and the 16-bit activation tile of every k-tile, and one or two consumer
warpgroups.  A consumer warpgroup pulls its 64 weight rows of a k-block from smem into
registers, converts e4m3 -> bf16/fp16 (``w8a16_fp8_gemm_sm100_utils.convert_e4m3_fragment``)
and issues wgmma with A from registers and B from smem, software-pipelined over the
k-blocks of a stage as CUTLASS's register-sourced mixed-input mainloop does.  The
epilogue applies the per-tensor scale and the per-weight-row bias in fp32, converts and
TMA-stores (tile_m x 32) subtiles.

Tiles: (128, A) with two consumer warpgroups (the C++ cooperative tiles), (64, A) with one
(the C++ ping-pong tiles run as single-warpgroup tiles here).  The tile walk is the C++
grouped order: consecutive work items take the weight tiles of one activation tile;
``group_size`` activation tiles form an L2 group that walks all weight tiles together.

Work decomposition: the first ``body_items`` tiles are plain (whole K, output written
by the epilogue); the tiles after them are the *tail*, each cut into ``tail_splits``
K-slices that write fp32 partial tiles into a compact workspace (one tile_m x tile_a
block per (tile, slice)).  The slices of a tile then reduce it among themselves: each
CTA publishes its block, counts in on the tile's counter, waits for all slices and
sums its own share of the tile's columns, applying scale and bias and casting into
the output (the CUTLASS stream-K "separate reduction" shape; the slices of one tile
run concurrently, so every CTA holds at most one tail item).  Split-K is body 0
(every tile split); the stream-K codes run the whole waves plain and split only the
last partial wave over the idle SMs.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup

from .w8a16_fp8_gemm_sm100_utils import (
    atomic_add_acq_rel_gpu_u64,
    bulk_store_complete_fence,
    convert_e4m3_words,
    ld_acquire_gpu_u64,
    load_e4m3_unit,
    nanosleep,
)

# The tail tiles' arrival counters only ever grow.  Every slice of a tile adds
# ``TAIL_COUNTER_STRIDE // splits``, so a complete tile advances its counter by exactly
# one stride whatever its split count (the stride is lcm(1..16); 16 is the most slices
# a tile is cut into), and a slice waits for the next stride boundary above the value
# it saw.  No reset, so no second round trip, and a counter that a different launch
# used with another split count is still at a boundary.  Launches to overflow a
# 64-bit counter: 2^63 / 720720, never.
TAIL_COUNTER_STRIDE = 720720


@cute.jit
def grouped_tile_at(
    tile_idx: cutlass.Int32,
    weight_tiles: cutlass.Int32,
    act_tiles: cutlass.Int32,
    group: cutlass.Int32,
):
    """Tile ``tile_idx`` of the C++ grouped walk order -> (weight tile, activation
    tile): ``group`` activation tiles form a group whose weight tiles are walked
    together (weight tile fast within the group), the last group narrower when the
    count does not divide.  Shared by the GEMM kernel and the tail reduce."""
    full_groups = act_tiles // group
    group_tiles = group * weight_tiles
    full_tiles = full_groups * group_tiles
    m_tile = cutlass.Int32(0)
    n_tile = cutlass.Int32(0)
    if tile_idx < full_tiles:
        group_idx = tile_idx // group_tiles
        tig = tile_idx % group_tiles
        m_tile = tig // group
        n_tile = group_idx * group + tig % group
    else:
        rem = act_tiles - full_groups * group
        t2 = tile_idx - full_tiles
        m_tile = t2 // rem
        n_tile = full_groups * group + t2 % rem
    return m_tile, n_tile


class Sm90W8A16Fp8GemmKernel:
    def __init__(
        self,
        tile_shape_mn: tuple[int, int],
        has_bias: bool,
        group_size: int = 1,
        has_tail: bool = False,
        enable_pdl: bool = False,
    ):
        # tile_shape_mn: (weight rows, activation rows) per CTA tile; 128 weight rows run
        # on two consumer warpgroups, 64 on one.  Activation rows: a multiple of 8, at
        # most 256 (the wgmma N extent).
        tile_w, tile_a = tile_shape_mn
        if tile_w not in (64, 128):
            raise ValueError(
                f"W8A16 sm90: weight rows per tile must be 64 or 128, not {tile_w}"
            )
        if tile_a % 8 != 0 or not 8 <= tile_a <= 256:
            raise ValueError(
                f"W8A16 sm90: activation rows per tile {tile_a} not a multiple of 8 <= 256"
            )
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.tile_shape_mnk = (tile_w, tile_a, 64)
        self.has_bias = has_bias
        self.group_size = group_size
        # has_tail: the kernel also carries the fp32 partial-tile epilogue (split tail
        # work items write to the workspace; the reduce applies scale / bias).
        self.has_tail = has_tail
        self.enable_pdl = enable_pdl
        self.acc_dtype = cutlass.Float32
        # EXPERIMENT: cut the tail fix-up: 1 = partial store only, 2 = + publish, count
        # and spin, 3 = + reduce loads (no output store), 0 = everything.
        import os

        self.fixup_cut = int(os.environ.get("W8A16_DSL_FIXUP_CUT", "0"))
        # EXPERIMENT: 0 disables the shared-item k-split under ping-pong.
        self.ksplit_enabled = os.environ.get("W8A16_DSL_KSPLIT", "0") == "1"

        # Two consumer warpgroups either way.  Cooperative (128 weight rows): both work
        # on one tile, 64 rows each.  Ping-pong (64 rows): each owns every other tile of
        # the CTA's sequence; an ordered barrier serializes their mainloops (the tensor
        # cores run one tile at a time) and their epilogues, so one warpgroup's epilogue
        # overlaps the other's mainloop, as CUTLASS's KernelTmaWarpSpecializedPingpong.
        self.pingpong = tile_w == 64
        self.atom_layout_mnk = (1, 1, 1) if self.pingpong else (2, 1, 1)
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = 2
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        # Threads sharing one tile's epilogue.
        self.num_epi_threads = 128 if self.pingpong else 256
        self.load_warp_id = 0
        # Register budgets (the 64K register file): 128 x 40 + 256 x 232 = 64512.
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024

    # ------------------------------------------------------------- attributes
    def _setup_attributes(self):
        # The wgmma reads A (the converted weight) from registers and B from smem.
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.mma_dtype,
            self.mma_dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
            a_source=warpgroup.OperandSource.RMEM,
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        assert self.tile_shape_mnk[2] % mma_inst_shape_k == 0
        self.num_k_blocks = self.tile_shape_mnk[2] // mma_inst_shape_k

        tile_m, tile_a, tile_k = self.tile_shape_mnk
        # Epilogue subtile: all weight rows of the tile x 32 activation rows.
        self.epi_tile = (tile_m, 32)
        self.epi_stage = 2
        self.ab_stage = self._compute_stages()

        # Weight (fp8, K-major): 64 bytes per row per stage -> the 64-byte swizzle atom;
        # activation (16-bit, K-major): 128 bytes per row -> 128-byte swizzle.
        a_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.a_layout, self.a_dtype, tile_k),
            self.a_dtype,
        )
        self.a_smem_layout_staged = cute.tile_to_shape(
            a_atom, (tile_m, tile_k, self.ab_stage), order=(0, 1, 2)
        )
        b_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(self.b_layout, self.b_dtype, tile_k),
            self.b_dtype,
        )
        self.b_smem_layout_staged = cute.tile_to_shape(
            b_atom, (tile_a, tile_k, self.ab_stage), order=(0, 1, 2)
        )
        # C is M-major (weight rows contiguous).
        c_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.c_layout, self.c_dtype, self.epi_tile[0]
            ),
            self.c_dtype,
        )
        self.epi_smem_layout_staged = cute.tile_to_shape(
            c_atom, (*self.epi_tile, self.epi_stage), order=(1, 0, 2)
        )
        # The fp32 partial tiles of the split tail: same subtile, 32-bit elements; the
        # buffer is aliased with the C buffer (a tile is one or the other).
        self.w_dtype = cutlass.Float32
        w_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.c_layout, self.w_dtype, self.epi_tile[0]
            ),
            self.w_dtype,
        )
        self.w_smem_layout_staged = cute.tile_to_shape(
            w_atom, (*self.epi_tile, self.epi_stage), order=(1, 0, 2)
        )

    def _compute_stages(self) -> int:
        tile_m, tile_a, tile_k = self.tile_shape_mnk
        ab_bytes = (
            tile_m * tile_k * self.a_dtype.width + tile_a * tile_k * self.b_dtype.width
        ) // 8
        epi_width = max(self.c_dtype.width, 32 if self.has_tail else 0)
        epi_bytes = (
            self.epi_tile[0] * self.epi_tile[1] * epi_width // 8 * self.epi_stage
        )
        bias_bytes = 2 * 4 * tile_m
        mbar_bytes = 1024
        stages = (self.smem_capacity - epi_bytes - bias_bytes - mbar_bytes) // ab_bytes
        if stages < 2:
            raise ValueError(
                f"W8A16 sm90: tile {self.tile_shape_mnk} leaves fewer than 2 stages"
            )
        return stages

    # ------------------------------------------------------------------ entry
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        c: cute.Tensor,
        w: cute.Tensor,
        counters: cute.Tensor,
        act_rows: cutlass.Int32,
        body_items: cutlass.Int32,
        tail_splits: cutlass.Int32,
        num_persistent: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.mma_dtype = self.b_dtype
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup_attributes()

        tile_m, tile_a, tile_k = self.tile_shape_mnk
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a, a_smem_layout, (tile_m, tile_k)
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), b, b_smem_layout, (tile_a, tile_k)
        )
        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile
        )
        # The workspace store atom (fp32 partial tiles); without a tail it is built on
        # the (unused) workspace view all the same so the kernel signature is fixed.
        w_smem_layout = cute.slice_(self.w_smem_layout_staged, (None, None, 0))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), w, w_smem_layout, self.epi_tile
        )
        self.tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            # Ping-pong ordering: [phase (mainloop, epilogue)][consumer warpgroup].
            order_barriers: cute.struct.MemRange[cutlass.Int64, 4]
            # The bias of the tile's weight rows, fp32, double-buffered by tile parity.
            bias_tile: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 2 * tile_m], 16
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # Epilogue staging in C elements; with a split tail the fp32 partial
            # subtiles alias the same bytes (a tile is one or the other).
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    max(
                        cute.cosize(self.epi_smem_layout_staged),
                        (
                            cute.cosize(self.w_smem_layout_staged)
                            * 32
                            // self.c_dtype.width
                            if self.has_tail
                            else 0
                        ),
                    ),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_w,
            tma_tensor_w,
            w,
            c,
            counters,
            scale,
            bias,
            act_rows,
            body_items,
            tail_splits,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.w_smem_layout_staged,
        ).launch(
            grid=[num_persistent, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_mnl: cute.Tensor,
        mW_plain: cute.Tensor,
        mC_plain: cute.Tensor,
        counters: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        act_rows: cutlass.Int32,
        body_items: cutlass.Int32,
        tail_splits: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.ComposedLayout,
    ):
        tile_m, tile_a, tile_k = self.tile_shape_mnk
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
            if cutlass.const_expr(self.has_tail):
                cpasync.prefetch_descriptor(tma_atom_w)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # One pipeline for both operands: warp 0 of the producer warpgroup issues the two
        # TMA loads of a stage against one transaction barrier; every consumer warp
        # arrives on the empty barrier.
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        # Arrivals per stage: the warps of the warpgroup(s) consuming it (one warpgroup
        # per tile under ping-pong, both under cooperative).
        ab_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 4 if self.pingpong else 8
        )
        ab_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        order_bar = storage.order_barriers.data_ptr()
        if cutlass.const_expr(self.pingpong):
            if warp_idx == 0:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(4):
                        cute.arch.mbarrier_init(order_bar + i, 128)
        pipeline.pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sW = cute.make_tensor(
            cute.recast_ptr(
                storage.sC.data_ptr(), w_smem_layout_staged.inner, dtype=self.w_dtype
            ),
            w_smem_layout_staged.outer,
        )
        # Ping-pong, CTA with a single work item: group 1's accumulators cross to
        # group 0 through the (then idle) activation stages, (element, thread) with
        # consecutive threads 16 bytes apart.
        acc_per_thread = tile_m * tile_a // 128
        sRed = cute.make_tensor(
            cute.recast_ptr(storage.sB.data_ptr(), dtype=self.acc_dtype),
            cute.make_layout(
                ((4, acc_per_thread // 4), 128), stride=((1, 512), 4)
            ),
        )
        sBias = storage.bias_tile.get_tensor(
            cute.make_layout((tile_m, 2), stride=(1, tile_m))
        )

        # (bM, bK, RestM, RestK, L) / (bN, bK, RestN, RestK, L) / (bM, bN, RestM, RestN, L)
        gA_mkl = cute.local_tile(mA_mkl, (tile_m, tile_k), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, (tile_a, tile_k), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, (tile_m, tile_a), (None, None, None))
        gW_mnl = cute.local_tile(mW_mnl, (tile_m, tile_a), (None, None, None))
        # Fix-up views in 8-row vectors: the workspace (8, tile_m / 8, block columns),
        # the output (8, weight rows / 8, activation rows) and the bias (8, rows / 8).
        # The output column stride is spelled 8 * (rows / 8) so the compiler can see
        # that every 8-row vector is 16-byte aligned (as a plain dynamic row count it
        # emitted eight 2-byte stores).
        n_row_vecs = cute.size(mC_plain.shape[0]) // 8
        wv = cute.make_tensor(
            mW_plain.iterator,
            cute.make_layout(
                (8, tile_m // 8, cute.size(mW_plain.shape[1])), stride=(1, 8, tile_m)
            ),
        )
        cv = cute.make_tensor(
            mC_plain.iterator,
            cute.make_layout(
                (8, n_row_vecs, cute.size(mC_plain.shape[1])),
                stride=(1, 8, 8 * n_row_vecs),
            ),
        )
        bv = cute.make_tensor(
            bias.iterator, cute.make_layout((8, n_row_vecs), stride=(1, 8))
        )

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # Work items: body_items plain tiles (the grouped walk order of _tile_at, whole
        # K), then the tail tiles, each as tail_splits consecutive items (one K-slice
        # each, fp32 partial to the workspace).  Every warp computes the same sequence:
        # bidx, bidx + grid, ...
        weight_rows = cutlass.Int32(cute.size(mA_mkl.shape[0]))
        weight_tiles = cutlass.Int32(cute.ceil_div(weight_rows, tile_m))
        act_tiles = cutlass.Int32(cute.ceil_div(act_rows, tile_a))
        k_tiles_total = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
        total_tiles = weight_tiles * act_tiles
        total_work = body_items + (total_tiles - body_items) * tail_splits
        group = cutlass.min(cutlass.Int32(self.group_size), act_tiles)

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        pipeline.pipeline_init_wait(cluster_shape_mn=(1, 1))

        # ---- producer warpgroup: warp 0 loads both operands of every k-tile.
        if warp_group_idx < self.num_dma_warp_groups:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            if warp_idx == self.load_warp_id:
                producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                lin = cutlass.Int32(bidx)
                while lin < total_work:
                    tile_idx, m_tile, n_tile, k_base, k_tile_cnt, is_tail, _ = (
                        self._work_item(
                            lin,
                            body_items,
                            tail_splits,
                            k_tiles_total,
                            weight_tiles,
                            act_tiles,
                            group,
                        )
                    )
                    for k in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        ab_pipeline.producer_acquire(producer_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, m_tile, k_base + k, 0)],
                            tAsA[(None, producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                producer_state
                            ),
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, n_tile, k_base + k, 0)],
                            tBsB[(None, producer_state.index)],
                            tma_bar_ptr=ab_pipeline.producer_get_barrier(
                                producer_state
                            ),
                        )
                        ab_pipeline.producer_commit(producer_state)
                        producer_state.advance()
                    lin += gdx
                ab_pipeline.producer_tail(producer_state)

        # ---- consumer warpgroups: register-sourced wgmma mainloop, fused epilogue.
        if warp_group_idx >= self.num_dma_warp_groups:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            if cutlass.const_expr(self.pingpong):
                if warp_group_idx == self.num_dma_warp_groups:
                    self._consumer(
                        0,
                        cutlass.Int32(0),
                        tiled_mma,
                        ab_pipeline,
                        order_bar,
                        sA,
                        storage.sA.data_ptr(),
                        a_smem_layout_staged,
                        sB,
                        sC,
                        sW,
                        sRed,
                        sBias,
                        tma_atom_c,
                        gC_mnl,
                        tma_atom_w,
                        gW_mnl,
                        wv,
                        cv,
                        bv,
                        counters,
                        scale,
                        bias,
                        weight_rows,
                        act_rows,
                        weight_tiles,
                        act_tiles,
                        k_tiles_total,
                        body_items,
                        tail_splits,
                        total_work,
                        group,
                    )
                else:
                    self._consumer(
                        1,
                        cutlass.Int32(0),
                        tiled_mma,
                        ab_pipeline,
                        order_bar,
                        sA,
                        storage.sA.data_ptr(),
                        a_smem_layout_staged,
                        sB,
                        sC,
                        sW,
                        sRed,
                        sBias,
                        tma_atom_c,
                        gC_mnl,
                        tma_atom_w,
                        gW_mnl,
                        wv,
                        cv,
                        bv,
                        counters,
                        scale,
                        bias,
                        weight_rows,
                        act_rows,
                        weight_tiles,
                        act_tiles,
                        k_tiles_total,
                        body_items,
                        tail_splits,
                        total_work,
                        group,
                    )
            else:
                self._consumer(
                    0,
                    (warp_group_idx - self.num_dma_warp_groups)
                    * self.num_threads_per_warp_group,
                    tiled_mma,
                    ab_pipeline,
                    order_bar,
                    sA,
                    storage.sA.data_ptr(),
                    a_smem_layout_staged,
                    sB,
                    sC,
                    sW,
                    sRed,
                    sBias,
                    tma_atom_c,
                    gC_mnl,
                    tma_atom_w,
                    gW_mnl,
                    wv,
                    cv,
                    bv,
                    counters,
                    scale,
                    bias,
                    weight_rows,
                    act_rows,
                    weight_tiles,
                    act_tiles,
                    k_tiles_total,
                    body_items,
                    tail_splits,
                    total_work,
                    group,
                )

    # ------------------------------------------------------------- consumer
    @cute.jit
    def _consumer(
        self,
        cg: cutlass.Constexpr,
        mma_slice: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        ab_pipeline: pipeline.PipelineTmaAsync,
        order_bar: cute.Pointer,
        sA: cute.Tensor,
        sA_base: cute.Pointer,
        a_smem_layout_staged: cute.ComposedLayout,
        sB: cute.Tensor,
        sC: cute.Tensor,
        sW: cute.Tensor,
        sRed: cute.Tensor,
        sBias: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        gW_mnl: cute.Tensor,
        wv: cute.Tensor,
        cv: cute.Tensor,
        bv: cute.Tensor,
        counters: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        weight_rows: cutlass.Int32,
        act_rows: cutlass.Int32,
        weight_tiles: cutlass.Int32,
        act_tiles: cutlass.Int32,
        k_tiles_total: cutlass.Int32,
        body_items: cutlass.Int32,
        tail_splits: cutlass.Int32,
        total_work: cutlass.Int32,
        group: cutlass.Int32,
    ):
        """One consumer role: ``cg`` is the ping-pong consumer group (0 or 1; 0 under
        cooperative, where both warpgroups run this together), ``mma_slice`` the
        thread's slice of the tiled MMA."""
        tile_m, tile_a, tile_k = self.tile_shape_mnk
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Thread index within the tiled MMA / epilogue copies, and within its warpgroup.
        ctid = tidx % 128 if self.pingpong else tidx - 128
        wg_tid = tidx % 128
        epi_store_warp = 4 + 4 * cg if self.pingpong else 4
        epi_barrier = pipeline.NamedBarrier(
            barrier_id=1 + cg, num_threads=self.num_epi_threads
        )
        thr_mma = tiled_mma.get_slice(mma_slice)
        tCsA = thr_mma.partition_A(sA)  # (MMA, MMA_M, MMA_K, STAGE) smem
        tCsB = thr_mma.partition_B(sB)
        tCrB = thr_mma.make_fragment_B(tCsB)  # smem descriptors
        # A in registers: the converted (16-bit) fragment the wgmma reads, and the
        # fp8 fragment of the same shape the smem copy fills.
        tCrA_mma = thr_mma.make_fragment_A(tCsA[(None, None, None, 0)])
        num_k_blocks = cute.size(tCrA_mma, mode=[2])
        k_wait_max = min(num_k_blocks - 1, 7)
        # A fragment loads: the wgmma m64k16 A fragment of a thread is four 16-bit units
        # (two consecutive k of one row): rows r0 and r0 + 8, k offsets c0 and c0 + 8
        # within the k-block, r0 = warp * 16 + lane / 4 of its 64 rows, c0 = (lane % 4)
        # * 2.  Their byte offsets within the stage's K_SW64 layout (64-byte rows,
        # Swizzle<2, 4, 3>: bits [5:4] ^= bits [8:7]) are computed once per thread; the
        # loop adds the stage base and loads through load_convert_e4m3_units.
        assert tile_k == 64, "the hand-swizzled A addressing assumes 64-byte rows"
        lane = ctid % 32
        wg_rows0 = 0 if self.pingpong else (ctid // 128) * 64
        r0 = wg_rows0 + ((ctid % 128) // 32) * 16 + lane // 4
        c0 = (lane % 4) * 2
        a_unit_off = []
        for kb in cutlass.range_constexpr(num_k_blocks):
            for i in cutlass.range_constexpr(4):
                off = (r0 + 8 * (i & 1)) * tile_k + 16 * kb + c0 + 8 * (i >> 1)
                a_unit_off.append(off ^ (((off >> 7) & 3) << 4))
        a_smem_addr = sA_base.toint()
        a_stage_bytes = tile_m * tile_k

        # The MMA slice is per warpgroup (wgmma operands and accumulator layout are
        # warpgroup-uniform), so partition_C of the tile's identity gives each element
        # the warpgroup's row base (+ the fragment's +8 row component); the thread's
        # own rows of the m64 accumulator are warp * 16 + lane / 4 (+ 8).
        tile_coords = cute.make_identity_tensor((tile_m, tile_a))
        tCcAcc = thr_mma.partition_C(tile_coords)
        thread_row = ((ctid % 128) // 32) * 16 + (ctid % 32) // 4
        accumulators = cute.make_rmem_tensor(tCcAcc.shape, self.acc_dtype)

        # Epilogue copies, registers -> smem -> global (TMA).  The output path uses
        # stmatrix (16-bit C); the split tail's fp32 partials use plain 32-bit stores.
        # Subtile s of the retiled accumulator is its flat chunk [s * size, (s + 1) *
        # size) (n fast): the n8 blocks are the fragment's slowest mode.  For the
        # stmatrix copy the chunk's order is the smem fragment's order; the 32-bit copy's
        # fragment is (atom, rows 2, (col pair 2, n8 blocks)) while the accumulator
        # chunk runs col pair, row, n8 block, so the partials go through that
        # permutation.
        copy_atom_r2s = sm90_utils.get_smem_store_op(
            self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
        )
        copy_atom_c = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
            self.c_dtype,
        )
        tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_c_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(ctid)
        tRS_sD = thr_copy_r2s.partition_D(sC)  # (R2S, R2S_M, R2S_N, EPI_STAGE)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)  # (R2S, R2S_M, R2S_N) full tile
        tRS_cAcc = tiled_copy_r2s.retile(tCcAcc)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD = cute.make_rmem_tensor(rD_shape[:3], self.acc_dtype)
        tRS_rD_out = cute.make_rmem_tensor(rD_shape[:3], self.c_dtype)
        m_subs = tile_m // self.epi_tile[0]
        n_subs = tile_a // self.epi_tile[1]
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_epi_threads
            ),
        )

        read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        # Ping-pong: consumer group cg takes the CTA's work items cg, cg + 2, ...; the
        # producer fills the stages in item order, so this group starts cg tiles of
        # stages in and skips the other group's tile after each of its own.  Its order
        # barrier phase starts at 1 for group 0 (passes at once) and 0 for group 1.
        work_step = gdx * (2 if self.pingpong else 1)
        lin = cutlass.Int32(bidx + cg * gdx)
        order_phase = cutlass.Int32(1 - cg)
        # Ping-pong with fewer items than CTAs (this CTA has only item bidx): both
        # groups work on that item, group cg on the k-tiles of parity cg, each skipping
        # the other's stage after its own, and group 1 hands its accumulators to group
        # 0 through smem before the epilogue (k_split).  The groups then run their
        # mainloops concurrently, so the order barriers are bypassed.
        k_split = cutlass.Boolean(False)
        if cutlass.const_expr(self.pingpong and self.ksplit_enabled):
            if bidx < total_work and bidx + gdx >= total_work:
                _, _, _, _, k_cnt0, _, _ = self._work_item(
                    cutlass.Int32(bidx),
                    body_items,
                    tail_splits,
                    k_tiles_total,
                    weight_tiles,
                    act_tiles,
                    group,
                )
                k_split = k_cnt0 >= 2
        reduce_barrier = pipeline.NamedBarrier(barrier_id=3, num_threads=256)
        if cutlass.const_expr(self.pingpong and cg == 1):
            if k_split:
                lin = cutlass.Int32(bidx)
                read_state.advance()
                release_state.advance()
            else:
                # Skip the stages of the other group's first item (work item bidx).
                _, _, _, _, k_skip0, _, _ = self._work_item(
                    cutlass.Int32(bidx),
                    body_items,
                    tail_splits,
                    k_tiles_total,
                    weight_tiles,
                    act_tiles,
                    group,
                )
                for _ in cutlass.range(0, k_skip0, 1, unroll=1):
                    read_state.advance()
                    release_state.advance()
        tiles_done = cutlass.Int32(0)
        k_skip = cutlass.Int32(0)  # loop-carried: a stable type on every path
        while lin < total_work:
            tile_idx, m_tile, n_tile, k_base, k_tile_cnt, is_tail, slice_idx = (
                self._work_item(
                    lin,
                    body_items,
                    tail_splits,
                    k_tiles_total,
                    weight_tiles,
                    act_tiles,
                    group,
                )
            )
            if cutlass.const_expr(self.pingpong):
                if not k_split:
                    cute.arch.mbarrier_wait(order_bar + cg, order_phase)
            # This group's k-tiles of the item (every other one under k_split).
            k_tiles_own = k_tile_cnt
            if k_split:
                k_tiles_own = (k_tile_cnt + 1 - cg) // 2

            # ---- mainloop
            accumulators.fill(0.0)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            read_state.reset_count()
            release_state.reset_count()

            # First k-tile: no wgmma in flight, so no wait before the converts.
            ab_pipeline.consumer_wait(read_state)
            # Loads run two k-blocks ahead of the wgmma, converts one ahead, as the
            # CUTLASS mainloop: the smem latency and the ALU work hide behind the
            # tensor core.
            stage_addr = a_smem_addr + read_state.index * a_stage_bytes
            a_words = {}
            a_words[0] = self._a_loads(0, stage_addr, a_unit_off)
            if cutlass.const_expr(num_k_blocks > 1):
                a_words[1] = self._a_loads(1, stage_addr, a_unit_off)
            self._a_convert(tCrA_mma, 0, a_words[0])
            warpgroup.fence()
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_mma[(None, None, kb)],
                    tCrB[(None, None, kb, read_state.index)],
                    accumulators,
                )
                warpgroup.commit_group()
                if cutlass.const_expr(kb + 2 < num_k_blocks):
                    a_words[kb + 2] = self._a_loads(kb + 2, stage_addr, a_unit_off)
                if cutlass.const_expr(kb + 1 < num_k_blocks):
                    self._a_convert(tCrA_mma, kb + 1, a_words[kb + 1])
            warpgroup.fence()
            read_state.advance()
            if k_split:
                read_state.advance()

            # Steady state: the wgmma of k-block kb issued, at most k_wait_max groups
            # pending, then the previous stage is released at the last k-block (all
            # its wgmma are done) and the registers of k-block kb + 1 are refilled
            # (their reader, k-block kb + 1 of the previous k-tile, is done).
            for _k_tile in cutlass.range(1, k_tiles_own, 1, unroll=1):
                ab_pipeline.consumer_wait(read_state)
                stage_addr_k = a_smem_addr + read_state.index * a_stage_bytes
                a_words_k = {}
                a_words_k[0] = self._a_loads(0, stage_addr_k, a_unit_off)
                if cutlass.const_expr(num_k_blocks > 1):
                    a_words_k[1] = self._a_loads(1, stage_addr_k, a_unit_off)
                warpgroup.wait_group(k_wait_max)
                self._a_convert(tCrA_mma, 0, a_words_k[0])
                warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        tCrA_mma[(None, None, kb)],
                        tCrB[(None, None, kb, read_state.index)],
                        accumulators,
                    )
                    warpgroup.commit_group()
                    warpgroup.wait_group(k_wait_max)
                    if cutlass.const_expr(kb == num_k_blocks - 1):
                        ab_pipeline.consumer_release(release_state)
                        release_state.advance()
                        if k_split:
                            release_state.advance()
                    if cutlass.const_expr(kb + 2 < num_k_blocks):
                        a_words_k[kb + 2] = self._a_loads(
                            kb + 2, stage_addr_k, a_unit_off
                        )
                    if cutlass.const_expr(kb + 1 < num_k_blocks):
                        self._a_convert(tCrA_mma, kb + 1, a_words_k[kb + 1])
                warpgroup.fence()
                read_state.advance()
                if k_split:
                    read_state.advance()
            if cutlass.const_expr(self.pingpong):
                if not k_split:
                    # The other group's mainloop may start; this one drains behind it.
                    cute.arch.mbarrier_arrive(order_bar + (1 - cg))
            warpgroup.wait_group(0)
            ab_pipeline.consumer_release(release_state)
            release_state.advance()
            # Under ping-pong only one group runs the epilogue of a k_split item.
            runs_epilogue = cutlass.Boolean(True)
            if cutlass.const_expr(self.pingpong):
                if k_split:
                    # Both groups are past their last wgmma after the barrier, and the
                    # producer has nothing more to load, so the stages are free.
                    reduce_barrier.arrive_and_wait()
                    if cutlass.const_expr(cg == 1):
                        for i in cutlass.range_constexpr(cute.size(accumulators)):
                            sRed[(i, ctid)] = accumulators[i]
                        runs_epilogue = cutlass.Boolean(False)
                    reduce_barrier.arrive_and_wait()
                    if cutlass.const_expr(cg == 0):
                        for i in cutlass.range_constexpr(cute.size(accumulators)):
                            accumulators[i] = accumulators[i] + sRed[(i, ctid)]
                else:
                    # Skip the stages of the other group's next item (if it exists).
                    lin_other = lin + gdx
                    k_skip = cutlass.Int32(0)
                    if lin_other < total_work:
                        _, _, _, _, k_cnt_other, _, _ = self._work_item(
                            lin_other,
                            body_items,
                            tail_splits,
                            k_tiles_total,
                            weight_tiles,
                            act_tiles,
                            group,
                        )
                        k_skip = k_cnt_other
                    for _ in cutlass.range(0, k_skip, 1, unroll=1):
                        read_state.advance()
                        release_state.advance()
                    cute.arch.mbarrier_wait(order_bar + 2 + cg, order_phase)

            # ---- epilogue.  Plain tile: D = scale * acc (+ bias[row]), cast, TMA store
            # into the output per (tile_m x 32) subtile.  Tail item: the fp32
            # accumulator into its workspace block.
            if runs_epilogue:
                self._epilogue(
                    cg,
                    lin,
                    tile_idx,
                    m_tile,
                    n_tile,
                    is_tail,
                    slice_idx,
                    tiles_done,
                    tidx,
                    warp_idx,
                    ctid,
                    thread_row,
                    epi_store_warp,
                    epi_barrier,
                    accumulators,
                    tiled_copy_r2s,
                    tma_atom_c,
                    tma_store_pipeline,
                    sC,
                    tRS_sD,
                    tRS_rAcc,
                    tRS_cAcc,
                    tRS_rD,
                    tRS_rD_out,
                    sW,
                    sBias,
                    tma_atom_w,
                    gC_mnl,
                    gW_mnl,
                    wv,
                    cv,
                    bv,
                    counters,
                    scale,
                    bias,
                    weight_rows,
                    act_rows,
                    body_items,
                    tail_splits,
                )
            if cutlass.const_expr(self.pingpong):
                if not k_split:
                    # The other group reuses the epilogue smem next: every TMA store
                    # of this tile must have read it (the store wait is per issuing
                    # thread).
                    tma_store_pipeline.producer_tail()
                    epi_barrier.arrive_and_wait()
                    cute.arch.mbarrier_arrive(order_bar + 2 + (1 - cg))
                    order_phase ^= 1
            tiles_done += 1
            lin += work_step
        tma_store_pipeline.producer_tail()

    @cute.jit
    def _epilogue(
        self,
        cg: cutlass.Constexpr,
        lin: cutlass.Int32,
        tile_idx: cutlass.Int32,
        m_tile: cutlass.Int32,
        n_tile: cutlass.Int32,
        is_tail: cutlass.Boolean,
        slice_idx: cutlass.Int32,
        tiles_done: cutlass.Int32,
        tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        ctid: cutlass.Int32,
        thread_row: cutlass.Int32,
        epi_store_warp: cutlass.Constexpr,
        epi_barrier: pipeline.NamedBarrier,
        accumulators: cute.Tensor,
        tiled_copy_r2s: cute.TiledCopy,
        tma_atom_c: cute.CopyAtom,
        tma_store_pipeline: pipeline.PipelineTmaStore,
        sC: cute.Tensor,
        tRS_sD: cute.Tensor,
        tRS_rAcc: cute.Tensor,
        tRS_cAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        tRS_rD_out: cute.Tensor,
        sW: cute.Tensor,
        sBias: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        gC_mnl: cute.Tensor,
        gW_mnl: cute.Tensor,
        wv: cute.Tensor,
        cv: cute.Tensor,
        bv: cute.Tensor,
        counters: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        weight_rows: cutlass.Int32,
        act_rows: cutlass.Int32,
        body_items: cutlass.Int32,
        tail_splits: cutlass.Int32,
    ):
        """The epilogue of one work item: the output tile (scale, bias, cast, TMA
        store) or, for a tail item, its fp32 partial block and the split fix-up."""
        tile_m, tile_a, _ = self.tile_shape_mnk
        wg_tid = tidx % 128
        m_subs = tile_m // self.epi_tile[0]
        n_subs = tile_a // self.epi_tile[1]
        if cutlass.const_expr(self.has_tail):
            copy_atom_w = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.w_dtype, num_bits_per_copy=32
            )
            tiled_copy_w_atom = cute.make_tiled_copy_C_atom(copy_atom_w, self.tiled_mma)
            tiled_copy_r2s_w = cute.make_tiled_copy_S(copy_atom_w, tiled_copy_w_atom)
            thr_copy_r2s_w = tiled_copy_r2s_w.get_slice(ctid)
            tRS_sW = thr_copy_r2s_w.partition_D(sW)
            tRS_rAcc_w = tiled_copy_r2s_w.retile(accumulators)
            rW_shape = cute.shape(thr_copy_r2s_w.partition_S(sW))
            tRS_rW = cute.make_rmem_tensor(rW_shape[:3], self.w_dtype)
            size_tRS_rW = cute.size(tRS_rW)
            assert cute.size(rW_shape[0]) == 1 and cute.size(rW_shape[1]) == 2, rW_shape
            assert cute.size(rW_shape[2]) * 2 == size_tRS_rW, rW_shape
            n8_blocks = cute.size(rW_shape[2]) // 2
        scale_val = cutlass.Float32(scale[0])
        # Bias staging buffer: this group's own under ping-pong (the epilogues of
        # the two groups are serialized), tile parity under cooperative.
        bias_buf = cutlass.Int32(cg) if self.pingpong else tiles_done % 2
        if cutlass.const_expr(self.has_bias):
            if not is_tail:
                if wg_tid < tile_m:
                    bias_row = m_tile * tile_m + wg_tid
                    bias_val = cutlass.Float32(0.0)
                    if bias_row < weight_rows:
                        bias_val = cutlass.Float32(bias[bias_row])
                    sBias[(wg_tid, bias_buf)] = bias_val
                epi_barrier.arrive_and_wait()

        num_prev_subtiles = tiles_done * (m_subs * n_subs)
        # One (tile, slice) workspace block per tail item.  The output epilogue is
        # a helper so the tail branch exists only when the tail is compiled in.
        tail_block = lin - body_items
        if cutlass.const_expr(self.has_tail):
            if is_tail:
                gW_tile = gW_mnl[(None, None, 0, tail_block, 0)]
                tCgW_epi = cute.zipped_divide(gW_tile, self.epi_tile)
                bSG_sW, bSG_gW = cpasync.tma_partition(
                    tma_atom_w,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sW, 0, 2),
                    tCgW_epi,
                )
                for m_sub in cutlass.range_constexpr(0 if self.fixup_cut == 4 else m_subs):
                    for n_sub in cutlass.range_constexpr(n_subs):
                        base = (m_sub * n_subs + n_sub) * size_tRS_rW
                        for b in cutlass.range_constexpr(n8_blocks):
                            for r in cutlass.range_constexpr(2):
                                for c in cutlass.range_constexpr(2):
                                    tRS_rW[(0, r, c + 2 * b)] = tRS_rAcc_w[
                                        base + c + 2 * r + 4 * b
                                    ]
                        epi_idx = m_sub * n_subs + n_sub
                        epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
                        cute.copy(
                            tiled_copy_r2s_w,
                            tRS_rW,
                            tRS_sW[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        # One barrier per subtile: the storing warp first waits
                        # for the previous TMA store to finish reading its buffer
                        # (the one the next subtile writes), so after the barrier
                        # everyone may write it and the store of this subtile can
                        # be issued.
                        if warp_idx == epi_store_warp:
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        epi_barrier.arrive_and_wait()
                        if warp_idx == epi_store_warp:
                            cute.copy(
                                tma_atom_w,
                                bSG_sW[(None, epi_buffer)],
                                bSG_gW[(None, (m_sub, n_sub))],
                            )
                            tma_store_pipeline.producer_commit()
                self._tail_fixup(
                    1 + cg,
                    tile_idx - body_items,
                    slice_idx,
                    tail_splits,
                    m_tile,
                    n_tile,
                    act_rows,
                    weight_rows,
                    wv,
                    cv,
                    bv,
                    counters,
                    scale,
                    warp_idx,
                    epi_store_warp,
                    ctid,
                )
            else:
                self._epilogue_output(
                    1 + cg,
                    tiled_copy_r2s,
                    tma_atom_c,
                    tma_store_pipeline,
                    sC,
                    tRS_sD,
                    tRS_rAcc,
                    tRS_cAcc,
                    tRS_rD,
                    tRS_rD_out,
                    sBias,
                    gC_mnl,
                    m_tile,
                    n_tile,
                    scale_val,
                    bias_buf,
                    thread_row,
                    num_prev_subtiles,
                    warp_idx,
                    epi_store_warp,
                )
        else:
            self._epilogue_output(
                1 + cg,
                tiled_copy_r2s,
                tma_atom_c,
                tma_store_pipeline,
                sC,
                tRS_sD,
                tRS_rAcc,
                tRS_cAcc,
                tRS_rD,
                tRS_rD_out,
                sBias,
                gC_mnl,
                m_tile,
                n_tile,
                scale_val,
                bias_buf,
                thread_row,
                num_prev_subtiles,
                warp_idx,
                epi_store_warp,
            )

    # ---------------------------------------------------------- tail fix-up
    @cute.jit
    def _tail_fixup(
        self,
        barrier_id: cutlass.Constexpr,
        tail_tile: cutlass.Int32,
        slice_idx: cutlass.Int32,
        tail_splits: cutlass.Int32,
        m_tile: cutlass.Int32,
        n_tile: cutlass.Int32,
        act_rows: cutlass.Int32,
        weight_rows: cutlass.Int32,
        wv: cute.Tensor,
        cv: cute.Tensor,
        bv: cute.Tensor,
        counters: cute.Tensor,
        scale: cute.Tensor,
        warp_idx: cutlass.Int32,
        epi_store_warp: cutlass.Constexpr,
        etid: cutlass.Int32,
    ):
        """After this CTA's partial block of tail tile ``tail_tile`` is stored: publish
        it (bulk stores complete, release), count in on the tile's counter, wait for
        the tile's ``tail_splits`` slices, then reduce this slice's share of the tile's
        activation columns over all blocks, apply scale and bias and store the 16-bit
        output.  The counter is monotonic (see ``TAIL_COUNTER_STRIDE``): the add's
        return value tells the last slice it is last, so only the others poll."""
        tile_m, tile_a, _ = self.tile_shape_mnk
        epi_barrier = pipeline.NamedBarrier(
            barrier_id=barrier_id, num_threads=self.num_epi_threads
        )
        counter_addr = counters.iterator.toint() + tail_tile * 8
        if cutlass.const_expr(self.fixup_cut == 1 or self.fixup_cut == 4):
            return
        # ``etid``: the thread's index within the tile's epilogue group (both consumer
        # warpgroups under cooperative, its own under ping-pong).
        if warp_idx == epi_store_warp:
            bulk_store_complete_fence()
        epi_barrier.arrive_and_wait()
        if etid == 0:
            step = cutlass.Int64(TAIL_COUNTER_STRIDE // tail_splits)
            old = atomic_add_acq_rel_gpu_u64(counter_addr, step)
            target = (old // TAIL_COUNTER_STRIDE + 1) * TAIL_COUNTER_STRIDE
            seen = old + step
            while seen < target:
                nanosleep(64)
                seen = ld_acquire_gpu_u64(counter_addr)
        epi_barrier.arrive_and_wait()
        if cutlass.const_expr(self.fixup_cut == 2):
            return

        # This slice's columns: [slice * q + min(slice, rem), ...), q = tile_a / splits.
        q = tile_a // tail_splits
        rem = tile_a % tail_splits
        col_begin = slice_idx * q + cutlass.min(slice_idx, rem)
        col_cnt = q
        if slice_idx < rem:
            col_cnt = q + 1
        vecs_per_col = tile_m // 8
        block0 = tail_tile * tail_splits
        scale_val = cutlass.Float32(scale[0])
        act_base = n_tile * tile_a
        w_base = m_tile * tile_m
        items = col_cnt * vecs_per_col
        for it in cutlass.range(etid, items, self.num_epi_threads, unroll=1):
            col = col_begin + it // vecs_per_col
            rvec = it % vecs_per_col
            act_row = act_base + col
            w_row = w_base + rvec * 8
            if act_row < act_rows:
                if w_row < weight_rows:
                    # Unrolled so the slices' loads are all in flight before the first
                    # add (a thread usually owns one vector; serial loads would cost a
                    # global-memory latency per slice).
                    acc = wv[(None, rvec, block0 * tile_a + col)].load()
                    for sl in cutlass.range(1, tail_splits, 1, unroll=8):
                        acc = (
                            acc + wv[(None, rvec, (block0 + sl) * tile_a + col)].load()
                        )
                    acc = scale_val * acc
                    if cutlass.const_expr(self.has_bias):
                        acc = acc + bv[(None, w_row // 8)].load().to(cutlass.Float32)
                    if cutlass.const_expr(self.fixup_cut == 3):
                        if acc[0] == 12345.0:
                            cv[(None, w_row // 8, act_row)].store(acc.to(cv.element_type))
                    else:
                        cv[(None, w_row // 8, act_row)].store(acc.to(cv.element_type))

    # ------------------------------------------------------- output epilogue
    @cute.jit
    def _epilogue_output(
        self,
        barrier_id: cutlass.Constexpr,
        tiled_copy_r2s: cute.TiledCopy,
        tma_atom_c: cute.CopyAtom,
        tma_store_pipeline: pipeline.PipelineTmaStore,
        sC: cute.Tensor,
        tRS_sD: cute.Tensor,
        tRS_rAcc: cute.Tensor,
        tRS_cAcc: cute.Tensor,
        tRS_rD: cute.Tensor,
        tRS_rD_out: cute.Tensor,
        sBias: cute.Tensor,
        gC_mnl: cute.Tensor,
        m_tile: cutlass.Int32,
        n_tile: cutlass.Int32,
        scale_val: cutlass.Float32,
        bias_buf: cutlass.Int32,
        thread_row: cutlass.Int32,
        num_prev_subtiles: cutlass.Int32,
        warp_idx: cutlass.Int32,
        epi_store_warp: cutlass.Constexpr,
    ):
        """D = scale * acc (+ bias[row]), cast, TMA store into the output per
        (tile_m x 32) subtile."""
        tile_m, tile_a, _ = self.tile_shape_mnk
        m_subs = tile_m // self.epi_tile[0]
        n_subs = tile_a // self.epi_tile[1]
        size_tRS_rD = cute.size(tRS_rD)
        epi_barrier = pipeline.NamedBarrier(
            barrier_id=barrier_id, num_threads=self.num_epi_threads
        )
        gC_tile = gC_mnl[(None, None, m_tile, n_tile, 0)]
        tCgC_epi = cute.zipped_divide(gC_tile, self.epi_tile)
        bSG_sD, bSG_gD = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            tCgC_epi,
        )
        for m_sub in cutlass.range_constexpr(m_subs):
            for n_sub in cutlass.range_constexpr(n_subs):
                base = (m_sub * n_subs + n_sub) * size_tRS_rD
                if cutlass.const_expr(self.has_bias):
                    for v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[v] = (
                            scale_val * tRS_rAcc[base + v]
                            + sBias[(tRS_cAcc[base + v][0] + thread_row, bias_buf)]
                        )
                else:
                    for v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[v] = scale_val * tRS_rAcc[base + v]
                tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                epi_idx = m_sub * n_subs + n_sub
                epi_buffer = (num_prev_subtiles + epi_idx) % self.epi_stage
                cute.copy(
                    tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                # One barrier per subtile (see the tail path): the storing warp waits
                # for the previous store's smem read before the barrier.
                if warp_idx == epi_store_warp:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                epi_barrier.arrive_and_wait()
                if warp_idx == epi_store_warp:
                    cute.copy(
                        tma_atom_c,
                        bSG_sD[(None, epi_buffer)],
                        bSG_gD[(None, (m_sub, n_sub))],
                    )
                    tma_store_pipeline.producer_commit()

    # --------------------------------------------------------------- helpers
    @cute.jit
    def _work_item(
        self,
        lin: cutlass.Int32,
        body_items: cutlass.Int32,
        tail_splits: cutlass.Int32,
        k_tiles_total: cutlass.Int32,
        weight_tiles: cutlass.Int32,
        act_tiles: cutlass.Int32,
        group: cutlass.Int32,
    ):
        """Work item ``lin`` -> (tile index, weight tile, activation tile, first k-tile,
        k-tile count, is_tail).  Items below ``body_items`` are whole tiles; the tail
        tiles follow as ``tail_splits`` consecutive items each (near-equal K-slices,
        the first ``k_tiles % splits`` slices one k-tile longer)."""
        tile_idx = lin
        slice_idx = cutlass.Int32(0)
        is_tail = lin >= body_items
        if is_tail:
            tile_idx = body_items + (lin - body_items) // tail_splits
            slice_idx = (lin - body_items) % tail_splits
        splits = cutlass.Int32(1)
        if is_tail:
            splits = tail_splits
        q = k_tiles_total // splits
        rem = k_tiles_total % splits
        k_base = slice_idx * q + cutlass.min(slice_idx, rem)
        k_cnt = q
        if slice_idx < rem:
            k_cnt = q + 1
        m_tile, n_tile = grouped_tile_at(tile_idx, weight_tiles, act_tiles, group)
        return tile_idx, m_tile, n_tile, k_base, k_cnt, is_tail, slice_idx

    # Plain (trace-time) methods: they emit ops inline at the call site.
    def _a_loads(self, kb: int, stage_addr, a_unit_off: list) -> list:
        """The four 16-bit units of k-block ``kb`` of the thread's A fragment, loaded
        from the stage at shared address ``stage_addr`` into 32-bit registers."""
        return [load_e4m3_unit(stage_addr + a_unit_off[4 * kb + i]) for i in range(4)]

    def _a_convert(self, tCrA_mma: cute.Tensor, kb: int, words: list) -> None:
        tCrA_mma[(None, None, kb)].store(
            convert_e4m3_words(words, self.mma_dtype, native_cvt=False)
        )

    # ----------------------------------------------------------------- launch
    @cute.jit
    def wrapper(
        self,
        weight_ptr: cute.Pointer,
        activation_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        workspace_ptr: cute.Pointer,
        counters_ptr: cute.Pointer,
        n: cutlass.Int64,
        m: cutlass.Int64,
        k: cutlass.Int64,
        tail_blocks: cutlass.Int64,
        body_items: cutlass.Int32,
        tail_splits: cutlass.Int32,
        num_persistent: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Weight [n, k] fp8 and activation [m, k] 16-bit, K-major; out [m, n] row-major
        is C[n, m, 1].  The workspace holds ``tail_blocks`` fp32 tiles of
        (tile_m, tile_a), one per tail work item, as W[tile_m, tail_blocks * tile_a, 1]
        (weight rows contiguous, as C); ``counters`` are the tail tiles' 64-bit
        monotonic arrival counters (``TAIL_COUNTER_STRIDE``), one per tail tile."""
        tile_m, tile_a, _ = self.tile_shape_mnk
        weight = cute.make_tensor(
            weight_ptr, cute.make_ordered_layout((n, k, 1), order=(1, 0, 2))
        )
        activation = cute.make_tensor(
            activation_ptr, cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        output = cute.make_tensor(
            output_ptr, cute.make_ordered_layout((n, m, 1), order=(0, 1, 2))
        )
        workspace = cute.make_tensor(
            workspace_ptr,
            cute.make_ordered_layout(
                (tile_m, tail_blocks * tile_a, 1), order=(0, 1, 2)
            ),
        )
        counters = cute.make_tensor(counters_ptr, cute.make_layout((tail_blocks,)))
        scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        bias = cute.make_tensor(bias_ptr, cute.make_layout((n,)))
        return self(
            weight,
            activation,
            scale,
            bias,
            output,
            workspace,
            counters,
            cutlass.Int32(m),
            body_items,
            tail_splits,
            num_persistent=num_persistent,
            stream=stream,
        )


__all__ = ["Sm90W8A16Fp8GemmKernel", "grouped_tile_at"]
