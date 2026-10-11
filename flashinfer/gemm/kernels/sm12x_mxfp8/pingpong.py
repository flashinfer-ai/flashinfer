# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Persistent ping-pong MXFP8 GEMM for SM120 / SM121 (large M).

out[M, N] = A[M, K] @ B[N, K]^T, E4M3 operands with UE8M0 scales per 32
elements in the F8_128x4 layout produced by
``mxfp8_quantize(..., is_sf_swizzled_layout=True)``; nothing is repacked.

The structure follows the CUTLASS CuTe DSL example
``examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/
dense_blockscaled_gemm_persistent_pingpong.py``, which provides the SM120
block-scaled warp MMA setup, the scale-factor shared-memory layouts and the
ping-pong warp-group ordering. Changes:

- runtime M (one compile per N, K and configuration), static N and K;
- grouped tile rasterization with an exact tail decode, sized so the live A
  panels stay resident in L2;
- optional K64 stages sharing one K128 scale chunk, doubling the pipeline
  depth inside the 99 KB shared-memory budget;
- a 128x64 N tile that reads half of a 128-row scale block;
- an optional cooperative 256x128 tile (both warp groups share a B tile);
- a coalesced predicated epilogue for output rows that are not 16-byte
  aligned (TMA store otherwise);
- programmatic dependent launch: the first weight stage is loaded before
  ``griddepcontrol.wait``; activations, later stages and outputs after it.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync

from .common import SF_CHUNK, SMEM_BYTES

SF_VEC = 32


class Sm12xMxfp8PingpongGemm:
    """Host wrapper; ``__call__(a, b, sfa, sfb, c, num_ctas)``.

    Args:
        n, k: problem size (compile-time).
        tile_n: 128 or 64. tile_k: 128 or 64.
        epi_tile: epilogue sub-tile (rows, cols); epi_stages: TMA store stages.
        group: M tiles per rasterization group (A panels resident in L2).
        coop: cooperative 256x128 CTA tile (requires tile_n 128, tile_k 64).
        early_mma: issue the last k-block MMAs of a stage before waiting on
            the next stage.
    """

    def __init__(
        self,
        n,
        k,
        tile_n=128,
        tile_k=128,
        epi_tile=(64, 32),
        epi_stages=None,
        group=8,
        coop=False,
        early_mma=False,
        out_dtype=cutlass.BFloat16,
    ):
        assert tile_k in (64, 128) and tile_n in (64, 128)
        assert not coop or (tile_n == 128 and tile_k == 64)
        self.n = n
        self.k = k
        self.tile_k = tile_k
        self.tile_n = tile_n
        self.epi_tile = tuple(epi_tile)
        self.epi_stages_cfg = epi_stages
        self.group = group
        self.coop = coop
        self.early_mma = early_mma
        # Output rows that are not 16-byte aligned cannot use the TMA store.
        self.direct_epilogue = (n * 2) % 16 != 0
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = cutlass.Float8E4M3FN
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.c_dtype = out_dtype
        # tile_shape_mnk is the per-warp-group MMA tile; cta_m the CTA's M extent.
        self.tile_shape_mnk = (128, tile_n, tile_k)
        self.cta_m = 256 if coop else 128
        self.sf_shared = tile_k == 64
        self.num_mma_warps = 8
        self.tma_load_warp_id = self.num_mma_warps
        self.threads_per_cta = ((self.num_mma_warps + 1) * 32 + 127) // 128 * 128
        self.buffer_align_bytes = 1024
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        self.epilog_sync_barrier_wg1 = pipeline.NamedBarrier(
            barrier_id=3, num_threads=128
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.num_n_tiles = (n + tile_n - 1) // tile_n
        self.k_tile_cnt = (k + tile_k - 1) // tile_k

    # ------------------------------------------------------------------ setup
    def _setup_attributes(self):
        mma_op = cute.nvgpu.warp.MmaMXF8Op(self.ab_dtype, self.acc_dtype, self.sf_dtype)
        perm = sm120_utils.get_permutation_mnk(self.tile_shape_mnk, SF_VEC, True)
        self.tiled_mma = cute.make_tiled_mma(
            mma_op, cute.make_layout((2, 2, 1)), permutation_mnk=perm
        )
        sfa_tile = (self.cta_m, 128, 128)
        sf_tile = (128, 128, 128)
        sfa_1 = blockscaled_utils.sm120_make_smem_layout_sfa(
            self.tiled_mma, sfa_tile, SF_VEC, 1
        )
        sfb_1 = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma, sf_tile, SF_VEC, 1
        )
        tm, tn, tk = self.tile_shape_mnk
        ab_bytes = self.cta_m * tk + tn * tk
        sf_bytes = cute.size(cute.filter_zeros(sfa_1).shape) + cute.size(
            cute.filter_zeros(sfb_1).shape
        )
        if self.direct_epilogue:
            epi_stage = 2
        else:
            epi_max = (tn // self.epi_tile[1]) * (tm // self.epi_tile[0])
            epi_stage = min(epi_max, self.epi_stages_cfg or 4)
        epi_bytes = self.epi_tile[0] * self.epi_tile[1] * 2 * epi_stage
        if self.coop:
            epi_bytes *= 2  # one set of epilogue buffers per warp group
        ab_stage = (SMEM_BYTES - 1024 - 1024 - epi_bytes) // (ab_bytes + sf_bytes)
        assert ab_stage >= 2
        self.ab_stage = ab_stage
        self.epi_stage = epi_stage

        a_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR, self.ab_dtype, tk
            ),
            self.ab_dtype,
        )
        self.a_smem_layout_staged = cute.tile_to_shape(
            a_atom, (self.cta_m, tk, ab_stage), order=(0, 1, 2)
        )
        self.b_smem_layout_staged = cute.tile_to_shape(
            a_atom, (tn, tk, ab_stage), order=(0, 1, 2)
        )
        self.sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
            self.tiled_mma, sfa_tile, SF_VEC, ab_stage
        )
        # Per-warp-group view of a 256-row SFA block: one 128-row sub-block
        # (byte offset SF_CHUNK * warp group), same strides.
        self.sfa_mma_layout_staged = self.sfa_smem_layout_staged
        if self.coop:

            def one_block(x):
                if isinstance(x, tuple):
                    if len(x) == 2 and x[0] == (32, 4) and x[1] == 2:
                        return ((32, 4), 1)
                    return tuple(one_block(v) for v in x)
                return x

            lay = self.sfa_smem_layout_staged
            self.sfa_mma_layout_staged = cute.make_layout(
                one_block(lay.shape), stride=lay.stride
            )
        self.sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            self.tiled_mma, sf_tile, SF_VEC, ab_stage
        )
        # The SFB TMA always fills a 128-row scale block. A 64-wide N tile reads
        # one half of it: rows 32 * j + i for j in {2h, 2h + 1} sit at byte
        # offset 8 * h with the 4-row interleave narrowed from 4 to 2.
        self.sfb_mma_layout_staged = self.sfb_smem_layout_staged
        if tn == 64:

            def narrow(x):
                if isinstance(x, tuple):
                    if x == (32, 4):
                        return (32, 2)
                    return tuple(narrow(v) for v in x)
                return x

            lay = self.sfb_smem_layout_staged
            self.sfb_mma_layout_staged = cute.make_layout(
                narrow(lay.shape), stride=lay.stride
            )
        c_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR, self.c_dtype, self.epi_tile[1]
            ),
            self.c_dtype,
        )
        self.epi_smem_layout_staged = cute.tile_to_shape(
            c_atom, (*self.epi_tile, self.epi_stage), order=(0, 1, 2)
        )
        self.epi_cosize = cute.cosize(self.epi_smem_layout_staged)

    # --------------------------------------------------------------- host jit
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        num_ctas: Int32,
        stream,
    ):
        """a: [M, K] E4M3, b: [N, K] E4M3, sfa / sfb: flat UE8M0 bytes, c: [M, N]."""
        self._setup_attributes()
        m = cute.size(a, mode=[0])
        n, k = self.n, self.k
        a3 = cute.make_tensor(
            a.iterator, cute.make_layout((m, k, 1), stride=(k, 1, m * k))
        )
        b3 = cute.make_tensor(
            b.iterator, cute.make_layout((n, k, 1), stride=(k, 1, n * k))
        )
        c3 = cute.make_tensor(
            c.iterator, cute.make_layout((m, n, 1), stride=(n, 1, m * n))
        )
        c3_tma = c3
        if cutlass.const_expr(self.direct_epilogue):
            # The TMA store is unused here, but its descriptor is still encoded
            # and needs a 16-byte row pitch.
            n_pad = (n + 7) // 8 * 8
            c3_tma = cute.make_tensor(
                c.iterator,
                cute.make_layout((m, n_pad, 1), stride=(n_pad, 1, m * n_pad)),
            )
        sfa_ptr = cute.recast_ptr(sfa.iterator, dtype=self.sf_dtype)
        sfb_ptr = cute.recast_ptr(sfb.iterator, dtype=self.sf_dtype)
        sfa3 = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a3.shape, SF_VEC)
        )
        sfb3 = cute.make_tensor(
            sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b3.shape, SF_VEC)
        )

        _, tn, tk = self.tile_shape_mnk
        a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a3, a_smem, (self.cta_m, tk)
        )
        tma_atom_b, tma_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), b3, b_smem, (tn, tk)
        )
        sfa_smem = cute.slice_(self.sfa_smem_layout_staged, (None, None, 0))
        sfb_smem = cute.slice_(self.sfb_smem_layout_staged, (None, None, 0))
        tma_atom_sfa, tma_sfa = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            sfa3,
            sfa_smem,
            (self.cta_m, 128),
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_sfb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            sfb3,
            sfb_smem,
            (128, 128),
            internal_type=cutlass.Int16,
        )
        epi_smem = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c3_tma, epi_smem, self.epi_tile
        )

        num_m_tiles = (m + self.cta_m - 1) // self.cta_m

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            # PipelineOrder(depth=2, length=2): depth * length mbarriers.
            order_barrier_ptr: cute.struct.MemRange[cutlass.Int64, 4]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, self.epi_cosize * (2 if self.coop else 1)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            tma_atom_sfa,
            tma_sfa,
            tma_atom_sfb,
            tma_sfb,
            tma_atom_c,
            tma_c,
            c3,
            self.tiled_mma,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.sfb_mma_layout_staged,
            self.sfa_mma_layout_staged,
            self.epi_smem_layout_staged,
            num_m_tiles,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    # ------------------------------------------------------------ scheduling
    @cute.jit
    def tile_coord(self, linear: Int32, num_m_tiles: Int32):
        """Grouped rasterization with exact tail handling (no padded tiles).

        ``group`` is the largest group whose A panels fit the L2 budget; it is
        clamped to the runtime tile count so small M forms a single group.
        """
        # Overshooting the L2 budget by up to 25% beats a lone trailing group,
        # which would stream all of B from DRAM again for one M-tile row.
        gmax = max(1, self.group * 128 // self.cta_m)
        g = cutlass.min(num_m_tiles, gmax)
        if num_m_tiles <= gmax * 5 // 4:
            g = num_m_tiles
        # Groups of g M tiles; sweep N inside a group (A panels stay in L2).
        per_group = g * self.num_n_tiles
        gid = linear // per_group
        first_m = gid * g
        gsz = cutlass.min(num_m_tiles - first_m, g)
        r = linear - gid * per_group
        m_idx = first_m + r % gsz
        n_idx = r // gsz
        return m_idx, n_idx

    @cute.jit
    def advance(self, state: pipeline.PipelineState, iterations):
        if iterations < state.stages and ((state._index + iterations) >= state.stages):
            state._phase ^= 1
        if (
            iterations >= state.stages
            and (((state._index + iterations) // state.stages) % 2) == 1
        ):
            state._phase ^= 1
        state._index = (state._index + iterations) % state.stages
        state._count += iterations
        return state

    # ---------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_tma: cute.Tensor,
        mC: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        sfb_mma_layout_staged: cute.Layout,
        sfa_mma_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        num_m_tiles: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            if cutlass.const_expr(not self.direct_epilogue):
                cpasync.prefetch_descriptor(tma_atom_c)

        tm, tn, tk = self.tile_shape_mnk
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.ab_dtype, a_smem_layout)
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_warps if self.coop else self.num_mma_warps // 2,
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        order_barrier = pipeline.PipelineOrder.create(
            barrier_storage=storage.order_barrier_ptr.data_ptr(),
            depth=2,
            length=2,
            group_id=warp_group_idx,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        order_state = order_barrier.state

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        if cutlass.const_expr(self.coop):
            sC = cute.make_tensor(
                sC.iterator
                + cute.assume(warp_group_idx * self.epi_cosize, divby=self.epi_cosize),
                sC.layout,
            )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sSFB_mma = cute.make_tensor(sSFB.iterator, sfb_mma_layout_staged)

        # (bM, bK, loopM, loopK, loopL)
        gA = cute.local_tile(mA, (self.cta_m, tk), (None, None, None))
        gB = cute.local_tile(mB, (tn, tk), (None, None, None))
        gSFA = cute.local_tile(mSFA, (self.cta_m, 128), (None, None, None))
        gSFB = cute.local_tile(mSFB, (128, 128), (None, None, None))
        gC_tma = cute.local_tile(mC_tma, (tm, tn), (None, None, None))

        thr_mma = tiled_mma.get_slice(tidx % 128)
        one = cute.make_layout(1)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, one, cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2)
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, one, cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2)
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            0,
            one,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            0,
            one,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # Operand views for this warp group's MMA rows.
        sA_mma = sA
        sSFA_mma = sSFA
        if cutlass.const_expr(self.coop):
            sA_mma = cute.local_tile(
                sA, (128, tk, self.ab_stage), (warp_group_idx, 0, 0)
            )
            sSFA_mma = cute.make_tensor(
                sSFA.iterator + warp_group_idx * SF_CHUNK, sfa_mma_layout_staged
            )
        tCsA = thr_mma.partition_A(sA_mma)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA = sm120_utils.partition_fragment_SFA(
            sSFA_mma[None, None, 0], thr_mma, tidx
        )
        tCrSFB = sm120_utils.partition_fragment_SFB(
            sSFB_mma[None, None, 0], thr_mma, tidx
        )
        tCgC = thr_mma.partition_C(gC_tma[None, None, 0, 0, 0])
        accumulators = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)

        cute.arch.sync_threads()

        k_tile_cnt = self.k_tile_cnt
        total_units = num_m_tiles * self.num_n_tiles

        prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        if warp_idx >= self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            if warp_idx == self.tma_load_warp_id:
                linear = bidx
                while linear < total_units:
                    m_idx, n_idx = self.tile_coord(linear, num_m_tiles)
                    tAgA_t = tAgA[(None, m_idx, None, 0)]
                    tBgB_t = tBgB[(None, n_idx, None, 0)]
                    tAgSFA_t = tAgSFA[(None, m_idx, None, 0)]
                    tBgSFB_t = tBgSFB[(None, n_idx // (128 // tn), None, 0)]
                    prod_state.reset_count()
                    for kt in cutlass.range(k_tile_cnt, unroll=1):
                        mainloop_pipeline.producer_acquire(prod_state)
                        bar = mainloop_pipeline.producer_get_barrier(prod_state)
                        cute.copy(
                            tma_atom_b,
                            tBgB_t[(None, kt)],
                            tBsB[(None, prod_state.index)],
                            tma_bar_ptr=bar,
                        )
                        kt_sf = kt
                        if cutlass.const_expr(self.sf_shared):
                            kt_sf = kt // 2
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_t[(None, kt_sf)],
                            tBsSFB[(None, prod_state.index)],
                            tma_bar_ptr=bar,
                        )
                        # Returns at once after the first call.
                        cute.arch.griddepcontrol_wait()
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_t[(None, kt_sf)],
                            tAsSFA[(None, prod_state.index)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_atom_a,
                            tAgA_t[(None, kt)],
                            tAsA[(None, prod_state.index)],
                            tma_bar_ptr=bar,
                        )
                        mainloop_pipeline.producer_commit(prod_state)
                        prod_state.advance()
                    linear += gdim
                cute.arch.griddepcontrol_launch_dependents()
                mainloop_pipeline.producer_tail(prod_state)
        else:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            cute.arch.griddepcontrol_wait()
            num_k_blocks = cute.size(tCrA, mode=[2])

            smem_copy_A = cute.make_tiled_copy_A(
                cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.ab_dtype
                ),
                tiled_mma,
            )
            smem_copy_B = cute.make_tiled_copy_B(
                cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), self.ab_dtype
                ),
                tiled_mma,
            )
            sf_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
            smem_copy_SFA = cute.make_tiled_copy(
                sf_atom,
                sm120_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_copy_SFB = cute.make_tiled_copy(
                sf_atom,
                sm120_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            thr_copy_A = smem_copy_A.get_slice(tidx % 128)
            thr_copy_B = smem_copy_B.get_slice(tidx % 128)
            tCsA_v = thr_copy_A.partition_S(sA_mma)
            tCrA_v = thr_copy_A.retile(tCrA)
            tCsB_v = thr_copy_B.partition_S(sB)
            tCrB_v = thr_copy_B.retile(tCrB)
            thr_copy_SFA = smem_copy_SFA.get_slice(tidx % 128)
            thr_copy_SFB = smem_copy_SFB.get_slice(tidx % 128)
            tCsSFA_v = cute.filter_zeros(thr_copy_SFA.partition_S(sSFA_mma))
            tCrSFA_v = cute.filter_zeros(thr_copy_SFA.retile(tCrSFA))
            tCsSFB_v0 = cute.filter_zeros(thr_copy_SFB.partition_S(sSFB_mma))
            tCrSFB_v = cute.filter_zeros(thr_copy_SFB.retile(tCrSFB))

            linear = bidx
            if cutlass.const_expr(not self.coop):
                if warp_group_idx == 1:
                    linear += gdim
                    cons_state = self.advance(cons_state, k_tile_cnt)

            while linear < total_units:
                m_idx, n_idx = self.tile_coord(linear, num_m_tiles)
                tCsSFB_v = tCsSFB_v0
                if cutlass.const_expr(tn == 64):
                    tCsSFB_v = cute.make_tensor(
                        tCsSFB_v0.iterator + (n_idx % 2) * 8, tCsSFB_v0.layout
                    )
                accumulators.fill(0.0)
                cons_state.reset_count()
                if cutlass.const_expr(not self.coop):
                    order_barrier.wait(order_state)

                # Software-pipelined mainloop: the shared-to-register copy of
                # k-block kb + 1 (possibly from the next stage) overlaps the MMAs
                # of kb. With early_mma the MMAs of a stage's last k-block are
                # issued before blocking on the next stage's TMA barrier.
                mainloop_pipeline.consumer_wait(cons_state)
                self._load_kblock(
                    smem_copy_A,
                    smem_copy_B,
                    smem_copy_SFA,
                    smem_copy_SFB,
                    tCsA_v,
                    tCrA_v,
                    tCsB_v,
                    tCrB_v,
                    tCsSFA_v,
                    tCrSFA_v,
                    tCsSFB_v,
                    tCrSFB_v,
                    cons_state.index,
                    0,
                    0,
                    self._sf_off(0, num_k_blocks),
                )
                for kt in cutlass.range(k_tile_cnt - 1, unroll=1):
                    sf_off = self._sf_off(kt, num_k_blocks)
                    sf_off_next = self._sf_off(kt + 1, num_k_blocks)
                    for kb in cutlass.range_constexpr(num_k_blocks):
                        if kb == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(cons_state)
                            cons_state.advance()
                            if cutlass.const_expr(self.early_mma):
                                self._mma_kblock(
                                    tiled_mma,
                                    accumulators,
                                    tCrA,
                                    tCrSFA,
                                    tCrB,
                                    tCrSFB,
                                    kb,
                                )
                            mainloop_pipeline.consumer_wait(cons_state)
                            self._load_kblock(
                                smem_copy_A,
                                smem_copy_B,
                                smem_copy_SFA,
                                smem_copy_SFB,
                                tCsA_v,
                                tCrA_v,
                                tCsB_v,
                                tCrB_v,
                                tCsSFA_v,
                                tCrSFA_v,
                                tCsSFB_v,
                                tCrSFB_v,
                                cons_state.index,
                                0,
                                0,
                                sf_off_next,
                            )
                            if cutlass.const_expr(not self.early_mma):
                                self._mma_kblock(
                                    tiled_mma,
                                    accumulators,
                                    tCrA,
                                    tCrSFA,
                                    tCrB,
                                    tCrSFB,
                                    kb,
                                )
                        else:
                            self._load_kblock(
                                smem_copy_A,
                                smem_copy_B,
                                smem_copy_SFA,
                                smem_copy_SFB,
                                tCsA_v,
                                tCrA_v,
                                tCsB_v,
                                tCrB_v,
                                tCsSFA_v,
                                tCrSFA_v,
                                tCsSFB_v,
                                tCrSFB_v,
                                cons_state.index,
                                kb + 1,
                                kb + 1,
                                sf_off,
                            )
                            self._mma_kblock(
                                tiled_mma, accumulators, tCrA, tCrSFA, tCrB, tCrSFB, kb
                            )
                sf_off_last = self._sf_off(k_tile_cnt - 1, num_k_blocks)
                for kb in cutlass.range_constexpr(num_k_blocks):
                    if kb == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(cons_state)
                        cons_state.advance()
                    else:
                        self._load_kblock(
                            smem_copy_A,
                            smem_copy_B,
                            smem_copy_SFA,
                            smem_copy_SFB,
                            tCsA_v,
                            tCrA_v,
                            tCsB_v,
                            tCrB_v,
                            tCsSFA_v,
                            tCrSFA_v,
                            tCsSFB_v,
                            tCrSFB_v,
                            cons_state.index,
                            kb + 1,
                            kb + 1,
                            sf_off_last,
                        )
                    self._mma_kblock(
                        tiled_mma, accumulators, tCrA, tCrSFA, tCrB, tCrSFB, kb
                    )

                if cutlass.const_expr(not self.coop):
                    order_state = order_barrier.arrive(order_state)
                    order_barrier.wait(order_state)
                    cons_state = self.advance(cons_state, k_tile_cnt)

                m_epi = m_idx
                if cutlass.const_expr(self.coop):
                    m_epi = m_idx * 2 + warp_group_idx
                self._epilogue(
                    tiled_mma,
                    accumulators,
                    sC,
                    tma_atom_c,
                    gC_tma,
                    m_epi,
                    n_idx,
                    tidx,
                    warp_idx,
                    warp_group_idx,
                    mC[None, None, 0],
                )
                if cutlass.const_expr(self.coop):
                    linear += gdim
                else:
                    linear += 2 * gdim
                    order_state = order_barrier.arrive(order_state)

    @cute.jit
    def _sf_off(self, kt, num_k_blocks: cutlass.Constexpr):
        """K64 stages read the half of the shared K128 scale chunk they cover."""
        off = 0
        if cutlass.const_expr(self.sf_shared):
            off = (kt % 2) * num_k_blocks
        return off

    @cute.jit
    def _load_kblock(
        self,
        copy_a,
        copy_b,
        copy_sfa,
        copy_sfb,
        tCsA_v,
        tCrA_v,
        tCsB_v,
        tCrB_v,
        tCsSFA_v,
        tCrSFA_v,
        tCsSFB_v,
        tCrSFB_v,
        stage,
        src_kb: cutlass.Constexpr,
        dst_kb: cutlass.Constexpr,
        sf_off,
    ):
        cute.copy(copy_a, tCsA_v[None, None, src_kb, stage], tCrA_v[None, None, dst_kb])
        cute.copy(copy_b, tCsB_v[None, None, src_kb, stage], tCrB_v[None, None, dst_kb])
        cute.copy(
            copy_sfa,
            tCsSFA_v[None, None, src_kb + sf_off, stage],
            tCrSFA_v[None, None, dst_kb],
        )
        cute.copy(
            copy_sfb,
            tCsSFB_v[None, None, src_kb + sf_off, stage],
            tCrSFB_v[None, None, dst_kb],
        )

    @cute.jit
    def _mma_kblock(
        self, tiled_mma, acc, tCrA, tCrSFA, tCrB, tCrSFB, kb: cutlass.Constexpr
    ):
        cute.gemm(
            tiled_mma,
            acc,
            [tCrA[None, None, kb], tCrSFA[None, None, kb]],
            [tCrB[None, None, kb], tCrSFB[None, None, kb]],
            acc,
        )

    @cute.jit
    def _epilogue(
        self,
        tiled_mma,
        accumulators,
        sC,
        tma_atom_c,
        gC_tma,
        m_idx,
        n_idx,
        tidx,
        warp_idx,
        warp_group_idx,
        mC,
    ):
        """Registers -> swizzled shared sub-tiles (stmatrix) -> global, by TMA
        store or, for row pitches that are not 16-byte aligned, by coalesced
        predicated stores from all 128 threads of the warp group."""
        tm, tn, _ = self.tile_shape_mnk
        copy_atom_r2s = sm120_utils.sm120_get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(False, 2), self.c_dtype
        )
        tiled_copy_C_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_atom)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx % 128)
        tRS_sD = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

        gC_tile = gC_tma[(None, None, m_idx, n_idx, 0)]
        bSG_sD, bSG_gD = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.zipped_divide(gC_tile, self.epi_tile),
        )
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
        )

        epi_rest_m = bSG_gD.shape[1][0]
        epi_rest_n = bSG_gD.shape[1][1]
        mma_tile_m = tm // cute.size(tRS_rAcc, mode=[1])
        mma_tile_n = tn // cute.size(tRS_rAcc, mode=[2])
        mm_per_epi = self.epi_tile[0] // mma_tile_m
        nn_per_epi = self.epi_tile[1] // mma_tile_n
        for epi_m in cutlass.range_constexpr(epi_rest_m):
            for epi_n in cutlass.range_constexpr(epi_rest_n):
                for mn in cutlass.range_constexpr(nn_per_epi):
                    for mm in cutlass.range_constexpr(mm_per_epi):
                        dst = tRS_rD[(None, mm, mn)]
                        src = tRS_rAcc[
                            (None, epi_m * mm_per_epi + mm, epi_n * nn_per_epi + mn)
                        ]
                        for e in cutlass.range_constexpr(cute.size(dst)):
                            dst[e] = src[e]
                tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
                tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))
                epi_buffer = (epi_m * epi_rest_n + epi_n) % self.epi_stage
                self._epi_barrier(warp_group_idx)
                cute.copy(
                    tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                self._epi_barrier(warp_group_idx)
                if cutlass.const_expr(self.direct_epilogue):
                    em, en = self.epi_tile
                    m = cute.size(mC, mode=[0])
                    t = tidx % 128
                    row0 = m_idx * tm + epi_m * em
                    col0 = n_idx * tn + epi_n * en
                    for i in cutlass.range_constexpr(em * en // 128):
                        idx = i * 128 + t
                        r = idx // en
                        cc = idx - r * en
                        if row0 + r < m and col0 + cc < self.n:
                            mC[row0 + r, col0 + cc] = sC[r, cc, epi_buffer]
                elif warp_idx % 4 == 0:
                    cute.copy(
                        tma_atom_c,
                        bSG_sD[(None, epi_buffer)],
                        bSG_gD[(None, (epi_m, epi_n))],
                    )
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()
        if cutlass.const_expr(not self.direct_epilogue):
            if warp_idx % 4 == 0:
                tma_store_pipeline.producer_tail()

    @cute.jit
    def _epi_barrier(self, warp_group_idx):
        if warp_group_idx == 0:
            self.epilog_sync_barrier.arrive_and_wait()
        else:
            self.epilog_sync_barrier_wg1.arrive_and_wait()
