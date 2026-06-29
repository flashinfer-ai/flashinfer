# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
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

"""bf16 x fp4 dense GEMM for Blackwell (SM100/103/120/121).

Built on top of ``dense_gemm_bf16_blackwell.py`` from cutlass examples at
https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py
"""

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32, Uint32

from ....cute_dsl.fp4_common import (
    cvt_s0e5m3_to_f16x2_broadcast,
    f16x2_to_f32x2,
    fp4_decode_4bytes,
    get_smem_ptr_as_int32,
    half2_mul,
    ld_shared_v2_u32,
)
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as T
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def cvt_bf16x2_to_f16x2_via_f32(packed_bf16x2: Uint32, *, loc=None, ip=None) -> Uint32:
    """Packed bf16x2 (u32) -> f16x2 (u32) via f32 intermediate.

    Used by the fp16-MMA path (use_fp16_mma=1) to convert A's ldmatrix
    output (bf16 bit pattern from sA) to fp16 bit pattern for the fp16
    MMA inputs.  Direct ``cvt.rn.f16x2.bf16x2`` is rejected by ptxas on
    sm_100a / CUDA 13.1; this packed via-f32 path is 4 PTX instrs per
    pair (mov.b32 unpack + 2x cvt.f32.bf16 + 1x cvt.rn.f16x2.f32).
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(packed_bf16x2).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 b_lo, b_hi;
                .reg .f32 f_lo, f_hi;
                mov.b32 {b_lo, b_hi}, $1;
                cvt.f32.bf16 f_lo, b_lo;
                cvt.f32.bf16 f_hi, b_hi;
                cvt.rn.f16x2.f32 $0, f_hi, f_lo;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f16x2_unpack(
    packed_h2: Uint32, *, loc=None, ip=None
) -> Tuple["cutlass.Float16", "cutlass.Float16"]:
    """Unpack f16x2 (u32) into (f16_lo, f16_hi).  Free at HW level --
    just a register-rename (mov.b32 {h_lo, h_hi}, packed).  Used by the
    fp16-MMA path to bypass the f16->f32->bf16 cvt chain that the
    default bf16-MMA path needs after hmul2."""
    from cutlass import Float16

    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f16, f16)>"),
        [Uint32(packed_h2).ir_value(loc=loc, ip=ip)],
        "mov.b32 {$0, $1}, $2;",
        "=h,=h,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float16(llvm.extractvalue(T.f16(), res, [0], loc=loc, ip=ip)),
        Float16(llvm.extractvalue(T.f16(), res, [1], loc=loc, ip=ip)),
    )


# FP4 weight packing constants (must match _cute_dsl_pack_fp4_weight in
# flashinfer/gemm/gemm_bf16_fp4_cute_dsl.py).
_PACK_TILE_K: cutlass.Constexpr = 16
_PACK_TILE_N: cutlass.Constexpr = 64
_PACK_INTS_PER_TILE: cutlass.Constexpr = 128  # 128 int32 per (16K x 64N) block


class BlackwellDenseGemmBf16Fp4Kernel:
    """Warp-MMA dense GEMM for Blackwell, FP4-weight A bf16/fp16 input.

    A: (M, K, L) bf16/fp16.
    B: (K // 16, N * 2, L) int32 -- packed FP4 (see prepare).
    B_sf: (K // 16, N, L) uint8 -- per-group scales (S0E5M3).
    alpha: (1,) fp32 -- global scalar scale.
    C: (M, N, L) bf16/fp16; fp32 accumulator cast at write.

    """

    GROUP_SIZE: cutlass.Constexpr = 16

    def __init__(
        self,
        acc_dtype,
        tile_shape_mnk,
        epi_stage: int = 4,
        pipeline_depth: int = 1,
        atom_layout: Tuple[int, int, int] = (2, 2, 1),
        epi_tile_override: Optional[Tuple[int, int]] = None,
        # 1 = fp16 MMA (default): MmaF16BF16Op uses Float16.  Dequant writes
        #     fp16 directly to tCrB (skipping the f16->f32->bf16 cvt chain
        #     the bf16 path needs after hmul2).  A is bf16 in SMEM ->
        #     ldmatrix into bf16 staging fragment -> in-register packed
        #     bf16->fp16 cvt -> fp16 tCrA.  Slightly *more* accurate than
        #     bf16 MMA for well-behaved inputs since fp16's 10-bit mantissa
        #     beats bf16's 7-bit at the multiply step; accumulator stays
        #     fp32 in both modes.
        # 0 = bf16 MMA: original path.  Safer for workloads with very
        #     large activation magnitudes (|A| > ~30000) since bf16's
        #     wider exponent range avoids saturation in the A cvt.
        use_fp16_mma: int = 1,
        enable_pdl: bool = True,
        tile_swizzle: int = 1,
        raster_along_m: bool = True,
    ):
        """bf16 x fp4 kernel.

        Args:
            acc_dtype: accumulator dtype (always Float32 for this kernel).
            tile_shape_mnk: CTA tile shape.
            epi_stage: TMA-store pipeline depth.  Default 4 balances
                cross-tile overlap (epi_stage > 1 lets next-tile compute
                overlap with this-tile store) against SMEM available for
                ab_stage (deeper ab_stage hides TMA-load latency, which
                dominates the small-M shape's stalls).  At M=4 with the
                default 64-CTA grid on 148 SMs, each SM gets at most
                one tile, so epi_stage > 1 doesn't help much locally;
                we keep 2-4 to preserve persistent-scheduler benefits
                at larger M.
        """
        self.acc_dtype = acc_dtype
        self.cluster_shape_mnk = (1, 1, 1)
        self.tile_shape_mnk = tuple(tile_shape_mnk)
        self.epi_stage_target = int(epi_stage)
        if self.epi_stage_target < 1:
            raise ValueError(f"epi_stage must be >= 1 (got {epi_stage})")
        self.pipeline_depth = int(pipeline_depth)
        self.use_fp16_mma = int(use_fp16_mma)
        self.tile_swizzle = int(tile_swizzle)
        self.raster_along_m = bool(raster_along_m)
        self.enable_pdl = bool(enable_pdl)
        # Optional override for the epilogue tile shape.
        self.epi_tile_override = (
            tuple(epi_tile_override) if epi_tile_override is not None else None
        )
        self.tiled_mma = None
        # num_mcast_ctas_a / num_mcast_ctas_b are derived from the cluster
        # shape in __call__ before use; left unset here so their type is
        # inferred from that assignment (matches the sibling kernels).
        self.is_a_mcast = False
        self.is_b_mcast = False

        if self.tile_shape_mnk[1] % _PACK_TILE_N != 0:
            raise ValueError(
                f"bf16 x fp4 requires tile_N % {_PACK_TILE_N} == 0 "
                f"(got tile_N={self.tile_shape_mnk[1]})"
            )
        if self.tile_shape_mnk[2] % _PACK_TILE_K != 0:
            raise ValueError(
                f"bf16 x fp4 requires tile_K % {_PACK_TILE_K} == 0 "
                f"(got tile_K={self.tile_shape_mnk[2]})"
            )

        self.occupancy = 1
        # 2x2 atom layout: 4 MMA warps arranged as 2 M-warps x 2 N-warps.
        self.atom_layout = tuple(atom_layout)
        if self.atom_layout not in ((2, 2, 1), (1, 2, 1)):
            raise ValueError(
                f"Unsupported atom_layout {self.atom_layout!r}; "
                "expected (2,2,1) or (1,2,1)"
            )
        self.num_mma_warps = (
            self.atom_layout[0] * self.atom_layout[1] * self.atom_layout[2]
        )
        self.num_dma_warps = 1
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + self.num_dma_warps
        ) * self.num_threads_per_warp
        # SM100/103 expose >= SM120 SMEM/CTA
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        self.mma_inst_mnk = (16, 8, 16)
        # MmaF16BF16Op accepts ab_dtype in {Float16, BFloat16}.  We pick via
        # b_compute_dtype so use_fp16_mma=1 swings the whole MMA to fp16
        # (both A and B fragments will be fp16-typed).
        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.b_compute_dtype,
            self.acc_dtype,
            self.mma_inst_mnk,
        )
        tC = cute.make_layout(self.atom_layout)
        permutation_mnk = (
            self.atom_layout[0] * self.mma_inst_mnk[0],
            # *2 trick: each warp covers two atom-N tiles in one ldmatrix.x4
            self.atom_layout[1] * self.mma_inst_mnk[1] * 2,
            self.atom_layout[2] * self.mma_inst_mnk[2],
        )
        self.tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        self.num_mcast_ctas_a = self.cluster_shape_mnk[1]
        self.num_mcast_ctas_b = self.cluster_shape_mnk[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk,
            self.c_dtype,
            is_cooperative=False,
            epi_tile_override=self.epi_tile_override,
        )

        # B-side smem is packed int32 (4 bytes per logical FP4
        # pair).  Stage budget uses int32 B + uint8 scales; bf16 phantom
        # layout is only used by partition_B/make_fragment_B.
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
            self.GROUP_SIZE,
            self.epi_stage_target,
        )

        if self.ab_stage == 0:
            raise RuntimeError(
                "ab_stage == 0: not enough shared memory for this tile shape "
                f"({self.tile_shape_mnk}) at occupancy {self.occupancy}."
            )

        (
            self.a_smem_layout_staged,
            self.b_packed_smem_layout_staged,
            self.b_sf_smem_layout_staged,
            self.b_bf16_logical_layout,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_compute_dtype,
            self.b_layout_compute,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.GROUP_SIZE,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b_packed: cute.Tensor,
        b_sf: cute.Tensor,
        c: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        from cutlass import Float16

        self.a_dtype = a.element_type
        # MMA operand dtype: bf16 by default matches A, or fp16 when
        # use_fp16_mma=1 (lets us skip the f16->f32->bf16 cvt chain on B).
        if cutlass.const_expr(self.use_fp16_mma == 1):
            self.b_compute_dtype = Float16
        else:
            self.b_compute_dtype = a.element_type
        self.c_dtype = c.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout_compute = utils.LayoutEnum.ROW_MAJOR
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype.width != 16):
            raise TypeError(f"a_dtype must be 16-bit (bf16/fp16), got {self.a_dtype}")
        if cutlass.const_expr(self.a_dtype != self.c_dtype):
            raise TypeError(
                f"a_dtype and c_dtype must match, got {self.a_dtype} vs {self.c_dtype}"
            )

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )

        # Packed B TMA: tile is (tile_K // 16, 2 * tile_N) int32.
        b_packed_tma_tile = (
            self.tile_shape_mnk[2] // _PACK_TILE_K,
            2 * self.tile_shape_mnk[1],
        )
        tma_atom_b_packed, tma_tensor_b_packed = self._make_tma_atoms_and_tensors(
            b_packed,
            self.b_packed_smem_layout_staged,
            b_packed_tma_tile,
            1,
        )

        # B scale TMA: tile is (tile_K // group_size, tile_N) uint8.
        # Small per-tile load -- this replaces ``tile_K // group_size``
        # gmem-direct loads per thread per K-block in the dequant path.
        b_sf_tma_tile = (
            self.tile_shape_mnk[2] // self.GROUP_SIZE,
            self.tile_shape_mnk[1],
        )
        tma_atom_b_sf, tma_tensor_b_sf = self._make_tma_atoms_and_tensors(
            b_sf,
            self.b_sf_smem_layout_staged,
            b_sf_tma_tile,
            1,
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
            self.tile_swizzle,
            self.raster_along_m,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB_packed: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Int32, cute.cosize(self.b_packed_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB_sf: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Uint8, cute.cosize(self.b_sf_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b_packed,
            tma_tensor_b_packed,
            tma_atom_b_sf,
            tma_tensor_b_sf,
            tma_atom_c,
            tma_tensor_c,
            alpha,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_packed_smem_layout_staged,
            self.b_sf_smem_layout_staged,
            self.b_bf16_logical_layout,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b_packed: cute.CopyAtom,
        mB_packed_kn: cute.Tensor,
        tma_atom_b_sf: cute.CopyAtom,
        mB_sf_kn: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mAlpha: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_packed_smem_layout_staged: cute.Layout,
        b_sf_smem_layout_staged: cute.Layout,
        b_bf16_logical_layout: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors from warp 0.
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b_packed)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b_sf)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_packed_smem_layout = cute.slice_(b_packed_smem_layout_staged, (None, None, 0))
        b_sf_smem_layout = cute.slice_(b_sf_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(cutlass.Int32, b_packed_smem_layout)
            + cute.size_in_bytes(cutlass.Uint8, b_sf_smem_layout)
        )

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # Both B-side tensors are plain (non-swizzled) staged layouts.
        sB_packed = storage.sB_packed.get_tensor(b_packed_smem_layout_staged)
        sB_sf = storage.sB_sf.get_tensor(b_sf_smem_layout_staged)
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # Packed B: (K // 16, N * 2, L); tile = (tile_K // 16, 2 * tile_N).
        b_packed_tile_shape = (
            self.tile_shape_mnk[2] // _PACK_TILE_K,
            2 * self.tile_shape_mnk[1],
        )
        gB_packed_kn = cute.local_tile(
            mB_packed_kn,
            b_packed_tile_shape,
            (None, None, None),
        )
        # Scales: (K // 16, N, L); per-tile slice has shape
        # (tile_K // group_size, tile_N) = (tile_K // 16, tile_N).  Dequant
        # indexes directly per K-block + per N-coord.
        gB_sf_kn = cute.local_tile(
            mB_sf_kn,
            (self.tile_shape_mnk[2] // self.GROUP_SIZE, self.tile_shape_mnk[1]),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partition for A: (m, k) -> per-CTA partition.
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA partition for B (packed int32).
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBpacked_s, tBpacked_g = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b_packed,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_packed, 0, 2),
            cute.group_modes(gB_packed_kn, 0, 2),
        )

        # TMA partition for B_sf (uint8 scales).  Shares the b_cta_crd
        # (no N multicast since cluster_shape_mnk = (1,1,1)).
        tBsf_s, tBsf_g = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b_sf,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB_sf, 0, 2),
            cute.group_modes(gB_sf_kn, 0, 2),
        )

        # B partition uses a phantom bf16 layout on top of the sB_packed
        # int32 storage (recast pointer + bf16 logical layout).  This
        # gives ``partition_B``/``make_fragment_B`` the right fragment
        # shape; the data is never read through this view -- we always
        # decode FP4 ourselves.
        sB_phantom = cute.make_tensor(
            cute.recast_ptr(sB_packed.iterator, dtype=self.b_compute_dtype),
            b_bf16_logical_layout,
        )
        tCsA = thr_mma.partition_A(sA)
        tCsB_phantom = thr_mma.partition_B(sB_phantom)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB_phantom[None, None, None, 0])

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            pipeline.sync(barrier_id=1)

        # PDL bookend (start): wait for the prior grid to finish so this
        # kernel's TMA loads see the producer kernel's writes.
        cute.arch.griddepcontrol_wait()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group: warps [0, num_mma_warps) compute.
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # ldmatrix is only for A.  B is filled in-register from sB_packed
            # via FP4 decode + per-group scale.
            #
            # use_fp16_mma=1: tCrA is fp16-typed (matches the fp16 MMA), but
            # sA is bf16 in SMEM.  ldmatrix into a bf16 staging fragment,
            # then per-K-block convert bf16 -> fp16 in-register before MMA
            # reads tCrA.  See the inline cvt at each ldmatrix site.
            from cutlass import BFloat16

            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            if cutlass.const_expr(self.use_fp16_mma == 1):
                tCrA_bf16 = cute.make_fragment_like(tCrA, BFloat16)
                tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA_bf16)
            else:
                tCrA_bf16 = tCrA
                tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)

            alpha_val = Float32(mAlpha[Int32(0)])

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                accumulators.fill(0.0)

                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]

                # Prologue: prefetch tCrA[0] and tCrB[0] only when running with
                # pipeline_depth >= 1.  With depth=0 the dequant for block k
                # happens inside the same iteration as gemm(k) -- no prefetch.
                if cutlass.const_expr(self.pipeline_depth >= 1):
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_p[None, None, 0],
                        tCrA_copy_view[None, None, 0],
                    )
                    if cutlass.const_expr(self.use_fp16_mma == 1):
                        self._cvt_a_bf16_to_fp16_one_k_block(tCrA, tCrA_bf16, 0)
                    self._dequant_b_to_register(
                        sB_packed,
                        sB_sf,
                        tCrB,
                        tidx,
                        mainloop_consumer_state.index,
                        mainloop_consumer_state.count,
                        0,
                    )

                for _k_tile in cutlass.range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if cutlass.const_expr(self.pipeline_depth == 0):
                            # No-prefetch: dequant(k) just before gemm(k),
                            # all on the CURRENT K-tile's stage.  Release +
                            # advance + wait happens AFTER gemm of the last
                            # block so we don't drop SMEM under our own read.
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_idx],
                                tCrA_copy_view[None, None, k_block_idx],
                            )
                            if cutlass.const_expr(self.use_fp16_mma == 1):
                                self._cvt_a_bf16_to_fp16_one_k_block(
                                    tCrA, tCrA_bf16, k_block_idx
                                )
                            self._dequant_b_to_register(
                                sB_packed,
                                sB_sf,
                                tCrB,
                                tidx,
                                mainloop_consumer_state.index,
                                mainloop_consumer_state.count,
                                k_block_idx,
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[None, None, k_block_idx],
                                tCrB[None, None, k_block_idx],
                                accumulators,
                            )
                            if k_block_idx == num_k_blocks - 1:
                                mainloop_pipeline.consumer_release(
                                    mainloop_consumer_state
                                )
                                mainloop_consumer_state.advance()
                                peek_ab_full_status = cutlass.Boolean(1)
                                peek_ab_full_status = (
                                    mainloop_pipeline.consumer_try_wait(
                                        mainloop_consumer_state
                                    )
                                )
                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )
                        else:
                            # 1-stage prefetch: dequant(k+1) while gemm(k).
                            if k_block_idx == num_k_blocks - 1:
                                mainloop_pipeline.consumer_release(
                                    mainloop_consumer_state
                                )
                                mainloop_consumer_state.advance()

                                peek_ab_full_status = cutlass.Boolean(1)
                                peek_ab_full_status = (
                                    mainloop_pipeline.consumer_try_wait(
                                        mainloop_consumer_state
                                    )
                                )

                                tCsA_p = tCsA_copy_view[
                                    None, None, None, mainloop_consumer_state.index
                                ]
                                mainloop_pipeline.consumer_wait(
                                    mainloop_consumer_state, peek_ab_full_status
                                )

                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                            if cutlass.const_expr(self.use_fp16_mma == 1):
                                self._cvt_a_bf16_to_fp16_one_k_block(
                                    tCrA, tCrA_bf16, k_block_next
                                )
                            self._dequant_b_to_register(
                                sB_packed,
                                sB_sf,
                                tCrB,
                                tidx,
                                mainloop_consumer_state.index,
                                mainloop_consumer_state.count,
                                k_block_next,
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[None, None, k_block_idx],
                                tCrB[None, None, k_block_idx],
                                accumulators,
                            )
                # Hoist out last k_tile (no further loads after the last k_block)
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if cutlass.const_expr(self.pipeline_depth == 0):
                        # No-prefetch path for last K-tile.  Release happens
                        # AFTER gemm of the last block (kernel exits then).
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_idx],
                            tCrA_copy_view[None, None, k_block_idx],
                        )
                        if cutlass.const_expr(self.use_fp16_mma == 1):
                            self._cvt_a_bf16_to_fp16_one_k_block(
                                tCrA, tCrA_bf16, k_block_idx
                            )
                        self._dequant_b_to_register(
                            sB_packed,
                            sB_sf,
                            tCrB,
                            tidx,
                            mainloop_consumer_state.index,
                            mainloop_consumer_state.count,
                            k_block_idx,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                    else:
                        # 1-stage prefetch path for last K-tile.
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                        if k_block_next > 0:
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_block_next],
                                tCrA_copy_view[None, None, k_block_next],
                            )
                            if cutlass.const_expr(self.use_fp16_mma == 1):
                                self._cvt_a_bf16_to_fp16_one_k_block(
                                    tCrA, tCrA_bf16, k_block_next
                                )
                            self._dequant_b_to_register(
                                sB_packed,
                                sB_sf,
                                tCrB,
                                tidx,
                                mainloop_consumer_state.index,
                                mainloop_consumer_state.count,
                                k_block_next,
                            )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[None, None, k_block_idx],
                            tCrB[None, None, k_block_idx],
                            accumulators,
                        )

                # Epilogue: accumulator -> smem -> gmem via R2S (StMatrix.x4)
                # + TMA bulk store.
                copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        4,
                    ),
                    self.c_dtype,
                )

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)

                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )

                # Epilogue: iterate (epi_m, epi_n) explicitly
                # and use (mma_m, mma_n) mode indexing into tRS_rAcc so
                # the loop works for any epi_tile_m / epi_tile_n.  Also
                # supports OOB-iteration skipping when m_actual < tile_M.
                epi_rest_m = cute.size(tcgc_for_tma_partition, mode=[1, 0])
                epi_rest_n = cute.size(tcgc_for_tma_partition, mode=[1, 1])
                epi_tile_m = self.epi_tile[0]
                epi_tile_n = self.epi_tile[1]
                # mma_tile_{m,n} = per-mma-atom (M,N) size.  tRS_rAcc has
                # shape (atom_v, mma_m, mma_n); modes 1, 2 give the atom
                # counts in M, N.
                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                MmaMPerEpiM = epi_tile_m // mma_tile_m
                MmaNPerEpiN = epi_tile_n // mma_tile_n

                tma_store_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_warps * self.num_threads_per_warp,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=tma_store_producer_group,
                )

                # Skip OOB epilogue iterations when actual M < tile_M.
                m_actual = cute.size(mC_mnl, mode=[0])
                cta_m_offset = tile_coord_mnl[0] * Int32(self.tile_shape_mnk[0])

                # kept_count cycles in lockstep with the TMA-store pipeline.
                # epi_buffer = kept_count % num_stages stays in sync with
                # producer_commit/acquire calls, regardless of how many
                # iterations are skipped.
                kept_count = 0
                for epi_n in cutlass.range_constexpr(epi_rest_n):
                    for epi_m in cutlass.range_constexpr(epi_rest_m):
                        epi_m_global_start = cta_m_offset + Int32(epi_m * epi_tile_m)
                        if epi_m_global_start < m_actual:
                            # Copy this epi-tile's slice of acc -> tRS_rD
                            # using b12x-style (mma_m, mma_n) indexing.
                            for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                                for mma_m_in_epi in cutlass.range_constexpr(
                                    MmaMPerEpiM
                                ):
                                    mma_n = epi_n * MmaNPerEpiN + mma_n_in_epi
                                    mma_m = epi_m * MmaMPerEpiM + mma_m_in_epi
                                    tRS_rD_slice = tRS_rD[
                                        (None, mma_m_in_epi, mma_n_in_epi)
                                    ]
                                    tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                    for elem_idx in cutlass.range_constexpr(
                                        cute.size(tRS_rD_slice)
                                    ):
                                        tRS_rD_slice[elem_idx] = tRS_rAcc_slice[
                                            elem_idx
                                        ]

                            tRS_rD_out = cute.make_rmem_tensor(
                                tRS_rD_layout.shape, self.c_dtype
                            )
                            # Apply the global alpha here, once, on the
                            # fp32 accumulator (hoisted out of the
                            # per-K-block dequant): one rounding instead
                            # of folding alpha into every B scale.
                            acc_vec = tRS_rD.load()
                            tRS_rD_out.store((alpha_val * acc_vec).to(self.c_dtype))

                            epi_buffer = kept_count % cute.size(tRS_sD, mode=[3])
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rD_out,
                                tRS_sD[(None, None, None, epi_buffer)],
                            )
                            cute.arch.fence_proxy("async.shared", space="cta")
                            self.epilog_sync_barrier.arrive_and_wait()

                            if warp_idx == 0:
                                cute.copy(
                                    tma_atom_c,
                                    bSG_sD[(None, epi_buffer)],
                                    bSG_gD[(None, (epi_m, epi_n))],
                                )
                                tma_store_pipeline.producer_commit()
                                tma_store_pipeline.producer_acquire()
                            kept_count = kept_count + 1

                tma_store_pipeline.producer_tail()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
        # Single DMA warp: issues all 3 TMA descriptors (A, B_packed, B_sf)
        # back-to-back into the same stage barrier per K-tile.
        elif warp_idx == self.num_mma_warps:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBpacked_g_kn = tBpacked_g[
                    (None, None, tile_coord_mnl[1], tile_coord_mnl[2])
                ]
                tBsf_g_kn = tBsf_g[(None, None, tile_coord_mnl[1], tile_coord_mnl[2])]
                mainloop_producer_state.reset_count()
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    barrier_ptr = mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    )

                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=barrier_ptr,
                        mcast_mask=a_mcast_mask,
                    )

                    tBpacked_g_k = tBpacked_g_kn[(None, mainloop_producer_state.count)]
                    tBpacked_s_pipe = tBpacked_s[(None, mainloop_producer_state.index)]
                    cute.copy(
                        tma_atom_b_packed,
                        tBpacked_g_k,
                        tBpacked_s_pipe,
                        tma_bar_ptr=barrier_ptr,
                        mcast_mask=b_mcast_mask,
                    )

                    tBsf_g_k = tBsf_g_kn[(None, mainloop_producer_state.count)]
                    tBsf_s_pipe = tBsf_s[(None, mainloop_producer_state.index)]
                    cute.copy(
                        tma_atom_b_sf,
                        tBsf_g_k,
                        tBsf_s_pipe,
                        tma_bar_ptr=barrier_ptr,
                        mcast_mask=b_mcast_mask,
                    )

                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        cute.arch.griddepcontrol_launch_dependents()
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        epi_tile: Tuple[int, int],
        c_dtype: Type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
        group_size: int = 16,
        epi_stage: int = 4,
    ) -> Tuple[int, int]:
        """Stage budget accounting for A + B (packed int32) + B_sf (uint8).

        Per (16K x 64N) packed block: 128 int32 = 512 bytes.
        Per (group_size K x tile_N N) scale block: tile_N bytes.
        """
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_bytes = c_bytes_per_stage * epi_stage

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_bytes_per_stage = cute.size(a_shape) * a_dtype.width // 8

        # B side: packed int32, 128 int32 per (16K x 64N) block.
        packed_blocks_per_tile = (tile_shape_mnk[2] // _PACK_TILE_K) * (
            tile_shape_mnk[1] // _PACK_TILE_N
        )
        b_packed_bytes_per_stage = packed_blocks_per_tile * _PACK_INTS_PER_TILE * 4

        # Scale tile: 1 byte per (K-group, N).  Small compared to B_packed.
        b_sf_bytes_per_stage = (tile_shape_mnk[2] // group_size) * tile_shape_mnk[1]

        ab_bytes_per_stage = (
            a_bytes_per_stage + b_packed_bytes_per_stage + b_sf_bytes_per_stage
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            (smem_capacity - occupancy * 1024) // occupancy
            - mbar_helpers_bytes
            - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: Tuple[int, int, int],
        epi_tile: Tuple[int, int],
        a_dtype: Type[cutlass.Numeric],
        a_layout: cute.Layout,
        b_compute_dtype: Type[cutlass.Numeric],
        b_layout_compute: cute.Layout,
        ab_stage: int,
        c_dtype: Type[cutlass.Numeric],
        c_layout: cute.Layout,
        epi_stage: int,
        group_size: int,
    ):
        """Returns (sA, sB_packed, sB_sf, b_bf16_phantom, sC) layouts.

        ``sB_packed_layout`` is a plain (non-swizzled) staged int32 layout
        ``(tile_K // 16, 2 * tile_N, ab_stage)``.

        ``sB_sf_layout`` is a plain staged uint8 layout
        ``(tile_K // group_size, tile_N, ab_stage)`` -- one byte per
        (K-group, N) cell.  Loaded via TMA once per K-tile, read from
        SMEM in the dequant inner loop (replaces the gmem-direct path).

        ``b_bf16_logical_layout`` is the phantom bf16 layout used only by
        ``partition_B`` / ``make_fragment_B`` for fragment-shape
        determination -- there is no real bf16 SMEM allocation.
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        # sB_packed: (tile_K // 16) rows x (2 * tile_N) int32 cols x stage.
        b_packed_smem_layout_staged = cute.make_ordered_layout(
            (
                tile_shape_mnk[2] // _PACK_TILE_K,
                2 * tile_shape_mnk[1],
                ab_stage,
            ),
            order=(1, 0, 2),
        )

        # sB_sf: (tile_K // group_size) rows x tile_N uint8 cols x stage.
        # Order (1, 0, 2) -> N innermost (= contiguous load lane), K next,
        # stage outermost.  Matches the gmem layout.
        b_sf_smem_layout_staged = cute.make_ordered_layout(
            (
                tile_shape_mnk[2] // group_size,
                tile_shape_mnk[1],
                ab_stage,
            ),
            order=(1, 0, 2),
        )

        # bf16 phantom layout for partition_B / make_fragment_B.
        b_bf16_logical_layout = sm90_utils.make_smem_layout_b(
            b_layout_compute,
            tile_shape_mnk,
            b_compute_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )
        return (
            a_smem_layout_staged,
            b_packed_smem_layout_staged,
            b_sf_smem_layout_staged,
            b_bf16_logical_layout,
            epi_smem_layout_staged,
        )

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: Tuple[int, int, int],
        max_active_clusters: cutlass.Constexpr,
        tile_swizzle: cutlass.Constexpr = 1,
        raster_along_m: cutlass.Constexpr = True,
    ):
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (1, 1, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size=tile_swizzle,
            raster_along_m=raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: Tuple[int, int],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )
        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @cute.jit
    def _cvt_a_bf16_to_fp16_one_k_block(
        self,
        tCrA_dst,
        tCrA_bf16_src,
        k_block: cutlass.Constexpr,
    ):
        """In-register bf16 -> fp16 cvt for one K-block of A.

        Recasts both fragments to Uint32 (each u32 packs 2 16-bit elems)
        and applies `cvt_bf16x2_to_f16x2_via_f32` per pair.  The packed
        narrowing cvt (`cvt.rn.f16x2.f32`) combines what would be two
        scalar `cvt.rn.f16.f32` instructions if we used cute's default
        `.to(Float16)` lowering.

        bf16/f16 typed inline-asm constraints don't compile on sm_100a
        (NVVM rejects them), so we keep everything in Uint32 pair
        representation via cute.recast_tensor.
        """
        bf_u32 = cute.recast_tensor(tCrA_bf16_src[None, None, k_block], Uint32)
        fp_u32 = cute.recast_tensor(tCrA_dst[None, None, k_block], Uint32)
        n_pairs = cute.size(bf_u32)
        for i in cutlass.range_constexpr(n_pairs):
            fp_u32[i] = cvt_bf16x2_to_f16x2_via_f32(Uint32(bf_u32[i]))

    @cute.jit
    def _dequant_b_to_register(
        self,
        sB_packed: cute.Tensor,
        sB_sf: cute.Tensor,
        tCrB: cute.Tensor,
        tidx: Int32,
        stage_idx: Int32,
        k_tile_idx: Int32,
        k_block_idx: cutlass.Constexpr,
    ):
        """Decode 2 int32 per thread per K-block into 16 fp16 fragment slots.

        Per the MMA partition for atom_layout (2,2,1) + *2 on N:

          tc_row     = (lane % 4) * 2          in {0, 2, 4, 6}
          tc_col     = lane // 4               in [0, 8)
          n_warp_idx = warp_idx // 2           in {0, 1}
          base_n    = n_warp_idx * 8 + tc_col  in [0, 16)

        Each thread covers:
          K = {tc_row, tc_row+1, tc_row+8, tc_row+9}
          N = {base_n, base_n+16, base_n+32, base_n+48}

        Two int32s per K-block per thread; sB_packed offsets:
          u32_0 @ sB_packed[k_block_idx, n_warp_idx * 64 + lane * 2 + 0, stage]
          u32_1 @ sB_packed[k_block_idx, n_warp_idx * 64 + lane * 2 + 1, stage]

        Byte layout inside each int32 (see ``_cute_dsl_pack_fp4_weight`` in
        ``flashinfer/gemm/gemm_bf16_fp4_cute_dsl.py``):
          u32_0:
            byte 0: K=tc_row,    tc_row+1 at N=base_n        -> (mma_i=0,1, nn=0)
            byte 1: K=tc_row+8,  tc_row+9 at N=base_n        -> (mma_i=2,3, nn=0)
            byte 2: K=tc_row,    tc_row+1 at N=base_n+16     -> (mma_i=0,1, nn=1)
            byte 3: K=tc_row+8,  tc_row+9 at N=base_n+16     -> (mma_i=2,3, nn=1)
          u32_1 mirrors with N=base_n+32 and N=base_n+48 (nn=2, nn=3).
        """
        lane = tidx % Int32(32)
        warp = tidx // Int32(32)
        # n_warp_idx maps warp -> N-stripe.  With (2,2,1) the 4 warps are
        # arranged as 2 M-warps * 2 N-warps, so n_warp_idx = warp // 2.
        # With (1,2,1) there are 2 warps total (no M-warp dim), so each
        # warp is its own N-stripe: n_warp_idx = warp.  In general
        # n_warp_idx = warp // atom_layout[0].
        n_warp_idx = warp // Int32(self.atom_layout[0])
        tc_col = lane // Int32(4)
        base_n_in_tile = n_warp_idx * Int32(8) + tc_col

        # tile_N is built from tile_N // 64 independent packed 64-N blocks.
        # The hand-coded byte->fragment mapping below covers exactly one such
        # 64-N block (4 nn slots); for tile_N=128 we loop it over both blocks.
        # Block n_blk lives at sB_packed int32 columns [n_blk*128, +128), sB_sf
        # N-rows [n_blk*64, +64), and writes tCrB nn in [n_blk*4, +4).  For
        # tile_N=64 this loops once.
        num_n_blocks = cutlass.const_expr(self.tile_shape_mnk[1] // 64)

        # _write_hmul2 captures tCrB / k_block_idx; nn and mma-row passed per
        # write.  fp16 MMA writes fp16 directly; bf16 MMA goes via fp32 (no
        # packed f16x2 -> bf16x2 cvt on sm_100a).
        if cutlass.const_expr(self.use_fp16_mma == 1):

            def _write_hmul2(h2, scale_h2, mma_i_low, nn):
                scaled_h2 = half2_mul(h2, scale_h2)
                f_lo, f_hi = f16x2_unpack(scaled_h2)
                tCrB[mma_i_low, nn, k_block_idx] = f_lo
                tCrB[mma_i_low + 1, nn, k_block_idx] = f_hi
        else:

            def _write_hmul2(h2, scale_h2, mma_i_low, nn):
                scaled_h2 = half2_mul(h2, scale_h2)
                f_lo, f_hi = f16x2_to_f32x2(scaled_h2)
                tCrB[mma_i_low, nn, k_block_idx] = f_lo.to(self.b_compute_dtype)
                tCrB[mma_i_low + 1, nn, k_block_idx] = f_hi.to(self.b_compute_dtype)

        for n_blk in cutlass.range_constexpr(num_n_blocks):
            n_col_off = Int32(n_blk * 128)  # int32 column offset of this 64-N block
            n_sf_off = Int32(n_blk * 64)  # SF N-row offset of this 64-N block
            nn0 = n_blk * 4  # base tCrB N-fragment slot for this block

            # Single ld.shared.v2.u32 (8-byte load): u32_pos_base is always even
            # (= n_blk*128 + n_warp*64 + lane*2), so the offset is 8-byte aligned.
            u32_pos_base = n_col_off + n_warp_idx * Int32(64) + lane * Int32(2)
            smem_addr_b = get_smem_ptr_as_int32(
                sB_packed, sB_packed.layout((k_block_idx, u32_pos_base, stage_idx))
            )
            u32_0, u32_1 = ld_shared_v2_u32(smem_addr_b)

            # Per-group scale loads (S0E5M3).  4 distinct N positions per block.
            sf_n = base_n_in_tile + n_sf_off
            sf_byte_0 = Uint32(sB_sf[k_block_idx, sf_n + Int32(0), stage_idx])
            sf_byte_1 = Uint32(sB_sf[k_block_idx, sf_n + Int32(16), stage_idx])
            sf_byte_2 = Uint32(sB_sf[k_block_idx, sf_n + Int32(32), stage_idx])
            sf_byte_3 = Uint32(sB_sf[k_block_idx, sf_n + Int32(48), stage_idx])

            h0_a, h0_b, h0_c, h0_d = fp4_decode_4bytes(u32_0)
            h1_a, h1_b, h1_c, h1_d = fp4_decode_4bytes(u32_1)

            # S0E5M3 scale -> f16x2 broadcast in one mul.lo.u32.  alpha is
            # applied once in the epilogue, so the scale never needs an f32 form.
            sc_n0 = cvt_s0e5m3_to_f16x2_broadcast(sf_byte_0)
            sc_n1 = cvt_s0e5m3_to_f16x2_broadcast(sf_byte_1)
            sc_n2 = cvt_s0e5m3_to_f16x2_broadcast(sf_byte_2)
            sc_n3 = cvt_s0e5m3_to_f16x2_broadcast(sf_byte_3)
            _write_hmul2(h0_a, sc_n0, 0, nn0 + 0)
            _write_hmul2(h0_b, sc_n0, 2, nn0 + 0)
            _write_hmul2(h0_c, sc_n1, 0, nn0 + 1)
            _write_hmul2(h0_d, sc_n1, 2, nn0 + 1)
            _write_hmul2(h1_a, sc_n2, 0, nn0 + 2)
            _write_hmul2(h1_b, sc_n2, 2, nn0 + 2)
            _write_hmul2(h1_c, sc_n3, 0, nn0 + 3)
            _write_hmul2(h1_d, sc_n3, 2, nn0 + 3)

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB_packed: cute.Tensor,
        mB_sf: cute.Tensor,
        mC: cute.Tensor,
        mAlpha: cute.Tensor,
        l: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
    ):
        """bf16 x fp4 wrapper for the FlashInfer compile interface.

        Args:
            mA:        (m, k) input tensor A, bf16 or fp16.
            mB_packed: (k // 16, n * 2) int32 -- packed FP4.
            mB_sf:     (k // 16, n) uint8 -- FP8-E4M3 per-group scales.
            mC:        (m, n) output tensor C, bf16 or fp16.
            mAlpha:    (1,) fp32 global scale.
            l: batch dimension (Constexpr); typically 1.
            max_active_clusters: Constexpr from get_max_active_clusters(1).
            current_stream: CUDA stream (TVM-FFI fake stream).
        """
        m = cute.size(mA, mode=[0])
        k = cute.size(mA, mode=[1])
        n = cute.size(mC, mode=[1])
        k_tiles = k // _PACK_TILE_K
        n_packed = 2 * n
        k_sf_groups = k // self.GROUP_SIZE

        a_tensor = cute.make_tensor(
            mA.iterator,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        b_packed_tensor = cute.make_tensor(
            mB_packed.iterator,
            layout=cute.make_ordered_layout((k_tiles, n_packed, l), order=(1, 0, 2)),
        )
        b_sf_tensor = cute.make_tensor(
            mB_sf.iterator,
            layout=cute.make_ordered_layout((k_sf_groups, n, l), order=(1, 0, 2)),
        )
        c_tensor = cute.make_tensor(
            mC.iterator,
            layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
        )

        self(
            a_tensor,
            b_packed_tensor,
            b_sf_tensor,
            c_tensor,
            mAlpha,
            max_active_clusters,
            current_stream,
        )
