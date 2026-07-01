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
"""
CUTE DSL translation of the TGV GEMM kernel (tgv_gemm.cuh).

This is a low-latency Blackwell GEMM kernel: C = A * B
- A (M, K, L) with K contiguous (UmmaMajor::K)
- B (N, K, L) with K contiguous
- C (M, N, L) with M contiguous

Features:
- TMA loads for A and B matrices (GMEM -> SMEM)
- tcgen05.mma for matrix multiply-accumulate in TMEM
- Multi-stage pipeline for overlapping TMA and MMA
- Non-TMA-store epilogue (TMEM -> RMEM -> GMEM with direct store)
- 1 SM mode (no 2CTA instructions)
- 1x1 cluster (no multicast)
- PDL (Programmatic Dependent Launch) support via griddepcontrol

NVFP4 blockscaled variant of the TGV GEMM kernel.
Default config: CTA_M=128, CTA_N=8, CTA_K=256, DMA_Stage=8
  TypeA=Float4E2M1FN, TypeB=Float4E2M1FN, TypeC=Float16, AccType=float
  SF=Float8E4M3FN, sf_vec_size=16
  UmmaMajorA=Major::K, UmmaMajorB=Major::K

C++ warp assignment (256 threads, 8 warps):
  Warp 0: DMA_A - loads A tiles via TMA
  Warp 1: DMA_B - loads B tiles via TMA
  Warp 2: MMA - performs tcgen05.mma
  Warp 3: unused
  Warps 4-7: EPILOG - TMEM->RMEM->GMEM store
"""

from typing import NamedTuple, Tuple, Type
from dataclasses import dataclass
from functools import lru_cache
import cuda.bindings.driver as cuda
import argparse
import os
import sys
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_ptr
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.arch.smem import map_dsmem_ptr
from cutlass.cute.nvgpu.cpasync import CopyDsmemStoreOp
from cutlass.pipeline import (
    PipelineAsync, PipelineAsyncUmma, PipelineUmmaAsync,
    CooperativeGroup, Agent,
    pipeline_init_arrive, pipeline_init_wait,
    make_pipeline_state, PipelineUserType,
)


# ============================================================
# DSMEM primitives for cluster split-k reduction
# ============================================================


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes. Matches C++ WorkTileInfo struct.
    For non-persistent static scheduler, CTA id is the work tile info."""
    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32   # kblock range [K_idx_start, K_idx_end)
    K_idx_end: cutlass.Int32

class TgvGemmKernel:
    """
    Low-latency Blackwell GEMM kernel, literal translation from tgv_gemm.cuh.

    Uses raw mbarriers for synchronization with 8-warp specialization:
    warp 0 = DMA_A, warp 1 = DMA_B, warp 2 = MMA, warps 4-7 = EPILOG.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 128,
        cta_n: int = 8,
        cta_k: int = 256,
        num_ab_stage: int = 8,
        num_sfb_tmem_stage: int = 4,
        sf_vec_size: int = 16,
        use_pdl: bool = False,
        pdl_count: int = -1,
        split_k: int = 1,
        sfb_tmem_store: bool = False,
    ):
        self.acc_dtype = acc_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.num_sfb_tmem_stage = num_sfb_tmem_stage
        self.sf_vec_size = sf_vec_size
        self.use_pdl = use_pdl
        self.pdl_count = pdl_count
        self.split_k = split_k
        self.sfb_tmem_store = sfb_tmem_store

        # DSMEM mailbox sizing for cluster split-k distributed reduction
        # each epilog thread owns (cta_m * cta_n / 128) float32 accumulator elements
        self._mailbox_elems_per_thread =  (cta_m * cta_n) // 128
        # distributed reduction: each CTA reduces 1/split_k of the output tile
        # shard size in elements per thread (must divide evenly)
        self._shard_elems_per_thread = self._mailbox_elems_per_thread // max(split_k, 1)
        if split_k > 1 and self._mailbox_elems_per_thread % split_k != 0:
            raise ValueError(
                f"elems_per_thread={self._mailbox_elems_per_thread} must be divisible by split_k={split_k}"
            )
        # per-CTA mailbox: receives (split_k-1) shards from peer CTAs
        self._mailbox_total_elems = max(split_k - 1, 0) * 128 * self._shard_elems_per_thread
        # TX bytes: each peer sends shard_elems_per_thread * 128 * 4B to this CTA
        self._mailbox_tx_per_sender = 128 * self._shard_elems_per_thread * 4
        self._mailbox_tx_total = max(split_k - 1, 0) * self._mailbox_tx_per_sender

        # Fixed configuration matching C++ kernel
        self.threads_per_cta = 384     # 12 warps (3 active + 4 SFB + 1 unused + 4 epilog)
        self.use_2cta_instrs = False   # 1 SM mode
        self.cluster_shape_mn = (1, 1) # No multicast, 1x1 cluster
        self.cta_group = tcgen05.CtaGroup.ONE

    def _setup_attributes(self):
        """Set up derived config. Corresponds to C++ gemm_host() setup section."""
        mma_tiler_mn = (self.cta_m, self.cta_n)

        # Blockscaled MMA atom for NVFP4 (Float4E2M1FN + Float8E4M3FN scale factors)
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size,
            self.cta_group, mma_tiler_mn,
        )
        assert self.cta_group == tcgen05.CtaGroup.ONE

        # Derive mma_inst_tile_k = NumMma_K = CTA_K / Mma_K
        # For NVFP4: Mma_K = 64, so CTA_K=256 -> mma_inst_tile_k = 4
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        # Shared SFB TMEM layout constants (used by both sfb_warp and mma_warp)
        self._sf_atom_mn = 32
        self._sf_per_mma_k = mma_inst_shape_k // self.sf_vec_size
        self._num_n_atoms = 2 # padded to 2 cols (tcgen05.st path)
        self._sfb_lane_stride = 1 << 18
        self._num_sfa_tmem_cols = (self.cta_m // self._sf_atom_mn) * self.mma_inst_tile_k
        if self.sfb_tmem_store:
            self._sfb_tmem_cols_per_stage = self.mma_inst_tile_k * self._num_n_atoms
        else:
            self._sfb_tmem_cols_per_stage = self.mma_inst_tile_k * 4  # standard padded

        # assert TMEM column budget: acc + SFA + SFB(staged) <= 256 (half of TMEM)
        total_tmem_cols = (
            self.cta_n                                                    # accumulator
            + self._num_sfa_tmem_cols                                     # SFA
            + self._sfb_tmem_cols_per_stage * self.num_sfb_tmem_stage     # SFB (staged)
        )
        assert total_tmem_cols <= 256, (
            f"TMEM column budget exceeded: {total_tmem_cols} > 256 "
            f"(acc={self.cta_n}, sfa={self._num_sfa_tmem_cols}, "
            f"sfb={self._sfb_tmem_cols_per_stage}*{self.num_sfb_tmem_stage}="
            f"{self._sfb_tmem_cols_per_stage * self.num_sfb_tmem_stage})"
        )

        # Maps a CTA's linear rank within its cluster to 4D (V, M, N, K) coordinates.
        # For 1SM + 1x1xS cluster: K dimension = split_k, CTAs along K share M/N tile.
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, self.split_k)),
            (tiled_mma.thr_id.shape,),
        )

        # Create SMEM layouts for A and B with swizzle
        # sA_layout: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, (self.cta_m, self.cta_n, self.cta_k), self.a_dtype, self.num_ab_stage
        )
        # sB_layout: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, (self.cta_m, self.cta_n, self.cta_k), self.b_dtype, self.num_ab_stage
        )

        # SFA SMEM layout: ((Atom_M, Atom_K), MMA_M, MMA_K, STAGE)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, (self.cta_m, self.cta_n, self.cta_k),
            self.sf_vec_size, self.num_ab_stage,
        )
        # SFB SMEM layouts: compute both flat (tcgen05.st path) and padded (tcgen05.cp path), select based on sfb_tmem_store
        sfb_n = self.cta_n
        sfb_k = self.cta_k // self.sf_vec_size
        self._sfb_smem_layout_flat = cute.make_layout(
            (sfb_n, sfb_k, self.num_ab_stage),
            stride=(sfb_k, 1, sfb_n * sfb_k),
        )
        # padded layout for tcgen05.cp TMA and make_tmem_layout_sfb
        self._mma_tiler_sfb = (self.cta_m, cute.round_up(self.cta_n, 128), self.cta_k)
        self._tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size,
            tcgen05.CtaGroup.ONE, (self.cta_m, cute.round_up(self.cta_n, 128)),
        )
        self._sfb_smem_layout_padded = blockscaled_utils.make_smem_layout_sfb(
            self._tiled_mma_sfb, self._mma_tiler_sfb,
            self.sf_vec_size, self.num_ab_stage,
        )
        if self.sfb_tmem_store:
            self.sfb_smem_layout_staged = self._sfb_smem_layout_flat
        else:
            self.sfb_smem_layout_staged = self._sfb_smem_layout_padded

        # Accumulator shape for TMEM allocation size calculation
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    @cute.jit
    def _call_tmem_store(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_mnkl: Tuple,
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch (tcgen05.st SFB path)."""
        m, n, k, l = problem_mnkl
        a = cute.make_tensor(a_ptr, cute.make_ordered_layout(
            (cute.assume(m, 32), k, l), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, cute.make_ordered_layout(
            (cute.assume(n, 32), k, l), order=(1, 0, 2)))
        c = cute.make_tensor(c_ptr, cute.make_ordered_layout(
            (cute.assume(m, 32), n, l), order=(0, 1, 2)))
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.c_dtype = c_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        sfa = cute.make_tensor(sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size))
        sfb = cute.make_tensor(sfb_ptr, cute.make_ordered_layout(
            (n, cute.assume(k // self.sf_vec_size, 16), l), order=(1, 0, 2)))
        self._setup_attributes()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size,
            self.cta_group, (self.cta_m, self.cta_n),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, (self.cta_m, self.cta_n, self.cta_k), tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, (self.cta_m, self.cta_n, self.cta_k), tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, sfa, sfa_smem_layout, (self.cta_m, self.cta_n, self.cta_k), tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )
        sfb_smem_layout = cute.slice_(self._sfb_smem_layout_flat, (None, None, 0))
        sfb_k = self.cta_k // self.sf_vec_size
        sfb_tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
            sfb_tma_op, sfb, sfb_smem_layout, (self.cta_n, sfb_k),
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.tma_bytes_a = (a_copy_size + sfa_copy_size) * atom_thr_size
        self.tma_bytes_b = (b_copy_size + sfb_copy_size) * atom_thr_size
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_m),
            cute.ceil_div(c.layout.shape[1], self.cta_n),
            c.layout.shape[2] * self.split_k,
        )
        self.kernel_tmem_store(
            tiled_mma, self._tiled_mma_sfb,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa,
            tma_atom_sfb, tma_tensor_sfb,
            c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self._sfb_smem_layout_flat,
            self._sfb_smem_layout_padded,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, self.split_k),
            stream=stream,
            use_pdl=self.use_pdl,
        )
        return

    @cute.jit
    def _call_s2t_copy(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_mnkl: Tuple,
        stream: cuda.CUstream,
    ):
        """Host-side setup and kernel launch (SMEM-to-TMEM copy SFB path)."""
        m, n, k, l = problem_mnkl
        a = cute.make_tensor(a_ptr, cute.make_ordered_layout(
            (cute.assume(m, 32), k, l), order=(1, 0, 2)))
        b = cute.make_tensor(b_ptr, cute.make_ordered_layout(
            (cute.assume(n, 32), k, l), order=(1, 0, 2)))
        c = cute.make_tensor(c_ptr, cute.make_ordered_layout(
            (cute.assume(m, 32), n, l), order=(0, 1, 2)))
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.c_dtype = c_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        sfa = cute.make_tensor(sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size))
        n_padded_sfb = cute.round_up(n, 128)
        b_shape_for_sfb = cute.make_ordered_layout(
            (n_padded_sfb, k, l), order=(1, 0, 2)).shape
        sfb = cute.make_tensor(sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(
            b_shape_for_sfb, self.sf_vec_size))
        self._setup_attributes()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size,
            self.cta_group, (self.cta_m, self.cta_n),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, (self.cta_m, self.cta_n, self.cta_k), tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, (self.cta_m, self.cta_n, self.cta_k), tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, sfa, sfa_smem_layout, (self.cta_m, self.cta_n, self.cta_k), tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, self._tiled_mma_sfb.thr_id)
        sfb_smem_layout = cute.slice_(self._sfb_smem_layout_padded, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb, sfb_smem_layout, self._mma_tiler_sfb, self._tiled_mma_sfb,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.tma_bytes_a = (a_copy_size + sfa_copy_size) * atom_thr_size
        self.tma_bytes_b = (b_copy_size + sfb_copy_size) * atom_thr_size
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_m),
            cute.ceil_div(c.layout.shape[1], self.cta_n),
            c.layout.shape[2] * self.split_k,
        )
        self.kernel(
            tiled_mma, self._tiled_mma_sfb,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa,
            tma_atom_sfb, tma_tensor_sfb,
            c,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self._sfb_smem_layout_flat,
            self._sfb_smem_layout_padded,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, self.split_k),
            stream=stream,
            use_pdl=self.use_pdl,
        )
        return

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_mnkl: Tuple,
        stream: cuda.CUstream,
    ):
        """Host-side setup — dispatches to tcgen05.st or SMEM-to-TMEM copy path."""
        if self.sfb_tmem_store:  # constexpr: Python bool, only taken branch is traced
            self._call_tmem_store(a_ptr, sfa_ptr, b_ptr, sfb_ptr, c_ptr, problem_mnkl, stream)
        else:
            self._call_s2t_copy(a_ptr, sfa_ptr, b_ptr, sfb_ptr, c_ptr, problem_mnkl, stream)

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,              # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor for A
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,              # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor for B
        tma_atom_sfa: cute.CopyAtom,
        sfa: cute.Tensor,                 # SFA TMA coordinate tensor
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,            # SFB TMA coordinate tensor
        mC_mnl: cute.Tensor,              # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        b_smem_layout_staged: cute.ComposedLayout,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_flat_staged: cute.Layout,    # flat 3D layout for tcgen05.st SFB
        sfb_smem_layout_padded_staged: cute.Layout,  # padded 4D layout for tcgen05.cp SFB + TMEM
    ):
        """
        GPU device kernel. 1:1 translation of C++ gemm_device().

        Slim kernel: only SMEM alloc + barrier init + warp dispatch.
        All tensor partitioning is done inside warp functions.

        256 threads, 8 warps:
          Warp 0: DMA_A  - loads A tiles via TMA
          Warp 1: DMA_B  - loads B tiles via TMA
          Warp 2: MMA    - performs tcgen05.mma
          Warp 3: unused
          Warps 4-7: EPILOG - TMEM->RMEM->GMEM store
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        # Prefetch TMA descriptors (warp 0 only)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        # ============================================================
        # cluster split-k: compute split_rank and l_idx from cluster topology
        # ============================================================
        _, _, split_rank = cute.arch.block_in_cluster_idx()
        _, _, l_idx = cute.arch.cluster_idx()

        # ============================================================
        # WorkTileInfo with per-split K-range
        # ============================================================
        total_k_tiles = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_k)
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        k_tiles_per_split = cute.ceil_div(total_k_tiles, self.split_k)
        k_start = split_rank * k_tiles_per_split
        if k_start > total_k_tiles:
            k_start = total_k_tiles
        k_end = k_start + k_tiles_per_split
        if k_end > total_k_tiles:
            k_end = total_k_tiles
        work_tile_info = WorkTileInfo(
            M_idx=bidx // cute.size(tiled_mma.thr_id.shape),
            N_idx=bidy,
            L_idx=l_idx,
            K_idx_start=k_start,
            K_idx_end=k_end,
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        # block_in_cluster_coord_vmnk: this CTA's (V, M, N, K) coordinate within its cluster.
        # Used by tma_partition to set up TMA multicast masks:
        #   [2] (N) → A's tma_partition (A is multicast along N)
        #   [1] (M) → B's tma_partition (B is multicast along M)
        # For 1x1 cluster: always (0, 0, 0, 0), trivially no multicast.
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # ============================================================
        # SMEM allocation (SharedStorage struct for barriers + TMEM ptr)
        # ============================================================
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage

        # Shared storage struct (matches C++ SharedStorage)
        # There are 2 kinds of barriers used here:
        #   1. Transaction barriers (tma_mma_full): 64-bit in SMEM, support transaction byte
        #      counting for TMA completion tracking. set_barrier_transaction_bytes arrives and
        #      sets the expected TX count; TMA increments TX bytes as data arrives in SMEM.
        #      When both arrival count and TX count are met, the barrier phase flips.
        #   2. Cluster barriers (all others): 64-bit in SMEM, simpler arrive/wait semantics.
        #      Used for thread-to-thread synchronization within/across warps.
        MailboxElems = self._mailbox_total_elems  # (split_k-1) * 128 * elems_per_thread

        @cute.struct
        class SharedStorage:
            # AB pipeline barriers: full[0..DMA_Stage-1] + empty[DMA_Stage..2*DMA_Stage-1]
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, DMA_Stage * 2]
            # SFB pipeline barriers (unused in tcgen05.cp path, but allocated for consistent layout)
            sfb_pipeline_bars: cute.struct.MemRange[cutlass.Int64, SFB_Stage * 2]
            # MMA→epilog pipeline: full[0] + empty[1]
            mma_epilog_bars: cute.struct.MemRange[cutlass.Int64, 2]
            # raw mbarriers (multi-phase tmem alloc, split-k dsmem)
            tmem_allocation_result_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            dsmem_mailbox_barrier: cutlass.Int64
            dsmem_mailbox: cute.struct.MemRange[cutlass.Float32, MailboxElems]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ============================================================
        # Pipeline creation (replaces manual mbarrier_init loops)
        # defer_sync=True: pipeline init does NOT insert __syncthreads;
        # we fence + sync manually below after all inits are done.
        #
        # Transaction barriers (used for AB full) track both arrival count
        # and TX byte count; the phase flips when both are satisfied.
        # ============================================================

        # AB pipeline: DMA_A + DMA_B → MMA (2 producer arrivals, 1 consumer)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=DMA_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage

        # MMA→epilog pipeline (1-stage): MMA tcgen05.commit → epilog wait
        mma_epilog_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.mma_epilog_bars.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        mma_epilog_full_bar = mma_epilog_pipeline.sync_object_full.barrier_storage

        # raw barrier pointers
        tmem_alloc_result_bar = storage.tmem_allocation_result_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        mailbox_mbar = storage.dsmem_mailbox_barrier.ptr
        dsmem_mailbox_ptr = storage.dsmem_mailbox.data_ptr()
        # SFB barrier pointers (uninitialized — unused in tcgen05.cp path)
        # layout: full[0..SFB_Stage-1], empty[SFB_Stage..2*SFB_Stage-1]
        sfb_full_bar = storage.sfb_pipeline_bars.data_ptr()
        sfb_empty_bar = sfb_full_bar + SFB_Stage

        # ---- raw mbarrier init (tmem_alloc + dsmem only) ----
        # pipeline API handles AB/MMA-epilog mbarrier_init; only raw barriers remain
        if warp_idx == 0:
            with cute.arch.elect_one():
                # 2-phase barrier for tmem lifetime:
                #   phase 0->1: mma alloc done, tmem_base_ptr now valid for epi warp to read
                #   phase 1->0: epilog warp is done reading tmem, mma can dealloc
                # 32 (MMA) + 128 (EPILOG warps 8-11) = 160
                cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128)
                if self.split_k > 1:
                    cute.arch.mbarrier_init(mailbox_mbar, 1)

        # barrier init needs to be visible to all warps before proceeding
        cluster_layout = cute.make_layout((*self.cluster_shape_mn, self.split_k)) if self.split_k > 1 else None
        pipeline_init_arrive(cluster_shape_mn=cluster_layout, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_layout)

        # ============================================================
        # SMEM Tensor Allocation
        # C++: SharedStorage contains alignas(128) ArrayEngine for A and B
        # ============================================================
        # sA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) with swizzle
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # sB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) with swizzle
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # sSFA: scale factor SMEM (no swizzle)
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        # allocate SMEM for SFB using both layouts (flat for tcgen05.st, padded for tcgen05.cp)
        # only the active one is actually used at runtime; both are always allocated
        # so the kernel signature/types are consistent regardless of sfb_tmem_store
        sSFB_flat = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_flat_staged,
            byte_alignment=128,
        )
        sSFB_padded = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_padded_staged,
            byte_alignment=128,
        )

        # ============================================================
        # Warp dispatch (C++: gemm_device warp_idx dispatch)
        # Each warp specializes in a different role:
        #   Warp 0 (threads 0-31):      DMA_A — loads A tiles via TMA
        #   Warp 1 (threads 32-63):     DMA_B — loads B tiles via TMA
        #   Warp 2 (threads 64-95):     MMA   — performs tcgen05.mma
        #   Warps 3-6 (threads 96-223): SFB   — lds + tcgen05.st to TMEM (sfb_tmem_store only)
        #   Warp 7 (threads 224-255):   unused
        #   Warps 8-11 (threads 256-383): EPILOG — TMEM->RMEM->GMEM store
        # ============================================================
        if warp_idx == 0:
            self.dma_a_warp(
                ab_full_bar, ab_empty_bar,
                tma_atom_a, mA_mkl, sA,
                tma_atom_sfa, sfa, sSFA,
                tiled_mma, cluster_layout_vmnk, block_in_cluster_coord_vmnk,
                mma_tile_coord_v, work_tile_info, k_tile_count,
            )
        elif warp_idx == 1:
            if self.sfb_tmem_store:
                self.dma_b_warp_tmem_store(
                    ab_full_bar, ab_empty_bar,
                    tma_atom_b, mB_nkl, sB,
                    tma_atom_sfb, mSFB_nkl, sSFB_flat,
                    tiled_mma, cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk, mma_tile_coord_v,
                    work_tile_info, k_tile_count,
                )
            else:
                self.dma_b_warp_s2t_copy(
                    ab_full_bar, ab_empty_bar,
                    tma_atom_b, mB_nkl, sB,
                    tma_atom_sfb, mSFB_nkl, sSFB_padded,
                    tiled_mma, tiled_mma_sfb, cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk, mma_tile_coord_v,
                    work_tile_info, k_tile_count,
                )
        elif warp_idx == 2:
            self.mma_warp(
                ab_full_bar, ab_empty_bar,
                mma_epilog_full_bar, tmem_alloc_result_bar,
                sfb_full_bar, sfb_empty_bar,
                tiled_mma, sA, sB, sSFA, sSFB_padded,
                sfa_smem_layout_staged, sfb_smem_layout_padded_staged,
                tmem_base_smem_ptr, k_tile_count,
            )
        elif warp_idx >= 8:
            epi_tid = tidx - 256
            self.epilog_warp(
                mma_epilog_full_bar, tmem_alloc_result_bar,
                tiled_mma, mC_mnl,
                tmem_base_smem_ptr,
                mailbox_mbar, dsmem_mailbox_ptr,
                epi_tid, mma_tile_coord_v, work_tile_info,
                split_rank,
            )

        # Final sync — C++: __syncthreads()
        cute.arch.barrier()
        return

    @cute.kernel
    def kernel_tmem_store(
        self,
        tiled_mma: cute.TiledMma, tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom, sfa: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor, cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_flat_staged: cute.Layout,
        sfb_smem_layout_padded_staged: cute.Layout,
    ):
        """tcgen05.st SFB variant of the kernel — calls dma_b_warp_tmem_store without branching."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
        _, _, split_rank = cute.arch.block_in_cluster_idx()
        _, _, l_idx = cute.arch.cluster_idx()
        total_k_tiles = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_k)
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        k_tiles_per_split = cute.ceil_div(total_k_tiles, self.split_k)
        k_start = split_rank * k_tiles_per_split
        if k_start > total_k_tiles:
            k_start = total_k_tiles
        k_end = k_start + k_tiles_per_split
        if k_end > total_k_tiles:
            k_end = total_k_tiles
        work_tile_info = WorkTileInfo(
            M_idx=bidx // cute.size(tiled_mma.thr_id.shape),
            N_idx=bidy, L_idx=l_idx,
            K_idx_start=k_start, K_idx_end=k_end)
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage
        MailboxElems = self._mailbox_total_elems
        @cute.struct
        class SharedStorage:
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, DMA_Stage * 2]
            sfb_pipeline_bars: cute.struct.MemRange[cutlass.Int64, SFB_Stage * 2]
            mma_epilog_bars: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_allocation_result_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            dsmem_mailbox_barrier: cutlass.Int64
            dsmem_mailbox: cute.struct.MemRange[cutlass.Float32, MailboxElems]
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # ---- pipeline creation (defer_sync=True: we fence + sync manually below) ----

        # AB pipeline: DMA_A + DMA_B → MMA (2 producer arrivals, 1 consumer)
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=DMA_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage

        # SFB pipeline: 4 SFB warps (elect_one each) → MMA (tcgen05.commit)
        sfb_pipeline = PipelineAsyncUmma.create(
            barrier_storage=storage.sfb_pipeline_bars.data_ptr(),
            num_stages=SFB_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 4),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            defer_sync=True,
        )
        sfb_full_bar = sfb_pipeline.sync_object_full.barrier_storage
        sfb_empty_bar = sfb_pipeline.sync_object_empty.barrier_storage

        # MMA→epilog pipeline (1-stage): MMA tcgen05.commit → epilog wait
        mma_epilog_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.mma_epilog_bars.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        mma_epilog_full_bar = mma_epilog_pipeline.sync_object_full.barrier_storage

        tmem_alloc_result_bar = storage.tmem_allocation_result_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        mailbox_mbar = storage.dsmem_mailbox_barrier.ptr
        dsmem_mailbox_ptr = storage.dsmem_mailbox.data_ptr()

        # raw mbarrier init (pipeline API handles AB/SFB/MMA-epilog)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128 + 128)
                if self.split_k > 1:
                    cute.arch.mbarrier_init(mailbox_mbar, 1)

        cluster_layout = cute.make_layout((*self.cluster_shape_mn, self.split_k)) if self.split_k > 1 else None
        pipeline_init_arrive(cluster_shape_mn=cluster_layout, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_layout)

        sA = smem.allocate_tensor(element_type=self.a_dtype, layout=a_smem_layout_staged.outer,
            byte_alignment=128, swizzle=a_smem_layout_staged.inner)
        sB = smem.allocate_tensor(element_type=self.b_dtype, layout=b_smem_layout_staged.outer,
            byte_alignment=128, swizzle=b_smem_layout_staged.inner)
        sSFA = smem.allocate_tensor(element_type=self.sf_dtype, layout=sfa_smem_layout_staged, byte_alignment=128)
        sSFB_flat = smem.allocate_tensor(element_type=self.sf_dtype, layout=sfb_smem_layout_flat_staged, byte_alignment=128)
        sSFB_padded = smem.allocate_tensor(element_type=self.sf_dtype, layout=sfb_smem_layout_padded_staged, byte_alignment=128)

        if warp_idx == 0:
            self.dma_a_warp(
                ab_full_bar, ab_empty_bar,
                tma_atom_a, mA_mkl, sA, tma_atom_sfa, sfa, sSFA,
                tiled_mma, cluster_layout_vmnk, block_in_cluster_coord_vmnk,
                mma_tile_coord_v, work_tile_info, k_tile_count)
        elif warp_idx == 1:
            self.dma_b_warp_tmem_store(
                ab_full_bar, ab_empty_bar,
                tma_atom_b, mB_nkl, sB, tma_atom_sfb, mSFB_nkl, sSFB_flat,
                tiled_mma, cluster_layout_vmnk,
                block_in_cluster_coord_vmnk, mma_tile_coord_v,
                work_tile_info, k_tile_count)
        elif warp_idx == 2:
            self.mma_warp(
                ab_full_bar, ab_empty_bar,
                mma_epilog_full_bar, tmem_alloc_result_bar,
                sfb_full_bar, sfb_empty_bar,
                tiled_mma, sA, sB, sSFA, sSFB_padded,
                sfa_smem_layout_staged, sfb_smem_layout_padded_staged,
                tmem_base_smem_ptr, k_tile_count)
        elif warp_idx >= 3 and warp_idx <= 6:
            subpartition_idx = warp_idx - 3
            self.sfb_warp(
                ab_full_bar, sfb_full_bar, sfb_empty_bar,
                tmem_alloc_result_bar, tmem_base_smem_ptr,
                sSFB_flat, k_tile_count, subpartition_idx)
        elif warp_idx >= 8:
            epi_tid = tidx - 256
            self.epilog_warp(
                mma_epilog_full_bar, tmem_alloc_result_bar,
                tiled_mma, mC_mnl, tmem_base_smem_ptr,
                mailbox_mbar, dsmem_mailbox_ptr,
                epi_tid, mma_tile_coord_v, work_tile_info, split_rank)
        cute.arch.barrier()
        return

    @cute.jit
    def dma_a_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,              # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor
        sA: cute.Tensor,                  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        sSFA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: Tuple,             # WorkTileInfo: (M_idx, N_idx, L_idx, K_idx_start, K_idx_end)
        k_tile_count: cutlass.Int32,
    ):
        """DMA_A warp: loads A tiles via TMA. C++ DMA_A_warp() lines 204-318."""
        DMA_Stage = self.num_ab_stage

        # ---- Self-contained tensor partitioning (matches C++) ----
        # Partitioned tensors use tXgY naming convention:
        #   tX = partitioning pattern applied to tensor gY
        #   tC = tensor partitioned into the MMA shape, i.e. ((Mma_M, Mma_K), ...)
        #   tA = tensor partitioned into the TMA shape, i.e. (TMA, ...)
        #   g = gmem, s = smem, t = tmem, r = rmem

        # Tile mA_mkl (Gemm_M, Gemm_K, Gemm_L) into CTA-level tiles
        # C++: local_tile(mA, make_shape(CTA_M, CTA_K), make_coord(M_idx, _, L_idx))
        # gA_mkl: (CTA_M, CTA_K, Tiles_M, Tiles_K, Gemm_L) — all tiles
        gA_mkl = cute.local_tile(
            mA_mkl, (self.cta_m, self.cta_k), (None, None, None)
        )
        # Partition gmem tensor into the MMA shape
        # In 1SM mode, tiled_mma has 1 "thread", so get_slice(0) gives the only partition
        # C++: ThrMMA cta_mma = tiled_mma.get_slice(0)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, Tiles_M, Tiles_K, Gemm_L)
        tCgA = thr_mma.partition_A(gA_mkl)

        # ---- TMA partition for cluster-level multicast ----
        # tma_partition sets up TMA descriptors that can multicast data to multiple CTAs
        # in the same cluster that share the same tile. It takes 3 key arguments:
        #   1. This CTA's coordinate within the multicast group
        #   2. The multicast group layout (how many CTAs participate)
        #   3. The SMEM/GMEM tensors to partition
        #
        # For A: all CTAs along the N dimension of the cluster share the same A tile
        # (same M, same K, different N). So the multicast group is the N-slice of the
        # cluster layout. For a 2x3 cluster, 3 CTAs along N would share each A tile.
        #
        # a_cta_layout: layout of CTAs that share A data = N-dimension of cluster.
        #   Extracted by slicing cluster_layout_vmnk at V=0, M=0, K=0, keeping N free.
        #   For 1x1 cluster: shape (_1,) — just this CTA, no multicast.
        # block_in_cluster_coord_vmnk[2]: this CTA's position within that N multicast group.
        #   For 1x1 cluster: always 0.
        #
        # group_modes groups ((Mma_M, Mma_K), NumMma_M, NumMma_K) into a single mode
        # so the TMA handles the entire SMEM tile in one copy call.
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],  # this CTA's N coord within cluster
            a_cta_layout,
            cute.group_modes(sA, 0, 3),       # (((Mma_M, Mma_K), NumMma_M, NumMma_K), DMA_Stage)
            cute.group_modes(tCgA, 0, 3),      # (((Mma_M, Mma_K), NumMma_M, NumMma_K), Tiles_M, Tiles_K, Gemm_L)
        )
        # tAsA: ((TMA, NumTma_K), DMA_Stage) — SMEM destination for each pipeline stage
        # tAgA: ((TMA, NumTma_K), Tiles_M, Tiles_K, Gemm_L) — GMEM source tiles

        # Slice to this CTA's M tile and batch index, keep K tiles and TMA modes free
        # C++: implicit via work_tile_info.M_idx and work_tile_info.L_idx in local_tile
        # tAgA after slice: ((TMA, NumTma_K), Tiles_K)
        tAgA = tAgA[(None, work_tile_info.M_idx, None, work_tile_info.L_idx)]

        # SFA partition (same cta_layout as A — SFA multicasts along N)
        gSFA_mkl = cute.local_tile(mSFA_mkl, (self.cta_m, self.cta_k), (None, None, None))
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tAgSFA = tAgSFA[(None, work_tile_info.M_idx, None, work_tile_info.L_idx)]

        # ---- K-loop with PipelineState ----
        # producer state starts at phase 1 (empty barriers init to phase 0 = ready)
        ab_state = make_pipeline_state(PipelineUserType.Producer, DMA_Stage)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            # wait for MMA to signal SMEM slot empty (auto phase tracking)
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)

            # set transaction bytes on full barrier
            # C++: set_barrier_transaction_bytes(..., tma_transaction_bytes)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    ab_full_bar + ab_state.index, self.tma_bytes_a)

            # issue TMA loads for A + SFA into this pipeline stage
            # CRITICAL: cute.copy for TMA must be OUTSIDE elect_one() —
            # TMA partition internally ensures only thread 0 issues cp.async.bulk;
            # wrapping in elect_one() causes GPU deadlock in DSL.
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(tma_atom_a,
                tAgA[(None, k_tile_global)], tAsA[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index)
            cute.copy(tma_atom_sfa,
                tAgSFA[(None, k_tile_global)], tAsSFA[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index)

            ab_state.advance() # update phase and index/stage

            # PDL: launch dependents at the computed k_tile, or unconditionally at end
            if self.use_pdl:
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if self.use_pdl:
            cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def dma_b_warp_tmem_store(
        self,
        ab_full_bar, ab_empty_bar,
        tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor, sB: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB_nkl: cute.Tensor, sSFB: cute.Tensor,
        tiled_mma: cute.TiledMma, cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple, mma_tile_coord_v: cutlass.Int32,
        work_tile_info: Tuple, k_tile_count: cutlass.Int32,
    ):
        """DMA_B warp (tcgen05.st SFB path): flat TMA for SFB."""
        DMA_Stage = self.num_ab_stage
        gB_nkl = cute.local_tile(mB_nkl, (self.cta_n, self.cta_k), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tBgB = tBgB[(None, work_tile_info.N_idx, None, work_tile_info.L_idx)]
        sfb_k = self.cta_k // self.sf_vec_size
        gSFB_nkl = cute.local_tile(mSFB_nkl, (self.cta_n, sfb_k), (None, None, None))
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb, 0, cute.make_layout(1),
            cute.group_modes(sSFB, 0, 2), cute.group_modes(gSFB_nkl, 0, 2))
        tBgSFB = tBgSFB[(None, work_tile_info.N_idx, None, work_tile_info.L_idx)]
        if self.use_pdl:
            cute.arch.griddepcontrol_wait()

        ab_state = make_pipeline_state(PipelineUserType.Producer, DMA_Stage)

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    ab_full_bar + ab_state.index, self.tma_bytes_b)
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(tma_atom_b, tBgB[(None, k_tile_global)], tBsB[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index)
            cute.copy(tma_atom_sfb, tBgSFB[(None, k_tile_global)], tBsSFB[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index)
            ab_state.advance()

    @cute.jit
    def dma_b_warp_s2t_copy(
        self,
        ab_full_bar,
        ab_empty_bar,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,              # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor
        sB: cute.Tensor,                  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,            # SFB TMA coordinate tensor (N padded to 128)
        sSFB: cute.Tensor,                # SFB SMEM tensor (padded BlockScaled, staged)
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: Tuple,             # WorkTileInfo: (M_idx, N_idx, L_idx, K_idx_start, K_idx_end)
        k_tile_count: cutlass.Int32,
    ):
        """DMA_B warp: loads B tiles via TMA. C++ DMA_B_warp() lines 327-390."""
        DMA_Stage = self.num_ab_stage

        # ---- Self-contained tensor partitioning (similar to DMA_A_warp) ----
        # Tile mB_nkl (Gemm_N, Gemm_K, Gemm_L) into CTA-level tiles
        # C++: local_tile(mB, make_shape(CTA_N, CTA_K), make_coord(N_idx, _, L_idx))
        # gB_nkl: (CTA_N, CTA_K, Tiles_N, Tiles_K, Gemm_L)
        gB_nkl = cute.local_tile(
            mB_nkl, (self.cta_n, self.cta_k), (None, None, None)
        )
        # C++: ThrMMA cta_mma = tiled_mma.get_slice(0)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, Tiles_N, Tiles_K, Gemm_L)
        tCgB = thr_mma.partition_B(gB_nkl)

        # Same pattern as A, but for B the multicast group is along M.
        # All CTAs along M share the same B tile (same N, same K, different M).
        # b_cta_layout: layout of CTAs that share B data = M-dimension of cluster.
        #   Extracted by slicing cluster_layout_vmnk at V=0, N=0, K=0, keeping M free.
        #   For 1x1 cluster: shape (_1,) — just this CTA, no multicast.
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],  # this CTA's M coord within cluster
            b_cta_layout,
            cute.group_modes(sB, 0, 3),       # (((Mma_N, Mma_K), NumMma_N, NumMma_K), DMA_Stage)
            cute.group_modes(tCgB, 0, 3),      # (((Mma_N, Mma_K), NumMma_N, NumMma_K), Tiles_N, Tiles_K, Gemm_L)
        )
        # tBsB: ((TMA, NumTma_K), DMA_Stage)
        # tBgB: ((TMA, NumTma_K), Tiles_N, Tiles_K, Gemm_L)

        # Slice to this CTA's N tile and batch index
        # tBgB after slice: ((TMA, NumTma_K), Tiles_K)
        tBgB = tBgB[(None, work_tile_info.N_idx, None, work_tile_info.L_idx)]

        # SFB TMA partition (same pattern as B but using padded tiled_mma_sfb)
        n_padded = cute.round_up(self.cta_n, 128)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, (n_padded, self.cta_k), (None, None, None)
        )
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tBgSFB = tBgSFB[(None, work_tile_info.N_idx, None, work_tile_info.L_idx)]

        # PDL: wait on dependent grids only for B (the activation tensor)
        if self.use_pdl:
            cute.arch.griddepcontrol_wait()

        ab_state = make_pipeline_state(PipelineUserType.Producer, DMA_Stage)

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    ab_full_bar + ab_state.index, self.tma_bytes_b)
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(tma_atom_b,
                tBgB[(None, k_tile_global)], tBsB[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index)
            cute.copy(tma_atom_sfb,
                tBgSFB[(None, k_tile_global)], tBsSFB[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index)
            ab_state.advance()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make tiled copy for SMEM→TMEM load of scale factors, then partition."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def sfb_warp(
        self,
        ab_full_bar,
        sfb_full_bar,
        sfb_empty_bar,
        tmem_alloc_result_bar,
        tmem_base_smem_ptr,
        sSFB: cute.Tensor,
        k_tile_count: cutlass.Int32,
        subpartition_idx: cutlass.Int32,
    ):
        """SFB warp: writes SFB scale factors from SMEM to one TMEM subpartition via lds + tcgen05.st."""
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage
        sfb_k = self.cta_k // self.sf_vec_size
        # participate in TMEM alloc sync — need the TMEM base address
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        # retrieve TMEM base ptr and compute SFB TMEM address for this subpartition
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, 16, tmem_base_smem_ptr,
        )
        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self._num_sfa_tmem_cols + self.cta_n,
            dtype=self.sf_dtype)

        # staged TMEM layout: add SFB_Stage as outermost mode
        sfb_kblock_stride = self._num_n_atoms * self._sf_per_mma_k
        sfb_stage_stride = self.mma_inst_tile_k * sfb_kblock_stride
        sp_tmem_staged = cute.make_tensor(sfb_tmem_ptr, cute.make_layout(
            ((self._sf_atom_mn, (1, self._sf_per_mma_k)),
             1, self.mma_inst_tile_k, SFB_Stage),
            stride=((self._sfb_lane_stride, (0, 1)),
                    0, sfb_kblock_stride, sfb_stage_stride)))

        # single-stage slice for copy plan (layout same across stages)
        sp_tmem_single = sp_tmem_staged[(None, None, None, 0)]
        sp_tmem_compact_single = cute.filter_zeros(sp_tmem_single)
        # staged compact for destination partitioning
        sp_tmem_compact_staged = cute.filter_zeros(sp_tmem_staged)

        tidx_in_warp = cute.arch.lane_idx()
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(1)), self.sf_dtype,
        )
        tmem_store_tiled = tcgen05.make_tmem_copy(tmem_store_atom, sp_tmem_compact_single)
        tmem_store_thr = tmem_store_tiled.get_slice(tidx_in_warp)
        tStDst_staged = tmem_store_thr.partition_D(sp_tmem_compact_staged)
        tStSrc = tmem_store_thr.partition_S(sp_tmem_compact_single)
        rSFB = cute.make_rmem_tensor(tStSrc.shape, self.sf_dtype)

        # pipeline states: observe AB full (read-only), produce SFB TMEM
        ab_observer = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
        sfb_state = make_pipeline_state(PipelineUserType.Producer, SFB_Stage)

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            # observe AB full (read-only — SFB warps don't release the AB slot)
            cute.arch.mbarrier_wait(ab_full_bar + ab_observer.index, ab_observer.phase)

            # LDS: load SFB from SMEM to RMEM
            sSFB_stage = sSFB[(None, None, ab_observer.index)]
            smem_row = tidx_in_warp
            if smem_row >= self.cta_n:
                smem_row = cutlass.Int32(0)
            for i in cutlass.range(cute.size(rSFB)):
                rSFB[i] = sSFB_stage[smem_row + i * self.cta_n]

            # wait for MMA to consume this TMEM SFB stage (auto phase via PipelineState)
            cute.arch.mbarrier_wait(sfb_empty_bar + sfb_state.index, sfb_state.phase)

            # tcgen05.st: write to staged TMEM destination
            cute.copy(tmem_store_tiled, rSFB, tStDst_staged[(None, None, None, None, sfb_state.index)])
            cute.arch.fence_view_async_tmem_store()

            # signal MMA that this stage's SFB is in TMEM
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(sfb_full_bar + sfb_state.index)

            ab_observer.advance()
            sfb_state.advance()

        # SFB warps done — arrive at tmem_alloc for dealloc phase
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

    @cute.jit
    def mma_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        sfb_full_bar,
        sfb_empty_bar,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,                  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,                  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_padded_staged: cute.Layout,
        tmem_base_smem_ptr,
        k_tile_count: cutlass.Int32,
    ):
        """MMA warp: performs blockscaled tcgen05.mma with scale factors."""
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage

        # ---- MMA Fragment Allocation ----
        # We allocate "fragments" which are SMEM descriptors that serve as inputs to cute.gemm.
        # For tcgen05.mma operations:
        #   - Matrices A and B are sourced from SMEM
        #   - tCrA and tCrB provide descriptor views of sA and sB respectively
        #   - The first mode of each descriptor represents the SMEM for a single MMA instruction
        # C++: tCrA = cta_mma.make_fragment_A(tCsA)
        # tCrA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — SMEM descriptors for A
        tCrA = tiled_mma.make_fragment_A(sA)
        # tCrB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — SMEM descriptors for B
        tCrB = tiled_mma.make_fragment_B(sB)

        # TMEM Accumulator shape
        # On SM100, accumulators are stored exclusively in tensor memory (TMEM).
        # make_fragment_C creates a TMEM tensor view with the appropriate layout.
        # tCtAcc is a view of the accumulator tensor; its tmem base ptr is unset until alloc_tmem.
        # acc_shape / tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        # For bf16 M64 N8 K16: (((_16,_4),_8),_1,_1) — Mma_M=(16,4)=(Mma_M_per_subp, NumSubp), Mma_N=8
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # ---- TMEM Allocation ----
        # Only use half of TMEM to allow overlapping with next CTA
        # TMEM has 128 lanes, 512 columns, each word is 4B, 256KB total
        # Our accumulator uses CTA_N * sizeof(AccType) columns
        # C++: tmem_allocator.allocate(Sm100TmemCapacityColumns / 2, &shared_storage.tmem_base_ptr)
        num_tmem_cols = 256  # SM100_TMEM_CAPACITY_COLUMNS / 2
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_smem_ptr)

        # Notify epilog warp that TMEM allocation is complete
        # 32 threads (MMA warp) arrive; NO WAIT here — just arrives and continues
        # C++: arrive_barrier(shared_storage.tmem_allocation_result_barrier)
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # Relinquish TMEM allocation lock early so that the next prefetch CTA can be launched
        # C++: tmem_allocator.release_allocation_lock()
        cute.arch.relinquish_tmem_alloc_permit()

        # Retrieve TMEM ptr from SMEM and create the accumulator tensor view
        # C++: tCtAcc.data() = shared_storage.tmem_base_ptr
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, 16, tmem_base_smem_ptr,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # ---- SFA/SFB TMEM tensors and S2T copy partition ----
        sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + self.cta_n, dtype=self.sf_dtype)
        sfa_mem_single = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, tiled_mma.shape_mnk, self.sf_vec_size, sfa_mem_single)
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self._num_sfa_tmem_cols + self.cta_n, dtype=self.sf_dtype)

        # SFA: SMEM-to-TMEM copy (used by both paths)
        (tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)

        # ---- MMA mainloop (SFB TMEM setup + k-loop, branched by sfb_tmem_store) ----
        if self.sfb_tmem_store:
            # tcgen05.st path: staged SFB TMEM layout with _num_n_atoms=2
            # bypassing blockscaled_utils.make_tmem_layout_sfb to build our own layout
            # because that utility pads cta_n up to 128 and thus assumes _num_n_atoms = 4
            # even if mma_instruction only requires mma subpartition to be padded to 2 cols
            # instead of 4. So the only change:
            # new layout we use [same except for _num_n_atoms]:
            # ((((32,2),4),(16,4)),1,4):((((262144,4),8388608),(0,1)),0,8)
            # = ((((32 rows,2 cols),4 subp),(16 elems,4 sfs)),1,4 k-blocks, 4 num_sfb_stages)
            sfb_kblock_stride = self._num_n_atoms * self._sf_per_mma_k
            sfb_stage_stride = self.mma_inst_tile_k * sfb_kblock_stride
            tCtSFB_layout_staged = cute.make_layout(
                (
                    (
                        (
                            (self._sf_atom_mn, self._num_n_atoms),
                            128 // self._sf_atom_mn
                        ),
                        (self.sf_vec_size, self._sf_per_mma_k)
                    ),
                    1,
                    self.mma_inst_tile_k,
                    SFB_Stage
                ),
                stride=(
                    (
                        (
                            (self._sfb_lane_stride, self._sf_per_mma_k),
                            self._sf_atom_mn * self._sfb_lane_stride
                        ),
                        (0, 1)
                    ),
                    0,
                    sfb_kblock_stride,
                    sfb_stage_stride
                )
            )
            tCtSFB_staged = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout_staged)

            # ---- MMA mainloop with try_wait/wait overlap (tcgen05.st path) ----
            # try_wait pattern: non-blocking mbarrier.try_wait returns a predicate
            # so subsequent MMA instructions can issue without spin-looping.
            # if the try_wait failed, the next iteration does a blocking wait.
            ab_state = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
            sfb_state = make_pipeline_state(PipelineUserType.Consumer, SFB_Stage)
            smem_wait_done = cutlass.Boolean(False)
            sfb_wait_done = cutlass.Boolean(False)

            for k_tile in cutlass.range(k_tile_count):
                if ~smem_wait_done:
                    cute.arch.mbarrier_wait(ab_full_bar + ab_state.index, ab_state.phase)

                old_ab_index = ab_state.index
                ab_state.advance()

                old_sfb_index = sfb_state.index
                old_sfb_phase = sfb_state.phase
                sfb_state.advance()

                # reset: assume next stage is not ready unless try_wait succeeds below
                smem_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    smem_wait_done = cute.arch.mbarrier_try_wait(
                        ab_full_bar + ab_state.index, ab_state.phase)

                # copy SFA from SMEM to TMEM (tcgen05.cp)
                s2t_stage_coord = (None, None, None, None, old_ab_index)
                cute.copy(tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)

                # wait for SFB warps to finish writing this TMEM stage
                if ~sfb_wait_done:
                    cute.arch.mbarrier_wait(sfb_full_bar + old_sfb_index, old_sfb_phase)

                # slice staged SFB TMEM to this stage for GEMM
                tCtSFB_mma = tCtSFB_staged[(None, None, None, old_sfb_index)]

                # blockscaled GEMM: A*SFA * B*SFB -> Acc
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                tile_crd = (None, None, None, old_ab_index)
                cute.gemm(
                    tiled_mma, tCtAcc,
                    [tCrA[tile_crd], tCtSFA],
                    [tCrB[tile_crd], tCtSFB_mma],
                    tCtAcc,
                )

                with cute.arch.elect_one():
                    # signal SFB warps that TMEM SFB stage consumed + signal DMA SMEM slot empty
                    tcgen05.commit(sfb_empty_bar + old_sfb_index, None, self.cta_group)
                    tcgen05.commit(ab_empty_bar + old_ab_index, None, self.cta_group)

                # try_wait on next iteration's SFB TMEM stage (non-blocking peek)
                sfb_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    sfb_wait_done = cute.arch.mbarrier_try_wait(
                        sfb_full_bar + sfb_state.index, sfb_state.phase)
        else:
            # tcgen05.cp path: use standard make_tmem_layout_sfb (single-stage, overwritten each k-tile)
            sfb_smem_single = cute.slice_(sfb_smem_layout_padded_staged, (None, None, None, 0))
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, tiled_mma.shape_mnk, self.sf_vec_size, sfb_smem_single)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # SFB: tcgen05.cp (hardware SMEM→TMEM copy, replaces tcgen05.st SFB warps)
            (tiled_copy_s2t_sfb, tCsSFB_compact_s2t, tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            # ---- MMA mainloop with try_wait/wait overlap (tcgen05.cp path) ----
            # try_wait pattern: non-blocking mbarrier.try_wait returns a predicate
            # so subsequent MMA instructions can issue without spin-looping.
            # if the try_wait failed, the next iteration does a blocking wait.
            ab_state = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
            smem_wait_done = cutlass.Boolean(False)

            for k_tile in cutlass.range(k_tile_count):
                if ~smem_wait_done:
                    cute.arch.mbarrier_wait(ab_full_bar + ab_state.index, ab_state.phase)

                old_ab_index = ab_state.index
                ab_state.advance()

                # reset: assume next stage is not ready unless try_wait succeeds below
                smem_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    smem_wait_done = cute.arch.mbarrier_try_wait(
                        ab_full_bar + ab_state.index, ab_state.phase)

                # copy SFA from SMEM to TMEM (tcgen05.cp)
                s2t_stage_coord = (None, None, None, None, old_ab_index)
                cute.copy(tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)

                # copy SFB from SMEM to TMEM (tcgen05.cp — no SFB warps needed)
                cute.copy(tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t[s2t_stage_coord], tCtSFB_compact_s2t)

                # blockscaled GEMM: A*SFA * B*SFB -> Acc
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                tile_crd = (None, None, None, old_ab_index)
                cute.gemm(
                    tiled_mma, tCtAcc,
                    [tCrA[tile_crd], tCtSFA],
                    [tCrB[tile_crd], tCtSFB],
                    tCtAcc,
                )

                with cute.arch.elect_one():
                    tcgen05.commit(ab_empty_bar + old_ab_index, None, self.cta_group)

        # signal epilog: MMA done, accumulator ready (via tcgen05.commit on full barrier)
        with cute.arch.elect_one():
            tcgen05.commit(mma_epilog_full_bar, None, self.cta_group)

        # TMEM dealloc protocol
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols)

    @cute.jit
    def epilog_warp(
        self,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        tiled_mma: cute.TiledMma,
        mC_mnl: cute.Tensor,              # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        tmem_base_smem_ptr,
        mailbox_mbar,                # local SMEM transaction barrier; peer CTAs signal it via st.async
        dsmem_mailbox_ptr,                 # SMEM pointer for DSMEM stores and reads
        epi_tid: cutlass.Int32,            # thread id within epilog warps (0-127)
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: Tuple,             # WorkTileInfo: (M_idx, N_idx, L_idx, K_idx_start, K_idx_end)
        split_rank: cutlass.Int32,
    ):
        """EPILOG warp: TMEM -> RMEM -> type convert -> GMEM. C++ EPILOG_warp() lines 600-838."""

        # ---- Self-contained tensor partitioning (matches C++) ----
        # Get the local tile of C for this CTA
        # C++: local_tile(mC, make_shape(CTA_M, CTA_N), make_coord(M_idx, N_idx, L_idx))
        # gC_mnl: (CTA_M, CTA_N, Tiles_M, Tiles_N, Gemm_L)
        gC_mnl = cute.local_tile(
            mC_mnl, (self.cta_m, self.cta_n), (None, None, None)
        )
        # C++: ThrMMA cta_mma = tiled_mma.get_slice(0)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgC: ((Mma_M, Mma_N), NumMma_M, NumMma_N, Tiles_M, Tiles_N, Gemm_L)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Since tCtAcc is a view of the accumulator tensor, it's safe to create a new view
        # in the epilog warp too. The layout is the same as in the MMA warp.
        # acc_shape / tCtAcc_fake: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # ---- Arrive + wait tmem_allocation_result_barrier phase 0 ----
        # 128 threads from EPILOG warps (4-7) arrive
        # C++: arrive_barrier(shared_storage.tmem_allocation_result_barrier)
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        # Wait for MMA warp to complete TMEM allocation
        # Initial phase=0, it will flip to 1 when TMEM is allocated, so we wait for old phase 0
        # C++: wait_barrier(shared_storage.tmem_allocation_result_barrier, 0)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        # Update TMEM base ptr of the accumulator tensor view
        # C++: tCtAcc.data() = shared_storage.tmem_base_ptr
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype, 16, tmem_base_smem_ptr,
        )
        # tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N) — TMEM accumulator view
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # ---- Create tiled copy for TMEM -> RMEM ----
        # C++: TiledCopy tiled_t2r_copy = make_tmem_copy(
        #          TMEM::op_repeater<SM100_TMEM_LOAD_16dp256b1x, acc_col_bits>(), tCtAcc)
        # For M=64, M/N major output, we use SM100_TMEM_LOAD_16dp256b1x:
        #   - 16dp version because M=64, each sub-partition uses 16dp of TMEM in MMA
        #   - 256b = 32B per load, loads 16dp x 8col per tcgen05.ld instruction
        # The op_repeater automatically figures out the tcgen05.ld repeat count based on CTA_N
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            (self.cta_m, self.cta_n),
            self.use_2cta_instrs,
        )

        # C++: TiledCopy tiled_t2r_copy = make_tmem_copy(op, tCtAcc)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc[((None, None), 0, 0)]
        )
        # Epilog tid is 0-127 (threads 128-255 offset by -128)
        # C++: ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(tid)
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)

        # tD means the partitioning pattern of tcgen05.ld (TMEM->RMEM copy)
        # tDtAcc: per-subpartition view of the accumulator tensor (TMEM source)
        # Shape: (CpyS, NumCpy_M, NumCpy_N)
        # For bf16 M64 N8, tcgen05.ld atom SM100_TMEM_LOAD_16dp256b1x:
        #   CpyS = (CpyS_N, CpyS_M) = (8, 16) — 16dp x 8col per tcgen05.ld instruction
        #   NumCpy_M = 1, NumCpy_N = CTA_N / CpyS_N = 1
        # C++: tDtAcc = thr_t2r_copy.partition_S(tCtAcc)
        tDtAcc = thr_copy_t2r.partition_S(tCtAcc[((None, None), 0, 0)])

        # Partition tCgC for 2 reasons:
        # 1. Get post-partition shape for allocating rmem space for the accumulator
        # 2. Partition it for storing the result back to gmem (per-thread slice)
        # tDgC represents which values in tCgC are stored in this thread's rmem
        # after tcgen05.ld from TMEM to RMEM
        # Shape: (CpyD, NumCpy_M, NumCpy_N, Tiles_M, Tiles_N, Gemm_L)
        # For bf16 M64 N8: CpyD = (2,2) — 4 registers per thread per tcgen05.ld
        # C++: tDgC = thr_t2r_copy.partition_D(tCgC)
        # DSL tCgC has extra grid dims; partition with them, create rmem before slicing
        tDgC = thr_copy_t2r.partition_D(
            tCgC[((None, None), 0, 0, None, None, None)]
        )

        # Allocate per-thread rmem space for the accumulator
        # tDrAcc and tDrC have the same shape as the single-tile slice of tDgC
        # Shape: (CpyD, NumCpy_M, NumCpy_N)
        # C++: Tensor tDrAcc = make_tensor<AccType>(shape(tDgC))
        # IMPORTANT: Create rmem tensors BEFORE slicing tDgC with work_tile_info,
        # because slicing reduces rank and breaks shape extraction
        epi_frag_shape = tDgC[(None, None, None, 0, 0, 0)].shape
        # tDrAcc: (CpyD, NumCpy_M, NumCpy_N) — per-thread accumulator in rmem (AccType)
        tDrAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)
        # tDrC: (CpyD, NumCpy_M, NumCpy_N) — per-thread converted output in rmem (TypeC)
        tDrC = cute.make_rmem_tensor(epi_frag_shape, self.c_dtype)

        # Slice gmem partition to this CTA's output tile
        # tDgC after slice: (CpyD, NumCpy_M, NumCpy_N) — per-thread GMEM store destinations
        tDgC = tDgC[(None, None, None, work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx)]

        # ---- Bounds-check predicate (C++: make_identity_tensor + elem_less) ----
        # An identity tensor maps (m, n, l) coordinates to the payload tuple (m, n, l).
        # We use this to check if each thread's store coordinates are within bounds,
        # because the problem shape may not be a multiple of the tile size.
        # For TMA loads, OOB is handled automatically; for stg stores, we need explicit predicates.
        # C++: Tensor coordC = make_identity_tensor(shape(mC))
        coordC = cute.make_identity_tensor(mC_mnl.shape)     # (M, N, L) -> (m, n, l)
        # Create the local tile of coordC, same tiling as gC
        # C++: local_tile(coordC, make_shape(CTA_M, CTA_N), make_coord(M_idx, N_idx, L_idx))
        gCcoord = cute.local_tile(
            coordC, (self.cta_m, self.cta_n), (None, None, None)
        )
        # tCcC: ((Mma_M, Mma_N), NumMma_M, NumMma_N, Tiles_M, Tiles_N, Gemm_L)
        tCcC = thr_mma.partition_C(gCcoord)
        # tDcC: (CpyD, NumCpy_M, NumCpy_N, Tiles_M, Tiles_N, Gemm_L)
        # — per-thread coordinate tensor, same shape as tDgC but payload is (m,n,l) coord
        tDcC = thr_copy_t2r.partition_D(
            tCcC[((None, None), 0, 0, None, None, None)]
        )
        # Slice to this CTA's tile: (CpyD, NumCpy_M, NumCpy_N)
        tDcC = tDcC[(None, None, None, work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx)]
        # Construct predicate tensor: compare each coordinate with problem shape (M, N, L)
        # tDpC(t) = true if coordinate is in-bounds, false if out-of-bounds
        # C++: tDpC(t) = elem_less(tDcC(t), shape(mC))
        tDpC = cute.make_rmem_tensor(tDcC.shape, cutlass.Boolean)  # (CpyD, NumCpy_M, NumCpy_N)
        for i in range(cute.size(tDpC)):
            tDpC[i] = cute.elem_less(tDcC[i], mC_mnl.shape)

        # ---- Wait for MMA to finish (PipelineUmmaAsync consumer) ----
        mma_epilog_state = make_pipeline_state(PipelineUserType.Consumer, 1)
        cute.arch.mbarrier_wait(mma_epilog_full_bar + mma_epilog_state.index, mma_epilog_state.phase)

        # ---- Copy TMEM -> RMEM (single copy, no subtile loop) ----
        # The copy operation is:
        #   for each NumCpy_M:
        #     for each NumCpy_N:
        #       tcgen05.ld.16dp256bit.x1: CpyS -> CpyD
        # Each tcgen05.ld instruction copies 16dp x 8col (CpyS) for each TMEM subpartition
        # to 128 threads' rmem (4 registers (CpyD) per thread)
        # C++: copy(tiled_t2r_copy, tDtAcc, tDrAcc)
        cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)

        # ---- Signal MMA warp that TMEM read is done ----
        # Wait for tcgen05.ld to finish so we can safely deallocate TMEM
        # C++: cutlass::arch::fence_view_async_tmem_load()
        cute.arch.fence_view_async_tmem_load()
        # C++: arrive_barrier(shared_storage.tmem_allocation_result_barrier)
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # ---- Type conversion and GMEM store (with optional DSMEM split-k reduction) ----
        elems_per_thread = self._mailbox_elems_per_thread
        if self.split_k == 1:
            # direct path: convert AccType -> TypeC and store to GMEM
            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            cute.basic_copy_if(tDpC, tDrC, tDgC)
        else:
            # cluster split-k via DSMEM scatter with DISTRIBUTED REDUCTION:
            # the output tile is sharded across all split_k CTAs by flat element index.
            # each CTA sends shard-targeted partials to every peer, reduces its own
            # shard from all received partials + its own, and writes its shard to GMEM.
            #
            # shard assignment: CTA r owns flat elements [r*shard_ept, (r+1)*shard_ept)
            # where shard_ept = elems_per_thread / split_k.
            shard_ept = self._shard_elems_per_thread

            # every CTA arms its own mailbox transaction barrier BEFORE the scatter
            # to ensure it's armed before any peer's stores arrive
            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mailbox_mbar, self._mailbox_tx_total)

            # intra-CTA epilog sync: all 128 threads must finish TMEM→RMEM before scatter
            cute.arch.barrier(barrier_id=15, number_of_threads=128)

            # scatter: send each peer's shard of our partial accumulator to that peer.
            # each st.async.shared::cluster stores 4B and adds 4B to the remote
            # CTA's mailbox barrier TX counter.
            #
            # MAILBOX LAYOUT (thread-interleaved, per-CTA):
            #   addr = sender_idx * 128 * shard_ept + i * 128 + epi_tid
            # where sender_idx is this CTA's slot in the peer's mailbox (0..split_k-2),
            # i is the element index within the shard (0..shard_ept-1), and
            # epi_tid is the thread ID (0..127) as the innermost dimension for coalescing.
            dsmem_store_atom = cute.make_copy_atom(CopyDsmemStoreOp(), cutlass.Float32)
            scatter_val = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
            for peer in range(self.split_k):
                if peer != split_rank:
                    # elements belonging to peer's shard: [peer*shard_ept, (peer+1)*shard_ept)
                    shard_start = peer * shard_ept
                    # our slot in peer's mailbox: skip peer's own rank in the sender list
                    sender_idx = split_rank - (1 if split_rank > cutlass.Int32(peer) else 0)
                    mailbox_base = sender_idx * 128 * shard_ept
                    remote_mbar = map_dsmem_ptr(mailbox_mbar, cutlass.Int32(peer))
                    for i in range(shard_ept):
                        scatter_val[0] = tDrAcc[shard_start + i]
                        remote_ptr = map_dsmem_ptr(
                            dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid,
                            cutlass.Int32(peer))
                        dst = cute.make_tensor(remote_ptr, cute.make_layout(1))
                        cute.copy(dsmem_store_atom, scatter_val, dst, mbar_ptr=remote_mbar)

            # wait for all peer shards to arrive in our local mailbox
            cute.arch.mbarrier_wait(mailbox_mbar, 0)

            # reduce: accumulate received shards with our own partial.
            # my shard is at flat indices [split_rank*shard_ept, (split_rank+1)*shard_ept).
            # the mailbox contains (split_k-1) copies at sender_idx slots, same layout.
            my_shard_start = split_rank * shard_ept
            for s in range(self.split_k - 1):
                mailbox_base = s * 128 * shard_ept
                for i in range(shard_ept):
                    tDrAcc[my_shard_start + i] = tDrAcc[my_shard_start + i] + \
                        (dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid).load()

            # mask out non-shard predicates so basic_copy_if only writes our shard
            for i in range(elems_per_thread):
                if i < my_shard_start or i >= my_shard_start + shard_ept:
                    tDpC[i] = cutlass.Boolean(False)

            # convert and write our shard to GMEM
            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            cute.basic_copy_if(tDpC, tDrC, tDgC)


# ============================================================
# blockscaled format config
# ============================================================

@dataclass(frozen=True)
class BlockscaledFormat:
    name: str
    ab_torch_dtype: torch.dtype
    ab_cutlass_dtype: type
    sf_torch_dtype: torch.dtype
    sf_cutlass_dtype: type
    c_torch_dtype: torch.dtype
    c_cutlass_dtype: type
    sf_vec_size: int
    mma_k: int
    default_cta_k: int
    bytes_per_element: float  # 0.5 for fp4, 1.0 for fp8

NVFP4 = BlockscaledFormat(
    name="nvfp4",
    ab_torch_dtype=torch.float4_e2m1fn_x2,
    ab_cutlass_dtype=cutlass.Float4E2M1FN,
    sf_torch_dtype=torch.float8_e4m3fn,
    sf_cutlass_dtype=cutlass.Float8E4M3FN,
    c_torch_dtype=torch.bfloat16,
    c_cutlass_dtype=cutlass.BFloat16,
    sf_vec_size=16,
    mma_k=64,
    default_cta_k=256,
    bytes_per_element=0.5,
)

MXFP4 = BlockscaledFormat(
    name="mxfp4",
    ab_torch_dtype=torch.float4_e2m1fn_x2,
    ab_cutlass_dtype=cutlass.Float4E2M1FN,
    sf_torch_dtype=torch.float8_e8m0fnu,
    sf_cutlass_dtype=cutlass.Float8E8M0FNU,
    c_torch_dtype=torch.bfloat16,
    c_cutlass_dtype=cutlass.BFloat16,
    sf_vec_size=32,
    mma_k=64,
    default_cta_k=256,
    bytes_per_element=0.5,
)

MXFP8 = BlockscaledFormat(
    name="mxfp8",
    ab_torch_dtype=torch.float8_e4m3fn,
    ab_cutlass_dtype=cutlass.Float8E4M3FN,
    sf_torch_dtype=torch.float8_e8m0fnu,
    sf_cutlass_dtype=cutlass.Float8E8M0FNU,
    c_torch_dtype=torch.bfloat16,
    c_cutlass_dtype=cutlass.BFloat16,
    sf_vec_size=32,
    mma_k=32,
    default_cta_k=128,
    bytes_per_element=1.0,
)

FORMAT_MAP = {"nvfp4": NVFP4, "mxfp4": MXFP4, "mxfp8": MXFP8}


# ============================================================
# blockscaled helpers
# ============================================================

def ceil_div(a, b):
    return (a + b - 1) // b


def reorder_scale_factors(sf, mn, k, sf_vec_size):
    """Reorder scale factors from simple (MN, sf_K, L) to the BlockScaledBasicChunk
    atom layout expected by TMA and the s2t copy.

    Allocates in mma_shape and permutes to get interleaved (16, 4) strides
    matching tile_atom_to_shape_SF. Does NOT call .contiguous().
    """
    sf_k = ceil_div(k, sf_vec_size)
    atom_mn = 128  # 32 * 4
    atom_k = 4     # Mma_K / sf_vec_size = 64 / 16

    rest_mn = ceil_div(mn, atom_mn)
    rest_k = ceil_div(sf_k, atom_k)
    padded_mn = rest_mn * atom_mn
    padded_k = rest_k * atom_k

    original_dtype = sf.dtype
    l = sf.shape[2]

    sf_f32 = sf.to(torch.float32)
    if padded_mn != mn or padded_k != sf_k:
        sf_f32 = torch.nn.functional.pad(
            sf_f32, (0, 0, 0, padded_k - sf_k, 0, padded_mn - mn),
        )

    sf_f32 = sf_f32.view(rest_mn, atom_mn, rest_k, atom_k, l)
    sf_f32 = sf_f32.view(rest_mn, 4, 32, rest_k, atom_k, l)

    mma_shape = (l, rest_mn, rest_k, 32, 4, atom_k)
    out = torch.zeros(mma_shape, dtype=original_dtype, device=sf.device)
    out.permute(3, 4, 1, 5, 2, 0).copy_(sf_f32.permute(2, 1, 0, 4, 3, 5).to(original_dtype))

    return out


def to_blocked(input_matrix):
    """Convert (MN, sf_K) scale factors to blocked format for torch._scaled_mm."""
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    if padded_rows != rows or padded_cols != cols:
        original_dtype = input_matrix.dtype
        input_float32 = input_matrix.to(torch.float32)
        padded = torch.nn.functional.pad(
            input_float32, (0, padded_cols - cols, 0, padded_rows - rows),
        )
        if original_dtype != input_float32.dtype:
            padded = padded.to(original_dtype)
    else:
        padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def _reference_scaled_mm_nvfp4(a, b, sfa, sfb, c, mnkl):
    """Reference NVFP4 GEMM using to_blocked + torch._scaled_mm (native path)."""
    m, n, k, l = mnkl
    c_ref = torch.clone(c)
    n_padded = ceil_div(n, 16) * 16
    for l_idx in range(l):
        scale_a = to_blocked(sfa[:, :, l_idx])
        a_slice = a[:, :, l_idx].contiguous()
        b_slice = b[:, :, l_idx].contiguous()
        sfb_slice = sfb[:, :, l_idx]
        if n_padded != n:
            b_pad_bytes = torch.zeros((n_padded, b_slice.shape[1]), dtype=torch.int8, device=b_slice.device)
            b_pad_bytes[:n, :] = b_slice.view(dtype=torch.int8)
            b_slice = b_pad_bytes.view(dtype=b_slice.dtype)
            sfb_pad = torch.zeros((n_padded, sfb_slice.shape[1]), dtype=torch.float32, device=sfb_slice.device)
            sfb_pad[:n, :] = sfb_slice.to(torch.float32)
            sfb_slice = sfb_pad.to(sfb_slice.dtype)
        scale_b = to_blocked(sfb_slice)
        res = torch._scaled_mm(
            a_slice, b_slice.transpose(0, 1),
            scale_a.cuda(), scale_b.cuda(),
            bias=None, out_dtype=c_ref.dtype,
        )
        c_ref[:, :n, l_idx] = res[:, :n]
    return c_ref


# Float4E2M1FN LUT: 16 values indexed by 4-bit pattern [sign(1), exp(2), mantissa(1)]
_FP4_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,      # positive: s=0
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, # negative: s=1
], dtype=torch.float32)


def _decode_fp4x2_to_f32(packed_int8):
    """Decode packed float4_e2m1fn_x2 tensor to float32 on CPU.
    Each byte holds two fp4 values: low nibble = first element, high nibble = second.
    Returns a tensor with K dimension doubled.
    """
    raw = packed_int8.view(torch.int8).cpu().to(torch.int32)
    lo = raw & 0x0F
    hi = (raw >> 4) & 0x0F
    lo_f32 = _FP4_LUT[lo.to(torch.int64)]
    hi_f32 = _FP4_LUT[hi.to(torch.int64)]
    return torch.stack([lo_f32, hi_f32], dim=-1).reshape(*packed_int8.shape[:-1], -1)


def _reference_scaled_mm_emulated(a, b, sfa, sfb, c, mnkl, fmt):
    """Emulated reference: dequantize + float32 matmul. Works for all formats."""
    m, n, k, l = mnkl
    sf_vec_size = fmt.sf_vec_size
    c_ref = torch.clone(c)

    for l_idx in range(l):
        # decode data to float32
        if fmt.bytes_per_element == 0.5:
            a_f32 = _decode_fp4x2_to_f32(a[:, :, l_idx])
            b_f32 = _decode_fp4x2_to_f32(b[:, :, l_idx])
        else:
            a_f32 = a[:, :, l_idx].cpu().to(torch.float32)
            b_f32 = b[:, :, l_idx].cpu().to(torch.float32)

        # expand SFs: (MN, sf_K) -> (MN, K) by repeat_interleave then truncate
        sfa_f32 = torch.repeat_interleave(
            sfa[:, :, l_idx].cpu().to(torch.float32), sf_vec_size, dim=1)[:, :k]
        sfb_f32 = torch.repeat_interleave(
            sfb[:, :, l_idx].cpu().to(torch.float32), sf_vec_size, dim=1)[:, :k]

        # dequantize: element-wise multiply data * SF
        a_dequant = a_f32 * sfa_f32
        b_dequant = b_f32 * sfb_f32

        # matmul: C[m,n] = sum_k A[m,k] * B[n,k]
        ref = a_dequant @ b_dequant.T
        c_ref[:, :n, l_idx] = ref[:, :n].to(c.dtype).to(c.device)

    return c_ref



def _reference_scaled_mm_mxfp8(a, b, sfa, sfb, c, mnkl):
    """Reference MXF8 blockscaled GEMM using to_blocked + torch._scaled_mm (1x32 path)."""
    m, n, k, l = mnkl
    c_ref = torch.clone(c)
    n_padded = ceil_div(n, 16) * 16
    for l_idx in range(l):
        scale_a = to_blocked(sfa[:, :, l_idx])
        a_slice = a[:, :, l_idx].contiguous()
        b_slice = b[:, :, l_idx].contiguous()
        sfb_slice = sfb[:, :, l_idx]
        if n_padded != n:
            b_pad = torch.zeros((n_padded, b_slice.shape[1]), dtype=b_slice.dtype, device=b_slice.device)
            b_pad[:n, :] = b_slice
            b_slice = b_pad
            sfb_pad = torch.zeros((n_padded, sfb_slice.shape[1]), dtype=torch.float32, device=sfb_slice.device)
            sfb_pad[:n, :] = sfb_slice.to(torch.float32)
            sfb_slice = sfb_pad.to(sfb_slice.dtype)
        scale_b = to_blocked(sfb_slice)
        res = torch._scaled_mm(
            a_slice, b_slice.T,
            scale_a.cuda(), scale_b.cuda(),
            bias=None, out_dtype=c_ref.dtype,
        )
        c_ref[:, :n, l_idx] = res[:, :n]
    return c_ref


def reference_scaled_mm(a, b, sfa, sfb, c, mnkl, fmt=None):
    """Reference blockscaled GEMM — dispatches based on format.
    NVFP4: uses torch._scaled_mm (to_blocked + blockwise 1x16).
    MXF8: uses torch._scaled_mm (to_blocked + blockwise 1x32).
    MXFP4: uses emulated dequantize + float32 matmul (no native torch path).
    """
    if fmt is None:
        fmt = NVFP4
    if fmt is NVFP4:
        return _reference_scaled_mm_nvfp4(a, b, sfa, sfb, c, mnkl)
    elif fmt is MXFP8:
        return _reference_scaled_mm_mxfp8(a, b, sfa, sfb, c, mnkl)
    else:
        # MXFP4 and any other format: use emulated dequantize path
        return _reference_scaled_mm_emulated(a, b, sfa, sfb, c, mnkl, fmt)


def make_nvfp4_tensors(m, n, k, l, sf_vec_size=16, c_dtype=torch.float16, sfb_tmem_store=False):
    """Create NVFP4 A/B tensors with scale factors and output tensor."""
    sf_k = ceil_div(k, sf_vec_size)

    # A/B as random int8, viewed as float4_e2m1fn_x2
    # K-major layout: (MN, K//2_packed, L)
    a = torch.empty((l, m, k // 2), dtype=torch.int8, device="cuda").permute(1, 2, 0)
    b = torch.empty((l, n, k // 2), dtype=torch.int8, device="cuda").permute(1, 2, 0)
    a.copy_(torch.randint(-2, 2, a.shape, dtype=torch.int8, device="cuda"))
    b.copy_(torch.randint(-2, 2, b.shape, dtype=torch.int8, device="cuda"))
    a = a.view(dtype=torch.float4_e2m1fn_x2)
    b = b.view(dtype=torch.float4_e2m1fn_x2)

    # Scale factors: simple layout (MN, sf_K, L) — kept for reference
    sfa_simple = (
        torch.randint(0, 3, (l, m, sf_k), dtype=torch.uint8, device="cuda")
        .permute(1, 2, 0).to(dtype=torch.float8_e4m3fn)
    )
    sfb_simple = (
        torch.randint(0, 3, (l, n, sf_k), dtype=torch.uint8, device="cuda")
        .permute(1, 2, 0).to(dtype=torch.float8_e4m3fn)
    )

    # SFA: atom-reordered for TMA/tcgen05.cp
    sfa_reordered = reorder_scale_factors(sfa_simple, m, k, sf_vec_size)

    if sfb_tmem_store:
        # tcgen05.st path: SFB as K-major layout matching kernel's order=(1,0,2)
        sfb_reordered = sfb_simple.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    else:
        # tcgen05.cp path: SFB pad N to 128 and atom-reorder for TMA (BlockScaledBasicChunk)
        n_padded_sfb = ceil_div(n, 128) * 128
        sfb_padded = torch.zeros(
            (n_padded_sfb, sf_k, l), dtype=sfb_simple.dtype, device=sfb_simple.device)
        sfb_padded[:n, :, :] = sfb_simple
        sfb_reordered = reorder_scale_factors(sfb_padded, n_padded_sfb, k, sf_vec_size)

    # C: col-major (M, N, L) with M-contiguous
    c = torch.zeros((l, n, m), dtype=c_dtype, device="cuda").permute(2, 1, 0)

    return a, b, sfa_reordered, sfb_reordered, c, sfa_simple, sfb_simple


def make_blockscaled_tensors(m, n, k, l, fmt, sfb_tmem_store=False):
    """Create blockscaled A/B tensors with scale factors and output tensor for any format."""
    sf_vec_size = fmt.sf_vec_size
    sf_k = ceil_div(k, sf_vec_size)

    if fmt.bytes_per_element == 0.5:
        # FP4: packed as int8, view as float4_e2m1fn_x2
        a = torch.empty((l, m, k // 2), dtype=torch.int8, device="cuda").permute(1, 2, 0)
        b = torch.empty((l, n, k // 2), dtype=torch.int8, device="cuda").permute(1, 2, 0)
        a.copy_(torch.randint(-2, 2, a.shape, dtype=torch.int8, device="cuda"))
        b.copy_(torch.randint(-2, 2, b.shape, dtype=torch.int8, device="cuda"))
        a = a.view(dtype=fmt.ab_torch_dtype)
        b = b.view(dtype=fmt.ab_torch_dtype)
    else:
        # FP8: create as small unsigned ints, reinterpret as fp8.
        # avoid 0x7F/0xFF (NaN in E4M3FN) by using values 0..3
        a = torch.randint(0, 4, (l, m, k), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        b = torch.randint(0, 4, (l, n, k), dtype=torch.uint8, device="cuda").permute(1, 2, 0)
        a = a.view(dtype=fmt.ab_torch_dtype)
        b = b.view(dtype=fmt.ab_torch_dtype)

    # scale factors: simple layout (MN, sf_K, L) — use randint(1,3) to avoid zero exponents
    # for Float8E8M0FNU (which would produce NaN/denorm)
    sf_low = 1 if fmt.sf_torch_dtype == torch.float8_e8m0fnu else 0
    sfa_simple = (
        torch.randint(sf_low, 3, (l, m, sf_k), dtype=torch.uint8, device="cuda")
        .permute(1, 2, 0).to(dtype=fmt.sf_torch_dtype)
    )
    sfb_simple = (
        torch.randint(sf_low, 3, (l, n, sf_k), dtype=torch.uint8, device="cuda")
        .permute(1, 2, 0).to(dtype=fmt.sf_torch_dtype)
    )

    sfa_reordered = reorder_scale_factors(sfa_simple, m, k, sf_vec_size)

    if sfb_tmem_store:
        sfb_reordered = sfb_simple.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    else:
        n_padded_sfb = ceil_div(n, 128) * 128
        sfb_padded = torch.zeros(
            (n_padded_sfb, sf_k, l), dtype=sfb_simple.dtype, device=sfb_simple.device)
        sfb_padded[:n, :, :] = sfb_simple
        sfb_reordered = reorder_scale_factors(sfb_padded, n_padded_sfb, k, sf_vec_size)

    c = torch.zeros((l, n, m), dtype=fmt.c_torch_dtype, device="cuda").permute(2, 1, 0)
    return a, b, sfa_reordered, sfb_reordered, c, sfa_simple, sfb_simple


def to_cute_tensors(A, B, SFA, SFB, C, fmt):
    """Convert PyTorch tensors to CuTe pointers using format-specific dtypes."""
    a_ = make_ptr(fmt.ab_cutlass_dtype, A.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ = make_ptr(fmt.ab_cutlass_dtype, B.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    sfa_ = make_ptr(fmt.sf_cutlass_dtype, SFA.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
    sfb_ = make_ptr(fmt.sf_cutlass_dtype, SFB.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    c_ = make_ptr(fmt.c_cutlass_dtype, C.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    return a_, b_, sfa_, sfb_, c_


def to_cute_tensors_nvfp4(A_fp4, B_fp4, SFA, SFB, C, sf_vec_size=16):
    """Convert NVFP4 PyTorch tensors to CuTe pointers. (backward compat wrapper)"""
    return to_cute_tensors(A_fp4, B_fp4, SFA, SFB, C, NVFP4)


# ============================================================
# Compilation and execution helpers
# ============================================================


def compile_tgv_gemm_nvfp4(a_fp4, b_fp4, sfa, sfb, c, acc_dtype,
                            problem_mnkl=None, sf_vec_size=16,
                            cta_m=128, cta_n=8, cta_k=256,
                            num_ab_stage=8, num_sfb_tmem_stage=4,
                            use_pdl=False, pdl_count=-1,
                            split_k=1, sfb_tmem_store=False):
    gemm = TgvGemmKernel(
        acc_dtype=acc_dtype,
        cta_m=cta_m, cta_n=cta_n, cta_k=cta_k,
        num_ab_stage=num_ab_stage,
        num_sfb_tmem_stage=num_sfb_tmem_stage,
        sf_vec_size=sf_vec_size,
        use_pdl=use_pdl, pdl_count=pdl_count,
        split_k=split_k,
        sfb_tmem_store=sfb_tmem_store,
    )
    stream = make_fake_stream()
    # constexpr if in __call__ handles the dispatch — only the taken branch is traced
    return cute.compile(gemm, a_fp4, sfa, b_fp4, sfb, c, problem_mnkl, stream,
                        options="--generate-line-info")


def verify(m, n, k, l, fmt=NVFP4, sf_vec_size=None, c_dtype=None,
           cta_m=128, cta_n=8, cta_k=None, num_ab_stage=8,
           num_sfb_tmem_stage=4,
           atol=1e-1, rtol=1e-3, use_pdl=False, split_k=1,
           sfb_tmem_store=False):
    """Verify blockscaled TGV GEMM kernel against torch._scaled_mm."""
    if sf_vec_size is None:
        sf_vec_size = fmt.sf_vec_size
    if c_dtype is None:
        c_dtype = fmt.c_torch_dtype
    if cta_k is None:
        cta_k = fmt.default_cta_k

    print(f"=== {fmt.name.upper()} Correctness Test ===")
    print(f"Problem: M={m}, N={n}, K={k}, L={l}")
    print(f"Config: CTA=({cta_m}, {cta_n}, {cta_k}), AB stages={num_ab_stage}, "
          f"SFB TMEM stages={num_sfb_tmem_stage}, PDL={use_pdl}, split_k={split_k}, "
          f"sfb_tmem_store={sfb_tmem_store}")

    a, b, sfa_reordered, sfb_reordered, c, sfa_simple, sfb_simple = make_blockscaled_tensors(
        m, n, k, l, fmt, sfb_tmem_store=sfb_tmem_store,
    )
    a_, b_, sfa_, sfb_, c_ = to_cute_tensors(a, b, sfa_reordered, sfb_reordered, c, fmt)

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    problem_mnkl = (cutlass.Int32(m), cutlass.Int32(n), cutlass.Int32(k), cutlass.Int32(l))

    print("Compiling DSL kernel...")
    fn = compile_tgv_gemm_nvfp4(
        a_, b_, sfa_, sfb_, c_, cutlass.Float32,
        problem_mnkl=problem_mnkl, sf_vec_size=sf_vec_size,
        cta_m=cta_m, cta_n=cta_n, cta_k=cta_k,
        num_ab_stage=num_ab_stage, num_sfb_tmem_stage=num_sfb_tmem_stage,
        use_pdl=use_pdl, split_k=split_k, sfb_tmem_store=sfb_tmem_store,
    )
    print("Running DSL kernel...")
    fn(a_, sfa_, b_, sfb_, c_, problem_mnkl, stream)
    torch.cuda.synchronize()

    c_ref = reference_scaled_mm(a, b, sfa_simple, sfb_simple, c, (m, n, k, l), fmt)
    c_float = c.to(torch.float32)
    ref_float = c_ref.to(torch.float32)

    max_diff = torch.max(torch.abs(c_float - ref_float)).item()
    mean_diff = torch.mean(torch.abs(c_float - ref_float)).item()
    relative_diff = max_diff / (torch.max(torch.abs(ref_float)).item() + 1e-8)

    print(f"Max absolute difference:  {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Max relative difference:  {relative_diff:.6f}")

    try:
        torch.testing.assert_close(c_float, ref_float, atol=atol, rtol=rtol)
        print("Correctness check PASSED")
        return True
    except AssertionError as e:  # noqa: F821
        print(f"Correctness check FAILED: {e}")
        return False

"""
Testing harness for the blockscaled CUTE DSL TGV GEMM kernel.
Supports NVFP4, MXFP4, and MXFP8 formats.

Usage:
    python tgv_gemm_nvfp4_dsl.py
    python tgv_gemm_nvfp4_dsl.py --format mxfp4
    python tgv_gemm_nvfp4_dsl.py --format mxfp8 --m 3072 --n 8 --k 4096
    python tgv_gemm_nvfp4_dsl.py --benchmark
    python tgv_gemm_nvfp4_dsl.py --format mxfp8 --benchmark --skip-verify
"""
def benchmark(m, n, k, l, fmt=NVFP4, sf_vec_size=None, c_dtype=None,
              cta_m=128, cta_n=8, cta_k=None, num_ab_stage=8,
              num_sfb_tmem_stage=4,
              warmup_iters=10, bench_iters=1000, num_workspaces=20,
              use_pdl=False, split_k=1, sfb_tmem_store=False):
    """Benchmark blockscaled TGV GEMM using CUDA graph with L2 cache thrashing."""
    if sf_vec_size is None:
        sf_vec_size = fmt.sf_vec_size
    if c_dtype is None:
        c_dtype = fmt.c_torch_dtype
    if cta_k is None:
        cta_k = fmt.default_cta_k

    print(f"\n=== {fmt.name.upper()} Benchmark ===")
    print(f"Problem: M={m}, N={n}, K={k}, L={l}")
    print(f"Config: CTA=({cta_m}, {cta_n}, {cta_k}), AB stages={num_ab_stage}, "
          f"SFB TMEM stages={num_sfb_tmem_stage}, PDL={use_pdl}, split_k={split_k}, "
          f"sfb_tmem_store={sfb_tmem_store}")

    num_launched_sms = ((m + cta_m - 1) // cta_m) * ((n + cta_n - 1) // cta_n) * l * split_k
    print(f"Launched SMs: {num_launched_sms}")
    print(f"Workspaces (L2 thrash): {num_workspaces}, Iterations: {bench_iters}")

    workspaces = []
    _torch_refs = []  # prevent GC of backing tensors while pointers are live
    for _ in range(num_workspaces):
        a, b, sfa_r, sfb_r, c, sfa_s, sfb_s = make_blockscaled_tensors(m, n, k, l, fmt, sfb_tmem_store=sfb_tmem_store)
        a_, b_, sfa_, sfb_, c_ = to_cute_tensors(a, b, sfa_r, sfb_r, c, fmt)
        workspaces.append((a_, b_, sfa_, sfb_, c_))
        _torch_refs.append((a, b, sfa_r, sfb_r, c))

    problem_mnkl = (cutlass.Int32(m), cutlass.Int32(n), cutlass.Int32(k), cutlass.Int32(l))

    a0_, b0_, sfa0_, sfb0_, c0_ = workspaces[0]
    compiled_fn = compile_tgv_gemm_nvfp4(
        a0_, b0_, sfa0_, sfb0_, c0_, cutlass.Float32,
        problem_mnkl=problem_mnkl, sf_vec_size=sf_vec_size,
        cta_m=cta_m, cta_n=cta_n, cta_k=cta_k,
        num_ab_stage=num_ab_stage, num_sfb_tmem_stage=num_sfb_tmem_stage,
        use_pdl=use_pdl, split_k=split_k, sfb_tmem_store=sfb_tmem_store,
    )

    graph_stream = torch.cuda.Stream()
    stream = cuda.CUstream(graph_stream.cuda_stream)

    print("Warming up...")
    with torch.cuda.stream(graph_stream):
        for ws_idx in range(num_workspaces):
            a_, b_, sfa_, sfb_, c_ = workspaces[ws_idx]
            compiled_fn(a_, sfa_, b_, sfb_, c_, problem_mnkl, stream)
    torch.cuda.synchronize()

    print("Capturing CUDA graph...")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=graph_stream):
        for i in range(bench_iters):
            ws_idx = i % num_workspaces
            a_, b_, sfa_, sfb_, c_ = workspaces[ws_idx]
            compiled_fn(a_, sfa_, b_, sfb_, c_, problem_mnkl, stream)

    for _ in range(warmup_iters):
        g.replay()
    torch.cuda.synchronize()

    print("Benchmarking...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.profiler.start()
    start_event.record()
    g.replay()
    end_event.record()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / bench_iters
    avg_us = avg_ms * 1000.0

    ops = 2 * m * n * k * l
    gflops = ops / (avg_ms * 1e6)

    bpe = fmt.bytes_per_element
    sf_k = ceil_div(k, sf_vec_size)
    bytes_a = int(m * k * l * bpe)
    bytes_b = int(n * k * l * bpe)
    bytes_sfa = m * sf_k * l  # SF is always 1 byte (FP8)
    bytes_sfb = n * sf_k * l
    bytes_c = m * n * l * 2   # BF16 = 2 bytes
    bytes_per_iter = bytes_a + bytes_b + bytes_sfa + bytes_sfb + bytes_c
    dram_bw_gbps = bytes_per_iter / (avg_ms / 1000.0) / 1e9

    print(f"\n=== Benchmark Results ===")
    print(f"Average time: {avg_ms:.4f} ms ({avg_us:.2f} us)")
    print(f"GFLOPS:       {gflops:.2f}")
    print(f"DRAM BW:      {dram_bw_gbps:.2f} GB/s")
    print(f"=========================")

    return avg_us


def main():
    parser = argparse.ArgumentParser(description="Blockscaled TGV GEMM DSL kernel test harness")
    parser.add_argument("--format", choices=list(FORMAT_MAP.keys()), default="nvfp4",
                        help="Blockscaled format (default: nvfp4)")
    parser.add_argument("--m", type=int, default=3072, help="M dimension")
    parser.add_argument("--n", type=int, default=8, help="N dimension")
    parser.add_argument("--k", type=int, default=4096, help="K dimension")
    parser.add_argument("--l", type=int, default=6, help="Batch dimension") # 1 full wave
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--skip-verify", action="store_true", help="Skip correctness check")
    parser.add_argument("--warmup", type=int, default=10, help="Benchmark warmup (graph replays)")
    parser.add_argument("--iters", type=int, default=10000, help="Benchmark iterations (in graph)")
    parser.add_argument("--workspaces", type=int, default=20, help="Number of workspace sets for L2 thrashing")
    parser.add_argument("--cta-m", type=int, default=128, help="CTA M tile size")
    parser.add_argument("--cta-n", type=int, default=8, help="CTA N tile size")
    parser.add_argument("--cta-k", type=int, default=None, help="CTA K tile size (default: format-dependent)")
    parser.add_argument("--stages", type=int, default=8, help="Number of A/B SMEM pipeline stages")
    parser.add_argument("--sfb-tmem-stages", type=int, default=4, dest="num_sfb_tmem_stage", help="num SFB TMEM stages")
    parser.add_argument("--use-pdl", action="store_true", help="Enable PDL (Programmatic Dependent Launch)")
    parser.add_argument("--split-k", type=int, default=1, help="cluster split-k factor (default: 1, no split)")
    parser.add_argument("--sfb-tmem-store", action="store_true", help="Use lds + tcgen05.st SFB path (default: SMEM-to-TMEM copy)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        sys.exit(1)

    fmt = FORMAT_MAP[args.format]

    # force tcgen05.cp path for MXF4/MXF8 (tcgen05.st not yet validated for these formats)
    if fmt is not NVFP4 and args.sfb_tmem_store:
        print(f"WARNING: --sfb-tmem-store is only validated for nvfp4, forcing tcgen05.cp path for {fmt.name}")
        args.sfb_tmem_store = False

    # default cta_k from format if not explicitly set
    cta_k = args.cta_k if args.cta_k is not None else fmt.default_cta_k

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print(f"Format: {fmt.name}")

    success = True

    if not args.skip_verify:
        success = verify(
            args.m, args.n, args.k, args.l, fmt=fmt,
            cta_m=args.cta_m, cta_n=args.cta_n, cta_k=cta_k,
            num_ab_stage=args.stages,
            num_sfb_tmem_stage=args.num_sfb_tmem_stage,
            use_pdl=args.use_pdl, split_k=args.split_k,
            sfb_tmem_store=args.sfb_tmem_store,
        )
        if not success:
            sys.exit(1)

    if args.benchmark:
        benchmark(
            args.m, args.n, args.k, args.l, fmt=fmt,
            cta_m=args.cta_m, cta_n=args.cta_n, cta_k=cta_k,
            num_ab_stage=args.stages,
            num_sfb_tmem_stage=args.num_sfb_tmem_stage,
            warmup_iters=args.warmup, bench_iters=args.iters,
            num_workspaces=args.workspaces, use_pdl=args.use_pdl,
            split_k=args.split_k, sfb_tmem_store=args.sfb_tmem_store,
        )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()