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
CUTE DSL implementation of the Low-Latency Blackwell GEMM kernel.

This is a low-latency Blackwell GEMM kernel: C = A * B, with optional bias
- A (M, K, L) with K contiguous (UmmaMajor::K)
- B (N, K, L) with K contiguous
- C (M, N, L) with M contiguous

Features:
- TMA loads for A and B matrices (GMEM -> SMEM)
- Block-scaled matrix multiply-accumulate in TMEM
- Multi-stage pipeline for overlapping TMA and MMA
- Non-TMA-store epilogue (TMEM -> RMEM -> GMEM with direct store)
- Optional M-element bias, broadcast across N and L
- 1 SM mode (no 2CTA instructions)
- 1x1 M/N cluster with optional cluster split-K
- Programmatic Dependent Launch support

Blockscaled variant of the Low-Latency Blackwell GEMM kernel.
Default config: CTA_M=128, CTA_N=8, CTA_K=256, DMA_Stage=8
  TypeA=Float4E2M1FN, TypeB=Float4E2M1FN, TypeC=BFloat16, AccType=float
  SF=Float8E4M3FN, sf_vec_size=16
  UmmaMajorA=Major::K, UmmaMajorB=Major::K

Warp assignment (384 threads, 12 warps):
  Warp 0: DMA_A - loads A tiles and scale factors
  Warp 1: DMA_B - loads B tiles and scale factors
  Warp 2: MMA - performs the block-scaled matrix multiply
  Warps 3-6: SFB - stage B scale factors in TMEM for the register-mediated path
  Warp 7: unused
  Warps 8-11: EPILOG - copy accumulators to registers, reduce split-K, and store C
"""

import argparse
from typing import NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.testing
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream, make_ptr
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.utils.gemm.tensor_utils import decode_float4e2m1fn
from cutlass.pipeline import (
    PipelineAsync,
    PipelineAsyncUmma,
    PipelineUmmaAsync,
    CooperativeGroup,
    Agent,
    pipeline_init_arrive,
    pipeline_init_wait,
    make_pipeline_state,
    PipelineUserType,
)


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes.
    For non-persistent static scheduler, CTA id is the work tile info."""

    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32  # kblock range [K_idx_start, K_idx_end)
    K_idx_end: cutlass.Int32


_MAX_AB_STAGES = 12
_SUPPORTED_SPLIT_K = (1, 2, 4, 8)
_SMEM_CAPACITY_BYTES = cutlass.utils.get_smem_capacity_in_bytes("sm_100")


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _smem_bytes(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    cta_k: int,
    num_ab_stage: int,
    num_sfb_tmem_stage: int,
    split_k: int,
) -> int:
    """Mirror the kernel's shared-memory allocator closely enough for tactics."""
    mailbox_elems = max(split_k - 1, 0) * 128 * (8 // split_k)
    cursor = (
        16 * num_ab_stage
        + 16 * num_sfb_tmem_stage
        + 16  # MMA-to-epilogue barriers
        + 8  # TMEM allocation barrier
        + 8  # bias-load barrier
        + 4  # TMEM base pointer
    )
    cursor = _align_up(cursor, 8) + 8 + mailbox_elems * 4

    mixed_width = a_dtype.width != b_dtype.width
    a_smem_width = 8 if mixed_width and a_dtype.width < 8 else a_dtype.width
    b_smem_width = 8 if mixed_width and b_dtype.width < 8 else b_dtype.width
    sf_k = cta_k // sf_vec_size
    buffer_sizes = (
        128 * cta_k * a_smem_width // 8 * num_ab_stage,
        8 * cta_k * b_smem_width // 8 * num_ab_stage,
        128 * sf_k * num_ab_stage,
        8 * sf_k * num_ab_stage,
        128 * sf_k * num_ab_stage,
    )
    for size in buffer_sizes:
        cursor = _align_up(cursor, 128) + size
    return cursor


def autotune_tactics(
    problem_sizes_mnkl: tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
) -> list[tuple[int, int, int, int]]:
    """Enumerate every distinct direct-SFB specialization for a problem.

    A tactic is ``(cta_k, ab_stages, sfb_tmem_stages, split_k)``.  The public
    dispatch paths use the direct SMEM-to-TMEM SFB copy, where the SFB TMEM
    stage count is not consumed by the mainloop, so it is canonically one.
    CTA-K covers every MMA-K multiple that divides K, AB stages cover the full
    supported range, and ``can_implement`` applies the TMEM/SMEM limits.
    """
    _, _, k, _ = problem_sizes_mnkl
    mma_k = 64 if a_dtype.width == b_dtype.width == 4 else 32
    tactics = []
    for cta_k in range(mma_k, k + 1, mma_k):
        if k % cta_k:
            continue
        for num_ab_stage in range(1, _MAX_AB_STAGES + 1):
            for split_k in _SUPPORTED_SPLIT_K:
                tactic = (cta_k, num_ab_stage, 1, split_k)
                if LowLatencyBlockscaledGemmKernel.can_implement(
                    problem_sizes_mnkl,
                    a_dtype,
                    b_dtype,
                    sf_dtype,
                    sf_vec_size,
                    c_dtype,
                    mma_tiler_mnk=(128, 8, cta_k),
                    num_ab_stage=num_ab_stage,
                    num_sfb_tmem_stage=1,
                    split_k=split_k,
                ):
                    tactics.append(tactic)

    default_cta_k = 4 * mma_k
    return sorted(
        tactics,
        key=lambda tactic: (
            tactic != (default_cta_k, 8, 1, 1),
            abs(tactic[0] - default_cta_k),
            abs(tactic[1] - 8),
            tactic[3],
            tactic[0],
            tactic[1],
        ),
    )


class LowLatencyBlockscaledGemmKernel:
    """Low-latency, warp-specialized Blackwell block-scaled GEMM.

    The 12 warps are assigned to two TMA load roles, one MMA role, four
    optional scale-factor producer roles, four epilogue roles, and one unused
    warp.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        mma_tiler_mnk: tuple[int, int, int] = (128, 8, 256),
        num_ab_stage: int = 8,
        num_sfb_tmem_stage: int = 4,
        sf_vec_size: int = 16,
        use_pdl: bool = False,
        pdl_count: int = -1,
        split_k: int = 1,
        sfb_tmem_store: bool = False,
        use_scale: bool = False,
        use_bias: bool = False,
    ):
        """Initialize the Low-Latency blockscaled GEMM kernel configuration.

        :param acc_dtype: Data type for the MMA accumulator (split-K requires Float32).
        :type acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mnk: CTA tile shape (M, N, K).
        :type mma_tiler_mnk: tuple[int, int, int]
        :param num_ab_stage: Number of A/B SMEM pipeline stages.
        :type num_ab_stage: int
        :param num_sfb_tmem_stage: Number of staged SFB TMEM buffers.
        :type num_sfb_tmem_stage: int
        :param sf_vec_size: Scale factor vector size (16 for NVFP4, 32 for MX).
        :type sf_vec_size: int
        :param use_pdl: Enable Programmatic Dependent Launch.
        :type use_pdl: bool
        :param pdl_count: K-tile index to launch dependent grids at; -1 launches at the end.
        :type pdl_count: int
        :param split_k: Cluster split-K factor (1 = no split).
        :type split_k: int
        :param sfb_tmem_store: Use the register-mediated SFB staging path
            instead of the direct SMEM-to-TMEM copy path.
        :type sfb_tmem_store: bool
        :param use_scale: Apply one output scale in accumulator precision.
        :type use_scale: bool
        :param use_bias: Add an M-element bias in the epilogue, broadcast across N and L.
        :type use_bias: bool
        """
        if split_k > 1 and acc_dtype is not cutlass.Float32:
            raise ValueError(
                "block-scaled split-k requires acc_dtype=cutlass.Float32 because "
                "SM100 block-scaled MMA produces Float32 accumulators"
            )

        cta_m, cta_n, cta_k = mma_tiler_mnk
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
        self.use_scale = use_scale
        self.use_bias = use_bias

        # Size the distributed shared memory mailbox used by split-K
        self._mailbox_elems_per_thread = (cta_m * cta_n) // 128
        self._shard_elems_per_thread = self._mailbox_elems_per_thread // max(split_k, 1)
        if split_k > 1 and self._mailbox_elems_per_thread % split_k != 0:
            raise ValueError(
                f"elems_per_thread={self._mailbox_elems_per_thread} must be "
                f"divisible by split_k={split_k}"
            )
        self._mailbox_total_elems = (
            max(split_k - 1, 0) * 128 * self._shard_elems_per_thread
        )
        self._mailbox_tx_per_sender = 128 * self._shard_elems_per_thread * 4
        self._mailbox_tx_total = max(split_k - 1, 0) * self._mailbox_tx_per_sender

        # Fixed configuration
        self.threads_per_cta = 384  # 12 warps (3 active + 4 SFB + 1 unused + 4 epilog)
        self.use_2cta_instrs = False  # 1 SM mode
        self.cluster_shape_mn = (1, 1)  # No multicast, 1x1 cluster
        self.cta_group = tcgen05.CtaGroup.ONE

    def _setup_attributes(self):
        """Set up derived config."""
        mma_tiler_mn = (self.cta_m, self.cta_n)

        # Build the block-scaled MMA atom
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            mma_tiler_mn,
        )
        assert self.cta_group == tcgen05.CtaGroup.ONE

        # Number of MMA instructions along the CTA K tile
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        # Shared SFB layout constants used by the SFB and MMA warps
        self._sf_atom_mn = 32
        self._sf_per_mma_k = mma_inst_shape_k // self.sf_vec_size
        self._num_n_atoms = 2  # two columns for register-mediated SFB staging
        self._sfb_lane_stride = 1 << 18
        self._num_sfa_tmem_cols = (
            self.cta_m // self._sf_atom_mn
        ) * self.mma_inst_tile_k
        if cutlass.const_expr(self.sfb_tmem_store):
            self._sfb_tmem_cols_per_stage = self.mma_inst_tile_k * self._num_n_atoms
        else:
            self._sfb_tmem_cols_per_stage = self.mma_inst_tile_k * 4  # standard padded

        # Keep the accumulator and staged scale factors within half of TMEM
        total_tmem_cols = (
            self.cta_n  # accumulator
            + self._num_sfa_tmem_cols  # SFA
            + self._sfb_tmem_cols_per_stage * self.num_sfb_tmem_stage  # SFB (staged)
        )
        assert total_tmem_cols <= 256, (
            f"TMEM column budget exceeded: {total_tmem_cols} > 256 "
            f"(acc={self.cta_n}, sfa={self._num_sfa_tmem_cols}, "
            f"sfb={self._sfb_tmem_cols_per_stage}*{self.num_sfb_tmem_stage}="
            f"{self._sfb_tmem_cols_per_stage * self.num_sfb_tmem_stage})"
        )

        # Map cluster ranks to (V, M, N, K); split-K varies the final coordinate
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, self.split_k)),
            (tiled_mma.thr_id.shape,),
        )

        # Create SMEM layouts for A and B with swizzle
        # sA_layout: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            (self.cta_m, self.cta_n, self.cta_k),
            self.smem_alloc_a_dtype,
            self.num_ab_stage,
        )
        # sB_layout: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            (self.cta_m, self.cta_n, self.cta_k),
            self.smem_alloc_b_dtype,
            self.num_ab_stage,
        )

        # SFA SMEM layout: ((Atom_M, Atom_K), MMA_M, MMA_K, STAGE)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            (self.cta_m, self.cta_n, self.cta_k),
            self.sf_vec_size,
            self.num_ab_stage,
        )
        # Compact layout for register-mediated SFB staging
        sfb_n = self.cta_n
        sfb_k = self.cta_k // self.sf_vec_size
        self._sfb_smem_layout_flat = cute.make_layout(
            (sfb_n, sfb_k, self.num_ab_stage),
            stride=(sfb_k, 1, sfb_n * sfb_k),
        )
        # Padded layout for direct SFB staging
        self._mma_tiler_sfb = (self.cta_m, cute.round_up(self.cta_n, 128), self.cta_k)
        self._tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (self.cta_m, cute.round_up(self.cta_n, 128)),
        )
        self._sfb_smem_layout_padded = blockscaled_utils.make_smem_layout_sfb(
            self._tiled_mma_sfb,
            self._mma_tiler_sfb,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        if cutlass.const_expr(self.sfb_tmem_store):
            self.sfb_smem_layout_staged = self._sfb_smem_layout_flat
        else:
            self.sfb_smem_layout_staged = self._sfb_smem_layout_padded

        # Accumulator shape for TMEM allocation size calculation
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        scale_ptr: Optional[cute.Pointer],
        bias_ptr: Optional[cute.Pointer],
        problem_mnkl: Tuple,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM kernel warp specialized for DMA_A, DMA_B, MMA, SFB, and EPILOG.
        - DMA_A: loads A tiles via TMA
        - DMA_B: loads B tiles via TMA
        - MMA: performs the block-scaled matrix multiply
        - SFB: stages SFB into TMEM through registers (sfb_tmem_store only)
        - EPILOG: TMEM->RMEM->GMEM store

        :param a_ptr: Pointer to the A operand.
        :type a_ptr: cute.Pointer
        :param sfa_ptr: Pointer to the SFA operand.
        :type sfa_ptr: cute.Pointer
        :param b_ptr: Pointer to the B operand.
        :type b_ptr: cute.Pointer
        :param sfb_ptr: Pointer to the SFB operand.
        :type sfb_ptr: cute.Pointer
        :param c_ptr: Pointer to the C operand.
        :type c_ptr: cute.Pointer
        :param scale_ptr: Pointer to the optional FP32 output scale.
        :type scale_ptr: Optional[cute.Pointer]
        :param bias_ptr: Pointer to the optional M-element bias, or None.
        :type bias_ptr: Optional[cute.Pointer]
        :param problem_mnkl: Problem shape (M, N, K, L).
        :type problem_mnkl: Tuple
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        """
        m, n, k, l = problem_mnkl
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((m, k, l), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout((m, n, l), order=(0, 1, 2))
        )
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.c_dtype = c_ptr.value_type
        if cutlass.const_expr(self.use_scale):
            assert scale_ptr is not None
            scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        else:
            assert scale_ptr is None
            scale = None
        if cutlass.const_expr(self.use_bias):
            assert bias_ptr is not None
            bias = cute.make_tensor(
                bias_ptr,
                cute.make_layout((m, n, l), stride=(1, 0, 0)),
            )
            self.bias_dtype = bias_ptr.value_type
        else:
            assert bias_ptr is None
            bias = None
            self.bias_dtype = self.c_dtype
        self.sf_dtype = sfa_ptr.value_type
        self.mxf8f6f4 = self.needs_unpack_tma(self.a_dtype, self.b_dtype)
        self.smem_alloc_a_dtype = (
            cutlass.Int8 if self.mxf8f6f4 and self.a_dtype.width < 8 else self.a_dtype
        )
        self.smem_alloc_b_dtype = (
            cutlass.Int8 if self.mxf8f6f4 and self.b_dtype.width < 8 else self.b_dtype
        )
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = cutlass.utils.LayoutEnum.from_tensor(c)
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb = cute.make_tensor(
                sfb_ptr,
                cute.make_ordered_layout(
                    (n, cute.assume(k // self.sf_vec_size, 16), l), order=(1, 0, 2)
                ),
            )
        else:
            # Give each CTA N tile a separate 128-row SFB atom
            n_padded_sfb = cute.ceil_div(n, self.cta_n) * cute.round_up(self.cta_n, 128)
            b_shape_for_sfb = cute.make_ordered_layout(
                (n_padded_sfb, k, l), order=(1, 0, 2)
            ).shape
            sfb = cute.make_tensor(
                sfb_ptr,
                blockscaled_utils.tile_atom_to_shape_SF(
                    b_shape_for_sfb, self.sf_vec_size
                ),
            )
        self._setup_attributes()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.cta_m, self.cta_n),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_alloc_a_dtype
            if self.mxf8f6f4 and self.a_dtype.width < 8
            else None,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_alloc_b_dtype
            if self.mxf8f6f4 and self.b_dtype.width < 8
            else None,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb_smem_layout = cute.slice_(self._sfb_smem_layout_flat, (None, None, 0))
            sfb_k = self.cta_k // self.sf_vec_size
            sfb_tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
                sfb_tma_op,
                sfb,
                sfb_smem_layout,
                (self.cta_n, sfb_k),
            )
        else:
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, self._tiled_mma_sfb.thr_id
            )
            sfb_smem_layout = cute.slice_(
                self._sfb_smem_layout_padded, (None, None, None, 0)
            )
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                sfb,
                sfb_smem_layout,
                self._mma_tiler_sfb,
                self._tiled_mma_sfb,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        # Count input bytes before unpacking mixed-width operands
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
            tiled_mma,
            self._tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c,
            scale,
            bias,
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

    @staticmethod
    def needs_unpack_tma(
        a_dtype: Type[cutlass.Numeric], b_dtype: Type[cutlass.Numeric]
    ) -> bool:
        """Return whether mixed-width operands use byte containers in shared memory.

        Mixed-width block-scaled MMA requires uniform one-byte containers. The
        load path therefore unpacks a sub-byte operand when A and B widths
        differ.
        """
        return a_dtype.width != b_dtype.width

    @staticmethod
    def can_implement(
        problem_sizes_mnkl: tuple[int, int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mnk: tuple[int, int, int | None] = (128, 8, None),
        num_ab_stage: int = 8,
        num_sfb_tmem_stage: int = 4,
        split_k: int = 1,
        sfb_tmem_store: bool = False,
    ) -> bool:
        """Return whether the kernel supports the problem and configuration."""
        operand_dtypes = {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }
        if a_dtype not in operand_dtypes or b_dtype not in operand_dtypes:
            return False
        if c_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
            return False
        is_nvfp4 = sf_dtype is cutlass.Float8E4M3FN
        if is_nvfp4:
            if not (
                sf_vec_size == 16
                and a_dtype is cutlass.Float4E2M1FN
                and b_dtype is cutlass.Float4E2M1FN
            ):
                return False
        elif sf_dtype is cutlass.Float8E8M0FNU:
            if sf_vec_size != 32:
                return False
        else:
            return False

        if sfb_tmem_store and not is_nvfp4:
            return False

        m, n, k, l = problem_sizes_mnkl
        if min(m, n, k, l) <= 0:
            return False

        cta_m, cta_n, cta_k = mma_tiler_mnk
        mma_k = 64 if a_dtype.width == 4 and b_dtype.width == 4 else 32
        if cta_k is None:
            cta_k = 4 * mma_k
        if (cta_m, cta_n) != (128, 8) or cta_k <= 0 or k % cta_k != 0:
            return False
        if cta_k % mma_k != 0:
            return False
        # MX block-scaled TMA layouts are built from four 32-wide MMA-K atoms.
        # Smaller/non-128-multiple tiles fail the DSL's CTA V-map equivalence.
        if sf_vec_size == 32 and cta_k % 128 != 0:
            return False
        if (
            num_ab_stage <= 0
            or num_ab_stage > _MAX_AB_STAGES
            or num_sfb_tmem_stage <= 0
            or split_k <= 0
        ):
            return False

        output_values_per_thread = (cta_m * cta_n) // 128
        if cta_m * cta_n % 128 != 0:
            return False
        if output_values_per_thread % split_k != 0:
            return False

        mma_tiles_k = cta_k // mma_k
        sfa_columns = (cta_m // 32) * mma_tiles_k
        sfb_columns_per_stage = mma_tiles_k * (2 if sfb_tmem_store else 4)
        tmem_columns = cta_n + sfa_columns + sfb_columns_per_stage * num_sfb_tmem_stage
        if cta_m % 32 != 0 or tmem_columns > 256:
            return False
        return (
            _smem_bytes(
                a_dtype,
                b_dtype,
                sf_vec_size,
                cta_k,
                num_ab_stage,
                num_sfb_tmem_stage,
                split_k,
            )
            <= _SMEM_CAPACITY_BYTES
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor for A
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor for B
        tma_atom_sfa: cute.CopyAtom,
        sfa: cute.Tensor,  # SFA TMA coordinate tensor
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,  # SFB TMA coordinate tensor
        mC_mnl: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        mScale: Optional[cute.Tensor],  # single output scale
        mBias_mnl: Optional[cute.Tensor],  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        b_smem_layout_staged: cute.ComposedLayout,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_flat_staged: cute.Layout,  # compact 3D SFB layout
        sfb_smem_layout_padded_staged: cute.Layout,  # padded 4D SFB layout
    ):
        """GPU device kernel: SMEM alloc, barrier init, warp dispatch.

        384 threads, 12 warps: 0=DMA_A, 1=DMA_B, 2=MMA, 3-6=SFB
        (register-mediated path only), 7=unused, 8-11=EPILOG. Tensor
        partitioning is done inside the warp functions.
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

        # Derive the split rank and batch index from the cluster topology
        _, _, split_rank = cute.arch.block_in_cluster_idx()
        _, _, l_idx = cute.arch.cluster_idx()

        # Assign this CTA its split-K range
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
        # tma_partition uses the M/N coordinates to build multicast masks
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # SMEM allocation (SharedStorage struct for barriers + TMEM ptr)
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage

        # AB full barriers track both arrivals and TMA transaction bytes. The
        # remaining barriers use plain arrive/wait synchronization
        MailboxElems = self._mailbox_total_elems  # (split_k-1) * 128 * elems_per_thread

        @cute.struct
        class SharedStorage:
            # AB pipeline barriers: full[0..DMA_Stage-1] + empty[DMA_Stage..2*DMA_Stage-1]
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, DMA_Stage * 2]
            # SFB pipeline barriers (unused on the direct-copy path)
            sfb_pipeline_bars: cute.struct.MemRange[cutlass.Int64, SFB_Stage * 2]
            # MMA→epilog pipeline: full[0] + empty[1]
            mma_epilog_bars: cute.struct.MemRange[cutlass.Int64, 2]
            # Standalone barriers coordinate tensor memory and split-K reduction
            tmem_allocation_result_barrier: cutlass.Int64
            tma_epilog_full_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            dsmem_mailbox_barrier: cutlass.Int64
            dsmem_mailbox: cute.struct.MemRange[cutlass.Float32, MailboxElems]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Defer synchronization until every pipeline and standalone barrier is ready
        # Two TMA producer warps feed one MMA consumer warp
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=DMA_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage

        # One-stage MMA-to-epilogue completion pipeline
        mma_epilog_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.mma_epilog_bars.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        mma_epilog_full_bar = mma_epilog_pipeline.sync_object_full.barrier_storage

        # Raw barrier pointers
        tmem_alloc_result_bar = storage.tmem_allocation_result_barrier.ptr
        tma_epilog_full_bar = storage.tma_epilog_full_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        mailbox_mbar = storage.dsmem_mailbox_barrier.ptr
        dsmem_mailbox_ptr = storage.dsmem_mailbox.data_ptr()
        # The register-mediated path has a separate SFB pipeline
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb_pipeline = PipelineAsyncUmma.create(
                barrier_storage=storage.sfb_pipeline_bars.data_ptr(),
                num_stages=SFB_Stage,
                producer_group=CooperativeGroup(Agent.Thread, 4),
                consumer_group=CooperativeGroup(Agent.Thread),
                defer_sync=True,
            )
            sfb_full_bar = sfb_pipeline.sync_object_full.barrier_storage
            sfb_empty_bar = sfb_pipeline.sync_object_empty.barrier_storage
        else:
            sfb_full_bar = storage.sfb_pipeline_bars.data_ptr()
            sfb_empty_bar = sfb_full_bar + SFB_Stage

        # Initialize standalone TMEM and split-K barriers
        if warp_idx == 0:
            with cute.arch.elect_one():
                # The TMEM barrier covers allocation and release across MMA,
                # epilogue, and the optional SFB warps
                if cutlass.const_expr(self.sfb_tmem_store):
                    cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128 + 128)
                else:
                    cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128)
                if cutlass.const_expr(self.use_bias):
                    # All 32 B-load threads arrive after issuing activation loads
                    cute.arch.mbarrier_init(tma_epilog_full_bar, 32)
                if self.split_k > 1:
                    cute.arch.mbarrier_init(mailbox_mbar, 1)

        # Publish barrier initialization to every warp
        cluster_layout = (
            cute.make_layout((*self.cluster_shape_mn, self.split_k))
            if self.split_k > 1
            else None
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_layout)

        # SMEM Tensor Allocation
        # sA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) with swizzle
        sA = smem.allocate_tensor(
            element_type=self.smem_alloc_a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # sB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) with swizzle
        sB = smem.allocate_tensor(
            element_type=self.smem_alloc_b_dtype,
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
        # Allocate both SFB layouts to keep the configurations type-compatible
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

        # Warp dispatch
        # Each warp specializes in a different role:
        #   Warp 0 (threads 0-31):      TMA load A
        #   Warp 1 (threads 32-63):     TMA load B
        #   Warp 2 (threads 64-95):     MMA   — performs block-scaled MMA
        #   Warps 3-6 (threads 96-223): SFB producers
        #   Warp 7 (threads 224-255):   unused
        #   Warps 8-11 (threads 256-383): epilogue
        if warp_idx == 0:
            self.dma_a_warp(
                ab_full_bar,
                ab_empty_bar,
                tma_atom_a,
                mA_mkl,
                sA,
                tma_atom_sfa,
                sfa,
                sSFA,
                tiled_mma,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                work_tile_info,
                k_tile_count,
            )
        elif warp_idx == 1:
            sSFB = sSFB_flat if cutlass.const_expr(self.sfb_tmem_store) else sSFB_padded
            self.dma_b_warp(
                ab_full_bar,
                ab_empty_bar,
                tma_epilog_full_bar,
                tma_atom_b,
                mB_nkl,
                sB,
                tma_atom_sfb,
                mSFB_nkl,
                sSFB,
                tiled_mma,
                tiled_mma_sfb,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                work_tile_info,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                ab_full_bar,
                ab_empty_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                sfb_full_bar,
                sfb_empty_bar,
                tiled_mma,
                sA,
                sB,
                sSFA,
                sSFB_padded,
                sfa_smem_layout_staged,
                sfb_smem_layout_padded_staged,
                tmem_base_smem_ptr,
                k_tile_count,
            )
        elif warp_idx >= 8:
            epi_tid = tidx - 256
            self.epilog_warp(
                tma_epilog_full_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                tiled_mma,
                mC_mnl,
                mScale,
                mBias_mnl,
                tmem_base_smem_ptr,
                mailbox_mbar,
                dsmem_mailbox_ptr,
                epi_tid,
                mma_tile_coord_v,
                work_tile_info,
                split_rank,
                k_tile_count,
            )

        # Register-mediated path only: warps 3-6 stage SFB into TMEM (idle on direct copy)
        if cutlass.const_expr(self.sfb_tmem_store):
            if warp_idx >= 3 and warp_idx <= 6:
                self.sfb_warp(
                    ab_full_bar,
                    sfb_full_bar,
                    sfb_empty_bar,
                    tmem_alloc_result_bar,
                    tmem_base_smem_ptr,
                    sSFB_flat,
                    k_tile_count,
                )

        # Final sync
        cute.arch.barrier()
        return

    @cute.jit
    def dma_a_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        sSFA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: WorkTileInfo,
        k_tile_count: cutlass.Int32,
    ):
        """DMA_A warp: loads A tiles via TMA."""
        DMA_Stage = self.num_ab_stage

        # Tile mA_mkl (Gemm_M, Gemm_K, Gemm_L) into CTA-level tiles
        # gA_mkl: (CTA_M, CTA_K, Tiles_M, Tiles_K, Gemm_L) — all tiles
        gA_mkl = cute.local_tile(mA_mkl, (self.cta_m, self.cta_k), (None, None, None))
        # In 1SM mode, tiled_mma has one partition
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, Tiles_M, Tiles_K, Gemm_L)
        tCgA = thr_mma.partition_A(gA_mkl)

        # A tiles multicast along the cluster N dimension. group_modes folds the
        # MMA modes into one TMA transfer; a 1x1 cluster reduces to a local copy
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],  # this CTA's N coord within cluster
            a_cta_layout,
            cute.group_modes(
                sA, 0, 3
            ),  # (((Mma_M, Mma_K), NumMma_M, NumMma_K), DMA_Stage)
            cute.group_modes(
                tCgA, 0, 3
            ),  # (((Mma_M, Mma_K), NumMma_M, NumMma_K), Tiles_M, Tiles_K, Gemm_L)
        )
        # tAsA: ((TMA, NumTma_K), DMA_Stage) — SMEM destination for each pipeline stage
        # tAgA: ((TMA, NumTma_K), Tiles_M, Tiles_K, Gemm_L) — GMEM source tiles

        # Slice to this CTA's M tile and batch index, keep K tiles and TMA modes free
        # tAgA after slice: ((TMA, NumTma_K), Tiles_K)
        tAgA = tAgA[(None, work_tile_info.M_idx, None, work_tile_info.L_idx)]

        # SFA partition (same cta_layout as A — SFA multicasts along N)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, (self.cta_m, self.cta_k), (None, None, None)
        )
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

        # Empty barriers start at phase 0, so the producer starts at phase 1
        ab_state = make_pipeline_state(PipelineUserType.Producer, DMA_Stage)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            # wait for MMA to signal SMEM slot empty (auto phase tracking)
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)

            # set transaction bytes on full barrier
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    ab_full_bar + ab_state.index, self.tma_bytes_a
                )

            # tma_partition elects the issuer; another election around cute.copy
            # can prevent participating threads from making progress
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile_global)],
                tAsA[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, k_tile_global)],
                tAsSFA[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index,
            )

            ab_state.advance()  # update phase and index/stage

            # PDL: launch dependents at the computed k_tile, or unconditionally at end
            if self.use_pdl:
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if self.use_pdl:
            cute.arch.griddepcontrol_launch_dependents()

        # Producer tail: keep shared memory alive until MMA releases every stage
        for _k_tile in cutlass.range(DMA_Stage, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)
            ab_state.advance()

    @cute.jit
    def dma_b_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        tma_epilog_full_bar,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,  # SFB TMA coordinate tensor
        sSFB: cute.Tensor,  # SFB SMEM tensor (flat or padded, matching the path)
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,  # only used by the direct-copy (padded) SFB path
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: WorkTileInfo,
        k_tile_count: cutlass.Int32,
    ):
        """DMA_B warp: loads B tiles + SFB scale factors via TMA.

        Only the SFB TMA partition differs between the two SFB paths.
        """
        DMA_Stage = self.num_ab_stage

        # Tile mB_nkl (Gemm_N, Gemm_K, Gemm_L) into CTA-level tiles
        # gB_nkl: (CTA_N, CTA_K, Tiles_N, Tiles_K, Gemm_L)
        gB_nkl = cute.local_tile(mB_nkl, (self.cta_n, self.cta_k), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, Tiles_N, Tiles_K, Gemm_L)
        tCgB = thr_mma.partition_B(gB_nkl)

        # B tiles multicast along the cluster M dimension
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],  # this CTA's M coord within cluster
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        # tBsB: ((TMA, NumTma_K), DMA_Stage)
        # tBgB: ((TMA, NumTma_K), Tiles_N, Tiles_K, Gemm_L)

        # Slice to this CTA's N tile and batch index
        tBgB = tBgB[(None, work_tile_info.N_idx, None, work_tile_info.L_idx)]

        if cutlass.const_expr(self.sfb_tmem_store):
            # Register-mediated path: flat packed SFB layout, partitioned directly
            sfb_k = self.cta_k // self.sf_vec_size
            gSFB_nkl = cute.local_tile(
                mSFB_nkl, (self.cta_n, sfb_k), (None, None, None)
            )
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                0,
                cute.make_layout(1),
                cute.group_modes(sSFB, 0, 2),
                cute.group_modes(gSFB_nkl, 0, 2),
            )
        else:
            # Direct-copy path: padded SFB partitioned like B via tiled_mma_sfb
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
                    ab_full_bar + ab_state.index, self.tma_bytes_b
                )
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile_global)],
                tBsB[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB[(None, k_tile_global)],
                tBsSFB[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index,
            )
            ab_state.advance()

        # Release bias loads after the B-load warp satisfies the PDL dependency
        if cutlass.const_expr(self.use_bias):
            cute.arch.mbarrier_arrive(tma_epilog_full_bar)

        # Producer tail: keep shared memory alive until MMA releases every stage
        for _k_tile in cutlass.range(DMA_Stage, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)
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
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
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
    ):
        """Move one SFB subpartition through registers into tensor memory."""
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage
        # Join the TMEM allocation barrier before reading the base address
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        # Offset this subpartition past the accumulator and SFA storage
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self._num_sfa_tmem_cols + self.cta_n, dtype=self.sf_dtype
        )

        # staged TMEM layout: add SFB_Stage as outermost mode
        sfb_kblock_stride = self._num_n_atoms * self._sf_per_mma_k
        sfb_stage_stride = self.mma_inst_tile_k * sfb_kblock_stride
        sp_tmem_staged = cute.make_tensor(
            sfb_tmem_ptr,
            cute.make_layout(
                (
                    (self._sf_atom_mn, (1, self._sf_per_mma_k)),
                    1,
                    self.mma_inst_tile_k,
                    SFB_Stage,
                ),
                stride=(
                    (self._sfb_lane_stride, (0, 1)),
                    0,
                    sfb_kblock_stride,
                    sfb_stage_stride,
                ),
            ),
        )

        # single-stage slice for copy plan (layout same across stages)
        sp_tmem_single = sp_tmem_staged[(None, None, None, 0)]
        sp_tmem_compact_single = cute.filter_zeros(sp_tmem_single)
        # staged compact for destination partitioning
        sp_tmem_compact_staged = cute.filter_zeros(sp_tmem_staged)

        tidx_in_warp = cute.arch.lane_idx()
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(1)),
            self.sf_dtype,
        )
        tmem_store_tiled = tcgen05.make_tmem_copy(
            tmem_store_atom, sp_tmem_compact_single
        )
        tmem_store_thr = tmem_store_tiled.get_slice(tidx_in_warp)
        tStDst_staged = tmem_store_thr.partition_D(sp_tmem_compact_staged)
        tStSrc = tmem_store_thr.partition_S(sp_tmem_compact_single)
        rSFB = cute.make_rmem_tensor(tStSrc.shape, self.sf_dtype)

        # Observe the AB pipeline and produce into the SFB pipeline
        ab_observer = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
        sfb_state = make_pipeline_state(PipelineUserType.Producer, SFB_Stage)

        for _k_tile in cutlass.range(k_tile_count, unroll=1):
            # SFB warps observe AB completion but do not release the slot
            cute.arch.mbarrier_wait(ab_full_bar + ab_observer.index, ab_observer.phase)

            # Load SFB from shared memory into registers
            sSFB_stage = sSFB[(None, None, ab_observer.index)]
            smem_row = tidx_in_warp
            if smem_row >= self.cta_n:
                smem_row = cutlass.Int32(0)
            for i in cutlass.range(cute.size(rSFB)):
                rSFB[i] = sSFB_stage[smem_row + i * self.cta_n]

            # Wait until MMA releases this SFB stage
            cute.arch.mbarrier_wait(sfb_empty_bar + sfb_state.index, sfb_state.phase)

            # Store the register fragment in its staged tensor memory destination
            cute.copy(
                tmem_store_tiled,
                rSFB,
                tStDst_staged[(None, None, None, None, sfb_state.index)],
            )
            cute.arch.fence_view_async_tmem_store()

            # Publish the completed SFB stage to MMA
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(sfb_full_bar + sfb_state.index)

            ab_observer.advance()
            sfb_state.advance()

        # Join the TMEM release phase
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
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_padded_staged: cute.Layout,
        tmem_base_smem_ptr,
        k_tile_count: cutlass.Int32,
    ):
        """Perform block-scaled MMA with the staged scale factors."""
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage

        # A and B fragments are SMEM descriptors consumed by cute.gemm
        # tCrA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — SMEM descriptors for A
        tCrA = tiled_mma.make_fragment_A(sA)
        # tCrB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — SMEM descriptors for B
        tCrB = tiled_mma.make_fragment_B(sB)

        # Build the accumulator layout before attaching its TMEM base pointer
        # acc_shape / tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # Reserve half of TMEM so the next CTA can overlap allocation
        num_tmem_cols = 256  # SM100_TMEM_CAPACITY_COLUMNS / 2
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_smem_ptr)

        # All 32 MMA threads publish the allocation without waiting
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # Relinquish TMEM allocation lock early so that the next prefetch CTA can be launched
        cute.arch.relinquish_tmem_alloc_permit()

        # Retrieve TMEM ptr from SMEM and create the accumulator tensor view
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # SFA/SFB TMEM tensors and S2T copy partition
        sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + self.cta_n, dtype=self.sf_dtype)
        sfa_mem_single = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, tiled_mma.shape_mnk, self.sf_vec_size, sfa_mem_single
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self._num_sfa_tmem_cols + self.cta_n, dtype=self.sf_dtype
        )

        # SFA: SMEM-to-TMEM copy
        (
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t,
            tCtSFA_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)

        # MMA mainloop
        if cutlass.const_expr(self.sfb_tmem_store):
            # Register-mediated staging uses two N atoms per SFB subpartition
            sfb_kblock_stride = self._num_n_atoms * self._sf_per_mma_k
            sfb_stage_stride = self.mma_inst_tile_k * sfb_kblock_stride
            tCtSFB_layout_staged = cute.make_layout(
                (
                    (
                        (
                            (self._sf_atom_mn, self._num_n_atoms),
                            128 // self._sf_atom_mn,
                        ),
                        (self.sf_vec_size, self._sf_per_mma_k),
                    ),
                    1,
                    self.mma_inst_tile_k,
                    SFB_Stage,
                ),
                stride=(
                    (
                        (
                            (self._sfb_lane_stride, self._sf_per_mma_k),
                            self._sf_atom_mn * self._sfb_lane_stride,
                        ),
                        (0, 1),
                    ),
                    0,
                    sfb_kblock_stride,
                    sfb_stage_stride,
                ),
            )
            tCtSFB_staged = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout_staged)

            # Register-mediated SFB mainloop
            ab_state = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
            sfb_state = make_pipeline_state(PipelineUserType.Consumer, SFB_Stage)
            smem_wait_done = cutlass.Boolean(False)
            sfb_wait_done = cutlass.Boolean(False)

            for k_tile in cutlass.range(k_tile_count):
                if ~smem_wait_done:
                    cute.arch.mbarrier_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                old_ab_index = ab_state.index
                ab_state.advance()

                old_sfb_index = sfb_state.index
                old_sfb_phase = sfb_state.phase
                sfb_state.advance()

                smem_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    smem_wait_done = cute.arch.mbarrier_try_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                # Copy SFA from shared memory to tensor memory
                s2t_stage_coord = (None, None, None, None, old_ab_index)
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord],
                    tCtSFA_compact_s2t,
                )

                # Wait for the SFB warps to finish this TMEM stage
                if ~sfb_wait_done:
                    cute.arch.mbarrier_wait(sfb_full_bar + old_sfb_index, old_sfb_phase)

                # Select the staged SFB tensor used by this GEMM
                tCtSFB_mma = tCtSFB_staged[(None, None, None, old_sfb_index)]

                # A*SFA * B*SFB -> Acc
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                tile_crd = (None, None, None, old_ab_index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    [tCrA[tile_crd], tCtSFA],
                    [tCrB[tile_crd], tCtSFB_mma],
                    tCtAcc,
                )

                with cute.arch.elect_one():
                    # Release the SFB and operand pipeline stages
                    tcgen05.commit(sfb_empty_bar + old_sfb_index, None, self.cta_group)
                    tcgen05.commit(ab_empty_bar + old_ab_index, None, self.cta_group)

                sfb_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    sfb_wait_done = cute.arch.mbarrier_try_wait(
                        sfb_full_bar + sfb_state.index, sfb_state.phase
                    )
        else:
            # Direct staging reuses one standard SFB tensor memory stage
            sfb_smem_single = cute.slice_(
                sfb_smem_layout_padded_staged, (None, None, None, 0)
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, tiled_mma.shape_mnk, self.sf_vec_size, sfb_smem_single
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Configure the direct SFB copy from shared to tensor memory
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            # Direct SFB staging mainloop
            ab_state = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
            smem_wait_done = cutlass.Boolean(False)

            for k_tile in cutlass.range(k_tile_count):
                if ~smem_wait_done:
                    cute.arch.mbarrier_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                old_ab_index = ab_state.index
                ab_state.advance()

                smem_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    smem_wait_done = cute.arch.mbarrier_try_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                # Copy SFA from shared memory to tensor memory
                s2t_stage_coord = (None, None, None, None, old_ab_index)
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord],
                    tCtSFA_compact_s2t,
                )

                # Copy SFB directly from shared memory to tensor memory
                cute.copy(
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t[s2t_stage_coord],
                    tCtSFB_compact_s2t,
                )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                tile_crd = (None, None, None, old_ab_index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    [tCrA[tile_crd], tCtSFA],
                    [tCrB[tile_crd], tCtSFB],
                    tCtAcc,
                )

                with cute.arch.elect_one():
                    tcgen05.commit(ab_empty_bar + old_ab_index, None, self.cta_group)

        # Signal that the accumulator is ready for the epilogue
        with cute.arch.elect_one():
            tcgen05.commit(mma_epilog_full_bar, None, self.cta_group)

        # Release the tensor memory allocation
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols)

    @cute.jit
    def epilog_warp(
        self,
        tma_epilog_full_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        tiled_mma: cute.TiledMma,
        mC_mnl: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        mScale: Optional[cute.Tensor],  # single output scale
        mBias_mnl: Optional[cute.Tensor],  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        tmem_base_smem_ptr,
        mailbox_mbar,  # local transaction barrier signaled by peer CTAs
        dsmem_mailbox_ptr,  # SMEM pointer for DSMEM stores and reads
        epi_tid: cutlass.Int32,  # thread id within epilog warps (0-127)
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: WorkTileInfo,
        split_rank: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        """EPILOG warp: TMEM -> RMEM -> type convert -> GMEM."""

        # Get this CTA's output tile
        # gC_mnl: (CTA_M, CTA_N, Tiles_M, Tiles_N, Gemm_L)
        gC_mnl = cute.local_tile(mC_mnl, (self.cta_m, self.cta_n), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgC: ((Mma_M, Mma_N), NumMma_M, NumMma_N, Tiles_M, Tiles_N, Gemm_L)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Recreate the MMA accumulator layout; attach its TMEM base pointer below
        # acc_shape / tCtAcc_fake: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # All 128 epilogue threads arrive, then wait for MMA to publish TMEM
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        # Update TMEM base ptr of the accumulator tensor view
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        # tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N) — TMEM accumulator view
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # Select the tensor-to-register copy atom for this CTA
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            (self.cta_m, self.cta_n),
            self.use_2cta_instrs,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc[((None, None), 0, 0)]
        )
        # Epilogue tid is 0-127 (threads 256-383 offset by 256)
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)

        # tD describes the per-thread tensor-to-register partition
        # tDtAcc: per-subpartition view of the accumulator tensor (TMEM source)
        # Shape: (CpyS, NumCpy_M, NumCpy_N)
        tDtAcc = thr_copy_t2r.partition_S(tCtAcc[((None, None), 0, 0)])

        # Partition C before slicing so its full shape can size the register fragment
        # Shape: (CpyD, NumCpy_M, NumCpy_N, Tiles_M, Tiles_N, Gemm_L)
        # DSL tCgC has extra grid dims; partition with them, create rmem before slicing
        tDgC = thr_copy_t2r.partition_D(tCgC[((None, None), 0, 0, None, None, None)])

        # Allocate register fragments before slicing; slicing changes the rank
        # Shape: (CpyD, NumCpy_M, NumCpy_N)
        epi_frag_shape = tDgC[(None, None, None, 0, 0, 0)].shape
        # tDrAcc: (CpyD, NumCpy_M, NumCpy_N) — per-thread accumulator in rmem (AccType)
        tDrAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)
        # tDrC: (CpyD, NumCpy_M, NumCpy_N) — per-thread converted output in rmem (TypeC)
        tDrC = cute.make_rmem_tensor(epi_frag_shape, self.c_dtype)

        # Slice gmem partition to this CTA's output tile
        # tDgC after slice: (CpyD, NumCpy_M, NumCpy_N) — per-thread GMEM store destinations
        tDgC = tDgC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]

        # Direct stores need explicit bounds checks; TMA loads handle OOB coordinates
        coordC = cute.make_identity_tensor(mC_mnl.shape)  # (M, N, L) -> (m, n, l)
        # Create the local tile of coordC, same tiling as gC
        gCcoord = cute.local_tile(coordC, (self.cta_m, self.cta_n), (None, None, None))
        # tCcC: ((Mma_M, Mma_N), NumMma_M, NumMma_N, Tiles_M, Tiles_N, Gemm_L)
        tCcC = thr_mma.partition_C(gCcoord)
        # tDcC: (CpyD, NumCpy_M, NumCpy_N, Tiles_M, Tiles_N, Gemm_L)
        # — per-thread coordinate tensor, same shape as tDgC but payload is (m,n,l) coord
        tDcC = thr_copy_t2r.partition_D(tCcC[((None, None), 0, 0, None, None, None)])
        # Slice to this CTA's tile: (CpyD, NumCpy_M, NumCpy_N)
        tDcC = tDcC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]
        # Construct predicate tensor: compare each coordinate with problem shape (M, N, L)
        # tDpredC(t) = true if coordinate is in-bounds, false if out-of-bounds
        tDpredC = cute.make_rmem_tensor(
            tDcC.shape, cutlass.Boolean
        )  # (CpyD, NumCpy_M, NumCpy_N)
        for i in range(cute.size(tDpredC)):
            tDpredC[i] = cute.elem_less(tDcC[i], mC_mnl.shape)

        if cutlass.const_expr(self.use_scale):
            scale_value = mScale[0].to(self.acc_dtype)

        # Use C's register mapping for the M-broadcast bias
        if cutlass.const_expr(self.use_bias):
            gBias_mnl = cute.local_tile(
                mBias_mnl, (self.cta_m, self.cta_n), (None, None, None)
            )
            tCgBias = thr_mma.partition_C(gBias_mnl)
            tDgBias = thr_copy_t2r.partition_D(
                tCgBias[((None, None), 0, 0, None, None, None)]
            )
            tDgBias = tDgBias[
                (
                    None,
                    None,
                    None,
                    work_tile_info.M_idx,
                    work_tile_info.N_idx,
                    work_tile_info.L_idx,
                )
            ]
            tDrBias = cute.make_rmem_tensor(epi_frag_shape, self.bias_dtype)
            tDrBiasAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)

            # Wait for the B-load warp to satisfy the bias producer dependency
            cute.arch.mbarrier_wait(tma_epilog_full_bar, 0)
            cute.basic_copy_if(tDpredC, tDgBias, tDrBias)
            tDrBiasAcc.store(tDrBias.load().to(self.acc_dtype))

        # Wait for MMA to finish (PipelineUmmaAsync consumer)
        mma_epilog_state = make_pipeline_state(PipelineUserType.Consumer, 1)
        cute.arch.mbarrier_wait(
            mma_epilog_full_bar + mma_epilog_state.index, mma_epilog_state.phase
        )

        # Copy the accumulator to the 128 epilogue threads' registers
        cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)

        # Fence asynchronous TMEM reads before releasing the allocation
        cute.arch.fence_view_async_tmem_load()

        # Empty K partitions contribute zero to the split-K reduction
        if cutlass.const_expr(self.split_k > 1):
            if k_tile_count == 0:
                tDrAcc.fill(0.0)

        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # Type conversion and GMEM store (with optional DSMEM split-k reduction)
        elems_per_thread = self._mailbox_elems_per_thread
        if self.split_k == 1:
            # Apply the scale in accumulator precision before output conversion
            if cutlass.const_expr(self.use_scale):
                tDrAcc.store(tDrAcc.load() * scale_value)
            # Add the bias in accumulator precision before output conversion
            if cutlass.const_expr(self.use_bias):
                tDrAcc.store(tDrAcc.load() + tDrBiasAcc.load())
            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            cute.basic_copy_if(tDpredC, tDrC, tDgC)
        else:
            # Each CTA reduces and writes one output shard
            shard_ept = self._shard_elems_per_thread

            # Arm the local mailbox before peer stores can arrive
            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mailbox_mbar, self._mailbox_tx_total
                )

            # Complete tensor memory loads before the distributed scatter
            cute.arch.barrier(barrier_id=15, number_of_threads=128)

            for peer in range(self.split_k):
                if peer != split_rank:
                    # Elements in peer's shard: [peer*shard_ept, (peer+1)*shard_ept)
                    shard_start = peer * shard_ept
                    # Skip the peer's own rank when selecting its sender slot
                    sender_idx = split_rank - (
                        1 if split_rank > cutlass.Int32(peer) else 0
                    )
                    mailbox_base = sender_idx * 128 * shard_ept
                    for i in range(shard_ept):
                        cute.arch.store_async_dsmem(
                            dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid,
                            tDrAcc[shard_start + i].bitcast(cutlass.Int32),
                            mailbox_mbar,
                            cutlass.Int32(peer),
                        )

            cute.arch.mbarrier_wait(mailbox_mbar, 0)

            my_shard_start = split_rank * shard_ept
            for s in range(self.split_k - 1):
                mailbox_base = s * 128 * shard_ept
                for i in range(shard_ept):
                    tDrAcc[my_shard_start + i] = (
                        tDrAcc[my_shard_start + i]
                        + (dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid).load()
                    )

            # Each output belongs to exactly one shard, so scale and bias apply once.
            if cutlass.const_expr(self.use_scale):
                tDrAcc.store(tDrAcc.load() * scale_value)
            if cutlass.const_expr(self.use_bias):
                tDrAcc.store(tDrAcc.load() + tDrBiasAcc.load())

            for i in range(elems_per_thread):
                if i < my_shard_start or i >= my_shard_start + shard_ept:
                    tDpredC[i] = cutlass.Boolean(False)

            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            cute.basic_copy_if(tDpredC, tDrC, tDgC)


# Blockscaled helpers


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
    atom_k = 4  # Mma_K / sf_vec_size = 64 / 16

    rest_mn = ceil_div(mn, atom_mn)
    rest_k = ceil_div(sf_k, atom_k)
    padded_mn = rest_mn * atom_mn
    padded_k = rest_k * atom_k

    original_dtype = sf.dtype
    l = sf.shape[2]

    sf_f32 = sf.to(torch.float32)
    if padded_mn != mn or padded_k != sf_k:
        sf_f32 = torch.nn.functional.pad(
            sf_f32,
            (0, 0, 0, padded_k - sf_k, 0, padded_mn - mn),
        )

    sf_f32 = sf_f32.view(rest_mn, atom_mn, rest_k, atom_k, l)
    sf_f32 = sf_f32.view(rest_mn, 4, 32, rest_k, atom_k, l)

    mma_shape = (l, rest_mn, rest_k, 32, 4, atom_k)
    out = torch.zeros(mma_shape, dtype=original_dtype, device=sf.device)
    out.permute(3, 4, 1, 5, 2, 0).copy_(
        sf_f32.permute(2, 1, 0, 4, 3, 5).to(original_dtype)
    )

    return out


def reorder_scale_factors_per_cta(sf, cta_mn, k, sf_vec_size):
    """Pack each CTA MN tile into a separate 128-row scale-factor atom."""
    atom_mn = 128
    atom_k = 4
    if cta_mn <= 0 or cta_mn > atom_mn:
        raise ValueError(f"cta_mn must be in [1, {atom_mn}], got {cta_mn}")

    mn, input_sf_k, l = sf.shape
    sf_k = ceil_div(k, sf_vec_size)
    if input_sf_k != sf_k:
        raise ValueError(f"expected {sf_k} scale columns, got {input_sf_k}")

    rest_mn = ceil_div(mn, cta_mn)
    rest_k = ceil_div(sf_k, atom_k)
    padded_mn = rest_mn * cta_mn
    padded_k = rest_k * atom_k

    original_dtype = sf.dtype
    sf_f32 = sf.to(torch.float32)
    if padded_mn != mn or padded_k != sf_k:
        sf_f32 = torch.nn.functional.pad(
            sf_f32,
            (0, 0, 0, padded_k - sf_k, 0, padded_mn - mn),
        )

    # Pad each CTA tile independently
    sf_f32 = sf_f32.view(rest_mn, cta_mn, rest_k, atom_k, l)
    atom_tiles = torch.zeros(
        (rest_mn, atom_mn, rest_k, atom_k, l),
        dtype=torch.float32,
        device=sf.device,
    )
    atom_tiles[:, :cta_mn, :, :, :] = sf_f32
    atom_tiles = atom_tiles.view(rest_mn, 4, 32, rest_k, atom_k, l)

    mma_shape = (l, rest_mn, rest_k, 32, 4, atom_k)
    out = torch.zeros(mma_shape, dtype=original_dtype, device=sf.device)
    out.permute(3, 4, 1, 5, 2, 0).copy_(
        atom_tiles.permute(2, 1, 0, 4, 3, 5).to(original_dtype)
    )
    return out


def _decode_ab_to_f32(x):
    """Decode an A/B operand ``(MN, K_phys, L)`` to dense float32 ``(MN, K, L)``
    on CPU. fp4 operands are unpacked with the DSL's ``decode_float4e2m1fn``;
    fp8 and other native floats cast directly."""
    if x.dtype != torch.float4_e2m1fn_x2:
        return x.cpu().to(torch.float32)
    u8 = x.view(torch.uint8).cpu().permute(0, 2, 1).contiguous()  # (MN, L, K//2)
    mn, l, k_half = u8.shape
    total = u8.numel()
    padded = torch.zeros(1, 2 * total, 1, dtype=torch.uint8)
    padded[0, :total, 0] = u8.reshape(-1)
    decoded = decode_float4e2m1fn(padded).reshape(mn, l, 2 * k_half)  # (MN, L, K)
    return decoded.permute(0, 2, 1).contiguous()  # (MN, K, L)


def compare(a, b, sfa_ref, sfb_ref, c, c_dtype, atol, rtol, scale=None, bias=None):
    """Reference check ``D = scale * (A * SFA) @ (B * SFB) + bias``."""
    a_f32 = _decode_ab_to_f32(a)
    b_f32 = _decode_ab_to_f32(b)
    ref = torch.einsum("mkl,nkl->mnl", a_f32 * sfa_ref, b_f32 * sfb_ref)
    if scale is not None:
        ref = ref * scale.cpu().to(torch.float32)
    if bias is not None:
        # FP32 bias accumulation, broadcast across N and L
        ref = ref + bias.cpu().to(torch.float32)[:, None, None]
    # Match the epilogue: round the f32 reference through the output dtype
    ref = ref.to(cutlass_torch.dtype(c_dtype)).to(torch.float32)
    gpu_result = c.detach().cpu().to(torch.float32)
    torch.testing.assert_close(gpu_result, ref, atol=atol, rtol=rtol)


def make_blockscaled_tensors(
    mnkl,
    a_dtype,
    b_dtype,
    sf_dtype,
    sf_vec_size,
    c_dtype,
    sfb_tmem_store=False,
    cta_n=8,
):
    """Create blockscaled A/B tensors with scale factors and output tensor for any format."""
    m, n, k, l = mnkl
    sf_k = ceil_div(k, sf_vec_size)

    def make_operand(mn, torch_dtype):
        if torch_dtype == torch.float4_e2m1fn_x2:
            # FP4 is packed two elements per byte along K
            operand = torch.empty(
                (l, mn, k // 2), dtype=torch.int8, device="cuda"
            ).permute(1, 2, 0)
            operand.copy_(
                torch.randint(-2, 2, operand.shape, dtype=torch.int8, device="cuda")
            )
            return operand.view(dtype=torch_dtype)
        return (
            torch.randint(-2, 3, (l, mn, k), dtype=torch.int8, device="cuda")
            .to(dtype=torch_dtype)
            .permute(1, 2, 0)
        )

    a = make_operand(m, cutlass_torch.dtype(a_dtype))
    b = make_operand(n, cutlass_torch.dtype(b_dtype))

    # Avoid zero E8M0 exponents, which decode as NaN/denorm
    sf_torch_dtype = cutlass_torch.dtype(sf_dtype)
    sf_low = 1 if sf_torch_dtype == torch.float8_e8m0fnu else 0
    sfa_simple = (
        torch.randint(sf_low, 3, (l, m, sf_k), dtype=torch.uint8, device="cuda")
        .permute(1, 2, 0)
        .to(dtype=sf_torch_dtype)
    )
    sfb_simple = (
        torch.randint(sf_low, 3, (l, n, sf_k), dtype=torch.uint8, device="cuda")
        .permute(1, 2, 0)
        .to(dtype=sf_torch_dtype)
    )

    sfa_reordered = reorder_scale_factors(sfa_simple, m, k, sf_vec_size)

    if sfb_tmem_store:
        sfb_reordered = sfb_simple.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    else:
        sfb_reordered = reorder_scale_factors_per_cta(sfb_simple, cta_n, k, sf_vec_size)

    c = torch.zeros(
        (l, n, m), dtype=cutlass_torch.dtype(c_dtype), device="cuda"
    ).permute(2, 1, 0)
    return a, b, sfa_reordered, sfb_reordered, c, sfa_simple, sfb_simple


def _gmem_ptr_from_torch(tensor: torch.Tensor, assumed_align: int) -> cute.Pointer:
    """Create a CuTe pointer whose element type comes from the torch tensor."""
    element_type = from_dlpack(tensor).element_type
    return make_ptr(
        element_type,
        tensor.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=assumed_align,
    )


def to_cute_tensors(
    A: torch.Tensor,
    B: torch.Tensor,
    SFA: torch.Tensor,
    SFB: torch.Tensor,
    C: torch.Tensor,
    scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
):
    """Return ``(a, b, sfa, sfb, c, scale, bias)`` CuTe pointers.

    Scale is ``None`` or one FP32 device value. Bias is ``None`` or a contiguous
    M-element vector.
    """
    if scale is not None:
        if scale.numel() != 1 or scale.dtype != torch.float32:
            raise ValueError("scale must contain one FP32 value")
        if scale.device != C.device:
            raise ValueError("scale must be on the same device as C")
    if bias is not None:
        expected_shape = (C.shape[0],)
        if tuple(bias.shape) != expected_shape:
            raise ValueError(
                f"bias must have shape {expected_shape}, got {tuple(bias.shape)}"
            )
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")
        if bias.device != C.device:
            raise ValueError(
                f"bias must be on the same device as C: expected {C.device}, "
                f"got {bias.device}"
            )

    a_ = _gmem_ptr_from_torch(A, 16)
    b_ = _gmem_ptr_from_torch(B, 16)
    sfa_ = _gmem_ptr_from_torch(SFA, 32)
    sfb_ = _gmem_ptr_from_torch(SFB, 16)
    c_ = _gmem_ptr_from_torch(C, 16)
    scale_ = _gmem_ptr_from_torch(scale, 4) if scale is not None else None
    bias_ = _gmem_ptr_from_torch(bias, 2) if bias is not None else None
    return a_, b_, sfa_, sfb_, c_, scale_, bias_


"""
Testing harness for the blockscaled CuTe DSL Low-Latency Blackwell GEMM kernel.

Usage:
    python low_latency_blockscaled_gemm.py
    python low_latency_blockscaled_gemm.py --a_dtype Float4E2M1FN --b_dtype Float4E2M1FN \
        --sf_dtype Float8E4M3FN --sf_vec_size 16
    python low_latency_blockscaled_gemm.py --a_dtype Float8E4M3FN --b_dtype Float8E4M3FN \
        --sf_dtype Float8E8M0FNU --sf_vec_size 32 --problem_sizes_mnkl 3072,8,4096,6
    python low_latency_blockscaled_gemm.py --a_dtype Float8E4M3FN --b_dtype Float4E2M1FN \
        --sf_dtype Float8E8M0FNU --sf_vec_size 32
    python low_latency_blockscaled_gemm.py --a_dtype Float4E2M1FN --b_dtype Float8E4M3FN \
        --sf_dtype Float8E8M0FNU --sf_vec_size 32
"""


def run(
    problem_sizes_mnkl: tuple[int, int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    c_dtype: Type[cutlass.Numeric],
    mma_tiler_mnk: tuple[int, int, int | None],
    num_ab_stage: int,
    num_sfb_tmem_stage: int,
    use_pdl: bool,
    split_k: int,
    sfb_tmem_store: bool,
    warmup_iterations: int = 10,
    iterations: int = 10000,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    use_scale: bool = False,
    use_bias: bool = False,
    pdl_count: int = -1,
    workspace_count: int | None = None,
    **kwargs,
):
    """Run the Low-Latency Blackwell blockscaled GEMM example with specified configurations.

    :param num_ab_stage: Number of A/B SMEM pipeline stages.
    :type num_ab_stage: int
    :param num_sfb_tmem_stage: Number of staged SFB TMEM buffers.
    :type num_sfb_tmem_stage: int
    :param use_pdl: Enable Programmatic Dependent Launch.
    :type use_pdl: bool
    :param split_k: Cluster split-K factor.
    :type split_k: int
    :param sfb_tmem_store: Use the register-mediated SFB staging path.
    :type sfb_tmem_store: bool
    :param use_cold_l2: Cycle through enough workspaces to keep L2 cold while benchmarking.
    :type use_cold_l2: bool
    :param use_scale: Apply one FP32 output scale in the epilogue.
    :type use_scale: bool
    :param use_bias: Add an M-element bias in the epilogue.
    :type use_bias: bool

    :return: Average benchmark time in microseconds, or None if not benchmarked.
    :rtype: float | None
    """
    m, n, k, l = problem_sizes_mnkl
    cta_m, cta_n, cta_k = mma_tiler_mnk

    # Reject unsupported configurations
    if not LowLatencyBlockscaledGemmKernel.can_implement(
        problem_sizes_mnkl,
        a_dtype,
        b_dtype,
        sf_dtype,
        sf_vec_size,
        c_dtype,
        mma_tiler_mnk,
        num_ab_stage,
        num_sfb_tmem_stage,
        split_k,
        sfb_tmem_store,
    ):
        raise cutlass.testing.CantImplementError(
            f"Unsupported testcase: a={a_dtype}, b={b_dtype}, sf={sf_dtype}, "
            f"sf_vec_size={sf_vec_size}, c={c_dtype}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    # Register-mediated SFB staging is only validated for NVFP4
    is_nvfp4 = (
        a_dtype is cutlass.Float4E2M1FN
        and b_dtype is cutlass.Float4E2M1FN
        and sf_dtype is cutlass.Float8E4M3FN
        and sf_vec_size == 16
    )
    if sfb_tmem_store and not is_nvfp4:
        print(
            "WARNING: sfb_tmem_store is only validated for nvfp4, forcing "
            "the direct-copy path"
        )
        sfb_tmem_store = False

    # Derive CTA K from the operand dtypes when omitted
    if cta_k is None:
        cta_k = (
            256
            if (a_dtype is cutlass.Float4E2M1FN and b_dtype is cutlass.Float4E2M1FN)
            else 128
        )

    print("Running Blackwell Low-Latency blockscaled GEMM test with:")
    print(f"Problem (M, N, K, L): {m}, {n}, {k}, {l}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, SF dtype: {sf_dtype}, "
        f"SF Vec size: {sf_vec_size}"
    )
    print(f"C dtype: {c_dtype}")
    print(f"Mma Tiler (M, N, K): {(cta_m, cta_n, cta_k)}")
    print(f"AB stages: {num_ab_stage}, SFB TMEM stages: {num_sfb_tmem_stage}")
    print(
        f"PDL: {use_pdl}, Split-K: {split_k}, SFB TMEM store: {sfb_tmem_store}, "
        f"Scale: {use_scale}, Bias: {use_bias}"
    )
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {'True' if use_cold_l2 else 'False'}")

    # Build one workspace and compile the kernel
    a, b, sfa_reordered, sfb_reordered, c, sfa_simple, sfb_simple = (
        make_blockscaled_tensors(
            problem_sizes_mnkl,
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            sfb_tmem_store=sfb_tmem_store,
            cta_n=cta_n,
        )
    )
    scale = (
        torch.tensor(0.125, dtype=torch.float32, device="cuda") if use_scale else None
    )
    bias = (
        torch.randint(-2, 3, (m,), dtype=torch.float32, device="cuda").to(
            cutlass_torch.dtype(c_dtype)
        )
        if use_bias
        else None
    )
    a_, b_, sfa_, sfb_, c_, scale_, bias_ = to_cute_tensors(
        a, b, sfa_reordered, sfb_reordered, c, scale=scale, bias=bias
    )
    problem_mnkl = (
        cutlass.Int32(m),
        cutlass.Int32(n),
        cutlass.Int32(k),
        cutlass.Int32(l),
    )
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    print("Compiling DSL kernel...")
    gemm = LowLatencyBlockscaledGemmKernel(
        acc_dtype=cutlass.Float32,
        mma_tiler_mnk=(cta_m, cta_n, cta_k),
        num_ab_stage=num_ab_stage,
        num_sfb_tmem_stage=num_sfb_tmem_stage,
        sf_vec_size=sf_vec_size,
        use_pdl=use_pdl,
        pdl_count=pdl_count,
        split_k=split_k,
        sfb_tmem_store=sfb_tmem_store,
        use_scale=scale_ is not None,
        use_bias=bias_ is not None,
    )
    # Compile against a fake stream; the real stream is bound at launch time
    compiled_fn = cute.compile(
        gemm,
        a_,
        sfa_,
        b_,
        sfb_,
        c_,
        scale_,
        bias_,
        problem_mnkl,
        make_fake_stream(),
        options="--generate-line-info",
    )

    # Compare one run against the dequantized reference
    if not skip_ref_check:
        compiled_fn(a_, sfa_, b_, sfb_, c_, scale_, bias_, problem_mnkl, stream)
        print("Verifying results...")

        torch.cuda.synchronize()
        # Expand per-block scale factors (MN, sf_K, L) to dense per-element (MN, K, L)
        sfa_ref = torch.repeat_interleave(
            sfa_simple.cpu().to(torch.float32), sf_vec_size, dim=1
        )[:, :k, :]
        sfb_ref = torch.repeat_interleave(
            sfb_simple.cpu().to(torch.float32), sf_vec_size, dim=1
        )[:, :k, :]
        compare(
            a,
            b,
            sfa_ref,
            sfb_ref,
            c,
            c_dtype,
            atol=1e-1,
            rtol=1e-3,
            scale=scale,
            bias=bias,
        )

    # Benchmark CUDA graph replays on a dedicated stream
    bench_stream = torch.cuda.Stream()
    bench_cu_stream = cuda.CUstream(bench_stream.cuda_stream)

    def make_workspace():
        wa, wb, wsfa, wsfb, wc, _sfa_s, _sfb_s = make_blockscaled_tensors(
            problem_sizes_mnkl,
            a_dtype,
            b_dtype,
            sf_dtype,
            sf_vec_size,
            c_dtype,
            sfb_tmem_store=sfb_tmem_store,
            cta_n=cta_n,
        )
        wscale = (
            torch.tensor(0.125, dtype=torch.float32, device="cuda")
            if use_scale
            else None
        )
        wbias = (
            torch.randint(-2, 3, (m,), dtype=torch.float32, device="cuda").to(
                cutlass_torch.dtype(c_dtype)
            )
            if use_bias
            else None
        )
        return to_cute_tensors(wa, wb, wsfa, wsfb, wc, scale=wscale, bias=wbias), (
            wa,
            wb,
            wsfa,
            wsfb,
            wc,
            wscale,
            wbias,
        )

    if workspace_count is None:
        workspace_count = 2  # a single workspace inflates the measured memory SOL
        if use_cold_l2:
            _, refs0 = make_workspace()
            one_workspace_bytes = sum(
                t.numel() * t.element_size() for t in refs0 if t is not None
            )
            workspace_count = cutlass.testing.get_workspace_count(
                one_workspace_bytes, warmup_iterations, iterations
            )

    def workspace_generator():
        (wa_, wb_, wsfa_, wsfb_, wc_, wscale_, wbias_), refs = make_workspace()
        args = cutlass.testing.JitArguments(
            wa_, wsfa_, wb_, wsfb_, wc_, wscale_, wbias_, problem_mnkl, bench_cu_stream
        )
        args.add_to_scope([t for t in refs if t is not None])
        return args

    exec_time = cutlass.testing.benchmark(
        compiled_fn,
        workspace_generator=workspace_generator,
        workspace_count=workspace_count,
        stream=bench_cu_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=True,
    )

    runtime_s = exec_time / 1.0e6
    flop = 2 * m * n * k * l
    gflops = (flop / 1.0e9) / runtime_s

    # Achieved DRAM bandwidth for this tiny-N, memory-bound kernel
    sf_k = ceil_div(k, sf_vec_size)
    bytes_per_iter = (
        int(m * k * l * a_dtype.width / 8)  # A
        + int(n * k * l * b_dtype.width / 8)  # B
        + m * sf_k * l  # SFA (1 byte/elem)
        + n * sf_k * l  # SFB (1 byte/elem)
        + int(m * n * l * c_dtype.width / 8)  # C
        + (4 if use_scale else 0)  # scale
        + (int(m * c_dtype.width / 8) if use_bias else 0)  # bias
    )
    dram_bw_gbps = bytes_per_iter / runtime_s / 1e9

    # Theoretical peak DDR bandwidth from the device's memory clock + bus width
    cuda.cuInit(0)
    _, _device = cuda.cuDeviceGet(0)
    _, memory_clock_khz = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, _device
    )
    _, bus_width_bits = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, _device
    )
    peak_bw_gbps = 2.0 * memory_clock_khz * (bus_width_bits / 8) / 1e6
    pct_sol = 100.0 * (dram_bw_gbps / peak_bw_gbps)

    print("Average Runtime : ", exec_time / 1000, "ms")
    print("GFLOPS          : ", gflops)
    print("DRAM BW         : ", dram_bw_gbps, "GB/s")
    print(
        "GPU peak BW     : ",
        peak_bw_gbps,
        f"GB/s (clock: {memory_clock_khz / 1e3:.0f} MHz, bus: {bus_width_bits} bits)",
    )
    print("Memory SOL      : ", pct_sol, "%")

    return exec_time


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError as err:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            ) from err

    parser = argparse.ArgumentParser(
        description="Blockscaled Low-Latency Blackwell GEMM on Blackwell."
    )
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument(
        "--problem_sizes_mnkl",
        type=parse_comma_separated_ints,
        default=(3072, 8, 4096, 6),
        help="Problem shape (M, N, K, L) as comma-separated ints, e.g. 3072,8,4096,6",
    )
    parser.add_argument(
        "--skip_ref_check", action="store_true", help="Skip reference checking"
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=10,
        help="Benchmark warmup (graph replays)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Benchmark iterations (in graph)",
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Cycle through enough workspaces to keep the L2 cache cold",
    )
    parser.add_argument(
        "--mma_tiler_mnk",
        type=parse_comma_separated_ints,
        default=(128, 8, None),
        help="CTA tile shape as 'M,N' (K dtype-derived) or 'M,N,K', e.g. 128,8 or 128,8,256",
    )
    parser.add_argument(
        "--stages", type=int, default=8, help="Number of A/B SMEM pipeline stages"
    )
    parser.add_argument(
        "--sfb_tmem_stages",
        type=int,
        default=4,
        dest="num_sfb_tmem_stage",
        help="num SFB TMEM stages",
    )
    parser.add_argument(
        "--use_pdl",
        action="store_true",
        help="Enable PDL (Programmatic Dependent Launch)",
    )
    parser.add_argument(
        "--split_k",
        type=int,
        default=1,
        help="cluster split-k factor (default: 1, no split)",
    )
    parser.add_argument(
        "--sfb_tmem_store",
        action="store_true",
        help="Use register-mediated SFB staging (default: SMEM-to-TMEM copy)",
    )
    parser.add_argument(
        "--use_scale",
        action="store_true",
        help="Apply one FP32 output scale in the epilogue",
    )
    parser.add_argument(
        "--use_bias",
        action="store_true",
        help="Add an M-element bias in the epilogue",
    )
    args = parser.parse_args()

    if len(args.problem_sizes_mnkl) != 4:
        parser.error("--problem_sizes_mnkl must contain exactly 4 values (M, N, K, L)")

    if len(args.mma_tiler_mnk) not in (2, 3):
        parser.error(
            "--mma_tiler_mnk must contain 2 (M,N; K dtype-derived) or 3 (M,N,K) values"
        )
    # 2 values -> K is derived from the dtypes (None sentinel)
    mma_tiler_mnk = (
        args.mma_tiler_mnk
        if len(args.mma_tiler_mnk) == 3
        else (*args.mma_tiler_mnk, None)
    )

    run(
        args.problem_sizes_mnkl,
        a_dtype=args.a_dtype,
        b_dtype=args.b_dtype,
        sf_dtype=args.sf_dtype,
        c_dtype=args.c_dtype,
        sf_vec_size=args.sf_vec_size,
        mma_tiler_mnk=mma_tiler_mnk,
        num_ab_stage=args.stages,
        num_sfb_tmem_stage=args.num_sfb_tmem_stage,
        use_pdl=args.use_pdl,
        split_k=args.split_k,
        sfb_tmem_store=args.sfb_tmem_store,
        skip_ref_check=args.skip_ref_check,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        use_cold_l2=args.use_cold_l2,
        use_scale=args.use_scale,
        use_bias=args.use_bias,
    )
    print("PASS")
