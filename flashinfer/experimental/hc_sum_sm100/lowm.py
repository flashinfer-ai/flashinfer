# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Provenance:
#   * Cluster split-K BF16/FP16 GEMM: flashinfer-ai/flashinfer PR #4266 at
#     629147317d4149a12e53bcef27808bac380c283f.
#   * HC sum-state epilogues: SGLang PR #40552 branch at
#     9cdd6624fe115979b0a0e6c588d9de5ddfeaafe2
#     (sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue, a
#     modified copy of the kernel above), Apache-2.0.
#
# ****************************************************************************
# MODIFICATION NOTICE: this file is a modified version of the sources above.
# Changes made for FlashInfer (2026-11-04):
#   * Only the two HC sum-state fusion families of the low-M route remain:
#     branch-packed Down+Inject (alpha-only epilogue) and Up+Mix (raw gate
#     epilogue with next-sum reset). The plain GEMM, bias, SiLU, unfused gate,
#     pre-normalized and historical direct-epilogue entry points are removed.
#   * The producer and MMA warps, the shared-memory model and the DSMEM
#     helpers are reused from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk
#     instead of a vendored copy. They carry FlashInfer #7443's TMEM release
#     ordering and tcgen05 fences; the epilogue below adopts the same fences
#     and the cluster-scope acquire of peer partials.
#   * Tactics are restricted locally to the validated HC set (Down split-K 16
#     included) and to SM100/SM103, leaving the stable autotuner unchanged.
#   * Compiled kernels use FlashInfer's CuTe-DSL disk cache and TVM-FFI with
#     the current PyTorch stream; the SGLang and msgspec dependencies are gone.
#   * Tiles, K partition, MMA order, reduction order, FP32 statistics
#     consumption, BF16/FP16 rounding, store predicates and launch shared-memory
#     sizes are unchanged.
# ****************************************************************************
"""Low-M (T <= 24) HC sum-state GEMMs on SM100/SM103.

Both kernels run the cluster split-K GEMM of
``flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk`` with an HC epilogue
that consumes FP32 per-branch square sums ``sum_sq[T, C]`` (C = 4) and rebuilds
``rsqrt(sum_sq / H + eps)`` in the kernel; no inverse-RMS buffer exists.

* Down+Inject (:func:`run_splitk_dense_silu_aux`): ``A[T, C*H]`` raw residual
  times the branch-major weight ``B[C, H, Npad]`` (a transpose view of the
  contiguous ``[C, Npad, H]`` storage shared with the large-T route). Each of
  the 16 cluster ranks covers a K slice inside one branch and scales its FP32
  partial by that branch's inverse RMS before the DSMEM reduction. Output tiles
  ``[0, lowrank)`` store ``bf16(silu(scale * acc))``; the trailing Inject tile
  stores only FP32 ``alpha = 2 * sigmoid(scale * acc)`` and leaves its columns
  of ``out`` untouched.
* Up+Mix (:func:`run_splitk_dense_gate`): ``A[T, Kpad]`` SiLU low-rank times the
  permuted, zero-padded Up weight; the epilogue normalizes the raw residual
  per branch, rounds it to the storage dtype, mixes the C branches with their
  sigmoid gates, and clears the NEXT square sums.

The public ``A[M, K] @ B[K, N]`` problem is swapped internally: kernel-M carries
public N (output columns) and kernel-N carries public M (tokens).
"""

from __future__ import annotations

import functools
import math
import struct

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cute_math
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32
from cutlass.cute import experimental as cute_ext

from ...gemm.kernels import dense_bf16_gemm_sm100_splitk as _splitk
from ...gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    _AB_BUFFER_ALIGN_BYTES,
    _CTA_K,
    _FP32_BYTES,
    _MAILBOX_ALIGN_BYTES,
    _MBARRIER_BYTES,
    _SMEM_CAPACITY_BYTES,
    _TMEM_POINTER_BYTES,
    OWNER_RANK,
    SplitKDenseGemmKernel,
    SplitKTactic,
    _align_up,
    _from_dlpack_dynamic,
    _make_layout_tensor,
    _mbarrier_wait_acquire_cluster,
    _smem_bytes,
    _store_shared_remote_v4,
    _tcgen05_fence,
)
from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

__all__ = [
    "SplitKTactic",
    "run_splitk_dense_gate",
    "run_splitk_dense_silu_aux",
]

#: HC branch count; the kernels are specialized for exactly four branches.
_HC_COUNT = 4

#: Largest token count served by the low-M route.
_MAX_ROWS = 24

#: Hidden-size granularity and bound: at split-K 16 each branch's H / 128 K
#: tiles divide evenly across its four ranks.
_HIDDEN_ALIGN = 512
_MAX_HIDDEN = 16384

#: Validated tactics. Down: four ranks per branch; Up: one CTA per tile, with
#: kernel-N (tokens) 8/16/32.
_DOWN_TACTICS = frozenset({SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=3)})
_UP_TACTICS = frozenset(
    SplitKTactic(mma_m=64, mma_n=mma_n, split_k=1, ab_stages=3) for mma_n in (8, 16, 32)
)

#: Compute capabilities this backend serves.
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))

_SUPPORTED_TORCH_DTYPES = (torch.bfloat16, torch.float16)

#: Base-pointer alignment required of every tensor operand.
_TENSOR_ALIGN_BYTES = 32

#: TMA global strides must be multiples of 16 bytes.
_TMA_STRIDE_ALIGN_BYTES = 16

#: Named barrier for the gate epilogue's SMEM staging round-trip.
_GATE_BARRIER_ID = 7

#: Alignment of the gate epilogue SMEM tile.
_GATE_TILE_ALIGN_BYTES = 16

_CUTE_DSL_MODULE = "hc_sum_sm100_lowm"


def _launch_smem_bytes(tactic: SplitKTactic, gate: bool) -> int:
    """Compact dynamic SMEM: the tactic's buffers, plus the FP32 gate tile."""
    smem_bytes = _smem_bytes(tactic, tactic.ab_stages)
    if gate:
        # Include the FP32 gate tile and its allocator alignment.
        smem_bytes = (
            _align_up(smem_bytes, _GATE_TILE_ALIGN_BYTES)
            + tactic.mma_m * tactic.mma_n * _FP32_BYTES
        )
    return smem_bytes


def _sigmoid_f32(v):
    return cute_math.rcp(cute_math.exp(v * -1.0) + 1.0)


class _HCSumStateGemmKernel(SplitKDenseGemmKernel):
    """Cluster split-K GEMM with an HC sum-state epilogue.

    ``gate=False`` is branch-packed Down+Inject; ``gate=True`` is Up+Mix. The
    DMA and MMA warps are inherited unchanged from the core kernel.
    """

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        gate: bool,
        use_pdl: bool,
        epilogue_scale: float,
        epilogue_aux_start: int,
        rms_eps: float,
        rms_inv_h: float,
    ) -> None:
        super().__init__(tactic=tactic, use_pdl=use_pdl, has_bias=False)
        self.gate = gate
        self.epilogue_scale = epilogue_scale
        self.epilogue_aux_start = epilogue_aux_start
        self.rms_eps = rms_eps
        self.rms_inv_h = rms_inv_h
        # Compact sizing lets CTAs share an SM; the inherited MMA warp releases
        # TMEM only after the epilogue has read it (#7443).
        self.launch_smem_bytes = _launch_smem_bytes(tactic, gate)
        if self.launch_smem_bytes > _SMEM_CAPACITY_BYTES:
            raise ValueError(
                f"tactic {tactic} needs {self.launch_smem_bytes} B of shared "
                f"memory; only {_SMEM_CAPACITY_BYTES} B available"
            )
        if gate:
            if tactic.split_k != 1:
                raise ValueError("the gate epilogue requires split_k=1")
            if tactic.mma_m % _HC_COUNT:
                raise ValueError(
                    f"HC count {_HC_COUNT} must divide mma_m={tactic.mma_m}"
                )
            gate_out_elems = (tactic.mma_m // _HC_COUNT) * tactic.mma_n
            if gate_out_elems % self.epilog_threads:
                raise ValueError(
                    f"gate tile ({tactic.mma_m}, {tactic.mma_n}) gives "
                    f"{gate_out_elems} outputs; must be a multiple of "
                    f"{self.epilog_threads}"
                )
        else:
            if tactic.split_k % _HC_COUNT:
                raise ValueError("raw Down requires branch-aligned cluster split-K")
            if epilogue_aux_start <= 0 or epilogue_aux_start % tactic.mma_m:
                raise ValueError("the Inject tile must start at an mma_m boundary")

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        norm_weight: cute.Tensor,
        x: cute.Tensor,
        out: cute.Tensor,
        sum_sq: cute.Tensor,
        next_sum_sq: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, norm_weight, x, out, sum_sq, next_sum_sq).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.launch_smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # Weights (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # Activations (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # Down: low-rank output; Up: residual shape anchor
        mNorm: cute.Tensor,  # Up: broadcast permuted RMS weight; Down: dead
        mX: cute.Tensor,  # Up: raw residual [T, C*H]; Down: dead
        mOut: cute.Tensor,  # Up: mixed output [T, H]; Down: FP32 alpha [T, C]
        mSumSq: cute.Tensor,  # Read-only CURRENT square sums [T, C]
        mNextSum: cute.Tensor,  # Up: write-only NEXT-sum reset; Down: dead
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Allocation order and alignments are mirrored by _launch_smem_bytes.
        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        # Unused without bias, but kept for the inherited DMA warp and the
        # shared-memory layout.
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if cutlass.const_expr(self.gate):
            gate_tile = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout((self.cta_m, self.cta_n)),
                alignment=_GATE_TILE_ALIGN_BYTES,
            )
        else:
            gate_tile = mailbox

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        if cutlass.const_expr(self.gate):
            weight_batch = l_idx
            k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
            weight_k_start = split_rank * k_tile_count
        else:
            # mA is [N,H,C], backed by the large-T [C,Npad,H] storage.
            # Each rank stays inside one branch; mB still spans the full C*H.
            ranks_per_branch = self.split_k // _HC_COUNT
            weight_batch = split_rank // ranks_per_branch
            k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // ranks_per_branch
            weight_k_start = (split_rank % ranks_per_branch) * k_tile_count
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(
                    mA, (self.cta_m, self.cta_k), (bidx, None, weight_batch)
                ),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                weight_k_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.hc_epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.local_tile(mNorm, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                gate_tile,
                mX,
                mOut,
                mSumSq,
                mNextSum,
                bidx,
                n_idx,
            )

    @cute.experimental.jit
    def hc_epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        gNorm_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        gate_tile,
        mX: cute.Tensor,
        mOut: cute.Tensor,
        mSumSq: cute.Tensor,
        mNextSum: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
    ):
        if cutlass.const_expr(self.use_pdl):
            # The A/weight producer can release the next grid early. Scalar
            # state accesses must also wait, independently of the B DMA warp.
            cute.arch.griddepcontrol_wait()
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _tcgen05_fence(before=False)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        if cutlass.const_expr(self.gate):
            # These per-output inverses do not depend on MMA results. Prepare
            # them after PDL/TMEM setup, keeping current sums read-only.
            early_j_per_tile = self.cta_m // _HC_COUNT
            early_iterations = early_j_per_tile * self.cta_n // self.epilog_threads
            rEarlyGateInv = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                cute.make_layout((_HC_COUNT, early_iterations)),
                alignment=32,
            )
            for early_it in cutlass.range_constexpr(early_iterations):
                early_row = (
                    n_idx * self.cta_n
                    + (early_it * self.epilog_threads + epi_tid) // early_j_per_tile
                )
                for early_g in cutlass.range_constexpr(_HC_COUNT):
                    early_inv = cutlass.Float32(0.0)
                    if early_row < cute.size(mOut, mode=[0]):
                        early_inv = cute.math.rsqrt(
                            mSumSq[early_row, early_g] * self.rms_inv_h + self.rms_eps,
                            fastmath=True,
                        )
                    rEarlyGateInv[early_g, early_it] = early_inv
        else:
            rD = cute_ext.allocate(
                c_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            # Identity coordinates use rAcc's TMEM->RMEM partition. Kernel axes
            # are swapped relative to the public GEMM: coordinate 0 is the
            # output column, coordinate 1 the token row.
            tCoord = thr_t2r.partition_D(
                cute.flat_divide(cute.make_identity_tensor(epi_tile), epi_tile)
            )[(None, None, None, 0, 0)]
            # Each split rank reads CURRENT sums and reconstructs inverse RMS
            # locally for the one branch its K slice covers.
            branch = split_rank // (self.split_k // _HC_COUNT)
            rEarlyInv = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            for early_idx in cutlass.range_constexpr(cute.size(rmem_layout)):
                early_row = n_idx * self.cta_n + tCoord[early_idx][1]
                early_inv = cutlass.Float32(0.0)
                if early_row < cute.size(mSumSq, mode=[0]):
                    early_inv = cute.math.rsqrt(
                        mSumSq[early_row, branch] * self.rms_inv_h + self.rms_eps,
                        fastmath=True,
                    )
                rEarlyInv[early_idx] = early_inv

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _tcgen05_fence(before=False)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        _tcgen05_fence(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(not self.gate):
            # A rank's contiguous K partition stays inside exactly one HC
            # branch. Scale its FP32 partial BEFORE inter-branch reduction.
            for value_idx in cutlass.range_constexpr(cute.size(rmem_layout)):
                row = n_idx * self.cta_n + tCoord[value_idx][1]
                if row < cute.size(mSumSq, mode=[0]):
                    rAcc[value_idx] = rAcc[value_idx] * rEarlyInv[value_idx]

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1):
            assert cute.size(rmem_layout) == self.mailbox_elements // (
                (self.split_k - 1) * self.epilog_threads
            )
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(
                self.epilog_threads * values_per_thread
            )
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                _mbarrier_wait_acquire_cluster(bar_reduce, Int32(0))
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer
                                + epi_tid * values_per_thread
                                + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            if cutlass.const_expr(self.gate):
                rSig = cute_ext.allocate(
                    self.acc_dtype,
                    cute.AddressSpace.rmem,
                    rmem_layout,
                    alignment=32,
                )
                rSig.store(_sigmoid_f32(rAcc.load()))
                sGate_epi = cute.flat_divide(gate_tile, epi_tile)
                cute_ext.partition_and_copy(thr_t2r, rSig, sGate_epi[None, None, 0, 0])
                cute.arch.barrier(
                    barrier_id=_GATE_BARRIER_ID,
                    number_of_threads=self.epilog_threads,
                )

                group = _HC_COUNT
                rows = cute.size(mOut, mode=[0])
                hs = cute.size(mOut, mode=[1])
                # One CTA per token tile clears NEXT sums after the dependency
                # wait. Apply's normal stream launch waits for this reset.
                if bidx == 0:
                    for reset_it in cutlass.range_constexpr(
                        cute.ceil_div(self.cta_n * group, self.epilog_threads)
                    ):
                        reset_idx = reset_it * self.epilog_threads + epi_tid
                        reset_row = n_idx * self.cta_n + reset_idx // group
                        if reset_idx < self.cta_n * group and reset_row < rows:
                            mNextSum[reset_row, reset_idx % group] = cutlass.Float32(
                                0.0
                            )
                j_per_tile = self.cta_m // group
                total_out = j_per_tile * self.cta_n
                for it in cutlass.range_constexpr(total_out // self.epilog_threads):
                    elem = it * self.epilog_threads + epi_tid
                    j_local = elem % j_per_tile
                    m_local = elem // j_per_tile
                    m_global = n_idx * self.cta_n + m_local
                    j_global = bidx * j_per_tile + j_local
                    if m_global < rows:
                        gated = cutlass.Float32(0.0)
                        for g in cutlass.range_constexpr(group):
                            sig = gate_tile[j_local * group + g, m_local]
                            xv = mX[m_global, g * hs + j_global].to(cutlass.Float32)
                            weight = gNorm_tile[j_local * group + g, m_local].to(
                                cutlass.Float32
                            )
                            # Match materialized RMSNorm's activation dtype
                            # before the gate multiply; gamma is 1+Wnorm.
                            xv = (
                                (
                                    xv
                                    * rEarlyGateInv[g, it]
                                    * (cutlass.Float32(1.0) + weight)
                                )
                                .to(c_dtype)
                                .to(cutlass.Float32)
                            )
                            gated = gated + sig * xv
                        mOut[m_global, j_global] = (gated * self.epilogue_scale).to(
                            c_dtype
                        )
            else:
                tile_start = bidx * self.cta_m
                aux_start = self.epilogue_aux_start
                if (
                    tile_start < aux_start + _HC_COUNT
                    and tile_start + self.cta_m > aux_start
                ):
                    # The trailing CTA-aligned Inject tile writes FP32 alpha
                    # straight from the FP32 accumulators.
                    rows = cute.size(mOut, mode=[0])
                    for value_idx in cutlass.range_constexpr(cute.size(rmem_layout)):
                        coord = tCoord[value_idx]
                        j_global = bidx * self.cta_m + coord[0]
                        m_global = n_idx * self.cta_n + coord[1]
                        if (
                            m_global < rows
                            and j_global >= aux_start
                            and j_global < aux_start + _HC_COUNT
                        ):
                            mOut[m_global, j_global - aux_start] = cutlass.Float32(
                                2.0
                            ) * _sigmoid_f32(rAcc[value_idx] * self.epilogue_scale)
                # Down tiles keep their SiLU, storage cast and output store.
                # The Inject tile never stores into the low-rank output.
                if tile_start < aux_start:
                    scaled = rAcc.load() * self.epilogue_scale
                    rAcc.store(scaled * _sigmoid_f32(scaled))
                    rD.store(rAcc.load().to(c_dtype))
                    # Preserve TMEM coordinates; the copy predicates tails.
                    cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        # The reduction mbarrier covers remote stores; no cluster barrier needed.


@cute.experimental.jit
def _launch_down(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    alpha: cute.Tensor,
    sum_sq: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        c,
        c,
        alpha,
        sum_sq,
        # Only the gate epilogue resets sums, so this operand is dead in Down.
        # Reuse the input instead of allocating a buffer.
        sum_sq,
        stream,
    )


@cute.experimental.jit
def _launch_up(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    x: cute.Tensor,
    out: cute.Tensor,
    sum_sq: cute.Tensor,
    norm_weight: cute.Tensor,
    next_sum_sq: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        norm_weight,
        x,
        out,
        sum_sq,
        next_sum_sq,
        stream,
    )


def _compile_tensors(dtype: torch.dtype, gate: bool):
    """Placeholders on the current device, with the original's extents. Every
    extent and non-unit stride is dynamic; leading dimensions match the
    host-validated runtime views."""
    m, n, k = 64, 8, _CTA_K
    tensors = [
        (_make_layout_tensor((1 if gate else _HC_COUNT, n, k), dtype, 2), 2),
        (_make_layout_tensor((1, k, m), dtype, 1), 1),
        (_make_layout_tensor((1, n, m), dtype, 1), 1),
    ]
    stats = torch.empty((n, _HC_COUNT), dtype=torch.float32, device="cuda")
    if gate:
        tensors += [
            (torch.empty((n, m), dtype=dtype, device="cuda"), 1),
            (torch.empty((n, m), dtype=dtype, device="cuda"), 1),
            (stats, 1),
            (
                torch.empty((m,), dtype=dtype, device="cuda").as_strided(
                    (m, n, 1), (1, 0, 0)
                ),
                0,
            ),
            (torch.empty_like(stats), 1),
        ]
    else:
        tensors += [(torch.empty_like(stats), 1), (stats, 1)]
    return tuple(_from_dlpack_dynamic(t, leading) for t, leading in tensors)


def _float_key(value: float) -> str:
    # Exact IEEE-754 bits; '-' and '+' would collide after symbol sanitizing.
    return struct.pack(">d", value).hex()


def _kernel_source_files() -> tuple[str, ...]:
    """Source files whose content invalidates the on-disk kernel cache."""
    return (__file__, _splitk.__file__)


@functools.cache
def _get_compiled_kernel(
    device_index: int,
    dtype: torch.dtype,
    tactic: SplitKTactic,
    gate: bool,
    use_pdl: bool,
    scale: float,
    aux_start: int,
    rms_eps: float,
    rms_inv_h: float,
):
    capability = torch.cuda.get_device_capability(device_index)
    if capability not in _SUPPORTED_CAPABILITIES:
        raise ValueError(
            f"the HC sum-state low-M GEMM supports SM100/SM103; device "
            f"{device_index} has compute capability {capability}"
        )
    if dtype not in _SUPPORTED_TORCH_DTYPES:
        raise ValueError(
            f"the HC sum-state low-M GEMM supports {_SUPPORTED_TORCH_DTYPES}; "
            f"got {dtype}"
        )
    kernel = _HCSumStateGemmKernel(
        tactic=tactic,
        gate=gate,
        use_pdl=use_pdl,
        epilogue_scale=scale,
        epilogue_aux_start=aux_start,
        rms_eps=rms_eps,
        rms_inv_h=rms_inv_h,
    )

    def compile_kernel():
        return cute_ext.compile(
            _launch_up if gate else _launch_down,
            kernel,
            *_compile_tensors(dtype, gate),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    with torch.cuda.device(device_index):
        return build_and_load_cute_dsl_kernel(
            _CUTE_DSL_MODULE,
            f"{'up' if gate else 'down'}_{str(dtype).removeprefix('torch.')}"
            f"_m{tactic.mma_m}_n{tactic.mma_n}_split{tactic.split_k}"
            f"_stages{tactic.ab_stages}_pdl{int(use_pdl)}_aux{aux_start}"
            f"_scale{_float_key(scale)}_eps{_float_key(rms_eps)}"
            f"_invh{_float_key(rms_inv_h)}",
            compile_kernel,
            extra_key_files=_kernel_source_files(),
        )


def _check_tensors(**tensors) -> None:
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch tensor")


def _is_row_major(tensor: torch.Tensor, cols: int) -> bool:
    """Row-major; padded leading strides are allowed."""
    return tensor.stride(1) == 1 and tensor.stride(0) >= cols


def _is_tma_stride(tensor: torch.Tensor, stride: int) -> bool:
    return stride * tensor.element_size() % _TMA_STRIDE_ALIGN_BYTES == 0


def _check_operands(a: torch.Tensor, others: tuple[torch.Tensor, ...]) -> None:
    if a.device.type != "cuda" or any(t.device != a.device for t in others):
        raise ValueError("all tensors must be on the same CUDA device")
    if a.dtype not in _SUPPORTED_TORCH_DTYPES or any(
        t.dtype != a.dtype for t in others
    ):
        raise ValueError("matrix operands must share BF16 or FP16 dtype")
    if any(t.data_ptr() % _TENSOR_ALIGN_BYTES for t in (a, *others)):
        raise ValueError(f"matrix operands must be {_TENSOR_ALIGN_BYTES}-byte aligned")


def _check_rows(rows: int) -> None:
    if not 1 <= rows <= _MAX_ROWS:
        raise ValueError(f"the low-M route requires 1 <= T <= {_MAX_ROWS}, got {rows}")


def _check_branch_stats(
    tensor: torch.Tensor, name: str, rows: int, device: torch.device
) -> None:
    if (
        tensor.shape != (rows, _HC_COUNT)
        or tensor.dtype != torch.float32
        or tensor.device != device
        or not tensor.is_contiguous()
        or tensor.data_ptr() % _TENSOR_ALIGN_BYTES
    ):
        raise ValueError(
            f"{name} must be a {_TENSOR_ALIGN_BYTES}-byte-aligned contiguous FP32 "
            f"[{rows}, {_HC_COUNT}] tensor on {device}"
        )


def _check_rms_eps(rms_eps: float) -> None:
    if not math.isfinite(rms_eps) or rms_eps <= 0:
        raise ValueError(f"sum-state RMS requires a positive finite eps, got {rms_eps}")


def run_splitk_dense_silu_aux(
    a,
    b,
    out,
    aux,
    pdl: bool,
    tactic: SplitKTactic,
    scale: float,
    aux_start: int,
    *,
    sum_sq,
    rms_eps: float = 1e-6,
    branch_packed: bool = True,
):
    """Branch-packed HC Down+Inject consuming FP32 square sums.

    ``a`` is the raw residual ``[T, C*H]`` (row-major, T <= 24, C = 4, H a
    multiple of 512 up to 16384). ``b`` is ``[C, H, Npad]``, a transpose view of
    the contiguous ``[C, Npad, H]`` storage shared with the large-T route.
    ``out.shape[1]`` supplies the logical ``N = aux_start + C`` (``<= Npad``);
    physical padding is neither computed nor stored. ``sum_sq[T, C]`` holds the
    CURRENT FP32 per-branch square sums and is only read; each split rank
    scales its FP32 partial by ``rsqrt(sum_sq / H + rms_eps)`` of its branch
    before the cluster reduction.

    ``out[:, :aux_start]`` receives ``silu(scale * acc)`` in the storage dtype.
    ``aux[T, C]`` receives FP32 ``2 * sigmoid(scale * acc)`` of the trailing
    Inject columns directly from the FP32 accumulators; those columns of
    ``out`` are left untouched, so a zero-initialized padded low-rank buffer
    keeps its zero padding for the Up GEMM. ``aux_start`` must be a multiple of
    ``tactic.mma_m``. Only ``branch_packed=True`` is implemented.
    """
    _check_tensors(a=a, b=b, out=out, aux=aux, sum_sq=sum_sq)
    if not branch_packed:
        raise ValueError("only the branch-packed HC Down+Inject is implemented")
    if tactic not in _DOWN_TACTICS:
        raise ValueError(f"Down tactic must be one of {sorted(_DOWN_TACTICS, key=str)}")
    if a.ndim != 2 or b.ndim != 3 or out.ndim != 2:
        raise ValueError("Down requires A[T,C*H], B[C,H,Npad] and out[T,N]")
    branches, hidden, npad = b.shape
    rows, k = a.shape
    n = out.shape[1]
    if (
        branches != _HC_COUNT
        or hidden % _HIDDEN_ALIGN
        or not 0 < hidden <= _MAX_HIDDEN
        or k != branches * hidden
        or b.stride() != (npad * hidden, 1, hidden)
    ):
        raise ValueError(
            f"expected C={_HC_COUNT}, H a multiple of {_HIDDEN_ALIGN} up to "
            f"{_MAX_HIDDEN}, full C*H input and branch-major weight storage"
        )
    if (
        aux_start <= 0
        or aux_start % tactic.mma_m
        or n != aux_start + _HC_COUNT
        or n > npad
        or out.shape[0] != rows
    ):
        raise ValueError(
            f"out must be [T, aux_start + {_HC_COUNT}] within Npad={npad}, with "
            f"aux_start a positive multiple of mma_m={tactic.mma_m}"
        )
    _check_rows(rows)
    _check_operands(a, (b, out))
    if not _is_row_major(a, k) or not _is_tma_stride(a, a.stride(0)):
        raise ValueError("a must be row-major with a 16-byte-aligned row stride")
    if not _is_row_major(out, n):
        raise ValueError("out must be row-major (padded leading stride allowed)")
    _check_branch_stats(aux, "aux", rows, a.device)
    _check_branch_stats(sum_sq, "sum_sq", rows, a.device)
    _check_rms_eps(rms_eps)

    with torch.cuda.device(a.device):
        compiled = _get_compiled_kernel(
            a.get_device(),
            a.dtype,
            tactic,
            False,
            bool(pdl),
            float(scale),
            aux_start,
            float(rms_eps),
            1.0 / hidden,
        )
        compiled(
            b[:, :, :n].transpose(-2, -1),
            a.unsqueeze(0).transpose(-2, -1),
            out.unsqueeze(0).transpose(-2, -1),
            aux,
            sum_sq,
        )
    return out, aux


def run_splitk_dense_gate(
    a,
    b,
    x,
    out,
    pdl: bool,
    tactic: SplitKTactic,
    scale: float,
    group: int,
    *,
    norm_weight_permuted,
    sum_sq,
    next_sum_sq,
    rms_eps: float = 1e-6,
):
    """HC Up+Mix consuming raw residual, FP32 square sums and RMS weight.

    ``logits = A @ B`` with ``A[T, Kpad]`` row-major and ``B[Kpad, C*H]``
    column-major, pre-permuted so ``n = j*C + g``; ``x[T, C*H]`` is the raw
    residual and the logits matrix is never materialized. With
    ``inv[m, g] = rsqrt(sum_sq[m, g] / H + rms_eps)``::

        out[m, j] = dtype(scale * sum_g sigmoid(logits[m, j*C + g])
                          * fp32(dtype(x[m, g*H + j] * inv[m, g]
                                       * (1 + w[j*C + g]))))

    where ``w`` is ``norm_weight_permuted[C*H]``. ``sum_sq`` is only read;
    ``next_sum_sq[T, C]`` is a separate output that is cleared after the
    dependency wait and must not overlap ``sum_sq``. ``group`` must be 4.
    """
    _check_tensors(
        a=a,
        b=b,
        x=x,
        out=out,
        norm_weight_permuted=norm_weight_permuted,
        sum_sq=sum_sq,
        next_sum_sq=next_sum_sq,
    )
    if tactic not in _UP_TACTICS:
        raise ValueError(f"Up tactic must be one of {sorted(_UP_TACTICS, key=str)}")
    if group != _HC_COUNT:
        raise ValueError(f"the HC gate requires group={_HC_COUNT}, got {group}")
    if a.ndim != 2 or b.ndim != 2 or x.ndim != 2 or out.ndim != 2:
        raise ValueError("Up requires 2D A, B, x and out")
    rows, k = a.shape
    n = b.shape[1]
    hidden = n // group
    if (
        b.shape[0] != k
        or k <= 0
        or k % _CTA_K
        or n % group
        or hidden % _HIDDEN_ALIGN
        or not 0 < hidden <= _MAX_HIDDEN
    ):
        raise ValueError(
            f"Up requires K a positive multiple of {_CTA_K} and N = {group}*H with "
            f"H a multiple of {_HIDDEN_ALIGN} up to {_MAX_HIDDEN}; got "
            f"a {tuple(a.shape)}, b {tuple(b.shape)}"
        )
    if x.shape != (rows, n) or out.shape != (rows, hidden):
        raise ValueError(
            f"x must be {(rows, n)} and out {(rows, hidden)}; got "
            f"{tuple(x.shape)} and {tuple(out.shape)}"
        )
    _check_rows(rows)
    _check_operands(a, (b, x, out, norm_weight_permuted))
    if not _is_row_major(a, k) or not _is_tma_stride(a, a.stride(0)):
        raise ValueError("a must be row-major with a 16-byte-aligned row stride")
    if not (b.stride(0) == 1 and b.stride(1) >= k and _is_tma_stride(b, b.stride(1))):
        raise ValueError("b must be column-major with a 16-byte-aligned stride")
    if not _is_row_major(x, n) or not _is_row_major(out, hidden):
        raise ValueError("x and out must be row-major (padded leading stride allowed)")
    if norm_weight_permuted.shape != (n,) or not norm_weight_permuted.is_contiguous():
        raise ValueError(f"norm_weight_permuted must be contiguous [{n}]")
    _check_branch_stats(sum_sq, "sum_sq", rows, a.device)
    _check_branch_stats(next_sum_sq, "next_sum_sq", rows, a.device)
    # Both statistics tensors are contiguous; reject partial overlap too.
    stat_bytes = sum_sq.numel() * sum_sq.element_size()
    if (
        next_sum_sq.data_ptr() < sum_sq.data_ptr() + stat_bytes
        and sum_sq.data_ptr() < next_sum_sq.data_ptr() + stat_bytes
    ):
        raise ValueError("next_sum_sq and the current sum_sq must not overlap")
    _check_rms_eps(rms_eps)

    with torch.cuda.device(a.device):
        compiled = _get_compiled_kernel(
            a.get_device(),
            a.dtype,
            tactic,
            True,
            bool(pdl),
            float(scale),
            0,
            float(rms_eps),
            1.0 / hidden,
        )
        compiled(
            b.unsqueeze(0).transpose(-2, -1),
            a.unsqueeze(0).transpose(-2, -1),
            x.unsqueeze(0).transpose(-2, -1),
            x,
            out,
            sum_sq,
            norm_weight_permuted.as_strided((n, rows, 1), (1, 0, 0)),
            next_sum_sq,
        )
    return out
