# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Blackwell low-M BF16 GEMM with a fused SwiGLU epilogue.

This is a deliberately narrow prototype built on the cluster split-K dense
GEMM.  Given canonical gate/up weights, the public problem is::

    gate_bf16 = bf16(a[M, K] @ weight_gate[N, K].T)
    up_bf16 = bf16(a[M, K] @ weight_up[N, K].T)
    out[M, N] = bf16(silu(gate_bf16.float()) * up_bf16.float())

``b`` uses the same column-major view as :func:`flashinfer.mm_bf16`, but its
physical rows must be prepared in 16-row pairs::

    [up_0:16, gate_0:16, up_16:32, gate_16:32, ...]

Consequently one 128-row tensor-core output tile contains both operands for a
64-row output tile.  Only the owner rank applies SwiGLU, after the FP32 split-K
reduction.  Before activation, both accumulator fragments are rounded to BF16
and promoted back to FP32.  That preserves the BF16 hand-off of the unfused
``BF16 GEMM -> SwiGLU`` composition without a global-memory round trip.

V1 intentionally supports BF16 only, no bias, no clamp, an internal 128-row
MMA output tile, and logical output widths divisible by 64.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch as _torch
from cutlass import Int32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05

from .dense_bf16_gemm_sm100_splitk import (
    OWNER_RANK,
    SplitKDenseGemmKernel,
    SplitKTactic,
    _AB_BUFFER_ALIGN_BYTES,
    _AB_ELEMENT_BYTES,
    _CTA_K,
    _MAILBOX_ALIGN_BYTES,
    _MAX_M as _DENSE_MAX_M,
    _MBARRIER_BYTES,
    _TMEM_POINTER_BYTES,
    _bmm_no_bias,
    _from_dlpack_dynamic,
    _make_layout_tensor,
    _store_shared_remote_v4,
    _to_cute_swap,
    autotune_tactics as _dense_autotune_tactics,
    validate_tactic as _validate_dense_tactic,
)

_RAW_CTA_M = 128
_OUTPUT_CTA_M = _RAW_CTA_M // 2
_OUTPUT_ALIGNMENT = 64
_EPILOG_THREADS = 128
_THREADS_PER_CTA = 256
_LOG2_E = 1.4426950408889634

#: Largest M this kernel serves. Deliberately above the dense split-K
#: module's ``_MAX_M``: that bound makes ``mm_bf16``'s cute-dsl backend
#: decline large-M problems so ``auto`` routes elsewhere, whereas this op
#: has no other backend to route to. The bound here is what has been
#: measured, not a hardware limit -- the kernel is correct well beyond it,
#: but the tactic selector is only calibrated this far.
_MAX_M = 128

#: Fixed per-K-tile cost in the tactic ranking model, expressed in the same
#: units as ``mma_n`` so one K tile costs ``_K_TILE_FIXED_COST + mma_n``. Only
#: the ratios matter: this value reproduces the measured 1.00/1.26/1.84 relative
#: cost of mma_n = 8/16/32 at an equal number of K tiles per CTA.
_K_TILE_FIXED_COST = 24

#: Occupancy bounds used when the device cannot be queried (no CUDA context).
#: Under-estimating either only makes the search less aggressive, and tactic
#: selection is a performance choice, never a correctness input.
_FALLBACK_SM_COUNT = 148
_FALLBACK_L2_BYTES = 126 * 1024 * 1024


def validate_swiglu_tactic(
    tactic: SplitKTactic,
    m: int,
    n: int,
    k: int,
) -> None:
    """Validate a tactic for logical output shape ``(m, n)``."""
    if tactic.mma_m != _RAW_CTA_M:
        raise ValueError(
            f"BF16 SwiGLU v1 requires mma_m={_RAW_CTA_M}, got {tactic.mma_m}"
        )
    if n <= 0 or n % _OUTPUT_ALIGNMENT:
        raise ValueError(
            "BF16 SwiGLU v1 requires positive N divisible by "
            f"{_OUTPUT_ALIGNMENT}, got {n}"
        )
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"BF16 SwiGLU requires 1 <= M <= {_MAX_M}, got {m}")
    # M is checked above; clamp it for the dense validator so its own low-M
    # bound -- mm_bf16's backend-dispatch policy -- does not apply here.
    _validate_dense_tactic(tactic, min(m, _DENSE_MAX_M), 2 * n, k)

    values_per_half_thread = (_OUTPUT_CTA_M * tactic.mma_n) // _EPILOG_THREADS
    if values_per_half_thread % 4:
        raise ValueError(
            "each epilogue thread must own a multiple of four values per "
            f"gate/up half, got {values_per_half_thread} for {tactic}"
        )


def _swiglu_tactic_space(m: int, n: int, k: int) -> list[SplitKTactic]:
    """Return the dense split-K tactic space compatible with paired tiles.

    Candidate generator for :func:`default_swiglu_tactic`, not an autotuner
    hook: this kernel selects its tactic from a cost model rather than by
    profiling, so nothing else enumerates this space.
    """
    if n <= 0 or n % _OUTPUT_ALIGNMENT:
        return []
    tactics = []
    # Clamp only the dense generator's own low-M bound (see
    # ``validate_swiglu_tactic``); every candidate is re-checked against the
    # real M below, and the cost model sizes its tiles from the real M too.
    for tactic in _dense_autotune_tactics(min(m, _DENSE_MAX_M), 2 * n, k):
        if tactic.mma_m != _RAW_CTA_M:
            continue
        try:
            validate_swiglu_tactic(tactic, m, n, k)
        except ValueError:
            continue
        tactics.append(tactic)
    return tactics


@functools.cache
def _occupancy_limits() -> tuple[int, int]:
    """SM count and L2 capacity that bound the default tactic search.

    Read from the ambient current device. This is a ranking bound only, so
    reading a peer device on a multi-GPU host is harmless.
    """
    try:
        props = _torch.cuda.get_device_properties(None)
    except Exception:
        return _FALLBACK_SM_COUNT, _FALLBACK_L2_BYTES
    return props.multi_processor_count, props.L2_cache_size


def _swiglu_tactic_footprint(
    tactic: SplitKTactic, m: int, n: int, k: int
) -> tuple[int, int]:
    """Return ``(cta_count, weight_bytes_read)`` for ``tactic``.

    Every kernel-N tile re-reads the whole prepared weight, so shrinking
    ``mma_n`` buys parallelism at the cost of multiplying weight traffic --
    free while the total stays L2-resident, ruinous once it does not.
    """
    n_tiles = -(-m // tactic.mma_n)
    ctas = (2 * n // _RAW_CTA_M) * n_tiles * tactic.split_k
    return ctas, 2 * n * k * _AB_ELEMENT_BYTES * n_tiles


def _swiglu_tactic_cost(tactic: SplitKTactic, k: int) -> float:
    """Relative per-CTA cost: K tiles owned times the per-K-tile tile-width cost.

    With one wave and an L2-resident weight the kernel is per-CTA-work bound,
    not parallelism bound, so this -- and not the CTA count -- is what to
    minimize. See ``_K_TILE_FIXED_COST``.
    """
    return (k // _CTA_K) / tactic.split_k * (_K_TILE_FIXED_COST + tactic.mma_n)


@functools.cache
def _select_swiglu_tactic(
    m: int, n: int, k: int, sm_count: int, l2_bytes: int
) -> SplitKTactic:
    candidates = _swiglu_tactic_space(m, n, k)
    if not candidates:
        raise ValueError(f"no BF16 SwiGLU split-K tactic for M={m}, N={n}, K={k}")

    footprints = {
        tactic: _swiglu_tactic_footprint(tactic, m, n, k) for tactic in candidates
    }
    # Spilling into a second wave, or pushing the replicated weight out of L2,
    # each costs more than any difference the cost model can express. Treat both
    # as hard bounds and relax them only when nothing satisfies them, the wave
    # bound last as the more expensive to violate.
    for bound_waves, bound_l2 in ((True, True), (True, False), (False, True)):
        feasible = [
            tactic
            for tactic in candidates
            if (not bound_waves or footprints[tactic][0] <= sm_count)
            and (not bound_l2 or footprints[tactic][1] <= l2_bytes)
        ]
        if feasible:
            candidates = feasible
            break

    return min(
        candidates,
        key=lambda tactic: (
            _swiglu_tactic_cost(tactic, k),
            -tactic.ab_stages,
            tactic.mma_n,
            footprints[tactic][0],
        ),
    )


def default_swiglu_tactic(m: int, n: int, k: int) -> SplitKTactic:
    """Choose the cheapest tactic that fills one wave without leaving L2.

    The dense heuristic cannot be reused here: it picks ``mma_n``/``split_k``
    jointly with ``mma_m``, and this kernel is locked to the paired 128-row
    tile, so inheriting its choice for ``mma_m=64`` halves the grid.
    """
    if n <= 0 or n % _OUTPUT_ALIGNMENT:
        raise ValueError(
            "BF16 SwiGLU v1 requires positive N divisible by "
            f"{_OUTPUT_ALIGNMENT}, got {n}"
        )
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"BF16 SwiGLU requires 1 <= M <= {_MAX_M}, got {m}")
    return _select_swiglu_tactic(m, n, k, *_occupancy_limits())


class SplitKSwiGLUDenseGemmKernel(SplitKDenseGemmKernel):
    """Split-K BF16 GEMM whose owner CTA stores one SwiGLU half-tile."""

    def __init__(self, *, tactic: SplitKTactic, use_pdl: bool) -> None:
        if tactic.mma_m != _RAW_CTA_M:
            raise ValueError(
                f"paired SwiGLU epilogue requires mma_m={_RAW_CTA_M}, "
                f"got {tactic.mma_m}"
            )
        super().__init__(tactic=tactic, use_pdl=use_pdl, has_bias=False)
        self.threads_per_cta = _THREADS_PER_CTA
        self.epilog_threads = _EPILOG_THREADS
        self.output_cta_m = _OUTPUT_CTA_M
        self.values_per_half_thread = (
            self.output_cta_m * self.cta_n
        ) // self.epilog_threads
        if self.values_per_half_thread % 4:
            raise ValueError(
                "remote split-K stores require four FP32 values per transaction; "
                f"got {self.values_per_half_thread} values per half/thread"
            )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # ``a`` is the swapped, prepared weight.  Its raw extent is twice C's
        # logical output extent, so it -- not C -- defines the grid-x tile count.
        self.kernel(a, b, c, bias).launch(
            grid=(
                cute.ceil_div(a.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # prepared (2 * Out_N, K, L), K-major
        mB: cute.Tensor,  # activation (M, K, L), K-major
        mC: cute.Tensor,  # activated (Out_N, M, L), Out_N-major
        mBias: cute.Tensor,  # compile-wrapper placeholder; intentionally unused
    ):
        """Allocate the inherited mainloop storage and dispatch paired epilogue."""
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
        c_tiler_mn = (self.output_cta_m, self.cta_n)

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
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                # One 32-thread MMA warp plus four 32-thread epilogue warps.
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
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
            self.epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.size(mC, mode=[1]) - n_idx * self.cta_n,
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
            )

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        valid_output_n: cutlass.Int32,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
    ):
        """Reduce paired FP32 fragments, round to BF16, apply SwiGLU, store."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        # Mirror the W4 gated epilogue: one full 128-row T2R transaction
        # produces an RMEM fragment with a dedicated up/gate mode.
        gated_epi_tile = (self.cta_m, self.cta_n)
        gated_tAcc_epi = cute.flat_divide(acc_view, gated_epi_tile)
        gated_tiled_copy_t2r = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                gated_epi_tile,
                False,
            ),
            gated_tAcc_epi[(None, None, 0, 0)],
        )
        gated_thr_copy_t2r = gated_tiled_copy_t2r.get_slice(epi_tid)
        gated_tTR_tAcc = gated_thr_copy_t2r.partition_S(gated_tAcc_epi)

        gated_identity = cute.make_identity_tensor((self.cta_m, self.cta_n))
        gated_identity_epi = cute.flat_divide(gated_identity, gated_epi_tile)
        gated_tTR_identity = gated_thr_copy_t2r.partition_D(gated_identity_epi)
        gated_tTR_rAcc = cute.make_rmem_tensor(
            gated_tTR_identity[(None, None, None, 0, 0)].shape,
            self.acc_dtype,
        )
        rUp = cute.coalesce(cute.flatten(gated_tTR_rAcc[(None, 0, None)]))
        rGate = cute.coalesce(cute.flatten(gated_tTR_rAcc[(None, 1, None)]))
        up_coords = cute.coalesce(
            cute.flatten(gated_tTR_identity[(None, None, None, 0, 0)][(None, 0, None)])
        )
        half_values = cutlass.const_expr(cute.size(rUp))
        assert half_values == self.values_per_half_thread
        assert cute.size(rGate) == half_values
        assert cute.size(up_coords) == half_values

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute.copy(
            gated_tiled_copy_t2r,
            gated_tTR_tAcc[(None, None, None, 0, 0)],
            gated_tTR_rAcc,
        )
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish both half-fragments into one per-thread mailbox record.
        # Rank 0 performs the reduction before the non-linear epilogue.
        if cutlass.const_expr(self.split_k > 1):
            values_per_thread = cutlass.const_expr(2 * half_values)
            values_per_peer = cutlass.const_expr(
                self.epilog_threads * values_per_thread
            )
            if split_rank != OWNER_RANK:
                peer_base = (
                    split_rank - Int32(1)
                ) * values_per_peer + epi_tid * values_per_thread
                for value_idx in cutlass.range_constexpr(0, half_values, 4):
                    _store_shared_remote_v4(
                        rUp[value_idx],
                        rUp[value_idx + 1],
                        rUp[value_idx + 2],
                        rUp[value_idx + 3],
                        mailbox.iterator + peer_base + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
                    _store_shared_remote_v4(
                        rGate[value_idx],
                        rGate[value_idx + 1],
                        rGate[value_idx + 2],
                        rGate[value_idx + 3],
                        mailbox.iterator + peer_base + half_values + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    peer_base = peer * values_per_peer + epi_tid * values_per_thread
                    for value_idx in cutlass.range_constexpr(half_values):
                        rUp[value_idx] = rUp[value_idx] + mailbox[peer_base + value_idx]
                        rGate[value_idx] = (
                            rGate[value_idx]
                            + mailbox[peer_base + half_values + value_idx]
                        )

        if split_rank == OWNER_RANK:
            # Reproduce the materialized BF16 GEMM output contract in registers:
            # FP32 accumulator -> BF16 -> FP32, then evaluate SwiGLU in FP32.
            rUp.store(rUp.load().to(cutlass.BFloat16).to(self.acc_dtype))
            rGate.store(rGate.load().to(cutlass.BFloat16).to(self.acc_dtype))
            for value_idx in cutlass.range_constexpr(0, half_values, 2):
                gate_pair = (rGate[value_idx], rGate[value_idx + 1])
                up_pair = (rUp[value_idx], rUp[value_idx + 1])
                gate_log2e = cute.arch.mul_packed_f32x2(
                    gate_pair,
                    (-cutlass.Float32(_LOG2_E), -cutlass.Float32(_LOG2_E)),
                )
                sigmoid = cute.arch.add_packed_f32x2(
                    (
                        cute.math.exp2(gate_log2e[0], fastmath=True),
                        cute.math.exp2(gate_log2e[1], fastmath=True),
                    ),
                    (1.0, 1.0),
                )
                sigmoid = (
                    cute.arch.rcp_approx(sigmoid[0]),
                    cute.arch.rcp_approx(sigmoid[1]),
                )
                silu = cute.arch.mul_packed_f32x2(gate_pair, sigmoid)
                result = cute.arch.mul_packed_f32x2(up_pair, silu)
                rUp[value_idx] = result[0]
                rUp[value_idx + 1] = result[1]

                coord_0 = up_coords[value_idx]
                coord_1 = up_coords[value_idx + 1]
                output_m_0 = coord_0[0] // 32 * 16 + coord_0[0] % 16
                output_m_1 = coord_1[0] // 32 * 16 + coord_1[0] % 16
                output_n_0 = coord_0[1] % self.cta_n
                output_n_1 = coord_1[1] % self.cta_n
                if output_n_0 < valid_output_n:
                    gD_tile[(output_m_0, output_n_0)] = c_dtype(result[0])
                if output_n_1 < valid_output_n:
                    gD_tile[(output_m_1, output_n_1)] = c_dtype(result[1])


def _make_compile_repr_tensors(
    dtype: _torch.dtype,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):
    """Trace-time stand-ins; C carries half of A's extent per paired tile."""
    m, raw_n, logical_n, k, batch = 8, _RAW_CTA_M, _OUTPUT_CTA_M, _CTA_K, 1
    return tuple(
        _from_dlpack_dynamic(
            _make_layout_tensor(shape, dtype, leading_dim), leading_dim
        )
        for shape, leading_dim in (
            ((batch, raw_n, k), a_leading),
            ((batch, k, m), b_leading),
            ((batch, logical_n, m), c_leading),
        )
    )


@functools.cache
def _get_compiled_splitk_swiglu_kernel(
    device_index: int,
    dtype,
    tactic: SplitKTactic,
    use_pdl: bool,
    leading_dims: tuple[int, int, int],
):
    if dtype != _torch.bfloat16:
        raise ValueError(f"split-K SwiGLU GEMM supports BF16 only; got {dtype}")
    # Representative tensors and the compiled module must live on the device
    # that will run the kernel.
    with _torch.cuda.device(device_index):
        kernel = SplitKSwiGLUDenseGemmKernel(tactic=tactic, use_pdl=use_pdl)
        compile_tensors = _make_compile_repr_tensors(dtype, *leading_dims)
        stream = _cuda.CUstream(_torch.cuda.current_stream(device_index).cuda_stream)
        return cute_ext.compile(_bmm_no_bias, kernel, *compile_tensors, stream)


def _dense_tensors_overlap(lhs: _torch.Tensor, rhs: _torch.Tensor) -> bool:
    """Return whether two validated dense tensors overlap in storage."""
    if lhs.device != rhs.device or lhs.numel() == 0 or rhs.numel() == 0:
        return False
    lhs_start = lhs.data_ptr()
    rhs_start = rhs.data_ptr()
    lhs_end = lhs_start + lhs.numel() * lhs.element_size()
    rhs_end = rhs_start + rhs.numel() * rhs.element_size()
    return lhs_start < rhs_end and rhs_start < lhs_end


def _validate_runtime_tensors(a, b, out) -> tuple[int, int, int]:
    if any(not isinstance(tensor, _torch.Tensor) for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("split-K SwiGLU GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or b.device != a.device or out.device != a.device:
        raise ValueError("a, b, and out must be on the same CUDA device")
    if a.dtype != _torch.bfloat16 or b.dtype != a.dtype or out.dtype != a.dtype:
        raise ValueError("a, b, and out must share BF16 dtype")
    if not a.is_contiguous():
        raise ValueError("a must be row-major contiguous")
    if not b.T.is_contiguous():
        raise ValueError("prepared b must be column-major (b.T contiguous)")
    if not out.is_contiguous():
        raise ValueError("out must be row-major contiguous")
    if _dense_tensors_overlap(out, a):
        raise ValueError("out must not overlap a storage")
    if _dense_tensors_overlap(out, b):
        raise ValueError("out must not overlap b storage")
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    raw_n = b.shape[1]
    if raw_n % 2:
        raise ValueError(f"prepared b width must be even, got {raw_n}")
    n = raw_n // 2
    if n % _OUTPUT_ALIGNMENT:
        raise ValueError(
            f"logical output width must be divisible by {_OUTPUT_ALIGNMENT}, got {n}"
        )
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    return m, n, k


def run_splitk_swiglu(
    a,
    b,
    out,
    pdl: bool,
    tactic: SplitKTactic,
):
    """Run strict-BF16 ``SwiGLU(A @ B)`` with prepared interleaved ``B``."""
    m, n, k = _validate_runtime_tensors(a, b, out)
    validate_swiglu_tactic(tactic, m, n, k)
    device_index = a.device.index
    if device_index is None:
        device_index = _torch.cuda.current_device()
    with _torch.cuda.device(device_index):
        *cute_tensors, _bias, leading_dims = _to_cute_swap(a, b, out, None)
        compiled = _get_compiled_splitk_swiglu_kernel(
            device_index=device_index,
            dtype=a.dtype,
            tactic=tactic,
            use_pdl=bool(pdl),
            leading_dims=leading_dims,
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream(device_index).cuda_stream)
        compiled(*cute_tensors, stream)
    return out


__all__ = [
    "SplitKSwiGLUDenseGemmKernel",
    "SplitKTactic",
    "default_swiglu_tactic",
    "run_splitk_swiglu",
    "validate_swiglu_tactic",
]
