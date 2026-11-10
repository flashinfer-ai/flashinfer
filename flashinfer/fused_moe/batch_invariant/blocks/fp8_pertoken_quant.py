# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The per-token e4m3 quantizer of the class ``moe.e4m3.<act>.pertoken`` (cache module ``batch_invariant_moe_fp8pt``): one bf16 row
in, its e4m3 codes and one fp32 scale out, the stores redirected to the grouped operand's rows when a routing is given.

Term per row (:mod:`..spec_fp8`): ``amax = fmaxc-tree over fabs(x_i)`` (``pertoken_row_reduction``: exact in every association, so
the register tree, the lane shuffles and the warp exchange below compute it whatever their order), ``m = div(448, nzor(amax, 448))``
(``pertoken_mult``: 1 for a zero row), ``q_i = cvt.rn.satfinite.e4m3(mul(x_i, m))`` (``pertoken_quant_elem``, the paired conversion),
``s = div(1, m)`` (``pertoken_scale``: the scale the GEMM epilogue reads).

Placements: ``k = 0`` stores row ``t`` at row ``t`` (the intermediate's requantization); ``k >= 1`` stores token ``t``'s codes and
scale at the ``k`` dense rows ``dest[t k + j]`` of the grouped operand (the input quantization fused with the scatter: one launch;
a dropped slot, ``dest = -1``, stores nothing). ``W`` warps per row (a power of two, at most 8), ``32 W`` lanes holding 8-element
vectors in interleaved slabs; ``4 // W`` rows per CTA (one for ``W >= 4``). The row is read after the programmatic-dependent-launch
wait (it is the predecessor's output); the one-wave trigger follows the stores.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8

from ....batch_invariant import jit
from ....batch_invariant.quantization.cute_ext import CuteF32
from ....batch_invariant.quantization.cvt_fp8 import cvt_f32x2_to_fp8x2
from .. import pdl
from ..spec_fp8 import (
    PT_ELEM,
    PT_LANES,
    pertoken_leaf,
    pertoken_mult,
    pertoken_quant_elem,
    pertoken_scale,
)

MODULE = "batch_invariant_moe_fp8pt"
VEC = PT_ELEM  # bf16 elements per 16-byte load (one leaf vector of the reduction)
LANES = PT_LANES


def warps_per_row_quant(K: int) -> int:
    """The quantizer's placement at row length ``K``: one warp holds up to 1024 elements (four vectors per lane), doubled up to 8 warps."""
    w = 1
    while w < 8 and 1024 * w < K:
        w *= 2
    return w


def _select(cond, a, b):
    """``a`` if ``cond`` else ``b`` (``arith.select``: no arithmetic)."""
    from cutlass._mlir.dialects import arith

    return Float32(
        arith.select(cond.ir_value(), Float32(a).ir_value(), Float32(b).ir_value())
    )


class PerTokenQuantE4m3Kernel:
    """``q[dest, :] = cvt_e4m3(mul(x[t, :], m_t))``, ``s[dest] = div(1, m_t)`` per row ``t < M`` (``K`` static, a multiple of 8)."""

    def __init__(self, K: int, W: int, k: int):
        assert K % VEC == 0 and W in (1, 2, 4, 8) and k >= 0, (K, W, k)
        self.K, self.W, self.k = K, W, k
        self.rows_per_cta = max(1, 4 // W)
        self.threads = LANES * W * self.rows_per_cta
        self.n_vec = K // VEC
        self.vecs_per_lane = -(-self.n_vec // (LANES * W))

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        mDest: cute.Tensor,
        M: Int32,
        stream,
    ):
        self.kernel(mX, mQ, mS, mDest, M).launch(
            grid=[cute.ceil_div(M, self.rows_per_cta), 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        mDest: cute.Tensor,
        M: Int32,
    ):
        fp = CuteF32
        W, RPC, NV, VPL, k = (
            self.W,
            self.rows_per_cta,
            self.n_vec,
            self.vecs_per_lane,
            self.k,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        warp = tidx // Int32(LANES)
        row_in_cta = warp // Int32(W)
        w = warp % Int32(W)
        t_raw = bidx * Int32(RPC) + row_in_cta
        valid_row = t_raw < M
        t = cutlass.min(t_raw, M - Int32(1))
        cute.arch.griddepcontrol_wait()  # the rows are the predecessor's output
        atom_ld = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        atom_st = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=64
        )
        row_x = mX[(t, None)]
        frags = [cute.make_rmem_tensor((VEC,), mX.element_type) for _ in range(VPL)]
        # the loads: lane ``lane`` of warp ``w`` holds the vectors ``(i W + w) 32 + lane``, i < VPL (clamped past the row: not stored)
        for i in cutlass.range_constexpr(VPL):
            j = (Int32(i * W) + w) * Int32(LANES) + lane
            jc = cutlass.min(j, Int32(NV - 1))
            cute.copy(atom_ld, cute.local_tile(row_x, (VEC,), (jc,)), frags[i])
        # the row amax: the register tree over the lane's leaves, the lane shuffles, the warp exchange (every association is the term)
        acc = fp.const(0.0)  # the identity of fmaxc on the non-negative leaves
        for i in cutlass.range_constexpr(VPL):
            j = (Int32(i * W) + w) * Int32(LANES) + lane
            live = j < Int32(NV)
            for e in cutlass.range_constexpr(VEC):
                a = pertoken_leaf(fp, frags[i][e].to(Float32))
                if cutlass.const_expr(NV % (LANES * W) != 0):
                    a = _select(
                        live, a, fp.const(0.0)
                    )  # an absent vector contributes +0, the identity of fmaxc
                acc = fp.fmaxc(acc, a)
        for off in (16, 8, 4, 2, 1):
            acc = fp.fmaxc(acc, cute.arch.shuffle_sync_bfly(acc, offset=off))
        if cutlass.const_expr(W > 1):
            smem = cutlass.memory.SmemAllocator()
            xbuf = smem.allocate_tensor(
                Float32, cute.make_layout((RPC, W)), byte_alignment=16
            )
            if lane == Int32(0):
                xbuf[(row_in_cta, w)] = acc
            cute.arch.barrier()
            amax = xbuf[(row_in_cta, 0)]
            for ww in cutlass.range_constexpr(1, W):
                amax = fp.fmaxc(amax, xbuf[(row_in_cta, ww)])
        else:
            amax = acc
        m = pertoken_mult(fp, amax)
        s = pertoken_scale(fp, m)
        # the codes of the lane's vectors, converted once (the paired conversion), stored below to every destination
        outs = [cute.make_rmem_tensor((VEC,), Uint8) for _ in range(VPL)]
        for i in cutlass.range_constexpr(VPL):
            for q in cutlass.range_constexpr(0, VEC, 2):
                pr = cvt_f32x2_to_fp8x2(
                    pertoken_quant_elem(fp, frags[i][q].to(Float32), m),
                    pertoken_quant_elem(fp, frags[i][q + 1].to(Float32), m),
                    "e4m3",
                )
                outs[i][q] = (pr & 0xFF).to(Uint8)
                outs[i][q + 1] = (pr >> 8).to(Uint8)
        # the destinations: row t itself (k = 0) or the k dense rows of the token's slots
        n_dst = 1 if k == 0 else k
        for d in cutlass.range_constexpr(n_dst):
            if cutlass.const_expr(k == 0):
                p = t
                ok = valid_row
            else:
                p_raw = mDest[
                    cutlass.min(t * Int32(k) + Int32(d), mDest.shape[0] - Int32(1))
                ]
                ok = valid_row & (p_raw >= Int32(0))
                p = cutlass.max(p_raw, Int32(0))
            row_q = mQ[(p, None)]
            for i in cutlass.range_constexpr(VPL):
                j = (Int32(i * W) + w) * Int32(LANES) + lane
                jc = cutlass.min(j, Int32(NV - 1))
                if ok & (j < Int32(NV)):
                    cute.copy(atom_st, outs[i], cute.local_tile(row_q, (VEC,), (jc,)))
            if ok & (w == Int32(0)) & (lane == Int32(0)):
                mS[p] = s
        pdl.trigger_one_wave()


def kernel_name(K: int, W: int, k: int) -> str:
    return jit.sanitize_symbol_name(f"moe_e4m3pt_quant_K{K}_w{W}_k{k}")


@functools.cache
def get_compiled_pertoken_quant(K: int, W: int, k: int):
    """``kernel(x [M, K] bf16, q [R, K] u8, s [R] f32, dest [M k] i32 (any 1-element i32 tensor for k = 0), M)``: the per-token quantizer
    at row length ``K`` with ``W`` warps per row (``warps_per_row_quant(K)``), stores at the rows themselves (``k = 0``) or at the ``k`` dense
    rows ``dest[t k + j]`` of token ``t``."""

    def compile_fn():
        sym_m, sym_r, sym_n = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)
        x_fake = cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (sym_m, K),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        q_fake = cute.runtime.make_fake_tensor(
            Uint8, (sym_r, K), (cute.sym_int64(divisibility=16), 1), assumed_align=16
        )
        s_fake = cute.runtime.make_fake_tensor(Float32, (sym_r,), (1,), assumed_align=4)
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            PerTokenQuantE4m3Kernel(K, W, k),
            x_fake,
            q_fake,
            s_fake,
            d_fake,
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, kernel_name(K, W, k), compile_fn)
