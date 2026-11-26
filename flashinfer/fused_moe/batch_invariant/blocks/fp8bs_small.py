# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 MoE class's small kernels: the standalone ``fp8bs.quant`` kernel (rows of 16-bit values to e4m3 codes and
one fp32 scale per 128-element block, the scales in the ``MN`` layout ``(K / 128, M)`` the block-scale GEMM's row-scale ring
reads or in the ``K`` layout ``(M, K / 128)``: a placement of the store, one kernel, the tensor's own strides) and the gather
of the activations' block scales to the expanded rows (``sf_perm[kb, p] = sf[kb, src[p] // k]``: a copy, no arithmetic).

Quantizer placement: a warp pass covers two blocks (sixteen lanes per block, eight consecutive elements per lane: one 16-byte
load, one 8-byte store); ``wpr`` warps share a row and take its block pairs round-robin (a placement: every block's amax is
the plan's tree ``[(8, reg), (16, lane)]`` through ``canon.init_state`` / ``canon.run`` whichever warp holds it). ``K`` and
the row strides are runtime arguments; a kernel is compiled per (input dtype, warps per row, PDL, block). Nothing
here decides a numeric order: the reduction is the engine's plan, the chains are :mod:`..spec_fp8bs`'s, the conversion the
declared ``cvt.rn.satfinite.e4m3x2.f32``.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8, Uint32

from ....batch_invariant import canon, jit
from ....batch_invariant.fp_interface.cute import get_cutlass_dtype
from ....batch_invariant.quantization.blocks import LanePeers, pack_block_codes
from ....batch_invariant.quantization.cute_ext import CuteF32
from .. import pdl
from ..spec_fp8bs import (
    BLOCK,
    block_amax_placement,
    block_chain,
    block_reduction,
    check_block,
    leaf_value,
    quant_pre,
)

WARPS = 8  # warps per CTA
LANES = 32
V = 8  # elements per lane: one 16-byte load of 16-bit values, one block per sixteen lanes
LANES_PER_BLOCK = BLOCK // V
SCATTER_THREADS = 256
SCALE_BLOCKS_PER_THREAD = (
    8  # block columns per thread of the scale gather, their loads issued together
)
#: The fused row + scale gather runs the row gather's block (``permute.GatherKernel``: ``WARPS`` warps of ``LANES`` lanes).
GATHER_WARPS = 4
#: The MoE classes' one JIT module (the bodies' cache key: an edit here moves every object's key).
MODULE = "batch_invariant_moe"
GATHER_LANES = 32


def lanes_per_block(block: int) -> int:
    return check_block(block) // V


def blocks_per_pass(block: int) -> int:
    """Blocks one warp pass of the quantizer covers: 2 at the 128-wide class, 8 at the 32-wide one."""
    return LANES // lanes_per_block(block)


class _Loader:
    """``canon.init_state`` leaf loader: the spec's leaf (``fabs`` of the widened element) for the register key."""

    def __init__(self, frag):
        self.frag = frag

    def __call__(self, key):
        return leaf_value(CuteF32, self.frag[key[0]].to(Float32))


class BSQuantKernel:
    """``codes[m, k] = cvt_e4m3(mul(x[m, k], inv(m, k // block)))``, ``s[m, k // block]`` for rows of 16-bit values; ``block``
    128 (sixteen lanes per block, two blocks per warp pass) or 32 (four lanes per block, eight blocks per pass)."""

    def __init__(self, dtype: cutlass.Numeric, wpr: int, block: int = BLOCK):
        assert dtype.width == 16, dtype
        assert wpr >= 1 and wpr & (wpr - 1) == 0 and wpr <= 8, wpr
        self.dtype, self.wpr, self.block = dtype, wpr, check_block(block)
        self.lpb, self.bpp = lanes_per_block(block), blocks_per_pass(block)
        self.red = block_reduction(leaf_x=None, block=block)
        self.plan = canon.plan(self.red, block_amax_placement(V, block))
        assert self.plan.units["lane"] == self.lpb and self.plan.n_reg == 1

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(mX, mY, mS, M, K, enable_pdl).launch(
            grid=[cute.ceil_div(M * Int32(self.wpr), WARPS), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
        K: Int32,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(enable_pdl):
            cute.arch.griddepcontrol_wait()
            pdl.trigger_one_wave()  # the one-wave kernels' shared site, right after the wait
        self.rows(mX, mY, mS, M, K, bidx * WARPS + tidx // LANES)

    @cute.jit
    def rows(
        self,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
        K: Int32,
        g: Int32,
    ):
        """The quantizer's warp ``g`` (row ``g // wpr``, part ``g % wpr``): the body of :meth:`kernel`, also run by the plan +
        quantizer launch (:class:`PlanQuantKernel`) in the CTAs after the plan's."""
        fp = CuteF32
        wpr = self.wpr
        lane = cute.arch.lane_idx()
        row_raw = g // Int32(wpr)
        part = g % Int32(wpr)
        valid = row_raw < M
        row = cutlass.min(row_raw, M - Int32(1))
        lpb, bpp = self.lpb, self.bpp
        which = lane // Int32(lpb)  # which of the pass's blocks this lane holds
        sub = lane % Int32(lpb)  # the lane's eight elements inside the block
        nb = K // Int32(self.block)
        n_groups = (nb + Int32(bpp - 1)) // Int32(bpp)  # block groups of one pass
        n_mine = (n_groups - part + Int32(wpr - 1)) // Int32(wpr)
        atom_x = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=V * mX.element_type.width,
        )
        atom_y = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=8 * V
        )
        row_x = mX[(row, None)]  # 1-D row views (a (1, V) tile mis-vectorises)
        row_y = mY[(row, None)]
        frag = cute.make_rmem_tensor((V,), mX.element_type)
        fy_words = cute.make_rmem_tensor(cute.make_layout((V // 4,)), Uint32)
        fy = cute.make_tensor(
            cute.recast_ptr(fy_words.iterator, dtype=Uint8), cute.make_layout((V,))
        )
        for jj in cutlass.range(0, n_mine, 1):
            blk = (jj * Int32(wpr) + part) * Int32(bpp) + which
            live = valid & (blk < nb)
            blk_c = cutlass.min(blk, nb - Int32(1))
            chunk = blk_c * Int32(lpb) + sub
            cute.copy(
                atom_x, cute.local_tile(row_x, (V,), (chunk,)), frag
            )  # the lane's eight elements (clamped: a dead lane's bytes are never stored)
            # ===== the block amax: the engine's plan (the register tree fused with the load, then the xor butterfly) =====
            state, start = canon.init_state(self.plan, fp, _Loader(frag))
            amax = canon.run(self.plan, state, start, LanePeers(), fp)
            # ===== the chains: the scale, its reciprocal, the element conversion =====
            s, inv = block_chain(fp, amax)
            ys = [quant_pre(fp, frag[v].to(Float32), inv) for v in range(V)]
            for w, word in enumerate(pack_block_codes(fp, ys, "e4m3")):
                fy_words[w] = word
            if live:
                cute.copy(atom_y, fy, cute.local_tile(row_y, (V,), (chunk,)))
            if live & (sub == Int32(0)):
                mS[(row, blk_c)] = s


class PlanQuantKernel:
    """The pre-routed permutation plan and the activation quantizer in one launch: CTAs ``[0, E)`` run the plan's CTA
    (:meth:`..permute.PlanKernel.cta` at this kernel's thread count), the CTAs after them the quantizer's warps
    (:meth:`BSQuantKernel.rows`). The two read disjoint inputs (the expert ids; the bf16 rows) and write disjoint outputs, so
    the launch is a placement of the two kernels: the same integers, the same bits. Every CTA waits for the predecessor
    grid before its reads (the plan's rule) and none triggers."""

    THREADS = WARPS * LANES

    def __init__(self, dtype: cutlass.Numeric, wpr: int, block: int = BLOCK):
        from ..permute import PlanKernel

        self.quant = BSQuantKernel(dtype, wpr, block)
        self.plan = PlanKernel(threads=self.THREADS)

    @cute.jit
    def __call__(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        n: Int32,
        E: Int32,
        M: Int32,
        K: Int32,
        stream,
    ):
        n_quant = cute.ceil_div(M * Int32(self.quant.wpr), WARPS)
        self.kernel(mExp, mCounts, mDest, mSrc, mX, mY, mS, n, E, M, K).launch(
            grid=[E + n_quant, 1, 1],
            block=[self.THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mX: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        n: Int32,
        E: Int32,
        M: Int32,
        K: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        if bidx < E:
            self.plan.cta(mExp, mCounts, mDest, mSrc, mDest, mDest, n, E, bidx, tidx)
        else:
            self.quant.rows(mX, mY, mS, M, K, (bidx - E) * WARPS + tidx // LANES)


class GatherRowsScalesKernel:
    """``x_perm[p, :] = x[src[p] // k, :]`` (the first ``n_rows_blocks`` blocks: ``wpr`` warps per expanded row, 16-byte vectors,
    the row gather's body) and ``sf_perm[kb, p] = sf[kb, src[p] // k]`` (the remaining blocks: thread = expanded row,
    ``SCALE_BLOCKS_PER_THREAD`` block columns per thread, the scale gather's body) in one launch; ``-1`` entries (dropped slots)
    gather nothing. A placement of the two copies: one launch on the chain instead of two."""

    def __init__(self, k: int, wpr: int = 1):
        assert wpr >= 1 and wpr & (wpr - 1) == 0, wpr
        self.k, self.wpr = k, wpr

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSrc: cute.Tensor,
        mXp: cute.Tensor,
        mS: cute.Tensor,
        mSp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        KB: Int32,
        stream,
    ):
        n_rows_blocks = cute.ceil_div(Tk * Int32(self.wpr), GATHER_WARPS)
        n_scale_blocks = cute.ceil_div(Tk, GATHER_WARPS * GATHER_LANES) * cute.ceil_div(
            KB, SCALE_BLOCKS_PER_THREAD
        )
        self.kernel(mX, mSrc, mXp, mS, mSp, Tk, B, KB, n_rows_blocks).launch(
            grid=[n_rows_blocks + n_scale_blocks, 1, 1],
            block=[GATHER_WARPS * GATHER_LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mSrc: cute.Tensor,
        mXp: cute.Tensor,
        mS: cute.Tensor,
        mSp: cute.Tensor,
        Tk: Int32,
        B: Int32,
        KB: Int32,
        n_rows_blocks: Int32,
    ):
        wpr = self.wpr
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        if bidx < n_rows_blocks:
            # the rows (``permute.GatherKernel``'s body on the e4m3 bytes)
            g = bidx * GATHER_WARPS + tidx // GATHER_LANES
            p_raw = g // Int32(wpr)
            part = g % Int32(wpr)
            valid = p_raw < Tk
            p = cutlass.min(p_raw, Tk - Int32(1))
            src = mSrc[p]
            valid = valid & (src >= Int32(0))  # -1: a dropped slot, nothing to gather
            t = cutlass.max(src, Int32(0)) // Int32(self.k)
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Uint8, num_bits_per_copy=128
            )
            n_vec = B // Int32(16)
            frag = cute.make_rmem_tensor((16,), Uint8)
            row_x = mX[(t, None)]
            row_p = mXp[(p, None)]
            n_slabs = (n_vec + Int32(GATHER_LANES - 1)) // Int32(GATHER_LANES)
            n_mine = (n_slabs - part + Int32(wpr - 1)) // Int32(wpr)
            for jj in cutlass.range(0, n_mine, 1):
                j = (jj * Int32(wpr) + part) * GATHER_LANES + lane
                jc = cutlass.min(j, n_vec - Int32(1))
                cute.copy(atom, cute.local_tile(row_x, (16,), (jc,)), frag)
                if valid & (j < n_vec):
                    cute.copy(atom, frag, cute.local_tile(row_p, (16,), (jc,)))
        else:
            # the scale columns (thread = expanded row, its block columns' loads issued together)
            b = bidx - n_rows_blocks
            n_pb = cute.ceil_div(Tk, GATHER_WARPS * GATHER_LANES)
            bx = b % n_pb
            by = b // n_pb
            p_raw = bx * Int32(GATHER_WARPS * GATHER_LANES) + tidx
            valid = p_raw < Tk
            p = cutlass.min(p_raw, Tk - Int32(1))
            src = mSrc[p]
            valid = valid & (src >= Int32(0))
            tt = cutlass.max(src, Int32(0)) // Int32(self.k)
            kb0 = by * SCALE_BLOCKS_PER_THREAD
            vals = []
            for j in cutlass.range_constexpr(SCALE_BLOCKS_PER_THREAD):
                vals.append(mS[(cutlass.min(kb0 + j, KB - Int32(1)), tt)])
            for j in cutlass.range_constexpr(SCALE_BLOCKS_PER_THREAD):
                kb = kb0 + j
                if valid:
                    if kb < KB:
                        mSp[(kb, p)] = vals[j]
        pdl.trigger_one_wave()


# ----------------------------------------------------------------------------- the compile wrappers (module ``batch_invariant_moe``)


def quant_kernel_name(
    dtype_str: str, wpr: int, enable_pdl: bool = True, block: int = BLOCK
) -> str:
    """The standalone quantizer's name: the input dtype, the warps per row (a placement), PDL and the block size where it is
    not the 128-wide class's; never M or K (the trigger placement is the module constant of :mod:`..pdl`)."""
    return jit.sanitize_symbol_name(
        f"fp8bs_quant_{dtype_str}_w{wpr}_pdl{int(enable_pdl)}"
        + (f"_b{block}" if check_block(block) != BLOCK else "")
    )


def gather_rows_scales_name(k: int, wpr: int) -> str:
    return jit.sanitize_symbol_name(
        f"moe_gather_rows_scales_k{k}" + (f"_w{wpr}" if wpr > 1 else "")
    )


@functools.cache
def get_compiled_fp8bs_quant(
    dtype_str: str, wpr: int, enable_pdl: bool = True, block: int = BLOCK
):
    """``kernel(x [M, K] 16-bit, codes [M, K] u8, s (M, KB) f32 with free strides, M, K)``; one kernel per (dtype, warps per row,
    PDL, block), in the MoE classes' module."""
    dtype = get_cutlass_dtype(dtype_str)
    name = quant_kernel_name(dtype_str, wpr, enable_pdl, block)

    def compile_fn(**compile_options):
        sym_m, sym_k, sym_kb = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=block),
            cute.sym_int(32),
        )
        x = cute.runtime.make_fake_tensor(
            dtype, (sym_m, sym_k), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
        y = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_m, sym_k),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        s = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            BSQuantKernel(dtype, wpr, block),
            x,
            y,
            s,
            Int32(1),
            Int32(block),
            enable_pdl,
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(MODULE, name, compile_fn)


def plan_quant_kernel_name(dtype_str: str, wpr: int, block: int = BLOCK) -> str:
    """The plan + quantizer launch's name: the quantizer's input dtype, warps per row and block (the dense plan)."""
    return jit.sanitize_symbol_name(
        f"fp8bs_plan_quant_{dtype_str}_w{wpr}"
        + (f"_b{block}" if check_block(block) != BLOCK else "")
    )


@functools.cache
def get_compiled_plan_quant(dtype_str: str, wpr: int, block: int = BLOCK):
    """``kernel(experts [T k] i32, counts [E] i32 (written), dest [T k] i32, src [T k] i32, x [M, K] 16-bit, codes [M, K] u8,
    s (M, KB) f32 with free strides, n, E, M, K)``: the dense permutation plan and the quantizer in one launch."""
    dtype = get_cutlass_dtype(dtype_str)

    def compile_fn(**compile_options):
        sym_n, sym_e = cute.sym_int(32), cute.sym_int(32)
        sym_m, sym_k, sym_kb = (
            cute.sym_int(32),
            cute.sym_int(32, divisibility=block),
            cute.sym_int(32),
        )
        e_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        s_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        x = cute.runtime.make_fake_tensor(
            dtype, (sym_m, sym_k), (cute.sym_int64(divisibility=8), 1), assumed_align=16
        )
        y = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_m, sym_k),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        sc = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_kb),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            PlanQuantKernel(dtype, wpr, block),
            e_fake,
            c_fake,
            d_fake,
            s_fake,
            x,
            y,
            sc,
            Int32(1),
            Int32(1),
            Int32(1),
            Int32(block),
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(
        MODULE, plan_quant_kernel_name(dtype_str, wpr, block), compile_fn
    )


@functools.cache
def get_compiled_gather_rows_scales(k: int, wpr: int = 1):
    """``kernel(x_bytes [T, B] u8, src [>= Tk] i32, x_perm [>= Tk, B] u8, sf (KB, T) f32 any strides, sf_perm (KB, >= Tk) f32,
    Tk, B, KB)``: the fused row + scale gather (one launch); ``wpr`` warps per row (``permute.warps_per_row``)."""

    def compile_fn(**compile_options):
        sym_t, sym_n, sym_p = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)
        sym_b = cute.sym_int(32, divisibility=16)
        sym_kb, sym_t2 = cute.sym_int(32), cute.sym_int(32)
        x_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_t, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        src_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        xp_fake = cute.runtime.make_fake_tensor(
            Uint8,
            (sym_p, sym_b),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        s_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_kb, sym_t2),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=4,
        )
        sp_fake = cute.runtime.make_fake_tensor(
            Float32, (sym_kb, sym_p), (cute.sym_int64(), 1), assumed_align=4
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            GatherRowsScalesKernel(k, wpr),
            x_fake,
            src_fake,
            xp_fake,
            s_fake,
            sp_fake,
            Int32(1),
            Int32(16),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
            **compile_options,
        )

    return jit.build_and_load(MODULE, gather_rows_scales_name(k, wpr), compile_fn)
