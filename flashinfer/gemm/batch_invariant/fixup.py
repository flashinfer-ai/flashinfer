# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Split-K fixup: ``out[m, n] = chain( ((+0 + W[m,n,0]) + W[m,n,1]) + ... )`` over the C fp32 chunk partials, in
chunk order (the ``chunk`` fold digit of the chunked specification), the chain being the class's epilogue chain
(``cvt.rn`` for bf16 and the MXFP8 classes; ``mul.rn alpha`` then ``cvt.rn`` for the fp8 classes, ``alpha = mul.rn(a_scale,
b_scale)`` formed on the device from the two per-tensor scales, and for the 4-bit classes, whose ``alpha`` is the caller's
scalar; ``add.rn bias`` of the output column then ``cvt.rn`` for the
biased bf16 class: the bias enters once, here, after the whole chunk fold, and the partials are bias-free). All math through ``CuteF32``; vectorised over n with 16-byte
loads; columns predicated; the grid's y extent is capped (``MAX_GRID_Y``) and each block strides over the rows, so
any M launches."""

from __future__ import annotations

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ...batch_invariant import jit
from ...batch_invariant.fp_interface.cute import CuteF32, get_cutlass_dtype
from .blocks.compile import MODULE
from .spec import EpilogueChain

VEC = 4  # fp32 per 16-byte load
THREADS = 256
MAX_GRID_Y = 65535  # the hardware's grid.y limit; rows beyond it are strided over


class SplitKFixup:
    def __init__(self, C: int, out_dtype, chain: EpilogueChain):
        self.C, self.out_dtype, self.chain = C, out_dtype, chain

    @cute.jit
    def __call__(
        self, mW: cute.Tensor, mY: cute.Tensor, M: Int32, N: Int32, rows: Int32, stream
    ):
        """``rows = min(M, MAX_GRID_Y)`` is the grid's y extent (chosen by the caller); block ``by`` handles rows ``by, by + rows, ...``."""
        n_vec = cute.ceil_div(N, VEC)
        self.kernel(mW, mY, None, None, None, M, N, n_vec, rows).launch(
            grid=[cute.ceil_div(n_vec, THREADS), rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mSa: Optional[cute.Tensor],
        mSb: Optional[cute.Tensor],
        mBias: Optional[cute.Tensor],
        M: Int32,
        N: Int32,
        n_vec: Int32,
        rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        v = bx * THREADS + tidx
        alpha = None
        if cutlass.const_expr(self.chain.scaled):
            if cutlass.const_expr(mSb is None):
                alpha = mSa[0]  # the 4-bit classes: the caller's scalar as given
            else:
                alpha = CuteF32.mul(
                    mSa[0], mSb[0]
                )  # alpha = mul.rn(a_scale, b_scale), once per block
        if v < n_vec:
            n0 = v * VEC
            for m in cutlass.range(by, M, rows, unroll=1):
                for j in cutlass.range_constexpr(VEC):
                    n = n0 + j
                    if n < N:
                        acc = CuteF32.const(0.0)
                        for c in cutlass.range_constexpr(self.C):
                            acc = CuteF32.add(acc, mW[m, n, c])
                        # the chain once, after the chunk fold: the fp32 ops, then the single conversion
                        bias = None
                        if cutlass.const_expr(self.chain.biased):
                            bias = CuteF32.widen(mBias[n])
                        mY[m, n] = self.chain.apply_pre(CuteF32, acc, alpha, bias).to(
                            mY.element_type
                        )


class SplitKFixupScaled(SplitKFixup):
    """The fp8 classes' fixup: ``alpha = mul.rn(a_scale, b_scale)`` then the chain ``mul.rn alpha``, ``cvt.rn``."""

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mSa: cute.Tensor,
        mSb: cute.Tensor,
        M: Int32,
        N: Int32,
        rows: Int32,
        stream,
    ):
        n_vec = cute.ceil_div(N, VEC)
        self.kernel(mW, mY, mSa, mSb, None, M, N, n_vec, rows).launch(
            grid=[cute.ceil_div(n_vec, THREADS), rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


class SplitKFixupAlpha(SplitKFixup):
    """The 4-bit classes' fixup: the chain ``mul.rn alpha`` with the caller's 1-element alpha, then ``cvt.rn``."""

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mAlpha: cute.Tensor,
        M: Int32,
        N: Int32,
        rows: Int32,
        stream,
    ):
        n_vec = cute.ceil_div(N, VEC)
        self.kernel(mW, mY, mAlpha, None, None, M, N, n_vec, rows).launch(
            grid=[cute.ceil_div(n_vec, THREADS), rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


class SplitKFixupBias(SplitKFixup):
    """The biased bf16 class's fixup: the chain ``add.rn bias`` of the output column, then ``cvt.rn``; the (N,) bf16 bias
    is one more operand."""

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mBias: cute.Tensor,
        M: Int32,
        N: Int32,
        rows: Int32,
        stream,
    ):
        n_vec = cute.ceil_div(N, VEC)
        self.kernel(mW, mY, None, None, mBias, M, N, n_vec, rows).launch(
            grid=[cute.ceil_div(n_vec, THREADS), rows, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


@functools.cache
def get_fixup(
    C: int,
    out_dtype_name: str = "bf16",
    scaled: bool = False,
    bias: bool = False,
    alpha_given: bool = False,
):
    """One fixup kernel per chunk count, output format and chain; M and N are runtime arguments. ``scaled``: the fp8
    classes' chain (two per-tensor scales); ``alpha_given``: the 4-bit classes' chain (one scalar)."""
    assert not (scaled and bias) and not (alpha_given and (scaled or bias)), (
        "one chain per class"
    )
    out_dt = get_cutlass_dtype(out_dtype_name)
    chain = EpilogueChain.for_class(out_dtype_name, scaled or alpha_given, bias)
    name = jit.sanitize_symbol_name(
        f"splitk_fixup_C{C}_{out_dtype_name}"
        + ("_scaled" if scaled else "")
        + ("_alpha" if alpha_given else "")
        + ("_bias" if bias else "")
    )

    def compile_fn():
        sym_m, sym_n = cute.sym_int(64), cute.sym_int(32, divisibility=4)
        w_fake = cute.runtime.make_fake_tensor(
            Float32,
            (sym_m, sym_n, C),
            (cute.sym_int64(divisibility=4), 1, cute.sym_int64(divisibility=4)),
            assumed_align=16,
        )
        y_fake = cute.runtime.make_fake_tensor(
            out_dt,
            (sym_m, sym_n),
            (cute.sym_int64(divisibility=8), 1),
            assumed_align=16,
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        if bias:
            bias_fake = cute.runtime.make_fake_tensor(
                get_cutlass_dtype("bf16"), (sym_n,), (1,), assumed_align=16
            )
            return cute.compile(
                SplitKFixupBias(C, out_dt, chain),
                w_fake,
                y_fake,
                bias_fake,
                Int32(1),
                Int32(1),
                Int32(1),
                stream_fake,
                options="--enable-tvm-ffi",
            )
        if alpha_given:
            alpha_fake = cute.runtime.make_fake_tensor(
                Float32, (1,), (1,), assumed_align=4
            )
            return cute.compile(
                SplitKFixupAlpha(C, out_dt, chain),
                w_fake,
                y_fake,
                alpha_fake,
                Int32(1),
                Int32(1),
                Int32(1),
                stream_fake,
                options="--enable-tvm-ffi",
            )
        if scaled:
            sa_fake = cute.runtime.make_fake_tensor(
                Float32, (1,), (1,), assumed_align=4
            )
            sb_fake = cute.runtime.make_fake_tensor(
                Float32, (1,), (1,), assumed_align=4
            )
            return cute.compile(
                SplitKFixupScaled(C, out_dt, chain),
                w_fake,
                y_fake,
                sa_fake,
                sb_fake,
                Int32(1),
                Int32(1),
                Int32(1),
                stream_fake,
                options="--enable-tvm-ffi",
            )
        return cute.compile(
            SplitKFixup(C, out_dt, chain),
            w_fake,
            y_fake,
            Int32(1),
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, name, compile_fn)
