"""Route reduction for the per-token MoE.

``Y[t, :] = sum_s weights[t, s] * X[row(t, s), :]`` in BF16, with ``row(t, s) = inverse[t * topk + s]``
(grouped layouts) or ``t * topk + s`` (direct GEMM outputs).

Short batches (``vec=4``):
- grid (tokens, H / block), ``block`` = 128 x warps columns, each thread 4 adjacent columns: block 512 / 4 warps for
  grouped layouts, 128 / 1 warp after the direct GEMM;
- the 8 route rows are read from ``inverse`` first, then (with PDL) ``griddepcontrol.launch_dependents``
  and ``wait``, so the index loads overlap the producer;
- per route one masked ``ld.global.v2.b32`` (4 BF16; ``mov 0`` then ``@p ld``, p = h < H, inline PTX [P4])
  and one weight load, in slot order 0, 2, 4, 6, 1, 3, 5, 7;
- the fixed FP32 order that keeps every schedule bit-identical: even and odd slots form two chains,
  ``fma(x0, w0, x2 * w2)`` then ``fma(x4, w4, .)``, ``fma(x6, w6, .)``, and the two chains are added (an odd
  topk: the even chain has one route more; a chain of one route is ``x * w``; topk 1: the even chain alone);
  all packed ``.rn.f32x2``;
- ``cvt.rn.bf16x2.f32`` and one ``st.global.v2.b32``.

The index and weight loads and the store are not predicated (the grid covers H exactly, asserted).
Inline PTX: ``HANDWRITTEN_PTX.md`` [P4].

Long batches (``vec=8``, ``reduce_shape``): each thread takes 8 columns with one 16-byte
``cute.arch.load`` per route, and the 8 routed rows and weights are read with two 16-byte loads
each. No inline PTX; the same per-element arithmetic, so both forms give the same output. ``reduce_shape`` uses the
4-column P4 form below 2048 tokens.
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .gemm import _asm, cvt_bf16x2, use_pdl

GMEM = cute.AddressSpace.gmem


def _pair_f32(words, p):
    """Columns 2p, 2p+1 of a loaded group as FP32 (``cvt.f32.bf16``), converted where they are used."""
    b = cute.recast_tensor(words, BFloat16)
    return (b[2 * p].to(Float32), b[2 * p + 1].to(Float32))


@dsl_user_op
def _ld_words_pred(addr, pred, *, loc=None, ip=None):
    """Masked 8-byte load [P4]: ``mov.b32 0`` into both words, then ``@p ld.global.v2.b32``.

    ptxas issues all eight route loads of this form before the math; written as a CuTeDSL load
    (``cute.arch.load``, or under a DSL ``if`` into a zeroed tensor) it moves two of them behind the
    first FMAs."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(pred).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p; setp.ne.b32 p, $3, 0; mov.b32 $0, 0; mov.b32 $1, 0; "
        "@p ld.global.v2.b32 {$0, $1}, [$2]; }",
        "=r,=r,l,r",
        True,
        loc,
        ip,
    )
    words = cute.make_rmem_tensor((2,), Uint32)
    for i in range(2):
        words[i] = Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip))
    return words


def reduce_shape(tokens: int, h: int = 2048):
    """(block, num_warps, vec) of the grouped reduction for ``tokens`` (the shape for h = 2048, see the module
    doc). For other h that shape is kept if its block divides h, else the widest block that does
    (h must be a multiple of 128, as the GEMM tiles require)."""
    if tokens >= 16384:
        block, warps, vec = 1024, 4, 8
    elif tokens >= 2048:
        block, warps, vec = 512, 2, 8
    else:
        block, warps, vec = 512, 4, 4
    if h % block == 0:
        return block, warps, vec
    assert h % 128 == 0, f"hidden size {h} must be a multiple of 128"
    for warps in (4, 2, 1):
        if h % (32 * vec * warps) == 0:
            return 32 * vec * warps, warps, vec
    return 128, 1, 4


def _ld_x4(base, index, dtype):
    """Four consecutive 32-bit values at ``base + 4 index`` (16-byte aligned) with one ``ld.global.v4``."""
    p = cute.make_ptr(dtype, base + Int64(index) * 4, GMEM, assumed_align=16)
    v = cute.TensorSSA(
        cute.arch.load(p, ir.VectorType.get([4], dtype.mlir_type)), (4,), dtype
    )
    return [v[i] for i in range(4)]


class ReduceRoutes:
    """Weighted sum of each token's ``topk`` routed rows.

    Args:
        h: Hidden size (columns).
        topk: Routes per token (8).
        block: Columns per CTA (``vec`` per thread).
        num_warps: Warps per CTA (block / (32 vec)).
        inverse: Rows come from ``inverse[t * topk + s]``; else row ``t * topk + s``.
        launch_pdl: ``griddepcontrol.launch_dependents`` and ``wait`` after the index loads.
        vec: Columns per thread: 4 (P4 loads) or 8 (16-byte interface loads).
        clear: CTA (0, 0) zeroes ``clear`` Int32 at ``clr_ptr`` after the wait: the expert route counts of
            ``SelectExperts(ranks=True)`` (read by the scatter, long done), ready for the next call.
    """

    def __init__(
        self, h, topk, block, num_warps, *, inverse, launch_pdl, vec=4, clear=0
    ):
        # even and odd slots form two chains of topk / 2 routes; an odd topk gives the even chain one route more
        # (topk 1: the even chain alone)
        assert topk >= 1, f"topk {topk}"
        assert vec in (4, 8) and block == 32 * vec * num_warps and h % block == 0
        self.h, self.topk, self.block, self.num_warps = h, topk, block, num_warps
        self.inverse, self.launch_pdl, self.vec = inverse, launch_pdl, vec
        assert clear == 0 or launch_pdl, "the counts are zeroed after the wait"
        self.clear = clear

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        inv_ptr: cute.Pointer,
        clr_ptr: cute.Pointer,
        tokens: Int32,
        stream,
    ):
        w = cute.make_tensor(w_ptr, cute.make_layout(tokens * self.topk))
        inv = cute.make_tensor(inv_ptr, cute.make_layout(tokens * self.topk))
        self.kernel(x_ptr, w, y_ptr, inv, clr_ptr).launch(
            grid=[tokens, self.h // self.block, 1],
            block=[32 * self.num_warps, 1, 1],
            stream=stream,
            use_pdl=use_pdl(self),
        )

    @cute.kernel
    def kernel(
        self,
        x_ptr: cute.Pointer,
        w: cute.Tensor,
        y_ptr: cute.Pointer,
        inv: cute.Tensor,
        clr_ptr: cute.Pointer,
    ):
        H, T = self.h, self.topk
        token, blk, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        first = token * T
        V = self.vec
        wide = V == 8
        if cutlass.const_expr(self.inverse and wide and T % 4 == 0):
            rows = []
            for q in cutlass.range_constexpr(T // 4):
                rows += _ld_x4(inv.iterator.toint(), first + 4 * q, Int32)
        elif cutlass.const_expr(self.inverse):
            rows = [inv[first + s] for s in range(T)]
        else:
            rows = [first + s for s in range(T)]
        if cutlass.const_expr(self.launch_pdl):
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.clear > 0):
            if (token == 0) & (blk == 0):
                NTH = 32 * self.num_warps
                clr = cute.make_tensor(
                    cute.make_ptr(Int32, clr_ptr.toint(), GMEM, assumed_align=4),
                    cute.make_layout(self.clear),
                )
                for i in cutlass.range_constexpr((self.clear + NTH - 1) // NTH):
                    if tidx + i * NTH < self.clear:
                        clr[tidx + i * NTH] = Int32(0)
        col = blk * self.block + tidx * V
        xb = x_ptr.toint()
        order = list(range(0, T, 2)) + list(range(1, T, 2))
        xs, ws = {}, {}
        colb = Int64(col) * 2
        live = Int32(blk < self.h // self.block)  # h < H (always true for this grid)
        if cutlass.const_expr(wide and T % 4 == 0):
            wv = []
            for q in cutlass.range_constexpr(T // 4):
                wv += _ld_x4(w.iterator.toint(), first + 4 * q, Float32)
        elif cutlass.const_expr(wide):
            wv = [w[first + s] for s in range(T)]
        for i in cutlass.range_constexpr(T):
            s = order[i]
            addr = xb + Int64(rows[s] * H) * 2 + colb
            if cutlass.const_expr(wide):
                words = cute.make_rmem_tensor((4,), Uint32)
                words.store(
                    cute.TensorSSA(
                        cute.arch.load(
                            cute.make_ptr(Uint32, addr, GMEM, assumed_align=16),
                            ir.VectorType.get([4], Uint32.mlir_type),
                        ),
                        (4,),
                        Uint32,
                    )
                )
                xs[s] = words
                ws[s] = wv[s]
            else:
                xs[s] = _ld_words_pred(addr, live)
                ws[s] = w[first + s]
        out = []
        for p in cutlass.range_constexpr(V // 2):  # column pairs (0, 1), (2, 3), ...
            chains = []
            for start in cutlass.range_constexpr(
                min(T, 2)
            ):  # even, odd slots (topk 1: no odd chain)
                slots = list(range(start, T, 2))
                a = slots[0]
                if cutlass.const_expr(len(slots) == 1):  # one route in the chain
                    acc = cute.arch.mul_packed_f32x2(
                        (ws[a], ws[a]), _pair_f32(xs[a], p), rnd="rn"
                    )
                else:
                    b = slots[1]
                    acc = cute.arch.mul_packed_f32x2(
                        (ws[b], ws[b]), _pair_f32(xs[b], p), rnd="rn"
                    )
                    acc = cute.arch.fma_packed_f32x2(
                        _pair_f32(xs[a], p), (ws[a], ws[a]), acc, rnd="rn"
                    )
                for j in cutlass.range_constexpr(2, len(slots)):
                    s = slots[j]
                    acc = cute.arch.fma_packed_f32x2(
                        _pair_f32(xs[s], p), (ws[s], ws[s]), acc, rnd="rn"
                    )
                chains.append(acc)
            if cutlass.const_expr(T == 1):
                y = chains[0]
            else:
                y = cute.arch.add_packed_f32x2(chains[0], chains[1], rnd="rn")
            out.append(cvt_bf16x2(y[0], y[1], packed=True))
        r = cute.make_rmem_tensor((V // 2,), Uint32)
        for i in cutlass.range_constexpr(V // 2):
            r[i] = out[i]
        dst = cute.make_ptr(
            Uint32,
            y_ptr.toint() + (Int64(token) * H + col) * 2,
            GMEM,
            assumed_align=2 * V,
        )
        cute.arch.store(dst, r.load().ir_value())
