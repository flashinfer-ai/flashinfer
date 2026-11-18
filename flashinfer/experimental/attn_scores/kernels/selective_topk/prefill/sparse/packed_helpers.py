# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cutlass

import cutlass.cute as cute


from cutlass import Int32


from cutlass._mlir.dialects import llvm

from cutlass._mlir.dialects import llvm as mlir_llvm


from cutlass.cutlass_dsl import T, dsl_user_op


RES_B = 0

RES_M = 1

RES_ABOVE = 2

RES_TOT = 3


def warp_add_i32(v):
    """__reduce_add_sync on Int32 (scan_cross_w two-redux stage)."""
    return cute.arch.warp_redux_sync(v, "add")


def ballot(pred):
    """__ballot_sync(FULLM, pred) -> Int32 mask."""
    return cute.arch.vote_ballot_sync(pred)


def clz_i32(x):
    """__clz as Int32."""
    return cutlass.Int32(cute.arch.clz(x))


@cute.jit
def hi_bit_or_zero(msk):
    """CUDA `msk ? (31 - __clz(msk)) : 0`."""
    r = cutlass.Int32(0)
    if msk != cutlass.Int32(0):
        r = cutlass.Int32(31) - clz_i32(msk)
    return r


@cute.jit
def _shfl_up_add(val, lane, offset: cutlass.Constexpr):
    """Inclusive scan step; lanes below offset discard the shuffled value."""
    other = cute.arch.shuffle_sync_up(val, offset, mask_and_clamp=0)
    if lane >= cutlass.Int32(offset):
        val = val + other
    return val


@cute.jit
def _shfl_down_add(val, lane, offset: cutlass.Constexpr):
    """Suffix scan step; lanes at or beyond 32 - offset discard the shuffled value."""
    other = cute.arch.shuffle_sync_down(val, offset, mask_and_clamp=31)
    if lane + cutlass.Int32(offset) < cutlass.Int32(32):
        val = val + other
    return val


@cute.jit
def warp_incl_scan_add(val, lane):
    """5-step inclusive __shfl_up_sync add scan."""
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_up_add(val, lane, o)
    return val


@cute.jit
def warp_suffix_scan_add(val, lane):
    """5-step __shfl_down_sync suffix add scan."""
    for o in [1, 2, 4, 8, 16]:
        val = _shfl_down_add(val, lane, o)
    return val


@cute.jit
def scan_cross_w(
    s_hist,
    s_ws,
    target,
    tidx,
    s_res,
    blk: cutlass.Constexpr,
    nb: cutlass.Constexpr,
    vector: cutlass.Constexpr = False,
):
    BPT = nb // blk
    NW = blk // 32
    lane = tidx & cutlass.Int32(31)
    wid = tidx >> cutlass.Int32(5)
    loc = cute.make_rmem_tensor((BPT,), cutlass.Int32)
    base = tidx * cutlass.Int32(BPT)
    sm = cutlass.Int32(0)
    if cutlass.const_expr(vector and BPT in (2, 4)):
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=32 * BPT
        )
        hist_span = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int32,
                s_hist.iterator.toint() + base * cutlass.Int32(4),
                cute.AddressSpace.smem,
                assumed_align=4 * BPT,
            ),
            cute.make_layout((BPT,)),
        )
        cute.copy(atom, hist_span, loc)
    else:
        for i in cutlass.range_constexpr(BPT):
            loc[i] = s_hist[base + cutlass.Int32(i)]
    for i in cutlass.range_constexpr(BPT):
        sm = sm + loc[i]
    w = warp_incl_scan_add(sm, lane)
    if lane == cutlass.Int32(31):
        s_ws[wid] = w
    cute.arch.barrier()  # Synchronize shared scan results.
    vv = cutlass.Int32(0)
    if lane < cutlass.Int32(NW):
        vv = s_ws[lane]
    tot = cutlass.Int32(warp_add_i32(vv))
    sel = cutlass.Int32(0)
    if lane < wid:
        sel = vv
    off = cutlass.Int32(warp_add_i32(sel))
    after = tot - (off + w)
    if tidx == cutlass.Int32(0):
        s_res[RES_TOT] = tot
    for i in cutlass.range_constexpr(BPT - 1, -1, -1):
        cq = loc[i]
        if cutlass.const_expr(vector and BPT in (2, 4)):
            loc[i] = after
        else:
            s_hist[base + Int32(i)] = after
        gb = base + cutlass.Int32(i)
        cross = cutlass.Int32(0)
        if after < target:
            if (after + cq) >= target:
                cross = cutlass.Int32(1)
            if gb == cutlass.Int32(0):
                cross = cutlass.Int32(1)
        if cross != cutlass.Int32(0):
            s_res[RES_B] = gb
            s_res[RES_ABOVE] = after
            s_res[RES_M] = cq
        after = after + cq

    if cutlass.const_expr(vector and BPT in (2, 4)):
        cute.copy(atom, loc, hist_span)


@cute.jit
def scan_cross_w_parts(
    s_hx,
    s_ws,
    target,
    tidx,
    s_res,
    part,
    blk: cutlass.Constexpr,
    nb: cutlass.Constexpr,
    parts: cutlass.Constexpr,
):
    BPT = max(1, nb // blk)
    NW = blk // 32
    lane = tidx & cutlass.Int32(31)
    wid = tidx >> cutlass.Int32(5)
    loc = cute.make_rmem_tensor((BPT,), cutlass.Int32)
    pre = cute.make_rmem_tensor((BPT,), cutlass.Int32)
    base = tidx * cutlass.Int32(BPT)
    sm = cutlass.Int32(0)
    for i in cutlass.range_constexpr(BPT):
        t = cutlass.Int32(0)
        pr = cutlass.Int32(0)
        if cutlass.const_expr(nb >= blk) or tidx < cutlass.Int32(nb):
            for r in cutlass.range_constexpr(parts):
                v = s_hx[cutlass.Int32(r * nb) + base + cutlass.Int32(i)]
                t = t + v
                if cutlass.Int32(r) < part:  # predicated add
                    pr = pr + v
        loc[i] = t
        pre[i] = pr
    for i in cutlass.range_constexpr(BPT):
        sm = sm + loc[i]
    w = warp_incl_scan_add(sm, lane)
    if lane == cutlass.Int32(31):
        s_ws[wid] = w
    cute.arch.barrier()  # Synchronize shared scan results.
    vv = cutlass.Int32(0)
    if lane < cutlass.Int32(NW):
        vv = s_ws[lane]
    tot = cutlass.Int32(warp_add_i32(vv))
    sel = cutlass.Int32(0)
    if lane < wid:
        sel = vv
    off = cutlass.Int32(warp_add_i32(sel))
    after = tot - (off + w)
    if tidx == cutlass.Int32(0):
        s_res[RES_TOT] = tot
    for i in cutlass.range_constexpr(BPT - 1, -1, -1):
        cq = loc[i]
        gb = base + cutlass.Int32(i)
        if cutlass.const_expr(nb >= blk) or tidx < cutlass.Int32(nb):
            s_hx[gb] = after + pre[i]
            cross = cutlass.Int32(0)
            if after < target:
                if (after + cq) >= target:
                    cross = cutlass.Int32(1)
                if gb == cutlass.Int32(0):
                    cross = cutlass.Int32(1)
            if cross != cutlass.Int32(0):
                s_res[RES_B] = gb
                s_res[RES_ABOVE] = after
                s_res[RES_M] = cq
        after = after + cq


@dsl_user_op
def _lds_v4_u32(saddr, *, loc=None, ip=None):
    """ld.shared.v4.u32 at a plain Int32 byte address -> four Int32 words."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [saddr.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [0])),
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [1])),
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [2])),
        cutlass.Int32(llvm.extractvalue(T.i32(), ret, [3])),
    )


@dsl_user_op
def _sts_v4_u32(saddr, a, b, c, d, *, loc=None, ip=None):
    """st.shared.v4.u32 [saddr], {a, b, c, d}."""
    mlir_llvm.inline_asm(
        res=None,
        operands_=[
            saddr.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
        ],
        asm_string="st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        constraints="r,r,r,r,r",
        has_side_effects=True,
        asm_dialect=mlir_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_g_nc_v2_b32(gaddr, *, loc=None, ip=None):
    """Load four 16-bit elements as two Uint32 words from an 8-byte-aligned address."""
    ret = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [gaddr.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Uint32(llvm.extractvalue(T.i32(), ret, [0])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), ret, [1])),
    )


@dsl_user_op
def _ld_g_nc_u16(gaddr, *, loc=None, ip=None):
    """`ld.global.nc.u16` scalar 16-bit gather, zero-extended into a u32 reg."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [gaddr.ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u16 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def fkey16_of_code16(c):
    """Dense monotone 16-bit key of a 16-bit code (format-agnostic; equals fkey(f32) >> 16 for bf16)."""
    sign = c >> cutlass.Uint32(15)
    return (c ^ ((cutlass.Uint32(0) - sign) | cutlass.Uint32(0x8000))) & cutlass.Uint32(
        0xFFFF
    )


@dsl_user_op
def _red_shared_add1_if_eq(addr, x, y, *, loc=None, ip=None):
    """Increment shared memory when x == y, using predication without branching."""
    llvm.inline_asm(
        res=None,
        operands_=[
            addr.ir_value(loc=loc, ip=ip),
            x.ir_value(loc=loc, ip=ip),
            y.ir_value(loc=loc, ip=ip),
        ],
        asm_string="{\n\t.reg .pred p;\n\tsetp.eq.u32 p, $1, $2;\n\t"
        "@p red.relaxed.cta.shared.add.u32 [$0], 1;\n}",
        constraints="r,r,r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _atom_shared_add1_if_ge(addr, key, thr, *, loc=None, ip=None):
    """Ticket of `@(key >= thr) atom.shared.add.u32 [addr], 1` (the old value), or 0x40000000 when the
    predicate is false (no atomic issued).  Unsigned compare; addr Int32, key / thr Uint32."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [
                addr.ir_value(loc=loc, ip=ip),
                key.ir_value(loc=loc, ip=ip),
                thr.ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ge.u32 p, $2, $3;\n\tmov.u32 $0, 0x40000000;\n\t"
            "@p atom.relaxed.cta.shared.add.u32 $0, [$1], 1;\n}",
            "=r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def find_cross256_cursors(
    saddr,
    target,
    tidx,
    s_res,
    bias,
    tie_cnt,
    next_hist,
    pad_count,
    cancel_pad: cutlass.Constexpr,
):
    """Scan a 256-bin histogram with warp 0, without a barrier.

    Lane l owns bins 8l..8l+7. Bins become bias plus the count above them;
    the crossing lane publishes RES_B, RES_M, RES_ABOVE and tie_cnt[1].
    cancel_pad subtracts padding from next_hist[0] when the crossing bin is 0.
    """
    if tidx < cutlass.Int32(32):
        lane = tidx
        row_addr = saddr + lane * cutlass.Int32(32)
        h0, h1, h2, h3 = _lds_v4_u32(row_addr)
        h4, h5, h6, h7 = _lds_v4_u32(row_addr + cutlass.Int32(16))
        part = ((h0 + h1) + (h2 + h3)) + ((h4 + h5) + (h6 + h7))
        v = warp_suffix_scan_add(part, lane)  # sum of the partials of lanes >= lane
        msk = ballot(v >= target)
        L = hi_bit_or_zero(msk)
        acc = v - part  # keys in the lanes above this lane
        # cursors: count strictly above each bin of the lane, biased
        c7 = bias + acc
        c6 = c7 + h7
        c5 = c6 + h6
        c4 = c5 + h5
        c3 = c4 + h4
        c2 = c3 + h3
        c1 = c2 + h2
        c0 = c1 + h1
        _sts_v4_u32(row_addr, c0, c1, c2, c3)
        _sts_v4_u32(row_addr + cutlass.Int32(16), c4, c5, c6, c7)
        B = cutlass.Int32(0)
        M = cutlass.Int32(0)
        A = cutlass.Int32(0)
        found = cutlass.Int32(0)
        hs = (h0, h1, h2, h3, h4, h5, h6, h7)
        for jj in cutlass.range_constexpr(8):
            j = 7 - jj
            hj = hs[j]
            nxt = acc + hj
            if found == cutlass.Int32(0):
                if nxt >= target:
                    B = lane * cutlass.Int32(8) + cutlass.Int32(j)
                    M = hj
                    A = acc
                    found = cutlass.Int32(1)
                else:
                    acc = nxt
        if lane == L:
            s_res[RES_B] = B
            s_res[RES_M] = M
            s_res[RES_ABOVE] = bias + A
            tie_cnt[1] = bias + A
            if cutlass.const_expr(cancel_pad):
                if cutlass.Int32(0) == B:
                    next_hist[0] = cutlass.Int32(0) - pad_count


STATIC_WORDS = 128  # Shared-memory control region.


@dsl_user_op
def _smem_addr_reg__reg(addr, *, loc=None, ip=None):
    """Keep a CTA-shared byte address in one register.

    The non-duplicable asm result prevents repeated shared-window address
    materialization in divergent blocks.
    """
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [addr.ir_value(loc=loc, ip=ip)],
            "mov.u32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_shared_add1__reg(addr, *, loc=None, ip=None):
    """Increment a shared histogram bin without returning the previous count.

    Uses red.shared with relaxed CTA ordering and an Int32 byte address.
    """
    llvm.inline_asm(
        res=None,
        operands_=[addr.ir_value(loc=loc, ip=ip)],
        asm_string="red.relaxed.cta.shared.add.u32 [$0], 1;",
        constraints="r",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _smem_view(dtype, sbase, word_off: int, length: int, align: int = 16):
    """Typed tensor view at a constexpr word offset into the smem window."""
    p = cute.make_ptr(
        dtype,
        sbase + cutlass.Int32(word_off * 4),
        cute.AddressSpace.smem,
        assumed_align=align,
    )
    return cute.make_tensor(p, cute.make_layout((length,)))
