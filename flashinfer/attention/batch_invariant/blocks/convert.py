# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The conversion stage of the fp8-KV kernels (classes ``attention.e4m3kv`` / ``attention.e5m2kv``): the load warp and the two
otherwise idle warps of the last warpgroup (``operands.Roles.convert_warps``, 96 threads) read the fp8 K and V tiles of every unit
from global memory, widen each element to bf16 and write the ring slot bytes the TMA producer writes for a bf16 cache holding those
values. The MMA warp, the softmax and the correction consume the ring exactly as in the bf16 kernel: the conversion is a
placement of the class's exact ``widen`` leaves (``spec.kv_widen``), never an arithmetic step.

The NVFP4 variant (``st.fp4``) reads 8 packed bytes (16 e2m1 nibbles of one key) and the block's e4m3 scale byte per group and
forms ``value * scale`` in binary32 (``cvt.rn.f16x2.e2m1x2`` / ``cvt.rn.f16x2.e4m3x2``, ``cvt.f32.f16``, ``mul.rn.f32``): a
product of at most six significant bits within the bf16 range, exact at every step, then ``cvt.rn.bf16x2.f32`` moves bits. The
ring then holds the class's ``dq4`` leaves (``spec.kv_dq4``).

Exactness: every e4m3 and e5m2 value is a binary16 value, every binary16 value is a binary32 value, and every fp8 value has at most
four significant bits, so the hardware chain ``cvt.rn.f16x2.<fmt>x2`` -> ``cvt.f32.f16`` -> ``cvt.rn.bf16x2.f32`` moves bits and
rounds nothing (a NaN stays a NaN, an e5m2 infinity an infinity). The elements of a 16-byte global load (16 consecutive ``d`` of one
key) land in two 16-byte granules of the SW128-swizzled slot (``mma.zero_v_tail`` addresses the same swizzle), so the slot ends up
with element ``(key, d)`` at ``(d % 64) + 64 key + slot_keys * 64 * (d // 64)`` before the swizzle, as ``operands.ring_view_layout``
places it.

The slot order is the TMA producer's (``producer.load_loop``: ``K_lo``, then per tile ``K_{n+1}, V_n``, or with two S stages
``K_lo, K_{lo+1}`` then ``V_n, K_{n+2}``), so the MMA warp's consumer sequence is unchanged; the ring pipeline is a thread-to-UMMA
pipeline (``PipelineAsyncUmma``) whose full barrier takes the 96 arrivals after each thread's ``fence.proxy.async``. Keys past
the request's pages read a clamped valid page (the TMA producer's rule) and keys past a ragged pool its last token: the softmax
masks them and the MMA warp zeroes the V tail, as for the junk of a bf16 pool.
"""

from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .operands import HALF
from .producer import load_q_tile, q_tx_bytes

GROUP = 16  # elements per conversion group: one 16-byte global load of one key's consecutive d, two 16-byte ring granules


def _cvt_asm(cvt: str) -> str:
    """Four fp8 bytes of one 32-bit word to four bf16 in two 32-bit words (element i of the input at byte i, at half-word i of
    the output): ``cvt`` is the fp8-pair-to-f16x2 instruction of the format."""
    return f"""
    {{
        .reg .b16 lo16, hi16, a0, a1, a2, a3;
        .reg .b32 h2a, h2b;
        .reg .f32 f0, f1, f2, f3;
        mov.b32 {{lo16, hi16}}, $2;
        {cvt} h2a, lo16;
        {cvt} h2b, hi16;
        mov.b32 {{a0, a1}}, h2a;
        mov.b32 {{a2, a3}}, h2b;
        cvt.f32.f16 f0, a0;
        cvt.f32.f16 f1, a1;
        cvt.f32.f16 f2, a2;
        cvt.f32.f16 f3, a3;
        cvt.rn.bf16x2.f32 $0, f1, f0;
        cvt.rn.bf16x2.f32 $1, f3, f2;
    }}
    """


_E4M3_ASM = _cvt_asm("cvt.rn.f16x2.e4m3x2")
_E5M2_ASM = _cvt_asm("cvt.rn.f16x2.e5m2x2")

# four packed bytes = eight e2m1 nibbles (element 2i in the low nibble of byte i) times one binary32 scale -> eight bf16 in four
# words; every product is exact (see the module docstring)
_E2M1_ASM = """
    {
        .reg .b8 b0, b1, b2, b3;
        .reg .b32 h0, h1, h2, h3;
        .reg .b16 a0, a1, a2, a3, a4, a5, a6, a7;
        .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;
        mov.b32 {b0, b1, b2, b3}, $4;
        cvt.rn.f16x2.e2m1x2 h0, b0;
        cvt.rn.f16x2.e2m1x2 h1, b1;
        cvt.rn.f16x2.e2m1x2 h2, b2;
        cvt.rn.f16x2.e2m1x2 h3, b3;
        mov.b32 {a0, a1}, h0;
        mov.b32 {a2, a3}, h1;
        mov.b32 {a4, a5}, h2;
        mov.b32 {a6, a7}, h3;
        cvt.f32.f16 f0, a0;
        cvt.f32.f16 f1, a1;
        cvt.f32.f16 f2, a2;
        cvt.f32.f16 f3, a3;
        cvt.f32.f16 f4, a4;
        cvt.f32.f16 f5, a5;
        cvt.f32.f16 f6, a6;
        cvt.f32.f16 f7, a7;
        mul.rn.f32 f0, f0, $5;
        mul.rn.f32 f1, f1, $5;
        mul.rn.f32 f2, f2, $5;
        mul.rn.f32 f3, f3, $5;
        mul.rn.f32 f4, f4, $5;
        mul.rn.f32 f5, f5, $5;
        mul.rn.f32 f6, f6, $5;
        mul.rn.f32 f7, f7, $5;
        cvt.rn.bf16x2.f32 $0, f1, f0;
        cvt.rn.bf16x2.f32 $1, f3, f2;
        cvt.rn.bf16x2.f32 $2, f5, f4;
        cvt.rn.bf16x2.f32 $3, f7, f6;
    }
    """


@dsl_user_op
def e4m3_byte_to_f32(b: Uint32, *, loc=None, ip=None) -> Float32:
    """The e4m3 block scale (the low byte of ``b``) as binary32, exactly."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(b).ir_value(loc=loc, ip=ip)],
            """
    {
        .reg .b16 h, lo, hi;
        .reg .b32 p;
        cvt.u16.u32 h, $1;
        cvt.rn.f16x2.e4m3x2 p, h;
        mov.b32 {lo, hi}, p;
        cvt.f32.f16 $0, lo;
    }
    """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e2m1x8_scaled_to_bf16x8(
    packed: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Eight e2m1 nibbles times their block scale, as eight bf16 in four words, exactly."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [
            Uint32(packed).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip),
        ],
        _E2M1_ASM,
        "=r,=r,=r,=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    )


def _widen4(asm: str, packed: Uint32, loc, ip) -> Tuple[Uint32, Uint32]:
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        asm,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def e4m3x4_to_bf16x4(packed: Uint32, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """Four e4m3 bytes to four bf16 (two packed words), exactly."""
    return _widen4(_E4M3_ASM, packed, loc, ip)


@dsl_user_op
def e5m2x4_to_bf16x4(packed: Uint32, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """Four e5m2 bytes to four bf16 (two packed words), exactly."""
    return _widen4(_E5M2_ASM, packed, loc, ip)


@cute.jit
def widen_group(st, r_in: cute.Tensor, r_out: cute.Tensor):
    """``GROUP`` fp8 elements (four 32-bit words of ``r_in``) to ``GROUP`` bf16 (eight words of ``r_out``), in element order."""
    for j in cutlass.range_constexpr(GROUP // 4):
        if cutlass.const_expr(st.kv_fmt == "e4m3"):
            lo, hi = e4m3x4_to_bf16x4(r_in[j])
        else:
            lo, hi = e5m2x4_to_bf16x4(r_in[j])
        r_out[2 * j] = lo
        r_out[2 * j + 1] = hi


@cute.jit
def dequant_group(r_in: cute.Tensor, sf: Float32, r_out: cute.Tensor):
    """``GROUP`` e2m1 nibbles (two 32-bit words of ``r_in``) times the block scale ``sf`` to ``GROUP`` bf16 (eight words), in element
    order."""
    for j in cutlass.range_constexpr(GROUP // 8):
        w0, w1, w2, w3 = e2m1x8_scaled_to_bf16x8(r_in[j], sf)
        r_out[4 * j] = w0
        r_out[4 * j + 1] = w1
        r_out[4 * j + 2] = w2
        r_out[4 * j + 3] = w3


@cute.jit
def convert_slot(
    st,
    key0: Int32,
    kv_prod,
    mKV: cute.Tensor,
    mSF: cute.Tensor,
    sKV_u32: cute.Tensor,
    mKvIndices: cute.Tensor,
    kv_ptr0: Int32,
    n_pages: Int32,
    kv_head: Int32,
    ctid: Int32,
):
    """One ring slot (``slot_keys`` keys of K or V from absolute key ``key0``) of the low-bit cache ``mKV`` (``(D, page, Hkv,
    num_pages)`` bytes for fp8, ``(D / 2, ...)`` packed nibbles for NVFP4 with the block scales ``mSF`` ``(D / 16, ...)``; under
    ``st.ragged`` one page of ``total_kv`` tokens): acquire it, convert the slot's ``slot_keys x D`` elements in 16-element groups
    dealt round-robin over the stage's threads, fence and commit."""
    CT = st.roles.convert_threads
    GPK = st.D // GROUP  # groups per key
    n_groups = st.slot_keys * GPK
    EB = GROUP // 2 if st.fp4 else GROUP  # bytes of one group in the cache
    h = kv_prod.acquire_and_advance()
    r_in = cute.make_rmem_tensor(cute.make_layout((EB // 4,)), Uint32)
    r_in8 = cute.make_tensor(
        cute.recast_ptr(r_in.iterator, dtype=cutlass.Uint8), cute.make_layout((EB,))
    )
    r_out = cute.make_rmem_tensor(cute.make_layout((GROUP // 2,)), Uint32)
    for i in cutlass.range_constexpr((n_groups + CT - 1) // CT):
        gidx = ctid + i * CT
        if gidx < n_groups:
            key = gidx // GPK
            d0 = (gidx % GPK) * GROUP
            ka = key0 + key
            e0 = (
                d0 // 2 if cutlass.const_expr(st.fp4) else d0
            )  # the group's first byte in the cache row
            if cutlass.const_expr(st.ragged):
                tok = kv_ptr0 + ka
                last = mKV.shape[1] - 1
                if tok > last:
                    tok = last
                off = cute.crd2idx((e0, tok, kv_head, Int32(0)), mKV.layout)
                if cutlass.const_expr(st.fp4):
                    sfoff = cute.crd2idx(
                        (d0 // GROUP, tok, kv_head, Int32(0)), mSF.layout
                    )
            else:
                pslot = ka // st.page
                if pslot > n_pages - 1:
                    pslot = n_pages - 1
                page_id = mKvIndices[kv_ptr0 + pslot]
                off = cute.crd2idx((e0, ka % st.page, kv_head, page_id), mKV.layout)
                if cutlass.const_expr(st.fp4):
                    sfoff = cute.crd2idx(
                        (d0 // GROUP, ka % st.page, kv_head, page_id), mSF.layout
                    )
            cute.autovec_copy(
                cute.make_tensor(
                    mKV.iterator + cute.assume(off, divby=EB),
                    cute.make_layout((EB,)),
                ),
                r_in8,
            )
            if cutlass.const_expr(st.fp4):
                sf_byte = cute.make_tensor(
                    mSF.iterator + sfoff, cute.make_layout((1,))
                )[0]
                dequant_group(r_in, e4m3_byte_to_f32(sf_byte.to(Uint32)), r_out)
            else:
                widen_group(st, r_in, r_out)
            # the two 16-byte granules of the group in the SW128 slot (bf16 element offsets, then 32-bit words)
            base = (
                h.index * st.kv_slot_elems
                + (d0 // HALF) * (st.slot_keys * HALF)
                + key * HALF
            )
            gran = (d0 % HALF) // 8
            g0 = base + ((gran ^ (key % 8)) * 8)
            g1 = base + (((gran + 1) ^ (key % 8)) * 8)
            cute.autovec_copy(
                cute.make_tensor(r_out.iterator, cute.make_layout((4,))),
                cute.make_tensor(
                    sKV_u32.iterator + cute.assume(g0 // 2, divby=4),
                    cute.make_layout((4,)),
                ),
            )
            cute.autovec_copy(
                cute.make_tensor(r_out.iterator + 4, cute.make_layout((4,))),
                cute.make_tensor(
                    sKV_u32.iterator + cute.assume(g1 // 2, divby=4),
                    cute.make_layout((4,)),
                ),
            )
    cute.arch.fence_view_async_shared()
    h.commit()
    return kv_prod


@cute.jit
def convert_tile(
    st,
    t: Int32,
    kv_prod,
    mKV: cute.Tensor,
    mSF: cute.Tensor,
    sKV_u32: cute.Tensor,
    mKvIndices: cute.Tensor,
    kv_ptr0: Int32,
    n_pages: Int32,
    kv_head: Int32,
    ctid: Int32,
):
    """One K or V tile = ``slots_per_tile`` consecutive ring slots in key order (``producer.issue_tile``)."""
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        kv_prod = convert_slot(
            st,
            t * st.T + sl * st.slot_keys,
            kv_prod,
            mKV,
            mSF,
            sKV_u32,
            mKvIndices,
            kv_ptr0,
            n_pages,
            kv_head,
            ctid,
        )
    return kv_prod


@cute.jit
def convert_unit(
    st,
    kv_prod,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mKsf: cute.Tensor,
    mVsf: cute.Tensor,
    sKV_u32: cute.Tensor,
    mKvIndptr: cute.Tensor,
    mKvIndices: cute.Tensor,
    req: Int32,
    kv_head: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
    ctid: Int32,
):
    """The K and V tiles of one unit into the ring, in the TMA producer's slot order (``producer.load_loop``)."""
    LA = st.qk_ahead
    kv_ptr0 = mKvIndptr[req]
    n_pages = mKvIndptr[req + 1] - kv_ptr0
    N = tile_hi - tile_lo
    kv_prod = convert_tile(
        st,
        tile_lo,
        kv_prod,
        mK,
        mKsf,
        sKV_u32,
        mKvIndices,
        kv_ptr0,
        n_pages,
        kv_head,
        ctid,
    )
    if cutlass.const_expr(LA == 2):
        if N > 1:
            kv_prod = convert_tile(
                st,
                tile_lo + 1,
                kv_prod,
                mK,
                mKsf,
                sKV_u32,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
                ctid,
            )
        for n in range(N):
            kv_prod = convert_tile(
                st,
                tile_lo + n,
                kv_prod,
                mV,
                mVsf,
                sKV_u32,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
                ctid,
            )
            if n + 2 < N:
                kv_prod = convert_tile(
                    st,
                    tile_lo + n + 2,
                    kv_prod,
                    mK,
                    mKsf,
                    sKV_u32,
                    mKvIndices,
                    kv_ptr0,
                    n_pages,
                    kv_head,
                    ctid,
                )
    else:
        for n in range(N):
            if n + 1 < N:
                kv_prod = convert_tile(
                    st,
                    tile_lo + n + 1,
                    kv_prod,
                    mK,
                    mKsf,
                    sKV_u32,
                    mKvIndices,
                    kv_ptr0,
                    n_pages,
                    kv_head,
                    ctid,
                )
            kv_prod = convert_tile(
                st,
                tile_lo + n,
                kv_prod,
                mV,
                mVsf,
                sKV_u32,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
                ctid,
            )
    return kv_prod


@cute.jit
def load_q_unit(
    st,
    atom_q: cute.CopyAtom,
    tQ: cute.Tensor,
    atom_q_rep: cute.CopyAtom,
    tQr: cute.Tensor,
    sQ_view: cute.Tensor,
    sQ_rep_view: cute.Tensor,
    q_prod,
    tok_abs: Int32,
    kv_head: Int32,
    rows_valid: Int32,
):
    """The unit's Q tile by TMA (the load warp alone), as ``producer.load_loop`` issues it."""
    qh = q_prod.acquire_and_advance(expected_tx=q_tx_bytes(st, rows_valid))
    load_q_tile(
        st,
        atom_q,
        tQ,
        atom_q_rep,
        tQr,
        sQ_view,
        sQ_rep_view,
        qh,
        qh.index,
        tok_abs,
        kv_head,
        rows_valid,
    )
    return q_prod
