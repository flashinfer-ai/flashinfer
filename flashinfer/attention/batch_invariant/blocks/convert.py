# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The conversion stage of the fp8-KV kernels (classes ``attention.e4m3kv`` / ``attention.e5m2kv``): the load warp's TMA brings the
raw fp8 K and V tiles of every unit into a staging buffer (``stage_unit``: the page boxes of the bf16 producer, one byte per
element), and the conversion warps (``operands.Roles.convert_warps``) widen each element to bf16 and write the ring slot bytes the
TMA producer writes for a bf16 cache holding those values (``convert_unit``). The MMA warp, the softmax and the correction consume
the ring exactly as in the bf16 kernel: the conversion is a placement of the class's exact ``widen`` leaves (``spec.kv_widen``),
never an arithmetic step.

The NVFP4 variant (``st.fp4``) stages 8 packed bytes (16 e2m1 nibbles of one key) per group, reads the block's e4m3 scale byte
from global memory and forms ``value * scale`` in binary32 (``cvt.rn.f16x2.e2m1x2`` / ``cvt.rn.f16x2.e4m3x2``, ``cvt.f32.f16``,
``mul.rn.f32``): a product of at most six significant bits within the bf16 range, exact at every step, then ``cvt.rn.bf16x2.f32``
moves bits. The ring then holds the class's ``dq4`` leaves (``spec.kv_dq4``). The V block scales of a paged cache are read in
the linear layout or in the 4-token swizzle (``sf_swizzle_coord``, ``st.v_sf_swizzled``): the two layouts store the same scale
bytes at different addresses, so the leaves -- and with them the class and its fingerprint -- do not depend on the layout.

Exactness: every e4m3 and e5m2 value is a binary16 value, every binary16 value is a binary32 value, and every fp8 value has at most
four significant bits, so the hardware chain ``cvt.rn.f16x2.<fmt>x2`` -> ``cvt.f32.f16`` -> ``cvt.rn.bf16x2.f32`` moves bits and
rounds nothing (a NaN stays a NaN, an e5m2 infinity an infinity). The 16 elements of a group (16 consecutive ``d`` of one key)
land in two 16-byte granules of the SW128-swizzled slot (``mma.zero_v_tail`` addresses the same swizzle), so the slot ends up
with element ``(key, d)`` at ``(d % 64) + 64 key + slot_keys * 64 * (d // 64)`` before the swizzle, as ``operands.ring_view_layout``
places it.

A staging stage is ``raw_keys`` keys of raw cache rows (``operands.StaticPlan``); a ring slot is ``raw_per_slot``
consecutive stages. The stage order is the TMA producer's slot order (``producer.load_loop``: ``K_lo``, then per tile
``K_{n+1}, V_n``, or with two S stages ``K_lo, K_{lo+1}`` then ``V_n, K_{n+2}``), so the MMA warp's consumer sequence is
unchanged; the ring pipeline is a thread-to-UMMA pipeline (``PipelineAsyncUmma``) whose full barrier takes one arrival per
conversion thread after its ``fence.proxy.async``. Keys past the request's pages stage a clamped valid page (the TMA producer's
rule) and keys past a ragged pool are zero-filled by the TMA: the softmax masks them and the MMA warp zeroes the V tail, as for
the junk of a bf16 pool.
"""

from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

from .operands import HALF
from .producer import load_q_tile, q_tx_bytes

GROUP = 16  # elements per conversion group: 16 consecutive d of one key (16 staged bytes, 8 for NVFP4), two 16-byte ring granules


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


def sf_swizzle_coord(n_blk: int, blk, tok):
    """Where the 4-token swizzle of a paged NVFP4 cache stores the block scale of ``(blk, tok)`` -- block ``blk`` (of ``n_blk`` =
    ``D / 16`` per key) of the token at offset ``tok`` in its page: within each group of four consecutive tokens of a page and
    head, the ``4 x n_blk`` scale bytes are stored at token ``(tok // 4) * 4 + blk // (n_blk / 4)``, block
    ``(blk % (n_blk / 4)) * 4 + tok % 4`` (the layout trtllm-gen reads the V block scales in, which serving engines write for
    it). A permutation of the scale bytes inside the four-token group, so the page size must be a multiple of four tokens; the
    scales of a page tail (a last page filled to fewer than a multiple of four tokens) lie in the rows of its group like any
    other. Plain Python on Python or DSL integers: the addressing has one statement for the kernel and the tests."""
    assert n_blk % 4 == 0, n_blk
    per = n_blk // 4
    return (blk % per) * 4 + tok % 4, (tok // 4) * 4 + blk // per


SF_AHEAD = 4  # rounds the NVFP4 block scale of a group is read from global memory ahead of its conversion


@cute.jit
def sf_offset(
    st,
    mSF: cute.Tensor,
    blk: Int32,
    tok: Int32,
    kv_head: Int32,
    page_id: Int32,
    sf_swizzled: cutlass.Constexpr,
):
    """The offset in the NVFP4 block scales ``mSF`` (``(D / 16, page, Hkv, num_pages)`` bytes) of block ``blk`` of token ``tok``
    of page ``page_id``: the linear layout (the cache's own geometry), or under ``sf_swizzled`` the 4-token swizzle
    (``sf_swizzle_coord``) -- another address of the same scale byte."""
    if cutlass.const_expr(sf_swizzled):
        return cute.crd2idx(
            sf_swizzle_coord(st.D // GROUP, blk, tok) + (kv_head, page_id), mSF.layout
        )
    return cute.crd2idx((blk, tok, kv_head, page_id), mSF.layout)


@cute.jit
def load_sf(
    st,
    key: Int32,
    ka0: Int32,
    blk: Int32,
    pg0: Int32,
    mSF: cute.Tensor,
    sPg: cute.Tensor,
    kv_ptr0: Int32,
    kv_head: Int32,
    r_sf: cute.Tensor,
    sf_swizzled: cutlass.Constexpr,
):
    """The e4m3 block scale byte of block ``blk`` of stage key ``key`` (clamped to the stage; absolute key ``ka0 + key``) into
    ``r_sf[0]``. The key's page is the one the load warp staged its cache bytes from (``sPg[pg0 + box]``); a ragged pool clamps
    the token to the pool."""
    if key > st.raw_keys - 1:
        key = Int32(st.raw_keys - 1)
    ka = ka0 + key
    if cutlass.const_expr(st.ragged):
        tok = kv_ptr0 + ka
        last = mSF.shape[1] - 1
        if tok > last:
            tok = last
        sfoff = sf_offset(st, mSF, blk, tok, kv_head, Int32(0), False)
    else:
        page_id = sPg[pg0 + key // st.raw_box_rows]
        sfoff = sf_offset(st, mSF, blk, ka % st.page, kv_head, page_id, sf_swizzled)
    r_sf[0] = cute.make_tensor(mSF.iterator + sfoff, cute.make_layout((1,)))[0].to(
        Uint32
    )


@cute.jit
def stage_slot(
    st,
    key0: Int32,
    raw_prod,
    atom: cute.CopyAtom,
    tg: cute.Tensor,
    ts: cute.Tensor,
    sPg: cute.Tensor,
    mKvIndices: cute.Tensor,
    kv_ptr0: Int32,
    n_pages: Int32,
    kv_head: Int32,
):
    """The raw cache bytes of one ring slot (``slot_keys`` keys of K or V from absolute key ``key0``): ``raw_per_slot`` staging
    stages, each acquired and filled by ``raw_boxes`` TMA boxes of ``raw_box_rows`` keys of one page (``producer.issue_slot``'s
    page and ragged addressing). The NVFP4 kernels record every box's page in the stage's side entries ``sPg`` before the box
    is issued (the conversion threads address the block scales with it)."""
    page = st.page
    for sub in cutlass.range_constexpr(st.raw_per_slot):
        h = raw_prod.acquire_and_advance()
        for i in cutlass.range_constexpr(st.raw_boxes):
            k0 = key0 + sub * st.raw_keys + i * st.raw_box_rows
            if cutlass.const_expr(st.ragged):
                for half in cutlass.range_constexpr(st.raw_halves):
                    cute.copy(
                        atom,
                        tg[(None, (half, k0 // st.raw_box_rows, kv_head, 0))],
                        ts[(None, i, half, h.index)],
                        tma_bar_ptr=h.barrier,
                    )
            else:
                pslot = k0 // page
                if cutlass.const_expr(page <= st.raw_keys):
                    box_in_page = Int32(0)
                else:
                    box_in_page = (k0 % page) // st.raw_box_rows
                if pslot > n_pages - 1:
                    pslot = n_pages - 1
                page_id = mKvIndices[kv_ptr0 + pslot]
                if cutlass.const_expr(st.fp4):
                    sPg[h.index * st.raw_boxes + i] = page_id
                for half in cutlass.range_constexpr(st.raw_halves):
                    cute.copy(
                        atom,
                        tg[(None, (half, box_in_page, kv_head, page_id))],
                        ts[(None, i, half, h.index)],
                        tma_bar_ptr=h.barrier,
                    )
    return raw_prod


@cute.jit
def stage_tile(
    st,
    t: Int32,
    raw_prod,
    atom: cute.CopyAtom,
    tg: cute.Tensor,
    ts: cute.Tensor,
    sPg: cute.Tensor,
    mKvIndices: cute.Tensor,
    kv_ptr0: Int32,
    n_pages: Int32,
    kv_head: Int32,
):
    """One K or V tile = ``slots_per_tile`` consecutive ring slots in key order (``producer.issue_tile``)."""
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        raw_prod = stage_slot(
            st,
            t * st.T + sl * st.slot_keys,
            raw_prod,
            atom,
            tg,
            ts,
            sPg,
            mKvIndices,
            kv_ptr0,
            n_pages,
            kv_head,
        )
    return raw_prod


@cute.jit
def stage_unit(
    st,
    raw_prod,
    atom_k: cute.CopyAtom,
    tK: cute.Tensor,
    atom_v: cute.CopyAtom,
    tV: cute.Tensor,
    sRaw_view: cute.Tensor,
    sPg: cute.Tensor,
    mKvIndptr: cute.Tensor,
    mKvIndices: cute.Tensor,
    req: Int32,
    kv_head: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
):
    """The raw K and V tiles of one unit into the staging buffer, in the TMA producer's slot order (``producer.load_loop``)."""
    LA = st.qk_ahead
    kv_ptr0 = mKvIndptr[req]
    n_pages = mKvIndptr[req + 1] - kv_ptr0
    if cutlass.const_expr(st.ragged):
        tK = cute.domain_offset((0, kv_ptr0, 0, 0), tK)
        tV = cute.domain_offset((0, kv_ptr0, 0, 0), tV)
    box = (st.raw_bw, st.raw_box_rows, 1, 1)
    gK = cute.zipped_divide(tK, box)
    gV = cute.zipped_divide(tV, box)
    sRg = cute.group_modes(
        sRaw_view, 0, 4
    )  # ((raw_bw, rows, 1, 1), boxes, halves, stages)
    tKsK, tKgK = cpasync.tma_partition(atom_k, 0, cute.make_layout(1), sRg, gK)
    tVsV, tVgV = cpasync.tma_partition(atom_v, 0, cute.make_layout(1), sRg, gV)
    N = tile_hi - tile_lo
    raw_prod = stage_tile(
        st,
        tile_lo,
        raw_prod,
        atom_k,
        tKgK,
        tKsK,
        sPg,
        mKvIndices,
        kv_ptr0,
        n_pages,
        kv_head,
    )
    if cutlass.const_expr(LA == 2):
        if N > 1:
            raw_prod = stage_tile(
                st,
                tile_lo + 1,
                raw_prod,
                atom_k,
                tKgK,
                tKsK,
                sPg,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
            )
        for n in range(N):
            raw_prod = stage_tile(
                st,
                tile_lo + n,
                raw_prod,
                atom_v,
                tVgV,
                tVsV,
                sPg,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
            )
            if n + 2 < N:
                raw_prod = stage_tile(
                    st,
                    tile_lo + n + 2,
                    raw_prod,
                    atom_k,
                    tKgK,
                    tKsK,
                    sPg,
                    mKvIndices,
                    kv_ptr0,
                    n_pages,
                    kv_head,
                )
    else:
        for n in range(N):
            if n + 1 < N:
                raw_prod = stage_tile(
                    st,
                    tile_lo + n + 1,
                    raw_prod,
                    atom_k,
                    tKgK,
                    tKsK,
                    sPg,
                    mKvIndices,
                    kv_ptr0,
                    n_pages,
                    kv_head,
                )
            raw_prod = stage_tile(
                st,
                tile_lo + n,
                raw_prod,
                atom_v,
                tVgV,
                tVsV,
                sPg,
                mKvIndices,
                kv_ptr0,
                n_pages,
                kv_head,
            )
    return raw_prod


@cute.jit
def load_group(
    st,
    key: Int32,
    rbase: Int32,
    sRaw: cute.Tensor,
    r_g8: cute.Tensor,
    G: cutlass.Constexpr,
):
    """The staged bytes of ``G`` consecutive groups (stage key ``key``, clamped to the stage; ``rbase`` the byte offset of the
    thread's ``d0`` of the stage's key 0) into ``r_g8``."""
    EB = G * (GROUP // 2 if st.fp4 else GROUP)
    if key > st.raw_keys - 1:
        key = Int32(st.raw_keys - 1)
    if cutlass.const_expr(st.raw_box_stride == st.raw_bw * st.raw_box_rows):
        koff = key * st.raw_bw
    else:
        koff = (key // st.raw_box_rows) * st.raw_box_stride + (
            key % st.raw_box_rows
        ) * st.raw_bw
    cute.autovec_copy(
        cute.make_tensor(
            sRaw.iterator + cute.assume(rbase + koff, divby=EB),
            cute.make_layout((EB,)),
        ),
        r_g8,
    )


@cute.jit
def convert_slot(
    st,
    key0: Int32,
    kv_prod,
    raw_cons,
    mSF: cute.Tensor,
    sKV_u32: cute.Tensor,
    sRaw: cute.Tensor,
    sPg: cute.Tensor,
    kv_ptr0: Int32,
    kv_head: Int32,
    ctid: Int32,
    sf_swizzled: cutlass.Constexpr,
):
    """One ring slot (``slot_keys`` keys of K or V from absolute key ``key0``) from its staged raw bytes (``sRaw``: fp8 bytes, or
    packed NVFP4 nibbles with the block scales ``mSF`` ``(D / 16, page, Hkv, num_pages)`` read from global memory through the
    stage's page entries ``sPg``): acquire it, convert stage after stage in 16-element groups and commit. Thread ``ctid`` of the
    stage holds ``convert_groups`` consecutive groups (unit ``ctid % UPK`` of ``UPK`` per key) of the keys ``ctid // UPK + round *
    KPI`` of a stage; the loop is compact and reads a round ahead of its conversion.
    ``sf_swizzled``: the block scales of ``mSF`` are stored in the 4-token swizzle (``sf_offset``)."""
    CT = st.roles.convert_threads
    G = st.convert_groups  # consecutive groups of one key a thread converts per round
    UPK = st.D // (GROUP * G)  # thread units per key
    KPI = CT // UPK  # keys the stage's threads cover per round
    EB = GROUP // 2 if st.fp4 else GROUP  # bytes of one group in the cache
    EW = EB // 4
    h = kv_prod.acquire_and_advance()
    r_in = cute.make_rmem_tensor(cute.make_layout((2 * G * EW,)), Uint32)
    r_in8 = cute.make_tensor(
        cute.recast_ptr(r_in.iterator, dtype=cutlass.Uint8),
        cute.make_layout((2 * G * EB,)),
    )
    r_sf = cute.make_rmem_tensor(cute.make_layout(((SF_AHEAD + 1) * G,)), Uint32)
    r_out = cute.make_rmem_tensor(cute.make_layout((GROUP // 2,)), Uint32)
    kt = ctid // UPK
    d0 = (ctid % UPK) * (GROUP * G)
    e0 = (
        d0 // 2 if cutlass.const_expr(st.fp4) else d0
    )  # the unit's first byte in the cache row
    for sub in cutlass.range_constexpr(st.raw_per_slot):
        rh = raw_cons.wait_and_advance()
        rbase = (
            rh.index * st.raw_stage_bytes
            + (e0 // st.raw_bw) * st.raw_half_bytes
            + e0 % st.raw_bw
        )
        skey = sub * st.raw_keys
        pg0 = rh.index * st.raw_boxes
        # the thread's rounds of the stage: its keys kt, kt + KPI, ... below raw_keys (none for a thread past a small stage)
        n_rounds = (st.raw_keys - kt + KPI - 1) // KPI
        if cutlass.const_expr(st.raw_keys % KPI == 0):
            n_rounds = Int32(st.raw_keys // KPI)
        load_group(st, kt, rbase, sRaw, r_in8, G)
        if cutlass.const_expr(st.fp4):
            for a in cutlass.range_constexpr(SF_AHEAD):
                for g in cutlass.range_constexpr(G):
                    load_sf(
                        st,
                        kt + a * KPI,
                        key0 + skey,
                        d0 // GROUP + g,
                        pg0,
                        mSF,
                        sPg,
                        kv_ptr0,
                        kv_head,
                        cute.make_tensor(
                            r_sf.iterator + a * G + g, cute.make_layout((1,))
                        ),
                        sf_swizzled,
                    )
        for i in cutlass.range(n_rounds, unroll=1):
            key = kt + i * KPI
            load_group(
                st,
                key + KPI,
                rbase,
                sRaw,
                cute.make_tensor(r_in8.iterator + G * EB, cute.make_layout((G * EB,))),
                G,
            )
            if cutlass.const_expr(st.fp4):
                for g in cutlass.range_constexpr(G):
                    load_sf(
                        st,
                        key + SF_AHEAD * KPI,
                        key0 + skey,
                        d0 // GROUP + g,
                        pg0,
                        mSF,
                        sPg,
                        kv_ptr0,
                        kv_head,
                        cute.make_tensor(
                            r_sf.iterator + SF_AHEAD * G + g, cute.make_layout((1,))
                        ),
                        sf_swizzled,
                    )
            skk = skey + key
            for g in cutlass.range_constexpr(G):
                r_g = cute.make_tensor(r_in.iterator + g * EW, cute.make_layout((EW,)))
                if cutlass.const_expr(st.fp4):
                    dequant_group(r_g, e4m3_byte_to_f32(r_sf[g]), r_out)
                else:
                    widen_group(st, r_g, r_out)
                # the two 16-byte granules of the group in the SW128 slot (bf16 element offsets, then 32-bit words)
                dg = d0 + g * GROUP
                base = (
                    h.index * st.kv_slot_elems
                    + (dg // HALF) * (st.slot_keys * HALF)
                    + skk * HALF
                )
                gran = (dg % HALF) // 8
                g0 = base + ((gran ^ (skk % 8)) * 8)
                g1 = base + (((gran + 1) ^ (skk % 8)) * 8)
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
            for j in cutlass.range_constexpr(G * EW):
                r_in[j] = r_in[G * EW + j]
            if cutlass.const_expr(st.fp4):
                for a in cutlass.range_constexpr(SF_AHEAD * G):
                    r_sf[a] = r_sf[a + G]
        rh.release()
    cute.arch.fence_view_async_shared()
    h.commit()
    return kv_prod, raw_cons


@cute.jit
def convert_tile(
    st,
    t: Int32,
    kv_prod,
    raw_cons,
    mSF: cute.Tensor,
    sKV_u32: cute.Tensor,
    sRaw: cute.Tensor,
    sPg: cute.Tensor,
    kv_ptr0: Int32,
    kv_head: Int32,
    ctid: Int32,
    sf_swizzled: cutlass.Constexpr,
):
    """One K or V tile = ``slots_per_tile`` consecutive ring slots in key order (``producer.issue_tile``)."""
    for sl in cutlass.range_constexpr(st.slots_per_tile):
        kv_prod, raw_cons = convert_slot(
            st,
            t * st.T + sl * st.slot_keys,
            kv_prod,
            raw_cons,
            mSF,
            sKV_u32,
            sRaw,
            sPg,
            kv_ptr0,
            kv_head,
            ctid,
            sf_swizzled,
        )
    return kv_prod, raw_cons


@cute.jit
def convert_unit(
    st,
    kv_prod,
    raw_cons,
    mKsf: cute.Tensor,
    mVsf: cute.Tensor,
    sKV_u32: cute.Tensor,
    sRaw: cute.Tensor,
    sPg: cute.Tensor,
    mKvIndptr: cute.Tensor,
    req: Int32,
    kv_head: Int32,
    tile_lo: Int32,
    tile_hi: Int32,
    ctid: Int32,
):
    """The staged K and V tiles of one unit into the ring, in the TMA producer's slot order (``producer.load_loop``): the order
    ``stage_unit`` staged them in. The K block scales of an NVFP4 cache are linear; the V block scales are linear or swizzled as
    the plan states (``st.v_sf_swizzled``)."""
    LA = st.qk_ahead
    kv_ptr0 = mKvIndptr[req]
    N = tile_hi - tile_lo
    kv_prod, raw_cons = convert_tile(
        st,
        tile_lo,
        kv_prod,
        raw_cons,
        mKsf,
        sKV_u32,
        sRaw,
        sPg,
        kv_ptr0,
        kv_head,
        ctid,
        False,
    )
    if cutlass.const_expr(LA == 2):
        if N > 1:
            kv_prod, raw_cons = convert_tile(
                st,
                tile_lo + 1,
                kv_prod,
                raw_cons,
                mKsf,
                sKV_u32,
                sRaw,
                sPg,
                kv_ptr0,
                kv_head,
                ctid,
                False,
            )
        for n in range(N):
            kv_prod, raw_cons = convert_tile(
                st,
                tile_lo + n,
                kv_prod,
                raw_cons,
                mVsf,
                sKV_u32,
                sRaw,
                sPg,
                kv_ptr0,
                kv_head,
                ctid,
                st.v_sf_swizzled,
            )
            if n + 2 < N:
                kv_prod, raw_cons = convert_tile(
                    st,
                    tile_lo + n + 2,
                    kv_prod,
                    raw_cons,
                    mKsf,
                    sKV_u32,
                    sRaw,
                    sPg,
                    kv_ptr0,
                    kv_head,
                    ctid,
                    False,
                )
    else:
        for n in range(N):
            if n + 1 < N:
                kv_prod, raw_cons = convert_tile(
                    st,
                    tile_lo + n + 1,
                    kv_prod,
                    raw_cons,
                    mKsf,
                    sKV_u32,
                    sRaw,
                    sPg,
                    kv_ptr0,
                    kv_head,
                    ctid,
                    False,
                )
            kv_prod, raw_cons = convert_tile(
                st,
                tile_lo + n,
                kv_prod,
                raw_cons,
                mVsf,
                sKV_u32,
                sRaw,
                sPg,
                kv_ptr0,
                kv_head,
                ctid,
                st.v_sf_swizzled,
            )
    return kv_prod, raw_cons


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
