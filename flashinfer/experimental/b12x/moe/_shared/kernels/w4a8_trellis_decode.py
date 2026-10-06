"""Shared trellis decode primitives for the W4A8 kernel family.

Module-level pieces used by both the monolithic dynamic kernel and the
materialized phase kernels: the raw-window B stager, the t256 ring window
extraction and value-table decode, the butterfly pairing into m16n8k32
operand order, the within-K32 activation permutation that the native B lane
order requires, and the warp-level normalized H128 used at the activation
boundary.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    get_ptr_as_int64,
    ld_shared_u32,
    packed_decode_lut_e4m3_to_e4m3x8,
    packed_decode_lut_e4m3_direct_to_e4m3x8,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits


@cute.jit
def _w4a8_stage_trellis_b_tile(
    tr_u32: cute.Tensor,
    smem_base: Int32,
    tile_base: Int64,
    k16_stride_u32: Int32,
    bits: cutlass.Constexpr,
    tidx: Int32,
    tcnt: cutlass.Constexpr,
    k16_rows: cutlass.Constexpr = 8,
):
    """Stage one K128xN128 trellis B tile as eight contiguous K16 spans.

    The gmem trellis layout keeps the eight N16 window blocks of one K16 row
    contiguous (64*bits u32); rows are ``k16_stride_u32`` apart. The staged
    tile preserves k16-major order, so a consumer window base is
    ``(k16_local * 8 + n16_local) * 8 * bits`` u32 from ``smem_base``.
    """

    span_chunks = 16 * bits
    transfers = k16_rows * span_chunks
    limit = Int64(tr_u32.shape[0]) - Int64(4)
    for i in cutlass.range_constexpr((transfers + tcnt - 1) // tcnt):
        idx = tidx + Int32(i * tcnt)
        if idx < Int32(transfers):
            k16_local = idx // Int32(span_chunks)
            chunk = idx - k16_local * Int32(span_chunks)
            src = (
                tile_base
                + Int64(k16_local) * Int64(k16_stride_u32)
                + Int64(chunk * Int32(4))
            )
            # The pipeline prefetches epochs past the final task, where the
            # task metadata no longer addresses a real expert; predicate the
            # copy so junk prefetches cannot leave the payload allocation.
            if src >= Int64(0) and src <= limit:
                cp_async4_shared_global(
                    smem_base + (idx << Int32(4)),
                    get_ptr_as_int64(tr_u32, src),
                )


# The native trellis B decode leaves each lane's bytes in t256 order; the
# MMA is exact iff the A quantizer writes this fixed within-K32 permutation
# (tests/moe/test_w4a8_fragment_probe.py::_NATIVE_MMA_K32_PERM). The UE8M0
# scale group is unchanged because the permutation stays inside one K32.
_W4A8_TRELLIS_A_K32_PERM = (
    0, 1, 8, 9, 4, 5, 12, 13,
    2, 3, 10, 11, 6, 7, 14, 15,
    20, 21, 28, 29, 16, 17, 24, 25,
    22, 23, 30, 31, 18, 19, 26, 27,
)


@cute.jit
def _w4a8_trellis_permute_k32(values: cute.Tensor) -> cute.Tensor:
    permuted = cute.make_rmem_tensor((32,), cutlass.Float32)
    for i in cutlass.range_constexpr(32):
        permuted[i] = values[_W4A8_TRELLIS_A_K32_PERM[i]]
    return permuted


@cute.jit
def _w4a8_had128_quad(
    v0: cutlass.Float32,
    v1: cutlass.Float32,
    v2: cutlass.Float32,
    v3: cutlass.Float32,
    lane: Int32,
):
    """Normalized Sylvester H128 over one warp; lane holds columns lane*4..+3.

    Intra-lane H4 on the four consecutive values, then five butterfly rounds
    across the warp. Natural-order in and out; scale 1/sqrt(128).
    """

    s0 = v0 + v1
    d0 = v0 - v1
    s1 = v2 + v3
    d1 = v2 - v3
    h0 = s0 + s1
    h1 = d0 + d1
    h2 = s0 - s1
    h3 = d0 - d1
    for i in cutlass.range_constexpr(5):
        st = 1 << i
        p0 = cute.arch.shuffle_sync_bfly(h0, offset=st)
        p1 = cute.arch.shuffle_sync_bfly(h1, offset=st)
        p2 = cute.arch.shuffle_sync_bfly(h2, offset=st)
        p3 = cute.arch.shuffle_sync_bfly(h3, offset=st)
        if (lane & Int32(st)) != Int32(0):
            h0 = p0 - h0
            h1 = p1 - h1
            h2 = p2 - h2
            h3 = p3 - h3
        else:
            h0 = p0 + h0
            h1 = p1 + h1
            h2 = p2 + h2
            h3 = p3 + h3
    scale = cutlass.Float32(0.088388347648)
    return h0 * scale, h1 * scale, h2 * scale, h3 * scale


@cute.jit
def _w4a8_trellis_lane_geom(lane: Int32, bits: cutlass.Constexpr):
    """t256 ring geometry for this lane, hoistable across the K32 loops.

    ``ia``/``ib`` are the two ring word indices covering the lane's
    overlapping 16-bit windows and ``s2`` the merge shift; they depend only on
    the lane id and the bitrate. The W4A8 kernels always decode the full
    eight-weight span starting at the lane origin.
    """

    ia, ib, s2, _ = trellis256_lane_geom_bits(lane, 0, 8, bits)
    return ia, ib, s2


@cute.jit
def _w4a8_trellis_decode_half(
    smem_base: Int32,
    base_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    n_high: Int32,
    bits: cutlass.Constexpr,
    lut_addr: Int64,
    direct_lut: cutlass.Constexpr = False,
) -> Uint32:
    """Decode one K16 window's E4M3x4 word for one n8 half of an N16 tile."""

    a = Uint32(ld_shared_u32(smem_base + ((base_u32 + ia) << Int32(2))))
    b = Uint32(ld_shared_u32(smem_base + ((base_u32 + ib) << Int32(2))))
    merged = (Int64(a) << Int64(32)) | Int64(b)
    win_a = Uint32(merged >> Int64(s2))
    win_b = Uint32(merged >> Int64(s2 + Int32(4 * int(bits))))
    if cutlass.const_expr(direct_lut):
        lo, hi = packed_decode_lut_e4m3_direct_to_e4m3x8(
            win_a,
            win_b,
            lut_addr,
            int(bits),
            rate_indexed=True,
        )
    else:
        lo, hi = packed_decode_lut_e4m3_to_e4m3x8(
            win_a,
            win_b,
            lut_addr,
            int(bits),
            value_table_in_shared=True,
        )
    value = lo
    if n_high != Int32(0):
        value = hi
    return value


@cute.jit
def _w4a8_trellis_pair_words(
    smem_base: Int32,
    lane: Int32,
    base0_u32: Int32,
    base1_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    n_high: Int32,
    bits: cutlass.Constexpr,
    lut_addr: Int64,
    direct_lut: cutlass.Constexpr = False,
):
    """(b0, b1) QMMA E4M3 words for one n8 half and one K32 block.

    ``base0_u32``/``base1_u32`` address the two adjacent K16 window blocks of
    the K32 in staged shared memory. The butterfly redistributes the lanes'
    K quarters between the two decoded windows into MMA operand order; the
    contract is pinned by tests/moe/test_w4a8_fragment_probe.py.
    """

    e0 = _w4a8_trellis_decode_half(
        smem_base, base0_u32, ia, ib, s2, n_high, bits, lut_addr, direct_lut
    )
    e1 = _w4a8_trellis_decode_half(
        smem_base, base1_u32, ia, ib, s2, n_high, bits, lut_addr, direct_lut
    )
    c = lane & Int32(3)
    own = e0
    send = e1
    if c >= Int32(2):
        own = e1
        send = e0
    peer = Uint32(cute.arch.shuffle_sync_bfly(send, offset=2))
    return own, peer

@cute.jit
def _w4a8_trellis_decode_both(
    smem_base: Int32,
    base_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
    lut_addr: Int64,
    lut_in_smem: cutlass.Constexpr = True,
    direct_lut: cutlass.Constexpr = False,
):
    """Decode one K16 window's E4M3x4 words for both n8 halves."""

    a = Uint32(ld_shared_u32(smem_base + ((base_u32 + ia) << Int32(2))))
    b = Uint32(ld_shared_u32(smem_base + ((base_u32 + ib) << Int32(2))))
    merged = (Int64(a) << Int64(32)) | Int64(b)
    win_a = Uint32(merged >> Int64(s2))
    win_b = Uint32(merged >> Int64(s2 + Int32(4 * int(bits))))
    if cutlass.const_expr(direct_lut):
        lo, hi = packed_decode_lut_e4m3_direct_to_e4m3x8(
            win_a,
            win_b,
            lut_addr,
            int(bits),
            rate_indexed=True,
        )
    else:
        lo, hi = packed_decode_lut_e4m3_to_e4m3x8(
            win_a,
            win_b,
            lut_addr,
            int(bits),
            value_table_in_shared=bool(lut_in_smem),
        )
    return lo, hi


@cute.jit
def _w4a8_trellis_pair_words_both(
    smem_base: Int32,
    lane: Int32,
    base0_u32: Int32,
    base1_u32: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    bits: cutlass.Constexpr,
    lut_addr: Int64,
    lut_in_smem: cutlass.Constexpr = True,
    direct_lut: cutlass.Constexpr = False,
):
    """(b0, b1) QMMA words for both n8 halves of one N16 tile and one K32.

    Same butterfly contract as ``_w4a8_trellis_pair_words`` with both halves
    retained; consumers whose warp owns adjacent n8 fragments avoid the
    discarded-half decode entirely.
    """

    e0_lo, e0_hi = _w4a8_trellis_decode_both(
        smem_base, base0_u32, ia, ib, s2, bits, lut_addr, lut_in_smem,
        direct_lut,
    )
    e1_lo, e1_hi = _w4a8_trellis_decode_both(
        smem_base, base1_u32, ia, ib, s2, bits, lut_addr, lut_in_smem,
        direct_lut,
    )
    c = lane & Int32(3)
    own_lo = e0_lo
    own_hi = e0_hi
    send_lo = e1_lo
    send_hi = e1_hi
    if c >= Int32(2):
        own_lo = e1_lo
        own_hi = e1_hi
        send_lo = e0_lo
        send_hi = e0_hi
    peer_lo = Uint32(cute.arch.shuffle_sync_bfly(send_lo, offset=2))
    peer_hi = Uint32(cute.arch.shuffle_sync_bfly(send_hi, offset=2))
    return own_lo, peer_lo, own_hi, peer_hi
