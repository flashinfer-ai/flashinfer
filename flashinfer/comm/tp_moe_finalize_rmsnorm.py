# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""TP MoE finalize and post-expert RMSNorm on SM100/SM103.

One candidate-owned kernel, no framework collective.  The input is the raw
permuted GEMM2 buffer plus its expanded-to-permuted row map; resolving that map
inside the kernel avoids materializing the logical [T,K,H] partial-row tensor.
The TP group size P must exceed one and evenly divide K_EXPERTS. Each rank
owns KO = K_EXPERTS / P expert slots:

  phase 1 (scatter by expert owner): rank r owns slots [r*KO, (r+1)*KO).
      Each rank pushes its partial rows for other owners straight from
      registers into those owners' Lamport slots, using unicast stores.
  phase 2 (local finalize of the owned experts): the owner sums the BF16
      partial rows from all ranks in rank order in FP32, rounds once to BF16,
      RMS-normalizes over the full H, applies the routing weight in FP32,
      and accumulates its owned slots into an FP32 partial output row.
  phase 3 (all-reduce the partial output): one multimem.st broadcast of the
      FP32 [T,H] partial row, then each rank sums the partials in rank order
      and rounds once to BF16.

Sharding expert slots distributes normalization work across ranks. The second
communication hop exchanges only the smaller [T,H] partial outputs.

Both hops use the same flag-free Lamport transport: destination slots are
pre-armed with a negative-zero sentinel (0x80008000 for the BF16 hop,
0x80000000 for the FP32 hop), so a payload word that still reads as the sentinel
has not landed yet and the data write *is* the release.  Live payload words are
scrubbed from -0.0 to +0.0 (numerically identical) so they can never be mistaken
for "not arrived".

Buffer generations: both regions are triple-buffered.  A launch reads
generation ``g``, writes ``g`` and re-arms ``g+2`` (== ``g-1``); a peer cannot
reach ``g+2`` before this rank has published ``g+1``, so re-arming never races a
live push.  The counter is one int32 per CTA slot in device memory, read at
entry and advanced at exit, so a captured graph replays correctly.

Decomposition: a token is owned by a thread-block *cluster* of ``S`` CTAs that
split the H axis; every CTA pushes, polls, normalizes and writes only its own H
band.  The single piece of cross-CTA state is the per-expert sums of
squares, exchanged through distributed shared memory with one cluster barrier.
``S`` is a launch parameter: small token counts leave most of the machine idle
with one CTA per token, so S grows until the grid covers the SM array.
"""

import logging
from typing import Any, TypedDict

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
import torch
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32, Pointer
from cutlass.cutlass_dsl import T as MLIR_T
from cutlass.cutlass_dsl import dsl_user_op

from ..api_logging import flashinfer_api
from ..trace.templates.comm import tp_moe_finalize_rmsnorm_trace
from ..utils import register_custom_op, register_fake_op

try:
    import cuda.bindings.driver as _drv
except ImportError:  # pragma: no cover
    import cuda.cuda as _drv

_STREAMS: dict[int, Any] = {}
logger = logging.getLogger(__name__)

_HAS_CLUSTER = hasattr(cute.arch, "cluster_arrive") and hasattr(
    cute.arch, "cluster_wait"
)


def _cur_stream(dev_ord, h=None):
    if h is None:
        h = torch._C._cuda_getCurrentRawStream(dev_ord)
    s = _STREAMS.get(h)
    if s is None:
        s = _drv.CUstream(h)
        _STREAMS[h] = s
    return s


K_EXPERTS = 8
_COMPUTE_CAPABILITIES = ((10, 0), (10, 3))
H_DIM = 4096
VEC = 8
NSLOT = 256  # >= max CTAs in the grid; one Lamport slot per CTA
NGEN = 3
# packed pair of BF16 negative zeros / one FP32 negative zero
SENT_A = 0x80008000
SENT_B = 0x80000000
SENT_A_I32 = SENT_A - (1 << 32)
SENT_B_I32 = SENT_B - (1 << 32)


# --------------------------------------------------------------------------- #
# low level PTX helpers
# --------------------------------------------------------------------------- #
@dsl_user_op
def ld_global_v4(
    ptr: Pointer, *, loc=None, ip=None
) -> tuple[Int32, Int32, Int32, Int32]:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    rs = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [ptr_int],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    regs = [llvm.extractvalue(MLIR_T.i32(), rs, [i]) for i in range(4)]
    return regs[0], regs[1], regs[2], regs[3]


@dsl_user_op
def ld_global_v4_volatile(
    ptr: Pointer, *, loc=None, ip=None
) -> tuple[Int32, Int32, Int32, Int32]:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    rs = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [ptr_int],
        "ld.volatile.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    regs = [llvm.extractvalue(MLIR_T.i32(), rs, [i]) for i in range(4)]
    return regs[0], regs[1], regs[2], regs[3]


@dsl_user_op
def st_global_v4(
    ptr: Pointer, x: Int32, y: Int32, z: Int32, w: Int32, *, loc=None, ip=None
) -> None:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    vx = [Int32(v).ir_value(loc=loc, ip=ip) for v in (x, y, z, w)]
    llvm.inline_asm(
        None,
        [ptr_int] + vx,
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def multimem_st_v4(
    ptr: Pointer, x: Int32, y: Int32, z: Int32, w: Int32, *, loc=None, ip=None
) -> None:
    """One store that the NVSwitch fans out to every rank's copy."""
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    vx = [Int32(v).ir_value(loc=loc, ip=ip) for v in (x, y, z, w)]
    llvm.inline_asm(
        None,
        [ptr_int] + vx,
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_proxy_alias(*, loc=None, ip=None) -> None:
    """Order multicast stores before polling through their unicast aliases."""
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.alias;",
        "",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def _scrub(sent):
    @dsl_user_op
    def f(v: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(
            llvm.inline_asm(
                MLIR_T.i32(),
                [Int32(v).ir_value(loc=loc, ip=ip)],
                "{ .reg .pred scp;\n"
                "  setp.eq.u32 scp, $1, " + str(sent) + ";\n"
                "  selp.b32 $0, 0, $1, scp; }",
                "=r,r",
                has_side_effects=False,
                asm_dialect=0,
                loc=loc,
                ip=ip,
            )
        )

    return f


# Map the sentinel bit pattern to +0.0.  The two are numerically identical, so
# this never changes the result; it only guarantees a live payload word can
# never be mistaken for "not yet arrived".
scrub_a = _scrub(SENT_A)
scrub_b = _scrub(SENT_B)


def _probe_asm(offsets, sent) -> str:
    """Branch-free readiness probe over the given 128-bit slot byte offsets.

    Returns 1 when no 32-bit word still holds the sentinel.  Deliberately has
    no PTX label so the block stays safe to duplicate (loop unrolling, inlining)
    — the spin itself is expressed as a DSL ``while`` around this probe.
    """
    n_vec = len(offsets)
    nw = n_vec * 4
    lines = ["{", " .reg .b32 pw<" + str(nw) + ">;", " .reg .pred pqr;"]
    for k in range(n_vec):
        off = offsets[k]
        addr = "$1" if off == 0 else "$1+" + str(off)
        lines.append(
            f" ld.volatile.global.v4.u32 {{pw{4 * k},pw{4 * k + 1},"
            f"pw{4 * k + 2},pw{4 * k + 3}}}, [{addr}];"
        )
    lines.append(" setp.ne.u32 pqr, pw0, " + str(sent) + ";")
    for i in range(1, nw):
        lines.append(f" setp.ne.and.u32 pqr, pw{i}, {sent}, pqr;")
    lines.append(" selp.b32 $0, 1, 0, pqr;")
    lines.append("}")
    return "\n".join(lines)


@dsl_user_op
def lamport_probe_a(
    ptr: Pointer, offsets: cutlass.Constexpr, *, loc=None, ip=None
) -> Int32:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            MLIR_T.i32(),
            [ptr_int],
            _probe_asm(list(offsets), SENT_A),
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def lamport_probe_b(
    ptr: Pointer, offsets: cutlass.Constexpr, *, loc=None, ip=None
) -> Int32:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            MLIR_T.i32(),
            [ptr_int],
            _probe_asm(list(offsets), SENT_B),
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_bf16x2_to_2xf32(
    bf16x2_b32: Int32, *, loc=None, ip=None
) -> tuple[Float32, Float32]:
    ret = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [Int32(bf16x2_b32).ir_value(loc=loc, ip=ip)],
        "{ .reg .b16 hl, hh;\n"
        "  mov.b32 {hl, hh}, $2;\n"
        "  cvt.f32.bf16 $0, hl;\n"
        "  cvt.f32.bf16 $1, hh; }",
        "=f,=f,r",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(MLIR_T.f32(), ret, [0])),
        Float32(llvm.extractvalue(MLIR_T.f32(), ret, [1])),
    )


@dsl_user_op
def cvt_2xf32_to_bf16x2(f_lo: Float32, f_hi: Float32, *, loc=None, ip=None) -> Int32:
    return llvm.inline_asm(
        MLIR_T.i32(),
        [
            Float32(f_lo).ir_value(loc=loc, ip=ip),
            Float32(f_hi).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bitcast_i32_f32(v: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            MLIR_T.f32(),
            [Int32(v).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bitcast_f32_i32(v: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            MLIR_T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_gen(ptr: Pointer, *, loc=None, ip=None) -> Int32:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int32(
        llvm.inline_asm(
            MLIR_T.i32(),
            [ptr_int],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_gen(ptr: Pointer, v: Int32, *, loc=None, ip=None) -> None:
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr_int, Int32(v).ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def map_smem_rank(smem_ptr: Pointer, rank: Int32, *, loc=None, ip=None) -> Int32:
    """Translate a local SMEM address to the same slot in a cluster peer."""
    a = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            MLIR_T.i32(),
            [a, Int32(rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_smem_remote_f32(addr: Int32, val: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Float32(val).ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


def _pick_split(n_tok, nsm):
    """H-split factor: grow the grid until it covers the SM array."""
    if not _HAS_CLUSTER:
        return 1
    s = 1
    while s < 16 and n_tok * s * 2 <= nsm:
        s *= 2
    return s


# --------------------------------------------------------------------------- #
# kernel factory
# --------------------------------------------------------------------------- #
def _make_program(
    n_tok,
    num_rows,
    row_stride,
    grid_y,
    split,
    nthr,
    mode,
    eps,
    gain_center,
    expert_weights_fp32,
    ws,
    use_mc,
    addrs,
    my_rank,
):
    inv_h = 1.0 / float(H_DIM)
    n_iter = (n_tok + grid_y - 1) // grid_y
    comm = ws > 1
    nwarps = nthr // 32
    band = nthr * VEC  # H columns owned by one CTA
    peers = [r for r in range(ws) if r != my_rank]
    KO = K_EXPERTS // ws if comm else K_EXPERTS  # experts owned by this rank
    NLOC = KO if comm else K_EXPERTS
    my_experts = (
        [my_rank * KO + i for i in range(KO)] if comm else list(range(K_EXPERTS))
    )

    # element counts of the two Lamport regions
    A_ELEMS = NGEN * NSLOT * KO * ws * H_DIM  # BF16 elements
    A_BYTES = A_ELEMS * 2

    a_lam = addrs["lam_uc"]
    a_lam_mc = addrs["lam_mc"] if use_mc else addrs["lam_uc"]
    a_lam_peers = addrs["lam_peers"]
    a_gen = addrs["gen"]

    def _pin(dt, a, al=16):
        return cute.make_ptr(dt, a, cutlass.AddressSpace.gmem, assumed_align=al)

    def a_off(gm, slot, ko, src):
        # BF16 element offset of scattered expert row [gen][slot][ko][src]
        return (((gm * NSLOT + slot) * KO + ko) * ws + src) * H_DIM

    def b_off(gm, slot, src):
        # FP32 element offset of the partial output row [gen][slot][src]
        return ((gm * NSLOT + slot) * ws + src) * H_DIM

    # probe offsets are byte deltas from the (ko=0, src=0) anchor
    _pa_off = [
        (a_off(0, 0, ko, peers[r]) - a_off(0, 0, 0, 0)) * 2
        for ko in range(KO)
        for r in range(len(peers))
    ]
    _pb_off = [
        (b_off(0, 0, peers[r]) - b_off(0, 0, 0) + j * VEC // 2) * 4
        for r in range(len(peers))
        for j in range(2)
    ]

    @cute.jit
    def rearm(gen: Int32, slot: Int32, h0: Int32):
        """Re-arm generation g+2 (== g-1) of both Lamport regions.

        Runs for *every* generation this CTA slot advances through, including
        the tail iterations where ``tok >= n_tok`` leaves the slot idle: the
        slot still consumes a generation, so skipping the re-arm would leave a
        later launch polling a buffer that still holds live data from three
        generations back and it would accept those stale rows immediately.
        Each CTA arms only its own H band, so the S CTAs of a cluster together
        cover the whole region.
        """
        gc = (gen + 2) % NGEN
        pa = _pin(cutlass.BFloat16, a_lam)
        pb = _pin(cutlass.Float32, a_lam + A_BYTES)
        sa = cutlass.Int32(SENT_A_I32)
        sb = cutlass.Int32(SENT_B_I32)
        for ko in cutlass.range_constexpr(KO):
            for r in cutlass.range_constexpr(len(peers)):
                st_global_v4(pa + (a_off(gc, slot, ko, peers[r]) + h0), sa, sa, sa, sa)
        for r in cutlass.range_constexpr(len(peers)):
            base = b_off(gc, slot, peers[r]) + h0
            for j in cutlass.range_constexpr(2):
                st_global_v4(pb + (base + j * (VEC // 2)), sb, sb, sb, sb)

    @cute.jit
    def finalize_tile(
        gIn: cute.Tensor,
        gW: cute.Tensor,
        gIdx: cute.Tensor,
        gNw: cute.Tensor,
        gOut: cute.Tensor,
        sred: cute.Tensor,
        srstd: cute.Tensor,
        sxch: cute.Tensor,
        tok: Int32,
        tidx: Int32,
        sx: Int32,
        slot: Int32,
        gen: Int32,
        h0: Int32,
    ):
        lane = tidx % 32
        warp = tidx // 32

        # Issue the small vector loads first; they overlap the fabric wait.
        # FP32 routing weights occupy two 128-bit vectors, while the original
        # BF16 path remains one vector.
        nrm = ld_global_v4(gNw.iterator + h0)
        wgt_lo = ld_global_v4(gW.iterator + tok * K_EXPERTS)
        if cutlass.const_expr(expert_weights_fp32):
            wgt_hi = ld_global_v4(gW.iterator + tok * K_EXPERTS + 4)

        # Resolve physical GEMM2 rows directly from the expanded-to-permuted
        # map.  A missing or out-of-range slot contributes literal zero without
        # touching the fallback row: GEMM workspaces are allowed to contain
        # unwritten NaN/Inf values, for which multiplying by zero is not safe.
        # The predicate is uniform across the CTA because every thread handles
        # the same token and compile-time expert slot.
        mine = []
        for k in cutlass.range_constexpr(K_EXPERTS):
            permuted_idx = cutlass.Int32(gIdx[tok, k])
            is_valid = permuted_idx >= 0 and permuted_idx < num_rows
            q0 = cutlass.Int32(0)
            q1 = cutlass.Int32(0)
            q2 = cutlass.Int32(0)
            q3 = cutlass.Int32(0)
            if is_valid:
                q0, q1, q2, q3 = ld_global_v4(
                    gIn.iterator + (permuted_idx * row_stride + h0)
                )
            mine.append((scrub_a(q0), scrub_a(q1), scrub_a(q2), scrub_a(q3)))

        packed = []
        if cutlass.const_expr(comm):
            gm = gen % NGEN
            pa = _pin(cutlass.BFloat16, a_lam)

            # ---- phase 1: unicast every non-owned expert row to its owner.
            # One store per expert row; nothing is broadcast, so the rank's
            # fabric egress here is (ws - 1) / ws of its payload.
            for k in cutlass.range_constexpr(K_EXPERTS):
                o = k // KO
                if cutlass.const_expr(o != my_rank):
                    pp = _pin(cutlass.BFloat16, a_lam_peers[o])
                    q = mine[k]
                    st_global_v4(
                        pp + (a_off(gm, slot, k % KO, my_rank) + h0),
                        q[0],
                        q[1],
                        q[2],
                        q[3],
                    )

            # ---- FP32 cross-rank accumulate of the owned expert rows, in rank
            # order.  One spin covers every incoming slot, so the arrival test
            # is a single memory round rather than one per source rank.
            anchor = pa + (a_off(gm, slot, 0, 0) + h0)
            ok = lamport_probe_a(anchor, _pa_off)
            while ok == 0:
                ok = lamport_probe_a(anchor, _pa_off)

            acc = []
            for ko in cutlass.range_constexpr(KO):
                v = [cutlass.Float32(0.0) for _ in range(2 * 4)]
                for src in cutlass.range_constexpr(ws):
                    if cutlass.const_expr(src == my_rank):
                        dv = mine[my_experts[ko]]
                    else:
                        dv = ld_global_v4_volatile(pa + (a_off(gm, slot, ko, src) + h0))
                    for j in cutlass.range_constexpr(4):
                        lo, hi = cvt_bf16x2_to_2xf32(dv[j])
                        v[2 * j] = v[2 * j] + lo
                        v[2 * j + 1] = v[2 * j + 1] + hi
                acc.append(v)

            for ko in cutlass.range_constexpr(KO):
                packed.append(
                    tuple(
                        cvt_2xf32_to_bf16x2(acc[ko][2 * j], acc[ko][2 * j + 1])
                        for j in range(4)
                    )
                )

            # every incoming row is now in registers, so the sentinel re-arm can
            # be issued here and drain behind the RMS reduction and the epilogue
            rearm(gen, slot, h0)
        else:
            for k in cutlass.range_constexpr(K_EXPERTS):
                packed.append(mine[k])

        nw = []
        for r2 in cutlass.range_constexpr(4):
            lo, hi = cvt_bf16x2_to_2xf32(nrm[r2])
            nw.append(lo + gain_center)
            nw.append(hi + gain_center)
        ew = []
        if cutlass.const_expr(expert_weights_fp32):
            for r2 in cutlass.range_constexpr(4):
                ew.append(bitcast_i32_f32(wgt_lo[r2]))
            for r2 in cutlass.range_constexpr(4):
                ew.append(bitcast_i32_f32(wgt_hi[r2]))
        else:
            for r2 in cutlass.range_constexpr(4):
                lo, hi = cvt_bf16x2_to_2xf32(wgt_lo[r2])
                ew.append(lo)
                ew.append(hi)

        ssq = []
        for ko in cutlass.range_constexpr(NLOC):
            q = packed[ko]
            s = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(4):
                lo, hi = cvt_bf16x2_to_2xf32(q[j])
                if cutlass.const_expr(mode == 1):
                    lo = lo * nw[2 * j]
                    hi = hi * nw[2 * j + 1]
                s = s + lo * lo + hi * hi
            ssq.append(s)

        for ko in cutlass.range_constexpr(NLOC):
            v = ssq[ko]
            for si in cutlass.range_constexpr(5):
                v = v + cute.arch.shuffle_sync_bfly(v, offset=(1 << si))
            ssq[ko] = v
        if lane == 0:
            for ko in cutlass.range_constexpr(NLOC):
                sred[ko * nwarps + warp] = ssq[ko]
        cute.arch.barrier()

        if cutlass.const_expr(split == 1):
            if tidx < 32 * NLOC:
                kk = tidx // 32
                ll = tidx % 32
                v = cutlass.Float32(0.0)
                if ll < nwarps:
                    v = sred[kk * nwarps + ll]
                for si in cutlass.range_constexpr(5):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=(1 << si))
                if ll == 0:
                    srstd[kk] = cute.rsqrt(v * inv_h + eps)
            cute.arch.barrier()
        else:
            # the per-expert partials fold inside the CTA, then scatter into
            # every cluster peer's SMEM; one cluster barrier makes the whole
            # S x NLOC table local everywhere.
            if tidx < NLOC:
                v = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(nwarps):
                    v = v + sred[tidx * nwarps + w]
                slotp = sxch.iterator + (sx * NLOC + tidx)
                for d in cutlass.range_constexpr(split):
                    st_smem_remote_f32(map_smem_rank(slotp, cutlass.Int32(d)), v)
            cute.arch.barrier()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if tidx < NLOC:
                t = cutlass.Float32(0.0)
                for d in cutlass.range_constexpr(split):
                    t = t + sxch[d * NLOC + tidx]
                srstd[tidx] = cute.rsqrt(t * inv_h + eps)
            cute.arch.barrier()

        # ---- routing-weight multiply and ordered accumulation of the experts
        # this rank owns
        accs = [cutlass.Float32(0.0) for _ in range(8)]
        for ko in cutlass.range_constexpr(NLOC):
            r = srstd[ko]
            wk = ew[my_experts[ko]]
            for j in cutlass.range_constexpr(4):
                lo, hi = cvt_bf16x2_to_2xf32(packed[ko][j])
                if cutlass.const_expr(mode == 1):
                    lo = lo * nw[2 * j]
                    hi = hi * nw[2 * j + 1]
                ylo = lo * r
                yhi = hi * r
                if cutlass.const_expr(mode == 0):
                    ylo = ylo * nw[2 * j]
                    yhi = yhi * nw[2 * j + 1]
                accs[2 * j] = accs[2 * j] + ylo * wk
                accs[2 * j + 1] = accs[2 * j + 1] + yhi * wk

        if cutlass.const_expr(comm):
            gm = gen % NGEN
            pb = _pin(cutlass.Float32, a_lam + A_BYTES)
            pb_mc = _pin(cutlass.Float32, a_lam_mc + A_BYTES)

            # ---- phase 3: broadcast the FP32 partial output row.  This is a
            # quarter of the BF16 payload, so the multicast fan-out is cheap.
            mo = [bitcast_f32_i32(x) for x in accs]
            mo = [scrub_b(x) for x in mo]
            src_b = b_off(gm, slot, my_rank) + h0
            for j in cutlass.range_constexpr(2):
                q = mo[4 * j : 4 * j + 4]
                if cutlass.const_expr(use_mc):
                    multimem_st_v4(
                        pb_mc + (src_b + j * (VEC // 2)), q[0], q[1], q[2], q[3]
                    )
                else:
                    for r in cutlass.range_constexpr(len(peers)):
                        pp = _pin(cutlass.Float32, a_lam_peers[peers[r]] + A_BYTES)
                        st_global_v4(
                            pp + (b_off(gm, slot, my_rank) + h0 + j * (VEC // 2)),
                            q[0],
                            q[1],
                            q[2],
                            q[3],
                        )
            if cutlass.const_expr(use_mc):
                fence_proxy_alias()

            anchor_b = pb + (b_off(gm, slot, 0) + h0)
            okb = lamport_probe_b(anchor_b, _pb_off)
            while okb == 0:
                okb = lamport_probe_b(anchor_b, _pb_off)

            out = [cutlass.Float32(0.0) for _ in range(8)]
            for src in cutlass.range_constexpr(ws):
                if cutlass.const_expr(src == my_rank):
                    for j in cutlass.range_constexpr(8):
                        out[j] = out[j] + bitcast_i32_f32(mo[j])
                else:
                    for j in cutlass.range_constexpr(2):
                        dv = ld_global_v4_volatile(
                            pb + (b_off(gm, slot, src) + h0 + j * (VEC // 2))
                        )
                        for i in cutlass.range_constexpr(4):
                            out[4 * j + i] = out[4 * j + i] + bitcast_i32_f32(dv[i])
            accs = out

        o0 = cvt_2xf32_to_bf16x2(accs[0], accs[1])
        o1 = cvt_2xf32_to_bf16x2(accs[2], accs[3])
        o2 = cvt_2xf32_to_bf16x2(accs[4], accs[5])
        o3 = cvt_2xf32_to_bf16x2(accs[6], accs[7])
        st_global_v4(gOut.iterator + (tok * H_DIM + h0), o0, o1, o2, o3)

    @cute.kernel
    def fused_kernel(
        gIn: cute.Tensor,
        gW: cute.Tensor,
        gIdx: cute.Tensor,
        gNw: cute.Tensor,
        gOut: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        sx, cta, _ = cute.arch.block_idx()

        smem = cutlass_utils.SmemAllocator()
        sred = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(nwarps * NLOC),
            byte_alignment=16,
        )
        srstd = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(NLOC),
            byte_alignment=16,
        )
        sxch = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(split * NLOC),
            byte_alignment=16,
        )
        h0 = sx * band + tidx * VEC

        g0 = cutlass.Int32(0)
        if cutlass.const_expr(comm):
            g0 = ld_gen(_pin(cutlass.Int32, a_gen, 4) + cta)

        for it in cutlass.range(n_iter):
            tok = cta + it * grid_y
            if tok < n_tok:
                finalize_tile(
                    gIn,
                    gW,
                    gIdx,
                    gNw,
                    gOut,
                    sred,
                    srstd,
                    sxch,
                    tok,
                    tidx,
                    sx,
                    cta,
                    g0 + it,
                    h0,
                )
            else:
                # idle tail iteration: the slot still burns a generation, so it
                # must still hand a sentinel-armed buffer to a later launch
                if cutlass.const_expr(comm):
                    rearm(g0 + it, cta, h0)

        # every CTA of the cluster has read the counter by now (they all passed
        # the cluster barrier inside the first tile), so one writer is safe.
        if cutlass.const_expr(comm) and tidx == 0 and sx == 0:
            st_gen(_pin(cutlass.Int32, a_gen, 4) + cta, g0 + n_iter)

    @cute.jit
    def host(
        gIn: cute.Tensor,
        gW: cute.Tensor,
        gIdx: cute.Tensor,
        gNw: cute.Tensor,
        gOut: cute.Tensor,
        stream: _drv.CUstream,
    ):
        if cutlass.const_expr(split > 1):
            fused_kernel(gIn, gW, gIdx, gNw, gOut).launch(
                grid=(split, grid_y, 1),
                block=(nthr, 1, 1),
                cluster=(split, 1, 1),
                min_blocks_per_mp=1,
                stream=stream,
            )
        else:
            fused_kernel(gIn, gW, gIdx, gNw, gOut).launch(
                grid=(1, grid_y, 1),
                block=(nthr, 1, 1),
                min_blocks_per_mp=1,
                stream=stream,
            )

    return host


# --------------------------------------------------------------------------- #
# host side
# --------------------------------------------------------------------------- #
class _Ws:
    initialized = False
    available = False
    lam = None
    hdl = None
    gen: torch.Tensor | None = None
    addrs: "_WorkspaceAddresses | None" = None
    use_mc = False
    ws = 1
    rank = 0
    nsm = 148
    dev_ord = 0
    group = None


_WS = _Ws()


class _WorkspaceAddresses(TypedDict):
    lam_uc: int
    lam_mc: int
    lam_peers: list[int]
    gen: int


_PROGRAMS: dict[tuple[int, int, int, int, int, float, float], "_Program"] = {}


def _group_all(
    local_ok: bool,
    device: torch.device,
    group,
    flag: torch.Tensor | None = None,
) -> bool:
    """Return a rank-consistent logical AND of an initialization verdict."""
    import torch.distributed as dist

    if flag is None:
        flag = torch.empty((1,), dtype=torch.int32, device=device)
    flag.fill_(int(local_ok))
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def init_tp_moe_finalize_rmsnorm(group=None) -> bool:
    """Collectively initialize the process-wide TP multicast workspace.

    ``group`` is the exact tensor-parallel group. Its size must exceed one and
    evenly divide ``K_EXPERTS``. The default process group is used when it is
    omitted. Availability is reduced across the group, so every rank returns
    the same verdict.
    """
    if _WS.initialized:
        if group is not None and group is not _WS.group:
            raise RuntimeError(
                "TP MoE finalize is already initialized for another group"
            )
        return _WS.available

    import torch.distributed as dist

    if group is None:
        group = dist.group.WORLD
    _WS.group = group
    world_size = dist.get_world_size(group)
    if world_size <= 1 or K_EXPERTS % world_size != 0:
        _WS.initialized = True
        return False

    device = torch.device("cuda", torch.accelerator.current_device_index())
    props = torch.cuda.get_device_properties(device)
    local_supported = (props.major, props.minor) in _COMPUTE_CAPABILITIES
    local_supported = local_supported and props.multi_processor_count <= NSLOT
    if not _group_all(local_supported, device, group):
        if not local_supported:
            logger.warning(
                "TP MoE finalize requires SM100/SM103 and at most %d SMs; got SM%d%d "
                "with %d SMs.",
                NSLOT,
                props.major,
                props.minor,
                props.multi_processor_count,
            )
        else:
            logger.warning(
                "TP MoE finalize is unavailable because another rank does not "
                "satisfy the device or workspace-slot contract."
            )
        _WS.initialized = True
        return False

    _WS.nsm = props.multi_processor_count
    _WS.dev_ord = torch.cuda.current_device()
    _WS.ws = world_size
    _WS.rank = dist.get_rank(group)
    _WS.addrs = {"lam_uc": 0, "lam_mc": 0, "lam_peers": [], "gen": 0}

    import torch.distributed._symmetric_memory as symm_mem

    ko = K_EXPERTS // _WS.ws
    a_i32 = NGEN * NSLOT * ko * _WS.ws * H_DIM // 2  # BF16 region, in int32 words
    b_i32 = NGEN * NSLOT * _WS.ws * H_DIM  # FP32 region, in int32 words
    allocation_flag = torch.empty((1,), dtype=torch.int32, device=device)
    lam = None
    generation = None
    allocation_error = None
    try:
        lam = symm_mem.empty((a_i32 + b_i32,), dtype=torch.int32, device=device)
        generation = torch.zeros((NSLOT,), dtype=torch.int32, device=device)
    except RuntimeError as error:
        allocation_error = error
    if not _group_all(allocation_error is None, device, group, flag=allocation_flag):
        if allocation_error is not None:
            logger.warning(
                "TP MoE finalize workspace allocation failed; using the native "
                "finalize: %s",
                allocation_error,
            )
        else:
            logger.warning(
                "TP MoE finalize is unavailable because another rank could not "
                "allocate its workspace."
            )
        _WS.initialized = True
        return False
    assert lam is not None and generation is not None
    lam[:a_i32].fill_(SENT_A_I32)
    lam[a_i32:].fill_(SENT_B_I32)
    _WS.gen = generation
    torch.cuda.synchronize()
    hdl = symm_mem.rendezvous(lam, group=group)
    _WS.lam, _WS.hdl = lam, hdl

    mc = int(getattr(hdl, "multicast_ptr", 0) or 0)
    if not _group_all(mc != 0, device, group):
        if mc == 0:
            logger.warning(
                "TP MoE finalize requires NVLink multicast memory; "
                "using the native finalize."
            )
        else:
            logger.warning(
                "TP MoE finalize is unavailable because another rank has no "
                "NVLink multicast mapping."
            )
        _WS.initialized = True
        return False
    _WS.use_mc = True
    try:
        peers = [int(p) for p in hdl.buffer_ptrs]
    except (AttributeError, TypeError):
        peers = [
            hdl.get_buffer(r, (a_i32 + b_i32,), torch.int32).data_ptr()
            for r in range(_WS.ws)
        ]
    _WS.addrs = {
        "lam_uc": lam.data_ptr(),
        "lam_mc": mc,
        "lam_peers": peers,
        "gen": generation.data_ptr(),
    }
    torch.accelerator.synchronize(device)
    dist.barrier(group=group)
    _WS.available = True
    _WS.initialized = True
    return True


class _Program:
    __slots__ = ("fn", "ffi")

    def __init__(self, fn, ffi):
        self.fn = fn
        self.ffi = ffi

    def __call__(self, a, b, c, d, e, stream):
        if self.ffi:
            self.fn(a, b, c, d, e, stream)
        else:
            self.fn(
                from_dlpack(a, assumed_align=16),
                from_dlpack(b, assumed_align=16),
                from_dlpack(c, assumed_align=16),
                from_dlpack(d, assumed_align=16),
                from_dlpack(e, assumed_align=16),
                stream,
            )


def _build(key, a, b, c, d, e, stream):
    # The generated program embeds process-local symmetric-memory and multicast
    # addresses. It must therefore remain in the process cache rather than the
    # persistent CuTe-DSL cache, whose objects are reusable across processes.
    n_tok, num_rows, row_stride, expert_weights_fp32, mode, eps, gain_center = key
    split = _pick_split(n_tok, _WS.nsm)
    nthr = H_DIM // (VEC * split)
    grid_y = min(n_tok, _WS.nsm // split)
    addrs = _WS.addrs
    if addrs is None:
        raise RuntimeError("TP MoE finalize workspace addresses are unavailable")
    host = _make_program(
        n_tok,
        num_rows,
        row_stride,
        grid_y,
        split,
        nthr,
        mode,
        eps,
        gain_center,
        expert_weights_fp32,
        _WS.ws,
        _WS.use_mc,
        addrs,
        _WS.rank,
    )
    try:
        args = [
            from_dlpack(x, assumed_align=16, enable_tvm_ffi=True)
            for x in (a, b, c, d, e)
        ]
        compiled = cute.compile(host, *args, stream, options="--enable-tvm-ffi")
        return _Program(compiled, True)
    except Exception as error:
        logger.warning(
            "TVM-FFI compilation failed for TP MoE finalize; using the "
            "process-local CuTe callable: %s",
            error,
        )
        args = [from_dlpack(x, assumed_align=16) for x in (a, b, c, d, e)]
        return _Program(cute.compile(host, *args, stream), False)


def _run(
    gemm2_permuted,
    expert_weights,
    expanded_idx_to_permuted_idx,
    norm_weight,
    eps,
    gain_center,
    weight_before_norm,
    output,
):
    n_tok = expanded_idx_to_permuted_idx.shape[0]
    if n_tok == 0:
        return
    if not _WS.available:
        raise RuntimeError("TP MoE finalize workspace was not initialized")
    key = (
        n_tok,
        gemm2_permuted.shape[0],
        gemm2_permuted.stride(0),
        int(expert_weights.dtype == torch.float32),
        int(weight_before_norm),
        float(eps),
        float(gain_center),
    )
    capturing = torch.cuda.is_current_stream_capturing()
    stream = _cur_stream(_WS.dev_ord)
    prog = _PROGRAMS.get(key)
    if capturing and prog is None:
        raise RuntimeError(
            "TP MoE finalize specialization must be compiled by a non-capturing "
            "warmup before CUDA graph capture"
        )
    if prog is None:
        prog = _build(
            key,
            gemm2_permuted,
            expert_weights,
            expanded_idx_to_permuted_idx,
            norm_weight,
            output,
            stream,
        )
        _PROGRAMS[key] = prog
    prog(
        gemm2_permuted,
        expert_weights,
        expanded_idx_to_permuted_idx,
        norm_weight,
        output,
        stream,
    )


def run(
    gemm2_permuted,
    expert_weights,
    expanded_idx_to_permuted_idx,
    norm_weight,
    eps,
    gain_center,
    weight_before_norm,
    output,
):
    """Direct entry point with the same ordering contract as the public API."""
    if not init_tp_moe_finalize_rmsnorm():
        raise RuntimeError("TP multicast finalize is unavailable")
    if not supports_tp_moe_finalize_rmsnorm(
        gemm2_permuted,
        expert_weights,
        expanded_idx_to_permuted_idx,
        norm_weight,
        output,
    ):
        raise ValueError("tensors do not satisfy the TP MoE finalize contract")
    _run(
        gemm2_permuted,
        expert_weights,
        expanded_idx_to_permuted_idx,
        norm_weight,
        eps,
        gain_center,
        weight_before_norm,
        output,
    )


def supports_tp_moe_finalize_rmsnorm(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the TP SM100/SM103 kernel contract."""
    if (
        not _WS.available
        or gemm2_permuted.ndim != 2
        or expert_weights.ndim != 2
        or expanded_idx_to_permuted_idx.ndim != 2
        or norm_weight.ndim != 1
        or output.ndim != 2
    ):
        return False
    workspace_device = torch.device("cuda", _WS.dev_ord)
    num_tokens = expanded_idx_to_permuted_idx.shape[0]
    max_input_offset = (
        0
        if gemm2_permuted.shape[0] == 0
        else (gemm2_permuted.shape[0] - 1) * gemm2_permuted.stride(0) + H_DIM
    )
    bf16_tensors = (gemm2_permuted, norm_weight, output)
    return (
        gemm2_permuted.shape[1] >= H_DIM
        and gemm2_permuted.stride(1) == 1
        and gemm2_permuted.stride(0) % VEC == 0
        and (num_tokens == 0 or gemm2_permuted.shape[0] > 0)
        and max_input_offset <= torch.iinfo(torch.int32).max
        and expert_weights.shape == (num_tokens, K_EXPERTS)
        and expanded_idx_to_permuted_idx.shape == (num_tokens, K_EXPERTS)
        and norm_weight.shape == (H_DIM,)
        and output.shape == (num_tokens, H_DIM)
        and expert_weights.dtype in (torch.bfloat16, torch.float32)
        and expert_weights.is_cuda
        and expert_weights.is_contiguous()
        and expert_weights.device == gemm2_permuted.device
        and expert_weights.device == workspace_device
        and expert_weights.data_ptr() % 16 == 0
        and all(
            tensor.dtype == torch.bfloat16
            and tensor.is_cuda
            and tensor.is_contiguous()
            and tensor.device == gemm2_permuted.device
            and tensor.device == workspace_device
            and tensor.data_ptr() % 16 == 0
            for tensor in bf16_tensors
        )
        and expanded_idx_to_permuted_idx.dtype == torch.int32
        and expanded_idx_to_permuted_idx.is_cuda
        and expanded_idx_to_permuted_idx.is_contiguous()
        and expanded_idx_to_permuted_idx.device == gemm2_permuted.device
        and expanded_idx_to_permuted_idx.device == workspace_device
        and expanded_idx_to_permuted_idx.data_ptr() % 16 == 0
    )


def _tp_moe_finalize_impl(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    gain_center: float,
    weight_before_norm: bool,
) -> None:
    if not supports_tp_moe_finalize_rmsnorm(
        gemm2_permuted,
        expert_weights,
        expanded_idx_to_permuted_idx,
        norm_weight,
        output,
    ):
        raise ValueError("tensors do not satisfy the TP MoE finalize contract")
    _run(
        gemm2_permuted,
        expert_weights,
        expanded_idx_to_permuted_idx,
        norm_weight,
        eps,
        gain_center,
        weight_before_norm,
        output,
    )


def _tp_moe_finalize_fake(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    gain_center: float,
    weight_before_norm: bool,
) -> None:
    return None


_tp_moe_finalize_impl = register_custom_op(
    "flashinfer::tp_moe_finalize_rmsnorm", mutates_args=("output",)
)(_tp_moe_finalize_impl)
_tp_moe_finalize_fake = register_fake_op("flashinfer::tp_moe_finalize_rmsnorm")(
    _tp_moe_finalize_fake
)


@flashinfer_api(trace=tp_moe_finalize_rmsnorm_trace)
def tp_moe_finalize_rmsnorm(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
    *,
    eps: float,
    gain_center: float,
    weight_before_norm: bool,
) -> None:
    """Gather and finalize TP expert rows, then apply per-expert RMSNorm.

    ``gemm2_permuted`` contains the raw, unweighted GEMM2 rows in physical
    expert order. ``expanded_idx_to_permuted_idx[t, k]`` selects the physical
    row for logical token ``t`` and expert slot ``k``; a negative or
    out-of-range index contributes zero. The input may have a padded trailing
    dimension, but its first 4096 columns and every row start must be contiguous
    and 16-byte aligned. ``expert_weights`` may be BF16 or FP32; either dtype is
    converted to FP32 registers and remains FP32 through the routing reduction.

    Initialize the workspace collectively with
    :func:`init_tp_moe_finalize_rmsnorm` before the first invocation. The
    operation writes its BF16 result to ``output``. All invocations share one
    process-wide Lamport workspace and must not overlap, including CUDA graph
    replays. Every rank must issue the same call sequence. When switching streams,
    the caller must order their work with synchronization or event dependencies;
    this operation does not insert synchronization between streams. Warm up every
    specialization outside CUDA graph capture before capturing it.
    """
    _tp_moe_finalize_impl(
        gemm2_permuted,
        expert_weights,
        expanded_idx_to_permuted_idx,
        norm_weight,
        output,
        eps,
        gain_center,
        weight_before_norm,
    )
