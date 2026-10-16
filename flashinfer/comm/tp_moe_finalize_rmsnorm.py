# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""TP MoE finalize and post-expert RMSNorm on SM100/SM103/SM107.

Candidate-owned collectives, no framework collective. The input is the raw
permuted GEMM2 buffer plus its expanded-to-permuted row map; resolving that map
inside the kernel avoids materializing the logical [T,K,H] partial-row tensor.
The TP group size P must exceed one and evenly divide K_EXPERTS. Each rank
owns KO = K_EXPERTS / P expert slots on the small-token path:

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
squares, exchanged through distributed shared memory with startup and completion
cluster barriers.
``S`` is a launch parameter: small token counts leave most of the machine idle
with one CTA per token, so S grows until the grid covers the SM array.

From ``64 / P`` tokens on, rows are owned by token instead: rank
``token % P`` sums every expert's partial rows in rank order, normalizes and
accumulates the experts in FP32, then multicasts the final BF16 row. This moves
less data than the expert-owner path (the second hop carries BF16 output rows,
not FP32 partials).

  single cooperative launch (padded token count <= SM count and the compiled
      grid passes a collective residency check): CTA ``(b, g, d)`` covers H
      band ``b`` of token ``g * P + d``. It pushes the token's rows to rank
      ``d``, reduces some experts of the token this rank owns in slot ``g``,
      and copies rank ``d``'s multicast row out. Both hops are Lamport
      transfers with their own triple-buffered generations; the owner's
      experts are spread over ``XP`` CTAs of the slot (combined through a
      local scratch by the last arriver) or kept in one CTA when that is
      faster. Every CTA is co-resident and pushes before it polls.
  chunked (more tokens): separate staging, owner, and copy launches with
      release/acquire flags. The copy waits for every owner before input
      storage is reused; the next owner waits for every rank's next stage, so
      all previous output reads finish before overwrite.
"""

import logging
import tempfile
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
from ..cute_dsl.availability import is_cute_dsl_arch_supported
from ..trace.templates.comm import tp_moe_finalize_rmsnorm_trace
from ..utils import register_custom_op, register_fake_op
from .mnnvl_cutedsl.cute_dsl_primitives import (
    map_shared_to_peer as map_smem_rank,
    store_shared_cluster_f32 as st_smem_remote_f32,
)
from .mnnvl_cutedsl.runtime import collective_support

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
_COMPUTE_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
H_DIM = 4096
VEC = 8
NSLOT = 256  # >= max CTAs in the grid; one Lamport slot per CTA
NGEN = 3
_MAX_SPLIT = 16
# In-flight 128-bit peer-row loads per thread when an owner batches its loads.
_PEER_BATCH_VECS = 32
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
def ld_acquire(ptr: Pointer, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            MLIR_T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_release(ptr: Pointer, value: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.release.sys.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_acq_rel(ptr: Pointer, value: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            MLIR_T.i32(),
            [
                ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


def _peer_batch(ws):
    """Experts whose peer rows (ws - 1 vectors each) fit one load batch."""
    return max(1, min(K_EXPERTS, _PEER_BATCH_VECS // max(1, ws - 1)))


def _pick_split(n_tok, nsm):
    """H-split factor: grow the grid until it covers the SM array."""
    if not _HAS_CLUSTER:
        return 1
    s = 1
    while s < _MAX_SPLIT and n_tok * s * 2 <= nsm:
        s *= 2
    return s


# --------------------------------------------------------------------------- #
# kernel factory
# --------------------------------------------------------------------------- #
def _make_program(
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
    token_owner=False,
    single=False,
    xpar=None,
    push=None,
):
    inv_h = 1.0 / float(H_DIM)
    comm = ws > 1
    nwarps = nthr // 32
    band = nthr * VEC  # H columns owned by one CTA
    peers = [r for r in range(ws) if r != my_rank]
    KO = K_EXPERTS if token_owner else K_EXPERTS // ws if comm else K_EXPERTS
    NLOC = KO if comm else K_EXPERTS
    my_experts = (
        [my_rank * KO + i for i in range(KO)]
        if comm and not token_owner
        else list(range(K_EXPERTS))
    )

    # element counts of the two Lamport regions
    A_ELEMS = NGEN * NSLOT * KO * ws * H_DIM  # BF16 elements
    A_BYTES = A_ELEMS * 2

    a_lam = addrs["token_uc"] if token_owner else addrs["lam_uc"]
    a_lam_mc = (
        (addrs["token_mc"] if token_owner else addrs["lam_mc"]) if use_mc else a_lam
    )
    a_lam_peers = addrs["token_peers"] if token_owner else addrs["lam_peers"]
    a_gen = addrs["token_gen"] if token_owner else addrs["gen"]
    token_slots = addrs["token_slots"]
    if token_owner:
        A_BYTES = token_slots * K_EXPERTS * ws * H_DIM * 2
    flag_bytes = A_BYTES + token_slots * ws * H_DIM * 2
    # Token-path owners load this many experts' peer rows in one batch, so the
    # loads share one memory latency.
    load_grp = _peer_batch(ws)
    # Rows pushed per load batch; the chunked path pushes all of them at once.
    push_grp = K_EXPERTS
    NS = NSLOT  # slots per Lamport generation
    if single:
        # Lamport transport with its own generations: triple-buffered staged
        # rows, then the owners' output rows; plus local combine scratch.
        assert xpar is not None and push is not None
        push_grp = push
        NS = addrs["tl_slots"]
        a_lam = addrs["tl_uc"]
        a_lam_mc = addrs["tl_mc"] if use_mc else a_lam
        a_lam_peers = addrs["tl_peers"]
        A_BYTES = NGEN * NS * K_EXPERTS * ws * H_DIM * 2
        a_gen = addrs["tl_gen"]
        a_done = addrs["tl_done"]
        a_y = addrs["tl_y"]
        a_ycnt = addrs["tl_ycnt"]

    def _pin(dt, a, al=16):
        return cute.make_ptr(dt, a, cutlass.AddressSpace.gmem, assumed_align=al)

    def a_off(gm, slot, ko, src):
        # BF16 element offset of scattered expert row [gen][slot][ko][src]
        if token_owner and not single:
            return ((slot * K_EXPERTS + ko) * ws + src) * H_DIM
        return (((gm * NS + slot) * KO + ko) * ws + src) * H_DIM

    def b_off(gm, slot, src):
        # Element offset of the output row [gen][slot][src]: FP32 partials on
        # the expert path, BF16 owner rows on the token paths.
        if token_owner and not single:
            return (slot * ws + src) * H_DIM
        return ((gm * NS + slot) * ws + src) * H_DIM

    def flag_off(slot, phase, src):
        return (slot * 2 + phase) * ws + src

    @cute.jit
    def push_rows(
        gIn: cute.Tensor,
        gIdx: cute.Tensor,
        tok: Int32,
        dest: cutlass.Constexpr,
        gm: Int32,
        slot: Int32,
        h0: Int32,
    ):
        num_rows = gIn.shape[0]
        pp = _pin(cutlass.BFloat16, a_lam_peers[dest])
        # Rows are loaded push_grp at a time before their stores: a store waits
        # for its data, so a larger group overlaps more load latency, at the
        # cost of burstier fabric traffic when many CTAs push at once. The
        # index row comes first: the asm stores order memory, so an index load
        # inside the loop would add a dependent latency per batch.
        idxs = [cutlass.Int32(gIdx[tok, k]) for k in range(K_EXPERTS)]
        for k0 in cutlass.range_constexpr(0, K_EXPERTS, push_grp):
            rows = []
            for k in cutlass.range_constexpr(k0, k0 + push_grp):
                idx = idxs[k]
                q0, q1, q2, q3 = (
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                )
                if idx >= 0 and idx < num_rows:
                    q0, q1, q2, q3 = ld_global_v4(gIn.iterator + idx * row_stride + h0)
                rows.append((q0, q1, q2, q3))
            for k in cutlass.range_constexpr(k0, k0 + push_grp):
                q = rows[k - k0]
                if cutlass.const_expr(single):
                    # the single launch polls these rows, so scrub the sentinel
                    q = tuple(scrub_a(w) for w in q)
                st_global_v4(
                    pp + a_off(gm, slot, k, my_rank) + h0, q[0], q[1], q[2], q[3]
                )

    @cute.jit
    def wait_phase(slot: Int32, phase: cutlass.Constexpr, gen: Int32, tidx: Int32):
        if tidx < ws and tidx != my_rank:
            flags = _pin(cutlass.Int32, a_lam + flag_bytes, 4)
            ptr = flags + flag_off(slot, phase, tidx)
            value = ld_acquire(ptr)
            while value != gen:
                value = ld_acquire(ptr)
        cute.arch.barrier()

    @cute.jit
    def publish_phase(slot: Int32, phase: cutlass.Constexpr, gen: Int32, tidx: Int32):
        # CTA synchronization carries every writer's accesses into the
        # system release/acquire chain, as in multi_gpu_barrier.
        cute.arch.barrier()
        if tidx < ws and tidx != my_rank:
            peer = cutlass.Int64(a_lam_peers[0])
            for r in cutlass.range_constexpr(len(peers)):
                if tidx == peers[r]:
                    peer = cutlass.Int64(a_lam_peers[peers[r]])
            flags = _pin(cutlass.Int32, peer + flag_bytes, 4)
            st_release(flags + flag_off(slot, phase, my_rank), gen)
        cute.arch.barrier()

    @cute.jit
    def scatter_tokens(
        gIn: cute.Tensor,
        gIdx: cute.Tensor,
        tok: Int32,
        slot: Int32,
        gen: Int32,
        tidx: Int32,
        h0: Int32,
    ):
        n_tok = gIdx.shape[0]
        # The previous copy waited for every owner, which had consumed all
        # input rows before publishing its output. Input storage is now free.
        for dest in cutlass.range_constexpr(ws):
            if cutlass.const_expr(dest != my_rank):
                if tok < n_tok and tok % ws == dest:
                    push_rows(gIn, gIdx, tok, dest, cutlass.Int32(0), slot, h0)
        cute.arch.barrier()
        if tidx == 0:
            for dest in cutlass.range_constexpr(ws):
                if cutlass.const_expr(dest != my_rank):
                    if tok % ws == dest:
                        flags = _pin(cutlass.Int32, a_lam_peers[dest] + flag_bytes, 4)
                        st_release(flags + flag_off(slot, 0, my_rank), gen)

    @cute.jit
    def copy_tokens(
        gOut: cute.Tensor,
        base: Int32,
        slot: Int32,
        gen: Int32,
        tidx: Int32,
        h0: Int32,
    ):
        n_tok = gOut.shape[0]
        wait_phase(slot, 1, gen, tidx)
        pb = _pin(cutlass.BFloat16, a_lam + A_BYTES)
        for src in cutlass.range_constexpr(ws):
            tok = base + src
            if tok < n_tok:
                q = ld_global_v4(pb + b_off(0, slot, src) + h0)
                st_global_v4(gOut.iterator + tok * H_DIM + h0, q[0], q[1], q[2], q[3])
        # The next owner waits for every rank's next stage, which follows this
        # copy on that rank's stream. It cannot overwrite any live output row.
        cute.arch.barrier()

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
    def advance_idle(gen: Int32, slot: Int32, h0: Int32):
        # Publishing the intervening generation proves peers have consumed the
        # previous one before rearm clears it, even when this slot has no token.
        gm = gen % NGEN
        zero = cutlass.Int32(0)
        for k in cutlass.range_constexpr(K_EXPERTS):
            owner = k // KO
            if cutlass.const_expr(owner != my_rank):
                pp = _pin(cutlass.BFloat16, a_lam_peers[owner])
                st_global_v4(
                    pp + a_off(gm, slot, k % KO, my_rank) + h0, zero, zero, zero, zero
                )
        pa = _pin(cutlass.BFloat16, a_lam)
        ok = lamport_probe_a(pa + a_off(gm, slot, 0, 0) + h0, _pa_off)
        while ok == 0:
            ok = lamport_probe_a(pa + a_off(gm, slot, 0, 0) + h0, _pa_off)
        rearm(gen, slot, h0)
        pb = _pin(cutlass.Float32, a_lam + A_BYTES)
        pb_mc = _pin(cutlass.Float32, a_lam_mc + A_BYTES)
        for j in cutlass.range_constexpr(2):
            multimem_st_v4(
                pb_mc + b_off(gm, slot, my_rank) + h0 + j * (VEC // 2),
                zero,
                zero,
                zero,
                zero,
            )
        fence_proxy_alias()
        ok = lamport_probe_b(pb + b_off(gm, slot, 0) + h0, _pb_off)
        while ok == 0:
            ok = lamport_probe_b(pb + b_off(gm, slot, 0) + h0, _pb_off)

    # ---- straight-line math shared by finalize_tile and single_kernel. Plain
    # Python: they emit into the caller's trace and hold no runtime branches.
    def decode_gains(nrm, wgt_lo, wgt_hi):
        """Norm gains (with gain_center) and FP32 routing weights."""
        nw = []
        for q in range(4):
            lo, hi = cvt_bf16x2_to_2xf32(nrm[q])
            nw.append(lo + gain_center)
            nw.append(hi + gain_center)
        ew = []
        if expert_weights_fp32:
            ew = [bitcast_i32_f32(w) for w in (*wgt_lo, *wgt_hi)]
        else:
            for q in range(4):
                lo, hi = cvt_bf16x2_to_2xf32(wgt_lo[q])
                ew.append(lo)
                ew.append(hi)
        return nw, ew

    def sum_ranks(rows):
        """FP32 sum of one expert's BF16 rows, in the given (rank) order."""
        v = [cutlass.Float32(0.0) for _ in range(8)]
        for dv in rows:
            for q in range(4):
                lo, hi = cvt_bf16x2_to_2xf32(dv[q])
                v[2 * q] = v[2 * q] + lo
                v[2 * q + 1] = v[2 * q + 1] + hi
        return v

    def pack8(v):
        return tuple(cvt_2xf32_to_bf16x2(v[2 * q], v[2 * q + 1]) for q in range(4))

    def warp_ssq(row, nw):
        """Warp-reduced sum of squares of one packed BF16 expert row."""
        s = cutlass.Float32(0.0)
        for q in range(4):
            lo, hi = cvt_bf16x2_to_2xf32(row[q])
            if mode == 1:
                lo = lo * nw[2 * q]
                hi = hi * nw[2 * q + 1]
            s = s + lo * lo + hi * hi
        for si in range(5):
            s = s + cute.arch.shuffle_sync_bfly(s, offset=(1 << si))
        return s

    def norm_row(row, rs, nw):
        """RMS-normalized, gained FP32 values of one packed BF16 expert row."""
        y = []
        for q in range(4):
            lo, hi = cvt_bf16x2_to_2xf32(row[q])
            if mode == 1:
                lo = lo * nw[2 * q]
                hi = hi * nw[2 * q + 1]
            lo = lo * rs
            hi = hi * rs
            if mode == 0:
                lo = lo * nw[2 * q]
                hi = hi * nw[2 * q + 1]
            y.append(lo)
            y.append(hi)
        return y

    @cute.jit
    def rms_finish(
        n: cutlass.Constexpr,
        sred: cute.Tensor,
        srstd: cute.Tensor,
        sxch: cute.Tensor,
        tidx: Int32,
        sx: Int32,
    ):
        """Fold the per-warp sums of squares in ``sred`` into ``srstd``."""
        if cutlass.const_expr(split == 1):
            if tidx < 32 * n:
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
            # S x n table local everywhere.
            cute.arch.cluster_wait()
            if tidx < n:
                v = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(nwarps):
                    v = v + sred[tidx * nwarps + w]
                slotp = sxch.iterator + (sx * n + tidx)
                for d in cutlass.range_constexpr(split):
                    st_smem_remote_f32(map_smem_rank(slotp, cutlass.Int32(d)), v)
            cute.arch.barrier()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if tidx < n:
                t = cutlass.Float32(0.0)
                for d in cutlass.range_constexpr(split):
                    t = t + sxch[d * n + tidx]
                srstd[tidx] = cute.rsqrt(t * inv_h + eps)
            cute.arch.barrier()

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
        if cutlass.const_expr(split > 1):
            cute.arch.cluster_arrive_relaxed()
        lane = tidx % 32
        warp = tidx // 32

        num_rows = gIn.shape[0]
        # Issue the small vector loads first; they overlap the fabric wait.
        # FP32 routing weights occupy two 128-bit vectors, while the original
        # BF16 path remains one vector.
        nrm = ld_global_v4(gNw.iterator + h0)
        wgt_lo = ld_global_v4(gW.iterator + tok * K_EXPERTS)
        wgt_hi = None
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
                if cutlass.const_expr(not token_owner and o != my_rank):
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
            if cutlass.const_expr(not token_owner):
                anchor = pa + (a_off(gm, slot, 0, 0) + h0)
                ok = lamport_probe_a(anchor, _pa_off)
                while ok == 0:
                    ok = lamport_probe_a(anchor, _pa_off)
            acc = []
            pre = {}
            for ko in cutlass.range_constexpr(KO):
                if cutlass.const_expr(token_owner and ko % load_grp == 0):
                    # Issue a whole group of peer rows before the first use, so
                    # the loads share one memory latency instead of serializing.
                    for kk in cutlass.range_constexpr(ko, min(ko + load_grp, KO)):
                        for r in cutlass.range_constexpr(len(peers)):
                            pre[(kk, peers[r])] = ld_global_v4_volatile(
                                pa + (a_off(gm, slot, kk, peers[r]) + h0)
                            )
                rows = []
                for src in cutlass.range_constexpr(ws):
                    if cutlass.const_expr(src == my_rank):
                        rows.append(mine[my_experts[ko]])
                    elif cutlass.const_expr(token_owner):
                        rows.append(pre[(ko, src)])
                    else:
                        rows.append(
                            ld_global_v4_volatile(pa + (a_off(gm, slot, ko, src) + h0))
                        )
                acc.append(sum_ranks(rows))
            packed = [pack8(v) for v in acc]

            # every incoming row is now in registers, so the sentinel re-arm can
            # be issued here and drain behind the RMS reduction and the epilogue
            if cutlass.const_expr(not token_owner):
                rearm(gen, slot, h0)
        else:
            for k in cutlass.range_constexpr(K_EXPERTS):
                packed.append(mine[k])

        nw, ew = decode_gains(nrm, wgt_lo, wgt_hi)
        ssq = [warp_ssq(packed[ko], nw) for ko in range(NLOC)]
        if lane == 0:
            for ko in cutlass.range_constexpr(NLOC):
                sred[ko * nwarps + warp] = ssq[ko]
        cute.arch.barrier()
        rms_finish(NLOC, sred, srstd, sxch, tidx, sx)

        # ---- routing-weight multiply and ordered accumulation of the experts
        # this rank owns
        accs = [cutlass.Float32(0.0) for _ in range(8)]
        for ko in cutlass.range_constexpr(NLOC):
            wk = ew[my_experts[ko]]
            y = norm_row(packed[ko], srstd[ko], nw)
            for i in cutlass.range_constexpr(8):
                accs[i] = accs[i] + y[i] * wk

        if cutlass.const_expr(comm and not token_owner):
            gm = gen % NGEN
            pb = _pin(cutlass.Float32, a_lam + A_BYTES)
            pb_mc = _pin(cutlass.Float32, a_lam_mc + A_BYTES)

            # ---- phase 3: broadcast the FP32 partial output row.  This is a
            # quarter of the BF16 payload, so the multicast fan-out is cheap.
            mo = [bitcast_f32_i32(x) for x in accs]
            mo = [scrub_b(x) for x in mo]
            for j in cutlass.range_constexpr(2):
                q = mo[4 * j : 4 * j + 4]
                if cutlass.const_expr(use_mc):
                    # Pack adjacent sectors instead of two half-sector stores
                    # with a 32-byte stride between lanes.
                    source_lane = lane // 2 + j * 16
                    q = []
                    for i in cutlass.range_constexpr(4):
                        lo = cute.arch.shuffle_sync(mo[i], source_lane)
                        hi = cute.arch.shuffle_sync(mo[i + 4], source_lane)
                        q.append(lo if lane % 2 == 0 else hi)
                    packed_h = h0 - lane * VEC + lane * (VEC // 2) + j * 32 * (VEC // 2)
                    multimem_st_v4(
                        pb_mc + b_off(gm, slot, my_rank) + packed_h,
                        q[0],
                        q[1],
                        q[2],
                        q[3],
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

        o0, o1, o2, o3 = pack8(accs)
        if cutlass.const_expr(token_owner):
            pb_mc = _pin(cutlass.BFloat16, a_lam_mc + A_BYTES)
            multimem_st_v4(pb_mc + b_off(0, slot, my_rank) + h0, o0, o1, o2, o3)
            fence_proxy_alias()
        else:
            st_global_v4(gOut.iterator + (tok * H_DIM + h0), o0, o1, o2, o3)

    @cute.kernel
    def fused_kernel(
        gIn: cute.Tensor,
        gW: cute.Tensor,
        gIdx: cute.Tensor,
        gNw: cute.Tensor,
        gOut: cute.Tensor,
        start: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        sx, cta, _ = cute.arch.block_idx()
        n_tok = gOut.shape[0]
        active_grid_y = cutlass.min(n_tok, grid_y)
        n_iter = (
            1
            if cutlass.const_expr(token_owner)
            else cute.ceil_div(n_tok, active_grid_y)
        )

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
            tok = cta + it * active_grid_y
            if cutlass.const_expr(token_owner):
                base = tok * ws + start
                tok = base + my_rank
                wait_phase(cta, 0, g0 + it + 1, tidx)
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
                if cutlass.const_expr(comm and not token_owner):
                    advance_idle(g0 + it, cta, h0)
            if cutlass.const_expr(token_owner):
                publish_phase(cta, 1, g0 + it + 1, tidx)

        # every CTA of the cluster has read the counter by now (they all passed
        # the cluster barrier inside the first tile), so one writer is safe.
        if cutlass.const_expr(comm and not token_owner) and tidx == 0 and sx == 0:
            st_gen(_pin(cutlass.Int32, a_gen, 4) + cta, (g0 + n_iter) % NGEN)

    # The owned token's experts are spread over XP CTAs of its slot: CTA d < XP
    # reduces experts d, d + XP, ... (NJ of them). XP == 1 keeps every expert in
    # one CTA and skips the cross-CTA combine; larger XP balances the reduction
    # across the grid at the cost of that combine.
    XP = xpar if single else ws
    NJ = K_EXPERTS // XP
    JG = min(NJ, load_grp)  # experts per batch of peer loads / sentinel probe
    _pj_off = [
        [
            (a_off(0, 0, jj * XP, peers[r]) - a_off(0, 0, j0 * XP, 0)) * 2
            for jj in range(j0, min(j0 + JG, NJ))
            for r in range(len(peers))
        ]
        for j0 in range(0, NJ, JG)
    ]

    @cute.jit
    def rearm_a1(gen: Int32, slot: Int32, d: Int32, h0: Int32):
        # Re-arm generation g+2 of the peer rows this CTA consumed.
        gc = (gen + 2) % NGEN
        pa = _pin(cutlass.BFloat16, a_lam)
        sa = cutlass.Int32(SENT_A_I32)
        for jj in cutlass.range_constexpr(NJ):
            for r in cutlass.range_constexpr(len(peers)):
                st_global_v4(
                    pa + (a_off(gc, slot, d + jj * XP, peers[r]) + h0), sa, sa, sa, sa
                )

    def publish_row(acc, gOut: cute.Tensor, t_own, gm, slot, h0):
        """Write the owned token's output row and multicast it to every rank."""
        o = pack8(acc)
        st_global_v4(gOut.iterator + (t_own * H_DIM + h0), *o)
        # Peers poll this copy, so it must never read as the sentinel. No
        # fence.proxy.alias: this launch never reaches the row through its
        # unicast alias again (the re-arm targets another generation), and
        # waiting for the multicast to land cost ~2.5 us per launch.
        pb_mc = _pin(cutlass.BFloat16, a_lam_mc + A_BYTES)
        multimem_st_v4(pb_mc + b_off(gm, slot, my_rank) + h0, *(scrub_a(w) for w in o))

    @cute.jit
    def rearm_o1(gen: Int32, slot: Int32, owner: Int32, h0: Int32):
        gc = (gen + 2) % NGEN
        pb = _pin(cutlass.BFloat16, a_lam + A_BYTES)
        sa = cutlass.Int32(SENT_A_I32)
        st_global_v4(pb + (b_off(gc, slot, owner) + h0), sa, sa, sa, sa)

    @cute.kernel
    def single_kernel(
        gIn: cute.Tensor,
        gW: cute.Tensor,
        gIdx: cute.Tensor,
        gNw: cute.Tensor,
        gOut: cute.Tensor,
    ):
        """Whole token-owner dataflow in one launch.

        Grid ``(split, slots, ws)``: CTA ``(b, g, d)`` covers H band ``b`` of
        token ``g * ws + d``, which rank ``d`` owns. Each CTA

        A. pushes its token's rows (scrubbed of the sentinel) to rank ``d``;
        B. if ``d < XP``, for the token this rank owns in slot ``g``, reduces
           experts ``d, d + XP, ...`` across ranks in rank order and
           RMS-normalizes them;
        C. with ``XP == 1`` combines the experts in registers; otherwise stores
           the weighted FP32 rows to a local scratch, and the last of the
           ``XP`` CTAs on band ``b`` sums them in expert order. Either way the
           output row is written and multicast;
        D. polls rank ``d``'s multicast row and copies it out.

        Spreading the owner's experts over the slot's CTAs balances the
        reduction across the grid. Both cross-rank hops are flag-free Lamport
        transfers. All CTAs are co-resident and none polls before it pushed,
        and C never waits, so the cross-rank waits cannot cycle.
        """
        tidx, _, _ = cute.arch.thread_idx()
        sx, slot, d = cute.arch.block_idx()
        n_tok = gOut.shape[0]
        num_rows = gIn.shape[0]
        lane = tidx % 32
        warp = tidx // 32

        smem = cutlass_utils.SmemAllocator()
        sred = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(nwarps * NJ),
            byte_alignment=16,
        )
        srstd = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(NJ),
            byte_alignment=16,
        )
        sxch = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout(split * NJ),
            byte_alignment=16,
        )
        slast = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_layout(1),
            byte_alignment=16,
        )
        h0 = sx * band + tidx * VEC
        tok = slot * ws + d
        t_own = slot * ws + my_rank
        gen_ptr = _pin(cutlass.Int32, a_gen, 4) + slot
        gen = ld_gen(gen_ptr)
        gm = gen % NGEN

        if cutlass.const_expr(split > 1):
            if t_own < n_tok and d < XP:
                cute.arch.cluster_arrive_relaxed()

        # ---- A: push this CTA's token band to its owner
        if tok < n_tok:
            for dest in cutlass.range_constexpr(ws):
                if cutlass.const_expr(dest != my_rank):
                    if d == dest:
                        push_rows(gIn, gIdx, tok, dest, gm, slot, h0)

        pa = _pin(cutlass.BFloat16, a_lam)
        pb = _pin(cutlass.BFloat16, a_lam + A_BYTES)
        py = _pin(cutlass.Float32, a_y)
        if t_own < n_tok and d < XP:
            # ---- B: experts d, d + XP, ... of the owned token. Issue every
            # local load before the arrival wait so they overlap it.
            nrm = ld_global_v4(gNw.iterator + h0)
            wgt_lo = ld_global_v4(gW.iterator + t_own * K_EXPERTS)
            wgt_hi = None
            if cutlass.const_expr(expert_weights_fp32):
                wgt_hi = ld_global_v4(gW.iterator + t_own * K_EXPERTS + 4)
            mine = []
            for jj in cutlass.range_constexpr(NJ):
                permuted_idx = cutlass.Int32(gIdx[t_own, d + jj * XP])
                q0 = cutlass.Int32(0)
                q1 = cutlass.Int32(0)
                q2 = cutlass.Int32(0)
                q3 = cutlass.Int32(0)
                if permuted_idx >= 0 and permuted_idx < num_rows:
                    q0, q1, q2, q3 = ld_global_v4(
                        gIn.iterator + (permuted_idx * row_stride + h0)
                    )
                mine.append((scrub_a(q0), scrub_a(q1), scrub_a(q2), scrub_a(q3)))

            pre = {}
            for gi in cutlass.range_constexpr(len(_pj_off)):
                j0 = gi * JG
                anchor = pa + (a_off(gm, slot, d + j0 * XP, 0) + h0)
                ok = lamport_probe_a(anchor, _pj_off[gi])
                while ok == 0:
                    ok = lamport_probe_a(anchor, _pj_off[gi])
                for jj in cutlass.range_constexpr(j0, min(j0 + JG, NJ)):
                    for r in cutlass.range_constexpr(len(peers)):
                        pre[(jj, peers[r])] = ld_global_v4_volatile(
                            pa + (a_off(gm, slot, d + jj * XP, peers[r]) + h0)
                        )
            # Statements, not a conditional expression: the DSL rewrites
            # `a if c else b` inside kernels, which broke this rank-order pick.
            packed = []
            for jj in cutlass.range_constexpr(NJ):
                rows = []
                for src in cutlass.range_constexpr(ws):
                    if cutlass.const_expr(src == my_rank):
                        rows.append(mine[jj])
                    else:
                        rows.append(pre[(jj, src)])
                packed.append(pack8(sum_ranks(rows)))

            nw, ew = decode_gains(nrm, wgt_lo, wgt_hi)
            ssq = [warp_ssq(packed[jj], nw) for jj in range(NJ)]
            if lane == 0:
                for jj in cutlass.range_constexpr(NJ):
                    sred[jj * nwarps + warp] = ssq[jj]
            cute.arch.barrier()
            rms_finish(NJ, sred, srstd, sxch, tidx, sx)

            if cutlass.const_expr(XP == 1):
                # every expert is in registers: combine in expert order here
                acc = [cutlass.Float32(0.0) for _ in range(8)]
                for jj in cutlass.range_constexpr(NJ):
                    y = norm_row(packed[jj], srstd[jj], nw)
                    for i in cutlass.range_constexpr(8):
                        acc[i] = acc[i] + y[i] * ew[jj]
                publish_row(acc, gOut, t_own, gm, slot, h0)
            else:
                for jj in cutlass.range_constexpr(NJ):
                    k = d + jj * XP
                    wk = ew[0]
                    for kk in cutlass.range_constexpr(1, K_EXPERTS):
                        if k == kk:
                            wk = ew[kk]
                    y = [
                        bitcast_f32_i32(v * wk)
                        for v in norm_row(packed[jj], srstd[jj], nw)
                    ]
                    yp = py + ((slot * K_EXPERTS + k) * H_DIM + h0)
                    st_global_v4(yp, y[0], y[1], y[2], y[3])
                    st_global_v4(yp + 4, y[4], y[5], y[6], y[7])

                # ---- C: the slot's last expert CTA on this band combines them
                cute.arch.fence_acq_rel_gpu()
                cute.arch.barrier()
                if tidx == 0:
                    cnt = _pin(cutlass.Int32, a_ycnt, 4) + (slot * split + sx)
                    old = atomic_add_acq_rel(cnt, cutlass.Int32(1))
                    last = cutlass.Int32(0)
                    if old == XP - 1:
                        st_gen(cnt, cutlass.Int32(0))
                        last = cutlass.Int32(1)
                    slast[0] = last
                cute.arch.barrier()
                if slast[0] == 1:
                    cute.arch.fence_acq_rel_gpu()
                    acc = [cutlass.Float32(0.0) for _ in range(8)]
                    for kk in cutlass.range_constexpr(K_EXPERTS):
                        yrow = py + ((slot * K_EXPERTS + kk) * H_DIM + h0)
                        ya = ld_global_v4_volatile(yrow)
                        yb = ld_global_v4_volatile(yrow + 4)
                        for q in cutlass.range_constexpr(4):
                            acc[q] = acc[q] + bitcast_i32_f32(ya[q])
                            acc[4 + q] = acc[4 + q] + bitcast_i32_f32(yb[q])
                    publish_row(acc, gOut, t_own, gm, slot, h0)
        # Re-arm generation g+2 of this CTA's incoming rows on every launch,
        # including launches where this rank owns no token in the slot: the
        # slot's generation advances regardless, and a buffer left holding an
        # older launch's rows would pass the next arrival probe immediately.
        if d < XP:
            rearm_a1(gen, slot, d, h0)

        # ---- D: copy the row rank d multicast for this CTA's token
        if tok < n_tok and d != my_rank:
            arow = pb + (b_off(gm, slot, d) + h0)
            okd = lamport_probe_a(arow, [0])
            while okd == 0:
                okd = lamport_probe_a(arow, [0])
            qd = ld_global_v4_volatile(arow)
            st_global_v4(gOut.iterator + tok * H_DIM + h0, qd[0], qd[1], qd[2], qd[3])
        # Only polled rows need re-arming: this rank's own row receives just its
        # own multicast and is never polled here.
        if d != my_rank:
            rearm_o1(gen, slot, d, h0)
        cute.arch.barrier()

        # the slot's last CTA publishes the next generation
        if tidx == 0:
            done = _pin(cutlass.Int32, a_done, 4) + slot
            old = atomic_add_acq_rel(done, cutlass.Int32(1))
            if old == ws * split - 1:
                st_gen(done, cutlass.Int32(0))
                st_gen(gen_ptr, (gen + 1) % NGEN)

    @cute.kernel
    def stage_kernel(gIn: cute.Tensor, gIdx: cute.Tensor, start: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        sx, cta, _ = cute.arch.block_idx()
        slot = cta // ws
        gen = ld_gen(_pin(cutlass.Int32, a_gen, 4) + slot) + 1
        scatter_tokens(gIn, gIdx, start + cta, slot, gen, tidx, sx * band + tidx * VEC)

    @cute.kernel
    def copy_kernel(gOut: cute.Tensor, start: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        sx, cta, _ = cute.arch.block_idx()
        gen = ld_gen(_pin(cutlass.Int32, a_gen, 4) + cta) + 1
        if cutlass.const_expr(split > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        copy_tokens(gOut, start + cta * ws, cta, gen, tidx, sx * band + tidx * VEC)
        if cutlass.const_expr(split > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        if tidx == 0 and sx == 0:
            st_gen(_pin(cutlass.Int32, a_gen, 4) + cta, gen)

    @cute.jit
    def host(
        gIn: cute.Tensor,
        gW: cute.Tensor,
        gIdx: cute.Tensor,
        gNw: cute.Tensor,
        gOut: cute.Tensor,
        stream: _drv.CUstream,
    ):
        n_tok = gOut.shape[0]
        tiles = cute.ceil_div(n_tok, ws)
        if cutlass.const_expr(single and split > 1):
            single_kernel(gIn, gW, gIdx, gNw, gOut).launch(
                grid=(split, tiles, ws),
                block=(nthr, 1, 1),
                cluster=(split, 1, 1),
                min_blocks_per_mp=1,
                cooperative=True,
                stream=stream,
            )
        elif cutlass.const_expr(single):
            single_kernel(gIn, gW, gIdx, gNw, gOut).launch(
                grid=(1, tiles, ws),
                block=(nthr, 1, 1),
                min_blocks_per_mp=1,
                cooperative=True,
                stream=stream,
            )
        elif cutlass.const_expr(token_owner):
            for chunk in cutlass.range(cute.ceil_div(n_tok, grid_y * ws)):
                start = chunk * grid_y * ws
                groups = cutlass.min(grid_y, cute.ceil_div(n_tok - start, ws))
                stage_kernel(gIn, gIdx, cutlass.Int32(start)).launch(
                    grid=(split, groups * ws, 1),
                    block=(nthr, 1, 1),
                    stream=stream,
                )
                if cutlass.const_expr(split > 1):
                    fused_kernel(gIn, gW, gIdx, gNw, gOut, cutlass.Int32(start)).launch(
                        grid=(split, groups, 1),
                        block=(nthr, 1, 1),
                        cluster=(split, 1, 1),
                        stream=stream,
                    )
                    copy_kernel(gOut, cutlass.Int32(start)).launch(
                        grid=(split, groups, 1),
                        block=(nthr, 1, 1),
                        cluster=(split, 1, 1),
                        stream=stream,
                    )
                else:
                    fused_kernel(gIn, gW, gIdx, gNw, gOut, cutlass.Int32(start)).launch(
                        grid=(split, groups, 1),
                        block=(nthr, 1, 1),
                        stream=stream,
                    )
                    copy_kernel(gOut, cutlass.Int32(start)).launch(
                        grid=(split, groups, 1),
                        block=(nthr, 1, 1),
                        stream=stream,
                    )
        elif cutlass.const_expr(split > 1):
            fused_kernel(gIn, gW, gIdx, gNw, gOut, cutlass.Int32(0)).launch(
                grid=(split, cutlass.min(n_tok, grid_y), 1),
                block=(nthr, 1, 1),
                cluster=(split, 1, 1),
                min_blocks_per_mp=1,
                cooperative=True,
                stream=stream,
            )
        else:
            fused_kernel(gIn, gW, gIdx, gNw, gOut, cutlass.Int32(0)).launch(
                grid=(1, cutlass.min(n_tok, grid_y), 1),
                block=(nthr, 1, 1),
                min_blocks_per_mp=1,
                cooperative=True,
                stream=stream,
            )

    return host


def _token_crossover(nsm):
    # Below about 64 / P tokens the expert-owner path is faster: each owner
    # reduces only K / P experts, so its single CTA finishes sooner than a token
    # owner's all-expert reduction. Above it, the token-owner path moves less
    # data. Measured on GB300 (TP4 crossover 16, TP8 crossover 8).
    return max(1, 64 // _WS.ws)


def _select_path(n_tok, nsm):
    if n_tok < _token_crossover(nsm):
        return "expert"
    # One CTA per token, all co-resident: a single launch with no kernel-boundary
    # ordering. Larger counts chunk through the three-launch pipeline.
    if _single_ctas(n_tok) > nsm:
        return "token"
    return "token1"


def _single_xpar(n_tok):
    """CTAs that share one owned token's experts in the single launch."""
    # One CTA per owned token avoids the cross-CTA combine. It wins while all
    # of a token's peer rows fit one load batch and the per-rank token count is
    # small (TP4: T <= 48 on GB300); otherwise spreading the experts over the
    # slot balances the reduction. Measured on 4x/8x GB300, H=4096, top-k 8.
    ws = _WS.ws
    if _peer_batch(ws) == K_EXPERTS and n_tok <= 12 * ws:
        return 1
    return ws


def _single_push(n_tok):
    """Expert rows a CTA loads before storing them in the single launch."""
    # Two rows per batch overlap most of the load latency. When one CTA owns
    # all of a token's experts at mid T, burstier pushes delay that CTA's
    # inputs, so rows go out one at a time (GB300, TP4 T=32..48).
    if _single_xpar(n_tok) == 1 and n_tok >= 32:
        return 1
    return 2


def _single_ctas(n_tok):
    """Token-sized CTAs of the single launch: one per token, padded to full slots."""
    return (n_tok + _WS.ws - 1) // _WS.ws * _WS.ws


def _config(n_tok):
    """Compile-time ``(path, rows, split, xpar, push)`` for ``n_tok`` tokens.

    Grids are sized for a power-of-two row bucket, so nearby token counts share
    one program; the kernels read the real count from the tensor shapes. The
    path and its tuning follow the real count, which they were measured on.
    """
    ws, nsm = _WS.ws, _WS.nsm
    path = _select_path(n_tok, nsm)
    rows = 1 << (n_tok - 1).bit_length()
    if path == "expert":
        return path, rows, _pick_split(n_tok, nsm), 0, 0
    if path == "token1":
        # The bucket must keep every padded slot of the single launch, which
        # the real count fits, inside the SM count and the workspace slots.
        rows = min(rows, nsm // ws * ws)
        split = _pick_split(_single_ctas(n_tok), nsm)
        return path, rows, split, _single_xpar(n_tok), _single_push(n_tok)
    return path, min(rows, 1 << (nsm - 1).bit_length()), 1, 0, 0


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
    scratch: tuple[torch.Tensor, torch.Tensor] | None = None


_WS = _Ws()


class _WorkspaceAddresses(TypedDict):
    lam_uc: int
    lam_mc: int
    lam_peers: list[int]
    gen: int
    token_uc: int
    token_mc: int
    token_peers: list[int]
    token_gen: int
    token_slots: int
    tl_done: int
    tl_gen: int
    tl_uc: int
    tl_mc: int
    tl_peers: list[int]
    tl_slots: int
    tl_y: int
    tl_ycnt: int


_PROGRAMS: dict[tuple, "_Program"] = {}


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
    capability = (props.major, props.minor)
    local_supported = (
        capability in _COMPUTE_CAPABILITIES
        and is_cute_dsl_arch_supported(*capability)
        and props.multi_processor_count <= NSLOT
    )
    if not collective_support(local_supported, group, device):
        if not local_supported:
            logger.warning(
                "TP MoE finalize requires SM100/SM103/SM107, a CuTe DSL that "
                "targets it, and at most %d SMs; got SM%d%d with %d SMs.",
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

    sm_count = torch.tensor(
        props.multi_processor_count, dtype=torch.int32, device=device
    )
    dist.all_reduce(sm_count, op=dist.ReduceOp.MIN, group=group)
    _WS.nsm = int(sm_count.item())
    _WS.dev_ord = torch.cuda.current_device()
    _WS.ws = world_size
    _WS.rank = dist.get_rank(group)

    import torch.distributed._symmetric_memory as symm_mem

    ko = K_EXPERTS // _WS.ws
    a_i32 = NGEN * NSLOT * ko * _WS.ws * H_DIM // 2  # BF16 region, in int32 words
    b_i32 = NGEN * NSLOT * _WS.ws * H_DIM  # FP32 region, in int32 words
    token_i32 = _WS.nsm * (K_EXPERTS + 1) * _WS.ws * H_DIM // 2
    token_flags_i32 = _WS.nsm * 2 * _WS.ws
    # The single launch only runs while its padded slots fit the SM count.
    tl_slots = _WS.nsm // _WS.ws
    tl_i32 = NGEN * tl_slots * (K_EXPERTS + 1) * _WS.ws * H_DIM // 2
    allocation_flag = torch.empty((1,), dtype=torch.int32, device=device)
    lam = None
    generation = None
    scratch = None
    allocation_error = None
    try:
        lam = symm_mem.empty(
            (a_i32 + b_i32 + token_i32 + token_flags_i32 + tl_i32,),
            dtype=torch.int32,
            device=device,
        )
        generation = torch.zeros((4 * NSLOT,), dtype=torch.int32, device=device)
        # Local scratch of the single launch: weighted FP32 expert rows per
        # slot and one arrival counter per (slot, H band).
        scratch = (
            torch.empty(
                (tl_slots * K_EXPERTS * H_DIM,), dtype=torch.float32, device=device
            ),
            torch.zeros((tl_slots * _MAX_SPLIT,), dtype=torch.int32, device=device),
        )
    except RuntimeError as error:
        allocation_error = error
    if not collective_support(
        allocation_error is None, group, device, flag=allocation_flag
    ):
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
    assert lam is not None and generation is not None and scratch is not None
    lam[:a_i32].fill_(SENT_A_I32)
    lam[a_i32 : a_i32 + b_i32].fill_(SENT_B_I32)
    lam[a_i32 + b_i32 + token_i32 : a_i32 + b_i32 + token_i32 + token_flags_i32].zero_()
    tl_base = a_i32 + b_i32 + token_i32 + token_flags_i32
    lam[tl_base:].fill_(SENT_A_I32)
    _WS.gen = generation
    _WS.scratch = scratch
    tl_y, tl_ycnt = scratch
    torch.cuda.synchronize()
    hdl = symm_mem.rendezvous(lam, group=group)
    _WS.lam, _WS.hdl = lam, hdl

    mc = int(getattr(hdl, "multicast_ptr", 0) or 0)
    if not collective_support(mc != 0, group, device):
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
            hdl.get_buffer(r, (lam.numel(),), torch.int32).data_ptr()
            for r in range(_WS.ws)
        ]
    _WS.addrs = {
        "lam_uc": lam.data_ptr(),
        "lam_mc": mc,
        "lam_peers": peers,
        "gen": generation.data_ptr(),
        "token_uc": lam.data_ptr() + (a_i32 + b_i32) * 4,
        "token_mc": mc + (a_i32 + b_i32) * 4,
        "token_peers": [p + (a_i32 + b_i32) * 4 for p in peers],
        "token_gen": generation.data_ptr() + NSLOT * 4,
        "tl_done": generation.data_ptr() + 2 * NSLOT * 4,
        # The single launch keeps its own generations: the multi-launch token
        # path advances token_gen per slot, which would desynchronize the
        # single launch's Lamport buffers from their re-arm schedule.
        "tl_gen": generation.data_ptr() + 3 * NSLOT * 4,
        "tl_uc": lam.data_ptr() + tl_base * 4,
        "tl_mc": mc + tl_base * 4,
        "tl_peers": [p + tl_base * 4 for p in peers],
        "tl_slots": tl_slots,
        "tl_y": tl_y.data_ptr(),
        "tl_ycnt": tl_ycnt.data_ptr(),
        "token_slots": _WS.nsm,
    }
    torch.accelerator.synchronize(device)
    dist.barrier(group=group)
    _WS.available = True
    _WS.initialized = True
    return True


class _Program:
    __slots__ = ("fn", "ffi", "path")

    def __init__(self, fn, ffi, path):
        self.fn = fn
        self.ffi = ffi
        # The transport actually compiled; differs from the key on fallback.
        self.path = path

    def __call__(self, a, b, c, d, e, stream):
        if self.ffi:
            self.fn(a, b, c, d, e, stream)
        else:
            self.fn(
                from_dlpack(a, assumed_align=16).mark_layout_dynamic(
                    leading_dim=a.ndim - 1
                ),
                from_dlpack(b, assumed_align=16).mark_layout_dynamic(
                    leading_dim=b.ndim - 1
                ),
                from_dlpack(c, assumed_align=16).mark_layout_dynamic(
                    leading_dim=c.ndim - 1
                ),
                from_dlpack(d, assumed_align=16).mark_layout_dynamic(
                    leading_dim=d.ndim - 1
                ),
                from_dlpack(e, assumed_align=16).mark_layout_dynamic(
                    leading_dim=e.ndim - 1
                ),
                stream,
            )


def _checked(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA driver call failed: {result[0]}")
    return result[1] if len(result) > 1 else None


def _grid_fits(compiled, *, split, groups, nthr, smem, stream):
    """Whether the cooperative grid of a single-kernel program fits the GPU.

    TVM-FFI programs expose no CUDA function handle, so the kept cubin is
    loaded into the current context for the occupancy query only; the launch
    still goes through ``compiled``.
    """
    (name,) = compiled.kernel_info
    module = _checked(_drv.cuModuleLoadData(compiled.artifacts.CUBIN))
    try:
        fn = _checked(_drv.cuModuleGetFunction(module, name.encode()))
        if split == 1:
            active = _checked(
                _drv.cuOccupancyMaxActiveBlocksPerMultiprocessor(fn, nthr, smem)
            )
            return active * _WS.nsm >= groups
        _checked(
            _drv.cuFuncSetAttribute(
                fn,
                _drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
                1,
            )
        )
        cluster = _drv.CUlaunchAttribute()
        cluster.id = _drv.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        cluster.value.clusterDim.x = split
        cluster.value.clusterDim.y = 1
        cluster.value.clusterDim.z = 1
        config = _drv.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = split, groups, 1
        config.blockDimX, config.blockDimY, config.blockDimZ = nthr, 1, 1
        config.sharedMemBytes = smem
        config.hStream = stream
        config.attrs = [cluster]
        config.numAttrs = 1
        return _checked(_drv.cuOccupancyMaxActiveClusters(fn, config)) >= groups
    finally:
        _drv.cuModuleUnload(module)


def _compile(host, a, b, c, d, e, stream, options, path):
    try:
        args = [
            from_dlpack(x, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
                leading_dim=x.ndim - 1
            )
            for x in (a, b, c, d, e)
        ]
        compiled = cute.compile(
            host, *args, stream, options=f"--enable-tvm-ffi {options}"
        )
        return _Program(compiled, True, path)
    except Exception as error:
        logger.warning(
            "TVM-FFI compilation failed for TP MoE finalize; using the "
            "process-local CuTe callable: %s",
            error,
        )
        args = [
            from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=x.ndim - 1)
            for x in (a, b, c, d, e)
        ]
        return _Program(cute.compile(host, *args, stream, options=options), False, path)


def _build(key, a, b, c, d, e, stream):
    # The generated program embeds process-local symmetric-memory and multicast
    # addresses. It must therefore remain in the process cache rather than the
    # persistent CuTe-DSL cache, whose objects are reusable across processes.
    path, rows, split, xpar, push, row_stride, expert_weights_fp32, mode, eps, gain = (
        key
    )
    tiles = (rows + _WS.ws - 1) // _WS.ws if path != "expert" else rows
    nthr = H_DIM // (VEC * split)
    grid_y = min(tiles, _WS.nsm // split)
    addrs = _WS.addrs
    if addrs is None:
        raise RuntimeError("TP MoE finalize workspace addresses are unavailable")
    # Every rank votes once per specialization, so a local failure (compile,
    # occupancy query) becomes a group-wide fallback or error instead of
    # leaving the other ranks waiting in a collective or a kernel.
    program = None
    try:
        host = _make_program(
            row_stride,
            grid_y,
            split,
            nthr,
            mode,
            eps,
            gain,
            expert_weights_fp32,
            _WS.ws,
            _WS.use_mc,
            addrs,
            _WS.rank,
            token_owner=path != "expert",
            single=path == "token1",
            xpar=xpar or None,
            push=push or None,
        )
        if path == "token":
            program = _compile(host, a, b, c, d, e, stream, "", path)
            ok = True
        else:
            with tempfile.TemporaryDirectory() as dump_dir:
                program = _compile(
                    host,
                    a,
                    b,
                    c,
                    d,
                    e,
                    stream,
                    f"--keep-cubin --dump-dir={dump_dir}",
                    path,
                )
            groups = tiles * _WS.ws if path == "token1" else grid_y
            experts = K_EXPERTS // (xpar if path == "token1" else _WS.ws)
            # Conservative padding for the reduction arrays and single-path flag.
            smem = 4 * experts * (nthr // 32 + 1 + split) + 64
            ok = _grid_fits(
                program.fn,
                split=split,
                groups=groups,
                nthr=nthr,
                smem=smem,
                stream=stream,
            )
    except Exception as error:
        logger.warning("TP MoE finalize %s path failed on this rank: %s", path, error)
        ok = False
    device = torch.device("cuda", _WS.dev_ord)
    if collective_support(ok, _WS.group, device):
        assert program is not None
        return program
    if path == "token":
        raise RuntimeError("TP MoE finalize compilation failed on at least one rank")
    # All ranks choose the staged transport together. Kernel boundaries remove
    # the requirement that the full grid be resident at once.
    logger.warning(
        "TP MoE finalize %s path is unavailable for %d padded tokens; "
        "using the staged token path.",
        path,
        rows,
    )
    return _build(("token", rows, 1, 0, 0, *key[5:]), a, b, c, d, e, stream)


def _program_key(
    n_tok, gemm2_permuted, expert_weights, weight_before_norm, eps, gain_center
):
    return (
        *_config(n_tok),
        gemm2_permuted.stride(0),
        int(expert_weights.dtype == torch.float32),
        int(weight_before_norm),
        float(eps),
        float(gain_center),
    )


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
    key = _program_key(
        n_tok, gemm2_permuted, expert_weights, weight_before_norm, eps, gain_center
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


def supports_tp_moe_finalize_rmsnorm(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_weight: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the TP SM100/SM103/SM107 kernel contract."""
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
    """TP MoE finalize plus per-expert RMSNorm, for hidden size 4096 and top-k 8.

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

    Parameters
    ----------
    gemm2_permuted : torch.Tensor
        BF16 GEMM2 output of shape ``(rows, >= 4096)`` in physical expert order.
        Rows may be padded, but the last dimension must have unit stride and the
        row stride must be a multiple of 8 elements.
    expert_weights : torch.Tensor
        BF16 or FP32 routing weights of shape ``(num_tokens, 8)``.
    expanded_idx_to_permuted_idx : torch.Tensor
        Int32 tensor of shape ``(num_tokens, 8)`` mapping each (token, top-k slot)
        to a row of ``gemm2_permuted``.
    norm_weight : torch.Tensor
        BF16 RMSNorm weight of shape ``(4096,)``.
    output : torch.Tensor
        BF16 destination of shape ``(num_tokens, 4096)``.
    eps : float
        Epsilon added to the RMS denominator.
    gain_center : float
        Constant added to ``norm_weight`` in FP32.
    weight_before_norm : bool
        Whether ``norm_weight`` scales the activations before the RMS reduction
        instead of after normalization.

    All tensors must be contiguous (except the padded ``gemm2_permuted``),
    16-byte aligned, and on the device the workspace was initialized on.
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
