"""Scatter of the pre-quantized input rows for the per-token MoE. The input arrives quantized (FlashInfer's
per-token NVFP4 activation format: packed E2M1 [M, H / 2], linear E4M3 block scales [M, H / 16], row scales [M]),
so the kernels only copy it.

``Scatter`` (short batches, PDL): 64 threads per token load the token's FP4 row (16-byte chunks tp, tp + 64, ..),
its block-scale words (4 scales each, words tp, tp + 64, ..) and its row scale before the PDL wait, then store
them to the token's ``topk`` routed rows: FP4 rows [rows, H / 2] (``st.global.v4``), block scales packed per
``block_m``-row tile in the 128x4 layout (one word per 4 blocks), row scales [rows].

Short batches place the routes themselves (``rank_place``): ``SelectExperts`` leaves each route's rank
within its expert in ``inverse`` and the experts' route counts in ``ecount``. After
the wait, warp 0 of the CTA scans the counts rounded up to tiles (experts in index order) into shared
``base[E]`` (CTA 0's warp 0 also writes each tile's expert and limit and the row count), and each route's
row is base[expert] + rank. The dependents read routing before their own wait (the gate/up GEMM its row
count and tile experts for the weight prefetch), so the rows and tiles are written and fenced before
``launch_dependents``.
Several tokens may share a CTA (``tokens_per_cta`` groups of 64 threads): the MoE uses about 128 CTAs.
With ``fused`` (``CuteMoE(use_fused_finalize=True)``) the scatter also stores each routed row's token and routing
weight and zeroes the tokens' output rows, into which the down GEMM then adds the routes (``GroupedGemm(fused_out)``).
"""

import cutlass
import cutlass.cute as cute
import cutlass.memory
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir

from .gemm import _MAXNREG, use_pdl

GMEM = cute.AddressSpace.gmem


def _v4():
    return ir.VectorType.get([4], Uint32.mlir_type)


def ld_v4(addr):
    """16 bytes at a 16-byte aligned global address (``ld.global.v4.b32``), as a vector value."""
    return cute.arch.load(cute.make_ptr(Uint32, addr, GMEM, assumed_align=16), _v4())


def st_v4(addr, v):
    cute.arch.store(cute.make_ptr(Uint32, addr, GMEM, assumed_align=16), v)


def ld_u32(addr):
    return cute.arch.load(cute.make_ptr(Uint32, addr, GMEM, assumed_align=4), Uint32)


def st_u32(addr, v):
    cute.arch.store(cute.make_ptr(Uint32, addr, GMEM, assumed_align=4), Uint32(v))


@cute.jit
def store_chunks(
    store: cutlass.Constexpr,
    base,
    vals,
    tp,
    TH: cutlass.Constexpr,
    n: cutlass.Constexpr,
    stride: cutlass.Constexpr,
):
    """Thread tp of TH stores ``vals[j]`` at ``base + stride * (tp + TH j)`` for its items tp + TH j < n (``store``:
    st_v4 or st_u32; only the last round is predicated, where TH does not divide n): a row copied in chunks."""
    for j in cutlass.range_constexpr(len(vals)):
        if cutlass.const_expr((j + 1) * TH > n):
            if tp + TH * j < n:
                store(base + stride * (tp + TH * j), vals[j])
        else:
            store(base + stride * (tp + TH * j), vals[j])


@cute.jit
def zero_row(
    y_ptr: cute.Pointer, token, tp, H: cutlass.Constexpr, TH: cutlass.Constexpr
):
    """Thread tp of TH: zero the token's BF16 output row [H] (16-byte chunks tp + TH j)."""
    NC = H // 8
    z = cute.make_rmem_tensor((4,), Uint32)
    z.fill(0)
    zv = z.load().ir_value()
    store_chunks(
        st_v4,
        y_ptr.toint() + Int64(token) * (H * 2),
        [zv] * ((NC + TH - 1) // TH),
        tp,
        TH,
        NC,
        16,
    )


@cute.jit
def route_info(
    w_ptr: cute.Pointer,
    rtok_ptr: cute.Pointer,
    rwt_ptr: cute.Pointer,
    token,
    slot: cutlass.Constexpr,
    row,
    T: cutlass.Constexpr,
):
    """The fused finalize's per-row route: the row's token (Int32 [rows]) and routing weight (FP32 [rows])."""
    wv = cute.arch.load(
        cute.make_ptr(
            Float32,
            w_ptr.toint() + (Int64(token) * T + slot) * 4,
            GMEM,
            assumed_align=4,
        ),
        Float32,
    )
    cute.arch.store(
        cute.make_ptr(Int32, rtok_ptr.toint() + Int64(row) * 4, GMEM, assumed_align=4),
        Int32(token),
    )
    cute.arch.store(
        cute.make_ptr(Float32, rwt_ptr.toint() + Int64(row) * 4, GMEM, assumed_align=4),
        wv,
    )


def load_row(
    q_in,
    sf_in,
    gs_in,
    token,
    tp,
    K: cutlass.Constexpr,
    TH: cutlass.Constexpr,
    rows=True,
    row_scale=True,
):
    """Thread tp of TH: chunks tp + TH j of the token's FP4 row (16 bytes, clamped to the last chunk), words
    tp + TH j of its linear block scales (clamped) and its row scale (``row_scale`` False, per-tensor input: none
    loaded, 1)."""
    NQ, NS = K // 32, K // 64
    qrow = q_in.toint() + Int64(token) * (K // 2)
    srow = sf_in.toint() + Int64(token) * (K // 16)
    qv = (
        [
            ld_v4(
                qrow
                + 16
                * (
                    cutlass.min(tp + TH * j, NQ - 1)
                    if (j + 1) * TH > NQ
                    else tp + TH * j
                )
            )
            for j in range((NQ + TH - 1) // TH)
        ]
        if rows
        else []
    )
    sw = [
        ld_u32(
            srow
            + 4
            * (cutlass.min(tp + TH * j, NS - 1) if (j + 1) * TH > NS else tp + TH * j)
        )
        for j in range((NS + TH - 1) // TH)
    ]
    if not row_scale:
        return qv, sw, Float32(1.0)
    gs = cute.arch.load(
        cute.make_ptr(Float32, gs_in.toint() + Int64(token) * 4, GMEM, assumed_align=4),
        Float32,
    )
    return qv, sw, gs


@cute.jit
def _scatter_store(
    qv,
    sw,
    gs,
    row,
    slot: cutlass.Constexpr,
    token,
    t,
    qb,
    sb,
    gb,
    w_ptr,
    rtok_ptr,
    rwt_ptr,
    K: cutlass.Constexpr,
    T: cutlass.Constexpr,
    BM: cutlass.Constexpr,
    rowtok: cutlass.Constexpr,
    sf_linear: cutlass.Constexpr,
    fused: cutlass.Constexpr,
    row_scale: cutlass.Constexpr = True,
):
    """Scatter: the token's copies (thread t of 64) into its routed row ``row`` of slot ``slot`` (the row scale only
    with ``row_scale``)."""
    NQ, NS = K // 32, K // 64
    store_chunks(st_v4, qb + Int64(row) * (K // 2), qv, t, 64, NQ, 16)
    if cutlass.const_expr(rowtok):
        if t == 0:
            st_u32(sb + Int64(row) * 4, Uint32(token))
    elif cutlass.const_expr(sf_linear):
        store_chunks(st_u32, sb + Int64(row) * (K // 16), sw, t, 64, NS, 4)
    else:
        # 128x4-packed layout of BM-row tiles: word w (blocks 4 w .. 4 w + 3) of tile row srow at
        # srow / 128 * (K / 64 * 512) + w * 512 + srow % 32 * 16 + srow % 128 / 32 * 4
        srow = row // BM * 128 + row % BM
        soff = sb + Int64(
            srow // 128 * (NS * 512) + srow % 32 * 16 + srow % 128 // 32 * 4
        )
        store_chunks(st_u32, soff, sw, t, 64, NS, 512)
    if cutlass.const_expr(row_scale):
        if t == 0:
            cute.arch.store(
                cute.make_ptr(Float32, gb + Int64(row) * 4, GMEM, assumed_align=4), gs
            )
    if cutlass.const_expr(fused):
        if t == 1:  # the row's token and routing weight
            wv = cute.arch.load(
                cute.make_ptr(
                    Float32,
                    w_ptr.toint() + (Int64(token) * T + slot) * 4,
                    GMEM,
                    assumed_align=4,
                ),
                Float32,
            )
            cute.arch.store(
                cute.make_ptr(
                    Int32, rtok_ptr.toint() + Int64(row) * 4, GMEM, assumed_align=4
                ),
                Int32(token),
            )
            cute.arch.store(
                cute.make_ptr(
                    Float32, rwt_ptr.toint() + Int64(row) * 4, GMEM, assumed_align=4
                ),
                wv,
            )


class Scatter:
    """Store each token's pre-quantized row to its routed rows (short batches; long batches: ``ScatterChunks``).

    Args:
        k: Hidden size, a multiple of 64.
        topk: Routes per token.
        block_m: Rows per expert tile (the 128x4 block-scale packing).
        num_experts: E.
        tokens_per_cta: Tokens per CTA of ``64 * tokens_per_cta`` threads; the grid is
            ceil(tokens / tokens_per_cta) and missing tokens of the last CTA store nothing.
        rowtok: The gate/up GEMM loads the block scales from the input (``GroupedGemm(x_sf_gather=1)``): instead of
            the scales, store each route's token at ``sf_ptr`` (Int32 [rows]).
        sf_linear: Store the block scales row-major [rows, K / 16] (``GroupedGemm(x_sf_gather=3)``) instead of packed
            per tile.
        fused: For the down GEMM's fused finalize: after the wait, also store each routed row's token and routing weight
            (Int32 / FP32 [rows] at rtok_ptr / rwt_ptr; the weights [tokens, topk] at w_ptr) and zero the token's
            output row (BF16 [tokens, K] at y_ptr). Otherwise those pointers are not used.
        base: The rank's first expert (expert parallelism): local expert = id - base. Routes whose local expert is
            outside [0, E) get no row (inverse -1, written by SelectExperts and kept) and are not copied.
        row_scales: Copy the tokens' row scales (gs_in -> gs_ptr); False for per-tensor input
            (``moe.CuteMoE(per_token_scale=False)``: none, the gate/up's ``unit_a_gs``).
    """

    def __init__(
        self,
        k,
        topk,
        block_m,
        *,
        num_experts=128,
        tokens_per_cta=1,
        rowtok=False,
        sf_linear=False,
        fused=False,
        early=False,
        base=0,
        row_scales=True,
    ):
        assert k % 64 == 0, f"hidden size {k} must be a multiple of 64"
        # early (block_m 64): the gate/up places the tiles itself (GroupedGemm place_tiles 2): trigger right after the
        # wait, write no tile table (the rows and inverse as without early)
        self.early = early
        # PDL (griddepcontrol.wait / launch_dependents) always: the short batches place the routes from their ranks
        self.k, self.topk, self.bm, self.launch_pdl, self.E = (
            k,
            topk,
            block_m,
            True,
            num_experts,
        )
        self.tpc = tokens_per_cta
        self.threads = 64 * self.tpc
        self.rowtok, self.sf_linear, self.fused = rowtok, sf_linear, fused
        self.base = base
        self.row_scales = row_scales

    @cute.jit
    def __call__(
        self,
        q_in: cute.Pointer,
        sf_in: cute.Pointer,
        gs_in: cute.Pointer,
        inv_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        experts_ptr: cute.Pointer,
        limits_ptr: cute.Pointer,
        count_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        rtok_ptr: cute.Pointer,
        rwt_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        tokens: Int32,
        stream,
    ):
        @cute.struct
        class Storage:
            # first row of each expert's routes
            sbase: cute.struct.MemRange[Int32, self.E]

        self.storage = Storage
        n = tokens * self.topk
        inv = cute.make_tensor(inv_ptr, cute.make_layout(n))
        ids = cute.make_tensor(ids_ptr, cute.make_layout(n))
        ecount = cute.make_tensor(
            partial_ptr, cute.make_layout(self.E)
        )  # routes per expert
        self.kernel(
            q_in,
            sf_in,
            gs_in,
            inv,
            q_ptr,
            sf_ptr,
            gs_ptr,
            ids,
            ecount,
            experts_ptr,
            limits_ptr,
            count_ptr,
            w_ptr,
            rtok_ptr,
            rwt_ptr,
            y_ptr,
            tokens,
        ).launch(
            grid=[(tokens + self.tpc - 1) // self.tpc, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=use_pdl(self),
        )

    @cute.kernel
    def kernel(
        self,
        q_in: cute.Pointer,
        sf_in: cute.Pointer,
        gs_in: cute.Pointer,
        inv: cute.Tensor,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        ids: cute.Tensor,
        ecount: cute.Tensor,
        experts_ptr: cute.Pointer,
        limits_ptr: cute.Pointer,
        count_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        rtok_ptr: cute.Pointer,
        rwt_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        ntokens: Int32,
    ):
        K, T, BM = self.k, self.topk, self.bm
        bidx, _, _ = cute.arch.block_idx()
        tall, _, _ = cute.arch.thread_idx()
        grp = tall // 64  # token group of this thread (tokens_per_cta)
        t = tall % 64
        token_g = bidx * self.tpc + grp
        valid = token_g < ntokens
        token = Int32(
            cutlass.select_(valid, token_g, ntokens - 1)
        )  # loads of a missing token read the last one
        if cutlass.const_expr(self.tpc == 1):
            token = bidx
            valid = True
        st = cutlass.memory.SmemAllocator().allocate(self.storage)

        # ---- the token's row (an input of the MoE, not written by the routing kernels)
        qv, sw, gs = load_row(
            q_in, sf_in, gs_in, token, t, K, 64, row_scale=self.row_scales
        )

        cute.arch.griddepcontrol_wait()  # launch_dependents once the routing is placed (below; early: now)
        if cutlass.const_expr(self.early):
            cute.arch.griddepcontrol_launch_dependents()
        rows = self._rank_rows(
            st,
            t,
            t % 32,
            bidx,
            tall,
            token,
            valid,
            ids,
            ecount,
            inv,
            experts_ptr,
            limits_ptr,
            count_ptr,
        )
        # publish inverse[], the tiles and the row count before the dependents may launch
        if cutlass.const_expr(not self.early):
            cute.arch.fence_acq_rel_gpu()
            cute.arch.barrier()
            cute.arch.griddepcontrol_launch_dependents()

        # ---- copies to the routed rows (a missing token of the last CTA stores nothing)
        if valid:
            qb, sb, gb = q_ptr.toint(), sf_ptr.toint(), gs_ptr.toint()
            if cutlass.const_expr(self.fused):
                zero_row(y_ptr, token, t, K, 64)
            rowtok, sf_linear, fused, rsc = (
                self.rowtok,
                self.sf_linear,
                self.fused,
                self.row_scales,
            )
            for slot in cutlass.range_constexpr(T):
                row = rows[slot]
                if row >= 0:  # a skipped route has no row
                    _scatter_store(
                        qv,
                        sw,
                        gs,
                        row,
                        slot,
                        token,
                        t,
                        qb,
                        sb,
                        gb,
                        w_ptr,
                        rtok_ptr,
                        rwt_ptr,
                        K,
                        T,
                        BM,
                        rowtok,
                        sf_linear,
                        fused,
                        rsc,
                    )

    @cute.jit
    def _rank_rows(
        self,
        st,
        t,
        lane,
        bidx,
        tall,
        token,
        valid,
        ids,
        ecount,
        inv,
        experts_ptr,
        limits_ptr,
        count_ptr,
    ):
        """Rows of the token's routes from their ranks (``rank_place``); see the module docstring. Warp 0 of
        the CTA: lane l scans experts EPL l .. EPL l + EPL - 1 (warp scan of the lane sums), one scan per CTA."""
        T, E, BM = self.topk, self.E, self.bm
        EPL = (E + 31) // 32
        exact = E % (32 * EPL) == 0
        sbase = st.sbase.get_tensor(cute.make_layout(E))
        # the token's experts and ranks do not depend on the scan: issued first
        es = [ids[token * T + s] for s in range(T)]
        ranks = [inv[token * T + s] for s in range(T)]
        oks = [True] * T
        if cutlass.const_expr(self.base):
            es = [e - self.base for e in es]
        oks = [(e >= 0) & (e < E) for e in es]
        es = [Int32(cutlass.select_(oks[s], es[s], Int32(0))) for s in range(T)]
        if tall < 32:
            cnt, pc = [], []
            for j in cutlass.range_constexpr(EPL):
                e = EPL * lane + j
                c = Int32(0)
                if cutlass.const_expr(exact):
                    c = ecount[e]
                else:
                    c = Int32(
                        cutlass.select_(
                            e < E,
                            ecount[Int32(cutlass.select_(e < E, e, E - 1))],
                            Int32(0),
                        )
                    )
                cnt.append(c)
                pc.append((c + BM - 1) // BM * BM)
            v = pc[0]
            for j in cutlass.range_constexpr(1, EPL):
                v = v + pc[j]
            incl = v
            for d in cutlass.range_constexpr(5):
                u = cute.arch.shuffle_sync_up(incl, offset=1 << d)
                incl = incl + Int32(cutlass.select_(lane >= (1 << d), u, Int32(0)))
            base = [incl - v]
            for j in cutlass.range_constexpr(1, EPL):
                base.append(base[j - 1] + pc[j - 1])
            for j in cutlass.range_constexpr(EPL):
                if cutlass.const_expr(exact):  # noqa: SIM114 (compile-time branch; the elif is a runtime check)
                    sbase[EPL * lane + j] = base[j]
                elif EPL * lane + j < E:
                    sbase[EPL * lane + j] = base[j]
            if bidx == (
                -1 if self.early else 0
            ):  # early: the gate/up writes the tile table
                ex = cute.make_tensor(
                    cute.make_ptr(Int32, experts_ptr.toint(), GMEM, assumed_align=4),
                    cute.make_layout(1 << 30),
                )
                lim = cute.make_tensor(
                    cute.make_ptr(Int32, limits_ptr.toint(), GMEM, assumed_align=4),
                    cute.make_layout(1 << 30),
                )
                for j in cutlass.range_constexpr(EPL):
                    first = base[j] // BM
                    for i in cutlass.range(
                        pc[j] // BM, unroll=1
                    ):  # experts past E have no routes
                        ex[first + i] = EPL * lane + j
                        lim[first + i] = base[j] + cnt[j]
                if lane == 31:
                    cute.make_tensor(
                        cute.make_ptr(Int32, count_ptr.toint(), GMEM, assumed_align=4),
                        cute.make_layout(1),
                    )[0] = incl
        cute.arch.barrier()  # base[] written; every thread holds its token's ranks before they are overwritten
        rows = [sbase[es[s]] + ranks[s] for s in range(T)]
        rows = [Int32(cutlass.select_(oks[s], rows[s], Int32(-1))) for s in range(T)]
        mine = rows[T - 1]
        for s in cutlass.range_constexpr(T - 2, -1, -1):
            mine = Int32(cutlass.select_(t == s, rows[s], mine))
        if (t < T) & valid:
            inv[token * T + t] = mine
        return rows


@cute.jit
def _chunk_store(
    sw,
    gs,
    row,
    slot: cutlass.Constexpr,
    token,
    tp,
    qb,
    sb,
    gb,
    w_ptr,
    rtok_ptr,
    rwt_ptr,
    K: cutlass.Constexpr,
    T: cutlass.Constexpr,
    fused: cutlass.Constexpr,
    row_scale: cutlass.Constexpr = True,
):
    """ScatterChunks: the token's copies (thread tp of 64) into its routed row ``row`` of slot ``slot``: the row's
    token (the gate/up gathers the FP4 rows itself, ``GroupedGemm(x_gather)``), block scales and row scale (only with
    ``row_scale``)."""
    NS = K // 64
    if tp == 0:
        cute.arch.store(
            cute.make_ptr(Int32, qb + Int64(row) * 4, GMEM, assumed_align=4),
            Int32(token),
        )
    store_chunks(st_u32, sb + Int64(row) * (K // 16), sw, tp, 64, NS, 4)
    if cutlass.const_expr(row_scale):
        if tp == 0:
            cute.arch.store(
                cute.make_ptr(Float32, gb + Int64(row) * 4, GMEM, assumed_align=4), gs
            )
    if cutlass.const_expr(fused):
        if tp == 1:
            route_info(w_ptr, rtok_ptr, rwt_ptr, token, slot, row, T)


class ScatterChunks:
    """Long batches: scatter with in-kernel placement, organized around the routing kernel's counting chunks (the
    input arrives quantized; each routed row gets its token, for the token-major gate/up that gathers the FP4 rows
    itself, and copies of the token's row-major block scales and row scale).

    A CTA takes ``tokens_per_cta`` tokens of one chunk (two warps per token), issues all its global loads first (the
    tokens' block scales and row scales, the experts' totals and the chunk's offsets, the chunk's ids), scans the
    padded totals (threads e < E, ceil(E / 32) warps), ranks its routes among the chunk's earlier routes of the same
    expert, writes ``inverse`` and the tile metadata (tiles b, b + grid, ...; last warp, warp reductions; with ``pairs``
    also the pair tiles q = b, b + grid, ... from a scan of each expert's ceil(blocks / 2): second-to-last warp) and stores
    the tokens from the rows in shared memory. The ranks: where the CTA's routes fill at least a warp
    (``warp_rank``), thread r holds route r of the chunk, its rank among its warp's routes from ``match.any`` plus the
    earlier warps' counts of the expert (shared memory, per warp and expert); else a loop over the chunk's ids per
    route (cheaper where a CTA takes a few routes: the whole chunk's match per CTA repeats across the chunk's CTAs, and
    ``match`` is slow to issue). A CTA needs at
    least max(chunk, E) threads (``moe``: tokens_per_cta = max(2, next power of 2 of max(chunk, E) / 64)). A chunk is
    one routing CTA's 16 tokens: 128 routes at topk 8, 96 at topk 6, 160 at topk 10.
    """

    def __init__(
        self,
        k,
        topk,
        *,
        num_experts=128,
        chunk=128,
        tokens_per_cta=2,
        pad_m=128,
        fused=False,
        base=0,
        row_scales=True,
        pairs=False,
    ):
        # pairs (the fused down GEMM as CTA pairs, GroupedGemm(pair_table)): also the pair tiles: each expert's 128-row
        # blocks two by two (the last one alone at an odd count); pairs[0] the pair tiles' count, pairs[1 + q] pair
        # tile q's first row block
        self.pairs = pairs
        # routes whose id is outside [0, E) get no row (inverse -1) and are not copied; base: the rank's first expert
        # (expert parallelism), local expert = id - base; row_scales: as Scatter's (False:
        # per-tensor input, no row scales copied)
        self.base = base
        self.row_scales = row_scales
        # the token-major gate/up gathers the input rows (GroupedGemm x_gather): each row's token (Int32 at q_ptr)
        # instead of the FP4 row
        # any hidden size (a multiple of 64) and E (the expert scan takes ceil(E / 32) warps; threads past E add
        # nothing); chunk: the routes of one routing CTA's tokens (moe: ROUTE_TOKENS * topk, 128 at topk 8)
        assert chunk % topk == 0 and k % 64 == 0, (
            f"long batches: whole tokens per chunk (got hidden {k}, {num_experts} experts, topk {topk}, chunk {chunk})"
        )
        self.k, self.topk, self.bm, self.E, self.chunk = (
            k,
            topk,
            128,
            num_experts,
            chunk,
        )  # 128-row tiles
        # pad_m: each expert's rows padded to pad_m (256: the 2-CTA gate/up's pair tiles); the tile metadata stays per
        # 128 rows
        assert pad_m in (128, 256)
        self.pad_m = pad_m
        # tokens_per_cta: a CTA takes 1 / sub of a chunk (sub = chunk / topk / tokens_per_cta) and repeats the
        # chunk's scan; it needs >= 128 threads for the experts and the chunk's ids
        self.tpc = tokens_per_cta
        self.sub = chunk // topk // tokens_per_cta
        assert self.sub * tokens_per_cta * topk == chunk and 64 * tokens_per_cta >= max(
            chunk, num_experts
        ), (
            f"tokens_per_cta {tokens_per_cta}: must divide the chunk's {chunk // topk} tokens, 64 threads each >= "
            f"max(chunk {chunk}, E {num_experts})"
        )
        self.threads = 64 * self.tpc
        # fused (the fused finalize): store each row's token and routing weight (w_ptr [tokens, topk] -> rtok_ptr,
        # rwt_ptr) for the token-major down GEMM's adds; the output (y_ptr, unused here) is zeroed by the gate/up
        # (GroupedGemm zero_rows), off this kernel's serial pass
        self.fused = fused
        # warp_rank (see the class docstring): the per-warp route counts per expert of the chunk's warps but the last
        # (whose counts no route reads) in shared memory; else the chunk's ids
        self.warp_rank = tokens_per_cta * topk >= 32
        self.cnt_n = ((chunk + 31) // 32 - 1) * num_experts if self.warp_rank else 0
        self.ids_n = 0 if self.warp_rank else chunk

    @cute.jit
    def __call__(
        self,
        q_in: cute.Pointer,
        sf_in: cute.Pointer,
        gs_in: cute.Pointer,
        inv_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        ids_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        experts_ptr: cute.Pointer,
        limits_ptr: cute.Pointer,
        count_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        rtok_ptr: cute.Pointer,
        rwt_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        pairs_ptr: cute.Pointer,
        tokens: Int32,
        stream,
    ):
        NP = self.E if self.pairs else 0

        @cute.struct
        class Storage:
            wt: cute.struct.MemRange[Int32, max(4, (self.E + 31) // 32)]
            wtp: cute.struct.MemRange[Int32, max(4, (self.E + 31) // 32) if NP else 0]
            pend: cute.struct.MemRange[
                Int32, NP
            ]  # pairs: end of the expert's pair tiles
            pbase: cute.struct.MemRange[
                Int32, NP
            ]  # pairs: first block - 2 x first pair tile
            ids: cute.struct.MemRange[Int32, self.ids_n]
            start: cute.struct.MemRange[
                Int32, self.E
            ]  # first row of the expert's routes in this chunk
            end: cute.struct.MemRange[Int32, self.E]  # end of the expert's padded rows
            lim: cute.struct.MemRange[Int32, self.E]  # end of the expert's routes
            rows: cute.struct.MemRange[Int32, self.chunk]
            cnt: cute.struct.MemRange[Int32, self.cnt_n]  # [warp, expert] route counts

        self.storage = Storage
        n = tokens * self.topk
        inv = cute.make_tensor(inv_ptr, cute.make_layout(n))
        ids = cute.make_tensor(ids_ptr, cute.make_layout(n))
        chunks = (n + self.chunk - 1) // self.chunk
        partial = cute.make_tensor(partial_ptr, cute.make_layout((chunks + 1) * self.E))
        # 1024 threads per CTA: 64 registers (.reqntid alone caps it there too, so no fallback is needed)
        _MAXNREG[0] = 64 if self.threads > 512 else None
        try:
            self.kernel(
                q_in,
                sf_in,
                gs_in,
                inv,
                q_ptr,
                sf_ptr,
                gs_ptr,
                ids,
                partial,
                experts_ptr,
                limits_ptr,
                count_ptr,
                w_ptr,
                rtok_ptr,
                rwt_ptr,
                y_ptr,
                pairs_ptr,
                tokens,
                chunks,
            ).launch(
                grid=[(tokens + self.tpc - 1) // self.tpc, 1, 1],
                block=[self.threads, 1, 1],
                stream=stream,
            )
        finally:
            _MAXNREG[0] = None

    @cute.kernel
    def kernel(
        self,
        q_in: cute.Pointer,
        sf_in: cute.Pointer,
        gs_in: cute.Pointer,
        inv: cute.Tensor,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        ids: cute.Tensor,
        partial: cute.Tensor,
        experts_ptr: cute.Pointer,
        limits_ptr: cute.Pointer,
        count_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        rtok_ptr: cute.Pointer,
        rwt_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        pairs_ptr: cute.Pointer,
        tokens: Int32,
        nch: Int32,
    ):
        K, T, E, CH, BM = self.k, self.topk, self.E, self.chunk, self.bm
        RPC = self.tpc * T  # routes of the CTA
        b, _, _ = cute.arch.block_idx()
        nb, _, _ = cute.arch.grid_dim()
        c = b // self.sub
        part = b % self.sub
        t, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = t % 32
        p = t // 64  # token slot of the warp pair
        tp = t % 64  # thread within the pair
        st = cutlass.memory.SmemAllocator().allocate(self.storage)
        wt = st.wt.get_tensor(cute.make_layout(max(4, (E + 31) // 32)))
        start_s = st.start.get_tensor(cute.make_layout(E))
        end_s = st.end.get_tensor(cute.make_layout(E))
        lim_s = st.lim.get_tensor(cute.make_layout(E))
        rows_s = st.rows.get_tensor(cute.make_layout(CH))
        WR, NZ = self.warp_rank, self.cnt_n
        if cutlass.const_expr(not WR):
            ids_s = st.ids.get_tensor(cute.make_layout(CH))
        routes = tokens * T
        first = c * CH
        token = b * self.tpc + p

        # ---- every global load first: the token's row (clamped past the last token), the experts' totals and
        # this chunk's offsets (thread e < E), the chunk's ids (thread r < CH)
        _, sw, gs = load_row(
            q_in,
            sf_in,
            gs_in,
            cutlass.min(token, tokens - 1),
            tp,
            K,
            64,
            rows=False,
            row_scale=self.row_scales,
        )
        e = cutlass.min(t, E - 1)
        tot = partial[nch * E + e]
        off = partial[c * E + e]
        my_route = first + t
        my_id = ids[cutlass.min(my_route, routes - 1)]
        my_id = Int32(cutlass.select_(my_route < routes, my_id, Int32(E)))
        if cutlass.const_expr(self.base):
            my_id = my_id - self.base
        my_id = Int32(cutlass.select_((my_id >= 0) & (my_id < E), my_id, Int32(E)))

        # ---- placement, part 1: scan of the padded totals over the experts (thread e < E in warps 0 .. NWE - 1)
        NWE = (E + 31) // 32
        PM = self.pad_m
        pad = (tot + PM - 1) // PM * PM
        if cutlass.const_expr(E % 32 != 0):
            pad = Int32(
                cutlass.select_(t < E, pad, Int32(0))
            )  # threads past E in the last expert warp
        v = pad
        for d in cutlass.range_constexpr(5):
            u = cute.arch.shuffle_sync_up(v, offset=1 << d)
            v = v + Int32(cutlass.select_(lane >= (1 << d), u, Int32(0)))
        if t < 32 * NWE:
            if lane == 31:
                wt[warp] = v
        if cutlass.const_expr(self.pairs):
            # the experts' pair tiles: ceil(blocks / 2), scanned as the padded totals
            wtp = st.wtp.get_tensor(cute.make_layout(max(4, NWE)))
            npr = (pad // BM + 1) // 2
            vp = npr
            for d in cutlass.range_constexpr(5):
                u = cute.arch.shuffle_sync_up(vp, offset=1 << d)
                vp = vp + Int32(cutlass.select_(lane >= (1 << d), u, Int32(0)))
            if t < 32 * NWE:
                if lane == 31:
                    wtp[warp] = vp
        # ranks, part 1 (warp_rank): thread r < CH holds route r of the chunk (others the sentinel E); its group of
        # equal experts in its warp (match.any); the counts table zeroed before the barrier, written after it
        NWC = (CH + 31) // 32  # warps holding the chunk's routes
        mv = Int32(cutlass.select_(t < CH, my_id, Int32(E)))
        grp = Uint32(0)
        below = Uint32(0)
        if cutlass.const_expr(NZ > 0):
            cnt_s = st.cnt.get_tensor(cute.make_layout(NZ))
            for j in cutlass.range_constexpr((NZ + self.threads - 1) // self.threads):
                if cutlass.const_expr((j + 1) * self.threads <= NZ):
                    cnt_s[t + j * self.threads] = Int32(0)
                else:
                    if t + j * self.threads < NZ:
                        cnt_s[t + j * self.threads] = Int32(0)
        if cutlass.const_expr(WR):
            if warp < NWC:
                grp = cute.arch.match_sync(0xFFFFFFFF, mv)
                below = grp & cute.arch.lanemask_lt()
        else:
            if t < CH:
                ids_s[t] = my_id
        cute.arch.barrier()
        if cutlass.const_expr(NZ > 0):
            # the group's first lane writes its warp's count of the expert
            if warp < NWC - 1:
                if (below == Uint32(0)) & (mv < E):
                    cnt_s[warp * E + mv] = Int32(cute.arch.popc(grp))
        ws = [wt[j] for j in range(NWE)]
        total = ws[0]
        for j in cutlass.range_constexpr(1, NWE):
            total = total + ws[j]
        if t < E:
            end = v
            for j in cutlass.range_constexpr(NWE - 1):
                end = end + Int32(cutlass.select_(warp >= j + 1, ws[j], Int32(0)))
            end_s[t] = end
            lim_s[t] = end - pad + tot
            start_s[t] = end - pad + off
        if b == 0:
            if t == 0:
                cute.make_tensor(
                    cute.make_ptr(Int32, count_ptr.toint(), GMEM, assumed_align=4),
                    cute.make_layout(1),
                )[0] = total
        totalp = Int32(0)
        if cutlass.const_expr(self.pairs):
            pend_s = st.pend.get_tensor(cute.make_layout(E))
            pbase_s = st.pbase.get_tensor(cute.make_layout(E))
            wsp = [wtp[j] for j in range(NWE)]
            totalp = wsp[0]
            for j in cutlass.range_constexpr(1, NWE):
                totalp = totalp + wsp[j]
            if t < E:
                pe = vp
                for j in cutlass.range_constexpr(NWE - 1):
                    pe = pe + Int32(cutlass.select_(warp >= j + 1, wsp[j], Int32(0)))
                pend_s[t] = pe
                pbase_s[t] = (end_s[t] - pad) // BM - 2 * (pe - npr)
            if b == 0:
                if t == 0:
                    cute.make_tensor(
                        cute.make_ptr(Int32, pairs_ptr.toint(), GMEM, assumed_align=4),
                        cute.make_layout(1),
                    )[0] = totalp

        cute.arch.barrier()  # publishes start / end / lim / ids / the counts

        # ---- placement, part 2: the rank of each of the CTA's routes among the chunk's earlier routes of its expert
        if cutlass.const_expr(WR):
            # route t of the chunk (this CTA's part: lo <= t < lo + RPC): its rank in its warp plus the earlier warps'
            # counts
            lo = part * RPC
            if (t >= lo) & (t < lo + RPC) & (t < CH) & (first + t < routes):
                wrank = Int32(cute.arch.popc(below))
                if cutlass.const_expr(NZ > 0):
                    mc = cutlass.min(mv, E - 1)
                    for j in cutlass.range_constexpr(NWC - 1):
                        wrank = wrank + Int32(
                            cutlass.select_(warp > j, cnt_s[j * E + mc], Int32(0))
                        )
                wrow = Int32(0)
                wrow = Int32(
                    cutlass.select_(
                        mv < E, start_s[cutlass.min(mv, E - 1)] + wrank, Int32(-1)
                    )
                )
                rows_s[t - lo] = wrow
                inv[first + t] = wrow
        else:
            # route part * RPC + t (t < RPC): a loop over the chunk's ids
            if t < RPC:
                rl = part * RPC + t  # route within the chunk
                rid = ids_s[rl]
                rank = Int32(0)
                for i in cutlass.range_constexpr(CH - 1):
                    rank = rank + Int32((ids_s[i] == rid) & (i < rl))
                if first + rl < routes:
                    row = Int32(0)
                    row = Int32(
                        cutlass.select_(
                            rid < E,
                            start_s[cutlass.min(rid, E - 1)] + rank,
                            Int32(-1),
                        )
                    )
                    rows_s[t] = row
                    inv[first + rl] = row
        # ---- tiles b, b + grid, ...: owner = experts whose padded rows end at or before the tile (last warp)
        if warp == self.threads // 32 - 1:
            tile = b
            while tile < total // BM:
                cnt = Int32(0)
                for j in cutlass.range_constexpr(E // 32):
                    cnt = cnt + Int32(end_s[lane + 32 * j] <= tile * BM)
                if cutlass.const_expr(
                    E % 32 != 0
                ):  # the last, partial group of 32 experts
                    j = E // 32
                    ej = cutlass.min(lane + 32 * j, E - 1)
                    cnt = cnt + Int32((lane + 32 * j < E) & (end_s[ej] <= tile * BM))
                owner = cute.arch.warp_redux_sync(cnt, "add")
                if lane == 0:
                    cute.make_tensor(
                        cute.make_ptr(
                            Int32,
                            experts_ptr.toint() + Int64(tile) * 4,
                            GMEM,
                            assumed_align=4,
                        ),
                        cute.make_layout(1),
                    )[0] = owner
                    cute.make_tensor(
                        cute.make_ptr(
                            Int32,
                            limits_ptr.toint() + Int64(tile) * 4,
                            GMEM,
                            assumed_align=4,
                        ),
                        cute.make_layout(1),
                    )[0] = lim_s[owner]
                tile += nb
        if cutlass.const_expr(self.pairs):
            # pair tiles q = b, b + grid, ...: owner = experts whose pair tiles end at or before q
            if warp == self.threads // 32 - 2:
                q = b
                while q < totalp:
                    cnt = Int32(0)
                    for j in cutlass.range_constexpr(E // 32):
                        cnt = cnt + Int32(pend_s[lane + 32 * j] <= q)
                    if cutlass.const_expr(E % 32 != 0):
                        j = E // 32
                        ej = cutlass.min(lane + 32 * j, E - 1)
                        cnt = cnt + Int32((lane + 32 * j < E) & (pend_s[ej] <= q))
                    owner = cute.arch.warp_redux_sync(cnt, "add")
                    if lane == 0:
                        cute.make_tensor(
                            cute.make_ptr(
                                Int32,
                                pairs_ptr.toint() + Int64(q + 1) * 4,
                                GMEM,
                                assumed_align=4,
                            ),
                            cute.make_layout(1),
                        )[0] = pbase_s[owner] + 2 * q
                    q += nb

        cute.arch.barrier()  # rows_s

        # ---- copies of the token to its routed rows (row-major scales [rows, K / 16])
        if token < tokens:
            qb = q_ptr.toint()
            sb = sf_ptr.toint()
            gb = gs_ptr.toint()
            fused, rsc = self.fused, self.row_scales
            for slot in cutlass.range_constexpr(T):
                row = rows_s[p * T + slot]
                if row >= 0:  # a skipped route has no row
                    _chunk_store(
                        sw,
                        gs,
                        row,
                        slot,
                        token,
                        tp,
                        qb,
                        sb,
                        gb,
                        w_ptr,
                        rtok_ptr,
                        rwt_ptr,
                        K,
                        T,
                        fused,
                        rsc,
                    )
