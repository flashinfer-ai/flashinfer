# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# Adapted from humanfia/kda-for-kda-release; see licenses/LICENSE.kda-for-kda.

# ruff: noqa: SIM117, F841
# Keep the imported TIRx builder contexts and trace-time assignments intact.
"""KDA forward, packed varlen: persistent warp-specialized CTAs over (sequence, head) items, G-form chain.

Family: fused item-parallel, G-form chain over packed sequences, distributed over min(#SMs, H * num_seqs)
CTAs (equal-length items go to the least loaded CTAs, so packed-varlen shapes use all SMs instead of one
CTA per head).

Chunk size 64.  The per-channel log2 decay is factored around two integer half
references (r0 = rint(gamma_31 / 2) for tokens 0-31, eps = min(0, rint(e + F + 0.5))
for tokens 32-63, with the chunk total e clamped through e_cl = e - eps for the
state scale) so every operand exponent stays within +-~122 bits.  Per chunk the
recurrent chain is
   S_bf -> G = S k2^T -> (bf16) -> v_new = u - G T'^T -> (bf16) -> S += v_new kA -> decay -> S_bf

The delta lands before the decay, so the pre-decay accumulator holds the state times
2^-e_cl (up to 2^(CLAMP_F + 0.5)).  The TMEM state therefore lives at 2^-S_SHIFT: T' carries
2^-S_SHIFT (so u, v_new and v_new_bf do too), q2 / k2 and the staged Aqk carry 2^S_SHIFT
(so G and O come out unscaled).  A piece loaded from initial_state runs its first chunk's
G / O_I on the unscaled image (q2 / k2 without 2^S_SHIFT) and rescales S_acc before that
chunk's delta; a piece that ends in final_state leaves the frame through its last decay
factor; handoffs stay in the frame.
This centers the representable state range (about 2^-66 .. 2^66 times the true values)
instead of leaving ~8 bits of headroom above the true state.

Varlen: a one-warp prologue expands cu_seqlens into an SMEM item table (token offset,
valid rows, sequence id, first/last flags); every role loops over the same flat item
index so all barrier parities stay item-based.  Rows past a sequence end are masked in
the prep role (k, q -> 0, log-decay -> 0, beta -> 0), so a partial tail chunk leaves the
state untouched for those rows.  The state role loads initial_state[seq] into TMEM at
each sequence's first chunk and writes final_state[seq] after its last chunk.  Output
rows are written with predicated 16-byte global stores from the transposed SMEM staging
tile (no fixed-box TMA store can be clipped at an interior sequence boundary).

Scheduling: the host picks whole-item LPT lists or chunk-linear splits with fp32
continuation handoffs from a cost model (see _host_item_table).  Initial-state loads and
final-state stores use 256-bit streaming fp32 operations.
"""

import math

import tvm
import tirx_kernels.tirx_lite as txl
from tvm.backend.cuda.cpp.descriptors import (
    encode_instr_descriptor_dense_uint32 as _idesc,
)

BT, D = 64, 128
RCP_LN2 = 1.0 / math.log(2.0)
GATE_C = -2.5 * RCP_LN2
EPS = 1e-6
MAX_ITEMS = 160


MMA = "tcgen05.mma.cta_group::1.kind::f16"
MMA_WS = "tcgen05.mma.ws.cta_group::1.kind::f16"
TMA3 = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
TMA2 = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
CACHE_EVICT_FIRST = txl.uint64(0x12F0000000000000)
COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
LD32x64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
ST32x64 = "tcgen05.st.sync.aligned.32x32b.x64.b32"
ST32x32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
ST32x16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
LD16x8 = "tcgen05.ld.sync.aligned.16x256b.x8.b32"
LD16x2 = "tcgen05.ld.sync.aligned.16x256b.x2.b32"
LD16x4 = "tcgen05.ld.sync.aligned.16x256b.x4.b32"


CLAMP_F = 120.0
# log2 scale of the TMEM state frame (see the module docstring)
S_SHIFT = 60
LD32x32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
LD32x16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMA_S2G3 = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQ = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
FENCE_AFTER = "tcgen05.fence::after_thread_sync"
FENCE_BEFORE = "tcgen05.fence::before_thread_sync"


C_SACC, C_SBF, C_VNBF, C_OACC, C_GACC, C_AA0, C_GBF = 0, 128, 192, 224, 288, 352, 480
N_COLS = 512

F32, BF = "float32", "bfloat16"
ID_AQK64 = _idesc(64, 64, 16, F32, BF, BF, False, False)
ID_U = _idesc(128, 64, 16, F32, BF, BF, True, True)
ID_VN = _idesc(128, 64, 16, F32, BF, BF, False, True, neg_a=True)
ID_G64 = _idesc(128, 64, 16, F32, BF, BF, False, False)
ID_DL = _idesc(128, 128, 16, F32, BF, BF, False, True)
ID_OX = _idesc(128, 64, 16, F32, BF, BF, False, True)

W_STATE = list(range(0, 4))
W_LD, W_MMI, W_MMC, W_RND = 4, 5, 6, 7
W_INTRA = list(range(8, 12))
W_PREP = list(range(12, 20))
NWARPS = 20

NB_PREP, NB_INTRA, NB_STATE = 1, 2, 3


def build_kernel(
    H,
    arch,
    state_regs=112,
    wg_regs=40,
    intra_regs=104,
    prep_regs=112,
    max_items=MAX_ITEMS,
):
    """Persistent CTAs, each walking its host-built item list (see _host_item_table).

    Item lists up to MAX_ITEMS long are staged in SMEM; longer ones (long sequences on few
    (sequence, head) units) are read from global memory in place."""
    assert H % 8 == 0
    global_items = max_items > MAX_ITEMS

    def kda_fwd(
        q_map,
        k_map,
        v_map,
        g_map,
        beta_map,
        o_map,
        out,
        A_log,
        dt_bias,
        h0,
        final_state,
        hand,
        flags,
        items,
        item_counts,
        num_ctas,
        scale,
    ):
        cta = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        tid = txl.thread_id()

        smem = txl.smem_pool()
        tmem_addr = smem.alloc((1,), txl.u32)
        item_tbl = smem.alloc((2 * min(max_items, MAX_ITEMS),), txl.u32, align=16)
        item_meta = smem.alloc((4,), txl.u32, align=16)
        adt_s = smem.alloc((128 + 4,), txl.f32, align=16)
        rsq = smem.alloc((2 * 4 * 64,), txl.f32, align=16)
        beta_s = smem.alloc((2 * 64 * 8,), txl.bf16, align=128)
        bsig = smem.alloc((3 * 64,), txl.f32, align=16)
        dvec = smem.alloc((3 * 128,), txl.f32, align=16)

        tot = smem.alloc((4 * 128,), txl.f32, align=16)

        def mbar(count, depth=1):
            b = txl.MBarrier(smem, depth)
            b.init(count)
            return b

        ring_full = mbar(1, 2)
        v_full = mbar(1)
        rfull = mbar(8, 2)

        aa_r = mbar(1, 2)
        akk_r = mbar(1, 2)
        aa_free = mbar(4, 2)
        T_ready = mbar(4, 2)
        dvec_ready = mbar(2, 3)
        dvec_free = mbar(4, 3)
        g_done = mbar(1)
        g_ready = mbar(4)
        u_done = mbar(1, 2)
        Aqk_ready = mbar(4)
        S_ready = mbar(4, 1)
        vnew_done = mbar(1, 2)
        kA_free = mbar(1, 2)
        oi_done = mbar(1)
        o_done = mbar(1)
        o_free = mbar(4)
        aqk_stage_free = mbar(1)

        pool = smem.pool
        TOFF = {}

        def tile_alloc(name, shape):
            view = smem.alloc(shape, txl.bf16, swizzle=txl.SW128B)
            nbytes = 2
            for d_ in shape:
                nbytes *= d_
            TOFF[name] = pool.offset - nbytes
            return view

        pool.move_base_to((pool.offset + 1023) // 1024 * 1024)
        # interleaved kq ring tile, rows [k 0-31 | q 0-31 | k 32-63 | q 32-63]: token t of x (k 0, q 1) is row (t // 32) * 64 + 32 * x + t % 32, so a rounds half reads its k- and q-targets as one N=64 operand
        kq_t = tile_alloc("kq", (2, 2 * BT, D))
        KQ_Q_OFF = (
            32 * 128
        )  # byte offset of a q row from the k row of the same token (32 rows of one 128-B slab)
        g_t = tile_alloc("g", (2, BT, D))
        STAGE_UNITS = BT * D * 2 // 16
        v_t = tile_alloc("v", (BT, D))
        pool.move_base_to((pool.offset + 1023) // 1024 * 1024)
        kA = tile_alloc("kA", (2, BT, D))
        q2t = tile_alloc("q2", (BT, D))
        k2t = tile_alloc("k2", (BT, D))
        Aqk_s = tile_alloc("Aqk", (64, 64))
        LTT = tile_alloc("LTT", (64, 64))
        TpT_t = tile_alloc("TpT", (64, 64))

        TpTlo_t = tile_alloc("TpTlo", (64, 64))
        o_st = tile_alloc("ost", (2, 32, 64))
        base0 = TOFF["kq"]
        for name in TOFF:
            TOFF[name] -= base0
        assert min(TOFF.values()) == 0

        # this CTA's host-built item table (see _host_item_table) -> SMEM by one warp; read only after the CTA barrier below
        with txl.If(warp == 0), txl.Then():
            cnt = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(cnt, item_counts.ptr_to([cta]))
            if not global_items:
                with txl.serial(2 * max_items // 32) as it_:
                    wi = it_ * 32 + lane
                    with txl.If(wi < 2 * cnt), txl.Then():
                        wv = txl.local_scalar("uint32")
                        txl.ptx.ld.global_.u32(
                            wv, items.ptr_to([cta * (2 * max_items) + wi])
                        )
                        txl.ptx.st.shared.u32(item_tbl.ptr_to([wi]), wv)
            with txl.If(lane == 0), txl.Then():
                txl.ptx.st.shared.u32(item_meta.ptr_to([0]), txl.Cast("uint32", cnt))
            # the item table aliases nothing, but the ring tiles are filled by TMA (async proxy) right after
            txl.ptx.fence.proxy.async_.shared__cta()

        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        with txl.If(warp == 0), txl.Then():
            txl.ptx[TMEM_ALLOC](txl.address_of(tmem_addr[0]), txl.uint32(N_COLS))
            txl.cuda.warp_sync()
        txl.cuda.cta_sync()

        def n_items_local():
            n_items = txl.local_scalar("int32")
            w = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(w, item_meta.ptr_to([0]))
            txl.assign(n_items, txl.Cast("int32", w))
            return n_items

        def item_word(idx):
            w = txl.local_scalar("uint32")
            if global_items:
                txl.ptx.ld.global_.u32(
                    w, items.ptr_to([cta * (2 * max_items) + 2 * idx])
                )
            else:
                txl.ptx.ld.shared.u32(w, item_tbl.ptr_to([2 * idx]))
            return w

        def item_word1(idx):
            w = txl.local_scalar("uint32")
            if global_items:
                txl.ptx.ld.global_.u32(
                    w, items.ptr_to([cta * (2 * max_items) + 2 * idx + 1])
                )
            else:
                txl.ptx.ld.shared.u32(w, item_tbl.ptr_to([2 * idx + 1]))
            return w

        def item_tok0(w):
            lo = txl.bitwise_and(w, txl.uint32(0xFFFF))
            hi = txl.shift_left(
                txl.bitwise_and(txl.shift_right(w, txl.uint32(23)), txl.uint32(0x1F)),
                txl.uint32(16),
            )
            return txl.Cast("int32", txl.bitwise_or(lo, hi))

        def item_nvalid(w):
            return txl.Cast(
                "int32",
                txl.bitwise_and(txl.shift_right(w, txl.uint32(16)), txl.uint32(0x7F)),
            )

        def item_first(w):
            return txl.bitwise_and(w, txl.uint32(1 << 30)) != txl.uint32(0)

        def item_last(w):
            return txl.bitwise_and(w, txl.uint32(1 << 31)) != txl.uint32(0)

        def item_src_hand(w):
            """Continuation piece: its state comes from handoff slot `cta` after the producer's flag."""
            return txl.bitwise_and(w, txl.uint32(1 << 29)) != txl.uint32(0)

        def item_dst_hand(w):
            """Head piece: its state goes to handoff slot `cta + 1`, published with a release flag."""
            return txl.bitwise_and(w, txl.uint32(1 << 28)) != txl.uint32(0)

        def item_seq(w1):
            return txl.Cast("int32", txl.bitwise_and(w1, txl.uint32(0xFFFF)))

        def item_head(w1):
            return txl.Cast("int32", txl.shift_right(w1, txl.uint32(16)))

        tbase = txl.local_scalar("uint32")
        txl.ptx.ld.shared.u32(tbase, tmem_addr.ptr_to([0]))

        tbase = txl.uniform(tbase)

        def tmem(col, lane_off=0):
            # plain add: the allocation starts at lane 0 and the column sum stays below 2^16, so no address field overflows and constant offsets fold into the tcgen05 immediates
            if isinstance(col, int):
                return tbase + txl.uint32((lane_off << 16) + col)
            return tbase + txl.uint32(lane_off << 16) + txl.Cast("uint32", col)

        def aa_col(b2, extra=0):
            return C_AA0 + extra + 64 * b2

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def warrive(b, idx):
            txl.cuda.warp_sync()
            with txl.If(lane == 0), txl.Then():
                b.arrive(idx)

        def irange(name):
            token = txl.alloc_local((1,), "uint32")
            txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        def iend(token):
            txl.cuda.iket.range_end(token[0])

        def mark(name):
            txl.cuda.iket.mark(name)

        def fwait(b, stage, parity, name=None):
            tok = irange(name) if name else None
            ready = txl.local_scalar("uint32", init=txl.uint32(0))
            # no suspend-time hint: a 2-instruction spin loop keeps the hot code small
            with txl.While(ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.shared.b64(
                    ready, b.ptr_to([stage]), txl.Cast("uint32", parity)
                )
            iend(tok)

        ZERO4 = (txl.uint32(0),) * 4

        def mma(d, aop, bop, idesc, acc):
            if not isinstance(acc, bool):
                acc = txl.local_scalar(
                    "uint32", init=txl.Select(acc, txl.uint32(1), txl.uint32(0))
                )
            txl.ptx[MMA](
                txl.Cast("uint32", d),
                aop,
                bop,
                txl.uint32(idesc),
                *ZERO4,
                txl.ptx.pred(acc),
            )

        def mma_ws(d, aop, bop, idesc, acc, collector):
            txl.ptx[f"{MMA_WS}.collector::b{collector}::fill"](
                txl.Cast("uint32", d),
                aop,
                bop,
                txl.uint32(idesc),
                txl.ptx.pred(acc),
                txl.uint64(0),
            )

        def mma_ws_lastuse(d, aop, bop, idesc, acc, collector):
            txl.ptx[f"{MMA_WS}.collector::b{collector}::lastuse"](
                txl.Cast("uint32", d),
                aop,
                bop,
                txl.uint32(idesc),
                txl.ptx.pred(acc),
                txl.uint64(0),
            )

        def commit(ptr):
            txl.ptx[COMMIT](ptr)

        PARENT_ROWS = {
            "kq": 2 * BT,
            "g": BT,
            "v": BT,
            "kA": BT,
            "q2": BT,
            "k2": BT,
            "Aqk": 64,
            "LTT": 64,
            "TpT": 64,
            "TpTlo": 64,
        }
        LBO_UNITS = sorted({r * 8 for r in PARENT_ROWS.values()} | {0})

        def make_templates():
            t = {}
            for ldo in LBO_UNITS:
                d = txl.SmemDescriptor()
                d.init(kq_t[0].ptr_to(0, 0), ldo=ldo, sdo=64, swizzle=3)
                t[ldo] = d.desc
            return t

        def tile_desc(tmpl, name, cols, major, kp=0, stage_units=None):
            """Descriptor of tile `name` at K step `kp` (a Python int, or a loop variable for rolled K loops)."""
            lbo = PARENT_ROWS[name] * 8
            ldo = 0 if (major == "k" and cols <= 64) else lbo
            if major == "k":
                step = (kp % 4) * 2 + (kp // 4) * lbo
            else:
                step = kp * 128
            if isinstance(kp, int):
                off = (TOFF[name] >> 4) + step
                d = tmpl[ldo] + txl.uint64(off) if off else tmpl[ldo]
            else:
                d = tmpl[ldo] + txl.Cast("uint64", step + (TOFF[name] >> 4))
            if stage_units is not None:
                d = d + txl.Cast("uint64", stage_units)
            return d

        def bf16x2(lo, hi):
            r = txl.local_scalar("uint32")
            txl.ptx.cvt.rn.bf16x2.f32(r, hi, lo)
            return r

        def unpack(u):
            lo = txl.reinterpret("float32", txl.shift_left(u, txl.uint32(16)))
            hi = txl.reinterpret("float32", txl.bitwise_and(u, txl.uint32(0xFFFF0000)))
            return lo, hi

        def hmul2(a, b):
            r = txl.local_scalar("uint32")
            txl.ptx.mul.rn.bf16x2(r, a, b)
            return r

        def hfma2(a, b, c):
            r = txl.local_scalar("uint32")
            txl.ptx.fma.rn.bf16x2(r, a, b, c)
            return r

        def ex2(x):
            r = txl.local_scalar("float32")
            txl.ptx.ex2.approx.ftz.f32(r, x)
            return r

        sp = txl.specialize()
        r_state = sp.role("state", warps=W_STATE, regs=state_regs)
        wg1 = sp.warpgroup("wg1", warps=range(4, 8), regs=wg_regs)
        r_ld = sp.role("load", warps=[W_LD], group=wg1)
        r_mmi = sp.role("mma_i", warps=[W_MMI], group=wg1)
        r_mmc = sp.role("mma_c", warps=[W_MMC], group=wg1)
        r_rnd = sp.role("rounds", warps=[W_RND], group=wg1)
        r_intra = sp.role("intra", warps=W_INTRA, regs=intra_regs)
        r_prep = sp.role("prep", warps=W_PREP, regs=prep_regs)

        def load_body():
            n_items = n_items_local()

            def issue_loads(cc):
                sc = cc % 2
                with txl.If(elected()), txl.Then():
                    tok0 = txl.local_scalar("int32", init=item_tok0(item_word(cc)))
                    h = txl.local_scalar("int32", init=item_head(item_word1(cc)))
                    mb = txl.cuda.cvta_generic_to_shared(ring_full.ptr_to([sc]))
                    # k / q: 4 KB boxes (64 channels x 32 tokens x 1 channel slab) into the interleaved kq tile
                    for tx, tmap in ((0, k_map), (1, q_map)):
                        for hf_ in range(2):
                            for sl_ in range(2):
                                txl.ptx[TMA3](
                                    kq_t[sc].ptr_to(64 * hf_ + 32 * tx, 64 * sl_),
                                    txl.address_of(tmap),
                                    txl.int32(0),
                                    tok0 + 32 * hf_,
                                    txl.Cast("int32", 2 * h + sl_),
                                    mb,
                                    CACHE_EVICT_FIRST,
                                )
                    txl.ptx[TMA3](
                        g_t[sc].ptr_to(0, 0),
                        txl.address_of(g_map),
                        txl.int32(0),
                        tok0,
                        txl.Cast("int32", 2 * h),
                        mb,
                        CACHE_EVICT_FIRST,
                    )
                    txl.ptx[TMA2](
                        beta_s.ptr_to([sc * 512]),
                        txl.address_of(beta_map),
                        txl.Cast("int32", (h // 8) * 8),
                        tok0,
                        mb,
                        CACHE_EVICT_FIRST,
                    )
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        ring_full.ptr_to([sc]), txl.uint32(3 * BT * D * 2 + 1024)
                    )

            issue_loads(txl.int32(0))
            with txl.If(n_items > 1), txl.Then():
                issue_loads(txl.int32(1))
            with txl.serial(n_items, unroll=False) as c:
                with txl.If(tvm.tirx.all(c >= 1, c + 1 < n_items)), txl.Then():
                    fwait(aa_r, (c + 1) % 2, ((c - 1) // 2) % 2, "ld-wait-aar")
                    txl.ptx.fence.proxy.async_.shared__cta()
                    issue_loads(c + 1)
                with txl.If(c >= 1), txl.Then():
                    fwait(u_done, (c + 1) % 2, ((c - 1) // 2) % 2, "ld-wait-udone")
                with txl.If(elected()), txl.Then():
                    tok0v = txl.local_scalar("int32", init=item_tok0(item_word(c)))
                    hv = txl.local_scalar("int32", init=item_head(item_word1(c)))
                    mbv = txl.cuda.cvta_generic_to_shared(v_full.ptr_to([0]))
                    txl.ptx[TMA3](
                        v_t.ptr_to(0, 0),
                        txl.address_of(v_map),
                        txl.int32(0),
                        tok0v,
                        txl.Cast("int32", 2 * hv),
                        mbv,
                        CACHE_EVICT_FIRST,
                    )
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        v_full.ptr_to([0]), txl.uint32(BT * D * 2)
                    )

        def rounds_body(halves):
            """The assigned independent halves of the intra-chunk products.  Target half h (tokens 32h..32h+31) of the
            round accumulator lives at lane offset 16h of AA[b2]: D_h[m][n] = source_h[m] . target_h[n],
            columns 0-31 = Akk^T (target = k * 2^(gamma_t - r_h)), 32-63 = Aqk^T (target = q * scale * 2^(gamma_t - r_h)).
            Half 0 (r_0 = gamma_31 / 2): source = k * 2^(r_0 - gamma_m) in g_t[b2] rows 0-31 (rows 32-63 are the
            raw gate tile, finite and always masked).  Half 1 (r_1 = eps): source = kA (the delta operand
            k * 2^(eps - gamma_m)).  Targets: k_t[b2] row t = k-target of token t, q_t[b2] row t = q-target of
            token t (prep writes each row in place of the raw row it read); half h uses rows 32h..32h+31 of both
            tiles as two N=32 operands (columns 0-31 and 32-63 of the accumulator)."""
            n_items = n_items_local()
            with txl.If(elected()), txl.Then():
                tmpl = make_templates()
                with txl.serial(n_items, unroll=False) as c:
                    b2 = c % 2
                    su = b2 * STAGE_UNITS
                    with txl.If(c >= 2), txl.Then():
                        fwait(aa_free, b2, (c // 2 + 1) % 2, "rd-wait-aafree")
                    fwait(rfull, b2, (c // 2) % 2, "rd-wait-rfull")
                    for hh_ in halves:
                        tok_round = irange(f"rd-half{hh_}")
                        mark("rd-issue")
                        a_name = "g" if hh_ == 0 else "kA"
                        # B operand: rows 64 hh_ .. 64 hh_ + 63 of the kq tile = [k-targets of half hh_ | q-targets] -> accumulator columns 0-31 = Akk^T, 32-63 = Aqk^T; the K loop is rolled (run-time descriptor offsets) to keep the hot code small
                        with txl.serial(8, unroll=False) as kp:
                            bd = tile_desc(
                                tmpl, "kq", 128, "k", kp, 2 * su
                            ) + txl.uint64(512 * hh_)
                            mma(
                                tmem(aa_col(b2), 16 * hh_),
                                tile_desc(tmpl, a_name, 128, "k", kp, su),
                                bd,
                                ID_AQK64,
                                kp != 0,
                            )
                        commit((akk_r if hh_ == 0 else aa_r).ptr_to([b2]))
                        iend(tok_round)

        def mmc_body():
            """Recurrence MMAs: G^T = S^T k2^T; O = S^T q2^T; u^T = v^T T'^T; v_new^T = u^T - G_bf^T T'^T;
            S^T += v_new^T kA; O += v_new^T Aqk^T."""
            n_items = n_items_local()
            with txl.If(elected()), txl.Then():
                tmpl = make_templates()
                with txl.serial(n_items, unroll=False) as c:
                    b2 = c % 2
                    su = b2 * STAGE_UNITS
                    fwait(rfull, b2, (c // 2) % 2, "mm-wait-rfull")
                    fwait(S_ready, 0, c % 2, "mm-wait-sready")
                    mark("mm-issue-G")
                    # rolled K loops for the plain MMAs (G, O_I, delta, OX) keep the hot code small
                    with txl.serial(8, unroll=False) as kp:
                        mma(
                            tmem(C_GACC, 0),
                            txl.Cast("uint32", tmem(C_SBF + 8 * kp, 0)),
                            tile_desc(tmpl, "k2", 128, "k", kp),
                            ID_G64,
                            kp != 0,
                        )
                    commit(g_done.ptr_to([0]))
                    with txl.If(c >= 1), txl.Then():
                        fwait(o_free, 0, (c + 1) % 2, "mm-wait-ofree")
                    mark("mm-issue-OI")
                    with txl.serial(8, unroll=False) as kp:
                        mma(
                            tmem(C_OACC, 0),
                            txl.Cast("uint32", tmem(C_SBF + 8 * kp, 0)),
                            tile_desc(tmpl, "q2", 128, "k", kp),
                            ID_G64,
                            kp != 0,
                        )
                    commit(oi_done.ptr_to([0]))
                    fwait(T_ready, b2, (c // 2) % 2, "mm-wait-Tready")
                    fwait(v_full, 0, c % 2, "mm-wait-vfull")
                    mark("mm-issue-u")
                    for kp in range(4):
                        mma_ws(
                            tmem(aa_col(b2), 0),
                            tile_desc(tmpl, "v", 128, "mn", kp),
                            tile_desc(tmpl, "TpT", 64, "mn", kp),
                            ID_U,
                            kp != 0,
                            kp,
                        )
                    commit(u_done.ptr_to([b2]))
                    fwait(g_ready, 0, c % 2, "mm-wait-gready")
                    mark("mm-issue-vnew")
                    for kp in range(4):
                        mma_ws_lastuse(
                            tmem(aa_col(b2), 0),
                            txl.Cast("uint32", tmem(C_GBF + 8 * kp, 0)),
                            tile_desc(tmpl, "TpT", 64, "mn", kp),
                            ID_VN,
                            True,
                            kp,
                        )
                    for kp in range(4):
                        mma(
                            tmem(aa_col(b2), 0),
                            txl.Cast("uint32", tmem(C_GBF + 8 * kp, 0)),
                            tile_desc(tmpl, "TpTlo", 64, "mn", kp),
                            ID_VN,
                            True,
                        )
                    commit(vnew_done.ptr_to([b2]))
                    fwait(aa_free, b2, (c // 2) % 2, "mm-wait-aafree")
                    mark("mm-issue-delta")
                    with txl.serial(4, unroll=False) as kp:
                        mma(
                            tmem(C_SACC, 0),
                            txl.Cast("uint32", tmem(C_VNBF + 8 * kp, 0)),
                            tile_desc(tmpl, "kA", 128, "mn", kp, su),
                            ID_DL,
                            True,
                        )
                    commit(kA_free.ptr_to([b2]))
                    fwait(Aqk_ready, 0, c % 2, "mm-wait-aqk")
                    mark("mm-issue-OX")
                    with txl.serial(4, unroll=False) as kp:
                        mma(
                            tmem(C_OACC, 0),
                            txl.Cast("uint32", tmem(C_VNBF + 8 * kp, 0)),
                            tile_desc(tmpl, "Aqk", 64, "mn", kp),
                            ID_OX,
                            True,
                        )
                    commit(o_done.ptr_to([0]))

        def state_body():
            v_idx = tid
            regs = txl.alloc_local((64,), "float32")
            packed = txl.alloc_local((16,), "uint32")

            def st_packed(col, base):
                for j in range(16):
                    txl.assign(
                        packed[j], bf16x2(regs[base + 2 * j], regs[base + 2 * j + 1])
                    )
                txl.ptx[ST32x16](tmem(col), *[packed[j] for j in range(16)])

            n_items = n_items_local()

            def load_state_from(buf, base, external=False):
                """[v][k] fp32 block at element offset `base` (V-first, lane = v row) -> S_acc fp32 and S_bf bf16 in TMEM, then S_ready.

                Handoff slots hold the 2^-S_SHIFT frame.  initial_state loads unscaled (a multiply here sits on the
                piece-boundary critical path); that piece's first chunk runs G / O_I unscaled and enter_frame() rescales
                S_acc before its delta."""

                for half_ in range(2):
                    if external:
                        for u in range(8):
                            txl.ptx["ld.global.L1::no_allocate.L2::evict_first.v8.f32"](
                                *[regs[8 * u + j] for j in range(8)],
                                buf.ptr_to([base + 64 * half_ + 8 * u]),
                            )
                    else:
                        for u in range(16):
                            txl.ptx.ld.global_.v4.f32(
                                regs[4 * u],
                                regs[4 * u + 1],
                                regs[4 * u + 2],
                                regs[4 * u + 3],
                                buf.ptr_to([base + 64 * half_ + 4 * u]),
                            )
                    txl.ptx[ST32x64](
                        tmem(C_SACC + 64 * half_), *[regs[j] for j in range(64)]
                    )
                    st_packed(C_SBF + 32 * half_, 0)
                    st_packed(C_SBF + 32 * half_ + 16, 32)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                warrive(S_ready, 0)

            def load_state(seq, h):
                """initial_state[seq][h] -> TMEM."""
                base = txl.local_scalar(
                    "int32", init=(seq * H + h) * (D * D) + v_idx * D
                )
                load_state_from(h0, base, external=True)

            def load_state_hand():
                """Continuation piece: acquire the producer CTA's flag, then load its fp32 handoff state.

                Warp 0 polls (all lanes read the same word); the other state warps wait at the role barrier, which
                orders their loads after the acquire.  The flag is reset here so every launch starts from zero."""
                with txl.If(warp == 0), txl.Then():
                    tok_fl = irange("st-wait-hand")
                    fl = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.While(fl == 0):
                        txl.ptx.ld.acquire.gpu.global_.s32(fl, flags.ptr_to([cta]))
                        with txl.If(fl == 0), txl.Then():
                            txl.cuda.nano_sleep(txl.uint64(256))
                    iend(tok_fl)
                    txl.cuda.warp_sync()
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.global_.s32(flags.ptr_to([cta]), txl.int32(0))
                txl.ptx.bar.sync(txl.uint32(NB_STATE), txl.uint32(128))
                base = txl.local_scalar("int32", init=cta * (D * D) + v_idx * D)
                load_state_from(hand, base)

            def store_state_to(buf, fbase, external=False):
                """S_acc (after the piece's last decay) -> [v][k] fp32 block at element offset `fbase`, one v row per lane.

                Handoff slots keep the 2^-S_SHIFT frame; a piece ending in final_state already left it in its last decay (prep's dvec)."""
                for half_ in range(2):
                    txl.ptx[LD32x64](
                        *[regs[j] for j in range(64)], tmem(C_SACC + 64 * half_)
                    )
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    if external:
                        for u in range(8):
                            txl.ptx["st.global.L1::no_allocate.L2::evict_first.v8.f32"](
                                buf.ptr_to([fbase + 64 * half_ + 8 * u]),
                                *[regs[8 * u + j] for j in range(8)],
                            )
                    else:
                        for u in range(16):
                            txl.ptx.st.global_.v4.f32(
                                buf.ptr_to([fbase + 64 * half_ + 4 * u]),
                                regs[4 * u],
                                regs[4 * u + 1],
                                regs[4 * u + 2],
                                regs[4 * u + 3],
                            )

            def store_state(seq, h):
                """-> final_state[seq][h]."""
                fbase = txl.local_scalar(
                    "int32", init=(seq * H + h) * (D * D) + v_idx * D
                )
                store_state_to(final_state, fbase, external=True)

            def store_state_hand():
                """Head piece: store the fp32 state into its handoff slot, then publish it.

                Every thread fences its own stores, the role barrier collects them, and thread 0 releases the flag
                at gpu scope."""
                fbase = txl.local_scalar(
                    "int32", init=(cta + txl.int32(1)) * (D * D) + v_idx * D
                )
                store_state_to(hand, fbase)
                txl.ptx.fence.acq_rel.gpu()
                txl.ptx.bar.sync(txl.uint32(NB_STATE), txl.uint32(128))
                with txl.If(tid == 0), txl.Then():
                    txl.ptx.fence.acq_rel.gpu()
                    txl.ptx.st.release.gpu.global_.s32(
                        flags.ptr_to([cta + txl.int32(1)]), txl.int32(1)
                    )

            def epilogue(c1, tok0, nvalid, h):
                """Stage both token halves through SMEM; full chunks go out by TMA, tail chunks by predicated stores."""
                fwait(o_done, 0, c1 % 2, "st-wait-odone")
                tok_epi = irange("st-epilogue")
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx[FENCE_AFTER]()
                for half in range(2):
                    txl.ptx[LD16x8](
                        *[regs[32 * half + j] for j in range(32)],
                        tmem(C_OACC, 16 * half),
                    )
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                warrive(o_free, 0)
                for g_ in range(2):
                    for half in range(2):
                        b = 32 * half
                        for m in range(4):
                            u = 4 * g_ + m
                            for hh in range(2):
                                txl.assign(
                                    packed[2 * m + hh],
                                    bf16x2(
                                        regs[b + 4 * u + 2 * hh],
                                        regs[b + 4 * u + 2 * hh + 1],
                                    ),
                                )
                        for hh in range(2):
                            if g_ == 0:
                                dst = o_st[warp // 2].ptr_to(
                                    lane, 32 * (warp % 2) + 16 * half + 8 * hh
                                )
                            else:
                                dst = Aqk_s.ptr_to(
                                    32 * (warp // 2) + lane,
                                    32 * (warp % 2) + 16 * half + 8 * hh,
                                )
                            txl.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                dst,
                                packed[hh],
                                packed[2 + hh],
                                packed[4 + hh],
                                packed[6 + hh],
                            )
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx.bar.sync(txl.uint32(NB_STATE), txl.uint32(128))
                with txl.If(nvalid == BT):
                    with txl.Then():
                        with txl.If(warp == 0), txl.Then():
                            with txl.If(lane == 0), txl.Then():
                                for g_ in range(2):
                                    src = (
                                        o_st[0].ptr_to(0, 0)
                                        if g_ == 0
                                        else Aqk_s.ptr_to(0, 0)
                                    )
                                    txl.ptx[TMA_S2G3](
                                        txl.address_of(o_map),
                                        txl.int32(0),
                                        tok0 + 32 * g_,
                                        txl.Cast("int32", 2 * h),
                                        src,
                                        CACHE_EVICT_FIRST,
                                    )
                                txl.ptx.cp.async_.bulk.commit_group()
                                txl.ptx.cp.async_.bulk.wait_group.read(0)
                                aqk_stage_free.arrive(0)
                    with txl.Else():
                        j8 = tid % 8
                        trow = tid // 8
                        obase = txl.local_scalar(
                            "int32", init=(tok0 * H + h) * D + 8 * j8
                        )
                        ov = txl.alloc_local((4,), "uint32")
                        for g_ in range(2):
                            with txl.serial(2, unroll=False) as p:
                                for th_ in range(2):
                                    t = trow + 16 * th_
                                    if g_ == 0:
                                        src = o_st[p].ptr_to(t, 8 * j8)
                                    else:
                                        src = Aqk_s.ptr_to(32 * p + t, 8 * j8)
                                    txl.ptx.ld.shared.v4.b32(
                                        ov[0], ov[1], ov[2], ov[3], src
                                    )
                                    txl.ptx.st.global_.v4.b32(
                                        out.ptr_to(
                                            [obase + (32 * g_ + t) * (H * D) + 64 * p]
                                        ),
                                        ov[0],
                                        ov[1],
                                        ov[2],
                                        ov[3],
                                        pred=(32 * g_ + t < nvalid),
                                    )
                        txl.ptx.bar.sync(txl.uint32(NB_STATE), txl.uint32(128))
                        with txl.If(warp == 0), txl.Then():
                            with txl.If(lane == 0), txl.Then():
                                aqk_stage_free.arrive(0)
                iend(tok_epi)

            def gconv(c):
                """G^T (fp32, GACC) -> bf16 GBF."""
                fwait(g_done, 0, c % 2, "st-wait-gdone")
                tok_g = irange("st-gconv")
                txl.ptx[FENCE_AFTER]()
                txl.ptx[LD32x64](*[regs[j] for j in range(64)], tmem(C_GACC))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                st_packed(C_GBF, 0)
                st_packed(C_GBF + 16, 32)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                warrive(g_ready, 0)
                iend(tok_g)

            def enter_frame():
                """S_acc (initial_state, unscaled) -> 2^-S_SHIFT frame, while the u / v_new MMAs run; the v_new
                conversion's wait::st and fence order it before this chunk's delta."""
                with txl.serial(2, unroll=False) as hf:
                    txl.ptx[LD32x64](
                        *[regs[j] for j in range(64)], tmem(C_SACC + 64 * hf)
                    )
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    for j in range(0, 64, 2):
                        pair = txl.local_scalar("uint64")
                        txl.ptx.mov.b64(pair, regs[j], regs[j + 1])
                        txl.ptx.mul.rn.f32x2(
                            pair,
                            pair,
                            txl.cuda.make_float2(
                                txl.float32(2.0**-S_SHIFT), txl.float32(2.0**-S_SHIFT)
                            ),
                        )
                        txl.ptx.mov.b64(regs[j], regs[j + 1], pair)
                    txl.ptx[ST32x64](
                        tmem(C_SACC + 64 * hf), *[regs[j] for j in range(64)]
                    )

            def load_piece_state(idx):
                """Item idx starts a piece: bring its state into TMEM (handoff slot or initial_state), arrive S_ready."""
                iwn = item_word(idx)
                iw1n = item_word1(idx)
                tok_ld = irange("st-load")
                with txl.If(item_src_hand(iwn)):
                    with txl.Then():
                        load_state_hand()
                    with txl.Else():
                        load_state(item_seq(iw1n), item_head(iw1n))
                iend(tok_ld)

            # a piece's first item has its state loaded at the end of the previous piece's last item, ahead of that item's epilogue; only item 0 is loaded here
            with txl.If(n_items > 0), txl.Then():
                load_piece_state(txl.int32(0))
            with txl.serial(n_items, unroll=False) as c:
                s = c % 2
                iw = item_word(c)
                iw1 = item_word1(c)
                hcur = txl.local_scalar("int32", init=item_head(iw1))
                with txl.If(item_last(iw)), txl.Then():
                    with txl.If(c + 1 < n_items), txl.Then():
                        # the next piece's initial state (64 KB, contiguous) into L2 while this item runs
                        iwn = item_word(c + 1)
                        with txl.If(txl.Not(item_src_hand(iwn))), txl.Then():
                            with txl.If(tid == 0), txl.Then():
                                iw1n = item_word1(c + 1)
                                pf_base = txl.local_scalar(
                                    "int32",
                                    init=(item_seq(iw1n) * H + item_head(iw1n))
                                    * (D * D),
                                )
                                txl.ptx["cp.async.bulk.prefetch.L2.global"](
                                    h0.ptr_to([pf_base]), txl.uint32(D * D * 4)
                                )
                gconv(c)
                with (
                    txl.If(tvm.tirx.all(item_first(iw), txl.Not(item_src_hand(iw)))),
                    txl.Then(),
                ):
                    enter_frame()
                fwait(vnew_done, s, (c // 2) % 2, "st-wait-vnew")
                tok_vn = irange("st-vnconv")
                txl.ptx[FENCE_AFTER]()
                txl.ptx[LD32x64](*[regs[j] for j in range(64)], tmem(aa_col(s)))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                st_packed(C_VNBF, 0)
                st_packed(C_VNBF + 16, 32)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                warrive(aa_free, s)
                iend(tok_vn)

                fwait(kA_free, s, (c // 2) % 2, "st-wait-delta")
                txl.ptx[FENCE_AFTER]()
                fwait(dvec_ready, c % 3, (c // 3) % 2, "st-wait-dvec")
                tok_dec = irange("st-decay")

                # decay: the four 32-column quarters as a 2-iteration loop with an unconditional prefetch of the next two, to keep the hot code small
                def ld_q(col, b):
                    txl.ptx[LD32x32](*[regs[b + j] for j in range(32)], tmem(col))

                def proc_q(qi, b):
                    dv4 = txl.alloc_local((4,), "float32")
                    for j in range(0, 32, 4):
                        txl.ptx.ld.shared.v4.f32(
                            dv4[0],
                            dv4[1],
                            dv4[2],
                            dv4[3],
                            dvec.ptr_to([(c % 3) * 128 + 32 * qi + j]),
                        )
                        for m in range(0, 4, 2):
                            pair = txl.local_scalar("uint64")
                            txl.ptx.mov.b64(pair, regs[b + j + m], regs[b + j + m + 1])
                            txl.ptx.mul.rn.ftz.f32x2(
                                pair, pair, txl.cuda.make_float2(dv4[m], dv4[m + 1])
                            )
                            txl.ptx.mov.b64(regs[b + j + m], regs[b + j + m + 1], pair)
                    for j in range(16):
                        txl.assign(
                            packed[j], bf16x2(regs[b + 2 * j], regs[b + 2 * j + 1])
                        )
                    txl.ptx[ST32x16](
                        tmem(C_SBF + 16 * qi), *[packed[j] for j in range(16)]
                    )
                    txl.ptx[ST32x32](
                        tmem(C_SACC + 32 * qi), *[regs[b + j] for j in range(32)]
                    )

                ld_q(C_SACC, 0)
                ld_q(C_SACC + 32, 32)
                with txl.serial(2, unroll=False) as kq_:
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    proc_q(2 * kq_, 0)
                    # the last iteration's prefetch reads the complete, read-only bf16 v_new columns instead, so no in-flight store is re-read
                    ld_q(txl.Select(kq_ == 0, C_SACC + 64, C_VNBF), 0)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    proc_q(2 * kq_ + 1, 32)
                    ld_q(txl.Select(kq_ == 0, C_SACC + 96, C_VNBF), 32)
                warrive(dvec_free, c % 3)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                with txl.If(item_last(iw)):
                    with txl.Then():
                        tok_st = irange("st-store")
                        with txl.If(item_dst_hand(iw)):
                            with txl.Then():
                                store_state_hand()
                            with txl.Else():
                                store_state(item_seq(iw1), hcur)
                        iend(tok_st)
                        with txl.If(c + 1 < n_items), txl.Then():
                            load_piece_state(c + 1)
                    with txl.Else():
                        warrive(S_ready, 0)
                iend(tok_dec)
                epilogue(c, item_tok0(iw), item_nvalid(iw), hcur)

        def intra_body():
            """L = strict-lower(Akk) beta -> T = (I+L)^-1 -> T' = T diag(beta); Aqk_s."""
            q = warp - W_INTRA[0]
            r16 = lane // 4
            c4 = lane % 4
            aqk = txl.alloc_local((32,), "float32")
            acc = txl.alloc_local((8,), "float32")
            a_frag = txl.alloc_local((4,), "uint32")
            b_frag = txl.alloc_local((4,), "uint32")
            aM = txl.alloc_local((4,), "uint32")
            bM = txl.alloc_local((4,), "uint32")
            aP = txl.alloc_local((4,), "uint32")
            bP4 = txl.alloc_local((4,), "uint32")
            bP8 = txl.alloc_local((4,), "uint32")
            tA = [txl.alloc_local((4,), "uint32") for _ in range(3)]
            bL = [txl.alloc_local((4,), "uint32") for _ in range(3)]

            def round_coords(L, reg):
                rep, rem = divmod(reg, 4)
                hh, cc = divmod(rem, 2)
                i = 8 * rep + 2 * c4 + cc
                j = q * 16 + r16 + 8 * hh
                return j, i, L == 0, rep // 2

            def isync():
                txl.ptx.bar.sync(txl.uint32(NB_INTRA), txl.uint32(128))

            def frag_addr(tile, rb, cb):
                return tile.ptr_to(rb + lane % 16, cb + (lane // 16) * 8)

            def ld_b(B, br, bc, dst):
                txl.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                    dst[0], dst[1], dst[2], dst[3], frag_addr(B, br, bc)
                )

            def st_frag(tile, rb, cb, src):
                txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                    frag_addr(tile, rb, cb), src[0], src[1], src[2], src[3]
                )

            def movm(dst, src):
                for z in range(4):
                    txl.ptx.movmatrix.sync.aligned.m8n8.trans.b16(dst[z], src[z])

            def mma_frag(a, b, clear):
                if clear:
                    for z in range(8):
                        txl.assign(acc[z], txl.float32(0.0))
                for nh in range(2):
                    z = 4 * nh
                    txl.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
                        acc[z],
                        acc[z + 1],
                        acc[z + 2],
                        acc[z + 3],
                        a[0],
                        a[1],
                        a[2],
                        a[3],
                        b[2 * nh],
                        b[2 * nh + 1],
                        acc[z],
                        acc[z + 1],
                        acc[z + 2],
                        acc[z + 3],
                    )

            def pack_acc(dst, neg=False, rs=None):
                for z in range(4):
                    v0, v1 = acc[2 * z], acc[2 * z + 1]
                    if rs is not None:
                        v0, v1 = v0 * rs[z % 2], v1 * rs[z % 2]
                    p = bf16x2(v0, v1)
                    txl.assign(
                        dst[z], txl.bitwise_xor(p, txl.uint32(0x80008000)) if neg else p
                    )

            def identity_minus_acc():
                """acc <- I - acc (one FADD per element; identical to negating and adding the identity)."""
                for nh in range(2):
                    z = 4 * nh
                    txl.assign(
                        acc[z],
                        txl.Select(
                            r16 == 8 * nh + 2 * c4, txl.float32(1.0), txl.float32(0.0)
                        )
                        - acc[z],
                    )
                    txl.assign(
                        acc[z + 1],
                        txl.Select(
                            r16 == 8 * nh + 2 * c4 + 1,
                            txl.float32(1.0),
                            txl.float32(0.0),
                        )
                        - acc[z + 1],
                    )
                    txl.assign(
                        acc[z + 2],
                        txl.Select(
                            r16 + 8 == 8 * nh + 2 * c4,
                            txl.float32(1.0),
                            txl.float32(0.0),
                        )
                        - acc[z + 2],
                    )
                    txl.assign(
                        acc[z + 3],
                        txl.Select(
                            r16 + 8 == 8 * nh + 2 * c4 + 1,
                            txl.float32(1.0),
                            txl.float32(0.0),
                        )
                        - acc[z + 3],
                    )

            def add_identity():
                for nh in range(2):
                    z = 4 * nh
                    txl.assign(
                        acc[z],
                        acc[z]
                        + txl.Select(
                            r16 == 8 * nh + 2 * c4, txl.float32(1.0), txl.float32(0.0)
                        ),
                    )
                    txl.assign(
                        acc[z + 1],
                        acc[z + 1]
                        + txl.Select(
                            r16 == 8 * nh + 2 * c4 + 1,
                            txl.float32(1.0),
                            txl.float32(0.0),
                        ),
                    )
                    txl.assign(
                        acc[z + 2],
                        acc[z + 2]
                        + txl.Select(
                            r16 + 8 == 8 * nh + 2 * c4,
                            txl.float32(1.0),
                            txl.float32(0.0),
                        ),
                    )
                    txl.assign(
                        acc[z + 3],
                        acc[z + 3]
                        + txl.Select(
                            r16 + 8 == 8 * nh + 2 * c4 + 1,
                            txl.float32(1.0),
                            txl.float32(0.0),
                        ),
                    )

            def beta_rows(sb, J):
                """Row scales of T' = T diag(beta), carrying the state frame's 2^-S_SHIFT."""
                b0 = txl.local_scalar("float32")
                b1 = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(b0, bsig.ptr_to([sb * 64 + 16 * J + r16]))
                txl.ptx.ld.shared.f32(b1, bsig.ptr_to([sb * 64 + 16 * J + r16 + 8]))
                f = txl.float32(2.0**-S_SHIFT)
                return (b0 * f, b1 * f)

            AQK_SHIFT2 = txl.uint32(
                ((127 + S_SHIFT) << 7) * 0x10001
            )  # bf16x2(2^S_SHIFT, 2^S_SHIFT)
            LT = LTT
            TT = LTT
            TpT = TpT_t
            TpTlo = TpTlo_t
            a_lo = txl.alloc_local((4,), "uint32")

            def pack_acc_rs_hilo(dst_hi, dst_lo, rs):
                """Row-scaled C fragment -> bf16 hi/lo A fragments (T'^T = beta_j T^T); hi + lo carries ~16 mantissa bits because the block chains sum columns of T that cancel to ~(1-beta)^15."""
                for z in range(4):
                    v0 = txl.local_scalar("float32", init=acc[2 * z] * rs[z % 2])
                    v1 = txl.local_scalar("float32", init=acc[2 * z + 1] * rs[z % 2])
                    hi = bf16x2(v0, v1)
                    h0, h1 = unpack(hi)
                    txl.assign(dst_hi[z], hi)
                    txl.assign(dst_lo[z], bf16x2(v0 - h0, v1 - h1))

            with txl.If(q == 3), txl.Then():
                z = txl.uint32(0)
                for m in range(6):
                    n = m * 32 + lane
                    rr, hf = (n % 32) // 2, n % 2
                    UJ = [1, 2, 3, 2, 3, 3][m]
                    UI = [0, 0, 0, 1, 1, 2][m]
                    txl.ptx.st.shared.v4.u32(
                        TpT.ptr_to(16 * UJ + rr, 16 * UI + 8 * hf), z, z, z, z
                    )
                    txl.ptx.st.shared.v4.u32(
                        TpTlo.ptr_to(16 * UJ + rr, 16 * UI + 8 * hf), z, z, z, z
                    )
            txl.ptx.fence.proxy.async_.shared__cta()
            n_items = n_items_local()
            with txl.serial(n_items, unroll=False) as c:
                s = c % 2
                sb = c % 3

                fwait(akk_r, s, (c // 2) % 2, "in-wait-akk")
                with txl.If(q >= 2), txl.Then():
                    fwait(aa_r, s, (c // 2) % 2, "in-wait-aar")
                tok_in = irange("in-transform")
                txl.ptx[FENCE_AFTER]()
                bcol = txl.alloc_local((16,), "float32")
                for I in range(4):
                    for r2 in range(2):
                        txl.ptx.ld.shared.v2.f32(
                            bcol[I * 4 + r2 * 2],
                            bcol[I * 4 + r2 * 2 + 1],
                            bsig.ptr_to([sb * 64 + 16 * I + 8 * r2 + 2 * c4]),
                        )
                dblk = txl.alloc_local((8,), "float32")

                txl.ptx[LD16x2](
                    *[dblk[j] for j in range(8)],
                    tbase
                    + txl.shift_left(txl.Cast("uint32", 16 * (q // 2)), txl.uint32(16))
                    + txl.Cast("uint32", aa_col(s, 0) + 16 * (q % 2)),
                )
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                bdiag = txl.alloc_local((4,), "float32")
                for rep_ in range(2):
                    txl.ptx.ld.shared.v2.f32(
                        bdiag[2 * rep_],
                        bdiag[2 * rep_ + 1],
                        bsig.ptr_to([sb * 64 + 16 * q + 8 * rep_ + 2 * c4]),
                    )

                for reg in range(0, 8, 2):
                    rep_, rem = divmod(reg, 4)
                    hh = rem // 2
                    j = q * 16 + r16 + 8 * hh
                    i = 16 * q + 8 * rep_ + 2 * c4
                    lv = []
                    for cc in range(2):
                        lv.append(
                            txl.local_scalar(
                                "float32",
                                init=txl.Select(
                                    i + cc > j,
                                    dblk[reg + cc] * bdiag[2 * rep_ + cc],
                                    txl.float32(0.0),
                                ),
                            )
                        )
                    hpack = bf16x2(lv[0], lv[1])
                    txl.assign(aM[reg // 2], hpack)
                movm(bM, aM)

                aqk_pk = txl.alloc_local((16,), "uint32")
                ltw = txl.alloc_local((16,), "uint32")
                npk = 0
                fwait(aa_r, s, (c // 2) % 2, "in-wait-aar2")
                txl.ptx[FENCE_AFTER]()
                for L in (1, 0):
                    for hh_ in range(2):
                        txl.ptx[LD16x4](
                            *[aqk[16 * hh_ + j] for j in range(16)],
                            tmem(aa_col(s, 32 * (1 - L)), 16 * hh_),
                        )
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    # packed word reg // 2 = 2 rep + hh holds rows q 16 + r16 + 8 hh, columns 8 rep + 2 c4 .. +1 (mma C order): stored as stmatrix.x4 16x16 blocks at frag_addr(tile, 16 q, 16 n); L^T blocks on or below the diagonal (n <= q) are skipped
                    for reg in range(0, 32, 2):
                        j, i, is_aqk, I = round_coords(L, reg)
                        rep_ = reg // 4
                        if is_aqk:
                            av = [
                                txl.local_scalar(
                                    "float32",
                                    init=txl.Select(
                                        i + cc >= j, aqk[reg + cc], txl.float32(0.0)
                                    ),
                                )
                                for cc in range(2)
                            ]
                            # staged Aqk carries 2^S_SHIFT against the scaled v_new_bf it multiplies (exact bf16 scaling)
                            txl.assign(
                                aqk_pk[npk], hmul2(bf16x2(av[0], av[1]), AQK_SHIFT2)
                            )
                            npk += 1
                            continue
                        lv = []
                        for cc in range(2):
                            bidx = I * 4 + (rep_ % 2) * 2 + cc
                            lv.append(
                                txl.local_scalar(
                                    "float32", init=aqk[reg + cc] * bcol[bidx]
                                )
                            )
                        txl.assign(ltw[reg // 2], bf16x2(lv[0], lv[1]))
                    if L == 1:
                        for n_ in range(1, 4):
                            with txl.If(q < n_), txl.Then():
                                txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                                    frag_addr(LT, 16 * q, 16 * n_),
                                    ltw[4 * n_],
                                    ltw[4 * n_ + 1],
                                    ltw[4 * n_ + 2],
                                    ltw[4 * n_ + 3],
                                )

                mma_frag(aM, bM, True)
                pack_acc(aP)
                movm(b_frag, aP)
                mma_frag(aP, b_frag, True)
                pack_acc(a_frag)
                movm(bP4, a_frag)
                mma_frag(a_frag, bP4, True)
                pack_acc(a_frag)
                movm(bP8, a_frag)
                for z in range(4):
                    lo, hi = unpack(aP[z])
                    txl.assign(acc[2 * z], lo)
                    txl.assign(acc[2 * z + 1], hi)
                add_identity()
                pack_acc(a_frag)
                mma_frag(a_frag, bP4, False)
                pack_acc(a_frag)
                mma_frag(a_frag, bP8, False)
                pack_acc(a_frag)
                movm(b_frag, a_frag)
                # T = Q - M Q as one accumulate onto Q with the B fragment negated (sign flips are exact)
                for z in range(4):
                    txl.assign(
                        b_frag[z], txl.bitwise_xor(b_frag[z], txl.uint32(0x80008000))
                    )
                mma_frag(aM, b_frag, False)
                pack_acc(tA[0])

                movm(b_frag, tA[0])
                mma_frag(aM, b_frag, True)
                for z in range(4):
                    th0, th1 = unpack(tA[0][z])
                    txl.assign(acc[2 * z], acc[2 * z] + th0)
                    txl.assign(acc[2 * z + 1], acc[2 * z + 1] + th1)
                identity_minus_acc()
                pack_acc(aP)
                movm(bP4, aP)
                mma_frag(tA[0], bP4, True)
                for z in range(4):
                    th0, th1 = unpack(tA[0][z])
                    txl.assign(acc[2 * z], acc[2 * z] + th0)
                    txl.assign(acc[2 * z + 1], acc[2 * z + 1] + th1)
                pack_acc(tA[0])
                with txl.If(c >= 1), txl.Then():
                    fwait(vnew_done, (c + 1) % 2, ((c - 1) // 2) % 2, "in-wait-vnew")
                    txl.ptx.fence.proxy.async_.shared__cta()
                st_frag(TT, 16 * q, 16 * q, tA[0])
                # acc holds +T here: scale its rows by +beta
                rs = beta_rows(sb, q)
                pack_acc_rs_hilo(a_frag, a_lo, rs)
                st_frag(TpT, 16 * q, 16 * q, a_frag)
                st_frag(TpTlo, 16 * q, 16 * q, a_lo)
                isync()

                # beta is constant across the off-diagonal block-column solve: reuse the diagonal's row scales at every chain step
                chain_rs = (txl.float32(0.0) - rs[0], txl.float32(0.0) - rs[1])

                # software-pipelined chain: step s+1's ldmatrix operands are fetched during step s; unconsumed fetches are clamped to an already written block
                bL2 = [txl.alloc_local((4,), "uint32") for _ in range(3)]
                b_frag2 = txl.alloc_local((4,), "uint32")

                def ld_ops(I_rt, dbL, db):
                    # unconsumed operands are clamped onto the diagonal block of the same column (always written)
                    I_c = txl.min(I_rt, 3)
                    for n in range(3):
                        ld_b(LT, 16 * txl.min(q + n, I_c), 16 * I_c, dbL[n])
                    ld_b(TT, 16 * I_c, 16 * I_c, db)

                with txl.If(q < 3), txl.Then():
                    ld_ops(q + 1, bL, b_frag)
                with txl.serial(3 - q, unroll=False) as step:
                    I_rt = q + 1 + step
                    mma_frag(tA[0], bL[0], True)
                    for n in range(1, 3):
                        with txl.If(n <= step), txl.Then():
                            mma_frag(tA[n], bL[n], False)
                    pack_acc(a_frag)
                    mma_frag(a_frag, b_frag, True)
                    ld_ops(I_rt + 1, bL2, b_frag2)
                    with txl.If(step == 0), txl.Then():
                        pack_acc(tA[1], neg=True)
                    with txl.If(step == 1), txl.Then():
                        pack_acc(tA[2], neg=True)
                    pack_acc_rs_hilo(a_frag, a_lo, chain_rs)
                    st_frag(TpT, 16 * q, 16 * I_rt, a_frag)
                    st_frag(TpTlo, 16 * q, 16 * I_rt, a_lo)
                    for n in range(3):
                        for z in range(4):
                            txl.assign(bL[n][z], bL2[n][z])
                    for z in range(4):
                        txl.assign(b_frag[z], b_frag2[z])

                isync()
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx[FENCE_BEFORE]()
                warrive(T_ready, s)
                iend(tok_in)

                with txl.If(c >= 1), txl.Then():
                    fwait(o_done, 0, (c + 1) % 2, "in-wait-odone")
                    fwait(aqk_stage_free, 0, (c + 1) % 2, "in-wait-aqkstage")
                    txl.ptx.fence.proxy.async_.shared__cta()
                for n_ in range(4):
                    txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        frag_addr(Aqk_s, 16 * q, 16 * n_),
                        aqk_pk[4 * n_],
                        aqk_pk[4 * n_ + 1],
                        aqk_pk[4 * n_ + 2],
                        aqk_pk[4 * n_ + 3],
                    )
                txl.ptx.fence.proxy.async_.shared__cta()
                warrive(Aqk_ready, 0)

        def norm_body():
            """Use the otherwise idle MMA-I warp for row norms and beta."""
            n_items = n_items_local()
            with txl.serial(n_items, unroll=False) as c:
                iw = item_word(c)
                iw1 = item_word1(c)
                hcur = item_head(iw1)
                nvalid = item_nvalid(iw)
                s = c % 2
                with txl.If(item_first(iw)), txl.Then():
                    txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))
                tok_norm = irange("nr-work")
                for hf in range(2):
                    row = lane + 32 * hf
                    for which in range(2):
                        n0 = txl.local_scalar("float32")
                        n1 = txl.local_scalar("float32")
                        norm = txl.local_scalar("float32")
                        txl.ptx.ld.shared.f32(
                            n0, rsq.ptr_to([s * 256 + which * 128 + row])
                        )
                        txl.ptx.ld.shared.f32(
                            n1, rsq.ptr_to([s * 256 + which * 128 + 64 + row])
                        )
                        txl.ptx.rsqrt.approx.ftz.f32(norm, n0 + n1 + txl.float32(EPS))
                        fac = norm * scale if which == 0 else norm
                        fac = txl.Select(row < nvalid, fac, txl.float32(0.0))
                        txl.ptx.st.shared.f32(
                            rsq.ptr_to([s * 256 + which * 128 + row]), fac
                        )
                    rawb = txl.local_scalar("uint16")
                    txl.ptx.ld.shared.b16(
                        rawb, beta_s.ptr_to([s * 512 + row * 8 + hcur % 8])
                    )
                    bv = txl.reinterpret(
                        "float32",
                        txl.shift_left(txl.Cast("uint32", rawb), txl.uint32(16)),
                    )
                    sg = txl.idioms.sigmoid_tanh_approx_f32(bv)
                    txl.ptx.st.shared.f32(
                        bsig.ptr_to([(c % 3) * 64 + row]),
                        txl.Select(row < nvalid, sg, txl.float32(0.0)),
                    )
                iend(tok_norm)
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))

        def prep_body():
            """Gate/cumsum/normalize/scaled-copy producer: thread = 4 tokens x 8 channels."""
            w = warp - W_PREP[0]
            I = w // 2
            half = w % 2
            cg8 = lane % 8
            j4 = lane // 8
            d0 = (half * 8 + cg8) * 8
            tl0 = I * 16 + 4 * j4
            th = I // 2

            cX = txl.alloc_local((16,), "float32")
            cY = txl.alloc_local((16,), "float32")
            sqx = txl.alloc_local((4,), "float32")
            sqy = txl.alloc_local((4,), "float32")
            qraw = txl.alloc_local((4,), "uint32")
            kraw = txl.alloc_local((4,), "uint32")
            ra = txl.local_scalar("uint32")
            rb = txl.local_scalar("uint32")

            # roff[i]: byte offset of this thread's row tl0 + i (channels d0..d0+7) in a [64][128] tile; in the [128][128] kq tile the k row sits at roff + half * 8 KB + th * 4 KB (same swizzle) and the q row KQ_Q_OFF further
            sh0 = txl.local_scalar(
                "uint32", init=txl.cuda.cvta_generic_to_shared(g_t[0].ptr_to(0, 0))
            )
            roff = [
                txl.local_scalar(
                    "uint32",
                    init=txl.cuda.cvta_generic_to_shared(g_t[0].ptr_to(tl0 + i, d0))
                    - sh0,
                )
                for i in range(4)
            ]
            KQ_K_OFF = txl.local_scalar(
                "uint32", init=txl.Cast("uint32", half * 8192 + th * KQ_Q_OFF)
            )
            KQ_QQ_OFF = txl.local_scalar("uint32", init=KQ_K_OFF + txl.uint32(KQ_Q_OFF))

            def tptr_off(view, r):
                return txl.ptx.addr(view.ptr_to(0, 0), r)

            RINT_C, RINT_BITS = 12582912.0, 0x4B400000

            def add_rint_c(x):
                t = txl.local_scalar("float32")
                txl.ptx.add.rn.f32(
                    t, txl.local_scalar("float32", init=x), txl.float32(RINT_C)
                )
                return t

            def rint(x):
                r = txl.local_scalar("float32")
                txl.ptx.sub.rn.f32(r, add_rint_c(x), txl.float32(RINT_C))
                return r

            def shfl_xor(x, xr):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(
                    peer,
                    txl.reinterpret("uint32", x),
                    txl.uint32(xr),
                    txl.uint32(31),
                    txl.uint32(0xFFFFFFFF),
                )
                return txl.reinterpret("float32", peer)

            def shfl_up(x, delta):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.up.b32(
                    peer,
                    txl.reinterpret("uint32", x),
                    txl.uint32(delta),
                    txl.uint32(0),
                    txl.uint32(0xFFFFFFFF),
                )
                return txl.reinterpret("float32", peer)

            def ld4u(dst, off, ptr):
                txl.ptx.ld.shared.v4.b32(
                    dst[off], dst[off + 1], dst[off + 2], dst[off + 3], ptr
                )

            def st4u(ptr, src, off):
                txl.ptx.st.shared.v4.b32(
                    ptr, src[off], src[off + 1], src[off + 2], src[off + 3]
                )

            def rcp(x):
                r = txl.local_scalar("float32")
                txl.ptx.rcp.approx.ftz.f32(r, x)
                return r

            def fadd2f(a0, a1, b0, b1):
                """Two independent RN fp32 additions in one packed instruction."""
                packed = txl.local_scalar("uint64")
                o0 = txl.local_scalar("float32")
                o1 = txl.local_scalar("float32")
                txl.ptx.add.rn.f32x2(
                    packed,
                    txl.cuda.make_float2(a0, a1),
                    txl.cuda.make_float2(b0, b1),
                )
                txl.ptx.mov.b64(o0, o1, packed)
                return o0, o1

            def fmul2f(a0, a1, b0, b1):
                """Two independent RN fp32 products in one packed instruction (no .ftz: subnormal results survive)."""
                packed = txl.local_scalar("uint64")
                o0 = txl.local_scalar("float32")
                o1 = txl.local_scalar("float32")
                txl.ptx.mul.rn.f32x2(
                    packed,
                    txl.cuda.make_float2(a0, a1),
                    txl.cuda.make_float2(b0, b1),
                )
                txl.ptx.mov.b64(o0, o1, packed)
                return o0, o1

            def ffma2f(a0, a1, b0, b1, c0, c1):
                """Two independent RN fp32 FMAs in one packed instruction."""
                packed = txl.local_scalar("uint64")
                o0 = txl.local_scalar("float32")
                o1 = txl.local_scalar("float32")
                txl.ptx.fma.rn.f32x2(
                    packed,
                    txl.cuda.make_float2(a0, a1),
                    txl.cuda.make_float2(b0, b1),
                    txl.cuda.make_float2(c0, c1),
                )
                txl.ptx.mov.b64(o0, o1, packed)
                return o0, o1

            adt2 = txl.alloc_local((8,), "float32")
            a_e2 = txl.local_scalar("float32")

            def head_consts(hh):
                """Per-head gate constants (exp(A_log)/2 and its dt_bias products) -> adt_s -> registers."""
                with txl.If(I == 0), txl.Then():
                    a_h = txl.local_scalar("float32")
                    txl.ptx.ld.global_.f32(a_h, A_log.ptr_to([hh]))
                    a_e2v = txl.local_scalar(
                        "float32",
                        init=ex2(a_h * txl.float32(RCP_LN2)) * txl.float32(0.5),
                    )
                    with txl.If(j4 == 0), txl.Then():
                        dtb = txl.alloc_local((8,), "float32")
                        txl.ptx.ld.global_.v4.f32(
                            dtb[0],
                            dtb[1],
                            dtb[2],
                            dtb[3],
                            dt_bias.ptr_to([hh * D + d0]),
                        )
                        txl.ptx.ld.global_.v4.f32(
                            dtb[4],
                            dtb[5],
                            dtb[6],
                            dtb[7],
                            dt_bias.ptr_to([hh * D + d0 + 4]),
                        )
                        txl.ptx.st.shared.v4.f32(
                            adt_s.ptr_to([d0]),
                            a_e2v * dtb[0],
                            a_e2v * dtb[1],
                            a_e2v * dtb[2],
                            a_e2v * dtb[3],
                        )
                        txl.ptx.st.shared.v4.f32(
                            adt_s.ptr_to([d0 + 4]),
                            a_e2v * dtb[4],
                            a_e2v * dtb[5],
                            a_e2v * dtb[6],
                            a_e2v * dtb[7],
                        )
                    with txl.If(tvm.tirx.all(half == 0, lane == 0)), txl.Then():
                        txl.ptx.st.shared.f32(adt_s.ptr_to([128]), a_e2v)
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))
                txl.ptx.ld.shared.v4.f32(
                    adt2[0], adt2[1], adt2[2], adt2[3], adt_s.ptr_to([d0])
                )
                txl.ptx.ld.shared.v4.f32(
                    adt2[4], adt2[5], adt2[6], adt2[7], adt_s.ptr_to([d0 + 4])
                )
                txl.ptx.ld.shared.f32(a_e2, adt_s.ptr_to([128]))

            n_items = n_items_local()
            with txl.serial(n_items, unroll=False) as c:
                s = c % 2
                sb = c % 3
                iw = item_word(c)
                iw1 = item_word1(c)
                hcur = txl.local_scalar("int32", init=item_head(iw1))
                nvalid = txl.local_scalar("int32", init=item_nvalid(iw))

                with txl.If(item_first(iw)), txl.Then():
                    head_consts(hcur)
                fwait(ring_full, s, (c // 2) % 2, "pr-wait-ring")
                tok_pa = irange("pr-phaseA")
                kqs, gs_ = kq_t[s], g_t[s]

                g8 = txl.alloc_local((4,), "uint32")
                for m in range(8):
                    txl.assign(cY[8 + m], txl.float32(0.0))
                txl.assign(ra, roff[0])
                txl.assign(rb, roff[1])

                with txl.serial(2, unroll=False) as it:
                    tokA = tl0 + 2 * it
                    for a in range(2):
                        r = ra if a == 0 else rb
                        va = tokA + a < nvalid
                        gate_c = txl.local_scalar(
                            "float32",
                            init=txl.Select(va, txl.float32(GATE_C), txl.float32(0.0)),
                        )
                        ld4u(g8, 0, tptr_off(gs_, r))
                        ld4u(qraw, 0, tptr_off(kqs, r + KQ_QQ_OFF))
                        ld4u(kraw, 0, tptr_off(kqs, r + KQ_K_OFF))
                        sqa = txl.local_scalar("uint32", init=txl.uint32(0))
                        ska = txl.local_scalar("uint32", init=txl.uint32(0))
                        for p in range(4):
                            g0, g1 = unpack(g8[p])
                            x0, x1 = ffma2f(
                                g0, g1, a_e2, a_e2, adt2[2 * p], adt2[2 * p + 1]
                            )
                            th0 = txl.local_scalar("float32")
                            th1 = txl.local_scalar("float32")
                            txl.ptx.tanh.approx.f32(th0, x0)
                            txl.ptx.tanh.approx.f32(th1, x1)
                            # rows past the sequence end get a zero log-decay: the mask is the FMA's constant
                            gl0, gl1 = ffma2f(th0, th1, gate_c, gate_c, gate_c, gate_c)

                            gl0, gl1 = fadd2f(
                                cY[8 * (1 - a) + 2 * p],
                                cY[8 * (1 - a) + 2 * p + 1],
                                gl0,
                                gl1,
                            )
                            txl.assign(cY[a * 8 + 2 * p], gl0)
                            txl.assign(cY[a * 8 + 2 * p + 1], gl1)
                            txl.assign(sqa, hfma2(qraw[p], qraw[p], sqa))
                            txl.assign(ska, hfma2(kraw[p], kraw[p], ska))
                        q0, q1 = unpack(sqa)
                        k0, k1 = unpack(ska)
                        txl.assign(sqy[a], q0 + q1)
                        txl.assign(sqy[2 + a], k0 + k1)
                    with txl.If(it == 0), txl.Then():
                        for m in range(16):
                            txl.assign(cX[m], cY[m])
                        for m in range(4):
                            txl.assign(sqx[m], sqy[m])
                        txl.assign(ra, roff[2])
                        txl.assign(rb, roff[3])

                xin = [txl.local_scalar("float32", init=cY[8 + m]) for m in range(8)]
                for step in (1, 2):
                    for m in range(8):
                        y = shfl_up(xin[m], 8 * step)
                        txl.assign(xin[m], txl.Select(j4 >= step, xin[m] + y, xin[m]))
                with txl.If(j4 == 3), txl.Then():
                    txl.ptx.st.shared.v4.f32(
                        tot.ptr_to([I * 128 + d0]), xin[0], xin[1], xin[2], xin[3]
                    )
                    txl.ptx.st.shared.v4.f32(
                        tot.ptr_to([I * 128 + d0 + 4]), xin[4], xin[5], xin[6], xin[7]
                    )
                excl = [
                    txl.local_scalar("float32", init=xin[m] - cY[8 + m])
                    for m in range(8)
                ]

                cur = [sqx[0], sqx[1], sqy[0], sqy[1], sqx[2], sqx[3], sqy[2], sqy[3]]
                for xr in (4, 2, 1):
                    nb = len(cur) // 2
                    hi = txl.Cast("bool", txl.bitwise_and(lane, txl.int32(xr)))
                    nxt = []
                    for m in range(nb):
                        send = txl.local_scalar(
                            "float32", init=txl.Select(hi, cur[m], cur[nb + m])
                        )
                        keep = txl.Select(hi, cur[nb + m], cur[m])
                        nxt.append(
                            txl.local_scalar("float32", init=keep + shfl_xor(send, xr))
                        )
                    cur = nxt
                txl.ptx.st.shared.f32(
                    rsq.ptr_to(
                        [s * 256 + (cg8 // 4) * 128 + half * 64 + tl0 + cg8 % 4]
                    ),
                    cur[0],
                )
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))

                iend(tok_pa)
                tok_pc = irange("pr-consts")

                cp = half * 32 + lane
                tt = txl.alloc_local((8,), "float32")
                for J in range(4):
                    txl.ptx.ld.shared.v2.f32(
                        tt[2 * J], tt[2 * J + 1], tot.ptr_to([J * 128 + 2 * cp])
                    )
                # adjacent channels share the prefix/rounding chain: packed f32x2 ops keep each channel's operation order
                pref = [[txl.float32(0.0)] + [None] * 4 for _ in range(2)]
                for J in range(4):
                    pref[0][J + 1], pref[1][J + 1] = fadd2f(
                        pref[0][J], pref[1][J], tt[2 * J], tt[2 * J + 1]
                    )

                def rint2(x0, x1):
                    y0, y1 = fadd2f(x0, x1, txl.float32(RINT_C), txl.float32(RINT_C))
                    return fadd2f(y0, y1, txl.float32(-RINT_C), txl.float32(-RINT_C))

                e = [pref[0][4], pref[1][4]]
                eps_arg = fadd2f(
                    e[0], e[1], txl.float32(CLAMP_F + 0.5), txl.float32(CLAMP_F + 0.5)
                )
                eps_round = rint2(*eps_arg)
                eps = [
                    txl.local_scalar("float32", init=txl.min(x, txl.float32(0.0)))
                    for x in eps_round
                ]
                e_cl = fadd2f(e[0], e[1], -eps[0], -eps[1])
                r0 = rint2(pref[0][2] * txl.float32(0.5), pref[1][2] * txl.float32(0.5))
                ref = [
                    txl.local_scalar("float32", init=txl.Select(th == 0, r0[m], eps[m]))
                    for m in range(2)
                ]
                r = [pref[m][3] for m in range(2)]
                for m in range(2):
                    for J in (2, 1, 0):
                        r[m] = txl.Select(I == J, pref[m][J], r[m])
                lane_off = fadd2f(r[0], r[1], -ref[0], -ref[1])
                ca = [
                    txl.local_scalar("float32", init=txl.max(x, txl.float32(-126.0)))
                    for x in ref
                ]
                cb_arg = fadd2f(ref[0], ref[1], -ca[0], -ca[1])
                cb = [
                    txl.local_scalar("float32", init=txl.max(x, txl.float32(-126.0)))
                    for x in cb_arg
                ]
                dx = fadd2f(eps[0], eps[1], -r0[0], -r0[1])
                da = [
                    txl.local_scalar("float32", init=txl.max(x, txl.float32(-126.0)))
                    for x in dx
                ]
                db_arg = fadd2f(dx[0], dx[1], -da[0], -da[1])
                db = [
                    txl.local_scalar("float32", init=txl.max(x, txl.float32(-126.0)))
                    for x in db_arg
                ]
                scal = [ca, cb, da, db]
                lane_c = txl.alloc_local((4,), "uint32")
                # the first chunk of a piece loaded from initial_state still sees an unscaled S_bf
                shift = txl.Select(
                    tvm.tirx.all(item_first(iw), txl.Not(item_src_hand(iw))),
                    txl.uint32(0),
                    txl.uint32(S_SHIFT),
                )
                for z in range(4):
                    nr = fadd2f(
                        scal[z][0], scal[z][1], txl.float32(RINT_C), txl.float32(RINT_C)
                    )
                    # 2^ca also carries the state frame's 2^S_SHIFT (ca >= -126, so it stays normal)
                    power = [
                        txl.reinterpret(
                            "float32",
                            txl.shift_left(
                                txl.reinterpret("uint32", nr[m])
                                - txl.uint32(RINT_BITS - 127)
                                + (shift if z == 0 else txl.uint32(0)),
                                txl.uint32(23),
                            ),
                        )
                        for m in range(2)
                    ]
                    txl.assign(lane_c[z], bf16x2(power[0], power[1]))
                # the q2/k2 clamp product cc = 2^S_SHIFT * 2^ca * 2^cb is per-channel, so it is formed once and broadcast; the kA factors 2^da, 2^db stay separate (their fused product can underflow where the staged one does not)
                cc_lane = txl.local_scalar("uint32", init=hmul2(lane_c[0], lane_c[1]))
                lane_b = [cc_lane, lane_c[2], lane_c[3]]

                with txl.If(I == 0), txl.Then():
                    with txl.If(c >= 3), txl.Then():
                        fwait(dvec_free, c % 3, ((c - 3) // 3) % 2, "pr-wait-dvecfree")
                    # a piece ending in final_state leaves the 2^-S_SHIFT frame in its last decay
                    unscale = txl.Select(
                        tvm.tirx.all(item_last(iw), txl.Not(item_dst_hand(iw))),
                        txl.float32(S_SHIFT),
                        txl.float32(0.0),
                    )
                    txl.ptx.st.shared.v2.f32(
                        dvec.ptr_to([(c % 3) * 128 + 2 * cp]),
                        ex2(e_cl[0] + unscale),
                        ex2(e_cl[1] + unscale),
                    )
                    warrive(dvec_ready, c % 3)

                cst4 = txl.alloc_local((12,), "uint32")
                for p in range(4):
                    src_lane = txl.Cast("uint32", 4 * cg8 + p)
                    for m in range(2):
                        v = txl.local_scalar("uint32")
                        txl.ptx.shfl_sync.idx.b32(
                            v,
                            txl.reinterpret("uint32", lane_off[m]),
                            src_lane,
                            txl.uint32(31),
                            txl.uint32(0xFFFFFFFF),
                        )
                        txl.assign(
                            excl[2 * p + m],
                            excl[2 * p + m] + txl.reinterpret("float32", v),
                        )
                    for z in range(3):
                        v = txl.local_scalar("uint32")
                        txl.ptx.shfl_sync.idx.b32(
                            v,
                            lane_b[z],
                            src_lane,
                            txl.uint32(31),
                            txl.uint32(0xFFFFFFFF),
                        )
                        txl.assign(cst4[3 * p + z], v)

                # all tot reads finish before reuse, and the helper's fp32 norm factors and beta are visible before phase B publishes rfull
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))
                iend(tok_pc)

                with txl.If(c >= 1), txl.Then():
                    fwait(aa_r, (c + 1) % 2, ((c - 1) // 2) % 2, "pr-wait-aar")
                    fwait(oi_done, 0, (c + 1) % 2, "pr-wait-oidone")
                    txl.ptx.fence.proxy.async_.shared__cta()
                tok_pb = irange("pr-phaseB")
                xq = txl.alloc_local((4,), "uint32")
                xk = txl.alloc_local((4,), "uint32")
                q2v = txl.alloc_local((4,), "uint32")
                k2v = txl.alloc_local((4,), "uint32")
                kf = txl.alloc_local((4,), "uint32")
                kaa = txl.alloc_local((4,), "uint32")
                rq2 = txl.alloc_local((2,), "uint32")
                rk2 = txl.alloc_local((2,), "uint32")

                txl.assign(ra, roff[0])
                txl.assign(rb, roff[1])
                with txl.serial(2, unroll=False) as it:
                    tokA = tl0 + 2 * it
                    txl.ptx.ld.shared.v2.b32(
                        rq2[0], rq2[1], rsq.ptr_to([s * 256 + tokA])
                    )
                    txl.ptx.ld.shared.v2.b32(
                        rk2[0], rk2[1], rsq.ptr_to([s * 256 + 128 + tokA])
                    )
                    for a in range(2):
                        r = ra if a == 0 else rb
                        ld4u(qraw, 0, tptr_off(kqs, r + KQ_QQ_OFF))
                        ld4u(kraw, 0, tptr_off(kqs, r + KQ_K_OFF))
                        fq = txl.reinterpret("float32", rq2[a])
                        fk = txl.reinterpret("float32", rk2[a])
                        # q / k decorations in fp32 with a single rounding to bf16 (norm factor, decay factor and
                        # their product; 1/Ex from its own fp32 reciprocal).  Packed bf16 chains here cost up to four
                        # roundings per operand, and the former one-step bf16 Newton 1/Ex was biased by -0.14%.
                        # mul.rn.f32x2 keeps subnormal products: late half-1 targets of strong-decay chunks sit
                        # near 2^-120 and must not be flushed.
                        for p in range(4):
                            x0, x1 = fadd2f(
                                cX[a * 8 + 2 * p],
                                cX[a * 8 + 2 * p + 1],
                                excl[2 * p],
                                excl[2 * p + 1],
                            )
                            ex0, ex1 = ex2(x0), ex2(x1)
                            rx0, rx1 = rcp(ex0), rcp(ex1)
                            q0, q1 = unpack(qraw[p])
                            k0, k1 = unpack(kraw[p])
                            qn0, qn1 = fmul2f(q0, q1, fq, fq)
                            kn0, kn1 = fmul2f(k0, k1, fk, fk)
                            txl.assign(xq[p], bf16x2(*fmul2f(qn0, qn1, ex0, ex1)))
                            txl.assign(xk[p], bf16x2(*fmul2f(kn0, kn1, ex0, ex1)))
                            txl.assign(kf[p], bf16x2(*fmul2f(kn0, kn1, rx0, rx1)))
                        for p in range(4):
                            # q2 and k2 share the fused power-of-two clamp product cc = cst4[3 p]
                            txl.assign(q2v[p], hmul2(xq[p], cst4[3 * p]))
                            txl.assign(k2v[p], hmul2(xk[p], cst4[3 * p]))
                        st4u(tptr_off(q2t, r), q2v, 0)
                        st4u(tptr_off(k2t, r), k2v, 0)

                        st4u(tptr_off(kqs, r + KQ_K_OFF), xk, 0)
                        st4u(tptr_off(kqs, r + KQ_QQ_OFF), xq, 0)
                        with txl.If(th == 0):
                            with txl.Then():
                                for p in range(4):
                                    txl.assign(
                                        kaa[p],
                                        hmul2(
                                            hmul2(kf[p], cst4[3 * p + 1]),
                                            cst4[3 * p + 2],
                                        ),
                                    )
                                st4u(tptr_off(g_t[s], r), kf, 0)
                                st4u(tptr_off(kA[s], r), kaa, 0)
                            with txl.Else():
                                st4u(tptr_off(kA[s], r), kf, 0)
                    with txl.If(it == 0), txl.Then():
                        for m in range(16):
                            txl.assign(cX[m], cY[m])
                        txl.assign(ra, roff[2])
                        txl.assign(rb, roff[3])
                txl.ptx.fence.proxy.async_.shared__cta()
                iend(tok_pb)
                warrive(rfull, s)

        with r_state:
            state_body()
        with wg1:
            with r_ld:
                load_body()
            with r_mmi:
                norm_body()
            with r_mmc:
                mmc_body()
            with r_rnd:
                rounds_body((0, 1))
        with r_intra:
            intra_body()
        with r_prep:
            prep_body()

        txl.cuda.cta_sync()
        with txl.If(warp == 0), txl.Then():
            txl.ptx[TMEM_RELINQ]()
            txl.ptx[TMEM_DEALLOC](tbase, txl.uint32(N_COLS))

    kda_fwd.__annotations__ = {
        "q_map": txl.TensorMap,
        "k_map": txl.TensorMap,
        "v_map": txl.TensorMap,
        "g_map": txl.TensorMap,
        "beta_map": txl.TensorMap,
        "o_map": txl.TensorMap,
        "out": txl.gptr[txl.bf16],
        "A_log": txl.gptr[txl.f32, (H,)],
        "dt_bias": txl.gptr[txl.f32, (H * D,)],
        "h0": txl.gptr[txl.f32],
        "final_state": txl.gptr[txl.f32],
        "hand": txl.gptr[txl.f32],
        "flags": txl.gptr[txl.i32],
        "items": txl.gptr[txl.u32],
        "item_counts": txl.gptr[txl.i32],
        "num_ctas": txl.i32,
        "scale": txl.f32,
    }
    return txl.kernel(warps=NWARPS, arch=arch, min_blocks_per_sm=1, grid="num_ctas")(
        kda_fwd
    )
