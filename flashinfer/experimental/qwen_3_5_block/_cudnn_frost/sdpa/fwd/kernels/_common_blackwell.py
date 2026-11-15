# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import NamedTuple, Optional

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_WORK_PREFIX_OFF, THD_WORK_META_WORDS, THD_REMAP_OFF
import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm
from cutlass._mlir.dialects import arith

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.scheduler import (
    SCHED_LPT,
    SCHED_LPT_L2,
    SCHED_NATURAL,
    lpt_tile_coords,
    lpt_l2_tile_coords,
)

# KvLoopBounds / compute_kv_loop_bounds moved to frost.tile_dsl.mask (the
# backward needs the same tile-level bounds); re-exported here so the twelve
# prefill kernels that import them from this module keep working.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mask import (  # noqa: F401
    MASK_CAUSAL,
    MASK_PADDED,
    MASK_SWA,
    KvLoopBounds,
    compute_kv_loop_bounds,
    _div_up,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import MBarrier, Producer, Scope
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import fmul2, ffma2, opaque_f32_zero, fmax_f32
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_shared_v4, st_global_v4, st_shared_v4, tma_load_tile

# The O-swizzle selector lives on the base config line (config_sm107 carries a
# byte-identical copy).  Importing it from config_sm100 keeps this cross-arch
# module off the Rubin package (engine-contract S8: a shared module never leans
# on one arch's package) and is cycle-free -- config_sm100 imports only
# tile_dsl.constants, never this package.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import o_swz_bytes as _o_swz_bytes


@cute.jit
def sanitize_mxfp8_thd_v_sf_padding(sv_sf, kv_tile_idx, seqlen_kv):
    """Replace fully padded V scale blocks before block-scaled BMM2.

    F8_128x4 stores four 32-token V scale blocks in each 4-byte row group.
    The packed THD tail may leave whole blocks outside the sequence; their data
    TMA-loads as zero, but an E8M0 NaN scale would still make ``0 * NaN`` poison
    the accumulator.  Preserve blocks containing valid tokens and replace only
    the fully padded scale bytes with the finite scale 1.0 (0x7f).
    """
    remaining = seqlen_kv - kv_tile_idx * cutlass.Int32(128)
    valid_blocks = (remaining + cutlass.Int32(31)) // cutlass.Int32(32)
    if valid_blocks < cutlass.Int32(4):
        lane = cute.arch.thread_idx()[0] % cutlass.Int32(32)
        keep_mask = (cutlass.Int32(1) << (valid_blocks * cutlass.Int32(8))) - cutlass.Int32(1)
        fill_mask = cutlass.Int32(0x7F7F7F7F) & ~keep_mask
        for row_group in cutlass.range_constexpr(4):
            byte_offset = lane * cutlass.Int32(16) + cutlass.Int32(row_group * 4)
            word = Pointer(sv_sf.base.subview(byte_offset).data_ptr(), dtype=cutlass.Int32)
            word.store((word.load(alignment=4) & keep_mask) | fill_mask, alignment=4)
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def sanitize_mxfp8_v_sf_padding_pair(sv_sf, kv_tile_idx, seqlen_kv, peer_cta_id, cta_group: cutlass.Constexpr[int]):
    """Replace the fully padded V scale blocks in this CTA's slab and, under ``cta_group`` 2, in the peer CTA's slab at
    the same offset through cluster-mapped loads and stores.

    Called by the leader's MMA warp after its ``mb_v_full`` wait and before the ``tcgen05.cp`` that reads each CTA's
    slab into that CTA's tensor memory; the peer runs no MMA warp of its own.
    """
    sanitize_mxfp8_thd_v_sf_padding(sv_sf, kv_tile_idx, seqlen_kv)
    if cutlass.const_expr(cta_group == 2):
        remaining = seqlen_kv - kv_tile_idx * cutlass.Int32(128)
        valid_blocks = (remaining + cutlass.Int32(31)) // cutlass.Int32(32)
        if valid_blocks < cutlass.Int32(4):
            lane = cute.arch.thread_idx()[0] % cutlass.Int32(32)
            keep_mask = (cutlass.Int32(1) << (valid_blocks * cutlass.Int32(8))) - cutlass.Int32(1)
            fill_mask = cutlass.Int32(0x7F7F7F7F) & ~keep_mask
            for row_group in cutlass.range_constexpr(4):
                byte_offset = lane * cutlass.Int32(16) + cutlass.Int32(row_group * 4)
                peer_word = nvvm.mapa(sv_sf.base.subview(byte_offset), peer_cta_id).ir_value()
                old = nvvm.inline_ptx("ld.shared::cluster.b32 $0, [$1];", write_only_types=[cutlass.Int32], read_only_args=[peer_word])
                nvvm.inline_ptx("st.shared::cluster.b32 [$0], $1;", read_only_args=[peer_word, (old & keep_mask) | fill_mask])
            nvvm.inline_ptx("fence.proxy.async.shared::cluster;")
            nvvm.bar_warp_sync(cute.arch.FULL_MASK)


class Bars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object

    mb_bmm1_done: object
    mb_bmm2_done: object
    mb_bmm2_ready: object

    mb_stat_full: object
    mb_stat_empty: object

    # Per-sub-tile "final stats consumed" gate (correction -> MMA, cross-CTA
    # to the leader under cga2).  The per-tile (total_max, total_sum) publish
    # rides the HEAD of the S_acc slot (LAYOUT.STATS_OFF + qs*STATS_STRIDE),
    # so the NEXT tile's prologue BMM1 — gated only on q_full/k_full —
    # overwrites it.  The mb_stat_full/mb_stat_empty handshake only orders
    # softmax vs correction; nothing ordered MMA's next-tile BMM1 after
    # correction's epilogue stats read (or after softmax's final stats
    # write).  One arrive per tile per sub-tile from the correction epilogue;
    # the leader MMA waits it before the prologue BMM1 into that S slot.
    mb_stats_read: object

    mb_o_full: object
    mb_o_empty: object

    mb_tmem_dealloc: object
    mb_empty_mainloop: object

    mb_q_o_alias: object
    # Return edge of the Q∪O alias gate (see the soundness note in
    # make_classic_bars): TMA-LDG arrives after consuming each alias-gate
    # phase; TMA-STG waits it before its next alias arrive.
    mb_qo_slab_free: object

    # Fused epilogue gate (O := O * sigmoid(G)) staging-tile handshake; None
    # unless the bars were built with epilogue_gate=True.  Table + lane
    # arithmetic: make_d256_bars.
    mb_gate_full: object = None
    mb_gate_empty: object = None
    # Resident additive-bias block handshake (the sm107 d128-file tiles at
    # CFG.BIAS_MODE == 1); None unless built with bias_resident=True.  Two
    # ITEM-level rows (lane ledger in make_classic_bars):
    #   mb_bias_full   TMA_LOAD, init ONE_LANE: the TMA-LDG warp arms it once per
    #                  non-empty work item (pred=elect_sync -> 1 lane) with the
    #                  sum of the block's FULL tile boxes; both softmax WGs wait
    #                  it once per item before their first bias read.
    #   mb_bias_empty  THREAD, init BIAS_EMPTY_ARRIVERS = the softmax warps (8):
    #                  each softmax warp arrives once per non-empty item after
    #                  its LAST read of the block (`if elect_sync():` -> 1 lane x
    #                  8 warps); the TMA-LDG warp waits it before the NEXT item's
    #                  fill (consumer phase pre-armed at 1: the first fill waits
    #                  on nothing).  An EMPTY item (zero kv tiles) touches
    #                  neither row on either side.
    mb_bias_full: object = None
    mb_bias_empty: object = None
    # The STREAMED bias ring (CFG.BIAS_MODE == 2; None unless built with bias_ring_stages > 0): STAGES_BIAS stages, one bias
    # tile (BIAS_BLOCK_ROWS x TILE_N) per kv tile.
    #   mb_bias_ring_full[s]   TMA_LOAD, init ONE_LANE: the TMA-LDG warp arms stage s once per kv tile (expect_tx = the tile's
    #                          FULL boxes, one elected lane) and queues the tile; both softmax warpgroups wait it before the
    #                          tile's first bias read (256 lanes wait; waits do not count).
    #   mb_bias_ring_empty[s]  THREAD, init BIAS_EMPTY_ARRIVERS = the softmax warps (8): one elected arrive per softmax warp per
    #                          kv tile after the stage's last bias word is consumed; the TMA-LDG warp waits it before refilling
    #                          the stage (pre-armed on its side: the first STAGES_BIAS fills wait on nothing).  On the
    #                          streamed_multicast ring (CFG.BIAS_MODE == 3) the init is BIAS_EMPTY_ARRIVERS x BIAS_MCAST_CTAS
    #                          (make_classic_bars(bias_ring_empty_arrivers=): 8 own + 8 peer = 16): the PEER CTA's softmax warps
    #                          credit this CTA's stage too, through the .release.cta cluster arrive (arrive_on_peer(release_cta=True),
    #                          a read-then-credit across CTAs -- never the relaxed form; the kernel's barrier table has the row).
    mb_bias_ring_full: object = None
    mb_bias_ring_empty: object = None


class D256Bars(NamedTuple):
    mb_q_full: object
    mb_q_o_alias: object
    mb_tmastg_go: object

    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object

    mb_bmm1_done: object
    mb_bmm2_done: object
    mb_bmm2_ready: object

    mb_stat_full: object
    mb_stat_empty: object

    mb_o_full: object
    mb_o_empty: object

    mb_empty_mainloop: object
    mb_tmem_dealloc: object

    # Fused epilogue gate staging-tile handshake (None unless
    # epilogue_gate=True) -- see make_d256_bars.
    mb_gate_full: object = None
    mb_gate_empty: object = None


def make_d256_bars(CFG, *, N_O_CHUNKS: int, epilogue_gate: bool = False) -> D256Bars:
    """Barrier bundle for the Q∪O-aliased d256 pipeline.

    ``epilogue_gate`` adds the two LOCAL barriers of the fused epilogue gate
    (O := O * sigmoid(G)); both stay ``None`` otherwise so every existing
    kernel traces byte-identically.  Their table (identical on the f16 and
    the per-tensor FP8 d256 kernels):

      mb_gate_full   Producer.TMA_LOAD, Scope.LOCAL.  ONE arrive_expect_tx per
                     tile from the TMA-LDG warp, ``pred=nvvm.elect_sync()``
                     -> 1 issuing lane x 1 call = 1 == init CFG.ONE_LANE.
                     Bytes: the gate tile's subtiles, cta_group=1 (shared::cta),
                     ALL to this CTA's mbar -- no CTA_MMA factor (P9 routing
                     applies to the cta_group::2 tensor form only).  Consumer:
                     the correction warpgroup, 128 lanes wait once per tile,
                     phase starts 0 (wait-then-arrive, P2).
      mb_gate_empty  Producer.THREAD, Scope.LOCAL.  BARE ``.arrive()`` from every
                     correction lane once per tile after its last gate LDS ->
                     128 issuing lanes x 1 call = 128 == init CFG.CORR_LANES
                     (THREAD has no ``pred=`` path; same form as mb_o_full).
                     Consumer: the TMA-LDG warp, one wait per tile at the TOP
                     of its tile iteration, phase PRE-ARMED at 1 (P5b).
      P14: the gate load is issued on BOTH arms of the empty-mainloop branch
                     and the epilogue runs on every tile, so each tile is
                     exactly one wait + one arrive on each bar at every shape.
      P15: no cross-CTA arrive on either bar -> no drain owed.
    """
    SOFTMAX_LANES_TOTAL = CFG.SOFTMAX_LANES * CFG.CTA_MMA
    CORR_LANES_TOTAL = CFG.CORR_LANES * CFG.CTA_MMA
    SOFTMAX_PLUS_CORR_TOTAL = SOFTMAX_LANES_TOTAL + CORR_LANES_TOTAL
    KV_EMPTY_ARRIVERS = (CFG.CGA_M // CFG.CTA_MMA) + CFG.CGA_N - 1
    N_BMM2_CHUNKS = CFG.N_BMM2_CHUNKS

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return D256Bars(
        mb_q_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_o_alias=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        mb_tmastg_go=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.THREAD),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_empty=MBarrier(_alloc(CFG.STAGES_KV), stages=CFG.STAGES_KV, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_bmm1_done=MBarrier(_alloc(2), stages=2, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(2), stages=2, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(2 * N_BMM2_CHUNKS),
            stages=2 * N_BMM2_CHUNKS,
            init_count=tuple(SOFTMAX_PLUS_CORR_TOTAL if (s % N_BMM2_CHUNKS) == 0 else SOFTMAX_LANES_TOTAL for s in range(2 * N_BMM2_CHUNKS)),
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_stat_full=MBarrier(_alloc(1), stages=1, init_count=CFG.SOFTMAX_LANES, producer=Producer.THREAD),
        mb_stat_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_full=MBarrier(_alloc(N_O_CHUNKS), stages=N_O_CHUNKS, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.THREAD),
        mb_gate_full=(MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD) if epilogue_gate else None),
        mb_gate_empty=(MBarrier(_alloc(1), stages=1, init_count=CFG.CORR_LANES, producer=Producer.THREAD) if epilogue_gate else None),
    )


class D512X2Bars(NamedTuple):
    """Barrier bundle of the d512 2x2-datapath forward (sm100/prefill_d512_f16_2x2.py)."""

    mb_q_full: object
    mb_q_empty: object

    mb_k_full: object
    mb_k_empty: object
    mb_v_full: object
    mb_v_empty: object

    mb_bmm1_done: object
    mb_bmm2_done: object
    mb_bmm2_ready: object
    mb_p_full: object

    mb_stat_full: object
    mb_stat_empty: object

    mb_o_full: object
    mb_o_empty: object

    mb_empty_mainloop: object
    mb_tmem_dealloc: object


def make_d512_2x2_bars(CFG, *, N_O_CHUNKS: int, STAT_STAGES: int = 2, cross_pair_poll: bool = True) -> D512X2Bars:
    """Barrier bundle for the d512 2x2-datapath pipeline.  MBARRIER LEDGER (per CTA; "x2 CTAs" =
    both CTAs of the pair arrive on the LEADER's copy; every init count is the exact per-phase
    arrival sum, P3):

    WAIT FORM RULE (``cross_pair_poll``, default True -- the kernel's POLL_CROSS_PAIR_WAITS constant; only the
    time-slicing negative control flips it): every barrier whose phase can be completed by an operation issued
    from the OTHER pair of the cluster is waited with the NON-BLOCKING ``test_wait.parity`` poll
    (``MBarrier(poll=True)`` -> ``barrier.wait_poll``, POLL_TIGHT_ITERS tight tests then a timer nanosleep
    between tests): mb_k_empty / mb_v_empty (both pair leaders' commits,
    mask 0xF), mb_o_empty (the twin's arrive_on_peer), mb_k_full / mb_v_full (the twin pair's TMA
    complete_tx).  A parked waiter (the default ``try_wait`` hint form AND the hint-less spin) loses such a
    wake-up under GPU time-slicing (d512_bprop lane, B200 2026-10-01: hang within 2-74 launches; the poll
    200/200).  Pair-local barriers (q_*, bmm1/2_done, bmm2_ready, p_full, stat_*, o_full, empty_mainloop,
    tmem_dealloc) keep the default form.

      mb_q_full       [1]  init 1 (ONE_LANE), TMA_LOAD.  Leader TMA-LDG lane expect_tx(Q bytes of the
                           PAIR = TILE_M*TILE_K*BPE*CTA_MMA) once per tile, pred = is_leader & elect;
                           the bytes of BOTH CTAs' cta_group::2 Q loads complete on the leader's copy
                           (P9 routing).  Waiter: leader MMA, phase 0.
      mb_q_empty      [1]  init 1, MMA_COMMIT.  Leader elected commit after the tile's last BMM1,
                           pair multicast.  Waiters: TMA-LDG of both CTAs, pre-armed phase 1.
      mb_k_full   [STAGES_K_SUB]  init 1, TMA_LOAD.  Leader expect_tx(64 KiB = the bytes landing in
                           the PAIR per 32 KiB-per-CTA sub-chunk) per sub-chunk; KV_SHARE=2: 4 issuers
                           x 16 KiB (each twin issues 2 of the 4 subtiles into both twins), KV_SHARE=1:
                           2 x 32 KiB own-bit.  Waiter: leader MMA.
      mb_k_empty  [STAGES_K_SUB]  init KV_EMPTY_ARRIVERS = CGA_M // CTA_MMA, MMA_COMMIT.  EACH pair
                           leader's elected commit, mask 0xF (KV_SHARE=2) / pair mask (1): a slot is
                           refilled only after BOTH pairs' MMAs read it (the twin's bytes land in MY
                           slot).  Waiter: TMA-LDG (own copy), pre-armed 1.
      mb_v_full / mb_v_empty [STAGES_V_SUB]  as K, V sub-chunks (2 x 16 KiB subtiles per CTA).
      mb_bmm1_done [XFER_STAGES]  init 1, MMA_COMMIT.  Leader commit, pair mask, 1x/iteration.
                           Waiters: 128 softmax lanes (per-parity phase bit).
      mb_bmm2_done [XFER_STAGES]  init 1, MMA_COMMIT.  Leader commit, pair mask, 1x/iteration (empty
                           tile: 1 commit after mb_empty_mainloop).  Waiters: correction lanes, exactly
                           one in-loop wait per iteration (either ballot arm) + the final tile wait.
      mb_bmm2_ready [XFER_STAGES * N_BMM2_CHUNKS]  init PAIR_LANES = 128 corr lanes x 2 CTAs, LEADER
                           (relaxed; TMEM data ordered by tcgen05.wait::st), Scope.LEADER.  Every
                           correction lane once per slot per iteration from exactly one ballot arm;
                           iteration kv_left arrives both slots directly.  Waiter: leader MMA before
                           BMM2 N-block c.
      mb_p_full   [XFER_STAGES]  init PAIR_LANES = 128 softmax lanes x 2 CTAs, LEADER_RELEASE,
                           Scope.LEADER.  Every softmax lane after its store_swizzled of P and
                           fence_proxy("async.shared", space="cta"): arrive_on_leader_release
                           (.release.cta on the mapa'd leader bar; probe release_p: 0/1920
                           mismatches), 1x/iteration.  Waiter: leader MMA before BMM2(i).
      mb_stat_full [STAT_STAGES]  init SOFTMAX_LANES (128), THREAD.  Every softmax lane after its
                           alpha (or tile-end stats) tcgen05.st + wait::st: 1x/iteration + 1x/tile
                           end; the ring advances on both.  Waiters: correction lanes, phase 0.
                           The payload of ring step s lives in slot-s columns ONLY (alpha at 384+s,
                           the tile stats at 386+2s / 387+2s): a fixed stats pair is protected by the
                           ring only if the next writer waits the SAME slot's empty, which an EMPTY
                           tile (one ring step) does not -- the dead-row race the sm107 lane found.
      mb_stat_empty [STAT_STAGES]  init CORR_LANES (128), THREAD.  Every correction lane once per
                           consumed ring step.  Waiters: softmax lanes before the next store into
                           that slot, pre-armed 1.
      mb_o_full   [N_O_CHUNKS]  init O_CHUNK_ARRIVERS = 64, THREAD.  The 64 lanes of column half
                           (chunk // 4) (warps 4+2h, 5+2h), one arrive per 8 KiB O subtile after
                           fence_proxy; the fp32-partials arm arrives identically (protocol only).
                           Waiter: TMA-STG per chunk.
      mb_o_empty  [1]  init O_EMPTY_ARRIVERS = ONE_WARP * KV_SHARE (64 / 32), THREAD.  PAIR-WIDE
                           O u V alias gate: after commit + wait_group.read 0 every TMA-STG lane
                           arrives on its OWN copy (32) and, under KV_SHARE=2, arrive_on_peer on the
                           TWIN's (cta ^ 2) copy (+32) -- the twin's V(t+1) multicast lands in MY sVO,
                           so MY store must gate it too.  1x/tile per arriver.  Waiters (non-consuming,
                           pre-armed 1): correction before its sO writes (also covers the twin's store:
                           a shared phase, not a data need); TMA-LDG before the first V load of the next
                           tile (advance-only on empty tiles) and once more at kernel end so the twin's
                           last remote arrive never targets an exited CTA.
      mb_empty_mainloop [1]  init PAIR_LANES, LEADER, Scope.LEADER.  Every correction lane x 2 CTAs
                           on an EMPTY tile.  Waiter: leader MMA.
      mb_tmem_dealloc [1]  init PAIR_LANES, THREAD.  Every correction lane: arrive() local +
                           arrive_on_peer(cta ^ 1) -> 128 + 128 per CTA.  Waiters: both MMA warps.
    Gone vs the role-split CfgD512 bundle: p/alpha/stats xfer full/empty, q_utccp_done, s_acc_empty
    (the S/P slot reuse rides the MMA thread's issue order: p_full(i) -> BMM2(i) -> BMM1(i+XFER)).
    """
    n_pairs = CFG.CGA_M // CFG.CTA_MMA
    if CFG.KV_EMPTY_ARRIVERS != n_pairs:
        raise ValueError(f"d512 2x2 bars: KV_EMPTY_ARRIVERS ({CFG.KV_EMPTY_ARRIVERS}) != CGA_M // CTA_MMA ({n_pairs})")
    if CFG.O_CHUNK_ARRIVERS != CFG.CORR_LANES // 2:
        raise ValueError(f"d512 2x2 bars: O_CHUNK_ARRIVERS ({CFG.O_CHUNK_ARRIVERS}) != the 64 lanes of one column half")
    if CFG.O_EMPTY_ARRIVERS != CFG.ONE_WARP * CFG.KV_SHARE:
        raise ValueError(
            f"d512 2x2 bars: O_EMPTY_ARRIVERS ({CFG.O_EMPTY_ARRIVERS}) != ONE_WARP * KV_SHARE ({CFG.ONE_WARP * CFG.KV_SHARE}): own + twin TMA-STG warps"
        )
    if CFG.PAIR_LANES != CFG.SOFTMAX_LANES * CFG.CTA_MMA or CFG.PAIR_LANES != CFG.CORR_LANES * CFG.CTA_MMA:
        raise ValueError("d512 2x2 bars: PAIR_LANES must be the 128 lanes of a compute warpgroup x CTA_MMA")

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return D512X2Bars(
        mb_q_full=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_k_full=MBarrier(_alloc(CFG.STAGES_K_SUB), stages=CFG.STAGES_K_SUB, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD, poll=cross_pair_poll),
        mb_k_empty=MBarrier(
            _alloc(CFG.STAGES_K_SUB), stages=CFG.STAGES_K_SUB, init_count=CFG.KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT, poll=cross_pair_poll
        ),
        mb_v_full=MBarrier(_alloc(CFG.STAGES_V_SUB), stages=CFG.STAGES_V_SUB, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD, poll=cross_pair_poll),
        mb_v_empty=MBarrier(
            _alloc(CFG.STAGES_V_SUB), stages=CFG.STAGES_V_SUB, init_count=CFG.KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT, poll=cross_pair_poll
        ),
        mb_bmm1_done=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS),
            stages=CFG.XFER_STAGES * CFG.N_BMM2_CHUNKS,
            init_count=CFG.PAIR_LANES,
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_p_full=MBarrier(_alloc(CFG.XFER_STAGES), stages=CFG.XFER_STAGES, init_count=CFG.PAIR_LANES, producer=Producer.LEADER_RELEASE, scope=Scope.LEADER),
        mb_stat_full=MBarrier(_alloc(STAT_STAGES), stages=STAT_STAGES, init_count=CFG.SOFTMAX_LANES, producer=Producer.THREAD),
        mb_stat_empty=MBarrier(_alloc(STAT_STAGES), stages=STAT_STAGES, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_full=MBarrier(_alloc(N_O_CHUNKS), stages=N_O_CHUNKS, init_count=CFG.O_CHUNK_ARRIVERS, producer=Producer.THREAD),
        mb_o_empty=MBarrier(_alloc(1), stages=1, init_count=CFG.O_EMPTY_ARRIVERS, producer=Producer.THREAD, poll=cross_pair_poll),
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=CFG.PAIR_LANES, producer=Producer.LEADER, scope=Scope.LEADER),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CFG.PAIR_LANES, producer=Producer.THREAD),
    )


def make_classic_bars(
    CFG,
    s_stages: Optional[int] = None,
    *,
    epilogue_gate: bool = False,
    q_full_arrivers: int = 1,
    o_parts: int = 1,
    bias_resident: bool = False,
    bias_ring_stages: int = 0,
    bias_ring_empty_arrivers: Optional[int] = None,
    stages_k: Optional[int] = None,
    stages_v: Optional[int] = None,
) -> Bars:
    """The classic pipeline's barrier set.

    ``stages_k`` / ``stages_v``: the K and V ring depths when a kernel runs them at different depths (the d192
    half-input cga1 tile: one K stage, two V stages); None (every other caller) = ``CFG.STAGES_KV`` for both.

    ``s_stages`` is the S/P TMEM slot ring depth the BMM1-done / BMM2-ready
    handshakes run over: one stage per Q sub-tile on the classic pipeline
    (``CFG.TILES_Q``, the default), or the number of slots a single-sub-tile
    kernel alternates its S between so BMM1(i+1) can overlap softmax(i)
    (``sm100/decode_d128_f16.py`` passes 2).  Every other barrier keeps its
    per-sub-tile count.

    ``bias_resident``: the two item-level rows of the resident additive-bias
    block (``Bars.mb_bias_full`` / ``mb_bias_empty``; the issuing-lane ledger is on the
    fields).  Init counts: ONE_LANE (one elected expect_tx arrive per item) and
    ``CFG.BIAS_EMPTY_ARRIVERS`` (one elected THREAD arrive per softmax warp per
    item = SOFTMAX_WARPGROUPS x SOFTMAX_WG_WARPS, pinned by the config
    validator).  False (every other caller) leaves them None and traces the
    pre-bias bundle byte-for-byte.

    ``bias_ring_stages``: the STREAMED bias ring's two rows (``Bars.mb_bias_ring_full`` /
    ``mb_bias_ring_empty``, that many stages each; the same two init counts per stage
    -- ONE_LANE for the TMA-LDG warp's per-kv-tile expect_tx, ``CFG.BIAS_EMPTY_ARRIVERS``
    for the softmax warps' per-kv-tile elected release).  0 (every other caller)
    leaves them None.  A kernel passes one of ``bias_resident`` / ``bias_ring_stages``,
    never both: the two bias forms fold on ``CFG.BIAS_MODE``.

    ``bias_ring_empty_arrivers`` (appended): the ``mb_bias_ring_empty`` init count when it is NOT the CTA's own softmax
    warps alone -- the ``streamed_multicast`` ring (``CFG.BIAS_MODE == 3``) releases every stage from BOTH CTAs of the
    cluster (one elected local arrive + one elected ``.release.cta`` cluster arrive (``arrive_on_peer(release_cta=True)``,
    a READ-then-credit across CTAs -- never the relaxed form) per softmax warp), so each CTA's ring counts
    ``BIAS_EMPTY_ARRIVERS x 2`` per phase.  None (every other caller) = ``CFG.BIAS_EMPTY_ARRIVERS``.
    """
    if bias_resident and bias_ring_stages:
        raise ValueError("make_classic_bars: bias_resident and bias_ring_stages are the two bias FORMS (CFG.BIAS_MODE 1 / 2); pass one")
    if bias_ring_stages < 0:
        raise ValueError(f"make_classic_bars: bias_ring_stages must be >= 0 (got {bias_ring_stages})")
    if bias_ring_empty_arrivers is not None and (not bias_ring_stages or int(bias_ring_empty_arrivers) < 1):
        raise ValueError(f"make_classic_bars: bias_ring_empty_arrivers={bias_ring_empty_arrivers} needs a ring (bias_ring_stages={bias_ring_stages}) and >= 1")
    # Read CFG.BIAS_EMPTY_ARRIVERS only when a bias form asks for it: the shared SM100 bodies (the d128 decode tile, the
    # SM100 prefill files) build their bars through this factory with Cfg records that carry no bias fields.
    ring_empty_count = int(bias_ring_empty_arrivers) if bias_ring_empty_arrivers is not None else (CFG.BIAS_EMPTY_ARRIVERS if bias_ring_stages else 0)
    # ``epilogue_gate``: same two LOCAL gate barriers as make_d256_bars (table
    # there); no classic (d128/d192) kernel passes True yet, so today every
    # caller gets None and traces unchanged.  The init counts (ONE_LANE /
    # CORR_LANES) were derived for the d256 BODIES only: a classic kernel that
    # flips this must first splice the gate seams into ITS TMA-LDG and
    # correction groups and re-audit both rows per P3 (SUM(issuing lanes) ==
    # init) against those bodies -- the counts are not inherited from here.
    SOFTMAX_PLUS_CORR_TOTAL = CFG.SOFTMAX_LANES * 2 * CFG.CTA_MMA
    SOFTMAX_LANES_TOTAL = CFG.SOFTMAX_LANES * CFG.CTA_MMA
    CORR_LANES_TOTAL = CFG.CORR_LANES * CFG.CTA_MMA
    KV_EMPTY_ARRIVERS = (CFG.CGA_M // CFG.CTA_MMA) + CFG.CGA_N - 1
    N_BMM2_CHUNKS = CFG.N_BMM2_CHUNKS
    N_S = CFG.TILES_Q if s_stages is None else int(s_stages)
    STAGES_K = CFG.STAGES_KV if stages_k is None else int(stages_k)
    STAGES_V = CFG.STAGES_KV if stages_v is None else int(stages_v)

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    return Bars(
        # q_full_arrivers > 1: besides the leader's expect_tx arrive, each peer CTA's TMA-LDG warp arrives once
        # after filling its own SF_Q tile by hand (the MXFP8 PackGQA gather, sm100/prefill_d128_mxfp8.py).
        mb_q_full=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE + q_full_arrivers - 1, producer=Producer.TMA_LOAD),
        mb_k_full=MBarrier(_alloc(STAGES_K), stages=STAGES_K, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_full=MBarrier(_alloc(STAGES_V), stages=STAGES_V, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_k_empty=MBarrier(_alloc(STAGES_K), stages=STAGES_K, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_empty=MBarrier(_alloc(STAGES_V), stages=STAGES_V, init_count=KV_EMPTY_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_bmm1_done=MBarrier(_alloc(N_S), stages=N_S, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_done=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_bmm2_ready=MBarrier(
            _alloc(N_S * N_BMM2_CHUNKS),
            stages=N_S * N_BMM2_CHUNKS,
            init_count=tuple(SOFTMAX_PLUS_CORR_TOTAL if (s % N_BMM2_CHUNKS) == 0 else SOFTMAX_LANES_TOTAL for s in range(N_S * N_BMM2_CHUNKS)),
            producer=Producer.LEADER,
            scope=Scope.LEADER,
        ),
        mb_stat_full=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.SOFTMAX_LANES, producer=Producer.THREAD),
        mb_stat_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_stats_read=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CORR_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        # ``o_parts`` > 1: the O slab of each sub-tile is published (and TMA-stored) per column block.
        mb_o_full=MBarrier(_alloc(CFG.TILES_Q * o_parts), stages=CFG.TILES_Q * o_parts, init_count=CFG.CORR_LANES, producer=Producer.THREAD),
        mb_o_empty=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.THREAD),
        mb_empty_mainloop=MBarrier(_alloc(1), stages=1, init_count=CORR_LANES_TOTAL, producer=Producer.LEADER, scope=Scope.LEADER),
        # Q∪O alias gate FULL/EMPTY pair.  mb_q_o_alias alone is UNSOUND:
        # mbarrier parity waits deadlock once a producer runs >= 2 phases
        # ahead, and on EMPTY tiles (zero-KV varlen sequences) the
        # corr -> STG -> alias-arrive chain has NO dependency on TMA-LDG, so
        # a delayed LDG warp loses the race and its bootstrap parity credit
        # is consumed by a real arrive (observed: LDG parked forever at the
        # tile-1 alias wait with the barrier already in phase 1, deadlocking
        # the whole cluster).  mb_qo_slab_free is the return edge: LDG
        # arrives it right after consuming each alias phase and STG waits it
        # before each alias arrive, bounding either side's lead to one phase
        # by construction.
        mb_q_o_alias=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_qo_slab_free=MBarrier(_alloc(CFG.TILES_Q), stages=CFG.TILES_Q, init_count=CFG.ONE_WARP, producer=Producer.THREAD),
        mb_gate_full=(MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD) if epilogue_gate else None),
        mb_gate_empty=(MBarrier(_alloc(1), stages=1, init_count=CFG.CORR_LANES, producer=Producer.THREAD) if epilogue_gate else None),
        mb_bias_full=(MBarrier(_alloc(1), stages=1, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD) if bias_resident else None),
        mb_bias_empty=(MBarrier(_alloc(1), stages=1, init_count=CFG.BIAS_EMPTY_ARRIVERS, producer=Producer.THREAD) if bias_resident else None),
        mb_bias_ring_full=(
            MBarrier(_alloc(bias_ring_stages), stages=bias_ring_stages, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD) if bias_ring_stages else None
        ),
        mb_bias_ring_empty=(
            MBarrier(_alloc(bias_ring_stages), stages=bias_ring_stages, init_count=ring_empty_count, producer=Producer.THREAD) if bias_ring_stages else None
        ),
    )


def row_max_for_exp2(total_max):
    """Canonical masked-softmax row-max guard (FlashAttention / cuDNN form).

    The f16 prefill kernels mask scores with true ``-inf``
    (``apply_mask_chunk(..., mask_value=float("-inf"))``) and start the running
    ``total_max`` at ``-inf``, so for a row that has not yet seen a live column
    the (scaled) row max is still exactly ``-inf``. Using it directly would
    make every masked exp2 argument ``-inf - (-inf) == NaN`` (or, with the old
    finite sentinel, ``0`` and hence a bogus P == 1 per masked column). The
    canonical fix — FlashAttention's ``max == -INFINITY ? 0.f : max * scale``
    and cuDNN's ``(total_max == NEG_INFINITY) ? 0.0f : total_max`` — is a
    single compare + select substituting 0 AT THE POINT OF USE:

      * masked scores become ``exp2(-inf - 0) == 0``, so a fully-masked
        iteration ships P == 0 to BMM2 and contributes nothing;
      * a fully-masked row naturally ends the tile with ``total_sum == 0``,
        which the epilogue turns into O := 0 / LSE := -inf;
      * substituted into BOTH alpha operands, any iteration that does not move
        the max (fully-masked ones included, since ``-inf`` can never clear
        the RESCALE_THRESHOLD update) yields alpha == exp2(0 - 0) == 1
        exactly, so the all_alpha_one ballot keeps firing.

    Callers clamp the alpha exponent with ``min(prev_safe - new_safe, 0)`` so
    the one transition that can lower the safe max (dead -> alive, 0 ->
    real*scale < 0) cannot overflow exp2; total_sum is still exactly 0 there,
    so any finite alpha is exact.
    """
    still_dead = total_max == cutlass.Float32(float("-inf"))
    return cutlass.Float32(
        arith.select(
            still_dead.ir_value(),
            cutlass.Float32(0.0).ir_value(),
            total_max.ir_value(),
        )
    )


@cute.jit
def running_max_step_finite_sentinel(raw_max, current_max, total_max, neg_inf, rescale_threshold, masked: cutlass.Constexpr[bool]):
    """One KV tile's online-softmax running-max step for the kernels that mask with the FINITE sentinel
    (``apply_mask_chunk``'s default ``_NEG_INF_BITS``, the ``neg_inf`` their running ``total_max`` also starts at).

    ``raw_max`` is the tile's UNSCALED row max -- exactly ``neg_inf`` iff every column of this tile is masked --
    and ``current_max`` the same max in the exp2 domain (``raw_max * scale_log2``, or ``raw_max`` itself under a
    pre-folded scale).  Returns ``(total_max, alpha, exp_shift)``: the running max, the rescale factor of the
    accumulator and the row-sum (``exp2(old - new)``; ``exp2(neg_inf) == 0`` on the row's first live tile, whose
    stat the correction warp consumes unread), and the value this tile's scores are shifted by before ``exp2``.

    The guard (``masked``; a MASK_NONE build has no masked column and folds it out, byte-identical): a tile that
    is fully masked BEFORE the row's first live key -- a left band or a bottom-right diagonal clearing a whole
    tile, the row's keys in a LATER tile -- is ``tile_dead``.  Letting the sentinel become the running max breaks
    the shift: scaled by ``scale_log2 > 1`` it overflows to -inf and ``-inf - (-inf)`` is NaN (NaN P into BMM2,
    NaN O); below that the shift reads 0 and every masked column publishes P = 1, mass that only ``alpha = 0`` at
    the next live tile wipes -- a multiply by zero of garbage.  So the dead tile is SELECTED out of the state:
    ``total_max`` stays ``neg_inf`` (``is_first`` then starts the online softmax at the first live tile), ``alpha``
    is ``exp2(neg_inf - neg_inf) = 1`` exactly (the accumulator and the row-sum are still their zero-trip state)
    and ``exp_shift`` is 0 -- :func:`row_max_for_exp2`'s form for the finite sentinel -- so every column lands at
    ``exp2(sentinel * scale)`` = +0 on every arm (f32 ``ex2.approx``, the f16x2 arms, the fused f32x2 -> f16x2
    shift): the path a masked column of a LIVE tile already takes, with a finite shift.  A dead tile AFTER a live
    one needs nothing: the finite running max never clears the threshold, alpha is ``exp2(0) = 1`` and the masked
    columns read ``sentinel * scale - m`` -> 0.  A row with no live key at all ends its loop at ``(neg_inf, 0)``
    and stays with the epilogue's geometry select (``_kv_empty`` / ``_row_empty``)."""
    old_total_max = total_max
    is_first = total_max == neg_inf
    update_cond = is_first | ((current_max - total_max) > rescale_threshold)
    first_live = is_first
    if cutlass.const_expr(masked):
        tile_dead = is_first & (raw_max == neg_inf)
        update_cond = update_cond & ~tile_dead
        first_live = is_first & ~tile_dead
    total_max = cutlass.Float32(arith.select(update_cond.ir_value(), current_max.ir_value(), total_max.ir_value()))
    exp_input = cutlass.Float32(arith.select(first_live.ir_value(), neg_inf.ir_value(), (old_total_max - total_max).ir_value()))
    alpha = cute.math.exp2(exp_input, fastmath=True)
    exp_shift = total_max
    if cutlass.const_expr(masked):
        exp_shift = cutlass.Float32(arith.select(tile_dead.ir_value(), cutlass.Float32(0.0).ir_value(), total_max.ir_value()))
    return total_max, alpha, exp_shift


def assert_tile_n_supported(CFG):
    """Import-time gate: kernels on the ``reg_S_a``/``reg_S_b`` softmax body
    require N_BMM2_CHUNKS == 2 (TILE_N == 128)."""
    if CFG.N_BMM2_CHUNKS != 2:
        raise NotImplementedError(f"this kernel currently requires TILE_N=128 (got TILE_N={CFG.TILE_N})")


def load_o_chunks_grouped(tmem_base, tmem_o_off, tile_o: int, chunk: int, group: int):
    """Every fp32 O chunk (``tile_o // chunk`` of them, ``chunk`` columns each) of a TMEM accumulator, loaded ``group`` at a time with ONE wait per
    group -- the per-chunk ``ld`` + ``wait`` of the original schedule pays a serialized TMEM round trip per chunk.  Every chunk stays live until the
    caller consumes it, so ``tile_o`` fp32 registers per thread must fit the epilogue's budget."""
    chunks = []
    for g0 in range(0, tile_o // chunk, group):
        loaded = [
            nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + cutlass.Int32(tmem_o_off + (g0 + k) * chunk), cutlass.Float32), num=chunk)
            for k in range(min(group, tile_o // chunk - g0))
        ]
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        chunks.extend(loaded)
    return tuple(chunks)


@cute.jit
def store_fp32_partial_tile(
    o_partial_f32,
    tmem_base,
    tmem_o_off,
    inv_sum,
    row_dead,
    row_valid,
    o_batch,
    q_row_global,
    row_head_idx,
    tile_o: cutlass.Constexpr[int],
    chunk: cutlass.Constexpr[int],
    load_group: cutlass.Constexpr[int] = 1,
) -> None:
    """Store one Q row's O tile as fp32, straight from TMEM to the workspace.

    The staged path casts the accumulator into the SMEM O tile and TMA-stores
    that, which ties the partial's width to the tile's.  Here the accumulator
    goes to global directly, so the partial can be fp32 while the tile -- and
    therefore the SMEM budget -- is untouched.  Shared by every flavor whose
    epilogue holds its O accumulator in TMEM.

    ``chunk`` must be the flavor's OWN TMEM read width (its ``O_CHUNK``): the
    O region is not uniformly addressable across flavors, so reading it in a
    different stride than the staged epilogue does silently returns the wrong
    columns rather than failing.  ``load_group`` > 1 loads that many chunks per wait
    (:func:`load_o_chunks_grouped`: the whole ``tile_o`` is live); 1 is the original wait per chunk.

    ``row_dead`` cannot be dropped in favour of the zeroed ``inv_sum`` every
    caller already computes: an empty mainloop never wrote O TMEM, so the load
    can return NaN, and ``NaN * 0.0`` is NaN, not zero.  The staged paths avoid
    this by not loading at all for such rows; here the select does it.
    """
    op = cutlass.make_array_view(o_partial_f32)
    # The slab carries the graph's ACTUAL d_v, which an ENVELOPE flavor routinely
    # exceeds -- d_v=64 runs on the d128 tile, so tile_o overshoots each row by 64
    # columns.  The staged TMA path clipped that to the tensor extent; a direct
    # store has to bound itself or it writes into the next head's row, and off the
    # end of the slab on the last one.  Read off the tensor rather than passed in,
    # so it cannot drift from the buffer actually bound.
    d_v = cutlass.const_expr(o_partial_f32.shape[3])
    # Use develop's 16-byte primitive when the current row supports it. Keep
    # the scalar fallback for legal narrow or unaligned caller bindings.
    grouped = load_o_chunks_grouped(tmem_base, tmem_o_off, tile_o, chunk, load_group) if cutlass.const_expr(load_group > 1) else None
    for blk in cutlass.range_constexpr(tile_o // chunk):
        if cutlass.const_expr(grouped is None):
            addr = tmem_base + cutlass.Int32(tmem_o_off + blk * chunk)
            vals = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(addr, cutlass.Float32), num=chunk)
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        else:
            vals = grouped[blk]
        scaled = vals * inv_sum
        if row_valid:
            row_out = op[o_batch, q_row_global, row_head_idx, :]
            if cutlass.const_expr(d_v % 4 == 0 and chunk % 4 == 0 and o_partial_f32.stride[3] == 1):
                row_ptr = op.data_ptr((o_batch, q_row_global, row_head_idx, 0))
                if (row_ptr.toint(cutlass.Int64) & cutlass.Int64(15)) == 0:
                    for group in cutlass.range_constexpr(chunk // 4):
                        if cutlass.const_expr(blk * chunk + group * 4 < d_v):
                            values = [
                                cutlass.Float32(arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), scaled[group * 4 + j].ir_value()))
                                for j in range(4)
                            ]
                            st_global_v4(row_ptr.toint(cutlass.Int64) + cutlass.Int64((blk * chunk + group * 4) * 4), values, cutlass.Float32)
                else:
                    for j in cutlass.range_constexpr(chunk):
                        if cutlass.const_expr(blk * chunk + j < d_v):
                            row_out[cutlass.Int32(blk * chunk + j)] = cutlass.Float32(
                                arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), scaled[j].ir_value())
                            )
            else:
                for j in cutlass.range_constexpr(chunk):
                    if cutlass.const_expr(blk * chunk + j < d_v):
                        row_out[cutlass.Int32(blk * chunk + j)] = cutlass.Float32(
                            arith.select(row_dead.ir_value(), cutlass.Float32(0.0).ir_value(), scaled[j].ir_value())
                        )


@cute.jit
def store_fp32_partial_tile_coalesced(
    o_partial_f32,
    tmem_base,
    tmem_o_off,
    inv_sum,
    row_dead,
    row_valid,
    o_batch,
    q_row_global,
    row_head_idx,
    tile_o: cutlass.Constexpr[int],
    chunk: cutlass.Constexpr[int],
    scratch_addr,
    lane,
) -> None:
    """:func:`store_fp32_partial_tile` with whole-line global stores.

    The direct store writes one 16-byte piece of 32 DIFFERENT rows per warp
    instruction (rows are D*4 or H*D*4 bytes apart), so every instruction is a
    32-sector request with half of each sector unused, and a 256-row CTA's
    128 KiB of partials costs ~8k sector writes.  Here each warp first parks
    its 32-row x ``chunk``-column fp32 block in SMEM (``scratch_addr``: this
    WARP's 32 * chunk * 4 bytes in the shared window, idle under the split
    epilogue -- the O tile is never staged there), then reads it back
    column-major: lane l stores 16-byte segment ``l % SEGS`` of row
    ``l // SEGS``, so one instruction writes ``32 / SEGS`` complete rows of
    ``chunk * 4`` contiguous bytes (whole 128-byte lines at chunk 32).  The
    target row's coordinates come from the lane that owns it (three warp
    shuffles); invalid rows (past S_q) are skipped, dead rows store zeros as
    before.  The 16-byte segments are rotated by the row inside each 128-byte
    bank row so both the row-major write and the column-major read are
    bank-conflict free.  Requires the 16-byte-store conditions of the direct
    path (d_v % 4 == 0, chunk % 4 == 0, unit D stride, 16-byte aligned slab);
    the caller falls back to :func:`store_fp32_partial_tile` otherwise.
    """
    op = cutlass.make_array_view(o_partial_f32)
    d_v = cutlass.const_expr(o_partial_f32.shape[3])
    SEGS = chunk // 4
    RPI = 32 // SEGS
    ROW_BYTES = chunk * 4
    ROWS_PER_BANKROW = max(1, 128 // ROW_BYTES)
    zero = cutlass.Float32(0.0)
    valid_i = cutlass.Int32(arith.select(row_valid.ir_value(), cutlass.Int32(1).ir_value(), cutlass.Int32(0).ir_value()))
    my_seg_rot = lane // cutlass.Int32(ROWS_PER_BANKROW)
    rd_row0 = lane // cutlass.Int32(SEGS)
    rd_seg = lane % cutlass.Int32(SEGS)
    for blk in cutlass.range_constexpr(tile_o // chunk):
        addr = tmem_base + cutlass.Int32(tmem_o_off + blk * chunk)
        vals = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(addr, cutlass.Float32), num=chunk)
        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
        scaled = vals * inv_sum
        for seg in cutlass.range_constexpr(SEGS):
            values = [cutlass.Float32(arith.select(row_dead.ir_value(), zero.ir_value(), scaled[seg * 4 + j].ir_value())) for j in range(4)]
            rot = (cutlass.Int32(seg) + my_seg_rot) % cutlass.Int32(SEGS)
            st_shared_v4(scratch_addr + lane * cutlass.Int32(ROW_BYTES) + rot * cutlass.Int32(16), values, cutlass.Float32)
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        for k in cutlass.range_constexpr(SEGS):
            row = rd_row0 + cutlass.Int32(k * RPI)
            rot = (rd_seg + row // cutlass.Int32(ROWS_PER_BANKROW)) % cutlass.Int32(SEGS)
            v = ld_shared_v4(scratch_addr + row * cutlass.Int32(ROW_BYTES) + rot * cutlass.Int32(16), cutlass.Float32)
            q_t = nvvm.shfl_sync(0xFFFFFFFF, q_row_global, row, 31, kind=nvvm.Shfl.IDX)
            h_t = nvvm.shfl_sync(0xFFFFFFFF, row_head_idx, row, 31, kind=nvvm.Shfl.IDX)
            ok_t = nvvm.shfl_sync(0xFFFFFFFF, valid_i, row, 31, kind=nvvm.Shfl.IDX)
            col = cutlass.Int32(blk * chunk) + rd_seg * cutlass.Int32(4)
            if (ok_t != cutlass.Int32(0)) & (col < cutlass.Int32(d_v)):
                row_ptr = op.data_ptr((o_batch, q_t, h_t, 0))
                st_global_v4(
                    row_ptr.toint(cutlass.Int64) + cutlass.Int64(col) * cutlass.Int64(4),
                    [cutlass.Float32(v[0]), cutlass.Float32(v[1]), cutlass.Float32(v[2]), cutlass.Float32(v[3])],
                    cutlass.Float32,
                )
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)


class SplitHelpers(NamedTuple):
    """Split-aware decode / bounds closures, plus the two flags kernels fold on."""

    SPLIT_KV: int
    # True when a tile's KV range can come out empty (right <= left).  See
    # make_split_helpers for why KV split makes this reachable without a mask.
    MAY_BE_EMPTY: bool
    split_chunk: object
    decode_initial_split: object
    decode_payload_split: object
    bounds_for_tile_split: object
    nomask_range_split: object
    partial_batch: object


def make_split_helpers(CFG, *, bounds_for_tile, dispatch_decode_initial, dispatch_decode_payload) -> SplitHelpers:
    """Split-aware decode / bounds closures shared by the SM100 prefill flavors.

    ``bounds_for_tile`` is the caller's own bounds closure, taking
    ``(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_addr,
    batch_idx, qh_per_kh)`` — flavors differ in whether they apply the
    dead-Q-tile trim, so the split narrowing composes on top of whatever they
    already do.  ``qh_per_kh`` (trailing, default 1) is the PACKING GROUP
    size -- the kernel's HEADS_PER_TILE (``CFG.PACK_G``), which under partial
    PackGQA is a proper divisor of the graph's GQA ratio -- so the split
    chunks the tile's PACKED token-span bounds and packing and KV split
    compose; without CFG.PACK_GQA the bounds fold to the classic
    single-head-per-tile form and the argument is ignored.
    At SPLIT_KV == 1 every closure below folds away and the traced code is the
    classic single-pass kernel.
    """
    SPLIT_KV = int(getattr(CFG, "SPLIT_KV", 1))

    # The split index rides the BATCH axis (grid.z), not grid.x: the decode
    # already recovers the batch coordinate on BOTH the blockIdx and the
    # scheduler-handout paths -- it is the high half of the packed head|batch
    # word -- so a composite z = batch + split*B travels with it for free, with
    # no in-place mutation of the shared tile id and no dependence on the grid's
    # x extent (which is q_clusters * CGA_M, NOT the n_q_supers the kernel is
    # handed, on any flavor where CGA_M != CTA_MMA -- d512).

    # Can a tile's KV range come out EMPTY (right <= left)?
    #
    # Before KV split the answer was "only under a mask" — a SWA/causal/padded
    # tile can fall entirely outside the band — so the empty-tile handshake
    # (mb_empty_mainloop: correction arrives, MMA waits, TMA-LDG skips its
    # loads) was gated on MASK_FLAGS != 0 and folded away at MASK_NONE, where
    # [0, S_kv/TILE_N) is never empty.
    #
    # KV split breaks that WITHOUT a mask: a split past the end of a short range
    # legitimately gets zero tiles.  Correction detects empties with a RUNTIME
    # test, so if the gate const-folds to False the warp groups disagree —
    # correction jumps to its epilogue while MMA waits on mb_q_full and TMA-LDG
    # issues loads nobody consumes — and the kernel deadlocks.
    MAY_BE_EMPTY = (CFG.MASK_FLAGS != 0) or (SPLIT_KV > 1)

    # Which grid does this flavor launch?  SCHED_NATURAL uses a 3-D
    # (q_super, head, batch) grid; the LPT policy flattens everything into x.
    # This must use the flavor's effective CFG policy so the helper and launch
    # grid follow the same compile-time specialization.
    IS_LPT = CFG.SCHEDULER_POLICY != SCHED_NATURAL

    @cute.jit
    def _lpt_split_of(raw, n_q_supers, n_qh, n_batch):
        """(within-split raw x, split) for the flattened LPT grid.

        The LPT tile space is q_tiles * n_qh * n_batch clusters; KV split
        appends SPLIT_KV copies of it, split-major, so the split is the high
        digit of the cluster index.  The CGA lane (raw % CGA_M) is preserved so
        the caller's decode still sees a well-formed x coordinate.
        """
        cga = cutlass.Int32(CFG.CGA_M)
        linear = raw // cga
        # Q supers per cluster: CTA_MMA on the pair kernels, CGA_M on the d512 2x2 kernel
        # (Cfg.Q_SUPERS_PER_CLUSTER; absent on every other Cfg, so the trace is unchanged).
        q_tiles = n_q_supers // cutlass.Int32(int(getattr(CFG, "Q_SUPERS_PER_CLUSTER", CFG.CTA_MMA)))
        per_split = q_tiles * n_qh * n_batch
        split = linear // per_split
        rest = (linear % per_split) * cga + (raw % cga)
        return rest, split

    @cute.jit
    def _split_chunk(left, right, split_idx):
        """Cut ``[left, right)`` into SPLIT_KV near-equal chunks.

        The FIRST ``rem`` splits get one extra tile, so chunk sizes differ by at
        most 1 however the mask has already narrowed the range — a balanced cut
        matters because the slowest split sets the critical path.  A split past
        the end collapses to ``lo == hi``; the existing empty-mainloop path then
        writes O := 0 / LSE := -inf, exactly the identity of the combine's
        log-sum-exp, so no special case is needed downstream.
        """
        n_tiles = right - left
        per = n_tiles // cutlass.Int32(SPLIT_KV)
        rem = n_tiles % cutlass.Int32(SPLIT_KV)
        lo = left + split_idx * per + cute.math.min(split_idx, rem)
        extra = cutlass.Int32(
            arith.select(
                (split_idx < rem).ir_value(),
                cutlass.Int32(1).ir_value(),
                cutlass.Int32(0).ir_value(),
            )
        )
        return lo, lo + per + extra

    @cute.jit
    def _decode_initial_split(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
        """decode_initial + this tile's split index.

        NATURAL: the split rides the BATCH axis (see the note above on why not
        grid.x).  The host launches z = B * SPLIT_KV, so the split falls out of
        the DECODED batch coordinate as ``b // n_batch``, leaving the real batch
        as ``b % n_batch``.

        LPT / LPT_L2: the grid is flat, so there is no batch axis to ride and the
        split is folded into the linear tile id instead; ``_lpt_split_of`` peels
        it back off before the flavor's dispatcher sees the id.

        ``qh_per_kh`` / ``seqlen_kv`` are the LPT_L2 cost-model inputs; they are
        opaque here and forwarded to the flavor's dispatcher unchanged.
        """
        if cutlass.const_expr(SPLIT_KV > 1 and IS_LPT and not getattr(CFG, "THD_VARLEN", 0)):
            raw, split = _lpt_split_of(bidx, n_q_supers, n_qh, n_batch)
            q, h, b = dispatch_decode_initial(raw, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)
            return q, h, b, split
        q, h, b = dispatch_decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)
        if cutlass.const_expr(SPLIT_KV == 1):
            return q, h, b, cutlass.Int32(0)
        return q, h, b % n_batch, b // n_batch

    @cute.jit
    def _decode_payload_split(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
        """decode_payload + split index; ``t0`` is the try_cancel cluster-base id."""
        if cutlass.const_expr(SPLIT_KV > 1 and IS_LPT and not getattr(CFG, "THD_VARLEN", 0)):
            raw, split = _lpt_split_of(t0, n_q_supers, n_qh, n_batch)
            q, h, b = dispatch_decode_payload(raw, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)
            return q, h, b, split
        q, h, b = dispatch_decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh, seqlen_kv)
        if cutlass.const_expr(SPLIT_KV == 1):
            return q, h, b, cutlass.Int32(0)
        return q, h, b % n_batch, b // n_batch

    @cute.jit
    def _bounds_for_tile_split(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, split_idx, qh_per_kh: int = 1):
        """The flavor's (possibly packed) bounds, narrowed to this split's slice
        of the KV range.

        Splitting the ALREADY-masked ``[left, right)`` rather than the raw KV
        extent is what keeps causal / SWA correct AND balanced: each split gets
        an equal share of the tile's real work, not of the sequence — and under
        PackGQA that range is already in packed token-span units, so the two
        features compose.  The unmasked band is clamped into the slice, which
        preserves the ``left <= unmasked_lo <= unmasked_hi <= right`` invariant
        the mainloop relies on, because clamping is monotone.
        """
        b = bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, qh_per_kh)
        if cutlass.const_expr(SPLIT_KV == 1):
            return b
        lo, hi = _split_chunk(b.left, b.right, split_idx)
        return KvLoopBounds(
            left=lo,
            unmasked_lo=cute.math.min(cute.math.max(b.unmasked_lo, lo), hi),
            unmasked_hi=cute.math.min(cute.math.max(b.unmasked_hi, lo), hi),
            right=hi,
        )

    @cute.jit
    def _nomask_range_split(seqlen_kv, split_idx):
        """MASK_NONE fast-path KV range, split-aware.

        MMA / TMA-LDG take this path while softmax / correction go through
        bounds_for_tile_split; every warp group must land on the SAME chunk
        boundaries or their mbarrier handshakes desync.  The split branch
        therefore divides with the div-up compute_kv_loop_bounds uses, not the
        floor of the historical fast path.
        """
        if cutlass.const_expr(SPLIT_KV == 1):
            return cutlass.Int32(0), seqlen_kv // cutlass.Int32(CFG.TILE_N)
        n_tiles = (seqlen_kv + cutlass.Int32(CFG.TILE_N - 1)) // cutlass.Int32(CFG.TILE_N)
        return _split_chunk(cutlass.Int32(0), n_tiles, split_idx)

    @cute.jit
    def _partial_batch(batch_idx, split_idx, n_batch):
        """Batch coord of this split's partial O / LSE slot (split-major).

        Stacking the partials on the BATCH axis (extent B*SPLIT_KV) means the O
        TMA descriptor is untouched — only the coord shifts.  Folds to batch_idx
        at SPLIT_KV == 1.
        """
        if cutlass.const_expr(SPLIT_KV == 1):
            return batch_idx
        return batch_idx + split_idx * n_batch

    return SplitHelpers(
        SPLIT_KV=SPLIT_KV,
        MAY_BE_EMPTY=MAY_BE_EMPTY,
        split_chunk=_split_chunk,
        decode_initial_split=_decode_initial_split,
        decode_payload_split=_decode_payload_split,
        bounds_for_tile_split=_bounds_for_tile_split,
        nomask_range_split=_nomask_range_split,
        partial_batch=_partial_batch,
    )


@cute.jit
def decode_linear_tile_lpt_grouped(linear, q_h, batch, q_tiles, lpt_head_group: cutlass.Constexpr[int]):
    head_group = cutlass.Int32(lpt_head_group)
    group_span = q_tiles * head_group
    group_idx = linear // group_span
    group_offset = linear % group_span
    row_rank = group_offset // head_group
    within = group_idx * head_group + group_offset % head_group
    row = (q_tiles - cutlass.Int32(1)) - row_rank
    head = within % q_h
    batch_idx = within // q_h
    return row, head, batch_idx


class SdpaHelpers(NamedTuple):
    decode_initial: object
    decode_payload: object
    bounds_for_tile: object
    bounds_for_tile_qtrim: object
    resolve_seqlen_kv: object
    resolve_seqlen_q: object
    thd_decode: object
    dispatch_decode_initial: object
    dispatch_decode_payload: object
    thd_tma_offsets: object
    thd_sf_tile_bases: object


def make_sdpa_helpers(
    CFG,
    lpt_q_tiles_in_cga_units: bool = False,
    grouped_lpt: bool = False,
    lpt_head_group: int = 1,
    lpt_q_tiles: int = 0,
    kv_shared_cluster: bool = False,
    bounded_thd_decode: bool = False,
) -> SdpaHelpers:
    """Decode / bounds / THD closures for a flavor.

    ``bounded_thd_decode`` uses the device-sorted batch remap to locate a live
    THD work item without scanning later sequences. Equal endpoint tile counts
    admit direct indexing, including unequal lengths in the same tile bucket.
    Callers opt in after qualifying their geometry; other flavors keep the
    original decoder. Neither path reads sequence lengths on the host.

    ``kv_shared_cluster`` (default False = every existing kernel's trace, byte for byte): the
    CLUSTER-UNION geometry of the d512 2x2-datapath kernel, whose CGA_M CTAs each own their
    OWN ``TILE_M`` Q rows and share one K/V stream (two cta_group::2 pairs tied by TMA
    multicast and 2-count ``k/v_empty`` barriers).  Every CTA of such a cluster must run the
    IDENTICAL KV range, so the bounds height is the whole cluster's rows
    (``TILES_Q * TILE_M * CGA_M * CGA_N``) and the Q super-tile decode counts CGA_M supers
    per cluster (the caller passes its cluster rank ``cta_id_x`` where the pair kernels pass
    ``cta_in_pair``).  The default keeps the pair geometry: ``cga_tile_m = TILES_Q * TILE_M *
    CTA_MMA`` and CTA_MMA supers per cluster -- the role-split d512's two pairs own the SAME
    rows, so its cluster is still CTA_MMA supers tall (judge-confirmed: the default arm halves
    a 2x2 cluster's height to 128 and lets its two pairs compute different KV ranges under a
    mask, which deadlocks the shared ring).
    """
    _cga_m = getattr(CFG, "CGA_M", 1)
    _cta_mma = getattr(CFG, "CTA_MMA", 1)
    # Q super-tiles (TILES_Q * TILE_M-row CTA tiles) per cluster: the unit the blocked NATURAL
    # decode, the LPT q_super expansion, the THD unit decode and the bounds height all share.
    _q_supers_per_cluster = (_cga_m * getattr(CFG, "CGA_N", 1)) if kv_shared_cluster else _cta_mma
    # Live Q sub-tiles: TILES_Q, or CfgD128.Q_TILES_LIVE (1) under the two-KV-stream
    # mode, where sub-tile 1 re-uses sub-tile 0's rows; the Q span, row bases and
    # bounds follow it.  Read only when that mode is on: the decode-tile configs
    # inherit CfgD128's field defaults with a TILES_Q of their own.
    q_tiles_live = CFG.Q_TILES_LIVE if getattr(CFG, "DUAL_KV_STREAM", 0) else CFG.TILES_Q
    cga_tile_m = q_tiles_live * CFG.TILE_M * _q_supers_per_cluster
    # PackGQA head grouping for the LPT_L2 decode: the grid's head axis is in
    # PACKED heads (QH / PACK_G) and QH_PER_KH // PACK_G of them read one KV
    # head -- 1 under full packing, G / p under partial packing (d128 / d256
    # f16).  A Cfg without PACK_G packs the whole group.
    _pack_g = int(getattr(CFG, "PACK_G", 0)) or int(getattr(CFG, "QH_PER_KH", 1))
    _packed_heads_per_kv = max(1, int(getattr(CFG, "QH_PER_KH", 1)) // _pack_g) if CFG.PACK_GQA else 1
    thd_token_tile_m = cga_tile_m // (_pack_g if CFG.PACK_GQA else 1)

    @cute.jit
    def _lpt_linear(block_id):
        if cutlass.const_expr(_cga_m > 1):
            return block_id // cutlass.Int32(_cga_m)
        return block_id

    @cute.jit
    def _lpt_q_super(row, cta_in_pair):
        if cutlass.const_expr(_q_supers_per_cluster > 1):
            return row * cutlass.Int32(_q_supers_per_cluster) + cta_in_pair
        return row

    if CFG.SCHEDULER_POLICY == SCHED_NATURAL:
        if CFG.SPLIT_PIPELINE == 1:

            @cute.jit
            def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                blocked_row = (bidx // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(_q_supers_per_cluster) + cta_in_pair
                return blocked_row, bidy, bidz

            @cute.jit
            def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                blocked_row = (t0 // cutlass.Int32(CFG.CGA_M)) * cutlass.Int32(_q_supers_per_cluster) + cta_in_pair
                head = t1 & cutlass.Int32(0xFFFF)
                batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
                return blocked_row, head, batch

        else:

            @cute.jit
            def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                return bidx, bidy, bidz

            @cute.jit
            def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                head = t1 & cutlass.Int32(0xFFFF)
                batch = (t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF)
                return t0 + cta_in_pair, head, batch

    elif CFG.SCHEDULER_POLICY == SCHED_LPT_L2:
        _kv_bytes_per_row = (CFG.TILE_K + CFG.TILE_O) * CFG.BPE
        _l2_bytes = CFG.L2_SIZE_MIB * 1024 * 1024

        @cute.jit
        def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
            linear = _lpt_linear(bidx)
            q_tiles = n_q_supers // cutlass.Int32(_q_supers_per_cluster) if lpt_q_tiles_in_cga_units else n_q_supers
            if cutlass.const_expr(qh_per_kh is None or seqlen_kv is None):
                raise ValueError("SCHED_LPT_L2 decode requires qh_per_kh and seqlen_kv at every call site")
            row, head, batch = lpt_l2_tile_coords(linear, n_qh, n_batch, q_tiles, qh_per_kh, seqlen_kv, _kv_bytes_per_row, _l2_bytes)
            return _lpt_q_super(row, cta_in_pair), head, batch

        @cute.jit
        def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
            linear = _lpt_linear(t0)
            q_tiles = n_q_supers // cutlass.Int32(_q_supers_per_cluster) if lpt_q_tiles_in_cga_units else n_q_supers
            if cutlass.const_expr(qh_per_kh is None or seqlen_kv is None):
                raise ValueError("SCHED_LPT_L2 decode requires qh_per_kh and seqlen_kv at every call site")
            row, head, batch = lpt_l2_tile_coords(linear, n_qh, n_batch, q_tiles, qh_per_kh, seqlen_kv, _kv_bytes_per_row, _l2_bytes)
            return _lpt_q_super(row, cta_in_pair), head, batch

    else:
        if grouped_lpt:
            if lpt_q_tiles > 0:

                @cute.jit
                def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                    row, head, batch = decode_linear_tile_lpt_grouped(_lpt_linear(bidx), n_qh, n_batch, cutlass.Int32(lpt_q_tiles), lpt_head_group)
                    return _lpt_q_super(row, cta_in_pair), head, batch

                @cute.jit
                def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                    row, head, batch = decode_linear_tile_lpt_grouped(_lpt_linear(t0), n_qh, n_batch, cutlass.Int32(lpt_q_tiles), lpt_head_group)
                    return _lpt_q_super(row, cta_in_pair), head, batch

            else:

                @cute.jit
                def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                    q_tiles = n_q_supers // cutlass.Int32(_q_supers_per_cluster) if lpt_q_tiles_in_cga_units else n_q_supers
                    row, head, batch = decode_linear_tile_lpt_grouped(_lpt_linear(bidx), n_qh, n_batch, q_tiles, lpt_head_group)
                    return _lpt_q_super(row, cta_in_pair), head, batch

                @cute.jit
                def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                    q_tiles = n_q_supers // cutlass.Int32(_q_supers_per_cluster) if lpt_q_tiles_in_cga_units else n_q_supers
                    row, head, batch = decode_linear_tile_lpt_grouped(_lpt_linear(t0), n_qh, n_batch, q_tiles, lpt_head_group)
                    return _lpt_q_super(row, cta_in_pair), head, batch

        else:

            @cute.jit
            def _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                linear = _lpt_linear(bidx)
                q_tiles = n_q_supers // cutlass.Int32(_q_supers_per_cluster) if lpt_q_tiles_in_cga_units else n_q_supers
                row, head, batch = lpt_tile_coords(linear, n_qh, n_batch, q_tiles)
                return _lpt_q_super(row, cta_in_pair), head, batch

            @cute.jit
            def _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh=None, seqlen_kv=None):
                linear = _lpt_linear(t0)
                q_tiles = n_q_supers // cutlass.Int32(_q_supers_per_cluster) if lpt_q_tiles_in_cga_units else n_q_supers
                row, head, batch = lpt_tile_coords(linear, n_qh, n_batch, q_tiles)
                return _lpt_q_super(row, cta_in_pair), head, batch

    @cute.jit
    def _bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, qh_per_kh: int = 1):
        # Token capacity of one CGA super-tile: TILES_Q * TILE_M rows hold
        # rows / p tokens when packing (CFG.PACK_GQA; ``qh_per_kh`` is the
        # caller's packed group p = HEADS_PER_TILE), else one token per row.
        tokens_per_super = (q_tiles_live * CFG.TILE_M) // qh_per_kh if CFG.PACK_GQA else q_tiles_live * CFG.TILE_M
        cga_base_super = q_super_idx - cta_in_pair
        q_row_coord = cga_base_super * cutlass.Int32(tokens_per_super)
        return compute_kv_loop_bounds(
            q_row_coord,
            seqlen_q,
            seqlen_kv,
            CFG.WINDOW_LEFT,
            CFG.MASK_FLAGS,
            CFG.TILE_N,
            cga_tile_m // qh_per_kh if CFG.PACK_GQA else cga_tile_m,
            bottom_right=bool(CFG.BOTTOM_RIGHT),
            window_right=int(CFG.WINDOW_RIGHT),
        )

    @cute.jit
    def _bounds_for_tile_qtrim(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, seq_q_lens_addr, batch_idx, qh_per_kh: int = 1):
        """bounds_for_tile + cuDNN-style dead-Q-tile KV-loop collapse.

        Mirrors cuDNN fort (mma_pipeline_op_native_sdpa_prefill_sm100_nonfp8
        .cpp:916-921): when the CGA tile's base Q row is at/past this batch's
        actual Q length (SEQ_Q_LENS_PRESENT dense padded-Q trim; q lens are
        the SEPARATE (B,)-int32 ``seq_q_lens_addr`` kernel parameter — cuDNN
        SEQLEN_Q / FA seqused_q style — ``0`` unless the flag is set, so
        the read below folds out with the branch), collapse the KV loop to
        empty (right := left, matching the SWA empty-tile machinery) — the
        grid stays padded-sized and a dead tile costs prologue+epilogue only.
        q_len_b == 0 (whole batch dead) collapses every tile since the base
        row coord is always >= 0.  Under PackGQA the dead-tile
        compare is the tile's base TOKEN (rows // G) against the per-batch
        Q length — the same token-space value the packed bounds use.  Either way
        the row coord is the SAME cga-base value _bounds_for_tile uses and q
        lens are per-batch constants, so every warp group calling this helper
        sees identical (collapsed) bounds and the barrier handshakes stay in
        lockstep.  The epilogue's SEQ_Q_LENS_PRESENT trim (applied after the
        sink fold) already forces O := 0 / LSE := -inf for every row of a
        collapsed tile.
        """
        b = _bounds_for_tile(q_super_idx, seqlen_q, seqlen_kv, cta_in_pair, qh_per_kh)
        if cutlass.const_expr(int(getattr(CFG, "SEQ_Q_LENS_PRESENT", 0)) == 1):
            tokens_per_super = (q_tiles_live * CFG.TILE_M) // qh_per_kh if CFG.PACK_GQA else q_tiles_live * CFG.TILE_M
            cga_base_super = q_super_idx - cta_in_pair
            q_row_coord = cga_base_super * cutlass.Int32(tokens_per_super)
            arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
            q_len_b = cutlass.Int32(arr[cutlass.Int32(batch_idx)])  # batch_idx may arrive as a raw arith value from the decode
            tile_dead = q_row_coord >= q_len_b
            dead_lo = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.unmasked_lo.ir_value()))
            dead_hi = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.unmasked_hi.ir_value()))
            dead_right = cutlass.Int32(arith.select(tile_dead.ir_value(), b.left.ir_value(), b.right.ir_value()))
            return KvLoopBounds(left=b.left, unmasked_lo=dead_lo, unmasked_hi=dead_hi, right=dead_right)
        return b

    @cute.jit
    def _resolve_seqlen_kv(seq_kv_lens_tensor, batch_idx, scalar_seqlen_kv):
        if cutlass.const_expr(CFG.SEQ_KV_LENS_PRESENT == 1):
            arr = cutlass.make_array_view(seq_kv_lens_tensor)
            return cutlass.Int32(arr[cutlass.Int32(batch_idx)])  # batch_idx may be a raw arith value after a payload read
        return scalar_seqlen_kv

    @cute.jit
    def _resolve_seqlen_q(seq_kv_lens_tensor, batch_idx, scalar_seqlen_q, n_batch, seq_q_lens_addr=0):
        """Per-batch Q length for the bottom-right causal diagonal.

        Bottom-right anchors the diagonal at the per-batch corner
        (seq_len_q[b], seq_len_kv[b]).  THD reads the actual Q length as the
        cu_seqlen_q difference from the packed [kv_lens | cu_q | cu_kv] metadata
        buffer (same layout _thd_decode / _thd_tma_offsets read); dense padded
        graphs carrying per-batch Q lengths read the SEPARATE (B,)-int32
        ``seq_q_lens_addr`` (cuDNN SEQLEN_Q style), clamped to [0, S_q].
        KV-only padding (and every non-BR mask, where seqlen_q only feeds the
        unused diagonal) keeps the scalar S_q, so both reads fold out unless
        CAUSAL_BOTTOM_RIGHT is set together with THD_VARLEN or
        SEQ_Q_LENS_PRESENT (mutually exclusive by _validate_params).
        """
        if cutlass.const_expr(int(getattr(CFG, "THD_VARLEN", 0)) == 1 and int(CFG.BOTTOM_RIGHT) == 1):
            cu = cutlass.make_array_view(seq_kv_lens_tensor)
            q0 = n_batch
            return cutlass.Int32(cu[q0 + batch_idx + cutlass.Int32(1)]) - cutlass.Int32(cu[q0 + batch_idx])
        if cutlass.const_expr(int(getattr(CFG, "SEQ_Q_LENS_PRESENT", 0)) == 1 and int(CFG.BOTTOM_RIGHT) == 1):
            arr = cute.make_tensor(cute.make_ptr(cutlass.Int32, seq_q_lens_addr, cute.AddressSpace.gmem, assumed_align=4), cute.make_layout(1 << 24))
            return cute.math.max(cutlass.Int32(0), cute.math.min(cutlass.Int32(arr[cutlass.Int32(batch_idx)]), scalar_seqlen_q))
        return scalar_seqlen_q

    _thd_on = int(getattr(CFG, "THD_VARLEN", 0))

    @cute.jit
    def _thd_decode_prefix(linear_cta, meta_t, n_batch, n_qh, cta_in_pair):
        u = linear_cta // cutlass.Int32(CFG.CGA_M)
        split = u % cutlass.Int32(CFG.SPLIT_KV)
        u = u // cutlass.Int32(CFG.SPLIT_KV)
        meta = cutlass.make_array_view(meta_t)
        b, local = n_batch, cutlass.Int32(0)
        if n_batch == cutlass.Int32(1):
            b, local = cutlass.Int32(0), u
        elif n_batch > cutlass.Int32(1):
            prefix0 = THD_WORK_PREFIX_OFF(n_batch)
            if u < cutlass.Int32(meta[prefix0 + n_batch]):
                lo, hi = cutlass.Int32(0), n_batch
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if cutlass.Int32(meta[prefix0 + mid + 1]) <= u:
                        lo = mid + 1
                    else:
                        hi = mid
                b = cutlass.Int32(meta[THD_REMAP_OFF(n_batch) + lo])
                local = u - cutlass.Int32(meta[prefix0 + lo])
        f_batch, f_head, f_qc = n_batch, cutlass.Int32(0), cutlass.Int32(0)
        if b < n_batch:
            q_len = cutlass.Int32(meta[n_batch + b + 1]) - cutlass.Int32(meta[n_batch + b])
            cb = (q_len + cutlass.Int32(thd_token_tile_m - 1)) // cutlass.Int32(thd_token_tile_m)
            if local < cb * n_qh:
                # Policies reorder the SAME live THD work list; they never decode
                # the padded rectangular cache-shape envelope. NATURAL retains
                # head-major ascending rows. LPT visits the heavier causal rows
                # across all heads first; LPT_L2 keeps a KV-sharing head group
                # together while reversing its rows. Other flavors retain their
                # head-major THD order until their policy contracts are validated.
                head = local // cb
                row = local % cb
                if cutlass.const_expr(CFG.DTYPE_QKV in (2, 3) and CFG.TILE_K in (64, 128, 256)):
                    if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_LPT):
                        head = local % n_qh
                        row = cb - 1 - local // n_qh
                    elif cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_LPT_L2):
                        group = cutlass.Int32(_packed_heads_per_kv if CFG.PACK_GQA else CFG.QH_PER_KH)
                        head = (local // (cb * group)) * group + local % group
                        row = cb - 1 - (local // group) % cb
                f_batch, f_head, f_qc = b, head, row
        return f_qc * cutlass.Int32(_q_supers_per_cluster) + cta_in_pair, f_head, f_batch + split * n_batch

    @cute.jit
    def _thd_decode(linear_cta, seq_kv_lens_t, n_batch, n_qh, cta_in_pair):
        """Unit id -> (q_super, head, batch [+ split * n_batch]) for every THD flavor.

        Through the remap-ordered work prefix the setup launch publishes
        (``write_thd_work_prefix``; ``THD_WORK_PREFIX_OFF``): a binary search over
        B + 1 words. Dead units (uid past the live total) keep
        ``batch == n_batch``: in-bounds metadata reads make every downstream
        consumer a no-op (empty KV range, no O/LSE writes).
        """
        return _thd_decode_prefix(linear_cta, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)

    @cute.jit
    def _thd_decode_bounded(linear_cta, seq_kv_lens_t, n_batch, n_qh, cta_in_pair):
        u = linear_cta // cutlass.Int32(CFG.CGA_M)
        split = cutlass.Int32(0)
        if cutlass.const_expr(getattr(CFG, "SPLIT_KV", 1) > 1):
            # Split is the low digit of the live ragged work list. Both
            # initial admission and persistent claims use this same mapping.
            split = u % cutlass.Int32(CFG.SPLIT_KV)
            u = u // cutlass.Int32(CFG.SPLIT_KV)
        cu = cutlass.make_array_view(seq_kv_lens_t)
        cuq0 = n_batch
        acc = cutlass.Int32(0)
        # DEAD-unit sentinel (issue #552 over-launch): a unit no sequence
        # claims (u >= sum of live units) keeps batch == n_batch.  That index
        # makes every downstream consumer a no-op through IN-BOUNDS metadata
        # reads: _resolve_seqlen_kv reads meta[n_batch] = cu_q[0] = 0 (empty
        # KV range in every role), the epilogue's per-sequence Q length
        # cu[2n+1]-cu[2n] goes negative (LSE predicate never fires), and the
        # O-store role skips the TMA store explicitly (batch >= n_batch).
        f_batch = n_batch
        f_head = cutlass.Int32(0)
        f_qc = cutlass.Int32(0)
        found_cb = cutlass.Int32(1)
        found_local = cutlass.Int32(0)
        done = cutlass.Int32(0)
        remap0 = cutlass.Int32(3) * n_batch + cutlass.Int32(2)
        search = cutlass.Int32(1)
        # batch_remap is sorted by decreasing Q length. Rounding length to
        # tile count is monotone, so equal endpoint counts prove that EVERY
        # batch has the same count, including unequal lengths in one bucket.
        # All observations are current device metadata; no host promise.
        if n_batch > cutlass.Int32(0):
            first_b = cutlass.Int32(cu[remap0])
            last_b = cutlass.Int32(cu[remap0 + n_batch - cutlass.Int32(1)])
            first_s = cutlass.Int32(cu[cuq0 + first_b + cutlass.Int32(1)]) - cutlass.Int32(cu[cuq0 + first_b])
            last_s = cutlass.Int32(cu[cuq0 + last_b + cutlass.Int32(1)]) - cutlass.Int32(cu[cuq0 + last_b])
            first_cb = (first_s + cutlass.Int32(thd_token_tile_m - 1)) // cutlass.Int32(thd_token_tile_m)
            last_cb = (last_s + cutlass.Int32(thd_token_tile_m - 1)) // cutlass.Int32(thd_token_tile_m)
            if first_cb == last_cb:
                search = cutlass.Int32(0)
                if first_cb > cutlass.Int32(0):
                    per_batch = first_cb * n_qh
                    ordinal = u // per_batch
                    if ordinal < n_batch:
                        f_batch = cutlass.Int32(cu[remap0 + ordinal])
                        found_cb = first_cb
                        found_local = u % per_batch
                        done = cutlass.Int32(1)
        i = cutlass.Int32(0)
        # Preserve longest-first remapping and the DEAD-unit sentinel. Empty
        # batches consume zero work. No host knowledge of lengths is needed.
        while (i < n_batch) & (done == cutlass.Int32(0)) & (search != cutlass.Int32(0)):
            b = cutlass.Int32(cu[remap0 + i])
            s_i = cutlass.Int32(cu[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[cuq0 + b])
            cb = (s_i + cutlass.Int32(thd_token_tile_m - 1)) // cutlass.Int32(thd_token_tile_m)
            units_b = cb * n_qh
            if u < acc + units_b:
                f_batch = b
                found_cb = cb
                found_local = u - acc
                done = cutlass.Int32(1)
            acc = acc + units_b
            i = i + cutlass.Int32(1)
        # Quotient/remainder decoding is needed only for the matching batch.
        # cb is positive here; keep head/row zero for an unclaimed work unit.
        if done != cutlass.Int32(0):
            f_head = found_local // found_cb
            f_qc = found_local % found_cb
            if cutlass.const_expr(CFG.DTYPE_QKV in (2, 3) and CFG.TILE_K in (64, 128, 256)):
                if cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_LPT):
                    f_head = found_local % n_qh
                    f_qc = found_cb - cutlass.Int32(1) - found_local // n_qh
                elif cutlass.const_expr(CFG.SCHEDULER_POLICY == SCHED_LPT_L2):
                    group = cutlass.Int32(_packed_heads_per_kv if CFG.PACK_GQA else CFG.QH_PER_KH)
                    f_head = (found_local // (found_cb * group)) * group + found_local % group
                    f_qc = found_cb - cutlass.Int32(1) - (found_local // group) % found_cb
        q_super = f_qc * cutlass.Int32(_q_supers_per_cluster) + cta_in_pair
        return q_super, f_head, f_batch + split * n_batch

    _thd_decode_impl = _thd_decode_bounded if bounded_thd_decode else _thd_decode

    @cute.jit
    def _dispatch_decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
        if cutlass.const_expr(_thd_on):
            return _thd_decode_impl(bidx, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)
        if cutlass.const_expr(CFG.PACK_GQA):
            qh_per_kh = cutlass.Int32(_packed_heads_per_kv)
        return _decode_initial(bidx, bidy, bidz, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh, seqlen_kv)

    @cute.jit
    def _dispatch_decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, seq_kv_lens_t, qh_per_kh=None, seqlen_kv=None):
        if cutlass.const_expr(_thd_on):
            return _thd_decode_impl(t0, seq_kv_lens_t, n_batch, n_qh, cta_in_pair)
        if cutlass.const_expr(CFG.PACK_GQA):
            qh_per_kh = cutlass.Int32(_packed_heads_per_kv)
        return _decode_payload(t0, t1, cta_in_pair, n_q_supers, n_qh, n_batch, qh_per_kh, seqlen_kv)

    @cute.jit
    def _thd_tma_offsets(seq_kv_lens_t, batch_idx, n_batch):
        if cutlass.const_expr(_thd_on):
            cu = cutlass.make_array_view(seq_kv_lens_t)
            q_off = cutlass.Int32(cu[n_batch + batch_idx])
            kv_off = cutlass.Int32(cu[cutlass.Int32(2) * n_batch + cutlass.Int32(1) + batch_idx])
            return q_off, kv_off, cutlass.Int32(0)
        return cutlass.Int32(0), cutlass.Int32(0), batch_idx

    @cute.jit
    def _thd_sf_tile_bases(seq_kv_lens_t, batch_idx, n_batch):
        if cutlass.const_expr(_thd_on):
            cu = cutlass.make_array_view(seq_kv_lens_t)
            q0 = n_batch
            k0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
            sfq = cutlass.Int32(0)
            sfk = cutlass.Int32(0)
            for b in cutlass.range(0, batch_idx, 1, unroll=1):
                s_q = cutlass.Int32(cu[q0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[q0 + b])
                s_kv = cutlass.Int32(cu[k0 + b + cutlass.Int32(1)]) - cutlass.Int32(cu[k0 + b])
                sfq = sfq + (s_q + cutlass.Int32(CFG.TILE_M - 1)) // cutlass.Int32(CFG.TILE_M)
                sfk = sfk + (s_kv + cutlass.Int32(CFG.TILE_N - 1)) // cutlass.Int32(CFG.TILE_N)
            return cute.arch.make_warp_uniform(sfq), cute.arch.make_warp_uniform(sfk)
        return cutlass.Int32(0), cutlass.Int32(0)

    return SdpaHelpers(
        decode_initial=_decode_initial,
        decode_payload=_decode_payload,
        bounds_for_tile=_bounds_for_tile,
        bounds_for_tile_qtrim=_bounds_for_tile_qtrim,
        resolve_seqlen_kv=_resolve_seqlen_kv,
        resolve_seqlen_q=_resolve_seqlen_q,
        thd_decode=_thd_decode_impl,
        dispatch_decode_initial=_dispatch_decode_initial,
        dispatch_decode_payload=_dispatch_decode_payload,
        thd_tma_offsets=_thd_tma_offsets,
        thd_sf_tile_bases=_thd_sf_tile_bases,
    )


# ======================================================================# Fused epilogue gate  --  O := O * sigmoid(G)   (Rubin d256 f16 / per-tensor FP8 / MXFP8)
# =============================================================================
# Shared between the three SM107 d256 prefill kernels (``sm107/prefill_d256_f16.py``,
# ``sm107/prefill_d256_fp8.py``, ``sm107/prefill_d256_mxfp8.py``); each kernel splices these under
# ``cutlass.const_expr(CFG.EPILOGUE_GATE)`` at its ``EPILOGUE_FUSION_SEAM``
# tokens (one per splice; a source test counts them per kernel).  The gate ``SmemTile(`` itself is constructed INLINE in each kernel
# (multi-line, ``desc_version=DESC_VERSION``) so the per-kernel "every SmemTile
# takes the module DESC_VERSION" source test keeps counting truthfully; this
# module owns the geometry, the barriers, the TMA issue, the chunk-offset
# arithmetic and the packed epilogue math.
#
# The gate tile is per-CTA and full-width even under cga2: the decode folds
# ``cta_in_pair`` into ``q_super_idx``, so each CTA of a pair owns a distinct
# 128-row Q block and all TILE_O columns (only K rows / V columns split per
# CTA).  Hence ``cta_group=1`` and NO ``* CFG.CTA_MMA`` on the transaction bytes
# -- contrast ``qTmaTransactionBytes = qBufferElems * BPE * CTA_MMA`` (P9).


class GateGeometry(NamedTuple):
    """Host-side gate staging geometry, derived from GATE's OWN byte width.

    Never from O's: on the FP8 kernel with e4m3 O the two differ (O walks 2
    subtiles of 128 elems; the bf16 gate walks 4 of 64), and a mis-derivation
    passes every bf16-O test and fails only e4m3-O."""

    swz_bytes: int  # o_swz_bytes(TILE_O, gate_bpe)            -> 128 at d256 / 2 B
    tma_iters: int  # (TILE_O * gate_bpe) // swz_bytes         -> 4 subtiles
    tma_granu_elems: int  # swz_bytes // gate_bpe                   -> 64 elems per subtile row
    buffer_elems: int  # TILE_M * TILE_O (ONE q tile per load)   -> 32768 elems (1 stage)
    tx_bytes: int  # buffer_elems * gate_bpe                 -> 65536 B, NO * CTA_MMA
    box_dims: tuple  # (1, TILE_M, 1, tma_granu_elems)
    layout_enum: int  # the kernel's _SWZ_ENUM[swz_bytes]         (128: 2, 64: 4, 32: 6)
    smem_swizzle: object  # cutlass.Swizzle({128: 3, 64: 2, 32: 1}[swz_bytes], 4, 3)
    d_block: int  # TILE_O // tma_iters                     -> 64 elems per lane row inside one subtile
    granu_local: int  # TILE_M * d_block                        -> 8192 elems per subtile


def gate_geometry(CFG, *, gate_bpe: int, swz_enum: dict) -> GateGeometry:
    """Pure host arithmetic; called at module import by every d256 module.

    Swizzle has BOTH jobs here (frost-kernels.md S5): the TMA descriptor WRITES
    the tile under ``swz_bytes`` (so the SMEM side must match it -- correctness)
    AND correction lanes READ it one q row per lane at a ``d_block * gate_bpe``
    = 128 B stride (= the bank cycle, so an unswizzled tile would be a 32-way
    conflict).  ``Swizzle(3, 4, 3)`` has ``MBase + SShift = 7 = log2(128 B)``.
    """
    # ONE q tile per ``issue_gate_load`` call: ``tma_load_tile`` delivers
    # ``tma_iters * TILE_M * tma_granu_elems`` = ``TILE_M * TILE_O`` elements, and
    # ``expect_tx`` must equal exactly the bytes delivered (P3) -- so the buffer
    # and the transaction are sized per Q TILE, never per ``TILES_Q``.  A flavor
    # with TILES_Q > 1 (d128/d192) needs a per-tile issue loop + a per-tile
    # arm before the gate can be wired there; refuse at import rather than
    # over-arm the mbar (an over-count is a hang, not an error).
    if CFG.TILES_Q != 1:
        raise ValueError(f"gate_geometry: the epilogue gate stages ONE q tile per load (TILES_Q == 1); got TILES_Q={CFG.TILES_Q}")
    swz_bytes = _o_swz_bytes(CFG.TILE_O, gate_bpe)
    tma_iters = (CFG.TILE_O * gate_bpe) // swz_bytes
    tma_granu_elems = swz_bytes // gate_bpe
    buffer_elems = CFG.TILE_M * CFG.TILE_O
    d_block = CFG.TILE_O // tma_iters
    return GateGeometry(
        swz_bytes=swz_bytes,
        tma_iters=tma_iters,
        tma_granu_elems=tma_granu_elems,
        buffer_elems=buffer_elems,
        tx_bytes=buffer_elems * gate_bpe,
        box_dims=(1, CFG.TILE_M, 1, tma_granu_elems),
        layout_enum=swz_enum[swz_bytes],
        smem_swizzle=cutlass.Swizzle({128: 3, 64: 2, 32: 1}[swz_bytes], 4, 3),
        d_block=d_block,
        granu_local=CFG.TILE_M * d_block,
    )


@cute.jit
def issue_gate_load(sGate, tma_gate, mb_gate_full, tx_bytes: cutlass.Constexpr[int], head_idx, q_row, tma_batch):
    """TMA-LDG warp: arm ``expect_tx`` (ONE lane, P16 value predicate) and queue
    the gate tile's subtiles with ``cta_group=1`` so every byte lands on THIS
    CTA's mbar.  Folds to nothing when ``sGate`` is None (gate compiled out).

    Call it on BOTH arms of the empty-mainloop branch (P14): the epilogue runs
    on every tile -- empty-KV and padded-Q-trimmed ones included -- and waits
    ``mb_gate_full`` each time, so every tile must consume exactly one load.
    Issue position: AFTER the kv loop.  The TMA engine serves requests in
    order and the gate is the one operand with slack (consumed only in the
    epilogue); queued before K/V it delays the operands BMM1 is blocked on
    (measured -2.4 % on the fused kernel for the after-Q position).
    """
    if cutlass.const_expr(sGate is not None):
        mb_gate_full.arrive(n_bytes=tx_bytes, pred=nvvm.elect_sync())
        tma_load_tile(
            sGate,
            tma_gate(cutlass.Int32(0), head_idx, q_row, tma_batch),
            mb_gate_full.smem_ptr,
            cta_group=1,
        )


def gate_chunk_smem_offset(chunk_idx_total: int, o_chunk: int, gg: GateGeometry, tid_in_wg) -> cutlass.Int32:
    """Trace-time helper (plain Python, like ``row_max_reduction``): SMEM element
    offset of this lane's ``o_chunk``-wide gate chunk.  Subtile-major walk in
    GATE geometry (lane = one q row, ``d_block`` elems per row per subtile)."""
    c0 = chunk_idx_total * o_chunk
    return cutlass.Int32((c0 // gg.d_block) * gg.granu_local + (c0 % gg.d_block)) + tid_in_wg * cutlass.Int32(gg.d_block)


def load_gate_chunk(sGate_base, smem_offset, gg: GateGeometry, o_chunk: int) -> list:
    """Trace-time helper: one swizzled LDS of ``o_chunk`` gate elements, widened
    to fp32, as a Python list.

    Alignment 32 is the TRUTHFUL byte alignment of a 16-elem x 2 B chunk at
    offsets 0 / 32 / 64 / 96 B inside the 128 B row (64 would overstate the odd
    chunks); CuTe vectorises at 16 B either way (LDS.128 x 2 per chunk).
    Plain ``for`` + ``append``: ``range_constexpr`` is statement-only and a
    comprehension is not preprocessed (frost-gotchas)."""
    vec = sGate_base.subview(smem_offset).data_ptr().load_swizzled(gg.smem_swizzle, 32, count=o_chunk).to(cutlass.Float32)
    out = []
    for i in range(o_chunk):
        out.append(vec[i])
    return out


def gate_epilogue_pairs(o_scaled, g_vals: list, half_opaque, o_chunk: int) -> list:
    """Trace-time helper: the gate math, algebraically minimised and PACKED.

        sigmoid(g) = tanh(g/2)/2 + 1/2
        o*inv_sum*sigmoid(g) = h*tanh(g/2) + h        with h = o*(inv_sum/2)

    The caller has ALREADY folded the 1/2 into ``inv_sum`` (``gate_inv_sum``),
    so ``o_scaled`` IS ``h`` and both of sigmoid's constants are gone.  Per PAIR
    of elements: 1 FMUL2 (g/2) + 2 MUFU.TANH + 1 FFMA2 (h*t + h, multiplicand
    and addend the same register).  Verified in SASS as the delta against the
    gate compiled out (256 elems/tile): +128 FMUL2, +128 FFMA2, +256 MUFU.TANH,
    0 scalar FMUL/FFMA.  ``half_opaque`` must be ``gate_half_opaque()`` -- a
    constant float reaching ``inline_ptx`` ICEs libNVVM (frost-tile-dsl S7).

    The caller applies the dead-row SELECT per element AFTER these values
    (never a multiply-by-zero: the TMEM residue behind ``h`` can be NaN).

    A caller that also reports ``Amax_O`` folds it on ``o_scaled`` (= h, the
    UNGATED value halved -- exactly, see ``gate_inv_sum``) and doubles the
    running max once per tile: Amax_O is a statistic of the sdpa node, which
    precedes the gate, so it must not depend on G
    (``sm107/prefill_d256_fp8.py`` and ``sm107/prefill_d256_mxfp8.py``,
    corr_chunk / corr_release)."""
    out = []
    for p in range(o_chunk // 2):
        a, b = g_vals[2 * p], g_vals[2 * p + 1]
        ha, hb = o_scaled[2 * p], o_scaled[2 * p + 1]
        sa, sb = fmul2(a, b, half_opaque, half_opaque)
        ta = cute.math.tanh(sa, approx=True)
        tb = cute.math.tanh(sb, approx=True)
        va, vb = ffma2(ta, tb, ha, hb, ha, hb)
        out.append(va)
        out.append(vb)
    return out


def gate_inv_sum(inv_sum):
    """``inv_sum / 2`` -- folds sigmoid's 1/2 into the scale the epilogue applies anyway.

    Exact for the amax fold: ``o * (inv_sum * 0.5)`` is ``0.5 * RN(o * inv_sum)``
    bit-for-bit whenever the product is a normal fp32 (a power-of-two scale
    commutes with RN), and ``inv_sum <= scale_o * descale_v`` (``sum >= 1``; on
    MXFP8, which has no per-tensor scale, ``inv_sum <= 1``) -- an upper bound
    only, since a sink can shrink ``inv_sum`` without limit -- keeps the tile's
    max element a normal fp32, which is what lets the FP8 / MXFP8 kernels'
    Amax_O fold consume ``h`` and double once per tile."""
    return inv_sum * cutlass.Float32(0.5)


def o_epilogue_convert_store(o_scaled, row_empty, amax_acc, smem_ptr, *, n: int, apply_select: bool, out_dtype, swizzle):
    """One O block of the sg1 epilogue AFTER the ``o_fp32 * beta`` multiply: SELECT the dead-row zero, fold |O| into the
    running Amax_O, pack to the O dtype, swizzled SMEM store.  Trace-time helper (plain Python over traced values, like
    :func:`gate_epilogue_pairs`): it emits straight-line IR at the call site and holds NO control flow, so a caller may
    place it under a runtime branch (the d512 kernels' dead-row fast path) -- never a collective op inside it.

    ``apply_select=True`` is the classic body: ``select(row_empty, 0, x)`` per element (a SELECT, never ``* 0`` -- the TMEM
    residue of an empty row can be a NaN bit pattern, sdpa-invariants.md s2) and the amax fold over the SUBSTITUTED values,
    so a dead row cannot poison Amax_O.  ``apply_select=False`` is the same body with ``select(False, 0, x) == x`` folded:
    legal ONLY when the caller has proven no lane of the executing warp holds a dead row (a warp-uniform ``vote.any`` on
    ``row_empty``), which makes the two arms bit-identical.

    ``amax_acc`` is the fp32 Amax_O accumulator (an :func:`opaque_f32_zero`-seeded value, folded through ``fmax_f32`` so it
    lowers to FMNMX3, frost-tile-dsl.md s9) or ``None`` for a kernel without Amax_O (half-precision O); the new accumulator
    (or ``None``) is returned.  ``o_scaled[i]`` are the ``n`` fp32 elements of the block; ``smem_ptr`` is the block's swizzled
    SMEM destination (``store_swizzled`` at ``alignment=64``, the epilogue's 16-B granule)."""
    elems = []
    for i in range(n):
        e = o_scaled[i]
        if apply_select:
            e = cutlass.Float32(arith.select(row_empty.ir_value(), cutlass.Float32(0.0).ir_value(), e.ir_value()))
        elems.append(e)
    if amax_acc is not None:
        for e in elems:
            amax_acc = fmax_f32(amax_acc, cute.math.abs(e))
    o_out = cutlass.Vector.from_elements(tuple(elems), cutlass.Float32).to(out_dtype)
    smem_ptr.store_swizzled(o_out, alignment=64, swizzle=swizzle)
    return amax_acc


def gate_half_opaque():
    """An opaque 0.5f for ``fmul2``: a constant float operand into inline_ptx
    gets the ``n`` immediate constraint and ICEs libNVVM (frost-tile-dsl S7).
    Hoist it once per tile, not per chunk."""
    return opaque_f32_zero() + cutlass.Float32(0.5)


class SdpaOperandTensors(NamedTuple):
    """The tensors a prefill host hands to the TMA encodes and the launch, built from pointers."""

    q: cute.Tensor
    k: cute.Tensor
    v: cute.Tensor
    o: cute.Tensor
    lse: Optional[cute.Tensor]
    sinks: cute.Tensor
    meta: cute.Tensor
    o_desc: cute.Tensor
    thd_q_lens: Optional[cute.Tensor]
    thd_kv_lens: Optional[cute.Tensor]
    o_partial: Optional[cute.Tensor]
    block_table: Optional[cute.Tensor]
    block_table_v: Optional[cute.Tensor]


def _vec(ptr, n):
    return cute.make_tensor(ptr, cute.make_layout((n,), stride=(1,)))


def _bshd(ptr, batch, seq, heads, d, strides, thd):
    """(B, S, H, D) over the caller's (batch, seq, head) strides. A packed THD operand has
    batch extent 1 and binds the seq stride there (never stepped, GitHub #980)."""
    bs, ss, hs = (cutlass.Int64(s) for s in strides)  # Int64 leaves: TMA-unit scaling happens in the leaf's width
    if thd:
        return cute.make_tensor(ptr, cute.make_layout((1, seq, heads, d), stride=(ss, ss, hs, 1)))
    return cute.make_tensor(ptr, cute.make_layout((batch, seq, heads, d), stride=(bs, ss, hs, 1)))


def sdpa_operand_tensors(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    sinks_ptr,
    meta_ptr,
    o_desc_ptr,
    problem_size,
    q_strides,
    k_strides,
    v_strides,
    o_strides,
    lse_strides,
    lse_ext,
    thd_q_lens_ptr,
    thd_kv_lens_ptr,
    thd_lens_form,
    o_partial_ptr,
    *,
    d_qk,
    d_v,
    lse_kind,
    thd,
    split_kv,
    tensor_map_qwords,
    paged=False,
    page_size=0,
    block_table_ptr=None,
    block_table_v_ptr=None,
    table_strides=(0, 0),
    n_pages=0,
    o_pack=1,
    table_v_strides=None,
) -> SdpaOperandTensors:
    """Pointer + stride prologue shared by every SM100 / SM107 prefill host (called while the
    host traces, so the branches below are static).

    ``problem_size`` is (B, QH, KH, SQ, SKV, 0); THD passes packed totals as SQ/SKV, paged K/V pass
    max_pages*page_size. Strides are the caller's (batch, seq, head) element strides; paged K/V are
    (n_pages, page_size, KH, D) pools with (page, row, head) strides. ``lse_kind``: "dense"
    (B*split_kv, QH, SQ) in ``lse_strides``; "token" (SQ, QH) packed; "head" (1, QH, lse_ext);
    "padded" (B, QH, lse_ext, 1) in ``lse_strides``. ``lse_ptr`` None compiles the store out."""
    B, QH, KH, SQ, SKV, _ = problem_size
    q = _bshd(q_ptr, B, SQ, QH, d_qk, q_strides, thd)
    packed_split = thd and split_kv > 1
    o = _bshd(o_ptr, split_kv if packed_split else B * split_kv, SQ, QH, d_v // o_pack, o_strides, thd and not packed_split)
    if paged:
        k = _bshd(k_ptr, n_pages, page_size, KH, d_qk, k_strides, False)
        v = _bshd(v_ptr, n_pages, page_size, KH, d_v, v_strides, False)
    else:
        k = _bshd(k_ptr, B, SKV, KH, d_qk, k_strides, thd)
        v = _bshd(v_ptr, B, SKV, KH, d_v, v_strides, thd)
    if lse_ptr is None:
        lse = None
    elif lse_kind == "token":
        lse = cute.make_tensor(lse_ptr, cute.make_layout((SQ, QH), stride=(QH, 1)))
    elif lse_kind == "head":
        lse = cute.make_tensor(lse_ptr, cute.make_layout((1, QH, lse_ext), stride=(QH * lse_ext, lse_ext, 1)))
    elif lse_kind == "padded":
        l0, l1, l2 = lse_strides
        lse = cute.make_tensor(lse_ptr, cute.make_layout((B, QH, lse_ext, 1), stride=(l0, l1, l2, 1)))
    else:
        l0, l1, l2 = lse_strides
        lse = cute.make_tensor(lse_ptr, cute.make_layout((split_kv if packed_split else B * split_kv, QH, SQ), stride=(l0, l1, l2)))
    sinks = _vec(sinks_ptr, QH)
    if thd:
        # [seq_kv(B) | cu_q(B+1) | cu_k(B+1) | remap(B) | live | ctr | unit prefix(B+1)] and (B + 3) O/K/V descriptor slots
        meta = _vec(meta_ptr, THD_WORK_META_WORDS(B))
        o_desc = _vec(o_desc_ptr, (B + 3) * tensor_map_qwords)
        q_lens = None if thd_q_lens_ptr is None else _vec(thd_q_lens_ptr, B + (thd_lens_form & 1))
        kv_lens = None if thd_kv_lens_ptr is None else _vec(thd_kv_lens_ptr, B + ((thd_lens_form >> 1) & 1))
    else:
        meta, o_desc, q_lens, kv_lens = _vec(meta_ptr, B), _vec(o_desc_ptr, 1), None, None
    o_partial = None if o_partial_ptr is None else o  # the split slab: same layout as O, written in fp32
    table = table_v = None
    if paged:
        max_pages = SKV // cutlass.Int32(page_size)
        t_bs, t_ps = table_strides
        table = cute.make_tensor(block_table_ptr, cute.make_layout((B, max_pages), stride=(t_bs, t_ps)))
        v_bs, v_ps = table_strides if table_v_strides is None else table_v_strides
        table_v = cute.make_tensor(block_table_v_ptr, cute.make_layout((B, max_pages), stride=(v_bs, v_ps)))
    return SdpaOperandTensors(q, k, v, o, lse, sinks, meta, o_desc, q_lens, kv_lens, o_partial, table, table_v)


def sdpa_gate_tensor(gate_ptr, problem_size, d_v, gate_strides, thd=False):
    """The fused epilogue gate G (O's (B, SQ, QH, d_v) shape) over the caller's (batch, seq, head)
    strides; None folds the gate path out (the module's CFG.EPILOGUE_GATE must agree).

    ``thd`` (appended; the kernel passes its ``CFG.THD_VARLEN``): under THD the gate is the SAME packed
    ``[1, T, QH, d_v]`` map Q / O take (``_bshd(..., thd=True)``: batch extent 1 binding the token stride,
    never stepped), so the TMA-LDG warp's ``issue_gate_load(..., q_row_base + q_seq_off, tma_batch=0)``
    addresses the gate rows of each sequence exactly as it addresses its Q rows.  The dense form is
    unchanged (the default keeps every caller's trace byte-identical)."""
    if gate_ptr is None:
        return None
    B, QH, _, SQ, _, _ = problem_size
    return _bshd(gate_ptr, B, SQ, QH, d_v, gate_strides, thd)


def sdpa_bias_tensor(bias_ptr, problem_size, bias_strides, bias_extents):
    """The additive attention bias ``[B_b, H_b, S_q, S_kv]`` (``B_b`` in {1, B}, ``H_b`` in {1, QH}: the graph's
    broadcast dims, bound as runtime extents so one artifact serves every broadcast form) over the caller's
    (batch, head, q-row) element strides; the kv column is contiguous.  None folds the bias path out (the
    module's CFG.BIAS_MODE must agree).  Dense only."""
    if bias_ptr is None:
        return None
    _, _, _, SQ, SKV, _ = problem_size
    b_b, h_b = bias_extents
    sb, sh, sq = (cutlass.Int64(s) for s in bias_strides)
    return cute.make_tensor(bias_ptr, cute.make_layout((b_b, h_b, SQ, SKV), stride=(sb, sh, sq, 1)))


def sdpa_kv_mask_tensor(kv_mask_ptr, problem_size, kv_mask_batch_stride):
    """The per-key boolean mask ``[B_m, S_kv]`` (the graph's ``[1|B, 1, 1, S_kv]`` BOOLEAN tensor read as bytes; a
    batch stride of 0 broadcasts one row over the batch) in the caller's batch element stride; the key is
    contiguous.  None folds the kv_mask path out (CFG.KV_MASK_FORM must agree)."""
    if kv_mask_ptr is None:
        return None
    B, _, _, _, SKV, _ = problem_size
    return cute.make_tensor(kv_mask_ptr, cute.make_layout((B, SKV), stride=(cutlass.Int64(kv_mask_batch_stride), 1)))


def sdpa_score_tensor(score_ptr, problem_size, score_strides):
    """One score side output (``score_max`` or ``score_sum_exp``): a dense fp32 ``(B, QH, SQ)`` row
    statistic in the caller's declared (batch, head, row) element strides -- the Stats geometry
    without its unit axis.  None folds the store out of the kernel (the DSL specializes on None),
    so a plan that does not request the output traces byte-identically to before the slot existed.
    Dense only: the THD / split builds decline the outputs at ``compile()``."""
    if score_ptr is None:
        return None
    B, QH, _, SQ, _, _ = problem_size
    s0, s1, s2 = score_strides
    return cute.make_tensor(score_ptr, cute.make_layout((B, QH, SQ), stride=(s0, s1, s2)))
