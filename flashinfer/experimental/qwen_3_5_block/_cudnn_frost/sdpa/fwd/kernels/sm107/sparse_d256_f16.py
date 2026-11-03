# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Rubin (SM107) index-list SPARSE prefill SDPA forward -- d_qk = d_v = 256, FP16/BF16, swap-AB, gathered K/V.

HEADER FIRST: this module carries its pipeline tables (warp map, barrier table with the issuing-lane ledger, SMEM table with
swizzle + why, TMEM map, the one-helper invariant, the degenerate-input matrix) ahead of the body, so a reader checks the
body against them instead of reconstructing the pipeline from the code.  The body is built from the SM100 swap-AB decode
tile (``sm100/decode_d256_f16.py``: the MMA descriptors, the softmax orientation, the epilogue) plus the ``tma_gather4``
loader; the config twin is ``config_sm107.make_cfg_d256_sparse``.

WHAT IT COMPUTES.  For query row ``r`` of sequence ``b`` at position ``p`` with ``n_vis = min(p + 1, L_b)`` visible tokens
and the caller's block list ``ids(r)`` (``[T_q, BLOCK_TOPK]`` int32, valid prefix then -1, blocks of BLOCK_SIZE = 4 tokens):

    V(r) = { t < n_vis : floor(t / 4) in ids(r) }  UNION  { t : 4 floor(n_vis / 4) <= t < n_vis }     (the open tail block, always visible)
    O[r] = softmax_fp32(q_r . K[V(r)]^T * scale) . V[V(r)];   LSE[r] = max + log(sum)  (natural log, the exact fp32 sum)

A listed block strictly past the open one (4 blk >= n_vis) contributes nothing (masked key by key); an entry that lists the
OPEN block itself (blk == floor(n_vis / 4) with n_vis % 4 != 0) or repeats another entry is attended TWICE -- a duplicate of
the appended tail / of the other entry (the softmax over the multiset; the list contract forbids both -- the library's host
check is FORM-only and never reads a list, the test tree's ``block_ids_contract_violations`` detector is what rejects them);
-1 entries at index < count are "no key" (zero rows, -inf) and still cost fabric bytes; entries at index
>= count are never read, so an extra entry placed at index count is never read either.  count = clamp(block_lens[r], 0,
n_sel_default) when block_lens is given, else n_sel_default = min(BLOCK_TOPK, floor((p + 1) / 4)) -- a device min / max,
never a host read.  An empty V(r) (L_b = 0, p < 0, a padded Q row, an empty selection with no tail) lands O = 0 / LSE = -inf
through a SELECT on the emptiness predicate, never ``residue * inv_sum``; padded Q rows store nothing.  The position p is the
row index (top-left) or, under CFG.BOTTOM_RIGHT, row + (L_b - S_q_b) (the dense tiles' bottom-right diagonal: the decode and
verify rows sit at the END of the sequence); the DECODE FORM below shares one list per sequence anchored at the step-0 position.

WORK ITEM = (query token t, KV head h): the G query heads of t (12 at 24/2, G <= 16) form the MMA N axis of the swap-AB tile,

    BMM1  S^T[128 keys, 16] = K_tile[128 x 256] (A, K-major SW128) . Q^T[16 x 256] (B, K-major SW128)   16 k-steps of 16
    BMM2  O^T[128 d, 16]   += V^T[128 d x 128 keys] (A, MN-major) . P^T[128 keys x 16] (B, MN-major)    2 d-blocks x 8 k-steps

over n_tiles 128-key tiles, each 32 four-token blocks GATHERED by ``tma_gather4`` (one issue = 4 rows x one 128-B column box;
4 boxes per 256-wide row per operand; 128 issues = 64 KiB per operand tile).  One CTA per SM, cga1, 20 warps (16 at one softmax
column group), persistent
through the shared scheduler (BSHD: the grid is the work list, 2 x T_q items decoded arithmetically; THD: the device-counter
form on an occupancy-sized grid).  Q^T and the block list of item i+1 are prefetched under item i's KV loop; the epilogue gate
of item i (``O *= sigmoid(G)``, compiled out unless CFG.EPILOGUE_GATE) is TMA-staged into the freed sQ slot of item i.

WARP MAP (the DEFAULT = TWO softmax column groups: 20 warps = 640 threads; ONE group (TemplateParams.qsa_softmax_groups = 1) = 16 warps
= 512 threads, its ids in parentheses; every warpgroup is role-homogeneous because ``setmaxnreg`` is warpgroup-collective; the gather /
MMA / TMA / scheduler / spare ids FOLLOW SOFTMAX_WARPS -- CFG.GATHER_WARP_BASE / MMA_WARP_ID / TMALDG_WARP_ID / SCHED_WARP_ID /
SPARE_WARP_ID -- and the body dispatches on those names, never on a literal; table re-based on the two-group default 2026-10-09)

    warps 0-7 (0-3)    WG0-1 (WG0)   softmax + epilogue: 128 lanes per column group = the 128 key lanes of S^T / P^T = the 128 d lanes of O^T;
                                     column group g = warps 4g..4g+3 owns columns 8g..8g+7 (ONE group: warps 0-3 own all 16 columns) -- the
                                     SOFTMAX SPLIT below
    warps 8-15 (4-11)  WG2-3 (1-2)   GATHER issuers w = warp - GATHER_WARP_BASE: blocks 4w..4w+3 of every K and V tile; the quad of ids by ONE
                                     warp-uniform 16-B LDS from sIds BEFORE the ring wait (the quad is 16-B aligned: BLOCK_TOPK % 4 == 0), the
                                     SELECT per index; PAGED arm: ONE block-table lookup per block of the quad right there (page = block_table[b,
                                     4 blk // PAGE_SIZE], an in-bounds read, -1 for a dead / past-the-length block, a page index past the table
                                     or a -1 page); 4 blocks x 4 boxes = 16 gather4 per operand tile from the elected lane; ONE
                                     arrive_expect_tx(KV_TX_BYTES_PER_WARP) per operand stage; kv_state carried across items; ring drain at exit
    warp 16 (12)       WG4 (3)       MMA issue (CFG.MMA_WARP_ID; + tmem_alloc / dealloc once per CTA): 4 commits per tile (s_full, kv_empty[K],
                                     bmm2_done, kv_empty[V]) + 1 per item (q_empty)
    warp 17 (13)       WG4 (3)       TMA-LDG (CFG.TMALDG_WARP_ID): Q^T(i+1) (4 subtiles, Q_TX_BYTES = the BOX bytes), the BLOCK_TOPK x 4 B ids bulk
                                     copy of item i+1, the gate of item i into its freed sQ slot (gated arm); NO K/V bytes; exit drains PER ARM:
                                     mb_ids_empty x STAGES_IDS always; ungated mb_q_empty x STAGES_Q; gated: the exit-path mb_gate_empty wait
                                     BEFORE the last item's gate (the wait Q(n_items) would have taken), then mb_gate_empty x STAGES_GATE
                                     (every mb_q_empty commit is consumed by its item's gate load -- nothing of row 2 is left to drain)
    warp 18 (14)       WG4 (3)       scheduler (CFG.SCHED_WARP_ID; scheduler_warp_loop for BSHD, the persistent claim-counter form for THD) --
                                     never credits
    warp 19 (15)       WG4 (3)       spare (CFG.SPARE_WARP_ID): passes the init sync, then exits (no CTA-wide barrier-0 sync may follow that one
                                     in the body)

SOFTMAX SPLIT (CFG.SOFTMAX_GROUPS = 2, the DEFAULT since its A/B/A; TemplateParams.qsa_softmax_groups = 1 renders the one-group body; a
PERFORMANCE knob -- the same function bitwise): the softmax role becomes TWO 4-warp column GROUPS, warps 0-3 (WG0) columns 0-7 and warps 4-7 (WG1) columns 8-15 of S^T / P^T /
O^T over the SAME 128 lanes (grp = warp // 4, col0 = 8 grp, lane = tidx & 127; a warp's TMEM lanes are its quadrant's, warp % 4, so warp 4
shares lanes 0-31 with warp 0 and the column offset is a plain address add); the gather warps move to 8-15, MMA / TMA-LDG / scheduler /
spare to 16 / 17 / 18 / 19.  Per tile each group loads ITS 8 S^T columns (tcgen05_ld num=8), reduces 8 column maxes through ITS slice of
``red`` ([3][8 warps][8 cols] = the same 768 B) behind ITS named barrier (2 + grp, 128 threads), packs ITS 16-B P^T chunk per lane row
(word offset 4 grp = exactly the one-group loop's chunk c = grp under the same Swizzle(1, 4, 3)), rescales ITS 8 columns of both O^T
d-blocks, and in the epilogue sums / stores ITS columns (col0 + j < G).  Every softmax-fired barrier init doubles (SOFTMAX_LANES 256,
IDS_EMPTY_ARRIVERS 16, READ_TILE_ARRIVERS 18, named barrier 1 at 288); the per-lane mask work is duplicated in both groups.  At ONE
group every split term folds away (``cutlass.const_expr(SOFTMAX_GROUPS > 1)``): the default rendering's cubin is byte-identical to the
pre-split one (28801073 ungated / ceeba558 gated, 2026-10-09).  Registers at 20 warps: 96 FLAT (65536 / 640 rounded down to 8); host
sm_107a trace-compiles of 21 two-group renderings (gated / paged / THD / split-KV / bottom-right / f16 / G = 1, 5, 12, 16 / top-4): STL 0 /
LDL 0 on 20, G = 16 gated 1 / 1 in the prologue only (vs 6 / 6 + 3 per TILE at one group, REG 128); the declared split
8 x 136 + 8 x 72 + 4 x 64 = 1920 == 96 x 20; the fallback ``gather_warps=4``: 16 warps, 128 flat, 8 x 160 + 4 x 96 + 4 x 96 = 2048.

Every warp except the scheduler credits the scheduler slot (read_tile_id_arrive) at the TOP of its item, right after reading
the payload -> READ_TILE_ARRIVERS = SOFTMAX_WARPS + 1 + 1 + GATHER_WARPS = 18 (14 at one column group).  Registers at ONE column group
(the two-group numbers are in the SOFTMAX SPLIT paragraph above): a 512-thread launch gives ptxas 128 per thread
FLAT (``.reqntid 512`` and no ``.maxnreg`` from the DSL; every setmaxnreg is dropped).  Host trace-compile of the decode body
under a forced 512-thread rendering (sm_107a): REG 104, STL 0, LDL 0 on the dense-padded and the causal specializations.  The
declared split for when ``.maxnreg`` lands: 4 x 240 + 8 x 96 + 4 x 80 = 2048 == 128 x 16 (SUM(role regs x warps) == entry x
warps; every count % 8 == 0).  The named fallback ``make_cfg_d256_sparse(gather_warps=4)``: 12 warps, 168 flat,
4 x 240 + 4 x 96 + 4 x 168 = 2016 == 168 x 12.  Every other population is declined by the config: a gather count that does
not divide the 32 blocks of a tile (5 -> 13 warps, 6 -> 14 warps) by the divisibility predicate, one that divides them but
does not fill a warpgroup (2 -> 10 warps) by the warpgroup predicate; the factory itself admits gather_warps in {4, 8} only.

MEASURED LEVERS (2026-10-08; the Rubin perf class, 212 SMs at the 2376 MHz lock, sampled 2364 MHz; A/B/A x 3 rounds x 100 launches
against this kernel with a control pair in the same process (<= 0.04 %: this kernel's twin pairs <= 0.024 %, the variants' own
<= 0.040 %; a NEUTRAL verdict = |speed-up| <= max(2 x the control spread, 0.3 %)); B = 1, 24 / 2, bf16, top_k 512, S = 8K / 32K,
the real layer-3 lists and random lists; every variant bitwise-equal to this kernel's O / LSE; speed-up = shipped / variant - 1):
the hint-less ring-wait spin (SPIN_RING_WAITS True) -0.51..-0.54 % on every cell -> False stands; 4 gather warps (12 warps, 168
registers flat) -35.3 % @8K / -36.4 % @32K (~5200 clk per K + V pair vs ~3370: the per-warp issue loop of 32 serialized gather4
binds) -> 8 stands, and 12 gather warps is not a population (32 % 12 != 0; 20 warps = 96 registers flat < the 128 this body needs
at 0 spills); the L2 EVICT_NORMAL policy on the K / V gathers -0.00..-0.02 % = NEUTRAL (inside the control pairs) -> EVICT_LAST
stands; the exact live-bytes arm (expect_tx = n_live x 2 KiB per warp, the dead quads STS-zeroed by the warp's lanes instead of TMA-OOB
zero-fill) -9.5 % @8K / -10.6 % @32K -> the full-box arm with TMA zero-fill stands (that arm fences generic -> async and
warp-syncs at EVERY fill before the arrive, plus the zero stores per dead block: more gather-warp latency than the <= 4.3 % of
fabric bytes it saves are worth on a body-bound core; a fence only on fills that zeroed a quad was not measured -- the
lever's ceiling is still those bytes); a SEPARATE gate slot
(229.4 KiB, the oversized carveout) is unmeasured -- it needs the gated rendering.

BARRIER TABLE (per CTA; init = the EXACT per-phase arrival sum; "lanes" resolved from the guard form at the call site --
nothing in tile_dsl.barrier elects for you; cga1: every commit is cta_group::1, no cluster fence, no cross-CTA arrive.  The primary
values are the DEFAULT rendering's (TWO softmax column groups, 20 warps); the ONE-group values follow in parentheses; producers are
named by ROLE -- the MMA warp = CFG.MMA_WARP_ID, the TMA warp = CFG.TMALDG_WARP_ID (16 / 17 at the default, 12 / 13 at one group).
Table re-based on the two-group default 2026-10-09; before that it led with the one-group rendering.)

    #  name [stages]        producer     issuing lanes (guard form)                                   SUM == init  consumer / wait                   start  carried state
    1  mb_q_full[2]         TMA_LOAD     the TMA warp, arrive(n_bytes=Q_TX_BYTES, pred=elect_sync()); tma_load_tile x 4 subtiles
                                         Q_TX_BYTES = Q_BOX_ROWS x TILE_K x BPE = G x 512 B (6144 at G=12; the BOX, never the 8 KiB slot)
                                                                                                        1 == 1       MMA before the item's first BMM1     0      q_state(2), once per item
    2  mb_q_empty[2]        MMA_COMMIT   the MMA warp, arrive(cta_group=1, pred=elect_p) once per item after the item's LAST BMM1
                                                                                                        1 == 1       the TMA warp before gate(i) / Q(i+2) 1 | 0  per item.  UNGATED: the pre-armed qe_state (start 1) waits
                                                                                                                                                                 before Q(j), the last STAGES_Q commits drained at exit.  GATED: the
                                                                                                                                                                 wait-then-arrive qg_state (start 0) consumes commit i before gate(i);
                                                                                                                                                                 n commits, n waits -- no drain (the slot's WAR gate for Q(i+2) is row 12)
    3  mb_ids_full[2]       TMA_LOAD     the TMA warp, arrive(n_bytes=IDS_TX_BYTES, pred=elect_sync()); bulk_copy(sIds[slot], block_ids[item_row], IDS_TX_BYTES)
                                         IDS_TX_BYTES = BLOCK_TOPK x 4 (a 16-B multiple); the copy runs for EVERY item, dead ones included
                                         (item_row = the token's flat row, or the SEQUENCE under CFG.LIST_PER_SEQUENCE: one row copied by every item of it)
                                                                                                        1 == 1       8 gather + 8 softmax warps (4 at one group), item top
                                                                                                                                                          0      ids_state(2), once per item
    4  mb_ids_empty[2]      THREAD       `if elect_sync(): arrive()` from each of the 16 consuming warps after its last read of the slot: 8 gather +
                                         8 softmax (both column groups read their lane's block id; IDS_EMPTY_ARRIVERS = GATHER_WARPS + SOFTMAX_WARPS;
                                         12 at one group)
                                                                                                        16 == 16 (12 == 12)   the TMA warp before refilling (item i+2)
                                                                                                                                                          1      per item; drained at exit by the TMA warp
    5  mb_kv_full[3]        TMA_LOAD     each gather warp: arrive(n_bytes=KV_TX_BYTES_PER_WARP, pred=elect_sync()) once per stage fill,
                                         then its 16 gather4 x 512 B = 8192 B (OOB / -1 rows zero-filled AND credited); 8 x 8192 = the 64 KiB stage
                                                                                                        8 == 8       MMA before BMM1 (K) / BMM2 (V)       0      kv_state(3) per LOAD, order K(0) K(1) V(0) K(2) V(1) ... V(n-1),
                                                                                                                                                                 CARRIED across items in the gather warps AND the MMA warp
    6  mb_kv_empty[3]       MMA_COMMIT   the MMA warp, pred=elect_p after the last MMA reading the stage (K: with s_full's commit; V: with bmm2_done's)
                                                                                                        1 == 1       all 8 gather warps before refilling 1      the same kv_state; each gather warp drains STAGES_KV at exit
    7  mb_s_full[2]         MMA_COMMIT   the MMA warp, pred=elect_p right after BMM1(t)               1 == 1       8 softmax warps (4), top of tile t   0      s_state(2) per TILE, never reset at an item boundary
    8  mb_s_empty[2]        THREAD       bare arrive() from all 256 softmax lanes (SOFTMAX_LANES), each column group after ITS tcgen05_ld(num=8) ->
                                         tcgen05_wait(LOAD) -> tcgen05_fence (128 at one group, after its num=16 load)
                                                                                                        256 == 256 (128 == 128)   MMA before BMM1(t+2) reuses the slot
                                                                                                                                                          1      s_empty_state(2) per tile, carried
    9  mb_p_full[2]         THREAD       bare arrive() from all 256 lanes, each column group after ITS 16-B P^T chunk store_swizzled + fence_proxy + ITS
                                         O^T rescale + tcgen05_fence: both groups' chunks and both rescales have landed when the MMA warp's wait
                                         returns (128 at one group)
                                                                                                        256 == 256 (128 == 128)   MMA before BMM2(t)
                                                                                                                                                          0      p_state(2) per tile, carried
   10  mb_bmm2_done[2]      MMA_COMMIT   the MMA warp, pred=elect_p after BMM2(t)'s two d-blocks     1 == 1       softmax before the O^T rescale (tile t+1) and once in the epilogue (last tile)
                                                                                                                                                          0      bmm2_state(2) per tile; the epilogue's wait advances it.  Also the P^T slot's
                                                                                                                                                                 WAR gate: P^T(t) -> sP[t % 2] is stored AFTER the wait on bmm2_done(t-2) taken at iteration t-1
   11  mb_gate_full[1]      TMA_LOAD     the TMA warp, arrive(n_bytes=GATE_TX_BYTES, pred=elect_sync()); the gate box (1, 1 token, G heads, 64) x 4 subtiles
                                         into the freed sQ[i % 2]; GATE_TX_BYTES = Q_BOX_ROWS x TILE_O x GATE_BPE = G x 512 B (the BOX)   [gated arm]
                                                                                                        1 == 1       all 256 softmax lanes (128), epilogue(i)
                                                                                                                                                          0      gf_state(1): wait-then-arrive, advance(.., STAGES_GATE) once per item
   12  mb_gate_empty[1]     THREAD       bare arrive() from all 256 softmax lanes after the item's last gate LDS (SOFTMAX_LANES; 128 at one group)   [gated arm]
                                                                                                        256 == 256 (128 == 128)   the TMA warp before Q(i+2) into the slot
                                                                                                                                                          1      ge_state(1): pre-armed, advance(.., STAGES_GATE) once per item: Q(1)'s wait is
                                                                                                                                                                 the pass (Q(0) skips it), Q(j) waits epilogue(j - 2); the exit path waits
                                                                                                                                                                 epilogue(n - 2) BEFORE gate(n - 1) is issued (the wait Q(n) would have taken;
                                                                                                                                                                 the pass itself at n = 1), then drains x STAGES_GATE = epilogue(n - 1).  Never
                                                                                                                                                                 2 behind: completion i needs gate(i), issued only after the wait that consumed
                                                                                                                                                                 i - 1 (Q(i + 1)'s in the loop, the exit-path one for the last item)
   13  sched.mb_scheduler[2]  TMA_LOAD (CLC response form) | THREAD (persistent form)
                                         the scheduler warp's elected arrive_expect_tx(16) + the CLC response (CLC) / its elected arrive after the
                                         4 payload stores (persistent; no expect_tx at cga1)
                                                                                                        1 == 1       every consuming warp reads the payload      tile_dsl.scheduler
   14  sched.mb_read_tile_id[2]  THREAD  read_tile_id_arrive(mb, cga_size=1) = one elected lane per calling warp; callers = 8 softmax + 1 MMA + 1 TMA +
                                         8 gather = 18 (4 + 1 + 1 + 8 = 14 at one group; READ_TILE_ARRIVERS, config-derived)
                                                                                                        18 == 18 (14 == 14)   the scheduler before refilling the slot     tile_dsl.scheduler
   15  mb_tmem_dealloc[1]   THREAD       bare arrive() from all 256 softmax lanes at exit (128 at one group)
                                                                                                        256 == 256 (128 == 128)   MMA before tmem_dealloc    0      once
   16  named barrier 1      bar          MMA warp barrier_cta_arrive (32) + EVERY softmax warp barrier_cta_sync, thread_count 32 x (SOFTMAX_WARPS + 1) = 288
                                         (160 at one group)   the TMEM-base hand-off, once
   17  named barrier 2 (+g) bar          column group g's 4 warps ONLY, id 2 + g (a runtime Int32 id; the lone group's id is the constant 2), thread_count
                                         128: the column-max exchange per tile + the per-lane sum per item; the groups never wait on each other
   18  barrier 0            bar          all THREADS_PER_CTA lanes once (640; 512 at one group), after fence_mbarrier_init (the spare warp exits only after it)

Init: ``if warp_idx == 0: if nvvm.elect_sync():`` every stage of every ring (``.init()`` is per stage) + the two scheduler rings
(ONE_LANE / READ_TILE_ARRIVERS); fence_mbarrier_init + barrier_cta_sync OUTSIDE the branch.  Phase discipline: every ring
slot and parity above comes from a PipelineState CARRIED across work items in every role that touches it and advanced at
every producer fire / consumer wait -- NEVER recomputed from a per-item tile index, and never a manual ``^ 1`` flip (rows 11 / 12
included since 2026-10-09: the single-stage gate barriers' consumer parities are ``ge_state`` / ``gf_state``, advanced through
``advance(state, CFG.STAGES_GATE)`` -- a second flip site is how P14 drifts).  The decode tile's arithmetic parities
(``_kv_slot``, ``(t // 2) & 1``, ``((t // 2) - 1) & 1``) are correct only because that kernel is not persistent (fresh
barriers per unit); in a persistent CTA they HANG at n_tiles = 1 with two items (s_empty), read a stale S^T slot at
n_tiles = 3 (s_full) and read a stale K/V slot whenever 2 x n_tiles % 3 != 0.  Every item runs >= 1 tile, so the per-item
rings (1-4, 11-14) advance exactly once per item and the per-tile rings (5-10) at least once; the epilogue runs for every item.
Exit drains (every arrive has a reachable wait, and the waiter is never TWO completions behind a barrier: wait() returns when the
parity differs from the waited one, so parity(k) == parity(k + 2) parks it forever -- a hang with correct numerics): the TMA warp waits
mb_ids_empty x STAGES_IDS and, ungated, mb_q_empty x STAGES_Q (pre-armed state); gated, every mb_q_empty commit is consumed by its
item's gate load (nothing left) and mb_gate_empty is consumed in ORDER as well as in count -- n_items completions against n_items
waits: n_items - 1 before Q(1 .. n_items - 1) (Q(1)'s the pre-armed pass, Q(j)'s = epilogue(j - 2)), the exit-path wait taken
BEFORE the last item's gate is issued (= epilogue(n_items - 2); the pass itself at n_items = 1, taken while nothing can have
arrived), the exit drain x STAGES_GATE (= epilogue(n_items - 1)); completion i needs gate(i), and gate(i) is issued only after the
wait that consumed completion i - 1, so the consumer is never two behind (a drain of STAGES_GATE + 1 waits with no exit-path wait
was right in count and wrong in order: epilogue(n_items - 1) could land before the first drain wait and alias its parity); every
gather warp waits mb_kv_empty x STAGES_KV -- each with its carried state.  Gated arm, the slot's order: Q(i) -> MMA reads ->
mb_q_empty(i) -> gate(i) -> mb_gate_full -> epilogue(i) reads -> mb_gate_empty -> Q(i + 2); the TMA warp issues gate(i) AFTER
queuing Q(i + 1) / ids(i + 1) (their waits are the EARLY ones: mb_gate_empty(i - 1) / mb_ids_empty; the gate's is the item's
LAST BMM1) and before its next credit; the softmax warps find the item's gate at slot i % 2 through a carried qslot_state.

PDL SITES (``use_pdl``, a compile-time kernel argument and the launch attribute, appended 2026-10-21; NO barrier row, ring depth,
SMEM buffer or TMEM column changes -- the table above is the ``use_pdl=False`` rendering's and the ``use_pdl=True`` rendering's alike,
and the two compute the same function bitwise).  Under the programmatic-stream-serialization attribute this grid may START while
its predecessor on the stream (the norm + RoPE / prepare launch that writes the slab, the write-through that fills the pools, the
selector that writes the block list, the stack's kernels that write the lengths and the table) still runs, so EVERY global read of a
predecessor's output sits below a ``griddepcontrol.wait`` (``wait_on_dependent_grids()``) of the reading role, and no global WRITE
sits above one.  Per role, every global read, and where its wait is:

    TMA warp      reads: block_ids (the ids bulk copy, barrier row 3), the Q^T box (the slab; row 1), the gate tile (the slab's GATE
                  columns; row 11, gated arm).  Wait: at the top of ``_tmaldg_warp_group``, before the prologue's ``_decode_item`` and
                  ``_tma_issue_item`` (the tensor-map prefetches above it read kernel PARAMETERS, not a predecessor's output).
    gather warps  reads: ``_item_bounds``' inputs -- seq_kv_lens (the stack's lengths) and block_lens (the selector's counts) --, the
                  block table (``_gather_block_pages``, paged arm) and the K / V rows (``tma_gather4`` of the pools or the dense
                  operands).  Wait: at the top of ``_gather_warp_group``, before the prologue's ``_decode_item`` and the first
                  ``_item_bounds``.  (Dense BSHD: ``_decode_item`` reads nothing global -- the grid position; the THD arm's metadata
                  read is below the wait too, but see the prologue note.)
    MMA warp      reads: ``_item_bounds``' inputs (seq_kv_lens, block_lens) only -- its operands arrive through SMEM.  Wait: after the
                  TMEM allocation, the named-barrier hand-off and the descriptor builds (its prologue), before ``_decode_item``.
    softmax warps reads: ``_item_bounds``' inputs (seq_kv_lens, block_lens); WRITES: O / the partial slabs and the LSE (the epilogue).
                  Wait: after the TMEM hand-off (``barrier_cta_sync`` on named barrier 1 + the SMEM read of the TMEM base), before
                  ``_decode_item`` -- every global write of the role is below it.
    scheduler     reads nothing global on the BSHD / CLC form (the CLC responses are hardware; the payload ring is SMEM): no wait.
    spare warp    exits after the init sync: no wait.
    the release   NONE explicit: no CTA executes ``griddepcontrol.launch_dependents``, so each CTA counts as released when it EXITS (the
                  PTX rule) and the dependent grid (the split combine, or the out projection at split 1) launches at this grid's
                  completion through the programmatic edge.  Its own wait blocks on this grid's completion and memory flush wherever a
                  trigger would sit, so the placement is a performance choice, never a correctness one -- and it is a MEASURED one
                  (2026-10-21, the cc 10.7 class: a 212-SM gate of the switched block chain and a per-kernel timeline of one replay on
                  a 204-SM part under two instruments).  The first form of this paragraph released right after ``fence_mbarrier_init +
                  barrier_cta_sync``, before the role dispatch: the combine's CTAs then sat resident on the idle SMs from this grid's
                  first microsecond, spinning in their wait through the whole mainloop, and after this grid completed they needed
                  7.4-7.9 us for 2.75 us of work (+4.6-5.1 us at that one link) while this kernel ran 0.4-0.7 us slower with them
                  resident -- the whole 3.2-4.4 us per-step loss the switched chain showed at B <= 4 (the graph's launch overhead did
                  not move with the number of programmatic nodes).  The SAME edge triggered at this grid's completion cost nothing
                  (2.85 vs 2.75 us), and the dependents the norm + RoPE, cache-write and combine kernels launch early (short, non-CLC
                  primaries) gain 0.2-0.8 us each.  Rule for a long CLC-scheduled primary: no release above its mainloop (trigger at
                  completion, or after the CTA's last item); a prologue release is a measured exception.  The TMEM-footprint argument
                  the prologue release needed (no dependent CTA could co-reside with this one -- 226,688 B of the 227 KiB carveout, one
                  CTA per SM -- so a dependent allocating TMEM before its own wait could never hold columns this CTA's later allocation
                  needs) is moot: nothing is released above the MMA warp's allocation, or at all.

The PRE-DISPATCH prologue holds ONE global read ABOVE every role wait: ``exit_if_dead_thd_cluster(seq_kv_lens_tensor, ...)`` under
``CFG.THD_VARLEN == 1`` reads the live-unit total ``meta[4B + 2]`` -- the SETUP LAUNCH's device value, written by the kernel launched
right before this one on the same stream -- before any mbarrier init, and the THD scheduler's claim counter reads ``meta[4B + 3]`` the
same way.  A dependent launch of the THD rendering would therefore read the metadata the setup launch has not written yet, so THD x
``use_pdl`` is a typed decline of the adapter (``SparseGqaFwdDslSm107.check_support``: ``thd=True`` with ``use_pdl=True``), never a wait
placed above the exit (which would serialize every CTA of the grid behind the predecessor anyway).  The early-exit CTAs of the THD form
count as released by completing, like every CTA of this kernel.  The PDL scope is the dense BSHD and paged decode chain.

THE ONE HELPER ``_item_bounds``: n_tiles, count, the tail block id(s), dead and the split chunk are computed by ONE pure
helper at the item-start site of EVERY role that loops over tiles (softmax, MMA, the 8 gather warps, the TMA warp) over the
same inputs (the item ``_decode_item`` decoded from the payload -- ONE decode for every role and every item, the CTA's
first item included -- the block_lens pointer, the seq lens); nothing about them is staged through SMEM.

    tok           = the Q row (BSHD: ctaid.x % S_q, the split ctaid.x // S_q)  |  tok_packed - cu_q[b]  (THD)
    delta         = eff_seqlen_kv_b - eff_seqlen_q_b  (CFG.BOTTOM_RIGHT)  |  0
    pos           = tok + delta                                  the row's KEY-SPACE position (the mask: key_abs <= pos)
    anchor        = delta = pos_0, the step-0 position  (CFG.LIST_PER_SEQUENCE)  |  pos
    n_vis         = clamp(min(pos + 1, eff_seqlen_kv_b), 0)     the VISIBLE range (a padded sequence shorter than pos keeps its own tail)
    n_anchor      = clamp(min(anchor + 1, eff_seqlen_kv_b), 0)
    n_sel_default = min(CFG.BLOCK_TOPK, floor(clamp(anchor + 1, 0) / 4))
    count         = clamp(block_lens[item_row], 0, n_sel_default)  when given, else n_sel_default
    tail_lo       = floor(n_anchor / 4)                           the first block NOT complete at the anchor
    n_tail        = max(0, ceil(n_vis / 4) - tail_lo)             the blocks covering [4 tail_lo, n_vis): 0 / 1, or 2 under the shared
                                                                  list at S_q <= 4 (every token from the step-0 tail start to the
                                                                  row's own position -- the block completed between pos_0 and pos_j
                                                                  stays visible); 0 under the pure-list test arm (INCLUDE_OPEN_BLOCK = 0)
    dead          = (pos < 0) | (tok >= eff_seqlen_q_b) | (eff_seqlen_kv_b == 0) | (count + n_tail == 0)
    n_tiles       = max(1, ceil((count + n_tail) / 32))          CLAMPED >= 1: a dead item runs ONE tile of -1 rows
    CFG.SPLIT_KV = S > 1: per = n_tiles // S, rem = n_tiles % S, tile0 = split x per + min(split, rem), n_sp = per + (split < rem);
                  the chunk [tile0, tile0 + n_sp) is THIS work item's; n_sp == 0 (S > the item's tiles) -> dead (one clamped tile)
    live          = tok < eff_seqlen_q_b                          the store predicate (a padded Q row stores nothing; computed at the store)

A dead item's 32 block ids read as -1 (the gather and softmax lanes SELECT -1 over the loaded word, read at an address
clamped into the slot), TMA zero-fills and credits the 128 KiB, every lane is masked, l_tot = 0 and the row_dead SELECT lands
O = 0 / LSE = -inf (no store for a padded Q row) -- every ring advances exactly as on a live item.  A gather ROW coordinate
is kv_row_base + 4 blk + r for a live row (kv_row_base = batch x S_kv on the dense map, cu_k[b] under THD) and -1 (TMA
zero-fill, bytes credited) for a -1 block, a block past the sequence's length and the straddling rows of its last block, so
no other sequence's row -- or its NaN -- reaches an MMA under P = 0.  Per key lane l of tile i: blk = sIds[32 (tile0 + i) + l // 4]
(or the tail id / -1 by the SELECT), key_abs = 4 blk + l % 4, valid = (blk >= 0) & (key_abs <= pos) & (key_abs < eff_seqlen_kv_b);
s[l, j] = valid ? s_raw * scale_log2 : -inf for the 16 columns -- the open block's future rows hold REAL keys of other
tokens, so this mask is mandatory (TMA-OOB zero-fill does not do its job).

THD ARM (CFG.THD_VARLEN; the config forces SEQ_KV_LENS_PRESENT): the operands are PACKED -- Q / O [1, T_q, H, D], K / V
[1, T_kv, H_kv, D], block_ids [T_q, BLOCK_TOPK] with the block ids RELATIVE TO THE SEQUENCE (block j of sequence b = its
tokens [4 j, 4 j + 4), i.e. packed rows cu_k[b] + 4 j .. + 3), LSE [1, H, T_q] -- and ``seq_kv_lens`` is the int32 METADATA
workspace of THD_META_WORDS(B) = 4 B + 4 words the host launches the shared setup kernel (``thd_helpers``) to fill from the
caller's lengths (per-sequence [B], or cumulative [B + 1] per the lens-form bits), BEFORE the main launch on the same stream:

    [0 .. B-1] = seq_kv_lens   [B .. 2B] = cu_q   [2B+1 .. 3B+1] = cu_k   [3B+2 .. 4B+1] = remap (unread here)
    [4B+2]     = live = T_live x H_kv, T_live = cu_q[B]   (one unit per packed Q token per KV head)
    [4B+3]     = ctr  = the persistent claim counter, seeded with the grid size

so eff_seqlen_kv_b = meta[b] is read by the SAME SEQ_KV_LENS_PRESENT path as a dense per-batch length.  The scheduler takes
the persistent form (``scheduler_warp_loop_persistent``: a unit id from the counter against ``live``, published as the payload
word (uid, 0, valid, 0) through local SMEM + one elected arrive -- barrier row 13 keeps init ONE_LANE); the grid is
occupancy-sized (one CTA per SM, at most one per unit), CTA c takes unit c first, and a CTA at or past ``live`` EXITS at entry
before any mbarrier init, TMEM allocation or credit.  ``_decode_item`` turns a unit id into the item in EVERY role: head =
uid // T_live, tok_packed = uid % T_live (the token runs fastest: concurrent units are adjacent tokens of one KV head, whose
lists overlap in L2), the sequence b by a binary search over cu_q (an empty sequence can never hold a token), pos = tok_packed
- cu_q[b], eff_seqlen_q_b = cu_q[b + 1] - cu_q[b], item_row = tok_packed (the block_ids / block_lens row and the Q / O / LSE
token coordinate, batch coordinate 0), kv_row_base = cu_k[b].  No per-sequence descriptor exists: the Q map and the gather maps
span the packed CAPACITIES (T_q / T_kv rows, each >= 1 by the adapter's contract -- a zero extent is an INVALID map) and every
coordinate stays inside the live totals by construction (tok_packed < T_live; a key past the sequence's length takes row -1).
Degenerate THD rows: a sequence with Q tokens and S_kv_b = 0 -> every one of its items is dead (one -1 tile; O = 0 / LSE = -inf
STORED); S_q_b = 0 -> no unit (nothing stored); S_q_b > S_kv_b -> n_vis = min(pos + 1, S_kv_b) (the tail follows the visible
range); the capacity tail past cu_q[B] is never written and the K / V rows past cu_k[B] are never read; B = 1 packed is the
same function as dense B = 1, S = T (bitwise: the same gathers in the same order, only the item-to-CTA assignment differs).
The missed-coordinate signature of a THD port (a dense coordinate on a load: LSE exactly log(S_kv_b), constant across a
sequence) is excluded by construction -- the three load sites (the ids bulk copy, the Q box, the gather rows) take item_row /
(out_tok, out_batch) / kv_row_base from ``_decode_item`` and nothing else.

SMEM TABLE (declaration order = address order; descriptor-read operands first so every MMA operand starts below 256 KiB,
lane-only buffers last so no alignment padding is spent; 1024-B alignment modelled by the config validator)

    name [stages]   bytes     WRITER, how                       READER, how                      lane stride  swizzle + WHY                                   start
    sQ[2]           16,384    TMA: Q(i+1) 4 subtiles of (G rows x 64), then gate(i) into the freed slot (same box orientation);
                              rows G..15 of BOTH slots zeroed by STS once per CTA + fence_proxy, never written again
                                                                MMA descriptor (B of BMM1, K-major, SBO 8 x 128 B); the gate: each column group's 128 lanes, lane = d,
                                                                G x LDS.16 at load_swizzled offsets
                                                                                                 n/a / 2 B    SW128 -- a descriptor reads it AND the per-lane
                                                                                                              reads follow the same SW128 offsets             0 (1024-aligned slots)
    sP[2]            8,192    256 softmax lanes (128 at one column group): group g store_swizzled(ITS 16-B chunk of a lane row's 32 B, word offset 4 g,
                              Swizzle(1, 4, 3)) -- the same bytes the one-group loop's chunk c = g writes
                                                                MMA descriptor (B of BMM2, MN-major, 32-B atom, SBO 8 x 32 B)
                                                                                                 32 B         32-B atom Swizzle(1, 4, 3) -- BOTH jobs: the
                                                                                                              descriptor decodes it (P_SWZ_BYTES = N_Q x BPE)
                                                                                                              and it spreads the 32-B lane stride            16,384
    sKV[3]         196,608    8 gather warps, tma_gather4 x 128 per stage: box b of quad q at b x 16 KiB + q x 512 B (512-B quads inside
                              1024-B sub-boxes = the tiled SW128 layout)
                                                                MMA descriptors over the same bytes: K-major A of BMM1 (SBO 1 KiB) and
                                                                MN-major A of BMM2 (V^T: LBO 16 KiB between d atoms, SBO 1 KiB)
                                                                                                 n/a          SW128 -- both descriptors read it; the gather
                                                                                                              map writes under the same SW128                24,576; sKV[2] at 155,648 = 152 KiB
    sIds[2]          4,224    the TMA warp, ONE cp.async.bulk of IDS_TX_BYTES from block_ids[item_row]; no STS (slot = IDS_TX_BYTES + 64 B reserve, unread)
                                                                gather warps: ONE uniform LDS.128 of blocks 32 i + 4 w .. + 3; softmax lanes: LDS.32 of
                                                                block 32 i + lane // 4 (4 adjacent lanes share a word); both clamped into the slot
                                                                                                 4 B          none -- no descriptor reads it and 32 lanes x 4 B
                                                                                                              is one bank cycle: neither justification applies  221,184 (16-B aligned bulk-copy destination)
    red                768    softmax warps: one warp-uniform fp32 per (slot, warp, column), 3 slots (tile parity 0 / 1, epilogue) x SOFTMAX_WARPS x
                              COLS_PER_GROUP = 4 x 16 or 8 x 8 (two column groups: group g reads warps 4 g .. 4 g + 3 of the slot) -- 192 words either way
                                                                the group's 4 warps after named barrier 2
                                                                                                 4 B          none (one address per warp per column)          225,408
    misc               512    tmem_ptr (16 B), 15 mbarrier arrays (29 stages, 16-B padded: 272 B), the payload ring (64 B) -- 352 B in a 512 B reserve
                              (+ the two 1-stage gate arrays, 32 B, on the EPILOGUE_GATE rendering: 384 B; the total is unchanged)        226,176
    total          226,688 B = 221.4 KiB  < 232,448 B (227 KiB, the standard carveout; L1 keeps its 100 KiB) by 5,760 B

Descriptor version: the LAST MMA-operand start is sKV[STAGES_KV - 1] = 155,648 B < 262,144 -> DESC_VERSION = 0, ONE module
constant passed to every SmemTile (a version-0 descriptor's 14-bit start_address wraps at 256 KiB; version 1 is not a
transparent widening, so it is derived, never "set to be safe").  A separate gate slot (+8 KiB) or ring depth 4 (+64 KiB)
needs the oversized carveout (ALLOW_OVERSIZED_SHARED_MEMORY, L1 -> 8 kB) -- measured levers, not defaults.  Proxy fences:
after the P^T generic stores (before mb_p_full), after the once-per-CTA sQ tail-row zeroing; the TMEM path is ordered by
tcgen05_fence + the commits; the sIds bulk copy -> LDS and gate TMA -> LDS edges are ordered by their mbarrier waits.

TMEM MAP (64 columns per CTA, allocated once, is_exclusive=False -- <= 512 columns; at two softmax column groups group g touches columns
[8 g, 8 g + 8) of the active S^T slot and of BOTH O^T d-blocks -- disjoint column sets, the same 128 lanes): S^T slot 0 at [0, 16), slot 1 at
[16, 32), O^T d-block 0 at [32, 48), d-block 1 at [48, 64).  The S^T slot count IS the depth of rows 7-10 and of sP: CFG.STAGES_S
(= len(S_ACC_OFF) = 2, pinned by the config), read once as STAGES_S and spelled at every ring site -- never a literal 2.  No TMEM slot is assumed zero: BMM2(0) of every item overwrites
(accumulate=False) and every dead column ends in the SELECT, never residue * 0.

GATHER TENSOR MAP: one 2-D map per operand, tokens OUTER, the token's FULL ROW inner, box (64 elems, 1 row) = one 128-B
SW128 span; the head and the d-box live in the COLUMN coordinate (h x D + box x 64; + o_k / o_v on the block's slab), every
batch / sequence / page term in the ROW coordinate (b x S + 4 blk + r; cu_seqlens_k[b] + 4 blk + r; the paged form below).
Stride contract: last dim contiguous, head stride a multiple of 64 elements, token stride a multiple of 8 elements, extents
< 2^32, page_size % 4 == 0.

PAGED ARM (CFG.PAGED_KV; a const_expr branch of the gather warps only -- the softmax, MMA and TMA warps and every barrier /
SMEM row are the dense arm's): K / V are page POOLS [num_pages, H_kv, PAGE_SIZE, D] addressed through a (B, max_pages) int32
block table; the per-batch KV length (seq_kv_lens, REQUIRED) is the visible range.  The 2-D map's rows are the pool's token
rows at the TOKEN stride (pitch = the page_size axis' stride): rows = num_pages x rows_per_page, inner = (H_kv - 1) x
col_head_stride + D.  The row of block blk's token r = page x rows_per_page + h x rows_per_head + (4 blk) % PAGE_SIZE + r
with page = block_table[b, (4 blk) // PAGE_SIZE] -- PAGE_SIZE % 4 == 0, so a block never straddles pages and ONE lookup per
block suffices; the lookup is hoisted into the gather warps' ids quad before the ring wait.  NHD pools (token stride
H_kv x D, head stride D): rows_per_page = PAGE_SIZE, rows_per_head = 0, the head in the column (h x D + box x 64); HND pools
(token stride D, head stride PAGE_SIZE x D): rows_per_page = H_kv x PAGE_SIZE, rows_per_head = PAGE_SIZE, the column box x
64 -- the adapter derives the three per-operand numbers from the strides (one compiled kernel serves both).  A -1 block, a
block whose first token is at or past the batch's KV length (its page index may name another sequence's stale page), a page
index past the table and a -1 table entry (the page -1 convention of the dense paged tiles) all take row -1: TMA zero-fill,
bytes credited -- so the rows of a partially filled last page beyond the KV length (another sequence's tokens, or garbage)
never reach an MMA, and the straddling rows of the open block are masked exactly as in the dense arm (key_abs < the length).

DECODE FORM (CFG.BOTTOM_RIGHT + CFG.LIST_PER_SEQUENCE + CFG.SPLIT_KV, each a const_expr arm; the dense rendering is unchanged):
the serving step's S_q <= 4 rows per sequence sit at the END of the sequence (BOTTOM_RIGHT: pos = tok + L_b - S_q), share ONE
block list (LIST_PER_SEQUENCE: block_ids [B, BLOCK_TOPK], block_lens [B]; item_row = b, so every item of the sequence -- every
KV head, split and token -- bulk-copies the same row) anchored at the step-0 position pos_0 = L_b - S_q (the count from pos_0,
the tail = the blocks from floor((pos_0 + 1) / 4) to the row's own block: 0-2 ids, derived, never in sIds), and at serving batch
(2 B items of <= 17 tiles each) the tiles of every item are cut into SPLIT_KV chunks, each its OWN work item: grid x = tok + S_q x
split, the softmax lanes store fp32 partials O_s = acc / l_s and LSE_s = m_s + log l_s (natural log) at the split-major
workspace batch b + s x B with the SAME st.global path as the half O (the tensor's element type is fp32), the dead SELECT per
element first, and ``sm100/split_combine.py`` reduces them (the gate, when requested, rides the combine: the kernel's own
EPILOGUE_GATE is refused under a split by the config).  An empty chunk is a DEAD item (one clamped tile of -1 rows -> O_s = 0 /
LSE_s = -inf, exactly the combine's identity), so every ring advances exactly as on a live item (P14); SPLIT_KV <=
MAX_TILES_PER_ITEM (a larger split has an empty chunk on every item).  No barrier init count, ring depth, SMEM buffer or TMEM
column changes: the per-sequence list is a SOURCE change of barrier row 3, the split a tile-range change of rows 5-10 plus the
epilogue's store target.  THD never composes with the three arms (the config refuses each pair).
MTP rows as single-token items (the shared list at S_q <= 4): every row is its OWN work item re-gathering the sequence's list --
per item IDS_TX_BYTES of ids (2 KiB at top_k 512; the copy runs for every item, dead ones included) plus (count + n_tail) x 4 rows x
2 operands x 512 B of K / V (514 blocks = 2056 KiB at a saturated list with two tail ids; 17 tiles x 128 KiB credited with the
zero-fill), so a step's bytes are S_q x the single-row step's: 2 KV heads x 4 rows x ~2.1 MiB = ~16 MiB per sequence per layer at
S_q = 4 (~4 MiB at S_q = 1), and the sequence's list row is copied S_q x H_kv x SPLIT_KV times.  The 64-wide single-item form
that would gather the shared list once is not built.

DEGENERATE-INPUT MATRIX (every row names its handling site): empty selection / pos < 0 -> dead -> one -1 tile -> the
row_dead SELECT; a query with 0 complete blocks (pos in {0, 1, 2}) -> count 0, has_open 1, the key_abs <= pos term;
(pos + 1) % 4 == 0 -> has_open 0; n_tiles in {1, 3, 4, 17} -> the carried PipelineStates; -1 inside the valid prefix -> "no
key" (blk >= 0), bytes still fetched; a block id past the sequence -> key_abs < eff_seqlen_kv_b, TMA zero-fills past the
tensor; a block violating block-causality -> key_abs <= pos (a future block contributes nothing; the open block listed inside
the count duplicates the tail); duplicates -> counted twice (contract); S_kv = 0 -> dead (one
tile, no host read); padded Q rows -> dead, no store; B x H > 1 with n_tiles = 1 -> carried states; block_lens out of range
-> clamped on device; B >= 2 / the slab stride -> the ROW coordinate carries b x S (cu_k[b] under THD), the COLUMN
coordinate the head and the column offset; one CTA / one item -> cga1 init counts; THD: S_kv_b = 0 with Q tokens -> dead items
stored, S_q_b = 0 -> no unit, the packed capacity tails untouched, a CTA past the live total exits at entry; a paged block whose
page is -1 / past the table -> row -1 (zero rows, the keys masked by the length); THD x paged -> a typed decline at config /
check_support (the sequence's row offset composes with a dense tensor only); page_size % 4 != 0, top_k outside [4, 512] or
% 4 != 0, G > 16 -> typed declines at config / check_support.  Decode form: kv_len_b = 0 -> pos_0 = -S_q -> every row dead (stored
0 / -inf; under a split every chunk dead -> the combine's all-dead identity); kv_len_b < S_q -> the rows with pos_j < 0 dead, the
rest live; (L_b - S_q + 1) % 4 == 0 -> the step-0 row has no tail and row j >= 1 one tail block (the block that completed between
pos_0 and pos_j); SPLIT_KV > the item's tiles -> the empty chunks are dead items; SPLIT_KV == the item's tiles -> one tile per item
(n_kv = 1 with B x H x S items: the carried states); S_q > 4 under the shared list, list_per_sequence without bottom_right, any of
the three arms under THD -> typed declines at config / check_support.  The denominator floor (1e-30) sits inside the reciprocal and the log only and
is always followed by the row_dead SELECT, so neither LSE nor O ever carries it.
"""

# Config comes from the FROST template loader (FROST_TEMPLATE_PARAMS is injected before this body runs); the plain-import
# default is the 24/2 geometry (12 query heads per KV head) at top_k = 512, bf16, so a plain import stays usable standalone.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import TemplateParams, make_cfg_d256_sparse

PARAMS: TemplateParams = globals().get(
    "FROST_TEMPLATE_PARAMS", TemplateParams(dtype_qkv=2, cta_mma=1, pack_gqa=True, qh_per_kh=12, qsa_block_topk=512, qsa_block_size=4)
)
CFG, _TMA = make_cfg_d256_sparse(PARAMS)

# tcgen05 SMEM-descriptor version for EVERY SmemTile in this module -- ONE decision point, DERIVED by the config from the
# layout (the start of the last K/V stage against the 14-bit version-0 window), wired into every construction below and
# never re-literalled at a tile (test_sm107_every_smem_tile_takes_the_module_desc_version's convention).
DESC_VERSION: int = CFG.DESC_VERSION
# Retry form of the per-KV-iteration RING waits: every such site is spelled ``.wait(..., spin=SPIN_RING_WAITS)``, never a
# literal; the sign is a MEASURED per-kernel fact.  This kernel's own A/B/A (2026-10-08, the Rubin perf class, 212 SMs at the
# 2376 MHz lock, sampled 2364 MHz, 3 rounds x 100 launches, control pairs <= 0.04 %; B = 1, 24 / 2, bf16, top_k 512) read
# the spin at -0.54 / -0.53 % @8K and -0.51 / -0.52 % @32K (real / random lists) -- a LOSS on every cell, so it stays False
# (the header's MEASURED LEVERS paragraph).
SPIN_RING_WAITS: bool = False
Cfg = type(CFG)

# CuTe DSL version gate (python/cudnn/AGENTS.md Rule 7): sm_107a needs the public 4.8.0 wheel; below it every compile on a
# Rubin box dies inside the DSL (``KeyError: 'sm_107a'``).  The adapter declines first (SparseGqaFwdDslSm107.check_support,
# BEFORE this module is loaded); this is the typed backstop for a direct template load, placed before the kernel body so the
# message names the installed version, never a DSL internal.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error

_DSL_ARCH_ERROR = cutedsl_arch_requirement_error((10, 7))
if _DSL_ARCH_ERROR is not None:
    raise NotImplementedError(f"sparse_d256_f16 (sm107): {_DSL_ARCH_ERROR}")

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached as _compile_cached, template_key as _template_key
from functools import lru_cache
from typing import Callable, NamedTuple, Optional, Tuple

from cutlass.experimental import primitives as nvvm
from cutlass.experimental.cuda import tensor_map as tmap
from cutlass._mlir.dialects import arith
from cutlass.base_dsl.typing import Pointer

import cutlass
from cutlass.experimental import primitives as prims
import cutlass.cute as cute
import cuda.bindings.driver as _cuda_driver  # noqa: F401

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import MBarrier, PipelineState, Producer, advance, wait, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mma import mma_ss
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import TMA_L2_EVICT_LAST, bulk_copy, opaque_i64, tma_gather4, tma_load_tile
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.handles import GmemTileTma, MmaDesc, SmemTile
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tmem import tmem_alloc, tmem_dealloc
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.scheduler import Sched, read_clc_payload, read_tile_id_arrive, scheduler_warp_loop, scheduler_warp_loop_persistent
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.thd import THD_CTR_OFF, THD_CU_Q_TOTAL_OFF, THD_LIVE_OFF, THD_META_WORDS, THD_SETUP_THREADS, exit_if_dead_thd_cluster
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import fp32_to_fp16

# The visible-set helper shared with the lean mma.sync decode tile (ONE helper for both bodies; the SELECT spellings travel with it).
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.sm107._sparse_item_common import _select_f32, _select_i32, block_id_at, visible_bounds
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels._common_blackwell import _bshd, _vec, gate_epilogue_pairs, gate_half_opaque, gate_inv_sum, row_max_for_exp2

# The THD metadata setup launch (per-sequence lengths -> [seq_kv_lens | cu_q | cu_k | remap | live | ctr]) is the shared one of
# the dense forwards: a second copy of the metadata LAYOUT would be a silent-wrong-answer waiting to happen (tile_dsl/thd.py).
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.thd_helpers import build_thd_meta_kernel as _build_thd_meta_kernel

if CFG.DTYPE_QKV == 2:
    STORAGE_DTYPE = cutlass.BFloat16
else:
    STORAGE_DTYPE = cutlass.Float16
MMA_KIND = nvvm.Tcgen05MMAKind.F16
CTA_GROUP_KIND = nvvm.CTAGroup.CTA_1

# === geometry, every value from CFG (frost-kernels.md section 6: derive, never re-literal) ===
N_Q = CFG.N_Q
TILE_N = CFG.TILE_N
TILE_K = CFG.TILE_K
TILE_O = CFG.TILE_O
BPE = CFG.BPE
STAGES_KV = CFG.STAGES_KV
STAGES_Q = CFG.STAGES_Q
STAGES_IDS = CFG.STAGES_IDS
# The S^T / P^T ring depth (barrier rows 7-10 and the sP buffers): the S^T TMEM slots of the map, read ONCE here and spelled at every
# ring site below -- the four MBarriers, every advance() of s_state / s_empty_state / p_state / bmm2_state / t_state, the sP Array and
# SmemTile, the init loop, the red scratch's parity slots.  Never a literal: a depth changed at some sites and not others is a stale S^T
# slot read from the first reuse (frost-kernels.md section 6).  The config pins STAGES_S == len(S_ACC_OFF) == 2; the check below is the
# trace-time twin for a direct template load.
STAGES_S = CFG.STAGES_S
SCHEDULER_STAGES = CFG.SCHEDULER_STAGES
BLOCK_SIZE = CFG.BLOCK_SIZE
BLOCKS_PER_TILE = CFG.BLOCKS_PER_TILE
BLOCKS_PER_WARP = CFG.BLOCKS_PER_WARP
BLOCK_TOPK = CFG.BLOCK_TOPK
GATHER_BOXES = CFG.GATHER_BOXES
GATHER_BOX_ELEMS = CFG.GATHER_BOX_ELEMS
# The paged arm: PAGE_SIZE tokens per pool page (0 on the dense arm, where no paged code is traced); PAGE_SIZE % BLOCK_SIZE == 0
# (the config's predicate) is what makes ONE block-table lookup per block sufficient.
PAGED_KV = CFG.PAGED_KV
PAGE_SIZE = CFG.PAGE_SIZE
# The decode form's three arms (const_expr folds; the dense rendering traces exactly the prefill program).
SPLIT_KV = CFG.SPLIT_KV
LIST_PER_SEQUENCE = CFG.LIST_PER_SEQUENCE
BOTTOM_RIGHT = CFG.BOTTOM_RIGHT

# The work item's Q^T box: ONE token x its G query heads (Q_BOX_ROWS = G rows of the N_Q = 16 tile; rows G..15 are the
# once-zeroed tail whose columns are computed and never stored).
G = CFG.QH_PER_KH
Q_BOX_TOKENS = CFG.Q_BOX_TOKENS
Q_BOX_ROWS = CFG.Q_BOX_ROWS

# TMA granule for Q (the K-major B operand of BMM1): one 128 B swizzle atom of the head dim = 64 half elements.  The gather
# box of K/V is the same 128-B span, so one SW128 layout serves the TMA-loaded Q and the gathered K/V tiles alike.
GRANU_ELEMS = _TMA.QK_GRANU_ELEMS
QK_ITERS = _TMA.QK_ITERS
if _TMA.VO_GRANU_ELEMS != GRANU_ELEMS or GATHER_BOX_ELEMS != GRANU_ELEMS:
    raise ValueError("sparse_d256_f16: Q, K and V must share the 128 B swizzle granule (the gather box is one such span)")

# The gate's per-lane SMEM read (EPILOGUE_GATE): lane = d, one 2-B element per head row out of the SW128 Q^T slot, so the 16-B
# chunk index (bits 4-6 of the byte offset = the column's chunk inside the 128-B row) is XORed with the row & 7 -- the pattern
# the TMA wrote and the Q^T descriptor decodes.  Derived from the swizzle span and BPE, never literals (frost-kernels.md section 6).
_GATE_ROW_SHIFT = GRANU_ELEMS.bit_length() - 1  # 6: d // 64 = the 64-wide subtile
_GATE_CHUNK_ELEMS = 16 // BPE  # 8 elements per 16-B chunk
_GATE_CHUNK_SHIFT = _GATE_CHUNK_ELEMS.bit_length() - 1  # 3
_GATE_SWZ_ROW_MASK = CFG.Q_SWZ_BYTES // 16 - 1  # 7: SW128 = 8 chunks per row, the row index folded modulo 8
if (1 << _GATE_ROW_SHIFT) != GRANU_ELEMS or (1 << _GATE_CHUNK_SHIFT) != _GATE_CHUNK_ELEMS:
    raise ValueError("sparse_d256_f16: the gate's SMEM read needs a power-of-two swizzle granule and chunk")

qBufferElems = N_Q * TILE_K
kvBufferElems = TILE_N * TILE_K
pBufferWords = TILE_N * N_Q // 2  # P^T as packed half pairs: N_Q / 2 words per key row
SUB_BOX_ELEMS = TILE_N * GATHER_BOX_ELEMS  # one 128-B column box of a K/V tile = TILE_N rows x 128 B = 16 KiB
IDS_SLOT_WORDS = CFG.IDS_SLOT_BYTES // 4

# Transaction bytes -- every one the BOX / the copy length, never a buffer size (barrier rows 1, 3, 5).
Q_TX_BYTES = CFG.Q_TX_BYTES
IDS_TX_BYTES = CFG.IDS_TX_BYTES
KV_TX_BYTES_PER_WARP = CFG.KV_TX_BYTES_PER_WARP
GATE_TX_BYTES = CFG.GATE_TX_BYTES  # the gate BOX (= Q_BOX_ROWS x TILE_O x GATE_BPE), the EPILOGUE_GATE arm's row 11

_SWZ_ENUM = {128: 2, 64: 4, 32: 6}
_SWZ_BITS = {128: 3, 64: 2, 32: 1}
SMEM_LAYOUT_QK = _SWZ_ENUM[CFG.Q_SWZ_BYTES]
SMEM_LAYOUT_P = _SWZ_ENUM[CFG.P_SWZ_BYTES]
# The generic-proxy P^T stores apply the same XOR pattern the UMMA descriptor decodes (Swizzle<B, 4, 3>).
_P_SMEM_SWIZZLE = cutlass.Swizzle(_SWZ_BITS[CFG.P_SWZ_BYTES], 4, 3)
P_ROW_WORDS = N_Q // 2  # the FULL P^T row of a key lane (every group's columns)

# K-major operands (K as A of BMM1, Q^T as B of BMM1): SBO = 8 rows x 128 B.
STRIDE_BYTE_OFFSET_QK = 8 * CFG.K_SWZ_BYTES
# V^T as the MN-major A of BMM2: 128 d columns span two 64-wide swizzle atoms, LBO steps between them (TILE_N rows x 128 B
# each); SBO = 8 key rows.
LEADING_BYTE_OFFSET_VT = TILE_N * CFG.V_SWZ_BYTES
STRIDE_BYTE_OFFSET_VT = 8 * CFG.V_SWZ_BYTES
# P^T as the MN-major B of BMM2: one atom wide (N_Q * 2 B), SBO = 8 key rows.
STRIDE_BYTE_OFFSET_P = 8 * CFG.P_SWZ_BYTES

# TMEM: STAGES_S S^T slots then the two O^T d-blocks, N_Q fp32 columns each (CFG.S_ACC_OFF / CFG.O_OFF, allocated once per CTA).
S_ACC_OFF = CFG.S_ACC_OFF
if len(S_ACC_OFF) != STAGES_S:
    raise ValueError(f"sparse_d256_f16: STAGES_S={STAGES_S} must equal the number of S^T TMEM slots (len(S_ACC_OFF)={len(S_ACC_OFF)})")
O_OFF = CFG.O_OFF
TMEM_COLS = CFG.TMEM_COLS
O_BLOCKS = TILE_O // 128

# The softmax role = SOFTMAX_GROUPS column groups of SOFTMAX_WARPS_PER_GROUP (4) warps each; group g owns columns [g COLS, (g + 1) COLS)
# of S^T / P^T / O^T over all 128 key / d lanes (one group: 4 warps x 16 columns = the 16 Q rows of the item -- today's body; two
# groups: 2 x 4 warps x 8 columns, the softmax chain per warp halved -- the header's SOFTMAX SPLIT paragraph).  Every count below is CFG's.
COLS = CFG.COLS_PER_GROUP
SOFTMAX_GROUPS = CFG.SOFTMAX_GROUPS
SOFTMAX_WARPS_PER_GROUP = CFG.SOFTMAX_WARPS_PER_GROUP
GROUP_LANES = 32 * SOFTMAX_WARPS_PER_GROUP  # 128 = the key rows of a tile = the lanes of ONE group
SOFTMAX_WARPS = CFG.SOFTMAX_WARPS  # all groups
SOFTMAX_LANES = CFG.SOFTMAX_LANES  # all groups: the init of every bare-arrive barrier the softmax role fires
GATHER_WARP_BASE = CFG.GATHER_WARP_BASE
GATHER_WARPS = CFG.GATHER_WARPS
MMA_WARP_ID = CFG.MMA_WARP_ID
TMALDG_WARP_ID = CFG.TMALDG_WARP_ID
SCHED_WARP_ID = CFG.SCHED_WARP_ID
P_CHUNKS = (COLS * BPE) // 16  # 16-B chunks of one key row's P^T slice written by ONE group (2 at one group, 1 at two)
P_GROUP_WORDS = COLS // 2  # the words ONE group packs: its COLS columns as half pairs
# Gate rows the epilogue reads per lane: the item's G head rows at one group; at two groups the group's COLS rows (rows past G read the
# slot's once-zeroed tail -- finite, and their columns are never stored).
N_GATE_ROWS = Q_BOX_ROWS if SOFTMAX_GROUPS == 1 else COLS
# Named barriers: 1 = TMEM base hand-off (every softmax warp + the MMA warp), 2 + g = column group g's own cross-warp reductions
# (ids 2 .. 2 + SOFTMAX_GROUPS - 1, thread_count = the group's 128 lanes; the groups never wait on each other).
_BAR_TMEM = 1
_BAR_SOFTMAX = 2
BAR_TMEM_THREADS = CFG.BAR_TMEM_THREADS
BAR_SOFTMAX_THREADS = CFG.BAR_SOFTMAX_THREADS
# Cross-warp reduction scratch: [STAGES_S tile parities + 1 epilogue][softmax warp (all groups)][COLS]; the epilogue slot is index STAGES_S;
# 192 words at either group count, group g reads the slice of its own four warps.
RED_SLOTS = STAGES_S + 1
RED_WORDS = RED_SLOTS * SOFTMAX_WARPS * COLS

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


class SparseBars(NamedTuple):
    mb_q_full: object
    mb_q_empty: object
    mb_ids_full: object
    mb_ids_empty: object
    mb_kv_full: object
    mb_kv_empty: object
    mb_s_full: object
    mb_s_empty: object
    mb_p_full: object
    mb_bmm2_done: object
    mb_tmem_dealloc: object
    mb_gate_full: object  # rows 11 / 12: the EPILOGUE_GATE arm only (None on the ungated rendering)
    mb_gate_empty: object


def make_sparse_bars() -> SparseBars:
    """The barrier table's rows 1-10 and 15 as typed MBarriers; init counts from CFG (the EXACT per-phase arrival sums the
    header's ledger derives: ONE_LANE for every TMA_LOAD / MMA_COMMIT producer, KV_FULL_ARRIVERS = one expect_tx per gather
    warp, IDS_EMPTY_ARRIVERS = one elected lane per list-reading warp, SOFTMAX_LANES for the bare lane arrives).  Rows 11 / 12
    (the gate's TMA_LOAD full barrier: ONE_LANE, and its THREAD empty barrier: SOFTMAX_LANES, one stage each -- the gate of
    item i aliases the item's own sQ slot, so one slot is in flight at a time) exist on the EPILOGUE_GATE rendering only."""

    def _alloc(n):
        return cutlass.Array(cutlass.Int64, n, alignment=16, space=cutlass.AddressSpace.smem)

    gate_full = MBarrier(_alloc(CFG.STAGES_GATE), stages=CFG.STAGES_GATE, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD) if CFG.EPILOGUE_GATE else None
    gate_empty = MBarrier(_alloc(CFG.STAGES_GATE), stages=CFG.STAGES_GATE, init_count=SOFTMAX_LANES, producer=Producer.THREAD) if CFG.EPILOGUE_GATE else None
    return SparseBars(
        mb_q_full=MBarrier(_alloc(STAGES_Q), stages=STAGES_Q, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_empty=MBarrier(_alloc(STAGES_Q), stages=STAGES_Q, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_ids_full=MBarrier(_alloc(STAGES_IDS), stages=STAGES_IDS, init_count=CFG.ONE_LANE, producer=Producer.TMA_LOAD),
        mb_ids_empty=MBarrier(_alloc(STAGES_IDS), stages=STAGES_IDS, init_count=CFG.IDS_EMPTY_ARRIVERS, producer=Producer.THREAD),
        mb_kv_full=MBarrier(_alloc(STAGES_KV), stages=STAGES_KV, init_count=CFG.KV_FULL_ARRIVERS, producer=Producer.TMA_LOAD),
        mb_kv_empty=MBarrier(_alloc(STAGES_KV), stages=STAGES_KV, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_s_full=MBarrier(_alloc(STAGES_S), stages=STAGES_S, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_s_empty=MBarrier(_alloc(STAGES_S), stages=STAGES_S, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_p_full=MBarrier(_alloc(STAGES_S), stages=STAGES_S, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_bmm2_done=MBarrier(_alloc(STAGES_S), stages=STAGES_S, init_count=CFG.ONE_LANE, producer=Producer.MMA_COMMIT),
        mb_tmem_dealloc=MBarrier(_alloc(1), stages=1, init_count=SOFTMAX_LANES, producer=Producer.THREAD),
        mb_gate_full=gate_full,
        mb_gate_empty=gate_empty,
    )


@cute.jit
def _item_bounds(tok, batch, item_row, eff_seqlen_q, block_lens_tensor: Optional[cute.Tensor], seq_kv_lens_tensor, seqlen_kv, split):
    """THE ONE HELPER (header): the row's key-space position, the entries read from the item's list, the appended tail block(s),
    the dead flag, the tile count and -- under CFG.SPLIT_KV -- this work item's chunk of the tiles, computed by EVERY role that
    loops over tiles from the SAME inputs (the item ``_decode_item`` handed it: the Q row ``tok``, the batch / sequence, the flat
    ``block_ids`` / ``block_lens`` row ``item_row``, the sequence's Q length ``eff_seqlen_q`` and the split index), nothing
    staged through SMEM.

    The VISIBLE SET (position, count, tail, dead) is ``_sparse_item_common.visible_bounds`` -- the helper this module SHARES with the
    lean ``mma.sync`` decode tile (``sparse_lean_d256_f16.py``), every CFG knob handed over as a constexpr parameter (the cubins of
    this module are byte-identical with the arithmetic inlined and with it shared, the md5 record in
    ``test_sdpa_fwd_sparse_lean_sm107.py``; the visible-token count the helper returns is unused here); its arithmetic (eff_seqlen_kv, pos, the anchor, n_sel_default, count, tail_lo, n_tail,
    dead) is documented there and in the header above.  This module's own geometry follows:
    n_tiles = max(1, ceil((count + n_tail) / BLOCKS_PER_TILE)) -- CLAMPED >= 1: a dead item runs ONE tile of -1 rows.
    SPLIT_KV > 1: the item's tiles are cut into SPLIT_KV contiguous chunks (per = n // S, rem = n % S: chunk s = [s per + min(s, rem),
    + per + (s < rem))) and this work item runs chunk ``split`` from ``tile0``; an EMPTY chunk is a dead item (one clamped tile).
    The store predicate is the Q ROW's liveness (``tok < eff_seqlen_q``), computed by the softmax group at the store.
    """
    pos, count, n_tail, tail_lo, dead, n_listed, eff_seqlen_kv, _n_vis = visible_bounds(
        tok,
        batch,
        item_row,
        eff_seqlen_q,
        block_lens_tensor,
        seq_kv_lens_tensor,
        seqlen_kv,
        seq_kv_lens_present=CFG.SEQ_KV_LENS_PRESENT,
        bottom_right=BOTTOM_RIGHT,
        list_per_sequence=LIST_PER_SEQUENCE,
        include_open_block=CFG.INCLUDE_OPEN_BLOCK,
        block_size=BLOCK_SIZE,
        block_topk=BLOCK_TOPK,
    )
    zero = cutlass.Int32(0)
    n_tiles = cute.math.max(cutlass.Int32(1), (n_listed + cutlass.Int32(BLOCKS_PER_TILE - 1)) // cutlass.Int32(BLOCKS_PER_TILE))
    tile0 = zero
    if cutlass.const_expr(SPLIT_KV > 1):
        per = n_tiles // cutlass.Int32(SPLIT_KV)
        rem = n_tiles % cutlass.Int32(SPLIT_KV)
        tile0 = split * per + cute.math.min(split, rem)
        n_sp = per + _select_i32(split < rem, 1, 0)
        dead = dead | (n_sp == zero)
        n_tiles = cute.math.max(cutlass.Int32(1), n_sp)
    return pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0


@cute.jit
def _block_id_at(idx, loaded, count, n_tail, tail_lo, dead):
    """The block id at list index ``idx`` of the current item (the staged word while ``idx < count``, the tail blocks at
    ``idx == count .. count + n_tail - 1``, -1 past that and for every index of a dead item): the shared
    ``_sparse_item_common.block_id_at`` with this module's constants -- ONE spelling of the visible set for both bodies."""
    return block_id_at(idx, loaded, count, n_tail, tail_lo, dead, list_per_sequence=LIST_PER_SEQUENCE, block_size=BLOCK_SIZE)


@cute.jit
def _thd_find_batch(meta, n_batch, tok_packed):
    """The sequence holding packed Q token ``tok_packed``: the LAST ``b`` with ``cu_q[b] <= tok_packed`` -- a binary search over
    the ``B + 1`` Q prefix sums at ``meta[B .. 2B]`` (an empty sequence has ``cu_q[b] == cu_q[b + 1]`` and can never hold the
    token, so it is stepped over).  Loop-carried ``lo`` / ``hi`` as SELECTs: ceil(log2(B + 1)) iterations, one LDG each."""
    cuq0 = n_batch
    lo = cutlass.Int32(0)
    hi = n_batch
    while hi - lo > cutlass.Int32(1):
        mid = (lo + hi) // cutlass.Int32(2)
        take = cutlass.Int32(meta[cuq0 + mid]) <= tok_packed
        lo = _select_i32(take, mid, lo)
        hi = _select_i32(take, hi, mid)
    return lo


@cute.jit
def _decode_item(t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv):
    """ONE decode of a work item for EVERY role and EVERY item (the CTA's first item from its grid position, every later one
    from the scheduler payload) -> ``(tok, head, batch, item_row, kv_row_base, eff_seqlen_q, out_tok, out_batch, split)``:

    * BSHD (the CLC form): the grid is (S_q x SPLIT_KV, H_kv, B) and the payload packs ``ctaid.x`` / ``ctaid.y | ctaid.z <<
      16``; ``tok = x % S_q`` and ``split = x // S_q`` (SPLIT_KV > 1; else ``tok = x``, ``split = 0`` -- the MTP rows of one
      (sequence, head), which gather the SAME tiles, stay adjacent in launch order), ``item_row = batch x S_q + tok`` (the flat
      ``block_ids`` / ``block_lens`` row) or ``batch`` under LIST_PER_SEQUENCE (one list per sequence), ``kv_row_base = batch x
      S_kv`` (the gather map's rows are the tokens of every batch), the Q / O / LSE coordinates are ``(tok, batch)``; the
      key-space position (bottom-right or not) is ``_item_bounds``' business.
    * THD (the persistent claim-counter form, CFG.THD_VARLEN): the payload is the unit id ``uid`` in ``[0, T_live x H_kv)``
      with ``T_live = cu_q[B]`` (the metadata's device value): ``head = uid // T_live``, ``tok_packed = uid % T_live`` (the token
      runs fastest, so adjacent units -- adjacent tokens of one KV head, whose lists overlap -- gather concurrently), the
      sequence by ``_thd_find_batch``, ``tok = tok_packed - cu_q[b]`` (sequence-relative: the list's block ids are the
      sequence's own), ``item_row = tok_packed``, ``kv_row_base = cu_k[b]`` (the packed K / V rows of the sequence), the Q / O /
      LSE coordinates ``(tok_packed, 0)`` on the ``[1, T, H, D]`` operands; ``eff_seqlen_q = cu_q[b + 1] - cu_q[b]``; ``split =
      0`` (the three decode-form arms never render under THD).
    """
    split = cutlass.Int32(0)
    if cutlass.const_expr(CFG.THD_VARLEN == 1):
        meta = cutlass.make_array_view(seq_kv_lens_tensor)
        uid = cute.arch.make_warp_uniform(t0)
        t_live = cutlass.Int32(meta[THD_CU_Q_TOTAL_OFF(n_batch)])
        head = cute.arch.make_warp_uniform(uid // t_live)
        tok_packed = cute.arch.make_warp_uniform(uid - head * t_live)
        batch = cute.arch.make_warp_uniform(_thd_find_batch(meta, n_batch, tok_packed))
        cu_q_b = cutlass.Int32(meta[n_batch + batch])
        tok = tok_packed - cu_q_b
        eff_seqlen_q = cutlass.Int32(meta[n_batch + batch + cutlass.Int32(1)]) - cu_q_b
        item_row = tok_packed
        kv_row_base = cutlass.Int32(meta[cutlass.Int32(2) * n_batch + cutlass.Int32(1) + batch])
        out_tok = tok_packed
        out_batch = cutlass.Int32(0)
    else:
        x = cute.arch.make_warp_uniform(t0)
        tok = x
        if cutlass.const_expr(SPLIT_KV > 1):
            split = cute.arch.make_warp_uniform(x // seqlen_q)
            tok = x - split * seqlen_q
        head = cute.arch.make_warp_uniform(t1 & cutlass.Int32(0xFFFF))
        batch = cute.arch.make_warp_uniform((t1 >> cutlass.Int32(16)) & cutlass.Int32(0xFFFF))
        eff_seqlen_q = seqlen_q
        if cutlass.const_expr(LIST_PER_SEQUENCE == 1):
            item_row = batch
        else:
            item_row = batch * seqlen_q + tok
        kv_row_base = batch * seqlen_kv
        out_tok = tok
        out_batch = batch
    return tok, head, batch, item_row, kv_row_base, eff_seqlen_q, out_tok, out_batch, split


# === TMA-LDG warp (CFG.TMALDG_WARP_ID: 17 at two softmax column groups, 13 at one): Q^T and the block list of the NEXT item, one item
# === ahead; the gate of the CURRENT item (gated arm) =======================================================================


@cute.jit
def _tma_issue_item(
    tma_q,
    sQ,
    sIds_raw,
    bars,
    ids_base,
    item_row,
    head,
    out_tok,
    out_batch,
    q_idx,
    q_phase,
    qe_idx,
    qe_phase,
    i_idx,
    i_phase,
    ie_idx,
    ie_phase,
    ge_idx,
    ge_phase,
    first: cutlass.Constexpr[bool],
):
    """Issue the two loads of one work item: the ``BLOCK_TOPK x 4`` B list row ``item_row`` by ONE bulk copy into
    ``sIds[slot]`` (barrier row 3; unconditional -- dead items included, the row exists for every token) and the ``Q^T`` box
    of the token's G heads at the Q coordinate ``(out_tok, out_batch)`` into ``sQ[slot]`` (row 1; ``Q_TX_BYTES`` = the BOX
    bytes).  Both coordinates come from ``_decode_item`` (under THD the packed token row and batch 0: a dense coordinate that
    survived the port would read batch 0's rows for every sequence).  The ids slot is taken after its ``_empty`` wait (row
    4, pre-armed).  The Q slot's wait is the arm's: ungated, ``mb_q_empty[slot]`` (row 2, pre-armed: Q(0), Q(1) pass fresh);
    gated, ``mb_gate_empty`` (row 12, pre-armed, ONE stage: Q(1) passes fresh, Q(j) waits epilogue(j - 2) -- the consumer of
    gate(j - 2) in this same slot), skipped for the prologue's Q(0) (``first``: a fresh slot, and the pre-armed pass is
    Q(1)'s); the wait Q(n_items) would have taken is the caller's exit-path wait before the last gate.  Returns the advanced
    PipelineStates as (idx, phase) pairs, the gate-empty consumer's (``ge``, one stage = CFG.STAGES_GATE, advanced per wait like
    every other ring -- never a scalar parity toggle) included."""
    # --- the list (row 4 wait -> row 3 arm + copy)
    bars.mb_ids_empty[ie_idx].wait(ie_phase)
    bars.mb_ids_full[i_idx].arrive(n_bytes=IDS_TX_BYTES, pred=nvvm.elect_sync())
    bulk_copy(
        sIds_raw.subview(i_idx * cutlass.Int32(IDS_SLOT_WORDS)), ids_base + item_row * cutlass.Int32(BLOCK_TOPK), IDS_TX_BYTES, bars.mb_ids_full[i_idx].smem_ptr
    )
    # --- Q^T (the slot wait -> row 1 arm + 4 subtiles)
    if cutlass.const_expr(CFG.EPILOGUE_GATE):
        if cutlass.const_expr(not first):
            bars.mb_gate_empty.wait(ge_phase)
            ge_state = advance(PipelineState(idx=ge_idx, phase=ge_phase), CFG.STAGES_GATE)
            ge_idx, ge_phase = ge_state.idx, ge_state.phase
    else:
        bars.mb_q_empty[qe_idx].wait(qe_phase)
    bars.mb_q_full[q_idx].arrive(n_bytes=Q_TX_BYTES, pred=nvvm.elect_sync())
    tma_load_tile(sQ[q_idx], tma_q(cutlass.Int32(0), head * cutlass.Int32(G), out_tok, out_batch), bars.mb_q_full[q_idx].smem_ptr, cta_group=1)
    q_state = advance(PipelineState(idx=q_idx, phase=q_phase), STAGES_Q)
    qe_state = advance(PipelineState(idx=qe_idx, phase=qe_phase), STAGES_Q)
    i_state = advance(PipelineState(idx=i_idx, phase=i_phase), STAGES_IDS)
    ie_state = advance(PipelineState(idx=ie_idx, phase=ie_phase), STAGES_IDS)
    return q_state.idx, q_state.phase, qe_state.idx, qe_state.phase, i_state.idx, i_state.phase, ie_state.idx, ie_state.phase, ge_idx, ge_phase


@cute.jit
def _tma_issue_gate(tma_gate, sQ, bars, out_tok, head, out_batch, qg_idx, qg_phase):
    """Gated arm: the gate tile of the CURRENT item -- O's shape, the token's G query heads = exactly the item's Q^T box, the
    same 4-D tensor-map form and SW128 -- into the item's OWN sQ slot once the MMA warp freed it (row 2's commit after the
    item's last BMM1; ``qg`` is the wait-then-arrive consumer state: item i waits completion i // 2 of slot i % 2 = its own
    commit).  The coordinate is the item's OUTPUT coordinate ``(out_tok, out_batch)`` of ``_decode_item`` -- the gate is
    O-shaped, so under THD it is the packed token row and batch 0 on the ``[1, T_q, H, D]`` operand, exactly where O's rows
    go; under the paged read it is the dense ``(tok, batch)`` (the gate is never paged).  Arms row 11 with ``GATE_TX_BYTES``
    = the BOX bytes (Q_BOX_ROWS x TILE_O x GATE_BPE, never the 8 KiB slot).  Issued for every item, dead ones included, so
    rows 11 / 12 cycle exactly once per item on every path (P14)."""
    bars.mb_q_empty[qg_idx].wait(qg_phase)
    bars.mb_gate_full.arrive(n_bytes=GATE_TX_BYTES, pred=nvvm.elect_sync())
    tma_load_tile(sQ[qg_idx], tma_gate(cutlass.Int32(0), head * cutlass.Int32(G), out_tok, out_batch), bars.mb_gate_full.smem_ptr, cta_group=1)
    qg_state = advance(PipelineState(idx=qg_idx, phase=qg_phase), STAGES_Q)
    return qg_state.idx, qg_state.phase


@cute.jit
def _tmaldg_warp_group(
    tma_q_desc,
    tma_gate_desc,
    sQ,
    sIds_raw,
    bars,
    sched,
    block_ids_tensor,
    seq_kv_lens_tensor,
    n_batch,
    seqlen_q,
    seqlen_kv,
    t0_init,
    t1_init,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    tma_q = GmemTileTma(tma_q_desc)
    tma_gate = GmemTileTma(tma_gate_desc) if cutlass.const_expr(CFG.EPILOGUE_GATE) else None
    ids_base = Pointer(block_ids_tensor.iterator.raw_ptr(), dtype=cutlass.Int32)
    # Producer states (slot to fill next) and the pre-armed consumer states of the matching _empty rings (P5b: Q(0), Q(1)
    # and ids(0), ids(1) pass fresh barriers) -- all four carried across items.  Gated arm: qg_state is the gate's
    # wait-then-arrive consumer state of mb_q_empty (every commit consumed by its item's gate load, nothing left to drain) and
    # ge_state the pre-armed mb_gate_empty consumer (one stage = CFG.STAGES_GATE, advanced per wait like every other ring: Q(1)
    # passes fresh; Q(j) waits epilogue(j - 2)); the ungated arm's qe_state keeps its role there and is unused here.
    q_state = PipelineState.start(phase=0)
    qe_state = PipelineState.start(phase=1)
    i_state = PipelineState.start(phase=0)
    ie_state = PipelineState.start(phase=1)
    qg_state = PipelineState.start(phase=0)
    ge_state = PipelineState.start(phase=1)

    if cutlass.const_expr(use_pdl):
        # PDL (header, PDL SITES): this role's first global reads are the prologue's ids bulk copy (block_ids: the selector's / the
        # caller's list) and the Q^T box (the slab a projection / norm + RoPE launch wrote); the tensor-map prefetches above are kernel
        # parameters.  Nothing of the role is read or written above this wait.
        wait_on_dependent_grids()
    # Prologue: the first item's loads (its coordinates are this CTA's grid position, decoded like every later payload).
    _tok, head, _batch, item_row, _kvb, _sq, out_tok, out_batch, _split = _decode_item(t0_init, t1_init, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv)
    qi, qp, qei, qep, ii, ip, iei, iep, gei, gep = _tma_issue_item(
        tma_q,
        sQ,
        sIds_raw,
        bars,
        ids_base,
        item_row,
        head,
        out_tok,
        out_batch,
        q_state.idx,
        q_state.phase,
        qe_state.idx,
        qe_state.phase,
        i_state.idx,
        i_state.phase,
        ie_state.idx,
        ie_state.phase,
        ge_state.idx,
        ge_state.phase,
        True,
    )
    q_state = PipelineState(idx=qi, phase=qp)
    qe_state = PipelineState(idx=qei, phase=qep)
    i_state = PipelineState(idx=ii, phase=ip)
    ie_state = PipelineState(idx=iei, phase=iep)
    ge_state = PipelineState(idx=gei, phase=gep)

    # The CURRENT item's output coordinate (the gate's: O-shaped, (out_tok, out_batch) of _decode_item -- the packed token row
    # and batch 0 under THD); the payload below names the NEXT one.
    cur_out_tok, cur_head, cur_out_batch = out_tok, head, out_batch
    is_valid = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid > cutlass.Int32(0):
        # Credit the scheduler slot at the TOP of the item (row 14), then learn the NEXT item and prefetch its Q^T and list
        # under the current item's KV loop.  The _empty waits inside the issue bound this warp at one item ahead.
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), 1)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        t0, t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        _tok, head, _batch, item_row, _kvb, _sq, out_tok, out_batch, _split = _decode_item(t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv)
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        sched_state = advance(sched_state, SCHEDULER_STAGES)
        if is_valid > cutlass.Int32(0):
            qi, qp, qei, qep, ii, ip, iei, iep, gei, gep = _tma_issue_item(
                tma_q,
                sQ,
                sIds_raw,
                bars,
                ids_base,
                item_row,
                head,
                out_tok,
                out_batch,
                q_state.idx,
                q_state.phase,
                qe_state.idx,
                qe_state.phase,
                i_state.idx,
                i_state.phase,
                ie_state.idx,
                ie_state.phase,
                ge_state.idx,
                ge_state.phase,
                False,
            )
            q_state = PipelineState(idx=qi, phase=qp)
            qe_state = PipelineState(idx=qei, phase=qep)
            i_state = PipelineState(idx=ii, phase=ip)
            ie_state = PipelineState(idx=iei, phase=iep)
            ge_state = PipelineState(idx=gei, phase=gep)
        if cutlass.const_expr(CFG.EPILOGUE_GATE):
            # The gate of the CURRENT item, after the next item's prefetch was queued (the Q / ids prefetch keeps its place
            # under this item's KV loop; the gate's wait is the item's LAST BMM1 -- the late one) and before the next credit
            # (which the softmax warps give only after epilogue(i), later than this issue).
            if is_valid <= cutlass.Int32(0):
                # The LAST item (row 12's exit path): no Q(i + 1) is issued, so the mb_gate_empty wait Q(i + 1) would have taken --
                # the one that consumes epilogue(i - 1)'s release of the other slot -- is taken HERE, before gate(i) exists.  ORDER,
                # not count: wait() returns when the barrier's parity differs from the waited one, so a consumer that falls TWO
                # completions behind a single-stage barrier spins forever (parity(i - 1) == parity(i + 1)) -- a hang with correct
                # numerics.  Taken here, completion i cannot land before this wait (it needs gate(i)), so the consumer is never more
                # than one completion behind, and the exit drain below consumes exactly STAGES_GATE = 1 completion (epilogue(i)).
                # n_items = 1: this is the pre-armed pass, taken while nothing can have arrived (gate(0) is issued after it).
                bars.mb_gate_empty.wait(ge_state.phase)
                ge_state = advance(ge_state, CFG.STAGES_GATE)
            qgi, qgp = _tma_issue_gate(tma_gate, sQ, bars, cur_out_tok, cur_head, cur_out_batch, qg_state.idx, qg_state.phase)
            qg_state = PipelineState(idx=qgi, phase=qgp)
        cur_out_tok, cur_head, cur_out_batch = out_tok, head, out_batch

    # Exit drains (P6): the last STAGES_IDS ids_empty arrives need a waiter; the carried states name exactly those phases.
    for _ in cutlass.range_constexpr(STAGES_IDS):
        bars.mb_ids_empty[ie_state.idx].wait(ie_state.phase)
        ie_state = advance(ie_state, STAGES_IDS)
    if cutlass.const_expr(CFG.EPILOGUE_GATE):
        # Row 12: n_items completions against n_items waits -- n_items - 1 before Q(1 .. n_items - 1) (Q(1)'s the pre-armed pass,
        # Q(j)'s = epilogue(j - 2)) plus the exit-path wait taken above BEFORE the last gate (= epilogue(n_items - 2); the pass
        # itself at n_items = 1) -> exactly STAGES_GATE = 1 unconsumed (epilogue(n_items - 1)) for every n_items >= 1, and the
        # consumer never two completions behind the single-stage barrier.  Row 2 needs no drain here: every mb_q_empty commit
        # was consumed by its item's gate load.
        for _ in cutlass.range_constexpr(CFG.STAGES_GATE):
            bars.mb_gate_empty.wait(ge_state.phase)
            ge_state = advance(ge_state, CFG.STAGES_GATE)
    else:
        # The last STAGES_Q q_empty commits need a waiter.
        for _ in cutlass.range_constexpr(STAGES_Q):
            bars.mb_q_empty[qe_state.idx].wait(qe_state.phase)
            qe_state = advance(qe_state, STAGES_Q)


# === GATHER warps (CFG.GATHER_WARP_BASE .. + 7: 8-15 at two softmax column groups, 4-11 at one): the K/V ring, block by block =====


@cute.jit
def _gather_block_ids(sIds_raw, slot_base, tile, w, count, n_tail, tail_lo, dead):
    """This warp's BLOCKS_PER_WARP block ids of tile ``tile`` (the item's ABSOLUTE tile index: the split chunk's ``tile0 + i``):
    list indices ``32 tile + BLOCKS_PER_WARP w + q``.  The staged words are read with warp-uniform 16-B vector loads (the quad
    is 16-B aligned: BLOCK_TOPK % 4 == 0 and the slot start is) BEFORE the ring wait, at an address clamped into the slot; the
    SELECT (``_block_id_at``) substitutes the tail id(s) / -1 / the dead item's -1 per index.  Returns a Vector of
    BLOCKS_PER_WARP Int32."""
    first = tile * cutlass.Int32(BLOCKS_PER_TILE) + w * cutlass.Int32(BLOCKS_PER_WARP)
    ld_first = cute.math.min(first, cutlass.Int32(BLOCK_TOPK - BLOCKS_PER_WARP))
    ids = []
    for qq in cutlass.range_constexpr(BLOCKS_PER_WARP // 4):
        v = sIds_raw.load(slot_base + ld_first + cutlass.Int32(qq * 4), vector_size=4, alignment=16)
        for r in cutlass.range_constexpr(4):
            q = qq * 4 + r
            ids.append(_block_id_at(first + cutlass.Int32(q), cutlass.Int32(v[r]), count, n_tail, tail_lo, dead))
    return cutlass.Vector.from_elements(tuple(ids), cutlass.Int32)


@cute.jit
def _gather_block_pages(blks, block_table_tensor, batch, max_pages, eff_seqlen_kv):
    """PAGED arm: the pool PAGE of each block of this warp's quad -- ONE block-table lookup per block, hoisted here (the ids
    step, before the ring wait) so the fill does arithmetic only: ``page = block_table[batch, (4 blk) // PAGE_SIZE]`` read at
    an index clamped into the table row (always in bounds), then -1 unless the block is live (``blk >= 0``), its first token
    is inside the batch's KV length (a block at or past it may name another sequence's stale page), the page index is inside
    the table and the entry is not -1 (the page -1 convention of the dense paged tiles).  A -1 page becomes row -1 at the
    fill: TMA zero-fill, bytes credited.  Returns a Vector of BLOCKS_PER_WARP Int32."""
    bt = cutlass.make_array_view(block_table_tensor)
    last = cute.math.max(max_pages - cutlass.Int32(1), cutlass.Int32(0))
    pages = []
    for q in cutlass.range_constexpr(BLOCKS_PER_WARP):
        blk = cutlass.Int32(blks[q])
        key0 = blk * cutlass.Int32(BLOCK_SIZE)
        page_idx = key0 // cutlass.Int32(PAGE_SIZE)
        rd = cute.math.min(cute.math.max(page_idx, cutlass.Int32(0)), last)
        page = cutlass.Int32(bt[batch, rd])
        ok = (blk >= cutlass.Int32(0)) & (key0 < eff_seqlen_kv) & (page_idx < max_pages) & (page >= cutlass.Int32(0))
        pages.append(_select_i32(ok, page, cutlass.Int32(-1)))
    return cutlass.Vector.from_elements(tuple(pages), cutlass.Int32)


@cute.jit
def _gather_fill(desc, sKV_raw, stage_idx, mbar, w, blks, pages, kv_row_base, rows_per_page, head_rows, eff_seqlen_kv, col_base, hint):
    """ONE warp's share of one K or V stage: BLOCKS_PER_WARP blocks x GATHER_BOXES column boxes = 16 ``tma_gather4`` from
    the ELECTED lane (the caller elects).  Block ``q`` of this warp = quad ``BLOCKS_PER_WARP w + q`` of the tile, landing at
    ``box * 16 KiB + quad * 512 B`` inside the stage -- a 512-B-aligned quad in a 1024-B-aligned sub-box, so the swizzle the
    hardware applies equals a tiled load's (the tile then reads under the dense descriptors).  Row coordinates carry the
    item's ``kv_row_base`` (``batch x S_kv`` on the dense map whose rows are the tokens of every batch; ``cu_k[b]`` under THD,
    the sequence's first packed K / V row -- the block ids are the sequence's own) on the dense arm, and on the PAGED arm the
    pool row ``page x rows_per_page + head_rows + (4 blk) % PAGE_SIZE`` (``pages`` from ``_gather_block_pages``; ``head_rows``
    = ``head x rows_per_head``, 0 for an NHD pool whose head lives in the column; ``kv_row_base`` is unread there -- THD x
    paged is a typed decline); a ``-1`` block (or page), a block past the sequence's length or the straddling rows of its
    last block take row ``-1`` -> TMA zero-fills and still credits the bytes, so no foreign batch's / sequence's row (or its
    NaN) ever reaches an MMA under P = 0."""
    stage_off = stage_idx * cutlass.Int32(kvBufferElems)
    quad0 = w * cutlass.Int32(BLOCKS_PER_WARP)
    for q in cutlass.range_constexpr(BLOCKS_PER_WARP):
        blk = cutlass.Int32(blks[q])
        key0 = blk * cutlass.Int32(BLOCK_SIZE)
        if cutlass.const_expr(PAGED_KV):
            page = cutlass.Int32(pages[q])
            in_page = key0 - (key0 // cutlass.Int32(PAGE_SIZE)) * cutlass.Int32(PAGE_SIZE)
            base = page * rows_per_page + head_rows + in_page
            live = page >= cutlass.Int32(0)
        else:
            base = kv_row_base + key0
            live = blk >= cutlass.Int32(0)
        rows = []
        for r in cutlass.range_constexpr(BLOCK_SIZE):
            rows.append(_select_i32(live & ((key0 + cutlass.Int32(r)) < eff_seqlen_kv), base + cutlass.Int32(r), cutlass.Int32(-1)))
        quad_elems = (quad0 + cutlass.Int32(q)) * cutlass.Int32(BLOCK_SIZE * GATHER_BOX_ELEMS)
        for b in cutlass.range_constexpr(GATHER_BOXES):
            tma_gather4(
                desc,
                sKV_raw.subview(stage_off + cutlass.Int32(b * SUB_BOX_ELEMS) + quad_elems),
                mbar,
                col_base + cutlass.Int32(b * GATHER_BOX_ELEMS),
                rows[0],
                rows[1],
                rows[2],
                rows[3],
                cta_group=1,
                l2_hint=hint,
            )


@cute.jit
def _gather_warp_group(
    tma_k_desc,
    tma_v_desc,
    sKV_raw,
    sIds_raw,
    bars,
    sched,
    block_lens_tensor,
    seq_kv_lens_tensor,
    seqlen_q,
    seqlen_kv,
    k_head_stride,
    v_head_stride,
    w,
    n_batch,
    t0_init,
    t1_init,
    block_table_tensor: Optional[cute.Tensor] = None,
    max_pages=None,
    k_rows_per_page=None,
    k_rows_per_head=None,
    v_rows_per_page=None,
    v_rows_per_head=None,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    if cutlass.const_expr(use_pdl):
        # PDL (header, PDL SITES): this role's first global reads are _item_bounds' lengths / counts, then the block table (paged arm)
        # and the K / V rows of the gathers -- all a predecessor's output; nothing is read above this wait.
        wait_on_dependent_grids()
    hint = opaque_i64(TMA_L2_EVICT_LAST)
    # kv_state: the stage to fill next (producer of mb_kv_full) == the mb_kv_empty wait (pre-armed: the first STAGES_KV
    # fills pass fresh barriers); advanced per LOAD in the K(0), K(1), V(0), K(2), V(1), ... order the MMA warp consumes,
    # CARRIED across items (never recomputed from a per-item tile index -- barrier rows 5 / 6).
    kv_state = PipelineState.start(phase=1)
    ids_state = PipelineState.start(phase=0)
    tok, head, batch, item_row, kv_row_base, eff_seqlen_q, _ot, _ob, split = _decode_item(t0_init, t1_init, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv)
    is_valid = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), 1)
        _pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0 = _item_bounds(
            tok, batch, item_row, eff_seqlen_q, block_lens_tensor, seq_kv_lens_tensor, seqlen_kv, split
        )
        col_k = head * k_head_stride
        col_v = head * v_head_stride
        # PAGED arm: the per-item row terms of the head (0 for an NHD pool: its head lives in the column coordinate).
        head_rows_k = head * k_rows_per_head if cutlass.const_expr(PAGED_KV) else None
        head_rows_v = head * v_rows_per_head if cutlass.const_expr(PAGED_KV) else None
        slot_base = ids_state.idx * cutlass.Int32(IDS_SLOT_WORDS)
        bars.mb_ids_full[ids_state.idx].wait(ids_state.phase)

        # Ring order K(0), K(1), V(0), K(2), V(1), ...: K runs one tile ahead of V so BMM1(t+1) overlaps the softmax of t.
        # PAGED arm: the block-table lookups of a tile's quad ride with its ids (before the ring wait), K and V share them.
        blks_cur = _gather_block_ids(sIds_raw, slot_base, tile0, w, count, n_tail, tail_lo, dead)
        pages_cur = _gather_block_pages(blks_cur, block_table_tensor, batch, max_pages, eff_seqlen_kv) if cutlass.const_expr(PAGED_KV) else None
        bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES_PER_WARP, pred=nvvm.elect_sync())
        if nvvm.elect_sync():
            _gather_fill(
                tma_k_desc,
                sKV_raw,
                kv_state.idx,
                bars.mb_kv_full[kv_state.idx].smem_ptr,
                w,
                blks_cur,
                pages_cur,
                kv_row_base,
                k_rows_per_page,
                head_rows_k,
                eff_seqlen_kv,
                col_k,
                hint,
            )
        kv_state = advance(kv_state, STAGES_KV)
        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            # The next tile's ids are read before this iteration's ring waits (an in-bounds address even past the last tile).
            t_next = cute.math.min(i + cutlass.Int32(1), n_tiles - cutlass.Int32(1))
            if cutlass.const_expr(SPLIT_KV > 1):
                t_next = tile0 + t_next
            blks_next = _gather_block_ids(sIds_raw, slot_base, t_next, w, count, n_tail, tail_lo, dead)
            pages_next = _gather_block_pages(blks_next, block_table_tensor, batch, max_pages, eff_seqlen_kv) if cutlass.const_expr(PAGED_KV) else None
            if i + cutlass.Int32(1) < n_tiles:
                bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES_PER_WARP, pred=nvvm.elect_sync())
                if nvvm.elect_sync():
                    _gather_fill(
                        tma_k_desc,
                        sKV_raw,
                        kv_state.idx,
                        bars.mb_kv_full[kv_state.idx].smem_ptr,
                        w,
                        blks_next,
                        pages_next,
                        kv_row_base,
                        k_rows_per_page,
                        head_rows_k,
                        eff_seqlen_kv,
                        col_k,
                        hint,
                    )
                kv_state = advance(kv_state, STAGES_KV)
            bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_kv_full[kv_state.idx].arrive(n_bytes=KV_TX_BYTES_PER_WARP, pred=nvvm.elect_sync())
            if nvvm.elect_sync():
                _gather_fill(
                    tma_v_desc,
                    sKV_raw,
                    kv_state.idx,
                    bars.mb_kv_full[kv_state.idx].smem_ptr,
                    w,
                    blks_cur,
                    pages_cur,
                    kv_row_base,
                    v_rows_per_page,
                    head_rows_v,
                    eff_seqlen_kv,
                    col_v,
                    hint,
                )
            kv_state = advance(kv_state, STAGES_KV)
            blks_cur = blks_next
            pages_cur = pages_next

        # The last read of this item's list slot (row 4: ONE lane per consuming warp).
        if nvvm.elect_sync():
            bars.mb_ids_empty[ids_state.idx].arrive()
        ids_state = advance(ids_state, STAGES_IDS)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        t0, t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        tok, head, batch, item_row, kv_row_base, eff_seqlen_q, _ot, _ob, split = _decode_item(t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv)
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        sched_state = advance(sched_state, SCHEDULER_STAGES)

    # Exit drain (P6): the MMA warp's last STAGES_KV kv_empty commits need a waiter; the carried state names their phases.
    for _ in cutlass.range_constexpr(STAGES_KV):
        bars.mb_kv_empty[kv_state.idx].wait(kv_state.phase)
        kv_state = advance(kv_state, STAGES_KV)


# === MMA warp (CFG.MMA_WARP_ID: 16 at two softmax column groups, 12 at one) ================================================


@cute.jit
def _mma_warp_group(
    sQ,
    sK,
    sVt,
    sP,
    tmem_ptr_i32,
    bars,
    sched,
    block_lens_tensor,
    seq_kv_lens_tensor,
    n_batch,
    seqlen_q,
    seqlen_kv,
    t0_init,
    t1_init,
    use_pdl: cutlass.Constexpr[bool] = False,
):
    tmem_alloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)
    nvvm.barrier_cta_arrive(_BAR_TMEM, BAR_TMEM_THREADS)

    tmem_raw = nvvm.make_tmem_ptr(tmem_ptr_i32.load(), cutlass.Int8)

    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=TILE_N,
        k_dim=0,
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=STORAGE_DTYPE,
        b_dtype=STORAGE_DTYPE,
        n_dim=N_Q,
        m_dim=128,
        a_major=1,
        b_major=1,
        k_dim=0,
    )
    # BMM1: S^T = K . Q^T -- K [128 keys x TILE_K] K-major, Q^T [N_Q x TILE_K] K-major.
    bmm1_desc = MmaDesc(
        M=TILE_N,
        N=N_Q,
        K=TILE_K,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=MMA_KIND,
    )
    # BMM2: O^T = V^T . P^T -- V^T [128 d x 128 keys] MN-major (d contiguous), P^T [N_Q x 128 keys] MN-major.
    bmm2_desc = MmaDesc(
        M=128,
        N=N_Q,
        K=TILE_N,
        bpe_a=BPE,
        bpe_b=BPE,
        tile_k_hw=CFG.TILE_K_HW,
        atranspose=True,
        btranspose=True,
        cta_group=1,
        idesc=idesc_pv,
        kind=MMA_KIND,
    )

    # Every ring slot and parity below comes from a PipelineState CARRIED across work items (barrier rows 1, 2, 5-10):
    #   q_state      -- mb_q_full wait (consumer) and the slot of the mb_q_empty commit, once per item;
    #   kv_state     -- mb_kv_full wait + the slot of the mb_kv_empty commit, per LOAD in the gather warps' order;
    #   s_state      -- the S^T slot of BMM1(t) = the mb_s_full commit slot, per tile;
    #   s_empty_state-- mb_s_empty wait (pre-armed: BMM1(0), BMM1(1) pass fresh), per tile;
    #   p_state      -- mb_p_full wait, per tile;  bmm2_state -- the mb_bmm2_done commit slot, per tile.
    q_state = PipelineState.start(phase=0)
    kv_state = PipelineState.start(phase=0)
    s_state = PipelineState.start(phase=0)
    s_empty_state = PipelineState.start(phase=1)
    p_state = PipelineState.start(phase=0)
    bmm2_state = PipelineState.start(phase=0)

    if cutlass.const_expr(use_pdl):
        # PDL (header, PDL SITES): the TMEM allocation, the named-barrier hand-off and the descriptor builds above are this role's
        # prologue; its only global reads are _item_bounds' lengths / counts, below (the operands arrive through SMEM).
        wait_on_dependent_grids()
    tok, head, batch, item_row, _kvb, eff_seqlen_q, _ot, _ob, split = _decode_item(t0_init, t1_init, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv)
    is_valid = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), 1)
        _pos, _count, _n_tail, _tail_lo, _dead, n_tiles, _ekv, _tile0 = _item_bounds(
            tok, batch, item_row, eff_seqlen_q, block_lens_tensor, seq_kv_lens_tensor, seqlen_kv, split
        )

        bars.mb_q_full[q_state.idx].wait(q_state.phase)
        desc_Q = sQ[q_state.idx].desc()

        # Prologue: BMM1(0).  The item's LAST BMM1 (here when n_tiles == 1) also commits mb_q_empty (row 2): the commit tracks
        # every prior MMA of this thread, so it fires once the last read of sQ[slot] is done.
        bars.mb_kv_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
        bars.mb_s_empty[s_empty_state.idx].wait(s_empty_state.phase, spin=SPIN_RING_WAITS)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
        mma_ss(bmm1_desc, sK[kv_state.idx].desc(), desc_Q, tmem_raw.subview(cutlass.Int32(S_ACC_OFF[0]) + s_state.idx * cutlass.Int32(N_Q)))
        elect_p = nvvm.elect_sync()
        bars.mb_s_full[s_state.idx].arrive(cta_group=1, pred=elect_p)
        bars.mb_kv_empty[kv_state.idx].arrive(cta_group=1, pred=elect_p)
        bars.mb_q_empty[q_state.idx].arrive(cta_group=1, pred=elect_p & (n_tiles == cutlass.Int32(1)))
        kv_state = advance(kv_state, STAGES_KV)
        s_state = advance(s_state, STAGES_S)
        s_empty_state = advance(s_empty_state, STAGES_S)

        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            # BMM1(i + 1) ahead of BMM2(i): the softmax of tile i overlaps the next score tile, and the K slot is released as
            # early as possible.
            if i + cutlass.Int32(1) < n_tiles:
                bars.mb_kv_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
                bars.mb_s_empty[s_empty_state.idx].wait(s_empty_state.phase, spin=SPIN_RING_WAITS)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                mma_ss(bmm1_desc, sK[kv_state.idx].desc(), desc_Q, tmem_raw.subview(cutlass.Int32(S_ACC_OFF[0]) + s_state.idx * cutlass.Int32(N_Q)))
                elect_p1 = nvvm.elect_sync()
                bars.mb_s_full[s_state.idx].arrive(cta_group=1, pred=elect_p1)
                bars.mb_kv_empty[kv_state.idx].arrive(cta_group=1, pred=elect_p1)
                bars.mb_q_empty[q_state.idx].arrive(cta_group=1, pred=elect_p1 & ((i + cutlass.Int32(2)) == n_tiles))
                kv_state = advance(kv_state, STAGES_KV)
                s_state = advance(s_state, STAGES_S)
                s_empty_state = advance(s_empty_state, STAGES_S)

            # BMM2(i): V(i) is the next load of the ring, P^T(i) the softmax's publish.
            bars.mb_kv_full[kv_state.idx].wait(kv_state.phase, spin=SPIN_RING_WAITS)
            bars.mb_p_full[p_state.idx].wait(p_state.phase, spin=SPIN_RING_WAITS)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            desc_P = sP[p_state.idx].desc()
            accum = cutlass.Boolean(i > cutlass.Int32(0))
            for blk in cutlass.range_constexpr(O_BLOCKS):
                mma_ss(
                    bmm2_desc,
                    sVt[kv_state.idx].shifted(blk * 2 * TILE_N * GRANU_ELEMS).desc(),
                    desc_P,
                    tmem_raw.subview(cutlass.Int32(O_OFF[blk])),
                    accumulate=accum,
                )
            elect_p2 = nvvm.elect_sync()
            bars.mb_bmm2_done[bmm2_state.idx].arrive(cta_group=1, pred=elect_p2)
            bars.mb_kv_empty[kv_state.idx].arrive(cta_group=1, pred=elect_p2)
            kv_state = advance(kv_state, STAGES_KV)
            p_state = advance(p_state, STAGES_S)
            bmm2_state = advance(bmm2_state, STAGES_S)

        q_state = advance(q_state, STAGES_Q)

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        t0, t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        tok, head, batch, item_row, _kvb, eff_seqlen_q, _ot, _ob, split = _decode_item(t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv)
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        sched_state = advance(sched_state, SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.wait(cutlass.Int32(0))
    tmem_dealloc(tmem_ptr_i32, TMEM_COLS, CTA_GROUP_KIND)


# === SOFTMAX warps (0 .. SOFTMAX_WARPS - 1): 128 key lanes per column group, COLS columns of the item's query heads per group =====


# The hot max / min sites of the softmax body are spelled ``ftz=True`` so they lower to FMNMX (fused FMNMX3 in the dependent
# chains); the bare ``cute.math.max`` default is ``arith.maxnumf`` = a compare + select pair per op (FSETP + FSEL), which was
# 54 % of the executed softmax instructions on this body's decode twin.  Numerics: ``ftz`` flushes sub-normal OPERANDS, so the
# two spellings can differ only where a column max is itself an fp32 sub-normal (|s| < 2^-126) or a signed zero -- O is
# unchanged there (the exp2 of a flushed sub-normal is 1.0 either way) and LSE moves by at most that sub-normal (<= 1.2e-38,
# and only with a single live key -- with more, ``m + log(l)`` rounds it away); no measured cell reached it, and O / LSE were
# BITWISE identical on every one (0 mismatching elements).  MEASURED against the compare-select
# spelling on a Rubin perf node (212 SMs, 2376 MHz lock, A/B/A x 3 rounds, control pairs <= 0.03 %): +13.4 / +13.4 / +13.5 %
# at S = 8K / 32K / 128K on recorded layer lists (3369 -> 2971, 3301 -> 2910, 3288 -> 2896 clk per K + V tile pair), the
# per-tile body -9.9 % (2894 -> 2607 clk steady state), the per-item fixed cost -21 %; O and LSE bitwise identical.
@cute.jit
def _warp_reduce_max(x):
    for off in cutlass.range_constexpr(5):
        x = cute.math.max(x, cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY)), ftz=True)
    return x


@cute.jit
def _warp_reduce_sum(x):
    for off in cutlass.range_constexpr(5):
        x = x + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, x, 16 >> off, 31, kind=nvvm.Shfl.BFLY))
    return x


@cute.jit
def _softmax_warp_group(
    tmem_ptr_i32,
    bars,
    sched,
    red_smem,
    sP_raw,
    sIds_raw,
    o_tensor,
    lse_tensor: Optional[cute.Tensor],
    block_lens_tensor: Optional[cute.Tensor],
    seq_kv_lens_tensor,
    n_batch,
    seqlen_q,
    seqlen_kv,
    t0_init,
    t1_init,
    scale_log2: cutlass.Float32,
    sQ_raw=None,  # the gated arm reads the item's gate out of its freed Q^T slot (appended; unused ungated)
    use_pdl: cutlass.Constexpr[bool] = False,
):
    nvvm.barrier_cta_sync(barrier_id=_BAR_TMEM, thread_count=BAR_TMEM_THREADS)
    tidx = cute.arch.thread_idx()[0]
    sm_warp = tidx // cutlass.Int32(32)  # 0 .. SOFTMAX_WARPS - 1 (all groups)
    # The column group (warp-uniform) and its first column; the key row of S^T / P^T = the d row of O^T is the lane WITHIN the group.
    # At ONE group the three are the constants 0 / 0 / tidx, so that rendering traces today's program.
    grp = (sm_warp // cutlass.Int32(SOFTMAX_WARPS_PER_GROUP)) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else cutlass.Int32(0)
    col0 = (grp * cutlass.Int32(COLS)) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else cutlass.Int32(0)
    lane = (tidx & cutlass.Int32(GROUP_LANES - 1)) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else tidx
    bar_softmax = (cutlass.Int32(_BAR_SOFTMAX) + grp) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else _BAR_SOFTMAX
    tmem_base = tmem_ptr_i32.load()
    red_ptr = Pointer(red_smem.data_ptr(), dtype=cutlass.Float32)

    NEG_INF = cutlass.Float32(float("-inf"))
    ZERO = cutlass.Float32(0.0)
    ONE = cutlass.Float32(1.0)
    LN2_F = cutlass.Float32(LN2)
    TINY = cutlass.Float32(1e-30)

    # Carried states (barrier rows 3, 7-10): t_state = the per-tile slot / parity of mb_s_full (wait), mb_s_empty and
    # mb_p_full (this group's arrives); bmm2_state = the mb_bmm2_done wait, one per tile, one iteration behind; ids_state =
    # the mb_ids_full wait and the mb_ids_empty arrive slot, once per item.  None is ever recomputed from a per-item index.
    t_state = PipelineState.start(phase=0)
    bmm2_state = PipelineState.start(phase=0)
    ids_state = PipelineState.start(phase=0)
    # Gated arm (barrier rows 11 / 12): gf_state = the mb_gate_full consumer state (wait-then-arrive, one stage = CFG.STAGES_GATE,
    # advanced per wait like every other ring: the k-th wait returns on gate(k)), qslot_state = the item's Q^T slot (i % 2, the same
    # item sequence the TMA and MMA warps walk) that holds its gate.
    # The gate's per-lane address parts: lane = d, so sub (the 64-wide subtile) and c (the column inside it) are item-invariant.
    gf_state = PipelineState.start(phase=0)
    qslot_state = PipelineState.start(phase=0)

    if cutlass.const_expr(use_pdl):
        # PDL (header, PDL SITES): the TMEM hand-off above is this role's prologue; its global reads (_item_bounds' lengths / counts)
        # and every global WRITE of the role (O / the partial slabs, the LSE) are below this wait.
        wait_on_dependent_grids()
    tok, head, batch, item_row, _kvb, eff_seqlen_q, out_tok, out_batch, split = _decode_item(t0_init, t1_init, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv)
    is_valid = cutlass.Int32(1)
    sched_state = PipelineState.start()
    while is_valid > cutlass.Int32(0):
        read_tile_id_arrive(sched.mb_read_tile_id.subview(sched_state.idx), 1)
        pos, count, n_tail, tail_lo, dead, n_tiles, eff_seqlen_kv, tile0 = _item_bounds(
            tok, batch, item_row, eff_seqlen_q, block_lens_tensor, seq_kv_lens_tensor, seqlen_kv, split
        )
        slot_base = ids_state.idx * cutlass.Int32(IDS_SLOT_WORDS)
        bars.mb_ids_full[ids_state.idx].wait(ids_state.phase)

        # Running column state: max in the scaled log2 domain (-inf = no live key yet), the thread's partial sum.
        m_vec = cutlass.Vector.from_elements(tuple(NEG_INF for _ in range(COLS)), cutlass.Float32)
        l_vec = cutlass.Vector.from_elements(tuple(ZERO for _ in range(COLS)), cutlass.Float32)

        for i in cutlass.range(0, n_tiles, 1, unroll=1):
            par = t_state.idx
            s_off = cutlass.Int32(S_ACC_OFF[0]) + par * cutlass.Int32(N_Q)
            if cutlass.const_expr(SOFTMAX_GROUPS > 1):
                s_off = s_off + col0  # the group's COLS columns of the slot

            bars.mb_s_full[par].wait(t_state.phase, spin=SPIN_RING_WAITS)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
            s_raw = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + s_off, cutlass.Float32), num=COLS)
            # The LOAD wait is what ORDERS the arrive after the read (frost-kernels.md section 3).
            nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.mb_s_empty[par].arrive()

            # Per-thread validity (header: the in-block mask depends on the key index alone -- all 16 columns share pos).
            # Lane l of tile i holds key 4 blk + l % 4 with blk = the list entry 32 (tile0 + i) + l // 4 (tail / -1 by the SELECT).
            t_abs = i
            if cutlass.const_expr(SPLIT_KV > 1):
                t_abs = tile0 + i
            idx = t_abs * cutlass.Int32(BLOCKS_PER_TILE) + (lane >> cutlass.Int32(2))
            ld_idx = cute.math.min(idx, cutlass.Int32(BLOCK_TOPK - 1))
            word = cutlass.Int32(sIds_raw.load(slot_base + ld_idx))
            blk = _block_id_at(idx, word, count, n_tail, tail_lo, dead)
            key_abs = blk * cutlass.Int32(BLOCK_SIZE) + (lane & cutlass.Int32(BLOCK_SIZE - 1))
            valid = (blk >= cutlass.Int32(0)) & (key_abs <= pos) & (key_abs < eff_seqlen_kv)
            s_cols = []
            for j in cutlass.range_constexpr(COLS):
                s_cols.append(_select_f32(valid, cutlass.Float32(s_raw[j]) * scale_log2, NEG_INF))

            # Column max over the tile's 128 keys: butterfly inside the warp, then the group's four warps exchange through
            # SMEM (slot = the tile's ring parity, so the single barrier per tile also orders the next reuse of the slot;
            # the slot is [SOFTMAX_WARPS][COLS] and group g reads warps 4 g .. 4 g + 3 of it after ITS named barrier).
            red_base = par * cutlass.Int32(SOFTMAX_WARPS * COLS) + sm_warp * cutlass.Int32(COLS)
            col_max = []
            for j in cutlass.range_constexpr(COLS):
                cm = _warp_reduce_max(s_cols[j])
                (red_ptr + (red_base + cutlass.Int32(j))).store(cm)
                col_max.append(cm)
            nvvm.barrier_cta_sync(barrier_id=bar_softmax, thread_count=BAR_SOFTMAX_THREADS)
            red_read = par * cutlass.Int32(SOFTMAX_WARPS * COLS)
            if cutlass.const_expr(SOFTMAX_GROUPS > 1):
                red_read = red_read + grp * cutlass.Int32(SOFTMAX_WARPS_PER_GROUP * COLS)
            tile_max = []
            for j in cutlass.range_constexpr(COLS):
                tm = col_max[j]
                for w in cutlass.range_constexpr(SOFTMAX_WARPS_PER_GROUP):
                    tm = cute.math.max(tm, cutlass.Float32((red_ptr + (red_read + cutlass.Int32(w * COLS + j))).load()), ftz=True)
                tile_max.append(tm)

            # Online update per column (identical in every lane: alpha is uniform, so O^T's rescale is exact).
            m_new = []
            l_new = []
            alpha = []
            all_one = None
            p_cols = []
            for j in cutlass.range_constexpr(COLS):
                m_old_j = cutlass.Float32(m_vec[j])
                m_new_j = cute.math.max(m_old_j, tile_max[j], ftz=True)
                ms_old = row_max_for_exp2(m_old_j)
                ms_new = row_max_for_exp2(m_new_j)
                alpha_j = cute.math.exp2(cute.math.min(ms_old - ms_new, ZERO, ftz=True), fastmath=True)
                p_j = cute.math.exp2(s_cols[j] - ms_new, fastmath=True)
                l_new.append(cutlass.Float32(l_vec[j]) * alpha_j + p_j)
                m_new.append(m_new_j)
                alpha.append(alpha_j)
                p_cols.append(p_j)
                is_one = alpha_j == ONE
                all_one = is_one if all_one is None else (all_one & is_one)
            m_vec = cutlass.Vector.from_elements(tuple(m_new), cutlass.Float32)
            l_vec = cutlass.Vector.from_elements(tuple(l_new), cutlass.Float32)

            # P^T(t) -> sP[par] as packed half pairs, row = the thread's key, swizzled the way the BMM2 B descriptor decodes
            # it.  WAR on the slot: BMM2(t-2) last read it, and its completion was waited at iteration t-1 (below), BEFORE
            # this store -- the store-then-wait order inside an iteration is exactly sufficient (header row 10).
            p_words = []
            for w in cutlass.range_constexpr(P_GROUP_WORDS):
                p_words.append(fp32_to_fp16(p_cols[2 * w], p_cols[2 * w + 1], dtype=STORAGE_DTYPE))
            p_row_base = par * cutlass.Int32(pBufferWords) + lane * cutlass.Int32(P_ROW_WORDS)
            if cutlass.const_expr(SOFTMAX_GROUPS > 1):
                # group g writes word offset 4 g of the lane's 32-B row = the chunk c = g of the one-group loop, under the same swizzle
                p_row_base = p_row_base + grp * cutlass.Int32(P_GROUP_WORDS)
            for c in cutlass.range_constexpr(P_CHUNKS):
                chunk = cutlass.Vector.from_elements(tuple(p_words[4 * c + k] for k in range(4)), cutlass.Int32)
                p_ptr = Pointer(sP_raw.subview(p_row_base + cutlass.Int32(4 * c)).data_ptr(), dtype=cutlass.Int32)
                p_ptr.store_swizzled(chunk, _P_SMEM_SWIZZLE, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")

            # O^T(t-1) is complete once BMM2(t-1) commits; rescale this group's columns by alpha before BMM2(t) accumulates
            # on top.  Skipped when no column's max moved.
            if i > cutlass.Int32(0):
                bars.mb_bmm2_done[bmm2_state.idx].wait(bmm2_state.phase, spin=SPIN_RING_WAITS)
                bmm2_state = advance(bmm2_state, STAGES_S)
                nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)
                if ~all_one:
                    for blk_o in cutlass.range_constexpr(O_BLOCKS):
                        o_col = cutlass.Int32(O_OFF[blk_o])
                        if cutlass.const_expr(SOFTMAX_GROUPS > 1):
                            o_col = o_col + col0
                        o_ptr = nvvm.make_tmem_ptr(tmem_base + o_col, cutlass.Float32)
                        o_vals = nvvm.tcgen05_ld("32x32b", o_ptr, num=COLS)
                        nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                        o_scaled = cutlass.Vector.from_elements(tuple(cutlass.Float32(o_vals[j]) * alpha[j] for j in range(COLS)), cutlass.Float32)
                        nvvm.tcgen05_st("32x32b", o_ptr, o_scaled)
                    nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.STORE)
            # Publish P^T(t) (and the rescaled O^T) to the MMA warp.
            nvvm.tcgen05_fence(nvvm.Tcgen05Fence.BEFORE_THREAD_SYNC)
            bars.mb_p_full[par].arrive()
            t_state = advance(t_state, STAGES_S)

        # The last read of this item's list slot (row 4: ONE lane per consuming warp).
        if nvvm.elect_sync():
            bars.mb_ids_empty[ids_state.idx].arrive()
        ids_state = advance(ids_state, STAGES_IDS)

        # --- epilogue: the item's last BMM2 must land before O^T is read (one wait per tile: this is the last tile's).
        bars.mb_bmm2_done[bmm2_state.idx].wait(bmm2_state.phase)
        bmm2_state = advance(bmm2_state, STAGES_S)
        nvvm.tcgen05_fence(nvvm.Tcgen05Fence.AFTER_THREAD_SYNC)

        # Column sums over the 128 lanes (once per item), via the epilogue slot.
        red_epi = cutlass.Int32(STAGES_S * SOFTMAX_WARPS * COLS) + sm_warp * cutlass.Int32(COLS)
        for j in cutlass.range_constexpr(COLS):
            (red_ptr + (red_epi + cutlass.Int32(j))).store(_warp_reduce_sum(cutlass.Float32(l_vec[j])))
        nvvm.barrier_cta_sync(barrier_id=bar_softmax, thread_count=BAR_SOFTMAX_THREADS)
        red_epi_read = cutlass.Int32(STAGES_S * SOFTMAX_WARPS * COLS)
        if cutlass.const_expr(SOFTMAX_GROUPS > 1):
            red_epi_read = red_epi_read + grp * cutlass.Int32(SOFTMAX_WARPS_PER_GROUP * COLS)

        lse_cols = []
        inv_cols = []
        dead_cols = []
        for j in cutlass.range_constexpr(COLS):
            l_tot = ZERO
            for w in cutlass.range_constexpr(SOFTMAX_WARPS_PER_GROUP):
                l_tot = l_tot + cutlass.Float32((red_ptr + (red_epi_read + cutlass.Int32(w * COLS + j))).load())
            m_raw = cutlass.Float32(m_vec[j])  # -inf while the column has no live key
            m_nat = row_max_for_exp2(m_raw) * LN2_F
            row_dead = l_tot <= ZERO
            # The denominator floor sits inside the log and the reciprocal ONLY and is always followed by the row_dead
            # SELECT, so neither LSE nor O ever carries it (sdpa-invariants.md sections 2-4): keep these lines together.
            lse_j = m_nat + cute.math.log(cute.math.max(l_tot, TINY), fastmath=True)
            inv_j = ONE / cute.math.max(l_tot, TINY)
            lse_j = _select_f32(row_dead, NEG_INF, lse_j)
            inv_j = _select_f32(row_dead, ZERO, inv_j)
            if cutlass.const_expr(CFG.EPILOGUE_GATE):
                # Gated arm: sigmoid's 1/2 folded into the scale the epilogue applies anyway (exact, a power of two), so
                # o * inv_j below IS the h of h * tanh(g / 2) + h (gate_epilogue_pairs).  The dead-row SELECT still runs per
                # element AFTER the gate fma -- never inv = 0 times a residue that may be NaN (sdpa-invariants section 2).
                inv_j = gate_inv_sum(inv_j)
            if cutlass.const_expr(CFG.STATS_LOG2):
                lse_j = lse_j * cutlass.Float32(LOG2E)
            lse_cols.append(lse_j)
            inv_cols.append(inv_j)
            dead_cols.append(row_dead)

        # Stores: column j = query head (head * G + j) of the item's token at its output coordinate (out_tok, out_batch) --
        # the dense (tok, batch) or the packed (tok_packed, 0) of _decode_item; the zero tail columns j >= G are never stored,
        # and a padded Q row (the Q ROW past the sequence's Q length) stores nothing.  SPLIT_KV > 1: the tensors are the fp32
        # partial slabs and the batch coordinate is split-major, b + s x B (the combine's convention).
        live_row = pos < eff_seqlen_q
        if cutlass.const_expr(BOTTOM_RIGHT == 1):
            live_row = tok < eff_seqlen_q  # the row's liveness is the Q row's, not its key-space position's
        part_batch = out_batch
        if cutlass.const_expr(SPLIT_KV > 1):
            part_batch = out_batch + split * n_batch
        head_base = head * cutlass.Int32(G)
        # Two groups: column j of group g is head col0 + j; the columns at or past G (the zero tail) are never stored.
        n_live_cols = (cutlass.Int32(Q_BOX_ROWS) - col0) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else cutlass.Int32(Q_BOX_ROWS)
        if cutlass.const_expr(lse_tensor is not None):
            lse_arr = cutlass.make_array_view(lse_tensor)
            for j in cutlass.range_constexpr(COLS):
                if cutlass.const_expr(SOFTMAX_GROUPS > 1 or j < Q_BOX_ROWS):
                    col = (col0 + cutlass.Int32(j)) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else cutlass.Int32(j)
                    ok = ((lane == col) & live_row & (cutlass.Int32(j) < n_live_cols)) if cutlass.const_expr(SOFTMAX_GROUPS > 1) else ((lane == col) & live_row)
                    if ok:
                        lse_arr[part_batch, head_base + col, out_tok] = lse_cols[j]

        # --- the fused epilogue gate (CFG.EPILOGUE_GATE; barrier rows 11 / 12): ONE wait per item on the gate the TMA warp
        # staged into this item's freed Q^T slot after its last BMM1 (it has had the last tile's softmax, BMM2, the column
        # sums and the LSE to land), the opaque 0.5 hoisted once per item (frost-tile-dsl section 7), the slot = i % 2.
        gate_slot_elems = cutlass.Int32(0)
        half_opaque = ZERO
        if cutlass.const_expr(CFG.EPILOGUE_GATE):
            half_opaque = gate_half_opaque()
            bars.mb_gate_full.wait(gf_state.phase)
            gf_state = advance(gf_state, CFG.STAGES_GATE)
            gate_slot_elems = qslot_state.idx * cutlass.Int32(qBufferElems)
            qslot_state = advance(qslot_state, STAGES_Q)

        # O[q, d] = O^T[d, q] / l[q]: the thread index is d (and d + 128).
        oo = cutlass.make_array_view(o_tensor)
        D_V = cutlass.const_expr(o_tensor.shape[3])
        for blk_o in cutlass.range_constexpr(O_BLOCKS):
            if cutlass.const_expr(blk_o * 128 < D_V):
                d_idx = lane + cutlass.Int32(blk_o * 128)
                o_col_e = cutlass.Int32(O_OFF[blk_o])
                if cutlass.const_expr(SOFTMAX_GROUPS > 1):
                    o_col_e = o_col_e + col0
                o_vals = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(tmem_base + o_col_e, cutlass.Float32), num=COLS)
                nvvm.tcgen05_wait(kind=nvvm.Tcgen05Wait.LOAD)
                if cutlass.const_expr(CFG.EPILOGUE_GATE):
                    # The thread's gate column (lane = d) over the item's Q_BOX_ROWS head rows, read out of the SW128 slot at the
                    # offsets the TMA wrote (the header's SMEM table): subtile = d // 64, the 16-B chunk index of the 2-B column
                    # inside the 128-B row XORed with (row & 7) -- the same pattern the Q^T descriptor decodes.  32 lanes read 32
                    # adjacent columns of ONE row: four 16-B chunks, two lanes per bank word -> one wavefront, no conflict.
                    sub = d_idx >> cutlass.Int32(_GATE_ROW_SHIFT)
                    c = d_idx & cutlass.Int32(GRANU_ELEMS - 1)
                    lane_part = gate_slot_elems + sub * cutlass.Int32(N_Q * GRANU_ELEMS) + (c & cutlass.Int32(_GATE_CHUNK_ELEMS - 1))
                    if cutlass.const_expr(SOFTMAX_GROUPS > 1):
                        # the group's rows start at head row col0 (a multiple of 8, so (col0 + j) & 7 == j: the XOR term is j's)
                        lane_part = lane_part + col0 * cutlass.Int32(GRANU_ELEMS)
                    chunk = c >> cutlass.Int32(_GATE_CHUNK_SHIFT)
                    g_vals = []
                    h_vals = []
                    for j in cutlass.range_constexpr(N_GATE_ROWS):
                        off_j = (
                            lane_part + cutlass.Int32(j * GRANU_ELEMS) + ((chunk ^ cutlass.Int32(j & _GATE_SWZ_ROW_MASK)) << cutlass.Int32(_GATE_CHUNK_SHIFT))
                        )
                        g_vals.append(STORAGE_DTYPE(sQ_raw.load(off_j)).to(cutlass.Float32))  # one LDS.U16, widened
                        h_vals.append(cutlass.Float32(o_vals[j]) * inv_cols[j])  # h: inv_cols carries sigmoid's 1/2
                    if cutlass.const_expr(N_GATE_ROWS % 2 == 1):
                        # gate_epilogue_pairs walks PAIRS of head rows: an odd G repeats its last element (the same packed
                        # instructions as the even case; the duplicate result is dropped below).
                        g_vals.append(g_vals[N_GATE_ROWS - 1])
                        h_vals.append(h_vals[N_GATE_ROWS - 1])
                    gated = gate_epilogue_pairs(h_vals, g_vals, half_opaque, len(h_vals))
                    if (d_idx < cutlass.Int32(D_V)) & live_row:
                        for j in cutlass.range_constexpr(N_GATE_ROWS):
                            # The dead-column SELECT per element, AFTER the gate fma (sdpa-invariants section 2).
                            val = _select_f32(dead_cols[j], ZERO, gated[j])
                            if cutlass.const_expr(SOFTMAX_GROUPS > 1):
                                if cutlass.Int32(j) < n_live_cols:  # the group's columns at or past G are the zero tail: never stored
                                    oo[
                                        cutlass.Int64(part_batch),
                                        cutlass.Int64(out_tok),
                                        cutlass.Int64(head_base + col0 + cutlass.Int32(j)),
                                        cutlass.Int64(d_idx),
                                    ] = val.to(o_tensor.element_type)
                            else:
                                oo[cutlass.Int64(part_batch), cutlass.Int64(out_tok), cutlass.Int64(head_base + cutlass.Int32(j)), cutlass.Int64(d_idx)] = (
                                    val.to(o_tensor.element_type)
                                )
                else:
                    if (d_idx < cutlass.Int32(D_V)) & live_row:
                        for j in cutlass.range_constexpr(COLS):
                            if cutlass.const_expr(SOFTMAX_GROUPS > 1 or j < Q_BOX_ROWS):
                                val = _select_f32(dead_cols[j], ZERO, cutlass.Float32(o_vals[j]) * inv_cols[j])
                                if cutlass.const_expr(SOFTMAX_GROUPS > 1):
                                    if cutlass.Int32(j) < n_live_cols:  # the group's columns at or past G are the zero tail: never stored
                                        oo[
                                            cutlass.Int64(part_batch),
                                            cutlass.Int64(out_tok),
                                            cutlass.Int64(head_base + col0 + cutlass.Int32(j)),
                                            cutlass.Int64(d_idx),
                                        ] = val.to(o_tensor.element_type)
                                else:
                                    oo[cutlass.Int64(part_batch), cutlass.Int64(out_tok), cutlass.Int64(head_base + cutlass.Int32(j)), cutlass.Int64(d_idx)] = (
                                        val.to(o_tensor.element_type)
                                    )
        if cutlass.const_expr(CFG.EPILOGUE_GATE):
            # Release the gate's slot to the TMA warp (row 12: a BARE arrive from every softmax lane after the item's last gate
            # LDS; generic read -> arrive -> producer wait -> TMA write needs no proxy fence).
            bars.mb_gate_empty.arrive()

        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        wait(sched.mb_scheduler.subview(sched_state.idx), sched_state.phase)
        t0, t1, nxt_v = read_clc_payload(sched, sched_state.idx * cutlass.Int32(8))
        tok, head, batch, item_row, _kvb, eff_seqlen_q, out_tok, out_batch, split = _decode_item(t0, t1, seq_kv_lens_tensor, n_batch, seqlen_q, seqlen_kv)
        is_valid = cute.arch.make_warp_uniform(nxt_v)
        sched_state = advance(sched_state, SCHEDULER_STAGES)

    bars.mb_tmem_dealloc.arrive()


# === the kernel ===========================================================================================================


@cute.kernel
def _kernel(
    tma_q_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_k_desc: cutlass.GridConstant[tmap.TensorMap],
    tma_v_desc: cutlass.GridConstant[tmap.TensorMap],
    o_tensor: cute.Tensor,
    lse_tensor: Optional[cute.Tensor],
    block_ids_tensor: cute.Tensor,
    block_lens_tensor: Optional[cute.Tensor],
    seq_kv_lens_tensor: cute.Tensor,
    seqlen_q: cutlass.Int32,
    seqlen_kv: cutlass.Int32,
    k_head_stride: cutlass.Int32,
    v_head_stride: cutlass.Int32,
    scale_softmax_log2: cutlass.Float32,
    block_table_tensor: Optional[cute.Tensor],
    max_pages: cutlass.Int32,
    k_rows_per_page: cutlass.Int32,
    k_rows_per_head: cutlass.Int32,
    v_rows_per_page: cutlass.Int32,
    v_rows_per_head: cutlass.Int32,
    n_batch: cutlass.Int32,
    # The gate's LOAD descriptor (EPILOGUE_GATE): Q's 4-D box form over the gate tensor.  Appended LAST so the ungated
    # rendering's parameter offsets are untouched; ungated it aliases tma_q_desc (a GridConstant tensor map is always a valid
    # one) and is never read -- CFG.EPILOGUE_GATE stays the single fold flag.
    tma_gate_desc: cutlass.GridConstant[tmap.TensorMap] = None,
    # Programmatic dependent launch (the launch attribute is set by the host entry under the same flag; header, PDL SITES): every role
    # waits on the dependent grid before its first global read of a predecessor's output; no explicit release -- the CTA exit triggers
    # the dependent grid at this grid's completion (the measured placement).  Appended LAST; the False rendering traces the program
    # above unchanged.
    use_pdl: cutlass.Constexpr[bool] = False,
) -> None:
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tidx, _, _ = cute.arch.thread_idx()
    # BSHD: the grid IS the work list -- x = query token, y = KV head, z = batch; a CTA's first item is its own position, later
    # ones come from the CLC scheduler's cancelled CTAs.  THD: the grid is occupancy-sized (x = the first unit id, y = z = 0)
    # and later units come from the device claim counter.  Both forms hand _decode_item the same two payload words.
    bidx = cute.arch.block_idx()[0]
    bidy = cute.arch.block_idx()[1]
    bidz = cute.arch.block_idx()[2]
    t0_init = bidx
    t1_init = bidy | (bidz << cutlass.Int32(16))
    if cutlass.const_expr(CFG.THD_VARLEN == 1):
        # A CTA whose first unit id is at or past the live total (meta[4B + 2], the setup launch's device value) has no work:
        # every thread exits HERE, before any mbarrier init, TMEM allocation or scheduler credit (the occupancy-sized grid may
        # exceed the packed work; the counter then hands the resident CTAs nothing past the total either).
        exit_if_dead_thd_cluster(seq_kv_lens_tensor, n_batch, 1)

    # SMEM in the header table's order: the descriptor-read operands first (1024-B aligned), the register-addressed buffers last.
    sQ_raw = cutlass.Array(STORAGE_DTYPE, STAGES_Q * qBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sP_raw = cutlass.Array(cutlass.Int32, STAGES_S * pBufferWords, alignment=1024, space=cutlass.AddressSpace.smem)
    sKV_raw = cutlass.Array(STORAGE_DTYPE, STAGES_KV * kvBufferElems, alignment=1024, space=cutlass.AddressSpace.smem)
    sIds_raw = cutlass.Array(cutlass.Int32, STAGES_IDS * IDS_SLOT_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    red_smem = cutlass.Array(cutlass.Float32, RED_WORDS, alignment=16, space=cutlass.AddressSpace.smem)
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, alignment=16, space=cutlass.AddressSpace.smem)

    # Q^T: the K-major B operand of BMM1 (N_Q rows x TILE_K per slot), one TMA sub-tile (N_Q rows x 128 B) per 64-wide granule.
    sQ = SmemTile(
        base=sQ_raw,
        elems_per_stage=qBufferElems,
        stages=STAGES_Q,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        tma_loads_per_tile=QK_ITERS,
        tma_granu_elems=GRANU_ELEMS,
        tma_subtile_stride_elems=N_Q * GRANU_ELEMS,
        desc_version=DESC_VERSION,
    )
    # The gathered K/V ring, seen as the K-major A operand of BMM1 (TILE_N rows x TILE_K) ...
    sK = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES_KV,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_QK,
        layout=SMEM_LAYOUT_QK,
        desc_version=DESC_VERSION,
    )
    # ... and as the MN-major A operand of BMM2 (V^T: d contiguous, LBO steps the 64-wide d atoms).  Same bytes, two
    # descriptors -- the gather map writes under the same SW128 the descriptors decode.
    sVt = SmemTile(
        base=sKV_raw,
        elems_per_stage=kvBufferElems,
        stages=STAGES_KV,
        leading_byte_offset=LEADING_BYTE_OFFSET_VT,
        stride_byte_offset=STRIDE_BYTE_OFFSET_VT,
        layout=SMEM_LAYOUT_QK,
        desc_version=DESC_VERSION,
    )
    # P^T: the MN-major B operand of BMM2, STAGES_S buffers of TILE_N key rows.
    sP = SmemTile(
        base=sP_raw,
        elems_per_stage=pBufferWords,
        stages=STAGES_S,
        leading_byte_offset=0,
        stride_byte_offset=STRIDE_BYTE_OFFSET_P,
        layout=SMEM_LAYOUT_P,
        desc_version=DESC_VERSION,
    )

    bars = make_sparse_bars()
    sched = Sched(
        **{
            "mb_scheduler": cutlass.Array(cutlass.Int64, SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "mb_read_tile_id": cutlass.Array(cutlass.Int64, SCHEDULER_STAGES, alignment=16, space=cutlass.AddressSpace.smem),
            "tile_id_smem": cutlass.Array(cutlass.Int32, SCHEDULER_STAGES * 8, alignment=16, space=cutlass.AddressSpace.smem),
            "bidx_init": bidx,
            "bidy_init": bidy,
            "bidz_init": bidz,
        }
    )

    # P4: ONE warp, ONE lane initialises every stage of every ring (.init() is per stage); fence + CTA sync outside.
    if warp_idx == 0:
        if nvvm.elect_sync():
            for s in cutlass.range_constexpr(STAGES_Q):
                bars.mb_q_full[s].init()
                bars.mb_q_empty[s].init()
            for s in cutlass.range_constexpr(STAGES_IDS):
                bars.mb_ids_full[s].init()
                bars.mb_ids_empty[s].init()
            for s in cutlass.range_constexpr(STAGES_KV):
                bars.mb_kv_full[s].init()
                bars.mb_kv_empty[s].init()
            for p in cutlass.range_constexpr(STAGES_S):
                bars.mb_s_full[p].init()
                bars.mb_s_empty[p].init()
                bars.mb_p_full[p].init()
                bars.mb_bmm2_done[p].init()
            bars.mb_tmem_dealloc.init()
            if cutlass.const_expr(CFG.EPILOGUE_GATE):
                bars.mb_gate_full.init()  # rows 11 / 12: one stage each
                bars.mb_gate_empty.init()
            for s in cutlass.range_constexpr(SCHEDULER_STAGES):
                nvvm.mbarrier_init(sched.mb_scheduler.subview(s), CFG.ONE_LANE)
                nvvm.mbarrier_init(sched.mb_read_tile_id.subview(s), CFG.READ_TILE_ARRIVERS)

    if cutlass.const_expr(Q_BOX_ROWS < N_Q):
        # Zero the Q^T tail rows the TMA box never covers (rows G..15 of BOTH slots) once per CTA, so their never-stored
        # columns stay finite; the generic stores are published to the async proxy before any MMA reads the slots.
        if warp_idx < SOFTMAX_WARPS:
            zero4 = cutlass.Vector.from_elements(tuple(cutlass.Int32(0) for _ in range(4)), cutlass.Int32)
            tail_chunks = (N_Q - Q_BOX_ROWS) * 8  # 16 B chunks per sub-tile
            n_chunks = STAGES_Q * QK_ITERS * tail_chunks
            for it in cutlass.range_constexpr((n_chunks + SOFTMAX_LANES - 1) // SOFTMAX_LANES):
                chunk = cutlass.Int32(it * SOFTMAX_LANES) + tidx
                if chunk < cutlass.Int32(n_chunks):
                    slot = chunk // cutlass.Int32(QK_ITERS * tail_chunks)
                    rem = chunk % cutlass.Int32(QK_ITERS * tail_chunks)
                    sub = rem // cutlass.Int32(tail_chunks)
                    within = rem % cutlass.Int32(tail_chunks)
                    elem_off = (
                        slot * cutlass.Int32(qBufferElems)
                        + sub * cutlass.Int32(N_Q * GRANU_ELEMS)
                        + cutlass.Int32(Q_BOX_ROWS * GRANU_ELEMS)
                        + within * cutlass.Int32(8)
                    )
                    Pointer(sQ_raw.subview(elem_off).data_ptr(), dtype=cutlass.Int32).store(zero4, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()
    # No CTA-wide barrier-0 sync may follow this one: the spare warp exits here.
    # PDL (header, PDL SITES): NO release here, or anywhere -- the CTA exit is the trigger of the dependent grid (a release at this
    # point made the split combine's CTAs resident through the core's whole run and cost 3.2-4.4 us per step at B <= 4 on the cc 10.7
    # class, the measured placement); each role's own wait follows in its body.

    if warp_idx < SOFTMAX_WARPS:
        _softmax_warp_group(
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            red_smem=red_smem,
            sP_raw=sP_raw,
            sIds_raw=sIds_raw,
            o_tensor=o_tensor,
            lse_tensor=lse_tensor,
            block_lens_tensor=block_lens_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_batch=n_batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            t0_init=t0_init,
            t1_init=t1_init,
            scale_log2=scale_softmax_log2,
            sQ_raw=sQ_raw,
            use_pdl=use_pdl,
        )
    elif warp_idx < GATHER_WARP_BASE + GATHER_WARPS:
        _gather_warp_group(
            tma_k_desc=tma_k_desc,
            tma_v_desc=tma_v_desc,
            sKV_raw=sKV_raw,
            sIds_raw=sIds_raw,
            bars=bars,
            sched=sched,
            block_lens_tensor=block_lens_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            k_head_stride=k_head_stride,
            v_head_stride=v_head_stride,
            w=warp_idx - cutlass.Int32(GATHER_WARP_BASE),
            n_batch=n_batch,
            t0_init=t0_init,
            t1_init=t1_init,
            block_table_tensor=block_table_tensor,
            max_pages=max_pages,
            k_rows_per_page=k_rows_per_page,
            k_rows_per_head=k_rows_per_head,
            v_rows_per_page=v_rows_per_page,
            v_rows_per_head=v_rows_per_head,
            use_pdl=use_pdl,
        )
    elif warp_idx == MMA_WARP_ID:
        _mma_warp_group(
            sQ=sQ,
            sK=sK,
            sVt=sVt,
            sP=sP,
            tmem_ptr_i32=tmem_ptr_i32,
            bars=bars,
            sched=sched,
            block_lens_tensor=block_lens_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_batch=n_batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            t0_init=t0_init,
            t1_init=t1_init,
            use_pdl=use_pdl,
        )
    elif warp_idx == TMALDG_WARP_ID:
        nvvm.prefetch_tensormap(tma_q_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_k_desc.get_ptr())
        nvvm.prefetch_tensormap(tma_v_desc.get_ptr())
        if cutlass.const_expr(CFG.EPILOGUE_GATE):
            nvvm.prefetch_tensormap(tma_gate_desc.get_ptr())
        _tmaldg_warp_group(
            tma_q_desc=tma_q_desc,
            tma_gate_desc=tma_gate_desc,
            sQ=sQ,
            sIds_raw=sIds_raw,
            bars=bars,
            sched=sched,
            block_ids_tensor=block_ids_tensor,
            seq_kv_lens_tensor=seq_kv_lens_tensor,
            n_batch=n_batch,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            t0_init=t0_init,
            t1_init=t1_init,
            use_pdl=use_pdl,
        )
    elif warp_idx == SCHED_WARP_ID:
        cta_id_x = cutlass.Int32(0)
        is_cga_first_cta = cta_id_x == cutlass.Int32(0)
        if cutlass.const_expr(CFG.THD_VARLEN == 1):
            # THD: the persistent form -- units claimed from the device counter at meta[4B + 3] against the live total at
            # meta[4B + 2] (both written by the setup launch); the payload is (uid, 0, valid, 0) published through local SMEM
            # and one elected arrive (barrier row 13, init ONE_LANE in both forms).
            scheduler_warp_loop_persistent(sched, SCHEDULER_STAGES, is_cga_first_cta, seq_kv_lens_tensor, THD_CTR_OFF(n_batch), THD_LIVE_OFF(n_batch), 1, 1)
        else:
            scheduler_warp_loop(sched, SCHEDULER_STAGES, is_cga_first_cta, 1)
    # else: the spare warp (CFG.SPARE_WARP_ID: 19 at two softmax column groups, 15 at one) exits.


_kernel.set_name_prefix("cudnn", remove_cutlass_symbol=True)


# === host =================================================================================================================


@cute.jit
def _host(
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    o_ptr: cute.Pointer,
    lse_ptr: Optional[cute.Pointer],
    block_ids_ptr: cute.Pointer,
    block_lens_ptr: Optional[cute.Pointer],
    seq_kv_lens_ptr: cute.Pointer,
    problem_size: Tuple[int, int, int, int, int],
    q_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    k_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    v_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    o_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    lse_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64],
    scale_softmax_log2: cutlass.Float32,
    block_table_ptr: Optional[cute.Pointer],
    paged_geom: Tuple[int, int, int, int, int, int],
    thd_q_lens_ptr: Optional[cute.Pointer] = None,
    thd_kv_lens_ptr: Optional[cute.Pointer] = None,
    thd_lens_form: Optional[cutlass.Int32] = None,
    n_ctas: Optional[cutlass.Int32] = None,
    gate_ptr: Optional[cute.Pointer] = None,
    gate_strides: Tuple[cutlass.Int64, cutlass.Int64, cutlass.Int64] = (cutlass.Int64(0), cutlass.Int64(0), cutlass.Int64(0)),
    use_pdl: cutlass.Constexpr[bool] = False,
    stream: _cuda_driver.CUstream = None,
) -> None:
    """Bind the explicit pointer ABI and launch the sparse core.

    ``problem_size`` = (B, QH, KH, SQ, SKV); strides are the caller's (batch, seq, head) ELEMENT strides (Int64 leaves);
    ``block_ids_ptr`` -> int32 [B x SQ, BLOCK_TOPK] contiguous, ``block_lens_ptr`` -> int32 [B x SQ] or None,
    ``seq_kv_lens_ptr`` -> int32 [B] (read only under SEQ_KV_LENS_PRESENT).  The K/V gather maps are 2-D (tokens of EVERY
    batch as rows, the token's full row as columns), so the batch stride must be SKV x the token stride (or B == 1) -- the
    adapter validates that contract; this entry assumes validated operands.  ``gate_ptr`` (appended) -> the fused epilogue
    gate G in O's (B, SQ, QH, D) shape at its own (batch, seq, head) strides (a slab column slice binds with no copy);
    present iff CFG.EPILOGUE_GATE (a module specialization), None otherwise -- a mismatch is refused at trace time.  The
    gate takes O's form on every arm: the packed ``[1, T_q, QH, D]`` operand (batch extent 1) under THD, the dense one under
    the paged read (the gate is never paged; it composes with both arms -- it touches the Q^T slot ring and the epilogue only).

    PAGED arm (CFG.PAGED_KV): ``k_strides`` / ``v_strides`` = (page stride, TOKEN stride, COLUMN head stride -- 0 for an HND
    pool whose head folds into the row), ``block_table_ptr`` -> int32 (B, max_pages) contiguous, ``paged_geom`` =
    (num_pages, max_pages, k rows_per_page, k rows_per_head, v rows_per_page, v rows_per_head); the gather maps' rows are
    the pool's token rows (num_pages x rows_per_page); SKV is the table's capacity (max_pages x PAGE_SIZE), the visible range
    comes from ``seq_kv_lens``.  On the dense arm ``block_table_ptr`` is None and ``paged_geom`` is unread.

    DECODE FORM.  ``CFG.BOTTOM_RIGHT``: no ABI change (the position offset is derived on device from the lengths).
    ``CFG.LIST_PER_SEQUENCE``: ``block_ids_ptr`` -> int32 [B, BLOCK_TOPK] contiguous and ``block_lens_ptr`` -> int32 [B] (one row per
    SEQUENCE, read by every item of it).  ``CFG.SPLIT_KV = S > 1``: ``o_ptr`` -> the fp32 O PARTIAL slab (B x S, SQ, QH, D) and
    ``lse_ptr`` -> the fp32 LSE partial slab (B x S, QH, SQ), both split-major on the batch axis (b + s x B) at the strides given
    (``o_strides`` / ``lse_strides`` are the SLABS' strides -- the adapter passes the compact ones -- and ``lse_ptr`` is REQUIRED:
    the per-split LSE drives the combine); the grid is (SQ x S, KH, B) and ``sm100/split_combine.py`` reduces the slabs into the
    caller's O / LSE afterwards (the adapter's second launch; the gate, when given, rides it).

    PDL (``use_pdl``, a compile-time constant of this entry -- the adapter's ``compile(use_pdl=)`` key): the main kernel launches with
    the programmatic-stream-serialization attribute and carries the griddepcontrol sites (header, PDL SITES); the THD setup launch never
    takes the attribute (THD x ``use_pdl`` is the adapter's typed decline: the main kernel reads the setup launch's metadata above every
    role wait).  The same math, bitwise.

    THD (CFG.THD_VARLEN): ``B`` is the number of SEQUENCES and ``SQ`` / ``SKV`` the packed CAPACITIES of the ``[1, T, H, D]``
    operands (batch extent 1; the live totals are device values); ``seq_kv_lens_ptr`` -> the int32 metadata workspace of
    ``THD_META_WORDS(B)`` words that the setup launch fills from ``thd_q_lens_ptr`` / ``thd_kv_lens_ptr`` (per-sequence lengths
    ``[B]``, or cumulative ``[B + 1]`` per ``thd_lens_form`` bit 0 / 1): ``[seq_kv_lens(B) | cu_q(B + 1) | cu_k(B + 1) | remap(B) |
    live | ctr]``; ``block_ids_ptr`` -> int32 [SQ, BLOCK_TOPK] (one row per packed token); ``n_ctas`` the occupancy-sized grid
    (one CTA per SM, at most one per unit) that also seeds the claim counter.  THD x paged is never rendered (a validator
    predicate): under THD ``n_q_batches`` is 1 and the gather maps span the packed capacity ``SKV``.
    """
    B, QH, KH, SQ, SKV = problem_size
    num_pages, max_pages, k_rows_per_page, k_rows_per_head, v_rows_per_page, v_rows_per_head = paged_geom
    thd = bool(CFG.THD_VARLEN)
    n_q_batches = 1 if thd else B  # a packed operand has batch extent 1
    n_o_batches = B * SPLIT_KV  # SPLIT_KV > 1: the split-major partial slabs' batch axis (b + s x B); THD (never split) keeps extent 1
    q_tensor = _bshd(q_ptr, B, SQ, QH, TILE_K, q_strides, thd)
    o_tensor = _bshd(o_ptr, n_o_batches, SQ, QH, TILE_O, o_strides, thd)
    if cutlass.const_expr((gate_ptr is not None) != bool(CFG.EPILOGUE_GATE)):
        raise ValueError(f"sparse_d256_f16: gate_ptr presence ({gate_ptr is not None}) must match CFG.EPILOGUE_GATE={CFG.EPILOGUE_GATE}")
    # The gate is O-shaped on every arm: O's (B, SQ, QH, D) form, batch extent 1 under THD (the same `thd` flag as O).
    gate_tensor = _bshd(gate_ptr, B, SQ, QH, TILE_O, gate_strides, thd) if cutlass.const_expr(gate_ptr is not None) else None
    lse_tensor = None
    if cutlass.const_expr(lse_ptr is not None):
        l0, l1, l2 = lse_strides
        lse_tensor = cute.make_tensor(lse_ptr, cute.make_layout((1 if thd else n_o_batches, QH, SQ), stride=(l0, l1, l2)))
    # The list rows: one per packed / dense token, or ONE per sequence under LIST_PER_SEQUENCE (block_ids [B, BLOCK_TOPK]).
    n_list_rows = B if cutlass.const_expr(LIST_PER_SEQUENCE == 1) else n_q_batches * SQ
    block_ids_tensor = _vec(block_ids_ptr, n_list_rows * BLOCK_TOPK)
    block_lens_tensor = None
    if cutlass.const_expr(block_lens_ptr is not None):
        block_lens_tensor = _vec(block_lens_ptr, n_list_rows)
    seq_kv_lens_tensor = _vec(seq_kv_lens_ptr, THD_META_WORDS(B) if thd else B)
    block_table_tensor = None
    if cutlass.const_expr(block_table_ptr is not None):
        block_table_tensor = cute.make_tensor(block_table_ptr, cute.make_layout((B, max_pages), stride=(max_pages, 1)))

    def _tma_swz(byte_w: int):
        return tmap.TensorMapSwizzle.s128b if byte_w == 128 else tmap.TensorMapSwizzle.s64b if byte_w == 64 else tmap.TensorMapSwizzle.s32b

    # Q^T box: ONE token x G heads of one KV group (row = head within the group), the 4-D map over (D, QH, SQ, B).
    stride_order = (3, 2, 1, 0)
    box_q = (1, Q_BOX_TOKENS, Q_BOX_ROWS, GRANU_ELEMS)
    tma_q_desc = tmap.create_tensor_map_tiled(
        global_address=q_tensor.iterator.toint(),
        dtype=q_tensor.element_type,
        global_dims=tuple(q_tensor.shape[i] for i in stride_order),
        global_strides=tuple(cutlass.Int64(q_tensor.stride[i]) * q_tensor.element_type.width // 128 for i in stride_order[1:]),
        box_dims=tuple(box_q[i] for i in stride_order),
        swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
        l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
    )
    # The gate's LOAD descriptor (EPILOGUE_GATE): the SAME box orientation as Q^T over the gate tensor -- ONE token x the
    # item's G heads x 64 elements per subtile, SW128 -- so the box bytes are GATE_TX_BYTES = Q_BOX_ROWS x TILE_O x GATE_BPE
    # and the tile lands in the freed Q^T slot under the swizzle the epilogue's per-lane reads decode.  Ungated it aliases
    # tma_q_desc (always a valid tensor map, never read).
    tma_gate_desc = (
        tmap.create_tensor_map_tiled(
            global_address=gate_tensor.iterator.toint(),
            dtype=gate_tensor.element_type,
            global_dims=tuple(gate_tensor.shape[i] for i in stride_order),
            global_strides=tuple(cutlass.Int64(gate_tensor.stride[i]) * gate_tensor.element_type.width // 128 for i in stride_order[1:]),
            box_dims=tuple(box_q[i] for i in stride_order),
            swizzle=_tma_swz(CFG.Q_SWZ_BYTES),
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )
        if cutlass.const_expr(gate_tensor is not None)
        else tma_q_desc
    )

    # K / V gather maps: ONE 2-D map per operand, rows = the tokens of every batch (B x SKV, pitch = the token stride;
    # PAGED arm: the pool's token rows, num_pages x rows_per_page, at the same pitch), columns = the token's full row
    # ((KH - 1) x head stride + D elements, contiguous); box (64 elements, 1 row) = one 128-B SW128 span.  The head and
    # the column box live in the COLUMN coordinate (gather warps: head x head_stride + box x 64), the batch / page term in
    # the ROW coordinate (batch x SKV + 4 blk + r; page x rows_per_page + head x rows_per_head + (4 blk) % PAGE_SIZE + r).
    def _gather_desc(ptr, strides, rows_per_page):
        _bs, ss, hs = strides
        rows = cutlass.Int32(num_pages) * cutlass.Int32(rows_per_page) if cutlass.const_expr(PAGED_KV) else cutlass.Int32(n_q_batches) * cutlass.Int32(SKV)
        inner = (cutlass.Int32(KH) - cutlass.Int32(1)) * cutlass.Int32(hs) + cutlass.Int32(TILE_K)
        return tmap.create_tensor_map_tiled(
            global_address=ptr.toint(),
            dtype=STORAGE_DTYPE,
            global_dims=(inner, rows),
            global_strides=(cutlass.Int64(ss) * BPE // 16,),
            box_dims=(GATHER_BOX_ELEMS, 1),
            swizzle=_tma_swz(CFG.K_SWZ_BYTES),
            l2_promotion=tmap.TensorMapL2Promotion.l2_128b,
        )

    tma_k_desc = _gather_desc(k_ptr, k_strides, k_rows_per_page)
    tma_v_desc = _gather_desc(v_ptr, v_strides, v_rows_per_page)

    if cutlass.const_expr(CFG.THD_VARLEN):
        # THD: the setup launch writes the metadata (lengths -> prefix sums, the live unit total T_live x KH with one unit per
        # packed token per KV head, the claim counter seeded at the grid size) into the workspace the main kernel reads as its
        # ``seq_kv_lens`` tensor; the grid is occupancy-sized (``n_ctas``), every CTA takes unit blockIdx.x first and then
        # claims from the counter; a CTA past the live total exits at entry.
        q_lens_t = _vec(thd_q_lens_ptr, B + 1)
        kv_lens_t = _vec(thd_kv_lens_ptr, B + 1)
        _build_thd_meta_kernel(
            seq_kv_lens_tensor,
            q_lens_t,
            kv_lens_t,
            thd_lens_form,
            cutlass.Int32(B),
            cutlass.Int32(KH),
            cutlass.Int32(1),
            n_ctas,
        ).launch(grid=(1, 1, 1), block=(THD_SETUP_THREADS, 1, 1), stream=stream)
        grid_shape = (n_ctas, cutlass.Int32(1), cutlass.Int32(1))
    else:
        # The grid is the work list: one CTA per (query token x split chunk, KV head, batch) -- x = tok + SQ x split; the CLC
        # scheduler hands cancelled CTAs to the resident ones (one CTA per SM through the 221 KiB SMEM footprint).
        grid_shape = (SQ * SPLIT_KV, KH, B)
    _kernel(
        tma_q_desc,
        tma_k_desc,
        tma_v_desc,
        o_tensor,
        lse_tensor,
        block_ids_tensor,
        block_lens_tensor,
        seq_kv_lens_tensor,
        cutlass.Int32(SQ),
        cutlass.Int32(SKV),
        cutlass.Int32(k_strides[2]),
        cutlass.Int32(v_strides[2]),
        scale_softmax_log2,
        block_table_tensor,
        cutlass.Int32(max_pages),
        cutlass.Int32(k_rows_per_page),
        cutlass.Int32(k_rows_per_head),
        cutlass.Int32(v_rows_per_page),
        cutlass.Int32(v_rows_per_head),
        cutlass.Int32(B),
        tma_gate_desc,
        use_pdl,
    ).launch(
        grid=grid_shape,
        block=[CFG.THREADS_PER_CTA, 1, 1],
        cluster=(1, 1, 1),
        stream=stream,
        use_pdl=use_pdl,  # the programmatic-stream-serialization attribute, paired with the kernel's griddepcontrol sites
    )


EXPLICIT_ABI = True  # pointer/int host entry; the adapter builds the argument list itself


@lru_cache(maxsize=None)
def compile(  # noqa: A001
    has_lse: bool = True,
    has_block_lens: bool = False,
    use_pdl: bool = False,
) -> Callable:
    """Compile the pointer ABI: ``has_lse`` folds the LSE store in / out, ``has_block_lens`` the per-row count read; the
    paged arm's block table is present exactly when the config is paged (CFG.PAGED_KV, part of the template's identity).
    Head dims are the config's (d_qk = d_v = 256, exact); shapes and strides are runtime arguments.  SPLIT_KV > 1 (the
    config's): the O pointer is the fp32 PARTIAL slab and ``has_lse`` must be True (the per-split LSE drives the combine).
    ``use_pdl`` (appended; in the key) traces the programmatic-dependent-launch form: the launch attribute plus the per-role
    ``griddepcontrol`` wait before the first dependent global read, and NO explicit release -- the CTA exit triggers the dependent
    grid at this grid's completion (header, PDL SITES: the measured placement) -- the same math, bitwise; the False rendering is the
    program as before."""
    if SPLIT_KV > 1 and not has_lse:
        raise ValueError("sparse_d256_f16: split_kv > 1 requires has_lse=True (the per-split LSE partial drives the combine)")
    _cache_key = _template_key(globals(), locals(), "compile")
    gmem = cute.AddressSpace.gmem

    def P(dtype, align=16):
        return cute.runtime.make_ptr(dtype, 16, gmem, assumed_align=align)  # fake: type only

    i64_3 = (cutlass.Int64(0),) * 3  # stride slots: Int64 leaves, see _host
    thd = bool(CFG.THD_VARLEN)
    return _compile_cached(
        _host,
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(STORAGE_DTYPE),
        P(cutlass.Float32) if SPLIT_KV > 1 else P(STORAGE_DTYPE),  # the fp32 partial slab under a split, else the caller's O
        P(cutlass.Float32, 4) if has_lse else None,
        P(cutlass.Int32, 16),
        P(cutlass.Int32, 4) if has_block_lens else None,
        P(cutlass.Int32, 4),
        (0, 0, 0, 0, 0),
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        i64_3,
        cutlass.Float32(0.0),
        P(cutlass.Int32, 4) if PAGED_KV else None,
        (0, 0, 0, 0, 0, 0),
        P(cutlass.Int32, 4) if thd else None,
        P(cutlass.Int32, 4) if thd else None,
        cutlass.Int32(0) if thd else None,
        cutlass.Int32(0) if thd else None,
        # The gate is a MODULE specialization (TemplateParams.epilogue_gate -> CFG.EPILOGUE_GATE): its pointer slot exists on
        # the gated rendering only, its strides are runtime host arguments, so nothing of it is in the compile key.
        P(STORAGE_DTYPE) if CFG.EPILOGUE_GATE else None,
        i64_3,
        bool(use_pdl),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
        cache_key=_cache_key,
        symbol="frost_sdpa_fwd_sparse",
    )
