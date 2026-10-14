# MoE Monokernel — Design

A single persistent CUDA kernel that executes an entire FP8 block-wise MoE
layer for small decode batches (BS ≤ 8) on Hopper (sm_90a): routing → up
projection (+SiLU+re-quant) → down projection → weighted combine.  One launch,
no intermediate kernel boundaries, capturable into a CUDA Graph.

Fixed shape (`Dims_BS8_E256_N512_K2048_BlockFP8_WGMMA_TMA`): E=256 experts,
N=512 (`moe_intermediate_size` per half), K=2048 hidden, M ≤ 8 tokens.

```text
GM in:  activations [BS,K] bf16      router_logits [BS,E] bf16
        w_up   [E,2N,K] fp8 + block scales [E,2N/128,K/128] fp32
        w_down [E,K,N]  fp8 + block scales [E,K/128,N/128]  fp32
GM out: activations_out [BS,K] bf16
```

## Execution model

- `GRID_SIZE` blocks (default 128), `BLOCK_SIZE = 384` threads = 12 warps.
  The whole grid is launched ONCE and persists through all five phases —
  no blocks are created or retired between stages; a block changes *role*
  (which expert / which output tile it works on) at each phase boundary.
- `__launch_bounds__(BLOCK_SIZE, 1)` + `GRID_SIZE <= SM count` pin one block
  per SM.  This co-residency invariant is what makes the cross-block
  flag/sentinel handoffs (below) deadlock-free: every block is scheduled from
  launch, so a block spinning on another block's flag can never wait on a
  block that has not yet been scheduled.
- Warp roles (fixed for the whole kernel; the same physical warps take
  phase-specific duties, detailed per stage below):
    - warps 0–7 (threads 0–255): **calc warps**, forming two WGMMA
    warpgroups — WG0 = warps 0–3, WG1 = warps 4–7.  A WGMMA is issued by
    all 128 threads of a warpgroup together.
    - warp 8 lane 0 (thread 256): the single **TMA launcher thread** —
    issues every `cp.async.bulk.tensor.2d` and arms every mbarrier in the
    block, in all phases.
    - warps 8–11 (threads 256–383): **prefetch warps** (PF0–PF3) — scale
    loads, the deferred up-proj epilogue, the deferred down-proj
    accumulate, and (warps 1–11) the Phase-2 quantization.

The stage geometry is fixed by the shape's `KernelConfig` (`GRID_SIZE = 128`,
`DOWN_COL_TILE` = DCT = 256, `K_STEP_UP` = KUP = 256, `K_STEP_DOWN` = KDN = 256,
`UP_W_SLOTS` = SLOTS = 4; `UP_COL_HALVES` = UCH is derived).  All stage geometry
follows from these and the (E, N, K) dims:

| derived quantity  | formula                                 | value | meaning                             |
| ----------------- | --------------------------------------- | ----- | ----------------------------------- |
| `UCH`             | `max(1, 2N·DCT / (128·K))`              | 1     | 128-row M-atoms per up-block        |
| `UP_GRID`         | `2N / (128·UCH)`                        | 8     | blocks per expert, up-proj          |
| `UP_GROUPS`       | `GRID_SIZE / UP_GRID`                   | 16    | experts in parallel, up-proj        |
| `K_TILES_UP`      | `K / KUP`                               | 8     | up-proj outer K iterations          |
| `K_SUBSTEPS_UP`   | `KUP / 128`                             | 2     | 128-K substeps per iteration        |
| `UP_ARM_DISTANCE` | `max(1, SLOTS − 2)`                     | 2     | weight-TMA prefetch distance        |
| `DOWN_GRID`       | `K / DCT`                               | 8     | blocks per expert, down-proj        |
| `DOWN_GROUPS`     | `GRID_SIZE / DOWN_GRID`                 | 16    | experts in parallel, down-proj      |
| `DOWN_COL_HALVES` | `DCT / 128`                             | 2     | 128-col WGMMA passes per K-step     |
| `K_TILES_DOWN`    | `N / KDN`                               | 2     | down-proj outer K iterations        |
| `K_SUBSTEPS_DOWN` | `KDN / 128`                             | 2     | 128-K substeps per iteration        |
| `K_BLOCKS_TOTAL`  | `K / 128`                               | 16    | routing-window / quantization atoms |

`UP_GROUPS == DOWN_GROUPS` (16), so the site-#2 producer/consumer sets coincide.
The routing-window tile is 32 KB (BS=8).  These constraints hold: `GRID_SIZE` is
a multiple of `UP_GRID` and `DOWN_GRID` and ≤ SM count; `K_TILES_UP` is a
multiple of SLOTS (cross-expert stitch slot alignment); `K_TILES_DOWN` is even
(the inter-expert lookahead lands in the slot the next expert reads); SHM fits
the 224 KB budget.

## Phase pipeline

```text
Phase 1  routing (calc warps: topK)              ∥  routing-window TMA:
                                                    full [BS,K] bf16 tile
                                                    → SHM bf16_in_full,
                                                    completion on bar_rwin
Phase 2  warp 0: prepare_moe_topk            ∥  warps 1..11: wait bar_rwin,
         (expert tally / prefix sums /              quantize bf16 → fp8_act_full
          slot assignment)                          + per-128-K act_scale
         __syncthreads()  — publishes both
Phase 3  up-projection (WGMMA + weight-TMA pipeline)
         → spec->temp_fp8 (expert-sorted rows) + temp_act_scale
Site #2  NO barrier — sentinel handoff: each act-scale cell doubles as the
         readiness flag for its fp8 payload; the down-proj polls the cells
         it consumes (per-expert granularity)
Phase 4  down-projection (WGMMA + weight/activation TMA double-buffer)
         → atomicAdd into spec->down_partial_out [BS,K] fp32
Site #3  NO barrier — flag counters: each block bumps its col stripe's
         arrival counter; only that stripe's Phase-5 writer polls it
Phase 5  fp32 → bf16 cast + writeback (first DOWN_GRID blocks only)
```

Zero-fill of `down_partial_out` happens at kernel entry with no cross-block
sync: every block runs the full routing+quantize+up-proj pipeline (tens of µs)
before its first Phase-4 atomicAdd, while the zero-fill retires within the
launch's first ~µs.  Phase 5 `=`-writes every output element, so no output
pre-zero pass or extra grid sync exists.

## Stage-by-stage execution detail

### Phase 1 — routing ∥ input prefetch (all GRID_SIZE blocks, replicated)

Every block executes Phase 1 identically and independently — routing and
the input tile are needed by every block later, and each block has its own
SHM, so the work is *replicated* across the grid rather than partitioned
(the work is tiny: 8 tokens × E logits).  No cross-block communication.

Within one block, two things run concurrently:

- **Calc warps 0–7 — top-k routing (`topK`)**: one warp per token
  (warp w handles token w; warps ≥ `batch_size` return immediately).
  Within a warp, lane t owns experts `{t, t+32, t+64, ...}` in registers
  (`E/32` per lane — 8 for E=256).  The warp runs `top_k` rounds of
  warp-reduce-argmax and lane 0 writes the token's ids/weights to
  `topk_ids_flat` / `topk_weights_flat`.  Then `sync_calc_threads()`
  (a 256-thread `bar.sync`) joins the 8 warps.
- **Warp 8 lane 0 — routing-window TMA**: arms `bar_rwin` once with the
  full tile's byte count, then issues `K_BLOCKS_TOTAL = K/128 = 16` bulk
  TMAs, one 8-token × 128-K bf16 box each, filling
  `bf16_in_full`.  Warp 8 lanes 1–31 and warps 9–11 are idle in Phase 1.

### Phase 2 — routing tables ∥ quantization (all GRID_SIZE blocks, replicated)

Warp split within each block:

- **Warp 0 (lanes 0–31) — `prepare_moe_topk`**: single-warp, three
  sub-phases (A tally, B fused prefix scans, C slot assignment) building
  `experts[]`, `expert_count`, `expert_slot_start[]`,
  `expert_routed_count[]`, `sorted_slot[]`, `down_rank[][]`.  Warp 0 never
  reads `bf16_in_full`, so it skips the `bar_rwin` wait.
- **Warps 1–11 (11 warps, 352 threads) — `routing_phase_quantize`**: each
  thread first waits on `bar_rwin` (the Phase-1 TMA completion), then the
  11 warps split the `BS × K_BLOCKS_TOTAL` (token, 128-K-block) pairs
  (128 pairs) stride-11 by warp.  Each pair = one warp call to
  `moe_streaming_quantize_k128`: 32 lanes × 4 bf16 values, warp-reduce
  max, fp8 quantize into `fp8_act_full`, one scale into `act_scale`.
  Note this uses calc warps 1–7 too — the Phase-1 role split does not
  apply here; only warp 0 is reserved.

One trailing `__syncthreads()` publishes both sides to all 12 warps.
There is intentionally no other sync between warp 0 and warps 1–11 — they
write disjoint SHM.

### Phase 3 — up-projection (grid partitioned: UP_GROUPS × UP_GRID)

Block assignment: `up_group = blockIdx.x / UP_GRID`,
`up_block_idx = blockIdx.x % UP_GRID`.

- **Blocks per expert**: `UP_GRID = 2N/(128·UCH)` blocks jointly produce
  one expert's full `2N` intermediate rows; block `up_block_idx` owns rows
  `[up_block_idx · 128·UCH, +128·UCH)`.  Here: 8 blocks × 128 rows (64 gate
  + 64 up features each, interleaved — UCH = 1).
- **Experts in parallel**: `UP_GROUPS = GRID_SIZE/UP_GRID` = 16 expert groups
  run concurrently, one active expert per group at a time.
- **Expert loop**: group g iterates the *active* expert list (built by
  Phase 2, ascending id, length `expert_count ≤ min(E, BS·top_k)`) as
  `e = g, g + UP_GROUPS, g + 2·UP_GROUPS, ...`.  Each group therefore
  visits ≤ `ceil(expert_count / UP_GROUPS)` experts — with BS=8, top_k=8
  at most 64 are active, so ≤ 4 per group at UP_GROUPS=16.  A group whose
  index exceeds `expert_count` skips Phase 3 entirely and waits at the
  site-#2 barrier.
- **Per expert, warp duties**:
    - warps 0–7 (WG0 + WG1): the K-loop — `K_TILES_UP = K/KUP = 8`
    iterations, each
    waiting `bar_w[s % SLOTS]` then chaining 4 WGMMAs per 128-K substep
    per M-atom, with scale-apply at every 128-K boundary.  WG0 computes
    SHM weight rows [0..63] of each atom, WG1 rows [64..127].  At the
    K-loop tail they do the per-lane `rw·up·silu(gate)` combine into
    `post_silu_scratch` and snapshot the rank cache.  8 calc threads
    (one per token) also populate the per-expert routing cache at the
    K-loop top.
    - warp 8 lane 0 (launcher): inside each K-iteration, arms + TMAs the
    weight slot `UP_ARM_DISTANCE = max(1, SLOTS−2)` iterations ahead; on
    the last `UP_ARM_DISTANCE` iterations it stitches the *next* expert's
    first tiles instead, so the pipeline never drains across experts.
    - warp 8 (lanes 0–31, as PF warp 0): cp.async-prefetches the *next*
    expert's block-scale tile into the `up_scale` ping-pong during the
    current K-loop.
    - warps 8–11 (PF0–PF3): the **deferred epilogue of the previous
    expert** — one warp per token, `ceil(BS/4)` waves spaced across the
    first `K_TILES_UP - 1` iterations: read `post_silu_scratch`,
    warp-reduce max over the block's 64 features,
    fp8-quantize, store to `temp_fp8[sorted_slot_row]` +
    `temp_act_scale`.  The last visited expert has no successor and
    drains inline on calc warps after the loop (one token per warp).

### Site #2 — Phase 3 → 4 handoff (sentinel, no barrier)

No barrier counter is involved.  Each `temp_act_scale` cell is
release-published by its producing up-proj warp *after* the covering fp8
payload segment (`moe_publish_act_scale`), and the down-projection polls
exactly the cells it consumes until they turn non-sentinel before reading
the payload.  This gives per-expert granularity — a down-block starts an
expert as soon as THAT expert's rows are published (consumers wait on data, so
no barrier-counter protocol is needed).  See "Cross-block synchronization"
for the sentinel value and reset discipline.

### Phase 4 — down-projection (grid re-partitioned: DOWN_GROUPS × DOWN_GRID)

The same 128 blocks re-map: `down_group = blockIdx.x / DOWN_GRID`,
`down_block_idx = blockIdx.x % DOWN_GRID`,
`base_col = down_block_idx · DOWN_COL_TILE`.

- **Blocks per expert**: `DOWN_GRID = K/DCT = 8` blocks jointly
  cover one expert's `K` output columns; each block owns `DCT = 256`
  columns = `DOWN_COL_HALVES = DCT/128 = 2` sequential
  128-col WGMMA passes per K-step.
- **Experts in parallel**: `DOWN_GROUPS = GRID_SIZE/DOWN_GRID = 16` groups.
- **Expert loop**: `e = down_group, down_group + DOWN_GROUPS, ...` over
  the same active-expert list.  Note the up and down loops visit experts
  in a different interleaving; correctness needs only that Phase 3
  finished the expert before Phase 4 reads it, which site #2 guarantees.
- **Per expert, warp duties**:
    - warps 0–7: K-loop of `K_TILES_DOWN = N/KDN = 2` iterations,
    each
    waiting `bar_w[s&1]` + `bar_a[s&1]` (weight + activation double
    buffers) then running the lo/hi WGMMA chains per substep per
    col-half with scale-apply; at the loop tail they write the
    accumulators to `down_out` in SHM.
    - warp 8 lane 0 (launcher): prefetches K-step s+1 during step s; on
    the last step prefetches the *next expert's* step-0 weight +
    activation tiles instead (inter-expert lookahead).  The activation
    tile is one bulk TMA per 128-K substep covering all ≤ 8 routed rows
    of the expert (fetched from the contiguous `temp_fp8` slab).
    - warp 8 (PF0): loads the expert's per-token activation scales;
    warp 9 (PF1): loads the expert's weight scales — both once per
    expert, in parallel, before the K-loop.
    - warps 8–11 (128 PF threads): the **deferred accumulate of the
    previous expert** — `out_accum[tok][col] += down_out[col][rank]`,
    the (tok, col) plane sliced across the first `K_TILES_DOWN − 1`
    iterations (a single slice at s=0, since `K_TILES_DOWN = 2`).  The
    last visited expert's accumulate runs after the loop with all 12
    warps.
- After the expert loop: every block `atomicAdd`s its
  `out_accum[BS][DOWN_COL_TILE]` slice into the global
  `down_partial_out[BS][K]` (so each output cell receives
  `DOWN_GROUPS` atomic adds).

### Site #3 — Phase 4 → 5 handoff (flag counters, no barrier)

`down_partial_out` is atomicAdd-accumulated, so readiness can't live in the
data.  Instead every block bumps its col stripe's parity-selected arrival
counter (`down_ready[parity][blockIdx.x % DOWN_GRID]`, fence + add) after its
Phase-4 adds, and ONLY the stripe's single Phase-5 writer polls it up to
`DOWN_GROUPS`.  The other blocks publish and run to kernel exit — no
grid-wide mutual spin (the old col-stripe barrier made all GRID_SIZE blocks
wait).

### Phase 5 — writeback (first DOWN_GRID blocks only)

Only blocks with `blockIdx.x < DOWN_GRID` (8) write: all 384
threads of each stream-cast the block's own `DCT`-column stripe of
`down_partial_out` from fp32 to bf16 in `activations_out`, and zero-fill
the padding tokens `[batch_size, BS)`.  The remaining
`GRID_SIZE − DOWN_GRID` blocks (120) exit after the site-#3
barrier.

## Grid carve

Both projections partition the grid into groups that process different
experts in parallel:

- Up: each block owns `UP_COL_HALVES` (UCH) stacked 128-row WGMMA M-atoms of
  the `[2N, K]` weight matrix.
  `UP_GRID = 2N / (128·UCH)` blocks cover one expert;
  `UP_GROUPS = GRID_SIZE / UP_GRID` experts run in parallel.
  Group g handles experts `g, g+UP_GROUPS, ...` in `shmem->experts[]` order.
- Down: each block owns `DOWN_COL_TILE` (DCT) output columns of `[K]`.
  `DOWN_GRID = K / DCT`, `DOWN_GROUPS = GRID_SIZE / DOWN_GRID`.

**Producer/consumer coupling.**  The site-#2 handoff is symmetric: the 8
blocks that produced an expert's `temp_fp8` rows are exactly the 8 blocks that
consume them, i.e. `UP_GROUPS == DOWN_GROUPS` (16).  This holds because
`UCH = 2N·DCT / (128·K)` = 1 for this shape.

**Interleaved up-proj weights (UCH == 1).**  One 128-row A-tile packs 64 gate
+ 64 up rows in the gate/up *pair layout*.  The tensor must be pre-interleaved
in Python (`interleave_for_tma_wgmma_up`) so a single 128×128 TMA fetches one
full WGMMA A-tile.  Each lane then holds gate and up for the same output
feature, so silu(gate)·up is a register-local combine.

## Up-projection mechanics (Phase 3 deep-dive)

Per-expert loop; per expert a K-loop over `K_TILES_UP = K / K_STEP_UP` outer
steps, each step = `K_SUBSTEPS_UP` 128-K substeps (128 = SWZ128 atom width =
FP8 block-scale granularity).

Weight-TMA lookahead pipeline: `UP_W_SLOTS` (S) physical `bar_w`/`w_wgmma`
slots, arm distance `A = max(1, S-2)`.  At iter s the launcher arms slot
`(s+A) % S` for logical iter s+A; calc warps wait `bar_w[s % S]`.  The last A
iters of an expert *stitch* the next expert's iters [0, A) into slots [0, A),
so the mbarrier parity chain carries across the expert boundary with no
barrier reinit (this requires `K_TILES_UP % S == 0`).  A slot is never
re-armed before its previous consumer wait completed
(wraparound safety: `S >= A + 2` by construction).  The per-slot parity
registers are hoisted OUT of the expert loop (like the down-proj's): a slot
completes `K_TILES_UP / S` phases per expert, and when that quotient is odd
(e.g. K_TILES=12, S=4) the mbarrier ends the expert at phase 1 — a
per-expert parity reset would then let the next expert's first wait pass on
the stale phase and corrupt the arm/wait pairing.

Per 128-K substep, each WG chains 4 `wgmma.mma_async.m64n8k32.e4m3` reading:

- A = weight tile from `w_wgmma` (SWZ128 canonical Major::K, LBO=16,
  SBO=1024, swizzle=1),
- B = activations from `fp8_act_full` (SWIZZLE_NONE; LBO=144 — see SHM),

then applies `weight_scale × act_scale` at the 128-K boundary into fp32
accumulators.

Cross-expert latency hiding:

- the next expert's block-scale tile is prefetched via `cp.async` into a
  ping-pong `up_scale[2]` buffer during the current K-loop;
- a per-expert routing cache (`up_rank_for_tok` / `up_rw_for_tok`) is
  populated by 8 calc threads with one 16-B vector load per token, replacing
  a dependent SHM scan; the same pre-K-loop `__syncthreads()` publishes both.

**Deferred epilogue.**  At the K-loop tail, calc warps only do the per-lane
`rw·up·silu(gate)` combine (`__fdividef` + `__expf`) and store fp32 to
`post_silu_scratch[feature][tok]`, plus snapshot the rank cache to
`up_rank_for_tok_prev`.  The expensive part — warp-reduce max, fp8 quantize,
GM stores into `temp_fp8`/`temp_act_scale` — is *deferred to prefetch warps
inside the NEXT expert's K-loop* (one warp per token, waves spaced across the
first `K_TILES-1` iterations so DRAM store bursts don't bunch and the last
iter stays free for the cross-expert stitch).  This is numerically
schedule-invariant: `post_silu_scratch` holds the previous expert's values
for the whole current K-loop, and routing tables are immutable.  The last
expert in a block's range has no successor and drains inline on calc warps
after the loop.

Output layout: the writeback row is `sorted_slot[tok·top_k + k]` — Phase 2
assigns each routed (token, expert) pair a row so that each expert's tokens
occupy a *contiguous slab* `[expert_slot_start[id], +routed_count)` of
`temp_fp8`.  That contiguity is what lets Phase 4 fetch a whole expert's
activations with one bulk TMA.  One fp8 scale per (row, up-block) goes to
`temp_act_scale` (block size along N = UCH·64 features).

## Down-projection mechanics (Phase 4 deep-dive)

Per-expert loop with stride `DOWN_GROUPS` starting at `down_group`.  Per
expert:

- Prefetch warps load the expert's weight scales (warp 9) and the per-token
  activation scales for the *whole* expert (warp 8) — hoisted out of the
  K-loop.  The activation-scale SHM layout is `[block][tok]`
  (bank-conflict-free broadcast in the scale-apply).
- K-loop over `K_TILES_DOWN = N / K_STEP_DOWN` with a 2-slot weight +
  activation TMA double-buffer (`bar_w[0..1]`, `bar_a[0..1]`, reinitialized in
  the down-proj prologue).  The launcher prefetches step s+1 during step s;
  at the last step it instead prefetches the *next expert's* step-0 tiles
  (inter-expert lookahead — requires `K_TILES_DOWN` even so the freed slot is
  the one the next expert's s=0 wait reads).  Parity state is hoisted out of
  the expert loop and never reset.
- When `routed_count == 0` for an expert, no activation TMA is armed; the
  WGMMA computes on garbage, but the rank-filtered accumulate never reads or
  accumulates those results (every token's `rank` is `0xFF` for an unrouted
  expert), so the garbage cannot reach the output.
- Epilogue: accumulators → `down_out[DCT][8]` in SHM.  The
  `out_accum[tok][col] += down_out[col][rank]` accumulate for the *previous*
  expert runs deferred on prefetch warps, sliced across the first
  `K_TILES_DOWN - 1` K-steps (the last step stays clean so the read of
  `down_out` fully drains before this expert's epilogue overwrites it).
  `rank = down_rank[expert_id][tok]` was recorded once in routing Phase C
  (0xFF = token not routed to that expert) — nothing is recomputed here.
- After the expert loop: final accumulate for the last expert (all warps),
  then `atomicAdd` of `out_accum` into the single global
  `down_partial_out[BS][K]` buffer.  Phase 5 reads each cell exactly once —
  no cross-group reduction pass.

## Routing mechanics (Phase 1/2 deep-dive)

`topK` (one warp per token, experts distributed lane-cyclically,
`NUM_EXPERTS % 32 == 0` keeps score arrays in registers):

- Fast path (softmax+renormalize, or sigmoid): select top-k on raw logits
  (activations are monotone), then exponentiate only the k winners; for
  softmax+renorm the global denominator cancels.  Softmax+no-renorm needs the
  full denominator and falls back to a full warp softmax.
- Ties break toward the lowest expert *index* (matching vLLM `topk_softmax`),
  not the lowest lane.
- Optional GLM-style biased selection: rank by `sigmoid(logit) + bias[e]`,
  weight stays the unbiased sigmoid (recovered as `metric - bias`);
  `routed_scaling_factor` is folded into the shared normalizer.

`prepare_moe_topk` (warp 0 only, 3 phases):

- A: vectorized zero of `expert_routed_count`, 0xFF-seed of `down_rank`,
  tally via `__match_any_sync` with routed pair eids cached in registers.
- B: fused dual warp scan (routed-count prefix + active-expert prefix) →
  `expert_slot_start[]` (packed u16 stores), `experts[]` (ascending eid), and
  `expert_count`.
- C: intra-expert rank per pair via `__match_any_sync` + cross-chunk carry →
  `sorted_slot[pair]` and `down_rank[eid][tok]`.

## Shared memory (`MoE_SHM`, ≤ 224 KB)

The dominant space is a union whose members have strictly disjoint lifetimes
(separated by the Phase-2 trailing sync and the site-#2 barrier):

| view                                   | phase | size (N512_K2048 cfg0) |
| -------------------------------------- | ----- | --------------- |
| `bf16_in_full[K/128][BS][128]`         | 1–2   | 32 KB           |
| `w_wgmma[UP_W_SLOTS][M_total][128]`    | 3     | 64 KB           |
| `w_down_wgmma[2][DCT·K_SUBSTEPS][128]` | 4     | dominates       |

Other notable fields:

- `fp8_act_full[K/128][8][T_TILE+1][16]` — single-buffer fp8 activations for
  the whole K range (Phase 3 reads with no slot alternation).  The 9th
  token row per 16-B chunk is padding: it moves the chunk stride from 128 B
  to 144 B so the routing-quantize stores and the WGMMA B reads are
  bank-conflict-free.  The WGMMA B descriptor's `LBO = 144` steps over the
  pad.
- `a_down_wgmma[2][K_SUBSTEPS_DOWN][8][8][16]` — down-proj activation
  double-buffer (SWZ128 atoms).
- `partial_result` union: `wgmma_out[128][9]` / `down_out[DCT][8]` /
  `post_silu_scratch[UCH·128][9]` — the +1 column padding makes the
  `[col][tok]` read pattern bijective over banks (gcd(9,32)=1).
- mbarriers (`bar_w[UP_W_SLOTS]`, `bar_a[2]`, `bar_rwin`), `alignas(16)`.
- Routing tables: `expert_slot_start[E]` (u16, `alignas(16)` — Phase B emits
  packed STS.128 stores), `expert_routed_count[E]` (u8),
  `sorted_slot[BS·8]` (u8), `down_rank[E][BS]` (u8, `alignas(16)` for the
  uint4 seed), `up_rank_for_tok[_prev][BS]` + `up_rw_for_tok[BS]` (V2 only).
- `act_scale[K/128][BS]` — transposed `[blk][tok]` layout for conflict-free
  scale-apply broadcasts.

## Global scratchpad (`MoEGemmSpec`)

Persistent GM workspace, one per process, zeroed once on first launch.
`mono_moe` allocates it automatically when the `scratchpad` argument is
omitted; callers that want to reuse a buffer across launches allocate one
explicitly with `alloc_scratchpad()`, sized by `get_scratchpad_size_bytes()`
(the `monomoe_scratchpad_size` binding, i.e. `sizeof(MoEGemmSpec<Dims>)`):

- `temp_fp8[TEMP_ROWS][N]` + `temp_act_scale[TEMP_ROWS][N/DOWN_ACT_BLOCK]` —
  Phase 3 → Phase 4 handoff, expert-sorted rows.
  **Layout invariant:** the host computes the device address of `temp_fp8`
  as `scratchpad + TEMP_FP8_OFFSET` when building the down-activation TMA
  descriptor, so no field may ever be inserted before `temp_fp8`; new fields
  (handoff flags) go at the tail.  A `static_assert` in
  `monomoe_binding.cu` enforces this.
- `down_partial_out[BS][K]` fp32 — Phase 4 atomicAdd target.
- sentinel-handoff tail state: `temp_act_scale_alt` (the second scale
  buffer), `launch_flip[GRID_SIZE]` (per-block private launch counters →
  buffer parity), `down_ready[2][DOWN_GRID]` (Phase-4→5 arrival counters).

## Cross-block synchronization (flag/sentinel handoffs)

There are no grid barriers.  The kernel launches via plain
`cudaLaunchKernel` (CUDA-Graph capturable) and orders its two cross-block
handoffs through the data path, so consumers wait only on the values they
actually need.  Both handoffs rely on the one-block-per-SM co-residency
invariant (a spinning consumer needs its producers scheduled), enforced at
compile time by `__launch_bounds__(BLOCK_SIZE, 1)` and at runtime by the
`GRID_SIZE <= SM count` check in the wrapper.

**Site #2 (Phase 3 → 4), sentinel-in-data:** each `temp_act_scale` cell
doubles as the readiness flag for the fp8 payload segment it covers.  The
producing warp stores the payload, then `__syncwarp()` +
`__threadfence()` + `atomicExch` of the scale (`moe_publish_act_scale`),
clamped to >= FLT_MIN so the sentinel `+0.0f` is never a valid value.  The
down-projection polls exactly the cells of the expert it is about to
consume (device-scope atomic reads) before reading scales or issuing the
activation TMA; the inter-expert lookahead TMA has its own sweep-poll
(`moe_wait_expert_scales_published`).  This gives per-expert granularity —
down work for an expert starts as soon as that expert's rows are published.

**Site #3 (Phase 4 → 5), arrival flags:** `down_partial_out` is
atomicAdd-accumulated, so readiness cannot live in the data (a partial sum
looks complete).  Each block instead bumps its col stripe's arrival
counter (`__syncthreads()`, `__threadfence()`, `atomicAdd(+1)`) and runs
to exit; only the stripe's single Phase-5 writer polls the counter up to
`DOWN_GROUPS`.

**Reset discipline (both sites):** flag state must never leak across
launches, so it is double-buffered by launch parity: every block bumps its
private `launch_flip` word once per launch (all blocks agree on parity
with no cross-block sync), the current parity's state is consumed, and the
OTHER parity's state is zero-refilled in the prologue, off the critical
path.  A `torch.zeros` scratchpad allocation establishes the invariant for
the first launch — no host-side re-init is ever needed, and the scheme is
CUDA-Graph-replay safe (parity keeps alternating across replays).

## TMA descriptors

Four `CUtensorMap`s are built host-side per launch (`moe_tma.cu`) and passed
as `__grid_constant__` kernel parameters:

| descriptor       | tensor                               | box     | swizzle |
| ---------------- | ------------------------------------ | ------- | ------- |
| up weights       | `[E·2N, K]` fp8 (interleaved or raw) | 128×128 | 128B    |
| activations      | `[BS, K]` bf16                       | 8×128   | none    |
| down weights     | `[E·K, N]` fp8 (raw)                 | 128×128 | 128B    |
| down activations | `temp_fp8 [rows, N]` fp8             | 8×128   | 128B    |

The TMA hardware applies the 8-row × 128-B core-matrix XOR swizzle at write
time, producing the canonical CUTLASS Major::K B128 layout that the WGMMA
descriptors read (`LBO=16`, `SBO=1024`, swizzle=1).  The activation
descriptor is SWIZZLE_NONE with a compact 8×128 box; the SHM destination is
therefore *tile-major* `[K/128][BS][128]` (each box gets its own 2 KB slab —
a `[BS][K]` layout would make consecutive boxes overlap because the TMA
writes with the box's own row stride, not the destination's logical stride).
The TMA `boxDim` cap of 256/axis is why the routing window is 16 separate
issues and why DCT=384 uses three 128-row weight TMAs per substep.
