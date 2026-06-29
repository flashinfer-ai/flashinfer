# Plan: Persistent main kernel (checkpointing SSU two-kernel split)

## Goal

Turn the main kernel into a **1D grid-stride persistent loop over single-head
work-units**. Launch `min(cta_per_sm · NUM_SMS, total_work)` CTAs; each CTA
grid-strides over work-units instead of the GPU launching one CTA per work-unit.

Two wins, both targeting the **no-write path** (the only place we lose to
triton-replay-pm — see *Why* below):

1. **Co-residency with conv1d.** Fewer main CTAs leave SM room so the main starts
   *during* conv1d (under PDL), like triton-pm's persistent main, instead of being
   contention-blocked until the 1-per-work-unit grid drains.
2. **L2 / smem reuse of per-group `C` / `old_B`.** With a head-fastest flatten,
   consecutive work-units are consecutive heads of one `(d_tile, seq)` → same group
   → their `C` (and write-path `old_B`) stay hot, and a later step can skip the
   reload outright. This **subsumes the old `MAIN_HEADS_PER_CTA` head-tiling** —
   the persistent loop is the single mechanism for "do more heads per CTA without
   more CTAs."

Default `cta_per_sm` high ⇒ `grid == total_work` ⇒ one work-unit/CTA ⇒
**bit-identical to today's non-persistent main**.

**Decision: ditch the MHC knob.** Work-unit = one head (MHC=1). No
`FLASHINFER_SSU_MAIN_HEADS_PER_CTA` env / Python handle / auto-heuristic. The
grid-stride loop replaces head-tiling.

## Why — write/no-write split at PRODUCTION max_window=16 (B300, 2026-06-25)

Per-path isolation, `bench_checkpointing_ssu.py` (CUPTI, SSU-kernel-only, mtp=6, bf16),
after this session's no-write work (cumAdt batch-prefetch + barrier cut, per-warp
loads, old_x predicate+split). prev_k=0/8 = no-write, 12/16 = write.

| kernel | b512 nw0 | nw8 | wr12 | wr16 | b1024 nw0 | nw8 | wr12 | wr16 |
|--------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| **cuda-incr-2k** | **50.5** | **53.3** | **75.9** | **76.4** | **90.5** | **94.7** | **144.6** | **145.4** |
| cuda-incr (mono) | 62.7 | 63.1 | 82.3 | 82.3 | 113.3 | 114.1 | 160.2 | 160.1 |
| triton-replay-pm | 42.1 | 45.2 | 94.4 | 95.5 | 75.4 | 80.1 | 184.1 | 186.5 |

- **Write: we crush triton-pm** (b512 −19µs, b1024 **−40µs**). The write path needs nothing.
- **No-write: triton-pm wins** (b512 +8µs, b1024 +15µs). This is the persistent-main edge.
- We beat the monolithic on both paths/batches.

**The no-write deficit is the per-step 16 KB state reload, NOT the rectangle.**
Confirmed our no-write path IS the rectangle trick (equations doc §1.1 OUT.3 =
`CB_old @ old_x`, `replay_state_mma` skipped) and it's FLOP-optimal (avoids the
DSTATE contraction a replay would cost). NCU: `long_scoreboard` 35% sits on the
`C @ state` (OUT.1) state load (`:639`/`:530`). Persistence (resident state +
co-residency) is the lever — exactly this plan.

MHC×STAGES sweep @ b=1024 (mixed, conv1d) confirmed MHC=4/STAGES=1 is already
optimal among the non-persistent knobs (124.98µs), so further head-tiling/pipeline
tuning is exhausted — the only remaining lever is persistence.

The persistent main (v1.0, MHC=1 single-head work-units) **regresses b=1024**.
`cta_per_sm` sweep (grid = min(cta_per_sm·NUM_SMS≈148, total_work=32768)):

| cta_per_sm | grid | median µs (pre-pipeline) | + setup-prefetch |
|---:|---:|---:|---:|
| 1 | 148 | 908 💀 | — |
| 2 | 296 | 477 | — |
| 4 | 592 | 258 | — |
| 8 | 1184 | 149.9 | 149.3 |
| 16 | 2368 | 144.4 | 144.4 |
| 32 | 4736 | — | 143.4 |
| default (huge) | 32768 | 146.6 | 145.6 |

Bars: old MHC=4 (non-persistent) **124.6**, triton-replay-pm **120.4**.

Findings:
- **cta_per_sm < occupancy (8) is pointless** — at b=1024 the GPU is saturated
  (total_work ≫ occupancy·NUM_SMS), so cutting CTAs just under-occupies (cps=1 →
  1 CTA/SM → 6× slower). Co-residency room is a *small-batch (cliff)* lever, not b=1024.
- **Setup-prefetch (software-pipeline the per-work-unit scalar metadata loads): no
  measured effect** (≤1µs, within noise). The dependent scalar loads were not the
  bottleneck — the compiler already overlapped them.
- **The ~20µs regression is MHC=1 vs MHC=4.** Single-head work-units reload per-group
  `C`/`old_B` *every head* (16384 CTAs) vs MHC=4's load-once-reuse-4-heads (4096 CTAs)
  — 4× the C/old_B traffic + per-CTA barrier/setup overhead. The grid-stride "head-fastest"
  flatten only gives **L2** reuse across concurrent CTAs (the stride = gridDim.x jumps over
  the same-group block, so a CTA's consecutive iterations are different groups → no *smem*
  reuse, and step-7 skip-reload is moot).

## NGROUPS template + division-free unflatten (2026-06-25)

NCU on the MHC=1 persistent main showed the new `wait` stall (14% no-write) was the
**unflatten's runtime integer divisions** (`tile % ngroups`, `/ ngroups`, `% batch`,
`/ batch` — `ngroups`/`batch` runtime → `MUFU`/slow division).  The old 3D grid read
`blockIdx.{x,y,z}` directly (zero divisions).

Fix: **expose `num_groups` as a jinja template param** (`NGROUPS`, URI key `_ng_`) so
`NHEADS = NGROUPS·HEADS_PER_GROUP` is compile-time, and **reorder the flatten** so the
unflatten divides only by compile-time constants (NHEADS, D_SPLIT, HEADS_PER_GROUP);
`seq` is the top quotient so `batch` is never a divisor.  Also ripped `MAIN_HEADS_PER_CTA`
(MHC=1) + `PIPELINE_STAGES` from the kernel template (replaced by `NGROUPS`); `head_loop`'s
dead MHC>1 inner loop left for a follow-up cleanup.

Result (b=1024, mixed+conv1d) — **~10µs across the board**, far more than the stall %
implied (the divisions gate the state-load issue + dispatch on the critical path):

| cta_per_sm | pre-NGROUPS | division-free | Δ |
|---:|---:|---:|---:|
| 8  | 149.3 | 139.6 | −9.7 |
| 16 | 144.4 | **133.6** | −10.8 |
| 32 | 143.4 | 134.1 | −9.3 |
| grid==total_work | 145.6 | 137.0 | −8.6 |

- Persistence helps ~3µs (cps=16 133.6 < grid==total_work 137.0); **default set to cps=16**
  (slightly oversubscribed beats exactly-resident cps=8 = 139.6).
- Regression vs MHC=4 (124.6) cut from +20 → **+9µs**; gap to triton-pm (120.4) now +13µs.

**Remaining +9µs vs MHC=4 = the per-work-unit C/old_B load** (single head/work-unit reloads
C; MHC=4 amortized over 4 heads).  The grid-stride's stride jumps over same-group blocks → a
CTA never gets consecutive same-group work-units → no smem C-reuse.  Next lever (both keep
MHC=1): **block-cyclic work assignment** (contiguous head-fastest run per CTA → enable
skip-reload of C/old_B) OR bring back head-tiling as the work-unit size.  **Decide after the
no-write ncu diff.**

## No-write ncu diff — division-free unflatten landed (b=512, mw=16, 2026-06-25)

NCU `--set full`, no-write (prev_k=0), `checkpointing_ssu_main_kernel`.  "grid==tw" =
`FLASHINFER_SSU_MAIN_CTA_PER_SM=1<<20` (one work-unit/CTA, apples-to-apples vs the old
profile); "cps=16" = the new production default.

| metric | prev (grid==tw, **with** div) | div-free (grid==tw) | div-free (**cps=16**) |
|--------|-----:|-----:|-----:|
| duration | 52.35 µs | 49.50 µs | **48.93 µs** |
| grid | 16384 | 16384 | 2368 (2 waves) |
| achieved occ | 44.9% | 43.9% | 46.9% |
| compute (SM) | 55.0% | 48.7% | 43.0% |
| memory | 60.9% | 63.2% | 64.4% |
| DRAM total | 160.2 MB | 151.2 MB | 151.7 MB |
| long_scoreboard | 31.3% | 35.0% | 35.7% |
| **wait** | **14.1%** | 12.1% | **11.3%** |
| barrier | 12.7% | 11.7% | 14.6% |

- **−3.4µs (−6.5%)** on no-write b=512 (52.35 → 48.93).  Pure division removal = −2.85µs
  (same grid); persistent loop adds −0.57µs.
- **The `wait` divisions are gone.** Old top-`wait` PCs `:818`/`:821` were `UIMAD`/`MUFU`/
  `UISETP`/`UIADD3` (the `%ngroups` / `/batch` ops) — they no longer appear; the new top
  `wait` is genuine `HMMA` latency (`mma_sm80.hpp:238`).  DRAM also −9MB (division codegen
  freed registers → less spill).
- **Now cleanly state-load-bound.** `long_scoreboard` 35.7%, of which **62% at `:534`**
  (`__syncwarp` draining the `state + old_x` cp.async prefetch) and **23% at `:598`**
  (the `cb_scaled`/C LDG).  No-write has too little compute to hide the per-work-unit
  state reload + C/old_B re-fetch.

## Decision: head-tiling fused to the group (next lever)

The two LSB hotspots are exactly what MHC=4 amortized: C/old_B reload (`:598`) is *per-group*
(shared across `HEADS_PER_GROUP` heads), and the state-load hide (`:534`) needs more in-flight
MMA per work-unit.  **Chosen lever: make the work-unit a tile of contiguous same-group heads**
(recovers both MHC=4 benefits inside the persistent, division-free grid), tied to the group
structure rather than a free knob:

- `total_work = D_SPLIT · batch · NGROUPS` (was `· NHEADS`); work-unit = one group.
- Inner loop over `HEADS_PER_GROUP` heads reusing the resident C/old_B (loaded once per
  work-unit) and pipelining `HEADS_PER_GROUP` state loads.
- If a full group is too coarse for occupancy at small batch, the tile is a compile-time
  divisor of `HEADS_PER_GROUP` (a TILE constant, still no runtime division).
- Rejected alternative: block-cyclic single-head assignment (CTA gets a contiguous head-run
  but work-unit stays single-head) — recovers smem C-reuse but not the state-load hiding, and
  needs a per-CTA run-length the grid-stride doesn't naturally give.

## State-pipeline (STATE_PIPE=2) + register-resident meta ring (B200, 2026-06-26)

Path actually taken (instead of the group head-tiling decided above): pipeline the **full
per-work-unit bundle** across `MAIN_STATE_PIPE = 2` smem slots — a cross-iteration cp.async
ring — to hide the per-work-unit state reload the no-write path can't cover with compute.
Depth-2 makes the bundle (C/old_B/z/old_x/x/state, STATE_PIPE-strided) ~38.4 KB/CTA →
**smem-bound at 5 blocks/SM** (independent of registers).

`HeadMetaSSU` trimmed to **`{tile, cache_slot}`** only; everything else derived on the fly
(`derive_head`: `d_tile`/`seq`/`head`/`group` via constexpr divides, `buf_read`/`prev_k` one
load each off the resolved `cache_slot`, `seq_len = NPREDICTED` constexpr). The ring
prefetches `cache_slot` STAGES ahead (the chain head `sbi[seq]`) so the consumer never pays
the `sbi[seq]→prev/buf` chain.

### Bug found + fixed: the meta ring lived in LOCAL memory

The prefetched ring `meta[STAGES]` was indexed by the **runtime** `slot = j % STAGES`, so the
compiler put the whole array in **local memory**. "Prefetched cache_slot" was then read back
from local at ~the same latency as the global load it replaced. NCU (lineinfo, `--set full`,
no-write b1024): it was the **#1 `long_scoreboard` site** (`:1043`, `is_valid` reading
`cache_slot`, 34.3% of long_scoreboard) plus **3.3 MB local-memory traffic** (1.46M ld /
1.80M st sectors).

**Fix:** convert the ring to a register-resident **shift-register** (`head_meta[]`) indexed
ONLY by compile-time constants (`head_meta[0]`, `head_meta[k]` in `#pragma unroll`,
`head_meta[STAGES-1]`) so SROA keeps it in registers; the runtime `slot` is kept SEPARATE and
only addresses the smem ring (a runtime offset into shared memory is not local memory). The
physical smem layout (unit `u` in buffer `u % STAGES`) and cp.async FIFO depth are unchanged →
bit-identical.

### Results (no-write prev_k=0, bf16, mtp=6, mw=16, `bench_checkpointing_ssu.py` CUPTI)

| variant | b512 | b1024 |
|---|---:|---:|
| trimmed ring, `meta[]` in LOCAL mem | 74.2 | 142.2 |
| **register-resident shift-register** | **68.4** | **130.8** |
| Δ | **−7.8%** | **−8.0%** |
| triton-replay-pm (bar) | 41.9 | 75.3 |

NCU diff (lineinfo `--set full`, no-write b1024, `checkpointing_ssu_main_kernel`):

| metric | `meta[]` LOCAL | shift-register |
|---|---:|---:|
| duration | 131.8 µs | 121.3 µs |
| registers / smem / blocks | 87 / 38.4 KB / 5 | 87 / 38.4 KB / 5 |
| long_scoreboard | 37.0% | 27.7% |
| local spill traffic (ld/st sectors) | 1.46M / 1.80M | 0 / 0 |
| INT compute (PC samples) | 49.7% | 44.2% |
| tensor core | 4.9% | 9.4% |

- Occupancy unchanged (5 blocks, smem-bound) — this was a **latency** fix, not occupancy.
- Bit-exact across the full suite (persistent, two_kernel mhc1/2/4, d_split2, pipeline_stages2).
- New #1 `long_scoreboard` = `:1067`, `mc_of`'s `prev_ptr[cache_slot]` reload (the
  recompute-vs-store follow-up). Other top sites are now genuine compute (HMMA
  `common.cuh:1279` 24.9%, cumAdt STS `main.cuh:267` 16.9%).

### Follow-ups landed: prev_k in entry + CB→smem (B200, 2026-06-26)

Two latency cuts on top of the shift-register, no-write b1024 / b512 (`bench_checkpointing_ssu.py`
CUPTI; ncu = lineinfo `--set full` b1024):

| step | b512 | b1024 | ncu dur | long_scoreboard | notes |
|---|---:|---:|---:|---:|---|
| shift-register | 68.4 | 130.8 | 121.3 | 27.7% | (above) |
| **+ prev_k in entry** | 66.5 | 126.2 | 116.8 | 21.2% | prev_num_accepted loaded ONCE at fetch |
| **+ CB→smem cp.async** | **64.8** | **120.8** | **110.85** | 15.9% | CB A-operand from smem, not LDG |

- **prev_k → `HeadMetaSSU` (1b).**  Was the #1 long_scoreboard site (`mc_of`/`derive_head` reloaded
  `prev_num_accepted[cache_slot]` 3–5×/unit on the critical path).  Resolved at `fetch` (with
  cache_slot, prefetched STAGES ahead, guarded `!= pad_slot_id`); `mc_of`/`derive_head` read the
  register.  +1 int/entry, still register-resident.  INT compute 44.2→35.5%.
- **CB → smem (`load_cb_async`).**  `load_cb_fragA` reads `Pack[lane]` *by lane*, so all NUM_WARPS
  warps read the SAME 32 Packs → the just-in-time path did **128 redundant LDGs** for 32 unique
  Packs, and the output HMMA stalled on them (`common.cuh:1279`, 27% of long_scoreboard).  Now W3
  (idle in `load_x`'s post-gdc set) cp.async's the fragA-native blocks into 2 tiny smem buffers
  (cb_new ≤512 B, cb_old ≤512 B; +2 KB total, **occupancy held at 5 blocks**) inside the post-gdc
  group; consume is an LDS.  cb_old gated by `!must_checkpoint` (mirrors old_B).  CB HMMA stall gone.

**Cumulative this session: b1024 142.2→120.8 µs (−15.0%), b512 74.2→64.8 µs (−12.7%).**  New #1
long_scoreboard is `load_cumAdt`'s SYNCHRONOUS LDG+STS (`main.cuh:274`, 32%) — cumAdt is the last
post-gdc input not on cp.async.

### Full write/no-write comparison — we still LOSE to the monolith on no-write (B200, e829c35e)

Whole-kernel CUPTI (`bench_checkpointing_ssu.py`, mtp=6, mw=16, bf16; `cuda-incr-2k` = precompute +
persistent main, the precompute overlapped under PDL).  prev_k 0/8 = no-write, 12/16 = write
(must_checkpoint = prev_k+6 > 16).  **Best per column in bold.**

**b512**

| kernel | nw0 | nw8 | wr12 | wr16 |
|--------|----:|----:|-----:|-----:|
| cuda-incr-2k (ours)  | 64.8 | 75.4 | 95.8 | 97.1 |
| cuda-incr (monolith) | 62.6 | **63.0** | **82.3** | **82.3** |
| triton-replay        | 72.1 | 77.9 | 139.9 | 141.6 |
| triton-replay-pm     | **41.9** | **45.3** | 94.9 | 95.7 |

**b1024**

| kernel | nw0 | nw8 | wr12 | wr16 |
|--------|----:|----:|-----:|-----:|
| cuda-incr-2k (ours)  | 120.8 | 141.5 | 184.1 | 187.0 |
| cuda-incr (monolith) | 113.3 | **114.1** | **159.6** | **159.7** |
| triton-replay        | 130.6 | 152.3 | 225.0 | 227.3 |
| triton-replay-pm     | **75.5** | **80.3** | 184.4 | 186.6 |

**The 2k persistent main is SLOWER than the monolith on the no-write path it was built to win**
(b512 nw0 64.8 vs 62.6; b1024 nw0 120.8 vs 113.3) and the gap widens with prev_k (nw8: 75.4 vs 63.0;
141.5 vs 114.1).  triton-pm dominates no-write (b1024 75.5).  On write the monolith is fastest and
we ≈ triton-pm.  This contradicts the original B300 "Why" table (which claimed the 2k crushed
triton-pm on write) — either a B300→B200 inversion or a structural regression in the persistent
pipeline.

**RESOLVED — the depth-2 pipeline itself is the regression.**  Setting `MAIN_STATE_PIPE=1`
(single-buffer, no cross-iteration overlap; all the latency cuts above still apply — they're
pipe-independent) is **~33% faster**:

| no-write | pipe=2 | **pipe=1** | monolith | triton-pm |
|----------|------:|-------:|--------:|----------:|
| b512 CUPTI  | 64.8 | **49.0** | 62.6 | 41.9 |
| b1024 CUPTI | 120.8 | **90.0** | 113.1 | 75.5 |
| ncu dur (b1024) | 110.85 | **83.4** | — | — |
| registers / smem / blocks / occ | 88 / 40.4 KB / 5 / 28.7% | **64 / 20.2 KB / 8 / 47.2%** | — | — |

pipe=1 has **zero spills** (maxnreg(64) → exactly 8 blocks) and flips us from *losing* to the
monolith to *beating* it (b1024 90 vs 113).  Why depth-2 loses: the `state` buffer is 16 KB of the
20 KB bundle, so double-buffering it pushes smem 20→40 KB → **5 blocks instead of 8**.  On this
kernel **occupancy (more warps to hide the state reload) beats prefetch (a cp.async ring at half the
warps)** — the monolith was implicitly right to single-buffer.  This also **supersedes Plan B
(state-only pipelining)**: state is the 16 KB pole, so any depth-2 variant that double-buffers it
lands ≥28 KB / ≤7 blocks — still worse than depth-1's 20 KB / 8 blocks.  **depth-1 is optimal here.**

**cumAdt → cp.async (`load_cumAdt_async`)** landed on top of this (the last synchronous post-gdc
load; was the #1 long_scoreboard at pipe=2, 32%).  At pipe=2: b512 64.8→62.9, b1024 120.8→117.25,
ncu 110.85→107.46; the cumAdt STS stall is gone, leaving `wait` (20%) + scattered per-work-unit
address-arithmetic long_scoreboard — i.e. pipe=2 is now drained of memory stalls and PURELY
occupancy-bound at 5 blocks, still ~24 µs (ncu) behind pipe=1.  cumAdt async is pipe-independent so
it helps pipe=1 too (it was 14.8% of pipe=1's long_scoreboard).

### Tried + REVERTED: warp-templated head_loop (B200, 2026-06-27)

To kill the per-iteration warp-dispatch branch (`post_gdc`'s `if (warp==X)`, ~5% of the `wait`
bucket = ~1% of total), templated the whole loop on `int WARP` (`head_loop<…,WARP>`) and dispatched
once at kernel entry, so the loader guards fold to `if constexpr`.  **Net REGRESSION at pipe=2:**

| no-write | pipe=2 +cumAdt | + warp-templated head_loop |
|----------|---------------:|---------------------------:|
| b512 CUPTI  | 62.9 | 73.3 |
| b1024 CUPTI | 117.25 | 143.1 |
| ncu dur (b1024) | 107.46 | 136.54 |

ncu root cause: **`no_instruction` jumped to 31.8%** (the new #1 stall; issue rate 39.9→30.9%,
eligible warps 0.63→0.48).  Templating the *whole* loop 4×-inlines the heavy tensor-core path
(replay/output, which never branches on warp — pure tid arithmetic), so four distinct instruction
streams go resident → **i-cache thrashing**.  Registers 87→96 (hit the maxnreg cap), **no spills**,
occupancy unchanged (5 blocks) — confirming it's purely instruction fetch, not regs/occupancy.

**Lesson:** warp specialization via whole-loop templating is wrong when the warp-uniform code (the
MMA) dwarfs the warp-divergent code (the loaders).  The branch it removes (~1%) is far cheaper than
the i-cache cost of 4×ing the MMA.  A per-call dispatch (template only the loaders, MMA shared,
runtime warp) avoids the bloat but keeps a per-iteration dispatch — not worth it for ~1%.  **Reverted
to the cumAdt-async version (62.9 / 117.25).**  The warp branch is an occupancy symptom; the real
lever remains pipe=1.

### D_val → head_meta, and the barrier investigation (B200, 2026-06-27)

After reverting the warp-templating, profiled the no-write **barrier** stall (22.8%, the #1 stall at
pipe=2, the `:1222` "publish drained bundle" `__syncthreads`).

**D_val → `head_meta`:** `D_val = params.D[first_head]` was the only un-prefetched output input — a
per-work-unit global LDG in `compute_output_and_store`, and ncu pinned 81.8% of the barrier to it.
Hoisted it into the entry (resolved at `fetch`, like `prev_k`; +1 float/entry).  CUPTI no-write:
b512 62.9→61.6, b1024 117.25→114.3 (−3 µs); ncu (prev_k=1) 110.6→108.8.  **But the barrier %
didn't move (22.8→23.0)** — confirming D_val was a *compiler-hoist artifact* (the latency-tolerant
LDG was scheduled just before the barrier, so the PC parked there during the rendezvous; not the
cause).  Real −3 µs from cutting `wait`/long_scoreboard, though.  KEEPER.

**The actual barrier laggard (prev_k>0):** with D_val gone, the #1 long_scoreboard (40%) resolves to
`load_head:201` — `old_cumAdt_slot[prev_k-1] = oca_ptr[...]`, a **synchronous, single-lane
(`warp==0 && lane==0`), uncoalesced** global load of the β-decay tail, sitting in the *prefetch*
path.  One lane does a ~700 ns LDG → that lane is the laggard → all 127 other threads idle at the
`:1222` rendezvous.  That's how a single scalar becomes ~23% barrier.  DRAM is only **38.9%** (not
BW-bound), so it's pure latency-skew, not a memory wall.  The full `old_cumAdt` smem load
(`load_old_dt_cumAdt`) is **write-path only**; no-write folds the window into precomputed `cb_old` and
needs just this tail — so there's no smem copy to reuse, the gmem load is the source.
**Fix (in progress):** hoist `old_cumAdt[prev_k-1]` into `head_meta` (resolve at fetch, same pattern
as prev_k/D_val) → deletes the synchronous single-lane LDG, which feeds *both* the long_scoreboard
and the barrier.  prev_k>0-specific (guarded off at prev_k=0, which is why prev_k=0 is already faster).

### Next levers

- **Flip to pipe=1** (the measured 33% win above) and attack ITS #1 stall — the under-prefetched meta
  fetch (`:1109`, depth-1 prefetches meta only 1 ahead) — with a **deeper register-only meta ring**
  (decoupled from the state-ring depth; 3 ints/entry, zero smem cost).
- The warp-branch and HMMA stalls are NOT worth chasing on pipe=2 (occupancy-bound at 5 blocks).

## TODO

- [ ] **Pad-slot test coverage for the N-stage ring.** The persistent test runs
  `state_batch_indices=None` (no pads), and no test injects `pad_slot_id`, so the
  mid-loop skip path (`cache_slot == pad_slot_id` → `process_head` skipped between valid
  tiles, empty `__pipeline_commit()` holding the FIFO slot) is **untested**.  The
  out-of-range form of the skip *is* exercised (CTAs grid-stride past their work-units →
  epilogue empty commits) and passes, so the empty-commit + `wait_prior(STAGES-1)`
  accounting is validated; the gap is specifically a pad *between* valid work-units.
  Add a variant to `test_persistent_main_matches_monolithic` with some
  `state_batch_indices` entries = `pad_slot_id` (reference monolith skips them too) to
  cover it.  Believed-correct (skip is CTA-uniform → barriers stay matched), just unproven.

## Tests

```bash
uv run pytest tests/mamba/test_checkpointing_ssu.py -v -s -x -k "two_kernel or persistent"
```
Don't run the full mamba suite — it's long.

## Ncu
This runs the no-write benchmark with ncu
```bash
export FLASHINFER_JIT_LINEINFO=1 SSU_TAG=22.0a BATCH=1024 MTP=6 WINDOW=16 DTYPE=bf16 PNAT=0.1
${CUDA_HOME}/ncu --target-processes all --set full --import-source yes \
    --kernel-name "regex:checkpointing_ssu_(precompute|main)_kernel" \
    --launch-skip 0 --launch-count 100 -f \
    -o /home/scratch.ishovkun_gpu/benchmarks/mamba_decode/ssu_2k_v${SSU_TAG}_b${BATCH}_${DTYPE}_${MTP}_pnat${PNAT} \
    uv run python benchmarks/bench_checkpointing_ssu.py \
    --warmup 0 --iters 1 --no-l2-flush --no-cuda-graph \
    --batch-sizes ${BATCH} --mtp-lengths ${MTP} --max-window ${WINDOW} --state-dtypes ${DTYPE} \
    --prev-tokens-fracs ${PNAT} \
    --output -
```

# Non-mixed single-mode bench

```bash
export FLASHINFER_JIT_LINEINFO=1 SSU_TAG=22.0 BATCH=1024 MTP=6 WINDOW=16 DTYPE=bf16 PNAT=0.1
    uv run python benchmarks/bench_checkpointing_ssu.py \
    --batch-sizes ${BATCH} --mtp-lengths ${MTP} --max-window ${WINDOW} --state-dtypes ${DTYPE} \
    --prev-tokens-fracs ${PNAT}
```
