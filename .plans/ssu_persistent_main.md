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

## Choices (locked)

1. **Replace** the non-persistent main — the persistent loop subsumes it
   (`cta_per_sm` high ⇒ `grid == total_work` ⇒ one work-unit/CTA ⇒ identical).
2. **`cta_per_sm` via env** `FLASHINFER_SSU_MAIN_CTA_PER_SM` (read per-launch so the
   test can vary it; baked into a heuristic later).
3. **Ditch MHC** — work-unit = single head. Flatten **head-fastest**
   (`tile = (d_tile·batch + seq)·nheads + head`) so consecutive work-units are
   consecutive heads of one `(d_tile, seq)` → per-group `C`/`old_B` stay L2-hot.
4. **One-shot gdc_wait** — fires only on a CTA's FIRST outer iteration
   (`DO_GDC_WAIT` template param on `head_loop`, gates ONLY
   `cudaGridDependencySynchronize()`; the surrounding barriers that publish
   state/cumAdt/x stay on every iteration).

## Test (xfail-first)

- [ ] **T_PERSIST** `tests/mamba/test_checkpointing_ssu.py::test_persistent_main_matches_monolithic`
  — set `FLASHINFER_SSU_MAIN_CTA_PER_SM` small so **`work_units > occupancy · NUM_SMS`**
  (grid genuinely can't hold all work-units → each CTA grid-strides ≥2×). Assert
  bit-exact vs monolithic (`out`, `state`, `old_x`, `old_B`, `old_dt`, `old_cumAdt`)
  on BOTH the no-write and write paths. **Initially xfail** (no loop yet). *EASY*

## Steps

1. [ ] **T_PERSIST xfail** test as above. *EASY*
2. [ ] **Ditch the MHC knob**: launcher dispatches the main at MHC=1 only; remove
   `FLASHINFER_SSU_MAIN_HEADS_PER_CTA` env read, the auto-heuristic, and the Python
   `main_heads_per_cta` handle. (`head_loop`'s MHC>1 inner loop / STAGES=2 cross-head
   paths become dead at MHC=1 — leave correct-at-MHC=1, clean up in a follow-up.)
   `launch_checkpointing_ssu.cuh`, `flashinfer/...` Python wrapper. *MEDIUM*
3. [ ] **Env knob** `FLASHINFER_SSU_MAIN_CTA_PER_SM` + cached `NUM_SMS`
   (`cudaDevAttrMultiProcessorCount`, static like the precompute's `sm_count`).
   `launch_checkpointing_ssu.cuh`. *EASY*
4. [ ] **1D persistent grid**: main grid → `dim3(min(cta_per_sm·NUM_SMS, total_work))`,
   `total_work = D_SPLIT · batch · nheads`. Default `cta_per_sm` ⇒ `grid == total_work`.
   `launch_checkpointing_ssu.cuh`. *MEDIUM*
5. [ ] **Grid-stride loop + `DO_GDC_WAIT`** in `checkpointing_ssu_main_kernel`:
   wrap the per-work-unit body (unflatten head-fastest → d_tile/seq/head/group, per-slot
   setup, `head_loop`) in `for (int tile = blockIdx.x; tile < total_work; tile += gridDim.x)`.
   Dispatch `head_loop<DO_GDC_WAIT = (tile == blockIdx.x)>` (compile-time via if/else on the
   runtime first-iter check). Add `DO_GDC_WAIT` template param to `head_loop` gating only the
   `cudaGridDependencySynchronize()`. Loop-boundary `__syncthreads()` at the bottom so
   iteration N's output/store_old_x finish before N+1 overwrites smem.
   `kernel_checkpointing_ssu_main.cuh` (kernel body L763-848 + `head_loop`). *HARD*
6. [ ] **Make T_PERSIST pass**; confirm **T0** (`test_two_kernel_matches_monolithic`)
   and mw=16 still bit-exact at default `cta_per_sm` (grid == total_work). *MEDIUM*
7. [ ] *(follow-up, perf)* **Skip `C`/`old_B` reload** when the next tile shares
   `(seq, group)` — recovers MHC's smem-level reuse on top of L2. *MEDIUM*

## How to measure overlap (conv1d + precompute + main)

`SSU_CUPTI_DEBUG=1` with the mixed bench's `--with-conv1d` prints per-kernel
start/end timestamps (relative to the first kernel) to stderr:

```bash
SSU_CUPTI_DEBUG=1 uv run python benchmarks/bench_ssu_checkpoint_mixed.py \
    --kernels cuda-incr-2k,triton-replay-pm --batch-sizes 1024 --with-conv1d \
    --iters 5 --warmup 3 --output-prefix - 2>&1 | grep -E "CUPTI-DEBUG|start="
```
```
[CUPTI-DEBUG] tag=... iter=0 n_kernels=3
  start=  0.000 end=  3.840 µs  _causal_conv1d_update_kernel
  start=  2.528 end=  7.296 µs  _checkpointing_ssu_precompute_kernel...
  start=  3.649 end= 10.176 µs  _checkpointing_ssu_main_kernel...
```
`start` is relative to the first kernel. The main's `start` shows whether it
co-resides with conv1d (persistent target: main starts during conv1d, like triton-pm).

## Findings archive (B300, 2026-06-25)

**Per-head barriers** (no-write hot path): cut from 3→2/head (dropped the redundant
post-replay barrier; cumAdt batch-prefetch removed per-head STS + drain). NCU
(MHC=4 auto, b=512, no-write): duration 44.0µs, barrier 13.5%, `long_scoreboard`
35.4% (state load — the persistent target), occupancy 45%, eligible warps 0.93.

**Load balance**: per-head loads distributed one-tensor-per-warp (old_x→W0/W1 split
+ predicated to prev_k, x→W2, z→W3) — all single-warp (conflict-free). Earlier
CTA-wide loads introduced a 2-way LDGSTS conflict (d_split=2 half-width write); the
per-warp form removes it.

**No-write progression** (CUPTI, b=512, prev_k=0, mw=8): 51.5 (postbarrier) → 50.7
(CTA-wide x) → 50.9 (CTA all) → **48.8** (predicate+split+per-warp, new best).

**Target**: close the no-write gap to triton-replay-pm via persistence
(b=1024: 90.5 vs 75.4 = **15µs**; b=512: 50.5 vs 42.1 = **8µs**).

## Persistent main results @ b=1024 (mixed PNAT + conv1d, 2026-06-25)

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

### Next levers

- **cumAdt → cp.async:** fold cumAdt into the post-gdc cp.async group (same pattern as CB) — it's
  now the #1 long_scoreboard site (synchronous LDG+STS).
- **2 (Plan B):** state-only pipelining — ring holds only `state`, C/old_B/x/cumAdt
  single-buffered → smem ~40→~24 KB → **8 blocks**, to hide the residual long_scoreboard via
  occupancy.

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
