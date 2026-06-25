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

## Tests

```bash
uv run pytest tests/mamba/test_checkpointing_ssu.py -v -s -x -k "two_kernel or persistent"
```
Don't run the full mamba suite — it's long.
