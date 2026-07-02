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

## Tests

```bash
uv run pytest tests/mamba/test_checkpointing_ssu.py -v -s -x -k "two_kernel or persistent"
```
Don't run the full mamba suite — it's long.

## Ncu
Source-correlated profile of the precompute + main kernels.  **PNAT is now the ABSOLUTE
previously-accepted-token count** (`--pnat`, comma list; clamped to `[0, WINDOW]`): PNAT=0 → no-write;
`PNAT + MTP > WINDOW` → the write/checkpoint path.  `--set full --import-source yes` gives the
stall/PC-sampling metrics and needs `FLASHINFER_JIT_LINEINFO=1` (NOT `JIT_DEBUG` — see CLAUDE.local.md).
`--warmup 0 --iters 1 --no-l2-flush --no-cuda-graph` so ncu profiles a single clean launch each.
```bash
export FLASHINFER_JIT_LINEINFO=1 SSU_TAG=30 BATCH=1024 MTP=6 WINDOW=16 DTYPE=bf16 PNAT=0
${CUDA_HOME}/ncu --target-processes all --set full --import-source yes \
    --kernel-name "regex:checkpointing_ssu_(precompute|main)_kernel" \
    --launch-skip 0 --launch-count 100 -f \
    -o /home/scratch.ishovkun_gpu/benchmarks/mamba_decode/ssu_2k_v${SSU_TAG}_b${BATCH}_${DTYPE}_${MTP}_pnat${PNAT} \
    uv run python benchmarks/bench_checkpointing_ssu.py \
    --warmup 0 --iters 1 --no-l2-flush --no-cuda-graph \
    --batch-sizes ${BATCH} --mtp-lengths ${MTP} --max-window ${WINDOW} --state-dtypes ${DTYPE} \
    --pnat ${PNAT} \
    --output -
```
Read the report with the ncu_reader (CLAUDE.local.md), e.g. by kernel-name substring:
```bash
uv run /home/scratch.ishovkun_gpu/code/ncu_reader/ncu_reader.py -i <report>.ncu-rep -f main
```

# Single-config bench (CUPTI timing)

One (batch, mtp, pnat, dtype) point via CUPTI (auto-falls back to CUDA events).  `--pnat` = absolute
PNAT list (comma-sep); `--mtp-lengths` = NPREDICTED.  The output table's PNAT column is `pnat`.  Runs
`cuda-incr` (monolith) + `cuda-incr-2k` (operand-swap split) + the Triton refs.
```bash
export FLASHINFER_JIT_LINEINFO=1 BATCH=1024 MTP=6 WINDOW=16 DTYPE=bf16 PNAT=0
uv run python benchmarks/bench_checkpointing_ssu.py --cupti \
    --batch-sizes ${BATCH} --mtp-lengths ${MTP} --max-window ${WINDOW} --state-dtypes ${DTYPE} \
    --pnat ${PNAT}
```

# PNAT sweep + plot (collect script)

`collect_checkpointing_ssu_runs.py` sweeps **PNAT on the x-axis** at fixed **NPREDICTED** — one line per
(kernel, npredicted) — and writes a CSV + PNG to `mamba_decode/` and `mamba_decode/img/` (the user runs
an external viewer there, so pick a fresh `--tag`, don't clobber).  `--pnat`/`--npredicted` are comma
lists; `--tag N` suffixes the output `_vN`.  Plots cuda-incr-2k / cuda-incr / triton-incr (kernel→color,
npredicted→linestyle); other CSV impls (triton-replay-pm etc.) are collected but not drawn.
```bash
uv run python benchmarks/collect_checkpointing_ssu_runs.py \
    --batch-size 512 --state-dtypes bf16 --cupti \
    --pnat 0,1,4,8,12,14 --npredicted 4,8 --max-window 16 --tag 25
```
Re-plot from an existing CSV without re-running: `--plot-only <csv>`.

# Mixed benchmark
```bash
export FLASHINFER_JIT_LINEINFO=1
SSU_CUPTI_DEBUG=1 uv run python benchmarks/bench_ssu_checkpoint_mixed.py \
    --batch-sizes 1024 -K 1 --with-conv1d --output-prefix -
```
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
**Fix v1 (meta-hoist) → REVERTED:** hoisted `old_cumAdt[prev_k-1]` into `head_meta` (resolve at
fetch, like prev_k/D_val).  Bit-exact, but **regressed prev_k=0 by +11 µs** (b1024 114→125) — the
case that doesn't even use it.  A/B (neuter the fetch load → 114; full → 125) isolated it to the
guarded global-load chain added to `fetch`: **no register/occupancy/spill change (87/5/0)**, pure
**codegen/i-cache** — `fetch` is inlined in the hot loop and at 5 blocks it won't absorb extra code
(same failure mode as the warp-templating).  Reverted.

**Fix v2 (cp.async in place) → LANDED.**  Kept the load in `load_head` (already the prefetch path,
no fetch bloat) but swapped the synchronous single-lane LDG→STS for `__pipeline_memcpy_async` — W0
issues + continues instead of blocking, and the tail drains with the bundle.  **One-line change, big
win on the no-write-with-history path:**

| no-write b1024 | nw0 (pk=0) | nw8 (pk=8) | | b512 | nw0 | nw8 |
|----------------|-----------:|-----------:|-|------|----:|----:|
| `e829c35e`     | 120.8 | 141.5 | | | 64.8 | 75.4 |
| + D_val + cumAdt-async + **cp.async-tail** | **112.5** | **118.3** | | | **61.1** | **65.1** |
| Δ | −8.3 | **−23.2** | | | −3.7 | **−10.3** |

ncu (prev_k=8): **barrier 22.8%→9.6%** (the laggard is gone), `:201` old_cumAdt dropped out of the
long_scoreboard top sites entirely; 87 regs / 5 blocks unchanged; prev_k=0 unchanged (in-place
cp.async never touches the common path).  We now **beat the monolith on nw0** (112.5 vs 113.3) and
nearly tie on nw8 (118.3 vs 114.1, was +27 behind).  Write path (wr12/16) untouched — cp.async-tail
is no-write-only.  **Lesson:** at 5 blocks, fix stalls *in place* (swap sync→async) rather than
adding hot-loop code (meta-hoist / warp-templating both regressed via codegen/i-cache).

### Next levers

- **Flip to pipe=1** (the measured 33% win above) and attack ITS #1 stall — the under-prefetched meta
  fetch (`:1109`, depth-1 prefetches meta only 1 ahead) — with a **deeper register-only meta ring**
  (decoupled from the state-ring depth; 3 ints/entry, zero smem cost).
- The warp-branch and HMMA stalls are NOT worth chasing on pipe=2 (occupancy-bound at 5 blocks).

## Why triton-pm wins no-write: it's the INDEX MATH, not the matmuls (B200, 2026-06-27)

Profiled triton-replay-pm's `_persistent_main_kernel` (no-write, prev_k=8, b1024) vs our
`checkpointing_ssu_main_kernel`, same `dump_int_overhead.py` static SASS analysis + per-kernel CUPTI.

**Clean per-kernel CUPTI (no ncu artifacts) — the gap is the MAIN, not precompute, not PDL:**

| | precompute | main | whole path |
|---|---:|---:|---:|
| ours (cuda-incr-2k) | 13.6 µs | ~110 µs | 120.7 |
| triton-pm | 14.9 µs | **~70.6 µs** | 79.0 |

Our precompute is *faster* (13.6 < 14.9); PDL overlap is ~3 µs (irrelevant at b=1024).  All ~40 µs is
the main.  (NB: ncu base-clock durations LIE here — they show our main faster; trust the CUPTI.)

**Same memory, 3.3× the instructions.**  DRAM read is identical (318.5 vs 317.8 MB) — same math, same
data.  But: ours 50.06 M dynamic instructions vs triton 15.61 M; static SASS ours 3432 vs 1840.

| static SASS | triton-pm | ours |
|---|---:|---:|
| total | 1840 | 3432 |
| **INT** | 555 (30%) | **1831 (53%)** |
| UNIFORM | 232 (13%) | 183 (5%) |
| LD/ST | 245 | 696 |
| TENSOR | 108 (6%) | 54 (2%) |

**It's address math, on the wrong datapath.**  Neither kernel is matmul-bound (tensor 2–6%).  We
execute 3.3× more INT (IMAD/LEA/LOP3/SHF) — almost all **CuTe per-access swizzle/layout/pointer
recomputation** (`pointer_base:155` 220 INT @4%, `stride/swizzle/tensor_impl/copy` ~200, plus our own
`main.cuh` load/output/loop lines), **re-run every work-unit**.  Triton's heavy INT is **hoisted** —
its 96-IMAD index block (`:1352`) and swizzle (`:2168`) sit at ~0% time (computed once); its hot
samples are loads.  And triton offloads warp-uniform address math to the **uniform datapath** (13% vs
our 5%; `UIMAD`/`ULEA`/`ULDC`), keeping the main pipe free.

**The 3× LD/ST is the same root.**  Breakdown: `LDSM` (cooperative-MMA ldmatrix) is *equal* (3.5 vs
2.9%) — not our problem.  Our excess is `LDC` (param re-loads from constant mem: **6.1% vs 0.2%**) +
`LDS` (smem operand re-reads: **5.6% vs 0.6%**).  We re-fetch strides/pointers and re-read swizzled
smem per access because CuTe recomputes addressing AND our **87-register budget can't cache it**;
triton's **190 registers** hold params+operands resident → near-zero LDC/LDS.

**Reframe:** more *registers* (to cache params + hoist invariant addressing) is what cuts the
INT/LDC/LDS count — not occupancy.  triton trades occupancy (10%) for registers (190) + hoisting; we
do the opposite (28% / 87).  For this address-heavy memory-streamer, triton's bet wins.

### Exact executed-instruction count vs triton — the north star (B200, 2026-06-29)

Replaces the rough 50.06M/15.61M above with the exact **dynamic executed** mix
(`sass__inst_executed_per_opcode_category`, ncu `--metrics`, nw0 b1024).  Ours = current
committed main (C@state A-operand hand-roll + 2-way accumulator split, 99.9 µs);
triton = `_persistent_main_kernel` nowrite-half (61 µs; the 4.8 µs launches are the empty
write-half at prev_k=0).

| executed instructions | ours (split) | triton-pm | ratio |
|---|---:|---:|---:|
| **total** | **49,410,880** | **14,139,640** | **3.49×** |
| **integer** | **26,420,736 (53.5%)** | **5,745,000 (40.6%)** | **4.60×** |
| load/store | 7,227,520 (14.6%) | — | |
| uniform datapath | 5,395,328 (10.9%) | — | |
| floating point | 3,801,088 (7.7%) | — | |
| control | 3,820,800 (7.7%) | — | |

Key reframing: **triton is also integer-heavy (40.6%)** — it doesn't avoid INT, it executes
**3.5× fewer instructions total** (4.6× fewer integer).  Our IPC is *higher* (we're only 1.64×
slower on 3.5× more instructions → triton runs lower-IPC / lower-occupancy).  So the
**optimization north star is the instruction count itself**: drive total 49.4M → 14M, integer
26.4M → 5.75M.  The A-hoist + acc-split shaved ~5 µs of stalls but barely dented the *count*
(both were latency/ILP wins, not instruction-count wins) — the 3.5× gap is structural:
per-work-unit CuTe addressing recompute, small tiles (more loop iterations), and the
warp-specialized prefetch.  Closing it is the staged plan below.

### SASS dumps for source-correlated comparison (B200, 2026-06-30)

Captured paired `--set full --import-source yes` ncu reports + extracted SASS for the **exact
profiled no-write main kernels** (b1024, mtp=6, mw=16, bf16, prev_k=0), to diff our SASS against
triton-pm's and attribute the 4× body-size gap to source lines.  All artifacts +
provenance/commands in:
**`/home/scratch.ishovkun_gpu/benchmarking/mamba_decode/README_sass_compare.md`**

- `{ours,triton_pm}_main_nw0_b1024.ncu-rep` — full reports (Nsight Compute GUI: correlated
  Source/SASS + PC stalls).
- `*.sass.txt` — authoritative SASS from the report (exact profiled cubin):
  `offset | SASS | #PC-samples | top-2 stalls`.
- `*.sass_src.txt` — `nvdisasm -gi`, SASS interleaved with `//## File … line N`
  (ours → `kernel_checkpointing_ssu_main.cuh`; triton → `replay_selective_state_update.py`).
  Offsets cross-reference the `.sass.txt` by `/*offset*/`.

Static SASS (matches the dynamic north star): **ours 3440 vs triton 856 = 4.0× bigger body**,
both ~50% integer (ours 53.2%, triton 50.8%).  So triton does *not* avoid integer ops
proportionally — its kernel body is simply ~4× smaller (fewer unrolled iterations / bigger
per-thread tiles / hoisted addressing).  Triton no-write half = grid 888 / 168 regs / 60.7 µs;
ours = grid 2368 / 90 regs / 100.7 µs.

### SASS-diff findings — the 3.5× gap is `prefetch_async` + CuTe-addressed `process_head` (2026-06-30)

Per-source-line **dynamic executed** attribution (ncu "Instructions Executed" × nvdisasm `-gi`
line map; totals reconcile to the north star 49.41M / 14.14M).  Lines = `kernel_checkpointing_ssu_main.cuh`.

**OURS — 49.41M executed (53% int), concentrated in TWO inlined helpers in `head_loop`:**

| line | helper | exec | %tot | int | ldst | tc | fp |
|---|---|---:|---:|---:|---:|---:|---:|
| **1228** | `process_head<false>` (the no-write compute) | **19.1M** | **39%** | 10.46M | 3.40M | 1.22M | 2.31M |
| **1240** | `prefetch_async` (steady bundle stager) | **14.6M** | **30%** | 8.85M | 2.32M | **0** | **0** |
| 1237 | `fetch_head_meta` | 3.56M | 7% | 0.98M | 0.53M | 0 | 0 |
| 1196 | `compute_output_and_store<false>` | 1.80M | 4% | 0.95M | 0.33M | 0.09M | 0.18M |
| 1221/1216/1220/1219/1223 | driver loop / syncthreads / valid checks | ~3.6M | 7% | mostly int | | | |

The two hot lines = **68% of executed, 73% of all integer** (19.3M of 26.4M int).

**TRITON — 14.14M executed (41% int), spread thin (no line > 12%):**

| line | what | exec | %tot | int | ldst | tc | fp |
|---|---|---:|---:|---:|---:|---:|---:|
| 1903 | `tl.dot(C_tile, stateᵀ)` (C@state) | 1.70M | 12% | **0** | 0.52M | 1.05M | 0 |
| 1858 | `C_tile = tl.load(...)` | 0.92M | 6% | 0.38M | 0.44M | 0 | 0 |
| 2168 | `tl.range(...)` persistent-loop header | 0.87M | 6% | 0.40M | 0.03M | 0 | 0 |
| 1793 | `state = state_tma.load(...)` | 0.85M | 6% | 0.21M | 0.17M | 0 | 0 |
| 1909 | `tl.dot(CB, x_window)` (token) | 0.82M | 6% | 0.03M | 0.26M | 0.13M | 0.26M |
| 2172 | unflatten `pid_h = tile//(...)` | 0.79M | 6% | 0.69M | 0 | 0 | 0 |

**Two structural differences, both attackable:**

1. **`prefetch_async` (line 1240) is 30% of executed / 8.85M integer with ZERO matmul/fp — a stage
   triton does not have.**  It is *pure address arithmetic + LDGSTS* to hand-stage the
   gmem→smem bundle (state/C/x/cumAdt/old) with CuTe per-element swizzled addressing, re-derived
   every work-unit.  Triton loads operands straight into the matmul via TMA/coalesced `tl.load`
   (`:1793` state TMA, `:1858` C) with addressing the compiler hoists; its `num_stages` software
   pipeline + `warp_specialize` (`tl.range`, `:2168`) do the staging with ~0.4M int for the *whole*
   loop.  **This 14.6M is the single biggest, cleanest target.**
2. **Triton's matmuls carry ~0 integer** (`:1903` C@state = 0 int; `:1909` token = 32K int) — operand
   addresses computed once, tensor core streams.  OUR `process_head` (`:1228`) interleaves **10.46M
   integer** into the same matmul region — CuTe regenerates LDSM/operand addresses per MMA tile per
   work-unit.  Most of process_head's integer is hoistable addressing, not real compute.

So **both** hot lines are dominated by address math triton hoists/elides; the matmul FLOPs are tiny
in both.  Confirms the north star and pins the order: **(a) kill/shrink `prefetch_async`'s
per-work-unit re-addressing** (its 8.85M int has zero compute — biggest single win), then
**(b) hoist `process_head`'s CuTe operand addressing** (staged plan #1/#2 below).

Separately (static, not dynamic): `process_head<true>` (`:1224`, 440 insts), `replay_state`
(`:1165`, 281), `compute_output<true>` (`:1192`, 225) are **compiled-but-dead in no-write**
(`mc_of` always false at prev_k=0) — ~27% of the 3440 static SASS is dead write-path code inflating
i-cache / register pressure (90 regs → 5 blocks).  Triton sidesteps this with its **two-launch
write/nowrite split** (`WRITE_CHECKPOINT` constexpr).  A kernel-level `MUST_CHECKPOINT` template +
two launches would lean the no-write specialization (helps occupancy, not the dynamic count).

### Index-math simplification plan (staged)

1. **Hoist loop-invariant CuTe setup out of the per-work-unit path.**  `output_head_2k` /
   `compute_output_and_store` / `add_init_out_ring` / `replay_state_mma_ring` rebuild `make_tiled_mma`,
   `get_slice(tid)`, `partition_fragment_*`, the s2r `Copy_Atom`s, and the swizzled layouts **every
   work-unit** — all depend only on `tid`/types, not the work-unit.  Build them ONCE in `head_loop`
   before the steady loop, pass by ref.  Per-work-unit cost drops to base-pointer + slot offset.
   Target: collapse the `pointer_base`/`stride`/`swizzle` INT (re-run → once).
2. **Template / precompute the hot smem offsets.**  Where a swizzled address is `base + f(lane)` with
   `f(lane)` work-unit-invariant, precompute `f(lane)` once (or make it a `constexpr` offset table)
   and hand-roll the per-access add — instead of CuTe regenerating the swizzle (`LOP3`/`SHF`) each
   time.  Hot targets from ncu: the `add_init_out` C@state operand, the output store, `:457`/`:208`.
2b. **Kill the LDC param re-loads.**  Hoist `params.*_stride_*` / base pointers into local registers
   once at `head_loop` entry so nvcc stops re-issuing `LDC` per access (6.1%→~0).
3. **Uniform-ize the bases** so nvcc emits `UIMAD`/`ULEA` for the warp-invariant address parts (push
   work onto the idle uniform datapath), leaving only the per-lane swizzle on the main pipe.

Start with **#1** (biggest, cleanest): hoist the CuTe MMA/partition/layout objects to `head_loop`.

### Stage 0 → hoist REVERTED, only the reg-cap kept (2026-06-30)

**Update:** the manual byteoffA hoist was a dead end and is **reverted**.  A/B at cap 128 showed the
hoist is *net-negative* (nw0 108.8 with hoist vs **104.8 without**) — `compute_C_byteoffA` boxes in
the compiler, which hoists addressing better on its own once given register room.  **The entire win
is the reg-cap lift 96→128** (nw0 110.6→104.8, nw8 117.1→110.2, ~−6 µs; total executed −6.5%).
Kept: `SSU_MAIN_MAXNREG_PIPE=128` only.  **Lesson confirmed: integer COUNT does not move with
hoisting or registers — only with the access pattern (structural).**  Next = the x/old_x fusion
(3 matmuls → Triton's 2) to actually cut the access count.  Original mechanism notes below.

### (superseded) byteoffA hoist + reg-cap lift (mechanism proof) (2026-06-30)

Hoisted the C@state A-operand swizzled byte offsets (`byteoffA[8]`) out of the per-work-unit
`pipelined_kloop_gemm` up to `head_loop` (`compute_C_byteoffA` once per thread → threaded
`byteoffA_ext` through process_head→compute_output_and_store→output_head_2k→add_init_out_ring→
pipelined_kloop_gemm; monolith passes nullptr → unchanged).  Bit-exact (4 two-kernel tests pass).

**The reg cap, not the hoist, is the lever.**  ncu `--set full` int-mix, nw0 b1024:

| build | regs / blocks | total exec | integer | uniform | nw0 / nw8 µs |
|---|---|---:|---:|---:|---:|
| baseline (cap 96) | 90 / 5 | 49,410,880 | 26,420,736 | 4,572,800 | 110.56 / 117.10 |
| hoist, cap 96 | 89 / 5 | 50,564,928 | 25,577,472 | **6,618,688** | 110.56 / 117.10 |
| **hoist, cap 128** | **118 / 4** | **46,192,320** | 27,291,776 | **805,376** | **108.82 / 114.55** |
| hoist, cap 255 | 150 / 3 | — | — | — | 130.67 / 136.35 💀 |

- At **cap 96** the hoist had no room (90 + 8 offsets > 96) so the compiler **REMATERIALIZED**
  `compute_C_byteoffA` every work-unit onto the **uniform** pipe (uniform 4.57M→6.62M, total UP,
  timing flat).  Confirmed **NOT a spill** (ncu_reader "no register spills", all
  `*_register_spilling*` = 0, local ld/st sectors = 0).  Pure rematerialization.
- At **cap 128 / 4 blocks** the offsets stay live → the per-work-unit address recompute **vanishes**
  (uniform 4.57M→**0.8M**, total executed **49.4M→46.2M, −6.5%**), no spills, **nw0 −1.7 / nw8 −2.5 µs**.
  The int↔uniform split reshuffled (int rose as uniform collapsed) — the meaningful metric is
  **total executed**, which dropped.
- **Fully uncapping (255) is a trap**: the compiler grabs 150 regs → 3 blocks → +20 µs.
- Knob: `SSU_MAIN_MAXNREG_PIPE` (default **128**, was an implicit 96).  This is the Triton regime
  (more regs, fewer blocks) and the **right baseline to land the bigger B-operand + partition_C
  hoist on** — register headroom now exists to hold their offsets without rematerializing.

**Mechanism proven:** hoist-to-head_loop + register headroom eliminates per-work-unit CuTe address
recompute (byteoffA was a small slice — the large `pointer_base` 4.49M lives in the B operands /
partition_C, Stage 1).

### ROOT CAUSE — ldmatrix swizzle-PERIOD mismatch, not the swizzle (instruction-level SASS-diff, B200, 2026-06-30)

Read the paired SASS (`{ours,triton_pm}_main_nw0_b1024.sass.txt`) instruction-by-instruction.
The 4× integer gap is **operand smem stride vs the swizzle period**, decided per-fragment INSIDE
each `cute::gemm` — which is exactly why every hoist (byteoffA, head_loop, reg-cap) was noise.
**It is NOT the swizzle: both kernels use `Swizzle<3,3,3>`.**

**Quantified — integer/address insts interleaved *between* the LDSMs of a matmul burst:**

| | triton-pm | ours |
|---|---:|---:|
| LDSM total | 29 | 86 |
| int/addr insts between LDSM in a burst | **1** | **58** |
| → integer per ldmatrix | **0.03** | **0.67** |
| LDSM addressed via a uniform (UR) warp offset | **28/29** | **1/86** |
| distinct base regs | 10 (R120→8 frags, R119→8 frags) | 32 (~4 frags/base) |

Triton steady-state load (`:0x1bd0`): all 8 frags = `[R120 + UR9 + imm]`, imm ∈ {0,0x800,…,0x3800},
**zero** integer between the loads; `R120` (per-lane swizzle) computed ONCE, `UR9` (warp K-base) is uniform.
Ours hot burst (`:0xaf30` → `copy_sm75.hpp:172` → `main.cuh:1165`): a `SEL R81,0x20,-0x20` + `IMAD.IADD`
chain rebuilds a fresh base every 4 frags; addrs = base±{0,0x20,0x40,0x60} × {0,+0x400}.

**Bit-level why.** `Swizzle<3,3,3>` on bf16 has period `2^(B+M+S)=2^9` elem = **0x400 B**; it XORs
byte-bits {0x10,0x20,0x40} from source byte-bits {0x80,0x100,0x200}.
- **Triton** frag step 0x800 (and 0x2000) is *above* both XOR regions ⇒ `swz(x+0x800)=swz(x)+0x800`
  exactly ⇒ pure immediate, swizzle folded into the base once, still conflict-free. 8×0x800 = 0x4000 =
  the 16 KB state tile.
- **Ours** frag step 0x20 (one dstate K-tile = 16 bf16, in a `[T,dstate]` row-major operand) is *inside*
  the XOR region ⇒ adding it carries into bit 0x80, flipping the mask ⇒ `swz(x+0x20)≠swz(x)+0x20` ⇒ the
  conditional ±0x20/±0x40. Our `+0x400` step *is* an immediate (0x400 is above the swizzle); the mixed
  `±0x20 + imm-0x400` pattern is the period boundary showing through.

Plus triton offloads the warp-varying address part to the **uniform datapath** (28/29 vs our 1/86),
keeping the main integer pipe free.

**Why hoisting was structurally doomed.** The recompute is per-fragment, intrinsic to ONE matmul's
K-walk — not a per-work-unit invariant — so head_loop/byteoffA hoists cannot touch it (confirmed: count
didn't move). Same swizzle as triton ⇒ nothing to fix in the swizzle. Second, multiplicative factor:
**86 vs 29 LDSM** (3 dots vs triton's 2, smaller tiles, grid 2368 vs 888). Gap ≈ (~20× addr-int per
ldmatrix) × (~3× ldmatrix) = the 4×.

**Fix direction (a real operand-layout redesign, not a hoist).** Lay the matmul operands in smem so
EVERY MMA-fragment stride is ≥ one swizzle period (0x400 B): make the contraction dim (dstate) the
**major** smem dim with a fat stride, not the tightly-packed inner dim. That is what triton's
`state_tma.load` + `tl.trans(state)` give for free — dstate K-tiles land 0x800 apart, the XOR drops out
of the walk, ptxas folds everything into immediates on a single base. Co-design the cp.async/TMA store
layout with the ldmatrix read so the K-walk steps over whole periods. **Prototype + inspect ptxas on a
one-off (confirm the immediate-walk codegen) BEFORE touching the live path.**

### Prototype CONFIRMS the fix at codegen level (standalone, B200, 2026-06-30)

Standalone CuTe probe (`benchmarking/mamba_decode/cstate_swizzle_probe.cu`, copies only the
swizzle/layout helpers — does NOT include or touch the SHARED `_common.cuh`). Reproduces the exact
C@state matmul (A=C[16,128], B=state[N,128], `SM80_16x8x16_TN`, `Layout<Shape<_1,_4>>`, 8 K-tiles)
two ways; `cuobjdump -sass` on the sm_100a cubin:

| | baseline (K-inner, current) | **K-major + LDSM.T** |
|---|---:|---:|
| LDSM / HMMA | 16 / 8 | **16 / 8** (same) |
| distinct base regs | **8** | **2** (one per operand) |
| integer in burst / ldsm | **0.50** | **0.12** |
| LDSM uniform-addressed | 2/16 | **16/16** |

- **baseline** emits the af30 pathology verbatim — `SEL R28,R28,0xffffffe0` (±0x20), `SEL R6,R6,0xffffffc0`
  (±0x40), `IMAD.IADD`/`IADD3` chains building 8 bases interleaved with the LDSMs.
- **K-major** (store operands `rc<bf16,128(K),64(MN)>`, read via `make_swizzled_layout_rc_transpose`
  + transpose ldmatrix) → both operands `[R+UR+imm·0x800]`: ONE base + UNIFORM warp offset + immediate
  K-walk at 0x800 = 2× the 0x400 swizzle period. Triton's signature, **at identical LDSM/HMMA count**.
- Atom width matters: B with `U16x2_LDSM_T` (half payload) doubled the count to 24 LDSM; the
  width-matched `U16x4_LDSM_T` (B) / `U16x8_LDSM_T` (A) restore 16. (codegen-only probe; numerics
  not checked — irrelevant to the address-emission question.)

**Implementation cost discovered — the smem asymmetry:** swz_rc needs COLS ≥ ATOM_COLS(64), so K-major
stores the operand as `[K=128, MN→pad64]`.
- **state (B): FREE.** `[128,64]` = 8192 elem = 16 KB = exactly today's `[64,128]` state buffer.
- **C (A): 4× bloat.** `[16,128]`=4 KB → `[128,64]`=16 KB (M=16 pads to 64). At 8 blocks/20 KB this
  would wreck occupancy. So C can't naively go K-major; needs packing (e.g. share the 64-col atom across
  4 d-tiles / heads) or stays K-inner.
- C is the SHARED A-operand (loaded once, reused across N-tiles); state is reloaded per N-tile → state's
  LDSMs dominate the integer. **First move: take state K-major only (free smem, kills the dominant
  half), measure end-to-end; tackle C packing separately.** Also still TBD: the gmem→smem **store** must
  write K-major (transposed cp.async / TMA descriptor) — the probe used a flat fill, so store-side cost
  is unmodeled.

### 07-01: prototype reproduces Triton's compile-time LDSM addressing (K-major state + LDSM_T)

Prototypes in `benchmarking/mamba_decode/frag_proto/` (`gemm_cmp.cu` = full C@stateᵀ load+gemm+store,
`ldsm_addr.cu` = swizzle address probe, `frag_load.cu` = load-only). sm_100a, `cuobjdump` / `-Xptxas -v`.

**1. Swizzle choice is NOT an INT lever** (`ldsm_addr.cu`, state-A `[16,128]`, `LDSM_N x4`, 8 K-tiles):
`<3,3,3>` = 75 INT / `<3,4,3>` = 80 INT, **both 8 LDSM**. Larger period ⇒ finer per-row phase (m mod 8
vs m mod 4) ⇒ *more* distinct address offsets (`<3,3,3>`: 4 bases + reused `+0x800` imm + 2-bit SEL;
`<3,4,3>`: 7 regs via IMAD.IADD chains + 3-bit `R2P` phase). Confirms 06-30 "NOT the swizzle"; `<3,4,3>`
is a small INT *loss*. (Bank-conflict-freeness for the x4 read is a separate, runtime-only question.)

**2. What Triton actually does (verified from its TTGIR + SASS — NO transpose, NO K-major storage, NO
different formulation).** State smem = `#ttg.nvmma_shared<swizzlingByteWidth=128, transposed=false>`,
`[dim, dstate]` (dstate contiguous), TMA-loaded (`async_tma_copy_global_to_local`); the output is
`init_out = tt.dot(C_tile, tl.trans(state)) = C@stateᵀ = [T, dim]` — identical formulation + layout to
ours. The 4× is pure CODEGEN: **(a)** TMA does the gmem→smem addressing in hardware (≈0 int), and **(b)**
Triton lowers `local_load(nvmma_shared) → dot_op(mma)` to LDSM `[base + UR + immediate]` — base computed
ONCE, warp-varying part on the UNIFORM datapath, K-walk as **compile-time immediates** (step 0x800),
ZERO int between the 8 loads. Ours (`cute::copy` from `make_swizzled_layout_rc`) re-derives the swizzle
per K-tile on the main int pipe (`SEL ±0x20` + `IMAD.IADD` chains). So it is NOT the swizzle and NOT a
layout transpose — it's the ldmatrix ADDRESS EMISSION.

**3. We reproduced Triton's emission in CuTe** (`gemm_cmp.cu` `k_optimal`): store the state operand with
the contraction dim (dstate) **MAJOR** — physical `[dstate, dim]` — and read it via a transpose view
(`swz_rc_T`) + transpose ldmatrix (`SM75_U16x8_LDSM_T`). The K(dstate)-tile then walks the major
(physical-row) dim → step 0x800 ≥ Swizzle period (0x400) → the swizzle folds into the base and the
**K-walk becomes compile-time immediates**. SASS, state operand, side-by-side:

    k_optimal (ours):  LDSM.16.MT88.4  [R3+UR4]   [R3+UR4+0x800]   ...  [R3+UR4+0x3800]
    Triton:            LDSM.16.M88.4   [R120+UR9] [R120+UR9+0x800] ...  [R120+UR9+0x3800]

Base computed ONCE, warp-varying offset on the UNIFORM register (UR4), K-walk = compile-time immediates
(0x800·k), zero int between the 8 loads. Full C@stateᵀ (load+gemm+store, 4 warps):

| variant | total | INT | HMMA | LDSM | regs |
|---|---:|---:|---:|---:|---:|
| current `C@stateᵀ` (state=B x2, natural `<3,3,3>`) | 247 | 99 | 16 | 24 | 32 |
| transp `state@Cᵀ` (state=A x4, natural `<3,3,3>`) | 240 | 109 | 8 | 16 | 30 |
| transp `state@Cᵀ` (state=A x4, natural `<3,4,3>`) | 249 | 119 | 8 | 16 | 30 |
| **OPTIMAL `state@Cᵀ` (state=A, K-major + LDSM_T)** | 240 | **106** | 8 | 16 | **28** |

**Isolation masks the payoff (critical).** Even with the perfect `[base+UR+imm]` emission, `k_optimal`'s
total INT (106) is still ABOVE `current` (99): in a standalone kernel the compiler has the registers to
strength-reduce `current`'s natural addressing anyway. The 4× only shows up IN-CONTEXT (real kernel:
118 regs / 4 blocks + head-loop + 3 competing matmuls ⇒ no room to strength-reduce ⇒ the natural layout
emits the 58-int/burst recompute = the 200 `pointer_base`; K-major forces the immediate emission
*regardless* of pressure). So the prototype's job is done — the recipe is proven at the codegen level;
**only an in-context kernel run can show the speedup.** Prototype counts codegen faithfully (real gemm,
stored, no spills), not numerically verified.

**No runtime transpose.** The state cache is internal — store it **dstate-major**
`[cache_slots, nheads, dstate, dim]` and have every accessor (main read, replay, checkpoint write) use
that layout consistently. The cp.async load is then a plain contiguous copy into `[dstate, dim]` smem =
exactly the K-major smem the recipe needs, with **NO transpose on the load**; the `LDSM_T` is in-hardware
(free). The mtp≤8 HMMA-halve caveat still applies to the transposed output (NPRED=6→N-pad 8; gate on
`NPREDICTED ≤ 8`).

### LANDED: operand-swap output `[DIM,NPRED]` — correct + faster (B200, 2026-07-01)

Shipped the operand swap (candidate B) end-to-end. **Precompute** (`scale_store_cb`) stores CB/CB_old
**fragB-native** via the per-lane register select `{raw[0],raw[1],raw[4],raw[5]}` (proven bit-exact by
`frag_proto/cb_fragb_select_probe.cu`) — halves the CB store/traffic at mtp≤8. **Main** (`output_head_2k`,
`add_init_out_ring`, `add_cbx_swapped`, `add_D_skip_swapped`, `compute_z_gating_swapped`):
- OUT.1 `state@Cᵀ → [d,t]`: state=A (`SM75_U32x4_LDSM_N`), C=B (`SM75_U32x2_LDSM_N`), no transpose/layout
  change (state smem is already dstate-major), **reusing the shared `pipelined_kloop_gemm`** with A/B
  swapped (monolith's call untouched).
- OUT.2/3: x/old_x=A (`SM75_U16x8/U16x4_LDSM_T` transpose from the `[token,d]` smem), CB=B (fragB from smem).
- Epilogue on `[DIM,NPRED]`: decay broadcast over M-rows (`stride(0,1)`, `cumAdt[t]`), D·x/z via the `[d,token]`
  transpose views (scalar), **scatter store** (strided `[d,t]`→gmem `[token,d]`, no smem bounce).
- Both write + no-write share the machinery (write adds `store_state`, skips OUT.3), per user "swap both".

**Generic / d_split:** warps split M via `NUM_M_WARPS = D_PER_CTA/16`; at d_split=2 (D_PER_CTA=32) only 2
warps do the output MMA (guarded `if (m_active)`), the rest still do `store_state`/`store_old_x`. **mtp>8**
works: CB is built as a **per-output-N-tile `partition_fragment_B` ARRAY** (one m16n8 N-atom each — a single
multi-N-tile `partition_fragment_B` trips `make_fragment_like` on the identity tensor), filled by one
vectorized LDS then distributed (`raw.val[on·REGS_PER + r]`, typed from the fragment element =
`cutlass::bfloat16_t`, not `MMA_prop::operand_t`).

**Fixed IMA (mtp>8 × d_split=2 × grid-stride, 2026-07-02).** `collect_checkpointing_ssu_runs.py` b=512
crashed at mtp≥10: the `m_active` (`warp < NUM_M_WARPS=D_PER_CTA/16`) guard was on the `NUM_N_TILES==1`
branches only — the mtp>8 `==2` branches ran `add_init_out_ring`→`pipelined_kloop_gemm`→`ldmatrix_x4_addr`
on warps ≥ NUM_M_WARPS with an out-of-range `Shape<NUM_M_WARPS,1>` partition → OOB `__shared__` LDSM
(silent garbage at small batch; a hard fault once grid-stride kicked in, batch ≳ 74).  compute-sanitizer
pinned it to warp 2 at `common.cuh:1358` ← `output_head_2k` `==2` branch.  Fix: guard both `==2` branches
(OUT.1 + epilogues) with `m_active`, `store_state`/`store_old_x` stay cooperative — mirrors the `==1` fix.

**Validated (bit-for-bit vs monolithic, `test_checkpointing_ssu.py`):** `test_two_kernel_matches_monolithic`
(mhc {1,2,4}, k∈{0,2,T}, ±PDL), `test_two_kernel_pipeline_stages2`, `test_two_kernel_d_split2`; standalone
mtp=10/mw=16 (d_split=1) and mtp=16/mw=16 **d_split=2, mhc=4, grid-strided b=96** checks (k∈{0,8,16}) match
monolithic; `collect_checkpointing_ssu_runs.py b=512 bf16 --cupti` completes clean for mtp∈{4..16};
compute-sanitizer memcheck-clean (0 device faults). **Perf:** win confirmed by the user (exact CUPTI vs the
104.4/109.3 µs `c4ca6b72` baseline still TBD — rerun `bench_checkpointing_ssu.py` no-write b=1024 mtp=6 with
`FLASHINFER_JIT_LINEINFO=1`).

**Follow-ups:** ncu INT/HMMA/LDSM vs baseline; the D·x-from-x-A-fragment reuse (TODO below); mtp>8 loads x
twice (once per output N-tile — redundant but correctness-only path).

### Perf: closed most of the triton-replay-pm gap (B200, b=512, bf16, 2026-07-02)

Benchmark reworked — `collect_checkpointing_ssu_runs.py` now sweeps **PNAT** (previously-accepted tokens,
`--pnat 0,1,4,8,12,14`) at fixed **NPREDICTED** (`--npredicted 4,8`), one line per (kernel, npredicted);
`--tag N` suffixes the output `_vN`.  So these are **NOT** directly comparable to the old mtp-swept
b=1024 104.4/109.3 µs baseline (different batch + x-axis).

No-write (PNAT=0), b=512, bf16, CUPTI, iters=20 — median µs:

| NPREDICTED | incremental (triton) | cuda-incr (mono) | **cuda-incr-2k (swap)** | triton-replay-pm (north star) |
|---:|---:|---:|---:|---:|
| 4  | 89.7 | 62.7 | **50.6** | 41.1 |
| 6  | 91.7 | 62.6 | **51.5** | 41.7 |
| 8  | 98.5 | 64.8 | **51.7** | 42.1 |
| 16 | 99.5 | 75.6 | **64.5** | 47.2 |

The operand swap makes **cuda-incr-2k beat the monolithic by ~18–20%** (mtp=8: 51.7 vs 64.8) and beat every
Triton variant except triton-replay-pm.  Gap to the north-star **triton-replay-pm narrowed from ~1.5×
(mono) to ~1.24× (swap)** (mtp=8: 64.8→51.7 vs 42.1).  The residual ~1.24× is consistent with the 06-27→07-01
root cause: triton-pm's remaining edge is the **INDEX MATH** (compile-time-immediate LDSM addressing), which
the operand swap did *not* touch.

Write path (PNAT+NPREDICTED > max_window=16): checkpoint cliff to ~90 µs, roughly at parity with the
monolith — the shared replay + state-write dominates there and the swap's output savings are smallest (the
split-tax; confirmed NOT the d_split=2 idle-warp guard — d_split=1 was no faster).

### LANDED: meta pipeline — smem ring + register queue (B200, 2026-07-02, `512e9f1a` + `69cae8cd`)

ncu v30.1 (d_split=1) showed the steady loop latency-bound on **per-unit metadata**, not the 16 KB
state: `long_scoreboard` 26.1% #1, top sites = the `D_ptr[head]` refetch (22.9%) and the
`sbi[seq] → {prev_num_accepted, cache_buf_idx}` dependent chase (12.7%), issued only STAGES=2 ahead —
uncoverable at 25% occupancy.

Two-level fix (both occupancy-neutral, `d_split`/pipeline structure untouched):

1. **Smem meta ring** (`512e9f1a`, ncu v26/v27): `fill_meta(base)` resolves 32 units warp-wide
   (lane l → unit base+l) into a 512 B ring `{cache_slot int32, pnat, buf_read, D}`; refill every
   `META_RING−STAGES` units with slot wraparound. All 4 warps fill redundantly → each warp reads its
   own writes → `__syncwarp()` only, **no block barrier**; window overlap `[f, f+STAGES)` rewrites
   identical values (benign race). Ring sized 512 B deliberately: headroom to the 4-blocks/SM smem
   cliff is exactly 640 B/block (57,728 B/block ×4 on the 228 KB SM), so `tile` is re-derived and
   `cache_slot` stored int32 (pads canonicalized to −1). **82.0 → 79.1 µs**, long_scoreboard
   26.1 → 16.1%.
2. **Register queue** (`69cae8cd`, ncu v28): v27 showed the cost moved to the ring's LDS consumers
   (short_scoreboard #1 21.1%: `is_valid` ISETP 12.3%, cache_slot LDS 10.2%) — steady-state
   `wait_prior` retires instantly, hiding nothing. `meta_q[STAGES]` shift-register (constant-index,
   SROA) appended at the iteration bottom from `load_meta`, consumed as process-meta STAGES
   iterations later → `is_valid`/`derive_head`/prefetch address math read pure registers.
   **79.1 → 77.5 µs**; both v27 short_scoreboard sites gone (residual ISETP 8.3% at the bottom
   append); regs 119 → 125 (cap 128, no spills).

Also in `512e9f1a`: `derive_head` returns `HeadCoords`, derived ONCE per `prefetch_async` (was
re-derived in both gdc halves), load-free non-varlen; `j`→`tile`; single-letter rename sweep;
`test_two_kernel_meta_ring_refill` (CTA_PER_SM=1, ~64 units/CTA → two wraparound refills, random
per-slot pnat mixes write/no-write inside every window).

**Cumulative main-kernel: 82.0 → 77.5 µs (−5.5%); wall (both kernels) 97.25 → 93.95 µs; gap to
triton-replay-pm 1.24× → 1.19×** (b=1024 bf16 mtp=8 pnat=4).

ncu v28 residuals — the meta lever is EXHAUSTED, profile is now a broad floor:
- `short_scoreboard` 19.2%: FMUL@624 (`beta_extra` ← `old_cumAdt[prev_k−1]` LDS→`__expf` chain,
  22% of cat) + HMMA waits + `cuda_bf16` IMAD — pre-existing compute-consume chains.
- `long_scoreboard` 17.5%: prologue BAR@1283 (one-time fill+state chase per CTA, 17.4% of cat),
  IMAD@171 `ox_base` (17.2% — possibly LDC stride rematerialization at 125 regs), fill's own
  STS@1234 (9.1%, by-design amortized).
- `mio_throttle` 17.3% + `barrier` 11.2% + `wait` 13.9%: cp.async/LDSM issue pressure + pipeline
  sync at 25% occupancy — structural.

**beta_extra hoist tried (2026-07-02): wall-FLAT, but diagnostic.** Moving the
`old_cumAdt[prev_k−1]` LDS + `__expf` from the no-write epilogue up to `compute_output_and_store`
(ahead of the CB fragment loads) moved the FMUL stall site verbatim (624 → the hoist line, same
~20% of short_scoreboard) — ptxas keeps LDS→FMUL adjacent, and under 17% `mio_throttle` the LDS
waits on the saturated MIO queue wherever it issues.  ⇒ The output phase is **MIO-ISSUE-bound,
not scheduling-bound**: shuffling LDS consumers is dead; only REMOVING LDS traffic pays.
Kept anyway: regs 125 → 118 (headroom off the 128 cliff).

Next levers:
- **beta via the meta ring**: fetch the `old_cumAdt[cache_slot, buf_read, head, prev_k−1]` tail in
  `fill_meta` (3rd dependent LDG in the 32-wide chase, store `__expf` of it), consume as a register
  from `meta_q` → deletes the per-unit LDS+FMUL site AND the no-write pre-gdc tail cp.async (less
  LDGSTS → less mio_throttle).  Race-safe: checkpoint writes target buffer `1−buf_read`, reads
  `buf_read`.  Smem: pack `pnat|buf_read` into one int32 so the ring stays 512 B (cliff math holds).
- Occupancy (needs regs ≤ 102 **and** smem ≤ 45 KB simultaneously — hard).

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

- [ ] **D·x from the x A-fragment (operand-swap main).** Since x's A-fragment `x[d,j]` is
  already in registers after OUT.2, `D·x[t,d] = D·x[d, j=t]` could be pulled from that
  fragment instead of a fresh smem read — eliminating the D·x smem load entirely. Register
  reindex needed; defer unless ncu flags it.
