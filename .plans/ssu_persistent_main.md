# Plan: Persistent main kernel (checkpointing SSU two-kernel split)

## Goal

Generalize the main kernel into a **1D grid-stride persistent loop** over
`(d_tile, seq, head)` work-units, launching `min(cta_per_sm·NUM_SMS, total_work)`
CTAs instead of one CTA per work-unit. With a large enough `cta_per_sm` the grid
equals `total_work` and each CTA does exactly one work-unit → **bit-identical to
today's non-persistent main**. With a smaller `cta_per_sm` the main leaves SM
room to co-reside with conv1d (and the precompute) on the cliff.

`cta_per_sm` is a tunable env knob now (mirrors the precompute's
`FLASHINFER_SSU_HEADS_PER_CTA`); once the persistent kernel exists we sweep the
**precompute `hc` × main `cta_per_sm`** grid jointly to find the co-residency
balance, then bake a heuristic. The current head-tiling heuristic is parked
(it over-tiles — `hc=2` already beats `hc=1` on the cliff, the precompute-CTA and
main-CTA counts are coupled through co-residency).

Motivation (cliff timeline, b=64): triton-pm's persistent main starts at 2.91µs
(during conv1d); our 1024-CTA main is contention-blocked until ~5.06µs. Closing
that ≈ the 1.3µs gap to triton.

## Current kernel state (as of this plan)

The CUDA main kernel (`checkpointing_ssu_main_kernel`) already has:

- **`head_loop` with `MAIN_HEADS_PER_CTA`** (`kernel_checkpointing_ssu_main.cuh` L302): a
  compile-time loop over consecutive heads within ONE work-unit. Grid is
  `dim3(D_SPLIT, batch, ngroups * HEAD_TILES)` where `HEAD_TILES = HEADS_PER_GROUP /
  MAIN_HEADS_PER_CTA`.  **`MAIN_HEADS_PER_CTA` default is now `0` (auto-heuristic).**
- **One-shot gdc_wait via `IS_FIRST`** (`replay_head`, L249-251): `if constexpr (IS_FIRST)
  cudaGridDependencySynchronize()` — called only for the first head in `head_loop`, guarded
  by a `__syncthreads()` just before it (which both publishes replayed state cross-warp and
  acts as the fence for gdc visibility).
- **`__maxnreg__(64)`** on the main kernel (L448), giving 8 blocks/SM at 50% occupancy.
- **`STAGES=2` double-buffer** for the state pipeline (controlled by
  `FLASHINFER_SSU_MAIN_PIPELINE_STAGES`). Correctness-tested after the fixes in
  commit c2508c2d.

What is **NOT yet** in the kernel:
- A true grid-stride outer loop over multiple `(d_tile, seq, head_tile)` work-units per CTA.
  Each CTA currently processes exactly one work-unit (one `(d_tile, seq, head_tile)` triplet).

## Fixes and knob changes (commit c2508c2d, 2026-06-24)

Three correctness fixes in the STAGES=2 code path plus two new perf knobs.

### Fix 1: cumAdt smem stride mismatch

`CheckpointingSsuMainStorage` allocated `float cumAdt[STATE_PIPE * NPREDICTED]` but
all readers used `NPREDICTED_PAD_MMA_M = next_multiple_of<16>(NPREDICTED)` as the
slot stride. For T=6: NPREDICTED=6, pad=16 → `buf=1` write landed at offset 6 while
the read expected offset 16 → corrupted output.

**Fix** (`kernel_checkpointing_ssu_main.cuh`, smem struct ~L98):
```cpp
// BEFORE:
float cumAdt[STATE_PIPE * NPREDICTED];
// AFTER:
float cumAdt[STATE_PIPE * NPREDICTED_PAD_MMA_M];
```
All three write sites updated to use `* NPREDICTED_PAD_MMA_M` stride.

### Fix 2: Phantom pipeline commits

`__pipeline_commit()` for the G_tiles group (state + old_tiles prefetch) was placed
**outside** `if (has_next)`, pushing an empty pipeline group into the queue on every
last-head iteration (`has_next=false`). The subsequent `wait_prior(1)` had to drain
that empty group, which also consumed the `x_h` group → serializing the HBM load
of `x_h` with MMA on the critical path at the end of every CTA.

**Fix** (`kernel_checkpointing_ssu_main.cuh`, STAGES=1 and STAGES=2 loops):
```cpp
// AFTER (both commits moved inside if (has_next)):
if (has_next) {
    prefetch_state<...>(...);
    load_head<...>(...);
    __pipeline_commit();  // G_tiles: state_{h+1} + old_tiles_{h+1}
}
if (has_next) {
    load_x<...>(...);
    __pipeline_commit();  // G_xnext: x_{h+1}
}
// wait comment updated:
// has_next:  3 in-flight = [x_h, G_tiles(h+1), G_xnext(h+1)].
//            wait_prior(3) keeps all 3 in flight.
// !has_next: 1 in-flight = [x_h].  wait_prior(1) = no stall.
__pipeline_wait_prior(has_next ? 3 : 1);
```

### Fix 3: Static env-var prevents monkeypatch

`static int const main_pipeline_stages` initialized once per template instantiation
per process. `monkeypatch.setenv("FLASHINFER_SSU_MAIN_PIPELINE_STAGES", "2")` after
the first call had no effect. Removed `static` → reads env on every launch.

### New knob: MHC auto-heuristic

`main_heads_per_cta=0` (Python default changed from `1` to `0`) triggers the launcher
auto-heuristic:

```cpp
// launch_checkpointing_ssu.cuh
// heuristic: batch>=512 → MHC=4 (amortizes C/old_B; b=512: 75.6->72.3us, b=1024: 145->131us)
{
  int const mhc_env = [] {  // non-static: monkeypatch-safe
    char const* e = std::getenv("FLASHINFER_SSU_MAIN_HEADS_PER_CTA");
    return e ? std::atoi(e) : 0;
  }();
  if (mhc_env > 0)
    main_heads_per_cta = mhc_env;
  else if (main_heads_per_cta == 0)
    main_heads_per_cta = (params.batch >= 512) ? 4 : 1;
}
```

`FLASHINFER_SSU_MAIN_HEADS_PER_CTA=N` overrides the heuristic for benchmarking.
csrc validation updated to allow `0` as the auto-sentinel.

### New default: DS=2 in benchmark

`bench_checkpointing_ssu.py` now defaults `FLASHINFER_SSU_D_SPLIT=2` (was `1`).
DS=2 doubles the main CTA count (256→512 at b=16) while halving D_PER_CTA (128→64),
saving 2.04µs at b=16. The mixed benchmark default was also updated.

### Test: `test_two_kernel_pipeline_stages2`

Added to `tests/mamba/test_checkpointing_ssu.py` (before
`test_checkpointing_ssu_pdl_bf16`). Exercises the STAGES=2 code path via
`monkeypatch.setenv("FLASHINFER_SSU_MAIN_PIPELINE_STAGES", "2")` with MHC=2 vs
MHC=1 reference, checking both the no-write and write paths at batch=2, T=6,
max_window=8.

## Choices (locked)

1. **Replace** the non-persistent main — the persistent loop subsumes it
   (`cta_per_sm` high ⇒ `grid == total_work` ⇒ one work-unit/CTA ⇒ identical).
2. **`cta_per_sm` via env** (`FLASHINFER_SSU_MAIN_CTA_PER_SM`), heuristic later.
3. **Flatten head-fastest** (`tile = (d_tile*batch + seq)*nheads + head`) so
   consecutive work-units are consecutive heads of one `(seq, group)` → their
   per-group `C` / `old_B` stay L2-hot (and a later step can reuse them outright,
   "don't reload C/B for the next head").

## Test (xfail-first)

- [ ] **T_PERSIST** `tests/mamba/test_checkpointing_ssu.py::test_persistent_main_matches_monolithic`
  — set `FLASHINFER_SSU_MAIN_CTA_PER_SM=1` (force `grid << total_work` so each CTA
  loops ≥2×), run the two-kernel at a batch with `batch·nheads > NUM_SMS`, assert
  bit-exact vs monolithic (`out`, `state`, `old_x`, `old_B`, `old_dt`,
  `old_cumAdt`) on both the no-write and write paths. **Initially xfail** (no loop
  yet). *EASY*

## Steps

1. [ ] **Env knob** `FLASHINFER_SSU_MAIN_CTA_PER_SM` in the launcher (read
   per-launch so T_PERSIST can vary it; replaced by a heuristic later).
   `include/flashinfer/mamba/launch_checkpointing_ssu.cuh` (main launch, ~L149). *EASY*
2. [ ] **1D persistent grid**: main grid `dim3(D_SPLIT, batch, ngroups*HEAD_TILES)` →
   `dim3(min(cta_per_sm·NUM_SMS, total_work))`, `total_work = D_SPLIT·batch·nheads` (at
   `MAIN_HEADS_PER_CTA=1`, i.e., `HEAD_TILES=HEADS_PER_GROUP`).
   `NUM_SMS` from `cudaDevAttrMultiProcessorCount` (cache statically, like the
   precompute's `sm_count`). Default `cta_per_sm` ⇒ `grid == total_work`.
   `launch_checkpointing_ssu.cuh` ~L149-155. *MEDIUM*
3. [ ] **Grid-stride loop in the kernel body**: wrap the per-work-unit body
   (setup + `prefetch_state` + `load_head` + `replay_head` + `load_x` + `output_head`) in
   `for (int tile = blockIdx.x; tile < total_work; tile += gridDim.x) { ... }`,
   replacing the `blockIdx.{x,y,z}` reads with the unflatten
   `head = tile % nheads; t = tile / nheads; seq = t % batch; d_tile = t / batch;
   group_idx = head / HEADS_PER_GROUP`.
   **IS_FIRST split required**: the existing `IS_FIRST` template param in `replay_head`
   and `load_x` serves two independent purposes that diverge in the persistent loop:
   - `load_x IS_FIRST=true` → reload C from gmem (needed for EVERY outer iteration
     since group_idx changes per work-unit → different C address)
   - `replay_head IS_FIRST=true` → fire `cudaGridDependencySynchronize()` (needed ONLY
     for the FIRST outer iteration, `tile == blockIdx.x`)
   These must be split into two separate template params in `head_loop` / `replay_head`:
   `IS_LOAD_C_FIRST` (always true when h==0 inside head_loop) and `DO_GDC_WAIT`
   (compile-time bool, true only for first outer tile). In the kernel body:
   ```cpp
   if (tile == blockIdx.x) {
       head_loop<..., DO_GDC_WAIT=true>(...);
   } else {
       head_loop<..., DO_GDC_WAIT=false>(...);
   }
   ```
   This preserves compile-time elimination of the gdc_wait for iteration > 0.
   Precompute/conv1d complete once; subsequent iterations read `cb_scaled` /
   `cumAdt_vec` without re-waiting. Iteration-0's `prefetch_state` + `load_head`
   (Phase-1) still runs *before* `replay_head` so it overlaps conv1d/precompute.
   Pre-GDC correctness verified (2026-06-24): all G0/G1 reads use `buf_read`
   (the double-buffer read slot); precompute always writes to `buf_write` — no
   race regardless of gdc_wait timing.
   `include/flashinfer/mamba/kernel_checkpointing_ssu_main.cuh` (kernel body, starting at L467). *HARD*
4. [ ] **Loop-boundary `__syncthreads()`** at the bottom of the outer grid-stride loop
   so iteration N's `output_head` / `store_old_x` finish reading/writing smem before
   iteration N+1's `prefetch_state` / `load_head` overwrite the same buffers. Note:
   `output_head` already has an internal `__syncthreads()` at L272 (fences `cumAdt` and
   the x/C pipeline data cross-warp before the output MMA), but that is mid-function; the
   loop-boundary sync is a separate, additional barrier at the bottom of the loop body. *EASY*
5. [ ] **Make T_PERSIST pass** — debug the loop/unflatten until bit-exact. *MEDIUM*
6. [ ] **T0 still passes** (`test_two_kernel_matches_monolithic`) at default
   `cta_per_sm` (grid == total_work ⇒ unchanged behavior). *EASY*

## How to measure kernel overlap (conv1d + precompute + main)

Use `SSU_CUPTI_DEBUG=1` with the mixed bench's `--with-conv1d` flag.  CUPTI
prints per-kernel start/end timestamps (relative to the first kernel) for the
first iteration of each PNAT sample, to stderr.

```bash
# Single HPC value, batch=16, kernels=cuda-incr-2k + triton-replay for comparison:
SSU_CUPTI_DEBUG=1 FLASHINFER_SSU_HEADS_PER_CTA=8 \
  uv run python benchmarks/bench_ssu_checkpoint_mixed.py \
    --kernels cuda-incr-2k,triton-replay \
    --batch-sizes 16 --with-conv1d \
    --iters 5 --warmup 3 --output-prefix - \
  2>&1 | grep -E "CUPTI-DEBUG|start="

# Sweep HPC values (run the command above with HPC=4, 8, 16):
for hpc in 4 8 16; do
  echo "=== HPC=$hpc ==="
  SSU_CUPTI_DEBUG=1 FLASHINFER_SSU_HEADS_PER_CTA=$hpc \
    uv run python benchmarks/bench_ssu_checkpoint_mixed.py \
      --kernels cuda-incr-2k --batch-sizes 16 --with-conv1d \
      --iters 3 --warmup 2 --output-prefix - \
    2>&1 | grep "start="
done
```

The output looks like:
```
[CUPTI-DEBUG] tag=mixed_b16_T6_W16_cuda-incr-2k_s0 iter=0 n_kernels=3
  start=   0.000  end=   3.840 µs  _causal_conv1d_update_kernel
  start=   2.528  end=   7.296 µs  _checkpointing_ssu_precompute_kernel...
  start=   3.649  end=  10.176 µs  _checkpointing_ssu_main_kernel...
```

`start` is relative to the first kernel's start.  The PDL trigger time from
conv1d → precompute is the precompute's `start` value.  Total wall = last `end`.

To measure NCU resource usage of the precompute at different HPC values:
```bash
FLASHINFER_SSU_HEADS_PER_CTA=8 $CUDA_HOME/ncu \
  --target-processes all \
  --kernel-name "checkpointing_ssu_precompute_kernel" \
  --metrics "launch__registers_per_thread,launch__shared_mem_per_block_dynamic,\
launch__waves_per_multiprocessor,launch__block_size,launch__grid_size" \
  --launch-count 1 \
  uv run python benchmarks/bench_ssu_checkpoint_mixed.py \
    --kernels cuda-incr-2k --batch-sizes 16 --no-cupti --no-cuda-graph \
    --iters 1 --warmup 0 --output-prefix - 2>&1 | grep -E "registers|shared|waves|block|grid"
```

## Investigation findings: conv1d PDL trigger delay (batch=16, B300)

Measured with `SSU_CUPTI_DEBUG=1` + `bench_ssu_checkpoint_mixed.py --with-conv1d`.

### Timeline (with conv1d, batch=16)

| config | conv1d end | precompute start | precompute dur | main dur | wall total |
|--------|-----------|-----------------|----------------|----------|------------|
| ours HPC=8  | 3.84µs | 2.53µs | 4.77µs | 6.53µs | **10.18µs** |
| ours HPC=16 | 4.19µs | 1.12µs | 5.60µs | 10.46µs | **12.96µs** |
| triton      | 4.16µs | 1.12µs | 5.28µs |  7.10µs |  **9.15µs** |

The individual precompute and main kernel durations for HPC=8 are **faster** than
Triton's. The gap is entirely that our precompute starts 1.4µs later (conv1d fires
the PDL trigger at 2.53µs instead of 1.12µs).

### The 512-thread threshold

Tested HPC=4 (64 CTAs, 128 threads), HPC=8 (32 CTAs, 256 threads), HPC=16 (16 CTAs,
512 threads). NCU shows **identical** resource footprint for all three:

| HPC | CTAs | block | waves/SM | regs/thread | smem/CTA |
|-----|------|-------|----------|-------------|----------|
|  4  |  64  |  128  |   0.04   |     36      |  10.9 KB |
|  8  |  32  |  256  |   0.04   |     36      |  11.7 KB |
| 16  |  16  |  512  |   0.04   |     36      |  13.2 KB |

0.04 waves/SM means the entire precompute grid occupies ~5-6 SMs out of 168 —
trivial fraction. SM footprint, registers, and smem are virtually identical yet:
- HPC=4 and HPC=8 (128/256-thread blocks): trigger fires at **2.53µs**
- HPC=16 (512-thread block): trigger fires at **1.12µs**

The only visible difference is block size. This appears to be a **Blackwell
block-scheduler artifact**: at 512 threads/CTA the GPU makes the PDL reservation
earlier relative to conv1d's execution, even though resource usage is the same.

### HPC=16 breaks the main

With HPC=16, precompute has only 1 CTA per batch item (16 total). It fires its
PDL trigger to the main after completing just one head worth of work per CTA. The
main kernel starts at 2.49µs but then **stalls for 10.46µs** while waiting for
cb_scaled/cumAdt_vec data for heads 2–16 that the single precompute CTA hasn't
produced yet. Result: 12.96µs total — far worse than HPC=8.

### Implication for persistent main

If we can make the main not stall under HPC=16 (i.e., the persistent main is
either light enough to co-reside or structured to not block on precompute data
it hasn't received yet), we unlock the 1.12µs trigger timing and could reach
≈8.7µs wall — **faster than Triton** (9.15µs).

## Joint knob sweep: precompute (HPC, NW) × main (MHC, DS), batch=16, B300, 2026-06-24

### Sweep command
```bash
for hpc in 4 8; do for nw in 4 8; do for mhc in 1 2; do for ds in 1 2; do
  FLASHINFER_SSU_HEADS_PER_CTA=$hpc FLASHINFER_SSU_PRECOMPUTE_NUM_WARPS=$nw \
  FLASHINFER_SSU_MAIN_HEADS_PER_CTA=$mhc FLASHINFER_SSU_D_SPLIT=$ds \
  SSU_CUPTI_DEBUG=1 uv run python benchmarks/bench_ssu_checkpoint_mixed.py \
    --kernels cuda-incr-2k --batch-sizes 16 --with-conv1d \
    --iters 5 --warmup 3 --output-prefix - 2>&1 | grep "start="
done; done; done; done
```

### Results (sorted by wall_end)

| HPC | NW | MHC | DS | pre_start | main_start | wall_end |
|-----|----|-----|----|-----------|------------|----------|
|   8 |  8 |   1 |  2 |  0.93µs   |   2.66µs   | **10.30µs** |
|   4 |  8 |   1 |  2 |  2.51µs   |   3.65µs   | **10.32µs** |
|   4 |  8 |   1 |  1 |  2.62µs   |   3.74µs   |  10.69µs |
|   4 |  4 |   1 |  1 |  2.53µs   |   3.50µs   |  11.50µs |
|   4 |  4 |   1 |  2 |  2.58µs   |   3.58µs   |  11.78µs |
|   4 |  8 |   2 |  2 |  2.61µs   |   3.66µs   |  12.18µs |
|   8 |  8 |   1 |  1 |  0.96µs   |   2.67µs   |  12.34µs |
|   8 |  4 |   1 |  2 |  2.62µs   |   3.46µs   |  12.67µs |
|   8 |  4 |   1 |  1 |  2.61µs   |   3.49µs   |  12.80µs |
|   4 |  8 |   2 |  1 |  2.54µs   |   3.62µs   |  13.12µs |
|  ...| ...|  ...|... |    ...    |    ...     |    ...   |

(MHC=2 configs are all ≥12.18µs — MHC=1 is always better)

### Key findings

1. **DS=2 is a large win for main**: at HPC=8,NW=8,MHC=1: DS=1→12.34µs, DS=2→10.30µs
   (2.04µs improvement). DS=2 doubles the main CTA count (256→512) by splitting
   DIM=128 across two CTAs; better L2 utilization. **Make DS=2 the default for DIM≥64.**

2. **MHC=1 always beats MHC=2**: best MHC=2 is 12.18µs; best MHC=1 is 10.30µs.
   The CTA-count reduction from MHC=2 kills bandwidth more than C reuse helps.

3. **NW=8 always beats NW=4**: e.g. HPC=4,MHC=1,DS=2: NW=4→11.78µs, NW=8→10.32µs.

4. **Best config**: HPC=8, NW=8, MHC=1, DS=2 = 10.30µs (essentially tied with
   HPC=4, NW=8, MHC=1, DS=2 = 10.32µs). HPC=8 wins by 0.02µs due to earlier trigger.

### Side-by-side vs triton (best config, 5 measured iters)

```
kernel                    pre_start  pre_end  main_start  wall    stall   post-gdc-exec
ours (HPC=8,NW=8,MHC=1,DS=2)  0.93µs   6.37µs    2.59µs   10.24µs  3.78µs    3.87µs
triton (persistent main)       0.96µs   6.85µs    1.82µs    9.25µs  5.02µs    2.40µs
```

- Our precompute finishes **0.48µs FASTER** (6.37 vs 6.85µs).
- Triton main starts **0.77µs earlier** (fewer CTAs → faster GPU scheduling).
- Triton post-gdc execution is **1.47µs faster** (2.40 vs 3.87µs) — L2 cache hot
  from persistent loop reusing C/old_B across work-units per CTA.
- **Total gap: 0.99µs** (10.24 vs 9.25µs).

### Persistent main target

With persistent main matching triton's L2 efficiency (2.40µs post-gdc exec) and
earlier scheduling (1.82µs main start), and keeping our faster precompute (6.37µs end):
- stall = 6.37 - 1.82 = 4.55µs (triton: 5.02µs — ours shorter since precompute faster)
- wall = 1.82 + 4.55 + 2.40 = **8.77µs — ~0.5µs faster than triton**

The entire remaining gap is execution efficiency (L2 cache hot), NOT trigger timing.
HPC=8 (early trigger 0.93µs) is already the best we can get without persistent main.

## Large-batch investigation (batch=512, B300, 2026-06-24)

### Write vs no-write benchmark (bench_checkpointing_ssu.py)

Command:
```bash
FLASHINFER_SSU_MAIN_HEADS_PER_CTA=0 uv run python benchmarks/bench_checkpointing_ssu.py \
  --batch-sizes 512 --mtp-lengths 6 --max-window 8 \
  --prev-tokens-fracs 0,1.0 --cupti --state-dtypes bf16 \
  --no-flashinfer-dump --iters 50 --warmup 10
```

**No-write case only** (the path most of our optimization targets — `must_checkpoint=false`):
drop the `1.0` fraction so only `prev_k=0` runs. Halves the run (skips the write
recompile/timing) and isolates the no-write kernel under CUPTI:
```bash
FLASHINFER_SSU_MAIN_HEADS_PER_CTA=0 uv run python benchmarks/bench_checkpointing_ssu.py \
  --batch-sizes 512 --mtp-lengths 6 --max-window 8 \
  --prev-tokens-fracs 0 --cupti --state-dtypes bf16 \
  --no-flashinfer-dump --iters 50 --warmup 10
```
`--prev-tokens-fracs` is `prev_k / mtp_len`: with `mtp_len=6, max_window=8` only
`frac ≤ 2/6 ≈ 0.33` (i.e. `prev_k ≤ 2`) stays on the no-write path; `frac=0` is the
canonical no-write point. `--cupti` measures the SSU kernel(s) directly (no conv1d
span), so per-head barrier / pipeline changes show up undiluted.

`must_checkpoint = (prev_k + mtp_len > max_window)`:
- `prev_k=0`: 0+6=6 ≤ 8 → **no-write** path
- `prev_k=6`: 6+6=12 > 8 → **write** path (checkpoints state back to cache)

| kernel | no-write (prev_k=0) | write (prev_k=6) | write penalty |
|--------|---------------------|-----------------|---------------|
| `cuda-incr-2k` (MHC=4 auto) | **54.0µs** | **74.4µs** | +20.4µs (+38%) |
| `cuda-incr` (monolithic)    |  58.8µs  |  77.5µs  | +18.7µs |
| `triton-replay`             |  70.5µs  | 130.5µs  | +60.0µs |
| `triton-replay-pm`          |  43.0µs  |  97.7µs  | +54.7µs |
| `incremental` (flashinfer)  |  92.9µs  | 106.0µs  | +13.1µs |

**Key takeaways:**
- `cuda-incr-2k` **beats triton-replay-pm on write** (74.4 vs 97.7µs, +31% faster).
  Our write path avoids triton-pm's double-launch overhead.
- `triton-replay-pm` wins on **no-write** (43.0 vs 54.0µs, 26% gap). This is the
  remaining target for the persistent main.
- `triton-replay` (non-persistent) is the worst on write — confirms that the
  persistent-main benefit is specifically the C/old_B L2 cache reuse.

### Per-head barrier reduction (2026-06-25)

Hoisted both `replay_head` `__syncthreads()` into the callers, inlined
`replay_state_mma` (the wrapper became meaningless), and dropped the **redundant
post-replay barrier**: the unconditional pre-output `__syncthreads()` (loop) /
pre-gdc `__syncthreads()` (head 0) already publishes the in-place-replayed state,
and nothing reads state cross-warp in between (only gmem CB LDGs). The pre-replay
barrier stays, write-path only. Per-head barriers: **no-write 3→2, write 4→3**;
head-0 unchanged. (Earlier in the same change, the no-write `output_head_2k` x-fence
that had been wrongly removed was restored — it fences warp-0-loaded `smem.x` to all
warps; a distributed per-warp x load can't replace it because a column slice is a
sub-row → bank-conflicted cp.async write. See `kernel_checkpointing_ssu_main.cuh`.)

Re-measured no-write (same command, `--prev-tokens-fracs 0`, B300):

| kernel | no-write before | no-write after | Δ |
|--------|----------------:|---------------:|----:|
| **`cuda-incr-2k`** (MHC=4 auto) | 54.0µs | **51.2µs** | **−2.8µs (−5.1%)** |
| `cuda-incr` (monolithic)    | 58.8µs | 58.3µs | ~flat (untouched) |
| `triton-replay`             | 70.5µs | 72.2µs | ~noise |
| `triton-replay-pm`          | 43.0µs | 41.9µs | ~noise |
| `incremental` (flashinfer)  | 92.9µs | 91.3µs | ~noise |

The unchanged kernels are within run-to-run noise → methodology stable. The −2.8µs
on `cuda-incr-2k` is the dropped barrier across MHC=4 heads. Gap to `triton-replay-pm`
narrows 11.0→9.4µs; the remainder is the persistent-main launch/co-residency edge,
not barriers. Correctness: `test_two_kernel_matches_monolithic` green (mhc1/2/4,
PDL + non-PDL, write + no-write).

### NCU profiling at batch=512 (main kernel only)

Captured with `FLASHINFER_JIT_LINEINFO=1`:
```bash
# Warm up JIT then capture:
FLASHINFER_JIT_LINEINFO=1 FLASHINFER_SSU_MAIN_HEADS_PER_CTA=1 \
  $CUDA_HOME/ncu --set full \
    --kernel-name regex:checkpointing_ssu_main_kernel \
    --launch-count 3 --target-processes all \
    -o /home/scratch.ishovkun_gpu/benchmarks/mamba_decode/ssu_main_mhc1_b512 \
  uv run python benchmarks/bench_checkpointing_ssu.py \
    --batch-sizes 512 --mtp-lengths 6 --max-window 8 \
    --prev-tokens-fracs 0 --state-dtypes bf16 \
    --no-triton-replay --no-triton-replay-pm --no-flashinfer-dump \
    --no-cuda-graph --iters 5 --warmup 0 --no-cupti
# repeat with HEADS_PER_CTA=2 → ssu_main_mhc2_b512
```

NCU reports:
- MHC=1: `/home/scratch.ishovkun_gpu/benchmarks/mamba_decode/ssu_main_mhc1_b512.ncu-rep`
- MHC=2: `/home/scratch.ishovkun_gpu/benchmarks/mamba_decode/ssu_main_mhc2_b512.ncu-rep`

Read with: `uv run /home/scratch.ishovkun_gpu/code/ncu_reader/ncu_reader.py -i <file> -n 0`

### NCU results summary

| metric | MHC=1 | MHC=2 | delta |
|--------|-------|-------|-------|
| Grid | (2,512,16) | (2,512,8) | 2× fewer CTAs |
| Kernel duration | 93.44µs | 84.35µs | −9.1µs (−10%) |
| Waves/SM | 13.84 | 6.92 | 2× fewer waves |
| Achieved occupancy | 45.7% | 46.8% | ≈same |
| DRAM reads | 149.31 MB | 149.35 MB | identical |
| L2 cache util | 21.3% | 23.6% | MHC=2 better |
| Long scoreboard stall | 28.8% | 25.2% | MHC=2 better |
| Barrier stall | 18.3% | 17.4% | ≈same |
| mio_throttle | 4.0% | 7.8% | new in MHC=2 |

**MHC=2 is 9µs faster despite identical DRAM bytes** — the improvement is entirely L2
cache efficiency: C/old_B loads for head `h+1` hit in L2 because head `h` (same group)
warmed it. mio_throttle (LDSM smem matrix loads) is new at MHC=2, suggesting the smem
bandwidth is becoming a secondary limiter as L2 pressure eases.

### Top stalls and their meaning

**#1: long_scoreboard at `kernel_checkpointing_ssu_main.cuh:332` (NOP)**
- MHC=1: 45.3% of long_scoreboard; MHC=2: 30.6%
- This is the `__pipeline_wait_prior` site (after the cp.async commit in the
  `has_next` block). The NOP is inserted by the compiler to absorb the cp.async
  latency before the state can be consumed. Indicates the pipeline depth is
  insufficient to hide the state prefetch latency.

**#2: barrier stall at `kernel_checkpointing_ssu_common.cuh:1255` (LDG)**
- MHC=1: 82.3% of barrier stalls; MHC=2: 67.6%
- This LDG is the global load of per-group C/old_B matrix data, gated by a
  `BAR.SYNC.DEFER_BLOCKING` at `kernel_checkpointing_ssu_main.cuh:361` (the
  `__syncthreads()` before replay_head). Warps stall waiting for other warps to
  finish loading C/old_B from HBM. This is the primary reason persistent main
  wins: it keeps C/old_B hot in L2 across work-units.

**#3: short_scoreboard on HMMA / IMAD**
- Tensor core and integer pipeline latency; unavoidable without structural changes.

### Persistent main impact at b=512

With persistent main (grid << total_work), C/old_B for iteration `k+1` is already
in L2 from iteration `k` (same or adjacent group). This directly attacks the #2
stall. The #1 stall (cp.async pipeline) may also improve if the persistent loop
can overlap state loads across outer iterations.

**Target**: match triton-replay-pm's no-write performance (43.0µs). Current gap:
54.0 − 43.0 = **11.0µs** at b=512.

** TEsts
```bash
uv run pytest tests/mamba/test_checkpointing_ssu.py -v -s -x -k "two_kernel"
```
don't run the full test as it's pretty long.
