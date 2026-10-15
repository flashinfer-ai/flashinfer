# W8A16 FP8 GEMM: optimization ladder summary

`flashinfer.gemm.w8a16_fp8_gemm` computes `out = act @ (weight.float() * scale).T + bias`
for an FP8 e4m3 `[N, K]` weight with one per-tensor fp32 scale (the TRT-LLM FP8QDQ
checkpoint layout) and a bf16 / fp16 activation, dequantizing the weight inside the
kernel.  It replaces TRT-LLM's `W8A16FP8LinearMethod.apply` cast-copy path
(`weight.to(dtype).mul_(scale); F.linear`).  Hopper only (`sm_90a`: wgmma + TMA).

The kernel was built as a ladder of single-change commits on branch `w8a16-ladder`
(every rung a commit, losers kept, never rebased).  Each commit message carries the
measured table, the pinned-clock ncu numbers and the spill check.  This file is the
index.

## Benchmark

`benchmarks/bench_w8a16_fp8_gemm.py`: CUPTI timing (L2 flush before every
iteration, `cupti.finalize` once per process), 108 rows = 6 Cosmos3-Super layers
x TP 2 / 4 / 8 x M in {1024, 2048, 4096, 8192, 44160, 88320}; baselines are cuBLAS
bf16 GEMM on the dequantized weight (the ceiling) and the TRT-LLM cast-copy path;
every row is checked against the cast-copy reference
(`max|err| <= max(2 x cast_err, 2e-3 x max|ref|)`).  `--shape N,K --m M` pins one
configuration; `--impls kernel --iters 1 --warmup 0 --no-cupti --no-l2-flush` is the
form used under `ncu`.

Layers (N, K, shard): und.q (8192, 5120, col), und.kv (1024, 5120, col),
gen.qkv (10240, 5120, col), attn.out (5120, 8192, row), mlp.gate_up (25600, 5120, col),
mlp.down (5120, 25600, row); column-parallel shards N by TP, row-parallel shards K.

## Final state (rung 37; rung 38 is an env-only loser on top)

Geometric mean of per-row time ratios over the 18 rows of each M class.

| M | H100 kernel / cuBLAS | H100 kernel / cast-copy | H200 kernel / cuBLAS | H200 kernel / cast-copy | H200 / H100 kernel time |
|---|---|---|---|---|---|
| 1024 | 1.32 | 0.46 | 1.33 | 0.46 | 0.98 |
| 2048 | 1.18 | 0.60 | 1.22 | 0.61 | 0.99 |
| 4096 | 1.08 | 0.72 | 1.11 | 0.74 | 1.00 |
| 8192 | 1.06 | 0.84 | 1.06 | 0.85 | 0.98 |
| 44160 | 1.04 | 1.03 | 1.05 | 1.03 | 0.99 |
| 88320 | 1.04 | 1.01 | 1.04 | 1.04 | 0.99 |

H100 SXM (s4124-0129) and H200 (s4124-0128), free-running clocks.  Pinned-clock ncu
on 12800x5120 M=8192 (`Cfg<128,256,4>`, persistent): tensor pipe 96.0% on both parts,
SM throughput 91.8%, 0 spills.  Where the dispatcher splits K (narrow N, short M) the
mainloop kernel runs at 57-72% tensor pipe plus a bandwidth-bound reduce kernel; those
rows are 20-50 us against a cuBLAS of 15-35 us.

The plain checkpoint layout is the shipped path; `prepare_w8a16_fp8_weight` +
`weight_prepared=True` (ldmatrix fragment loads) was opt-in and not faster on H100/H200
(both removed in H7 together with the hand kernel).
Spill-free on every instantiation the dispatcher can launch (128x256 and 128x128, plain
and prepared, persistent and split-K, bf16 and fp16; the reduce kernel).

Rung 29 (the earlier ship point) to rung 37 on H100, kernel / cuBLAS: 1.50 -> 1.32,
1.27 -> 1.18, 1.16 -> 1.08, 1.10 -> 1.06, 1.09 -> 1.04, 1.09 -> 1.04 for the six M
classes; large-M tensor pipe 92.3% -> 96.0%.

Re-verified after rung 38 (shared kernel code touched): auto-path table within noise of
rung 37, pinned 96.0% tensor pipe on 12800x5120 M=8192, all instantiations spill-free,
130 tests under auto and every forced tile.

## Knobs

| Variable | Effect |
|---|---|
| `FLASHINFER_W8A16_FP8_TILE` | `128x256`, `128x128`, `256x128` or `64x256`: pins the tile instead of the cost model (auto only ever picks the first two). Read once per process. |
| `FLASHINFER_W8A16_FP8_SPLITS` | Pins the split-K count (clamped to >= 8 K slabs per split, <= 8 splits). `0` / unset = model. Read once per process. |

Dispatch (host side, `W8A16Fp8GemmPickTile` / `W8A16Fp8GemmSplits`): for each tile and
split count the modelled time is
`ceil(units * splits / SMs) * tileN * tileM * (K / splits) / eff + [splits > 1] (24M + 8 * M * N * splits)`
with eff 1.00 for 128x256 and 0.65 for 128x128; the cheapest plan launches.  Both
constants were fitted to forced-tile and forced-split tables on H100.

## Rungs (commit subjects carry the details)

Kernel structure by rung 29: TMA loads (SW128 activation, SW64 weight) into a 4-stage
mbarrier pipeline, 1 producer + 2 consumer warpgroups (`setmaxnreg` 24 / 240), fp8 ->
16-bit conversion into registers and `wgmma m64n256k16` with A from registers, grouped
raster, persistent tiles, split-K through an fp32 workspace for under-filled grids.

| Rung | Commit | Change | Result |
|---|---|---|---|
| 30 | 5093170a7 | tile-shape dispatch (128x256 / 256x128 / 128x128) by eff x last-wave fill | LOSER: early return handed under-filled problems to 128x128; split-K not in the score |
| 30b | e6d579e84 | split-K-aware cost model fitted to the forced-tile tables | within 0.2% of the per-row oracle; 256x128 dropped from auto (never wins, spills 456 B plain) |
| 31 | 38262cd77 | TMA-store epilogue (per-WG swizzled staging buffer, `cp.async.bulk.tensor` store) | +1.5-3% at large M; introduced a 64 B spill |
| 32 | 90f5219e1 | staging stores through the shared window (`st.shared`, 32-bit addresses) | spill gone; +1-3% everywhere; tensor pipe 95.3% |
| 33 | 9b9e891c0 | shuffle-paired 32-bit staging stores (one convert per pair) | marginal (+~1% pinned); tensor pipe 96.1% |
| 34 | e3d34e102 | split-K count from the cost model (linear reduce price) | mixed: -2.6% at M=1024, over-splits several und.kv rows |
| 35 | 6a799fdac | in-kernel split-K fixup by the last-arriving CTA | LOSER: latency-bound fixup loop (+17% at M=1024) |
| 36 | b1871d781 | batched loads in that fixup | LOSER: one CTA per tile cannot match a whole-GPU reduce; counter memset costs the saved launch |
| 37 | f89b9ef45 | two-kernel split restored; reduce priced as 24M + 8 x M x N x splits | WIN: picks a measured plan on all 108 rows at 1.004x the best measured |
| 38 | 08c193e74 | 64x256 tile, K slabs split between the consumer warpgroups (keeps m64n256) | LOSER: L2-bound, 43% tensor pipe; env-only |

Earlier losers kept on the branch: rung 22 (epilogue skew), 25-26 (stream-K: L2
locality and a 1248 B spill), 28 (textbook separate-tile ping-pong, 0.75x).

## Why the losers lost

- **Rung 30**: a wave-fill score that returns early for "any tile at two waves" picks the
  least efficient tile first; the model has to compare modelled time, not fill.
- **Rungs 35-36**: the serial fixup puts the whole reduction of a tile on one CTA.  With
  4 tiles x 8 splits that is 4 CTAs doing what 132 SMs did before, and the per-launch
  counter memset is a graph node the benchmark (correctly) counts.  Pinned, the fused
  kernel was 10% faster than the pair on the best row and still lost end to end.
- **Rung 38**: a 64-row weight tile moves 1.8x the activation bytes per MAC through L2;
  128x256 already runs L2 at 70% while the tensor pipe is at 96%, so the narrow tile
  saturates L2 at 43% tensor pipe.  Splitting K between the two warpgroups and
  exchanging accumulator halves through smem worked (tests pass, 0 spills) but cannot
  beat the bandwidth wall.  First attempt also indexed the accumulator array with the
  runtime warpgroup id, which pushed all 128 accumulators to local memory (29% tensor
  pipe) until the half index was made a compile-time constant.
- **Prepared weight layout**: the ldmatrix fragment path is not faster than the plain
  layout on H100 / H200 and its non-split instantiation spilled 184 B before rung 32;
  it stays opt-in.

## Results directory

`/home/scratch.ishovkun_gpu/w8a16-results/` (computelab scratch): `rung<N>_*_full.csv`
(108-row tables, one per tile / device), `rung<N>_ncu_*.txt` (full `ncu --set full`
dumps with `sm__pipe_tensor_op_hmma_cycles_active`), `rung<N>_log.txt`.

# Blackwell branch (sm_100a, B200)

Branch `w8a16-blackwell` continues the ladder on Blackwell.  Same API, same tests, same
benchmark (plus TP 1, so 144 rows); a separate JIT module picked by compute capability
(`gen_w8a16_fp8_gemm_sm100_module`), the Hopper module is untouched.  No prepared weight
layout on Blackwell.

## Hopper, CUTLASS line (versions H1 / H2 / H3, Tue 13 Oct 2026)

`include/flashinfer/gemm/w8a16_fp8_gemm_sm90.cuh`: the same op on CUTLASS's sm90
mixed-input mainloop (`sm90_mma_tma_gmma_rs_warpspecialized_mixed_input`: TMA to smem, each
consumer warpgroup converts its 64 weight rows in registers, wgmma with A from registers),
built next to the hand kernel so the two compete per shape through the Python op's `algo`
argument (`W8A16_FP8_GEMM_ALGOS_SM90`: `hand`, `cutlass_128x256`, `cutlass_128x192`,
`cutlass_128x160`, `cutlass_128x128`, `cutlass_64x256` (ping-pong), `cutlass_128x256_streamk`,
`cutlass_128x192_streamk`; env `FLASHINFER_W8A16_FP8_ALGO_SM90`).  Same operand mapping,
epilogue fusion and 4-wide converter specialisation as the Blackwell header (the ALU path
for bf16, Hopper has no packed e4m3 -> bf16 cvt; native `cvt.rn.f16x2.e4m3x2` for fp16).

Why: the hand kernel's tile height and scheduler are hand-written, so the wave-fill menu and
stream-K that paid off on Blackwell (B13 / B14) would each be a week of kernel work there;
on CUTLASS they are template arguments.

### What the H100 said

Measurement had to change twice.  The node (s4124-0125) throttles thermally under a sweep
(the `SW Thermal Slowdown` counter climbs, cuBLAS itself swings up to 2.5x between
processes at M >= 4096), and a fixed back-to-back variant order parks the middle variants
in the hot state every pass.  The usable method was `sweepH1d`: every launch timed by
`ncu --metrics gpu__time_duration.sum` (clocks locked to base, caches flushed before each
kernel), one variant per process with the bf16 cuBLAS GEMM measured next to it, ratios
per row.  Caveat: the flush between the hand kernel's split-K GEMM and its reduce kernel
inflates those rows a little (the partials would normally be read from L2), which is why
the N <= 1024 rows stay on the hand kernel even where the table calls them a tie.

Geomean kernel / cuBLAS over the 24 shapes of each M class, locked clocks:

| M | hand (v37) | c128x256 | c128x192 | c128x128 | c64x256 | c128x256 stream-K | c128x192 stream-K | best CUTLASS per row | oracle (all) |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1.34 | 1.66 | 1.90 | 1.55 | 1.29 | 1.40 | 1.74 | 1.24 | 1.18 |
| 2048 | 1.20 | 1.33 | 1.82 | 1.37 | 1.15 | 1.26 | 1.66 | 1.13 | 1.10 |
| 4096 | 1.09 | 1.15 | 1.35 | 1.35 | 1.12 | 1.15 | 1.32 | 1.06 | 1.06 |
| 8192 | 1.05 | 1.09 | 1.30 | 1.31 | 1.20 | 1.10 | 1.29 | 1.05 | 1.03 |
| 44160 | 1.03 | 1.11 | 1.14 | 1.33 | 1.71 | 1.09 | 1.12 | 1.06 | 1.00 |
| 88320 | 0.99 | 1.08 | 1.09 | 1.31 | 1.75 | 1.09 | 1.10 | 0.99 | 0.94 |

Per-row winners: hand 53, 128x256 32, 64x256 27, 128x256 stream-K 18, 128x192 stream-K 8,
128x192 3, 128x128 3.  (The 128x160 tile ran in the free-running sweeps only: a dud, one
win in 144 rows, and under ncu it executes twice the instructions of the 256 tile.)

* **Mainloop (H1 + swizzle).** CUTLASS's persistent scheduler defaults to no L2 tile
  grouping; with it the 128x256 mainloop read 2.5x the DRAM bytes of the hand kernel and
  ran 12% slower at full waves (ncu `--set full`, gen.qkv tp1 M=8192: 1.261 vs 1.121 ms,
  tensor pipe 90.5% vs 91.7%, DRAM 56% vs 23% of peak).  `max_swizzle_size = 8`
  (`FLASHINFER_W8A16_FP8_SM90_SWIZZLE`, raster order `FLASHINFER_W8A16_FP8_SM90_RASTER`
  does not matter at 8) closes it on that shape: 1.117 ms.  Across the table it is still
  0-10% behind the hand kernel at full waves, so the hand kernel keeps those rows.
* **Tile heights are not free here.** 256 > 192 > 160 > 128 at full waves: the
  register-sourced mainloop converts the weight tile once per activation tile, so a
  narrower tile converts more per flop (on Blackwell the transform warps are off the MMA's
  critical path, which is why the 160-row tile worked there).  The wave-fill menu does not
  transfer; what does is stream-K (long K, partial first wave: 25-30% over the hand
  kernel's split-K on attn.out / mlp.down / gen.qkv tp2 at M=1024) and the 64x256
  ping-pong tile (short K or modest N: two tiles in flight per SM, 10-25%).
* **Narrow N** (und.kv, N <= 1024): CUTLASS's stream-K and forced split-K have a ~40 us
  floor on these shapes (37-51 us vs the hand kernel's 14-27 us at M=1024); the hand
  kernel's batched split-K + tiny reduce stays.

### Version H2: the dispatcher (`W8A16Sm90PickAlgo`)

Waves of the 128x256 CTA tile (`ceil(N/128) x ceil(M/256)` units over the SM count):
N <= 1024 -> hand.  <= 1.5 waves: K >= 4096 and N >= 5120 -> 128x256 stream-K; short K
(<= 3200) or N <= 2048 (unless N <= 1024 at M >= 4096) -> 64x256; else hand.  1.5-3
waves: K >= 12800 or N >= 8192 -> stream-K; N >= 5120 (or N >= 2560 at M >= 4096) ->
64x256; else hand.  > 3 waves: M <= 2048 with K >= 4096 -> stream-K; else hand.  Explicit
`algo` and the env variable override it; `weight_prepared=True` always runs the hand kernel.

| M | hand (v37) / cuBLAS | H2 auto / cuBLAS | oracle / cuBLAS | picks |
|---|---|---|---|---|
| 1024 | 1.34 | 1.19 | 1.18 | stream-K 10, 64x256 8, hand 6 |
| 2048 | 1.20 | 1.12 | 1.10 | 64x256 11, stream-K 7, hand 6 |
| 4096 | 1.09 | 1.08 | 1.06 | hand 21, 64x256 3 |
| 8192 | 1.05 | 1.05 | 1.03 | hand |
| 44160 | 1.03 | 1.03 | 1.00 | hand |
| 88320 | 0.99 | 0.99 | 0.94 | hand |

Worst row against the hand kernel: gen.qkv tp8 M=2048 (+7%).  Free-running spot check
(attn.out tp1 M=1024: auto 120 us, hand 164, cuBLAS 109; attn.out tp8: 21.6 / 26.6 /
19.4) agrees with the locked-clock table.  Not pursued: the 0-10% full-wave gap of the
CUTLASS mainloop (WARPGROUP wait is 33% of its PC samples vs 14% in the hand kernel, with
fewer instructions executed overall; a cluster of 2 with activation multicast is the
untried knob), and the narrow-N rows at M >= 8192 where 128x128 wins 8-30% over the hand
kernel's split-K.

Results: `/home/scratch.ishovkun_gpu/w8a16-results/hopper/` (`sweepH1b_*`, `sweepH1c_*`
free-running per-variant; `sweepH1d_{rows,ncu}_*` the locked-clock sweep; `ncuH1_*`
`--set full` reports of the hand kernel and the 256 / 192 / 160 / 128 tiles, and the nine
swizzle / raster settings).

**Read the H1 / H2 CUTLASS numbers above with the next section in mind: every CUTLASS
column was taken with the scheduler computing padding tiles on the shapes whose tile counts
are not multiples of 8 (N = 1280, 2560, 3200, 6400 and M = 44160 / 88320), which is where
the "0-10% full-wave gap" came from.**

### Version H3: why CUTLASS looked slower, and the dispatcher refitted (Tue 13 Oct 2026)

The question was why the CUTLASS mainloop trailed the hand kernel at all when the one
`--set full` profile said the two were equal.  `ncu --set full --import-source yes` on the
shape with the widest gap, gen.qkv tp8 M=8192 (N=1280, K=5120): the CUTLASS kernel ran
247 us to the hand kernel's 162 with the tensor pipe *busier* (89.6% vs 72.7%), and the
opcode counts said why: 1,310,720 HGMMA to the hand kernel's 819,200, 1.6x, with the same
factor on L2 sectors, smem wavefronts and instructions.  It was not running the same work
slower; it was doing 60% more work.  `tile_scheduler_params.h` rounds *both* tile counts
up to a multiple of `swizzle x cluster` (`problem_blocks_m = round_up(.., (1 << log_swizzle)
* cluster_m)`, same in the stream-K scheduler) and the kernel runs the padding tiles: TMA
zero-fills them, the epilogue drops them.  With the swizzle 8 that H1 made the default,
N=1280 is 16 weight tiles instead of 10 (+60%), N=3200 is 32 instead of 25 (+28%), N=6400
is 56 instead of 50 (+12%), N=2560 is 24 instead of 20 (+20%), and M=44160 / 88320 pay 1.7-2%
on the activation side.  The 12% swizzle-8 win of H1 was measured on gen.qkv tp1 (80 x 32
tiles): a shape the padding cannot touch.  Forcing swizzle 1 on N=1280 gives 159.5 us,
i.e. the hand kernel's time with fewer instructions.

Fix, first half: `W8A16Sm90PickSwizzle` caps `max_swizzle_size` per problem at the largest
group whose padding is at most 0.5% of the tiles (CUTLASS's own shrink rule for small tile
counts is replicated so the waste is computed for the size it will actually use), else 1;
0.5% because 1.7% of padding at M=44160 cost 3-5% (attn.out tp1: 4.60 vs 4.82 ms).  The
env variable is now the cap, not the value.

Fix, second half, found when the dispatcher's first `auto` sweep ran mlp.gate_up tp1 at
M=44160 in 31 ms instead of 15: the swizzle was never the real knob.  CUTLASS's raster
heuristic takes "along N" (consecutive CTAs walk the activation tiles of one weight tile)
whenever there are at least as many weight tiles as activation tiles, N >= M/2 here, and
for this operand layout that streams the 16-bit activation instead of the fp8 weight:
2x slower once the activation outgrows L2 (mlp.gate_up tp1 31.1 vs 14.9 ms, mlp.down tp1
33.2 vs 14.5, gen.qkv tp1 12.9 vs 5.8 at M=44160; gen.qkv tp1 M=8192 1.155 vs 1.046 ms).
The H1 "swizzle 8 recovers 12%" was the grouping papering over that raster on a shape the
padding could not touch.  With the raster forced along the weight tiles the group buys
nothing (gen.qkv tp1 M=8192: 1.046 vs 1.054 ms; equal on five other shapes), so the
defaults are now raster `m` and swizzle cap 1 (`FLASHINFER_W8A16_FP8_SM90_RASTER`,
`FLASHINFER_W8A16_FP8_SM90_SWIZZLE`).

The whole CUTLASS column was re-swept at locked clocks (`sweepH3_*`, H100 ipp2-0148 for
M <= 8192; its GPU then fell off the bus and ipp2-0069 took M = 44160 / 88320; the hand
kernel re-measured alongside).  Also swept: `W8A16Sm90Cfg<128, 256, 64, 2>`, a cluster of
two CTAs sharing the activation tile by TMA multicast, the hand kernel's arrangement.
Geomean kernel / cuBLAS per M class:

| M | hand (v37) | c128x256 | c128x256 cluster 2 | c128x128 | c64x256 | c128x256 stream-K | best CUTLASS per row | oracle (all) |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1.32 | 1.65 | 1.66 | 1.56 | 1.29 | 1.40 | 1.24 | 1.17 |
| 2048 | 1.20 | 1.33 | 1.34 | 1.38 | 1.13 | 1.24 | 1.11 | 1.08 |
| 4096 | 1.09 | 1.15 | 1.15 | 1.30 | 1.12 | 1.12 | 1.05 | 1.05 |
| 8192 | 1.04 | 1.06 | 1.06 | 1.25 | 1.16 | 1.07 | 1.01 | 1.01 |
| 44160 | 1.03 | 1.04 | 1.04 | 1.25 | 1.74 | 1.06 | 1.03 | 1.02 |
| 88320 | 1.03 | 1.02 | 1.03 | 1.28 | 1.67 | 1.06 | 1.01 | 1.01 |

What changed against the H1 table: with N > 512 the CUTLASS 128x256 tile and the hand
kernel are the same kernel to within 2% on every row, at every M (M=4096 still shows 1.15
vs 1.09 only because of the und.kv N <= 512 rows, where the hand kernel's split-K is
unmatched below M=8192).  The cluster-2 multicast changes nothing (+-2% everywhere, not
instantiated).  The H1 "hand beats cuBLAS by 15% at M=88320" rows did not reproduce on
either new node: the H1 sweep's cuBLAS p90 spread at that M was 1.44, so they were the
first node's thermal state, not a kernel property.  What is left on Hopper is wave
quantisation of a tile that cannot be made shorter (the register-sourced conversion), and
the 44160 rows were still measured with 1.7% padding (the cap was tightened to 0.5%
afterwards; the swizzle-1 probe above says they move to the hand kernel's numbers).

The dispatcher (`W8A16Sm90PickAlgo`) was refitted to read the 128x256 tile's wave count and
the fill of its last wave: one partial wave -> stream-K when K >= 4096 and the wave is at
least 0.7 full, else the 64x256 ping-pong tile; more than 1.5 waves -> 128x256 when the last
wave is >= 0.8 full, stream-K when it is a sliver (< 0.15, K >= 2048, and only up to 16
waves, beyond which the sliver is not worth stream-K's fix-up traffic), ping-pong in
between up to 3 waves, 128x256 beyond; N <= 512 -> the hand kernel's
split-K up to M=4096 (units <= 32), 128x128 under 0.75 of a wave, 128x256 above.  Evaluated
row by row on the sweep:

| M | hand (v37) / cuBLAS | H2 rule on this data | H3 rule | oracle / cuBLAS |
|---|---|---|---|---|
| 1024 | 1.32 | 1.17 | 1.18 | 1.17 |
| 2048 | 1.20 | 1.11 | 1.09 | 1.08 |
| 4096 | 1.09 | 1.08 | 1.06 | 1.05 |
| 8192 | 1.04 | 1.04 | 1.02 | 1.01 |
| 44160 | 1.03 | 1.03 | 1.04 | 1.02 |
| 88320 | 1.03 | 1.03 | 1.02 | 1.01 |

Picks over the 144 rows: 128x256 85, stream-K 24, 64x256 24, hand 8, 128x128 3.  Worst
row against the per-row oracle: und.kv tp4 M=4096 (+12%, a 27 us kernel).  The measured
`auto` column (dispatcher running, same method, raster `m`, no padding) is in the table
below; the 44160 rows improve on the variant sweep, which still carried 1.7% of padding.

| M | hand (v37) / cuBLAS | H3 auto / cuBLAS (measured) | oracle / cuBLAS |
|---|---|---|---|
| 1024 | 1.32 | 1.19 | 1.16 |
| 2048 | 1.20 | 1.10 | 1.08 |
| 4096 | 1.09 | 1.06 | 1.04 |
| 8192 | 1.04 | 1.02 | 1.01 |
| 44160 | 1.03 | 1.01 | 1.01 |
| 88320 | 1.03 | 1.01 | 1.00 |

Against the hand kernel row by row: faster by more than 10% on 29 of the 144 rows (up to
28% on attn.out tp1 / mlp.down tp4 at M=1024), slower by more than 5% on 4 (worst und.kv
tp4 M=2048, +10% on a 17 us kernel, where the dispatcher does call the hand kernel; that
is run-to-run noise at that size).  Where the Hopper line now stands: at cuBLAS from
M=8192 up, and the remaining 2-19% at smaller M is the wave quantisation of a 128x256 tile
that the register-sourced conversion keeps from getting shorter.  The hand kernel is called
for 8 of the 144 rows (N <= 512, M <= 4096); deleting it would cost those rows 5-30%.

**Free-running clocks (Wed 14 Oct 2026, a4u8g-0015).**  Everything above was timed by ncu
at its locked base clock, which is the right tool for counting instructions and finding the
padding tiles but not the number a user sees: on these shapes the GPU sits on its 700 W cap
and the SM clock is set by how many joules per flop each kernel spends.  Re-measured the way
the repo benchmarks measure (CUPTI, L2 flushed, free clocks, `--algo auto hand` back to back
in one process; `freeH3_r*.csv`), with `nvidia-smi dmon` sampling next to it:

| M | hand (v37) / cuBLAS | H3 auto / cuBLAS | mean cuBLAS TFLOP/s | SM clock, busy samples |
|---|---|---|---|---|
| 1024 | 1.27 | 1.15 | 655 | 1980 MHz |
| 2048 | 1.17 | 1.07 | 711 | |
| 4096 | 1.06 | 1.03 | 737 | |
| 8192 | 1.04 | 1.01 | 746 | 1350-1965 |
| 44160 | 1.03 | 1.05 | 716 | 1260-1935, median 1410 |
| 88320 | 1.02 | 1.02 | 671 | 1245-1980, median 1395 |

The "TFLOP/s falls at large M" shape of every line in the plots, cuBLAS included, is the
power cap: from M=8192 up the GPU runs at 1.25-1.5 GHz instead of 1.98, i.e. ~700 of the
989 TFLOP/s headline.  Whether the in-kernel dequant costs clock was checked directly on
und.q tp4 M=88320 (each implementation alone for ~4 s, two passes, 20 s cool-down between;
`clock_*_p{1,2}.txt`): cuBLAS 2545 us at 596-672 W / 1384-1536 MHz, H3 auto 2524-2549 us at
614-661 W / 1444-1462 MHz, hand 2630 us at 617-622 W / 1421-1512 MHz, CUTLASS 128x256 2549
us at 675 W / 1369-1384 MHz.  Same power, same clock, same time: the conversion does not
buy a lower clock.  Row-to-row at M >= 44160 the free-running ratios scatter +-10% (e.g.
und.q tp4 44160 1.14, 88320 1.00; attn.out tp4 88320 0.92) where the locked-clock sweep had
them all within 1-3%: that is the cap moving the clock between the back-to-back
measurements, and the 44160 geomean of 1.05 against 1.01 locked is that noise, not a
regression.  Rows that are 15% off in a single free-running pass at these sizes should be
repeated before being believed, in either direction (H1's "hand beats cuBLAS 15% at 88320"
was one such row).  Hopper plot: `plots/w8a16_h100_tflops_grid.png` now draws this sweep,
auto (H3) and hand (v37) side by side.

Results: `sweepH3_{rows,ncu}_*` (variants and `auto`), `freeH3_r*.csv` / `freeH3_dmon_*` /
`clock_*` (free-running validation), `ncuH3_*` (`--set full` with source
of both kernels on gen.qkv tp8 / mlp.gate_up tp8 / mlp.down tp4 M=88320 / gen.qkv tp1),
`swz_*` and `swz44_*` (the swizzle / raster probes), all under
`/home/scratch.ishovkun_gpu/w8a16-results/hopper/`.

### Version H4: large M, measured cleanly, and an L2 grouping that does not pad (Wed 14 Oct 2026)

Phase 1 of H4 was the 48 rows at M = 44160 / 88320, where the free-running H3 sweep had
`auto` at 1.05 / 1.02 of cuBLAS with single rows at 1.10-1.17 (und.kv tp1 44160 1.17, und.q
tp4 44160 1.14, attn.out tp8 44160 1.10, mlp.gate_up tp1 44160 1.08, mlp.down tp4 88320
1.13) while the locked-clock sweep had them all within 1-3%.  (The narrow-N / small-M work
was parked half-way: see the end of this section.)

**Most of those rows were the measurement.**  Two of the four H100s this ran on were not
usable: ipp2-0720 and ipp2-0711 dropped off the PCIe bus mid-sweep (Xid 79, node needs a
reboot), and s4124-0130 was thermally throttling hard enough to halve every number
(`SW Thermal Slowdown` counter climbing, SM clock 435-1590 MHz under load, cuBLAS
gen.qkv tp1 M=44160 at 13.1 ms instead of 6.4).  A bench pass on that node gave geomeans of
1.14 / 1.12 with rows from 0.8 to 1.9, i.e. noise.  The usable node (s4124-0129) stays at
1245-1665 MHz on the 700 W cap with no thermal events; and the bench got `--rounds R`: each
row's implementations are timed R times in rotated order and the median taken, so the
clock drifting under the power cap lands on every column alike instead of on whichever
column happened to run hot.  With R = 3, two passes, the H3 dispatcher itself measures
1.020 / 1.016 (column `cutlass_128x256`-equivalent rows below) and nothing scatters past
+-8%.  Clock: one implementation alone for ~4 s on mlp.down tp4 M=88320 (`h4p4_clock_*`):
cuBLAS 1432-1504 MHz / 550-629 W, auto 1346-1414 MHz / 574-604 W, hand 1180-1440 MHz:
the same power-capped clock band within its own run-to-run spread, so no row here is
explained by energy per flop.

**What was real, part 1: und.kv at TP 2-8 (N = 128-512).**  2-4 weight tiles x 173-345
activation tiles; the 128x256 tile ran them at 1.06-1.10 of cuBLAS.  The 128x192 tile is
0.93-1.03 on all six rows (`h4p4_kv_*`, 3 rounds): with so few weight tiles the run is
bound by streaming the activation once, and a 192-row tile gives 33% more CTAs in the last
wave (173 x 2 = 346 tiles of 256 rows = 2.62 waves; 230 x 2 = 460 tiles of 192 = 3.48).
Dispatcher: N <= 512 and M >= 16384 -> 128x192.

**What was real, part 2: L2 traffic (the Blackwell B15 lead).**  With swizzle 1 (H3's
default, because CUTLASS's swizzle pads the tile grid) the 132 CTAs in flight walk ~132
weight tiles of one or two activation tiles, so each activation tile re-reads the weight
from HBM once it no longer fits in L2 next to the activation band.  ncu (locked clocks,
counters only, `h4n_*`, `h4g_ncu_*`) agrees: DRAM bytes read per kernel, CUTLASS 128x256
ungrouped vs cuBLAS:

| row | cuBLAS | H3 (no group) | group 2 | group 4 | group 8 |
|---|---|---|---|---|---|
| mlp.gate_up tp1 M=44160 | 12.2 GB, 14.13 ms | 22.6 GB, 1.055x | 13.7 GB, 1.030x | **8.4 GB, 1.024x** | 9.9 GB, 1.032x |
| mlp.gate_up tp1 M=88320 | 24.9 GB, 28.35 ms | 45.0 GB, 1.051x | 29.7 GB, 1.031x | **20.0 GB, 1.023x** | 25.9 GB, 1.033x |
| und.q tp1 M=44160 | 4.2 GB, 4.49 ms | 5.2 GB, 1.037x | 4.5 GB, 1.029x | **3.1 GB, 1.022x** | 3.4 GB, 1.033x |
| gen.qkv tp1 M=44160 | 7.1 GB, 5.78 ms | 8.1 GB, 1.013x | 5.3 GB, 0.999x | **3.7 GB, 0.992x** | 4.2 GB, 1.000x |
| mlp.down tp1 M=44160 | 26.3 GB, 14.85 ms | 22.8 GB, 0.979x | **18.1 GB, 0.967x** | 18.5 GB, 0.974x | 29.3 GB, 1.009x |
| und.q tp4 M=44160 | 0.93 GB, 1.08 ms | **0.67 GB, 1.013x** | 0.76 GB, 1.020x | 0.87 GB, 1.029x | 1.13 GB, 1.041x |
| attn.out tp8 M=44160 | 0.21 GB, 0.59 ms | **0.10 GB, 1.033x** | 0.12 GB | 0.14 GB | 0.16 GB, 1.043x |

The 10 MB weight of und.q tp4 already stays in L2; there the group only makes the
activation re-read, and that row's H3 "1.14" was the noise above (1.01 measured cleanly,
ncu 1.013).  Same for attn.out tp8: H3 already reads half of cuBLAS's bytes, the remaining
3-5% is not memory.  Where the weight is >= 40 MB (und.q / gen.qkv / mlp.gate_up / mlp.down /
attn.out tp1, mlp.gate_up / mlp.down tp2), a group of 4 activation tiles cuts DRAM reads
1.6-2.7x and buys 1-3% of time; 2 is the better size for the 40-weight-tile shapes (N =
5120) and 8 is worse than 4 everywhere (an 8-tile activation band, 8 x 256 x K x 2 B = 21 MB
at K = 5120, crowds the weight out of L2 again).  The time win is small because these
kernels run at 720-750 TFLOP/s already and HBM was never the bound; it is the traffic (and
the HBM energy under the power cap) that moves.

**The scheduler** (`W8A16Sm90GroupedTileScheduler` in `w8a16_fp8_gemm_sm90.cuh`,
selected by a tag through CUTLASS's `TileSchedulerSelector`): a subclass of CUTLASS's sm90
persistent scheduler that keeps its swizzle at 1 (so grid size, work count and everything
the kernel asks of the scheduler are CUTLASS's own, no padding) and replaces only the
linear-index -> (weight tile, activation tile) map: groups of G activation tiles, the
weight tiles walked inside a group, and the last group narrower instead of padded (the
hand kernel's rasterisation).  G = 1 is CUTLASS's order exactly.  All plain cluster-1
variants use it; the stream-K variants keep CUTLASS's scheduler.  `W8A16Sm90PickGroup`:
G = 4 if N*K >= 40 MB and >= 64 weight tiles, G = 2 if N*K >= 40 MB otherwise, 1 below 40
MB or at M < 16384 (at M = 4096 / 8192 the group cuts reads 20-30% but the time moves
+-1%, so those rows keep the H3 schedule bit for bit).  Env `FLASHINFER_W8A16_FP8_SM90_GROUP`
overrides it: cached as absent if unset at the first read, re-read per call only when it
was set then (the bench sets it before the first GEMM, so `--algo auto@g<n>` columns
compare group sizes in one process).  The index map uses FastDivmod precomputed on the host
(CUTLASS's own scheduler does the same).

**Raster per shape** (option c of the brief) was not pursued: with the weight grouped the
`n` raster would only stream the activation instead, which is the 2x loss H3 already
documented.

Result, free-running, CUPTI, L2 flushed, `--rounds 3`, two passes on s4124-0129, ratio to
the cuBLAS measured in the same rounds (`h4_conf_p{1,2}_r*.csv`).  "H3 sweep" is the
single-pass freeH3 column the brief started from (a different node).  "H3 rule, this node"
is the H3 choice re-measured here: the `auto@g1` column (same tiles as H3, grouping off),
and on und.kv tp2-8 the `cutlass_128x256` column (H3's tile there; group 1 on those rows).
Each H4 change on its own: the tile change shows in `auto@g1` on und.kv, the grouping in
`auto` vs `auto@g1` elsewhere.

| row | M | H3 sweep | H3 rule, this node | **H4 auto** | hand |
|---|---|---|---|---|---|
| und.kv tp1 | 44160 | 1.17 | 1.02 | 1.00 | 1.00 |
| und.kv tp2 | 44160 | 1.08 | 1.10 | **0.95** | 1.12 |
| und.kv tp8 | 44160 | 1.13 | 1.08 | **1.00** | 1.09 |
| und.kv tp2 | 88320 | 1.09 | 1.06 | **1.00** | 1.06 |
| und.kv tp4 | 88320 | 1.06 | 1.09 | **0.98** | 1.18 |
| und.q tp4 | 44160 | 1.14 | 1.02 | 1.01 | 1.00 |
| und.q tp8 | 44160 | 1.11 | 0.99 | 1.00 | 0.99 |
| attn.out tp8 | 44160 | 1.10 | 1.04 | 1.05 | 1.04 |
| mlp.gate_up tp1 | 44160 | 1.08 | 1.06 | **1.03** | 1.01 |
| mlp.gate_up tp1 | 88320 | 1.04 | 1.07 | **1.03** | 1.03 |
| mlp.gate_up tp2 | 44160 | 1.07 | 1.04 | **1.00** | 0.99 |
| mlp.gate_up tp2 | 88320 | 1.05 | 1.07 | **1.05** | 1.04 |
| mlp.down tp4 | 88320 | 1.13 | 1.05 | 1.02 | 1.06 |
| attn.out tp2 | 88320 | 1.11 | 1.00 | 1.04 | 1.04 |

| M | H3 rule / cuBLAS (this node) | **H4 auto** | hand (v37) |
|---|---|---|---|
| 44160 | 1.022 | **1.009** | 1.021 |
| 88320 | 1.014 | **1.003** | 1.024 |

So the "1.10-1.17" rows split three ways: und.q tp4 / tp8, und.kv tp1, mlp.down tp4,
attn.out tp2 were the old node's clock (1.00-1.05 here with the H3 code); und.kv tp2-8 were
real and are fixed by the 128x192 tile (6-11%); mlp.gate_up tp1 / tp2 were real and are the
L2 traffic, fixed partly by the grouping (3-4%; the ncu table above says the remaining
2-3% is not DRAM).  Rows over 1.04 left: attn.out tp8 44160 (1.05; ncu 1.033 with half of
cuBLAS's DRAM reads: a 0.6 ms kernel, the wave tail of 40 x 173 tiles on 132 SMs, not
memory), gen.qkv tp4 44160 (1.05, ncu 1.02), mlp.gate_up tp2 88320 (1.05, the same in every
column incl. hand), attn.out tp2 88320 (1.04, 1.00 at group 1 in this run, 1.01-1.04 at
group 2 under ncu: within the spread).  Nothing got slower beyond +-4% single-row scatter.

Whole grid after the change (one free-running pass at M <= 8192 with 2 rounds, `h4_grid_*`;
M <= 8192 rows run the H3 code paths unchanged): 1.149 / 1.077 / 1.037 / 1.018 for M =
1024-8192 against 1.146 / 1.073 / 1.026 / 1.007 in the freeH3 sweep, the same within this
node-to-node spread.

**Follow-up: the "3-5% at G = 1" on mlp.gate_up tp1 was noise.**  The H4 table shows
mlp.gate_up tp1 at 1.06-1.07 with the grouped scheduler at G = 1 against 1.00-1.02 for "the
H3 rule", which looked like the scheduler costing 3-5% on its own.  Settled with a
temporary build carrying both schedulers (s4124-0106, `h4ab_*`), same process, 5
interleaved rounds per row: stock CUTLASS scheduler / grouped G = 1 with plain integer
division / grouped G = 1 with FastDivmod / grouped G = 4:

| row | stock | G=1 plain div | G=1 FastDivmod | G=4 |
|---|---|---|---|---|
| mlp.gate_up tp1 44160 | 1.053 | 1.070 | 1.067 | **1.009** |
| mlp.gate_up tp2 44160 | 1.059 | 1.060 | 1.057 | **1.010** |
| mlp.gate_up tp1 88320 | 1.05 | 1.06 | 1.07 | **1.02** |
| mlp.gate_up tp2 88320 | 1.05 | 1.05 | 1.04 | **1.01** |
| geomean, 12 rows (gate_up / und.q / gen.qkv tp1-2), 44160 | 1.039 | 1.034 | 1.032 | 1.024 |
| geomean, same, 88320 | 1.019 | 1.022 | 1.026 | 1.010 |

ncu on mlp.gate_up tp1 44160 (locked clocks, counters only): 14.87 / 14.83 / 14.99 /
14.77 ms, 3316 / 3320 / 3313 / 3305 M instructions, DRAM read 22.3 / 18.1 / 22.7 / 13.5 GB.
The stock scheduler and the grouped one at G = 1 are the same kernel (instruction counts
within 0.2%, times within 1%; the DRAM bytes of the ungrouped runs vary 18-23 GB between
launches because which weight slices survive in L2 depends on CTA timing).  The H4 table's
"H3 rule" 1.00-1.02 on that row was the spread of that particular run.  The FastDivmod map
stays (no cost, matches CUTLASS); `is_last_tile` now only compares the next index against
the work count instead of running the map.  The grouping itself is confirmed: G = 4 is
4-5% under every G = 1 column on the four gate_up rows, in the same process.

**Floor check of the rows still above 1.04, and one dispatcher miss.**  Each pinned to
128x256 / 64x256 / 128x192 at group 1 / 2 / 4 and to 128x256 stream-K, 5 interleaved
rounds (`h4fl_*`, `h4fl2_*`):

* gen.qkv tp4 44160: auto 1.024, best 128x192 at group 1 0.989; a two-pass repeat gives
  1.01 vs 1.00.  Nothing beats auto by more than 3%: a floor.
* mlp.gate_up tp2 88320: auto (group 4) 1.020, group 2 1.008 (the order flips between runs),
  stream-K 1.045, 128x192 1.08-1.12, 64x256 1.31-1.65.  A floor.
* attn.out tp8 (N = 5120, K = 1024): the 64x256 ping-pong tile, ungrouped, beats 128x256 in
  every run: 0.94 / 0.99 vs 1.00 / 1.01 at M = 44160, 0.97 vs 1.02 at 88320.  16 k-tiles per
  tile is a short mainloop, and two tiles in flight per SM hide the epilogue behind it.  It
  does not carry to longer K: 6-23% worse on attn.out / mlp.down / gen.qkv tp2 / tp4 and
  mlp.down tp8 (K = 2048-12800), a tie on attn.out tp4 (K = 2048).  Dispatcher: K <= 1024 and
  M >= 16384 -> 64x256 (the weight is 5 MB, so it stays ungrouped); of the 48 large-M rows
  only attn.out tp8 matches.

Re-confirm with the rule (`h4c2_*`, 48 rows, `--rounds 3`, two passes): geomean auto /
cuBLAS 1.008 / 1.008 at M = 44160 / 88320 (128x256 pinned 1.018 / 1.007, hand 1.026 /
1.020).  attn.out tp8: 0.96 / 0.96 at 44160 (128x256 in the same rounds 1.02 / 1.04), 1.06 /
1.01 at 88320 (1.17 / 0.99), and 1.01 vs 1.11 in a 5-round repeat.  Six rows moved by more
than 3% against the previous confirm (gen.qkv tp2 44160, mlp.down tp2 44160 / 88320,
attn.out tp4 44160, und.q tp4 88320, gen.qkv tp8 88320) although their code path did not
change (the new rule does not fire on them); a 5-round, two-pass repeat (`h4c3_*`) puts all
six at 0.98-1.02, equal to 128x256 pinned in the same rounds: run-to-run spread at 3 rounds.

**Parked: narrow N / small M** (`~/code/worktrees/flashinfer/h4-parked/`, a patch plus
the M = 1024 / 2048 sweep).  Tried 128x64 (cooperative), 64x128 and 64x64 (ping-pong), and
plain split-K through the stream-K scheduler (`DecompositionMode::SplitK`) on 128x64 /
128x128 / 128x256.  Small tiles lose 1.5-2.5x on wide N (the register-sourced conversion
per activation tile again); 64x128 is the one that wins rows: und.q tp8 and und.kv tp1 at
M=1024 (1.38 -> 1.11 and 1.36 -> 1.07 of cuBLAS) and und.kv tp2 at M=2048 (1.36 -> 1.09).
CUTLASS's split-K does not: 24-45 us where the hand kernel's split-K + reduce takes 14 us
(und.kv tp8 M=1024); its reduction runs serially through the workspace and the deterministic
mode serialises the peers.  gen.qkv tp8 (N=1280) at M <= 2048 stays at 1.24-1.52 with
nothing on the menu better than the hand kernel.

Results: `h4n_*` (ncu, 48 rows x groups 1/2/4/8), `h4g_ncu_*` (ncu incl. group 16 and
hand on 7 rows), `h4p4_*` (und.kv tiles, mlp.down groups, clock-alone runs),
`h4_conf_*` (the table), `h4_grid_*` (whole grid), `h4f_*`, `h4m_*` (M = 4096 / 8192 group
counters), `h4_probe_r{1024,2048}.csv` (the parked small-M sweep), under
`/home/scratch.ishovkun_gpu/w8a16-results/hopper/`.

### Version H5: the gap that was not there (Wed 14 Oct 2026)

The H100 plot (drawn from the single-pass `freeH3_*` sweep) showed `auto` above the torch
cast-copy column (dequantize the weight to bf16, then cuBLAS) on the small-N rows at M =
44160 / 88320 (und.kv tp1-8, und.q tp8, gen.qkv tp8, N <= 1280): e.g. und.kv tp1 44160 1.116
of cast, gen.qkv tp8 88320 1.109, und.kv tp8 44160 1.067.  The suspicion was that the
register-sourced mainloop idles the tensor pipe while the math warpgroups convert the fp8
weight fragments, which a shared-memory conversion stage would hide.  No code changed in
H5; this section is the measurement that says why not.

**The cast column, current code (c93a53ba9), 5 interleaved rounds, cast / cuBLAS / auto
timed in the same rounds** (s4124-0106, no thermal events, `h5b_*`):

| row | N | M | cast / cuBLAS | auto / cuBLAS | **auto / cast** |
|---|---|---|---|---|---|
| und.q tp1 | 8192 | 44160 | 1.025 | 1.006 | 0.982 |
| und.kv tp1 | 1024 | 44160 | 1.042 | 0.993 | 0.952 |
| gen.qkv tp1 | 10240 | 44160 | 1.031 | 1.012 | 0.981 |
| und.q tp2 | 4096 | 44160 | 1.055 | 1.005 | 0.953 |
| und.kv tp2 | 512 | 44160 | 1.060 | 0.972 | 0.917 |
| gen.qkv tp2 | 5120 | 44160 | 1.077 | 1.006 | 0.934 |
| und.q tp4 | 2048 | 44160 | 1.069 | 1.018 | 0.952 |
| und.kv tp4 | 256 | 44160 | 1.057 | 0.987 | 0.933 |
| gen.qkv tp4 | 2560 | 44160 | 1.051 | 1.024 | 0.974 |
| und.q tp8 | 1024 | 44160 | 1.074 | 1.004 | 0.935 |
| und.kv tp8 | 128 | 44160 | 1.054 | 0.995 | 0.944 |
| gen.qkv tp8 | 1280 | 44160 | 1.065 | 0.997 | 0.936 |
| und.q tp1 | 8192 | 88320 | 1.027 | 1.035 | 1.008 |
| und.kv tp1 | 1024 | 88320 | 1.038 | 1.037 | 0.999 |
| gen.qkv tp1 | 10240 | 88320 | 1.016 | 1.010 | 0.995 |
| und.q tp2 | 4096 | 88320 | 1.028 | 0.988 | 0.961 |
| und.kv tp2 | 512 | 88320 | 1.062 | 0.996 | 0.938 |
| gen.qkv tp2 | 5120 | 88320 | 1.024 | 1.000 | 0.976 |
| und.q tp4 | 2048 | 88320 | 1.044 | 1.014 | 0.972 |
| und.kv tp4 | 256 | 88320 | 1.036 | 0.990 | 0.956 |
| gen.qkv tp4 | 2560 | 88320 | 1.059 | 0.996 | 0.941 |
| und.q tp8 | 1024 | 88320 | 1.021 | 1.019 | 0.998 |
| und.kv tp8 | 128 | 88320 | 1.039 | 1.008 | 0.970 |
| gen.qkv tp8 | 1280 | 88320 | 1.008 | 1.026 | 1.018 |

The flagged rows were the H3 code measured in one pass on a different node (the same rows
now: 0.952 / 0.944 / 0.935 / 1.018 / 0.998 / 0.938 of cast).  The cast is also not a rounding
error at this size: it costs 1-8% over cuBLAS.

**ncu, auto vs the cuBLAS kernel, M = 88320** (locked clocks, counters only, `--set full`,
one launch each, SM clocks equal within 1% per row; `h5n_*.ncu-repz`):

| row | tiles (ours / cuBLAS) | time | tensor pipe active | DRAM read | instructions | issue slots busy |
|---|---|---|---|---|---|---|
| und.q tp8 (N=1024) | 128x256 / 256x128 | 1.101 / 1.093 ms (+0.7%) | 96.2 / 96.9% | 1.04 / 1.15 GB | 264 / 109 M | 27.6 / 11.6% |
| gen.qkv tp8 (N=1280) | 128x256 / 320x128 | 1.425 / 1.391 ms (+2.4%) | 96.2 / 95.9% | 1.13 / 1.21 GB | 330 / 190 M | 26.8 / 15.9% |
| und.kv tp4 (N=256) | 128x192 / 128x192 | 340.7 / 343.9 us (-0.9%) | 73.9 / 72.7% | 0.91 / 0.92 GB | 87 / 45 M | 27.5 / 13.9% |
| und.kv tp8 (N=128) | 128x256 / 128x256 | 337.4 / 333.7 us (+1.1%) | 39.2 / 42.5% | 0.91 / 0.91 GB | 44 / 14 M | 13.5 / 4.4% |
| und.q tp1 (N=8192, reference) | 128x256 / 192x192 | 9.30 / 9.10 ms (+2.2%) | 96.5 / 97.1% | 6.98 / 8.87 GB | 2109 / 974 M | 27.7 / 12.9% |

* The tensor pipe does not idle during the conversion: 96% active for both kernels where
  the GEMM is MMA-bound.  The conversion makes us issue 2-3x the instructions, but the issue
  slots are 27% busy; it fits beside the wgmma.  A shared-memory conversion stage (the
  Blackwell design) would buy nothing here and was not built.
* und.kv at N = 128-256 is HBM-bound for both kernels: they read the same 0.91 GB per launch
  (the 1.8 GB activation, streamed once, the rest from L2) at about 2.7 TB/s.  We are at
  cuBLAS within +-1%.
* We read the same or fewer DRAM bytes than cuBLAS on every row (the m raster walks the
  weight tiles fastest, so the activation streams once).
* The remaining 0.7-2.4% at equal tensor-pipe utilisation is the wave tail of the tile
  choice (gen.qkv tp8: 3450 tiles of 128x256 = 26.1 waves, a 14%-full last wave), which the
  H4 floor check already showed is not worth a dispatcher rule (64x256 1.00, 128x192 1.02,
  auto 1.01 at 5 rounds).

**Whole grid for the plot** (`freeH4_r*.csv`, all 24 shapes x 6 M, one M class per process,
`--rounds 5`, `--impls cast cublas kernel --algo auto hand`, same node).  Geomean per M
class, lower is better:

| M | auto / cuBLAS | hand (v37) / cuBLAS | cast / cuBLAS | auto / cast | rows with auto > cast |
|---|---|---|---|---|---|
| 1024 | 1.134 | 1.263 | 2.802 | 0.405 | 0 / 24 |
| 2048 | 1.069 | 1.165 | 2.002 | 0.534 | 0 / 24 |
| 4096 | 1.033 | 1.075 | 1.544 | 0.670 | 0 / 24 |
| 8192 | 1.018 | 1.049 | 1.291 | 0.789 | 0 / 24 |
| 44160 | 1.007 | 1.026 | 1.055 | 0.955 | 1 / 24 |
| 88320 | 1.011 | 1.026 | 1.029 | 0.982 | 4 / 24 |

The five rows over cast are all within 1.5% of it, and all are tp1 or long-K shapes where
the dequant is a small share of the work (cast / cuBLAS 1.01-1.02): gen.qkv tp1 44160 /
88320 (1.004 / 1.004), und.kv tp1 88320 (1.015), attn.out tp1 88320 (1.000), mlp.down tp4
88320 (1.013).  None is a small-N row.

**Does the group size need to scale with K (the Blackwell B15 finding)?**  On B200 the
last gap on the wide rows was the SM clock following the DRAM bytes under the power cap,
and the group had to shrink as K grew.  Checked on H100 for the K-extremes of the grouped
rows, mlp.down tp1 / tp2 (K = 25600 / 12800) and mlp.gate_up tp1 / tp2 (K = 5120), at
G = 1 / 2 / 4 / 8: ncu with `--clock-control none` (free-running clock, counters only:
time, DRAM read bytes, `sm__cycles_elapsed.avg.per_second`, cycle count), plus a 5-round
free-running pass (`h5kg_*`):

| row | cuBLAS | G=1 | G=2 | G=4 | G=8 |
|---|---|---|---|---|---|
| mlp.down tp1 44160 | 14.60 ms, 22.0 GB, 1.52 GHz | 14.87, 25.6, 1.49 | 14.65, 22.8, 1.51 | **14.50, 18.3, 1.52** | 14.84, 23.6, 1.49 |
| mlp.down tp1 88320 | 29.32, 45.9, 1.50 | 29.64, 49.8, 1.48 | 30.39, 38.7, 1.44 | **29.18, 40.5, 1.50** | 29.89, 52.5, 1.47 |
| mlp.down tp2 44160 | 7.25, 9.9, 1.54 | 7.32, 10.4, 1.53 | 7.24, 9.8, 1.54 | **7.21, 7.8, 1.54** | 7.32, 9.9, 1.52 |
| mlp.down tp2 88320 | 14.51, 20.7, 1.53 | 14.70, 23.1, 1.51 | 14.54, 21.0, 1.52 | **14.49, 17.0, 1.52** | 14.72, 21.0, 1.50 |
| mlp.gate_up tp1 44160 | 14.60, 16.4, 1.54 | 14.47, 11.2, 1.55 | 14.74, 19.1, 1.52 | 14.62, 11.9, 1.54 | 14.63, 10.2, 1.54 |
| mlp.gate_up tp1 88320 | 29.30, 33.9, 1.53 | 29.22, 27.6, 1.53 | 29.70, 39.4, 1.51 | 29.43, 26.4, 1.52 | 29.70, 27.0, 1.51 |
| mlp.gate_up tp2 44160 | 7.22, 7.5, 1.55 | 7.19, 5.7, 1.56 | 7.28, 8.6, 1.54 | 7.24, 5.8, 1.55 | 7.31, 6.1, 1.55 |
| mlp.gate_up tp2 88320 | 14.56, 15.6, 1.54 | 14.55, 13.4, 1.54 | 14.72, 18.4, 1.52 | 14.63, 12.6, 1.53 | 14.73, 13.1, 1.52 |

Free-running, 5 rounds, ratio to cuBLAS (G = 1 / 2 / 4 / 8): mlp.down tp1 1.00 / 1.00 /
1.01 / 1.01 at 44160 and 0.99 / 1.00 / 1.01 / 1.02 at 88320; mlp.down tp2 1.00 / 1.00 / 1.00 /
1.01 and 1.01 / 1.00 / 1.01 / 1.04; mlp.gate_up tp1 1.08 / 1.02 / 1.02 / 1.03 and 1.07 / 1.05 /
1.02 / 1.04; mlp.gate_up tp2 1.04 / 1.01 / 1.01 / 1.01 and 1.05 / 1.05 / 1.01 / 1.03.

* The clock does follow the bytes on H100, but weakly: over all 120 launches (bytes and clock
  normalised per row) the correlation is -0.48 and the slope is -2.7% of clock per doubling of
  DRAM bytes (bytes range 0.47-1.75x of the row median, clock range 0.92-1.03x).  The cycle
  count per kernel varies by only +-1.5%, so the time differences are clock differences.
* At K = 12800 / 25600 the shipped group for mlp.down (G = 2, fewer than 64 weight tiles)
  is not the byte-minimum on three of the four rows: G = 4 reads about 20% fewer bytes there
  (22.8 -> 18.3, 9.8 -> 7.8, 21.0 -> 17.0 GB) and 5% more on mlp.down tp1 88320 (38.7 ->
  40.5 GB), and its clock is equal or higher on all four (1.44 -> 1.50 GHz on that last
  row, so the clock there does not follow the bytes).  In time, G = 2 and G = 4 are within
  1% at 5 rounds on all four mlp.down rows, so this is not worth changing the rule.  G = 8
  is worse everywhere (more bytes, lower clock).  Single-launch ncu byte counts at G = 1-2
  scatter by +-20% (which weight slices survive in L2 depends on CTA timing), so only
  differences larger than that are read here.
* So the B15 effect exists on H100 but is about a quarter of its size on B200 (H100 HBM3 at
  700 W gives the memory a smaller share of the power budget).  The group rule stays K-blind.
  The G = 1 vs G = 4 gap on mlp.gate_up (2-5% at 5 rounds) is the one place where it shows.

Results: `h5b_*` (cast column), `h5n_*` (ncu reports and extracts), `h5kg_*` (group vs K,
free-clock ncu and 5-round pass), `freeH4_r*.csv` /
`freeH4_dmon_r*.txt` (the plot grid; also packed as `../freeH4.tgz`), under
`/home/scratch.ishovkun_gpu/w8a16-results/hopper/`.

### Version H6: 21 of 24 at M=88320, the converter's energy floor (Wed 14 Oct 2026)

Goal: every one of the 24 rows at M = 88320 at or under 1.02 of cuBLAS (median, two
independent runs agreeing) and at or under the cast-copy column.  No dispatcher change
came out of it; the three rows that miss are explained below.

**Measurement first.**  The 5-round short bursts (`--iters 10-20 --rounds 5`) are not good
enough at this size: the same row flips between 0.98 and 1.05 between runs (e.g.
mlp.down tp4 1.016 / 1.048 / 1.050, attn.out tp4 1.091 / 1.051 / 1.009).  Launch bursts of
10-20 back-to-back kernels alternating with other kernels never settle the power state.
`--iters 200 --warmup 20 --rounds 3` (the Blackwell recipe) gives two runs that agree
within 1% on 16 of 24 rows and within 2% on the rest (`h6L_*`).  Every number below uses it.
The bench also got two fixes for this (commit dcf90de64): kernel columns share one output
buffer (16 columns at M = 88320 ran out of memory), and `--rounds` rotates the column list
by an even share per round, so every column visits the front, middle and back.

**The 24 rows, auto vs cuBLAS / cast, two runs** (`h6L_p{1,2}_r88320.csv`):

| row | auto / cuBLAS | cast / cuBLAS |
|---|---|---|
| und.q tp1 | 1.013 / 1.014 | 1.026 / 1.027 |
| und.kv tp1 | 0.998 / 1.006 | 1.006 / 1.011 |
| gen.qkv tp1 | 1.001 / 1.000 | 1.017 / 1.017 |
| attn.out tp1 | **1.019 / 1.030** | 1.016 / 1.026 |
| mlp.gate_up tp1 | 1.021 / 1.010 | 1.025 / 1.017 |
| mlp.down tp1 | 0.992 / 0.995 | 1.016 / 1.015 |
| und.q tp2 | 0.987 / 0.994 | 1.027 / 1.026 |
| und.kv tp2 | 0.967 / 0.978 | 1.010 / 0.987 |
| gen.qkv tp2 | 0.999 / 0.998 | 1.027 / 1.027 |
| attn.out tp2 | 1.002 / 0.994 | 1.016 / 1.017 |
| mlp.gate_up tp2 | 1.036 / 1.016 | 1.027 / 1.023 |
| mlp.down tp2 | 1.003 / 0.997 | 1.027 / 1.024 |
| und.q tp4 | 1.009 / 1.028 | 1.016 / 1.034 |
| und.kv tp4 | 0.981 / 0.979 | 1.018 / 1.007 |
| gen.qkv tp4 | 1.007 / 1.013 | 1.014 / 1.021 |
| attn.out tp4 | 1.000 / 1.001 | 1.024 / 1.023 |
| mlp.gate_up tp4 | 1.003 / 1.004 | 1.016 / 1.016 |
| mlp.down tp4 | **1.031 / 1.030** | 1.027 / 1.026 |
| und.q tp8 | 1.008 / 1.009 | 1.002 / 1.029 |
| und.kv tp8 | 1.003 / 1.003 | 1.041 / 1.042 |
| gen.qkv tp8 | 1.039 / 1.030 | 0.989 / 0.994 |
| attn.out tp8 | **1.036 / 1.017** | 1.039 / 0.999 |
| mlp.gate_up tp8 | 0.994 / 1.005 | 1.016 / 1.026 |
| mlp.down tp8 | 1.012 / 0.992 | 1.026 / 1.015 |

The rows over the bar in either run were re-measured with every variant in the same rounds
(auto, hand, 128x256 stream-K, 128x256 at G = 1 / 2 / 4, 64x256, 128x192 at G = 1 / 2; long
recipe, two passes, `h6o_*`):

| row | auto | best other variant | cast | verdict |
|---|---|---|---|---|
| mlp.down tp4 (K=6400) | 1.031 / 1.030 | all variants 1.030-1.040 | 1.023 / 1.027 | **fails, nothing helps** |
| attn.out tp1 (K=8192) | 1.025 / 1.029 | 128x256 G=2 / G=1 1.024 / 1.028 | 1.022 / 1.026 | **fails by 0.5-1%, at cast** |
| attn.out tp8 (K=1024) | 1.037 / 1.031 | 128x256 G=4 1.002 / 1.039, 128x192 G=2 1.010 / 1.039 | 1.014 / 1.024 | **fails, no variant wins both passes** |
| gen.qkv tp8 | 1.018 / 1.019 | hand 1.013 / 1.002 | 1.020 / 1.013 | passes |
| mlp.gate_up tp2 | 1.018 / 1.013 | hand 1.017 / 1.012 | 1.027 / 1.022 | passes |
| und.q tp4 | 0.998 / 0.988 | 128x256 G=1 0.996 / 0.997 | 1.014 / 1.004 | passes |
| mlp.gate_up tp1 | 1.018 / 1.009 | hand 1.015 / 1.011 | 1.025 / 1.014 | passes |

So 21 of 24 rows pass; mlp.down tp4, attn.out tp1 and attn.out tp8 sit at 1.025-1.035.

**Why: cycles are fine, the clock is not.**  ncu on our kernel and the cuBLAS kernel at M =
88320, once at locked base clock and once free-running (`--clock-control none`), counters
only (`h6n_*`).  Ratios ours / cuBLAS:

| row | locked: time / cycles | free-running: time / cycles / SM clock | DRAM bytes | instructions | issue slots busy (ours / cuBLAS) |
|---|---|---|---|---|---|
| mlp.down tp4 | 0.995 / 0.997 | 1.008 / 0.991 / **0.980** | 0.64 | 1.73 | 27.7 / 15.8% |
| attn.out tp1 | 0.988 / 0.997 | 1.008 / 0.989 / **0.980** | 0.62 | 1.73 | 27.6 / 15.7% |
| attn.out tp8 | 0.974 / 0.977 | 1.018 / 0.974 / **0.955** | 0.50 | 1.82 | 30.6 / 16.3% |
| und.q tp2 (passes) | 0.977 / 0.986 | 0.964 / 0.980 / 1.016 | 0.49 | 1.71 | 27.7 / 15.9% |
| mlp.down tp1 (passes) | 0.982 / 0.990 | 0.980 / 0.991 / 1.012 | 0.73 | 1.72 | 27.5 / 15.8% |

Our kernel takes 1-3% fewer cycles than cuBLAS on every row and is faster on all of them at
locked clocks.  At free-running clocks the failing rows lose 2-4.5% of SM clock, although
we read 36-50% fewer DRAM bytes.  So it is not memory energy (the Blackwell B15 effect); it
is compute energy.  The in-register fp8 -> bf16 conversion (two prmt, two shift-and-mask,
two bf16x2 multiplies per four weights) makes us issue 1.7-1.8x cuBLAS's instructions, and
under the 700 W cap that ALU and register-file power is paid in clock.  The same split
(fewer cycles, lower clock, many more instructions) is what the Blackwell branch measures
on its three remaining rows.  Rows where we pass are the ones where the tile choice gains
more cycles than the conversion costs in clock.

**What was rejected:**
* A shared-memory conversion stage: same number of conversions, plus smem traffic.  It does
  not reduce the energy per flop.
* Dropping the bf16x2 multiply in the converter by folding 2^120 into the epilogue scale.
  That scales every product by 2^-120 before the fp32 accumulate; any act x w below 2^-6
  becomes an fp32 denormal inside the MMA, which the tensor cores flush to zero.  With
  weights ~0.02 and activations ~1 that is a large share of the terms.  The every-fp8-value
  test would not catch it (identity activation, products = weights).
* Per-row dispatcher picks: none wins on both passes on the three failing rows.

**fp16 as a probe of the converter cost.**  fp16 activations use the native
`cvt.rn.f16x2.e4m3x2` (one instruction per two weights instead of three).  Long recipe, two
passes, ratio to the cuBLAS of the same dtype (`h6f16_*`):

| row | bf16 auto / cuBLAS | fp16 auto / cuBLAS | ours fp16 / bf16 (time) | cuBLAS fp16 / bf16 (time) |
|---|---|---|---|---|
| mlp.down tp4 | 1.030 / 1.029 | 0.968 / 0.964 | 1.001 | 1.067 |
| attn.out tp1 | 1.030 / 1.029 | 0.997 / 0.997 | 1.001 | 1.033 |
| attn.out tp8 | 1.033 / 1.051 | 0.995 / 0.976 | 0.985 | 1.041 |
| und.q tp2 | 0.986 / 0.986 | 0.951 / 0.944 | 1.006 | 1.047 |

The fp16 ratios pass, but not because our kernel got faster: our time is the same in both
dtypes (+-1.5%), and it is cuBLAS that is 3-7% slower in fp16 than in bf16.  ncu on
mlp.down tp4 (free-running): the instruction ratio falls from 1.73 to 1.16 with the native
convert and our clock rises from 0.980 to 1.025 of cuBLAS's.  But part of that is cuBLAS's own
clock dropping in fp16 (1.583 -> 1.507 GHz, ours 1.551 -> 1.544 GHz).  So a cheaper bf16
converter would not, on this evidence, be worth the full 3%; the fp16 run bounds the gain at
"our clock stops being the lowest", which here means about 1-2%.  Not pursued.

**The tightest recipe on the three rows** (ipp2-0073, no thermal events, SM clock
1320-1425 MHz; `--iters 500 --warmup 50 --rounds 5`, attn.out tp8 `--iters 1500 --warmup
100`, cuBLAS + auto, three independent repeats, `h6t_*`):

| row | rep 1 | rep 2 | rep 3 | cuBLAS us | ours us |
|---|---|---|---|---|---|
| mlp.down tp4 | 1.020 | 1.019 | 1.019 | 8224 / 8236 / 8236 | 8390 / 8392 / 8393 |
| attn.out tp1 | 1.019 | 1.019 | 1.019 | 10488 / 10488 / 10490 | 10685 / 10685 / 10685 |
| attn.out tp8 | 1.041 | 1.026 | 1.039 | 1410 / 1430 / 1412 | 1467 / 1467 / 1467 |

Our kernel's time repeats to 0.1% on all three rows.  At this length mlp.down tp4 and
attn.out tp1 sit at the 1.02 bar (1.019-1.020, spread 0.001), so their 1.025-1.031 in the
200-launch table above was the spread of that recipe.  attn.out tp8 (a 1.4 ms kernel)
stays at 1.026-1.041; there the spread is cuBLAS's own (1410-1430 us).  So, measured as
tightly as this bench allows: 23 of 24 rows at or under 1.02, attn.out tp8 at about 1.035.

**Where the extra instructions go** (ncu source page, instructions executed per SASS line
summed by opcode, mlp.down tp4 M = 88320, `h7o_*`):

| opcode | ours (M warp instr.) | share | cuBLAS (M) | share |
|---|---|---|---|---|
| total | 1659 | | 963 | |
| PRMT | 265 | 16.0% | | |
| LOP3 | 202 | 12.2% | 11 | 1.1% |
| LDS | 177 | 10.6% | 9 | 0.9% |
| HMUL2 | 177 | 10.6% | | |
| IMAD | 174 | 10.5% | | |
| WARPGROUP | 88 | 5.3% | 57 | 5.9% |
| UIADD3 | 79 | 4.7% | 463 | 48.1% |
| HGMMA | 44 | 2.7% | 177 | 18.3% |
| SYNCS | 35 | 2.1% | 29 | 3.1% |
| BRA | 35 | 2.1% | 32 | 3.3% |

(cuBLAS issues 4x the HGMMA instructions because its atom is 64x64x16 where ours is
64x256x16; same flops.)  Barrier polling (SYNCS + BRA) is 4% of our instructions and 6%
of cuBLAS's, so the sleeping-poll fix that helped on Blackwell has nothing to work on here.
The extra ~700 M instructions are the conversion path.  In the SASS of the 128x256 k-loop
that is 14 instructions per 4 weights, not the 6 of the C++ converter: 4 LDS.U16 + 2 PRMT
(CUTLASS's mixed-input smem-to-register copy delivers the 8-bit fragment as 16-bit loads
and packs them), then 2 PRMT (sign replicate), 2 shifts, 2 LOP3 masks, 2 HMUL2.  A byte
permute cannot place the e4m3 exponent into the bf16 exponent field (that needs a 4-bit
shift), so the reachable cut in the converter itself is about 2 of those 14.

**What fewer instructions would buy: the slope, measured.**  The converter's 8 ALU ops were
issued a second time on a dummy value, kept alive by an XOR into the result under a mask
that is zero at run time (a `__device__` variable, so ptxas cannot fold it; numerics
unchanged; build-flag-only, not shipped).  That adds about 24% instructions and shows what
an instruction costs under the power cap.  ncu free-running clock plus the 500-launch
recipe (2 repeats):

| row | instructions (ours / cuBLAS) | SM clock (ours / cuBLAS) | cycles (ours / cuBLAS) | auto / cuBLAS | our time |
|---|---|---|---|---|---|
| mlp.down tp4, as shipped | 1632 M (1.73x) | 0.971 | 0.996 | 1.044 / 1.012 | 8330 / 8337 us |
| mlp.down tp4, + probe | 2019 M (2.14x) | 0.960 | 0.995 | 1.050 / 1.035 | 8426 / 8499 us (+1.6%) |
| attn.out tp1, as shipped | 2072 M (1.73x) | 0.975 | 0.996 | 1.019 / 1.019 | 10682 / 10681 us |
| attn.out tp1, + probe | 2567 M (2.14x) | 0.964 | 0.998 | 1.031 / 1.031 | 10815 / 10815 us (+1.2%) |

+24% instructions costs about 1.1% of SM clock and 1.2-1.6% of time, at unchanged cycle
counts.  Linearised, removing all ~700 M extra instructions (ours at 1.0x cuBLAS) would
buy about 3%; the reachable 2-of-14 cut about 0.3%.  So no converter change reaches the
remaining row: it would need each converted weight tile reused across more MMAs than
wgmma's N = 256 allows.

Results: `h6s_*`, `h6s2_*` (short-burst sweeps, superseded), `h6L_*` (24 rows, long recipe),
`h6o_*` (variants on the offender rows), `h6n_*` (locked vs free-running ncu), `h6f16_*` (the
fp16 probe), `h6t_*` (the tightest recipe), `h7o_*` (opcode counts), `h7p_*` (the slope probe), under `/home/scratch.ishovkun_gpu/w8a16-results/hopper/`.

### Version H7: small M, step 2a, the hand kernel's split-K routed by wave count (Thu 15 Oct 2026)

Small-M goal: every one of the 24 rows at M in {1, 16, 32, 64, 128, 256, 512, 1024, 2048,
4096} at or under the cast-copy path, and within 1.02 of bare cuBLAS where possible.
Protocol for every number here: 300 launches, 30 warm-up, 5 rotated rounds, two full passes,
mean of the per-pass medians; ipp2-0073.  Pass-to-pass agreement of the ratios: median
0.3%, p90 1.8%.

**Baseline (f8eb255ed, `smallM_all_*`).**  Our kernel was already at or under cast-copy on all
240 cells (cast-copy is 3.8-8.2x cuBLAS at M <= 256, 1.3-1.5x at 4096).  Against bare
cuBLAS only 29 of 240 cells were within 1.02.  The hand kernel was the fastest of {auto,
hand, CUTLASS 128x128, CUTLASS 64x256} on 137 of 240 cells and beat auto by more than 5% on
133, but the dispatcher sent only the three und.kv N <= 512 rows to it.  The reason: at
small M the 128x256 grid is a fraction of a wave (mlp.down tp1 at M = 1 is 40 tiles on 132
SMs) and the CUTLASS tiles leave most SMs idle, while the hand kernel's cost model splits K
across up to 8 blocks per tile and reduces in fp32 (mlp.down tp1 M = 1: 64x256 229 us,
hand 82 us, cuBLAS 104 us).  On the three und.kv rows the hand kernel already runs, the
best CUTLASS tile is 81-141% slower at M <= 512, so CUTLASS cannot retire it as the menu
stands.

**Rule** (`W8A16Sm90PickAlgo`, M <= 4096; waves = 128x256 tiles / SMs): hand if the grid is
at most one wave at M <= 512, except short K (<= 3200) at M = 129-256 where there are too
few K slabs to split and 64x256 wins; or if it is 1.4-1.6 waves (mlp.gate_up), where the
split evens the half-empty second wave.  Between 1 and 1.4 waves CUTLASS keeps the row
(gen.qkv tp1 M = 512: hand 1.28x).  On the baseline data: 9 cells more than 2% behind the
better of auto / hand, worst 1.07 (gen.qkv tp8 M = 2048).

**Result** (`s2a_*`): 108 of 240 cells within 1.02 of cuBLAS (was 29), 240 of 240 at or under
cast-copy.  Geomean auto / cuBLAS per M:

| M | before | after |
|---|---|---|
| 1 | 2.076 | 1.056 |
| 16 | 2.061 | 1.040 |
| 32 | 2.012 | 1.019 |
| 64 | 1.967 | 1.004 |
| 128 | 1.808 | 0.984 |
| 256 | 1.498 | 1.077 |
| 512 | 1.326 | 1.198 |
| 1024 | 1.144 | 1.138 |
| 2048 | 1.086 | 1.081 |
| 4096 | 1.025 | 1.029 |

Three cells measured more than 3% slower than before (gen.qkv tp1 M = 256 and 4096, und.q
tp2 M = 4096), all on rows whose pick did not change (128x256 before and after; 128x256
pinned in the same rounds 0.99-1.02): measurement slot.  M = 88320 is untouched by the rule
and re-measured at the H6 numbers.  Worst cell after: und.kv tp8 M = 1 at 1.73x cuBLAS, on
the hand kernel; the 30 worst cells are all narrow N (N <= 1280) at M <= 512, where both
kernels are 1.2-1.7x cuBLAS: next step.  Per-cell table: `hopper-mail/step2a_routing.md`.

**Step 2b: the 64x128 ping-pong tile** (`cutlass_64x128`, algo code 9).  Measured against
the 2a default on 216 cells (M <= 2048, `s2b_*`): it beats it by more than 3% on 16 cells, by up
to 34% (attn.out tp4 M = 128: 1.44 -> 0.94 of cuBLAS; und.kv tp1 / und.q tp8 M = 1024: 1.33 ->
1.06).  All 16 are cells where its grid (64-row weight tiles x 128-row activation tiles) is
80-128 tiles, 0.6-1.0 of a wave: one tile per SM and no split-K round trip.  Above one wave it
is 1.3-1.9x of the 128-row tiles (the register-sourced conversion runs once per activation
tile), and at M < 128 with long K split-K wins.

The dispatcher was refitted against the new menu (not stacked on 2a): at M <= 4096, 64x128 for
a 0.6-1.0 wave grid when M >= 128 and K <= 5120 or K <= 1024; hand split-K for at most one
128x256 wave at M <= 512 (except short K at M = 129-256), and up to two waves at M <= 128; at
1-2 waves the tile by last-wave fill (K <= 1024 -> 64x256; 128x256 last wave >= 0.8 full ->
128x256; 64x256 last wave >= 0.6 -> 64x256; <= 1.3 waves -> stream-K at K >= 4096 or 64x256;
else 128x256).  The 2a hand window at 1.4-1.6 waves was a fit artifact: every 1-2 wave cell of
the 24 rows at M = 256..4096 (51 cells, all variants pinned, `s2c_*`) has the hand kernel within
1% of 128x256 there, because its split model picks no split.  Fit check over the 267 measured
cells: none more than 3% behind the per-cell best.

Result (`s2d_*`, `s2e_*`): 112 of 240 cells within 1.02 of cuBLAS (2a 108, baseline 29), 240 of
240 at or under cast-copy, auto within 2% of the hand kernel on 232 of 240.  Geomean auto /
cuBLAS at M = 1 / 16 / 32 / 64 / 128 / 256 / 512 / 1024 / 2048 / 4096: 1.060 / 1.048 / 1.032 /
1.015 / 0.956 / 1.063 / 1.189 / 1.120 / 1.069 / 1.024.  M = 88320 untouched (max 1.033,
attn.out tp8).  Worst cell: und.kv tp8 M = 1 at 1.72x cuBLAS, on the hand kernel.  Per-cell
table: `hopper-mail/step2b_64x128.md`.

**Step 3.1: split-K on the CUTLASS path** (`cutlass_128x128_splitk`, `cutlass_128x256_splitk`; not
yet in `auto`).  Same structure as the Blackwell `2sm256x128_splitk`: the GEMM runs as one launch
of a batch of `splits` GEMMs over equal K-slices (A / B batch stride = the slice length, fp32
partials batch stride M x N in the workspace), the epilogue stores plain fp32, and our flat reduce
applies scale and bias once.  The grouped tile scheduler now decodes the batch index.  Split count
from the hand kernel's cost model, ported, restricted to equal slices of at least 8 k-tiles.

Against the hand kernel (all splits in {2, 4, 5, 8} pinned, both tiles, plus the plain tiles;
`s31b_*`): the CUTLASS best is within 1.02 of hand on 115 of 144 cells at M <= 512, geomean 0.936.
It beats hand by 5-15% on every narrow-N row (same 128x128 tile and stage count, none of the hand
kernel's split-tile staging).  It still loses 5-22% on long K at M <= 256 (mlp.down tp1 / tp2): the
hand model picks 3 splits there and 400 / 200 k-tiles do not divide by 3, so the equal-slice batch
cannot run it; the next step is unequal slices or a (tile, k-range) work-unit scheduler.
Details: `hopper-mail/step3_1_splitk.md`.

**Scope change (Thu 15 Oct 2026):** Cosmos shapes at M in {1024, 2048, 4096, 8192, 44160, 88320}
only; everything below M = 1024 is out of scope, and the bench recipe is 50 launches x 3 rotated
rounds, one pass, all columns.  The narrow-activation tiles of the first step-3.2 plan (128x16 /
32 / 64, 64x16 / 32, a 128-deep k-slab and a weight-multicast pair, all split-K) were instanced and
matched the hand kernel's numerics on five shapes, but are not landed: they target M <= 64.  The
probe patch is kept in `h4-parked/step3_2_narrow_tiles_probe.patch`.

**Step 3.2: one cost model for auto at M <= 8192** (`W8A16Sm90PickByCost`, replacing the twelve
threshold branches; the hand kernel is no longer an `auto` candidate).  Each candidate (5 plain
tiles, 2 stream-K, 2 split-K x splits) gets a time from: c_tile x K per tile (measured ns per K of
one tile alone on an SM: 128x256 11.37, 128x192 9.00, 128x128 7.45, 64x256 6.15, 64x128 4.50) plus a
2 us x A/256 epilogue for the cooperative tiles; plain tiles in ceil(tiles / SMs) waves; stream-K
with CUTLASS's own decomposition (data-parallel unless the last wave is under half full, then the
last up-to-two waves in equal k-iteration units over min(SMs, k-tiles / 8) CTAs) at 1.1x per
iteration + 8 us + 2 us per peer; split-K s in ceil(tiles x s / SMs) waves of a K/s tile + 2 us +
2e-6 us per fp32 partial element, partials capped at 64 MB.  The split-K variants take their split
count from the same model.  Fit (two runs, 96 cells, all 9 variants pinned, `s32m_*` / `s32m2_*`):
the pick is within noise of the per-cell best variant on 94 of 96.

Result (`s32c_*`): auto within 1.02 of cuBLAS on 41 of 96 cells (was 31); at or under cast-copy on
96 of 96.  The 1.02 bar is not reachable with this menu: a perfect per-cell pick gets 36-42 of 96.
The remaining classes are narrow N under a wave (und.kv / und.q tp8 / gen.qkv tp8: 1.07-1.24 with
the best variant; cuBLAS uses a 64 x 16-64 tile with clusters there), the 1-2 wave band at M = 1024
(1.08-1.19), and the big rows at 2-6 waves (1.05-1.15).  At this recipe the same pinned variant on a
300+ us kernel moves 5-15% between runs, so single-cell differences of that size are not read as
regressions.  M >= 44160 runs the H6 rules unchanged.  Table: `hopper-mail/step3_2_costmodel.md`.

**Step 3.4: the hand-written kernel is retired.**  `auto` stopped using it in step 3.2, and the best
CUTLASS variant was within 2% of it on 70 of the 72 Cosmos3 cells at M = 1024-4096 (the two misses,
gen.qkv tp2 and mlp.down tp4 at M = 4096, 1.035 and 1.048, at the bench's run-to-run spread).  Deleted:
`include/flashinfer/gemm/w8a16_fp8_gemm.cuh` (the version-37 wgmma kernel, its tile / split-K cost
model and split-K reduce), the `prepare_w8a16_fp8_weight` / `weight_prepared` path, the `hand` algo
(code 1 reserved, never reused), and the `K % 16` / `N % 8` fallback that routed misaligned shapes to
the hand kernel (which then failed with a bare `invalid argument`).  Misaligned K / N are now rejected
by the Python op with a ValueError before launch.  The rung history above (rungs 0-38, H1-H6) still
describes that kernel; it is the record of how the CUTLASS menu got here.

**Rename after the hand kernel was retired.**  With one code path left, the header is now
`include/flashinfer/gemm/w8a16_fp8_gemm_sm90.cuh` (matching `w8a16_fp8_gemm_sm100.cuh`) and the
`cutlass_` prefix is gone from the Hopper `algo` names (`128x256`, `128x192`, `128x160`, `128x128`,
`64x256`, `64x128`, `128x256_streamk`, `128x192_streamk`, `128x128_splitk`, `128x256_splitk`), the
`W8A16Sm90Algo` enum (`k128x256`, ...) and the launcher functions (`W8A16Fp8GemmSm90`,
`W8A16Fp8GemmSm90WorkspaceBytes`).  Numeric codes unchanged (1 still reserved).  The sections above
keep the names as they were when measured (`cutlass_128x256` = `128x256`).

**Measurement recipe (Thu 15 Oct 2026).**  A 50-launch burst never reaches the 700 W power cap: the
whole process runs at boost clock (1965 MHz, 225-360 W), and the capped steady state (1410-1450 MHz
at 693 W on a 1.4 ms kernel) starts ~0.8 s into a burst on the H100 and ~6.5 s on the H200.  So
50 x 3 measures a different operating point, and on long kernels the ratio flips (mlp.down tp2
M = 8192, 128x256: 1.022-1.040 at 50 x 3, 0.983-0.986 at 500 x 3, three repeats each).  The bench
got `--warmup-seconds` (an untimed burst once per process) and `--burst-seconds` (each timed burst
sized by time: iters = clamp(ceil(s / call), 50, 5000)); proposed recipe 10 s / 2 s / 3 rounds,
pending Igor's decision.  Details: `hopper-mail/recipe_experiment.md`.

**Step 3.2b: 64x64 / 64x32 ping-pong split-K tiles, not landed.**  Measured on the narrow-N rows
(und.kv tp1-8, und.q tp8, gen.qkv tp8) at M = 1024-4096 with the steady-state recipe, auto and
maxed stage counts, a 128-deep k-slab and a cluster-2 weight-multicast pair.  They win on two
cells (und.kv tp8, N = 128, at M = 1024 / 2048: about 11% each, 11-15 us kernels) and are 1.3-4.6x
slower everywhere else.  The register-sourced conversion runs once per activation tile, so a
64- or 32-row activation tile doubles or quadruples the conversion work per flop.  Stage count
and multicast change nothing measurable.  Patch kept in `h4-parked/step3_2b_64x64_64x32_probe.patch`;
table in `hopper-mail/step3_2b_narrow_tiles.md`.

## Why not the Hopper design

tcgen05.mma takes both operands from shared memory (or A from TMEM), so the fp8 -> bf16
conversion cannot ride in registers in front of the MMA.  CUTLASS's sm100 mixed-input
kernel converts in a separate transform warpgroup (TMA -> smem -> registers -> TMEM /
smem -> MMA); rung B0 wires it up as-is, rung B2 forks it into
`include/flashinfer/gemm/w8a16_sm100/` (kernel, collective, builder; keyed on W8A16 dispatch
tags, listed in `.clang-format-ignore` to stay diffable against CUTLASS 4.4) so the warp
layout, tile shapes and TMEM budget can change.  Mapping: the fp8 weight `[N, K]` is the
builder's A (M' = N), the activation `[M, K]` is B (N' = M), D `[M', N']` column-major is
`out[M, N]` row-major; the per-tensor scale is the epilogue alpha (device pointer), the bias
a per-row broadcast in the fused epilogue.

## Final state (version B14)

Geometric mean of per-row time ratios over the 24 rows of each M class, B200
(umb-b200-236, 148 SMs), free-running clocks, CUDA 13.6, torch 2.12, Tue 13 Oct 2026
(`sweepB14b`).  Every variant is measured back to back on the same tensors (`bench
--algo auto 2sm256x128 2sm256x192 2sm256x160 2sm256x128_splitk 2sm256x128_streamk`),
one M class per process, two passes, per-row minimum.  (A single pass over the whole
144-row table is 5% median / 20% p90 noisy between repeats on this box: the large-M
rows heat the part and the small rows that follow swing; one M class per pass brings
that to 1-2% median / 10% p90.)  The 1SM 128x128 column is from the B12 table.

Caveat on the fixed-variant columns: within a row the bench times the variants in a
fixed order, `auto` first, and on the power-bound wide rows the part is hotter by the
time the pinned variants run.  mlp.gate_up tp2 (N=12800) at M=8192 is the worst case:
`auto` measures 686 us and the pinned 256x128 tile, the same binary `auto` selected,
968 us.  So on those rows the fixed columns are biased slow and the "oracle" can read
worse than `auto`.  The plots take the per-row minimum over the variants *and* `auto`;
a future sweep should interleave the variants per row (or pin clocks) instead.

| M | auto / cuBLAS | oracle / cuBLAS | 256x128 | 256x160 | 256x192 | 1SM 128x128 | 256x128 split-K | 256x128 stream-K | auto / oracle |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 1.08 | 1.07 | 1.20 | 1.22 | 1.38 | 1.25 | 1.26 | 1.43 | 1.01 |
| 2048 | 1.03 | 1.01 | 1.10 | 1.13 | 1.14 | 1.20 | 1.28 | 1.25 | 1.01 |
| 4096 | 1.00 | 0.98 | 1.03 | 1.05 | 1.07 | 1.12 | 1.27 | 1.08 | 1.02 |
| 8192 | 1.00 | 0.97 | 1.03 | 1.02 | 1.04 | 1.11 | 1.31 | 1.05 | 1.03 |
| 44160 | 1.01 | 0.96 | 1.02 | 0.99 | 1.01 | 1.10 | 1.32 | 1.06 | 1.05 |
| 88320 | 1.03 | 1.00 | 1.04 | 1.03 | 1.05 | 1.15 | 1.37 | 1.11 | 1.03 |

Overall auto / cuBLAS 1.023, oracle (best variant per row) / cuBLAS 0.999, auto / oracle
1.024; 67 of 144 rows have a variant at or under cuBLAS.  Who wins per row: the 256x128
tile 45, 256x160 38, 256x192 36, stream-K 14, split-K 11 (the narrow und.kv rows).
The B14 step: M=1024 1.112 -> 1.076, M=2048 1.038 -> 1.027, M=44160 1.026 -> 1.008
(the 160-row tile takes 10 of 24 rows there), by giving the wave-fill model a third
tile height to land near whole waves.  Rows still > 10% behind cuBLAS: a handful of
M=1024 rows with N=5120 and K <= 3200 (attn.out tp4 / tp8, mlp.down tp8, 1.13-1.15x:
even at 1.89 waves a 14-27 us kernel pays its un-overlapped prologue and epilogue) and
the widest power-bound rows (mlp.gate_up N=25600 at M >= 44160, 1.25-1.30x, clocks at
~1.2 GHz).  The gap between auto and the oracle (2.4%) is mostly the und.q tp2 rows
(N=4096, K=5120) at M >= 4096, which are bimodal on this box across all four passes of
two sweeps: at M=8192 the 128-row tile measures 227 or 316 us, the 192-row tile 209 to
339, cuBLAS 311 to 380, while the stream-K variant sits at 211 to 236 every time.  The
per-row minimum therefore hands the oracle to stream-K; whether the slow mode is a
clock / thermal state or a memory-system pathology of that shape (power-of-two N, 10 KB
activation row stride) that the stream-K tile order dodges is not established.  Not
pursued yet.

### Knobs and the `algo` argument

`flashinfer.gemm.w8a16_fp8_gemm(..., algo=)` (Blackwell) pins a variant per call:
`"2sm256x128"`, `"2sm256x160"`, `"2sm256x192"`, `"2sm256x256"`, `"1sm128x128"`,
`"2sm256x128_splitk"`, `"2sm256x128_streamk"`; `None` / `"auto"` is the launcher's model.  The codes are the
`W8A16Sm100Tile` enum values.  `FLASHINFER_W8A16_FP8_TILE_SM100` still pins a tile
process-wide (same names).  `FLASHINFER_W8A16_FP8_STREAMK_MODE` (`heuristic` |
`streamk` | `splitk` | `dataparallel`) and `FLASHINFER_W8A16_FP8_STREAMK_REDUCTION`
(`deterministic` | `nondeterministic`) pin the stream-K scheduler's decomposition and
peer-reduction order for A/B runs.  The benchmark takes `--algo a b ...` and emits one
`kernel:<algo>` column per variant, measured back to back; that is what the tables
above come from.

### The dispatcher after versions B11 / B12 / B13 / B14 (`W8A16Sm100PickTile`)

0. Stream-K (version B13, 256x128 tile, `W8A16Sm100UseStreamK`): more than half the
   clusters busy, two or three waves with the last one at most a third full, and >= 64
   k-tiles per tile.  On the Cosmos3 table that is the three M=1024 N=5120 rows with
   K >= 8192 (attn.out tp1, mlp.down tp1 / tp2), 7-8% faster than split-K there and now
   at or under cuBLAS on two of them.
1. Split-K (version B12, 256x128 tile, `W8A16Sm100SplitK`): (a) fewer cluster tiles than
   half the resident clusters: the largest split that still fits in one wave; (b) K >=
   100 k-tiles and a partial last wave: the smallest split with >= 50 k-tiles per slice
   that cuts waves x k-tiles-per-slice by >= 15%.  Short-K problems never split (the fp32
   partials cost more than the fill buys).
2. Otherwise the wave-fill cost model `ceil(units / clusters) x height / eff` over three
   tile heights: 256x128 (eff 1.00), 256x160 (eff 0.96, version B14) and 256x192 (eff
   1.02, refit from the B9 value of 1.22 once the native convert took the transform off
   the critical path).  The point of the 160-row tile is not speed per tile but a tile
   count that lands near a whole number of waves: N=5120 at M=1024 is 160 tiles of 128
   rows (2.16 waves, last wave 16% full) or 140 tiles of 160 rows (1.89 waves, 89%
   full), and the taller tile's surplus rows in the last M-tile are masked by the
   TMA / epilogue predication for free; it measures 11-16% faster there.  The 256x256
   smem-A tile lost every row and is `algo`-only; so is the 1SM tile.

Split-K runs the mixed-input kernel as one batched launch over K-slices (batch stride =
slice length for A and B, one fp32 M x N partial per slice for D, alpha = 1), then
`W8A16Sm100SplitKReduceKernel` sums the slices, applies the per-tensor scale and the
bias and casts; the fp32 epilogue uses a 128x16 epilogue tile to stay inside the
227 KB smem budget next to the 6/4 mainloop stages.

Stream-K runs the same kernel under CUTLASS's `PersistentTileSchedulerSm100StreamK`
(a fourth template flag on `W8A16Sm100Gemm`; the forked kernel body already consumed
the per-unit k-range and called the scheduler's fixup).  The full waves stay
data-parallel; the tiles of the last wave(s) are cut along K into equal iteration
counts for every resident cluster, the peers of a tile add their fp32 partials through
the scheduler workspace, and only the final peer runs the scale / bias / cast epilogue,
so no reduce kernel.  What it costs (Tue 13 Oct 2026, M=1024, N=5120, torch profiler):
the per-launch workspace memset is 1.5 us; the kernel itself is 28.7 us against 21.2 us
for the plain tile at K=2048, because every K fragment pays a fresh pipeline prologue
and a 128 KB accumulator round trip through global memory, roughly 8-10 k-tiles' worth
of mainloop.  Deterministic vs nondeterministic peer order makes no difference (within
1%).  Hence the rule: only tiles >= 64 k-tiles deep, and only when the tail is most of
the grid (at four or more waves, M=2048, it loses 5-11%).

### Rung B10 state, for the record


Geometric mean of per-row time ratios over the 24 rows of each M class, B200
(umb-b200-261, 148 SMs), free-running clocks, auto dispatch.  Per row the minimum
over four runs of the table (B10) and three runs with the rung B9 converter forced
back in (`FLASHINFER_EXTRA_CUDAFLAGS=-DW8A16_FORCE_ALU_CONVERT`), all in one
session on one node (Mon 12 Oct 2026, CUDA 13.6, torch 2.12).

| M | B9 / cuBLAS | B10 / cuBLAS | B9 / cast-copy | B10 / cast-copy | B10 / B9 | rows at or under cuBLAS |
|---|---|---|---|---|---|---|
| 1024 | 1.36 | 1.24 | 0.33 | 0.31 | 0.89 | 0 / 24 |
| 2048 | 1.09 | 1.06 | 0.38 | 0.37 | 0.93 | 10 / 24 |
| 4096 | 1.09 | 1.03 | 0.56 | 0.52 | 0.94 | 10 / 24 |
| 8192 | 1.08 | 1.02 | 0.72 | 0.69 | 0.95 | 8 / 24 |
| 44160 | 1.11 | 1.03 | 1.04 | 0.96 | 0.92 | 10 / 24 |
| 88320 | 1.09 | 1.04 | 1.07 | 1.01 | 0.94 | 9 / 24 |

Overall B10 / B9 0.928 (geomean over 144 rows), B10 / cuBLAS 1.065; 47 rows at or
under cuBLAS, 82 within 5%.  Rows still more than 10% behind cuBLAS: all of M=1024
(wave quantisation: one 256-row weight tile per cluster pair), the narrow und.kv rows
(N=128..512, split-K territory), and the widest power-bound rows (mlp.gate_up
N=25600 at M>=44160, 1.28x, clocks at ~1.2 GHz).  68 tests pass, 0 spills
(74-78 registers), SASS carries 224 `F2FP.BF16.E4M3.UNPACK_B` and no HMUL2.

Known follow-up (rung B11): the dispatcher's efficiency factors were fitted with the
rung B1 converter.  With the native convert the 256x256 smem-A tile loses everywhere
measured (it only won while the convert was expensive), and on e.g. 12800x5120 M=1024
and 2048x5120 M=2048 auto picks it (84.6 / 32.8 us) where the 256x128 tile does
79.4 / 31.0 us and beats cuBLAS (79.8 / 29.5 us).

### Rung B9 state, for the record

| M | kernel / cuBLAS | kernel / cast-copy | kernel / rung B0 |
|---|---|---|---|
| 1024 | 1.40 | 0.36 | 1.03 |
| 2048 | 1.13 | 0.40 | 0.91 |
| 4096 | 1.10 | 0.55 | 0.88 |
| 8192 | 1.09 | 0.73 | 0.84 |
| 44160 | 1.11 | 1.04 | 0.82 |
| 88320 | 1.10 | 1.08 | 0.83 |

Pinned-clock ncu, auto dispatch: 12800x5120 M=8192 (192-row tile) tensor pipe 91.6%
active / 89.7% elapsed at 753 us; 25600x5120 M=44160 (192) 92.5% / 92.3%; 5120x5120
M=4096 (192) 87.6% / 83.0%, 126.5 us against cuBLAS 129; 1024x5120 M=2048 (128-row tile)
56.5% / 40.2%.  cuBLAS itself (nvjet 128x256 2-CTA 6-stage) reaches ~92% on the large
shape.  0 spills on every instantiation; 68 tests under auto and every pinned tile;
144/144 rows within the error gate.  Best rows 1.65 PFLOPS.

Ceiling probes (`-DW8A16_PROBE_NO_CONVERT`, transform copies without converting; perf
only): the 256x128 TMEM-A skeleton reaches 98.9% tensor pipe and 553 us (cuBLAS 644), the
256x256 smem-A skeleton 94.3% and 606 us.  The conversion instruction stream is the
whole remaining cost; four instructions per converted pair (prmt, shift, mask, bf16x2
multiply) is the floor found so far.

## Knobs

| Variable | Effect |
|---|---|
| `FLASHINFER_W8A16_FP8_TILE_SM100` | `2sm256x128`, `2sm256x192`, `2sm256x256` or `1sm128x128`: pins the tile instead of the cost model (auto picks among the first three). Read once per process. |

Dispatch (`W8A16Sm100PickTile`): per tile, `ceil(units / (SMs / 2)) * tileA / eff` with
`units = ceil(N / 256) * ceil(M / tileA)` and eff 1.00 / 1.22 / 1.10 for the 128 / 192 /
256-row tiles (fitted offline on the B3 / B4 / B8 tables, geomean 1.008x the per-row
oracle).

## Rungs

| Rung | Commit | Change | Result |
|---|---|---|---|
| B0 | 987ad3481 | CUTLASS 4.4 sm100 mixed-input mainloop, 2SM 256x128x128, swap-and-transpose mapping | baseline: 1.54 / 1.38 / 1.30 / 1.30 / 1.34 / 1.31 x cuBLAS; 69.9% tensor pipe |
| B1 | 64b4a341b | pure-ALU e4m3 -> bf16 / f16 converter (sign-replicating prmt, shift, mask, x2^120) | +2-4% at large M; 73.3%; exact on all 254 finite encodings (new test) |
| B2 | 7205301ec | fork kernel / collective / builder into FlashInfer, new dispatch tags | parity (identical ncu counts) |
| B3 | 162fa988b | two transform warpgroups splitting the k-blocks (compile-time group index) | small win: 75.5% |
| B4 | aea818528 | 2SM 256x256x64 tile, converted A in smem (2SM SS atom added to the forked builder) | MIXED: 78.7% pinned, wins wide rows 7-20%, loses narrow ones up to 2x; env-only |
| B5 | 49fe38868 | four transform warpgroups | LOSER: 65.8% / 65.2%; +41% instructions (pipeline handshakes per warp) |
| B5b | c33bec27c | back to two warpgroups | restore |
| B6 | 6be0c7230 | pipe-balanced converter (mask first, IMAD shift) | LOSER: -4 to -5 points; IMAD latency on the dependent chain |
| B6b | 8367b0aa4 | restore the B1 converter | restore |
| B7 | 82c8b2f34 | dispatch between 256x128 and 256x256 by a wave-fill cost model | WIN: 1.46 / 1.31 / 1.20 / 1.22 / 1.21 / 1.18 |
| B8 | 3c32207a4 | 2SM 256x192x128 tile with A in TMEM (2 acc stages x 192 + 2 A stages x 64 = 512 TMEM cols) | WIN: 91.9% active / 89.1% elapsed on 12800x5120 M=8192; best tile on 20-22 of 24 rows at M >= 8192 |
| B9 | 62885dae2 | three-way dispatch | WIN: 1.40 / 1.13 / 1.10 / 1.09 / 1.11 / 1.10 x cuBLAS |
| B10 | d24ed43a9 | native `cvt.rn.bf16x2.e4m3x2` / `cvt.rn.f16x2.e4m3x2` converter (one F2FP per pair; ALU path kept behind `W8A16_FORCE_ALU_CONVERT` and for nvcc < 13.2) | WIN: 1.24 / 1.06 / 1.03 / 1.02 / 1.03 / 1.04 x cuBLAS, 0.928x B9 time |

| B11 | (this commit) | `algo` argument on the Python op (codes 1-5, FFI `int64 algo`), `bench --algo a b ...` back-to-back columns, dispatcher refit: 256x192 eff 1.22 -> 1.02, 256x256 dropped from auto | WIN: auto / oracle 1.040 -> 1.014 |
| B12 | fec33517e | batched split-K for the 256x128 tile (fp32 partials + reduce kernel, 128x16 fp32 epilogue tile), split rule for under-filled grids and long-K partial waves | WIN: M=1024 1.24 -> 1.10 x cuBLAS, M=2048 1.06 -> 1.04, und.kv rows 1.3-1.8x -> ~1.0x |
| B14 | (this commit) | 256x160 tile (`2sm256x160`, 2 x 160 acc + 3 x 64 A TMEM columns) as a third tile height in the wave-fill model (eff 0.96) | WIN: M=1024 1.112 -> 1.076 x cuBLAS, M=2048 1.038 -> 1.027, M=44160 1.026 -> 1.008; N=5120 M=1024 rows 11-16% faster; ncu (line info, `ncu_reader`) first showed the small-M loss is wave fill + un-overlapped prologue/epilogue, every stall on a pipeline mbarrier, 54% tensor pipe vs 92% at M=8192 |
| B13 | cf6fb8984 + 4994d9979 | 256x128 tile under CUTLASS's SM100 stream-K scheduler (`2sm256x128_streamk`), dispatcher rule for a mostly-empty tail wave over >= 64-k-tile tiles | SMALL WIN: 3 rows (M=1024 N=5120 K >= 8192) 7-8% faster, two now under cuBLAS; M=1024 geomean 1.114 -> 1.103; everywhere else stream-K loses (1.43x at M=1024 overall when forced: per-fragment prologue + 128 KB accumulator round trip) |

### Why B10 works (PerfSim, gb100 model, Mon 12 Oct 2026)

A PerfSim + Morph run of the rung B9 kernel (one launch, 2SM 256x192x128 tile,
M=192 N=256 K=4096, 1.8 GHz model, `/home/scratch.ishovkun_gpu/w8a16_perfsim/`,
PIC report `pic_m192d/report`) gives the per-warp timeline the silicon profile
cannot: per 128-deep k-block the MMA takes 768 cycles, the measured period is 817
(MMA pipe 93.9% in the mainloop, matching ncu's 92%).  The MMA warp issues its eight
`UTCHMMA.2CTA` in 33 cycles and then waits on a long dependency 74% of the mainloop:
each of the 16 transform warps needs ~200 instructions and ~750 cycles per k-block
(the prmt / shift / mask / multiply chain at ~4 cycles per instruction, two transform
warps per sub-partition contending), and the last TMEM store -> mbarrier -> 2-CTA
barrier -> MMA issue handoff costs a fixed ~226 cycles.  The convert chain therefore
runs ~50 cycles longer than the MMA every k-block; the issue slots themselves are
only 45% busy, so it is chain length, not issue bandwidth.  The native convert cuts
the chain to ~110 instructions per warp per k-block, which is what B10 measures.  The
same conclusion came independently out of the Kernel Factory campaign on this kernel
(every agent's first win was the native cvt, 7.6-12.8% on single shapes).

## Why the losers lost

- **B5**: every transform warp pays the per-k-tile pipeline handshakes (try_wait, acquire,
  named barrier, commit) and steals issue slots from the MMA and TMA warps; the convert is
  instruction-count bound, not latency bound.  (First build dispatched only two of the
  four groups, leaving half of every k-tile as zeros; an identity-activation probe found it.)
- **B6**: same four instructions per pair, but the IMAD on the dependent prmt -> shift ->
  mask -> multiply chain costs more latency than the SHF it replaced, and two transform
  warps per sub-partition cannot hide it.
- **B4 on narrow problems**: a 256x256 cluster tile leaves 32 clusters on 148 SMs for
  N=1024, M=2048; the dispatcher (B7/B9) keeps the 128-row tile there.
- **B13 stream-K on short K**: the idea (split only the tail wave's tiles along K across
  the idle clusters) is right, and the arithmetic says 160 tiles x 40 k-tiles over 74
  clusters should drop from 3 waves to 2.16.  But each fragment re-fills the TMA /
  transform pipeline and dumps and re-reads a 128 KB fp32 accumulator, which costs
  about as much as 8-10 k-tiles of mainloop; with 8-40 k-tiles per tile that is more
  than the 0.84 wave saved.  It pays from ~64 k-tiles per tile upwards, which is where
  the rule draws the line.  The und.kv rows (8-32 tiles) stay with batched split-K,
  whose slices need no handshake (11.9 us vs 34 us for stream-K at N=128).

## Results directory

`/home/scratch.ishovkun_gpu/w8a16-results/blackwell/`: `rungB<N>_*_b200_full.csv`,
`rungB<N>_ncu_*.txt`, `rungB5_sass.txt` / `rungB6_sass.txt` (cuobjdump of the module),
`sweepB12d_b200_r<M>_<i>.csv` (the per-M back-to-back variant sweeps behind the B12
table; `sweepB11i_*`, `sweepB12*_*` are the noisier whole-table sweeps),
`sweepB13_b200_r<M>_<i>.csv` (all M, stream-K forced as a column) and
`sweepB13b_b200_r{1024,2048}_<i>.csv` (the B13 dispatcher remeasure),
`sweepB14_b200_r{1024,2048}_<i>.csv` (128 / 160 / 192 tiles) and
`sweepB14b_b200_r<M>_<i>.csv` (the version B14 final table, all variants, all M),
`ncuB13_<tile>_m<M>.ncu-repz` (full ncu with line info, N=5120 K=4096).  Plots:
`plots/w8a16_b200_plots.py` in the main checkout renders the ladder figures from these
CSVs.
