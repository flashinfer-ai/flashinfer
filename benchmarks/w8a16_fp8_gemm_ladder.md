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
`weight_prepared=True` (ldmatrix fragment loads) is opt-in and not faster on H100/H200.
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

### Version B15: tile-order swizzle for the weights that overflow L2 (Wed 14 Oct 2026)

Igor's reading of the B14 plot: "mlp.gate_up tp1 is far from the peak at large M", and
unlike the other large-M rows this one is not noise.  Both B14 passes agree: at M=44160
/ 88320 every variant sits at 930-1150 TFLOP/s against cuBLAS's 1380-1390.  N=25600,
K=5120: a 131 MB fp8 weight, just over B200's L2.  The Blackwell launcher never set
`max_swizzle_size` (CUTLASS default 0 = plain raster) or `raster_order` (heuristic), so
the 74 in-flight clusters walked the 100 weight tiles of one activation tile, then the
next activation tile re-read the whole weight: 345 x 131 MB = 45 GB of L2 misses at
M=44160, nearly the kernel's whole compute time in HBM traffic.  tp2 (65 MB) mostly fits,
hence its 4% loss.  Same disease the Hopper line fixed in H3 (there it was the raster
direction); it had never been applied here.

New knobs `FLASHINFER_W8A16_FP8_SWIZZLE_SM100` / `_RASTER_SM100` (both sm100 schedulers
take them), then swizzle 0 / 2 / 4 / 8 / 16 x raster heuristic / m / n on umb-b200-260,
free-running, CUPTI, L2 flushed (`swzB_*`).  Kernel / cuBLAS, heuristic raster:

| swizzle | gate_up tp1 M=44160, 256x128 | same, 256x192 | M=88320, 256x128 | 256x192 | und.q tp2 M=44160 (control), 256x128 |
|---|---|---|---|---|---|
| 0 (B14) | 1.52 | 1.20 | 1.43 | 1.18 | 0.98 |
| 2 | 1.32 | 1.29 | 1.23 | 1.13 | 0.99 |
| 4 | 1.18 | 1.11 | 1.16 | 1.05 | 1.00 |
| 8 | 1.07 | 0.99 | 1.04 | 1.01 | 1.00 |
| 16 | 1.05 | 0.99 | 1.01 | 0.99 | 1.00 |

Raster `m` equals the heuristic (which resolves to `m` here); `n` without a swizzle is
1.35-1.55x on every row.  gen.qkv tp1 (N=10240, 52 MB) wobbles +-10% in both
directions with cuBLAS's own reference varying 5% between runs: the power cap.

Swizzle 8 as a blanket default, full 24 x 6 grid, `auto`, swizzle 0 vs 8 back to back
(`b15_s{0,8}_r*`): geomeans 1.05 / 1.08 at M=1024, 1.02 / 1.03, 0.99 / 1.00, 0.99 / 1.01,
1.03 / 1.03, 1.04 / 1.04.  gate_up tp1 1.32 -> 0.98 / 0.99, mlp.down tp2 8-16% faster at
M >= 44160, but the M=1024 rows lost for real (both of two repeats, `b15small_*`):
mlp.down tp1 172 -> 209 us, mlp.down tp2 85 -> 101 us, attn.out tp1 57 -> 66 us.  At
that size the whole problem is L2-resident and the swizzle only scrambles the wave
packing of a two-wave grid.  Hence the rule in `W8A16Sm100PickSwizzle`: swizzle 8 when
the fp8 weight is >= 64 MB (half of L2) and M >= 4096, else 0; the env pins a value for
every shape.  The sm100 scheduler does not pad the grid to the swizzle size (residual
tiles are not remapped), so the H3 padding guard is not needed here.

Final B15 state (`sweepB15_b200_r*`, umb-b200-260, one pass per M class, `--algo auto
2sm256x128 2sm256x192 2sm256x128_streamk` plus the torch cast-copy op, 110 tests pass),
geomean kernel / cuBLAS per M class:

| M | auto (B15) | auto (B14) | 256x128 | 256x192 | 256x128 stream-K | torch cast-copy | oracle of the four | mean cuBLAS TFLOP/s |
|---|---|---|---|---|---|---|---|---|
| 1024 | 1.05 | 1.08 | 1.17 | 1.35 | 1.43 | 3.79 | 1.05 | 1198 |
| 2048 | 1.02 | 1.03 | 1.10 | 1.14 | 1.28 | 2.75 | 1.01 | 1415 |
| 4096 | 0.99 | 1.00 | 1.02 | 1.07 | 1.11 | 1.94 | 0.99 | 1477 |
| 8192 | 0.98 | 1.00 | 1.00 | 1.03 | 1.06 | 1.54 | 0.97 | 1496 |
| 44160 | 0.98 | 1.01 | 1.00 | 1.00 | 1.04 | 1.10 | 0.96 | 1458 |
| 88320 | 1.00 | 1.03 | 1.01 | 1.02 | 1.09 | 1.03 | 0.97 | 1441 |

mlp.gate_up tp1 in this pass: 1.06 at M=44160, 1.04 at 88320 (0.98 / 0.99 in the
swizzle-0-vs-8 grid an hour earlier; that spread is the box).  The B14 remark "widest
power-bound rows 1.25-1.30x, clocks at 1.2 GHz" was wrong about the cause: the clock
sits there for cuBLAS too, the gap was the L2 re-stream.  Plot:
`plots/w8a16_b200_tflops_grid.png` is drawn from this sweep.

What is left on the two widest rows (umb-b200-240, each implementation alone for 400
launches with `nvidia-smi dmon` sampling, `clockB_*`): mlp.gate_up tp1 M=88320 cuBLAS
17045 us at 1002 W / 1130 MHz, auto 17374 us (1.019) at 910 W / 1095-1231 MHz, pinned
256x192 17285 us (1.014) at 987 W / 1132 MHz; tp2 cuBLAS 8528 us at 1002 W / 1127 MHz,
auto 8907 us (1.045) at 934 W / 1080 MHz, 256x192 8931 us at 989 W / 1117 MHz.  Same
clock and slightly less power than cuBLAS, so the remaining 2-5% is not the dequant's
energy; it is whatever cuBLAS's tile order still saves in L2 traffic.  Swizzle 16 / 32
on tp2 (`swzC_*`) is indistinguishable from 8 inside the +-10% single-pass scatter of
this box (cuBLAS itself read 6912 to 8212 us on tp1 M=44160 across three passes minutes
apart).

TMA L2 eviction hints (`FLASHINFER_W8A16_FP8_L2HINT_SM100`; the forked collective now
carries `cache_hint_a` / `cache_hint_b` into the A and B TMA loads; `l2hint_p*`): three
interleaved passes, median kernel / cuBLAS on gate_up tp1 M=44160 / 88320: normal 1.05 /
1.00, weight evict-last 1.04 / 1.01, activation evict-first 1.31 / 1.29, both 1.34 / 1.28;
tp2: 1.16 / 1.00, 1.17 / 1.11, 1.14 / 1.13, 1.19 / 1.08; the und.q tp2 control is 0.96 under
every setting.  Evict-last on the weight buys nothing measurable; evict-first on the
activation costs 30% on tp1 because, under the swizzle, the activation tile is exactly the
operand the eight in-flight weight tiles share.  LOSER; the knob stays at `normal`.  What
is left for those two rows is 256 B L2 promotion on the weight descriptor, untried.

Cluster (2, 2, 1) with the weight tile multicast to two 2SM pairs on neighbouring
activation tiles (`2sm256x128_c2` / `2sm256x192_c2`, algo codes 8 / 9, never chosen by
`auto`; `c2_p*`): two passes of five back-to-back columns on umb-b200-240, kernel / cuBLAS.
mlp.down tp1 M=44160: plain 128 1.16-1.21, c2 1.26; plain 192 1.11-1.13, c2 1.24-1.25.
mlp.gate_up tp2 M=44160: 128 1.11-1.13 vs c2 1.06-1.13; 192 0.96-1.09 vs c2 1.08-1.16.
und.q tp2 (control): 128 0.90-1.06 vs c2 1.05-1.31.  LOSER everywhere: the weight's
L2 -> SMEM traffic was not the bottleneck, and a 4-CTA cluster packs worse.  Kept as
algo-only variants for A/B runs.

### Version B16: the clock follows the DRAM bytes; swizzle scaled by 1/K (Wed 14 Oct 2026)

Why mlp.down tp1 (N=5120, K=25600) and mlp.gate_up tp2 were still 1.10-1.15 of cuBLAS
at M >= 44160 with every tile.  ncu `--set full`, locked clocks, mlp.down tp1 M=88320,
our auto (256x192) vs cuBLAS (nvjet 256x256, k-tile 64):

| | ours | cuBLAS |
|---|---|---|
| tensor pipe active | 97.1% | 96.4% |
| SM cycles elapsed | 19.21 M | 19.56 M |
| SM clock during the kernel | 1.188 GHz | 1.333 GHz |
| time | 16.17 ms | 14.68 ms |
| DRAM read | 40 GB | 20.7 GB |
| L2 bytes | 164 GB | 125 GB |
| SMEM wavefronts | 597 M | 9.5 M |

In cycles we are 2% faster than cuBLAS and both tensor pipes are saturated; the whole gap
is the SM clock, 11% lower for our kernel even under ncu's lock, because the chip's
power / current limiter follows the memory traffic.  Which traffic: sweeping the swizzle
on this row (`b16_swz`) leaves the cycle count at 19.2 M in every setting while the clock
moves with the DRAM bytes one to one (shared-memory traffic does not matter: the 256x256
tile has 2x the SMEM wavefronts and the highest clock):

| mlp.down tp1 M=88320, 256x192 | swizzle 2 | 4 | 8 (B15) | 16 | 20 | raster n, 0 |
|---|---|---|---|---|---|---|
| DRAM read | 36 GB | 26 GB | 39 GB | 69 GB | 77 GB | 93 GB |
| SM clock | 1.22 | 1.27 | 1.20 | 1.08 | 1.06 | 1.01 GHz |
| time | 15.7 | 15.1 | 16.0 | 17.9 | 18.3 | 19.1 ms |

| mlp.gate_up tp2 M=44160, 256x128 | 4 | 8 (B15) | 16 | 32 |
|---|---|---|---|---|
| DRAM read | 4.1 | 3.2 | 1.9 | 3.7 GB |
| time | 3.82 | 3.75 | 3.69 | 3.88 ms |

The B15 constant 8 was fitted on K=5120.  The band of S weight tiles (S x 256 x K bytes)
wants to stay around 10-26 MB, so S scales with 1/K: `W8A16Sm100PickSwizzle` now returns
24 MB / (256 K) rounded down to a power of two in [2, 16] (16 at K=5120, 8 at 12800, 4
at 25600) for weights >= 64 MB at M >= 4096.  Checks on the interpolated rows (`b16_swz2`):
mlp.down tp2 (K=12800) M=88320: 4 / 8 / 16 -> 9.9 / 8.5 / 19.5 GB; mlp.gate_up tp1
(K=5120) M=88320: 8 / 16 -> 9.0 / 6.8 GB, 14.17 / 14.04 ms; mlp.down tp1 M=44160: 2 / 4 /
8 -> 17.6 / 12.7 / 15.7 GB; mlp.down tp1 M=8192: 0 / 4 -> 4.2 / 2.9 GB, 1.45 / 1.38 ms.
Not pursued: raster `n` is 1.3-2x worse without a group and never better with one; the
256x256 tile reaches cuBLAS's clock (1.315 GHz, 26-29 GB) but needs 8% more cycles for its
SMEM-staged A operand.

Confirm (umbriel-b200-017, free-running, three passes alternating the rule and the B15
constant 8 per pass, median kernel / cuBLAS; `b16c_*`; 122 tests pass).  The K=12800
rows get swizzle 8 under both settings, so they calibrate the pass-to-pass noise (+-3%):

| row | B15 (8) | B16 (rule) |
|---|---|---|
| mlp.down tp1 (K=25600) M=44160 | 1.155 | 0.995 |
| mlp.down tp1 M=88320 | 1.051 | 1.025 |
| mlp.gate_up tp1 (K=5120) M=44160 | 1.012 | 0.978 |
| mlp.gate_up tp1 M=88320 | 1.017 | 0.998 |
| mlp.gate_up tp2 M=44160 | 1.098 | 1.059 |
| mlp.gate_up tp2 M=88320 | 1.049 | 1.052 |
| mlp.down tp2 (K=12800, same code) M=44160 / 88320 | 1.108 / 0.977 | 1.130 / 0.970 |
| M=4096 / 8192 rows | 0.98-1.04 | 0.85-1.05 |

mlp.gate_up tp2 at 1.05-1.06 is what is left on Blackwell at large M (2.4 vs our 1.9 GB
of DRAM is not it; the remaining clock difference there is unexplained).

### Version B17: every row at M=88320, and how to measure one (Wed 14 Oct 2026)

Igor's goal: every one of the 24 shapes at M=88320 on par with cuBLAS or better, no
averages.  Bar: median kernel / cuBLAS <= 1.02, two independent runs agreeing.

**Measurement first.**  Three things that looked like kernel gaps were the bench:
1. Fixed column order.  The bench timed `cast`, then `cublas`, then the kernel columns, in
   that order, every row; on the 1000 W part the first column runs coolest.  That is how
   the cast-copy path (a dequant kernel plus the same cuBLAS GEMM) "beat" standalone cuBLAS
   by 10% on mlp.gate_up tp4 (3649 vs 4042 us).  ncu, one launch each, same clock:
   cast-path GEMM 3.46 ms, standalone cuBLAS 3.46 ms, ours 3.40 ms.  The bench now has
   `--rounds R` (ported from the Hopper line): R passes per row with the column order
   rotated, every implementation included, median per column.
2. Short bursts.  With 20 launches per column the GPU never reaches a steady power state
   and rows flip 0.95 <-> 1.04 between two 5-round runs (`flush_*`); the per-iteration L2
   flush is not the cause (same scatter without it).  With 200 launches per column and 3
   rotated rounds, two independent runs agree within 1% on nearly every row (`long_*`).
   That is the method for any per-row claim at large M; it is also the production regime
   for a 5-17 ms kernel.  Isolated 1000-launch bursts (`lim_*`): und.q tp2 ours 2540 us at
   482 W / 1756 MHz vs cuBLAS 2616 us at 658 W / 1522 MHz; mlp.down tp4 3948 vs 4114 us.
3. The best-variant line of the plots (minimum over six noisy columns) is gone.

**Two fixes.**  und.kv tp8 (N=128): the 2SM tiles pad the 128-row weight to 256 and run
twice the MMA for the same bytes; the 1SM 128x128 tile has no padding.  Steady state at
M=88320: auto 1.13 -> 1.00 (`W8A16Sm100PickTile`: N <= 128 and M >= 16384 -> 1SM).  And
the swizzle rule rounds to the nearest integer instead of a power of two (the sm100
scheduler takes any group size), so K=25600 gets 4, not 2.

**Where the fine grid ends** (`b17_fine`, steady state, kernel / cuBLAS at M=88320):
mlp.down tp1 (K=25600) swizzle 3 / 4 / 5 / 6: auto 1.04 / 1.04 / 1.02 / 1.03, 192-row tile
1.04 / 1.03 / 1.02 / 1.03; mlp.down tp2 (K=12800) 6 / 8 / 10 / 12: 1.02 / 1.04 / 1.02 /
1.08 (160-row tile 1.02 / 1.02 / 1.02 / 1.08); mlp.gate_up tp2 12 / 16 / 20 / 24: 1.00 /
1.00 / 1.02 / 1.02.  mlp.down tp1 / tp2 sit at 1.02-1.04 with every tile order: cuBLAS
runs them with a 256x256 tile (half our L2 bytes per flop, 125 vs 150-164 GB), and our
256x256 tile reaches its clock but spends 8% more cycles staging the converted A in SMEM
because two 256-column accumulator stages fill TMEM.  A 256x224 tile with A in TMEM
would need a single A stage (2 x 224 + 64 = 512 columns): untried, the next lever.

**Verdict at M=88320** (`b17v_*`, two independent steady-state runs): 21 of 24 rows at or
under 1.02 x cuBLAS, 15 of them under 0.99 (und.q tp8 0.92, und.kv tp1 0.95, mlp.down
tp4 0.95, attn.out tp1 0.96, ...).  mlp.down tp2 1.024 / 1.024 and mlp.gate_up tp2 1.022 /
1.022 in the two-column runs, 1.02-1.03 with every tile in a four-column run: on par
within the run-to-run band, not under.  attn.out tp8 (N=5120, K=1024): 0.998 / 1.064 in
the two-column runs, 1.07-1.15 across six columns, best tile the 160-row one at
1.03-1.13, cuBLAS itself 582-630 us between runs; a 0.6 ms kernel with eight k-steps per
tile, where the per-tile prologue / epilogue is the whole story.  Open.

What the three are made of (ncu, `--clock-control none`, one launch each, M=88320):

| row | | time | SM clock | SM cycles | DRAM read | instructions |
|---|---|---|---|---|---|---|
| attn.out tp8 | cuBLAS 128x256 | 0.556 ms | 1.415 GHz | 0.79 M | 0.25 GB | 0.04 G |
| | ours 256x160 | 0.578 ms | 1.448 GHz | 0.84 M | 0.19 GB | 0.22 G |
| mlp.gate_up tp2 | cuBLAS 128x256 | 6.99 ms | 1.373 GHz | 9.59 M | 4.6 GB | 0.20 G |
| | ours (auto) | 7.25 ms | 1.335 GHz | 9.68 M | 5.9 GB | 2.08 G |
| mlp.down tp2 | cuBLAS 256x256 | 7.16 ms | 1.386 GHz | 9.92 M | 11.3 GB | 0.10 G |
| | ours (auto, 192) | 7.24 ms | 1.333 GHz | 9.65 M | 8.3 GB | 1.98 G |

Three different floors.  attn.out tp8 is cycles: 6% more at a higher clock; with eight
k-steps per tile the transform-to-MMA hand-off at every tile start is not hidden.
mlp.gate_up tp2 is DRAM (27% more bytes than cuBLAS at M=88320; the band rule was fitted
at 44160) and clock.  mlp.down tp2 is neither: 3% fewer cycles and 27% fewer DRAM bytes
than cuBLAS, yet a 4% lower clock, with 20x the instruction count.  That is the
converter's own energy: the transform warps' instruction stream costs clock on a
power-capped part even when the memory system is quiet.  Conversions per flop fall with
the activation tile height, and the measured order on that row follows it exactly (128:
1.065, 160: 1.033, 192: 1.024), but the 256-row tile pays 7% for its SMEM-staged A.  The
way through is a 256-row activation tile with A in TMEM (one accumulator stage, or a
224-row tile with one A stage), a mainloop change, not a knob.

Tried (version B18 experiment): a 256x224 tile with a 64-deep k-tile, which fits TMEM
with A still there (2 x 224 + 2 x 32 = 512 columns).  Compiles, passes the tests, and is
slower than auto on every row (mlp.gate_up tp2 1.08-1.10 vs 1.03, mlp.down tp2 1.08 vs
1.02, und.q tp2 1.04-1.05 vs 0.96, attn.out tp8 1.23-1.25 vs 1.13, mlp.down tp1 1.06 vs
1.04): the halved k-tile costs more pipeline than the saved conversions buy.  Removed
from the build together with the cluster-(2,2,1) variants (`W8A16Sm100Cfg<256, 224, 64,
2, true>` and `W8A16Sm100Cfg<256, {128,192}, 128, 2, true, false, 2>` rebuild them;
the `kClusterN` template parameter stays).

Definitive numbers for the three (500 launches per column, 5 rotated rounds, three
independent repeats; attn.out tp8 with 1500 launches), kernel / cuBLAS at M=88320:
mlp.gate_up tp2 1.024 / 1.022 / 1.022; mlp.down tp2 1.024 / 1.026 / 1.025; attn.out tp8
1.042 / 1.042 / 1.042 (the 256x256 tile 1.139 there, the 192-row tile the same as auto).
So at M=88320 Blackwell ends at 21 of 24 rows at or under 1.02 x cuBLAS and three rows
2.2-4.2% over, all three with fewer SM cycles than cuBLAS and a lower clock under the
1000 W cap: the energy of converting on the fly.

Checked and closed: putting the weight on the B operand to get a 256-row activation tile
without the TMEM limit.  The fork's collective sets `SwapAB = !IsATransformed` and swaps
the roles internally, so the converted operand is always the MMA's A (TMEM or
SMEM-staged); conversions per flop are 1 / (activation rows per cluster tile) whichever
way the launcher passes the operands, and the only 256-row-activation configuration is the
256x256 SMEM-A tile already measured (1.07-1.14 on these rows).  Cutting the conversion
energy below the 192-row tile therefore needs a different mainloop (a converted B operand
in SMEM with the activation as a plain TMA A operand, which the fork does not have), not a
configuration.

### Version B19 experiment: the warps that spin (Wed 14 Oct 2026, evening)

Dynamic opcode counts (ncu source page, instructions executed per SASS line, mlp.down tp1
M=88320, our auto kernel): 5.1 G warp instructions, of which F2FP (the conversion) 0.95 G,
IMAD 0.68 G, and SYNCS 0.77 G + BRA 0.50 G + NANOSLEEP 0.31 G = 1.58 G (31%) are
barrier-poll loops.  The hottest single site is one TRYWAIT / NANOSLEEP / BRA loop
(`barrier.h:419`, CUTLASS's blocking `wait()`) re-polled ~226 M times: the hardware
suspend inside `mbarrier.try_wait` lasts tens of ns whatever its hint, so a warp that
waits long keeps issuing.  cuBLAS's warps show 79% of their PC samples suspended on one
NANOSLEEP.SYNCS.  Hopper does not have this (the agent's opcode table: SYNCS + BRA 4% of
ours vs 6% of cuBLAS's there).

Peek-and-sleep waits (`__nanosleep` between non-blocking tries, macros
`W8A16_WAIT_SLEEP_NS` / `W8A16_MMA_WAIT_SLEEP_NS`, default 0):
- transform warps' two per-k-step waits, 100 / 300 / 1000 ns: instruction count 1.98 G
  unchanged, ratios unchanged (`b19_sleep`).  They were already suspended.
- epilogue warps' accumulator wait, 500 / 2000 ns: 1.98 -> 1.92 G, clock unchanged.
- MMA warp's two per-k-step waits plus the two TMA producers' acquires, 50 ns: the
  exact-value test kernel deadlocked (63 min at 100% utilisation, 250 W, killed).  The
  TMA-producer part is reverted; the MMA-warp part is kept behind its macro for one bounded
  retry.  The remaining ~1.4 G of polls are therefore in the MMA warp and the TMA warps,
  the two that wait between every k-step.
- MMA warp's two per-k-step waits alone, 100 ns, bounded retry on umb-b200-263: builds and
  passes the exact-value test (the deadlock was the TMA-producer part), steady state vs the
  0 baseline on the same node: mlp.gate_up tp2 1.041 / 1.041, mlp.down tp2 1.031 / 1.028,
  und.q tp2 0.973 / 0.967, attn.out tp8 1.030 / 1.039.  Nothing.
What it could have been worth, by the Hopper agent's energy-slope probe (+24% instructions
-> -1.1% clock at equal cycles): all 1.5 G poll instructions are ~1.5% of clock, enough for
the two tp2 rows, not for attn.out tp8.  Measured, none of the three sleeping waits bought
any of it, so the polls that remain are not where the energy goes, or the sleep costs what
it saves.  The experimental wait code is reverted; the finding stands as a note.  Blackwell
closes at 21 of 24 rows at or under 1.02 x cuBLAS at M=88320 (mlp.gate_up tp2 and mlp.down
tp2 at 1.02-1.03, attn.out tp8 at 1.03-1.04), the Hopper line at 23 of 24 (attn.out tp8 at
1.03-1.04).  Both residues sit at the energy / per-tile floor of converting on the fly.

### Version B20: where the clock really goes (Wed 14 Oct 2026, night)

Two probes on umb-b200-263, steady state, kernel / cuBLAS at M=88320 (normal build in the
same session: gate_up tp2 1.041, mlp.down tp2 1.028, und.q tp2 0.968, attn.out tp8 1.020;
ncu free clock on mlp.down tp2: 1.255 GHz, 9.65 M cycles, 1.98 G instructions):
- `W8A16_PROBE_NO_CONVERT` (transform warps skip the cvt; the MMA's A operand stays zero;
  same loads, TMEM stores, pipeline): 0.839 / 0.838 / 0.783 / 0.804; 1.562 GHz, 9.59 M
  cycles, 1.15 G instructions.
- `W8A16_PROBE_CVT_ZERO` (cvt instructions execute on zero input, so the MMA still sees
  zeros): 0.848 / 0.853 / 0.802 / 0.812; 1.513 GHz, 9.62 M cycles, 1.62 G instructions.
Same cycles in all three builds; the F2FP instructions cost ~3% of clock; the DATA costs
~20%: real bf16 weight values written to TMEM (STTM), read from TMEM by the MMA and fed
through the datapath against zeros.  cuBLAS runs the same dequantised bf16 values through
its MMA at 1.37-1.39 GHz, but from SMEM.  Our 256x256 tile, the one variant with A staged
in SMEM, ran at 1.315 GHz in the B16 table (cuBLAS 1.333) and lost only on cycles to its
64-deep k-tile.  So the energy term is the TMEM round trip of the converted operand, not
the conversion.

Experiment: `kAInSmem` variants of the 192- and 128-row tiles (k-tile 64, the SMEM budget
allows nothing deeper), steady state, two repeats, kernel / cuBLAS:

| row, M=88320 | auto (A in TMEM) | 192 rows, A in SMEM | 128 rows, A in SMEM |
|---|---|---|---|
| mlp.gate_up tp2 | 1.034-1.042 | 1.274-1.275 | 1.662-1.673 |
| mlp.down tp2 | 1.021 | 1.242-1.243 | 1.619 |
| und.q tp2 | 0.968 | 1.220-1.222 | 1.588-1.609 |
| attn.out tp8 | 1.022 | 1.294-1.302 | 1.667-1.671 |

ncu, mlp.down tp2: 192-row SMEM-A 1.502 GHz but 14.1 M cycles (auto: 1.255 GHz, 9.65 M);
128-row SMEM-A 1.760 GHz but 21.0 M cycles.  The clock comes back exactly as the probes
predicted and the SMEM path (transform STS of the bf16 operand, SS MMA reads, halved
k-tile) costs far more cycles than the clock buys.  The 256x256 SMEM-A tile remains the
best compromise of that family (1.07-1.14).  Both variants removed from the build
(`W8A16Sm100Cfg<256, {192,128}, 64, 2, true, true>` rebuilds them); the two probe
macros `W8A16_PROBE_NO_CONVERT` / `W8A16_PROBE_CVT_ZERO` stay (no cost, documented).

One more for the short-K row (version B21 experiment): the 160-row tile with a 64-deep
k-tile and A in TMEM (six 32-column A stages), meant to amortise the per-tile pipeline fill
on attn.out tp8 over 16 k-steps.  Steady state, two repeats: attn.out tp8 1.17 / 1.20 vs
auto 1.03 / 1.04; attn.out tp4 1.13 / 1.14 vs 1.00; the tp2 rows 1.13-1.15 vs 1.02-1.04.
The halved MMA step costs more than the deeper buffering saves.  Removed.

Conclusion for the Blackwell line: the 2-4% left on mlp.gate_up tp2, mlp.down tp2 and
attn.out tp8 at M=88320 is the energy of carrying the converted operand through TMEM,
a property of this mainloop's design (A in TMEM so that two accumulator stages and the
operand fit).  Closing it needs an A path that is neither TMEM nor SMEM-bandwidth-bound,
which this fork does not have.

Lesson for the whole line, both arches: on a power-capped part a GEMM at 97% tensor-pipe
utilisation is not "compute-bound, done"; every DRAM byte costs clock, and cuBLAS's edge on
the wide rows was never its mainloop but its L2 footprint.

On the "oracle" line of the plots: five interleaved passes of every variant on the und.q
rows (`undq_p*`) put `auto` at or within 1-2% of the median-best variant on all 18 rows,
while a single pass of the same kernel on und.q tp4 at M >= 44160 ranges 0.74-1.22 of
cuBLAS.  The plot's best-variant line was the minimum over six variants of one pass, which
on such a row always lands under `auto`; the plots now take per-row medians over seven
passes (`sweepB15m_p*`) for every line, and the best-variant line is the minimum over the
variant medians.

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
| B15 | (this commit) | scheduler swizzle 8 for weights >= 64 MB at M >= 4096 (`W8A16Sm100PickSwizzle`; `FLASHINFER_W8A16_FP8_SWIZZLE_SM100` / `_RASTER_SM100` pin) | WIN: mlp.gate_up tp1 (N=25600) at M=44160 / 88320 1.32 -> 0.98 / 0.99 x cuBLAS, mlp.down tp2 8-16% faster there; no other row moves beyond noise |
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
