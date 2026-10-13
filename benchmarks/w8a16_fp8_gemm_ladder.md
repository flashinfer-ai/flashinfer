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

## Final state (rung B10)

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
| B10 | (this commit) | native `cvt.rn.bf16x2.e4m3x2` / `cvt.rn.f16x2.e4m3x2` converter (one F2FP per pair; ALU path kept behind `W8A16_FORCE_ALU_CONVERT` and for nvcc < 13.2) | WIN: 1.24 / 1.06 / 1.03 / 1.02 / 1.03 / 1.04 x cuBLAS, 0.928x B9 time |

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

## Results directory

`/home/scratch.ishovkun_gpu/w8a16-results/blackwell/`: `rungB<N>_*_b200_full.csv`,
`rungB<N>_ncu_*.txt`, `rungB5_sass.txt` / `rungB6_sass.txt` (cuobjdump of the module).
