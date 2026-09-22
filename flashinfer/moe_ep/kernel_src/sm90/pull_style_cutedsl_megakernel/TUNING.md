# SM90 pull-style FP8/MXFP4 MegaMoE tuning + performance notes

This document collects the performance work on the `sm90_fp8_fp8_bf16_pull_cutedsl` mega
backend: the measured microbenchmark results, the benchmark methodology
behind those numbers, the knob surface as it exists today, and the open
perf levers.  It is the companion to `SKILL.md` (drop-update workflow) and
mirrors the structure of the SM100 tree's `TUNING.md`.

Unless noted otherwise, every measurement comes from a single 4x NVIDIA
H200 141GB node (EP=4) taken in one session: the two microbenchmark tables
below and the companion `benchmarks/pull_and_push_comparison.md` are one
2026-09-19 session (current table: group_hint 264 on every bucket up to
8192, tail-split pair tasks, FC2 16-byte stores, fused swap-AB generate_c
store, compact pull buffer, swap-AB N=8 rows; sweep and pull/push
comparison back to back on one 4x H200 node with the SM clock locked at
1830 MHz, like the 2026-09-12 table), while the same-node before/after
paragraphs are the 2026-09-19 (small-bucket group_hint) and 2026-09-18
(tail-split port) sessions — both on a node running its 1980 MHz boost
clock unlocked, which costs the long blockwise 8192 / 32768 points 3-5%
through power-cap dips but leaves the interleaved A/B ratios valid — and
the 2026-09-12 (compact pull buffer) and 2026-09-03 (fold layout)
sessions, each on its own node (never compare
numbers across sessions; node-to-node offsets of ~2% are documented in
"Next levers") — at the
kernel drop's DSV4-Pro P03
geometry: **384 experts, top-6, hidden 7168, intermediate 3072
(post-SwiGLU; gate+up 6144), gate_up_clamp 10.0**, tokens-per-rank swept
8..32768 in powers of two (13 points) — the same geometry and knobs as the
kernel team's `moe_hopper_fp8/run_token_sweep_benchmark.py`.  The launch
config per point is the drop's token-bucket heuristic table
(`moe_hopper_fp8/heuristic_config.py`, geometry derived from the kernel
team's 2026-08-19 four-rank H200 sweep at the same vendored kernel
sources, plus the locally added per-bucket `token_back_mode` column from
the 2026-08-23 epi-vs-reuse sweep, and the kernel team's 2026-09-18/19
tail-split / group_hint / token-back retunes — see the knob list below).
Raw rows: `benchmark_data/20260919/20260919_064746_mega_sm90_heuristic_both.csv`
(local archive, not committed).

## Upstream FP8 microbenchmark results (2026-09-19, heuristic launch configs, max-rank µs)

These tables are the upstream #5338 measurements, not a rerun of this MXFP4
integration. New integration measurements are reported separately.

Two timed series per point — the difference is WHAT each call includes:

- **`compute` — the pure compute path.**  Inputs are staged ONCE, then
  `MegaKernelBackend.compute(output=None)` is timed repeatedly: **fused
  mega kernel (dispatch + FC1 + SwiGLU + FC2 + combine) + standalone
  top-k reduce, zero-copy output** (the result stays in the workspace; no
  staging, no output copy).  This is the number to quote for kernel work
  and to compare against the drop's `mega_us + topk_us`.
- **`e2e` — the full production path.**  `MoEEpLayer.forward` is timed:
  **input validation + bf16→fp8 staging quantization + everything in
  `compute` + output copy** into the caller's tensor.  This is the
  serving-relevant number; it has no drop counterpart.

`e2e` time is therefore always ≥ `compute` time (so
`critical_tflops_e2e` ≤ `critical_tflops_compute`); the gap is the
staging + copy overhead quantified in "e2e overhead" below.  TFLOPS use
the drop's per-rank formula (`routed = tok/rank × topk`,
`flops = 2·routed·hidden·(gateup + downproj)`) over the max-rank
(critical-path) time.

The `token back` / `group hint` / `tail split` columns are the per-bucket
`token_back_mode` / `group_hint` / `tail_split_pairs` the heuristic table
now selects (`epi` = `epi_warps`, `reuse` = `reuse_dispatch_warps`; blank
group hint = one wave of clusters).  All other knobs are at their config
defaults — notably `active_dispatch_warps=1` (see "The knob surface").

**per_tensor** — peak 1004 TFLOPS/rank:

| tok/rank | heuristic config                   | token back | group hint | tail split | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|:----------:|-----------:|:----------:|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB M256N16 CGA2x1             |    epi     |        264 |            |      759.5 |    8.3 |    874.5 |        7.2 |
|       16 | swap-AB ping-pong M128N16 CGA1x2   |    epi     |        264 |            |     1079.3 |   11.8 |   1226.3 |       10.3 |
|       32 | swap-AB M256N8 CGA2x1              |    epi     |        264 |            |     1356.3 |   18.7 |   1480.7 |       17.1 |
|       64 | swap-AB M128N8 CGA1x2              |    epi     |        264 |            |     1494.9 |   33.9 |   1617.6 |       31.4 |
|      128 | swap-AB ping-pong M128N8 CGA1x2    |    epi     |        264 |            |     1531.0 |   66.3 |   1653.9 |       61.4 |
|      256 | swap-AB M256N32 CGA2x1             |    epi     |        264 |            |     1653.5 |  122.7 |   1754.0 |      115.7 |
|      512 | swap-AB M256N64 CGA1x1             |    epi     |        264 |            |     1685.6 |  240.8 |   1815.0 |      223.6 |
|     1024 | swap-AB ping-pong M128N64 CGA1x2   |    epi     |        264 |    yes     |     1744.2 |  465.4 |   1845.4 |      439.9 |
|     2048 | swap-AB ping-pong M128N128 CGA1x2  |    epi     |        264 |    yes     |     2239.7 |  724.9 |   2355.4 |      689.3 |
|     4096 | swap-AB ping-pong M128N128 CGA1x2  |    epi     |        264 |    yes     |     3679.3 |  882.5 |   3870.3 |      839.0 |
|     8192 | swap-AB ping-pong M128N128 CGA1x2  |    epi     |        264 |    yes     |     6935.0 |  936.4 |   7201.6 |      901.7 |
|    16384 | swap-AB ping-pong M128N128 CGA1x2  |    epi     |            |    yes     |    13148.5 |  987.8 |  13648.2 |      951.6 |
|    32768 | swap-AB ping-pong M128N128 CGA1x2  |    epi     |            |    yes     |    25877.2 | 1003.8 |  26978.5 |      962.8 |

**blockwise** — peak 852 TFLOPS/rank:

| tok/rank | heuristic config                   | token back | group hint | tail split | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|:----------:|-----------:|:----------:|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB M256N16 CGA2x1             |    epi     |        264 |            |      782.9 |    8.1 |    963.7 |        6.6 |
|       16 | swap-AB M256N16 CGA1x1             |    epi     |        264 |            |     1121.5 |   11.3 |   1323.0 |        9.6 |
|       32 | swap-AB ping-pong M128N16 CGA1x2   |    epi     |        264 |    yes     |     1359.2 |   18.7 |   1569.4 |       16.2 |
|       64 | swap-AB M256N32 CGA2x1             |    epi     |        264 |            |     1561.2 |   32.5 |   1769.5 |       28.7 |
|      128 | swap-AB M256N16 CGA2x1             |    epi     |        264 |            |     1575.0 |   64.4 |   1781.3 |       57.0 |
|      256 | swap-AB ping-pong M128N32 CGA1x2   |    epi     |        264 |            |     1694.1 |  119.8 |   1893.6 |      107.2 |
|      512 | non-swap M64N256 CGA1x1            |    epi     |        264 |            |     1729.2 |  234.7 |   1939.2 |      209.3 |
|     1024 | non-swap M64N256 CGA2x2            |    epi     |        264 |            |     2015.6 |  402.7 |   2226.2 |      364.6 |
|     2048 | non-swap M64N256 CGA2x2            |    epi     |        264 |            |     2763.9 |  587.4 |   3008.6 |      539.6 |
|     4096 | non-swap M64N256 CGA1x1            |    epi     |            |            |     5421.1 |  599.0 |   5784.5 |      561.3 |
|     8192 | non-swap M64N256 CGA2x1            |    epi     |        264 |    yes     |     8191.9 |  792.7 |   8839.1 |      734.7 |
|    16384 | non-swap M64N256 CGA1x2            |   reuse    |            |            |    17097.8 |  759.6 |  18637.3 |      696.9 |
|    32768 | non-swap M64N256 CGA2x1            |   reuse    |            |            |    30479.6 |  852.2 |  33407.6 |      777.5 |

**Before/after on one node (2026-09-19 session, unlocked 1980 MHz node)** —
`group_hint` 264 on the 8-256 buckets (kernel team's retune; every bucket
up to 8192 now carries it) versus the 2026-09-18 table, same node, 2
interleaved rounds, e2e median: geomean **+3.70%** (per_tensor +3.89%,
blockwise +3.50%); compute series +3.99%.  Per bucket (e2e): pt16
**+12.9%**, pt64 **+10.4%**, pt128 **+14.1%**, pt256 +4.6%, pt8 +3.6%,
pt32 +3.2%; bw32 **+18.3%**, bw256 **+19.2%**, bw64 +3.7%, bw8 +3.1%,
bw128 +3.0%, bw16 +1.7%; 512 and up unchanged (within ±1.5%, their rows
did not move).  At these token counts each expert has only a few rows,
so a launch is one pass over the active experts' weights: with the
one-wave default group the FC2 tiles of an expert are handed out right
behind its 48 FC1 tiles and the activation TMA producer spends ~25% of
its time in the fc2 fc1_done spin (kernel team's 4-rank IKET trace);
six experts per group removes the spin.  All 26 points reference-checked
`pass` in both trees and both rounds.

**Before/after on one node (2026-09-18 session, unlocked 1980 MHz node)** —
the kernel team's tail-split port (tail-split pair tasks, per-bucket
`group_hint`, FC2 epilogue 16-byte stores, retuned heuristic table) versus
the pre-port tree (main `01045366`, 2026-09-12 table), same node, 2
interleaved rounds, e2e median: geomean **+5.98%** (per_tensor **+8.34%**,
blockwise **+3.68%**); compute series geomean +6.44% (per_tensor +8.92%,
blockwise +4.02%).  Per bucket (e2e): pt1024 **+20.7%**, pt2048 **+28.8%**,
pt4096 **+27.3%**, pt8192 **+18.5%**, pt16384 +6.0%, pt32768 +11.0%, pt512
+3.3%; bw1024 **+20.1%**, bw2048 **+17.5%**, bw8192 +6.9%, bw512 +3.0%,
bw4096 +2.2%, bw32 +1.7%; every other bucket within ±1% (the port is
neutral where the table row did not change).  The per_tensor 1024-32768
gains are the swap-AB ping-pong M128N128 cga(1,2,1) + tail-split rows
replacing the non-swap M64N128 / M64N256 rows; the blockwise 1024 / 2048
gains are `group_hint` 264 with the write-back moved from
reuse_dispatch_warps to epi_warps.  The FlashInfer numbers reproduce the
kernel team's drop-harness A/B within a few points per bucket.  Sweep
pass/fail: all 26 points reference-checked `pass` in both trees and both
rounds; the new rows are additionally pinned bit-exact by the multirank
`tail_split_pairs` cases.

**Before/after on one node (2026-09-12 session, this table's node)** —
`compact_pull_buffer` (default True) versus the 4-slot pull buffer, same
tree, generate_c off, 2 interleaved rounds, e2e median: per_tensor
geomean **+1.35%** (pt16384 +5.4, pt4096 +3.4, pt512 +2.4, pt2048 +2.3;
the N256 cooperative and N128 ping-pong configs gain one AB stage),
blockwise **−0.25%** (no bucket beyond −1.2%; the blockwise N256 config
keeps 4 stages — see the knob entry); geomean e2e +0.55%, compute +0.79%.
The 16-byte generate_c stores do not touch the generate_c=off path
(off-vs-off within noise, −0.2%).  Against the 2026-09-10 table (a
different node) the peaks moved 903 → 971 (per_tensor) and 828 → 848
(blockwise) TFLOPS/rank; per the rule below, only the same-node
before/after numbers are attributable to the code.

**Swap-AB N=8 rows (2026-09-10 session, the node of the previous table)**: old table vs
new table, interleaved 2 rounds, e2e median — pt64 **+14.1%**, pt128
+4.7%; pt32 moved from non-swap M64N256 to cooperative swap M256N8 CGA2x1:
+5.7% / +6.4% e2e (compute +7.4% / +7.1%) against the non-swap row, the
ping-pong swap M128N8 twin only ties it.  pt16 (+1.8%) and bw64 (+1.7..
+2.9%) also measured positive at N=8 but were judged too small to move
the table; every other bucket within ±1%.  Details and the rejected
buckets in "Next levers" item 4.

**Before/after on one node (2026-09-03 session, table without the N=8
rows)** — the fold default: `fold_producer_warps` + the re-calibrated
table — blockwise non-swap 512-32768 cooperative M64N256, per_tensor 8
cooperative, per_tensor 64 basic; versus the pre-fold default — producer
warpgroup + FC1 store offload + the 2026-08-19 table — re-run on the same
node the same hour, compute TFLOPS: per_tensor geomean **+0.76%** (pt8
+5.4, pt64 +2.7, pt512 +2.7; worst pt16384 −1.5), blockwise geomean
**+12.20%** (bw512 +12.7, bw1024 +13.2, bw2048 +22.3, bw4096 +16.7,
bw8192 +29.8, bw16384 +27.4, bw32768 +44.5; the swap buckets 8-256 are
within ±3%).

Do not compare these absolute numbers with earlier revisions of this table:
nodes with identical clocks and software differ by ~2% for identical
configs.  Concretely, the previous revision (2026-08-30, a different node)
listed per_tensor peak 936 / blockwise 594; re-running that exact pre-fold
config on this node gives geomean −2.04% (per_tensor) / −1.95% (blockwise)
purely from the node, on top of which the new default adds +0.76% /
+12.20%.  The apparent per_tensor decline between revisions is therefore
the node, not the code.  Every decision in this document was made on
interleaved same-node A/Bs, never on cross-session sweeps.

### e2e overhead (the production path)

`e2e` minus `compute` is ~120 µs (per_tensor) / ~210 µs (blockwise, whose
staging quant also emits the per-128 activation scales) at small token
counts, growing to ~2.9 ms at 32768 — dominated by the torch-composed staging quant plus
the output copy.  The SM100 tree eliminated the analogous cost with a
fused single-launch quant+repack kernel (`FLASHINFER_MEGA_FUSED_STAGE`);
the SM90 tree has no counterpart yet — this is the top e2e lever (see
"Next levers").

## The knob surface

Geometry and behavior knobs are explicit
`Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig` fields, resolved once per
session at workspace allocation.  On top of the explicit fields the tree
now carries the SM100-style tuning stack (`shim/tuner.py`,
`shim/autotune.py`, `shim/knob_cache.py`): the config's `knobs=` field
accepts a knob dict, `"auto"` (collective online autotune on first
compute, winner persisted to the knob cache), or `None` (cache lookup,
then the heuristic table).  The autotune candidate set is the heuristic
winner plus every geometry that wins some bucket of the table (19 today
— tile / cluster / ping-pong together with the row's `group_hint` and
`tail_split_pairs`, derived programmatically) crossed with both validated
token-back modes — 38 candidates.

- `fp8_scale_mode` — `"per_tensor"` (per-expert weight scalar + static
  activation calibration scalars, identical on all EP ranks by contract) or
  `"blockwise"` (DeepGEMM-style 128-block fp32 scales; requires
  hidden/intermediate %128).
- `swap_ab` + `pingpong` + `mma_tiler_mnk` + `cluster_shape_mnk` — layout,
  scheduling, tile, and CGA shape.  Leave ALL four `None` to use the drop's
  token-bucket heuristic table (`moe_hopper_fp8/heuristic_config.py`, keyed
  on `fp8_scale_mode` and max tokens/rank, derived from the 2026-08-19
  four-rank H200 DSV4 sweep; re-calibrated on 2026-09-02 under the folded
  warp layout: blockwise non-swap 512-32768 -> cooperative M64N256,
  per_tensor 8 -> cooperative swap M256N16, per_tensor 64 -> basic swap
  M128N64, see `fold_producer_warps` below; 2026-09-10: swap-AB N=8 for
  per_tensor 32/64/128, see "Next levers" item 4; 2026-09-18: per_tensor
  1024-32768 -> swap-AB ping-pong M128N128 cga(1,2,1) with tail-split
  pair tasks, see `tail_split_pairs` / `group_hint` below);
  setting any one switches to
  manual mode with
  drop-driver defaults for the rest (non-swap (64, 128, 128), swap-AB
  (256, 32, 128), (128, 32, 128) with ping-pong; cluster (1, 1, 1)).
  Kernel-legal geometry: non-swap M∈{64}, N∈{128,256}; swap-AB M∈{128,256},
  N∈{8,16,32,64,128} (N=8 = wgmma m64n8k32, admitted 2026-09-10 and used by
  the per_tensor 32/64/128 rows; bit-exact for per_tensor
  ping-pong / cooperative and blockwise ping-pong / cooperative
  since 2026-09-10 — the blockwise token-scale box (n×4 fp32 = n·16 B) used
  to ride B's cluster-M multicast, whose per-CTA sub-box is n/cluster_m·16 B
  = 64 B at n=8, below TMA's 128 B smem-destination alignment; the kernel now
  loads that box non-multicast whenever the sub-box is not a 128 B multiple,
  n≥16 configs are untouched); K=128; CGA (m,n)∈{(1,1),(2,1),(1,2),(2,2)},
  k=1.  For reference, DeepGEMM's SM100 mega kernel cannot go below a
  16-token block at all: it always swaps A/B onto a 2-CTA UMMA_M=256, whose
  N must be a multiple of 16 (`BLOCK_M % 16 == 0` static_assert); its
  `kMinCandidateBlockM = 8` only sizes pool padding.
  Ping-pong needs one physical warpgroup per task tile: N=128 non-swap,
  M=128 swap-AB.
- `load_balance_mode` — `"static"` (default, used by the correctness
  tests) or `"atomic_counter"` (the drop's perf-sweep setting; used by the
  benchmark for reference parity).
- `token_back_mode` — `epi_warps`, `reuse_dispatch_warps`, or
  `standalone_warps` (four dedicated token-back warps).  Left unset it
  follows the per-token-bucket heuristic table (2026-08-23 four-rank H200
  sweep: epi_warps small/mid buckets, reuse_dispatch_warps at the
  GEMM-bound tail; the 2026-09-18 retune moved every per_tensor bucket
  and blockwise 1024-8192 back to epi_warps — with `group_hint` 264 the
  epilogue-warp write-back wins or ties there — so reuse_dispatch_warps
  remains only on blockwise >= 16384) and is a tuner candidate axis.  All six token_back x reduce combinations are
  kernel-supported; `epi_warps` / `reuse_dispatch_warps` /
  `standalone_warps` are all bit-validated by the `mega_sm90` multirank
  oracles.  `token_back_by_dispatch` remains as a legacy bool alias
  (True -> `reuse_dispatch_warps`).
- `in_kernel_fc2_reduce` — REDG atomic-add combine (bf16 unordered sum,
  nondeterministic; validated in `mega_sm90` with the roundoff-envelope
  band, not measured in the sweep above).
- `active_dispatch_warps` — how many of the 4 dispatch warps do token-comm
  work AT ALL (prep + barrier + pull + reuse token-back; 1/2/4, default 1).
  The physical layout stays at 4 (setmaxnreg is warpgroup-granular); warps
  beyond the count skip the whole dispatch body and only rejoin at
  kernel_tail.  With the default count of 1 those three idle slots host
  the TMA-A / TMA-B / scheduler roles (`fold_producer_warps`, below).
- `generate_c` (default False; training forward) — the FC1 epilogue also
  writes the raw pre-SwiGLU gate+up accumulator (pre-clamp, pre-routing-
  weight, dequantized) as BF16 to an expert-major pool tensor
  `fc1_c[pool_row, 2·intermediate]` in the kernel's gate/up-interleaved
  column order, each local expert's segment padded to 128 rows (pad rows
  stay zero; row order inside a segment is the dispatch arrival order) —
  the same contract as the Blackwell MXFP8 kernel's `generate_c`.  Read it
  back from `symm_buffer.fc1_c` after the launch; segment offsets are
  `sum(round_up(count[i], 128), i < e)` over the per-expert routed-token
  counts.  Store path (both layouts): the values go straight from the
  wgmma fragments to GMEM as 16-byte stores after a lane transpose —
  non-swap: a quad holds 64 consecutive raw columns of one row for each
  pair of gate/up groups, two `shfl.bfly` levels across `lane_mod` give
  every lane one 16-byte chunk (one `st.global.v4` per (row, pair), a quad
  writes a 64-byte-aligned 64-byte run); swap-AB: the eight lane groups
  hold one 16-byte chunk per (token, m_sub, gate/up), two 32-bit butterfly
  levels on lane-group bits 2/1 plus a `prmt` half-word exchange on bit 0
  leave each lane group one chunk (1 STG.128 per token group instead of 8
  STG.16).  On swap-AB the store is fused into the SwiGLU pass (one token
  group at a time, packing the already scaled gate/up registers with
  `cvt.rn.bf16x2` before the lane transpose): a separate raw-C pass kept
  all 128 accumulator registers live and made the M128N128 ping-pong
  kernel spill 868 B, which turned a 3 µs pass into a 25 µs SwiGLU
  epilogue (17-22% generate_c overhead at 2048-32768 tokens in the drop
  harness); fused, spills are zero and the drop's overhead is 0.7-2.4%, with
  the generate_c=off kernels PTX-identical.  The GMEM pointer carries an explicit 16-byte alignment
  assumption (`make_ptr(..., assumed_align=16)`): a plain `autovec_copy`
  into the BF16 view has no alignment fact and degrades to 16-bit stores
  — the first version's "bf16x2" pair stores compiled to `STG.E.U16`,
  which was the whole generate_c overhead.  The row/column predicate is
  exact (`intermediate_gateup % 64 == 0`).  No SMEM staging: a TMA-staged
  variant (two (64x64) BF16 stages per epilogue warpgroup) was measured
  and rejected — the 16 KB per warpgroup comes out of the AB pipeline
  budget and drops the N256 configs from 4 to 3 stages (−10..−38% on
  those buckets) while gaining only +2..+3% where it fits.
  Cost when on (FlashInfer bench, 4x H200, 2 interleaved generate_c rounds
  against one generate_c=off run, e2e median, 2026-09-18 node at 1980 MHz):
  geomean +1.1% (per_tensor) / +0.5% (blockwise); the swap-AB M128N128
  tail-split rows pay pt2048 +2.5, pt4096 +2.8, pt8192 +1.0, pt16384 +3.1,
  pt32768 +2.4% — before the fused store they paid +14.9 / +14.5 / +16.3 /
  +5.9 / +14.2% on the same node (per_tensor geomean +5.1%), and with the
  16-bit stores the non-swap rows of the older table paid −8..−16%.  Off
  costs nothing: the store path is compiled out (same-node off-vs-off
  between the fused and the separate-pass build within ±1.6% per bucket,
  one round).  Test
  `test_..._generate_c` (both layouts x both scale modes vs the multi-rank
  torch reference's `return_fc1_gateup`).
- `compact_pull_buffer` (default True) — size the dispatch pull buffer by
  the *active* dispatch warps (`active_dispatch_warps` x
  `max(hidden_bytes, tb_chunk_bytes)` = 7 KiB at hidden=7168) instead of
  one `hidden_bytes` slot per physical dispatch warp (28 KiB): the idle
  warps never enter the dispatch body and the reuse token-back walkers
  are exactly the active warps, so the three idle slots were dead SMEM.
  Token-comm misc SMEM 31264 -> 9760 B per CTA and the difference goes to
  AB pipeline stages: per_tensor N256 coop 4 -> 5, N128 ping-pong 7 -> 8,
  swap-AB tiles +1; blockwise N256 coop stays at 4 (its 1 KiB/stage of
  block scales plus the 16 KiB output ring leave it ~3.6 KiB short of a
  fifth stage — shaving that, e.g. one fewer fc2 store slot, is the
  follow-up).  Measured (same bench, generate_c off, 2 interleaved
  rounds): per_tensor e2e +1.35% (pt16384 +5.4, pt4096 +3.4, pt512 +2.4,
  pt2048 +2.3), blockwise −0.25% (no bucket beyond −1.2%); geomean e2e
  +0.55%, compute +0.79%.  Output-invariant; `False` restores the 4-slot
  buffer for A/B (`--no-compact-pull-buffer` on the bench).
- `fold_producer_warps` (default True; requires `active_dispatch_warps == 1`)
  — folds the TMA-A / TMA-B / scheduler warps into the three idle
  dispatch-warpgroup slots and drops the separate producer warpgroup
  (including the epi_aux store-server warp): `[epi][disp0 tma_a tma_b
  sched][tb?]`, 128 fewer threads per CTA and the 2-WG (N256 / swap M256)
  register budget falls from the 65536 cap to 60160.  `dispatch_warp_id`
  stays a 4-tuple with the repurposed warps idling through the dispatch
  hook, so every TokenComm count (num_dispatch_warps, the 128-thread
  nvlink barrier, the kernel-tail range) is unchanged; only the register
  budget and `num_other_warps` shrink.  No epi_aux warp means no FC1 store
  server, so `fc1_early_done_publish` is forced on (swap-AB early-pub,
  once actually active, matches the offload's swap gain).  The freed
  budget is fed back to the epilogue by `fit_epi_registers()` (216→232
  on 2-WG kernels; N128 is already at the 256 ceiling) — measured neutral
  by itself (±0.7%), so registers are NOT the lever; the layout is.
  Measured 4x H200 1830 MHz, interleaved 2x2: vs the old layout with
  early-pub geomean +2.05% (bw1024/2048 +7.3%, nothing negative); vs the
  previous default (old layout + store offload) geomean +1.13% (swap
  non-pp pt512 +3.0, bw1024/2048 +3.4..+4.0; only the GEMM-bound tail
  pt16384 −1.3 / bw4096 −0.8).
  The big consequence is scheduling.  Terminology: **basic** = one
  epilogue WG owns a task tile (non-swap N128 / swap M128, no ping-pong);
  **ping-pong** = two WGs alternate basic-size tiles; **cooperative** = two
  WGs split one doubled tile (N256 / M256).  A bucket's ping-pong twin is
  therefore the HALVED tile + pingpong, and its cooperative twin the
  DOUBLED tile − pingpong; every bucket was re-measured against both twins
  with its own cluster shape / accum / token-back preserved (2x2,
  interleaved, per-run outliers inspected; every bucket has all three
  modes measured).  per_tensor: cooperative buckets hold against both
  twins (basic −3..−14, ping-pong −2..−16) and ping-pong buckets hold
  against their cooperative twin (−4..−16) and, with two exceptions,
  their basic twin — pt8 → cooperative swap M256N16 (+3.9%) and pt64 →
  basic swap M128N64 (+3.9%) won consistently across three interleaved
  runs on two nodes and moved.  blockwise swap buckets likewise hold (coop
  8/16/64/128 vs basic −4..−7 / vs pp −8..−19; pp 32/256 vs basic −1..−3 /
  vs coop −6..−7).
  blockwise non-swap 512-32768 flip: cooperative M64N256 beats the basic
  M64N128 (and its ping-pong twin, which gains only bw2048+ +4..+10 and
  loses at 512) by bw512 +11.7, bw1024 +8.7, bw2048 +14.4, bw4096 +13.1,
  bw8192 +17.9, bw16384 +16.2, bw32768 +29.4 (geomean +15.8%), bit-exact
  (`test_..._blockwise_coop_n256`, cga 1x1 and 2x2) — the table now
  selects it; directly re-measured twins on a third node: cooperative
  ahead of basic by 8..30% and of ping-pong by 9..23% at every one of
  those buckets.  Attribution: the epilogue register refit is not the lever
  (coop at 232 vs 216 regs: −0.4%); the layout is — under the old
  producer-warpgroup layout the same cooperative tile runs 13.6% BEHIND
  basic and the same ping-pong 24% behind, and the fold makes both 2-WG
  modes ~43% faster on blockwise non-swap.  Why the old layout penalises
  2-WG blockwise so heavily is an open question (see Next levers).
  Standalone token-back was re-measured under the fold too and loses on
  every bucket (pt −1..−7%, bw −2..−9%).  Validated bit-exact by
  `test_..._fold_producer_warps` (1-WG / 2-WG / swap) plus the full
  multirank suite under the new default.
- `fc1_store_offload` (default True, but inactive under the default folded
  layout — it needs the epi_aux warp, i.e. `fold_producer_warps=False` or
  `active_dispatch_warps != 1`) — the empty warp runs an FC1 store
  server: the epilogue only R2S-stages FC1 output and hands
  (slot, dest, done-flag) over a per-WG smem mailbox FIFO; the server
  issues the TMA store, waits full completion, and release-publishes
  fc1_done immediately — hoisting the publication ahead of the epilogue's
  consume_next stall and boundary barrier (that hoist, not the store work
  itself, is the win: at small tiles the sched-descriptor wait otherwise
  defers every fc1_done by ~a tile).  Self-gating to non-ping-pong
  kernels (ping-pong's retire section already publishes before
  consume_next, and coop+offload measured 8-30% behind ping-pong);
  covers BOTH non-swap and swap-AB non-pp.  For swap-AB the offload is
  strictly better than in-epilogue early publication because it keeps the
  baseline's store/consume_next overlap (the server drains while the
  epilogue sits in consume) AND hoists the publish — early publication has
  to move the drain before consume and loses that overlap, netting ~0.
  A dynamic register fit sizes the store server to the CTA's remaining
  64K budget (reclaims the 2-WG epilogue's 216→200 headroom; 224 regs at
  1 WG, 152 at 2 WG) and falls back to `fc1_early_done_publish` when even
  a lean 88-reg server does not fit.  The 2-WG (N256 / swap M256) path
  runs a dual-FIFO server (one slot per epilogue WG) with the fc2 spin
  threshold scaled x2 (each WG-half publishes +1).  Measured (4x H200,
  1830 MHz, per-bucket vs no-offload): non-swap bw512 +21~23%,
  bw1024-4096 +6~8%; swap-AB pt256 +15%, pt512 +13%, bw128 +10% (the
  heuristic's swap non-pp buckets).  Correctness: multirank swap_ab and
  torch-oracle tests pass bit-exact with the offload active.
- `fc1_early_done_publish` (default False) — lighter variant: the epilogue
  itself publishes fc1_done right after its store drains, before
  consume_next.  Implemented for every layout (non-swap 1/2-WG, non-swap
  pp, swap-AB non-pp with a pre-consume drain; the fc2 spin threshold
  scales x2 for 2-WG non-pp tasks).  Recovers most of the offload's
  mid-token win on N128 (bw512 +21%) but measured net-neutral-to-negative
  everywhere else: big tiles pay the fence+red on the epi stream
  (bw8192-16384 −1~2%), swap-AB non-pp loses the store/consume_next
  overlap its baseline drain placement had, and non-swap pp's retire
  section already publishes early.  Superseded by the offload wherever
  that is active and auto-enabled as its register-fit fallback; kept as a
  tuner axis.
  Output-invariant: sustained pull bandwidth ~= warps x SMs x hidden_bytes
  / read-RTT, and on H200 two warps per SM is ~2x the bandwidth-delay
  product — enough headroom while keeping the NVLink read queue shallow
  (a 2026-08-29 clock-locked sweep measured 1/2 warps ahead of 4 by 1-3%
  in the dispatch-sensitive 256..8192 buckets, flat elsewhere).
- `group_hint` (default None = one wave of clusters, `max_active_clusters`
  from the DSL's occupancy query) — scheduler group size in FC1 cluster
  tiles.  Several experts per group keep the FC2 tiles of one expert from
  spinning on `fc1_done` right behind that expert's own FC1 tiles, which
  only matters while experts are small (few tiles each).  Left unset it
  follows the heuristic table's per-bucket value (264 on every bucket up
  to 8192 except blockwise 4096: 2026-09-18 for 512-8192, where the kernel
  team's same-node 4x H200 A/B measured −3..−16% and it lets epi_warps
  overtake reuse_dispatch_warps at blockwise 1024 / 2048; 2026-09-19 for
  8-256, where the one-wave group made the activation TMA producer spend
  ~25% of its time in the fc2 fc1_done spin — FlashInfer same-node A/B
  pt16/64/128 +10..+14%, bw32/256 +18..+19%, the rest +2..+5%).  An explicit value wins over
  the table; a knob-cache / `knobs=` entry that moves the geometry drops
  the table's value (it was tuned with that geometry).  Output-invariant
  tuner axis.
- `tail_split_pairs` (default None = the heuristic table's per-bucket
  value; legal only with a 2-CTA token cluster and a 1-CTA weight cluster,
  i.e. swap-AB cga (1, 2, 1) or non-swap cga (2, 1, 1)) — an expert whose
  token count is an odd number of CTA tiles ends in a cluster block whose
  second CTA has no tokens.  The scheduler turns that tail block into
  *pair tasks*: both CTAs compute the single valid token tile against
  adjacent weight tiles, so the expert contributes `ceil(W/2)` tail tasks
  instead of `W` and neither CTA idles; the weight operand of a pair task
  is loaded through a second, non-multicast TMA atom picked per work tile
  by the weight producer warp (blockwise weight scales already follow the
  CTA's decoded weight tile).  The dispatch-driven token-back walkers
  expect `token_cluster * ceil(W/2)` fc2_done publishes for a split tail
  (`fc2_publishes_per_split_tail_tile`; the CTA tile size detects the odd
  count).  The 2026-09-18 retune made this the default for per_tensor
  1024-32768 (swap-AB ping-pong M128N128 cga(1,2,1) replaces the non-swap
  M64N128 / M64N256 rows: the kernel team's same-node A/B measured −25 /
  −24 / −19 / −9 / −13% at 2048 / 4096 / 8192 / 16384 / 32768, −15% at 1024
  with group_hint) and for blockwise 32 (−3%) / 8192 (−8%).  `FP8_TAIL_SPLIT=1`
  turns it on for every selected config whose geometry qualifies (drop
  harness and shim alike, through `heuristic_config`).  Bit-exact with the
  plain schedule (`test_..._tail_split_pairs`: both layouts x both
  token-back placements at an odd CTA-tile count, plus the drop's
  `test_tail_split_sched.py` scheduler contract).  generate_c runs the same
  rows: the swap-AB raw fc1_c store is fused into the SwiGLU pass (per
  token group, from the already scaled gate/up registers), so the M128N128
  ping-pong kernel no longer spills with it on — the drop's 4-rank
  generate_c overhead fell from 17-22% to 0.7-2.4% at 2048-32768 and the
  earlier per_tensor 16384 training override was dropped again.
- `dedup_dispatch`, `grouped_token_back`, `combine_format` — top-k dedup
  on dispatch / combine and the quantized combine wire; see
  `dedup_topk_design.md`.
- `fp8_accum_mode`, `kind` (e4m3/e5m2), clamps.

## Humming MXFP4 tuning and cache

The MXFP4 backend supports fused execution only and reuses the shared
collective scorer and persistent JSON cache. `knobs="auto"` times compatible
candidates with three warmups and ten synchronized `perf_counter` samples:
each rank takes its median, then `MAX` across ranks determines the score. `knobs=None` does
no timing; it resolves the cache and then the routing/token-bucket heuristic.
An explicit complete tactic bypasses both.

Inputs are contiguous CUDA `uint8` `PrequantizedMoEWeights`. For `E` local
experts, hidden `H`, and post-SwiGLU width `I`, packed E2M1 payloads have shapes
`w13[E,2I,H/2]`, `w2[E,H,I/2]`; raw K32 E8M0 scales have shapes
`w13_scale[E,2I,H/32]`, `w2_scale[E,H,I/32]`. H and I must be multiples of 128.
Activations use E4M3 and one FP32 scale per routed row, replicated into the
four-column communication layout. No BF16-weight fallback or persistent FP8
weight conversion is performed.

The geometry union has 17 deduplicated tactics from the block-permutation and
published-exact winners plus two H20-derived anchors. Eligible optional
strategies originally expanded it to 23 candidates for H7168/I3072/E384/EP4.
For that model/EP domain, the live union now retains those 23 and adds three
measured large-token configurations plus 12 distinct tail-pair neighbors: 38
candidates total. Each neighbor retains the original tile, communication and
group/stage settings, changes the cluster to `(1,2,1)` and uses whole-tile
readiness. Existing N8 selection is retained where legal. Every model considers
this same bounded extension catalog, filtering each candidate by tile alignment
and the existing N8/readiness eligibility rules. There is no exact-model
whitelist for tail pairs; the final count depends on which candidates are legal
for the model and EP size. The frozen default buckets are unchanged.

Online and offline tuning call the same
`hopper_mxfp4_optimization_candidates`; token capacity changes ordering, not
the union. Only the complete canonical ordered
list may write a production cache entry. A subset, reordered list or
`--max-candidates` run still applies its measured winner without persisting it
under the complete-union identity.

```bash
torchrun --standalone --nproc_per_node=4 -m flashinfer.moe_ep.tune \
  --dtype sm90_mxfp4 --routing-profile block_permutation_v1 \
  --hidden 7168 --intermediate 3072 --num-experts 384 --topk 6 \
  --max-tokens 8 32 64 128 256 512 1024 2048 \
  --gate-up-clamp 10 --warmup-iters 3 --timed-iters 10 --seed 0
```

Use a fresh `FLASHINFER_MOE_EP_KNOB_CACHE` path for tuning and retain it for
replay. Omit `--live-tokens`, or set it equal to every listed capacity: the
cache has no independent live-token axis. The MXFP4 CLI defaults clamp to 10;
a runtime `gate_up_clamp=None` has a separate identity and can be tuned online.

Schema-v1 cache rows match device, precision/scale mode, EP size, model shape,
top-k and clamp. MXFP4 additionally requires exact routing profile, compute
capability, SM count, and tuning provenance. Provenance binds the fused format,
shipped manifests and live candidate union. Missing/stale identity fields
cause a miss. Token capacity selects an exact bucket, otherwise the smallest
bucket above the request, otherwise the largest below. Ordinary FP8 matching
is unchanged. Tactics must satisfy the current field, type and geometry
validation before use.

### Guarded implementation and optional strategies

`mxfp4_policy.py` owns the MXFP4 domain; `fused_comm_policy.py` owns the shared
BF16 output-layout and dispatch-count capabilities:

- Paired FC2 BF16 stores and row-address reuse require M256/N64, complete
  channel clusters (`H > 0`, `H % (256 * cluster_m) == 0`), BF16 combine and
  direct epilogue token-back without deduplication or in-kernel reduction.
  MXFP4 retains its K256 domain. GPU qualification includes H4096/6144/7168/8192
  with CGA2x1x1; other aligned shapes are a layout rule, not exhaustive GPU
  coverage. Other layouts retain scalar stores.
- Contiguous 512-byte auxiliary-offset copies retain the existing stage
  completion protocol for non-pingpong K256. K128 keeps its original path.
- Zero-count elision skips zero local contributions without removing the grid
  rendezvous, empty-expert/rank completion broadcasts, system fence or reset.
  MXFP4 enables it only without dispatch deduplication.

FP8 keeps paired stores and zero-count elision **disabled by default**:
measured benefits were workload-dependent. Independent internal diagnostic
controls exercise all four combinations; they add no public tuning axis.
The shared compiled identity includes `fused_comm_v1`. MXFP4's defaults remain
unchanged. Its compact pull buffer and training `generate_c` paths remain off;
FP8 retains main's settings. Per-tensor FP8 has no corresponding weight-offset
payload; blockwise FP8 uses a Float32 scale per warpgroup/K128. Tail-N8 and
segmented readiness also require different FP8 layout/scale handling and are
not enabled for FP8.

Two optional MXFP4 strategies expand the tuning domain:

- `fc2_tail_n8=False`: within the original H7168 M256N64K256 domain, use N8
  math for an FC2 task with at most eight valid tokens. Physical N64 staging
  and output layout do not change; the public frontend does not gain tile N8.
- `fc1_ready_mode="tile"`: `"k256"` additionally requires
  H7168/I3072/E384/EP4, CGA2x1x1, effective early FC1 publication and no store
  offload. It publishes 48 K64 completion bits after both post-SwiGLU data and
  scales store; FC2 acquires four bits per K256 segment. Int64 counter slots,
  workspace reset and compile identity change together.

MXFP4 also supports main's `tail_split_pairs` through the existing complete
`knobs` mapping. It requires swap-AB with `cluster_shape_mnk=(1, 2, 1)` and
`fc1_ready_mode="tile"`; packed weights and their auxiliary offsets follow the
same task mapping. This is scheduling inside one fused kernel, unrelated to
removed Green Context execution. The bounded normal tuning union above can
select it through existing `knobs="auto"` or offline tuning; no new public
flag is required. An untuned heuristic still defaults to `False`. An explicit
two-candidate comparison is a subset experiment, not a full 38-candidate tune,
and cannot populate the production cache under its complete-union identity.

A historical 17-field tactic resets all three strategies to their defaults.
The older 19-field form preserves its N8/readiness values and adds
`tail_split_pairs=False`; normalized tactics have 20 fields. N8 and readiness
must still appear together, while `tail_split_pairs` is independently optional.
Unsupported combinations reject before allocation. Applying a legacy tactic
releases the old workspace. Benchmark spellings remain `--mxfp4-fc2-tail-n8`
and `--mxfp4-fc1-ready-mode k256`; no additional tail-pair CLI option is added.
Actual strategies enter the runtime and compiled identities. The cache
provenance records the exact extension strategies and per-candidate eligibility
policy. Cache rows from the previous model-whitelisted domain miss after this
change, so newly eligible candidates participate in the next live tune. Frozen
manifests remain unchanged.
`fused_local_v2` invalidates earlier fused winners after
correcting the bulk weight-offset completion notification. Bulk copies retain
their transaction-byte tracking and use ordinary producer barrier arrivals;
non-bulk `cp.async` copies retain their existing completion tracking.

### Tiny-value quantization

The fused Mega input and post-SwiGLU quantizers preserve normal/zero arithmetic.
For `0 < amax < 448e-30`, use the finite FP32 multiplier
`q = 448 / max(amax, FP32(448 / FLT_MAX))` and separately round `d = RN(1/q)`.
Quantize with retained q: recomputing it from subnormal d may overflow. The
smallest d is `2**-128`; unrepresentable output may still round to zero.
This correction is MXFP4-only and requires Mega token communication in the
shared epilogue. It keeps the Torch amax reduction and uses a cached CuTe
quantizer on the current stream; warm it eagerly before ordinary Graph capture.

Safe-quantization and multirank tests cover normal byte equality, FP32 boundary
values, separate q/d references, empty batches, stream/Graph replay and complete
tiny outputs. References use effective Humming weights, not original
unquantized weights. Policy/config/tuner tests cover strategy and cache guards.

### Direct benchmark replay

```bash
torchrun --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py \
  --backend sm90_fp8_mxfp4_bf16_pull_cutedsl --scale-mode mxfp4_hybrid \
  --mxfp4-tactic-source cache_or_heuristic --tokens 32,2048 \
  --routing-mode block_permutation --no-sparse-data --warmup 10 --iters 50
```

Check the resolved tactic: a cache miss uses the heuristic. FP8 and MXFP4 use
the same direct CUDA-event timing and compute/E2E boundaries; CSV rows carry
`compute_launch_mode=direct`. MMA overrides use `--mma-tiler M,N[,K]`;
omitting K preserves the historical K128 behavior. MXFP4 accepts K128/K256;
the FP8 benchmark keeps K128. Cluster overrides use `--cga M,N`, with K=1,
for explicit MXFP4 and manual FP8 layouts. FP8 tuner candidates can be passed
as `--fp8-knobs-json` objects, including their boolean `tail_split_pairs` field.
Historical tables above belong to
their recorded source/session; the PR reports new measurements separately.
Requested clocks alone do not establish a fixed observed clock.

## Sweep methodology + environment (reproduce recipe)

**Hardware / software.**  One H200 node, 4x NVIDIA H200 141GB (sm_90,
cc 9.0) over NVLink, SM clock locked at 1830 MHz on all four GPUs
(`nvidia-smi --query-gpu=clocks.sm,clocks.max.sm` reports 1830 / 1980 MHz;
every sweep script records it to `clocks.txt` before timing, and no
row in this document was taken unlocked).  Python 3.12, torch `2.12.0+cu130`,
`nvshmem4py-cu13`, **`nvidia-cutlass-dsl 4.6.0`** — the measurement
environment, pinned separately from the repository's supported range
(`requirements.txt` >= 4.6.2a0, `pyproject.toml` cu13 extra >= 4.7.0a0);
the drop pins `4.5.0dev0`, and 4.6.0 compiles and runs this SM90 tree.  Whether the SM100
tree's ">=4.6.1 perf floor" finding applies to the SM90 kernels is
UNTESTED — worth one A/B run.

**Harness.**  `benchmarks/bench_moe_ep_sm90_mega.py`, direct launch for both
FP8 and MXFP4, with one torchrun process
per GPU:

```bash
torchrun --nproc_per_node=4 benchmarks/bench_moe_ep_sm90_mega.py
```

Defaults: heuristic launch configs, tokens 8..32768 (13 points) ×
{per_tensor, blockwise} = 26 rows, drop-recipe fp8 payloads, 5s cooldown
before each timed series, results archived to
`benchmark_data/<date>/<date>_<time>_mega_sm90_<order>_<scale>.csv`
(directories auto-created; the CSV carries the resolved heuristic config
columns).  Axes: `--tokens`, `--scale-mode`,
`--swap-ab`/`--no-swap-ab`/`--both-orders` (fixed layouts instead of the
heuristic), `--mma-tiler M,N`, `--kind`, `--token-back`,
`--load-balance-mode`, `--no-sparse-data`, `--cooldown-s`, `--output-csv`.

**Problem.**  The drop's block-permutation balanced routing over all 384
experts; fp8 payloads per the drop perf recipe (`--no-sparse-data`
switches to dense quantized-randn model data).
per_tensor activation scales are static config scalars identical on every
rank.  Env parity with the drop harness: `NCCL_NVLS_ENABLE=0`,
`NVSHMEM_DISABLE_NVLS=1`.

The shared benchmark timer performs warmup, synchronization, and CUDA-event
timing without graph
capture or replay. Ordinary library CUDA Graph support remains available and
has separate functional regression tests. Historical Graph timings are a
different launch protocol and must not be mixed with direct-launch results.

**Timed regions** (both barrier+sync-fenced per iteration, per-rank CUDA
events, warmup 3 + 20 timed iters, matching the drop's counts):

- `compute` — pre-staged inputs, repeated `compute(output=None)`
  (zero-copy view).  Closest to the drop's `mega_us + topk_us`.
- `e2e` — full `layer.forward` (validation + staging quant + kernel +
  output copy).  The serving-relevant number; no drop counterpart.

Reported per point: min/max/mean/median across ranks of the per-rank
means; `critical_tflops_*` uses the max-rank time.

**Comparison rule** (inherited from the SM100 lessons): match the problem
shape, the data recipe, the routing, AND the timed region before comparing
— note in particular that FI `compute` includes the standalone TopkReduce
tail and the host-launch gap inside its CUDA-event window, while the
drop's `*_mega_us` columns are profiler-extracted kernel time only.

## Next levers

1. **Fused staging kernel** — port the SM100 tree's single-launch
   quant+repack (`shim/quant_stage.py` / `FLASHINFER_MEGA_FUSED_STAGE`)
   to the SM90 fp8 staging path; it is the bulk of the e2e overhead.
2. **Heuristic re-calibration** — the token-bucket table was derived with
   a mega-only, no-post-warmup-alignment metric; under a metric that
   includes the top-k reduce and aligns ranks after warmup, the per-bucket
   winners may shift (especially the small-token CGA choices).  Blockwise
   non-swap 512-32768 already moved to cooperative M64N256 on 2026-09-02
   (fold layout); re-derive the rest when the kernel team's next sweep
   lands.
3. **Old-layout 2-WG pathology (blockwise non-swap)** — with the producer
   warpgroup present, both 2-WG modes are crippled at 512-32768: ping-pong
   runs 24% behind basic and cooperative 13.6% behind basic, while under
   the fold the SAME configs (identical registers, identical publication
   path) are ~43% faster and win.  128 idle threads cannot explain that;
   something in the old 16-warp layout (a barrier the extra warps
   participate in, or SM scheduler / register-file pressure) hurts the
   2-WG epilogue specifically.  Worth a PIC/IKET trace before anyone
   re-enables the old layout.
4. **Swap-AB N=8** — DONE for the rows where it pays (2026-09-10).  The tile
   is legal and bit-exact in all four modes (blockwise cooperative M256N8
   fixed 2026-09-10: the token-scale TMA box is loaded non-multicast when the
   multicast sub-box would be < 128 B; `compute-sanitizer` with
   `CUTE_DSL_LINEINFO=1` pinned the fault to the sf `cute.copy`, 0 errors
   after the fix).  Same-node interleaved A/B on 4x H200 at 1830 MHz
   (`--swap-token-tile 8` vs the table, 2 rounds, e2e median) over every
   swap bucket ≤ 256: pt64 basic M128N8 **+13.5%** (16 tokens/expert fill
   two N=8 tiles instead of one N=64 tile that is 3/4 padding) and pt128
   ping-pong M128N8 +4.6% → switched; pt16 ping-pong M128N8 +1.8% and bw64
   cooperative M256N8 +1.7% (+2.9% in the table-vs-table confirmation)
   measured positive but were judged too small to change (kept at N=16 /
   N=32); pt8 / bw8 / bw16 / bw32 are within the ±1% run noise
   (+0.2..+1.2%, left at N=16) and bw128 / bw256 / pt256 lose 15-42%
   (every N≥16 tile is already full there, so N=8 only doubles the tile
   count).  Old-table vs new-table confirmation on the same node (2
   rounds): pt64 +14.1%, pt128 +4.7% e2e; all unchanged buckets within
   ±1%.  pt32 was the one
   non-swap small bucket (`--swap-token-tile` does not touch it); tested
   separately in manual mode (`--swap-ab --mma-tiler 256,8 --cga 2,1`,
   2 rounds): cooperative swap M256N8 CGA2x1 beats non-swap M64N256 by
   +5.7% / +6.4% e2e, the ping-pong swap M128N8 CGA1x2 twin ties it → pt32
   switched to the cooperative swap row.  Remaining idea: an intermediate N
   (e.g. N=16 for pt128 instead of N=32/N=8) was not swept.
   Sanitizer caveat (not N=8 specific, reproduced at N=16):
   with `load_balance_mode=atomic_counter` and a 1x1 cluster the scheduler's
   DSMEM fan-out (`store_i32_to_peer_cluster_smem_async`, lane 0 writing its
   own CTA through `st.async.shared::cluster`) is flagged by compute-sanitizer
   as "Invalid __shared__ write / Cluster needs to have at least 2 blocks"
   (528 reports per bench run) even though the run is bit-exact; sanitize
   with `--load-balance-mode static` or a cluster of >= 2 CTAs.
5. **Old-layout + store-offload intermittent hang** — `--no-fold-producer-warps`
   (offload active) hung once at blockwise 4096 (reuse token-back) after a
   clean pass of the identical config minutes earlier (2026-09-02 07:04).
   The path is off by default now but still reachable with
   `active_dispatch_warps=2/4`; needs a deadlock probe before that knob is
   recommended.
6. **DSL runtime A/B** — rerun one column on `nvidia-cutlass-dsl>=4.6.1`
   to check whether the SM100 perf-floor finding transfers to SM90.
7. **CUDA-graph capture** — the SM100 mega layer's warmup+capture path is
   kernel-agnostic; validate it on sm90_fp8_fp8_bf16_pull_cutedsl (`test_mega_cuda_graph`
   analog) for decode serving.
