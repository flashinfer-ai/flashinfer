# SM90 pull-style FP8 MegaMoE tuning + performance notes

This document collects the performance work on the `sm90_fp8_fp8_bf16_pull_cutedsl` mega
backend: the measured microbenchmark results, the benchmark methodology
behind those numbers, the knob surface as it exists today, and the open
perf levers.  It is the companion to `SKILL.md` (drop-update workflow) and
mirrors the structure of the SM100 tree's `TUNING.md`.

Unless noted otherwise, every measurement comes from a single H200 node
(4x NVIDIA H200 141GB, SM clock locked at 1830 MHz, EP=4) taken in one
session: the two microbenchmark tables below and the companion
`benchmarks/pull_and_push_comparison.md` are one 2026-09-12 session
(current table: compact pull buffer default, 16-byte generate_c stores,
swap-AB N=8 rows; sweep and pull/push comparison back to back on the same
node), while the same-node before/after paragraphs are the 2026-09-12
(compact pull buffer) and 2026-09-03 (fold layout) sessions, each on its
own node (never compare numbers across sessions; node-to-node offsets of
~2% are documented in "Next levers") — at the
kernel drop's DSV4-Pro P03
geometry: **384 experts, top-6, hidden 7168, intermediate 3072
(post-SwiGLU; gate+up 6144), gate_up_clamp 10.0**, tokens-per-rank swept
8..32768 in powers of two (13 points) — the same geometry and knobs as the
kernel team's `moe_hopper_fp8/run_token_sweep_benchmark.py`.  The launch
config per point is the drop's token-bucket heuristic table
(`moe_hopper_fp8/heuristic_config.py`, geometry derived from the kernel
team's 2026-08-19 four-rank H200 sweep at the same vendored kernel
sources, plus the locally added per-bucket `token_back_mode` column from
the 2026-08-23 epi-vs-reuse sweep — see the knob list below).
Raw rows: `benchmark_data/20260912/20260912_025139_mega_sm90_heuristic_both.csv`
(local archive, not committed).

## Microbenchmark results (2026-09-12, heuristic launch configs, max-rank µs)

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

The `token back` column is the per-bucket `token_back_mode` the heuristic
table now selects (`epi` = `epi_warps`, `reuse` = `reuse_dispatch_warps`).
All other knobs are at their config defaults — notably
`active_dispatch_warps=1` (see "The knob surface"), which lifts the
large-token buckets by up to ~7% over the previous 4-warp fixed layout.

**per_tensor** — peak 971 TFLOPS/rank:

| tok/rank | heuristic config                   | token back | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|:----------:|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB M256N16 CGA2x1             |    epi     |      759.9 |    8.3 |    881.1 |        7.0 |
|       16 | swap-AB ping-pong M128N16 CGA1x2   |    epi     |     1220.5 |   10.4 |   1347.1 |        9.2 |
|       32 | swap-AB M256N8 CGA2x1              |    epi     |     1371.6 |   18.5 |   1488.3 |       16.6 |
|       64 | swap-AB M128N8 CGA1x2              |    epi     |     1652.7 |   30.7 |   1770.5 |       28.0 |
|      128 | swap-AB ping-pong M128N8 CGA1x2    |    epi     |     1746.8 |   58.0 |   1879.3 |       53.0 |
|      256 | swap-AB M256N32 CGA2x1             |    epi     |     1646.2 |  123.1 |   1783.1 |      111.7 |
|      512 | swap-AB M256N64 CGA1x1             |    epi     |     1728.8 |  234.6 |   1859.1 |      214.3 |
|     1024 | swap-AB ping-pong M128N64 CGA1x2   |    epi     |     2083.3 |  384.9 |   2190.3 |      363.2 |
|     2048 | non-swap ping-pong M64N128 CGA2x1  |    epi     |     2942.5 |  549.7 |   3065.7 |      528.2 |
|     4096 | non-swap ping-pong M64N128 CGA2x2  |    epi     |     4860.6 |  670.1 |   5041.6 |      643.1 |
|     8192 | swap-AB ping-pong M128N64 CGA1x2   |    epi     |     8317.2 |  788.5 |   8410.8 |      775.6 |
|    16384 | non-swap M64N256 CGA2x1            |   reuse    |    13401.7 |  971.0 |  14227.3 |      916.5 |
|    32768 | non-swap ping-pong M64N128 CGA2x2  |   reuse    |    27559.7 |  941.7 |  29199.6 |      888.4 |

**blockwise** — peak 848 TFLOPS/rank:

| tok/rank | heuristic config                   | token back | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|:----------:|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB M256N16 CGA2x1             |    epi     |      778.0 |    8.0 |    986.7 |        6.2 |
|       16 | swap-AB M256N16 CGA1x1             |    epi     |     1123.2 |   11.3 |   1326.0 |        9.3 |
|       32 | swap-AB ping-pong M128N16 CGA1x2   |    epi     |     1642.8 |   15.4 |   1860.2 |       13.4 |
|       64 | swap-AB M256N32 CGA2x1             |    epi     |     1587.5 |   31.9 |   1803.6 |       27.2 |
|      128 | swap-AB M256N16 CGA2x1             |    epi     |     1594.5 |   63.5 |   1821.7 |       54.7 |
|      256 | swap-AB ping-pong M128N32 CGA1x2   |    epi     |     2037.9 |   99.5 |   2257.5 |       88.3 |
|      512 | non-swap M64N256 CGA1x1            |    epi     |     1735.7 |  233.2 |   1960.4 |      206.7 |
|     1024 | non-swap M64N256 CGA2x2            |   reuse    |     2418.3 |  327.2 |   2593.3 |      312.1 |
|     2048 | non-swap M64N256 CGA2x2            |   reuse    |     3256.2 |  488.7 |   3482.8 |      465.5 |
|     4096 | non-swap M64N256 CGA1x1            |   reuse    |     5439.4 |  594.3 |   5854.0 |      554.9 |
|     8192 | non-swap M64N256 CGA2x1            |   reuse    |     8863.3 |  735.0 |   9565.4 |      684.6 |
|    16384 | non-swap M64N256 CGA1x2            |   reuse    |    17226.9 |  754.9 |  18605.7 |      699.5 |
|    32768 | non-swap M64N256 CGA2x1            |   reuse    |    30579.5 |  847.7 |  33548.0 |      773.6 |

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

`e2e` minus `compute` is ~140-280 µs at small token counts growing to
~1.6-2.9 ms at 32768 — dominated by the torch-composed staging quant plus
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
winner plus every geometry that wins some bucket of the table (16 today,
derived programmatically) crossed with both validated token-back modes —
32 candidates.

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
  per_tensor 32/64/128, see "Next levers" item 4);
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
  follows the per-token-bucket heuristic table (epi_warps small/mid
  buckets, reuse_dispatch_warps at the GEMM-bound tail — per_tensor
  >= 16384, blockwise >= 1024; 2026-08-23 four-rank H200 sweep) and is a
  tuner candidate axis.  All six token_back x reduce combinations are
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
  STG.16).  The GMEM pointer carries an explicit 16-byte alignment
  assumption (`make_ptr(..., assumed_align=16)`): a plain `autovec_copy`
  into the BF16 view has no alignment fact and degrades to 16-bit stores
  — the first version's "bf16x2" pair stores compiled to `STG.E.U16`,
  which was the whole generate_c overhead.  The row/column predicate is
  exact (`intermediate_gateup % 64 == 0`).  No SMEM staging: a TMA-staged
  variant (two (64x64) BF16 stages per epilogue warpgroup) was measured
  and rejected — the 16 KB per warpgroup comes out of the AB pipeline
  budget and drops the N256 configs from 4 to 3 stages (−10..−38% on
  those buckets) while gaining only +2..+3% where it fits.
  Cost when on (FlashInfer bench, 4x H200 at 1830 MHz, 2 interleaved
  rounds, e2e median): every bucket within −1..−3% of the same tree with
  generate_c off (geomean e2e −0.9%, compute −1.2%; was −5.2% / −5.4%
  with the 16-bit stores: non-swap buckets −8..−16%).  Off costs nothing:
  the store path is compiled out (off-vs-off sweeps within noise).  Test
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
- `dedup_dispatch`, `grouped_token_back`, `combine_format` — top-k dedup
  on dispatch / combine and the quantized combine wire; see
  `dedup_topk_design.md`.
- `fp8_accum_mode`, `kind` (e4m3/e5m2), clamps.

## Sweep methodology + environment (reproduce recipe)

**Hardware / software.**  One H200 node, 4x NVIDIA H200 141GB (sm_90,
cc 9.0) over NVLink, SM clock locked at 1830 MHz on all four GPUs
(`nvidia-smi --query-gpu=clocks.sm,clocks.max.sm` reports 1830 / 1980 MHz;
every sweep script records it to `clocks.txt` before timing, and no
row in this document was taken unlocked).  Python 3.12, torch `2.12.0+cu130`,
`nvshmem4py-cu13`, **`nvidia-cutlass-dsl 4.6.0`** (the drop pins
`4.5.0dev0`; 4.6.0 compiles and runs this SM90 tree).  Whether the SM100
tree's ">=4.6.1 perf floor" finding applies to the SM90 kernels is
UNTESTED — worth one A/B run.

**Harness.**  `benchmarks/bench_moe_ep_sm90_mega.py`, one torchrun process
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
