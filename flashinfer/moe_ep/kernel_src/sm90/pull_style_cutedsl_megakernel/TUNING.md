# SM90 pull-style FP8 MegaMoE tuning + performance notes

This document collects the performance work on the `sm90_fp8_fp8_bf16_pull_cutedsl` mega
backend: the measured microbenchmark results, the benchmark methodology
behind those numbers, the knob surface as it exists today, and the open
perf levers.  It is the companion to `SKILL.md` (drop-update workflow) and
mirrors the structure of the SM100 tree's `TUNING.md`.

Unless noted otherwise, all measurements were taken 2026-09-03 on a single
H200 node (4x NVIDIA H200 141GB, SM clock locked at 1830 MHz, EP=4) in one
session — the two microbenchmark tables, the same-node before/after
paragraph and the companion `benchmarks/pull_and_push_comparison.md` all
come from that one node, back to back — at the
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
Raw rows: `benchmark_data/20260903/20260903_002209_mega_sm90_heuristic_both.csv`.

## Microbenchmark results (2026-09-03, heuristic launch configs, max-rank µs)

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

**per_tensor** — peak 902 TFLOPS/rank:

| tok/rank | heuristic config                   | token back | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|:----------:|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB M256N16 CGA2x1             |    epi     |      786.1 |    8.1 |    944.7 |        6.7 |
|       16 | swap-AB ping-pong M128N16 CGA1x2   |    epi     |     1242.1 |   10.2 |   1409.7 |        9.0 |
|       32 | non-swap M64N256 CGA1x1            |    epi     |     1553.2 |   16.3 |   1646.7 |       15.4 |
|       64 | swap-AB M128N64 CGA1x2             |    epi     |     1930.7 |   26.3 |   2104.8 |       24.1 |
|      128 | swap-AB ping-pong M128N32 CGA1x2   |    epi     |     1833.8 |   55.3 |   2010.2 |       50.5 |
|      256 | swap-AB M256N32 CGA2x1             |    epi     |     1694.9 |  119.7 |   1877.6 |      108.1 |
|      512 | swap-AB M256N64 CGA1x1             |    epi     |     1823.2 |  222.6 |   2010.2 |      201.9 |
|     1024 | swap-AB ping-pong M128N64 CGA1x2   |    epi     |     2158.6 |  376.1 |   2317.8 |      350.2 |
|     2048 | non-swap ping-pong M64N128 CGA2x1  |    epi     |     3062.1 |  530.2 |   3206.2 |      506.4 |
|     4096 | non-swap ping-pong M64N128 CGA2x2  |    epi     |     5076.3 |  639.6 |   5269.9 |      616.1 |
|     8192 | swap-AB ping-pong M128N64 CGA1x2   |    epi     |     8535.1 |  760.9 |   8780.3 |      739.6 |
|    16384 | non-swap M64N256 CGA2x1            |   reuse    |    14393.4 |  902.4 |  15194.1 |      854.8 |
|    32768 | non-swap ping-pong M64N128 CGA2x2  |   reuse    |    28978.9 |  896.4 |  30673.7 |      846.9 |

**blockwise** — peak 830 TFLOPS/rank:

| tok/rank | heuristic config                   | token back | compute µs | TFLOPS | e2e µs   | e2e TFLOPS |
|---------:|------------------------------------|:----------:|-----------:|-------:|---------:|-----------:|
|        8 | swap-AB M256N16 CGA2x1             |    epi     |      799.2 |    7.9 |   1063.9 |        6.0 |
|       16 | swap-AB M256N16 CGA1x1             |    epi     |     1162.0 |   10.9 |   1426.2 |        8.9 |
|       32 | swap-AB ping-pong M128N16 CGA1x2   |    epi     |     1652.7 |   15.3 |   1929.9 |       13.1 |
|       64 | swap-AB M256N32 CGA2x1             |    epi     |     1659.1 |   30.6 |   1977.2 |       25.7 |
|      128 | swap-AB M256N16 CGA2x1             |    epi     |     1670.2 |   60.8 |   1973.2 |       51.4 |
|      256 | swap-AB ping-pong M128N32 CGA1x2   |    epi     |     2057.2 |   98.7 |   2337.5 |       86.8 |
|      512 | non-swap M64N256 CGA1x1            |    epi     |     1789.6 |  226.8 |   2029.8 |      199.9 |
|     1024 | non-swap M64N256 CGA2x2            |   reuse    |     2521.2 |  322.0 |   2689.4 |      301.8 |
|     2048 | non-swap M64N256 CGA2x2            |   reuse    |     3387.8 |  479.2 |   3538.9 |      458.8 |
|     4096 | non-swap M64N256 CGA1x1            |   reuse    |     5606.4 |  579.1 |   5971.9 |      543.7 |
|     8192 | non-swap M64N256 CGA2x1            |   reuse    |     8979.3 |  723.2 |   9691.9 |      670.0 |
|    16384 | non-swap M64N256 CGA1x2            |   reuse    |    17565.9 |  739.4 |  18981.2 |      684.3 |
|    32768 | non-swap M64N256 CGA2x1            |   reuse    |    31304.9 |  829.8 |  34137.6 |      760.9 |

**Before/after on the same node** (the table above is the new default:
`fold_producer_warps` + the re-calibrated table — blockwise non-swap
512-32768 cooperative M64N256, per_tensor 8 cooperative, per_tensor 64
basic; the pre-fold default — producer warpgroup + FC1 store offload + the
2026-08-19 table — re-run on the same node the same hour, compute TFLOPS):
per_tensor geomean **+0.76%** (pt8 +5.4, pt64 +2.7, pt512 +2.7; worst
pt16384 −1.5), blockwise geomean **+12.20%** (bw512 +12.7, bw1024 +13.2,
bw2048 +22.3, bw4096 +16.7, bw8192 +29.8, bw16384 +27.4, bw32768 +44.5;
the swap buckets 8-256 are within ±3%).

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

`e2e` minus `compute` is ~150-280 µs at small token counts growing to
~1.7-3.0 ms at 32768 — dominated by the torch-composed staging quant plus
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
  M128N64, see `fold_producer_warps` below); setting any one switches to
  manual mode with
  drop-driver defaults for the rest (non-swap (64, 128, 128), swap-AB
  (256, 32, 128), (128, 32, 128) with ping-pong; cluster (1, 1, 1)).
  Kernel-legal geometry: non-swap M∈{64}, N∈{128,256}; swap-AB M∈{128,256},
  N∈{16,32,64,128}; K=128; CGA (m,n)∈{(1,1),(2,1),(1,2),(2,2)}, k=1.
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
4. **Old-layout + store-offload intermittent hang** — `--no-fold-producer-warps`
   (offload active) hung once at blockwise 4096 (reuse token-back) after a
   clean pass of the identical config minutes earlier (2026-09-02 07:04).
   The path is off by default now but still reachable with
   `active_dispatch_warps=2/4`; needs a deadlock probe before that knob is
   recommended.
5. **DSL runtime A/B** — rerun one column on `nvidia-cutlass-dsl>=4.6.1`
   to check whether the SM100 perf-floor finding transfers to SM90.
6. **CUDA-graph capture** — the SM100 mega layer's warmup+capture path is
   kernel-agnostic; validate it on sm90_fp8_fp8_bf16_pull_cutedsl (`test_mega_cuda_graph`
   analog) for decode serving.
