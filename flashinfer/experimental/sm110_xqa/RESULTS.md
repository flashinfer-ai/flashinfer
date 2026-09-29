# Thor SM110 validation

Native CUDA/TVM-FFI integration was validated on NVIDIA Thor (20 SMs, compute capability 11.0), with CUDA 13.4.59, Python 3.12.3 and PyTorch 2.14.0a0+b2c75dd062.nv26.09. Native JIT compilation passed for all seven frozen routes.

The pre-integration frozen schedule and native export use identical inputs and physical selections. Each row uses six alternating pairs, 250 ms warmup per round, 256 CUPTI samples, cold L2, no timer fallback, and the complete recorded kernel chain. Values below are medians of round medians. The acceptance rule is `abs(export / schedule - 1) <= 0.03` for every row.

| Shape | Frozen schedule (us) | Native export (us) | Absolute difference |
| --- | ---: | ---: | ---: |
| tree_fp16_contiguous | 61.6080 | 60.9285 | 1.103% |
| tree_fp16_paged | 63.4085 | 62.5523 | 1.350% |
| tree_fp8_contiguous | 59.2160 | 60.8087 | 2.690% |
| tree_fp8_paged | 62.0800 | 62.9605 | 1.418% |
| decode_b1_c1024 | 26.2882 | 26.3923 | 0.396% |
| decode_b2_c512 | 26.3203 | 26.2565 | 0.242% |
| decode_b4_c256 | 24.7680 | 24.8160 | 0.194% |

**All 7 rows pass; maximum difference is 2.690%.** The input dimensions and precision are recorded in `benchmarks/sm110_xqa_shapes.json`. This measures integration fidelity; it does not establish parity with other XQA implementations.

Correctness: 35 JIT metadata cases and 54 GPU cases passed. GPU coverage includes reference output checks, ragged lengths, packed and paged tree input, repeated replay, current-stream dependencies, graph capture, workspace ownership and alignment. Output comparisons use `atol=rtol=0.01`. The paired comparison additionally checks correctness before and after timing, input immutability and decode counter reset.

Synchronization: all seven rows passed separate Compute Sanitizer `synccheck` and `racecheck` invocations (14 passes, no timeouts). Each invocation had a process-tree wall-clock limit of 20 seconds. Per-row outcomes are in `validation.json`.

Physical turnaround: native build plus tests 92.43 s (37.81 s compiling seven routes); native sanitizers 137.39 s; paired performance acceptance 298.60 s including queue time, with 153.01 s in the measurement payload. GPU kernel runtime is shown separately in the table.

Physical selections: tree M64/N256, eight copy warps; FP8 raw staging/prefetch; prefix fastpath; paged address hoisting; FP16 K prefetch and V/QK overlap. Decode uses 256-token partitions, fused last-CTA merge, half-warp merge and cached statistics when P>1; P=1 dispatches to the specialization without cached statistics. All measured rows launch one kernel per replay.

The source manifest is immutable and records status at generation time. Execution evidence is kept here and in `validation.json`. The decorated public entry point passes all 89 tests (35 metadata and 54 GPU) in 11.76 s, and the public CUPTI benchmark completes all seven rows. Ruff 0.12.8 formatting/lint passes; formatting preserves the Python AST. Frozen device-source bytes and manifest remain unchanged.

An independently installed wheel contains every manifest-listed CUDA file, native headers and the thin public API. Fresh native JIT and two replays each pass for D128 P=1/P=4 and paged FP8 tree attention with the source checkout absent from Python's import path. Packaging/install/native smoke takes 83.37 s (39.55/5.17/26.35 s for those phases); the full final-entry-point/benchmark/package step takes 121.04 s.

This is a scoped sparse-source packaging smoke in the existing development environment. It uses TVM-FFI `0.1.dev1+gd1fd51222`, which is outside the declared `>=0.1.11,<0.2` dependency range, with dependency resolution disabled. It does not establish a dependency-resolved installation or complete release-wheel readiness.

## Register-MMA D512 tree routes (`kernel="register_mma"`)

The four `tree_*_mma` routes wrap the register `mma.sync` XQA schedule (one 32-row Q tile per CTA over all 512 output columns, eight QK warps and eight PV warps, no tensor memory) behind the same public API. Validation ran on the same NVIDIA Thor class of machine (20 SMs, compute capability 11.0), node `sr250v3-0681`, CUDA 13.4, PyTorch 2.15.0a0+875d815502.nvinternal.main, cupti-python 13.4.0, with the frozen sources compiled by FlashInfer's native JIT.

Artifact parity uses the frozen schedule's own launcher and the exported TVM-FFI entry on identical tensors in one process: six alternating paired rounds (round order alternates schedule-first / export-first), 250 ms warmup and 256 CUPTI cold-L2 samples per round, no timer fallback, correctness against the FP32 oracle before and after timing. Values are medians of round medians; the acceptance rule is `abs(export / schedule - 1) <= 0.03` per row.

| Shape | Frozen schedule (us) | Native export (us) | Absolute difference | Cake-first | Export-first |
| --- | ---: | ---: | ---: | ---: | ---: |
| tree_fp16_contiguous_mma | 29.5198 | 29.6645 | 0.490% | 1.0033 | 1.0071 |
| tree_fp16_paged_mma | 34.5525 | 34.8003 | 0.717% | 1.0055 | 1.0088 |
| tree_fp8_contiguous_mma | 23.9682 | 23.9520 | 0.068% | 1.0000 | 0.9986 |
| tree_fp8_paged_mma | 25.7523 | 25.6400 | 0.436% | 0.9975 | 0.9950 |

**All 4 register-MMA rows pass; maximum difference is 0.717%.**

Public ledger benchmark (`benchmarks/bench_sm110_xqa.py`, 250.0 ms warmup, 256 CUPTI cold-L2 samples per row, one process for all 11 rows). The register-MMA rows are listed next to the tcgen05 rows of the same shape measured in the same process; the ratio is the tcgen05 latency over the register-MMA latency and describes these two frozen schedules only.

| Ledger row | Kernel family | Native latency (us) | Ratio to the tcgen05 row |
| --- | --- | ---: | ---: |
| tree_fp16_contiguous | tcgen05 | 69.216 | 1 (reference row) |
| tree_fp16_paged | tcgen05 | 73.152 | 1 (reference row) |
| tree_fp8_contiguous | tcgen05 | 61.361 | 1 (reference row) |
| tree_fp8_paged | tcgen05 | 61.008 | 1 (reference row) |
| tree_fp16_contiguous_mma | register_mma | 31.585 | 2.191x faster |
| tree_fp16_paged_mma | register_mma | 38.112 | 1.919x faster |
| tree_fp8_contiguous_mma | register_mma | 24.000 | 2.557x faster |
| tree_fp8_paged_mma | register_mma | 26.080 | 2.339x faster |

Correctness: JIT metadata suite 35 passed, 19 warnings in 1.24s; GPU suite 89 passed, 21 warnings in 61.15s (0:01:01) (the register-MMA cases reuse the tcgen05 numerical ledger, packed and paged inputs, replay and rejection checks at `atol=rtol=0.01`).

Synchronization: separate Compute Sanitizer `synccheck` and `racecheck` invocations per route, each under a hard 20 s process-tree wall-clock limit (a timeout would be recorded as skipped, never retried; `memcheck` is not run). Every `synccheck` run reports 0 errors. `racecheck` completes on every route with 0 errors and reports warning-level hazards (fp16_contiguous 144, fp16_paged 125, fp8_contiguous 4, fp8_paged 12); the flagged sites are the schedule's mbarrier-ordered shared-memory handoffs (`racecheck` does not model `mbarrier` waits), see `validation.json` for the per-route opcode summary.

| Route | Tool | Verdict | Wall | Summary |
| --- | --- | --- | ---: | --- |
| fp16_contiguous | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| fp16_contiguous | racecheck | pass_with_warnings | 19 s | ========= RACECHECK SUMMARY: 100 hazards displayed (0 errors, 144 warnings) |
| fp16_paged | synccheck | pass | 7 s | ========= ERROR SUMMARY: 0 errors |
| fp16_paged | racecheck | pass_with_warnings | 19 s | ========= RACECHECK SUMMARY: 100 hazards displayed (0 errors, 125 warnings) |
| fp8_contiguous | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| fp8_contiguous | racecheck | pass_with_warnings | 12 s | ========= RACECHECK SUMMARY: 4 hazards displayed (0 errors, 4 warnings) |
| fp8_paged | synccheck | pass | 7 s | ========= ERROR SUMMARY: 0 errors |
| fp8_paged | racecheck | pass_with_warnings | 13 s | ========= RACECHECK SUMMARY: 12 hazards displayed (0 errors, 12 warnings) |

Physical turnaround (managed step seconds on the Thor node): patch apply 2, hook tooling and environment 8, upstream pre-commit hooks 5, JIT metadata tests 9, GPU suite 69, ledger benchmark 12 (5.4 s payload), artifact parity 127 (124.8 s payload), sanitizers 13. GPU kernel runtime is shown separately in the tables.

## Split register-MMA FP16 page128 route (`kernel="register_mma_split"`, `kernel="register_mma_auto"`)

`tree_fp16_paged_mma_split` wraps the same register `mma.sync` XQA schedule split over the two halves of the KV sequence: two eight-warp groups per CTA (512 threads, one 32-row Q tile over all 512 output columns, grid unchanged), each with its own K/V staging ring, merge their unnormalised partials and row statistics in shared memory before one group writes the output. It is frozen for FP16 page128 KV, the one D512 cache mode where it is faster than the `register_mma` route on both validated Thor nodes (1.06-1.08x cold-L2, 1.04-1.05x warm in the schedule harness); `register_mma_auto` selects it there and `register_mma` for the other three cache modes. Validation ran on NVIDIA Thor `sr250v3-0677` (20 SMs, CUDA 13.4, PyTorch 2.15.0a0+875d815502.nvinternal.main) in one container session.

Artifact parity (frozen split schedule launcher vs the exported TVM-FFI entry, identical tensors, one process, six alternating paired rounds, 250 ms warmup, 256 CUPTI cold-L2 samples per round, gate 3%):

| Shape | Frozen schedule (us) | Native export (us) | Absolute difference | Cake-first | Export-first |
| --- | ---: | ---: | ---: | ---: | ---: |
| tree_fp16_paged_mma_split | 34.1600 | 34.3285 | 0.493% | 1.0009 | 1.0070 |

**The split row passes (difference 0.493%).**

Public ledger benchmark (`benchmarks/bench_sm110_xqa.py`, one process for all 12 rows, same protocol as above):

| Ledger row | Kernel family | Native latency (us) | Ratio |
| --- | --- | ---: | ---: |
| tree_fp16_paged | tcgen05 | 70.928 | 1 (reference row) |
| tree_fp16_paged_mma | register_mma | 44.928 | 1.579x faster than tcgen05 |
| tree_fp16_paged_mma_split | register_mma_split | 40.897 | 1.734x faster than tcgen05, 1.099x faster than register_mma |

Correctness: JIT metadata suite 35 passed; GPU suite 109 passed (adds the FP16 page128 cache cases for `register_mma_split` and `register_mma_auto`, the auto fallback to `register_mma` on the other cache modes and the rejection of `register_mma_split` off its frozen mode). pre-commit clean.

Synchronization (same protocol, hard 20 s limit, `memcheck` not run):

| Route | Tool | Verdict | Wall | Summary |
| --- | --- | --- | ---: | --- |
| fp16_paged (split) | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| fp16_paged (split) | racecheck | pass_with_warnings | 19 s | ========= RACECHECK SUMMARY: 53 hazards displayed (0 errors, 53 warnings) |

The racecheck warnings map to the same instruction classes as the `register_mma` routes (`LDGSTS.E.BYPASS.128` staging writes and `LDSM.16.MT88.4` fragment reads): asynchronous `cp.async` staging writes and `ldmatrix` reads of the K/V rings that are ordered by `cp.async.mbarrier.arrive` / `cp.async.wait_group` and mbarrier phases the tool does not model; no error-class hazard.

## tcgen05/TMEM D512 tree routes (`kernel="tmem"`, `kernel="auto"`)

The four `tree_*_tmem` routes wrap the tcgen05 tensor-memory schedule of the D512 tree kernel: one 128-row Q tile x one 256-column output half per 512-thread CTA (grid `(2, heads * ceil(q_len * ratio / 128), batch)`, 230400 B dynamic shared memory), K/V and Q fetched once per thread-block cluster by TMA multicast. Each route ships two compiled forms: the `(2, 2, 1)` cluster form pairs the two Q tiles of one KV head (K/V + Q multicast) and the `(2, 1, 1)` form (`*_q_kernel.cu`) multicasts Q only for heads with an odd Q-tile count; the binding selects the form from the Q-tile parity. Under the strict cold-L2 protocol the tmem route is faster than `register_mma_auto` and than the upstream Edge XQA source on every D512 cache mode on every validated Thor node, so `kernel="auto"` selected it for all four modes (round 9 below: for E4M3 KV whose heads have an even Q-tile count `auto` now selects the pair route). Validation ran on NVIDIA Thor `sr250v3-0666` (20 SMs, CUDA 13.4, PyTorch 2.15.0a0+875d815502.nvinternal.main) in one container session.

Artifact parity (frozen schedule launcher vs the exported TVM-FFI entry, identical tensors, one process, six alternating paired rounds, 250 ms warmup, 256 CUPTI cold-L2 samples per round, gate 3%):

| Shape | Frozen schedule (us) | Native export (us) | Absolute difference | Cake-first | Export-first |
| --- | ---: | ---: | ---: | ---: | ---: |
| tree_fp16_contiguous_tmem | 30.5760 | 30.3680 | 0.680% | 0.9937 | 0.9948 |
| tree_fp16_paged_tmem | 30.6400 | 30.7680 | 0.418% | 1.0010 | 1.0063 |
| tree_fp8_contiguous_tmem | 23.7518 | 23.7280 | 0.100% | 1.0000 | 0.9993 |
| tree_fp8_paged_tmem | 24.0000 | 23.8558 | 0.601% | 0.9947 | 0.9933 |

**All 4 tmem rows pass; maximum difference is 0.680%.**

Public ledger benchmark (`benchmarks/bench_sm110_xqa.py`, one process for all 16 rows, same protocol as above):

| Ledger row | Kernel family | Native latency (us) | Ratio |
| --- | --- | ---: | ---: |
| tree_fp16_contiguous | tcgen05 | 69.680 | 1 (reference row) |
| tree_fp16_contiguous_mma | register_mma | 31.408 | 2.219x faster than tcgen05 |
| tree_fp16_contiguous_tmem | tmem | 31.168 | 2.236x faster than tcgen05, 1.008x faster than the previous `auto` row |
| tree_fp16_paged | tcgen05 | 73.216 | 1 (reference row) |
| tree_fp16_paged_mma_split | register_mma_split | 34.687 | 2.111x faster than tcgen05 |
| tree_fp16_paged_tmem | tmem | 29.712 | 2.464x faster than tcgen05, 1.167x faster than the previous `auto` row |
| tree_fp8_contiguous | tcgen05 | 61.088 | 1 (reference row) |
| tree_fp8_contiguous_mma | register_mma | 24.544 | 2.489x faster than tcgen05 |
| tree_fp8_contiguous_tmem | tmem | 23.887 | 2.557x faster than tcgen05, 1.027x faster than the previous `auto` row |
| tree_fp8_paged | tcgen05 | 60.112 | 1 (reference row) |
| tree_fp8_paged_mma | register_mma | 25.760 | 2.334x faster than tcgen05 |
| tree_fp8_paged_tmem | tmem | 24.288 | 2.475x faster than tcgen05, 1.061x faster than the previous `auto` row |

Correctness: JIT metadata suite 35 passed; GPU suite 177 passed (adds the tmem cache cases on all four modes with even and odd Q-tile counts, i.e. both cluster forms, `auto` routing to the tmem routes and the JIT fixture for the new family). pre-commit clean (ruff-format rewrote one blank line in test_sm110_xqa.py on the first run; the formatted file is the one validated and committed; all other hooks passed).

Synchronization (same protocol, hard 20 s limit, `memcheck` not run):

| Route | Tool | Verdict | Wall | Summary |
| --- | --- | --- | ---: | --- |
| fp16_contiguous (tmem) | synccheck | pass | 10 s | ========= ERROR SUMMARY: 0 errors |
| fp16_contiguous (tmem) | racecheck | pass | 8 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| fp16_paged (tmem) | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| fp16_paged (tmem) | racecheck | pass | 9 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| fp8_contiguous (tmem) | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| fp8_contiguous (tmem) | racecheck | pass | 9 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| fp8_paged (tmem) | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| fp8_paged (tmem) | racecheck | pass | 10 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |

Both tools completed within the 20 s limit on the exported routes (8-10 s each) with no hazards. The same kernel under the frozen-schedule benchmark harness reached the 20 s racecheck deadline in every earlier run (a different launcher and process setup); that difference was not investigated.

## Round 9: per-GQA-ratio tmem kernels and the cta_group::2 pair routes (`kernel="pair"`, `kernel="auto"`)

Each `tree_*_tmem` route now ships one frozen trace per GQA ratio (2, 4, 8 and 16) and cluster form; the binding selects the kernel from the ratio and the Q-tile parity and encodes Q with the ratio's box (`(64, ratio, 128 / ratio)`), lifting the ratio-8 freeze of the previous round. The two new `tree_fp8_{contiguous,paged}_pair` routes run the same tcgen05/TMEM schedule as two `cta_group::2` CTA pairs in a `(4, 1, 1)` cluster per two 128-row Q tiles of a KV head (grid `(4, heads * ceil(q_len * ratio / 128) / 2, batch)`, 230400 B dynamic shared memory, one kernel per GQA ratio): the pair leader issues one M256 MMA stream for both Q tiles, each CTA holds one token half of every K chunk and one column half of every V chunk, and the raw E4M3 V rows are multicast to the pair. Under the strict cold-L2 protocol the pair route is faster than the tmem route on both E4M3 cache modes on every validated Thor node (1.034-1.071 per process, pooled 1.037-1.049), so `kernel="auto"` selects it for E4M3 KV whose heads have an even Q-tile count and `tmem` otherwise. Validation ran on NVIDIA Thor `sr250v3-0667` (20 SMs, CUDA 13.4, PyTorch 2.15.0a0+875d815502.nvinternal.main) in one container session.

Artifact parity (frozen schedule launcher vs the exported TVM-FFI entry, identical tensors, one process, six alternating paired rounds, 250 ms warmup, 256 CUPTI cold-L2 samples per round, gate 3%):

| Shape | Frozen schedule (us) | Native export (us) | Absolute difference | Cake-first | Export-first |
| --- | ---: | ---: | ---: | ---: | ---: |
| tree_fp16_contiguous_tmem | 30.9520 | 30.7520 | 0.646% | 0.9928 | 0.9959 |
| tree_fp16_paged_tmem | 29.8242 | 29.8720 | 0.160% | 1.0022 | 1.0000 |
| tree_fp8_contiguous_tmem | 23.9125 | 23.8723 | 0.168% | 0.9987 | 0.9967 |
| tree_fp8_paged_tmem | 24.0320 | 23.8160 | 0.899% | 0.9907 | 0.9880 |
| tree_fp8_contiguous_pair | 22.7200 | 22.8640 | 0.634% | 1.0056 | 1.0070 |
| tree_fp8_paged_pair | 22.6803 | 22.7045 | 0.107% | 1.0014 | 1.0000 |

**All 6 rows (four tmem, two pair) pass; maximum difference is 0.899%.**

Public ledger benchmark (`benchmarks/bench_sm110_xqa.py`, one process for all 18 rows, same protocol as above):

| Ledger row | Kernel family | Native latency (us) | Ratio |
| --- | --- | ---: | ---: |
| tree_fp16_contiguous | tcgen05 | 59.345 | 1 (reference row) |
| tree_fp16_contiguous_tmem | tmem | 30.208 | 1.965x faster than tcgen05 |
| tree_fp16_paged | tcgen05 | 61.505 | 1 (reference row) |
| tree_fp16_paged_tmem | tmem | 30.944 | 1.988x faster than tcgen05 |
| tree_fp8_contiguous | tcgen05 | 58.880 | 1 (reference row) |
| tree_fp8_contiguous_tmem | tmem | 23.840 | 2.470x faster than tcgen05 |
| tree_fp8_contiguous_pair | pair | 22.305 | 2.640x faster than tcgen05, 1.069x faster than the previous `auto` row (tmem) |
| tree_fp8_paged | tcgen05 | 59.680 | 1 (reference row) |
| tree_fp8_paged_tmem | tmem | 23.904 | 2.497x faster than tcgen05 |
| tree_fp8_paged_pair | pair | 22.912 | 2.605x faster than tcgen05, 1.043x faster than the previous `auto` row (tmem) |

Correctness: JIT metadata suite 44 passed; GPU suite 306 passed (adds the tmem cache cases at GQA ratios 2/4/8/16 with even and odd Q-tile counts, the pair cases on both E4M3 modes at every ratio, `auto` routing to pair / tmem, the pair rejections of FP16 KV and odd Q-tile counts, and the JIT fixture for the per-ratio schema). pre-commit clean.

Synchronization (same protocol, hard 20 s limit, `memcheck` not run):

| Route | Tool | Verdict | Wall | Summary |
| --- | --- | --- | ---: | --- |
| tree_fp16_contiguous_tmem | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| tree_fp16_contiguous_tmem | racecheck | pass | 9 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| tree_fp16_paged_tmem | synccheck | pass | 7 s | ========= ERROR SUMMARY: 0 errors |
| tree_fp16_paged_tmem | racecheck | pass | 9 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| tree_fp8_contiguous_tmem | synccheck | pass | 7 s | ========= ERROR SUMMARY: 0 errors |
| tree_fp8_contiguous_tmem | racecheck | pass | 8 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| tree_fp8_paged_tmem | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| tree_fp8_paged_tmem | racecheck | pass | 9 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| tree_fp8_contiguous_pair | synccheck | pass | 7 s | ========= ERROR SUMMARY: 0 errors |
| tree_fp8_contiguous_pair | racecheck | pass | 8 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| tree_fp8_paged_pair | synccheck | pass | 8 s | ========= ERROR SUMMARY: 0 errors |
| tree_fp8_paged_pair | racecheck | pass | 8 s | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
