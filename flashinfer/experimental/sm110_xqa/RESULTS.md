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
