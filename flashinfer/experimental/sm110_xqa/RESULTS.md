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
