# CAKE KDA K1-parallel validation report

This report records the correctness, sanitizer, routing, and performance checks
for the owner/helper CAKE KDA prefill implementation. GPU validation ran on
implementation commit `2c2925ece209ac4e7a0e8151b8b186598c0740cd`. Its rebased
equivalent is `039bd3a4561899dbf567b237f5113f2065ed9aa1`; `git range-diff` shows
the 17 implementation commits are unchanged, and the intervening upstream
commits do not touch KDA paths. The final child commit only updates this report.

## Scope

The change preserves CAKE's fully fused K1 + K2 structure. One owner CTA keeps
the ordered K2 recurrence and TMEM-resident state, while helper CTAs prepare
independent K1 chunks into a generation-tagged global-memory mailbox. The owner
consumes ready packets in token order, so K2 can overlap later K1 work without
waiting for a batch-wide K1 phase.

The original fallback contract is intentionally unchanged:

- fixed `B=1, H=64` may use CAKE-M64;
- every other fixed shape falls back to CAKE-M128;
- packed varlen falls back to CAKE-M128.

The M64 baseline is specialized for fixed `B=1, H=64`; it is not a valid
small-head oracle. Tests now reject attempts to force it for `H=1`, `H=4`, or
`H=8`.

## Conservative dispatch

All automatic helper routes use C4. M128 C8 remains available only as an
explicitly forced validation route; M64 rejects C8.

| GPU | Layout | Shape condition | Route |
| --- | --- | --- | --- |
| B200 / CC 10.0 | fixed | `B=1, H=1, T>=4096` | M64 helper, C4/D10 |
| B200 / CC 10.0 | fixed or packed | TMA-supported `H`, average `T>=2048`, `B*H<=8` | M128 helper, C4/D15 |
| B200 / CC 10.0 | fixed or packed | TMA-supported `H`, average `T>=2048`, `B*H<=32` | M128 helper, C4/D30 |
| B300 / CC 10.3 | fixed | TMA-supported `H`, `T>=2048`, `B*H<=8` | M128 helper, C4/D30 |
| B300 / CC 10.3 | fixed | TMA-supported `H`, `T>=2048`, `B*H<=32` | M128 helper, C4/D45 |
| B300 / CC 10.3 | packed | any | original CAKE fallback |
| Any | any | outside the rows above | original CAKE fallback |

Packed routing uses the host-known average length `total_tokens / B`; it does
not copy or read `cu_seqlens` on the CPU. This is deliberately conservative for
highly skewed batches.

## Source-build and functional validation

Both GPU runs explicitly replaced `FLASHINFER_AOT_DIR` with a nonexistent
directory and used a fresh `FLASHINFER_WORKSPACE_BASE`. This prevents the
installed `flashinfer-jit-cache` package from satisfying module loads and proves
that the tested binaries were built from this commit.

On each GPU, NVCC generated four fresh SM100f modules:

- `flash_kda_bf16_fused_m64_sm100f`;
- `flash_kda_bf16_fused_m128_sm100f`;
- `flash_kda_bf16_fused_m64_k1_parallel_sm100f`;
- `flash_kda_bf16_fused_m128_k1_parallel_sm100f`.

The focused suites were:

```text
tests/jit/test_flash_kda_jit.py
tests/kda/test_recurrent_kda_prefill.py
```

| Device | CUDA / target | Result |
| --- | --- | --- |
| B200-class, 148 SM, CC 10.0 | CUDA 13.0, `sm100f` | 129 passed |
| B300-class, 148 SM, CC 10.3 | CUDA 13.0, `sm100f` | 126 passed, 3 B200-only skips |

Coverage includes route selection, FFI ABI, mailbox bounds, unsupported route
rejection, M64/M128 baselines, M64 C4 helper, M128 C4/C8 helpers, fixed and
packed inputs, bitwise output/state comparison, reference comparison, nondefault
streams, in-place state updates, and CUDA Graph capture/replay.

## Compute Sanitizer

The sanitizer subset exercised five physical paths: M128 baseline, M64
baseline, M128 C4 helper, forced M128 C8 helper, and M64 C4 helper.

| Device | memcheck | synccheck |
| --- | --- | --- |
| B200 / CC 10.0 | 5 passed, 0 errors | 5 passed, 0 errors |
| B300 / CC 10.3 | 5 passed, 0 errors | 5 passed, 0 errors |

On the B300 environment, CUTLASS Python initialization performs optional
`cuGetProcAddress_v2` probes that return `CUDA_ERROR_INVALID_VALUE`. The first
memcheck run counted 21 such host API probes even though all five tests passed.
The recorded B300 sanitizer results use `--report-api-errors no`, which suppresses
host API-return diagnostics but continues to report device memory and
synchronization errors. B200 required no suppression.

## B200 source-built performance smoke

Cold-L2 CUDA-event timings were rerun against the source-built modules with two
deterministically shuffled rounds. The baseline is CAKE-M128 for all rows below;
the invalid small-head M64 baseline is never used.

| B | T | H | Selected route | CAKE-M128 | Owner/helper | Speedup |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 2048 | 1 | M128 C4/D15 | 0.144480 ms | 0.119872 ms | 1.205x |
| 1 | 2048 | 4 | M128 C4/D15 | 0.144480 ms | 0.121888 ms | 1.185x |
| 1 | 2048 | 8 | M128 C4/D15 | 0.142048 ms | 0.119712 ms | 1.187x |
| 1 | 4096 | 1 | M64 C4/D10 | 0.269424 ms | 0.199120 ms | 1.353x |
| 1 | 4096 | 4 | M128 C4/D15 | 0.270496 ms | 0.216064 ms | 1.252x |
| 1 | 4096 | 8 | M128 C4/D15 | 0.267168 ms | 0.211008 ms | 1.266x |

This smoke test is intended to verify the final route and binary, not to replace
the broader tuning sweeps. Automatic benchmark routes are checked for output and
final-state correctness before timing.

## Local PR gates

- rebased directly onto the current upstream `main` used for this work;
- `pre-commit run --all-files` passes;
- `git diff --check` passes;
- all commits carry `Signed-off-by` trailers;
- the worktree is clean after the report commit;
- no official upstream pull request has been opened.

The complete upstream CI matrix and maintainer-owned performance infrastructure
remain the final merge gates.
