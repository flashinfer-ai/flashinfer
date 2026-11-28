# Sage FP8 BSA #9 integration validation

Status: local integration and B200 validation complete. SM103 and SM107 still
require hardware CI.

## Source and scope

Baseline: FlashInfer `26075a1331ff91530706567b59389c23cc143648`.
Port reference: Block-Sparse-Attention `9698fce43bc14deda49f27b371a395821d8c8a32`
([BSA #9](https://github.com/NVIDIA-JerryChen/Block-Sparse-Attention/pull/9)),
including prerequisite native-tail and synchronization changes.

This includes [FlashInfer #5090](https://github.com/flashinfer-ai/flashinfer/pull/5090)'s
BF16 tail masking, true-extent TMA reads and wrapper acceptance of nonaligned KV.
It additionally supports native Sage FP8 Q/KV tails, variable sparse counts,
rank-1/2/3 block sizes and empty direct-API rows. Physical and declared valid
lengths are intersected, even if the user declares a tail size of 64. The wrapper
still requires aligned Q and nonempty sparse rows.

CuTe FP8 admission remains MHA, B=1, H in {4,8}, D=128. Cake eligibility and
`auto` preference are unchanged. The existing SM107 entrance is retained but
unqualified on this machine. SM120 is outside this change.

Implemented: FP8 max tree, select mask, K-scale prefetch, shared softmax body,
176/104 register allocation, persistent epilogue storage, TMA output and
SM-count/work-dependent split selection. Unsplit FP8 and split combine write
aligned caller outputs directly. The compile-key schema is bumped.

Safety fixes found and validated during integration:

- Remove FI's divergent CTA barrier inside CLC consumer advancement.
- Synchronize FP8 TMEM allocation and teardown across the CTA.
- Use explicit `barrier.cta.sync` with a compiler memory clobber for partial-CTA
  waits: CuTe 4.8's `wait_unaligned()` still lowers through implicitly aligned
  `nvvm.barrier`, which failed BF16 TMEM synccheck.
- Order V-scale cache reuse before overwrite, including the epilogue warp of
  explicitly forced CLC + split-KV. The latter reproduced four race warnings
  before the added warp sync and zero after the fix.
- Keep BF16's original stats-slot timing. Applying the upstream deferred-wait
  protocol to BF16 caused a 3%-6% regression and was excluded.

## Environment and timing

Measured 2026-11-28, NVIDIA B200 (148 SMs), driver 615.71.10, Python 3.12,
PyTorch 2.11.0a0+a6c236b9fd.nv26.03.46836102, CUDA 13.2, CuTe DSL 4.8.1,
TVM-FFI 0.1.16, CUDA bindings 13.2.0, cupti-python 13.2.0 and CUPTI library
13.2.86. Dependencies are isolated under `/tmp`; system packages were not
replaced. No flashinfer-jit-cache distribution was installed.

Performance: B=1, H=4, D=128, seed 42, preallocated BF16 BSHD output; top-k
counts 64-token blocks. Serial runs on GPU 0, clocks unlocked, compilation and
warmup excluded. GPU numbers are CUPTI medians, without CUDA-event fallback.
`gpu_us` spans the first to last operation kernel, including inter-launch gaps;
it is not the sum of isolated kernel durations. Eager timing uses warm L2 unless
`--cold-l2` is set; graph replay always flushes L2. Host timing is mean warmed
asynchronous enqueue time over 100 calls, excluding final synchronization.
Memory is peak additional PyTorch allocation above inputs/caller output; it
excludes compiler memory, allocator reservations and device shared memory.
These are operation benchmarks, not serving startup or KV-cache-fit measurements.

## CuTe FP8 performance

Graph replay:

| Sq | Sk | KV blocks | Splits requested | Before (us) | After (us) | Reduction |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4096 | 8192 | 32 | 1 | 39.072 | 22.656 | 42.01% |
| 4096 | 8192 | 128 | 1 | 80.640 | 62.624 | 22.34% |
| 16384 | 16384 | 256 | 1 | 444.415 | 385.727 | 13.21% |
| 64 | 16384 | 256 | auto | 26.688 | 19.936 | 25.30% |
| 8192 | 65536 | 1024 | auto | 1049.470 | 830.814 | 20.83% |

Eager GPU span, warm L2:

| Sq | Sk | KV blocks | Splits requested | Before (us) | After (us) | Reduction |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4096 | 8192 | 32 | 1 | 35.648 | 21.984 | 38.33% |
| 4096 | 8192 | 128 | 1 | 77.664 | 62.145 | 19.98% |
| 16384 | 16384 | 256 | 1 | 442.623 | 380.671 | 14.00% |
| 64 | 16384 | 256 | auto | 44.864 | 30.496 | 32.03% |
| 8192 | 65536 | 1024 | auto | 1045.086 | 831.663 | 20.42% |

Host enqueue:

| Sq | Sk | KV blocks | Splits requested | Before (us) | After (us) | Reduction |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4096 | 8192 | 32 | 1 | 74.465 | 60.421 | 18.86% |
| 4096 | 8192 | 128 | 1 | 75.119 | 60.232 | 19.82% |
| 16384 | 16384 | 256 | 1 | 73.995 | 60.656 | 18.03% |
| 64 | 16384 | 256 | auto | 125.368 | 116.763 | 6.86% |
| 8192 | 65536 | 1024 | auto | 125.907 | 60.858 | 51.66% |

Additional peak allocation:

| Sq | Sk | KV blocks | Splits requested | Before (MiB) | After (MiB) |
| --- | --- | --- | --- | ---: | ---: |
| 4096 | 8192 | 32 | 1 | 8.062 | 0.062 |
| 4096 | 8192 | 128 | 1 | 8.062 | 0.062 |
| 16384 | 16384 | 256 | 1 | 32.250 | 0.250 |
| 64 | 16384 | 256 | auto | 1.071 | 2.017 |
| 8192 | 65536 | 1024 | auto | 72.625 | 0.125 |

Short-Q auto changes from 8 to 16 splits: faster replay costs about 0.95 MiB
additional temporary memory. Long-Q/high-top-k changes from 4 splits to 1,
eliminating FP32 partial workspace. This tradeoff is explicit, not a universal
memory reduction.

## Fused quantization plus attention

Both versions use the same existing `sage_fp8_quantize_sm100` quantizer and
BF16 source inputs. Includes its allocations/launches and attention, excluding
sparse-index construction. Graph replay:

| Sq | Sk | KV blocks | Splits requested | Before (us) | After (us) | Reduction |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4096 | 8192 | 32 | 1 | 49.215 | 33.232 | 32.48% |
| 4096 | 8192 | 128 | 1 | 91.328 | 73.536 | 19.48% |
| 16384 | 16384 | 256 | 1 | 459.551 | 397.472 | 13.51% |
| 64 | 16384 | 256 | auto | 38.944 | 32.448 | 16.68% |
| 8192 | 65536 | 1024 | auto | 1085.214 | 870.495 | 19.79% |

## Cold-L2 eager FP8

| Sq | Sk | KV blocks | Splits requested | Before (us) | After (us) | Reduction |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4096 | 8192 | 32 | 1 | 40.720 | 22.576 | 44.56% |
| 4096 | 8192 | 128 | 1 | 82.367 | 62.496 | 24.13% |
| 16384 | 16384 | 256 | 1 | 446.127 | 380.543 | 14.70% |
| 64 | 16384 | 256 | auto | 45.632 | 30.976 | 32.12% |
| 8192 | 65536 | 1024 | auto | 1051.230 | 830.255 | 21.02% |

## BF16 control

Graph replay after restoring BF16 stats-slot timing and adding valid partial-CTA waits:

| Sq | Sk | KV blocks | Splits requested | Before (us) | After (us) | Reduction |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 4096 | 8192 | 32 | 1 | 37.408 | 37.520 | -0.30% |
| 4096 | 8192 | 128 | 1 | 82.112 | 82.112 | 0.00% |
| 16384 | 16384 | 256 | 1 | 515.903 | 522.623 | -1.30% |
| 64 | 16384 | 256 | auto | 74.559 | 74.592 | -0.04% |
| 8192 | 65536 | 1024 | auto | 1037.647 | 1057.726 | -1.94% |

Negative reductions indicate slower control measurements. The final control
shows up to about 2% additional latency relative to the baseline, versus the
initial 3%-6% regression. Clocks were unlocked; the data does not distinguish
small clock variation from the cost of the necessary synchronization repair.
No zero-regression BF16 claim is made.

## Cake comparison

Same performance inputs, prequantized attention graph replay:

| Sq | Sk | KV blocks | Old CuTe (us) | New CuTe (us) | Cake (us) |
| --- | --- | --- | ---: | ---: | ---: |
| 4096 | 8192 | 32 | 39.072 | 22.656 | 21.856 |
| 4096 | 8192 | 128 | 80.640 | 62.624 | 63.328 |
| 16384 | 16384 | 256 | 444.415 | 385.727 | 401.663 |
| 64 | 16384 | 256 | 26.688 | 19.936 | 33.872 |
| 8192 | 65536 | 1024 | 1049.470 | 830.814 | 829.087 |

Cake ignores the split argument; CuTe requests match the first table. CuTe is
not universally faster than Cake. Eligible `auto`/split=1 calls still use Cake,
which retains lower host overhead. Explicit `kv_splits="auto"` uses CuTe as
before and benefits from the new implementation. No broad routing change was made.

## Precision: identical inputs, three implementations

On a second B200, the old/new input byte hashes match for every common case.
Reference: FP32 attention on the exactly dequantized FP8 Q/K/V, not the original
BF16 tensors. This isolates attention arithmetic/P quantization from Q/K/V
quantization error. TF32 is disabled by the highest matmul precision setting.

| Case | Old CuTe RMSE | New CuTe RMSE | Cake RMSE | New vs old bitwise |
| --- | ---: | ---: | ---: | --- |
| aligned_dense | 0.002242315 | 0.002242315 | 0.002229939 | True |
| aligned_sparse | 0.001595032 | 0.001595032 | 0.001693546 | True |
| multiwave | 0.001199247 | 0.001199247 | 0.001244837 | True |
| short_q_auto | 0.000313893 | 0.000300991 | 0.000330418 | False |

The short-Q auto case changes partition count (8 to 16), so output need not be
bitwise identical; its new/old maximum absolute difference is 0.0013427734375.
The other three aligned cases are bitwise identical. All reference checks pass.
New-only tail/padding cases intentionally do not launch the unsafe/unsupported
old FP8 path:

| Case | New CuTe RMSE | Cake RMSE | New CuTe LSE max abs | Cake LSE max abs |
| --- | ---: | ---: | ---: | ---: |
| kv_tail | 0.002900960 | 0.003127513 | 1.43e-06 | 9.54e-07 |
| short_q_tail | 0.003560526 | 0.003670109 | 4.77e-07 | 4.77e-07 |
| padding | 0.003508379 | 0.003727082 | 1.43e-06 | 9.54e-07 |
| variable_tail | 0.003484399 | 0.003572670 | 9.54e-07 | 1.43e-06 |

Raw JSON additionally records maximum absolute error, relative L2, cosine,
input hashes, and pairwise errors. Empty rows require exact zero output and
negative-infinite LSE, independently of the error tolerance.

## Direct reproduction of #5090

Q=K=0; 65 valid V rows equal 1. Backing K/V storage is fully allocated for 128
rows with zero padding, but the passed tensor extent is 65. This safely exposes
incorrect softmax padding without launching an out-of-allocation TMA read.
The mathematically correct output is exactly 1.

| | Old | New |
| --- | ---: | ---: |
| Output (all elements) | 0.5078125 | 1.0 |
| Max absolute error | 0.4921875 | 0.0 |
| Wrapper plan(N=65) | rejects divisibility | accepts; output error 0 |

This reproduces the padding correctness failure. It does not intentionally
execute a physically out-of-allocation read in the old kernel. New physical-tail
memory safety is checked separately with memcheck.

## Validation

- Full affected regression: 141 passed, 33 skipped by existing guards across
  the new boundary file, existing VSA/BSA tests and Cake/auto API tests.
- CPU split policy: 9 passed with no GPU visible.
- Final memcheck/synccheck/racecheck: seven representative tests each, including
  BF16/FP8 multiwave CLC, explicit CLC + split and changing graph metadata.
- Final initcheck: five FP8 multiwave/split/graph cases. TMA store completeness
  is independently checked with NaN sentinels and dequantized references;
  no sanitizer error suppression is used.
- Larger physical-tail memcheck: 45 passed, 0 errors.

The matrix includes Sk=1/63/64/65/127/128/193; Q tails including Sq=1;
Sq != Sk; per-head block sizes; sizes exceeding physical tails; empty and fully
masked rows; invalid unused index slots; split 1/2/4/8/16/auto; static/CLC parity;
aligned/offset/strided caller output; and replay with changing data, scales,
indices, counts and sizes. Eager auto uses active counts; capture uses index
capacity without a host readback.

## Reproduction

Run from each checkout root in the compatible environment above. Copy the same
benchmark scripts to the baseline checkout for old measurements.

```bash
python benchmarks/bench_bsa_sage_fp8.py --output attn.json
python benchmarks/bench_bsa_sage_fp8.py --quantize --output quantized.json
python benchmarks/bench_bsa_sage_fp8.py --backends cute --bf16 --output bf16.json
python benchmarks/bench_bsa_sage_fp8.py --backends cute --cold-l2 --output cold.json
# Before checkout:
python benchmarks/check_bsa_sage_fp8_accuracy.py --label before --output before.json --tensor-output before.pt
# After checkout, using before.pt:
python benchmarks/check_bsa_sage_fp8_accuracy.py --label after --baseline before.pt --output after.json --tensor-output after.pt
# Run on each checkout with the corresponding label:
python benchmarks/check_bsa_sage_fp8_accuracy.py --label after --reproduce-tail --output tail.json --tensor-output tail.pt
pytest tests/attention/test_bsa_sage_fp8_sm100.py tests/attention/test_vsa_block_sparse.py tests/attention/test_cake_sage_block_sparse_attention_sm100_api.py
CUDA_VISIBLE_DEVICES='' pytest tests/cpu/attention/test_bsa_fp8_split_policy.py
```

Prefix the selected pytest command with `compute-sanitizer --tool memcheck
--error-exitcode 99`, and separately use `synccheck`, `racecheck`, `initcheck`.
Changed-file hooks include mypy and Ruff. SM103/B300 and SM107/Rubin still need
hardware CI; no performance or sanitizer qualification on those devices is claimed.

Implementation and validation were AI-assisted.
