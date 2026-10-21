# SM120 GDN pipeline32

The opt-in p32 implementation precomputes chunk-local transforms, then runs a
32-token consumer with four warp groups (state, delta, output and loading).
Its grid is `(2, heads, sequences)`: two V=64 CTAs per head/sequence, with 512
threads per CTA. This does not imply two resident CTAs on one SM. Inputs use
ping-pong shared memory and ready/free mbarriers; accumulations are FP32.

## Opt-in and applicability

```python
import os
from flashinfer.gdn_prefill import chunk_gated_delta_rule

os.environ["FLASHINFER_GDN_SM120_PIPELINE32"] = "1"
o, state = chunk_gated_delta_rule(
    q, k, v, g=alpha, beta=beta, initial_state=initial_state,
    cu_seqlens=cu_seqlens, output_final_state=True,
    use_cp=False, backend="flashinfer",
)
```

**This is not a general SM12x speedup. Performance depends on GPU SKU, sequence
length and per-rank head count. Measure your workload before enabling it.**
No automatic length threshold is introduced. The default route is unchanged.
Only exact capability `(12, 0)` without `state_indices` can select p32; state
pools, SM121 and an unavailable p32 import retain the official implementation.
Other SKUs have not been remeasured here; the review reports regressions on
RTX 5090 and RTX PRO 6000 Blackwell Max-Q. No benefit on those SKUs is claimed.

Q/K/V must be contiguous FP16/BF16 tensors `[tokens, heads, 128]`; gates are
multiplicative FP32 and state is contiguous FP32 `[sequences, heads, V, K]`.
The public QK normalization flag is supported: Q/K are normalized before
p32 dispatch. Only `FLASHINFER_GDN_SM120_PIPELINE32` selects this implementation;
downstream environment aliases are no longer read.

For low-level checkpoints, reserve at least `ceil(total_tokens / interval)`
entries. The interval must be a positive multiple of 32 (the public API keeps
its own restrictions). Checkpoint starts must describe valid packed,
non-overlapping per-sequence ranges. Capacity is checked before compilation or
launch without copying GPU metadata to the CPU, including during graph capture.
Only complete intervals are written; unused capacity is not initialized.

## Reproduce validation and timing

Use this checkout and explicitly activate your CUDA Python environment.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -p no:cacheprovider -rA \
  tests/gdn/test_prefill_pipeline32.py tests/gdn/test_pipeline32_host.py
CUDA_VISIBLE_DEVICES=0 compute-sanitizer --tool memcheck --error-exitcode 99 \
  --report-api-errors no --print-limit 50 python -m pytest -q -p no:cacheprovider \
  tests/gdn/test_prefill_pipeline32.py
pre-commit run --files $(git diff --name-only)

# FLA is optional. FI-only mode can also compare against official FI fuse.
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_gdn_sm120_pipeline32.py \
  --model '27B TP1' --seqlen 1x --fi-path all --check-acc \
  --backend cupti --warmup 10 --repeats 100 --json /tmp/fi-only.json

# Add --fla to time FLA and use it as the accuracy reference.
for round in 1 2 3; do
  for tp in 1 2; do
    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_gdn_sm120_pipeline32.py \
      --model "27B TP$tp" --seqlen 1x --fi-path all --fla --check-acc \
      --fla-chunk-size 64 --warmup 10 --repeats 100 --backend cupti \
      --json "/tmp/tp${tp}-round${round}.json"
  done
done
```

`--backend cupti` uses `flashinfer.testing.bench_gpu_time(enable_cupti=True,
cold_l2_cache=True)` and reports the median of independent invocation samples.
Install `cupti-python>=13` with a CUPTI library matching your CUDA version.
The driver sums GPU kernel durations for each invocation, excludes CPU launch
gaps, and flushes L2 between iterations. It does not divide a single event span
by a batch of consecutive graph replays. Event and single-call graph timing
remain explicit alternatives, not the reported CUPTI results.

All implementations share inputs. Timed FI callables include log-to-linear
gate conversion, initial/final KV/VK state transpose and index conversion;
p32 includes both precompute and consumer. FLA uses the same explicit chunk
size for accuracy and timing. Accuracy checks both output and final state with
relative L2 <= 0.02, rejecting nonfinite values. TP1/TP2 here mean per-rank head
shapes `(Hqk,Hv)=(16,48)/(8,24)`, not a multi-GPU distributed benchmark. D=128,
BF16, one sequence, and lengths 2K, 4K, 8K, 16K and 32K.

Results record source SHA-256 hashes, base PR head, versions, GPU name, commands
and timing boundary. Official fused/CP sources are unchanged from PR base
`3637083a3a3d08538cce60937cd876b17779c6f0`; the old FlashInfer 0.6.18 table is
superseded by the measurements below, not mixed into the new comparison.

## Validation and measured results

2026-10-21: NVIDIA RTX 6000D, driver 580.126.09, CUDA 13.0,
Torch 2.13.0+cu130, source version 0.7.2 at PR head `38c70db5ec` plus these
review fixes (exact source hashes in JSON), FLA 0.5.2, Triton 3.7.1 and
CUPTI Python 13.0.1. Three interleaved TP1/TP2 rounds, each 10 warmups and
100 timed invocations; table is the median of each round's median.
Above 1x means p32 is faster. These values supersede the old historical table.

| TP shape | Tokens | FLA ms | Fused ms | CP ms | p32 ms | vs fused | vs CP | vs FLA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP1 | 2048 | 0.334 | 0.244 | 0.313 | 0.329 | 0.74x | 0.95x | 1.01x |
| TP1 | 4096 | 0.660 | 0.435 | 0.443 | 0.535 | 0.81x | 0.83x | 1.23x |
| TP1 | 8192 | 1.293 | 1.160 | 0.865 | 0.952 | 1.22x | 0.91x | 1.36x |
| TP1 | 16384 | 2.583 | 2.879 | 1.542 | 1.801 | 1.60x | 0.86x | 1.43x |
| TP1 | 32768 | 5.186 | 5.998 | 3.842 | 3.486 | 1.72x | 1.10x | 1.49x |
| TP2 | 2048 | 0.221 | 0.237 | 0.225 | 0.312 | 0.76x | 0.72x | 0.71x |
| TP2 | 4096 | 0.354 | 0.420 | 0.322 | 0.508 | 0.83x | 0.63x | 0.70x |
| TP2 | 8192 | 0.718 | 1.066 | 0.455 | 0.905 | 1.18x | 0.50x | 0.79x |
| TP2 | 16384 | 1.405 | 2.896 | 0.859 | 1.701 | 1.70x | 0.51x | 0.83x |
| TP2 | 32768 | 2.804 | 6.020 | 1.507 | 3.285 | 1.83x | 0.46x | 0.85x |

Short inputs regress against official fused/CP. On this SKU TP1 at 32K beats
both official routes, while TP2 remains slower than CP at every tested length.
No portable crossover or default selection policy is inferred from one SKU.

Validation: 24 tests passed, including the original 8 normalization cases,
both index-dtype orders with/without checkpoints, checkpoint references and
capacity, CUDA graph replay, state-pool fallback, simulated SM121 routing,
missing-kernel fallback, and FI-only execution with FLA imports blocked.
Memcheck: 17 GPU tests passed, **0 device-memory errors**; API error reporting
was disabled (`--report-api-errors no`), so this is not an API-clean claim.
Mypy passed on 4 changed source files. All fixed-version project pre-commit
hooks passed for the 9 changed files. Hook sources and wheels were staged
offline; hook logic and the project configuration were unchanged. SM121 is a routing simulation on SM120;
no physical SM121, RTX 5090 or Max-Q validation is claimed.

Results and exact commands are recorded in
`results/gdn_sm120_pipeline32_rtx6000d.json`.

Timing does not establish that a particular MMA throttle was eliminated.
No full-model or TTFT speedup, or performance on untested SKUs, is claimed.
