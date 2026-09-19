# Complete Ulysses examples with native FlashInfer attention

Follow-up to [the architecture experiments](ulysses_architecture_experiments.md),
tracking [#5316](https://github.com/flashinfer-ai/flashinfer/issues/5316).
The example adapters are benchmark-local, not new public APIs or new kernels.

## Explicit backend selection

| Example `--attention` | Existing FlashInfer compute | Architecture | Compute dtype |
| --- | --- | --- | --- |
| `fa3-bf16` | `BatchPrefillWithRaggedKVCacheWrapper(backend="fa3")` | SM90 | BF16 |
| `fa3-fp8` | Same native FA3 wrapper with explicit Q/K/V descales | SM90 | E4M3 Q/K/V, BF16 O |
| `trtllm-bf16` | `trtllm_ragged_attention_deepseek(backend="trtllm-gen")` | SM100 | BF16 |
| `trtllm-fp8` | Same native TRT-LLM Gen API | SM100 | E4M3 Q/K/V, BF16 O |
| `sm120-bf16` | `bsa_attn_sm120_blk64_fwd` with all blocks active | SM120 | BF16 |
| `sm120-fp8` | `sm120_fmha_fp8_ragged_prefill` | SM120 | E4M3 Q/K/V, BF16 O |
| `sm120-sage` | `quantize_sage_qkv_sm120` + `bsa_attn_sm120_blk64_sage_fwd(backend="cute_dsl")` | SM120 | QK INT8 / PV FP8, BF16 O |

No external FlashAttention or Sage installation/patch is used by these examples.
The SM100 example is **not distributed FA4**: ordinary in-tree TRT-LLM Gen is
composed with stream-level Ulysses overlap. The separate remote-Q/owner-O FA4
experiment and its hardware/dependency requirements are unchanged.

SM120 BSA metadata selects every block: this is dense attention semantics, not
a sparsity-based speedup claim. All-active index storage is quadratic in the
number of 64-token blocks; setup/memory costs should be considered for long S.
Sage uses the in-tree block/channel quantization and smoothing implementation,
not an external Sage recipe or the separate experimental K-mean helper.

## Runnable complete comparison

From a FlashInfer source checkout with its normal kernel prerequisites installed:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
torchrun --standalone --nproc-per-node=2 benchmarks/comm/bench_ulysses_native_attention.py \
  --attention sm120-sage --sequence 512 --used 497 --heads 8 --schedule 2,2
```

Use `--nproc-per-node=4 --schedule 1,1` or
`--nproc-per-node=8 --schedule 1` for that H=8 fixture. The latter has only one
head per rank and cannot exploit multiple head chunks. Select an admitted
backend from the table for other architectures. An H3-like physical shape is:

```bash
torchrun --standalone --nproc-per-node=4 benchmarks/comm/bench_ulysses_native_attention.py \
  --attention sm120-sage --sequence 37888 --used 37807 --heads 56 --schedule 7,7
```

To exercise coarse projection with native attention as well:

```bash
torchrun --standalone --nproc-per-node=2 benchmarks/comm/bench_ulysses_grouped_producer.py \
  --attention sm120-sage --seq 512 --heads 8 --dim 128 --hidden 256 --schedule 2,2
```

The QKV-to-O example compares four modes:

1. `ordinary`: three input scatters, one native attention call, output gather.
2. `whole_fused`: one fused-QKV scatter, same native attention, output gather.
3. `chunk_serial`: chosen head bands, serial communication/compute.
4. `chunk_overlap`: same bands and kernels, input/compute/output stream overlap.

The **same selected compute kernel and quantization recipe** are used by all
four. Baseline is not SDPA; SDPA is only the independent correctness reference
outside the measured interval. Output includes per-case samples, max-rank
median latency, speedup versus ordinary and relative RMSE versus BF16 SDPA.
Order rotates across cases each iteration. Native quantization, layout copies,
allocations inside run, native attention and communication are timed. Metadata,
workspace setup, compilation and warmup are excluded. This is an operator
pipeline, not a full model and not a pure attention-kernel microbenchmark.

## Numerical and lifecycle boundaries

- Communication remains **BF16 for Q/K/V/O**, including FP8 and Sage compute.
  This isolates scheduling experiments from low-precision wire experiments.
  The separate per-head quant-pack primitive is not substituted into this path.
- FP8 examples use a fixed scalar `--fp8-scale` (default 1/32) identically on
  every rank/head, making quantization independent of the chunk schedule.
  This is a synthetic-input recipe, not calibrated model scales. Quantization
  saturates outside its range; supply appropriate scales and validate accuracy.
- B=1, D=128, non-causal MHA, H/S divisible by U. `--used` defines one valid
  global prefix. Padding KV is cropped before quantization/attention and cannot
  affect valid tokens. Tail output is zero, **not** the second sequence of
  `cu=[0,used,S]`; consumers must discard it. No causal/multi-sample contract.
- Rank agreement and local kernel preparation checks precede timed communication.
  There is no silent fallback to SDPA, BF16 compute, or another backend.
- Every simultaneously consumed chunk owns separate persistent attention/output
  storage. Caller-stream joins protect reuse. No reentrancy or CUDA Graph claim.
- Independent BF16 reference tolerances are 0.01 for BF16 and 0.06 for lossy
  modes; comparison against ordinary **same-backend** output uses atol=0.002,
  rtol=0.01. These are synthetic correctness checks, not model-quality gates.

## Current validation (2026-09-18)

Available hardware: **one RTX PRO 6000 Blackwell Workstation Edition**,
Torch `2.15.0.dev20260915+cu134`. Actual SM120 kernels executed; SM90/SM100
tests are hardware-gated and await their machines. U=1 has no inter-GPU A2A and
does **not** validate multi-GPU transport or performance.

Single-rank smoke: S=256, used=193, H=4, D=128, schedule=[1,3], two warmups,
eight rotating-order samples per mode. Milliseconds:

| Native mode | Ordinary | Whole fused | Chunk serial | Chunk overlap | Relative RMSE vs BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SM120 BF16 | 0.377456 | 0.485728 | 1.009088 | 1.018512 | 0.001835 |
| SM120 FP8 | 0.265920 | 0.352736 | 0.781312 | 0.774944 | 0.051096 |
| SM120 Sage | 0.623744 | 0.685456 | 1.404640 | 1.475248 | 0.036461 |

These small single-rank cases are **slower when chunked**. No positive multi-GPU
claim or speedup inheritance is made. Lossy relative errors are reported rather
than treated as proof of model quality. Native Sage producer smoke also passed
(U1/S128/H4/D128/K64, groups=[1,3]): whole projection 1.587072 ms versus coarse
1.631232 ms, five samples. Raw smoke samples are in
`ulysses_native_sm120_smoke.json` beside this document.

```bash
python -m pytest tests/experimental/ulysses/test_native_attention.py -q
```

Tests cover real-kernel prefix correctness, padding isolation, non-default
streams, changed-input reuse, uneven/small head chunks, and a complete single-
rank pipeline. Next acceptance gates are SM90/SM100 execution and U2/U4/U8
paired correctness/performance on all three architectures. Native-kernel
availability is not a guarantee that a head-chunk schedule wins.
