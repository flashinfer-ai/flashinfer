# SM110 GQA decode results

## Primary comparison: paired kernel timing against XQA

Fresh measurements on 2026-09-09 compare the exported kernels at commit
`105a1efd2bae8a0ab8ab3a0895f2114a019b5754` with a fixed, ahead-of-time XQA
specialization on NVIDIA Thor (compute capability 11.0, 20 SMs), CUDA 13.5,
and a PyTorch 2.14 development build. XQA is the kernel-performance baseline;
PyTorch is used separately as a correctness oracle and supplementary API
comparison.

| Shape | Valid lengths | Exported SM110 kernel | XQA | Speedup vs. XQA |
| --- | --- | ---: | ---: | ---: |
| B1, capacity 64 | `[1]` | 0.004512 ms | 0.005792 ms | 1.284x |
| B4, capacity 256 | `[64, 127, 191, 256]` | 0.026048 ms | 0.032768 ms | 1.258x |
| B1, capacity 1024 | `[1024]` | 0.035745 ms | 0.041984 ms | 1.175x |
| B1, capacity 4096 | `[3968]` | 0.117633 ms | 0.124513 ms | 1.058x |

All four exported-kernel medians are lower than XQA's: a 1.058x–1.284x
speedup, or 5.53%–22.10% lower kernel latency in this run.

### Measurement boundary and protocol

- All four shapes were measured in one process, with identical Q, KV, and
  sequence-length storage shared by both arms and separate preallocated
  output buffers. No rows were resumed from an earlier process.
- Each timed callback launches one prepared kernel. Compilation, tensor-view
  preparation, output allocation, and public API sequence-length validation
  are outside both timing regions. These are **kernel-only** results, not
  end-to-end `sm110_gqa_decode` API latency.
- Timing uses CUPTI GPU activity spans with an L2 flush before every sample,
  three counterbalanced paired groups, 100 ms warmup and 1000 ms target
  measurement duration per arm, and a 6x duration safety factor. CUPTI is
  required; there is no event-timing fallback. Device clocks were not locked.
- CUPTI records were checked for exactly one launch and one expected kernel
  activity per sample: `kernel_sm110_gqa_decode_short` for capacity 64,
  `kernel_sm110_gqa_decode_long` for the other rows, and XQA's `kernel_mha`.
- Inputs are FP32 normal samples scaled by 0.25 and cast to FP16, with seeds
  95601, 95602, 95603, and 95604 in table order; `q_scale=1.0`. All rows use
  32 query heads, 8 KV heads, head dimension 128, and one query token.

The public API, prepared exported kernel, and XQA all passed an independent
FP32 attention oracle cast to FP16, with `atol=rtol=1e-2`. All outputs were
finite and Q, KV, and sequence-length inputs remained unchanged. The largest
absolute error across the four shapes was `6.103515625e-5`.

Artifact identity:

- Exported JIT module: `sm110_gqa_decode_a8bdde430c0d0714b134`.
- Exported JIT library SHA-256:
  `b34b21598b57455430b683abbd0dd85f4b31be745b0f6eaf2b91afa6c4f5af94`.
- XQA cubin SHA-256:
  `259b035bf26b9dc92cd9682802933107e71f5d4171685cbf37c8b3278c441786`.

This comparison loads the fixed XQA artifact directly; it does not exercise
FlashInfer's public XQA dispatcher. The paired driver and XQA artifact are
not bundled in this PR. The command below reproduces only the supplementary
SDPA comparison, which has a different timing boundary.

## Supplementary comparison: public API against PyTorch SDPA

Measurements were collected on the same GPU type with CUDA 13.5 and a
PyTorch 2.14 development build. The public
`benchmarks/bench_sm110_gqa_decode.py` benchmark runs both implementations in
one process on identical tensors and reports median GPU activity spans from
CUPTI with an L2 flush before every sample. Both timed calls allocate and
return their output tensor; neither reuses a caller-provided output buffer.
The candidate measurement includes the public API's sequence-length validation.

This target-native public benchmark uses PyTorch SDPA as its reference; it
does not replace the XQA kernel-performance baseline above.

| Shape | Valid lengths | SM110 GQA decode | PyTorch SDPA | Speedup |
| --- | --- | ---: | ---: | ---: |
| B1, capacity 64 | `[1]` | 0.161282 ms | 0.541734 ms | 3.36x |
| B4, capacity 256 | `[64, 127, 191, 256]` | 0.184097 ms | 0.632294 ms | 3.43x |
| B1, capacity 1024 | `[1024]` | 0.193154 ms | 0.629159 ms | 3.26x |
| B1, capacity 4096 | `[3968]` | 0.277633 ms | 2.361688 ms | 8.51x |

Correctness was checked against an independent FP32 attention calculation and
then cast to FP16 with `atol=1e-2` and `rtol=1e-2`. All four rows passed, all
outputs were finite, and Q, KV, and sequence-length inputs remained unchanged.

Run the benchmark with:

```bash
python benchmarks/bench_sm110_gqa_decode.py
```
