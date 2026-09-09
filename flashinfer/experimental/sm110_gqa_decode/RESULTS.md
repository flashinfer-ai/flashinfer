# SM110 GQA decode results

Measurements were collected on NVIDIA Thor with compute capability 11.0,
CUDA 13.5, and a PyTorch 2.14 development build. The public
`benchmarks/bench_sm110_gqa_decode.py` benchmark runs both implementations in
one process on identical tensors and reports median GPU activity spans from
CUPTI with an L2 flush before every sample. Both timed calls allocate and
return their output tensor; neither reuses a caller-provided output buffer.
The candidate measurement includes the public API's sequence-length validation.

The public XQA backend does not expose an SM110 route, so this target-native
benchmark uses PyTorch SDPA as its available public reference.

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
