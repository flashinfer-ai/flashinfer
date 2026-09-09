# SM110 GQA decode results

Measurements were collected on NVIDIA Thor with compute capability 11.0,
CUDA 13.4, and a PyTorch 2.14 development build. The public
`benchmarks/bench_sm110_gqa_decode.py` benchmark runs both implementations in
one process on identical tensors and reports median GPU activity spans from
CUPTI with an L2 flush before every sample.

The public XQA backend does not expose an SM110 route, so this target-native
benchmark uses PyTorch SDPA as its available public reference.

| Shape | Valid lengths | SM110 GQA decode | PyTorch SDPA | Speedup |
| --- | --- | ---: | ---: | ---: |
| B1, capacity 64 | `[1]` | 0.004832 ms | 0.170625 ms | 35.31x |
| B4, capacity 256 | `[64, 127, 191, 256]` | 0.026113 ms | 0.259938 ms | 9.95x |
| B1, capacity 1024 | `[1024]` | 0.036000 ms | 0.256642 ms | 7.13x |
| B1, capacity 4096 | `[3968]` | 0.122433 ms | 0.530915 ms | 4.34x |

Correctness was checked against an independent FP32 attention calculation and
then cast to FP16 with `atol=1e-2` and `rtol=1e-2`. All four rows passed, all
outputs were finite, and Q, KV, and sequence-length inputs remained unchanged.

Run the benchmark with:

```bash
python benchmarks/bench_sm110_gqa_decode.py
```
