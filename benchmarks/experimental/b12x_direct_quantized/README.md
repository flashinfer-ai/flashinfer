# SM120 low-token quantized MoE experiments

This directory contains the low-token (`M=1..8`) SM120 experiments and the
benchmark used to compare the public dispatch against a pristine B12x
checkout. Two precision modes are implemented:

- `W4A16`: BF16 activations/output with B12x-compatible E2M1 FP4 weights.
- `NVFP4`/`W4A4`: the existing optimized input quantizer, FP4 gate/up, fused
  SwiGLU-to-FP4 quantization, FP4 down projection, and BF16 output.

Weight block scales are unswizzled and folded once at model load. The public
W4A16 dispatcher selects a measured Tensor Core launch for the Qwen and JoyAI
shapes and retains the scalar Direct kernel as a safe fallback. Both precision
paths consume precomputed routes directly and are CUDA Graph safe after
warm-up.

The Direct SM12x path requires CUDA 12.9 or newer. Other FlashInfer kernels may
support CUDA 12.8, but this path is gated because its SM12x JIT normalization
and CUDA FP4 conversion types require the newer toolkit.

Run a CUDA Graph benchmark with warm-up:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/experimental/b12x_direct_quantized/bench.py \
  --preset qwen --warmup 100 --iterations 1000 --csv /tmp/qwen.csv
```

Use `--tune-direct` for a launch-policy search. B12x comparisons use all-local
experts because `B12xMoEWrapper` does not currently support expert parallelism.
The benchmark reports failures per mode instead of silently dropping a shape.

## RTX PRO 6000 Blackwell Server Edition results

The checked-in final CSV uses one fresh process per M, 100 warm-up CUDA Graph
replays, 1,000 measured replays, 64 experts, and top-k 8. Latency is in
microseconds. The baseline is a pristine checkout at `1afe1cd`; its validation
was relaxed only for the tile configurations already selected by B12x, without
changing the baseline kernel or launch policy.

| Shape / mode | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | Minimum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen H2048/I512 W4A16 | 1.333x | 1.122x | 1.443x | 1.081x | 1.302x | 1.706x | 1.381x | 1.534x | 1.081x |
| Qwen H2048/I512 NVFP4 | 1.429x | 1.626x | 1.506x | 1.470x | 1.462x | 1.498x | 2.282x | 2.194x | 1.429x |
| JoyAI H2048/I768 W4A16 | 1.126x | 1.210x | 1.249x | 1.428x | 1.570x | 1.507x | 1.313x | 1.413x | 1.126x |
| JoyAI H2048/I768 NVFP4 | 1.668x | 2.601x | 1.426x | 1.345x | 1.377x | 2.096x | 1.874x | 1.760x | 1.345x |

All 32 measured points exceed the 1.08x acceptance threshold. The complete
baseline latency, optimized latency, speedup, and numerical error are recorded
in `results/b12x_direct_vs_pristine_final.csv`.

Additional vLLM modular-dispatch, CUDA Graph, and host Compute Sanitizer data
is recorded in
`results/vllm_sm120_integration_sanitizer_20260813.md`. The report includes the
W4A16 racecheck follow-up observed on the target H2048/I512 Tensor Core path.
