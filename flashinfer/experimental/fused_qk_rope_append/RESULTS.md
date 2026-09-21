# Fused QK/RoPE/append results

Measured on one NVIDIA B200 with CUDA 13.3 and PyTorch 2.11.0+cu130. Timings
are median kernel time over 30 CUPTI measurements after 10 warmups, with the L2
cache flushed before every measurement. The comparison pipeline uses the
stable FlashInfer RMSNorm, RoPE, and paged KV append primitives. Both paths
consume the same BF16-rounded norm weights, and the benchmark validates Q, K,
and V before timing.

| Shape | Fused | Stable primitive pipeline | Speedup |
| --- | ---: | ---: | ---: |
| B=32, Q=1, Hq=8, Hkv=1, D=128, context=2048 | 0.005280 ms | 0.030368 ms | 5.752x |
| B=8, Q=16, Hq=8, Hkv=1, D=128, context=2048 | 0.005344 ms | 0.034015 ms | 6.365x |

Reproduce from the repository root:

```bash
python benchmarks/bench_fused_qk_rope_append.py \
  --batch-size 32 --qo-len 1 --context-len 2048
python benchmarks/bench_fused_qk_rope_append.py \
  --batch-size 8 --qo-len 16 --context-len 2048
```

These numbers demonstrate launch and memory-traffic savings for the exact
supported shapes; they are not a claim about other head configurations. SM90
correctness and performance still need to be run on H100/H200 hardware.
