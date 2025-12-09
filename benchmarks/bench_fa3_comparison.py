import torch
import random

torch.manual_seed(42)
random.seed(42)
device = "cuda"
dtype = torch.float16

import sys

sys.path.insert(0, "/home/zhye/flash-attention/hopper")
from flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func
import flashinfer
from flashinfer.testing import (
    bench_gpu_time_with_cuda_event as bench_gpu_time_with_cupti,
)

head_dim = 128


def calc_tflops(batch_size, seq_len, num_qo_heads, head_dim, time_ms, causal=True):
    """Calculate TFLOPS for attention.

    FLOPs = 4 * batch_size * seq_len^2 * num_heads * head_dim (for non-causal)
    For causal, multiply by 0.5
    """
    flops = 4 * batch_size * seq_len * seq_len * num_qo_heads * head_dim
    if causal:
        flops = flops * 0.5
    tflops = flops / (time_ms / 1000) / 1e12
    return tflops


def calc_tflops_varlen(seq_lens, num_qo_heads, head_dim, time_ms, causal=True):
    """Calculate TFLOPS for variable length attention."""
    total_flops = sum(4 * s * s * num_qo_heads * head_dim for s in seq_lens)
    if causal:
        total_flops = total_flops * 0.5
    tflops = total_flops / (time_ms / 1000) / 1e12
    return tflops


def bench_fn(fn):
    """Benchmark a function and return median time in ms."""
    times = bench_gpu_time_with_cupti(fn, l2_flush=True)
    return sorted(times)[len(times) // 2]  # median


print("Comprehensive benchmark: FlashInfer vs FA3 (using CUPTI)")
print("=" * 115)

# bs=1 tests
print("\n--- bs=1 Single Prefill ---")
print(
    f"{'seq_len':<10} {'heads':<12} {'FlashInfer (ms)':<18} {'FA3 (ms)':<15} {'diff':<10} {'FI TFLOPS':<12} {'FA3 TFLOPS':<12}"
)
print("-" * 100)

for seq_len in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
    for num_qo_heads, num_kv_heads in [(32, 8), (32, 32)]:
        q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            backend="fa3",
        )
        wrapper.plan(
            cu_seqlens,
            cu_seqlens,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            causal=True,
        )

        fi_time = bench_fn(lambda: wrapper.run(q, k, v))
        fa3_time = bench_fn(
            lambda: fa3_varlen_func(
                q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len, causal=True
            )
        )

        diff = (fi_time - fa3_time) / fa3_time * 100
        fi_tflops = calc_tflops(
            1, seq_len, num_qo_heads, head_dim, fi_time, causal=True
        )
        fa3_tflops = calc_tflops(
            1, seq_len, num_qo_heads, head_dim, fa3_time, causal=True
        )
        heads_str = f"{num_qo_heads}/{num_kv_heads}"
        print(
            f"{seq_len:<10} {heads_str:<12} {fi_time:<18.3f} {fa3_time:<15.3f} {diff:+.1f}%{'':5} {fi_tflops:<12.1f} {fa3_tflops:<12.1f}"
        )

# Batch prefill tests
print("\n--- Batch Prefill ---")
print(
    f"{'Config':<35} {'FlashInfer (ms)':<18} {'FA3 (ms)':<15} {'diff':<10} {'FI TFLOPS':<12} {'FA3 TFLOPS':<12}"
)
print("-" * 115)

batch_configs = [
    (8, 512, 32, 8),
    (8, 1024, 32, 8),
    (8, 2048, 32, 8),
    (8, 4096, 32, 8),
    (8, 8192, 32, 8),
    (8, 512, 32, 32),
    (8, 1024, 32, 32),
    (8, 2048, 32, 32),
    (8, 4096, 32, 32),
    (8, 8192, 32, 32),
    (4, 16384, 32, 8),
    (4, 16384, 32, 32),
    (2, 32768, 32, 8),
    (2, 32768, 32, 32),
]

for batch_size, seq_len, num_qo_heads, num_kv_heads in batch_configs:
    qo_lens = [seq_len] * batch_size
    total_q = sum(qo_lens)

    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_q, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_q, num_kv_heads, head_dim, dtype=dtype, device=device)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).numpy()),
        dtype=torch.int32,
        device=device,
    )

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device), backend="fa3"
    )
    wrapper.plan(
        cu_seqlens,
        cu_seqlens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        causal=True,
    )

    fi_time = bench_fn(lambda: wrapper.run(q, k, v))
    fa3_time = bench_fn(
        lambda: fa3_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, seq_len, seq_len, causal=True
        )
    )

    config_str = f"bs={batch_size}, seq={seq_len}, h={num_qo_heads}/{num_kv_heads}"
    diff = (fi_time - fa3_time) / fa3_time * 100
    fi_tflops = calc_tflops(
        batch_size, seq_len, num_qo_heads, head_dim, fi_time, causal=True
    )
    fa3_tflops = calc_tflops(
        batch_size, seq_len, num_qo_heads, head_dim, fa3_time, causal=True
    )
    print(
        f"{config_str:<35} {fi_time:<18.3f} {fa3_time:<15.3f} {diff:+.1f}%{'':5} {fi_tflops:<12.1f} {fa3_tflops:<12.1f}"
    )


# Variable sequence length tests
print("\n--- Variable Sequence Length Batch Prefill ---")
print(
    f"{'Config':<40} {'FlashInfer (ms)':<18} {'FA3 (ms)':<15} {'diff':<10} {'FI TFLOPS':<12} {'FA3 TFLOPS':<12}"
)
print("-" * 120)

varlen_configs = [
    # (batch_size, min_len, max_len, num_qo_heads, num_kv_heads)
    (16, 64, 512, 32, 8),
    (16, 128, 1024, 32, 8),
    (16, 256, 2048, 32, 8),
    (8, 512, 4096, 32, 8),
    (4, 1024, 8192, 32, 8),
    (4, 2048, 16384, 32, 8),
    (2, 4096, 32768, 32, 8),
    (16, 64, 512, 32, 32),
    (16, 128, 1024, 32, 32),
    (16, 256, 2048, 32, 32),
    (8, 512, 4096, 32, 32),
    (4, 1024, 8192, 32, 32),
    (4, 2048, 16384, 32, 32),
]

for batch_size, min_len, max_len, num_qo_heads, num_kv_heads in varlen_configs:
    seq_lens = [random.randint(min_len, max_len) for _ in range(batch_size)]
    total_tokens = sum(seq_lens)
    max_seqlen = max(seq_lens)

    q = torch.randn(total_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).numpy()),
        dtype=torch.int32,
        device=device,
    )

    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device), backend="fa3"
    )
    wrapper.plan(
        cu_seqlens,
        cu_seqlens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        causal=True,
    )

    fi_time = bench_fn(lambda: wrapper.run(q, k, v))
    fa3_time = bench_fn(
        lambda: fa3_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )
    )

    fi_tflops = calc_tflops_varlen(
        seq_lens, num_qo_heads, head_dim, fi_time, causal=True
    )
    fa3_tflops = calc_tflops_varlen(
        seq_lens, num_qo_heads, head_dim, fa3_time, causal=True
    )

    config_str = (
        f"bs={batch_size}, len=[{min_len}-{max_len}], h={num_qo_heads}/{num_kv_heads}"
    )
    diff = (fi_time - fa3_time) / fa3_time * 100
    print(
        f"{config_str:<40} {fi_time:<18.3f} {fa3_time:<15.3f} {diff:+.1f}%{'':5} {fi_tflops:<12.1f} {fa3_tflops:<12.1f}"
    )

# FP8 tests (FA3 only, FlashInfer FP8 FA3 backend has compilation issues)
print("\n--- FP8 Batch Prefill (FA3 only) ---")
print(f"{'Config':<35} {'FA3 (ms)':<15} {'FA3 TFLOPS':<12}")
print("-" * 70)

fp8_dtype = torch.float8_e4m3fn

fp8_configs = [
    (8, 2048, 32, 8),
    (8, 4096, 32, 8),
    (8, 8192, 32, 8),
    (4, 16384, 32, 8),
]

for batch_size, seq_len, num_qo_heads, num_kv_heads in fp8_configs:
    qo_lens = [seq_len] * batch_size
    total_q = sum(qo_lens)

    # Create FP8 tensors with proper scaling
    q_fp16 = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16 = torch.randn(
        total_q, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16 = torch.randn(
        total_q, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    q_fp8 = q_fp16.to(fp8_dtype)
    k_fp8 = k_fp16.to(fp8_dtype)
    v_fp8 = v_fp16.to(fp8_dtype)

    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).numpy()),
        dtype=torch.int32,
        device=device,
    )

    try:
        fa3_time = bench_fn(
            lambda: fa3_varlen_func(
                q_fp8,
                k_fp8,
                v_fp8,
                cu_seqlens,
                cu_seqlens,
                seq_len,
                seq_len,
                causal=True,
            )
        )
        fa3_tflops = calc_tflops(
            batch_size, seq_len, num_qo_heads, head_dim, fa3_time, causal=True
        )
        config_str = f"bs={batch_size}, seq={seq_len}, h={num_qo_heads}/{num_kv_heads}"
        print(f"{config_str:<35} {fa3_time:<15.3f} {fa3_tflops:<12.1f}")
    except Exception as e:
        config_str = f"bs={batch_size}, seq={seq_len}, h={num_qo_heads}/{num_kv_heads}"
        print(f"{config_str:<35} FA3 FP8 failed: {e}")
