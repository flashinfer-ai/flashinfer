import torch
import random

torch.manual_seed(42)
random.seed(42)
device = "cuda"
dtype = torch.float16

from flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func
from flash_attn_interface import flash_attn_with_kvcache as fa3_kvcache_func
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

# FP8 tests
print("\n--- FP8 Batch Prefill ---")
print(
    f"{'Config':<35} {'FlashInfer (ms)':<18} {'FA3 (ms)':<15} {'diff':<10} {'FI TFLOPS':<12} {'FA3 TFLOPS':<12}"
)
print("-" * 115)

fp8_dtype = torch.float8_e4m3fn


def per_head_symmetric_quant(x, quant_dtype):
    """Per-head symmetric quantization to FP8."""
    o_min_val, o_max_val = (
        (-448.0, 448.0) if quant_dtype == torch.float8_e4m3fn else (-57344, 57344)
    )
    x_max_val = x.abs().amax(dim=(0, 2)).to(dtype=torch.float32)
    s_out = torch.clamp(x_max_val / o_max_val, min=1e-6)
    s_out_broadcast = s_out.view(1, -1, 1)
    q_x_out = torch.clamp(x / s_out_broadcast, min=o_min_val, max=o_max_val).to(
        dtype=quant_dtype
    )
    return q_x_out, s_out


fp8_configs = [
    (8, 2048, 32, 8),
    (8, 4096, 32, 8),
    (8, 8192, 32, 8),
    (4, 16384, 32, 8),
]

for batch_size, seq_len, num_qo_heads, num_kv_heads in fp8_configs:
    qo_lens = [seq_len] * batch_size
    total_q = sum(qo_lens)

    # Create FP16 tensors first
    q_fp16 = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16 = torch.randn(
        total_q, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16 = torch.randn(
        total_q, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # Quantize to FP8 with proper scaling
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, fp8_dtype)
    k_fp8, s_k = per_head_symmetric_quant(k_fp16, fp8_dtype)
    v_fp8, s_v = per_head_symmetric_quant(v_fp16, fp8_dtype)

    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).numpy()),
        dtype=torch.int32,
        device=device,
    )

    config_str = f"bs={batch_size}, seq={seq_len}, h={num_qo_heads}/{num_kv_heads}"

    # Benchmark FlashInfer FP8
    fi_time = None
    fi_tflops = None
    try:
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
            q_data_type=fp8_dtype,
            kv_data_type=fp8_dtype,
            o_data_type=torch.float16,  # Output is FP16
        )
        fi_time = bench_fn(lambda: wrapper.run(q_fp8, k_fp8, v_fp8, s_q, s_k, s_v))
        fi_tflops = calc_tflops(
            batch_size, seq_len, num_qo_heads, head_dim, fi_time, causal=True
        )
    except Exception:
        fi_time = None
        fi_tflops = None

    # Benchmark FA3 FP8
    fa3_time = None
    fa3_tflops = None
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
    except Exception:
        fa3_time = None
        fa3_tflops = None

    if fi_time is not None and fa3_time is not None:
        diff = (fi_time - fa3_time) / fa3_time * 100
        print(
            f"{config_str:<35} {fi_time:<18.3f} {fa3_time:<15.3f} {diff:+.1f}%{'':5} {fi_tflops:<12.1f} {fa3_tflops:<12.1f}"
        )
    elif fi_time is not None:
        print(
            f"{config_str:<35} {fi_time:<18.3f} {'N/A':<15} {'N/A':<10} {fi_tflops:<12.1f} {'N/A':<12}"
        )
    elif fa3_time is not None:
        print(
            f"{config_str:<35} {'N/A':<18} {fa3_time:<15.3f} {'N/A':<10} {'N/A':<12} {fa3_tflops:<12.1f}"
        )
    else:
        print(
            f"{config_str:<35} {'N/A':<18} {'N/A':<15} {'N/A':<10} {'N/A':<12} {'N/A':<12}"
        )

# FP16 Paged KV Cache tests
print("\n--- FP16 Paged KV Cache Prefill ---")
print(
    f"{'Config':<45} {'FlashInfer (ms)':<18} {'FA3 (ms)':<15} {'diff':<10} {'FI TFLOPS':<12} {'FA3 TFLOPS':<12}"
)
print("-" * 125)

fp16_paged_configs = [
    # (batch_size, seq_len, num_qo_heads, num_kv_heads, page_size)
    # page_size=1
    (8, 2048, 32, 8, 1),
    (8, 4096, 32, 8, 1),
    (8, 8192, 32, 8, 1),
    (4, 16384, 32, 8, 1),
    # page_size=16
    (8, 2048, 32, 8, 16),
    (8, 4096, 32, 8, 16),
    (8, 8192, 32, 8, 16),
    (4, 16384, 32, 8, 16),
]

for batch_size, seq_len, num_qo_heads, num_kv_heads, page_size in fp16_paged_configs:
    qo_lens = [seq_len] * batch_size
    kv_lens = [seq_len] * batch_size
    total_q = sum(qo_lens)
    total_kv_pages = sum((kv_len + page_size - 1) // page_size for kv_len in kv_lens)

    # FP16 tensors
    q_fp16 = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )

    # Paged KV cache: (num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_data_fp16 = torch.randn(
        total_kv_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )

    # Page indices for each request
    kv_indptr = torch.tensor(
        [0]
        + [
            sum((kv_lens[i] + page_size - 1) // page_size for i in range(j + 1))
            for j in range(batch_size)
        ],
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.arange(total_kv_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        [((kv_len - 1) % page_size) + 1 for kv_len in kv_lens],
        dtype=torch.int32,
        device=device,
    )
    qo_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).numpy()),
        dtype=torch.int32,
        device=device,
    )

    config_str = f"bs={batch_size}, seq={seq_len}, h={num_qo_heads}/{num_kv_heads}, page={page_size}"

    # Benchmark FlashInfer FP16 Paged
    fi_time = None
    fi_tflops = None
    try:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            "NHD",
            backend="fa3",
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.float16,
        )
        fi_time = bench_fn(lambda: wrapper.run(q_fp16, kv_data_fp16))
        fi_tflops = calc_tflops(
            batch_size, seq_len, num_qo_heads, head_dim, fi_time, causal=True
        )
    except Exception as e:
        print(f"FlashInfer error: {e}")
        fi_time = None
        fi_tflops = None

    # FA3 paged attention
    fa3_time = None
    fa3_tflops = None
    try:
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        max_num_blocks_per_seq = num_pages_per_seq

        # Create FA3 paged KV cache: (num_blocks, page_size, num_kv_heads, head_dim)
        k_cache_fa3 = kv_data_fp16[
            :, 0, :, :, :
        ]  # (num_pages, page_size, num_kv_heads, head_dim)
        v_cache_fa3 = kv_data_fp16[:, 1, :, :, :]

        # Create page table: (batch_size, max_num_blocks_per_seq)
        page_table = torch.zeros(
            batch_size, max_num_blocks_per_seq, dtype=torch.int32, device=device
        )
        for b in range(batch_size):
            start_page = b * num_pages_per_seq
            for p in range(num_pages_per_seq):
                page_table[b, p] = start_page + p

        # Q for FA3: (batch_size, seq_len, num_qo_heads, head_dim)
        q_fa3 = q_fp16.reshape(batch_size, seq_len, num_qo_heads, head_dim)

        # cache_seqlens
        cache_seqlens = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        fa3_time = bench_fn(
            lambda: fa3_kvcache_func(
                q_fa3,
                k_cache_fa3,
                v_cache_fa3,
                cache_seqlens=cache_seqlens,
                page_table=page_table,
                causal=True,
            )
        )
        fa3_tflops = calc_tflops(
            batch_size, seq_len, num_qo_heads, head_dim, fa3_time, causal=True
        )
    except Exception as e:
        print(f"FA3 paged error: {e}")
        fa3_time = None
        fa3_tflops = None

    if fi_time is not None and fa3_time is not None:
        diff = (fi_time - fa3_time) / fa3_time * 100
        print(
            f"{config_str:<45} {fi_time:<18.3f} {fa3_time:<15.3f} {diff:>+.1f}%{'':<4} {fi_tflops:<12.1f} {fa3_tflops:<12.1f}"
        )
    elif fi_time is not None:
        print(
            f"{config_str:<45} {fi_time:<18.3f} {'N/A':<15} {'N/A':<10} {fi_tflops:<12.1f} {'N/A':<12}"
        )
    else:
        print(
            f"{config_str:<45} {'N/A':<18} {'N/A':<15} {'N/A':<10} {'N/A':<12} {'N/A':<12}"
        )

# FP8 Paged KV Cache tests
print("\n--- FP8 Paged KV Cache Prefill ---")
print(
    f"{'Config':<45} {'FlashInfer (ms)':<18} {'FA3 (ms)':<15} {'diff':<10} {'FI TFLOPS':<12} {'FA3 TFLOPS':<12}"
)
print("-" * 125)

fp8_paged_configs = [
    # (batch_size, seq_len, num_qo_heads, num_kv_heads, page_size)
    # page_size=1
    (8, 2048, 32, 8, 1),
    (8, 4096, 32, 8, 1),
    (8, 8192, 32, 8, 1),
    (4, 16384, 32, 8, 1),
    # page_size=16
    (8, 2048, 32, 8, 16),
    (8, 4096, 32, 8, 16),
    (8, 8192, 32, 8, 16),
    (4, 16384, 32, 8, 16),
]

for batch_size, seq_len, num_qo_heads, num_kv_heads, page_size in fp8_paged_configs:
    qo_lens = [seq_len] * batch_size
    kv_lens = [seq_len] * batch_size
    total_q = sum(qo_lens)
    total_kv_pages = sum((kv_len + page_size - 1) // page_size for kv_len in kv_lens)

    # Create FP16 tensors first
    q_fp16 = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )

    # Paged KV cache: (num_pages, 2, page_size, num_kv_heads, head_dim)
    kv_data_fp16 = torch.randn(
        total_kv_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )

    # Page indices for each request
    kv_indptr = torch.tensor(
        [0]
        + [
            sum((kv_lens[i] + page_size - 1) // page_size for i in range(j + 1))
            for j in range(batch_size)
        ],
        dtype=torch.int32,
        device=device,
    )
    kv_indices = torch.arange(total_kv_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        [((kv_len - 1) % page_size) + 1 for kv_len in kv_lens],
        dtype=torch.int32,
        device=device,
    )
    qo_indptr = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(qo_lens), 0).numpy()),
        dtype=torch.int32,
        device=device,
    )

    # Quantize Q to FP8
    q_fp8, s_q = per_head_symmetric_quant(q_fp16, fp8_dtype)

    # For paged KV, we need to quantize differently
    k_fp16 = kv_data_fp16[:, 0, :, :, :].reshape(-1, num_kv_heads, head_dim)
    v_fp16 = kv_data_fp16[:, 1, :, :, :].reshape(-1, num_kv_heads, head_dim)
    k_fp8, s_k = per_head_symmetric_quant(k_fp16, fp8_dtype)
    v_fp8, s_v = per_head_symmetric_quant(v_fp16, fp8_dtype)

    # Reshape back to paged format
    kv_data_fp8 = torch.stack(
        [
            k_fp8.reshape(total_kv_pages, page_size, num_kv_heads, head_dim),
            v_fp8.reshape(total_kv_pages, page_size, num_kv_heads, head_dim),
        ],
        dim=1,
    )

    config_str = f"bs={batch_size}, seq={seq_len}, h={num_qo_heads}/{num_kv_heads}, page={page_size}"

    # Benchmark FlashInfer FP8 Paged
    fi_time = None
    fi_tflops = None
    try:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
            "NHD",
            backend="fa3",
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=fp8_dtype,
            kv_data_type=fp8_dtype,
            o_data_type=torch.float16,
        )
        fi_time = bench_fn(lambda: wrapper.run(q_fp8, kv_data_fp8, s_q, s_k, s_v))
        fi_tflops = calc_tflops(
            batch_size, seq_len, num_qo_heads, head_dim, fi_time, causal=True
        )
    except Exception as e:
        print(f"FlashInfer error: {e}")
        fi_time = None
        fi_tflops = None

    # FA3 paged attention
    fa3_time = None
    fa3_tflops = None
    try:
        # FA3 paged format: (num_blocks, page_size, num_kv_heads, head_dim)
        num_pages_per_seq = (seq_len + page_size - 1) // page_size
        max_num_blocks_per_seq = num_pages_per_seq

        # Create FA3 paged KV cache
        k_cache_fa3 = k_fp8.reshape(total_kv_pages, page_size, num_kv_heads, head_dim)
        v_cache_fa3 = v_fp8.reshape(total_kv_pages, page_size, num_kv_heads, head_dim)

        # Create page table: (batch_size, max_num_blocks_per_seq)
        page_table = torch.zeros(
            batch_size, max_num_blocks_per_seq, dtype=torch.int32, device=device
        )
        for b in range(batch_size):
            start_page = b * num_pages_per_seq
            for p in range(num_pages_per_seq):
                page_table[b, p] = start_page + p

        # Q for FA3: (batch_size, seq_len, num_qo_heads, head_dim)
        q_fa3 = q_fp8.reshape(batch_size, seq_len, num_qo_heads, head_dim)

        # cache_seqlens: actual sequence lengths
        cache_seqlens = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, device=device
        )

        # descale tensors for FP8
        # FA3 expects per-head descale: shape (batch_size, num_kv_heads) for GQA
        k_descale_fa3 = s_k.squeeze().unsqueeze(0).expand(batch_size, -1).contiguous()
        v_descale_fa3 = s_v.squeeze().unsqueeze(0).expand(batch_size, -1).contiguous()
        # q_descale should also be (batch_size, num_kv_heads) - one scale per kv head group
        q_descale_fa3 = (
            s_q.squeeze()
            .reshape(num_kv_heads, num_qo_heads // num_kv_heads)
            .mean(dim=1)
        )
        q_descale_fa3 = q_descale_fa3.unsqueeze(0).expand(batch_size, -1).contiguous()

        fa3_time = bench_fn(
            lambda: fa3_kvcache_func(
                q_fa3,
                k_cache_fa3,
                v_cache_fa3,
                cache_seqlens=cache_seqlens,
                page_table=page_table,
                q_descale=q_descale_fa3,
                k_descale=k_descale_fa3,
                v_descale=v_descale_fa3,
                causal=True,
            )
        )
        fa3_tflops = calc_tflops(
            batch_size, seq_len, num_qo_heads, head_dim, fa3_time, causal=True
        )
    except Exception as e:
        print(f"FA3 paged error: {e}")
        fa3_time = None
        fa3_tflops = None

    if fi_time is not None and fa3_time is not None:
        diff = (fi_time - fa3_time) / fa3_time * 100
        print(
            f"{config_str:<45} {fi_time:<18.3f} {fa3_time:<15.3f} {diff:>+.1f}%{'':<4} {fi_tflops:<12.1f} {fa3_tflops:<12.1f}"
        )
    elif fi_time is not None:
        print(
            f"{config_str:<45} {fi_time:<18.3f} {'N/A':<15} {'N/A':<10} {fi_tflops:<12.1f} {'N/A':<12}"
        )
    else:
        print(
            f"{config_str:<45} {'N/A':<18} {'N/A':<15} {'N/A':<10} {'N/A':<12} {'N/A':<12}"
        )
