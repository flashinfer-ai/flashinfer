import torch
import triton

import flashinfer


def bench_single_prefill(seq_len, num_heads, causal, head_dim):
    num_qo_heads = num_kv_heads = num_heads
    q = torch.randn(seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda")

    sm80_ms, sm90_ms = (
        triton.testing.do_bench(
            lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
                q, k, v, causal=causal, backend=backend
            ),
            warmup=100,
            rep=1000,
        )
        for backend in ["fa2", "fa3"]
    )

    q = torch.randn(
        seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    ).to(dtype=torch.float8_e4m3fn)
    k = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    ).to(dtype=torch.float8_e4m3fn)
    v = torch.randn(
        seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    ).to(dtype=torch.float8_e4m3fn)

    fp8_sm90_ms = triton.testing.do_bench(
        lambda: flashinfer.single_prefill_with_kv_cache_return_lse(
            q, k, v, causal=causal, backend="fa3", o_dtype=torch.half
        ),
        warmup=100,
        rep=1000,
    )

    def flops(ms):
        if causal:
            return seq_len * seq_len * num_qo_heads * head_dim * 2 / ms / 1e9
        else:
            return seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1e9

    print(
        f"bench_single_prefill (seq_len={seq_len}, num_heads={num_heads}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s, fa3-fp8: {flops(fp8_sm90_ms):.3f} TFLOPs/s"
    )


if __name__ == "__main__":
    for seq_len in [4096, 8192, 16384]:
        for num_heads in [24, 32]:
            for causal in [True, False]:
                for head_dim in [64, 128, 256]:
                    bench_single_prefill(seq_len, num_heads, causal, head_dim)
