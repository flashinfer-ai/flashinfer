import argparse
from typing import cast

import torch
from triton.testing import do_bench

import flashinfer


def generate_cos_sin_f32_cache(max_seq_len, head_dim, theta=1e4):
    position = torch.arange(max_seq_len).float().unsqueeze(1)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.cat([freqs, freqs], dim=-1).contiguous()
    
    args = position * freqs
    sin_cache = torch.sin(args)
    cos_cache = torch.cos(args)
    return cos_cache, sin_cache


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 19, 99, 128])
    parser.add_argument("--append-len", nargs="+", type=int, default=[1, 128, 1024])
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    eps = 1e-6
    dtype = torch.float16
    num_qo_heads = args.num_qo_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim

    # Loop over each combination of batch_size, hidden_size, and dtype
    for batch_size in args.batch_sizes:
        for append_len in args.append_len:
            for use_cos_sin_cache in [False, True]:
                # Define tensors with the correct dtype

                q = torch.randn(
                    (batch_size * append_len, num_qo_heads, args.head_dim),
                    dtype=dtype,
                    device="cuda",
                )
                k = torch.randn(
                    (batch_size * append_len, num_kv_heads, args.head_dim),
                    dtype=dtype,
                    device="cuda",
                )
                pos_ids = torch.repeat_interleave(
                    torch.arange(append_len, dtype=torch.int32, device=q.device),
                    batch_size,
                )
                cos_cache, sin_cache = generate_cos_sin_f32_cache(4096, head_dim)
                cos_cache = cos_cache.to(q.device)
                sin_cache = sin_cache.to(q.device)

                @torch.cuda.nvtx.range(
                    f"apply_rope batch_size={batch_size}, append_len={append_len}, num_qo_heads={num_qo_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
                )
                def fn() -> None:
                    if use_cos_sin_cache:
                        flashinfer.apply_rope_with_cos_sin_cache(
                            q, k, cos_cache, sin_cache, pos_ids
                        )
                    else:
                        flashinfer.apply_rope_pos_ids(q, k, pos_ids)

                # Run benchmarking
                latency_ms = cast(float, do_bench(fn))
                throughput = (
                    q.numel() * q.element_size() * 2 + k.numel() * k.element_size() * 2
                ) / (latency_ms * 1e-3)
                print(
                    f"batch_size: {batch_size:3},",
                    f"append_len: {append_len:5},",
                    f"num_qo_heads: {num_qo_heads:5},",
                    f"num_kv_heads: {num_kv_heads:5},",
                    f"head_dim: {head_dim:5},",
                    f"use_cos_sin_cache: {use_cos_sin_cache},",
                    f"latency: {latency_ms*1e3:2.0f}us,",
                    f"throughput: {throughput*1e-9:7.3f}GB/s",
                )

        print("---")

    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
