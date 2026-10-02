"""Benchmark fused QK RMSNorm/RoPE/paged append against stable primitives."""

from __future__ import annotations

import argparse
import statistics

import torch

import flashinfer
import flashinfer.norm
from flashinfer.testing import bench_gpu_time


def _median_ms(fn, enable_cupti: bool) -> float:
    return statistics.median(
        bench_gpu_time(
            fn,
            enable_cupti=enable_cupti,
            dry_run_iters=10,
            repeat_iters=30,
            cold_l2_cache=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--qo-len", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--cuda-events", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    q_heads, kv_heads, head_dim = 8, 1, 128
    rows = args.batch_size * args.qo_len
    final_len = args.context_len + args.qo_len
    pages_per_request = (final_len + args.page_size - 1) // args.page_size
    num_pages = args.batch_size * pages_per_request
    width = (q_heads + 2 * kv_heads) * head_dim

    qkv = torch.randn(rows, width, device=device, dtype=torch.bfloat16)
    q = qkv[:, : q_heads * head_dim].view(rows, q_heads, head_dim)
    k = qkv[:, q_heads * head_dim : (q_heads + kv_heads) * head_dim].view(
        rows, kv_heads, head_dim
    )
    v = qkv[:, (q_heads + kv_heads) * head_dim :].view(rows, kv_heads, head_dim)
    # The imported kernel consumes FP32 norm weights, while the stable
    # ``rmsnorm`` primitive requires weights to match its BF16 input. Use the
    # same BF16-rounded values in both paths.
    q_weight_bf16 = torch.rand(head_dim, device=device, dtype=torch.bfloat16) + 0.5
    k_weight_bf16 = torch.rand(head_dim, device=device, dtype=torch.bfloat16) + 0.5
    q_weight = q_weight_bf16.float()
    k_weight = k_weight_bf16.float()
    q_indptr = (
        torch.arange(args.batch_size + 1, device=device, dtype=torch.int32)
        * args.qo_len
    )
    seq_lens = torch.full(
        (args.batch_size,), final_len, device=device, dtype=torch.int32
    )
    page_indices = torch.arange(num_pages, device=device, dtype=torch.int32).view(
        args.batch_size, pages_per_request
    )
    kv_indptr = (
        torch.arange(args.batch_size + 1, device=device, dtype=torch.int32)
        * pages_per_request
    )
    kv_last_page_len = torch.full(
        (args.batch_size,),
        (final_len - 1) % args.page_size + 1,
        device=device,
        dtype=torch.int32,
    )
    batch_indices = (
        torch.arange(args.batch_size, device=device)
        .repeat_interleave(args.qo_len)
        .int()
    )
    positions = (
        torch.arange(args.qo_len, device=device)
        .repeat(args.batch_size)
        .add(args.context_len)
        .int()
    )
    pos_ids = positions
    inv_freq = 1.0 / (
        1e4
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(
        torch.arange(final_len, device=device, dtype=torch.float32), inv_freq
    )
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    fused_cache = (
        torch.empty(
            num_pages,
            args.page_size,
            kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        ),
        torch.empty(
            num_pages,
            args.page_size,
            kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        ),
    )
    baseline_cache = tuple(torch.empty_like(x) for x in fused_cache)
    fused_q = torch.empty_like(q)
    norm_q = torch.empty_like(q)
    norm_k = torch.empty_like(k)

    def fused() -> None:
        flashinfer.rope.fused_qk_rmsnorm_rope_append_paged_kv_cache(
            qkv,
            cos_sin,
            seq_lens,
            q_indptr,
            page_indices,
            fused_cache,
            args.qo_len > 1,
            q_weight,
            k_weight,
            2,
            out_q=fused_q,
        )

    def baseline() -> None:
        flashinfer.norm.rmsnorm(q, q_weight_bf16, out=norm_q)
        flashinfer.norm.rmsnorm(k, k_weight_bf16, out=norm_k)
        flashinfer.rope.apply_rope_pos_ids_inplace(norm_q, norm_k, pos_ids)
        flashinfer.append_paged_kv_cache(
            norm_k,
            v,
            batch_indices,
            positions,
            baseline_cache,
            page_indices.flatten(),
            kv_indptr,
            kv_last_page_len,
        )

    fused()
    baseline()
    torch.cuda.synchronize()
    torch.testing.assert_close(fused_q, norm_q, atol=2e-2, rtol=2e-2)
    for batch, position in zip(batch_indices.tolist(), positions.tolist(), strict=True):
        page = int(page_indices[batch, position // args.page_size])
        offset = position % args.page_size
        torch.testing.assert_close(
            fused_cache[0][page, offset],
            baseline_cache[0][page, offset],
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            fused_cache[1][page, offset],
            baseline_cache[1][page, offset],
            atol=0,
            rtol=0,
        )

    enable_cupti = not args.cuda_events
    fused_ms = _median_ms(fused, enable_cupti)
    baseline_ms = _median_ms(baseline, enable_cupti)
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(
        f"shape=B{args.batch_size},Q{args.qo_len},Hq{q_heads},Hkv{kv_heads},"
        f"D{head_dim},context={args.context_len},page={args.page_size}"
    )
    print(f"fused_ms={fused_ms:.6f}")
    print(f"stable_primitive_pipeline_ms={baseline_ms:.6f}")
    print(f"speedup={baseline_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
