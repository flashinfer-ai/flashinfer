"""Benchmark: MIS v1 (delimited) vs MIS v2 (delimiterless).

Measures GPU attention kernel time for multi-item scoring with paged KV cache.

Both V1 and V2 exclude prefix tokens from Q (prefix lives only in KV cache).
The only difference is delimiter tokens: V1 includes them, V2 does not.

The attention kernel speedup from removing delimiters is modest. The primary
benefit of MIS v2 is reducing total sequence length through the entire model
(attention + MLP + layernorm at every layer), which this benchmark does not
measure. The savings are most significant when the delimiter-to-item-token
ratio is high (many short items).
"""

import torch
import numpy as np

from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


def bench_gpu_time(fn, warmup=10, repeat=100):
    """Measure GPU kernel time in ms using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def build_mis_v1_inputs(batch_size, prefix_len, item_lens):
    """Build MIS v1 inputs: Q = delimiters + item tokens (no prefix in Q).

    KV = prefix + delimiters + item tokens.
    """
    token_pos = []
    for item_len in item_lens:
        token_pos.append(0)  # delimiter
        for j in range(1, item_len + 1):
            token_pos.append(j)

    qo_len = len(token_pos)  # delimiters + items only
    kv_len = prefix_len + qo_len
    max_item_len = max(item_lens)

    token_pos_tensor = torch.tensor(token_pos * batch_size, dtype=torch.uint16).cuda()

    return (
        qo_len,
        kv_len,
        torch.tensor([prefix_len] * batch_size, dtype=torch.uint32).cuda(),
        token_pos_tensor,
        len(token_pos),
        torch.tensor([max_item_len] * batch_size, dtype=torch.uint16).cuda(),
    )


def build_mis_v2_inputs(batch_size, prefix_len, item_lens):
    """Build MIS v2 inputs: Q = item tokens only (no delimiters, no prefix in Q).

    KV = prefix + item tokens.
    """
    total_items_tokens = sum(item_lens)
    qo_len = total_items_tokens
    kv_len = prefix_len + total_items_tokens

    offsets = [0]
    for il in item_lens:
        offsets.append(offsets[-1] + il)

    offsets_replicated = offsets * batch_size

    return (
        qo_len,
        kv_len,
        torch.tensor([prefix_len] * batch_size, dtype=torch.uint32).cuda(),
        torch.tensor(offsets_replicated, dtype=torch.uint32).cuda(),
        len(offsets),
    )


def run_benchmark(
    batch_size, prefix_len, item_lens,
    num_qo_heads, num_kv_heads, head_dim, page_size,
    warmup=10, repeat=100,
):
    """Run benchmark comparing MIS v1 and MIS v2 (fair: both exclude prefix from Q)."""
    device = "cuda"
    results = {}

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # ---- MIS v1 (with delimiters, no prefix in Q) ----
    (
        qo_len_v1, kv_len_v1, prefix_len_ptr_v1,
        token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr,
    ) = build_mis_v1_inputs(batch_size, prefix_len, item_lens)

    q_v1 = torch.randn(batch_size * qo_len_v1, num_qo_heads, head_dim, device=device, dtype=torch.float16)
    num_pages_v1 = (kv_len_v1 + page_size - 1) // page_size
    total_pages_v1 = num_pages_v1 * batch_size
    kv_data_v1 = torch.randn(total_pages_v1, 2, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    q_indptr_v1 = (torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len_v1).cuda()
    kv_indptr_v1 = (torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_v1).cuda()
    kv_indices_v1 = torch.arange(0, total_pages_v1, dtype=torch.int32, device=device)
    kv_last_page_v1 = torch.full((batch_size,), (kv_len_v1 - 1) % page_size + 1, dtype=torch.int32, device=device)

    wrapper_v1 = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper_v1.plan(
        q_indptr_v1, kv_indptr_v1, kv_indices_v1, kv_last_page_v1,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True, pos_encoding_mode="NONE",
        prefix_len_ptr=prefix_len_ptr_v1,
        token_pos_in_items_ptr=token_pos_in_items_ptr,
        token_pos_in_items_len=token_pos_in_items_len,
        max_item_len_ptr=max_item_len_ptr,
    )
    times_v1 = bench_gpu_time(lambda: wrapper_v1.run(q_v1, kv_data_v1), warmup=warmup, repeat=repeat)
    results["v1"] = times_v1

    # ---- MIS v2 (no delimiters, no prefix in Q) ----
    (
        qo_len_v2, kv_len_v2, prefix_len_ptr_v2,
        item_offsets, item_offsets_len,
    ) = build_mis_v2_inputs(batch_size, prefix_len, item_lens)

    q_v2 = torch.randn(batch_size * qo_len_v2, num_qo_heads, head_dim, device=device, dtype=torch.float16)
    num_pages_v2 = (kv_len_v2 + page_size - 1) // page_size
    total_pages_v2 = num_pages_v2 * batch_size
    kv_data_v2 = torch.randn(total_pages_v2, 2, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    q_indptr_v2 = (torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len_v2).cuda()
    kv_indptr_v2 = (torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_v2).cuda()
    kv_indices_v2 = torch.arange(0, total_pages_v2, dtype=torch.int32, device=device)
    kv_last_page_v2 = torch.full((batch_size,), (kv_len_v2 - 1) % page_size + 1, dtype=torch.int32, device=device)

    wrapper_v2 = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper_v2.plan(
        q_indptr_v2, kv_indptr_v2, kv_indices_v2, kv_last_page_v2,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True, pos_encoding_mode="NONE",
        prefix_len_ptr=prefix_len_ptr_v2,
        item_offsets=item_offsets,
        item_offsets_len=item_offsets_len,
    )
    times_v2 = bench_gpu_time(lambda: wrapper_v2.run(q_v2, kv_data_v2), warmup=warmup, repeat=repeat)
    results["v2"] = times_v2

    return results, {
        "qo_len_v1": qo_len_v1, "kv_len_v1": kv_len_v1,
        "qo_len_v2": qo_len_v2, "kv_len_v2": kv_len_v2,
        "num_delimiters": len(item_lens),
    }


def main():
    print("=" * 80)
    print("MIS v1 (delimited) vs MIS v2 (delimiterless) — Attention Kernel Benchmark")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("NOTE: Both V1 and V2 exclude prefix from Q. The only difference is")
    print("delimiter tokens. This measures attention kernel time only — the full")
    print("model benefit (MLP, layernorm savings) from shorter sequences is not")
    print("captured here.")
    print()

    # Configs focus on high delimiter-to-item ratios where v2 matters
    configs = [
        # High delimiter ratio — v2's sweet spot
        (256, [3] * 100,   "256 prefix, 100x3 items (25% delims)"),
        (256, [3] * 200,   "256 prefix, 200x3 items (25% delims)"),
        (256, [3] * 500,   "256 prefix, 500x3 items (25% delims)"),
        (512, [5] * 100,   "512 prefix, 100x5 items (17% delims)"),
        (512, [5] * 200,   "512 prefix, 200x5 items (17% delims)"),
        (256, [10] * 100,  "256 prefix, 100x10 items (9% delims)"),
        (256, [10] * 200,  "256 prefix, 200x10 items (9% delims)"),
        # Low delimiter ratio — minimal difference expected
        (512, [64] * 20,   "512 prefix, 20x64 items (1.5% delims)"),
        (1024, [128] * 10, "1024 prefix, 10x128 items (0.8% delims)"),
        (4096, [512] * 4,  "4096 prefix, 4x512 items (0.2% delims)"),
    ]

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    batch_sizes = [1, 10]
    repeat = 100

    print(f"Config: {num_qo_heads} QO heads, {num_kv_heads} KV heads, "
          f"{head_dim} head_dim, page_size={page_size}")
    print()

    for batch_size in batch_sizes:
        print(f"{'=' * 40} batch_size={batch_size} {'=' * 40}")
        header = (f"{'Config':<45} {'V1 qo':<8} {'V2 qo':<8} {'Delims':<8} "
                  f"{'V1 (ms)':<12} {'V2 (ms)':<12} {'Speedup':<10} {'Delim%':<8}")
        print(header)
        print("-" * len(header))

        for prefix_len, item_lens, desc in configs:
            try:
                results, lens = run_benchmark(
                    batch_size, prefix_len, item_lens,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    warmup=10, repeat=repeat,
                )

                mean_v1 = np.mean(results["v1"])
                mean_v2 = np.mean(results["v2"])
                speedup = mean_v1 / mean_v2
                delim_pct = lens["num_delimiters"] / lens["qo_len_v1"] * 100

                print(
                    f"{desc:<45} {lens['qo_len_v1']:<8} {lens['qo_len_v2']:<8} "
                    f"{lens['num_delimiters']:<8} {mean_v1:<12.4f} {mean_v2:<12.4f} "
                    f"{speedup:<10.2f}x {delim_pct:<8.1f}%"
                )
            except Exception as e:
                import traceback
                print(f"{desc:<45} ERROR: {e}")
                traceback.print_exc()

        print()

    print("Legend:")
    print("  V1 qo/V2 qo: Q sequence length (V1 includes delimiters, V2 does not)")
    print("  Delims: number of delimiter tokens saved by V2")
    print("  Delim%: delimiters as fraction of V1's qo_len")
    print("  Speedup: V1 time / V2 time (>1 = V2 faster)")
    print()
    print("  The attention kernel speedup is modest. The primary benefit of MIS v2")
    print("  is reducing total sequence length through the full transformer stack")
    print("  (attention + MLP + layernorm), which compounds across layers.")


if __name__ == "__main__":
    main()
