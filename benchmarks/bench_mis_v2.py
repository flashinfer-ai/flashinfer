"""Benchmark: MIS v1 (delimited) vs MIS v2 (delimiterless) vs custom_mask baseline.

Measures GPU kernel time for multi-item scoring attention with paged KV cache.
"""

import time
import torch
import numpy as np

import flashinfer
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


def bench_gpu_time(fn, warmup=5, repeat=50):
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


def build_mis_v1_inputs(batch_size, prefix_len, item_lens, delimiter=True):
    """Build MIS v1 inputs: sequence includes prefix + [delim item]* tokens."""
    # V1: prefix tokens + (delimiter + item_tokens) for each item
    # token_pos_in_items: 0 for delimiter, 1..N for item tokens, and prefix tokens
    total_items_with_delim = sum(1 + il for il in item_lens)  # 1 delimiter per item
    qo_len = prefix_len + total_items_with_delim
    kv_len = qo_len

    # Build token_pos_in_items for the full sequence
    token_pos = []
    # Prefix tokens
    for i in range(prefix_len):
        token_pos.append(i)
    # Items with delimiters
    for item_len in item_lens:
        token_pos.append(0)  # delimiter
        for j in range(1, item_len + 1):
            token_pos.append(j)

    max_item_len = max(item_lens)

    return (
        qo_len,
        kv_len,
        torch.tensor([prefix_len], dtype=torch.uint32).cuda(),
        torch.tensor(token_pos, dtype=torch.uint16).cuda(),
        len(token_pos),
        torch.tensor([max_item_len], dtype=torch.uint16).cuda(),
    )


def build_mis_v2_inputs(batch_size, prefix_len, item_lens):
    """Build MIS v2 inputs: no delimiters, uses item_offsets CSR format."""
    total_items_tokens = sum(item_lens)
    qo_len = total_items_tokens  # only item tokens in Q (prefix in KV cache)
    kv_len = prefix_len + total_items_tokens

    # item_offsets: CSR-style [0, len1, len1+len2, ...]
    offsets = [0]
    for il in item_lens:
        offsets.append(offsets[-1] + il)

    return (
        qo_len,
        kv_len,
        torch.tensor([prefix_len], dtype=torch.uint32).cuda(),
        torch.tensor(offsets, dtype=torch.uint32).cuda(),
        len(offsets),
    )


def build_custom_mask_v2(prefix_len, item_lens, qo_len, kv_len):
    """Build a custom mask equivalent to MIS v2 for reference benchmarking."""
    offsets = [0]
    for il in item_lens:
        offsets.append(offsets[-1] + il)

    custom_mask = torch.zeros(qo_len, kv_len, dtype=torch.bool, device="cuda")
    for q_idx in range(qo_len):
        kv_pos = prefix_len + q_idx
        # Find which item this Q belongs to
        for j in range(len(offsets) - 1):
            if offsets[j] <= q_idx < offsets[j + 1]:
                item_start_rel = offsets[j]
                break
        # Attend to all prefix tokens
        custom_mask[q_idx, :prefix_len] = True
        # Attend to same item tokens, causally
        kv_item_start = prefix_len + item_start_rel
        custom_mask[q_idx, kv_item_start : kv_pos + 1] = True

    return custom_mask.unsqueeze(0).reshape(-1)


def run_benchmark(
    batch_size,
    prefix_len,
    item_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    warmup=10,
    repeat=100,
):
    """Run benchmark comparing MIS v1, MIS v2, and custom_mask."""
    device = "cuda"
    results = {}

    # ---- MIS v2 (delimiterless) ----
    (
        qo_len_v2,
        kv_len_v2,
        prefix_len_ptr_v2,
        item_offsets,
        item_offsets_len,
    ) = build_mis_v2_inputs(batch_size, prefix_len, item_lens)

    q_v2 = torch.randn(batch_size * qo_len_v2, num_qo_heads, head_dim, device=device, dtype=torch.float16)
    num_pages_v2 = (kv_len_v2 + page_size - 1) // page_size
    total_pages_v2 = num_pages_v2 * batch_size
    kv_data_v2 = torch.randn(total_pages_v2, 2, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    q_indptr_v2 = (torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len_v2).cuda()
    kv_indptr_v2 = (torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_v2).cuda()
    kv_indices_v2 = torch.arange(0, total_pages_v2, dtype=torch.int32, device=device)
    kv_last_page_v2 = torch.full((batch_size,), (kv_len_v2 - 1) % page_size + 1, dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_v2 = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper_v2.plan(
        q_indptr_v2, kv_indptr_v2, kv_indices_v2, kv_last_page_v2,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        pos_encoding_mode="NONE",
        prefix_len_ptr=prefix_len_ptr_v2,
        item_offsets=item_offsets,
        item_offsets_len=item_offsets_len,
    )
    times_v2 = bench_gpu_time(lambda: wrapper_v2.run(q_v2, kv_data_v2), warmup=warmup, repeat=repeat)
    results["MIS_v2_delimiterless"] = times_v2

    # ---- MIS v1 (with delimiters) ----
    (
        qo_len_v1,
        kv_len_v1,
        prefix_len_ptr_v1,
        token_pos_in_items_ptr,
        token_pos_in_items_len,
        max_item_len_ptr,
    ) = build_mis_v1_inputs(batch_size, prefix_len, item_lens)

    q_v1 = torch.randn(batch_size * qo_len_v1, num_qo_heads, head_dim, device=device, dtype=torch.float16)
    num_pages_v1 = (kv_len_v1 + page_size - 1) // page_size
    total_pages_v1 = num_pages_v1 * batch_size
    kv_data_v1 = torch.randn(total_pages_v1, 2, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    q_indptr_v1 = (torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len_v1).cuda()
    kv_indptr_v1 = (torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_v1).cuda()
    kv_indices_v1 = torch.arange(0, total_pages_v1, dtype=torch.int32, device=device)
    kv_last_page_v1 = torch.full((batch_size,), (kv_len_v1 - 1) % page_size + 1, dtype=torch.int32, device=device)

    wrapper_v1 = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper_v1.plan(
        q_indptr_v1, kv_indptr_v1, kv_indices_v1, kv_last_page_v1,
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        pos_encoding_mode="NONE",
        prefix_len_ptr=prefix_len_ptr_v1,
        token_pos_in_items_ptr=token_pos_in_items_ptr,
        token_pos_in_items_len=token_pos_in_items_len,
        max_item_len_ptr=max_item_len_ptr,
    )
    times_v1 = bench_gpu_time(lambda: wrapper_v1.run(q_v1, kv_data_v1), warmup=warmup, repeat=repeat)
    results["MIS_v1_delimited"] = times_v1

    # ---- Custom mask baseline (equivalent to MIS v2 mask, but using generic custom_mask path) ----
    # single_prefill only supports batch_size=1
    if batch_size == 1:
        custom_mask = build_custom_mask_v2(prefix_len, item_lens, qo_len_v2, kv_len_v2)

        # For custom mask, we use single_prefill (ragged) which supports custom_mask
        # Reconstruct full KV from paged format for single_prefill
        k_full = kv_data_v2[:num_pages_v2, 0].reshape(-1, num_kv_heads, head_dim)[:kv_len_v2]
        v_full = kv_data_v2[:num_pages_v2, 1].reshape(-1, num_kv_heads, head_dim)[:kv_len_v2]
        q_cm = q_v2[:qo_len_v2]

        times_cm = bench_gpu_time(
            lambda: flashinfer.prefill.single_prefill_with_kv_cache(
                q_cm, k_full, v_full, causal=True, custom_mask=custom_mask,
            ),
            warmup=warmup,
            repeat=repeat,
        )
        results["custom_mask_baseline"] = times_cm

    return results, {
        "qo_len_v1": qo_len_v1, "kv_len_v1": kv_len_v1,
        "qo_len_v2": qo_len_v2, "kv_len_v2": kv_len_v2,
    }


def main():
    print("=" * 80)
    print("MIS v1 (delimited) vs MIS v2 (delimiterless) vs Custom Mask Benchmark")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    configs = [
        # (prefix_len, item_lens, description)
        (512, [64] * 5, "512 prefix, 5x64 items"),
        (1024, [128] * 10, "1024 prefix, 10x128 items"),
        (2048, [256] * 5, "2048 prefix, 5x256 items"),
        (4096, [512] * 4, "4096 prefix, 4x512 items"),
        (512, [32, 64, 128, 256], "512 prefix, mixed items [32,64,128,256]"),
        (1024, [64] * 20, "1024 prefix, 20x64 items"),
    ]

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    batch_sizes = [1, 10, 100]
    repeat = 100

    print(f"Config: {num_qo_heads} QO heads, {num_kv_heads} KV heads, {head_dim} head_dim, page_size={page_size}")
    print()

    for batch_size in batch_sizes:
        print(f"{'=' * 40} batch_size={batch_size} {'=' * 40}")
        if batch_size == 1:
            header = f"{'Config':<45} {'QO/KV len':<20} {'MIS v1 (ms)':<15} {'MIS v2 (ms)':<15} {'CustomMask (ms)':<18} {'v2 vs v1':<12} {'v2 vs CM':<12}"
        else:
            header = f"{'Config':<45} {'QO/KV len':<20} {'MIS v1 (ms)':<15} {'MIS v2 (ms)':<15} {'v2 vs v1':<12}"
        print(header)
        print("-" * len(header))

        for prefix_len, item_lens, desc in configs:
            try:
                results, lens = run_benchmark(
                    batch_size, prefix_len, item_lens,
                    num_qo_heads, num_kv_heads, head_dim, page_size,
                    warmup=10, repeat=repeat,
                )

                mean_v1 = np.mean(results["MIS_v1_delimited"])
                mean_v2 = np.mean(results["MIS_v2_delimiterless"])

                speedup_v2_vs_v1 = mean_v1 / mean_v2

                lens_str = f"{lens['qo_len_v2']}/{lens['kv_len_v2']}"

                if batch_size == 1:
                    mean_cm = np.mean(results["custom_mask_baseline"])
                    speedup_v2_vs_cm = mean_cm / mean_v2
                    print(
                        f"{desc:<45} {lens_str:<20} "
                        f"{mean_v1:<15.4f} {mean_v2:<15.4f} {mean_cm:<18.4f} "
                        f"{speedup_v2_vs_v1:<12.2f}x {speedup_v2_vs_cm:<12.2f}x"
                    )
                else:
                    print(
                        f"{desc:<45} {lens_str:<20} "
                        f"{mean_v1:<15.4f} {mean_v2:<15.4f} "
                        f"{speedup_v2_vs_v1:<12.2f}x"
                    )
            except Exception as e:
                import traceback
                print(f"{desc:<45} ERROR: {e}")
                traceback.print_exc()

        print()

    print("Legend:")
    print("  MIS v1: Multi-item scoring with delimiter tokens (kMultiItemScoring)")
    print("  MIS v2: Delimiterless multi-item scoring (kMultiItemScoringV2)")
    print("  CustomMask: single_prefill with explicit custom mask (reference baseline, batch_size=1 only)")
    print("  v2 vs v1: speedup of v2 over v1 (>1 = v2 faster)")
    print("  v2 vs CM: speedup of v2 over custom mask (>1 = v2 faster)")


if __name__ == "__main__":
    main()
