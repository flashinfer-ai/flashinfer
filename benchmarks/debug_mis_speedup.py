"""Fair benchmark: MIS v1 vs v2 with same qo_len treatment.

Both v1 and v2 exclude prefix from Q — the only difference is delimiter tokens.
Focuses on configs where delimiter savings matter (many small items).
"""

import torch
import numpy as np
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


def bench_gpu_time(fn, warmup=10, repeat=100):
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
    return np.mean(times), np.std(times)


def run_v1_no_prefix_q(workspace, prefix_len, item_lens, num_qo_heads, num_kv_heads, head_dim, page_size):
    """V1 with prefix NOT in Q — only delimiters + item tokens in Q."""
    device = "cuda"
    token_pos = []
    for il in item_lens:
        token_pos.append(0)  # delimiter
        for j in range(1, il + 1):
            token_pos.append(j)

    qo_len = len(token_pos)  # delimiters + items
    kv_len = prefix_len + qo_len

    q = torch.randn(qo_len, num_qo_heads, head_dim, device=device, dtype=torch.float16)
    num_pages = (kv_len + page_size - 1) // page_size
    kv_data = torch.randn(num_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper.plan(
        torch.tensor([0, qo_len], dtype=torch.int32, device=device),
        torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        torch.arange(num_pages, dtype=torch.int32, device=device),
        torch.tensor([(kv_len - 1) % page_size + 1], dtype=torch.int32, device=device),
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        prefix_len_ptr=torch.tensor([prefix_len], dtype=torch.uint32, device=device),
        token_pos_in_items_ptr=torch.tensor(token_pos, dtype=torch.uint16, device=device),
        token_pos_in_items_len=len(token_pos),
        max_item_len_ptr=torch.tensor([max(item_lens)], dtype=torch.uint16, device=device),
    )
    mean, std = bench_gpu_time(lambda: wrapper.run(q, kv_data))
    return mean, qo_len, kv_len


def run_v2(workspace, prefix_len, item_lens, num_qo_heads, num_kv_heads, head_dim, page_size):
    """V2 — only item tokens in Q (no delimiters, no prefix)."""
    device = "cuda"
    total_item_tokens = sum(item_lens)
    qo_len = total_item_tokens
    kv_len = prefix_len + total_item_tokens

    offsets = [0]
    for il in item_lens:
        offsets.append(offsets[-1] + il)

    q = torch.randn(qo_len, num_qo_heads, head_dim, device=device, dtype=torch.float16)
    num_pages = (kv_len + page_size - 1) // page_size
    kv_data = torch.randn(num_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
    wrapper.plan(
        torch.tensor([0, qo_len], dtype=torch.int32, device=device),
        torch.tensor([0, num_pages], dtype=torch.int32, device=device),
        torch.arange(num_pages, dtype=torch.int32, device=device),
        torch.tensor([(kv_len - 1) % page_size + 1], dtype=torch.int32, device=device),
        num_qo_heads, num_kv_heads, head_dim, page_size,
        causal=True,
        prefix_len_ptr=torch.tensor([prefix_len], dtype=torch.uint32, device=device),
        item_offsets=torch.tensor(offsets, dtype=torch.uint32, device=device),
        item_offsets_len=len(offsets),
    )
    mean, std = bench_gpu_time(lambda: wrapper.run(q, kv_data))
    return mean, qo_len, kv_len


def main():
    device = "cuda"
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Heads: {num_qo_heads}/{num_kv_heads}, head_dim={head_dim}, page_size={page_size}")
    print()
    print("Both V1 and V2 exclude prefix from Q.")
    print("V1 Q = delimiters + items, V2 Q = items only.")
    print("The ONLY difference is delimiter tokens.")
    print()

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    configs = [
        # (prefix_len, item_lens, description)
        # High delimiter-to-item ratio (where v2 should shine)
        (256, [3] * 100,    "256 prefix, 100 items x 3 tok"),
        (256, [3] * 200,    "256 prefix, 200 items x 3 tok"),
        (256, [3] * 500,    "256 prefix, 500 items x 3 tok"),
        (512, [5] * 100,    "512 prefix, 100 items x 5 tok"),
        (512, [5] * 200,    "512 prefix, 200 items x 5 tok"),
        (256, [10] * 100,   "256 prefix, 100 items x 10 tok"),
        (256, [10] * 200,   "256 prefix, 200 items x 10 tok"),
        # Low delimiter ratio (original benchmark style — v2 should barely win)
        (4096, [512] * 4,   "4096 prefix, 4 items x 512 tok"),
        (1024, [128] * 10,  "1024 prefix, 10 items x 128 tok"),
        (512, [64] * 20,    "512 prefix, 20 items x 64 tok"),
    ]

    header = f"{'Config':<40} {'V1 qo':<8} {'V2 qo':<8} {'Delims':<8} {'V1 (ms)':<12} {'V2 (ms)':<12} {'Speedup':<10} {'Delim %':<8}"
    print(header)
    print("-" * len(header))

    for prefix_len, item_lens, desc in configs:
        try:
            t_v1, qo_v1, kv_v1 = run_v1_no_prefix_q(
                workspace, prefix_len, item_lens,
                num_qo_heads, num_kv_heads, head_dim, page_size,
            )
            t_v2, qo_v2, kv_v2 = run_v2(
                workspace, prefix_len, item_lens,
                num_qo_heads, num_kv_heads, head_dim, page_size,
            )
            num_delims = len(item_lens)
            delim_pct = num_delims / qo_v1 * 100

            print(
                f"{desc:<40} {qo_v1:<8} {qo_v2:<8} {num_delims:<8} "
                f"{t_v1:<12.4f} {t_v2:<12.4f} {t_v1/t_v2:<10.2f}x {delim_pct:<8.1f}%"
            )
        except Exception as e:
            import traceback
            print(f"{desc:<40} ERROR: {e}")
            traceback.print_exc()

    print()
    print("Delim % = delimiter tokens as fraction of V1's qo_len")
    print("Speedup > 1 means V2 is faster")


if __name__ == "__main__":
    main()
