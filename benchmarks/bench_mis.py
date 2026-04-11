"""Benchmark: Multi-Item Scoring (MIS) performance."""

import time
import torch
import numpy as np

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


def bench_plan_time(plan_fn, warmup=3, repeat=20):
    """Measure plan() wall-clock time in ms."""
    for _ in range(warmup):
        plan_fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        plan_fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def build_mis_inputs(batch_size, prefix_len, item_lens):
    """Build MIS inputs: uses item_offsets CSR format."""
    total_items_tokens = sum(item_lens)
    qo_len = total_items_tokens
    kv_len = prefix_len + total_items_tokens

    offsets = [0]
    for l in item_lens:
        offsets.append(offsets[-1] + l)

    max_item_len = max(item_lens)

    return (
        qo_len,
        kv_len,
        torch.tensor([prefix_len], dtype=torch.uint32).cuda(),
        torch.tensor(offsets, dtype=torch.uint32).cuda(),
        len(offsets),
        torch.tensor([max_item_len], dtype=torch.uint16).cuda(),
    )


def old_plan_preprocess(item_offsets, item_offsets_len, prefix_len_ptr):
    """Reproduce the ORIGINAL plan() preprocessing for timing comparison.

    This is the old Python loop with .item() calls that we're replacing.
    """
    batch_size = len(prefix_len_ptr) if prefix_len_ptr is not None else 1
    item_start_list = []
    max_items_tokens = 0
    for b in range(batch_size):
        offsets_b = item_offsets[b * item_offsets_len:(b + 1) * item_offsets_len]
        num_offsets = (offsets_b.to(torch.int64) > 0).sum().item() + 1
        valid_offsets = offsets_b[:num_offsets].long()
        total_tokens = valid_offsets[-1].item()
        max_items_tokens = max(max_items_tokens, total_tokens)
        item_start_b = torch.zeros(total_tokens, dtype=torch.uint32, device=item_offsets.device)
        for i in range(num_offsets - 1):
            start = valid_offsets[i].item()
            end = valid_offsets[i + 1].item()
            item_start_b[start:end] = start
        item_start_list.append(item_start_b)
    padded = []
    for ist in item_start_list:
        if len(ist) < max_items_tokens:
            ist = torch.cat([ist, torch.zeros(max_items_tokens - len(ist), dtype=torch.uint32, device=ist.device)])
        padded.append(ist)
    return torch.cat(padded), max_items_tokens


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
    device = "cuda"
    results = {}

    (
        qo_len, kv_len,
        prefix_len_ptr, item_offsets, item_offsets_len, max_item_len_ptr,
    ) = build_mis_inputs(batch_size, prefix_len, item_lens)

    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim, device=device, dtype=torch.float16)
    num_pages = (kv_len + page_size - 1) // page_size
    total_pages = num_pages * batch_size
    kv_data = torch.randn(total_pages, 2, page_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    q_indptr = (torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len).cuda()
    kv_indptr = (torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages).cuda()
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=device)
    kv_last_page = torch.full((batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # New plan: pass item_offsets + max_item_len_ptr directly (zero expansion)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    def plan_new():
        wrapper.plan(
            q_indptr, kv_indptr, kv_indices, kv_last_page,
            num_qo_heads, num_kv_heads, head_dim, page_size,
            causal=True, pos_encoding_mode="NONE",
            prefix_len_ptr=prefix_len_ptr,
            item_offsets=item_offsets,
            item_offsets_len=item_offsets_len,
            max_item_len_ptr=max_item_len_ptr,
        )
    plan_new()
    results["kern"] = bench_gpu_time(lambda: wrapper.run(q, kv_data), warmup=warmup, repeat=repeat)
    results["plan_new"] = bench_plan_time(plan_new)

    # Old plan: Python loop with .item() calls
    results["plan_old"] = bench_plan_time(
        lambda: old_plan_preprocess(item_offsets, item_offsets_len, prefix_len_ptr),
        warmup=3, repeat=20,
    )

    return results, {"qo_len": qo_len, "kv_len": kv_len}


def main():
    NUM_LAYERS = 28  # Qwen3-0.6B

    print("=" * 80)
    print("MIS Benchmark")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    configs = [
        # Uniform item lengths
        (512, [64] * 5, "512 prefix, 5x64 uniform"),
        (1024, [128] * 10, "1024 prefix, 10x128 uniform"),
        (2048, [256] * 5, "2048 prefix, 5x256 uniform"),
        (4096, [512] * 4, "4096 prefix, 4x512 uniform"),
        (1024, [64] * 20, "1024 prefix, 20x64 uniform"),
        (1024, [64] * 50, "1024 prefix, 50x64 uniform"),
        (2048, [32] * 100, "2048 prefix, 100x32 uniform"),
        # Variable item lengths (moderate variation, realistic)
        (512, [20, 30, 25, 35, 40], "512 prefix, 5 mixed [20..40]"),
        (256, [15, 25, 20, 30, 10] * 10, "256 prefix, 50 mixed [10..30]"),
        (512, list(range(20, 45, 5)) * 10, "512 prefix, 50 ramp [20..40]"),
        (1024, [40, 60, 50, 70, 55] * 10, "1024 prefix, 50 mixed [40..70]"),
        (512, [25, 35, 30, 45, 20] * 20, "512 prefix, 100 mixed [20..45]"),
    ]

    num_qo_heads = 16
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    batch_size = 1

    print(f"Config: {num_qo_heads} QO heads, {num_kv_heads} KV heads, {head_dim} head_dim, page_size={page_size}")
    print(f"End-to-end = plan + {NUM_LAYERS} layers * kernel  (all times in ms)")
    print()

    header = (
        f"{'Config':<35} {'QO/KV':<12} "
        f"{'kernel':<9} "
        f"{'OldPlan':<9} {'NewPlan':<9} {'Plan+':<7} "
        f"{'e2e old':<9} {'e2e new':<9} {'e2e+':<7}"
    )
    print(header)
    print("-" * len(header))

    for prefix_len, item_lens, desc in configs:
        try:
            results, lens = run_benchmark(
                batch_size, prefix_len, item_lens,
                num_qo_heads, num_kv_heads, head_dim, page_size,
            )

            kern = np.mean(results["kern"])
            plan_old = np.mean(results["plan_old"])
            plan_new = np.mean(results["plan_new"])
            plan_speedup = plan_old / plan_new

            e2e_old = plan_old + NUM_LAYERS * kern
            e2e_new = plan_new + NUM_LAYERS * kern
            e2e_speedup = e2e_old / e2e_new

            lens_str = f"{lens['qo_len']}/{lens['kv_len']}"

            print(
                f"{desc:<35} {lens_str:<12} "
                f"{kern:<9.4f} "
                f"{plan_old:<9.4f} {plan_new:<9.4f} {plan_speedup:<7.2f} "
                f"{e2e_old:<9.2f} {e2e_new:<9.2f} {e2e_speedup:<7.2f}"
            )
        except Exception as e:
            import traceback
            print(f"{desc:<35} ERROR: {e}")
            traceback.print_exc()

    print()
    print("Legend:")
    print("  kernel: MIS GPU kernel time (ms)")
    print("  OldPlan: original plan() with Python .item() loop (ms)")
    print("  NewPlan: optimized plan() with direct passthrough (ms)")
    print("  Plan+: plan speedup (>1 = new faster)")
    print(f"  e2e old / e2e new: plan + {NUM_LAYERS} layers * kernel (ms)")
    print("  e2e+: end-to-end speedup (>1 = new faster)")


if __name__ == "__main__":
    main()
