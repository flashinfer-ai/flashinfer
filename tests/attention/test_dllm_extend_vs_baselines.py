"""dLLM Extend: BBE/V2 vs Baselines Speedup

Unlike the step-by-step prefill pipeline, in the extend scenario the KV cache
is already fully populated and Q only processes a small segment (corresponding
to dLLM re-sampling / verification of an intermediate block).

Key property: Q_end << KV_end → block_extend plan optimization takes effect.

Compares 4 approaches:
  - [Baseline 1] SGLang Cascade: Ragged(current) + Paged(prefix) + merge_state
  - [Baseline 2] Custom Mask: single_prefill_with_kv_cache(custom_mask=...)
  - [V2] block_extend: single_prefill_with_kv_cache(block_extend=True, ...)
  - [BBE] BatchBlockExpanding Ragged: BatchPrefillWithRaggedKVCacheWrapper

All approaches use CUDA Graph to eliminate launch overhead.
"""

import torch
import math
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    merge_state,
    single_prefill_with_kv_cache,
)


def compute_block_extend_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dllm_block_size: int,
    q_offset: int = 0,
    sm_scale: float = None,
) -> torch.Tensor:
    """Compute block-extend attention reference using custom_mask."""
    qo_len = q.shape[0]
    kv_len = k.shape[0]
    head_dim = q.shape[-1]
    device = q.device
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    q_pos = torch.arange(qo_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device)
    q_block = q_pos.unsqueeze(1) // dllm_block_size
    k_block = k_pos.unsqueeze(0) // dllm_block_size
    mask_2d = (q_block >= k_block).to(torch.uint8)
    return single_prefill_with_kv_cache(
        q,
        k,
        v,
        custom_mask=mask_2d,
        sm_scale=sm_scale,
    )


def run(
    tokens_per_request: int = 8192,
    dllm_block_size: int = 32,
    chunk_sizes: list = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    page_size: int = 16,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
):
    if chunk_sizes is None:
        chunk_sizes = [32, 64, 128, 256]

    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim**0.5)
    B = dllm_block_size
    WS_MB = 256

    # Place Q in the middle of the sequence so Q_end << KV_end
    q_offset_base = tokens_per_request // 2

    print(f"\n{'=' * 90}")
    print("dLLM Extend: BBE/V2 vs Baselines Speedup")
    print(f"{'=' * 90}")
    print("Configuration:")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  dllm_block_size     = {dllm_block_size}")
    print(f"  chunk_sizes         = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print(
        f"  q_offset_base       = {q_offset_base} (mid-sequence, ensures Q_end << KV_end)"
    )
    print("\nScenario:")
    eff_examples = []
    for cs in chunk_sizes:
        q_end = q_offset_base + cs
        eff = min(tokens_per_request, ((q_end - 1) // B + 1) * B)
        eff_examples.append(
            f"chunk={cs}: eff_kv={eff} ({eff / tokens_per_request * 100:.1f}%)"
        )
    print("  " + ", ".join(eff_examples))
    print("  All approaches: single call (1 step), with CUDA Graph")

    results = {}

    for chunk_size in chunk_sizes:
        if tokens_per_request % chunk_size != 0:
            continue

        q_offset = q_offset_base

        print(f"\n{'-' * 90}")
        print(
            f"chunk_size = {chunk_size}  (Q={chunk_size} @ offset={q_offset}, KV={tokens_per_request})"
        )
        print(f"{'-' * 90}")

        # Build Q (extend chunk) and full K, V
        q_extend = torch.randn(
            chunk_size, num_heads, head_dim, dtype=dtype, device=device
        )
        k_full = torch.randn(
            tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_full = torch.randn(
            tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Compute block ranges
        q_block_start = q_offset // B
        q_block_end = (q_offset + chunk_size - 1) // B
        num_q_blocks = q_block_end - q_block_start + 1
        kv_current_start = q_block_start * B
        kv_current_end = min((q_block_end + 1) * B, tokens_per_request)
        kv_prefix_end = kv_current_start

        if verbose:
            print(
                f"  Q  block range: [{q_block_start}, {q_block_end}] ({num_q_blocks} blocks)"
            )
            print(f"  KV current:     [{kv_current_start}, {kv_current_end})")
            print(f"  KV prefix:      [0, {kv_prefix_end})")

        # ================================================================
        # [Baseline 1] SGLang Cascade
        # ================================================================
        print("  [Cascade]  SGLang Cascade...")
        ws_cascade = torch.empty(WS_MB * 1024 * 1024, dtype=torch.uint8, device=device)

        k_current = k_full[kv_current_start:kv_current_end]
        v_current = v_full[kv_current_start:kv_current_end]
        kv_curr_len = kv_current_end - kv_current_start
        qo_indptr = torch.tensor([0, chunk_size], dtype=torch.int32, device=device)

        wrapper_curr = BatchPrefillWithRaggedKVCacheWrapper(ws_cascade, kv_layout="NHD")
        wrapper_curr.plan(
            qo_indptr=qo_indptr,
            kv_indptr=torch.tensor([0, kv_curr_len], dtype=torch.int32, device=device),
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=False,
            sm_scale=sm_scale,
        )

        o_curr = torch.empty_like(q_extend)
        s_curr = torch.empty(chunk_size, num_heads, dtype=torch.float32, device=device)
        o_prefix = torch.empty_like(q_extend)
        s_prefix = torch.empty(
            chunk_size, num_heads, dtype=torch.float32, device=device
        )

        has_prefix = kv_prefix_end > 0
        wrapper_pfx = None
        paged_kv_cache = None

        if has_prefix:
            num_prefix_pages = (kv_prefix_end + page_size - 1) // page_size
            last_page_len = kv_prefix_end - (num_prefix_pages - 1) * page_size

            paged_kv_cache = torch.zeros(
                num_prefix_pages,
                2,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for p in range(num_prefix_pages):
                start = p * page_size
                end = min(start + page_size, kv_prefix_end)
                actual = end - start
                paged_kv_cache[p, 0, :actual] = k_full[start:end]
                paged_kv_cache[p, 1, :actual] = v_full[start:end]

            paged_kv_indices = torch.arange(
                num_prefix_pages, dtype=torch.int32, device=device
            )
            paged_kv_indptr = torch.tensor(
                [0, num_prefix_pages], dtype=torch.int32, device=device
            )
            paged_kv_last_page_len = torch.tensor(
                [last_page_len], dtype=torch.int32, device=device
            )

            wrapper_pfx = BatchPrefillWithPagedKVCacheWrapper(
                ws_cascade, kv_layout="NHD"
            )
            wrapper_pfx.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=False,
                sm_scale=sm_scale,
            )

        def run_cascade_once():
            if has_prefix:
                o1, s1 = wrapper_curr.run_return_lse(q_extend, k_current, v_current)  # noqa: F821
                o_curr.copy_(o1)
                s_curr.copy_(s1)
                o2, s2 = wrapper_pfx.run(q_extend, paged_kv_cache, return_lse=True)  # noqa: F821
                o_prefix.copy_(o2)
                s_prefix.copy_(s2)
                o_out, _ = merge_state(o_curr, s_curr, o_prefix, s_prefix)
            else:
                o_out = wrapper_curr.run(q_extend, k_current, v_current)  # noqa: F821
            return o_out

        # CUDA Graph
        for _ in range(max(1, warmup_iters // 2)):
            run_cascade_once()
        torch.cuda.synchronize()
        cascade_stream = torch.cuda.Stream()
        with torch.cuda.stream(cascade_stream):
            run_cascade_once()
        cascade_stream.synchronize()
        cascade_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cascade_graph, stream=cascade_stream):
            run_cascade_once()  # noqa: F841 (CUDA Graph capture, output unused)
        for _ in range(max(1, warmup_iters // 2)):
            cascade_graph.replay()
        torch.cuda.synchronize()
        se, ee = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        se.record()
        for _ in range(bench_iters):
            cascade_graph.replay()
        ee.record()
        torch.cuda.synchronize()
        cascade_us = se.elapsed_time(ee) / bench_iters * 1000
        print(f"    => {cascade_us:.1f} us")
        del cascade_graph, wrapper_curr
        if wrapper_pfx is not None:
            del wrapper_pfx
        torch.cuda.empty_cache()

        # ================================================================
        # [Baseline 2] Custom Mask
        # ================================================================
        print("  [CustMsk] Custom Mask (single_prefill + custom_mask)...")
        q_pos = torch.arange(chunk_size, device=device) + q_offset
        k_pos = torch.arange(tokens_per_request, device=device)
        q_block = q_pos.unsqueeze(1) // B
        k_block = k_pos.unsqueeze(0) // B
        mask_2d = (q_block >= k_block).to(torch.uint8)
        cm_out = torch.empty_like(q_extend)

        def run_cm_once():
            cm_out.copy_(
                single_prefill_with_kv_cache(
                    q_extend,
                    k_full,
                    v_full,
                    custom_mask=mask_2d,
                    sm_scale=sm_scale,
                    backend="fa2",
                )
            )

        for _ in range(max(1, warmup_iters // 2)):
            run_cm_once()
        torch.cuda.synchronize()
        cm_stream = torch.cuda.Stream()
        with torch.cuda.stream(cm_stream):
            run_cm_once()
        cm_stream.synchronize()
        cm_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cm_graph, stream=cm_stream):
            run_cm_once()
        for _ in range(max(1, warmup_iters // 2)):
            cm_graph.replay()
        torch.cuda.synchronize()
        se, ee = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        se.record()
        for _ in range(bench_iters):
            cm_graph.replay()
        ee.record()
        torch.cuda.synchronize()
        cm_us = se.elapsed_time(ee) / bench_iters * 1000
        print(f"    => {cm_us:.1f} us")
        del cm_graph
        torch.cuda.empty_cache()

        # ================================================================
        # [V2] block_extend_attention
        # ================================================================
        print("  [V2    ] block_extend (single_prefill, block_extend=True)...")
        v2_out = torch.empty_like(q_extend)

        def run_v2_once():
            v2_out.copy_(
                single_prefill_with_kv_cache(
                    q_extend,
                    k_full,
                    v_full,
                    block_extend=True,
                    block_size=dllm_block_size,
                    q_offset=q_offset,
                    sm_scale=sm_scale,
                )
            )

        for _ in range(max(1, warmup_iters // 2)):
            run_v2_once()
        torch.cuda.synchronize()
        v2_stream = torch.cuda.Stream()
        with torch.cuda.stream(v2_stream):
            run_v2_once()
        v2_stream.synchronize()
        v2_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(v2_graph, stream=v2_stream):
            run_v2_once()
        for _ in range(max(1, warmup_iters // 2)):
            v2_graph.replay()
        torch.cuda.synchronize()
        se, ee = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        se.record()
        for _ in range(bench_iters):
            v2_graph.replay()
        ee.record()
        torch.cuda.synchronize()
        v2_us = se.elapsed_time(ee) / bench_iters * 1000
        print(f"    => {v2_us:.1f} us")
        del v2_graph
        torch.cuda.empty_cache()

        # ================================================================
        # [BBE] BatchBlockExpanding Ragged
        # ================================================================
        print("  [BBE   ] BatchBlockExpanding Ragged...")
        ws_bbe = torch.empty(WS_MB * 1024 * 1024, dtype=torch.uint8, device=device)
        bbe_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            ws_bbe,
            kv_layout="NHD",
            block_extend=True,
            block_size=dllm_block_size,
        )
        bbe_wrapper.plan(
            qo_indptr=qo_indptr,
            kv_indptr=torch.tensor(
                [0, tokens_per_request], dtype=torch.int32, device=device
            ),
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_data_type=dtype,
            sm_scale=sm_scale,
            q_offsets=torch.tensor([q_offset], dtype=torch.int32, device=device),
        )
        bbe_out = torch.empty_like(q_extend)

        def run_bbe_once():
            bbe_out.copy_(bbe_wrapper.run(q_extend, k_full, v_full))  # noqa: F821

        for _ in range(max(1, warmup_iters // 2)):
            run_bbe_once()
        torch.cuda.synchronize()
        bbe_stream = torch.cuda.Stream()
        with torch.cuda.stream(bbe_stream):
            run_bbe_once()
        bbe_stream.synchronize()
        bbe_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bbe_graph, stream=bbe_stream):
            run_bbe_once()
        for _ in range(max(1, warmup_iters // 2)):
            bbe_graph.replay()
        torch.cuda.synchronize()
        se, ee = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        se.record()
        for _ in range(bench_iters):
            bbe_graph.replay()
        ee.record()
        torch.cuda.synchronize()
        bbe_us = se.elapsed_time(ee) / bench_iters * 1000

        # Plan metrics
        pv = bbe_wrapper._plan_info
        bbe_padded_bs = int(pv[0])
        bbe_split_kv = bool(pv[-1])
        bbe_seen_kv = -1
        if len(pv) > 4:
            iws = bbe_wrapper._int_workspace_buffer
            off = int(pv[4])
            if 0 <= off < iws.numel():
                bbe_seen_kv = int(iws.view(torch.int32)[off // 4].item())
        bbe_kv_chunk = -1
        if len(pv) >= 15:
            iws = bbe_wrapper._int_workspace_buffer
            off = int(pv[9])
            if 0 <= off < iws.numel():
                bbe_kv_chunk = int(iws.view(torch.int32)[off // 4].item())

        print(
            f"    => {bbe_us:.1f} us  (pad_bs={bbe_padded_bs}, split_kv={bbe_split_kv}, "
            f"seen_kv={bbe_seen_kv}, kv_chunk={bbe_kv_chunk})"
        )
        del bbe_graph, bbe_wrapper, ws_bbe
        torch.cuda.empty_cache()

        # Correctness (BBE/V2 vs CustomMask reference)
        cm_ref = compute_block_extend_reference(
            q_extend,
            k_full,
            v_full,
            dllm_block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
        )
        bbe_diff = (bbe_out - cm_ref).abs().max().item()
        v2_diff = (v2_out - cm_ref).abs().max().item()
        tol = 1e-2
        if verbose or bbe_diff >= tol or v2_diff >= tol:
            print(
                f"  Correctness vs CustomMask: BBE max_diff={bbe_diff:.6f}, V2 max_diff={v2_diff:.6f}"
            )

        q_end = q_offset + chunk_size
        eff_kv = min(tokens_per_request, ((q_end - 1) // B + 1) * B)

        results[f"chunk{chunk_size}"] = {
            "chunk_size": chunk_size,
            "q_offset": q_offset,
            "eff_kv": eff_kv,
            "cascade_us": cascade_us,
            "cm_us": cm_us,
            "v2_us": v2_us,
            "bbe_us": bbe_us,
            "bbe_padded_bs": bbe_padded_bs,
            "bbe_split_kv": bbe_split_kv,
            "bbe_seen_kv": bbe_seen_kv,
            "bbe_kv_chunk": bbe_kv_chunk,
            "bbe_diff": bbe_diff,
            "v2_diff": v2_diff,
        }

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 90}")
    print("Summary: dLLM Extend (Q_end << KV_end, plan optimization active)")
    print(f"{'=' * 90}")

    print(
        f"\n{'chunk':>6} {'eff_kv':>7} | "
        f"{'Casc(us)':>8} {'CMsk(us)':>8} {'V2(us)':>8} {'BBE(us)':>8} | "
        f"{'BBE/Cas':>8} {'BBE/CM':>8} {'V2/CM':>8} | "
        f"{'plan':>12}"
    )
    print(
        f"{'-' * 6}-+-{'-' * 7}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-"
        f"{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 12}"
    )

    for key in sorted(results.keys(), key=lambda k: results[k]["chunk_size"]):
        r = results[key]
        speedup_bb_cas = r["cascade_us"] / r["bbe_us"]
        speedup_bb_cm = r["cm_us"] / r["bbe_us"]
        speedup_v2_cm = r["cm_us"] / r["v2_us"]
        plan_str = f"pad={r['bbe_padded_bs']} skv={r['bbe_split_kv']}"
        print(
            f"{r['chunk_size']:>6} {r['eff_kv']:>7} | "
            f"{r['cascade_us']:>7.1f} {r['cm_us']:>7.1f} "
            f"{r['v2_us']:>7.1f} {r['bbe_us']:>7.1f} | "
            f"{speedup_bb_cas:>7.2f}x {speedup_bb_cm:>7.2f}x "
            f"{speedup_v2_cm:>7.2f}x | {plan_str:>12}"
        )

    print("\nNotes:")
    print(
        f"  - Scenario: KV={tokens_per_request} fully populated, Q={chunk_sizes}@offset={q_offset_base}"
    )
    print("  - Q_end << KV_end → block_extend plan optimization active")
    print("  - BBE/Cas: BBE speedup vs SGLang Cascade")
    print("  - BBE/CM : BBE speedup vs Custom Mask")
    print("  - V2/CM  : V2 block_extend speedup vs Custom Mask")
    print("  - plan   : BBE scheduler decision (pad=padding, skv=split_kv)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="dLLM Extend speedup comparison")
    parser.add_argument(
        "--tokens_per_request", type=int, default=8192, help="KV cache total length"
    )
    parser.add_argument(
        "--dllm_block_size", type=int, default=32, help="DLLM block size"
    )
    parser.add_argument(
        "--chunk_sizes", type=str, default="32,64,128,256", help="Q size list"
    )
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()

    run(
        tokens_per_request=args.tokens_per_request,
        dllm_block_size=args.dllm_block_size,
        chunk_sizes=[int(x) for x in args.chunk_sizes.split(",")],
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        verbose=args.verbose,
    )
