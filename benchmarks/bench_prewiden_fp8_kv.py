"""Benchmark the opt-in FP8-KV pre-widen path against the FP8 and BF16 paths.

Usage:
    python benchmarks/bench_prewiden_fp8_kv.py --kv-len 8192 --q-len 8192
    python benchmarks/bench_prewiden_fp8_kv.py --kv-len 32768 --q-len 8192

The three arms share the same K/V values and page table:
  * fp8:      FA2 with BF16 Q and FP8 K/V (in-kernel repack).
  * prewiden: FA2 with BF16 Q and FP8 K/V, gathered+widened once per run.
  * bf16:     FA2 with BF16 Q and pre-widened BF16 K/V (the upper bound).
"""

import argparse
import statistics

import torch

import flashinfer


def _make_inputs(q_len, kv_len, hq, hkv, head_dim, page_size, dtype, layout, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    q = (torch.randn(q_len, hq, head_dim, generator=g, device=dev) * 0.5).to(
        torch.bfloat16
    )
    n_pages = (kv_len + page_size - 1) // page_size
    pool = n_pages + 7
    idx = torch.randperm(pool, generator=g, device=dev).to(torch.int32)[:n_pages]
    shape = (
        (pool, hkv, page_size, head_dim)
        if layout == "HND"
        else (pool, page_size, hkv, head_dim)
    )
    k = torch.randn(shape, generator=g, device=dev).to(dtype)
    v = torch.randn(shape, generator=g, device=dev).to(dtype)
    return q, k, v, idx


def _make_wrapper(
    layout, q_len, kv_len, hq, hkv, head_dim, page_size, idx, kv_dtype, prewiden
):
    dev = torch.device("cuda")
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend="fa2", prewiden_fp8_kv=prewiden
    )
    wrapper.plan(
        torch.tensor([0, q_len], dtype=torch.int32, device=dev),
        torch.tensor([0, idx.numel()], dtype=torch.int32, device=dev),
        idx,
        torch.tensor(
            [kv_len - (idx.numel() - 1) * page_size], dtype=torch.int32, device=dev
        ),
        hq,
        hkv,
        head_dim,
        page_size,
        causal=True,
        sm_scale=1.0 / head_dim**0.5,
        q_data_type=torch.bfloat16,
        kv_data_type=kv_dtype,
        o_data_type=torch.bfloat16,
        fixed_split_size=-1,
        disable_split_kv=False,
    )
    return wrapper


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return statistics.median(times), times[0], times[-1]


def _graph_time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return statistics.median(times), times[0], times[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int, default=8192)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--hq", type=int, default=24)
    parser.add_argument("--hkv", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--layout", choices=["HND", "NHD"], default="HND")
    parser.add_argument("--dtype", choices=["e4m3", "e5m2"], default="e4m3")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    kv_dtype = torch.float8_e4m3fn if args.dtype == "e4m3" else torch.float8_e5m2
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    q, k, v, idx = _make_inputs(
        args.q_len,
        args.kv_len,
        args.hq,
        args.hkv,
        args.head_dim,
        args.page_size,
        kv_dtype,
        args.layout,
    )
    k16 = k.index_select(0, idx.long()).to(torch.bfloat16)
    v16 = v.index_select(0, idx.long()).to(torch.bfloat16)
    compact = torch.arange(idx.numel(), dtype=torch.int32, device="cuda")

    common = (
        args.q_len,
        args.kv_len,
        args.hq,
        args.hkv,
        args.head_dim,
        args.page_size,
    )
    w_fp8 = _make_wrapper(args.layout, *common, idx, kv_dtype, prewiden=False)
    w_pw = _make_wrapper(args.layout, *common, idx, kv_dtype, prewiden=True)
    w_bf16 = _make_wrapper(
        args.layout, *common, compact, torch.bfloat16, prewiden=False
    )

    out_fp8 = torch.empty(
        args.q_len, args.hq, args.head_dim, dtype=torch.bfloat16, device="cuda"
    )
    out_pw = torch.empty_like(out_fp8)
    out_bf16 = torch.empty_like(out_fp8)

    def run_fp8():
        w_fp8.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=out_fp8)

    def run_pw():
        w_pw.run(q, (k, v), k_scale=1.125, v_scale=0.875, out=out_pw)

    def run_bf16():
        w_bf16.run(q, (k16, v16), k_scale=1.125, v_scale=0.875, out=out_bf16)

    run_fp8()
    run_pw()
    run_bf16()
    torch.cuda.synchronize()
    assert torch.equal(out_pw, out_bf16), "prewiden must match the BF16 kernel"
    diff = (out_pw.float() - out_fp8.float()).abs()
    print(
        f"prewiden vs fp8: max abs diff {diff.max().item():.6f}, "
        f"mean {diff.mean().item():.6f}"
    )

    for label, fn in (
        ("fp8", run_fp8),
        ("prewiden", run_pw),
        ("bf16", run_bf16),
    ):
        eager = _time(fn, args.warmup, args.iters)
        graph = _graph_time(fn, args.warmup, args.iters)
        print(
            f"{label:9s} eager median {eager[0]:8.3f} ms "
            f"[{eager[1]:.3f}, {eager[2]:.3f}]  "
            f"graph median {graph[0]:8.3f} ms [{graph[1]:.3f}, {graph[2]:.3f}]"
        )


if __name__ == "__main__":
    main()
