"""Reproducible SM100 blk64 attention-only benchmark (prequantized inputs).

Run before and after a kernel change with identical arguments. Compilation and
warmup are excluded; GPU timing uses CUPTI with CUDA-event fallback. Host timing
measures warmed enqueue latency. Graph replay timing flushes L2 each iteration.
"""

import argparse
import importlib.metadata
import json
import statistics
import time

import torch

from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd
from flashinfer.testing import bench_gpu_time


def run(backend, sq, sk, topk, splits, cold_l2, quantize=False, bf16=False):
    torch.manual_seed(42)
    heads, dim = 4, 128
    q, k, v = [
        torch.randn(1, s, heads, dim, device="cuda", dtype=torch.bfloat16)
        for s in (sq, sk, sk)
    ]
    qs = q.float().abs().amax(-1).clamp_min(1e-6) / 448
    ks = k.float().reshape(1, sk // 16, 16, heads, dim).abs().amax((2, 4))
    ks = ks.clamp_min(1e-6) / 448
    vs = v.float().abs().amax((0, 1)).clamp_min(1e-6) / 448
    q8 = (q.float() / qs[..., None]).to(torch.float8_e4m3fn)
    k8 = (k.float() / ks.repeat_interleave(16, 1)[..., None]).to(torch.float8_e4m3fn)
    v8 = (v.float() / vs).to(torch.float8_e4m3fn)
    qs, ks = qs.transpose(1, 2).contiguous(), ks.transpose(1, 2).contiguous()
    index = torch.arange(topk, dtype=torch.int32, device="cuda")
    index = index.expand(1, heads, (sq + 63) // 64, topk).contiguous()
    out = torch.empty_like(q)

    def fn():
        if bf16:
            tensors = q, k, v, None, None, None
        elif quantize:
            from flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake import (
                sage_fp8_quantize_sm100,
            )

            tensors = sage_fp8_quantize_sm100(q, k, v)
        else:
            tensors = q8, k8, v8, qs, ks, vs
        fq, fk, fv, fqs, fks, fvs = tensors
        if fvs is not None and fvs.ndim == 3:
            fvs = fvs[0]  # B=1; use the V-scale form shared by both backends.
        return bsa_attn_sm100_blk64_fwd(
            fq,
            fk,
            fv,
            index,
            topk,
            q_scale=fqs,
            k_scale=fks,
            v_scale=fvs,
            backend=backend,
            kv_splits=splits,
            out=out,
        )

    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    live_bytes = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak_extra_bytes = torch.cuda.max_memory_allocated() - live_bytes
    gpu_ms = statistics.median(
        bench_gpu_time(fn, enable_cupti=True, cold_l2_cache=cold_l2)
    )
    start = time.perf_counter()
    for _ in range(100):
        fn()
    host_us = (time.perf_counter() - start) * 1e4
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph_ms = statistics.median(
        bench_gpu_time(graph.replay, enable_cupti=True, cold_l2_cache=True)
    )
    return dict(
        backend=backend,
        sq=sq,
        sk=sk,
        topk=topk,
        splits=splits,
        gpu_us=gpu_ms * 1000,
        host_enqueue_us=host_us,
        graph_us=graph_ms * 1000,
        cold_l2=cold_l2,
        graph_cold_l2=True,
        peak_extra_bytes=peak_extra_bytes,
        quantization_included=quantize,
        dtype="bf16" if bf16 else "sage_fp8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["cute", "cake", "auto"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    if args.bf16 and (args.quantize or "cake" in args.backends):
        parser.error("--bf16 requires --backends cute (or auto) without --quantize")
    results = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cutlass=importlib.metadata.version("nvidia-cutlass-dsl"),
        tvm_ffi=importlib.metadata.version("apache-tvm-ffi"),
        cuda_bindings=importlib.metadata.version("cuda-bindings"),
        sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
        heads=4,
        head_dim=128,
        cases=[],
    )
    for backend in args.backends:
        for sq, sk, topk, splits in [
            (4096, 8192, 32, 1),
            (4096, 8192, 128, 1),
            (16384, 16384, 256, 1),
            (64, 16384, 256, "auto"),
            (8192, 65536, 1024, "auto"),
        ]:
            row = run(
                backend, sq, sk, topk, splits, args.cold_l2, args.quantize, args.bf16
            )
            results["cases"].append(row)
            print(json.dumps(row), flush=True)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
