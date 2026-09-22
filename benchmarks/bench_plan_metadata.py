"""Measure host planning latency and metadata synchronization (issue #3338).

Run this script on both revisions with the same arguments, for example:
    python benchmarks/bench_plan_metadata.py --output plan-metadata.json

Times include plan()'s CPU work and any waits for the GPU, but exclude JIT
warmup and the explicit synchronization around each sample. The busy case
queues a 4096x4096 FP16 GEMM immediately before timing plan(). These are
planning microbenchmarks, not attention kernel or whole-model timings.
"""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

import flashinfer
from flashinfer.mla import MLAPlanMetadata


def make_plan(kind, mode, batch, kv_len, page_size):
    qo = torch.arange(batch + 1, dtype=torch.int32)
    pages = (kv_len + page_size - 1) // page_size
    kv = qo * (kv_len if kind == "ragged" else pages)
    lengths = torch.full((batch,), kv_len, dtype=torch.int32)
    last = torch.full((batch,), (kv_len - 1) % page_size + 1, dtype=torch.int32)
    indices = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(batch)
    if mode == "gpu":
        qo, kv, lengths, last = (x.cuda() for x in (qo, kv, lengths, last))
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    if kind == "paged":
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, backend="fa2"
        )
        return lambda: wrapper.plan(
            qo,
            kv,
            indices,
            last,
            32,
            8,
            128,
            page_size,
            causal=True,
            q_data_type=torch.float16,
        )
    if kind == "ragged":
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, backend="fa2"
        )
        return lambda: wrapper.plan(
            qo,
            kv,
            32,
            8,
            128,
            causal=True,
            q_data_type=torch.float16,
        )
    if kind == "decode":
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, backend="fa2", use_tensor_cores=False
        )
        return lambda: wrapper.plan(
            kv,
            indices,
            last,
            32,
            8,
            128,
            page_size,
            q_data_type=torch.float16,
        )
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    metadata = MLAPlanMetadata.csr(qo, kv, indices, lengths)
    return lambda: wrapper.plan(
        metadata=metadata,
        num_heads=32,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=page_size,
        causal=True,
        sm_scale=576**-0.5,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("batch_size", "kv_len", "page_size", "iterations"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be strictly positive")
    source_root = Path(__file__).resolve().parents[1]
    if not Path(flashinfer.__file__).resolve().is_relative_to(source_root):
        raise RuntimeError(
            f"Benchmark must import this checkout. Set PYTHONPATH={source_root}"
        )
    report = {
        "revision": subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source": str(Path(flashinfer.__file__).resolve()),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "shape": {
            "batch_size": args.batch_size,
            "q_len": 1,
            "kv_len": args.kv_len,
            "page_size": args.page_size,
            "heads": 32,
            "kv_heads": 8,
            "attention_head_dim": 128,
            "mla_head_dims": [512, 64],
            "dtype": "float16",
        },
        "iterations": args.iterations,
        "profile_iterations": 3,
        "results": {},
    }
    a = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    b, c = torch.randn_like(a), torch.empty_like(a)
    for _ in range(10):
        torch.mm(a, b, out=c)
    for kind in ("paged", "ragged", "decode", "mla"):
        plans = {
            mode: make_plan(kind, mode, args.batch_size, args.kv_len, args.page_size)
            for mode in ("cpu", "gpu")
        }
        for _ in range(10):
            for plan in plans.values():
                plan()
        torch.cuda.synchronize()
        result = {}
        for busy in (False, True):
            samples = {mode: [] for mode in plans}
            for i in range(args.iterations):
                for mode in list(plans) if i % 2 else list(reversed(plans)):
                    torch.cuda.synchronize()
                    if busy:
                        torch.mm(a, b, out=c)
                    start = time.perf_counter_ns()
                    plans[mode]()
                    samples[mode].append((time.perf_counter_ns() - start) / 1000)
                    torch.cuda.synchronize()
            result["busy_us" if busy else "idle_us"] = {
                mode: {"median": statistics.median(values), "samples": values}
                for mode, values in samples.items()
            }
        result["profiles"] = {}
        for mode, plan in plans.items():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(3):
                    with torch.profiler.record_function("metadata_plan"):
                        plan()
            counts = {}
            for event in prof.events():
                if any(
                    name in event.name
                    for name in (
                        "DtoH",
                        "cudaStreamSynchronize",
                        "cudaDeviceSynchronize",
                        "cudaMemcpyAsync",
                    )
                ):
                    counts[event.name] = counts.get(event.name, 0) + 1
            result["profiles"][mode] = counts
        report["results"][kind] = result
        print(
            kind,
            {
                key: {mode: data["median"] for mode, data in values.items()}
                for key, values in result.items()
                if key.endswith("_us")
            },
            flush=True,
        )
        print(result["profiles"], flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
