"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Benchmark the experimental NVFP4 attention backend on SM103.

For every ``[B,H,S,128]`` row this reports the attention launch (CUPTI, cold
L2, median over the measured iterations) and the preparation step
(``prepare_nvfp4_attention``: quantization + scale packing + binding) as host
wall time, the number of GPU kernels it launches, the bytes it copies host to
device and its peak allocated memory. Run it before and after a host-path
change to compare both halves of a call.

Usage::

    python benchmarks/bench_cake_nvfp4_attention.py [--rows b4_h8_s4096 ...]
        [--prepare-repeats 5] [--json out.json]
"""

import argparse
import json
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from flashinfer.prefill import prepare_nvfp4_attention
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.utils import get_compute_capability

ROWS = {
    "b4_h8_s4096": (4, 8, 4096),
    "b1_h8_s32768": (1, 8, 32768),
    "b8_h32_s8192": (8, 32, 8192),
}


def _prepare_profile(q, k, v, out):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        start = time.perf_counter()
        runner = prepare_nvfp4_attention(q, k, v, out, backend="cake")
        torch.cuda.synchronize()
        host_ms = (time.perf_counter() - start) * 1e3
    events = trace.events()
    kernels = sum(
        1 for e in events if e.device_type.name == "CUDA" and "Memcpy" not in e.name
    )
    h2d_copies = sum(
        1 for e in events if e.device_type.name == "CUDA" and "Memcpy HtoD" in e.name
    )
    peak_bytes = torch.cuda.max_memory_allocated() - before
    return runner, dict(
        prepare_host_ms=host_ms,
        prepare_kernels=kernels,
        prepare_h2d_copies=h2d_copies,
        prepare_peak_bytes=peak_bytes,
    )


def _summarize_prepare(samples):
    """Fold the per-repeat preparation samples into one record.

    Launch and copy counts are one value when every repeat agrees and the
    sorted distinct values otherwise. The peak allocation is the maximum over
    the repeats, so it is always a number the table can print in MiB.
    """
    record = {
        "prepare_host_ms_median": statistics.median(
            s["prepare_host_ms"] for s in samples
        ),
        "prepare_peak_bytes": max(s["prepare_peak_bytes"] for s in samples),
    }
    for key in ("prepare_kernels", "prepare_h2d_copies"):
        values = {s[key] for s in samples}
        record[key] = values.pop() if len(values) == 1 else sorted(values)
    return record


def bench_row(name, repeats):
    batch, heads, seqlen = ROWS[name]
    torch.manual_seed(42)
    q = torch.randn((batch, heads, seqlen, 128), dtype=torch.bfloat16, device="cuda")
    k, v = torch.randn_like(q), torch.randn_like(q)
    out = torch.empty_like(q)
    # Warm the JIT library and the cached device facts before measuring.
    prepare_nvfp4_attention(q, k, v, out, backend="cake")()
    torch.cuda.synchronize()
    samples = []
    runner = None
    for _ in range(repeats):
        runner, sample = _prepare_profile(q, k, v, out)
        samples.append(sample)
    record = {
        "row": name,
        "batch": batch,
        "heads": heads,
        "seqlen": seqlen,
        **_summarize_prepare(samples),
    }
    times = bench_gpu_time_with_cupti(runner, cold_l2_cache=True)
    record["attention_ms_median"] = float(statistics.median(times))
    flops = 4.0 * batch * heads * seqlen * seqlen * 128
    record["attention_tflops"] = flops / (record["attention_ms_median"] * 1e-3) / 1e12
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
    parser.add_argument("--prepare-repeats", type=int, default=5)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    if get_compute_capability(torch.device("cuda", torch.cuda.current_device())) != (
        10,
        3,
    ):
        raise SystemExit("NVFP4 attention requires an SM103 GPU")
    records = [bench_row(name, args.prepare_repeats) for name in args.rows]
    print(
        f"{'row':<16}{'attention ms':>14}{'TFLOPS':>10}{'prepare ms':>12}"
        f"{'kernels':>9}{'H2D':>6}{'peak MiB':>10}"
    )
    for r in records:
        print(
            f"{r['row']:<16}{r['attention_ms_median']:>14.4f}{r['attention_tflops']:>10.1f}"
            f"{r['prepare_host_ms_median']:>12.2f}{str(r['prepare_kernels']):>9}"
            f"{str(r['prepare_h2d_copies']):>6}{r['prepare_peak_bytes'] / 2**20:>10.1f}"
        )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"device": torch.cuda.get_device_name(), "rows": records}, f, indent=2
            )


if __name__ == "__main__":
    main()
