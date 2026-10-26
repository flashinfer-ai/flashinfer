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

"""Benchmark the experimental FP8 attention backend (SM90).

For every ``[B, H, S, 128]`` row this reports the complete BF16 -> BF16 call
(statistics + quantization + attention, replayed from a CUDA graph; CUPTI
kernel time with a cold L2, median over the measured iterations), its
TFLOP/s over the dense score and output products, the preparation host time
(``prepare_fp8_attention``: validation, workspace carving, binding) and the
workspace bytes.  ``--baseline sdpa`` adds PyTorch's BF16
``scaled_dot_product_attention`` on the same tensors under the same protocol.

Usage::

    python benchmarks/bench_cake_fp8_attention.py [--rows R1 R2 ...]
        [--layout HND|NHD] [--baseline none|sdpa] [--json out.json]
"""

import argparse
import json
import statistics
import time

import torch

from flashinfer.prefill import fp8_attention_workspace_bytes, prepare_fp8_attention
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.utils import get_compute_capability

# (batch, seqlen, heads): the owner's rows of the dense video-diffusion workload.
ROWS = {
    "R1": (1, 4800, 40),
    "R2": (1, 32760, 40),
    "R3": (1, 75600, 40),
    "R4": (1, 32760, 5),
    "R5": (1, 75600, 5),
    "R6": (2, 32760, 40),
    "R7": (1, 32760, 12),
}
SUPPORTED = {(9, 0): "sm_90a"}


def _inputs(batch, heads, seqlen, layout, seed=42):
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seqlen, 128)
    tensors = []
    for _ in range(3):
        x = torch.randn(shape, generator=generator).to(
            dtype=torch.bfloat16, device="cuda"
        )
        if layout == "NHD":
            x = x.transpose(1, 2).contiguous().transpose(1, 2)
        tensors.append(x)
    out = (
        torch.empty_like(tensors[0])
        if layout == "HND"
        else torch.empty_like(tensors[0].transpose(1, 2)).transpose(1, 2)
    )
    return (*tensors, out)


def _timed(fn):
    times = bench_gpu_time_with_cupti(fn, cold_l2_cache=True, use_cuda_graph=True)
    return float(statistics.median(times))


def bench_row(name, layout, baseline, prepare_repeats):
    batch, seqlen, heads = ROWS[name]
    q, k, v, out = _inputs(batch, heads, seqlen, layout)
    runner = prepare_fp8_attention(q, k, v, out, backend="cake")
    runner()  # warm the JIT libraries and the cached device facts
    torch.cuda.synchronize()
    host_ms = []
    for _ in range(prepare_repeats):
        start = time.perf_counter()
        runner = prepare_fp8_attention(
            q, k, v, out, backend="cake", workspace=runner.workspace
        )
        torch.cuda.synchronize()
        host_ms.append((time.perf_counter() - start) * 1e3)
    record = {
        "row": name,
        "batch": batch,
        "heads": heads,
        "seqlen": seqlen,
        "layout": layout,
        "arch": runner.route_metadata["arch"],
        "recipe": runner.route_metadata["recipe"],
        "qk_format": runner.route_metadata["qk_format"],
        "workspace_bytes": fp8_attention_workspace_bytes(
            batch, heads, seqlen, q.device
        ),
        "prepare_host_ms_median": statistics.median(host_ms),
        "call_ms_median": _timed(runner.launch),
    }
    flops = 4.0 * batch * heads * seqlen * seqlen * 128
    record["call_tflops"] = flops / (record["call_ms_median"] * 1e-3) / 1e12
    if baseline == "sdpa":
        sdpa = torch.nn.functional.scaled_dot_product_attention
        record["sdpa_ms_median"] = _timed(lambda: sdpa(q, k, v))
        record["speedup_vs_sdpa"] = record["sdpa_ms_median"] / record["call_ms_median"]
    return record


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
    parser.add_argument("--layout", default="HND", choices=("HND", "NHD"))
    parser.add_argument("--baseline", default="none", choices=("none", "sdpa"))
    parser.add_argument("--prepare-repeats", type=int, default=5)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()
    capability = get_compute_capability(
        torch.device("cuda", torch.cuda.current_device())
    )
    if tuple(capability) not in SUPPORTED:
        raise SystemExit(f"FP8 attention requires an SM90 GPU, got {capability}")
    records = [
        bench_row(name, args.layout, args.baseline, args.prepare_repeats)
        for name in args.rows
    ]
    header = f"{'row':<5}{'B':>3}{'S':>7}{'H':>4}{'call ms':>10}{'TFLOP/s':>9}{'prep ms':>9}{'ws MiB':>8}"
    if args.baseline == "sdpa":
        header += f"{'sdpa ms':>10}{'speedup':>9}"
    print(
        f"{torch.cuda.get_device_name()}  {records[0]['arch']}  CUDA {torch.version.cuda}  layout {args.layout}"
    )
    print(header)
    for r in records:
        line = (
            f"{r['row']:<5}{r['batch']:>3}{r['seqlen']:>7}{r['heads']:>4}{r['call_ms_median']:>10.3f}"
            f"{r['call_tflops']:>9.1f}{r['prepare_host_ms_median']:>9.2f}{r['workspace_bytes'] / 2**20:>8.1f}"
        )
        if args.baseline == "sdpa":
            line += f"{r['sdpa_ms_median']:>10.3f}{r['speedup_vs_sdpa']:>9.2f}"
        print(line)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {
                    "device": torch.cuda.get_device_name(),
                    "cuda": torch.version.cuda,
                    "rows": records,
                },
                handle,
                indent=1,
            )


if __name__ == "__main__":
    main()
