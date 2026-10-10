# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit-only FP8 SIMT benchmark, with hot and evicted CUDA graph timings.

Run: python benchmarks/bench_frost_low_latency_fp8.py --out result.json
M is the number of independent activation rows, not the BMM batch dimension.
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
import flashinfer
from flashinfer.autotuner import autotune

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument(
    "--cupti",
    action="store_true",
    help="Also collect GPU kernel spans with CUPTI",
)
p.add_argument("--rows", default="1,2,4,8,16,32,64")
p.add_argument("--shapes", default="1024x1024,4096x4096,6144x5120,8192x8192")
p.add_argument(
    "--backends",
    default="cublas,cutlass,cudnn,trtllm_low_latency,cutedsl_low_latency,frost-low-latency",
)
a = p.parse_args()
# The profiler's teardown affects subsequent cuBLAS capture in this environment.
# Isolate each profiled shape so every provider gets a fresh CUDA context.
if a.cupti and len(a.rows.split(",")) * len(a.shapes.split(",")) > 1:
    combined = None
    with tempfile.TemporaryDirectory(prefix="frost-fp8-") as temp:
        for shape in a.shapes.split(","):
            for rows in a.rows.split(","):
                output = Path(temp) / "case.json"
                subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--out",
                        str(output),
                        "--rows",
                        rows,
                        "--shapes",
                        shape,
                        "--backends",
                        a.backends,
                        "--cupti",
                    ],
                    check=True,
                )
                case = json.loads(output.read_text())
                if combined is None:
                    combined = {**case, "args": vars(a), "cases": []}
                combined["cases"].extend(case["cases"])
                Path(a.out).write_text(json.dumps(combined, indent=2))
    raise SystemExit(0)
if a.cupti:
    from cupti import cupti  # noqa: F401 - fail closed if profiling is unavailable
    from flashinfer.testing import bench_gpu_time

torch.manual_seed(123)
result = dict(
    gpu=str(torch.cuda.get_device_properties(0)),
    torch=torch.__version__,
    cuda=torch.version.cuda,
    flashinfer=flashinfer.__version__,
    args=vars(a),
    cases=[],
)
for k, n in [map(int, s.split("x")) for s in a.shapes.split(",")]:
    for m in map(int, a.rows.split(",")):
        x = torch.randn(1, m, k, device="cuda").to(torch.float8_e4m3fn)
        w = torch.randn(1, n, k, device="cuda").to(torch.float8_e4m3fn).transpose(1, 2)
        sa = torch.tensor([0.25], device="cuda")
        sb = torch.tensor([1.5], device="cuda")
        alpha = sa * sb
        packed_w = None
        if "trtllm_low_latency" in a.backends.split(","):
            packed_w = flashinfer.prepare_low_latency_gemm_weights(w[0].T, {})
        ref = ((x.double() @ w.double()) * alpha.double()).to(torch.bfloat16)
        graphs = {}
        case = dict(m=m, k=k, n=n, backends={})
        for backend in a.backends.split(","):
            if backend == "cutedsl_low_latency" and m > 8:
                continue
            y = torch.empty((1, m, n), device="cuda", dtype=torch.bfloat16)
            if backend in ("cutedsl_low_latency", "trtllm_low_latency"):

                def fn():
                    return flashinfer.mm_fp8(
                        x[0],
                        packed_w if backend == "trtllm_low_latency" else w[0].T,
                        alpha=alpha,
                        out=y[0],
                        backend=backend,
                    )
            else:

                def fn():
                    return flashinfer.bmm_fp8(
                        x, w, sa, sb, torch.bfloat16, out=y, backend=backend
                    )

            try:
                with autotune():
                    fn()
                fn()
                torch.cuda.synchronize()
                torch.testing.assert_close(y, ref, rtol=0.008, atol=0.003)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                # One graph with 100 calls amortizes host submission and event overhead.
                hot_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(hot_graph):
                    for _ in range(100):
                        fn()
                graphs[backend] = (g, hot_graph, y)
                case["backends"][backend] = dict(hot_us=[])
            except Exception as e:
                case["backends"][backend] = dict(error=str(e))
                print("BACKEND_ERROR", m, k, n, backend, repr(e), flush=True)
                if backend == "frost-low-latency":
                    raise
        for repeat in range(7):
            names = list(graphs)
            if repeat % 2:
                names.reverse()
            for name in names:
                g, hot_graph, y = graphs[name]
                g.replay()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                hot_graph.replay()
                end.record()
                end.synchronize()
                case["backends"][name]["hot_us"].append(start.elapsed_time(end) * 10)
                y.fill_(float("nan"))
                g.replay()
                torch.testing.assert_close(y, ref, rtol=0.008, atol=0.003)
        if a.cupti:
            for name, (g, _, y) in graphs.items():
                for cold in (False, True):
                    samples = bench_gpu_time(
                        g.replay,
                        enable_cupti=True,
                        use_cuda_graph=False,  # g.replay already submits a captured graph
                        dry_run_iters=5,
                        repeat_iters=30,
                        cold_l2_cache=cold,
                    )
                    mode = "cold" if cold else "hot"
                    case["backends"][name][f"cupti_{mode}_us"] = [
                        v * 1000 for v in samples
                    ]
                    case["backends"][name][f"cupti_{mode}_median_us"] = (
                        statistics.median(samples) * 1000
                    )
                torch.testing.assert_close(y, ref, rtol=0.008, atol=0.003)
        for v in case["backends"].values():
            if "hot_us" in v:
                v["hot_median_us"] = statistics.median(v["hot_us"])
        result["cases"].append(case)
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(
            "CASE",
            {key: val for key, val in case.items() if key != "backends"},
            {
                name: {k: v for k, v in val.items() if not isinstance(v, list)}
                for name, val in case["backends"].items()
            },
            flush=True,
        )
