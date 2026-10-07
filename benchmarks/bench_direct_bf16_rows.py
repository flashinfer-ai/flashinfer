# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare the original and expanded row schedules through mm_bf16.

Example: python benchmarks/bench_direct_bf16_rows.py --m 16 --k 5120 --n 96 --out rows.json --cupti
Run each shape in a fresh process. The baseline retains every original tactic
and only removes the newly added smaller row schedules. Both sides autotune
all available CuTe DSL algorithms; no particular winner is forced.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import flashinfer
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm import gemm_base
from flashinfer.gemm.kernels import dense_bf16_gemm_direct as direct

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=16)
parser.add_argument("--k", type=int, default=5120)
parser.add_argument("--n", type=int, default=96)
parser.add_argument("--out", required=True)
parser.add_argument("--cupti", action="store_true")
parser.add_argument(
    "--direct-schedules",
    action="store_true",
    help="Also measure every original and added direct tactic to separate kernel gains from selection",
)
parser.add_argument(
    "--default-tuning",
    action="store_true",
    help="Use default buckets and profiling repeat policy",
)
args = parser.parse_args()
tuning_kwargs = (
    {}
    if args.default_tuning
    else dict(tuning_buckets=[args.m], cuda_graph_profile_replays=20)
)
if args.cupti:
    from cupti import cupti  # noqa: F401 - fail closed if unavailable
    from flashinfer.testing import bench_gpu_time

torch.manual_seed(123)
a = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
b = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16).T
ref = (a.double() @ b.double()).bfloat16()
expanded = direct.autotune_tactics


def original_row_space(m, n, k):
    default = direct.default_tactic(m, n, k)
    return [t for t in expanded(m, n, k) if t.rows_per_block == default.rows_per_block]


result = dict(
    args=vars(args),
    gpu=str(torch.cuda.get_device_properties(0)),
    torch=torch.__version__,
    cuda=torch.version.cuda,
    flashinfer=flashinfer.__version__,
    providers={},
)
graphs = {}
providers = [
    "original-rows",
    "expanded-rows",
    "cublaslt",
    "tinygemm",
    "cudnn",
    "cutlass",
    "cutile",
]
fixed_tactics = {}
if args.direct_schedules:
    fixed_tactics = {
        f"direct-{t.block_size}-{t.outputs_per_block}-{t.rows_per_block}": t
        for t in expanded(args.m, args.n, args.k)
    }
    providers.extend(fixed_tactics)
try:
    for name in providers:
        direct.autotune_tactics = (
            original_row_space if name == "original-rows" else expanded
        )
        gemm_base._cute_dsl_direct_bf16_gemm_runner.cache_clear()
        tuner = AutoTuner.get()
        tuner.clear_cache()
        backend = "cute-dsl" if name.endswith("-rows") else name
        y = torch.empty_like(ref)

        fixed_tactic = fixed_tactics.get(name)

        def fn():
            if fixed_tactic is not None:
                return direct.run_direct_dense(a, b, y, False, fixed_tactic)
            # Match the tuning policy when looking up the winner during capture.
            with autotune(False, **tuning_kwargs):
                return flashinfer.mm_bf16(a, b, out=y, backend=backend)

        try:
            with autotune(**tuning_kwargs):
                fn()
            torch.testing.assert_close(fn(), ref, rtol=0.008, atol=0.005)
            selected = (
                repr(fixed_tactic)
                if fixed_tactic is not None
                else repr(tuner.profiling_cache)
            )
            single, repeated = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(single):
                fn()
            with torch.cuda.graph(repeated):
                for _ in range(100):
                    fn()
            graphs[name] = single, repeated, y
            result["providers"][name] = dict(selected=selected, hot_us=[])
        except Exception as error:
            result["providers"][name] = dict(error=str(error))
            if name.endswith("-rows"):
                raise
finally:
    direct.autotune_tactics = expanded
    gemm_base._cute_dsl_direct_bf16_gemm_runner.cache_clear()

for repeat in range(7):
    names = list(graphs)
    if repeat % 2:
        names.reverse()
    for name in names:
        single, repeated, y = graphs[name]
        repeated.replay()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        repeated.replay()
        end.record()
        end.synchronize()
        result["providers"][name]["hot_us"].append(start.elapsed_time(end) * 10)
        y.fill_(float("nan"))
        single.replay()
        torch.testing.assert_close(y, ref, rtol=0.008, atol=0.005)

for name, (single, _, y) in graphs.items():
    row = result["providers"][name]
    row["hot_median_us"] = statistics.median(row["hot_us"])
    if args.cupti:
        samples = bench_gpu_time(
            single.replay,
            enable_cupti=True,
            use_cuda_graph=False,
            dry_run_iters=5,
            repeat_iters=30,
            cold_l2_cache=True,
        )
        row["cold_cupti_us"] = [t * 1000 for t in samples]
        row["cold_cupti_median_us"] = statistics.median(samples) * 1000
        torch.testing.assert_close(y, ref, rtol=0.008, atol=0.005)
Path(args.out).write_text(json.dumps(result, indent=2))
print(
    {
        name: {
            k: v
            for k, v in row.items()
            if k in ("hot_median_us", "cold_cupti_median_us", "error")
        }
        for name, row in result["providers"].items()
    }
)

# Release captured resources while the CUDA/Python runtimes are still alive.
for single, repeated, _ in graphs.values():
    single.reset()
    repeated.reset()
graphs.clear()
torch.cuda.synchronize()
