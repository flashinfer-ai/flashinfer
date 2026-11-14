# SPDX-License-Identifier: Apache-2.0
"""NCU-ready HC-only harness; no host/event timings are reported.

Warm the exact source/shape with JIT enabled first, verify Ninja reports no work,
then profile the measurement window, for example::

    ncu --profile-from-start off \
      --replay-mode application --cache-control none --clock-control none \
      --kernel-name-base demangled --kernel-name 'regex:.*' \
      --metrics gpu__time_duration.sum --csv --log-file hc.csv \
      python benchmarks/experimental/bench_hc_sum_sm100.py --profile --rows 24

NVCC JIT modules in this FlashInfer version validate freshness through Ninja on
the first load in each process, even when an artifact exists. The harness warms
before profiling and disables new JIT before entering the measurement window.
The measured work is mix followed by apply with a prepared core output.
Bootstrap, packing, allocation, warmup and correctness checks are outside the
profiler window. Sum all stages within each invocation before aggregating.
"""

import argparse
import json
import os

import torch

from flashinfer.hyperconnection import prepare_hc_sum_weights


def inputs(rows, hidden, lowrank, dtype):
    torch.manual_seed(42)
    kw = dict(device="cuda", dtype=dtype)
    down = torch.randn(lowrank, 4 * hidden, **kw) * 0.01
    up = torch.randn(4 * hidden, lowrank, **kw) * 0.01
    inject = torch.randn(4, 4 * hidden, **kw) * 0.01
    norm = torch.randn(4, hidden, **kw) * 0.01
    residual = torch.randn(rows, 4 * hidden, **kw)
    core = torch.randn(rows, hidden, **kw)
    return down, up, inject, norm, residual, core


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--lowrank", type=int, default=320)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    dtype = getattr(torch, args.dtype)
    down, up, inject, norm, residual, core = inputs(
        args.rows, args.hidden_size, args.lowrank, dtype
    )
    plan = prepare_hc_sum_weights(down, up, inject, norm)
    ws = plan.allocate_workspace(args.rows, deterministic=args.deterministic)
    sums = plan.bootstrap(residual)

    def run():
        plan.mix(residual, sums, ws)
        return plan.apply(core, residual, ws)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    os.environ["FLASHINFER_DISABLE_JIT"] = "1"
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for index in range(args.iterations):
        with torch.cuda.nvtx.range(f"HC/T{args.rows}/iteration{index}"):
            run()
    torch.cuda.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    assert torch.isfinite(ws.residual).all() and torch.isfinite(ws.sum_sq).all()
    print(
        json.dumps(
            {
                "rows": args.rows,
                "hidden_size": args.hidden_size,
                "lowrank": args.lowrank,
                "dtype": args.dtype,
                "deterministic": args.deterministic,
                "iterations": args.iterations,
                "gpu": torch.cuda.get_device_name(),
                "profile": args.profile,
                "finite": True,
            }
        )
    )


if __name__ == "__main__":
    main()
