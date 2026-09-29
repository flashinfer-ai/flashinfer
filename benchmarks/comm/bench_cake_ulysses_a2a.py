"""Benchmark three scatters plus independent gather with cold-L2 CUPTI.

Run on an SM100 NVLink node with, for example:
    torchrun --standalone --nproc-per-node=8 \
        benchmarks/comm/bench_cake_ulysses_a2a.py --json results.json

Both backends use the same input and caller-owned output tensors. Each sample
includes all four staging-to-output copies. CUPTI unavailability is an error.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import warnings

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.pytorch.ulysses_a2a_export.interface import prepare, shapes
from flashinfer.testing import bench_gpu_time_with_cupti


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeat_iters < 1 or args.warmup_iters < 0:
        parser.error("repeat-iters must be positive and warmup-iters nonnegative")

    from cupti import cupti  # noqa: F401
    from importlib.metadata import version

    if int(version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("CUPTI >= 13 is required")
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")

    started = time.monotonic()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rows = []
    try:
        world = dist.get_world_size()
        rank = dist.get_rank()
        selected = [
            row for row in shapes(world) if args.all_shapes or row["performance"]
        ]
        if not selected:
            raise ValueError(f"no portfolio shapes for world size {world}")
        for shape in selected:
            nvlink = prepare(shape, backend="nvlink")
            try:
                nccl = prepare(
                    shape, backend="nccl", inputs=nvlink.inputs, outputs=nvlink.outputs
                )
                try:
                    nvlink.check()
                    nccl.check()
                    results = {}
                    for backend, case in (("nvlink", nvlink), ("nccl", nccl)):
                        samples = bench_gpu_time_with_cupti(
                            case.run,
                            dry_run_iters=args.warmup_iters,
                            repeat_iters=args.repeat_iters,
                            cold_l2_cache=True,
                            use_cuda_graph=False,
                            aggregate_op=max,
                        )
                        results[backend] = {
                            "median_ms": statistics.median(samples),
                            "samples_ms": samples,
                        }
                    rows.append(
                        {
                            "shape": shape,
                            "correctness": "bit_exact",
                            "timing_backend": "cupti",
                            "cold_l2_cache": True,
                            "rank_aggregation": "max_then_median",
                            "results": results,
                            "speedup_over_nccl": results["nccl"]["median_ms"]
                            / results["nvlink"]["median_ms"],
                        }
                    )
                finally:
                    nccl.close()
            finally:
                nvlink.close()
        if rank == 0:
            report = {
                "world_size": world,
                "gpu": torch.cuda.get_device_name(),
                "compute_capability": torch.cuda.get_device_capability(),
                "runtime_seconds": time.monotonic() - started,
                "rows": rows,
            }
            encoded = json.dumps(report, indent=2)
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(encoded + "\n")
            print(encoded)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
