"""Benchmark three scatters plus independent gather with cold-L2 CUPTI.

Run on an SM100/SM103 NVLink node with, for example:
    torchrun --standalone --nproc-per-node=8 \
        benchmarks/comm/bench_cake_ulysses_a2a.py --json results.json

Three arms share the same input and caller-owned output tensors: the Cake
package route of the NVLink backend (``cake``), FlashInfer's portable NVLink
kernel (``portable``) and the NCCL backend (``nccl``).  A fourth arm,
``cake_registered``, reads the same inputs and writes outputs from the
communicator's ``allocate_output`` (the Cake direct route).  Each sample is one
``PreparedCase.run`` (three ``scatter_heads`` and one ``gather_heads``) and
includes every copy the arm's kernel issues.  Every sample starts rank-aligned:
the ranks synchronize their devices and meet at a host-side (gloo) barrier
before each call, so the synchronizing kernels are not timed against another
rank's launch skew; the per-sample statistic is the rank maximum, the row's the
median.  CUPTI unavailability is an error.  Add ``--all-shapes`` to measure
every portfolio row of the launched world.  The ``cake`` and
``cake_registered`` arms are reported as ``null`` with a reason where the
package delivers no route for the device.
"""

import argparse
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flashinfer.testing import bench_gpu_time_with_cupti

from tests.comm.cake_ulysses_a2a_fixture import (
    cake_direct_available,
    cake_route_available,
    prepare,
    shapes,
)


def _aligned(case, barrier_group):
    """``case.run`` behind a device synchronize and a host barrier: every rank starts the sample together."""

    def run():
        torch.cuda.synchronize()
        dist.barrier(group=barrier_group)
        return case.run()

    return run


def _measure(case, barrier_group, args):
    samples = bench_gpu_time_with_cupti(
        _aligned(case, barrier_group),
        dry_run_iters=args.warmup_iters,
        repeat_iters=args.repeat_iters,
        cold_l2_cache=True,
        use_cuda_graph=False,
        aggregate_op=max,
    )
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--repeat-iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeat_iters < 1 or args.warmup_iters < 0:
        parser.error("repeat-iters must be positive and warmup-iters nonnegative")

    from importlib.metadata import version

    from cupti import cupti  # noqa: F401

    if int(version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("CUPTI >= 13 is required")
    warnings.filterwarnings("error", message=".*Falling back to CUDA events.*")

    started = time.monotonic()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    # the rank-alignment barrier is host-side so it adds no GPU activity to a sample
    barrier_group = dist.new_group(backend="gloo")
    rows = []
    try:
        world = dist.get_world_size()
        rank = dist.get_rank()
        selected = [
            row for row in shapes(world) if args.all_shapes or row["performance"]
        ]
        if not selected:
            raise ValueError(f"no portfolio shapes for world size {world}")
        cake_available = cake_route_available(world)
        direct_available = cake_available and cake_direct_available(world)
        for shape in selected:
            portable = prepare(shape, backend="nvlink", nvlink_kernel="portable")
            cases = {"portable": portable}
            try:
                if cake_available:
                    cases["cake"] = prepare(
                        shape,
                        backend="nvlink",
                        nvlink_kernel="cake",
                        inputs=portable.inputs,
                        outputs=portable.outputs,
                    )
                if direct_available:
                    cases["cake_registered"] = prepare(
                        shape,
                        backend="nvlink",
                        nvlink_kernel="cake",
                        inputs=portable.inputs,
                        registered=True,
                    )
                cases["nccl"] = prepare(
                    shape,
                    backend="nccl",
                    inputs=portable.inputs,
                    outputs=portable.outputs,
                )
                results = {}
                for _name, case in cases.items():
                    case.check()
                for name, case in cases.items():
                    results[name] = _measure(case, barrier_group, args)
                if not cake_available:
                    results["cake"] = None
                if not direct_available:
                    results["cake_registered"] = None
                row = {
                    "shape": shape,
                    "correctness": "bit_exact",
                    "timing_backend": "cupti",
                    "cold_l2_cache": True,
                    "rank_alignment": "device_sync_then_host_barrier_per_sample",
                    "rank_aggregation": "max_then_median",
                    "results": results,
                    "speedup_portable_over_nccl": results["nccl"]["median_ms"]
                    / results["portable"]["median_ms"],
                }
                if cake_available:
                    row["speedup_cake_over_nccl"] = (
                        results["nccl"]["median_ms"] / results["cake"]["median_ms"]
                    )
                    row["speedup_cake_over_portable"] = (
                        results["portable"]["median_ms"] / results["cake"]["median_ms"]
                    )
                else:
                    row["cake_unavailable"] = (
                        "the package delivers no route for this device and dtype"
                    )
                if direct_available:
                    row["speedup_cake_registered_over_nccl"] = (
                        results["nccl"]["median_ms"]
                        / results["cake_registered"]["median_ms"]
                    )
                    row["speedup_cake_registered_over_cake"] = (
                        results["cake"]["median_ms"]
                        / results["cake_registered"]["median_ms"]
                    )
                else:
                    row["cake_registered_unavailable"] = (
                        "the package delivers no direct route for this device and dtype"
                    )
                rows.append(row)
            finally:
                for case in reversed(list(cases.values())):
                    case.close()
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
