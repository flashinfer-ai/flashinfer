# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare standalone Lowp orchestration with the prepared communicator API.

Run from the FlashInfer repository root with the checkout on PYTHONPATH:
  torchrun --standalone --nproc-per-node=8 benchmarks/comm/bench_ulysses_qkv.py \
    --lengths 4736 4737 --output ulysses-qkv-microbench.json

Both variants use preallocated payload, stats-gather and six final outputs.
Both retain existing local_stats/finalize_stats temporary allocations and
copy the per-rank V scale into independent final storage. This compares
the same output-ownership contract. Preparation/JIT is outside timing.

CUDA event spans include launch starvation and collective waits; they are
NOT sums of kernel execution times. CPU enqueue time can include blocking
inside NCCL/PyTorch. Synchronized wall time measures complete batch latency.
No attention, output gather, CUDA graphs, profiler or video generation.
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from flashinfer.comm import UlyssesCommunicator
from flashinfer.comm import _ulysses_lowp as lowp


def measure(fn, iterations):
    # This is a batch boundary, outside the timed region. The operation
    # itself has exactly one statistics AG and one payload A2A per call.
    dist.barrier()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    begin = time.perf_counter()
    for _ in range(iterations):
        fn()
    enqueued = time.perf_counter()
    stop.record()
    stop.synchronize()
    complete = time.perf_counter()
    return {
        "cpu_enqueue_ms": (enqueued - begin) * 1000 / iterations,
        "synchronized_wall_ms": (complete - begin) * 1000 / iterations,
        "cuda_event_span_ms": start.elapsed_time(stop) / iterations,
    }


def benchmark_case(args, rank, world, length):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    batch, heads, head_dim = args.batch, args.heads, args.head_dim
    dtype = getattr(torch, args.dtype)
    shape = (batch, length, heads, head_dim)
    total = length * world
    used = total if args.used is None else args.used
    if not 0 < used <= total:
        raise ValueError("--used must lie in (0, world_size * local_sequence]")
    torch.manual_seed(2101 + rank)
    q, k, v = (torch.randn(shape, device=device, dtype=dtype) for _ in range(3))
    local_live = max(0, min(length, used - rank * length))
    for tensor in (q, k, v):
        tensor[:, local_live:].zero_()
    cap = lowp.capability(device)
    if cap["device_capability"] not in ((9, 0), (12, 0)) or not cap["supported"]:
        raise RuntimeError(f"Requires supported SM90 or SM120 kernels: {cap}")
    layout = getattr(lowp, cap["layout_class"])(head_dim=head_dim)
    local_heads = heads // world
    head_start = rank * local_heads
    spec = layout.payload_spec(
        batch_size=batch, local_sequence=length, num_heads=heads, world_size=world
    )
    q_width, k_width = layout.scale_widths(used)
    send_u8 = torch.empty(
        (world, spec["chunk_bytes"]), dtype=torch.uint8, device=device
    )
    recv_u8 = torch.empty_like(send_u8)
    stats_gather = torch.empty(
        world * batch * heads * (6 * head_dim + 2), dtype=torch.float32, device=device
    )
    standalone_out = (
        torch.empty(
            (batch, total, local_heads, head_dim), dtype=torch.int8, device=device
        ),
        torch.empty(
            (batch, total, local_heads, head_dim), dtype=torch.int8, device=device
        ),
        torch.empty(
            (batch, head_dim, local_heads, spec["padded_sequence"]),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        torch.empty((batch, local_heads, q_width), dtype=torch.float32, device=device),
        torch.empty((batch, local_heads, k_width), dtype=torch.float32, device=device),
        torch.empty((batch, local_heads, head_dim), dtype=torch.float32, device=device),
    )

    def standalone():
        # Match the communicator's device guard and NVTX instrumentation.
        with torch.cuda.device(device):
            with torch.cuda.nvtx.range("lowp_local_stats"):
                send, ctx = layout.local_stats(
                    q, k, v, rank=rank, world_size=world, used_sequence=used
                )
            with torch.cuda.nvtx.range("lowp_stats_allgather"):
                dist.all_gather_into_tensor(stats_gather, send, group=dist.group.WORLD)
            with torch.cuda.nvtx.range("lowp_finalize_stats"):
                stats = layout.finalize_stats(stats_gather, ctx, k)
            with torch.cuda.nvtx.range("lowp_quant_pack"):
                layout.quant_and_pack(q, k, v, stats, out=send_u8)
            with torch.cuda.nvtx.range("lowp_input_a2a"):
                dist.all_to_all_single(recv_u8, send_u8, group=dist.group.WORLD)
            with torch.cuda.nvtx.range("lowp_unpack"):
                layout.unpack_for_sage(
                    recv_u8,
                    batch_size=batch,
                    local_sequence=length,
                    local_heads=local_heads,
                    world_size=world,
                    scale_sequence=used,
                    out=standalone_out[:5],
                )
            with torch.cuda.nvtx.range("lowp_v_scale"):
                standalone_out[5].copy_(
                    stats.v_scale_global[:, head_start : head_start + local_heads]
                )
        return standalone_out

    with UlyssesCommunicator(
        dist.group.WORLD,
        max_elems=q.numel(),
        dtype=q.dtype,
        backend="nccl",
        device=device,
    ) as comm:
        workspace = comm.prepare_qkv(shape, used_sequence=used)
        encoded = comm.scatter_qkv(q, k, v, workspace=workspace)

        def communicator():
            return comm.scatter_qkv(q, k, v, workspace=workspace, out=encoded)

        standalone()
        checks = [
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            for a, b in zip(standalone_out, encoded[:6], strict=True)
        ]
        checks += [
            torch.equal(send_u8, workspace._send_buffer),
            torch.equal(recv_u8, workspace._recv_buffer),
        ]
        all_checks = [None] * world
        dist.all_gather_object(all_checks, checks)
        if not all(all(checks) for checks in all_checks):
            raise AssertionError(f"Byte comparison failed: {all_checks}")
        variants = {
            "standalone_preallocated": standalone,
            "communicator_reused_out": communicator,
        }
        for _ in range(args.warmup):
            standalone()
            communicator()
        torch.cuda.synchronize()
        samples = {name: [] for name in variants}
        for round_index in range(args.rounds):
            names = list(variants) if round_index % 2 == 0 else list(reversed(variants))
            for name in names:
                samples[name].append(measure(variants[name], args.iterations))
        all_samples = [None] * world
        dist.all_gather_object(all_samples, samples)
        storage_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                workspace._send_buffer,
                workspace._recv_buffer,
                workspace._stats_gather,
            )
        )
        result = {
            "shape": list(shape),
            "used_sequence": used,
            "world_size": world,
            "layout": workspace.layout,
            "dtype": str(q.dtype),
            "byte_comparison": "all six outputs and send/recv payloads equal on every rank",
            "workspace_bytes_per_rank": storage_bytes,
            "samples_per_rank": all_samples,
            "summary": {},
        }
        for name in variants:
            result["summary"][name] = {
                metric: statistics.median(
                    max(rank_samples[name][i][metric] for rank_samples in all_samples)
                    for i in range(args.rounds)
                )
                for metric in (
                    "cpu_enqueue_ms",
                    "synchronized_wall_ms",
                    "cuda_event_span_ms",
                )
            }
        old = result["summary"]["standalone_preallocated"]
        new = result["summary"]["communicator_reused_out"]
        result["communicator_overhead_percent"] = {
            metric: (new[metric] / old[metric] - 1) * 100 for metric in old
        }
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4736])
    parser.add_argument(
        "--used", type=int, help="Global live prefix; defaults to the full sequence."
    )
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(
        value <= 0
        for value in [
            *args.lengths,
            args.heads,
            args.batch,
            args.warmup,
            args.iterations,
            args.rounds,
        ]
    ):
        raise ValueError("Lengths, batch, heads and iteration counts must be positive")
    if (
        os.environ.get("FLASHINFER_LOGLEVEL", "0") != "0"
        or os.environ.get("FLASHINFER_TRACE_DUMP", "0") == "1"
    ):
        raise RuntimeError(
            "Disable FlashInfer logging and trace auto-dump for performance measurements"
        )
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world not in (2, 4, 8) or args.heads % world:
            raise ValueError("Requires P2/P4/P8 and --heads divisible by world size")
        report = {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "seed_per_rank": "2101 + rank",
            "warmup": args.warmup,
            "iterations_per_round": args.iterations,
            "rounds": args.rounds,
            "aggregation": "median across rounds of the maximum rank per-call batch-average latency",
            "timing_note": "CUDA event spans include CPU launch starvation and collective waits; not pure kernel sums. Both variants explicitly copy V scale into independent preallocated storage.",
            "cases": [
                benchmark_case(args, rank, world, length) for length in args.lengths
            ],
        }
        if rank == 0:
            brief = {
                **report,
                "cases": [
                    {
                        key: value
                        for key, value in case.items()
                        if key != "samples_per_rank"
                    }
                    for case in report["cases"]
                ],
            }
            print(json.dumps(brief, indent=2), flush=True)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
