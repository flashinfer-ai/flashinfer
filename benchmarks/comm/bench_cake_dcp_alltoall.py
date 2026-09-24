"""Benchmark the DCP all-to-all exchange with cold-L2 CUPTI timing.

Run on an SM100/SM103 MNNVL node, for example:
    torchrun --standalone --nproc-per-node=4 \
        benchmarks/comm/bench_cake_dcp_alltoall.py --json results.json

Every rank drives the same preallocated-output call; the reported time is the
slowest rank per shape (the collective completes when the last rank does).
Build with an exact ``FLASHINFER_CUDA_ARCH_LIST`` (``10.0a`` or ``10.3a``) to
measure the generated kernels, or any other list for the portable helix path.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flashinfer.comm import (  # noqa: E402
    decode_cp_a2a_alltoall,
    decode_cp_a2a_allocate_mnnvl_workspace,
    decode_cp_a2a_init_workspace,
)
from flashinfer.comm.comm_backend import TorchDistBackend  # noqa: E402
from flashinfer.comm.mapping import Mapping  # noqa: E402
from flashinfer.comm.mnnvl import MnnvlConfig  # noqa: E402
from flashinfer.testing import bench_gpu_time_with_cupti  # noqa: E402

BATCHES = (1, 16, 64, 128)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--stats-dim", type=int, default=2)
    parser.add_argument("--repeat-iters", type=int, default=50)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    started = time.monotonic()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    # Coordinate over gloo: the exchange itself runs on the MNNVL workspace, and
    # the backend only needs object collectives and barriers. Keeping NCCL (and
    # its watchdog thread) out of the process avoids racing the CUPTI session.
    dist.init_process_group("gloo")
    rows = []
    try:
        world = dist.get_world_size()
        mapping = Mapping(
            world_size=world, rank=rank, cp_size=world, tp_size=1, pp_size=1
        )
        workspace = decode_cp_a2a_allocate_mnnvl_workspace(
            mapping, mnnvl_config=MnnvlConfig(comm_backend=TorchDistBackend())
        )
        decode_cp_a2a_init_workspace(workspace, rank, world)
        torch.cuda.synchronize()
        dist.barrier()

        for batch in args.batches:
            torch.manual_seed(0xA2A + rank)
            partial_o = torch.randn(
                batch, world, args.head_dim, dtype=torch.bfloat16, device="cuda"
            )
            softmax_stats = torch.randn(
                batch, world, args.stats_dim, dtype=torch.float32, device="cuda"
            )
            out = (torch.empty_like(partial_o), torch.empty_like(softmax_stats))

            def run():
                decode_cp_a2a_alltoall(
                    partial_o, softmax_stats, workspace, rank, world, out=out
                )

            dist.barrier()
            samples = bench_gpu_time_with_cupti(
                run,
                dry_run_iters=args.warmup_iters,
                repeat_iters=args.repeat_iters,
                cold_l2_cache=True,
            )
            local_median = torch.tensor(
                [sorted(samples)[len(samples) // 2]], dtype=torch.float64
            )
            dist.all_reduce(local_median, op=dist.ReduceOp.MAX)
            bytes_per_rank = (
                partial_o.numel() * partial_o.element_size()
                + softmax_stats.numel() * softmax_stats.element_size()
            )
            row = {
                "cp_size": world,
                "batch": batch,
                "head_dim": args.head_dim,
                "stats_dim": args.stats_dim,
                "median_us_max_rank": local_median.item() * 1e3,
                "payload_bytes_per_rank": bytes_per_rank,
            }
            rows.append(row)
            if rank == 0:
                print(
                    f"cp{world} b{batch} d{args.head_dim} s{args.stats_dim}: "
                    f"{row['median_us_max_rank']:.3f} us (slowest rank median)"
                )
    finally:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0 and args.json:
        args.json.write_text(
            json.dumps(
                {"rows": rows, "elapsed_s": time.monotonic() - started}, indent=2
            )
        )


if __name__ == "__main__":
    main()
