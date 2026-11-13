# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DCP All-to-All Microbenchmark: Native LL128 FIFO vs NCCL Baseline

Measures single kernel-level latency for the DCP A2A communication op:
  - Native: decode_cp_a2a_alltoall (fused LL128 FIFO kernel via MNNVL)
  - NCCL baseline: 2x torch.distributed.all_to_all_single (partial_o + softmax_stats)

This is NOT an end-to-end pipeline benchmark. It measures raw communication
kernel time only.

Launch:
    mpirun --allow-run-as-root --oversubscribe -np 4 \
        python benchmarks/bench_dcp_alltoall.py

    mpirun --allow-run-as-root --oversubscribe -np 2 \
        python benchmarks/bench_dcp_alltoall.py --batch_sizes 1 16 64

Options:
    --batch_sizes     : Batch sizes to benchmark (default: 1 16 64 128)
    --head_dim        : Head dimension D (default: 128)
    --stats_dim       : Stats dimension S (default: 2)
    --warmup          : Warmup iterations (default: 50)
    --iters           : Timed iterations (default: 200)
    --skip_nccl       : Skip NCCL baseline (only run native)
    --skip_native     : Skip native (only run NCCL baseline)

Requires:
    - SM90+ GPU (Hopper/Blackwell)
    - MNNVL support (multi-GPU fabric memory)
    - mpi4py
"""

import argparse
import os
import socket

import numpy as np
import pynvml
import torch
import torch.distributed as dist
from mpi4py import MPI

from flashinfer.comm import (
    decode_cp_a2a_alltoall,
    decode_cp_a2a_init_workspace,
    decode_cp_a2a_workspace_size,
)
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import MnnvlMemory, MpiComm


def _to_torch(t):
    """Convert a tvm_ffi.core.Tensor (or any DLPack object) to torch.Tensor."""
    if isinstance(t, torch.Tensor):
        return t
    return torch.from_dlpack(t)


def setup_mpi():
    """Initialize MPI and set CUDA device. Returns (rank, world_size, comm, local_rank)."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Compute local rank from hostname
    hostname = socket.gethostname()
    all_hostnames = comm.allgather(hostname)
    local_rank = sum(1 for i in range(rank) if all_hostnames[i] == hostname)
    torch.cuda.set_device(local_rank)

    return rank, world_size, comm, local_rank


def setup_nccl(rank, world_size):
    """Initialize torch.distributed NCCL backend."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def allocate_mnnvl_workspace(rank, cp_size, mpi_comm):
    """Allocate MNNVL workspace for native DCP A2A."""
    pynvml.nvmlInit()
    MnnvlMemory.initialize()
    MnnvlMemory.comm = MpiComm()

    mapping = Mapping(
        world_size=cp_size,
        rank=rank,
        cp_size=cp_size,
        tp_size=1,
        pp_size=1,
    )

    ws_bytes = decode_cp_a2a_workspace_size(cp_size)
    mnnvl_mem = MnnvlMemory(mapping, ws_bytes)
    workspace = mnnvl_mem.as_torch_strided_tensor(torch.int64)
    workspace._mnnvl_mem = mnnvl_mem  # prevent GC
    return workspace


def bench_native(
    workspace,
    rank,
    cp_size,
    batch_size,
    head_dim,
    stats_dim,
    dtype,
    warmup,
    iters,
    mpi_comm,
):
    """Benchmark native decode_cp_a2a_alltoall. Returns list of per-iteration times in ms."""
    # Init workspace once — FIFO supports reuse across iterations
    decode_cp_a2a_init_workspace(workspace, rank, cp_size)
    torch.cuda.synchronize()
    mpi_comm.Barrier()

    partial_o = torch.randn(batch_size, cp_size, head_dim, dtype=dtype, device="cuda")
    softmax_stats = torch.randn(
        batch_size, cp_size, stats_dim, dtype=torch.float32, device="cuda"
    )

    # Warmup
    for _ in range(warmup):
        recv_o, recv_s = decode_cp_a2a_alltoall(
            partial_o, softmax_stats, workspace, rank, cp_size
        )
        torch.cuda.synchronize()
        mpi_comm.Barrier()

    # Timed iterations
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        mpi_comm.Barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        recv_o, recv_s = decode_cp_a2a_alltoall(
            partial_o, softmax_stats, workspace, rank, cp_size
        )
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))
        mpi_comm.Barrier()

    return times


def bench_nccl(
    rank, cp_size, batch_size, head_dim, stats_dim, dtype, warmup, iters, mpi_comm
):
    """Benchmark NCCL 2x all_to_all_single. Returns list of per-iteration times in ms."""
    group = dist.group.WORLD

    # partial_o: [B, cp_size, D] — each rank sends chunk [B, 1, D] to each peer
    partial_o = torch.randn(batch_size, cp_size, head_dim, dtype=dtype, device="cuda")
    softmax_stats = torch.randn(
        batch_size, cp_size, stats_dim, dtype=torch.float32, device="cuda"
    )

    # Flatten for all_to_all_single: [B * cp_size * D] split into cp_size equal chunks
    send_o = partial_o.contiguous().view(-1)
    recv_o = torch.empty_like(send_o)
    send_s = softmax_stats.contiguous().view(-1)
    recv_s = torch.empty_like(send_s)

    # Warmup
    for _ in range(warmup):
        dist.all_to_all_single(recv_o, send_o, group=group)
        dist.all_to_all_single(recv_s, send_s, group=group)
        torch.cuda.synchronize()
        mpi_comm.Barrier()

    # Timed iterations
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        mpi_comm.Barrier()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        dist.all_to_all_single(recv_o, send_o, group=group)
        dist.all_to_all_single(recv_s, send_s, group=group)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))
        mpi_comm.Barrier()

    return times


def compute_stats(all_times, iters):
    """Compute p50/p95/mean from gathered per-rank times.

    Takes max across ranks per iteration (communication is synchronous),
    then reports percentiles.
    """
    per_iter_max = [
        max(all_times[r][i] for r in range(len(all_times))) for i in range(iters)
    ]
    return {
        "p50": float(np.percentile(per_iter_max, 50)),
        "p95": float(np.percentile(per_iter_max, 95)),
        "mean": float(np.mean(per_iter_max)),
    }


def main():
    parser = argparse.ArgumentParser(description="DCP A2A Microbenchmark")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 16, 64, 128])
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--stats_dim", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--skip_nccl", action="store_true")
    parser.add_argument("--skip_native", action="store_true")
    args = parser.parse_args()

    rank, world_size, mpi_comm, local_rank = setup_mpi()
    cp_size = world_size
    dtype = torch.bfloat16

    if rank == 0:
        print(f"=== DCP A2A Benchmark (cp_size={cp_size}, {cp_size} GPUs) ===")
        print()

    # Initialize NCCL if needed
    if not args.skip_nccl:
        setup_nccl(rank, world_size)

    # Allocate MNNVL workspace if needed
    workspace = None
    if not args.skip_native:
        workspace = allocate_mnnvl_workspace(rank, cp_size, mpi_comm)

    for batch_size in args.batch_sizes:
        nccl_stats = None
        native_stats = None

        # NCCL baseline
        if not args.skip_nccl:
            times = bench_nccl(
                rank,
                cp_size,
                batch_size,
                args.head_dim,
                args.stats_dim,
                dtype,
                args.warmup,
                args.iters,
                mpi_comm,
            )
            all_times = mpi_comm.allgather(times)
            nccl_stats = compute_stats(all_times, args.iters)

        # Native DCP A2A
        if not args.skip_native and workspace is not None:
            times = bench_native(
                workspace,
                rank,
                cp_size,
                batch_size,
                args.head_dim,
                args.stats_dim,
                dtype,
                args.warmup,
                args.iters,
                mpi_comm,
            )
            all_times = mpi_comm.allgather(times)
            native_stats = compute_stats(all_times, args.iters)

        # Print results (rank 0 only)
        if rank == 0:
            print(
                f"  batch={batch_size}, head_dim={args.head_dim}, "
                f"stats_dim={args.stats_dim}, dtype=bf16"
            )
            if nccl_stats:
                print(
                    f"    NCCL (2x all_to_all_single):  "
                    f"p50={nccl_stats['p50']:.3f}ms  "
                    f"p95={nccl_stats['p95']:.3f}ms  "
                    f"mean={nccl_stats['mean']:.3f}ms"
                )
            if native_stats:
                print(
                    f"    Native (decode_cp_a2a_alltoall):    "
                    f"p50={native_stats['p50']:.3f}ms  "
                    f"p95={native_stats['p95']:.3f}ms  "
                    f"mean={native_stats['mean']:.3f}ms"
                )
            if nccl_stats and native_stats and native_stats["p50"] > 0:
                speedup = nccl_stats["p50"] / native_stats["p50"]
                print(f"    Speedup: {speedup:.1f}x")
            print()

    # Cleanup
    if not args.skip_nccl and dist.is_initialized():
        dist.destroy_process_group()

    # Prevent segfault at exit: MnnvlMemory uses a bump allocator that
    # doesn't support individual frees. If Python GC destroys the workspace
    # tensor during interpreter shutdown, it triggers a segfault in
    # TensorImpl::~TensorImpl. Calling os._exit() skips GC entirely.
    mpi_comm.Barrier()
    MPI.Finalize()
    os._exit(0)


if __name__ == "__main__":
    main()
