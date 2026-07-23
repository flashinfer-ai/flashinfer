# Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
# Copyright (c) 2025 by FlashInfer team.
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

"""
Test script for FP8 quantized two-shot AllReduce.

Run with pytest:
    pytest tests/comm/test_quantized_allreduce.py -vv -s

Run standalone:
    python tests/comm/test_quantized_allreduce.py --correctness
    python tests/comm/test_quantized_allreduce.py --benchmark
    python tests/comm/test_quantized_allreduce.py --scale-sweep
"""

import argparse
import os
import random

# Suppress NCCL watchdog timeout during Triton JIT compilation
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

from flashinfer.comm import quantized_all_reduce
from flashinfer.utils import get_compute_capability


def _supported_world_size() -> int:
    """Largest power-of-2 world size supported by block_size=4096."""
    n = torch.cuda.device_count()
    ws = 1
    while ws * 2 <= n and ws * 2 <= 8:
        ws *= 2
    return ws


def ref_all_reduce_bf16(inp: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Reference BF16 AllReduce using NCCL — the baseline without quantization."""
    out = inp.clone()
    dist.all_reduce(out, group=group)
    return out


def setup(rank: int, world_size: int, port: int):
    """Initialize distributed process group."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    symm_mem.enable_symm_mem_for_group(group.group_name)
    torch.manual_seed(42 + rank)
    return device, group


def run_correctness(rank: int, world_size: int, port: int):
    device, group = setup(rank, world_size, port)

    # Test across sizes: 32KB to 64MB
    sizes_numel = [
        16384,  # 32KB
        65536,  # 128KB
        262144,  # 512KB
        1048576,  # 2MB
        4194304,  # 8MB
        16777216,  # 32MB
        33554432,  # 64MB
    ]

    for numel in sizes_numel:
        inp = torch.randn(numel, dtype=torch.bfloat16, device=device)
        dist.barrier(group)

        out_quant = quantized_all_reduce(inp, group)
        out_ref = ref_all_reduce_bf16(inp, group)  # BF16 baseline (not FP32)

        # FP8 quantization introduces bounded error proportional to world_size
        torch.testing.assert_close(
            out_quant,
            out_ref,
            atol=1.0,
            rtol=0.1,
            msg=f"Failed at numel={numel}, rank={rank}",
        )
        if rank == 0:
            max_err = (out_quant - out_ref).abs().max().item()
            print(f"  numel={numel:>10d}: max_abs_err={max_err:.6f} PASS")

    if rank == 0:
        print("All correctness checks passed!")
    dist.destroy_process_group()


def run_scale_group_sweep(rank: int, world_size: int, port: int):
    device, group = setup(rank, world_size, port)

    numel = 4194304  # 8MB
    inp = torch.randn(numel, dtype=torch.bfloat16, device=device)
    out_ref = ref_all_reduce_bf16(inp, group)  # BF16 baseline (not FP32)

    for sg in [128, 256, 512, 1024]:
        dist.barrier(group)
        out = quantized_all_reduce(inp, group, scale_group=sg)
        max_err = (out - out_ref).abs().max().item()
        if rank == 0:
            print(f"  scale_group={sg:>4d}: max_abs_err={max_err:.6f}")
        torch.testing.assert_close(out, out_ref, atol=1.0, rtol=0.1)

    if rank == 0:
        print("Scale group sweep passed!")
    dist.destroy_process_group()


def run_edge_cases(rank: int, world_size: int, port: int):
    device, group = setup(rank, world_size, port)

    edge_sizes = [
        256 * world_size,  # minimum: SCALE_GROUP * ws (one group per stripe)
        4096,  # exactly one BLOCK_SIZE
        4096 * world_size,  # exactly one stride_per_program
        4096 * world_size + 8,  # one stride + small remainder
        8,  # absolute minimum (numel % 8 == 0)
    ]

    for numel in edge_sizes:
        inp = torch.randn(numel, dtype=torch.bfloat16, device=device)
        dist.barrier(group)

        out_quant = quantized_all_reduce(inp, group)
        out_ref = ref_all_reduce_bf16(inp, group)  # BF16 baseline (not FP32)

        max_err = (out_quant - out_ref).abs().max().item()
        passed = max_err < 1.0 and not torch.isnan(out_quant).any()
        assert passed, f"Edge case failed at numel={numel}, max_err={max_err}"
        if rank == 0:
            print(f"  edge numel={numel:>10d}: max_abs_err={max_err:.6f} PASS")

    if rank == 0:
        print("Edge case tests passed!")
    dist.destroy_process_group()


def run_benchmark(rank: int, world_size: int, port: int):
    device, group = setup(rank, world_size, port)

    sizes_numel = [65536, 262144, 1048576, 4194304, 16777216, 67108864]

    if rank == 0:
        print(f"{'numel':>12} | {'quant_ms':>10} | {'nccl_ms':>10} | {'speedup':>8}")
        print("-" * 52)

    for numel in sizes_numel:
        inp = torch.randn(numel, dtype=torch.bfloat16, device=device)
        dist.barrier(group)

        for _ in range(10):
            quantized_all_reduce(inp, group)
            ref_all_reduce_bf16(inp, group)

        dist.barrier(group)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(50):
            quantized_all_reduce(inp, group)
        e.record()
        e.synchronize()
        quant_ms = s.elapsed_time(e) / 50

        dist.barrier(group)
        s.record()
        for _ in range(50):
            ref_all_reduce_bf16(inp, group)
        e.record()
        e.synchronize()
        nccl_ms = s.elapsed_time(e) / 50

        if rank == 0:
            speedup = nccl_ms / quant_ms if quant_ms > 0 else float("inf")
            print(
                f"{numel:>12d} | {quant_ms:>10.3f} | {nccl_ms:>10.3f}"
                f" | {speedup:>7.2f}x"
            )

    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Tests require at least 2 CUDA devices",
)
@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda:0"))[0] < 9,
    reason="Requires SM90+ (Hopper or later)",
)
def test_quantized_allreduce_correctness():
    port = random.randint(30000, 60000)
    world_size = _supported_world_size()
    mp.spawn(run_correctness, args=(world_size, port), nprocs=world_size, join=True)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Tests require at least 2 CUDA devices",
)
@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda:0"))[0] < 9,
    reason="Requires SM90+ (Hopper or later)",
)
def test_quantized_allreduce_scale_groups():
    port = random.randint(30000, 60000)
    world_size = _supported_world_size()
    mp.spawn(
        run_scale_group_sweep, args=(world_size, port), nprocs=world_size, join=True
    )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Tests require at least 2 CUDA devices",
)
@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda:0"))[0] < 9,
    reason="Requires SM90+ (Hopper or later)",
)
def test_quantized_allreduce_edge_cases():
    """Test edge cases: small tensors, exact block boundaries."""
    port = random.randint(30000, 60000)
    world_size = _supported_world_size()
    mp.spawn(run_edge_cases, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--scale-sweep", action="store_true")
    args = parser.parse_args()

    port = random.randint(30000, 60000)
    world_size = _supported_world_size()
    spawn_kwargs = dict(args=(world_size, port), nprocs=world_size, join=True)

    if args.benchmark:
        mp.spawn(run_benchmark, **spawn_kwargs)
    elif args.scale_sweep:
        mp.spawn(run_scale_group_sweep, **spawn_kwargs)
    else:
        mp.spawn(run_correctness, **spawn_kwargs)
