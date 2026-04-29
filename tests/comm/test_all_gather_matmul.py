# Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Test script for all-gather and matrix multiplication kernels.

Run with pytest:
    pytest tests/comm/test_all_gather_matmul.py -vv -s

Run standalone:
    python test_all_gather_matmul.py --correctness
    python test_all_gather_matmul.py --benchmark
    python test_all_gather_matmul.py --profile

Other options:
    --dtype: Data type for input and weight tensors (default: bfloat16)
"""

import argparse
import pytest
import random
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp

from flashinfer.comm import all_gather_matmul
from flashinfer.utils import get_compute_capability

HID = 8192
OUT_HID = 2048


# Reference gather then matmul implementation
def ref_gather_matmul(
    inp: torch.Tensor,
    w: torch.Tensor,
    group: dist.ProcessGroup,
):
    world_size = dist.get_world_size(group)
    ag_scratch = torch.empty(
        (world_size * inp.shape[0], inp.shape[1]), device=inp.device, dtype=inp.dtype
    )
    dist.all_gather_into_tensor(ag_scratch, inp, group=group)
    out = ag_scratch @ w
    return out


def setup(rank: int, world_size: int, port: int):
    """Initialize distributed process group and return common state."""
    print(f"Rank {rank} of {world_size} is initializing")
    symm_mem.set_backend("NVSHMEM")
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
    torch.manual_seed(rank + 52)
    return device, group


def unit_test(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device, group = setup(rank, world_size, port)
    w = torch.randn((HID, OUT_HID), device=device, dtype=dtype)
    inp = symm_mem.empty(16 * 1024, HID, device=device, dtype=dtype).normal_()
    out_push = all_gather_matmul(inp, w, group, verbose=True)
    expected_out = ref_gather_matmul(inp, w, group)
    torch.testing.assert_close(out_push, expected_out, atol=1e-1, rtol=1e-2)
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Tests require at least 2 CUDA devices",
)
@pytest.mark.skipif(
    get_compute_capability(torch.device("cuda:0"))[0] < 9,
    reason="Tests runs only on SM90+ devices",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_all_gather_matmul(dtype: torch.dtype):
    import os
    import sys

    # mp.spawn starts fresh interpreters that need to re-import this module;
    # ensure the repo root is on sys.path so 'tests.comm' is findable.
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    port = random.randint(30000, 60000)
    world_size = torch.cuda.device_count()
    mp.spawn(unit_test, args=(world_size, port, dtype), nprocs=world_size, join=True)


def run_profile(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device, group = setup(rank, world_size, port)
    w = torch.randn((HID, OUT_HID), device=device, dtype=dtype)
    inp = symm_mem.empty(16 * 1024, HID, device=device, dtype=dtype).normal_()
    # Warmup
    all_gather_matmul(inp, w, group, verbose=True)
    ref_gather_matmul(inp, w, group)
    # Synchronize timer
    dist.barrier(group)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(3):
            all_gather_matmul(inp, w, group)
            ref_gather_matmul(inp, w, group)
            prof.step()
    torch.cuda.synchronize()
    gpu_arch = torch.cuda.get_device_properties(device).name.replace(" ", "_")
    prof.export_chrome_trace(f"{gpu_arch}_rank{rank}.json")
    dist.destroy_process_group()


def run_correctness(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device, group = setup(rank, world_size, port)
    w = torch.randn((HID, OUT_HID), device=device, dtype=dtype)
    bs_list = [2**15, 2**14, 2**13, 2**12]
    # Non-power-of-two sizes
    bs_list += [19 * 1024]
    for bs in bs_list:
        inp = symm_mem.empty(bs, HID, device=device, dtype=dtype)
        for _ in range(10):
            inp.normal_()
            dist.barrier(group)
            out_push = all_gather_matmul(inp, w, group)
            expected_out = ref_gather_matmul(inp, w, group)
            torch.testing.assert_close(out_push, expected_out, atol=1e-1, rtol=1e-2)
        print(f"Rank {rank} of {world_size}: Correctness check passed (bs={bs})")
    dist.destroy_process_group()


def run_benchmark(rank: int, world_size: int, port: int, dtype: torch.dtype):
    device, group = setup(rank, world_size, port)
    w = torch.randn((HID, OUT_HID), device=device, dtype=dtype)

    # Warmup
    inp = symm_mem.empty(16 * 1024, HID, device=device, dtype=dtype).normal_()
    all_gather_matmul(inp, w, group)

    # Benchmark timing (CUDA events) for both implementations.
    # Notes:
    # - The timed region is "end-to-end" host call -> kernel(s) complete.
    # - For the wait+signal path, the function already synchronizes main/comm streams.
    bench_warmup = 10
    bench_iters = 50

    def _bench_one_mean_ms(fn, *, expected_out=None, check_name: str = ""):
        # Ensure all ranks start each benchmark section together.
        dist.barrier(group)
        torch.cuda.synchronize(device)
        # Warmup (separate from compile warmup above; helps stabilize caches).
        for _ in range(bench_warmup):
            out = fn()
            if expected_out is not None:
                torch.testing.assert_close(
                    out,
                    expected_out,
                    atol=1e-1,
                    rtol=1e-2,
                    msg=(
                        f"Benchmark correctness check failed "
                        f"(rank={rank}, impl={check_name})"
                    ),
                )
        dist.barrier(group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_ms = 0.0
        for _ in range(bench_iters):
            start.record()
            fn()
            end.record()
            end.synchronize()
            total_ms += float(start.elapsed_time(end))
        mean_ms = total_ms / bench_iters
        local = torch.tensor([mean_ms], device=device, dtype=torch.float32)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local, group=group)
        if rank != 0:
            return None
        means = torch.stack(gathered, dim=0).cpu().flatten()  # (world,)
        avg_mean = means.mean().item()
        max_mean = means.max().item()
        return avg_mean, max_mean

    bs_list = [2**16, 2**15, 2**14, 2**13, 2**12, 2**11, 2**10]
    if rank == 0:
        print(f"[benchmark] iters={bench_iters}, warmup={bench_warmup}")
        bs_w = 8
        col_w = 13
        inner_gap = " "
        group_w = col_w * 2 + len(inner_gap)
        speedup_w = 9
        vbar = " | "
        header_top = (
            f"{'':>{bs_w}}{vbar}"
            f"{'push':^{group_w}}{vbar}"
            f"{'ref':^{group_w}}{vbar}"
            f"{'speedup':^{speedup_w}}"
        )
        header_bottom = (
            f"{'# tokens':>{bs_w}}{vbar}"
            f"{'avg_mean':>{col_w}}{inner_gap}{'max_mean':>{col_w}}{vbar}"
            f"{'avg_mean':>{col_w}}{inner_gap}{'max_mean':>{col_w}}{vbar}"
            f"{'avg':>{speedup_w}}"
        )
        sep = "-" * len(header_bottom)
        print(sep)
        print(header_top)
        print(header_bottom)
        print(sep)

    for bs in bs_list:
        # Fresh symmetric allocations per bs (avoid relying on slicing semantics).
        inp = symm_mem.empty(bs, HID, device=device, dtype=dtype).normal_()
        # Make sure all ranks have initialized their inputs for this bs.
        dist.barrier(group)
        push = _bench_one_mean_ms(
            lambda: all_gather_matmul(inp, w, group),
            check_name="overlapped",
        )
        ref = _bench_one_mean_ms(
            lambda: ref_gather_matmul(inp, w, group),
            check_name="ref",
        )
        if rank == 0:
            push_avg, push_max = push
            ref_avg, ref_max = ref
            speedup = ref_avg / push_avg
            print(
                f"{bs:8d}{vbar}"
                f"{push_avg:13.3f} {push_max:13.3f}{vbar}"
                f"{ref_avg:13.3f} {ref_max:13.3f}{vbar}"
                f"{speedup:>{speedup_w}.2f}x"
            )

    if rank == 0:
        print(sep)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Check correctness across a range of batch sizes",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run with torch profiler and export Chrome trace",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark GPU time of push+signal vs reference implementations",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for input and weight tensors (default: bfloat16)",
    )
    args = parser.parse_args()

    # IP port number for multi-process rendezvous
    port = random.randint(30000, 60000)
    world_size = torch.cuda.device_count()
    dtype = getattr(torch, args.dtype)
    spawn_kwargs = dict(args=(world_size, port, dtype), nprocs=world_size, join=True)

    if args.profile:
        mp.spawn(run_profile, **spawn_kwargs)
    elif args.correctness:
        mp.spawn(run_correctness, **spawn_kwargs)
    elif args.benchmark:
        mp.spawn(run_benchmark, **spawn_kwargs)
    else:
        mp.spawn(unit_test, **spawn_kwargs)
