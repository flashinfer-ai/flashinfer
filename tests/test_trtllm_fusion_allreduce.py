import ctypes
import multiprocessing as mp
import random
import socket
import unittest
from typing import Any, List, Optional

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
# import flashinfer.trtllm_all_reduce as trtllm_all_reduce


def _run_correctness_worker(world_size, rank, distributed_init_port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        device = torch.device(f"cuda:{rank}")
        max_size = 8192 * 1024
        meta_ptrs = comm.create_shared_buffer(comm.meta_size() + max_size, group=group)

        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        buffer_ptrs = comm.create_shared_buffer(max_size, group=group)

        custom_ptr = comm.init_custom_ar(meta_ptrs, rank_data, rank, True)
        comm.register_buffer(custom_ptr, buffer_ptrs)

        test_sizes = [
            512,
            2560,
            4096,
            5120,
            7680,
            32768,
            262144,
            524288,
            1048576,
            2097152,
        ]
        num_ctas = [1, 2, 4, 8, 16, 32, 36]
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        test_loop = 10

        for test_size in test_sizes:
            for num_cta in num_ctas:
                for dtype in dtypes:
                    for _ in range(test_loop):
                        inp1 = torch.randint(
                            1, 16, (test_size,), dtype=dtype, device=device
                        )
                        inp1_ref = inp1.clone()
                        out1 = torch.empty_like(inp1)

                        comm.all_reduce(
                            custom_ptr,
                            inp1,
                            out1,
                            buffer_ptrs[rank],
                            max_size,
                            num_cta,
                        )

                        dist.all_reduce(inp1_ref, group=group)

                        torch.testing.assert_close(out1, inp1_ref)
    finally:
        dist.barrier(group=group)
        if custom_ptr is not None:
            comm.dispose(custom_ptr)
        if buffer_ptrs:
            comm.free_shared_buffer(buffer_ptrs, group)
        if meta_ptrs:
            comm.free_shared_buffer(meta_ptrs, group)

        dist.destroy_process_group(group=group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_trtllm_fusion_allreduce(world_size: int):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")


if __name__ == "__main__":
    # Initialize process group for compilation test
    distributed_init_method = f"tcp://localhost:{get_open_port()}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=0,
        world_size=1,
    )
    group = list(range(dist.get_world_size()))
    
    # Placeholder tests for JIT compilation
    device = torch.device("cuda:0")
    hidden_dim = 1024
    seq_len = 32
    num_experts = 8

    # Test allreduce compilation
    input_tensor = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=device)
    residual = torch.randn_like(input_tensor)
    norm_weight = torch.randn(hidden_dim, dtype=torch.float16, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)  # 1MB workspace

    # Test basic allreduce
    outputs = comm.allreduce(input_tensor)
    print("Basic allreduce compiled successfully")

    # Test fused residual + RMS norm allreduce
    outputs = comm.allreduce(
        input_tensor,
        residual=residual,
        norm_weight=norm_weight,
        workspace=workspace,
        op=1,  # RESIDUAL_RMS_NORM
    )
    print("Fused residual + RMS norm allreduce compiled successfully")

    # Test MoE allreduce compilation
    device_num_experts = torch.tensor([num_experts], dtype=torch.int32, device=device)
    scale_input = torch.randn(num_experts, seq_len, dtype=torch.float32, device=device)
    active_experts_token_input = torch.randn(num_experts, seq_len, hidden_dim, dtype=torch.float16, device=device)
    token_input = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=device)

    outputs = comm.moe_allreduce(
        residual=residual,
        norm_weight=norm_weight,
        device_num_experts=device_num_experts,
        scale_input=scale_input,
        active_experts_token_input=active_experts_token_input,
        token_input=token_input,
        workspace=workspace,
    )
    print("MoE allreduce compiled successfully")

