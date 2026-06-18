# flashinfer: adapted from sglang + vllm
# refer to sgl-kernel/tests/test_custom_allreduce.py from sglang

import logging
import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm

# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/cuda_wrapper.py


logger = logging.getLogger(__name__)


def _initialize_process_group(world_size, rank, distributed_init_port):
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
    return group


def _run_correctness_worker(world_size, rank, distributed_init_port):
    try:
        group = _initialize_process_group(world_size, rank, distributed_init_port)
        device = torch.device(f"cuda:{rank}")
        max_size = 8192 * 1024
        meta_ptrs = comm.create_shared_buffer(
            comm.vllm_meta_size() + max_size, group=group
        )

        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        buffer_ptrs = comm.create_shared_buffer(max_size, group=group)

        custom_ptr = comm.vllm_init_custom_ar(meta_ptrs, rank_data, rank, True)
        comm.vllm_register_buffer(custom_ptr, buffer_ptrs)

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

                        comm.vllm_all_reduce(
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
            comm.vllm_dispose(custom_ptr)
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


def _run_graph_buffer_ipc_meta_worker(
    world_size: int, rank: int, distributed_init_port: int
):
    """Test get_graph_buffer_ipc_meta function with CUDA graph capture."""

    custom_ptr = None
    meta_ptrs = None

    try:
        # Setup
        group = _initialize_process_group(world_size, rank, distributed_init_port)
        device = torch.device(f"cuda:{rank}")
        max_size = 8192 * 1024
        meta_ptrs = comm.create_shared_buffer(
            comm.vllm_meta_size() + max_size, group=group
        )
        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        custom_ptr = comm.vllm_init_custom_ar(meta_ptrs, rank_data, rank, True)

        # Test 1: Empty state before graph capture
        handle_bytes, offsets = comm.vllm_get_graph_buffer_ipc_meta(custom_ptr)
        assert len(handle_bytes) == 0 and len(offsets) == 0, (
            "Expected empty buffers before graph capture"
        )

        # Test 2: Capture graph and validate IPC metadata structure
        test_size = 4096
        num_cta = 16
        dtype = torch.float16

        inp1 = torch.randn(test_size, dtype=dtype, device=device)
        inp2 = torch.randn(test_size, dtype=dtype, device=device)
        out1 = torch.empty_like(inp1)
        out2 = torch.empty_like(inp2)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=None):
            comm.vllm_all_reduce(custom_ptr, inp1, out1, 0, 0, num_cta)
            comm.vllm_all_reduce(custom_ptr, inp2, out2, 0, 0, num_cta)

        handle_bytes, offsets = comm.vllm_get_graph_buffer_ipc_meta(custom_ptr)

        # Validate structure: 2 buffers, correct handle size (64 bytes each)
        ipc_handle_size = 64
        expected_num_buffers = 2
        assert len(offsets) == expected_num_buffers, (
            f"Expected {expected_num_buffers} offsets, got {len(offsets)}"
        )
        assert len(handle_bytes) == ipc_handle_size * expected_num_buffers, (
            f"Expected {ipc_handle_size * expected_num_buffers} handle bytes"
        )
        assert all(isinstance(o, int) and o >= 0 for o in offsets), (
            "All offsets should be non-negative integers"
        )

        # Test 3: Distributed gather and register graph buffers
        all_handle_bytes = [None] * world_size
        all_offsets = [None] * world_size

        dist.all_gather_object(all_handle_bytes, handle_bytes, group=group)
        dist.all_gather_object(all_offsets, offsets, group=group)

        # All ranks should have same number of buffers
        assert all(len(off) == expected_num_buffers for off in all_offsets), (
            "All ranks should have same number of buffers"
        )

        comm.vllm_register_graph_buffers(custom_ptr, all_handle_bytes, all_offsets)

        # Test 4: Graph replay produces correct results
        inp1_test = torch.randn(test_size, dtype=dtype, device=device)
        inp2_test = torch.randn(test_size, dtype=dtype, device=device)

        inp1.copy_(inp1_test)
        inp2.copy_(inp2_test)

        g.replay()
        torch.cuda.synchronize()

        # Verify with NCCL reference
        inp1_ref = inp1_test.clone()
        inp2_ref = inp2_test.clone()
        dist.all_reduce(inp1_ref, group=group)
        dist.all_reduce(inp2_ref, group=group)

        torch.testing.assert_close(out1, inp1_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out2, inp2_ref, rtol=1e-3, atol=1e-3)

    finally:
        dist.barrier(group=group)
        if custom_ptr is not None:
            comm.vllm_dispose(custom_ptr)
        if meta_ptrs:
            comm.free_shared_buffer(meta_ptrs, group)
        dist.destroy_process_group(group=group)


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
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


@pytest.mark.parametrize("world_size", [2, 4])
def test_vllm_custom_allreduce(world_size):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(
        world_size,
        _run_correctness_worker,
        target_args=(),
    )
    print(f"custom allreduce tp = {world_size}: OK")


@pytest.mark.parametrize("world_size", [2, 4])
def test_get_graph_buffer_ipc_meta(world_size: int):
    """Test get_graph_buffer_ipc_meta function with CUDA graph capture."""
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running get_graph_buffer_ipc_meta test for world_size={world_size}")
    multi_process_parallel(
        world_size,
        _run_graph_buffer_ipc_meta_worker,
        target_args=(),
    )
    print(f"get_graph_buffer_ipc_meta test for world_size={world_size}: OK")
