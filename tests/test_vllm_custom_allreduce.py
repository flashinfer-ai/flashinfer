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
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(
        world_size,
        _run_correctness_worker,
        target_args=(),
    )
    print(f"custom allreduce tp = {world_size}: OK")
