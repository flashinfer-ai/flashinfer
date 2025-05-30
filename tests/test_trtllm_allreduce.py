import multiprocessing as mp
import socket
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm


# todo: temp for test
maxBatchSize = 1
maxBeamWidth = 3
maxSeqLen = 65536
hiddenSize = 64


def _run_correctness_worker(world_size, rank, dtype, distributed_init_port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",  # todo: do we need to support other backends?
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        device = torch.device(f"cuda:{rank}")
        token_nums = [
            512,
            2560,
            4096,
        ]
        strategy_codes = [0, 1, 2, 3, 4, 5, 6]
        config_codes = [0, 1]
        fusion_op_codes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # dtypes = [torch.float32, torch.float16, torch.bfloat16]
        launch_with_pdl = [True, False]

        # create ipc memory
        # todo(29 May): create ipc memory buffer instead of lists of ipc memory???
        # todo(yingyi): lamport should be init only when can_access_peer is true
        # init per world_size? 
        ipc_handles = comm.trtllm_create_ipc_buffer_for_all_reduce(rank, world_size, maxSeqLen, hiddenSize, group=group)

        test_loop = 3

        for token_num in token_nums:
            for strategy_code in strategy_codes:
                for config_code in config_codes:
                    for fusion_op_code in fusion_op_codes:
                        for launch_with_pdl in launch_with_pdl:
                            for _ in range(test_loop):
                                message_size = token_num * hiddenSize
                                inp1 = torch.rand(
                                    message_size, dtype=dtype, device=device
                                )
                                inp1_ref = inp1.clone()
                                out1 = torch.empty_like(inp1)

                                all_reduce_params = comm.trtllm_create_all_reduce_params(
                                    message_size,
                                    world_size,
                                    rank,
                                    inp1.data_ptr(),
                                    out1.data_ptr(),
                                    ipc_handles,
                                )

                                comm.trtllm_custom_all_reduce(
                                    all_reduce_params,
                                    inp1,
                                    out1,
                                    world_size,
                                    rank,
                                    message_size,
                                    fusion_op_code,
                                    strategy_code,
                                    config_code,
                                    launch_with_pdl,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                )
                                dist.all_reduce(inp1_ref, group=group)

                                torch.testing.assert_close(out1, inp1_ref)
    finally:
        dist.barrier(group=group)

        # todo(yingyi): should we have a comm manager for all tllm allreduce
        # if custom_ptr is not None:
        #     comm.dispose(custom_ptr)

        # todo(yingyi): free ipc handles
        for ipc_handle in ipc_handles:
            comm.free_shared_buffer(ipc_handle, group)

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
    world_size: int, dtype: torch.dtype, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, dtype, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_trtllm_custom_allreduce(world_size, dtype):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        raise ValueError(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")

    multi_process_parallel(
        world_size,
        dtype,
        _run_correctness_worker,
        target_args=(),
    )
    print(f"custom allreduce tp = {world_size}: OK")


if __name__ == "__main__":
    # compile tests
    mod = comm.get_comm_module()

    # add py interface binding tests
    # comm.trtllm_lamport_initialize_all(
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    # )
    # comm.trtllm_custom_all_reduce(
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     torch.empty(1024, dtype=torch.float32, device="cuda"),
    #     2,
    #     0,
    #     1024,
    #     0,
    #     0,
    #     0,
    #     True,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    #     None,
    # )

    # todo: add real tests above
