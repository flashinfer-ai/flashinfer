import argparse
import multiprocessing as mp

import numpy as np
import torch

from flashinfer.comm.mixed_comm import MixedComm
from flashinfer.testing.utils import bench_gpu_time


dtype_map = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def print_first_rank(info=None, flush=True):
    if torch.distributed.get_rank() == 0:
        if info is not None:
            print(info, flush=flush)
        else:
            print(flush=flush)


def print_duration(info, func, input_args, input_kwargs=None):
    duration_list = bench_gpu_time(
        func,
        input_args=input_args,
        input_kwargs=input_kwargs,
        dry_run_time_ms=10,
        repeat_time_ms=100,
        use_cuda_graph=True,
        num_iters_within_graph=10,
    )
    duration_us = (
        torch.tensor(duration_list).median() / torch.distributed.get_world_size() * 1000
    )
    torch.distributed.reduce(duration_us, dst=0)
    print_first_rank(f"{info}: {duration_us.item():.3f} us")


def bench_allreduce_allgather(mixed_comm, data, local_bs):
    print_first_rank(f"bench_allreduce_allgather: {local_bs=}")
    x_in = data[:local_bs]
    for mode in mixed_comm.valid_mode_list:
        print_duration(
            f"{mode.name}",
            mixed_comm.allreduce_allgather,
            (x_in, mode),
        )
    print_first_rank()


def bench_reducescatter_allreduce(mixed_comm, data, local_bs):
    print_first_rank(f"bench_reducescatter_allreduce: {local_bs=}")
    para_info = mixed_comm.para_info
    x_in = data[: local_bs * para_info.dp_size]
    for mode in mixed_comm.valid_mode_list:
        print_duration(
            f"{mode.name}",
            mixed_comm.reducescatter_allreduce,
            (x_in, mode),
        )
    print_first_rank()


@torch.inference_mode()
def _run_worker(local_rank, local_size, args):
    local_bs_list = args.local_bs_list
    hidden_size = args.hidden_size
    dtype = dtype_map[args.dtype]
    local_tp_size = args.local_tp_size
    local_dp_size = args.local_dp_size
    inter_tp_size = args.inter_tp_size
    inter_dp_size = args.inter_dp_size
    node_id = args.node_id
    num_nodes = inter_tp_size * inter_dp_size
    world_rank = node_id * local_size + local_rank
    world_size = num_nodes * local_size
    torch.cuda.set_device(local_rank)
    torch.random.manual_seed(world_rank)
    np.random.seed(0)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=world_rank,
        world_size=world_size,
        device_id=device,
        init_method=args.dist_init_method,
    )
    local_size_all = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(local_size_all, local_size)
    assert all([val == local_size for val in local_size_all]), (
        "local_size must be the same on all ranks"
    )
    max_local_bs = max(local_bs_list)
    mixed_comm = MixedComm(
        world_rank=world_rank,
        world_size=world_size,
        local_rank=local_rank,
        local_size=local_size,
        node_id=node_id,
        num_nodes=num_nodes,
        local_tp_size=local_tp_size,
        local_dp_size=local_dp_size,
        inter_tp_size=inter_tp_size,
        inter_dp_size=inter_dp_size,
        max_local_bs=max_local_bs,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
        maybe_use_trtllm_comm=True,
    )
    data = torch.empty(
        [max_local_bs * world_size, hidden_size],
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    for local_bs in local_bs_list:
        bench_allreduce_allgather(mixed_comm, data, local_bs)
        bench_reducescatter_allreduce(mixed_comm, data, local_bs)
    del mixed_comm
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_bs_list", type=int, nargs="+", required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    parser.add_argument("--local_tp_size", type=int, required=True)
    parser.add_argument("--local_dp_size", type=int, required=True)
    parser.add_argument("--inter_tp_size", type=int, default=1)
    parser.add_argument("--inter_dp_size", type=int, default=1)
    parser.add_argument("--node_id", type=int, default=0)
    parser.add_argument("--dist_init_method", type=str, default="tcp://localhost:29501")
    args = parser.parse_args()
    num_local_gpus = torch.cuda.device_count()
    local_size = args.local_tp_size * args.local_dp_size
    assert local_size > 1, "local_size must be greater than 1"
    assert local_size <= num_local_gpus, (
        f"At least {local_size} local GPUs are required, but only {num_local_gpus} are available"
    )
    if args.node_id == 0:
        print(args, flush=True)
    mp.set_start_method("spawn", force=True)
    process_list = []
    for local_rank in range(local_size):
        process = mp.Process(
            target=_run_worker,
            args=(local_rank, local_size, args),
            name=f"Worker-{local_rank}",
        )
        process.start()
        process_list.append(process)
    for idx, process in enumerate(process_list):
        process.join()
        assert process.exitcode == 0, (
            f"Process {idx} failed with exit code {process.exitcode}"
        )


if __name__ == "__main__":
    main()
