import argparse
import multiprocessing as mp
import statistics

import torch

from flashinfer.comm.mixed_comm import MixedCommHandler, MixedCommOp, run_mixed_comm
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
    duration_us = statistics.mean(duration_list) * 1000
    print_first_rank(f"{info}: {duration_us:.3f} us")


def bench_op(op, mixed_comm_handler, data, local_bs):
    print_first_rank(f"{op.name}: {local_bs=}")
    if op in [MixedCommOp.REDUCESCATTER, MixedCommOp.REDUCESCATTER_ALLREDUCE]:
        x_in = data[: local_bs * mixed_comm_handler.para_info.dp_size]
    else:
        x_in = data[:local_bs]
    for mode in mixed_comm_handler.valid_mode_list:
        print_duration(
            f"{mode.name}",
            run_mixed_comm,
            (op, mixed_comm_handler, x_in),
            input_kwargs={"mode": mode},
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
    inter_rank = args.node_id
    inter_size = inter_tp_size * inter_dp_size
    world_rank = inter_rank * local_size + local_rank
    world_size = inter_size * local_size
    torch.cuda.set_device(local_rank)
    torch.random.manual_seed(world_rank)
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
    assert all(val == local_size for val in local_size_all), (
        "local_size must be the same on all ranks"
    )
    max_local_bs = max(local_bs_list)
    mixed_comm_handler = MixedCommHandler(
        world_rank=world_rank,
        world_size=world_size,
        local_rank=local_rank,
        local_size=local_size,
        inter_rank=inter_rank,
        inter_size=inter_size,
        local_tp_size=local_tp_size,
        local_dp_size=local_dp_size,
        inter_tp_size=inter_tp_size,
        inter_dp_size=inter_dp_size,
        dtype=dtype,
        device=device,
    )
    data = torch.empty(
        [max_local_bs * mixed_comm_handler.para_info.dp_size, hidden_size],
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    for (
        op,
        mode,
    ), max_block_size_dict in mixed_comm_handler.max_block_size_dict.items():
        print_first_rank(f"{op.name=}, {mode.name=}, {max_block_size_dict=}")
    print_first_rank()
    for local_bs in local_bs_list:
        for op in mixed_comm_handler.valid_op_list:
            bench_op(op, mixed_comm_handler, data, local_bs)
    mixed_comm_handler.shutdown()
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
