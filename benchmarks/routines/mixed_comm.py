"""
Mixed Communication Benchmark Routine

This module provides benchmarking for mixed communication operations
(allreduce+allgather, reducescatter+allreduce) using FlashInfer's
MixedCommHandler. Uses multiprocessing + torch.distributed for
multi-GPU orchestration.

Supports operations: allreduce, allgather, reducescatter,
    allreduce_allgather, reducescatter_allreduce
Supports modes: all valid fused and NCCL modes, plus AUTOTUNE

Launch examples:
    # Basic benchmark on 8 GPUs (4 TP x 2 DP)
    python benchmarks/flashinfer_benchmark.py \\
        --routine mixed_comm \\
        --local_bs_list 1 4 16 \\
        --hidden_size 4096 \\
        --dtype bfloat16 \\
        --local_tp_size 4 \\
        --local_dp_size 2

    # Multi-node benchmark
    python benchmarks/flashinfer_benchmark.py \\
        --routine mixed_comm \\
        --local_bs_list 1 4 16 \\
        --hidden_size 4096 \\
        --dtype bfloat16 \\
        --local_tp_size 4 \\
        --local_dp_size 2 \\
        --inter_tp_size 2 \\
        --inter_dp_size 1 \\
        --node_id 0 \\
        --dist_init_method tcp://192.168.1.1:29501
"""

import multiprocessing as mp
import statistics
from collections import defaultdict

import torch

from flashinfer.comm.mixed_comm import (
    MixedCommHandler,
    MixedCommOp,
    run_mixed_comm,
    _ceil_div,
)
from flashinfer.testing.utils import bench_gpu_time

try:
    from .flashinfer_benchmark_utils import dtype_str_to_torch_dtype
except ImportError:
    from flashinfer_benchmark_utils import dtype_str_to_torch_dtype


@torch.inference_mode()
def _run_worker(local_rank, local_size, args, result_queue):
    """Worker process: set up distributed, run benchmarks, send results back via queue."""
    inter_tp_size = args.inter_tp_size
    inter_dp_size = args.inter_dp_size
    inter_rank = args.node_id
    inter_size = inter_tp_size * inter_dp_size
    world_rank = inter_rank * local_size + local_rank
    world_size = inter_size * local_size
    dtype = dtype_str_to_torch_dtype(args.dtype)

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

    handler = MixedCommHandler(
        world_rank=world_rank,
        world_size=world_size,
        local_rank=local_rank,
        local_size=local_size,
        inter_rank=inter_rank,
        inter_size=inter_size,
        local_tp_size=args.local_tp_size,
        local_dp_size=args.local_dp_size,
        inter_tp_size=inter_tp_size,
        inter_dp_size=inter_dp_size,
        dtype=dtype,
        device=device,
    )

    max_local_bs = max(args.local_bs_list)
    data = torch.empty(
        [max_local_bs * handler.para_info.dp_size, args.hidden_size],
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)

    def _print_rank0(*print_args, **print_kwargs):
        """Print only from world rank 0."""
        if world_rank == 0:
            print_kwargs.setdefault("flush", True)
            print(*print_args, **print_kwargs)

    for (op, mode), max_block_size_dict in handler.max_block_size_dict.items():
        _print_rank0(f"{op.name=}, {mode.name=}, {max_block_size_dict=}")
    _print_rank0()

    results = []

    for local_bs in args.local_bs_list:
        for op in handler.valid_op_list:
            if op in [MixedCommOp.REDUCESCATTER, MixedCommOp.REDUCESCATTER_ALLREDUCE]:
                x_in = data[: local_bs * handler.para_info.dp_size]
            else:
                x_in = data[:local_bs]

            _print_rank0(f"{op.name}: {local_bs=}")

            for mode in handler.valid_mode_list:
                # Note: shared args --num_iters and --dry_run_iters are not used
                # here. Time-based warmup/measurement is used instead.
                duration_list = bench_gpu_time(
                    run_mixed_comm,
                    input_args=(op, handler, x_in),
                    input_kwargs={"mode": mode},
                    dry_run_time_ms=10,
                    repeat_time_ms=100,
                    use_cuda_graph=True,
                    num_iters_within_graph=10,
                )

                if world_rank == 0:
                    median_time_ms = statistics.median(duration_list)
                    median_us = median_time_ms * 1000
                    tp_size = handler.para_info.tp_size
                    world_size = handler.para_info.world_size
                    if op == MixedCommOp.ALLREDUCE:
                        data_bytes_base = _ceil_div(
                            x_in.numel() * x_in.element_size(), world_size
                        )
                        message_bytes = 2 * (world_size - 1) * data_bytes_base
                    elif op == MixedCommOp.ALLGATHER:
                        data_bytes_base = x_in.numel() * x_in.element_size()
                        message_bytes = (world_size - 1) * data_bytes_base
                    elif op == MixedCommOp.REDUCESCATTER:
                        data_bytes_base = _ceil_div(
                            x_in.numel() * x_in.element_size(), world_size
                        )
                        message_bytes = (world_size - 1) * data_bytes_base
                    elif op == MixedCommOp.ALLREDUCE_ALLGATHER:
                        data_bytes_base = _ceil_div(
                            x_in.numel() * x_in.element_size(), tp_size
                        )
                        message_bytes = (tp_size + world_size - 2) * data_bytes_base
                    elif op == MixedCommOp.REDUCESCATTER_ALLREDUCE:
                        data_bytes_base = _ceil_div(
                            x_in.numel() * x_in.element_size(), world_size
                        )
                        message_bytes = (world_size + tp_size - 2) * data_bytes_base
                    else:
                        raise ValueError(f"Unsupported op: {op.name}")
                    gb_per_sec = (
                        message_bytes / (median_time_ms * 1e-3) / 1e9
                        if median_time_ms > 0
                        else 0.0
                    )

                    label = f"{op.name}_{mode.name}"
                    _print_rank0(
                        f"  {mode.name}: {median_us:.3f} us, {gb_per_sec:.3f} GB/sec"
                    )

                    cur_res = defaultdict(str)
                    cur_res["routine"] = args.routine
                    cur_res["median_time"] = median_time_ms
                    cur_res["tflops"] = "N/A"
                    cur_res["tb_per_sec"] = gb_per_sec / 1000
                    cur_res["backend"] = mode.name
                    cur_res["resolved_backend"] = label
                    cur_res["hidden_size"] = args.hidden_size
                    cur_res["input_dtype"] = args.dtype
                    cur_res["local_bs"] = local_bs
                    cur_res["op_name"] = op.name
                    cur_res["mode_name"] = mode.name
                    cur_res["local_tp_size"] = handler.para_info.local_tp_size
                    cur_res["local_dp_size"] = handler.para_info.local_dp_size
                    cur_res["inter_tp_size"] = handler.para_info.inter_tp_size
                    cur_res["inter_dp_size"] = handler.para_info.inter_dp_size
                    results.append(cur_res)

            _print_rank0()

    handler.shutdown()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

    if world_rank == 0:
        result_queue.put(results)


def run_mixed_comm_test(args):
    """Entry point from flashinfer_benchmark.py."""
    if args.routine == "mixed_comm":
        return test_mixed_comm(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_mixed_comm_args(line, parser):
    """Parse mixed-comm-specific command line arguments."""
    parser.add_argument("--local_bs_list", type=int, nargs="+", required=True)
    parser.add_argument("--hidden_size", type=int, required=True)
    parser.add_argument(
        "--dtype", type=str, required=True, choices=["float16", "bfloat16"]
    )
    parser.add_argument("--local_tp_size", type=int, required=True)
    parser.add_argument("--local_dp_size", type=int, required=True)
    parser.add_argument("--inter_tp_size", type=int, default=1)
    parser.add_argument("--inter_dp_size", type=int, default=1)
    parser.add_argument("--node_id", type=int, default=0)
    parser.add_argument("--dist_init_method", type=str, default="tcp://localhost:29501")
    args = parser.parse_args(line)
    return args


def test_mixed_comm(args):
    """Benchmark mixed communication operations via multiprocessing workers.

    Spawns one process per local GPU, each running torch.distributed.
    Rank 0 collects and returns benchmark results via a queue.
    """
    num_local_gpus = torch.cuda.device_count()
    local_size = args.local_tp_size * args.local_dp_size
    assert local_size > 1, "local_size must be greater than 1"
    assert local_size <= num_local_gpus, (
        f"At least {local_size} local GPUs are required, "
        f"but only {num_local_gpus} are available"
    )

    if args.node_id == 0:
        print(args, flush=True)

    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()

    process_list = []
    for local_rank in range(local_size):
        process = mp.Process(
            target=_run_worker,
            args=(local_rank, local_size, args, result_queue),
            name=f"Worker-{local_rank}",
        )
        process.start()
        process_list.append(process)

    # Poll workers so we can fail fast if any worker exits non-zero.
    # Without this, a crashed worker leaves peers stuck in collectives
    # and the parent blocks forever on join().
    failed = None
    while any(p.is_alive() for p in process_list):
        for idx, p in enumerate(process_list):
            p.join(timeout=10.0)
            if p.exitcode is not None and p.exitcode != 0 and failed is None:
                failed = idx
                break
        if failed is not None:
            for p in process_list:
                if p.is_alive():
                    p.terminate()
            for p in process_list:
                p.join(timeout=5.0)
            break

    if failed is not None:
        raise RuntimeError(
            f"Worker {failed} failed with exit code {process_list[failed].exitcode}"
        )

    if not result_queue.empty():
        return result_queue.get()
    return []
