import math
import multiprocessing as mp
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from flashinfer.utils import get_compute_capability
from flashinfer.comm.mixed_comm import (
    MixedCommHandler,
    MixedCommMode,
    MixedCommOp,
    run_mixed_comm,
)


_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))


def prepare_data(max_local_bs, hidden_size, world_size, dtype, device):
    x_ag = torch.empty(
        [max_local_bs, hidden_size],
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    x_rs = torch.empty(
        [max_local_bs * world_size, hidden_size],
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    data = {
        MixedCommOp.ALLREDUCE: x_ag,
        MixedCommOp.ALLGATHER: x_ag,
        MixedCommOp.REDUCESCATTER: x_rs,
        MixedCommOp.ALLREDUCE_ALLGATHER: x_ag,
        MixedCommOp.REDUCESCATTER_ALLREDUCE: x_rs,
    }
    return data


def check_op(
    op,
    mixed_comm_handler,
    data,
    local_bs_list,
    info_dict_base,
    rtol=2e-2,
    atol=2e-1,
):
    tp_size = mixed_comm_handler.para_info.tp_size
    dp_size = mixed_comm_handler.para_info.dp_size
    if MixedCommMode.NCCL_TP_DP in mixed_comm_handler.valid_mode_list:
        ref_mode = MixedCommMode.NCCL_TP_DP
    else:
        ref_mode = MixedCommMode.NCCL_ONE
    if op in [MixedCommOp.REDUCESCATTER, MixedCommOp.REDUCESCATTER_ALLREDUCE]:
        local_bs_coef = dp_size
    else:
        local_bs_coef = 1
    data_sel = data[op] / math.sqrt(mixed_comm_handler.para_info.tp_size)
    for idx_data, local_bs in enumerate(local_bs_list):
        x_in = data_sel[: local_bs * local_bs_coef]
        ref_out = run_mixed_comm(op, mixed_comm_handler, x_in, mode=ref_mode)
        for mode in mixed_comm_handler.valid_mode_list:
            if mode == ref_mode:
                continue
            info_dict = {
                **info_dict_base,
                "mode": mode,
                "idx_data": idx_data,
                "local_bs": local_bs,
            }
            x_out = run_mixed_comm(op, mixed_comm_handler, x_in, mode=mode)
            valid_precision = torch.allclose(x_out, ref_out, rtol=rtol, atol=atol)
            if not valid_precision:
                diff_abs = (x_out - ref_out).abs()
                index_sel = torch.argsort(diff_abs.flatten(), descending=True)[:10]
                diff_abs_sel = diff_abs.flatten()[index_sel]
                x_out_sel = x_out.flatten()[index_sel]
                ref_out_sel = ref_out.flatten()[index_sel]
                raise AssertionError(
                    f"{op.name} precision check failed: {info_dict}\n"
                    f"{diff_abs_sel=}\n"
                    f"{x_out_sel=}\n"
                    f"{ref_out_sel=}\n"
                    f"{x_out=}\n"
                    f"{ref_out=}"
                )
            if tp_size > 1:
                x_out_all = torch.empty(
                    [tp_size, x_out.shape[0], *x_out.shape[1:]],
                    dtype=x_out.dtype,
                    device=x_out.device,
                )
                torch.distributed.all_gather_into_tensor(
                    x_out_all,
                    x_out,
                    group=mixed_comm_handler.tp_comm_group,
                )
                valid_consistency = torch.equal(
                    x_out_all, x_out[None].expand(tp_size, -1, -1)
                )
                if not valid_consistency:
                    raise AssertionError(
                        f"{op.name} consistency check failed: {info_dict}\n"
                        f"{x_out=}\n"
                        f"{x_out_all=}"
                    )


def _run_worker(
    local_rank,
    local_size,
    inter_rank,
    inter_size,
    dtype,
    dist_init_method,
):
    max_local_bs = 4096
    num_data = 100
    hidden_size = 8192
    world_rank = inter_rank * local_size + local_rank
    world_size = inter_size * local_size
    torch.cuda.set_device(local_rank)
    torch.random.manual_seed(world_rank)
    np.random.seed(0)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=world_rank,
        world_size=world_size,
        device_id=device,
        init_method=dist_init_method,
    )
    local_size_all = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(local_size_all, local_size)
    assert all(val == local_size for val in local_size_all), (
        "local_size must be the same on all ranks"
    )
    local_tp_size_list = [
        val for val in range(1, local_size + 1) if local_size % val == 0
    ]
    inter_tp_size_list = [
        val for val in range(1, inter_size + 1) if inter_size % val == 0
    ]
    data = prepare_data(max_local_bs, hidden_size, world_size, dtype, device)
    local_bs_list = (
        np.random.permutation(max_local_bs - 2)[: num_data - 2] + 2
    ).tolist() + [1, max_local_bs]
    for local_tp_size, inter_tp_size in product(local_tp_size_list, inter_tp_size_list):
        local_dp_size = local_size // local_tp_size
        inter_dp_size = inter_size // inter_tp_size
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
        info_dict_base = {
            "local_rank": local_rank,
            "inter_rank": inter_rank,
            "local_tp_size": local_tp_size,
            "local_dp_size": local_dp_size,
            "inter_tp_size": inter_tp_size,
            "inter_dp_size": inter_dp_size,
            "dtype": dtype,
        }
        if world_rank == 0:
            valid_op_list = [val.name for val in mixed_comm_handler.valid_op_list]
            valid_mode_list = [val.name for val in mixed_comm_handler.valid_mode_list]
            print(info_dict_base, flush=True)
            print(f"{valid_op_list=}", flush=True)
            print(f"{valid_mode_list=}", flush=True)
            for op, mode in product(
                mixed_comm_handler.valid_op_list, mixed_comm_handler.valid_mode_list
            ):
                max_block_size_dict = mixed_comm_handler.max_block_size_dict.get(
                    (op, mode), None
                )
                print(f"{op.name=}, {mode.name=}, {max_block_size_dict=}", flush=True)
            print(flush=True)
        for op in mixed_comm_handler.valid_op_list:
            check_op(op, mixed_comm_handler, data, local_bs_list, info_dict_base)
        mixed_comm_handler.shutdown()
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("local_size", [2, 4, 8])
def test_mixed_comm(local_size, num_nodes, node_id, dtype, dist_init_method):
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not run_mixed_comm.is_compute_capability_supported(compute_capability_number):
        pytest.skip(
            f"run_mixed_comm not supported on current compute capability."
            f"Detected sm{compute_capability_number}."
        )
    num_local_gpus = torch.cuda.device_count()
    if local_size > num_local_gpus:
        pytest.skip(
            f"At least {local_size} local GPUs are required, but only {num_local_gpus} are available"
        )
    mp.set_start_method("spawn", force=True)
    process_list = []
    for local_rank in range(local_size):
        process = mp.Process(
            target=_run_worker,
            args=(local_rank, local_size, node_id, num_nodes, dtype, dist_init_method),
            name=f"Worker-{local_rank}",
        )
        process.start()
        process_list.append(process)
    # Poll workers and fail fast if any exits non-zero, to avoid deadlocks
    # where peers are stuck in collectives after a worker crash.
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
