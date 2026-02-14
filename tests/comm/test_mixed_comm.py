import multiprocessing as mp
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch

from flashinfer.comm.mixed_comm import MixedComm, MixedCommMode


_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.append(str(_project_root))


def prepare_data(max_local_bs, hidden_size, world_size, dtype, device, num_data):
    x_ar_ag = torch.empty(
        [num_data, max_local_bs, hidden_size],
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    x_rs_ar = torch.empty(
        [num_data, max_local_bs * world_size, hidden_size],
        dtype=dtype,
        device=device,
    ).uniform_(-0.5, 0.5)
    data = {"ar_ag": x_ar_ag, "rs_ar": x_rs_ar}
    return data


def check_allreduce_allgather(
    mixed_comm,
    data,
    local_bs_list,
    info_dict_base,
    ref_mode=MixedCommMode.NCCL_TP_DP,
    rtol=2e-2,
    atol=2e-1,
):
    tp_size = mixed_comm.para_info.tp_size
    ref_out_list = [
        mixed_comm.allreduce_allgather(data[idx_data, :local_bs], ref_mode)
        for idx_data, local_bs in enumerate(local_bs_list)
    ]
    for mode in mixed_comm.valid_mode_list:
        if mode == ref_mode:
            continue
        block_size_list = mixed_comm.valid_block_size_dict["ar_ag"][mode]
        for block_size, (idx_data, local_bs) in product(
            block_size_list, enumerate(local_bs_list)
        ):
            info_dict = {
                **info_dict_base,
                "mode": mode,
                "block_size": block_size,
                "idx_data": idx_data,
                "local_bs": local_bs,
            }
            x_in = data[idx_data, :local_bs]
            x_out = mixed_comm.allreduce_allgather(x_in, mode, block_size)
            ref_out = ref_out_list[idx_data]
            assert torch.allclose(x_out, ref_out, rtol=rtol, atol=atol), (
                f"AR & AG precision check failed: {info_dict}"
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
                    group=mixed_comm.tp_comm_group,
                )
                assert torch.equal(x_out_all, x_out[None].expand(tp_size, -1, -1)), (
                    f"AR & AG consistency check failed: {info_dict}"
                )


def check_reducescatter_allreduce(
    mixed_comm,
    data,
    local_bs_list,
    info_dict_base,
    ref_mode=MixedCommMode.NCCL_TP_DP,
    rtol=2e-2,
    atol=2e-1,
):
    tp_size = mixed_comm.para_info.tp_size
    dp_size = mixed_comm.para_info.dp_size
    ref_out_list = [
        mixed_comm.reducescatter_allreduce(
            data[idx_data, : local_bs * dp_size], ref_mode
        )
        for idx_data, local_bs in enumerate(local_bs_list)
    ]
    for mode in mixed_comm.valid_mode_list:
        if mode == ref_mode:
            continue
        block_size_list = mixed_comm.valid_block_size_dict["rs_ar"][mode]
        for block_size, (idx_data, local_bs) in product(
            block_size_list, enumerate(local_bs_list)
        ):
            info_dict = {
                **info_dict_base,
                "mode": mode,
                "block_size": block_size,
                "idx_data": idx_data,
                "local_bs": local_bs,
            }
            x_in = data[idx_data, : local_bs * dp_size]
            x_out = mixed_comm.reducescatter_allreduce(x_in, mode, block_size)
            ref_out = ref_out_list[idx_data]
            assert torch.allclose(x_out, ref_out, rtol=rtol, atol=atol), (
                f"RS & AR precision check failed: {info_dict}"
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
                    group=mixed_comm.tp_comm_group,
                )
                assert torch.equal(x_out_all, x_out[None].expand(tp_size, -1, -1)), (
                    f"RS & AR consistency check failed: {info_dict}"
                )


def _run_worker(local_rank, local_size, node_id, num_nodes, dtype, dist_init_method):
    max_local_bs = 256
    num_data = 100
    hidden_size = 4096
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
        init_method=dist_init_method,
    )
    local_size_all = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(local_size_all, local_size)
    assert all([val == local_size for val in local_size_all]), (
        "local_size must be the same on all ranks"
    )
    local_tp_size_list = [
        val for val in range(1, local_size + 1) if local_size % val == 0
    ]
    inter_tp_size_list = [
        val for val in range(1, num_nodes + 1) if num_nodes % val == 0
    ]
    data = prepare_data(max_local_bs, hidden_size, world_size, dtype, device, num_data)
    local_bs_list = (
        np.random.permutation(max_local_bs - 2)[: num_data - 2] + 2
    ).tolist() + [1, max_local_bs]
    for local_tp_size, inter_tp_size in product(local_tp_size_list, inter_tp_size_list):
        local_dp_size = local_size // local_tp_size
        inter_dp_size = num_nodes // inter_tp_size
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
        info_dict_base = {
            "local_tp_size": local_tp_size,
            "local_dp_size": local_dp_size,
            "inter_tp_size": inter_tp_size,
            "inter_dp_size": inter_dp_size,
            "dtype": dtype,
        }
        check_allreduce_allgather(
            mixed_comm,
            data["ar_ag"],
            local_bs_list,
            info_dict_base,
        )
        check_reducescatter_allreduce(
            mixed_comm,
            data["rs_ar"],
            local_bs_list,
            info_dict_base,
        )
        del mixed_comm
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("local_size", [2, 4, 8])
def test_mixed_comm(local_size, num_nodes, node_id, dtype, dist_init_method):
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
    for idx, process in enumerate(process_list):
        process.join()
        assert process.exitcode == 0, (
            f"Process {idx} failed with exit code {process.exitcode}"
        )
