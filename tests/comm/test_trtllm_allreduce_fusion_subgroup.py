"""
Test allreduce fusion workspace creation with TP sub-groups.

Requires: 4 GPUs (2 TP groups of size 2).

Uses torch.distributed.launcher to spawn workers the same way
torchrun does. This is necessary because the bug only manifests under
torchrun-style initialization, where broadcast_object_list interprets `src` as
a global rank. With mp.Process-style init the bug does not reproduce.
"""

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend


def _worker_subgroup(tp_size: int, hidden_dim: int, use_unified_api: bool):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    assert world_size == 4, f"Requires 4 GPUs, got {world_size}"

    my_tp_group = None
    for start in range(0, world_size, tp_size):
        ranks = list(range(start, start + tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            my_tp_group = group

    assert my_tp_group is not None
    tp_rank = dist.get_rank(group=my_tp_group)

    comm_backend = TorchDistBackend(group=my_tp_group)

    if use_unified_api:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=tp_size,
            rank=tp_rank,
            max_token_num=1024,
            hidden_dim=hidden_dim,
            dtype=torch.bfloat16,
            comm_backend=comm_backend,
        )
    else:
        comm.trtllm_create_ipc_workspace_for_all_reduce_fusion(
            tp_rank=tp_rank,
            tp_size=tp_size,
            max_token_num=1024,
            hidden_dim=hidden_dim,
            group=my_tp_group,
        )
        workspace = None

    if workspace is not None:
        workspace.destroy()

    dist.destroy_process_group()


def _launch_subgroup_test(tp_size: int, hidden_dim: int, use_unified_api: bool):
    """Launch 4 workers via elastic_launch, same as torchrun."""
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

    config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=4,
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        start_method="spawn",
    )
    elastic_launch(config, _worker_subgroup)(tp_size, hidden_dim, use_unified_api)


@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("hidden_dim", [1024, 7168])
@pytest.mark.parametrize("use_unified_api", [False, True])
def test_trtllm_allreduce_fusion_subgroup(tp_size, hidden_dim, use_unified_api):
    available_gpus = torch.cuda.device_count()
    if available_gpus < 4:
        pytest.skip(f"Test requires 4 GPUs, only {available_gpus} available")

    _launch_subgroup_test(tp_size, hidden_dim, use_unified_api)
