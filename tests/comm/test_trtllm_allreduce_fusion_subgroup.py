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


def _check_ws2_specialized_call(workspace, group, tp_rank: int) -> None:
    """One kARResidualRMSNorm call in the range of the specialized two-rank kernel on an
    ``auto`` workspace of a TP sub-group: on compute capability 10.3 auto picks the
    specialized backend (the state was agreed over the sub-group) and the kernel
    dispatches; elsewhere auto builds the stock backend; the result is correct either
    way."""
    from flashinfer.comm._allreduce_fusion_ws2_specialized import (
        _allreduce_fusion_ws2_specialized_stats as stats,
    )
    from flashinfer.utils import get_compute_capability

    target = get_compute_capability(torch.device("cuda")) == (10, 3)
    is_specialized = isinstance(workspace, comm.SpecializedAllReduceFusionWorkspace)
    assert is_specialized == target, stats()["last_state_rejection"]
    assert getattr(workspace, "_ar_ws2", None) is not None or not target
    lo, hi = stats()["allowlist_num_tokens"][0]
    num_tokens, hidden_dim = min((lo + hi) // 2, 5461), 6144
    own = torch.Generator(device="cuda").manual_seed(11 + tp_rank)
    shared = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(num_tokens, hidden_dim, generator=own, device="cuda").bfloat16()
    residual = torch.randn(
        num_tokens, hidden_dim, generator=shared, device="cuda"
    ).bfloat16()
    gamma = (0.1 * torch.randn(hidden_dim, generator=shared, device="cuda")).bfloat16()
    x_all = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(x_all, x, group=group)
    residual_out = torch.empty_like(x)
    norm_out = torch.empty_like(x)
    before = stats()["dispatch_count"]
    comm.allreduce_fusion(
        input=x,
        workspace=workspace,
        pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm,
        launch_with_pdl=True,
        residual_out=residual_out,
        norm_out=norm_out,
        residual_in=residual,
        rms_gamma=gamma,
        rms_eps=1e-6,
        fp32_acc=True,
        weight_bias=1.0,
    )
    torch.cuda.synchronize()
    assert stats()["dispatch_count"] - before == int(target)
    s = torch.stack([t.float() for t in x_all]).sum(0).bfloat16()
    h = (s.float() + residual.float()).bfloat16()
    hf = h.float()
    norm = (
        hf
        * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6)
        * (1.0 + gamma.float())
    )
    torch.testing.assert_close(residual_out.float(), hf, atol=1e-2, rtol=1.6e-2)
    torch.testing.assert_close(norm_out.float(), norm, atol=1e-2, rtol=1.6e-2)


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
            # 6144: auto with the vLLM workspace size, so that it picks the specialized
            # backend (its state agreed over the sub-group) on compute capability 10.3.
            backend="auto" if hidden_dim == 6144 else "trtllm",
            world_size=tp_size,
            rank=tp_rank,
            max_token_num=5461 if hidden_dim == 6144 else 1024,
            hidden_dim=hidden_dim,
            dtype=torch.bfloat16,
            comm_backend=comm_backend,
        )
        if hidden_dim == 6144:
            _check_ws2_specialized_call(workspace, my_tp_group, tp_rank)
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
@pytest.mark.parametrize("hidden_dim", [1024, 6144, 7168])
@pytest.mark.parametrize("use_unified_api", [False, True])
def test_trtllm_allreduce_fusion_subgroup(tp_size, hidden_dim, use_unified_api):
    available_gpus = torch.cuda.device_count()
    if available_gpus < 4:
        pytest.skip(f"Test requires 4 GPUs, only {available_gpus} available")

    _launch_subgroup_test(tp_size, hidden_dim, use_unified_api)
