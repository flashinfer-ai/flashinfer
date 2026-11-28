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


_WS2_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"


def _ws2_group_is_target(group) -> bool:
    """Whether every rank of ``group`` has compute capability 10.3: the sub-group-wide
    condition the creation vote decides on (one rank's GPU alone does not decide it)."""
    from flashinfer.utils import get_compute_capability

    flags = [None] * dist.get_world_size(group)
    local = get_compute_capability(torch.device("cuda")) == (10, 3)
    dist.all_gather_object(flags, local, group=group)
    return all(flags)


def _check_ws2_specialized_call(workspace, group, expect_specialized: bool) -> None:
    """One kARResidualRMSNorm call in the range of the specialized two-rank kernel on an
    ``auto`` workspace of a TP sub-group. The route is the sub-group's collective decision:
    the specialized backend (its state agreed over the sub-group) with one launch per rank
    when ``expect_specialized``, else the stock workspace with no specialized launch; every
    rank of the sub-group takes the same route, and the result is correct either way."""
    from flashinfer.comm._allreduce_fusion_ws2_specialized import (
        _allreduce_fusion_ws2_specialized_stats as stats,
    )

    is_specialized = isinstance(workspace, comm.SpecializedAllReduceFusionWorkspace)
    assert is_specialized == expect_specialized, stats()["last_state_rejection"]
    if expect_specialized:
        assert workspace.has_specialized_kernels, stats()["last_state_rejection"]
        assert workspace.fallback_backend in ("mnnvl", "trtllm")
    lo, hi = stats()["allowlist_num_tokens"][0]
    num_tokens, hidden_dim = min((lo + hi) // 2, 5461), 6144
    # x differs on every global rank (the two sub-groups get different values); residual and
    # gamma are shared, as in tensor parallelism.
    own = torch.Generator(device="cuda").manual_seed(11 + dist.get_rank())
    shared = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(num_tokens, hidden_dim, generator=own, device="cuda").bfloat16()
    residual = torch.randn(
        num_tokens, hidden_dim, generator=shared, device="cuda"
    ).bfloat16()
    gamma = (0.1 * torch.randn(hidden_dim, generator=shared, device="cuda")).bfloat16()
    # The reference reduces over this rank's sub-group only.
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
    launches = stats()["dispatch_count"] - before
    assert launches == int(expect_specialized)
    routes = [None] * dist.get_world_size(group)
    dist.all_gather_object(routes, (is_specialized, launches), group=group)
    assert len(set(routes)) == 1, (
        f"the sub-group's ranks took different routes: {routes}"
    )
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


def _check_ws2_specialized_veto(
    group, comm_backend, tp_size: int, tp_rank: int
) -> None:
    """A veto by one rank of the sub-group (its kill switch set at creation): ``auto`` builds
    the stock workspace on every rank of the sub-group, and no call launches the kernel."""
    import os

    if tp_rank == 1:
        os.environ[_WS2_DISABLE_ENV] = "1"
    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="auto",
            world_size=tp_size,
            rank=tp_rank,
            max_token_num=5461,
            hidden_dim=6144,
            dtype=torch.bfloat16,
            comm_backend=comm_backend,
            group=group,
        )
    finally:
        os.environ.pop(_WS2_DISABLE_ENV, None)
    _check_ws2_specialized_call(workspace, group, expect_specialized=False)
    workspace.destroy()


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
            group=my_tp_group,
        )
        if hidden_dim == 6144:
            _check_ws2_specialized_call(
                workspace, my_tp_group, _ws2_group_is_target(my_tp_group)
            )
            _check_ws2_specialized_veto(my_tp_group, comm_backend, tp_size, tp_rank)
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
