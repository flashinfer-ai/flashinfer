"""Replay LL windows with changing routes, weights and empty receivers."""

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep import (
    BootstrapConfig,
    CombineInputParams,
    DispatchInputParams,
    EpAlgorithm,
    EpLayout,
    FleetAlgoKnobNcclEpConfig,
    FleetParams,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
    HandleParams,
    create_fleet,
)
from tests.moe_ep.test_nccl_ep_release_multirank import (
    _assert_collective,
    _assert_collective_close,
    _assert_receive_ownership,
    released_runtime as released_runtime,
)

pytestmark = [pytest.mark.nvep, pytest.mark.gpu_4]


@pytest.mark.parametrize("adopted", [False, True])
@pytest.mark.parametrize("hidden", [2048, 5120])
@pytest.mark.parametrize("windows", [False, True])
def test_dispatch_window_graph_and_empty_receivers(
    released_runtime,  # noqa: F811 - shared pytest fixture
    hidden,
    windows,
    adopted,
):
    rank, world = released_runtime
    owner = None
    if adopted:
        from nccl.ep.interop.torch import get_nccl_comm_from_group

        owner = get_nccl_comm_from_group()
    tokens, capacity = 4, 8
    fleet = create_fleet(
        BootstrapConfig(
            rank=rank,
            world_size=world,
            stream=torch.cuda.current_stream().cuda_stream,
            nccl_comm=owner.ptr if owner is not None else None,
        ),
        FleetParams(
            num_experts=2 * world,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.RANK_MAJOR,
        ),
        [FleetAlgoKnobNcclEpConfig(num_topk=2, dispatch_zero_copy=windows)],
        backend="nccl_ep",
    )
    row = torch.arange(tokens, device="cuda", dtype=torch.int64)
    own = row % 2 + rank * 2
    peer = row % 2 + ((rank + 1) % world) * 2
    ids = torch.stack((own, peer), dim=1)
    weights = torch.tensor([0.25, 0.75], device="cuda").expand(tokens, 2).contiguous()
    values = torch.full(
        (tokens, hidden), float(rank + 1), device="cuda", dtype=torch.bfloat16
    )
    values[0].zero_()
    out = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    handle = fleet.create_handle(
        HandleParams(ids),
        [
            HandleAlgoKnobTopKWeights(weights),
            HandleAlgoKnobUserStream(torch.cuda.current_stream().cuda_stream),
        ],
    )

    received = None

    def step():
        nonlocal received
        handle.update(HandleParams(ids))
        received = handle.dispatch(DispatchInputParams(x=[values]))
        local = received.expert_tensors.float()
        if received.recv_topk_idx is not None:
            is_local = (received.recv_topk_idx >= 0) & (received.recv_topk_idx < 2)
            gain = received.recv_topk_weights.masked_fill(~is_local, 0).sum(-1)
            local *= gain.reshape(*local.shape[:-1], 1)
        return handle.combine(
            CombineInputParams(
                x=[local.to(torch.bfloat16)], out=out, out_is_stable=True
            )
        ).x

    graph = None
    try:
        step()
        step()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            result = step()
        for phase in range(3):
            if phase == 1:
                # Only rank zero owns these experts; every other receiver empties.
                ids[:, 0].fill_(0)
                ids[:, 1].fill_(1)
                weights.mul_(0.5)
            elif phase == 2:
                ids[:, 0].copy_(own)
                ids[:, 1].copy_(peer)
                values.copy_(torch.full_like(out, float(rank + 2)))
                weights.mul_(3)
            graph.replay()
            torch.cuda.synchronize()
            expected = (values.float() * weights.sum(-1, keepdim=True)).to(
                torch.bfloat16
            )
            _assert_collective_close(result, expected)
            _assert_receive_ownership(
                received, rank, world, tokens, concentrated=phase == 1
            )
        _assert_collective(
            len(fleet._dispatch_windows) == (1 if windows else 0),
            "unexpected dispatch-window count",
        )
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()
        handle.destroy()
        fleet.destroy()
        _assert_collective(
            not fleet._dispatch_windows, "dispatch windows remain after teardown"
        )
        dist.barrier()
        if owner is not None:
            from nccl.core import NcclRedOp

            check = torch.ones(1, device="cuda")
            owner.allreduce(
                check,
                check,
                NcclRedOp.SUM,
                stream=torch.cuda.current_stream().cuda_stream,
            )
            torch.cuda.synchronize()
            _assert_collective_close(check, torch.full_like(check, world))
            owner.destroy()
