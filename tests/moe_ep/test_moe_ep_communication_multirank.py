"""Multi-GPU dispatch/combine correctness of the MoEEpCommunication backends.

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_communication_multirank.py -v

Each rank applies a synthetic expert computation to the rows it receives: a
token routed to expert ``e`` with weight ``w`` contributes ``w * (e + 1) * x``.
After combine every token must equal ``sum_k w_k * (e_k + 1) * x`` computed
locally, which checks routing, payload transport, invalid-row handling and the
cross-rank reduction together.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

_PG_TIMEOUT = timedelta(minutes=60)
_BACKENDS = [
    ("nvlink_one_sided", "trtllm"),
    ("nvlink_one_sided", "cake"),
    ("nvlink_two_sided", None),
    ("nccl_ep", None),
]


def _init_dist():
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=_PG_TIMEOUT,
        )
    return dist.get_rank(), dist.get_world_size()


def _backend_config(name, kernel):
    import torch

    from flashinfer.moe_ep import (
        NCCLEPConfig,
        NVLinkOneSidedConfig,
        NVLinkTwoSidedConfig,
        available_communication_backends,
    )

    if name not in available_communication_backends():
        pytest.skip(f"{name} is not available on this machine")
    if name == "nvlink_one_sided":
        if kernel == "cake" and torch.cuda.get_device_capability() not in (
            (10, 0),
            (10, 3),
        ):
            pytest.skip("the cake kernels need compute capability 10.0 or 10.3")
        return NVLinkOneSidedConfig(kernel=kernel)
    if name == "nvlink_two_sided":
        return NVLinkTwoSidedConfig()
    return NCCLEPConfig()


def _random_routing(num_tokens, num_experts, top_k, generator):
    import torch

    scores = torch.rand(num_tokens, num_experts, generator=generator)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(torch.rand(num_tokens, top_k, generator=generator), -1)
    return topk_ids.cuda(), topk_weights.cuda()


@pytest.mark.gpu_2
@pytest.mark.parametrize(("backend", "kernel"), _BACKENDS)
def test_dispatch_combine_matches_local_reference(backend, kernel):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import BootstrapConfig, MoEEpCommParams, create_communication

    rank, world_size = _init_dist()
    config = _backend_config(backend, kernel)
    num_experts = 4 * world_size
    top_k = 4
    hidden = 2048
    max_tokens_per_rank = 32

    generator = torch.Generator().manual_seed(1234 + rank)
    num_tokens = int(
        torch.randint(1, max_tokens_per_rank + 1, (1,), generator=generator)
    )
    step_max = torch.tensor([num_tokens], device="cuda")
    dist.all_reduce(step_max, op=dist.ReduceOp.MAX)

    x = torch.randn(num_tokens, hidden, generator=generator).to("cuda", torch.bfloat16)
    topk_ids, topk_weights = _random_routing(num_tokens, num_experts, top_k, generator)

    comm = create_communication(
        BootstrapConfig(world_size=world_size, rank=rank),
        MoEEpCommParams(
            num_experts=num_experts,
            top_k=top_k,
            max_tokens_per_rank=max_tokens_per_rank,
            hidden_size=hidden,
        ),
        config,
    )
    try:
        for _ in range(2):  # the second round reuses the workspace
            received = comm.dispatch(
                x, topk_ids, topk_weights, max_tokens_per_rank=int(step_max)
            )
            rows = comm.ep_size * received.tokens_per_rank
            assert received.hidden_states.shape == (rows, hidden)
            assert received.topk_ids.shape == (rows, top_k)

            first_local = rank * comm.num_local_experts
            ids = received.topk_ids.long()
            is_local = (ids >= first_local) & (
                ids < first_local + comm.num_local_experts
            )
            row_scale = torch.where(
                is_local,
                received.topk_weights.float() * (ids + 1).float(),
                torch.zeros((), device="cuda"),
            ).sum(-1, keepdim=True)
            expert_output = torch.where(
                is_local.any(-1, keepdim=True),
                received.hidden_states.float() * row_scale,
                torch.zeros((), device="cuda"),
            ).to(torch.bfloat16)

            out = comm.combine(expert_output)

            reference = x.float() * (topk_weights * (topk_ids.long() + 1).float()).sum(
                -1, keepdim=True
            )
            torch.testing.assert_close(out.float(), reference, rtol=2e-2, atol=5e-2)
    finally:
        dist.barrier()
        comm.destroy()


@pytest.mark.gpu_2
def test_split_layer_identity_round_trip_over_nvlink_one_sided():
    """MoEEpSplitLayer drives a communication backend end to end.

    The identity kernel returns received rows unchanged and combine sums them
    over ranks, so each token comes back multiplied by the number of distinct
    ranks its experts live on.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        IdentityConfig,
        MoEEpLayer,
        MoEEpTensors,
        SplitConfig,
        dummy_moe_weights,
    )

    rank, world_size = _init_dist()
    config = _backend_config("nvlink_one_sided", "trtllm")
    num_experts = 4 * world_size
    hidden = 1024
    generator = torch.Generator().manual_seed(99 + rank)
    x = torch.randn(16, hidden, generator=generator).to("cuda", torch.bfloat16)
    topk_ids, topk_weights = _random_routing(16, num_experts, 2, generator)

    layer = MoEEpLayer(
        BootstrapConfig(world_size=world_size, rank=rank),
        FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=16,
            token_hidden_size=hidden,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.RANK_MAJOR,
        ),
        dummy_moe_weights(num_local_experts=4, hidden=hidden),
        backend=SplitConfig(comm=config, kernel=IdentityConfig()),
    )
    try:
        out = layer(
            MoEEpTensors(
                hidden_states=x,
                topk_ids=topk_ids.long(),
                topk_weights=topk_weights,
            )
        )
        target_ranks = topk_ids.long() // 4
        num_target_ranks = torch.stack(
            [(target_ranks == r).any(-1) for r in range(world_size)], -1
        ).sum(-1, keepdim=True)
        torch.testing.assert_close(out.float(), x.float() * num_target_ranks.float())
    finally:
        dist.barrier()
        layer.destroy()
