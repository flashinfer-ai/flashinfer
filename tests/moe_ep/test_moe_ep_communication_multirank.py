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
# One-sided variants pin the transport: "fence" never uses CFT counted writes,
# "cft" uses them for every step and skips where the platform lacks them.
_BACKENDS = [
    "nvlink_one_sided:fence",
    "nvlink_one_sided:cft",
    "cake",
    "nvlink_two_sided",
]
_ONE_SIDED_MODES = ["fence", "cft"]


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


def _backend_config(backend, **options):
    from flashinfer.moe_ep import (
        CakeAlltoAllConfig,
        NVLinkOneSidedConfig,
        NVLinkTwoSidedConfig,
        available_communication_backends,
    )

    name, _, mode = backend.partition(":")
    if name not in available_communication_backends():
        pytest.skip(f"{name} is not available on this machine")
    if mode:
        options["cft"] = mode == "cft"
    return {
        "nvlink_one_sided": NVLinkOneSidedConfig,
        "cake": CakeAlltoAllConfig,
        "nvlink_two_sided": NVLinkTwoSidedConfig,
    }[name](**options)


def _skip_without_cft(backend, comm):
    """Skip a CFT variant on platforms without counted writes; all ranks agree."""
    if backend.endswith(":cft") and not comm.cft_enabled:
        comm.destroy()
        pytest.skip("CFT counted writes are not available on this machine")


def _random_routing(num_tokens, num_experts, top_k, generator):
    import torch

    scores = torch.rand(num_tokens, num_experts, generator=generator)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(torch.rand(num_tokens, top_k, generator=generator), -1)
    return topk_ids.cuda(), topk_weights.cuda()


def _identity_round_trip_reference(x, topk_ids, world_size, experts_per_rank):
    """Split-layer output with the identity kernel: combine sums one unchanged
    copy of each token per distinct rank its experts live on."""
    import torch

    target_ranks = topk_ids.long() // experts_per_rank
    num_target_ranks = torch.stack(
        [(target_ranks == r).any(-1) for r in range(world_size)], -1
    ).sum(-1, keepdim=True)
    return x.float() * num_target_ranks.float()


@pytest.mark.gpu_2
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("backend", _BACKENDS)
def test_dispatch_combine_matches_local_reference(backend, top_k):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import BootstrapConfig, MoEEpCommParams, create_communication

    rank, world_size = _init_dist()
    config = _backend_config(backend)
    num_experts = 4 * world_size
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
    if backend.startswith("nvlink_one_sided"):
        _skip_without_cft(backend, comm)
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
    config = _backend_config("nvlink_one_sided")
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
        torch.testing.assert_close(
            out.float(), _identity_round_trip_reference(x, topk_ids, world_size, 4)
        )
    finally:
        dist.barrier()
        layer.destroy()


@pytest.mark.gpu_2
@pytest.mark.parametrize("backend", _BACKENDS)
def test_split_layer_cuda_graph_replays_new_inputs(backend):
    """A captured MoEEpSplitLayer forward serves inputs rewritten in place."""
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
    config = _backend_config(backend)
    num_tokens, hidden, top_k = 16, 1024, 2
    num_experts = 4 * world_size
    generator = torch.Generator().manual_seed(7 + rank)
    t = MoEEpTensors(
        hidden_states=torch.empty(
            num_tokens, hidden, dtype=torch.bfloat16, device="cuda"
        ),
        topk_ids=torch.empty(num_tokens, top_k, dtype=torch.int64, device="cuda"),
        topk_weights=torch.empty(num_tokens, top_k, dtype=torch.float32, device="cuda"),
    )

    def load_next_step():
        topk_ids, topk_weights = _random_routing(
            num_tokens, num_experts, top_k, generator
        )
        t.hidden_states.copy_(torch.randn(num_tokens, hidden, generator=generator))
        t.topk_ids.copy_(topk_ids)
        t.topk_weights.copy_(topk_weights)

    layer = MoEEpLayer(
        BootstrapConfig(world_size=world_size, rank=rank),
        FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=num_tokens,
            token_hidden_size=hidden,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.RANK_MAJOR,
        ),
        dummy_moe_weights(num_local_experts=4, hidden=hidden),
        backend=SplitConfig(comm=config, kernel=IdentityConfig()),
    )
    try:
        load_next_step()
        state = layer.create_graph_state(t)
        if backend.endswith(":cft") and not layer._communication.cft_enabled:
            pytest.skip("CFT counted writes are not available on this machine")
        layer.forward(t, graph_state=state)  # eager warmup
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = layer.forward(t, graph_state=state)
        for _ in range(3):
            load_next_step()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                out.float(),
                _identity_round_trip_reference(
                    t.hidden_states, t.topk_ids, world_size, 4
                ),
            )
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        layer.destroy()


@pytest.mark.gpu_2
@pytest.mark.parametrize("mode", _ONE_SIDED_MODES)
def test_nvlink_one_sided_low_precision_eplb_and_rank_mask(mode):
    """FP8 combine, EPLB statistics and an all-active rank mask in one round."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import BootstrapConfig, MoEEpCommParams, create_communication

    rank, world_size = _init_dist()
    backend = f"nvlink_one_sided:{mode}"
    num_experts = 4 * world_size
    top_k, hidden, max_tokens_per_rank = 8, 2048, 32
    config = _backend_config(
        backend,
        use_low_precision_combine=True,
        eplb_stats_num_experts=num_experts,
        enable_rank_mask=True,
    )
    generator = torch.Generator().manual_seed(4321 + rank)
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
    _skip_without_cft(backend, comm)
    try:
        stats_base = torch.arange(num_experts, dtype=torch.int32, device="cuda")
        mask = comm.active_rank_mask(range(world_size))
        received = comm.dispatch(
            x,
            topk_ids,
            topk_weights,
            max_tokens_per_rank=int(step_max),
            eplb_local_stats=stats_base + 1000 * rank,
            active_rank_mask=mask,
        )
        torch.testing.assert_close(
            received.eplb_gathered_stats,
            torch.stack([stats_base + 1000 * r for r in range(world_size)]),
        )
        first_local = rank * comm.num_local_experts
        ids = received.topk_ids.long()
        is_local = (ids >= first_local) & (ids < first_local + comm.num_local_experts)
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

        out = comm.combine(expert_output, active_rank_mask=mask)

        reference = x.float() * (topk_weights * (topk_ids.long() + 1).float()).sum(
            -1, keepdim=True
        )
        # Each rank's partial result crosses the wire as FP8 e4m3.
        error = (out.float() - reference).norm() / reference.norm()
        assert error < 0.08, f"relative error {error:.4f}"
    finally:
        dist.barrier()
        comm.destroy()
