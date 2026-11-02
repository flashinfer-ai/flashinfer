"""Multi-GPU dispatch/combine correctness of the MoEEpCommunication backends.

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_communication_multirank.py -v
or via an MPI launcher, including across nodes:
    mpirun -np 8 pytest tests/moe_ep/test_moe_ep_communication_multirank.py -v

Each rank applies a synthetic expert computation to the rows it receives: a
token routed to expert ``e`` with weight ``w`` contributes ``w * (e + 1) * x``.
After combine every token must equal ``sum_k w_k * (e_k + 1) * x`` computed
locally, which checks routing, payload transport, invalid-row handling and the
cross-rank reduction together.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import timedelta

import pytest

from ._mpi_launch import adopt_mpi_launch

_PG_TIMEOUT = timedelta(minutes=60)
_BACKENDS = ["nvlink_one_sided", "cake", "nvlink_two_sided"]
# At import, so that collection's multi-rank checks see WORLD_SIZE.
adopt_mpi_launch()


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

    if backend not in available_communication_backends():
        pytest.skip(f"{backend} is not available on this machine")
    return {
        "nvlink_one_sided": NVLinkOneSidedConfig,
        "cake": CakeAlltoAllConfig,
        "nvlink_two_sided": NVLinkTwoSidedConfig,
    }[backend](**options)


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
    try:
        for _ in range(2):  # the second round reuses the workspace
            received = comm.dispatch(
                x, topk_ids, topk_weights, runtime_max_tokens_per_rank=int(step_max)
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
        layer.forward(t, graph_state=state)  # eager warmup
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = layer.forward(t, graph_state=state)
        with pytest.raises(ValueError, match="runtime_max_tokens_per_rank"):
            layer.forward(
                dataclasses.replace(t, runtime_max_tokens_per_rank=num_tokens),
                graph_state=state,
            )
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


def _split_candidate(quant_pair, prims_ts):
    from flashinfer.fused_moe.api import (
        CuteDslConfig,
        PrimsTsConfig,
        QuantFormat,
        TrtllmFp4Config,
        TrtllmFp8BlockConfig,
    )

    weight = QuantFormat[quant_pair[0]]
    if prims_ts:
        return PrimsTsConfig()
    if weight is QuantFormat.NVFP4:
        return TrtllmFp4Config()
    if weight is QuantFormat.DeepSeekFp8:
        return TrtllmFp8BlockConfig()
    return CuteDslConfig()


def _fused_moe_split_config(
    comm, quant_pair, *, rank, world_size, num_experts, prims_ts=False
):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
    )
    from flashinfer.moe_ep import FusedMoeKernelConfig, SplitConfig

    weight, activation = (QuantFormat[name] for name in quant_pair)
    local_num_experts = num_experts // world_size
    candidate = _split_candidate(quant_pair, prims_ts)
    moe_config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=2),
        quant=QuantConfig(weight=weight, activation=activation),
        experts=ExpertConfig(
            intermediate_size=512,
            local_expert_offset=rank * local_num_experts,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(candidate,)),
        execution=ExecutionConfig(tune_max_num_tokens=16 * world_size),
    )
    return SplitConfig(comm=comm, kernel=FusedMoeKernelConfig(moe_config=moe_config))


@pytest.mark.gpu_2
@pytest.mark.parametrize(
    "quant_pair, prims_ts",
    [
        (("NVFP4", "NVFP4"), False),
        (("MXFP4", "MXFP8"), False),
        (("DeepSeekFp8", "DeepSeekFp8"), False),
        (("NVFP4", "NVFP4"), True),
        (("MXFP4", "MXFP8"), True),
        (("MXFP8", "MXFP8"), True),
        (("DeepSeekFp8", "DeepSeekFp8"), True),
    ],
    ids=[
        "nvfp4",
        "mxfp4_mxfp8",
        "deepseek_fp8",
        "prims_ts-nvfp4",
        "prims_ts-mxfp4_mxfp8",
        "prims_ts-mxfp8",
        "prims_ts-deepseek_fp8",
    ],
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_split_layer_quantizes_tokens_before_dispatch(backend, quant_pair, prims_ts):
    """Quantized activations travel over the wire and match receiver-side
    quantization.

    NVFP4, MXFP8 and DeepSeek FP8 quantize each token row on its own, so a
    layer that quantizes before dispatch must produce the output of one that
    dispatches BF16 rows and quantizes them on the receiving rank.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.fused_moe.api import QuantFormat
    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
    )

    if prims_ts:
        from flashinfer.prims_ts.utils import is_prims_ts_available

        if not is_prims_ts_available():
            pytest.skip("Prims-TS needs nvidia-cutlass-dsl with cutlass.experimental")
    candidate = _split_candidate(quant_pair, prims_ts)
    major, minor = torch.cuda.get_device_capability()
    if not candidate.supported(major * 10 + minor):
        pytest.skip(f"{type(candidate).__name__} does not support SM{major}{minor}")
    rank, world_size = _init_dist()
    _backend_config(backend)
    num_tokens, hidden, intermediate = 16, 1024, 512
    num_experts = 4 * world_size
    generator = torch.Generator().manual_seed(23 + rank)
    x = torch.randn(num_tokens, hidden, generator=generator).to("cuda", torch.bfloat16)
    topk_ids, topk_weights = _random_routing(num_tokens, num_experts, 2, generator)
    t = MoEEpTensors(
        hidden_states=x, topk_ids=topk_ids.long(), topk_weights=topk_weights
    )
    weight_generator = torch.Generator().manual_seed(1000 + rank)
    w13 = torch.randn(4, 2 * intermediate, hidden, generator=weight_generator) * 0.05
    w2 = torch.randn(4, hidden, intermediate, generator=weight_generator) * 0.05

    def build_layer():
        return MoEEpLayer(
            BootstrapConfig(world_size=world_size, rank=rank),
            FleetParams(
                num_experts=num_experts,
                max_tokens_per_rank=num_tokens,
                token_hidden_size=hidden,
                algorithm=EpAlgorithm.LOW_LATENCY,
                layout=EpLayout.RANK_MAJOR,
            ),
            MoEWeightPack(
                w13=w13.to("cuda", torch.bfloat16), w2=w2.to("cuda", torch.bfloat16)
            ),
            backend=_fused_moe_split_config(
                _backend_config(backend),
                quant_pair,
                rank=rank,
                world_size=world_size,
                num_experts=num_experts,
                prims_ts=prims_ts,
            ),
        )

    quantized = build_layer()
    unquantized = build_layer()
    # Dispatch BF16 rows, so the receiving rank quantizes them.
    unquantized._kernel.dispatch_format = lambda: None
    try:
        out = quantized(t)
        expected = unquantized(t)
        assert (
            quantized._communication.params.dispatch_format
            is QuantFormat[quant_pair[1]]
        )
        assert unquantized._communication.params.dispatch_format is None
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        quantized.destroy()
        unquantized.destroy()


@pytest.mark.gpu_2
@pytest.mark.parametrize("backend", _BACKENDS)
def test_split_layer_dispatches_runtime_max_tokens_per_rank(backend):
    """MoEEpTensors.runtime_max_tokens_per_rank sizes the layer's dispatch.

    Ranks hold different token counts below the capacity; the dispatch
    reserves only the busiest rank's count per source rank and the round
    trip stays exact.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        IdentityConfig,
        MoEEpConfigError,
        MoEEpLayer,
        MoEEpTensors,
        SplitConfig,
        dummy_moe_weights,
    )

    rank, world_size = _init_dist()
    config = _backend_config(backend)
    capacity, hidden, top_k = 32, 1024, 2
    num_experts = 4 * world_size
    counts = [3 + 2 * r for r in range(world_size)]
    runtime_max = max(counts)
    num_tokens = counts[rank]
    generator = torch.Generator().manual_seed(21 + rank)
    x = torch.randn(num_tokens, hidden, generator=generator).to("cuda", torch.bfloat16)
    topk_ids, topk_weights = _random_routing(num_tokens, num_experts, top_k, generator)

    layer = MoEEpLayer(
        BootstrapConfig(world_size=world_size, rank=rank),
        FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.RANK_MAJOR,
        ),
        dummy_moe_weights(num_local_experts=4, hidden=hidden),
        backend=SplitConfig(comm=config, kernel=IdentityConfig()),
    )
    try:
        comm = layer._ensure_communication(top_k)
        dispatch = comm.dispatch
        tokens_per_rank = []

        def recording_dispatch(*args, **kwargs):
            result = dispatch(*args, **kwargs)
            tokens_per_rank.append(result.tokens_per_rank)
            return result

        comm.dispatch = recording_dispatch
        out = layer(
            MoEEpTensors(
                hidden_states=x,
                topk_ids=topk_ids.long(),
                topk_weights=topk_weights,
                runtime_max_tokens_per_rank=runtime_max,
            )
        )
        assert tokens_per_rank == [runtime_max]
        torch.testing.assert_close(
            out.float(), _identity_round_trip_reference(x, topk_ids, world_size, 4)
        )
        with pytest.raises(MoEEpConfigError, match="runtime_max_tokens_per_rank"):
            layer(
                MoEEpTensors(
                    hidden_states=x,
                    topk_ids=topk_ids.long(),
                    topk_weights=topk_weights,
                    runtime_max_tokens_per_rank=num_tokens - 1,
                )
            )
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        layer.destroy()


@pytest.mark.gpu_2
def test_nvlink_one_sided_low_precision_eplb_and_rank_mask():
    """FP8 combine, EPLB statistics and an all-active rank mask in one round."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import BootstrapConfig, MoEEpCommParams, create_communication

    rank, world_size = _init_dist()
    backend = "nvlink_one_sided"
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
    try:
        stats_base = torch.arange(num_experts, dtype=torch.int32, device="cuda")
        mask = comm.active_rank_mask(range(world_size))
        received = comm.dispatch(
            x,
            topk_ids,
            topk_weights,
            runtime_max_tokens_per_rank=int(step_max),
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


@pytest.mark.gpu_2
@pytest.mark.parametrize("backend", ["nvlink_one_sided", "cake"])
def test_checkpoint_restore_keeps_captured_graph_valid(backend):
    """A graph captured before checkpoint_prepare/checkpoint_restore replays
    correctly afterwards; repeated calls do nothing."""
    import torch
    import torch.distributed as dist

    from flashinfer.comm.mnnvl import MnnvlMemory, TorchDistBackend
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
    num_tokens, hidden, top_k = 16, 1024, 4
    num_experts = 4 * world_size
    generator = torch.Generator().manual_seed(11 + rank)
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

    def replay_and_check(graph, out):
        load_next_step()
        graph.replay()
        torch.cuda.synchronize()
        # Up to top_k BF16 copies are summed, so compare at BF16 precision.
        torch.testing.assert_close(
            out.float(),
            _identity_round_trip_reference(t.hidden_states, t.topk_ids, world_size, 4),
            rtol=2e-2,
            atol=5e-2,
        )

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
        comm = layer._communication
        layer.forward(t, graph_state=state)  # eager warmup
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = layer.forward(t, graph_state=state)
        replay_and_check(graph, out)

        memory = comm._state["mnnvl_mem"]
        cft_enabled = getattr(comm, "_cft_enabled", None)
        comm.checkpoint_prepare()
        comm.checkpoint_prepare()
        assert not MnnvlMemory.allocated_map[memory.ptr].mapped
        fresh_backend = TorchDistBackend()
        comm.checkpoint_restore(fresh_backend)
        comm.checkpoint_restore(fresh_backend)
        assert MnnvlMemory.allocated_map[memory.ptr].mapped
        if backend == "nvlink_one_sided":
            assert comm._cft_enabled == cft_enabled

        for _ in range(2):
            replay_and_check(graph, out)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        layer.destroy()
