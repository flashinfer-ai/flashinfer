"""W4A16 (NVFP4 weights x BF16 activations) MoE-EP split layer under a CUDA graph.

Launched via torchrun::

    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_split_layer_w4a16_cudagraph_multirank.py -v -m "nvep and gpu_4"

``test_split_layer_cudagraph_multirank.py`` captures the same layer with the
*identity* inner kernel, which isolates the transport. This file captures the
whole chain -- dispatch -> quantized grouped GEMM -> combine -- in one graph,
with real expert weights, which is what a served model actually runs.

The quantization is weight-only: ``QuantConfig(weight=NVFP4, activation=BF16)``.
Activations cross the wire as BF16 (``mxfp8_dispatch`` stays off) and reach the
MMA as BF16; only the weights are 4-bit. ``build_activation_pack`` passes BF16
rows through untouched for both W4A16 encodings, and ``prepare.py`` quantizes
the canonical BF16 weight pack into NVFP4 at layer construction.

``CuteDslConfig`` is pinned as the sole backend candidate because on SM100 it
is the one that serves ``(NVFP4, BF16)``: ``_CUTLASS_W4A16_ARCHS = (90,)`` is
Hopper-only and ``_CUTILE_W4A16_ARCHS = (89, 90, 120, 121)`` omits 100.
Verified on GB200 -- ``MoELayer`` with that candidate resolves to
``[CuteDslRunner]``, and leaving the default candidate list in place would make
the test's coverage depend on whichever backend happened to win autotune.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

_PG_TIMEOUT = timedelta(minutes=60)

# Kept small: the capture, not the shape, is under test. Both are multiples of
# 128 so the FP4 scale-factor layout is happy, and HIDDEN is 2048 because
# nccl_ep LOW_LATENCY instantiates its kernels only for a fixed set of hidden
# sizes -- 2048, 2560, 4096, 5120, 6144, 7168, 8192
# (contrib/nccl_ep/device/macros.cuh SWITCH_HIDDEN). Anything else aborts the
# process inside device/low_latency.cu, so 2048 is the smallest legal choice.
TOKENS_PER_RANK = 64
NUM_EXPERTS = 8
HIDDEN = 2048
INTERMEDIATE = 512
TOPK = 2


def _init_dist():
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=_PG_TIMEOUT,
        )
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    return rank, world


def _build_layer(rank, world_size, backend):
    """Split layer whose inner kernel is a real W4A16 fused MoE."""
    import torch

    from flashinfer.fused_moe.api import (
        BackendOptions,
        CuteDslConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
    )
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpSplitLayer,
        MoEWeightPack,
        NcclEpConfig,
        NvepConfig,
        SplitConfig,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    local_num_experts = NUM_EXPERTS // world_size
    local_expert_offset = rank * local_num_experts
    # EXPERT_MAJOR recv buffer is [local_experts, cap, hidden] with
    # cap = tokens_per_rank * world, so the compute sees this many rows.
    compute_max_tokens = local_num_experts * TOKENS_PER_RANK * world_size

    moe_config = MoEConfig(
        routing=RoutingConfig(num_experts=NUM_EXPERTS, top_k=TOPK),
        # The point of this test: 4-bit weights, 16-bit activations.
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.BF16),
        experts=ExpertConfig(
            intermediate_size=INTERMEDIATE,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(CuteDslConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=compute_max_tokens),
    )

    # Canonical weights are plain BF16; prepare.py quantizes them to NVFP4
    # during preprocess_weights, driven by moe_config.quant.
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    w13 = (
        torch.randn(
            local_num_experts,
            2 * INTERMEDIATE,
            HIDDEN,
            dtype=torch.bfloat16,
            device=device,
            generator=g,
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            local_num_experts,
            HIDDEN,
            INTERMEDIATE,
            dtype=torch.bfloat16,
            device=device,
            generator=g,
        )
        * 0.05
    )

    layer = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
        ),
        fleet_params=FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=TOKENS_PER_RANK,
            token_hidden_size=HIDDEN,
            dtype_bytes=2,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=SplitConfig(
            comm=NcclEpConfig() if backend == "nccl_ep" else NvepConfig(),
            kernel=FusedMoeKernelConfig(moe_config=moe_config),
        ),
    )
    return layer


def _make_tensors(rank):
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    x = torch.randn(
        TOKENS_PER_RANK, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g
    )
    ids = torch.randint(
        0,
        NUM_EXPERTS,
        (TOKENS_PER_RANK, TOPK),
        device="cuda",
        dtype=torch.int64,
        generator=g,
    )
    w = torch.softmax(
        torch.randn(TOKENS_PER_RANK, TOPK, device="cuda", generator=g), dim=-1
    )
    return MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=w)


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_w4a16_split_layer_chain_is_capturable(backend):
    """Capture dispatch -> NVFP4xBF16 MoE -> combine as one graph, then replay.

    Asserts, in increasing order of what they would catch:

    1. the chain runs eagerly at all, and produces finite, non-trivial output
       (a quant path that silently degenerates to zeros would otherwise sail
       through every graph assertion below);
    2. capture completes and replay does not fault;
    3. replay reproduces the eager result;
    4. replay actually re-runs -- the activations are rewritten in place
       between replays, and a graph that replayed nothing would leave the
       previous contents in the output buffer and still satisfy (2) and (3).
    """
    import torch
    import torch.distributed as dist

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer = _build_layer(rank, world_size, backend)
    t = _make_tensors(rank)

    # All collective work runs first and results are stashed; assertions come
    # afterwards, once the layer is torn down. A bare assert mid-test aborts
    # one rank inside a collective and strands the rest at the next barrier.
    state = layer.create_graph_state(t)
    layer.forward(t, graph_state=state)  # warmup: compiles + autotunes
    eager = layer.forward(t, graph_state=state).clone()
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer.forward(t, graph_state=state)
    dist.barrier()

    graph.replay()
    torch.cuda.synchronize()
    replay_same = captured.clone()
    dist.barrier()

    t.hidden_states.mul_(-3.0)
    graph.replay()
    torch.cuda.synchronize()
    replay_newx = captured.clone()
    dist.barrier()

    state.destroy()
    layer.destroy()
    dist.barrier()

    # --- assertions: no collectives beyond this point ---------------------
    # (1) the quant path produced something real
    assert torch.isfinite(eager).all(), "eager output has non-finite values"
    assert eager.abs().max().item() > 0.0, (
        "eager output is all zeros -- the W4A16 path degenerated, and every "
        "graph assertion below would pass vacuously"
    )
    # (2)+(3) replay is faithful. NVFP4 weights are quantized once at layer
    # construction and the same kernel runs both times, so eager and replay
    # should agree tightly; the tolerance covers only reduction-order noise.
    torch.testing.assert_close(replay_same, eager, atol=5e-2, rtol=5e-2)
    # (4) replay re-ran rather than serving a stale buffer
    assert not torch.allclose(replay_newx, replay_same, atol=5e-2, rtol=5e-2), (
        "replay did not track activations rewritten between replays"
    )


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize("backend", ["nccl_ep"])
def test_w4a16_graph_matches_eager_across_routing_changes(backend):
    """Rewriting the routing between replays must change what the graph serves.

    Activations alone cannot witness this: dispatch could ignore the ids
    entirely and a changed ``x`` would still change the output. Rewriting
    ``topk_ids`` in place isolates the routing half of the chain.
    """
    import torch
    import torch.distributed as dist

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer = _build_layer(rank, world_size, backend)
    t = _make_tensors(rank)

    state = layer.create_graph_state(t)
    layer.forward(t, graph_state=state)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer.forward(t, graph_state=state)
    dist.barrier()

    graph.replay()
    torch.cuda.synchronize()
    before = captured.clone()
    dist.barrier()

    # Derived from the current ids rather than a fresh draw, so every rank is
    # guaranteed a different-but-valid assignment and none can diverge.
    t.topk_ids.copy_((t.topk_ids + 1) % NUM_EXPERTS)
    graph.replay()
    torch.cuda.synchronize()
    after = captured.clone()
    dist.barrier()

    state.destroy()
    layer.destroy()
    dist.barrier()

    assert torch.isfinite(before).all() and torch.isfinite(after).all()
    assert not torch.allclose(after, before, atol=5e-2, rtol=5e-2), (
        "replay did not track topk_ids rewritten between replays"
    )
