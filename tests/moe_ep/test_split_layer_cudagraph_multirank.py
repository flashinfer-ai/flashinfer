"""CUDA-graph capture of ``MoEEpSplitLayer.forward``, on 4+ GPUs.

Launched via torchrun::

    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_split_layer_cudagraph_multirank.py -v -m "nvep and gpu_4"

``test_moe_ep_cudagraph_multirank.py`` covers the same property one layer down,
driving ``Fleet``/``Handle`` directly, because the split *layer* could not
express the capture recipe at all: its forward created a Handle per call and
destroyed it in a ``finally``, so a captured graph replayed against freed
memory. This file covers the layer API that closes that gap --
``create_graph_state()`` holds one handle across forwards (the allocating half
outside the capture, ``update()`` recorded inside it) and pins the buffers the
graph binds.

Both comm backends are exercised. They reach the same place by different
routes: nccl_ep needs the real ``ncclEpInitHandle``/``ncclEpUpdateHandle``
split, while nixl_ep has no handle-init step and instead must drop its
host-side recv hook under capture (a host callback runs once at capture and
never again on replay).
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

_PG_TIMEOUT = timedelta(minutes=60)

NUM_TOKENS, NUM_EXPERTS, HIDDEN, TOPK = 64, 8, 4096, 4


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
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MoEEpSplitLayer,
        dummy_moe_weights,
    )

    return MoEEpSplitLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            stream=torch.cuda.current_stream().cuda_stream,
        ),
        fleet_params=FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=NUM_TOKENS,
            token_hidden_size=HIDDEN,
            dtype_bytes=2,
        ),
        weights=dummy_moe_weights(
            num_local_experts=NUM_EXPERTS // world_size, hidden=HIDDEN
        ),
        backend=backend,
    )


def _make_tensors(rank, seed):
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    g = torch.Generator(device="cuda").manual_seed(seed + rank)
    x = torch.randn(
        NUM_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g
    )
    ids = torch.randint(
        0,
        NUM_EXPERTS,
        (NUM_TOKENS, TOPK),
        device="cuda",
        dtype=torch.int64,
        generator=g,
    )
    # softmax => weights sum to 1, so the identity round trip returns x.
    w = torch.softmax(torch.randn(NUM_TOKENS, TOPK, device="cuda", generator=g), dim=-1)
    return MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=w)


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_split_layer_forward_is_capturable(backend):
    """Capture ``forward(graph_state=...)``, then replay across changed input.

    Asserts, in increasing order of what they would catch:

    1. capture completes;
    2. replay does not fault and reproduces the eager result. This is the
       regression the graph state exists for; with a handle created per
       forward it fails here with an illegal memory access;
    3. replay actually re-runs. The activations are rewritten in place between
       replays, so a graph that replayed nothing would leave the previous
       contents in the output buffer and still satisfy (1) and (2). This is
       the one that matters.
    """
    import torch
    import torch.distributed as dist

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer = _build_layer(rank, world_size, backend)
    t = _make_tensors(rank, 1234)

    # All collective work runs first and results are stashed; the assertions
    # come afterwards, once the layer is torn down. A bare assert mid-test
    # aborts one rank inside a collective and strands the rest at the next
    # barrier, turning a one-line failure into a wedged multi-hour job.
    state = layer.create_graph_state(t)
    layer.forward(t, graph_state=state)  # warmup, still eager
    eager = layer.forward(t, graph_state=state).clone()
    x_orig = t.hidden_states.clone()
    torch.cuda.synchronize()
    dist.barrier()

    # (1) Capture. create_graph_state() ran outside it; update() is inside.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer.forward(t, graph_state=state)
    dist.barrier()

    # (2) Replay must not fault and must reproduce eager.
    graph.replay()
    torch.cuda.synchronize()
    replay_same = captured.clone()
    dist.barrier()

    # (3) Rewrite the activations in place and replay. Only a round trip that
    # actually re-ran tracks the change.
    t.hidden_states.mul_(-3.0)
    graph.replay()
    torch.cuda.synchronize()
    replay_newx = captured.clone()
    dist.barrier()

    state.destroy()
    layer.destroy()
    dist.barrier()

    # --- assertions: no collectives beyond this point ---------------------
    torch.testing.assert_close(replay_same, eager, atol=5e-2, rtol=5e-2)
    # LL writes every slot of its recv buffer and the weights sum to 1, so the
    # identity round trip returns x exactly.
    torch.testing.assert_close(eager, x_orig, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(replay_newx, x_orig * -3.0, atol=5e-2, rtol=5e-2)


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_split_layer_forward_without_state_rejects_capture(backend):
    """The stateless forward must refuse capture rather than fault on replay.

    Capturing the per-forward create/destroy path succeeds silently and only
    fails at replay, as an illegal memory access. A caller who forgets
    ``create_graph_state()`` deserves the error at the point of the mistake.
    """
    import torch
    import torch.distributed as dist

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer = _build_layer(rank, world_size, backend)
    t = _make_tensors(rank, 99)

    layer.forward(t)  # eager still works, unchanged
    torch.cuda.synchronize()
    dist.barrier()

    raised = None
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            layer.forward(t)
    except Exception as e:
        raised = f"{type(e).__name__}: {e}"
    del graph
    torch.cuda.synchronize()
    dist.barrier()

    layer.destroy()
    dist.barrier()

    assert raised is not None, "capturing forward() without a graph_state must raise"
    assert "create_graph_state" in raised, raised


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize("backend", ["nccl_ep", "nixl_ep"])
def test_split_layer_graph_state_rejects_rebound_buffers(backend):
    """A graph binds addresses once, so a swapped-in tensor must be rejected.

    Passing fresh ids to a stated forward is the quiet failure this guards:
    eagerly it would appear to work, but the captured graph keeps reading the
    buffer it saw, so replays would silently serve stale routing.
    """
    import torch
    import torch.distributed as dist

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer = _build_layer(rank, world_size, backend)
    t = _make_tensors(rank, 7)

    state = layer.create_graph_state(t)
    layer.forward(t, graph_state=state)
    torch.cuda.synchronize()

    swapped = _make_tensors(rank, 8)
    swapped.hidden_states = t.hidden_states  # only the ids differ
    swapped.topk_weights = t.topk_weights
    raised = None
    try:
        layer.forward(swapped, graph_state=state)
    except ValueError as e:
        raised = str(e)
    torch.cuda.synchronize()
    dist.barrier()

    state.destroy()
    layer.destroy()
    dist.barrier()

    assert raised is not None, "a rebound topk_ids must be rejected"
    assert "topk_ids" in raised, raised
