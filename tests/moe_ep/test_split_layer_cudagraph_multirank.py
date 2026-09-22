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
split, while nixl_ep has no handle-init step and uses a combined send/recv
kernel launch under capture, with the same device-side arrival waits.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

pytestmark = [
    pytest.mark.nvep,
    pytest.mark.gpu_4,
    pytest.mark.usefixtures("require_split_backend"),
]

_PG_TIMEOUT = timedelta(minutes=60)

NUM_TOKENS, NUM_EXPERTS, HIDDEN, TOPK = 64, 8, 4096, 4


def _init_dist():
    """Join the torchrun process group and pin this rank to its own device.

    The device pin has to come first: ``_build_layer`` captures
    ``torch.cuda.current_stream()`` and the transport registers its buffers on
    the current device, so a rank left on cuda:0 would advertise memory on a
    GPU its peers are not talking to.
    """
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


def _build_layer(rank, world_size, backend, algorithm=None):
    """Split layer with the default identity inner kernel.

    No expert math means a mismatch can only come from the transport or from
    the graph, which is all these tests are about. HIDDEN stays at 4096
    because nccl_ep LOW_LATENCY instantiates its kernels only for a fixed set
    of hidden sizes (2048, 2560, 4096, 5120, 6144, 7168, 8192); anything else
    aborts the process inside the device code.
    """
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        EpAlgorithm,
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
            algorithm=algorithm or EpAlgorithm.LOW_LATENCY,
        ),
        weights=dummy_moe_weights(
            num_local_experts=NUM_EXPERTS // world_size, hidden=HIDDEN
        ),
        backend=backend,
    )


def _make_tensors(rank, seed):
    """Per-rank inputs whose identity round trip is exactly ``x``.

    ``topk_weights`` comes from a softmax, so every token's combine weights
    sum to 1 and the identity kernel returns the activations unchanged. That
    closed form is what the LOW_LATENCY assertions compare against, so an
    arbitrary weight draw would silently reduce them to "replay matches
    eager" and stop witnessing the transport.
    """
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


# nixl_ep is LL-only (MVP), so only nccl_ep carries the HT case. HT reaches
# capture differently: its recv buffer is sized statically (max_per_rank *
# world), so no host sync sneaks into the captured region.
@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize(
    "backend,algo_name",
    [
        ("nccl_ep", "low_latency"),
        ("nccl_ep", "high_throughput"),
        ("nixl_ep", "low_latency"),
    ],
)
def test_split_layer_forward_is_capturable(backend, algo_name):
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

    from flashinfer.moe_ep import EpAlgorithm

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    algorithm = {
        "low_latency": EpAlgorithm.LOW_LATENCY,
        "high_throughput": EpAlgorithm.HIGH_THROUGHPUT,
    }[algo_name]
    layer = _build_layer(rank, world_size, backend, algorithm)
    t = _make_tensors(rank, 1234)

    # All collective work runs first and results are stashed; the assertions
    # come afterwards, once the layer is torn down. A bare assert mid-test
    # aborts one rank inside a collective and strands the rest at the next
    # barrier, turning a one-line failure into a wedged multi-hour job.
    state = layer.create_graph_state(t)
    layer.forward(t, graph_state=state)  # warmup, still eager
    eager = layer.forward(t, graph_state=state).clone()
    x_orig = t.hidden_states.clone()
    t.hidden_states.mul_(-3.0)
    eager_newx = layer.forward(t, graph_state=state).clone()
    t.hidden_states.copy_(x_orig)
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
    # Replay must reproduce eager. This is the capture property itself and
    # holds for both algorithms.
    torch.testing.assert_close(replay_same, eager, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(replay_newx, eager_newx, atol=5e-2, rtol=5e-2)
    # A replay that re-ran tracks the rewritten activations; one that did not
    # would still hold the previous result. True for both algorithms, and the
    # assertion that actually distinguishes them.
    assert not torch.allclose(replay_newx, replay_same, atol=5e-2, rtol=5e-2), (
        "replay did not track activations rewritten between replays"
    )
    if algorithm is EpAlgorithm.LOW_LATENCY:
        # LL writes every slot of its recv buffer and the weights sum to 1, so
        # the identity round trip returns x exactly, under any routing. HT only
        # fills the rows it actually received, so it has no such closed form.
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


# nccl_ep only. The ownership check lives in MoEEpSplitGraphState, above the
# comm backend, so running it a second time over nixl_ep would only rebuild a
# second transport to reach the identical raise.
@pytest.mark.nvep
@pytest.mark.gpu_4
def test_split_layer_rejects_foreign_graph_state():
    """A state carries one layer's handle, so a second layer must refuse it.

    Fleets are per layer: the state's persistent handle was created by its own
    layer's fleet -- a different communicator and a different set of transport
    buffers -- and the stated path never touches the calling layer's fleet at
    all. Accepting a foreign state therefore sends this layer's tokens over the
    other layer's transport and writes the result into the other state's ``out``
    buffer. In the normal case (every MoE layer shares FleetParams) no shape
    disagrees anywhere, so nothing faults: layer B's result simply overwrites
    layer A's and both forwards hand back the same tensor. A model with N MoE
    layers holds N layers and N states, which is exactly the shape of code an
    off-by-one over ``states[i]`` lives in.

    Also asserts that a non-state object is rejected by type rather than
    reaching ``_check`` and dying with an AttributeError, and that the guard
    fires before ``handle.update()`` -- otherwise a call that is ultimately
    refused would still have rebound the owner's routing on its way out.

    That ordering needs a witness of its own, and two obvious ones do not
    work: a refusal passing the owner's own ``t`` would rebind identical
    values even from a late guard (a no-op), and any owner forward run
    afterwards re-runs ``update()`` and repairs a clobbered binding. So the
    second refusal below passes routing the owner never bound, and the
    handle's binding is snapshotted around it with no owner forward in
    between.
    """
    import torch
    import torch.distributed as dist

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer_a = _build_layer(rank, world_size, "nccl_ep")
    layer_b = _build_layer(rank, world_size, "nccl_ep")
    t = _make_tensors(rank, 4321)

    # All collective work runs first and results are stashed; the assertions
    # come afterwards, once both layers are torn down. A bare assert mid-test
    # aborts one rank inside a collective and strands the rest at the next
    # barrier.
    state_a = layer_a.create_graph_state(t)
    before = layer_a.forward(t, graph_state=state_a).clone()
    # What routing the owner's persistent handle is currently bound to.
    # NcclEpHandle._topk_idx is private, and reaching into it is deliberate:
    # it is the only observable of that binding, and handle.update() is what
    # rewrites it, so it is exactly the witness for "the guard ran first".
    # The equality assertion at the bottom pins it to t.topk_ids, so a rename
    # or a change of meaning fails loudly instead of quietly voiding the
    # witness. That is why this test is nccl_ep-only twice over.
    bound_ptr = state_a._handle._topk_idx.data_ptr()
    bound_ids = state_a._handle._topk_idx.clone()
    torch.cuda.synchronize()
    dist.barrier()

    # Every rank makes the same mixup, so the old code completes the round
    # trip on all ranks and this test fails on the assertion below rather than
    # hanging half the job.
    cross = None
    try:
        layer_b.forward(t, graph_state=state_a)
    except ValueError as e:
        cross = f"{type(e).__name__}: {e}"
    torch.cuda.synchronize()
    dist.barrier()

    # The same mixup, now carrying routing the owner never bound. A guard
    # placed after handle.update() would leave the owner's handle pointing at
    # THESE ids; the snapshot is taken immediately, with no intervening owner
    # forward to re-run update() and repair it.
    foreign = _make_tensors(rank, 4322)
    cross_foreign = None
    try:
        layer_b.forward(foreign, graph_state=state_a)
    except ValueError as e:
        cross_foreign = f"{type(e).__name__}: {e}"
    after_ptr = state_a._handle._topk_idx.data_ptr()
    after_ids = state_a._handle._topk_idx.clone()
    torch.cuda.synchronize()
    dist.barrier()

    wrong_type = None
    try:
        layer_a.forward(t, graph_state=object())
    except TypeError as e:
        wrong_type = f"{type(e).__name__}: {e}"
    torch.cuda.synchronize()
    dist.barrier()

    # The owning layer must be unharmed by both refusals.
    after = layer_a.forward(t, graph_state=state_a).clone()
    torch.cuda.synchronize()
    dist.barrier()

    state_a.destroy()
    layer_a.destroy()
    layer_b.destroy()
    dist.barrier()

    # --- assertions: no collectives beyond this point ---------------------
    assert cross is not None, (
        "forward() accepted a graph state belonging to a different layer; "
        "the round trip ran on the other layer's handle and wrote the other "
        "state's out buffer"
    )
    assert "different" in cross and "MoEEpSplitLayer" in cross, cross
    assert wrong_type is not None, "forward() must reject a non-state graph_state"
    assert "create_graph_state" in wrong_type, wrong_type

    # The ordering witness. Preconditions first, so it cannot degenerate into
    # a tautology: the handle must really track t.topk_ids, and the refused
    # call must really have carried a different buffer.
    assert bound_ptr == t.topk_ids.data_ptr(), (
        "the handle's bound routing is no longer readable as "
        "_handle._topk_idx, so the ordering assertion below witnesses nothing"
    )
    assert foreign.topk_ids.data_ptr() != bound_ptr, (
        "the refused call must carry a topk_ids buffer the owner never bound"
    )
    assert cross_foreign is not None, "the second cross-layer call was accepted too"
    assert after_ptr == bound_ptr, (
        "a refused forward rebound the owner's persistent handle to the "
        "caller's routing: the ownership guard runs after handle.update(), so "
        "a rejected call still corrupts the owner's next replay"
    )
    assert torch.equal(after_ids, bound_ids), (
        "the owner's bound routing changed across a refused forward"
    )

    # Weaker, and deliberately labelled as such: this says the refusals leave
    # the owner's state usable, nothing about ordering. The forward that
    # produced `after` re-ran update() and would have repaired a clobbered
    # binding, which is what the data_ptr check above is for.
    torch.testing.assert_close(after, before, atol=5e-2, rtol=5e-2)


# nccl_ep only, like the ownership test above: the warmth guard lives in
# MoEEpSplitLayer.forward, above the comm backend, so a nixl_ep copy would
# only build a second transport to reach the identical raise.
@pytest.mark.nvep
@pytest.mark.gpu_4
def test_split_layer_capture_rejects_never_forwarded_layer():
    """A capture cannot be the layer's first round trip.

    ``create_graph_state()`` builds the fleet and the handle but runs no round
    trip, so on its own it warms nothing the capture needs: the inner MoE
    kernel is still unbuilt, and selecting its backend captures a graph of its
    own -- nested capture is illegal. Without the guard, the failure surfaces
    from inside the kernel and names nothing the caller wrote.
    """
    import torch
    import torch.distributed as dist

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer = _build_layer(rank, world_size, "nccl_ep")
    t = _make_tensors(rank, 202)

    # Collectives first, assertions after teardown (see the note in
    # test_split_layer_forward_is_capturable).
    state = layer.create_graph_state(t)
    torch.cuda.synchronize()
    dist.barrier()

    raised = None
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            layer.forward(t, graph_state=state)
    except Exception as e:
        raised = f"{type(e).__name__}: {e}"
    del graph
    torch.cuda.synchronize()
    dist.barrier()

    state.destroy()
    layer.destroy()
    dist.barrier()

    # --- assertions: no collectives beyond this point ---------------------
    assert raised is not None, (
        "capturing the layer's first ever round trip must be refused; it "
        "reaches the inner kernel's lazy build inside the capture"
    )
    # The type matters as much as the text: it separates "the guard fired"
    # from "the capture died of something else and the message happened to
    # mention a forward".
    assert raised.startswith("MoEEpConfigError"), raised
    assert "forward" in raised, raised


@pytest.mark.nvep
@pytest.mark.gpu_4
def test_split_layer_capture_accepts_stateless_warmup():
    """Pins the warmth guard's granularity: per layer, not per graph state.

    This test is not the coverage for the warmup guard itself -- delete the
    guard outright and this still passes, because all it asserts is "no raise"
    plus a replay that ran. The guard is held up by
    ``test_split_layer_capture_rejects_never_forwarded_layer``.

    What it does fail against is one specific wrong implementation: warmth
    tracked per graph state. A stateless eager forward warms what a capture
    needs -- the inner kernel's lazy build and the transport's cold first
    round trip, both of which belong to the layer and its fleet -- but it
    touches no state, so a per-state flag would refuse the capture that
    follows. That sequence is legal, and live: a server warms its layers with
    plain forwards before it decides to capture at all.
    """
    import torch
    import torch.distributed as dist

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer = _build_layer(rank, world_size, "nccl_ep")
    t = _make_tensors(rank, 303)

    # Stateless warmup only. It never touches the state's handle -- there is
    # no state yet.
    warm = layer.forward(t).clone()
    torch.cuda.synchronize()
    dist.barrier()

    state = layer.create_graph_state(t)
    torch.cuda.synchronize()
    dist.barrier()

    raised = None
    captured = None
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            captured = layer.forward(t, graph_state=state)
    except Exception as e:
        raised = f"{type(e).__name__}: {e}"
    torch.cuda.synchronize()
    dist.barrier()

    # Every rank decides this the same way (the guard is a local check), so
    # the ranks cannot disagree about whether to replay.
    replayed = None
    if raised is None:
        graph.replay()
        torch.cuda.synchronize()
        replayed = captured.clone()
    dist.barrier()

    # Retire the graph before freeing the buffers it holds pointers into.
    del graph
    state.destroy()
    layer.destroy()
    dist.barrier()

    # --- assertions: no collectives beyond this point ---------------------
    assert raised is None, (
        "capture was refused after an eager stateless forward, which is "
        f"exactly the warmup a capture needs: {raised}"
    )
    # Not just "it did not raise": a capture that recorded nothing useful
    # would still get here. Replay-vs-eager and the identity closed form are
    # covered in full by test_split_layer_forward_is_capturable.
    torch.testing.assert_close(replayed, warm, atol=5e-2, rtol=5e-2)


@pytest.mark.nvep
@pytest.mark.gpu_4
def test_split_layer_destroyed_rejects_reuse():
    """After destroy() the layer must refuse, not quietly rebuild.

    ``_ensure_fleet()`` only tests for ``None``, so without the flag a forward
    after destroy() builds a brand-new fleet on a comm runtime that has
    already been finalized.

    The layer runs one eager forward before ``destroy()``, so the fleet and
    the comm runtime it finalizes actually exist -- otherwise ``destroy()``
    has nothing to tear down and the test passes whatever it does.
    Everything after ``destroy()`` is plain Python and issues nothing.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import MoEEpConfigError

    rank, world_size = _init_dist()
    assert world_size >= 4, f"needs >=4 ranks, got {world_size}"

    layer = _build_layer(rank, world_size, "nccl_ep")
    t = _make_tensors(rank, 404)

    # Collectives first, assertions after teardown (see the note in
    # test_split_layer_forward_is_capturable).
    layer.forward(t)
    torch.cuda.synchronize()
    dist.barrier()

    layer.destroy()
    dist.barrier()

    # Both raise before touching the device, so a failure here strands no
    # rank in a collective; results are still stashed to match the file.
    forward_err = None
    try:
        layer.forward(t)
    except MoEEpConfigError as e:
        forward_err = str(e)

    create_err = None
    try:
        layer.create_graph_state(t)
    except MoEEpConfigError as e:
        create_err = str(e)

    assert forward_err is not None, "forward() on a destroyed layer must raise"
    assert "destroyed" in forward_err, forward_err
    assert create_err is not None, (
        "create_graph_state() on a destroyed layer must raise"
    )
    assert "destroyed" in create_err, create_err
