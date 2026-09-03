"""CUDA-graph capture of the LL split path, on 4+ GPUs.

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_moe_ep_cudagraph_multirank.py -v -m "nvep and gpu_4"

This is the test ``Handle.update()`` exists for. Creating a Handle per forward
cannot be captured: a graph records the device pointers it sees, so a handle
destroyed at the end of the captured forward leaves the replay dereferencing
freed memory. Observed on 4xB200 through vLLM as capture succeeding (PIECEWISE
35/35, FULL 35/35) and the first replay raising "an illegal memory access was
encountered" -- silent at capture, crash at replay.

The fix mirrors NCCL-EP's own recipe (``contrib/nccl_ep/ep_test.cu``,
``--use_cuda_graph``): ``ncclEpInitHandle`` stays OUTSIDE the capture and
``ncclEpUpdateHandle`` is recorded INSIDE it. Here that is ``create_handle()``
before the capture and ``handle.update()`` within it.

Note this drives ``Fleet``/``Handle`` directly rather than ``MoEEpLayer``.
``MoEEpLayer.forward`` creates and destroys a handle every call by design, so
it cannot express the split; the layer-level follow-up is the vLLM adapter.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest

_PG_TIMEOUT = timedelta(minutes=60)


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


class _Rig:
    """One Fleet plus the static buffers a captured graph will bind to."""

    def __init__(
        self, rank, world_size, num_tokens, num_experts, hidden, topk, seed, algorithm
    ):
        import torch

        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            create_fleet,
        )

        g = torch.Generator(device="cuda").manual_seed(seed + rank)
        self.x = torch.randn(
            num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
        )
        self.topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, topk),
            device="cuda",
            dtype=torch.int64,
            generator=g,
        )
        # softmax => weights sum to 1, so an identity round trip returns x.
        self.topk_weights = torch.softmax(
            torch.randn(num_tokens, topk, device="cuda", generator=g), dim=-1
        )
        # Combine writes here. A graph binds this address once, so it must be
        # preallocated rather than an empty_like() per call.
        self.out = torch.empty_like(self.x)
        self._sig = None

        self.fleet = create_fleet(
            BootstrapConfig(
                world_size=world_size,
                rank=rank,
                stream=torch.cuda.current_stream().cuda_stream,
            ),
            FleetParams(
                num_experts=num_experts,
                max_tokens_per_rank=num_tokens,
                token_hidden_size=hidden,
                dtype_bytes=2,
                algorithm=algorithm,
            ),
            [],
            backend="nccl_ep",
        )

    def create_handle(self):
        """The InitHandle half: called ONCE, outside any capture."""
        from flashinfer.moe_ep import HandleParams
        from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights

        self.handle = self.fleet.create_handle(
            HandleParams(topk_ids=self.topk_ids),
            algo_knobs=[HandleAlgoKnobTopKWeights(weights=self.topk_weights)],
        )
        return self.handle

    # Scale applied to the expert tensors between dispatch and combine. Any
    # value but 1.0 works; the point is that the "compute" is NOT the identity,
    # so a combine that failed to replay cannot be mistaken for a correct pass
    # through. With weights summing to 1 the round trip returns GAIN * x.
    GAIN = 2.0

    def step(self):
        """One update+dispatch+combine -- the sequence placed under capture.

        Mirrors ``SplitLayer.forward``, minus the per-forward create/destroy
        that makes that path uncapturable. The expert "compute" is a constant
        scale rather than the identity: an identity round trip returns x, which
        a stale output buffer could also do, so it cannot witness combine
        actually re-running.
        """
        from flashinfer.moe_ep import (
            CombineInputParams,
            DispatchInputParams,
            HandleParams,
        )

        self.handle.update(HandleParams(topk_ids=self.topk_ids))
        d = self.handle.dispatch(DispatchInputParams(x=[self.x]))
        # Whatever the transport reports about where tokens went. Held as a
        # live reference, not a copy: under capture this is the buffer the
        # replayed kernels write, which is exactly what we want to re-read.
        self._sig = d.expert_counts
        if self._sig is None:
            self._sig = d.recv_topk_idx
        d.expert_tensors.mul_(self.GAIN)
        c = self.handle.combine(CombineInputParams(x=[d.expert_tensors], out=self.out))
        self.handle.complete()
        return c.x

    def routing_signature(self):
        """Snapshot of what the transport last routed, or None if unreported.

        Comparing outputs cannot distinguish "update() replayed" from "update()
        skipped" -- an identity round trip returns x under any routing. So ask
        the transport directly instead.
        """
        return None if self._sig is None else self._sig.detach().clone()

    def destroy(self):
        self.handle.destroy()
        self.fleet.destroy()


@pytest.mark.nvep
@pytest.mark.gpu_4
@pytest.mark.parametrize("algo_name", ["low_latency", "high_throughput"])
def test_round_trip_is_capturable_with_a_persistent_handle(algo_name):
    """Capture update+dispatch+combine, then replay it across changed routing.

    Three properties, in increasing order of what they would catch:

    1. capture completes;
    2. replay does not fault. This is the regression that motivated
       ``Handle.update()``; with a handle created per forward it fails here
       with an illegal memory access;
    3. replay actually re-runs routing. The routing buffer is rewritten in
       place between replays and the transport must follow it. A graph frozen
       to capture-time routing would still pass (1) and (2) while silently
       serving stale experts, so this is the one that matters.
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

    num_tokens, num_experts, hidden, topk = 64, 8, 4096, 4
    rig = _Rig(rank, world_size, num_tokens, num_experts, hidden, topk, 1234, algorithm)
    rig.create_handle()

    # All collective work runs first and results are stashed; the assertions
    # come afterwards, once the fleet is torn down. A bare assert mid-test
    # aborts one rank inside a collective and strands the rest at the next
    # barrier, turning a one-line failure into a wedged multi-hour job.
    #
    # Warmup runs on the CURRENT stream, not a side stream. The handle issues
    # transport work on its own stream (see NcclEpHandle._op_stream), so a
    # side-stream clone would read the output buffer while the transport is
    # still writing it on another stream, and the comparison would be a race
    # rather than a check.
    rig.step()
    eager = rig.step().clone()
    torch.cuda.synchronize()
    dist.barrier()

    # (1) Capture. update() is inside; create_handle() was not.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = rig.step()
    dist.barrier()

    # (2) Replay must not fault, and must reproduce the eager result.
    graph.replay()
    torch.cuda.synchronize()
    replay_same = captured.clone()
    sig_same = rig.routing_signature()
    dist.barrier()

    # (3) Rewrite routing in place and replay again. Derived from the current
    # ids rather than a fresh random draw so every rank is guaranteed a
    # different-but-valid assignment and no rank can diverge into a skip.
    rig.topk_ids.copy_((rig.topk_ids + 1) % num_experts)
    graph.replay()
    torch.cuda.synchronize()
    replay_new = captured.clone()
    sig_new = rig.routing_signature()
    dist.barrier()

    # (4) Rewrite the ACTIVATIONS in place and replay. The routing signature
    # in (3) witnesses update+dispatch, but a combine that never replayed
    # would leave the previous contents in the output buffer and still satisfy
    # it. Only a replayed combine tracks a changed x.
    rig.x.mul_(-3.0)
    graph.replay()
    torch.cuda.synchronize()
    replay_newx = captured.clone()
    dist.barrier()

    rig.destroy()
    dist.barrier()

    # --- assertions: no collectives beyond this point ---------------------
    # Replaying the graph must reproduce what eager produced. This is the
    # capture property itself and applies to both algorithms.
    torch.testing.assert_close(replay_same, eager, atol=5e-2, rtol=5e-2)

    if algorithm is EpAlgorithm.LOW_LATENCY:
        # LL writes every slot of its [local_expert, max_tokens, hidden] recv
        # buffer, so the round trip is GAIN * x exactly (weights sum to 1),
        # under any routing -- which is why (3) has to interrogate the
        # transport rather than the output. rig.x was scaled by -3 in (4), so
        # compare the pre-(4) results against the ORIGINAL activations.
        x_before = rig.x / -3.0
        torch.testing.assert_close(eager, x_before * _Rig.GAIN, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(
            replay_new, x_before * _Rig.GAIN, atol=5e-2, rtol=5e-2
        )
        # The activation change must show up: this is what proves combine
        # replayed rather than the buffer merely holding a stale result.
        torch.testing.assert_close(replay_newx, rig.x * _Rig.GAIN, atol=5e-2, rtol=5e-2)
    else:
        # HT sizes its recv buffer to the static max (max_tokens_per_rank *
        # world) and dispatch only writes the slots that actually received
        # tokens. An identity pass-through therefore hands combine the
        # unwritten remainder, so "round trip returns x" is not a property HT
        # has -- real HT consumers either compute over the whole static buffer
        # or trim to recv_total_counter. Capture correctness is still fully
        # covered: replay must match eager above, and routing must track below.
        assert torch.isfinite(replay_same.float()).all()
        assert not torch.allclose(replay_newx, replay_new, atol=1e-3), (
            "output did not change when the activations did -- combine was "
            "not replayed inside the graph"
        )

    assert sig_same is not None, (
        "transport reported no routing signature (expert_counts and "
        "recv_topk_idx both None); cannot prove update() was replayed"
    )
    assert not torch.equal(sig_same, sig_new), (
        "routing signature did not change when topk_ids changed -- "
        "ncclEpUpdateHandle was not replayed inside the graph "
        f"(same={sig_same.flatten()[:8].tolist()}, "
        f"new={sig_new.flatten()[:8].tolist()})"
    )
    print(f"rank {rank}: {algo_name} capture + replay OK across changed routing")
