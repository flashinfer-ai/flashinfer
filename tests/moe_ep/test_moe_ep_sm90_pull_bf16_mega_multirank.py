"""Multi-rank tests for MoEEpMegaLayer (sm90_bf16_bf16_bf16_pull_cutedsl).

Launched via torchrun (4 Hopper GPUs)::

    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_sm90_pull_bf16_mega_multirank.py -v -m "gpu_4 and arch_hopper"

Every rank builds the same GLOBAL expert bank from a fixed seed and hands
the layer its local slice, so each rank checks its own tokens against the
pure-torch BF16 oracle over the global bank (``_sm90_pull_bf16_reference``) with
no gathers: real cross-rank NVSHMEM dispatch/combine vs independent math.
Routing rounds run back to back on one layer -- balanced, uneven (one rank
idle), masked, hot, all-remote, recovery -- then a lockstep CUDA graph
replays over in-place-mutated inputs.

Process isolation: imports the SM90 kernel tree (exclusive with SM100);
excluded from run_tests.sh ``unit`` and run by the ``mega_sm90`` target.
"""

from __future__ import annotations

import os

import pytest

from ._sm90_pull_bf16_reference import assert_bf16_close, sm90_bf16_moe_reference

pytest.importorskip("flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel")

GATE_UP_CLAMP = 10.0
HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOPK, MAX_TOKENS = 2048, 1024, 8, 4, 64


def _require_cuda():
    import torch

    from flashinfer.utils import is_sm90a_supported

    if not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("Requires SM90a")


def _launcher_ranks() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _global_weights():
    """Identical on every rank (fixed seed): the oracle's global expert bank."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(13)
    w13 = torch.randn(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=g
    ) * (HIDDEN**-0.5)
    w2 = torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=g) * (
        INTERMEDIATE**-0.5
    )
    return w13.to(torch.bfloat16), w2.to(torch.bfloat16)


def _routing_round(rank: int, world_size: int, name: str):
    """Per-rank ``MoEEpTensors`` for one round (see the module docstring)."""
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    x = torch.randn(
        MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(MAX_TOKENS, NUM_EXPERTS, device="cuda", generator=g)
    num_local = NUM_EXPERTS // world_size
    if name == "all_remote":
        scores[:, rank * num_local : (rank + 1) * num_local] = float("-inf")
    weights, ids = torch.topk(scores, TOPK, dim=-1, sorted=False)
    weights = torch.softmax(weights, dim=-1)
    if name == "balanced":
        # Guarantee cross-rank traffic: token 0 hits one expert per rank.
        forced = torch.arange(min(TOPK, world_size), device="cuda") * num_local
        ids[0, : forced.numel()] = forced
    n = MAX_TOKENS
    if name == "uneven":
        n = 0 if rank == 1 else max(MAX_TOKENS - 13 * rank, 1)
    elif name == "masked":
        ids[::3, 0] = -1
        ids[1::5, -1] = -1
        ids[5] = -1
    elif name == "hot":
        ids[:] = torch.arange(TOPK, device="cuda")
    return MoEEpTensors(hidden_states=x[:n], topk_ids=ids[:n], topk_weights=weights[:n])


def _make_layer(rank: int, world_size: int, w13, w2, **cfg):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    num_local = NUM_EXPERTS // world_size
    local = slice(rank * num_local, (rank + 1) * num_local)
    return MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=MAX_TOKENS,
            token_hidden_size=HIDDEN,
        ),
        weights=MoEWeightPack(w13=w13[local].contiguous(), w2=w2[local].contiguous()),
        backend=MegaConfig(
            megakernel=Sm90_Bf16_Bf16_Bf16_PullCutedsl_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=TOPK,
                gate_up_clamp=GATE_UP_CLAMP,
                **cfg,
            ),
        ),
    )


def _run_routing_rounds(rank: int, world_size: int, **cfg) -> int:
    import torch
    import torch.distributed as dist

    w13, w2 = _global_weights()
    layer = _make_layer(rank, world_size, w13, w2, **cfg)
    rounds = ["balanced", "uneven", "masked", "hot"]
    if world_size > 1:
        rounds.append("all_remote")
    rounds.append("balanced")
    try:
        for name in rounds:
            t = _routing_round(rank, world_size, name)
            y = layer.forward(t)
            torch.cuda.synchronize()
            dist.barrier()
            n = t.hidden_states.shape[0]
            assert y.shape == (n, HIDDEN), (rank, name)
            if n:
                ref = sm90_bf16_moe_reference(
                    t.hidden_states,
                    t.topk_ids,
                    t.topk_weights,
                    w13,
                    w2,
                    gate_up_clamp=GATE_UP_CLAMP,
                )
                assert_bf16_close(y, ref, label=f"rank {rank} {name}")
                if name == "masked":
                    assert torch.all(y[5] == 0), (rank, name)
        return rank
    finally:
        layer.destroy()
        torch.cuda.synchronize()
        dist.barrier()


_CASES = {
    "heuristic": {},
    "swap_pingpong_m128n128_k64": dict(
        swap_ab=True,
        pingpong=True,
        mma_tiler_mnk=(128, 128, 64),
        cluster_shape_mnk=(1, 2, 1),
    ),
    "native_m64n128": dict(swap_ab=False, mma_tiler_mnk=(64, 128, 128)),
    "native_coop_cga2x2": dict(
        swap_ab=False, mma_tiler_mnk=(64, 256, 128), cluster_shape_mnk=(2, 2, 1)
    ),
    "swap_m256n32": dict(swap_ab=True, mma_tiler_mnk=(256, 32, 128)),
    "swap_pingpong_cga1x2": dict(
        swap_ab=True,
        pingpong=True,
        mma_tiler_mnk=(128, 64, 128),
        cluster_shape_mnk=(1, 2, 1),
    ),
    "dedup_reuse_dispatch": dict(
        dedup_dispatch=True, token_back_mode="reuse_dispatch_warps"
    ),
    "grouped_token_back": dict(
        swap_ab=True, token_back_mode="reuse_dispatch_warps", grouped_token_back=True
    ),
}


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("case", sorted(_CASES))
def test_moe_ep_sm90_pull_bf16_mega_routing_rounds(case):
    """Real cross-rank BF16 EP vs the global-bank torch oracle, round by round."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    _run_routing_rounds(rank, world_size, **_CASES[case])


def _run_graph_lockstep(rank: int, world_size: int, **cfg) -> int:
    import torch
    import torch.distributed as dist

    w13, w2 = _global_weights()
    layer = _make_layer(rank, world_size, w13, w2, **cfg)
    graph = None
    try:
        t = _routing_round(rank, world_size, "balanced")
        layer.warmup()
        dist.barrier()
        y_eager = layer.forward(t).clone()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y_graph = layer.forward(t)
        dist.barrier()
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
        assert torch.equal(y_graph, y_eager), f"rank {rank}: replay != eager"

        masked = _routing_round(rank, world_size, "masked")
        t.topk_ids.copy_(masked.topk_ids)
        t.hidden_states.mul_(-0.5)
        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        y_replay = y_graph.clone()
        y_eager2 = layer.forward(t)
        torch.cuda.synchronize()
        dist.barrier()
        assert torch.equal(y_replay, y_eager2), f"rank {rank}: masked replay != eager"
        return rank
    finally:
        if graph is not None:
            graph.reset()
        layer.destroy()
        torch.cuda.synchronize()
        dist.barrier()


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("swap_ab", [False, True])
def test_moe_ep_sm90_pull_bf16_mega_graph_lockstep(swap_ab):
    """Collective warmup -> capture -> lockstep replays == eager (then masked inputs)."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    _run_graph_lockstep(rank, world_size, swap_ab=swap_ab)
