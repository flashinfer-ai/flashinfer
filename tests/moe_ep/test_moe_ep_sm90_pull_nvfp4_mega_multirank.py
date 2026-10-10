"""Multi-rank tests for MoEEpLayer (sm90_bf16_nvfp4_bf16_pull_cutedsl, W4A16).

Launched via torchrun (4 Hopper GPUs)::

    torchrun --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_sm90_pull_nvfp4_mega_multirank.py -v -m "gpu_4 and arch_hopper"

Every rank builds the same GLOBAL NVFP4 expert bank (packed E2M1, E4M3
per-16 scales, FP32 per-expert alphas) from a fixed seed and hands the layer
its local slice; each rank checks its own tokens against the BF16 oracle on
the dequantized global bank (the decode is exact), with no gathers.  The
routing rounds of the BF16 multirank test (balanced, uneven with one idle
rank, masked, hot, all-remote, recovery) run back to back on one layer, plus
a round with runtime ``MoEEpTensors`` alphas; then a lockstep CUDA graph
replays over in-place-mutated inputs and alphas.

Process isolation: imports the SM90 kernel tree (exclusive with SM100);
excluded from run_tests.sh ``unit`` and run by the ``mega_sm90`` target.
"""

from __future__ import annotations

import dataclasses

import pytest

from ._sm90_pull_bf16_reference import assert_bf16_close, sm90_bf16_moe_reference
from .test_moe_ep_sm90_pull_bf16_mega_multirank import (
    GATE_UP_CLAMP,
    HIDDEN,
    INTERMEDIATE,
    MAX_TOKENS,
    NUM_EXPERTS,
    TOPK,
    _launcher_ranks,
    _require_cuda,
    _routing_round,
)

pytest.importorskip("flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel")


def _global_bank():
    """Identical on every rank (fixed seed): packed legs, scale legs, alphas."""
    import torch

    g = torch.Generator(device="cuda").manual_seed(13)
    packed, scales = [], []
    for n, k in ((2 * INTERMEDIATE, HIDDEN), (HIDDEN, INTERMEDIATE)):
        codes = torch.randint(
            0, 16, (NUM_EXPERTS, n, k), device="cuda", generator=g, dtype=torch.uint8
        )
        packed.append((codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous())
        scales.append(
            (
                (torch.rand(NUM_EXPERTS, n, k // 16, device="cuda", generator=g) + 0.5)
                * (k**-0.5)
                / 3
            ).to(torch.float8_e4m3fn)
        )
    alphas = [
        torch.rand(NUM_EXPERTS, device="cuda", generator=g) + 0.5 for _ in range(2)
    ]
    return packed, scales, alphas


def _dequant(packed, scales, alphas):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.common.nvfp4 import (
        dequantize_nvfp4,
    )

    return tuple(
        dequantize_nvfp4(p, s, a)
        for p, s, a in zip(packed, scales, alphas, strict=True)
    )


def _local(rank: int, world_size: int) -> slice:
    num_local = NUM_EXPERTS // world_size
    return slice(rank * num_local, (rank + 1) * num_local)


def _make_layer(rank: int, world_size: int, packed, scales, alphas, **cfg):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    local = _local(rank, world_size)
    return MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=MAX_TOKENS,
            token_hidden_size=HIDDEN,
        ),
        weights=MoEWeightPack(
            w13=packed[0][local].contiguous(),
            w2=packed[1][local].contiguous(),
            w13_scale=scales[0][local].contiguous(),
            w2_scale=scales[1][local].contiguous(),
        ),
        backend=MegaConfig(
            megakernel=Sm90_Bf16_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=TOPK,
                gate_up_clamp=GATE_UP_CLAMP,
                fc1_alpha=alphas[0][local].contiguous(),
                fc2_alpha=alphas[1][local].contiguous(),
                **cfg,
            ),
        ),
    )


def _runtime_alphas(alphas):
    """Global runtime-alpha bank (every rank derives the same one)."""
    return [a.flip(0) * 1.5 for a in alphas]


def _run_routing_rounds(rank: int, world_size: int, **cfg) -> int:
    import torch
    import torch.distributed as dist

    packed, scales, alphas = _global_bank()
    w13, w2 = _dequant(packed, scales, alphas)
    runtime = _runtime_alphas(alphas)
    w13_rt, w2_rt = _dequant(packed, scales, runtime)
    layer = _make_layer(rank, world_size, packed, scales, alphas, **cfg)
    rounds = ["balanced", "uneven", "masked", "hot"]
    if world_size > 1:
        rounds.append("all_remote")
    rounds += ["runtime_alpha", "balanced"]
    local = _local(rank, world_size)
    try:
        for name in rounds:
            if name == "runtime_alpha":
                t = dataclasses.replace(
                    _routing_round(rank, world_size, "balanced"),
                    fc1_alpha=runtime[0][local].contiguous(),
                    fc2_alpha=runtime[1][local].contiguous(),
                )
                ref_w13, ref_w2 = w13_rt, w2_rt
            else:
                t = _routing_round(rank, world_size, name)
                ref_w13, ref_w2 = w13, w2
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
                    ref_w13,
                    ref_w2,
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


# W4A16 is swap-AB only; the heuristic rows are all swap-AB.
_CASES = {
    "heuristic": {},
    "swap_m256n32": dict(mma_tiler_mnk=(256, 32, 128)),
    "swap_m256n16_cga2x1": dict(
        mma_tiler_mnk=(256, 16, 128), cluster_shape_mnk=(2, 1, 1)
    ),
    "swap_pingpong_cga1x2": dict(
        pingpong=True, mma_tiler_mnk=(128, 64, 128), cluster_shape_mnk=(1, 2, 1)
    ),
    "dedup_reuse_dispatch": dict(
        mma_tiler_mnk=(256, 32, 128),
        dedup_dispatch=True,
        token_back_mode="reuse_dispatch_warps",
    ),
    "grouped_token_back": dict(
        mma_tiler_mnk=(256, 32, 128),
        token_back_mode="reuse_dispatch_warps",
        grouped_token_back=True,
    ),
}


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("case", sorted(_CASES))
def test_moe_ep_sm90_pull_nvfp4_mega_routing_rounds(case):
    """Real cross-rank W4A16 EP vs the global-bank torch oracle, round by round."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    _run_routing_rounds(rank, world_size, **_CASES[case])


def _run_graph_lockstep(rank: int, world_size: int, **cfg) -> int:
    import torch
    import torch.distributed as dist

    packed, scales, alphas = _global_bank()
    layer = _make_layer(rank, world_size, packed, scales, alphas, **cfg)
    local = _local(rank, world_size)
    graph = None
    try:
        t = dataclasses.replace(
            _routing_round(rank, world_size, "balanced"),
            fc1_alpha=alphas[0][local].clone(),
            fc2_alpha=alphas[1][local].clone(),
        )
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
        t.fc1_alpha.mul_(1.25)
        t.fc2_alpha.copy_(t.fc2_alpha.flip(0))
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
def test_moe_ep_sm90_pull_nvfp4_mega_graph_lockstep():
    """Collective warmup -> capture -> lockstep replays == eager (then masked inputs, new alphas)."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    _run_graph_lockstep(rank, world_size)
