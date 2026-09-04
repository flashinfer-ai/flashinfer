"""SM120 NVFP4 x NVFP4 FlashInfer MegaMoE integration tests."""

from __future__ import annotations

import os

import pytest
import torch


def _packed_e2m1(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randint(
        0,
        256,
        shape,
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    ).view(torch.float4_e2m1fn_x2)


def _problem(
    rank: int,
    world_size: int,
    *,
    tokens: int,
    capacity: int,
):
    from flashinfer.moe_ep import MoEEpTensors, MoEWeightPack

    hidden = 1024
    intermediate = 1024
    experts = 8
    top_k = 2
    local_experts = experts // world_size
    generator = torch.Generator(device="cuda").manual_seed(91 + rank)
    weights = MoEWeightPack(
        _packed_e2m1((local_experts, 2 * intermediate, hidden // 2), generator),
        _packed_e2m1((local_experts, hidden, intermediate // 2), generator),
        torch.ones(
            (local_experts, 2 * intermediate, hidden // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        torch.ones(
            (local_experts, hidden, intermediate // 16),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
    )
    hidden_states = (
        torch.randn(
            (tokens, hidden),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    rows = torch.arange(tokens, device="cuda")
    topk_ids = torch.stack(
        ((rows * 3 + rank) % experts, (rows * 5 + rank + 1) % experts), 1
    ).long()
    inputs = MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=torch.full(
            (tokens, top_k), 0.5, dtype=torch.float32, device="cuda"
        ),
    )
    return {
        "capacity": capacity,
        "experts": experts,
        "hidden": hidden,
        "intermediate": intermediate,
        "top_k": top_k,
        "weights": weights,
        "inputs": inputs,
    }


def _make_layer(rank: int, world_size: int, problem: dict):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world_size, rank=rank),
        fleet_params=FleetParams(
            num_experts=problem["experts"],
            max_tokens_per_rank=problem["capacity"],
            token_hidden_size=problem["hidden"],
        ),
        weights=problem["weights"],
        backend=MegaConfig(
            megakernel=Sm120_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["top_k"],
                gate_up_clamp=10.0,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


@pytest.mark.arch_sm120
def test_sm120_nvfp4_single_rank_replay_and_outer_cuda_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    problem = _problem(0, 1, tokens=16, capacity=16)
    layer = _make_layer(0, 1, problem)
    try:
        layer.warmup(problem["inputs"])
        eager0 = layer(problem["inputs"]).clone()
        eager1 = layer(problem["inputs"]).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(problem["inputs"])
        graph.replay()
        replay0 = captured.clone()
        graph.replay()
        replay1 = captured.clone()
        torch.cuda.synchronize()
        assert torch.isfinite(eager0).all()
        torch.testing.assert_close(eager0, eager1, atol=0.0, rtol=0.0)
        torch.testing.assert_close(eager0, replay0, atol=0.0, rtol=0.0)
        torch.testing.assert_close(replay0, replay1, atol=0.0, rtol=0.0)
    finally:
        layer.destroy()


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_sm120_nvfp4_four_rank_imbalanced_second_epoch() -> None:
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    tokens_by_rank = (17, 16, 7, 1)
    problem = _problem(
        rank,
        world_size,
        tokens=tokens_by_rank[rank],
        capacity=32,
    )
    layer = _make_layer(rank, world_size, problem)
    try:
        outputs = []
        for _ in range(3):
            layer.stage_inputs(
                problem["inputs"], compile_tokens_per_rank=max(tokens_by_rank)
            )
            outputs.append(layer.compute_staged(output=None).clone())
        torch.cuda.synchronize()
        dist.barrier()
        assert torch.isfinite(outputs[0]).all()
        for output in outputs[1:]:
            torch.testing.assert_close(outputs[0], output, atol=0.0, rtol=0.0)
    finally:
        layer.destroy()
