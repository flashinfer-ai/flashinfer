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
    top_k: int = 2,
):
    from flashinfer.moe_ep import MoEEpTensors, MoEWeightPack

    hidden = 1024
    intermediate = 1024
    experts = 8
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
    slots = torch.arange(top_k, device="cuda")
    topk_ids = ((rows[:, None] * 3 + rank + slots) % experts).to(torch.int32)
    inputs = MoEEpTensors(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=torch.full(
            (tokens, top_k), 1.0 / top_k, dtype=torch.float32, device="cuda"
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


def _make_layer(rank: int, world_size: int, problem: dict, *, knobs=None):
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
                knobs=knobs,
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


def _owner_grouped_reference(routes, ids, *, world_size, local_experts):
    partials = []
    for owner in range(world_size):
        value = torch.zeros_like(routes[:, 0], dtype=torch.float32)
        for slot in range(ids.shape[1]):
            selected = (ids[:, slot] // local_experts) == owner
            value += routes[:, slot].float() * selected[:, None]
        partials.append(value.bfloat16())
    total = torch.zeros_like(routes[:, 0], dtype=torch.float32)
    for partial in partials:
        total += partial.float()
    return total.bfloat16()


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
@pytest.mark.parametrize("rank_local_combine", (False, True))
def test_sm120_nvfp4_rank_cache_and_combine_graph_epochs(rank_local_combine) -> None:
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    tokens = (124, 63, 7, 0)[rank]
    bucket = 128
    problem = _problem(rank, 4, tokens=tokens, capacity=256, top_k=6)
    base_knobs = {"dispatch_rank_cache": False, "rank_local_combine": False}
    candidate_knobs = {
        "dispatch_rank_cache": True,
        "rank_local_combine": rank_local_combine,
    }
    baseline = _make_layer(rank, 4, problem, knobs=base_knobs)
    candidate = _make_layer(rank, 4, problem, knobs=candidate_knobs)
    second_layer = _make_layer(rank, 4, problem, knobs=candidate_knobs)
    try:
        inputs = problem["inputs"]
        for layer in (baseline, candidate, second_layer):
            layer.stage_inputs(inputs, compile_tokens_per_rank=bucket)
            layer.compute_staged(output=None)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidate.stage_inputs(inputs, compile_tokens_per_rank=bucket)
            captured0 = candidate.compute_staged(output=None).clone()
            second_layer.stage_inputs(inputs, compile_tokens_per_rank=bucket)
            captured1 = second_layer.compute_staged(output=None).clone()

        storage = candidate._workspace._storages[bucket]
        assert candidate._workspace is second_layer._workspace
        for epoch in range(3):
            # Change the source rows and omit an expert owner in epoch 1.
            active_rows = tokens if epoch != 1 else tokens // 2
            ids = inputs.topk_ids
            rows = torch.arange(tokens, device="cuda")[:, None]
            slots = torch.arange(problem["top_k"], device="cuda")[None, :]
            ids.copy_((rows * 3 + slots + rank + epoch) % (6 if epoch == 1 else 8))
            ids[active_rows:].fill_(-1)
            inputs.hidden_states.mul_(-0.5 if epoch == 1 else 1.25)
            inputs.topk_weights.mul_(0.75)
            torch.cuda.synchronize()
            dist.barrier()
            baseline.stage_inputs(inputs, compile_tokens_per_rank=bucket)
            expected_direct = baseline.compute_staged(output=None).clone()
            routes = (
                baseline._workspace._storages[bucket].combine_output[:tokens].clone()
            )
            expected = (
                _owner_grouped_reference(routes, ids, world_size=4, local_experts=2)
                if rank_local_combine
                else expected_direct
            )
            torch.cuda.synchronize()
            dist.barrier()
            initial_generation = (
                int(storage.rank_combine_ready[-1, 0]) if rank_local_combine else 0
            )
            for _ in range(40):
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    captured0[:active_rows], expected[:active_rows], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    captured1[:active_rows], expected[:active_rows], atol=0, rtol=0
                )
                assert torch.isfinite(captured0[:active_rows]).all()
            if rank_local_combine:
                assert int(storage.rank_combine_ready[-1, 0]) == initial_generation + 80
            dist.barrier()
    finally:
        second_layer.destroy()
        candidate.destroy()
        baseline.destroy()


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
