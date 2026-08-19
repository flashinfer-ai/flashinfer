"""SM120 MXFP4 x MXFP8 FlashInfer MegaMoE integration tests."""

from __future__ import annotations

import os

import pytest
import torch


def _packed_e2m1(
    shape: tuple[int, ...], generator: torch.Generator
) -> torch.Tensor:
    """Sparse finite E2M1 pairs with logical values in {0, 0.5}."""

    low = torch.randint(0, 2, shape, dtype=torch.uint8, device="cuda", generator=generator)
    high = torch.randint(0, 2, shape, dtype=torch.uint8, device="cuda", generator=generator)
    return (low | (high << 4)).view(torch.float4_e2m1fn_x2)


def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    raw = packed.view(torch.uint8)
    output = torch.empty((*raw.shape, 2), dtype=torch.float32, device=raw.device)
    output[..., 0] = lut[(raw & 0x0F).to(torch.int64)]
    output[..., 1] = lut[(raw >> 4).to(torch.int64)]
    return output.flatten(-2)


def _dequant(data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    expanded = scales.to(torch.float32).repeat_interleave(32, dim=-1)
    return data.to(torch.float32) * expanded[..., : data.shape[-1]]


def _torch_reference(problem: dict) -> torch.Tensor:
    from common.host_utils import mxfp8_quantize_per_block_32_row

    inputs = problem["inputs"]
    weights = problem["weights"]
    activation_q, activation_scale = mxfp8_quantize_per_block_32_row(
        inputs.hidden_states.to(torch.float32), torch.float8_e4m3fn
    )
    activation = _dequant(activation_q, activation_scale)
    tokens = inputs.hidden_states.shape[0]
    terms = torch.zeros(
        tokens,
        problem["topk"],
        problem["hidden"],
        dtype=torch.float32,
        device="cuda",
    )
    for expert in range(problem["experts"]):
        row, slot = torch.where(inputs.topk_ids == expert)
        if row.numel() == 0:
            continue
        w13 = _dequant(
            _unpack_e2m1(weights.w13[expert]), weights.w13_scale[expert]
        )
        fc1 = activation[row] @ w13.transpose(0, 1)
        gate, up = fc1.chunk(2, dim=-1)
        swiglu = torch.nn.functional.silu(gate) * up
        swiglu *= inputs.topk_weights[row, slot].unsqueeze(1)
        fc1_q, fc1_scale = mxfp8_quantize_per_block_32_row(
            swiglu, torch.float8_e4m3fn
        )
        fc1_dequant = _dequant(fc1_q, fc1_scale)
        w2 = _dequant(
            _unpack_e2m1(weights.w2[expert]), weights.w2_scale[expert]
        )
        terms[row, slot] = fc1_dequant @ w2.transpose(0, 1)
    return terms.sum(dim=1)


def _problem(
    rank: int,
    world_size: int,
    *,
    tokens: int = 64,
    capacity: int = 64,
    skewed_routing: bool = False,
    seed_offset: int = 0,
):
    from flashinfer.moe_ep import MoEWeightPack, MoEEpTensors

    hidden = 1024
    intermediate = 1024
    experts = 8
    topk = 4
    assert experts % world_size == 0
    local_experts = experts // world_size
    generator = torch.Generator(device="cuda").manual_seed(
        20260815 + rank + seed_offset
    )
    w13 = _packed_e2m1(
        (local_experts, 2 * intermediate, hidden // 2), generator
    )
    w2 = _packed_e2m1(
        (local_experts, hidden, intermediate // 2), generator
    )
    scale_dtype = torch.float8_e8m0fnu
    w13_scale = torch.full(
        (local_experts, 2 * intermediate, hidden // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(scale_dtype)
    w2_scale = torch.full(
        (local_experts, hidden, intermediate // 32),
        127,
        dtype=torch.uint8,
        device="cuda",
    ).view(scale_dtype)
    hidden_states = torch.randn(
        tokens,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ) * 0.125
    slots = torch.arange(topk, device="cuda", dtype=torch.int64)
    topk_ids = (
        torch.arange(tokens, device="cuda", dtype=torch.int64).unsqueeze(1) + slots
    ) % experts
    if skewed_routing:
        topk_ids[:, 0] = 0
        topk_ids[:, 1] = 1
        topk_ids[:, 2] = 2 + (torch.arange(tokens, device="cuda") % 2)
        topk_ids[:, 3] = 4 + (torch.arange(tokens, device="cuda") % 4)
    topk_weights = torch.full(
        (tokens, topk), 1.0 / topk, dtype=torch.float32, device="cuda"
    )
    return {
        "hidden": hidden,
        "intermediate": intermediate,
        "experts": experts,
        "topk": topk,
        "capacity": capacity,
        "weights": MoEWeightPack(w13, w2, w13_scale, w2_scale),
        "inputs": MoEEpTensors(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        ),
    }


def _all_gather_cat(tensor: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    contiguous = tensor.contiguous()
    byte_wire = contiguous.element_size() == 1 and contiguous.dtype != torch.uint8
    wire = contiguous.view(torch.uint8) if byte_wire else contiguous
    gathered = [torch.empty_like(wire) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, wire)
    result = torch.cat(gathered, dim=0)
    return result.view(contiguous.dtype) if byte_wire else result


def _global_weight_problem(problem: dict) -> dict:
    from flashinfer.moe_ep import MoEWeightPack

    weights = problem["weights"]
    return dict(
        problem,
        weights=MoEWeightPack(
            _all_gather_cat(weights.w13),
            _all_gather_cat(weights.w2),
            _all_gather_cat(weights.w13_scale),
            _all_gather_cat(weights.w2_scale),
        ),
    )


def _make_layer(rank: int, world_size: int, problem: dict):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
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
            megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=problem["intermediate"],
                top_k=problem["topk"],
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


@pytest.mark.arch_sm120
def test_sm120_w4a8_single_rank_replay_and_cuda_graph() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    problem = _problem(0, 1)
    layer = _make_layer(0, 1, problem)
    try:
        layer.warmup(problem["inputs"])
        eager0 = layer(problem["inputs"]).clone()
        eager1 = layer(problem["inputs"]).clone()
        torch.cuda.synchronize()
        assert torch.isfinite(eager0).all()
        torch.testing.assert_close(eager0, eager1, atol=0.0, rtol=0.0)
        reference = _torch_reference(problem)
        rel_l2 = (
            (eager0.float() - reference).norm()
            / reference.norm().clamp_min(1e-6)
        )
        assert rel_l2.item() < 0.03, f"single-rank rel_l2={rel_l2.item():.5f}"

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(problem["inputs"])
        graph.replay()
        replay0 = captured.clone()
        graph.replay()
        replay1 = captured.clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(replay0, replay1, atol=0.0, rtol=0.0)
        torch.testing.assert_close(eager0, replay0, atol=0.0, rtol=0.0)
    finally:
        layer.destroy()


@pytest.mark.arch_sm120
def test_sm120_w4a8_two_layers_share_workspace() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")

    first = _problem(0, 1)
    second = _problem(0, 1, seed_offset=97)
    second["inputs"] = first["inputs"]
    layer1 = _make_layer(0, 1, first)
    layer2 = _make_layer(0, 1, second)
    try:
        layer1.warmup(first["inputs"])
        layer2.warmup(second["inputs"])
        assert layer1._workspace is layer2._workspace
        assert len(layer1._workspace._frontends) == 2

        output1 = layer1(first["inputs"]).clone()
        output2 = layer2(second["inputs"]).clone()
        torch.cuda.synchronize()
        for output, problem in ((output1, first), (output2, second)):
            reference = _torch_reference(problem)
            rel_l2 = (
                (output.float() - reference).norm()
                / reference.norm().clamp_min(1e-6)
            )
            assert rel_l2.item() < 0.03

        layer1.destroy()
        output2_after_release = layer2(second["inputs"]).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output2,
            output2_after_release,
            atol=0.0,
            rtol=0.0,
        )
    finally:
        layer1.destroy()
        layer2.destroy()


@pytest.mark.gpu_4
@pytest.mark.arch_sm120
def test_sm120_w4a8_four_rank_tail_wave_and_cuda_graph_replay() -> None:
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 4:
        pytest.skip("requires exactly four ranks")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Exercise a real tail wave: every rank enters the collective kernel, but
    # the amount of useful work differs substantially. The one-row rank also
    # guards against accidentally requiring every rank to fill a tile.
    tokens_by_rank = (67, 64, 19, 1)
    problem = _problem(
        rank,
        world_size,
        tokens=tokens_by_rank[rank],
        capacity=80,
        skewed_routing=True,
    )
    layer = _make_layer(rank, world_size, problem)
    try:
        layer.warmup(problem["inputs"])
        eager0 = layer(problem["inputs"]).clone()
        eager1 = layer(problem["inputs"]).clone()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(eager0, eager1, atol=0.0, rtol=0.0)

        reference = _torch_reference(_global_weight_problem(problem))
        rel_l2 = (
            (eager0.float() - reference).norm()
            / reference.norm().clamp_min(1e-6)
        )
        assert rel_l2.item() < 0.03, f"rank {rank} rel_l2={rel_l2.item():.5f}"

        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(problem["inputs"])
        dist.barrier()
        replays = []
        for _ in range(16):
            graph.replay()
            replays.append(captured.clone())
        torch.cuda.synchronize()
        dist.barrier()
        for replay in replays:
            torch.testing.assert_close(replays[0], replay, atol=0.0, rtol=0.0)
        torch.testing.assert_close(eager0, replays[0], atol=0.0, rtol=0.0)
    finally:
        layer.destroy()
