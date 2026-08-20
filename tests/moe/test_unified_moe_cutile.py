"""Unified source-integrated cuTile BF16 MoE tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer.cutile import is_cuda_tile_available
from flashinfer.fused_moe import (
    ActivationConfig,
    BackendOptions,
    CuTileBf16Config,
    CuTileBf16Runner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS


def _config(
    *,
    num_experts: int = 4,
    top_k: int = 2,
    intermediate_size: int = 256,
    tune_max_num_tokens: int = 128,
) -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=ActivationConfig.swiglu,
        backend=BackendOptions((CuTileBf16Config(),)),
        execution=ExecutionConfig(
            do_finalize=True,
            enable_pdl=False,
            tune_max_num_tokens=tune_max_num_tokens,
        ),
    )


def test_cutile_bf16_config_architectures_and_registration():
    assert [arch for arch in range(140) if CuTileBf16Config.supported(arch)] == [
        89,
        90,
        120,
        121,
    ]
    assert _BACKEND_RUNNERS[CuTileBf16Config] is CuTileBf16Runner
    assert repr(CuTileBf16Config()) == "CuTileBf16Config()"


@pytest.mark.parametrize(
    ("arch", "extra_configs"),
    (
        (89, ((256, 32, 2), (256, 32, 1))),
        (90, ((256, 64, 2), (256, 32, 2), (256, 32, 1))),
        (120, ((256, 32, 2),)),
        (121, ((256, 32, 2),)),
    ),
)
def test_cutile_bf16_architecture_specific_gemm_configs(arch, extra_configs):
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner._device_arch = arch
    configs = runner._gemm_configs(k_in=2048, n=1536)

    assert configs[:4] == [
        (128, 32, 4),
        (128, 128, 1),
        (128, 64, 1),
        (64, 64, 4),
    ]
    assert configs[4:] == list(extra_configs)
    assert runner._block_sizes == (32, 64, 128)


@pytest.mark.parametrize("arch", (89, 90, 120, 121))
@pytest.mark.parametrize("num_tokens", (1, 1024))
def test_cutile_bf16_tactic_shortlist(arch, num_tokens):
    runner = CuTileBf16Runner.__new__(CuTileBf16Runner)
    runner._built = True
    runner._device_arch = arch
    runner.config = _config(
        num_experts=64,
        top_k=8,
        intermediate_size=768,
        tune_max_num_tokens=1024,
    )
    inputs = [
        torch.empty(num_tokens, 2048),
        torch.empty(num_tokens, 2048),
        torch.empty(num_tokens, 8, dtype=torch.int32),
        torch.empty(num_tokens, 8),
        torch.empty(0),
        torch.empty(0),
    ]

    tactics = runner.get_valid_tactics(inputs, None)

    expected_blocks = {32, 64} if num_tokens == 1 or arch in (120, 121) else {64, 128}
    assert len(tactics) == 12
    assert {tactic[0] for tactic in tactics} == expected_blocks
    assert all(len(tactic) == 7 for tactic in tactics)
    assert any(tactic[1] == 256 or tactic[4] == 256 for tactic in tactics)


def test_prepare_cutile_bf16_weights_converts_native_layout():
    num_experts, hidden_size, intermediate_size = 2, 16, 8
    w1 = torch.arange(
        num_experts * 2 * intermediate_size * hidden_size,
        dtype=torch.float32,
    ).reshape(num_experts, 2 * intermediate_size, hidden_size)
    w2 = torch.arange(
        num_experts * hidden_size * intermediate_size,
        dtype=torch.float32,
    ).reshape(num_experts, hidden_size, intermediate_size)
    w1 = w1.to(torch.bfloat16)
    w2 = w2.to(torch.bfloat16)

    view = CuTileBf16Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    up, gate = w1.chunk(2, dim=1)
    assert set(view) == {"w1", "w2"}
    assert view["w1"].is_contiguous() and view["w2"].is_contiguous()
    torch.testing.assert_close(view["w1"], torch.cat((gate, up), dim=1).transpose(1, 2))
    torch.testing.assert_close(view["w2"], w2.transpose(1, 2))


def _reference_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    intermediate_size = w2.shape[2]
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    expanded_hidden = (
        hidden_states[:, None, :]
        .expand(num_tokens, top_k, hidden_size)
        .reshape(-1, hidden_size)
    )
    gemm1 = torch.bmm(w1[flat_ids], expanded_hidden.unsqueeze(-1)).squeeze(-1)
    up, gate = gemm1.split(intermediate_size, dim=-1)
    intermediate = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    gemm2 = torch.bmm(w2[flat_ids], intermediate.unsqueeze(-1)).squeeze(-1)
    return (
        (
            gemm2.reshape(num_tokens, top_k, hidden_size).float()
            * topk_weights.unsqueeze(-1)
        )
        .sum(dim=1)
        .to(torch.bfloat16)
    )


def _cutile_device_is_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor in (89, 90, 120, 121) and is_cuda_tile_available()


@pytest.mark.skipif(
    not _cutile_device_is_supported(),
    reason="requires a working cuTile toolchain on SM89/SM90/SM120/SM121",
)
@pytest.mark.parametrize("num_tokens", (1, 17, 128))
def test_cutile_bf16_runner_matches_reference(num_tokens):
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_experts, top_k = 4, 2
    hidden_size, intermediate_size = 128, 256
    config = _config(
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        tune_max_num_tokens=128,
    )
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    canonical_w1 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    canonical_w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_ids = (
        torch.arange(num_tokens * top_k, dtype=torch.int32, device=device)
        .reshape(num_tokens, top_k)
        .remainder(num_experts)
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    native = CuTileBf16Config.prepare_weights(
        canonical_w1,
        canonical_w2,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cutile_bf16", native)
    activations = MoEActivationPack(
        hidden_states,
        None,
        topk_ids,
        topk_weights,
    )
    runner = CuTileBf16Runner(config, device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(activations, weights)

    actual = runner.forward(inputs, tactic=-1)
    expected = _reference_moe(
        hidden_states, topk_ids, topk_weights, canonical_w1, canonical_w2
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-1)
    tuned_gemms = {
        89: (256, 32, 2, 256, 32, 1),
        90: (256, 64, 2, 256, 64, 2),
        120: (256, 32, 2, 256, 32, 2),
        121: (256, 32, 2, 256, 32, 2),
    }[runner._device_arch]
    for block_size in runner._block_sizes:
        tuned_actual = runner.forward(inputs, tactic=(block_size, *tuned_gemms))
        torch.testing.assert_close(tuned_actual, expected, rtol=3e-2, atol=5e-1)


@pytest.mark.skipif(
    not _cutile_device_is_supported(),
    reason="requires a working cuTile toolchain on SM89/SM90/SM120/SM121",
)
def test_cutile_bf16_runner_is_cuda_graph_capturable():
    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size = 4, 128, 256
    config = _config(intermediate_size=intermediate_size, tune_max_num_tokens=4)
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    canonical_w1 = torch.randn(
        4,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    canonical_w2 = torch.randn(
        4,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    ids = torch.arange(8, dtype=torch.int32, device=device).reshape(4, 2) % 4
    routing_weights = torch.full((4, 2), 0.5, device=device)
    weights = MoEWeightPack()
    weights.prepare_for(
        "cutile_bf16",
        CuTileBf16Config.prepare_weights(
            canonical_w1,
            canonical_w2,
            num_local_experts=4,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ),
    )
    runner = CuTileBf16Runner(config, device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(
        MoEActivationPack(hidden_states, None, ids, routing_weights), weights
    )
    tuned_gemms = {
        89: (256, 32, 2, 256, 32, 1),
        90: (256, 64, 2, 256, 64, 2),
        120: (256, 32, 2, 256, 32, 2),
        121: (256, 32, 2, 256, 32, 2),
    }[runner._device_arch]
    tactic = (128, *tuned_gemms)
    runner.forward(inputs, tactic=tactic)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic=tactic)
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_moe(
        hidden_states, ids, routing_weights, canonical_w1, canonical_w2
    )
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=5e-1)


@pytest.mark.skipif(
    not _cutile_device_is_supported(),
    reason="requires a working cuTile toolchain on SM89/SM90/SM120/SM121",
)
def test_cutile_bf16_runs_through_unified_layer():
    torch.manual_seed(1)
    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size = 4, 128, 256
    config = _config(intermediate_size=intermediate_size, tune_max_num_tokens=4)
    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    )
    canonical_w1 = torch.randn(
        4,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    canonical_w2 = torch.randn(
        4,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    ids = torch.arange(8, dtype=torch.int32, device=device).reshape(4, 2) % 4
    routing_weights = torch.full((4, 2), 0.5, device=device)
    weights = MoEWeightPack()
    weights.prepare_for(
        "cutile_bf16",
        CuTileBf16Config.prepare_weights(
            canonical_w1,
            canonical_w2,
            num_local_experts=4,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ),
    )

    layer = MoELayer(config, device)
    actual = layer(
        MoEActivationPack(hidden_states, None, ids, routing_weights), weights
    )
    expected = _reference_moe(
        hidden_states, ids, routing_weights, canonical_w1, canonical_w2
    )

    assert layer.winner_backend == "cutile_bf16"
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-1)
