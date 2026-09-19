"""GPU correctness coverage for the standalone MegaMOE FC12 backend."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MegaMoeFc12Config,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
)
from flashinfer.utils import is_sm100a_supported


@pytest.mark.skipif(
    not (torch.cuda.is_available() and is_sm100a_supported(torch.device("cuda"))),
    reason="requires SM100 CuTe-DSL MegaMOE FC12",
)
def test_fc12_processes_cluster_routed_rows():
    """Regression for the cluster routing path's local-expert histogram."""
    torch.manual_seed(0)
    num_tokens, num_experts, top_k = 1024, 512, 8
    hidden_size, intermediate_size = 256, 256
    x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.zeros((num_tokens, top_k), device="cuda", dtype=torch.int32)
    topk_weights = torch.full(
        (num_tokens, top_k),
        1 / top_k,
        device="cuda",
        dtype=torch.bfloat16,
    )
    w13 = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.05
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_num_experts=num_experts,
        ),
        backend=BackendOptions(candidates=(MegaMoeFc12Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=num_tokens),
    )
    weights = MoEWeightPack()
    weights.prepare_for(
        "megamoe_fc12",
        MegaMoeFc12Config.prepare_weights(
            w13,
            w2,
            variant=QuantVariant.BF16,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        ),
    )
    activation = MoEActivationPack(x, None, topk_ids, topk_weights)

    layer = MoELayer(config)
    runner = layer.runners[0]
    assert runner._enable_pdl
    inputs = runner.pack_inputs(activation, weights)
    actual = runner.forward(inputs)
    torch.cuda.synchronize()

    gate, up = F.linear(x.float(), w13[0].float()).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * up, w2[0].float()).bfloat16()
    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.1)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner.forward(inputs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=0.1, rtol=0.1)
