"""Kimi K3.1 Pro routed MoE usage on EP1, EP8 or EP16.

Uses synthetic sparse weights, not checkpoint accuracy validation. Routing,
latent projections, post-MoE RMSNorm and shared experts remain caller-owned.
"""

import pytest
import torch
import torch.distributed as dist

from flashinfer import moe_ep
from tests.moe_ep.test_sm100_kimi_k3 import (
    _check,
    _inputs,
    _reference,
    _runtime,
    _weights,
)

pytestmark = [pytest.mark.arch_blackwell, pytest.mark.node_exclusive]

# moonshotai/Kimi-K3.1-Pro config.json, revision d9e99b0792a5e93b78ea9bea62a68167e36ddd9e.
TEXT_CONFIG = {
    "routed_expert_hidden_size": 3584,
    "num_experts": 896,
    "moe_intermediate_size": 3072,
    "num_experts_per_token": 16,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
}


@pytest.mark.parametrize("combine_dtype", ["bf16", "mxfp8"])
def test_kimi_k31_public_api_eager_and_graph(combine_dtype):
    rank, world = _runtime()
    quant_kind = "mxfp4_mxfp8"
    config = moe_ep.Sm100_KimiK3_MegaMoeConfig(
        intermediate_size=TEXT_CONFIG["moe_intermediate_size"],
        top_k=TEXT_CONFIG["num_experts_per_token"],
        activation=TEXT_CONFIG["hidden_act"],
        situ_beta=TEXT_CONFIG["activation_situ_beta"],
        situ_linear_beta=TEXT_CONFIG["activation_situ_linear_beta"],
        quant_kind=quant_kind,
        combine_dtype=combine_dtype,
    )
    layer = moe_ep.MoEEpMegaLayer(
        bootstrap=moe_ep.BootstrapConfig(
            world_size=world,
            rank=rank,
            auto_bootstrap=False,
            process_group=dist.group.WORLD if world > 1 else None,
            device=torch.cuda.current_device(),
        ),
        fleet_params=moe_ep.FleetParams(
            num_experts=TEXT_CONFIG["num_experts"],
            token_hidden_size=TEXT_CONFIG["routed_expert_hidden_size"],
            max_tokens_per_rank=256,
        ),
        weights=_weights(rank, world, quant_kind),
        backend=moe_ep.MegaConfig(megakernel=config, quantize_input=True),
    )
    graph = None
    try:
        workspace = layer.create_workspace(256)
        sample = _inputs(17, rank, world, 0, mask_routes=False)
        # Input is already projected to routed width; route scores are normalized.
        tensors = moe_ep.MoEEpTensors(
            hidden_states=sample.hidden_states,
            topk_ids=sample.topk_ids,
            topk_weights=sample.topk_weights
            / sample.topk_weights.sum(dim=-1, keepdim=True),
        )
        layer.warmup(tensors, workspace=workspace)
        eager = layer(tensors, workspace=workspace).clone()
        assert eager.shape == (17, TEXT_CONFIG["routed_expert_hidden_size"])
        _check(eager, _reference(tensors, quant_kind, combine_dtype))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(tensors, workspace=workspace)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)

        changed = _inputs(17, rank, world, 1, mask_routes=False)
        changed.topk_weights /= changed.topk_weights.sum(dim=-1, keepdim=True)
        for name in ("hidden_states", "topk_ids", "topk_weights"):
            getattr(tensors, name).copy_(getattr(changed, name))
        graph.replay()
        torch.cuda.synchronize()
        _check(captured, _reference(tensors, quant_kind, combine_dtype))
    finally:
        graph = None
        torch.cuda.synchronize()
        layer.destroy()
