"""Experimental Firefly backend: a per-token NVFP4 routed MoE on SM100 / SM103 through MoELayer.

Listing FireflyNvfp4Config() as a backend candidate is the opt-in (it warns once as an
experimental backend). The activations are quantized to per-token NVFP4 outside the layer.
"""

import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    FireflyNvfp4Config,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SwiGLU,
)
from flashinfer.quantization import nvfp4_quantize_per_token_cute_dsl

num_experts, top_k, hidden, inter, num_tokens = 64, 8, 2048, 1024, 32
device = torch.device("cuda")
torch.manual_seed(33550336)

# Canonical BF16 experts: w1 [E, 2I, H] with the rows [up, gate], w2 [E, H, I].
w1 = (torch.randn(num_experts, 2 * inter, hidden, device=device) * 0.02).bfloat16()
w2 = (torch.randn(num_experts, hidden, inter, device=device) * 0.02).bfloat16()
weights = MoEWeightPack()
weights.prepare_for(
    "firefly_nvfp4",
    FireflyNvfp4Config.prepare_weights(
        w1,
        w2,
        num_local_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=inter,
        activation=SwiGLU(),
    ),
)

config = MoEConfig(
    routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
    quant=QuantConfig(
        weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4, per_token_scale=True
    ),
    experts=ExpertConfig(intermediate_size=inter),
    activation=SwiGLU(),
    execution=ExecutionConfig(tune_max_num_tokens=1024),
    backend=BackendOptions(candidates=(FireflyNvfp4Config(),)),
)
layer = MoELayer(config)

# Per-token NVFP4 activations: uint8 [M, H/2], linear E4M3 block scales [M, H/16],
# FP32 per-token scales [M]; pre-routed ids and weights.
x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
q, sf, per_token_scale = nvfp4_quantize_per_token_cute_dsl(
    x, torch.tensor([1.0 / 2688.0], device=device), sf_layout=2
)
logits = torch.randn(num_tokens, num_experts, device=device)
topk_weights, topk_ids = torch.sigmoid(logits).topk(top_k, dim=-1)

out = layer(
    MoEActivationPack(
        hidden_states_q=q,
        hidden_states_scale=sf.view(torch.float8_e4m3fn).reshape(num_tokens, -1),
        topk_ids=topk_ids.int(),
        topk_weights=topk_weights,
        per_token_scale=per_token_scale.reshape(-1),
    ),
    weights,
)
print(layer.winner_backend, out.shape, out.dtype)
