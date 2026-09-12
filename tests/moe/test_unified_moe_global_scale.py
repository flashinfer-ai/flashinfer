import pytest
import torch

from flashinfer.fused_moe.api import (
    BackendOptions,
    CuteDslConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SwiGLU,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.layer import MoELayer


def _sm100_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


@pytest.mark.skipif(not _sm100_available(), reason="requires SM100")
@pytest.mark.parametrize(
    "backend_config,backend_key",
    [
        (CuteDslConfig(), "cute_dsl"),
        (TrtllmFp4Config(), "trtllm_fp4_routed"),
    ],
)
def test_nvfp4_calibrated_global_scales_match_reference(backend_config, backend_key):
    """Non-unit activation, weight, and intermediate scales reach the kernel."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, hidden_size = 32, 256
    intermediate_size, num_experts, top_k = 256, 8, 2

    x = (torch.randn(num_tokens, hidden_size, device=device) * 0.5).to(torch.bfloat16)
    w1 = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, device=device)
        * 0.05
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device) * 0.05
    ).to(torch.bfloat16)
    logits = torch.randn(num_tokens, num_experts, device=device)
    topk_logits, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).float()
    topk_ids = topk_ids.to(torch.int32)

    fp4_range = 448.0 * 6.0
    a1_gs = (fp4_range / x.float().abs().max()).reshape(1)
    w1_gs = fp4_range / w1.float().abs().amax(dim=(1, 2))
    w2_gs = fp4_range / w2.float().abs().amax(dim=(1, 2))
    gemm1 = torch.einsum("th,eoh->teo", x.float(), w1.float())
    intermediate = (
        torch.nn.functional.silu(gemm1[..., intermediate_size:])
        * gemm1[..., :intermediate_size]
    )
    a2_gs = (fp4_range / intermediate.abs().max()).reshape(1)

    quant = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
    x_q, x_sf = TrtllmFp4Config.prepare_activations(
        x, quant=quant, hidden_states_scale_global=a1_gs
    )
    activations = MoEActivationPack(
        hidden_states_q=x_q,
        hidden_states_scale=x_sf,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states_scale_global=a1_gs,
    )
    weights = MoEWeightPack()
    weights.prepare_for(
        backend_key,
        backend_config.prepare_weights(
            w1,
            w2,
            quant=quant,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=SwiGLU(),
            device=device,
            gemm1_scales_global=w1_gs,
            gemm2_scales_global=w2_gs,
            intermediate_scale_global=a2_gs,
        ),
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=quant,
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(backend_config,)),
    )
    runner = MoELayer(config).runners[0]
    output = runner.forward(runner.pack_inputs(activations, weights), tactic=-1)

    reference = torch.zeros(num_tokens, hidden_size, device=device)
    for token in range(num_tokens):
        for slot in range(top_k):
            expert = int(topk_ids[token, slot])
            fc1 = torch.mv(w1[expert].float(), x[token].float())
            activated = (
                torch.nn.functional.silu(fc1[intermediate_size:])
                * fc1[:intermediate_size]
            )
            reference[token] += topk_weights[token, slot] * torch.mv(
                w2[expert].float(), activated
            )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), reference, rtol=0.5, atol=0.12)
