"""SM12x MXFP8 x MXFP4 unified runner against a pure-Torch reference."""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fused_moe import (
    BackendOptions, ExecutionConfig, ExpertConfig, MoEActivationPack, MoEConfig,
    MoELayer, MoEWeightPack, QuantConfig, QuantVariant, RoutingConfig,
    SM12xMxfp8Mxfp4Config, SiTU, SwiGLU,
)
from flashinfer.utils import is_sm120a_supported
from tests.moe.test_sm12x_moe_gemm_mxfp8_mxfp4 import (
    calc_diff, dequant_e2m1_codes, mxfp4_weight_quantize, mxfp8_act_quantize,
    pack_mxfp4_moe_sfb,
)

pytestmark = pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available")


def _dequant_weight(q, sf):
    n, k2 = q.shape
    k = 2 * k2
    codes = torch.empty(n, k, dtype=torch.uint8, device=q.device)
    codes[:, 0::2] = q & 0x0F
    codes[:, 1::2] = (q >> 4) & 0x0F
    return (dequant_e2m1_codes(codes).reshape(n, k // 32, 32)
            * sf[:, :, None]).reshape(n, k)


def _quantize_weights(w):
    qs, sfs = zip(*(mxfp4_weight_quantize(x) for x in w))
    return torch.stack(qs), pack_mxfp4_moe_sfb(sfs), sfs


def _activation(x, intermediate, activation):
    linear, gate = x[:intermediate], x[intermediate:]
    if isinstance(activation, SwiGLU):
        gate = gate.clamp(max=activation.limit)
        linear = linear.clamp(-activation.limit, activation.limit)
        return gate * torch.sigmoid(gate) * linear
    return (activation.gate_scale * torch.tanh(gate / activation.gate_scale)
            * torch.sigmoid(gate) * activation.linear_scale
            * torch.tanh(linear / activation.linear_scale))


def _reference(x, ids, route_weights, w1q, w1sf, w2q, w2sf, activation):
    xq, xsf = mxfp8_act_quantize(x)
    xdeq = (xq.float().reshape(x.shape[0], -1, 128) * xsf[:, :, None]).reshape_as(x)
    w1 = [_dequant_weight(q, sf) for q, sf in zip(w1q, w1sf)]
    w2 = [_dequant_weight(q, sf) for q, sf in zip(w2q, w2sf)]
    out = torch.zeros_like(x, dtype=torch.float32)
    intermediate = 2 * w2q[0].shape[1]
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            hidden = _activation(xdeq[token].float() @ w1[expert].t(), intermediate, activation)
            hq, hsf = mxfp8_act_quantize(hidden[None])
            hdeq = (hq.float().reshape(1, -1, 128) * hsf[:, :, None]).reshape(-1)
            out[token] += route_weights[token, slot] * (hdeq @ w2[expert].t())
    return out


@pytest.mark.parametrize(
    "activation", [SwiGLU(limit=0.25), SiTU(gate_scale=4.0, linear_scale=25.0)]
)
def test_sm12x_mxfp8_mxfp4_unified_runner_matches_reference(activation):
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    torch.manual_seed(7)
    tokens, experts, top_k, hidden, intermediate = 8, 4, 2, 512, 512
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 10
    w1 = (torch.randn(experts, 2 * intermediate, hidden, device="cuda",
                      dtype=torch.bfloat16) / math.sqrt(hidden))
    w2 = (torch.randn(experts, hidden, intermediate, device="cuda",
                      dtype=torch.bfloat16) / math.sqrt(intermediate))
    token = torch.arange(tokens, device="cuda")
    ids = torch.stack([token % experts, (token + 1) % experts], dim=1).to(torch.int32)
    route_weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=1)
    w1q, w1sf_packed, w1sf = _quantize_weights(w1)
    w2q, w2sf_packed, w2sf = _quantize_weights(w2)

    act_pack = MoEActivationPack(
        hidden_states_q=x, hidden_states_scale=None, topk_ids=ids,
        topk_weights=route_weights,
    )
    weight_pack = MoEWeightPack()
    backend = SM12xMxfp8Mxfp4Config()
    weight_pack.prepare_for(
        "sm12x_mxfp8_mxfp4",
        backend.prepare_weights(w1q, w1sf_packed, w2q, w2sf_packed),
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.MXFP4, per_token_scale=False),
        experts=ExpertConfig(intermediate_size=intermediate, local_num_experts=experts),
        activation=activation,
        backend=BackendOptions(candidates=(backend,)),
        execution=ExecutionConfig(enable_pdl=False),
    )
    got = MoELayer(config, device=x.device)(act_pack, weight_pack)
    ref = _reference(x, ids, route_weights, w1q, w1sf, w2q, w2sf, activation)
    assert calc_diff(got.float(), ref) < 2e-2
