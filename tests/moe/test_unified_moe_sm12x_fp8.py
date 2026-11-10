"""SM12x DeepSeek FP8 unified runner against a pure-Torch reference."""

import math

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SM12xFp8Config,
    SwiGLU,
)
from flashinfer.testing.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
from flashinfer.utils import is_sm120a_supported
from tests.moe.test_cute_dsl_sm12x_fp8 import calc_diff


def _cuda_13_or_newer() -> bool:
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return get_cuda_version().major >= 13
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not is_cute_dsl_available(), reason="cute_dsl not available"),
    pytest.mark.skipif(
        not _cuda_13_or_newer(), reason="SM12x FP8 requires CUDA 13 or later"
    ),
]


def _dequant_weight(q, sf):
    n, k = q.shape
    return (
        q.float().reshape(n // 128, 128, k // 128, 128) * sf[:, None, :, None]
    ).reshape(n, k)


def _quantize_weights(w):
    qs, sfs = zip(*(per_block_cast_to_fp8(x) for x in w), strict=True)
    return torch.stack(qs), torch.stack(sfs).transpose(-1, -2).contiguous(), sfs


def _reference(x, ids, route_weights, w1q, w1sf, w2q, w2sf, activation):
    xq, xsf = per_token_cast_to_fp8(x)
    xdeq = (xq.float().reshape(x.shape[0], -1, 128) * xsf[:, :, None]).reshape_as(x)
    w1 = [_dequant_weight(q, sf) for q, sf in zip(w1q, w1sf, strict=True)]
    w2 = [_dequant_weight(q, sf) for q, sf in zip(w2q, w2sf, strict=True)]
    out = torch.zeros_like(x, dtype=torch.float32)
    intermediate = w2q[0].shape[1]
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            if expert < 0:  # unrouted pair
                continue
            gate_up = xdeq[token].float() @ w1[expert].t()
            linear, gate = gate_up[:intermediate], gate_up[intermediate:]
            gate = gate.clamp(max=activation.limit)
            linear = linear.clamp(-activation.limit, activation.limit)
            hidden = gate * torch.sigmoid(gate) * linear
            hq, hsf = per_token_cast_to_fp8(hidden[None])
            hdeq = (hq.float().reshape(1, -1, 128) * hsf[:, :, None]).reshape(-1)
            out[token] += route_weights[token, slot] * (hdeq @ w2[expert].t())
    return out


@pytest.mark.parametrize(
    "activation,enable_pdl", ((SwiGLU(), False), (SwiGLU(limit=0.25), None))
)
def test_sm12x_fp8_unified_runner_matches_reference(activation, enable_pdl):
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    torch.manual_seed(7)
    tokens, experts, top_k, hidden, intermediate = 8, 4, 2, 512, 512
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 10
    w1 = torch.randn(
        experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(hidden)
    w2 = torch.randn(
        experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(intermediate)
    token = torch.arange(tokens, device="cuda")
    ids = torch.stack([token % experts, (token + 1) % experts], dim=1).to(torch.int32)
    route_weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=1)
    w1q, w1sf_packed, w1sf = _quantize_weights(w1)
    w2q, w2sf_packed, w2sf = _quantize_weights(w2)

    act_pack = MoEActivationPack(
        hidden_states_q=x,
        hidden_states_scale=None,
        topk_ids=ids,
        topk_weights=route_weights,
    )
    weight_pack = MoEWeightPack()
    backend = SM12xFp8Config()
    weight_pack.prepare_for(
        "sm12x_fp8",
        backend.prepare_weights(w1q, w1sf_packed, w2q, w2sf_packed),
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=top_k),
        quant=QuantConfig(
            weight=QuantFormat.DeepSeekFp8,
            activation=QuantFormat.DeepSeekFp8,
            per_token_scale=False,
        ),
        experts=ExpertConfig(intermediate_size=intermediate, local_num_experts=experts),
        activation=activation,
        backend=BackendOptions(candidates=(backend,)),
        execution=ExecutionConfig(enable_pdl=enable_pdl),
    )
    got = MoELayer(config, device=x.device)(act_pack, weight_pack)
    ref = _reference(x, ids, route_weights, w1q, w1sf, w2q, w2sf, activation)
    assert calc_diff(got.float(), ref) < 2e-2


def _mask_unrouted(ids, pattern):
    """Mark unrouted slots the way vLLM pads CUDA-graph rows: expert id -1,
    with the routing weight left non-zero, so only the id can skip them."""
    num_tokens, top_k = ids.shape
    if pattern == "tail":
        masked = torch.zeros_like(ids, dtype=torch.bool)
        masked[num_tokens - max(1, num_tokens // 4) :] = True
    elif pattern == "all":
        masked = torch.ones_like(ids, dtype=torch.bool)
    elif pattern == "mixed":
        token = torch.arange(num_tokens, device=ids.device)[:, None]
        slot = torch.arange(top_k, device=ids.device)[None, :]
        masked = (token + slot) % 2 == 0
    else:
        raise ValueError(pattern)
    return ids.masked_fill(masked, -1)


@pytest.mark.parametrize("pattern", ["tail", "all", "mixed"])
@pytest.mark.parametrize("tokens", [8, 80], ids=["decode", "prefill"])
def test_sm12x_fp8_unified_runner_skips_unrouted(tokens, pattern):
    """A negative expert id is an unrouted pair (vLLM CUDA-graph padding):
    it must contribute nothing, so fully unrouted tokens are exact zeros and
    the routed pairs still match the reference."""
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    torch.manual_seed(11)
    # top_k * tokens <= 256 takes the single-CTA decode route, > 256 the prefill one.
    experts, top_k, hidden, intermediate = 32, 4, 512, 512
    activation = SwiGLU()
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 10
    w1 = torch.randn(
        experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(hidden)
    w2 = torch.randn(
        experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(intermediate)
    token = torch.arange(tokens, device="cuda")
    ids = torch.stack(
        [(token + 7 * slot) % experts for slot in range(top_k)], dim=1
    ).to(torch.int32)
    route_weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=1)
    w1q, w1sf_packed, w1sf = _quantize_weights(w1)
    w2q, w2sf_packed, w2sf = _quantize_weights(w2)
    weight_pack = MoEWeightPack()
    backend = SM12xFp8Config()
    weight_pack.prepare_for(
        "sm12x_fp8",
        backend.prepare_weights(w1q, w1sf_packed, w2q, w2sf_packed),
    )
    config = MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=top_k),
        quant=QuantConfig(
            weight=QuantFormat.DeepSeekFp8,
            activation=QuantFormat.DeepSeekFp8,
            per_token_scale=False,
        ),
        experts=ExpertConfig(intermediate_size=intermediate, local_num_experts=experts),
        activation=activation,
        backend=BackendOptions(candidates=(backend,)),
        execution=ExecutionConfig(enable_pdl=False),
    )
    layer = MoELayer(config, device=x.device)

    def run(topk_ids):
        act_pack = MoEActivationPack(
            hidden_states_q=x,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=route_weights,
        )
        return layer(act_pack, weight_pack)

    # An all-routed call first leaves stale rows in the recycled scratch.
    run(ids)
    masked_ids = _mask_unrouted(ids, pattern)
    got = run(masked_ids).float()
    torch.cuda.synchronize()
    assert torch.isfinite(got).all()
    fully_masked = (masked_ids < 0).all(dim=1)
    assert torch.equal(got[fully_masked], torch.zeros_like(got[fully_masked]))
    routed = ~fully_masked
    if routed.any():
        ref = _reference(x, masked_ids, route_weights, w1q, w1sf, w2q, w2sf, activation)
        assert calc_diff(got[routed], ref[routed]) < 2e-2
