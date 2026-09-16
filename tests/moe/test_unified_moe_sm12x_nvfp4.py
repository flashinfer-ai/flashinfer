"""SM12x NVFP4 unified runner against the composed low-level op chain."""

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
    SM12xNvfp4Config,
    SwiGLU,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp4_fc1_act_q1 import (
    cute_dsl_sm12x_fc1_act_q1_nvfp4,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp4_fc2_finalize import (
    cute_dsl_sm12x_fc2_finalize_nvfp4,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_fp4_q0_route_triton import (
    nvfp4_q0_route_triton,
)
from flashinfer.fused_moe.runners import SM12xNvfp4Runner
from flashinfer.utils import is_sm120a_supported

pytestmark = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="cute_dsl not available"
)


def _pack_e2m1(codes):
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _pack_sf_128x4(sf):
    rows, cols = sf.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_cols = (cols + 3) // 4 * 4
    row = torch.arange(rows, device=sf.device, dtype=torch.int64)[:, None]
    col = torch.arange(cols, device=sf.device, dtype=torch.int64)[None, :]
    index = (
        ((row >> 7) * (padded_cols >> 2) + (col >> 2)) * 512
        + (row & 31) * 16
        + ((row & 127) >> 5) * 4
        + (col & 3)
    )
    out = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8, device=sf.device)
    out[index] = sf
    return out.view(padded_rows, padded_cols)


def _make_weight(rows, columns, seed):
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None]
    col = torch.arange(columns, device="cuda", dtype=torch.int64)[None, :]
    codes = ((row * 5 + col * 3 + seed) & 7).to(torch.uint8)
    codes |= (((row + col + seed) & 1) << 3).to(torch.uint8)
    sf_col = torch.arange(columns // 16, device="cuda", dtype=torch.int64)[None, :]
    choices = (
        torch.tensor([0.0625, 0.125, 0.25, 0.5], device="cuda", dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    sf = choices[(row * 3 + sf_col * 5 + seed) & 3]
    return _pack_e2m1(codes), _pack_sf_128x4(sf)


@pytest.mark.parametrize("tokens", [8, 129])
def test_sm12x_nvfp4_unified_runner_matches_composed_ops(tokens):
    if not (torch.cuda.is_available() and is_sm120a_supported(torch.device("cuda"))):
        pytest.skip("requires an SM120a device")
    torch.manual_seed(11)
    experts, top_k, hidden, intermediate = 4, 2, 512, 512
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 10
    token = torch.arange(tokens, device="cuda")
    ids = torch.stack([token % experts, (token + 1) % experts], dim=1).to(torch.int32)
    route_weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=1)
    w1, w1_sf = zip(
        *(
            _make_weight(2 * intermediate, hidden, 7 + expert)
            for expert in range(experts)
        ),
        strict=True,
    )
    w2, w2_sf = zip(
        *(_make_weight(hidden, intermediate, 17 + expert) for expert in range(experts)),
        strict=True,
    )
    q0_scale = torch.tensor([96.0], dtype=torch.float32, device="cuda")
    q1_scale = torch.tensor([96.0], dtype=torch.float32, device="cuda")
    inverse_q0 = q0_scale.reciprocal().expand(experts).contiguous()
    w2_alpha = q1_scale.reciprocal().contiguous()
    native = SM12xNvfp4Config.prepare_weights(
        torch.stack(w1),
        torch.cat(w1_sf),
        inverse_q0,
        inverse_q0.clone(),
        torch.stack(w2),
        torch.cat(w2_sf),
        w2_alpha,
        q0_scale,
        q1_scale,
    )
    weights = MoEWeightPack()
    weights.prepare_for("sm12x_nvfp4", native)
    act = MoEActivationPack(x, None, ids, route_weights)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=top_k),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=intermediate, local_num_experts=experts),
        activation=SwiGLU(),
        backend=BackendOptions(candidates=(SM12xNvfp4Config(),)),
        execution=ExecutionConfig(enable_pdl=False),
    )
    got = MoELayer(config, device=x.device)(act, weights)

    offsets, token_map, pair_scales, q0, sf0 = nvfp4_q0_route_triton(
        x, ids, route_weights, experts, q0_scale, enable_pdl=False
    )
    q1, sf1 = cute_dsl_sm12x_fc1_act_q1_nvfp4(
        q0,
        sf0,
        native["w1_weight"],
        native["w1_weight_sf"],
        offsets,
        native["w1_up_scale"],
        native["w1_gate_scale"],
        q1_scale,
        tune=False,
    )
    ref = cute_dsl_sm12x_fc2_finalize_nvfp4(
        q1,
        sf1,
        native["w2_weight"],
        native["w2_weight_sf"],
        offsets,
        token_map,
        pair_scales,
        w2_alpha,
        tokens,
        tune=False,
    )
    torch.testing.assert_close(got, ref, rtol=0, atol=0.0625)

    for bad_id in (-1, experts):
        ids[0, 0] = bad_id
        with pytest.raises(ValueError, match="0 <= id"):
            MoELayer(config, device=x.device)(act, weights)
    ids[0, 0] = 0

    runner = object.__new__(SM12xNvfp4Runner)
    runner._topk_validation_receipt = None
    runner._validate_expert_id_range(ids, experts)
    graph = torch.cuda.CUDAGraph()
    captured_ids = torch.empty_like(ids)
    with torch.cuda.graph(graph):
        runner._validate_expert_id_range(ids, experts)
        captured_ids.copy_(ids)
    graph.replay()
    assert torch.equal(captured_ids, ids)

    pdl_config = MoEConfig(
        routing=config.routing,
        quant=config.quant,
        experts=config.experts,
        activation=config.activation,
        backend=config.backend,
        execution=ExecutionConfig(),
    )
    with pytest.raises(NotImplementedError, match="does not support PDL"):
        SM12xNvfp4Runner(pdl_config, x.device)._check_support()
