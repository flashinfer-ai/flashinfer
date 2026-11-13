"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""References and graph contracts for routed NVFP4 preprocessing."""

import pytest
import torch

from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe.cute_dsl.moe_utils import moe_permute, moe_sort
from flashinfer.fused_moe.cute_dsl.fused_moe import (
    CuteDslMoEWrapper,
    cute_dsl_fused_moe,
)
from flashinfer.fused_moe.cute_dsl.tuner import (
    CuteDslFusedMoERunner,
    DEFAULT_BLACKWELL_MOE_TACTIC,
)
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import get_compute_capability
from tests.moe.pre_fc2_reference_quantizer import quantize_reference as _quantize


@pytest.fixture(autouse=True)
def _standard_fp4_recipe(monkeypatch):
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "0")


def _mapping(tokens, topk, device):
    routes = torch.arange(tokens * topk, device=device)
    live = routes[routes % 3 != 1]
    rows = torch.arange(live.numel(), device=device) * 2 + 7
    padded = ((tokens * topk * 2 + 17 + 127) // 128) * 128
    inverse = torch.full((tokens * topk,), -1, device=device, dtype=torch.int32)
    inverse[live] = rows.to(torch.int32)
    return inverse.view(tokens, topk), live, rows, padded


def _linear_scales(sf, rows, width):
    mma = convert_sf_to_mma_layout(sf, sf.shape[0], width)
    blocks = torch.arange(width // 16, device=sf.device)[None, :]
    rows = rows[:, None]
    return mma[
        rows % 32,
        rows % 128 // 32,
        rows // 128,
        blocks % 4,
        blocks // 4,
        0,
    ]


@pytest.fixture
def gpu():
    device = torch.device("cuda")
    if not torch.cuda.is_available() or get_compute_capability(device) not in [
        (10, 0),
        (10, 3),
    ]:
        pytest.skip("Requires SM100/SM103")
    return device


@pytest.mark.parametrize("tokens,width", [(1, 128), (17, 768), (129, 2048)])
@pytest.mark.parametrize("sf_dtype", [torch.uint8, torch.float8_e4m3fn])
def test_permutation_preserves_packed_data_scales_and_replays(
    tokens, width, sf_dtype, gpu
):
    torch.manual_seed(33)
    ids = (torch.arange(tokens * 2, device=gpu).reshape(tokens, 2) % 8).int()
    weights = torch.ones((tokens, 2), device=gpu)

    def sort():
        return moe_sort(
            ids,
            weights,
            8,
            2,
            local_expert_offset=2,
            num_local_experts=3,
            tile_tokens_dim=128,
        )

    tiles, limits, inverse, mapping, _, active = sort()
    live = (inverse.flatten() >= 0).nonzero().flatten()
    padded = mapping.numel()
    x = torch.randint(0, 256, (tokens, width // 2), device=gpu, dtype=torch.uint8)
    sf = torch.randint(
        1, 120, (tokens, width // 16), device=gpu, dtype=torch.uint8
    ).view(sf_dtype)
    out = torch.empty((padded, width // 2), device=gpu, dtype=torch.uint8)
    out_sf = torch.empty((padded, width // 16), device=gpu, dtype=torch.uint8)

    def run():
        out.fill_(165)
        out_sf.fill_(165)
        _, new_limits, new_inverse, new_mapping, _, new_active = sort()
        inverse.copy_(new_inverse)
        moe_permute(
            x,
            out,
            new_limits,
            new_mapping,
            new_active,
            padded,
            2,
            128,
            input_sf=sf,
            permuted_sf=out_sf,
        )

    def check():
        current_rows = inverse.flatten()[live].long()
        torch.testing.assert_close(out[current_rows], x[live // 2], rtol=0, atol=0)
        torch.testing.assert_close(
            _linear_scales(out_sf, current_rows, width),
            sf.view(torch.uint8)[live // 2],
            rtol=0,
            atol=0,
        )
        untouched = torch.ones(padded, device=gpu, dtype=torch.bool)
        untouched[current_rows] = False
        assert (out[untouched] == 165).all()
        assert (
            _linear_scales(out_sf, torch.arange(padded, device=gpu)[untouched], width)
            == 165
        ).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.bitwise_xor_(0x88)
    sf.view(torch.uint8).bitwise_xor_(0x40)
    ids.copy_(ids.flip(0))
    graph.replay()
    live = (inverse.flatten() >= 0).nonzero().flatten()
    check()


# 513 tokens x top-2 exceeds the 1024-route threshold for one program per route.
@pytest.mark.parametrize(
    "tokens,width", [(1, 128), (17, 512), (129, 4096), (513, 4096)]
)
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_weighting_precedes_quantization_with_bf16_rounding(
    tokens, width, weight_dtype, gpu
):
    torch.manual_seed(74)
    inverse, live, rows, padded = _mapping(tokens, 2, gpu)
    x = (torch.randn(tokens * 2, width, device=gpu) * 0.35).bfloat16()
    weights = (
        torch.linspace(0.0, 1.75, tokens * 2, device=gpu)
        .view(tokens, 2)
        .to(weight_dtype)
    )
    scale = torch.tensor([32.0], device=gpu)
    clamp = torch.tensor([16.0], device=gpu)
    out = torch.empty((padded, width // 2), device=gpu, dtype=torch.uint8)
    out_sf = torch.empty((padded, width // 16), device=gpu, dtype=torch.uint8)

    def run():
        out.fill_(165)
        out_sf.fill_(165)
        return _quantize(
            x, scale, weights, inverse, torch.zeros_like(inverse), 0, clamp, out, out_sf
        )

    def check():
        current_rows = inverse.flatten()[live].long()
        expected = (16.0 * torch.tanh(x[live].float() / 16.0)).bfloat16()
        expected = expected.float().relu().square().bfloat16()
        expected = (expected.float() * weights.flatten()[live, None].float()).bfloat16()
        reference_q, reference_sf = fp4_quantize(
            expected, scale, is_sf_swizzled_layout=False
        )
        actual_sf = _linear_scales(out_sf, current_rows, width)
        assert (actual_sf != reference_sf.view(torch.uint8)).float().mean() < 0.002
        assert (out[current_rows] != reference_q).float().mean() < 0.002
        untouched = torch.ones(padded, device=gpu, dtype=torch.bool)
        untouched[current_rows] = False
        assert (out[untouched] == 165).all()
        assert (
            _linear_scales(out_sf, torch.arange(padded, device=gpu)[untouched], width)
            == 165
        ).all()

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    x.neg_()
    weights.copy_(weights.flip(1))
    inverse.copy_(inverse.flip(0))
    live = (inverse.flatten() >= 0).nonzero().flatten()
    graph.replay()
    check()


@pytest.mark.parametrize(
    "invalid",
    ["activation", "clamp", "output", "w4a8", "w4a16", "swap_ab", "tile_size"],
)
def test_invalid_pre_fc2_configuration_rejected_without_cuda(invalid):
    kwargs = dict(
        num_experts=8,
        top_k=2,
        hidden_size=128,
        intermediate_size=128,
        device="cpu",
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([16.0]),
        routing_weights_before_fc2=True,
    )
    if invalid == "activation":
        kwargs.update(
            activation_type=ActivationType.Swiglu.value, clamped_relu2_limit=None
        )
    elif invalid == "clamp":
        kwargs["clamped_relu2_limit"] = None
    elif invalid == "output":
        kwargs["output_dtype"] = torch.float16
    elif invalid == "swap_ab":
        kwargs["swap_ab"] = True
    elif invalid == "tile_size":
        kwargs["tile_size"] = 256
    else:
        kwargs["quant_mode"] = invalid
    with pytest.raises(ValueError):
        CuteDslMoEWrapper(**kwargs)


def test_pre_fc2_runner_cache_and_tactic_isolation():
    def make(enabled):
        return CuteDslFusedMoERunner(
            lambda **kw: kw,
            num_experts=8,
            top_k=2,
            num_local_experts=3,
            activation_type=ActivationType.ClampedRelu2.value,
            clamped_relu2_limit=torch.tensor([16.0]),
            routing_weights_before_fc2=enabled,
        )

    legacy, weighted = make(False), make(True)
    assert hash(legacy) != hash(weighted)
    assert legacy.get_cache_key_extras([]) != weighted.get_cache_key_extras([])
    with pytest.raises(ValueError, match="128x128"):
        weighted.forward(
            [], tactic=(256, ((256, 128), (2, 1), False), ((256, 128), (2, 1), False))
        )


def test_permute_rejects_one_sided_fp4_scales_without_cuda():
    with pytest.raises(ValueError, match="both input_sf and permuted_sf"):
        moe_permute(
            torch.empty((1, 64), dtype=torch.uint8),
            torch.empty((128, 64), dtype=torch.uint8),
            torch.empty(1, dtype=torch.int32),
            torch.empty(128, dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
            128,
            2,
            128,
            input_sf=torch.empty((1, 8), dtype=torch.uint8),
        )


@pytest.mark.parametrize("tokens,width", [(17, 768), (129, 2048)])
def test_fp4_permutation_without_scales_moves_only_values(tokens, width, gpu):
    """Callers that permute scales themselves (e.g. cuDNN) pass no scale buffers."""
    ids = (torch.arange(tokens * 2, device=gpu).reshape(tokens, 2) % 8).int()
    tiles, limits, inverse, mapping, _, active = moe_sort(
        ids,
        torch.ones((tokens, 2), device=gpu),
        8,
        2,
        local_expert_offset=2,
        num_local_experts=3,
        tile_tokens_dim=128,
    )
    live = (inverse.flatten() >= 0).nonzero().flatten()
    rows = inverse.flatten()[live].long()
    padded = mapping.numel()
    x = torch.randint(0, 256, (tokens, width // 2), device=gpu, dtype=torch.uint8)
    out = torch.full((padded, width // 2), 165, device=gpu, dtype=torch.uint8)
    moe_permute(x, out, limits, mapping, active, padded, 2, 128)
    torch.testing.assert_close(out[rows], x[live // 2], rtol=0, atol=0)
    untouched = torch.ones(padded, device=gpu, dtype=torch.bool)
    untouched[rows] = False
    assert (out[untouched] == 165).all()


def _decode(q, sf):
    """Independent packed-E2M1 decode with row-major block scales."""
    table = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=q.device)
    nibble = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).long()
    values = table[nibble & 7] * torch.where(nibble < 8, 1.0, -1.0)
    return values * sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)


def _case(tokens, device, per_expert_scale, hidden=128, intermediate=256, top_k=2):
    torch.manual_seed(53)
    local = 3
    x_bf16 = (torch.randn(tokens, hidden, device=device) * 0.2).bfloat16()
    w1_bf16 = (torch.randn(local, intermediate, hidden, device=device) * 0.1).bfloat16()
    w2_bf16 = (torch.randn(local, hidden, intermediate, device=device) * 0.1).bfloat16()
    one = torch.ones(1, device=device)
    x, x_sf = fp4_quantize(x_bf16, one, is_sf_swizzled_layout=False)

    def weight(tensor):
        experts, rows, width = tensor.shape
        q, sf = fp4_quantize(tensor.flatten(0, 1), one, is_sf_swizzled_layout=False)
        _, swizzled = fp4_quantize(
            tensor.flatten(0, 1), one, is_sf_swizzled_layout=True
        )
        mma = convert_sf_to_mma_layout(swizzled, rows, width, num_groups=experts)
        return (
            q.reshape(experts, rows, width // 2),
            mma,
            _decode(q, sf).reshape(experts, rows, width),
        )

    w1, sf1, ref1 = weight(w1_bf16)
    w2, sf2, ref2 = weight(w2_bf16)
    ids = (
        (torch.arange(tokens * top_k, device=device).reshape(tokens, top_k) + 2) % 8
    ).int()
    weights = torch.linspace(0.05, 0.8, tokens * top_k, device=device).reshape(
        tokens, top_k
    )
    scale2 = torch.tensor(
        [16.0, 64.0, 128.0] if per_expert_scale else [32.0], device=device
    )
    alpha1 = torch.tensor([0.8, 1.0, 1.2], device=device)
    alpha2 = torch.tensor([1.2, 1.0, 0.8], device=device) / scale2
    inputs = dict(
        x=x,
        x_sf=x_sf,
        token_selected_experts=ids,
        token_final_scales=weights,
        w1_weight=w1,
        w1_weight_sf=sf1,
        w1_alpha=alpha1,
        fc2_input_scale=scale2,
        w2_weight=w2,
        w2_weight_sf=sf2,
        w2_alpha=alpha2,
    )
    return inputs, (_decode(x, x_sf), ref1, ref2)


def _reference(inputs, decoded, clamp, fused):
    x, w1, w2 = decoded
    result = torch.zeros_like(x, dtype=torch.bfloat16 if fused else torch.float32)
    ids, weights = inputs["token_selected_experts"], inputs["token_final_scales"]
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot]) - 2
            if not 0 <= expert < 3:
                continue
            raw = (
                x[token : token + 1] @ w1[expert].T * inputs["w1_alpha"][expert]
            ).bfloat16()
            act = (clamp * torch.tanh(raw.float() / clamp)).bfloat16()
            act = act.float().relu().square().bfloat16()
            act = (act.float() * weights[token, slot]).bfloat16()
            scale = inputs["fc2_input_scale"]
            scale = scale[expert : expert + 1] if scale.numel() > 1 else scale
            q, sf = fp4_quantize(act, scale, is_sf_swizzled_layout=False)
            route = (
                _decode(q, sf) @ w2[expert].T * inputs["w2_alpha"][expert]
            ).bfloat16()
            result[token] += route[0]
    return result.bfloat16()


@pytest.mark.parametrize(
    "tokens,hidden,intermediate",
    [(1, 128, 256), (17, 128, 256), (129, 128, 256), (17, 768, 4096)],
)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("per_expert_scale", [False, True])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_existing_moe_apis_preserve_weighted_quantization_and_replay(
    tokens, hidden, intermediate, fused, per_expert_scale, weight_dtype, gpu
):
    inputs, decoded = _case(tokens, gpu, per_expert_scale, hidden, intermediate)
    inputs["token_final_scales"] = inputs["token_final_scales"].to(weight_dtype)
    options = dict(
        num_experts=8,
        top_k=2,
        num_local_experts=3,
        local_expert_offset=2,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([0.15], device=gpu),
        use_fused_finalize=fused,
        routing_weights_before_fc2=True,
        enable_pdl=False,
    )
    wrapper = CuteDslMoEWrapper(
        hidden_size=hidden,
        intermediate_size=intermediate,
        use_cuda_graph=True,
        **options,
    )
    actual = wrapper.run(**inputs, tactic=DEFAULT_BLACKWELL_MOE_TACTIC)
    functional = cute_dsl_fused_moe(
        **inputs, **options, tactic=DEFAULT_BLACKWELL_MOE_TACTIC
    )
    expected = _reference(inputs, decoded, options["clamped_relu2_limit"], fused)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=2e-4)
    torch.testing.assert_close(functional, actual, rtol=0.02, atol=2e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(**inputs, tactic=DEFAULT_BLACKWELL_MOE_TACTIC)
    inputs["token_selected_experts"].copy_(inputs["token_selected_experts"].flip(0))
    inputs["token_final_scales"].mul_(0.37)
    options["clamped_relu2_limit"].mul_(0.5)
    inputs["x"].bitwise_xor_(0x88)
    decoded = (_decode(inputs["x"], inputs["x_sf"]), decoded[1], decoded[2])
    graph.replay()
    torch.testing.assert_close(
        captured,
        _reference(inputs, decoded, options["clamped_relu2_limit"], fused),
        rtol=0.03,
        atol=2e-4,
    )
    # A rank with no local routes must produce zeros without reading unwritten FC1 rows.
    inputs["token_selected_experts"].zero_()
    graph.replay()
    assert torch.count_nonzero(captured) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
def test_existing_permute_non_fp4_dtypes(dtype, gpu):
    ids = (torch.arange(34, device=gpu).reshape(17, 2) % 8).int()
    weights = torch.ones((17, 2), device=gpu)
    _, limits, inverse, mapping, _, active = moe_sort(
        ids,
        weights,
        8,
        2,
        local_expert_offset=2,
        num_local_experts=3,
        tile_tokens_dim=128,
    )
    x = torch.randn((17, 128), device=gpu).to(dtype)
    out = torch.full((mapping.numel(), 128), 0.5, device=gpu, dtype=dtype)
    moe_permute(x, out, limits, mapping, active, mapping.numel(), 2, 128)
    live = (inverse.flatten() >= 0).nonzero().flatten()
    rows = inverse.flatten()[live].long()
    torch.testing.assert_close(out.float()[rows], x.float()[live // 2], rtol=0, atol=0)
    untouched = torch.ones(out.shape[0], device=gpu, dtype=torch.bool)
    untouched[rows] = False
    assert (out.float()[untouched] == 0.5).all()


def test_pre_fc2_empty_batch(gpu):
    inputs, _ = _case(1, gpu, False)
    for name in ("x", "x_sf", "token_selected_experts", "token_final_scales"):
        inputs[name] = inputs[name][:0]
    opts = dict(
        num_experts=8,
        top_k=2,
        num_local_experts=3,
        local_expert_offset=2,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([0.15], device=gpu),
        routing_weights_before_fc2=True,
    )
    wrapper = CuteDslMoEWrapper(hidden_size=128, intermediate_size=256, **opts)
    assert wrapper.run(**inputs, tactic=DEFAULT_BLACKWELL_MOE_TACTIC).shape == (0, 128)
    assert cute_dsl_fused_moe(
        **inputs, **opts, tactic=DEFAULT_BLACKWELL_MOE_TACTIC
    ).shape == (0, 128)


def _host_pre_fc2_inputs():
    return dict(
        x=torch.empty((2, 64), dtype=torch.uint8),
        x_sf=torch.empty((2, 8), dtype=torch.uint8),
        token_selected_experts=torch.zeros((2, 1), dtype=torch.int32),
        token_final_scales=torch.full((2, 1), 0.5),
        w1_weight=torch.empty((1, 128, 64), dtype=torch.uint8),
        w1_weight_sf=torch.empty(128 * 128 // 16, dtype=torch.uint8),
        w1_alpha=torch.ones(1),
        fc2_input_scale=torch.ones(1),
        w2_weight=torch.empty((1, 128, 64), dtype=torch.uint8),
        w2_weight_sf=torch.empty(128 * 128 // 16, dtype=torch.uint8),
        w2_alpha=torch.ones(1),
    )


@pytest.mark.parametrize(
    "name,shape",
    [
        ("x", ()),
        ("x", (128,)),
        ("x", (1, 1, 64)),
        ("w1_weight", ()),
        ("w1_weight", (128,)),
        ("w1_weight", (128, 64)),
        ("w2_weight", ()),
        ("w2_weight", (128,)),
        ("w2_weight", (128, 64)),
    ],
)
def test_pre_fc2_rejects_invalid_rank_before_dimension_checks(name, shape, monkeypatch):
    from flashinfer.fused_moe.cute_dsl import _routed_nvfp4 as routed

    # Rank validation needs only tensor metadata, not a CUDA allocation.
    monkeypatch.setattr(routed, "get_compute_capability", lambda device: (10, 3))
    inputs = _host_pre_fc2_inputs()
    inputs[name] = torch.empty(shape, dtype=torch.uint8)
    with pytest.raises(ValueError, match="rank 2.*rank 3"):
        _validate_host_pre_fc2_inputs(routed, inputs)


def _validate_host_pre_fc2_inputs(routed, inputs):
    routed._validate_pre_fc2_inputs(
        *(
            inputs[name]
            for name in (
                "x",
                "x_sf",
                "token_selected_experts",
                "token_final_scales",
                "w1_weight",
                "w1_weight_sf",
                "w1_alpha",
                "w2_weight",
                "w2_weight_sf",
                "w2_alpha",
                "fc2_input_scale",
            )
        )
    )


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("w1_weight_sf", torch.empty(1, dtype=torch.uint8), ValueError),
        ("w2_weight_sf", torch.empty(1, dtype=torch.uint8), ValueError),
        ("w1_weight_sf", torch.empty(1024, dtype=torch.float32), TypeError),
        ("w2_weight_sf", None, TypeError),
        ("w1_alpha", torch.ones(2), ValueError),
        ("w2_alpha", torch.ones(1, dtype=torch.bfloat16), ValueError),
        ("w2_alpha", None, ValueError),
        ("w1_alpha", torch.ones(1, dtype=torch.float64), ValueError),
    ],
)
def test_pre_fc2_rejects_invalid_weight_scales_and_alphas(
    name, value, error, monkeypatch
):
    """GEMM operands read through raw pointers are validated up front."""
    from flashinfer.fused_moe.cute_dsl import _routed_nvfp4 as routed

    monkeypatch.setattr(routed, "get_compute_capability", lambda device: (10, 3))
    inputs = _host_pre_fc2_inputs()
    _validate_host_pre_fc2_inputs(routed, inputs)
    inputs[name] = value
    with pytest.raises(error, match=name):
        _validate_host_pre_fc2_inputs(routed, inputs)


@pytest.mark.parametrize("invalid", ["strided", "device", "dtype", "shape"])
def test_pre_fc2_output_rejected_before_routing(invalid, monkeypatch, gpu):
    import importlib

    api = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")

    def forbidden_sort(*args, **kwargs):
        raise AssertionError("Invalid output reached GPU routing")

    monkeypatch.setattr(api, "moe_sort", forbidden_sort)
    output = torch.empty((2, 128), dtype=torch.bfloat16, device=gpu)
    error, match = ValueError, "contiguous"
    if invalid == "strided":
        output = torch.empty((2, 256), dtype=torch.bfloat16, device=gpu)[:, ::2]
    elif invalid == "device":
        output = torch.empty((2, 128), dtype=torch.bfloat16, device="meta")
        match = "same device"
    elif invalid == "dtype":
        output = output.half()
        error, match = TypeError, "bfloat16"
    else:
        output = torch.empty((2, 256), dtype=torch.bfloat16, device=gpu)
        match = "shape"
    with pytest.raises(error, match=match):
        cute_dsl_fused_moe(
            **{k: v.to(gpu) for k, v in _host_pre_fc2_inputs().items()},
            num_experts=1,
            top_k=1,
            activation_type=ActivationType.ClampedRelu2.value,
            clamped_relu2_limit=torch.tensor([16.0], device=gpu),
            routing_weights_before_fc2=True,
            moe_output=output,
            tactic=DEFAULT_BLACKWELL_MOE_TACTIC,
        )


def test_pre_fc2_per_token_scales_rejected_before_routing(monkeypatch, gpu):
    import importlib

    api = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")
    inputs = {k: v.to(gpu) for k, v in _host_pre_fc2_inputs().items()}
    options = dict(
        num_experts=1,
        top_k=1,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([16.0], device=gpu),
        routing_weights_before_fc2=True,
    )
    wrapper = CuteDslMoEWrapper(hidden_size=128, intermediate_size=128, **options)

    def forbidden_sort(*args, **kwargs):
        raise AssertionError("Unsupported per-token scaling reached GPU routing")

    monkeypatch.setattr(api, "moe_sort", forbidden_sort)
    per_token_scale = torch.ones(2, device=gpu)
    with pytest.raises(ValueError, match="per_token_scale"):
        wrapper.run(**inputs, per_token_scale=per_token_scale)
    with pytest.raises(ValueError, match="per_token_scale"):
        cute_dsl_fused_moe(**inputs, **options, per_token_scale=per_token_scale)


@pytest.mark.parametrize("invalid", ["top_k", "local_count", "local_range"])
def test_pre_fc2_routing_metadata_rejected_before_sort(invalid, monkeypatch, gpu):
    import importlib

    api = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")
    inputs = {k: v.to(gpu) for k, v in _host_pre_fc2_inputs().items()}
    options = dict(
        num_experts=2,
        num_local_experts=1,
        top_k=1,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([16.0], device=gpu),
        routing_weights_before_fc2=True,
    )
    if invalid == "top_k":
        options["top_k"] = 2
    elif invalid == "local_count":
        options["num_local_experts"] = 2
    else:
        options["local_expert_offset"] = 2

    def forbidden_sort(*args, **kwargs):
        raise AssertionError("Incompatible expert metadata reached GPU routing")

    monkeypatch.setattr(api, "moe_sort", forbidden_sort)
    with pytest.raises(ValueError, match="top_k|num_local_experts|Local expert range"):
        cute_dsl_fused_moe(**inputs, **options, tactic=DEFAULT_BLACKWELL_MOE_TACTIC)


def test_pre_fc2_autotuning_preserves_weighted_reference(gpu):
    from flashinfer import autotune

    inputs, decoded = _case(17, gpu, True)
    clamp = torch.tensor([0.15], device=gpu)
    options = dict(
        num_experts=8,
        num_local_experts=3,
        local_expert_offset=2,
        top_k=2,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=clamp,
        routing_weights_before_fc2=True,
    )
    with autotune(True):
        result = cute_dsl_fused_moe(**inputs, **options)
    torch.testing.assert_close(
        result, _reference(inputs, decoded, clamp, True), rtol=0.03, atol=2e-4
    )


def test_pre_fc2_trace_reference_matches_weighted_experts(gpu):
    from flashinfer.trace.templates.moe import cute_dsl_fused_moe_trace

    inputs, decoded = _case(17, gpu, True)
    clamp = torch.tensor([0.15], device=gpu)
    actual = cute_dsl_fused_moe_trace.reference(
        **inputs,
        num_experts=8,
        top_k=2,
        local_expert_offset=2,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=clamp,
        routing_weights_before_fc2=True,
    )
    torch.testing.assert_close(
        actual, _reference(inputs, decoded, clamp, False), rtol=0.03, atol=2e-4
    )


@pytest.mark.parametrize("weight_name", ["w1_weight", "w2_weight"])
def test_pre_fc2_rejects_strided_weights_before_routing(weight_name, monkeypatch, gpu):
    """Raw-pointer grouped GEMM weights must retain their dense K-major layout."""
    import importlib

    api = importlib.import_module("flashinfer.fused_moe.cute_dsl.fused_moe")

    def forbidden_sort(*args, **kwargs):
        raise AssertionError("Invalid expert weights reached GPU routing")

    monkeypatch.setattr(api, "moe_sort", forbidden_sort)
    inputs, _ = _case(1, gpu, False)
    w = inputs[weight_name]
    storage = torch.empty((*w.shape[:-1], w.shape[-1] * 2), dtype=w.dtype, device=gpu)
    inputs[weight_name] = storage[..., ::2]
    with pytest.raises(ValueError, match="Expert weights must be contiguous"):
        cute_dsl_fused_moe(
            **inputs,
            num_experts=8,
            top_k=2,
            num_local_experts=3,
            local_expert_offset=2,
            activation_type=ActivationType.ClampedRelu2.value,
            clamped_relu2_limit=torch.tensor([16.0], device=gpu),
            routing_weights_before_fc2=True,
            tactic=DEFAULT_BLACKWELL_MOE_TACTIC,
        )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "contributions",
    [
        (256.0, 1.0, -256.0),
        (22.625, 0.015869140625, 0.921875),
        (33554432.0, 1.0, -33554432.0),
    ],
)
@pytest.mark.parametrize("slot_order", [(0, 1, 2), (0, 2, 1)])
def test_three_local_routes_distinguish_bf16_reduction(
    contributions, fused, slot_order, gpu
):
    """Accept every legal BF16 reduction order, with exact rounding-sensitive inputs."""
    import itertools
    from flashinfer.trace.templates.moe import cute_dsl_fused_moe_trace

    inputs, _ = _case(1, gpu, False, hidden=128, intermediate=128, top_k=3)
    inputs["x"].fill_(0x22)
    inputs["x_sf"].view(torch.uint8).fill_(56)
    inputs["w1_weight"].fill_(0x22)
    inputs["w1_weight_sf"].view(torch.uint8).fill_(56)
    inputs["w2_weight"].fill_(0x22)
    inputs["w2_weight_sf"].view(torch.uint8).fill_(56)
    for e, value in enumerate(contributions):
        if value < 0:
            inputs["w2_weight"][e].fill_(0xAA)
    inputs["w1_alpha"].fill_(2.46875 / 128)
    inputs["fc2_input_scale"].fill_(1)
    # The rounded activation quantizes to 6, so FC2 sums 128 * 6 before alpha.
    inputs["w2_alpha"].copy_(torch.tensor(contributions, device=gpu).abs() / (128 * 6))
    inputs["token_selected_experts"].copy_(
        torch.tensor([[e + 2 for e in slot_order]], device=gpu)
    )
    inputs["token_final_scales"].fill_(1)
    opts = dict(
        num_experts=8,
        top_k=3,
        num_local_experts=3,
        local_expert_offset=2,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([16.0], device=gpu),
        routing_weights_before_fc2=True,
        use_fused_finalize=fused,
    )
    actual = cute_dsl_fused_moe(**inputs, **opts, tactic=DEFAULT_BLACKWELL_MOE_TACTIC)
    values = torch.tensor(contributions, device=gpu, dtype=torch.bfloat16)
    exact = torch.zeros((), device=gpu, dtype=torch.float32)
    for slot in slot_order:
        exact += values[slot].float()
    exact = exact.bfloat16()
    allowed = []
    for order in itertools.permutations(range(3)):
        acc = torch.zeros((), device=gpu, dtype=torch.bfloat16)
        for i in order:
            acc += values[i]
        allowed.append(acc)
    allowed = torch.stack(allowed)
    if fused:
        assert (actual[..., None] == allowed).any(-1).all()
    else:
        assert (actual == exact).all()
    if all(v > 0 for v in contributions):
        assert (allowed != exact).all(), "Fixture must detect an FP32 accumulator"
    reference = cute_dsl_fused_moe_trace.reference(**inputs, **opts)
    assert cute_dsl_fused_moe_trace.check(reference, actual, rtol=0, atol=0)
    bound = getattr(
        reference, "_cute_dsl_bf16_reduction_bound", torch.zeros((), device=gpu)
    )
    invalid = actual.float() + 2 * bound.max().float() + 32
    assert not cute_dsl_fused_moe_trace.check(reference, invalid, rtol=0, atol=0)


_BITWISE_SMALL_CASES = [
    (tokens, hidden, intermediate, per_expert, weight_dtype, regime, clamp)
    for tokens, hidden, intermediate in [(17, 128, 256), (300, 768, 4096)]
    for per_expert in (False, True)
    for weight_dtype in (torch.float32, torch.bfloat16)
    for regime in ("small", "calibrated", "saturating")
    for clamp in (0.15, 4.0)
]


def _fc1_bitwise_case(
    tokens,
    hidden,
    intermediate,
    per_expert_scale,
    weight_dtype,
    regime,
    clamp_value,
    gpu,
    all_local=False,
):
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion as gather_gemm,
    )
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
        blockscaled_contiguous_grouped_gemm_finalize_fusion as grouped_gemm,
    )
    from flashinfer.fused_moe.cute_dsl.moe_utils import (
        PRE_FC2_CLUSTER_SHAPE,
        PRE_FC2_MMA_SHAPE,
    )

    inputs, _ = _case(tokens, gpu, per_expert_scale, hidden, intermediate)
    x, x_sf = inputs["x"], inputs["x_sf"]
    ids = inputs["token_selected_experts"]
    if all_local:
        ids.copy_(2 + torch.arange(ids.numel(), device=gpu).reshape(ids.shape) % 3)
    weights = inputs["token_final_scales"].clone()
    # A zero weight yields an all-zero block: scale 0 and no division.
    weights.view(-1)[0] = 0
    weights = weights.to(weight_dtype)
    clamp = torch.tensor([clamp_value], device=gpu)
    top_k = ids.shape[1]
    tiles, limits, inverse, mapping, _, active = moe_sort(
        token_selected_experts=ids,
        token_final_scales=weights,
        num_experts=8,
        top_k=top_k,
        local_expert_offset=2,
        num_local_experts=3,
        tile_tokens_dim=128,
        enable_pdl=False,
    )
    rows = mapping.numel()
    common = dict(
        tile_idx_to_expert_idx=tiles,
        num_non_exiting_tiles=active,
        tile_idx_to_mn_limit=limits,
        a_dtype="float4_e2m1fn",
        b_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        sf_vec_size=16,
        mma_tiler_mn=PRE_FC2_MMA_SHAPE,
        cluster_shape_mn=PRE_FC2_CLUSTER_SHAPE,
        enable_pdl=False,
    )

    # Reference: BF16 FC1 rows, then the standalone weighted quantizer.
    permuted = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=gpu)
    permuted_sf = torch.empty((rows, hidden // 16), dtype=torch.uint8, device=gpu)
    moe_permute(
        x,
        permuted,
        limits,
        mapping,
        active,
        rows,
        top_k,
        128,
        enable_pdl=False,
        input_sf=x_sf.view(tokens, hidden // 16),
        permuted_sf=permuted_sf,
    )
    raw = torch.empty((tokens * top_k, intermediate), dtype=torch.bfloat16, device=gpu)
    grouped_gemm(
        a=permuted,
        b=inputs["w1_weight"],
        a_scale=convert_sf_to_mma_layout(permuted_sf, rows, hidden),
        b_scale=inputs["w1_weight_sf"],
        alpha=inputs["w1_alpha"],
        permuted_idx_to_expanded_idx=mapping,
        token_final_scales=torch.ones_like(weights, dtype=torch.float32),
        out=raw,
        use_fused_finalize=False,
        **common,
    )
    scale2 = inputs["fc2_input_scale"]
    if regime != "small":
        # Calibrate the FC2 input scale so block scales reach the E4M3 maximum
        # (448); "saturating" overshoots it fourfold.
        local = inverse.reshape(-1) >= 0
        act = (clamp * torch.tanh(raw[local].float() / clamp)).relu().square()
        act = act * weights.reshape(-1)[local, None].float()
        calibrated = 448.0 * 6.0 / act.abs().max().item()
        if regime == "saturating":
            calibrated *= 4.0
        spread = [0.5, 1.0, 2.0] if per_expert_scale else [1.0]
        scale2 = torch.tensor([calibrated * f for f in spread], device=gpu)
    expected = torch.empty((rows, intermediate // 2), dtype=torch.uint8, device=gpu)
    expected_sf = torch.empty((rows, intermediate // 16), dtype=torch.uint8, device=gpu)
    _quantize(raw, scale2, weights, inverse, ids, 2, clamp, expected, expected_sf)
    del raw

    actual, actual_sf = gather_gemm(
        a=x,
        b=inputs["w1_weight"],
        a_scale=x_sf,
        b_scale=inputs["w1_weight_sf"],
        alpha=inputs["w1_alpha"],
        token_id_mapping=mapping,
        global_scale=scale2 if per_expert_scale else scale2.expand(3).contiguous(),
        topk=top_k,
        c_dtype="float4_e2m1fn",
        quantize_output=True,
        activation_type=ActivationType.ClampedRelu2.value,
        gated=False,
        clamped_relu2_limit=clamp,
        routing_weights=weights.reshape(-1).float(),
        **common,
    )
    live_tiles = int(active.item())
    valid = torch.cat(
        [torch.arange(t * 128, int(limits[t]), device=gpu) for t in range(live_tiles)]
    )
    assert valid.numel() == int((inverse >= 0).sum())
    expected_scales = _linear_scales(expected_sf, valid, intermediate)
    if regime == "saturating":
        # The fixture must reach the E4M3 maximum, 448 (code 0x7E).
        assert (expected_scales == 0x7E).any()
    q_mismatch = (actual[valid] != expected[valid]).float().mean().item()
    sf_mismatch = (
        (
            _linear_scales(
                actual_sf.view(torch.uint8).reshape(rows, intermediate // 16),
                valid,
                intermediate,
            )
            != expected_scales
        )
        .float()
        .mean()
        .item()
    )
    assert q_mismatch == 0 and sf_mismatch == 0, (q_mismatch, sf_mismatch)
    return rows


@pytest.mark.parametrize(
    "tokens,hidden,intermediate,per_expert_scale,weight_dtype,regime,clamp",
    _BITWISE_SMALL_CASES,
)
def test_fused_fc1_epilogue_matches_separate_quantizer_bitwise(
    tokens, hidden, intermediate, per_expert_scale, weight_dtype, regime, clamp, gpu
):
    """The gather GEMM1 epilogue reproduces BF16 FC1 output plus the quantizer."""
    _fc1_bitwise_case(
        tokens,
        hidden,
        intermediate,
        per_expert_scale,
        weight_dtype,
        regime,
        clamp,
        gpu,
    )


def test_fused_fc1_epilogue_addresses_beyond_signed_int32(gpu):
    """FP4 output rows past 2**31 bytes keep their bitwise reference."""
    intermediate = 8192
    rows = _fc1_bitwise_case(
        270_000,
        128,
        intermediate,
        True,
        torch.float32,
        "calibrated",
        0.15,
        gpu,
        all_local=True,
    )
    assert rows * intermediate // 2 > 2**31


@pytest.mark.parametrize("fused", [False, True])
def test_pre_fc2_accepts_single_value_fc2_scale_of_any_rank(fused, gpu):
    """A one-element FC2 input scale is broadcast regardless of its rank."""
    inputs, _ = _case(17, gpu, False)
    options = dict(
        num_experts=8,
        top_k=2,
        num_local_experts=3,
        local_expert_offset=2,
        activation_type=ActivationType.ClampedRelu2.value,
        clamped_relu2_limit=torch.tensor([0.15], device=gpu),
        use_fused_finalize=fused,
        routing_weights_before_fc2=True,
        enable_pdl=False,
    )
    flat = cute_dsl_fused_moe(**inputs, **options, tactic=DEFAULT_BLACKWELL_MOE_TACTIC)
    inputs["fc2_input_scale"] = inputs["fc2_input_scale"].reshape(1, 1)
    square = cute_dsl_fused_moe(
        **inputs, **options, tactic=DEFAULT_BLACKWELL_MOE_TACTIC
    )
    assert torch.equal(flat, square)
