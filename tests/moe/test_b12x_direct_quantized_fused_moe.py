"""Accuracy and CUDA Graph tests for experimental B12x Direct FP4 MoE."""

from __future__ import annotations

import pytest
import torch

from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fp4_quantization import fp4_quantize
from flashinfer.fused_moe.cute_dsl import B12xMoEWrapper
from flashinfer.fused_moe.b12x_direct_quantized import (
    prepare_b12x_direct_w4a16_scales,
    b12x_direct_nvfp4_fused_moe,
    b12x_direct_nvfp4_fused_moe_workspace,
    b12x_direct_w4a16_fused_moe,
    b12x_direct_w4a16_fused_moe_workspace,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="B12x Direct quantized MoE requires SM120",
)


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1(value: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(value)
    magnitude = value.abs()
    result = torch.zeros_like(magnitude)
    result[(magnitude >= 0.25) & (magnitude < 0.75)] = 0.5
    result[(magnitude >= 0.75) & (magnitude <= 1.25)] = 1.0
    result[(magnitude > 1.25) & (magnitude < 1.75)] = 1.5
    result[(magnitude >= 1.75) & (magnitude <= 2.5)] = 2.0
    result[(magnitude > 2.5) & (magnitude < 3.5)] = 3.0
    result[(magnitude >= 3.5) & (magnitude <= 5.0)] = 4.0
    result[magnitude > 5.0] = 6.0
    return result * sign


def _quant_dequant_activation(
    value: torch.Tensor,
    *,
    global_encode_scale: float = 448.0,
    bf16_dequant_scale: bool,
) -> torch.Tensor:
    grouped = value.float().reshape(*value.shape[:-1], -1, 16)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    encoded = (global_encode_scale * amax / 6.0).to(torch.float8_e4m3fn).float()
    dequant = encoded / global_encode_scale
    if bf16_dequant_scale:
        dequant = dequant.to(torch.bfloat16).float()
    scaled = torch.where(dequant == 0, torch.zeros_like(grouped), grouped / dequant)
    return (_round_e2m1(scaled) * dequant).reshape_as(value)


def _dequant_weight(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    values = torch.tensor(
        _E2M1_VALUES + tuple(-value for value in _E2M1_VALUES),
        dtype=torch.float32,
        device=packed.device,
    )
    raw = packed.view(torch.uint8)
    unpacked = torch.empty(
        (*raw.shape[:-1], raw.shape[-1] * 2), dtype=torch.uint8, device=raw.device
    )
    unpacked[..., 0::2] = raw & 0x0F
    unpacked[..., 1::2] = (raw >> 4) & 0x0F
    return values[unpacked.long()] * scales.float().repeat_interleave(16, dim=-1)


def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = weight.shape
    packed, scales = fp4_quantize(
        weight.reshape(-1, shape[-1]),
        torch.ones(1, dtype=torch.float32, device=weight.device),
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    packed = packed.reshape(*shape[:-1], shape[-1] // 2).contiguous()
    scales = (
        scales.view(torch.float8_e4m3fn)
        .reshape(*shape[:-1], shape[-1] // 16)
        .to(torch.bfloat16)
        .contiguous()
    )
    return packed, scales


def _make_case(num_tokens: int):
    torch.manual_seed(2026 + num_tokens)
    hidden, intermediate, experts, topk = 512, 256, 4, 2
    device = torch.device("cuda")
    x = (
        torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device) * 0.1
    ).contiguous()
    w1, s1 = _quantize_weight(
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    )
    w2, s2 = _quantize_weight(
        torch.randn(
            experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    )
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(num_tokens)]
    ).to(torch.int32)
    route_weights = torch.softmax(
        torch.randn(num_tokens, topk, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    return x, ids, route_weights, w1, s1, w2, s2


def _quantize_modelopt_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = weight.shape
    rows = shape[-2]
    experts = shape[0]
    packed_flat, swizzled = fp4_quantize(
        weight.reshape(-1, shape[-1]),
        torch.ones(1, dtype=torch.float32, device=weight.device),
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )
    packed = packed_flat.reshape(*shape[:-1], shape[-1] // 2).contiguous()
    mma_scales = convert_sf_to_mma_layout(
        swizzled,
        m=rows,
        k=shape[-1],
        num_groups=experts,
        sf_vec_size=16,
    )
    alphas = torch.ones(experts, dtype=torch.float32, device=weight.device)
    direct_scales = prepare_b12x_direct_w4a16_scales(
        mma_scales,
        alphas,
        rows=rows,
        cols=shape[-1],
    )
    return packed, mma_scales, direct_scales


def _make_target_case(num_tokens: int, intermediate: int = 512):
    torch.manual_seed(1701)
    hidden, experts, topk = 2048, 64, 8
    device = torch.device("cuda")
    x = (
        torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device) * 0.1
    ).contiguous()
    w1, w1_sf, w1_direct_sf = _quantize_modelopt_weight(
        torch.randn(
            experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    )
    w2, w2_sf, w2_direct_sf = _quantize_modelopt_weight(
        torch.randn(
            experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.02
    )
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(num_tokens)]
    ).to(torch.int32)
    route_weights = torch.softmax(
        torch.randn(num_tokens, topk, dtype=torch.float32, device=device), dim=-1
    ).contiguous()
    return (
        x,
        ids,
        route_weights,
        w1,
        w1_sf,
        w1_direct_sf,
        w2,
        w2_sf,
        w2_direct_sf,
    )


def _reference(case, *, quantize_activations: bool) -> torch.Tensor:
    x, ids, route_weights, w1, s1, w2, s2 = case
    hidden = x.float()
    if quantize_activations:
        hidden = _quant_dequant_activation(hidden, bf16_dequant_scale=False)
    w1_dequant = _dequant_weight(w1, s1)
    w2_dequant = _dequant_weight(w2, s2)
    intermediate_size = w2.shape[2] * 2
    output = torch.zeros_like(hidden)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            projection = torch.mv(w1_dequant[expert], hidden[token])
            activated = (
                torch.nn.functional.silu(projection[intermediate_size:])
                * projection[:intermediate_size]
            )
            if quantize_activations:
                activated = _quant_dequant_activation(
                    activated, bf16_dequant_scale=True
                )
            else:
                activated = activated.to(torch.bfloat16).float()
            output[token] += (
                torch.mv(w2_dequant[expert], activated) * route_weights[token, slot]
            )
    return output.to(torch.bfloat16)


@pytest.mark.parametrize("num_tokens", [1, 2, 4])
def test_b12x_direct_w4a16_accuracy(num_tokens: int):
    case = _make_case(num_tokens)
    x, ids, route_weights, w1, s1, w2, s2 = case
    actual = b12x_direct_w4a16_fused_moe(x, ids, route_weights, w1, s1, w2, s2)
    expected = _reference(case, quantize_activations=False)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 2, 4])
def test_b12x_direct_nvfp4_accuracy(num_tokens: int):
    case = _make_case(num_tokens)
    x, ids, route_weights, w1, s1, w2, s2 = case
    actual = b12x_direct_nvfp4_fused_moe(
        x,
        ids,
        route_weights,
        w1,
        s1,
        w2,
        s2,
        outputs_per_warp=2,
        num_threads=256,
    )
    expected = _reference(case, quantize_activations=True)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=5e-2)


@pytest.mark.parametrize("mode", ["w4a16", "nvfp4"])
def test_b12x_direct_quantized_cuda_graph(mode: str):
    case = _make_case(2)
    x, ids, route_weights, w1, s1, w2, s2 = case
    output = torch.empty_like(x)
    if mode == "w4a16":
        workspace = b12x_direct_w4a16_fused_moe_workspace(2, 2, 256, device=x.device)

        def run():
            return b12x_direct_w4a16_fused_moe(
                x,
                ids,
                route_weights,
                w1,
                s1,
                w2,
                s2,
                output=output,
                workspace=workspace,
            )

    else:
        workspace = b12x_direct_nvfp4_fused_moe_workspace(
            2, 2, 512, 256, device=x.device
        )

        def run():
            return b12x_direct_nvfp4_fused_moe(
                x,
                ids,
                route_weights,
                w1,
                s1,
                w2,
                s2,
                output=output,
                workspace=workspace,
            )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(case, quantize_activations=mode == "nvfp4")
    if mode == "nvfp4":
        torch.testing.assert_close(captured, expected, atol=1e-4, rtol=5e-2)
    else:
        torch.testing.assert_close(captured, expected, atol=3e-5, rtol=3e-2)


@pytest.mark.parametrize(
    ("mode", "num_tokens"),
    [("w4a16", 3), ("w4a16", 5), ("nvfp4", 3)],
)
def test_b12x_direct_target_cuda_graph_dynamic_routes(mode: str, num_tokens: int):
    """Exercise Direct tensor-core/hybrid dispatch with graph-time route data."""
    (
        x,
        ids,
        route_weights,
        w1,
        w1_sf,
        w1_direct_sf,
        w2,
        w2_sf,
        w2_direct_sf,
    ) = _make_target_case(num_tokens)
    output = torch.empty_like(x)
    topk = int(ids.shape[1])
    alphas = torch.ones(w1.shape[0], dtype=torch.float32, device=x.device)
    reference = B12xMoEWrapper(
        num_experts=int(w1.shape[0]),
        top_k=topk,
        hidden_size=int(x.shape[1]),
        intermediate_size=int(w2.shape[2]) * 2,
        quant_mode=mode,
        use_cuda_graph=True,
        max_num_tokens=num_tokens,
    )

    def run_reference() -> torch.Tensor:
        return reference.run(
            x=x,
            w1_weight=w1,
            w1_weight_sf=w1_sf,
            w1_alpha=alphas,
            fc2_input_scale=torch.ones(1, dtype=torch.float32, device=x.device),
            w2_weight=w2,
            w2_weight_sf=w2_sf,
            w2_alpha=alphas,
            token_selected_experts=ids,
            token_final_scales=route_weights,
        )

    original_reference = run_reference().clone()
    ids.copy_(torch.roll(ids, shifts=1, dims=1))
    changed_reference = run_reference().clone()
    ids.copy_(torch.roll(ids, shifts=-1, dims=1))

    if mode == "w4a16":
        workspace = b12x_direct_w4a16_fused_moe_workspace(
            num_tokens, topk, int(w2.shape[2]) * 2, device=x.device
        )

        def run():
            return b12x_direct_w4a16_fused_moe(
                x,
                ids,
                route_weights,
                w1,
                w1_direct_sf,
                w2,
                w2_direct_sf,
                output=output,
                workspace=workspace,
            )

    else:
        workspace = b12x_direct_nvfp4_fused_moe_workspace(
            num_tokens,
            topk,
            int(x.shape[1]),
            int(w2.shape[2]) * 2,
            device=x.device,
        )

        def run():
            return b12x_direct_nvfp4_fused_moe(
                x,
                ids,
                route_weights,
                w1,
                w1_direct_sf,
                w2,
                w2_direct_sf,
                output=output,
                workspace=workspace,
            )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    original_output = output.clone()
    ids.copy_(torch.roll(ids, shifts=1, dims=1))
    graph.replay()
    torch.cuda.synchronize()
    changed_output = output.clone()
    assert not torch.equal(original_output, changed_output)
    if mode == "w4a16":
        torch.testing.assert_close(
            original_output, original_reference, atol=3e-5, rtol=3e-2
        )
        torch.testing.assert_close(
            changed_output, changed_reference, atol=3e-5, rtol=3e-2
        )
    else:
        torch.testing.assert_close(
            original_output, original_reference, atol=2e-3, rtol=6e-2
        )
        torch.testing.assert_close(
            changed_output, changed_reference, atol=2e-3, rtol=6e-2
        )


@pytest.mark.parametrize(
    ("mode", "num_tokens"),
    [("w4a16", 3), ("w4a16", 5), ("nvfp4", 3)],
)
def test_b12x_direct_target_cuda_graph_sanitizer_smoke(mode: str, num_tokens: int):
    """Direct-only graph smoke test for compute-sanitizer tools."""
    (
        x,
        ids,
        route_weights,
        w1,
        _w1_sf,
        w1_direct_sf,
        w2,
        _w2_sf,
        w2_direct_sf,
    ) = _make_target_case(num_tokens)
    output = torch.empty_like(x)
    topk = int(ids.shape[1])
    if mode == "w4a16":
        workspace = b12x_direct_w4a16_fused_moe_workspace(
            num_tokens, topk, int(w2.shape[2]) * 2, device=x.device
        )

        def run():
            return b12x_direct_w4a16_fused_moe(
                x,
                ids,
                route_weights,
                w1,
                w1_direct_sf,
                w2,
                w2_direct_sf,
                output=output,
                workspace=workspace,
            )

    else:
        workspace = b12x_direct_nvfp4_fused_moe_workspace(
            num_tokens,
            topk,
            int(x.shape[1]),
            int(w2.shape[2]) * 2,
            device=x.device,
        )

        def run():
            return b12x_direct_nvfp4_fused_moe(
                x,
                ids,
                route_weights,
                w1,
                w1_direct_sf,
                w2,
                w2_direct_sf,
                output=output,
                workspace=workspace,
            )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    original = output.clone()
    ids.copy_(torch.roll(ids, shifts=1, dims=1))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert not torch.equal(original, output)
