"""
Copyright (c) 2025 by FlashInfer team.

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

from contextlib import nullcontext

import pytest
from flashinfer.fused_moe.core import ActivationType
import torch
from torch.nn import functional as F

import flashinfer.fused_moe as fused_moe
from flashinfer.utils import (
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm12x_supported,
)
from flashinfer import (
    autotune,
    fp4_quantize,
    mxfp4_dequantize,
    mxfp4_quantize,
    mxfp8_dequantize_host,
    mxfp8_quantize,
    mxfp4_dequantize_host,
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn


def dynamic_per_tensor_fp8_quant(x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    fp8_traits_max = FLOAT8_E4M3_MAX
    fp8_traits_min = -FLOAT8_E4M3_MAX
    fp8_max = torch.tensor(fp8_traits_max).float()
    one = torch.tensor(1.0).float()

    x_max = x.abs().max().float()
    scale = x_max / fp8_max
    iscale = one / scale
    out = (x.float() * iscale).clamp(fp8_traits_min, fp8_traits_max).to(FP8_DTYPE)
    return out, scale.view((1,))


def gen_tensor(shape, dtype, stype=None, scale=1.0):
    x = torch.randn(*shape, dtype=dtype).cuda() * scale
    return x.to(stype) if stype else x


def cast_to_representable(x):
    x_q, x_scale = dynamic_per_tensor_fp8_quant(x)
    x = x_q.to(x.dtype) * x_scale.to(x.dtype)
    return x


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0 : k // block_size]


def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1ToFloat = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def break_int4_bytes_to_int8(packed):
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    return torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1)


def dequantize_int4_to_dtype(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
    weight_scale_2: torch.Tensor = None,
) -> torch.Tensor:
    # unpack: [N, K//2] -> [N, K]
    unpacked = break_int4_bytes_to_int8(packed_weight)
    scale_expanded = weight_scale.repeat_interleave(group_size, dim=1)
    dequant = unpacked.float() * scale_expanded.float()
    if weight_scale_2 is not None:
        dequant = dequant / weight_scale_2.float()
    return dequant.to(dtype)


def compute_routing(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def torch_moe_nvfp4(a, w1, w2, topk, topk_weight, topk_ids, activation_type):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    # score = torch.softmax(score, dim=-1, dtype=torch.float32)
    # topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # w1 needs to be swapped in terms of gate and up_proj

    if activation_type == ActivationType.Swiglu:

        def act(weight, mask):
            m = weight.shape[0]
            assert m % 2 == 0
            w1_expert, w3_expert = weight[m // 2 :, :], weight[: m // 2, :]
            return F.silu(a[mask] @ w1_expert.t()) * (a[mask] @ w3_expert.t())

    elif activation_type == ActivationType.Relu2:

        def act(weight, mask):
            return F.relu(a[mask] @ weight.t()) ** 2

    else:
        raise ValueError(f"Unsupported activation type {activation_type}")

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter = act(w1[i], mask)
            inter_gs = torch.tensor(1.0).cuda()
            inter_q, inter_blockscale = fp4_quantize(inter, inter_gs)
            inter = dequantize_nvfp4_to_dtype(
                inter_q,
                inter_blockscale,
                inter_gs,
                dtype=inter.dtype,
                device=inter.device,
                block_size=16,
            ).cuda()
            out[mask] = inter @ w2[i].transpose(0, 1)
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def torch_moe_w4a8(
    num_experts,
    x,
    w31_weight,
    w2_weight,
    selected_experts,
    routing_weights,
    fc1_input_scale,
    fc2_input_scale,
    fc1_pre_quant_scale,
    fc2_pre_quant_scale,
    fc1_weight_scale_2,
    fc2_weight_scale_2,
):
    dtype = x.dtype
    results = torch.zeros_like(x)

    for expert_id in range(num_experts):
        mask = selected_experts == expert_id
        if not mask.sum():
            continue
        batch_idx, nth_expert = torch.where(mask)

        w31_expert = w31_weight[expert_id]  # [2N, K]
        w2_expert = w2_weight[expert_id]  # [K, N]
        w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)

        expert_inputs = x[batch_idx]
        if fc1_input_scale is not None:
            scale1 = fc1_input_scale[expert_id]

        if fc1_pre_quant_scale is not None:
            expert_inputs_scaled = expert_inputs * fc1_pre_quant_scale[expert_id]
        else:
            expert_inputs_scaled = expert_inputs
        inp_q = (
            torch.clamp(expert_inputs_scaled / scale1, -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(dtype)
        )
        x1 = (inp_q @ w1_expert.t()) * scale1
        x2 = (inp_q @ w3_expert.t()) * scale1
        if fc1_weight_scale_2 is not None:
            ws2 = fc1_weight_scale_2[expert_id]
            x1 = x1 * ws2.to(dtype)
            x2 = x2 * ws2.to(dtype)

        inter = F.silu(x1) * x2

        if fc2_input_scale is not None:
            scale2 = fc2_input_scale[expert_id]
        if fc2_pre_quant_scale is not None:
            inter_scaled = inter * fc2_pre_quant_scale[expert_id]
        else:
            inter_scaled = inter
        inter_q = (
            torch.clamp(inter_scaled / scale2, -448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .to(dtype)
        )
        output = (inter_q @ w2_expert.t()) * scale2

        if fc2_weight_scale_2 is not None:
            ws2 = fc2_weight_scale_2[expert_id]
            output = output * ws2.to(dtype)

        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output

    return results.view_as(x)


def compute_with_experts(
    num_experts,
    x,
    w31_weight,
    w2_weight,
    selected_experts,
    routing_weights,
    alpha=None,
    beta=None,
    limit=None,
):
    results = torch.zeros_like(x)
    for expert_id in range(num_experts):
        mask = selected_experts == expert_id
        if not mask.sum():
            continue
        batch_idx, nth_expert = torch.where(mask)
        w31_expert = w31_weight[expert_id]  # [2 * intermediate_size, hidden_size]
        w2_expert = w2_weight[expert_id]  # [hidden_size, intermediate_size]

        # Split w13 into w1 and w3
        w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)

        expert_inputs = x[batch_idx]
        if alpha is not None and limit is not None and beta is not None:
            # SwiGLUBias
            x1 = expert_inputs @ w1_expert.t()
            x1 = x1.clamp_(min=None, max=limit)
            x1_scaled = x1 * torch.sigmoid(alpha * x1)
            x2 = expert_inputs @ w3_expert.t()
            x2 = x2.clamp_(min=-limit, max=limit) + beta

            inter = x1_scaled * x2
        else:
            inter = F.silu(expert_inputs @ w1_expert.t()) * (
                expert_inputs @ w3_expert.t()
            )
        output = inter @ w2_expert.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output
    return results.view_as(x)


# Test configurations
BATCH_SIZES = [
    1,
]
HIDDEN_SIZES = [
    128,
]
NUM_EXPERTS = [2]
TOP_K_VALUES = [2]
INTERMEDIATE_SIZES = [
    128,
]
EP_NUM_EXPERTS = [8]
EP_TOP_K = [2]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe(batch_size, hidden_size, num_experts, top_k, intermediate_size):
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16).cuda() / 5
    router_logits = torch.randn(batch_size, num_experts, dtype=torch.float32).cuda()
    w31_weight = (
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float16
        ).cuda()
        / 5
    )
    w2_weight = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.float16
        ).cuda()
        / 5
    )

    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )
    flash_output = torch.empty_like(ref_output)
    flash_output = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        w31_weight,
        w2_weight,
        flash_output.dtype,
        output=flash_output,
        quant_scales=None,
    )

    torch.testing.assert_close(ref_output, flash_output[0], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("otype, wtype", [(torch.float16, torch.float8_e4m3fn)])
def test_moe_fp8(
    batch_size, hidden_size, num_experts, top_k, intermediate_size, otype, wtype
):
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    input_shape = (batch_size, hidden_size)
    w31_shape = (num_experts, 2 * intermediate_size, hidden_size)
    w2_shape = (num_experts, hidden_size, intermediate_size)
    x = cast_to_representable(gen_tensor(input_shape, otype))
    router_logits = gen_tensor((batch_size, num_experts), otype)

    # Create weight tensors
    w31_weight = gen_tensor(w31_shape, otype, wtype)
    w2_weight = gen_tensor(w2_shape, otype, wtype)
    w31_scales = torch.empty(num_experts, 2, dtype=otype).cuda()
    w2_scales = torch.empty(num_experts, 1, dtype=otype).cuda()

    w31_dequantized = gen_tensor(w31_shape, otype)
    w2_dequantized = gen_tensor(w2_shape, otype)
    for expert_id in range(num_experts):
        w31 = cast_to_representable(gen_tensor(w31_shape[1:], otype, scale=0.1))
        w2 = cast_to_representable(gen_tensor(w2_shape[1:], otype, scale=0.09))

        w31_quant, s31 = dynamic_per_tensor_fp8_quant(w31)
        w2_quant, s2 = dynamic_per_tensor_fp8_quant(w2)

        w31_weight.data[expert_id].copy_(w31_quant)
        w2_weight.data[expert_id].copy_(w2_quant)
        w31_scales.data[expert_id].copy_(s31)
        w2_scales.data[expert_id].copy_(s2)
        w31_dequantized.data[expert_id].copy_(torch.mul(w31_quant.to(dtype=otype), s31))
        w2_dequantized.data[expert_id].copy_(torch.mul(w2_quant.to(dtype=otype), s2))

    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    ref_output = compute_with_experts(
        num_experts,
        x,
        w31_dequantized,
        w2_dequantized,
        selected_experts,
        routing_weights,
    )
    flash_output = torch.empty_like(ref_output)
    # For fp8, the hidden_state expects quantized.
    _, w1_scales = torch.chunk(w31_scales, 2, dim=-1)
    x_quant, hidden_states_scale = dynamic_per_tensor_fp8_quant(x)
    hidden_states_scale = torch.tensor(hidden_states_scale[0]).cuda()
    quant_scales = [
        torch.squeeze(w1_scales * hidden_states_scale).float(),
        torch.tensor(1.0).cuda(),
        torch.squeeze(1.0 * w2_scales).float(),
        hidden_states_scale,
    ]

    _ = fused_moe.cutlass_fused_moe(
        x_quant,
        selected_experts.to(torch.int),
        routing_weights,
        w31_weight,
        w2_weight,
        otype,
        quant_scales=quant_scales,
        output=flash_output,
    )
    torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize(
    "otype, wtype",
    [(torch.float16, torch.float8_e4m3fn), (torch.bfloat16, torch.float8_e4m3fn)],
)
@pytest.mark.parametrize("quantized_input", [False, True])
@pytest.mark.parametrize(
    "activation_type",
    [ActivationType.Swiglu, ActivationType.Relu2],
    ids=["swiglu", "relu2"],
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10, 11, 12],
    reason="NVFP4 is only supported on SM100, SM110 and SM120/SM121",
)
def test_moe_nvfp4(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    wtype,
    quantized_input,
    activation_type,
):
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    quant_blocksize = 16
    round_up = lambda x, y: (x + y - 1) // y * y
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    w1_n = 2 * n if activation_type == ActivationType.Swiglu else n
    w1 = torch.randn((e, w1_n, k), device="cuda", dtype=otype) / 10

    sf_w1_2n = round_up(w1_n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_blockscale_cutlass = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )

    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_q = torch.empty((e, w1_n, k // 2), device="cuda", dtype=torch.uint8)
    w1_q_cutlass = torch.empty((e, w1_n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1).max().to(torch.float32)
        w2_amax = torch.abs(w2).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])

        w1_q_cutlass[expert], w1_blockscale_cutlass[expert] = fp4_quantize(
            w1[expert], w1_gs[expert]
        )

        w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])

    x = torch.randn(m, k, dtype=otype).cuda()
    a1_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(x).max().to(
        torch.float32
    ).cuda()
    a1_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    # quant_scales format
    # auto const fc1_act_global = quant_scales.value()[0];
    # auto const fc1_weight_block = quant_scales.value()[1];
    # auto const fc1_global = quant_scales.value()[2];
    # auto const fc2_act_global = quant_scales.value()[3];
    # auto const fc2_weight_block = quant_scales.value()[4];
    # auto const fc2_global = quant_scales.value()[5];
    flash_output = torch.zeros_like(x)

    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]
    hidden_states = x
    input_sf = None
    if quantized_input:
        hidden_states, input_sf = fp4_quantize(x, a1_gs)
    _ = fused_moe.cutlass_fused_moe(
        hidden_states,
        selected_experts.to(torch.int),
        routing_weights,
        w1_q.contiguous().view(torch.long),
        w2_q.contiguous().view(torch.long),
        otype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        output=flash_output,
        activation_type=activation_type,
    )

    # Ref check
    a_fp4, a_scale_interleaved = fp4_quantize(x, a1_gs)
    _, m_k = a_fp4.shape
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a1_gs,
        dtype=otype,
        device=x.device,
        block_size=quant_blocksize,
    )

    w1_d = torch.empty((e, w1_n, k), device="cuda", dtype=otype)
    w2_d = torch.empty((e, k, n), device="cuda", dtype=otype)

    for idx in range(0, e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx],
            w1_blockscale[idx],
            w1_gs[idx],
            dtype=w1.dtype,
            device=w1.device,
            block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx],
            w2_blockscale[idx],
            w2_gs[idx],
            dtype=w2.dtype,
            device=w2.device,
            block_size=quant_blocksize,
        )

    ref_output = torch_moe_nvfp4(
        a_in_dtype,
        w1_d,
        w2_d,
        top_k,
        routing_weights,
        selected_experts,
        activation_type,
    )
    torch.testing.assert_close(ref_output, flash_output, rtol=2e-1, atol=2e-1)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", EP_NUM_EXPERTS)
@pytest.mark.parametrize("top_k", EP_TOP_K)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe_expert_parallel(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """
    Test expert parallelism with X GPUs and Y experts.
    Each GPU handles one expert and results are reduced.

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts (must be 2 for this test)
        top_k: Number of experts to route to per token
        intermediate_size: Intermediate dimension size
        activation: Activation function type
    """
    # This test is specifically for 2 GPUs and 2 experts
    # GPU 0 (ep_rank=0) handles expert 0
    # GPU 1 (ep_rank=1) handles expert 1
    ep_size = num_experts // 2
    torch.manual_seed(42)

    # Create input tensors
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16).cuda()

    # Create weight tensors - each GPU will have one expert
    w31_weight = (
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float16
        ).cuda()
        / 10
    )
    w2_weight = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.float16
        ).cuda()
        / 10
    )

    selected_experts = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()

    routing_weights = torch.randn((batch_size, top_k)).cuda()
    routing_weights = F.softmax(routing_weights, dim=1)
    ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )

    outputs = []
    flash_output = torch.zeros_like(ref_output)
    for ep_rank in range(ep_size):
        # Create output tensor for this GPU
        out_hidden_states_local = torch.zeros_like(x)

        # Compute expert start and end positions for this rank
        experts_per_rank = (
            num_experts // ep_size
        )  # 2 GPUs, so each gets half the experts
        expert_start = ep_rank * experts_per_rank
        expert_end = expert_start + experts_per_rank  # if ep_rank < 1 else num_experts

        w31_weight_local = w31_weight[
            expert_start:expert_end, :
        ]  # Get only the experts for this rank
        w2_weight_local = w2_weight[
            expert_start:expert_end, :
        ]  # Get only the experts for this rank

        _ = fused_moe.cutlass_fused_moe(
            x.contiguous(),
            selected_experts.to(torch.int),
            routing_weights,
            w31_weight_local.contiguous(),
            w2_weight_local.contiguous(),
            x.dtype,
            ep_size=ep_size,
            ep_rank=ep_rank,
            quant_scales=None,
            output=out_hidden_states_local,
        )
        outputs.append(out_hidden_states_local)

    # Reduce results from all GPUs
    for ep_rank in range(ep_size):
        flash_output += outputs[ep_rank]  # [batch_size, num_experts]
    torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)


TP_SIZES = [2, 4]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe_tensor_parallel(
    batch_size, hidden_size, num_experts, tp_size, intermediate_size
):
    """
    Test tensor parallelism with:
    - w31 sharded along second dimension (non-contracting)
    - w2 sharded along third dimension (contracting)
    - All-reduce to sum partial results

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        top_k: Number of experts to route to per token
        intermediate_size: Intermediate dimension size
        activation: Activation function type
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    top_k = 2
    # Create input tensors
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16).cuda()

    # Create weight tensors
    w31_weight = (
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float16
        ).cuda()
        / 10
    )
    w2_weight = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.float16
        ).cuda()
        / 10
    )

    # Generate unique random expert indices for each token
    selected_experts = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()

    routing_weights = torch.randn((batch_size, top_k)).cuda()
    routing_weights = F.softmax(routing_weights, dim=1)

    # Run reference implementation (no parallelism)
    ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )

    # Simulate tensor parallelism on # TP GPUs
    outputs = []
    for tp_rank in range(tp_size):
        # Create output tensor for this GPU
        out_hidden_states_local = torch.zeros_like(x)

        # Shard w31 along second dimension (intermediate_size)
        # First split w31 into w3 and w1
        w3_weight, w1_weight = torch.chunk(
            w31_weight, 2, dim=1
        )  # [num_experts, intermediate_size, hidden_size] each

        # Shard w3 and w1 separately
        w3_shard_size = intermediate_size // tp_size
        w3_start = tp_rank * w3_shard_size
        w3_end = w3_start + w3_shard_size
        w3_weight_local = w3_weight[:, w3_start:w3_end, :]

        w1_shard_size = intermediate_size // tp_size
        w1_start = tp_rank * w1_shard_size
        w1_end = w1_start + w1_shard_size
        w1_weight_local = w1_weight[:, w1_start:w1_end, :]

        # Stack the sharded weights back together
        w31_weight_local = torch.cat([w3_weight_local, w1_weight_local], dim=1)

        # Shard w2 along third dimension (intermediate_size)
        w2_shard_size = intermediate_size // tp_size
        w2_start = tp_rank * w2_shard_size
        w2_end = w2_start + w2_shard_size
        w2_weight_local = w2_weight[:, :, w2_start:w2_end]

        _ = fused_moe.cutlass_fused_moe(
            x.contiguous(),
            selected_experts.to(torch.int),
            routing_weights,
            w31_weight_local.contiguous(),
            w2_weight_local.contiguous(),
            x.dtype,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_scales=None,
            output=out_hidden_states_local,
        )
        outputs.append(out_hidden_states_local)

    # All-reduce to sum partial results from all GPUs
    flash_output = sum(outputs)
    torch.testing.assert_close(ref_output, flash_output, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", EP_NUM_EXPERTS)
@pytest.mark.parametrize("top_k", EP_TOP_K)
@pytest.mark.parametrize("tp_size", TP_SIZES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe_tensor_expert_parallel(
    batch_size, hidden_size, num_experts, top_k, tp_size, intermediate_size
):
    """
    Test combined tensor parallelism and expert parallelism:
    - Expert parallelism: Distribute experts across GPUs
    - Tensor parallelism: For each expert's weights:
        - w31 sharded along second dimension (non-contracting)
        - w2 sharded along third dimension (contracting)
    - All-reduce to sum partial results

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        tp_size: Number of GPUs for tensor parallelism
        intermediate_size: Intermediate dimension size
    """
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_size, dtype=torch.float16).cuda()
    w31_weight = (
        torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.float16
        ).cuda()
        / 10
    )
    w2_weight = (
        torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.float16
        ).cuda()
        / 10
    )

    # Generate unique random expert indices for each token
    selected_experts = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()

    routing_weights = torch.randn((batch_size, top_k)).cuda()
    routing_weights = F.softmax(routing_weights, dim=1)

    # Run reference implementation (no parallelism)
    ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )

    # Simulate combined parallelism
    ep_size = num_experts // 2  # Number of GPUs for expert parallelism
    outputs = []

    # For each expert parallel rank
    for ep_rank in range(ep_size):
        # Get experts for this rank
        experts_per_rank = num_experts // ep_size
        expert_start = ep_rank * experts_per_rank
        expert_end = expert_start + experts_per_rank

        # Get expert weights for this rank
        w31_weight_ep = w31_weight[
            expert_start:expert_end, :
        ]  # [experts_per_rank, 2*intermediate_size, hidden_size]
        w2_weight_ep = w2_weight[
            expert_start:expert_end, :
        ]  # [experts_per_rank, hidden_size, intermediate_size]

        # For each tensor parallel rank
        for tp_rank in range(tp_size):
            # Create output tensor for this GPU
            out_hidden_states_local = torch.zeros_like(x)

            # Split w31 into w3 and w1
            w3_weight, w1_weight = torch.chunk(w31_weight_ep, 2, dim=1)

            # Shard w3 and w1 separately
            w3_shard_size = intermediate_size // tp_size
            w3_start = tp_rank * w3_shard_size
            w3_end = w3_start + w3_shard_size
            w3_weight_local = w3_weight[:, w3_start:w3_end, :]

            w1_shard_size = intermediate_size // tp_size
            w1_start = tp_rank * w1_shard_size
            w1_end = w1_start + w1_shard_size
            w1_weight_local = w1_weight[:, w1_start:w1_end, :]

            # Stack the sharded weights back together
            w31_weight_local = torch.cat([w3_weight_local, w1_weight_local], dim=1)

            # Shard w2 along third dimension
            w2_shard_size = intermediate_size // tp_size
            w2_start = tp_rank * w2_shard_size
            w2_end = w2_start + w2_shard_size
            w2_weight_local = w2_weight_ep[:, :, w2_start:w2_end]

            # Call flashinfer implementation with both parallelisms
            out_hidden_states_local = fused_moe.cutlass_fused_moe(
                x.contiguous(),
                selected_experts.to(torch.int),
                routing_weights,
                w31_weight_local.contiguous(),
                w2_weight_local.contiguous(),
                x.dtype,
                tp_size=tp_size,
                tp_rank=tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                quant_scales=None,
            )
            outputs.append(out_hidden_states_local[0])

    # All-reduce to sum partial results from all GPUs
    flash_output = sum(outputs)
    torch.testing.assert_close(ref_output, flash_output, rtol=1e-2, atol=1e-2)


def ceil_div(a: int, b: int) -> int:
    return -(a // -b)


def per_block_cast_to_fp8(
    x: torch.Tensor, block_size_n: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, block_size_n) * block_size_n),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def per_token_group_quant_fp8(x, group_size, eps=1e-10, dtype=torch.float8_e4m3fn):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert x.shape[-1] % group_size == 0, (
        "the last dimension of `x` cannot be divisible by `group_size`"
    )
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


def dequantize_block(
    x_quant: torch.Tensor,
    scales: torch.Tensor,
    dtype: torch.dtype,
    original_shape: tuple,
) -> torch.Tensor:
    """
    Dequantize a block-quantized tensor.

    Args:
        x_quant: Quantized tensor
        scales: Block scaling factors
        dtype: Target dtype for dequantization
        original_shape: Original shape of the tensor before padding

    Returns:
        torch.Tensor: Dequantized tensor
    """
    # Reshape scales to match block structure

    def transform_dim(a: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Move target dim to last position if not already last
        if dim != -1:
            a = a.transpose(dim, -1)
        # Broadcast and reshape
        a_broadcasted = a.unsqueeze(-1).expand(*a.shape, 128)
        a_reshaped = a_broadcasted.reshape(*a.shape[:-1], a.shape[-1] * 128)
        # Move back if needed
        if dim != -1:
            a_reshaped = a_reshaped.transpose(dim, -1)
        return a_reshaped

    if x_quant.dim() == 2:  # For activation tensors [batch_size, hidden_size]
        batch_size, hidden_size = x_quant.shape
        num_blocks = (hidden_size + 127) // 128
        scales = scales.view(batch_size, num_blocks, 1).expand(-1, -1, 128)
        scales = scales[:, :, : hidden_size % 128] if hidden_size % 128 != 0 else scales
    else:  # For weight tensors [..., in_dim, out_dim]
        *_dims, in_dim, out_dim = x_quant.shape

        # Transform both dimensions
        scales = transform_dim(scales, -1)  # Last dim
        scales = transform_dim(scales, -2)  # Second-to-last dim

        # Handle padding
        if in_dim % 128 != 0:
            scales = scales[..., : in_dim % 128, :]
        if out_dim % 128 != 0:
            scales = scales[..., :, : out_dim % 128]

    x_dequant = x_quant.to(dtype) * scales.to(dtype)
    return x_dequant.view(original_shape)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
def test_moe_fp8_block_scaling(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """
    Test MoE with FP8 block scaling (Deepseek style):
    - Activation: BF16 (unquantized)
    - Weights: FP8 with 128x128 block scaling
    - Each block has its own scaling factor

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        top_k: Number of experts to route to per token
        intermediate_size: Intermediate dimension size
    """
    torch.manual_seed(42)
    otype = torch.bfloat16

    x = torch.randn(batch_size, hidden_size, dtype=otype).cuda()

    w31_weight = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=otype).cuda()
        / 10
    )
    w2_weight = (
        torch.randn(num_experts, hidden_size, intermediate_size, dtype=otype).cuda()
        / 10
    )

    # Generate unique random expert indices for each token
    selected_experts = torch.stack(
        [torch.randperm(num_experts)[:top_k] for _ in range(batch_size)]
    ).cuda()

    routing_weights = torch.randn((batch_size, top_k)).cuda()
    routing_weights = F.softmax(routing_weights, dim=1)

    # Quantize input and weights
    x_quant, x_scales = per_token_group_quant_fp8(x, group_size=128)

    w31_dequant = torch.empty_like(w31_weight)
    w2_dequant = torch.empty_like(w2_weight)
    w31_quant = torch.empty_like(w31_weight).to(torch.float8_e4m3fn)
    w2_quant = torch.empty_like(w2_weight).to(torch.float8_e4m3fn)
    w31_scales = torch.zeros(
        num_experts,
        ceil_div(2 * intermediate_size, 128),
        ceil_div(hidden_size, 128),
        dtype=torch.float32,
    ).cuda()
    w2_scales = torch.zeros(
        num_experts,
        ceil_div(hidden_size, 128),
        ceil_div(intermediate_size, 128),
        dtype=torch.float32,
    ).cuda()

    for expert_id in range(num_experts):
        w31, w31_s = per_block_cast_to_fp8(w31_weight[expert_id, :])
        w2, w2_s = per_block_cast_to_fp8(w2_weight[expert_id, :])
        w31_quant.data[expert_id].copy_(w31)
        w31_scales.data[expert_id].copy_(w31_s)
        w2_quant.data[expert_id].copy_(w2)
        w2_scales.data[expert_id].copy_(w2_s)
    # Dequantize for verification
    x_dequant = dequantize_block(x_quant, x_scales, x.dtype, x.shape)
    w31_dequant = dequantize_block(
        w31_quant, w31_scales, w31_weight.dtype, w31_weight.shape
    )
    w2_dequant = dequantize_block(w2_quant, w2_scales, w2_weight.dtype, w2_weight.shape)

    # Run reference implementation with dequantized tensors
    ref_output = compute_with_experts(
        num_experts,
        x_dequant,
        w31_dequant,
        w2_dequant,
        selected_experts,
        routing_weights,
    )

    flash_output = torch.zeros_like(x)

    execption_context = (
        pytest.raises(NotImplementedError)
        if torch.cuda.get_device_capability()[0] != 9
        else nullcontext()
    )

    with execption_context:
        _ = fused_moe.cutlass_fused_moe(
            x.contiguous(),
            selected_experts.to(torch.int),
            routing_weights,
            w31_quant.contiguous(),
            w2_quant.contiguous(),
            otype,
            use_deepseek_fp8_block_scale=True,
            quant_scales=[w31_scales.contiguous(), w2_scales.contiguous()],
            output=flash_output,
        )

        torch.testing.assert_close(flash_output, ref_output, rtol=1e-1, atol=1e-1)


def quant_mxfp4_batches(a, num_experts):
    quant_a = []
    sfs = []
    for i in range(num_experts):
        a_fp4, a_sf = mxfp4_quantize(a[i].cuda())
        quant_a.append(a_fp4)
        sfs.append(a_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)

    return result_quant_a, result_sfs


def quant_mxfp8_batches(a, num_experts):
    quant_a = []
    sfs = []
    for i in range(num_experts):
        a_fp8, a_sf = mxfp8_quantize(a[i].cuda(), True, 32)
        quant_a.append(a_fp8)
        sfs.append(a_sf)

    result_quant_a = torch.stack(quant_a)
    result_sfs = torch.stack(sfs)

    return result_quant_a, result_sfs


def pack_mxfp8_scales_u8_to_int32_batches(
    scale_u8: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    num_experts = scale_u8.size(0)
    aligned_rows = ceil_div(rows, 128) * 128
    k_scales = cols // 32
    aligned_k_scales = ceil_div(k_scales, 4) * 4
    return (
        scale_u8.contiguous()
        .view(num_experts, aligned_rows, aligned_k_scales)
        .view(torch.int32)
        .contiguous()
    )


def dequant_mxfp4_batches(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
):
    num_batches = mat_fp4.size(0)

    scale_tensor = scale_tensor.view(num_batches, -1)

    return torch.stack(
        [
            mxfp4_dequantize(mat_fp4[b, :, :], scale_tensor[b, :])
            for b in range(num_batches)
        ]
    )


def dequant_mxfp8_batches(
    mat_fp8: torch.Tensor,
    scale_tensor: torch.Tensor,
):
    num_batches = mat_fp8.size(0)

    scale_tensor = scale_tensor.view(num_batches, -1)

    return torch.stack(
        [
            mxfp8_dequantize_host(
                mat_fp8[b, :, :].cpu().view(torch.uint8),
                scale_tensor[b, :].cpu().view(torch.uint8).reshape(-1),
                True,
            )
            for b in range(num_batches)
        ]
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("otype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("alpha", "beta", "limit"), [(None, None, None), (0.5, 0.0, 7.0), (1.702, 1.0, 7.0)]
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10, 11, 12],
    reason="MXFP8xMXFP4 is only supported on SM100, SM110 and SM120/SM121",
)
def test_moe_mxfp8_mxfp4(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    alpha,
    beta,
    limit,
):
    """
    Test MoE with MXFP8 activations and MXFP4 weights.
    Uses mxfp8_quantize for activations and fp4_quantize for weights.
    """
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    x = torch.randn(m, k, dtype=otype).cuda()
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10

    mxfp8_x, mxfp8_x_sf = mxfp8_quantize(x, True, 32)

    mxfp4_w1, mxfp4_w1_scale = quant_mxfp4_batches(w1, e)
    mxfp4_w2, mxfp4_w2_scale = quant_mxfp4_batches(w2, e)

    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    fake_input_scale = torch.ones(e, device=x.device)

    quant_scales = [
        mxfp4_w1_scale.view(torch.int32),
        fake_input_scale,
        mxfp4_w2_scale.view(torch.int32),
        fake_input_scale,
    ]

    flash_output = torch.zeros_like(x)

    if alpha is not None and limit is not None and beta is not None:
        alpha_t = torch.ones(e, device=x.device) * alpha
        limit_t = torch.ones(e, device=x.device) * limit
        beta_t = torch.ones(e, device=x.device) * beta
    else:
        alpha_t = None
        limit_t = None
        beta_t = None

    # Call cutlass_fused_moe with MXFP8 activations and MXFP4 weights
    _ = fused_moe.cutlass_fused_moe(
        mxfp8_x,
        selected_experts.to(torch.int),
        routing_weights,
        mxfp4_w1.contiguous().view(torch.long),
        mxfp4_w2.contiguous().view(torch.long),
        otype,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        input_sf=mxfp8_x_sf,
        use_mxfp8_act_scaling=True,
        output=flash_output,
    )

    dq_mxfp8_x = (
        mxfp8_dequantize_host(
            mxfp8_x.cpu().view(torch.uint8),
            mxfp8_x_sf.cpu().view(torch.uint8).reshape(-1),
            True,
        )
        .cuda()
        .to(otype)
    )

    dq_mfxp4_w1 = (
        dequant_mxfp4_batches(
            mxfp4_w1.cpu().view(torch.uint8),
            mxfp4_w1_scale.cpu().view(torch.uint8).reshape(-1),
        )
        .cuda()
        .to(otype)
    )

    dq_mfxp4_w2 = (
        dequant_mxfp4_batches(
            mxfp4_w2.cpu().view(torch.uint8),
            mxfp4_w2_scale.cpu().view(torch.uint8).reshape(-1),
        )
        .cuda()
        .to(otype)
    )

    # Use original weights for reference computation
    ref_output = compute_with_experts(
        e,
        dq_mxfp8_x,
        dq_mfxp4_w1,
        dq_mfxp4_w2,
        selected_experts,
        routing_weights,
        alpha,
        beta,
        limit,
    )

    torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("otype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("alpha", "beta", "limit"), [(None, None, None), (0.5, 0.0, 7.0), (1.702, 1.0, 7.0)]
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10],
    reason="MXFP8xMXFP8 is only supported on SM100 for now",
)
def test_moe_mxfp8_mxfp8(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    alpha,
    beta,
    limit,
):
    """Test MoE with MXFP8 activations and MXFP8 weights."""
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    x = torch.randn(m, k, dtype=otype).cuda()
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10

    mxfp8_x, mxfp8_x_sf = mxfp8_quantize(x, True, 32)
    mxfp8_w1, mxfp8_w1_scale = quant_mxfp8_batches(w1, e)
    mxfp8_w2, mxfp8_w2_scale = quant_mxfp8_batches(w2, e)
    mxfp8_w1_scale_i32 = pack_mxfp8_scales_u8_to_int32_batches(mxfp8_w1_scale, 2 * n, k)
    mxfp8_w2_scale_i32 = pack_mxfp8_scales_u8_to_int32_batches(mxfp8_w2_scale, k, n)

    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    fake_input_scale = torch.ones(e, device=x.device, dtype=torch.float32)
    quant_scales = [
        mxfp8_w1_scale_i32,
        fake_input_scale,
        mxfp8_w2_scale_i32,
        fake_input_scale,
    ]

    flash_output = torch.zeros_like(x)

    if alpha is not None and limit is not None and beta is not None:
        alpha_t = torch.ones(e, device=x.device) * alpha
        limit_t = torch.ones(e, device=x.device) * limit
        beta_t = torch.ones(e, device=x.device) * beta
    else:
        alpha_t = None
        limit_t = None
        beta_t = None

    _ = fused_moe.cutlass_fused_moe(
        mxfp8_x,
        selected_experts.to(torch.int),
        routing_weights,
        mxfp8_w1.contiguous(),
        mxfp8_w2.contiguous(),
        otype,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        input_sf=mxfp8_x_sf,
        use_mxfp8_act_scaling=True,
        output=flash_output,
    )

    dq_mxfp8_x = (
        mxfp8_dequantize_host(
            mxfp8_x.cpu().view(torch.uint8),
            mxfp8_x_sf.cpu().view(torch.uint8).reshape(-1),
            True,
        )
        .cuda()
        .to(otype)
    )
    dq_mxfp8_w1 = dequant_mxfp8_batches(mxfp8_w1, mxfp8_w1_scale).cuda().to(otype)
    dq_mxfp8_w2 = dequant_mxfp8_batches(mxfp8_w2, mxfp8_w2_scale).cuda().to(otype)

    ref_output = compute_with_experts(
        e,
        dq_mxfp8_x,
        dq_mxfp8_w1,
        dq_mxfp8_w2,
        selected_experts,
        routing_weights,
        alpha,
        beta,
        limit,
    )

    torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)


def dequant_mxfp4_batches_host(
    mat_fp4: torch.Tensor,
    scale_tensor: torch.Tensor,
):
    return torch.stack(
        [
            mxfp4_dequantize_host(mat_fp4[b, :, :], scale_tensor[b, :, :])
            for b in range(mat_fp4.size(0))
        ]
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize(
    ("alpha", "beta", "limit"), [(None, None, None), (0.5, 0.0, 7.0), (1.702, 1.0, 7.0)]
)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] != 9,
    reason="BF16xMXFP4 is only supported on SM90",
)
def test_moe_bf16_mxfp4(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    alpha,
    beta,
    limit,
):
    """
    Test MoE with bf16 activations and MXFP4 weights.
    Uses bf16 for activations and fp4_quantize for weights.
    """
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    x = torch.randn(m, k, dtype=torch.bfloat16).cuda()
    w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, n // 2), device="cuda", dtype=torch.uint8)

    w1_scale = torch.randint(
        118, 123, (e, 2 * n, k // 32), device="cuda", dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (e, k, n // 32), device="cuda", dtype=torch.uint8
    )

    router_logits = torch.randn(m, e, dtype=torch.bfloat16).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    flash_output = torch.zeros_like(x)

    if alpha is not None and limit is not None and beta is not None:
        alpha_t = torch.ones(e, device=x.device) * alpha
        limit_t = torch.ones(e, device=x.device) * limit
        beta_t = torch.ones(e, device=x.device) * beta
    else:
        alpha_t = None
        limit_t = None
        beta_t = None

    pad_size = hidden_size - x.shape[1]
    x_pad = torch.nn.functional.pad(x, (0, pad_size))

    # SM90 mixed-input path reads weights / scales in an interleaved byte
    # layout (see ``interleave_moe_{weights,scales}_for_sm90_mixed_gemm``
    # and the LDSM + LUT pipeline ported from TRT-LLM PR #12451). Raw
    # weights produce stale output.
    w1_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
        w1.contiguous().view(torch.uint8), "fp4"
    )
    w2_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
        w2.contiguous().view(torch.uint8), "fp4"
    )
    w1_scale_il = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(w1_scale)
    w2_scale_il = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(w2_scale)

    quant_scales = [
        w1_scale_il.view(torch.int32),
        w2_scale_il.view(torch.int32),
    ]

    # Call cutlass_fused_moe with BF16 activations and MXFP4 weights
    _ = fused_moe.cutlass_fused_moe(
        x_pad,
        selected_experts.to(torch.int),
        routing_weights,
        w1_il,
        w2_il,
        torch.bfloat16,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=quant_scales,
        use_w4_group_scaling=True,
        output=flash_output,
    )

    dq_mfxp4_w1 = (
        dequant_mxfp4_batches_host(
            w1.cpu(),
            w1_scale.cpu(),
        )
        .cuda()
        .to(torch.bfloat16)
    )

    dq_mfxp4_w2 = (
        dequant_mxfp4_batches_host(
            w2.cpu(),
            w2_scale.cpu(),
        )
        .cuda()
        .to(torch.bfloat16)
    )

    # Use original weights for reference computation
    ref_output = compute_with_experts(
        e,
        x,
        dq_mfxp4_w1,
        dq_mfxp4_w2,
        selected_experts,
        routing_weights,
        alpha,
        beta,
        limit,
    )

    torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_autotune", [False, True])
def test_moe_w4a8(
    batch_size: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    dtype: torch.dtype,
    use_autotune: bool,
):
    """Test MoE with W4A8 quantization (INT4 weights, FP8 activations)."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("W4A8 is only supported on SM90")
    if top_k > num_experts:
        pytest.skip("top_k must be <= num_experts")

    torch.manual_seed(42)
    group_size = 128
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    affine_coeff = 0.005

    x = torch.randn(m, k, dtype=dtype, device="cuda")
    router_logits = torch.randn(m, e, dtype=dtype, device="cuda")
    w1_weight = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")
    w2_weight = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device="cuda")
    w3_weight = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device="cuda")

    # per group weight
    w1_scale = (
        torch.randn(e, n, k // group_size, dtype=dtype, device="cuda") * affine_coeff
    )
    w2_scale = (
        torch.randn(e, k, n // group_size, dtype=dtype, device="cuda") * affine_coeff
    )
    w3_scale = (
        torch.randn(e, n, k // group_size, dtype=dtype, device="cuda") * affine_coeff
    )

    # per channel pre quant scales
    w1_pre_quant_scale = torch.rand(e, k, dtype=dtype, device="cuda") * 0.1 + 0.95
    w2_pre_quant_scale = torch.rand(e, n, dtype=dtype, device="cuda") * 0.1 + 0.95
    w3_pre_quant_scale = torch.rand(e, k, dtype=dtype, device="cuda") * 0.1 + 0.95

    input_scale = torch.rand(e, 1, dtype=torch.float32, device="cuda") * 0.2 + 0.1
    weight_scale_2 = torch.ones(e, 1, dtype=torch.float32, device="cuda")

    fc1_weights = torch.cat([w3_weight, w1_weight], dim=1)
    fc2_weights = w2_weight

    # Weight byte interleave required by the SM90 mixed-input GEMM
    # (ported from TRT-LLM PR #12451). Scale reshape+permute is done below.
    fc1_weights_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
        fc1_weights.contiguous().view(torch.uint8), "int4"
    )
    fc2_weights_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
        fc2_weights.contiguous().view(torch.uint8), "int4"
    )

    def interleave_weights(w: torch.Tensor, dim: int) -> torch.Tensor:
        # Factors are chosen based on TRTLLM's quantization.py
        interleave_factor = 4 if dim % 512 == 0 else (2 if dim % 256 == 0 else 1)
        s = w.shape
        w_interleaved = (
            w.reshape(s[0], s[1], s[2] // interleave_factor, interleave_factor)
            .permute(0, 2, 1, 3)
            .reshape(s[0], s[2] // interleave_factor, s[1] * interleave_factor)
            .contiguous()
        )
        return w_interleaved

    w3_w1_scales = torch.cat([w3_scale, w1_scale], dim=1)
    w3_w1_scales_int = interleave_weights(w3_w1_scales, k)
    w2_scales_int = interleave_weights(w2_scale, n)

    # act scales
    w3_w1_pre_quant_max = torch.max(w1_pre_quant_scale, w3_pre_quant_scale)
    w3_w1_input_scale_max = input_scale.max()
    fc31_act_scale = (w3_w1_pre_quant_max / w3_w1_input_scale_max).to(dtype)
    fc2_act_scale = (w2_pre_quant_scale / input_scale).to(dtype).unsqueeze(-1)

    fc31_alpha = (weight_scale_2.squeeze(-1) * w3_w1_input_scale_max).float()
    fc2_alpha = (weight_scale_2.squeeze(-1) * input_scale.squeeze(-1)).float()

    zero_1 = torch.empty(0, dtype=dtype, device="cuda")
    zero_2 = torch.empty(0, dtype=dtype, device="cuda")

    # SM90 requires bfloat16 bit patterns
    sm = (
        torch.cuda.get_device_capability()[0] * 10
        + torch.cuda.get_device_capability()[1]
    )
    if sm >= 90:
        w3_w1_scales_out = w3_w1_scales_int.to(torch.bfloat16).view(dtype)
        w2_scales_out = w2_scales_int.to(torch.bfloat16).view(dtype)
        fc31_act_out = fc31_act_scale.to(torch.bfloat16).view(dtype)
        fc2_act_out = fc2_act_scale.to(torch.bfloat16).view(dtype)
    else:
        w3_w1_scales_out = w3_w1_scales_int.to(dtype)
        w2_scales_out = w2_scales_int.to(dtype)
        fc31_act_out = fc31_act_scale
        fc2_act_out = fc2_act_scale

    quant_scales = (
        w3_w1_scales_out,
        w2_scales_out,
        fc31_act_out,
        fc2_act_out,
        zero_1,
        zero_2,
        fc31_alpha,
        fc2_alpha,
    )

    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    selected_experts_int32 = selected_experts.to(torch.int32)

    flash_output = torch.zeros_like(x)
    with autotune(True) if use_autotune else nullcontext():
        _ = fused_moe.cutlass_fused_moe(
            x,
            selected_experts_int32,
            routing_weights,
            fc1_weights_il,
            fc2_weights_il,
            dtype,
            quant_scales=quant_scales,
            use_w4_group_scaling=True,
            output=flash_output,
            use_packed_weights=True,
        )

    w31_weight_list = []
    w2_weight_list = []

    for e_idx in range(num_experts):
        w1_w = w1_weight[e_idx]  # [N, K//2]
        w3_w = w3_weight[e_idx]  # [N, K//2]
        w2_w = w2_weight[e_idx]  # [K, N//2]
        w1_s = w1_scale[e_idx]  # [N, K//group_size]
        w3_s = w3_scale[e_idx]  # [N, K//group_size]
        w2_s = w2_scale[e_idx]  # [K, N//group_size]
        ws2 = weight_scale_2[e_idx]  # [1]

        # dequant w1 and w3: [N, K//2] -> [N, K]
        w1_dequant = dequantize_int4_to_dtype(w1_w, w1_s, group_size, dtype, ws2)
        w3_dequant = dequantize_int4_to_dtype(w3_w, w3_s, group_size, dtype, ws2)

        # dequant w2: [K, N//2] -> [K, N]
        w2_dequant = dequantize_int4_to_dtype(w2_w, w2_s, group_size, dtype, ws2)

        w31 = torch.cat([w3_dequant, w1_dequant], dim=0)  # [2N, K]

        w31_weight_list.append(w31)
        w2_weight_list.append(w2_dequant)

    w31_weight_dequant = torch.stack(w31_weight_list, dim=0)  # [e, 2N, K]
    w2_weight_dequant = torch.stack(w2_weight_list, dim=0)  # [e, K, N]

    ref_output = torch_moe_w4a8(
        num_experts,
        x,
        w31_weight_dequant,
        w2_weight_dequant,
        selected_experts,
        routing_weights,
        fc1_input_scale=input_scale.squeeze(-1),
        fc2_input_scale=input_scale.squeeze(-1),
        fc1_pre_quant_scale=torch.max(w1_pre_quant_scale, w3_pre_quant_scale),
        fc2_pre_quant_scale=w2_pre_quant_scale,
        fc1_weight_scale_2=weight_scale_2.squeeze(-1),
        fc2_weight_scale_2=weight_scale_2.squeeze(-1),
    )
    torch.testing.assert_close(ref_output, flash_output, rtol=1e-2, atol=1e-1)


@pytest.mark.skipif(
    not is_sm100a_supported(torch.device("cuda")),
    reason="NVFP4 is only supported on SM100+",
)
def test_moe_nvfp4_unswizzled_input_sf():
    """Test cutlass_fused_moe with swizzled_input_sf=False (linear layout input_sf).

    In FP4 allgather/alltoall scenarios, the input scaling factors received after
    communication are in linear layout (not swizzled). This test verifies that
    passing swizzled_input_sf=False produces the same output as first swizzling
    the input_sf and passing swizzled_input_sf=True.
    """
    torch.manual_seed(42)
    batch_size = 32
    hidden_size = 128
    intermediate_size = 128
    num_experts = 4
    top_k = 2
    otype = torch.float16
    quant_blocksize = 16

    def round_up(x, y):
        return (x + y - 1) // y * y

    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size
    w1_n = 2 * n  # Swiglu

    w1 = torch.randn((e, w1_n, k), device="cuda", dtype=otype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10

    sf_w1_2n = round_up(w1_n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)

    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_q = torch.empty((e, w1_n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1[expert]).max().to(torch.float32)
        w2_amax = torch.abs(w2[expert]).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
        w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])
        w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])

    x = torch.randn(m, k, dtype=otype).cuda()
    a1_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]

    # Quantize input with swizzled layout (default)
    hidden_states_swizzled, input_sf_swizzled = fp4_quantize(
        x, a1_gs, is_sf_swizzled_layout=True
    )
    # Quantize input with linear layout (as received after allgather/alltoall)
    hidden_states_linear, input_sf_linear = fp4_quantize(
        x, a1_gs, is_sf_swizzled_layout=False
    )

    # Both quantizations should produce the same quantized values
    assert torch.equal(hidden_states_swizzled, hidden_states_linear)
    # The SF buffers must differ — otherwise the test would pass trivially
    # even if fp4_quantize ignored is_sf_swizzled_layout
    assert not torch.equal(input_sf_swizzled, input_sf_linear), (
        "input_sf_swizzled and input_sf_linear should have different layouts"
    )

    output_swizzled = torch.zeros(m, k, dtype=otype, device="cuda")
    output_linear = torch.zeros(m, k, dtype=otype, device="cuda")

    # swizzled_input_sf=True with swizzled input_sf (default behavior)
    fused_moe.cutlass_fused_moe(
        hidden_states_swizzled,
        selected_experts.to(torch.int),
        routing_weights,
        w1_q.contiguous().view(torch.long),
        w2_q.contiguous().view(torch.long),
        otype,
        quant_scales=quant_scales,
        input_sf=input_sf_swizzled,
        swizzled_input_sf=True,
        output=output_swizzled,
    )

    # swizzled_input_sf=False with linear input_sf (post-allgather scenario)
    fused_moe.cutlass_fused_moe(
        hidden_states_linear,
        selected_experts.to(torch.int),
        routing_weights,
        w1_q.contiguous().view(torch.long),
        w2_q.contiguous().view(torch.long),
        otype,
        quant_scales=quant_scales,
        input_sf=input_sf_linear,
        swizzled_input_sf=False,
        output=output_linear,
    )

    torch.testing.assert_close(output_swizzled, output_linear, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    "hidden_size, intermediate_size",
    [
        # hidden_size=288: 288/16=18 scale cols, round_up(18,4)=20 -> padding.
        # Exercises the weight_scale_vec_size snap fix (issue #2847).
        (288, 128),
        # Non-aligned hidden AND intermediate — exercises K-dim padding in BOTH
        # expandInputRows (FC1 SFs, hidden→padded_hidden) and doActivation (FC2 SFs,
        # inter→padded_inter).
        (288, 192),  # hidden 288→384, inter 192→256: K-dim padding in both kernels
        (160, 192),  # hidden 160→256, inter 192→256: K-dim padding in both kernels
        (320, 160),  # hidden 320→384, inter 160→256: K-dim padding in both kernels
        # Aligned hidden, non-aligned intermediate — only doActivation K-dim padding
        (
            256,
            192,
        ),  # hidden 256→256 (no pad), inter 192→256: K-dim padding in doActivation only
    ],
)
@pytest.mark.parametrize("num_experts", [2])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize(
    "otype, wtype",
    [(torch.bfloat16, torch.float8_e4m3fn)],
)
@pytest.mark.parametrize("quantized_input", [False])
@pytest.mark.parametrize(
    "activation_type",
    [ActivationType.Swiglu],
    ids=["swiglu"],
)
@pytest.mark.skipif(
    not is_sm100a_supported(torch.device("cuda"))
    and not is_sm12x_supported(torch.device("cuda")),
    reason="NVFP4 is only supported on SM100+",
)
def test_moe_nvfp4_unaligned_hidden_size(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    wtype,
    quantized_input,
    activation_type,
):
    """Test NVFP4 MoE with hidden_size not aligned to sf_block_size * 4.

    When hidden_size/sf_block_size is not a multiple of 4, block_scale_interleave
    pads the scale columns, inflating numel(). This caused weight_scale_vec_size
    to be computed incorrectly (e.g. 31 instead of 32). See issue #2847.
    """
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    quant_blocksize = 16

    def round_up(x, y):
        return (x + y - 1) // y * y

    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    w1_n = 2 * n if activation_type == ActivationType.Swiglu else n
    w1 = torch.randn((e, w1_n, k), device="cuda", dtype=otype) / 10

    sf_w1_2n = round_up(w1_n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )

    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_q = torch.empty((e, w1_n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1[expert]).max().to(torch.float32)
        w2_amax = torch.abs(w2[expert]).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])
        w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])

    x = torch.randn(m, k, dtype=otype).cuda()
    a1_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]

    hidden_states = x
    input_sf = None
    if quantized_input:
        hidden_states, input_sf = fp4_quantize(x, a1_gs)
    flash_output = torch.zeros_like(x)
    _ = fused_moe.cutlass_fused_moe(
        hidden_states,
        selected_experts.to(torch.int),
        routing_weights,
        w1_q.contiguous().view(torch.long),
        w2_q.contiguous().view(torch.long),
        otype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        output=flash_output,
        activation_type=activation_type,
    )

    # Ref check
    a_fp4, a_scale_interleaved = fp4_quantize(x, a1_gs)
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a1_gs,
        dtype=otype,
        device=x.device,
        block_size=quant_blocksize,
    )

    w1_d = torch.empty((e, w1_n, k), device="cuda", dtype=otype)
    w2_d = torch.empty((e, k, n), device="cuda", dtype=otype)

    for idx in range(0, e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx],
            w1_blockscale[idx],
            w1_gs[idx],
            dtype=w1.dtype,
            device=w1.device,
            block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx],
            w2_blockscale[idx],
            w2_gs[idx],
            dtype=w2.dtype,
            device=w2.device,
            block_size=quant_blocksize,
        )

    ref_output = torch_moe_nvfp4(
        a_in_dtype,
        w1_d,
        w2_d,
        top_k,
        routing_weights,
        selected_experts,
        activation_type,
    )
    torch.testing.assert_close(ref_output, flash_output, rtol=2e-1, atol=2e-1)


# NOTE: No MXFP8xMXFP4 unaligned-hidden_size test here because the MXFP4 MoE
# kernel requires hidden_size % 128 == 0, and when that holds, hidden_size / 32
# is always a multiple of 4, so the block_scale_interleave column-padding that
# triggers the weight_scale_vec_size bug cannot occur.


# ============================================================================
# Tests for N-dim SF padding removal safety
# ============================================================================
# The N-dim SF padding (zeroing extra token rows beyond tokens_to_expert up to
# MinNDimAlignment) was removed because CUTLASS grouped GEMM sets gemm_m =
# tokens_to_expert per expert and never reads scale factors for rows beyond
# that. These tests exercise configurations where empty experts and uninitialized
# SF padding rows could cause incorrect results if the GEMM did read them.
#
# Key configurations that stress-test the removal:
# - num_experts >> top_k: many empty experts with uninitialized SF regions
# - Various batch sizes: different amounts of N-dim padding per expert
# - Large hidden/intermediate sizes: more SF buffer area at risk
# - Non-aligned intermediate sizes: confirms K-dim padding (still present) works

NDIM_PADDING_BATCH_SIZES = [1, 4, 8]
NDIM_PADDING_HIDDEN_SIZES = [2048]
NDIM_PADDING_NUM_EXPERTS = [128]
NDIM_PADDING_TOP_K = [8]
NDIM_PADDING_INTERMEDIATE_SIZES = [768, 1024]


@pytest.mark.parametrize("batch_size", NDIM_PADDING_BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", NDIM_PADDING_HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NDIM_PADDING_NUM_EXPERTS)
@pytest.mark.parametrize("top_k", NDIM_PADDING_TOP_K)
@pytest.mark.parametrize("intermediate_size", NDIM_PADDING_INTERMEDIATE_SIZES)
@pytest.mark.parametrize("quantized_input", [False, True])
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10, 11, 12],
    reason="NVFP4 is only supported on SM100, SM110 and SM120/SM121",
)
def test_moe_nvfp4_ndim_padding_safety(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    quantized_input,
):
    """Test that N-dim SF padding removal is safe with many empty experts.

    With num_experts=128 and top_k=8, 120 experts have no tokens. Their SF
    buffer regions contain uninitialized data. This test verifies the CUTLASS
    grouped GEMM produces correct results despite those uninitialized regions.
    """
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    otype = torch.bfloat16
    quant_blocksize = 16
    round_up = lambda x, y: (x + y - 1) // y * y
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    w1_n = 2 * n  # Swiglu gated
    w1 = torch.randn((e, w1_n, k), device="cuda", dtype=otype) / 10

    sf_w1_2n = round_up(w1_n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )

    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_q = torch.empty((e, w1_n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1).max().to(torch.float32)
        w2_amax = torch.abs(w2).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
        w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])
        w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])

    x = torch.randn(m, k, dtype=otype).cuda()
    a1_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    flash_output = torch.zeros_like(x)

    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]
    hidden_states = x
    input_sf = None
    if quantized_input:
        hidden_states, input_sf = fp4_quantize(x, a1_gs)

    _ = fused_moe.cutlass_fused_moe(
        hidden_states,
        selected_experts.to(torch.int),
        routing_weights,
        w1_q.contiguous().view(torch.long),
        w2_q.contiguous().view(torch.long),
        otype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        output=flash_output,
    )

    # Reference: dequantize and compute in high precision
    a_fp4, a_scale_interleaved = fp4_quantize(x, a1_gs)
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_scale_interleaved,
        a1_gs,
        dtype=otype,
        device=x.device,
        block_size=quant_blocksize,
    )

    w1_d = torch.empty((e, w1_n, k), device="cuda", dtype=otype)
    w2_d = torch.empty((e, k, n), device="cuda", dtype=otype)

    for idx in range(0, e):
        w1_d[idx] = dequantize_nvfp4_to_dtype(
            w1_q[idx],
            w1_blockscale[idx],
            w1_gs[idx],
            dtype=w1.dtype,
            device=w1.device,
            block_size=quant_blocksize,
        )
        w2_d[idx] = dequantize_nvfp4_to_dtype(
            w2_q[idx],
            w2_blockscale[idx],
            w2_gs[idx],
            dtype=w2.dtype,
            device=w2.device,
            block_size=quant_blocksize,
        )

    ref_output = torch_moe_nvfp4(
        a_in_dtype,
        w1_d,
        w2_d,
        top_k,
        routing_weights,
        selected_experts,
        ActivationType.Swiglu,
    )
    # Two-tier tolerance for FP4 at larger K dimensions (2048 vs 128 in existing tests):
    # 1. Tight: >=95% of elements within atol=0.5 (baseline on SM120 is ~98%+).
    #    If N-dim padding corruption occurs, this drops dramatically.
    # 2. Relaxed: 100% within atol=2.0. Catches catastrophic NaN/corruption.
    abs_diff = (ref_output - flash_output).abs()
    tight_match_rate = (abs_diff <= 0.5).float().mean().item()
    assert tight_match_rate >= 0.95, (
        f"Only {tight_match_rate * 100:.1f}% of elements within tight tolerance (0.5). "
        f"Expected >=95%."
    )
    assert abs_diff.max().item() <= 2.0, (
        f"Max absolute difference {abs_diff.max().item():.4f} exceeds relaxed tolerance (2.0)."
    )


@pytest.mark.parametrize("batch_size", NDIM_PADDING_BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", NDIM_PADDING_HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NDIM_PADDING_NUM_EXPERTS)
@pytest.mark.parametrize("top_k", NDIM_PADDING_TOP_K)
@pytest.mark.parametrize("intermediate_size", NDIM_PADDING_INTERMEDIATE_SIZES)
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] not in [10, 11, 12],
    reason="MXFP8xMXFP4 is only supported on SM100, SM110 and SM120/SM121",
)
def test_moe_mxfp8_mxfp4_ndim_padding_safety(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
):
    """Test that N-dim SF padding removal is safe for MXFP8xMXFP4 with many empty experts.

    Same rationale as test_moe_nvfp4_ndim_padding_safety but for the MXFP8 activation +
    MXFP4 weight path, which also had N-dim SF padding that was removed.
    """
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    otype = torch.bfloat16
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    x = torch.randn(m, k, dtype=otype).cuda()
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10

    mxfp8_x, mxfp8_x_sf = mxfp8_quantize(x, True, 32)

    mxfp4_w1, mxfp4_w1_scale = quant_mxfp4_batches(w1, e)
    mxfp4_w2, mxfp4_w2_scale = quant_mxfp4_batches(w2, e)

    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    fake_input_scale = torch.ones(e, device=x.device)

    quant_scales = [
        mxfp4_w1_scale.view(torch.int32),
        fake_input_scale,
        mxfp4_w2_scale.view(torch.int32),
        fake_input_scale,
    ]

    flash_output = torch.zeros_like(x)

    _ = fused_moe.cutlass_fused_moe(
        mxfp8_x,
        selected_experts.to(torch.int),
        routing_weights,
        mxfp4_w1.contiguous().view(torch.long),
        mxfp4_w2.contiguous().view(torch.long),
        otype,
        quant_scales=quant_scales,
        input_sf=mxfp8_x_sf,
        use_mxfp8_act_scaling=True,
        output=flash_output,
    )

    # Reference: dequantize and compute in high precision
    dq_mxfp8_x = (
        mxfp8_dequantize_host(
            mxfp8_x.cpu().view(torch.uint8),
            mxfp8_x_sf.cpu().view(torch.uint8).reshape(-1),
            True,
        )
        .cuda()
        .to(otype)
    )

    dq_mfxp4_w1 = (
        dequant_mxfp4_batches(
            mxfp4_w1.cpu().view(torch.uint8),
            mxfp4_w1_scale.cpu().view(torch.uint8).reshape(-1),
        )
        .cuda()
        .to(otype)
    )

    dq_mfxp4_w2 = (
        dequant_mxfp4_batches(
            mxfp4_w2.cpu().view(torch.uint8),
            mxfp4_w2_scale.cpu().view(torch.uint8).reshape(-1),
        )
        .cuda()
        .to(otype)
    )

    ref_output = compute_with_experts(
        e,
        dq_mxfp8_x,
        dq_mfxp4_w1,
        dq_mfxp4_w2,
        selected_experts,
        routing_weights,
    )

    # Two-tier tolerance — MXFP8×MXFP4 has significantly higher error than NVFP4 due to
    # two levels of block scaling. Baseline on SM120 is ~76-85% at atol=0.5.
    # 1. Tight: >=95% within atol=1.0. 2. Relaxed: 100% within atol=3.0.
    abs_diff = (ref_output - flash_output).abs()
    tight_match_rate = (abs_diff <= 1.0).float().mean().item()
    assert tight_match_rate >= 0.95, (
        f"Only {tight_match_rate * 100:.1f}% of elements within tight tolerance (1.0). "
        f"Expected >=95%."
    )
    assert abs_diff.max().item() <= 3.0, (
        f"Max absolute difference {abs_diff.max().item():.4f} exceeds relaxed tolerance (3.0)."
    )


# ============================================================================
# SM90 mixed-input MoE tests — PR #3084
#
# Exercise the W4A16 (MXFP4 x BF16) and W4A8 (INT4 x FP8) paths with the
# preprocessing helpers exposed by this PR: weights go through
# ``interleave_moe_weights_for_sm90_mixed_gemm``, MXFP4 block scales go
# through ``interleave_moe_scales_for_sm90_mixed_gemm``, and W4A8 weight
# scales use a local group-wise reshape+permute (factor = 4 / 2 / 1 based on
# whether K is divisible by 512 / 256) to match the W4A8 kernel layout.
# ============================================================================


_MXFP4_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _dequant_mxfp4_on_device(
    w_fp4: torch.Tensor, w_scale: torch.Tensor
) -> torch.Tensor:
    """GPU dequant for a batched MXFP4 tensor. Avoids the host round-trip
    of ``dequant_mxfp4_batches_host`` and — crucially — allows the caller to
    pass only the active-expert slice, which at e=256 / h=4096 / n=2048 is
    the difference between fitting and OOMing a reference dequant on H200.
    """
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=w_fp4.device)
    lo = w_fp4 & 0x0F
    hi = (w_fp4 >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(*w_fp4.shape[:-1], -1)
    values = lut[nib.long()]
    scale = torch.exp2(w_scale.to(torch.float32) - 127.0)
    scale = scale.repeat_interleave(32, dim=-1)
    return (values * scale).to(torch.bfloat16)


def _compute_with_active_experts(
    active_experts,
    x,
    w31_by_expert,
    w2_by_expert,
    selected_experts,
    routing_weights,
    alpha=None,
    beta=None,
    limit=None,
):
    results = torch.zeros_like(x)
    for expert_id in active_experts.tolist():
        mask = selected_experts == expert_id
        if not mask.any():
            continue
        batch_idx, nth_expert = torch.where(mask)
        w3_expert, w1_expert = torch.chunk(w31_by_expert[expert_id], 2, dim=0)
        w2_expert = w2_by_expert[expert_id]
        expert_inputs = x[batch_idx]
        if alpha is not None and limit is not None and beta is not None:
            x1 = expert_inputs @ w1_expert.t()
            x1 = x1.clamp_(min=None, max=limit)
            x1_scaled = x1 * torch.sigmoid(alpha * x1)
            x2 = expert_inputs @ w3_expert.t()
            x2 = x2.clamp_(min=-limit, max=limit) + beta
            inter = x1_scaled * x2
        else:
            inter = F.silu(expert_inputs @ w1_expert.t()) * (
                expert_inputs @ w3_expert.t()
            )
        output = inter @ w2_expert.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output
    return results


W4A16_CORRECTNESS_CONFIGS = [
    (1, 128, 2, 2, 128),
    (4, 128, 4, 2, 128),
    (4, 768, 8, 2, 512),
    (4, 2048, 8, 4, 1024),
    (4, 4096, 8, 4, 2048),
]

W4A16_COVERAGE_CONFIGS = [
    (1, 4096, 256, 6, 2048),
    (4, 2048, 256, 6, 1024),
    (4, 4096, 8, 2, 2048),
    (4, 4096, 256, 1, 2048),
    (4, 4096, 256, 8, 2048),
]

W4A16_ACTIVATION_CONFIGS = [
    (4, 4096, 8, 4, 2048, None, None, None),
    (4, 4096, 8, 4, 2048, 0.5, 0.0, 7.0),
    (4, 4096, 8, 4, 2048, 1.702, 1.0, 7.0),
]


def _run_w4a16_moe_hopper(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    alpha=None,
    beta=None,
    limit=None,
    strict_correctness=True,
):
    torch.manual_seed(42)
    device = torch.device("cuda")
    e, m, n, k = num_experts, batch_size, intermediate_size, hidden_size

    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    w1 = torch.randint(0, 256, (e, 2 * n, k // 2), device=device, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, k, n // 2), device=device, dtype=torch.uint8)
    w1_scale = torch.randint(
        118, 123, (e, 2 * n, k // 32), device=device, dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (e, k, n // 32), device=device, dtype=torch.uint8
    )

    router_logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    if alpha is not None:
        alpha_t = torch.ones(e, device=device) * alpha
        limit_t = torch.ones(e, device=device) * limit
        beta_t = torch.ones(e, device=device) * beta
    else:
        alpha_t = limit_t = beta_t = None

    w1_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w1, "fp4")
    w2_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(w2, "fp4")
    w1_scale_il = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(w1_scale)
    w2_scale_il = fused_moe.interleave_moe_scales_for_sm90_mixed_gemm(w2_scale)

    flash_output = torch.zeros_like(x)
    fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int),
        routing_weights,
        w1_il,
        w2_il,
        torch.bfloat16,
        swiglu_alpha=alpha_t,
        swiglu_limit=limit_t,
        swiglu_beta=beta_t,
        quant_scales=[w1_scale_il.view(torch.int32), w2_scale_il.view(torch.int32)],
        use_w4_group_scaling=True,
        output=flash_output,
    )

    active = torch.unique(selected_experts.flatten())
    active_w1 = _dequant_mxfp4_on_device(w1[active], w1_scale[active])
    active_w2 = _dequant_mxfp4_on_device(w2[active], w2_scale[active])
    w31_by_expert = {eid: active_w1[i] for i, eid in enumerate(active.tolist())}
    w2_by_expert = {eid: active_w2[i] for i, eid in enumerate(active.tolist())}
    ref_output = _compute_with_active_experts(
        active,
        x,
        w31_by_expert,
        w2_by_expert,
        selected_experts,
        routing_weights,
        alpha,
        beta,
        limit,
    )
    if strict_correctness:
        torch.testing.assert_close(ref_output, flash_output, rtol=1e-1, atol=1e-1)
    else:
        diff = (ref_output.float() - flash_output.float()).abs()
        tol = 0.1 + 1e-1 * ref_output.float().abs()
        close_pct = (diff <= tol).float().mean().item()
        assert close_pct >= 0.999, (
            f"Only {close_pct:.4%} of elements within tolerance (need >= 99.9%). "
            f"max_abs_err={diff.max().item():.4f}"
        )


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")),
    reason="W4A16 MoE (Hopper mixed-input) requires SM90",
)
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size",
    W4A16_CORRECTNESS_CONFIGS,
    ids=[f"m{c[0]}_h{c[1]}_e{c[2]}_k{c[3]}" for c in W4A16_CORRECTNESS_CONFIGS],
)
def test_moe_bf16_mxfp4_hopper_correctness(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    _run_w4a16_moe_hopper(
        batch_size, hidden_size, num_experts, top_k, intermediate_size
    )


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")),
    reason="W4A16 MoE (Hopper mixed-input) requires SM90",
)
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size",
    W4A16_COVERAGE_CONFIGS,
    ids=[f"m{c[0]}_h{c[1]}_e{c[2]}_k{c[3]}_n{c[4]}" for c in W4A16_COVERAGE_CONFIGS],
)
def test_moe_bf16_mxfp4_hopper_coverage(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    if top_k > num_experts:
        pytest.skip(f"top_k ({top_k}) > num_experts ({num_experts})")
    _run_w4a16_moe_hopper(
        batch_size,
        hidden_size,
        num_experts,
        top_k,
        intermediate_size,
        strict_correctness=False,
    )


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")),
    reason="W4A16 MoE (Hopper mixed-input) requires SM90",
)
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size,alpha,beta,limit",
    W4A16_ACTIVATION_CONFIGS,
    ids=["swiglu_default", "alpha_0.5", "alpha_1.702"],
)
def test_moe_bf16_mxfp4_hopper_activations(
    batch_size, hidden_size, num_experts, top_k, intermediate_size, alpha, beta, limit
):
    _run_w4a16_moe_hopper(
        batch_size,
        hidden_size,
        num_experts,
        top_k,
        intermediate_size,
        alpha,
        beta,
        limit,
    )


# W4A8 Hopper interleaved path.
#
# Strict-tolerance envelope: h == intermediate_size == 512 with e == 2 only.
# Larger shapes exceed assert_close(rtol=1e-2, atol=1e-1) because of FP8 + INT4
# accumulation noise — the upstream ``test_moe_w4a8`` above stays inside the
# same envelope for the same reason (verified on H200: e=2/h=2048 and
# e=8/h=512 both fail against a float32 PyTorch reference).
W4A8_CORRECTNESS_CONFIGS = [
    (1, 512, 2, 2, 512),
    (4, 512, 2, 2, 512),
]


def _run_w4a8_moe_hopper(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    dtype=torch.bfloat16,
    use_autotune=False,
):
    torch.manual_seed(42)
    group_size = 128
    e, m, n, k = num_experts, batch_size, intermediate_size, hidden_size
    affine_coeff = 0.005
    device = torch.device("cuda")

    x = torch.randn(m, k, dtype=dtype, device=device)
    router_logits = torch.randn(m, e, dtype=dtype, device=device)
    w1_weight = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=device)
    w2_weight = torch.randint(0, 256, (e, k, n // 2), dtype=torch.uint8, device=device)
    w3_weight = torch.randint(0, 256, (e, n, k // 2), dtype=torch.uint8, device=device)

    w1_scale = (
        torch.randn(e, n, k // group_size, dtype=dtype, device=device) * affine_coeff
    )
    w2_scale = (
        torch.randn(e, k, n // group_size, dtype=dtype, device=device) * affine_coeff
    )
    w3_scale = (
        torch.randn(e, n, k // group_size, dtype=dtype, device=device) * affine_coeff
    )
    w1_pre_quant_scale = torch.rand(e, k, dtype=dtype, device=device) * 0.1 + 0.95
    w2_pre_quant_scale = torch.rand(e, n, dtype=dtype, device=device) * 0.1 + 0.95
    w3_pre_quant_scale = torch.rand(e, k, dtype=dtype, device=device) * 0.1 + 0.95
    input_scale = torch.rand(e, 1, dtype=torch.float32, device=device) * 0.2 + 0.1
    weight_scale_2 = torch.ones(e, 1, dtype=torch.float32, device=device)

    fc1_weights = torch.cat([w3_weight, w1_weight], dim=1)
    fc2_weights = w2_weight
    fc1_weights_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
        fc1_weights.contiguous().view(torch.uint8), "int4"
    )
    fc2_weights_il = fused_moe.interleave_moe_weights_for_sm90_mixed_gemm(
        fc2_weights.contiguous().view(torch.uint8), "int4"
    )

    def _interleave_scales(w, dim):
        factor = 4 if dim % 512 == 0 else (2 if dim % 256 == 0 else 1)
        s = w.shape
        return (
            w.reshape(s[0], s[1], s[2] // factor, factor)
            .permute(0, 2, 1, 3)
            .reshape(s[0], s[2] // factor, s[1] * factor)
            .contiguous()
        )

    w3_w1_scales_int = _interleave_scales(torch.cat([w3_scale, w1_scale], dim=1), k)
    w2_scales_int = _interleave_scales(w2_scale, n)
    # Weight scales: bf16 bit-pattern trick; act scales stay in native dtype.
    w3_w1_scales_out = w3_w1_scales_int.to(torch.bfloat16).view(dtype)
    w2_scales_out = w2_scales_int.to(torch.bfloat16).view(dtype)

    w3_w1_input_scale_max = input_scale.max()
    fc31_act_scale = (
        torch.max(w1_pre_quant_scale, w3_pre_quant_scale) / w3_w1_input_scale_max
    ).to(dtype)
    fc2_act_scale = (w2_pre_quant_scale / input_scale).to(dtype).unsqueeze(-1)
    fc31_alpha = (weight_scale_2.squeeze(-1) * w3_w1_input_scale_max).float()
    fc2_alpha = (weight_scale_2.squeeze(-1) * input_scale.squeeze(-1)).float()
    zero_1 = torch.empty(0, dtype=dtype, device=device)
    zero_2 = torch.empty(0, dtype=dtype, device=device)

    quant_scales = (
        w3_w1_scales_out,
        w2_scales_out,
        fc31_act_scale,
        fc2_act_scale,
        zero_1,
        zero_2,
        fc31_alpha,
        fc2_alpha,
    )

    routing_weights, selected_experts = compute_routing(router_logits, top_k)
    flash_output = torch.zeros_like(x)
    with autotune(True) if use_autotune else nullcontext():
        fused_moe.cutlass_fused_moe(
            x,
            selected_experts.to(torch.int32),
            routing_weights,
            fc1_weights_il,
            fc2_weights_il,
            dtype,
            quant_scales=quant_scales,
            use_w4_group_scaling=True,
            output=flash_output,
            use_packed_weights=True,
        )

    w31_list, w2_list = [], []
    for e_idx in range(num_experts):
        ws2 = weight_scale_2[e_idx]
        w1_dq = dequantize_int4_to_dtype(
            w1_weight[e_idx], w1_scale[e_idx], group_size, dtype, ws2
        )
        w3_dq = dequantize_int4_to_dtype(
            w3_weight[e_idx], w3_scale[e_idx], group_size, dtype, ws2
        )
        w2_dq = dequantize_int4_to_dtype(
            w2_weight[e_idx], w2_scale[e_idx], group_size, dtype, ws2
        )
        w31_list.append(torch.cat([w3_dq, w1_dq], dim=0))
        w2_list.append(w2_dq)

    # Broadcast max over experts; see comment on fc31_act_scale above.
    fc1_input_scale_for_ref = torch.full_like(
        input_scale.squeeze(-1), w3_w1_input_scale_max.item()
    )
    ref_output = torch_moe_w4a8(
        num_experts,
        x,
        torch.stack(w31_list, dim=0),
        torch.stack(w2_list, dim=0),
        selected_experts,
        routing_weights,
        fc1_input_scale=fc1_input_scale_for_ref,
        fc2_input_scale=input_scale.squeeze(-1),
        fc1_pre_quant_scale=torch.max(w1_pre_quant_scale, w3_pre_quant_scale),
        fc2_pre_quant_scale=w2_pre_quant_scale,
        fc1_weight_scale_2=weight_scale_2.squeeze(-1),
        fc2_weight_scale_2=weight_scale_2.squeeze(-1),
    )
    torch.testing.assert_close(ref_output, flash_output, rtol=1e-2, atol=1e-1)


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")),
    reason="W4A8 MoE (Hopper mixed-input) requires SM90",
)
@pytest.mark.parametrize(
    "batch_size,hidden_size,num_experts,top_k,intermediate_size",
    W4A8_CORRECTNESS_CONFIGS,
    ids=[f"m{c[0]}_h{c[1]}_e{c[2]}_k{c[3]}" for c in W4A8_CORRECTNESS_CONFIGS],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_moe_w4a8_hopper_correctness(
    batch_size, hidden_size, num_experts, top_k, intermediate_size, dtype
):
    _run_w4a8_moe_hopper(
        batch_size, hidden_size, num_experts, top_k, intermediate_size, dtype=dtype
    )


@pytest.mark.skipif(
    not is_sm90a_supported(torch.device("cuda")),
    reason="W4A8 MoE (Hopper mixed-input) requires SM90",
)
def test_moe_w4a8_hopper_autotune():
    _run_w4a8_moe_hopper(4, 512, 2, 2, 512, dtype=torch.bfloat16, use_autotune=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
