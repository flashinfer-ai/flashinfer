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

import pytest
import torch
from torch.nn import functional as F

import flashinfer.fused_moe as fused_moe
from flashinfer import (
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
    return out[0:m, 0:k]


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


def torch_moe_nvfp4(a, w1, w2, topk, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    # score = torch.softmax(score, dim=-1, dtype=torch.float32)
    # topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # w1 needs to be swapped in terms of gate and up_proj

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            m = w1[i].shape[0]
            assert m % 2 == 0
            w1_expert, w3_expert = w1[i][m // 2 :, :], w1[i][: m // 2, :]
            inter = F.silu(a[mask] @ w1_expert.t()) * (a[mask] @ w3_expert.t())
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
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] != 10,
    reason="NVFP4 is only supported on SM100",
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

    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype) / 10
    w1_cutlass = torch.cat((w1[:, n:, :], w1[:, :n, :]), dim=1).contiguous()

    sf_w1_2n = round_up(2 * n, 128)
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
    w1_q = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w1_q_cutlass = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
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
            w1_cutlass[expert], w1_gs[expert]
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

    w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=otype)
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

    w1_q_cutlass = torch.cat((w1_q[:, n:, :], w1_q[:, :n, :]), dim=1).contiguous()
    w1_blockscale_cutlass = torch.cat(
        (w1_blockscale[:, n:, :], w1_blockscale[:, :n, :]), dim=1
    ).contiguous()
    ref_output = torch_moe_nvfp4(
        a_in_dtype, w1_d, w2_d, top_k, routing_weights, selected_experts
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
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] != 10,
    reason="FP8 block scaling is only supported on SM100",
)
def test_moe_fp8_block_scaling(
    batch_size, hidden_size, num_experts, top_k, intermediate_size
):
    """
    Test MoE with FP8 block scaling (Deepseek style):
    - Activation: 128x1 blocks
    - Weights: 128x128 blocks
    - Each block has its own scaling factor

    Args:
        batch_size: Batch size for the input
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        top_k: Number of experts to route to per token
        intermediate_size: Intermediate dimension size
        Only support bf16 for hidden_states
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

    # Run reference implementation (no quantization)
    _ref_output = compute_with_experts(
        num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
    )

    # Quantize input and weights
    x_quant, x_scales = per_token_group_quant_fp8(x, group_size=128)

    w31_dequant = torch.empty_like(w31_weight)
    w2_dequant = torch.empty_like(w2_weight)
    w31_quant = torch.empty_like(w31_weight).to(torch.float8_e4m3fn)
    w2_quant = torch.empty_like(w2_weight).to(torch.float8_e4m3fn)
    w31_scales = torch.randn(
        num_experts,
        ceil_div(2 * intermediate_size, 128),
        ceil_div(hidden_size, 128),
        dtype=torch.float32,
    ).cuda()
    w2_scales = torch.randn(
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
    # Dequantize for verificationa
    x_dequant = dequantize_block(x_quant, x_scales, x.dtype, x.shape)
    w31_dequant = dequantize_block(
        w31_quant, w31_scales, w31_weight.dtype, w31_weight.shape
    )
    w2_dequant = dequantize_block(w2_quant, w2_scales, w2_weight.dtype, w2_weight.shape)

    # Run reference implementation with dequantized tensors
    _ref_output = compute_with_experts(
        num_experts,
        x_dequant,
        w31_dequant,
        w2_dequant,
        selected_experts,
        routing_weights,
    )
    quant_scales = [
        w31_scales,  # .view(-1),  # W31 scales
        w2_scales,  # .view(-1),  # W2 scales
    ]

    # Call flashinfer implementation with block scaling and expect NotImplementedError
    with pytest.raises(
        NotImplementedError,
        match="DeepSeek FP8 Block Scaling is not yet implemented in CUTLASS for Blackwell",
    ):
        _ = fused_moe.cutlass_fused_moe(
            x.contiguous(),
            selected_experts.to(torch.int),
            routing_weights,
            w31_quant.contiguous(),
            w2_quant.contiguous(),
            otype,
            tp_size=1,
            tp_rank=0,
            use_deepseek_fp8_block_scale=True,
            quant_scales=quant_scales,
        )


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
    torch.cuda.get_device_capability()[0] != 10,
    reason="MXFP8xMXFP4 is only supported on SM100",
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

    quant_scales = [
        w1_scale.view(torch.int32),
        w2_scale.view(torch.int32),
    ]

    # Call cutlass_fused_moe with BF16 activations and MXFP4 weights
    _ = fused_moe.cutlass_fused_moe(
        x_pad,
        selected_experts.to(torch.int),
        routing_weights,
        w1.contiguous().view(torch.uint8),
        w2.contiguous().view(torch.uint8),
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
