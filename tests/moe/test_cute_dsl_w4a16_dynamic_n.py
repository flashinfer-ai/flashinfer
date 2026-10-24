# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Tail-addressing and live-routing regressions for split SM100 W4A16 MoE."""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_arch_supported, is_cute_dsl_available
from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)

from .utils import create_moe_tensors


@pytest.fixture(autouse=True)
def require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    capability = torch.cuda.get_device_capability(0)
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Requires Blackwell SM100 or SM103")
    if not is_cute_dsl_available() or not is_cute_dsl_arch_supported(
        *capability, native_only=True
    ):
        pytest.skip("Installed CuTe DSL does not support this GPU")


@pytest.mark.parametrize(
    "mma_m,route_tile,cluster_m",
    [(128, n, 1) for n in (8, 16, 32, 64, 128)]
    + [(128, 64, 2)]
    + [(256, n, 2) for n in (16, 32, 64, 128, 192)],
)
@pytest.mark.parametrize("activation_type", [None, ActivationType.Relu2])
def test_dynamic_n_grouped_gemm_tails(mma_m, route_tile, cluster_m, activation_type):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import _run_grouped_gemm

    rows, k, num_experts = 256, 512, 2
    weight = torch.full(
        (num_experts, rows, k // 2), 0x22, dtype=torch.uint8, device="cuda"
    )
    row = torch.arange(rows, device="cuda")
    expert = torch.arange(num_experts, device="cuda")[:, None]
    # Distinct rows and experts reveal a misplaced TMEM row/column; all values
    # and products are exactly representable before the final BF16 conversion.
    scale_codes = (0x20 + (7 * row + row // 16 + 8 * expert) % 24).to(torch.uint8)
    scale_codes = scale_codes[:, :, None].expand(-1, -1, k // 16).contiguous()
    weight_sf = convert_sf_to_mma_layout(
        block_scale_interleave(scale_codes), m=rows, k=k, num_groups=num_experts
    )
    row_scales = scale_codes[:, :, 0].contiguous().view(torch.float8_e4m3fn).float()
    alpha = torch.tensor([0.5, 1.25], dtype=torch.float32, device="cuda")
    activations = torch.zeros((2 * route_tile, k), dtype=torch.bfloat16, device="cuda")
    token_values = (torch.arange(2 * route_tile, device="cuda") % 31 + 1).float()
    token_values[1::2] *= -1
    k_values = (torch.arange(k // 16, device="cuda") % 4 + 1).float() / 16
    payload = (token_values[:, None] * k_values).to(torch.bfloat16)
    output = torch.empty((2 * route_tile, rows), dtype=torch.bfloat16, device="cuda")
    tile_experts = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    limits = torch.tensor(
        [route_tile, 2 * route_tile], dtype=torch.int32, device="cuda"
    )
    active_tiles = torch.tensor([2], dtype=torch.int32, device="cuda")

    # Shrink from a full tile, cross both 8- and 16-column MMA boundaries, then
    # grow again using the same compiled kernel and buffer addresses.
    tails = sorted(
        {
            n
            for boundary in range(0, route_tile + 1, 8)
            for n in (boundary - 1, boundary, boundary + 1)
            if 0 < n < route_tile
        }
    )
    for iteration, tail in enumerate([route_tile, 0, *tails, route_tile]):
        valid_rows = route_tile + tail
        activations.zero_()
        activations[:, ::16] = payload if iteration % 2 == 0 else -payload
        activations[valid_rows:] = float("nan")
        limits[1] = valid_rows
        output.fill_(-123)
        _run_grouped_gemm(
            weight=weight,
            weight_sf=weight_sf,
            activations=activations,
            tile_idx_to_expert_idx=tile_experts,
            tile_idx_to_mn_limit=limits,
            num_non_exiting_tiles=active_tiles,
            alpha=alpha,
            output=output,
            num_local_experts=num_experts,
            activation_type=activation_type,
            swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
            swiglu_beta=DEFAULT_SWIGLU_BETA,
            swiglu_limit=DEFAULT_SWIGLU_LIMIT,
            situ_beta=None,
            situ_linear_beta=None,
            use_fused_finalize=False,
            permuted_idx_to_expanded_idx=None,
            token_final_scales=None,
            enable_pdl=False,
            tactic=((mma_m, route_tile, 256), (cluster_m, 1), True),
        )
        expected = torch.full_like(output, -123)
        token_experts = torch.arange(valid_rows, device="cuda") // route_tile
        values = activations[:valid_rows].float().sum(dim=1)[:, None]
        values = values * row_scales[token_experts] * alpha[token_experts, None]
        if activation_type == ActivationType.Relu2:
            values = values.relu().square()
        expected[:valid_rows] = values.to(torch.bfloat16)
        # Includes the untouched invalid rows, so writing stale accumulator
        # columns cannot pass merely because the live prefix was correct.
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "mma_m,route_tile,activation_type,use_fused_finalize",
    [
        pytest.param(128, 64, ActivationType.Swiglu, False, id="1cta-swiglu-unfused"),
        pytest.param(128, 64, ActivationType.GegluTanh, True, id="1cta-geglu-fused"),
        pytest.param(256, 128, ActivationType.Swiglu, True, id="2cta-swiglu-fused"),
        pytest.param(
            256, 128, ActivationType.GegluTanh, False, id="2cta-geglu-unfused"
        ),
        pytest.param(256, 64, ActivationType.Relu2, True, id="2cta-relu2-fused"),
        pytest.param(
            256, 192, ActivationType.Relu2, False, id="2cta-n192-relu2-unfused"
        ),
    ],
)
def test_dynamic_n_graph_routing(
    mma_m, route_tile, activation_type, use_fused_finalize
):
    from flashinfer.cute_dsl.utils import convert_sf_from_mma_layout
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import launch_w4a16_moe
    from flashinfer.quantization import e2m1_and_ufp8sf_scale_to_float

    num_tokens, hidden_size, intermediate_size = 129, 256, 512
    num_experts, top_k = 4, 2
    gated = activation_type != ActivationType.Relu2
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
        gated=gated,
        use_nontrivial_alphas=False,
    )
    tensors["token_final_scales"][:] = torch.tensor([0.25, 0.75], device="cuda")
    routing = tensors["token_selected_experts"]
    output = torch.empty_like(tensors["x_bf16"])
    gemm_tactic = ((mma_m, route_tile, 256), (2, 1), True)

    def run():
        return launch_w4a16_moe(
            x=tensors["x_bf16"],
            token_selected_experts=routing,
            token_final_scales=tensors["token_final_scales"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            num_experts=num_experts,
            num_local_experts=num_experts,
            local_expert_offset=0,
            moe_output=output,
            use_fused_finalize=use_fused_finalize,
            enable_pdl=False,
            activation_type=activation_type,
            tactic=(gemm_tactic, gemm_tactic),
        )

    def dequantize(name):
        packed = tensors[f"{name}_weight"]
        experts, rows, packed_k = packed.shape
        scales = convert_sf_from_mma_layout(
            tensors[f"{name}_weight_sf"], rows, packed_k * 2, experts
        )
        return (
            e2m1_and_ufp8sf_scale_to_float(
                packed.reshape(-1, packed_k).cpu(),
                scales.view(torch.uint8).flatten().cpu(),
                torch.ones(1),
                sf_vec_size=16,
                ufp8_type=1,
                is_sf_swizzled_layout=True,
            )
            .reshape(experts, rows, packed_k * 2)
            .cuda()
            .bfloat16()
            .float()
        )

    # Use the actual quantized weights, not their pre-quantization inputs.
    w1, w2 = dequantize("w1"), dequantize("w2")
    if gated:
        w1 = w1.reshape(num_experts, intermediate_size // 64, 2, 64, hidden_size)
        w1 = w1.transpose(1, 2).reshape(num_experts, 2 * intermediate_size, hidden_size)
    fc1 = torch.matmul(tensors["x_bf16"].float(), w1.transpose(1, 2))
    if gated:
        up, gate = fc1.chunk(2, dim=-1)
        if activation_type == ActivationType.Swiglu:
            intermediate = up * torch.nn.functional.silu(gate)
        else:
            intermediate = up * torch.nn.functional.gelu(gate, approximate="tanh")
    else:
        intermediate = fc1.relu().square()
    fc2 = torch.bmm(intermediate.bfloat16().float(), w2.transpose(1, 2))
    scores = torch.tensor([0.25, 0.25, 0.75, 0.75], device="cuda")[:, None, None]
    if use_fused_finalize:
        contributions = (fc2 * scores).bfloat16().float()
    else:
        contributions = fc2.bfloat16().float() * scores
    reference_pairs = torch.stack(
        [contributions[0] + contributions[2], contributions[1] + contributions[3]]
    ).bfloat16()

    routing[:, 0] = 0
    routing[:, 1] = 2
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # Expert counts change without changing the graph, tensor sizes or addresses.
    # Start large, shrink over instruction boundaries, then grow to a full tile.
    for split in (129, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129):
        choices = (torch.arange(num_tokens, device="cuda") >= split).to(torch.int32)
        routing[:, 0] = choices
        routing[:, 1] = choices + 2
        output.fill_(float("nan"))
        graph.replay()
        captured = output.clone()
        expected = reference_pairs[
            choices.long(), torch.arange(num_tokens, device="cuda")
        ]
        torch.testing.assert_close(captured, expected, rtol=0.015, atol=0.005)
        output.fill_(float("nan"))
        run()
        # Two contributions per output make the BF16 atomic sum independent
        # of route order, so replay and eager execution must agree bit for bit.
        torch.testing.assert_close(captured, output, rtol=0, atol=0)
