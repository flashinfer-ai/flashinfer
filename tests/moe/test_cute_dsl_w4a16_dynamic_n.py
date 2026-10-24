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
from tests.test_helpers.w4a16_reference import (
    assert_w4a16_bits,
    canonical_split_w4a16_weights,
    combine_w4a16_split_terms,
    w4a16_reference,
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
        assert_w4a16_bits(output, expected)


@pytest.mark.parametrize(
    "mma_m,route_tile,activation_type,use_fused_finalize,activation_options,nonlocal_routes",
    [
        pytest.param(
            128, 64, ActivationType.Swiglu, False, {}, False, id="1cta-swiglu-unfused"
        ),
        pytest.param(
            128, 64, ActivationType.GegluTanh, True, {}, False, id="1cta-geglu-fused"
        ),
        pytest.param(
            256, 128, ActivationType.Swiglu, True, {}, False, id="2cta-swiglu-fused"
        ),
        pytest.param(
            256,
            128,
            ActivationType.GegluTanh,
            False,
            {},
            False,
            id="2cta-geglu-unfused",
        ),
        pytest.param(
            256, 64, ActivationType.Relu2, True, {}, False, id="2cta-relu2-fused"
        ),
        pytest.param(
            256,
            192,
            ActivationType.Relu2,
            False,
            {},
            False,
            id="2cta-n192-relu2-unfused",
        ),
        pytest.param(
            128,
            64,
            ActivationType.Swiglu,
            False,
            {"swiglu_alpha": 1.7, "swiglu_beta": 0.2, "swiglu_limit": 0.5},
            False,
            id="1cta-parameterized-swiglu-unfused",
        ),
        pytest.param(
            256,
            128,
            ActivationType.Swiglu,
            True,
            {"situ_beta": 1.3, "situ_linear_beta": 2.7},
            False,
            id="2cta-situ-fused",
        ),
        pytest.param(
            256,
            64,
            ActivationType.Swiglu,
            False,
            {},
            True,
            id="2cta-swiglu-nonlocal-unfused",
        ),
    ],
)
def test_dynamic_n_graph_routing(
    mma_m,
    route_tile,
    activation_type,
    use_fused_finalize,
    activation_options,
    nonlocal_routes,
):
    _check_dynamic_n_graph_routing(
        mma_m,
        route_tile,
        activation_type,
        use_fused_finalize,
        activation_options,
        nonlocal_routes,
    )


def test_dynamic_n_model_width_graph_routing():
    _check_dynamic_n_graph_routing(
        256,
        64,
        ActivationType.Swiglu,
        False,
        {},
        False,
        problem_shape=(17, 7168, 2048),
    )


def _check_dynamic_n_graph_routing(
    mma_m,
    route_tile,
    activation_type,
    use_fused_finalize,
    activation_options,
    nonlocal_routes,
    *,
    problem_shape=(129, 256, 512),
):
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import launch_w4a16_moe

    num_tokens, hidden_size, intermediate_size = problem_shape
    num_local_experts, top_k = 4, 2
    num_experts = 8 if nonlocal_routes else num_local_experts
    local_expert_offset = 2 if nonlocal_routes else 0
    gated = activation_type != ActivationType.Relu2
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        top_k=top_k,
        gated=gated,
        use_nontrivial_alphas=True,
        vary_block_scales=True,
    )
    # FP32 weights and alphas exercise both the FC2 epilogue's scale order and
    # the different BF16 rounding points of fused and unfused finalization.
    tensors["token_final_scales"][:] = torch.tensor([0.3127, 0.6873], device="cuda")
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
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            moe_output=output,
            use_fused_finalize=use_fused_finalize,
            enable_pdl=False,
            activation_type=activation_type,
            tactic=(gemm_tactic, gemm_tactic),
            **activation_options,
        )

    weights = canonical_split_w4a16_weights(tensors, gated)
    activation = {
        ActivationType.Swiglu: "swiglu",
        ActivationType.GegluTanh: "geglu_tanh",
        ActivationType.Relu2: "relu2",
    }[activation_type]
    if activation_options.get("situ_beta") is not None:
        activation = "situ"
    route_pairs = [
        (local_expert_offset, local_expert_offset + 2),
        (local_expert_offset + 1, local_expert_offset + 3),
    ]
    if nonlocal_routes:
        route_pairs += [(0, local_expert_offset + 2), (0, 1)]
    route_pairs = torch.tensor(route_pairs, dtype=torch.int32, device="cuda")
    reference_pairs = []
    for pair in route_pairs:
        ids = pair.expand_as(routing)
        reference_options = dict(activation_options)
        reference_options["gate_up_clamp"] = reference_options.pop("swiglu_limit", None)
        reference_pairs.append(
            w4a16_reference(
                {
                    "hidden_states": tensors["x_bf16"],
                    "topk_ids": ids,
                    "topk_weights": tensors["token_final_scales"],
                    "activation": activation,
                    **reference_options,
                },
                weights,
                fc1_alpha=tensors["w1_alpha"],
                fc2_alpha=tensors["w2_alpha"],
                backend="split",
                local_expert_offset=local_expert_offset,
                use_fused_finalize=use_fused_finalize,
            )
        )
    reference_pairs = torch.stack(reference_pairs)

    routing[:] = route_pairs[0]
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # Expert counts change without changing the graph, tensor sizes or addresses.
    # Start large, shrink over instruction boundaries, then grow to a full tile.
    boundaries = (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128)
    splits = (num_tokens, *(n for n in boundaries if n < num_tokens), num_tokens)
    for split in splits:
        choices = (torch.arange(num_tokens, device="cuda") >= split).to(torch.int32)
        if nonlocal_routes:
            choices = choices * 2 + torch.arange(num_tokens, device="cuda") % 2
        routing[:] = route_pairs[choices.long()]
        output.fill_(float("nan"))
        graph.replay()
        captured = output.clone()
        expected = reference_pairs[
            choices.long(), torch.arange(num_tokens, device="cuda")
        ]
        assert_w4a16_bits(captured, expected)
        output.fill_(float("nan"))
        run()
        # Two contributions per output make the BF16 atomic sum independent
        # of route order, so replay and eager execution must agree bit for bit.
        assert_w4a16_bits(captured, output)


@pytest.mark.parametrize("input_is_expanded", [False, True])
def test_split_finalize_fma_contract(input_is_expanded):
    from flashinfer.fused_moe.cute_dsl.blackwell.moe_finalize import moe_unpermute

    pytest.importorskip("triton")
    num_tokens, top_k, hidden_size = 8, 8, 8
    terms = torch.full(
        (num_tokens, top_k, hidden_size),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    scores = torch.full((num_tokens, top_k), float("nan"), device="cuda")
    local = torch.zeros((num_tokens, top_k), dtype=torch.bool, device="cuda")

    # The exact residual is 2**-30. Separately rounding the final multiply
    # before the add instead produces zero. Visit all eight route slots.
    local[0] = True
    terms[0] = 0
    scores[0] = 0.314159
    terms[0, 0], scores[0, 0] = -1, 1 + 2**-7 + 2**-23
    terms[0, 7], scores[0, 7] = 1 + 2**-7, 1 + 2**-23
    # Both BF16 and FP32 subnormal operands are preserved, as are subnormal
    # products; the final BF16 conversion must not silently flush them either.
    local[1:5, 3] = True
    terms[1, 3], scores[1, 3] = 2**-133, 2**20
    terms[2, 3], scores[2, 3] = 2**30, 2**-140
    terms[3, 3], scores[3, 3] = 2**-120, 2**-10
    terms[4, 3], scores[4, 3] = -(2**-120), 2**-10
    # Slot zero uses multiply, preserving -0. A later -0 uses FMA with +0.
    # The all-nonlocal row remains +0 without reading its NaN inputs.
    local[5, 0] = local[7, 3] = True
    terms[5, 0], scores[5, 0] = -0.0, 1.0
    terms[7, 3], scores[7, 3] = -0.0, 1.0
    expected = torch.zeros(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    expected[0] = 2**-30
    expected[1] = 2**-113
    expected[2] = 2**-110
    expected[3] = 2**-130
    expected[4] = -(2**-130)
    expected[5] = -0.0
    reference = combine_w4a16_split_terms(terms, scores, local)
    assert_w4a16_bits(reference, expected)

    expanded = terms.reshape(num_tokens * top_k, hidden_size)
    route_rows = torch.arange(num_tokens * top_k, device="cuda", dtype=torch.int32)
    if input_is_expanded:
        expert_output = expanded.contiguous()
        mapping = route_rows.reshape(num_tokens, top_k)
    else:
        # A nonidentity permutation makes addressing errors observable too.
        expert_output = expanded.flip(0).contiguous()
        mapping = route_rows.flip(0).reshape(num_tokens, top_k)
    mapping = mapping.masked_fill(~local, -1).contiguous()
    output = torch.full_like(expected, float("nan"))
    moe_unpermute(
        expert_output,
        output,
        mapping,
        scores,
        num_tokens,
        top_k,
        input_is_expanded=input_is_expanded,
    )
    assert_w4a16_bits(output, reference)
