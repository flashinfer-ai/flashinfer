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

from flashinfer.utils import get_compute_capability

from tests.moe.trtllm_gen_fused_moe_utils import (
    ActivationType,
    BF16Moe,
    FP4Moe,
    FP8BlockScaleMoe,
    FP8PerTensorMoe,
    Fp8QuantizationType,
    MxInt4BlockScaleMoe,
    QuantMode,
    RoutingMethodType,
    WeightLayout,
    moe_args,
    pack_topk_for_routed_moe,
    routing_reference_renormalize,
    run_moe_test,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
)

pytestmark = pytest.mark.long_running


@pytest.fixture(scope="module")
def cache_permute_indices():
    return {}


# Test: Sigmoid routing (Sigmoid -> TopK, no renormalization)
@pytest.mark.parametrize("num_tokens", [8, 768, 3072])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [1024, 768, 512, 384])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(BF16Moe(), id="BF16xBF16"),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="FP8_Block_DeepSeek",
        ),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8),
            id="FP8_Block_MxFp8",
        ),
        pytest.param(FP8PerTensorMoe(), id="FP8_Tensor"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4), id="NvFP4xNvFP4"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8), id="MxFP4xMxFP8"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_Bf16), id="MxFP4xBf16"),
        pytest.param(MxInt4BlockScaleMoe(), id="MxInt4xBf16"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Sigmoid,
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384, 768, 1024],
                "enable_autotune": True,
            },
            id="Sigmoid_128e_top8",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="NoShuffle_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.BlockMajorK,
                "compatible_moe_impls": [
                    FP8BlockScaleMoe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
            },
            id="Shuffled_BlockMajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu, id="Swiglu"),
        pytest.param(ActivationType.Geglu, id="Geglu"),
    ],
)
def test_sigmoid_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
):
    """Test Sigmoid routing configurations (Sigmoid -> TopK, no renormalization)."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
    )


# Test: DeepSeekV3 routing
@pytest.mark.parametrize("num_tokens", [8, 768, 3072])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [2688, 2048, 1024, 768, 512, 384])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(FP8PerTensorMoe(), id="FP8_PerTensor"),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="FP8_Block_DeepSeek",
        ),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8),
            id="FP8_Block_MxFp8",
        ),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4), id="NvFP4xNvFP4"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8), id="MxFP4xMxFP8"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_Bf16), id="MxFP4xBf16"),
        pytest.param(MxInt4BlockScaleMoe(), id="MxInt4xBf16"),
        pytest.param(BF16Moe(), id="Bf16xBf16"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 512,
                "top_k": 22,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [BF16Moe, FP8PerTensorMoe, FP4Moe],
                "compatible_intermediate_size": [2688],
                "compatible_activation_types": [ActivationType.Relu2],
                "enable_autotune": True,
            },
            id="nemotron_3_super",
        ),
        pytest.param(
            {
                "num_experts": 384,
                "top_k": 8,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP4Moe, FP8BlockScaleMoe],
                "compatible_intermediate_size": [1024, 2048],
                "compatible_activation_types": [
                    ActivationType.Swiglu,
                    ActivationType.Geglu,
                ],
                "enable_autotune": True,
            },
            id="kimi_k2",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [
                    FP4Moe,
                    FP8BlockScaleMoe,
                    MxInt4BlockScaleMoe,
                    BF16Moe,
                ],
                "compatible_intermediate_size": [512, 1024, 2048],
                "compatible_activation_types": [
                    ActivationType.Swiglu,
                    ActivationType.Geglu,
                ],
                "enable_autotune": True,
            },
            id="DSv3",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "num_fused_shared_experts": 1,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "compatible_activation_types": [ActivationType.Swiglu],
                "enable_autotune": False,
            },
            id="DSv3_fused_shared_1",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "num_fused_shared_experts": 2,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "compatible_activation_types": [ActivationType.Swiglu],
                "enable_autotune": False,
            },
            id="DSv3_fused_shared_2",
        ),
        pytest.param(
            {
                "num_experts": 72,
                "top_k": 6,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP4Moe, FP8BlockScaleMoe],
                "compatible_intermediate_size": [384, 768],
                "compatible_activation_types": [
                    ActivationType.Swiglu,
                    ActivationType.Geglu,
                ],
                "enable_autotune": False,
            },
            id="DSLite",
        ),
        pytest.param(
            {
                "num_experts": 160,
                "top_k": 8,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP4Moe, FP8BlockScaleMoe, BF16Moe],
                "compatible_intermediate_size": [512, 1024, 1536],
                "compatible_activation_types": [
                    ActivationType.Swiglu,
                    ActivationType.Geglu,
                ],
                "enable_autotune": False,
            },
            id="GLM4_MoE",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="NoShuffle_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.BlockMajorK,
                "compatible_moe_impls": [
                    FP8BlockScaleMoe,
                    MxInt4BlockScaleMoe,
                    BF16Moe,
                ],
            },
            id="Shuffled_BlockMajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu, id="Swiglu"),
        pytest.param(ActivationType.Geglu, id="Geglu"),
        pytest.param(ActivationType.Relu2, id="Relu2"),
    ],
)
@pytest.mark.parametrize(
    "routing_logits_dtype",
    [
        pytest.param(torch.float32, id="FP32_logits"),
    ],
)
def test_deepseekv3_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    routing_logits_dtype,
    cache_permute_indices,
):
    """Test DeepSeekV3 routing configurations."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
        routing_logits_dtype,
    )


# Test: TopK routing
@pytest.mark.parametrize("num_tokens", [8, 128])  # Limited for GeGlu
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [384, 512, 768, 1024])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4), id="NvFP4xNvFP4"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8), id="MxFP4xMxFP8"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 16,
                "top_k": 2,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.TopK,
                "compatible_moe_impls": [FP4Moe],
                "compatible_intermediate_size": [512, 768, 1024],
                "enable_autotune": True,
            },
            id="TopK",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu, id="Swiglu"),
        pytest.param(ActivationType.Geglu, id="Geglu"),
    ],
)
@pytest.mark.parametrize(
    "routing_logits_dtype",
    [
        pytest.param(torch.float32, id="FP32_logits"),
        pytest.param(torch.bfloat16, id="BF16_logits"),
    ],
)
def test_topk_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    routing_logits_dtype,
    cache_permute_indices,
):
    """Test TopK routing configuration."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
        routing_logits_dtype,
    )


# Test: Llama4 routing
@pytest.mark.parametrize("num_tokens", [8, 768, 3072])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(FP8PerTensorMoe(), id="FP8_Tensor"),
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4), id="FP4_MoE"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 1,
                "padding": 8,
                "n_groups": 0,
                "top_k_groups": 0,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.Llama4,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe],
                "compatible_intermediate_size": [1024, 2048],
                "enable_autotune": True,
            },
            id="Llama4",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu, id="Swiglu"),
    ],
)
@pytest.mark.parametrize(
    "routing_logits_dtype",
    [
        pytest.param(torch.bfloat16, id="BF16_logits"),
    ],
)
def test_llama4_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    routing_logits_dtype,
    cache_permute_indices,
):
    """Test Llama4 routing configuration with FP8 per-tensor."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
        routing_logits_dtype,
    )


@pytest.mark.parametrize("num_tokens", [32, 768, 3072])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [2048, 1024, 768, 512])
@pytest.mark.parametrize("bias", ["gemm2", "gemm1", "gemm1_and_gemm2"])
def test_mxfp4_moe_gemm_bias(
    num_tokens, hidden_size, intermediate_size, bias, cache_permute_indices
):
    """Test MXFP4 MoE with GEMM bias support."""
    # TODO NVFP4 is currently broken
    num_experts = 8
    top_k = 2
    device = "cuda"

    gemm1_bias = None
    gemm2_bias = None
    if "gemm1" in bias:
        gemm1_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=device, dtype=torch.float32
        )
    if "gemm2" in bias:
        gemm2_bias = torch.randn(
            (num_experts, hidden_size), device=device, dtype=torch.float32
        )

    run_moe_test(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_impl=FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8),
        routing_config={
            "num_experts": num_experts,
            "top_k": top_k,
            "padding": 8,
            "n_groups": None,
            "top_k_groups": None,
            "routed_scaling": None,
            "has_routing_bias": False,
            "routing_method_type": RoutingMethodType.Renormalize,
            "compatible_moe_impls": [FP4Moe],
            "compatible_intermediate_size": [512, 768, 1024, 2048],
            "enable_autotune": True,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.bfloat16,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
    )


@pytest.mark.parametrize("num_tokens", [32])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize("bias", ["gemm2", "gemm1"])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(FP4Moe(quant_mode=QuantMode.FP4_MXFP4_MXFP8), id="MxFP4xMxFP8"),
    ],
)
def test_fp4_moe_gemm_bias_changes_output(
    num_tokens,
    hidden_size,
    intermediate_size,
    bias,
    moe_impl,
    cache_permute_indices,
):
    """Test FP4 MoE GEMM bias support changes the kernel output."""
    num_experts = 8
    top_k = 2
    device = "cuda"
    routing_config = {
        "num_experts": num_experts,
        "top_k": top_k,
        "padding": 8,
        "n_groups": None,
        "top_k_groups": None,
        "routed_scaling": None,
        "has_routing_bias": False,
        "routing_method_type": RoutingMethodType.Renormalize,
        "compatible_moe_impls": [FP4Moe],
        "compatible_intermediate_size": [512, 768, 1024, 2048],
        "enable_autotune": True,
    }
    weight_processing = {
        "use_shuffled_weight": True,
        "layout": WeightLayout.MajorK,
        "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
    }

    gemm1_bias = None
    gemm2_bias = None
    if "gemm1" in bias:
        gemm1_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=device, dtype=torch.float32
        )
    if "gemm2" in bias:
        gemm2_bias = torch.randn(
            (num_experts, hidden_size), device=device, dtype=torch.float32
        )

    _, output_with_bias, _ = run_moe_test(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_impl=moe_impl,
        routing_config=routing_config,
        weight_processing=weight_processing,
        activation_type=ActivationType.Swiglu,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.bfloat16,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
    )
    _, output_without_bias, _ = run_moe_test(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_impl=moe_impl,
        routing_config=routing_config,
        weight_processing=weight_processing,
        activation_type=ActivationType.Swiglu,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.bfloat16,
    )

    # Sanity check to ensure the bias is actually changing the output
    # If the weights and activations are too large we might not see a difference which would invalidate the tests
    # Also useful for debugging if the bias is skipped vs incorrect
    assert not torch.allclose(
        output_with_bias, output_without_bias, atol=1e-3, rtol=1e-3
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 64, 256, 1000, 4000])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("intermediate_size", [512, 1024])
@pytest.mark.parametrize(
    "zero_hidden_states",
    [
        pytest.param(True, id="ZeroHiddenStates"),
        pytest.param(False, id="RandomHiddenStates"),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 32,
                "top_k": 4,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512, 1024],
                "compatible_activation_types": [ActivationType.Relu2],
                "enable_autotune": False,
            },
            id="E32_K4",
        ),
        pytest.param(
            {
                "num_experts": 64,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512, 1024],
                "compatible_activation_types": [ActivationType.Relu2],
                "enable_autotune": False,
            },
            id="E64_K8",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
    ],
)
def test_mxfp8_block_scale_moe_relu2_non_gated(
    num_tokens,
    hidden_size,
    intermediate_size,
    zero_hidden_states,
    routing_config,
    weight_processing,
    cache_permute_indices,
):
    """Test MXFP8 block-scale TRTLLM MoE with non-gated RELU2."""
    run_moe_test(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_impl=FP8BlockScaleMoe(
            fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8
        ),
        routing_config=routing_config,
        weight_processing=weight_processing,
        activation_type=ActivationType.Relu2,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.bfloat16,
        zero_hidden_states=zero_hidden_states,
    )


def test_mxfp8_block_scale_moe_relu2_deepseekv3_topk22(cache_permute_indices):
    """Targeted coverage for MXFP8 non-gated Relu2 with DeepSeekV3 routing top_k=22."""
    run_moe_test(
        num_tokens=128,
        hidden_size=1024,
        intermediate_size=512,
        moe_impl=FP8BlockScaleMoe(
            fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8
        ),
        routing_config={
            # top_k=22 is only supported when num_experts > NumKimiK2Experts (384)
            "num_experts": 512,
            "top_k": 22,
            "padding": 8,
            "n_groups": 1,
            "top_k_groups": 1,
            "routed_scaling": 2.5,
            "has_routing_bias": True,
            "routing_method_type": RoutingMethodType.DeepSeekV3,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_intermediate_size": [512],
            "compatible_activation_types": [ActivationType.Relu2],
            "enable_autotune": False,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP8BlockScaleMoe],
        },
        activation_type=ActivationType.Relu2,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.float32,
    )


@pytest.mark.parametrize(
    "autotune_case",
    [
        pytest.param(
            {
                "num_tokens": 1,
                "hidden_size": 1024,
                "intermediate_size": 1024,
                "quant_mode": QuantMode.FP8_BLOCK_SCALE_MXFP8,
                "activation_type": ActivationType.Relu2,
                "num_experts": 64,
                "top_k": 8,
                "routing_method_type": RoutingMethodType.Renormalize,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
            },
            id="MxFp8_Relu2_T1_H1024_I1024_K8",
        ),
        pytest.param(
            {
                "num_tokens": 64,
                "hidden_size": 512,
                "intermediate_size": 512,
                "quant_mode": QuantMode.FP8_BLOCK_SCALE_MXFP8,
                "activation_type": ActivationType.Relu2,
                "num_experts": 512,
                "top_k": 22,
                # top_k=22 is only valid on DeepSeekV3 routing path for large expert counts.
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
            },
            id="MxFp8_Relu2_T64_H512_I512_K22",
        ),
        pytest.param(
            {
                "num_tokens": 256,
                "hidden_size": 1024,
                "intermediate_size": 512,
                "quant_mode": QuantMode.FP8_BLOCK_SCALE_DEEPSEEK,
                "activation_type": ActivationType.Swiglu,
                "num_experts": 256,
                "top_k": 8,
                "routing_method_type": RoutingMethodType.Renormalize,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
            },
            id="DeepSeek_Swiglu_T256_H1024_I512_K8",
        ),
    ],
)
def test_fp8_block_scale_autotune_valid_configs(autotune_case, cache_permute_indices):
    """Autotune smoke matrix to exercise C++ getValidConfigs across FP8 modes/shapes."""
    run_moe_test(
        num_tokens=autotune_case["num_tokens"],
        hidden_size=autotune_case["hidden_size"],
        intermediate_size=autotune_case["intermediate_size"],
        moe_impl=FP8BlockScaleMoe(fp8_quantization_type=autotune_case["quant_mode"]),
        routing_config={
            "num_experts": autotune_case["num_experts"],
            "top_k": autotune_case["top_k"],
            "padding": 8,
            "n_groups": autotune_case["n_groups"],
            "top_k_groups": autotune_case["top_k_groups"],
            "routed_scaling": autotune_case["routed_scaling"],
            "has_routing_bias": autotune_case["has_routing_bias"],
            "routing_method_type": autotune_case["routing_method_type"],
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_intermediate_size": [autotune_case["intermediate_size"]],
            "compatible_activation_types": [autotune_case["activation_type"]],
            "enable_autotune": True,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP8BlockScaleMoe],
        },
        activation_type=autotune_case["activation_type"],
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.float32,
        zero_hidden_states=False,
    )


@pytest.mark.parametrize(
    "autotune_case",
    [
        pytest.param(
            {
                "num_tokens": 64,
                "hidden_size": 1024,
                "intermediate_size": 1024,
                "activation_type": ActivationType.Swiglu,
                "num_experts": 64,
                "top_k": 8,
            },
            id="PerTensor_Swiglu_T64_H1024_I1024_K8",
        ),
        pytest.param(
            {
                "num_tokens": 32,
                "hidden_size": 512,
                "intermediate_size": 512,
                "activation_type": ActivationType.Relu2,
                "num_experts": 64,
                "top_k": 8,
            },
            id="PerTensor_Relu2_T32_H512_I512_K8",
        ),
    ],
)
def test_fp8_per_tensor_autotune_valid_configs_nonefp8(
    autotune_case, cache_permute_indices
):
    """Exercise per-tensor autotune path that uses NoneFp8 in valid-config dispatch."""
    run_moe_test(
        num_tokens=autotune_case["num_tokens"],
        hidden_size=autotune_case["hidden_size"],
        intermediate_size=autotune_case["intermediate_size"],
        moe_impl=FP8PerTensorMoe(),
        routing_config={
            "num_experts": autotune_case["num_experts"],
            "top_k": autotune_case["top_k"],
            "padding": 8,
            "n_groups": None,
            "top_k_groups": None,
            "routed_scaling": None,
            "has_routing_bias": False,
            "routing_method_type": RoutingMethodType.Renormalize,
            "compatible_moe_impls": [FP8PerTensorMoe],
            "compatible_intermediate_size": [autotune_case["intermediate_size"]],
            "compatible_activation_types": [autotune_case["activation_type"]],
            "enable_autotune": True,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP8PerTensorMoe],
        },
        activation_type=autotune_case["activation_type"],
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.bfloat16,
        zero_hidden_states=False,
    )


@pytest.mark.parametrize(
    "num_tokens",
    [5, 8, 12, 16],
    ids=lambda t: f"T{t}",
)
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="FP8_Block_DeepSeek",
        ),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 64,
                "top_k": 4,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="Renormalize_64e_top4",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="NoShuffle_MajorK",
        ),
    ],
)
@pytest.mark.parametrize("activation_type", [ActivationType.Swiglu])
def test_dyn_block_kernel_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
):
    """Test token counts 5-16 that exercise the dynamic block kernel path (BlockKernelMaxNumTokens < tokens <= DynBlockKernelMaxNumTokens)."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices=False,
    )


@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="FP8_Block_DeepSeek",
        ),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 1024,
                "top_k": 8,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 1.0,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="DeepSeekV3_1024e_top8",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="NoShuffle_MajorK",
        ),
    ],
)
@pytest.mark.parametrize("activation_type", [ActivationType.Swiglu])
def test_tier_1024_experts_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
):
    """Test 1024-expert routing to exercise Tier<1024, 32> in SigmoidBias+ScaledSumNormalize policy."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices=False,
    )


# num_tokens is chosen to straddle the dispatch thresholds in routingCustom::run
# (see trtllm_fused_moe_routing_custom.cu):
#   - tokens == 8  : dyn-block kernel path (tokens <= DynBlockKernelMaxNumTokens=16,
#                    numExperts <= DynBlockKernelMaxNumExperts=512)
#   - tokens == 32 : block-per-token "split" path on the single-cluster kernel
#                    (17 <= tokens <= 256, numExperts >= 160, policy pair opts into
#                    PolicyPairSupportsBlockPerToken) — exercises the
#                    routingIndicesBlockScoresKernel path added for this feature.
@pytest.mark.parametrize("num_tokens", [8, 32])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="FP8_Block_DeepSeek",
        ),
        pytest.param(BF16Moe(), id="BF16xBF16"),
        pytest.param(FP8PerTensorMoe(), id="FP8_PerTensor"),
        pytest.param(
            FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
            id="NvFP4xNvFP4",
        ),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        # DeepSeekV3 + nGroup == 1 routes through routingCustom with
        # (SigmoidBiasPreprocess, ScaledSumNormalizePostprocess) — the policy pair
        # that opts into the block-per-token BlockScores kernel.
        pytest.param(
            {
                "num_experts": 384,
                "top_k": 6,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [
                    FP8BlockScaleMoe,
                    BF16Moe,
                    FP8PerTensorMoe,
                    FP4Moe,
                ],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="DeepSeekV3_ngroup1_384e_top6",
        ),
        pytest.param(
            {
                # top_k=22 requires num_experts > NumKimiK2Experts (384).
                "num_experts": 512,
                "top_k": 22,
                "padding": 8,
                "n_groups": 1,
                "top_k_groups": 1,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [
                    FP8BlockScaleMoe,
                    BF16Moe,
                    FP8PerTensorMoe,
                    FP4Moe,
                ],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="DeepSeekV3_ngroup1_512e_top22",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="NoShuffle_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP4Moe, FP8PerTensorMoe, FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.BlockMajorK,
                "compatible_moe_impls": [
                    FP8BlockScaleMoe,
                    MxInt4BlockScaleMoe,
                    BF16Moe,
                ],
            },
            id="Shuffled_BlockMajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [
        pytest.param(ActivationType.Swiglu, id="Swiglu"),
        pytest.param(ActivationType.Relu2, id="Relu2"),
    ],
)
def test_deepseek_ngroup1_block_per_token_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
):
    """Exercise the block-per-token BlockScores kernel in routingCustom.

    DeepSeekV3 with n_group == 1 dispatches to routingCustom with the
    (SigmoidBiasPreprocess, ScaledSumNormalizePostprocess) policy pair, which opts
    into PolicyPairSupportsBlockPerToken. For num_experts >= 160 and
    17 <= num_tokens <= 256, routingIndicesBlockScoresKernel replaces the
    fused single-cluster kernel. We intentionally use independent
    parametrization here; incompatible combinations are filtered by skip_checks.

    Covered tiers:
      - Tier<384, 8>  via num_experts=384, top_k=6
      - Tier<512, 22> via num_experts=512, top_k=22
    """
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices=cache_permute_indices,
    )


@pytest.mark.parametrize("num_tokens", [8])
@pytest.mark.parametrize("hidden_size", [512])
@pytest.mark.parametrize("intermediate_size", [512])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="FP8_Block_DeepSeek",
        ),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 6,
                "padding": 8,
                "n_groups": 8,
                "top_k_groups": 4,
                "routed_scaling": 2.5,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="DeepSeekV3_256e",
        ),
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="Renormalize_128e",
        ),
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Default,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="Default_128e",
        ),
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.SigmoidRenorm,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="SigmoidRenorm_128e",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 6,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.MiniMax2,
                "compatible_moe_impls": [FP8BlockScaleMoe],
                "compatible_intermediate_size": [512],
                "enable_autotune": False,
            },
            id="MiniMax2_256e",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": False,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="NoShuffle_MajorK",
        ),
    ],
)
@pytest.mark.parametrize("activation_type", [ActivationType.Swiglu])
@pytest.mark.parametrize(
    "routing_logits_dtype",
    [
        pytest.param(torch.bfloat16, id="BF16_logits"),
        pytest.param(torch.float32, id="FP32_logits"),
    ],
)
@pytest.mark.parametrize(
    "routing_bias_dtype",
    [
        pytest.param(None, id="default_bias"),
        pytest.param(torch.float32, id="FP32_bias"),
    ],
)
def test_routing_dtype_flexibility(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    routing_logits_dtype,
    routing_bias_dtype,
):
    """Test that routing works with both bfloat16 and float32 logits/bias across all routing methods."""
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices=False,
        routing_logits_dtype=routing_logits_dtype,
        routing_bias_dtype=routing_bias_dtype,
    )


def test_bf16_moe_swiglu_oa_activation_param_validation():
    """BF16 SwiGLU OA params are rejected before dispatch for non-SwiGLU activations."""
    kwargs = {
        "routing_logits": torch.empty((1, 1), dtype=torch.bfloat16),
        "routing_bias": None,
        "hidden_states": torch.empty((1, 128), dtype=torch.bfloat16),
        "gemm1_weights": torch.empty((1, 2, 128), dtype=torch.bfloat16),
        "gemm2_weights": torch.empty((1, 128, 1), dtype=torch.bfloat16),
        "num_experts": 1,
        "top_k": 1,
        "n_group": None,
        "topk_group": None,
        "intermediate_size": 1,
        "local_expert_offset": 0,
        "local_num_experts": 1,
        "routed_scaling_factor": None,
        "routing_method_type": RoutingMethodType.Renormalize.value,
        "activation_type": ActivationType.Geglu.value,
    }
    per_expert = torch.ones((1,), dtype=torch.float32)

    with pytest.raises(ValueError, match=r"ActivationType\.Swiglu"):
        trtllm_bf16_moe(**kwargs, gemm1_alpha=per_expert)

    routed_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in ("routing_logits", "routing_bias")
    }
    routed_kwargs["topk_ids"] = torch.empty((1, 1), dtype=torch.int32)

    with pytest.raises(ValueError, match=r"ActivationType\.Swiglu"):
        trtllm_bf16_routed_moe(**routed_kwargs, gemm1_clamp_limit=per_expert)


def test_bf16_moe_swiglu_oa_activation_params(cache_permute_indices):
    """TRT-LLM Gen BF16 MoE applies raw fused FC1 SwiGLU OA params."""
    if not torch.cuda.is_available():
        pytest.skip("TRT-LLM Gen BF16 MoE test requires CUDA.")
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("TRT-LLM Gen BF16 MoE requires SM10.x.")

    num_experts = 64
    num_tokens = 8
    hidden_size = 512
    intermediate_size = 512
    top_k = 1
    padding = 8
    routing_method_type = RoutingMethodType.Renormalize
    weight_processing = {
        "use_shuffled_weight": True,
        "layout": WeightLayout.BlockMajorK,
    }

    selected_experts = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    routing_logits = torch.full(
        (num_tokens, num_experts), -80.0, device="cuda", dtype=torch.bfloat16
    )
    routing_logits[torch.arange(num_tokens, device="cuda"), selected_experts] = 80.0
    permute_info, scores = routing_reference_renormalize(
        routing_logits, top_k, num_experts, padding
    )

    hidden_states = torch.zeros(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    hidden_states[:, 0] = torch.tensor(
        [-3.0, -1.0, 0.25, 1.0, 3.0, -4.0, 2.0, 0.5],
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden_states[:, 1] = torch.tensor(
        [-3.0, -0.5, 0.75, 2.5, 4.0, 6.0, -8.0, 1.5],
        device="cuda",
        dtype=torch.bfloat16,
    )

    gemm1_weights = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.zeros(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for expert_idx in range(num_tokens):
        gemm1_weights[expert_idx, 0, 0] = 1.0
        gemm1_weights[expert_idx, intermediate_size, 1] = 1.0
        gemm2_weights[expert_idx, 0, 0] = 1.0

    moe_impl = BF16Moe()
    moe_impl._cache_permute_indices = cache_permute_indices

    def per_expert(value):
        return torch.full((num_experts,), value, device="cuda", dtype=torch.float32)

    def run_case(gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None):
        weights_data = moe_impl.quantize_weights(
            gemm1_weights, gemm2_weights, hidden_states
        )
        inputs_data = moe_impl.quantize_inputs(
            hidden_states, weights_data["hidden_states_scale_global"]
        )
        quant_data = {**weights_data, **inputs_data}
        args = moe_args(
            num_tokens,
            num_experts,
            hidden_size,
            intermediate_size,
            top_k,
            padding,
            quant_data["hidden_states"],
            quant_data["hidden_states_scale"],
            quant_data["hidden_states_scale_global"],
            scores,
            quant_data["gemm1_weights"],
            quant_data["gemm1_scales"],
            quant_data["gemm1_scales_global"],
            quant_data["gemm2_weights"],
            quant_data["gemm2_scales"],
            quant_data["gemm2_scales_global"],
            permute_info,
            False,
            ActivationType.Swiglu,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )
        output_ref, args_dequant = moe_impl.compute_reference(args)
        output_actual = moe_impl.compute_production(
            args_dequant,
            args,
            expert_logits=routing_logits,
            routing_bias=None,
            hidden_states_orig=hidden_states,
            gemm1_weights_orig=gemm1_weights,
            gemm2_weights_orig=gemm2_weights,
            n_groups=None,
            top_k_groups=None,
            routed_scaling=None,
            routing_method_type=routing_method_type,
            weight_processing=weight_processing,
            enable_pdl=True,
            hidden_states_quant=inputs_data["hidden_states"],
            enable_autotune=False,
            norm_topk_prob=True,
        )
        return output_ref.to(torch.float), output_actual.to(torch.float)

    output_default_ref, output_default = run_case()
    output_noop_ref, output_noop = run_case(
        gemm1_alpha=per_expert(1.0),
        gemm1_beta=per_expert(0.0),
        gemm1_clamp_limit=per_expert(1.0e9),
    )

    alpha = per_expert(1.702)
    beta = per_expert(1.0)
    clamp_limit = per_expert(2.0)
    output_oa_ref, output_oa = run_case(
        gemm1_alpha=alpha,
        gemm1_clamp_limit=clamp_limit,
    )
    output_beta_oa_ref, output_beta_oa = run_case(
        gemm1_alpha=alpha,
        gemm1_beta=beta,
        gemm1_clamp_limit=clamp_limit,
    )

    torch.testing.assert_close(output_default, output_default_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(output_noop, output_noop_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(output_oa, output_oa_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(output_beta_oa, output_beta_oa_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(output_default, output_noop, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(output_default, output_oa, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(output_oa, output_beta_oa, atol=1e-2, rtol=1e-2)


def test_fp8_block_scale_routed_activation_type_relu2_smoke():
    """Smoke test routed FP8 block-scale call path with explicit non-gated activation_type."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    torch.manual_seed(0)
    device = torch.device("cuda:0")

    num_tokens = 32
    hidden_size = 512
    intermediate_size = 512
    num_experts = 64
    top_k = 8
    routing_method_type = RoutingMethodType.Renormalize
    activation_type = ActivationType.Relu2.value
    fp8_quantization_type = Fp8QuantizationType.MxFp8

    routing_logits = torch.randn((num_tokens, num_experts), device=device).to(
        torch.bfloat16
    )
    hidden_states = torch.randn((num_tokens, hidden_size), device=device).to(
        torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (num_experts, intermediate_size, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (num_experts, hidden_size, intermediate_size),
        device=device,
        dtype=torch.bfloat16,
    )

    quant_impl = FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8)
    quant_weights = quant_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )
    quant_inputs = quant_impl.quantize_inputs(hidden_states)

    output_ref = trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=quant_inputs["hidden_states"],
        hidden_states_scale=quant_inputs["hidden_states_scale"],
        gemm1_weights=quant_weights["gemm1_weights"],
        gemm1_weights_scale=quant_weights["gemm1_scales"],
        gemm2_weights=quant_weights["gemm2_weights"],
        gemm2_weights_scale=quant_weights["gemm2_scales"],
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK.value,
        enable_pdl=True,
        fp8_quantization_type=fp8_quantization_type,
        activation_type=activation_type,
    ).to(torch.float)

    permute_info, expert_weights_full = routing_reference_renormalize(
        routing_logits, top_k, num_experts, 8
    )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_full.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)
    packed_topk_ids = pack_topk_for_routed_moe(topk_ids, expert_weights)

    output_routed = trtllm_fp8_block_scale_routed_moe(
        topk_ids=packed_topk_ids,
        routing_bias=None,
        hidden_states=quant_inputs["hidden_states"],
        hidden_states_scale=quant_inputs["hidden_states_scale"],
        gemm1_weights=quant_weights["gemm1_weights"],
        gemm1_weights_scale=quant_weights["gemm1_scales"],
        gemm2_weights=quant_weights["gemm2_weights"],
        gemm2_weights_scale=quant_weights["gemm2_scales"],
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=routing_method_type.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK.value,
        enable_pdl=True,
        fp8_quantization_type=fp8_quantization_type,
        activation_type=activation_type,
    ).to(torch.float)

    close = torch.isclose(output_ref, output_routed, atol=1e-2, rtol=1e-2)
    mismatch_pct = (~close).float().mean().item() * 100
    assert mismatch_pct < 10, f"Mismatch percentage is {mismatch_pct:.2f}%"


def test_fp8_block_scale_moe_swiglu_oa_activation_param_validation():
    """FP8 block-scale OA params are currently scoped to MxFp8 SwiGLU."""
    kwargs = {
        "routing_logits": torch.empty((1, 1), dtype=torch.bfloat16),
        "routing_bias": None,
        "hidden_states": torch.empty((1, 1), dtype=torch.bfloat16),
        "hidden_states_scale": torch.empty((1, 1), dtype=torch.float32),
        "gemm1_weights": torch.empty((1, 2, 1), dtype=torch.bfloat16),
        "gemm1_weights_scale": torch.empty((1, 1, 1), dtype=torch.float32),
        "gemm2_weights": torch.empty((1, 1, 1), dtype=torch.bfloat16),
        "gemm2_weights_scale": torch.empty((1, 1, 1), dtype=torch.float32),
        "num_experts": 1,
        "top_k": 1,
        "n_group": None,
        "topk_group": None,
        "intermediate_size": 1,
        "local_expert_offset": 0,
        "local_num_experts": 1,
        "routed_scaling_factor": None,
        "routing_method_type": RoutingMethodType.Renormalize.value,
    }
    per_expert = torch.ones((1,), dtype=torch.float32)

    with pytest.raises(ValueError, match="Fp8QuantizationType.MxFp8"):
        trtllm_fp8_block_scale_moe(
            **kwargs,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
            activation_type=ActivationType.Swiglu.value,
            gemm1_alpha=per_expert,
        )

    with pytest.raises(ValueError, match="ActivationType.Swiglu"):
        trtllm_fp8_block_scale_moe(
            **kwargs,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            activation_type=ActivationType.Geglu.value,
            gemm1_clamp_limit=per_expert,
        )

    routed_kwargs = {
        key: value for key, value in kwargs.items() if key != "routing_logits"
    }
    routed_kwargs["topk_ids"] = torch.empty((1, 1), dtype=torch.int32)

    with pytest.raises(ValueError, match="Fp8QuantizationType.MxFp8"):
        trtllm_fp8_block_scale_routed_moe(
            **routed_kwargs,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
            activation_type=ActivationType.Swiglu.value,
            gemm1_beta=per_expert,
        )

    with pytest.raises(ValueError, match="ActivationType.Swiglu"):
        trtllm_fp8_block_scale_routed_moe(
            **routed_kwargs,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            activation_type=ActivationType.Geglu.value,
            gemm1_alpha=per_expert,
        )


def test_fp8_block_scale_moe_fused_shared_experts_reject_ep():
    """Fused shared experts must reject expert-parallel (EP) configurations.

    The routing kernel maps a shared expert's global id ``num_experts + k`` to a
    weight row as ``global_id - local_expert_offset``, which only lands at the
    intended local slot when all routed experts are local. EP configurations
    (non-zero ``local_expert_offset`` or ``local_num_experts < num_experts``)
    must therefore raise instead of silently producing wrong results. The guard
    is a cheap host-side check, so this test does not require a GPU.
    """
    num_experts = 4
    base_kwargs = {
        "routing_logits": torch.empty((1, num_experts), dtype=torch.bfloat16),
        "routing_bias": None,
        "hidden_states": torch.empty((1, 1), dtype=torch.bfloat16),
        "hidden_states_scale": torch.empty((1, 1), dtype=torch.float32),
        "gemm1_weights": torch.empty((1, 2, 1), dtype=torch.bfloat16),
        "gemm1_weights_scale": torch.empty((1, 1, 1), dtype=torch.float32),
        "gemm2_weights": torch.empty((1, 1, 1), dtype=torch.bfloat16),
        "gemm2_weights_scale": torch.empty((1, 1, 1), dtype=torch.float32),
        "num_experts": num_experts,
        "top_k": 1,
        "n_group": None,
        "topk_group": None,
        "intermediate_size": 1,
        "routed_scaling_factor": None,
        "routing_method_type": RoutingMethodType.DeepSeekV3.value,
        "num_fused_shared_experts": 1,
    }

    # Non-zero local_expert_offset (this rank does not own the first expert).
    with pytest.raises(ValueError, match="expert parallelism"):
        trtllm_fp8_block_scale_moe(
            **base_kwargs, local_expert_offset=2, local_num_experts=num_experts
        )

    # Sharded experts: local_num_experts < num_experts.
    with pytest.raises(ValueError, match="expert parallelism"):
        trtllm_fp8_block_scale_moe(
            **base_kwargs, local_expert_offset=0, local_num_experts=num_experts // 2
        )


def test_mxfp8_block_scale_moe_swiglu_oa_activation_params(cache_permute_indices):
    """TRT-LLM Gen MxFp8 MoE applies raw fused FC1 SwiGLU OA params."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    num_experts = 64
    num_tokens = 8
    hidden_size = 512
    intermediate_size = 512
    top_k = 1
    padding = 8
    routing_method_type = RoutingMethodType.Renormalize
    weight_processing = {
        "use_shuffled_weight": True,
        "layout": WeightLayout.MajorK,
        "compatible_moe_impls": [FP8BlockScaleMoe],
    }

    hidden_states = torch.zeros(
        (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
    )
    hidden_states[:, 0] = torch.tensor(
        [-3.0, -1.0, 0.25, 1.0, 3.0, -4.0, 2.0, 0.5],
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden_states[:, 1] = torch.tensor(
        [-3.0, -0.5, 0.75, 2.5, 4.0, 6.0, -8.0, 1.5],
        device="cuda",
        dtype=torch.bfloat16,
    )

    selected_experts = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    expert_logits = torch.full(
        (num_tokens, num_experts), -80.0, device="cuda", dtype=torch.bfloat16
    )
    expert_logits[torch.arange(num_tokens, device="cuda"), selected_experts] = 80.0
    permute_info, scores = routing_reference_renormalize(
        expert_logits, top_k, num_experts, padding
    )

    gemm1_weights = torch.zeros(
        (num_experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.zeros(
        (num_experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    for expert_idx in range(num_tokens):
        gemm1_weights[expert_idx, 0, 0] = 1.0
        gemm1_weights[expert_idx, intermediate_size, 1] = 1.0
        gemm2_weights[expert_idx, 0, 0] = 1.0

    moe_impl = FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8)
    moe_impl._cache_permute_indices = cache_permute_indices

    def per_expert(value):
        return torch.full((num_experts,), value, device="cuda", dtype=torch.float32)

    def run_case(gemm1_alpha=None, gemm1_beta=None, gemm1_clamp_limit=None):
        weights_data = moe_impl.quantize_weights(
            gemm1_weights, gemm2_weights, hidden_states
        )
        inputs_data = moe_impl.quantize_inputs(
            hidden_states, weights_data["hidden_states_scale_global"]
        )
        quant_data = {**weights_data, **inputs_data}
        args = moe_args(
            num_tokens,
            num_experts,
            hidden_size,
            intermediate_size,
            top_k,
            padding,
            quant_data["hidden_states"],
            quant_data["hidden_states_scale"],
            quant_data["hidden_states_scale_global"],
            scores,
            quant_data["gemm1_weights"],
            quant_data["gemm1_scales"],
            quant_data["gemm1_scales_global"],
            quant_data["gemm2_weights"],
            quant_data["gemm2_scales"],
            quant_data["gemm2_scales_global"],
            permute_info,
            False,
            ActivationType.Swiglu,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

        output_ref, args_dequant = moe_impl.compute_reference(args)
        static_data = moe_impl.prepare_static_weights_for_kernel(
            args_dequant,
            args,
            gemm1_weights,
            gemm2_weights,
            hidden_size,
            intermediate_size,
            num_experts,
            weight_processing,
        )
        output_actual = moe_impl.call_moe(
            static_data,
            hidden_states,
            None,
            expert_logits=expert_logits,
            routing_bias=None,
            num_experts=num_experts,
            top_k=top_k,
            n_groups=None,
            top_k_groups=None,
            intermediate_size=intermediate_size,
            routed_scaling=None,
            routing_method_type=routing_method_type,
            do_finalize=True,
            activation_type=ActivationType.Swiglu,
            hidden_states_scale=inputs_data["hidden_states_scale"],
            hidden_states_quant=inputs_data["hidden_states"],
            enable_autotune=False,
            enable_pdl=True,
            gemm1_bias=None,
            gemm2_bias=None,
            gemm1_lora_delta=None,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            permute_info=permute_info,
            norm_topk_prob=True,
        )
        return output_ref, output_actual

    output_default_ref, output_default = run_case()
    output_noop_ref, output_noop = run_case(
        gemm1_alpha=per_expert(1.0), gemm1_clamp_limit=per_expert(1.0e9)
    )

    torch.testing.assert_close(output_default, output_default_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(output_noop, output_noop_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(output_default, output_noop, atol=1e-2, rtol=1e-2)

    alpha = per_expert(1.702)
    beta = per_expert(1.0)
    clamp_limit = per_expert(2.0)
    output_oa_ref, output_oa = run_case(
        gemm1_alpha=alpha, gemm1_clamp_limit=clamp_limit
    )
    output_beta_oa_ref, output_beta_oa = run_case(
        gemm1_alpha=alpha, gemm1_beta=beta, gemm1_clamp_limit=clamp_limit
    )

    # Match the existing MXFP8xMXFP8 MoE test tolerance. This deterministic
    # identity-like setup still exercises MxFp8 quantization, block scales, and
    # the fused SwiGLU epilogue, so exact arithmetic parity is not expected.
    torch.testing.assert_close(output_oa, output_oa_ref, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(output_beta_oa, output_beta_oa_ref, atol=1e-1, rtol=1e-1)
    assert not torch.allclose(output_default, output_oa, atol=1e-2, rtol=1e-2)
    assert not torch.allclose(output_oa, output_beta_oa, atol=1e-2, rtol=1e-2)


# ====================================================================================
# MoE LoRA: gemm1_lora_delta
# ====================================================================================
#
# These tests ride on the shared `run_moe_test` path: the delta is threaded
# through moe_args → run_moe_dequant (which applies it before SwiGlu) and the
# matching kernel argument, and the dequant reference vs kernel comparison
# uses the same `check_accuracy` tolerances as the rest of the suite.


@pytest.mark.parametrize("num_tokens", [8, 128])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [1024])
@pytest.mark.parametrize(
    "moe_impl",
    [
        pytest.param(MxInt4BlockScaleMoe(), id="MxInt4xBf16"),
        pytest.param(BF16Moe(), id="BF16xBF16"),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8),
            id="MxFp8",
        ),
        pytest.param(
            FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
            id="DSFp8",
        ),
    ],
)
@pytest.mark.parametrize(
    "routing_config",
    [
        pytest.param(
            {
                "num_experts": 128,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                    FP8BlockScaleMoe,
                ],
                "compatible_intermediate_size": [1024],
                "enable_autotune": False,
            },
            id="Renorm",
        ),
    ],
)
@pytest.mark.parametrize(
    "weight_processing",
    [
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.BlockMajorK,
                "compatible_moe_impls": [
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                    FP8BlockScaleMoe,
                ],
            },
            id="Shuffled_BlockMajorK",
        ),
        pytest.param(
            {
                "use_shuffled_weight": True,
                "layout": WeightLayout.MajorK,
                "compatible_moe_impls": [FP8BlockScaleMoe],
            },
            id="Shuffled_MajorK",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_type",
    [pytest.param(ActivationType.Swiglu, id="Swiglu")],
)
def test_moe_lora_delta(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
):
    """Runs the standard MoE reference/kernel comparison with a non-None
    `gemm1_lora_delta` threaded through run_moe_test.  We compare a zero delta
    against a deterministic non-zero delta on the same routed path so the test
    fails if LoRA is silently dropped from both reference and production."""
    top_k = routing_config["top_k"]
    zero_delta = torch.zeros(
        num_tokens, top_k, 2 * intermediate_size, dtype=torch.bfloat16, device="cuda"
    )
    delta = torch.full_like(zero_delta, 4)

    zero_reference, _, _ = run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
        gemm1_lora_delta=zero_delta,
    )

    delta_reference, _, delta_args_dequant = run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        moe_impl,
        routing_config,
        weight_processing,
        activation_type,
        cache_permute_indices,
        gemm1_lora_delta=delta,
    )

    torch.testing.assert_close(delta_args_dequant.gemm1_lora_delta, delta)
    assert (delta_reference - zero_reference).abs().max().item() > 0.05


def test_fp4_block_scale_deepseekv3_unfinalized_weight_dtype(cache_permute_indices):
    """Regression for #3595.

    With fp32 DeepSeekV3 routing logits and ``do_finalize=False``, the returned
    ``expert_weights`` must be bfloat16: the trtllm-gen routing kernel always
    emits bf16 expert weights, and the FP4 op returns that buffer verbatim.
    Before the fix the buffer was allocated with ``routing_logits.dtype`` (fp32),
    so callers received bf16 data mislabeled as fp32.
    """
    run_moe_test(
        num_tokens=128,
        hidden_size=1024,
        intermediate_size=1024,
        moe_impl=FP4Moe(quant_mode=QuantMode.FP4_NVFP4_NVFP4),
        routing_config={
            "num_experts": 256,
            "top_k": 8,
            "padding": 8,
            "n_groups": 8,
            "top_k_groups": 4,
            "routed_scaling": 2.5,
            "has_routing_bias": True,
            "routing_method_type": RoutingMethodType.DeepSeekV3,
            "compatible_moe_impls": [FP4Moe],
            "compatible_intermediate_size": [1024],
            "compatible_activation_types": [ActivationType.Swiglu],
            "enable_autotune": False,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP4Moe],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.float32,
        verify_unfinalized_weight_dtype=True,
    )
