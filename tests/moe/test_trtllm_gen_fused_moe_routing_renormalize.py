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

from tests.moe.trtllm_gen_fused_moe_common import (
    ActivationType,
    BF16Moe,
    FP4Moe,
    FP8BlockScaleMoe,
    FP8PerTensorMoe,
    MxInt4BlockScaleMoe,
    QuantMode,
    RoutingMethodType,
    WeightLayout,
    run_moe_test,
)


@pytest.fixture(scope="module")
def cache_permute_indices():
    return {}


# Test: Renormalize routing
@pytest.mark.parametrize(
    "zero_hidden_states",
    [
        pytest.param(True, id="ZeroHiddenStates"),
        pytest.param(False, id="RandomHiddenStates"),
    ],
)
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
                "routing_method_type": RoutingMethodType.Renormalize,
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
            id="Qwen3_MOE",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 8,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384, 1024],
                "enable_autotune": False,
            },
            id="Renorm",
        ),
        pytest.param(
            {
                "num_experts": 512,
                "top_k": 10,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [512],
                "enable_autotune": True,
            },
            id="Qwen3_next",
        ),
        pytest.param(
            {
                "num_experts": 2048,
                "top_k": 32,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.Renormalize,
                "compatible_moe_impls": [
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384],
                "enable_autotune": True,
            },
            id="RoutingRenormalize_large_experts",
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
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384, 768, 1024],
                "enable_autotune": False,
            },
            id="Default_128e_top8",
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
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384, 768, 1024],
                "enable_autotune": False,
            },
            id="SigmoidRenorm_128e_top8",
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
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384, 768, 1024],
                "enable_autotune": False,
            },
            id="MiniMax2_256e_top6_no_scale",
        ),
        pytest.param(
            {
                "num_experts": 256,
                "top_k": 6,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": 3.0,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.MiniMax2,
                "compatible_moe_impls": [
                    FP8PerTensorMoe,
                    FP8BlockScaleMoe,
                    FP4Moe,
                    BF16Moe,
                    MxInt4BlockScaleMoe,
                ],
                "compatible_intermediate_size": [384, 768, 1024],
                "enable_autotune": False,
            },
            id="MiniMax2_256e_top6_scale3",
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
@pytest.mark.parametrize(
    "routing_logits_dtype",
    [
        pytest.param(torch.float32, id="FP32_logits"),
        pytest.param(torch.bfloat16, id="BF16_logits"),
    ],
)
def test_renormalize_routing(
    num_tokens,
    hidden_size,
    intermediate_size,
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    cache_permute_indices,
    routing_logits_dtype,
    zero_hidden_states,
):
    """Test Renormalize routing configurations."""
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
        zero_hidden_states=zero_hidden_states,
    )
