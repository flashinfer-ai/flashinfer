# NOTE for future contributors (incl. AI agents): keep this file lean. Randomized
# breadth (shapes, token counts) belongs in tests/moe/test_unified_moe_fuzz.py --
# extend its axes/adapters. This file exists for the quant x routing x layout
# kernel-selection matrix and for paths the fuzzer cannot express; add cases only
# as deliberate regression anchors.

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

from tests.moe.trtllm_gen_fused_moe_utils import (
    ActivationType,
    FP8BlockScaleMoe,
    FP8PerChannelMoe,
    FP8PerTensorMoe,
    QuantMode,
    RENORMALIZE_ACTIVATION_TYPES,
    RENORMALIZE_HIDDEN_SIZES,
    RENORMALIZE_INTERMEDIATE_SIZES,
    RENORMALIZE_NUM_TOKENS,
    RENORMALIZE_ROUTING_CONFIGS,
    RENORMALIZE_ROUTING_LOGITS_DTYPES,
    RENORMALIZE_WEIGHT_PROCESSING,
    RENORMALIZE_ZERO_HIDDEN_STATES,
    RoutingMethodType,
    WeightLayout,
    run_moe_test,
)

pytestmark = pytest.mark.long_running


@pytest.fixture(scope="module")
def cache_permute_indices():
    return {}


MOE_IMPLS = [
    pytest.param(
        FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
        id="FP8_Block_DeepSeek",
    ),
    pytest.param(
        FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8),
        id="FP8_Block_MxFp8",
    ),
    pytest.param(FP8PerTensorMoe(), id="FP8_Tensor"),
]


@pytest.mark.parametrize("zero_hidden_states", RENORMALIZE_ZERO_HIDDEN_STATES)
@pytest.mark.parametrize("num_tokens", RENORMALIZE_NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", RENORMALIZE_HIDDEN_SIZES)
@pytest.mark.parametrize("intermediate_size", RENORMALIZE_INTERMEDIATE_SIZES)
@pytest.mark.parametrize("moe_impl", MOE_IMPLS)
@pytest.mark.parametrize("routing_config", RENORMALIZE_ROUTING_CONFIGS)
@pytest.mark.parametrize("weight_processing", RENORMALIZE_WEIGHT_PROCESSING)
@pytest.mark.parametrize("activation_type", RENORMALIZE_ACTIVATION_TYPES)
@pytest.mark.parametrize("routing_logits_dtype", RENORMALIZE_ROUTING_LOGITS_DTYPES)
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


@pytest.mark.parametrize("num_tokens", [8, 768, 3072])
@pytest.mark.parametrize("hidden_size", [1024])
@pytest.mark.parametrize("intermediate_size", [384, 512, 768, 1024])
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
                "compatible_moe_impls": [FP8PerChannelMoe],
                "compatible_intermediate_size": [384, 512, 768, 1024],
                "enable_autotune": True,
            },
            id="Renorm_128e",
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
def test_fp8_per_channel_renormalize(
    num_tokens,
    hidden_size,
    intermediate_size,
    routing_config,
    activation_type,
    cache_permute_indices,
):
    run_moe_test(
        num_tokens,
        hidden_size,
        intermediate_size,
        FP8PerChannelMoe(),
        routing_config,
        {
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP8PerChannelMoe],
        },
        activation_type,
        cache_permute_indices,
    )
