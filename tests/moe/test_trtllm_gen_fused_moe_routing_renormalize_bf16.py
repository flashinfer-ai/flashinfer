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
    BF16Moe,
    MxInt4BlockScaleMoe,
    RENORMALIZE_ACTIVATION_TYPES,
    RENORMALIZE_HIDDEN_SIZES,
    RENORMALIZE_INTERMEDIATE_SIZES,
    RENORMALIZE_NUM_TOKENS,
    RENORMALIZE_ROUTING_CONFIGS,
    RENORMALIZE_ROUTING_LOGITS_DTYPES,
    RENORMALIZE_WEIGHT_PROCESSING,
    RENORMALIZE_ZERO_HIDDEN_STATES,
    run_moe_test,
)

pytestmark = pytest.mark.long_running


@pytest.fixture(scope="module")
def cache_permute_indices():
    return {}


MOE_IMPLS = [
    pytest.param(BF16Moe(), id="BF16xBF16"),
    pytest.param(MxInt4BlockScaleMoe(), id="MxInt4xBf16"),
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
