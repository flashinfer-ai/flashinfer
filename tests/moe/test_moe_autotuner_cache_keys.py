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

from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.core import Fp8QuantizationType, MoETuningSetup

HIDDEN_SIZE = 256
NUM_EXPERTS = 8
TOP_K = 2
SCALE_DIM = HIDDEN_SIZE // 128


def _assert_cache_key_match(config, inputs):
    """Core assertion: at least one tuning profile must produce a cache key
    that matches the inference cache key for the given inputs."""
    tuner = AutoTuner()
    inference_shapes = tuple(tuner._get_input_sizes(inputs))
    inference_key = AutoTuner._find_nearest_profile(inference_shapes, config)

    profiles = tuner._generate_optimization_profiles(config, inputs)
    tuning_keys = {
        AutoTuner._find_nearest_profile(p.get_opt_shapes(), config) for p in profiles
    }
    assert inference_key in tuning_keys, (
        f"No tuning profile matches the inference cache key.\n"
        f"  inference key: {inference_key}\n"
        f"  tuning keys:   {tuning_keys}"
    )


@pytest.mark.parametrize("num_tokens", [512, 4096, 8192])
def test_deepseek_fp8_routing_from_logits(num_tokens):
    config = MoETuningSetup.select_fp8_tuning_config(
        has_routing_logits=True,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
    )
    output = torch.empty(num_tokens, HIDDEN_SIZE)
    inputs = MoETuningSetup.build_fp8_moe_inputs(
        routing_logits=torch.empty(num_tokens, NUM_EXPERTS),
        hidden_states=torch.empty(num_tokens, HIDDEN_SIZE),
        hidden_states_scale=torch.empty(SCALE_DIM, num_tokens),
        output=output,
    )
    _assert_cache_key_match(config, inputs)


@pytest.mark.parametrize("num_tokens", [512, 4096, 8192])
def test_mxfp8_routing_from_logits(num_tokens):
    config = MoETuningSetup.select_fp8_tuning_config(
        has_routing_logits=True,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
    )
    output = torch.empty(num_tokens, HIDDEN_SIZE)
    inputs = MoETuningSetup.build_fp8_moe_inputs(
        routing_logits=torch.empty(num_tokens, NUM_EXPERTS),
        hidden_states=torch.empty(num_tokens, HIDDEN_SIZE),
        hidden_states_scale=torch.empty(num_tokens, SCALE_DIM),
        output=output,
    )
    _assert_cache_key_match(config, inputs)


@pytest.mark.parametrize("num_tokens", [512, 4096, 8192])
def test_deepseek_fp8_precomputed_routing(num_tokens):
    config = MoETuningSetup.select_fp8_tuning_config(
        has_routing_logits=False,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
    )
    output = torch.empty(num_tokens, HIDDEN_SIZE)
    inputs = MoETuningSetup.build_fp8_moe_inputs(
        routing_logits=None,
        hidden_states=torch.empty(num_tokens, HIDDEN_SIZE),
        hidden_states_scale=torch.empty(SCALE_DIM, num_tokens),
        output=output,
        topk_ids=torch.empty(num_tokens, TOP_K, dtype=torch.int32),
        expert_weights=torch.empty(num_tokens, TOP_K),
    )
    _assert_cache_key_match(config, inputs)


@pytest.mark.parametrize("num_tokens", [512, 4096, 8192])
def test_no_scale_config(num_tokens):
    config = MoETuningSetup.tuning_config_no_hidden_states_scales
    inputs = [
        torch.empty(num_tokens, HIDDEN_SIZE),
        torch.empty(num_tokens, NUM_EXPERTS),
        torch.empty(num_tokens, TOP_K, dtype=torch.int32),
        torch.empty(num_tokens, TOP_K),
        torch.empty(num_tokens, HIDDEN_SIZE),
    ]
    _assert_cache_key_match(config, inputs)


@pytest.mark.parametrize("max_tokens", [4096, 16384])
def test_max_tokens_respected(max_tokens):
    """Tokens at max_tokens must still hit the cache after refine."""
    MoETuningSetup.refine_tuning_config(max_tokens)
    config = MoETuningSetup.select_fp8_tuning_config(
        has_routing_logits=True,
        fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
    )
    output = torch.empty(max_tokens, HIDDEN_SIZE)
    inputs = MoETuningSetup.build_fp8_moe_inputs(
        routing_logits=torch.empty(max_tokens, NUM_EXPERTS),
        hidden_states=torch.empty(max_tokens, HIDDEN_SIZE),
        hidden_states_scale=torch.empty(SCALE_DIM, max_tokens),
        output=output,
    )
    _assert_cache_key_match(config, inputs)


def test_select_config_deepseek_with_routing():
    config = MoETuningSetup.select_fp8_tuning_config(
        True, Fp8QuantizationType.DeepSeekFp8
    )
    assert config is MoETuningSetup.tuning_config_routing_from_logits_deepseek_fp8


def test_select_config_deepseek_precomputed():
    config = MoETuningSetup.select_fp8_tuning_config(
        False, Fp8QuantizationType.DeepSeekFp8
    )
    assert config is MoETuningSetup.tuning_config_precomputed_routing_deepseek_fp8


def test_select_config_mxfp8_with_routing():
    config = MoETuningSetup.select_fp8_tuning_config(True, Fp8QuantizationType.MxFp8)
    assert config is MoETuningSetup.tuning_config_routing_from_logits


def test_select_config_mxfp8_precomputed():
    config = MoETuningSetup.select_fp8_tuning_config(False, Fp8QuantizationType.MxFp8)
    assert config is MoETuningSetup.tuning_config_precomputed_routing
