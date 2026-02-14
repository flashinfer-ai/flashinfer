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
from enum import IntEnum
from flashinfer import ActivationType, RoutingMethodType
from flashinfer.utils import get_compute_capability
from flashinfer.fused_moe import WeightLayout


class QuantMode(IntEnum):
    """Supported quantization modes for MoE testing."""

    FP4_NVFP4_NVFP4 = 1
    FP4_MXFP4_MXFP8 = 2
    FP4_MXFP4_Bf16 = 3
    FP8_BLOCK_SCALE_DEEPSEEK = 4
    FP8_BLOCK_SCALE_MXFP8 = 5
    FP8_PER_TENSOR = 6
    BF16 = 7
    MXINT4_BF16_BF16 = 8


NON_GATED_ACTIVATION_SUPPORTED_QUANT_MODES = [
    QuantMode.FP4_NVFP4_NVFP4,
    QuantMode.FP8_PER_TENSOR,
]


def is_gated_activation(activation_type: ActivationType) -> bool:
    return activation_type in [
        ActivationType.Swiglu,
        ActivationType.Geglu,
        ActivationType.SwigluBias,
    ]


def skip_checks(
    moe_impl,
    routing_config,
    weight_processing,
    activation_type,
    num_tokens,
    hidden_size,
    intermediate_size,
    zero_hidden_states=False,
):
    """Common skip logic for all tests."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    # Check moe_impl class by name to avoid circular imports
    is_fp4_moe = type(moe_impl).__name__ == "FP4Moe"
    is_fp8_block_scale_moe = type(moe_impl).__name__ == "FP8BlockScaleMoe"

    # Skip zero hidden states tests for non-FP8 Block Scale MoE implementations
    if zero_hidden_states and not is_fp8_block_scale_moe:
        pytest.skip("Skipping zero hidden states tests for non-FP8 Block Scale MoE.")

    # Skip incompatible combinations
    if activation_type == ActivationType.Geglu and (
        not is_fp4_moe
        or moe_impl.quant_mode != QuantMode.FP4_NVFP4_NVFP4
        or routing_config["routing_method_type"] != RoutingMethodType.TopK
        or num_tokens > 128
    ):
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {activation_type} + {routing_config['routing_method_type']} + {num_tokens}"
        )
    elif activation_type == ActivationType.Swiglu and (
        hidden_size > 1024 or intermediate_size > 1024
    ):
        pytest.skip(
            f"Skip for testing speed: {activation_type} + {hidden_size} + {intermediate_size}"
        )

    compatible_activation_types = routing_config.get(
        "compatible_activation_types", None
    )
    if (
        compatible_activation_types is not None
        and activation_type not in compatible_activation_types
    ):
        pytest.skip(
            f"Incompatible: activation_type={activation_type} not in compatible_activation_types ({compatible_activation_types})"
        )

    if (
        not is_gated_activation(activation_type)
        and moe_impl.quant_mode not in NON_GATED_ACTIVATION_SUPPORTED_QUANT_MODES
    ):
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {activation_type=} + quant_mode={moe_impl.quant_mode}: non-gated activations only supported with these quant modes: {NON_GATED_ACTIVATION_SUPPORTED_QUANT_MODES}"
        )

    # Skip large intermediate sizes for configurations with many experts
    if routing_config["num_experts"] > 512 and intermediate_size > 512:
        pytest.skip(
            f"Skipping for testing speed: intermediate_size={intermediate_size} with {routing_config['num_experts']} experts"
        )

    if type(moe_impl) not in routing_config["compatible_moe_impls"]:
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {routing_config['routing_method_type'].name}"
        )
    if type(moe_impl) not in weight_processing["compatible_moe_impls"]:
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {weight_processing['use_shuffled_weight']} + {weight_processing['layout']}"
        )
    if (
        is_fp8_block_scale_moe
        and moe_impl.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8
        and not weight_processing["use_shuffled_weight"]
    ):
        pytest.skip("use_shuffled_weight must be true for MxFp8.")
    if (
        is_fp8_block_scale_moe
        and moe_impl.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8
        and weight_processing["layout"] != WeightLayout.MajorK
    ):
        pytest.skip("weight_layout must be MajorK for MxFp8.")

    if intermediate_size not in routing_config["compatible_intermediate_size"]:
        pytest.skip(
            f"Incompatible: intermediate_size={intermediate_size} with {routing_config['routing_method_type'].name} routing ({routing_config['num_experts']} experts)"
        )

    if moe_impl.quant_mode == QuantMode.MXINT4_BF16_BF16 and (
        intermediate_size % 256 != 0 or hidden_size % 256 != 0
    ):
        pytest.skip(
            f"Incompatible: intermediate_size={intermediate_size} or hidden_size={hidden_size} with MXINT4_BF16_BF16 quantization"
        )

    # TODO(jimmzhou): enable MxFP4xBf16 on SM103
    if (
        is_fp4_moe
        and moe_impl.quant_mode == QuantMode.FP4_MXFP4_Bf16
        and compute_capability[0] == 10
        and compute_capability[1] == 3
    ):
        pytest.xfail(
            "Note(jimmzhou): Make MxFP4xBf16 nonfunctional on SM103 to avoid B200 regression"
        )
