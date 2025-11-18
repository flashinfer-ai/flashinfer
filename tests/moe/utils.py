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
from flashinfer import GatedActType, RoutingMethodType
from flashinfer.utils import get_compute_capability


class QuantMode(IntEnum):
    """Supported quantization modes for MoE testing."""

    FP4_NVFP4_NVFP4 = 1
    FP4_MXFP4_MXFP8 = 2
    FP4_MXFP4_Bf16 = 3
    FP8_BLOCK_SCALE = 4
    FP8_PER_TENSOR = 5
    BF16 = 6


def skip_checks(
    moe_impl,
    routing_config,
    weight_processing,
    gated_act_type,
    num_tokens,
    hidden_size,
    intermediate_size,
):
    """Common skip logic for all tests."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    # Check if moe_impl is FP4Moe by class name to avoid circular imports
    is_fp4_moe = type(moe_impl).__name__ == "FP4Moe"

    # Skip incompatible combinations
    if gated_act_type == GatedActType.GeGlu and (
        not is_fp4_moe
        or moe_impl.quant_mode != QuantMode.FP4_NVFP4_NVFP4
        or routing_config["routing_method_type"] != RoutingMethodType.TopK
        or num_tokens > 128
    ):
        pytest.skip(
            f"Incompatible: {moe_impl.name} + {gated_act_type} + {routing_config['routing_method_type']} + {num_tokens}"
        )
    elif gated_act_type == GatedActType.SwiGlu and (
        hidden_size > 1024 or intermediate_size > 1024
    ):
        pytest.skip(
            f"Skip for testing speed: {gated_act_type} + {hidden_size} + {intermediate_size}"
        )

    # Skip large intermediate sizes for configurations with many experts
    if routing_config["num_experts"] >= 512 and intermediate_size > 512:
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
    if intermediate_size not in routing_config["compatible_intermediate_size"]:
        pytest.skip(
            f"Incompatible: intermediate_size={intermediate_size} with {routing_config['routing_method_type'].name} routing ({routing_config['num_experts']} experts)"
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
