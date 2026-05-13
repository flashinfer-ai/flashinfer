"""
Copyright (c) 2026 by FlashInfer team.

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


def _packed_ones(*shape):
    return torch.full(shape, 0x22, dtype=torch.uint8)


def test_b12x_wrapper_trace_does_not_expose_constructor_precision():
    from flashinfer.trace.templates.moe import (
        b12x_fused_moe_trace,
        b12x_moe_wrapper_run_trace,
    )

    assert "activation_precision" in b12x_fused_moe_trace.inputs
    assert "activation_precision" not in b12x_moe_wrapper_run_trace.inputs


def test_b12x_reference_uses_activation_precision_and_fc2_scale():
    from flashinfer.trace.templates.moe import b12x_fused_moe_trace

    x = torch.ones((1, 2), dtype=torch.bfloat16)
    w1_weight = _packed_ones(1, 32, 1)
    w1_weight_sf = torch.ones((1, 32, 1), dtype=torch.float8_e4m3fn)
    w2_weight = _packed_ones(1, 2, 8)
    w2_weight_sf = torch.ones((1, 2, 1), dtype=torch.float8_e4m3fn)
    token_selected_experts = torch.zeros((1, 1), dtype=torch.int32)
    token_final_scales = torch.ones((1, 1), dtype=torch.float32)
    alpha = torch.ones((1,), dtype=torch.float32)

    common_kwargs = dict(
        x=x,
        w1_weight=w1_weight,
        w1_weight_sf=w1_weight_sf,
        w2_weight=w2_weight,
        w2_weight_sf=w2_weight_sf,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=1,
        top_k=1,
        w1_alpha=alpha,
        w2_alpha=alpha,
    )

    bf16 = b12x_fused_moe_trace.reference(
        **common_kwargs,
        activation_precision="bf16",
    )

    with pytest.raises(ValueError, match="fc2_input_scale is required"):
        b12x_fused_moe_trace.reference(
            **common_kwargs,
            activation_precision="fp4",
        )

    fp4 = b12x_fused_moe_trace.reference(
        **common_kwargs,
        fc2_input_scale=torch.ones((1,), dtype=torch.float32),
        activation_precision="fp4",
    )

    assert fp4.shape == bf16.shape
    assert not torch.equal(fp4, bf16)
