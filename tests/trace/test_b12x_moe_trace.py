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


def test_b12x_wrapper_trace_keeps_constructor_quantization_config():
    from flashinfer.trace.templates.moe import (
        b12x_fused_moe_trace,
        b12x_moe_wrapper_run_trace,
    )

    assert "activation_precision" in b12x_fused_moe_trace.inputs
    assert "quant_mode" in b12x_fused_moe_trace.inputs
    assert "source_format" in b12x_fused_moe_trace.inputs
    assert "activation_precision" not in b12x_moe_wrapper_run_trace.inputs
    assert "quant_mode" in b12x_moe_wrapper_run_trace.inputs
    assert "source_format" in b12x_moe_wrapper_run_trace.inputs


def test_b12x_reference_uses_activation_precision_and_fc2_scale():
    from flashinfer.trace.templates.moe import (
        b12x_fused_moe_trace,
        b12x_moe_wrapper_run_trace,
    )

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
        quant_mode="w4a16",
    )
    wrapper_bf16 = b12x_moe_wrapper_run_trace.reference(
        **common_kwargs,
        quant_mode="w4a16",
        source_format="compressed_tensors",
    )

    with pytest.raises(ValueError, match="fc2_input_scale is required"):
        b12x_fused_moe_trace.reference(
            **common_kwargs,
            activation_precision="fp4",
        )

    with pytest.raises(ValueError, match=r"compressed_tensors.*w4a16"):
        b12x_fused_moe_trace.reference(
            **common_kwargs,
            fc2_input_scale=torch.ones((1,), dtype=torch.float32),
            quant_mode="nvfp4",
            source_format="compressed_tensors",
        )

    fp4 = b12x_fused_moe_trace.reference(
        **common_kwargs,
        fc2_input_scale=torch.ones((1,), dtype=torch.float32),
        activation_precision="fp4",
    )

    assert fp4.shape == bf16.shape
    assert torch.equal(wrapper_bf16, bf16)
    assert not torch.equal(fp4, bf16)


def test_b12x_reference_applies_expert_map():
    from flashinfer.trace.templates.moe import b12x_fused_moe_trace

    torch.manual_seed(0)
    num_experts, top_k, num_tokens = 4, 2, 6
    hidden, inter = 32, 32
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16) * 0.25
    # Small scales keep outputs near 1 so atol stays meaningful in bf16.
    w1 = torch.randint(0, 256, (num_experts, 2 * inter, hidden // 2), dtype=torch.uint8)
    w1_sf = torch.full(
        (num_experts, 2 * inter, hidden // 16), 0.0625, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    w2 = torch.randint(0, 256, (num_experts, hidden, inter // 2), dtype=torch.uint8)
    w2_sf = torch.full(
        (num_experts, hidden, inter // 16), 0.0625, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32)
    topk_weights = torch.rand((num_tokens, top_k), dtype=torch.float32)
    alpha = torch.ones((num_experts,), dtype=torch.float32)

    def ref(local_ids, expert_map):
        sel = torch.tensor(local_ids, dtype=torch.long)
        return b12x_fused_moe_trace.reference(
            x=x,
            w1_weight=w1.index_select(0, sel),
            w1_weight_sf=w1_sf.index_select(0, sel),
            w2_weight=w2.index_select(0, sel),
            w2_weight_sf=w2_sf.index_select(0, sel),
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            num_experts=num_experts,
            top_k=top_k,
            w1_alpha=alpha.index_select(0, sel),
            w2_alpha=alpha.index_select(0, sel),
            quant_mode="w4a16",
            expert_map=expert_map,
        )

    full = ref(range(num_experts), None)
    identity = ref(
        range(num_experts),
        torch.arange(num_experts, dtype=torch.int32),
    )
    assert torch.equal(identity, full)

    # Round-robin two-rank split: partials must sum to the full output.
    even = ref([0, 2], torch.tensor([0, -1, 1, -1], dtype=torch.int32))
    odd = ref([1, 3], torch.tensor([-1, 0, -1, 1], dtype=torch.int32))
    assert torch.count_nonzero(full).item() > 0
    torch.testing.assert_close(even + odd, full, rtol=0.02, atol=0.02)
