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

# Tests for issue #2657: trtllm_fp4_block_scale_moe bf16 hidden_states support

import pytest
import torch

from flashinfer.utils import get_compute_capability
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_moe,
    WeightLayout,
    convert_to_block_layout,
)
from flashinfer import (
    ActivationType,
    fp4_quantize,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.fp4_quantization import block_scale_interleave


def skip_if_not_sm100():
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [10]:
        pytest.skip("trtllm_fp4_block_scale_moe requires SM100 (Blackwell) GPU")


def make_mxfp4_weights(num_experts, out_features, in_features, device):
    """Build minimal MxE2m1 (MXFP4) weight + scale tensors for testing."""
    w = torch.randint(0, 256, (num_experts, out_features, in_features // 2), dtype=torch.uint8, device=device)
    # MxE2m1 scale shape: [num_experts, out_features, in_features // 32]
    scale = torch.ones(num_experts, out_features, in_features // 32, dtype=torch.float8_e4m3fn, device=device)
    return w, scale


def test_bf16_hidden_states_scale_is_optional():
    """Verify hidden_states_scale has a default of None (no TypeError when omitted).

    Regression test for: TypeError: trtllm_fp4_block_scale_moe() missing 1 required
    positional argument: 'hidden_states_scale'
    """
    import inspect
    sig = inspect.signature(trtllm_fp4_block_scale_moe)
    param = sig.parameters.get("hidden_states_scale")
    assert param is not None, "hidden_states_scale parameter not found"
    assert param.default is None, (
        f"hidden_states_scale should default to None, got {param.default!r}"
    )


def test_bf16_hidden_states_wrong_weight_format_raises():
    """Calling with bf16 hidden_states + non-MxE2m1 weights must raise a clear ValueError.

    Regression test for: RuntimeError: Check failed: (mDtypeWeights == btg::Dtype::MxE2m1)
    is false: Only MxE2m1 weights are supported by block scale MoE with Bfloat16, E4m3 or
    MxE4m3 activation
    """
    skip_if_not_sm100()

    device = torch.device("cuda")
    num_tokens, hidden_size, intermediate_size = 8, 128, 128
    num_experts, top_k = 2, 1

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    # E2m1 (NvFP4) weights: scale shape [E, out, in//16] — NOT MxE2m1
    gemm1_w = torch.randint(0, 256, (num_experts, 2 * intermediate_size, hidden_size // 2), dtype=torch.uint8, device=device)
    gemm1_ws = torch.ones(num_experts, 2 * intermediate_size, hidden_size // 16, dtype=torch.float8_e4m3fn, device=device)
    gemm2_w = torch.randint(0, 256, (num_experts, hidden_size, intermediate_size // 2), dtype=torch.uint8, device=device)
    gemm2_ws = torch.ones(num_experts, hidden_size, intermediate_size // 16, dtype=torch.float8_e4m3fn, device=device)

    routing_logits = torch.zeros(num_tokens, num_experts, dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="MxE2m1"):
        trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=hidden_states,
            gemm1_weights=gemm1_w,
            gemm1_weights_scale=gemm1_ws,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_w,
            gemm2_weights_scale=gemm2_ws,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            # hidden_states_scale intentionally omitted (defaults to None)
        )
