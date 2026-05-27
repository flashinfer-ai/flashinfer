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

from flashinfer import RoutingMethodType
from flashinfer.fused_moe import trtllm_mxint4_block_scale_moe
from flashinfer.fused_moe.core import _infer_trtllm_moe_output_hidden_size
from flashinfer.utils import device_support_pdl, get_compute_capability


@pytest.mark.parametrize(
    ("hidden_size", "valid_hidden_size", "expected"),
    [
        (3072, None, 3072),
        (3072, 2880, 2944),
        (512, 64, 128),
    ],
)
def test_infer_trtllm_moe_output_hidden_size(hidden_size, valid_hidden_size, expected):
    assert (
        _infer_trtllm_moe_output_hidden_size(hidden_size, valid_hidden_size) == expected
    )


@pytest.mark.parametrize("valid_hidden_size", [0, -1, 129])
def test_infer_trtllm_moe_output_hidden_size_rejects_invalid(
    valid_hidden_size,
):
    with pytest.raises(ValueError):
        _infer_trtllm_moe_output_hidden_size(128, valid_hidden_size)


def test_trtllm_mxint4_moe_valid_hidden_size_matches_unpadded_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TRT-LLM MoE kernels.")
    if get_compute_capability(torch.device("cuda"))[0] not in [10]:
        pytest.skip("TRT-LLM MoE tests are only guaranteed on SM100/SM103 GPUs.")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    num_tokens = 8
    num_experts = 16
    top_k = 2
    padded_hidden_size = 512
    valid_hidden_size = 256
    intermediate_size = 512

    routing_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )
    hidden_states = torch.randn(
        num_tokens, padded_hidden_size, device=device, dtype=torch.bfloat16
    )

    gemm1_weights = torch.randint(
        0,
        256,
        (num_experts, 2 * intermediate_size, padded_hidden_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    gemm1_weights_scale = torch.randn(
        num_experts,
        2 * intermediate_size,
        padded_hidden_size // 32,
        dtype=torch.bfloat16,
        device=device,
    )
    gemm2_weights = torch.randint(
        0,
        256,
        (num_experts, valid_hidden_size, intermediate_size // 2),
        dtype=torch.uint8,
        device=device,
    )
    gemm2_weights_scale = torch.randn(
        num_experts,
        valid_hidden_size,
        intermediate_size // 32,
        dtype=torch.bfloat16,
        device=device,
    )

    common_kwargs = dict(
        routing_logits=routing_logits,
        routing_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        do_finalize=True,
        enable_pdl=device_support_pdl(device),
    )

    valid_output = trtllm_mxint4_block_scale_moe(
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        valid_hidden_size=valid_hidden_size,
        **common_kwargs,
    )[0].to(torch.float)

    reference_output = trtllm_mxint4_block_scale_moe(
        hidden_states=hidden_states[:, :valid_hidden_size].contiguous(),
        gemm1_weights=gemm1_weights[:, :, : valid_hidden_size // 2].contiguous(),
        gemm1_weights_scale=gemm1_weights_scale[
            :, :, : valid_hidden_size // 32
        ].contiguous(),
        **common_kwargs,
    )[0].to(torch.float)

    assert (
        valid_output.shape == reference_output.shape == (num_tokens, valid_hidden_size)
    )
    mask = torch.isclose(valid_output, reference_output, rtol=1e-2, atol=1e-2)
    mismatch_pct = (~mask).float().mean().item() * 100
    assert mismatch_pct < 10, f"Mismatch percentage is {mismatch_pct:.2f}%"
