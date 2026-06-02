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


# Numerical test for valid_hidden_size.
#
# IMPORTANT: trtllm_mxint4_block_scale_moe consumes mxint4-quantized AND
# *shuffled* weights (see prepare_static_weights_for_kernel in
# test_trtllm_gen_fused_moe.py). The kernel's internal int4 weight + block-scale
# layout depends on the full (padded) hidden_size K, so a naive
# "slice the raw weights and compare to a second kernel call" self-consistency
# check is NOT a valid reference -- the sliced layout no longer matches what the
# kernel expects. Instead we build properly quantized+shuffled padded weights,
# run the kernel with valid_hidden_size, and compare against a dequantized float
# reference computed over only the valid hidden region. Dequantize-then-slice is
# valid because mxint4 quantization is independent per 32-element block.
#
# Shapes are selective and modeled on test_trtllm_gen_fused_moe.py's Renormalize
# config (num_experts=128, top_k=8). MxInt4 requires hidden_size % 256 == 0 and
# intermediate_size % 256 == 0; valid_hidden_size is kept a multiple of 256 so it
# also satisfies roundUp(valid_hidden_size, 128) == valid_hidden_size.
@pytest.mark.parametrize(
    ("padded_hidden_size", "valid_hidden_size", "intermediate_size"),
    [
        (1024, 512, 768),
        (1024, 768, 1024),
    ],
)
def test_trtllm_mxint4_moe_valid_hidden_size_matches_dequant_reference(
    padded_hidden_size, valid_hidden_size, intermediate_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TRT-LLM MoE kernels.")
    if get_compute_capability(torch.device("cuda"))[0] not in [10]:
        pytest.skip("TRT-LLM MoE tests are only guaranteed on SM100/SM103 GPUs.")

    # Imported here (not at module scope) so collection does not require the heavy
    # harness / CUDA-only deps on machines that only run the unit-level checks above.
    from types import SimpleNamespace

    from flashinfer import ActivationType

    from .test_trtllm_gen_fused_moe import (
        MxInt4BlockScaleMoe,
        check_accuracy,
        mxint4_quantize,
        routing_reference_renormalize,
        run_moe_reference_mxint4,
    )

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    num_tokens = 8
    num_experts = 128
    top_k = 8
    padding = 8

    # Float source tensors at the padded hidden size. GEMM2 output width equals the
    # valid hidden size (== hidden_size_output, since valid_hidden_size % 128 == 0).
    expert_logits = torch.randn(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states = 2 * torch.randn(
        num_tokens, padded_hidden_size, device=device, dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        num_experts,
        2 * intermediate_size,
        padded_hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        num_experts,
        valid_hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    )

    # Quantize (mxint4) + shuffle into the layout the kernel expects for the padded K.
    # Quantize per-tensor (not via MxInt4BlockScaleMoe.quantize_weights) because the
    # GEMM2 output width (valid_hidden_size) differs from the GEMM1 contraction width
    # (padded_hidden_size); that helper assumes they are equal.
    sf_vec_size = 32
    gemm1_weights_int4, gemm1_scales = mxint4_quantize(gemm1_weights, sf_vec_size)
    gemm2_weights_int4, gemm2_scales = mxint4_quantize(gemm2_weights, sf_vec_size)
    quant = SimpleNamespace(
        gemm1_weights=gemm1_weights_int4,
        gemm1_scales=gemm1_scales.to(torch.bfloat16).reshape(
            num_experts, 2 * intermediate_size, padded_hidden_size // sf_vec_size
        ),
        gemm2_weights=gemm2_weights_int4,
        gemm2_scales=gemm2_scales.to(torch.bfloat16).reshape(
            num_experts, valid_hidden_size, intermediate_size // sf_vec_size
        ),
    )
    moe = MxInt4BlockScaleMoe()
    moe._cache_permute_indices = {}
    static = moe.prepare_static_weights_for_kernel(
        None,
        quant,
        None,
        None,
        padded_hidden_size,
        intermediate_size,
        num_experts,
        None,
    )

    kernel_output = trtllm_mxint4_block_scale_moe(
        routing_logits=expert_logits,
        routing_bias=None,
        hidden_states=hidden_states,
        gemm1_weights=static["gemm1_weights"],
        gemm1_weights_scale=static["gemm1_scales"],
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=static["gemm2_weights"],
        gemm2_weights_scale=static["gemm2_scales"],
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
        valid_hidden_size=valid_hidden_size,
    )[0].to(torch.float)

    # Dequantized float reference over only the valid hidden region. Slicing the
    # quantized weights/scales here is valid because each 32-element block is
    # quantized independently, so dequantize(slice) == slice(dequantize).
    permute_info, _ = routing_reference_renormalize(
        expert_logits, top_k, num_experts, padding
    )
    ref_args = SimpleNamespace(
        num_tokens=num_tokens,
        num_experts=num_experts,
        hidden_size=valid_hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        padding=padding,
        hidden_states=hidden_states[:, :valid_hidden_size].contiguous(),
        expert_logits=expert_logits,
        gemm1_weights=quant.gemm1_weights[:, :, : valid_hidden_size // 2].contiguous(),
        gemm1_scales=quant.gemm1_scales[:, :, : valid_hidden_size // 32].contiguous(),
        gemm2_weights=quant.gemm2_weights,
        gemm2_scales=quant.gemm2_scales,
        permute_info=permute_info,
        use_routing_scales_on_input=False,
        activation_type=ActivationType.Swiglu,
        gemm1_bias=None,
        gemm2_bias=None,
        gemm1_lora_delta=None,
    )
    reference_output, _ = run_moe_reference_mxint4(ref_args)
    reference_output = reference_output.to(torch.float)

    assert (
        kernel_output.shape == reference_output.shape == (num_tokens, valid_hidden_size)
    )
    # MxInt4 tolerances mirror MxInt4BlockScaleMoe.get_tolerances().
    check_accuracy(reference_output, kernel_output, atol=0.1, rtol=0.85, percent=0.925)
