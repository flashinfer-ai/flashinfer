# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""Which activation and finalize kernels the trtllm-gen fused MoE launches at a given token count.

The output tests cannot tell the glue kernels apart: for finite inputs ``activationDeepSeekKernelV2``
writes the same bytes as ``activationDeepSeekKernel`` in every row the old kernel writes, and
``finalizeKernelV2`` the same values as ``finalizeKernelVecLoad``, the kernel it replaces (the
small-batch finalize sums the top-k in a different fp32 order, below any output tolerance). This test
reads the launched kernel names from the profiler instead and checks the selection
rule of ``activation::run()`` and ``finalize::run()`` in ``trtllm_fused_moe_dev_kernel.cu`` on both
sides of each threshold: the activation switches at ``DEEP_SEEK_ACTIVATION_V2_MIN_TOKENS`` (64 tokens);
the finalize switches when the per-token grid reaches ``FINALIZE_SMALL_GRID_CTA_BOUND`` (1184 CTAs,
i.e. ``ceil(hidden / 256) * tokens``) while ``tokens * slices`` stays within 32 CTAs per SM. With a
4096-wide hidden size that is 16 CTAs per token, so 64 tokens flip the activation only and 128 tokens
flip both; 8 tokens keep the small-batch kernels of both.
"""

import pytest
import torch

from flashinfer.utils import get_compute_capability

from tests.moe.trtllm_gen_fused_moe_utils import (
    FP8BlockScaleMoe,
    Fp8QuantizationType,
    QuantMode,
    RoutingMethodType,
    WeightLayout,
    trtllm_fp8_block_scale_moe,
)
from tests.cake.test_helpers.cake_profiler import cuda_kernel_names

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1024
NUM_EXPERTS = 32
TOP_K = 8

ACTIVATION_KERNELS = ("activationDeepSeekKernelV2", "activationDeepSeekKernel")
FINALIZE_KERNELS = (
    "finalizeKernelV2",
    "finalizeKernelVecLoad",
    "finalizeKernelSmallBatch",
    "finalizeKernel",
)


def _launched(names, kernels):
    """The kernels of ``kernels`` among the profiler names (a template name is ``<kernel><...>``)."""
    found = []
    for name in names:
        for kernel in kernels:
            if kernel + "<" in name or kernel + "(" in name:
                found.append(kernel)
                break
    return found


def _skip_unless_sm100_class():
    major, _ = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("the trtllm-gen fused MoE runs on SM100-class GPUs only")


def _make_call(num_tokens, enable_pdl):
    torch.manual_seed(1234 + num_tokens)
    device = torch.device("cuda:0")
    routing_logits = torch.randn(
        (num_tokens, NUM_EXPERTS), device=device, dtype=torch.bfloat16
    )
    hidden_states = torch.randn(
        (num_tokens, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    gemm1_weights = torch.randn(
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )
    quant_impl = FP8BlockScaleMoe(
        fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
    )
    quant_weights = quant_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )
    quant_inputs = quant_impl.quantize_inputs(hidden_states)

    def call():
        return trtllm_fp8_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=quant_inputs["hidden_states"],
            hidden_states_scale=quant_inputs["hidden_states_scale"],
            gemm1_weights=quant_weights["gemm1_weights"],
            gemm1_weights_scale=quant_weights["gemm1_scales"],
            gemm2_weights=quant_weights["gemm2_weights"],
            gemm2_weights_scale=quant_weights["gemm2_scales"],
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            n_group=None,
            topk_group=None,
            intermediate_size=INTERMEDIATE_SIZE,
            local_expert_offset=0,
            local_num_experts=NUM_EXPERTS,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.Renormalize.value,
            use_shuffled_weight=False,
            weight_layout=WeightLayout.MajorK.value,
            enable_pdl=enable_pdl,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
        )

    return call


@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize(
    "num_tokens, activation_kernel, finalize_kernel",
    [
        (8, "activationDeepSeekKernel", "finalizeKernelSmallBatch"),
        (64, "activationDeepSeekKernelV2", "finalizeKernelSmallBatch"),
        (128, "activationDeepSeekKernelV2", "finalizeKernelV2"),
    ],
)
def test_glue_kernel_selection(
    num_tokens, activation_kernel, finalize_kernel, enable_pdl
):
    _skip_unless_sm100_class()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    # finalizeKernelV2 also needs tokens * slices <= 32 * SMs; 128 tokens x 4 slices here.
    assert 32 * sms >= 128 * (HIDDEN_SIZE // 8 // 128), sms
    call = _make_call(num_tokens, enable_pdl)
    call()  # JIT build and warm-up outside the profiled window
    torch.cuda.synchronize()
    names = cuda_kernel_names(call)
    assert _launched(names, ACTIVATION_KERNELS) == [activation_kernel], names
    assert _launched(names, FINALIZE_KERNELS) == [finalize_kernel], names
