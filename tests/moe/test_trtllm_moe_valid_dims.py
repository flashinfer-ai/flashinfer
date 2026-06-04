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


# Numerical test for valid_intermediate_size.
#
# valid_intermediate_size pads intermediate_size: GEMM1 produces only the valid
# 2*valid_intermediate_size rows of its [gate; up] output, and GEMM2 contracts
# only over valid_intermediate_size. The dequantized float reference therefore
# uses a valid sub-problem with intermediate=valid_intermediate_size whose GEMM1
# weight rows are gate[0:valid] concatenated with up[0:valid] (NOT a flat slice
# of the [gate; up] tensor), and whose GEMM2 contraction width is valid. As with
# the valid_hidden_size test, weights must be properly mxint4-quantized+shuffled,
# and dequantize-then-select is valid because quantization is per-32-block.
#
# Hidden is left unpadded here to isolate the valid_intermediate_size path.
@pytest.mark.parametrize(
    ("hidden_size", "padded_intermediate_size", "valid_intermediate_size"),
    [
        (1024, 1024, 512),
        (1024, 768, 512),
    ],
)
def test_trtllm_mxint4_moe_valid_intermediate_size_matches_dequant_reference(
    hidden_size, padded_intermediate_size, valid_intermediate_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TRT-LLM MoE kernels.")
    if get_compute_capability(torch.device("cuda"))[0] not in [10]:
        pytest.skip("TRT-LLM MoE tests are only guaranteed on SM100/SM103 GPUs.")

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

    expert_logits = torch.randn(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states = 2 * torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    # GEMM1 output rows are laid out [gate(padded_int); up(padded_int)].
    gemm1_weights = torch.randn(
        num_experts,
        2 * padded_intermediate_size,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        num_experts,
        hidden_size,
        padded_intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    )

    sf_vec_size = 32
    gemm1_weights_int4, gemm1_scales = mxint4_quantize(gemm1_weights, sf_vec_size)
    gemm2_weights_int4, gemm2_scales = mxint4_quantize(gemm2_weights, sf_vec_size)
    quant = SimpleNamespace(
        gemm1_weights=gemm1_weights_int4,
        gemm1_scales=gemm1_scales.to(torch.bfloat16).reshape(
            num_experts, 2 * padded_intermediate_size, hidden_size // sf_vec_size
        ),
        gemm2_weights=gemm2_weights_int4,
        gemm2_scales=gemm2_scales.to(torch.bfloat16).reshape(
            num_experts, hidden_size, padded_intermediate_size // sf_vec_size
        ),
    )
    moe = MxInt4BlockScaleMoe()
    moe._cache_permute_indices = {}
    static = moe.prepare_static_weights_for_kernel(
        None,
        quant,
        None,
        None,
        hidden_size,
        padded_intermediate_size,
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
        intermediate_size=padded_intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        do_finalize=True,
        enable_pdl=device_support_pdl(device),
        valid_intermediate_size=valid_intermediate_size,
    )[0].to(torch.float)

    # Valid sub-problem: gate[0:valid] ++ up[0:valid] for GEMM1, GEMM2 K -> valid.
    pi = padded_intermediate_size
    vi = valid_intermediate_size
    ref_gemm1_weights = torch.cat(
        [quant.gemm1_weights[:, :vi, :], quant.gemm1_weights[:, pi : pi + vi, :]], dim=1
    ).contiguous()
    ref_gemm1_scales = torch.cat(
        [quant.gemm1_scales[:, :vi, :], quant.gemm1_scales[:, pi : pi + vi, :]], dim=1
    ).contiguous()

    permute_info, _ = routing_reference_renormalize(
        expert_logits, top_k, num_experts, padding
    )
    ref_args = SimpleNamespace(
        num_tokens=num_tokens,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=valid_intermediate_size,
        top_k=top_k,
        padding=padding,
        hidden_states=hidden_states,
        expert_logits=expert_logits,
        gemm1_weights=ref_gemm1_weights,
        gemm1_scales=ref_gemm1_scales,
        gemm2_weights=quant.gemm2_weights[:, :, : vi // 2].contiguous(),
        gemm2_scales=quant.gemm2_scales[:, :, : vi // sf_vec_size].contiguous(),
        permute_info=permute_info,
        use_routing_scales_on_input=False,
        activation_type=ActivationType.Swiglu,
        gemm1_bias=None,
        gemm2_bias=None,
        gemm1_lora_delta=None,
    )
    reference_output, _ = run_moe_reference_mxint4(ref_args)
    reference_output = reference_output.to(torch.float)

    assert kernel_output.shape == reference_output.shape == (num_tokens, hidden_size)
    # MxInt4 tolerances mirror MxInt4BlockScaleMoe.get_tolerances().
    check_accuracy(reference_output, kernel_output, atol=0.1, rtol=0.85, percent=0.925)


# Numerical test for the FP4 (mxfp4 x mxfp8) path with a padded intermediate_size --
# the FP4 path is the one reported in issue #2372 (trtllm_fp4_block_scale_moe output
# diverging from trtllm-gen when dimensions are padded), which is what
# valid_hidden_size / valid_intermediate_size were added to fix.
#
# Only intermediate_size is padded here (hidden is left unpadded). This keeps the
# GEMM2 output width equal to hidden_size, which the harness FP4
# prepare_static_weights_for_kernel requires (it reshapes GEMM2 by a single
# hidden_size, i.e. it assumes GEMM2-output == GEMM1-K). The combined
# valid_hidden_size + valid_intermediate_size FP4 case needs an asymmetric shuffle
# helper and is left as a follow-up; valid_hidden_size is already covered on the
# mxint4 path, which shares the same set_valid_moe_dims plumbing.
#
# mxfp4 weight quantization uses a per-tensor global scale of 1.0 (see
# calculate_fp4_global_scale_factor for use_ue8m0) and mxfp8 activation quant is
# per-32-block, so both are block-independent: the float reference is built by
# re-quantizing the sliced valid sub-problem (quantize(slice) == slice(quantize) at
# dequant precision). GEMM1 is gated, so the valid sub-problem's GEMM1 rows are
# gate[0:valid_int] concatenated with up[0:valid_int].
@pytest.mark.parametrize(
    ("hidden_size", "padded_intermediate_size", "valid_intermediate_size"),
    [
        (1024, 1024, 512),
        (1024, 768, 512),
    ],
)
def test_trtllm_fp4_mxfp4_moe_valid_intermediate_size_matches_dequant_reference(
    hidden_size, padded_intermediate_size, valid_intermediate_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TRT-LLM MoE kernels.")
    if get_compute_capability(torch.device("cuda"))[0] not in [10]:
        pytest.skip("TRT-LLM MoE tests are only guaranteed on SM100/SM103 GPUs.")

    from flashinfer import ActivationType
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe

    from .test_trtllm_gen_fused_moe import (
        FP4Moe,
        check_accuracy,
        moe_args,
        routing_reference_renormalize,
        run_moe_reference_fp4,
    )
    from .utils import QuantMode

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    num_tokens = 8
    num_experts = 128
    top_k = 8
    padding = 8
    quant_mode = QuantMode.FP4_MXFP4_MXFP8
    activation = ActivationType.Swiglu.value
    pi = padded_intermediate_size
    vi = valid_intermediate_size

    moe = FP4Moe(quant_mode=quant_mode)
    moe._cache_permute_indices = {}

    expert_logits = torch.randn(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states = 2 * torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    # GEMM1 output rows are laid out [gate(padded_int); up(padded_int)]; GEMM2 output
    # width is hidden_size (unpadded), keeping the problem square for prepare().
    gemm1_weights = torch.randn(
        num_experts, 2 * pi, hidden_size, device=device, dtype=torch.bfloat16
    )
    gemm2_weights = torch.randn(
        num_experts, hidden_size, pi, device=device, dtype=torch.bfloat16
    )

    permute_info, _ = routing_reference_renormalize(
        expert_logits, top_k, num_experts, padding
    )

    def build_args(w1f, w2f, hid, isize, swizzle_inputs):
        weights_data = moe.quantize_weights(w1f, w2f, hid)
        inputs_data = moe.quantize_inputs(
            hid, weights_data["hidden_states_scale_global"], is_swizzling=swizzle_inputs
        )
        return moe_args(
            num_tokens,
            num_experts,
            hidden_size,
            isize,
            top_k,
            padding,
            inputs_data["hidden_states"],
            inputs_data["hidden_states_scale"],
            weights_data["hidden_states_scale_global"],
            expert_logits,
            weights_data["gemm1_weights"],
            weights_data["gemm1_scales"],
            weights_data["gemm1_scales_global"],
            weights_data["gemm2_weights"],
            weights_data["gemm2_scales"],
            weights_data["gemm2_scales_global"],
            permute_info,
            False,
            activation,
        )

    # Reference: FP4 dequant MoE over the valid sub-problem (re-quantize sliced floats).
    w1_valid = torch.cat(
        [gemm1_weights[:, :vi, :], gemm1_weights[:, pi : pi + vi, :]], dim=1
    ).contiguous()
    w2_valid = gemm2_weights[:, :, :vi].contiguous()
    args_valid = build_args(w1_valid, w2_valid, hidden_states, vi, swizzle_inputs=True)
    reference_output, _ = run_moe_reference_fp4(args_valid, quant_mode)
    reference_output = reference_output.to(torch.float)

    # Kernel: properly quantized+shuffled PADDED weights, run with valid_intermediate_size.
    args_pad = build_args(
        gemm1_weights, gemm2_weights, hidden_states, pi, swizzle_inputs=True
    )
    # prepare_static_weights_for_kernel needs args_dequant.c_global_sf; obtain it from
    # the (square) padded reference run.
    _, args_dequant_pad = run_moe_reference_fp4(args_pad, quant_mode)
    static = moe.prepare_static_weights_for_kernel(
        args_dequant_pad,
        args_pad,
        gemm1_weights,
        gemm2_weights,
        hidden_size,
        pi,
        num_experts,
        None,
    )
    # Kernel-side input quant uses the non-swizzled scale layout (see CUDAGraphMoE).
    kernel_inputs = moe.quantize_inputs(
        hidden_states, args_pad.hidden_states_scale_global, is_swizzling=False
    )
    kernel_output = trtllm_fp4_block_scale_moe(
        routing_logits=expert_logits,
        routing_bias=None,
        hidden_states=kernel_inputs["hidden_states"],
        hidden_states_scale=kernel_inputs["hidden_states_scale"],
        gemm1_weights=static["gemm1_weights_fp4_shuffled"],
        gemm1_weights_scale=static["gemm1_scales_fp4_shuffled"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=static["gemm2_weights_fp4_shuffled"],
        gemm2_weights_scale=static["gemm2_scales_fp4_shuffled"],
        gemm2_bias=None,
        output1_scale_scalar=static["scale_c_fc1"],
        output1_scale_gate_scalar=static["scale_gate_fc1"],
        output2_scale_scalar=static["scale_c_fc2"],
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=pi,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        activation_type=activation,
        do_finalize=True,
        enable_pdl=device_support_pdl(device),
        valid_intermediate_size=vi,
    )[0].to(torch.float)

    assert kernel_output.shape == reference_output.shape == (num_tokens, hidden_size)
    # FP4 tolerances mirror FP4Moe.get_tolerances().
    check_accuracy(reference_output, kernel_output, atol=0.1, rtol=0.85, percent=0.92)


# Numerical test for the FP4 (mxfp4 x mxfp8) path with BOTH hidden_size and
# intermediate_size padded -- the exact scenario from issue #2372
# (trtllm_fp4_block_scale_moe output diverging from trtllm-gen when both dims are
# padded).
#
# The harness FP4 prepare_static_weights_for_kernel reshapes GEMM2 by a single
# hidden_size and so assumes GEMM2-output == GEMM1-K. With valid_hidden_size those
# differ (GEMM1 K = padded_hidden_size, GEMM2 output = hidden_size_output =
# valid_hidden_size), so we inline an asymmetric shuffle that mirrors that helper
# but reshapes GEMM1 at padded_hidden_size and GEMM2 at hidden_size_output. The
# float reference re-quantizes the sliced valid sub-problem (valid because mxfp4
# global scale is 1.0 and mxfp8 is per-32-block); GEMM1 is gated so its valid rows
# are gate[0:valid_int] concatenated with up[0:valid_int].
@pytest.mark.parametrize(
    (
        "padded_hidden_size",
        "valid_hidden_size",
        "padded_intermediate_size",
        "valid_intermediate_size",
    ),
    [
        (1024, 512, 1024, 512),
        (1024, 768, 768, 512),
    ],
)
def test_trtllm_fp4_mxfp4_moe_valid_dims_matches_dequant_reference(
    padded_hidden_size,
    valid_hidden_size,
    padded_intermediate_size,
    valid_intermediate_size,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TRT-LLM MoE kernels.")
    if get_compute_capability(torch.device("cuda"))[0] not in [10]:
        pytest.skip("TRT-LLM MoE tests are only guaranteed on SM100/SM103 GPUs.")

    from flashinfer import ActivationType
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    from .test_trtllm_gen_fused_moe import (
        FP4Moe,
        check_accuracy,
        moe_args,
        quant_fp4_batches,
        routing_reference_renormalize,
        run_moe_reference_fp4,
    )
    from .utils import QuantMode

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    num_tokens = 8
    num_experts = 128
    top_k = 8
    padding = 8
    quant_mode = QuantMode.FP4_MXFP4_MXFP8
    activation = ActivationType.Swiglu.value
    pi = padded_intermediate_size
    vi = valid_intermediate_size
    sf_vec_size = 32  # mxfp4

    moe = FP4Moe(quant_mode=quant_mode)
    moe._cache_permute_indices = {}

    expert_logits = torch.randn(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states = 2 * torch.randn(
        num_tokens, padded_hidden_size, device=device, dtype=torch.bfloat16
    )
    # GEMM1 K = padded_hidden_size; GEMM2 output = valid_hidden_size (hidden_size_output).
    gemm1_weights = torch.randn(
        num_experts,
        2 * pi,
        padded_hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    gemm2_weights = torch.randn(
        num_experts,
        valid_hidden_size,
        pi,
        device=device,
        dtype=torch.bfloat16,
    )

    permute_info, _ = routing_reference_renormalize(
        expert_logits, top_k, num_experts, padding
    )

    def build_args(w1f, w2f, hid, hsize, isize, swizzle_inputs):
        weights_data = moe.quantize_weights(w1f, w2f, hid)
        inputs_data = moe.quantize_inputs(
            hid, weights_data["hidden_states_scale_global"], is_swizzling=swizzle_inputs
        )
        return moe_args(
            num_tokens,
            num_experts,
            hsize,
            isize,
            top_k,
            padding,
            inputs_data["hidden_states"],
            inputs_data["hidden_states_scale"],
            weights_data["hidden_states_scale_global"],
            expert_logits,
            weights_data["gemm1_weights"],
            weights_data["gemm1_scales"],
            weights_data["gemm1_scales_global"],
            weights_data["gemm2_weights"],
            weights_data["gemm2_scales"],
            weights_data["gemm2_scales_global"],
            permute_info,
            False,
            activation,
        )

    # Reference: FP4 dequant MoE over the valid sub-problem (re-quantize sliced floats).
    w1_valid = torch.cat(
        [
            gemm1_weights[:, :vi, :valid_hidden_size],
            gemm1_weights[:, pi : pi + vi, :valid_hidden_size],
        ],
        dim=1,
    ).contiguous()
    w2_valid = gemm2_weights[:, :, :vi].contiguous()
    hidden_valid = hidden_states[:, :valid_hidden_size].contiguous()
    args_valid = build_args(
        w1_valid, w2_valid, hidden_valid, valid_hidden_size, vi, swizzle_inputs=True
    )
    reference_output, _ = run_moe_reference_fp4(args_valid, quant_mode)
    reference_output = reference_output.to(torch.float)

    # Kernel weights: asymmetric shuffle (GEMM1 at padded_hidden, GEMM2 at
    # hidden_size_output == valid_hidden_size). Mirrors FP4Moe.prepare_static_weights_for_kernel.
    args_pad = build_args(
        gemm1_weights, gemm2_weights, hidden_states, padded_hidden_size, pi, True
    )
    epilogue_tile_m = 128
    _, g1_sf_linear, _ = quant_fp4_batches(gemm1_weights, num_experts, True, False)
    _, g2_sf_linear, _ = quant_fp4_batches(gemm2_weights, num_experts, True, False)
    g1_fp4 = args_pad.gemm1_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * pi, padded_hidden_size // 2
    )
    g1_sf_linear = g1_sf_linear.view(torch.float8_e4m3fn).reshape(
        num_experts, 2 * pi, padded_hidden_size // sf_vec_size
    )
    g2_fp4 = args_pad.gemm2_weights.view(torch.float8_e4m3fn).reshape(
        num_experts, valid_hidden_size, pi // 2
    )
    g2_sf_linear = g2_sf_linear.view(torch.float8_e4m3fn).reshape(
        num_experts, valid_hidden_size, pi // sf_vec_size
    )

    g1_w_shuf, g1_s_shuf, g2_w_shuf, g2_s_shuf = [], [], [], []
    for i in range(num_experts):
        p = _maybe_get_cached_w3_w1_permute_indices(
            moe._cache_permute_indices,
            g1_fp4[i].view(torch.uint8),
            epilogue_tile_m,
            is_gated_act_gemm=True,
        )
        g1_w_shuf.append(g1_fp4[i].view(torch.uint8)[p.to(device)].contiguous())
        psf = _maybe_get_cached_w3_w1_permute_indices(
            moe._cache_permute_indices,
            g1_sf_linear[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        g1_s_shuf.append(
            block_scale_interleave(
                g1_sf_linear[i].view(torch.uint8)[psf.to(device)].contiguous()
            )
        )
        p = get_w2_permute_indices_with_cache(
            moe._cache_permute_indices, g2_fp4[i].view(torch.uint8), epilogue_tile_m
        )
        g2_w_shuf.append(g2_fp4[i].view(torch.uint8)[p.to(device)].contiguous())
        psf = get_w2_permute_indices_with_cache(
            moe._cache_permute_indices,
            g2_sf_linear[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        g2_s_shuf.append(
            block_scale_interleave(
                g2_sf_linear[i].view(torch.uint8)[psf.to(device)].contiguous()
            )
        )
    gemm1_weights_shuffled = torch.stack(g1_w_shuf)
    gemm1_scales_shuffled = (
        torch.stack(g1_s_shuf)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, 2 * pi, padded_hidden_size // sf_vec_size)
    )
    gemm2_weights_shuffled = torch.stack(g2_w_shuf)
    gemm2_scales_shuffled = (
        torch.stack(g2_s_shuf)
        .view(torch.float8_e4m3fn)
        .reshape(num_experts, valid_hidden_size, pi // sf_vec_size)
    )
    # mxfp4 x mxfp8: c_global_sf and all per-tensor global scales are 1.0, so the
    # epilogue scale scalars are all 1.0 (one per expert).
    scale_c_fc1 = (1.0 / args_pad.gemm1_scales_global) * (
        1.0 / args_pad.hidden_states_scale_global
    )
    scale_gate_fc1 = scale_c_fc1
    scale_c_fc2 = 1.0 / args_pad.gemm2_scales_global

    kernel_inputs = moe.quantize_inputs(
        hidden_states, args_pad.hidden_states_scale_global, is_swizzling=False
    )
    kernel_output = trtllm_fp4_block_scale_moe(
        routing_logits=expert_logits,
        routing_bias=None,
        hidden_states=kernel_inputs["hidden_states"],
        hidden_states_scale=kernel_inputs["hidden_states_scale"],
        gemm1_weights=gemm1_weights_shuffled,
        gemm1_weights_scale=gemm1_scales_shuffled,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=gemm2_weights_shuffled,
        gemm2_weights_scale=gemm2_scales_shuffled,
        gemm2_bias=None,
        output1_scale_scalar=scale_c_fc1,
        output1_scale_gate_scalar=scale_gate_fc1,
        output2_scale_scalar=scale_c_fc2,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=pi,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        activation_type=activation,
        do_finalize=True,
        enable_pdl=device_support_pdl(device),
        valid_hidden_size=valid_hidden_size,
        valid_intermediate_size=vi,
    )[0].to(torch.float)

    assert kernel_output.shape == reference_output.shape == (
        num_tokens,
        valid_hidden_size,
    )
    # FP4 tolerances mirror FP4Moe.get_tolerances().
    check_accuracy(reference_output, kernel_output, atol=0.1, rtol=0.85, percent=0.92)
