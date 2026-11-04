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

import math

import pytest
import torch

from flashinfer import (
    ActivationType,
    RoutingMethodType,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.fp4_quantization import block_scale_interleave
from flashinfer.fused_moe import (
    pad_trtllm_moe_weights,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
)
from flashinfer.fused_moe.core import (
    Fp8QuantizationType,
    _maybe_get_cached_w3_w1_permute_indices,
    _round_up,
    get_w2_permute_indices_with_cache,
)
from flashinfer.utils import device_support_pdl, get_compute_capability

from .trtllm_gen_fused_moe_utils import (
    FP4Moe,
    FP8BlockScaleMoe,
    check_accuracy,
    moe_args,
    quant_fp4_batches,
    routing_reference_renormalize,
)
from .utils import QuantMode, is_gated_activation

# Nemotron Nano expert geometry (TP1 and TP2) and the #3206 TP4 shard.
_SHAPES = [(2688, 1856), (2688, 928), (1024, 176)]
_CONTAM = 42.0


def _skip_unless_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for TRT-LLM MoE kernels.")
    if get_compute_capability(torch.device("cuda"))[0] not in [10]:
        pytest.skip("TRT-LLM MoE tests are only guaranteed on SM100/SM103 GPUs.")


@pytest.mark.parametrize("hidden_size,intermediate_size", _SHAPES)
@pytest.mark.parametrize("elems_per_byte,sf_vec_size", [(1, 32), (2, 16)])
@pytest.mark.parametrize("is_gated_act", [True, False])
def test_pad_trtllm_moe_weights_round_trip(
    hidden_size, intermediate_size, elems_per_byte, sf_vec_size, is_gated_act
):
    if intermediate_size % sf_vec_size:
        pytest.skip("intermediate_size must be a multiple of the scale block")
    torch.manual_seed(0)
    num_experts = 2
    factor = 2 if is_gated_act else 1
    rows1 = factor * intermediate_size

    def rand_bytes(*shape):
        return torch.randint(1, 255, shape, dtype=torch.uint8)

    w1 = rand_bytes(num_experts, rows1, hidden_size // elems_per_byte)
    s1 = rand_bytes(num_experts, rows1, hidden_size // sf_vec_size)
    w2 = rand_bytes(num_experts, hidden_size, intermediate_size // elems_per_byte)
    s2 = rand_bytes(num_experts, hidden_size, intermediate_size // sf_vec_size)

    padded = pad_trtllm_moe_weights(
        w1.view(torch.float8_e4m3fn),
        s1,
        w2.view(torch.float8_e4m3fn),
        s2,
        hidden_size,
        intermediate_size,
        is_gated_act=is_gated_act,
    )
    pi, ph = padded.intermediate_size, padded.hidden_size
    gemm2_rows = _round_up(hidden_size, 128)
    assert pi % 128 == 0 and pi >= intermediate_size
    assert ph % 512 == 0 and ph >= hidden_size
    assert padded.valid_hidden_size == hidden_size
    assert padded.valid_intermediate_size == intermediate_size
    assert padded.gemm1_weights.dtype == torch.float8_e4m3fn
    assert padded.gemm1_weights.shape == (
        num_experts,
        factor * pi,
        ph // elems_per_byte,
    )
    assert padded.gemm1_weights_scale.shape == (
        num_experts,
        factor * pi,
        ph // sf_vec_size,
    )
    assert padded.gemm2_weights.shape == (num_experts, gemm2_rows, pi // elems_per_byte)
    assert padded.gemm2_weights_scale.shape == (
        num_experts,
        gemm2_rows,
        pi // sf_vec_size,
    )

    neutral = 127 if sf_vec_size == 32 else 0x38
    for src, dst, fill, cols in (
        (w1, padded.gemm1_weights.view(torch.uint8), 0, hidden_size // elems_per_byte),
        (s1, padded.gemm1_weights_scale, neutral, hidden_size // sf_vec_size),
    ):
        mask = torch.zeros_like(dst, dtype=torch.bool)
        for h in range(factor):
            dst_half = dst[:, h * pi : (h + 1) * pi]
            src_half = src[:, h * intermediate_size : (h + 1) * intermediate_size]
            torch.testing.assert_close(dst_half[:, :intermediate_size, :cols], src_half)
            mask[:, h * pi : h * pi + intermediate_size, :cols] = True
        assert (dst[~mask] == fill).all()
    for src, dst, fill, cols in (
        (
            w2,
            padded.gemm2_weights.view(torch.uint8),
            0,
            intermediate_size // elems_per_byte,
        ),
        (s2, padded.gemm2_weights_scale, neutral, intermediate_size // sf_vec_size),
    ):
        torch.testing.assert_close(dst[:, :hidden_size, :cols], src)
        mask = torch.zeros_like(dst, dtype=torch.bool)
        mask[:, :hidden_size, :cols] = True
        assert (dst[~mask] == fill).all()


def test_pad_trtllm_moe_weights_rejects_mismatched_shapes():
    w1 = torch.zeros(2, 2 * 64, 128, dtype=torch.float8_e4m3fn)
    s1 = torch.zeros(2, 2 * 64, 128 // 32, dtype=torch.uint8)
    w2 = torch.zeros(2, 128, 64, dtype=torch.float8_e4m3fn)
    s2 = torch.zeros(2, 128, 64 // 32, dtype=torch.uint8)
    pad_trtllm_moe_weights(w1, s1, w2, s2, 128, 64)
    with pytest.raises(ValueError):
        pad_trtllm_moe_weights(w1, s1, w2, s2, 128, 64, is_gated_act=False)
    with pytest.raises(ValueError):
        pad_trtllm_moe_weights(w1, s1, w2[:, :64], s2, 128, 64)
    with pytest.raises(ValueError):
        pad_trtllm_moe_weights(w1, s1, w2, s2, 128, 64, hidden_alignment=256 + 64)


def _run_padded_moe(
    quant_mode,
    activation_type,
    valid_hidden_size,
    valid_intermediate_size,
    hidden_alignment=512,
    num_tokens=64,
    num_experts=8,
    top_k=2,
):
    """Run the padded kernel at ``(valid_hidden_size, valid_intermediate_size)``.

    Returns ``(kernel_output, reference_output, tolerances)``; the reference is the
    dequant MoE over the unpadded problem.
    """
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    padding = 8
    vh, vi = valid_hidden_size, valid_intermediate_size
    gated = is_gated_activation(activation_type)
    factor = 2 if gated else 1
    activation = activation_type.value

    if quant_mode == QuantMode.FP8_BLOCK_SCALE_MXFP8:
        moe = FP8BlockScaleMoe(fp8_quantization_type=quant_mode)
    else:
        moe = FP4Moe(quant_mode=quant_mode)
    moe._cache_permute_indices = {}

    expert_logits = torch.randn(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states = 2 * torch.randn(num_tokens, vh, device=device, dtype=torch.bfloat16)
    gemm1_weights = torch.randn(
        num_experts, factor * vi, vh, device=device, dtype=torch.bfloat16
    ) / math.sqrt(vh)
    gemm2_weights = torch.randn(
        num_experts, vh, vi, device=device, dtype=torch.bfloat16
    ) / math.sqrt(vi)
    permute_info, _ = routing_reference_renormalize(
        expert_logits, top_k, num_experts, padding
    )

    # Reference: the unpadded problem through the dequant MoE.
    weights_data = moe.quantize_weights(gemm1_weights, gemm2_weights, hidden_states)
    inputs_data = moe.quantize_inputs(
        hidden_states, weights_data["hidden_states_scale_global"]
    )
    args_valid = moe_args(
        num_tokens,
        num_experts,
        vh,
        vi,
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
    reference_output, args_dequant = moe.compute_reference(args_valid)
    reference_output = reference_output.to(torch.float)

    # Kernel: pad the linear-layout quantized weights, then shuffle as usual.
    if quant_mode == QuantMode.FP8_BLOCK_SCALE_MXFP8:
        sf_vec_size = 32
        w1_lin = weights_data["gemm1_weights"]
        s1_lin = weights_data["gemm1_scales"].reshape(
            num_experts, factor * vi, vh // sf_vec_size
        )
        w2_lin = weights_data["gemm2_weights"]
        s2_lin = weights_data["gemm2_scales"].reshape(
            num_experts, vh, vi // sf_vec_size
        )
    else:
        sf_vec_size = moe.sf_vec_size
        _, s1_lin, _ = quant_fp4_batches(
            gemm1_weights, num_experts, moe.is_mxfp4, False
        )
        _, s2_lin, _ = quant_fp4_batches(
            gemm2_weights, num_experts, moe.is_mxfp4, False
        )
        w1_lin = (
            weights_data["gemm1_weights"]
            .view(torch.uint8)
            .reshape(num_experts, factor * vi, vh // 2)
        )
        s1_lin = s1_lin.view(torch.uint8).reshape(
            num_experts, factor * vi, vh // sf_vec_size
        )
        w2_lin = (
            weights_data["gemm2_weights"]
            .view(torch.uint8)
            .reshape(num_experts, vh, vi // 2)
        )
        s2_lin = s2_lin.view(torch.uint8).reshape(num_experts, vh, vi // sf_vec_size)

    padded = pad_trtllm_moe_weights(
        w1_lin,
        s1_lin,
        w2_lin,
        s2_lin,
        vh,
        vi,
        is_gated_act=gated,
        hidden_alignment=hidden_alignment,
    )
    ph, pi = padded.hidden_size, padded.intermediate_size
    gemm2_rows = padded.gemm2_weights.shape[1]
    epilogue_tile_m = 128

    g1_w, g1_s, g2_w, g2_s = [], [], [], []
    for e in range(num_experts):
        w1 = padded.gemm1_weights[e].view(torch.uint8)
        s1 = padded.gemm1_weights_scale[e].view(torch.uint8)
        w2 = padded.gemm2_weights[e].view(torch.uint8)
        s2 = padded.gemm2_weights_scale[e].view(torch.uint8)
        if quant_mode == QuantMode.FP8_BLOCK_SCALE_MXFP8:
            if gated:
                w1 = reorder_rows_for_gated_act_gemm(w1)
                s1 = reorder_rows_for_gated_act_gemm(s1)
            g1_w.append(shuffle_matrix_a(w1, epilogue_tile_m))
            g1_s.append(shuffle_matrix_sf_a(s1, epilogue_tile_m))
            g2_w.append(shuffle_matrix_a(w2, epilogue_tile_m))
            g2_s.append(shuffle_matrix_sf_a(s2, epilogue_tile_m))
        else:
            p = _maybe_get_cached_w3_w1_permute_indices(
                moe._cache_permute_indices, w1, epilogue_tile_m, is_gated_act_gemm=gated
            )
            g1_w.append(w1[p.to(device)].contiguous())
            p = _maybe_get_cached_w3_w1_permute_indices(
                moe._cache_permute_indices,
                s1,
                epilogue_tile_m,
                num_elts_per_sf=sf_vec_size,
                is_gated_act_gemm=gated,
            )
            g1_s.append(block_scale_interleave(s1[p.to(device)].contiguous()))
            p = get_w2_permute_indices_with_cache(
                moe._cache_permute_indices, w2, epilogue_tile_m
            )
            g2_w.append(w2[p.to(device)].contiguous())
            p = get_w2_permute_indices_with_cache(
                moe._cache_permute_indices,
                s2,
                epilogue_tile_m,
                num_elts_per_sf=sf_vec_size,
            )
            g2_s.append(block_scale_interleave(s2[p.to(device)].contiguous()))
    gemm1_weights_k = torch.stack(g1_w)
    gemm2_weights_k = torch.stack(g2_w)
    if quant_mode == QuantMode.FP8_BLOCK_SCALE_MXFP8:
        gemm1_weights_k = gemm1_weights_k.view(torch.float8_e4m3fn)
        gemm2_weights_k = gemm2_weights_k.view(torch.float8_e4m3fn)
    gemm1_scales_k = torch.stack(g1_s).reshape(
        num_experts, factor * pi, ph // sf_vec_size
    )
    gemm2_scales_k = torch.stack(g2_s).reshape(
        num_experts, gemm2_rows, pi // sf_vec_size
    )

    # Poison the activation pad: the kernel must not read past roundUp(valid_hidden_size, 128).
    hidden_padded = torch.full(
        (num_tokens, ph), _CONTAM, device=device, dtype=torch.bfloat16
    )
    hidden_padded[:, :vh] = hidden_states
    kernel_inputs = moe.quantize_inputs(
        hidden_padded, weights_data["hidden_states_scale_global"], is_swizzling=False
    )
    common = dict(
        routing_bias=None,
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
        valid_hidden_size=vh,
        valid_intermediate_size=vi,
    )
    if quant_mode == QuantMode.FP8_BLOCK_SCALE_MXFP8:
        out = trtllm_fp8_block_scale_moe(
            routing_logits=expert_logits,
            hidden_states=kernel_inputs["hidden_states"],
            hidden_states_scale=kernel_inputs["hidden_states_scale"],
            gemm1_weights=gemm1_weights_k,
            gemm1_weights_scale=gemm1_scales_k,
            gemm2_weights=gemm2_weights_k,
            gemm2_weights_scale=gemm2_scales_k,
            use_shuffled_weight=True,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            **common,
        )
    else:
        g1_global = weights_data["gemm1_scales_global"]
        g2_global = weights_data["gemm2_scales_global"]
        hs_global = weights_data["hidden_states_scale_global"]
        c_global_sf = args_dequant.c_global_sf
        scale_gate_fc1 = (1.0 / g1_global) * (1.0 / hs_global)
        if gated:
            scale_c_fc1 = c_global_sf * scale_gate_fc1
        else:
            scale_c_fc1 = torch.full_like(g1_global, c_global_sf)
        scale_c_fc2 = (1.0 / c_global_sf) * (1.0 / g2_global)
        out = trtllm_fp4_block_scale_moe(
            routing_logits=expert_logits,
            hidden_states=kernel_inputs["hidden_states"],
            hidden_states_scale=kernel_inputs["hidden_states_scale"],
            gemm1_weights=gemm1_weights_k,
            gemm1_weights_scale=gemm1_scales_k.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=gemm2_weights_k,
            gemm2_weights_scale=gemm2_scales_k.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=scale_c_fc1,
            output1_scale_gate_scalar=scale_gate_fc1,
            output2_scale_scalar=scale_c_fc2,
            **common,
        )
    kernel_output = (out[0] if isinstance(out, list) else out).to(torch.float)
    return kernel_output, reference_output, moe.get_tolerances()


@pytest.mark.parametrize(
    "quant_mode", [QuantMode.FP8_BLOCK_SCALE_MXFP8, QuantMode.FP4_NVFP4_NVFP4]
)
@pytest.mark.parametrize(
    "activation_type", [ActivationType.Swiglu, ActivationType.Relu2]
)
@pytest.mark.parametrize("valid_hidden_size,valid_intermediate_size", _SHAPES)
def test_padded_moe_matches_unpadded_reference(
    quant_mode, activation_type, valid_hidden_size, valid_intermediate_size
):
    _skip_unless_sm100()
    if quant_mode == QuantMode.FP8_BLOCK_SCALE_MXFP8 and valid_intermediate_size % 32:
        pytest.skip("MxFP8 needs a 32-aligned intermediate_size")
    kernel_output, reference_output, tol = _run_padded_moe(
        quant_mode, activation_type, valid_hidden_size, valid_intermediate_size
    )
    assert kernel_output.shape == reference_output.shape == (64, valid_hidden_size)
    check_accuracy(
        reference_output, kernel_output, tol["atol"], tol["rtol"], tol["percent"]
    )
