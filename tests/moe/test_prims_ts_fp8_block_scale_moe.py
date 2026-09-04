# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.autotuner import autotune
from .trtllm_gen_fused_moe_utils import (
    FP8BlockScaleMoe,
    MoeGemmBackend,
    check_accuracy,
    moe_args,
    pack_topk_for_routed_moe,
    routing_reference_renormalize,
    run_moe_test,
)
from .utils import QuantMode
from flashinfer.fused_moe import (
    RoutingMethodType,
    WeightLayout,
    prims_ts_fp8_block_scale_moe,
    prims_ts_fp8_block_scale_routed_moe,
)
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.tllm_enums import ActivationType, Fp8QuantizationType
from flashinfer.utils import device_support_pdl, get_compute_capability


@pytest.fixture(scope="module")
def cache_permute_indices():
    return {}


@pytest.mark.parametrize(
    ("quant_mode", "case_id"),
    [
        pytest.param(QuantMode.FP8_BLOCK_SCALE_MXFP8, "mxfp8", id="MxFp8"),
        pytest.param(QuantMode.FP8_BLOCK_SCALE_DEEPSEEK, "deepseek", id="DeepSeekFp8"),
    ],
)
def test_prims_ts_fp8_block_scale_moe_smoke(
    quant_mode,
    case_id,
    cache_permute_indices,
):
    if case_id == "deepseek":
        num_tokens = 128
        hidden_size = 512
        intermediate_size = 512
        num_experts = 64
        top_k = 8
    else:
        num_tokens = 32
        hidden_size = 512
        intermediate_size = 512
        num_experts = 64
        top_k = 8

    run_moe_test(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_impl=FP8BlockScaleMoe(fp8_quantization_type=quant_mode),
        routing_config={
            "num_experts": num_experts,
            "top_k": top_k,
            "padding": 8,
            "n_groups": None,
            "top_k_groups": None,
            "routed_scaling": None,
            "has_routing_bias": False,
            "routing_method_type": RoutingMethodType.Renormalize,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_intermediate_size": [intermediate_size],
            "compatible_activation_types": [ActivationType.Swiglu],
            "enable_autotune": False,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_gemm_backends": [MoeGemmBackend.PRIMS_TS],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.bfloat16,
        moe_gemm_backend=MoeGemmBackend.PRIMS_TS,
    )


def test_prims_ts_deepseek_fp8_block_scale_tile16_smoke(
    cache_permute_indices,
):
    run_moe_test(
        num_tokens=128,
        hidden_size=512,
        intermediate_size=512,
        moe_impl=FP8BlockScaleMoe(
            fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
        ),
        routing_config={
            "num_experts": 64,
            "top_k": 8,
            "padding": 8,
            "n_groups": None,
            "top_k_groups": None,
            "routed_scaling": None,
            "has_routing_bias": False,
            "routing_method_type": RoutingMethodType.Renormalize,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_intermediate_size": [512],
            "compatible_activation_types": [ActivationType.Swiglu],
            "enable_autotune": False,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_gemm_backends": [MoeGemmBackend.PRIMS_TS],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.bfloat16,
        moe_gemm_backend=MoeGemmBackend.PRIMS_TS,
    )


def test_prims_ts_deepseek_fp8_accepts_fp32_logits(cache_permute_indices):
    run_moe_test(
        num_tokens=32,
        hidden_size=512,
        intermediate_size=512,
        moe_impl=FP8BlockScaleMoe(
            fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_DEEPSEEK
        ),
        routing_config={
            "num_experts": 64,
            "top_k": 4,
            "padding": 8,
            "n_groups": 8,
            "top_k_groups": 4,
            "routed_scaling": 2.5,
            "has_routing_bias": True,
            "routing_method_type": RoutingMethodType.DeepSeekV3,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_intermediate_size": [512],
            "compatible_activation_types": [ActivationType.Swiglu],
            "enable_autotune": False,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_gemm_backends": [MoeGemmBackend.PRIMS_TS],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.float32,
        moe_gemm_backend=MoeGemmBackend.PRIMS_TS,
    )


@pytest.mark.parametrize("bias", ["gemm2", "gemm1", "gemm1_and_gemm2"])
@pytest.mark.parametrize(
    "moe_gemm_backend",
    [
        pytest.param(MoeGemmBackend.TRTLLM, id="TRTLLM"),
        pytest.param(MoeGemmBackend.PRIMS_TS, id="PrimsTS"),
    ],
)
def test_prims_ts_mxfp8_block_scale_bias(
    bias,
    moe_gemm_backend,
    cache_permute_indices,
):
    num_tokens = 32
    hidden_size = 512
    intermediate_size = 512
    num_experts = 64
    top_k = 8
    device = "cuda"

    gemm1_bias = None
    gemm2_bias = None
    if "gemm1" in bias:
        gemm1_bias = torch.randn(
            (num_experts, 2 * intermediate_size), device=device, dtype=torch.float32
        )
    if "gemm2" in bias:
        gemm2_bias = torch.randn(
            (num_experts, hidden_size), device=device, dtype=torch.float32
        )

    run_moe_test(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_impl=FP8BlockScaleMoe(
            fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8
        ),
        routing_config={
            "num_experts": num_experts,
            "top_k": top_k,
            "padding": 8,
            "n_groups": None,
            "top_k_groups": None,
            "routed_scaling": None,
            "has_routing_bias": False,
            "routing_method_type": RoutingMethodType.Renormalize,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_intermediate_size": [intermediate_size],
            "compatible_activation_types": [ActivationType.Swiglu],
            "enable_autotune": False,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": WeightLayout.MajorK,
            "compatible_moe_impls": [FP8BlockScaleMoe],
            "compatible_gemm_backends": [moe_gemm_backend],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices=cache_permute_indices,
        routing_logits_dtype=torch.bfloat16,
        moe_gemm_backend=moe_gemm_backend,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
    )


def test_prims_ts_mxfp8_block_scale_routed_modes_match_logits(
    cache_permute_indices,
):
    """Packed and unpacked MXFP8 Prims-TS routed inputs match the logits path."""
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    if not is_prims_ts_available():
        pytest.skip("Prims-TS dependencies are unavailable")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    num_tokens = 8
    hidden_size = 128
    intermediate_size = 128
    num_experts = 8
    top_k = 2
    padding = 8
    activation_type = ActivationType.Swiglu

    moe_impl = FP8BlockScaleMoe(fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8)
    moe_impl._cache_permute_indices = cache_permute_indices
    hidden_states = torch.randn(
        (num_tokens, hidden_size), device=device, dtype=torch.bfloat16
    )
    gemm1_weights = (
        torch.randn(
            (num_experts, 2 * intermediate_size, hidden_size),
            device=device,
            dtype=torch.bfloat16,
        )
        / hidden_size**0.5
    )
    gemm2_weights = (
        torch.randn(
            (num_experts, hidden_size, intermediate_size),
            device=device,
            dtype=torch.bfloat16,
        )
        / intermediate_size**0.5
    )
    routing_logits = torch.randn(
        (num_tokens, num_experts), device=device, dtype=torch.bfloat16
    )

    permute_info, scores = routing_reference_renormalize(
        routing_logits, top_k, num_experts, padding
    )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    topk_weights = scores.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)

    weights_data = moe_impl.quantize_weights(
        gemm1_weights, gemm2_weights, hidden_states
    )
    inputs_data = moe_impl.quantize_inputs(
        hidden_states, weights_data["hidden_states_scale_global"]
    )
    quant_data = {**weights_data, **inputs_data}
    args = moe_args(
        num_tokens,
        num_experts,
        hidden_size,
        intermediate_size,
        top_k,
        padding,
        quant_data["hidden_states"],
        quant_data["hidden_states_scale"],
        quant_data["hidden_states_scale_global"],
        scores,
        quant_data["gemm1_weights"],
        quant_data["gemm1_scales"],
        quant_data["gemm1_scales_global"],
        quant_data["gemm2_weights"],
        quant_data["gemm2_scales"],
        quant_data["gemm2_scales_global"],
        permute_info,
        False,
        activation_type,
    )
    _, args_dequant = moe_impl.compute_reference(args)
    static_data = moe_impl.prepare_static_weights_for_kernel(
        args_dequant,
        args,
        gemm1_weights,
        gemm2_weights,
        hidden_size,
        intermediate_size,
        num_experts,
        {"use_shuffled_weight": True, "layout": WeightLayout.MajorK},
    )

    common_kwargs = dict(
        routing_bias=None,
        hidden_states=quant_data["hidden_states"],
        hidden_states_scale=quant_data["hidden_states_scale"],
        gemm1_weights=static_data["gemm1_weights"],
        gemm1_weights_scale=static_data["gemm1_scales"],
        gemm2_weights=static_data["gemm2_weights"],
        gemm2_weights_scale=static_data["gemm2_scales"],
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        use_shuffled_weight=True,
        weight_layout=WeightLayout.MajorK,
        do_finalize=True,
        enable_pdl=device_support_pdl(device),
        tune_max_num_tokens=4096,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=activation_type.value,
        norm_topk_prob=True,
    )
    routed_kwargs = dict(common_kwargs)
    routed_kwargs.pop("norm_topk_prob")

    with autotune(False):
        logits_output = prims_ts_fp8_block_scale_moe(
            routing_logits=routing_logits,
            **common_kwargs,
        ).to(torch.float)
        packed_output = prims_ts_fp8_block_scale_routed_moe(
            pack_topk_for_routed_moe(topk_ids, topk_weights),
            **routed_kwargs,
        ).to(torch.float)
        unpacked_output = prims_ts_fp8_block_scale_routed_moe(
            (topk_ids, topk_weights),
            **routed_kwargs,
        ).to(torch.float)

    check_accuracy(logits_output, packed_output, atol=1e-2, rtol=1e-2, percent=0.99)
    check_accuracy(logits_output, unpacked_output, atol=1e-2, rtol=1e-2, percent=0.99)
