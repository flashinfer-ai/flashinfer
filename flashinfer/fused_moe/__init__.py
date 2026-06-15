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

# Unified MoE API
from .api import (  # noqa: F401
    ActivationConfig,
    BackendOptions,
    CuteDslConfig,
    CutlassConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    TrtllmBf16Config,
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmMxInt4Config,
)
from .layer import MoELayer  # noqa: F401
from .runners import CuteDslNvfp4Runner, TrtllmFp4RoutedRunner  # noqa: F401

# Legacy flat-argument APIs (unchanged, not deprecated)
from .core import (
    convert_to_block_layout,
    cutlass_fused_moe,
    interleave_moe_scales_for_sm90_mixed_gemm,
    interleave_moe_weights_for_sm90_mixed_gemm,
    gen_cutlass_fused_moe_sm120_module,
    gen_cutlass_fused_moe_sm103_module,
    gen_cutlass_fused_moe_sm100_module,
    gen_cutlass_fused_moe_sm90_module,
    gen_trtllm_gen_fused_moe_sm100_module,
    reorder_rows_for_gated_act_gemm,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_mxint4_block_scale_moe,
    trtllm_mxint4_block_scale_routed_moe,
)

from ..tllm_enums import (
    ActivationType,
    Fp8QuantizationType,
    WeightLayout,
    RoutingMethodType,
)

from .fused_routing_dsv3 import (  # noqa: F401
    fused_topk_deepseek as fused_topk_deepseek,
)

from .bgmv_moe import (  # noqa: F401
    bgmv_moe as bgmv_moe,
    bgmv_moe_shrink as bgmv_moe_shrink,
    bgmv_moe_expand as bgmv_moe_expand,
    fill_w_ptr as fill_w_ptr,
    has_bgmv_moe as has_bgmv_moe,
)

# CuteDSL MoE APIs (conditionally imported if cute_dsl available)
try:
    from .cute_dsl import (
        cute_dsl_fused_moe_nvfp4,
        CuteDslMoEWrapper,
        b12x_fused_moe,
        B12xMoEWrapper,
    )

    _cute_dsl_available = True
except ImportError:
    _cute_dsl_available = False

__all__ = [
    # Unified API
    "ActivationConfig",
    "BackendOptions",
    "CuteDslConfig",
    "CutlassConfig",
    "ExecutionConfig",
    "ExpertConfig",
    "CuteDslNvfp4Runner",
    "MoEActivationPack",
    "MoEConfig",
    "MoELayer",
    "MoEWeightPack",
    "TrtllmFp4RoutedRunner",
    "QuantConfig",
    "QuantVariant",
    "RoutingConfig",
    "TrtllmBf16Config",
    "TrtllmFp4Config",
    "TrtllmFp8BlockConfig",
    "TrtllmFp8PerTensorConfig",
    "TrtllmMxInt4Config",
    # Legacy flat APIs
    "ActivationType",
    "Fp8QuantizationType",
    "RoutingMethodType",
    "WeightLayout",
    "convert_to_block_layout",
    "cutlass_fused_moe",
    "interleave_moe_scales_for_sm90_mixed_gemm",
    "interleave_moe_weights_for_sm90_mixed_gemm",
    "gen_cutlass_fused_moe_sm120_module",
    "gen_cutlass_fused_moe_sm103_module",
    "gen_cutlass_fused_moe_sm100_module",
    "gen_cutlass_fused_moe_sm90_module",
    "gen_trtllm_gen_fused_moe_sm100_module",
    "reorder_rows_for_gated_act_gemm",
    "trtllm_bf16_moe",
    "trtllm_bf16_routed_moe",
    "trtllm_fp4_block_scale_moe",
    "trtllm_fp4_block_scale_routed_moe",
    "trtllm_fp8_block_scale_moe",
    "trtllm_fp8_block_scale_routed_moe",
    "trtllm_fp8_per_tensor_scale_moe",
    "trtllm_mxint4_block_scale_moe",
    "trtllm_mxint4_block_scale_routed_moe",
    "fused_topk_deepseek",
    "bgmv_moe",
    "bgmv_moe_shrink",
    "bgmv_moe_expand",
    "fill_w_ptr",
    "has_bgmv_moe",
]

# Add CuteDSL exports if available
if _cute_dsl_available:
    __all__ += [
        "cute_dsl_fused_moe_nvfp4",
        "CuteDslMoEWrapper",
        "b12x_fused_moe",
        "B12xMoEWrapper",
    ]
