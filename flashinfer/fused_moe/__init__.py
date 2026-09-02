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

import warnings

# Unified MoE API
from .api import (  # noqa: F401
    # Typed activation values
    ActivationConfig,
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    ReLU,
    ReLU2,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
    # Unified configs and packs
    B12xNvfp4Config,
    B12xW4A16Config,
    BackendOptions,
    CuteDslConfig,
    CutlassBf16Config,
    CutlassFp8BlockConfig,
    CutlassFp8PerTensorConfig,
    CutlassHummingConfig,
    CutlassMxfp8Config,
    CutlassMxfp8Mxfp4Config,
    CutlassNvfp4Config,
    CutlassW4A16Config,
    CutlassW4A8Config,
    CuTileBf16Config,
    CuTileNvfp4Config,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
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
from .da_runtime import (  # noqa: F401
    trtllm_moe_acquire_da_graph_leases,
    trtllm_moe_da_diagnostics,
    trtllm_moe_release_da_resources,
)
from .runners import (  # noqa: F401
    B12xNvfp4Runner,
    B12xW4A16Runner,
    CutlassBf16Runner,
    CutlassFp8BlockRunner,
    CutlassFp8PerTensorRunner,
    CutlassHummingRunner,
    CutlassMxfp8Mxfp4Runner,
    CutlassMxfp8Runner,
    CutlassNvfp4Runner,
    CutlassW4A16Runner,
    CutlassW4A8Runner,
    CuTileBf16Runner,
    CuTileNvfp4Runner,
    CuteDslRunner,
    TrtllmBf16RoutedRunner,
    TrtllmFp4RoutedRunner,
    TrtllmFp8BlockRunner,
    TrtllmFp8PerTensorRunner,
    TrtllmMxInt4RoutedRunner,
)

# Legacy flat-argument APIs (unchanged, not deprecated)
from .core import (
    RoutingInputMode,
    TrtllmMoERoutingMetadata,
    TrtllmMoERoutingMetadataSlot,
    convert_to_block_layout,
    cutlass_fused_moe,
    cutlass_fused_moe_workspace_size,
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
    trtllm_fp8_per_channel_scale_moe,
    trtllm_fp8_per_channel_scale_routed_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_fp8_per_tensor_scale_routed_moe,
    trtllm_moe_allocate_routing_metadata,
    trtllm_moe_allocate_routing_metadata_multi_tile,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_mxint4_block_scale_moe,
    trtllm_mxint4_block_scale_routed_moe,
)

from .prepare import (
    interleave_moe_scales_for_sm90_mixed_gemm,
    interleave_moe_weights_for_sm90_mixed_gemm,
    preprocess_moe_weights_for_sm90_mixed_gemm_humming,
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

from .hash_topk import (  # noqa: F401
    hash_topk as hash_topk,
)

from .trtllm_gen_routing import (  # noqa: F401
    TrtllmGenRoutingResult as TrtllmGenRoutingResult,
    trtllm_gen_routing as trtllm_gen_routing,
)

from .bgmv_moe import (  # noqa: F401
    BGMVMoEBlackwellPlan as BGMVMoEBlackwellPlan,
    bgmv_moe as bgmv_moe,
    bgmv_moe_shrink as bgmv_moe_shrink,
    bgmv_moe_expand as bgmv_moe_expand,
    fill_w_ptr as fill_w_ptr,
    has_bgmv_moe as has_bgmv_moe,
    prepare_bgmv_moe as prepare_bgmv_moe,
)
from .moe_lora_delta import (  # noqa: F401
    bgmv_moe_gemm1_lora_delta as bgmv_moe_gemm1_lora_delta,
    bgmv_moe_gemm2_lora_delta as bgmv_moe_gemm2_lora_delta,
)
from .monomoe import (  # noqa: F401
    mono_moe as mono_moe,
    has_monomoe as has_monomoe,
    alloc_scratchpad as alloc_scratchpad,
    get_scratchpad_size_bytes as get_scratchpad_size_bytes,
    interleave_for_tma_wgmma_up as interleave_for_tma_wgmma_up,
)

# CuteDSL MoE APIs (conditionally imported if cute_dsl available)
try:
    from .cute_dsl import (
        cute_dsl_fused_moe,
        CuteDslMoEWrapper,
        cute_dsl_fused_moe_nvfp4,
        cute_dsl_fused_moe_mxfp8_mxfp4,
        CuteDslMxfp8Mxfp4MoEWrapper,
        b12x_fused_moe,
        B12xMoEWrapper,
        cute_dsl_fused_moe_bf16,
        CuteDslBf16MoEWrapper,
    )

    _cute_dsl_available = True
except ImportError:
    _cute_dsl_available = False


def __getattr__(name: str):
    if name == "CuteDslNvfp4Runner":
        warnings.warn(
            "CuteDslNvfp4Runner is deprecated; use CuteDslRunner instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return CuteDslRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Typed activation values
    "ActivationConfig",
    "GELU",
    "SwiGLU",
    "SiTU",
    "GeGLU",
    "ReLU",
    "ReLU2",
    "SiLU",
    "GeGLUTanh",
    "SwiGLUStep",
    "Identity",
    # Unified configs, packs, and runners
    "B12xNvfp4Config",
    "B12xNvfp4Runner",
    "B12xW4A16Config",
    "B12xW4A16Runner",
    "BackendOptions",
    "CuteDslConfig",
    "CutlassBf16Config",
    "CutlassBf16Runner",
    "CutlassFp8BlockConfig",
    "CutlassFp8BlockRunner",
    "CutlassFp8PerTensorConfig",
    "CutlassFp8PerTensorRunner",
    "CutlassHummingConfig",
    "CutlassHummingRunner",
    "CutlassMxfp8Config",
    "CutlassMxfp8Mxfp4Config",
    "CutlassMxfp8Mxfp4Runner",
    "CutlassMxfp8Runner",
    "CutlassNvfp4Config",
    "CutlassNvfp4Runner",
    "CutlassW4A16Config",
    "CutlassW4A16Runner",
    "CutlassW4A8Config",
    "CutlassW4A8Runner",
    "CuTileBf16Config",
    "CuTileBf16Runner",
    "CuTileNvfp4Config",
    "CuTileNvfp4Runner",
    "ExecutionConfig",
    "ExpertConfig",
    "CuteDslRunner",
    "CuteDslNvfp4Runner",
    "MoEActivationPack",
    "RoutingInputMode",
    "TrtllmMoERoutingMetadata",
    "TrtllmMoERoutingMetadataSlot",
    "trtllm_moe_allocate_routing_metadata",
    "trtllm_moe_allocate_routing_metadata_multi_tile",
    "trtllm_moe_acquire_da_graph_leases",
    "trtllm_moe_da_diagnostics",
    "trtllm_moe_release_da_resources",
    "MoEConfig",
    "MoEFinalizeConfig",
    "MoELayer",
    "MoEWeightPack",
    "TrtllmBf16RoutedRunner",
    "TrtllmFp4RoutedRunner",
    "TrtllmFp8BlockRunner",
    "TrtllmFp8PerTensorRunner",
    "TrtllmMxInt4RoutedRunner",
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
    "cutlass_fused_moe_workspace_size",
    "interleave_moe_scales_for_sm90_mixed_gemm",
    "interleave_moe_weights_for_sm90_mixed_gemm",
    "preprocess_moe_weights_for_sm90_mixed_gemm_humming",
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
    "trtllm_fp8_per_channel_scale_moe",
    "trtllm_fp8_per_channel_scale_routed_moe",
    "trtllm_fp8_per_tensor_scale_moe",
    "trtllm_fp8_per_tensor_scale_routed_moe",
    "trtllm_mxint4_block_scale_moe",
    "trtllm_mxint4_block_scale_routed_moe",
    "fused_topk_deepseek",
    "hash_topk",
    "TrtllmGenRoutingResult",
    "trtllm_gen_routing",
    "bgmv_moe",
    "BGMVMoEBlackwellPlan",
    "bgmv_moe_shrink",
    "bgmv_moe_expand",
    "bgmv_moe_gemm1_lora_delta",
    "bgmv_moe_gemm2_lora_delta",
    "fill_w_ptr",
    "has_bgmv_moe",
    "prepare_bgmv_moe",
    "mono_moe",
    "has_monomoe",
    "alloc_scratchpad",
    "get_scratchpad_size_bytes",
    "interleave_for_tma_wgmma_up",
]

# Add CuteDSL exports if available
if _cute_dsl_available:
    __all__ += [
        "cute_dsl_fused_moe",
        "cute_dsl_fused_moe_nvfp4",
        "cute_dsl_fused_moe_mxfp8_mxfp4",
        "CuteDslMoEWrapper",
        "CuteDslMxfp8Mxfp4MoEWrapper",
        "b12x_fused_moe",
        "B12xMoEWrapper",
        "cute_dsl_fused_moe_bf16",
        "CuteDslBf16MoEWrapper",
    ]
