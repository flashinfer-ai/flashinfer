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

"""
CuteDSL-based Fused MoE API for NVFP4 on Blackwell GPUs.

This module provides high-level APIs for running Mixture of Experts (MoE)
computations using CuteDSL kernels. The implementation follows TensorRT-LLM's
CuteDslFusedMoE pattern closely.

Auto-tuning Usage:
    The kernel supports automatic performance tuning via the `autotune` context
    manager. When called inside `with autotune(True):`, the kernel will profile
    different GEMM tactics and cache the best performing configuration.

    Example:
        >>> from flashinfer import autotune
        >>> from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4
        >>>
        >>> # Without tuning (uses cached or default tactics)
        >>> output = cute_dsl_fused_moe_nvfp4(x, x_sf, ..., num_experts=8, top_k=2)
        >>>
        >>> # With auto-tuning
        >>> with autotune(True):
        ...     output = cute_dsl_fused_moe_nvfp4(x, x_sf, ..., num_experts=8, top_k=2)

CUDA Graph Compatibility:
    This module is designed to be compatible with CUDA graph capture. Key features:
    - moe_sort returns tensors (no CPU-GPU sync with .item())
    - CUDA events and streams are pre-allocated at module level
    - When using CUDA graphs, pre-allocate the moe_output buffer before capture
"""

from typing import Optional, Tuple, Dict, Any

import torch

from ..api_logging import flashinfer_api
from ..autotuner import AutoTuner
from ..moe_utils import moe_sort, moe_output_memset, get_max_num_permuted_tokens


# Module-level cache for CUDA graph resources (events and streams)
# Pre-allocating these outside of CUDA graph capture is required for compatibility
_cuda_graph_resources: Dict[str, Any] = {}


def _get_cuda_graph_resources() -> Dict[str, Any]:
    """Get or create pre-allocated CUDA events and streams for CUDA graph compatibility.

    These resources must be created outside of CUDA graph capture to ensure
    that event.wait() and stream operations work correctly during graph capture.
    """
    if not _cuda_graph_resources:
        _cuda_graph_resources["main_event"] = torch.cuda.Event()
        _cuda_graph_resources["memset_event"] = torch.cuda.Event()
        _cuda_graph_resources["aux_stream"] = torch.cuda.Stream()
    return _cuda_graph_resources


from .blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import (
    blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4,
)
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
)
from .tuner import (
    CuteDslFusedMoENvfp4Runner,
)


def _cute_dsl_fused_moe_nvfp4_impl(
    # Input (already NVFP4 quantized)
    x: torch.Tensor,
    x_sf: torch.Tensor,
    # Routing
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    # GEMM1 weights (gate + up projection, fused)
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    # GEMM2 intermediate quantization scale
    fc2_input_scale: torch.Tensor,
    # GEMM2 weights (down projection)
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    # MoE configuration
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int = 0,
    # Tactic parameters (TRT-LLM style)
    tile_size: int = 128,
    gemm1_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm1_cluster_shape_mn: Tuple[int, int] = (1, 1),
    gemm2_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm2_cluster_shape_mn: Tuple[int, int] = (1, 1),
    # Options
    output_dtype: torch.dtype = torch.bfloat16,
    use_fused_finalize: bool = True,
    moe_output: Optional[torch.Tensor] = None,
    # Stream optimization (for fused finalize)
    aux_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Internal implementation of fused MoE with explicit tactic parameters.

    This function is called by the TunableRunner during auto-tuning.
    """
    num_tokens = token_selected_experts.size(0)
    hidden_size = w2_weight.size(1)  # GEMM2 output dim

    # Allocate output buffer if not provided
    if moe_output is None:
        moe_output = torch.empty(
            (num_tokens, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )

    # Step 1: Sort tokens by expert and generate mapping tensors
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_tokens_dim=tile_size,
    )

    # Set up stream synchronization for fused finalize
    # Use pre-allocated resources for CUDA graph compatibility
    if use_fused_finalize:
        resources = _get_cuda_graph_resources()
        main_event = resources["main_event"]
        memset_event = resources["memset_event"]

        if aux_stream is None:
            aux_stream = resources["aux_stream"]

        main_event.record()
        moe_output.record_stream(aux_stream)

    # Step 2: GEMM1 + SwiGLU (gather directly from input, no physical permute)
    # Output is quantized to NVFP4 for GEMM2
    intermediate, intermediate_sf = (
        blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4(
            a=x,
            b=w1_weight,
            a_scale=x_sf,
            b_scale=w1_weight_sf,
            alpha=w1_alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            token_id_mapping=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_scale=fc2_input_scale,
            topk=top_k,
            c_dtype="float4_e2m1fn",  # Output FP4 for GEMM2 input
            mma_tiler_mn=gemm1_mma_tiler_mn,
            cluster_shape_mn=gemm1_cluster_shape_mn,
        )
    )

    if use_fused_finalize:
        # Step 3: Async moe_output_memset on auxiliary stream
        max_num_permuted_tokens = get_max_num_permuted_tokens(
            num_tokens, top_k, num_local_experts, tile_size
        )

        with torch.cuda.stream(aux_stream):
            main_event.wait()
            moe_output_memset(
                output=moe_output,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                max_num_permuted_tokens=max_num_permuted_tokens,
                top_k=top_k,
                tile_size=tile_size,
            )
            memset_event.record()

        # Wait for memset to complete before GEMM2
        memset_event.wait()

        # Step 4: GEMM2 + Finalize (atomic scatter to output)
        blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
            a=intermediate,
            b=w2_weight,
            a_scale=intermediate_sf,
            b_scale=w2_weight_sf,
            alpha=w2_alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            out=moe_output,
            mma_tiler_mn=gemm2_mma_tiler_mn,
            cluster_shape_mn=gemm2_cluster_shape_mn,
        )
    else:
        # Non-fused path: separate GEMM2 and unpermute
        raise NotImplementedError(
            "Non-fused finalize path not yet implemented. Use use_fused_finalize=True."
        )

    return moe_output


@flashinfer_api
def cute_dsl_fused_moe_nvfp4(
    # Input (already NVFP4 quantized)
    x: torch.Tensor,
    x_sf: torch.Tensor,
    # Routing
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    # GEMM1 weights (gate + up projection, fused)
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    # GEMM2 intermediate quantization scale
    fc2_input_scale: torch.Tensor,
    # GEMM2 weights (down projection)
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    # MoE configuration
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    # Options
    output_dtype: torch.dtype = torch.bfloat16,
    use_fused_finalize: bool = True,
    moe_output: Optional[torch.Tensor] = None,
    # Stream optimization (for fused finalize)
    aux_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """
    Run fused MoE computation using CuteDSL NVFP4 kernels on Blackwell GPUs.

    This function implements the complete NVFP4 MoE pipeline:
    1. Sort tokens by expert assignment (moe_sort)
    2. GEMM1 + SwiGLU with gather (no physical permute needed)
    3. Async moe_output_memset on auxiliary stream (if use_fused_finalize)
    4. GEMM2 + Finalize with atomic scatter (if use_fused_finalize)

    Auto-tuning is controlled via the `autotune` context manager. When called
    inside `with autotune(True):`, this function will profile different GEMM
    tactics and cache the best performing configuration.

    Args:
        x: Input tensor, NVFP4 quantized. Shape: [num_tokens, hidden_size // 2].
           Dtype: torch.uint8 (packed FP4, 2 values per byte).
        x_sf: Scale factors for x. Shape: [num_tokens, hidden_size // sf_vec_size].
              Dtype: torch.uint8 (FP8 E4M3).
        token_selected_experts: Expert assignments. Shape: [num_tokens, top_k].
                               Dtype: torch.int32.
        token_final_scales: Routing weights. Shape: [num_tokens, top_k].
                           Dtype: torch.float32.
        w1_weight: GEMM1 weights (gate + up fused).
                   Shape: [num_experts, 2 * intermediate_size, hidden_size // 2].
                   Dtype: torch.uint8 (packed FP4).
        w1_weight_sf: Scale factors for w1_weight.
                      Shape: [num_experts, 2 * intermediate_size, hidden_size // sf_vec_size].
                      Dtype: torch.uint8.
        w1_alpha: Per-expert global scale for GEMM1. Shape: [num_experts].
                  Dtype: torch.float32.
        fc2_input_scale: Global scale for quantizing GEMM2 input.
                         Shape: [num_experts] or scalar.
                         Dtype: torch.float32.
        w2_weight: GEMM2 weights (down projection).
                   Shape: [num_experts, hidden_size, intermediate_size // 2].
                   Dtype: torch.uint8 (packed FP4).
        w2_weight_sf: Scale factors for w2_weight.
                      Shape: [num_experts, hidden_size, intermediate_size // sf_vec_size].
                      Dtype: torch.uint8.
        w2_alpha: Per-expert global scale for GEMM2. Shape: [num_experts].
                  Dtype: torch.float32.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        num_local_experts: Number of local experts (for expert parallelism).
                          Default: num_experts.
        local_expert_offset: Expert offset for expert parallelism. Default: 0.
        output_dtype: Output data type. Default: torch.bfloat16.
        use_fused_finalize: If True, fuse unpermute and scaling into GEMM2 epilogue.
                           Default: True.
        moe_output: Pre-allocated output buffer. Shape: [num_tokens, hidden_size].
                    If None, will be allocated.
        aux_stream: Auxiliary CUDA stream for moe_output_memset.
                    If None and use_fused_finalize, a module-level cached stream
                    will be used (enables CUDA graph compatibility).

    Returns:
        Output tensor. Shape: [num_tokens, hidden_size]. Dtype: output_dtype.

    Note:
        This function is CUDA graph compatible. When using CUDA graphs:
        - Pre-allocate moe_output buffer before graph capture
        - The aux_stream parameter can be None (uses cached stream) or a
          pre-allocated stream created before graph capture

    Example:
        >>> import torch
        >>> from flashinfer import autotune, fp4_quantize
        >>> from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4
        >>>
        >>> # Quantize input
        >>> x_fp16 = torch.randn(128, 4096, device="cuda", dtype=torch.float16)
        >>> x, x_sf = fp4_quantize(x_fp16)
        >>>
        >>> # Expert routing
        >>> token_selected_experts = torch.randint(0, 8, (128, 2), dtype=torch.int32, device="cuda")
        >>> token_final_scales = torch.softmax(torch.randn(128, 2, device="cuda"), dim=-1).float()
        >>>
        >>> # Run MoE without tuning (uses cached or default tactics)
        >>> output = cute_dsl_fused_moe_nvfp4(
        ...     x=x, x_sf=x_sf,
        ...     token_selected_experts=token_selected_experts,
        ...     token_final_scales=token_final_scales,
        ...     w1_weight=w1_weight, w1_weight_sf=w1_weight_sf, w1_alpha=w1_alpha,
        ...     fc2_input_scale=fc2_input_scale,
        ...     w2_weight=w2_weight, w2_weight_sf=w2_weight_sf, w2_alpha=w2_alpha,
        ...     num_experts=8, top_k=2,
        ... )
        >>>
        >>> # Run MoE with auto-tuning
        >>> with autotune(True):
        ...     output = cute_dsl_fused_moe_nvfp4(
        ...         x=x, x_sf=x_sf,
        ...         token_selected_experts=token_selected_experts,
        ...         token_final_scales=token_final_scales,
        ...         w1_weight=w1_weight, w1_weight_sf=w1_weight_sf, w1_alpha=w1_alpha,
        ...         fc2_input_scale=fc2_input_scale,
        ...         w2_weight=w2_weight, w2_weight_sf=w2_weight_sf, w2_alpha=w2_alpha,
        ...         num_experts=8, top_k=2,
        ...     )
    """
    if num_local_experts is None:
        num_local_experts = num_experts

    num_tokens = token_selected_experts.size(0)
    hidden_size = w2_weight.size(1)  # GEMM2 output dim

    # Allocate output buffer if not provided
    if moe_output is None:
        moe_output = torch.empty(
            (num_tokens, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )

    # Get auto-tuner
    tuner = AutoTuner.get()

    runner = CuteDslFusedMoENvfp4Runner(
        forward_impl=_cute_dsl_fused_moe_nvfp4_impl,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        use_fused_finalize=use_fused_finalize,
        output_dtype=output_dtype,
    )

    # Pack inputs for the runner
    inputs = [
        x,
        x_sf,
        token_selected_experts,
        token_final_scales,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        fc2_input_scale,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        moe_output,
    ]

    # choose_one handles both tuning mode and inference mode:
    # - In tuning mode (inside `with autotune(True):`): profiles all tactics
    # - In inference mode: uses cached result or fallback to default
    _, best_tactic = tuner.choose_one(
        "CuteDslFusedMoE::run_moe_nvfp4",
        [runner],
        CuteDslFusedMoENvfp4Runner.tuning_config,  # Use class-level tuning config
        inputs,
        aux_stream=aux_stream,
    )

    return runner(inputs, tactic=best_tactic, aux_stream=aux_stream)


__all__ = [
    "cute_dsl_fused_moe_nvfp4",
]
