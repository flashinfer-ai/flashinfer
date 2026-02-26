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
computations using CuteDSL kernels.

Two APIs are provided:

1. **Functional API** (`cute_dsl_fused_moe_nvfp4`):
   Simple function call with auto-tuning support via `autotune()` context.
   Best for: simple use cases, experimenting, auto-tuning.

2. **Wrapper API** (`CuteDslMoEWrapper`):
   Class-based API with pre-allocated buffers for CUDA graph compatibility.
   Best for: production inference with CUDA graphs, fine-grained control.

Both APIs share the same core implementation and support auto-tuning.

Example (Functional API):
    >>> from flashinfer.cute_dsl import cute_dsl_fused_moe_nvfp4
    >>> output = cute_dsl_fused_moe_nvfp4(x, x_sf, ..., num_experts=8, top_k=2)

Example (Wrapper API with CUDA Graph):
    >>> from flashinfer.cute_dsl import CuteDslMoEWrapper
    >>> moe = CuteDslMoEWrapper(num_experts=256, top_k=8, ..., use_cuda_graph=True)
    >>> # Warmup
    >>> for _ in range(3):
    ...     output = moe.run(x, x_sf, topk_ids, topk_weights, w1, w1_sf, ...)
    >>> # Capture
    >>> with torch.cuda.graph(g):
    ...     output = moe.run(x, x_sf, topk_ids, topk_weights, w1, w1_sf, ...)
    >>> # Replay
    >>> g.replay()
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ...api_logging import flashinfer_api
from ...autotuner import AutoTuner
from .moe_utils import (
    allocate_moe_sort_buffers,
    get_max_num_permuted_tokens,
    moe_output_memset,
    moe_sort,
)
from .blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion import (
    blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion_nvfp4,
)
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
)
from .tuner import (
    ALL_MOE_TACTICS,
    CuteDslFusedMoENvfp4Runner,
)


# =============================================================================
# Module-level Resources for CUDA Graph Compatibility
# =============================================================================

_cuda_graph_resources: Dict[str, Any] = {}


def _get_cuda_graph_resources() -> Dict[str, Any]:
    """Get or create pre-allocated CUDA events and streams.

    These resources must be created outside CUDA graph capture.
    """
    if not _cuda_graph_resources:
        _cuda_graph_resources["main_event"] = torch.cuda.Event()
        _cuda_graph_resources["memset_event"] = torch.cuda.Event()
        _cuda_graph_resources["aux_stream"] = torch.cuda.Stream()
    return _cuda_graph_resources


# =============================================================================
# Core Implementation (Shared by Functional and Wrapper APIs)
# =============================================================================


def _moe_core_impl(
    # Input
    x: torch.Tensor,
    x_sf: torch.Tensor,
    # Routing
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    # GEMM1 weights
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    # GEMM2 intermediate scale
    fc2_input_scale: torch.Tensor,
    # GEMM2 weights
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    # MoE config
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int = 0,
    # Tactic parameters
    tile_size: int = 128,
    gemm1_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm1_cluster_shape_mn: Tuple[int, int] = (1, 1),
    gemm2_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm2_cluster_shape_mn: Tuple[int, int] = (1, 1),
    # Pre-allocated buffers (for CUDA graph)
    moe_sort_buffers: Optional[Dict[str, torch.Tensor]] = None,
    gemm1_out: Optional[torch.Tensor] = None,
    gemm1_out_scale: Optional[torch.Tensor] = None,
    moe_output: Optional[torch.Tensor] = None,
    # Stream resources
    aux_stream: Optional[torch.cuda.Stream] = None,
    main_event: Optional[torch.cuda.Event] = None,
    memset_event: Optional[torch.cuda.Event] = None,
    # Options
    output_dtype: torch.dtype = torch.bfloat16,
    use_async_memset: bool = True,
) -> torch.Tensor:
    """Core MoE implementation shared by functional and wrapper APIs.

    This function handles:
    1. moe_sort: Token routing computation
    2. GEMM1 + SwiGLU: First projection with activation
    3. Async moe_output_memset: Zero output buffer (overlapped with GEMM1)
    4. GEMM2 + Finalize: Second projection with atomic scatter

    Args:
        x: Input tensor, NVFP4 quantized.
        x_sf: Scale factors for x.
        token_selected_experts: Expert assignments [num_tokens, top_k].
        token_final_scales: Routing weights [num_tokens, top_k].
        w1_weight: GEMM1 weights (gate + up fused).
        w1_weight_sf: Scale factors for w1_weight.
        w1_alpha: Per-expert global scale for GEMM1.
        fc2_input_scale: Global scale for GEMM2 input quantization.
        w2_weight: GEMM2 weights (down projection).
        w2_weight_sf: Scale factors for w2_weight.
        w2_alpha: Per-expert global scale for GEMM2.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        num_local_experts: Number of local experts (for EP).
        local_expert_offset: Expert offset for EP.
        tile_size: Tile size for moe_sort.
        gemm1_mma_tiler_mn: GEMM1 MMA tiler shape.
        gemm1_cluster_shape_mn: GEMM1 cluster shape.
        gemm2_mma_tiler_mn: GEMM2 MMA tiler shape.
        gemm2_cluster_shape_mn: GEMM2 cluster shape.
        moe_sort_buffers: Pre-allocated moe_sort output buffers.
        gemm1_out: Pre-allocated GEMM1 output buffer.
        gemm1_out_scale: Pre-allocated GEMM1 output scale buffer.
        moe_output: Pre-allocated final output buffer.
        aux_stream: Auxiliary CUDA stream for async memset.
        main_event: CUDA event for main stream.
        memset_event: CUDA event for memset completion.
        output_dtype: Output data type.
        use_async_memset: Use async memset on aux stream.

    Returns:
        Output tensor [num_tokens, hidden_size].
    """
    num_tokens = token_selected_experts.size(0)
    hidden_size = w2_weight.size(1)

    # Allocate output if not provided
    if moe_output is None:
        moe_output = torch.empty(
            (num_tokens, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )

    # Get stream resources if using async memset
    if use_async_memset:
        if aux_stream is None or main_event is None or memset_event is None:
            resources = _get_cuda_graph_resources()
            aux_stream = aux_stream or resources["aux_stream"]
            main_event = main_event or resources["main_event"]
            memset_event = memset_event or resources["memset_event"]

    # Step 1: Sort tokens by expert
    moe_sort_kwargs = moe_sort_buffers or {}
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
        **moe_sort_kwargs,
    )

    # Record event for async memset synchronization
    if use_async_memset:
        main_event.record()
        moe_output.record_stream(aux_stream)

    # Step 2: GEMM1 + SwiGLU
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
            out=gemm1_out,
            out_scale=gemm1_out_scale,
            global_scale=fc2_input_scale,
            topk=top_k,
            c_dtype="float4_e2m1fn",
            mma_tiler_mn=gemm1_mma_tiler_mn,
            cluster_shape_mn=gemm1_cluster_shape_mn,
        )
    )

    # Step 3: Async moe_output_memset on auxiliary stream
    if use_async_memset:
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
        memset_event.wait()
    else:
        # Simple zero without async
        moe_output[:num_tokens].zero_()

    # Step 4: GEMM2 + Finalize
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

    return moe_output[:num_tokens]


# =============================================================================
# Wrapper API (Class-based, CUDA Graph Compatible)
# =============================================================================


class CuteDslMoEWrapper:
    """Wrapper class for CuteDSL MoE with CUDA graph and auto-tuning support.

    This wrapper pre-allocates all necessary buffers when `use_cuda_graph=True`,
    enabling CUDA graph capture and replay. It also supports auto-tuning via
    the `tactic` parameter or by calling inside `autotune()` context.

    Attributes:
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        hidden_size: Hidden dimension size.
        intermediate_size: Intermediate dimension size.
        use_cuda_graph: Whether to pre-allocate buffers for CUDA graph.
        max_num_tokens: Maximum tokens (only used with use_cuda_graph=True).

    Example (CUDA Graph):
        >>> moe = CuteDslMoEWrapper(
        ...     num_experts=256, top_k=8,
        ...     hidden_size=7168, intermediate_size=2048,
        ...     use_cuda_graph=True, max_num_tokens=4096,
        ... )
        >>> # Warmup
        >>> for _ in range(3):
        ...     output = moe.run(x, x_sf, topk_ids, topk_weights, w1, w1_sf, ...)
        >>> # Capture
        >>> g = torch.cuda.CUDAGraph()
        >>> with torch.cuda.graph(g):
        ...     output = moe.run(x, x_sf, topk_ids, topk_weights, w1, w1_sf, ...)
        >>> # Replay
        >>> g.replay()

    Example (Auto-tuning):
        >>> moe = CuteDslMoEWrapper(num_experts=256, top_k=8, ...)
        >>> # Run with auto-tuning
        >>> with autotune(True):
        ...     output = moe.run(x, x_sf, topk_ids, topk_weights, w1, w1_sf, ...)
    """

    @flashinfer_api
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        use_cuda_graph: bool = False,
        max_num_tokens: int = 4096,
        num_local_experts: Optional[int] = None,
        local_expert_offset: int = 0,
        tile_size: int = 128,
        sf_vec_size: int = 16,
        output_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """Initialize the MoE wrapper.

        Args:
            num_experts: Total number of experts.
            top_k: Number of experts per token.
            hidden_size: Hidden dimension size.
            intermediate_size: Intermediate size (after SwiGLU reduction).
            use_cuda_graph: Pre-allocate buffers for CUDA graph compatibility.
            max_num_tokens: Maximum tokens (only for use_cuda_graph=True).
            num_local_experts: Local experts for EP. Default: num_experts.
            local_expert_offset: Expert offset for EP. Default: 0.
            tile_size: Tile size for moe_sort. Default: 128.
            sf_vec_size: Scale factor vector size. Default: 16.
            output_dtype: Output data type. Default: torch.bfloat16.
            device: Device for buffer allocation. Default: "cuda".
        """
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_cuda_graph = use_cuda_graph
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts or num_experts
        self.local_expert_offset = local_expert_offset
        self.tile_size = tile_size
        self.sf_vec_size = sf_vec_size
        self.output_dtype = output_dtype
        self.device = device

        # Pre-allocated buffers
        self._moe_sort_buffers: Optional[Dict[str, torch.Tensor]] = None
        self._gemm1_output: Optional[torch.Tensor] = None
        self._gemm1_output_scale: Optional[torch.Tensor] = None
        self._moe_output: Optional[torch.Tensor] = None
        self._aux_stream: Optional[torch.cuda.Stream] = None
        self._main_event: Optional[torch.cuda.Event] = None
        self._memset_event: Optional[torch.cuda.Event] = None

        # Create auto-tuner runner
        self._runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=self._forward_with_tactic,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=local_expert_offset,
            use_fused_finalize=True,
            output_dtype=output_dtype,
        )

        if use_cuda_graph:
            self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Pre-allocate all buffers for CUDA graph compatibility."""
        max_num_permuted_tokens = get_max_num_permuted_tokens(
            self.max_num_tokens, self.top_k, self.num_local_experts, self.tile_size
        )

        # moe_sort buffers
        self._moe_sort_buffers = allocate_moe_sort_buffers(
            num_tokens=self.max_num_tokens,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            tile_tokens_dim=self.tile_size,
            device=self.device,
        )

        # GEMM1 output (FP4 quantized)
        self._gemm1_output = torch.empty(
            (max_num_permuted_tokens, self.intermediate_size // 2),
            dtype=torch.uint8,
            device=self.device,
        )

        # GEMM1 output scale
        scale_size = max_num_permuted_tokens * (
            self.intermediate_size // self.sf_vec_size
        )
        self._gemm1_output_scale = torch.empty(
            (scale_size,), dtype=torch.uint8, device=self.device
        )

        # Final output
        self._moe_output = torch.empty(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.output_dtype,
            device=self.device,
        )

        # CUDA resources
        self._aux_stream = torch.cuda.Stream(device=self.device)
        self._main_event = torch.cuda.Event()
        self._memset_event = torch.cuda.Event()

    def _forward_with_tactic(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w1_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_sf: torch.Tensor,
        w2_alpha: torch.Tensor,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
        tile_size: int = 128,
        gemm1_mma_tiler_mn: Tuple[int, int] = (128, 128),
        gemm1_cluster_shape_mn: Tuple[int, int] = (1, 1),
        gemm2_mma_tiler_mn: Tuple[int, int] = (128, 128),
        gemm2_cluster_shape_mn: Tuple[int, int] = (1, 1),
        output_dtype: torch.dtype = torch.bfloat16,
        use_fused_finalize: bool = True,
        moe_output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward implementation called by auto-tuner."""
        return _moe_core_impl(
            x=x,
            x_sf=x_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            tile_size=tile_size,
            gemm1_mma_tiler_mn=gemm1_mma_tiler_mn,
            gemm1_cluster_shape_mn=gemm1_cluster_shape_mn,
            gemm2_mma_tiler_mn=gemm2_mma_tiler_mn,
            gemm2_cluster_shape_mn=gemm2_cluster_shape_mn,
            moe_sort_buffers=self._moe_sort_buffers if self.use_cuda_graph else None,
            gemm1_out=self._gemm1_output if self.use_cuda_graph else None,
            gemm1_out_scale=self._gemm1_output_scale if self.use_cuda_graph else None,
            moe_output=moe_output
            if moe_output is not None
            else (self._moe_output if self.use_cuda_graph else None),
            aux_stream=self._aux_stream,
            main_event=self._main_event,
            memset_event=self._memset_event,
            output_dtype=output_dtype,
            use_async_memset=True,
        )

    @flashinfer_api
    def run(
        self,
        x: torch.Tensor,
        x_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w1_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_sf: torch.Tensor,
        w2_alpha: torch.Tensor,
        tactic: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """Run MoE computation.

        This method is CUDA graph safe when use_cuda_graph=True.
        Supports auto-tuning via `tactic` parameter or `autotune()` context.

        Args:
            x: Input tensor, NVFP4 quantized [num_tokens, hidden_size // 2].
            x_sf: Scale factors for x.
            token_selected_experts: Expert assignments [num_tokens, top_k].
            token_final_scales: Routing weights [num_tokens, top_k].
            w1_weight: GEMM1 weights (gate + up fused).
            w1_weight_sf: Scale factors for w1_weight.
            w1_alpha: Per-expert global scale for GEMM1.
            fc2_input_scale: Global scale for GEMM2 input quantization.
            w2_weight: GEMM2 weights (down projection).
            w2_weight_sf: Scale factors for w2_weight.
            w2_alpha: Per-expert global scale for GEMM2.
            tactic: Tactic tuple or None for auto-selection.

        Returns:
            Output tensor [num_tokens, hidden_size].
        """
        num_tokens = token_selected_experts.size(0)

        if self.use_cuda_graph and num_tokens > self.max_num_tokens:
            raise ValueError(
                f"num_tokens ({num_tokens}) exceeds max_num_tokens ({self.max_num_tokens})"
            )

        # Allocate output buffer if not using pre-allocated one
        if self.use_cuda_graph:
            moe_output = self._moe_output
        else:
            moe_output = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=self.output_dtype,
                device=x.device,
            )

        # Use auto-tuner for tactic selection
        tuner = AutoTuner.get()

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

        if tactic is not None:
            # Use provided tactic
            return self._runner(inputs, tactic=tactic)

        # Let tuner choose tactic
        _, best_tactic = tuner.choose_one(
            "CuteDslMoEWrapper::run",
            [self._runner],
            CuteDslFusedMoENvfp4Runner.tuning_config,
            inputs,
        )

        return self._runner(inputs, tactic=best_tactic)

    def get_valid_tactics(self) -> list:
        """Return list of valid tactics for this MoE configuration."""
        return ALL_MOE_TACTICS


# =============================================================================
# Functional API (Simple Function Call)
# =============================================================================


def _cute_dsl_fused_moe_nvfp4_impl(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int = 0,
    tile_size: int = 128,
    gemm1_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm1_cluster_shape_mn: Tuple[int, int] = (1, 1),
    gemm2_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm2_cluster_shape_mn: Tuple[int, int] = (1, 1),
    output_dtype: torch.dtype = torch.bfloat16,
    use_fused_finalize: bool = True,
    moe_output: Optional[torch.Tensor] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Internal implementation called by auto-tuner for functional API."""
    return _moe_core_impl(
        x=x,
        x_sf=x_sf,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        w1_weight=w1_weight,
        w1_weight_sf=w1_weight_sf,
        w1_alpha=w1_alpha,
        fc2_input_scale=fc2_input_scale,
        w2_weight=w2_weight,
        w2_weight_sf=w2_weight_sf,
        w2_alpha=w2_alpha,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        tile_size=tile_size,
        gemm1_mma_tiler_mn=gemm1_mma_tiler_mn,
        gemm1_cluster_shape_mn=gemm1_cluster_shape_mn,
        gemm2_mma_tiler_mn=gemm2_mma_tiler_mn,
        gemm2_cluster_shape_mn=gemm2_cluster_shape_mn,
        moe_output=moe_output,
        aux_stream=aux_stream,
        output_dtype=output_dtype,
        use_async_memset=True,
    )


@flashinfer_api
def cute_dsl_fused_moe_nvfp4(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    output_dtype: torch.dtype = torch.bfloat16,
    use_fused_finalize: bool = True,
    moe_output: Optional[torch.Tensor] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
) -> torch.Tensor:
    """Run fused MoE computation using CuteDSL NVFP4 kernels.

    This is the simple functional API. For CUDA graph support, use
    `CuteDslMoEWrapper` instead.

    Auto-tuning is controlled via the `autotune()` context manager:

        >>> with autotune(True):
        ...     output = cute_dsl_fused_moe_nvfp4(...)

    Args:
        x: Input tensor, NVFP4 quantized [num_tokens, hidden_size // 2].
        x_sf: Scale factors for x.
        token_selected_experts: Expert assignments [num_tokens, top_k].
        token_final_scales: Routing weights [num_tokens, top_k].
        w1_weight: GEMM1 weights (gate + up fused).
        w1_weight_sf: Scale factors for w1_weight.
        w1_alpha: Per-expert global scale for GEMM1.
        fc2_input_scale: Global scale for GEMM2 input quantization.
        w2_weight: GEMM2 weights (down projection).
        w2_weight_sf: Scale factors for w2_weight.
        w2_alpha: Per-expert global scale for GEMM2.
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        num_local_experts: Local experts for EP. Default: num_experts.
        local_expert_offset: Expert offset for EP. Default: 0.
        output_dtype: Output data type. Default: torch.bfloat16.
        use_fused_finalize: Use fused finalize. Default: True.
        moe_output: Pre-allocated output buffer.
        aux_stream: Auxiliary CUDA stream.

    Returns:
        Output tensor [num_tokens, hidden_size].
    """
    if num_local_experts is None:
        num_local_experts = num_experts

    num_tokens = token_selected_experts.size(0)
    hidden_size = w2_weight.size(1)

    if moe_output is None:
        moe_output = torch.empty(
            (num_tokens, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )

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

    _, best_tactic = tuner.choose_one(
        "CuteDslFusedMoE::run_moe_nvfp4",
        [runner],
        CuteDslFusedMoENvfp4Runner.tuning_config,
        inputs,
        aux_stream=aux_stream,
    )

    return runner(inputs, tactic=best_tactic, aux_stream=aux_stream)


__all__ = [
    "cute_dsl_fused_moe_nvfp4",
    "CuteDslMoEWrapper",
]
