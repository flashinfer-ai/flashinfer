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
   Class-based API that holds persistent CUDA stream/event resources for
   async-memset overlap and CUDA graph compatibility.
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

import weakref

import torch

from ...api_logging import flashinfer_api
from ...trace.templates.moe import (
    cute_dsl_fused_moe_nvfp4_trace,
    cute_dsl_moe_wrapper_run_trace,
)
from ...tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from ...autotuner import AutoTuner
from ...cute_dsl.utils import convert_sf_to_mma_layout
from ...quantization.kernels.nvfp4_quantize import (
    SF_LAYOUT_128x4,
    nvfp4_quantize_per_token_cute_dsl,
)
from ...utils import supported_compute_capability
from .moe_utils import (
    moe_output_memset_inplace,
    moe_sort,
    normalize_cute_dsl_moe_activation_type,
)
from .blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
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


def _intermediate_c_dtype(output_dtype: torch.dtype) -> str:
    if output_dtype == torch.float16:
        return "float16"
    if output_dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(
        "CuTe-DSL MoE per-token FC2 input quantization supports only "
        f"torch.float16 and torch.bfloat16 intermediate dtypes, got {output_dtype}."
    )


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
    per_token_scale: Optional[torch.Tensor] = None,
    # Stream resources
    aux_stream: Optional[torch.cuda.Stream] = None,
    main_event: Optional[torch.cuda.Event] = None,
    memset_event: Optional[torch.cuda.Event] = None,
    # Options
    output_dtype: torch.dtype = torch.bfloat16,
    use_async_memset: bool = True,
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
) -> torch.Tensor:
    """Core MoE implementation shared by functional and wrapper APIs.

    This function handles:
    1. moe_sort: Token routing computation
    2. GEMM1 + SwiGLU: First projection with activation
    3. Async output zero: Zero output buffer (overlapped with GEMM1)
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
        per_token_scale: Optional per-token input row scale for GEMM1.
        aux_stream: Auxiliary CUDA stream for async memset.
        main_event: CUDA event for main stream.
        memset_event: CUDA event for memset completion.
        output_dtype: Output data type.
        use_async_memset: Use async memset on aux stream.
        activation_type: Activation type to apply after GEMM1. Use
            ActivationType.Swiglu for gated mode and ActivationType.Relu2 for
            non-gated mode; swiglu_oai is represented as Swiglu with
            non-default swiglu_alpha/beta/limit.
        swiglu_alpha: SwiGLU sigmoid multiplier.
        swiglu_beta: SwiGLU up-projection bias.
        swiglu_limit: SwiGLU clamp limit.

    Returns:
        Output tensor [num_tokens, hidden_size].
    """
    activation_type, gated = normalize_cute_dsl_moe_activation_type(activation_type)

    num_tokens = token_selected_experts.size(0)
    hidden_size = w2_weight.size(1)
    use_per_token_activation = per_token_scale is not None

    # Allocate output if not provided.  The caller (wrapper or functional
    # API) should pass a [:num_tokens] slice of the pre-allocated buffer
    # when using CUDA graphs.  The buffer is zeroed in Step 3 below.
    if moe_output is None:
        moe_output = torch.empty(
            (num_tokens, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )
    else:
        assert moe_output.size(0) == num_tokens, (
            f"moe_output must be sliced to num_tokens rows before calling "
            f"_moe_core_impl (got {moe_output.size(0)}, expected {num_tokens})"
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

    # Step 2: GEMM1 + activation
    output_kwargs: Dict[str, Any] = (
        {
            "out_scale": None,
            "global_scale": None,
            "a_per_token_scale": per_token_scale,
            "c_dtype": _intermediate_c_dtype(output_dtype),
        }
        if use_per_token_activation
        else {
            "out_scale": gemm1_out_scale,
            "global_scale": fc2_input_scale,
            "a_per_token_scale": None,
            "c_dtype": "float4_e2m1fn",
        }
    )
    intermediate_per_token_scale = None
    intermediate, intermediate_sf = (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4(
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
            **output_kwargs,
            topk=top_k,
            mma_tiler_mn=gemm1_mma_tiler_mn,
            cluster_shape_mn=gemm1_cluster_shape_mn,
            enable_pdl=enable_pdl,
            activation_type=activation_type.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            gated=gated,
        )
    )
    if use_per_token_activation:
        intermediate, intermediate_sf, intermediate_per_token_scale = (
            nvfp4_quantize_per_token_cute_dsl(
                intermediate,
                fc2_input_scale,
                sf_layout=SF_LAYOUT_128x4,
                enable_pdl=enable_pdl,
            )
        )
        intermediate_sf = convert_sf_to_mma_layout(
            intermediate_sf,
            m=intermediate.shape[0],
            k=intermediate.shape[1] * 2,
            num_groups=1,
            sf_vec_size=16,
        )

    # Step 3: Zero the active output slice before GEMM2 finalize.
    # Finalize uses atomic scatter-add into `moe_output`, so it must start
    # from zero each call. We zero only the active slice, not the full
    # preallocated buffer.
    #
    # `moe_output_memset_inplace` mirrors TRT-LLM's `moe_output_memset_inplace`
    # Path A (dense cudaMemsetAsync). TRT-LLM's Path B (sparse moeOutputMemset
    # kernel for the internal-alltoall case) is not exposed here — current
    # callers of this API handle all-to-all outside this function.
    #
    # The wrapper issues cudaMemsetAsync on the current PyTorch CUDA stream,
    # so the `with torch.cuda.stream(aux_stream):` context below correctly
    # places the memset on the aux stream for overlap with the main-stream
    # GEMM1.
    if use_async_memset:
        with torch.cuda.stream(aux_stream):
            main_event.wait()
            moe_output_memset_inplace(moe_output)
            memset_event.record()
        memset_event.wait()
    else:
        moe_output_memset_inplace(moe_output)

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
        a_per_token_scale=intermediate_per_token_scale,
        mma_tiler_mn=gemm2_mma_tiler_mn,
        cluster_shape_mn=gemm2_cluster_shape_mn,
        enable_pdl=enable_pdl,
    )

    return moe_output[:num_tokens]


# =============================================================================
# Wrapper API (Class-based, CUDA Graph Compatible)
# =============================================================================


class CuteDslMoEWrapper:
    """Wrapper class for CuteDSL MoE with CUDA graph and auto-tuning support.

    With `use_cuda_graph=True`, the wrapper creates persistent CUDA stream
    and event resources outside graph capture, enabling async-memset / GEMM1
    overlap during capture and replay. Auto-tuning is supported via the `tactic`
    parameter or `autotune()` context.

    Supported architectures: SM100, SM103.

    Attributes:
        num_experts: Total number of experts.
        top_k: Number of experts per token.
        hidden_size: Hidden dimension size.
        intermediate_size: Intermediate dimension size.
        use_cuda_graph: Whether the wrapper holds persistent stream/event
            resources for CUDA graph capture.
        max_num_tokens: Deprecated; accepted for backwards compatibility
            but ignored.

    Example (CUDA Graph):
        >>> moe = CuteDslMoEWrapper(
        ...     num_experts=256, top_k=8,
        ...     hidden_size=7168, intermediate_size=2048,
        ...     use_cuda_graph=True,
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

    @supported_compute_capability([100, 103])
    @flashinfer_api
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        use_cuda_graph: bool = False,
        max_num_tokens: Optional[int] = None,
        num_local_experts: Optional[int] = None,
        local_expert_offset: int = 0,
        tile_size: int = 128,
        sf_vec_size: int = 16,
        output_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        enable_pdl: bool = True,
        activation_type: int = ActivationType.Swiglu.value,
        swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
        swiglu_beta: float = DEFAULT_SWIGLU_BETA,
        swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    ):
        r"""Configure the CuTe-DSL NVFP4 fused-MoE wrapper.

        Parameters
        ----------
        num_experts : int
            Total number of experts.
        top_k : int
            Number of experts routed to per token.
        hidden_size : int
            Hidden dimension size.
        intermediate_size : int
            Intermediate dimension size (after SwiGLU reduction).
        use_cuda_graph : bool
            Create persistent CUDA stream/events for async-memset overlap.
            Required for CUDA graph capture, since streams and events must be
            created outside graph capture.  Defaults to ``False``.
        max_num_tokens : Optional[int]
            Deprecated; accepted for backwards compatibility but ignored.
        num_local_experts : Optional[int]
            Local experts for expert parallelism.  Defaults to
            ``num_experts``.
        local_expert_offset : int
            Offset of local experts in the global expert space.  Defaults
            to ``0``.
        tile_size : int
            Tile size for ``moe_sort``.  Defaults to ``128``.
        sf_vec_size : int
            Scale-factor vector size.  Defaults to ``16``.
        output_dtype : torch.dtype
            Output dtype.  Defaults to ``torch.bfloat16``.
        device : str
            Device on which to allocate buffers.  Defaults to ``"cuda"``.
        enable_pdl : bool
            Enable Programmatic Dependent Launch.  Defaults to ``True``.
        activation_type : int
            FC1 activation type. Use ``ActivationType.Swiglu`` for gated
            SwiGLU and ``ActivationType.Relu2`` for non-gated ReLU^2.
        swiglu_alpha, swiglu_beta, swiglu_limit : float
            SwiGLU parameters. ``swiglu_oai`` is represented as
            ``ActivationType.Swiglu`` with non-default values.
        """
        activation_type, gated = normalize_cute_dsl_moe_activation_type(activation_type)

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_cuda_graph = use_cuda_graph
        self.num_local_experts = num_local_experts or num_experts
        self.local_expert_offset = local_expert_offset
        self.tile_size = tile_size
        self.sf_vec_size = sf_vec_size
        self.output_dtype = output_dtype
        self.device = device
        self.enable_pdl = enable_pdl
        self.activation_type = activation_type
        self.gated = gated
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit

        # Persistent CUDA resources for async-memset / GEMM1 overlap. These
        # are created outside graph capture (so they can be reused inside it)
        # when ``use_cuda_graph=True``. When None, ``_moe_core_impl`` falls
        # back to module-level resources via ``_get_cuda_graph_resources``.
        self._aux_stream: Optional[torch.cuda.Stream] = None
        self._main_event: Optional[torch.cuda.Event] = None
        self._memset_event: Optional[torch.cuda.Event] = None

        wrapper_ref = weakref.ref(self)

        def _forward_with_tactic_weak(*args, **kwargs):
            wrapper = wrapper_ref()
            if wrapper is None:
                raise RuntimeError(
                    "CuteDslMoEWrapper was destroyed before runner invocation"
                )
            return wrapper._forward_with_tactic(*args, **kwargs)

        # Create auto-tuner runner. Use a weak trampoline instead of a bound
        # method so the runner cannot keep CUDA graph resources alive after the
        # wrapper drops out of scope.
        self._runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=_forward_with_tactic_weak,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=local_expert_offset,
            use_fused_finalize=True,
            output_dtype=output_dtype,
            enable_pdl=enable_pdl,
            activation_type=activation_type.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            use_per_token_activation=False,
        )
        self._per_token_runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=_forward_with_tactic_weak,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=local_expert_offset,
            use_fused_finalize=True,
            output_dtype=output_dtype,
            enable_pdl=enable_pdl,
            activation_type=activation_type.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            use_per_token_activation=True,
        )

        if use_cuda_graph:
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
        per_token_scale: Optional[torch.Tensor] = None,
        enable_pdl: bool = True,
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
            moe_sort_buffers=None,
            gemm1_out=None,
            gemm1_out_scale=None,
            moe_output=moe_output,
            per_token_scale=per_token_scale,
            aux_stream=self._aux_stream,
            main_event=self._main_event,
            memset_event=self._memset_event,
            output_dtype=output_dtype,
            use_async_memset=True,
            enable_pdl=enable_pdl,
            activation_type=self.activation_type.value,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
        )

    @flashinfer_api(trace=cute_dsl_moe_wrapper_run_trace)
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
        *,
        per_token_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""Run the CuTe-DSL NVFP4 fused-MoE forward pass.

        CUDA-graph safe when the wrapper was constructed with
        ``use_cuda_graph=True``.  Supports auto-tuning via the ``tactic``
        argument or the surrounding :func:`autotune` context manager.

        Parameters
        ----------
        x : torch.Tensor
            NVFP4-quantized input of shape ``[num_tokens, hidden_size // 2]``.
        x_sf : torch.Tensor
            Scale factors for ``x``.
        token_selected_experts : torch.Tensor
            Expert assignments of shape ``[num_tokens, top_k]``.
        token_final_scales : torch.Tensor
            Routing weights of shape ``[num_tokens, top_k]``.
        w1_weight : torch.Tensor
            GEMM1 weights (gate + up fused).
        w1_weight_sf : torch.Tensor
            Scale factors for ``w1_weight``.
        w1_alpha : torch.Tensor
            Per-expert global scale for GEMM1.
        fc2_input_scale : torch.Tensor
            Global scale for GEMM2 input quantization.
        w2_weight : torch.Tensor
            GEMM2 weights (down projection).
        w2_weight_sf : torch.Tensor
            Scale factors for ``w2_weight``.
        w2_alpha : torch.Tensor
            Per-expert global scale for GEMM2.
        tactic : Optional[Tuple]
            Tactic tuple, or ``None`` for auto-selection via the runtime
            tuner.
        per_token_scale : Optional[torch.Tensor]
            Per-token input row scale for GEMM1. Passing this enables the
            per-token activation path.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[num_tokens, hidden_size]``.
        """
        num_tokens = token_selected_experts.size(0)
        use_per_token_activation = per_token_scale is not None
        runner = self._per_token_runner if use_per_token_activation else self._runner

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
        ]
        if use_per_token_activation:
            inputs.append(per_token_scale)
        inputs.append(moe_output)

        if tactic is not None:
            # Use provided tactic
            return runner(inputs, tactic=tactic)

        # Let tuner choose tactic
        _, best_tactic = tuner.choose_one(
            f"CuteDslMoEWrapper::run::{self.activation_type.name}",
            [runner],
            runner.tuning_config,
            inputs,
        )

        return runner(inputs, tactic=best_tactic)

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
    per_token_scale: Optional[torch.Tensor] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
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
        per_token_scale=per_token_scale,
        aux_stream=aux_stream,
        output_dtype=output_dtype,
        use_async_memset=True,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )


@supported_compute_capability([100, 103])
@flashinfer_api(trace=cute_dsl_fused_moe_nvfp4_trace)
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
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    *,
    per_token_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Run a fused MoE forward pass using the CuTe-DSL NVFP4 kernels.

    Supported architectures: SM100, SM103.  This is the simple functional
    API; for CUDA-graph support use :class:`CuteDslMoEWrapper` instead.

    Auto-tuning is controlled by the :func:`autotune` context manager::

        with autotune(True):
            output = cute_dsl_fused_moe_nvfp4(...)

    Parameters
    ----------
    x : torch.Tensor
        NVFP4-quantized input of shape ``[num_tokens, hidden_size // 2]``.
    x_sf : torch.Tensor
        Scale factors for ``x``.
    token_selected_experts : torch.Tensor
        Expert assignments of shape ``[num_tokens, top_k]``.
    token_final_scales : torch.Tensor
        Routing weights of shape ``[num_tokens, top_k]``.
    w1_weight : torch.Tensor
        GEMM1 weights (gate + up fused).
    w1_weight_sf : torch.Tensor
        Scale factors for ``w1_weight``.
    w1_alpha : torch.Tensor
        Per-expert global scale for GEMM1.
    fc2_input_scale : torch.Tensor
        Global scale for GEMM2 input quantization.
    w2_weight : torch.Tensor
        GEMM2 weights (down projection).
    w2_weight_sf : torch.Tensor
        Scale factors for ``w2_weight``.
    w2_alpha : torch.Tensor
        Per-expert global scale for GEMM2.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts routed to per token.
    num_local_experts : Optional[int]
        Local experts for expert parallelism.  Defaults to ``num_experts``.
    local_expert_offset : int
        Offset of local experts in the global expert space.  Defaults to ``0``.
    output_dtype : torch.dtype
        Output dtype.  Defaults to ``torch.bfloat16``.
    use_fused_finalize : bool
        Whether to use the fused finalize path.  Defaults to ``True``.
    moe_output : Optional[torch.Tensor]
        Pre-allocated output buffer.  Allocated internally if ``None``.
    aux_stream : Optional[torch.cuda.Stream]
        Optional auxiliary CUDA stream used to overlap setup work with the
        main computation.
    enable_pdl : bool
        Enable Programmatic Dependent Launch.  Defaults to ``True``.
    activation_type : int
        FC1 activation type. Use ``ActivationType.Swiglu`` for gated SwiGLU
        and ``ActivationType.Relu2`` for non-gated ReLU^2. ``swiglu_oai`` is
        represented as ``ActivationType.Swiglu`` with non-default
        ``swiglu_alpha/beta/limit``.
    swiglu_alpha, swiglu_beta, swiglu_limit : float
        SwiGLU parameters.
    per_token_scale : Optional[torch.Tensor]
        Per-token input row scale for GEMM1. Passing this enables the
        per-token activation path.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``[num_tokens, hidden_size]``.
    """
    activation_type, _ = normalize_cute_dsl_moe_activation_type(activation_type)

    if num_local_experts is None:
        num_local_experts = num_experts
    use_per_token_activation = per_token_scale is not None

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
        enable_pdl=enable_pdl,
        activation_type=activation_type.value,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        use_per_token_activation=use_per_token_activation,
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
    ]
    if use_per_token_activation:
        inputs.append(per_token_scale)
    inputs.append(moe_output)

    _, best_tactic = tuner.choose_one(
        f"CuteDslFusedMoE::run_moe_nvfp4::{activation_type.name}",
        [runner],
        runner.tuning_config,
        inputs,
        aux_stream=aux_stream,
    )

    return runner(
        inputs,
        tactic=best_tactic,
        aux_stream=aux_stream,
    )


__all__ = [
    "cute_dsl_fused_moe_nvfp4",
    "CuteDslMoEWrapper",
]
