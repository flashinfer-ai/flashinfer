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
CuteDSL-based Fused MoE API for NVFP4 on Blackwell and Rubin GPUs.

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

Both APIs share the same mode-specific runners and support auto-tuning.

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
from ...cute_dsl.utils import require_cute_dsl_arch as _require_cute_dsl_arch_for
from ...quantization.kernels.nvfp4_quantize import (
    SF_LAYOUT_128x4,
    nvfp4_quantize_per_token_cute_dsl,
)
from ...utils import supported_compute_capability
from .moe_utils import (
    moe_output_memset_inplace,
    moe_sort,
    moe_unpermute,
    normalize_cute_dsl_moe_activation_type,
    validate_cute_dsl_moe_situ_config,
)
from .blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
    blockscaled_contiguous_gather_grouped_gemm_act_fusion_nvfp4,
)
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import (
    blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4,
)
from .tuner import (
    CuteDslFusedMoENvfp4Runner,
    CuteDslFusedMoEW4A16Runner,
    W4A16_MOE_TACTICS,
    _get_arch_tactics,
)

# =============================================================================
# Module-level Resources for CUDA Graph Compatibility
# =============================================================================

_cuda_graph_resources: Dict[str, Any] = {}
_PER_TOKEN_AUX_AMAX_MIN_TOKENS = 4


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
    x_sf: Optional[torch.Tensor],
    # Routing
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    # GEMM1 weights
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    # GEMM2 intermediate scale
    fc2_input_scale: Optional[torch.Tensor],
    # GEMM2 weights
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    # MoE config
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int = 0,
    # Tactic parameters (Blackwell)
    tile_size: int = 128,
    gemm1_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm1_cluster_shape_mn: Tuple[int, int] = (1, 1),
    gemm2_mma_tiler_mn: Tuple[int, int] = (128, 128),
    gemm2_cluster_shape_mn: Tuple[int, int] = (1, 1),
    # Tactic parameters (Rubin — when set, use SM107 kernel)
    gemm1_mma_tiler: Optional[Tuple[int, int, int]] = None,
    gemm1_mma_inst_shape: Optional[Tuple[int, int, int]] = None,
    gemm2_mma_tiler: Optional[Tuple[int, int, int]] = None,
    gemm2_mma_inst_shape: Optional[Tuple[int, int, int]] = None,
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
    use_fused_finalize: bool = True,
    enable_pdl: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
) -> torch.Tensor:
    """Core MoE implementation shared by functional and wrapper APIs.

    This function handles:
    1. moe_sort: Token routing computation
    2. GEMM1 + activation
    3. GEMM2 with optional atomic finalize
    4. Routing-weight reduction in deterministic mode

    Args:
        x: Input tensor, NVFP4 quantized.
        x_sf: Scale factors for x.
        token_selected_experts: Expert assignments [num_tokens, top_k].
        token_final_scales: Routing weights [num_tokens, top_k].
        w1_weight: GEMM1 weights (gate + up fused for gated activations, or a
            single projection for non-gated activations).
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
        use_fused_finalize: Use atomic fused finalize; otherwise use the
            deterministic two-stage finalize.
        activation_type: Activation type to apply after GEMM1. Use
            ActivationType.Swiglu for gated SwiGLU/OAI/SiTU,
            ActivationType.GegluTanh for tanh-approximate GeGLU, and
            ActivationType.Relu2 for non-gated mode. Setting situ_beta selects
            SiTU; swiglu_oai is represented as Swiglu with non-default
            swiglu_alpha/beta/limit.
        swiglu_alpha: SwiGLU sigmoid multiplier.
        swiglu_beta: SwiGLU up-projection bias.
        swiglu_limit: SwiGLU clamp limit.
        situ_beta: When set with ActivationType.Swiglu, use the SiTU gate.
        situ_linear_beta: Optional SiTU tanh clamp for the up branch.

    Returns:
        Output tensor [num_tokens, hidden_size].
    """
    activation, gated = normalize_cute_dsl_moe_activation_type(activation_type)
    if x.dtype != torch.uint8:
        raise TypeError(
            f"quant_mode='w4a4' requires x.dtype=torch.uint8, got {x.dtype}"
        )
    if x_sf is None:
        raise ValueError("x_sf is required when quant_mode='w4a4'")
    if fc2_input_scale is None:
        raise ValueError("fc2_input_scale is required when quant_mode='w4a4'")
    validate_cute_dsl_moe_situ_config(activation, situ_beta, situ_linear_beta)

    num_tokens = token_selected_experts.size(0)
    hidden_size = w2_weight.size(1)
    use_per_token_activation = per_token_scale is not None

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

    # Fused finalize overlaps output zeroing with GEMM1.
    if use_async_memset and use_fused_finalize:
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

    # For Rubin, round num_non_exiting_tiles to the next EVEN number to
    # prevent a cluster-synchronization deadlock. With cluster_shape_m=2,
    # two CTAs get consecutive tile indices; if the count is odd, one CTA
    # enters the cluster barrier while the other skips it.
    is_rubin = gemm1_mma_tiler is not None and gemm1_mma_inst_shape is not None
    if is_rubin:
        kernel_num_non_exiting_tiles = ((num_non_exiting_tiles + 1) // 2) * 2
    else:
        kernel_num_non_exiting_tiles = num_non_exiting_tiles

    # Record event for async memset synchronization
    if use_async_memset and use_fused_finalize:
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
    # The aux handoff has fixed producer/consumer overhead at very small M.
    # num_tokens is host-static, so CUDA graph capture specializes one path
    # without a device-side branch or synchronization.
    use_intermediate_amax = (
        use_per_token_activation and num_tokens >= _PER_TOKEN_AUX_AMAX_MIN_TOKENS
    )
    intermediate_per_token_scale = None
    intermediate_amax = None
    if use_intermediate_amax:
        intermediate_size = w1_weight.shape[1] // (2 if gated else 1)
        output_tile_n = gemm1_mma_tiler_mn[1] // (2 if gated else 1)
        num_output_tiles = (intermediate_size + output_tile_n - 1) // output_tile_n
        num_permuted_rows = permuted_idx_to_expanded_idx.shape[0]
        intermediate_amax = torch.empty(
            (
                (num_permuted_rows + 7) // 8,
                num_output_tiles,
                8,
            ),
            dtype=output_dtype,
            device=x.device,
        )
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
            num_non_exiting_tiles=kernel_num_non_exiting_tiles,
            out=gemm1_out,
            out_amax=intermediate_amax,
            **output_kwargs,
            topk=top_k,
            mma_tiler_mn=gemm1_mma_tiler_mn,
            cluster_shape_mn=gemm1_cluster_shape_mn,
            mma_tiler=gemm1_mma_tiler,
            mma_inst_shape=gemm1_mma_inst_shape,
            enable_pdl=enable_pdl,
            activation_type=activation.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
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
                input_amax=intermediate_amax,
                input_amax_valid_rows=(
                    total_num_padded_tokens if intermediate_amax is not None else None
                ),
            )
        )
        intermediate_sf = convert_sf_to_mma_layout(
            intermediate_sf,
            m=intermediate.shape[0],
            k=intermediate.shape[1] * 2,
            num_groups=1,
            sf_vec_size=16,
        )

    # Atomic finalize requires a zeroed token output. Deterministic finalize
    # writes each route to a unique expanded row.
    if use_fused_finalize:
        if use_async_memset:
            with torch.cuda.stream(aux_stream):
                main_event.wait()
                moe_output_memset_inplace(moe_output)
                memset_event.record()
            memset_event.wait()
        else:
            moe_output_memset_inplace(moe_output)
        gemm2_output = moe_output
    else:
        gemm2_output = torch.empty(
            (num_tokens * top_k, hidden_size),
            dtype=output_dtype,
            device=x.device,
        )

    # Step 3: GEMM2 with optional atomic finalize
    blockscaled_contiguous_grouped_gemm_finalize_fusion_nvfp4(
        a=intermediate,
        b=w2_weight,
        a_scale=intermediate_sf,
        b_scale=w2_weight_sf,
        alpha=w2_alpha,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        num_non_exiting_tiles=kernel_num_non_exiting_tiles,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        token_final_scales=token_final_scales,
        out=gemm2_output,
        a_per_token_scale=intermediate_per_token_scale,
        mma_tiler_mn=gemm2_mma_tiler_mn,
        cluster_shape_mn=gemm2_cluster_shape_mn,
        mma_tiler=gemm2_mma_tiler,
        mma_inst_shape=gemm2_mma_inst_shape,
        enable_pdl=enable_pdl,
        use_fused_finalize=use_fused_finalize,
    )

    # Step 4: Deterministic routing-weight reduction
    if not use_fused_finalize:
        moe_unpermute(
            permuted_input=gemm2_output,
            output=moe_output,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            topk_scales=token_final_scales,
            num_tokens=num_tokens,
            top_k=top_k,
            input_is_expanded=True,
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
        use_fused_finalize: Use atomic fused finalize; otherwise use the
            deterministic two-stage finalize.
        quant_mode: Selected W4A4 or W4A16 compute mode.
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

    @supported_compute_capability([100, 103, 107])
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
        situ_beta: Optional[float] = None,
        situ_linear_beta: Optional[float] = None,
        use_fused_finalize: bool = True,
        quant_mode: str = "w4a4",
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
            Intermediate dimension size after the fused activation.
        use_cuda_graph : bool
            Create persistent CUDA stream/events for W4A4 async-memset
            overlap. W4A16 is CUDA-graph safe without those resources.
            Defaults to ``False``.
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
            SwiGLU/SiTU, ``ActivationType.GegluTanh`` for tanh-approximate
            GeGLU, and ``ActivationType.Relu2`` for non-gated ReLU^2. Setting
            ``situ_beta`` selects SiTU.
        swiglu_alpha, swiglu_beta, swiglu_limit : float
            SwiGLU parameters. ``swiglu_oai`` is represented as
            ``ActivationType.Swiglu`` with non-default values.
        situ_beta : Optional[float]
            When set with ``ActivationType.Swiglu``, use the SiTU gate
            ``beta * tanh(gate / beta) * sigmoid(gate)``.
        situ_linear_beta : Optional[float]
            Optional SiTU tanh clamp for the up branch.
        use_fused_finalize : bool
            Use atomic fused finalize; otherwise use the deterministic
            two-stage finalize. Defaults to ``True``.
        quant_mode : str
            Compute mode: ``"w4a4"`` / ``"nvfp4"`` or ``"w4a16"``.
            Defaults to ``"w4a4"``.
        """
        activation, gated = normalize_cute_dsl_moe_activation_type(activation_type)
        quant_mode = quant_mode.lower()
        validate_cute_dsl_moe_situ_config(activation, situ_beta, situ_linear_beta)
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
        self.activation_type: ActivationType = activation
        self.gated = gated
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.use_fused_finalize = use_fused_finalize
        self.quant_mode = quant_mode

        # Persistent CUDA resources for async-memset / GEMM1 overlap. These
        # are created outside graph capture (so they can be reused inside it)
        # when ``use_cuda_graph=True``. When None, ``_moe_core_impl`` falls
        # back to module-level resources via ``_get_cuda_graph_resources``.
        self._aux_stream: Optional[torch.cuda.Stream] = None
        self._main_event: Optional[torch.cuda.Event] = None
        self._memset_event: Optional[torch.cuda.Event] = None

        self._runner: Optional[CuteDslFusedMoENvfp4Runner] = None
        self._per_token_runner: Optional[CuteDslFusedMoENvfp4Runner] = None
        self._w4a16_runner: Optional[CuteDslFusedMoEW4A16Runner] = None
        if quant_mode in ("nvfp4", "w4a4"):
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
                use_fused_finalize=use_fused_finalize,
                output_dtype=output_dtype,
                enable_pdl=enable_pdl,
                activation_type=activation.value,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                use_per_token_activation=False,
            )
            self._per_token_runner = CuteDslFusedMoENvfp4Runner(
                forward_impl=_forward_with_tactic_weak,
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=self.num_local_experts,
                local_expert_offset=local_expert_offset,
                use_fused_finalize=use_fused_finalize,
                output_dtype=output_dtype,
                enable_pdl=enable_pdl,
                activation_type=activation.value,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                use_per_token_activation=True,
            )

            if use_cuda_graph:
                self._aux_stream = torch.cuda.Stream(device=self.device)
                self._main_event = torch.cuda.Event()
                self._memset_event = torch.cuda.Event()
        elif quant_mode == "w4a16":
            self._w4a16_runner = CuteDslFusedMoEW4A16Runner(
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=self.num_local_experts,
                local_expert_offset=local_expert_offset,
                use_fused_finalize=use_fused_finalize,
                output_dtype=output_dtype,
                enable_pdl=enable_pdl,
                activation_type=activation.value,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
            )
        else:
            raise ValueError(
                f"quant_mode must be 'nvfp4'/'w4a4' or 'w4a16' (got {quant_mode!r})."
            )

    def _forward_with_tactic(
        self,
        x: torch.Tensor,
        x_sf: Optional[torch.Tensor],
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w1_alpha: torch.Tensor,
        fc2_input_scale: Optional[torch.Tensor],
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
        gemm1_mma_tiler=None,
        gemm1_mma_inst_shape=None,
        gemm2_mma_tiler=None,
        gemm2_mma_inst_shape=None,
        output_dtype: torch.dtype = torch.bfloat16,
        use_fused_finalize: bool = True,
        moe_output: Optional[torch.Tensor] = None,
        per_token_scale: Optional[torch.Tensor] = None,
        enable_pdl: bool = True,
        use_async_memset: bool = True,
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
            gemm1_mma_tiler=gemm1_mma_tiler,
            gemm1_mma_inst_shape=gemm1_mma_inst_shape,
            gemm2_mma_tiler=gemm2_mma_tiler,
            gemm2_mma_inst_shape=gemm2_mma_inst_shape,
            moe_sort_buffers=None,
            gemm1_out=None,
            gemm1_out_scale=None,
            moe_output=moe_output,
            per_token_scale=per_token_scale,
            aux_stream=self._aux_stream,
            main_event=self._main_event,
            memset_event=self._memset_event,
            output_dtype=output_dtype,
            use_async_memset=use_async_memset,
            use_fused_finalize=use_fused_finalize,
            enable_pdl=enable_pdl,
            activation_type=self.activation_type.value,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )

    @flashinfer_api(trace=cute_dsl_moe_wrapper_run_trace)
    def run(
        self,
        x: torch.Tensor,
        x_sf: Optional[torch.Tensor],
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w1_alpha: torch.Tensor,
        fc2_input_scale: Optional[torch.Tensor],
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
            Packed NVFP4 input for ``quant_mode="w4a4"`` or BF16 input for
            ``quant_mode="w4a16"``.
        x_sf : Optional[torch.Tensor]
            Scale factors for ``quant_mode="w4a4"``; must be ``None`` for
            ``quant_mode="w4a16"``.
        token_selected_experts : torch.Tensor
            Expert assignments of shape ``[num_tokens, top_k]``.
        token_final_scales : torch.Tensor
            Routing weights of shape ``[num_tokens, top_k]``.
        w1_weight : torch.Tensor
            GEMM1 weights (gate + up fused for gated activations, or a single
            projection for non-gated activations).
        w1_weight_sf : torch.Tensor
            Scale factors for ``w1_weight``.
        w1_alpha : torch.Tensor
            Per-expert global scale for GEMM1.
        fc2_input_scale : Optional[torch.Tensor]
            Global scale for W4A4 GEMM2 input quantization; must be ``None``
            for W4A16 because GEMM1 output stays in BF16.
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
            Optional W4A4 per-token input row scale for GEMM1.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``[num_tokens, hidden_size]``.
        """
        num_tokens = token_selected_experts.size(0)

        moe_output = torch.empty(
            (num_tokens, self.hidden_size),
            dtype=self.output_dtype,
            device=x.device,
        )

        # Use auto-tuner for tactic selection
        tuner = AutoTuner.get()
        runner: CuteDslFusedMoENvfp4Runner | CuteDslFusedMoEW4A16Runner | None

        if self.quant_mode in ("nvfp4", "w4a4"):
            use_per_token_activation = per_token_scale is not None
            runner = (
                self._per_token_runner if use_per_token_activation else self._runner
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
            activation_name = (
                "Situ" if self.situ_beta is not None else self.activation_type.name
            )
            op_name = f"CuteDslMoEWrapper::run::{activation_name}"
        elif self.quant_mode == "w4a16":
            if (
                x_sf is not None
                or fc2_input_scale is not None
                or per_token_scale is not None
            ):
                raise ValueError(
                    "x_sf, fc2_input_scale, and per_token_scale must be None "
                    "when quant_mode='w4a16'"
                )
            runner = self._w4a16_runner
            inputs = [
                x,
                token_selected_experts,
                token_final_scales,
                w1_weight,
                w1_weight_sf,
                w1_alpha,
                w2_weight,
                w2_weight_sf,
                w2_alpha,
                moe_output,
            ]
            activation_name = (
                "Situ" if self.situ_beta is not None else self.activation_type.name
            )
            op_name = f"CuteDslMoEWrapper::run::W4A16::{activation_name}"
        else:
            raise RuntimeError(f"Unexpected quant_mode {self.quant_mode!r}")

        if runner is None:
            raise RuntimeError(f"{self.quant_mode} runner was not initialized")
        if tactic is not None:
            return runner(inputs, tactic=tactic)

        _, best_tactic = tuner.choose_one(
            op_name,
            [runner],
            runner.tuning_config,
            inputs,
        )
        if self.quant_mode in ("nvfp4", "w4a4"):
            # Timed tactic runs retain the default async path; only this
            # selected-tactic execution is single-stream while tuning.
            runner_kwargs = {"use_async_memset": not tuner.is_tuning_mode}
        elif self.quant_mode == "w4a16":
            runner_kwargs = {}
        else:
            raise RuntimeError(f"Unexpected quant_mode {self.quant_mode!r}")
        return runner(inputs, tactic=best_tactic, **runner_kwargs)

    def get_valid_tactics(self) -> list:
        """Return list of valid tactics for this MoE configuration."""
        if self.quant_mode in ("nvfp4", "w4a4"):
            # _get_arch_tactics() replaces main's ALL_MOE_TACTICS: the tactic
            # list is now architecture-dependent (Blackwell vs Rubin).
            return _get_arch_tactics()
        elif self.quant_mode == "w4a16":
            return list(W4A16_MOE_TACTICS)
        else:
            raise RuntimeError(f"Unexpected quant_mode {self.quant_mode!r}")


# =============================================================================
# Functional API (Simple Function Call)
# =============================================================================


def _cute_dsl_fused_moe_nvfp4_impl(
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: Optional[torch.Tensor],
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
    gemm1_mma_tiler=None,
    gemm1_mma_inst_shape=None,
    gemm2_mma_tiler=None,
    gemm2_mma_inst_shape=None,
    output_dtype: torch.dtype = torch.bfloat16,
    use_fused_finalize: bool = True,
    moe_output: Optional[torch.Tensor] = None,
    per_token_scale: Optional[torch.Tensor] = None,
    aux_stream: Optional[torch.cuda.Stream] = None,
    enable_pdl: bool = True,
    use_async_memset: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
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
        gemm1_mma_tiler=gemm1_mma_tiler,
        gemm1_mma_inst_shape=gemm1_mma_inst_shape,
        gemm2_mma_tiler=gemm2_mma_tiler,
        gemm2_mma_inst_shape=gemm2_mma_inst_shape,
        moe_output=moe_output,
        per_token_scale=per_token_scale,
        aux_stream=aux_stream,
        output_dtype=output_dtype,
        use_async_memset=use_async_memset,
        use_fused_finalize=use_fused_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


@supported_compute_capability([100, 103, 107])
@flashinfer_api(trace=cute_dsl_fused_moe_nvfp4_trace)
def cute_dsl_fused_moe_nvfp4(
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: Optional[torch.Tensor],
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
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    *,
    quant_mode: str = "w4a4",
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
        Packed NVFP4 input for ``quant_mode="w4a4"`` or BF16 input for
        ``quant_mode="w4a16"``.
    x_sf : Optional[torch.Tensor]
        Scale factors for ``quant_mode="w4a4"``; must be ``None`` for
        ``quant_mode="w4a16"``.
    token_selected_experts : torch.Tensor
        Expert assignments of shape ``[num_tokens, top_k]``.
    token_final_scales : torch.Tensor
        Routing weights of shape ``[num_tokens, top_k]``.
    w1_weight : torch.Tensor
        GEMM1 weights (gate + up fused for gated activations, or a single
        projection for non-gated activations).
    w1_weight_sf : torch.Tensor
        Scale factors for ``w1_weight``.
    w1_alpha : torch.Tensor
        Per-expert global scale for GEMM1.
    fc2_input_scale : Optional[torch.Tensor]
        Global scale for W4A4 GEMM2 input quantization; must be ``None`` for
        W4A16 because GEMM1 output stays in BF16.
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
        Use atomic fused finalize; otherwise use the deterministic two-stage
        finalize. Defaults to ``True``.
    moe_output : Optional[torch.Tensor]
        Pre-allocated output buffer.  Allocated internally if ``None``.
    aux_stream : Optional[torch.cuda.Stream]
        Optional auxiliary CUDA stream used to overlap setup work with the
        main computation.
    enable_pdl : bool
        Enable Programmatic Dependent Launch.  Defaults to ``True``.
    activation_type : int
        FC1 activation type. Use ``ActivationType.Swiglu`` for gated
        SwiGLU/SiTU, ``ActivationType.GegluTanh`` for tanh-approximate GeGLU,
        and ``ActivationType.Relu2`` for non-gated ReLU^2. Setting
        ``situ_beta`` selects SiTU; ``swiglu_oai`` is represented as
        ``ActivationType.Swiglu`` with non-default ``swiglu_alpha/beta/limit``.
    swiglu_alpha, swiglu_beta, swiglu_limit : float
        SwiGLU parameters.
    quant_mode : str
        Compute mode: ``"w4a4"`` / ``"nvfp4"`` or ``"w4a16"``. Defaults
        to ``"w4a4"``.
    situ_beta : Optional[float]
        When set with ``ActivationType.Swiglu``, use the SiTU gate
        ``beta * tanh(gate / beta) * sigmoid(gate)``.
    situ_linear_beta : Optional[float]
        Optional SiTU tanh clamp for the up branch.
    per_token_scale : Optional[torch.Tensor]
        Optional W4A4 per-token input row scale for GEMM1.

    Returns
    -------
    torch.Tensor
        Output tensor of shape ``[num_tokens, hidden_size]``.
    """
    _require_cute_dsl_arch_for(x.device, native_only=True)
    activation, _ = normalize_cute_dsl_moe_activation_type(activation_type)
    quant_mode = quant_mode.lower()
    validate_cute_dsl_moe_situ_config(activation, situ_beta, situ_linear_beta)

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
    runner: CuteDslFusedMoENvfp4Runner | CuteDslFusedMoEW4A16Runner

    if quant_mode in ("nvfp4", "w4a4"):
        use_per_token_activation = per_token_scale is not None
        runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=_cute_dsl_fused_moe_nvfp4_impl,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            use_fused_finalize=use_fused_finalize,
            output_dtype=output_dtype,
            enable_pdl=enable_pdl,
            activation_type=activation.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
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

        activation_name = "Situ" if situ_beta is not None else activation.name
        op_name = f"CuteDslFusedMoE::run_moe_nvfp4::{activation_name}"
    elif quant_mode == "w4a16":
        if (
            x_sf is not None
            or fc2_input_scale is not None
            or per_token_scale is not None
        ):
            raise ValueError(
                "x_sf, fc2_input_scale, and per_token_scale must be None "
                "when quant_mode='w4a16'"
            )
        runner = CuteDslFusedMoEW4A16Runner(
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            use_fused_finalize=use_fused_finalize,
            output_dtype=output_dtype,
            enable_pdl=enable_pdl,
            activation_type=activation.value,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        inputs = [
            x,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            moe_output,
        ]
        activation_name = "Situ" if situ_beta is not None else activation.name
        op_name = f"CuteDslFusedMoE::run_moe_w4a16::{activation_name}"
    else:
        raise ValueError(
            f"quant_mode must be 'nvfp4'/'w4a4' or 'w4a16' (got {quant_mode!r})."
        )

    _, best_tactic = tuner.choose_one(
        op_name,
        [runner],
        runner.tuning_config,
        inputs,
        aux_stream=aux_stream,
    )
    if quant_mode in ("nvfp4", "w4a4"):
        runner_kwargs = {
            "aux_stream": aux_stream,
            "use_async_memset": not tuner.is_tuning_mode,
        }
    elif quant_mode == "w4a16":
        runner_kwargs = {"aux_stream": aux_stream}
    else:
        raise RuntimeError(f"Unexpected quant_mode {quant_mode!r}")
    return runner(inputs, tactic=best_tactic, **runner_kwargs)


__all__ = [
    "cute_dsl_fused_moe_nvfp4",
    "CuteDslMoEWrapper",
]
