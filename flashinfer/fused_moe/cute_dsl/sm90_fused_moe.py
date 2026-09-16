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

SM90 (Hopper) CuTe-DSL fused MoE, BF16/FP16.

Three kernels per MoE layer:
  1. ``moe_sort``            (C++/JIT routing index maps — no data movement)
  2. GEMM1: gather + grouped GEMM + SiLU-gating (permute fused in the A load)
  3. GEMM2: grouped GEMM + fused finalize (router-scaled scatter-reduce)

Design doc: docs/design_docs/cute_dsl_moe_sm90.md.
"""

from typing import Any, NamedTuple, Optional, Tuple

import torch

from ...api_logging import flashinfer_api
from ...autotuner import AutoTuner
from ...trace.templates.moe import cute_dsl_fused_moe_bf16_trace
from ...utils import get_compute_capability, supported_compute_capability
from .moe_utils import moe_output_memset_inplace, moe_sort, moe_unpermute
from .sm90_tuner import CuteDslFusedMoESm90Runner
from .sm90_contiguous_gather_grouped_gemm_act_fusion import (
    sm90_contiguous_gather_grouped_gemm_act_fusion,
)
from .sm90_contiguous_grouped_gemm_finalize_fusion import (
    sm90_contiguous_grouped_gemm_finalize_fusion,
)

__all__ = ["cute_dsl_fused_moe_bf16", "CuteDslBf16MoEWrapper"]


class _CudaGraphResources(NamedTuple):
    """Aux-stream resources for overlapping the finalize-destination zeroing
    with GEMM1 (event fork-join; CUDA-graph capturable)."""

    aux_stream: torch.cuda.Stream
    main_event: torch.cuda.Event
    memset_event: torch.cuda.Event


# Created lazily during warmup, before CUDA-graph capture, and reused for the
# process lifetime.
_cuda_graph_resources: Optional[_CudaGraphResources] = None


def _get_cuda_graph_resources() -> _CudaGraphResources:
    global _cuda_graph_resources
    if _cuda_graph_resources is None:
        _cuda_graph_resources = _CudaGraphResources(
            torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event()
        )
    return _cuda_graph_resources


def _moe_core_impl(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int = 0,
    moe_output: Optional[torch.Tensor] = None,
    intermediate_buffer: Optional[torch.Tensor] = None,
    tile_size: int = 128,
    gemm1_tile_shape_mn: Tuple[int, int] = (128, 64),
    gemm1_swizzle_size: int = 1,
    gemm2_tile_shape_mn: Tuple[int, int] = (128, 64),
    gemm2_cluster_shape_mn: Tuple[int, int] = (1, 1),
    gemm2_raster_along_m: bool = False,
    use_fused_finalize: bool = True,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """moe_sort + GEMM1 + GEMM2 pipeline for one tactic.

    The runner (:class:`~.sm90_tuner.CuteDslFusedMoESm90Runner`) fans the
    tactic tuple into the tile keywords; the kernel wrappers validate them
    against the shapes. ``num_local_experts`` is required -- the public entry
    points resolve the ``None``-means-``num_experts`` default.
    """
    # Fail fast on the wrong arch: the routing / DSL modules below can abort
    # the process (not raise) when driven on a non-Hopper GPU.
    major, minor = get_compute_capability(x.device)
    if major != 9:
        raise ValueError(
            f"cute_dsl_fused_moe_bf16 requires SM90 (Hopper). Got SM{major}{minor}."
        )

    num_tokens, hidden = x.shape

    if num_tokens == 0:
        # Empty batch (e.g. DP rank with no tokens this step): nothing to
        # route or compute; moe_sort's routing kernels assume >= 1 token.
        if moe_output is None:
            moe_output = torch.empty(0, hidden, dtype=x.dtype, device=x.device)
        return moe_output

    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_tokens_dim=tile_size,
        enable_pdl=enable_pdl,
    )
    permuted_m = tile_idx_to_expert_idx.numel() * tile_size

    inter = w1_weight.shape[1] // 2
    if intermediate_buffer is None:
        intermediate_buffer = torch.empty(
            permuted_m, inter, dtype=x.dtype, device=x.device
        )

    # Zero the finalize destination on an aux stream, overlapped with GEMM1
    # (the fused scatter-reduce accumulates into it). Events order the zeroing
    # after prior main-stream work and before GEMM2. Deterministic mode
    # skips the zeroing: GEMM2 writes each valid (token, slot) route to its
    # own expanded row and moe_unpermute fully overwrites moe_output.
    if moe_output is None:
        moe_output = torch.empty(num_tokens, hidden, dtype=x.dtype, device=x.device)
    main_stream = torch.cuda.current_stream()
    if use_fused_finalize:
        aux_stream, main_event, memset_event = _get_cuda_graph_resources()
        # Fork/join via events only. Tensor.record_stream is illegal during
        # CUDA-graph capture and redundant here: the join orders later
        # main-stream reuse of moe_output after the auxiliary-stream write.
        main_event.record(main_stream)
        with torch.cuda.stream(aux_stream):
            aux_stream.wait_event(main_event)
            # cudaMemsetAsync via the C++ binding avoids a tensor-operation
            # launch for the destination initialization.
            moe_output_memset_inplace(moe_output)
            memset_event.record(aux_stream)

    intermediate = sm90_contiguous_gather_grouped_gemm_act_fusion(
        x,
        w1_weight,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        out=intermediate_buffer,
        topk=top_k,
        permuted_m=permuted_m,
        tile_shape_mn=gemm1_tile_shape_mn,
        cluster_shape_mn=(1, 1),
        swizzle_size=gemm1_swizzle_size,
        enable_pdl=enable_pdl,
    )

    if use_fused_finalize:
        # The zeroing must complete before the finalize scatter-reduce.
        main_stream.wait_event(memset_event)
        gemm2_output = moe_output
    else:
        # Deterministic path: unscaled expert rows in expanded
        # (token * top_k + slot) order; scales applied by moe_unpermute.
        gemm2_output = torch.empty(
            num_tokens * top_k, hidden, dtype=x.dtype, device=x.device
        )

    sm90_contiguous_grouped_gemm_finalize_fusion(
        intermediate,
        w2_weight,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        token_final_scales,
        gemm2_output,
        topk=top_k,
        use_fused_finalize=use_fused_finalize,
        tile_shape_mn=gemm2_tile_shape_mn,
        cluster_shape_mn=gemm2_cluster_shape_mn,
        raster_along_m=gemm2_raster_along_m,
        enable_pdl=enable_pdl,
    )

    if not use_fused_finalize:
        # Fixed-order routing-weight reduction (bitwise-reproducible).
        moe_unpermute(
            permuted_input=gemm2_output,
            output=moe_output,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
            topk_scales=token_final_scales,
            num_tokens=num_tokens,
            top_k=top_k,
            enable_pdl=enable_pdl,
            input_is_expanded=True,
        )
    return moe_output


@supported_compute_capability([90])
@flashinfer_api(trace=cute_dsl_fused_moe_bf16_trace)
def cute_dsl_fused_moe_bf16(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: Optional[int] = None,
    local_expert_offset: int = 0,
    use_fused_finalize: bool = True,
    moe_output: Optional[torch.Tensor] = None,
    enable_pdl: bool = True,
    *,
    intermediate_buffer: Optional[torch.Tensor] = None,
    tactic: Optional[Tuple[Any, ...]] = None,
) -> torch.Tensor:
    """SM90 CuTe-DSL fused MoE forward (BF16/FP16, unquantized).

    ``out[t] = sum_k scale[t,k] * ffn_expert(x[t]; e[t,k])`` with
    ``ffn(x; e) = (silu(x @ w1_gate[e].T) * (x @ w1_up[e].T)) @ w2_weight[e].T``.

    Supported configuration:
        * Arch: SM90 (Hopper) only.
        * Dtypes: bf16 or fp16 activations and weights (must match), fp32
          accumulation; output dtype = input dtype. No quantized paths.
        * Activation: SwiGLU (SiLU-gated) only, fused into GEMM1.
        * Routing: pre-routed contract only — the caller runs the router and
          passes global expert ids plus **normalized** scales. ``top_k`` is a
          compile-time constant of the kernels.
        * Parallelism: TP by weight shapes; EP via ``num_local_experts`` +
          ``local_expert_offset`` (tokens routed entirely outside the local
          shard contribute zeros).
        * Shapes: ``hidden % 64 == 0`` (GEMM1's reduction moves whole
          64-element K tiles), ``2I % 64 == 0``, ``I % 32 == 0`` (weight
          interleave; GEMM2's K tail is zero-filled by TMA);
          ``num_tokens == 0`` is supported.
        * Execution: CUDA-graph capturable; PDL on by default; fused
          finalize (default) is atomic and not bitwise-reproducible —
          ``use_fused_finalize=False`` selects the deterministic path.

    Tile selection goes through the FlashInfer AutoTuner. Under the
    :func:`autotune` context every tactic the runner offers for the token
    bucket (:meth:`~.sm90_tuner.CuteDslFusedMoESm90Runner.get_valid_tactics`)
    is profiled and the per-bucket winner is cached; outside it the cached
    winner (or the fixed default, :data:`~.sm90_tuner.DEFAULT_SM90_MOE_TACTIC`:
    tile 128, 64-wide N tiles, cluster (1, 1), N-major walks) dispatches. An
    explicit ``tactic`` bypasses the tuner::

        with autotune(True):
            output = cute_dsl_fused_moe_bf16(...)

    Args:
        x: ``[num_tokens, hidden]`` bf16/fp16.
        token_selected_experts: ``[num_tokens, top_k]`` int32.
        token_final_scales: ``[num_tokens, top_k]`` float32, normalized by the
            caller.
        w1_weight: ``[num_local_experts, 2I, hidden]`` — up/gate interleaved at 32
            columns. Callers may cache this repack; the in-tree reference is
            :func:`~.sm90_contiguous_gather_grouped_gemm_act_fusion.interleave_up_gate_sm90`.
        w2_weight: ``[num_local_experts, hidden, I]``.
        num_experts: Total (global) expert count.
        top_k: Experts per token.
        num_local_experts: Experts held by this rank (EP shard); defaults to
            ``num_experts``.
        local_expert_offset: Global id of this shard's first expert.
        use_fused_finalize: True (default) fuses the router-scaled
            scatter-reduce into GEMM2. The top-k combine then accumulates in
            the output dtype (``cp.reduce.async.bulk.add``): one
            output-dtype rounding per route on top of the bf16/fp16
            intermediate hand-off, and not bitwise-reproducible across
            runs. False uses the deterministic two-stage path: GEMM2
            scatters unscaled rows in expanded (token, slot) order, then
            ``moe_unpermute`` applies the scales and combines in float32
            in a fixed order — one final rounding, at the cost of an extra
            kernel and the expanded intermediate.
        moe_output: Optional pre-allocated ``[num_tokens, hidden]`` output
            (contents overwritten; zeroed internally for the fused finalize).
        enable_pdl: True (default) launches both GEMMs (and the deterministic
            path's ``moe_unpermute``) with Programmatic Dependent Launch so
            each kernel's prologue overlaps its predecessor's tail. Numerics
            are unaffected. Part of the kernel compile cache key.
        intermediate_buffer: Optional pre-allocated GEMM1 output buffer
            (advanced, keyword-only).
        tactic: Optional ``(tile_size, gemm1_tactic, gemm2_tactic)`` tuple
            (see :mod:`~.sm90_tuner`) dispatched as given, validated by the
            kernel wrappers; ``None`` selects through the AutoTuner.

    Returns:
        ``[num_tokens, hidden]`` in x's dtype.
    """
    if num_local_experts is None:
        num_local_experts = num_experts

    num_tokens, hidden = x.shape
    if num_tokens == 0:
        # Empty batch: nothing to route; moe_sort assumes >= 1 token.
        if moe_output is None:
            moe_output = torch.empty(0, hidden, dtype=x.dtype, device=x.device)
        return moe_output
    if moe_output is None:
        moe_output = torch.empty(num_tokens, hidden, dtype=x.dtype, device=x.device)

    runner = CuteDslFusedMoESm90Runner(
        forward_impl=_moe_core_impl,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        use_fused_finalize=use_fused_finalize,
        enable_pdl=enable_pdl,
    )
    inputs = [
        x,
        token_selected_experts,
        token_final_scales,
        w1_weight,
        w2_weight,
        moe_output,
    ]
    if tactic is not None:
        return runner(inputs, tactic=tactic, intermediate_buffer=intermediate_buffer)
    _, best_tactic = AutoTuner.get().choose_one(
        "CuteDslFusedMoE::run_moe_sm90::Swiglu",
        [runner],
        runner.tuning_config,
        inputs,
    )
    return runner(inputs, tactic=best_tactic, intermediate_buffer=intermediate_buffer)


class CuteDslBf16MoEWrapper:
    """SM90 CuTe-DSL fused-MoE wrapper (bf16/fp16, unquantized).

    Holds the static MoE configuration so call sites only pass tensors.
    ``run`` is CUDA-graph capturable (the
    underlying pipeline is graph-safe) and delegates to
    :func:`cute_dsl_fused_moe_bf16`, which carries the API logging/trace
    decoration and documents the supported configuration.
    Auto-tuning is controlled by the :func:`autotune` context. Warm up the
    selected tactic once before CUDA-graph capture or serving so its
    process-local kernels are compiled.

    Example (auto-tuning):
        >>> moe = CuteDslBf16MoEWrapper(
        ...     num_experts=128, top_k=8, hidden_size=2048,
        ...     intermediate_size=768,
        ... )
        >>> with autotune(True):
        ...     out = moe.run(x, topk_ids, topk_weights, w1_weight, w2_weight)
    """

    @supported_compute_capability([90])
    @flashinfer_api
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_local_experts: Optional[int] = None,
        local_expert_offset: int = 0,
        output_dtype: torch.dtype = torch.bfloat16,
        enable_pdl: bool = True,
        use_fused_finalize: bool = True,
    ):
        """Configure the SM90 fused-MoE wrapper.

        Args:
            num_experts: Total (global) expert count.
            top_k: Experts per token.
            hidden_size: Model hidden dimension.
            intermediate_size: Per-rank expert intermediate dimension
                (``w1_weight`` is ``[E_local, 2*intermediate, hidden]`` interleaved,
                ``w2_weight`` is ``[E_local, hidden, intermediate]``).
            num_local_experts: Experts held by this rank (EP shard);
                defaults to ``num_experts``.
            local_expert_offset: Global id of this shard's first expert.
            output_dtype: Output (= activation/weight) dtype, bf16 or fp16;
                used to allocate ``moe_output`` when the caller does not
                provide one.
            enable_pdl: Launch the kernels with Programmatic Dependent
                Launch (default True; see :func:`cute_dsl_fused_moe_bf16`).
            use_fused_finalize: True (default) fuses the router-scaled
                scatter-reduce into GEMM2; False selects the
                bitwise-reproducible two-stage finalize.
        """
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = (
            num_local_experts if num_local_experts is not None else num_experts
        )
        self.local_expert_offset = local_expert_offset
        self.use_fused_finalize = use_fused_finalize
        self.output_dtype = output_dtype
        self.enable_pdl = enable_pdl

    def run(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        tactic: Optional[Tuple[Any, ...]] = None,
        moe_output: Optional[torch.Tensor] = None,
        intermediate_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the fused MoE forward; see :func:`cute_dsl_fused_moe_bf16`.

        ``tactic`` dispatches that tactic tuple as given; ``None`` selects
        through the AutoTuner (the cached winner for these shapes when one
        exists, otherwise the fixed default).
        """
        if moe_output is None:
            moe_output = torch.empty(
                x.shape[0], self.hidden_size, dtype=self.output_dtype, device=x.device
            )
        return cute_dsl_fused_moe_bf16(
            x,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w2_weight,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            moe_output=moe_output,
            intermediate_buffer=intermediate_buffer,
            use_fused_finalize=self.use_fused_finalize,
            enable_pdl=self.enable_pdl,
            tactic=tactic,
        )
