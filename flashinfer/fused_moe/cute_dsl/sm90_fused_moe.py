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

from typing import List, NamedTuple, Optional, Tuple

import torch

from ...api_logging import flashinfer_api
from ...autotuner import AutoTuner
from ...trace.templates.moe import cute_dsl_fused_moe_bf16_trace
from ...utils import get_compute_capability, supported_compute_capability
from .moe_utils import moe_output_memset_inplace, moe_sort, moe_unpermute
from .sm90_tuner import (
    _GEMM1_TILE_N_BY_TILE_SIZE,
    _GEMM2_TILE_N_BY_TILE_SIZE,
    CuteDslFusedMoESm90Runner,
    Sm90MoeTactic,
    _decode_sm90_moe_tactic,
    _default_gemm2_tile_k,
    _enumerate_sm90_moe_tactics,
    _gemm2_tactic_can_implement,
    _Sm90MoeTacticOverride,
)
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
    tile_size: Optional[int] = None,
    gemm1_tile_n: Optional[int] = None,
    gemm2_tile_n: Optional[int] = None,
    gemm2_tile_k: Optional[int] = None,
    gemm2_cluster_shape_mn: Optional[Tuple[int, int]] = None,
    gemm2_raster_along_m: Optional[bool] = None,
    use_fused_finalize: bool = True,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """moe_sort + GEMM1 + GEMM2 pipeline with explicit (or auto-selected)
    tile parameters; :func:`cute_dsl_fused_moe_bf16` dispatches here with
    the AutoTuner-selected tactic. ``num_local_experts`` is required — the
    public entry points resolve the ``None``-means-``num_experts`` default."""
    # Fail fast on the wrong arch: the routing / DSL modules below can abort
    # the process (not raise) when driven on a non-Hopper GPU.
    major, minor = get_compute_capability(x.device)
    if major != 9:
        raise ValueError(
            f"cute_dsl_fused_moe_bf16 requires SM90 (Hopper). Got SM{major}{minor}."
        )

    num_tokens, hidden = x.shape
    inter2 = w1_weight.shape[1]

    if num_tokens == 0:
        # Empty batch (e.g. DP rank with no tokens this step): nothing to
        # route or compute; moe_sort's routing kernels assume >= 1 token.
        if moe_output is None:
            moe_output = torch.empty(0, hidden, dtype=x.dtype, device=x.device)
        return moe_output

    inter_per_rank = w2_weight.shape[2]
    if tile_size is None:
        # Tile 64 halves the padded-MMA waste of small batches (each touched
        # expert pads to a full M-tile), but tile 128's fatter tiles win as
        # soon as experts average one full 64-row tile (kernel A/B at 64
        # rows/expert: GEMM1 +7..13%, GEMM2 +4..36% across per-rank I).
        # Tiny reductions (I < 192) amortize GEMM2's fixed per-tile cost so
        # poorly at tile 64 that they flip far earlier.
        avg_rows_per_expert = num_tokens * top_k / num_local_experts
        decode_limit = 16 if inter_per_rank < 192 else 64
        tile_size = 64 if avg_rows_per_expert < decode_limit else 128

    # Tile N must divide the (padded) GEMM N extents; pick the largest legal
    # candidate. Gated GEMM1 requires tile_n % 64 == 0.
    def _pick_tile_n(n, candidates):
        for c in candidates:
            if n % c == 0:
                return c
        raise ValueError(f"no supported tile N divides n={n}")

    # Prefer the largest tile N (fewer tiles, better B-load amortization and
    # fewer A re-reads); 2-warpgroup tiles (N > 128 at tile M 128) are
    # validated. At decode (tile_size 64) stick to 1-WG tiles — the wide-tile
    # register economics only pay off with 2 consumer warpgroups.
    if gemm1_tile_n is None:
        gemm1_tile_n = _pick_tile_n(inter2, _GEMM1_TILE_N_BY_TILE_SIZE[tile_size])
    if gemm2_tile_n is None:
        gemm2_tile_n = _pick_tile_n(hidden, _GEMM2_TILE_N_BY_TILE_SIZE[tile_size])

    # Resolve and validate every GEMM2 topology field before routing or
    # launching GEMM1. The fallback dispatch retains the measured policies;
    # tuned and explicit calls supply the cluster/raster axes directly.
    if gemm2_tile_k is None:
        gemm2_tile_k = _default_gemm2_tile_k(inter_per_rank, tile_size)
    if gemm2_cluster_shape_mn is None:
        gemm2_cluster_shape_mn = (
            (1, 2)
            if inter_per_rank >= 192 and num_tokens * top_k >= 256 * num_local_experts
            else (1, 1)
        )
        if not _gemm2_tactic_can_implement(
            hidden,
            inter_per_rank,
            (tile_size, gemm2_tile_n),
            gemm2_tile_k,
            gemm2_cluster_shape_mn,
        ):
            gemm2_cluster_shape_mn = (1, 1)
    elif not _gemm2_tactic_can_implement(
        hidden,
        inter_per_rank,
        (tile_size, gemm2_tile_n),
        gemm2_tile_k,
        gemm2_cluster_shape_mn,
    ):
        raise ValueError(
            "GEMM2 tactic cannot implement "
            f"hidden={hidden}, intermediate={inter_per_rank}, "
            f"tile_shape_mn={(tile_size, gemm2_tile_n)}, tile_k={gemm2_tile_k}, "
            f"cluster_shape_mn={gemm2_cluster_shape_mn}"
        )
    if gemm2_raster_along_m is None:
        out_bytes = num_tokens * hidden * x.element_size()
        if tile_size < 128 or inter_per_rank > 384:
            gemm2_raster_along_m = False
        elif inter_per_rank <= 192:
            gemm2_raster_along_m = out_bytes >= 32 * 1024 * 1024
        else:
            gemm2_raster_along_m = out_bytes >= 64 * 1024 * 1024
    elif not isinstance(gemm2_raster_along_m, bool):
        raise ValueError("gemm2_raster_along_m must be bool or None")

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

    # Keep GEMM1 unclustered. L2 can service concurrent same-expert B reads,
    # while clustered execution constrains CTA scheduling and requires both
    # members of each pair to traverse the pipeline.
    gemm1_cluster = (1, 1)

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
        tile_shape_mn=(tile_size, gemm1_tile_n),
        cluster_shape_mn=gemm1_cluster,
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
        tile_shape_mn=(tile_size, gemm2_tile_n),
        tile_k=gemm2_tile_k,
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
    tile_size: Optional[int] = None,
    gemm1_tile_n: Optional[int] = None,
    gemm2_tile_n: Optional[int] = None,
    gemm2_tile_k: Optional[int] = None,
    gemm2_cluster_shape_mn: Optional[Tuple[int, int]] = None,
    gemm2_raster_along_m: Optional[bool] = None,
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
        * Shapes: ``hidden % 64 == 0`` and (GEMM1 reduction) no tile-32
          fallback, ``2I % 64 == 0``, ``I % 32 == 0`` (weight interleave and
          GEMM2 tile-k 32 fallback for ``I % 64 != 0``); ``num_tokens == 0``
          is supported.
        * Execution: CUDA-graph capturable; PDL on by default; fused
          finalize (default) is atomic and not bitwise-reproducible —
          ``use_fused_finalize=False`` selects the deterministic path.

    Tile selection goes through the FlashInfer AutoTuner. Under the
    :func:`autotune` context
    every enumerated :class:`Sm90MoeTactic` (capped at the top-2 legal N
    tiles per GEMM) is profiled and the per-bucket winner is
    cached; outside it the cached winner (or the heuristic auto-selection,
    as the default tactic) dispatches. Explicit tile / cluster / raster /
    buffer keyword overrides bypass the tuner::

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
            (advanced, keyword-only; bypasses the tuner like the tile
            overrides).
        tile_size: Tile size shared by moe_sort and both GEMMs (64 or 128;
            keyword-only). Default None auto-selects: 64 below an average of
            64 rows per local expert (``num_tokens * top_k /
            num_local_experts``) — small/decode batches pad each expert to a
            full M-tile, so the smaller tile roughly halves the wasted MMA
            work; 128 from one full tile per expert up (fatter tiles
            amortize per-tile fixed costs and B loads). Tiny reductions
            (per-rank ``I < 192``) switch to 128 already at 16 rows per
            expert.
        gemm1_tile_n: N tile for GEMM1 (None auto-selects; keyword-only).
        gemm2_tile_n: N tile for GEMM2 (None auto-selects; keyword-only).
        gemm2_tile_k: K tile for GEMM2, 64 or 32 (None auto-selects via
            :func:`_default_gemm2_tile_k`: 32 when 64 does not divide the
            per-rank I, and on prefill tiles with I >= 384 where its doubled
            pipeline depth wins; 64 otherwise. Keyword-only).
        gemm2_cluster_shape_mn: GEMM2 CTA cluster shape, ``(1, 1)`` or
            ``(1, 2)`` (None applies the fallback heuristic; keyword-only).
            ``(1, 2)`` requires an even GEMM2 N-tile count.
        gemm2_raster_along_m: GEMM2 tile raster order (None auto-selects;
            keyword-only). M-major confines the finalize scatter-RMW working
            set to one L2-resident output column band per CTA wave, at the
            cost of re-reading each A tile once per N tile — the default
            enables it only where that trade wins: prefill tiles with a
            large output working set (>= 32 MiB for per-rank ``I <= 192``,
            >= 64 MiB above) and per-rank ``I <= 384``.

    Returns:
        ``[num_tokens, hidden]`` in x's dtype.
    """
    if num_local_experts is None:
        num_local_experts = num_experts

    # Explicit tile / buffer overrides are deterministic direct dispatch
    # (tests, benchmarks); the tuner only owns the auto-selected path.
    if (
        tile_size is not None
        or gemm1_tile_n is not None
        or gemm2_tile_n is not None
        or gemm2_tile_k is not None
        or gemm2_cluster_shape_mn is not None
        or gemm2_raster_along_m is not None
        or intermediate_buffer is not None
    ):
        return _moe_core_impl(
            x,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w2_weight,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            moe_output=moe_output,
            intermediate_buffer=intermediate_buffer,
            tile_size=tile_size,
            gemm1_tile_n=gemm1_tile_n,
            gemm2_tile_n=gemm2_tile_n,
            gemm2_tile_k=gemm2_tile_k,
            gemm2_cluster_shape_mn=gemm2_cluster_shape_mn,
            gemm2_raster_along_m=gemm2_raster_along_m,
            use_fused_finalize=use_fused_finalize,
            enable_pdl=enable_pdl,
        )

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
    _, best_tactic = AutoTuner.get().choose_one(
        "CuteDslFusedMoE::run_moe_sm90::Swiglu",
        [runner],
        runner.tuning_config,
        inputs,
    )
    return runner(inputs, tactic=best_tactic)


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
        tile_size: Optional[int] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        enable_pdl: bool = True,
        use_fused_finalize: bool = True,
        gemm1_tile_n: Optional[int] = None,
        gemm2_tile_n: Optional[int] = None,
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
            tile_size: Optional tile override; ``None`` auto-selects (see
                :func:`cute_dsl_fused_moe_bf16`).
            output_dtype: Output (= activation/weight) dtype, bf16 or fp16;
                used to allocate ``moe_output`` when the caller does not
                provide one.
            enable_pdl: Launch the kernels with Programmatic Dependent
                Launch (default True; see :func:`cute_dsl_fused_moe_bf16`).
            use_fused_finalize: True (default) fuses the router-scaled
                scatter-reduce into GEMM2; False selects the
                bitwise-reproducible two-stage finalize.
            gemm1_tile_n: Optional GEMM1 N-tile override.
            gemm2_tile_n: Optional GEMM2 N-tile override.
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
        self.tile_size = tile_size
        self.gemm1_tile_n = gemm1_tile_n
        self.gemm2_tile_n = gemm2_tile_n
        self.enable_pdl = enable_pdl

    def run(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        tactic: Optional[Sm90MoeTactic] = None,
        moe_output: Optional[torch.Tensor] = None,
        intermediate_buffer: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the fused MoE forward; see :func:`cute_dsl_fused_moe_bf16`.
        ``tactic`` overrides the instance tile config for this call."""
        if tactic is None:
            tactic_override = _Sm90MoeTacticOverride(
                self.tile_size,
                self.gemm1_tile_n,
                self.gemm2_tile_n,
                None,
                None,
                None,
            )
        else:
            tactic_override = _decode_sm90_moe_tactic(tactic)
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
            tile_size=tactic_override.tile_size,
            gemm1_tile_n=tactic_override.gemm1_tile_n,
            gemm2_tile_n=tactic_override.gemm2_tile_n,
            gemm2_tile_k=tactic_override.gemm2_tile_k,
            gemm2_cluster_shape_mn=tactic_override.gemm2_cluster_shape_mn,
            gemm2_raster_along_m=tactic_override.gemm2_raster_along_m,
            use_fused_finalize=self.use_fused_finalize,
            enable_pdl=self.enable_pdl,
        )

    def get_valid_tactics(self) -> List[Sm90MoeTactic]:
        """All tunable tactics for this wrapper's geometry."""
        return _enumerate_sm90_moe_tactics(
            2 * self.intermediate_size,
            self.hidden_size,
            self.intermediate_size,
        )
