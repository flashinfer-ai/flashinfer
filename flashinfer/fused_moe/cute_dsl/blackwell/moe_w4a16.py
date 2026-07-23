# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""SM100 NVFP4-weight, BF16-activation fused MoE launcher."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import get_max_active_clusters, make_ptr
from flashinfer.fused_moe.cute_dsl.moe_utils import (
    allocate_moe_sort_buffers,
    get_max_num_permuted_tokens,
    moe_output_memset_inplace,
    moe_permute,
    moe_sort,
    moe_unpermute,
    normalize_cute_dsl_moe_activation_type,
)
from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)

from .moe_w4a16_kernel import Sm100W4A16GroupedGemmKernel
from .moe_w4a16_permute import moe_permute_bf16_pdl
from .moe_w4a16_unpermute import moe_unpermute_bf16_pdl


W4A16GemmTactic = Tuple[Tuple[int, int], Tuple[int, int]]
W4A16MoeTactic = Tuple[int, W4A16GemmTactic, W4A16GemmTactic]

_MIN_GEMM1_M_CLUSTERS_FOR_PERMUTE_OVERLAP = 8
_SPARSE_UNPERMUTE_NUM_WARPS = 16
_DENSE_UNPERMUTE_NUM_WARPS = 32

# Fixed correctness fallback used when no tuned tactic is available. Runtime
# performance selection belongs to CuteDslFusedMoEW4A16Runner.
DEFAULT_W4A16_MOE_TACTIC: W4A16MoeTactic = (
    128,
    ((256, 128), (2, 1)),
    ((256, 128), (2, 1)),
)


@dataclass
class _W4A16Workspace:
    num_tokens_capacity: int
    moe_sort_buffers: Dict[str, torch.Tensor]
    hidden_workspace: torch.Tensor
    intermediate: torch.Tensor


_kernel_cache: Dict[Tuple, object] = {}


def _get_workspace(
    x: torch.Tensor,
    top_k: int,
    num_experts: int,
    num_local_experts: int,
    intermediate_size: int,
    route_tile: int,
    workspace_cache: Optional[Dict[Tuple, _W4A16Workspace]] = None,
) -> _W4A16Workspace:
    num_tokens = int(x.size(0))
    route_slots = get_max_num_permuted_tokens(
        num_tokens, top_k, num_local_experts, route_tile
    )
    key = (
        x.device,
        top_k,
        int(x.size(1)),
        int(intermediate_size),
        int(num_experts),
        int(num_local_experts),
        route_tile,
    )
    workspace = workspace_cache.get(key) if workspace_cache is not None else None
    if workspace is None or workspace.num_tokens_capacity < num_tokens:
        workspace = _W4A16Workspace(
            num_tokens_capacity=num_tokens,
            moe_sort_buffers=allocate_moe_sort_buffers(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=num_local_experts,
                tile_tokens_dim=route_tile,
                device=x.device,
            ),
            # Permuted input is dead before GEMM2 writes its output.
            hidden_workspace=torch.empty(
                (route_slots, x.size(1)), dtype=torch.bfloat16, device=x.device
            ),
            intermediate=torch.empty(
                (route_slots, intermediate_size),
                dtype=torch.bfloat16,
                device=x.device,
            ),
        )
        if workspace_cache is not None:
            workspace_cache[key] = workspace
    return workspace


def _get_compiled_kernel(
    num_local_experts: int,
    weight_ptr,
    weight_sf_ptr,
    activation_ptr,
    tile_idx_to_expert_idx_ptr,
    tile_idx_to_mn_limit_ptr,
    num_non_exiting_tiles_ptr,
    alpha_ptr,
    output_ptr,
    permuted_idx_to_expanded_idx_ptr,
    token_final_scales_ptr,
    m: int,
    n: int,
    k: int,
    num_tokens: int,
    top_k: int,
    max_active_clusters: int,
    stream,
    activation_type: Optional[ActivationType],
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
    use_fused_finalize: bool,
    enable_pdl: bool,
    use_clc_scheduler: bool,
    route_tile: int,
    mma_tiler_mk: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
):
    mma_tiler_m, mma_tiler_k = mma_tiler_mk
    use_2cta_instrs = mma_tiler_m == 256
    cache_key = (
        num_local_experts,
        activation_type,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        use_fused_finalize,
        enable_pdl,
        use_clc_scheduler,
        mma_tiler_m,
        mma_tiler_k,
        route_tile,
        cluster_shape_mn,
    )
    compiled = _kernel_cache.get(cache_key)
    if compiled is None:
        kernel = Sm100W4A16GroupedGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mnk=(mma_tiler_m, route_tile, mma_tiler_k),
            cluster_shape_mn=cluster_shape_mn,
            group_count=num_local_experts,
            activation_type=(
                int(activation_type) if activation_type is not None else None
            ),
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            use_fused_finalize=use_fused_finalize,
            enable_pdl=enable_pdl,
            use_clc_scheduler=use_clc_scheduler,
        )
        compiled = cute.compile(
            kernel.wrapper,
            weight_ptr,
            weight_sf_ptr,
            activation_ptr,
            tile_idx_to_expert_idx_ptr,
            tile_idx_to_mn_limit_ptr,
            num_non_exiting_tiles_ptr,
            alpha_ptr,
            output_ptr,
            permuted_idx_to_expanded_idx_ptr,
            token_final_scales_ptr,
            m,
            n,
            k,
            num_tokens,
            top_k,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
        _kernel_cache[cache_key] = compiled
    return compiled


def _run_grouped_gemm(
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    activations: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    alpha: torch.Tensor,
    output: torch.Tensor,
    num_local_experts: int,
    activation_type: Optional[ActivationType],
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
    use_fused_finalize: bool,
    permuted_idx_to_expanded_idx: Optional[torch.Tensor],
    token_final_scales: Optional[torch.Tensor],
    enable_pdl: bool,
    route_tile: int,
    tactic: W4A16GemmTactic,
) -> None:
    m = int(weight.size(1))
    k = int(weight.size(2)) * 2
    n = int(activations.size(0))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    mma_tiler_mk, cluster_shape_mn = tactic
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    cta_group_size = 2 if mma_tiler_mk[0] == 256 else 1
    cta_tile_m = mma_tiler_mk[0] // cta_group_size
    num_ctas_m = (m + cta_tile_m - 1) // cta_tile_m
    num_ctas_n = (n + route_tile - 1) // route_tile
    num_problem_clusters = (
        (num_ctas_m + cluster_shape_mn[0] - 1) // cluster_shape_mn[0]
    ) * ((num_ctas_n + cluster_shape_mn[1] - 1) // cluster_shape_mn[1])
    use_clc_scheduler = (
        activation_type is not None and num_problem_clusters > max_active_clusters
    )
    weight_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        weight.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    weight_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        weight_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    activation_ptr = make_ptr(
        cutlass.BFloat16,
        activations.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    tile_idx_to_expert_idx_ptr = make_ptr(
        cutlass.Int32,
        tile_idx_to_expert_idx.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    tile_idx_to_mn_limit_ptr = make_ptr(
        cutlass.Int32,
        tile_idx_to_mn_limit.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    num_non_exiting_tiles_ptr = make_ptr(
        cutlass.Int32,
        num_non_exiting_tiles.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    alpha_ptr = make_ptr(
        cutlass.Float32,
        alpha.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    output_ptr = make_ptr(
        cutlass.BFloat16,
        output.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    permuted_idx_to_expanded_idx_ptr = (
        make_ptr(
            cutlass.Int32,
            permuted_idx_to_expanded_idx.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if permuted_idx_to_expanded_idx is not None
        else None
    )
    token_final_scales_ptr = (
        make_ptr(
            cutlass.Float32,
            token_final_scales.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if token_final_scales is not None
        else None
    )
    num_tokens = int(output.size(0)) if use_fused_finalize else 0
    top_k = int(token_final_scales.size(1)) if token_final_scales is not None else 0
    compiled = _get_compiled_kernel(
        num_local_experts,
        weight_ptr,
        weight_sf_ptr,
        activation_ptr,
        tile_idx_to_expert_idx_ptr,
        tile_idx_to_mn_limit_ptr,
        num_non_exiting_tiles_ptr,
        alpha_ptr,
        output_ptr,
        permuted_idx_to_expanded_idx_ptr,
        token_final_scales_ptr,
        m,
        n,
        k,
        num_tokens,
        top_k,
        max_active_clusters,
        stream,
        activation_type,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        use_fused_finalize,
        enable_pdl,
        use_clc_scheduler,
        route_tile,
        mma_tiler_mk,
        cluster_shape_mn,
    )
    compiled(
        weight_ptr,
        weight_sf_ptr,
        activation_ptr,
        tile_idx_to_expert_idx_ptr,
        tile_idx_to_mn_limit_ptr,
        num_non_exiting_tiles_ptr,
        alpha_ptr,
        output_ptr,
        permuted_idx_to_expanded_idx_ptr,
        token_final_scales_ptr,
        m,
        n,
        k,
        num_tokens,
        top_k,
        stream=stream,
    )


def launch_w4a16_moe(
    x: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    local_expert_offset: int,
    moe_output: torch.Tensor,
    use_fused_finalize: bool,
    enable_pdl: bool,
    activation_type: ActivationType,
    swiglu_alpha: float = DEFAULT_SWIGLU_ALPHA,
    swiglu_beta: float = DEFAULT_SWIGLU_BETA,
    swiglu_limit: float = DEFAULT_SWIGLU_LIMIT,
    tactic: Optional[W4A16MoeTactic] = None,
    workspace_cache: Optional[Dict[Tuple, _W4A16Workspace]] = None,
) -> torch.Tensor:
    """Run BF16 activations against online-decoded NVFP4 expert weights."""
    top_k = int(token_selected_experts.size(1))
    intermediate_size = int(w2_weight.size(2)) * 2
    activation_type, gated = normalize_cute_dsl_moe_activation_type(activation_type)
    gemm1_output_size = intermediate_size * (2 if gated else 1)
    if int(w1_weight.size(1)) != gemm1_output_size:
        raise ValueError(
            f"w1_weight dim 1 must be {gemm1_output_size} for "
            f"{activation_type.name}, got {w1_weight.size(1)}"
        )
    if tactic is None:
        tactic = DEFAULT_W4A16_MOE_TACTIC
    route_tile, gemm1_tactic, gemm2_tactic = tactic
    gemm1_mma_m = gemm1_tactic[0][0]
    gemm1_cluster_m = gemm1_tactic[1][0]
    gemm1_m_cluster_width = gemm1_mma_m * gemm1_cluster_m
    num_gemm1_m_clusters = (
        gemm1_output_size + gemm1_m_cluster_width - 1
    ) // gemm1_m_cluster_width
    # Wide GEMM1s can overlap the copy when one producer CTA occupies each SM.
    # Narrow GEMM1s leave the copy on the critical path and use 1.5 CTAs per SM.
    permute_ctas_per_sm = (
        (1, 1)
        if num_gemm1_m_clusters >= _MIN_GEMM1_M_CLUSTERS_FOR_PERMUTE_OVERLAP
        else (3, 2)
    )
    workspace = _get_workspace(
        x,
        top_k,
        num_experts,
        num_local_experts,
        intermediate_size,
        route_tile,
        workspace_cache,
    )

    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        _,
        num_non_exiting_tiles,
    ) = moe_sort(
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_tokens_dim=route_tile,
        enable_pdl=enable_pdl,
        **workspace.moe_sort_buffers,
    )
    num_tokens = int(x.size(0))
    route_slots = get_max_num_permuted_tokens(
        num_tokens, top_k, num_local_experts, route_tile
    )
    num_route_tiles = route_slots // route_tile
    tile_idx_to_expert_idx = tile_idx_to_expert_idx[:num_route_tiles]
    tile_idx_to_mn_limit = tile_idx_to_mn_limit[:num_route_tiles]
    expanded_idx_to_permuted_idx = expanded_idx_to_permuted_idx[:num_tokens]
    permuted_idx_to_expanded_idx = permuted_idx_to_expanded_idx[:route_slots]
    hidden_workspace = workspace.hidden_workspace[:route_slots]
    intermediate = workspace.intermediate[:route_slots]
    if enable_pdl:
        moe_permute_bf16_pdl(
            input=x,
            permuted_output=hidden_workspace,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            max_num_permuted_tokens=route_slots,
            top_k=top_k,
            tile_size=route_tile,
            ctas_per_sm=permute_ctas_per_sm,
        )
    else:
        moe_permute(
            input=x,
            permuted_output=hidden_workspace,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            max_num_permuted_tokens=route_slots,
            top_k=top_k,
            tile_size=route_tile,
            enable_pdl=False,
        )
    _run_grouped_gemm(
        weight=w1_weight,
        weight_sf=w1_weight_sf,
        activations=hidden_workspace,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        alpha=w1_alpha,
        output=intermediate,
        num_local_experts=num_local_experts,
        activation_type=activation_type,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        use_fused_finalize=False,
        permuted_idx_to_expanded_idx=None,
        token_final_scales=None,
        enable_pdl=enable_pdl,
        route_tile=route_tile,
        tactic=gemm1_tactic,
    )
    gemm2_output = moe_output if use_fused_finalize else hidden_workspace
    if use_fused_finalize:
        moe_output_memset_inplace(moe_output)
    _run_grouped_gemm(
        weight=w2_weight,
        weight_sf=w2_weight_sf,
        activations=intermediate,
        tile_idx_to_expert_idx=tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        alpha=w2_alpha,
        output=gemm2_output,
        num_local_experts=num_local_experts,
        activation_type=None,
        swiglu_alpha=DEFAULT_SWIGLU_ALPHA,
        swiglu_beta=DEFAULT_SWIGLU_BETA,
        swiglu_limit=DEFAULT_SWIGLU_LIMIT,
        use_fused_finalize=use_fused_finalize,
        permuted_idx_to_expanded_idx=(
            permuted_idx_to_expanded_idx if use_fused_finalize else None
        ),
        token_final_scales=token_final_scales if use_fused_finalize else None,
        enable_pdl=enable_pdl,
        route_tile=route_tile,
        tactic=gemm2_tactic,
    )
    if not use_fused_finalize:
        if enable_pdl:
            # Sparse EP partitions favor a smaller CTA that can coexist with
            # GEMM2; denser reductions use all 32 warps for memory throughput.
            sparse_ep_partition = top_k * num_local_experts <= num_experts
            unpermute_num_warps = (
                _SPARSE_UNPERMUTE_NUM_WARPS
                if sparse_ep_partition
                else _DENSE_UNPERMUTE_NUM_WARPS
            )
            moe_unpermute_bf16_pdl(
                permuted_input=gemm2_output,
                output=moe_output,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                topk_scales=token_final_scales,
                num_tokens=int(x.size(0)),
                top_k=top_k,
                num_warps=unpermute_num_warps,
            )
        else:
            moe_unpermute(
                permuted_input=gemm2_output,
                output=moe_output,
                expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                topk_scales=token_final_scales,
                num_tokens=int(x.size(0)),
                top_k=top_k,
                enable_pdl=False,
            )
    return moe_output


__all__ = ["launch_w4a16_moe"]
