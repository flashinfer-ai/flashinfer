# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""SM100 NVFP4-weight, MXFP8-activation fused MoE launcher."""

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

from .moe_w4a8_kernel import Sm100W4A8GroupedGemmKernel


W4A8GemmTactic = Tuple[Tuple[int, int], bool, bool]
W4A8MoeTactic = Tuple[int, W4A8GemmTactic, W4A8GemmTactic]

DEFAULT_W4A8_MOE_TACTIC: W4A8MoeTactic = (
    128,
    ((1, 1), False, False),
    ((1, 1), False, False),
)


@dataclass
class _W4A8Workspace:
    num_tokens_capacity: int
    moe_sort_buffers: Dict[str, torch.Tensor]
    permuted_input: torch.Tensor
    permuted_input_sf: torch.Tensor
    intermediate: torch.Tensor
    intermediate_sf: torch.Tensor
    expanded_output: torch.Tensor


_kernel_cache: Dict[Tuple, object] = {}


def _get_workspace(
    x: torch.Tensor,
    top_k: int,
    num_experts: int,
    num_local_experts: int,
    intermediate_size: int,
    hidden_size: int,
    route_tile: int,
    workspace_cache: Optional[Dict[Tuple, _W4A8Workspace]] = None,
) -> _W4A8Workspace:
    num_tokens = int(x.size(0))
    route_slots = get_max_num_permuted_tokens(
        num_tokens, top_k, num_local_experts, route_tile
    )
    key = (
        x.device,
        top_k,
        hidden_size,
        intermediate_size,
        num_experts,
        num_local_experts,
        route_tile,
    )
    workspace = workspace_cache.get(key) if workspace_cache is not None else None
    if workspace is None or workspace.num_tokens_capacity < num_tokens:
        workspace = _W4A8Workspace(
            num_tokens_capacity=num_tokens,
            moe_sort_buffers=allocate_moe_sort_buffers(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=num_local_experts,
                tile_tokens_dim=route_tile,
                device=x.device,
            ),
            permuted_input=torch.empty(
                (route_slots, hidden_size),
                dtype=torch.float8_e4m3fn,
                device=x.device,
            ),
            permuted_input_sf=torch.empty(
                route_slots * (hidden_size // 32),
                dtype=torch.uint8,
                device=x.device,
            ),
            intermediate=torch.empty(
                (route_slots, intermediate_size),
                dtype=torch.float8_e4m3fn,
                device=x.device,
            ),
            intermediate_sf=torch.empty(
                route_slots * (intermediate_size // 32),
                dtype=torch.uint8,
                device=x.device,
            ),
            expanded_output=torch.empty(
                (num_tokens * top_k, hidden_size),
                dtype=torch.bfloat16,
                device=x.device,
            ),
        )
        if workspace_cache is not None:
            workspace_cache[key] = workspace
    return workspace


def _get_compiled_kernel(
    a_ptr,
    b_ptr,
    a_sf_ptr,
    b_sf_ptr,
    c_ptr,
    c_sf_ptr,
    alpha_ptr,
    tile_idx_to_expert_idx_ptr,
    tile_idx_to_mn_limit_ptr,
    token_id_mapping_ptr,
    num_non_exiting_tiles_ptr,
    permuted_idx_to_expanded_idx_ptr,
    token_final_scales_ptr,
    orig_m: int,
    m: int,
    n: int,
    k: int,
    num_local_experts: int,
    num_tokens: int,
    top_k: int,
    max_active_clusters: int,
    stream,
    cluster_shape_mn: Tuple[int, int],
    raster_along_m: bool,
    use_tma_activation: bool,
    enable_pdl: bool,
    activation_type: ActivationType,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
    gated: bool,
    fuse_activation: bool,
    use_fused_finalize: bool,
):
    key = (
        top_k,
        cluster_shape_mn,
        raster_along_m,
        use_tma_activation,
        enable_pdl,
        int(activation_type),
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        gated,
        fuse_activation,
        use_fused_finalize,
    )
    compiled = _kernel_cache.get(key)
    if compiled is None:
        kernel = Sm100W4A8GroupedGemmKernel(
            sf_vec_size=32,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=cluster_shape_mn,
            vectorized_f32=True,
            topk=top_k,
            raster_along_m=raster_along_m,
            enable_pdl=enable_pdl,
            activation_type=int(activation_type),
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            gated=gated,
            fuse_activation=fuse_activation,
            use_fused_finalize=use_fused_finalize,
            input_is_expanded=use_tma_activation or not fuse_activation,
            use_tma_activation=use_tma_activation,
        )
        compiled = cute.compile(
            kernel.wrapper,
            a_ptr,
            b_ptr,
            a_sf_ptr,
            b_sf_ptr,
            c_ptr,
            c_sf_ptr,
            alpha_ptr,
            tile_idx_to_expert_idx_ptr,
            tile_idx_to_mn_limit_ptr,
            token_id_mapping_ptr,
            num_non_exiting_tiles_ptr,
            None,
            None,
            permuted_idx_to_expanded_idx_ptr,
            token_final_scales_ptr,
            orig_m,
            m,
            n,
            k,
            num_local_experts,
            num_tokens,
            tile_size=128,
            scaling_vector_size=32,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
        _kernel_cache[key] = compiled
    return compiled


def _run_grouped_gemm(
    a: torch.Tensor,
    a_sf: torch.Tensor,
    weight: torch.Tensor,
    weight_sf: torch.Tensor,
    weight_alpha: torch.Tensor,
    output: torch.Tensor,
    output_sf: Optional[torch.Tensor],
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    token_id_mapping: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    permuted_idx_to_expanded_idx: Optional[torch.Tensor],
    token_final_scales: Optional[torch.Tensor],
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    enable_pdl: bool,
    activation_type: ActivationType,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
    gated: bool,
    fuse_activation: bool,
    use_fused_finalize: bool,
    tactic: W4A8GemmTactic,
) -> None:
    cluster_shape_mn, raster_along_m, use_tma_activation = tactic
    route_slots = int(token_id_mapping.numel())
    n = int(weight.size(1))
    k = int(a.size(1))
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    a_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        a.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    b_ptr = make_ptr(
        cutlass.Float4E2M1FN,
        weight.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    a_sf_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        a_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    b_sf_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        weight_sf.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    c_dtype = cutlass.Float8E4M3FN if fuse_activation else cutlass.BFloat16
    c_ptr = make_ptr(
        c_dtype,
        output.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    c_sf_ptr = (
        make_ptr(
            cutlass.Float8E8M0FNU,
            output_sf.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        if output_sf is not None
        else None
    )
    alpha_ptr = make_ptr(
        cutlass.Float32,
        weight_alpha.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    int_ptr = lambda tensor: make_ptr(  # noqa: E731
        cutlass.Int32,
        tensor.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    final_scale_ptr = (
        make_ptr(
            cutlass.Float32,
            token_final_scales.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=4,
        )
        if token_final_scales is not None
        else None
    )

    compiled = _get_compiled_kernel(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        c_sf_ptr,
        alpha_ptr,
        int_ptr(tile_idx_to_expert_idx),
        int_ptr(tile_idx_to_mn_limit),
        int_ptr(token_id_mapping),
        int_ptr(num_non_exiting_tiles),
        int_ptr(permuted_idx_to_expanded_idx)
        if permuted_idx_to_expanded_idx is not None
        else None,
        final_scale_ptr,
        int(a.size(0)),
        route_slots,
        n,
        k,
        num_local_experts,
        num_tokens,
        top_k,
        max_active_clusters,
        stream,
        cluster_shape_mn,
        raster_along_m,
        use_tma_activation,
        enable_pdl,
        activation_type,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        gated,
        fuse_activation,
        use_fused_finalize,
    )
    compiled(
        a_ptr,
        b_ptr,
        a_sf_ptr,
        b_sf_ptr,
        c_ptr,
        c_sf_ptr,
        alpha_ptr,
        int_ptr(tile_idx_to_expert_idx),
        int_ptr(tile_idx_to_mn_limit),
        int_ptr(token_id_mapping),
        int_ptr(num_non_exiting_tiles),
        None,
        None,
        int_ptr(permuted_idx_to_expanded_idx)
        if permuted_idx_to_expanded_idx is not None
        else None,
        final_scale_ptr,
        int(a.size(0)),
        route_slots,
        n,
        k,
        num_local_experts,
        num_tokens,
        stream=stream,
    )


def launch_w4a8_moe(
    x: torch.Tensor,
    x_sf: torch.Tensor,
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
    tactic: Optional[W4A8MoeTactic] = None,
    workspace_cache: Optional[Dict[Tuple, _W4A8Workspace]] = None,
) -> torch.Tensor:
    """Run MXFP8 activations against online-converted NVFP4 expert weights."""
    activation_type, gated = normalize_cute_dsl_moe_activation_type(activation_type)
    top_k = int(token_selected_experts.size(1))
    hidden_size = int(x.size(1))
    intermediate_size = int(w2_weight.size(2)) * 2
    if tactic is None:
        tactic = DEFAULT_W4A8_MOE_TACTIC
    route_tile, gemm1_tactic, gemm2_tactic = tactic
    workspace = _get_workspace(
        x,
        top_k,
        num_experts,
        num_local_experts,
        intermediate_size,
        hidden_size,
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
    intermediate = workspace.intermediate[:route_slots]
    gemm1_input = x
    gemm1_input_sf = x_sf
    if gemm1_tactic[2]:
        gemm1_input = workspace.permuted_input[:route_slots]
        gemm1_input_sf = workspace.permuted_input_sf
        moe_permute(
            input=x,
            permuted_output=gemm1_input,
            input_sf=x_sf,
            permuted_sf=gemm1_input_sf,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            max_num_permuted_tokens=route_slots,
            top_k=top_k,
            tile_size=route_tile,
            enable_pdl=enable_pdl,
        )

    _run_grouped_gemm(
        gemm1_input,
        gemm1_input_sf,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        intermediate,
        workspace.intermediate_sf,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        None,
        None,
        num_tokens,
        top_k,
        num_local_experts,
        enable_pdl,
        activation_type,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        gated,
        True,
        use_fused_finalize,
        gemm1_tactic,
    )

    gemm2_output = (
        moe_output
        if use_fused_finalize
        else workspace.expanded_output[: num_tokens * top_k]
    )
    if use_fused_finalize:
        moe_output_memset_inplace(moe_output)
    _run_grouped_gemm(
        intermediate,
        workspace.intermediate_sf,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        gemm2_output,
        None,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        permuted_idx_to_expanded_idx,
        token_final_scales,
        num_tokens,
        top_k,
        num_local_experts,
        enable_pdl,
        activation_type,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        gated,
        False,
        use_fused_finalize,
        gemm2_tactic,
    )
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
    return moe_output


__all__ = ["launch_w4a8_moe"]
