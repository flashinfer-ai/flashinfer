# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""SM100 NVFP4-weight, BF16-activation fused MoE launcher."""

from dataclasses import dataclass
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import get_max_active_clusters, make_ptr
from flashinfer.fused_moe.cute_dsl.moe_utils import (
    allocate_moe_sort_buffers,
    get_max_num_permuted_tokens,
    moe_permute,
    moe_sort,
    moe_swiglu,
    moe_unpermute,
)

from .moe_w4a16_kernel import Sm100W4A16GroupedGemmKernel


_ROUTE_TILE = 128
_MMA_TILER_MNK = (256, 128, 128)
_CLUSTER_SHAPE_MN = (2, 1)


@dataclass
class _W4A16Workspace:
    moe_sort_buffers: Dict[str, torch.Tensor]
    gathered_activations: torch.Tensor
    gemm1_output: torch.Tensor
    intermediate: torch.Tensor
    gemm2_output: torch.Tensor


_workspace_cache: Dict[Tuple, _W4A16Workspace] = {}
_kernel_cache: Dict[Tuple[int, bool], object] = {}
_linear_scale_cache: Dict[Tuple, torch.Tensor] = {}


def _get_workspace(
    x: torch.Tensor,
    top_k: int,
    num_experts: int,
    intermediate_size: int,
) -> _W4A16Workspace:
    num_tokens = int(x.size(0))
    route_slots = get_max_num_permuted_tokens(
        num_tokens, top_k, num_experts, _ROUTE_TILE
    )
    key = (
        x.device,
        route_slots,
        int(x.size(1)),
        int(intermediate_size),
        int(num_experts),
    )
    workspace = _workspace_cache.get(key)
    if workspace is None:
        workspace = _W4A16Workspace(
            moe_sort_buffers=allocate_moe_sort_buffers(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                num_local_experts=num_experts,
                tile_tokens_dim=_ROUTE_TILE,
                device=x.device,
            ),
            gathered_activations=torch.empty(
                (route_slots, x.size(1)), dtype=torch.bfloat16, device=x.device
            ),
            gemm1_output=torch.empty(
                (route_slots, 2 * intermediate_size),
                dtype=torch.bfloat16,
                device=x.device,
            ),
            intermediate=torch.empty(
                (route_slots, intermediate_size),
                dtype=torch.bfloat16,
                device=x.device,
            ),
            gemm2_output=torch.empty(
                (route_slots, x.size(1)), dtype=torch.bfloat16, device=x.device
            ),
        )
        _workspace_cache[key] = workspace
    return workspace


def _linearize_weight_sf(
    weight_sf: torch.Tensor, rows: int, k: int, num_experts: int
) -> torch.Tensor:
    key = (weight_sf.data_ptr(), tuple(weight_sf.shape), rows, k, num_experts)
    linear = _linear_scale_cache.get(key)
    if linear is not None:
        return linear
    expected_shape = (
        32,
        4,
        (rows + 127) // 128,
        4,
        ((k + 15) // 16 + 3) // 4,
        num_experts,
    )
    if tuple(weight_sf.shape) != expected_shape:
        raise ValueError(
            f"weight scales must have MMA layout {expected_shape}, "
            f"got {tuple(weight_sf.shape)}"
        )
    linear = (
        weight_sf.permute(5, 2, 1, 0, 4, 3)
        .contiguous()
        .reshape(num_experts, expected_shape[2] * 128, expected_shape[4] * 4)
    )
    linear = linear[:, :rows, : k // 16].transpose(1, 2).contiguous()
    _linear_scale_cache[key] = linear
    return linear


def _get_compiled_kernel(
    num_experts: int,
    weight_ptr,
    weight_sf_ptr,
    activation_ptr,
    tile_idx_to_expert_idx_ptr,
    tile_idx_to_mn_limit_ptr,
    num_non_exiting_tiles_ptr,
    alpha_ptr,
    output_ptr,
    m: int,
    n: int,
    k: int,
    max_active_clusters: int,
    stream,
    deinterleave_output: bool,
):
    cache_key = (num_experts, deinterleave_output)
    compiled = _kernel_cache.get(cache_key)
    if compiled is None:
        kernel = Sm100W4A16GroupedGemmKernel(
            scale_granularity_m=1,
            scale_granularity_k=16,
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=True,
            mma_tiler_mnk=_MMA_TILER_MNK,
            cluster_shape_mn=_CLUSTER_SHAPE_MN,
            group_count=num_experts,
            shuffle_a=False,
            deinterleave_output=deinterleave_output,
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
            m,
            n,
            k,
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
    num_experts: int,
    deinterleave_output: bool,
) -> None:
    m = int(weight.size(1))
    k = int(weight.size(2)) * 2
    n = int(activations.size(0))
    weight_sf = _linearize_weight_sf(weight_sf, m, k, num_experts)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = get_max_active_clusters(2)
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
    compiled = _get_compiled_kernel(
        num_experts,
        weight_ptr,
        weight_sf_ptr,
        activation_ptr,
        tile_idx_to_expert_idx_ptr,
        tile_idx_to_mn_limit_ptr,
        num_non_exiting_tiles_ptr,
        alpha_ptr,
        output_ptr,
        m,
        n,
        k,
        max_active_clusters,
        stream,
        deinterleave_output,
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
        m,
        n,
        k,
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
    moe_output: torch.Tensor,
) -> torch.Tensor:
    """Run BF16 activations against online-decoded NVFP4 expert weights."""
    num_experts = int(w1_weight.size(0))
    top_k = int(token_selected_experts.size(1))
    intermediate_size = int(w2_weight.size(2)) * 2
    workspace = _get_workspace(x, top_k, num_experts, intermediate_size)

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
        num_local_experts=num_experts,
        tile_tokens_dim=_ROUTE_TILE,
        **workspace.moe_sort_buffers,
    )
    route_slots = int(permuted_idx_to_expanded_idx.numel())
    moe_permute(
        input=x,
        permuted_output=workspace.gathered_activations,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
        num_non_exiting_tiles=num_non_exiting_tiles,
        max_num_permuted_tokens=route_slots,
        top_k=top_k,
        tile_size=_ROUTE_TILE,
    )
    _run_grouped_gemm(
        w1_weight,
        w1_weight_sf,
        workspace.gathered_activations,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        num_non_exiting_tiles,
        w1_alpha,
        workspace.gemm1_output,
        num_experts,
        True,
    )
    moe_swiglu(
        input=workspace.gemm1_output,
        output=workspace.intermediate,
        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
        num_non_exiting_tiles=num_non_exiting_tiles,
        max_num_permuted_tokens=route_slots,
        tile_size=_ROUTE_TILE,
    )
    _run_grouped_gemm(
        w2_weight,
        w2_weight_sf,
        workspace.intermediate,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        num_non_exiting_tiles,
        w2_alpha,
        workspace.gemm2_output,
        num_experts,
        False,
    )
    moe_unpermute(
        permuted_input=workspace.gemm2_output,
        output=moe_output,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        topk_scales=token_final_scales,
        num_tokens=int(x.size(0)),
        top_k=top_k,
    )
    return moe_output


__all__ = ["launch_w4a16_moe"]
