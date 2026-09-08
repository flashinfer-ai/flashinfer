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

Host wrapper for the SM90 MoE GEMM2: contiguous grouped GEMM + fused
finalize.
"""

from typing import Any, Dict, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from ...cute_dsl.utils import get_max_active_clusters, make_ptr
from ...utils import get_compute_capability
from .hopper.contiguous_grouped_gemm_finalize_fusion import (
    Sm90ContiguousGroupedGemmFinalizeFusionKernel,
)
from .hopper.utils import TORCH_TO_CUTLASS_DTYPE

_finalize_kernel_cache: Dict[Tuple, Any] = {}


def _get_compiled_finalize_kernel(
    permuted_m: int,
    n: int,
    k: int,
    num_experts: int,
    num_tokens: int,
    out_rows: int,
    a_ptr,
    b_ptr,
    out_ptr,
    tile_idx_ptr,
    mn_limit_ptr,
    token_id_ptr,
    num_tiles_ptr,
    token_scales_ptr,
    max_active_clusters: int,
    stream,
    ab_dtype: type,
    c_dtype: type,
    tile_shape_mn: Tuple[int, int],
    tile_k: int,
    cluster_shape_mn: Tuple[int, int],
    topk: int,
    use_fused_finalize: bool,
    raster_along_m: bool,
    enable_pdl: bool,
) -> Any:
    """Get or compile one GEMM2 specialization.

    Problem dimensions, pointers, and the CUDA stream are runtime parameters;
    pointer dtypes and the kernel-specializing parameters (tile/cluster/raster
    tactics, ``topk``, ``use_fused_finalize``, ``max_active_clusters``,
    ``enable_pdl``) form the process-local compile key.
    """
    cache_key = (
        ab_dtype,
        c_dtype,
        tile_shape_mn,
        tile_k,
        cluster_shape_mn,
        topk,
        use_fused_finalize,
        raster_along_m,
        max_active_clusters,
        enable_pdl,
    )
    compiled = _finalize_kernel_cache.get(cache_key)
    if compiled is not None:
        return compiled

    gemm = Sm90ContiguousGroupedGemmFinalizeFusionKernel(
        acc_dtype=cutlass.Float32,
        tile_shape_mn=tile_shape_mn,
        topk=topk,
        use_fused_finalize=use_fused_finalize,
        tile_k=tile_k,
        cluster_shape_mn=cluster_shape_mn,
        raster_along_m=raster_along_m,
        enable_pdl=enable_pdl,
    )
    compiled = cute.compile(
        gemm.wrapper,
        a_ptr,
        b_ptr,
        out_ptr,
        tile_idx_ptr,
        mn_limit_ptr,
        token_id_ptr,
        num_tiles_ptr,
        token_scales_ptr,
        permuted_m,
        n,
        k,
        num_experts,
        num_tokens,
        out_rows,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    _finalize_kernel_cache[cache_key] = compiled
    return compiled


def sm90_contiguous_grouped_gemm_finalize_fusion(
    a: torch.Tensor,
    w2_weight: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    token_final_scales: torch.Tensor,
    out: torch.Tensor,
    *,
    topk: int,
    use_fused_finalize: bool = True,
    tile_shape_mn: Tuple[int, int] = (128, 128),
    tile_k: int = 64,
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    raster_along_m: bool = False,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """MoE GEMM2 on SM90 with fused finalize.

    Fused mode: ``out[token, :] += router_scale * (a[row] @ w2_weight[e].T)``;
    the caller must pass a zero-initialized ``out [num_tokens, hidden]``.
    Deterministic mode scatters to ``out [num_tokens * topk, hidden]`` and
    leaves the routing-weight reduction to the host pipeline.
    """
    major, minor = get_compute_capability(a.device)
    if major != 9:
        raise ValueError(
            "sm90_contiguous_grouped_gemm_finalize_fusion requires SM90. "
            f"Got SM{major}{minor}."
        )
    if a.dtype not in TORCH_TO_CUTLASS_DTYPE or a.dtype != w2_weight.dtype:
        raise ValueError(f"unsupported/mismatched dtypes: {a.dtype}, {w2_weight.dtype}")
    if token_final_scales.dtype != torch.float32:
        raise ValueError("token_final_scales must be float32")

    permuted_m, k = a.shape
    num_local_experts, n, w_k = w2_weight.shape
    if w_k != k:
        raise ValueError(f"k mismatch: a k={k}, w2_weight k={w_k}")
    tile_size = tile_shape_mn[0]
    if permuted_m % tile_size != 0:
        raise ValueError(f"permuted_m={permuted_m} not a multiple of {tile_size}")
    if n % tile_shape_mn[1] != 0:
        raise ValueError(f"n={n} must be a multiple of tile_n={tile_shape_mn[1]}")
    num_tokens = token_final_scales.shape[0]
    out_rows = out.shape[0]
    expected_rows = num_tokens if use_fused_finalize else num_tokens * topk
    if out_rows != expected_rows or out.shape[1] != n:
        raise ValueError(f"out shape {tuple(out.shape)} != ({expected_rows}, {n})")

    if cluster_shape_mn == (1, 2) and (n // tile_shape_mn[1]) % 2 != 0:
        cluster_shape_mn = (1, 1)
    if k % tile_k != 0:
        raise ValueError(f"k={k} must be a multiple of tile_k={tile_k}")

    ab_dtype = TORCH_TO_CUTLASS_DTYPE[a.dtype]
    c_dtype = TORCH_TO_CUTLASS_DTYPE[out.dtype]
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )
    a_ptr = make_ptr(ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(
        ab_dtype, w2_weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    out_ptr = make_ptr(
        c_dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    tile_idx_ptr = make_ptr(
        cutlass.Int32,
        tile_idx_to_expert_idx.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    mn_limit_ptr = make_ptr(
        cutlass.Int32,
        tile_idx_to_mn_limit.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    token_id_ptr = make_ptr(
        cutlass.Int32,
        permuted_idx_to_expanded_idx.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    num_tiles_ptr = make_ptr(
        cutlass.Int32,
        num_non_exiting_tiles.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    token_scales_ptr = make_ptr(
        cutlass.Float32,
        token_final_scales.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream)

    compiled = _get_compiled_finalize_kernel(
        permuted_m=permuted_m,
        n=n,
        k=k,
        num_experts=num_local_experts,
        num_tokens=num_tokens,
        out_rows=out_rows,
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        out_ptr=out_ptr,
        tile_idx_ptr=tile_idx_ptr,
        mn_limit_ptr=mn_limit_ptr,
        token_id_ptr=token_id_ptr,
        num_tiles_ptr=num_tiles_ptr,
        token_scales_ptr=token_scales_ptr,
        max_active_clusters=max_active_clusters,
        stream=stream,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        tile_shape_mn=tile_shape_mn,
        tile_k=tile_k,
        cluster_shape_mn=cluster_shape_mn,
        topk=topk,
        use_fused_finalize=use_fused_finalize,
        raster_along_m=raster_along_m,
        enable_pdl=enable_pdl,
    )
    compiled(
        a_ptr,
        b_ptr,
        out_ptr,
        tile_idx_ptr,
        mn_limit_ptr,
        token_id_ptr,
        num_tiles_ptr,
        token_scales_ptr,
        permuted_m,
        n,
        k,
        num_local_experts,
        num_tokens,
        out_rows,
        stream=stream,
    )
    return out
