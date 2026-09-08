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

Host wrapper for the SM90 MoE GEMM1: gather grouped GEMM + fused gated
activation.
"""

from typing import Any, Dict, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda

from ...cute_dsl.utils import get_max_active_clusters, make_ptr
from ...utils import get_compute_capability
from .hopper.contiguous_gather_grouped_gemm_act_fusion import (
    Sm90ContiguousGatherGroupedGemmActFusionKernel,
)
from .hopper.utils import TORCH_TO_CUTLASS_DTYPE

_gather_kernel_cache: Dict[Tuple, Any] = {}


def _get_compiled_gather_kernel(
    orig_m: int,
    permuted_m: int,
    n: int,
    k: int,
    num_experts: int,
    a_ptr,
    b_ptr,
    c_ptr,
    tile_idx_ptr,
    mn_limit_ptr,
    token_id_ptr,
    num_tiles_ptr,
    max_active_clusters: int,
    stream,
    ab_dtype: type,
    c_dtype: type,
    tile_shape_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    topk: int,
    raster_along_m: bool,
    enable_pdl: bool,
) -> Any:
    """Get or compile one GEMM1 specialization.

    Problem dimensions, pointers, and the CUDA stream are runtime parameters;
    pointer dtypes and the kernel-specializing parameters (tile/cluster/raster
    tactics, ``topk``, ``max_active_clusters``, ``enable_pdl``) form the
    process-local compile key.
    """
    cache_key = (
        ab_dtype,
        c_dtype,
        tile_shape_mn,
        cluster_shape_mn,
        topk,
        raster_along_m,
        max_active_clusters,
        enable_pdl,
    )
    compiled = _gather_kernel_cache.get(cache_key)
    if compiled is not None:
        return compiled

    gemm = Sm90ContiguousGatherGroupedGemmActFusionKernel(
        acc_dtype=cutlass.Float32,
        tile_shape_mn=tile_shape_mn,
        topk=topk,
        cluster_shape_mn=cluster_shape_mn,
        raster_along_m=raster_along_m,
        enable_pdl=enable_pdl,
    )
    compiled = cute.compile(
        gemm.wrapper,
        a_ptr,
        b_ptr,
        c_ptr,
        tile_idx_ptr,
        mn_limit_ptr,
        token_id_ptr,
        num_tiles_ptr,
        orig_m,
        permuted_m,
        n,
        k,
        num_experts,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    _gather_kernel_cache[cache_key] = compiled
    return compiled


def _interleave_gated_halves(up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Interleave per-expert ``up``/``gate`` halves (each ``[E, I, K]``) at
    32-column granularity along the I dimension into the ``[E, 2I, K]``
    layout the SM90 gated GEMM1 epilogue expects:
    ``[up 0:32 | gate 0:32 | up 32:64 | ...]``."""
    e, inter, k = up.shape
    if inter % 32 != 0:
        raise ValueError(f"intermediate size {inter} must be a multiple of 32")
    # [E, inter/32, 2, 32, K]: dim2=0 -> up, dim2=1 -> gate.
    stacked = torch.stack(
        (
            up.reshape(e, inter // 32, 32, k),
            gate.reshape(e, inter // 32, 32, k),
        ),
        dim=2,
    )
    return stacked.reshape(e, 2 * inter, k).contiguous()


def interleave_up_gate_sm90(w_gate_up: torch.Tensor) -> torch.Tensor:
    """Repack ``[E, 2I, K]`` gate-first concatenated weights:
    ``[gate; up]``) into the 32-column up/gate interleave this kernel expects.

    Reference implementation, used by the in-tree tests — frameworks own
    their weight conversion and keep a local copy of this trivial reshape
    The result places each 32-column up block immediately before its matching
    gate block."""
    inter = w_gate_up.shape[1] // 2
    return _interleave_gated_halves(w_gate_up[:, inter:], w_gate_up[:, :inter])


def sm90_contiguous_gather_grouped_gemm_act_fusion(
    x: torch.Tensor,
    w1_weight: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    token_id_mapping: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    topk: int,
    permuted_m: int,
    tile_shape_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    raster_along_m: bool = False,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """MoE GEMM1 on SM90: gather + grouped GEMM + SiLU-gating, bf16/fp16.

    ``out[r, j] = silu(gate_j) * up_j`` with
    ``[up, gate] = x[token(r)] @ w1_weight[expert(r)].T`` for every valid permuted
    row ``r``; token permute is fused into the A load (never materialized).

    Args:
        x: Unpermuted activations ``[num_tokens, k]``, row-major bf16/fp16.
        w1_weight: Expert weights ``[num_local_experts, 2I, k]``, k contiguous,
            up/gate interleaved at 32 columns (see
            :func:`interleave_up_gate_sm90`).
        tile_idx_to_expert_idx / tile_idx_to_mn_limit /
        num_non_exiting_tiles: ``moe_sort`` outputs (tile size =
            ``tile_shape_mn[0]``).
        token_id_mapping: ``moe_sort``'s ``permuted_idx_to_expanded_idx``
            (``[permuted_m]`` int32; garbage on padding rows).
        out: Optional ``[permuted_m, I]`` output. Padding rows hold garbage.
        topk: MoE top-k (compile-time constant of the kernel).
        permuted_m: ``max_num_tiles * tile_m`` (padded row count).
        tile_shape_mn: CTA tile over the accumulator (N counts up+gate
            columns); ``tile_n % 64 == 0``.

    Returns:
        The output tensor ``[permuted_m, I]`` where ``I = w1_weight.shape[1] // 2``.
    """
    major, minor = get_compute_capability(x.device)
    if major != 9:
        raise ValueError(
            "sm90_contiguous_gather_grouped_gemm_act_fusion requires SM90. "
            f"Got SM{major}{minor}."
        )
    if x.dtype not in TORCH_TO_CUTLASS_DTYPE or x.dtype != w1_weight.dtype:
        raise ValueError(f"unsupported/mismatched dtypes: {x.dtype}, {w1_weight.dtype}")
    if not (x.dim() == 2 and x.is_contiguous()):
        raise ValueError("x must be a contiguous 2D [num_tokens, k] tensor")
    if not (w1_weight.dim() == 3 and w1_weight.stride(2) == 1):
        raise ValueError(
            "w1_weight must be [num_local_experts, 2I, k] with k contiguous"
        )

    orig_m, k = x.shape
    num_local_experts, n, w_k = w1_weight.shape
    if w_k != k:
        raise ValueError(f"k mismatch: x k={k}, w1_weight k={w_k}")
    if n % 64 != 0:
        raise ValueError(f"w1_weight 2I dim ({n}) must be a multiple of 64")
    # The epilogue writes full N tiles and the gather/mainloop assume whole
    # K tiles — partial tiles would read/write out of bounds.
    if n % tile_shape_mn[1] != 0:
        raise ValueError(f"n={n} must be a multiple of tile_n={tile_shape_mn[1]}")
    if k % 64 != 0:
        raise ValueError(f"k={k} must be a multiple of the K tile (64)")
    tile_m = tile_shape_mn[0]
    if permuted_m % tile_m != 0:
        raise ValueError(
            f"permuted_m={permuted_m} must be a multiple of tile_m={tile_m}"
        )
    if token_id_mapping.numel() != permuted_m:
        raise ValueError(
            f"token_id_mapping has {token_id_mapping.numel()} entries, "
            f"expected permuted_m={permuted_m}"
        )

    ab_dtype = TORCH_TO_CUTLASS_DTYPE[x.dtype]
    if out is None:
        out = torch.empty(permuted_m, n // 2, dtype=x.dtype, device=x.device)
    elif out.shape != (permuted_m, n // 2):
        raise ValueError(f"out shape {tuple(out.shape)} != ({permuted_m}, {n // 2})")
    elif not out.is_contiguous() or out.device != x.device:
        raise ValueError("out must be a contiguous tensor on x's device")
    c_dtype = TORCH_TO_CUTLASS_DTYPE[out.dtype]

    if cluster_shape_mn == (2, 1) and (permuted_m // tile_shape_mn[0]) % 2 != 0:
        cluster_shape_mn = (1, 1)  # odd M-tile count cannot pair
    max_active_clusters = get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    a_ptr = make_ptr(ab_dtype, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(
        ab_dtype, w1_weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(c_dtype, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
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
        token_id_mapping.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    num_tiles_ptr = make_ptr(
        cutlass.Int32,
        num_non_exiting_tiles.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)

    compiled = _get_compiled_gather_kernel(
        orig_m=orig_m,
        permuted_m=permuted_m,
        n=n,
        k=k,
        num_experts=num_local_experts,
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        c_ptr=c_ptr,
        tile_idx_ptr=tile_idx_ptr,
        mn_limit_ptr=mn_limit_ptr,
        token_id_ptr=token_id_ptr,
        num_tiles_ptr=num_tiles_ptr,
        max_active_clusters=max_active_clusters,
        stream=stream,
        ab_dtype=ab_dtype,
        c_dtype=c_dtype,
        tile_shape_mn=tile_shape_mn,
        cluster_shape_mn=cluster_shape_mn,
        topk=topk,
        raster_along_m=raster_along_m,
        enable_pdl=enable_pdl,
    )

    compiled(
        a_ptr,
        b_ptr,
        c_ptr,
        tile_idx_ptr,
        mn_limit_ptr,
        token_id_ptr,
        num_tiles_ptr,
        orig_m,
        permuted_m,
        n,
        k,
        num_local_experts,
        stream=stream,
    )
    return out
