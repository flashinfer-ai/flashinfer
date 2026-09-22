# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Swap-AB execution path for few-rows-per-expert MXFP4 x MXFP8 MoE.

Routes with a small number of rows per expert (decode, small prefill, MoE-TP
shards) run the two grouped GEMMs with the expert weights as the MMA-M operand
and 32-row groups of permuted activations as the MMA-N operand
(:mod:`.blackwell.blockscaled_swapab_grouped_gemm`). This module owns the host
side: row-group capacity, the permuted activation staging buffers, the compile
cache and the launch sequence.
"""

from typing import Any, Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import get_max_active_clusters, make_ptr

from .blackwell.blockscaled_swapab_grouped_gemm import (
    Sm100BlockScaledSwapAbGroupedGemmKernel,
)
from .moe_utils import get_max_num_tiles, moe_sort

import os as _os0

# 32-row groups by default; SWAPAB_NTILE=64/128 for experiments.
SWAP_ROW_TILE = int(_os0.environ.get("SWAPAB_NTILE", "32"))
import os as _os

# 4 K-blocks (128 K elements) per stage matches the reference kernels' SF
# shared-to-TMEM copy geometry; SWAPAB_KBLOCKS overrides for experiments.
SWAP_K_BLOCKS_PER_STAGE = int(_os.environ.get("SWAPAB_KBLOCKS", "4"))

_swapab_kernel_cache: Dict[Tuple, Any] = {}


def swap_row_capacity(
    num_tokens: int, top_k: int, num_local_experts: int, tile: int = SWAP_ROW_TILE
) -> Tuple[int, int]:
    """Return ``(row_groups, rows)`` for the swap path.

    ``rows`` is a multiple of 128 so the permuted activations and the GEMM1
    output share the 128-row scale-factor atom layout.
    """
    groups = get_max_num_tiles(num_tokens, top_k, num_local_experts, tile)
    rows = -(-(groups * tile) // 128) * 128
    return groups, rows


def sf_linear_to_atom_physical(sf_linear: torch.Tensor) -> torch.Tensor:
    """Linear ``[R, K/32]`` UE8M0 bytes -> physical ``[R/128, K/128, 32, 4, 4]``."""
    rows, blocks = sf_linear.shape
    return (
        sf_linear.view(rows // 128, 4, 32, blocks // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
    )


def sf_linear_to_group_tiled(
    sf_linear: torch.Tensor, tile: int = SWAP_ROW_TILE
) -> torch.Tensor:
    """Linear ``[R, K/32]`` -> group-tiled physical ``[R/tile, K/128, 32, 4, 4]``.

    Every ``tile``-row group owns a full 128-row SF atom tile in which its rows
    are replicated ``128 // tile`` times.  The swap-AB kernel loads one such tile
    per row group by TMA and reads it at the unshifted TMEM base, which is the
    only addressable option for 32-row groups (SF TMEM columns are 32 rows wide
    and a 1-column base shift is illegal for the MMA).
    """
    rows, blocks = sf_linear.shape
    groups = rows // tile
    fake = sf_linear.view(groups, tile, blocks).repeat(1, 128 // tile, 1)
    return sf_linear_to_atom_physical(fake.reshape(groups * 128, blocks))


def group_tiled_to_linear(
    physical: torch.Tensor, rows: int, tile: int = SWAP_ROW_TILE
) -> torch.Tensor:
    """Inverse of :func:`sf_linear_to_group_tiled` (replica 0 of every group)."""
    groups, k_tiles = physical.shape[:2]
    lin = physical.permute(0, 3, 2, 1, 4).reshape(groups * 128, k_tiles * 4)
    return lin.view(groups, 128 // tile, tile, k_tiles * 4)[:, 0].reshape(
        groups * tile, -1
    )[:rows]


def atom_physical_to_mma_view(physical: torch.Tensor) -> torch.Tensor:
    """Physical ``[R/128, K/128, 32, 4, 4]`` -> logical ``(32, 4, R/128, 4, K/128, 1)``."""
    m_tiles, k_tiles = physical.shape[:2]
    return physical.view(1, m_tiles, k_tiles, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)


def permute_rows_torch(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    top_k: int,
    out_x: torch.Tensor,
    out_sf_physical: torch.Tensor,
) -> None:
    """Materialize permuted activation rows (reference implementation).

    Padding rows (negative or out-of-range expanded index) are zero-filled.
    """
    rows = out_x.shape[0]
    num_tokens = x.shape[0]
    idx = permuted_idx_to_expanded_idx[:rows].to(torch.int64)
    valid = (idx >= 0) & (idx < num_tokens * top_k)
    tok = torch.where(valid, idx // top_k, torch.zeros_like(idx))
    gathered = x[tok].view(torch.uint8)
    out_x.view(torch.uint8).copy_(
        torch.where(valid[:, None], gathered, torch.zeros_like(gathered))
    )
    sf = x_sf[tok]
    sf = torch.where(valid[:, None], sf, torch.zeros_like(sf))
    out_sf_physical.copy_(sf_linear_to_group_tiled(sf))


def _gmem_ptr(dtype, tensor: Optional[torch.Tensor], align: int = 16):
    if tensor is None:
        return None
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


class PermuteRowsKernel:
    """Gather permuted activation rows and write scales in the SF atom layout.

    One CTA per permuted row: 256 threads copy the row as 8-byte words, the
    first ``H/32`` threads place the UE8M0 bytes at their atom-layout position.
    Padding rows (negative / out-of-range expanded index) are zero-filled.
    """

    @cute.jit
    def wrapper(
        self,
        x_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        pidx_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        out_sf_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        top_k: cutlass.Int32,
        hidden_words: cutlass.Int32,
        sf_blocks: cutlass.Int32,
        rows: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        x = cute.make_tensor(
            x_ptr,
            layout=cute.make_ordered_layout((num_tokens, hidden_words), order=(1, 0)),
        )
        out = cute.make_tensor(
            out_ptr, layout=cute.make_ordered_layout((rows, hidden_words), order=(1, 0))
        )
        sf = cute.make_tensor(
            sf_ptr,
            layout=cute.make_ordered_layout((num_tokens, sf_blocks), order=(1, 0)),
        )
        out_sf = cute.make_tensor(
            out_sf_ptr,
            layout=cute.make_layout((rows // SWAP_ROW_TILE * 128 * sf_blocks,)),
        )
        pidx = cute.make_tensor(pidx_ptr, layout=cute.make_layout((rows,)))
        self.kernel(x, sf, pidx, out, out_sf, num_tokens, top_k, sf_blocks).launch(
            grid=[rows, 1, 1], block=[256, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        sf: cute.Tensor,
        pidx: cute.Tensor,
        out: cute.Tensor,
        out_sf: cute.Tensor,
        num_tokens: cutlass.Int32,
        top_k: cutlass.Int32,
        sf_blocks: cutlass.Int32,
    ):
        r, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        idx = pidx[r]
        valid = (idx >= 0) & (idx < num_tokens * top_k)
        tok = cutlass.Int32(0)
        if valid:
            tok = idx // top_k
        hidden_words = x.shape[1]
        for c in cutlass.range(tidx, hidden_words, 256):
            v = cutlass.Int64(0)
            if valid:
                v = cutlass.Int64(x[(tok, c)])
            out[(r, c)] = v
        if tidx < sf_blocks:
            s = cutlass.Uint8(0)
            if valid:
                s = cutlass.Uint8(sf[(tok, tidx)])
            kt = tidx // 4
            ik = tidx % 4
            # Group-tiled SF layout: replica `rep` of this row inside the
            # group's 128-row fake tile (see sf_linear_to_group_tiled).
            g = r // SWAP_ROW_TILE
            for rep in cutlass.range_constexpr(128 // SWAP_ROW_TILE):
                row = rep * SWAP_ROW_TILE + (r % SWAP_ROW_TILE)
                im = row // 32
                om = row % 32
                off = ((g * (sf_blocks // 4) + kt) * 32 + om) * 16 + im * 4 + ik
                out_sf[off] = s


_permute_kernel_cache: Dict[str, Any] = {}


def permute_rows(
    x: torch.Tensor,
    x_sf: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    top_k: int,
    out_x: torch.Tensor,
    out_sf_physical: torch.Tensor,
    _prepared_launches: Optional[Dict[str, Any]] = None,
) -> None:
    """Device permute-copy (see :class:`PermuteRowsKernel`)."""
    rows, hidden = out_x.shape
    if hidden % 8:
        raise ValueError("hidden size must be a multiple of 8 bytes")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        _gmem_ptr(cutlass.Int64, x, 8),
        _gmem_ptr(cutlass.Uint8, x_sf, 4),
        _gmem_ptr(cutlass.Int32, permuted_idx_to_expanded_idx, 4),
        _gmem_ptr(cutlass.Int64, out_x, 8),
        _gmem_ptr(cutlass.Uint8, out_sf_physical, 16),
        x.shape[0],
        top_k,
        hidden // 8,
        hidden // 32,
        rows,
    )
    if "permute" not in _permute_kernel_cache:
        _permute_kernel_cache["permute"] = cute.compile(
            PermuteRowsKernel().wrapper, *args, stream=stream
        )
    compiled = _permute_kernel_cache["permute"]
    if _prepared_launches is not None:
        _prepared_launches["swap_permute"] = (compiled, args)
    compiled(*args, stream=stream)


def _get_compiled_swapab_kernel(
    *,
    epilogue_kind: str,
    n_tile: int,
    k_blocks_per_stage: int,
    top_k: int,
    beta_count: int,
    linear_beta_count: int,
    use_linear_beta: bool,
    enable_pdl: bool,
    compile_args: Tuple,
    max_active_clusters: int,
    stream: cuda.CUstream,
):
    import os
    import sys

    key = (
        epilogue_kind,
        n_tile,
        k_blocks_per_stage,
        top_k,
        beta_count,
        linear_beta_count,
        use_linear_beta,
        enable_pdl,
    )
    if key not in _swapab_kernel_cache:
        if os.environ.get("SWAPAB_DEBUG"):
            print(f"[swapab] compile {key}", file=sys.stderr, flush=True)
        kernel = Sm100BlockScaledSwapAbGroupedGemmKernel(
            sf_vec_size=32,
            n_tile=n_tile,
            k_blocks_per_stage=k_blocks_per_stage,
            epilogue_kind=epilogue_kind,
            enable_pdl=enable_pdl,
            use_linear_beta=use_linear_beta,
        )
        _swapab_kernel_cache[key] = cute.compile(
            kernel.wrapper,
            *compile_args,
            top_k=top_k,
            beta_count=beta_count,
            linear_beta_count=linear_beta_count,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )
        if os.environ.get("SWAPAB_DEBUG"):
            print(f"[swapab] compiled {key}", file=sys.stderr, flush=True)
    return _swapab_kernel_cache[key]


def swapab_gemm1_situ(
    *,
    w1: torch.Tensor,
    w1_sf: torch.Tensor,
    x_perm: torch.Tensor,
    x_perm_sf: torch.Tensor,
    act: torch.Tensor,
    act_sf: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    linear_beta: Optional[torch.Tensor],
    n_tile: int = SWAP_ROW_TILE,
    k_blocks_per_stage: int = SWAP_K_BLOCKS_PER_STAGE,
    enable_pdl: bool = False,
    _prepared_launches: Optional[Dict[str, Any]] = None,
) -> None:
    """GEMM1 (up/gate) + SiTU + MXFP8 requantization on the swap path.

    ``w1`` is the prepared ``[L, 2I, H/2]`` interleaved weight, ``x_perm`` the
    permuted rows ``[R, H]`` (E4M3) with ``x_perm_sf`` in the MMA atom layout,
    ``act`` the ``[R, I]`` E4M3 output and ``act_sf`` its atom-layout scales.
    """
    num_local_experts, rows_w, packed_k = w1.shape
    k = packed_k * 2
    rows_b = x_perm.shape[0]
    intermediate = rows_w // 2
    if act.shape != (rows_b, intermediate):
        raise ValueError(
            f"act must be [{rows_b}, {intermediate}], got {tuple(act.shape)}"
        )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = get_max_active_clusters(1)
    use_linear_beta = linear_beta is not None
    args = (
        _gmem_ptr(cutlass.Float4E2M1FN, w1, 32),
        _gmem_ptr(cutlass.Float8E4M3FN, x_perm, 32),
        _gmem_ptr(cutlass.Float8E8M0FNU, w1_sf, 16),
        _gmem_ptr(cutlass.Float8E8M0FNU, x_perm_sf, 16),
        _gmem_ptr(cutlass.Float8E4M3FN, act, 32),
        _gmem_ptr(cutlass.Uint8, act_sf, 16),
        _gmem_ptr(cutlass.Int32, tile_idx_to_expert_idx, 4),
        _gmem_ptr(cutlass.Int32, tile_idx_to_mn_limit, 4),
        _gmem_ptr(cutlass.Int32, num_non_exiting_tiles, 4),
        _gmem_ptr(cutlass.Float32, alpha, 4),
        None,
        None,
        _gmem_ptr(cutlass.Float32, beta, 4),
        _gmem_ptr(cutlass.Float32, linear_beta, 4),
        rows_w,
        k,
        num_local_experts,
        rows_b,
        rows_b,
        intermediate,
        1,
        tile_idx_to_expert_idx.shape[0],
    )
    compiled = _get_compiled_swapab_kernel(
        epilogue_kind="situ_mxfp8",
        n_tile=n_tile,
        k_blocks_per_stage=k_blocks_per_stage,
        top_k=1,
        beta_count=beta.numel(),
        linear_beta_count=linear_beta.numel() if use_linear_beta else 1,
        use_linear_beta=use_linear_beta,
        enable_pdl=enable_pdl,
        compile_args=args,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    if _prepared_launches is not None:
        _prepared_launches["swap_gemm1"] = (compiled, args)
    compiled(*args, stream=stream)


def swapab_gemm2(
    *,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    act: torch.Tensor,
    act_sf: torch.Tensor,
    out: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    alpha: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    top_k: int,
    finalize: bool = True,
    n_tile: int = SWAP_ROW_TILE,
    k_blocks_per_stage: int = SWAP_K_BLOCKS_PER_STAGE,
    enable_pdl: bool = False,
    _prepared_launches: Optional[Dict[str, Any]] = None,
) -> None:
    """GEMM2 (down) on the swap path.

    ``finalize=True`` reduce-adds ``alpha * route_weight * acc`` into the
    zero-initialised ``out[T, H]``; ``finalize=False`` writes ``alpha * acc``
    to ``out[T*top_k, H]`` rows indexed by expanded index.
    """
    num_local_experts, rows_w, packed_k = w2.shape
    k = packed_k * 2
    rows_b = act.shape[0]
    if act.shape[1] != k:
        raise ValueError("act columns must equal the W2 K dimension")
    if finalize:
        if token_final_scales is None or token_final_scales.dtype != torch.float32:
            raise ValueError("finalize requires float32 token_final_scales")
        num_tokens = token_final_scales.shape[0]
        if out.shape != (num_tokens, rows_w):
            raise ValueError("finalize output must be [T, H]")
    else:
        num_tokens = out.shape[0] // top_k
        if out.shape != (num_tokens * top_k, rows_w):
            raise ValueError("partial output must be [T*top_k, H]")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = get_max_active_clusters(1)
    args = (
        _gmem_ptr(cutlass.Float4E2M1FN, w2, 32),
        _gmem_ptr(cutlass.Float8E4M3FN, act, 32),
        _gmem_ptr(cutlass.Float8E8M0FNU, w2_sf, 16),
        _gmem_ptr(cutlass.Float8E8M0FNU, act_sf, 16),
        _gmem_ptr(cutlass.BFloat16, out, 16),
        None,
        _gmem_ptr(cutlass.Int32, tile_idx_to_expert_idx, 4),
        _gmem_ptr(cutlass.Int32, tile_idx_to_mn_limit, 4),
        _gmem_ptr(cutlass.Int32, num_non_exiting_tiles, 4),
        _gmem_ptr(cutlass.Float32, alpha, 4),
        _gmem_ptr(cutlass.Int32, permuted_idx_to_expanded_idx, 4),
        _gmem_ptr(cutlass.Float32, token_final_scales, 4) if finalize else None,
        None,
        None,
        rows_w,
        k,
        num_local_experts,
        rows_b,
        out.shape[0],
        rows_w,
        num_tokens,
        tile_idx_to_expert_idx.shape[0],
    )
    compiled = _get_compiled_swapab_kernel(
        epilogue_kind="finalize" if finalize else "partial",
        n_tile=n_tile,
        k_blocks_per_stage=k_blocks_per_stage,
        top_k=top_k,
        beta_count=1,
        linear_beta_count=1,
        use_linear_beta=False,
        enable_pdl=enable_pdl,
        compile_args=args,
        max_active_clusters=max_active_clusters,
        stream=stream,
    )
    if _prepared_launches is not None:
        _prepared_launches["swap_gemm2"] = (compiled, args)
    compiled(*args, stream=stream)


class SwapAbBuffers:
    """Workspace tensors for one ``(num_tokens, top_k, num_local_experts)``."""

    def __init__(
        self,
        *,
        num_tokens: int,
        top_k: int,
        num_local_experts: int,
        hidden: int,
        intermediate: int,
        device,
    ):
        self.groups, self.rows = swap_row_capacity(num_tokens, top_k, num_local_experts)
        r = self.rows
        self.x_perm = torch.empty((r, hidden), dtype=torch.float8_e4m3fn, device=device)
        # Group-tiled SF storage: one 128-row SF tile per row group.
        sf_tiles = r // SWAP_ROW_TILE
        self.x_perm_sf_physical = torch.empty(
            (sf_tiles, hidden // 128, 32, 4, 4), dtype=torch.uint8, device=device
        )
        self.act = torch.empty(
            (r, intermediate), dtype=torch.float8_e4m3fn, device=device
        )
        self.act_sf_physical = torch.empty(
            (sf_tiles, intermediate // 128, 32, 4, 4), dtype=torch.uint8, device=device
        )
        self.tile_idx_to_expert_idx = torch.empty(
            (self.groups,), dtype=torch.int32, device=device
        )
        self.tile_idx_to_mn_limit = torch.empty(
            (self.groups,), dtype=torch.int32, device=device
        )
        self.expanded_idx_to_permuted_idx = torch.empty(
            (num_tokens, top_k), dtype=torch.int32, device=device
        )
        self.permuted_idx_to_expanded_idx = torch.empty(
            (r,), dtype=torch.int32, device=device
        )
        self.total_num_padded_tokens = torch.empty(
            (1,), dtype=torch.int32, device=device
        )
        self.num_non_exiting_tiles = torch.empty((1,), dtype=torch.int32, device=device)
        self.expert_counts = torch.empty((2 * 4096,), dtype=torch.int32, device=device)

    def sort_kwargs(self) -> Dict[str, torch.Tensor]:
        return dict(
            out_tile_idx_to_expert_idx=self.tile_idx_to_expert_idx,
            out_tile_idx_to_mn_limit=self.tile_idx_to_mn_limit,
            out_expanded_idx_to_permuted_idx=self.expanded_idx_to_permuted_idx,
            out_permuted_idx_to_expanded_idx=self.permuted_idx_to_expanded_idx,
            out_total_num_padded_tokens=self.total_num_padded_tokens,
            out_num_non_exiting_tiles=self.num_non_exiting_tiles,
            out_expert_counts=self.expert_counts,
        )


def swapab_moe_forward(
    *,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    route_ids: torch.Tensor,
    route_weights: torch.Tensor,
    w1: torch.Tensor,
    w1_sf: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    w2_alpha: torch.Tensor,
    beta: torch.Tensor,
    linear_beta: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    local_expert_offset: int,
    output: torch.Tensor,
    buffers: SwapAbBuffers,
    finalize: bool = True,
    torch_permute: bool = False,
) -> torch.Tensor:
    """Full swap-path MoE: sort (32-row groups) -> permute -> GEMM1 -> GEMM2."""
    moe_sort(
        token_selected_experts=route_ids,
        token_final_scales=route_weights,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_tokens_dim=SWAP_ROW_TILE,
        **buffers.sort_kwargs(),
    )
    (permute_rows_torch if torch_permute else permute_rows)(
        x,
        x_sf,
        buffers.permuted_idx_to_expanded_idx,
        top_k,
        buffers.x_perm,
        buffers.x_perm_sf_physical,
    )
    swapab_gemm1_situ(
        w1=w1,
        w1_sf=w1_sf,
        x_perm=buffers.x_perm,
        x_perm_sf=buffers.x_perm_sf_physical,
        act=buffers.act,
        act_sf=buffers.act_sf_physical,
        tile_idx_to_expert_idx=buffers.tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=buffers.tile_idx_to_mn_limit,
        num_non_exiting_tiles=buffers.num_non_exiting_tiles,
        alpha=w1_alpha,
        beta=beta,
        linear_beta=linear_beta,
    )
    if finalize:
        output.zero_()
    swapab_gemm2(
        w2=w2,
        w2_sf=w2_sf,
        act=buffers.act,
        act_sf=buffers.act_sf_physical,
        out=output,
        tile_idx_to_expert_idx=buffers.tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=buffers.tile_idx_to_mn_limit,
        num_non_exiting_tiles=buffers.num_non_exiting_tiles,
        alpha=w2_alpha,
        permuted_idx_to_expanded_idx=buffers.permuted_idx_to_expanded_idx,
        token_final_scales=route_weights if finalize else None,
        top_k=top_k,
        finalize=finalize,
    )
    return output
