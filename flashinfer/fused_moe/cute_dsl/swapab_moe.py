# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Swap-AB execution path for few-rows-per-expert MXFP4 x MXFP8 MoE.

Routes with a small number of rows per expert (decode, small prefill, MoE-TP
shards) run the two grouped GEMMs with the expert weights as the MMA-M operand
and ``n_tile``-row groups of routed activations as the MMA-N operand
(:mod:`.blackwell.blockscaled_swapab_grouped_gemm`). The GEMM1 kernel gathers
the activation rows itself and the GEMM2 kernel reduces into the output it
zero-filled during GEMM1, so the whole forward is ``moe_sort`` + two kernels.
This module owns the host side: row-group capacity, workspace buffers, the
compile cache and the launch sequence.
"""

import os
from collections import OrderedDict
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

# Rows of routed activations per MMA-N tile. SWAPAB_NTILE overrides for
# experiments (8, 16, 32, 64 or 128).
SWAP_ROW_TILE = int(os.environ.get("SWAPAB_NTILE", "8"))
# K-blocks (of 32 elements) per mainloop stage: 4, 8 or 16 (SWAPAB_KBLOCKS).
_ENV_KBLOCKS = os.environ.get("SWAPAB_KBLOCKS")
SWAP_K_BLOCKS_PER_STAGE = int(_ENV_KBLOCKS or "4")


def gemm1_k_blocks_per_stage(n_tile: int) -> int:
    """Stage depth for GEMM1 (K = hidden size). 8-row groups are bound by the
    per-stage round trip once the weights sit in L2 (few hot experts), so they
    take 256-wide stages; wider groups stream from HBM and keep 128-wide
    stages with the deeper pipeline (B300: 32-row prefill tiles lose ~3% at 8)."""
    if _ENV_KBLOCKS:
        return int(_ENV_KBLOCKS)
    return 8 if n_tile == 8 else 4


# GEMM2 (K = intermediate shard) may use a different stage depth: 12 covers a
# 384-wide MoE-TP shard in one stage.
_ENV_KBLOCKS2 = os.environ.get("SWAPAB_KBLOCKS2")


def gemm2_k_blocks_per_stage(k: int) -> int:
    """Stage depth for GEMM2: one 384-wide stage when the intermediate shard
    allows it (a 384-wide MoE-TP shard becomes a single stage per tile),
    otherwise the generic 128-wide stage."""
    if _ENV_KBLOCKS2:
        return int(_ENV_KBLOCKS2)
    if k % 384 == 0:
        return 12
    return SWAP_K_BLOCKS_PER_STAGE


SWAP_PERF_PROBE = int(os.environ.get("SWAPAB_PROBE", "0"))
# L2 eviction priority for the streamed weight TMA loads; the encodings follow
# CUTLASS's SM90 TMA cache hints. The launchers take the policy per call
# (``weight_l2_hint``); SWAPAB_L2HINT = none | first | last overrides it.
TMA_L2_EVICT_FIRST = 0x12F0000000000000
TMA_L2_EVICT_LAST = 0x14F0000000000000
_L2_HINTS = {"none": None, "first": TMA_L2_EVICT_FIRST, "last": TMA_L2_EVICT_LAST}
_ENV_L2HINT = os.environ.get("SWAPAB_L2HINT")


def _resolve_weight_l2_hint(weight_l2_hint: Optional[int]) -> Optional[int]:
    if _ENV_L2HINT:
        return _L2_HINTS[_ENV_L2HINT]
    return weight_l2_hint


# Pipeline depth knobs (SWAPAB_STAGES / SWAPAB_ACC_STAGES override for tuning).
SWAP_MAX_AB_STAGES = int(os.environ.get("SWAPAB_STAGES", "12"))
SWAP_ACC_STAGES = int(os.environ.get("SWAPAB_ACC_STAGES", "2"))
SWAP_TILE_STAGES = int(os.environ.get("SWAPAB_TILE_STAGES", "8"))
# Where GEMM2 resolves its per-tile epilogue metadata (expert alpha, output
# row and route weight per routed column): "sched" = the scheduler warp fills
# a per-tile smem ring ahead of time, "epi" = the epilogue warps prefetch it
# one tile ahead in registers.
SWAP_META_IN_SCHED = os.environ.get("SWAPAB_META", "sched") != "epi"
# bit 0: tile-major W1 (GEMM1), bit 1: tile-major W2 (GEMM2). Default: both.
# Each 128x128 MMA tile becomes one contiguous 8 KB block so the weight TMA
# streams whole DRAM pages (B300: GEMM1 -2%, GEMM2 -2.5% at T=16 balanced).
SWAP_TILED_WEIGHTS = int(os.environ.get("SWAPAB_TILED_W", "3"))

_tiled_weight_cache: "OrderedDict[Tuple, Tuple[torch.Tensor, torch.Tensor]]" = (
    OrderedDict()
)
_TILED_WEIGHT_CACHE_MAX = 64


def tile_major_weights(w: torch.Tensor) -> torch.Tensor:
    """One-time relayout of packed MXFP4 weights ``(E, M, K/2)`` uint8 into
    tile-major ``(E, M/128, K/128, 128, 64)`` so every 128x128 MMA tile is a
    contiguous 8 KB block.

    Cached per source tensor. The entry keeps a reference to the source so its
    address cannot be recycled by the allocator while the entry is alive (the
    key includes the address); weights are treated as immutable.
    """
    key = (w.data_ptr(), tuple(w.shape), w.dtype, w.device)
    hit = _tiled_weight_cache.get(key)
    if (
        hit is not None
        and hit[0] is w
        or (hit is not None and hit[0].data_ptr() == w.data_ptr())
    ):
        _tiled_weight_cache.move_to_end(key)
        return hit[1]
    e, m, kb = w.shape
    if m % 128 or kb % 64:
        raise ValueError("tile-major weights need M % 128 == 0 and K % 128 == 0")
    t = w.view(e, m // 128, 128, kb // 64, 64).permute(0, 1, 3, 2, 4).contiguous()
    _tiled_weight_cache[key] = (w, t)
    while len(_tiled_weight_cache) > _TILED_WEIGHT_CACHE_MAX:
        _tiled_weight_cache.popitem(last=False)
    return t


_swapab_kernel_cache: Dict[Tuple, Any] = {}


def swap_row_capacity(
    num_tokens: int, top_k: int, num_local_experts: int, tile: int = SWAP_ROW_TILE
) -> Tuple[int, int]:
    """Return ``(row_groups, rows)`` for the swap path (``rows = groups * tile``)."""
    groups = get_max_num_tiles(num_tokens, top_k, num_local_experts, tile)
    return groups, groups * tile


def _gmem_ptr(dtype, tensor: Optional[torch.Tensor], align: int = 16):
    if tensor is None:
        return None
    return make_ptr(
        dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=align
    )


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
    tiled_a: bool = False,
    zero_fill: bool = False,
    weight_l2_hint: Optional[int] = None,
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
        SWAP_MAX_AB_STAGES,
        SWAP_ACC_STAGES,
        SWAP_TILE_STAGES,
        SWAP_META_IN_SCHED,
        SWAP_PERF_PROBE,
        weight_l2_hint,
        tiled_a,
        # ``zero_output`` is a compile-time specialisation (None vs pointer).
        zero_fill,
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
            max_ab_stages=SWAP_MAX_AB_STAGES,
            num_acc_stages=SWAP_ACC_STAGES,
            num_tile_stages=SWAP_TILE_STAGES,
            meta_in_sched=SWAP_META_IN_SCHED,
            perf_probe=SWAP_PERF_PROBE,
            weight_l2_hint=weight_l2_hint,
        )
        _swapab_kernel_cache[key] = cute.compile(
            kernel.wrapper,
            *compile_args,
            top_k=top_k,
            beta_count=beta_count,
            linear_beta_count=linear_beta_count,
            max_active_clusters=max_active_clusters,
            tiled_a=tiled_a,
            stream=stream,
        )
        if os.environ.get("SWAPAB_DEBUG"):
            print(f"[swapab] compiled {key}", file=sys.stderr, flush=True)
    return _swapab_kernel_cache[key]


def swapab_gemm1_situ(
    *,
    w1: torch.Tensor,
    w1_sf: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    act: torch.Tensor,
    act_sf: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    linear_beta: Optional[torch.Tensor],
    top_k: int,
    zero_output: Optional[torch.Tensor] = None,
    weight_l2_hint: Optional[int] = None,
    n_tile: int = SWAP_ROW_TILE,
    k_blocks_per_stage: Optional[int] = None,
    enable_pdl: bool = False,
    _prepared_launches: Optional[Dict[str, Any]] = None,
) -> None:
    """GEMM1 (up/gate) + SiTU + MXFP8 requantization on the swap path.

    ``w1`` is the prepared ``[L, 2I, H/2]`` interleaved weight, ``x`` the
    unpermuted ``[T, H]`` E4M3 activations with plain ``[T, H/32]`` UE8M0 scales
    ``x_sf``; rows are gathered through ``permuted_idx_to_expanded_idx``
    (``[R]``). ``act`` is the ``[R, I]`` E4M3 output and ``act_sf`` its plain
    ``[R, I/32]`` scales. ``zero_output`` (BF16, 16-byte multiple) is
    zero-filled by the kernel for the following finalize GEMM2.
    """
    num_local_experts, rows_w, packed_k = w1.shape
    k = packed_k * 2
    num_tokens = x.shape[0]
    rows = permuted_idx_to_expanded_idx.shape[0]
    intermediate = rows_w // 2
    if x.shape[1] != k or x_sf.shape != (num_tokens, k // 32):
        raise ValueError("x must be [T, H] with x_sf [T, H/32]")
    if act.shape != (rows, intermediate) or act_sf.shape != (rows, intermediate // 32):
        raise ValueError(
            f"act must be [{rows}, {intermediate}] with act_sf [{rows}, {intermediate // 32}]"
        )
    zero_words = 0
    if zero_output is not None:
        if zero_output.numel() * zero_output.element_size() % 8:
            raise ValueError("zero_output byte size must be a multiple of 8")
        zero_words = zero_output.numel() * zero_output.element_size() // 8
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = get_max_active_clusters(1)
    use_linear_beta = linear_beta is not None
    if k_blocks_per_stage is None:
        k_blocks_per_stage = gemm1_k_blocks_per_stage(n_tile)
    args = (
        _gmem_ptr(
            cutlass.Float4E2M1FN,
            tile_major_weights(w1) if (SWAP_TILED_WEIGHTS & 1) else w1,
            32,
        ),
        _gmem_ptr(cutlass.Float8E4M3FN, x, 16),
        _gmem_ptr(cutlass.Float8E8M0FNU, w1_sf, 16),
        _gmem_ptr(cutlass.Float8E8M0FNU, x_sf, 4),
        _gmem_ptr(cutlass.Float8E4M3FN, act, 16),
        _gmem_ptr(cutlass.Uint8, act_sf, 4),
        _gmem_ptr(cutlass.Int32, tile_idx_to_expert_idx, 4),
        _gmem_ptr(cutlass.Int32, tile_idx_to_mn_limit, 4),
        _gmem_ptr(cutlass.Int32, num_non_exiting_tiles, 4),
        _gmem_ptr(cutlass.Float32, alpha, 4),
        _gmem_ptr(cutlass.Int32, permuted_idx_to_expanded_idx, 4),
        None,
        _gmem_ptr(cutlass.Float32, beta, 4),
        _gmem_ptr(cutlass.Float32, linear_beta, 4),
        _gmem_ptr(cutlass.Int64, zero_output, 8),
        rows_w,
        k,
        num_local_experts,
        num_tokens,
        rows,
        rows,
        intermediate,
        num_tokens,
        tile_idx_to_expert_idx.shape[0],
        zero_words,
    )
    compiled = _get_compiled_swapab_kernel(
        epilogue_kind="situ_mxfp8",
        n_tile=n_tile,
        k_blocks_per_stage=k_blocks_per_stage,
        top_k=top_k,
        beta_count=beta.numel(),
        linear_beta_count=linear_beta.numel() if use_linear_beta else 1,
        use_linear_beta=use_linear_beta,
        enable_pdl=enable_pdl,
        compile_args=args,
        max_active_clusters=max_active_clusters,
        stream=stream,
        tiled_a=bool(SWAP_TILED_WEIGHTS & 1),
        zero_fill=zero_output is not None,
        weight_l2_hint=_resolve_weight_l2_hint(weight_l2_hint),
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
    k_blocks_per_stage: Optional[int] = None,
    enable_pdl: bool = False,
    weight_l2_hint: Optional[int] = None,
    _prepared_launches: Optional[Dict[str, Any]] = None,
) -> None:
    """GEMM2 (down) on the swap path.

    ``act`` / ``act_sf`` are the permuted ``[R, I]`` E4M3 rows and plain
    ``[R, I/32]`` scales written by GEMM1. ``finalize=True`` reduce-adds
    ``alpha * route_weight * acc`` into the zero-filled ``out[T, H]``;
    ``finalize=False`` (deferred finalize) writes ``alpha * acc`` to
    ``out[R', H]`` (``R' >= R``) in permuted row order, i.e. row
    ``expanded_idx_to_permuted_idx[t, k]`` holds expert output for
    ``(token t, slot k)``; padding rows are left untouched.
    """
    num_local_experts, rows_w, packed_k = w2.shape
    k = packed_k * 2
    if k_blocks_per_stage is None:
        k_blocks_per_stage = gemm2_k_blocks_per_stage(k)
    rows = act.shape[0]
    if act.shape[1] != k or act_sf.shape != (rows, k // 32):
        raise ValueError("act must be [R, I] with act_sf [R, I/32]")
    if permuted_idx_to_expanded_idx.shape[0] != rows:
        raise ValueError("permuted_idx_to_expanded_idx must have one entry per act row")
    if finalize:
        if token_final_scales is None or token_final_scales.dtype != torch.float32:
            raise ValueError("finalize requires float32 token_final_scales")
        num_tokens = token_final_scales.shape[0]
        if out.shape != (num_tokens, rows_w):
            raise ValueError("finalize output must be [T, H]")
    else:
        if out.ndim != 2 or out.shape[0] < rows or out.shape[1] != rows_w:
            raise ValueError(
                "deferred output must be [>= permuted rows, H] BF16 rows in "
                "permuted order"
            )
        num_tokens = out.shape[0]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    max_active_clusters = get_max_active_clusters(1)
    args = (
        _gmem_ptr(
            cutlass.Float4E2M1FN,
            tile_major_weights(w2) if (SWAP_TILED_WEIGHTS & 2) else w2,
            32,
        ),
        _gmem_ptr(cutlass.Float8E4M3FN, act, 16),
        _gmem_ptr(cutlass.Float8E8M0FNU, w2_sf, 16),
        _gmem_ptr(cutlass.Float8E8M0FNU, act_sf, 4),
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
        None,
        rows_w,
        k,
        num_local_experts,
        rows,
        rows,
        out.shape[0],
        rows_w,
        num_tokens,
        tile_idx_to_expert_idx.shape[0],
        0,
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
        tiled_a=bool(SWAP_TILED_WEIGHTS & 2),
        weight_l2_hint=_resolve_weight_l2_hint(weight_l2_hint),
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
        tile: int = SWAP_ROW_TILE,
    ):
        self.tile = tile
        self.groups, self.rows = swap_row_capacity(
            num_tokens, top_k, num_local_experts, tile
        )
        r = self.rows
        self.act = torch.empty(
            (r, intermediate), dtype=torch.float8_e4m3fn, device=device
        )
        self.act_sf = torch.empty(
            (r, intermediate // 32), dtype=torch.uint8, device=device
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
    enable_pdl: bool = False,
    _prepared_launches: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Full swap-path MoE: ``moe_sort`` (``tile``-row groups) -> GEMM1 -> GEMM2.

    ``finalize=False`` writes deferred rows (``alpha * acc`` per permuted row,
    no route weight) into ``output[buffers.rows, H]``; the caller finalizes
    with ``buffers.expanded_idx_to_permuted_idx`` and ``route_weights``.
    """
    moe_sort(
        token_selected_experts=route_ids,
        token_final_scales=route_weights,
        num_experts=num_experts,
        top_k=top_k,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_tokens_dim=buffers.tile,
        enable_pdl=enable_pdl,
        _prepared_launches=_prepared_launches,
        **buffers.sort_kwargs(),
    )
    swapab_gemm1_situ(
        w1=w1,
        w1_sf=w1_sf,
        x=x,
        x_sf=x_sf,
        permuted_idx_to_expanded_idx=buffers.permuted_idx_to_expanded_idx,
        act=buffers.act,
        act_sf=buffers.act_sf,
        tile_idx_to_expert_idx=buffers.tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=buffers.tile_idx_to_mn_limit,
        num_non_exiting_tiles=buffers.num_non_exiting_tiles,
        alpha=w1_alpha,
        beta=beta,
        linear_beta=linear_beta,
        top_k=top_k,
        zero_output=output if finalize else None,
        n_tile=buffers.tile,
        enable_pdl=enable_pdl,
        _prepared_launches=_prepared_launches,
    )
    swapab_gemm2(
        w2=w2,
        w2_sf=w2_sf,
        act=buffers.act,
        act_sf=buffers.act_sf,
        out=output,
        tile_idx_to_expert_idx=buffers.tile_idx_to_expert_idx,
        tile_idx_to_mn_limit=buffers.tile_idx_to_mn_limit,
        num_non_exiting_tiles=buffers.num_non_exiting_tiles,
        alpha=w2_alpha,
        permuted_idx_to_expanded_idx=buffers.permuted_idx_to_expanded_idx,
        token_final_scales=route_weights if finalize else None,
        top_k=top_k,
        finalize=finalize,
        n_tile=buffers.tile,
        enable_pdl=enable_pdl,
        _prepared_launches=_prepared_launches,
    )
    return output
