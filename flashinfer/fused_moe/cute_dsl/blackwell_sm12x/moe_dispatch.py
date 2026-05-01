"""SM120/SM121 MoE dispatch layer — workspace, compilation, and launch.

Ported from b12x's integration/tp_moe.py. Supports micro (tiny decode),
static (decode), and dynamic (prefill) backends with token-count-based
selection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import cutlass
import cutlass.cute as cute
import torch

from flashinfer.cute_dsl.utils import (
    convert_sf_from_mma_layout,
    get_max_active_clusters,
    get_num_sm,
    make_ptr,
)
from .moe_dynamic_kernel import MoEDynamicKernel
from .moe_micro_kernel import MoEMicroKernel
from .moe_static_kernel import MoEStaticKernel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NVFP4_BLOCK_SIZE = 16
_LEVEL_TILE_M = 128
_LEVEL_TILE_N = 128
SF_VEC_SIZE = 16

# Micro kernel cutover thresholds (routed pairs)
_MICRO_COMPACT_CUTOVER_PAIRS = 20
_MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK = 40

# MAC (max active clusters) tuning ladders from b12x decode profiling.
# Each entry is (max_routed_rows, optimal_mac).
_MICRO_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (2, 84),
    (4, 127),
    (8, 107),
    (10, 84),
    (16, 63),
    (20, 84),
)
_STATIC_MAC_LADDER: Tuple[Tuple[int, int], ...] = (
    (24, 148),
    (32, 169),
    (40, 132),
    (48, 149),
    (64, 134),
    (80, 175),
    (96, 171),
    (120, 125),
    (128, 130),
    (160, 171),
    (192, 166),
    (256, 141),
    (320, 158),
    (512, 175),
    (640, 188),
)


def _lookup_mac_ladder(
    ladder: Tuple[Tuple[int, int], ...], routed_rows: int
) -> int | None:
    """Look up optimal MAC from a tuning ladder. Returns None if no match."""
    for end_rows, mac in ladder:
        if routed_rows <= end_rows:
            return mac
    return None


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _select_moe_mma_tiler_mn(routed_rows: int, n: int) -> Tuple[int, int]:
    """Select optimal MoE tile shape based on routed rows and N dimension.

    Uses narrower 64x128 tiles when routed_rows <= 128 and default 128x128
    would leave SMs idle.
    """
    sm_count = get_num_sm(torch.device("cuda"))
    coarse_tile = (128, 128)
    coarse_tiles = ((routed_rows + coarse_tile[0] - 1) // coarse_tile[0]) * (
        (n + coarse_tile[1] - 1) // coarse_tile[1]
    )
    # Single-token decode often lands exactly on the "half the machine"
    # boundary. Keeping the coarse 128x128 tile there leaves the M dimension
    # badly underfilled, so take the narrow 64x128 tile inclusive of equality.
    if routed_rows <= 128 and coarse_tiles <= max(1, sm_count // 2):
        return (64, 128)
    return (128, 128)


def _as_grouped_scale_view(
    scale_storage: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Create 6D MMA-compatible scale factor view from swizzled storage."""
    batch = scale_storage.shape[0]
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(cols // SF_VEC_SIZE, 4)
    sf = scale_storage.view(torch.float8_e4m3fn)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class Sm120StaticMoEWorkspace:
    """Scratch buffers for one SM120 static MoE launch."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device

    # Buffers
    row_counts: torch.Tensor  # [state_E] int32
    token_map: torch.Tensor  # [state_E, max_rows] int32
    token_weights: torch.Tensor  # [state_E, max_rows] float32
    packed_input: torch.Tensor  # [state_E, max_rows, k//2] uint8
    packed_input_scale: torch.Tensor  # [state_E, rows_pad_k, cols_pad_k] uint8
    barrier_count: torch.Tensor  # [1] int32
    barrier_epoch: torch.Tensor  # [1] int32
    active_expert_count: torch.Tensor  # [1] int32
    weight_expert_ids: torch.Tensor  # [state_E] int32
    global_to_local_expert: torch.Tensor  # [weight_E] int32
    compact_topk_ids: torch.Tensor  # [state_E] int32, for micro kernel pre-pass

    # Views (set after allocation)
    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None


def allocate_sm120_static_workspace(
    *,
    state_E: int,
    weight_E: int,
    max_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
) -> Sm120StaticMoEWorkspace:
    """Allocate workspace buffers for the SM120 static MoE kernel."""
    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    workspace = Sm120StaticMoEWorkspace(
        state_E=state_E,
        weight_E=weight_E,
        max_rows=max_rows,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(state_E, max_rows, dtype=torch.int32, device=device),
        token_weights=torch.zeros(
            state_E, max_rows, dtype=torch.float32, device=device
        ),
        packed_input=torch.empty(
            state_E, max_rows, k // 2, dtype=torch.uint8, device=device
        ),
        packed_input_scale=torch.empty(
            state_E, rows_pad_k, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        active_expert_count=torch.zeros(1, dtype=torch.int32, device=device),
        weight_expert_ids=torch.arange(state_E, dtype=torch.int32, device=device),
        global_to_local_expert=torch.empty(weight_E, dtype=torch.int32, device=device),
        compact_topk_ids=torch.empty(state_E, dtype=torch.int32, device=device),
    )

    # Finalize views
    sf_dtype = cutlass.Float8E4M3FN
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0).view(
        torch.float4_e2m1fn_x2
    )
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return workspace


# ---------------------------------------------------------------------------
# Weight views
# ---------------------------------------------------------------------------
@dataclass
class _WeightViews:
    w13_fp4: object = None
    down_fp4: object = None
    sfb_w13_ptr: object = None
    sfb_down_ptr: object = None
    w1_alpha: torch.Tensor | None = None
    w2_alpha: torch.Tensor | None = None
    _w13_sf_storage: torch.Tensor | None = None
    _down_sf_storage: torch.Tensor | None = None


_WEIGHT_CACHE: Dict[Tuple, _WeightViews] = {}


def _get_weight_views(
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_alphas: torch.Tensor,
    n: int,
    k: int,
) -> _WeightViews:
    """Create permuted weight views for the static kernel.

    The kernel expects concatenated w13 data with shape [2*n, k//2, E]
    via a single TMA descriptor.
    """
    # The kernel splits w13 into gate/up halves by tile index. This only works
    # when the boundary between halves lands on a tile-aligned column.
    if n % _LEVEL_TILE_N != 0:
        raise ValueError(
            f"intermediate_size ({n}) must be a multiple of {_LEVEL_TILE_N} "
            f"for the SM120 MoE kernel's gate/up tile split."
        )

    key = (
        w1_fp4.data_ptr(),
        w1_blockscale.data_ptr(),
        w1_alphas.data_ptr(),
        w2_fp4.data_ptr(),
        w2_blockscale.data_ptr(),
        w2_alphas.data_ptr(),
    )
    cached = _WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached

    # Permute [E, w1_rows, k//2] -> [w1_rows, k//2, E] (view, no copy)
    # w1_rows is 2*n for gated (SiLU) or n for non-gated (ReLU2)
    w13 = w1_fp4.permute(1, 2, 0)
    down = w2_fp4.permute(1, 2, 0)

    # The kernel's TMA descriptors read scale factors in the physical storage
    # order produced by swizzle_block_scale: (batch, rows_padded, cols_padded).
    # convert_sf_to_mma_layout returns a strided 6D view over this storage.
    # We need the ORIGINAL physical storage, not .contiguous() of the view
    # (which would write in permuted logical order).
    # convert_sf_from_mma_layout reverses the permutation back to 2D swizzled.
    sf_dtype = cutlass.Float8E4M3FN
    w1_rows = w1_fp4.shape[1]  # 2*n for gated, n for non-gated
    w13_sf_contiguous = convert_sf_from_mma_layout(
        w1_blockscale,
        m=w1_rows,
        k=k,
        num_groups=w1_fp4.shape[0],  # num_local_experts
    ).contiguous()
    down_sf_contiguous = convert_sf_from_mma_layout(
        w2_blockscale,
        m=k,
        k=n,
        num_groups=w2_fp4.shape[0],
    ).contiguous()

    views = _WeightViews(
        w13_fp4=w13.view(torch.float4_e2m1fn_x2),
        down_fp4=down.view(torch.float4_e2m1fn_x2),
        sfb_w13_ptr=make_ptr(
            sf_dtype,
            w13_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        sfb_down_ptr=make_ptr(
            sf_dtype,
            down_sf_contiguous.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        w1_alpha=w1_alphas.contiguous().to(torch.float32),
        w2_alpha=w2_alphas.contiguous().to(torch.float32),
    )
    # Keep references to prevent GC of contiguous buffers
    views._w13_sf_storage = w13_sf_contiguous
    views._down_sf_storage = down_sf_contiguous
    _WEIGHT_CACHE[key] = views
    return views


# ---------------------------------------------------------------------------
# Kernel compilation cache
# ---------------------------------------------------------------------------
_STATIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_static_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    mac_override: int | None = None,
    activation: str = "silu",
):
    """Compile (or retrieve cached) the SM120 static MoE kernel."""
    sf_vec_size = 16
    sm_count = get_num_sm(torch.device("cuda"))
    mac = (
        mac_override
        if mac_override is not None
        else min(get_max_active_clusters(1), sm_count)
    )

    # Select tile size based on actual routed rows
    routed_rows = m * num_topk
    mma_tiler_mn = (128, 128)
    if num_topk > 1:
        mma_tiler_mn = _select_moe_mma_tiler_mn(routed_rows, n)

    cache_key = (
        "static",
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
    )
    cached = _STATIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel = MoEStaticKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1]),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
    )

    is_gated = activation == "silu"
    w1_rows = (2 if is_gated else 1) * n  # 2*n for gated, n for non-gated

    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    # Build fake tensors for compilation
    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype,
        (m * num_topk,),
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (m * num_topk,),
        assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (max_rows, k, state_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * max_rows * (k // 2),),
        assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * rows_pad_k * cols_pad_k,),
        assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (w1_rows, k, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (k, n, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (weight_E,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=16,
    )
    compiled = cute.compile(
        kernel,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        row_counts_fake,
        active_expert_count_fake,
        weight_expert_ids_fake,
        global_to_local_expert_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        mac,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )

    result = (compiled, mac)
    _STATIC_KERNEL_CACHE[cache_key] = result
    return result


_MICRO_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_micro_kernel(
    state_E: int,
    weight_E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    share_input_across_experts: bool = False,
    mac_override: int | None = None,
    activation: str = "silu",
):
    """Compile (or retrieve cached) the SM120 micro MoE kernel."""
    sf_vec_size = 16
    sm_count = get_num_sm(torch.device("cuda"))
    mac = (
        mac_override
        if mac_override is not None
        else min(get_max_active_clusters(1), sm_count)
    )

    # Micro always selects tile from routed rows (not just for multi-topk)
    routed_rows = m * num_topk
    mma_tiler_mn = _select_moe_mma_tiler_mn(routed_rows, n)

    cache_key = (
        "micro",
        state_E,
        weight_E,
        m,
        k,
        n,
        num_topk,
        max_rows,
        mac,
        mma_tiler_mn,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        share_input_across_experts,
        activation,
    )
    cached = _MICRO_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel = MoEMicroKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=mma_tiler_mn,
        output_tile_count_n=max(1, (n + mma_tiler_mn[1] - 1) // mma_tiler_mn[1]),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
        share_input_across_experts=share_input_across_experts,
    )

    is_gated = activation == "silu"
    w1_rows = (2 if is_gated else 1) * n

    rows_pad_k = _align_up(max_rows, 128)
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    # Build fake tensors for compilation (identical to static kernel)
    a_input_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8
    topk_ids_fake = cute.runtime.make_fake_compact_tensor(
        topk_ids_cutlass_dtype,
        (m * num_topk,),
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (m * num_topk,),
        assumed_align=4,
    )
    packed_a_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (max_rows, k, state_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * max_rows * (k // 2),),
        assumed_align=16,
    )
    scale_storage_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (state_E * rows_pad_k * cols_pad_k,),
        assumed_align=16,
    )
    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (w1_rows, k, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (k, n, weight_E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    active_expert_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (1,),
        assumed_align=4,
    )
    weight_expert_ids_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E,),
        assumed_align=4,
    )
    global_to_local_expert_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (weight_E,),
        assumed_align=4,
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (weight_E,),
        assumed_align=16,
    )
    scatter_fake = cute.runtime.make_fake_compact_tensor(
        a_dtype,
        (m, k),
        stride_order=(1, 0),
        assumed_align=16,
    )
    token_map_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=4,
    )
    token_weights_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype,
        (state_E, max_rows),
        stride_order=(1, 0),
        assumed_align=16,
    )
    compiled = cute.compile(
        kernel,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        row_counts_fake,
        active_expert_count_fake,
        weight_expert_ids_fake,
        global_to_local_expert_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        mac,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )

    result = (compiled, mac)
    _MICRO_KERNEL_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
def _expand_to_experts(t: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Broadcast a scalar or [1] tensor to [num_experts]."""
    if t.numel() == 1:
        return t.expand(num_experts).contiguous()
    return t.contiguous().to(torch.float32)


def launch_sm120_static_moe(
    *,
    workspace: Sm120StaticMoEWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    k: int,
    n: int,
    top_k: int,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
) -> torch.Tensor:
    """Launch the SM120 static or micro MoE kernel.

    Selects the micro kernel for tiny decode batches (routed_rows <= 20-40)
    and the static kernel otherwise. The micro path runs a Triton pre-pass
    to compact routing IDs before launching.
    """
    # Flatten routing tensors
    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)
    routed_rows = num_tokens * top_k

    # Capture whether input_gs was a single shared scalar BEFORE expansion:
    # the m=1 relu2 shared-input micro optimization only applies when every
    # expert sees the same FC1-input global scale.
    input_gs_is_shared = input_gs.numel() == 1

    # Broadcast scalar scales to per-expert [E] tensors
    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    # Decide micro vs static
    micro_cutover = _MICRO_COMPACT_CUTOVER_PAIRS
    if top_k > 1:
        micro_cutover = _MICRO_COMPACT_CUTOVER_PAIRS_MULTI_TOPK
    use_micro = routed_rows <= micro_cutover

    sm_count = get_num_sm(torch.device("cuda"))
    base_mac = min(get_max_active_clusters(1), sm_count)

    if use_micro:
        assert flat_ids.numel() <= workspace.compact_topk_ids.numel(), (
            f"compact_topk_ids buffer too small: "
            f"{workspace.compact_topk_ids.numel()} < {flat_ids.numel()}"
        )
        compact_ids = workspace.compact_topk_ids[: flat_ids.numel()]
        if num_tokens == 1:
            # A single token's top-k is already a dense unique expert set,
            # so we can build the compact local-id mapping on the host
            # without launching the Triton compaction kernel. The micro
            # kernel still reads weight_expert_ids the same way it does
            # for m>1; it just sees a pre-filled workspace.
            compact_ids.copy_(
                torch.arange(
                    flat_ids.numel(),
                    device=flat_ids.device,
                    dtype=torch.int32,
                )
            )
            workspace.weight_expert_ids[: flat_ids.numel()].copy_(
                flat_ids.to(torch.int32)
            )
            workspace.active_expert_count.fill_(flat_ids.numel())
        else:
            from .triton_compact import compact_topk_ids as _triton_compact_topk_ids

            _triton_compact_topk_ids(
                flat_ids,
                compact_ids,
                workspace.weight_expert_ids,
                workspace.active_expert_count,
            )
        launch_ids = compact_ids
        # Select micro MAC: min of tuned ladder, work tiles, and hardware limit.
        # The hardware cap (base_mac) prevents deadlocks on GPUs with fewer SMs
        # than the profiled tuning target.
        micro_work_tiles = max(1, routed_rows * max(1, (n + 128 - 1) // 128))
        tuned_mac = _lookup_mac_ladder(_MICRO_MAC_LADDER, routed_rows)
        micro_mac = min(tuned_mac or base_mac, micro_work_tiles, base_mac)
        # For m=1 relu2 with a shared FC1-input scale, all experts see the
        # same quantized activation — quantize once and share the packed
        # buffer slot across all K top-k pairs. Env override lets us flip
        # this off without a code change if a regression surfaces.
        share_input_across_experts = (
            activation == "relu2"
            and num_tokens == 1
            and input_gs_is_shared
            and os.environ.get("FLASHINFER_B12X_MICRO_SHARE_INPUT", "1") != "0"
        )
        compiled, mac = _get_micro_kernel(
            workspace.state_E,
            num_experts,
            num_tokens,
            k,
            n,
            top_k,
            workspace.max_rows,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            share_input_across_experts=share_input_across_experts,
            mac_override=micro_mac,
            activation=activation,
        )
    else:
        # Static path — use hardware default MAC (same as main).
        # MAC tuning for the static kernel is deferred to a follow-up
        # to avoid changing behavior for existing static workloads.
        compiled, mac = _get_static_kernel(
            workspace.state_E,
            num_experts,
            num_tokens,
            k,
            n,
            top_k,
            workspace.max_rows,
            topk_ids_dtype=torch.int32,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
        )
        launch_ids = flat_ids

    # With TVM-FFI env stream, the stream is managed automatically.
    # max_active_clusters is still a required positional arg for TVM-FFI.
    # Pointer arguments must be passed as raw ints (data_ptr()) at runtime.
    compiled(
        a,
        launch_ids,
        flat_weights,
        workspace.packed_a_view,
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat,
        workspace.scale_flat,
        workspace.barrier_count,
        workspace.barrier_epoch,
        weights.w13_fp4,
        weights._w13_sf_storage.data_ptr(),
        weights.down_fp4,
        weights._down_sf_storage.data_ptr(),
        workspace.row_counts,
        workspace.active_expert_count,
        workspace.weight_expert_ids,
        workspace.global_to_local_expert,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output,
        workspace.token_map,
        workspace.token_weights,
        mac,
    )

    return scatter_output


# ==========================================================================
# Dynamic backend
# ==========================================================================

_STATIC_COMPACT_CUTOVER_PAIRS = 640
_DYNAMIC_SLICE_CHUNK = 2


def select_sm120_moe_backend(*, num_tokens: int, num_topk: int) -> str:
    """Pick static or dynamic backend based on routed-pair count."""
    routed_rows = num_tokens * num_topk
    if routed_rows <= _STATIC_COMPACT_CUTOVER_PAIRS:
        return "static"
    return "dynamic"


# ---------------------------------------------------------------------------
# Dynamic workspace
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class Sm120DynamicMoEWorkspace:
    """Scratch buffers for one SM120 dynamic MoE launch."""

    state_E: int
    weight_E: int
    max_rows: int
    k: int
    n: int
    num_topk: int
    device: torch.device

    # Core buffers
    row_counts: torch.Tensor
    token_map: torch.Tensor
    token_weights: torch.Tensor
    packed_input: torch.Tensor
    packed_input_scale: torch.Tensor
    barrier_count: torch.Tensor
    barrier_epoch: torch.Tensor

    # Dynamic-specific
    routed_rows_capacity: int
    physical_tiles_capacity: int
    task_capacity: int
    expert_write_rows: torch.Tensor
    expert_tile_base: torch.Tensor
    pair_head: torch.Tensor
    producers_done_count: torch.Tensor
    all_work_published: torch.Tensor
    task_head: torch.Tensor
    task_tail: torch.Tensor
    task_ready: torch.Tensor
    task_expert: torch.Tensor
    task_m_tile: torch.Tensor
    task_slice_begin: torch.Tensor
    task_slice_count: torch.Tensor
    task_valid_rows: torch.Tensor
    tile_write_count: torch.Tensor

    # Views
    packed_a_view: torch.Tensor | None = None
    sfa_ptr: object = None
    packed_a_flat: torch.Tensor | None = None
    scale_flat: torch.Tensor | None = None


def _dynamic_task_geometry(state_E: int, n: int, routed_rows: int):
    """Compute task queue dimensions from problem geometry.

    Each active expert can introduce at most one additional physical tile
    beyond the base count (due to per-expert tail padding). The task queue
    holds one entry per (m_tile, slice_group) pair — NOT multiplied by E.
    """
    routed_rows = max(1, routed_rows)
    base_m_tiles = _align_up(routed_rows, _LEVEL_TILE_M) // _LEVEL_TILE_M
    active_expert_upper_bound = min(state_E, routed_rows)
    max_m_tiles = max(1, base_m_tiles + active_expert_upper_bound - 1)
    gate_tile_cnt = max(1, (n + _LEVEL_TILE_N - 1) // _LEVEL_TILE_N)
    slice_groups = max(
        1, (gate_tile_cnt + _DYNAMIC_SLICE_CHUNK - 1) // _DYNAMIC_SLICE_CHUNK
    )
    max_tasks = max_m_tiles * slice_groups
    return max_m_tiles, gate_tile_cnt, max_tasks


def allocate_sm120_dynamic_workspace(
    *,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
) -> Sm120DynamicMoEWorkspace:
    """Allocate workspace buffers for the SM120 dynamic MoE kernel."""
    physical_tiles, _, max_tasks = _dynamic_task_geometry(state_E, n, routed_rows)
    rows_padded = physical_tiles * _LEVEL_TILE_M
    cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    workspace = Sm120DynamicMoEWorkspace(
        state_E=state_E,
        weight_E=weight_E,
        max_rows=rows_padded,
        k=k,
        n=n,
        num_topk=num_topk,
        device=device,
        routed_rows_capacity=routed_rows,
        physical_tiles_capacity=physical_tiles,
        task_capacity=max_tasks,
        row_counts=torch.zeros(state_E, dtype=torch.int32, device=device),
        token_map=torch.zeros(rows_padded, dtype=torch.int32, device=device),
        token_weights=torch.zeros(rows_padded, dtype=torch.float32, device=device),
        packed_input=torch.empty(
            1, rows_padded, k // 2, dtype=torch.uint8, device=device
        ),
        packed_input_scale=torch.empty(
            rows_padded, cols_pad_k, dtype=torch.uint8, device=device
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
        expert_write_rows=torch.zeros(state_E, dtype=torch.int32, device=device),
        expert_tile_base=torch.zeros(state_E + 1, dtype=torch.int32, device=device),
        pair_head=torch.zeros(1, dtype=torch.int32, device=device),
        producers_done_count=torch.zeros(1, dtype=torch.int32, device=device),
        all_work_published=torch.zeros(1, dtype=torch.int32, device=device),
        task_head=torch.zeros(1, dtype=torch.int32, device=device),
        task_tail=torch.zeros(1, dtype=torch.int32, device=device),
        task_ready=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_expert=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_m_tile=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_slice_begin=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_slice_count=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        task_valid_rows=torch.zeros(max_tasks, dtype=torch.int32, device=device),
        tile_write_count=torch.zeros(physical_tiles, dtype=torch.int32, device=device),
    )

    # Finalize views
    sf_dtype = cutlass.Float8E4M3FN
    workspace.packed_a_view = workspace.packed_input.permute(1, 2, 0).view(
        torch.float4_e2m1fn_x2
    )
    workspace.packed_a_flat = workspace.packed_input.view(-1)
    workspace.scale_flat = workspace.packed_input_scale.view(-1)
    workspace.sfa_ptr = make_ptr(
        sf_dtype,
        workspace.packed_input_scale.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return workspace


# ---------------------------------------------------------------------------
# Dynamic kernel compilation
# ---------------------------------------------------------------------------


class _DynamicMoELaunch:
    """Thin JIT wrapper that makes num_tokens and max_rows runtime Int32."""

    def __init__(self, kernel, k, num_topk):
        self._kernel = kernel
        self._k = k
        self._half_k = k // 2
        self._num_topk = num_topk
        self._cols_pad_k = _align_up(k // _NVFP4_BLOCK_SIZE, 4)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        topk_ids_ptr: cute.Pointer,
        topk_weights_ptr: cute.Pointer,
        packed_a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        packed_a_storage_ptr: cute.Pointer,
        scale_storage_ptr: cute.Pointer,
        barrier_count: cute.Tensor,
        barrier_epoch: cute.Tensor,
        pair_head: cute.Tensor,
        producers_done_count: cute.Tensor,
        all_work_published: cute.Tensor,
        task_head: cute.Tensor,
        task_tail: cute.Tensor,
        task_ready_ptr: cute.Pointer,
        task_expert_ptr: cute.Pointer,
        task_m_tile_ptr: cute.Pointer,
        task_slice_begin_ptr: cute.Pointer,
        task_slice_count_ptr: cute.Pointer,
        task_valid_rows_ptr: cute.Pointer,
        tile_write_count_ptr: cute.Pointer,
        b_w13: cute.Tensor,
        sfb_w13_ptr: cute.Pointer,
        b_down: cute.Tensor,
        sfb_down_ptr: cute.Pointer,
        row_counts: cute.Tensor,
        expert_write_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        input_global_scale: cute.Tensor,
        alpha: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        scatter_ptr: cute.Pointer,
        token_map_ptr: cute.Pointer,
        token_weights_ptr: cute.Pointer,
        num_tokens: cutlass.Int32,
        max_rows: cutlass.Int32,
        rows_padded: cutlass.Int32,
        max_tasks: cutlass.Int32,
        max_phys_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ):
        a_input = cute.make_tensor(
            a_ptr, layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1))
        )
        topk_ids = cute.make_tensor(
            topk_ids_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        topk_weights_t = cute.make_tensor(
            topk_weights_ptr,
            layout=cute.make_layout((num_tokens * self._num_topk,), stride=(1,)),
        )
        scatter_output = cute.make_tensor(
            scatter_ptr,
            layout=cute.make_layout((num_tokens, self._k), stride=(self._k, 1)),
        )
        packed_a = cute.make_tensor(
            packed_a_ptr,
            layout=cute.make_layout(
                (rows_padded, self._k, 1), stride=(self._k, 1, rows_padded * self._k)
            ),
        )
        packed_a_storage = cute.make_tensor(
            packed_a_storage_ptr,
            layout=cute.make_layout((rows_padded * self._half_k,), stride=(1,)),
        )
        scale_storage = cute.make_tensor(
            scale_storage_ptr,
            layout=cute.make_layout((rows_padded * self._cols_pad_k,), stride=(1,)),
        )
        token_map = cute.make_tensor(
            token_map_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        token_weights_t = cute.make_tensor(
            token_weights_ptr, layout=cute.make_layout((rows_padded,), stride=(1,))
        )
        task_ready = cute.make_tensor(
            task_ready_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_expert = cute.make_tensor(
            task_expert_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_m_tile = cute.make_tensor(
            task_m_tile_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_begin = cute.make_tensor(
            task_slice_begin_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_slice_count = cute.make_tensor(
            task_slice_count_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        task_valid_rows = cute.make_tensor(
            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))
        )
        tile_write_count = cute.make_tensor(
            tile_write_count_ptr,
            layout=cute.make_layout((max_phys_tiles,), stride=(1,)),
        )
        self._kernel(
            a_input,
            topk_ids,
            topk_weights_t,
            packed_a,
            sfa_ptr,
            packed_a_storage,
            scale_storage,
            barrier_count,
            barrier_epoch,
            pair_head,
            producers_done_count,
            all_work_published,
            task_head,
            task_tail,
            task_ready,
            task_expert,
            task_m_tile,
            task_slice_begin,
            task_slice_count,
            task_valid_rows,
            tile_write_count,
            b_w13,
            sfb_w13_ptr,
            b_down,
            sfb_down_ptr,
            row_counts,
            expert_write_rows,
            expert_tile_base,
            input_global_scale,
            alpha,
            down_alpha,
            global_scale,
            scatter_output,
            token_map,
            token_weights_t,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )


_DYNAMIC_KERNEL_CACHE: Dict[Tuple, Tuple] = {}


def _get_dynamic_kernel(
    E: int,
    m: int,
    k: int,
    n: int,
    num_topk: int,
    max_rows: int,
    *,
    topk_ids_dtype: torch.dtype = torch.int32,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
):
    """Compile (or retrieve cached) the SM120 dynamic MoE kernel."""
    sf_vec_size = 16
    sm_count = get_num_sm(torch.device("cuda"))
    mac = min(get_max_active_clusters(1), sm_count)

    cache_key = (
        "dynamic",
        E,
        k,
        n,
        num_topk,
        mac,
        topk_ids_dtype,
        input_scales_are_reciprocal,
        fast_math,
        activation,
    )
    cached = _DYNAMIC_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    is_gated = activation == "silu"
    w1_rows = (2 if is_gated else 1) * n

    ab_dtype = cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN
    a_dtype = cutlass.BFloat16
    alpha_dtype = cutlass.Float32

    kernel = MoEDynamicKernel(
        sf_vec_size=sf_vec_size,
        mma_tiler_mn=(_LEVEL_TILE_M, _LEVEL_TILE_N),
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
    )
    launch = _DynamicMoELaunch(kernel, k=k, num_topk=num_topk)

    topk_ids_cutlass_dtype = (
        cutlass.Int32 if topk_ids_dtype == torch.int32 else cutlass.Int64
    )
    topk_ids_align = 4 if topk_ids_dtype == torch.int32 else 8

    # Runtime-shaped tensors passed as pointers
    a_input_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    topk_ids_fake = make_ptr(
        topk_ids_cutlass_dtype,
        topk_ids_align,
        cute.AddressSpace.gmem,
        assumed_align=topk_ids_align,
    )
    topk_weights_fake = make_ptr(
        cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    packed_a_fake = make_ptr(ab_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    sfa_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    packed_a_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )
    scale_storage_fake = make_ptr(
        cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    barrier_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    pair_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    producers_done_count_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    all_work_published_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_head_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )
    task_tail_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (1,), assumed_align=4
    )

    task_ready_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_expert_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_m_tile_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_begin_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_slice_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    task_valid_rows_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )
    tile_write_count_fake = make_ptr(
        cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4
    )

    b_w13_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (w1_rows, k, E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_w13_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    b_down_fake = cute.runtime.make_fake_compact_tensor(
        ab_dtype,
        (k, n, E),
        stride_order=(1, 0, 2),
        assumed_align=16,
    )
    sfb_down_fake = make_ptr(sf_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    row_counts_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4
    )
    expert_write_rows_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E,), assumed_align=4
    )
    expert_tile_base_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (E + 1,), assumed_align=4
    )
    input_gs_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    down_alpha_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    global_scale_fake = cute.runtime.make_fake_compact_tensor(
        alpha_dtype, (E,), assumed_align=16
    )
    scatter_fake = make_ptr(a_dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
    token_map_fake = make_ptr(cutlass.Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    token_weights_fake = make_ptr(
        alpha_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
    )

    compiled = cute.compile(
        launch,
        a_input_fake,
        topk_ids_fake,
        topk_weights_fake,
        packed_a_fake,
        sfa_fake,
        packed_a_storage_fake,
        scale_storage_fake,
        barrier_count_fake,
        barrier_epoch_fake,
        pair_head_fake,
        producers_done_count_fake,
        all_work_published_fake,
        task_head_fake,
        task_tail_fake,
        task_ready_fake,
        task_expert_fake,
        task_m_tile_fake,
        task_slice_begin_fake,
        task_slice_count_fake,
        task_valid_rows_fake,
        tile_write_count_fake,
        b_w13_fake,
        sfb_w13_fake,
        b_down_fake,
        sfb_down_fake,
        row_counts_fake,
        expert_write_rows_fake,
        expert_tile_base_fake,
        input_gs_fake,
        alpha_fake,
        down_alpha_fake,
        global_scale_fake,
        scatter_fake,
        token_map_fake,
        token_weights_fake,
        1,
        1,
        1,
        1,
        1,  # runtime Int32 placeholders
        mac,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )

    result = (compiled, mac)
    _DYNAMIC_KERNEL_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Dynamic launch
# ---------------------------------------------------------------------------
def launch_sm120_dynamic_moe(
    *,
    workspace: Sm120DynamicMoEWorkspace,
    weights: _WeightViews,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    input_gs: torch.Tensor,
    down_input_scale: torch.Tensor,
    scatter_output: torch.Tensor,
    num_experts: int,
    num_tokens: int,
    k: int,
    n: int,
    top_k: int,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
) -> torch.Tensor:
    """Launch the SM120 dynamic MoE kernel."""
    flat_ids = topk_ids.view(-1).to(torch.int32)
    flat_weights = topk_weights.view(-1).to(torch.float32)

    # Broadcast scalar scales to per-expert [E] tensors
    input_gs = _expand_to_experts(input_gs, num_experts)
    down_input_scale = _expand_to_experts(down_input_scale, num_experts)

    compiled, mac = _get_dynamic_kernel(
        num_experts,
        num_tokens,
        k,
        n,
        top_k,
        workspace.max_rows,
        topk_ids_dtype=torch.int32,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        fast_math=fast_math,
        activation=activation,
    )

    # Dynamic kernel: runtime-shaped args are DataPointer (pass data_ptr()),
    # fixed-shape args are Tensor (pass torch tensor directly).
    compiled(
        a.data_ptr(),
        flat_ids.data_ptr(),
        flat_weights.data_ptr(),
        workspace.packed_a_view.data_ptr(),
        workspace.packed_input_scale.data_ptr(),
        workspace.packed_a_flat.data_ptr(),
        workspace.scale_flat.data_ptr(),
        workspace.barrier_count,
        workspace.barrier_epoch,
        workspace.pair_head,
        workspace.producers_done_count,
        workspace.all_work_published,
        workspace.task_head,
        workspace.task_tail,
        workspace.task_ready.data_ptr(),
        workspace.task_expert.data_ptr(),
        workspace.task_m_tile.data_ptr(),
        workspace.task_slice_begin.data_ptr(),
        workspace.task_slice_count.data_ptr(),
        workspace.task_valid_rows.data_ptr(),
        workspace.tile_write_count.data_ptr(),
        weights.w13_fp4,
        weights._w13_sf_storage.data_ptr(),
        weights.down_fp4,
        weights._down_sf_storage.data_ptr(),
        workspace.row_counts,
        workspace.expert_write_rows,
        workspace.expert_tile_base,
        input_gs,
        weights.w1_alpha,
        weights.w2_alpha,
        down_input_scale,
        scatter_output.data_ptr(),
        workspace.token_map.data_ptr(),
        workspace.token_weights.data_ptr(),
        num_tokens,
        workspace.max_rows,
        workspace.physical_tiles_capacity * _LEVEL_TILE_M,
        workspace.task_capacity,
        workspace.physical_tiles_capacity,
        mac,
    )

    return scatter_output


# ==========================================================================
# Workspace cache (for functional API path)
# ==========================================================================

_Sm120Workspace = Union[Sm120StaticMoEWorkspace, Sm120DynamicMoEWorkspace]

# Keyed by (state_E, weight_E, k, n, top_k, device, backend).
# Stores the workspace with the largest max_rows seen for each key.
# Grows monotonically — never shrinks within a process.
_WORKSPACE_CACHE: Dict[Tuple, _Sm120Workspace] = {}


def _get_cached_workspace(
    *,
    backend: str,
    state_E: int,
    weight_E: int,
    routed_rows: int,
    k: int,
    n: int,
    num_topk: int,
    device: torch.device,
) -> _Sm120Workspace:
    """Get or allocate a cached workspace for the given problem shape.

    Reuses the cached workspace if it has enough capacity for the requested
    routed_rows. For static workspaces, max_rows is the direct capacity.
    For dynamic workspaces, routed_rows_capacity is used because the dynamic
    geometry (physical tiles, task queue slots) depends on the original
    routed_rows, not just max_rows.
    """
    cache_key = (state_E, weight_E, k, n, num_topk, str(device), backend)
    cached = _WORKSPACE_CACHE.get(cache_key)

    if cached is not None:
        if backend == "dynamic":
            if cached.routed_rows_capacity >= max(1, routed_rows):  # type: ignore[union-attr]
                return cached
        else:
            if cached.max_rows >= max(1, routed_rows):
                return cached

    # Allocate new workspace (or grow existing)
    workspace: _Sm120Workspace
    if backend == "dynamic":
        workspace = allocate_sm120_dynamic_workspace(
            state_E=state_E,
            weight_E=weight_E,
            routed_rows=routed_rows,
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
        )
    else:
        workspace = allocate_sm120_static_workspace(
            state_E=state_E,
            weight_E=weight_E,
            max_rows=max(1, routed_rows),
            k=k,
            n=n,
            num_topk=num_topk,
            device=device,
        )

    _WORKSPACE_CACHE[cache_key] = workspace
    return workspace


# ==========================================================================
# Unified dispatch
# ==========================================================================
def launch_sm120_moe(
    *,
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w1_weight_sf: torch.Tensor,
    w1_alpha: torch.Tensor,
    fc2_input_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_sf: torch.Tensor,
    w2_alpha: torch.Tensor,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    scatter_output: torch.Tensor,
    input_scales_are_reciprocal: bool = False,
    fast_math: bool = True,
    activation: str = "silu",
    _workspace=None,
    _weight_views=None,
) -> torch.Tensor:
    """Unified SM120 MoE dispatch — selects static or dynamic by token count.

    Optional _workspace and _weight_views can be pre-allocated and reused
    across calls to avoid per-call allocation overhead (wrapper path).
    When not provided (functional API path), a module-level workspace cache
    is used to avoid re-allocating on every call.
    """
    num_tokens = topk_ids.size(0)
    k = a.size(1)  # hidden_size
    is_gated = activation == "silu"
    # w1_weight.size(1) is 2*n for gated or n for non-gated
    intermediate_size = w1_weight.size(1) // 2 if is_gated else w1_weight.size(1)
    n = intermediate_size
    routed_rows = num_tokens * top_k

    weights = (
        _weight_views
        if _weight_views is not None
        else _get_weight_views(
            w1_fp4=w1_weight,
            w1_blockscale=w1_weight_sf,
            w2_fp4=w2_weight,
            w2_blockscale=w2_weight_sf,
            w1_alphas=w1_alpha,
            w2_alphas=w2_alpha,
            n=n,
            k=k,
        )
    )

    # Resolve workspace and backend selection.
    # When a pre-allocated workspace is provided (CUDA graph wrapper path),
    # infer the backend from the workspace type so they stay in sync —
    # the caller already committed to a backend at allocation time.
    if _workspace is not None:
        workspace = _workspace
        if isinstance(workspace, Sm120DynamicMoEWorkspace):
            backend = "dynamic"
        else:
            backend = "static"
    else:
        backend = select_sm120_moe_backend(num_tokens=num_tokens, num_topk=top_k)
        # The dynamic kernel indexes row_counts/expert_write_rows directly with
        # topk_ids but those buffers are sized with num_local_experts. Unless
        # num_local_experts == num_experts, fall back to the static backend which
        # has global-to-local expert remapping.
        if backend == "dynamic" and num_local_experts != num_experts:
            backend = "static"
        workspace = _get_cached_workspace(
            backend=backend,
            state_E=num_local_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=k,
            n=n,
            num_topk=top_k,
            device=a.device,
        )

    if backend == "dynamic":
        return launch_sm120_dynamic_moe(
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=w1_alpha,
            down_input_scale=fc2_input_scale,
            scatter_output=scatter_output,
            num_experts=num_experts,
            num_tokens=num_tokens,
            k=k,
            n=n,
            top_k=top_k,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
        )
    else:
        return launch_sm120_static_moe(
            workspace=workspace,
            weights=weights,
            a=a,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            input_gs=w1_alpha,
            down_input_scale=fc2_input_scale,
            scatter_output=scatter_output,
            num_experts=num_experts,
            num_tokens=num_tokens,
            k=k,
            n=n,
            top_k=top_k,
            input_scales_are_reciprocal=input_scales_are_reciprocal,
            fast_math=fast_math,
            activation=activation,
        )
