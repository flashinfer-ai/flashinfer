# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context launch geometry, hardware defaults and scheduler selection."""

import math
import torch
from dataclasses import dataclass
from ...common.hardware import get_max_active_clusters
from ...common.paged_kv import validate_native_hnd_tma_strides
from .validation import (
    CONTEXT_KV_TILE_N,
    CONTEXT_TILE_SIZE_Q,
    CUDA_GRID_YZ_MAX,
    read_indptr,
    read_int32_values,
    resolve_context_device,
    validate_base_tensors,
    validate_compact,
    validate_context_dtype_pair,
    validate_context_head_dim,
    validate_context_head_geometry,
    validate_context_mask,
    validate_context_page_size,
    validate_context_sage_config,
    validate_context_window_left,
    validate_device,
    validate_extent,
    validate_indptr_tensor,
    validate_output_dtype,
    validate_padded_data_extent,
    validate_paged_dtype_pair,
    validate_paged_metadata_tensor,
    validate_paged_runtime_inputs,
    validate_paged_runtime_metadata,
    validate_static_extent,
    validate_value_layout,
)
from ..sage.config import SageAttentionConfig, validate_int8_qk_device
from ..sage.layout import SageScaleLayout
from ..vc_attention import VCAttentionConfig
from typing import Literal, Optional


ContextScheduler = Literal[
    "nonpersistent",
    "static_persistent",
    "clc_dynamic_persistent",
]


@dataclass(frozen=True)
class ContextGeometry:
    """Validated request geometry used by the contiguous one-shot adapter."""

    device: torch.device
    device_index: int
    packed_query: bool
    batch_size: int
    total_q: int
    total_k: int
    max_q_len: int
    max_seq_len_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype: torch.dtype
    pv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool
    q_shape: tuple[int, ...]
    kv_shape: tuple[int, ...]
    head_dim_vo: int | None = None


@dataclass(frozen=True)
class ContextPlanGeometry:
    """Static compile geometry for one reusable contiguous context plan."""

    device: torch.device
    device_index: int
    packed_query: bool
    batch_size: int
    max_q_len: int
    max_seq_len_k: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype: torch.dtype
    pv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    causal_single_kv_tile: bool
    packed_dense_k_mask: bool
    head_dim_vo: int | None = None
    # VC-Attention recipe, or ``None`` for a plain plan.
    vc: Optional[VCAttentionConfig] = None
    # The plan's Sage recipe, part of the compile identity; ``None`` without
    # Sage attention.
    sage: SageAttentionConfig | None = None


@dataclass(frozen=True)
class ContextSchedulerProbe:
    """Static kernel traits and device capacity used by scheduler policy."""

    single_qkv_instance: bool
    logical_work_tiles: int
    max_active_clusters: int


@dataclass(frozen=True)
class PagedContextPlanGeometry:
    """Static compile geometry for one reusable paged plan."""

    device: torch.device
    device_index: int
    batch_size: int
    max_q_len: int
    max_kv_len: int
    page_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    qk_dtype: torch.dtype
    pv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: str
    window_left: int
    head_paired: bool
    uniform_packed_lengths: bool
    has_q_offset: bool
    paged_v_tail_is_zero: bool
    packed_dense_k_mask: bool


def contiguous_context_uses_clc_scheduler(
    *,
    single_qkv_instance: bool,
    head_paired: bool,
    packed_query: bool,
    uniform_packed_lengths: bool,
    is_causal: bool,
    has_q_offset: bool,
) -> bool:
    """Return whether contiguous persistence needs dynamic work discovery.

    Paired and zero-offset triangular task graphs use CLC to distribute their
    heavier work tiles. A bottom-right-offset single-instance fixed or
    uniform-packed plan has a near-uniform immutable domain, so its persistent
    CTAs can advance through the static queue without a CLC request/response
    on every tile. General packed plans retain CLC because live cumulative
    offsets can change the active query-tile domain.
    """

    return (
        head_paired
        or not single_qkv_instance
        or (packed_query and not uniform_packed_lengths)
        or (is_causal and not has_q_offset)
    )


def contiguous_context_uses_persistent_scheduler(
    *,
    single_qkv_instance: bool,
    head_paired: bool,
    packed_query: bool,
    uniform_packed_lengths: bool,
    logical_work_tiles: int,
    max_active_clusters: int,
    batch_size: int,
    num_qo_heads: int,
    is_causal: bool,
    has_q_offset: bool,
) -> bool:
    """Return whether contiguous work needs a persistent launch.

    Paired, live-ragged, and zero-offset triangular task graphs keep dynamic
    persistence. Near-uniform bottom-right-offset single-instance domains
    launch directly when they fit in one resident wave and use static
    persistence only when more work remains. Oversized logical Y/Z dimensions
    also require the flattened persistent grid.
    """

    return (
        head_paired
        or not single_qkv_instance
        or (packed_query and not uniform_packed_lengths)
        or (is_causal and not has_q_offset)
        or logical_work_tiles > max_active_clusters
        or batch_size > CUDA_GRID_YZ_MAX
        or num_qo_heads > CUDA_GRID_YZ_MAX
    )


def default_exp2_fma_pairs(device_index: int, cfg) -> int:
    """FMA-pipe exp2 pairs per 16-pair softmax chunk on SM100.

    Measured on B200. The paired D128 dense path is MUFU bound: with 16-bit V
    it takes 4, and with 8-bit V and P in SMEM, fixed-length plans take 3.
    Packed and paged 8-bit plans spend the FMA pipe on their masks and take 0.
    Single-QKV dense takes 2. Causal is issue bound and takes 0. Callers can
    set cfg.exp2_fma_pairs to any value on any path."""
    if torch.cuda.get_device_capability(device_index) != (10, 0) or cfg.is_causal:
        return 0
    if cfg.v_dtype.width == 8:
        return 0 if cfg.has_varlen or cfg.single_qkv_instance else 3
    if cfg.v_dtype.width != 16:
        return 0
    if cfg.pv_half_overlap:
        return 4
    return 2 if cfg.single_qkv_instance else 0


def default_exp2_packed_f16x2(device_index: int, cfg) -> bool:
    """Softmax exp2 as one packed f16x2 MUFU op per score pair, on for the SM107
    paths with 8-bit V. SM100 and SM103 split the packed op into two MUFU ops and
    keep the scalar form."""
    return (
        torch.cuda.get_device_capability(device_index) == (10, 7)
        and cfg.v_dtype.width == 8
    )


def default_fp8_psmem_early_token(device_index: int) -> bool:
    """Release the S0/S1 token early on the fp8 SMEM-P path, on for SM103."""
    return torch.cuda.get_device_capability(device_index) == (10, 3)


def default_kv_ring_stages(device_index: int, qk_bits: int, pv_bits: int) -> int:
    """K/V ring depth of the SM107 dense paths with 8-bit V, 0 elsewhere for the
    budget-derived depth. Deeper rings stall the softmax P stores behind the TMA
    fills, so the fp8 K ring stops at 6 stages, and the bf16 K ring takes the 4
    stages the larger SM107 shared memory fits."""
    if torch.cuda.get_device_capability(device_index) == (10, 7) and pv_bits == 8:
        return 6 if qk_bits == 8 else 4
    return 0


def default_two_cta_umma(device_index: int, qk_bits: int) -> bool:
    """Two-CTA UMMA for the dense paired D128 context kernel. On for SM103, where
    the kernel is tensor-pipe bound and halving the UMMA issues and the K/V
    traffic per SM pays. On SM107 only for bf16 QK, which is shared-memory-port
    bound on operand fetch, since fp8 QK is slower with it."""
    capability = torch.cuda.get_device_capability(device_index)
    if capability == (10, 3):
        return True
    return capability == (10, 7) and qk_bits == 16


def default_umma_fp8_k64(device_index: int, qk_bits: int) -> bool:
    """K=64 fp8 UMMA instructions for the PV of the SM107 bf16-QK paths. The
    fp8-QK paths are shared-memory-port bound on operand fetch and keep K=32."""
    return torch.cuda.get_device_capability(device_index) == (10, 7) and qk_bits == 16


def default_uses_ldtm_stat(device_index: int, qk_bits: int, pv_bits: int) -> bool:
    """Enable LDTM.STAT where the shared tcgen05 gate allows it. SM107 keeps it
    off for the dense QK-BF16 PV-FP8 path, which is faster with the register
    row max."""
    from ...device.tcgen05 import ldtm_stat_supported

    capability = torch.cuda.get_device_capability(device_index)
    if capability == (10, 7) and qk_bits == 16 and pv_bits == 8:
        return False
    return ldtm_stat_supported(capability)


def derive_has_q_offset(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    mask_type: str,
) -> bool:
    """Return whether any request has a bottom-right causal Q offset."""

    if mask_type != "causal":
        return False
    return any(
        k_length != q_length
        for q_length, k_length in zip(q_lengths, k_lengths, strict=True)
    )


def make_context_kernel(
    *,
    device_index: int,
    input_qk_dtype,
    input_pv_dtype,
    out_dtype,
    head_dim: int,
    head_dim_vo: int | None = None,
    mask_type: str,
    window_left: int,
    head_paired: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    has_q_offset: bool,
    causal_single_kv_tile: bool,
    scheduler: ContextScheduler,
    uses_ldtm_stat: bool,
    fp8_psmem_early_token: bool = False,
    kv_ring_stages: int = 0,
    umma_fp8_k64: bool = False,
    two_cta_umma: bool = False,
    page_size: int | None = None,
    max_kv_len: int | None = None,
    vc_attention: str | None = None,
    sage_scale_layout: SageScaleLayout | None = None,
):
    """Build one context kernel from its batch-independent static topology."""

    import cutlass

    from .kernel.kernel import FmhaTs

    use_paged_kv = page_size is not None
    # VC QK8 also fixes the K-scale extent through max_kv_len. Paged runtime
    # context only needs page geometry; its live KV bound is a launch operand.
    is_persistent = scheduler != "nonpersistent"
    is_clc_dynamic = scheduler == "clc_dynamic_persistent"
    paged_kwargs = (
        {"use_paged_kv": True, "num_tokens_per_page": page_size} if use_paged_kv else {}
    )
    if max_kv_len is not None:
        paged_kwargs["max_kv_len"] = max_kv_len
    fmha = FmhaTs(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        in_qk_dtype=input_qk_dtype,
        in_pv_dtype=input_pv_dtype,
        out_dtype=out_dtype,
        d=head_dim,
        d_v=head_dim_vo,
        is_persistent=is_persistent,
        fp8_psmem_early_token=fp8_psmem_early_token,
        kv_ring_stages=kv_ring_stages,
        umma_fp8_k64=umma_fp8_k64,
        two_cta_umma=two_cta_umma,
        is_causal=mask_type == "causal",
        has_variable_window=mask_type == "variable_window",
        balance_causal_workload=(
            scheduler == "static_persistent"
            and uses_heavy_first_static_causal_raster(
                mask_type=mask_type,
                window_left=window_left,
                has_q_offset=has_q_offset,
            )
        ),
        is_clc_dynamic=is_clc_dynamic,
        head_paired=head_paired,
        window_size_left=window_left if window_left > 0 else 0,
        h_r=num_qo_heads // num_kv_heads,
        enable_skip_correction=True,
        uses_ldtm_stat=uses_ldtm_stat,
        causal_single_kv_tile=(causal_single_kv_tile and not use_paged_kv),
        vc_attention=vc_attention,
        use_blackwell_tma_oob_workaround=(
            torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3))
        ),
        sage_scale_layout=sage_scale_layout,
        **paged_kwargs,
    )
    return fmha


def make_context_scheduler_probe(
    geometry: ContextPlanGeometry | PagedContextPlanGeometry,
    *,
    causal_single_kv_tile: bool,
    page_size: int | None = None,
    max_kv_len: int | None = None,
) -> ContextSchedulerProbe:
    """Build the common static probe and logical work count for policy."""

    import cutlass

    dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e8m0fnu: cutlass.Float8E8M0FNU,
        torch.int8: cutlass.Int8,
    }
    probe = make_context_kernel(
        device_index=geometry.device_index,
        input_qk_dtype=dtype_map[geometry.qk_dtype],
        input_pv_dtype=dtype_map[geometry.pv_dtype],
        out_dtype=dtype_map[geometry.out_dtype],
        head_dim=geometry.head_dim,
        head_dim_vo=getattr(geometry, "head_dim_vo", None),
        mask_type=geometry.mask_type,
        window_left=geometry.window_left,
        head_paired=geometry.head_paired,
        num_qo_heads=geometry.num_qo_heads,
        num_kv_heads=geometry.num_kv_heads,
        has_q_offset=geometry.has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        scheduler="static_persistent",
        uses_ldtm_stat=default_uses_ldtm_stat(
            geometry.device_index,
            geometry.qk_dtype.itemsize * 8,
            geometry.pv_dtype.itemsize * 8,
        ),
        page_size=page_size,
        max_kv_len=max_kv_len,
    )
    with torch.cuda.device(geometry.device_index):
        max_active_clusters = get_max_active_clusters(1)
    num_seq_tiles = (
        geometry.max_q_len + probe.cfg.cta_tiler[0] - 1
    ) // probe.cfg.cta_tiler[0]
    num_head_tiles = geometry.num_qo_heads // probe.cfg.work_tile_q_heads
    logical_work_tiles = geometry.batch_size * num_seq_tiles * num_head_tiles
    return ContextSchedulerProbe(
        single_qkv_instance=bool(probe.cfg.single_qkv_instance),
        logical_work_tiles=int(logical_work_tiles),
        max_active_clusters=max_active_clusters,
    )


def make_context_value_fake(dtype, shape):
    """Use one compiled V layout for compact and row-padded runtime tensors."""

    import cutlass.cute as cute

    alignment = 128 // dtype.width
    outer_strides = tuple(cute.sym_int64(divisibility=alignment) for _ in shape[:-2])
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=(*outer_strides, shape[-1], 1),
        assumed_align=16,
    )


def needs_packed_dense_k_mask(
    *,
    packed_query: bool,
    mask_type: str,
    k_lengths: tuple[int, ...],
) -> bool:
    """Return whether packed-contiguous or paged dense K needs right masking.

    Uniform K lengths that are exact K-tile multiples give every request the
    same full-tile domain.  That specialization needs no runtime ``seqlen_k``
    mask; mixed lengths or a partial final tile retain the general mask.
    """

    return (
        packed_query
        and mask_type == "dense"
        and (
            any(length != k_lengths[0] for length in k_lengths[1:])
            or k_lengths[0] % CONTEXT_KV_TILE_N != 0
        )
    )


def paged_context_uses_clc_scheduler(
    *,
    is_persistent: bool,
    single_qkv_instance: bool,
    is_causal: bool,
    uniform_packed_lengths: bool,
) -> bool:
    """Return whether a persistent paged plan needs the dynamic CLC queue."""

    return is_persistent and (
        not single_qkv_instance or (is_causal and not uniform_packed_lengths)
    )


def paged_context_uses_persistent_scheduler(
    *,
    mask_type: str,
    head_paired: bool,
    logical_work_tiles: int,
    max_active_clusters: int,
    batch_size: int,
    num_qo_heads: int,
) -> bool:
    """Return whether paged context should use the static persistent grid.

    A one-wave dense query-paired grid has no inter-wave launch imbalance to
    amortize, so it launches one CTA per logical tile.  Larger grids use the
    persistent scheduler to balance work across resident CTAs.  CUDA limits
    grid Y and Z to 65,535; retaining persistent scheduling for oversized
    public geometries keeps every otherwise-valid int32 plan launchable.
    """

    return (
        mask_type == "causal"
        or head_paired
        or logical_work_tiles > max_active_clusters
        or batch_size > CUDA_GRID_YZ_MAX
        or num_qo_heads > CUDA_GRID_YZ_MAX
    )


def resolve_context_plan_geometry(
    *,
    device: int | str | torch.device,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    head_dim_vo: int | None = None,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    packed_query: bool,
    mask_type: str,
    window_left: int,
    out_dtype: torch.dtype,
    sage: SageAttentionConfig | None = None,
) -> ContextPlanGeometry:
    """Validate explicit static bounds for a reusable contiguous plan."""

    validate_context_dtype_pair(qk_dtype, qk_dtype, pv_dtype, out_dtype)
    validate_context_mask(mask_type)
    window_left = validate_context_window_left(window_left, mask_type)
    if not isinstance(packed_query, bool):
        raise TypeError("packed_query must be a bool")
    if packed_query and mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' requires fixed [B, S, H, D] tensors"
        )
    if qk_dtype == torch.float8_e8m0fnu and (packed_query or window_left > 0):
        raise NotImplementedError(
            "MXFP8 Q/K requires fixed [B, S, H, D] tensors and no left window"
        )

    batch_size = validate_static_extent(batch_size, "batch_size")
    max_q_len = validate_static_extent(max_q_len, "max_q_len")
    max_kv_len = validate_static_extent(max_kv_len, "max_kv_len")
    num_qo_heads = validate_static_extent(num_qo_heads, "num_qo_heads")
    num_kv_heads = validate_static_extent(num_kv_heads, "num_kv_heads")
    head_dim = validate_static_extent(head_dim, "head_dim")
    head_dim_vo = (
        head_dim
        if head_dim_vo is None
        else validate_static_extent(head_dim_vo, "head_dim_vo")
    )
    device, device_index = resolve_context_device(device)

    validate_padded_data_extent(batch_size * max_q_len, "batch_size * max_q_len")
    validate_padded_data_extent(batch_size * max_kv_len, "batch_size * max_kv_len")
    head_ratio = validate_context_head_geometry(num_qo_heads, num_kv_heads)
    validate_context_head_dim(head_dim, head_dim, head_dim_vo)
    if (head_dim, head_dim_vo) == (192, 128) and window_left > 0:
        raise NotImplementedError(
            "QK/V=(192, 128) does not support a positive left window"
        )
    if not packed_query and mask_type == "causal" and max_q_len > max_kv_len:
        raise ValueError(
            "bottom-right causal context requires max_q_len <= max_kv_len; "
            f"got {max_q_len} and {max_kv_len}"
        )
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )

    # Packed request lengths are run inputs, so select conservative compile
    # facts that remain valid for every request within the plan capacities.
    has_q_offset = mask_type == "causal" and (packed_query or max_q_len != max_kv_len)
    geometry = ContextPlanGeometry(
        device=device,
        device_index=device_index,
        packed_query=packed_query,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_seq_len_k=max_kv_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dim_vo=head_dim_vo,
        qk_dtype=qk_dtype,
        pv_dtype=pv_dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=False,
        has_q_offset=has_q_offset,
        # The single-tile causal form keeps P in tensor memory, which MXFP8
        # Q/K cannot use.
        causal_single_kv_tile=(
            mask_type == "causal"
            and not packed_query
            and not head_paired
            and max_q_len == max_kv_len
            and max_kv_len <= CONTEXT_KV_TILE_N
            and qk_dtype != torch.float8_e8m0fnu
        ),
        packed_dense_k_mask=packed_query and mask_type == "dense",
        sage=sage,
    )
    if sage is not None:
        validate_context_sage_config(geometry, sage)
    # The recipe is rejected before the device, as in block-sparse plans.
    validate_int8_qk_device(qk_dtype, device_index)
    return geometry


def resolve_context_scheduler(geometry: ContextPlanGeometry) -> ContextScheduler:
    """Select a scheduler from plan-time work while keeping batch out of JIT."""

    probe = make_context_scheduler_probe(
        geometry,
        causal_single_kv_tile=geometry.causal_single_kv_tile,
    )
    if two_cta_umma_geometry_eligible(geometry):
        # Two-CTA pairs Q tiles through the grid, so no persistent scheduler.
        return "nonpersistent"
    is_persistent = contiguous_context_uses_persistent_scheduler(
        single_qkv_instance=probe.single_qkv_instance,
        head_paired=geometry.head_paired,
        packed_query=geometry.packed_query,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        logical_work_tiles=probe.logical_work_tiles,
        max_active_clusters=probe.max_active_clusters,
        batch_size=geometry.batch_size,
        num_qo_heads=geometry.num_qo_heads,
        is_causal=geometry.mask_type == "causal",
        has_q_offset=geometry.has_q_offset,
    )
    if not is_persistent:
        return "nonpersistent"
    # Fixed causal requests with one work item per head all traverse the same
    # KV domain. Static persistence avoids CLC traffic without a load-balance
    # penalty; live packed offsets and multiple Q work items still use CLC.
    if (
        geometry.mask_type == "causal"
        and not geometry.packed_query
        and not geometry.head_paired
        and geometry.window_left <= 0
        and probe.logical_work_tiles == geometry.batch_size * geometry.num_qo_heads
    ):
        return "static_persistent"
    if contiguous_context_uses_clc_scheduler(
        single_qkv_instance=probe.single_qkv_instance,
        head_paired=geometry.head_paired,
        packed_query=geometry.packed_query,
        uniform_packed_lengths=geometry.uniform_packed_lengths,
        is_causal=geometry.mask_type == "causal",
        has_q_offset=geometry.has_q_offset,
    ):
        return "clc_dynamic_persistent"
    return "static_persistent"


def resolve_geometry(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qo_indptr: Optional[torch.Tensor],
    kv_indptr: Optional[torch.Tensor],
    mask_type: str,
    window_left: int,
    out_dtype: torch.dtype,
) -> ContextGeometry:
    """Validate one-shot inputs and derive their static wrapper bounds."""

    validate_base_tensors(q, k, v, allow_mla=True)
    validate_output_dtype(out_dtype)
    validate_context_mask(mask_type)
    window_left = validate_context_window_left(window_left, mask_type)
    device_index = validate_device(q.device)
    device = torch.device("cuda", device_index)
    if (qo_indptr is None) != (kv_indptr is None):
        raise ValueError("qo_indptr and kv_indptr must be provided together")
    packed_query = qo_indptr is not None

    if packed_query:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError(
                "packed Q/K/V must use [total_tokens, H, D] storage; "
                f"got q rank {q.ndim} and k rank {k.ndim}"
            )
        total_q, num_qo_heads, q_head_dim = map(int, q.shape)
        total_k, num_kv_heads, kv_head_dim = map(int, k.shape)
        validate_padded_data_extent(total_q, "total_q")
        validate_padded_data_extent(total_k, "total_k")
        validate_compact(q, "q", "[total_q, Hq, D]")
        validate_compact(k, "k", "[total_k, Hkv, D]")
        validate_value_layout(v)
        assert qo_indptr is not None and kv_indptr is not None
        validate_indptr_tensor(qo_indptr, "qo_indptr", device=device)
        validate_indptr_tensor(kv_indptr, "kv_indptr", device=device)
        if qo_indptr.numel() != kv_indptr.numel():
            raise ValueError(
                "qo_indptr and kv_indptr must describe the same batch; "
                f"got {qo_indptr.numel() - 1} and {kv_indptr.numel() - 1} rows"
            )
        _, q_lengths = read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
        _, k_lengths = read_indptr(kv_indptr, "kv_indptr", expected_total=total_k)
        batch_size = len(q_lengths)
        validate_extent(batch_size, "batch_size")
        max_q_len = max(q_lengths)
        max_seq_len_k = max(k_lengths)
        q_shape = tuple(q.shape)
        kv_shape = tuple(k.shape)
    else:
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(
                "fixed Q/K/V must use [B, S, H, D] storage; "
                f"got q rank {q.ndim} and k rank {k.ndim}"
            )
        batch_size, max_q_len, num_qo_heads, q_head_dim = map(int, q.shape)
        k_batch, max_seq_len_k, num_kv_heads, kv_head_dim = map(int, k.shape)
        if batch_size != k_batch:
            raise ValueError(
                f"q and k batch dimensions must match; got {batch_size} and {k_batch}"
            )
        validate_extent(batch_size, "batch_size")
        validate_extent(max_q_len, "seq_len_q")
        validate_extent(max_seq_len_k, "seq_len_k")
        total_q = batch_size * max_q_len
        total_k = batch_size * max_seq_len_k
        validate_padded_data_extent(total_q, "B * seq_len_q")
        validate_padded_data_extent(total_k, "B * seq_len_k")
        validate_compact(q, "q", "[B, Sq, Hq, D]")
        validate_compact(k, "k", "[B, Sk, Hkv, D]")
        validate_value_layout(v)
        q_lengths = (max_q_len,) * batch_size
        k_lengths = (max_seq_len_k,) * batch_size
        q_shape = tuple(q.shape)
        kv_shape = tuple(k.shape)

    validate_context_head_dim(q_head_dim, kv_head_dim, int(v.shape[-1]))
    head_ratio = validate_context_head_geometry(num_qo_heads, num_kv_heads)
    if mask_type == "causal":
        for batch_idx, (q_length, k_length) in enumerate(
            zip(q_lengths, k_lengths, strict=True)
        ):
            if k_length > 0 and q_length > k_length:
                raise ValueError(
                    "bottom-right causal context requires Sq <= Sk for each "
                    f"request; got batch {batch_idx}: Sq={q_length}, Sk={k_length}"
                )

    if (q_head_dim, int(v.shape[-1])) == (192, 128) and window_left > 0:
        raise NotImplementedError("192/128 context does not support a left window")
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )
    has_q_offset = derive_has_q_offset(q_lengths, k_lengths, mask_type)
    uniform_packed_lengths = (
        packed_query
        and all(length == q_lengths[0] for length in q_lengths[1:])
        and all(length == k_lengths[0] for length in k_lengths[1:])
    )
    causal_single_kv_tile = (
        mask_type == "causal"
        and not packed_query
        and not head_paired
        and max_q_len == max_seq_len_k
        and max_seq_len_k <= 128
    )
    packed_dense_k_mask = needs_packed_dense_k_mask(
        packed_query=packed_query,
        mask_type=mask_type,
        k_lengths=k_lengths,
    )
    return ContextGeometry(
        device=device,
        device_index=device_index,
        packed_query=packed_query,
        batch_size=batch_size,
        total_q=total_q,
        total_k=total_k,
        max_q_len=max_q_len,
        max_seq_len_k=max_seq_len_k,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        head_dim_vo=int(v.shape[-1]),
        qk_dtype=q.dtype,
        pv_dtype=v.dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
        causal_single_kv_tile=causal_single_kv_tile,
        packed_dense_k_mask=packed_dense_k_mask,
        q_shape=q_shape,
        kv_shape=kv_shape,
    )


def resolve_paged_context_scheduler(
    geometry: PagedContextPlanGeometry,
) -> ContextScheduler:
    """Select paged-context scheduling from plan work, not JIT identity."""

    probe = make_context_scheduler_probe(
        geometry,
        causal_single_kv_tile=False,
        page_size=geometry.page_size,
        max_kv_len=geometry.max_kv_len,
    )
    is_persistent = paged_context_uses_persistent_scheduler(
        mask_type=geometry.mask_type,
        head_paired=geometry.head_paired,
        logical_work_tiles=probe.logical_work_tiles,
        max_active_clusters=probe.max_active_clusters,
        batch_size=geometry.batch_size,
        num_qo_heads=geometry.num_qo_heads,
    )
    if not is_persistent:
        return "nonpersistent"
    if paged_context_uses_clc_scheduler(
        is_persistent=True,
        single_qkv_instance=probe.single_qkv_instance,
        is_causal=geometry.mask_type == "causal",
        uniform_packed_lengths=geometry.uniform_packed_lengths,
    ):
        return "clc_dynamic_persistent"
    return "static_persistent"


def resolve_paged_geometry(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    qo_indptr: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    page_size: int,
    mask_type: str,
    window_left: int,
    out_dtype: torch.dtype,
) -> PagedContextPlanGeometry:
    """Validate one fixed-table paged launch and derive its plan bounds."""

    validate_context_mask(mask_type)
    if mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' is not supported for paged context"
        )
    validate_base_tensors(q, k_cache, v_cache)
    window_left = validate_context_window_left(window_left, mask_type)
    page_size = validate_context_page_size(page_size)
    device_index = validate_device(q.device)
    device = torch.device("cuda", device_index)

    if q.ndim != 3:
        raise ValueError(
            "paged context Q must use packed [total_q, Hq, D] storage; "
            f"got rank {q.ndim}"
        )
    if k_cache.ndim != 4:
        raise ValueError(
            "paged K/V caches must use [num_pages, Hkv, page_size, D] "
            f"storage; got rank {k_cache.ndim}"
        )
    total_q, num_qo_heads, q_head_dim = map(int, q.shape)
    num_physical_pages, num_kv_heads, cache_page_size, kv_head_dim = map(
        int, k_cache.shape
    )
    validate_padded_data_extent(total_q, "total_q")
    validate_extent(num_physical_pages, "num_physical_pages")
    if cache_page_size != page_size:
        raise ValueError(
            f"K/V cache page extent must equal page_size={page_size}; "
            f"got {cache_page_size}"
        )
    validate_context_head_dim(q_head_dim, kv_head_dim)
    validate_compact(q, "q", "[total_q, Hq, D]")
    validate_native_hnd_tma_strides(k_cache, "k_cache")
    validate_native_hnd_tma_strides(v_cache, "v_cache")

    validate_indptr_tensor(qo_indptr, "qo_indptr", device=device)
    batch_size = int(qo_indptr.numel()) - 1
    validate_extent(batch_size, "batch_size")
    _, q_lengths = read_indptr(qo_indptr, "qo_indptr", expected_total=total_q)
    validate_paged_metadata_tensor(seq_lens_kv, "seq_lens_kv", device=device)
    k_lengths = read_int32_values(
        seq_lens_kv,
        "seq_lens_kv",
        expected_count=batch_size,
    )
    for batch_idx, k_length in enumerate(k_lengths):
        if k_length < 0:
            raise ValueError(
                "seq_lens_kv entries must be nonnegative; "
                f"batch {batch_idx} has {k_length}"
            )
    max_q_len = max(q_lengths)
    max_kv_len = max(k_lengths)
    uniform_packed_lengths = all(length == max_q_len for length in q_lengths) and all(
        length == max_kv_len for length in k_lengths
    )
    has_q_offset = derive_has_q_offset(q_lengths, k_lengths, mask_type)

    geometry = resolve_paged_plan_geometry(
        device=device,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q_head_dim,
        qk_dtype=q.dtype,
        pv_dtype=v_cache.dtype,
        page_size=page_size,
        mask_type=mask_type,
        window_left=window_left,
        out_dtype=out_dtype,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
    )
    validate_paged_runtime_inputs(q, k_cache, v_cache, geometry)
    validate_paged_runtime_metadata(
        qo_indptr,
        block_tables,
        seq_lens_kv,
        geometry,
        total_q=total_q,
        num_physical_pages=num_physical_pages,
    )
    return geometry


def resolve_paged_plan_geometry(
    *,
    device: int | str | torch.device,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    qk_dtype: torch.dtype,
    pv_dtype: torch.dtype,
    page_size: int,
    mask_type: str,
    window_left: int,
    out_dtype: torch.dtype,
    uniform_packed_lengths: bool = False,
    has_q_offset: bool = True,
    paged_v_tail_is_zero: bool = False,
) -> PagedContextPlanGeometry:
    """Validate explicit static bounds for a reusable paged specialization."""

    validate_paged_dtype_pair(qk_dtype, qk_dtype, pv_dtype, out_dtype)
    validate_context_mask(mask_type)
    if not isinstance(uniform_packed_lengths, bool):
        raise TypeError("uniform_packed_lengths must be a bool")
    if not isinstance(has_q_offset, bool):
        raise TypeError("has_q_offset must be a bool")
    if not isinstance(paged_v_tail_is_zero, bool):
        raise TypeError("paged_v_tail_is_zero must be a bool")
    if mask_type == "variable_window":
        raise NotImplementedError(
            "mask_type='variable_window' is not supported for paged context"
        )
    window_left = validate_context_window_left(window_left, mask_type)
    page_size = validate_context_page_size(page_size)
    batch_size = validate_static_extent(batch_size, "batch_size")
    max_q_len = validate_static_extent(max_q_len, "max_q_len")
    max_kv_len = validate_static_extent(max_kv_len, "max_kv_len")
    num_qo_heads = validate_static_extent(num_qo_heads, "num_qo_heads")
    num_kv_heads = validate_static_extent(num_kv_heads, "num_kv_heads")
    head_dim = validate_static_extent(head_dim, "head_dim")
    device, device_index = resolve_context_device(device)

    validate_padded_data_extent(batch_size * max_q_len, "batch_size * max_q_len")
    head_ratio = validate_context_head_geometry(num_qo_heads, num_kv_heads)
    validate_context_head_dim(head_dim, head_dim)
    head_paired = window_left > 0
    if head_paired and (head_ratio <= 1 or head_ratio % 2 != 0):
        raise NotImplementedError(
            "a positive left window requires grouped-query attention with an "
            f"even Hq/Hkv ratio greater than one; got {head_ratio}"
        )

    # A causal Q offset is not part of dense attention semantics. Canonicalize
    # it out of the compile identity even when the conservative public default
    # was used, so dense callers keep one cache entry and the leaner kernel.
    has_q_offset = mask_type == "causal" and has_q_offset

    return PagedContextPlanGeometry(
        device=device,
        device_index=device_index,
        batch_size=batch_size,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        qk_dtype=qk_dtype,
        pv_dtype=pv_dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        head_paired=head_paired,
        uniform_packed_lengths=uniform_packed_lengths,
        has_q_offset=has_q_offset,
        paged_v_tail_is_zero=paged_v_tail_is_zero,
        packed_dense_k_mask=(
            mask_type == "dense"
            and (not uniform_packed_lengths or max_kv_len % CONTEXT_KV_TILE_N != 0)
        ),
    )


def two_cta_umma_geometry_eligible(geometry: ContextPlanGeometry) -> bool:
    """Dense contiguous MHA or GQA at D=128 with bf16 QK runs the two-CTA UMMA
    form, which pairs adjacent Q tiles of one head through the grid.
    The two-CTA launch is non-persistent with heads on grid Y and batch on
    grid Z. CUDA limits grid Y and Z to 65,535. Oversized geometries keep the
    persistent flattened grid so the valid int32 plans stay launchable."""
    # VC-Attention-QK16 always runs two-CTA. With P in SMEM, the bf16 K ring,
    # tile-mean ring and row-sum operands leave room for one K/V stage per CTA
    # at one CTA and three in the halved form. The VC tail needs two.
    # VC-Attention-QK8 keeps the E4M3 K ring and follows the device default.
    vc_qk16 = geometry.vc is not None and geometry.qk_dtype not in (
        torch.float8_e4m3fn,
        torch.int8,
    )
    two_cta = default_two_cta_umma(
        geometry.device_index, geometry.qk_dtype.itemsize * 8
    )
    if two_cta and torch.cuda.get_device_capability(geometry.device_index) == (10, 7):
        # Pairs are coarser work units. Keep them only where they fill their
        # waves at least as well as single CTAs fill theirs.
        sms = torch.cuda.get_device_properties(
            geometry.device_index
        ).multi_processor_count
        tiles = math.ceil(geometry.max_q_len / (2 * CONTEXT_TILE_SIZE_Q))
        heads = geometry.batch_size * geometry.num_qo_heads
        single_waves = heads * tiles / sms
        pair_waves = heads * math.ceil(tiles / 2) / (sms // 2)
        two_cta = pair_waves / math.ceil(pair_waves) >= single_waves / math.ceil(
            single_waves
        )
    return (
        (two_cta or vc_qk16)
        and geometry.head_dim == 128
        and geometry.head_dim_vo in (None, 128)
        and geometry.mask_type == "dense"
        and not geometry.packed_query
        and not geometry.head_paired
        and geometry.qk_dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        and not (
            geometry.batch_size > CUDA_GRID_YZ_MAX
            or geometry.num_qo_heads > CUDA_GRID_YZ_MAX
        )
    )


def uses_heavy_first_static_causal_raster(
    *,
    mask_type: str,
    window_left: int,
    has_q_offset: bool,
) -> bool:
    """Return whether a static paged plan has triangular causal tile costs.

    Zero-offset, unwindowed causal attention grows from short to long K-tile
    domains across Q tiles, so reversing the sequence raster retires the
    heaviest work first. A bottom-right offset makes the short-Q context rows
    similarly heavy, while a finite left window bounds their work; both keep
    the ordinary sequence-local raster.
    """

    return mask_type == "causal" and window_left < 0 and not has_q_offset
