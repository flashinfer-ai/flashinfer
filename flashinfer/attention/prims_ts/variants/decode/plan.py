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

"""Plan preparation behind ``PrimsTSBatchDecodePlan`` and the wrapper plan state."""

import torch
from collections.abc import Callable
from dataclasses import dataclass, replace
from ..block_sparse.common import num_sparse_pattern_heads
from ...common.attention_args import (
    validate_layout,
    validate_lse_mode,
    validate_mask,
    validate_page_size,
)
from ...common.paged_kv import (
    PagedKVCache,
    PagedKVScaleFactors,
    normalize_native_paged_kv_cache,
    normalize_paged_kv_scale_factors,
    validate_block_table_metadata,
)
from ...common.scales import GlobalScale, normalize_scale, scale_types
from ...common.tensors import (
    attention_dtype_key,
    resolve_cuda_device,
    validate_runtime_device,
)
from ...common.workspace import validate_workspace_buffer
from ..q_token_kv_block_sparse_metadata import selection as sparse_selection
from flashinfer.utils import get_device_sm_count
from .launch import (
    DecodeWorkspaceLayout,
    DecodeWorkspaceViews,
    DirectQ1CompileSpec,
    bind_decode_workspace,
    get_compiled_decode,
    make_decode_compile_spec,
    make_decode_workspace_layout,
    resolve_decode_launch_spec,
)
from .validation import (
    decode_output_shape,
    resolve_q_mode,
    validate_dtype_pair,
    validate_head_geometry,
    validate_max_kv_len,
    validate_out,
    validate_q,
    validate_qo_indptr,
    validate_storage_page_size,
    validate_window_left,
)
from typing import Literal, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import PrimsTSBatchDecodePlan
    from .kernel.config import FmhaDecodeConfig


@dataclass(frozen=True)
class DecodePlanState:
    """One complete static paged-decode plan published atomically.

    Sequence lengths are owned either by the plan or by every run. The page
    table and (for packed Q) query offsets remain per-run request metadata.
    """

    device: torch.device
    device_index: int
    batch_size: int
    seq_len_q: int
    packed_query: bool
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    max_kv_len: int
    q_dtype: torch.dtype
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: str
    window_left: int
    config: "FmhaDecodeConfig"
    workspace_buffer: torch.Tensor
    workspace_layout: DecodeWorkspaceLayout
    workspace: DecodeWorkspaceViews
    compiled_main: Callable[..., object]
    compiled_reducer: Optional[Callable[..., object]]
    kv_prefix_mode: Literal["dynamic", "planned_full"]
    kv_lengths_mode: Literal["dynamic", "planned_uniform_max"]
    policy: tuple[tuple[str, object], ...]
    softmax_scale: GlobalScale
    output_scale: GlobalScale
    storage_page_size: Optional[int] = None
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    requested_splits: int = 1
    planned_seq_lens_kv_host: Optional[tuple[int, ...]] = None
    planned_seq_lens_kv_device: Optional[torch.Tensor] = None
    fixed_q_offsets: Optional[torch.Tensor] = None


def prepare_batch_decode_plan(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    workspace_buffer: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    max_kv_len: int,
    *,
    seq_len_q: int,
    qo_indptr: Optional[torch.Tensor],
    max_q_len: Optional[int],
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    mask_type: Literal["dense", "causal"],
    window_left: int,
    kv_layout: Literal["HND", "NHD"],
    page_size: Optional[int],
    kv_scale_factors: Optional[PagedKVScaleFactors] = None,
    q_token_kv_block_sparse_page_memberships: Optional[torch.Tensor] = None,
    query_major_memberships: bool = False,
    use_q_token_kv_block_sparse_route: bool = False,
    use_pdl: bool = False,
    split_kv: bool = True,
    direct_q1_inputs: tuple[Optional[torch.Tensor], ...] = (),
    direct_q1_max_model_len: Optional[int] = None,
    direct_q1_sparse_block_size: int = 4,
    direct_q1_fixed_query_len: int = 0,
    count_current_block_in_topk: bool = True,
    share_pattern_across_kv_heads: bool = True,
    allow_head_aware_locators: bool = False,
    use_attention_sinks: bool = False,
    lse_mode: Literal["none", "base2"] = "none",
    softmax_scale: Optional[GlobalScale] = None,
    output_scale: GlobalScale = 1.0,
    balanced_scheduler: Optional[bool] = None,
) -> tuple["PrimsTSBatchDecodePlan", torch.Tensor]:
    """Validate and freeze one dense-block-table PrimTS launch contract."""

    lse_mode = validate_lse_mode(lse_mode)
    validate_layout(kv_layout)
    validate_mask(mask_type)
    window_left = validate_window_left(window_left, mask_type)
    packed_query, resolved_seq_len_q = resolve_q_mode(
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
        require_packed_max=True,
    )
    assert resolved_seq_len_q is not None
    seq_len_q = resolved_seq_len_q
    metadata_device, metadata_rows, max_num_pages = validate_block_table_metadata(
        block_table,
        seq_lens_kv,
    )
    normalized_cache = normalize_native_paged_kv_cache(
        kv_cache, expected_device=query.device, kv_layout=kv_layout
    )
    pattern_heads = num_sparse_pattern_heads(
        normalized_cache.num_kv_heads, share_pattern_across_kv_heads
    )
    if metadata_rows % pattern_heads:
        raise ValueError("metadata rows must be divisible by the pattern head count")
    batch_size = metadata_rows // pattern_heads
    validate_q(
        query,
        seq_len_q=seq_len_q,
        packed_query=packed_query,
        device=metadata_device,
        batch_size=batch_size,
    )
    if qo_indptr is not None:
        validate_qo_indptr(
            qo_indptr,
            expected_device=query.device,
            batch_size=batch_size,
        )
    k_cache = normalized_cache.k_cache
    v_cache = normalized_cache.v_cache
    # Packed NVFP4 without separate scale tensors stores interleaved rows:
    # D/2 payload bytes, then the D/16 E4M3 scale bytes of the same token.
    nvfp4_interleaved_kv = k_cache.dtype == torch.uint8 and kv_scale_factors is None
    if nvfp4_interleaved_kv:
        if not use_q_token_kv_block_sparse_route:
            raise ValueError(
                "packed NVFP4 torch.uint8 K/V requires kv_scale_factors=(K_SF, V_SF); "
                "interleaved NVFP4 rows are supported only by QToken sparse attention"
            )
        storage_head_dim = int(k_cache.shape[-1])
        if storage_head_dim * 16 % 9 != 0:
            raise ValueError(
                "interleaved NVFP4 K/V rows must hold D/2 payload bytes and "
                f"D/16 scale bytes; got {storage_head_dim} bytes per token"
            )
        normalized_cache = replace(
            normalized_cache, head_dim=storage_head_dim * 16 // 9
        )
    num_kv_heads = normalized_cache.num_kv_heads
    storage_page_size = normalized_cache.storage_page_size
    head_dim = normalized_cache.head_dim
    if int(query.shape[-1]) != head_dim:
        raise ValueError(
            "paged_kv_cache logical head dimension must match the query: "
            f"expected {int(query.shape[-1])}, got {head_dim}"
        )
    num_qo_heads = int(query.shape[-2])
    validate_head_geometry(num_qo_heads, num_kv_heads)
    page_size = validate_page_size(
        storage_page_size if page_size is None else page_size
    )
    storage_page_size = validate_storage_page_size(page_size, storage_page_size)
    max_kv_len = validate_max_kv_len(max_kv_len, "max_kv_len")
    required_page_columns = (max_kv_len + page_size - 1) // page_size
    if max_num_pages < required_page_columns:
        raise ValueError(
            "block_table must have at least ceil(max_kv_len / page_size) "
            f"columns ({required_page_columns}), got {max_num_pages}"
        )
    out_dtype = out_dtype
    if out_dtype is None:
        if out is not None and not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        out_dtype = out.dtype if out is not None else query.dtype
    elif not isinstance(out_dtype, torch.dtype):
        raise TypeError("out_dtype must be a torch.dtype")
    validate_dtype_pair(
        query.dtype,
        k_cache.dtype,
        v_cache.dtype,
        out_dtype,
    )
    k_sf_cache, v_sf_cache, k_sf_strides, v_sf_strides = (
        normalize_paged_kv_scale_factors(
            kv_scale_factors,
            k_cache=k_cache,
            logical_head_dim=head_dim,
            interleaved=nvfp4_interleaved_kv,
            strided_rows=use_q_token_kv_block_sparse_route,
        )
    )
    resolved_device, device_index = resolve_cuda_device(query.device)
    validate_runtime_device(resolved_device)
    softmax_scale = normalize_scale(
        head_dim**-0.5 if softmax_scale is None else softmax_scale,
        "softmax_scale",
        resolved_device,
    )
    output_scale = normalize_scale(output_scale, "output_scale", resolved_device)

    policy_args = (
        device_index,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_kv_len,
        seq_len_q,
        attention_dtype_key(query.dtype),
        attention_dtype_key(k_cache.dtype),
        attention_dtype_key(v_cache.dtype),
        attention_dtype_key(out_dtype),
        kv_layout,
        mask_type,
        packed_query,
        window_left,
        storage_page_size,
        use_q_token_kv_block_sparse_route,
        use_pdl,
        split_kv,
        share_pattern_across_kv_heads,
    )
    # Flat compact-HND locators view each cache as one 2-D map of page-fragment
    # rows: they need semantic subpages, contiguous K/V, an Int32 row index and
    # head-aware locators when there are several KV heads.
    flat_native_kv_cache = (
        use_q_token_kv_block_sparse_route
        and storage_page_size != page_size
        and (num_kv_heads == 1 or (allow_head_aware_locators and not direct_q1_inputs))
        and k_cache.is_contiguous()
        and v_cache.is_contiguous()
        and int(k_cache.shape[0]) * storage_page_size * num_kv_heads < (1 << 31)
    )
    spec = resolve_decode_launch_spec(
        *policy_args,
        use_attention_sinks=use_attention_sinks,
    )
    if balanced_scheduler is not False and (
        use_q_token_kv_block_sparse_route
        and packed_query
        and seq_len_q > 1
        and not direct_q1_inputs
        and spec.config.use_persistent_scheduler
        and spec.config.shares_sparse_pattern
    ):
        # Longest-first CLC dispatch pays when work spans more than one wave of
        # SMs (within one wave every route starts at once) and routes can read
        # more than one KV128 tile (one-tile routes leave nothing to balance).
        # The order costs about 1-2 us, which routes of up to four tiles repay
        # only over three or more waves. True forces it.
        tile = sparse_selection.TILE_SIZE_KV
        work_items = batch_size * num_kv_heads
        sm_count = get_device_sm_count(torch.device("cuda", device_index))
        enable = balanced_scheduler or (
            work_items > sm_count
            and max_kv_len > tile
            and (work_items >= 3 * sm_count or max_kv_len > 4 * tile)
        )
        if enable:
            spec = replace(spec, config=replace(spec.config, balanced_scheduler=True))
    nvfp4_keeps = spec.config.uses_keeps_transformed_kv
    policy_config = replace(spec.config, fixed_seqlen=True)
    if query_major_memberships:
        if not policy_config.selects_query_major_memberships:
            raise ValueError(
                "query-major memberships require one G2..G8 Keeps/page-4 CTA"
            )
        spec = replace(spec, config=replace(spec.config, query_major_memberships=True))
    if nvfp4_interleaved_kv and nvfp4_keeps:
        # The Keeps raw ring stages whole interleaved rows; it loads them by
        # gather4 from flat compact-HND caches, otherwise one copy per page
        # fragment (FmhaDecodeConfig.uses_nvfp4_gather4_kv).
        spec = replace(spec, config=replace(spec.config, use_nvfp4_interleaved_kv=True))
    elif nvfp4_interleaved_kv:
        if not spec.config.store_transformed_kv_in_tmem:
            raise NotImplementedError(
                "interleaved NVFP4 K/V requires the TMEM-transformed K/V profile"
            )
        from .kernel.device.kv_transform import has_direct_bf16_fp4_fp8_cvt

        # Its BF16 V transform converts E2M1x2 pairs with the direct BF16 PTX
        # conversion, which has no exact fallback before CUDA 13.2.
        if query.dtype == torch.bfloat16 and not has_direct_bf16_fp4_fp8_cvt():
            raise ValueError(
                "BF16-Q interleaved NVFP4 K/V on the TMEM-transformed recipe "
                "requires a CUDA 13.2+ toolkit"
            )
        from .kernel.config import nvfp4_interleaved_kv_profile

        spec = replace(
            spec, config=replace(spec.config, **nvfp4_interleaved_kv_profile())
        )
    if policy_config.uses_q_token_kv_block_sparse_page_membership:
        if q_token_kv_block_sparse_page_memberships is None:
            raise ValueError(
                "grouped QToken-KvBlock-Sparse-Attention requires a separate q_token_kv_block_sparse_page_memberships tensor"
            )
        # Imported here: the QToken package imports decode at module load.
        from ..q_token_kv_block_sparse_metadata.grouping import (
            validate_page_memberships,
        )

        validate_page_memberships(
            q_token_kv_block_sparse_page_memberships,
            expected_device=query.device,
            batch_size=metadata_rows,
            max_num_pages=max_num_pages,
            planes=policy_config.page_membership_planes,
        )
    else:
        if q_token_kv_block_sparse_page_memberships is not None:
            raise ValueError(
                "q_token_kv_block_sparse_page_memberships is valid only for grouped QToken-KvBlock-Sparse-Attention launches"
            )
        # Keep one compiled tensor ABI without requiring Q1 or dense callers to
        # allocate an unused table. The kernel constexpr-elides all reads.
        q_token_kv_block_sparse_page_memberships = block_table
    layout = make_decode_workspace_layout(
        spec.scratch_shapes,
        out_dtype,
        use_separate_reduction_kernel=spec.config.use_separate_reduction_kernel,
        use_split_kv=spec.config.use_split_kv,
        scratch_reserve=spec.scratch_reserve,
    )
    validate_workspace_buffer(
        workspace_buffer,
        device=query.device,
        required_bytes=layout.total_bytes,
    )
    output_shape = decode_output_shape(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        seq_len_q=seq_len_q,
        head_dim=head_dim,
        total_q_tokens=int(query.shape[0]) if packed_query else None,
    )
    if out is None:
        out = torch.empty(output_shape, device=query.device, dtype=out_dtype)
    else:
        validate_out(
            out,
            q=query,
            expected_shape=output_shape,
            seq_len_q=seq_len_q,
            packed_query=packed_query,
            out_dtype=out_dtype,
        )
    direct_q1_spec = None
    if direct_q1_inputs:
        assert use_q_token_kv_block_sparse_route and seq_len_q == 1
        assert direct_q1_max_model_len is not None
        assert direct_q1_inputs[0] is not None and direct_q1_inputs[3] is not None
        direct_q1_spec = DirectQ1CompileSpec(
            position_dtype_key=str(direct_q1_inputs[3].dtype).removeprefix("torch."),
            max_seq_len_kv=direct_q1_max_model_len,
            block_topk=int(direct_q1_inputs[0].shape[-1]),
            kv_block_size=direct_q1_sparse_block_size,
            count_current_block_in_topk=count_current_block_in_topk,
            page_capacity=max_num_pages,
            fixed_query_len=direct_q1_fixed_query_len,
        )
    from .kernel.config import NVFP4_GATHER4_MAX_FRAGMENT_BYTES

    flat_native_kv_tma = flat_native_kv_cache and (
        (
            spec.config.uses_staged_one_inst_tmem_p
            and (spec.config.use_bf16_qkv or spec.config.use_fp8_qkv)
        )
        # Interleaved NVFP4 gathers four flat fragment rows per copy. Tiles
        # of fewer fragments, or fragments wider than one gather4 box, keep
        # the head-aware 4-D paged copies: only the gather4 loader reads the
        # flat map's prestaged, head-offset rows.
        or (
            spec.config.use_nvfp4_interleaved_kv
            and (spec.config.tile_size_kv // page_size) % 4 == 0
            and page_size * spec.config.nvfp4_interleaved_row_bytes
            <= NVFP4_GATHER4_MAX_FRAGMENT_BYTES
        )
    )
    # Compact separate scale tensors view every page fragment's scale rows
    # as one row of a flat 2-D map, which the Keeps raw ring gathers four
    # fragments at a time (FmhaDecodeConfig.uses_nvfp4_scale_gather4).
    flat_nvfp4_scale_num_heads = 0
    if (
        use_q_token_kv_block_sparse_route
        and spec.config.uses_nvfp4_scale_tensors
        and k_sf_cache.is_contiguous()
        and v_sf_cache.is_contiguous()
        and k_sf_cache.numel() // max(1, page_size * k_sf_cache.shape[-1]) < (1 << 31)
        and replace(
            spec.config, flat_nvfp4_scale_num_heads=num_kv_heads
        ).uses_nvfp4_scale_gather4
    ):
        flat_nvfp4_scale_num_heads = num_kv_heads
    compile_spec = make_decode_compile_spec(
        spec,
        device_index=device_index,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_dtype_key=attention_dtype_key(query.dtype),
        k_dtype_key=attention_dtype_key(k_cache.dtype),
        v_dtype_key=attention_dtype_key(v_cache.dtype),
        out_dtype_key=attention_dtype_key(out_dtype),
        direct_q1_spec=direct_q1_spec,
        flat_native_kv_tma=flat_native_kv_tma,
        flat_nvfp4_scale_num_heads=flat_nvfp4_scale_num_heads,
        lse_mode=lse_mode,
        scale_types=scale_types(softmax_scale, output_scale),
    )
    compiled_main, compiled_reducer = get_compiled_decode(compile_spec)
    workspace = bind_decode_workspace(workspace_buffer, layout)
    # A fused split-KV launch uses a wrapping global completion counter.  The
    # final arriving CTA restores it to zero, so initialization belongs here,
    # once, rather than on every prepared-plan run.  The standalone reducer
    # does not read this counter, but the same initialization is harmless and
    # keeps one replay contract for every split policy.
    if layout.uses_split_kv:
        workspace.split_kv_counter.zero_()
    from . import PrimsTSBatchDecodePlan

    plan = PrimsTSBatchDecodePlan(
        query_shape=tuple(query.shape),
        query_stride=tuple(query.stride()),
        output_shape=tuple(out.shape),
        output_stride=tuple(out.stride()),
        device=query.device,
        q_dtype=query.dtype,
        out_dtype=out_dtype,
        head_dim=head_dim,
        cache=normalized_cache,
        k_sf_cache=k_sf_cache,
        v_sf_cache=v_sf_cache,
        k_sf_strides=k_sf_strides,
        v_sf_strides=v_sf_strides,
        seq_lens_kv=seq_lens_kv,
        qo_indptr=(
            qo_indptr
            if qo_indptr is not None
            else torch.arange(batch_size + 1, dtype=torch.int32, device=query.device)
            * seq_len_q
        ),
        block_table=block_table,
        q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
        workspace=workspace,
        compiled_main=compiled_main,
        compiled_reducer=compiled_reducer,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        direct_q1_inputs=direct_q1_inputs,
        flat_locator_heads=num_kv_heads if flat_native_kv_tma else 1,
        uses_route_order=bool(spec.config.uses_longest_first_route_order),
        lse_mode=lse_mode,
        use_attention_sinks=use_attention_sinks,
        requested_splits=spec.requested_splits or int(spec.config.splits_kv),
        q_capacity=seq_len_q,
        kv_capacity=max_kv_len,
        direct_model_capacity=direct_q1_max_model_len or 0,
        direct_page_capacity=max_num_pages,
        direct_fixed_query_len=direct_q1_fixed_query_len,
    )
    return plan, out
