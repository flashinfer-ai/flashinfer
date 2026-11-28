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

"""Task-scheduled paged decode with a FlashInfer-style plan/run lifecycle."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, Optional, Union

import torch

from ...common.scales import (
    GlobalScale,
    normalize_scale,
    resolve_scale,
    scale_types,
)

from flashinfer.api_logging import flashinfer_experimental_api

from flashinfer.trace.templates.attention import (
    attention_ts_decode_trace_dispatch,
    prims_ts_decode_wrapper_trace_dispatch,
)

from ..q_token_kv_block_sparse_metadata import selection as sparse_selection
from ...common.tensors import (
    attention_dtype_key,
    resolve_cuda_device,
    validate_runtime_device,
)
from ...common.workspace import (
    validate_workspace_buffer,
)
from ...common.attention_args import (
    resolve_kv_dtypes,
    validate_layout,
    validate_mask,
    validate_page_size,
    validate_positive_int,
    prepare_lse,
    resolve_max_sequence_length,
    validate_attention_sinks,
    validate_lse_mode,
)
from ...common.paged_kv import (
    NativePagedKVCache,
    PagedKVCache,
    PagedKVScaleFactors,
    normalize_paged_kv_cache_views,
    validate_block_table_metadata,
)
from .validation import (
    decode_output_shape,
    resolve_q_mode,
    validate_decode_query_head_extent,
    validate_dtype_pair,
    validate_head_dim,
    validate_head_geometry,
    validate_max_kv_len,
    validate_out,
    validate_packed_q_plan_values,
    validate_q,
    validate_qo_indptr,
    validate_seq_len_q,
    validate_storage_page_size,
    validate_window_left,
)
from .launch import (
    DecodeWorkspaceViews,
    bind_decode_workspace,
    decode_scratch_shapes,
    get_compiled_decode,
    launch_decode,
    make_decode_compile_spec,
    make_decode_workspace_layout,
    normalize_plan_seq_lens_kv,
    prepare_decode_runtime,
    prepare_decode_runtime_unchecked,
    resolve_decode_launch_spec,
    resolve_decode_workspace_layout,
    validate_decode_run_metadata_values,
)
from .plan import DecodePlanState, prepare_batch_decode_plan

if TYPE_CHECKING:
    pass


def get_prims_ts_batch_decode_workspace_size(
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_kv_len: int,
    *,
    seq_len_q: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    q_dtype: torch.dtype = torch.float16,
    k_dtype: Optional[torch.dtype] = None,
    v_dtype: Optional[torch.dtype] = None,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    kv_layout: Literal["HND", "NHD"] = "HND",
    storage_page_size: Optional[int] = None,
    device: Optional[Union[int, str, torch.device]] = None,
    split_kv: bool = True,
    use_attention_sinks: bool = False,
    validate: bool = True,
) -> int:
    """Return caller-workspace bytes for one automatic FMHA policy.

    The arguments resolve the same policy and scratch layout as
    :func:`batch_decode_with_paged_kv_cache`, without compiling attention kernels.
    Allocate at least the returned number of bytes as a contiguous
    ``torch.int8`` or ``torch.uint8`` CUDA tensor and zero it before its first
    FMHA launch. Re-zero a reused buffer whenever any workspace-layout input,
    including ``batch_size``, changes because the internal section offsets can
    move even when the compiled callable is reused. Alternatively, plans for one
    device may share one buffer of at least the largest size returned for them:
    zero the whole buffer once, bind the same full tensor to every plan (not
    slices), and plan and launch them on one stream. PrimTS may use any byte
    of the passed buffer: its zero-at-rest control state sits at the buffer
    end, and launches leave it zero. Fixed-Q launches use ``seq_len_q``.
    Packed-Q launches provide ``qo_indptr`` and the explicit static
    ``max_q_len`` bound used for workspace geometry and JIT policy.
    ``max_kv_len`` must be no larger than ``2,147,483,392`` so the padded
    256-token K/V tile endpoint remains representable as signed Int32.
    ``storage_page_size`` defaults to the semantic ``page_size``. A larger
    physical page is supported only for semantic page size four and activates
    encoded ``(physical page, subpage)`` locators.
    By default, this sizing helper validates that every cumulative-offset delta
    is positive and no larger than the bound. If ``device`` is omitted, it is
    inferred from ``qo_indptr`` for a packed launch. ``k_dtype`` defaults to ``q_dtype`` and
    ``v_dtype`` to ``k_dtype``; pass ``v_dtype=torch.float8_e4m3fn`` with
    BF16 Q/K to size the QK-BF16/PV-FP8 launch. ``kv_dtype`` is a
    compatibility alias for both K and V and must agree with explicit dtypes.

    ``split_kv`` permits automatic split fanout when True (default), or forces
    nonsplit execution when False. It is independent of Q layout and must
    match the value supplied to attention planning/launch.
    ``use_attention_sinks`` must also match planning because it can affect
    the selected policy and scratch layout.

    ``validate=False`` skips repeated input/device validation and packed-Q
    offset readbacks. The caller must ensure supported hardware, valid geometry,
    dtypes and metadata, and a static Q bound covering every request. Alias and
    device resolution are retained, as is the shared automatic-policy cache;
    a cold policy lookup may still query device properties or compile a hardware
    occupancy probe. Warm up outside CUDA graph capture.
    """

    if not isinstance(validate, bool):
        raise TypeError("validate must be a bool")
    if validate:
        batch_size = validate_positive_int(batch_size, "batch_size")
    packed_query, resolved_seq_len_q = resolve_q_mode(
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
        require_packed_max=True,
        validate=validate,
    )
    assert resolved_seq_len_q is not None
    seq_len_q = resolved_seq_len_q
    if validate:
        validate_head_geometry(num_qo_heads, num_kv_heads)
        validate_decode_query_head_extent(
            batch_size=batch_size,
            num_qo_heads=num_qo_heads,
            max_q_len=seq_len_q,
        )
        head_dim = validate_head_dim(head_dim)
        page_size = validate_page_size(page_size)
        storage_page_size = validate_storage_page_size(
            page_size,
            page_size if storage_page_size is None else storage_page_size,
        )
        max_kv_len = validate_max_kv_len(max_kv_len, "max_kv_len")
        validate_layout(kv_layout)
        validate_mask(mask_type)
        window_left = validate_window_left(window_left, mask_type)
    elif storage_page_size is None:
        storage_page_size = page_size
    k_dtype, v_dtype = resolve_kv_dtypes(q_dtype, k_dtype, v_dtype, kv_dtype)
    if out_dtype is None:
        out_dtype = q_dtype
    if validate:
        validate_dtype_pair(q_dtype, k_dtype, v_dtype, out_dtype)
    inferred_device = (
        qo_indptr.device
        if device is None and isinstance(qo_indptr, torch.Tensor)
        else device
    )
    resolved_device, _ = resolve_cuda_device(inferred_device)
    if validate and qo_indptr is not None:
        validate_qo_indptr(
            qo_indptr,
            expected_device=resolved_device,
            batch_size=batch_size,
        )
        validate_packed_q_plan_values(
            qo_indptr,
            max_q_len=seq_len_q,
        )

    return resolve_decode_workspace_layout(
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_kv_len,
        seq_len_q,
        q_dtype,
        k_dtype,
        v_dtype,
        out_dtype,
        kv_layout,
        mask_type,
        packed_query,
        window_left,
        storage_page_size,
        resolved_device,
        split_kv=split_kv,
        use_attention_sinks=use_attention_sinks,
        validate=validate,
    ).total_bytes


@dataclass(frozen=True)
class PrimsTSBatchDecodePlan:
    """Validated dense-page-table PrimTS state for framework hot paths.

    The plan retains the K/V cache and scale tensors, dense block-table storage, compiled
    callables, and typed workspace views validated at construction. Block-table
    and sequence-length values may change between completed launches, which is
    the contract needed by QToken-KvBlock-Sparse-Attention metadata builders. Query and output storage may
    also change, but must preserve the exact shape, dtype, device, and strides
    proven by the representative tensors passed to
    :func:`prepare_prims_ts_batch_decode_with_kv_cache`.

    ``run`` deliberately omits repeated cache-stride validation, workspace
    rebinding, and semantic-policy resolution. Storage overlap is not checked
    during preparation or execution. The framework must keep retained tensors
    alive, keep output/workspace disjoint from the inputs and each other, and
    not mutate metadata concurrently with a launch or replay that reads it.
    """

    query_shape: tuple[int, ...]
    query_stride: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_stride: tuple[int, ...]
    device: torch.device
    q_dtype: torch.dtype
    out_dtype: torch.dtype
    head_dim: int
    cache: NativePagedKVCache
    k_sf_cache: torch.Tensor
    v_sf_cache: torch.Tensor
    k_sf_strides: tuple[int, int, int]
    v_sf_strides: tuple[int, int, int]
    seq_lens_kv: torch.Tensor
    qo_indptr: Optional[torch.Tensor]
    block_table: torch.Tensor
    q_token_kv_block_sparse_page_memberships: torch.Tensor
    workspace: DecodeWorkspaceViews
    compiled_main: Callable[..., object]
    compiled_reducer: Optional[Callable[..., object]]
    softmax_scale: GlobalScale
    output_scale: GlobalScale
    direct_q1_inputs: tuple[Optional[torch.Tensor], ...] = ()
    flat_locator_heads: int = 1
    # Persistent grouped QToken launches read the metadata's route order.
    uses_route_order: bool = False
    lse_mode: str = "none"
    use_attention_sinks: bool = False
    requested_splits: int = 1
    q_capacity: int = 1
    kv_capacity: int = 1
    direct_model_capacity: int = 0
    direct_page_capacity: int = 0
    direct_fixed_query_len: int = 0

    def run(
        self,
        query: torch.Tensor,
        *,
        out: torch.Tensor,
        softmax_scale: Optional[GlobalScale] = None,
        output_scale: Optional[GlobalScale] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        attention_sinks: Optional[torch.Tensor] = None,
        validate: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Launch with lightweight checks against the validated plan.
        ``lse`` is a compact float32 output of shape ``query.shape[:-1]``
        (``q.shape[:-1]`` for FMHA). Enable its base with ``plan(lse_mode=...)``.
        It is allocated if omitted; ``return_lse`` controls only tuple return.
        Supply output/LSE buffers and ``validate=False`` for allocation-free
        capture. See the module guide for scale and buffer ownership contracts.

        Omitted scales inherit the preparation defaults. Each override must
        retain its planned host-float or CUDA FP32 tensor type. Tensor scales
        have shape () or (1,), reside on the query device, and must remain finite and
        positive; validation checks metadata without reading device values.

        ``attention_sinks`` is a compact CUDA float32 ``[Hq]`` vector of
        natural-log sink logits after QK scaling, with zero V contribution.
        Values must be finite or ``-inf``; this is a caller precondition.
        Validation checks tensor metadata without reading the sink values.
        """

        if (
            query.shape != self.query_shape
            or query.stride() != self.query_stride
            or query.device != self.device
            or query.dtype != self.q_dtype
        ):
            raise ValueError(
                "query must preserve the shape, strides, device, and dtype "
                "validated by the PrimTS plan"
            )
        if (
            out.shape != self.output_shape
            or out.stride() != self.output_stride
            or out.device != self.device
            or out.dtype != self.out_dtype
        ):
            raise ValueError(
                "out must preserve the shape, strides, device, and dtype "
                "validated by the PrimTS plan"
            )

        softmax_scale = resolve_scale(
            softmax_scale, self.softmax_scale, "softmax_scale", self.device
        )
        output_scale = resolve_scale(
            output_scale, self.output_scale, "output_scale", self.device
        )
        lse = prepare_lse(
            lse, q=query, mode=self.lse_mode, return_lse=return_lse, validate=validate
        )
        attention_sinks = validate_attention_sinks(
            attention_sinks,
            q=query,
            enabled=self.use_attention_sinks,
            validate=validate,
        )
        result = self.run_unchecked(
            query,
            out,
            softmax_scale,
            output_scale,
            lse=lse,
            attention_sinks=attention_sinks,
        )
        return (result, lse) if return_lse else result

    def run_unchecked(
        self,
        query: torch.Tensor,
        out: torch.Tensor,
        softmax_scale: GlobalScale,
        output_scale: GlobalScale,
        direct_q1_inputs: Optional[tuple[Optional[torch.Tensor], ...]] = None,
        lse: Optional[torch.Tensor] = None,
        attention_sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Launch state already proven by a framework-owned outer plan."""

        q_offsets = (
            self.workspace.cu_seqlens_q if self.qo_indptr is None else self.qo_indptr
        )
        direct_inputs = (
            self.direct_q1_inputs if direct_q1_inputs is None else direct_q1_inputs
        )
        if direct_inputs:
            direct_inputs = (
                *direct_inputs,
                self.direct_model_capacity,
                self.direct_page_capacity,
                self.direct_fixed_query_len,
            )
        query_view = query.view(-1, query.shape[-2], query.shape[-1])
        out_view = out.view(-1, out.shape[-2], out.shape[-1])
        self.compiled_main(
            query_view,
            self.cache.k_cache,
            self.cache.v_cache,
            self.k_sf_cache,
            self.v_sf_cache,
            out_view,
            None if lse is None else lse.view(-1, lse.shape[-1]),
            self.seq_lens_kv,
            q_offsets,
            self.block_table,
            self.q_token_kv_block_sparse_page_memberships,
            self.workspace.partial_o,
            self.workspace.partial_stats,
            self.workspace.split_kv_counter,
            self.workspace.attention_sinks
            if attention_sinks is None
            else attention_sinks,
            self.cache.num_physical_pages,
            self.cache.k_page_stride,
            self.cache.k_head_stride,
            self.cache.k_token_stride,
            self.cache.v_page_stride,
            self.cache.v_head_stride,
            self.cache.v_token_stride,
            self.k_sf_strides[0],
            self.v_sf_strides[0],
            *self.k_sf_strides[1:],
            *self.v_sf_strides[1:],
            softmax_scale,
            output_scale,
            self.requested_splits,
            self.q_capacity,
            direct_inputs,
            self.kv_capacity,
        )
        if self.compiled_reducer is not None:
            self.compiled_reducer(
                out_view,
                None if lse is None else lse.view(-1, lse.shape[-1]),
                self.seq_lens_kv,
                q_offsets,
                self.workspace.partial_o,
                self.workspace.partial_stats,
                self.workspace.attention_sinks
                if attention_sinks is None
                else attention_sinks,
                self.requested_splits,
                self.q_capacity,
                direct_inputs,
                self.kv_capacity,
            )
        return out


@flashinfer_experimental_api
def suggest_q_token_kv_block_sparse_group_size(
    batch_size: int,
    seq_len_q: int,
    selected_seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    multi_processor_count: int,
    *,
    head_dim: int = 256,
    q_dtype: torch.dtype = torch.bfloat16,
    split_kv: bool = True,
    kv_block_size: int = 4,
    union_exponent: float = sparse_selection.DEFAULT_UNION_EXPONENT,
    nvfp4_kv: bool = False,
    seq_len_kv: Optional[int] = None,
    packed_query: bool = False,
    nvfp4_interleaved_kv: bool = False,
) -> int:
    """Suggest a QToken-KvBlock-Sparse-Attention group (G1-G8, G16 or G32).

    This pure host-side policy uses the caller-provided SM count; it never
    queries device properties or reads tensors. For NVFP4 K/V
    (``nvfp4_kv=True``) it estimates every legal group's call latency: the
    group's expected K/V union (from ``union_exponent``) under the selected
    recipe's split fanout gives the KV tiles per work item and the number of
    SM waves; grouped metadata (G >= 2) and the split reduction add fixed
    costs. The cheapest group wins; ties prefer the smaller group. BF16/FP8
    caches keep the G1-G8 coverage rule (the largest group that fills the
    SMs, shrunk only for better SM coverage); the latency constants were
    fitted on NVFP4 only. Explicit G passed to plan/run is never overridden.

    Query lengths need not be divisible by the result. Packed prefill may end
    each request with a shorter route. Fixed-shape decode may pad its final
    ``[B, num_query_groups, G, Hq, D]`` route with consecutive semantic dummy
    rows and discard their outputs.

    ``selected_seq_len_kv`` is the maximum candidate K/V-token count for one
    query, including its causal tail; pass ``min(budget + tail, kv_length)``
    for short contexts. Configured split fanout uses union capacity to
    accommodate non-shared KV; the estimate counts the splits active on the
    expected union.

    Parameters
    ----------
    batch_size : int
        Number of requests in the launch.
    seq_len_q : int
        Maximum number of live query tokens per request.
    selected_seq_len_kv : int
        Maximum selected candidate K/V tokens per query, including the causal
        tail. This is not the original context length or global cache capacity.
    num_qo_heads : int
        Number of query/output heads.
    num_kv_heads : int
        Number of key/value heads.
    multi_processor_count : int
        Number of SMs available to the launch. Frameworks should query it once
        and pass the cached value; this function performs no device query.
    head_dim : int
        Common Q/K/V head dimension: 64, 128, or 256.
    q_dtype : torch.dtype
        Q/K/V dtype used to select the matching dense MMA recipe.
    split_kv : bool
        Allow automatic split-KV fanout when True (default); False forces
        one KV partition. Pass False for prefill and True for decode,
        independently of packed or fixed Q layout. Match this value in
        workspace sizing and attention planning. Ragged workloads use maximum
        Q length as a capacity estimate; no GPU sequence lengths are inspected.

    kv_block_size : int
        Semantic sparse block size: 4/8/16/32/64/128, default four. Used to
        bound grouped union work; storage-page size does not affect this bound.
    union_exponent : float
        Growth of a G-query group's K/V union, estimated as
        ``selected * G ** union_exponent``; in [0, 1]. The default is fitted
        to latency on real sparse-indexer outputs, whose unions grow about
        1.4x per doubling of G. Used with ``nvfp4_kv=True`` only.
    nvfp4_kv : bool
        Plan for NVFP4 K/V with the recipes ``plan`` takes for it (BF16 Q
        always takes KeepsMmaAb). For BF16 K/V that the caller converted from NVFP4 (see
        ``QTokenKvBlockSparsePagedTSWrapper.run``), pass ``False``; that
        suggestion stays within G1-G8, while the conversion only pays with
        TileQ128 groups, which the caller then selects explicitly.
    seq_len_kv : int, optional
        KV length of the longest request (its context including these
        queries). A group's union cannot exceed it, so short contexts (e.g. a
        prefill from an empty prefix) favor larger groups; pass it whenever it
        is known. ``None`` keeps the uncapped union estimate. Used with
        ``nvfp4_kv=True`` only.
    packed_query : bool
        Whether the launch passes packed queries (``qo_indptr``). G16 and G32
        are suggested only for NVFP4 K/V when True and ``kv_block_size`` is at
        most 16; ``plan`` rejects them for fixed-shape queries. Default False
        (G1-G8).
    nvfp4_interleaved_kv : bool
        Whether the NVFP4 K/V cache stores interleaved rows (payload and
        scales in one row, ``kv_scale_factors=None`` at run) rather than
        separate scale tensors; the two layouts have separate latency
        constants. Default False. Used with ``nvfp4_kv=True`` only.

    Returns
    -------
    int
        One of G1--G8, or G16/G32 for packed NVFP4 queries, subject to the
        TileQ128 head-capacity bound.
    """
    from ..q_token_kv_block_sparse_metadata.grouping import suggest_group_size

    return suggest_group_size(
        batch_size,
        seq_len_q,
        selected_seq_len_kv,
        num_qo_heads,
        num_kv_heads,
        multi_processor_count,
        head_dim=head_dim,
        q_dtype=q_dtype,
        split_kv=split_kv,
        kv_block_size=kv_block_size,
        union_exponent=union_exponent,
        nvfp4_kv=nvfp4_kv,
        seq_len_kv=seq_len_kv,
        packed_query=packed_query,
        nvfp4_interleaved_kv=nvfp4_interleaved_kv,
    )


@flashinfer_experimental_api
def validate_q_token_kv_block_sparse_group_size(
    query_start_loc_cpu: Optional[torch.Tensor],
    num_query_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    *,
    group_size: int = 1,
) -> int:
    """Validate a QToken-KvBlock-Sparse-Attention Q1--Q8 group.

    ``query_start_loc_cpu`` contains cumulative flattened-query offsets for
    real requests. ``num_query_tokens`` includes any inert CUDA-graph padding
    rows. Validation caps the maximum grouped query by both the kernel
    membership representation and TileQ128 capacity. Request lengths do not
    have to be divisible by ``group_size``: frameworks can partition each
    request into packed routes of at most that size without crossing a request
    or real/padding boundary.

    This function never chooses a group from workload size or occupancy. The
    framework owns ``group_size``; attention split-KV fanout is selected
    separately after the grouped launch grid is known.

    Parameters
    ----------
    query_start_loc_cpu : torch.Tensor or None
        CPU Int32 or Int64 cumulative offsets for the real requests. It is
        required when ``group_size`` is greater than one; ``None`` is accepted
        for Q1.
    num_query_tokens : int
        Total flattened query-row count, including any inert CUDA-graph
        padding rows.
    num_qo_heads : int
        Number of query/output heads.
    num_kv_heads : int
        Number of key/value heads.
    group_size : int
        Caller-selected number of query rows per QToken-KvBlock-Sparse-Attention route. Supported values
        are Q1--Q8, subject to the TileQ128 head-capacity bound. Defaults to
        one (no grouping).

    Returns
    -------
    int
        The validated ``group_size``.
    """
    from ..q_token_kv_block_sparse_metadata.grouping import validate_group_size

    return validate_group_size(
        query_start_loc_cpu,
        num_query_tokens,
        num_qo_heads,
        num_kv_heads,
        group_size=group_size,
    )


@flashinfer_experimental_api
def make_q_token_kv_block_sparse_qo_indptr(
    query_start_loc_cpu: torch.Tensor,
    num_query_tokens: int,
    *,
    group_size: int = 1,
    device: Optional[Union[int, str, torch.device]] = None,
) -> torch.Tensor:
    """Build request-safe QToken-KvBlock-Sparse-Attention route offsets.

    ``query_start_loc_cpu`` contains cumulative offsets for real requests.
    Every request is chunked independently, so its final route may contain
    fewer than ``group_size`` rows. The default is one (no query grouping).
    Any inert CUDA-graph padding suffix is partitioned separately. Pass the
    returned Int32 ``qo_indptr`` to a packed-Q wrapper planned with the same
    ``group_size``.

    Parameters
    ----------
    query_start_loc_cpu : torch.Tensor
        One-dimensional CPU Int32 or Int64 cumulative offsets for the real
        requests. The first entry must be zero and the offsets must be
        nondecreasing.
    num_query_tokens : int
        Total flattened query-row count, including any inert CUDA-graph
        padding suffix.
    group_size : int
        Maximum rows per request-safe query group, default one (no grouping).
    device : int, str, torch.device, or None
        Destination device for the returned offsets. By default, PyTorch's
        default tensor device is used.

    Returns
    -------
    torch.Tensor
        Contiguous Int32 cumulative route offsets. Every nonempty route has at
        most ``group_size`` rows and no route crosses a request boundary.
    """
    from ..q_token_kv_block_sparse_metadata.grouping import make_qo_indptr

    return make_qo_indptr(
        query_start_loc_cpu, num_query_tokens, group_size=group_size, device=device
    )


@flashinfer_experimental_api
def prepare_prims_ts_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    workspace_buffer: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    max_kv_len: int,
    *,
    out: torch.Tensor,
    kv_scale_factors: Optional[PagedKVScaleFactors] = None,
    seq_len_q: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    kv_layout: Literal["HND", "NHD"] = "HND",
    page_size: Optional[int] = None,
    split_kv: bool = True,
    use_attention_sinks: bool = False,
    lse_mode: Literal["none", "base2"] = "none",
    softmax_scale: Optional[GlobalScale] = None,
    output_scale: GlobalScale = 1.0,
) -> PrimsTSBatchDecodePlan:
    """Validate and prepare a reusable dense-block-table PrimTS launch.

    This setup API is intended for frameworks that keep K/V, block-table
    metadata, and workspace storage stable while changing their values between launches.
    Call :meth:`PrimsTSBatchDecodePlan.run` on the hot path.  The returned plan
    is CUDA-graph compatible as long as captured tensor storage remains alive.

    Output and workspace must be disjoint from each other and from all inputs.
    Storage overlap is a caller precondition and is not checked.

    Parameters
    ----------
    query : torch.Tensor
        Fixed ``[B, Hq, D]`` or ``[B, SQ, Hq, D]`` query, or packed
        ``[total_q, Hq, D]`` query when ``qo_indptr`` is supplied.
    kv_cache : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Combined or separate HND paged K/V storage.
    workspace_buffer : torch.Tensor
        Caller-owned byte workspace, zero-initialized before the first launch
        for this semantic configuration.
    block_table : torch.Tensor
        Contiguous CUDA Int32 dense page table shaped ``[B, max_pages]``.
    seq_lens_kv : torch.Tensor
        Positive live K/V sequence lengths, one per request.
    max_kv_len : int
        Static maximum K/V length used for policy selection and JIT caching.
    out : torch.Tensor
        Caller-owned output tensor whose storage remains stable across replays.
    kv_scale_factors : tuple[torch.Tensor, torch.Tensor], optional
        Required for packed NVFP4 K/V stored as uint8. K and V scales are
        FP8 tensors with width ``D / 16``; V scales use the TRTLLM-GEN
        4-token interleaved layout. The plan retains these tensors.
    seq_len_q : int
        Fixed query length when ``qo_indptr`` is omitted.
    qo_indptr : torch.Tensor, optional
        Cumulative Int32 query offsets selecting packed-query mode.
    max_q_len : int, optional
        Static maximum packed-query route length.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype``.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    kv_layout : {"HND", "NHD"}
        Layout of the paged K/V cache.
    page_size : int, optional
        Semantic page size; defaults to the physical cache-page extent.
    split_kv : bool
        Permit automatic split fanout (True, default), or force S1 (False).
        Independent of packed/fixed Q. Match workspace sizing's value.
    softmax_scale, output_scale : float or torch.Tensor, optional
        Independent QK and V/output multipliers, defaulting to ``head_dim**-0.5``
        and one. Each operand is a host float or a contiguous CUDA FP32 tensor
        of shape () or (1,) on the query device. Types select the compiled signature;
        values are retained as run defaults. Softmax uses natural units. Tensor
        contents must stay finite and positive and may change during replay.

    Returns
    -------
    PrimsTSBatchDecodePlan
        Reusable prepared launch plan. Call :meth:`PrimsTSBatchDecodePlan.run`
        after updating values in the prepared tensor storage.

    ``lse_mode="none"`` (default) keeps the output-only contract.
    Enable ``"base2"`` while planning to write final base-2 LSE.
    Writing LSE is a compilation feature; the output pointer stays runtime.

    ``use_attention_sinks=True`` enables the per-run ``attention_sinks``
    binding; supply it on every run. Defaults to ``False``. Sizing caller
    workspace must use the same sink capability as planning.
    """

    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a caller-owned torch.Tensor")
    plan, _ = prepare_batch_decode_plan(
        query,
        kv_cache,
        workspace_buffer,
        block_table,
        seq_lens_kv,
        max_kv_len,
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_q_len=max_q_len,
        out=out,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        kv_layout=kv_layout,
        page_size=page_size,
        kv_scale_factors=kv_scale_factors,
        split_kv=split_kv,
        use_q_token_kv_block_sparse_route=False,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        lse_mode=lse_mode,
        use_attention_sinks=use_attention_sinks,
    )
    return plan


class BatchDecodePagedTSWrapper:
    """Reusable paged decode with eager planning and shared scratch.

    Plan selects a cached execution from problem bounds and binds scratch.
    Each run supplies query, lengths, page tables and optional packed-Q offsets.
    Policies/compiled kernels are cached across layers; resources remain local.
    Concurrent streams or graph replays require separate wrappers/workspaces."""

    @flashinfer_experimental_api
    def __init__(self, kv_layout: Literal["HND", "NHD"] = "HND") -> None:
        """Initialize an unplanned wrapper with one static K/V layout.

        Parameters
        ----------
        kv_layout : {"HND", "NHD"}
            Physical K/V layout, ``"HND"`` or ``"NHD"``.
        """

        validate_layout(kv_layout)
        self.kv_layout = kv_layout
        self.plan_state: Optional[DecodePlanState] = None

    @property
    def policy(self) -> tuple[tuple[str, object], ...]:
        """Return the immutable policy record for the published plan."""

        return self.require_plan_state().policy

    @flashinfer_experimental_api
    def plan(
        self,
        device: Union[int, str, torch.device],
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        max_kv_len: int,
        *,
        max_q_len: int | None = None,
        packed_query: bool = False,
        q_dtype: torch.dtype = torch.float16,
        k_dtype: Optional[torch.dtype] = None,
        v_dtype: Optional[torch.dtype] = None,
        kv_dtype: Optional[torch.dtype] = None,
        out_dtype: Optional[torch.dtype] = None,
        mask_type: Literal["dense", "causal"] = "dense",
        lse_mode: Literal["none", "base2"] = "none",
        use_attention_sinks: bool = False,
        window_left: int = -1,
        seq_lens_kv: Optional[Union[Sequence[int], torch.Tensor]] = None,
        workspace_buffer: Optional[torch.Tensor] = None,
        storage_page_size: Optional[int] = None,
        split_kv: bool = True,
        validate: bool = True,
        initialize_workspace: bool = True,
        softmax_scale: Optional[GlobalScale] = None,
        output_scale: GlobalScale = 1.0,
    ) -> None:
        """Select/cache a bounded paged-decode execution and bind reusable scratch.

        batch_size is exact. max_q_len is the fixed query length, or an
        explicit per-request bound for packed queries. Live K/V lengths remain
        within max_kv_len. Exact Q/KV bounds remain runtime operands when topology is unchanged. seq_lens_kv optionally supplies host lengths copied into
        plan-owned CUDA storage; otherwise every run supplies its own lengths.
        Kernel selection and compilation/loading finish here, before capture.
        Replanning replaces state and invalidates its prior captured graphs.
        Scratch is sized for the selected policy and problem geometry. Caller
        scratch may be passed without a size query. Planning initializes control
        storage unless initialize_workspace=False transfers that to the caller.
        Active split counts are automatic runtime operands for main/reduction
        kernels; users never supply the count.

        ``softmax_scale`` and ``output_scale`` independently accept host floats
        or contiguous CUDA FP32 tensors of shape ``()`` or ``(1,)``. Their types select
        the compiled signature; run() inherits these defaults. Device values
        must be finite and positive and may change in place during graph replay.
        Softmax scales use natural, not base-2, units.
        """

        if packed_query and max_q_len is None:
            raise ValueError("packed Q requires max_q_len in plan()")
        if max_q_len is None:
            max_q_len = 1
        k_dtype, v_dtype = resolve_kv_dtypes(q_dtype, k_dtype, v_dtype, kv_dtype)
        if out_dtype is None:
            out_dtype = q_dtype
        seq_len_q = max_q_len
        storage_page_size = (
            page_size if storage_page_size is None else storage_page_size
        )
        lse_mode = validate_lse_mode(lse_mode)
        if not isinstance(use_attention_sinks, bool):
            raise TypeError("use_attention_sinks must be a bool")
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        if not isinstance(initialize_workspace, bool):
            raise TypeError("initialize_workspace must be a bool")
        if validate:
            if not isinstance(packed_query, bool):
                raise TypeError("packed_query must be a bool")
            batch_size = validate_positive_int(batch_size, "batch_size")
            head_dim = validate_head_dim(head_dim)
            page_size = validate_page_size(page_size)
            storage_page_size = validate_storage_page_size(page_size, storage_page_size)
            max_kv_len = validate_max_kv_len(max_kv_len, "max_kv_len")
            seq_len_q = validate_seq_len_q(max_q_len)
            validate_head_geometry(num_qo_heads, num_kv_heads)
            validate_decode_query_head_extent(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                max_q_len=seq_len_q,
            )
            validate_mask(mask_type)
            window_left = validate_window_left(window_left, mask_type)

            validate_dtype_pair(
                q_dtype,
                k_dtype,
                v_dtype,
                out_dtype,
            )

            planned_seq_lens = normalize_plan_seq_lens_kv(
                seq_lens_kv,
                batch_size=batch_size,
                max_kv_len=max_kv_len,
            )
            if (
                planned_seq_lens is not None
                and mask_type == "causal"
                and not packed_query
            ):
                for request_idx, kv_len in enumerate(planned_seq_lens):
                    if kv_len > 0 and seq_len_q > kv_len:
                        raise ValueError(
                            "causal decode requires every per-request Q length to be "
                            "no greater than its K/V length; request "
                            f"{request_idx} has Q={seq_len_q} and K/V={kv_len}"
                        )

        else:
            planned_seq_lens = normalize_plan_seq_lens_kv(
                seq_lens_kv, batch_size=batch_size, max_kv_len=max_kv_len
            )

        device, device_index = resolve_cuda_device(device)
        if validate:
            validate_runtime_device(device)
        softmax_scale = normalize_scale(
            head_dim**-0.5 if softmax_scale is None else softmax_scale,
            "softmax_scale",
            device,
        )
        output_scale = normalize_scale(output_scale, "output_scale", device)

        policy_args = (
            device_index,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_kv_len,
            seq_len_q,
            attention_dtype_key(q_dtype),
            attention_dtype_key(k_dtype),
            attention_dtype_key(v_dtype),
            attention_dtype_key(out_dtype),
            self.kv_layout,
            mask_type,
            packed_query,
            window_left,
            storage_page_size,
        )
        spec = resolve_decode_launch_spec(
            *policy_args, split_kv=split_kv, use_attention_sinks=use_attention_sinks
        )
        kv_prefix_mode: Literal["dynamic", "planned_full"] = "dynamic"
        kv_lengths_mode: Literal["dynamic", "planned_uniform_max"] = "dynamic"
        cfg = spec.config
        # Reuse declared uniform bounds. Single-Q GMEM keeps its runtime page window.
        uniform_kv = (
            not packed_query
            and not cfg.use_sliding_window_causal
            # Keep FP8 Keeps multi-Q arithmetic identical to standalone calls,
            # whose KV bounds are runtime metadata rather than static lengths.
            and not (
                cfg.use_keeps_mma_ab
                and cfg.use_fp8_qkv
                and head_dim == 128
                and seq_len_q > 1
                and cfg.o_dtype_bytes == 2
            )
            and planned_seq_lens is not None
            and all(length == max_kv_len for length in planned_seq_lens)
            and ((max_kv_len + cfg.tile_size_kv - 1) // cfg.tile_size_kv)
            % cfg.num_insts_kv
            == 0
        )
        fixed_uniform_kv = uniform_kv and (
            cfg.use_cluster_smem_reduction or not cfg.has_single_q_cta
        )
        if uniform_kv:
            from .kernel.selection import compute_runtime_active_splits_kv

            q_step = cfg.q_tokens_per_cta if cfg.uses_nontrivial_grouped_q_layout else 1
            if cfg.use_split_kv and all(
                compute_runtime_active_splits_kv(
                    valid_k=(max_kv_len - seq_len_q + min(q_base + q_step, seq_len_q))
                    if mask_type == "causal"
                    else max_kv_len,
                    tile_size_kv=cfg.tile_size_kv,
                    num_insts_kv=cfg.num_insts_kv,
                    configured_splits_kv=spec.requested_splits,
                )
                == spec.requested_splits
                for q_base in range(0, seq_len_q, q_step)
            ):
                kv_prefix_mode = "planned_full"
                fixed_uniform_kv = True
            if fixed_uniform_kv:
                kv_lengths_mode = "planned_uniform_max"
            if fixed_uniform_kv and spec.requested_splits == cfg.splits_kv:
                # The live bound equals this declared topology; partial activity
                # is still derived from the owned lengths, as before runtime buckets.
                # Static kernels use the capacity-strided layout; same bytes
                # since requested == capacity.
                static_cfg = replace(cfg, runtime_split_kv=False)
                spec = replace(
                    spec,
                    config=static_cfg,
                    scratch_shapes=decode_scratch_shapes(
                        static_cfg,
                        batch_size=batch_size,
                        num_qo_heads=num_qo_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        seq_len_q=seq_len_q,
                    ),
                )
        compile_spec = make_decode_compile_spec(
            spec,
            device_index=device_index,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            q_dtype_key=attention_dtype_key(q_dtype),
            k_dtype_key=attention_dtype_key(k_dtype),
            v_dtype_key=attention_dtype_key(v_dtype),
            out_dtype_key=attention_dtype_key(out_dtype),
            packed_query=packed_query,
            kv_prefix_mode=kv_prefix_mode,
            kv_lengths_mode=kv_lengths_mode,
            lse_mode=lse_mode,
            scale_types=scale_types(softmax_scale, output_scale),
            runtime_kv_lengths=not fixed_uniform_kv
            and (cfg.runtime_split_kv or not cfg.use_split_kv),
        )
        compiled_main, compiled_reducer = get_compiled_decode(compile_spec)
        policy = spec.policy + (
            ("kv_prefix_mode", compile_spec.kv_prefix_mode),
            ("kv_lengths_mode", compile_spec.kv_lengths_mode),
            ("split_kv_allowed", split_kv),
        )
        workspace_layout = make_decode_workspace_layout(
            spec.scratch_shapes,
            out_dtype,
            use_separate_reduction_kernel=(spec.config.use_separate_reduction_kernel),
            use_split_kv=spec.config.use_split_kv,
            scratch_reserve=spec.scratch_reserve,
        )
        if workspace_buffer is None:
            initialize_workspace = True
            workspace_buffer = torch.empty(
                workspace_layout.total_bytes,
                device=device,
                dtype=torch.int8,
            )
        elif validate:
            validate_workspace_buffer(
                workspace_buffer,
                device=device,
                required_bytes=workspace_layout.total_bytes,
            )
        workspace = bind_decode_workspace(workspace_buffer, workspace_layout)
        if initialize_workspace:
            workspace.split_kv_counter.zero_()
            workspace.cu_seqlens_q.zero_()
            workspace.attention_sinks.zero_()
        # Materialize from the normalized tuple so the plan never aliases
        # caller-owned host or device storage.
        planned_seq_lens_kv_device = (
            None
            if planned_seq_lens is None
            else torch.tensor(
                planned_seq_lens,
                dtype=torch.int32,
                device=device,
            )
        )

        candidate = DecodePlanState(
            softmax_scale=softmax_scale,
            output_scale=output_scale,
            device=device,
            device_index=device_index,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            packed_query=packed_query,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            max_kv_len=max_kv_len,
            q_dtype=q_dtype,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            out_dtype=out_dtype,
            mask_type=mask_type,
            window_left=window_left,
            config=spec.config,
            workspace_buffer=workspace_buffer,
            workspace_layout=workspace_layout,
            workspace=workspace,
            compiled_main=compiled_main,
            compiled_reducer=compiled_reducer,
            kv_prefix_mode=compile_spec.kv_prefix_mode,
            kv_lengths_mode=compile_spec.kv_lengths_mode,
            lse_mode=lse_mode,
            planned_seq_lens_kv_host=planned_seq_lens,
            planned_seq_lens_kv_device=planned_seq_lens_kv_device,
            policy=policy,
            storage_page_size=storage_page_size,
            use_attention_sinks=use_attention_sinks,
            requested_splits=spec.requested_splits or int(spec.config.splits_kv),
            fixed_q_offsets=(
                None
                if packed_query
                else torch.arange(batch_size + 1, device=device, dtype=torch.int32)
                * seq_len_q
            ),
        )
        # This is the only wrapper mutation. Any failure above leaves the
        # previous complete plan revision usable.
        self.plan_state = candidate

    def require_plan_state(self) -> DecodePlanState:
        state = self.plan_state
        if state is None:
            raise RuntimeError("plan() must be called before run()")
        return state

    @flashinfer_experimental_api(trace=prims_ts_decode_wrapper_trace_dispatch)
    def run(
        self,
        q: torch.Tensor,
        paged_kv_cache: PagedKVCache,
        seq_lens_kv: Optional[torch.Tensor],
        block_tables: torch.Tensor,
        *,
        kv_scale_factors: Optional[PagedKVScaleFactors] = None,
        qo_indptr: Optional[torch.Tensor] = None,
        softmax_scale: Optional[GlobalScale] = None,
        output_scale: Optional[GlobalScale] = None,
        max_q_len: int | None = None,
        max_kv_len: int | None = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        attention_sinks: Optional[torch.Tensor] = None,
        validate: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Launch paged FMHA using live lengths and request metadata.

        Uses the policy and compiled kernels prepared by plan(). Batch and query
        shape must match that plan; live K/V lengths stay within its capacity.
        Supply output/LSE buffers and validate=False during graph capture, and
        retain the wrapper and captured buffers for every replay.

        An enabled lse_mode="base2" plan writes compact FP32 log2(sum(exp(scores)))
        with the query shape minus its final dimension. return_lse controls tuple
        return only; an omitted enabled buffer is allocated. Output/LSE/workspace must
        be disjoint from inputs and live buffers. One wrapper serves one ordered lane.
        ``max_q_len`` and ``max_kv_len`` are runtime launch operands.
        Bounds must cover the live request lengths without exceeding plan
        capacity. Validation infers omitted bounds from the metadata already
        read and checks supplied bounds. With ``validate=False``, bounds are
        trusted; omitted packed-Q and KV bounds use plan capacities, and
        fixed-Q bounds come from the query shape. Captured
        launches can therefore replay with new metadata while the kernel reads
        each request's live lengths from that metadata.

        Nonpositive per-run K/V lengths mark inactive requests in every
        profile. Their page-table rows and data are not read; caller-owned
        output is left untouched (newly allocated inactive output is unspecified).
        Producers and reducers retain scheduler/PDL progress. Plan-owned lengths
        may also mark inactive requests but remain immutable; use per-run lengths
        for empty/live graph transitions.

        ``validate=True`` performs structural and value validation. It
        reads per-run metadata values back to the host.
        ``validate=False`` treats every run argument as a trusted binding,
        skips metadata validation and remains free of metadata
        device-to-host synchronization. K/V view selection, scale forwarding,
        and optional output allocation are unavoidable in both modes.

        Supply seq_lens_kv on every run unless plan() owns its CUDA copy. Packed queries require qo_indptr; fixed Q
        rejects it under validation. Q/K/V dtypes, cache layout and model semantics
        must match plan(). softmax_scale is the effective QK scale in regular,
        not log2, units (default 1/sqrt(D)); output_scale is the V/output scale
        (default 1). Omitted scales inherit plan() values. Each override must
        keep its planned float/tensor type. Tensor scales are contiguous CUDA
        FP32 shape () or (1,) inputs on the plan device; their finite-positive values
        are a caller precondition and are never read back to the host. Scale
        metadata and argument types are checked in both validation modes.

        Enabled attention_sinks requires compact CUDA FP32 [Hq] natural-log logits
        after QK scaling. Values must be finite or -inf and add zero V contribution;
        validation checks metadata without reading sink values."""

        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")

        state = self.require_plan_state()
        softmax_scale = resolve_scale(
            softmax_scale, state.softmax_scale, "softmax_scale", state.device
        )
        output_scale = resolve_scale(
            output_scale, state.output_scale, "output_scale", state.device
        )
        if state.planned_seq_lens_kv_device is not None:
            if seq_lens_kv is not None:
                raise ValueError(
                    "run seq_lens_kv must be None when plan() owns sequence lengths"
                )
            effective_seq_lens = state.planned_seq_lens_kv_device
        else:
            if seq_lens_kv is None:
                raise ValueError(
                    "seq_lens_kv is required in run() when plan() does not own lengths"
                )
            effective_seq_lens = seq_lens_kv

        runtime_qo_indptr = qo_indptr if state.packed_query else None
        if validate:
            (
                metadata_device,
                metadata_batch_size,
                _,
            ) = validate_block_table_metadata(block_tables, effective_seq_lens)
            if metadata_device != state.device:
                raise ValueError(
                    f"per-run metadata must be on {state.device}, got {metadata_device}"
                )
            if metadata_batch_size != state.batch_size:
                raise ValueError(
                    "per-run metadata batch size must match the plan "
                    f"({state.batch_size}), got {metadata_batch_size}"
                )
            if state.packed_query:
                if qo_indptr is None:
                    raise ValueError("qo_indptr is required for a packed-Q plan")
                validate_qo_indptr(
                    qo_indptr,
                    expected_device=state.device,
                    batch_size=state.batch_size,
                )
            elif qo_indptr is not None:
                raise ValueError("qo_indptr cannot be used with a fixed-Q plan")

            runtime = prepare_decode_runtime(
                q,
                paged_kv_cache,
                kv_scale_factors,
                device=state.device,
                batch_size=state.batch_size,
                seq_len_q=state.seq_len_q,
                packed_query=state.packed_query,
                num_qo_heads=state.num_qo_heads,
                num_kv_heads=state.num_kv_heads,
                head_dim=state.head_dim,
                page_size=state.storage_page_size or state.page_size,
                q_dtype=state.q_dtype,
                k_dtype=state.k_dtype,
                v_dtype=state.v_dtype,
                out_dtype=state.out_dtype,
                softmax_scale=softmax_scale,
                output_scale=output_scale,
                out=out,
                kv_layout=self.kv_layout,
            )
            actual_q, actual_kv = validate_decode_run_metadata_values(
                runtime,
                planned_seq_lens_kv_host=state.planned_seq_lens_kv_host,
                max_kv_len=state.max_kv_len,
                page_size=state.page_size,
                packed_query=state.packed_query,
                seq_len_q=state.seq_len_q,
                batch_size=state.batch_size,
                mask_type=state.mask_type,
                seq_lens_kv=effective_seq_lens,
                block_tables=block_tables,
                qo_indptr=runtime_qo_indptr,
            )
            max_q_len = resolve_max_sequence_length(
                max_q_len, actual_q, "max_q_len", state.seq_len_q
            )
            max_kv_len = resolve_max_sequence_length(
                max_kv_len, actual_kv, "max_kv_len", state.max_kv_len
            )
        else:
            runtime = prepare_decode_runtime_unchecked(
                q,
                paged_kv_cache,
                kv_scale_factors,
                out_dtype=state.out_dtype,
                softmax_scale=softmax_scale,
                output_scale=output_scale,
                out=out,
                kv_layout=self.kv_layout,
            )

        if max_q_len is None:
            max_q_len = (
                state.seq_len_q
                if state.packed_query
                else (1 if q.ndim == 3 else int(q.shape[1]))
            )
        if max_kv_len is None:
            max_kv_len = state.max_kv_len

        lse = prepare_lse(
            lse,
            q=q,
            mode=state.lse_mode,
            return_lse=return_lse,
            validate=validate,
        )
        attention_sinks = validate_attention_sinks(
            attention_sinks,
            q=q,
            enabled=state.use_attention_sinks,
            validate=validate,
        )
        result = launch_decode(
            runtime,
            seq_lens_kv=effective_seq_lens,
            qo_indptr=(
                runtime_qo_indptr if state.packed_query else state.fixed_q_offsets
            ),
            block_tables=block_tables,
            workspace=state.workspace,
            compiled_main=state.compiled_main,
            compiled_reducer=state.compiled_reducer,
            lse=lse,
            attention_sinks=attention_sinks,
            requested_splits=state.requested_splits,
            max_kv_len=max_kv_len,
            max_q_len=max_q_len,
        )
        return (result, lse) if return_lse else result


@flashinfer_experimental_api(trace=attention_ts_decode_trace_dispatch)
def batch_decode_with_paged_kv_cache(
    q: torch.Tensor,
    paged_kv_cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    *,
    kv_scale_factors: Optional[PagedKVScaleFactors] = None,
    seq_len_q: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    mask_type: Literal["dense", "causal"] = "dense",
    window_left: int = -1,
    kv_layout: Literal["HND", "NHD"] = "HND",
    softmax_scale: Optional[GlobalScale] = None,
    output_scale: GlobalScale = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    attention_sinks: Optional[torch.Tensor] = None,
    lse_mode: Literal["base2"] = "base2",
    out_dtype: Optional[torch.dtype] = None,
    page_size: Optional[int] = None,
    split_kv: bool = True,
    workspace_buffer: Optional[torch.Tensor] = None,
    max_kv_len: Optional[int] = None,
    validate: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """One-shot fixed or packed-Q paged decode from fixed page tables.

    SQ1 preserves the ``[B, Hq, D]`` query/output contract. For fixed
    ``seq_len_q>1``, query and output are both token-major
    ``[B, SQ, Hq, D]``. Providing cumulative ``qo_indptr`` selects packed
    ``[total_q, Hq, D]`` query/output; packed Q requires ``max_q_len``
    or a non-default ``seq_len_q`` alias. No transpose is hidden here.

    Without caller scratch, the convenience planner reads ``seq_lens_kv`` on
    the host to size its plan and retain an owned metadata copy. This mode
    is not CUDA-graph-capturable. See Notes for the explicit capture-safe mode.

    Parameters
    ----------
    q : torch.Tensor
        Fixed or packed query tensor.
    paged_kv_cache : torch.Tensor or tuple[torch.Tensor, torch.Tensor]
        Combined or separate paged K/V storage. K and V dtypes are taken from
        the tensors; a ``torch.float8_e4m3fn`` V with ``torch.bfloat16`` Q/K
        requires the separate ``(K, V)`` form.
    block_tables : torch.Tensor
        Fixed row-strided ``[B, C]`` page table. Rows may have padding between
        them, but each row must be contiguous.
    seq_lens_kv : torch.Tensor
        Per-request K/V sequence lengths with shape ``[B]``.
    kv_scale_factors : tuple[torch.Tensor, torch.Tensor], optional
        Required for packed NVFP4 K/V. Compact FP8 ``(K_SF, V_SF)`` tensors
        have shape ``[pages, Hkv, storage_page_size, D/16]``; K scales are token-major
        and V scales use the 4-token interleaved layout. Omit for other dtypes.
    seq_len_q : int
        Fixed query length when ``qo_indptr`` is omitted. In packed-query mode,
        a non-default value is a backward-compatible alias for
        ``max_q_len`` and must agree with it when both are provided.
    qo_indptr : torch.Tensor, optional
        Cumulative query offsets selecting packed-query mode.
    max_q_len : int, optional
        Per-request packed-query length capacity. Required for packed Q
        unless a non-default ``seq_len_q`` supplies the bound. It is never
        inferred from ``qo_indptr``. In fixed-query mode, it must equal
        ``seq_len_q``.
    mask_type : {"dense", "causal"}
        Attention mask mode.
    window_left : int
        Left sliding-window extent, or ``-1`` to disable the window.
    kv_layout : {"HND", "NHD"}
        Layout of the paged K/V cache.
    softmax_scale, output_scale : float or torch.Tensor, optional
        Independent QK and value/output multipliers, defaulting to
        ``head_dim**-0.5`` and one. Each may be a host float or a contiguous CUDA
        FP32 tensor of shape () or (1,) on the query device. Softmax uses natural
        units. Tensor values must be finite and positive; they are read only
        by the kernel, without device-to-host synchronization.
    out : torch.Tensor, optional
        Caller-owned output tensor.
    out_dtype : torch.dtype, optional
        Output dtype; defaults to ``out.dtype`` or the query dtype.
    page_size : int, optional
        Semantic page-table size, defaulting to the physical cache extent.
        A smaller supported value must divide the physical extent and uses
        encoded subpage locators.
    split_kv : bool
        Allow automatic split fanout (True, default), or force S1 (False),
        independently of packed/fixed query storage.
    workspace_buffer : torch.Tensor, optional
        Zero-initialized caller-owned byte scratch; re-zero when layout inputs
        (including batch size) change. Exclusive to one in-flight launch/graph.
    max_kv_len : int, optional
        Static K/V capacity; defaults to the metadata maximum with validation.
    validate : bool
        Validate tensors and metadata values (may synchronize), default True.
        False trusts the caller and requires workspace and explicit bounds;
        skips tensor and metadata validation. Invalid inputs have undefined behavior.

    Returns
    -------
    torch.Tensor
        The fixed or packed attention output.
    Notes
    -----
    Both owned and caller-provided scratch use the same wrapper plan/run path.
    For CUDA Graph capture, supply ``workspace_buffer``, ``max_kv_len``,
    ``out``, and ``validate=False``; packed Q also needs an explicit
    ``max_q_len`` (or its non-default ``seq_len_q`` alias).
    Warm this exact topology outside capture first. Retain stable tensor storage
    and mutate metadata only between completed launches/replays. All live K/V
    positive lengths must be within the static bound; nonpositive lengths mark
    inactive requests. Active page IDs must
    index the cache, and causal per-request Q lengths must not exceed K/V lengths.
    Packed offsets must start at zero, end at the query token count, and have
    strictly positive deltas within the Q bound. Scratch/output must not alias
    any inputs or each other. No metadata is copied to the host on the trusted
    explicit path. Kernel policy, necessary control resets, and output layout
    are unchanged. Missing output may be allocated only outside capture.


    ``return_lse=True`` returns ``(output, lse)``. ``lse_mode`` selects
    ``"base2"`` (the only supported base); it is used only when returning LSE
    or writing a caller-provided ``lse`` buffer.

    ``lse`` is an optional independent, compact float32 output with shape
    ``query.shape[:-1]`` (``q.shape[:-1]`` for FMHA). An enabled LSE plan
    allocates it when omitted, including when ``return_lse=False``. A supplied
    buffer is written regardless of ``return_lse``; the flag only controls
    the Python return type. LSE includes any sink mass and excludes output
    scaling. Keep LSE disjoint from output, inputs, and workspace. Supply
    both output buffers and disable validation for allocation-free capture.

    ``attention_sinks`` is a compact CUDA float32 vector ``[Hq]`` shared
    across batch/query positions. Values are natural-log sink logits after
    QK scaling, with zero V contribution. Finite values and ``-inf`` are
    accepted. Validation checks the tensor metadata only, not its values;
    the caller guarantees the values, with or without validation.
    Sink values may change between completed CUDA graph replays.
    """

    if not isinstance(validate, bool):
        raise TypeError("validate must be a bool")
    if validate or workspace_buffer is None or out is None:
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "CUDA graph capture requires workspace_buffer, max_kv_len, "
                "out, validate=False, and explicit packed-Q bounds; warm up first"
            )
    if not validate:
        if workspace_buffer is None:
            raise ValueError("validate=False requires workspace_buffer")
        if max_kv_len is None:
            raise ValueError("validate=False requires max_kv_len")
        if qo_indptr is not None and max_q_len is None and seq_len_q == 1:
            raise ValueError("validate=False requires max_q_len for packed Q")

    validate_layout(kv_layout)
    allocate_workspace = workspace_buffer is None
    if validate:
        packed_query, resolved_seq_len_q = resolve_q_mode(
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_q_len=max_q_len,
            require_packed_max=True,
        )
        metadata_device, batch_size, _ = validate_block_table_metadata(
            block_tables, seq_lens_kv
        )
        if metadata_device != q.device:
            raise ValueError(
                f"paged-KV metadata must be on {q.device}, got {metadata_device}"
            )
        if qo_indptr is not None:
            validate_qo_indptr(
                qo_indptr, expected_device=q.device, batch_size=batch_size
            )
        assert resolved_seq_len_q is not None
        validate_q(
            q,
            seq_len_q=resolved_seq_len_q,
            packed_query=packed_query,
            device=q.device,
            batch_size=batch_size,
        )
        k_cache, v_cache, _, num_kv_heads, storage_page_size, head_dim = (
            normalize_paged_kv_cache_views(
                paged_kv_cache, expected_device=q.device, kv_layout=kv_layout
            )
        )
    else:
        packed_query = qo_indptr is not None
        resolved_seq_len_q = (
            max_q_len if packed_query and max_q_len is not None else seq_len_q
        )
        batch_size = int(seq_lens_kv.shape[0])
        k_cache = (
            paged_kv_cache[:, 0]
            if isinstance(paged_kv_cache, torch.Tensor)
            else paged_kv_cache[0]
        )
        v_cache = (
            paged_kv_cache[:, 1]
            if isinstance(paged_kv_cache, torch.Tensor)
            else paged_kv_cache[1]
        )
        if kv_layout == "NHD":
            k_cache = k_cache.transpose(1, 2)
            v_cache = v_cache.transpose(1, 2)
        num_kv_heads, storage_page_size, storage_head_dim = map(int, k_cache.shape[1:])
        head_dim = (
            storage_head_dim * 2 if k_cache.dtype == torch.uint8 else storage_head_dim
        )

    page_size = storage_page_size if page_size is None else page_size
    num_qo_heads = int(q.shape[-2])
    if validate and out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    out_dtype = (
        out_dtype
        if out_dtype is not None
        else (out.dtype if out is not None else q.dtype)
    )
    if max_kv_len is None:
        max_kv_len = int(seq_lens_kv.max().item())
    if validate and out is not None:
        validate_out(
            out,
            q=q,
            expected_shape=decode_output_shape(
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                seq_len_q=resolved_seq_len_q,
                head_dim=head_dim,
                total_q_tokens=int(q.shape[0]) if packed_query else None,
            ),
            seq_len_q=resolved_seq_len_q,
            packed_query=packed_query,
            out_dtype=out_dtype,
        )
    wrapper = BatchDecodePagedTSWrapper(kv_layout=kv_layout)
    wrapper.plan(
        q.device,
        batch_size=batch_size,
        max_q_len=resolved_seq_len_q,
        max_kv_len=max_kv_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        storage_page_size=storage_page_size,
        split_kv=split_kv,
        packed_query=packed_query,
        q_dtype=q.dtype,
        k_dtype=k_cache.dtype,
        v_dtype=v_cache.dtype,
        out_dtype=out_dtype,
        mask_type=mask_type,
        window_left=window_left,
        use_attention_sinks=attention_sinks is not None,
        lse_mode=validate_lse_mode(lse_mode)
        if return_lse or lse is not None
        else "none",
        workspace_buffer=workspace_buffer,
        validate=validate,
        initialize_workspace=allocate_workspace,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )
    return wrapper.run(
        q,
        paged_kv_cache,
        seq_lens_kv,
        block_tables,
        kv_scale_factors=kv_scale_factors,
        qo_indptr=qo_indptr,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        out=out,
        lse=lse,
        return_lse=return_lse,
        attention_sinks=attention_sinks,
        validate=validate,
    )


__all__ = [
    "BatchDecodePagedTSWrapper",
    "PrimsTSBatchDecodePlan",
    "batch_decode_with_paged_kv_cache",
    "get_prims_ts_batch_decode_workspace_size",
    "make_q_token_kv_block_sparse_qo_indptr",
    "suggest_q_token_kv_block_sparse_group_size",
    "validate_q_token_kv_block_sparse_group_size",
    "prepare_prims_ts_batch_decode_with_kv_cache",
]
