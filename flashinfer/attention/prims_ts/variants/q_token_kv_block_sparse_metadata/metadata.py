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

"""QToken-KvBlock-Sparse metadata: workspace layout, plan state and the metadata build."""

import functools
import math
import torch
from dataclasses import dataclass, replace
from ..block_sparse.common import num_sparse_pattern_heads
from ...common.scales import GlobalScale, resolve_scale
from .selection import membership_planes
from flashinfer.utils import device_support_pdl
from typing import Any, Optional


MIN_BLOCK_SIZE = 4
BLOCK_SIZES = (4, 8, 16, 32, 64, 128)
MAX_BLOCK_TOPK = 512
# Reserve three high key bits for grouped-query membership tags and retain an
# out-of-range radix sentinel in the CUDA/CUB touched-union representation.
MAX_SEQ_LEN_KV = ((1 << 29) - 1) * MIN_BLOCK_SIZE
MEMBERSHIPS_PER_WORD = 4
INT32_LOCATOR_CAPACITY = 1 << 31
METADATA_WORKSPACE_ALIGNMENT = 256


@dataclass(frozen=True)
class MetadataPlan:
    """Unchecked metadata launch state with all geometry pre-resolved."""

    q_token_kv_block_sparse_page_indices: torch.Tensor
    q_token_kv_block_sparse_page_memberships: torch.Tensor
    seq_lens: torch.Tensor
    qo_indptr: Optional[torch.Tensor]
    metadata_run: Any
    group_size: int
    packed_query: bool
    sparse_block_size: int
    storage_page_size: int
    max_seq_len_kv: int
    release_attention_pdl: bool
    locator_head_multiplier: int = 1
    count_current_block_in_topk: bool = True
    fixed_query_len: int = 0
    # Longest-first route order for persistent attention, and the plan-owned
    # completion counter the metadata grid rearms after each launch.
    route_order: Optional[torch.Tensor] = None
    route_order_counter: Optional[torch.Tensor] = None
    # Shortest routes reordered at the end (0: all routes, longest first).
    route_order_tail: int = 0

    def run(
        self,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: Optional[torch.Tensor],
        query_positions: torch.Tensor,
    ) -> None:
        """Launch the prepared CUDA metadata kernel without shape resolution."""

        if self.packed_query:
            self.metadata_run(
                block_indices,
                block_table,
                token_to_request,
                query_positions,
                self.qo_indptr,
                self.q_token_kv_block_sparse_page_indices,
                self.q_token_kv_block_sparse_page_memberships,
                self.seq_lens,
                self.group_size,
                self.storage_page_size,
                self.sparse_block_size,
                self.max_seq_len_kv,
                self.release_attention_pdl,
                self.locator_head_multiplier,
                self.count_current_block_in_topk,
                self.fixed_query_len,
                self.route_order,
                self.route_order_counter,
                self.route_order_tail,
            )
        else:
            self.metadata_run(
                block_indices,
                block_table,
                token_to_request,
                query_positions,
                self.q_token_kv_block_sparse_page_indices,
                self.q_token_kv_block_sparse_page_memberships,
                self.seq_lens,
                self.group_size,
                self.storage_page_size,
                self.sparse_block_size,
                self.max_seq_len_kv,
                self.release_attention_pdl,
                self.locator_head_multiplier,
                self.count_current_block_in_topk,
                self.fixed_query_len,
            )


@dataclass(frozen=True)
class PlanConfig:
    """Capacity and dtype contract published by the public wrapper."""

    batch_size: int
    group_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    kv_block_size: int
    page_size: int
    block_topk: int
    max_seq_len_kv: int
    device: torch.device
    workspace_buffer: torch.Tensor
    packed_query: bool
    split_kv: bool
    share_pattern_across_kv_heads: bool
    count_current_block_in_topk: bool
    balanced_scheduler: Optional[bool]
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    out_dtype: torch.dtype
    sm_scale: GlobalScale
    v_scale: GlobalScale


@dataclass(frozen=True)
class TensorDescriptor:
    """Structural replacement-storage contract for a prepared QToken-KvBlock-Sparse-Attention plan."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class SparsePlan:
    """Prepared compact-metadata and PrimTS-attention launch state.

    The plan binds output/workspace storage and freezes input geometry once.
    Eager calls may pass new input storage with the same shapes, strides,
    devices, and dtypes; CUDA graph replay retains its usual stable-address
    requirement. The hot path performs synchronization-free replacement-storage
    checks followed by already-prepared launches. G1 resolves logical blocks
    directly inside attention, without launching the metadata kernel; grouped
    routes first build their union. The
    original tensors are validated during preparation. Callers must keep all
    replacement tensors alive and disjoint, and must not mutate them
    concurrently with a launch or CUDA-graph replay that reads them. Call
    :meth:`run` once before capture so both metadata and attention kernels are
    compiled and their workspace state is initialized outside the graph.
    """

    metadata_plan: MetadataPlan
    bmm1_scale: GlobalScale
    bmm2_scale: GlobalScale
    attention_plan: Any
    query: TensorDescriptor
    block_indices: TensorDescriptor
    block_table: TensorDescriptor
    token_to_request: Optional[TensorDescriptor]
    query_positions: TensorDescriptor
    out: TensorDescriptor
    fixed_query_group_size: Optional[int] = None

    def run(
        self,
        query: torch.Tensor,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: Optional[torch.Tensor],
        query_positions: torch.Tensor,
        *,
        out: torch.Tensor,
        sm_scale: Optional[GlobalScale] = None,
        v_scale: Optional[GlobalScale] = None,
    ) -> torch.Tensor:
        """Launch prepared QToken-KvBlock-Sparse-Attention metadata and attention on the current stream."""

        validate_plan_tensor(query, "query", self.query)
        validate_plan_tensor(block_indices, "block_indices", self.block_indices)
        validate_plan_tensor(block_table, "block_table", self.block_table)
        if self.token_to_request is None:
            if token_to_request is not None:
                raise ValueError("the fixed-Q plan derives request IDs")
        else:
            if token_to_request is None:
                raise ValueError("the prepared plan requires explicit request IDs")
            validate_plan_tensor(
                token_to_request,
                "token_to_request",
                self.token_to_request,
            )
        validate_plan_tensor(
            query_positions,
            "query_positions",
            self.query_positions,
        )
        validate_plan_tensor(out, "out", self.out)

        from ...common.tensors import validate_16byte_alignment

        validate_16byte_alignment(query, "query")
        validate_16byte_alignment(out, "out")

        scale_qk = resolve_scale(sm_scale, self.bmm1_scale, "sm_scale", query.device)
        scale_v = resolve_scale(v_scale, self.bmm2_scale, "v_scale", query.device)
        return self.run_unchecked(
            query,
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            out=out,
            sm_scale=scale_qk,
            v_scale=scale_v,
        )

    def run_unchecked(
        self,
        query: torch.Tensor,
        block_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_request: Optional[torch.Tensor],
        query_positions: torch.Tensor,
        *,
        out: torch.Tensor,
        sm_scale: GlobalScale,
        v_scale: GlobalScale,
    ) -> torch.Tensor:
        """Launch tensors and scales that satisfy the prepared plan's ABI."""
        direct_q1_inputs = None
        if self.attention_plan.direct_q1_inputs:
            direct_q1_inputs = (
                block_indices,
                block_table,
                token_to_request,
                query_positions,
            )
        else:
            self.metadata_plan.run(
                block_indices,
                block_table,
                token_to_request,
                query_positions,
            )
        attention_query = flatten_fixed_groups(query, self.fixed_query_group_size)
        attention_out = flatten_fixed_groups(out, self.fixed_query_group_size)
        self.attention_plan.run_unchecked(
            attention_query,
            attention_out,
            sm_scale,
            v_scale,
            direct_q1_inputs,
        )
        return out


@dataclass(frozen=True)
class WorkspaceViews:
    """Typed views bound to one caller-owned QToken-KvBlock-Sparse-Attention attention workspace."""

    q_token_kv_block_sparse_page_indices: torch.Tensor
    q_token_kv_block_sparse_page_memberships: torch.Tensor
    seq_lens: torch.Tensor
    attention_workspace_buffer: torch.Tensor
    route_order: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class WorkspaceLayout:
    """Layout of QToken-KvBlock-Sparse-Attention metadata outputs and disjoint kernel scratch.

    The dense page-index and packed-membership tables plus compact sequence
    lengths remain live from metadata construction through the attention
    launch. A decode split-KV plan must
    preserve its partials and self-resetting completion counters independently
    of metadata. A direct prefill plan has no split-KV storage; its small
    attention region contains only uniform-call-ABI placeholder tensors.
    Semantic inputs such as block tables and query mappings remain outside.
    """

    q_token_kv_block_sparse_page_indices_shape: tuple[int, int]
    q_token_kv_block_sparse_page_indices_bytes: int
    q_token_kv_block_sparse_page_memberships_shape: tuple[int, int]
    q_token_kv_block_sparse_page_memberships_byte_offset: int
    q_token_kv_block_sparse_page_memberships_bytes: int
    seq_lens_byte_offset: int
    seq_lens_bytes: int
    route_order_bytes: int
    attention_workspace_byte_offset: int
    attention_scratch_bytes: int
    uses_split_kv: bool
    query_major_memberships: bool
    max_seq_len: int
    total_bytes: int

    def bind(self, workspace_buffer: torch.Tensor) -> WorkspaceViews:
        """Return zero-copy typed views over a validated byte workspace."""

        validate_attention_workspace(workspace_buffer, self.total_bytes)
        workspace_bytes = workspace_buffer.reshape(-1).view(torch.uint8)
        q_token_kv_block_sparse_page_indices = (
            workspace_bytes[: self.q_token_kv_block_sparse_page_indices_bytes]
            .view(torch.int32)
            .view(self.q_token_kv_block_sparse_page_indices_shape)
        )
        q_token_kv_block_sparse_page_memberships = (
            workspace_bytes[
                self.q_token_kv_block_sparse_page_memberships_byte_offset : (
                    self.q_token_kv_block_sparse_page_memberships_byte_offset
                    + self.q_token_kv_block_sparse_page_memberships_bytes
                )
            ]
            .view(torch.int32)
            .view(self.q_token_kv_block_sparse_page_memberships_shape)
        )
        seq_lens = workspace_bytes[
            self.seq_lens_byte_offset : self.seq_lens_byte_offset + self.seq_lens_bytes
        ].view(torch.int32)
        route_order_byte_offset = self.seq_lens_byte_offset + self.seq_lens_bytes
        route_order = (
            workspace_bytes[
                route_order_byte_offset : route_order_byte_offset
                + self.route_order_bytes
            ].view(torch.int32)
            if self.route_order_bytes
            else None
        )
        attention_workspace_buffer = workspace_bytes[
            self.attention_workspace_byte_offset : self.attention_workspace_byte_offset
            + self.attention_scratch_bytes
        ]
        return WorkspaceViews(
            q_token_kv_block_sparse_page_indices=q_token_kv_block_sparse_page_indices,
            q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
            seq_lens=seq_lens,
            attention_workspace_buffer=attention_workspace_buffer,
            route_order=route_order,
        )


def align_up_workspace(value: int) -> int:
    return (
        (value + METADATA_WORKSPACE_ALIGNMENT - 1)
        // METADATA_WORKSPACE_ALIGNMENT
        * METADATA_WORKSPACE_ALIGNMENT
    )


def build_metadata(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    group_size: int,
    storage_page_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int = 4,
    qo_indptr: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    share_pattern_across_kv_heads: bool = True,
    count_current_block_in_topk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dense QToken-KvBlock-Sparse-Attention page indices and memberships from sparse-block IDs.

    ``block_indices`` is ``[T,K]`` when shared, otherwise ``[T,Hkv,K]``.
    ``block_table`` remains the dense ``[requests, max_storage_pages]`` physical
    table. Let F = gcd(sparse_block_size, storage_page_size). Output rows number
    groups * pattern_heads, with width G * (K+1) * (sparse_block_size/F).
    ``seq_lens`` gives each row's live token prefix. Memberships pack four
    fragment bytes per Int32; G1 has zero membership columns. Every locator is
    physical_page * (storage_page_size/F) + subpage. No CSR conversion is needed.

    Q1 uses one CUDA C++ direct-mapping kernel. Q2--Q8 use one CUDA C++
    map-or-sort union kernel per route. Neither path requires
    caller-provided scratch. Advanced callers that capture this raw path must
    preallocate ``out``, warm the same metadata geometry once before capture,
    and retain every tensor at a stable address through replay. The three
    ``out`` storage ranges must be mutually disjoint and must not alias inputs.
    Packed ``qo_indptr`` must be an Int32 device copy of CPU-validated route
    offsets and is checked structurally without a host-side value read.

    ``sparse_block_size`` defaults to four. The API validates power-of-two
    values in {4,8,16,32,64,128}. Logical blocks may cross physical cache
    pages; metadata emits only the physical fragments the loader needs.

    Parameters
    ----------
    block_indices : torch.Tensor
        CUDA Int32 selected logical sparse-block IDs shaped ``[T,K]`` or
        ``[T,Hkv,K]``, with a contiguous K dimension.
    block_table : torch.Tensor
        Dense CUDA Int32 physical-page table shaped
        ``[num_requests, max_storage_pages]`` with contiguous rows.
    token_to_request : torch.Tensor
        Contiguous CUDA Int32 request index for each flattened query row,
        shaped ``[num_query_tokens]``.
    query_positions : torch.Tensor
        Contiguous CUDA Int32 or Int64 absolute position for each flattened
        query row, shaped ``[num_query_tokens]``.
    group_size : int
        Maximum number of query rows represented by one route.
    storage_page_size : int
        Number of tokens in each physical K/V storage page.
    max_seq_len_kv : int
        Static maximum visible logical K/V length in tokens, including current
        query or MTP tokens. It must fit within each dense block-table row.
    sparse_block_size : int
        Logical sparse-block size in tokens: 4, 8, 16, 32, 64, or 128.
    qo_indptr : torch.Tensor, optional
        Contiguous CUDA Int32 cumulative route offsets for packed queries. If
        omitted, adjacent rows form fixed groups of exactly ``group_size``.
    out : tuple[torch.Tensor, torch.Tensor, torch.Tensor], optional
        Preallocated contiguous CUDA Int32 tensors for page indices, packed
        memberships, and live sequence lengths. Their shapes must match
        :func:`get_metadata_output_shapes`. If omitted, the three
        outputs are allocated internally.

    share_pattern_across_kv_heads : bool
        True (default) uses one pattern per query, shared by KV heads.
        False uses independent per-KV-head patterns. Match this setting in
        workspace sizing and planning; it does not change the attention grid.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens)``. Memberships
        pack four per-page bytes into each Int32 word and have zero columns for
        Q1.
    """

    return build_plan_metadata(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        sparse_block_size=sparse_block_size,
        qo_indptr=qo_indptr,
        out=out,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        count_current_block_in_topk=count_current_block_in_topk,
    )


def build_plan_metadata(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    group_size: int,
    storage_page_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int = 4,
    qo_indptr: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    release_attention_pdl: bool = False,
    share_pattern_across_kv_heads: bool = True,
    count_current_block_in_topk: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build dense QToken-KvBlock-Sparse-Attention indices and memberships from compact sparse-block IDs.

    ``block_indices`` contains logical sparse-block IDs with shape
    ``[num_query_tokens, block_topk]`` when shared, or ``[num_query_tokens,
    Hkv, block_topk]`` for per-head selections.  Adjacent Q2--Q8 rows must belong to
    the same request and have consecutive positions.  Q1 output locators are
    ordinary encoded subpage locators. Grouped routes return the same plain
    locators plus a separate Int32 table containing four 8-bit membership masks
    per word; membership bit ``i`` marks visibility for query ``i`` in the
    group. Valid locators must fit in nonnegative signed Int32. The combined
    attention APIs prove this from the cache capacity, while callers of this
    advanced metadata-only API own that input-value contract.

    Pass preallocated ``out`` tensors for CUDA graph capture. The page-index
    table has a fixed row width of
    ``group_size * (block_topk + 1) * fragments_per_block``; ``seq_lens``
    selects the live token prefix. Fragments divide both logical blocks and
    storage pages, so physical page IDs never imply logical adjacency.
    The membership output has one Int32 word per four page slots, or zero
    columns for Q1.
    Q1 uses one CUDA C++ CTA per route to map its selected blocks and causal
    tail directly. Q2--Q8 union the bounded
    ``group_size * (block_topk + 1)`` candidates with a shared map or radix
    sort, OR membership bits, and map the sorted unique logical blocks through
    the dense block table. A map is used only when it fits shared memory and
    requires no extra CTA waves. The combined API releases attention at
    metadata entry; the consumer wait still covers every output store.
    Packed ``qo_indptr`` must be an Int32 device copy of CPU-validated route
    offsets; this builder checks only its structural tensor contract and does
    not read route values back to the host.
    """

    validate_metadata_inputs(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size,
        storage_page_size,
        max_seq_len_kv,
        sparse_block_size,
        share_pattern_across_kv_heads,
        count_current_block_in_topk,
    )
    rows, block_topk = block_indices.shape[0], block_indices.shape[-1]
    pattern_heads = block_indices.shape[1] if block_indices.ndim == 3 else 1
    packed_query = qo_indptr is not None
    if packed_query:
        validate_qo_indptr_tensor(
            qo_indptr,
            expected_device=block_indices.device,
        )
    groups = int(qo_indptr.numel()) - 1 if packed_query else None
    expected_shapes = get_metadata_output_shapes(
        rows,
        block_topk,
        group_size,
        num_query_groups=groups,
        sparse_block_size=sparse_block_size,
        storage_page_size=storage_page_size,
        num_pattern_heads=pattern_heads,
    )
    if out is None:
        outputs = tuple(
            torch.empty(shape, dtype=torch.int32, device=block_indices.device)
            for shape in expected_shapes
        )
    else:
        if not isinstance(out, tuple) or len(out) != 3:
            raise TypeError(
                "out must be a (q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens) tuple"
            )
        outputs = out
        for tensor, shape in zip(outputs, expected_shapes, strict=True):
            if (
                tensor.shape != shape
                or tensor.dtype != torch.int32
                or tensor.device != block_indices.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "QToken-KvBlock-Sparse-Attention metadata outputs must be contiguous int32 tensors with "
                    f"shapes {expected_shapes} on {block_indices.device}"
                )
    (
        q_token_kv_block_sparse_page_indices,
        q_token_kv_block_sparse_page_memberships,
        seq_lens,
    ) = outputs
    groups = resolve_num_query_groups(rows, group_size, groups)
    if groups == 0:
        return outputs

    validate_wide_metadata_fits(
        group_size, max_seq_len_kv, sparse_block_size, block_indices.device.index
    )
    enable_pdl = device_support_pdl(block_indices.device)
    metadata_module = get_metadata_module()
    release_attention_pdl = enable_pdl and release_attention_pdl
    if packed_query:
        metadata_module.run_packed(
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            qo_indptr,
            q_token_kv_block_sparse_page_indices,
            q_token_kv_block_sparse_page_memberships,
            seq_lens,
            group_size,
            storage_page_size,
            sparse_block_size,
            max_seq_len_kv,
            release_attention_pdl,
            1,
            count_current_block_in_topk,
            0,
            None,
            None,
            0,
        )
    else:
        metadata_module.run_fixed(
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            q_token_kv_block_sparse_page_indices,
            q_token_kv_block_sparse_page_memberships,
            seq_lens,
            group_size,
            storage_page_size,
            sparse_block_size,
            max_seq_len_kv,
            release_attention_pdl,
            1,
            count_current_block_in_topk,
            0,
        )
    return outputs


def describe_tensor(
    tensor: torch.Tensor,
) -> TensorDescriptor:
    return TensorDescriptor(
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        device=tensor.device,
        dtype=tensor.dtype,
    )


def flatten_fixed_groups(
    tensor: torch.Tensor,
    group_size: Optional[int],
) -> torch.Tensor:
    """Return the lower-level decode view for a canonical fixed QToken-KvBlock-Sparse-Attention tensor."""

    if group_size is None:
        return tensor
    flattened = tensor.flatten(0, 1)
    return flattened.squeeze(1) if group_size == 1 else flattened


@functools.cache
def get_metadata_module() -> Any:
    """Build and cache the CUDA C++ QToken-KvBlock-Sparse-Attention metadata implementation."""

    from .....jit import gen_prims_ts_q_token_kv_block_sparse_metadata_module

    return gen_prims_ts_q_token_kv_block_sparse_metadata_module().build_and_load()


def get_metadata_output_shapes(
    num_query_tokens: int,
    block_topk: int,
    group_size: int,
    *,
    num_query_groups: Optional[int] = None,
    sparse_block_size: int = 4,
    storage_page_size: Optional[int] = None,
    num_pattern_heads: int = 1,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return ``(q_token_kv_block_sparse_page_indices, q_token_kv_block_sparse_page_memberships, seq_lens)`` shapes.

    Physical fragments have size gcd(sparse_block_size, storage_page_size).
    Metadata has one row per query group and pattern head. Grouped masks use
    four packed bytes per Int32 word. Q1 does not consume membership metadata
    and returns a zero-width membership shape.

    Parameters
    ----------
    num_query_tokens : int
        Number of flattened query rows represented by the metadata.
    block_topk : int
        Number of selected logical sparse blocks supplied for each query row.
    group_size : int
        Maximum number of query rows represented by one QToken-KvBlock-Sparse-Attention route.
    num_query_groups : int, optional
        Explicit route count for a packed query. If omitted, query rows are
        partitioned into fixed groups and ``num_query_tokens`` must be
        divisible by ``group_size``.
    sparse_block_size : int
        Logical sparse-block size in tokens: 4, 8, 16, 32, 64, or 128.
    storage_page_size : int, optional
        Physical storage-page size; defaults to sparse_block_size. A block
        contains sparse_block_size / gcd(sparse_block_size, storage_page_size)
        independently addressable fragments.
    num_pattern_heads : int
        One for shared patterns, otherwise the number of KV heads.

    Returns
    -------
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
        Shapes for ``q_token_kv_block_sparse_page_indices``, ``q_token_kv_block_sparse_page_memberships``, and
        ``seq_lens``, in that order.
    """

    validate_sparse_block_size(sparse_block_size)
    validate_shape_parameters(num_query_tokens, block_topk, group_size)
    groups = resolve_num_query_groups(
        num_query_tokens,
        group_size,
        num_query_groups,
    )
    if storage_page_size is None:
        storage_page_size = sparse_block_size
    validate_metadata_storage_page_size(storage_page_size)
    fragments_per_block = sparse_block_size // math.gcd(
        sparse_block_size, storage_page_size
    )
    page_capacity = group_size * (block_topk + 1) * fragments_per_block
    groups *= num_pattern_heads
    membership_words = (
        0
        if group_size == 1
        # G9-G32: ceil(G / 8) byte planes per route row (queries 8p..8p+7 in plane p).
        else membership_planes(group_size)
        * ((page_capacity + MEMBERSHIPS_PER_WORD - 1) // MEMBERSHIPS_PER_WORD)
    )
    return (groups, page_capacity), (groups, membership_words), (groups,)


def get_workspace_layout(
    num_query_tokens: int,
    block_topk: int,
    max_num_storage_pages: int,
    storage_page_size: int,
    group_size: int,
    *,
    max_seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_dtype: torch.dtype = torch.float16,
    kv_dtype: Optional[torch.dtype] = None,
    out_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device | str | int] = None,
    num_query_groups: Optional[int] = None,
    packed_query: bool = False,
    sparse_block_size: int = 4,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
    count_current_block_in_topk: bool = True,
) -> WorkspaceLayout:
    """Return the unified allocation layout for QToken-KvBlock-Sparse-Attention metadata and attention.

    The workspace owns the dense QToken-KvBlock-Sparse-Attention page-index table, grouped membership
    words, compact sequence lengths, and kernel scratch. The caller continues
    to provide the model block table, request mappings, and query positions as
    explicit semantic inputs.

    Metadata and attention use disjoint regions. The caller controls whether
    split-KV may be selected, independently of query layout. Split launches
    own dedicated partial-output, statistics and completion-counter storage.
    """

    sparse_block_size = validate_sparse_block_size(sparse_block_size)
    validate_shape_parameters(num_query_tokens, block_topk, group_size)
    if num_query_tokens == 0:
        raise ValueError(
            "num_query_tokens must be positive for QToken-KvBlock-Sparse-Attention attention"
        )
    if not isinstance(split_kv, bool):
        raise TypeError("split_kv must be a bool")
    if not isinstance(max_num_storage_pages, int) or isinstance(
        max_num_storage_pages, bool
    ):
        raise TypeError("max_num_storage_pages must be an integer")
    if max_num_storage_pages <= 0:
        raise ValueError("max_num_storage_pages must be positive")
    validate_metadata_storage_page_size(storage_page_size)
    max_seq_len_kv = validate_max_seq_len_kv(
        max_seq_len_kv,
        block_table_token_capacity=max_num_storage_pages * storage_page_size,
    )

    from ...common.tensors import attention_dtype_key, resolve_cuda_device
    from ..decode.launch import make_decode_workspace_layout, resolve_decode_launch_spec
    from .grouping import validate_group_capacity

    group_size = validate_group_capacity(
        group_size,
        num_qo_heads,
        num_kv_heads,
    )

    groups = resolve_num_query_groups(
        num_query_tokens,
        group_size,
        num_query_groups,
    )
    if not isinstance(count_current_block_in_topk, bool):
        raise TypeError("count_current_block_in_topk must be a bool")
    if not isinstance(share_pattern_across_kv_heads, bool):
        raise TypeError("share_pattern_across_kv_heads must be a bool")
    pattern_heads = num_sparse_pattern_heads(
        num_kv_heads, share_pattern_across_kv_heads
    )
    metadata_rows = groups * pattern_heads
    fragment_size = math.gcd(sparse_block_size, storage_page_size)
    block_capacity = min(
        group_size * (block_topk + 1),
        (max_seq_len_kv + sparse_block_size - 1) // sparse_block_size,
    )
    page_capacity = block_capacity * (sparse_block_size // fragment_size)
    # Padding is an allocation property, not additional selected KV work.
    max_seq_len = (
        block_topk * sparse_block_size
        + sparse_block_size
        - int(count_current_block_in_topk)
        if group_size == 1
        else page_capacity * fragment_size
    )
    max_seq_len = min(max_seq_len, max_seq_len_kv)
    if kv_dtype is None:
        kv_dtype = q_dtype
    if out_dtype is None:
        out_dtype = q_dtype
    _, device_index = resolve_cuda_device(device)
    validate_wide_metadata_fits(
        group_size, max_seq_len_kv, sparse_block_size, device_index
    )
    spec = resolve_decode_launch_spec(
        device_index,
        groups,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        fragment_size,
        max_seq_len,
        group_size,
        attention_dtype_key(q_dtype),
        attention_dtype_key(kv_dtype),
        attention_dtype_key(kv_dtype),
        attention_dtype_key(out_dtype),
        "HND",
        "causal",
        packed_query,
        -1,
        storage_page_size,
        True,
        True,
        split_kv,
        share_pattern_across_kv_heads,
    )
    # Eligible Keeps G2..G8/page-4 profiles consume one bit-plane per query.
    # Other profiles and the standalone metadata API retain page-byte masks.
    query_major_memberships = (
        sparse_block_size == 4
        and replace(spec.config, fixed_seqlen=True).selects_query_major_memberships
    )
    if query_major_memberships:
        page_capacity = (page_capacity + 31) // 32 * 32
    membership_words = (
        0
        if group_size == 1
        # G9-G32: ceil(G / 8) byte planes per route row (queries 8p..8p+7 in plane p).
        else membership_planes(group_size)
        * ((page_capacity + MEMBERSHIPS_PER_WORD - 1) // MEMBERSHIPS_PER_WORD)
    )
    q_token_kv_block_sparse_page_indices_shape = (metadata_rows, page_capacity)
    q_token_kv_block_sparse_page_indices_bytes = metadata_rows * page_capacity * 4
    q_token_kv_block_sparse_page_memberships_shape = (metadata_rows, membership_words)
    q_token_kv_block_sparse_page_memberships_byte_offset = align_up_workspace(
        q_token_kv_block_sparse_page_indices_bytes
    )
    q_token_kv_block_sparse_page_memberships_bytes = (
        metadata_rows * membership_words * 4
    )
    seq_lens_byte_offset = align_up_workspace(
        q_token_kv_block_sparse_page_memberships_byte_offset
        + q_token_kv_block_sparse_page_memberships_bytes
    )
    seq_lens_numel = metadata_rows
    seq_lens_bytes = seq_lens_numel * 4
    attention_layout = make_decode_workspace_layout(
        spec.scratch_shapes,
        out_dtype,
        use_separate_reduction_kernel=spec.config.use_separate_reduction_kernel,
        use_split_kv=spec.config.use_split_kv,
        scratch_reserve=spec.scratch_reserve,
    )
    attention_scratch_bytes = attention_layout.total_bytes
    uses_split_kv = attention_layout.uses_split_kv
    # Persistent grouped launches that share one pattern across KV heads take
    # routes longest first: the metadata writes their order right after
    # seq_lens, where the attention work queue reads it (seq_lens + routes).
    # Split-KV and static grids, and per-head patterns (even with one KV
    # head), read none; the same config properties gate the planner.
    route_order_bytes = (
        metadata_rows * 4
        if packed_query
        and group_size > 1
        and spec.config.shares_sparse_pattern
        and spec.config.use_persistent_scheduler
        else 0
    )
    attention_workspace_byte_offset = align_up_workspace(
        seq_lens_byte_offset + seq_lens_bytes + route_order_bytes
    )
    return WorkspaceLayout(
        q_token_kv_block_sparse_page_indices_shape=q_token_kv_block_sparse_page_indices_shape,
        q_token_kv_block_sparse_page_indices_bytes=q_token_kv_block_sparse_page_indices_bytes,
        q_token_kv_block_sparse_page_memberships_shape=q_token_kv_block_sparse_page_memberships_shape,
        q_token_kv_block_sparse_page_memberships_byte_offset=q_token_kv_block_sparse_page_memberships_byte_offset,
        q_token_kv_block_sparse_page_memberships_bytes=q_token_kv_block_sparse_page_memberships_bytes,
        seq_lens_byte_offset=seq_lens_byte_offset,
        seq_lens_bytes=seq_lens_bytes,
        route_order_bytes=route_order_bytes,
        attention_workspace_byte_offset=attention_workspace_byte_offset,
        attention_scratch_bytes=attention_scratch_bytes,
        uses_split_kv=uses_split_kv,
        query_major_memberships=query_major_memberships,
        max_seq_len=max_seq_len,
        total_bytes=attention_workspace_byte_offset + attention_scratch_bytes,
    )


def get_workspace_layout_from_tensors(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_topk: int,
    *,
    max_seq_len_kv: int,
    out_dtype: Optional[torch.dtype],
    qo_indptr: Optional[torch.Tensor] = None,
    group_size: int = 1,
    sparse_block_size: int = 4,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
    count_current_block_in_topk: bool = True,
) -> WorkspaceLayout:
    """Infer query geometry and size scratch for the selected sparse-block policy."""
    sparse_block_size = validate_sparse_block_size(sparse_block_size)
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if not query.is_cuda:
        raise ValueError("query must be a CUDA tensor")
    if not isinstance(k_cache, torch.Tensor):
        raise TypeError("k_cache must be a torch.Tensor")
    if not isinstance(block_table, torch.Tensor):
        raise TypeError("block_table must be a torch.Tensor")
    packed_query = qo_indptr is not None
    if packed_query:
        if query.ndim != 3:
            raise ValueError(
                "packed QToken-KvBlock-Sparse-Attention query must have shape [total_q,Hq,D]"
            )
        validate_qo_indptr_layout_tensor(qo_indptr)
        groups = int(qo_indptr.numel()) - 1
        num_query_tokens, num_qo_heads, head_dim = query.shape
    elif query.ndim == 5:
        batch_size, groups_per_request, query_group_size, num_qo_heads, head_dim = (
            query.shape
        )
        if group_size != query_group_size:
            raise ValueError("fixed group_size must match query.shape[2]")
        if batch_size <= 0 or groups_per_request <= 0:
            raise ValueError(
                "fixed QToken-KvBlock-Sparse-Attention batch and query-group counts must be positive"
            )
        groups = int(batch_size) * int(groups_per_request)
    else:
        raise ValueError(
            "query must be packed [total_q,Hq,D] with qo_indptr or fixed "
            "[B,Nq,G,Hq,D] without qo_indptr"
        )
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have shape [pages, Hkv, storage_page_size, D]")
    # Packed NVFP4 caches store two E2M1 values per byte, optionally followed
    # by the token's interleaved E4M3 block scales.
    cache_head_dims = (
        (head_dim // 2, head_dim // 2 + head_dim // 16)
        if k_cache.dtype == torch.uint8
        else (head_dim,)
    )
    if k_cache.device != query.device or k_cache.shape[3] not in cache_head_dims:
        raise ValueError("query and k_cache must share device and head dimension")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be a rank-two int32 tensor")
    if block_table.device != query.device:
        raise ValueError("block_table must be on the query device")
    validate_locator_capacity(
        k_cache,
        sparse_block_size=sparse_block_size,
    )
    if out_dtype is None:
        out_dtype = query.dtype
    return get_workspace_layout(
        int(query.shape[0]) if packed_query else groups * group_size,
        block_topk,
        block_table.shape[1],
        k_cache.shape[2],
        group_size,
        max_seq_len_kv=max_seq_len_kv,
        num_qo_heads=num_qo_heads,
        num_kv_heads=k_cache.shape[1],
        head_dim=head_dim,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=out_dtype,
        device=query.device,
        num_query_groups=groups,
        packed_query=packed_query,
        sparse_block_size=sparse_block_size,
        split_kv=split_kv,
        share_pattern_across_kv_heads=share_pattern_across_kv_heads,
        count_current_block_in_topk=count_current_block_in_topk,
    )


def prepare_metadata_plan(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: Optional[torch.Tensor],
    query_positions: torch.Tensor,
    q_token_kv_block_sparse_page_indices: torch.Tensor,
    q_token_kv_block_sparse_page_memberships: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    group_size: int,
    storage_page_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int = 4,
    qo_indptr: Optional[torch.Tensor] = None,
    query_major_memberships: bool = False,
    locator_head_multiplier: int = 1,
    count_current_block_in_topk: bool = True,
    fixed_query_len: int = 0,
) -> MetadataPlan:
    """Freeze validated metadata tensors and launch constants.

    Query-major memberships use a private logical [row, KV tile, query slot]
    layout: each uint32 masks 32 four-token union fragments for one query.
    The rank-two tensor flattens eight slots per tile, with unused slots and
    tail bits zero. This encoding must match the attention plan. Standalone
    metadata retains per-fragment membership bytes; Q/K/V layouts, the union,
    and page indices are unchanged.
    """

    sparse_block_size = validate_sparse_block_size(sparse_block_size)
    rows = block_indices.shape[0]
    packed_query = qo_indptr is not None
    resolve_num_query_groups(
        rows,
        group_size,
        int(qo_indptr.numel()) - 1 if packed_query else None,
    )
    max_seq_len_kv = validate_max_seq_len_kv(
        max_seq_len_kv,
        block_table_token_capacity=block_table.shape[1] * storage_page_size,
    )
    release_attention_pdl = device_support_pdl(block_indices.device)
    metadata_module = get_metadata_module()
    if query_major_memberships:
        metadata_run = (
            metadata_module.run_packed_query_major
            if packed_query
            else metadata_module.run_fixed_query_major
        )
    else:
        metadata_run = (
            metadata_module.run_packed if packed_query else metadata_module.run_fixed
        )
    return MetadataPlan(
        q_token_kv_block_sparse_page_indices=q_token_kv_block_sparse_page_indices,
        q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
        seq_lens=seq_lens,
        qo_indptr=qo_indptr,
        metadata_run=metadata_run,
        group_size=group_size,
        packed_query=packed_query,
        sparse_block_size=sparse_block_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        release_attention_pdl=release_attention_pdl,
        locator_head_multiplier=locator_head_multiplier,
        count_current_block_in_topk=count_current_block_in_topk,
        fixed_query_len=fixed_query_len,
    )


def resolve_num_query_groups(
    num_query_tokens: int,
    group_size: int,
    num_query_groups: Optional[int],
) -> int:
    """Resolve fixed or request-partitioned route count."""

    if num_query_groups is None:
        if num_query_tokens % group_size:
            raise ValueError(
                "num_query_tokens must be divisible by group_size without "
                "packed QToken-KvBlock-Sparse-Attention qo_indptr"
            )
        return num_query_tokens // group_size
    if not isinstance(num_query_groups, int) or isinstance(num_query_groups, bool):
        raise TypeError("num_query_groups must be an integer")
    if num_query_groups <= 0:
        raise ValueError("num_query_groups must be positive")
    min_groups = (num_query_tokens + group_size - 1) // group_size
    if num_query_groups < min_groups or num_query_groups > num_query_tokens:
        raise ValueError(
            "num_query_groups cannot partition num_query_tokens into nonempty "
            f"routes of size at most {group_size}"
        )
    return num_query_groups


def tensor_abi_key(
    tensor: Optional[torch.Tensor],
    *,
    include_storage: bool = False,
) -> object:
    """Return the structural ABI, optionally including retained storage."""

    if tensor is None:
        return None
    key: tuple[object, ...] = (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.device,
        tensor.dtype,
    )
    return key + (tensor.data_ptr(),) if include_storage else key


def validate_attention_workspace(
    workspace_buffer: torch.Tensor,
    required_bytes: int,
) -> None:
    if not isinstance(workspace_buffer, torch.Tensor):
        raise TypeError("workspace_buffer must be a torch.Tensor")
    if workspace_buffer.dtype not in (torch.int8, torch.uint8):
        raise TypeError("workspace_buffer must have dtype torch.int8 or torch.uint8")
    if not workspace_buffer.is_cuda:
        raise ValueError("workspace_buffer must be a CUDA tensor")
    if not workspace_buffer.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous")
    available_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    if available_bytes < required_bytes:
        raise ValueError(
            "workspace_buffer is too small: requires at least "
            f"{required_bytes} bytes, got {available_bytes}"
        )
    if workspace_buffer.data_ptr() % 32:
        raise ValueError("workspace_buffer data pointer must be 32-byte aligned")


def validate_locator_capacity(
    k_cache: torch.Tensor,
    *,
    sparse_block_size: int,
) -> None:
    """Keep every valid encoded cache locator nonnegative in Int32."""

    storage_page_size = int(k_cache.shape[2])
    validate_metadata_storage_page_size(storage_page_size)
    fragment_size = math.gcd(storage_page_size, sparse_block_size)
    num_encoded_locators = int(k_cache.shape[0]) * storage_page_size // fragment_size
    if num_encoded_locators > INT32_LOCATOR_CAPACITY:
        raise NotImplementedError(
            "the encoded QToken-KvBlock-Sparse-Attention cache-locator extent must fit in nonnegative "
            "signed int32: got "
            f"{num_encoded_locators}, limit is {INT32_LOCATOR_CAPACITY}"
        )


def validate_max_seq_len_kv(
    max_seq_len_kv: int,
    *,
    block_table_token_capacity: Optional[int] = None,
) -> int:
    """Validate the static logical K/V bound used by metadata kernels."""

    if not isinstance(max_seq_len_kv, int) or isinstance(max_seq_len_kv, bool):
        raise TypeError("max_seq_len_kv must be an integer")
    if max_seq_len_kv <= 0:
        raise ValueError("max_seq_len_kv must be positive")
    if max_seq_len_kv > MAX_SEQ_LEN_KV:
        raise ValueError(
            "max_seq_len_kv exceeds the current grouped QToken-KvBlock-Sparse-Attention radix-key bound "
            f"({MAX_SEQ_LEN_KV})"
        )
    if (
        block_table_token_capacity is not None
        and max_seq_len_kv > block_table_token_capacity
    ):
        raise ValueError(
            "max_seq_len_kv must not exceed the dense block-table token "
            f"capacity ({block_table_token_capacity})"
        )
    return max_seq_len_kv


def validate_metadata_inputs(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: Optional[torch.Tensor],
    query_positions: torch.Tensor,
    group_size: int,
    storage_page_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int,
    share_pattern_across_kv_heads: bool = True,
    count_current_block_in_topk: bool = True,
    fixed_query_len: int = 0,
) -> None:
    """Validate tensor structure without copying live metadata values to the host."""
    sparse_block_size = validate_sparse_block_size(sparse_block_size)
    for name, tensor in (
        ("block_indices", block_indices),
        ("block_table", block_table),
        ("token_to_request", token_to_request),
        ("query_positions", query_positions),
    ):
        if name == "token_to_request" and fixed_query_len and tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if not block_indices.is_cuda:
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention metadata inputs must be CUDA tensors"
        )
    if not isinstance(count_current_block_in_topk, bool):
        raise TypeError("count_current_block_in_topk must be a bool")
    if not isinstance(share_pattern_across_kv_heads, bool):
        raise TypeError("share_pattern_across_kv_heads must be a bool")
    rank = 2 if share_pattern_across_kv_heads else 3
    if block_indices.ndim != rank or block_indices.dtype != torch.int32:
        shape = "[T,K]" if share_pattern_across_kv_heads else "[T,Hkv,K]"
        raise ValueError(f"block_indices must be a rank-{rank} int32 tensor {shape}")
    if rank == 3 and block_indices.shape[1] <= 0:
        raise ValueError("block_indices must have a positive pattern head count")
    rows, block_topk = block_indices.shape[0], block_indices.shape[-1]
    validate_shape_parameters(rows, block_topk, group_size)
    if block_topk > MAX_BLOCK_TOPK:
        raise NotImplementedError(
            f"PrimTS QToken-KvBlock-Sparse-Attention currently supports block_topk <= {MAX_BLOCK_TOPK}"
        )
    validate_metadata_storage_page_size(storage_page_size)
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("block_table must be a nonempty rank-two int32 tensor")
    if not all(block_table.shape):
        raise ValueError("block_table must be nonempty")
    validate_max_seq_len_kv(
        max_seq_len_kv,
        block_table_token_capacity=block_table.shape[1] * storage_page_size,
    )
    if fixed_query_len:
        if (
            token_to_request is not None
            or fixed_query_len < 0
            or fixed_query_len > 2**31 - 1
            or fixed_query_len % group_size
            or rows != block_table.shape[0] * fixed_query_len
        ):
            raise ValueError("fixed Q must align complete query groups with requests")
        metadata_rows = block_table.shape[0]
    else:
        assert token_to_request is not None
        if token_to_request.shape != (rows,) or token_to_request.dtype != torch.int32:
            raise ValueError("token_to_request must be int32 with one value per row")
        metadata_rows = rows
    if query_positions.shape != (metadata_rows,) or query_positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(
            "positions/KV lengths must be int32/int64 with the expected extent"
        )
    tensors: tuple[torch.Tensor, ...] = (block_table, query_positions)
    if token_to_request is not None:
        tensors += (token_to_request,)
    if any(tensor.device != block_indices.device for tensor in tensors):
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention metadata inputs must share one CUDA device"
        )
    if block_indices.stride(-1) != 1 or block_table.stride(1) != 1:
        raise ValueError("block_indices and block_table rows must be contiguous")
    if block_table.stride(0) < block_table.shape[1]:
        raise ValueError(
            "block_table must be a non-overlapping dense row-strided page table"
        )
    if (
        token_to_request is not None and token_to_request.stride(0) != 1
    ) or query_positions.stride(0) != 1:
        raise ValueError(
            "per-row QToken-KvBlock-Sparse-Attention metadata must be contiguous"
        )


def validate_metadata_storage_page_size(storage_page_size: int) -> None:
    if not isinstance(storage_page_size, int) or isinstance(storage_page_size, bool):
        raise TypeError("storage_page_size must be an integer")
    if storage_page_size < 4 or storage_page_size % 4:
        raise ValueError("storage_page_size must be a positive multiple of four")


def validate_paged_kv_cache(
    paged_kv_cache: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and return the two rank-four QToken-KvBlock-Sparse-Attention cache tensors."""

    if (
        not isinstance(paged_kv_cache, tuple)
        or len(paged_kv_cache) != 2
        or not all(isinstance(cache, torch.Tensor) for cache in paged_kv_cache)
    ):
        raise TypeError("paged_kv_cache must be a (k_cache, v_cache) tuple")
    k_cache, v_cache = paged_kv_cache
    if k_cache.ndim != 4:
        raise ValueError(
            "K and V cache tensors must have shape [pages,Hkv,storage_page_size,D]"
        )
    if (
        v_cache.shape != k_cache.shape
        or v_cache.device != k_cache.device
        or v_cache.dtype != k_cache.dtype
    ):
        raise ValueError(
            "K and V cache tensors must have matching shapes, devices, and dtypes"
        )
    return k_cache, v_cache


def validate_plan_tensor(
    tensor: torch.Tensor,
    name: str,
    descriptor: TensorDescriptor,
) -> None:
    """Check one replacement tensor without inspecting device values."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        tuple(tensor.shape) != descriptor.shape
        or tuple(tensor.stride()) != descriptor.stride
        or tensor.device != descriptor.device
        or tensor.dtype != descriptor.dtype
    ):
        raise ValueError(
            f"{name} must preserve the shape, strides, device, and dtype "
            "validated by the QToken-KvBlock-Sparse-Attention plan"
        )


def validate_qo_indptr_layout_tensor(
    qo_indptr: torch.Tensor,
) -> None:
    """Validate the packed-Q offsets required to size QToken-KvBlock-Sparse-Attention workspace."""

    if (
        not isinstance(qo_indptr, torch.Tensor)
        or qo_indptr.ndim != 1
        or qo_indptr.numel() < 2
        or qo_indptr.dtype != torch.int32
    ):
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention qo_indptr must be a rank-one int32 tensor"
        )


def validate_qo_indptr_tensor(
    qo_indptr: torch.Tensor,
    *,
    expected_device: torch.device,
) -> None:
    """Validate the synchronization-free packed-Q offset tensor contract."""

    validate_qo_indptr_layout_tensor(qo_indptr)
    if qo_indptr.device != expected_device or not qo_indptr.is_contiguous():
        raise ValueError(
            "QToken-KvBlock-Sparse-Attention qo_indptr must be a contiguous CUDA int32 tensor on the query device"
        )


def validate_shape_parameters(
    num_query_tokens: int,
    block_topk: int,
    group_size: int,
) -> None:
    for value, name in (
        (num_query_tokens, "num_query_tokens"),
        (block_topk, "block_topk"),
        (group_size, "group_size"),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if num_query_tokens < 0:
        raise ValueError("num_query_tokens must be nonnegative")
    if block_topk <= 0:
        raise ValueError("block_topk must be positive")
    from .grouping import validate_group_value

    validate_group_value(group_size)


def validate_sparse_block_size(sparse_block_size: int) -> int:
    """Validate the generic sparse-block API and current specialization."""

    if not isinstance(sparse_block_size, int) or isinstance(sparse_block_size, bool):
        raise TypeError("sparse_block_size must be an integer")
    if sparse_block_size <= 0 or sparse_block_size & (sparse_block_size - 1):
        raise ValueError("sparse_block_size must be a positive power of two")
    if sparse_block_size not in BLOCK_SIZES:
        raise NotImplementedError("sparse_block_size must be 4, 8, 16, 32, 64, or 128")
    return sparse_block_size


def validate_wide_metadata_fits(
    group_size: int,
    max_seq_len_kv: int,
    sparse_block_size: int,
    device_index: int,
) -> None:
    """Reject G16/G32 when their metadata map cannot fit the device's SMEM.

    G9-G32 metadata keeps a map over the model's whole block prefix
    (``max_seq_len_kv / sparse_block_size`` blocks) in shared memory; there is
    no bounded fallback, so the check runs before any launch.
    """
    if group_size <= 8:
        return
    needed = wide_metadata_smem_bytes(group_size, max_seq_len_kv, sparse_block_size)
    available = torch.cuda.get_device_properties(
        device_index
    ).shared_memory_per_block_optin
    if needed < 0 or needed > available:
        raise ValueError(
            f"QToken-KvBlock-Sparse-Attention group_size={group_size} keeps a "
            f"metadata map of max_seq_len_kv={max_seq_len_kv} / "
            f"kv_block_size={sparse_block_size} blocks in shared memory, which "
            + (
                "exceeds the map's size limit"
                if needed < 0
                else f"needs {needed} bytes"
            )
            + f" (device limit {available}); use group_size <= 8, a larger "
            "kv_block_size or a smaller max_seq_len_kv"
        )


@functools.cache
def wide_metadata_smem_bytes(
    group_size: int, max_seq_len_kv: int, sparse_block_size: int
) -> int:
    module = get_metadata_module()
    return int(
        module.wide_metadata_smem_bytes(group_size, max_seq_len_kv, sparse_block_size)
    )
