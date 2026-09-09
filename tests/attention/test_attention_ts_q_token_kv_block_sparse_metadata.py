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

from __future__ import annotations

import inspect
import math

import pytest
import torch

from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
    QTokenKvBlockSparsePagedTSWrapper,
    _build_q_token_kv_block_sparse_metadata as build_prims_ts_q_token_kv_block_sparse_metadata,
    _get_q_token_kv_block_sparse_metadata_output_shapes as get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes,
    _get_prims_ts_q_token_kv_block_sparse_workspace_layout,
    _prepare_q_token_kv_block_sparse_attention as prepare_prims_ts_q_token_kv_block_sparse_attention,
    _validate_q_token_kv_block_sparse_workspace_aliasing,
    get_q_token_kv_block_sparse_workspace_size,
    q_token_kv_block_sparse_attention_with_paged_kv_cache,
)
from flashinfer.decode import (
    make_q_token_kv_block_sparse_qo_indptr as make_prims_ts_q_token_kv_block_sparse_qo_indptr,
    suggest_q_token_kv_block_sparse_group_size as suggest_prims_ts_q_token_kv_block_sparse_group_size,
)


def get_prims_ts_q_token_kv_block_sparse_workspace_size(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_topk: int,
    max_seq_len_kv: int,
    out_dtype: torch.dtype | None = None,
    qo_indptr: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
    sparse_block_size: int = 4,
) -> int:
    """Adapt internal accuracy cases to the renamed public workspace API."""

    return get_q_token_kv_block_sparse_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=block_topk,
        max_seq_len_kv=max_seq_len_kv,
        o_data_type=out_dtype,
        qo_indptr=qo_indptr,
        seq_len_q=max_seq_len_q,
        kv_block_size=sparse_block_size,
    )


def prims_ts_q_token_kv_block_sparse_attention(
    query: torch.Tensor,
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    max_seq_len_kv: int,
    bmm1_scale: float | None = None,
    bmm2_scale: float = 1.0,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    qo_indptr: torch.Tensor | None = None,
    max_seq_len_q: int | None = None,
    sparse_block_size: int = 4,
) -> torch.Tensor:
    """Adapt internal accuracy cases to the renamed public eager API."""

    return q_token_kv_block_sparse_attention_with_paged_kv_cache(
        query,
        paged_kv_cache,
        block_table,
        block_indices,
        token_to_request,
        query_positions,
        workspace_buffer,
        max_seq_len_kv=max_seq_len_kv,
        seq_len_q=max_seq_len_q,
        kv_block_size=sparse_block_size,
        sm_scale=bmm1_scale,
        v_scale=bmm2_scale,
        out=out,
        o_data_type=out_dtype,
        qo_indptr=qo_indptr,
    )


def test_q_token_kv_block_sparse_apis_are_available_from_flashinfer_decode() -> None:
    import flashinfer.decode as public_decode

    public_names = (
        "QTokenKvBlockSparsePagedTSWrapper",
        "get_q_token_kv_block_sparse_workspace_size",
        "make_q_token_kv_block_sparse_qo_indptr",
        "q_token_kv_block_sparse_attention_with_paged_kv_cache",
        "suggest_q_token_kv_block_sparse_group_size",
        "validate_q_token_kv_block_sparse_group_size",
    )

    for name in public_names:
        assert getattr(public_decode, name) is not None

    # Keep the retired spellings here to catch accidental compatibility aliases.
    for legacy_name in (
        "PrimsTSQSAPlan",
        "get_prims_ts_qsa_workspace_size",
        "prepare_prims_ts_qsa_attention",
        "prims_ts_qsa_attention",
    ):
        assert not hasattr(public_decode, legacy_name)


def test_q_token_kv_block_sparse_wrapper_matches_plan_run_grammar() -> None:
    plan_parameters = inspect.signature(
        QTokenKvBlockSparsePagedTSWrapper.plan
    ).parameters
    assert tuple(plan_parameters) == (
        "self",
        "batch_size",
        "seq_len_q",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "kv_block_size",
        "page_size",
        "block_topk",
        "max_seq_len_kv",
        "device",
        "workspace_buffer",
        "use_packed_q",
        "mask_type",
        "q_data_type",
        "kv_data_type",
        "o_data_type",
    )
    run_parameters = inspect.signature(QTokenKvBlockSparsePagedTSWrapper.run).parameters
    assert tuple(run_parameters) == (
        "self",
        "q",
        "paged_kv_cache",
        "block_table",
        "indexer_block_ids",
        "token_to_request",
        "query_positions",
        "qo_indptr",
        "sm_scale",
        "v_scale",
        "out",
    )


@pytest.mark.parametrize(
    "api",
    (
        get_q_token_kv_block_sparse_workspace_size,
        q_token_kv_block_sparse_attention_with_paged_kv_cache,
    ),
)
def test_q_token_kv_block_sparse_public_apis_expose_kv_block_size(api: object) -> None:
    parameter = inspect.signature(api).parameters["kv_block_size"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 4


@pytest.mark.parametrize(
    "api",
    (
        get_q_token_kv_block_sparse_workspace_size,
        q_token_kv_block_sparse_attention_with_paged_kv_cache,
    ),
)
def test_q_token_kv_block_sparse_apis_require_static_max_seq_len_kv(
    api: object,
) -> None:
    parameter = inspect.signature(api).parameters["max_seq_len_kv"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("request_offsets", "num_query_tokens", "group_size", "expected"),
    (
        ((0, 8), 8, 4, (0, 4, 8)),
        ((0, 1, 8), 8, 4, (0, 1, 5, 8)),
        (tuple(range(9)), 8, 4, tuple(range(9))),
        # CUDA-graph padding is a separate route and cannot fill the final
        # partial route of the last real request.
        ((0, 5), 8, 4, (0, 4, 5, 8)),
        # Query grouping is independent of sparse_block_size=4: Q5/MTP4
        # partitions eighty fixed-width rows into sixteen routes.
        ((0, 80), 80, 5, tuple(range(0, 81, 5))),
    ),
)
def test_q_token_kv_block_sparse_packed_route_count_respects_requests_and_query_group_size(
    request_offsets: tuple[int, ...],
    num_query_tokens: int,
    group_size: int,
    expected: tuple[int, ...],
) -> None:
    actual = make_prims_ts_q_token_kv_block_sparse_qo_indptr(
        torch.tensor(request_offsets, dtype=torch.int32),
        num_query_tokens,
        group_size=group_size,
        device="cpu",
    )
    assert tuple(actual.tolist()) == expected


def test_q_token_kv_block_sparse_suggested_group_supports_partial_packed_routes() -> (
    None
):
    group_size = suggest_prims_ts_q_token_kv_block_sparse_group_size(
        batch_size=2,
        seq_len_q=5,
        selected_seq_len_kv=2051,
        num_qo_heads=12,
        num_kv_heads=1,
        multi_processor_count=1,
    )
    assert group_size == 5
    actual = make_prims_ts_q_token_kv_block_sparse_qo_indptr(
        torch.tensor((0, 5, 8), dtype=torch.int32),
        8,
        group_size=group_size,
        device="cpu",
    )
    assert tuple(actual.tolist()) == (0, 5, 8)


def test_q_token_kv_block_sparse_block_size_default_matches_explicit_four() -> None:
    assert get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(20, 512, 4) == (
        get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            20,
            512,
            4,
            sparse_block_size=4,
        )
    )
    assert get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(20, 512, 1) == (
        (20, 513),
        (20, 0),
        (20,),
    )
    assert get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(20, 512, 4) == (
        (5, 4 * 513),
        (5, 513),
        (5,),
    )


@pytest.mark.parametrize("sparse_block_size", (0, -4, 3, 6))
def test_q_token_kv_block_sparse_block_size_must_be_positive_power_of_two(
    sparse_block_size: int,
) -> None:
    with pytest.raises(ValueError, match="positive power of two"):
        get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            4,
            8,
            4,
            sparse_block_size=sparse_block_size,
        )


@pytest.mark.parametrize("sparse_block_size", (None, True, 4.0, "4"))
def test_q_token_kv_block_sparse_block_size_must_be_integer(
    sparse_block_size: object,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            4,
            8,
            4,
            sparse_block_size=sparse_block_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("sparse_block_size", (1, 2, 8, 16))
def test_q_token_kv_block_sparse_other_power_of_two_sparse_block_sizes_are_not_implemented(
    sparse_block_size: int,
) -> None:
    with pytest.raises(NotImplementedError, match="only sparse_block_size=4"):
        get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            4,
            8,
            4,
            sparse_block_size=sparse_block_size,
        )


@pytest.mark.parametrize(
    "api",
    (
        prepare_prims_ts_q_token_kv_block_sparse_attention,
        prims_ts_q_token_kv_block_sparse_attention,
    ),
)
def test_q_token_kv_block_sparse_attention_apis_reject_malformed_cache_rank(
    api: object,
) -> None:
    """Cache validation must report the public shape contract, not IndexError."""

    query = torch.empty((1, 1, 1, 12, 256), dtype=torch.bfloat16)
    malformed_cache = torch.empty((1, 256), dtype=torch.bfloat16)
    block_indices = torch.empty((1, 1), dtype=torch.int32)
    block_table = torch.empty((1, 1), dtype=torch.int32)
    token_to_request = torch.empty((1,), dtype=torch.int32)
    query_positions = torch.empty((1,), dtype=torch.int64)
    workspace = torch.empty((1,), dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"K and V cache tensors must have shape"):
        api(  # type: ignore[operator]
            query,
            (malformed_cache, malformed_cache),
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            workspace,
            out=torch.empty_like(query),
            max_seq_len_kv=1,
        )


@pytest.mark.parametrize(
    "overlap_name",
    (
        "query",
        "k_cache",
        "v_cache",
        "block_indices",
        "block_table",
        "token_to_request",
        "query_positions",
        "qo_indptr",
        "out",
    ),
)
def test_q_token_kv_block_sparse_unified_workspace_alias_guard_covers_all_live_tensors(
    overlap_name: str,
) -> None:
    """Unified metadata/attention storage must not overwrite semantic inputs."""

    storage = torch.empty(96, dtype=torch.uint8)
    workspace = storage[:64]
    overlapping = storage[32:]
    tensors = {
        name: torch.empty(1, dtype=torch.uint8)
        for name in (
            "query",
            "k_cache",
            "v_cache",
            "block_indices",
            "block_table",
            "token_to_request",
            "query_positions",
            "qo_indptr",
            "out",
        )
    }
    tensors[overlap_name] = overlapping

    with pytest.raises(
        ValueError,
        match=rf"workspace_buffer must not overlap {overlap_name} storage",
    ):
        _validate_q_token_kv_block_sparse_workspace_aliasing(
            workspace,
            query=tensors["query"],
            k_cache=tensors["k_cache"],
            v_cache=tensors["v_cache"],
            block_indices=tensors["block_indices"],
            block_table=tensors["block_table"],
            token_to_request=tensors["token_to_request"],
            query_positions=tensors["query_positions"],
            qo_indptr=tensors["qo_indptr"],
            out=tensors["out"],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q_token_kv_block_sparse_workspace_size_rejects_signed_locator_overflow() -> (
    None
):
    """Plain page locators must remain nonnegative signed Int32 values."""

    sparse_block_size = 4
    storage_page_size = 16
    max_encoded_locators = 1 << 31
    oversized_num_pages = (
        max_encoded_locators // (storage_page_size // sparse_block_size) + 1
    )
    cache_storage = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    oversized_cache = torch.as_strided(
        cache_storage,
        (oversized_num_pages, 1, storage_page_size, 256),
        (0, 0, 0, 0),
    )
    query = torch.empty((1, 1, 4, 12, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((1, 1), dtype=torch.int32, device="cuda")

    with pytest.raises(
        NotImplementedError,
        match=r"signed int32.*limit is 2147483648",
    ):
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            oversized_cache,
            block_table,
            block_topk=1,
            max_seq_len_kv=block_table.shape[1] * storage_page_size,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unified_q_token_kv_block_sparse_workspace_8k_q_128k_kv_owns_metadata_outputs() -> (
    None
):
    layout = _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
        8192,
        512,
        2048,
        64,
        4,
        max_seq_len_kv=8192,
        num_qo_heads=24,
        num_kv_heads=2,
        head_dim=256,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device="cuda",
        num_query_groups=2048,
        use_packed_q=True,
    )

    assert layout.q_token_kv_block_sparse_page_indices_shape == (2048, 4 * 513)
    assert layout.q_token_kv_block_sparse_page_indices_bytes == 16_809_984
    assert layout.q_token_kv_block_sparse_page_memberships_shape == (2048, 513)
    assert layout.q_token_kv_block_sparse_page_memberships_bytes == 4_202_496
    assert not layout.uses_split_kv
    assert layout.total_bytes == (
        layout.attention_workspace_byte_offset + layout.attention_scratch_bytes
    )

    workspace = torch.empty(layout.total_bytes, dtype=torch.uint8, device="cuda")
    views = layout.bind(workspace)
    assert views.q_token_kv_block_sparse_page_indices.shape == (2048, 4 * 513)
    assert views.q_token_kv_block_sparse_page_indices.data_ptr() == workspace.data_ptr()
    assert views.q_token_kv_block_sparse_page_indices.numel() == 8192 * 513
    assert views.q_token_kv_block_sparse_page_memberships.shape == (2048, 513)
    assert views.q_token_kv_block_sparse_page_memberships.data_ptr() == (
        workspace.data_ptr()
        + layout.q_token_kv_block_sparse_page_memberships_byte_offset
    )
    assert views.q_token_kv_block_sparse_page_memberships.numel() == 2048 * 513
    assert views.seq_lens.numel() == 2048
    assert views.seq_lens.data_ptr() == (
        workspace.data_ptr() + layout.seq_lens_byte_offset
    )
    assert views.attention_workspace_buffer.data_ptr() == (
        workspace.data_ptr() + layout.attention_workspace_byte_offset
    )
    assert views.attention_workspace_buffer.numel() == layout.attention_scratch_bytes
    assert (
        views.q_token_kv_block_sparse_page_indices.data_ptr()
        + views.q_token_kv_block_sparse_page_indices.numel()
        * views.q_token_kv_block_sparse_page_indices.element_size()
        <= views.q_token_kv_block_sparse_page_memberships.data_ptr()
    )
    assert (
        views.q_token_kv_block_sparse_page_memberships.data_ptr()
        + views.q_token_kv_block_sparse_page_memberships.numel()
        * views.q_token_kv_block_sparse_page_memberships.element_size()
        <= views.seq_lens.data_ptr()
    )
    assert (
        views.seq_lens.data_ptr()
        + views.seq_lens.numel() * views.seq_lens.element_size()
        <= views.attention_workspace_buffer.data_ptr()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_q_token_kv_block_sparse_workspace_size_accepts_cpu_route_offsets() -> (
    None
):
    query = torch.empty((8, 12, 256), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((2, 64), dtype=torch.int32, device="cuda")
    cpu_qo_indptr = torch.tensor((0, 4, 5, 8), dtype=torch.int32)

    workspace_bytes = get_prims_ts_q_token_kv_block_sparse_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=8,
        max_seq_len_kv=block_table.shape[1] * k_cache.shape[2],
        qo_indptr=cpu_qo_indptr,
        max_seq_len_q=4,
    )

    assert workspace_bytes > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_q_token_kv_block_sparse_workspace_size_rejects_int64_route_offsets() -> (
    None
):
    query = torch.empty((8, 12, 256), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((2, 64), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="int32"):
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=8,
            max_seq_len_kv=block_table.shape[1] * k_cache.shape[2],
            qo_indptr=torch.tensor((0, 4, 5, 8), dtype=torch.int64),
            max_seq_len_q=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("query_shape", ((2, 12, 256), (2, 4, 12, 256)))
def test_q_token_kv_block_sparse_workspace_size_rejects_legacy_fixed_layouts(
    query_shape: tuple[int, ...],
) -> None:
    query = torch.empty(query_shape, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((2, 64), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="fixed.*B,Nq"):
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=8,
            max_seq_len_kv=block_table.shape[1] * k_cache.shape[2],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_q_token_kv_block_sparse_workspace_size_rejects_packed_route_bound() -> (
    None
):
    query = torch.empty((2, 1, 4, 12, 256), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((2, 64), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="only valid with packed"):
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=8,
            max_seq_len_kv=block_table.shape[1] * k_cache.shape[2],
            max_seq_len_q=4,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_packed_q", (False, True))
def test_q_token_kv_block_sparse_workspace_size_enforces_q64_group_capacity(
    use_packed_q: bool,
) -> None:
    group_size = 5
    num_qo_heads = 16
    if use_packed_q:
        query = torch.empty(
            (group_size, num_qo_heads, 256),
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_kwargs = {
            "qo_indptr": torch.tensor((0, group_size), dtype=torch.int32),
            "max_seq_len_q": group_size,
        }
    else:
        query = torch.empty(
            (1, 1, group_size, num_qo_heads, 256),
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_kwargs = {}
    k_cache = torch.empty((64, 1, 16, 256), dtype=torch.bfloat16, device="cuda")
    block_table = torch.empty((1, 64), dtype=torch.int32, device="cuda")

    with pytest.raises(ValueError, match="TileQ64/head capacity"):
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=8,
            max_seq_len_kv=block_table.shape[1] * k_cache.shape[2],
            **q_kwargs,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_q_token_kv_block_sparse_metadata_rejects_int64_route_offsets() -> None:
    blocks, table, requests, positions, storage_page_size = _make_case(4, 8)

    with pytest.raises(ValueError, match="int32"):
        build_prims_ts_q_token_kv_block_sparse_metadata(
            blocks,
            table,
            requests,
            positions,
            group_size=4,
            storage_page_size=storage_page_size,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            qo_indptr=torch.tensor((0, 4, 8), dtype=torch.int64, device="cuda"),
        )


def _reference(
    block_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    storage_page_size: int,
    group_size: int,
    qo_indptr: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = block_indices.device
    blocks = block_indices.cpu()
    table = block_table.cpu()
    requests = token_to_request.cpu()
    positions = query_positions.cpu()
    rows, block_topk = blocks.shape
    if qo_indptr is None:
        assert rows % group_size == 0
        route_offsets = list(range(0, rows + 1, group_size))
    else:
        route_offsets = [int(value) for value in qo_indptr.cpu().tolist()]
    groups = len(route_offsets) - 1
    page_capacity = group_size * (block_topk + 1)
    q_token_kv_block_sparse_page_indices = torch.full(
        (groups, page_capacity), -1, dtype=torch.int32
    )
    membership_words = 0 if group_size == 1 else (page_capacity + 3) // 4
    q_token_kv_block_sparse_page_memberships = torch.zeros(
        (groups, membership_words), dtype=torch.int32
    )
    seq_lens = torch.ones(groups, dtype=torch.int32)
    subpages_per_storage_page = storage_page_size // 4

    for group in range(groups):
        first_row = route_offsets[group]
        group_end = route_offsets[group + 1]
        group_query_len = group_end - first_row
        request = int(requests[first_row].item())
        first_position = int(positions[first_row].item())
        valid = 0 <= request < table.shape[0]
        for q_index in range(group_query_len):
            row = first_row + q_index
            valid &= int(requests[row].item()) == request
            valid &= int(positions[row].item()) == first_position + q_index
        if not valid:
            continue

        if group_size == 1:
            visible_tokens = first_position + 1
            complete_pages = min(visible_tokens // 4, block_topk)
            tail_tokens = visible_tokens % 4
            live_blocks = [int(value) for value in blocks[first_row, :complete_pages]]
            if tail_tokens:
                live_blocks.append(visible_tokens // 4)
            seq_lens[group] = complete_pages * 4 + tail_tokens
            for rank, logical_block in enumerate(live_blocks):
                logical_token = logical_block * 4
                storage_page, token_offset = divmod(logical_token, storage_page_size)
                physical_page = int(table[request, storage_page].item())
                q_token_kv_block_sparse_page_indices[group, rank] = (
                    physical_page * subpages_per_storage_page + token_offset // 4
                )
            continue

        membership_by_block: dict[int, int] = {}
        for q_index in range(group_query_len):
            row = first_row + q_index
            visible_tokens = int(positions[row].item()) + 1
            complete_pages = min(visible_tokens // 4, block_topk)
            for logical_block_tensor in blocks[row, :complete_pages]:
                logical_block = int(logical_block_tensor.item())
                if not 0 <= logical_block < visible_tokens // 4:
                    continue
                membership_by_block[logical_block] = membership_by_block.get(
                    logical_block, 0
                ) | (1 << q_index)
            if visible_tokens % 4:
                tail_block = visible_tokens // 4
                membership_by_block[tail_block] = membership_by_block.get(
                    tail_block, 0
                ) | (1 << q_index)

        for rank, (logical_block, membership) in enumerate(
            sorted(membership_by_block.items())
        ):
            logical_token = logical_block * 4
            storage_page, token_offset = divmod(logical_token, storage_page_size)
            physical_page = int(table[request, storage_page].item())
            locator = physical_page * subpages_per_storage_page + token_offset // 4
            q_token_kv_block_sparse_page_indices[group, rank] = locator
            membership_word = rank // 4
            membership_shift = (rank % 4) * 8
            q_token_kv_block_sparse_page_memberships[group, membership_word] |= (
                membership << membership_shift
            )
        last_tail = (int(positions[group_end - 1].item()) + 1) % 4
        tail_padding = 0 if last_tail == 0 else 4 - last_tail
        seq_lens[group] = max(
            len(membership_by_block) * 4 - tail_padding,
            1,
        )

    return (
        q_token_kv_block_sparse_page_indices.to(device),
        q_token_kv_block_sparse_page_memberships.to(device),
        seq_lens.to(device),
    )


def _assert_metadata_matches(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Compare the observable live route prefix of compact QToken-KvBlock-Sparse-Attention metadata."""

    actual_indices, actual_memberships, actual_seq_lens = actual
    expected_indices, expected_memberships, expected_seq_lens = expected
    assert actual_indices.shape == expected_indices.shape
    assert actual_memberships.shape == expected_memberships.shape
    torch.testing.assert_close(actual_seq_lens, expected_seq_lens)
    for group in range(expected_seq_lens.numel()):
        live_pages = (int(expected_seq_lens[group].item()) + 3) // 4
        torch.testing.assert_close(
            actual_indices[group, :live_pages],
            expected_indices[group, :live_pages],
        )
        if expected_memberships.shape[1] > 0:
            # Attention loads whole words, including the last word's padding.
            live_words = (live_pages + 3) // 4
            torch.testing.assert_close(
                actual_memberships[group, :live_words],
                expected_memberships[group, :live_words],
            )


def _make_case(group_size: int, block_topk: int, storage_page_size: int = 16):
    groups = 2
    rows = groups * group_size
    base_position = 4 * block_topk - 1
    positions = torch.tensor(
        [
            base_position + q_index
            for _group in range(groups)
            for q_index in range(group_size)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    requests = torch.arange(groups, dtype=torch.int32, device="cuda").repeat_interleave(
        group_size
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [torch.roll(base_blocks, row % 7) for row in range(rows)]
    ).contiguous()
    table_width = max(64, (4 * block_topk + 64) // storage_page_size)
    block_table = torch.arange(
        groups * table_width, dtype=torch.int32, device="cuda"
    ).reshape(groups, table_width)
    return blocks, block_table, requests, positions, storage_page_size


def _make_sort_union_case(group_size: int, packed: bool):
    """Build one route usable with the qualified 64/128-word pack buckets."""

    storage_page_size = 1600
    full_rows = group_size * 2
    rows = full_rows - 1 if packed else full_rows
    requests = torch.tensor(
        [0] * group_size + [1] * (rows - group_size),
        dtype=torch.int32,
        device="cuda",
    )
    positions = torch.tensor(
        [4107 + index for index in range(group_size)]
        + [4107 + index for index in range(rows - group_size)],
        dtype=torch.int64,
        device="cuda",
    )
    base_blocks = torch.arange(512, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [(base_blocks + 97 * (row % group_size)) % 1000 for row in range(rows)]
    ).contiguous()
    block_table = (
        torch.arange(12, dtype=torch.int32, device="cuda").reshape(2, 6) * 3 + 7
    )
    qo_indptr = None
    if packed:
        qo_indptr = torch.tensor(
            [0, group_size, rows],
            dtype=torch.int32,
            device="cuda",
        )
    return (
        blocks,
        block_table,
        requests,
        positions,
        storage_page_size,
        qo_indptr,
    )


def _make_wide_sort_union_case(
    group_size: int,
    packed: bool,
    max_seq_len_kv: int,
):
    """Build sparse routes spanning one or more causal pack tiles."""

    storage_page_size = 1600
    full_rows = group_size * 2
    rows = full_rows - 1 if packed else full_rows
    requests = torch.tensor(
        [0] * group_size + [1] * (rows - group_size),
        dtype=torch.int32,
        device="cuda",
    )
    first_position = max_seq_len_kv - group_size - 1
    positions = torch.tensor(
        [first_position + index for index in range(group_size)]
        + [first_position + index for index in range(rows - group_size)],
        dtype=torch.int64,
        device="cuda",
    )
    logical_block_bound = (max_seq_len_kv + 3) // 4
    block_ranks = torch.arange(512, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [
            (block_ranks * 61 + row * 193) % (logical_block_bound - 1)
            for row in range(rows)
        ]
    ).contiguous()
    table_width = (max_seq_len_kv + storage_page_size - 1) // storage_page_size
    block_table = (
        torch.arange(2 * table_width, dtype=torch.int32, device="cuda")
        .reshape(2, table_width)
        .mul(3)
        .add(7)
    )
    qo_indptr = None
    if packed:
        qo_indptr = torch.tensor(
            [0, group_size, rows],
            dtype=torch.int32,
            device="cuda",
        )
    return (
        blocks,
        block_table,
        requests,
        positions,
        storage_page_size,
        qo_indptr,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (2, 4, 5))
@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("max_seq_len_kv", (8192, 9600))
def test_q_token_kv_block_sparse_sort_union_matches_reference(
    group_size: int,
    packed: bool,
    max_seq_len_kv: int,
) -> None:
    (
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        qo_indptr,
    ) = _make_sort_union_case(group_size, packed)
    output_shapes = get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
        blocks.shape[0],
        blocks.shape[1],
        group_size,
        num_query_groups=2,
    )
    outputs = tuple(
        torch.full(shape, -7, dtype=torch.int32, device="cuda")
        for shape in output_shapes
    )
    actual = build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        qo_indptr=qo_indptr,
        out=outputs,
    )
    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
        qo_indptr,
    )
    _assert_metadata_matches(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("group_size", "packed", "max_seq_len_kv"),
    (
        (4, False, 16_385),
        (2, False, 32 * 1024),
        (4, True, 64 * 1024),
        (5, False, 128 * 1024),
    ),
)
def test_q_token_kv_block_sparse_sort_union_wide_context_matches_reference(
    group_size: int,
    packed: bool,
    max_seq_len_kv: int,
) -> None:
    (
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        qo_indptr,
    ) = _make_wide_sort_union_case(group_size, packed, max_seq_len_kv)
    outputs = tuple(
        torch.full(shape, -7, dtype=torch.int32, device="cuda")
        for shape in get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            blocks.shape[0],
            blocks.shape[1],
            group_size,
            num_query_groups=2,
        )
    )
    actual = build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        qo_indptr=qo_indptr,
        out=outputs,
    )
    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
        qo_indptr,
    )
    _assert_metadata_matches(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q_token_kv_block_sparse_sort_union_emits_partial_causal_tail() -> None:
    """Emit the partial causal tail after the sorted selected-page union."""

    group_size = 5
    max_seq_len_kv = 16_385
    sparse_block_size = 4
    storage_page_size = 16
    block_topk = 512
    logical_block_bound = (max_seq_len_kv + sparse_block_size - 1) // sparse_block_size
    assert (logical_block_bound + 31) // 32 == 2 * 64 + 1
    block_ranks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [
            (block_ranks * 61 + row * 193) % (logical_block_bound - 1)
            for row in range(group_size)
        ]
    ).contiguous()
    table_width = (max_seq_len_kv + storage_page_size - 1) // storage_page_size
    table = torch.arange(table_width, dtype=torch.int32, device="cuda").unsqueeze(0)
    requests = torch.zeros(group_size, dtype=torch.int32, device="cuda")
    positions = torch.arange(
        max_seq_len_kv - group_size,
        max_seq_len_kv,
        dtype=torch.int64,
        device="cuda",
    )
    outputs = tuple(
        torch.full(shape, -7, dtype=torch.int32, device="cuda")
        for shape in get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            group_size,
            block_topk,
            group_size,
            sparse_block_size=sparse_block_size,
        )
    )

    actual = build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        sparse_block_size=sparse_block_size,
        out=outputs,
    )
    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
    )
    _assert_metadata_matches(actual, expected)

    live_pages = (int(actual[2][0].item()) + sparse_block_size - 1) // sparse_block_size
    final_slot = live_pages - 1
    assert int(actual[0][0, final_slot].item()) == logical_block_bound - 1
    assert int(actual[1].view(torch.uint8)[0, final_slot].item()) == 1 << (
        group_size - 1
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("group_size", "packed"), ((4, True), (5, False)))
def test_q_token_kv_block_sparse_sort_union_graph_replay_is_exact(
    group_size: int,
    packed: bool,
) -> None:
    """Rebuild exact metadata as a captured route shrinks and grows."""

    max_seq_len_kv = 128 * 1024
    storage_page_size = 16
    rows = group_size - 1 if packed else group_size
    qo_indptr = (
        torch.tensor((0, rows), dtype=torch.int32, device="cuda") if packed else None
    )
    groups = 1
    block_topk = 512
    requests = torch.zeros(rows, dtype=torch.int32, device="cuda")
    block_ranks = torch.arange(block_topk, dtype=torch.int32, device="cuda")

    def make_blocks(logical_block_bound: int) -> torch.Tensor:
        return torch.stack(
            [
                (block_ranks * 61 + row * 193) % logical_block_bound
                for row in range(rows)
            ]
        ).contiguous()

    short_first_position = 8192 - rows + 1
    short_positions = torch.arange(
        short_first_position,
        short_first_position + rows,
        dtype=torch.int64,
        device="cuda",
    )
    long_positions = torch.arange(
        max_seq_len_kv - rows,
        max_seq_len_kv,
        dtype=torch.int64,
        device="cuda",
    )
    short_blocks = make_blocks(short_first_position // 4)
    long_blocks = make_blocks(max_seq_len_kv // 4 - 1)
    positions = short_positions.clone()
    blocks = short_blocks.clone()
    table_width = (max_seq_len_kv + storage_page_size - 1) // storage_page_size
    table = torch.arange(table_width, dtype=torch.int32, device="cuda").unsqueeze(0)
    outputs = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            rows,
            block_topk,
            group_size,
            num_query_groups=groups,
        )
    )

    def run() -> None:
        build_prims_ts_q_token_kv_block_sparse_metadata(
            blocks,
            table,
            requests,
            positions,
            group_size=group_size,
            storage_page_size=storage_page_size,
            max_seq_len_kv=max_seq_len_kv,
            qo_indptr=qo_indptr,
            out=outputs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    blocks.copy_(long_blocks)
    positions.copy_(long_positions)
    graph.replay()
    torch.cuda.synchronize()
    _assert_metadata_matches(
        outputs,
        _reference(
            blocks,
            table,
            requests,
            positions,
            storage_page_size,
            group_size,
            qo_indptr,
        ),
    )

    blocks.copy_(short_blocks)
    positions.copy_(short_positions)
    graph.replay()
    torch.cuda.synchronize()
    _assert_metadata_matches(
        outputs,
        _reference(
            blocks,
            table,
            requests,
            positions,
            storage_page_size,
            group_size,
            qo_indptr,
        ),
    )

    blocks.copy_(long_blocks)
    positions.copy_(long_positions)
    graph.replay()
    torch.cuda.synchronize()
    _assert_metadata_matches(
        outputs,
        _reference(
            blocks,
            table,
            requests,
            positions,
            storage_page_size,
            group_size,
            qo_indptr,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_q_token_kv_block_sparse_masks_blocks_beyond_small_logical_bound() -> (
    None
):
    """Ignore IDs above an actual-sequence bound despite a wide block table."""

    group_size = 2
    storage_page_size = 16
    max_seq_len_kv = 2051
    logical_block_bound = (max_seq_len_kv + 3) // 4
    blocks = torch.tensor(
        (
            (0, logical_block_bound - 1, logical_block_bound, 543),
            (0, logical_block_bound - 1, logical_block_bound, 543),
        ),
        dtype=torch.int32,
        device="cuda",
    )
    table = torch.arange(1024, dtype=torch.int32, device="cuda").unsqueeze(0)
    requests = torch.zeros(group_size, dtype=torch.int32, device="cuda")
    positions = torch.tensor((2049, 2050), dtype=torch.int64, device="cuda")
    outputs = tuple(
        torch.full(shape, -7, dtype=torch.int32, device="cuda")
        for shape in get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            blocks.shape[0],
            blocks.shape[1],
            group_size,
        )
    )

    (
        q_token_kv_block_sparse_page_indices,
        q_token_kv_block_sparse_page_memberships,
        seq_lens,
    ) = build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        out=outputs,
    )

    torch.testing.assert_close(
        q_token_kv_block_sparse_page_indices[0, :2],
        torch.tensor((0, logical_block_bound - 1), dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        q_token_kv_block_sparse_page_memberships.view(torch.uint8)[0, :2],
        torch.full((2,), 0b11, dtype=torch.uint8, device="cuda"),
    )
    torch.testing.assert_close(
        seq_lens,
        torch.tensor((7,), dtype=torch.int32, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("packed", (False, True))
def test_grouped_q_token_kv_block_sparse_rejects_selected_blocks_beyond_each_query_causal_prefix(
    packed: bool,
) -> None:
    """Do not admit a future complete block through the selected-index list."""

    group_size = 4
    block_topk = 8
    storage_page_size = 16
    max_seq_len_kv = 128
    blocks = torch.tensor(
        (
            (0, 4, 5, 6, -1, -1, -1, -1),
            (0, 4, 5, 6, -1, -1, -1, -1),
            (0, 4, 5, 6, -1, -1, -1, -1),
            (0, 4, 5, 6, 7, -1, -1, -1),
        ),
        dtype=torch.int32,
        device="cuda",
    )
    table = torch.arange(
        max_seq_len_kv // storage_page_size,
        dtype=torch.int32,
        device="cuda",
    ).unsqueeze(0)
    requests = torch.zeros(group_size, dtype=torch.int32, device="cuda")
    positions = torch.tensor((20, 21, 22, 23), dtype=torch.int64, device="cuda")
    qo_indptr = (
        torch.tensor((0, group_size), dtype=torch.int32, device="cuda")
        if packed
        else None
    )
    outputs = tuple(
        torch.full(shape, -7, dtype=torch.int32, device="cuda")
        for shape in get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            group_size,
            block_topk,
            group_size,
            num_query_groups=1,
        )
    )

    actual = build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_seq_len_kv,
        qo_indptr=qo_indptr,
        out=outputs,
    )
    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
        qo_indptr,
    )

    _assert_metadata_matches(actual, expected)
    torch.testing.assert_close(
        actual[0][0, :3],
        torch.tensor((0, 4, 5), dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        actual[1].view(torch.uint8)[0, :3],
        torch.full((3,), 0b1111, dtype=torch.uint8, device="cuda"),
    )
    torch.testing.assert_close(
        actual[2],
        torch.tensor((12,), dtype=torch.int32, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("max_num_storage_pages", (64, 1024))
def test_grouped_q_token_kv_block_sparse_empty_union_publishes_positive_seq_len(
    max_num_storage_pages: int,
) -> None:
    """Keep both grouped pack implementations inside the positive-length ABI."""

    group_size = 4
    block_topk = 8
    storage_page_size = 16
    blocks = torch.full((1, block_topk), -1, dtype=torch.int32, device="cuda")
    table = torch.zeros(
        (1, max_num_storage_pages),
        dtype=torch.int32,
        device="cuda",
    )
    requests = torch.zeros(1, dtype=torch.int32, device="cuda")
    # One packed partial group ending exactly on a page boundary has no
    # implicit causal-tail page. Invalid selected IDs therefore form a valid
    # zero-union stress case.
    positions = torch.tensor((3,), dtype=torch.int64, device="cuda")
    qo_indptr = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    outputs = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            blocks.shape[0],
            block_topk,
            group_size,
            num_query_groups=1,
        )
    )
    for tensor in outputs:
        tensor.fill_(0x55555555)

    (
        q_token_kv_block_sparse_page_indices,
        q_token_kv_block_sparse_page_memberships,
        seq_lens,
    ) = build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        qo_indptr=qo_indptr,
        out=outputs,
    )

    torch.testing.assert_close(seq_lens, torch.ones_like(seq_lens))
    torch.testing.assert_close(
        q_token_kv_block_sparse_page_indices[:, 0],
        torch.full_like(q_token_kv_block_sparse_page_indices[:, 0], -1),
    )
    torch.testing.assert_close(
        q_token_kv_block_sparse_page_memberships.view(torch.uint8)[:, 0],
        torch.zeros_like(
            q_token_kv_block_sparse_page_memberships.view(torch.uint8)[:, 0]
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q_token_kv_block_sparse_sort_union_cuda_graph_reloads_and_invalidates() -> (
    None
):
    (
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        _,
    ) = _make_sort_union_case(5, False)
    output_shapes = get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
        blocks.shape[0], blocks.shape[1], 5
    )
    outputs = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda") for shape in output_shapes
    )
    outputs[1].fill_(0x5A5A5A5A)

    def run() -> None:
        build_prims_ts_q_token_kv_block_sparse_metadata(
            blocks,
            table,
            requests,
            positions,
            group_size=5,
            storage_page_size=storage_page_size,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            out=outputs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    table.add_(101)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(blocks, table, requests, positions, storage_page_size, 5)
    _assert_metadata_matches(outputs, expected)

    positions[2].add_(1)
    graph.replay()
    torch.cuda.synchronize()
    assert outputs[2][0].item() == 1
    assert outputs[0][0, 0].item() == -1

    # Restore a valid group and grow its live union. Every newly observable
    # page index and membership word must be produced by this replay; bytes
    # beyond the rounded live word prefix remain intentionally unspecified.
    positions[2].sub_(1)
    blocks[2].add_(16)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(blocks, table, requests, positions, storage_page_size, 5)
    _assert_metadata_matches(outputs, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "api",
    (
        prepare_prims_ts_q_token_kv_block_sparse_attention,
        prims_ts_q_token_kv_block_sparse_attention,
    ),
)
@pytest.mark.parametrize("overlap_name", ("block_indices", "out"))
def test_q_token_kv_block_sparse_attention_apis_reject_unified_workspace_overlap(
    api: object,
    overlap_name: str,
) -> None:
    """Both entry points must invoke the unified workspace alias guard."""

    blocks, table, requests, positions, storage_page_size = _make_case(1, 8)
    query = torch.empty((2, 1, 1, 12, 256), dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(query)
    k_cache = torch.empty(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.empty_like(k_cache)
    overlap = blocks if overlap_name == "block_indices" else out
    workspace = overlap.reshape(-1).view(torch.uint8)

    with pytest.raises(
        ValueError,
        match=rf"workspace_buffer must not overlap {overlap_name} storage",
    ):
        api(  # type: ignore[operator]
            query,
            (k_cache, v_cache),
            blocks,
            table,
            requests,
            positions,
            workspace,
            out=out,
            max_seq_len_kv=table.shape[1] * storage_page_size,
        )


def _fixed_q_token_kv_block_sparse_shape(
    num_routes: int,
    group_size: int,
    num_qo_heads: int = 12,
    head_dim: int = 256,
) -> tuple[int, int, int, int, int]:
    """Return the canonical fixed decode shape [B, Nq, G, Hq, D]."""

    return (num_routes, 1, group_size, num_qo_heads, head_dim)


def _as_lower_level_decode_view(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten canonical fixed QToken-KvBlock-Sparse-Attention route axes without copying storage."""

    flattened = tensor.flatten(0, 1)
    return flattened.squeeze(1) if tensor.shape[2] == 1 else flattened


def _run_private_q_token_kv_block_sparse_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_token_kv_block_sparse_page_indices: torch.Tensor,
    q_token_kv_block_sparse_page_memberships: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    group_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Exercise the lower-level QToken-KvBlock-Sparse-Attention route without exposing it publicly."""

    from flashinfer.attention.prims_ts.decode import (
        _prepare_prims_ts_batch_decode_plan,
        _resolve_decode_workspace_layout,
    )

    layout = _resolve_decode_workspace_layout(
        int(query.shape[0]),
        int(query.shape[-2]),
        int(k_cache.shape[1]),
        int(query.shape[-1]),
        4,
        max_seq_len,
        group_size,
        query.dtype,
        k_cache.dtype,
        out.dtype,
        "HND",
        "causal",
        False,
        -1,
        int(k_cache.shape[2]),
        query.device,
        use_q_token_kv_block_sparse_route=True,
    )
    workspace = torch.zeros(layout.total_bytes, dtype=torch.uint8, device=query.device)
    if group_size == 1:
        assert q_token_kv_block_sparse_page_memberships.shape == (
            q_token_kv_block_sparse_page_indices.shape[0],
            0,
        )
    plan, prepared_out = _prepare_prims_ts_batch_decode_plan(
        query,
        (k_cache, v_cache),
        workspace,
        q_token_kv_block_sparse_page_indices,
        seq_lens,
        max_seq_len,
        seq_len_q=group_size,
        qo_indptr=None,
        max_seq_len_q=None,
        out=out,
        out_dtype=out.dtype,
        mask_type="causal",
        window_left=-1,
        kv_layout="HND",
        page_size=4,
        use_q_token_kv_block_sparse_route=True,
        q_token_kv_block_sparse_page_memberships=(
            q_token_kv_block_sparse_page_memberships if group_size > 1 else None
        ),
    )
    return plan.run(query, out=prepared_out)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("group_size", "segment_memberships", "expected_keeps"),
    (
        (2, (0b01, 0b10, 0b11, 0b01, 0b10), False),
        (4, (0b0001, 0b0010, 0b0100, 0b1111, 0b1000), True),
    ),
)
def test_grouped_split_q_token_kv_block_sparse_memberships_cross_kv128_boundaries(
    group_size: int,
    segment_memberships: tuple[int, ...],
    expected_keeps: bool,
) -> None:
    """Keep split-local membership addressing aligned to each KV128 tile.

    Every 32 page-four locators form one KV128 tile.  Using a different
    membership mask in each tile makes a split that restarts the membership
    pointer, or applies the locator stride to membership words, visibly route
    pages to the wrong query.  Q2 covers the production SwapsMmaAb path and Q4
    covers KeepsMmaAb; both resolve to a separate-reduction split-KV launch.
    """

    from flashinfer.attention.prims_ts import decode as decode_module

    batch_size = 1
    num_qo_heads = 12
    num_kv_heads = 1
    head_dim = 256
    sparse_page_size = 4
    storage_page_size = 16
    pages_per_kv128 = 128 // sparse_page_size
    num_q_token_kv_block_sparse_pages = len(segment_memberships) * pages_per_kv128
    max_seq_len = num_q_token_kv_block_sparse_pages * sparse_page_size

    decode_module._resolve_decode_launch_spec.cache_clear()
    spec = decode_module._resolve_decode_launch_spec(
        torch.cuda.current_device(),
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        sparse_page_size,
        max_seq_len,
        group_size,
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "HND",
        "causal",
        False,
        -1,
        storage_page_size,
        True,
        False,
    )
    assert spec.config.use_keeps_mma_ab is expected_keeps
    assert spec.config.tile_size_q == (64 if expected_keeps else 32)
    # The exact first-wave fanout is device-capacity dependent; this test only
    # requires a real multi-split launch to exercise split-local addressing.
    assert spec.config.splits_kv == spec.config.max_splits_kv
    assert spec.config.splits_kv > 1
    assert spec.config.use_separate_reduction_kernel

    q_token_kv_block_sparse_page_indices = torch.arange(
        num_q_token_kv_block_sparse_pages, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)
    membership_bytes = torch.tensor(
        [
            membership
            for membership in segment_memberships
            for _ in range(pages_per_kv128)
        ],
        dtype=torch.uint8,
        device="cuda",
    ).unsqueeze(0)
    assert membership_bytes.shape[1] % 4 == 0
    q_token_kv_block_sparse_page_memberships = membership_bytes.contiguous().view(
        torch.int32
    )
    seq_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device="cuda")

    query = torch.zeros(
        batch_size,
        group_size,
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    num_storage_pages = (
        num_q_token_kv_block_sparse_pages * sparse_page_size // storage_page_size
    )
    k_cache = torch.zeros(
        num_storage_pages,
        num_kv_heads,
        storage_page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # Each page-four locator owns a distinct vector, repeated for its four
    # tokens.  With Q=K=0, each query output is the exact mean of the vectors
    # selected by that query's membership bit.
    torch.manual_seed(1771 + group_size)
    page_values = torch.randn(
        num_q_token_kv_block_sparse_pages,
        head_dim,
        dtype=torch.float32,
        device="cuda",
    )
    v_cache = (
        page_values.to(torch.bfloat16)
        .unsqueeze(1)
        .expand(-1, sparse_page_size, -1)
        .reshape(num_storage_pages, num_kv_heads, storage_page_size, head_dim)
        .contiguous()
    )
    output = torch.empty_like(query)
    _run_private_q_token_kv_block_sparse_decode(
        query,
        k_cache,
        v_cache,
        q_token_kv_block_sparse_page_indices,
        q_token_kv_block_sparse_page_memberships,
        seq_lens,
        max_seq_len,
        group_size,
        output,
    )
    torch.cuda.synchronize()

    reference = torch.empty_like(output, dtype=torch.float32)
    for query_index in range(group_size):
        selected_pages = (membership_bytes[0] & (1 << query_index)).bool()
        assert selected_pages.any()
        selected_mean = page_values[selected_pages].mean(dim=0)
        reference[0, query_index] = selected_mean.unsqueeze(0).expand(num_qo_heads, -1)
    torch.testing.assert_close(output.float(), reference, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
def test_q_token_kv_block_sparse_attention_hides_workspace_metadata(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    num_routes = table.shape[0]
    num_qo_heads = 12
    head_dim = 256
    query_shape = _fixed_q_token_kv_block_sparse_shape(
        num_routes, group_size, num_qo_heads, head_dim
    )
    query = torch.zeros(query_shape, dtype=torch.bfloat16, device="cuda")
    num_physical_pages = int(table.max().item()) + 1
    k_cache = torch.zeros(
        num_physical_pages,
        1,
        storage_page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    workspace_bytes = get_prims_ts_q_token_kv_block_sparse_workspace_size(
        query,
        k_cache,
        table,
        block_topk=blocks.shape[1],
        max_seq_len_kv=table.shape[1] * storage_page_size,
        out_dtype=torch.bfloat16,
    )
    workspace_layout = _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
        blocks.shape[0],
        blocks.shape[1],
        table.shape[1],
        storage_page_size,
        group_size,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=1,
        head_dim=head_dim,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        device="cuda",
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
    views = workspace_layout.bind(workspace)
    output = torch.empty_like(query)
    calls: dict[str, object] = {}

    class FakeAttentionPlan:
        def _run_unchecked(
            self,
            actual_query: torch.Tensor,
            out: torch.Tensor,
            scale_qk: float,
            scale_v: float,
        ) -> torch.Tensor:
            expected_query = _as_lower_level_decode_view(query)
            expected_output = _as_lower_level_decode_view(output)
            assert actual_query.shape == expected_query.shape
            assert actual_query.data_ptr() == expected_query.data_ptr()
            assert out.shape == expected_output.shape
            assert out.data_ptr() == expected_output.data_ptr()
            assert scale_qk == pytest.approx(head_dim**-0.5)
            assert scale_v == 1.0
            return out

    def fake_prepare(
        actual_query: torch.Tensor,
        _paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        scratch_buffer: torch.Tensor,
        q_token_kv_block_sparse_page_indices: torch.Tensor,
        actual_seq_lens: torch.Tensor,
        _max_seq_len: int,
        **kwargs,
    ) -> tuple[FakeAttentionPlan, torch.Tensor]:
        calls["scratch"] = scratch_buffer
        calls["page_indices"] = q_token_kv_block_sparse_page_indices
        expected_query = _as_lower_level_decode_view(query)
        expected_output = _as_lower_level_decode_view(output)
        assert actual_query.shape == expected_query.shape
        assert actual_query.data_ptr() == expected_query.data_ptr()
        assert (
            q_token_kv_block_sparse_page_indices.data_ptr()
            == views.q_token_kv_block_sparse_page_indices.data_ptr()
        )
        q_token_kv_block_sparse_page_memberships = kwargs[
            "q_token_kv_block_sparse_page_memberships"
        ]
        if group_size == 1:
            assert q_token_kv_block_sparse_page_memberships is None
            assert views.q_token_kv_block_sparse_page_memberships.shape == (
                num_routes,
                0,
            )
        else:
            assert isinstance(q_token_kv_block_sparse_page_memberships, torch.Tensor)
            assert (
                q_token_kv_block_sparse_page_memberships.data_ptr()
                == views.q_token_kv_block_sparse_page_memberships.data_ptr()
            )
        assert actual_seq_lens.data_ptr() == views.seq_lens.data_ptr()
        assert kwargs["out"].shape == expected_output.shape
        assert kwargs["out"].data_ptr() == expected_output.data_ptr()
        assert scratch_buffer.data_ptr() == (
            workspace.data_ptr() + workspace_layout.attention_workspace_byte_offset
        )
        return FakeAttentionPlan(), expected_output

    monkeypatch.setattr(
        decode_module,
        "_prepare_prims_ts_batch_decode_plan",
        fake_prepare,
    )
    assert (
        prims_ts_q_token_kv_block_sparse_attention(
            query,
            (k_cache, v_cache),
            blocks,
            table,
            requests,
            positions,
            workspace,
            out=output,
            max_seq_len_kv=table.shape[1] * storage_page_size,
        )
        is output
    )

    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
    )
    q_token_kv_block_sparse_page_indices = calls["page_indices"]
    scratch = calls["scratch"]
    assert isinstance(q_token_kv_block_sparse_page_indices, torch.Tensor)
    assert isinstance(scratch, torch.Tensor)
    _assert_metadata_matches(
        (
            q_token_kv_block_sparse_page_indices,
            views.q_token_kv_block_sparse_page_memberships,
            views.seq_lens,
        ),
        expected,
    )
    assert scratch.data_ptr() > q_token_kv_block_sparse_page_indices.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 4))
def test_q_token_kv_block_sparse_fixed_5d_layout_flattens_route_axes_without_copy(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    batch_size = 2
    groups_per_request = 1
    query = torch.zeros(
        batch_size,
        groups_per_request,
        group_size,
        12,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty_like(query)
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    workspace = torch.empty(
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            table,
            block_topk=blocks.shape[1],
            max_seq_len_kv=table.shape[1] * storage_page_size,
            out_dtype=output.dtype,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    calls: dict[str, torch.Tensor] = {}

    class FakeAttentionPlan:
        def _run_unchecked(
            self,
            actual_query: torch.Tensor,
            out: torch.Tensor,
            scale_qk: float,
            scale_v: float,
        ) -> torch.Tensor:
            expected_shape = (
                (batch_size, 12, 256)
                if group_size == 1
                else (batch_size, group_size, 12, 256)
            )
            assert actual_query.shape == expected_shape
            assert out.shape == actual_query.shape
            assert actual_query.data_ptr() == query.data_ptr()
            assert out.data_ptr() == output.data_ptr()
            assert scale_qk == pytest.approx(256**-0.5)
            assert scale_v == pytest.approx(0.25)
            out.fill_(3)
            calls["query"] = actual_query
            return out

    def fake_prepare(
        actual_query: torch.Tensor,
        _paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
        _scratch_buffer: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[FakeAttentionPlan, torch.Tensor]:
        actual_output = kwargs["out"]
        assert isinstance(actual_output, torch.Tensor)
        expected_shape = (
            (batch_size, 12, 256)
            if group_size == 1
            else (batch_size, group_size, 12, 256)
        )
        assert actual_query.shape == expected_shape
        assert actual_output.shape == actual_query.shape
        assert actual_query.data_ptr() == query.data_ptr()
        assert actual_output.data_ptr() == output.data_ptr()
        return FakeAttentionPlan(), actual_output

    monkeypatch.setattr(
        decode_module,
        "_prepare_prims_ts_batch_decode_plan",
        fake_prepare,
    )
    result = prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        bmm2_scale=0.25,
        out=output,
        max_seq_len_kv=table.shape[1] * storage_page_size,
    )
    assert result is output
    assert "query" in calls
    assert torch.all(output == 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 4))
def test_prepared_q_token_kv_block_sparse_fixed_5d_plan_flattens_runtime_views(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    query = torch.zeros(2, 1, group_size, 12, 256, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    workspace = torch.empty(
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            table,
            block_topk=blocks.shape[1],
            max_seq_len_kv=table.shape[1] * storage_page_size,
            out_dtype=output.dtype,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    calls: dict[str, bool] = {}

    class FakeAttentionPlan:
        def _run_unchecked(
            self,
            actual_query: torch.Tensor,
            actual_output: torch.Tensor,
            _scale_qk: float,
            _scale_v: float,
        ) -> torch.Tensor:
            expected_shape = (
                (2, 12, 256) if group_size == 1 else (2, group_size, 12, 256)
            )
            assert actual_query.shape == expected_shape
            assert actual_output.shape == actual_query.shape
            assert actual_query.data_ptr() == query.data_ptr()
            assert actual_output.data_ptr() == output.data_ptr()
            actual_output.fill_(4)
            calls["ran"] = True
            return actual_output

    def fake_prepare(
        actual_query: torch.Tensor,
        _cache: tuple[torch.Tensor, torch.Tensor],
        _attention_workspace: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[FakeAttentionPlan, torch.Tensor]:
        actual_output = kwargs["out"]
        assert isinstance(actual_output, torch.Tensor)
        expected_shape = (2, 12, 256) if group_size == 1 else (2, group_size, 12, 256)
        assert actual_query.shape == expected_shape
        assert actual_output.shape == actual_query.shape
        return FakeAttentionPlan(), actual_output

    monkeypatch.setattr(
        decode_module,
        "_prepare_prims_ts_batch_decode_plan",
        fake_prepare,
    )
    plan = prepare_prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
        max_seq_len_kv=table.shape[1] * storage_page_size,
    )
    assert (
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=output,
        )
        is output
    )
    assert calls["ran"] is True
    assert torch.all(output == 4)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
def test_q_token_kv_block_sparse_fixed_decode_matches_packed_prefill_layout(
    group_size: int,
) -> None:
    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    routes = table.shape[0]
    torch.manual_seed(42)
    fixed_query = torch.randn(
        _fixed_q_token_kv_block_sparse_shape(routes, group_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    packed_query = fixed_query.reshape(blocks.shape[0], 12, 256)
    qo_indptr = torch.arange(
        0,
        blocks.shape[0] + 1,
        group_size,
        dtype=torch.int32,
        device="cuda",
    )
    k_cache = torch.randn(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)

    def run(
        query: torch.Tensor,
        *,
        packed_qo_indptr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        workspace = torch.empty(
            get_prims_ts_q_token_kv_block_sparse_workspace_size(
                query,
                k_cache,
                table,
                block_topk=blocks.shape[1],
                max_seq_len_kv=table.shape[1] * storage_page_size,
                out_dtype=output.dtype,
                qo_indptr=packed_qo_indptr,
                max_seq_len_q=group_size if packed_qo_indptr is not None else None,
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        return prims_ts_q_token_kv_block_sparse_attention(
            query,
            (k_cache, v_cache),
            blocks,
            table,
            requests,
            positions,
            workspace,
            out=output,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            qo_indptr=packed_qo_indptr,
            max_seq_len_q=group_size if packed_qo_indptr is not None else None,
        )

    fixed_output = run(fixed_query)
    packed_output = run(packed_query, packed_qo_indptr=qo_indptr)
    torch.testing.assert_close(fixed_output.reshape_as(packed_output), packed_output)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q_token_kv_block_sparse_fixed_decode_supports_suggested_padded_final_route() -> (
    None
):
    """Automatic Q2 may pad SQ3 with one disposable semantic row."""

    seq_len_q = 3
    group_size = suggest_prims_ts_q_token_kv_block_sparse_group_size(
        batch_size=1,
        seq_len_q=seq_len_q,
        selected_seq_len_kv=2051,
        num_qo_heads=12,
        num_kv_heads=1,
        multi_processor_count=1,
    )
    assert group_size == 2
    live_rows = seq_len_q
    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    rows = math.ceil(seq_len_q / group_size) * group_size
    blocks = blocks[:rows].clone()
    table = table[:1].clone()
    requests = torch.zeros(rows, dtype=torch.int32, device="cuda")
    positions = torch.arange(
        int(positions[0].item()),
        int(positions[0].item()) + rows,
        dtype=torch.int64,
        device="cuda",
    )
    torch.manual_seed(2026)
    fixed_query = torch.randn(
        1,
        rows // group_size,
        group_size,
        12,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    packed_query = fixed_query.reshape(rows, 12, 256)[:live_rows].contiguous()
    qo_indptr = torch.tensor(
        (0, group_size, live_rows), dtype=torch.int32, device="cuda"
    )
    k_cache = torch.randn(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)

    def run(
        query: torch.Tensor,
        actual_blocks: torch.Tensor,
        actual_requests: torch.Tensor,
        actual_positions: torch.Tensor,
        *,
        packed_qo_indptr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        workspace = torch.empty(
            get_prims_ts_q_token_kv_block_sparse_workspace_size(
                query,
                k_cache,
                table,
                block_topk=blocks.shape[1],
                max_seq_len_kv=table.shape[1] * storage_page_size,
                out_dtype=output.dtype,
                qo_indptr=packed_qo_indptr,
                max_seq_len_q=(group_size if packed_qo_indptr is not None else None),
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        return prims_ts_q_token_kv_block_sparse_attention(
            query,
            (k_cache, v_cache),
            actual_blocks,
            table,
            actual_requests,
            actual_positions,
            workspace,
            out=output,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            qo_indptr=packed_qo_indptr,
            max_seq_len_q=(group_size if packed_qo_indptr is not None else None),
        )

    fixed_output = run(fixed_query, blocks, requests, positions)
    packed_output = run(
        packed_query,
        blocks[:live_rows],
        requests[:live_rows],
        positions[:live_rows],
        packed_qo_indptr=qo_indptr,
    )
    assert torch.isfinite(fixed_output).all()
    torch.testing.assert_close(
        fixed_output.reshape(rows, 12, 256)[:live_rows],
        packed_output,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q_token_kv_block_sparse_fixed_multiple_query_groups_match_packed_layout() -> (
    None
):
    """Fixed [B,Nq,G,Hq,D] preserves route order when Nq is greater than one."""

    group_size = 4
    groups_per_request = 2
    blocks, table, _, _, storage_page_size = _make_case(group_size, 8)
    batch_size = table.shape[0]
    block_topk = blocks.shape[1]
    blocks = (
        blocks.reshape(batch_size, group_size, block_topk)
        .repeat_interleave(groups_per_request, dim=0)
        .reshape(batch_size * groups_per_request * group_size, block_topk)
        .contiguous()
    )
    requests = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    requests = requests.repeat_interleave(groups_per_request * group_size)
    base_position = 4 * block_topk - 1
    positions = (
        torch.arange(
            groups_per_request * group_size,
            dtype=torch.int64,
            device="cuda",
        )
        .repeat(batch_size)
        .add_(base_position)
    )
    torch.manual_seed(31415)
    fixed_query = torch.randn(
        batch_size,
        groups_per_request,
        group_size,
        12,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    packed_query = fixed_query.reshape(-1, 12, 256)
    qo_indptr = torch.arange(
        0,
        packed_query.shape[0] + 1,
        group_size,
        dtype=torch.int32,
        device="cuda",
    )
    k_cache = torch.randn(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)

    def run(
        query: torch.Tensor,
        *,
        packed_qo_indptr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        workspace = torch.empty(
            get_prims_ts_q_token_kv_block_sparse_workspace_size(
                query,
                k_cache,
                table,
                block_topk=block_topk,
                max_seq_len_kv=table.shape[1] * storage_page_size,
                out_dtype=output.dtype,
                qo_indptr=packed_qo_indptr,
                max_seq_len_q=(group_size if packed_qo_indptr is not None else None),
            ),
            dtype=torch.uint8,
            device="cuda",
        )
        return prims_ts_q_token_kv_block_sparse_attention(
            query,
            (k_cache, v_cache),
            blocks,
            table,
            requests,
            positions,
            workspace,
            out=output,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            qo_indptr=packed_qo_indptr,
            max_seq_len_q=group_size if packed_qo_indptr is not None else None,
        )

    fixed_output = run(fixed_query)
    packed_output = run(packed_query, packed_qo_indptr=qo_indptr)
    torch.testing.assert_close(fixed_output.reshape_as(packed_output), packed_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
def test_prepared_q_token_kv_block_sparse_plan_keeps_metadata_and_attention_scratch_disjoint(
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    num_routes = table.shape[0]
    query_shape = _fixed_q_token_kv_block_sparse_shape(num_routes, group_size)
    query = torch.zeros(query_shape, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.zeros_like(k_cache)
    output = torch.empty_like(query)
    layout = _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
        blocks.shape[0],
        blocks.shape[1],
        table.shape[1],
        storage_page_size,
        group_size,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=output.dtype,
        device="cuda",
    )
    workspace = torch.empty(layout.total_bytes, dtype=torch.uint8, device="cuda")
    views = layout.bind(workspace)
    calls: dict[str, object] = {}

    class FakeAttentionPlan:
        def _run_unchecked(
            self,
            actual_query: torch.Tensor,
            actual_output: torch.Tensor,
            scale_qk: float,
            scale_v: float,
        ) -> torch.Tensor:
            expected_query = _as_lower_level_decode_view(query)
            assert actual_query.shape == expected_query.shape
            assert actual_query.stride() == expected_query.stride()
            assert actual_output.shape == expected_query.shape
            assert actual_output.stride() == expected_query.stride()
            assert scale_qk == pytest.approx(256**-0.5)
            assert scale_v == 1.0
            actual_output.zero_()
            calls["ran"] = True
            calls["query_ptr"] = actual_query.data_ptr()
            calls["output_ptr"] = actual_output.data_ptr()
            return actual_output

    def fake_prepare(
        actual_query: torch.Tensor,
        _cache: tuple[torch.Tensor, torch.Tensor],
        attention_workspace: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> tuple[FakeAttentionPlan, torch.Tensor]:
        expected_query = _as_lower_level_decode_view(query)
        assert actual_query.shape == expected_query.shape
        assert actual_query.data_ptr() == expected_query.data_ptr()
        assert attention_workspace.data_ptr() == (
            workspace.data_ptr() + layout.attention_workspace_byte_offset
        )
        actual_output = kwargs["out"]
        assert isinstance(actual_output, torch.Tensor)
        return FakeAttentionPlan(), actual_output

    monkeypatch.setattr(
        decode_module,
        "_prepare_prims_ts_batch_decode_plan",
        fake_prepare,
    )
    plan = prepare_prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
        max_seq_len_kv=table.shape[1] * storage_page_size,
    )
    views.attention_workspace_buffer.fill_(0x5A)
    assert (
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=output,
        )
        is output
    )
    torch.cuda.synchronize()
    assert calls["ran"] is True
    assert torch.all(views.attention_workspace_buffer == 0x5A)
    assert (
        plan._metadata_plan.q_token_kv_block_sparse_page_indices.data_ptr()
        == views.q_token_kv_block_sparse_page_indices.data_ptr()
    )
    assert (
        plan._metadata_plan.q_token_kv_block_sparse_page_memberships.data_ptr()
        == views.q_token_kv_block_sparse_page_memberships.data_ptr()
    )
    assert plan._metadata_plan.seq_lens.data_ptr() == views.seq_lens.data_ptr()
    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
    )
    _assert_metadata_matches(
        (
            views.q_token_kv_block_sparse_page_indices,
            views.q_token_kv_block_sparse_page_memberships,
            views.seq_lens,
        ),
        expected,
    )

    replacement_query = query.clone()
    replacement_blocks = blocks.clone()
    replacement_table = table.clone()
    replacement_requests = requests.clone()
    replacement_positions = positions.clone()
    replacement_output = torch.empty_like(output)
    assert (
        plan.run(
            replacement_query,
            replacement_blocks,
            replacement_table,
            replacement_requests,
            replacement_positions,
            out=replacement_output,
        )
        is replacement_output
    )
    assert calls["query_ptr"] == replacement_query.data_ptr()
    assert calls["output_ptr"] == replacement_output.data_ptr()

    valid_args = (
        replacement_query,
        replacement_blocks,
        replacement_table,
        replacement_requests,
        replacement_positions,
    )
    invalid_inputs = (
        ("query", (replacement_query[:-1], *valid_args[1:])),
        ("block_indices", (valid_args[0], replacement_blocks[:-1], *valid_args[2:])),
        (
            "block_table",
            (
                valid_args[0],
                valid_args[1],
                replacement_table[:-1],
                *valid_args[3:],
            ),
        ),
        (
            "token_to_request",
            (
                valid_args[0],
                valid_args[1],
                valid_args[2],
                replacement_requests[:-1],
                valid_args[4],
            ),
        ),
        (
            "query_positions",
            (
                valid_args[0],
                valid_args[1],
                valid_args[2],
                valid_args[3],
                replacement_positions[:-1],
            ),
        ),
        ("out", valid_args),
    )
    for name, args in invalid_inputs:
        invalid_out = replacement_output[:-1] if name == "out" else replacement_output
        with pytest.raises(ValueError, match=name):
            plan.run(*args, out=invalid_out)

    wrong_stride_query = torch.empty(
        (
            *replacement_query.shape[:-2],
            replacement_query.shape[-1],
            replacement_query.shape[-2],
        ),
        dtype=replacement_query.dtype,
        device=replacement_query.device,
    ).transpose(-1, -2)
    with pytest.raises(ValueError, match="strides"):
        plan.run(wrong_stride_query, *valid_args[1:], out=replacement_output)

    misaligned_query = torch.empty(
        replacement_query.numel() + 1,
        dtype=replacement_query.dtype,
        device=replacement_query.device,
    )[1:].view(replacement_query.shape)
    assert misaligned_query.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        plan.run(misaligned_query, *valid_args[1:], out=replacement_output)

    misaligned_output = torch.empty(
        replacement_output.numel() + 1,
        dtype=replacement_output.dtype,
        device=replacement_output.device,
    )[1:].view(replacement_output.shape)
    assert misaligned_output.data_ptr() % 16
    with pytest.raises(ValueError, match="16-byte aligned"):
        plan.run(*valid_args, out=misaligned_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_packed_q_token_kv_block_sparse_does_not_materialize_route_offsets_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashinfer.attention.prims_ts import decode as decode_module

    group_size = 4
    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    query = torch.zeros((8, 12, 256), dtype=torch.bfloat16, device="cuda")
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty_like(query)
    route_offsets = make_prims_ts_q_token_kv_block_sparse_qo_indptr(
        torch.tensor((0, 5, 8), dtype=torch.int32),
        8,
        group_size=group_size,
    )
    qo_indptr = route_offsets.to("cuda")
    workspace = torch.empty(
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            table,
            block_topk=blocks.shape[1],
            max_seq_len_kv=table.shape[1] * storage_page_size,
            qo_indptr=qo_indptr,
            max_seq_len_q=group_size,
        ),
        dtype=torch.uint8,
        device="cuda",
    )

    def fake_get_compiled_decode(
        _compile_spec: object,
    ) -> tuple[object, object]:
        return lambda *_args: None, None

    monkeypatch.setattr(
        decode_module,
        "_get_compiled_decode",
        fake_get_compiled_decode,
    )
    original_to = torch.Tensor.to
    original_cpu = torch.Tensor.cpu
    original_tolist = torch.Tensor.tolist

    def reject_qo_indptr_host_copy(
        tensor: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        if tensor is qo_indptr and (args == ("cpu",) or kwargs.get("device") == "cpu"):
            raise AssertionError(
                "prepared QToken-KvBlock-Sparse-Attention must not copy qo_indptr to the host"
            )
        return original_to(tensor, *args, **kwargs)

    def reject_qo_indptr_cpu(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is qo_indptr:
            raise AssertionError(
                "prepared QToken-KvBlock-Sparse-Attention must not copy qo_indptr to the host"
            )
        return original_cpu(tensor)

    def reject_qo_indptr_tolist(tensor: torch.Tensor) -> list[object]:
        if tensor is qo_indptr:
            raise AssertionError(
                "prepared QToken-KvBlock-Sparse-Attention must not materialize qo_indptr values"
            )
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "to", reject_qo_indptr_host_copy)
    monkeypatch.setattr(torch.Tensor, "cpu", reject_qo_indptr_cpu)
    monkeypatch.setattr(torch.Tensor, "tolist", reject_qo_indptr_tolist)
    prepare_prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, k_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size,
    )


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
def test_prepared_q_token_kv_block_sparse_plan_replays_without_counter_reset(
    group_size: int,
) -> None:
    blocks, table, requests, positions, storage_page_size = _make_case(
        group_size,
        512,
    )
    query_shape = _fixed_q_token_kv_block_sparse_shape(table.shape[0], group_size)
    torch.manual_seed(4117 + group_size)
    query = torch.randn(query_shape, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(query)
    layout = _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
        blocks.shape[0],
        blocks.shape[1],
        table.shape[1],
        storage_page_size,
        group_size,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=output.dtype,
        device="cuda",
    )
    workspace = torch.full(
        (layout.total_bytes,),
        0x55,
        dtype=torch.uint8,
        device="cuda",
    )
    plan = prepare_prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
        max_seq_len_kv=table.shape[1] * storage_page_size,
    )

    def run() -> None:
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=output,
        )

    run()
    torch.cuda.synchronize()
    reference = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
    if layout.uses_split_kv:
        assert not torch.count_nonzero(
            plan._attention_plan._workspace.split_kv_counter
        ).item()


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prepared_q_token_kv_block_sparse_bf16_q1_tileq8_split8_matches_oracle_and_graph() -> (
    None
):
    """Exercise the production low-grid Q1 policy and its PDL reducer."""

    from flashinfer.attention.prims_ts import decode as decode_module

    batch_size = 16
    num_qo_heads = 6
    num_kv_heads = 1
    head_dim = 256
    block_topk = 512
    sparse_block_size = 4
    storage_page_size = 16
    context_length = block_topk * sparse_block_size + 3
    storage_pages_per_request = (
        context_length + storage_page_size - 1
    ) // storage_page_size

    device_index = torch.cuda.current_device()
    decode_module._resolve_decode_launch_spec.cache_clear()
    spec = decode_module._resolve_decode_launch_spec(
        device_index,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        sparse_block_size,
        context_length,
        1,
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "HND",
        "causal",
        False,
        -1,
        storage_page_size,
        True,
        True,
    )
    assert not spec.config.use_variable_seqlens_q
    assert spec.config.tile_size_q == 8
    assert not spec.config.use_keeps_mma_ab
    assert spec.config.tile_size_kv == 128
    assert spec.config.splits_kv == spec.config.max_splits_kv == 8
    assert spec.config.use_separate_reduction_kernel
    assert spec.config.use_parallel_separate_reduction_pdl
    assert spec.config.use_pdl

    standalone_spec = decode_module._resolve_decode_launch_spec(
        device_index,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        sparse_block_size,
        context_length,
        1,
        "bfloat16",
        "bfloat16",
        "bfloat16",
        "HND",
        "causal",
        False,
        -1,
        storage_page_size,
        True,
        False,
    )
    assert standalone_spec.config.uses_q_token_kv_block_sparse_page_route
    assert not standalone_spec.config.use_pdl

    torch.manual_seed(8117)
    query = (
        torch.randn(
            batch_size,
            1,
            1,
            num_qo_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.25
    )
    num_storage_pages = batch_size * storage_pages_per_request
    k_cache = (
        torch.randn(
            num_storage_pages,
            num_kv_heads,
            storage_page_size,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        * 0.25
    )
    v_cache = torch.randn_like(k_cache) * 0.25
    block_table = torch.arange(
        num_storage_pages,
        dtype=torch.int32,
        device="cuda",
    ).reshape(batch_size, storage_pages_per_request)
    block_indices = torch.arange(
        block_topk,
        dtype=torch.int32,
        device="cuda",
    ).repeat(batch_size, 1)
    token_to_request = torch.arange(
        batch_size,
        dtype=torch.int32,
        device="cuda",
    )
    query_positions = torch.full(
        (batch_size,),
        context_length - 1,
        dtype=torch.int64,
        device="cuda",
    )
    output = torch.empty_like(query)
    workspace = torch.empty(
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            block_table,
            block_topk=block_topk,
            max_seq_len_kv=block_table.shape[1] * storage_page_size,
            out_dtype=output.dtype,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    plan = prepare_prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        workspace,
        out=output,
        max_seq_len_kv=block_table.shape[1] * storage_page_size,
    )

    plan.run(
        query,
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        out=output,
    )
    torch.cuda.synchronize()

    def reference_for_context(visible_tokens: int) -> torch.Tensor:
        reference = torch.empty_like(query, dtype=torch.float32)
        scale = head_dim**-0.5
        for request in range(batch_size):
            page_ids = block_table[request].long()
            k = k_cache[page_ids, 0].reshape(-1, head_dim)[:visible_tokens].float()
            v = v_cache[page_ids, 0].reshape(-1, head_dim)[:visible_tokens].float()
            q = query[request, 0, 0].float()
            reference[request, 0, 0] = torch.softmax(q @ k.T * scale, dim=-1) @ v
        return reference

    reference = reference_for_context(context_length)
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan.run(
            query,
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            out=output,
        )
    # Contract the runtime union from all five configured splits to two.
    # The three pruned CTAs must acquire metadata, tear down TMEM, release the
    # reducer, and retire without entering TaskManager. Then expand to five
    # again to prove the captured graph does not retain stale split state.
    short_context_length = 259
    query_positions.fill_(short_context_length - 1)
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    short_reference = reference_for_context(short_context_length)
    torch.testing.assert_close(output.float(), short_reference, rtol=1e-2, atol=1e-2)
    assert not torch.count_nonzero(
        plan._attention_plan._workspace.split_kv_counter
    ).item()

    query_positions.fill_(context_length - 1)
    output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", (1, 2))
def test_prepared_q_token_kv_block_sparse_fp8_q1_small_cuda_graph(rows: int) -> None:
    """Cover tiny fixed Q1 decode graph buckets."""

    block_topk = 512
    num_qo_heads = 12
    head_dim = 256
    storage_page_size = 1600
    blocks, table, requests, positions, _ = _make_case(1, block_topk, storage_page_size)
    blocks = blocks[:rows].contiguous()
    requests = requests[:rows].contiguous()
    positions = positions[:rows].contiguous()
    table = table[:rows].contiguous()

    torch.manual_seed(4189 + rows)
    query = (
        torch.randn(
            _fixed_q_token_kv_block_sparse_shape(rows, 1, num_qo_heads, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    k_cache = (
        torch.randn(
            int(table.max().item()) + 1,
            1,
            storage_page_size,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    v_cache = (
        torch.randn_like(k_cache, dtype=torch.bfloat16)
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    output = torch.empty(query.shape, dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty(
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            table,
            block_topk=block_topk,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            out_dtype=output.dtype,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    plan = prepare_prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
        max_seq_len_kv=table.shape[1] * storage_page_size,
    )

    def run() -> None:
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=output,
        )

    run()
    torch.cuda.synchronize()
    reference = output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (4, 5))
def test_prepared_q_token_kv_block_sparse_fp8_grouped_tp4_bs256_cuda_graph(
    group_size: int,
) -> None:
    """Match TP4 vLLM MTP3/MTP4 graph geometries with a prepared QToken-KvBlock-Sparse-Attention plan."""

    batch_size = 256
    block_topk = 512
    num_qo_heads = 6
    head_dim = 256
    storage_page_size = 1600
    max_storage_pages = 88
    context_length = 8192
    rows = batch_size * group_size
    max_seq_len = group_size * (block_topk + 1) * 4

    torch.manual_seed(9425)
    query = (
        torch.randn(
            batch_size,
            1,
            group_size,
            num_qo_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    raw_query = _as_lower_level_decode_view(query)
    k_cache = (
        torch.randn(
            max_storage_pages,
            1,
            storage_page_size,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    v_cache = (
        torch.randn_like(k_cache, dtype=torch.bfloat16)
        .mul_(0.25)
        .to(torch.float8_e4m3fn)
    )
    block_table = torch.arange(
        max_storage_pages, dtype=torch.int32, device="cuda"
    ).repeat(batch_size, 1)
    token_to_request = torch.arange(
        batch_size, dtype=torch.int32, device="cuda"
    ).repeat_interleave(group_size)
    query_positions = (
        torch.arange(
            context_length - group_size,
            context_length,
            dtype=torch.int64,
            device="cuda",
        )
        .repeat(batch_size)
        .contiguous()
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    block_indices = torch.stack(
        [torch.roll(base_blocks, row % 7) for row in range(rows)]
    ).contiguous()

    raw_metadata_shapes = get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
        rows, block_topk, group_size
    )
    raw_page_indices, raw_page_memberships, raw_seq_lens = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in raw_metadata_shapes
    )
    build_prims_ts_q_token_kv_block_sparse_metadata(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_storage_pages * storage_page_size,
        out=(raw_page_indices, raw_page_memberships, raw_seq_lens),
    )
    raw_output = torch.empty(raw_query.shape, dtype=torch.bfloat16, device="cuda")
    _run_private_q_token_kv_block_sparse_decode(
        raw_query,
        k_cache,
        v_cache,
        raw_page_indices,
        raw_page_memberships,
        raw_seq_lens,
        max_seq_len,
        group_size,
        raw_output,
    )

    combined_bytes = get_prims_ts_q_token_kv_block_sparse_workspace_size(
        query,
        k_cache,
        block_table,
        block_topk=block_topk,
        max_seq_len_kv=max_storage_pages * storage_page_size,
        out_dtype=torch.bfloat16,
    )
    combined_workspace = torch.empty(combined_bytes, dtype=torch.uint8, device="cuda")
    prepared_output = torch.empty(query.shape, dtype=torch.bfloat16, device="cuda")
    plan = prepare_prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        combined_workspace,
        out=prepared_output,
        max_seq_len_kv=max_storage_pages * storage_page_size,
    )
    assert plan._metadata_plan.release_attention_pdl is True

    def run() -> None:
        plan.run(
            query,
            block_indices,
            block_table,
            token_to_request,
            query_positions,
            out=prepared_output,
        )

    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        _as_lower_level_decode_view(prepared_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        _as_lower_level_decode_view(prepared_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )

    # vLLM captures a fixed 1,024-token graph bucket even when only a short
    # live prefix belongs to requests. Exercise the same static graph with
    # almost every grouped route marked inert; metadata must publish the invalid-row
    # sentinel without letting attention dereference it.
    active_groups = 3
    token_to_request[active_groups * group_size :].zero_()
    query_positions[active_groups * group_size :].fill_(-1)
    raw_output.fill_(torch.nan)
    prepared_output.fill_(torch.nan)
    build_prims_ts_q_token_kv_block_sparse_metadata(
        block_indices,
        block_table,
        token_to_request,
        query_positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=max_storage_pages * storage_page_size,
        out=(raw_page_indices, raw_page_memberships, raw_seq_lens),
    )
    _run_private_q_token_kv_block_sparse_decode(
        raw_query,
        k_cache,
        v_cache,
        raw_page_indices,
        raw_page_memberships,
        raw_seq_lens,
        max_seq_len,
        group_size,
        raw_output,
    )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        plan._metadata_plan.seq_lens[active_groups:],
        torch.ones_like(plan._metadata_plan.seq_lens[active_groups:]),
    )
    assert torch.isfinite(prepared_output).all()
    torch.testing.assert_close(
        _as_lower_level_decode_view(prepared_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q_token_kv_block_sparse_attention_unified_workspace_matches_two_step_cuda_graph() -> (
    None
):
    group_size = 4
    block_topk = 512
    blocks, table, requests, positions, storage_page_size = _make_case(
        group_size,
        block_topk,
    )
    num_routes = table.shape[0]
    num_qo_heads = 12
    head_dim = 256
    torch.manual_seed(9231)
    query = torch.randn(
        _fixed_q_token_kv_block_sparse_shape(
            num_routes, group_size, num_qo_heads, head_dim
        ),
        dtype=torch.bfloat16,
        device="cuda",
    )
    raw_query = _as_lower_level_decode_view(query)
    num_physical_pages = int(table.max().item()) + 1
    k_cache = torch.randn(
        num_physical_pages,
        1,
        storage_page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)

    unified_bytes = get_prims_ts_q_token_kv_block_sparse_workspace_size(
        query,
        k_cache,
        table,
        block_topk=block_topk,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        out_dtype=torch.bfloat16,
    )
    unified_workspace = torch.full(
        (unified_bytes,),
        0x55,
        dtype=torch.uint8,
        device="cuda",
    )
    unified_layout = _get_prims_ts_q_token_kv_block_sparse_workspace_layout(
        blocks.shape[0],
        block_topk,
        table.shape[1],
        storage_page_size,
        group_size,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=1,
        head_dim=head_dim,
        q_dtype=query.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=torch.bfloat16,
        device="cuda",
    )
    unified_views = unified_layout.bind(unified_workspace)
    unified_output = torch.empty_like(query)
    prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        unified_workspace,
        out=unified_output,
        max_seq_len_kv=table.shape[1] * storage_page_size,
    )

    metadata_shapes = get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
        blocks.shape[0],
        block_topk,
        group_size,
    )
    raw_metadata = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in metadata_shapes
    )
    build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        out=raw_metadata,
    )
    max_seq_len = group_size * (block_topk + 1) * 4
    raw_output = torch.empty_like(raw_query)
    _run_private_q_token_kv_block_sparse_decode(
        raw_query,
        k_cache,
        v_cache,
        raw_metadata[0],
        raw_metadata[1],
        raw_metadata[2],
        max_seq_len,
        group_size,
        raw_output,
    )
    torch.cuda.synchronize()
    _assert_metadata_matches(
        (
            unified_views.q_token_kv_block_sparse_page_indices,
            unified_views.q_token_kv_block_sparse_page_memberships,
            unified_views.seq_lens,
        ),
        raw_metadata,
    )
    torch.testing.assert_close(
        _as_lower_level_decode_view(unified_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )

    graph_output = torch.empty_like(query)
    graph_wrapper = QTokenKvBlockSparsePagedTSWrapper()
    graph_wrapper.plan(
        num_routes,
        group_size,
        num_qo_heads,
        1,
        head_dim,
        4,
        storage_page_size,
        block_topk,
        table.shape[1] * storage_page_size,
        device=query.device,
        workspace_buffer=unified_workspace,
        q_data_type=query.dtype,
        kv_data_type=k_cache.dtype,
        o_data_type=graph_output.dtype,
    )
    graph_wrapper.run(
        query,
        (k_cache, v_cache),
        table,
        blocks,
        requests,
        positions,
        out=graph_output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_wrapper.run(
            query,
            (k_cache, v_cache),
            table,
            blocks,
            requests,
            positions,
            out=graph_output,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        _as_lower_level_decode_view(graph_output),
        raw_output,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
@pytest.mark.parametrize("block_topk", (8, 512))
@pytest.mark.parametrize("storage_page_size", (4, 16, 256))
def test_q_token_kv_block_sparse_metadata_matches_reference(
    group_size: int,
    block_topk: int,
    storage_page_size: int,
) -> None:
    blocks, table, requests, positions, _ = _make_case(
        group_size, block_topk, storage_page_size
    )
    output_shapes = get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
        blocks.shape[0], block_topk, group_size
    )
    outputs = tuple(
        torch.full(shape, -7, dtype=torch.int32, device="cuda")
        for shape in output_shapes
    )
    actual = build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        out=outputs,
    )
    expected = _reference(
        blocks, table, requests, positions, storage_page_size, group_size
    )
    _assert_metadata_matches(actual, expected)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_grouped_q_token_kv_block_sparse_physical_sparse_pages_match_torch_reference() -> (
    None
):
    """Decode page-four routes without observing poisoned metadata padding."""

    group_size = 4
    block_topk = 8
    num_qo_heads = 12
    head_dim = 256
    positions = torch.arange(31, 35, dtype=torch.int64, device="cuda")
    requests = torch.zeros(group_size, dtype=torch.int32, device="cuda")
    blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda").repeat(
        group_size, 1
    )
    table = torch.arange(64, dtype=torch.int32, device="cuda").unsqueeze(0)
    torch.manual_seed(9527)
    query = torch.randn(
        1,
        1,
        group_size,
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.randn(64, 1, 4, head_dim, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    output = torch.empty_like(query)
    # Poison metadata padding so the consumer cannot accidentally interpret
    # unspecified bytes beyond the live membership prefix as valid pages.
    workspace = torch.full(
        (
            get_prims_ts_q_token_kv_block_sparse_workspace_size(
                query,
                k_cache,
                table,
                block_topk=block_topk,
                max_seq_len_kv=table.shape[1] * k_cache.shape[2],
            ),
        ),
        0xFF,
        dtype=torch.uint8,
        device="cuda",
    )
    prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        workspace,
        out=output,
        max_seq_len_kv=table.shape[1] * k_cache.shape[2],
    )
    reference = torch.empty_like(output, dtype=torch.float32)
    for query_idx, position in enumerate(positions.tolist()):
        visible_tokens = position + 1
        full_pages = min(visible_tokens // 4, block_topk)
        token_chunks = [k_cache[page, 0].float() for page in range(full_pages)]
        value_chunks = [v_cache[page, 0].float() for page in range(full_pages)]
        tail_tokens = visible_tokens % 4
        if tail_tokens:
            tail_page = visible_tokens // 4
            token_chunks.append(k_cache[tail_page, 0, :tail_tokens].float())
            value_chunks.append(v_cache[tail_page, 0, :tail_tokens].float())
        keys = torch.cat(token_chunks).unsqueeze(1).expand(-1, num_qo_heads, -1)
        values = torch.cat(value_chunks).unsqueeze(1).expand(-1, num_qo_heads, -1)
        scores = (
            torch.einsum("hd,thd->ht", query[0, 0, query_idx].float(), keys)
            / head_dim**0.5
        )
        probability = torch.softmax(scores, dim=-1)
        reference[0, 0, query_idx] = torch.einsum("ht,thd->hd", probability, values)

    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_q_token_kv_block_sparse_metadata_handles_partial_request_groups() -> (
    None
):
    group_size = 4
    block_topk = 8
    storage_page_size = 16
    request_lengths = (5, 3)
    rows = sum(request_lengths)
    qo_indptr = torch.tensor([0, 4, 5, 8], dtype=torch.int32, device="cuda")
    requests = torch.tensor([0] * 5 + [1] * 3, dtype=torch.int32, device="cuda")
    positions = torch.tensor(
        [31 + index for index in range(5)] + [47 + index for index in range(3)],
        dtype=torch.int64,
        device="cuda",
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [torch.roll(base_blocks, row % 3) for row in range(rows)]
    ).contiguous()
    table = torch.arange(2 * 64, dtype=torch.int32, device="cuda").reshape(2, 64)
    groups = qo_indptr.numel() - 1
    output_shapes = get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
        rows,
        block_topk,
        group_size,
        num_query_groups=groups,
    )
    outputs = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda") for shape in output_shapes
    )
    actual = build_prims_ts_q_token_kv_block_sparse_metadata(
        blocks,
        table,
        requests,
        positions,
        group_size=group_size,
        storage_page_size=storage_page_size,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        qo_indptr=qo_indptr,
        out=outputs,
    )
    expected = _reference(
        blocks,
        table,
        requests,
        positions,
        storage_page_size,
        group_size,
        qo_indptr,
    )
    _assert_metadata_matches(actual, expected)


@pytest.mark.arch_blackwell
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (1, 2, 4, 5))
def test_packed_q_token_kv_block_sparse_groups_match_packed_q1_attention(
    group_size: int,
) -> None:
    block_topk = 512
    storage_page_size = 16
    rows = 8
    num_query_heads = 12
    head_dim = 256
    route_offsets = [0]
    for request_begin, request_end in ((0, 5), (5, 8)):
        route_begin = request_begin
        while route_begin < request_end:
            route_begin = min(route_begin + group_size, request_end)
            route_offsets.append(route_begin)
    qo_indptr = torch.tensor(route_offsets, dtype=torch.int32, device="cuda")
    q1_qo_indptr = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    requests = torch.tensor([0] * 5 + [1] * 3, dtype=torch.int32, device="cuda")
    positions = torch.tensor(
        [2047 + index for index in range(5)] + [2047 + index for index in range(3)],
        dtype=torch.int64,
        device="cuda",
    )
    base_blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda")
    blocks = torch.stack(
        [torch.roll(base_blocks, row % 3) for row in range(rows)]
    ).contiguous()
    table = torch.arange(2 * 129, dtype=torch.int32, device="cuda").reshape(2, 129)
    if group_size == 5:
        from flashinfer.attention.prims_ts import decode as decode_module

        decode_module._resolve_decode_launch_spec.cache_clear()
        split_spec = decode_module._resolve_decode_launch_spec(
            torch.cuda.current_device(),
            qo_indptr.numel() - 1,
            num_query_heads,
            1,
            head_dim,
            4,
            group_size * (block_topk + 1) * 4,
            group_size,
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "HND",
            "causal",
            True,
            -1,
            storage_page_size,
            True,
            True,
        )
        # Two packed route rows fit in one wave, so this case qualifies the
        # metadata acquire without introducing a split-reduction dependency.
        assert split_spec.config.splits_kv == 1
        assert split_spec.config.use_pdl
    # Zero Q/K makes softmax exactly uniform. Comparing the grouped result to
    # Q1 then checks route membership and packed output placement without
    # conflating them with different accumulation recurrences.
    query = torch.zeros(
        rows,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.zeros(
        int(table.max().item()) + 1,
        1,
        storage_page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = (
        torch.arange(k_cache.shape[0], dtype=torch.float32, device="cuda")
        .div(k_cache.shape[0])
        .to(query.dtype)
        .view(-1, 1, 1, 1)
        .expand_as(k_cache)
        .contiguous()
    )
    packed_output = torch.empty_like(query)
    q1_output = torch.empty_like(query)
    packed_workspace = torch.empty(
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            table,
            block_topk=block_topk,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            out_dtype=query.dtype,
            qo_indptr=qo_indptr,
            max_seq_len_q=group_size,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    q1_workspace = torch.empty(
        get_prims_ts_q_token_kv_block_sparse_workspace_size(
            query,
            k_cache,
            table,
            block_topk=block_topk,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            out_dtype=query.dtype,
            qo_indptr=q1_qo_indptr,
            max_seq_len_q=1,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        packed_workspace,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size,
        out=packed_output,
    )
    prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        q1_workspace,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        qo_indptr=q1_qo_indptr,
        max_seq_len_q=1,
        out=q1_output,
    )
    prepared_output = torch.empty_like(query)
    prepared_workspace = torch.empty_like(packed_workspace)
    plan = prepare_prims_ts_q_token_kv_block_sparse_attention(
        query,
        (k_cache, v_cache),
        blocks,
        table,
        requests,
        positions,
        prepared_workspace,
        max_seq_len_kv=table.shape[1] * storage_page_size,
        qo_indptr=qo_indptr,
        max_seq_len_q=group_size,
        out=prepared_output,
    )

    def run_prepared() -> None:
        plan.run(
            query,
            blocks,
            table,
            requests,
            positions,
            out=prepared_output,
        )

    run_prepared()
    torch.cuda.synchronize()
    torch.testing.assert_close(prepared_output, packed_output, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_prepared()
    prepared_output.fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(prepared_output, packed_output, rtol=0, atol=0)

    if group_size == 5:
        original_positions = positions.clone()
        original_q1_output = q1_output.clone()
        positions.copy_(
            torch.tensor(
                [258 + index for index in range(5)]
                + [386 + index for index in range(3)],
                dtype=positions.dtype,
                device=positions.device,
            )
        )
        prims_ts_q_token_kv_block_sparse_attention(
            query,
            (k_cache, v_cache),
            blocks,
            table,
            requests,
            positions,
            q1_workspace,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            qo_indptr=q1_qo_indptr,
            max_seq_len_q=1,
            out=q1_output,
        )
        prepared_output.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            prepared_output,
            q1_output,
            rtol=2e-2,
            atol=2e-2,
        )
        positions.copy_(original_positions)
        prepared_output.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(prepared_output, packed_output, rtol=0, atol=0)
        q1_output.copy_(original_q1_output)

    for row, request in enumerate(requests.tolist()):
        selected_v = v_cache[table[request, :128].long()].reshape(-1, head_dim)
        tail_tokens = (int(positions[row].item()) + 1) % 4
        if tail_tokens:
            tail_v = v_cache[table[request, 128].long(), 0, :tail_tokens]
            selected_v = torch.cat((selected_v, tail_v), dim=0)
        expected = (
            selected_v.float()
            .mean(dim=0)
            .to(query.dtype)
            .expand(num_query_heads, head_dim)
        )
        torch.testing.assert_close(
            packed_output[row],
            expected,
            rtol=2e-2,
            atol=2e-2,
            msg=lambda message,
            row=row: f"packed QToken-KvBlock-Sparse-Attention row {row}: {message}",
        )
        torch.testing.assert_close(
            q1_output[row],
            expected,
            rtol=2e-2,
            atol=2e-2,
            msg=lambda message,
            row=row: f"Q1 QToken-KvBlock-Sparse-Attention row {row}: {message}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_q1_graph_replay_overwrites_newly_live_page_indices() -> None:
    group_size = 1
    block_topk = 512
    blocks, table, requests, positions, storage_page_size = _make_case(
        group_size, block_topk
    )
    output_shapes = get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
        blocks.shape[0], block_topk, group_size
    )
    outputs = tuple(
        torch.full(shape, 0x12345, dtype=torch.int32, device="cuda")
        for shape in output_shapes
    )

    def run() -> None:
        build_prims_ts_q_token_kv_block_sparse_metadata(
            blocks,
            table,
            requests,
            positions,
            group_size=group_size,
            storage_page_size=storage_page_size,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            out=outputs,
        )

    # Capture with a short live prefix, then grow to the maximum 513-page Q1
    # route. Dead suffix values are intentionally unspecified; newly-live
    # values must always be overwritten on replay.
    positions.fill_(31)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    outputs[0].fill_(0x12345)
    positions.fill_(4 * block_topk + 2)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        blocks, table, requests, positions, storage_page_size, group_size
    )
    assert outputs[1].shape == (blocks.shape[0], 0)
    _assert_metadata_matches(outputs, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("group_size", (4, 5))
def test_grouped_metadata_cuda_graph_reloads_inputs(group_size: int) -> None:
    blocks, table, requests, positions, storage_page_size = _make_case(group_size, 8)
    long_positions = positions.clone()
    num_groups = blocks.shape[0] // group_size
    short_positions = (
        torch.arange(group_size, dtype=torch.int64, device="cuda")
        .repeat(num_groups)
        .add_(7)
    )
    positions.copy_(short_positions)
    output_shapes = get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
        blocks.shape[0], blocks.shape[1], group_size
    )
    outputs = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda") for shape in output_shapes
    )

    def run() -> None:
        build_prims_ts_q_token_kv_block_sparse_metadata(
            blocks,
            table,
            requests,
            positions,
            group_size=group_size,
            storage_page_size=storage_page_size,
            max_seq_len_kv=table.shape[1] * storage_page_size,
            out=outputs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    outputs[0].fill_(0x12345)
    outputs[1].fill_(0x12345)
    positions.copy_(long_positions)
    blocks.add_(16)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        blocks, table, requests, positions, storage_page_size, group_size
    )
    _assert_metadata_matches(outputs, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("group_size", "max_seq_len_kv", "block_topk", "long_start_position"),
    (
        # A smaller top-k route.
        (4, 1024, 64, 255),
        # A maximum top-k Q5 route.
        (5, 8192, 512, 2047),
    ),
)
def test_grouped_metadata_cuda_graph_reloads_positions(
    group_size: int,
    max_seq_len_kv: int,
    block_topk: int,
    long_start_position: int,
) -> None:
    """The sort-union kernel reloads shorter and longer graph inputs."""

    storage_page_size = 16
    blocks = torch.arange(block_topk, dtype=torch.int32, device="cuda").repeat(
        group_size, 1
    )
    table_width = (max_seq_len_kv + storage_page_size - 1) // storage_page_size
    table = torch.arange(table_width, dtype=torch.int32, device="cuda").unsqueeze(0)
    requests = torch.zeros(group_size, dtype=torch.int32, device="cuda")
    long_positions = torch.arange(
        long_start_position,
        long_start_position + group_size,
        dtype=torch.int64,
        device="cuda",
    )
    short_positions = torch.arange(
        31,
        31 + group_size,
        dtype=torch.int64,
        device="cuda",
    )
    positions = long_positions.clone()
    outputs = tuple(
        torch.empty(shape, dtype=torch.int32, device="cuda")
        for shape in get_prims_ts_q_token_kv_block_sparse_metadata_output_shapes(
            blocks.shape[0],
            block_topk,
            group_size,
        )
    )

    def run() -> None:
        build_prims_ts_q_token_kv_block_sparse_metadata(
            blocks,
            table,
            requests,
            positions,
            group_size=group_size,
            storage_page_size=storage_page_size,
            max_seq_len_kv=max_seq_len_kv,
            out=outputs,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # Poison outputs, then contract to a much shorter causal prefix.
    positions.copy_(short_positions)
    for output in outputs:
        output.fill_(0x12345)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        blocks, table, requests, positions, storage_page_size, group_size
    )
    _assert_metadata_matches(outputs, expected)

    # Growing the same captured plan must also reload the longer positions.
    positions.copy_(long_positions)
    for output in outputs:
        output.fill_(0x12345)
    graph.replay()
    torch.cuda.synchronize()
    expected = _reference(
        blocks, table, requests, positions, storage_page_size, group_size
    )
    _assert_metadata_matches(outputs, expected)
