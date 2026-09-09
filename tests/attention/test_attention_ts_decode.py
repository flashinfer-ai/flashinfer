# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact public-interface acceptance coverage for PrimTS FMHA decode."""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
from typing import Optional, Sequence, cast
import warnings

import pytest
import torch

pytest.importorskip(
    "cutlass",
    minversion="4.7.0",
    reason="PrimTS attention tests require nvidia-cutlass-dsl==4.7.0",
)

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float16, Float8E4M3FN, Int32
from cutlass.cute.runtime import make_ptr
from cutlass import utils as cutlass_utils
from cutlass.experimental.task_scheduling.memory import SmemAllocation

from flashinfer.attention.prims_ts import (
    BatchDecodePagedTSWrapper,
    batch_decode_with_paged_kv_cache,
)
from flashinfer.attention.prims_ts.decode import (
    _DECODE_MAX_KV_LEN,
    _DECODE_MAX_KV_TILE_SIZE,
    _DecodePlanState,
    _DecodeRuntime,
    _make_decode_workspace_layout,
    _planned_kv_domain_has_unpaired_tail,
    _validate_prims_ts_q_token_kv_block_sparse_group_layout,
    _validate_q_token_kv_block_sparse_route_offsets_cpu,
    _validate_decode_query_head_extent,
    _validate_decode_output_aliasing,
    _validate_decode_policy_kv_tile_size,
    _validate_decode_run_metadata_values,
    _validate_block_table_metadata,
    _validate_head_geometry,
    _validate_max_kv_len,
    _validate_storage_page_size,
)
from flashinfer.attention.prims_ts._tensor_aliasing import (
    _validate_tensor_does_not_overlap_inputs,
)
from flashinfer.attention.prims_ts.split_kv_mode_policy import select_split_kv_modes
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
    FmhaDecodeConfig,
    make_decode_config,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_constants import (
    Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS,
    Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIPS_PER_WORD,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_kernel import (
    _build_decode_gen_schedule,
    _decode_min_blocks_per_mp,
    build_decode_task_manager,
)
from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.helpers_kv_tile_idx import (
    _runtime_last_valid_page_idx,
    _runtime_total_kv_tiles,
)
from flashinfer.decode import (
    get_prims_ts_batch_decode_workspace_size,
    make_q_token_kv_block_sparse_qo_indptr as make_prims_ts_q_token_kv_block_sparse_qo_indptr,
    prepare_prims_ts_batch_decode_with_kv_cache,
    prims_ts_batch_decode_with_kv_cache,
    suggest_q_token_kv_block_sparse_group_size as suggest_prims_ts_q_token_kv_block_sparse_group_size,
    validate_q_token_kv_block_sparse_group_size as validate_prims_ts_q_token_kv_block_sparse_group_size,
)
from flashinfer.utils import is_sm100a_supported


@pytest.mark.parametrize(
    (
        "query_lengths",
        "num_query_tokens",
        "num_qo_heads",
        "num_kv_heads",
        "group_size",
        "expected_group_size",
    ),
    [
        pytest.param([8192], 8192, 24, 2, 4, 4, id="tp1-prefill-q4"),
        pytest.param([4] * 8, 32, 24, 2, 4, 4, id="tp1-small-q4"),
        pytest.param([4] * 8, 32, 12, 1, 4, 4, id="tp2-small-q4"),
        pytest.param([4] * 64, 256, 6, 1, 4, 4, id="tp4-q4"),
        pytest.param([4] * 64 + [0], 256, 24, 2, 4, 4, id="zero-length"),
        pytest.param([2] * 128, 256, 6, 1, 2, 2, id="mtp-q2"),
        pytest.param([4] * 38, 152, 12, 1, 2, 2, id="caller-q2"),
        pytest.param([4] * 64, 256, 20, 1, 4, None, id="tileq64-cap"),
        pytest.param([5] * 32, 160, 12, 1, 5, 5, id="mtp4-q5-bs32"),
        pytest.param([5] * 16, 80, 12, 1, 5, 5, id="mtp4-q5-bs16"),
        pytest.param([5] * 64, 320, 16, 1, 5, None, id="mtp4-tileq64-cap"),
        pytest.param([1] * 1024, 1024, 12, 1, 4, 4, id="sq1-boundaries"),
        pytest.param([4, 3], 7, 12, 1, 4, 4, id="variable-length"),
        pytest.param([4] * 19 + [0], 80, 24, 2, 4, 4, id="graph-padding"),
        pytest.param([4] * 19 + [0], 79, 24, 2, 4, 4, id="unaligned-padding"),
    ],
)
def test_prims_ts_q_token_kv_block_sparse_fixed_group_validation(
    query_lengths: list[int],
    num_query_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    group_size: int,
    expected_group_size: int | None,
) -> None:
    query_start_loc = [0]
    for query_length in query_lengths:
        query_start_loc.append(query_start_loc[-1] + query_length)
    args = (
        group_size,
        torch.tensor(query_start_loc, dtype=torch.int32),
        num_query_tokens,
        num_qo_heads,
        num_kv_heads,
    )
    if expected_group_size is None:
        with pytest.raises(ValueError):
            _validate_prims_ts_q_token_kv_block_sparse_group_layout(*args)
    else:
        assert (
            _validate_prims_ts_q_token_kv_block_sparse_group_layout(*args)
            == expected_group_size
        )


def test_prims_ts_q_token_kv_block_sparse_fixed_group_validation_is_public() -> None:
    num_query_tokens = 32
    query_start_loc_cpu = torch.tensor(
        [0, num_query_tokens],
        dtype=torch.int32,
    )
    assert (
        validate_prims_ts_q_token_kv_block_sparse_group_size(
            query_start_loc_cpu,
            num_query_tokens,
            12,
            1,
            group_size=4,
        )
        == 4
    )
    assert (
        validate_prims_ts_q_token_kv_block_sparse_group_size(
            None,
            num_query_tokens,
            12,
            1,
            group_size=1,
        )
        == 1
    )
    with pytest.raises(ValueError, match="query_start_loc"):
        validate_prims_ts_q_token_kv_block_sparse_group_size(
            None,
            num_query_tokens,
            12,
            1,
            group_size=4,
        )


@pytest.mark.parametrize(
    (
        "batch_size",
        "seq_len_q",
        "selected_seq_len_kv",
        "num_qo_heads",
        "num_kv_heads",
        "multi_processor_count",
        "expected_group_size",
    ),
    (
        # One MTP3 request cannot fill a wave even after useful split-KV, so
        # expose four independent Q1 routes instead of building one Q4 union.
        (1, 4, 2051, 12, 1, 152, 1),
        # Eight MTP3 requests have enough K/V work to fill a wave as Q4.
        (8, 4, 2051, 12, 1, 152, 4),
        # Q5 fits TileQ64 for Qwen's twelve query heads per K/V head.
        (16, 5, 2051, 12, 1, 152, 5),
        # Divisibility is not required: two Q2 routes cover three live rows.
        (64, 3, 2051, 12, 1, 152, 2),
        # A long packed-prefill request supplies independent routes directly.
        (1, 8192, 2051, 12, 1, 152, 5),
        # Head capacity, rather than SQ, caps the largest legal group.
        (8, 5, 2051, 16, 1, 152, 4),
    ),
)
def test_suggest_prims_ts_q_token_kv_block_sparse_group_size(
    batch_size: int,
    seq_len_q: int,
    selected_seq_len_kv: int,
    num_qo_heads: int,
    num_kv_heads: int,
    multi_processor_count: int,
    expected_group_size: int,
) -> None:
    assert (
        suggest_prims_ts_q_token_kv_block_sparse_group_size(
            batch_size,
            seq_len_q,
            selected_seq_len_kv,
            num_qo_heads,
            num_kv_heads,
            multi_processor_count,
        )
        == expected_group_size
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    (
        ("batch_size", 0),
        ("seq_len_q", True),
        ("selected_seq_len_kv", -1),
        ("multi_processor_count", 0),
    ),
)
def test_suggest_prims_ts_q_token_kv_block_sparse_group_size_rejects_invalid_extent(
    argument: str,
    value: object,
) -> None:
    arguments: dict[str, object] = {
        "batch_size": 8,
        "seq_len_q": 4,
        "selected_seq_len_kv": 2051,
        "num_qo_heads": 12,
        "num_kv_heads": 1,
        "multi_processor_count": 152,
    }
    arguments[argument] = value
    with pytest.raises((TypeError, ValueError)):
        suggest_prims_ts_q_token_kv_block_sparse_group_size(**arguments)


@pytest.mark.parametrize(
    ("query_starts", "num_query_tokens", "group_size", "expected"),
    [
        ([0, 4, 7], 7, 4, [0, 4, 7]),
        ([0, 7, 12], 12, 4, [0, 4, 7, 11, 12]),
        ([0, 4, 7], 10, 4, [0, 4, 7, 10]),
        ([0, 0, 3], 3, 4, [0, 3]),
        ([0, 5], 5, 5, [0, 5]),
        ([0, 2, 3], 3, 1, [0, 1, 2, 3]),
    ],
)
def test_make_prims_ts_q_token_kv_block_sparse_qo_indptr_keeps_request_boundaries(
    query_starts: list[int],
    num_query_tokens: int,
    group_size: int,
    expected: list[int],
) -> None:
    actual = make_prims_ts_q_token_kv_block_sparse_qo_indptr(
        torch.tensor(query_starts, dtype=torch.int32),
        num_query_tokens,
        group_size=group_size,
    )
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int32))


@pytest.mark.parametrize(
    ("route_offsets", "num_query_tokens", "group_size"),
    [
        ([1, 4], 4, 4),
        ([0, 4], 5, 4),
        ([0, 0, 4], 4, 4),
        ([0, 5], 5, 4),
    ],
)
def test_q_token_kv_block_sparse_cpu_route_offsets_must_be_safe_before_upload(
    route_offsets: list[int],
    num_query_tokens: int,
    group_size: int,
) -> None:
    with pytest.raises(ValueError):
        _validate_q_token_kv_block_sparse_route_offsets_cpu(
            route_offsets,
            num_query_tokens=num_query_tokens,
            group_size=group_size,
        )


_REQUIRES_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 0)
    or not is_sm100a_supported(torch.device("cuda")),
    reason=(
        "PrimTS FMHA decode is signoff-qualified on SM100; "
        "SM103/B300 and GB300 qualification is pending"
    ),
)

_REQUIRES_PAGE4_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
    or not is_sm100a_supported(torch.device("cuda")),
    reason="PrimTS page-4 decode requires an SM100a or SM103a GPU",
)

_REQUIRES_BLACKWELL_PRIMTS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
    or not is_sm100a_supported(torch.device("cuda")),
    reason="PrimTS grouped decode requires an SM100a or SM103a GPU",
)

_Q_TOKEN_KV_BLOCK_SPARSE_TP_HEAD_GEOMETRIES = (
    (1, 24, 2),
    (2, 12, 1),
    (4, 6, 1),
    (6, 4, 1),
    (8, 3, 1),
    (12, 2, 1),
    (24, 1, 1),
)


_FP8 = torch.float8_e4m3fn
_FP8_PROBABILITY_SCALE = 448.0
_FP8_KV_TILE_SIZE = 128
_FP8_NUM_KV_INSTANCES = 2


def _single_cta_wave_capacity(device: torch.device | str = "cuda") -> int:
    """Return the runtime SM count used by direct-versus-persistent policy."""

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


@cute.kernel
def _runtime_int32_kv_ceil_div_kernel(
    values: cute.Pointer,
    output: cute.Pointer,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Evaluate the production runtime K-tile/page helpers on device."""

    index, _, _ = cute.arch.block_idx()
    seq_len_kv = Int32(values[index])
    output[index * Int32(2)] = _runtime_total_kv_tiles(
        cfg,
        seq_len_kv,
        Int32(1),
        Int32(0),
    )
    output[index * Int32(2) + Int32(1)] = _runtime_last_valid_page_idx(
        cfg,
        seq_len_kv,
    )


@cute.jit
def _launch_runtime_int32_kv_ceil_div(
    values: cute.Pointer,
    output: cute.Pointer,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    count: cutlass.Constexpr[int],
) -> None:
    _runtime_int32_kv_ceil_div_kernel(values, output, cfg).launch(
        grid=(count, 1, 1),
        block=(1, 1, 1),
    )


@dataclass(frozen=True)
class _DecodeCase:
    """One deterministic fixed-table problem plus a CSR reference oracle."""

    q: torch.Tensor
    paged_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_tables: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    reference_real: torch.Tensor
    output_dtype: torch.dtype
    mask_type: str
    bmm1_scale: float
    bmm2_scale: float
    q_scale: float
    k_scale: float
    v_scale: float
    o_scale: float
    window_left: int = -1


def _stored(real: torch.Tensor, dtype: torch.dtype, scale: float) -> torch.Tensor:
    return (real / scale).to(dtype) if dtype == _FP8 else real.to(dtype)


def _seq_lens_from_csr(
    paged_kv_indptr: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    page_counts = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
    return (page_counts - 1) * page_size + paged_kv_last_page_len


def _block_tables_from_csr(
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    *,
    table_capacity: Optional[int] = None,
    row_stride: Optional[int] = None,
) -> torch.Tensor:
    """Build a row-strided fixed table with poisoned inactive tails."""

    indptr = tuple(int(value) for value in paged_kv_indptr.tolist())
    page_counts = tuple(
        end - begin for begin, end in zip(indptr[:-1], indptr[1:], strict=True)
    )
    capacity = max(page_counts) if table_capacity is None else table_capacity
    stride = capacity if row_stride is None else row_stride
    if capacity < max(page_counts) or stride < capacity:
        raise ValueError("test page-table geometry is too small")
    backing = torch.full(
        (len(page_counts), stride),
        -1,
        dtype=torch.int32,
        device=paged_kv_indices.device,
    )
    block_tables = backing[:, :capacity]
    for request_idx, (page_begin, page_end) in enumerate(
        zip(indptr[:-1], indptr[1:], strict=True)
    ):
        block_tables[request_idx, : page_end - page_begin].copy_(
            paged_kv_indices[page_begin:page_end]
        )
    return block_tables


def _dense_block_table_from_csr(
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    *,
    min_num_pages: int = 0,
) -> torch.Tensor:
    """Convert native CSR page IDs to the standalone PrimTS dense ABI."""

    offsets = [int(value) for value in paged_kv_indptr.cpu().tolist()]
    offset_pairs = tuple(zip(offsets[:-1], offsets[1:], strict=True))
    row_lengths = [end - begin for begin, end in offset_pairs]
    max_num_pages = max([1, min_num_pages, *row_lengths])
    block_table = paged_kv_indices.new_full(
        (len(row_lengths), max_num_pages),
        -1,
    )
    for row, (begin, end) in enumerate(offset_pairs):
        block_table[row, : end - begin] = paged_kv_indices[begin:end]
    return block_table


def _packed_q_token_kv_block_sparse_page_memberships_from_csr(
    paged_kv_indptr: torch.Tensor,
    page_memberships: torch.Tensor,
    *,
    min_num_pages: int = 0,
) -> torch.Tensor:
    """Pack separate per-page membership bytes into the dense QToken-KvBlock-Sparse-Attention ABI."""

    offsets = [int(value) for value in paged_kv_indptr.cpu().tolist()]
    offset_pairs = tuple(zip(offsets[:-1], offsets[1:], strict=True))
    row_lengths = [end - begin for begin, end in offset_pairs]
    max_num_pages = max([1, min_num_pages, *row_lengths])
    packed = page_memberships.new_zeros(
        (
            len(row_lengths),
            (max_num_pages + Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIPS_PER_WORD - 1)
            // Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIPS_PER_WORD,
        )
    )
    for row, (begin, end) in enumerate(offset_pairs):
        row_memberships = page_memberships[begin:end]
        for byte_idx in range(Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIPS_PER_WORD):
            byte_values = row_memberships[
                byte_idx::Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIPS_PER_WORD
            ]
            packed[row, : byte_values.numel()] |= byte_values << (
                byte_idx * Q_TOKEN_KV_BLOCK_SPARSE_PAGE_MEMBERSHIP_BITS
            )
    return packed


def _visible_kv_bounds(
    *,
    kv_len: int,
    seq_len_q: int,
    query_idx: int,
    mask_type: str,
    window_left: int = -1,
) -> tuple[int, int]:
    """Return FlashInfer's bottom-right decode mask interval ``[begin, end)``."""

    if mask_type == "dense":
        end = kv_len
    elif mask_type == "causal":
        end = kv_len - seq_len_q + query_idx + 1
    else:
        raise ValueError("mask_type must be 'dense' or 'causal'")
    if end <= 0:
        raise ValueError("every causal KV sequence must be at least as long as Q")
    begin = 0 if window_left < 0 else max(0, end - window_left - 1)
    return begin, end


@torch.no_grad()
def _decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    *,
    sm_scale: float,
    q_scale: float,
    k_scale: float,
    v_scale: float,
    mask_type: str,
    window_left: int = -1,
    qo_indptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Small independent FP32 oracle for HND paged GQA decode."""

    packed_q = qo_indptr is not None
    if packed_q:
        if q.ndim != 3:
            raise ValueError("packed Q must be [total_q, Hq, D]")
        batch_size = qo_indptr.numel() - 1
        num_qo_heads, head_dim = q.shape[-2:]
        q_tokens = q
        preserve_sq1_shape = False
    elif q.ndim == 3:
        q_tokens = q.unsqueeze(1)
        preserve_sq1_shape = True
    elif q.ndim == 4:
        q_tokens = q
        preserve_sq1_shape = False
    else:
        raise ValueError("Q must be [B, Hq, D] or [B, SQ, Hq, D]")
    if not packed_q:
        batch_size, _, num_qo_heads, head_dim = q_tokens.shape
    _, num_kv_heads, page_size, cache_head_dim = k_cache.shape
    if cache_head_dim != head_dim or num_qo_heads % num_kv_heads:
        raise ValueError("invalid GQA head geometry")

    group_size = num_qo_heads // num_kv_heads
    q_real = q_tokens.float() * q_scale
    k_real = k_cache.float() * k_scale
    v_real = v_cache.float() * v_scale
    output = torch.empty_like(q_real)
    for batch_idx in range(batch_size):
        page_begin = int(paged_kv_indptr[batch_idx].item())
        page_end = int(paged_kv_indptr[batch_idx + 1].item())
        page_ids = paged_kv_indices[page_begin:page_end].long()
        kv_len = (page_end - page_begin - 1) * page_size + int(
            paged_kv_last_page_len[batch_idx].item()
        )
        keys = (
            k_real[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        values = (
            v_real[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        if packed_q:
            q_begin = int(qo_indptr[batch_idx].item())
            q_end = int(qo_indptr[batch_idx + 1].item())
            request_queries = q_real[q_begin:q_end]
        else:
            request_queries = q_real[batch_idx]
            q_begin = 0
        seq_len_q = request_queries.shape[0]
        for query_idx in range(seq_len_q):
            begin, end = _visible_kv_bounds(
                kv_len=kv_len,
                seq_len_q=seq_len_q,
                query_idx=query_idx,
                mask_type=mask_type,
                window_left=window_left,
            )
            logits = torch.einsum(
                "hd,thd->ht",
                request_queries[query_idx],
                keys[begin:end],
            )
            probabilities = torch.softmax(logits * sm_scale, dim=-1)
            result = torch.einsum("ht,thd->hd", probabilities, values[begin:end])
            if packed_q:
                output[q_begin + query_idx] = result
            else:
                output[batch_idx, query_idx] = result
    return output[:, 0] if preserve_sq1_shape else output


@torch.no_grad()
def _fp8_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float,
    output_dtype: torch.dtype,
    o_scale: float,
    mask_type: str,
    window_left: int = -1,
    qo_indptr: Optional[torch.Tensor] = None,
    splits_kv: int = 1,
    num_insts_kv: int = _FP8_NUM_KV_INSTANCES,
    probability_scale: float = _FP8_PROBABILITY_SCALE,
) -> torch.Tensor:
    """Model the kernel's FP8-P operand and split/instance stream merge."""

    if q.dtype != _FP8 or k_cache.dtype != _FP8 or v_cache.dtype != _FP8:
        raise ValueError("the FP8 reference requires E4M3 Q, K, and V")
    if output_dtype not in (torch.float16, torch.bfloat16, _FP8):
        raise ValueError("FP8 FMHA decode supports float16, bfloat16, or E4M3 output")
    if splits_kv <= 0 or num_insts_kv <= 0:
        raise ValueError("splits_kv and num_insts_kv must be positive")
    if not math.isfinite(probability_scale) or probability_scale <= 0:
        raise ValueError("probability_scale must be finite and positive")

    packed_q = qo_indptr is not None
    if packed_q:
        if q.ndim != 3:
            raise ValueError("packed Q must be [total_q, Hq, D]")
        batch_size = qo_indptr.numel() - 1
        num_qo_heads, head_dim = q.shape[-2:]
        q_tokens = q
        preserve_sq1_shape = False
    elif q.ndim == 3:
        batch_size, num_qo_heads, head_dim = q.shape
        q_tokens = q.unsqueeze(1)
        preserve_sq1_shape = True
    elif q.ndim == 4:
        batch_size, _, num_qo_heads, head_dim = q.shape
        q_tokens = q
        preserve_sq1_shape = False
    else:
        raise ValueError("Q must be [B, Hq, D] or [B, SQ, Hq, D]")

    _, num_kv_heads, page_size, cache_head_dim = k_cache.shape
    if cache_head_dim != head_dim or num_qo_heads % num_kv_heads:
        raise ValueError("invalid GQA head geometry")
    group_size = num_qo_heads // num_kv_heads
    k_stored = k_cache.float()
    v_stored = v_cache.float()
    output_shape = (
        q.shape if packed_q else (batch_size, q_tokens.shape[1], num_qo_heads, head_dim)
    )
    output = torch.empty(output_shape, dtype=torch.float32, device=q.device)

    for batch_idx in range(batch_size):
        page_begin = int(paged_kv_indptr[batch_idx].item())
        page_end = int(paged_kv_indptr[batch_idx + 1].item())
        page_ids = paged_kv_indices[page_begin:page_end].long()
        kv_len = (page_end - page_begin - 1) * page_size + int(
            paged_kv_last_page_len[batch_idx].item()
        )
        keys = (
            k_stored[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        values = (
            v_stored[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .repeat_interleave(group_size, dim=1)
        )
        if packed_q:
            q_begin = int(qo_indptr[batch_idx].item())
            q_end = int(qo_indptr[batch_idx + 1].item())
            request_queries = q_tokens[q_begin:q_end].float()
        else:
            request_queries = q_tokens[batch_idx].float()
            q_begin = 0

        seq_len_q = request_queries.shape[0]
        for query_idx in range(seq_len_q):
            visible_begin, visible_end = _visible_kv_bounds(
                kv_len=kv_len,
                seq_len_q=seq_len_q,
                query_idx=query_idx,
                mask_type=mask_type,
                window_left=window_left,
            )
            visible_keys = keys[visible_begin:visible_end]
            visible_values = values[visible_begin:visible_end]
            scores = torch.einsum(
                "hd,thd->ht", request_queries[query_idx], visible_keys
            )
            num_tiles = (
                visible_keys.shape[0] + _FP8_KV_TILE_SIZE - 1
            ) // _FP8_KV_TILE_SIZE
            tiles_per_group = splits_kv * num_insts_kv
            groups_per_split = (num_tiles + tiles_per_group - 1) // tiles_per_group
            local_tiles = max(
                groups_per_split * num_insts_kv,
                num_insts_kv,
            )
            active_splits = (num_tiles + local_tiles - 1) // local_tiles
            stream_tiles = []
            for split_idx in range(active_splits):
                split_begin = split_idx * local_tiles
                split_end = min(split_begin + local_tiles, num_tiles)
                for instance_idx in range(num_insts_kv):
                    tile_indices = range(
                        split_begin + instance_idx,
                        split_end,
                        num_insts_kv,
                    )
                    if tile_indices.start < tile_indices.stop:
                        stream_tiles.append(tile_indices)

            stream_max = [
                torch.full(
                    (num_qo_heads,),
                    -torch.inf,
                    dtype=torch.float32,
                    device=q.device,
                )
                for _ in stream_tiles
            ]
            stream_sum = [
                torch.zeros(num_qo_heads, dtype=torch.float32, device=q.device)
                for _ in stream_tiles
            ]
            stream_acc = [
                torch.zeros(
                    (num_qo_heads, head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                for _ in stream_tiles
            ]
            stream_valid = [False] * len(stream_tiles)

            for stream_idx, tile_indices in enumerate(stream_tiles):
                for tile_idx in tile_indices:
                    tile_begin = tile_idx * _FP8_KV_TILE_SIZE
                    tile_end = min(
                        tile_begin + _FP8_KV_TILE_SIZE, visible_keys.shape[0]
                    )
                    tile_scores = scores[:, tile_begin:tile_end]
                    local_max = tile_scores.max(dim=-1).values
                    new_max = (
                        torch.maximum(stream_max[stream_idx], local_max)
                        if stream_valid[stream_idx]
                        else local_max
                    )
                    probabilities = (
                        torch.exp((tile_scores - new_max.unsqueeze(-1)) * bmm1_scale)
                        * probability_scale
                    )
                    quantized_probabilities = probabilities.to(_FP8).float()
                    local_sum = probabilities.sum(dim=-1)
                    tile_acc = torch.einsum(
                        "ht,thd->hd",
                        quantized_probabilities,
                        visible_values[tile_begin:tile_end],
                    )
                    if stream_valid[stream_idx]:
                        correction = torch.exp(
                            (stream_max[stream_idx] - new_max) * bmm1_scale
                        )
                        stream_sum[stream_idx] = (
                            stream_sum[stream_idx] * correction + local_sum
                        )
                        stream_acc[stream_idx] = (
                            stream_acc[stream_idx] * correction.unsqueeze(-1) + tile_acc
                        )
                    else:
                        stream_sum[stream_idx] = local_sum
                        stream_acc[stream_idx] = tile_acc
                        stream_valid[stream_idx] = True
                    stream_max[stream_idx] = new_max

            valid_streams = [
                (maximum, denominator, accumulator)
                for maximum, denominator, accumulator, valid in zip(
                    stream_max,
                    stream_sum,
                    stream_acc,
                    stream_valid,
                    strict=True,
                )
                if valid
            ]
            final_max = (
                torch.stack([maximum for maximum, _, _ in valid_streams])
                .max(dim=0)
                .values
            )
            final_sum = torch.zeros_like(final_max)
            final_acc = torch.zeros_like(stream_acc[0])
            for maximum, denominator, accumulator in valid_streams:
                correction = torch.exp((maximum - final_max) * bmm1_scale)
                final_sum += denominator * correction
                final_acc += accumulator * correction.unsqueeze(-1)
            output_real = final_acc / final_sum.unsqueeze(-1) * bmm2_scale
            output_real = output_real.to(output_dtype).float() * o_scale
            if packed_q:
                output[q_begin + query_idx] = output_real
            else:
                output[batch_idx, query_idx] = output_real
    return output[:, 0] if preserve_sq1_shape else output


def _make_decode_case(
    *,
    kv_lens: Sequence[int],
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len_q: int,
    page_size: int,
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    cache_form: str,
    mask_type: str,
    window_left: int = -1,
    device: str | torch.device,
    seed: int,
) -> _DecodeCase:
    """Create random data with deterministic nonidentity physical page IDs."""

    if mask_type == "causal" and min(kv_lens) < seq_len_q:
        raise ValueError("causal KV sequences must be at least as long as Q")
    batch_size = len(kv_lens)
    pages_per_request = tuple(
        (length + page_size - 1) // page_size for length in kv_lens
    )
    num_referenced_pages = sum(pages_per_request)
    num_physical_pages = num_referenced_pages + 3
    generator = torch.Generator(device="cpu").manual_seed(seed)
    page_ids = torch.randperm(num_physical_pages, generator=generator)[
        :num_referenced_pages
    ]
    if torch.equal(page_ids, torch.arange(num_referenced_pages)):
        page_ids = torch.roll(page_ids, 1)
    indptr = [0]
    for page_count in pages_per_request:
        indptr.append(indptr[-1] + page_count)

    q_shape = (
        (batch_size, num_qo_heads, head_dim)
        if seq_len_q == 1
        else (batch_size, seq_len_q, num_qo_heads, head_dim)
    )
    q_real = 0.25 * torch.randn(q_shape, generator=generator)
    k_real = 0.25 * torch.randn(
        num_physical_pages, num_kv_heads, page_size, head_dim, generator=generator
    )
    v_real = 0.25 * torch.randn(
        num_physical_pages, num_kv_heads, page_size, head_dim, generator=generator
    )
    q_scale, k_scale, v_scale = (
        (0.5, 0.25, 0.75) if qkv_dtype == _FP8 else (1.0, 1.0, 1.0)
    )
    o_scale = 0.625 if output_dtype == _FP8 else 1.0
    q = _stored(q_real, qkv_dtype, q_scale).to(device)
    k = _stored(k_real, qkv_dtype, k_scale).to(device)
    v = _stored(v_real, qkv_dtype, v_scale).to(device)
    indptr_tensor = torch.tensor(indptr, dtype=torch.int32, device=device)
    indices_tensor = page_ids.to(dtype=torch.int32, device=device)
    block_tables = _block_tables_from_csr(indptr_tensor, indices_tensor)
    last_page_lens = torch.tensor(
        [(length - 1) % page_size + 1 for length in kv_lens],
        dtype=torch.int32,
        device=device,
    )
    if cache_form == "combined":
        combined = torch.stack((k, v), dim=1)
        paged_kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = combined
        k, v = combined[:, 0], combined[:, 1]
    elif cache_form == "tuple":
        paged_kv_cache = (k, v)
    else:
        raise ValueError("cache_form must be 'combined' or 'tuple'")

    sm_scale = 1.0 / math.sqrt(head_dim)
    bmm1_scale = sm_scale * q_scale * k_scale
    bmm2_scale = v_scale / o_scale
    if qkv_dtype == _FP8:
        reference = _fp8_decode_reference(
            q,
            k,
            v,
            indptr_tensor,
            indices_tensor,
            last_page_lens,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            output_dtype=output_dtype,
            o_scale=o_scale,
            mask_type=mask_type,
            window_left=window_left,
        )
    else:
        reference = _decode_reference(
            q,
            k,
            v,
            indptr_tensor,
            indices_tensor,
            last_page_lens,
            sm_scale=sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            mask_type=mask_type,
            window_left=window_left,
        )
    return _DecodeCase(
        q=q,
        paged_kv_cache=paged_kv_cache,
        k_cache=k,
        v_cache=v,
        block_tables=block_tables,
        paged_kv_indptr=indptr_tensor,
        paged_kv_indices=indices_tensor,
        paged_kv_last_page_len=last_page_lens,
        reference_real=reference,
        output_dtype=output_dtype,
        mask_type=mask_type,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        o_scale=o_scale,
        window_left=window_left,
    )


def _pack_page4_cache_into_storage_pages(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    storage_page_size: int,
    physical_layout: str = "compact",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack semantic page-4 IDs into separate or K/V-packed cache storage."""

    semantic_page_size = int(k_cache.shape[2])
    assert semantic_page_size == 4
    assert storage_page_size % semantic_page_size == 0
    subpages_per_storage_page = storage_page_size // semantic_page_size
    num_storage_pages = (
        int(k_cache.shape[0]) + subpages_per_storage_page - 1
    ) // subpages_per_storage_page
    num_kv_heads = int(k_cache.shape[1])
    head_dim = int(k_cache.shape[3])
    if physical_layout == "compact":
        packed_shape = (
            num_storage_pages,
            num_kv_heads,
            storage_page_size,
            head_dim,
        )
        packed_k = torch.zeros(packed_shape, dtype=k_cache.dtype, device=k_cache.device)
        packed_v = torch.zeros_like(packed_k)
    elif physical_layout in ("kv-packed-hnd", "kv-packed-nhd"):
        physical_shape = (
            (
                num_storage_pages,
                num_kv_heads,
                storage_page_size,
                2 * head_dim,
            )
            if physical_layout == "kv-packed-hnd"
            else (
                num_storage_pages,
                storage_page_size,
                num_kv_heads,
                2 * head_dim,
            )
        )
        packed_kv = torch.zeros(
            physical_shape, dtype=k_cache.dtype, device=k_cache.device
        )
        logical_hnd = (
            packed_kv
            if physical_layout == "kv-packed-hnd"
            else packed_kv.permute(0, 2, 1, 3)
        )
        packed_k, packed_v = logical_hnd.split(head_dim, dim=-1)
    else:
        raise ValueError(f"unsupported physical_layout {physical_layout!r}")
    for locator in range(int(k_cache.shape[0])):
        physical_page = locator // subpages_per_storage_page
        subpage = locator % subpages_per_storage_page
        token_begin = subpage * semantic_page_size
        token_end = token_begin + semantic_page_size
        packed_k[physical_page, :, token_begin:token_end].copy_(k_cache[locator])
        packed_v[physical_page, :, token_begin:token_end].copy_(v_cache[locator])
    return packed_k, packed_v


def _ragged_lengths(
    batch_size: int, max_seq_len: int, page_size: int
) -> tuple[int, ...]:
    """Return deterministic positive lengths spanning half to full maximum K."""

    if batch_size <= 0 or max_seq_len <= 0 or page_size <= 0:
        raise ValueError("batch size, maximum length, and page size must be positive")
    if batch_size == 1:
        return (max_seq_len,)
    lower = max(page_size, max_seq_len // 2)
    span = max_seq_len - lower
    lengths = [max_seq_len]
    used = {max_seq_len}
    for batch_idx in range(1, batch_size):
        candidate = lower + ((batch_idx * 104729 + batch_size * 37) % span)
        candidate = min(candidate, max_seq_len - 1)
        if candidate % page_size == 0:
            candidate -= 1
        while candidate in used and candidate > lower:
            candidate -= 1
            if candidate % page_size == 0:
                candidate -= 1
        if candidate in used:
            raise AssertionError("ragged test geometry does not have enough lengths")
        used.add(candidate)
        lengths.append(candidate)
    return tuple(lengths)


def _case(
    batch_size: int,
    max_seq_len: int,
    num_qo_heads: int,
    head_dim: int,
    qkv_dtype: torch.dtype,
    seed: int,
    *,
    num_kv_heads: int = 1,
    seq_len_q: int = 1,
    page_size: int = 32,
    output_dtype: torch.dtype | None = None,
    cache_form: str = "combined",
    mask_type: str = "dense",
    window_left: int = -1,
) -> dict[str, object]:
    return {
        "kv_lens": _ragged_lengths(batch_size, max_seq_len, page_size),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "seq_len_q": seq_len_q,
        "page_size": page_size,
        "qkv_dtype": qkv_dtype,
        "output_dtype": qkv_dtype if output_dtype is None else output_dtype,
        "cache_form": cache_form,
        "mask_type": mask_type,
        "window_left": window_left,
        "seed": seed,
    }


_UNSET = object()


def _policy(
    mma_variant: str,
    tile_size_q: int,
    *,
    grouped: bool = True,
    tile_size_kv: int | object = _UNSET,
    splits: int | object = _UNSET,
    split: bool | object = _UNSET,
    cluster: bool | object = _UNSET,
    separate: bool | object = _UNSET,
    persistent: bool | object = _UNSET,
) -> dict[str, object]:
    """Build only the policy fields that a row intentionally contracts."""

    expected = {
        "mma_variant": mma_variant,
        "tile_size_q": tile_size_q,
        "groups_tokens_heads_q": grouped,
    }
    optional = {
        "tile_size_kv": tile_size_kv,
        "splits_kv": splits,
        "use_split_kv": split,
        "use_cluster_smem_reduction": cluster,
        "use_separate_reduction_kernel": separate,
        "use_persistent_scheduler": persistent,
    }
    expected.update(
        {key: value for key, value in optional.items() if value is not _UNSET}
    )
    return expected


def _param(
    case_kwargs: dict[str, object],
    expected_policy: dict[str, object],
    correction_pattern: str | None,
    *,
    exercise_all_paths: bool = False,
    id: str,
):
    return pytest.param(
        case_kwargs,
        expected_policy,
        correction_pattern,
        exercise_all_paths,
        id=id,
    )


_FMHA_CASES = (
    _param(
        _case(4, 4096, 8, 128, torch.bfloat16, 31001),
        _policy(
            "swaps_mma_ab",
            8,
            tile_size_kv=128,
            split=True,
            cluster=True,
            separate=False,
            persistent=False,
        ),
        "identity",
        exercise_all_paths=True,
        id="F1-bf16-q8-cluster-ragged",
    ),
    _param(
        _case(5, 4097, 64, 256, torch.float8_e4m3fn, 31002, num_kv_heads=4),
        _policy(
            "swaps_mma_ab",
            16,
            splits=6,
            cluster=True,
            separate=False,
            persistent=False,
        ),
        "mixed",
        id="F2-fp8-d256-q16-s6-cluster",
    ),
    _param(
        _case(
            38,
            2177,
            128,
            64,
            torch.bfloat16,
            31003,
            num_kv_heads=4,
            page_size=64,
            cache_form="tuple",
        ),
        _policy(
            "swaps_mma_ab",
            32,
            split=False,
            cluster=False,
            separate=False,
            persistent=True,
        ),
        None,
        id="F3-bf16-d64-q32-direct-page64",
    ),
    _param(
        _case(
            38,
            2048,
            32,
            128,
            torch.float16,
            31004,
            num_kv_heads=4,
            page_size=128,
        ),
        _policy(
            "swaps_mma_ab",
            8,
            tile_size_kv=128,
            split=False,
            cluster=False,
            separate=False,
            persistent=True,
        ),
        None,
        id="F4-fp16-q8-clc-page128",
    ),
    _param(
        _case(
            8,
            4097,
            24,
            128,
            torch.float8_e4m3fn,
            31005,
            seq_len_q=4,
            output_dtype=torch.float16,
            mask_type="causal",
        ),
        _policy(
            "swaps_mma_ab",
            32,
            splits=4,
            split=True,
            cluster=True,
            separate=False,
            persistent=False,
        ),
        "mixed",
        id="F5-fp8-fp16-sq4-swaps-q32-cluster",
    ),
    _param(
        _case(
            4,
            8192,
            64,
            64,
            torch.float8_e4m3fn,
            31006,
            num_kv_heads=4,
            seq_len_q=4,
            mask_type="causal",
        ),
        _policy(
            "keeps_mma_ab",
            64,
            splits=8,
            separate=False,
        ),
        "identity",
        id="F6-fp8-d64-sq4-q64-s8",
    ),
    _param(
        _case(
            4,
            8192,
            128,
            128,
            torch.float8_e4m3fn,
            31007,
            num_kv_heads=4,
            seq_len_q=4,
            cache_form="tuple",
            mask_type="causal",
        ),
        _policy(
            "keeps_mma_ab",
            128,
            splits=8,
            separate=True,
        ),
        "mixed",
        id="F7-fp8-sq4-q128-separate",
    ),
    _param(
        _case(
            2,
            2049,
            16,
            128,
            torch.bfloat16,
            31008,
            seq_len_q=4,
            page_size=16,
            mask_type="dense",
        ),
        _policy(
            "swaps_mma_ab",
            16,
            splits=5,
            cluster=True,
        ),
        "tail",
        id="F8-spec-dense-tail-visible",
    ),
    _param(
        _case(
            2,
            2049,
            16,
            128,
            torch.bfloat16,
            31008,
            seq_len_q=4,
            page_size=16,
            mask_type="causal",
        ),
        _policy(
            "swaps_mma_ab",
            16,
            splits=5,
            cluster=True,
        ),
        "tail",
        id="F9-spec-causal-tail-progressive",
    ),
    _param(
        _case(
            3,
            2051,
            32,
            256,
            torch.bfloat16,
            31009,
            num_kv_heads=4,
            page_size=16,
        ),
        _policy("swaps_mma_ab", 8),
        None,
        id="F10-bf16-d256-page16-staged",
    ),
    _param(
        _case(
            16,
            2051,
            96,
            128,
            torch.float8_e4m3fn,
            31010,
            num_kv_heads=8,
            seq_len_q=8,
            page_size=32,
            mask_type="causal",
        ),
        _policy("keeps_mma_ab", 128, splits=1, split=False),
        "tail",
        id="F11-fp8-sq8-q128-runtime-causal",
    ),
    _param(
        _case(
            1,
            2048,
            64,
            64,
            torch.bfloat16,
            31011,
            num_kv_heads=4,
        ),
        _policy(
            "swaps_mma_ab",
            8,
            grouped=False,
            splits=4,
            split=True,
            cluster=True,
            separate=False,
            persistent=False,
        ),
        None,
        exercise_all_paths=True,
        id="F12-bf16-q16-underfilled-head-bands",
    ),
)


def _plan_case(
    case,
    *,
    max_kv_len: int,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
):
    wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
    seq_len_q = 1 if case.q.ndim == 3 else int(case.q.shape[1])
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    )
    wrapper.plan(
        case.q.device,
        int(case.paged_kv_indptr.numel()) - 1,
        case.q.shape[-2],
        case.k_cache.shape[1],
        case.q.shape[-1],
        case.k_cache.shape[2],
        max_kv_len,
        max_seq_len_q=(seq_len_q if max_seq_len_q is None else max_seq_len_q),
        packed_query=qo_indptr is not None,
        q_data_type=case.q.dtype,
        kv_data_type=case.k_cache.dtype,
        o_data_type=case.output_dtype,
        mask_type=case.mask_type,
        window_left=case.window_left,
        seq_lens=seq_lens.cpu(),
    )
    return wrapper


def _assert_auto_policy(
    policy: dict[str, object],
    expected_b200: dict[str, object],
    *,
    device: torch.device,
) -> None:
    """Contract exact B200 coverage and portable Blackwell legality."""

    assert policy.get("source", "auto") == "auto"
    assert policy["mma_variant"] in ("swaps_mma_ab", "keeps_mma_ab")
    assert policy["tile_size_q"] in (8, 16, 32, 64, 128)
    assert policy["tile_size_kv"] in (128, 256)
    assert isinstance(policy["groups_tokens_heads_q"], bool)
    assert policy["query_layout"] == policy["output_layout"]
    splits_kv = int(policy["splits_kv"])
    assert 1 <= splits_kv <= int(policy["max_splits_kv"])
    use_split_kv = bool(policy["use_split_kv"])
    use_cluster = bool(policy["use_cluster_smem_reduction"])
    use_separate = bool(policy["use_separate_reduction_kernel"])
    assert not (use_cluster and use_separate)
    if not use_split_kv:
        assert splits_kv == 1
        assert not use_cluster
        assert not use_separate
    if use_cluster or use_separate:
        assert use_split_kv
    if "tile_size_kv" in expected_b200:
        assert policy["tile_size_kv"] == expected_b200["tile_size_kv"]

    if (
        torch.cuda.get_device_capability(device) == (10, 0)
        and policy["tile_size_kv"] == 128
        and _single_cta_wave_capacity(device) == 148
    ):
        for key, expected in expected_b200.items():
            assert policy[key] == expected, (key, policy, expected_b200)


def _run_case(
    wrapper,
    case,
    *,
    seq_lens: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    out=None,
    validate: bool = True,
):
    return wrapper.run(
        case.q,
        case.paged_kv_cache,
        seq_lens,
        case.block_tables,
        qo_indptr=qo_indptr,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out=out,
        validate=validate,
    )


def _run_standalone(
    case,
    seq_lens,
    *,
    max_kv_len: int,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    storage_page_size: Optional[int] = None,
):
    """Run the caller-workspace public entry point for wrapper parity."""

    seq_len_q = 1 if case.q.ndim == 3 else int(case.q.shape[1])
    # The standalone ABI owns persistent reduction counters in this buffer.
    if workspace_buffer is None:
        workspace_size = get_prims_ts_batch_decode_workspace_size(
            case.paged_kv_indptr.numel() - 1,
            case.q.shape[-2],
            case.k_cache.shape[1],
            case.q.shape[-1],
            case.k_cache.shape[2] if page_size is None else page_size,
            max_kv_len,
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            q_dtype=case.q.dtype,
            kv_dtype=case.k_cache.dtype,
            out_dtype=case.output_dtype,
            mask_type=case.mask_type,
            window_left=case.window_left,
            kv_layout="HND",
            storage_page_size=storage_page_size,
            device=case.q.device,
        )
        workspace = torch.zeros(workspace_size, dtype=torch.int8, device=case.q.device)
    else:
        workspace = workspace_buffer
    output = torch.empty_like(case.q, dtype=case.output_dtype) if out is None else out
    if block_table is None:
        block_table = case.block_tables
    result = prims_ts_batch_decode_with_kv_cache(
        case.q,
        case.paged_kv_cache,
        workspace,
        block_table,
        seq_lens,
        max_kv_len,
        seq_len_q=seq_len_q,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out=output,
        out_dtype=case.output_dtype,
        mask_type=case.mask_type,
        window_left=case.window_left,
        kv_layout="HND",
        page_size=page_size,
    )
    assert result is output
    return output


def _assert_case_correct(output, case):
    actual = output.float() * case.o_scale
    expected = case.reference_real
    assert output.shape == case.q.shape
    assert output.dtype == case.output_dtype
    assert torch.isfinite(actual).all()
    if case.output_dtype == _FP8:
        rtol, atol = 5e-2, 2e-3
    elif case.q.dtype == _FP8:
        rtol, atol = 1e-3, 2.5e-4
    else:
        rtol, atol = 1e-2, 1e-2
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _exercise_public_paths(
    wrapper,
    case,
    seq_lens,
    *,
    max_kv_len: int,
    exercise_all_paths: bool,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
):
    """Always check eager; reserve standalone/graph parity for anchor rows."""

    if case.q.dtype == _FP8:
        policy = dict(wrapper._policy)
        case = _with_reference(
            case,
            qo_indptr=qo_indptr,
            splits_kv=int(policy["splits_kv"]),
            num_insts_kv=int(policy["num_insts_kv"]),
        )
    eager = _run_case(
        wrapper,
        case,
        qo_indptr=qo_indptr,
    )
    _assert_case_correct(eager, case)
    if not exercise_all_paths:
        return eager

    standalone = _run_standalone(
        case,
        seq_lens,
        max_kv_len=max_kv_len,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    _assert_case_correct(standalone, case)
    torch.testing.assert_close(standalone, eager, rtol=0, atol=0)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(
            wrapper,
            case,
            qo_indptr=qo_indptr,
            out=graph_out,
            validate=False,
        )
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(graph_out, case)
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)
    return eager


def _with_reference(
    case,
    *,
    q: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    splits_kv: int = 1,
    num_insts_kv: int = _FP8_NUM_KV_INSTANCES,
):
    """Return a case whose oracle reflects its current mutable Q/K/V data."""

    query = case.q if q is None else q
    if query.dtype == _FP8:
        reference = _fp8_decode_reference(
            query,
            case.k_cache,
            case.v_cache,
            case.paged_kv_indptr,
            case.paged_kv_indices,
            case.paged_kv_last_page_len,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
            output_dtype=case.output_dtype,
            o_scale=case.o_scale,
            mask_type=case.mask_type,
            window_left=case.window_left,
            qo_indptr=qo_indptr,
            splits_kv=splits_kv,
            num_insts_kv=num_insts_kv,
        )
    else:
        reference = _decode_reference(
            query,
            case.k_cache,
            case.v_cache,
            case.paged_kv_indptr,
            case.paged_kv_indices,
            case.paged_kv_last_page_len,
            sm_scale=1.0 / math.sqrt(query.shape[-1]),
            q_scale=case.q_scale,
            k_scale=case.k_scale,
            v_scale=case.v_scale,
            mask_type=case.mask_type,
            window_left=case.window_left,
            qo_indptr=qo_indptr,
        )
    return replace(case, q=query, reference_real=reference)


def _pack_decode_case(
    case: _DecodeCase,
    q_lens: Sequence[int],
) -> tuple[_DecodeCase, torch.Tensor]:
    """Pack fixed-Q storage and return its cumulative runtime Q offsets."""

    if case.q.ndim != 4 or len(q_lens) != case.q.shape[0]:
        raise ValueError("packed-Q source must be [B, SQ, H, D] with B lengths")
    if min(q_lens) <= 0 or max(q_lens) > case.q.shape[1]:
        raise ValueError("packed Q lengths must be positive and within source SQ")
    offsets = [0]
    for q_len in q_lens:
        offsets.append(offsets[-1] + q_len)
    qo_indptr = torch.tensor(offsets, dtype=torch.int32, device=case.q.device)
    packed_q = torch.cat(
        [case.q[batch_idx, :q_len] for batch_idx, q_len in enumerate(q_lens)]
    ).contiguous()
    return _with_reference(case, q=packed_q, qo_indptr=qo_indptr), qo_indptr


def _exercise_auto_case(
    case: _DecodeCase,
    *,
    qo_indptr: Optional[torch.Tensor] = None,
    max_seq_len_q: Optional[int] = None,
    exercise_all_paths: bool = False,
) -> dict[str, object]:
    """Plan automatically and validate one eager public-interface launch."""

    page_size = int(case.k_cache.shape[2])
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        page_size,
    )
    max_kv_len = int(seq_lens.max().item())
    wrapper = _plan_case(
        case,
        max_kv_len=max_kv_len,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    policy = dict(wrapper._policy)
    _assert_auto_policy(policy, {}, device=case.q.device)
    _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=max_kv_len,
        exercise_all_paths=exercise_all_paths,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    return policy


def _exercise_explicit_kv256_case(
    monkeypatch: pytest.MonkeyPatch,
    case: _DecodeCase,
    *,
    exercise_all_paths: bool = False,
) -> dict[str, object]:
    """Run a public path with the qualified logical KV256 profile pinned."""

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    original_make_decode_config = fmha_decode_config.make_decode_config
    explicit_profile = {
        "use_keeps_mma_ab": True,
        "tile_size_q": 64,
        "tile_size_kv": 256,
        "groups_tokens_heads_q": True,
    }

    def _make_explicit_kv256_config(*args, **kwargs):
        source = kwargs.get("args")
        kwargs["args"] = (
            explicit_profile if source is None else (source, explicit_profile)
        )
        return original_make_decode_config(*args, **kwargs)

    monkeypatch.setattr(
        fmha_decode_config,
        "make_decode_config",
        _make_explicit_kv256_config,
    )
    decode_module._resolve_decode_launch_spec.cache_clear()
    decode_module._get_compiled_decode.cache_clear()
    try:
        return _exercise_auto_case(
            case,
            exercise_all_paths=exercise_all_paths,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()
        decode_module._get_compiled_decode.cache_clear()


@torch.no_grad()
def _apply_decode_correction_pattern(case, pattern: str):
    query = case.q.unsqueeze(1) if case.q.ndim == 3 else case.q
    query.zero_()
    logical_rows = torch.arange(
        query.shape[1] * query.shape[2], device=query.device
    ).view(query.shape[1], query.shape[2])
    if pattern == "identity":
        signs = torch.ones_like(logical_rows)
    elif pattern == "mixed":
        signs = torch.where(logical_rows.remainder(2) == 0, 1, -1)
    else:
        raise ValueError(f"unsupported correction pattern {pattern!r}")
    magnitude = 8 if query.dtype == torch.float8_e4m3fn else 1
    query[..., 0] = (magnitude * signs).to(query.dtype).unsqueeze(0)

    page_size = case.k_cache.shape[2]
    for batch_idx in range(query.shape[0]):
        page_begin = int(case.paged_kv_indptr[batch_idx].item())
        page_end = int(case.paged_kv_indptr[batch_idx + 1].item())
        page_ids = case.paged_kv_indices[page_begin:page_end].to(torch.long)
        logical_tokens = torch.arange(page_ids.numel() * page_size, device=query.device)
        stored_k = (32 - logical_tokens // 128).clamp_min(0).to(case.k_cache.dtype)
        case.k_cache[page_ids, :, :, 0] = stored_k.view(-1, 1, page_size).expand(
            -1, case.k_cache.shape[1], -1
        )

    return _with_reference(case)


@torch.no_grad()
def _apply_speculative_tail_markers(case):
    """Make dense and bottom-right causal SQ>1 results observably different."""

    query = case.q if case.q.ndim == 4 else case.q.unsqueeze(1)
    if query.shape[1] <= 1:
        raise ValueError("tail-marker coverage requires SQ>1 input")
    query.zero_()
    query[..., 0] = 8 if query.dtype == _FP8 else 1
    case.k_cache.zero_()
    case.v_cache.zero_()
    page_size = int(case.k_cache.shape[2])
    for batch_idx in range(query.shape[0]):
        page_begin = int(case.paged_kv_indptr[batch_idx].item())
        page_end = int(case.paged_kv_indptr[batch_idx + 1].item())
        page_ids = case.paged_kv_indices[page_begin:page_end].long()
        kv_len = (page_end - page_begin - 1) * page_size + int(
            case.paged_kv_last_page_len[batch_idx].item()
        )
        for tail_idx in range(query.shape[1] - 1):
            logical_token = kv_len - query.shape[1] + 1 + tail_idx
            page_id = page_ids[logical_token // page_size]
            page_offset = logical_token % page_size
            case.k_cache[page_id, :, page_offset, 0] = 80
            value_marker = (
                tail_idx + 1 if case.v_cache.dtype == _FP8 else 8 * (tail_idx + 1)
            )
            case.v_cache[page_id, :, page_offset, 0] = value_marker
    return _with_reference(case)


def test_attention_ts_decode_storage_page_size_contract() -> None:
    """Keep encoded locators explicit and restricted to semantic page four."""

    assert _validate_storage_page_size(4, 4) == 4
    assert _validate_storage_page_size(4, 16) == 16
    with pytest.raises(ValueError, match="divisible by page_size"):
        _validate_storage_page_size(4, 18)
    with pytest.raises(ValueError, match="require page_size=4"):
        _validate_storage_page_size(16, 32)
    with pytest.raises(TypeError, match="must be a positive integer"):
        _validate_storage_page_size(4, True)


def _decode_runtime_for_aliasing() -> _DecodeRuntime:
    """Build the smallest runtime object accepted by the alias validator."""

    return _DecodeRuntime(
        q=torch.empty(8),
        k_cache=torch.empty(8),
        v_cache=torch.empty(8),
        out=torch.empty(8),
        num_physical_pages=1,
        k_page_stride=8,
        k_head_stride=8,
        k_token_stride=8,
        v_page_stride=8,
        v_head_stride=8,
        v_token_stride=8,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
    )


def test_attention_ts_decode_alias_guard_covers_every_live_allocation() -> None:
    """The output may not reuse storage that remains live during a launch."""

    for aliased_name in (
        "k_cache",
        "v_cache",
        "seq_lens",
        "qo_indptr",
        "block_tables",
        "workspace_buffer",
    ):
        runtime = _decode_runtime_for_aliasing()
        metadata = {
            "seq_lens": torch.empty(8),
            "qo_indptr": torch.empty(8),
            "block_tables": torch.empty(8),
            "workspace_buffer": torch.empty(8),
        }
        if aliased_name in ("k_cache", "v_cache"):
            runtime = replace(runtime, **{aliased_name: runtime.out})
        else:
            metadata[aliased_name] = runtime.out

        with pytest.raises(
            ValueError,
            match=rf"out must not overlap {aliased_name} storage",
        ):
            _validate_decode_output_aliasing(runtime, **metadata)


def test_attention_ts_decode_public_query_geometry_guards() -> None:
    """Reject unsafe Q extents and head ratios above 128."""

    int32_max = 2**31 - 1
    _validate_decode_query_head_extent(
        batch_size=int32_max,
        num_qo_heads=1,
        max_seq_len_q=1,
    )
    with pytest.raises(
        NotImplementedError,
        match=r"batch_size \* max_seq_len_q \* num_qo_heads.*signed int32",
    ):
        _validate_decode_query_head_extent(
            batch_size=int32_max + 1,
            num_qo_heads=1,
            max_seq_len_q=1,
        )

    # The public sizing path must reject the unsafe semantic key before device
    # probing, policy resolution, or workspace arithmetic.
    with pytest.raises(
        NotImplementedError,
        match=r"batch_size \* max_seq_len_q \* num_qo_heads.*signed int32",
    ):
        get_prims_ts_batch_decode_workspace_size(
            batch_size=2,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            max_seq_len=128,
            seq_len_q=2**27,
            device="cuda:0",
        )

    for supported_ratio in range(1, 129):
        _validate_head_geometry(supported_ratio, 1)
    with pytest.raises(ValueError, match="must be divisible"):
        _validate_head_geometry(127, 2)
    with pytest.raises(ValueError, match=r"1 <= Hq/Hkv <= 128"):
        _validate_head_geometry(129, 1)


@pytest.mark.parametrize(
    ("head_ratio", "expected_mma", "expected_tile_q"),
    (
        (1, "swaps_mma_ab", 8),
        (8, "swaps_mma_ab", 8),
        (9, "swaps_mma_ab", 16),
        (16, "swaps_mma_ab", 16),
        (17, "swaps_mma_ab", 32),
        (32, "swaps_mma_ab", 32),
        (33, "keeps_mma_ab", 64),
        (64, "keeps_mma_ab", 64),
        (65, "keeps_mma_ab", 128),
        (128, "keeps_mma_ab", 128),
    ),
)
@pytest.mark.parametrize("head_dim", (64, 128, 256), ids=lambda value: f"d{value}")
def test_attention_ts_decode_fp8_head_ratio_buckets(
    monkeypatch: pytest.MonkeyPatch,
    head_ratio: int,
    expected_mma: str,
    expected_tile_q: int,
    head_dim: int,
) -> None:
    """Dispatch each supported ratio range to its smallest qualified Q tile."""

    from contextlib import nullcontext
    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 152 // cluster_size,
    )
    decode_module._resolve_decode_launch_spec.cache_clear()
    try:
        spec = decode_module._resolve_decode_launch_spec(
            0,
            256,
            head_ratio,
            1,
            head_dim,
            32,
            4096,
            1,
            "float8_e4m3fn",
            "float8_e4m3fn",
            "float8_e4m3fn",
            "HND",
            "dense",
            False,
            -1,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()

    policy = dict(spec.policy)
    assert policy["mma_variant"] == expected_mma
    assert policy["tile_size_q"] == expected_tile_q
    assert policy["groups_tokens_heads_q"] is True


def test_attention_ts_decode_reserves_int32_kv_tile_padding() -> None:
    """The public K/V bound leaves room for the final exclusive tile endpoint."""

    safe_max = 2**31 - _DECODE_MAX_KV_TILE_SIZE
    assert safe_max == _DECODE_MAX_KV_LEN
    assert _validate_max_kv_len(safe_max, "max_seq_len") == safe_max
    assert safe_max + _DECODE_MAX_KV_TILE_SIZE - 1 == 2**31 - 1
    assert (
        safe_max + _DECODE_MAX_KV_TILE_SIZE - 1
    ) // _DECODE_MAX_KV_TILE_SIZE * _DECODE_MAX_KV_TILE_SIZE <= 2**31 - 1
    assert (
        safe_max + 1 + _DECODE_MAX_KV_TILE_SIZE - 1
    ) // _DECODE_MAX_KV_TILE_SIZE * _DECODE_MAX_KV_TILE_SIZE == 2**31

    with pytest.raises(
        NotImplementedError,
        match=rf"max_seq_len must be <= {safe_max}.*signed int32",
    ):
        _validate_max_kv_len(safe_max + 1, "max_seq_len")

    _validate_decode_policy_kv_tile_size(
        type("Config", (), {"tile_size_kv": _DECODE_MAX_KV_TILE_SIZE})()
    )
    unsupported_tile_size = _DECODE_MAX_KV_TILE_SIZE * 2
    with pytest.raises(
        RuntimeError,
        match=(
            rf"K/V tile no larger than {_DECODE_MAX_KV_TILE_SIZE}"
            rf".*got {unsupported_tile_size}"
        ),
    ):
        _validate_decode_policy_kv_tile_size(
            type("Config", (), {"tile_size_kv": unsupported_tile_size})()
        )


def test_attention_ts_decode_workspace_rejects_unsafe_int32_kv_bound() -> None:
    """Workspace policy lookup rejects the first unsafe bound before CUDA work."""

    with pytest.raises(
        NotImplementedError, match=r"padded FMHA decode K/V coordinates"
    ):
        get_prims_ts_batch_decode_workspace_size(
            batch_size=1,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            max_seq_len=_DECODE_MAX_KV_LEN + 1,
            device="cuda:0",
        )


def test_attention_ts_decode_workspace_layout_uses_explicit_reducer_mode() -> None:
    """FP8 partial-O storage follows the selected reducer, not shape inference."""

    scratch_shapes = (
        (2, 3, 4, 5, 128),
        (2, 3, 4, 5),
        (2, 3, 5),
    )
    separate = _make_decode_workspace_layout(
        scratch_shapes,
        torch.float8_e4m3fn,
        use_separate_reduction_kernel=True,
    )
    inline = _make_decode_workspace_layout(
        scratch_shapes,
        torch.float8_e4m3fn,
        use_separate_reduction_kernel=False,
    )
    assert separate.partial_o.dtype == torch.bfloat16
    assert inline.partial_o.dtype == torch.float16


@pytest.mark.parametrize(
    ("max_kv_len", "expected"),
    ((128, True), (256, False), (257, True), (384, True), (385, False)),
)
def test_attention_ts_decode_planned_kv_domain_detects_unpaired_tail(
    max_kv_len: int,
    expected: bool,
) -> None:
    """Classify paired K128 instructions from the immutable plan bound."""

    config = FmhaDecodeConfig(tile_size_kv=128, num_insts_kv=2)
    assert _planned_kv_domain_has_unpaired_tail(config, max_kv_len) is expected


@pytest.mark.parametrize(
    (
        "tile_size_q",
        "tile_size_kv",
        "use_keeps_mma_ab",
        "total_kv_tiles",
        "expected_budgets",
    ),
    (
        (64, 128, True, 32, (None, None, None)),
        (128, 128, False, 32, (None, None, None)),
        (128, 128, True, 1, (184, 88, 56)),
        (64, 256, True, 31, (None, None, None)),
        (64, 256, True, 32, (152, 152, 56)),
    ),
)
def test_attention_ts_decode_register_reallocation_follows_task_graph(
    tile_size_q: int,
    tile_size_kv: int,
    use_keeps_mma_ab: bool,
    total_kv_tiles: int,
    expected_budgets: tuple[int | None, int | None, int | None],
) -> None:
    """Register hand-off follows topology and amortized KV256 loop length."""

    config = FmhaDecodeConfig(
        tile_size_q=tile_size_q,
        tile_size_kv=tile_size_kv,
        use_keeps_mma_ab=use_keeps_mma_ab,
        total_kv_tiles=total_kv_tiles,
    )
    budgets = (
        config.softmax_task_num_registers,
        config.correction_task_num_registers,
        config.mma_load_task_num_registers,
    )
    assert budgets == expected_budgets
    assert config.uses_task_register_reallocation is (expected_budgets[0] is not None)
    if config.uses_task_register_reallocation:
        assert all(value is not None and value % 8 == 0 for value in budgets)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_launch_and_plan_reject_unsafe_int32_kv_bound() -> None:
    """Standalone launch and reusable planning share the padded-coordinate cap."""

    device = torch.device("cuda")
    page_size = 16
    assert (
        get_prims_ts_batch_decode_workspace_size(
            batch_size=1,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=64,
            page_size=page_size,
            max_seq_len=_DECODE_MAX_KV_LEN,
            device=device,
        )
        > 0
    )
    q = torch.empty((1, 8, 64), dtype=torch.float16, device=device)
    kv_cache = torch.empty((1, 2, 1, page_size, 64), dtype=torch.float16, device=device)
    paged_kv_indices = torch.tensor((0,), dtype=torch.int32, device=device)
    block_tables = paged_kv_indices.view(1, 1)
    last_page_len = torch.tensor((page_size,), dtype=torch.int32, device=device)
    seq_lens = last_page_len.clone()
    unsafe_max = _DECODE_MAX_KV_LEN + 1

    with pytest.raises(
        NotImplementedError, match=r"padded FMHA decode K/V coordinates"
    ):
        prims_ts_batch_decode_with_kv_cache(
            q,
            kv_cache,
            torch.empty(1, dtype=torch.uint8, device=device),
            block_tables,
            seq_lens,
            unsafe_max,
        )

    with pytest.raises(
        NotImplementedError, match=r"padded FMHA decode K/V coordinates"
    ):
        BatchDecodePagedTSWrapper().plan(
            device,
            1,
            8,
            1,
            64,
            page_size,
            unsafe_max,
        )


def test_attention_ts_workspace_alias_guard() -> None:
    """Caller-owned scratch must be disjoint from every live allocation."""

    storage = torch.empty(64, dtype=torch.uint8)
    workspace = storage[:32]
    overlapping_query = storage[16:48]
    with pytest.raises(
        ValueError,
        match="workspace_buffer must not overlap query storage",
    ):
        _validate_tensor_does_not_overlap_inputs(
            workspace,
            "workspace_buffer",
            ("query", overlapping_query),
        )


@pytest.mark.parametrize(
    ("tile_size_q", "head_dim", "split_kv"),
    ((8, 64, 2), (16, 128, 32), (128, 256, 128)),
)
def test_attention_ts_decode_reduction_mode_order_is_structural(
    tile_size_q: int,
    head_dim: int,
    split_kv: int,
) -> None:
    """FMHA reducer order is independent of measured shape crossovers."""

    assert select_split_kv_modes(
        family="fmha_decode",
        topology="1cta",
        tile_size_q=tile_size_q,
        head_dim=head_dim,
        head_dim_per_cta_v=None,
        split_kv=split_kv,
        available_modes=(
            "gmem_reduction",
            "gmem_reduction_with_separate_kernel",
            "cluster_smem_reduction",
        ),
    ) == (
        "cluster_smem_reduction",
        "gmem_reduction_with_separate_kernel",
        "gmem_reduction",
    )


_DECODE_PUBLIC_SURFACES = (
    BatchDecodePagedTSWrapper.__init__,
    BatchDecodePagedTSWrapper.plan,
    BatchDecodePagedTSWrapper.run,
    batch_decode_with_paged_kv_cache,
    get_prims_ts_batch_decode_workspace_size,
    prims_ts_batch_decode_with_kv_cache,
)


def test_attention_ts_decode_wrapper_has_compile_oriented_contract() -> None:
    """Freeze the compile-oriented plan/run split."""

    assert tuple(inspect.signature(batch_decode_with_paged_kv_cache).parameters) == (
        "q",
        "paged_kv_cache",
        "block_tables",
        "seq_lens_kv",
        "seq_len_q",
        "qo_indptr",
        "max_seq_len_q",
        "mask_type",
        "window_left",
        "kv_layout",
        "bmm1_scale",
        "bmm2_scale",
        "out",
        "out_dtype",
        "page_size",
    )
    assert tuple(inspect.signature(BatchDecodePagedTSWrapper.__init__).parameters) == (
        "self",
        "kv_layout",
    )
    assert tuple(inspect.signature(BatchDecodePagedTSWrapper.plan).parameters) == (
        "self",
        "device",
        "batch_size",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "page_size",
        "max_kv_len",
        "max_seq_len_q",
        "packed_query",
        "q_data_type",
        "kv_data_type",
        "o_data_type",
        "mask_type",
        "window_left",
        "seq_lens",
        "workspace_buffer",
        "storage_page_size",
    )
    run_parameters = inspect.signature(BatchDecodePagedTSWrapper.run).parameters
    assert tuple(run_parameters) == (
        "self",
        "q",
        "paged_kv_cache",
        "seq_lens",
        "block_tables",
        "qo_indptr",
        "bmm1_scale",
        "bmm2_scale",
        "out",
        "validate",
    )
    assert run_parameters["validate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert run_parameters["validate"].default is True

    assert _DecodePlanState.__dataclass_params__.frozen is True
    assert {
        "planned_seq_lens_host",
        "planned_seq_lens_device",
    }.issubset(_DecodePlanState.__dataclass_fields__)
    request_fields = {"qo_indptr", "block_tables"}
    assert request_fields.isdisjoint(_DecodePlanState.__dataclass_fields__)


_FORBIDDEN_DECODE_TUNING_PARAMETERS = frozenset(
    {
        "args",
        "auto_tuner",
        "autotuner",
        "config",
        "enable_clc",
        "enable_pdl",
        "fixed_split_size",
        "groups_tokens_heads_q",
        "kv_stages",
        "mma_variant",
        "num_insts_kv",
        "num_stages",
        "num_warps",
        "o_stages",
        "profile",
        "q_stages",
        "reduction_mode",
        "schedule",
        "single_kv",
        "split_kv",
        "split_kv_mode",
        "splits_kv",
        "tile_size_kv",
        "tile_size_q",
        "use_cluster_smem_reduction",
        "use_keeps_mma_ab",
        "use_persistent_scheduler",
        "use_separate_reduction_kernel",
        "warp_specialization",
    }
)
_FORBIDDEN_TUNING_TOKEN_PREFIXES = frozenset(
    {
        "autotun",
        "clc",
        "config",
        "cta",
        "impl",
        "inst",
        "kernel",
        "mma",
        "pdl",
        "persist",
        "profil",
        "reduc",
        "schedul",
        "split",
        "stag",
        "tile",
        "warp",
    }
)
_FORBIDDEN_TUNING_TOKEN_SEQUENCES = (
    ("groups", "tokens", "heads"),
    ("single", "kv"),
    ("tensor", "cores"),
)


def test_attention_ts_decode_public_surfaces_have_no_internal_tuning_knobs() -> None:
    """Keep launch-policy controls private without freezing whole signatures."""

    violations = []
    for surface in _DECODE_PUBLIC_SURFACES:
        parameters = inspect.signature(surface).parameters
        for parameter in parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                violations.append(f"{surface.__qualname__}.**{parameter.name}")
                continue
            tokens = tuple(parameter.name.split("_"))
            has_forbidden_token = any(
                token.startswith(prefix)
                for token in tokens
                for prefix in _FORBIDDEN_TUNING_TOKEN_PREFIXES
            )
            has_forbidden_sequence = any(
                tokens[index : index + len(sequence)] == sequence
                for sequence in _FORBIDDEN_TUNING_TOKEN_SEQUENCES
                for index in range(len(tokens) - len(sequence) + 1)
            )
            if (
                parameter.name in _FORBIDDEN_DECODE_TUNING_PARAMETERS
                or has_forbidden_token
                or has_forbidden_sequence
            ):
                violations.append(f"{surface.__qualname__}.{parameter.name}")

    assert violations == []


def test_attention_ts_decode_bound_wrapper_trace_uses_plan_state():
    """Trace packed-Q shape and planned output dtype from the live wrapper."""
    from flashinfer.fi_trace import fi_trace

    wrapper = BatchDecodePagedTSWrapper()
    q = torch.empty((5, 32, 128), dtype=_FP8)
    k_cache = torch.empty((8, 4, 32, 128), dtype=_FP8)
    v_cache = torch.empty_like(k_cache)
    kwargs = {
        "q": q,
        "paged_kv_cache": (k_cache, v_cache),
        "seq_lens": torch.tensor((32, 64), dtype=torch.int32),
        "block_tables": torch.tensor(((0, -1), (1, 2)), dtype=torch.int32),
        "qo_indptr": torch.tensor((0, 2, 5), dtype=torch.int32),
    }

    with pytest.raises(
        ValueError,
        match=r"requires the live wrapper's plan state.*flashinfer\.fi_trace",
    ):
        wrapper.run.fi_trace(**kwargs)
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        fi_trace(wrapper.run, **kwargs)

    # A successful plan publishes one frozen state. A minimal stand-in keeps
    # this trace dispatcher contract test CPU-only.
    wrapper._plan_state = type(
        "_TraceDecodePlanState",
        (),
        {
            "page_size": 32,
            "storage_page_size": 32,
            "use_packed_q": True,
            "seq_len_q": 3,
            "output_dtype": torch.float16,
            "mask_type": "causal",
            "window_left": -1,
            "max_kv_len": 64,
            "kv_prefix_mode": "dynamic",
            "kv_lengths_mode": "dynamic",
            "planned_seq_lens_host": None,
            "planned_seq_lens_device": None,
        },
    )()
    for required_name in (
        "seq_lens",
        "block_tables",
        "qo_indptr",
    ):
        incomplete_kwargs = dict(kwargs)
        incomplete_kwargs.pop(required_name)
        with pytest.raises(ValueError, match=required_name):
            fi_trace(wrapper.run, **incomplete_kwargs)

    defn = fi_trace(wrapper.run, **kwargs)
    assert defn["name"].startswith("prims_ts_decode_wrapper_tuple_fp16_output_packed_q")
    assert defn["inputs"]["q"]["shape"] == [
        "total_q",
        "num_qo_heads",
        "head_dim",
    ]
    assert defn["outputs"]["output"] == {
        "shape": ["total_q", "num_qo_heads", "head_dim"],
        "dtype": "float16",
        "param": "out",
    }
    assert defn["axes"]["max_seq_len_q"]["value"] == 3
    assert defn["axes"]["max_kv_len"]["value"] == 64
    assert "mask:causal" in defn["tags"]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _decode_resources_by_name(resource_dependency_graph):
    resources_by_id = {}
    for resource, dependencies in resource_dependency_graph.items():
        resources_by_id[id(resource)] = resource
        for dependency in dependencies:
            resources_by_id[id(dependency)] = dependency
    return {resource.name: resource for resource in resources_by_id.values()}


def _make_contiguous_keeps_config(*, dtype, tile_size_q: int, headdim: int = 128):
    return make_decode_config(
        headdim=headdim,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": tile_size_q,
            "groups_tokens_heads_q": False,
        },
        seq_len_q=1,
        seq_len_kv=4096,
        batch_size=8,
        num_heads_q=4 * tile_size_q,
        num_heads_kv=4,
        qkv_dtype=dtype,
        o_dtype=Float16 if dtype == Float8E4M3FN else dtype,
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type="dense",
        auto_tuner=False,
    )


def _make_contiguous_kv256_config(
    *,
    dtype=BFloat16,
    persistent: bool | None = None,
    config_args: dict[str, object] | None = None,
    split_kv_mode: str = "disabled",
    splits_kv: int = 1,
):
    """Build the qualified Q64/KV256 profile for schedule-level tests."""

    args = {
        "use_keeps_mma_ab": True,
        "tile_size_q": 64,
        "tile_size_kv": 256,
        "groups_tokens_heads_q": True,
    }
    if persistent is not None:
        args["use_persistent_scheduler"] = persistent
    if config_args is not None:
        args.update(config_args)
    return make_decode_config(
        headdim=128,
        args=args,
        seq_len_q=64,
        seq_len_kv=4096,
        batch_size=1,
        num_heads_q=32,
        num_heads_kv=32,
        qkv_dtype=dtype,
        o_dtype=dtype,
        qkv_layout="contiguousKv",
        split_kv_mode=split_kv_mode,
        splits_kv=splits_kv,
        mask_type="dense",
        auto_tuner=False,
    )


def _make_paged_window_crossing_config(*, page_size: int):
    """Build two visible K tiles whose page IDs cross the 32-ID window."""

    pages_per_tile = 128 // page_size
    page_window_tiles = 32 // pages_per_tile
    return make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 64,
            "groups_tokens_heads_q": False,
        },
        seq_len_q=1,
        seq_len_kv=(page_window_tiles + 1) * 128,
        batch_size=1,
        num_heads_q=64,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=page_size,
        split_kv_mode="disabled",
        splits_kv=1,
        sliding_window_causal=True,
        attention_window_size=256,
        mask_type="causal",
        auto_tuner=False,
    )


def _build_decode_resources(cfg):
    cfg.total_kv_tiles = 32
    native_paged_kwargs = {}
    if cfg.uses_scattered_page_route:
        native_paged_kwargs = {
            "use_native_paged_kv": True,
            "tma_desc_k": object(),
            "tma_desc_v": object(),
            "page_idx_kv": object(),
            "page_table_stride": 513,
            "page_table_capacity": 513,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        (
            _tasks,
            resource_dependency_graph,
            _dma_consumer_release_labels,
            smem_allocator,
            tmem_allocator,
            _correction_resources,
        ) = _build_decode_gen_schedule(
            cfg,
            total_kv_tiles=cfg.total_kv_tiles,
            # Resource construction stores but does not dereference this marker.
            tma_desc_q=object(),
            **native_paged_kwargs,
        )
    return (
        _decode_resources_by_name(resource_dependency_graph),
        smem_allocator,
        tmem_allocator,
    )


def _assert_decode_smem_within_capacity(cfg, smem_allocator) -> None:
    unified_smem_bytes = (
        _align_up(smem_allocator.total_smem_bytes, 8)
        + smem_allocator.barrier_smem_bytes
    )
    launch_smem_bytes = _align_up(unified_smem_bytes, cfg.stensor_align)
    assert launch_smem_bytes <= cutlass_utils.get_smem_capacity_in_bytes("sm_100")


@pytest.mark.parametrize("page_size", (4, 16, 32, 64, 128))
def test_attention_ts_decode_page_offsets_cross_window_schedule_is_safe(
    page_size: int,
) -> None:
    """Exhaustively check the schedule when page loading crosses ID 32."""

    cfg = _make_paged_window_crossing_config(page_size=page_size)
    pages_per_tile = cfg.tile_size_kv // page_size
    page_window_tiles = 32 // pages_per_tile
    seq_len_kv = (page_window_tiles + 1) * cfg.tile_size_kv
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_decode_task_manager(
            cfg,
            seq_len_kv=seq_len_kv,
            batch_size=1,
            num_heads_kv=1,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )

    assert cfg.static_num_skipped_kv_tiles == page_window_tiles - 1
    assert cfg.total_kv_tiles == 2
    assert cfg.static_num_skipped_kv_tiles * pages_per_tile == 32 - pages_per_tile
    assert (cfg.static_num_skipped_kv_tiles + 1) * pages_per_tile == 32


@pytest.mark.parametrize(
    ("tp_size", "num_qo_heads", "num_kv_heads"),
    _Q_TOKEN_KV_BLOCK_SPARSE_TP_HEAD_GEOMETRIES,
    ids=("tp1", "tp2", "tp4", "tp6", "tp8", "tp12", "tp24"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PAGE4_PRIMTS_GPU
def test_attention_ts_decode_page4_q_token_kv_block_sparse_causal_all_tp_geometries(
    tp_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> None:
    """Cover every valid QToken-KvBlock-Sparse-Attention TP shape and every compacted tail length."""

    assert num_qo_heads == 24 // tp_size
    assert num_kv_heads == max(1, 2 // tp_size)
    case = _make_decode_case(
        kv_lens=(1, 3, 4, 2047, 2048, 2049, 2050, 2051),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=256,
        seq_len_q=1,
        page_size=4,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type="causal",
        device="cuda",
        seed=41000 + num_qo_heads + num_kv_heads,
    )

    _exercise_auto_case(case)


@pytest.mark.arch_blackwell
@_REQUIRES_PAGE4_PRIMTS_GPU
def test_attention_ts_decode_prepared_dynamic_block_table_eager_and_graph() -> None:
    """Prepared launches support padded page rows and dynamic Q storage."""

    storage_page_size = 16
    case = _make_decode_case(
        kv_lens=(2048,) * 8,
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        seq_len_q=1,
        page_size=4,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type="causal",
        device="cuda",
        seed=41505,
    )
    packed_cache = _pack_page4_cache_into_storage_pages(
        case.k_cache,
        case.v_cache,
        storage_page_size=storage_page_size,
    )
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        4,
    )
    workspace_size = get_prims_ts_batch_decode_workspace_size(
        case.q.shape[0],
        case.q.shape[1],
        packed_cache[0].shape[1],
        case.q.shape[2],
        4,
        2051,
        q_dtype=case.q.dtype,
        kv_dtype=packed_cache[0].dtype,
        out_dtype=case.output_dtype,
        mask_type="causal",
        storage_page_size=storage_page_size,
        device="cuda",
    )
    workspace = torch.full((workspace_size,), 0x55, dtype=torch.int8, device="cuda")
    representative_out = torch.empty_like(case.q, dtype=case.output_dtype)
    block_table = _dense_block_table_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_indices,
        min_num_pages=(2051 + 3) // 4,
    )
    padded_table = torch.full(
        (block_table.shape[0], block_table.shape[1] + 17),
        -1,
        dtype=torch.int32,
        device=block_table.device,
    )
    padded_table[:, : block_table.shape[1]].copy_(block_table)
    block_table = padded_table[:, : block_table.shape[1]]
    plan = prepare_prims_ts_batch_decode_with_kv_cache(
        case.q,
        packed_cache,
        workspace,
        block_table,
        seq_lens,
        2051,
        out=representative_out,
        out_dtype=case.output_dtype,
        mask_type="causal",
        page_size=4,
    )
    # This BS8/Hkv1 geometry resolves to fused split-KV on GB300. Starting
    # from nonzero workspace bytes proves preparation owns the sole explicit
    # counter initialization.
    assert plan._workspace.split_kv_counter.numel() > 1
    assert not torch.count_nonzero(plan._workspace.split_kv_counter).item()

    dynamic_q = case.q.clone()
    eager_out = torch.empty_like(representative_out)
    assert (
        plan.run(
            dynamic_q,
            out=eager_out,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
        )
        is eager_out
    )
    _assert_case_correct(eager_out, case)
    assert not torch.count_nonzero(plan._workspace.split_kv_counter).item()

    graph_out = torch.empty_like(representative_out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = plan.run(
            dynamic_q,
            out=graph_out,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
        )
    assert captured is graph_out
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(graph_out, case)
    assert not torch.count_nonzero(plan._workspace.split_kv_counter).item()

    with pytest.raises(ValueError, match="query must preserve"):
        plan.run(
            dynamic_q[:4],
            out=graph_out,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
        )


@pytest.mark.parametrize(
    "physical_layout",
    ("compact", "kv-packed-hnd", "kv-packed-nhd"),
    ids=("compact", "packed-hnd", "packed-nhd"),
)
@pytest.mark.parametrize(
    ("tp_size", "num_qo_heads", "num_kv_heads"),
    _Q_TOKEN_KV_BLOCK_SPARSE_TP_HEAD_GEOMETRIES,
    ids=("tp1", "tp2", "tp4", "tp6", "tp8", "tp12", "tp24"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PAGE4_PRIMTS_GPU
def test_attention_ts_decode_page4_encoded_subpages_all_tp_geometries(
    tp_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    physical_layout: str,
) -> None:
    """Decode page-4 locators across cache layouts for every valid TP."""

    assert num_qo_heads == 24 // tp_size
    assert num_kv_heads == max(1, 2 // tp_size)
    storage_page_size = 16
    kv_lens = (1, 3, 4, 2047, 2048, 2049, 2050, 2051)
    case = _make_decode_case(
        kv_lens=kv_lens,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=256,
        seq_len_q=1,
        page_size=4,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type="causal",
        device="cuda",
        seed=42000 + num_qo_heads + num_kv_heads,
    )
    packed_cache = _pack_page4_cache_into_storage_pages(
        case.k_cache,
        case.v_cache,
        storage_page_size=storage_page_size,
        physical_layout=physical_layout,
    )
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        4,
    )
    workspace_size = get_prims_ts_batch_decode_workspace_size(
        len(kv_lens),
        num_qo_heads,
        num_kv_heads,
        256,
        4,
        2051,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        mask_type="causal",
        storage_page_size=storage_page_size,
        device="cuda",
    )
    workspace = torch.zeros(workspace_size, dtype=torch.int8, device="cuda")
    output = torch.empty_like(case.q)
    block_table = _dense_block_table_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_indices,
        min_num_pages=(2051 + 3) // 4,
    )
    result = prims_ts_batch_decode_with_kv_cache(
        case.q,
        packed_cache,
        workspace,
        block_table,
        seq_lens,
        2051,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out=output,
        out_dtype=torch.bfloat16,
        mask_type="causal",
        page_size=4,
    )

    assert result is output
    _assert_case_correct(output, case)
    one_shot = batch_decode_with_paged_kv_cache(
        case.q,
        packed_cache,
        block_table,
        seq_lens,
        mask_type="causal",
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        page_size=4,
    )
    _assert_case_correct(one_shot, case)


@pytest.mark.arch_blackwell
@_REQUIRES_PAGE4_PRIMTS_GPU
@pytest.mark.parametrize("group_size", (2, 4, 5))
def test_attention_ts_decode_grouped_keeps_split_reduction(
    monkeypatch,
    group_size: int,
) -> None:
    """Merge grouped BF16 rows with the standard split-KV reducer."""

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    kv_tokens = 2528
    storage_page_size = 16
    issuers = 8
    cfg = fmha_decode_config.make_decode_config(
        headdim=256,
        args={
            "use_keeps_mma_ab": True,
            "use_q_token_kv_block_sparse_route": True,
            "groups_tokens_heads_q": True,
            "tile_size_q": 64,
            "tile_size_kv": 128,
            "head_dim_per_stage_kv": 128,
            "num_insts_kv": 1,
            "o_stages": 1,
            "use_persistent_scheduler": False,
            "correction_num_warps": 4,
            "mma_warp_idx": 12,
            "page_offsets_warp_idx": 13,
            "load_warp_idx": 16,
            "load_num_warps": issuers,
        },
        seq_len_q=group_size,
        seq_len_kv=kv_tokens,
        batch_size=1,
        num_heads_q=12,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=4,
        storage_tokens_per_page=storage_page_size,
        split_kv_mode="gmem_reduction_with_separate_kernel",
        splits_kv=2,
        max_splits_kv=2,
        mask_type="causal",
        auto_tuner=False,
    )
    assert cfg.splits_kv == 2
    assert cfg.use_separate_reduction_kernel
    spec = decode_module._decode_launch_spec_from_config(
        cfg,
        batch_size=1,
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        seq_len_q=group_size,
        max_active_clusters=(
            fmha_decode_config.get_max_active_clusters_for_cluster_size(1)
        ),
    )
    monkeypatch.setattr(
        decode_module,
        "_resolve_decode_launch_spec",
        lambda *_args, **_kwargs: spec,
    )
    decode_module._get_compiled_decode.cache_clear()

    case = _make_decode_case(
        kv_lens=(kv_tokens,),
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        seq_len_q=group_size,
        page_size=4,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type="causal",
        device="cuda",
        seed=42528,
    )
    packed_cache = _pack_page4_cache_into_storage_pages(
        case.k_cache,
        case.v_cache,
        storage_page_size=storage_page_size,
    )
    page_memberships = torch.full_like(
        case.paged_kv_indices,
        (1 << group_size) - 1,
    )
    seq_lens = torch.full((1,), kv_tokens, dtype=torch.int32, device="cuda")
    workspace_size = get_prims_ts_batch_decode_workspace_size(
        1,
        12,
        1,
        256,
        4,
        kv_tokens,
        seq_len_q=group_size,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        mask_type="causal",
        storage_page_size=storage_page_size,
        device="cuda",
    )
    output = torch.empty_like(case.q)
    block_table = _dense_block_table_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_indices,
    )
    q_token_kv_block_sparse_page_memberships = (
        _packed_q_token_kv_block_sparse_page_memberships_from_csr(
            case.paged_kv_indptr,
            page_memberships,
        )
    )
    workspace = torch.zeros(workspace_size, dtype=torch.uint8, device="cuda")
    try:
        plan, prepared_output = decode_module._prepare_prims_ts_batch_decode_plan(
            case.q,
            packed_cache,
            workspace,
            block_table,
            seq_lens,
            kv_tokens,
            seq_len_q=group_size,
            qo_indptr=None,
            max_seq_len_q=None,
            out=output,
            out_dtype=torch.bfloat16,
            mask_type="causal",
            window_left=-1,
            kv_layout="HND",
            page_size=4,
            q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
            use_q_token_kv_block_sparse_route=True,
        )
        result = plan.run(
            case.q,
            out=prepared_output,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
        )
        torch.cuda.synchronize()
    finally:
        decode_module._get_compiled_decode.cache_clear()

    assert result is output
    _assert_case_correct(output, case)


@pytest.mark.arch_blackwell
@_REQUIRES_PAGE4_PRIMTS_GPU
@pytest.mark.parametrize("kv_tokens", (240, 256, 2528))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_attention_ts_decode_q4_keeps_kv128_d256_page_membership(
    monkeypatch,
    kv_tokens: int,
    batch_size: int,
) -> None:
    """Apply each packed page-membership bit to the corresponding Q token."""

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    storage_page_size = 16
    kv_lens = tuple(kv_tokens - 32 * batch_idx for batch_idx in range(batch_size))
    config_args = {
        "use_keeps_mma_ab": True,
        "use_q_token_kv_block_sparse_route": True,
        "groups_tokens_heads_q": True,
        "tile_size_q": 64,
        "tile_size_kv": 128,
        "use_persistent_scheduler": False,
        "head_dim_per_stage_kv": 128,
        "num_insts_kv": 1,
        "o_stages": 1,
        "correction_num_warps": 4,
        "mma_warp_idx": 12,
        "page_offsets_warp_idx": 13,
        "load_warp_idx": 16,
        "load_num_warps": 8,
    }
    cfg = fmha_decode_config.make_decode_config(
        headdim=256,
        args=config_args,
        seq_len_q=4,
        seq_len_kv=kv_tokens,
        batch_size=batch_size,
        num_heads_q=12,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=4,
        storage_tokens_per_page=storage_page_size,
        split_kv_mode="disabled",
        mask_type="causal",
        auto_tuner=False,
    )
    spec = decode_module._decode_launch_spec_from_config(
        cfg,
        batch_size=batch_size,
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        seq_len_q=4,
        max_active_clusters=(
            fmha_decode_config.get_max_active_clusters_for_cluster_size(1)
        ),
    )
    monkeypatch.setattr(
        decode_module,
        "_resolve_decode_launch_spec",
        lambda *_args, **_kwargs: spec,
    )
    decode_module._get_compiled_decode.cache_clear()

    case = _make_decode_case(
        kv_lens=kv_lens,
        num_qo_heads=12,
        num_kv_heads=1,
        head_dim=256,
        seq_len_q=4,
        page_size=4,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type="causal",
        device="cuda",
        seed=42531,
    )
    if kv_tokens == 240:
        case.q.fill_(1)
        case.k_cache.zero_()
        case.v_cache.zero_()
        for token_rank in range(kv_tokens):
            page_rank, token_offset = divmod(token_rank, 4)
            page_id = int(case.paged_kv_indices[page_rank].item())
            case.v_cache[page_id, :, token_offset].fill_((token_rank + 1) / kv_tokens)
        target_token_rank = 128
        target_page_id = int(case.paged_kv_indices[target_token_rank // 4].item())
        case.k_cache[target_page_id, :, target_token_rank % 4].fill_(0.5)
    packed_cache = _pack_page4_cache_into_storage_pages(
        case.k_cache,
        case.v_cache,
        storage_page_size=storage_page_size,
        physical_layout="compact",
    )
    memberships = torch.empty_like(case.paged_kv_indices)
    for batch_idx in range(batch_size):
        page_begin = int(case.paged_kv_indptr[batch_idx].item())
        page_end = int(case.paged_kv_indptr[batch_idx + 1].item())
        page_ranks = torch.arange(
            page_end - page_begin, device="cuda", dtype=torch.int32
        )
        memberships[page_begin:page_end] = torch.bitwise_left_shift(
            torch.ones_like(page_ranks),
            page_ranks.remainder(4),
        )
        assigned_query = page_ranks.remainder(4)
        assigned_causal_end = kv_lens[batch_idx] - 4 + assigned_query + 1
        memberships[page_begin:page_end] *= page_ranks * 4 < assigned_causal_end
    if kv_tokens == 240:
        memberships.fill_(0xF)
    kernel_page_indices = case.paged_kv_indices
    kernel_page_memberships = memberships
    kernel_indptr = case.paged_kv_indptr
    if kv_tokens == 240 and batch_size == 1:
        # Model-facing grouped QToken-KvBlock-Sparse-Attention uses a fixed-capacity dense row. Make its
        # padding maximally hostile by pointing every unused slot at the
        # high-logit membership-full page; seq_lens, not row capacity, must
        # keep those entries inert.
        padding_pages = 4
        kernel_page_indices = torch.cat(
            (
                kernel_page_indices,
                kernel_page_indices[target_token_rank // 4].repeat(padding_pages),
            )
        )
        kernel_page_memberships = torch.cat(
            (
                kernel_page_memberships,
                kernel_page_memberships[target_token_rank // 4].repeat(padding_pages),
            )
        )
        kernel_indptr = torch.tensor(
            (0, kernel_page_indices.numel()),
            dtype=torch.int32,
            device="cuda",
        )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    workspace_size = get_prims_ts_batch_decode_workspace_size(
        batch_size,
        12,
        1,
        256,
        4,
        kv_tokens,
        seq_len_q=4,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        mask_type="causal",
        storage_page_size=storage_page_size,
        device="cuda",
    )
    output = torch.empty_like(case.q)
    block_table = _dense_block_table_from_csr(kernel_indptr, kernel_page_indices)
    q_token_kv_block_sparse_page_memberships = (
        _packed_q_token_kv_block_sparse_page_memberships_from_csr(
            kernel_indptr,
            kernel_page_memberships,
        )
    )
    workspace = torch.zeros(workspace_size, dtype=torch.uint8, device="cuda")
    try:
        plan, prepared_output = decode_module._prepare_prims_ts_batch_decode_plan(
            case.q,
            packed_cache,
            workspace,
            block_table,
            seq_lens,
            kv_tokens,
            seq_len_q=4,
            qo_indptr=None,
            max_seq_len_q=None,
            out=output,
            out_dtype=torch.bfloat16,
            mask_type="causal",
            window_left=-1,
            kv_layout="HND",
            page_size=4,
            q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
            use_q_token_kv_block_sparse_route=True,
        )
        plan.run(
            case.q,
            out=prepared_output,
            bmm1_scale=case.bmm1_scale,
            bmm2_scale=case.bmm2_scale,
        )
        torch.cuda.synchronize()
    finally:
        decode_module._get_compiled_decode.cache_clear()

    reference = torch.empty_like(output, dtype=torch.float32)
    for batch_idx in range(batch_size):
        page_begin = int(case.paged_kv_indptr[batch_idx].item())
        page_end = int(case.paged_kv_indptr[batch_idx + 1].item())
        page_ids = case.paged_kv_indices[page_begin:page_end].long()
        batch_kv_tokens = kv_lens[batch_idx]
        all_keys = (
            case.k_cache[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, 1, 256)[:batch_kv_tokens]
        )
        all_values = (
            case.v_cache[page_ids]
            .permute(0, 2, 1, 3)
            .reshape(-1, 1, 256)[:batch_kv_tokens]
        )
        batch_memberships = memberships[page_begin:page_end]
        token_ranks = torch.arange(batch_kv_tokens, device="cuda")
        for query_idx in range(4):
            member_tokens = (
                (batch_memberships & (1 << query_idx)) != 0
            ).repeat_interleave(4)[:batch_kv_tokens]
            member_tokens &= token_ranks < batch_kv_tokens - 4 + query_idx + 1
            keys = all_keys[member_tokens]
            values = all_values[member_tokens]
            keys = keys.repeat_interleave(12, dim=1)
            values = values.repeat_interleave(12, dim=1)
            scores = torch.einsum(
                "hd,khd->hk", case.q[batch_idx, query_idx].float(), keys.float()
            )
            probabilities = torch.softmax(scores * case.bmm1_scale, dim=-1)
            reference[batch_idx, query_idx] = torch.einsum(
                "hk,khd->hd", probabilities, values.float()
            )
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1e-2)


@pytest.mark.arch_blackwell
@_REQUIRES_PAGE4_PRIMTS_GPU
def test_attention_ts_decode_page4_inert_block_table_row_is_zero() -> None:
    """Reserve locator -1 plus length one for CUDA-graph padding rows."""

    num_qo_heads = 6
    num_kv_heads = 1
    head_dim = 256
    semantic_page_size = 4
    storage_page_size = 16
    q = torch.randn(
        2,
        num_qo_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_cache = torch.randn(
        1,
        num_kv_heads,
        storage_page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.tensor([0, -1], dtype=torch.int32, device="cuda")
    seq_lens = torch.ones(2, dtype=torch.int32, device="cuda")
    max_seq_len = 2051
    workspace_size = get_prims_ts_batch_decode_workspace_size(
        2,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        semantic_page_size,
        max_seq_len,
        q_dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        out_dtype=q.dtype,
        mask_type="causal",
        storage_page_size=storage_page_size,
        device="cuda",
    )
    workspace = torch.zeros(workspace_size, dtype=torch.int8, device="cuda")
    output = torch.empty_like(q)
    block_table = _dense_block_table_from_csr(
        paged_kv_indptr,
        paged_kv_indices,
        min_num_pages=(max_seq_len + semantic_page_size - 1) // semantic_page_size,
    )

    prims_ts_batch_decode_with_kv_cache(
        q,
        (k_cache, v_cache),
        workspace,
        block_table,
        seq_lens,
        max_seq_len,
        out=output,
        mask_type="causal",
        page_size=semantic_page_size,
    )

    expected_active = v_cache[0, :, 0].repeat_interleave(
        num_qo_heads // num_kv_heads,
        dim=0,
    )
    torch.testing.assert_close(output[0], expected_active, rtol=0, atol=0)
    torch.testing.assert_close(output[1], torch.zeros_like(output[1]), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("headdim", (64, 128))
def test_attention_ts_decode_q128_tmem_p_aliases_consumed_s_region(
    dtype,
    headdim: int,
) -> None:
    """Q128 P overlays only its own consumed S region within capacity."""

    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=128,
        headdim=headdim,
    )
    assert cfg.uses_two_inst_tmem_p
    resources, smem_allocator, tmem_allocator = _build_decode_resources(cfg)
    s0 = resources["tmemS0"]._alloc
    s1 = resources["tmemS1"]._alloc
    p0 = resources["smemP0"]._tmem_alloc
    p1 = resources["smemP1"]._tmem_alloc
    output = resources["tmemO"]._alloc

    for name in ("tmemSoftmaxLocal0", "tmemSoftmaxLocal1"):
        stats = resources[name]
        if cfg.keeps_stats_via_smem:
            assert stats._alloc is None
        else:
            assert stats._alloc is not None
    if cfg.streams_tmem_p_fragments:
        # Streamed P fragments overlay S from its first column so every
        # 16-column P store lands on consumed scores; O leads the layout.
        assert cfg.keeps_stats_via_smem
        assert output.offset == 0
        assert s0.offset == output.offset + output.num_columns
        assert p0.offset == s0.offset
        assert p1.offset == s1.offset
    elif cfg.keeps_stats_via_smem:
        assert p0.offset == s0.offset + cfg.tmem_stats_cols
        assert p1.offset == s1.offset + cfg.tmem_stats_cols
        assert output.offset >= s1.offset + s1.num_columns
    else:
        stats0 = resources["tmemSoftmaxLocal0"]._alloc
        stats1 = resources["tmemSoftmaxLocal1"]._alloc
        assert p0.offset == s0.offset
        assert p1.offset == s1.offset
        assert stats0.offset == s1.offset + s1.num_columns
        assert stats1.offset == stats0.offset + stats0.num_columns
        assert output.offset == stats1.offset + stats1.num_columns

    expected_p_cols = cfg.tile_size_kv * cfg.q_dtype_bytes // 4
    assert p0.num_columns == p1.num_columns == expected_p_cols
    assert s0.offset <= p0.offset
    assert p0.offset + p0.num_columns <= s0.offset + s0.num_columns
    assert s1.offset <= p1.offset
    assert p1.offset + p1.num_columns <= s1.offset + s1.num_columns
    assert resources["smemP0"]._alloc is None
    assert resources["smemP1"]._alloc is None
    assert {"smemK0", "smemK1", "smemV0", "smemV1"} <= resources.keys()
    assert tmem_allocator.total_tmem_columns == cfg.tmem_total_cols <= 512
    _assert_decode_smem_within_capacity(cfg, smem_allocator)


def test_attention_ts_decode_kv256_uses_fragment_ready_p_policy() -> None:
    """KV256 publishes four TMEM P fragments without a full/empty FIFO."""

    cfg = _make_contiguous_kv256_config()
    resources, smem_allocator, _tmem_allocator = _build_decode_resources(cfg)

    assert cfg.keeps_stats_via_smem
    assert resources["tmemSoftmaxLocal0"]._alloc is None
    assert resources["tmemSoftmaxLocal1"]._alloc is None
    assert resources["tmemO"]._alloc.offset == 0
    assert resources["smemP0"]._tmem_alloc.offset == resources["tmemS0"]._alloc.offset
    assert resources["smemP1"]._tmem_alloc.offset == resources["tmemS1"]._alloc.offset

    # One score generation advances both P-ready parity and the matching
    # two-stage O completion credit exactly once per instance. These structural
    # invariants keep the hand-managed fragment protocol aligned across
    # persistent work-tile boundaries.
    assert cfg.num_insts_kv == cfg.o_stages == 2
    assert resources["tmemS0"].pipeline_config.num_stages == 1
    assert resources["tmemS1"].pipeline_config.num_stages == 1
    assert resources["tmemO"].pipeline_config.num_stages == cfg.o_stages
    for name in ("smemP0", "smemP1"):
        p = resources[name]
        assert p.pipeline_config is None
        assert p.tmem_o_ref is resources["tmemO"]
        assert isinstance(p._fragment_ready_alloc, SmemAllocation)
        assert p._fragment_ready_alloc.size_bytes == (
            cfg.num_softmax_score_fragments * 8
        )
    _assert_decode_smem_within_capacity(cfg, smem_allocator)


@pytest.mark.parametrize("persistent", (False, True))
def test_attention_ts_decode_streamed_p_fragments_follow_kv_tile(
    persistent: bool,
) -> None:
    """KV256 splits its scores into streamed fragments; dense Q128 does not.

    Streamed profiles share one rolled fragment loop whose geometry follows
    the fragment register count, so the profile property and the fragment
    count are pinned together. They also run their two softmax groups
    unordered. Dense 16-bit Q128/KV128 keeps the complete P row because its
    route loop is not load-bound; only block-sparse Q128 streams. FP8 Q128
    publishes a complete packed row and never streams.
    """

    kv256 = _make_contiguous_kv256_config(persistent=persistent)
    assert kv256.streams_tmem_p_fragments
    assert kv256.softmax_score_fragment_regs == 32
    assert kv256.num_softmax_score_fragments == 4
    assert kv256.kv_block_size % kv256.softmax_score_fragment_regs == 0
    assert not kv256.uses_ordered_softmax_barrier

    q128 = _make_contiguous_keeps_config(dtype=BFloat16, tile_size_q=128)
    assert q128.tile_size_kv == 128 and q128.uses_two_inst_tmem_p
    assert not q128.streams_tmem_p_fragments
    assert q128.num_softmax_score_fragments == 1

    q128_fp8 = _make_contiguous_keeps_config(dtype=Float8E4M3FN, tile_size_q=128)
    assert q128_fp8.uses_two_inst_tmem_p
    assert q128_fp8.num_softmax_score_fragments == 1
    assert not q128_fp8.streams_tmem_p_fragments


def test_attention_ts_decode_kv256_static_skips_unmodeled_fragment_alias_check() -> (
    None
):
    """Keep structural validation enabled when TS cannot model P barriers."""

    cfg = _make_contiguous_kv256_config(persistent=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        task_manager = build_decode_task_manager(
            cfg,
            seq_len_kv=2 * cfg.tile_size_kv,
            batch_size=1,
            num_heads_kv=32,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )

    assert task_manager._exhaustive_deadlock_race_check is False


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type", "message"),
    (
        pytest.param(
            "mma_tile_n_bmm1",
            128,
            ValueError,
            "mma_tile_n_bmm1",
            id="mma-geometry",
        ),
        pytest.param(
            "kv_stages",
            4,
            ValueError,
            "KV256 shared-memory pipeline",
            id="pipeline-capacity",
        ),
        pytest.param(
            "load_warp_idx",
            14,
            ValueError,
            "load_warp_idx",
            id="static-load-role",
        ),
        pytest.param(
            "kv_stages",
            2.5,
            TypeError,
            "kv_stages must be a Python integer",
            id="fractional-kv-stages",
        ),
        pytest.param(
            "q_stages",
            True,
            TypeError,
            "q_stages must be a Python integer",
            id="boolean-q-stages",
        ),
        pytest.param(
            "load_warp_idx",
            13.0,
            TypeError,
            "load_warp_idx must be a Python integer",
            id="fractional-load-role",
        ),
    ),
)
def test_attention_ts_decode_kv256_rejects_incompatible_profile_overrides(
    field_name: str,
    invalid_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject unsupported KV256 geometry, capacity, types, and roles."""

    with pytest.raises(error_type, match=message):
        _make_contiguous_kv256_config(
            config_args={field_name: invalid_value},
        )


@pytest.mark.parametrize(
    ("persistent", "use_attention_sinks", "expected_error"),
    (
        pytest.param(False, False, None, id="static"),
        pytest.param(True, False, None, id="persistent-direct"),
        pytest.param(True, True, None, id="persistent-sinks"),
    ),
)
def test_attention_ts_decode_kv256_explicit_pipeline_depth_contract(
    persistent: bool,
    use_attention_sinks: bool,
    expected_error: str | None,
) -> None:
    """Two K/V stages are legal for every KV256 scheduler.

    The tail exchange owns its own SMEM, so no scheduler needs a spare ring
    stage for it; the pipeline depth is a pure throughput tunable.
    """

    config_args = {
        "kv_stages": 2,
        "use_attention_sinks": use_attention_sinks,
    }
    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            _make_contiguous_kv256_config(
                persistent=persistent,
                config_args=config_args,
            )
        return

    cfg = _make_contiguous_kv256_config(
        persistent=persistent,
        config_args=config_args,
    )
    assert cfg.kv_stages == 2
    assert cfg.use_persistent_scheduler is persistent
    assert cfg.use_attention_sinks is use_attention_sinks

    if not persistent:
        resources, smem_allocator, _tmem_allocator = _build_decode_resources(cfg)
        assert resources["smemKv"].pipeline_config.num_stages == 2
        _assert_decode_smem_within_capacity(cfg, smem_allocator)


@pytest.mark.parametrize("persistent", (False, True))
def test_attention_ts_decode_kv256_uses_dedicated_fragment_exchange(
    persistent: bool,
) -> None:
    """Direct KV256 correction owns a compact exchange outside the K/V ring."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.tmem_corr import (
        TmemCorrResource,
    )

    cfg = _make_contiguous_kv256_config(persistent=persistent)
    tmem_corr1 = TmemCorrResource(
        cfg=cfg,
        inst_id=1,
        pipeline_config=None,
        name="tmemCorr1",
    )
    exchange_alloc = tmem_corr1.get_smem_requirements()[-1]

    # The tail exchanges 128 float4 stats plus one padded D32 fragment per
    # logical output row, so the buffer no longer has to alias the K/V ring
    # and the load warp can stream the next tile while correction finishes.
    assert tmem_corr1._kv_tile_256_exchange_entries() * 4 == 11_264
    assert exchange_alloc.size_bytes == 11_264

    if not persistent:
        _resources, smem_allocator, _tmem_allocator = _build_decode_resources(cfg)
        _assert_decode_smem_within_capacity(cfg, smem_allocator)


@pytest.mark.parametrize(
    ("split_kv_mode", "uses_separate_reduction"),
    (
        pytest.param("gmem_reduction", False, id="fused"),
        pytest.param(
            "gmem_reduction_with_separate_kernel",
            True,
            id="separate",
        ),
    ),
)
def test_attention_ts_decode_kv256_split_uses_compact_exchange(
    split_kv_mode: str,
    uses_separate_reduction: bool,
) -> None:
    """Both KV256 split reducers exchange only 64 logical output rows."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_resources.tmem_corr import (
        TmemCorrResource,
    )

    cfg = _make_contiguous_kv256_config(
        dtype=Float16,
        split_kv_mode=split_kv_mode,
        splits_kv=2,
    )
    assert cfg.use_split_kv
    assert cfg.splits_kv == 2
    assert cfg.use_separate_reduction_kernel is uses_separate_reduction

    tmem_corr1 = TmemCorrResource(
        cfg=cfg,
        inst_id=1,
        pipeline_config=None,
        name="tmemCorr1",
    )
    exchange_alloc = tmem_corr1.get_smem_requirements()[-1]

    assert tmem_corr1._kv_tile_256_exchange_entries() * 4 == 11_264
    assert exchange_alloc.size_bytes == 11_264


def test_attention_ts_decode_kv256_register_launch_bound_is_amortized() -> None:
    """Only long KV256 mainloops pay for dynamic register hand-off."""

    cfg = _make_contiguous_kv256_config()
    assert _decode_min_blocks_per_mp(cfg, 31 * cfg.tile_size_kv) == 0
    assert _decode_min_blocks_per_mp(cfg, 31 * cfg.tile_size_kv + 1) == 1


def test_attention_ts_decode_kv256_register_budget_matches_launch_bound() -> None:
    """KV256 enables its 176/104/56 hand-off only once it is amortized."""

    cfg = make_decode_config(
        headdim=128,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 64,
            "tile_size_kv": 256,
            "groups_tokens_heads_q": True,
        },
        seq_len_q=64,
        seq_len_kv=8192,
        batch_size=1,
        num_heads_q=32,
        num_heads_kv=32,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="contiguousKv",
        split_kv_mode="disabled",
        splits_kv=1,
        mask_type="dense",
        auto_tuner=False,
    )

    short_cfg = replace(cfg, total_kv_tiles=31)
    long_cfg = replace(cfg, total_kv_tiles=32)
    assert (
        short_cfg.softmax_task_num_registers,
        short_cfg.correction_task_num_registers,
        short_cfg.mma_load_task_num_registers,
    ) == (None, None, None)
    assert (
        long_cfg.softmax_task_num_registers,
        long_cfg.correction_task_num_registers,
        long_cfg.mma_load_task_num_registers,
    ) == (152, 152, 56)
    assert cfg.tmem_s_cols == 128
    assert cfg.mma_tile_n_bmm1 == 256


@pytest.mark.parametrize("dtype", (BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("headdim", (64, 128))
def test_attention_ts_decode_q64_keeps_p_in_smem_within_capacity(
    dtype,
    headdim: int,
) -> None:
    """Q64 keeps P in SMEM so the next QK wave can reuse released S."""

    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=64,
        headdim=headdim,
    )
    assert not cfg.uses_tmem_p
    resources, smem_allocator, tmem_allocator = _build_decode_resources(cfg)
    s0 = resources["tmemS0"]._alloc
    s1 = resources["tmemS1"]._alloc
    output = resources["tmemO"]._alloc
    for name in ("tmemSoftmaxLocal0", "tmemSoftmaxLocal1"):
        stats = resources[name]
        if cfg.keeps_stats_via_smem:
            assert stats._alloc is None
        else:
            assert stats._alloc is not None
    if cfg.keeps_stats_via_smem:
        assert output.offset == 0
    else:
        stats0 = resources["tmemSoftmaxLocal0"]._alloc
        stats1 = resources["tmemSoftmaxLocal1"]._alloc
        assert stats0.offset == 0
        assert stats1.offset == stats0.offset + stats0.num_columns
        assert output.offset == stats1.offset + stats1.num_columns
    assert s0.offset == output.offset + output.num_columns
    assert s1.offset == s0.offset + s0.num_columns
    assert s1.offset + s1.num_columns == cfg.tmem_total_cols
    for name in ("smemP0", "smemP1"):
        p = resources[name]
        assert isinstance(p._alloc, SmemAllocation)
        assert p._alloc.size_bytes == cfg.smem_p_tile_bytes
        assert p._tmem_alloc is None
    assert "tmemStatsDone0" not in resources
    assert "tmemStatsDone1" not in resources
    assert tmem_allocator.total_tmem_columns == cfg.tmem_total_cols <= 512
    _assert_decode_smem_within_capacity(cfg, smem_allocator)


@pytest.mark.parametrize("dtype", (BFloat16, Float8E4M3FN))
@pytest.mark.parametrize("tile_size_q", (64, 128))
def test_attention_ts_decode_d256_staged_tmem_p_has_overwrite_gate(
    dtype,
    tile_size_q: int,
) -> None:
    """D256 P remains inside S and retains its overwrite-credit gate."""

    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=tile_size_q,
        headdim=256,
    )
    assert cfg.uses_staged_one_inst_tmem_p
    resources, smem_allocator, tmem_allocator = _build_decode_resources(cfg)
    s = resources["tmemS0"]._alloc
    p = resources["smemP0"]._tmem_alloc

    assert cfg.keeps_stats_via_smem
    assert resources["tmemSoftmaxLocal0"]._alloc is None
    assert p.offset == s.offset + cfg.tmem_stats_cols

    assert "tmemStatsDone0" in resources
    assert "tmemStatsDone1" not in resources
    assert resources["smemP0"]._alloc is None
    assert s.offset <= p.offset
    assert p.offset + p.num_columns <= s.offset + s.num_columns
    assert cfg.tmem_total_cols <= tmem_allocator.total_tmem_columns <= 512
    _assert_decode_smem_within_capacity(cfg, smem_allocator)


@pytest.mark.parametrize(
    ("tile_size_q", "dtype", "headdim"),
    (
        (128, BFloat16, 128),
        (128, Float8E4M3FN, 128),
        (64, BFloat16, 64),
        (64, Float8E4M3FN, 128),
    ),
    ids=("q128-bf16", "q128-fp8", "q64-bf16-d64", "q64-fp8-d128"),
)
def test_attention_ts_decode_keeps_alias_schedule_is_race_free(
    tile_size_q: int,
    dtype,
    headdim: int,
) -> None:
    """Exhaust HEAD, LOOP, and odd TAIL reuse for representative profiles."""

    cfg = _make_contiguous_keeps_config(
        dtype=dtype,
        tile_size_q=tile_size_q,
        headdim=headdim,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        build_decode_task_manager(
            cfg,
            seq_len_kv=3 * cfg.tile_size_kv,
            batch_size=1,
            num_heads_kv=4,
            verbose=False,
            skip_validation=False,
            exhaustive_deadlock_race_check=True,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PAGE4_PRIMTS_GPU
def test_attention_ts_decode_runtime_kv_ceil_div_covers_int32_domain() -> None:
    """Keep runtime K-tile and page ceilings safe through signed Int32 max."""

    int32_max = 2**31 - 1
    for page_size in (4, 16, 32, 64, 128):
        cfg = FmhaDecodeConfig(num_tokens_per_page=page_size)
        host_values = (
            0,
            1,
            page_size - 1,
            page_size,
            page_size + 1,
            int32_max - page_size + 1,
            int32_max - page_size + 2,
            int32_max,
        )
        values = torch.tensor(host_values, device="cuda", dtype=torch.int32)
        output = torch.empty(
            (2 * len(host_values),),
            device="cuda",
            dtype=torch.int32,
        )
        values_ptr = make_ptr(
            Int32,
            values.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        output_ptr = make_ptr(
            Int32,
            output.data_ptr(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )

        _launch_runtime_int32_kv_ceil_div(
            values_ptr,
            output_ptr,
            cfg,
            len(host_values),
        )

        expected = torch.tensor(
            tuple(
                (
                    (seq_len_kv + cfg.tile_size_kv - 1) // cfg.tile_size_kv,
                    max((seq_len_kv + page_size - 1) // page_size - 1, 0),
                )
                for seq_len_kv in host_values
            ),
            dtype=torch.int32,
        )
        torch.testing.assert_close(
            output.cpu().view(-1, 2),
            expected,
            rtol=0,
            atol=0,
        )


def test_attention_ts_decode_run_requires_plan():
    wrapper = BatchDecodePagedTSWrapper()
    with pytest.raises(RuntimeError, match=r"plan\(\) must be called before run\(\)"):
        wrapper.run(None, None, None, None)


def test_attention_ts_decode_run_validate_false_skips_explicit_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The trusted run path canonicalizes and launches without validators."""

    from flashinfer.attention.prims_ts import decode as decode_module

    wrapper = BatchDecodePagedTSWrapper()
    wrapper._plan_state = type(
        "_TrustedDecodePlanState",
        (),
        {
            "use_packed_q": False,
            "workspace": object(),
            "compiled_main": object(),
            "compiled_reducer": None,
            "planned_seq_lens_host": None,
            "planned_seq_lens_device": None,
        },
    )()
    runtime = object()
    runtime_seq_lens = object()
    output = object()

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("explicit validation must be skipped")

    monkeypatch.setattr(
        decode_module,
        "_validate_block_table_metadata",
        fail_validation,
    )
    monkeypatch.setattr(decode_module, "_prepare_decode_runtime", fail_validation)
    monkeypatch.setattr(
        decode_module,
        "_validate_decode_run_metadata_values",
        fail_validation,
    )
    monkeypatch.setattr(
        decode_module,
        "_validate_tensor_does_not_overlap_inputs",
        fail_validation,
    )
    monkeypatch.setattr(
        decode_module,
        "_validate_decode_output_aliasing",
        fail_validation,
    )
    monkeypatch.setattr(
        decode_module,
        "_prepare_decode_runtime_unchecked",
        lambda *_args, **_kwargs: runtime,
    )

    def launch(actual_runtime, *, seq_lens, **_kwargs):
        if actual_runtime is not runtime or seq_lens is not runtime_seq_lens:
            fail_validation()
        return output

    monkeypatch.setattr(decode_module, "_launch_decode", launch)

    assert (
        wrapper.run(
            object(),
            object(),
            runtime_seq_lens,
            object(),
            out=object(),
            validate=False,
        )
        is output
    )


def test_attention_ts_decode_run_validates_control_and_q_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation control is Boolean and Q offsets follow the planned mode."""

    from flashinfer.attention.prims_ts import decode as decode_module

    wrapper = BatchDecodePagedTSWrapper()
    wrapper._plan_state = type(
        "_DecodeQModePlanState",
        (),
        {
            "use_packed_q": True,
            "device": torch.device("cuda:0"),
            "batch_size": 2,
            "planned_seq_lens_host": None,
            "planned_seq_lens_device": None,
        },
    )()
    args = (object(), object(), object(), object())
    with pytest.raises(TypeError, match="validate must be a bool"):
        wrapper.run(*args, validate=1)

    monkeypatch.setattr(
        decode_module,
        "_validate_block_table_metadata",
        lambda *_args, **_kwargs: (torch.device("cuda:0"), 2, 1),
    )
    with pytest.raises(ValueError, match="qo_indptr is required"):
        wrapper.run(*args)

    wrapper._plan_state = type(
        "_DecodeQModePlanState",
        (),
        {
            "use_packed_q": False,
            "device": torch.device("cuda:0"),
            "batch_size": 2,
            "planned_seq_lens_host": None,
            "planned_seq_lens_device": None,
        },
    )()
    with pytest.raises(ValueError, match="qo_indptr cannot be used"):
        wrapper.run(*args, qo_indptr=object())


def test_attention_ts_decode_failed_replan_preserves_published_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed compile cannot tear down the previous complete revision."""

    from flashinfer.attention.prims_ts import decode as decode_module

    wrapper = BatchDecodePagedTSWrapper()
    previous_seq_lens = object()
    previous_state = type(
        "_PreviousDecodePlanState",
        (),
        {
            "planned_seq_lens_host": (128,),
            "planned_seq_lens_device": previous_seq_lens,
        },
    )()
    wrapper._plan_state = previous_state
    fake_spec = type("_DecodeLaunchSpec", (), {"config": object()})()
    monkeypatch.setattr(
        decode_module,
        "_resolve_cuda_device",
        lambda _device: (torch.device("cuda:0"), 0),
    )
    monkeypatch.setattr(
        decode_module,
        "_resolve_decode_launch_spec",
        lambda *_args: fake_spec,
    )
    monkeypatch.setattr(
        decode_module,
        "_make_decode_compile_spec",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        decode_module,
        "_get_compiled_decode",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic compile failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic compile failure"):
        wrapper.plan(torch.device("cuda:0"), 1, 8, 1, 64, 16, 128)
    assert wrapper._plan_state is previous_state
    assert wrapper._plan_state.planned_seq_lens_device is previous_seq_lens


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_plan_snapshots_seq_lens_and_replan_replaces_them() -> None:
    """Plan-owned GPU lengths are immutable snapshots of the host values."""

    case = _make_decode_case(
        kv_lens=(64, 128),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=torch.float16,
        output_dtype=torch.float16,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=20260908,
    )
    plan_seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    ).cpu()
    expected_first = plan_seq_lens.clone()
    wrapper = BatchDecodePagedTSWrapper()
    wrapper.plan(
        case.q.device,
        2,
        8,
        1,
        64,
        32,
        128,
        seq_lens=plan_seq_lens,
    )

    first_state = wrapper._require_plan_state()
    assert first_state.planned_seq_lens_host == (64, 128)
    assert first_state.planned_seq_lens_device is not None
    assert first_state.planned_seq_lens_device.dtype == torch.int32
    assert first_state.planned_seq_lens_device.device == case.q.device
    assert first_state.planned_seq_lens_device.is_contiguous()
    torch.testing.assert_close(
        first_state.planned_seq_lens_device.cpu(),
        expected_first,
        rtol=0,
        atol=0,
    )

    plan_seq_lens.fill_(1)
    torch.testing.assert_close(
        first_state.planned_seq_lens_device.cpu(),
        expected_first,
        rtol=0,
        atol=0,
    )
    output = _run_case(wrapper, case)
    _assert_case_correct(output, case)

    replacement = torch.tensor((32, 96), dtype=torch.int64)
    wrapper.plan(
        case.q.device,
        2,
        8,
        1,
        64,
        32,
        128,
        seq_lens=replacement,
    )
    second_state = wrapper._require_plan_state()
    assert second_state is not first_state
    assert second_state.planned_seq_lens_host == (32, 96)
    assert second_state.planned_seq_lens_device is not None
    assert second_state.planned_seq_lens_device.data_ptr() != (
        first_state.planned_seq_lens_device.data_ptr()
    )
    assert second_state.planned_seq_lens_device.tolist() == [32, 96]
    assert first_state.planned_seq_lens_device.tolist() == [64, 128]


@pytest.mark.parametrize("validate", (True, False), ids=("validated", "trusted"))
@pytest.mark.parametrize(
    ("planned_seq_lens", "runtime_seq_lens", "message"),
    (
        (object(), object(), "seq_lens must be None when plan\\(\\) owns"),
        (None, None, "seq_lens is required when plan\\(\\) does not own"),
    ),
    ids=("both-sources", "neither-source"),
)
def test_attention_ts_decode_run_requires_exactly_one_seq_lens_owner(
    planned_seq_lens: object,
    runtime_seq_lens: object,
    message: str,
    validate: bool,
) -> None:
    """Length ownership is unconditional, including on the trusted path."""

    wrapper = BatchDecodePagedTSWrapper()
    wrapper._plan_state = type(
        "_SeqLensOwnershipPlanState",
        (),
        {
            "planned_seq_lens_host": ((128,) if planned_seq_lens is not None else None),
            "planned_seq_lens_device": planned_seq_lens,
            "kv_prefix_mode": "dynamic",
            "kv_lengths_mode": (
                "planned_uniform_max" if planned_seq_lens is not None else "dynamic"
            ),
        },
    )()

    with pytest.raises(ValueError, match=message):
        wrapper.run(
            object(),
            object(),
            runtime_seq_lens,
            object(),
            validate=validate,
        )


def test_attention_ts_decode_planned_full_dynamic_uses_owned_seq_lens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full-prefix plan owns lengths even when length handling is dynamic."""

    from flashinfer.attention.prims_ts import decode as decode_module

    planned_seq_lens = object()
    runtime = object()
    output = object()
    wrapper = BatchDecodePagedTSWrapper()
    wrapper._plan_state = type(
        "_PlannedFullDynamicDecodePlanState",
        (),
        {
            "use_packed_q": False,
            "workspace": object(),
            "compiled_main": object(),
            "compiled_reducer": None,
            "planned_seq_lens_host": (128, 127),
            "planned_seq_lens_device": planned_seq_lens,
            "kv_prefix_mode": "planned_full",
            "kv_lengths_mode": "dynamic",
        },
    )()
    monkeypatch.setattr(
        decode_module,
        "_prepare_decode_runtime_unchecked",
        lambda *_args, **_kwargs: runtime,
    )

    def launch(actual_runtime, *, seq_lens, **_kwargs):
        assert actual_runtime is runtime
        assert seq_lens is planned_seq_lens
        return output

    monkeypatch.setattr(decode_module, "_launch_decode", launch)

    assert (
        wrapper.run(
            object(),
            object(),
            None,
            object(),
            validate=False,
        )
        is output
    )


def test_attention_ts_decode_plan_owned_validation_uses_host_seq_lens() -> None:
    """Validated runs do not copy the plan-owned GPU lengths back to the host."""

    class _NoDeviceReadback:
        def tolist(self):
            raise AssertionError("plan-owned seq_lens must not be read back")

    state = type(
        "_PlanOwnedDecodeState",
        (),
        {
            "planned_seq_lens_host": (16,),
            "max_kv_len": 16,
            "page_size": 16,
            "storage_page_size": 16,
            "use_packed_q": False,
            "seq_len_q": 1,
            "batch_size": 1,
            "mask_type": "dense",
        },
    )()
    runtime = type(
        "_PlanOwnedDecodeRuntime",
        (),
        {
            "num_physical_pages": 1,
            "q": torch.empty((1, 8, 64)),
        },
    )()
    _validate_decode_run_metadata_values(
        state,
        runtime,
        seq_lens=cast(torch.Tensor, _NoDeviceReadback()),
        block_tables=torch.tensor(((0,),), dtype=torch.int32),
        qo_indptr=None,
    )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_runtime_owned_seq_lens_remain_dynamic_in_graph() -> None:
    """Run-owned lengths may change values and identity across graph replays."""

    case = _make_decode_case(
        kv_lens=(128,),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=1,
        page_size=16,
        qkv_dtype=torch.float16,
        output_dtype=torch.float16,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=20260901,
    )
    wrapper = BatchDecodePagedTSWrapper()
    wrapper.plan(case.q.device, 1, 8, 1, 64, 16, 128)
    policy = dict(wrapper._policy)
    assert policy["kv_prefix_mode"] == "dynamic"
    assert policy["kv_lengths_mode"] == "dynamic"

    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    )
    output = _run_case(wrapper, case, seq_lens=seq_lens)
    _assert_case_correct(output, case)

    graph_out = torch.empty_like(output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(
            wrapper,
            case,
            seq_lens=seq_lens,
            out=graph_out,
            validate=False,
        )
    assert captured is graph_out

    shorter_len = 64
    shorter_page_count = shorter_len // int(case.k_cache.shape[2])
    shorter_case = replace(
        case,
        paged_kv_indptr=torch.tensor(
            (0, shorter_page_count), dtype=torch.int32, device=case.q.device
        ),
        paged_kv_indices=case.paged_kv_indices[:shorter_page_count],
        paged_kv_last_page_len=torch.tensor(
            (int(case.k_cache.shape[2]),),
            dtype=torch.int32,
            device=case.q.device,
        ),
    )
    shorter_case = _with_reference(shorter_case)
    seq_lens.fill_(shorter_len)
    graph.replay()
    torch.cuda.synchronize()
    _assert_case_correct(graph_out, shorter_case)

    rebound_seq_lens = seq_lens.clone()
    assert rebound_seq_lens.data_ptr() != seq_lens.data_ptr()
    rebound = _run_case(wrapper, case, seq_lens=rebound_seq_lens)
    _assert_case_correct(rebound, shorter_case)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_public_interfaces_reject_output_alias():
    max_kv_len = 128
    case = _make_decode_case(
        kv_lens=(max_kv_len,),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=1,
        page_size=16,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=20260718,
    )
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    )
    wrapper = _plan_case(case, max_kv_len=max_kv_len)

    with pytest.raises(ValueError, match="out must not overlap query storage"):
        _run_case(wrapper, case, out=case.q)
    with pytest.raises(ValueError, match="out must not overlap query storage"):
        _run_standalone(
            case,
            seq_lens,
            max_kv_len=max_kv_len,
            out=case.q,
        )


@pytest.mark.parametrize(
    ("runtime_seq_lens", "table_values", "message"),
    (
        ((0, 1), ((0, -101), (1, -102)), r"must be within \[1, 64\]"),
        ((1, 65), ((0, -101), (1, 2)), r"must be within \[1, 64\]"),
        ((64, 64), ((0,), (1,)), "does not have enough columns"),
        ((1, 1), ((0, -101), (8, -102)), "must index the physical K/V cache"),
    ),
    ids=(
        "sequence-empty",
        "sequence-too-long",
        "table-too-narrow",
        "active-page-id-out-of-range",
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_run_rejects_malformed_fixed_metadata_values(
    runtime_seq_lens,
    table_values,
    message,
):
    """Default run validation rejects malformed active fixed-table values."""

    device = torch.device("cuda")
    block_tables = torch.tensor(table_values, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(runtime_seq_lens, dtype=torch.int32, device=device)
    q = torch.empty((2, 8, 128), dtype=torch.float16, device=device)
    kv_cache = torch.empty(
        (4, 2, 1, 32, 128),
        dtype=torch.float16,
        device=device,
    )
    wrapper = BatchDecodePagedTSWrapper()
    wrapper.plan(device, 2, 8, 1, 128, 32, 64)
    with pytest.raises(ValueError, match=message):
        wrapper.run(
            q,
            kv_cache,
            seq_lens,
            block_tables,
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_block_table_structure_accepts_padded_rows() -> None:
    """The native table requires compact rows, not a compact outer stride."""

    device = torch.device("cuda")
    seq_lens = torch.tensor((1, 33), dtype=torch.int32, device=device)
    backing = torch.full((2, 4), -101, dtype=torch.int32, device=device)
    block_tables = backing[:, :2]
    block_tables[0, 0] = 0
    block_tables[1].copy_(torch.tensor((1, 2), dtype=torch.int32, device=device))

    assert block_tables.shape == (2, 2)
    assert block_tables.stride() == (4, 1)
    assert _validate_block_table_metadata(block_tables, seq_lens) == (
        seq_lens.device,
        2,
        2,
    )

    noncompact_rows = backing[:, ::2]
    with pytest.raises(ValueError, match="contiguous within each row"):
        _validate_block_table_metadata(noncompact_rows, seq_lens)

    overlapping_rows = torch.empty(3, dtype=torch.int32, device=device).as_strided(
        (2, 2), (1, 1)
    )
    with pytest.raises(ValueError, match="rows must not overlap"):
        _validate_block_table_metadata(overlapping_rows, seq_lens)


@pytest.mark.parametrize(
    ("seq_lens", "table_values", "message"),
    (
        ((0, 1), ((0, -101), (1, -102)), "values must be positive"),
        ((65, 1), ((0, 1), (2, -102)), "does not have enough columns"),
        ((1, 1), ((0, -101), (8, -102)), "must index the physical K/V cache"),
    ),
    ids=("sequence-empty", "table-too-narrow", "active-page-id-out-of-range"),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_one_shot_rejects_malformed_fixed_metadata(
    seq_lens,
    table_values,
    message,
) -> None:
    """The one-shot surface validates fixed-table values before planning."""

    device = torch.device("cuda")
    q = torch.empty((2, 8, 64), dtype=torch.float16, device=device)
    kv_cache = torch.empty((4, 2, 1, 32, 64), dtype=torch.float16, device=device)
    with pytest.raises(ValueError, match=message):
        batch_decode_with_paged_kv_cache(
            q,
            kv_cache,
            torch.tensor(table_values, dtype=torch.int32, device=device),
            torch.tensor(seq_lens, dtype=torch.int32, device=device),
        )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_one_shot_rejects_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-derived plan bounds are explicitly outside graph capture."""

    device = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="cannot derive host plan bounds"):
        batch_decode_with_paged_kv_cache(
            torch.empty((1, 8, 64), dtype=torch.float16, device=device),
            torch.empty((1, 2, 1, 32, 64), dtype=torch.float16, device=device),
            torch.tensor(((0,),), dtype=torch.int32, device=device),
            torch.tensor((1,), dtype=torch.int32, device=device),
        )


@pytest.mark.parametrize("packed_q", (False, True), ids=("fixed", "packed"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_rejects_per_request_causal_q_longer_than_kv(
    packed_q: bool,
):
    """Reject a short KV row even when another row satisfies the global bound."""

    device = torch.device("cuda")
    page_size = 16
    block_tables = torch.tensor(((0,), (1,)), dtype=torch.int32, device=device)
    seq_lens = torch.tensor((4, 16), dtype=torch.int32, device=device)
    qo_indptr = (
        torch.tensor([0, 5, 6], dtype=torch.int32, device=device) if packed_q else None
    )
    seq_len_q = 1 if packed_q else 8
    max_seq_len_q = 5 if packed_q else None
    q = torch.empty(
        (6, 8, 64) if packed_q else (2, 8, 8, 64),
        dtype=torch.float16,
        device=device,
    )
    kv_cache = torch.empty((2, 2, 1, page_size, 64), dtype=torch.float16, device=device)
    match = r"request 0 has Q=(5|8) and K/V=4"

    wrapper = BatchDecodePagedTSWrapper()
    wrapper.plan(
        device,
        2,
        8,
        1,
        64,
        page_size,
        16,
        max_seq_len_q=max_seq_len_q if packed_q else seq_len_q,
        packed_query=packed_q,
        mask_type="causal",
    )
    with pytest.raises(ValueError, match=match):
        wrapper.run(
            q,
            kv_cache,
            seq_lens,
            block_tables,
            qo_indptr=qo_indptr,
        )
    with pytest.raises(ValueError, match=match):
        batch_decode_with_paged_kv_cache(
            q,
            kv_cache,
            block_tables,
            seq_lens,
            seq_len_q=seq_len_q,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            mask_type="causal",
        )


def test_attention_ts_decode_shared_arch_guard_rejects_unsupported_gpu(monkeypatch):
    """Both public decode workspace APIs enforce the shared architecture guard."""

    from contextlib import nullcontext

    from flashinfer.mla import get_prims_ts_batch_mla_decode_workspace_size

    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (9, 0)
    )
    workspace_queries = (
        lambda: get_prims_ts_batch_decode_workspace_size(
            batch_size=1,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            page_size=32,
            max_seq_len=128,
            device="cuda:0",
        ),
        lambda: get_prims_ts_batch_mla_decode_workspace_size(
            batch_size=1,
            num_heads=8,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            page_size=32,
            max_seq_len=128,
            device="cuda:0",
        ),
    )
    for query in workspace_queries:
        with pytest.raises(
            NotImplementedError,
            match=r"requires an SM100a/B200.*GPU.*\(9, 0\)",
        ):
            query()


def test_attention_ts_decode_auto_launch_persists_only_above_one_sm_wave(
    monkeypatch,
):
    """A partial second CTA wave is the general persistence boundary."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        _select_auto_launch_mode,
    )

    class _FourSmHardware:
        def get_device_multiprocessor_count(self) -> int:
            return 4

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _FourSmHardware)
    modes = tuple(
        _select_auto_launch_mode(
            batch_size=ctas,
            num_heads_kv=1,
            seq_len_kv=128,
            num_q_tiles=1,
            tile_size_kv=128,
        )
        for ctas in (3, 4, 5)
    )
    assert modes == ("static", "static", "persistent")


@pytest.mark.parametrize(
    ("tile_size_kv", "threshold_tiles"),
    (
        pytest.param(128, 16, id="kv128"),
        pytest.param(256, 8, id="kv256"),
    ),
)
def test_attention_ts_decode_auto_split_threshold_scales_with_kv_tile(
    monkeypatch,
    tile_size_kv: int,
    threshold_tiles: int,
) -> None:
    """Equivalent scheduled KV work reaches split at either tile width."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    class _FourSmHardware:
        def get_device_multiprocessor_count(self) -> int:
            return 4

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _FourSmHardware)

    modes = tuple(
        fmha_decode_config._select_auto_launch_mode(
            batch_size=1,
            num_heads_kv=1,
            seq_len_kv=seq_len_kv,
            tile_size_kv=tile_size_kv,
        )
        for seq_len_kv in (
            (threshold_tiles - 1) * tile_size_kv,
            (threshold_tiles - 1) * tile_size_kv + 1,
        )
    )
    assert modes == ("static", "gmem_reduction")


@pytest.mark.parametrize("seq_len_q", (3, 5, 17))
def test_attention_ts_decode_config_accepts_arbitrary_positive_q_length(
    seq_len_q: int,
):
    """The public fixed-Q geometry is not restricted to power-of-two lengths."""

    from cutlass import BFloat16
    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        make_decode_config,
    )

    cfg = make_decode_config(
        headdim=128,
        seq_len_q=seq_len_q,
        seq_len_kv=257,
        batch_size=1,
        num_heads_q=8,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=32,
        mask_type="causal",
        auto_tuner=False,
    )
    assert cfg.max_seq_len_q == seq_len_q
    assert cfg.num_q_ctas == seq_len_q


@pytest.mark.parametrize(
    ("num_heads_q", "num_heads_kv"),
    (
        pytest.param(0, 1, id="zero-q-heads"),
        pytest.param(1, 0, id="zero-kv-heads"),
    ),
)
def test_attention_ts_decode_config_requires_positive_head_counts(
    num_heads_q: int,
    num_heads_kv: int,
) -> None:
    """Reject non-positive head counts before automatic profile selection."""

    with pytest.raises(ValueError, match="head counts must be positive"):
        make_decode_config(
            headdim=128,
            seq_len_q=1,
            seq_len_kv=2048,
            batch_size=1,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            qkv_dtype=BFloat16,
            o_dtype=BFloat16,
            qkv_layout="pagedKv",
            num_tokens_per_page=16,
            auto_tuner=True,
        )


def _make_auto_kv_tile_config(monkeypatch, **overrides):
    """Resolve one automatic KV-tile candidate without compiling a kernel."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: "static",
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda _cluster_size: 148,
    )
    kwargs = {
        "headdim": 128,
        "seq_len_q": 64,
        "seq_len_kv": 1024,
        "batch_size": 1,
        "num_heads_q": 32,
        "num_heads_kv": 32,
        "qkv_dtype": BFloat16,
        "o_dtype": BFloat16,
        "qkv_layout": "pagedKv",
        "num_tokens_per_page": 16,
        "mask_type": "dense",
        "auto_tuner": True,
    }
    return fmha_decode_config.make_decode_config(**{**kwargs, **overrides})


@pytest.mark.parametrize("dtype", (BFloat16, Float16))
def test_attention_ts_decode_auto_config_selects_kv256(monkeypatch, dtype):
    """A Q64 cost-model result promotes both qualified 16-bit dtypes."""

    dtype_args = {
        "seq_len_q": 64,
        "num_heads_q": 128,
        "num_heads_kv": 4,
        "qkv_dtype": dtype,
        "o_dtype": dtype,
        "num_tokens_per_page": 32,
        "mask_type": "causal",
    }
    cfg = _make_auto_kv_tile_config(monkeypatch, **dtype_args)
    explicit_reference = _make_auto_kv_tile_config(
        monkeypatch,
        **dtype_args,
        args={
            "use_keeps_mma_ab": True,
            "tile_size_q": 64,
            "tile_size_kv": 256,
            "groups_tokens_heads_q": True,
        },
    )

    assert cfg.use_keeps_mma_ab is True
    assert cfg.tile_size_q == 64
    assert cfg.tile_size_kv == 256
    assert cfg.groups_tokens_heads_q is True
    assert cfg.q_tokens_per_cta == 2
    register_cfg = replace(cfg, total_kv_tiles=32)
    assert register_cfg.softmax_task_num_registers == 152
    assert register_cfg.correction_task_num_registers == 152
    assert cfg == explicit_reference


@pytest.mark.parametrize(
    (
        "overrides",
        "expected_tile_size_q",
        "expected_tile_size_kv",
        "expected_keeps",
        "expected_splits",
        "expect_q_cost_model",
        "expected_events",
    ),
    (
        pytest.param(
            {
                "seq_len_q": 1,
                "num_heads_q": 32,
                "num_heads_kv": 4,
                "qkv_dtype": Float16,
                "o_dtype": Float16,
                "num_tokens_per_page": 32,
                "mask_type": "causal",
            },
            8,
            128,
            False,
            1,
            False,
            ("q", "kv", "launch"),
            id="sq1-bypasses-q-cost-and-keeps-kv128",
        ),
        pytest.param(
            {
                "seq_len_q": 64,
                "num_heads_q": 128,
                "num_heads_kv": 4,
                "num_tokens_per_page": 32,
                "mask_type": "causal",
            },
            64,
            256,
            True,
            1,
            True,
            ("q", "kv", "launch"),
            id="q-cost-selects-q64-before-kv256",
        ),
        pytest.param(
            {
                "seq_len_q": 8,
                "num_heads_q": 32,
                "num_heads_kv": 1,
                "args": {"use_variable_seqlens_q": True},
            },
            32,
            128,
            False,
            1,
            False,
            ("q", "kv", "launch"),
            id="variable-q",
        ),
        pytest.param(
            {
                "seq_len_q": 1,
                "num_heads_q": 64,
                "num_heads_kv": 4,
                "qkv_dtype": Float16,
                "o_dtype": Float16,
                "num_tokens_per_page": 32,
                "mask_type": "causal",
            },
            16,
            128,
            False,
            1,
            False,
            ("q", "kv", "launch"),
            id="sq1-ratio16-bypasses-q-cost-and-keeps-kv128",
        ),
        pytest.param(
            {
                "headdim": 256,
                "seq_len_q": 2,
                "seq_len_kv": 512,
                "num_heads_q": 16,
                "num_heads_kv": 1,
                "qkv_dtype": Float8E4M3FN,
                "o_dtype": Float16,
                "num_tokens_per_page": 32,
                "mask_type": "causal",
            },
            64,
            128,
            True,
            2,
            True,
            ("q", "kv"),
            id="d256-cost-uses-finalized-one-inst-profile",
        ),
        pytest.param(
            {
                "seq_len_q": 1,
                "num_heads_q": 32,
                "num_heads_kv": 4,
                "args": {
                    "use_keeps_mma_ab": True,
                    "tile_size_q": 64,
                    "tile_size_kv": 256,
                    "groups_tokens_heads_q": True,
                },
            },
            64,
            256,
            True,
            1,
            False,
            ("q", "kv", "launch"),
            id="explicit-kv256",
        ),
    ),
)
def test_attention_ts_decode_auto_kv_tile_policy(
    monkeypatch,
    overrides: dict[str, object],
    expected_tile_size_q: int,
    expected_tile_size_kv: int,
    expected_keeps: bool,
    expected_splits: int,
    expect_q_cost_model: bool,
    expected_events: tuple[str, ...],
) -> None:
    """Cover KV/Q composition at utilization and compatibility boundaries."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    events = []
    launch_kv_tiles = []
    launch_persistent_flags = []
    q_cost_kv_tiles = []
    grouped_q_selections = []
    original_q_selector = fmha_decode_config._apply_auto_grouped_q_mma_config
    original_kv_promoter = fmha_decode_config._try_apply_auto_kv256_profile
    original_launch_selector = fmha_decode_config._apply_auto_launch_mode
    original_make_q_recipe = fmha_decode_config.make_grouped_q_launch_candidate

    def _spy_q_selector(*args, **kwargs):
        events.append("q")
        selected = original_q_selector(*args, **kwargs)
        grouped_q_selections.append(selected)
        return selected

    def _spy_kv_promoter(*args, **kwargs):
        events.append("kv")
        return original_kv_promoter(*args, **kwargs)

    def _spy_make_q_recipe(*args, **kwargs):
        q_cost_kv_tiles.append(kwargs["tile_size_kv"])
        return original_make_q_recipe(*args, **kwargs)

    def _spy_launch_selector(*args, **kwargs):
        events.append("launch")
        launch_kv_tiles.append(args[0].tile_size_kv)
        launch_persistent_flags.append(args[0].use_persistent_scheduler)
        return original_launch_selector(*args, **kwargs)

    monkeypatch.setattr(
        fmha_decode_config,
        "_apply_auto_grouped_q_mma_config",
        _spy_q_selector,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "_try_apply_auto_kv256_profile",
        _spy_kv_promoter,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "make_grouped_q_launch_candidate",
        _spy_make_q_recipe,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "_apply_auto_launch_mode",
        _spy_launch_selector,
    )
    cfg = _make_auto_kv_tile_config(monkeypatch, **overrides)

    assert cfg.tile_size_q == expected_tile_size_q
    assert cfg.tile_size_kv == expected_tile_size_kv
    assert cfg.use_keeps_mma_ab is expected_keeps
    assert cfg.splits_kv == expected_splits
    assert len(grouped_q_selections) == 1
    assert (grouped_q_selections[0] is not None) is expect_q_cost_model
    assert bool(q_cost_kv_tiles) is expect_q_cost_model
    assert set(q_cost_kv_tiles) <= {128}
    assert tuple(events) == expected_events
    expected_launch_kv_tiles = (
        (expected_tile_size_kv,) if "launch" in expected_events else ()
    )
    assert tuple(launch_kv_tiles) == expected_launch_kv_tiles
    assert tuple(launch_persistent_flags) == (False,) * len(expected_launch_kv_tiles)


def test_attention_ts_decode_auto_config_selection_is_device_agnostic(monkeypatch):
    """Kernel config selection relies on the public wrapper's device guard."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _make_auto_kv_tile_config(
        monkeypatch,
        seq_len_q=64,
        num_heads_q=128,
        num_heads_kv=4,
        num_tokens_per_page=32,
        mask_type="causal",
    )

    assert cfg.tile_size_kv == 256


def test_attention_ts_decode_split_work_floor_is_explicit() -> None:
    """Specialized bounded routes may lower the dense split-work floor."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        select_splits_kv,
    )

    inputs = {
        "seq_len_kv": 2051,
        "batch_size": 1,
        "num_heads_kv": 1,
        "tile_size_kv": 128,
        "num_insts_kv": 2,
        "service_capacity": 152,
        "max_splits_kv": 8,
    }
    assert select_splits_kv(**inputs) == 5
    assert select_splits_kv(**inputs, min_loop_iters_per_split=1) == 8
    with pytest.raises(ValueError, match="min_loop_iters_per_split must be positive"):
        select_splits_kv(**inputs, min_loop_iters_per_split=0)


def test_attention_ts_decode_public_sq1_head_band_stays_kv128(monkeypatch) -> None:
    """Keep the legacy public Q8 override outside KV256 selection."""

    from contextlib import nullcontext

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: "gmem_reduction",
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "select_splits_kv",
        lambda **_kwargs: 4,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 148 // cluster_size,
    )
    make_config_calls = []
    kv_q_candidates = []
    original_make_decode_config = fmha_decode_config.make_decode_config
    original_kv_promoter = fmha_decode_config._try_apply_auto_kv256_profile

    def _record_make_decode_config(*args, **kwargs):
        cfg = original_make_decode_config(*args, **kwargs)
        make_config_calls.append(cfg)
        return cfg

    def _record_kv_promoter(*args, q_candidate, **kwargs):
        kv_q_candidates.append(q_candidate)
        return original_kv_promoter(*args, q_candidate=q_candidate, **kwargs)

    monkeypatch.setattr(
        fmha_decode_config,
        "make_decode_config",
        _record_make_decode_config,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "_try_apply_auto_kv256_profile",
        _record_kv_promoter,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    decode_module._resolve_decode_launch_spec.cache_clear()
    try:
        public_spec = decode_module._resolve_decode_launch_spec(
            0,
            1,
            16,
            1,
            128,
            32,
            2048,
            1,
            "float16",
            "float16",
            "float16",
            "HND",
            "causal",
            False,
            -1,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()

    assert len(make_config_calls) == 2
    grouped_cfg, head_band_cfg = make_config_calls
    assert grouped_cfg.tile_size_q == 16
    assert grouped_cfg.groups_tokens_heads_q is True
    assert grouped_cfg.tile_size_kv == 128
    assert head_band_cfg.tile_size_q == 8
    assert head_band_cfg.groups_tokens_heads_q is False
    assert head_band_cfg.tile_size_kv == 128
    assert kv_q_candidates == [None, None]
    assert public_spec.config.tile_size_q == 8
    assert public_spec.config.groups_tokens_heads_q is False
    assert public_spec.config.tile_size_kv == 128


def _resolve_q_token_kv_block_sparse_policy_for_test(
    monkeypatch,
    *,
    batch_size: int = 16,
    num_qo_heads: int = 12,
    num_kv_heads: int = 1,
    group_size: int = 4,
    max_kv_len: int = 2051,
    qkv_dtype: str = "bfloat16",
    output_dtype: str = "bfloat16",
    use_packed_q: bool = False,
    head_dim: int = 256,
    page_size: int = 4,
    mask_type: str = "causal",
    window_left: int = -1,
):
    """Resolve QToken-KvBlock-Sparse-Attention policy without requiring a CUDA device in host tests."""

    from contextlib import nullcontext

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 148 // cluster_size,
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    decode_module._resolve_decode_launch_spec.cache_clear()
    try:
        return decode_module._resolve_decode_launch_spec(
            0,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_kv_len,
            group_size,
            qkv_dtype,
            qkv_dtype,
            output_dtype,
            "HND",
            mask_type,
            use_packed_q,
            window_left,
            16,
            True,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()


@pytest.mark.parametrize(
    (
        "batch_size",
        "num_qo_heads",
        "num_kv_heads",
        "group_size",
        "max_kv_len",
        "qkv_dtype",
        "output_dtype",
        "use_packed_q",
        "expected_tile_size_q",
        "expected_keeps",
        "expected_splits",
    ),
    (
        pytest.param(
            16,
            6,
            1,
            1,
            2051,
            "bfloat16",
            "bfloat16",
            False,
            8,
            False,
            8,
            id="fixed-bf16-q1-split-tile8",
        ),
        pytest.param(
            16,
            12,
            1,
            1,
            2051,
            "bfloat16",
            "bfloat16",
            False,
            16,
            False,
            8,
            id="fixed-bf16-q1-tile16",
        ),
        pytest.param(
            256,
            6,
            1,
            1,
            2051,
            "bfloat16",
            "bfloat16",
            False,
            8,
            False,
            1,
            id="fixed-bf16-q1-direct-tile8",
        ),
        pytest.param(
            16,
            12,
            1,
            2,
            2051,
            "bfloat16",
            "bfloat16",
            False,
            32,
            False,
            5,
            id="fixed-bf16-q2-tile32",
        ),
        pytest.param(
            16,
            12,
            1,
            4,
            2051,
            "bfloat16",
            "bfloat16",
            False,
            64,
            True,
            9,
            id="fixed-bf16-q4-tile64",
        ),
        pytest.param(
            32,
            12,
            1,
            5,
            2051,
            "bfloat16",
            "bfloat16",
            False,
            64,
            True,
            4,
            id="fixed-bf16-q5-tile64",
        ),
        pytest.param(
            128,
            12,
            1,
            2,
            8208,
            "bfloat16",
            "bfloat16",
            True,
            32,
            False,
            1,
            id="packed-bf16-q2-nonsplit",
        ),
        pytest.param(
            16,
            12,
            1,
            2,
            2524,
            "float8_e4m3fn",
            "bfloat16",
            False,
            32,
            False,
            5,
            id="fixed-fp8-q2-split",
        ),
        pytest.param(
            256,
            12,
            1,
            2,
            2524,
            "float8_e4m3fn",
            "float16",
            False,
            64,
            True,
            1,
            id="fixed-fp8-q2-direct-keeps",
        ),
    ),
)
def test_attention_ts_decode_q_token_kv_block_sparse_policy_is_structural(
    monkeypatch,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    group_size: int,
    max_kv_len: int,
    qkv_dtype: str,
    output_dtype: str,
    use_packed_q: bool,
    expected_tile_size_q: int,
    expected_keeps: bool,
    expected_splits: int,
) -> None:
    """Cover the complete fixed-group QToken-KvBlock-Sparse-Attention policy without shape thresholds."""

    spec = _resolve_q_token_kv_block_sparse_policy_for_test(
        monkeypatch,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        group_size=group_size,
        max_kv_len=max_kv_len,
        qkv_dtype=qkv_dtype,
        output_dtype=output_dtype,
        use_packed_q=use_packed_q,
    )
    cfg = spec.config
    heads_q_per_kv = num_qo_heads // num_kv_heads
    group_rows = heads_q_per_kv * group_size

    assert cfg.use_q_token_kv_block_sparse_route
    assert cfg.groups_tokens_heads_q
    assert cfg.tile_size_q == expected_tile_size_q
    structural_tile_size_q = next(
        tile for tile in (8, 16, 32, 64) if group_rows <= tile
    )
    if qkv_dtype == "float8_e4m3fn" and expected_splits == 1:
        assert cfg.tile_size_q == 64
    else:
        assert cfg.tile_size_q == structural_tile_size_q
    assert cfg.q_tokens_per_cta >= group_size
    assert cfg.use_keeps_mma_ab is expected_keeps
    assert cfg.num_insts_kv == (1 if expected_keeps else 2)
    assert cfg.tile_size_kv == 128
    assert cfg.head_dim_per_stage_kv == 128
    assert cfg.use_variable_seqlens_q is use_packed_q
    assert cfg.use_persistent_scheduler is False
    assert cfg.load_num_warps == 8

    assert cfg.splits_kv == cfg.max_splits_kv == expected_splits
    assert cfg.use_split_kv is (expected_splits > 1)
    assert cfg.use_separate_reduction_kernel is (expected_splits > 1)
    assert cfg.use_cluster_smem_reduction is False
    if use_packed_q:
        assert expected_splits == 1
    else:
        base_grid = batch_size * num_kv_heads
        min_loop_iters = 1 if group_size == 1 else 2
        work_bound = math.ceil(max_kv_len / (128 * cfg.num_insts_kv * min_loop_iters))
        max_policy_splits = 8 if group_size == 1 else 16
        assert expected_splits <= min(max_policy_splits, work_bound)
        assert expected_splits == 1 or base_grid * expected_splits <= 148

    expected_partial_o_shape = (
        (
            batch_size,
            num_kv_heads,
            expected_splits,
            group_rows,
            256,
        )
        if expected_splits > 1
        else (1, 1, 1, 1, 1)
    )
    assert spec.scratch_shapes[0] == expected_partial_o_shape


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        pytest.param(
            {"group_size": 3},
            "group_size must be one of",
            id="unsupported-group",
        ),
        pytest.param(
            {"group_size": 5, "num_qo_heads": 16},
            "exceeds the TileQ64/head capacity",
            id="group-exceeds-tileq64",
        ),
        pytest.param(
            {"mask_type": "dense"},
            "causal non-windowed mask",
            id="dense-mask",
        ),
        pytest.param(
            {"window_left": 128},
            "causal non-windowed mask",
            id="sliding-window",
        ),
        pytest.param(
            {"page_size": 16},
            "sparse_block_size=4",
            id="non-page4",
        ),
        pytest.param(
            {"head_dim": 128},
            "head_dim=256",
            id="non-d256",
        ),
        pytest.param(
            {"qkv_dtype": "float16", "output_dtype": "float16"},
            "BF16 Q/K/V/output or FP8",
            id="unsupported-dtype",
        ),
    ),
)
def test_attention_ts_decode_q_token_kv_block_sparse_rejects_unsupported_contracts(
    monkeypatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    """Reject QToken-KvBlock-Sparse-Attention inputs outside the deliberately narrow public contract."""

    with pytest.raises(ValueError, match=message):
        _resolve_q_token_kv_block_sparse_policy_for_test(monkeypatch, **overrides)


def test_attention_ts_decode_fixed_storage_subpages_do_not_infer_q_token_kv_block_sparse(
    monkeypatch,
) -> None:
    """Keep generic fixed multi-Q tables on the storage-subpage route."""

    from contextlib import nullcontext

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 148 // cluster_size,
    )

    class _B200Hardware:
        def get_device_multiprocessor_count(self) -> int:
            return 148

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _B200Hardware)
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    decode_module._resolve_decode_launch_spec.cache_clear()
    try:
        spec = decode_module._resolve_decode_launch_spec(
            0,
            1,
            12,
            1,
            256,
            4,
            35,
            4,
            "bfloat16",
            "bfloat16",
            "bfloat16",
            "HND",
            "causal",
            False,
            -1,
            16,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()

    assert spec.config.has_storage_subpages
    assert spec.config.uses_scattered_page_route
    assert not spec.config.use_q_token_kv_block_sparse_route
    assert not spec.config.uses_q_token_kv_block_sparse_page_route
    assert not spec.config.uses_q_token_kv_block_sparse_page_membership


def test_attention_ts_decode_public_head_band_does_not_reduce_kv_fanout(
    monkeypatch,
) -> None:
    """Keep grouped Q when extra head-band CTAs would reduce KV fanout."""

    from contextlib import nullcontext

    from flashinfer.attention.prims_ts import decode as decode_module
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 148 // cluster_size,
    )

    class _B200Hardware:
        def get_device_multiprocessor_count(self) -> int:
            return 148

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _B200Hardware)
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    decode_module._resolve_decode_launch_spec.cache_clear()
    try:
        spec = decode_module._resolve_decode_launch_spec(
            0,
            3,
            32,
            1,
            128,
            32,
            8192,
            1,
            "float16",
            "float16",
            "float16",
            "HND",
            "causal",
            False,
            -1,
        )
    finally:
        decode_module._resolve_decode_launch_spec.cache_clear()

    cfg = spec.config

    assert cfg.tile_size_q == 32
    assert cfg.groups_tokens_heads_q is True
    assert cfg.tile_size_kv == 128
    assert cfg.use_split_kv is True
    assert cfg.splits_kv == 16
    assert cfg.max_splits_kv == 16
    assert cfg.use_cluster_smem_reduction is True
    assert cfg.use_separate_reduction_kernel is False
    assert cfg.use_persistent_scheduler is False


def test_attention_ts_decode_explicit_kv_does_not_change_sq1_q_policy(
    monkeypatch,
) -> None:
    """Keep explicit KV materialization independent of the SQ1 Q policy."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    monkeypatch.setattr(
        fmha_decode_config,
        "_select_auto_launch_mode",
        lambda **_kwargs: "static",
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "get_max_active_clusters_for_cluster_size",
        lambda cluster_size: 148 // cluster_size,
    )
    common = dict(
        headdim=64,
        seq_len_q=1,
        seq_len_kv=1024,
        batch_size=1,
        num_heads_q=16,
        num_heads_kv=1,
        qkv_dtype=Float16,
        o_dtype=Float16,
        qkv_layout="pagedKv",
        num_tokens_per_page=16,
        mask_type="dense",
        auto_tuner=True,
    )
    implicit = fmha_decode_config.make_decode_config(**common)
    explicit = fmha_decode_config.make_decode_config(
        **common,
        args={"tile_size_kv": 128},
    )

    assert implicit.tile_size_q == explicit.tile_size_q == 16
    assert implicit.groups_tokens_heads_q is explicit.groups_tokens_heads_q is True
    assert implicit.tile_size_kv == explicit.tile_size_kv == 128


@pytest.mark.parametrize(
    "overrides",
    (
        pytest.param({"headdim": 64}, id="head-dim-64"),
        pytest.param({"args": {"tile_size_kv": 128}}, id="explicit-kv128"),
        pytest.param(
            {"qkv_dtype": Float16, "o_dtype": BFloat16},
            id="mixed-16-bit-io",
        ),
    ),
)
def test_attention_ts_decode_auto_config_falls_back_to_kv128(
    monkeypatch,
    overrides,
):
    """An unqualified or explicitly pinned request retains generic KV128."""

    qualified_auto_shape = {
        "seq_len_q": 64,
        "num_heads_q": 128,
        "num_heads_kv": 4,
        "qkv_dtype": Float16,
        "o_dtype": Float16,
        "num_tokens_per_page": 32,
        "mask_type": "causal",
    }
    cfg = _make_auto_kv_tile_config(
        monkeypatch,
        **{**qualified_auto_shape, **overrides},
    )

    assert cfg.tile_size_kv == 128


def test_attention_ts_decode_q_cost_uses_kv128_with_explicit_kv256(
    monkeypatch,
) -> None:
    """Keep explicit KV materialization downstream of the TileQ cost model."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    q_cost_kv_tiles = []
    selected_q_candidates = []
    original_make_q_recipe = fmha_decode_config.make_grouped_q_launch_candidate
    original_kv_promoter = fmha_decode_config._try_apply_auto_kv256_profile

    def _record_q_cost_kv_tile(*args, **kwargs):
        q_cost_kv_tiles.append(kwargs["tile_size_kv"])
        return original_make_q_recipe(*args, **kwargs)

    def _record_q_winner(*args, q_candidate, **kwargs):
        selected_q_candidates.append(q_candidate)
        return original_kv_promoter(*args, q_candidate=q_candidate, **kwargs)

    monkeypatch.setattr(
        fmha_decode_config,
        "make_grouped_q_launch_candidate",
        _record_q_cost_kv_tile,
    )
    monkeypatch.setattr(
        fmha_decode_config,
        "_try_apply_auto_kv256_profile",
        _record_q_winner,
    )

    with pytest.raises(ValueError, match="KV256"):
        _make_auto_kv_tile_config(
            monkeypatch,
            seq_len_q=8,
            seq_len_kv=4096,
            num_heads_q=32,
            num_heads_kv=4,
            qkv_dtype=Float16,
            o_dtype=Float16,
            num_tokens_per_page=32,
            mask_type="causal",
            args={"tile_size_kv": 256},
        )

    assert q_cost_kv_tiles
    assert set(q_cost_kv_tiles) == {128}
    assert [candidate.tile_size_q for candidate in selected_q_candidates] == [8]


@pytest.mark.parametrize(
    ("packed_q", "sliding_window"),
    (
        pytest.param(True, False, id="packed-q"),
        pytest.param(False, True, id="sliding-window"),
        pytest.param(True, True, id="packed-q-sliding-window"),
    ),
)
def test_attention_ts_decode_runtime_q_features_use_structural_persistence(
    monkeypatch,
    packed_q: bool,
    sliding_window: bool,
):
    """Packed-Q and sliding-window grids persist only above one CTA wave."""

    from cutlass import BFloat16
    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    class _FourSmHardware:
        def get_device_multiprocessor_count(self) -> int:
            return 4

    monkeypatch.setattr(fmha_decode_config.utils, "HardwareInfo", _FourSmHardware)
    # Keep this launch-mode unit test on the generic KV128 profile. A selected
    # Q64 16-bit D128 auto candidate uses KV256 and changes the physical
    # Q CTAs used by the wave boundary below.
    config_args: dict[str, object] = {"tile_size_kv": 128}
    if packed_q:
        config_args["use_variable_seqlens_q"] = True
    common = dict(
        headdim=128,
        args=config_args,
        seq_len_q=3,
        seq_len_kv=257,
        num_heads_q=8,
        num_heads_kv=1,
        qkv_dtype=BFloat16,
        o_dtype=BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=32,
        split_kv_mode="disabled",
        mask_type="causal",
        sliding_window_causal=sliding_window,
        attention_window_size=128 if sliding_window else 0,
        auto_tuner=True,
    )
    one_wave = fmha_decode_config.make_decode_config(batch_size=1, **common)
    multi_wave = fmha_decode_config.make_decode_config(batch_size=2, **common)

    assert one_wave.use_persistent_scheduler is False
    assert multi_wave.use_persistent_scheduler is True
    assert multi_wave.use_split_kv is False


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_persistent_d256_graph_reloads_page_ids():
    """Persistence must cover a partial second wave and reload page IDs."""

    num_kv_heads = 4
    service_capacity = _single_cta_wave_capacity()
    batch_size = service_capacity // num_kv_heads + 1
    logical_work = batch_size * num_kv_heads
    assert (batch_size - 1) * num_kv_heads <= service_capacity < logical_work

    case = _make_decode_case(
        # Use the smallest whole batch that occupies a partial second CTA wave
        # on the GPU running this test.
        kv_lens=(257,) * batch_size,
        num_qo_heads=32,
        num_kv_heads=num_kv_heads,
        head_dim=256,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=31100,
    )
    with pytest.raises(
        ValueError,
        match=r"plan seq_lens values must be within \[1, 256\]",
    ):
        _plan_case(case, max_kv_len=256)

    wrapper = _plan_case(case, max_kv_len=257)
    policy = dict(wrapper._policy)
    assert policy["use_persistent_scheduler"] is True
    assert policy["kv_lengths_mode"] == "dynamic"

    eager = _run_case(wrapper, case).clone()
    _assert_case_correct(eager, case)

    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_case(
            wrapper,
            case,
            out=graph_out,
            validate=False,
        )
    assert captured is graph_out

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)

    table_ptr = case.block_tables.data_ptr()
    table_shape = case.block_tables.shape
    table_stride = case.block_tables.stride()
    original_page_ids = case.paged_kv_indices.clone()

    num_physical_pages = case.k_cache.shape[0]
    remapped_page_ids = (original_page_ids + 1) % num_physical_pages
    case.paged_kv_indices.copy_(remapped_page_ids)
    case.block_tables.copy_(remapped_page_ids.view_as(case.block_tables))
    assert case.block_tables.data_ptr() == table_ptr
    assert case.block_tables.shape == table_shape
    assert case.block_tables.stride() == table_stride
    assert not torch.equal(case.paged_kv_indices, original_page_ids)
    remapped_case = _with_reference(case)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_case_correct(graph_out, remapped_case)
    assert not torch.allclose(graph_out.float(), eager.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("kv_len", "expected_kv_lengths_mode"),
    ((256, "planned_uniform_max"), (257, "dynamic")),
)
def test_attention_ts_decode_static_fp8_d128_odd_kv_tail_is_finite(
    kv_len: int,
    expected_kv_lengths_mode: str,
):
    """Static grouped Q8 specializes paired tails and guards unpaired tails."""

    case = _make_decode_case(
        kv_lens=(kv_len,),
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=128,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=_FP8,
        output_dtype=_FP8,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=31103,
    )
    wrapper = _plan_case(case, max_kv_len=kv_len)
    policy = dict(wrapper._policy)
    assert policy["tile_size_q"] == 8
    assert policy["groups_tokens_heads_q"] is True
    assert policy["use_persistent_scheduler"] is False
    assert policy["use_split_kv"] is False
    assert policy["kv_lengths_mode"] == expected_kv_lengths_mode

    first_output = None
    for _ in range(3):
        output = _run_case(wrapper, case).clone()
        _assert_case_correct(output, case)
        if first_output is None:
            first_output = output
        else:
            torch.testing.assert_close(output, first_output, rtol=0, atol=0)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_standalone_graph_reloads_all_live_metadata():
    """Replay reloads Q offsets, lengths, and a padded fixed page table."""

    max_seq_len_q = 8
    max_kv_len = 257
    case = _make_decode_case(
        kv_lens=(65, max_kv_len),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=128,
        seq_len_q=max_seq_len_q,
        page_size=32,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="causal",
        device="cuda",
        seed=31101,
    )
    case, qo_indptr = _pack_decode_case(case, (1, 7))
    table_capacity = int(case.block_tables.shape[1])
    case = replace(
        case,
        block_tables=_block_tables_from_csr(
            case.paged_kv_indptr,
            case.paged_kv_indices,
            table_capacity=table_capacity,
            row_stride=2 * table_capacity,
        ),
    )
    assert case.block_tables.stride() == (2 * table_capacity, 1)
    assert torch.all(case.block_tables[0, 3:] == -1)
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    )
    block_table = _dense_block_table_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_indices,
        min_num_pages=(max_kv_len + case.k_cache.shape[2] - 1) // case.k_cache.shape[2],
    )

    eager = _run_standalone(
        case,
        seq_lens,
        max_kv_len=max_kv_len,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        block_table=block_table,
    )
    _assert_case_correct(eager, case)

    workspace_size = get_prims_ts_batch_decode_workspace_size(
        2,
        case.q.shape[-2],
        case.k_cache.shape[1],
        case.q.shape[-1],
        case.k_cache.shape[2],
        max_kv_len,
        seq_len_q=1,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        q_dtype=case.q.dtype,
        kv_dtype=case.k_cache.dtype,
        out_dtype=case.output_dtype,
        mask_type=case.mask_type,
        kv_layout="HND",
        device=case.q.device,
    )
    workspace = torch.zeros(workspace_size, dtype=torch.int8, device=case.q.device)
    graph_out = torch.full_like(eager, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run_standalone(
            case,
            seq_lens,
            max_kv_len=max_kv_len,
            qo_indptr=qo_indptr,
            max_seq_len_q=max_seq_len_q,
            out=graph_out,
            workspace_buffer=workspace,
            block_table=block_table,
        )
    assert captured is graph_out
    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, eager, rtol=0, atol=0)

    table_ptr = case.block_tables.data_ptr()
    table_shape = case.block_tables.shape
    table_stride = case.block_tables.stride()
    qo_indptr.copy_(torch.tensor((0, 4, 8), dtype=torch.int32, device="cuda"))
    case.paged_kv_indptr.copy_(
        torch.tensor((0, 9, 12), dtype=torch.int32, device="cuda")
    )
    seq_lens.copy_(torch.tensor((257, 65), dtype=torch.int32, device="cuda"))
    original_page_ids = case.paged_kv_indices.clone()
    remapped_page_ids = (original_page_ids + 1) % int(case.k_cache.shape[0])
    case.paged_kv_indices.copy_(remapped_page_ids)
    case.block_tables.fill_(-1)
    case.block_tables[0, :9].copy_(remapped_page_ids[:9])
    case.block_tables[1, :3].copy_(remapped_page_ids[9:])
    assert case.block_tables.data_ptr() == table_ptr
    assert case.block_tables.shape == table_shape
    assert case.block_tables.stride() == table_stride
    assert torch.all(case.block_tables[1, 3:] == -1)
    assert not torch.equal(case.paged_kv_indices, original_page_ids)
    block_table.copy_(
        _dense_block_table_from_csr(
            case.paged_kv_indptr,
            case.paged_kv_indices,
            min_num_pages=block_table.shape[1],
        )
    )
    replay_case = _with_reference(case, qo_indptr=qo_indptr)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    _assert_case_correct(graph_out, replay_case)
    assert not torch.allclose(graph_out.float(), eager.float(), rtol=1e-3, atol=1e-3)


def test_attention_ts_speculative_mask_oracle_distinguishes_tail_visibility():
    common = dict(
        kv_lens=(65,),
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=4,
        page_size=16,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        device="cpu",
        seed=31099,
    )
    dense = _apply_speculative_tail_markers(
        _make_decode_case(mask_type="dense", **common)
    )
    causal = _apply_speculative_tail_markers(
        _make_decode_case(mask_type="causal", **common)
    )
    # MTP/Eagle bottom-right causal hides the last SQ-1 tokens from the first
    # row, then reveals one token per row. DFlash/DSpark dense sees all tails.
    assert float(dense.reference_real[0, 0, 0, 0]) > 4
    assert float(causal.reference_real[0, 0, 0, 0]) == 0
    torch.testing.assert_close(
        causal.reference_real[:, -1], dense.reference_real[:, -1]
    )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_packed_q_sliding_window_public_parity():
    """Use cumulative Q lengths as the sole runtime ragged-Q definition."""

    q_lens = (1, 3, 8)
    max_seq_len_q = max(q_lens)
    case = _make_decode_case(
        kv_lens=(4097, 3071, 2053),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=max_seq_len_q,
        page_size=64,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="tuple",
        mask_type="causal",
        window_left=127,
        device="cuda",
        seed=31100,
    )
    case, qo_indptr = _pack_decode_case(case, q_lens)
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        int(case.k_cache.shape[2]),
    )
    wrapper = _plan_case(
        case,
        max_kv_len=int(seq_lens.max().item()),
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    policy = dict(wrapper._policy)
    assert policy["use_packed_q"] is True
    assert policy["max_seq_len_q"] == max_seq_len_q
    assert policy["window_left"] == 127

    eager = _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=int(seq_lens.max().item()),
        exercise_all_paths=True,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    one_shot = batch_decode_with_paged_kv_cache(
        case.q,
        case.paged_kv_cache,
        case.block_tables,
        seq_lens,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
        mask_type=case.mask_type,
        window_left=case.window_left,
        bmm1_scale=case.bmm1_scale,
        bmm2_scale=case.bmm2_scale,
        out_dtype=case.output_dtype,
    )
    _assert_case_correct(one_shot, case)
    torch.testing.assert_close(one_shot, eager, rtol=0, atol=0)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_packed_q_sliding_window_clc_persistent():
    """Run live packed offsets and sliding bounds through the CLC scheduler."""

    q_lens = tuple((3, 5, 7)[batch_idx % 3] for batch_idx in range(22))
    max_seq_len_q = max(q_lens)
    case = _make_decode_case(
        kv_lens=(257,) * len(q_lens),
        num_qo_heads=8,
        num_kv_heads=1,
        head_dim=64,
        seq_len_q=max_seq_len_q,
        page_size=32,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="causal",
        window_left=127,
        device="cuda",
        seed=31102,
    )
    case, qo_indptr = _pack_decode_case(case, q_lens)
    policy = _exercise_auto_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    assert policy["use_persistent_scheduler"] is True
    assert policy["use_split_kv"] is False


@pytest.mark.parametrize(
    (
        "qkv_dtype",
        "output_dtype",
        "head_dim",
        "seq_len_q",
        "work_tiles_per_batch",
    ),
    (
        pytest.param(_FP8, _FP8, 128, 1, 4, id="fp8-d128-fixed"),
        pytest.param(
            torch.bfloat16,
            torch.bfloat16,
            256,
            2,
            2,
            id="bf16-d256-sliding",
        ),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_clc_persistent_dtype_head_dim_product(
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    head_dim: int,
    seq_len_q: int,
    work_tiles_per_batch: int,
):
    """Cover fixed and sliding CLC across FP8/BF16 and D128/D256."""

    service_capacity = _single_cta_wave_capacity()
    batch_size = service_capacity // work_tiles_per_batch + 1
    logical_work = batch_size * work_tiles_per_batch
    assert (batch_size - 1) * work_tiles_per_batch <= service_capacity < logical_work
    num_kv_heads = 4 if seq_len_q == 1 else 1
    case = _make_decode_case(
        kv_lens=(257,) * batch_size,
        num_qo_heads=8 * num_kv_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=seq_len_q,
        page_size=32,
        qkv_dtype=qkv_dtype,
        output_dtype=output_dtype,
        cache_form="combined",
        mask_type="causal" if seq_len_q > 1 else "dense",
        window_left=127 if seq_len_q > 1 else -1,
        device="cuda",
        seed=31200 + head_dim,
    )
    policy = _exercise_auto_case(case)
    assert policy["use_persistent_scheduler"] is True
    assert policy["use_split_kv"] is False
    assert policy["kv_lengths_mode"] == "dynamic"


@pytest.mark.parametrize(
    (
        "qkv_dtype",
        "seq_len_q",
        "kv_len",
        "batch_work_tiles",
        "window_left",
    ),
    (
        pytest.param(_FP8, 1, 256, 4, 127, id="sq1-sliding-parity-inversion"),
        pytest.param(
            torch.bfloat16,
            145,
            512,
            40,
            -1,
            id="multi-q-causal-parity-inversion",
        ),
    ),
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_persistent_effective_k_domain_parity(
    qkv_dtype: torch.dtype,
    seq_len_q: int,
    kv_len: int,
    batch_work_tiles: int,
    window_left: int,
):
    """Keep Q/window-dependent odd K tails on runtime-safe resources."""

    service_capacity = _single_cta_wave_capacity()
    batch_size = service_capacity // batch_work_tiles + 1
    case = _make_decode_case(
        kv_lens=(kv_len,) * batch_size,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=128,
        seq_len_q=seq_len_q,
        page_size=32,
        qkv_dtype=qkv_dtype,
        output_dtype=qkv_dtype,
        cache_form="combined",
        mask_type="causal",
        window_left=window_left,
        device="cuda",
        seed=31300 + seq_len_q,
    )
    policy = _exercise_auto_case(case, exercise_all_paths=True)
    assert policy["use_persistent_scheduler"] is True
    assert policy["use_split_kv"] is False
    assert policy["kv_lengths_mode"] == "dynamic"


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_static_sliding_domain_uses_runtime_lengths():
    """A direct sliding plan guards the effective domain after leading skips."""

    case = _make_decode_case(
        kv_lens=(256,),
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=128,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=_FP8,
        output_dtype=_FP8,
        cache_form="combined",
        mask_type="causal",
        window_left=127,
        device="cuda",
        seed=31301,
    )
    policy = _exercise_auto_case(case, exercise_all_paths=True)
    assert policy["use_persistent_scheduler"] is False
    assert policy["use_split_kv"] is False
    assert policy["kv_lengths_mode"] == "dynamic"


@pytest.mark.parametrize(
    (
        "case_kwargs",
        "expected_policy",
        "correction_pattern",
        "exercise_all_paths",
    ),
    _FMHA_CASES,
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_compact_variable_k_acceptance(
    case_kwargs,
    expected_policy,
    correction_pattern,
    exercise_all_paths,
):
    """Exercise pairwise dtype/shape/mask policies with runtime-valued K."""

    case = _make_decode_case(device="cuda", **case_kwargs)
    if correction_pattern == "tail":
        case = _apply_speculative_tail_markers(case)
    elif correction_pattern is not None:
        case = _apply_decode_correction_pattern(case, correction_pattern)

    page_size = int(case.k_cache.shape[2])
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        page_size,
    )
    assert seq_lens.tolist() == list(case_kwargs["kv_lens"])
    assert torch.unique(seq_lens).numel() == seq_lens.numel()
    assert int(seq_lens.max().item()) == max(case_kwargs["kv_lens"])
    assert bool((seq_lens[1:] % page_size != 0).all().item())

    wrapper = _plan_case(case, max_kv_len=max(case_kwargs["kv_lens"]))
    policy = dict(wrapper._policy)
    _assert_auto_policy(policy, expected_policy, device=case.q.device)
    expected_kv_lengths_mode = (
        "planned_uniform_max"
        if all(
            length == max(case_kwargs["kv_lens"]) for length in case_kwargs["kv_lens"]
        )
        else "dynamic"
    )
    assert policy["kv_lengths_mode"] == expected_kv_lengths_mode

    _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=max(case_kwargs["kv_lens"]),
        exercise_all_paths=exercise_all_paths,
    )


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("qkv_dtype", (torch.bfloat16, torch.float16))
def test_attention_ts_decode_qfirst_auto_selects_q64_kv256(
    qkv_dtype: torch.dtype,
):
    """Exercise a cost-model Q64 result and derived KV256 public launch."""

    case = _make_decode_case(
        kv_lens=(1024,),
        num_qo_heads=128,
        num_kv_heads=4,
        head_dim=128,
        seq_len_q=64,
        page_size=32,
        qkv_dtype=qkv_dtype,
        output_dtype=qkv_dtype,
        cache_form="combined",
        mask_type="causal",
        device="cuda",
        seed=20260729,
    )

    policy = _exercise_auto_case(case)

    assert policy["tile_size_q"] == 64
    assert policy["tile_size_kv"] == 256
    assert policy["mma_variant"] == "keeps_mma_ab"
    assert policy["use_split_kv"] is False


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    ("qkv_dtype", "kv_len", "num_heads", "forced_mode", "expect_separate"),
    (
        pytest.param(
            torch.float16,
            8192,
            32,
            None,
            True,
            id="fp16-separate-multi-head",
        ),
        pytest.param(
            torch.float16,
            8192,
            16,
            "gmem_reduction",
            False,
            id="fp16-fused",
        ),
        pytest.param(
            torch.bfloat16,
            65536,
            1,
            None,
            True,
            id="bf16-separate-long-kv",
        ),
    ),
)
def test_attention_ts_decode_q64_kv256_split_launch(
    monkeypatch: pytest.MonkeyPatch,
    qkv_dtype: torch.dtype,
    kv_len: int,
    num_heads: int,
    forced_mode: str | None,
    expect_separate: bool,
) -> None:
    """Exercise KV256 split-KV across distinct reducers and fanout shapes."""

    from flashinfer.attention.prims_ts.kernels.fmha_decode import fmha_decode_config

    if forced_mode is not None:
        monkeypatch.setattr(
            fmha_decode_config,
            "select_split_kv_modes",
            lambda **_kwargs: (forced_mode,),
        )

    case = _make_decode_case(
        kv_lens=(kv_len,),
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=128,
        seq_len_q=64,
        page_size=128,
        qkv_dtype=qkv_dtype,
        output_dtype=qkv_dtype,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=20260805,
    )

    policy = _exercise_explicit_kv256_case(monkeypatch, case)

    assert policy["tile_size_kv"] == 256
    assert policy["use_split_kv"] is True
    assert policy["use_separate_reduction_kernel"] is expect_separate


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize(
    (
        "seq_len_q",
        "kv_lens",
        "page_size",
        "cache_form",
        "mask_type",
        "window_left",
        "persistent",
    ),
    (
        pytest.param(
            64,
            (129,),
            16,
            "combined",
            "causal",
            -1,
            False,
            id="single-kv-tile-tail",
        ),
        pytest.param(
            64,
            (769,),
            128,
            "tuple",
            "causal",
            127,
            False,
            id="static-window-kv-tail",
        ),
        pytest.param(
            512,
            # Two, four, and six physical KV tiles produce runtime loop
            # domains 0/1/2, exercising every modulo-three cursor advance in
            # one persistent launch.
            (257, 769, 1281),
            64,
            "combined",
            "dense",
            -1,
            True,
            id="persistent-ragged-rotating-cursor",
        ),
    ),
)
def test_attention_ts_decode_q64_kv256_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    seq_len_q: int,
    kv_lens: tuple[int, ...],
    page_size: int,
    cache_form: str,
    mask_type: str,
    window_left: int,
    persistent: bool,
):
    """Cover KV256 tails under windowed static and ragged persistent work."""

    case = _make_decode_case(
        kv_lens=kv_lens,
        num_qo_heads=32,
        num_kv_heads=32,
        head_dim=128,
        seq_len_q=seq_len_q,
        page_size=page_size,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form=cache_form,
        mask_type=mask_type,
        window_left=window_left,
        device="cuda",
        seed=20260804 + max(kv_lens),
    )

    policy = _exercise_explicit_kv256_case(monkeypatch, case)

    assert policy["tile_size_q"] == 64
    assert policy["tile_size_kv"] == 256
    assert policy["window_left"] == window_left
    assert policy["use_persistent_scheduler"] is persistent
    if persistent:
        assert policy["use_split_kv"] is False


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
@pytest.mark.parametrize("packed_query", (False, True), ids=("fixed", "packed"))
def test_attention_ts_decode_reuses_compiled_topology_across_batch_sizes(
    packed_query: bool,
):
    """One resolved paged-decode topology accepts different batch extents."""

    wrappers = []
    for batch_size in (2, 3):
        case = _make_decode_case(
            kv_lens=(2049,) * batch_size,
            num_qo_heads=16,
            num_kv_heads=2,
            head_dim=128,
            seq_len_q=2 if packed_query else 1,
            page_size=32,
            qkv_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cache_form="combined",
            mask_type="dense",
            device="cuda",
            seed=32900 + batch_size,
        )
        qo_indptr = None
        if packed_query:
            case, qo_indptr = _pack_decode_case(case, (1,) * batch_size)
        wrapper = _plan_case(
            case,
            max_kv_len=2049,
            qo_indptr=qo_indptr,
            max_seq_len_q=1 if packed_query else None,
        )
        output = _run_case(wrapper, case, qo_indptr=qo_indptr)
        _assert_case_correct(output, case)
        wrappers.append(wrapper)

    first_state = wrappers[0]._require_plan_state()
    second_state = wrappers[1]._require_plan_state()
    assert first_state.compiled_main is second_state.compiled_main
    assert first_state.compiled_reducer is second_state.compiled_reducer


@pytest.mark.parametrize("head_dim", (64, 128, 256), ids=lambda value: f"d{value}")
@pytest.mark.parametrize(
    "num_qo_heads_per_kv",
    (1, 8, 15, 16, 32),
    ids=lambda value: f"gqa{value}",
)
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_head_dim_gqa_product(
    head_dim: int,
    num_qo_heads_per_kv: int,
):
    """Cross head width with GQA ratio using automatic runtime-K planning."""

    num_kv_heads = 2
    max_kv_len = 2049
    page_size = 32
    kv_lens = _ragged_lengths(2, max_kv_len, page_size)
    case = _make_decode_case(
        kv_lens=kv_lens,
        num_qo_heads=num_kv_heads * num_qo_heads_per_kv,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=1,
        page_size=page_size,
        qkv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=33000 + head_dim + num_qo_heads_per_kv,
    )
    seq_lens = _seq_lens_from_csr(
        case.paged_kv_indptr,
        case.paged_kv_last_page_len,
        page_size,
    )
    assert seq_lens.tolist() == list(kv_lens)
    assert bool((seq_lens % page_size != 0).all().item())

    wrapper = _plan_case(case, max_kv_len=max_kv_len)
    policy = dict(wrapper._policy)
    _assert_auto_policy(policy, {}, device=case.q.device)
    if policy["groups_tokens_heads_q"]:
        assert int(policy["tile_size_q"]) >= num_qo_heads_per_kv
    else:
        assert int(policy["tile_size_q"]) == 8
        assert num_qo_heads_per_kv > int(policy["tile_size_q"])
    _exercise_public_paths(
        wrapper,
        case,
        seq_lens,
        max_kv_len=max_kv_len,
        exercise_all_paths=False,
    )


@pytest.mark.parametrize("head_dim", (64, 128, 256), ids=lambda value: f"d{value}")
@pytest.mark.parametrize("head_ratio", (33, 65), ids=lambda value: f"gqa{value}")
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_wide_fp8_gqa_accuracy(
    head_dim: int,
    head_ratio: int,
):
    """Check partial Q64/Q128 Keeps tiles against the FP8 reference."""

    case = _make_decode_case(
        kv_lens=(257, 193),
        num_qo_heads=head_ratio,
        num_kv_heads=1,
        head_dim=head_dim,
        seq_len_q=1,
        page_size=32,
        qkv_dtype=_FP8,
        output_dtype=_FP8,
        cache_form="combined",
        mask_type="dense",
        device="cuda",
        seed=33500 + head_dim + head_ratio,
    )
    policy = _exercise_auto_case(case)

    assert policy["mma_variant"] == "keeps_mma_ab"
    assert policy["tile_size_q"] == (64 if head_ratio <= 64 else 128)
    assert policy["groups_tokens_heads_q"] is True


@pytest.mark.parametrize(
    ("qkv_dtype", "output_dtype"),
    (
        pytest.param(torch.bfloat16, torch.bfloat16, id="bf16"),
        pytest.param(_FP8, _FP8, id="fp8"),
    ),
)
@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@pytest.mark.parametrize("seq_len_q", (2, 8), ids=lambda value: f"sq{value}")
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_fixed_q_dtype_mask_product(
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    mask_type: str,
    seq_len_q: int,
):
    """Cross the missing fixed speculative-Q bounds with dtype and mask."""

    case = _make_decode_case(
        device="cuda",
        **_case(
            2,
            2051,
            16,
            128,
            qkv_dtype,
            36000 + seq_len_q + (100 if qkv_dtype == _FP8 else 0),
            num_kv_heads=2,
            seq_len_q=seq_len_q,
            output_dtype=output_dtype,
            mask_type=mask_type,
        ),
    )
    case = _apply_speculative_tail_markers(case)
    _exercise_auto_case(case)


@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_non_power_of_two_q_above_16k_kv():
    """Exercise arbitrary fixed SQ and a KV bound beyond the former 16K cap."""

    case = _make_decode_case(
        device="cuda",
        **_case(
            1,
            32769,
            8,
            64,
            torch.bfloat16,
            36503,
            seq_len_q=3,
            mask_type="causal",
        ),
    )
    _exercise_auto_case(case)


@pytest.mark.parametrize(
    "q_lens",
    (
        pytest.param((1, 2, 1, 2), id="maxsq2"),
        pytest.param((1, 2, 4, 3), id="maxsq4"),
        pytest.param((1, 3, 8, 5), id="maxsq8"),
    ),
)
@pytest.mark.parametrize("mask_type", ("dense", "causal"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_packed_fp8_q_mask_product(
    q_lens: tuple[int, ...],
    mask_type: str,
):
    """Cross packed FP8 static Q bounds with dense and causal masks."""

    max_seq_len_q = max(q_lens)
    case = _make_decode_case(
        device="cuda",
        **_case(
            len(q_lens),
            2051,
            16,
            128,
            _FP8,
            37000 + max_seq_len_q,
            num_kv_heads=2,
            seq_len_q=max_seq_len_q,
            mask_type=mask_type,
        ),
    )
    case = _apply_speculative_tail_markers(case)
    case, qo_indptr = _pack_decode_case(case, q_lens)
    policy = _exercise_auto_case(
        case,
        qo_indptr=qo_indptr,
        max_seq_len_q=max_seq_len_q,
    )
    assert policy["use_packed_q"] is True
    assert policy["max_seq_len_q"] == max_seq_len_q


@pytest.mark.parametrize(
    ("qkv_dtype", "output_dtype", "correction_pattern"),
    (
        pytest.param(torch.float16, torch.float16, "identity", id="fp16"),
        pytest.param(_FP8, torch.float16, None, id="fp8-fp16"),
    ),
)
@pytest.mark.parametrize("head_dim", (64, 256), ids=lambda value: f"d{value}")
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_missing_dtype_head_dim_product(
    qkv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    correction_pattern: str | None,
    head_dim: int,
):
    """Fill unsupported-by-existing-tests dtype and head-width cells."""

    case = _make_decode_case(
        device="cuda",
        **_case(
            2,
            2049,
            16,
            head_dim,
            qkv_dtype,
            38000 + head_dim + (1 if qkv_dtype == _FP8 else 0),
            num_kv_heads=2,
            output_dtype=output_dtype,
        ),
    )
    if correction_pattern is not None:
        case = _apply_decode_correction_pattern(case, correction_pattern)
    _exercise_auto_case(case)


@pytest.mark.parametrize("page_size", (16, 32, 64, 128))
@pytest.mark.parametrize("cache_form", ("combined", "tuple"))
@pytest.mark.arch_blackwell
@_REQUIRES_PRIMTS_GPU
def test_attention_ts_decode_page_cache_sliding_product(
    page_size: int,
    cache_form: str,
):
    """Cross native page/cache forms and one padded outer page stride."""

    case = _make_decode_case(
        device="cuda",
        **_case(
            2,
            2051,
            8,
            64,
            torch.bfloat16,
            39000 + page_size + (1 if cache_form == "tuple" else 0),
            seq_len_q=2,
            page_size=page_size,
            cache_form=cache_form,
            mask_type="causal",
            window_left=127,
        ),
    )
    if page_size == 32 and cache_form == "tuple":
        compact_page_elements = case.k_cache[0].numel()
        padded_page_stride = compact_page_elements + 8

        def with_padded_outer_stride(cache: torch.Tensor) -> torch.Tensor:
            storage = cache.new_empty((cache.shape[0] * padded_page_stride,))
            padded = storage.as_strided(
                cache.shape,
                (padded_page_stride, *cache.stride()[1:]),
            )
            padded.copy_(cache)
            return padded

        padded_k = with_padded_outer_stride(case.k_cache)
        padded_v = with_padded_outer_stride(case.v_cache)
        assert padded_k.stride(0) > compact_page_elements
        assert padded_k.stride(0) * padded_k.element_size() % 16 == 0
        case = replace(
            case,
            paged_kv_cache=(padded_k, padded_v),
            k_cache=padded_k,
            v_cache=padded_v,
        )
        case = _with_reference(case)

    policy = _exercise_auto_case(case)
    assert policy["window_left"] == 127
