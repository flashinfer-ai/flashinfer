"""Host planner for the primary paged-attention backend.

This module models the host-side work decomposition used by FlashInfer's paged
attention kernels:

- choose `CTA_TILE_Q` from packed Q rows,
- choose `kv_chunk_size` on the host,
- emit exact `(request_idx, qo_tile_idx, kv_tile_idx)` worklists,
- emit `merge_indptr` / `o_indptr` for split reduction.

For decode CUDA graphs, this module also builds a capacity-only chunk-policy
LUT from model geometry and graph bucket limits.  The captured device updater
in ``graph_replay`` consumes that LUT from live device lengths; no live length
is read by this host planner.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal
from ._controls import paged_control

import torch

from ._tuning import GqaConfig

_FP8_KV_DTYPE = torch.float8_e4m3fn
_PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 24),
    (1024, 24),
    (2048, 24),
)
_PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES = (
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (16, 1),
    (32, 2),
    (64, 3),
    (128, 6),
    (256, 6),
    (512, 32),
    (1024, 32),
    (2048, 32),
)

_DEFAULT_GRAPH_CTAS_PER_SM = 2
_FP8_PV_REPACK_MAX_RESIDENT_WAVES = 16
_MSA_TOPK = 16
_MSA_BLOCK_TOKENS = 128
_BF16_MINIMAX_DECODE_MAX_CHUNKS = (
    (1, 512, 20),
    (1, 1, 16),
    (2, 513, 13),
    (2, 1, 12),
    (3, 512, 16),
    (3, 1, 14),
    (4, 1, 11),
)
_PagedMode = Literal["decode", "extend", "verify"]


def _merge_backend_supports_split_kv(
    *,
    output_dtype: torch.dtype,
    head_dim_vo: int,
) -> bool:
    return output_dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ) and head_dim_vo in (128, 256)


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _align_up(x: int, y: int) -> int:
    return _ceil_div(x, y) * y


def use_paged_extend_fp8_pv_repack(
    plan: "PagedPlan",
    *,
    resident_ctas_per_sm: int,
    num_sms: int | None = None,
) -> bool:
    """Select CTA-wide FP8 V widening for low-wave dense extend grids."""

    if (
        plan.mode != "extend"
        or plan.dtype != torch.bfloat16
        or plan.kv_dtype != _FP8_KV_DTYPE
        or int(plan.head_dim_qk) != 256
        or int(plan.head_dim_vo) != 256
        or int(plan.page_size) != 64
        or int(plan.cta_tile_q) != 64
        or int(plan.window_left) >= 0
        or bool(plan.msa_block_sparse)
    ):
        return False
    resident_ctas_per_sm = max(int(resident_ctas_per_sm), 1)
    if num_sms is None:
        num_sms = int(
            torch.cuda.get_device_properties(plan.device).multi_processor_count
        )
    num_sms = max(int(num_sms), 1)
    active_ctas = int(plan.num_qo_tiles) * int(plan.num_kv_heads)
    resident_wave_capacity = num_sms * resident_ctas_per_sm
    return active_ctas <= (
        _FP8_PV_REPACK_MAX_RESIDENT_WAVES * resident_wave_capacity
    )


def _msa_decode_chunk_tokens(policy_batch: int) -> int:
    if int(policy_batch) <= 2:
        return 128
    if int(policy_batch) <= 8:
        return 256
    return 512


def _msa_decode_chunk_pages(policy_batch: int, page_size: int) -> int:
    return _ceil_div(_msa_decode_chunk_tokens(policy_batch), int(page_size))


def _msa_effective_pages(cache_len: int, page_size: int) -> int:
    visible_blocks = max(_ceil_div(max(int(cache_len), 1), _MSA_BLOCK_TOKENS), 1)
    selected_blocks = min(_MSA_TOPK, visible_blocks)
    pages_per_block = _ceil_div(_MSA_BLOCK_TOKENS, int(page_size))
    tail_tokens = max(int(cache_len) - (visible_blocks - 1) * _MSA_BLOCK_TOKENS, 1)
    tail_pages = min(max(_ceil_div(tail_tokens, int(page_size)), 1), pages_per_block)
    return max((selected_blocks - 1) * pages_per_block + tail_pages, 1)


def _previous_power_of_two(x: int) -> int:
    x = max(int(x), 1)
    return 1 << (x.bit_length() - 1)


def _decode_graph_chunk_pages_env(name: str) -> int | None:
    raw = paged_control(name)
    if raw is None or raw.strip() == "":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _apply_decode_graph_chunk_pages_debug_policy(chunk_pages: int) -> int:
    forced = _decode_graph_chunk_pages_env(
        "B12X_PAGED_DECODE_GRAPH_CHUNK_PAGES"
    )
    if forced is not None:
        return forced
    minimum = _decode_graph_chunk_pages_env(
        "B12X_PAGED_DECODE_GRAPH_MIN_CHUNK_PAGES"
    )
    if minimum is not None:
        return max(int(chunk_pages), minimum)
    return int(chunk_pages)


def _bf16_minimax_decode_max_chunks(
    *,
    batch: int,
    max_effective_kv_pages: int,
) -> int | None:
    for tuned_batch, min_pages, max_chunks in _BF16_MINIMAX_DECODE_MAX_CHUNKS:
        if batch == tuned_batch and max_effective_kv_pages >= min_pages:
            return max_chunks
    return None


def _cap_decode_graph_chunk_pages(
    *,
    chunk_pages: int,
    max_effective_kv_pages: int,
    max_chunks_per_req: int | None,
) -> int:
    chunk_pages = max(int(chunk_pages), 1)
    if max_chunks_per_req is None:
        return chunk_pages
    max_chunks_per_req = max(int(max_chunks_per_req), 1)
    return max(
        chunk_pages, _ceil_div(max(int(max_effective_kv_pages), 1), max_chunks_per_req)
    )


def _window_start_page(
    *,
    cache_len: int,
    q_len: int,
    window_left: int,
    page_size: int,
) -> int:
    if window_left < 0:
        return 0
    # Paged attention uses the right-aligned causal convention.  For a request
    # with cached length K and query length Q, the first query row can see up to
    # key index K - Q.  Start from that earliest row so every later row remains
    # covered by the request-level page span.
    first_causal_key = max(int(cache_len) - max(int(q_len), 1), 0)
    first_window_key = max(first_causal_key - int(window_left), 0)
    return first_window_key // int(page_size)


def _graph_max_batch_size_if_split(
    *,
    device: torch.device,
    num_kv_heads: int,
    graph_ctas_per_sm: int,
) -> int:
    blocks_per_sm = int(graph_ctas_per_sm)
    if blocks_per_sm <= 0:
        raise ValueError("graph_ctas_per_sm must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return max((num_sms * blocks_per_sm) // num_kv_heads, 1)


def decode_graph_max_chunks_per_request_budget(
    *,
    device: torch.device,
    num_kv_heads: int,
    batch: int,
    graph_ctas_per_sm: int,
    query_tiles_per_request: int = 1,
) -> int:
    max_batch_size_if_split = _graph_max_batch_size_if_split(
        device=device,
        num_kv_heads=num_kv_heads,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )
    work_items_per_chunk = max(int(batch), 1) * max(
        int(query_tiles_per_request), 1
    )
    return max(max_batch_size_if_split // work_items_per_chunk, 1)


def _laguna_page128_one_wave_chunk_budget(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    query_tiles_per_request: int,
) -> int | None:
    """Return the capture-static one-wave budget for the KV128 Laguna family."""

    if not (
        q_dtype == torch.bfloat16
        and kv_dtype == _FP8_KV_DTYPE
        and int(num_q_heads) == 36
        and int(num_kv_heads) == 4
        and int(head_dim_qk) == 128
        and int(head_dim_vo) == 128
        and int(page_size) == 128
        and int(batch) == 1
        and tuple(torch.cuda.get_device_capability(device)) == (12, 0)
    ):
        return None
    # One work item launches one CTA per KV head.  Use the physical SM count,
    # not a model-specific constant, so the fixed graph ends exactly at one
    # machine wave on every SM120 SKU.  Residency remains a separate capacity
    # limit; this policy deliberately targets one of the two resident waves.
    return decode_graph_max_chunks_per_request_budget(
        device=device,
        num_kv_heads=num_kv_heads,
        batch=batch,
        graph_ctas_per_sm=1,
        query_tiles_per_request=query_tiles_per_request,
    )


def _scale_chunk_budget_for_sm_count(
    chunks_at_reference_count: int,
    *,
    num_sms: int,
    reference_sm_count: int,
) -> int:
    return max(
        (
            int(chunks_at_reference_count) * int(num_sms)
            + int(reference_sm_count) // 2
        )
        // int(reference_sm_count),
        1,
    )


def _h256_gqa8_decode_chunk_budget(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    max_effective_kv_pages: int,
    window_left: int,
) -> int | None:
    """Return the SM-scaled split budget for the H256 GQA8 contract."""

    if not (
        q_dtype == torch.bfloat16
        and kv_dtype in (torch.bfloat16, _FP8_KV_DTYPE)
        and int(num_q_heads) == 8
        and int(num_kv_heads) == 1
        and int(head_dim_qk) == 256
        and int(head_dim_vo) == 256
        and int(page_size) == 64
        and int(batch) == 1
        and (
            224 <= int(max_effective_kv_pages) <= 288
            or 480 <= int(max_effective_kv_pages) <= 544
            or 992 <= int(max_effective_kv_pages) <= 1056
        )
        and int(window_left) < 0
        and int(torch.cuda.get_device_capability(device)[0]) == 12
    ):
        return None
    # Preserve the measured chunks-per-SM knees while the device LUT adapts to
    # every live length captured by the same graph.
    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    reference_sm_count = 48
    max_pages = int(max_effective_kv_pages)
    if 224 <= max_pages <= 288:
        reference_chunks = 19 if kv_dtype == torch.bfloat16 else 22
    elif 480 <= max_pages <= 544:
        reference_chunks = 14 if kv_dtype == torch.bfloat16 else 18
    elif 992 <= max_pages <= 1056:
        reference_chunks = 17 if kv_dtype == torch.bfloat16 else 18
    else:
        return None
    return _scale_chunk_budget_for_sm_count(
        reference_chunks,
        num_sms=num_sms,
        reference_sm_count=reference_sm_count,
    )


def _h256_one_wave_decode_chunk_budget(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    num_kv_heads: int,
    query_tiles_per_request: int,
    window_left: int,
) -> int | None:
    """Cap dense H256 decode at one wave when it still fills most SMs.

    The H256 kernel is occupancy-heavy enough that a second captured wave adds
    merge and scheduling work without helping when the regular one-wave grid
    already covers the machine.  Keep two-wave capacity for batch/head
    geometries whose integer per-request split would leave more than one
    quarter of the SMs idle.
    """

    if not (
        q_dtype == torch.bfloat16
        and kv_dtype in (torch.bfloat16, _FP8_KV_DTYPE)
        and int(head_dim_qk) == 256
        and int(head_dim_vo) == 256
        and int(page_size) == 64
        and int(batch) >= 1
        and int(num_kv_heads) >= 1
        and int(query_tiles_per_request) >= 1
        and int(window_left) < 0
        and int(torch.cuda.get_device_capability(device)[0]) == 12
    ):
        return None

    num_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    ctas_per_request_chunk = (
        int(batch) * int(num_kv_heads) * int(query_tiles_per_request)
    )
    one_wave_chunks = num_sms // ctas_per_request_chunk
    if one_wave_chunks <= 0:
        return None
    active_ctas = one_wave_chunks * ctas_per_request_chunk
    if active_ctas * 4 < num_sms * 3:
        return None
    return int(one_wave_chunks)


def _is_laguna_fp8_gqa6_analytic_decode_graph(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    window_left: int,
) -> bool:
    """Identify the capture-static Laguna q=1 family with device split mapping."""

    return bool(
        q_dtype == torch.bfloat16
        and kv_dtype == _FP8_KV_DTYPE
        and int(num_q_heads) == 24
        and int(num_kv_heads) == 4
        and int(head_dim_qk) == 128
        and int(head_dim_vo) == 128
        and int(page_size) == 128
        and 1 <= int(batch) <= 8
        and int(window_left) < 0
        and tuple(torch.cuda.get_device_capability(device)) == (12, 0)
    )


def _is_laguna_fp8_gqa6_analytic_verify_graph(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    query_len: int,
    window_left: int,
) -> bool:
    """Identify the exact Laguna q=8 verifier device schedule."""

    return bool(
        q_dtype == torch.bfloat16
        and kv_dtype == _FP8_KV_DTYPE
        and int(num_q_heads) == 24
        and int(num_kv_heads) == 4
        and int(head_dim_qk) == 128
        and int(head_dim_vo) == 128
        and int(page_size) == 128
        and 1 <= int(batch) <= 8
        and int(query_len) == 8
        and int(window_left) < 0
        and tuple(torch.cuda.get_device_capability(device)) == (12, 0)
    )


def _is_laguna_fp8_gqa6_full_prefill_graph(
    *,
    device: torch.device,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    window_left: int,
) -> bool:
    """Identify Laguna's capture-static full-attention prefill family."""

    return bool(
        q_dtype == torch.bfloat16
        and kv_dtype == _FP8_KV_DTYPE
        and int(num_q_heads) == 24
        and int(num_kv_heads) == 4
        and int(head_dim_qk) == 128
        and int(head_dim_vo) == 128
        and int(page_size) == 128
        and 1 <= int(batch) <= 8
        and int(window_left) < 0
        and tuple(torch.cuda.get_device_capability(device)) == (12, 0)
    )


def _heuristic_decode_graph_ctas_per_sm(
    *,
    kv_dtype: torch.dtype,
    batch: int,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
) -> int:
    if page_size != 64:
        return _DEFAULT_GRAPH_CTAS_PER_SM
    if (
        kv_dtype == torch.bfloat16
        and page_size == 64
        and head_dim_qk == 128
        and head_dim_vo == 128
        and gqa_group_size == 6
        and batch == 2
    ):
        return 6
    if (
        kv_dtype == torch.bfloat16
        and page_size == 64
        and head_dim_qk == 128
        and head_dim_vo == 128
        and gqa_group_size == 6
        and batch in (3, 4)
    ):
        return 4
    if (
        kv_dtype in (torch.bfloat16, _FP8_KV_DTYPE)
        and page_size == 64
        and head_dim_qk == 128
        and head_dim_vo == 128
        and gqa_group_size == 6
        and batch <= 4
    ):
        return 1
    if head_dim_qk >= 256 and head_dim_vo >= 256 and gqa_group_size <= 8:
        if kv_dtype == torch.bfloat16 and batch <= 1:
            return 6
        if batch <= 8:
            return 2
        return 1
    if (
        kv_dtype == _FP8_KV_DTYPE
        and batch == 1
        and head_dim_qk <= 192
        and head_dim_vo <= 128
    ):
        return 1
    if head_dim_qk <= 192 and head_dim_vo <= 128:
        return 2
    if gqa_group_size > 8:
        return 2
    return _DEFAULT_GRAPH_CTAS_PER_SM


def _resolve_graph_ctas_per_sm(
    *,
    mode: _PagedMode,
    kv_dtype: torch.dtype,
    policy_batch: int,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    graph_ctas_per_sm: int | None,
) -> int:
    if graph_ctas_per_sm is not None:
        resolved_graph_ctas_per_sm = int(graph_ctas_per_sm)
        if resolved_graph_ctas_per_sm <= 0:
            raise ValueError("graph_ctas_per_sm must be positive")
        return resolved_graph_ctas_per_sm

    if mode not in ("decode", "verify"):
        return _DEFAULT_GRAPH_CTAS_PER_SM
    resolved_graph_ctas_per_sm = _heuristic_decode_graph_ctas_per_sm(
        kv_dtype=kv_dtype,
        batch=policy_batch,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
    )

    if resolved_graph_ctas_per_sm <= 0:
        raise ValueError("resolved graph_ctas_per_sm must be positive")
    return resolved_graph_ctas_per_sm


def resolve_decode_graph_ctas_per_sm(
    *,
    kv_dtype: torch.dtype,
    batch: int,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    graph_ctas_per_sm: int | None = None,
) -> int:
    return _resolve_graph_ctas_per_sm(
        mode="decode",
        kv_dtype=kv_dtype,
        policy_batch=max(int(batch), 1),
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )


def _metadata_to_cpu_int_list(t: torch.Tensor, *, name: str) -> list[int]:
    if t.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be torch.int32 or torch.int64")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return [int(v) for v in t.detach().cpu().tolist()]


def _estimate_new_batch_size(
    *,
    packed_qo_len_arr: list[int],
    kv_len_arr: list[int],
    qo_chunk_size: int,
    kv_chunk_size_pages: int,
) -> int:
    new_batch_size = 0
    for packed_qo_len, kv_len in zip(packed_qo_len_arr, kv_len_arr):
        num_tiles_q = _ceil_div(packed_qo_len, qo_chunk_size)
        num_chunks_kv = _ceil_div(max(kv_len, 1), kv_chunk_size_pages)
        new_batch_size += num_tiles_q * num_chunks_kv
    return int(new_batch_size)


def _lookup_chunk_pages_from_table(
    max_effective_kv_pages: int,
    table: tuple[tuple[int, int | None], ...],
) -> int | None:
    for max_pages, chunk_pages in table:
        if max_effective_kv_pages <= max_pages:
            return chunk_pages
    return None


def _q_lengths_from_cu_seqlens(cu_seqlens_q: torch.Tensor) -> list[int]:
    cu_seqlens_q_list = _metadata_to_cpu_int_list(cu_seqlens_q, name="cu_seqlens_q")
    q_lengths: list[int] = []
    for start, end in zip(cu_seqlens_q_list[:-1], cu_seqlens_q_list[1:]):
        if end < start:
            raise ValueError("cu_seqlens_q must be non-decreasing")
        q_lengths.append(end - start)
    return q_lengths


def _graph_policy_batch(
    *,
    mode: _PagedMode,
    batch: int,
    total_q: int,
) -> int:
    if mode == "verify":
        return max(int(total_q), 1)
    return max(int(batch), 1)


def _chunk_policy_batch(
    *,
    mode: _PagedMode,
    kv_dtype: torch.dtype,
    batch: int,
    total_q: int,
    head_dim_qk: int,
    head_dim_vo: int,
) -> int:
    if (
        mode == "verify"
        and kv_dtype == _FP8_KV_DTYPE
        and head_dim_qk <= 192
        and head_dim_vo <= 128
    ):
        # Small-head FP8 verifier rows share each request's KV span.  Treating
        # those rows as independent requests collapses Laguna B1 from the
        # available 31 chunks to 8.  Keep the existing policy for other
        # verifier families until they have their own graph-first evidence.
        return max(int(batch), 1)
    return _graph_policy_batch(mode=mode, batch=batch, total_q=total_q)


def _decode_graph_heuristic_max_chunks_per_req(
    *,
    kv_dtype: torch.dtype,
    batch: int,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
) -> int:
    batch = max(int(batch), 1)
    head_dim_qk = max(int(head_dim_qk), 1)
    head_dim_vo = max(int(head_dim_vo), 1)
    gqa_group_size = max(int(gqa_group_size), 1)

    if head_dim_qk >= 256 and head_dim_vo >= 256 and gqa_group_size <= 8:
        total_chunk_budget = 192
        divisor = batch
        min_chunks = 1
        max_chunks = 256
    elif (
        kv_dtype == _FP8_KV_DTYPE
        and page_size == 128
        and gqa_group_size == 6
        and head_dim_qk <= 192
        and head_dim_vo <= 128
        and batch <= 8
    ):
        # The M16 FP8 GQA6 decode kernel admits two CTAs per SM.  Reserve both
        # waves in the captured graph across all requests: the architecture
        # budget clamps this nominal 96-way split to the exact device-sized
        # limit (94 total work items on 188-SM Laguna).  Live lengths select
        # each request's useful prefix and balanced ranges inside the kernel.
        total_chunk_budget = 96
        divisor = batch
        min_chunks = 4
        max_chunks = 96
    elif (
        kv_dtype == _FP8_KV_DTYPE
        and batch == 1
        and head_dim_qk <= 192
        and head_dim_vo <= 128
    ):
        total_chunk_budget = 48
        divisor = 1
        min_chunks = 4
        max_chunks = 48
    elif head_dim_qk <= 192 and head_dim_vo <= 128:
        total_chunk_budget = 64
        divisor = max(batch, 2)
        min_chunks = 4
        max_chunks = 32
    elif gqa_group_size > 8:
        total_chunk_budget = 64
        divisor = max(batch, 2)
        min_chunks = 4
        max_chunks = 32
    else:
        total_chunk_budget = 128
        divisor = batch
        min_chunks = 2
        max_chunks = 128

    chunk_count = total_chunk_budget // max(divisor, 1)
    chunk_count = min(max(chunk_count, min_chunks), max_chunks)
    if max_chunks <= 32 and chunk_count >= 4:
        chunk_count = _previous_power_of_two(chunk_count)
    return max(chunk_count, 1)


def heuristic_decode_graph_chunk_pages(
    *,
    kv_dtype: torch.dtype,
    batch: int,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
    max_chunks_per_req: int | None = None,
) -> int:
    max_chunks = _decode_graph_heuristic_max_chunks_per_req(
        kv_dtype=kv_dtype,
        batch=batch,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
    )
    if (
        kv_dtype == torch.bfloat16
        and head_dim_qk == 128
        and head_dim_vo == 128
        and gqa_group_size == 6
    ):
        bf16_max_chunks = _bf16_minimax_decode_max_chunks(
            batch=batch,
            max_effective_kv_pages=max_effective_kv_pages,
        )
        if bf16_max_chunks is not None:
            max_chunks = min(max_chunks, bf16_max_chunks)
    if max_chunks_per_req is not None:
        max_chunks = min(max_chunks, max(int(max_chunks_per_req), 1))
    return max(_ceil_div(max(int(max_effective_kv_pages), 1), max_chunks), 1)


def infer_paged_mode(cu_seqlens_q: torch.Tensor) -> Literal["decode", "extend"]:
    q_lengths = _q_lengths_from_cu_seqlens(cu_seqlens_q)
    return (
        "decode" if q_lengths and all(q_len == 1 for q_len in q_lengths) else "extend"
    )


def _fa2_determine_cta_tile_q(avg_packed_qo_len: int, head_dim: int) -> int:
    # Faithful to FlashInfer's FA2DetermineCtaTileQ.
    if avg_packed_qo_len > 64 and head_dim < 256:
        return 128
    if avg_packed_qo_len > 16:
        return 64
    return 16


def _paged_determine_cta_tile_q(
    *,
    mode: _PagedMode,
    kv_dtype: torch.dtype,
    packed_qo_len: int,
    head_dim: int,
    page_size: int,
    max_effective_kv_pages: int,
) -> int:
    if (
        mode == "verify"
        and kv_dtype == _FP8_KV_DTYPE
        and packed_qo_len == 48
        and head_dim == 128
        and page_size == 128
    ):
        # Laguna's uniform q=8/GQA6 verifier has exactly 48 packed rows.
        # Keep them in one CTA so every paged K/V tile is reused across all
        # verifier rows.  This is a capture-static geometry decision; live KV
        # length still only controls device-side chunk validity.
        del max_effective_kv_pages
        return 64
    if mode in ("decode", "verify"):
        # The production paged decode kernel's exact-plane K/V TMA path is a
        # one-Q-warp, M16 kernel.  Larger GQA groups are represented by
        # multiple query-tile work items; selecting FA2's CTA64 here creates a
        # plan that the decode kernel cannot instantiate.
        del kv_dtype, head_dim, max_effective_kv_pages
        return 16
    if mode == "extend" and packed_qo_len <= 32:
        del kv_dtype, head_dim, max_effective_kv_pages
        return 16

    cta_tile_q = _fa2_determine_cta_tile_q(packed_qo_len, head_dim)
    del max_effective_kv_pages
    del mode, kv_dtype
    return cta_tile_q


def chunk_pages_for_family(
    *,
    mode: _PagedMode,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    policy_batch: int | None,
    graph_chunk_policy: bool,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
    max_chunks_per_req: int | None = None,
    cta_tile_q: int | None = None,
) -> int | None:
    if q_dtype != torch.bfloat16:
        return None
    is_default_geometry = (
        page_size == 64
        and head_dim_qk == 256
        and head_dim_vo == 256
        and gqa_group_size == 8
    )
    if (
        mode == "extend"
        and is_default_geometry
        and not graph_chunk_policy
        and kv_dtype == _FP8_KV_DTYPE
    ):
        return _lookup_chunk_pages_from_table(
            max_effective_kv_pages, _PAGED_EXTEND_FP8_CHUNK_TABLE_PAGES
        )
    if (
        mode == "extend"
        and is_default_geometry
        and not graph_chunk_policy
        and kv_dtype == torch.bfloat16
    ):
        return _lookup_chunk_pages_from_table(
            max_effective_kv_pages,
            _PAGED_EXTEND_BF16_CHUNK_TABLE_PAGES,
        )

    if (
        mode not in ("decode", "verify")
        or not graph_chunk_policy
        or policy_batch is None
        or page_size not in (64, 128)
    ):
        return None

    if (
        mode == "verify"
        and cta_tile_q == 64
        and kv_dtype == _FP8_KV_DTYPE
        and page_size == 128
        and head_dim_qk == 128
        and head_dim_vo == 128
        and gqa_group_size == 6
        and max_chunks_per_req is not None
    ):
        # The M64 verifier has only one query tile per request.  Spend the full
        # fixed graph work budget on KV splits; replay will shrink the same
        # grid analytically from the live device cache lengths.
        chunk_pages = _ceil_div(
            max(int(max_effective_kv_pages), 1),
            max(int(max_chunks_per_req), 1),
        )
    else:
        chunk_pages = heuristic_decode_graph_chunk_pages(
            kv_dtype=kv_dtype,
            batch=int(policy_batch),
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=max_effective_kv_pages,
            max_chunks_per_req=max_chunks_per_req,
        )

    chunk_pages = _apply_decode_graph_chunk_pages_debug_policy(chunk_pages)
    return _cap_decode_graph_chunk_pages(
        chunk_pages=chunk_pages,
        max_effective_kv_pages=max_effective_kv_pages,
        max_chunks_per_req=max_chunks_per_req,
    )


@dataclass(frozen=True)
class PagedPlanBudget:
    max_total_q: int | None = None
    max_batch: int | None = None
    max_page_table_width: int | None = None
    max_work_items: int | None = None
    max_partial_rows: int | None = None


@dataclass(frozen=True)
class PagedDecodeGraphCapacity:
    """Static capacity and policy for one decode CUDA-graph bucket.

    Every field depends only on model geometry and the graph bucket's maximum
    capacities.  Live cache lengths are intentionally absent so integrations
    can allocate once, capture once, and let the device replay scheduler select
    the active schedule.
    """

    graph_ctas_per_sm: int
    cta_tile_q: int
    query_tiles_per_request: int
    architecture_max_chunks_per_request: int
    max_chunks_per_request: int
    max_work_items: int
    max_partial_rows: int
    max_effective_kv_pages: int
    worst_page_count: int
    chunk_pages_lut: tuple[int, ...]


@dataclass(frozen=True)
class PagedVerifyGraphCapacity:
    """Fixed capacity for one uniform multi-token verifier graph bucket.

    Query length and request count are capture-time geometry. Live KV lengths
    are intentionally absent so replay policy stays entirely on the device.
    """

    graph_ctas_per_sm: int
    cta_tile_q: int
    max_work_items: int
    max_partial_rows: int
    max_effective_kv_pages: int
    kv_chunk_size_pages: int
    representative_cache_seqlen: int


@dataclass(frozen=True)
class PagedExtendGraphCapacity:
    """Fixed capacity for one multi-token extend graph bucket.

    Total query capacity and request count are capture-time geometry. Live
    per-request query/cache lengths are intentionally absent; the replay
    scheduler derives those ranges from device metadata.
    """

    graph_ctas_per_sm: int
    cta_tile_q: int
    max_work_items: int
    max_effective_kv_pages: int
    representative_cache_seqlen: int


def _estimate_prefill_plan_usage(
    *,
    packed_qo_len_arr: list[int],
    q_lengths: list[int],
    kv_len_arr: list[int],
    qo_chunk_size: int,
    kv_chunk_size_pages: int,
    force_split_kv: bool,
) -> tuple[int, int]:
    new_batch_size = 0
    total_num_partial_rows = 0
    split_kv = False
    for packed_qo_len, qo_len, kv_len in zip(packed_qo_len_arr, q_lengths, kv_len_arr):
        num_tiles_q = _ceil_div(packed_qo_len, qo_chunk_size)
        num_chunks_kv = _ceil_div(max(kv_len, 1), kv_chunk_size_pages)
        split_kv = split_kv or num_chunks_kv > 1
        new_batch_size += num_tiles_q * num_chunks_kv
        total_num_partial_rows += qo_len * num_chunks_kv
    if force_split_kv:
        split_kv = True
    return int(new_batch_size), int(total_num_partial_rows if split_kv else 0)


def _prefill_usage_fits_budget(
    *,
    budget: PagedPlanBudget | None,
    new_batch_size: int,
    total_num_partial_rows: int,
) -> bool:
    if budget is None:
        return True
    if budget.max_work_items is not None and new_batch_size > int(
        budget.max_work_items
    ):
        return False
    if budget.max_partial_rows is not None and total_num_partial_rows > int(
        budget.max_partial_rows
    ):
        return False
    return True


def _prefill_binary_search_kv_chunk_size(
    *,
    enable_cuda_graph: bool,
    max_batch_size_if_split: int,
    packed_qo_len_arr: list[int],
    q_lengths: list[int],
    kv_len_arr: list[int],
    qo_chunk_size: int,
    force_split_kv: bool,
    plan_budget: PagedPlanBudget | None = None,
    min_kv_chunk_size: int = 1,
) -> tuple[bool, int]:
    max_kv_len = max(max(kv_len_arr, default=1), 1)
    low = min_kv_chunk_size
    high = max_kv_len
    while low < high:
        mid = (low + high) // 2
        new_batch_size, total_num_partial_rows = _estimate_prefill_plan_usage(
            packed_qo_len_arr=packed_qo_len_arr,
            q_lengths=q_lengths,
            kv_len_arr=kv_len_arr,
            qo_chunk_size=qo_chunk_size,
            kv_chunk_size_pages=mid,
            force_split_kv=force_split_kv,
        )
        if new_batch_size > max_batch_size_if_split or not _prefill_usage_fits_budget(
            budget=plan_budget,
            new_batch_size=new_batch_size,
            total_num_partial_rows=total_num_partial_rows,
        ):
            low = mid + 1
        else:
            high = mid
    final_new_batch_size, final_total_num_partial_rows = _estimate_prefill_plan_usage(
        packed_qo_len_arr=packed_qo_len_arr,
        q_lengths=q_lengths,
        kv_len_arr=kv_len_arr,
        qo_chunk_size=qo_chunk_size,
        kv_chunk_size_pages=low,
        force_split_kv=force_split_kv,
    )
    if final_new_batch_size > max_batch_size_if_split or not _prefill_usage_fits_budget(
        budget=plan_budget,
        new_batch_size=final_new_batch_size,
        total_num_partial_rows=final_total_num_partial_rows,
    ):
        raise ValueError(
            "paged prefill plan exceeds the configured eager workspace budget"
        )
    return (enable_cuda_graph or low < max_kv_len, low)


def decode_chunk_pages_for_graph(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    batch: int | None = None,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
    max_chunks_per_req: int | None = None,
) -> int | None:
    return chunk_pages_for_family(
        mode="decode",
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        policy_batch=batch,
        graph_chunk_policy=True,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        max_effective_kv_pages=max_effective_kv_pages,
        max_chunks_per_req=max_chunks_per_req,
    )


def build_decode_chunk_pages_lut(
    *,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    batch: int | None = None,
    page_size: int,
    head_dim_qk: int,
    head_dim_vo: int,
    gqa_group_size: int,
    max_effective_kv_pages: int,
    max_chunks_per_req: int | None = None,
) -> tuple[int, ...]:
    if batch is None:
        raise ValueError("batch is required for decode graph chunk policy lookup")
    max_effective_kv_pages = max(int(max_effective_kv_pages), 1)
    lut: list[int] = []
    for page_count in range(1, max_effective_kv_pages + 1):
        chunk_pages = decode_chunk_pages_for_graph(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            batch=batch,
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=page_count,
            max_chunks_per_req=max_chunks_per_req,
        )
        if chunk_pages is None:
            raise ValueError(
                "decode graph chunk heuristic is unavailable for "
                f"q_dtype={q_dtype}, kv_dtype={kv_dtype}, batch={batch}, page_size={page_size}"
            )
        lut.append(int(chunk_pages))
    return tuple(lut)


def _plan_decode_graph_capacity_heuristic(
    *,
    device: torch.device | str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    max_cache_page_count: int,
    window_left: int = -1,
    graph_ctas_per_sm: int | None = None,
    max_work_items: int | None = None,
    max_partial_rows: int | None = None,
    force_split_kv: bool | None = None,
) -> PagedDecodeGraphCapacity:
    """Return the fixed capacity for a graph-safe decode bucket.

    Optional work/partial limits describe already-owned caller storage.  Auto
    policy is clamped to those limits and falls back to one-chunk direct decode
    when partial storage cannot support a merge.  Explicitly forced split fails
    closed instead of weakening the request.
    """

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("decode CUDA-graph capacity requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    batch = int(batch)
    num_q_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    head_dim_qk = int(head_dim_qk)
    head_dim_vo = int(head_dim_vo)
    page_size = int(page_size)
    max_cache_page_count = int(max_cache_page_count)
    window_left = int(window_left)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("query and KV head counts must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if head_dim_qk <= 0 or head_dim_vo <= 0:
        raise ValueError("head dimensions must be positive")
    if page_size not in (64, 128):
        raise ValueError("decode graph chunk policy requires page_size=64 or 128")
    if max_cache_page_count <= 0:
        raise ValueError("max_cache_page_count must be positive")
    if window_left < -1:
        raise ValueError(
            "window_left must be -1 for full attention or a non-negative token count"
        )

    gqa_group_size = num_q_heads // num_kv_heads
    max_effective_kv_pages = max_cache_page_count
    if window_left >= 0:
        max_effective_kv_pages = min(
            max_effective_kv_pages,
            max(1, _ceil_div(window_left + page_size, page_size)),
        )
    cta_tile_q = _paged_determine_cta_tile_q(
        mode="decode",
        kv_dtype=kv_dtype,
        packed_qo_len=gqa_group_size,
        head_dim=head_dim_qk,
        page_size=page_size,
        max_effective_kv_pages=max_effective_kv_pages,
    )
    query_tiles_per_request = _ceil_div(gqa_group_size, cta_tile_q)
    resolved_graph_ctas_per_sm = resolve_decode_graph_ctas_per_sm(
        kv_dtype=kv_dtype,
        batch=batch,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )
    architecture_max_chunks = decode_graph_max_chunks_per_request_budget(
        device=device,
        num_kv_heads=num_kv_heads,
        batch=batch,
        graph_ctas_per_sm=resolved_graph_ctas_per_sm,
        query_tiles_per_request=query_tiles_per_request,
    )
    architecture_max_work_items = _graph_max_batch_size_if_split(
        device=device,
        num_kv_heads=num_kv_heads,
        graph_ctas_per_sm=resolved_graph_ctas_per_sm,
    )
    analytic_total_work_budget: int | None = None
    if batch <= 2 and _is_laguna_fp8_gqa6_analytic_decode_graph(
        device=device,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        page_size=page_size,
        batch=batch,
        window_left=window_left,
    ):
        # The flat device schedule may distribute the remainder across
        # requests (24/24/23/23 for B4, 12/.../11 for B8) instead of throwing
        # away TPC slots through integer division.  Never reserve more active
        # chunks than the captured page capacity can supply at 64 rows/stage.
        analytic_total_work_budget = min(
            architecture_max_work_items,
            batch * max_cache_page_count * max(page_size // 64, 1),
        )
    direct_work_items = batch * query_tiles_per_request
    max_chunks_budget = int(architecture_max_chunks)
    one_wave_chunks = _laguna_page128_one_wave_chunk_budget(
        device=device,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        page_size=page_size,
        batch=batch,
        query_tiles_per_request=query_tiles_per_request,
    )
    if one_wave_chunks is not None:
        max_chunks_budget = min(max_chunks_budget, one_wave_chunks)
    gqa8_chunks = _h256_gqa8_decode_chunk_budget(
        device=device,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        page_size=page_size,
        batch=batch,
        max_effective_kv_pages=max_effective_kv_pages,
        window_left=window_left,
    )
    if gqa8_chunks is not None:
        max_chunks_budget = min(max_chunks_budget, gqa8_chunks)
    if graph_ctas_per_sm is None:
        h256_one_wave_chunks = _h256_one_wave_decode_chunk_budget(
            device=device,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            batch=batch,
            num_kv_heads=num_kv_heads,
            query_tiles_per_request=query_tiles_per_request,
            window_left=window_left,
        )
        if h256_one_wave_chunks is not None:
            max_chunks_budget = min(max_chunks_budget, h256_one_wave_chunks)

    split_policy_supported = q_dtype == torch.bfloat16 and (
        _merge_backend_supports_split_kv(
            output_dtype=q_dtype,
            head_dim_vo=head_dim_vo,
        )
    )
    if force_split_kv is True and not split_policy_supported:
        raise ValueError(
            "forced decode graph split requires BF16 queries and a supported "
            f"merge output dimension, got q_dtype={q_dtype}, head_dim_vo={head_dim_vo}"
        )
    if force_split_kv is False or not split_policy_supported:
        max_chunks_budget = 1
        analytic_total_work_budget = None

    if max_work_items is not None:
        max_work_items = int(max_work_items)
        if max_work_items < direct_work_items:
            raise ValueError(
                "decode graph direct schedule needs "
                f"{direct_work_items} work items, capacity is {max_work_items}"
            )
        max_chunks_budget = min(
            max_chunks_budget, max(max_work_items // direct_work_items, 1)
        )
        if analytic_total_work_budget is not None:
            analytic_total_work_budget = min(
                analytic_total_work_budget, max_work_items
            )
    if max_partial_rows is not None:
        max_partial_rows = int(max_partial_rows)
        if max_partial_rows < 0:
            raise ValueError("max_partial_rows must be non-negative")
        if force_split_kv is True and max_partial_rows < batch:
            raise ValueError(
                "forced decode graph split needs at least one partial row per "
                f"request ({batch}), capacity is {max_partial_rows}"
            )
        partial_chunk_capacity = (
            max(max_partial_rows // batch, 1)
            if max_partial_rows >= batch
            else 1
        )
        max_chunks_budget = min(max_chunks_budget, partial_chunk_capacity)
        if analytic_total_work_budget is not None:
            analytic_total_work_budget = min(
                analytic_total_work_budget, max_partial_rows
            )

    if split_policy_supported and force_split_kv is not False:
        chunk_pages_lut = build_decode_chunk_pages_lut(
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            batch=batch,
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=max_effective_kv_pages,
            max_chunks_per_req=max_chunks_budget,
        )
    else:
        # The replay updater still consumes a fixed LUT on direct graphs.  Each
        # entry covers the entire active span and therefore selects one chunk.
        chunk_pages_lut = tuple(
            page_count for page_count in range(1, max_effective_kv_pages + 1)
        )

    worst_page_count = 1
    max_chunks_per_request = 1
    for page_count, chunk_pages in enumerate(chunk_pages_lut, start=1):
        num_chunks = _ceil_div(page_count, int(chunk_pages))
        if num_chunks > max_chunks_per_request:
            max_chunks_per_request = num_chunks
            worst_page_count = page_count

    split_storage_required = force_split_kv is True or max_chunks_per_request > 1
    if (
        analytic_total_work_budget is not None
        and split_storage_required
        and analytic_total_work_budget >= direct_work_items
    ):
        capacity_max_work_items = int(analytic_total_work_budget)
        capacity_max_partial_rows = int(analytic_total_work_budget)
        capacity_max_chunks_per_request = _ceil_div(
            analytic_total_work_budget, batch
        )
        architecture_max_chunks = max(
            architecture_max_chunks, capacity_max_chunks_per_request
        )
    else:
        capacity_max_work_items = (
            batch * query_tiles_per_request * max_chunks_per_request
        )
        capacity_max_partial_rows = (
            batch * max_chunks_per_request if split_storage_required else 0
        )
        capacity_max_chunks_per_request = max_chunks_per_request
    return PagedDecodeGraphCapacity(
        graph_ctas_per_sm=int(resolved_graph_ctas_per_sm),
        cta_tile_q=int(cta_tile_q),
        query_tiles_per_request=int(query_tiles_per_request),
        architecture_max_chunks_per_request=int(architecture_max_chunks),
        max_chunks_per_request=int(capacity_max_chunks_per_request),
        max_work_items=int(capacity_max_work_items),
        max_partial_rows=int(capacity_max_partial_rows),
        max_effective_kv_pages=int(max_effective_kv_pages),
        worst_page_count=int(worst_page_count),
        chunk_pages_lut=tuple(int(v) for v in chunk_pages_lut),
    )


def plan_decode_graph_capacity(
    *,
    device: torch.device | str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    max_cache_page_count: int,
    window_left: int = -1,
    graph_ctas_per_sm: int | None = None,
    max_work_items: int | None = None,
    max_partial_rows: int | None = None,
    force_split_kv: bool | None = None,
    config: GqaConfig | None = None,
) -> PagedDecodeGraphCapacity:
    """Build a decode capacity from a prepared config or established heuristic."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    page_size = int(page_size)
    max_cache_page_count = int(max_cache_page_count)
    if config is None:
        return _plan_decode_graph_capacity_heuristic(
            device=resolved_device,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim_qk=int(head_dim_qk),
            head_dim_vo=int(head_dim_vo),
            page_size=page_size,
            batch=int(batch),
            max_cache_page_count=max_cache_page_count,
            window_left=int(window_left),
            graph_ctas_per_sm=graph_ctas_per_sm,
            max_work_items=max_work_items,
            max_partial_rows=max_partial_rows,
            force_split_kv=force_split_kv,
        )
    return PagedDecodeGraphCapacity(
        graph_ctas_per_sm=config.graph_ctas_per_sm,
        cta_tile_q=config.cta_tile_q,
        query_tiles_per_request=config.query_tiles_per_request,
        architecture_max_chunks_per_request=config.architecture_max_chunks_per_request,
        max_chunks_per_request=config.max_chunks_per_request,
        max_work_items=config.max_work_items,
        max_partial_rows=config.max_partial_rows,
        max_effective_kv_pages=config.max_effective_kv_pages,
        worst_page_count=config.worst_page_count,
        chunk_pages_lut=config.chunk_pages_lut(),
    )
def plan_extend_graph_capacity(
    *,
    device: torch.device | str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    total_q_capacity: int,
    max_cache_page_count: int,
    window_left: int = -1,
    graph_ctas_per_sm: int | None = None,
) -> PagedExtendGraphCapacity:
    """Return capture-static scratch capacity for an extend Q bucket."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("extend CUDA-graph capacity requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    batch = int(batch)
    total_q_capacity = int(total_q_capacity)
    num_q_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    head_dim_qk = int(head_dim_qk)
    head_dim_vo = int(head_dim_vo)
    page_size = int(page_size)
    max_cache_page_count = int(max_cache_page_count)
    window_left = int(window_left)
    if batch <= 0:
        raise ValueError("batch must be positive")
    if total_q_capacity < batch:
        raise ValueError(
            "extend total_q_capacity must cover one query per request: "
            f"got {total_q_capacity} < {batch}"
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("query and KV head counts must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if head_dim_qk <= 0 or head_dim_vo <= 0:
        raise ValueError("head dimensions must be positive")
    if page_size not in (64, 128):
        raise ValueError("extend graph policy requires page_size=64 or 128")
    if max_cache_page_count <= 0:
        raise ValueError("max_cache_page_count must be positive")
    if window_left < -1:
        raise ValueError(
            "window_left must be -1 for full attention or a non-negative token count"
        )

    gqa_group_size = num_q_heads // num_kv_heads
    max_q_len = total_q_capacity - batch + 1
    max_effective_kv_pages = max_cache_page_count
    if window_left >= 0:
        max_effective_kv_pages = min(
            max_cache_page_count,
            max(
                1,
                _ceil_div(max_q_len + window_left, page_size),
            ),
        )
    max_qo_len = max_q_len * gqa_group_size
    cta_tile_q = _paged_determine_cta_tile_q(
        mode="extend",
        kv_dtype=kv_dtype,
        packed_qo_len=max_qo_len,
        head_dim=head_dim_qk,
        page_size=page_size,
        max_effective_kv_pages=max_effective_kv_pages,
    )
    if (
        max_qo_len > 64
        and _is_laguna_fp8_gqa6_full_prefill_graph(
            device=device,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            batch=batch,
            window_left=window_left,
        )
    ):
        cta_tile_q = 64

    total_num_qo_tiles = (
        _ceil_div(total_q_capacity * gqa_group_size, cta_tile_q)
        + batch
        - 1
    )
    resolved_graph_ctas_per_sm = _resolve_graph_ctas_per_sm(
        mode="extend",
        kv_dtype=kv_dtype,
        policy_batch=batch,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )
    max_work_items = max(
        _graph_max_batch_size_if_split(
            device=device,
            num_kv_heads=num_kv_heads,
            graph_ctas_per_sm=resolved_graph_ctas_per_sm,
        ),
        total_num_qo_tiles,
    )
    return PagedExtendGraphCapacity(
        graph_ctas_per_sm=int(resolved_graph_ctas_per_sm),
        cta_tile_q=int(cta_tile_q),
        max_work_items=int(max_work_items),
        max_effective_kv_pages=int(max_effective_kv_pages),
        representative_cache_seqlen=int(max_cache_page_count * page_size),
    )


def plan_verify_graph_capacity(
    *,
    device: torch.device | str,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    batch: int,
    query_len: int,
    max_cache_page_count: int,
    window_left: int = -1,
    graph_ctas_per_sm: int | None = None,
) -> PagedVerifyGraphCapacity:
    """Return capture-static scratch capacity for verifier attention."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("verify CUDA-graph capacity requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    batch = int(batch)
    query_len = int(query_len)
    num_q_heads = int(num_q_heads)
    num_kv_heads = int(num_kv_heads)
    head_dim_qk = int(head_dim_qk)
    head_dim_vo = int(head_dim_vo)
    page_size = int(page_size)
    max_cache_page_count = int(max_cache_page_count)
    window_left = int(window_left)
    if batch <= 0 or query_len <= 1:
        raise ValueError("verify graph capacity requires batch > 0 and query_len > 1")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("query and KV head counts must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if head_dim_qk <= 0 or head_dim_vo <= 0:
        raise ValueError("head dimensions must be positive")
    if page_size not in (64, 128):
        raise ValueError("verify graph policy requires page_size=64 or 128")
    if max_cache_page_count <= 0:
        raise ValueError("max_cache_page_count must be positive")
    if window_left < -1:
        raise ValueError(
            "window_left must be -1 for full attention or a non-negative token count"
        )

    gqa_group_size = num_q_heads // num_kv_heads
    total_q = batch * query_len
    max_effective_kv_pages = max_cache_page_count
    representative_cache_seqlen = max_cache_page_count * page_size
    if window_left >= 0:
        # Cover both the earliest verifier row and the worst page alignment.
        # A page-aligned maximum length only exercises one fewer page when
        # window_left + query_len crosses a page boundary, so use a static
        # length ending at offset one for graph preparation.
        max_effective_kv_pages = min(
            max_effective_kv_pages,
            max(
                1,
                _ceil_div(window_left + query_len + page_size - 1, page_size),
            ),
        )
        representative_cache_seqlen = (
            1
            if max_cache_page_count == 1
            else (max_cache_page_count - 1) * page_size + 1
        )
    cta_tile_q = _paged_determine_cta_tile_q(
        mode="verify",
        kv_dtype=kv_dtype,
        packed_qo_len=query_len * gqa_group_size,
        head_dim=head_dim_qk,
        page_size=page_size,
        max_effective_kv_pages=max_effective_kv_pages,
    )
    total_num_qo_tiles = batch * _ceil_div(
        query_len * gqa_group_size, cta_tile_q
    )
    analytic_laguna_verify = _is_laguna_fp8_gqa6_analytic_verify_graph(
        device=device,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        page_size=page_size,
        batch=batch,
        query_len=query_len,
        window_left=window_left,
    )
    resolved_graph_ctas_per_sm = _resolve_graph_ctas_per_sm(
        mode="verify",
        kv_dtype=kv_dtype,
        policy_batch=total_q,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        graph_ctas_per_sm=graph_ctas_per_sm,
    )
    if (
        analytic_laguna_verify
        and batch == 1
        and graph_ctas_per_sm is None
        and max_effective_kv_pages <= 512
    ):
        resolved_graph_ctas_per_sm = 1
    max_work_items = max(
        _graph_max_batch_size_if_split(
            device=device,
            num_kv_heads=num_kv_heads,
            graph_ctas_per_sm=resolved_graph_ctas_per_sm,
        ),
        total_num_qo_tiles,
    )
    graph_max_chunks_per_req = max(
        max_work_items // max(total_num_qo_tiles, 1),
        1,
    )
    packed_qo_len_arr = [query_len * gqa_group_size] * batch
    q_lengths = [query_len] * batch
    kv_len_arr = [max_effective_kv_pages] * batch
    kv_chunk_size_pages = chunk_pages_for_family(
        mode="verify",
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        # The launch-grid budget above still uses total_q.  This only spends
        # already capture-static capacity on the proven small-head FP8 case.
        policy_batch=_chunk_policy_batch(
            mode="verify",
            kv_dtype=kv_dtype,
            batch=batch,
            total_q=total_q,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
        ),
        graph_chunk_policy=True,
        page_size=page_size,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        gqa_group_size=gqa_group_size,
        max_effective_kv_pages=max_effective_kv_pages,
        max_chunks_per_req=graph_max_chunks_per_req,
        cta_tile_q=cta_tile_q,
    )
    work_items = 0
    partial_rows = 0
    if kv_chunk_size_pages is not None:
        work_items, partial_rows = _estimate_prefill_plan_usage(
            packed_qo_len_arr=packed_qo_len_arr,
            q_lengths=q_lengths,
            kv_len_arr=kv_len_arr,
            qo_chunk_size=cta_tile_q,
            kv_chunk_size_pages=kv_chunk_size_pages,
            force_split_kv=True,
        )
        if work_items > max_work_items:
            kv_chunk_size_pages = None
    if kv_chunk_size_pages is None:
        _split_kv, kv_chunk_size_pages = _prefill_binary_search_kv_chunk_size(
            enable_cuda_graph=True,
            max_batch_size_if_split=max_work_items,
            packed_qo_len_arr=packed_qo_len_arr,
            q_lengths=q_lengths,
            kv_len_arr=kv_len_arr,
            qo_chunk_size=cta_tile_q,
            force_split_kv=True,
        )
        work_items, partial_rows = _estimate_prefill_plan_usage(
            packed_qo_len_arr=packed_qo_len_arr,
            q_lengths=q_lengths,
            kv_len_arr=kv_len_arr,
            qo_chunk_size=cta_tile_q,
            kv_chunk_size_pages=kv_chunk_size_pages,
            force_split_kv=True,
        )
    if work_items > max_work_items:
        raise RuntimeError("verify graph policy exceeded its fixed work-item capacity")

    if analytic_laguna_verify:
        # The exact verifier maps balanced 64-row stage ranges from the live
        # device length, so page-rounded host metadata must not discard the
        # final TPC slots (512 pages / ceil(512 / 94) would launch only 86).
        # Keep a uniform per-request rectangle for B2/B4/B8 and reserve eight
        # partial rows per split because every K/V tile is shared by q=8.
        analytic_chunks_per_request = max(max_work_items // batch, 1)
        partial_rows = batch * query_len * analytic_chunks_per_request

    return PagedVerifyGraphCapacity(
        graph_ctas_per_sm=int(resolved_graph_ctas_per_sm),
        cta_tile_q=int(cta_tile_q),
        max_work_items=int(max_work_items),
        max_partial_rows=int(partial_rows),
        max_effective_kv_pages=int(max_effective_kv_pages),
        kv_chunk_size_pages=int(kv_chunk_size_pages),
        representative_cache_seqlen=int(representative_cache_seqlen),
    )


@dataclass(frozen=True)
class PagedPlanKey:
    total_q: int
    num_q_heads: int
    head_dim_qk: int
    head_dim_vo: int
    k_cache_shape: tuple[int, ...]
    v_cache_shape: tuple[int, ...]
    page_table_shape: tuple[int, ...]
    dtype: torch.dtype
    kv_dtype: torch.dtype
    mode: _PagedMode
    cta_tile_q: int
    kv_chunk_size: int
    split_kv: bool
    fixed_split_size: int
    disable_split_kv: bool
    enable_cuda_graph: bool
    graph_chunk_policy: bool
    graph_ctas_per_sm: int
    window_left: int
    max_batch_size_if_split: int
    padded_batch_size: int
    new_batch_size: int
    num_qo_tiles: int
    total_num_partial_rows: int
    page_size: int
    num_kv_heads: int
    gqa_group_size: int
    msa_block_sparse: bool
    msa_union_tile: bool
    msa_topk: int
    msa_block_tokens: int
    device_index: int


@dataclass(frozen=True, kw_only=True)
class PagedPlan:
    key: PagedPlanKey
    request_indices: tuple[int, ...]
    qo_tile_indices: tuple[int, ...]
    kv_tile_indices: tuple[int, ...]
    merge_indptr: tuple[int, ...]
    o_indptr: tuple[int, ...]
    block_valid_mask: tuple[bool, ...]
    kv_window_start_tokens: tuple[int, ...]

    def __getattr__(self, name: str):
        return getattr(self.key, name)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.device_index)


def create_paged_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    mode: _PagedMode | None = None,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
    force_split_kv: bool | None = None,
    enable_cuda_graph: bool = False,
    graph_chunk_policy: bool = False,
    max_batch_size_if_split: int | None = None,
    graph_ctas_per_sm: int | None = None,
    plan_budget: PagedPlanBudget | None = None,
    window_left: int = -1,
    msa_block_sparse: bool = False,
    msa_union_tile: bool | None = None,
    host_q_lengths: tuple[int, ...] | None = None,
    host_cache_lengths: tuple[int, ...] | None = None,
) -> PagedPlan:
    if q.ndim != 3:
        raise ValueError(
            f"q must be rank-3 [total_q, q_heads, head_dim], got {tuple(q.shape)}"
        )
    if k_cache.ndim != 4:
        raise ValueError(
            f"k_cache must be rank-4 [num_pages, page_size, kv_heads, head_dim], got {tuple(k_cache.shape)}"
        )
    if v_cache.ndim != 4:
        raise ValueError(
            f"v_cache must be rank-4 [num_pages, page_size, kv_heads, head_dim_v], got {tuple(v_cache.shape)}"
        )
    if page_table.ndim != 2:
        raise ValueError(
            f"page_table must be rank-2 [batch, max_pages], got {tuple(page_table.shape)}"
        )
    if cache_seqlens.ndim != 1:
        raise ValueError(
            f"cache_seqlens must be rank-1 [batch], got {tuple(cache_seqlens.shape)}"
        )
    if cu_seqlens_q.ndim != 1:
        raise ValueError(
            f"cu_seqlens_q must be rank-1 [batch+1], got {tuple(cu_seqlens_q.shape)}"
        )
    if host_q_lengths is None and q.device.type != "cuda":
        raise ValueError("q must be on CUDA")
    if host_q_lengths is None and not (
        k_cache.device
        == v_cache.device
        == page_table.device
        == cache_seqlens.device
        == cu_seqlens_q.device
        == q.device
    ):
        raise ValueError("all inputs must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported q dtype {q.dtype}")
    if k_cache.dtype != v_cache.dtype:
        raise TypeError("k_cache and v_cache must have matching dtypes")
    if k_cache.dtype not in (torch.float16, torch.bfloat16, _FP8_KV_DTYPE):
        raise TypeError(f"unsupported kv dtype {k_cache.dtype}")
    if window_left < -1:
        raise ValueError(
            "window_left must be -1 for full attention or a non-negative token count"
        )
    msa_block_sparse = bool(msa_block_sparse)
    if msa_block_sparse and window_left >= 0:
        raise ValueError(
            "MSA block-sparse paged attention does not support window_left/SWA"
        )

    total_q, num_q_heads, head_dim_qk = [int(dim) for dim in q.shape]
    num_pages, page_size, num_kv_heads, head_dim_k = [int(dim) for dim in k_cache.shape]
    v_num_pages, v_page_size, v_num_kv_heads, head_dim_vo = [
        int(dim) for dim in v_cache.shape
    ]
    batch, max_pages_per_request = [int(dim) for dim in page_table.shape]

    if (
        num_pages != v_num_pages
        or page_size != v_page_size
        or num_kv_heads != v_num_kv_heads
    ):
        raise ValueError(
            "k_cache and v_cache structural shapes must match except head_dim"
        )
    if head_dim_k != head_dim_qk:
        raise ValueError(
            "primary paged backend expects head_dim_qk to match k_cache head_dim"
        )
    if page_size not in (64, 128):
        raise ValueError(
            f"primary paged backend expects page_size=64 or page_size=128, got {page_size}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    if tuple(cache_seqlens.shape) != (batch,):
        raise ValueError("cache_seqlens shape must match page_table batch")
    if tuple(cu_seqlens_q.shape) != (batch + 1,):
        raise ValueError("cu_seqlens_q shape must be [batch + 1]")

    if (host_q_lengths is None) != (host_cache_lengths is None):
        raise ValueError(
            "host_q_lengths and host_cache_lengths must be supplied together"
        )
    if host_q_lengths is None:
        q_lengths = _q_lengths_from_cu_seqlens(cu_seqlens_q)
        cache_lengths = _metadata_to_cpu_int_list(
            cache_seqlens, name="cache_seqlens"
        )
        inferred_mode = infer_paged_mode(cu_seqlens_q)
    else:
        q_lengths = [int(length) for length in host_q_lengths]
        cache_lengths = [int(length) for length in host_cache_lengths]
        if len(q_lengths) != batch or len(cache_lengths) != batch:
            raise ValueError("host length metadata must match page_table batch")
        if any(length <= 0 for length in q_lengths):
            raise ValueError("host query lengths must be positive")
        if sum(q_lengths) != total_q:
            raise ValueError("host query lengths must sum to q total_q")
        inferred_mode = (
            "decode" if batch > 0 and all(length == 1 for length in q_lengths)
            else "extend"
        )
    if any(cache_len <= 0 for cache_len in cache_lengths):
        raise ValueError("primary paged backend requires cache_seqlens > 0")
    cache_pages_arr = [_ceil_div(cache_len, page_size) for cache_len in cache_lengths]
    if any(cache_pages > max_pages_per_request for cache_pages in cache_pages_arr):
        raise ValueError("page_table width is smaller than required by cache_seqlens")
    mode = inferred_mode if mode is None else mode
    if msa_union_tile is None:
        msa_union_tile = (
            msa_block_sparse
            and mode == "extend"
            and paged_control("B12X_PAGED_MSA_UNION_PREFILL", "1")
            != "0"
        )
    else:
        msa_union_tile = bool(msa_union_tile)
    if msa_union_tile and not (msa_block_sparse and mode == "extend"):
        raise ValueError("MSA union-tile prefill requires an MSA extend plan")
    if msa_block_sparse:
        if mode not in ("decode", "extend"):
            raise ValueError(
                "MSA block-sparse paged attention currently supports decode and extend plans only"
            )
        if page_size not in (64, 128):
            raise ValueError(
                "MSA block-sparse paged attention requires page_size=64 or page_size=128"
            )
        if head_dim_qk != 128 or head_dim_vo != 128:
            raise ValueError(
                "MSA block-sparse paged attention requires head_dim_qk=head_dim_vo=128"
            )
        if k_cache.dtype not in (torch.bfloat16, _FP8_KV_DTYPE):
            raise TypeError(
                "MSA block-sparse paged attention supports bf16 or fp8 KV only"
            )
        if mode == "extend" and not msa_union_tile and k_cache.dtype == _FP8_KV_DTYPE:
            # The per-token (cta_tile_q=16) extend dequant path produces wrong
            # results with fp8 KV (pre-existing, never exercised before MSA;
            # see .msaport plan follow-ups). Union-tile is the production
            # prefill path and is fp8-correct.
            raise TypeError(
                "MSA fp8 extend requires union-tile prefill "
                "(per-token fp8 extend is not supported)"
            )
    if mode == "verify" and inferred_mode != "extend":
        raise ValueError(
            f"verify mode requires q_len > 1, got inferred mode {inferred_mode}"
        )
    auto_split_kv_candidate = (
        force_split_kv is None
        and not disable_split_kv
        and not msa_block_sparse
        and mode == "decode"
        and enable_cuda_graph
        and graph_chunk_policy
        # The graph chunk LUT is currently tuned and defined for BF16 Q.
        and q.dtype == torch.bfloat16
    )
    if force_split_kv is None:
        force_split_kv = mode == "verify" or (msa_block_sparse and mode == "decode")
    else:
        force_split_kv = bool(force_split_kv)
    if mode == "extend" and force_split_kv:
        raise ValueError("extend plans no longer support split-kv")
    if plan_budget is not None:
        if plan_budget.max_total_q is not None and total_q > int(
            plan_budget.max_total_q
        ):
            raise ValueError(
                f"paged plan total_q={total_q} exceeds workspace capacity {int(plan_budget.max_total_q)}"
            )
        if plan_budget.max_batch is not None and batch > int(plan_budget.max_batch):
            raise ValueError(
                f"paged plan batch={batch} exceeds workspace capacity {int(plan_budget.max_batch)}"
            )
        if plan_budget.max_page_table_width is not None and max_pages_per_request > int(
            plan_budget.max_page_table_width
        ):
            raise ValueError(
                "paged plan page_table width="
                f"{max_pages_per_request} exceeds workspace capacity {int(plan_budget.max_page_table_width)}"
            )

    gqa_group_size = num_q_heads // num_kv_heads
    if msa_block_sparse and gqa_group_size != 16:
        raise ValueError("MSA block-sparse paged attention requires gqa_group_size=16")
    packed_qo_len_arr = [q_len * gqa_group_size for q_len in q_lengths]
    kv_len_arr = list(cache_pages_arr)
    policy_batch = _graph_policy_batch(
        mode=mode,
        batch=batch,
        total_q=total_q,
    )

    if window_left >= 0:
        kv_window_start_pages = [
            min(
                _window_start_page(
                    cache_len=cache_len,
                    q_len=q_len,
                    window_left=window_left,
                    page_size=page_size,
                ),
                max(kv_pages - 1, 0),
            )
            for cache_len, q_len, kv_pages in zip(cache_lengths, q_lengths, kv_len_arr)
        ]
        effective_kv_len_arr = [
            max(kv_pages - start_page, 1)
            for kv_pages, start_page in zip(kv_len_arr, kv_window_start_pages)
        ]
    else:
        kv_window_start_pages = [0 for _ in kv_len_arr]
        effective_kv_len_arr = kv_len_arr

    if msa_block_sparse:
        if msa_union_tile and mode == "extend":
            effective_kv_len_arr = [
                _MSA_TOPK * 8 * (_MSA_BLOCK_TOKENS // page_size) for _ in cache_lengths
            ]
        else:
            effective_kv_len_arr = [
                _msa_effective_pages(cache_len, page_size)
                for cache_len in cache_lengths
            ]

    if enable_cuda_graph and window_left >= 0:
        max_graph_window_pages = max(1, _ceil_div(window_left + page_size, page_size))
        effective_kv_len_arr = [
            max(effective_pages, min(kv_pages, max_graph_window_pages))
            for effective_pages, kv_pages in zip(effective_kv_len_arr, kv_len_arr)
        ]
        kv_window_start_pages = [
            min(start_page, max(kv_pages - effective_pages, 0))
            for start_page, kv_pages, effective_pages in zip(
                kv_window_start_pages, kv_len_arr, effective_kv_len_arr
            )
        ]

    if enable_cuda_graph:
        total_num_rows = total_q
        if mode == "verify":
            # Verifier graph plans are prepared for an exact, uniform static
            # query length.  Preserve that shape across graph batches so the
            # q=8/GQA6 packed-row specialization remains M64 for B=1..8.
            max_qo_len = max(packed_qo_len_arr)
        else:
            max_seq_len = total_num_rows - batch + 1
            max_qo_len = max_seq_len * gqa_group_size
        cta_tile_q = _paged_determine_cta_tile_q(
            mode=mode,
            kv_dtype=k_cache.dtype,
            packed_qo_len=max_qo_len,
            head_dim=head_dim_qk,
            page_size=page_size,
            max_effective_kv_pages=max(max(effective_kv_len_arr), 1),
        )
        if (
            mode == "extend"
            and max_qo_len > 64
            and not msa_block_sparse
            and _is_laguna_fp8_gqa6_full_prefill_graph(
                device=q.device,
                q_dtype=q.dtype,
                kv_dtype=k_cache.dtype,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                page_size=page_size,
                batch=batch,
                window_left=window_left,
            )
        ):
            # M64 raises active-CTA residency for the exact Laguna prefill
            # family while retaining reuse across a full GQA6 query group.
            # Query capacity is static; live KV length does not participate.
            cta_tile_q = 64
        if mode == "decode":
            total_num_qo_tiles = batch * _ceil_div(
                gqa_group_size, cta_tile_q
            )
        elif mode == "verify":
            total_num_qo_tiles = sum(
                _ceil_div(packed_qo_len, cta_tile_q)
                for packed_qo_len in packed_qo_len_arr
            )
        else:
            total_num_qo_tiles = (
                _ceil_div(total_num_rows * gqa_group_size, cta_tile_q)
                + batch
                - 1
            )
    else:
        avg_packed_qo_len = sum(packed_qo_len_arr) // max(batch, 1)
        if (
            mode in ("extend", "verify")
            and plan_budget is not None
            and plan_budget.max_total_q is not None
        ):
            avg_packed_qo_len = max(
                avg_packed_qo_len,
                int(plan_budget.max_total_q) * gqa_group_size // max(batch, 1),
            )
        cta_tile_q = _paged_determine_cta_tile_q(
            mode=mode,
            kv_dtype=k_cache.dtype,
            packed_qo_len=avg_packed_qo_len,
            head_dim=head_dim_qk,
            page_size=page_size,
            max_effective_kv_pages=max(max(effective_kv_len_arr), 1),
        )
        total_num_qo_tiles = sum(
            _ceil_div(packed_qo_len, cta_tile_q) for packed_qo_len in packed_qo_len_arr
        )

    if msa_block_sparse and mode == "extend":
        cta_tile_q = gqa_group_size * (8 if msa_union_tile else 1)
        if enable_cuda_graph:
            total_num_qo_tiles = (
                _ceil_div(total_q * gqa_group_size, cta_tile_q) + batch - 1
            )
        else:
            total_num_qo_tiles = sum(
                _ceil_div(packed_qo_len, cta_tile_q)
                for packed_qo_len in packed_qo_len_arr
            )

    if msa_block_sparse and cta_tile_q not in (gqa_group_size, gqa_group_size * 8):
        raise ValueError(
            "MSA block-sparse paged attention requires one or eight tokens per query tile "
            f"(gqa_group_size={gqa_group_size}, cta_tile_q={cta_tile_q})"
        )

    kv_window_start_tokens = tuple(
        start_page * page_size for start_page in kv_window_start_pages
    )
    resolved_graph_ctas_per_sm = 0
    if enable_cuda_graph:
        resolved_graph_ctas_per_sm = _resolve_graph_ctas_per_sm(
            mode=mode,
            kv_dtype=k_cache.dtype,
            policy_batch=policy_batch,
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            graph_ctas_per_sm=graph_ctas_per_sm,
        )
        if (
            mode == "verify"
            and batch == 1
            and graph_ctas_per_sm is None
            and all(int(q_len) == 8 for q_len in q_lengths)
            and max(max(effective_kv_len_arr), 1) <= 512
            and _is_laguna_fp8_gqa6_analytic_verify_graph(
                device=q.device,
                q_dtype=q.dtype,
                kv_dtype=k_cache.dtype,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                page_size=page_size,
                batch=batch,
                query_len=8,
                window_left=window_left,
            )
        ):
            resolved_graph_ctas_per_sm = 1
    if max_batch_size_if_split is None:
        if enable_cuda_graph:
            max_batch_size_if_split = _graph_max_batch_size_if_split(
                device=q.device,
                num_kv_heads=num_kv_heads,
                graph_ctas_per_sm=resolved_graph_ctas_per_sm,
            )
        else:
            max_batch_size_if_split = max(total_num_qo_tiles, 1) * max(
                max(effective_kv_len_arr), 1
            )
    # Keep decode graph replay on the metadata-driven single-qtile path for now.
    regularized_decode_graph = False
    padded_batch_size = (
        max(max_batch_size_if_split, total_num_qo_tiles) if enable_cuda_graph else 0
    )
    if regularized_decode_graph:
        padded_batch_size = _align_up(padded_batch_size, max(batch, 1))

    merge_backend_supports_split_kv = _merge_backend_supports_split_kv(
        output_dtype=q.dtype,
        head_dim_vo=head_dim_vo,
    )
    if not msa_block_sparse and not merge_backend_supports_split_kv:
        if force_split_kv:
            raise ValueError(
                "forced split-kv requires a supported merge backend "
                f"for output_dtype={q.dtype}, head_dim_vo={head_dim_vo}"
            )
        disable_split_kv = True
    auto_split_kv = auto_split_kv_candidate and merge_backend_supports_split_kv
    if mode == "decode" and not force_split_kv and not auto_split_kv:
        disable_split_kv = True

    min_kv_chunk_size = max(128 // page_size, 1)
    if mode == "extend":
        required_kv_chunk_size_pages = max(max(effective_kv_len_arr), 1)
        split_kv = False
        disable_split_kv = True
        if fixed_split_size > 0:
            if fixed_split_size < required_kv_chunk_size_pages:
                raise ValueError(
                    "extend fixed_split_size must cover the full effective KV span when split-kv is disabled"
                )
            kv_chunk_size_pages = fixed_split_size
        else:
            kv_chunk_size_pages = required_kv_chunk_size_pages
    elif msa_block_sparse and force_split_kv:
        split_kv = False
        kv_chunk_size_pages = (
            int(fixed_split_size)
            if fixed_split_size > 0
            else _msa_decode_chunk_pages(policy_batch, page_size)
        )
        if kv_chunk_size_pages <= 0:
            raise ValueError("MSA split chunk size must be positive")
    elif disable_split_kv and not force_split_kv:
        split_kv = False
        kv_chunk_size_pages = (
            _ceil_div(_MSA_TOPK * _MSA_BLOCK_TOKENS, page_size)
            if msa_block_sparse
            else max(max(effective_kv_len_arr), 1)
        )
    elif fixed_split_size > 0:
        split_kv = False
        kv_chunk_size_pages = fixed_split_size
    else:
        graph_max_chunks_per_req = None
        if enable_cuda_graph and graph_chunk_policy:
            graph_max_chunks_per_req = max(
                max_batch_size_if_split // max(total_num_qo_tiles, 1), 1
            )
            if mode == "decode":
                one_wave_chunks = _laguna_page128_one_wave_chunk_budget(
                    device=q.device,
                    q_dtype=q.dtype,
                    kv_dtype=k_cache.dtype,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim_qk,
                    head_dim_vo=head_dim_vo,
                    page_size=page_size,
                    batch=policy_batch,
                    query_tiles_per_request=max(
                        total_num_qo_tiles // max(policy_batch, 1), 1
                    ),
                )
                if one_wave_chunks is not None:
                    graph_max_chunks_per_req = min(
                        graph_max_chunks_per_req, one_wave_chunks
                    )
        heuristic_kv_chunk_size_pages = chunk_pages_for_family(
            mode=mode,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            policy_batch=_chunk_policy_batch(
                mode=mode,
                kv_dtype=k_cache.dtype,
                batch=batch,
                total_q=total_q,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
            ),
            graph_chunk_policy=bool(enable_cuda_graph and graph_chunk_policy),
            page_size=page_size,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            gqa_group_size=gqa_group_size,
            max_effective_kv_pages=max(max(effective_kv_len_arr), 1),
            max_chunks_per_req=graph_max_chunks_per_req,
            cta_tile_q=cta_tile_q,
        )
        heuristic_fits_graph_budget = True
        heuristic_fits_plan_budget = True
        if heuristic_kv_chunk_size_pages is not None and enable_cuda_graph:
            heuristic_new_batch_size = _estimate_new_batch_size(
                packed_qo_len_arr=packed_qo_len_arr,
                kv_len_arr=effective_kv_len_arr,
                qo_chunk_size=cta_tile_q,
                kv_chunk_size_pages=heuristic_kv_chunk_size_pages,
            )
            heuristic_fits_graph_budget = heuristic_new_batch_size <= padded_batch_size
        if heuristic_kv_chunk_size_pages is not None:
            heuristic_new_batch_size, heuristic_total_num_partial_rows = (
                _estimate_prefill_plan_usage(
                    packed_qo_len_arr=packed_qo_len_arr,
                    q_lengths=q_lengths,
                    kv_len_arr=effective_kv_len_arr,
                    qo_chunk_size=cta_tile_q,
                    kv_chunk_size_pages=heuristic_kv_chunk_size_pages,
                    force_split_kv=bool(force_split_kv),
                )
            )
            heuristic_fits_plan_budget = _prefill_usage_fits_budget(
                budget=plan_budget,
                new_batch_size=heuristic_new_batch_size,
                total_num_partial_rows=heuristic_total_num_partial_rows,
            )
        if (
            heuristic_kv_chunk_size_pages is not None
            and heuristic_fits_graph_budget
            and heuristic_fits_plan_budget
        ):
            split_kv = False
            kv_chunk_size_pages = heuristic_kv_chunk_size_pages
        else:
            split_kv, kv_chunk_size_pages = _prefill_binary_search_kv_chunk_size(
                enable_cuda_graph=enable_cuda_graph,
                max_batch_size_if_split=max_batch_size_if_split,
                packed_qo_len_arr=packed_qo_len_arr,
                q_lengths=q_lengths,
                kv_len_arr=effective_kv_len_arr,
                qo_chunk_size=cta_tile_q,
                force_split_kv=bool(force_split_kv),
                plan_budget=plan_budget,
                min_kv_chunk_size=min_kv_chunk_size,
            )

    # `_prefill_binary_search_kv_chunk_size` reserves a split merge path for
    # every CUDA graph.  Auto policy is weaker than an explicit force: when the
    # caller's fixed work/partial capacity only permits one chunk per request,
    # retain direct decode instead of allocating or failing for a vacuous
    # one-chunk merge.  Explicit force and verify keep their historical
    # always-split behavior.
    if auto_split_kv:
        split_kv = any(
            _ceil_div(max(kv_len, 1), kv_chunk_size_pages) > 1
            for kv_len in effective_kv_len_arr
        )

    request_indices: list[int] = []
    qo_tile_indices: list[int] = []
    kv_tile_indices: list[int] = []
    merge_indptr: list[int] = [0]
    o_indptr: list[int] = [0]
    request_num_chunks_kv: list[int] = []
    new_batch_size = 0

    for request_idx, (packed_qo_len, qo_len, kv_len) in enumerate(
        zip(packed_qo_len_arr, q_lengths, effective_kv_len_arr)
    ):
        num_tiles_q = _ceil_div(packed_qo_len, cta_tile_q)
        num_chunks_kv = (
            1
            if disable_split_kv and not force_split_kv
            else _ceil_div(max(kv_len, 1), kv_chunk_size_pages)
        )
        request_num_chunks_kv.append(int(num_chunks_kv))
        if not disable_split_kv or force_split_kv:
            split_kv = split_kv or num_chunks_kv > 1
        for q_tile_idx in range(num_tiles_q):
            for kv_tile_idx in range(num_chunks_kv):
                new_batch_size += 1
                request_indices.append(request_idx)
                qo_tile_indices.append(q_tile_idx)
                kv_tile_indices.append(kv_tile_idx)
        for _ in range(qo_len):
            merge_indptr.append(merge_indptr[-1] + num_chunks_kv)
        o_indptr.append(o_indptr[-1] + qo_len * num_chunks_kv)

    padded_batch_size = (
        max(max_batch_size_if_split, total_num_qo_tiles)
        if enable_cuda_graph
        else new_batch_size
    )
    if msa_block_sparse and force_split_kv and not enable_cuda_graph:
        padded_batch_size = max(padded_batch_size, 128)
    if regularized_decode_graph:
        padded_batch_size = _align_up(padded_batch_size, max(batch, 1))
    if new_batch_size > padded_batch_size:
        raise ValueError(
            "new_batch_size exceeds padded_batch_size; fixed_split_size is incompatible with the chosen graph budget"
        )
    if force_split_kv:
        split_kv = True
    total_num_partial_rows = o_indptr[-1] if split_kv else 0
    if (
        split_kv
        and mode == "verify"
        and enable_cuda_graph
        and graph_chunk_policy
        and fixed_split_size < 0
        and _is_laguna_fp8_gqa6_analytic_verify_graph(
            device=q.device,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            batch=batch,
            query_len=(q_lengths[0] if q_lengths else 0),
            window_left=window_left,
        )
        and all(int(q_len) == 8 for q_len in q_lengths)
    ):
        # `padded_batch_size` is the capture-static forward work envelope.
        # The exact entry launches a uniform split rectangle and stores one
        # partial for each of its eight query positions.
        analytic_chunks_per_request = max(padded_batch_size // batch, 1)
        total_num_partial_rows = (
            batch * 8 * analytic_chunks_per_request
        )
    if (
        split_kv
        and mode == "decode"
        and enable_cuda_graph
        and graph_chunk_policy
        and fixed_split_size < 0
        and total_q == batch
        and batch <= 2
        and _is_laguna_fp8_gqa6_analytic_decode_graph(
            device=q.device,
            q_dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            batch=batch,
            window_left=window_left,
        )
    ):
        # The exact forward and merge entries use a flat partial-row arena.
        # Preserve every device-wide work slot, including the remainder that
        # cannot be represented by one uniform chunks-per-request integer.
        # The generic metadata remains valid in its prefix; only the analytic
        # entries consume the extra fixed-capacity rows.
        analytic_stage_capacity = sum(
            _ceil_div(max(int(cache_len), 1), 64)
            for cache_len in cache_lengths
        )
        total_num_partial_rows = max(
            total_num_partial_rows,
            min(int(max_batch_size_if_split), analytic_stage_capacity),
        )
    if not _prefill_usage_fits_budget(
        budget=plan_budget,
        new_batch_size=new_batch_size,
        total_num_partial_rows=total_num_partial_rows,
    ):
        raise ValueError(
            "paged prefill plan exceeds the configured eager workspace budget"
        )

    if regularized_decode_graph:
        max_chunks_per_req = padded_batch_size // max(batch, 1)
        block_valid_mask = [False for _ in range(padded_batch_size)]
        for request_idx, num_chunks_kv in enumerate(request_num_chunks_kv):
            base_idx = request_idx * max_chunks_per_req
            for kv_tile_idx in range(num_chunks_kv):
                block_valid_mask[base_idx + kv_tile_idx] = True
    else:
        block_valid_mask = [idx < new_batch_size for idx in range(padded_batch_size)]
    kv_chunk_size = kv_chunk_size_pages * page_size

    if (
        paged_control("B12X_DEBUG_PAGED_POLICY") == "1"
        and enable_cuda_graph
        and graph_chunk_policy
        and mode == "decode"
    ):
        print(
            "# b12x_paged_policy"
            f" batch={batch}"
            f" pages={max(max(effective_kv_len_arr), 1)}"
            f" graph_ctas_per_sm={resolved_graph_ctas_per_sm}"
            f" chunk_pages={kv_chunk_size_pages}"
            f" split_kv={int(split_kv)}"
            f" padded_batch_size={padded_batch_size}"
            f" new_batch_size={new_batch_size}",
            file=sys.stderr,
            flush=True,
        )

    key = PagedPlanKey(
        total_q=total_q,
        num_q_heads=num_q_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        k_cache_shape=tuple(int(dim) for dim in k_cache.shape),
        v_cache_shape=tuple(int(dim) for dim in v_cache.shape),
        page_table_shape=tuple(int(dim) for dim in page_table.shape),
        dtype=q.dtype,
        kv_dtype=k_cache.dtype,
        mode=mode,
        cta_tile_q=cta_tile_q,
        kv_chunk_size=kv_chunk_size,
        split_kv=split_kv,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
        enable_cuda_graph=enable_cuda_graph,
        graph_chunk_policy=graph_chunk_policy,
        graph_ctas_per_sm=resolved_graph_ctas_per_sm,
        window_left=window_left,
        max_batch_size_if_split=max_batch_size_if_split,
        padded_batch_size=padded_batch_size,
        new_batch_size=new_batch_size,
        num_qo_tiles=total_num_qo_tiles,
        total_num_partial_rows=total_num_partial_rows,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        gqa_group_size=gqa_group_size,
        msa_block_sparse=msa_block_sparse,
        msa_union_tile=bool(msa_union_tile),
        msa_topk=(_MSA_TOPK if msa_block_sparse else 0),
        msa_block_tokens=(_MSA_BLOCK_TOKENS if msa_block_sparse else 0),
        device_index=q.device.index
        if q.device.index is not None
        else torch.cuda.current_device(),
    )
    return PagedPlan(
        key=key,
        request_indices=tuple(request_indices),
        qo_tile_indices=tuple(qo_tile_indices),
        kv_tile_indices=tuple(kv_tile_indices),
        merge_indptr=tuple(merge_indptr),
        o_indptr=tuple(o_indptr),
        block_valid_mask=tuple(block_valid_mask),
        kv_window_start_tokens=kv_window_start_tokens,
    )
