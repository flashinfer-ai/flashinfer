"""Slow, allocation-heavy mathematical oracles for QSA tests and debugging.

Nothing in the GPU :mod:`b12x.attention.qsa` path dispatches here.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


def gemma_rmsnorm_reference(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply zero-centered Gemma RMSNorm and restore the input dtype."""
    source_dtype = value.dtype
    value_fp32 = value.float()
    variance = value_fp32.square().mean(dim=-1, keepdim=True)
    normalized = value_fp32 * torch.rsqrt(variance + float(eps))
    return (normalized * (1.0 + weight.float())).to(source_dtype)


def prepare_index_reference(
    index_query: torch.Tensor,
    raw_index_key: torch.Tensor,
    query_norm_weight: torch.Tensor,
    eps: float,
    rope: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rope_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize every index-query head with one shared weight, then rotate."""
    if index_query.ndim != 3:
        raise ValueError("index_query must have shape [rows, heads, dim]")
    if tuple(raw_index_key.shape) != (
        int(index_query.shape[0]),
        int(index_query.shape[2]),
    ):
        raise ValueError("raw_index_key must have shape [rows, index_head_dim]")
    if tuple(query_norm_weight.shape) != (int(index_query.shape[2]),):
        raise ValueError("query_norm_weight must be shared with shape [index_head_dim]")
    normalized = gemma_rmsnorm_reference(index_query, query_norm_weight, eps)
    return rope(normalized, rope_positions), raw_index_key


def stream_compress_reference(
    raw_keys: torch.Tensor,
    logical_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    raw_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    compress_ratio: int,
    key_norm_weight: torch.Tensor,
    eps: float,
    rope: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutate one persistent raw ring and return newly completed groups.

    Logical tags are authoritative.  Missing history raises instead of reading
    stale payload bytes.  Callers modeling speculative verification should run
    against a state snapshot and commit only the accepted logical state.
    """
    rows, head_dim = raw_keys.shape
    if tuple(logical_positions.shape) != (rows,):
        raise ValueError("logical_positions must have shape [rows]")
    if rope_positions.ndim != 2 or int(rope_positions.shape[0]) != rows:
        raise ValueError("rope_positions must have shape [rows, position_axes]")
    capacity = int(raw_ring.shape[0])
    if tuple(raw_ring.shape) != (capacity, head_dim):
        raise ValueError("raw_ring must have shape [capacity, index_head_dim]")
    if tuple(raw_logical_positions.shape) != (capacity,):
        raise ValueError("raw_logical_positions must have shape [capacity]")
    if tuple(raw_rope_positions.shape) != (
        capacity,
        int(rope_positions.shape[1]),
    ):
        raise ValueError("raw_rope_positions has incompatible shape")
    if capacity < int(compress_ratio):
        raise ValueError("raw ring capacity must cover one compression group")
    if tuple(key_norm_weight.shape) != (head_dim,):
        raise ValueError("key_norm_weight must have shape [index_head_dim]")

    group_ids: list[int] = []
    representatives: list[torch.Tensor] = []
    previous_position: int | None = None
    for row in range(rows):
        position = int(logical_positions[row].item())
        if position < 0:
            raise ValueError("logical positions must be nonnegative")
        if previous_position is not None and position != previous_position + 1:
            raise ValueError("logical positions must be chronological and contiguous")
        previous_position = position
        slot = position % capacity
        raw_ring[slot].copy_(raw_keys[row])
        raw_logical_positions[slot] = position
        raw_rope_positions[slot].copy_(rope_positions[row])

        if (position + 1) % int(compress_ratio):
            continue
        first = position - int(compress_ratio) + 1
        source_positions = torch.arange(
            first,
            position + 1,
            dtype=torch.int64,
            device=logical_positions.device,
        )
        source_slots = torch.remainder(source_positions, capacity)
        observed = raw_logical_positions[source_slots]
        if not torch.equal(observed, source_positions):
            raise RuntimeError(
                "raw selector ring is missing tagged history for a completed group"
            )
        pooled = raw_ring[source_slots].float().mean(dim=0).to(torch.bfloat16)
        normalized = gemma_rmsnorm_reference(pooled, key_norm_weight, eps)
        first_rope_position = raw_rope_positions[first % capacity]
        representatives.append(rope(normalized, first_rope_position))
        group_ids.append(position // int(compress_ratio))

    if not representatives:
        return (
            logical_positions.new_empty((0,), dtype=torch.int64),
            raw_keys.new_empty((0, head_dim)),
        )
    return (
        torch.tensor(group_ids, dtype=torch.int64, device=raw_keys.device),
        torch.stack(representatives),
    )


def packed_stream_compress_reference(
    raw_keys: torch.Tensor,
    logical_positions: torch.Tensor,
    rope_positions: torch.Tensor,
    raw_ring: torch.Tensor,
    raw_logical_positions: torch.Tensor,
    raw_rope_positions: torch.Tensor,
    *,
    prior_interval_start_position: int,
    num_accepted_tokens: int,
    is_prefilling: bool = False,
    compress_ratio: int,
    key_norm_weight: torch.Tensor,
    eps: float,
    rope: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Apply one packed speculative interval to one persistent selector state.

    ``num_accepted_tokens`` describes the preceding verification interval and
    includes its guaranteed first token.  Therefore the first logical position
    in this interval must equal ``prior_interval_start_position +
    num_accepted_tokens``.  Completed groups consume accepted tagged history
    from the old ring and replacement rows from this interval before any raw
    ring slot is overwritten.  The returned integer is the interval start that
    must be committed as the next call's prior anchor.
    """

    rows, head_dim = raw_keys.shape
    if rows <= 0:
        raise ValueError("a packed speculative interval must contain at least one row")
    if tuple(logical_positions.shape) != (rows,):
        raise ValueError("logical_positions must have shape [rows]")
    if rope_positions.ndim != 2 or int(rope_positions.shape[0]) != rows:
        raise ValueError("rope_positions must have shape [rows, position_axes]")
    capacity = int(raw_ring.shape[0])
    if tuple(raw_ring.shape) != (capacity, head_dim):
        raise ValueError("raw_ring must have shape [capacity, index_head_dim]")
    if tuple(raw_logical_positions.shape) != (capacity,):
        raise ValueError("raw_logical_positions must have shape [capacity]")
    if tuple(raw_rope_positions.shape) != (
        capacity,
        int(rope_positions.shape[1]),
    ):
        raise ValueError("raw_rope_positions has incompatible shape")
    if int(compress_ratio) <= 0 or capacity < int(compress_ratio):
        raise ValueError("raw ring capacity must cover one positive compression group")
    if tuple(key_norm_weight.shape) != (head_dim,):
        raise ValueError("key_norm_weight must have shape [index_head_dim]")
    if int(num_accepted_tokens) < 1:
        raise ValueError("num_accepted_tokens must include at least the first token")

    expected_positions = torch.arange(
        int(logical_positions[0].item()),
        int(logical_positions[0].item()) + rows,
        dtype=torch.int64,
        device=logical_positions.device,
    )
    if not torch.equal(logical_positions, expected_positions):
        raise ValueError("logical positions must be chronological and contiguous")
    current_first = int(logical_positions[0].item())
    if current_first < 0:
        raise ValueError("logical positions must be nonnegative")
    if int(prior_interval_start_position) == -1 and (
        int(num_accepted_tokens) != 1 or current_first != 0
    ):
        raise ValueError(
            "prior interval start -1 is reserved for one accepted token at "
            "position zero"
        )
    if current_first != int(prior_interval_start_position) + int(num_accepted_tokens):
        raise ValueError(
            "current interval start must equal prior interval start plus "
            "num_accepted_tokens"
        )

    group_ids: list[int] = []
    representatives: list[torch.Tensor] = []
    for row in range(rows):
        position = int(logical_positions[row].item())
        if (position + 1) % int(compress_ratio):
            continue
        group_first = position - int(compress_ratio) + 1
        source_keys: list[torch.Tensor] = []
        first_rope_position: torch.Tensor | None = None
        for source_position in range(group_first, position + 1):
            if source_position >= current_first:
                source_row = source_position - current_first
                if source_row >= rows:
                    raise RuntimeError(
                        "completed group extends past the packed interval"
                    )
                source_keys.append(raw_keys[source_row])
                if source_position == group_first:
                    first_rope_position = rope_positions[source_row]
            else:
                source_slot = source_position % capacity
                if int(raw_logical_positions[source_slot].item()) != source_position:
                    raise RuntimeError(
                        "raw selector ring is missing accepted tagged history for "
                        "a completed group"
                    )
                source_keys.append(raw_ring[source_slot])
                if source_position == group_first:
                    first_rope_position = raw_rope_positions[source_slot]
        assert first_rope_position is not None
        pooled = torch.stack(source_keys).float().mean(dim=0).to(torch.bfloat16)
        normalized = gemma_rmsnorm_reference(pooled, key_norm_weight, eps)
        representatives.append(rope(normalized, first_rope_position))
        group_ids.append(position // int(compress_ratio))

    # Persistent mutation follows all compression reads. Only the final ring
    # suffix survives a prefill transaction, and every destination is unique.
    for row in range(max(0, rows - capacity), rows):
        position = int(logical_positions[row].item())
        slot = position % capacity
        raw_ring[slot].copy_(raw_keys[row])
        raw_logical_positions[slot] = position
        raw_rope_positions[slot].copy_(rope_positions[row])

    if representatives:
        representative_tensor = torch.stack(representatives)
        group_id_tensor = torch.tensor(
            group_ids, dtype=torch.int64, device=raw_keys.device
        )
    else:
        representative_tensor = raw_keys.new_empty((0, head_dim))
        group_id_tensor = logical_positions.new_empty((0,), dtype=torch.int64)
    anchor = int(logical_positions[-1].item()) if is_prefilling else current_first
    return group_id_tensor, representative_tensor, anchor


def paged_store_compressed_reference(
    compressed_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_id: int,
    group_ids: torch.Tensor,
    representatives: torch.Tensor,
) -> None:
    """Store compressed group keys through a request-relative page table."""
    if compressed_cache.ndim != 3:
        raise ValueError("compressed_cache must have shape [pages, page, dim]")
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [requests, logical_pages]")
    if group_ids.dtype != torch.int64:
        raise TypeError("group_ids must use int64")
    if tuple(representatives.shape) != (
        int(group_ids.numel()),
        int(compressed_cache.shape[-1]),
    ):
        raise ValueError("representatives shape does not match group_ids/cache")
    page_size = int(compressed_cache.shape[1])
    for index, group_id_tensor in enumerate(group_ids):
        group_id = int(group_id_tensor.item())
        logical_page = group_id // page_size
        page_offset = group_id % page_size
        if not 0 <= logical_page < int(block_table.shape[1]):
            raise IndexError("compressed logical page is outside the page table")
        physical_page = int(block_table[int(request_id), logical_page].item())
        if not 0 <= physical_page < int(compressed_cache.shape[0]):
            raise IndexError("compressed page table contains an invalid physical page")
        compressed_cache[physical_page, page_offset].copy_(representatives[index])


def score_select_reference(
    index_query: torch.Tensor,
    compressed_keys: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_length: int,
    compress_ratio: int,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score completed groups and emit fixed-width original-token positions."""
    rows, _, head_dim = index_query.shape
    if compressed_keys.ndim != 2 or int(compressed_keys.shape[1]) != head_dim:
        raise ValueError("compressed_keys must have shape [groups, index_head_dim]")
    if tuple(query_positions.shape) != (rows,):
        raise ValueError("query_positions must have shape [rows]")
    if int(budget) % int(compress_ratio):
        raise ValueError("budget must be divisible by compress_ratio")
    group_budget = int(budget) // int(compress_ratio)
    width = int(budget) + int(compress_ratio) - 1
    selected = torch.full(
        (rows, width),
        -1,
        dtype=torch.int32,
        device=index_query.device,
    )
    scores_out = torch.full(
        (rows, int(compressed_keys.shape[0])),
        -torch.inf,
        dtype=torch.float32,
        device=index_query.device,
    )
    for row in range(rows):
        position = int(query_positions[row].item())
        eligible = min(
            (position + 1) // int(compress_ratio),
            int(sequence_length) // int(compress_ratio),
            int(compressed_keys.shape[0]),
        )
        if eligible:
            dots = torch.einsum(
                "hd,gd->hg",
                index_query[row].float(),
                compressed_keys[:eligible].float(),
            )
            scores = F.relu(dots).sum(dim=0) / math.sqrt(head_dim)
            scores_out[row, :eligible] = scores
            count = min(group_budget, eligible)
            # Stable descending order makes exact ties prefer lower group IDs.
            group_ids = torch.argsort(scores, descending=True, stable=True)[:count]
            expanded = (
                group_ids[:, None] * int(compress_ratio)
                + torch.arange(
                    int(compress_ratio),
                    dtype=torch.int64,
                    device=index_query.device,
                )[None, :]
            ).flatten()
        else:
            expanded = query_positions.new_empty((0,), dtype=torch.int64)
        tail_start = ((position + 1) // int(compress_ratio)) * int(compress_ratio)
        tail = torch.arange(
            tail_start,
            position + 1,
            dtype=torch.int64,
            device=index_query.device,
        )
        positions = torch.cat((expanded, tail))
        selected[row, : int(positions.numel())] = positions.to(torch.int32)
    return scores_out, selected


def physical_element_offsets_reference(
    physical_pages: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    page_stride_elements: int,
    token_stride_elements: int,
) -> torch.Tensor:
    """Return page-scaled offsets with the serving-required Int64 math."""
    if physical_pages.shape != page_offsets.shape:
        raise ValueError("physical_pages and page_offsets must have the same shape")
    return physical_pages.to(torch.int64) * int(page_stride_elements) + page_offsets.to(
        torch.int64
    ) * int(token_stride_elements)


def sparse_paged_gqa_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Gather exact original-token paged K/V and apply causal GQA."""
    rows, query_heads, head_dim = query.shape
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("K/V cache must have shape [pages, page, kv_heads, dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("K/V cache shapes must match")
    if int(key_cache.shape[-1]) != head_dim:
        raise ValueError("query and cache head dimensions must match")
    kv_heads = int(key_cache.shape[2])
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if tuple(request_ids.shape) != (rows,) or tuple(query_positions.shape) != (rows,):
        raise ValueError("request_ids/query_positions must have shape [rows]")
    if selected_positions.ndim != 2 or int(selected_positions.shape[0]) != rows:
        raise ValueError("selected_positions must have shape [rows, width]")
    if selected_positions.dtype != torch.int32:
        raise TypeError("selected_positions must use int32")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    heads_per_kv = query_heads // kv_heads
    head_to_kv = torch.arange(query_heads, device=query.device) // heads_per_kv
    output = torch.zeros_like(query)
    page_size = int(key_cache.shape[1])
    for row in range(rows):
        positions = selected_positions[row].to(torch.int64)
        positions = positions[
            (positions >= 0) & (positions <= query_positions[row].to(torch.int64))
        ]
        if not int(positions.numel()):
            continue
        logical_pages = torch.div(positions, page_size, rounding_mode="floor")
        if int(logical_pages.max().item()) >= int(block_table.shape[1]):
            raise IndexError("selected position is outside the main page table")
        physical_pages = block_table[int(request_ids[row].item()), logical_pages].to(
            torch.int64
        )
        if bool(
            ((physical_pages < 0) | (physical_pages >= key_cache.shape[0])).any().item()
        ):
            raise IndexError("main page table contains an invalid physical page")
        page_offsets = torch.remainder(positions, page_size)
        gathered_k = key_cache[physical_pages, page_offsets][:, head_to_kv]
        gathered_v = value_cache[physical_pages, page_offsets][:, head_to_kv]
        logits = torch.einsum(
            "hd,khd->hk", query[row].float(), gathered_k.float()
        ) * float(softmax_scale)
        probabilities = torch.softmax(logits, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, gathered_v.float()).to(
            query.dtype
        )
    return output


__all__ = [
    "gemma_rmsnorm_reference",
    "prepare_index_reference",
    "stream_compress_reference",
    "paged_store_compressed_reference",
    "score_select_reference",
    "physical_element_offsets_reference",
    "sparse_paged_gqa_reference",
]
