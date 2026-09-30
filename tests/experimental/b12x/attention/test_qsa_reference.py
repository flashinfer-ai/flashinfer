from __future__ import annotations

import math

import pytest
import torch

from b12x.attention.qsa.reference import (
    gemma_rmsnorm_reference,
    packed_stream_compress_reference,
    paged_store_compressed_reference,
    physical_element_offsets_reference,
    score_select_reference,
    sparse_paged_gqa_reference,
    stream_compress_reference,
)


def _identity_rope(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    del positions
    return value


def test_stream_compression_is_bf16_rounded_and_survives_call_boundaries() -> None:
    dim = 8
    ring = torch.empty((4, dim), dtype=torch.bfloat16)
    tags = torch.full((4,), -1, dtype=torch.int64)
    rope_tags = torch.full((4, 1), -1, dtype=torch.int64)
    weight = torch.linspace(-0.2, 0.2, dim, dtype=torch.float32)
    raw = (torch.arange(8 * dim, dtype=torch.float32).reshape(8, dim) / 17).to(
        torch.bfloat16
    )

    ids0, representatives0 = stream_compress_reference(
        raw[:3],
        torch.arange(3, dtype=torch.int64),
        torch.arange(3, dtype=torch.int64)[:, None],
        ring,
        tags,
        rope_tags,
        4,
        weight,
        1e-6,
        _identity_rope,
    )
    assert ids0.numel() == 0
    ids1, representatives1 = stream_compress_reference(
        raw[3:],
        torch.arange(3, 8, dtype=torch.int64),
        torch.arange(3, 8, dtype=torch.int64)[:, None],
        ring,
        tags,
        rope_tags,
        4,
        weight,
        1e-6,
        _identity_rope,
    )

    assert torch.equal(ids1, torch.tensor([0, 1], dtype=torch.int64))
    expected = torch.stack(
        [
            gemma_rmsnorm_reference(
                raw[start : start + 4].float().mean(0).to(torch.bfloat16),
                weight,
                1e-6,
            )
            for start in (0, 4)
        ]
    )
    assert torch.equal(representatives1, expected)
    assert torch.equal(tags, torch.arange(4, 8, dtype=torch.int64))


def test_stream_compression_rejects_stale_payload_with_wrong_tag() -> None:
    ring = torch.zeros((4, 8), dtype=torch.bfloat16)
    tags = torch.tensor([0, 1, -1, -1], dtype=torch.int64)
    rope_tags = torch.arange(4, dtype=torch.int64)[:, None]
    with pytest.raises(RuntimeError, match="missing tagged history"):
        stream_compress_reference(
            torch.ones((1, 8), dtype=torch.bfloat16),
            torch.tensor([3], dtype=torch.int64),
            torch.tensor([[3]], dtype=torch.int64),
            ring,
            tags,
            rope_tags,
            4,
            torch.zeros((8,), dtype=torch.float32),
            1e-6,
            _identity_rope,
        )


def test_packed_speculative_compression_replaces_rejected_groups_before_use() -> None:
    dim = 8
    ratio = 4
    ring = torch.empty((12, dim), dtype=torch.bfloat16)
    tags = torch.full((12,), -1, dtype=torch.int64)
    rope_tags = torch.full((12, 1), -1, dtype=torch.int64)
    weight = torch.linspace(-0.2, 0.2, dim, dtype=torch.float32)
    cache = torch.zeros((1, 2, dim), dtype=torch.bfloat16)
    table = torch.tensor([[0]], dtype=torch.int32)

    first_raw = torch.zeros((8, dim), dtype=torch.bfloat16)
    first_ids, first_representatives, anchor = packed_stream_compress_reference(
        first_raw,
        torch.arange(8, dtype=torch.int64),
        torch.arange(8, dtype=torch.int64)[:, None],
        ring,
        tags,
        rope_tags,
        prior_interval_start_position=-1,
        num_accepted_tokens=1,
        compress_ratio=ratio,
        key_norm_weight=weight,
        eps=1e-6,
        rope=_identity_rope,
    )
    assert torch.equal(first_ids, torch.tensor([0, 1], dtype=torch.int64))
    paged_store_compressed_reference(cache, table, 0, first_ids, first_representatives)
    stale_group_one = cache[0, 1].clone()

    replacement_raw = (
        torch.arange(3 * dim, dtype=torch.float32).reshape(3, dim) + 1
    ).to(torch.bfloat16)
    replacement_ids, replacement_representatives, anchor = (
        packed_stream_compress_reference(
            replacement_raw,
            torch.arange(2, 5, dtype=torch.int64),
            torch.arange(2, 5, dtype=torch.int64)[:, None],
            ring,
            tags,
            rope_tags,
            prior_interval_start_position=anchor,
            num_accepted_tokens=2,
            compress_ratio=ratio,
            key_norm_weight=weight,
            eps=1e-6,
            rope=_identity_rope,
        )
    )
    assert torch.equal(replacement_ids, torch.tensor([0], dtype=torch.int64))
    paged_store_compressed_reference(
        cache, table, 0, replacement_ids, replacement_representatives
    )
    assert torch.equal(cache[0, 1], stale_group_one)

    # At logical position four only group zero is eligible, so the rejected
    # speculative representative for group one cannot be selected.
    _, selected = score_select_reference(
        torch.ones((1, 1, dim), dtype=torch.bfloat16),
        cache[0],
        torch.tensor([4], dtype=torch.int64),
        5,
        ratio,
        2048,
    )
    assert torch.all(selected[0, :4] < 4)
    assert int(selected[0, 4]) == 4
    assert torch.all(selected[0, 5:] == -1)

    final_raw = (
        torch.arange(3 * dim, dtype=torch.float32).reshape(3, dim).flip(-1) + 3
    ).to(torch.bfloat16)
    final_ids, final_representatives, final_anchor = packed_stream_compress_reference(
        final_raw,
        torch.arange(5, 8, dtype=torch.int64),
        torch.arange(5, 8, dtype=torch.int64)[:, None],
        ring,
        tags,
        rope_tags,
        prior_interval_start_position=anchor,
        num_accepted_tokens=3,
        compress_ratio=ratio,
        key_norm_weight=weight,
        eps=1e-6,
        rope=_identity_rope,
    )
    assert torch.equal(final_ids, torch.tensor([1], dtype=torch.int64))
    assert final_anchor == 5
    paged_store_compressed_reference(cache, table, 0, final_ids, final_representatives)
    assert not torch.equal(cache[0, 1], stale_group_one)
    expected_group_one = gemma_rmsnorm_reference(
        torch.stack((replacement_raw[2], *final_raw))
        .float()
        .mean(0)
        .to(torch.bfloat16),
        weight,
        1e-6,
    )
    assert torch.equal(cache[0, 1], expected_group_one)


def test_packed_prefill_commits_only_final_ring_suffix_and_final_anchor() -> None:
    dim = 8
    capacity = 8
    rows = 19
    raw = torch.arange(rows * dim, dtype=torch.float32).reshape(rows, dim).to(
        torch.bfloat16
    )
    positions = torch.arange(rows, dtype=torch.int64)
    ring = torch.full((capacity, dim), -1, dtype=torch.bfloat16)
    tags = torch.full((capacity,), -1, dtype=torch.int64)
    rope_tags = torch.full((capacity, 1), -1, dtype=torch.int64)

    _, _, anchor = packed_stream_compress_reference(
        raw,
        positions,
        positions[:, None],
        ring,
        tags,
        rope_tags,
        prior_interval_start_position=-1,
        num_accepted_tokens=1,
        is_prefilling=True,
        compress_ratio=4,
        key_norm_weight=torch.zeros((dim,), dtype=torch.float32),
        eps=1e-6,
        rope=_identity_rope,
    )

    assert anchor == rows - 1
    for position in range(rows - capacity, rows):
        slot = position % capacity
        assert int(tags[slot]) == position
        assert torch.equal(ring[slot], raw[position])
        assert int(rope_tags[slot, 0]) == position
def test_packed_speculative_compression_rejects_inconsistent_acceptance() -> None:
    with pytest.raises(ValueError, match="prior interval start"):
        packed_stream_compress_reference(
            torch.ones((1, 8), dtype=torch.bfloat16),
            torch.tensor([2], dtype=torch.int64),
            torch.tensor([[2]], dtype=torch.int64),
            torch.empty((4, 8), dtype=torch.bfloat16),
            torch.full((4,), -1, dtype=torch.int64),
            torch.full((4, 1), -1, dtype=torch.int64),
            prior_interval_start_position=0,
            num_accepted_tokens=1,
            compress_ratio=4,
            key_norm_weight=torch.zeros((8,), dtype=torch.float32),
            eps=1e-6,
            rope=_identity_rope,
        )


@pytest.mark.parametrize("accepted", [2, 3])
def test_packed_speculative_compression_reserves_negative_one_anchor_for_position_zero(
    accepted: int,
) -> None:
    ring = torch.randn((4, 8), dtype=torch.float32).to(torch.bfloat16)
    tags = torch.tensor([19, 20, 21, 22], dtype=torch.int64)
    rope_tags = tags[:, None].clone()
    ring_before = ring.clone()
    tags_before = tags.clone()
    rope_before = rope_tags.clone()

    with pytest.raises(ValueError, match="reserved for one accepted token"):
        packed_stream_compress_reference(
            torch.ones((1, 8), dtype=torch.bfloat16),
            torch.tensor([accepted - 1], dtype=torch.int64),
            torch.tensor([[accepted - 1]], dtype=torch.int64),
            ring,
            tags,
            rope_tags,
            prior_interval_start_position=-1,
            num_accepted_tokens=accepted,
            compress_ratio=4,
            key_norm_weight=torch.zeros((8,), dtype=torch.float32),
            eps=1e-6,
            rope=_identity_rope,
        )

    assert torch.equal(ring, ring_before)
    assert torch.equal(tags, tags_before)
    assert torch.equal(rope_tags, rope_before)


@pytest.mark.parametrize(
    ("position", "expected_tail"),
    [(0, [0]), (2, [0, 1, 2]), (3, []), (4, [4])],
)
def test_score_selection_appends_exact_causal_tail(
    position: int,
    expected_tail: list[int],
) -> None:
    query = torch.ones((1, 2, 8), dtype=torch.bfloat16)
    keys = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8).to(torch.bfloat16)
    _, selected = score_select_reference(
        query,
        keys,
        torch.tensor([position], dtype=torch.int64),
        position + 1,
        4,
        2048,
    )
    eligible = (position + 1) // 4
    expanded = selected[0, : eligible * 4].tolist()
    tail = selected[0, eligible * 4 : eligible * 4 + len(expected_tail)].tolist()
    assert len(expanded) == eligible * 4
    assert tail == expected_tail
    assert torch.all(selected[0, eligible * 4 + len(expected_tail) :] == -1)


def test_score_selection_ties_prefer_lower_group_ids() -> None:
    query = torch.zeros((1, 4, 128), dtype=torch.bfloat16)
    keys = torch.zeros((520, 128), dtype=torch.bfloat16)
    _, selected = score_select_reference(
        query,
        keys,
        torch.tensor([2079], dtype=torch.int64),
        2080,
        4,
        2048,
    )
    assert torch.equal(selected[0, :2048], torch.arange(2048, dtype=torch.int32))
    assert torch.all(selected[0, 2048:] == -1)


def test_compressed_paged_store_uses_request_relative_group_pages() -> None:
    cache = torch.zeros((4, 2, 8), dtype=torch.bfloat16)
    table = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)
    groups = torch.tensor([0, 3], dtype=torch.int64)
    values = torch.stack([torch.arange(8), torch.arange(8) + 10]).to(torch.bfloat16)
    paged_store_compressed_reference(cache, table, 1, groups, values)
    assert torch.equal(cache[2, 0], values[0])
    assert torch.equal(cache[0, 1], values[1])


def test_page_scaled_reference_offsets_do_not_wrap_int32() -> None:
    stride = 2048
    physical_page = 2**31 // stride + 7
    offsets = physical_element_offsets_reference(
        torch.tensor([physical_page], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        page_stride_elements=stride,
        token_stride_elements=128,
    )
    assert offsets.dtype == torch.int64
    assert offsets.item() == physical_page * stride + 3 * 128
    assert offsets.item() > 2**31


def test_sparse_paged_gqa_reference_reads_exact_original_token_values() -> None:
    query = torch.ones((1, 4, 16), dtype=torch.bfloat16)
    key = torch.zeros((2, 4, 2, 16), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    key[1, 1].fill_(0.25)
    value[1, 1, 0].fill_(3)
    value[1, 1, 1].fill_(7)
    selected = torch.full((1, 5), -1, dtype=torch.int32)
    selected[0, 0] = 1
    output = sparse_paged_gqa_reference(
        query,
        key,
        value,
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        selected,
        torch.tensor([1], dtype=torch.int64),
        1.0 / math.sqrt(16),
    )
    assert torch.equal(output[0, :2], torch.full((2, 16), 3, dtype=torch.bfloat16))
    assert torch.equal(output[0, 2:], torch.full((2, 16), 7, dtype=torch.bfloat16))
