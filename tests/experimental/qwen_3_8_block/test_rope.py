# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the position-indexed RoPE tables (formed once at load from a vLLM-style cos_sin_cache) bitwise the per-token helper's rows on
every position, the padding past the cache's rows, the typed refusals; the per-token helper re-exported."""

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_reference import build_rope_tables


def _cos_sin_cache(max_pos, rope_dim, base=1_000_000.0, dtype=torch.float32):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    freqs = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv_freq
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_position_tables_are_bitwise_the_per_token_rows(dtype):
    from flashinfer.qwen_3_8_block import (
        qwen_3_8_block_rope_position_tables,
        qwen_3_8_block_rope_tables,
    )

    cache = _cos_sin_cache(4096, 64, dtype=dtype)
    cos_pos, sin_pos = qwen_3_8_block_rope_position_tables(cache)
    assert (
        cos_pos.shape == (1, 4096, 64)
        and cos_pos.dtype == dtype
        and cos_pos.is_contiguous()
    )
    positions = torch.arange(4096, dtype=torch.int64)
    cos, sin = qwen_3_8_block_rope_tables(positions, cache)
    assert torch.equal(cos_pos[0], cos) and torch.equal(sin_pos[0], sin), (
        "the position tables are the per-token rows at every position"
    )
    assert torch.equal(cos_pos[0, :, :32], cos_pos[0, :, 32:]), "duplicated halves"
    ref_cos, ref_sin = build_rope_tables(
        0, 64, batch=1, device="cpu", dtype=dtype, positions=positions[None]
    )
    torch.testing.assert_close(
        cos_pos, ref_cos, rtol=0, atol=2e-2 if dtype == torch.bfloat16 else 1e-5
    )
    torch.testing.assert_close(
        sin_pos, ref_sin, rtol=0, atol=2e-2 if dtype == torch.bfloat16 else 1e-5
    )


def test_position_tables_pad_to_the_table_reach_and_cast():
    from flashinfer.qwen_3_8_block import qwen_3_8_block_rope_position_tables

    cache = _cos_sin_cache(100, 64)
    cos_pos, sin_pos = qwen_3_8_block_rope_position_tables(
        cache, num_positions=112, dtype=torch.bfloat16
    )
    assert cos_pos.shape == (1, 112, 64) and cos_pos.dtype == torch.bfloat16
    ref, _ = qwen_3_8_block_rope_position_tables(cache.to(torch.bfloat16))
    assert torch.equal(cos_pos[0, :100], ref[0]) and torch.equal(
        cos_pos[0, 100:], ref[0, 99:100].expand(12, 64)
    ), "the rows past the cache repeat the last row (positions no token reaches)"
    short, _ = qwen_3_8_block_rope_position_tables(cache, num_positions=10)
    full, _ = qwen_3_8_block_rope_position_tables(cache)
    assert short.shape == (1, 10, 64) and short.dtype == cache.dtype == torch.float32, (
        "no dtype: the cache's own"
    )
    assert torch.equal(short[0], full[0, :10]), (
        "a shorter table is the prefix of the full one"
    )
    with pytest.raises(ValueError, match="num_positions"):
        qwen_3_8_block_rope_position_tables(cache, num_positions=0)
    with pytest.raises(ValueError, match="cos_sin_cache"):
        qwen_3_8_block_rope_position_tables(cache[:, :-1])
    with pytest.raises(ValueError, match="cos_sin_cache"):
        qwen_3_8_block_rope_position_tables(cache[0])


def test_per_token_helper_is_the_qwen_3_5_one():
    from flashinfer.qwen_3_5_block import qwen_3_5_block_rope_tables
    from flashinfer.experimental.qwen_3_8_block import rope

    assert rope.qwen_3_8_block_rope_tables is rope.qwen_3_5_block_rope_tables
    cache = _cos_sin_cache(64, 64)
    pos = torch.tensor([0, 5, 63])
    a = rope.qwen_3_8_block_rope_tables(pos, cache)
    b = qwen_3_5_block_rope_tables(pos, cache)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
