# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU: the RoPE table helper (positions + a vLLM-style cos_sin_cache) against the reference tables, into caller-owned outputs."""

import pytest
import torch

from tests.test_helpers.qwen_3_5_block_reference import build_rope_tables


def _cos_sin_cache(max_pos, rope_dim, base=1_000_000.0):
    """vLLM's ``RotaryEmbedding.cos_sin_cache``: ``[max_pos, rope_dim]`` = ``cos | sin`` halves of width ``rope_dim // 2``."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    freqs = torch.arange(max_pos, dtype=torch.float32)[:, None] * inv_freq
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_tables_match_the_reference(dtype):
    from flashinfer.qwen_3_5_block import qwen_3_5_block_rope_tables

    cache = _cos_sin_cache(4096, 64).to(dtype)
    positions = torch.tensor([0, 1, 5, 4095, 300, 300], dtype=torch.int64)
    cos, sin = qwen_3_5_block_rope_tables(positions, cache)
    ref_cos, ref_sin = build_rope_tables(
        0, 64, batch=1, device="cpu", dtype=dtype, positions=positions[None]
    )
    assert cos.shape == (6, 64) and cos.dtype == dtype
    torch.testing.assert_close(
        cos, ref_cos[0], rtol=0, atol=2e-2 if dtype == torch.bfloat16 else 1e-5
    )
    torch.testing.assert_close(
        sin, ref_sin[0], rtol=0, atol=2e-2 if dtype == torch.bfloat16 else 1e-5
    )
    assert torch.equal(cos[:, :32], cos[:, 32:]) and torch.equal(
        sin[:, :32], sin[:, 32:]
    ), "duplicated halves"
    out_cos, out_sin = torch.empty_like(cos), torch.empty_like(sin)
    c2, s2 = qwen_3_5_block_rope_tables(
        positions.to(torch.int32), cache, out_cos=out_cos, out_sin=out_sin
    )
    assert (
        c2 is out_cos
        and s2 is out_sin
        and torch.equal(out_cos, cos)
        and torch.equal(out_sin, sin)
    )
    with pytest.raises(ValueError, match="out_cos"):
        qwen_3_5_block_rope_tables(positions, cache, out_cos=torch.empty(6, 32))
    with pytest.raises(ValueError, match="positions"):
        qwen_3_5_block_rope_tables(positions.float(), cache)
