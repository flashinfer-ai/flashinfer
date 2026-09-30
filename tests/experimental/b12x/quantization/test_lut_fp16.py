from __future__ import annotations

import hashlib

import pytest
import torch

from b12x._lib.quant.lut_fp16 import (
    LUT_FP16_SEGMENT_TABLE_BYTES,
    LUT_FP16_SEGMENT_COUNT,
    LUT_FP16_SEGMENT_TABLE_SHA256,
    lut_fp16_decode_ranks_torch,
    lut_fp16_segment_table_cpu,
    lut_fp16_direct_table_cpu,
    lut_fp16_permutation,
)


def test_lut_fp16_segment_table_has_frozen_identity() -> None:
    payload = lut_fp16_segment_table_cpu()
    assert payload.dtype == torch.uint8
    assert payload.numel() == LUT_FP16_SEGMENT_TABLE_BYTES
    assert payload.view(torch.float16).numel() == 2 * LUT_FP16_SEGMENT_COUNT
    assert hashlib.sha256(payload.numpy().tobytes()).hexdigest() == (
        LUT_FP16_SEGMENT_TABLE_SHA256
    )


def test_lut_fp16_value_law_is_finite_sign_symmetric_and_close_to_gaussian() -> None:
    ranks = torch.arange(1 << 16, dtype=torch.int64)
    actual = lut_fp16_decode_ranks_torch(ranks)
    target = (
        1.5 * torch.special.ndtri((ranks.double() + 0.5) / (1 << 16))
    ).to(torch.float16)
    assert actual.dtype == torch.float16
    assert bool(torch.isfinite(actual).all())
    assert torch.equal(actual, -actual.flip(0))
    error = actual.float() - target.float()
    assert float(torch.sqrt(torch.mean(error.square()))) < 7.0e-4
    assert float(error.abs().max()) <= 4.0e-3


@pytest.mark.parametrize("bits", [5, 6])
def test_lut_fp16_direct_table_uses_bijective_permutation(bits: int) -> None:
    codewords = torch.arange(1 << 16, dtype=torch.int64)
    ranks = lut_fp16_permutation(codewords, bits)
    assert torch.equal(torch.sort(ranks).values, codewords)
    expected = lut_fp16_decode_ranks_torch(ranks)
    assert torch.equal(lut_fp16_direct_table_cpu(bits), expected)


@pytest.mark.parametrize("bits", [2, 3, 4, 7])
def test_lut_fp16_rejects_unsupported_rates(bits: int) -> None:
    with pytest.raises(ValueError, match="K5/K6"):
        lut_fp16_permutation(torch.zeros(1), bits)
