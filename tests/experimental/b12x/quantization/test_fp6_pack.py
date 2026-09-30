from __future__ import annotations

import torch

from b12x._lib.fp6 import (
    FLOAT6_E2M3_MAX,
    FLOAT6_E3M2_MAX,
    SF_VEC_SIZE_FP6,
    pack_4_fp6_codes,
    pack_fp6_codes_tensor,
    unpack_4_fp6_codes,
    unpack_fp6_packed_tensor,
    fp6_quantize_values_torch,
)


def test_pack_unpack_4_fp6_roundtrip() -> None:
    codes = (5, 10, 21, 63)
    b0, b1, b2 = pack_4_fp6_codes(*codes)
    assert unpack_4_fp6_codes(b0, b1, b2) == codes


def test_pack_fp6_codes_tensor_shape() -> None:
    codes = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    packed = pack_fp6_codes_tensor(codes)
    assert packed.shape == (2, 12)
    roundtrip = unpack_fp6_packed_tensor(packed, num_fp6=16)
    assert torch.equal(roundtrip, codes)


def test_fp6_quantize_respects_max() -> None:
    x = torch.tensor([FLOAT6_E3M2_MAX * 2.0, -FLOAT6_E2M3_MAX * 2.0], dtype=torch.float32)
    q3 = fp6_quantize_values_torch(x[:1], fmt="e3m2")
    q2 = fp6_quantize_values_torch(x[1:], fmt="e2m3")
    assert float(q3.abs().max()) <= FLOAT6_E3M2_MAX + 1e-3
    assert float(q2.abs().max()) <= FLOAT6_E2M3_MAX + 1e-3


def test_sf_vec_size_fp6_is_32() -> None:
    assert SF_VEC_SIZE_FP6 == 32
