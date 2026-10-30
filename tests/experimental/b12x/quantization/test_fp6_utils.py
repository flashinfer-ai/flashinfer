from __future__ import annotations

import cutlass
import pytest
import torch

from b12x._lib.utils import (
    MXFP6_MMA_K,
    MXFP6_SF_DTYPE_STR,
    MXFP6_SF_VEC_SIZE,
    cutlass_to_torch_dtype,
    get_cutlass_dtype,
    is_mxfp6_ab_dtype,
    is_mxfp6_ab_dtype_string,
    mxfp6_logical_k_from_packed_bytes,
    mxfp6_num_k_blocks,
    mxfp6_packed_k_bytes,
    mxfp6_tile_k,
    mxfp6_tile_shape_mnk,
    verify_mxfp6_smem_tile_k,
)


def test_get_cutlass_dtype_fp6() -> None:
    assert get_cutlass_dtype("float6_e3m2fn") is cutlass.Float6E3M2FN
    assert get_cutlass_dtype("float6_e2m3fn") is cutlass.Float6E2M3FN


def test_get_cutlass_dtype_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unsupported cutlass dtype"):
        get_cutlass_dtype("float6_unknown")


def test_cutlass_to_torch_dtype_fp6_storage_is_uint8() -> None:
    assert cutlass_to_torch_dtype(cutlass.Float6E3M2FN) is torch.uint8
    assert cutlass_to_torch_dtype(cutlass.Float6E2M3FN) is torch.uint8


def test_mxfp6_tile_k_and_k_blocks() -> None:
    assert mxfp6_tile_k() == 128
    assert mxfp6_num_k_blocks(128) == 4
    assert MXFP6_MMA_K == 32


def test_mxfp6_packed_k_bytes_roundtrip() -> None:
    for k in (32, 128, 4096):
        packed = mxfp6_packed_k_bytes(k)
        assert mxfp6_logical_k_from_packed_bytes(packed) == k
        assert packed == (3 * k) // 4


def test_mxfp6_tile_shape_mnk() -> None:
    assert mxfp6_tile_shape_mnk(128, 128) == (128, 128, 128)


def test_verify_mxfp6_smem_tile_k_default() -> None:
    assert verify_mxfp6_smem_tile_k(128) == 4


def test_is_mxfp6_ab_dtype_helpers() -> None:
    assert is_mxfp6_ab_dtype(cutlass.Float6E3M2FN)
    assert is_mxfp6_ab_dtype_string("float6_e2m3fn")
    assert not is_mxfp6_ab_dtype(cutlass.Float4E2M1FN)
    assert MXFP6_SF_VEC_SIZE == 32
    assert MXFP6_SF_DTYPE_STR == "float8_e8m0fnu"
    assert get_cutlass_dtype(MXFP6_SF_DTYPE_STR) is cutlass.Float8E8M0FNU
