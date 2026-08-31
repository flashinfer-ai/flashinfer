# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only routing and ABI tests for the SM120 GLM-5.3 NoPE specialization."""

import pytest

from flashinfer.mla._sparse_mla_sm120 import (
    _BPT_DSV3_2,
    _DECODE_GLM53_NOPE_DISPATCH,
    _MODEL_TYPE_DSV4,
    _MODEL_TYPE_GLM53_NOPE,
    _bytes_per_token_for_model_type,
    _decode_dsv3_2_dispatchable,
    _resolve_model_type,
)


def test_glm53_nope_model_resolution_and_abi() -> None:
    """Arbitrary FP32 scales disambiguate NoPE from the DSV4 512-wide path."""
    assert _resolve_model_type(512, "arbitrary_fp32") == _MODEL_TYPE_GLM53_NOPE
    assert _resolve_model_type(512, "auto") == _MODEL_TYPE_DSV4
    assert _bytes_per_token_for_model_type(_MODEL_TYPE_GLM53_NOPE) == 656
    assert _BPT_DSV3_2 == 656


@pytest.mark.parametrize("num_heads", [32, 64])
def test_glm53_nope_decode_envelope(num_heads: int) -> None:
    """Both TP2 and TP1 route only at the native top-k and 512+0 geometry."""
    assert (num_heads, 2176) in _DECODE_GLM53_NOPE_DISPATCH
    assert _decode_dsv3_2_dispatchable(
        64, num_heads, 2176, 512, 64, _MODEL_TYPE_GLM53_NOPE
    )
    assert not _decode_dsv3_2_dispatchable(
        64, num_heads, 2048, 512, 64, _MODEL_TYPE_GLM53_NOPE
    )
    assert not _decode_dsv3_2_dispatchable(
        64, num_heads, 2176, 576, 64, _MODEL_TYPE_GLM53_NOPE
    )


def test_glm53_nope_decode_envelope_rejects_uninstantiated_heads() -> None:
    """The standalone branch does not claim general arbitrary-head support."""
    assert not _decode_dsv3_2_dispatchable(
        64, 128, 2176, 512, 64, _MODEL_TYPE_GLM53_NOPE
    )
    assert not _decode_dsv3_2_dispatchable(
        64, 32, 2176, 512, 64, _MODEL_TYPE_DSV4
    )
