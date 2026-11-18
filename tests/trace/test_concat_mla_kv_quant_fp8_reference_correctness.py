"""Reference correctness test for the concat_mla_kv_quant_fp8 trace API."""

import pytest
import torch

from tests.trace.reference_utils import _assert_finite


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(num_tokens=2048, num_heads=12, kv_dim=256, rope_dim=64),
        dict(num_tokens=257, num_heads=8, kv_dim=256, rope_dim=64),
    ],
)
def test_concat_mla_kv_quant_fp8_reference_correctness(shape_kwargs):
    """flashinfer.concat_mla_kv_quant_fp8 kernel vs reference (byte-exact fp8)."""
    from flashinfer import concat_mla_kv_quant_fp8
    from flashinfer.trace.templates.attention import concat_mla_kv_quant_fp8_trace

    inputs = concat_mla_kv_quant_fp8_trace.init(**shape_kwargs)
    _assert_finite(inputs["kv_nope"], inputs["k_pe"])
    key_api, value_api = concat_mla_kv_quant_fp8(**inputs)
    key_ref, value_ref = concat_mla_kv_quant_fp8_trace.reference(**inputs)
    # fp8 outputs: compare the byte encodings (NaN-safe), the cast is exact.
    assert torch.equal(key_api.view(torch.uint8), key_ref.view(torch.uint8))
    assert torch.equal(value_api.view(torch.uint8), value_ref.view(torch.uint8))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
