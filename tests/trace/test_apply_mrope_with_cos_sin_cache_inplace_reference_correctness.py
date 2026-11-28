# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from flashinfer import apply_mrope_with_cos_sin_cache_inplace as api
from flashinfer.trace.templates.mrope import (
    apply_mrope_with_cos_sin_cache_inplace_trace as template,
)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_mrope_trace_reference(native, interleaved):
    inputs = template.init(
        nnz=117,
        cache_fp32=not native,
        section_t=24,
        section_h=20,
        section_w=20,
        mrope_interleaved=interleaved,
    )
    expected = template.reference(**inputs)
    assert api(**inputs) is None
    for actual, wanted in zip((inputs["query"], inputs["key"]), expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
