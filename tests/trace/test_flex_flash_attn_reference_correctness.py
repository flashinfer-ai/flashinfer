"""Reference correctness test for the magi_ffa_flex (NHD) trace API.

MagiAttention is an optional dependency, so this test is skipped (not failed)
where it is not installed — the opt-in CI workflow covers the real path.
"""

import pytest
import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(num_tokens_q=128),
        dict(num_tokens_q=96, num_qo_heads=4, num_kv_heads=2, head_dim=64),
    ],
)
def test_flex_flash_attn_reference_correctness(shape_kwargs):
    """flashinfer.magi_ffa.flex_flash_attn kernel vs reference."""
    pytest.importorskip("magi_attention")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MagiAttention FFA")

    import flashinfer.magi_ffa
    from flashinfer.trace.templates.attention import magi_ffa_flex_nhd_trace

    inputs = magi_ffa_flex_nhd_trace.init(**shape_kwargs)
    _assert_finite(inputs["q"], inputs["k"], inputs["v"])
    api = flashinfer.magi_ffa.flex_flash_attn(**inputs)
    ref = magi_ffa_flex_nhd_trace.reference(**inputs)
    _assert_finite(api, ref)
    _check(magi_ffa_flex_nhd_trace, ref, api)
    torch.cuda.synchronize()
