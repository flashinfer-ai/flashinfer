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


_SHAPE_CASES = [
    pytest.param(
        dict(num_tokens_q=128),
        False,
        id="self-attention-out",
    ),
    pytest.param(
        dict(
            num_tokens_q=96,
            num_tokens_kv=80,
            num_ranges=3,
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=128,
        ),
        True,
        id="cross-attention-out-lse",
    ),
]


@pytest.mark.parametrize(("shape_kwargs", "return_lse"), _SHAPE_CASES)
def test_flex_flash_attn_init_honors_variable_axes(shape_kwargs, return_lse):
    """The replay initializer must honor every variable trace axis."""
    from flashinfer.trace.templates.attention import magi_ffa_flex_nhd_trace

    inputs = magi_ffa_flex_nhd_trace.init(
        **shape_kwargs, return_lse=return_lse, device="cpu"
    )
    num_tokens_q = shape_kwargs["num_tokens_q"]
    num_tokens_kv = shape_kwargs.get("num_tokens_kv", num_tokens_q)
    num_ranges = shape_kwargs.get("num_ranges", min(2, num_tokens_q))

    assert inputs["q"].shape[0] == num_tokens_q
    assert inputs["k"].shape[0] == num_tokens_kv
    assert inputs["v"].shape[0] == num_tokens_kv
    assert inputs["q_ranges"].shape == (num_ranges, 2)
    assert inputs["k_ranges"].shape == (num_ranges, 2)
    assert inputs["attn_type_map"].shape == (num_ranges,)
    assert inputs.get("return_lse", False) is return_lse


@pytest.mark.parametrize(
    ("shape_kwargs", "return_lse"),
    _SHAPE_CASES,
)
def test_flex_flash_attn_reference_correctness(shape_kwargs, return_lse):
    """flashinfer.magi_ffa.flex_flash_attn kernel vs reference."""
    pytest.importorskip("magi_attention")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for MagiAttention FFA")

    import flashinfer.magi_ffa
    from flashinfer.trace.templates.attention import magi_ffa_flex_nhd_trace

    inputs = magi_ffa_flex_nhd_trace.init(**shape_kwargs, return_lse=return_lse)
    _assert_finite(inputs["q"], inputs["k"], inputs["v"])
    api = flashinfer.magi_ffa.flex_flash_attn(**inputs)
    ref = magi_ffa_flex_nhd_trace.reference(**inputs)
    api_outputs = api if isinstance(api, tuple) else (api,)
    ref_outputs = ref if isinstance(ref, tuple) else (ref,)
    _assert_finite(*api_outputs, *ref_outputs)
    _check(magi_ffa_flex_nhd_trace, ref, api)
    torch.cuda.synchronize()
