"""Reference correctness test for the rope_quantize_fp8 trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(
            nnz=16,
            num_q_heads=8,
            num_k_heads=2,
            rope_dim=64,
            no_rope_dim=64,
            max_seq_len=4096,
            rotary_dim=64,
        ),
        dict(
            nnz=8,
            num_q_heads=4,
            num_k_heads=1,
            rope_dim=64,
            no_rope_dim=64,
            max_seq_len=2048,
            rotary_dim=64,
        ),
    ],
)
def test_rope_quantize_fp8_reference_correctness(shape_kwargs):
    """flashinfer.rope.rope_quantize_fp8 (GQA layout) kernel vs reference."""
    from flashinfer.rope import rope_quantize_fp8
    from flashinfer.trace.templates.rope import rope_quantize_fp8_trace

    inputs = rope_quantize_fp8_trace.init(**shape_kwargs)
    _assert_finite(
        inputs["q_rope"], inputs["k_rope"], inputs["q_nope"], inputs["k_nope"]
    )
    q_r_api, k_r_api, q_n_api, k_n_api = rope_quantize_fp8(
        inputs["q_rope"],
        inputs["k_rope"],
        inputs["q_nope"],
        inputs["k_nope"],
        inputs["cos_sin_cache"],
        inputs["pos_ids"],
        is_neox=inputs["is_neox"],
    )
    q_r_ref, k_r_ref, q_n_ref, k_n_ref = rope_quantize_fp8_trace.reference(
        inputs["q_rope"],
        inputs["k_rope"],
        inputs["q_nope"],
        inputs["k_nope"],
        inputs["cos_sin_cache"],
        inputs["pos_ids"],
        is_neox=inputs["is_neox"],
    )
    _assert_finite(
        q_r_api, k_r_api, q_n_api, k_n_api, q_r_ref, k_r_ref, q_n_ref, k_n_ref
    )
    # Match tolerance used by tests/attention/test_rope.py's rope_quantize_fp8
    # coverage: generous rtol (2e-1) absorbs single-ULP FP8 rounding between
    # the CUDA kernel and torch's FP8 cast while still catching real bugs.
    _check(
        rope_quantize_fp8_trace,
        (q_r_ref.float(), k_r_ref.float(), q_n_ref.float(), k_n_ref.float()),
        (q_r_api.float(), k_r_api.float(), q_n_api.float(), k_n_api.float()),
        atol=1e-2,
        rtol=2e-1,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.mark.parametrize(
    "quantize_dtype,sat_min,sat_max",
    [
        (torch.float8_e5m2, -57344.0, 57344.0),
        (torch.float8_e4m3fn, -448.0, 448.0),
    ],
)
def test_rope_quantize_fp8_reference_saturation_follows_dtype(
    quantize_dtype, sat_min, sat_max
):
    """The clamp applied before the FP8 cast must follow quantize_dtype.

    pos_ids 0 with cos 1 / sin 0 keeps the rotation an identity, so the
    values reaching the quantizer are exactly the ones built below. All of
    them are exact in bf16; every value past the first lies above the e4m3
    max of 448, on both signs, with 61440 past the e5m2 max of 57344.
    """
    from flashinfer.trace.templates.rope import rope_quantize_fp8_trace

    cos_sin_cache = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    pos_ids = torch.zeros(4, dtype=torch.int32)
    values = torch.tensor(
        [448.0, 512.0, 2048.0, 28672.0, 57344.0, 61440.0, -59904.0, -512.0],
        dtype=torch.bfloat16,
    )
    q_rope = values.reshape(4, 1, 2)
    k_rope = q_rope.clone()
    q_r_ref, k_r_ref, _, _ = rope_quantize_fp8_trace.reference(
        q_rope,
        k_rope,
        None,
        None,
        cos_sin_cache,
        pos_ids,
        quantize_dtype=quantize_dtype,
    )
    expected = values.to(torch.float32).clamp(sat_min, sat_max).to(quantize_dtype)
    assert q_r_ref.dtype == quantize_dtype
    assert torch.equal(q_r_ref.reshape(-1).to(torch.float32), expected.float())
    assert torch.equal(k_r_ref.reshape(-1).to(torch.float32), expected.float())
    if quantize_dtype == torch.float8_e5m2:
        # e5m2 represents values up to 57344; clamping them at the e4m3
        # max of 448 would compress the representable range by 128x.
        assert (q_r_ref.float().abs() > 448.0).any()
