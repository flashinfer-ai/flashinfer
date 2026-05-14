"""Reference correctness test for the rope_quantize_fp8 trace API."""

import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


def test_rope_quantize_fp8_reference_correctness():
    """flashinfer.rope.rope_quantize_fp8 (GQA layout) kernel vs reference."""
    from flashinfer.rope import rope_quantize_fp8
    from flashinfer.trace.templates.rope import rope_quantize_fp8_trace

    inputs = rope_quantize_fp8_trace.init(
        nnz=16,
        num_q_heads=8,
        num_k_heads=2,
        rope_dim=64,
        no_rope_dim=64,
        max_seq_len=4096,
        rotary_dim=64,
    )
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
    _close(q_r_api.float(), q_r_ref.float(), atol=1e-2, rtol=2e-1)
    _close(k_r_api.float(), k_r_ref.float(), atol=1e-2, rtol=2e-1)
    _close(q_n_api.float(), q_n_ref.float(), atol=1e-2, rtol=2e-1)
    _close(k_n_api.float(), k_n_ref.float(), atol=1e-2, rtol=2e-1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
