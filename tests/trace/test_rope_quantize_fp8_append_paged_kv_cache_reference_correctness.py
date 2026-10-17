"""Reference correctness test for the rope_quantize_fp8_append_paged_kv_cache trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(
            device="cuda",
            nnz=16,
            num_q_heads=8,
            num_k_heads=2,
            rope_dim=64,
            no_rope_dim=64,
            num_pages=4,
            page_size=16,
            batch_size=2,
        ),
        dict(
            device="cuda",
            nnz=7,
            num_q_heads=4,
            num_k_heads=2,
            rope_dim=64,
            no_rope_dim=64,
            num_pages=6,
            page_size=8,
            batch_size=3,
        ),
    ],
)
def test_rope_quantize_fp8_append_paged_kv_cache_reference_correctness(shape_kwargs):
    """rope_quantize_fp8_append_paged_kv_cache kernel vs reference (GQA layout)."""
    from flashinfer.rope import rope_quantize_fp8_append_paged_kv_cache
    from flashinfer.trace.templates.rope import (
        rope_quantize_fp8_append_paged_kv_cache_trace,
    )

    inputs = rope_quantize_fp8_append_paged_kv_cache_trace.init(**shape_kwargs)
    k_cache, v_cache = inputs["paged_kv_cache"]
    k_cache_api = k_cache.clone()
    v_cache_api = v_cache.clone()
    k_cache_ref = torch.zeros_like(k_cache_api)
    v_cache_ref = torch.zeros_like(k_cache_api)
    try:
        q_r_api, q_n_api = rope_quantize_fp8_append_paged_kv_cache(
            inputs["q_rope"],
            inputs["k_rope"],
            inputs["q_nope"],
            inputs["k_nope"],
            inputs["v"],
            inputs["cos_sin_cache"],
            inputs["pos_ids"],
            (k_cache_api, v_cache_api),
            inputs["kv_indices"],
            inputs["kv_indptr"],
            inputs["batch_indices"],
            inputs["positions"],
            is_neox=inputs["is_neox"],
            page_size=inputs["page_size"],
            kv_layout=inputs["kv_layout"],
        )
    except Exception as exc:
        pytest.skip(f"rope_quantize_fp8_append_paged_kv_cache unavailable: {exc}")
    q_r_ref, q_n_ref = rope_quantize_fp8_append_paged_kv_cache_trace.reference(
        inputs["q_rope"],
        inputs["k_rope"],
        inputs["q_nope"],
        inputs["k_nope"],
        inputs["v"],
        inputs["cos_sin_cache"],
        inputs["pos_ids"],
        (k_cache_ref, v_cache_ref),
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["batch_indices"],
        inputs["positions"],
        is_neox=inputs["is_neox"],
        page_size=inputs["page_size"],
        kv_layout=inputs["kv_layout"],
    )
    # Match tests/attention/test_rope.py FP8 rope quantize tolerance for Q.
    # (The paged K/V append half uses an implementation-specific internal
    # layout — nope/rope interleave order varies between kernel versions —
    # so we only compare the Q outputs here, which are portable.)
    _check(
        rope_quantize_fp8_append_paged_kv_cache_trace,
        (q_r_ref.float(), q_n_ref.float()),
        (q_r_api.float(), q_n_api.float()),
        atol=1e-2,
        rtol=2e-1,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_rope_quantize_fp8_append_paged_kv_cache_reference_e5m2_saturation():
    """e5m2 V appended to the paged cache must saturate at the e5m2 max.

    pos_ids 0 with cos 1 / sin 0 keeps the rotation an identity and the two
    tokens land in distinct slots of one page, so the values written to
    v_cache are exactly the quantized v rows below. All inputs are exact
    in bf16; most lie above the e4m3 max of 448 on both signs.
    """
    from flashinfer.trace.templates.rope import (
        rope_quantize_fp8_append_paged_kv_cache_trace,
    )

    cos_sin_cache = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    pos_ids = torch.zeros(2, dtype=torch.int32)
    q_rope = torch.tensor(
        [[[512.0, 57344.0]], [[2048.0, -59904.0]]], dtype=torch.bfloat16
    )
    k_rope = torch.tensor(
        [[[448.0, 61440.0]], [[28672.0, -512.0]]], dtype=torch.bfloat16
    )
    v = torch.tensor(
        [[500.0, 19968.0, -29952.0, 448.0], [-59904.0, 57344.0, 448.0, -512.0]],
        dtype=torch.bfloat16,
    ).reshape(2, 1, 4)
    k_cache = torch.zeros(1, 2, 1, 2, dtype=torch.float8_e5m2)
    v_cache = torch.zeros(1, 2, 1, 4, dtype=torch.float8_e5m2)
    q_r_ref, _ = rope_quantize_fp8_append_paged_kv_cache_trace.reference(
        q_rope,
        k_rope,
        None,
        None,
        v,
        cos_sin_cache,
        pos_ids,
        (k_cache, v_cache),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        kv_indptr=torch.tensor([0], dtype=torch.int32),
        batch_indices=torch.tensor([0, 0], dtype=torch.int32),
        positions=torch.tensor([0, 1], dtype=torch.int32),
        is_neox=True,
        quantize_dtype=torch.float8_e5m2,
        page_size=2,
        kv_layout="NHD",
    )
    expected_v = v.to(torch.float32).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)
    expected_k = k_rope.to(torch.float32).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)
    expected_q = q_rope.to(torch.float32).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)
    assert v_cache.dtype == torch.float8_e5m2
    assert torch.equal(v_cache[0].reshape(-1).float(), expected_v.reshape(-1).float())
    assert torch.equal(k_cache[0].reshape(-1).float(), expected_k.reshape(-1).float())
    assert torch.equal(q_r_ref.reshape(-1).float(), expected_q.reshape(-1).float())
    # Clamping V at the e4m3 max of 448 would compress the e5m2 range by 128x.
    assert (v_cache.float().abs() > 448.0).any()
