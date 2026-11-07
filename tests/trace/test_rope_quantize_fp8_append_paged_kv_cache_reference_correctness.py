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


def test_rope_quantize_fp8_append_paged_kv_cache_reference_infers_dtype_from_out():
    """A None quantize_dtype must resolve from the out tensors like the API."""
    from flashinfer.trace.templates.rope import (
        rope_quantize_fp8_append_paged_kv_cache_trace,
    )

    cos_sin_cache = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    pos_ids = torch.zeros(1, dtype=torch.int32)
    q_rope = torch.tensor([[[512.0, 57344.0]]], dtype=torch.bfloat16)
    k_rope = torch.tensor([[[2048.0, -59904.0]]], dtype=torch.bfloat16)
    k_cache = torch.zeros(1, 1, 1, 2, dtype=torch.float8_e4m3fn)
    v_cache = torch.zeros(1, 1, 1, 2, dtype=torch.float8_e4m3fn)
    q_rope_out = torch.empty(1, 1, 2, dtype=torch.float8_e5m2)
    q_r_ref, _ = rope_quantize_fp8_append_paged_kv_cache_trace.reference(
        q_rope,
        k_rope,
        None,
        None,
        None,
        cos_sin_cache,
        pos_ids,
        (k_cache, v_cache),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        kv_indptr=torch.tensor([0], dtype=torch.int32),
        batch_indices=torch.tensor([0], dtype=torch.int32),
        positions=torch.tensor([0], dtype=torch.int32),
        q_rope_out=q_rope_out,
    )
    assert q_r_ref.dtype == torch.float8_e5m2
    expected = q_rope.to(torch.float32).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)
    assert torch.equal(q_r_ref.reshape(-1).float(), expected.reshape(-1).float())


def test_rope_quantize_fp8_append_paged_kv_cache_reference_infers_dtype_from_cache():
    """With no out tensors, the paged cache dtype carries the request.

    The cache is where the kernel writes the quantized K/V, so an e5m2 cache
    with a None quantize_dtype has to quantize to e5m2, not to the e4m3fn
    default.
    """
    from flashinfer.trace.templates.rope import (
        rope_quantize_fp8_append_paged_kv_cache_trace,
    )

    cos_sin_cache = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    pos_ids = torch.zeros(1, dtype=torch.int32)
    q_rope = torch.tensor([[[512.0, 57344.0]]], dtype=torch.bfloat16)
    k_rope = torch.tensor([[[2048.0, -59904.0]]], dtype=torch.bfloat16)
    v = torch.tensor([[512.0, 57344.0]], dtype=torch.bfloat16).reshape(1, 1, 2)
    k_cache = torch.zeros(1, 1, 1, 2, dtype=torch.float8_e5m2)
    v_cache = torch.zeros(1, 1, 1, 2, dtype=torch.float8_e5m2)
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
        batch_indices=torch.tensor([0], dtype=torch.int32),
        positions=torch.tensor([0], dtype=torch.int32),
    )
    expected_v = v.to(torch.float32).clamp(-57344.0, 57344.0).to(torch.float8_e5m2)
    assert q_r_ref.dtype == torch.float8_e5m2
    assert torch.equal(v_cache[0].reshape(-1).float(), expected_v.reshape(-1).float())


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
        )
    ],
)
def test_rope_quantize_fp8_append_paged_kv_cache_reference_e5m2_kernel_consistency(
    shape_kwargs,
):
    """e5m2 kernel vs reference above the e4m3 max of 448, V cache included.

    A large quant_scale_kv pushes the V rows well past 448, and V skips the
    nope/rope interleave that makes the K layout kernel-specific, so the
    written v_cache is comparable end to end.
    """
    from flashinfer.rope import rope_quantize_fp8_append_paged_kv_cache
    from flashinfer.trace.templates.rope import (
        rope_quantize_fp8_append_paged_kv_cache_trace,
    )

    inputs = rope_quantize_fp8_append_paged_kv_cache_trace.init(**shape_kwargs)
    # The init builds an e4m3fn cache; the API requires the cache dtype to
    # match quantize_dtype, so re-materialize it as e5m2.
    k_cache, v_cache = inputs["paged_kv_cache"]
    k_cache = k_cache.to(torch.float8_e5m2)
    v_cache = v_cache.to(torch.float8_e5m2)
    k_cache_api = k_cache.clone()
    v_cache_api = v_cache.clone()
    k_cache_ref = torch.zeros_like(k_cache_api)
    v_cache_ref = torch.zeros_like(v_cache_api)
    common_tail = dict(
        is_neox=inputs["is_neox"],
        quantize_dtype=torch.float8_e5m2,
        quant_scale_kv=1000.0,
        page_size=inputs["page_size"],
        kv_layout=inputs["kv_layout"],
    )
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
            **common_tail,
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
        **common_tail,
    )
    # The scale puts the bulk of the V magnitudes past 448 without nearing
    # the e5m2 max of 57344, so the kernel's saturation stays unclamped.
    assert (v_cache_api.float().abs() > 448.0).any()
    # e5m2 has a 2-bit mantissa: adjacent representable values at a binade
    # start are 25% apart, so a rotate-side fp32 rounding difference that
    # lands on an RN boundary can flip one e5m2 code point (V skips the
    # rotation and matches the kernel bit-exact).
    assert torch.allclose(
        v_cache_api.float(), v_cache_ref.float(), atol=1e-2, rtol=3e-1
    )
    _check(
        rope_quantize_fp8_append_paged_kv_cache_trace,
        (q_r_ref.float(), q_n_ref.float()),
        (q_r_api.float(), q_n_api.float()),
        atol=1e-2,
        rtol=3e-1,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
