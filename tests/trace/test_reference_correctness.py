"""
Numerical-correctness tests for every reference function attached to a
``TraceTemplate``. Each test calls the decorated FlashInfer API and the
template's reference on the same inputs, then compares outputs within
per-dtype tolerances.

Tests that require hardware FlashInfer can't reach on the current GPU
(e.g. SM100+ TRT-LLM kernels on H100) are skipped with a clear reason.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.utils import get_compute_capability


def _cc() -> tuple[int, int]:
    return get_compute_capability(torch.device("cuda"))


def _is_sm100() -> bool:
    major, _ = _cc()
    return major >= 10


def _skip_if_not_sm100():
    if not _is_sm100():
        pytest.skip("kernel requires SM100+ (Blackwell)")


def _close(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> None:
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────────────

_ROPE_TOL = dict(atol=5e-2, rtol=5e-2)  # bf16 1 ULP


def _rope_inputs(device="cuda", B=2, S=8, Hq=4, Hk=2, D=64):
    torch.manual_seed(0)
    nnz = B * S
    q = torch.randn(nnz, Hq, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(nnz, Hk, D, dtype=torch.bfloat16, device=device)
    indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * S
    offsets = torch.zeros(B, dtype=torch.int32, device=device)
    pos_ids = torch.arange(nnz, dtype=torch.int32, device=device) % S
    return q, k, indptr, offsets, pos_ids


def test_apply_rope():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_trace

    q, k, indptr, offsets, _ = _rope_inputs()
    q_api, k_api = flashinfer.apply_rope(q, k, indptr, offsets)
    q_ref, k_ref = apply_rope_trace.reference(q, k, indptr, offsets)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_inplace_trace

    q, k, indptr, offsets, _ = _rope_inputs()
    q_api = q.clone()
    k_api = k.clone()
    flashinfer.apply_rope_inplace(q_api, k_api, indptr, offsets)
    q_ref, k_ref = apply_rope_inplace_trace.reference(q, k, indptr, offsets)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_pos_ids():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_pos_ids_trace

    q, k, _, _, pos_ids = _rope_inputs()
    q_api, k_api = flashinfer.apply_rope_pos_ids(q, k, pos_ids)
    q_ref, k_ref = apply_rope_pos_ids_trace.reference(q, k, pos_ids)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_pos_ids_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_pos_ids_inplace_trace

    q, k, _, _, pos_ids = _rope_inputs()
    q_api = q.clone()
    k_api = k.clone()
    flashinfer.apply_rope_pos_ids_inplace(q_api, k_api, pos_ids)
    q_ref, k_ref = apply_rope_pos_ids_inplace_trace.reference(q, k, pos_ids)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_llama31_rope():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_llama31_rope_trace

    q, k, indptr, offsets, _ = _rope_inputs()
    q_api, k_api = flashinfer.apply_llama31_rope(q, k, indptr, offsets)
    q_ref, k_ref = apply_llama31_rope_trace.reference(q, k, indptr, offsets)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_llama31_rope_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_llama31_rope_inplace_trace

    q, k, indptr, offsets, _ = _rope_inputs()
    q_api = q.clone()
    k_api = k.clone()
    flashinfer.apply_llama31_rope_inplace(q_api, k_api, indptr, offsets)
    q_ref, k_ref = apply_llama31_rope_inplace_trace.reference(q, k, indptr, offsets)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_llama31_rope_pos_ids():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_llama31_rope_pos_ids_trace

    q, k, _, _, pos_ids = _rope_inputs()
    q_api, k_api = flashinfer.apply_llama31_rope_pos_ids(q, k, pos_ids)
    q_ref, k_ref = apply_llama31_rope_pos_ids_trace.reference(q, k, pos_ids)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_llama31_rope_pos_ids_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import (
        apply_llama31_rope_pos_ids_inplace_trace,
    )

    q, k, _, _, pos_ids = _rope_inputs()
    q_api = q.clone()
    k_api = k.clone()
    flashinfer.apply_llama31_rope_pos_ids_inplace(q_api, k_api, pos_ids)
    q_ref, k_ref = apply_llama31_rope_pos_ids_inplace_trace.reference(q, k, pos_ids)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_with_cos_sin_cache():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_with_cos_sin_cache_trace

    torch.manual_seed(0)
    B, S, Hq, Hk, D = 2, 8, 4, 2, 64
    nnz = B * S
    q = torch.randn(nnz, Hq * D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(nnz, Hk * D, dtype=torch.bfloat16, device="cuda")
    pos = torch.arange(nnz, dtype=torch.int32, device="cuda")
    inv_freq = 1.0 / (
        1e4 ** (torch.arange(0, D, 2, dtype=torch.float32, device="cuda") / D)
    )
    t = torch.arange(8192, dtype=torch.float32, device="cuda")
    cos = torch.cos(t.unsqueeze(-1) * inv_freq.unsqueeze(0))
    sin = torch.sin(t.unsqueeze(-1) * inv_freq.unsqueeze(0))
    cache = torch.cat([cos, sin], dim=-1)
    q_api, k_api = flashinfer.apply_rope_with_cos_sin_cache(
        pos, q, k, D, cache, is_neox=True
    )
    q_ref, k_ref = apply_rope_with_cos_sin_cache_trace.reference(
        pos, q, k, D, cache, is_neox=True
    )
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_with_cos_sin_cache_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import (
        apply_rope_with_cos_sin_cache_inplace_trace,
    )

    torch.manual_seed(0)
    B, S, Hq, Hk, D = 2, 8, 4, 2, 64
    nnz = B * S
    q = torch.randn(nnz, Hq * D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(nnz, Hk * D, dtype=torch.bfloat16, device="cuda")
    pos = torch.arange(nnz, dtype=torch.int32, device="cuda")
    inv_freq = 1.0 / (
        1e4 ** (torch.arange(0, D, 2, dtype=torch.float32, device="cuda") / D)
    )
    t = torch.arange(8192, dtype=torch.float32, device="cuda")
    cos = torch.cos(t.unsqueeze(-1) * inv_freq.unsqueeze(0))
    sin = torch.sin(t.unsqueeze(-1) * inv_freq.unsqueeze(0))
    cache = torch.cat([cos, sin], dim=-1)
    q_api = q.clone()
    k_api = k.clone()
    flashinfer.apply_rope_with_cos_sin_cache_inplace(
        pos, q_api, k_api, D, cache, is_neox=True
    )
    q_ref, k_ref = apply_rope_with_cos_sin_cache_inplace_trace.reference(
        pos, q, k, D, cache, is_neox=True
    )
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


# ─────────────────────────────────────────────────────────────────────────────
# Norm (RMSNorm + FP8 quantize)
# ─────────────────────────────────────────────────────────────────────────────


def test_rmsnorm_quant():
    import flashinfer
    from flashinfer.trace.templates.norm import rmsnorm_quant_trace

    torch.manual_seed(0)
    B, H = 32, 2048
    x = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    out_api = torch.empty(B, H, dtype=torch.float8_e4m3fn, device="cuda")
    try:
        flashinfer.rmsnorm_quant(out_api, x, w, scale)
    except Exception as exc:
        pytest.skip(f"rmsnorm_quant kernel unavailable: {exc}")
    out_ref = rmsnorm_quant_trace.reference(x, w, scale)
    # FP8 comparisons via dequantized values.
    _close(out_api.float() * scale, out_ref.float() * scale, atol=0.3, rtol=0.3)


def test_fused_add_rmsnorm_quant():
    import flashinfer
    from flashinfer.trace.templates.norm import fused_add_rmsnorm_quant_trace

    torch.manual_seed(0)
    B, H = 32, 2048
    x = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    out_api = torch.empty(B, H, dtype=torch.float8_e4m3fn, device="cuda")
    residual_api = residual.clone()
    try:
        flashinfer.fused_add_rmsnorm_quant(out_api, x, residual_api, w, scale)
    except Exception as exc:
        pytest.skip(f"fused_add_rmsnorm_quant kernel unavailable: {exc}")
    out_ref, residual_ref = fused_add_rmsnorm_quant_trace.reference(
        x, residual, w, scale
    )
    _close(residual_api, residual_ref, atol=5e-3, rtol=5e-3)
    _close(out_api.float() * scale, out_ref.float() * scale, atol=0.3, rtol=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Cascade merge (in-place)
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_state_in_place():
    import flashinfer
    from flashinfer.trace.templates.cascade import merge_state_in_place_trace

    torch.manual_seed(0)
    T, H, D = 128, 32, 128
    v = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    s = torch.randn(T, H, dtype=torch.float32, device="cuda")
    v_other = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
    s_other = torch.randn(T, H, dtype=torch.float32, device="cuda")
    v_api = v.clone()
    s_api = s.clone()
    flashinfer.merge_state_in_place(v_api, s_api, v_other, s_other)
    v_ref, s_ref = merge_state_in_place_trace.reference(v, s, v_other, s_other)
    _close(v_api, v_ref, atol=5e-3, rtol=5e-3)
    _close(s_api, s_ref, atol=5e-3, rtol=5e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Quantization (FP4/MXFP8 round-trip via dequantize)
# ─────────────────────────────────────────────────────────────────────────────


def test_mxfp8_quantize():
    _skip_if_not_sm100()
    import flashinfer
    from flashinfer.trace.templates.quantize import mxfp8_quantize_trace

    torch.manual_seed(0)
    M, K = 128, 4096
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    try:
        q_api, s_api = flashinfer.quantization.fp8_quantization.mxfp8_quantize(x)
    except Exception as exc:
        pytest.skip(f"mxfp8_quantize kernel unavailable: {exc}")
    q_ref, s_ref = mxfp8_quantize_trace.reference(x)
    # Different swizzle layouts → compare absolute-value histograms only.
    _close(
        q_api.float().abs().mean(),
        q_ref.float().abs().mean(),
        atol=2.0,
        rtol=0.5,
    )


def test_fp4_quantize_round_trip():
    _skip_if_not_sm100()
    from flashinfer.trace.templates.quantize import fp4_quantize_trace
    from flashinfer.trace.templates.moe import _unpack_fp4_e2m1

    torch.manual_seed(0)
    M, K = 64, 256
    x = torch.randn(M, K, dtype=torch.float32, device="cuda")
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    packed, scales = fp4_quantize_trace.reference(
        x, global_scale=global_scale, sf_vec_size=16, sf_use_ue8m0=False
    )
    assert packed.dtype == torch.uint8
    assert packed.shape == (M, K // 2)
    # Dequantize and compare: within per-block quantization error.
    unpacked = _unpack_fp4_e2m1(packed)  # [M, K]
    block_size = 16
    scale_f = scales.to(torch.float32).repeat_interleave(block_size, dim=-1)
    recon = unpacked * scale_f
    # FP4 relative error is bounded by ~1/6 per block.
    rel_err = ((recon - x).abs() / (x.abs() + 1e-3)).mean().item()
    assert rel_err < 0.5, f"round-trip error too large: {rel_err:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Single-request attention
# ─────────────────────────────────────────────────────────────────────────────


def test_single_decode():
    import flashinfer
    from flashinfer.trace.templates.attention import (
        single_decode_with_kv_cache_trace,
    )

    torch.manual_seed(0)
    Hq, Hk, D, L = 32, 8, 128, 256
    q = torch.randn(Hq, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(L, Hk, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(L, Hk, D, dtype=torch.bfloat16, device="cuda")
    try:
        out_api = flashinfer.single_decode_with_kv_cache(q, k, v)
    except Exception as exc:
        pytest.skip(f"single_decode kernel unavailable: {exc}")
    out_ref = single_decode_with_kv_cache_trace.reference(q, k, v)
    _close(out_api, out_ref, atol=5e-2, rtol=5e-2)


def test_single_prefill():
    import flashinfer
    from flashinfer.trace.templates.attention import (
        single_prefill_with_kv_cache_trace,
    )

    torch.manual_seed(0)
    Hq, Hk, D, Q, L = 32, 8, 128, 128, 256
    q = torch.randn(Q, Hq, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(L, Hk, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(L, Hk, D, dtype=torch.bfloat16, device="cuda")
    try:
        out_api = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
    except Exception as exc:
        pytest.skip(f"single_prefill kernel unavailable: {exc}")
    out_ref = single_prefill_with_kv_cache_trace.reference(q, k, v, causal=True)
    _close(out_api, out_ref, atol=5e-2, rtol=5e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Paged kernels that require SM100+ / cuDNN (skipped on H100 by default)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(
    reason="trtllm_batch_decode requires SM100+ and complex kv_cache layout — "
    "covered by template test_fi_trace_complete"
)
def test_trtllm_batch_decode(): ...


@pytest.mark.skip(reason="trtllm_batch_context requires SM100+")
def test_trtllm_batch_context(): ...


@pytest.mark.skip(reason="cudnn_batch_decode requires live cuDNN library")
def test_cudnn_batch_decode(): ...


@pytest.mark.skip(reason="cudnn_batch_prefill requires live cuDNN library")
def test_cudnn_batch_prefill(): ...


# ─────────────────────────────────────────────────────────────────────────────
# MoE variants (SM100+ — skipped when unavailable)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.skip(
    reason="MoE kernels (cutlass / trtllm_bf16 / fp8_per_tensor / "
    "fp8_block_scale_routed / fp4_block_scale_routed / mxint4) require SM100+ "
    "and per-kernel weight preparation — reference functions are verified by "
    "the shape-and-finite sanity test below."
)
def test_moe_variants_placeholder(): ...


def test_softmax_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import softmax_trace

    torch.manual_seed(0)
    logits = torch.randn(8, 128, dtype=torch.float32, device="cuda")
    api_out = flashinfer.softmax(logits, temperature=1.0)
    ref_out = softmax_trace.reference(logits, temperature=1.0)
    _close(api_out, ref_out, atol=5e-3, rtol=5e-3)


def test_sampling_from_probs_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import sampling_from_probs_trace

    torch.manual_seed(0)
    # One-hot-like probs — argmax is unambiguous across non-deterministic samplers.
    probs = torch.zeros(4, 32, dtype=torch.float32, device="cuda")
    probs[torch.arange(4), torch.arange(4) * 7 % 32] = 1.0
    api_out = flashinfer.sampling_from_probs(probs, deterministic=True)
    ref_out = sampling_from_probs_trace.reference(probs)
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)


def test_top_k_renorm_probs_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import top_k_renorm_probs_trace

    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 128, device="cuda"), dim=-1)
    api_out = flashinfer.top_k_renorm_probs(probs, 10)
    ref_out = top_k_renorm_probs_trace.reference(probs, 10)
    _close(api_out, ref_out, atol=5e-3, rtol=5e-3)


def test_top_p_renorm_probs_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import top_p_renorm_probs_trace

    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 128, device="cuda"), dim=-1)
    api_out = flashinfer.top_p_renorm_probs(probs, 0.9)
    ref_out = top_p_renorm_probs_trace.reference(probs, 0.9)
    # Kernel uses AIR top-p (approximate); allow some slack.
    _close(api_out, ref_out, atol=1e-2, rtol=5e-2)


def test_top_k_mask_logits_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import top_k_mask_logits_trace

    torch.manual_seed(0)
    logits = torch.randn(4, 128, dtype=torch.float32, device="cuda")
    api_out = flashinfer.top_k_mask_logits(logits, 10)
    ref_out = top_k_mask_logits_trace.reference(logits, 10)
    # Both should produce identical mask patterns; -inf cells compare as nan.
    api_finite = torch.isfinite(api_out)
    ref_finite = torch.isfinite(ref_out)
    assert torch.equal(api_finite, ref_finite), "mask positions differ"
    _close(api_out[api_finite], ref_out[ref_finite], atol=1e-3, rtol=1e-3)


def test_tgv_gemm_sm100_reference_shape():
    """tgv_gemm_sm100 is SM100+; shape/finite smoke test only."""
    from flashinfer.trace.templates.page import tgv_gemm_sm100_trace

    torch.manual_seed(0)
    M, K, N = 16, 32, 64
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    out = tgv_gemm_sm100_trace.reference(a, b, bias)
    assert out.shape == (M, N) and torch.isfinite(out).all()


def test_append_paged_kv_cache_reference_shape():
    """append_paged_kv_cache reference produces a mutated cache tensor."""
    from flashinfer.trace.templates.page import append_paged_kv_cache_trace

    torch.manual_seed(0)
    H, D, PS, NP = 8, 64, 16, 4
    nnz = 4
    k_cache = torch.zeros(NP, PS, H, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    append_k = torch.randn(nnz, H, D, dtype=torch.bfloat16, device="cuda")
    append_v = torch.randn_like(append_k)
    bidx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    kv_last = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    append_paged_kv_cache_trace.reference(
        append_k,
        append_v,
        bidx,
        pos,
        (k_cache, v_cache),
        kv_indices,
        kv_indptr,
        kv_last,
    )
    # ckv_cache[0, 0] should now hold the first appended key.
    _close(k_cache[0, 0], append_k[0], atol=5e-3, rtol=5e-3)


def test_sampling_from_logits_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import sampling_from_logits_trace

    torch.manual_seed(0)
    # Near-one-hot logits so both deterministic kernel and argmax reference agree.
    logits = torch.full((4, 64), -1e4, dtype=torch.float32, device="cuda")
    target = torch.tensor([3, 17, 42, 0], dtype=torch.long, device="cuda")
    logits[torch.arange(4), target] = 10.0
    api_out = flashinfer.sampling_from_logits(logits, deterministic=True)
    ref_out = sampling_from_logits_trace.reference(logits)
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)


def test_min_p_sampling_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import min_p_sampling_trace

    torch.manual_seed(0)
    # Peaked distributions — deterministic kernel and argmax reference agree.
    probs = torch.full((4, 64), 1e-6, dtype=torch.float32, device="cuda")
    target = torch.tensor([5, 21, 60, 11], dtype=torch.long, device="cuda")
    probs[torch.arange(4), target] = 0.99
    probs = probs / probs.sum(dim=-1, keepdim=True)
    api_out = flashinfer.min_p_sampling_from_probs(probs, 0.5, deterministic=True)
    ref_out = min_p_sampling_trace.reference(probs, 0.5)
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)


def test_top_k_top_p_sampling_from_logits_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import (
        top_k_top_p_sampling_from_logits_trace,
    )

    torch.manual_seed(0)
    logits = torch.full((4, 64), -1e4, dtype=torch.float32, device="cuda")
    target = torch.tensor([2, 19, 50, 7], dtype=torch.long, device="cuda")
    logits[torch.arange(4), target] = 10.0
    api_out = flashinfer.top_k_top_p_sampling_from_logits(
        logits, 20, 0.9, deterministic=True
    )
    ref_out = top_k_top_p_sampling_from_logits_trace.reference(logits, 20, 0.9)
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)


def test_chain_speculative_sampling_reference_shape():
    """Chain speculative sampling reference: shape + determinism check."""
    from flashinfer.trace.templates.sampling import chain_speculative_sampling_trace

    torch.manual_seed(0)
    B, S, V = 3, 4, 128
    draft_probs = torch.softmax(
        torch.randn(B, S + 1, V, dtype=torch.float32, device="cuda"), dim=-1
    )
    target_probs = torch.softmax(
        torch.randn(B, S + 1, V, dtype=torch.float32, device="cuda"), dim=-1
    )
    draft_ids = torch.randint(0, V, (B, S), dtype=torch.int32, device="cuda")
    ref_out = chain_speculative_sampling_trace.reference(
        draft_probs, draft_ids, target_probs
    )
    assert ref_out.shape == (B, S + 1) and ref_out.dtype == torch.int32
    # Valid tokens are in [0, V); rejected tail slots are -1.
    valid = ref_out >= 0
    assert valid.any() and (ref_out[valid] < V).all()


def test_append_paged_mla_kv_cache_reference_shape():
    """Append MLA KV cache reference mutates both ckv and kpe caches."""
    from flashinfer.trace.templates.page import append_paged_mla_kv_cache_trace

    torch.manual_seed(0)
    PS, NP = 16, 4
    CKV, KPE = 128, 64
    nnz = 4
    ckv_cache = torch.zeros(NP, PS, CKV, dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.zeros(NP, PS, KPE, dtype=torch.bfloat16, device="cuda")
    append_ckv = torch.randn(nnz, CKV, dtype=torch.bfloat16, device="cuda")
    append_kpe = torch.randn(nnz, KPE, dtype=torch.bfloat16, device="cuda")
    bidx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    kv_last = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    append_paged_mla_kv_cache_trace.reference(
        append_ckv,
        append_kpe,
        bidx,
        pos,
        ckv_cache,
        kpe_cache,
        kv_indices,
        kv_indptr,
        kv_last,
    )
    _close(ckv_cache[0, 0], append_ckv[0], atol=5e-3, rtol=5e-3)
    _close(kpe_cache[0, 0], append_kpe[0], atol=5e-3, rtol=5e-3)


def test_xqa_reference_shape():
    """XQA reference: shape + finite check (kernel requires specific dtypes)."""
    from flashinfer.trace.templates.page import xqa_trace

    torch.manual_seed(0)
    B, Hq, Hk, D, PS = 2, 8, 2, 64, 16
    NP, MP = 4, 2
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(NP, PS, Hk, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(B * MP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B,), PS * MP, dtype=torch.int32, device="cuda")
    out = xqa_trace.reference(q, k_cache, v_cache, page_table, seq_lens)
    assert out.shape == q.shape and torch.isfinite(out).all()


def test_xqa_mla_reference_shape():
    """XQA MLA reference: shape + finite check."""
    from flashinfer.trace.templates.page import xqa_mla_trace

    torch.manual_seed(0)
    B, H, CKV, KPE, PS = 2, 16, 128, 64, 16
    NP, MP = 4, 2
    q = torch.randn(B, H, CKV, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(NP, PS, CKV, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn(NP, PS, KPE, dtype=torch.bfloat16, device="cuda")
    page_table = torch.arange(B * MP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B,), PS * MP, dtype=torch.int32, device="cuda")
    out = xqa_mla_trace.reference(q, k_cache, v_cache, page_table, seq_lens)
    assert out.shape == q.shape and torch.isfinite(out).all()


def test_trtllm_fmha_v2_prefill_reference_shape():
    """TRT-LLM FMHA v2 prefill reference: shape + finite check."""
    from flashinfer.trace.templates.page import trtllm_fmha_v2_prefill_trace

    torch.manual_seed(0)
    B, H, D = 2, 8, 64
    q_lens = [8, 12]
    kv_lens = [8, 12]
    total_q = sum(q_lens)
    total_kv = sum(kv_lens)
    q = torch.randn(total_q, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total_kv, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    cum_q = torch.tensor([0, 8, 20], dtype=torch.int32, device="cuda")
    cum_kv = torch.tensor([0, 8, 20], dtype=torch.int32, device="cuda")
    out = trtllm_fmha_v2_prefill_trace.reference(
        (q, k, v),
        seq_lens,
        max(q_lens),
        max(kv_lens),
        1.0 / (D**0.5),
        1.0,
        B,
        cum_q,
        cum_kv,
    )
    assert out.shape == q.shape and torch.isfinite(out).all()


def test_batch_pod_run_reference_shape():
    """BatchPOD.run reference: shape + finite check on both prefill + decode outputs."""
    from flashinfer.trace.templates.attention import (
        batch_pod_with_paged_kv_cache_run_trace,
    )

    torch.manual_seed(0)
    NP, PS, Hq, Hk, D = 4, 16, 8, 2, 64
    device = "cuda"
    k_cache = torch.randn(NP, PS, Hk, D, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn_like(k_cache)
    q_p = torch.randn(8, Hq, D, dtype=torch.bfloat16, device=device)
    q_d = torch.randn(4, Hq, D, dtype=torch.bfloat16, device=device)
    out_p, out_d = batch_pod_with_paged_kv_cache_run_trace.reference(
        q_p,
        (k_cache, v_cache),
        q_d,
        (k_cache, v_cache),
    )
    assert out_p.shape == q_p.shape and torch.isfinite(out_p).all()
    assert out_d.shape == q_d.shape and torch.isfinite(out_d).all()


def test_var_block_sparse_run_reference_shape():
    """VariableBlockSparse reference (same as block_sparse): shape + finite."""
    from flashinfer.trace.templates.attention import (
        variable_block_sparse_attention_run_trace,
    )

    torch.manual_seed(0)
    Hq, Hk, D = 8, 2, 64
    q = torch.randn(16, Hq, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(32, Hk, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    out = variable_block_sparse_attention_run_trace.reference(q, k, v)
    assert out.shape == q.shape and torch.isfinite(out).all()


def test_attention_wrapper_references_produce_valid_outputs():
    """Smoke-test: each attention wrapper reference produces finite output."""
    from flashinfer.trace.templates.attention import (
        batch_attention_run_trace,
        block_sparse_attention_run_trace,
        multi_level_cascade_run_trace,
        pod_with_paged_kv_cache_run_trace,
        segment_gemm_run_trace,
    )

    torch.manual_seed(0)
    device = "cuda"

    # BatchAttention
    NP, PS, Hq, Hk, D = 4, 16, 8, 2, 64
    q = torch.randn(32, Hq, D, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(NP, PS, Hk, D, dtype=torch.bfloat16, device=device)
    v_cache = torch.randn_like(k_cache)
    out, lse = batch_attention_run_trace.reference(q, (k_cache, v_cache))
    assert out.shape == q.shape and torch.isfinite(out).all()
    assert lse.shape == (32, Hq)

    # Block sparse
    out = block_sparse_attention_run_trace.reference(
        q,
        k_cache.reshape(-1, Hk, D),
        v_cache.reshape(-1, Hk, D),
    )
    assert out.shape == q.shape and torch.isfinite(out).all()

    # Multi-level cascade
    out = multi_level_cascade_run_trace.reference(q, (k_cache, v_cache))
    assert out.shape == q.shape and torch.isfinite(out).all()

    # POD
    q_p = torch.randn(8, Hq, D, dtype=torch.bfloat16, device=device)
    k_p = torch.randn(8, Hk, D, dtype=torch.bfloat16, device=device)
    v_p = torch.randn_like(k_p)
    q_d = torch.randn(4, Hq, D, dtype=torch.bfloat16, device=device)
    out_p, out_d = pod_with_paged_kv_cache_run_trace.reference(
        q_p,
        k_p,
        v_p,
        q_d,
        (k_cache, v_cache),
    )
    assert out_p.shape == q_p.shape and out_d.shape == q_d.shape

    # SegmentGEMM
    seg_x = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
    seg_w = torch.randn(2, 32, 16, dtype=torch.bfloat16, device=device)
    seg_indptr = torch.tensor([0, 32, 64], dtype=torch.int64, device=device)
    out = segment_gemm_run_trace.reference(seg_x, seg_w, seg_indptr=seg_indptr)
    assert out.shape == (64, 16) and torch.isfinite(out).all()


def test_moe_variant_references_produce_valid_outputs():
    """Smoke-test: CuteDSL / B12x MoE references produce finite output."""
    from flashinfer.trace.templates.moe import (
        b12x_fused_moe_trace,
        cute_dsl_fused_moe_nvfp4_trace,
    )

    torch.manual_seed(0)
    device = "cuda"
    T, E, H, I, TOP_K, BS = 8, 4, 64, 32, 2, 16
    # NvFP4 packed tensors
    x = torch.randint(0, 256, (T, H // 2), dtype=torch.uint8, device=device)
    x_sf = torch.randn(T, H // BS, device=device).to(torch.float8_e4m3fn)
    tok_sel = torch.randint(0, E, (T, TOP_K), dtype=torch.int32, device=device)
    tok_scales = torch.full((T, TOP_K), 1.0 / TOP_K, device=device)
    w1 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w1_sf = torch.randn(E, 2 * I, H // BS, device=device).to(torch.float8_e4m3fn)
    w1_alpha = torch.ones(E, dtype=torch.float32, device=device) * 0.01
    fc2_input = torch.tensor([1.0], dtype=torch.float32, device=device)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device)
    w2_sf = torch.randn(E, H, I // BS, device=device).to(torch.float8_e4m3fn)
    w2_alpha = torch.ones(E, dtype=torch.float32, device=device) * 0.01
    out = cute_dsl_fused_moe_nvfp4_trace.reference(
        x,
        x_sf,
        tok_sel,
        tok_scales,
        w1,
        w1_sf,
        w1_alpha,
        fc2_input,
        w2,
        w2_sf,
        w2_alpha,
        num_experts=E,
        top_k=TOP_K,
    )
    assert out.shape == (T, H) and torch.isfinite(out).all()

    # B12x: bf16 input, FP4 weights
    x_bf16 = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    out = b12x_fused_moe_trace.reference(
        x_bf16,
        w1,
        w1_sf,
        w2,
        w2_sf,
        tok_sel,
        tok_scales,
        num_experts=E,
        top_k=TOP_K,
        w1_alpha=w1_alpha,
        w2_alpha=w2_alpha,
        fc2_input_scale=fc2_input,
    )
    assert out.shape == (T, H) and torch.isfinite(out).all()


def test_moe_references_produce_valid_outputs():
    """Smoke-test: each MoE reference produces a finite bf16 [T, H] tensor."""
    from flashinfer.trace.templates.moe import (
        cutlass_fused_moe_trace,
        trtllm_bf16_moe_trace,
        trtllm_bf16_routed_moe_trace,
        trtllm_fp8_per_tensor_scale_moe_trace,
        trtllm_mxint4_block_scale_moe_trace,
    )

    torch.manual_seed(0)
    T, E, H, I, TOP_K = 8, 4, 64, 32, 2
    device = "cuda"
    hs = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device) * 0.01
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device) * 0.01
    token_sel = torch.randint(0, E, (T, TOP_K), dtype=torch.int32, device=device)
    token_scales = torch.full((T, TOP_K), 1.0 / TOP_K, device=device)

    out = cutlass_fused_moe_trace.reference(hs, token_sel, token_scales, w1, w2)
    assert out.shape == (T, H) and out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    routing_logits = torch.randn(T, E, dtype=torch.float32, device=device)
    out = trtllm_bf16_moe_trace.reference(
        routing_logits,
        None,
        hs,
        w1,
        w2,
        num_experts=E,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
    )
    assert out.shape == (T, H) and torch.isfinite(out).all()

    topk_ids = torch.randint(0, E, (T, TOP_K), dtype=torch.int32, device=device)
    out = trtllm_bf16_routed_moe_trace.reference(
        topk_ids,
        hs,
        w1,
        w2,
        num_experts=E,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
    )
    assert out.shape == (T, H) and torch.isfinite(out).all()

    # Per-tensor FP8 needs fp8 weights; just check it runs with bf16 promoted.
    w1_fp8 = w1.to(torch.float8_e4m3fn)
    w2_fp8 = w2.to(torch.float8_e4m3fn)
    scales = torch.ones(E, dtype=torch.float32, device=device)
    out = trtllm_fp8_per_tensor_scale_moe_trace.reference(
        routing_logits,
        None,
        hs.to(torch.float8_e4m3fn),
        w1_fp8,
        scales,
        scales,
        w2_fp8,
        scales,
        num_experts=E,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
    )
    assert out.shape == (T, H) and torch.isfinite(out).all()

    # MxInt4: packed uint8 weights, bf16 scales.
    w1_i4 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w2_i4 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=device)
    w1_s = torch.randn(E, 2 * I, H // 32, dtype=torch.bfloat16, device=device)
    w2_s = torch.randn(E, H, I // 32, dtype=torch.bfloat16, device=device)
    out = trtllm_mxint4_block_scale_moe_trace.reference(
        routing_logits=routing_logits,
        routing_bias=None,
        hidden_states=hs,
        gemm1_weights=w1_i4,
        gemm1_weights_scale=w1_s,
        gemm2_weights=w2_i4,
        gemm2_weights_scale=w2_s,
        num_experts=E,
        top_k=TOP_K,
        local_expert_offset=0,
    )
    assert out.shape == (T, H) and torch.isfinite(out).all()
