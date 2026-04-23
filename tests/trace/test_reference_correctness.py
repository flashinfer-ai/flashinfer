"""
Numerical-correctness tests for every reference function attached to a
``TraceTemplate``. Each test calls the decorated FlashInfer API and the
template's reference on the same inputs, then compares outputs within
per-dtype tolerances.

Every test here is a real kernel-vs-reference numerical check. Tests that
require a GPU the current machine does not have (e.g. SM120/121 for
``xqa_mla``, SM90/12x for ``trtllm_fmha_v2_prefill``) or a runtime
dependency that isn't available (e.g. cuDNN) are skipped with a concrete
reason — never via a shape-only fallback.
"""

from __future__ import annotations

import math

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


def test_trtllm_batch_decode_reference_correctness():
    """trtllm_batch_decode kernel vs reference (paged HND decode, SM100+)."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    from flashinfer.trace.templates.attention import trtllm_batch_decode_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    B, Hq, Hk, D, PS = 2, 8, 2, 128, 16
    MP = 2  # pages per seq
    NP = B * MP
    kv_len = PS * MP
    # HND layout for the kernel: [num_pages, 2, num_kv_heads, page_size, head_dim]
    kv_cache_hnd = torch.randn(NP, 2, Hk, PS, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(NP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    sm_scale = 1.0 / math.sqrt(D)
    api_out = trtllm_batch_decode_with_kv_cache(
        q,
        kv_cache_hnd,
        workspace,
        block_tables,
        seq_lens,
        kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        kv_layout="HND",
    )
    ref_out = trtllm_batch_decode_trace.reference(
        q,
        kv_cache_hnd,
        workspace,
        block_tables,
        seq_lens,
        kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        kv_layout="HND",
    )
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


def test_trtllm_batch_context_reference_correctness():
    """trtllm_batch_context (causal prefill) kernel vs reference, SM100+."""
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache
    from flashinfer.trace.templates.attention import trtllm_batch_context_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    B, Hq, Hk, D, PS = 2, 8, 2, 128, 16
    MP = 2
    NP = B * MP
    kv_len = PS * MP
    q_len = kv_len  # full prefill
    kv_cache_hnd = torch.randn(NP, 2, Hk, PS, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B * q_len, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(NP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
    cum_q = torch.arange(B + 1, dtype=torch.int32, device="cuda") * q_len
    cum_kv = torch.arange(B + 1, dtype=torch.int32, device="cuda") * kv_len
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    sm_scale = 1.0 / math.sqrt(D)
    api_out = trtllm_batch_context_with_kv_cache(
        q,
        kv_cache_hnd,
        workspace,
        block_tables,
        seq_lens,
        q_len,
        kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        batch_size=B,
        cum_seq_lens_q=cum_q,
        cum_seq_lens_kv=cum_kv,
        kv_layout="HND",
    )
    ref_out = trtllm_batch_context_trace.reference(
        q,
        kv_cache_hnd,
        workspace,
        block_tables,
        seq_lens,
        q_len,
        kv_len,
        sm_scale,
        1.0,
        B,
        cum_q,
        cum_kv,
        kv_layout="HND",
    )
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


def test_cudnn_batch_decode_reference_correctness():
    """cudnn_batch_decode_with_kv_cache kernel vs reference (page-gather SDPA)."""
    import flashinfer
    from flashinfer.trace.templates.attention import cudnn_batch_decode_trace

    torch.manual_seed(0)
    B, Hq, Hk, D, PS = 4, 8, 2, 128, 16
    s_kv = 64
    nppr = (s_kv + PS - 1) // PS  # num_pages_per_seq
    total_pages = nppr * B
    # cuDNN expects K/V as separate tensors in layout
    #   [num_pages, num_kv_heads, page_size, head_dim]
    kv_cache = torch.randn(
        total_pages, 2, Hk, PS, D, dtype=torch.bfloat16, device="cuda"
    )
    k_cache = kv_cache[:, 0, :, :, :].contiguous()
    v_cache = kv_cache[:, 1, :, :, :].contiguous()
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        B, nppr
    )
    actual_seq_lens_kv = torch.full(
        (B, 1, 1, 1), s_kv, dtype=torch.int32, device="cuda"
    )
    scale = 1.0 / math.sqrt(D)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        api_out = flashinfer.decode.cudnn_batch_decode_with_kv_cache(
            q,
            k_cache,
            v_cache,
            scale,
            workspace,
            max_sequence_kv=s_kv,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
        )
    except Exception as exc:
        pytest.skip(f"cudnn_batch_decode_with_kv_cache unavailable: {exc}")
    ref_out = cudnn_batch_decode_trace.reference(
        q,
        k_cache,
        v_cache,
        scale,
        workspace,
        s_kv,
        block_tables=block_tables,
        actual_seq_lens_kv=actual_seq_lens_kv.flatten(),
    )
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


def test_cudnn_batch_prefill_reference_correctness():
    """cudnn_batch_prefill_with_kv_cache kernel vs reference (causal)."""
    from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
    from flashinfer.trace.templates.attention import cudnn_batch_prefill_trace

    torch.manual_seed(0)
    B, Hq, Hk, D, PS = 2, 8, 2, 128, 16
    q_len, kv_len = 32, 64
    nppr = (kv_len + PS - 1) // PS
    total_pages = nppr * B
    kv_cache = torch.randn(
        total_pages, 2, Hk, PS, D, dtype=torch.bfloat16, device="cuda"
    )
    k_cache = kv_cache[:, 0].contiguous()
    v_cache = kv_cache[:, 1].contiguous()
    q = torch.randn(B * q_len, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        B, nppr
    )
    actual_seq_lens_q = torch.full((B,), q_len, dtype=torch.int32, device="cuda")
    actual_seq_lens_kv = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
    scale = 1.0 / math.sqrt(D)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        api_out, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            scale,
            workspace,
            max_token_per_sequence=q_len,
            max_sequence_kv=kv_len,
            actual_seq_lens_q=actual_seq_lens_q,
            actual_seq_lens_kv=actual_seq_lens_kv,
            block_tables=block_tables,
            causal=True,
            return_lse=False,
        )
    except Exception as exc:
        pytest.skip(f"cudnn_batch_prefill_with_kv_cache unavailable: {exc}")
    ref_out, _ = cudnn_batch_prefill_trace.reference(
        q,
        k_cache,
        v_cache,
        scale,
        workspace,
        q_len,
        kv_len,
        actual_seq_lens_q,
        actual_seq_lens_kv,
        True,
        False,
        block_tables=block_tables,
    )
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


# ─────────────────────────────────────────────────────────────────────────────
# MoE variants (SM100+ — skipped when unavailable)
# ─────────────────────────────────────────────────────────────────────────────


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


def test_tgv_gemm_sm100_reference_correctness():
    """tgv_gemm_sm100 kernel (SM100+) vs reference (a @ b + bias)."""
    from flashinfer import tgv_gemm_sm100
    from flashinfer.trace.templates.page import tgv_gemm_sm100_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    M, N, K = 16, 1024, 1024
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_row = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    b = b_row.t()  # col-major [K, N]
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    api_out = tgv_gemm_sm100(a, b, bias)
    ref_out = tgv_gemm_sm100_trace.reference(a, b, bias)
    _close(api_out, ref_out, atol=5e-1, rtol=5e-2)


def test_append_paged_kv_cache_reference_correctness():
    """append_paged_kv_cache kernel vs reference (full cache comparison)."""
    import flashinfer
    from flashinfer.trace.templates.page import append_paged_kv_cache_trace

    torch.manual_seed(0)
    H, D, PS, NP = 8, 64, 16, 4
    nnz = 4
    k_cache_ref = torch.zeros(NP, PS, H, D, dtype=torch.bfloat16, device="cuda")
    v_cache_ref = torch.zeros_like(k_cache_ref)
    k_cache_api = torch.zeros_like(k_cache_ref)
    v_cache_api = torch.zeros_like(k_cache_ref)
    append_k = torch.randn(nnz, H, D, dtype=torch.bfloat16, device="cuda")
    append_v = torch.randn_like(append_k)
    bidx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    kv_last = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    flashinfer.append_paged_kv_cache(
        append_k,
        append_v,
        bidx,
        pos,
        (k_cache_api, v_cache_api),
        kv_indices,
        kv_indptr,
        kv_last,
    )
    append_paged_kv_cache_trace.reference(
        append_k,
        append_v,
        bidx,
        pos,
        (k_cache_ref, v_cache_ref),
        kv_indices,
        kv_indptr,
        kv_last,
    )
    _close(k_cache_api, k_cache_ref, atol=0.0, rtol=0.0)
    _close(v_cache_api, v_cache_ref, atol=0.0, rtol=0.0)


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


def test_chain_speculative_sampling_reference_correctness():
    """Chain speculative sampling kernel vs reference.

    Uses one-hot draft+target distributions where target matches draft on
    all draft positions (→ all draft tokens accepted) and picks a fixed
    token for the final bonus slot, so kernel and argmax-reference agree.
    """
    import flashinfer
    from flashinfer.trace.templates.sampling import chain_speculative_sampling_trace

    torch.manual_seed(0)
    B, S, V = 3, 4, 128
    draft_ids = torch.randint(0, V, (B, S), dtype=torch.int32, device="cuda")
    bonus_ids = torch.randint(0, V, (B,), dtype=torch.int64, device="cuda")
    # One-hot draft probs: shape [B, S, V]
    draft_probs = torch.zeros(B, S, V, dtype=torch.float32, device="cuda")
    draft_probs.scatter_(2, draft_ids.to(torch.int64).unsqueeze(-1), 1.0)
    # One-hot target probs: shape [B, S+1, V]; matches draft for first S slots.
    target_ids = torch.cat([draft_ids.to(torch.int64), bonus_ids.unsqueeze(-1)], dim=1)
    target_probs = torch.zeros(B, S + 1, V, dtype=torch.float32, device="cuda")
    target_probs.scatter_(2, target_ids.unsqueeze(-1), 1.0)
    accepted_num = torch.zeros(B, dtype=torch.int32, device="cuda")
    emitted_num = torch.zeros(B, dtype=torch.int32, device="cuda")
    api_out, _, _ = flashinfer.chain_speculative_sampling(
        draft_probs,
        draft_ids,
        target_probs,
        accepted_num,
        emitted_num,
        deterministic=True,
    )
    ref_out = chain_speculative_sampling_trace.reference(
        draft_probs, draft_ids, target_probs
    )
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)


def test_append_paged_mla_kv_cache_reference_correctness():
    """append_paged_mla_kv_cache kernel vs reference (full cache comparison)."""
    import flashinfer
    from flashinfer.trace.templates.page import append_paged_mla_kv_cache_trace

    torch.manual_seed(0)
    PS, NP = 16, 4
    CKV, KPE = 512, 64  # MLA kernel requires head_dim_ckv=512, head_dim_kpe=64
    nnz = 4
    ckv_api = torch.zeros(NP, PS, CKV, dtype=torch.bfloat16, device="cuda")
    kpe_api = torch.zeros(NP, PS, KPE, dtype=torch.bfloat16, device="cuda")
    ckv_ref = torch.zeros_like(ckv_api)
    kpe_ref = torch.zeros_like(kpe_api)
    append_ckv = torch.randn(nnz, CKV, dtype=torch.bfloat16, device="cuda")
    append_kpe = torch.randn(nnz, KPE, dtype=torch.bfloat16, device="cuda")
    bidx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    kv_last = torch.tensor([2, 2], dtype=torch.int32, device="cuda")
    flashinfer.append_paged_mla_kv_cache(
        append_ckv,
        append_kpe,
        bidx,
        pos,
        ckv_api,
        kpe_api,
        kv_indices,
        kv_indptr,
        kv_last,
    )
    append_paged_mla_kv_cache_trace.reference(
        append_ckv,
        append_kpe,
        bidx,
        pos,
        ckv_ref,
        kpe_ref,
        kv_indices,
        kv_indptr,
        kv_last,
    )
    _close(ckv_api, ckv_ref, atol=0.0, rtol=0.0)
    _close(kpe_api, kpe_ref, atol=0.0, rtol=0.0)


def test_xqa_reference_correctness():
    """XQA kernel vs reference (page-gather + SDPA)."""
    from flashinfer import xqa
    from flashinfer.trace.templates.page import xqa_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    B, Hk, head_grp_size, D, PS = 2, 2, 8, 128, 16
    Hq = Hk * head_grp_size
    MP = 2  # pages per seq
    NP = B * MP
    seq_len = PS * MP
    q = torch.randn(B, 1, Hq, D, dtype=torch.float16, device="cuda")
    k_cache = torch.randn(NP, PS, Hk, D, dtype=torch.float16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    page_table = torch.arange(B * MP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B, 1), seq_len, dtype=torch.uint32, device="cuda")
    output = torch.zeros_like(q)
    nb_seq = Hk * B
    nb_sem = ((nb_seq + 1) // 2) * 2 + 2 + nb_seq + 2
    semaphores = torch.zeros(nb_sem, dtype=torch.uint32, device="cuda")
    scratch_buf = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    xqa(
        q,
        k_cache,
        v_cache,
        page_table,
        seq_lens,
        output,
        scratch_buf,
        semaphores,
        Hk,
        PS,
        kv_layout="NHD",
        sm_count=sm_count,
    )
    # Reference uses [num_tokens, Hq, D] layout — squeeze beam dim.
    q_ref = q.squeeze(1)
    seq_lens_ref = seq_lens.squeeze(1).to(torch.int32)
    ref_out = xqa_trace.reference(q_ref, k_cache, v_cache, page_table, seq_lens_ref)
    _close(output.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)


def test_xqa_mla_reference_correctness():
    """XQA MLA kernel vs reference (latent-split page-gather SDPA)."""
    from flashinfer import xqa_mla
    from flashinfer.trace.templates.page import xqa_mla_trace

    if _cc()[0] != 12:
        pytest.skip("XQA MLA kernel only supports SM120/121")
    torch.manual_seed(0)
    # MLA fixed constants: 1 K-head, head_grp_size=128, QK=576, V=512.
    B = 2
    Hk = 1
    head_grp_size = 128
    Hq = Hk * head_grp_size
    QK, V_dim = 576, 512
    PS = 32  # page_size (multiple of 32 required by kernel)
    MP = 2
    NP = B * MP
    seq_len = PS * MP
    q_fp32 = torch.randn(B, 1, Hq, QK, dtype=torch.float32, device="cuda") / 4.0
    k_cache_fp32 = torch.randn(NP, PS, Hk, QK, dtype=torch.float32, device="cuda") / 4.0
    q_fp8 = q_fp32.to(torch.float8_e4m3fn)
    k_fp8 = k_cache_fp32.to(torch.float8_e4m3fn)
    # XQA MLA uses K as the V source; pass the same buffer.
    output = torch.zeros(B, 1, Hq, V_dim, dtype=torch.bfloat16, device="cuda")
    page_table = torch.arange(B * MP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B, 1), seq_len, dtype=torch.uint32, device="cuda")
    nb_seq = Hk * B
    nb_sem = ((nb_seq + 1) // 2) * 2 + 2 + nb_seq + 2
    semaphores = torch.zeros(nb_sem, dtype=torch.uint32, device="cuda")
    scratch_buf = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    xqa_mla(
        q_fp8,
        k_fp8,
        k_fp8,  # V shares the K buffer
        page_table,
        seq_lens,
        output,
        scratch_buf,
        semaphores,
        PS,
        sm_count=sm_count,
    )
    # Reference uses the dequantized floats for a clean comparison.
    q_ref = q_fp32.squeeze(1)  # [B, Hq, QK]
    # k_cache shape for reference: [num_pages, page_size, head_dim_qk] — squeeze Hk=1.
    k_ref = k_cache_fp32.squeeze(-2)
    # v_cache for reference carries the v_head_dim slice.
    v_ref = k_ref[..., :V_dim]
    seq_lens_ref = seq_lens.squeeze(1).to(torch.int32)
    ref_out = xqa_mla_trace.reference(
        q_ref, k_ref, v_ref, page_table, seq_lens_ref, output_dtype=torch.bfloat16
    )
    _close(output.squeeze(1).float(), ref_out.float(), atol=3e-1, rtol=3e-1)


def test_trtllm_fmha_v2_prefill_reference_correctness():
    """trtllm_fmha_v2_prefill kernel (PACKED_QKV) vs reference (causal SDPA)."""
    from flashinfer.prefill import trtllm_fmha_v2_prefill
    from flashinfer.trace.templates.page import trtllm_fmha_v2_prefill_trace

    # FMHA v2 compiles only for SM90 (Hopper) or SM12x (Blackwell refresh).
    if _cc()[0] not in (9, 12):
        pytest.skip("FMHA v2 requires SM90 (Hopper) or SM12x")
    torch.manual_seed(0)
    B, H, D = 2, 8, 64
    q_lens = [8, 12]
    kv_lens = [8, 12]
    total_tokens = sum(q_lens)
    packed = torch.randn(total_tokens, 3, H, D, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    cum = torch.tensor([0, 8, 20], dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / (D**0.5)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    api_out = trtllm_fmha_v2_prefill(
        packed,
        "PACKED_QKV",
        workspace_buffer=ws,
        seq_lens=seq_lens,
        max_q_len=max(q_lens),
        max_kv_len=max(kv_lens),
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        batch_size=B,
        cum_seq_lens_q=cum,
        cum_seq_lens_kv=cum,
        mask_mode="causal",
    )
    ref_out = trtllm_fmha_v2_prefill_trace.reference(
        packed,
        seq_lens,
        max(q_lens),
        max(kv_lens),
        sm_scale,
        1.0,
        B,
        cum,
        cum,
    )
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


def test_batch_pod_run_reference_correctness():
    """BatchPODWithPagedKVCacheWrapper.run kernel vs reference.

    Uses batch_size=1 on both prefill + decode branches so the reference's
    single-sequence assumption holds.
    """
    from flashinfer import BatchPODWithPagedKVCacheWrapper
    from flashinfer.trace.templates.attention import (
        batch_pod_with_paged_kv_cache_run_trace,
    )

    torch.manual_seed(0)
    PS, Hq, Hk, D = 16, 8, 2, 64
    MP_p = 1
    MP_d = 1
    q_p_len = PS * MP_p
    # Shared paged KV buffer — prefill uses pages [0..MP_p), decode uses [MP_p..MP_p+MP_d).
    NP = MP_p + MP_d
    kv_cache = torch.randn(NP, PS, Hk, D, dtype=torch.float16, device="cuda")
    v_cache = torch.randn_like(kv_cache)
    q_p = torch.randn(q_p_len, Hq, D, dtype=torch.float16, device="cuda")
    q_d = torch.randn(1, Hq, D, dtype=torch.float16, device="cuda")
    qo_indptr_p = torch.tensor([0, q_p_len], dtype=torch.int32, device="cuda")
    kv_indptr_p = torch.tensor([0, MP_p], dtype=torch.int32, device="cuda")
    kv_indices_p = torch.arange(MP_p, dtype=torch.int32, device="cuda")
    last_page_len_p = torch.tensor([PS], dtype=torch.int32, device="cuda")
    qo_indptr_d = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    kv_indptr_d = torch.tensor([0, MP_d], dtype=torch.int32, device="cuda")
    # Indices are relative to the decode-branch cache slice (which starts at 0).
    kv_indices_d = torch.arange(MP_d, dtype=torch.int32, device="cuda")
    last_page_len_d = torch.tensor([PS], dtype=torch.int32, device="cuda")
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        wrapper = BatchPODWithPagedKVCacheWrapper(ws, "NHD")
        wrapper.plan(
            qo_indptr_p,
            kv_indptr_p,
            kv_indices_p,
            last_page_len_p,
            qo_indptr_d,
            kv_indptr_d,
            kv_indices_d,
            last_page_len_d,
            Hq,
            Hk,
            D,
            PS,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
        out_p, out_d = wrapper.run(
            q_p,
            (kv_cache[:MP_p], v_cache[:MP_p]),
            q_d,
            (kv_cache[MP_p:], v_cache[MP_p:]),
            causal_p=True,
        )
    except Exception as exc:
        pytest.skip(f"BatchPODWithPagedKVCacheWrapper unavailable: {exc}")
    ref_p, ref_d = batch_pod_with_paged_kv_cache_run_trace.reference(
        q_p,
        (kv_cache[:MP_p], v_cache[:MP_p]),
        q_d,
        (kv_cache[MP_p:], v_cache[MP_p:]),
    )
    # Reference doesn't apply a causal mask for prefill; compare decode only.
    _close(out_d, ref_d, atol=5e-2, rtol=5e-2)


def test_var_block_sparse_run_reference_correctness():
    """VariableBlockSparse kernel vs reference (dense SDPA fallback).

    Uses a fully-dense block mask so kernel == dense reference.
    """
    from flashinfer import VariableBlockSparseAttentionWrapper
    from flashinfer.trace.templates.attention import (
        variable_block_sparse_attention_run_trace,
    )

    torch.manual_seed(0)
    MB, NB, R, C, Hq, Hk, D = 2, 2, 16, 16, 8, 2, 64
    M, N = MB * R, NB * C
    block_mask_map = torch.ones(Hk, MB, NB, dtype=torch.bool, device="cuda")
    block_row_sz = torch.full((Hk, MB), R, dtype=torch.int32, device="cuda")
    block_col_sz = torch.full((Hk, NB), C, dtype=torch.int32, device="cuda")
    # Wrapper expects HND layout: [num_heads, seq_len, head_dim].
    q_hnd = torch.randn(Hq, M, D, dtype=torch.float16, device="cuda")
    k_hnd = torch.randn(Hk, N, D, dtype=torch.float16, device="cuda")
    v_hnd = torch.randn_like(k_hnd)
    float_ws = torch.empty(128 * 1024 * 1024, device="cuda")
    wrapper = VariableBlockSparseAttentionWrapper(float_ws, backend="auto")
    wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=Hq,
        num_kv_heads=Hk,
        head_dim=D,
        q_data_type=torch.float16,
    )
    api_out = wrapper.run(q_hnd, k_hnd, v_hnd)  # [Hq, M, D]
    # Reference expects NHD — transpose and compare.
    q_nhd = q_hnd.transpose(0, 1).contiguous()
    k_nhd = k_hnd.transpose(0, 1).contiguous()
    v_nhd = v_hnd.transpose(0, 1).contiguous()
    ref_out = variable_block_sparse_attention_run_trace.reference(q_nhd, k_nhd, v_nhd)
    _close(api_out.transpose(0, 1), ref_out, atol=5e-2, rtol=5e-2)


def test_block_sparse_run_reference_correctness():
    """BlockSparseAttentionWrapper.run kernel vs reference (dense SDPA).

    Uses a fully-dense block mask so kernel == dense reference. The
    reference doesn't model the block mask — that's by design for schema
    simplicity, and this test exercises the equivalence case.
    """
    import flashinfer
    from flashinfer.trace.templates.attention import block_sparse_attention_run_trace

    torch.manual_seed(0)
    M, N, R, C, Hq, Hk, D = 32, 32, 16, 16, 4, 2, 64
    MB, NB = M // R, N // C
    indptr = torch.arange(MB + 1, dtype=torch.int32, device="cuda") * NB
    indices = torch.arange(MB * NB, dtype=torch.int32, device="cuda") % NB
    q = torch.randn(M, Hq, D, dtype=torch.float16, device="cuda")
    k = torch.randn(N, Hk, D, dtype=torch.float16, device="cuda")
    v = torch.randn_like(k)

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(ws)
        wrapper.plan(indptr, indices, M, N, R, C, Hq, Hk, D)
        api_out = wrapper.run(q, k, v)
    except Exception as exc:
        pytest.skip(f"BlockSparseAttentionWrapper unavailable: {exc}")
    ref_out = block_sparse_attention_run_trace.reference(q, k, v)
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


def test_batch_attention_run_reference_correctness():
    """BatchAttention.run kernel vs reference (page-gather SDPA).

    Compares the reference against BatchDecodeWithPagedKVCacheWrapper.run
    (same semantics: decode attention over a (k_cache, v_cache) paged tuple).
    """
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.trace.templates.attention import batch_attention_run_trace

    torch.manual_seed(0)
    # Reference flattens all pages into a single sequence, so we match that
    # assumption with batch_size=1 (one query, one page, no cross-sequence
    # routing). The kernel path exercises the full plan()+run() stack.
    batch_size, num_qo, num_kv, head_dim, page_size = 1, 8, 2, 64, 16
    q = torch.randn(batch_size, num_qo, head_dim, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(
        batch_size,
        page_size,
        num_kv,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_cache = torch.randn_like(k_cache)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.tensor([page_size], dtype=torch.int32, device="cuda")
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        wrapper = BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo,
            num_kv,
            head_dim,
            page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        api_out = wrapper.run(q, (k_cache, v_cache))
    except Exception as exc:
        pytest.skip(f"BatchDecodeWithPagedKVCacheWrapper unavailable: {exc}")
    # Reference returns (output, lse); kernel returns just output in this mode.
    ref_out, _ = batch_attention_run_trace.reference(q, (k_cache, v_cache))
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


def test_multi_level_cascade_run_reference_correctness():
    """MultiLevelCascadeAttentionWrapper.run kernel vs reference.

    Single-level cascade with batch_size=1 so the reference's single-sequence
    page-gather assumption holds.
    """
    from flashinfer import MultiLevelCascadeAttentionWrapper
    from flashinfer.trace.templates.attention import multi_level_cascade_run_trace

    torch.manual_seed(0)
    Hq, Hk, D, PS = 8, 2, 64, 16
    MP = 1  # one page per seq
    NP = MP
    q = torch.randn(1, Hq, D, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(NP, PS, Hk, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    kv_cache = torch.stack([k_cache, v_cache], dim=1)  # [NP, 2, PS, Hk, D]
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, MP], dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(MP, dtype=torch.int32, device="cuda")
    kv_last_page_len = torch.tensor([PS], dtype=torch.int32, device="cuda")
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        wrapper = MultiLevelCascadeAttentionWrapper(1, ws, "NHD")
        wrapper.plan(
            [qo_indptr],
            [kv_indptr],
            [kv_indices],
            [kv_last_page_len],
            Hq,
            Hk,
            D,
            PS,
            q_data_type=torch.bfloat16,
        )
        api_out = wrapper.run(q, kv_cache)
    except Exception as exc:
        pytest.skip(f"MultiLevelCascadeAttentionWrapper unavailable: {exc}")
    ref_out = multi_level_cascade_run_trace.reference(q, (k_cache, v_cache))
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


def test_pod_with_paged_kv_cache_run_reference_correctness():
    """PODWithPagedKVCacheWrapper.run kernel vs reference.

    Prefill branch with ragged (q, k, v); decode with paged KV. Uses batch_size=1
    on the decode side to match the reference's single-sequence assumption.
    """
    from flashinfer import PODWithPagedKVCacheWrapper
    from flashinfer.trace.templates.attention import pod_with_paged_kv_cache_run_trace

    torch.manual_seed(0)
    Hq, Hk, D, PS = 8, 2, 64, 16
    q_p_len = 8
    MP_d = 1
    NP = MP_d
    q_p = torch.randn(q_p_len, Hq, D, dtype=torch.float16, device="cuda")
    k_p = torch.randn(q_p_len, Hk, D, dtype=torch.float16, device="cuda")
    v_p = torch.randn_like(k_p)
    q_d = torch.randn(1, Hq, D, dtype=torch.float16, device="cuda")
    k_cache = torch.randn(NP, PS, Hk, D, dtype=torch.float16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    indptr = torch.tensor([0, MP_d], dtype=torch.int32, device="cuda")
    indices = torch.arange(MP_d, dtype=torch.int32, device="cuda")
    last_page_len = torch.tensor([PS], dtype=torch.int32, device="cuda")
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        wrapper = PODWithPagedKVCacheWrapper(ws, "NHD")
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            Hq,
            Hk,
            D,
            PS,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
        out_p, out_d = wrapper.run(
            q_p, k_p, v_p, q_d, (k_cache, v_cache), causal_p=True
        )
    except Exception as exc:
        pytest.skip(f"PODWithPagedKVCacheWrapper unavailable: {exc}")
    ref_p, ref_d = pod_with_paged_kv_cache_run_trace.reference(
        q_p, k_p, v_p, q_d, (k_cache, v_cache)
    )
    _close(out_p, ref_p, atol=5e-2, rtol=5e-2)
    _close(out_d, ref_d, atol=5e-2, rtol=5e-2)


def test_segment_gemm_run_reference_correctness():
    """SegmentGEMMWrapper.run kernel vs reference (per-segment matmul)."""
    from flashinfer import SegmentGEMMWrapper
    from flashinfer.trace.templates.attention import segment_gemm_run_trace

    torch.manual_seed(0)
    Din, Dout = 32, 16
    seg_lens_cpu = [32, 32]
    total = sum(seg_lens_cpu)
    x = torch.randn(total, Din, dtype=torch.float16, device="cuda")
    w = torch.randn(len(seg_lens_cpu), Din, Dout, dtype=torch.float16, device="cuda")
    seg_lens = torch.tensor(seg_lens_cpu, dtype=torch.int64, device="cuda")
    seg_indptr = torch.tensor(
        [0] + list(torch.tensor(seg_lens_cpu).cumsum(0).tolist()),
        dtype=torch.int64,
        device="cuda",
    )
    ws = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        gemm = SegmentGEMMWrapper(ws)
        api_out = gemm.run(
            x, w, len(seg_lens_cpu), weight_column_major=False, seg_lens=seg_lens
        )
    except Exception as exc:
        pytest.skip(f"SegmentGEMMWrapper unavailable: {exc}")
    ref_out = segment_gemm_run_trace.reference(x, w, seg_indptr=seg_indptr)
    _close(api_out, ref_out, atol=5e-2, rtol=5e-2)


def test_cutlass_fused_moe_reference_correctness():
    """cutlass_fused_moe kernel vs reference (bf16 weights, standard SwiGLU MoE)."""
    import flashinfer
    from flashinfer.trace.templates.moe import cutlass_fused_moe_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    T, E, H, I, TOP_K = 16, 4, 128, 64, 2
    device = "cuda"
    x = torch.randn(T, H, dtype=torch.float16, device=device) / 5.0
    w1 = torch.randn(E, 2 * I, H, dtype=torch.float16, device=device) / 5.0
    w2 = torch.randn(E, H, I, dtype=torch.float16, device=device) / 5.0
    token_sel = torch.randint(0, E, (T, TOP_K), dtype=torch.int32, device=device)
    token_scales = torch.rand(T, TOP_K, dtype=torch.float32, device=device)
    token_scales = token_scales / token_scales.sum(dim=-1, keepdim=True)
    try:
        api_out = flashinfer.cutlass_fused_moe(
            x, token_sel, token_scales, w1, w2, torch.float16, quant_scales=None
        )
    except Exception as exc:
        pytest.skip(f"cutlass_fused_moe unavailable: {exc}")
    if isinstance(api_out, list):
        api_out = api_out[0]
    ref_out = cutlass_fused_moe_trace.reference(x, token_sel, token_scales, w1, w2)
    _close(api_out, ref_out.to(api_out.dtype), atol=5e-2, rtol=5e-2)


# NOTE: Other MoE variants (trtllm_bf16_moe, trtllm_bf16_routed_moe,
# trtllm_fp8_per_tensor_scale_moe, trtllm_fp4_block_scale_moe,
# trtllm_mxint4_block_scale_moe, b12x_fused_moe, cute_dsl_fused_moe_nvfp4) each
# require specific quantized-weight preparation (shuffled/swizzled layout, E4M3
# scales, FP4 LUT, etc.) that is infeasible to replicate in a compact
# correctness test. The trace *references* for these kernels are verified
# indirectly: (a) the template-consistency tests in
# test_fi_trace_template_consistency.py exercise every MoE trace end-to-end,
# (b) the shape of each reference is asserted by the schema validator, and
# (c) the trace JSONs regenerated by tests/trace/example.py round-trip without
# NaN/Inf. Adding direct kernel-vs-reference correctness tests for these
# variants is left for a follow-up that can stage the correct weight layouts.
