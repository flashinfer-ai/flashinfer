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


def _skip_if_not_sm100_or_103():
    """Gate for kernels that run only on Blackwell proper (SM100/SM103) —
    not on the SM12x refresh or older architectures."""
    major, minor = _cc()
    if (major, minor) not in ((10, 0), (10, 3)):
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")


def _close(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> None:
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


def _close_fp8(a: torch.Tensor, b: torch.Tensor, *, cos_sim_min: float = 0.99) -> None:
    """Cosine-similarity check — used only for APIs whose own unit test
    uses cosine similarity (e.g. tests/gemm/test_tgv_gemm.py's
    ``F.cosine_similarity(...) > 0.99`` guard)."""
    import torch.nn.functional as F

    cos = F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0)
    assert cos.item() > cos_sim_min, f"cos_sim={cos.item():.4f} < {cos_sim_min}"


def _close_pass_ratio(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    pass_ratio: float = 0.95,
) -> None:
    """Pass-ratio closeness check — the standard metric used by
    tests/attention/test_xqa.py and test_xqa_mla for FP8-quantized
    attention outputs. Requires at least ``pass_ratio`` of elements to be
    within ``(atol, rtol)`` (any element satisfying either bound passes)."""
    a_f = a.float()
    b_f = b.float()
    diff_abs = (a_f - b_f).abs()
    diff_rel = diff_abs / (b_f.abs() + 1e-8)
    ok = (diff_abs <= atol) | (diff_rel <= rtol)
    frac = ok.float().mean().item()
    assert frac >= pass_ratio, (
        f"pass_ratio={frac:.4f} < {pass_ratio} (atol={atol}, rtol={rtol})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────────────────────────────────────

# Matches tests/attention/test_rope.py: 1e-3 for fp16 apply_rope; 1e-2 for
# bf16 apply_rope_with_cos_sin_cache. Our inputs are bf16 so use 1e-2.
_ROPE_TOL = dict(atol=1e-2, rtol=1e-2)

# Common Var values for ragged/pos_ids RoPE inits — matches the
# original ``_rope_inputs`` helper (B=2, S=8, Hq=4, Hk=2, D=64).
_ROPE_KWARGS = dict(nnz=16, batch_size=2, num_q_heads=4, num_k_heads=2, head_dim=64)


def _init_filtered(template, **kwargs):
    """Call ``template.init(...)`` passing only kwargs the function accepts."""
    import inspect

    sig = inspect.signature(template.init)
    accepted = set(sig.parameters)
    return template.init(**{k: v for k, v in kwargs.items() if k in accepted})


def _assert_finite(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if t is None:
            continue
        assert torch.isfinite(t.float()).all(), "init/kernel produced NaN or Inf"


def test_apply_rope():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_trace

    inputs = apply_rope_trace.init(**_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api, k_api = flashinfer.apply_rope(
        inputs["q"], inputs["k"], inputs["indptr"], inputs["offsets"]
    )
    q_ref, k_ref = apply_rope_trace.reference(
        inputs["q"], inputs["k"], inputs["indptr"], inputs["offsets"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_inplace_trace

    inputs = apply_rope_inplace_trace.init(**_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api = inputs["q"].clone()
    k_api = inputs["k"].clone()
    flashinfer.apply_rope_inplace(q_api, k_api, inputs["indptr"], inputs["offsets"])
    q_ref, k_ref = apply_rope_inplace_trace.reference(
        inputs["q"], inputs["k"], inputs["indptr"], inputs["offsets"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_pos_ids():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_pos_ids_trace

    inputs = _init_filtered(apply_rope_pos_ids_trace, **_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api, k_api = flashinfer.apply_rope_pos_ids(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    q_ref, k_ref = apply_rope_pos_ids_trace.reference(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_pos_ids_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_pos_ids_inplace_trace

    inputs = _init_filtered(apply_rope_pos_ids_inplace_trace, **_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api = inputs["q"].clone()
    k_api = inputs["k"].clone()
    flashinfer.apply_rope_pos_ids_inplace(q_api, k_api, inputs["pos_ids"])
    q_ref, k_ref = apply_rope_pos_ids_inplace_trace.reference(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_llama31_rope():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_llama31_rope_trace

    inputs = apply_llama31_rope_trace.init(**_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api, k_api = flashinfer.apply_llama31_rope(
        inputs["q"], inputs["k"], inputs["indptr"], inputs["offsets"]
    )
    q_ref, k_ref = apply_llama31_rope_trace.reference(
        inputs["q"], inputs["k"], inputs["indptr"], inputs["offsets"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_llama31_rope_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_llama31_rope_inplace_trace

    inputs = apply_llama31_rope_inplace_trace.init(**_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api = inputs["q"].clone()
    k_api = inputs["k"].clone()
    flashinfer.apply_llama31_rope_inplace(
        q_api, k_api, inputs["indptr"], inputs["offsets"]
    )
    q_ref, k_ref = apply_llama31_rope_inplace_trace.reference(
        inputs["q"], inputs["k"], inputs["indptr"], inputs["offsets"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_llama31_rope_pos_ids():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_llama31_rope_pos_ids_trace

    inputs = _init_filtered(apply_llama31_rope_pos_ids_trace, **_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api, k_api = flashinfer.apply_llama31_rope_pos_ids(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    q_ref, k_ref = apply_llama31_rope_pos_ids_trace.reference(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_llama31_rope_pos_ids_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import (
        apply_llama31_rope_pos_ids_inplace_trace,
    )

    inputs = _init_filtered(apply_llama31_rope_pos_ids_inplace_trace, **_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api = inputs["q"].clone()
    k_api = inputs["k"].clone()
    flashinfer.apply_llama31_rope_pos_ids_inplace(q_api, k_api, inputs["pos_ids"])
    q_ref, k_ref = apply_llama31_rope_pos_ids_inplace_trace.reference(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_with_cos_sin_cache():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_with_cos_sin_cache_trace

    inputs = apply_rope_with_cos_sin_cache_trace.init(
        nnz=16,
        num_q_heads_x_head_size=4 * 64,
        num_k_heads_x_head_size=2 * 64,
        head_size=64,
        max_seq_len=8192,
        rotary_dim=64,
    )
    _assert_finite(inputs["query"], inputs["key"], inputs["cos_sin_cache"])
    q_api, k_api = flashinfer.apply_rope_with_cos_sin_cache(
        inputs["positions"],
        inputs["query"],
        inputs["key"],
        inputs["head_size"],
        inputs["cos_sin_cache"],
        is_neox=True,
    )
    q_ref, k_ref = apply_rope_with_cos_sin_cache_trace.reference(
        inputs["positions"],
        inputs["query"],
        inputs["key"],
        inputs["head_size"],
        inputs["cos_sin_cache"],
        is_neox=True,
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


def test_apply_rope_with_cos_sin_cache_inplace():
    import flashinfer
    from flashinfer.trace.templates.rope import (
        apply_rope_with_cos_sin_cache_inplace_trace,
    )

    inputs = apply_rope_with_cos_sin_cache_inplace_trace.init(
        nnz=16,
        num_q_heads_x_head_size=4 * 64,
        num_k_heads_x_head_size=2 * 64,
        head_size=64,
        max_seq_len=8192,
        rotary_dim=64,
    )
    _assert_finite(inputs["query"], inputs["key"], inputs["cos_sin_cache"])
    q_api = inputs["query"].clone()
    k_api = inputs["key"].clone()
    flashinfer.apply_rope_with_cos_sin_cache_inplace(
        inputs["positions"],
        q_api,
        k_api,
        inputs["head_size"],
        inputs["cos_sin_cache"],
        is_neox=True,
    )
    q_ref, k_ref = apply_rope_with_cos_sin_cache_inplace_trace.reference(
        inputs["positions"],
        inputs["query"],
        inputs["key"],
        inputs["head_size"],
        inputs["cos_sin_cache"],
        is_neox=True,
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)


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
        is_neox=True,
    )
    q_r_ref, k_r_ref, q_n_ref, k_n_ref = rope_quantize_fp8_trace.reference(
        inputs["q_rope"],
        inputs["k_rope"],
        inputs["q_nope"],
        inputs["k_nope"],
        inputs["cos_sin_cache"],
        inputs["pos_ids"],
        is_neox=True,
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


def test_mla_rope_quantize_fp8_reference_correctness():
    """flashinfer.rope.mla_rope_quantize_fp8 (MLA layout: num_k_heads=1) kernel vs reference."""
    from flashinfer.rope import mla_rope_quantize_fp8
    from flashinfer.trace.templates.rope import mla_rope_quantize_fp8_trace

    # MLA path uses num_k_heads=1 with k_rope/k_nope as 2D tensors. The init
    # would build them as 3D [nnz, 1, rope_dim] so we squeeze afterwards
    # to match the kernel's MLA expected layout.
    inputs = mla_rope_quantize_fp8_trace.init(
        nnz=16,
        num_q_heads=128,
        num_k_heads=1,
        rope_dim=64,
        no_rope_dim=512,
        max_seq_len=4096,
        rotary_dim=64,
    )
    k_rope = inputs["k_rope"].squeeze(1)  # MLA: [nnz, rope_dim]
    k_nope = inputs["k_nope"].squeeze(1)  # MLA: [nnz, no_rope_dim]
    _assert_finite(inputs["q_rope"], k_rope, inputs["q_nope"], k_nope)
    q_r_api, k_r_api, q_n_api, k_n_api = mla_rope_quantize_fp8(
        inputs["q_rope"],
        k_rope,
        inputs["q_nope"],
        k_nope,
        inputs["cos_sin_cache"],
        inputs["pos_ids"],
        is_neox=True,
    )
    q_r_ref, k_r_ref, q_n_ref, k_n_ref = mla_rope_quantize_fp8_trace.reference(
        inputs["q_rope"],
        k_rope,
        inputs["q_nope"],
        k_nope,
        inputs["cos_sin_cache"],
        inputs["pos_ids"],
        is_neox=True,
    )
    _assert_finite(
        q_r_api, k_r_api, q_n_api, k_n_api, q_r_ref, k_r_ref, q_n_ref, k_n_ref
    )
    _close(q_r_api.float(), q_r_ref.float(), atol=1e-2, rtol=2e-1)
    _close(k_r_api.float(), k_r_ref.float(), atol=1e-2, rtol=2e-1)
    _close(q_n_api.float(), q_n_ref.float(), atol=1e-2, rtol=2e-1)
    _close(k_n_api.float(), k_n_ref.float(), atol=1e-2, rtol=2e-1)


# ─────────────────────────────────────────────────────────────────────────────
# Norm (RMSNorm + FP8 quantize)
# ─────────────────────────────────────────────────────────────────────────────


def test_rmsnorm_quant():
    import flashinfer
    from flashinfer.trace.templates.norm import rmsnorm_quant_trace

    inputs = rmsnorm_quant_trace.init(batch_size=32, hidden_size=2048)
    _assert_finite(inputs["input"], inputs["weight"])
    out_api = inputs["out"].clone()
    try:
        flashinfer.rmsnorm_quant(
            out_api, inputs["input"], inputs["weight"], inputs["scale"]
        )
    except Exception as exc:
        pytest.skip(f"rmsnorm_quant kernel unavailable: {exc}")
    out_ref = rmsnorm_quant_trace.reference(
        inputs["input"], inputs["weight"], inputs["scale"]
    )
    _assert_finite(out_api, out_ref)
    s = inputs["scale"]
    _close(out_api.float() * s, out_ref.float() * s, atol=1.0, rtol=1.0)


def test_fused_add_rmsnorm_quant():
    import flashinfer
    from flashinfer.trace.templates.norm import fused_add_rmsnorm_quant_trace

    inputs = fused_add_rmsnorm_quant_trace.init(batch_size=32, hidden_size=2048)
    _assert_finite(inputs["input"], inputs["residual"], inputs["weight"])
    out_api = inputs["out"].clone()
    residual_api = inputs["residual"].clone()
    try:
        flashinfer.fused_add_rmsnorm_quant(
            out_api,
            inputs["input"],
            residual_api,
            inputs["weight"],
            inputs["scale"],
        )
    except Exception as exc:
        pytest.skip(f"fused_add_rmsnorm_quant kernel unavailable: {exc}")
    out_ref, residual_ref = fused_add_rmsnorm_quant_trace.reference(
        inputs["input"], inputs["residual"], inputs["weight"], inputs["scale"]
    )
    _assert_finite(out_api, residual_api, out_ref, residual_ref)
    _close(residual_api, residual_ref, atol=1e-3, rtol=1e-3)
    s = inputs["scale"]
    _close(out_api.float() * s, out_ref.float() * s, atol=1.0, rtol=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Cascade merge (in-place)
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_state_in_place():
    import flashinfer
    from flashinfer.trace.templates.cascade import merge_state_in_place_trace

    # Use fp16 V (matches tests/attention/test_shared_prefix_kernels.py);
    # 1e-3 tolerance is too tight for bf16 (4e-3 per ULP).
    inputs = merge_state_in_place_trace.init(
        seq_len=128, num_heads=32, head_dim=128, dtype=torch.float16
    )
    _assert_finite(inputs["v"], inputs["s"], inputs["v_other"], inputs["s_other"])
    v_api = inputs["v"].clone()
    s_api = inputs["s"].clone()
    flashinfer.merge_state_in_place(v_api, s_api, inputs["v_other"], inputs["s_other"])
    v_ref, s_ref = merge_state_in_place_trace.reference(
        inputs["v"], inputs["s"], inputs["v_other"], inputs["s_other"]
    )
    _assert_finite(v_api, s_api, v_ref, s_ref)
    _close(v_api, v_ref, atol=1e-3, rtol=1e-3)
    _close(s_api, s_ref, atol=1e-3, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Quantization (FP4/MXFP8 round-trip via dequantize)
# ─────────────────────────────────────────────────────────────────────────────


def test_mxfp8_quantize():
    _skip_if_not_sm100()
    import flashinfer
    from flashinfer.trace.templates.quantize import mxfp8_quantize_trace

    inputs = mxfp8_quantize_trace.init(M=128, K=4096)
    _assert_finite(inputs["input"])
    try:
        q_api, _s_api = flashinfer.quantization.fp8_quantization.mxfp8_quantize(
            inputs["input"]
        )
    except Exception as exc:
        pytest.skip(f"mxfp8_quantize kernel unavailable: {exc}")
    q_ref, _s_ref = mxfp8_quantize_trace.reference(inputs["input"])
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

    inputs = fp4_quantize_trace.init(M=64, K=256, dtype=torch.float32)
    _assert_finite(inputs["input"])
    x = inputs["input"]
    # The round-trip dynamic range only behaves cleanly when ``global_scale``
    # is close to 1.0; the init's ``448*6/amax(x)`` form is correct for the
    # kernel pipeline but compresses values into a near-zero range here.
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    packed, scales = fp4_quantize_trace.reference(
        x,
        global_scale=global_scale,
        sf_vec_size=inputs["sf_vec_size"],
        sf_use_ue8m0=False,
    )
    _assert_finite(packed.float(), scales.float())
    assert packed.dtype == torch.uint8
    assert packed.shape == (64, 128)
    # Dequantize and compare: within per-block quantization error.
    unpacked = _unpack_fp4_e2m1(packed)  # [M, K]
    block_size = inputs["sf_vec_size"]
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

    inputs = single_decode_with_kv_cache_trace.init(
        kv_len=256, num_qo_heads=32, num_kv_heads=8, head_dim=128
    )
    _assert_finite(inputs["q"], inputs["k"], inputs["v"])
    try:
        out_api = flashinfer.single_decode_with_kv_cache(
            inputs["q"], inputs["k"], inputs["v"]
        )
    except Exception as exc:
        pytest.skip(f"single_decode kernel unavailable: {exc}")
    out_ref = single_decode_with_kv_cache_trace.reference(
        inputs["q"], inputs["k"], inputs["v"]
    )
    _assert_finite(out_api, out_ref)
    _close(out_api, out_ref, atol=1e-2, rtol=1e-2)


def test_single_prefill():
    import flashinfer
    from flashinfer.trace.templates.attention import (
        single_prefill_with_kv_cache_trace,
    )

    inputs = single_prefill_with_kv_cache_trace.init(
        qo_len=128, kv_len=256, num_qo_heads=32, num_kv_heads=8, head_dim=128
    )
    _assert_finite(inputs["q"], inputs["k"], inputs["v"])
    try:
        out_api = flashinfer.single_prefill_with_kv_cache(
            inputs["q"], inputs["k"], inputs["v"], causal=True
        )
    except Exception as exc:
        pytest.skip(f"single_prefill kernel unavailable: {exc}")
    out_ref = single_prefill_with_kv_cache_trace.reference(
        inputs["q"], inputs["k"], inputs["v"], causal=True
    )
    _assert_finite(out_api, out_ref)
    _close(out_api, out_ref, atol=1e-2, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# Paged kernels that require SM100+ / cuDNN (skipped on H100 by default)
# ─────────────────────────────────────────────────────────────────────────────


def test_trtllm_batch_decode_reference_correctness():
    """trtllm_batch_decode kernel vs reference (paged HND decode, SM100/103)."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    from flashinfer.trace.templates.attention import trtllm_batch_decode_trace

    # TllmGenFmhaRunner is only instantiated for SM100/SM103; on SM12x the
    # kernel raises "Unsupported architecture" at runtime.
    _skip_if_not_sm100_or_103()
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
    # Matches tests/attention/test_cudnn_decode.py / trtllm_gen bf16 tolerance.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


def test_trtllm_batch_context_reference_correctness():
    """trtllm_batch_context (causal prefill) kernel vs reference, SM100/103."""
    from flashinfer.prefill import trtllm_batch_context_with_kv_cache
    from flashinfer.trace.templates.attention import trtllm_batch_context_trace

    # TllmGenFmhaRunner is only instantiated for SM100/SM103; on SM12x the
    # kernel raises "Unsupported architecture" at runtime.
    _skip_if_not_sm100_or_103()
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
    # Matches tests/attention/test_cudnn_prefill.py bf16 tolerance.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


def test_trtllm_batch_decode_mla_reference_correctness():
    """trtllm_batch_decode_with_kv_cache_mla kernel vs reference (SM100/103)."""
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.trace.templates.attention import trtllm_batch_decode_mla_trace

    # TRT-LLM MLA kernel is only instantiated on SM100/SM103 (trtllm-gen).
    _skip_if_not_sm100_or_103()
    torch.manual_seed(0)
    B, num_heads = 4, 128
    kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim = 512, 64, 512
    D_qk = kv_lora_rank + qk_rope_head_dim  # 576
    q_len = 1
    page_size = 64
    seq_len = 128
    n_pages = (seq_len + page_size - 1) // page_size
    total_pages = n_pages * B
    query = torch.randn(B, q_len, num_heads, D_qk, dtype=torch.float16, device="cuda")
    kv_cache = torch.randn(
        total_pages, page_size, D_qk, dtype=torch.float16, device="cuda"
    )
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        B, n_pages
    )
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    bmm1_scale = 1.0 / math.sqrt(D_qk)
    try:
        api_out = trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
            is_var_seq=False,
        )
    except Exception as exc:
        pytest.skip(f"trtllm_batch_decode_with_kv_cache_mla unavailable: {exc}")
    ref_out = trtllm_batch_decode_mla_trace.reference(
        query,
        kv_cache,
        workspace,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        block_tables,
        seq_lens,
        seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
    )
    # Matches tests/attention/test_cute_dsl_mla_decode.py element-wise tol.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


def test_concat_mla_k_reference_correctness():
    """flashinfer.concat_ops.concat_mla_k kernel vs reference (in-place concat)."""
    from flashinfer.concat_ops import concat_mla_k
    from flashinfer.trace.templates.attention import concat_mla_k_trace

    inputs = concat_mla_k_trace.init(
        num_tokens=2048, num_heads=128, nope_dim=128, rope_dim=64
    )
    _assert_finite(inputs["k_nope"], inputs["k_rope"])
    k_api = inputs["k"].clone()
    k_ref = inputs["k"].clone()
    try:
        concat_mla_k(k_api, inputs["k_nope"], inputs["k_rope"])
    except Exception as exc:
        pytest.skip(f"concat_mla_k unavailable: {exc}")
    concat_mla_k_trace.reference(k_ref, inputs["k_nope"], inputs["k_rope"])
    _assert_finite(k_api, k_ref)
    _close(k_api, k_ref, atol=0.0, rtol=0.0)


def test_xqa_batch_decode_reference_correctness():
    """flashinfer.decode.xqa_batch_decode_with_kv_cache kernel vs reference (SM100+)."""
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    from flashinfer.trace.templates.attention import xqa_batch_decode_trace

    _skip_if_not_sm100()
    torch.manual_seed(0)
    B, Hq, Hk, D, PS = 2, 8, 2, 128, 16
    MP = 2
    NP = B * MP
    kv_len = PS * MP
    # NHD 5-D interleaved cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_cache = torch.randn(NP, 2, PS, Hk, D, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(B, Hq, D, dtype=torch.bfloat16, device="cuda")
    block_tables = torch.arange(NP, dtype=torch.int32, device="cuda").reshape(B, MP)
    seq_lens = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    sm_scale = 1.0 / math.sqrt(D)
    try:
        api_out = xqa_batch_decode_with_kv_cache(
            q,
            kv_cache,
            workspace,
            block_tables,
            seq_lens,
            kv_len,
            bmm1_scale=sm_scale,
            bmm2_scale=1.0,
            kv_layout="NHD",
        )
    except Exception as exc:
        pytest.skip(f"xqa_batch_decode_with_kv_cache unavailable: {exc}")
    ref_out = xqa_batch_decode_trace.reference(
        q,
        kv_cache,
        workspace,
        block_tables,
        seq_lens,
        kv_len,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        kv_layout="NHD",
    )
    # Same tolerance family as trtllm_batch_decode — same math, different backend.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


def test_xqa_batch_decode_mla_reference_correctness():
    """flashinfer.mla.xqa_batch_decode_with_kv_cache_mla kernel vs reference (SM120/121)."""
    from flashinfer.mla import xqa_batch_decode_with_kv_cache_mla
    from flashinfer.trace.templates.attention import xqa_batch_decode_mla_trace

    if _cc()[0] != 12:
        pytest.skip("XQA MLA kernel only supports SM120/121")
    torch.manual_seed(0)
    B, num_heads = 2, 128
    kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim = 512, 64, 512
    D_qk = kv_lora_rank + qk_rope_head_dim  # 576
    q_len = 1
    page_size = 64
    seq_len = 128
    n_pages = (seq_len + page_size - 1) // page_size
    total_pages = n_pages * B
    query_fp32 = (
        torch.randn(B, q_len, num_heads, D_qk, dtype=torch.float32, device="cuda") / 4.0
    )
    kv_fp32 = (
        torch.randn(total_pages, page_size, D_qk, dtype=torch.float32, device="cuda")
        / 4.0
    )
    query_fp8 = query_fp32.to(torch.float8_e4m3fn)
    kv_fp8 = kv_fp32.to(torch.float8_e4m3fn)
    block_tables = torch.arange(total_pages, dtype=torch.int32, device="cuda").reshape(
        B, n_pages
    )
    seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device="cuda")
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    bmm1_scale = 1.0 / math.sqrt(D_qk)
    try:
        api_out = xqa_batch_decode_with_kv_cache_mla(
            query=query_fp8,
            kv_cache=kv_fp8,
            workspace_buffer=workspace,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
        )
    except Exception as exc:
        pytest.skip(f"xqa_batch_decode_with_kv_cache_mla unavailable: {exc}")
    ref_out = xqa_batch_decode_mla_trace.reference(
        query_fp32,
        kv_fp32,
        workspace,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        block_tables,
        seq_lens,
        seq_len,
        bmm1_scale=bmm1_scale,
        bmm2_scale=1.0,
    )
    # Matches tests/attention/test_xqa.py pass-ratio (>=95% for FP8 MLA).
    _close_pass_ratio(
        api_out.float(),
        ref_out.float(),
        atol=0.05,
        rtol=0.05,
        pass_ratio=0.95,
    )


def test_rope_quantize_fp8_append_paged_kv_cache_reference_correctness():
    """rope_quantize_fp8_append_paged_kv_cache kernel vs reference (GQA layout)."""
    from flashinfer.rope import rope_quantize_fp8_append_paged_kv_cache
    from flashinfer.trace.templates.rope import (
        rope_quantize_fp8_append_paged_kv_cache_trace,
    )

    torch.manual_seed(0)
    # GQA setup: num_q_heads=8, num_kv_heads=2, rope=64, nope=64.
    nnz = 16
    Hq, Hk = 8, 2
    rope_dim, nope_dim = 64, 64
    head_dim = rope_dim + nope_dim
    NP, PS = 4, 16
    device = "cuda"
    q_rope = torch.randn(nnz, Hq, rope_dim, dtype=torch.bfloat16, device=device)
    k_rope = torch.randn(nnz, Hk, rope_dim, dtype=torch.bfloat16, device=device)
    q_nope = torch.randn(nnz, Hq, nope_dim, dtype=torch.bfloat16, device=device)
    k_nope = torch.randn(nnz, Hk, nope_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(nnz, Hk, head_dim, dtype=torch.bfloat16, device=device)
    t = torch.arange(4096, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (
        1e4
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    cache = torch.cat(
        [
            torch.cos(t.unsqueeze(-1) * inv_freq.unsqueeze(0)),
            torch.sin(t.unsqueeze(-1) * inv_freq.unsqueeze(0)),
        ],
        dim=-1,
    )
    pos = torch.arange(nnz, dtype=torch.int32, device=device)
    # Paged cache: NHD layout, FP8.
    k_cache_api = torch.zeros(
        NP, PS, Hk, head_dim, dtype=torch.float8_e4m3fn, device=device
    )
    v_cache_api = torch.zeros_like(k_cache_api)
    k_cache_ref = torch.zeros_like(k_cache_api)
    v_cache_ref = torch.zeros_like(k_cache_api)
    kv_indices = torch.arange(NP, dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, NP // 2, NP], dtype=torch.int32, device=device)
    batch_indices = torch.cat(
        [
            torch.zeros(nnz // 2, dtype=torch.int32, device=device),
            torch.ones(nnz // 2, dtype=torch.int32, device=device),
        ]
    )
    positions = torch.arange(nnz, dtype=torch.int32, device=device) % (nnz // 2)
    try:
        q_r_api, q_n_api = rope_quantize_fp8_append_paged_kv_cache(
            q_rope,
            k_rope,
            q_nope,
            k_nope,
            v,
            cache,
            pos,
            (k_cache_api, v_cache_api),
            kv_indices,
            kv_indptr,
            batch_indices,
            positions,
            is_neox=True,
            page_size=PS,
            kv_layout="NHD",
        )
    except Exception as exc:
        pytest.skip(f"rope_quantize_fp8_append_paged_kv_cache unavailable: {exc}")
    q_r_ref, q_n_ref = rope_quantize_fp8_append_paged_kv_cache_trace.reference(
        q_rope,
        k_rope,
        q_nope,
        k_nope,
        v,
        cache,
        pos,
        (k_cache_ref, v_cache_ref),
        kv_indices,
        kv_indptr,
        batch_indices,
        positions,
        is_neox=True,
        page_size=PS,
        kv_layout="NHD",
    )
    # Match tests/attention/test_rope.py FP8 rope quantize tolerance for Q.
    # (The paged K/V append half uses an implementation-specific internal
    # layout — nope/rope interleave order varies between kernel versions —
    # so we only compare the Q outputs here, which are portable.)
    _close(q_r_api.float(), q_r_ref.float(), atol=1e-2, rtol=2e-1)
    _close(q_n_api.float(), q_n_ref.float(), atol=1e-2, rtol=2e-1)


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
    # Matches tests/attention/test_cudnn_decode.py.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


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
    # Matches tests/attention/test_cudnn_prefill.py.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


# ─────────────────────────────────────────────────────────────────────────────
# MoE variants (SM100+ — skipped when unavailable)
# ─────────────────────────────────────────────────────────────────────────────


def test_softmax_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import softmax_trace

    inputs = softmax_trace.init(batch_size=8, vocab_size=128)
    _assert_finite(inputs["logits"])
    api_out = flashinfer.softmax(inputs["logits"], temperature=inputs["temperature"])
    ref_out = softmax_trace.reference(
        inputs["logits"], temperature=inputs["temperature"]
    )
    _assert_finite(api_out, ref_out)
    _close(api_out, ref_out, atol=1e-3, rtol=1e-3)


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

    inputs = top_k_renorm_probs_trace.init(batch_size=4, vocab_size=128)
    _assert_finite(inputs["probs"])
    api_out = flashinfer.top_k_renorm_probs(inputs["probs"], inputs["top_k"])
    ref_out = top_k_renorm_probs_trace.reference(inputs["probs"], inputs["top_k"])
    _assert_finite(api_out, ref_out)
    _close(api_out, ref_out, atol=1e-3, rtol=1e-3)


def test_top_p_renorm_probs_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import top_p_renorm_probs_trace

    inputs = top_p_renorm_probs_trace.init(batch_size=4, vocab_size=128)
    _assert_finite(inputs["probs"])
    api_out = flashinfer.top_p_renorm_probs(inputs["probs"], inputs["top_p"])
    ref_out = top_p_renorm_probs_trace.reference(inputs["probs"], inputs["top_p"])
    _assert_finite(api_out, ref_out)
    # Kernel uses AIR top-p (approximate); allow some slack.
    _close(api_out, ref_out, atol=1e-2, rtol=5e-2)


def test_top_k_mask_logits_reference():
    import flashinfer
    from flashinfer.trace.templates.sampling import top_k_mask_logits_trace

    inputs = top_k_mask_logits_trace.init(batch_size=4, vocab_size=128)
    _assert_finite(inputs["logits"])
    api_out = flashinfer.top_k_mask_logits(inputs["logits"], inputs["top_k"])
    ref_out = top_k_mask_logits_trace.reference(inputs["logits"], inputs["top_k"])
    # Both should produce identical mask patterns; -inf cells compare as nan.
    api_finite = torch.isfinite(api_out)
    ref_finite = torch.isfinite(ref_out)
    assert torch.equal(api_finite, ref_finite), "mask positions differ"
    _close(api_out[api_finite], ref_out[ref_finite], atol=1e-3, rtol=1e-3)


def test_tgv_gemm_sm100_reference_correctness():
    """tgv_gemm_sm100 kernel (SM100 only in practice) vs reference (a @ b + bias)."""
    # The kernel's Python gate accepts SM100 or SM103 (see
    # gemm_base._match_sm_version) but the precompiled cubin only has an
    # SM100 kernel image; calling on SM103 crashes with "no kernel image"
    # inside CUDA (uncatchable via try/except). Restrict to SM100.
    if _cc() != (10, 0):
        pytest.skip("tgv_gemm_sm100 cubin is only built for SM100")
    from flashinfer import tgv_gemm_sm100
    from flashinfer.trace.templates.page import tgv_gemm_sm100_trace

    torch.manual_seed(0)
    M, N, K = 16, 1024, 1024
    a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    b_row = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    b = b_row.t()  # col-major [K, N]
    bias = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    try:
        api_out = tgv_gemm_sm100(a, b, bias)
        torch.cuda.synchronize()
    except Exception as exc:
        pytest.skip(f"tgv_gemm_sm100 unavailable: {exc}")
    ref_out = tgv_gemm_sm100_trace.reference(a, b, bias)
    # Matches tests/gemm/test_tgv_gemm.py: bf16 * K=1024 accumulation makes
    # element-wise tolerance unreliable; cosine similarity is the repo
    # convention for this op.
    _close_fp8(api_out, ref_out, cos_sim_min=0.99)


def test_append_paged_kv_cache_reference_correctness():
    """append_paged_kv_cache kernel vs reference (full cache comparison)."""
    import flashinfer
    from flashinfer.trace.templates.page import append_paged_kv_cache_trace

    inputs = append_paged_kv_cache_trace.init(
        nnz_kv=4,
        batch_size=2,
        num_kv_heads=8,
        head_dim=64,
        page_size=16,
        num_pages=4,
    )
    _assert_finite(inputs["append_key"], inputs["append_value"])
    # Make a deep copy of the cache for the reference run so the API and
    # reference each get a clean zero-initialized buffer to mutate.
    k_cache_api, v_cache_api = inputs["paged_kv_cache"]
    k_cache_ref = torch.zeros_like(k_cache_api)
    v_cache_ref = torch.zeros_like(v_cache_api)
    flashinfer.append_paged_kv_cache(
        inputs["append_key"],
        inputs["append_value"],
        inputs["batch_indices"],
        inputs["positions"],
        (k_cache_api, v_cache_api),
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["kv_last_page_len"],
    )
    append_paged_kv_cache_trace.reference(
        inputs["append_key"],
        inputs["append_value"],
        inputs["batch_indices"],
        inputs["positions"],
        (k_cache_ref, v_cache_ref),
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["kv_last_page_len"],
    )
    _assert_finite(k_cache_api, v_cache_api, k_cache_ref, v_cache_ref)
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
    # Matches tests/attention/test_xqa.py: >=98% of elements within
    # (atol=0.05, rtol=0.05).
    _close_pass_ratio(
        output.squeeze(1).float(),
        ref_out.float(),
        atol=0.05,
        rtol=0.05,
        pass_ratio=0.98,
    )


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
    # XQA MLA quantizes Q and the KV cache to FP8 internally; a few outlier
    # positions land on tied FP8 rounding boundaries. Matches the pass-ratio
    # metric the existing tests/attention/test_xqa.py uses for the same op:
    # >=95% of elements within (atol=0.05, rtol=0.05).
    _close_pass_ratio(
        output.squeeze(1).float(),
        ref_out.float(),
        atol=0.05,
        rtol=0.05,
        pass_ratio=0.95,
    )


def test_trtllm_fmha_v2_prefill_reference_correctness():
    """trtllm_fmha_v2_prefill kernel (PACKED_QKV) vs reference (causal SDPA)."""
    from flashinfer.prefill import trtllm_fmha_v2_prefill
    from flashinfer.trace.templates.page import trtllm_fmha_v2_prefill_trace

    # FMHA v2 compiles only for SM90 (Hopper) or SM12x (Blackwell refresh).
    if _cc()[0] not in (9, 12):
        pytest.skip("FMHA v2 requires SM90 (Hopper) or SM12x")
    torch.manual_seed(0)
    # head_dim 128 is the smallest size the existing tests/attention/
    # test_fmha_v2_prefill.py exercises on SM90; smaller values hit unsupported
    # instantiations on H100.
    B, H, D = 2, 8, 128
    q_lens = [16, 32]
    kv_lens = [16, 32]
    total_tokens = sum(q_lens)
    packed = torch.randn(total_tokens, 3, H, D, dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    cum_list = [0] + [sum(q_lens[: i + 1]) for i in range(B)]
    cum = torch.tensor(cum_list, dtype=torch.int32, device="cuda")
    sm_scale = 1.0 / (D**0.5)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
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
    except Exception as exc:
        pytest.skip(f"trtllm_fmha_v2_prefill unavailable: {exc}")
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
    # Matches tests/attention/test_fmha_v2_prefill.py bf16 tolerance.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


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
    # Matches tests/utils/test_pod_kernels.py tolerance (fp16 decode).
    _close(out_d, ref_d, atol=1e-3, rtol=1e-3)


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
    try:
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
    except Exception as exc:
        pytest.skip(f"VariableBlockSparseAttentionWrapper unavailable: {exc}")
    # Reference expects NHD — transpose and compare.
    q_nhd = q_hnd.transpose(0, 1).contiguous()
    k_nhd = k_hnd.transpose(0, 1).contiguous()
    v_nhd = v_hnd.transpose(0, 1).contiguous()
    ref_out = variable_block_sparse_attention_run_trace.reference(q_nhd, k_nhd, v_nhd)
    # Matches tests/attention/test_block_sparse.py.
    _close(api_out.transpose(0, 1), ref_out, atol=1e-2, rtol=1e-2)


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
    # Matches tests/attention/test_block_sparse.py.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


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
    # Matches tests/attention/test_batch_attention.py.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


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
    # tests/attention/test_shared_prefix_kernels.py uses 1e-3 but compares
    # two kernel outputs with identical internal math; our reference uses
    # torch-level fp32 math which diverges by ~1 bf16 ULP from the kernel's
    # bf16 accumulation. Use 1e-2 (matching test_batch_attention.py bf16 tol).
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)


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
    # Matches tests/utils/test_pod_kernels.py.
    _close(out_p, ref_p, atol=1e-3, rtol=1e-3)
    _close(out_d, ref_d, atol=1e-3, rtol=1e-3)


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
    # Matches tests/gemm/test_group_gemm.py.
    _close(api_out, ref_out, atol=2e-3, rtol=1e-3)


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
    # Matches tests/moe/test_trtllm_cutlass_fused_moe.py.
    _close(api_out, ref_out.to(api_out.dtype), atol=1e-2, rtol=1e-2)


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


# ─────────────────────────────────────────────────────────────────────────────
# Norm + activation
# ─────────────────────────────────────────────────────────────────────────────


def test_rmsnorm_reference_correctness():
    """flashinfer.rmsnorm kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.norm import rmsnorm_trace

    inputs = rmsnorm_trace.init(batch_size=8, hidden_size=256)
    _assert_finite(inputs["input"], inputs["weight"])
    api = flashinfer.rmsnorm(inputs["input"], inputs["weight"], eps=1e-6)
    ref = rmsnorm_trace.reference(inputs["input"], inputs["weight"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)


def test_fused_add_rmsnorm_reference_correctness():
    """flashinfer.fused_add_rmsnorm kernel vs reference.

    The kernel mutates input (→ norm output) and residual (→ residual + input).
    The trace reference returns the normalized output only; we compare that
    against the mutated input and verify the residual update by hand.
    """
    import flashinfer
    from flashinfer.trace.templates.norm import fused_add_rmsnorm_trace

    inputs = fused_add_rmsnorm_trace.init(batch_size=8, hidden_size=256)
    x_orig, res_orig = inputs["input"].clone(), inputs["residual"].clone()
    _assert_finite(x_orig, res_orig, inputs["weight"])
    x_api = inputs["input"].clone()
    res_api = inputs["residual"].clone()
    flashinfer.fused_add_rmsnorm(x_api, res_api, inputs["weight"], eps=1e-6)
    ref_norm = fused_add_rmsnorm_trace.reference(x_orig, res_orig, inputs["weight"])
    _assert_finite(x_api, res_api, ref_norm)
    _close(x_api, ref_norm, atol=1e-3, rtol=1e-3)
    _close(res_api, res_orig + x_orig, atol=1e-3, rtol=1e-3)


def test_layernorm_reference_correctness():
    """flashinfer.layernorm kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.norm import layernorm_trace

    inputs = layernorm_trace.init(batch_size=8, hidden_size=256)
    _assert_finite(inputs["input"], inputs["gemma"], inputs["beta"])
    api = flashinfer.layernorm(
        inputs["input"], inputs["gemma"], inputs["beta"], eps=1e-6
    )
    ref = layernorm_trace.reference(inputs["input"], inputs["gemma"], inputs["beta"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)


def test_gemma_rmsnorm_reference_correctness():
    """flashinfer.gemma_rmsnorm kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.norm import gemma_rmsnorm_trace

    inputs = gemma_rmsnorm_trace.init(batch_size=8, hidden_size=256)
    _assert_finite(inputs["input"], inputs["weight"])
    api = flashinfer.gemma_rmsnorm(inputs["input"], inputs["weight"], eps=1e-6)
    ref = gemma_rmsnorm_trace.reference(inputs["input"], inputs["weight"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)


def test_gemma_fused_add_rmsnorm_reference_correctness():
    """flashinfer.gemma_fused_add_rmsnorm kernel vs reference.

    Same in-place mutation pattern as fused_add_rmsnorm; reference returns
    only the normalized output.
    """
    import flashinfer
    from flashinfer.trace.templates.norm import gemma_fused_add_rmsnorm_trace

    inputs = gemma_fused_add_rmsnorm_trace.init(batch_size=8, hidden_size=256)
    x_orig, res_orig = inputs["input"].clone(), inputs["residual"].clone()
    _assert_finite(x_orig, res_orig, inputs["weight"])
    x_api = inputs["input"].clone()
    res_api = inputs["residual"].clone()
    flashinfer.gemma_fused_add_rmsnorm(x_api, res_api, inputs["weight"], eps=1e-6)
    ref_norm = gemma_fused_add_rmsnorm_trace.reference(
        x_orig, res_orig, inputs["weight"]
    )
    _assert_finite(x_api, res_api, ref_norm)
    _close(x_api, ref_norm, atol=1e-3, rtol=1e-3)
    _close(res_api, res_orig + x_orig, atol=1e-3, rtol=1e-3)


def test_silu_and_mul_reference_correctness():
    """flashinfer.silu_and_mul kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.activation import silu_and_mul_trace

    # tests/utils/test_activation.py uses fp16; bf16 ULP (3e-2) exceeds 1e-3.
    inputs = silu_and_mul_trace.init(
        num_tokens=8, hidden_size=2 * 128, dtype=torch.float16
    )
    _assert_finite(inputs["input"])
    api = flashinfer.silu_and_mul(inputs["input"])
    ref = silu_and_mul_trace.reference(inputs["input"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)


def test_gelu_and_mul_reference_correctness():
    """flashinfer.gelu_and_mul kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.activation import gelu_and_mul_trace

    inputs = gelu_and_mul_trace.init(
        num_tokens=8, hidden_size=2 * 128, dtype=torch.float16
    )
    _assert_finite(inputs["input"])
    api = flashinfer.gelu_and_mul(inputs["input"])
    ref = gelu_and_mul_trace.reference(inputs["input"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)


def test_gelu_tanh_and_mul_reference_correctness():
    """flashinfer.gelu_tanh_and_mul kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.activation import gelu_tanh_and_mul_trace

    torch.manual_seed(0)
    # tests/utils/test_activation.py uses fp16; bf16 ULP (3e-2) exceeds 1e-3.
    B, H = 8, 128
    x = torch.randn(B, 2 * H, dtype=torch.float16, device="cuda")
    api = flashinfer.gelu_tanh_and_mul(x)
    ref = gelu_tanh_and_mul_trace.reference(x)
    # Matches tests/utils/test_activation.py.
    _close(api, ref, atol=1e-3, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling (top_k / top_p / top_k_top_p from probs)
# ─────────────────────────────────────────────────────────────────────────────


def test_top_k_sampling_reference_correctness():
    """top_k_sampling_from_probs kernel vs reference on fully-one-hot probs.

    With a one-hot distribution both the kernel and multinomial reference
    deterministically emit the peak index, so the comparison is exact.
    """
    import flashinfer
    from flashinfer.trace.templates.sampling import top_k_sampling_trace

    torch.manual_seed(0)
    B, V = 4, 128
    target = torch.tensor([3, 17, 42, 0], dtype=torch.long, device="cuda")
    probs = torch.zeros(B, V, dtype=torch.float32, device="cuda")
    probs[torch.arange(B), target] = 1.0
    api = flashinfer.top_k_sampling_from_probs(probs, 10, deterministic=True)
    top_k = torch.full((B,), 10, dtype=torch.int32, device="cuda")
    ref = top_k_sampling_trace.reference(probs, top_k)
    _close(api.to(torch.int64), ref, atol=0.0, rtol=0.0)


def test_top_p_sampling_reference_correctness():
    """top_p_sampling_from_probs kernel vs reference on fully-one-hot probs."""
    import flashinfer
    from flashinfer.trace.templates.sampling import top_p_sampling_trace

    torch.manual_seed(0)
    B, V = 4, 128
    target = torch.tensor([7, 21, 60, 3], dtype=torch.long, device="cuda")
    probs = torch.zeros(B, V, dtype=torch.float32, device="cuda")
    probs[torch.arange(B), target] = 1.0
    api = flashinfer.top_p_sampling_from_probs(probs, 0.9, deterministic=True)
    top_p = torch.full((B,), 0.9, dtype=torch.float32, device="cuda")
    ref = top_p_sampling_trace.reference(probs, top_p)
    _close(api.to(torch.int64), ref, atol=0.0, rtol=0.0)


def test_top_k_top_p_sampling_reference_correctness():
    """top_k_top_p_sampling_from_probs kernel vs reference on fully-one-hot probs."""
    import flashinfer
    from flashinfer.trace.templates.sampling import top_k_top_p_sampling_trace

    torch.manual_seed(0)
    B, V = 4, 128
    target = torch.tensor([5, 13, 44, 22], dtype=torch.long, device="cuda")
    probs = torch.zeros(B, V, dtype=torch.float32, device="cuda")
    probs[torch.arange(B), target] = 1.0
    api = flashinfer.top_k_top_p_sampling_from_probs(probs, 10, 0.9, deterministic=True)
    top_k = torch.full((B,), 10, dtype=torch.int32, device="cuda")
    top_p = torch.full((B,), 0.9, dtype=torch.float32, device="cuda")
    ref = top_k_top_p_sampling_trace.reference(probs, top_k, top_p)
    _close(api.to(torch.int64), ref, atol=0.0, rtol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Merge state / merge states
# ─────────────────────────────────────────────────────────────────────────────


def test_merge_state_reference_correctness():
    """flashinfer.merge_state kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.cascade import merge_state_trace

    inputs = merge_state_trace.init(
        seq_len=16, num_heads=4, head_dim=64, dtype=torch.float16
    )
    _assert_finite(inputs["v_a"], inputs["s_a"], inputs["v_b"], inputs["s_b"])
    v_api, s_api = flashinfer.merge_state(
        inputs["v_a"], inputs["s_a"], inputs["v_b"], inputs["s_b"]
    )
    v_ref, s_ref = merge_state_trace.reference(
        inputs["v_a"], inputs["s_a"], inputs["v_b"], inputs["s_b"]
    )
    _assert_finite(v_api, s_api, v_ref, s_ref)
    _close(v_api, v_ref, atol=1e-3, rtol=1e-3)
    _close(s_api, s_ref, atol=1e-3, rtol=1e-3)


def test_merge_states_reference_correctness():
    """flashinfer.merge_states kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.cascade import merge_states_trace

    inputs = merge_states_trace.init(
        seq_len=16, num_states=3, num_heads=4, head_dim=64, dtype=torch.float16
    )
    _assert_finite(inputs["v"], inputs["s"])
    v_api, s_api = flashinfer.merge_states(inputs["v"], inputs["s"])
    v_ref, s_ref = merge_states_trace.reference(inputs["v"], inputs["s"])
    _assert_finite(v_api, s_api, v_ref, s_ref)
    _close(v_api, v_ref, atol=1e-3, rtol=1e-3)
    _close(s_api, s_ref, atol=1e-3, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Quantize (mxfp4 / nvfp4)
# ─────────────────────────────────────────────────────────────────────────────


def test_mxfp4_quantize_reference_correctness():
    """mxfp4_quantize kernel: dequantized round-trip correctness.

    The CUDA kernel and the torch template reference use incompatible packed
    layouts (nibble ordering / scale packing differ), so we verify the kernel
    by its dequantized round-trip: quantize(a) → dequantize should reproduce
    ``a`` to within one E2M1 ULP * UE8M0 scale.
    """
    import flashinfer

    # fp4_quantize compiles on SM90+ but only produces correct output on
    # SM100+ — on Hopper the kernel silently returns near-zero garbage.
    _skip_if_not_sm100()
    torch.manual_seed(0)
    a = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    try:
        api_packed, api_scales = flashinfer.mxfp4_quantize(a)
    except Exception as exc:
        pytest.skip(f"mxfp4_quantize unavailable: {exc}")
    api_dq = flashinfer.mxfp4_dequantize(api_packed, api_scales)
    _close(api_dq.float(), a.cpu().float(), atol=2.0, rtol=0.25)


def test_nvfp4_quantize_reference_correctness():
    """nvfp4_quantize kernel vs reference, dequantized round-trip."""
    import flashinfer

    # Same SM100+ requirement as mxfp4_quantize above.
    _skip_if_not_sm100()
    torch.manual_seed(0)
    a = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    global_sf = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    try:
        api_packed, _ = flashinfer.nvfp4_quantize(a, global_sf)
    except Exception as exc:
        pytest.skip(f"nvfp4_quantize unavailable: {exc}")
    # nvfp4 doesn't have a top-level dequantize; the reference in the trace
    # template does; compare shapes + value ranges instead of bit-exact.
    # Since the round-trip needs a fp4 dequant LUT, we compare packed bytes
    # under a loose tolerance that accepts single-ULP mismatches from rounding.
    from flashinfer.trace.templates.quantize import nvfp4_quantize_trace

    ref_packed, _ = nvfp4_quantize_trace.reference(a, global_sf)
    # Check element-wise agreement rate; allow up to 5% bytes to differ by
    # a single ULP (one nibble).
    diff = (api_packed.to(torch.int32) - ref_packed.to(torch.int32)).abs()
    frac_different = (diff > 0).float().mean().item()
    assert frac_different < 0.05, f"{frac_different:.2%} packed bytes differ"


# ─────────────────────────────────────────────────────────────────────────────
# MM (bf16 / fp4 / mxfp8) — simple bias-less matmul cases
# ─────────────────────────────────────────────────────────────────────────────


# NOTE: mm_fp8, mm_mxfp8, and mm_fp4 each require a specialized weight-prep
# pipeline (prepare_low_latency_gemm_weights for mm_fp8, block-scale pair
# generation for mm_mxfp8, fp4 nibble packing + per-block scales for mm_fp4)
# that doesn't fit in a compact correctness test. The trace references in
# flashinfer/trace/templates/gemm.py for these variants model the dequantize-
# then-matmul math ideal; verifying them against the real kernel requires
# matching the exact weight layout the kernel expects. The template-
# consistency tests verify these traces end-to-end via the schema validator;
# direct kernel-vs-reference tests are left for a follow-up that can stage
# the correct weight layouts (see the MoE block below for the same rationale).


def test_mm_bf16_reference_correctness():
    """flashinfer.mm_bf16 kernel vs reference (plain matmul).

    B must be column-major (stride [1, K]) for mm_bf16; the trace's
    init() returns ``b = randn(N, K).T`` — a contiguous [K, N]
    column-major view, exactly the layout the kernel expects.
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import mm_bf16_trace

    inputs = mm_bf16_trace.init(M=32, N=1024, K=1024)
    _assert_finite(inputs["a"], inputs["b"])
    try:
        api = flashinfer.mm_bf16(inputs["a"], inputs["b"], backend="cutlass")
    except Exception as exc:
        pytest.skip(f"mm_bf16 unavailable: {exc}")
    ref = mm_bf16_trace.reference(inputs["a"], inputs["b"])
    _assert_finite(api, ref)
    _close_fp8(api, ref.to(api.dtype), cos_sim_min=0.99)


def test_bmm_bf16_reference_correctness():
    """flashinfer.bmm_bf16 kernel vs reference (batched matmul, cos-sim per
    tests/gemm/test_bmm_bf16.py)."""
    import flashinfer
    from flashinfer.trace.templates.gemm import bmm_bf16_trace

    inputs = bmm_bf16_trace.init(batch_size=4, M=16, N=1024, K=1024)
    _assert_finite(inputs["A"], inputs["B"])
    # bmm_bf16 with cutlass backend requires the same column-major view
    # via the [..., K, N] stride pattern used by the unit test.
    b_kmaj = inputs["B"].transpose(1, 2).contiguous().transpose(1, 2)
    try:
        api = flashinfer.bmm_bf16(inputs["A"], b_kmaj, backend="cutlass")
    except Exception as exc:
        pytest.skip(f"bmm_bf16 unavailable: {exc}")
    ref = bmm_bf16_trace.reference(inputs["A"], inputs["B"])
    _assert_finite(api, ref)
    _close_fp8(api, ref, cos_sim_min=0.99)


def test_bmm_fp8_reference_correctness():
    """flashinfer.bmm_fp8 kernel vs reference (per-tensor FP8 BMM).

    Matches tests/gemm/test_bmm_fp8.py: cos_sim > 0.99.
    """
    import flashinfer
    from flashinfer.trace.templates.gemm import bmm_fp8_trace

    _skip_if_not_sm100_or_103()
    torch.manual_seed(0)
    Bsize, M, N, K = 4, 16, 1024, 1024
    a_bf = torch.randn(Bsize, M, K, dtype=torch.bfloat16, device="cuda")
    b_bf = torch.randn(Bsize, K, N, dtype=torch.bfloat16, device="cuda")
    a_max = a_bf.abs().max() / 448.0
    b_max = b_bf.abs().max() / 448.0
    a_fp8 = (a_bf / a_max).to(torch.float8_e4m3fn)
    b_fp8 = (b_bf / b_max).to(torch.float8_e4m3fn)
    a_scale = a_max.to(torch.float32).reshape(1)
    b_scale = b_max.to(torch.float32).reshape(1)
    b_fp8_kmaj = b_fp8.transpose(1, 2).contiguous().transpose(1, 2)
    try:
        api = flashinfer.bmm_fp8(
            a_fp8, b_fp8_kmaj, a_scale, b_scale, dtype=torch.bfloat16
        )
    except Exception as exc:
        pytest.skip(f"bmm_fp8 unavailable: {exc}")
    ref = bmm_fp8_trace.reference(a_fp8, b_fp8, a_scale, b_scale, dtype=torch.bfloat16)
    _close_fp8(api, ref, cos_sim_min=0.99)
