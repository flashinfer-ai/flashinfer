"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math

import pytest
import torch
import torch.nn.functional as F


def _patch_cutlass_dsl_operand_major_mode():
    try:
        import cutlass.cute as cute
        from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
    except ImportError:
        return
    if not hasattr(cute.nvgpu, "OperandMajorMode"):
        cute.nvgpu.OperandMajorMode = OperandMajorMode


_patch_cutlass_dsl_operand_major_mode()

import flashinfer
from flashinfer.utils import is_sm120a_supported, is_sm121a_supported


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if not (is_sm120a_supported(device) or is_sm121a_supported(device)):
        pytest.skip("SM120 or SM121 GPU is required")


def _pad_seq_len_to_128(x):
    pad_len = (-x.shape[2]) % 128
    if pad_len == 0:
        return x.contiguous()
    return torch.nn.functional.pad(x, (0, 0, 0, pad_len), value=0).contiguous()


def _preprocess_qkv_ref(q, k, v):
    k = k - k.mean(dim=-2, keepdim=True)
    q, k, v = map(_pad_seq_len_to_128, (q, k, v))
    batch, num_heads, seq_len, head_dim = q.shape
    q_grouped = q.reshape(batch, num_heads, seq_len // 128, 128, head_dim)
    qm = q_grouped.mean(dim=3)
    q = (
        (q_grouped - qm.unsqueeze(3))
        .reshape(batch, num_heads, seq_len, head_dim)
        .contiguous()
    )
    qk_correction = (
        torch.matmul(qm, k.transpose(-2, -1))
        .repeat_interleave(128, dim=2)
        .to(torch.float32)
        .contiguous()
    )
    return q, k, v, qk_correction


def _reference_attention(q, k, v, causal):
    q, k, v, qk_correction = _preprocess_qkv_ref(q, k, v)
    sm_scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    scores = scores + qk_correction * sm_scale
    if causal:
        seq_len = q.shape[2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


def _expand_quantized_kv_to_qo_heads(quantized):
    q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction = quantized
    num_qo_heads = q_fp4.shape[1]
    num_kv_heads = k_fp4.shape[1]
    kv_indices = torch.arange(num_qo_heads, device=q_fp4.device) // (
        num_qo_heads // num_kv_heads
    )
    return (
        q_fp4,
        k_fp4.index_select(1, kv_indices).contiguous(),
        v_fp4_t.index_select(1, kv_indices).contiguous(),
        q_scale,
        k_scale.index_select(1, kv_indices).contiguous(),
        v_scale_t.index_select(1, kv_indices).contiguous(),
        qk_correction,
    )


def _run_nvfp4_attention_sm120_accuracy_case(
    batch,
    num_heads,
    seq_len,
    head_dim,
    causal,
    cos_threshold,
    mean_abs_err_threshold,
):
    _require_sm120()

    torch.manual_seed(42)
    q = torch.randn(
        (batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction = (
        flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)
    )

    out, lse = flashinfer.nvfp4_attention_sm120_fwd(
        q_fp4,
        k_fp4,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        qk_correction,
        sm_scale=head_dim**-0.5,
        causal=causal,
        return_lse=True,
    )

    torch.cuda.synchronize()
    ref = _reference_attention(q, k, v, causal)[:, :, :seq_len, :]
    out = out[:, :, :seq_len, :]
    lse = lse[:, :, :seq_len]

    assert out.shape == ref.shape
    assert lse.shape == (batch, num_heads, seq_len)
    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert not torch.isnan(lse).any()
    assert not torch.isinf(lse).any()

    mean_abs_err = (out.float() - ref.float()).abs().mean().item()
    cos_sim = F.cosine_similarity(
        out.float().reshape(1, -1), ref.float().reshape(1, -1)
    ).item()
    assert mean_abs_err <= mean_abs_err_threshold
    assert cos_sim >= cos_threshold


@pytest.mark.parametrize(
    (
        "batch",
        "num_heads",
        "seq_len",
        "head_dim",
        "causal",
        "cos_threshold",
        "mean_abs_err_threshold",
    ),
    [
        pytest.param(1, 4, 128, 64, False, 0.95, 0.08, id="s128-d64-noncausal"),
        pytest.param(1, 4, 256, 128, False, 0.95, 0.06, id="s256-d128-noncausal"),
        pytest.param(1, 4, 256, 128, True, 0.94, 0.09, id="s256-d128-causal"),
        pytest.param(1, 1, 4096, 64, False, 0.95, 0.02, id="s4096-d64-noncausal"),
        pytest.param(1, 1, 4096, 128, True, 0.95, 0.04, id="s4096-d128-causal"),
        pytest.param(1, 1, 8192, 64, False, 0.95, 0.02, id="s8192-d64-noncausal"),
    ],
)
@torch.inference_mode()
def test_nvfp4_attention_sm120_accuracy(
    batch,
    num_heads,
    seq_len,
    head_dim,
    causal,
    cos_threshold,
    mean_abs_err_threshold,
):
    _run_nvfp4_attention_sm120_accuracy_case(
        batch,
        num_heads,
        seq_len,
        head_dim,
        causal,
        cos_threshold,
        mean_abs_err_threshold,
    )


@pytest.mark.parametrize(
    ("num_qo_heads", "num_kv_heads", "seq_len", "head_dim", "causal", "per_block_mean"),
    [
        pytest.param(8, 4, 256, 64, False, True, id="gqa2-d64"),
        pytest.param(8, 2, 384, 128, True, True, id="gqa4-d128-causal"),
        pytest.param(8, 1, 256, 128, False, False, id="mqa-d128-global-mean"),
    ],
)
@torch.inference_mode()
def test_nvfp4_attention_sm120_gqa_matches_expanded_packed_oracle(
    num_qo_heads,
    num_kv_heads,
    seq_len,
    head_dim,
    causal,
    per_block_mean,
):
    _require_sm120()

    torch.manual_seed(42)
    q = torch.randn(
        (1, num_qo_heads, seq_len, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        (1, num_kv_heads, seq_len, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    quantized = flashinfer.nvfp4_attention_sm120_quantize_qkv(
        q, k, v, per_block_mean=per_block_mean
    )
    q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, correction = quantized
    num_q_blocks = seq_len // 128 if per_block_mean else 1
    assert q_fp4.shape == (1, num_qo_heads, seq_len, head_dim // 2)
    assert k_fp4.shape == (1, num_kv_heads, seq_len, head_dim // 2)
    assert v_fp4_t.shape == (1, num_kv_heads, head_dim, seq_len // 2)
    assert q_scale.shape == (1, num_qo_heads, seq_len, head_dim // 16)
    assert k_scale.shape == (1, num_kv_heads, seq_len, head_dim // 16)
    assert v_scale_t.shape == (1, num_kv_heads, head_dim, seq_len // 16)
    assert correction.shape == (1, num_qo_heads, num_q_blocks, seq_len)

    out_gqa, lse_gqa = flashinfer.nvfp4_attention_sm120_fwd(
        *quantized,
        causal=causal,
        per_block_mean=per_block_mean,
        return_lse=True,
    )
    out_expanded, lse_expanded = flashinfer.nvfp4_attention_sm120_fwd(
        *_expand_quantized_kv_to_qo_heads(quantized),
        causal=causal,
        per_block_mean=per_block_mean,
        return_lse=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_gqa, out_expanded, rtol=0, atol=0)
    torch.testing.assert_close(lse_gqa, lse_expanded, rtol=0, atol=0)

    kv_indices = torch.arange(num_qo_heads, device=q.device) // (
        num_qo_heads // num_kv_heads
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.index_select(1, kv_indices).float(),
        v.index_select(1, kv_indices).float(),
        is_causal=causal,
        scale=head_dim**-0.5,
    )
    cos_sim = F.cosine_similarity(
        out_gqa.float().reshape(1, -1), ref.reshape(1, -1)
    ).item()
    assert cos_sim >= 0.94


@torch.inference_mode()
def test_nvfp4_attention_sm120_rejects_nonuniform_gqa_ratio():
    _require_sm120()

    q = torch.randn((1, 6, 128, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    with pytest.raises(ValueError, match="num_qo_heads"):
        flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)


@torch.inference_mode()
def test_nvfp4_attention_sm120_rejects_kv_head_correction():
    _require_sm120()

    q = torch.randn((1, 8, 128, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    quantized = list(flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v))
    quantized[6] = quantized[6][:, :2].contiguous()

    with pytest.raises(ValueError, match="qk_correction"):
        flashinfer.nvfp4_attention_sm120_fwd(*quantized)


@pytest.mark.parametrize("per_block_mean", [True, False])
@torch.inference_mode()
def test_nvfp4_attention_sm120_structured_q_correction(per_block_mean):
    """Q with real block structure makes qk_correction large; a misaddressed
    correction tensor collapses accuracy (regression test for the expanded
    [B, H, S, S] correction layout the kernel never addressed)."""
    _require_sm120()

    torch.manual_seed(42)
    batch, num_heads, seq_len, head_dim = 2, 4, 1024, 128
    q = torch.randn(
        (batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16
    )
    if per_block_mean:
        # distinct mean per 128-token block, batch, and head
        bias = torch.randn(
            (batch, num_heads, seq_len // 128, 1, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        q = (q.view(batch, num_heads, seq_len // 128, 128, head_dim) + bias).view(
            batch, num_heads, seq_len, head_dim
        )
    else:
        # distinct mean per batch and head, constant across the sequence,
        # so the full-sequence Q centering removes it exactly
        bias = torch.randn(
            (batch, num_heads, 1, head_dim), device="cuda", dtype=torch.bfloat16
        )
        q = q + bias
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction = (
        flashinfer.nvfp4_attention_sm120_quantize_qkv(
            q, k, v, per_block_mean=per_block_mean
        )
    )
    expected_rows = seq_len // 128 if per_block_mean else 1
    assert qk_correction.shape == (batch, num_heads, expected_rows, seq_len)

    out = flashinfer.nvfp4_attention_sm120_fwd(
        q_fp4,
        k_fp4,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        qk_correction,
        sm_scale=head_dim**-0.5,
        causal=False,
        per_block_mean=per_block_mean,
        return_lse=False,
    )
    torch.cuda.synchronize()

    # smoothing + correction reproduce plain attention exactly, so the exact
    # output is the reference; the remaining gap is FP4 quantization error
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=head_dim**-0.5
    )
    cos_sim = F.cosine_similarity(out.float().reshape(1, -1), ref.reshape(1, -1)).item()
    assert cos_sim >= 0.95


@torch.inference_mode()
def test_nvfp4_attention_sm120_output_magnitude():
    """With uniform scores and V = ones the exact output is 1.0 everywhere;
    a mis-reduced row_sum shows up as a uniform scale error that cosine
    thresholds cannot see (regression test for the halved output)."""
    _require_sm120()

    batch, num_heads, seq_len, head_dim = 1, 2, 512, 128
    q = torch.zeros(
        (batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.zeros_like(q)
    v = torch.ones_like(q)

    quantized = flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out, lse = flashinfer.nvfp4_attention_sm120_fwd(
        *quantized, sm_scale=head_dim**-0.5, causal=False, return_lse=True
    )
    torch.cuda.synchronize()

    # the only remaining error is V quantization (~3%)
    assert (out.float() - 1.0).abs().max().item() <= 0.05
    # uniform scores: lse is exactly ln(seq_len)
    assert (lse.float() - math.log(seq_len)).abs().max().item() <= 0.02


@torch.inference_mode()
def test_nvfp4_attention_sm120_rejects_expanded_correction():
    """The old expanded [B, H, S, S] correction layout (which the kernel never
    addressed correctly) must be rejected by the shape validation."""
    _require_sm120()

    torch.manual_seed(42)
    batch, num_heads, seq_len, head_dim = 1, 2, 256, 128
    q = torch.randn(
        (batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    quantized = list(flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v))
    quantized[6] = quantized[6].repeat_interleave(128, dim=2).contiguous()

    with pytest.raises(ValueError, match="qk_correction"):
        flashinfer.nvfp4_attention_sm120_fwd(
            *quantized, sm_scale=head_dim**-0.5, causal=False
        )


@pytest.mark.parametrize("causal", [False, True])
@torch.inference_mode()
def test_nvfp4_attention_sm120_lse(causal):
    """lse must be the log-sum-exp of the scaled scores the kernel attends
    over (K is mean-centered by quantize_qkv, which shifts each row's lse)."""
    _require_sm120()

    torch.manual_seed(42)
    batch, num_heads, seq_len, head_dim = 2, 4, 1024, 128
    q = torch.randn(
        (batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    quantized = flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)
    _, lse = flashinfer.nvfp4_attention_sm120_fwd(
        *quantized, sm_scale=head_dim**-0.5, causal=causal, return_lse=True
    )
    torch.cuda.synchronize()

    k_centered = k.float() - k.float().mean(dim=-2, keepdim=True)
    scores = torch.matmul(q.float(), k_centered.transpose(-2, -1)) * head_dim**-0.5
    if causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1
        )
        scores.masked_fill_(mask, float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1)

    diff = (lse.float() - lse_ref).abs()
    assert diff.mean().item() <= 0.05
    assert diff.max().item() <= 0.5


@pytest.mark.parametrize("causal", [False, True])
@torch.inference_mode()
def test_nvfp4_attention_sm120_without_lse(causal):
    _require_sm120()

    torch.manual_seed(42)
    batch, num_heads, seq_len, head_dim = 1, 2, 256, 128
    q = torch.randn(
        (batch, num_heads, seq_len, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    quantized = flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out_default, lse_default = flashinfer.nvfp4_attention_sm120_fwd(
        *quantized, sm_scale=head_dim**-0.5, causal=causal
    )
    out_without_lse = flashinfer.nvfp4_attention_sm120_fwd(
        *quantized,
        sm_scale=head_dim**-0.5,
        causal=causal,
        return_lse=False,
    )
    torch.cuda.synchronize()

    assert isinstance(out_default, torch.Tensor)
    assert isinstance(lse_default, torch.Tensor)
    assert lse_default.shape == (batch, num_heads, seq_len)
    assert isinstance(out_without_lse, torch.Tensor)
    torch.testing.assert_close(out_without_lse, out_default, rtol=0, atol=5e-4)


@torch.inference_mode()
def test_nvfp4_attention_sm120_causal_mask_column_order():
    _require_sm120()

    seq_len = 128
    head_dim = 128
    q = torch.zeros((1, 1, seq_len, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.eye(seq_len, device="cuda", dtype=torch.bfloat16).reshape(
        1, 1, seq_len, head_dim
    )

    q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, qk_correction = (
        flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)
    )
    out = flashinfer.nvfp4_attention_sm120_fwd(
        q_fp4,
        k_fp4,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        qk_correction,
        sm_scale=head_dim**-0.5,
        causal=True,
        return_lse=False,
    )
    torch.cuda.synchronize()
    out = out[0, 0].float()

    ref = torch.zeros((seq_len, head_dim), device="cuda")
    for row in range(seq_len):
        ref[row, : row + 1] = 1.0 / (row + 1)

    suffix_max = torch.stack(
        [out[row, row + 1 :].abs().max() for row in range(seq_len - 1)]
    ).max()
    cos_sim = F.cosine_similarity(out.reshape(1, -1), ref.reshape(1, -1)).item()

    assert suffix_max <= 1e-5
    assert cos_sim >= 0.98


@pytest.mark.parametrize(
    (
        "seq_len_q",
        "seq_len_k",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "input_dtype",
        "out_dtype",
        "per_block_mean",
        "causal",
        "provided_out",
        "return_lse",
    ),
    [
        pytest.param(
            128,
            256,
            4,
            4,
            64,
            torch.bfloat16,
            torch.bfloat16,
            True,
            False,
            False,
            True,
            id="mha-m-lt-n-aligned-d64-bf16",
        ),
        pytest.param(
            193,
            321,
            8,
            2,
            128,
            torch.bfloat16,
            torch.float16,
            True,
            False,
            True,
            False,
            id="gqa-m-lt-n-unaligned-d128-fp16-out",
        ),
        pytest.param(
            385,
            193,
            4,
            1,
            64,
            torch.float16,
            torch.bfloat16,
            False,
            False,
            False,
            True,
            id="mqa-m-gt-n-unaligned-d64-fp16-in",
        ),
        pytest.param(
            257,
            257,
            4,
            4,
            128,
            torch.bfloat16,
            torch.bfloat16,
            True,
            True,
            True,
            True,
            id="mha-square-unaligned-d128-causal",
        ),
    ],
)
@torch.inference_mode()
def test_nvfp4_attention_sm120_rectangular(
    seq_len_q,
    seq_len_k,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    input_dtype,
    out_dtype,
    per_block_mean,
    causal,
    provided_out,
    return_lse,
):
    _require_sm120()

    torch.manual_seed(42)
    q = torch.randn(
        (1, num_qo_heads, seq_len_q, head_dim),
        device="cuda",
        dtype=input_dtype,
    )
    k = torch.randn(
        (1, num_kv_heads, seq_len_k, head_dim),
        device="cuda",
        dtype=input_dtype,
    )
    v = torch.randn_like(k)

    quantized = flashinfer.nvfp4_attention_sm120_quantize_qkv(
        q, k, v, per_block_mean=per_block_mean
    )
    q_fp4, k_fp4, v_fp4_t, q_scale, k_scale, v_scale_t, correction = quantized
    seq_len_q_pad = math.ceil(seq_len_q / 128) * 128
    seq_len_k_pad = math.ceil(seq_len_k / 128) * 128
    correction_rows = seq_len_q_pad // 128 if per_block_mean else 1
    assert q_fp4.shape == (1, num_qo_heads, seq_len_q_pad, head_dim // 2)
    assert k_fp4.shape == (1, num_kv_heads, seq_len_k_pad, head_dim // 2)
    assert v_fp4_t.shape == (1, num_kv_heads, head_dim, seq_len_k_pad // 2)
    assert q_scale.shape == (1, num_qo_heads, seq_len_q_pad, head_dim // 16)
    assert k_scale.shape == (1, num_kv_heads, seq_len_k_pad, head_dim // 16)
    assert v_scale_t.shape == (1, num_kv_heads, head_dim, seq_len_k_pad // 16)
    assert correction.shape == (1, num_qo_heads, correction_rows, seq_len_k_pad)

    out_buffer = None
    if provided_out:
        out_buffer = torch.empty(
            (1, num_qo_heads, seq_len_q_pad, head_dim),
            device="cuda",
            dtype=out_dtype,
        )
    result = flashinfer.nvfp4_attention_sm120_fwd(
        *quantized,
        per_block_mean=per_block_mean,
        out=out_buffer,
        out_dtype=out_dtype,
        causal=causal,
        return_lse=return_lse,
        unpadded_k_len=seq_len_k,
    )
    if return_lse:
        out, lse = result
        assert lse.shape == (1, num_qo_heads, seq_len_q_pad)
        assert torch.isfinite(lse[:, :, :seq_len_q]).all()
    else:
        out = result
    torch.cuda.synchronize()

    assert out.shape == (1, num_qo_heads, seq_len_q_pad, head_dim)
    assert out.dtype == out_dtype
    if provided_out:
        assert out is out_buffer

    kv_indices = torch.arange(num_qo_heads, device="cuda") // (
        num_qo_heads // num_kv_heads
    )
    k_expanded = k.index_select(1, kv_indices)
    v_expanded = v.index_select(1, kv_indices)
    ref = F.scaled_dot_product_attention(
        q.float(),
        k_expanded.float(),
        v_expanded.float(),
        is_causal=causal,
        scale=head_dim**-0.5,
    )
    out = out[:, :, :seq_len_q]
    cos_sim = F.cosine_similarity(out.float().reshape(1, -1), ref.reshape(1, -1)).item()
    mean_abs_err = (out.float() - ref).abs().mean().item()
    assert cos_sim >= 0.94
    assert mean_abs_err <= 0.10

    if return_lse:
        k_centered = k.float() - k.float().mean(dim=-2, keepdim=True)
        scores = torch.matmul(
            q.float(), k_centered.index_select(1, kv_indices).transpose(-2, -1)
        ) * (head_dim**-0.5)
        if causal:
            mask = torch.triu(
                torch.ones(
                    seq_len_q,
                    seq_len_k,
                    device="cuda",
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores.masked_fill_(mask, float("-inf"))
        lse_ref = torch.logsumexp(scores, dim=-1)
        lse_diff = (lse[:, :, :seq_len_q] - lse_ref).abs()
        assert lse_diff.mean().item() <= 0.05
        assert lse_diff.max().item() <= 0.5


@torch.inference_mode()
def test_nvfp4_attention_sm120_rectangular_matches_square_padded_oracle():
    _require_sm120()

    torch.manual_seed(42)
    seq_len_q, seq_len_k, head_dim = 192, 320, 128
    q = torch.randn((1, 4, seq_len_q, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, seq_len_k, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    rectangular = flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out_rectangular = flashinfer.nvfp4_attention_sm120_fwd(
        *rectangular, return_lse=False, unpadded_k_len=seq_len_k
    )

    q_square = F.pad(q, (0, 0, 0, seq_len_k - seq_len_q))
    square = flashinfer.nvfp4_attention_sm120_quantize_qkv(q_square, k, v)
    out_square = flashinfer.nvfp4_attention_sm120_fwd(
        *square, return_lse=False, unpadded_k_len=seq_len_k
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out_rectangular[:, :, :seq_len_q],
        out_square[:, :, :seq_len_q],
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_nvfp4_attention_sm120_unpadded_k_len_masks_tail_garbage():
    _require_sm120()

    torch.manual_seed(42)
    seq_len_q, seq_len_k, head_dim = 160, 192, 128
    q = torch.randn((1, 4, seq_len_q, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, seq_len_k, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    quantized = flashinfer.nvfp4_attention_sm120_quantize_qkv(q, k, v)

    baseline = flashinfer.nvfp4_attention_sm120_fwd(
        *quantized, return_lse=False, unpadded_k_len=seq_len_k
    )
    with_garbage = [tensor.clone() for tensor in quantized]
    with_garbage[1][:, :, seq_len_k:].fill_(0x77)
    with_garbage[2][:, :, :, seq_len_k // 2 :].fill_(0x77)
    with_garbage[4][:, :, seq_len_k:].fill_(1.0)
    with_garbage[6][..., seq_len_k:].fill_(1.0e4)
    actual = flashinfer.nvfp4_attention_sm120_fwd(
        *with_garbage, return_lse=False, unpadded_k_len=seq_len_k
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)

    for invalid_len in (0, quantized[1].shape[2] + 1):
        with pytest.raises(ValueError, match="unpadded_k_len"):
            flashinfer.nvfp4_attention_sm120_fwd(
                *quantized, return_lse=False, unpadded_k_len=invalid_len
            )


def test_nvfp4_split_kv_gate_dtype_logic():
    """The split-KV gate's dtype classification must accept NVFP4 KV and only NVFP4 KV.

    Split-KV was empirically observed to corrupt NVFP4 KV reads when a short
    query attends a long KV range (decode / prefix-cache extend), so plan()
    force-disables split-KV for NVFP4 KV on the affected architectures as a
    workaround (see _nvfp4_kv_requires_disabled_split_kv for the state of the
    root-cause analysis). FP8 / 16-bit KV are unaffected and must keep split-KV.
    This is a pure dtype-classification check (no GPU required) guarding that
    half of the contract; the architecture scoping is covered by
    test_nvfp4_split_kv_gate_arch_scope.
    """
    from flashinfer.jit.attention.utils import _is_nvfp4_kv_dtype

    # packed NVFP4 (uint8 is the run-path convention) and native fp4 -> NVFP4 KV
    assert _is_nvfp4_kv_dtype(torch.uint8)
    native_fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if native_fp4 is not None:
        assert _is_nvfp4_kv_dtype(native_fp4)

    # 16-bit and FP8 KV split-KV is fine -> never NVFP4 KV
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        assert not _is_nvfp4_kv_dtype(dtype)


def test_nvfp4_split_kv_gate_arch_scope(monkeypatch):
    """The gate must fire only on the architectures the corruption was seen on.

    SM120/121 stay gated; every other target keeps split-KV, at the low-batch
    decode saving quantified in the comment above
    _NVFP4_SPLIT_KV_BROKEN_ARCHS. The compute capability is faked, so this runs
    without a GPU.
    """
    from flashinfer import prefill

    device = torch.device("cuda:0")

    def gated_at(compute_capability):
        monkeypatch.setattr(
            prefill, "get_compute_capability", lambda _device: compute_capability
        )
        return prefill._nvfp4_kv_requires_disabled_split_kv(torch.uint8, device)

    for compute_capability in ((12, 0), (12, 1)):
        assert gated_at(compute_capability)
    for compute_capability in (
        (8, 0),
        (8, 9),
        (9, 0),
        (10, 0),
        (10, 3),
        (10, 7),
        (11, 0),
    ):
        assert not gated_at(compute_capability)

    # A non-NVFP4 KV cache is never gated, and must not cost an architecture
    # query to establish that: the dtype test comes first for every plan() call.
    def _unexpected_query(_device):
        raise AssertionError("architecture queried for a non-NVFP4 KV cache")

    monkeypatch.setattr(prefill, "get_compute_capability", _unexpected_query)
    for dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        assert not prefill._nvfp4_kv_requires_disabled_split_kv(dtype, device)
