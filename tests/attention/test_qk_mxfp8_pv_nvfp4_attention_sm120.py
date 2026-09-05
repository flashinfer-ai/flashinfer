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

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.utils import is_sm120a_supported, is_sm121a_supported


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if not (is_sm120a_supported(device) or is_sm121a_supported(device)):
        pytest.skip("SM120 or SM121 GPU is required")


def _reference_attention(q, k, v, causal, sm_scale):
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    # Match the public preprocessing recipe. K centering leaves the softmax
    # probabilities unchanged but changes the LSE by a row-wise constant.
    k = k - k.mean(dim=-2, keepdim=True)
    if num_qo_heads != num_kv_heads:
        group_size = num_qo_heads // num_kv_heads
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        qo_len, kv_len = q.shape[2], k.shape[2]
        qo_idx = torch.arange(qo_len, device=q.device)[:, None]
        kv_idx = torch.arange(kv_len, device=q.device)[None, :]
        scores.masked_fill_(kv_idx > qo_idx + kv_len - qo_len, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v.float()), torch.logsumexp(scores, dim=-1)


def _logical_ue8m0_codes(scale, num_tokens):
    """Undo the SM120 scale-factor byte layout for D128 MXFP8 tensors."""
    token = torch.arange(num_tokens, device=scale.device)
    token_local = token % 128
    base = (token // 128) * 512 + (token_local % 32) * 16 + (token_local // 32) * 4
    flat = scale.flatten(start_dim=2)
    return torch.stack([flat[:, :, base + index] for index in range(4)], dim=-1)


def _expected_mxfp8(x):
    blocks = x.float().reshape(*x.shape[:-1], 4, 32)
    amax = blocks.abs().amax(dim=-1)
    normalized = amax / 448.0
    codes = torch.where(
        amax == 0,
        torch.zeros_like(amax, dtype=torch.int32),
        torch.ceil(torch.log2(normalized)).to(torch.int32) + 127,
    )
    scales = torch.where(
        codes == 0,
        torch.zeros_like(amax),
        torch.exp2(codes.float() - 127),
    )
    normalized_blocks = torch.where(
        scales[..., None] == 0,
        torch.zeros_like(blocks),
        blocks / scales[..., None],
    )
    quantized = normalized_blocks.to(torch.float8_e4m3fn).float().reshape_as(x.float())
    return quantized, codes.to(torch.uint8)


def _k_physical_to_logical_tokens(num_tokens, device):
    physical = torch.arange(num_tokens, device=device)
    local = physical % 128
    residue = local % 32
    logical_local = (
        (local // 32) * 32 + (residue // 8) * 2 + ((residue % 8) // 2) * 8 + residue % 2
    )
    return (physical // 128) * 128 + logical_local


@pytest.mark.parametrize(
    (
        "num_qo_heads",
        "num_kv_heads",
        "qo_len",
        "kv_len",
        "causal",
        "out_dtype",
    ),
    [
        pytest.param(4, 4, 128, 128, False, torch.bfloat16, id="mha-noncausal"),
        pytest.param(4, 4, 256, 256, True, torch.float16, id="mha-causal-fp16"),
        pytest.param(8, 2, 193, 317, True, torch.bfloat16, id="gqa-causal-tails"),
        pytest.param(8, 1, 257, 193, False, torch.bfloat16, id="mqa-rectangular"),
    ],
)
@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_accuracy(
    num_qo_heads,
    num_kv_heads,
    qo_len,
    kv_len,
    causal,
    out_dtype,
):
    _require_sm120()
    torch.manual_seed(42)
    q = torch.randn((1, num_qo_heads, qo_len, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, num_kv_heads, kv_len, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sm_scale = 128**-0.5

    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out, lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized,
        sm_scale=sm_scale,
        causal=causal,
        out_dtype=out_dtype,
        return_lse=True,
        unpadded_q_len=qo_len,
        unpadded_k_len=kv_len,
    )
    ref, ref_lse = _reference_attention(q, k, v, causal, sm_scale)
    assert out.dtype == out_dtype
    out = out[:, :, :qo_len].float()
    lse = lse[:, :, :qo_len]

    assert lse.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.isfinite(lse).all()
    cosine = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    mean_abs_error = (out - ref).abs().mean().item()
    relative_l2 = torch.linalg.vector_norm(out - ref) / torch.linalg.vector_norm(ref)
    max_abs_error = (out - ref).abs().max().item()
    lse_max_error = (lse - ref_lse).abs().max().item()
    assert cosine >= 0.985
    assert mean_abs_error <= 0.07
    assert relative_l2.item() <= 0.20
    assert max_abs_error <= 0.75
    assert lse_max_error <= 0.15


@pytest.mark.parametrize("causal", [False, True])
@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_lse_specialization(causal):
    _require_sm120()
    torch.manual_seed(7)
    q = torch.randn((1, 8, 256, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)

    out_only = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized, causal=causal
    )
    out_with_lse, lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized, causal=causal, return_lse=True
    )

    assert isinstance(out_only, torch.Tensor)
    assert lse.shape == (1, 8, 256)
    torch.testing.assert_close(out_only, out_with_lse, rtol=0, atol=4e-3)


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_caller_buffers_and_scale_alias():
    _require_sm120()
    torch.manual_seed(11)
    q = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    out = torch.empty_like(q)
    lse = torch.empty((1, 4, 128), device="cuda", dtype=torch.float32)

    actual_out, actual_lse = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
        *quantized,
        softmax_scale=0.1,
        out=out,
        lse=lse,
        return_lse=True,
    )

    assert actual_out.data_ptr() == out.data_ptr()
    assert actual_lse.data_ptr() == lse.data_ptr()
    assert actual_out.dtype == torch.float16
    assert torch.isfinite(actual_out).all()
    assert torch.isfinite(actual_lse).all()


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_quantized_shapes():
    _require_sm120()
    q = torch.randn((2, 8, 129, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((2, 2, 257, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t = (
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    )

    assert q_fp8.shape == (2, 8, 256, 128)
    assert k_fp8.shape == (2, 2, 384, 128)
    assert v_fp4_t.shape == (2, 2, 128, 192)
    assert q_scale.shape == (2, 8, 256, 4)
    assert k_scale.shape == (2, 2, 384, 4)
    assert v_scale_t.shape == (2, 2, 128, 24)
    assert q_fp8.dtype == torch.float8_e4m3fn
    assert k_fp8.dtype == torch.float8_e4m3fn
    assert v_fp4_t.dtype == torch.uint8
    assert q_scale.dtype == torch.uint8
    assert k_scale.dtype == torch.uint8
    assert v_scale_t.dtype == torch.float8_e4m3fn


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_mxfp8_recipe():
    _require_sm120()
    torch.manual_seed(19)
    q = torch.randn((1, 2, 129, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 1, 257, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    q_fp8, k_fp8, _, q_scale, k_scale, _ = (
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    )
    q_padded = F.pad(q, (0, 0, 0, 127))
    k_centered = k - k.mean(dim=-2, keepdim=True)
    k_padded = F.pad(k_centered, (0, 0, 0, 127))
    expected_q, expected_q_codes = _expected_mxfp8(q_padded)
    expected_k, expected_k_codes = _expected_mxfp8(k_padded)

    torch.testing.assert_close(q_fp8.float(), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(
        _logical_ue8m0_codes(q_scale, q_fp8.shape[2]),
        expected_q_codes,
        rtol=0,
        atol=0,
    )

    physical_to_logical = _k_physical_to_logical_tokens(k_fp8.shape[2], k.device)
    torch.testing.assert_close(
        k_fp8.float(), expected_k[:, :, physical_to_logical], rtol=0, atol=0
    )
    torch.testing.assert_close(
        _logical_ue8m0_codes(k_scale, k_fp8.shape[2]),
        expected_k_codes[:, :, physical_to_logical],
        rtol=0,
        atol=0,
    )

    assert torch.count_nonzero(q_fp8[:, :, q.shape[2] :]) == 0
    assert torch.count_nonzero(k_fp8[:, :, k.shape[2] :]) == 0


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_rejects_invalid_lengths():
    _require_sm120()
    q = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)

    with pytest.raises(ValueError, match="unpadded_q_len"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(*quantized, unpadded_q_len=129)
    with pytest.raises(ValueError, match="unpadded_k_len"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(*quantized, unpadded_k_len=0)
    with pytest.raises(ValueError, match="Specify only one"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            *quantized, sm_scale=0.1, softmax_scale=0.1
        )

    empty = torch.empty((1, 4, 0, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="sequence lengths must be positive"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(empty, empty, empty)


@torch.inference_mode()
def test_qk_mxfp8_pv_nvfp4_attention_sm120_rejects_invalid_inputs():
    _require_sm120()
    q = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    with pytest.raises(ValueError, match="q must have dtype"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q.float(), k, v)
    with pytest.raises(ValueError, match="q must be contiguous"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(
            q.transpose(-1, -2), k, v
        )
    with pytest.raises(ValueError, match="num_qo_heads"):
        invalid_k = torch.randn((1, 3, 128, 128), device="cuda", dtype=torch.bfloat16)
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(
            q, invalid_k, invalid_k
        )
    with pytest.raises(ValueError, match="head_dim must be 128"):
        q64 = q[..., :64].contiguous()
        k64 = k[..., :64].contiguous()
        v64 = v[..., :64].contiguous()
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q64, k64, v64)

    quantized = flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_quantize_qkv(q, k, v)
    q_fp8, k_fp8, v_fp4_t, q_scale, k_scale, v_scale_t = quantized
    with pytest.raises(ValueError, match="q_scale shape"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            q_fp8,
            k_fp8,
            v_fp4_t,
            q_scale[..., :-1].contiguous(),
            k_scale,
            v_scale_t,
        )
    with pytest.raises(ValueError, match="q_fp8 and k_fp8"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            q_fp8.view(torch.uint8),
            k_fp8,
            v_fp4_t,
            q_scale,
            k_scale,
            v_scale_t,
        )
    with pytest.raises(ValueError, match="out shape"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            *quantized,
            out=torch.empty((1, 4, 127, 128), device="cuda", dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError, match="return_lse=True"):
        flashinfer.qk_mxfp8_pv_nvfp4_attention_sm120_fwd(
            *quantized,
            lse=torch.empty((1, 4, 128), device="cuda", dtype=torch.float32),
        )
