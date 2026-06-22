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
from flashinfer.utils import is_sm120a_supported


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip("SM120 GPU is required")


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
    out, _ = flashinfer.nvfp4_attention_sm120_fwd(
        q_fp4,
        k_fp4,
        v_fp4_t,
        q_scale,
        k_scale,
        v_scale_t,
        qk_correction,
        sm_scale=head_dim**-0.5,
        causal=True,
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
