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


def _require_sm12x():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not (
        is_sm120a_supported(torch.device("cuda"))
        or is_sm121a_supported(torch.device("cuda"))
    ):
        pytest.skip("SM120/SM121 GPU is required")


def _quantize_per_tensor(x: torch.Tensor):
    """x: [tokens, heads, dim] bf16 -> (fp8 e4m3, dequant scale)."""
    amax = x.abs().amax().clamp(min=1e-6)
    scale = (amax / 448.0).item()
    q = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def _reference_ragged_attention(q8, k8, v8, qo_indptr, kv_indptr, scales, causal):
    """fp32 reference on the dequantized ragged inputs (per request).

    Returns (out [total_q, Hq, D] fp32, lse [total_q, Hq] fp32).
    """
    sm_scale, q_scale, k_scale, v_scale = scales
    batch = qo_indptr.numel() - 1
    outs, lses = [], []
    for r in range(batch):
        qs, qe = int(qo_indptr[r]), int(qo_indptr[r + 1])
        ks, ke = int(kv_indptr[r]), int(kv_indptr[r + 1])
        q_r = q8[qs:qe].float() * q_scale  # [Lq, Hq, D]
        k_r = k8[ks:ke].float() * k_scale  # [Lk, Hkv, D]
        v_r = v8[ks:ke].float() * v_scale
        lq, lk = qe - qs, ke - ks
        group = q_r.shape[1] // k_r.shape[1]
        k_r = k_r.repeat_interleave(group, dim=1)
        v_r = v_r.repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_r, k_r) * sm_scale
        if causal:
            offset = lk - lq
            rows = torch.arange(lq, device=scores.device).unsqueeze(1)
            cols = torch.arange(lk, device=scores.device).unsqueeze(0)
            scores = scores.masked_fill(cols > rows + offset, float("-inf"))
        lse_r = torch.logsumexp(scores, dim=-1)  # [Hq, Lq]
        # Replay the kernel's P path: unnormalized weights are requantized to e4m3 with
        # a fixed scale of 256 for the PV MMA, while the softmax denominator is the
        # UNQUANTIZED row sum.
        m = scores.amax(dim=-1, keepdim=True)
        p_un = torch.exp(scores - m)
        p_q = (p_un * 256.0).to(torch.float8_e4m3fn).float() / 256.0
        o_r = torch.einsum("hqk,khd->qhd", p_q, v_r)
        o_r = o_r / p_un.sum(dim=-1).transpose(0, 1).unsqueeze(-1)
        outs.append(o_r)
        lses.append(lse_r.transpose(0, 1))
    return torch.cat(outs, dim=0), torch.cat(lses, dim=0)


def _run_case(seed, lens, num_qo_heads, num_kv_heads, causal, out_dtype):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    head_dim = 128
    qo_lens = [q for q, _ in lens]
    kv_lens = [k for _, k in lens]
    qo_indptr = torch.tensor(
        [0] + list(torch.tensor(qo_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    kv_indptr = torch.tensor(
        [0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    q = torch.randn(sum(qo_lens), num_qo_heads, head_dim, device=device).to(
        torch.bfloat16
    )
    k = torch.randn(sum(kv_lens), num_kv_heads, head_dim, device=device).to(
        torch.bfloat16
    )
    v = torch.randn(sum(kv_lens), num_kv_heads, head_dim, device=device).to(
        torch.bfloat16
    )
    q8, q_scale = _quantize_per_tensor(q)
    k8, k_scale = _quantize_per_tensor(k)
    v8, v_scale = _quantize_per_tensor(v)
    sm_scale = head_dim**-0.5

    out, lse = flashinfer.mxfp8_attention_sm120_fwd(
        q8,
        k8,
        v8,
        qo_indptr,
        kv_indptr,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        causal=causal,
        out_dtype=out_dtype,
    )
    o_ref, lse_ref = _reference_ragged_attention(
        q8, k8, v8, qo_indptr, kv_indptr, (sm_scale, q_scale, k_scale, v_scale), causal
    )
    o_diff = (out.float() - o_ref).abs()
    lse_diff = (lse - lse_ref).abs()
    assert o_diff.max().item() < 4e-2, f"out max abs diff {o_diff.max().item()}"
    assert o_diff.mean().item() < 3e-3, f"out mean abs diff {o_diff.mean().item()}"
    assert lse_diff.max().item() < 1e-2, f"lse max abs diff {lse_diff.max().item()}"


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_attention_sm120_ragged_gqa_causal(out_dtype):
    _require_sm12x()
    # 16 requests: mixed lengths, several append (kv_len > qo_len) cases.
    lens = [
        (128, 128),
        (37, 37),
        (256, 256),
        (64, 300),
        (1, 1),
        (200, 512),
        (130, 130),
        (64, 64),
        (511, 511),
        (96, 190),
        (128, 1024),
        (65, 65),
        (384, 384),
        (16, 77),
        (250, 250),
        (127, 129),
    ]
    _run_case(0, lens, 8, 2, True, out_dtype)


def test_mxfp8_attention_sm120_ragged_mha_noncausal():
    _require_sm12x()
    lens = [(64, 64), (100, 130), (300, 300), (33, 61)]
    _run_case(1, lens, 4, 4, False, torch.bfloat16)


def test_mxfp8_attention_sm120_single_long_request():
    _require_sm12x()
    _run_case(2, [(2048, 2048)], 16, 4, True, torch.bfloat16)


def test_mxfp8_attention_sm120_prefix_append():
    _require_sm12x()
    # pure append: short new query chunk over a long cached prefix
    lens = [(16, 1000), (128, 900), (1, 513)]
    _run_case(3, lens, 8, 8, True, torch.bfloat16)


if __name__ == "__main__":
    test_mxfp8_attention_sm120_ragged_gqa_causal(torch.bfloat16)
    test_mxfp8_attention_sm120_ragged_mha_noncausal()
    test_mxfp8_attention_sm120_single_long_request()
    test_mxfp8_attention_sm120_prefix_append()
    print("PASS")
