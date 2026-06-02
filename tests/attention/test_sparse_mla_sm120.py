# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Correctness tests for sparse-MLA paged attention on SM120.

Covers both decode (num_tokens <= 64) and prefill (num_tokens > 64) paths
against a PyTorch SDPA-with-sparse-mask reference, plus the dual-cache
prefill variant exclusive to DSv4:

* DSv4 (d_qk=512, FP8 FOOTER 584 B/token, page_block_size=64)
    - decode-dsv4   (single-cache)
    - prefill-dsv4  (single-cache + dual-cache page-size variants)
* DSv3.2 (d_qk=576, FP8 INLINE 656 B/token, page_block_size=64)
    - decode-dsv3_2
    - prefill-dsv3_2
* GLM_NSA (d_qk=576, FP8 INLINE arbitrary FP32 scales)
    - decode + prefill through kv_scale_format="arbitrary_fp32"

Quantization helpers port the upstream FlashMLA packed layouts.

Skipped on non-SM12x GPUs via :func:`is_sm12x_supported`.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.sparse_mla_sm120 import sparse_mla_sm120_paged_attention
from flashinfer.utils import is_sm12x_supported

pytestmark = pytest.mark.skipif(
    not is_sm12x_supported(torch.device("cuda")),
    reason="Sparse-MLA SM120 requires SM12x.",
)


def _make_sparse_mla_wrapper(
    *,
    d_v: int,
    device: torch.device,
) -> flashinfer.mla.BatchMLAPagedAttentionWrapper:
    return flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(1, dtype=torch.int8, device=device),
        backend="sparse-sm120",
        d_v=d_v,
    )


# ── Quantization helpers (ported from flash_mla_sm120/tests/test_decode.py) ──


def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    """Round inverse scale to the nearest power-of-2 (FlashMLA convention)."""
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Extract the IEEE-754 exponent byte of an FP32 power-of-2 scale."""
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_kv_dsv4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSV4 FP8 FOOTER format.

    Input  shape (nb, bs, 1, 512) bf16.
    Output shape (nb, bs, 1, 584) uint8 — physical layout per block:
        [0 : bs*576)        Token data (nope 448B FP8 + rope 128B BF16) per token
        [bs*576 : bs*584)   Scale footer (7×UE8M0 + 1 pad) per token
    """
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2  # 576
    scale_bytes = num_tiles + 1  # 8
    bpt = data_stride + scale_bytes  # 584
    nb, bs, hk, d = kv_bf16.shape
    assert d == 512 and hk == 1
    kv = kv_bf16.squeeze(2)

    block_bytes = bs * bpt
    result_flat = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[..., ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        ue8m0 = _fp32_to_ue8m0_bytes(scale)

        for tok in range(bs):
            data_off = tok * data_stride + ti * tile_size
            result_flat[:, data_off : data_off + tile_size] = fp8[:, tok].view(
                torch.uint8
            )
            scale_off = bs * data_stride + tok * scale_bytes + ti
            result_flat[:, scale_off] = ue8m0[:, tok]

    rope = kv[..., d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    rope = rope.reshape(nb, bs, d_rope * 2)
    for tok in range(bs):
        rope_off = tok * data_stride + d_nope
        result_flat[:, rope_off : rope_off + d_rope * 2] = rope[:, tok]

    return result_flat.view(nb, bs, 1, bpt)


def dequantize_kv_dsv4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSV4 FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_dsv4`."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7
    data_stride = d_nope + d_rope * 2
    scale_bytes = num_tiles + 1
    bpt = data_stride + scale_bytes
    nb, bs, _, _ = packed.shape
    result = torch.zeros(nb, bs, 512, dtype=torch.bfloat16, device=packed.device)
    p = packed.view(nb, bs * bpt)

    for tok in range(bs):
        data_off = tok * data_stride
        scale_off = bs * data_stride + tok * scale_bytes
        for ti in range(num_tiles):
            fp8_off = data_off + ti * tile_size
            fp8 = p[:, fp8_off : fp8_off + tile_size].view(torch.float8_e4m3fn).float()
            ue8m0 = p[:, scale_off + ti]
            scale = torch.pow(2.0, ue8m0.float() - 127.0)
            result[:, tok, ti * tile_size : (ti + 1) * tile_size] = (
                fp8 * scale.unsqueeze(-1)
            ).to(torch.bfloat16)
        rope_off = data_off + d_nope
        rope_bytes = p[:, rope_off : rope_off + d_rope * 2].contiguous()
        result[:, tok, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nb, d_rope)

    return result.view(nb, bs, 1, 512)


# ── DSv3.2 INLINE pack (656 B/token: FP8 nope + FP32 scales + BF16 rope) ─────


def quantize_kv_dsv3_2(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv3.2 FP8 INLINE format.

    Input  shape (nb, bs, 1, 576) bf16  (d_qk = D_NOPE 512 + D_ROPE 64).
    Output shape (nb, bs, 1, 656) uint8 — per-token layout:
        [0   : 512)  FP8 e4m3 nope (4 tiles × 128 elements)
        [512 : 528)  4 × FP32 power-of-2 scale (one per 128-elem tile)
        [528 : 656)  BF16 rope (64 elements × 2B)
    """
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4  # 16
    bpt = d_nope + scale_bytes + d_rope * 2  # 656
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
    nt = nb * bs  # total token count across all blocks
    kv = kv_bf16.reshape(nt, d)

    result = torch.zeros(nt, bpt, dtype=torch.uint8, device=kv.device)

    # FP8 nope tiles + FP32 power-of-2 scales (inline, not footer).
    for ti in range(num_tiles):
        tile = kv[:, ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)  # power-of-2 FP32
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[:, ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        # FP32 scale → 4 bytes inline at offset 512 + ti*4.
        result[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4] = (
            scale.view(torch.float32).view(torch.uint8).view(nt, 4)
        )

    # BF16 rope tail.
    rope = kv[:, d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    result[:, d_nope + scale_bytes :] = rope.view(nt, d_rope * 2)
    return result.view(nb, bs, 1, bpt)


def quantize_kv_glm_nsa(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into the 656B inline layout with arbitrary FP32 scales."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4
    bpt = d_nope + scale_bytes + d_rope * 2
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
    nt = nb * bs
    kv = kv_bf16.reshape(nt, d)
    result = torch.zeros(nt, bpt, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[:, ti * tile_size : (ti + 1) * tile_size].float()
        scale = (tile.abs().amax(dim=-1).clamp(min=1e-4) / 448.0).to(torch.float32)
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[:, ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        result[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4] = (
            scale.view(torch.float32).view(torch.uint8).view(nt, 4)
        )

    rope = kv[:, d_nope:].to(torch.bfloat16).contiguous().view(torch.uint8)
    result[:, d_nope + scale_bytes :] = rope.view(nt, d_rope * 2)
    return result.view(nb, bs, 1, bpt)


def _assert_has_non_pow2_inline_scales(packed: torch.Tensor) -> None:
    scales = packed.reshape(-1, 656)[:, 512:528].contiguous().view(torch.float32)
    log2_scales = scales.float().log2()
    assert torch.any((log2_scales - log2_scales.round()).abs() > 1e-3)


def dequantize_kv_dsv3_2(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSv3.2 FP8 INLINE → bf16. Inverse of :func:`quantize_kv_dsv3_2`."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4
    nb, bs, _, _ = packed.shape
    nt = nb * bs
    p = packed.reshape(nt, -1)

    result = torch.zeros(nt, d_nope + d_rope, dtype=torch.bfloat16, device=p.device)
    for ti in range(num_tiles):
        fp8 = (
            p[:, ti * tile_size : (ti + 1) * tile_size]
            .view(torch.float8_e4m3fn)
            .float()
        )
        scale = (
            p[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4]
            .contiguous()
            .view(torch.float32)
            .squeeze(-1)
        )
        result[:, ti * tile_size : (ti + 1) * tile_size] = (
            fp8 * scale.unsqueeze(-1)
        ).to(torch.bfloat16)
    rope_bytes = p[:, d_nope + scale_bytes :].contiguous()
    result[:, d_nope:] = rope_bytes.view(torch.bfloat16).reshape(nt, d_rope)
    return result.view(nb, bs, 1, d_nope + d_rope)


# ── PyTorch SDPA-with-sparse-mask reference ───────────────────────────────────


def _ref_sparse_attn(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense SDPA over the sparse-gathered KV. Returns (output_bf16, lse_log2).

    Honors ``attn_sink`` (FlashMLA V4 sink-merge convention) and
    ``topk_length`` (per-token valid-length mask).
    """
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    kv_flat = kv_dequant.view(-1, d_qk).float()
    q_f = q.float()

    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0
    if topk_length is not None:
        # Mark tokens beyond per-token length as invalid.
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (ar >= topk_length.unsqueeze(-1))

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(num_tokens, topk, d_qk)
    # logits: [num_tokens, num_heads, topk] = q @ K^T per (t, h)
    P = torch.einsum("thd,tkd->thk", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    lse_e = torch.logsumexp(P, dim=-1)  # natural-log LSE [t, h]
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :d_v])

    # Convert lse to log2 to match the kernel's epilogue convention.
    LN2 = float(torch.log(torch.tensor(2.0)).item())
    lse_log2 = lse_e / LN2

    if attn_sink is not None:
        # FlashMLA V4 per-head sink: output[t,h,:] *= sigmoid(lse_e[t,h] - sink[h]).
        sink = attn_sink.float()  # [num_heads]
        sink_log2 = sink / LN2  # [num_heads]
        factor = torch.sigmoid(lse_e.float() - sink.unsqueeze(0))  # [t, h]
        out_f = out_f * factor.unsqueeze(-1)  # broadcast over d_v
        # Merge sink into lse (in log2 space). Handle padded -inf head sinks.
        lse_log2 = torch.where(
            lse_log2 == float("-inf"),
            sink_log2.unsqueeze(0).expand_as(lse_log2),
            lse_log2 + torch.log2(1.0 + torch.exp2(sink_log2.unsqueeze(0) - lse_log2)),
        )

    return out_f.to(torch.bfloat16), lse_log2


def _make_decode_scratch(
    num_tokens: int,
    num_heads: int,
    topk: int,
    d_v: int,
    device: torch.device,
    *,
    extra_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_splits = (topk + 63) // 64 + (extra_topk + 63) // 64
    return (
        torch.empty(
            (num_tokens, num_heads, num_splits, d_v),
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.empty(
            (num_tokens, num_heads, num_splits),
            dtype=torch.float32,
            device=device,
        ),
    )


# ── Tests ────────────────────────────────────────────────────────────────────

_DSV4_DECODE_CONFIGS = [
    # (num_heads, topk)
    # h=8 cases exercise the VALID_HPB < HPB code path (small-TP corner);
    # cover all three topk values to confirm the dispatch table.
    (8, 128),
    (8, 512),
    (8, 1024),
    (16, 128),
    (32, 512),
    (64, 1024),
    (128, 1024),
]


@pytest.mark.parametrize("num_heads,topk", _DSV4_DECODE_CONFIGS)
@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_decode_dsv4(
    num_heads: int, topk: int, num_tokens: int, with_sink: bool
) -> None:
    """DSV4 decode-dsv3_2 path: num_tokens <= 64, d_qk=512, page_block_size=64."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size  # 4096

    # bf16 reference KV → FP8-packed kernel KV.
    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Mark half the slots invalid (-1); ensures the kernel actually masks the
    # sentinel rather than silently passing because the other logits dominate.
    indices[:, topk // 2 :] = -1

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    # Reference (uses dequantized kv).
    ref_out, ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, attn_sink=attn_sink
    )

    # Kernel: allocate output, call paged_attention.
    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)

    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    # FP8 KV + BF16 output tolerance. Matches the repo convention for FP8
    # attention tests (see tests/attention/test_xqa_mla_batch_decode.py).
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_decode_dsv4_topk_length_truncation() -> None:
    """Kernel must honor topk_length even when indices past the length point at
    valid slots — passing garbage past topk_length should not corrupt the answer.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, topk, num_tokens = 32, 512, 16
    topk_len = 128  # truncate to first 128 columns
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Positions past topk_len intentionally point at valid slots (random in
    # [0, s_kv)) to ensure the kernel actually uses topk_length to truncate.
    topk_length = torch.full((num_tokens,), topk_len, dtype=torch.int32, device=device)

    sm_scale = d_qk**-0.5

    # Reference: mask positions past topk_length to -1, pass without topk_length.
    ref_indices = indices.clone()
    ref_indices[:, topk_len:] = -1
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, ref_indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)
    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        topk_length=topk_length,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_decode_dsv4_routes_through_mla_functional_api() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads, topk = 16, 32, 128
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 32
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    topk_lens = torch.full((num_tokens, 1), topk, device=device, dtype=torch.int32)
    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    out, out_lse = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        q.unsqueeze(1),
        kv_packed,
        torch.empty(1, dtype=torch.int8, device=device),
        qk_nope_head_dim=512,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        block_tables=torch.empty((num_tokens, 0), dtype=torch.int32, device=device),
        seq_lens=torch.ones(num_tokens, dtype=torch.int32, device=device),
        max_seq_len=s_kv,
        sparse_mla_top_k=topk,
        sparse_mla_indices=indices.unsqueeze(1),
        sparse_mla_top_k_lens=topk_lens,
        bmm1_scale=sm_scale,
        backend="auto",
        return_lse=True,
    )

    torch.testing.assert_close(out.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_decode_dsv4_dual_more_than_32_splits() -> None:
    """Dual-cache decode must handle split counts above the old merge bound."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads = 1, 16
    topk, extra_topk = 128, 2176  # 2 + 34 = 36 splits.
    d_qk, d_v = 512, 512
    main_pbs, extra_pbs = 64, 2
    main_num_blocks = 16
    extra_num_blocks = (extra_topk + extra_pbs - 1) // extra_pbs
    main_s_kv = main_num_blocks * main_pbs
    extra_s_kv = extra_num_blocks * extra_pbs

    main_bf16 = (
        torch.randn(
            main_num_blocks, main_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    extra_bf16 = (
        torch.randn(
            extra_num_blocks, extra_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    main_packed = quantize_kv_dsv4(main_bf16)
    extra_packed = quantize_kv_dsv4(extra_bf16)
    main_dequant = dequantize_kv_dsv4(main_packed)
    extra_dequant = dequantize_kv_dsv4(extra_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    main_idx = torch.randint(
        0, main_s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    extra_idx = torch.randint(
        0, extra_s_kv, (num_tokens, extra_topk), device=device, dtype=torch.int32
    )

    sm_scale = d_qk**-0.5
    virtual_kv = torch.cat(
        [main_dequant.reshape(-1, d_qk), extra_dequant.reshape(-1, d_qk)], dim=0
    ).reshape(-1, 1, 1, d_qk)
    virtual_idx = torch.cat(
        [main_idx, torch.where(extra_idx < 0, extra_idx, extra_idx + main_s_kv)], dim=-1
    )
    ref_out, ref_lse = _ref_sparse_attn(q, virtual_kv, virtual_idx, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    wrapper = _make_sparse_mla_wrapper(
        d_v=d_v,
        device=device,
    )
    out_lse = wrapper.run_sparse_mla(
        q,
        main_packed,
        main_idx,
        output,
        sm_scale=sm_scale,
        extra_kv_cache=extra_packed,
        extra_sparse_indices=extra_idx,
        return_lse=True,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


_DSV3_2_DECODE_HEADS = [8, 16, 32, 64, 128]


@pytest.mark.parametrize("num_heads", _DSV3_2_DECODE_HEADS)
@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_decode_dsv3_2(
    num_heads: int, num_tokens: int, with_sink: bool
) -> None:
    """DSv3.2 decode-dsv3_2 path: d_qk=576, topk=2048, page_block_size=64."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    topk = 2048  # the only dispatched topk for decode-dsv3_2
    page_block_size = 64
    # Pool sized so topk valid slot ids fit; 64 blocks * 64 slots = 4096 slots.
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv3_2(kv_bf16)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Mark half the slots invalid (-1); ensures the kernel actually masks the
    # sentinel rather than silently passing because the other logits dominate.
    indices[:, topk // 2 :] = -1

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    ref_out, ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, attn_sink=attn_sink
    )

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)

    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_decode_dsv3_2_autotune_route() -> None:
    """DSv3.2 decode participates in AutoTuner through the public API."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    num_tokens, num_heads, topk = 1, 8, 128
    page_block_size = 64
    num_blocks = 4
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv3_2(kv_bf16)
    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    output = torch.empty(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)

    tuner = AutoTuner.get()
    tuner.clear_cache()
    with autotune(True, tuning_buckets=(num_tokens,)):
        sparse_mla_sm120_paged_attention(
            q,
            kv_packed,
            indices,
            output,
            out_lse,
            d_qk**-0.5,
            d_v=d_v,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )
    assert any(
        cache_key[0] == "sparse_mla_sm120_decode_dsv3_2"
        for cache_key in tuner.profiling_cache
    )
    tuner.clear_cache()


_DSV3_2_PREFILL_HEADS = [8, 16, 32, 64, 128]


@pytest.mark.parametrize("num_heads", _DSV3_2_PREFILL_HEADS)
@pytest.mark.parametrize("num_tokens", [128, 256])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_prefill_dsv3_2(
    num_heads: int, num_tokens: int, with_sink: bool
) -> None:
    """DSv3.2 prefill path: d_qk=576, topk=2048, page_block_size=64, T>64.

    Covers the SG kernel (NH=8 small-TP + NH=16) and the MG kernel
    (NH=32/64/128). num_tokens>64 routes through the prefill orchestrator;
    the kernel iterates all NI=TOPK/BI tiles per token and writes BF16
    output directly (no split-K, no merge).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    topk = 2048
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size  # 4096 slots

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv3_2(kv_bf16)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Mark half the slots invalid (-1); matches the decode-test masking
    # convention and ensures the prefill kernel can't pass by ignoring -1.
    indices[:, topk // 2 :] = -1

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    ref_out, ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, attn_sink=attn_sink
    )

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)

    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_decode_glm_nsa_arbitrary_fp32() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    num_tokens, num_heads, topk = 16, 16, 512
    page_block_size = 64
    num_blocks = 16
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_glm_nsa(kv_bf16)
    _assert_has_non_pow2_inline_scales(kv_packed)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    indices[:, topk // 2 :] = -1
    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)

    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        kv_scale_format="arbitrary_fp32",
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("num_heads", [8, 32])
def test_sparse_mla_sm120_prefill_glm_nsa_arbitrary_fp32(num_heads: int) -> None:
    torch.manual_seed(2)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    num_tokens, topk = 128, 2048
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_glm_nsa(kv_bf16)
    _assert_has_non_pow2_inline_scales(kv_packed)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    indices[:, topk // 2 :] = -1
    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)

    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        kv_scale_format="arbitrary_fp32",
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


_DSV4_PREFILL_CONFIGS = [
    # (num_heads, topk). DSv4 prefill envelope: NH ∈ {16, 32, 64, 128},
    # topk ∈ {128, 512, 1024}. NH=8 is not in the DSv4 prefill dispatch.
    (16, 128),
    (32, 512),
    (64, 1024),
    (128, 1024),
]


@pytest.mark.parametrize("num_heads,topk", _DSV4_PREFILL_CONFIGS)
@pytest.mark.parametrize("num_tokens", [128, 256])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_prefill_dsv4(
    num_heads: int, topk: int, num_tokens: int, with_sink: bool
) -> None:
    """DSv4 prefill (single-cache) path: d_qk=512, page_block_size=64, T>64.

    NH=16 dispatches through the SG kernel; NH=32/64/128 through the MG
    kernel. Dual-cache (extra_kv_cache) variants are not exercised here.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size  # 4096 slots

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Mark half the slots invalid (-1); same convention as the decode tests.
    indices[:, topk // 2 :] = -1

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    ref_out, ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, attn_sink=attn_sink
    )

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)

    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


_DSV4_PREFILL_DUAL_CONFIGS = [
    # (extra_topk, extra_pbs). Main is fixed at (topk=128, pbs=64).
    (128, 64),
    (512, 64),
    (512, 2),
]

_DSV4_PREFILL_DUAL_HEADS = [16, 32, 64, 128]


@pytest.mark.parametrize("num_heads", _DSV4_PREFILL_DUAL_HEADS)
@pytest.mark.parametrize("extra_topk,extra_pbs", _DSV4_PREFILL_DUAL_CONFIGS)
def test_sparse_mla_sm120_prefill_dsv4_dual(
    num_heads: int, extra_topk: int, extra_pbs: int
) -> None:
    """DSv4 prefill (dual-cache) path: main + extra KV with disjoint slot pools.

    Main cache: topk=128, pbs=64. Extra cache parameters cover both secondary
    cache page sizes. NH ∈ {16, 32, 64, 128} covers the MG dual-cache dispatch.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    topk = 128
    main_pbs = 64
    num_tokens = 128

    # Size both pools so all topk slot ids fit.
    main_num_blocks = 64
    main_s_kv = main_num_blocks * main_pbs  # 4096
    # Round extra block count up to fit extra_topk slots.
    extra_num_blocks = max((extra_topk + extra_pbs - 1) // extra_pbs * 2, 16)
    extra_s_kv = extra_num_blocks * extra_pbs

    # Main cache.
    main_bf16 = (
        torch.randn(
            main_num_blocks, main_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    main_packed = quantize_kv_dsv4(main_bf16)
    main_dequant = dequantize_kv_dsv4(main_packed)

    # Extra cache (independent quantization noise + slot pool).
    extra_bf16 = (
        torch.randn(
            extra_num_blocks, extra_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    extra_packed = quantize_kv_dsv4(extra_bf16)
    extra_dequant = dequantize_kv_dsv4(extra_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    main_idx = torch.randint(
        0, main_s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    extra_idx = torch.randint(
        0, extra_s_kv, (num_tokens, extra_topk), device=device, dtype=torch.int32
    )
    # Mark half of each cache's slots invalid (-1).
    main_idx[:, topk // 2 :] = -1
    extra_idx[:, extra_topk // 2 :] = -1

    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0

    sm_scale = d_qk**-0.5

    # Reference: treat dual cache as one virtual pool. Main occupies slots
    # [0, main_s_kv); extra occupies [main_s_kv, main_s_kv + extra_s_kv).
    # Shift extra indices by main_s_kv so the unified pool sees disjoint
    # index spaces. The kernel's running-softmax across both caches matches
    # dense softmax over the union.
    virtual_kv = torch.cat(
        [main_dequant.reshape(-1, d_qk), extra_dequant.reshape(-1, d_qk)], dim=0
    ).reshape(-1, 1, 1, d_qk)
    extra_idx_shifted = torch.where(extra_idx < 0, extra_idx, extra_idx + main_s_kv)
    virtual_idx = torch.cat([main_idx, extra_idx_shifted], dim=-1)

    ref_out, ref_lse = _ref_sparse_attn(
        q, virtual_kv, virtual_idx, sm_scale, d_v, attn_sink=attn_sink
    )

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)

    sparse_mla_sm120_paged_attention(
        q,
        main_packed,
        main_idx,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        extra_kv_cache=extra_packed,
        extra_indices=extra_idx,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_prefill_dsv4_dual_accepts_singleton_s_q_indices() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, num_tokens = 32, 128
    d_qk, d_v = 512, 512
    topk, extra_topk = 128, 128
    main_pbs, extra_pbs = 64, 64

    main_num_blocks, extra_num_blocks = 64, 64
    main_s_kv = main_num_blocks * main_pbs
    extra_s_kv = extra_num_blocks * extra_pbs

    main_bf16 = (
        torch.randn(
            main_num_blocks, main_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    extra_bf16 = (
        torch.randn(
            extra_num_blocks, extra_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    main_packed = quantize_kv_dsv4(main_bf16)
    extra_packed = quantize_kv_dsv4(extra_bf16)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    main_idx = torch.randint(
        0, main_s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    extra_idx = torch.randint(
        0, extra_s_kv, (num_tokens, extra_topk), device=device, dtype=torch.int32
    )
    main_idx[:, topk // 2 :] = -1
    extra_idx[:, extra_topk // 2 :] = -1
    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
    sm_scale = d_qk**-0.5

    def run(
        indices: torch.Tensor, extra_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.zeros(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
        )
        out_lse = torch.zeros(
            (num_tokens, num_heads), dtype=torch.float32, device=device
        )
        sparse_mla_sm120_paged_attention(
            q,
            main_packed,
            indices,
            output,
            out_lse,
            sm_scale,
            d_v=d_v,
            attn_sink=attn_sink,
            extra_kv_cache=extra_packed,
            extra_indices=extra_indices,
        )
        return output, out_lse

    out_2d, lse_2d = run(main_idx, extra_idx)
    out_3d, lse_3d = run(main_idx.unsqueeze(1), extra_idx.unsqueeze(1))

    torch.testing.assert_close(out_3d, out_2d, atol=0, rtol=0)
    torch.testing.assert_close(lse_3d, lse_2d, atol=0, rtol=0)


@pytest.mark.parametrize("extra_topk_len", [0, 128, 768])
def test_sparse_mla_sm120_prefill_dsv4_dual_extra_topk_length_truncation(
    extra_topk_len: int,
) -> None:
    """Dual-cache: extra_topk_length truncates the secondary window. Past-length
    extra_indices intentionally point at valid slots to verify the kernel masks
    them out via extra_topk_length. Over-declared lengths clamp to extra_topk.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, num_tokens = 64, 128
    d_qk, d_v = 512, 512
    topk = 128
    main_pbs = 64
    extra_topk = 512
    extra_pbs = 64

    main_num_blocks = 64
    main_s_kv = main_num_blocks * main_pbs
    extra_num_blocks = max((extra_topk + extra_pbs - 1) // extra_pbs * 2, 16)
    extra_s_kv = extra_num_blocks * extra_pbs

    main_bf16 = (
        torch.randn(
            main_num_blocks, main_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    main_packed = quantize_kv_dsv4(main_bf16)
    main_dequant = dequantize_kv_dsv4(main_packed)

    extra_bf16 = (
        torch.randn(
            extra_num_blocks, extra_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    extra_packed = quantize_kv_dsv4(extra_bf16)
    extra_dequant = dequantize_kv_dsv4(extra_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    main_idx = torch.randint(
        0, main_s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    extra_idx = torch.randint(
        0, extra_s_kv, (num_tokens, extra_topk), device=device, dtype=torch.int32
    )
    # Past-length entries point at valid slots; kernel must mask via length.
    extra_topk_length = torch.full(
        (num_tokens,), extra_topk_len, dtype=torch.int32, device=device
    )

    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
    sm_scale = d_qk**-0.5

    # Reference: mask extra entries past length to -1, then build the unified
    # virtual pool / virtual_idx.
    ref_extra_idx = extra_idx.clone()
    extra_topk_len_clamped = min(max(extra_topk_len, 0), extra_topk)
    ref_extra_idx[:, extra_topk_len_clamped:] = -1
    virtual_kv = torch.cat(
        [main_dequant.reshape(-1, d_qk), extra_dequant.reshape(-1, d_qk)], dim=0
    ).reshape(-1, 1, 1, d_qk)
    extra_idx_shifted = torch.where(
        ref_extra_idx < 0, ref_extra_idx, ref_extra_idx + main_s_kv
    )
    virtual_idx = torch.cat([main_idx, extra_idx_shifted], dim=-1)
    ref_out, ref_lse = _ref_sparse_attn(
        q, virtual_kv, virtual_idx, sm_scale, d_v, attn_sink=attn_sink
    )

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    sparse_mla_sm120_paged_attention(
        q,
        main_packed,
        main_idx,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        extra_kv_cache=extra_packed,
        extra_indices=extra_idx,
        extra_topk_length=extra_topk_length,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_prefill_dsv4_dual_zero_main_topk() -> None:
    """When main topk_length is zero, main indices past the runtime length may be
    uninitialized by callers and must not be read by the dual-cache prefill path.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, num_tokens = 32, 128
    d_qk, d_v = 512, 512
    topk = 128
    main_pbs = 64
    extra_topk = 128
    extra_pbs = 2

    main_num_blocks = 4
    main_s_kv = main_num_blocks * main_pbs
    extra_num_blocks = (extra_topk + extra_pbs - 1) // extra_pbs + 8
    extra_s_kv = extra_num_blocks * extra_pbs

    main_bf16 = (
        torch.randn(
            main_num_blocks, main_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    main_packed = quantize_kv_dsv4(main_bf16)
    main_dequant = dequantize_kv_dsv4(main_packed)

    extra_bf16 = (
        torch.randn(
            extra_num_blocks, extra_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    extra_packed = quantize_kv_dsv4(extra_bf16)
    extra_dequant = dequantize_kv_dsv4(extra_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    main_idx = torch.full(
        (num_tokens, topk), main_s_kv + 1_000_000, device=device, dtype=torch.int32
    )
    extra_idx = torch.randint(
        0, extra_s_kv, (num_tokens, extra_topk), device=device, dtype=torch.int32
    )
    topk_length = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    extra_topk_length = torch.full(
        (num_tokens,), extra_topk, dtype=torch.int32, device=device
    )
    sm_scale = d_qk**-0.5

    virtual_kv = torch.cat(
        [main_dequant.reshape(-1, d_qk), extra_dequant.reshape(-1, d_qk)], dim=0
    ).reshape(-1, 1, 1, d_qk)
    main_idx_ref = torch.full_like(main_idx, -1)
    extra_idx_shifted = extra_idx + main_s_kv
    virtual_idx = torch.cat([main_idx_ref, extra_idx_shifted], dim=-1)
    ref_out, ref_lse = _ref_sparse_attn(q, virtual_kv, virtual_idx, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    sparse_mla_sm120_paged_attention(
        q,
        main_packed,
        main_idx,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        topk_length=topk_length,
        extra_kv_cache=extra_packed,
        extra_indices=extra_idx,
        extra_topk_length=extra_topk_length,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_prefill_dsv3_2_sg_zero_topk_length() -> None:
    """SG prefill must not prefetch tile 0 when runtime topk_length is zero."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads = 128, 8
    d_qk, d_v = 576, 512
    topk = 2048

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    kv_cache = torch.empty((1, 64, 1, 656), dtype=torch.uint8, device=device)
    indices = torch.full(
        (num_tokens, topk), 1_000_000, dtype=torch.int32, device=device
    )
    topk_length = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    output = torch.empty(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)

    sparse_mla_sm120_paged_attention(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        d_qk**-0.5,
        d_v=d_v,
        topk_length=topk_length,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, torch.zeros_like(output))
    torch.testing.assert_close(out_lse, torch.full_like(out_lse, -1e30))


# Runtime topk_extra should not require a dedicated template instantiation.
@pytest.mark.parametrize("extra_topk,extra_pbs", [(1024, 2), (1664, 2), (1024, 64)])
def test_sparse_mla_sm120_prefill_dsv4_dual_runtime_extra_topk(
    extra_topk: int, extra_pbs: int
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, num_tokens = 64, 128
    d_qk, d_v = 512, 512
    topk = 128
    main_pbs = 64

    main_num_blocks = 64
    main_s_kv = main_num_blocks * main_pbs
    extra_num_blocks = max((extra_topk + extra_pbs - 1) // extra_pbs * 2, 16)
    extra_s_kv = extra_num_blocks * extra_pbs

    main_bf16 = (
        torch.randn(
            main_num_blocks, main_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    main_packed = quantize_kv_dsv4(main_bf16)
    main_dequant = dequantize_kv_dsv4(main_packed)

    extra_bf16 = (
        torch.randn(
            extra_num_blocks, extra_pbs, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    extra_packed = quantize_kv_dsv4(extra_bf16)
    extra_dequant = dequantize_kv_dsv4(extra_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    main_idx = torch.randint(
        0, main_s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    extra_idx = torch.randint(
        0, extra_s_kv, (num_tokens, extra_topk), device=device, dtype=torch.int32
    )
    main_idx[:, topk // 2 :] = -1
    extra_idx[:, extra_topk // 2 :] = -1

    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
    sm_scale = d_qk**-0.5

    virtual_kv = torch.cat(
        [main_dequant.reshape(-1, d_qk), extra_dequant.reshape(-1, d_qk)], dim=0
    ).reshape(-1, 1, 1, d_qk)
    extra_idx_shifted = torch.where(extra_idx < 0, extra_idx, extra_idx + main_s_kv)
    virtual_idx = torch.cat([main_idx, extra_idx_shifted], dim=-1)
    ref_out, ref_lse = _ref_sparse_attn(
        q, virtual_kv, virtual_idx, sm_scale, d_v, attn_sink=attn_sink
    )

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    sparse_mla_sm120_paged_attention(
        q,
        main_packed,
        main_idx,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        extra_kv_cache=extra_packed,
        extra_indices=extra_idx,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("model", ["dsv4", "dsv3_2"])
def test_sparse_mla_sm120_wrapper_class_run(model: str) -> None:
    """Smoke-test the wrapper class: construct once, call .run() across
    decode (T ∈ {1, 16, 64}) and prefill (T = 128) shapes for both V32 and
    DSv4 envelopes."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    page_block_size = 64
    num_blocks = 32

    if model == "dsv4":
        num_heads, topk = 32, 512
        d_qk, d_v = 512, 512
        quantize = quantize_kv_dsv4
    else:
        num_heads, topk = 32, 2048
        d_qk, d_v = 576, 512
        quantize = quantize_kv_dsv3_2

    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize(kv_bf16)

    wrapper = _make_sparse_mla_wrapper(
        d_v=d_v,
        device=device,
    )

    # 1, 16, 64 -> decode kernel; 128 -> prefill orchestrator.
    for num_tokens in (1, 16, 64, 128):
        q = (
            torch.randn(
                num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16
            )
            / 10.0
        ).clamp(-1, 1)
        indices = torch.randint(
            0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
        )
        output = torch.zeros(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
        )
        mid_out = None
        mid_lse = None
        if num_tokens <= 64:
            mid_out, mid_lse = _make_decode_scratch(
                num_tokens, num_heads, topk, d_v, device
            )
        # No exception means the dispatch path is wired correctly.
        wrapper.run_sparse_mla(
            q,
            kv_packed,
            indices,
            output,
            sm_scale=d_qk**-0.5,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )
