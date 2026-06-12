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

"""Correctness tests for sparse-MLA paged attention on SM120."""

from __future__ import annotations

import pytest
import torch

import flashinfer
from flashinfer.mla._sparse_mla_sm120 import (
    _sparse_mla_sm120_paged_attention as sparse_mla_sm120_paged_attention,
)
from flashinfer.utils import is_sm12x_supported

pytestmark = pytest.mark.skipif(
    not is_sm12x_supported(torch.device("cuda")),
    reason="Sparse-MLA SM120 requires SM12x.",
)


# Quantization helpers.


def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor) -> torch.Tensor:
    """Round inverse scale to the nearest power-of-2 (FlashMLA convention)."""
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil())


def _fp32_to_ue8m0_bytes(scale_fp32: torch.Tensor) -> torch.Tensor:
    """Extract the IEEE-754 exponent byte of an FP32 power-of-2 scale."""
    bits = scale_fp32.to(torch.float32).view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_kv_dsv4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv4 FP8 FOOTER format."""
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


# DSv3.2 INLINE pack.


def quantize_kv_dsv3_2(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv3.2 FP8 INLINE format."""
    d_nope, d_rope, tile_size, num_tiles = 512, 64, 128, 4
    scale_bytes = num_tiles * 4  # 16
    bpt = d_nope + scale_bytes + d_rope * 2  # 656
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
    nt = nb * bs  # total token count across all blocks
    kv = kv_bf16.reshape(nt, d)

    result = torch.zeros(nt, bpt, dtype=torch.uint8, device=kv.device)

    for ti in range(num_tiles):
        tile = kv[:, ti * tile_size : (ti + 1) * tile_size].float()
        amax = tile.abs().amax(dim=-1).clamp(min=1e-4)
        scale = _cast_scale_inv_to_ue8m0(amax / 448.0)  # power-of-2 FP32
        fp8 = (tile / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[:, ti * tile_size : (ti + 1) * tile_size] = fp8.view(torch.uint8)
        result[:, d_nope + ti * 4 : d_nope + (ti + 1) * 4] = (
            scale.view(torch.float32).view(torch.uint8).view(nt, 4)
        )

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


# PyTorch SDPA reference.


def _ref_sparse_attn(
    q: torch.Tensor,
    kv_dequant: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense SDPA over sparse-gathered KV."""
    num_tokens, num_heads, d_qk = q.shape
    topk = indices.shape[-1]

    kv_flat = kv_dequant.view(-1, d_qk).float()
    q_f = q.float()

    idx_fixed = indices.clamp(min=0)
    invalid = indices < 0
    if topk_length is not None:
        ar = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (ar >= topk_length.unsqueeze(-1))

    gathered = kv_flat.index_select(0, idx_fixed.view(-1)).view(num_tokens, topk, d_qk)
    P = torch.einsum("thd,tkd->thk", q_f, gathered) * sm_scale
    P[invalid.unsqueeze(1).expand_as(P)] = float("-inf")

    lse_e = torch.logsumexp(P, dim=-1)
    lse_safe = lse_e.clone()
    lse_safe[lse_safe == float("-inf")] = float("+inf")
    weights = torch.exp(P - lse_safe.unsqueeze(-1))
    out_f = torch.einsum("thk,tkd->thd", weights, gathered[..., :d_v])

    LN2 = float(torch.log(torch.tensor(2.0)).item())
    lse_log2 = lse_e / LN2

    if attn_sink is not None:
        sink = attn_sink.float()
        sink_log2 = sink / LN2
        factor = torch.sigmoid(lse_e.float() - sink.unsqueeze(0))
        out_f = out_f * factor.unsqueeze(-1)
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


_DSV4_DECODE_CONFIGS = [
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
    """DSv4 decode."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size  # 4096

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


def test_sparse_mla_sm120_decode_dsv4_topk_length_truncation() -> None:
    """DSv4 decode honors topk_length."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, topk, num_tokens = 32, 512, 16
    topk_len = 128
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
    topk_length = torch.full((num_tokens,), topk_len, dtype=torch.int32, device=device)

    sm_scale = d_qk**-0.5

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


def test_sparse_mla_sm120_decode_dsv4_public_api() -> None:
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
    sm_scale = d_qk**-0.5
    ref_out, _ = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    out = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q.unsqueeze(1),
        swa_kv_cache=kv_packed,
        workspace_buffer=torch.empty(1, dtype=torch.int8, device=device),
        sparse_indices=indices,
        swa_topk_lens=torch.full((num_tokens,), topk, device=device, dtype=torch.int32),
        bmm1_scale=sm_scale,
        kv_layout="NHD",
    )

    torch.testing.assert_close(out.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)

    with pytest.raises(ValueError, match="only supports BF16 query"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q.to(torch.float8_e4m3fn).unsqueeze(1),
            swa_kv_cache=kv_packed,
            workspace_buffer=torch.empty(1, dtype=torch.int8, device=device),
            sparse_indices=indices,
            swa_topk_lens=torch.full(
                (num_tokens,), topk, device=device, dtype=torch.int32
            ),
            bmm1_scale=sm_scale,
            kv_layout="NHD",
        )


def test_sparse_mla_sm120_decode_dsv4_dual_large_extra_topk() -> None:
    """DSv4 dual-cache decode handles large compressed top-k."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads = 1, 16
    topk, extra_topk = 128, 2176
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
    ref_out, _ = _ref_sparse_attn(q, virtual_kv, virtual_idx, sm_scale, d_v)

    output = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q.unsqueeze(1),
        swa_kv_cache=main_packed,
        workspace_buffer=torch.empty(1, dtype=torch.int8, device=device),
        sparse_indices=main_idx,
        compressed_kv_cache=extra_packed,
        swa_topk_lens=torch.full((num_tokens,), topk, dtype=torch.int32, device=device),
        extra_sparse_indices=extra_idx,
        extra_sparse_topk_lens=torch.full(
            (num_tokens,), extra_topk, dtype=torch.int32, device=device
        ),
        bmm1_scale=sm_scale,
        kv_layout="NHD",
    )

    torch.testing.assert_close(output.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)


_DSV3_2_DECODE_HEADS = [8, 16, 32, 64, 128]


@pytest.mark.parametrize("num_heads", _DSV3_2_DECODE_HEADS)
@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_decode_dsv3_2(
    num_heads: int, num_tokens: int, with_sink: bool
) -> None:
    """DSv3.2 decode."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    topk = 2048
    page_block_size = 64
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


def test_sparse_mla_sm120_v32_public_api_accepts_hnd_view() -> None:
    """SM120 v32 accepts HND KV layout."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    num_tokens, num_heads, topk = 4, 8, 128
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
    kv_hnd = kv_packed.transpose(1, 2)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    sm_scale = d_qk**-0.5
    ref_out, _ = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=kv_hnd,
        workspace_buffer=torch.empty(8 << 20, dtype=torch.uint8, device=device),
        qk_nope_head_dim=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=indices.unsqueeze(1),
        seq_lens=None,
        max_seq_len=topk,
        sparse_mla_top_k=topk,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        backend="sparse",
    )

    torch.testing.assert_close(out.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_v32_prefill_public_api_accepts_hnd_view() -> None:
    """SM120 v32 prefill accepts HND KV layout."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    num_tokens, num_heads, topk = 128, 8, 2048
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv3_2(kv_bf16)
    kv_hnd = kv_packed.transpose(1, 2)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    sm_scale = d_qk**-0.5
    ref_out, _ = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        query=q.unsqueeze(1),
        kv_cache=kv_hnd,
        workspace_buffer=torch.empty(8 << 20, dtype=torch.uint8, device=device),
        qk_nope_head_dim=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=indices.unsqueeze(1),
        seq_lens=torch.full((num_tokens,), topk, dtype=torch.int32, device=device),
        max_seq_len=topk,
        sparse_mla_top_k=topk,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        backend="sparse",
    )

    torch.testing.assert_close(out.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)


_DSV3_2_PREFILL_HEADS = [8, 16, 32, 64, 128]


@pytest.mark.parametrize("num_heads", _DSV3_2_PREFILL_HEADS)
@pytest.mark.parametrize("num_tokens", [128, 256])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_prefill_dsv3_2(
    num_heads: int, num_tokens: int, with_sink: bool
) -> None:
    """DSv3.2 prefill."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    topk = 2048
    page_block_size = 64
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
    """DSv4 prefill."""
    torch.manual_seed(0)
    device = torch.device("cuda")
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
    """DSv4 dual-cache prefill."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    topk = 128
    main_pbs = 64
    num_tokens = 128

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
    """DSv4 dual-cache prefill honors extra_topk_length."""
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
    extra_topk_length = torch.full(
        (num_tokens,), extra_topk_len, dtype=torch.int32, device=device
    )

    attn_sink = torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
    sm_scale = d_qk**-0.5

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
    """DSv4 dual-cache prefill handles zero main topk_length."""
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
    """DSv3.2 SG prefill handles zero topk_length."""
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


@pytest.mark.parametrize("extra_topk,extra_pbs", [(1024, 2), (1664, 2), (1024, 64)])
def test_sparse_mla_sm120_prefill_dsv4_dual_runtime_extra_topk(
    extra_topk: int, extra_pbs: int
) -> None:
    """DSv4 dual-cache prefill accepts runtime extra top-k."""
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
