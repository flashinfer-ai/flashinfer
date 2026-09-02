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
    _SparseMLAPagedAttentionRunner,
    _sparse_mla_sm120_paged_attention as sparse_mla_sm120_paged_attention,
)
from flashinfer.utils import is_sm12x_supported

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
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


def _quantize_kv_footer(
    kv_bf16: torch.Tensor,
    d_nope: int,
    d_rope: int,
    tile_size: int,
    scale_bytes: int,
) -> torch.Tensor:
    """Pack bf16 KV into an FP8 FOOTER-scale layout.

    Shared by DSv4 (448/64, tile 64, 7 scales + 1 pad byte) and DOTS3_SWA
    (1024/64, tile 128, 8 scales, no pad). Layout per block of ``bs`` tokens:
    ``[bs * (d_nope + d_rope*2) data | bs * scale_bytes footer]``.
    """
    num_tiles = d_nope // tile_size
    assert num_tiles * tile_size == d_nope
    assert scale_bytes >= num_tiles
    data_stride = d_nope + d_rope * 2
    bpt = data_stride + scale_bytes
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope + d_rope and hk == 1
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


def quantize_kv_dsv4(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DSv4 FP8 FOOTER format."""
    return _quantize_kv_footer(kv_bf16, 448, 64, 64, 8)


def quantize_kv_dots3_swa(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack bf16 KV into DOTS3_SWA FP8 FOOTER format (1160 B/token)."""
    return _quantize_kv_footer(kv_bf16, 1024, 64, 128, 8)


def _dequantize_kv_footer(
    packed: torch.Tensor,
    d_nope: int,
    d_rope: int,
    tile_size: int,
    scale_bytes: int,
) -> torch.Tensor:
    """Unpack an FP8 FOOTER layout → bf16. Inverse of :func:`_quantize_kv_footer`."""
    num_tiles = d_nope // tile_size
    data_stride = d_nope + d_rope * 2
    bpt = data_stride + scale_bytes
    d_qk = d_nope + d_rope
    nb, bs, _, _ = packed.shape
    result = torch.zeros(nb, bs, d_qk, dtype=torch.bfloat16, device=packed.device)
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

    return result.view(nb, bs, 1, d_qk)


def dequantize_kv_dsv4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DSV4 FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_dsv4`."""
    return _dequantize_kv_footer(packed, 448, 64, 64, 8)


def dequantize_kv_dots3_swa(packed: torch.Tensor) -> torch.Tensor:
    """Unpack DOTS3_SWA FP8 FOOTER → bf16. Inverse of :func:`quantize_kv_dots3_swa`."""
    return _dequantize_kv_footer(packed, 1024, 64, 128, 8)


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


def quantize_kv_glm53_nope(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Pack native NoPE KV into the 656B ABI with arbitrary FP32 scales."""
    d_nope, tile_size, num_tiles = 512, 128, 4
    bpt = 656
    nb, bs, hk, d = kv_bf16.shape
    assert d == d_nope and hk == 1
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

    # Bytes 528:656 are reserved padding in the stable packed-cache ABI.
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
    # BI is model-dependent: the DeepSeek family consumes 64 candidates per
    # iteration, DOTS3_SWA 32. num_splits must be derived with the same value
    # the kernel uses or the grid covers only part of each candidate list.
    bi = 32 if d_v == 1024 else 64
    num_splits = (topk + bi - 1) // bi + (extra_topk + bi - 1) // bi
    # The runtime-H decode kernels HPB-align the scratch head dim (the
    # dedicated num_heads=8 instantiation keeps the true count).
    from flashinfer.mla._sparse_mla_sm120_plan import _decode_scratch_heads

    scratch_heads = _decode_scratch_heads(num_heads)
    return (
        torch.empty(
            (num_tokens, scratch_heads, num_splits, d_v),
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.empty(
            (num_tokens, scratch_heads, num_splits),
            dtype=torch.float32,
            device=device,
        ),
    )


_DSV4_DECODE_CONFIGS = [
    (8, 128),
    (8, 192),
    (8, 256),
    (8, 512),
    (8, 1024),
    (16, 128),
    (16, 192),
    (16, 256),
    (32, 192),
    (32, 256),
    (32, 512),
    (64, 192),
    (64, 256),
    (64, 1024),
    (128, 192),
    (128, 256),
    (128, 1024),
    # Runtime-H instantiation: arbitrary head counts ride the NUM_HEADS=0
    # kernel (zero-Q-padded tile, HPB-aligned scratch). 12 exercises the
    # in-block pad path, 24 a remainder second block, 80 an exact multiple.
    (12, 512),
    (24, 256),
    (80, 128),
    # Runtime-topk: topk is a runtime kernel argument (the indices-row
    # width). 384 is a multiple of the BI=64 tile off the calibrated grid;
    # 500 exercises the partial tail chunk.
    (64, 384),
    (64, 500),
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


@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("row_stride", [656, 1088])
def test_sparse_mla_sm120_decode_dsv3_2_padded_row(
    num_heads: int, row_stride: int
) -> None:
    """DSv3.2 decode over a KV cache whose per-token rows are padded.

    A serving stack may pad the 656 B packed rows out to a wider stride so
    layer types with different geometries share one KV cache group. The packed
    payload stays at the row start; only the per-token advance changes.
    ``row_stride=656`` is the unpadded control — both must agree with the same
    reference.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    bpt = 656
    page_block_size, num_blocks, topk = 64, 64, 1024
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    packed = quantize_kv_dsv3_2(kv_bf16)  # [nb, pbs, 1, 656]
    kv_dequant = dequantize_kv_dsv3_2(packed)

    if row_stride == bpt:
        kv_cache = packed
    else:
        # Widen each token row, payload at the start, garbage in the padding so
        # a kernel that used the packed stride would read the wrong bytes.
        kv_cache = torch.randint(
            0,
            256,
            (num_blocks, page_block_size, 1, row_stride),
            dtype=torch.uint8,
            device=device,
        )
        kv_cache[..., :bpt] = packed
        kv_cache = kv_cache.contiguous()

    q = (
        torch.randn(
            num_tokens_ := 16, num_heads, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens_, topk), device=device, dtype=torch.int32
    )
    indices[:, topk // 2 :] = -1
    sm_scale = d_qk**-0.5

    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens_, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens_, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens_, num_heads, topk, d_v, device)

    sparse_mla_sm120_paged_attention(
        q,
        kv_cache,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-3, rtol=5e-3)


def test_sparse_mla_sm120_decode_dsv4_padded_row_rejected() -> None:
    """DSV4 decode over a padded-row KV cache fails loudly at the binding.

    Only the decode-v32 kernel honors stride_kv_row in its gather; the
    footer-scale layouts (DSV4, DOTS3_SWA, and the dual extra cache) assume
    tightly packed rows, so the binding rejects the padded cache instead of
    gathering the wrong bytes.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    bpt = 584
    page_block_size, num_blocks, topk = 64, 64, 1024
    num_tokens, num_heads = 16, 128
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    packed = quantize_kv_dsv4(kv_bf16)  # [nb, pbs, 1, 584]
    kv_padded = torch.randint(
        0, 256, (num_blocks, page_block_size, 1, 1024), dtype=torch.uint8, device=device
    )
    kv_padded[..., :bpt] = packed
    kv_padded = kv_padded.contiguous()

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)

    with pytest.raises(RuntimeError, match="tightly packed"):
        sparse_mla_sm120_paged_attention(
            q,
            kv_padded,
            indices,
            output,
            out_lse,
            d_qk**-0.5,
            d_v=d_v,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )


def test_sparse_mla_sm120_decode_dsv4_indices_rows_checked() -> None:
    """The decode binding rejects an indices tensor whose leading dimension
    does not match num_tokens (mirrors the prefill-side guard)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk = d_v = 512
    page_block_size, num_blocks, topk = 64, 64, 640
    num_tokens, num_heads = 8, 64

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0,
        num_blocks * page_block_size,
        (num_tokens, topk),
        device=device,
        dtype=torch.int32,
    )
    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)

    with pytest.raises(
        RuntimeError, match="indices leading dimension must match num_tokens"
    ):
        sparse_mla_sm120_paged_attention(
            q,
            kv_packed,
            indices[: num_tokens - 1],
            output,
            out_lse,
            d_qk**-0.5,
            d_v=d_v,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )


def test_sparse_mla_sm120_decode_dsv4_dots3_swa_rejects_dual_cache() -> None:
    """DOTS3_SWA has no dual-cache instantiation; the standalone decode entry
    rejects extra_kv_cache instead of running an untested path."""
    from flashinfer.mla import _sparse_mla_sm120 as sm

    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v, topk = 1088, 1024, 576
    page_block_size, num_blocks, num_tokens, num_heads = 64, 64, 8, 16

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dots3_swa(kv_bf16)
    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0,
        num_blocks * page_block_size,
        (num_tokens, topk),
        device=device,
        dtype=torch.int32,
    )
    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)

    with pytest.raises(RuntimeError, match="no dual-cache form"):
        sm.sparse_mla_sm120_decode_dsv4(
            q,
            kv_packed,
            indices,
            mid_out,
            mid_lse,
            output,
            out_lse,
            d_qk**-0.5,
            extra_kv_cache=kv_packed,
            extra_indices=indices,
        )


@pytest.mark.parametrize("num_heads", [8, 64])
def test_sparse_mla_sm120_decode_dots3_swa_no_topk_length(num_heads: int) -> None:
    """DOTS3_SWA with -1 padding and no topk_length.

    The kernel caps the candidate count at DecodeTileCfg<DOTS3_SWA>::WINDOW
    (513) on its own, so a caller that -1-pads the unused slots may omit
    topk_length entirely. Also pins the cap: slots in [WINDOW, TOPK) hold
    VALID indices, and the kernel must ignore them purely because they sit
    past the window.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v, topk, window = 1088, 1024, 576, 513
    page_block_size, num_blocks, num_tokens = 64, 64, 8
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dots3_swa(kv_bf16)
    kv_dequant = dequantize_kv_dots3_swa(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Valid entries only inside the window; -1 marks the rest of the window.
    indices[:, window // 2 : window] = -1
    # [window, topk) keeps valid indices — the WINDOW cap alone must drop them.
    sm_scale = d_qk**-0.5

    # Reference sees exactly the window, via an explicit length.
    ref_len = torch.full((num_tokens,), window, device=device, dtype=torch.int32)
    ref_out, ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, topk_length=ref_len
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
        mid_out=mid_out,
        mid_lse=mid_lse,
    )  # no topk_length

    torch.testing.assert_close(output, ref_out, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_heads", [8, 64])
def test_sparse_mla_sm120_decode_dots3_swa_window_boundary(num_heads: int) -> None:
    """Runtime-topk window boundary: topk=513 (exactly WINDOW) decodes
    correctly; topk=512 (buffer narrower than the window) raises a readable
    error naming the 513 minimum."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v, window = 1088, 1024, 513
    page_block_size, num_blocks, num_tokens = 64, 64, 4
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dots3_swa(kv_bf16)
    kv_dequant = dequantize_kv_dots3_swa(kv_packed)
    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    sm_scale = d_qk**-0.5

    # topk == 513: the tightest legal buffer; every slot is inside the window.
    indices = torch.randint(
        0, s_kv, (num_tokens, window), device=device, dtype=torch.int32
    )
    indices[:, window // 2 :] = -1
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)
    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, window, d_v, device)
    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )
    torch.testing.assert_close(output, ref_out, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-3, rtol=5e-3)

    # topk == 512: cannot hold the window — rejected with the 513 minimum.
    bad_indices = torch.randint(
        0, s_kv, (num_tokens, 512), device=device, dtype=torch.int32
    )
    with pytest.raises(ValueError, match=r"topk=512 is below .*topk >= 513"):
        sparse_mla_sm120_paged_attention(
            q,
            kv_packed,
            bad_indices,
            output,
            out_lse,
            sm_scale,
            d_v=d_v,
        )


_DOTS3_SWA_DECODE_CONFIGS = [(8, 576), (16, 576), (32, 576), (64, 576)]


@pytest.mark.parametrize("num_heads,topk", _DOTS3_SWA_DECODE_CONFIGS)
@pytest.mark.parametrize("num_tokens", [1, 16, 64])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_decode_dots3_swa(
    num_heads: int, topk: int, num_tokens: int, with_sink: bool
) -> None:
    """DOTS3_SWA sliding-window decode: d_qk 1088 / d_v 1024.

    This is the D_V != 512 path. Unlike DSv4 (d_v == d_qk, so rope is part of
    V), here d_v == d_nope: rope participates in QK but must not reach the
    output row, which the reference expresses as ``gathered[..., :d_v]``.

    Tolerances are 5e-3, not the 5e-2 used by the DSv4 cases above. Measured
    worst-case error over all 24 parametrizations is 6.1e-4 (output) and
    6.9e-5 (LSE), so 5e-2 would have left ~80x of slack. At 5e-3 a V-segment
    misalignment of one rope width perturbs the output by ~2.8e-2 and fails;
    at 5e-2 it passed, i.e. the loose tolerance made the check vacuous.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 1088, 1024
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dots3_swa(kv_bf16)
    kv_dequant = dequantize_kv_dots3_swa(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Per-token sliding-window lengths, as an SWA metadata builder would emit:
    # tokens early in a sequence see a partial window, the rest the full 513.
    # Every slot past the length keeps a VALID index (not the -1 sentinel), so a
    # kernel that ignored topk_length would fold real KV into the result and miss
    # the reference. That is what gives this test coverage of topk_length.
    window = 513
    topk_length = torch.randint(
        1, window + 1, (num_tokens,), device=device, dtype=torch.int32
    )
    topk_length[0] = window  # pin one token to the full window

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    ref_out, ref_lse = _ref_sparse_attn(
        q,
        kv_dequant,
        indices,
        sm_scale,
        d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
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
        topk_length=topk_length,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("topk,topk_len", [(192, 133), (256, 133), (512, 128)])
def test_sparse_mla_sm120_decode_dsv4_topk_length_truncation(
    num_heads: int, topk: int, topk_len: int
) -> None:
    """DSv4 decode honors topk_length."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens = 16
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


def test_sparse_mla_sm120_decode_unsupported_shape_fails_before_prefill() -> None:
    """A shape served by neither decode nor the prefill envelope raises.

    num_heads=256 is past the runtime-H decode ceiling (128) and outside
    every prefill head set, so the planner's None is converted to the
    diagnostic ValueError.
    """
    device = torch.device("cuda")
    num_tokens, num_heads, topk = 1, 256, 384
    d_qk = d_v = 512

    q = torch.empty((num_tokens, num_heads, d_qk), dtype=torch.bfloat16, device=device)
    kv_cache = torch.empty((1, 64, 1, 584), dtype=torch.uint8, device=device)
    indices = torch.zeros((num_tokens, topk), dtype=torch.int32, device=device)
    output = torch.empty(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="prefill envelope both reject"):
        sparse_mla_sm120_paged_attention(
            q,
            kv_cache,
            indices,
            output,
            out_lse,
            d_qk**-0.5,
            d_v=d_v,
        )


def test_sparse_mla_sm120_decode_empty_query() -> None:
    """Zero-token decode (EP rank with no tokens) returns empty outputs."""
    device = torch.device("cuda")
    num_heads, topk = 8, 128
    d_qk = 576

    kv_bf16 = (
        torch.randn(4, 64, 1, d_qk, device=device, dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    kv_hnd = quantize_kv_dsv3_2(kv_bf16).transpose(1, 2)

    query = torch.empty((0, 1, num_heads, d_qk), dtype=torch.bfloat16, device=device)
    block_tables = torch.empty((0, 1, topk), dtype=torch.int32, device=device)
    workspace = torch.empty(8 << 20, dtype=torch.uint8, device=device)
    kwargs = dict(
        query=query,
        kv_cache=kv_hnd,
        workspace_buffer=workspace,
        qk_nope_head_dim=512,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=block_tables,
        seq_lens=None,
        max_seq_len=64,
        sparse_mla_top_k=topk,
        bmm1_scale=d_qk**-0.5,
        bmm2_scale=1.0,
        backend="sparse",
    )

    out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(**kwargs)
    assert out.shape == (0, 1, num_heads, 512)

    # Caller-supplied buffers must pass through untouched (identity, not copy).
    user_out = torch.empty((0, 1, num_heads, 512), dtype=torch.bfloat16, device=device)
    user_lse = torch.empty((0, num_heads), dtype=torch.float32, device=device)
    out2, lse2 = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        **kwargs, out=user_out, lse=user_lse, return_lse=True
    )
    assert out2 is user_out
    assert lse2 is user_lse

    # return_lse without a caller buffer: fresh flat-shaped empty lse.
    _, lse3 = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
        **kwargs, return_lse=True
    )
    assert lse3.shape == (0, num_heads)
    assert lse3.dtype == torch.float32


def test_sparse_mla_sm120_decode_zero_tokens_direct() -> None:
    """Direct decode-wrapper call with T=0 returns empty outputs (no launch)."""
    from flashinfer.mla._sparse_mla_sm120 import sparse_mla_sm120_decode_dsv4

    device = torch.device("cuda")
    num_heads, topk, d_qk, d_v = 8, 128, 512, 512
    q = torch.empty((0, num_heads, d_qk), dtype=torch.bfloat16, device=device)
    kv_cache = torch.empty(4, 64 * 584, dtype=torch.uint8, device=device)
    indices = torch.empty((0, topk), dtype=torch.int32, device=device)
    mid_out = torch.empty((0, num_heads, 2, d_v), dtype=torch.bfloat16, device=device)
    mid_lse = torch.empty((0, num_heads, 2), dtype=torch.float32, device=device)
    output = torch.empty((0, num_heads, d_v), dtype=torch.bfloat16, device=device)
    out_lse = torch.empty((0, num_heads), dtype=torch.float32, device=device)

    returned = sparse_mla_sm120_decode_dsv4(
        q, kv_cache, indices, mid_out, mid_lse, output, out_lse, d_qk**-0.5
    )
    assert returned is output


@pytest.mark.parametrize("family", ["dsv4", "dsv3_2"])
def test_sparse_mla_sm120_decode_row_strided_indices(family: str) -> None:
    """Decode accepts indices as row-strided views of a wider buffer."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk = 512 if family == "dsv4" else 576
    d_v = 512
    num_tokens, num_heads, topk = 16, 8, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    if family == "dsv4":
        kv_packed = quantize_kv_dsv4(kv_bf16)
        kv_dequant = dequantize_kv_dsv4(kv_packed)
    else:
        kv_packed = quantize_kv_dsv3_2(kv_bf16)
        kv_dequant = dequantize_kv_dsv3_2(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)

    # The kernel only sees the [:, :topk] view of a wider backing buffer.
    wide = torch.full((num_tokens, topk + 128), -1, dtype=torch.int32, device=device)
    wide[:, :topk] = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    indices = wide[:, :topk]
    assert not indices.is_contiguous()

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
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "num_heads,topk,num_tokens,kv_layout",
    [
        (32, 128, 7, "NHD"),
        (32, 192, 7, "HND"),
        (32, 256, 7, "HND"),
        (8, 192, 128, "HND"),
        (8, 256, 128, "HND"),
    ],
)
def test_sparse_mla_sm120_dsv4_public_api(
    num_heads: int, topk: int, num_tokens: int, kv_layout: str
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
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
    workspace_buffer = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    swa_topk_lens = torch.full((num_tokens,), topk, device=device, dtype=torch.int32)
    seq_lens = torch.full((num_tokens,), s_kv, device=device, dtype=torch.int32)
    kv_cache = (
        kv_packed if kv_layout == "NHD" else kv_packed.transpose(1, 2).contiguous()
    )

    out = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q.unsqueeze(1),
        swa_kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        sparse_indices=indices,
        seq_lens=seq_lens,
        swa_topk_lens=swa_topk_lens,
        bmm1_scale=sm_scale,
        kv_layout=kv_layout,
    )

    torch.testing.assert_close(out.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)

    out_buffer = torch.empty_like(out)
    returned = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q.unsqueeze(1),
        swa_kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        sparse_indices=indices,
        seq_lens=seq_lens,
        out=out_buffer,
        swa_topk_lens=swa_topk_lens,
        bmm1_scale=sm_scale,
        kv_layout=kv_layout,
    )
    assert returned.data_ptr() == out_buffer.data_ptr()
    torch.testing.assert_close(out_buffer.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)

    with pytest.raises(ValueError, match="only supports BF16 query"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q.to(torch.float8_e4m3fn).unsqueeze(1),
            swa_kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            sparse_indices=indices,
            seq_lens=seq_lens,
            swa_topk_lens=swa_topk_lens,
            bmm1_scale=sm_scale,
            kv_layout=kv_layout,
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
# Runtime-H decode-dsv3_2: arbitrary head counts (remainder-block pad path).
_DSV3_2_DECODE_RUNTIME_HEADS = [12, 24]


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


@pytest.mark.parametrize("num_heads", _DSV3_2_DECODE_RUNTIME_HEADS)
@pytest.mark.parametrize("num_tokens", [1, 64])
def test_sparse_mla_sm120_decode_dsv3_2_runtime_h(
    num_heads: int, num_tokens: int
) -> None:
    """DSv3.2 decode at arbitrary head counts (runtime-H instantiation)."""
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


@pytest.mark.parametrize("num_heads", [8, 32, 64, 128])
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


# num_heads=8 exercises the runtime-H instantiation (GLM53_NOPE has dedicated
# 32/64 only), e.g. a 64-head layer at TP8.
@pytest.mark.parametrize("num_heads", [8, 32, 64])
def test_sparse_mla_sm120_decode_glm53_nope(num_heads: int) -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")
    d_qk = d_v = 512
    num_tokens, topk = 4, 2176
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_glm53_nope(kv_bf16)
    _assert_has_non_pow2_inline_scales(kv_packed)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)[..., :d_qk]

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


def test_sparse_mla_sm120_prefill_glm53_nope() -> None:
    torch.manual_seed(4)
    device = torch.device("cuda")
    d_qk = d_v = 512
    num_tokens, num_heads, topk = 65, 32, 2176
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_glm53_nope(kv_bf16)
    _assert_has_non_pow2_inline_scales(kv_packed)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)[..., :d_qk]

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


@pytest.mark.parametrize("num_heads", [64, 128])
def test_sparse_mla_sm120_prefill_glm53_nope_swapab(num_heads: int) -> None:
    """swapAB serves GLM53_NOPE at topk=2176; auto routes to it (bitwise)."""
    torch.manual_seed(11)
    device = torch.device("cuda")
    d_qk = d_v = 512
    num_tokens, topk = 128, 2176
    page_block_size = 64
    num_blocks = 128
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_glm53_nope(kv_bf16)
    _assert_has_non_pow2_inline_scales(kv_packed)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)[..., :d_qk]

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

    results = {}
    for impl in ("swapab", None):
        output = torch.zeros(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
        )
        out_lse = torch.zeros(
            (num_tokens, num_heads), dtype=torch.float32, device=device
        )
        sparse_mla_sm120_paged_attention(
            q,
            kv_packed,
            indices,
            output,
            out_lse,
            sm_scale,
            d_v=d_v,
            kv_scale_format="arbitrary_fp32",
            prefill_impl=impl,
        )
        torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)
        results[impl] = (output, out_lse)

    # The auto policy prefers swapAB at this eligible shape.
    assert torch.equal(results["swapab"][0], results[None][0])
    assert torch.equal(results["swapab"][1], results[None][1])


_DSV4_PREFILL_CONFIGS = [
    (8, 128),
    (8, 192),
    (8, 256),
    (8, 512),
    (8, 1024),
    (8, 2048),
    (16, 128),
    (16, 192),
    (16, 256),
    (32, 192),
    (32, 256),
    (32, 512),
    (64, 192),
    (64, 256),
    (64, 1024),
    (128, 192),
    (128, 256),
    (128, 1024),
]


@pytest.mark.parametrize("num_heads,topk", _DSV4_PREFILL_CONFIGS)
@pytest.mark.parametrize("num_tokens", [65, 128, 256])
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


@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("topk", [192, 256])
def test_sparse_mla_sm120_prefill_dsv4_topk_length_truncation(
    num_heads: int, topk: int
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk_len, num_tokens = 133, 128
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
    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        topk_length=topk_length,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


_DOTS3_SWA_PREFILL_HEADS = [8, 16, 32, 64]


@pytest.mark.parametrize("num_heads", _DOTS3_SWA_PREFILL_HEADS)
@pytest.mark.parametrize("num_tokens", [65, 128])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_prefill_dots3_swa(
    num_heads: int, num_tokens: int, with_sink: bool
) -> None:
    """DOTS3_SWA sliding-window prefill: d_qk 1088 / d_v 1024.

    SG-only, at PrefillTileCfg<DOTS3_SWA>'s BI=32 / 4-math-warp QK tile — the
    MG layout does not fit D_NOPE=1024 in sm120 smem, so num_heads > 16 is
    served by replicating one CTA per 16-head tile rather than by MG. That
    replication is exactly what num_heads 32 and 64 cover here.

    Tolerance is atol 1e-2 / rtol 5e-3, not the 5e-2 the DSv4 prefill cases
    use. Measured over all 16 parametrizations the worst output error is
    6.8e-3, with a mean of 1.4e-4 and only 5 of 8.4M elements above 5e-3 —
    ordinary bf16 accumulation tail, so 5e-3 is below the noise floor here even
    though DOTS3_SWA decode holds it. The check still has teeth: a V-segment
    misalignment of 8 or 64 elements moves the worst error to 0.48 / 0.59, ~50x
    the tolerance. At 5e-2 that defect would pass.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v, topk, window = 1088, 1024, 576, 513
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dots3_swa(kv_bf16)
    kv_dequant = dequantize_kv_dots3_swa(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Per-token window lengths, as an SWA metadata builder would emit. Slots
    # past the length keep VALID indices, so a kernel that ignored topk_length
    # would fold real KV into the result — that is what gives this coverage.
    topk_length = torch.randint(
        1, window + 1, (num_tokens,), device=device, dtype=torch.int32
    )
    topk_length[0] = window  # pin one token to the full window
    topk_length[1] = 0  # pin one token to an empty window (zero tiles)

    attn_sink = (
        torch.randn(num_heads, device=device, dtype=torch.float32) * 2.0
        if with_sink
        else None
    )

    sm_scale = d_qk**-0.5

    ref_out, ref_lse = _ref_sparse_attn(
        q,
        kv_dequant,
        indices,
        sm_scale,
        d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
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
        topk_length=topk_length,
    )

    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=5e-3)
    if attn_sink is None:
        # Empty row: kernel emits the -1e30 sentinel, the reference -inf.
        nonempty = topk_length.bool()
        torch.testing.assert_close(
            out_lse[nonempty], ref_lse[nonempty], atol=5e-3, rtol=5e-3
        )
        assert (out_lse[1] == -1e30).all()
    else:
        torch.testing.assert_close(out_lse, ref_lse, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_heads", [8, 64])
def test_sparse_mla_sm120_prefill_dots3_swa_no_topk_length(num_heads: int) -> None:
    """DOTS3_SWA prefill with -1 padding and no topk_length.

    Counterpart to the decode test of the same name: PrefillTileCfg's WINDOW
    caps the per-token candidate count inside the kernel, so a caller that
    -1-pads unused slots may omit topk_length. Slots in [WINDOW, TOPK) hold
    VALID indices, so only the cap can exclude them — without it the kernel
    would scan all 576 candidates and miss the reference.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v, topk, window = 1088, 1024, 576, 513
    page_block_size, num_blocks, num_tokens = 64, 64, 96
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dots3_swa(kv_bf16)
    kv_dequant = dequantize_kv_dots3_swa(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    # Valid entries only inside the window; -1 marks the rest of the window.
    # [window, topk) keeps valid indices — the WINDOW cap alone must drop them.
    indices[:, window // 2 : window] = -1

    sm_scale = d_qk**-0.5

    # Reference sees exactly the window, via an explicit length.
    ref_len = torch.full((num_tokens,), window, device=device, dtype=torch.int32)
    ref_out, ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, topk_length=ref_len
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
    )  # no topk_length

    # Same tolerance rationale as test_sparse_mla_sm120_prefill_dots3_swa.
    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=5e-3)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_heads", [16, 64])
def test_sparse_mla_sm120_prefill_dots3_swa_offpin_topk(num_heads: int) -> None:
    """DOTS3_SWA prefill at topk=640 (>= 513, whole tiles, not the 576 pin).

    topk is a runtime kernel argument; the kernel still clamps the scan to
    the 513-wide window, so the extra buffer rows only matter through
    topk_length. Same coverage shape and tolerance rationale as
    test_sparse_mla_sm120_prefill_dots3_swa.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v, topk, window = 1088, 1024, 640, 513
    page_block_size = 64
    num_blocks = 64
    num_tokens = 128
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dots3_swa(kv_bf16)
    kv_dequant = dequantize_kv_dots3_swa(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    topk_length = torch.randint(
        1, window + 1, (num_tokens,), device=device, dtype=torch.int32
    )
    topk_length[0] = window  # pin one token to the full window

    sm_scale = d_qk**-0.5

    ref_out, ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, topk_length=topk_length
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
        topk_length=topk_length,
    )

    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=5e-3)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("num_tokens", [16, 128])
def test_sparse_mla_sm120_dots3_swa_runner(num_tokens: int) -> None:
    """DOTS3_SWA through _SparseMLAPagedAttentionRunner.

    The runner is the integration surface a serving stack holds across steps:
    it owns the LSE buffer and allocates decode split-K scratch itself. Both
    of those are sized from ``d_v`` and from a candidate-tile width that is
    model-dependent (64 for the DeepSeek family, 32 here), so this covers
    wiring the module-level entry point cannot. ``num_tokens`` straddles the
    decode/prefill cutoff of 64 to exercise both dispatches.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v, topk, window = 1088, 1024, 576, 513
    num_heads, page_block_size, num_blocks = 16, 64, 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dots3_swa(kv_bf16)
    kv_dequant = dequantize_kv_dots3_swa(kv_packed)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    topk_length = torch.randint(
        1, window + 1, (num_tokens,), device=device, dtype=torch.int32
    )
    topk_length[0] = window
    sm_scale = d_qk**-0.5

    ref_out, ref_lse = _ref_sparse_attn(
        q, kv_dequant, indices, sm_scale, d_v, topk_length=topk_length
    )

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    runner = _SparseMLAPagedAttentionRunner(d_v=d_v, device=device)
    out_lse = runner.run(
        q,
        kv_packed,
        indices,
        output,
        sm_scale,
        topk_length=topk_length,
        return_lse=True,
    )

    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=5e-3)
    assert out_lse is not None
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-3, rtol=5e-3)


def test_sparse_mla_sm120_runner_rejects_unknown_d_v() -> None:
    """The runner accepts every model type's d_v and nothing else."""
    device = torch.device("cuda")
    for d_v in (512, 1024):
        _SparseMLAPagedAttentionRunner(d_v=d_v, device=device)
    with pytest.raises(ValueError, match="d_v"):
        _SparseMLAPagedAttentionRunner(d_v=768, device=device)


def test_sparse_mla_sm120_runner_wide_lse_buffer() -> None:
    """An LSE buffer wider than the call's head count must stay correct.

    The runner hands the kernels a non-contiguous column slice of a wider
    buffer, which the kernels honor through the out_lse row stride.
    Previously the decode path silently corrupted every row past the first
    while prefill raised on the same call shape. Covers both the
    constructor-pre-allocated buffer and a caller-passed one.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk = d_v = 512
    num_tokens, num_heads, topk = 8, 64, 640
    page_block_size, num_blocks = 64, 64
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
    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    def run(runner, out_lse=None):
        output = torch.zeros(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
        )
        lse = runner.run(
            q, kv_packed, indices, output, sm_scale, out_lse=out_lse, return_lse=True
        )
        torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
        assert lse is not None
        torch.testing.assert_close(lse, ref_lse, atol=5e-2, rtol=5e-2)

    # Constructor-pre-allocated buffer wider than the call's head count.
    run(
        _SparseMLAPagedAttentionRunner(
            max_num_tokens=num_tokens, max_num_heads=128, device=device
        )
    )
    # Caller-passed wide buffer.
    wide = torch.full((num_tokens, 128), float("nan"), device=device)
    run(_SparseMLAPagedAttentionRunner(device=device), out_lse=wide)


_DSV4_PREFILL_DUAL_HEADS = [8, 16, 32, 64, 128]

# (num_heads, topk, extra_topk, extra_pbs). topk=512: DeepSeek V4 Vision
# primary candidate set (H=32/64 shards, both extra-cache page layouts).
_DSV4_PREFILL_DUAL_CONFIGS = [
    (num_heads, 128, extra_topk, extra_pbs)
    for num_heads in _DSV4_PREFILL_DUAL_HEADS
    for extra_topk, extra_pbs in [(128, 64), (512, 64), (512, 2)]
] + [
    (32, 512, 512, 64),
    (32, 512, 128, 2),
    (64, 512, 512, 64),
    (64, 512, 128, 2),
]


@pytest.mark.parametrize(
    "num_heads,topk,extra_topk,extra_pbs", _DSV4_PREFILL_DUAL_CONFIGS
)
def test_sparse_mla_sm120_prefill_dsv4_dual(
    num_heads: int, topk: int, extra_topk: int, extra_pbs: int
) -> None:
    """DSv4 dual-cache prefill."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
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


@pytest.mark.parametrize("num_heads", [8, 64])
@pytest.mark.parametrize("extra_topk_len", [0, 128, 768])
def test_sparse_mla_sm120_prefill_dsv4_dual_extra_topk_length_truncation(
    num_heads: int, extra_topk_len: int
) -> None:
    """DSv4 dual-cache prefill honors extra_topk_length."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens = 128
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


def _make_dsv3_2_prefill_case(
    num_heads: int, num_tokens: int, topk: int = 2048
) -> tuple:
    """Shared inputs for prefill_impl tests (DSv3.2, pow2 inline scales)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
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

    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)
    return q, kv_packed, indices, sm_scale, d_v, ref_out, ref_lse


def _run_prefill_impl(case: tuple, prefill_impl) -> tuple:
    q, kv_packed, indices, sm_scale, d_v, _, _ = case
    num_tokens, num_heads = q.shape[0], q.shape[1]
    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=q.device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=q.device)
    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        prefill_impl=prefill_impl,
    )
    return output, out_lse


@pytest.mark.parametrize("num_heads", [8, 32, 64])
def test_sparse_mla_sm120_prefill_dsv3_2_offpin_topk(num_heads: int) -> None:
    """DSv3.2 prefill at topk=1024 (not the historical 2048 pin).

    topk is a runtime kernel argument: one instantiation per variant serves
    every whole-tile width. num_heads 8/32/64 route auto to SG/MG/swapAB, so
    the three V32 prefill variants are all exercised at an off-pin width.
    """
    case = _make_dsv3_2_prefill_case(num_heads, num_tokens=128, topk=1024)
    output, out_lse = _run_prefill_impl(case, None)
    torch.testing.assert_close(output, case[5], atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, case[6], atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_prefill_ragged_topk_rejected() -> None:
    """Prefill rejects an indices width that is not a whole number of 64-wide
    index tiles (the kernels issue whole tiles and do not mask the tail), and
    a DOTS3_SWA width below the 513-wide sliding window. Both checks live in
    the FFI binding, ahead of dispatch."""
    from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

    device = torch.device("cuda")
    num_tokens = 2
    module = _get_sparse_mla_sm120_decode_module()

    def call(d_qk: int, d_v: int, bpt: int, topk: int, model_type: int, variant: int):
        module.sparse_mla_sm120_paged_attention(
            torch.zeros(num_tokens, 64, d_qk, dtype=torch.bfloat16, device=device),
            torch.zeros(4, 64 * bpt, dtype=torch.uint8, device=device),
            torch.zeros(num_tokens, topk, dtype=torch.int32, device=device),
            torch.zeros(num_tokens, 64, d_v, dtype=torch.bfloat16, device=device),
            torch.zeros(num_tokens, 64, dtype=torch.float32, device=device),
            d_qk**-0.5,
            model_type,
            variant,
            None,
            None,
            None,
            None,
            None,
        )

    # topk=1000: 15 whole tiles plus a ragged 40-entry tail.
    with pytest.raises(RuntimeError, match=r"topk % 64 == 0"):
        call(512, 512, 584, 1000, 1, 2)  # DSV4, PREFILL_MG
    # DOTS3_SWA at topk=512: whole tiles, but below the sliding-window floor.
    with pytest.raises(RuntimeError, match=r"topk >= 513"):
        call(1088, 1024, 1160, 512, 4, 1)  # DOTS3_SWA, PREFILL_SG


@pytest.mark.parametrize("num_heads", [64, 128])
def test_sparse_mla_sm120_prefill_impl_swapab_forced(num_heads: int) -> None:
    """Forced swapAB matches the reference at an eligible DSv3.2 shape."""
    case = _make_dsv3_2_prefill_case(num_heads, num_tokens=128)
    output, out_lse = _run_prefill_impl(case, "swapab")
    torch.testing.assert_close(output, case[5], atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, case[6], atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("num_heads", [64, 128])
def test_sparse_mla_sm120_prefill_impl_mg_matches_swapab(num_heads: int) -> None:
    """Forced MG and forced swapAB agree on identical inputs (and the ref)."""
    case = _make_dsv3_2_prefill_case(num_heads, num_tokens=128)
    out_swapab, lse_swapab = _run_prefill_impl(case, "swapab")
    out_mg, lse_mg = _run_prefill_impl(case, "mg")
    torch.testing.assert_close(out_mg, out_swapab, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_mg, lse_swapab, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_mg, case[5], atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_mg, case[6], atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_inline_scale_rejects_padded_block_stride() -> None:
    """Non-contiguous inline-scale caches fail at the entry.

    The prefill kernels address inline-scale caches as a flat token array, so
    a padded block stride would be silently misread — and crossover can route
    any decode-form call there, so the rejection cannot wait for a
    prefill-routed call.
    """
    q, kv_packed, indices, sm_scale, d_v, _, _ = _make_dsv3_2_prefill_case(
        64, num_tokens=128
    )
    base = torch.zeros(
        kv_packed.shape[0] * 2,
        *kv_packed.shape[1:],
        dtype=kv_packed.dtype,
        device=kv_packed.device,
    )
    kv = base[::2]
    kv.copy_(kv_packed)

    output = torch.zeros(128, 64, d_v, dtype=torch.bfloat16, device=q.device)
    out_lse = torch.zeros(128, 64, dtype=torch.float32, device=q.device)
    with pytest.raises(ValueError, match="must be contiguous"):
        sparse_mla_sm120_paged_attention(
            q, kv, indices, output, out_lse, sm_scale, d_v=d_v
        )


def test_sparse_mla_sm120_inline_scale_prefill_rejects_padded_rows() -> None:
    """Padded-row inline-scale caches are decode-only: decode-v32 honors the
    row stride, and a prefill-routed call fails loudly at the binding."""
    q, kv_packed, indices, sm_scale, d_v, _, _ = _make_dsv3_2_prefill_case(
        64, num_tokens=128
    )
    kv = torch.zeros(
        *kv_packed.shape[:-1],
        kv_packed.shape[-1] + 16,
        dtype=kv_packed.dtype,
        device=kv_packed.device,
    )
    kv[..., : kv_packed.shape[-1]] = kv_packed

    output = torch.zeros(128, 64, d_v, dtype=torch.bfloat16, device=q.device)
    out_lse = torch.zeros(128, 64, dtype=torch.float32, device=q.device)
    with pytest.raises(RuntimeError, match="tightly packed rows"):
        sparse_mla_sm120_paged_attention(
            q, kv, indices, output, out_lse, sm_scale, d_v=d_v
        )


@pytest.mark.parametrize("num_heads", [64, 128])
def test_sparse_mla_sm120_prefill_impl_auto_matches_swapab(num_heads: int) -> None:
    """Auto dispatch keeps preferring swapAB where it is instantiated."""
    case = _make_dsv3_2_prefill_case(num_heads, num_tokens=128)
    out_auto, lse_auto = _run_prefill_impl(case, None)
    out_swapab, lse_swapab = _run_prefill_impl(case, "swapab")
    torch.testing.assert_close(out_auto, out_swapab, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_auto, lse_swapab, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "num_heads,topk,match",
    [
        (32, 2048, "num_heads"),
        (64, 1000, "topk"),  # ragged width: not a whole number of index tiles
    ],
)
def test_sparse_mla_sm120_prefill_impl_swapab_ineligible_dsv3_2(
    num_heads: int, topk: int, match: str
) -> None:
    """Forced swapAB rejects out-of-envelope DSv3.2 shapes in Python."""
    case = _make_dsv3_2_prefill_case(num_heads, num_tokens=128, topk=topk)
    with pytest.raises(ValueError, match=match):
        _run_prefill_impl(case, "swapab")


@pytest.mark.parametrize("num_heads", [8, 64])
def test_sparse_mla_sm120_decode_form_prefill_fallback(num_heads: int) -> None:
    """A decode-form call at DSv4 topk=2048 — historically prefill-routed
    (the old decode sets stopped at topk=1024) — is served by the
    runtime-topk decode kernel under the decode-first default and matches
    the reference."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, topk = 32, 2048
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

    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    # Runtime-topk decode requires the caller-side split-K scratch.
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)
    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_crossover_routing_spy(monkeypatch) -> None:
    """Injected crossover (decode_max_tokens=8): T=8 routes to the decode
    kernel, T=16 routes to prefill; both match the reference."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla import _sparse_mla_sm120_cpb as cpb_mod
    from flashinfer.mla import _sparse_mla_sm120_plan as plan_mod

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, topk = 64, 512
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
    sm_scale = d_qk**-0.5

    # Inject the crossover table in-process and invalidate the planner memo.
    dev_key = cpb_mod._device_key(device)
    monkeypatch.setattr(cpb_mod, "_maybe_load_disk", lambda: None)
    monkeypatch.setitem(cpb_mod._crossover, dev_key, {"dsv4|64|512": 8})
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)
    plan_mod._plan_memo.clear()

    real_decode = sm.sparse_mla_sm120_decode_dsv4
    calls = {"decode": 0}

    def spy(*args, **kwargs):
        calls["decode"] += 1
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(sm, "sparse_mla_sm120_decode_dsv4", spy)

    runner = sm._SparseMLAPagedAttentionRunner()
    for num_tokens, expect_decode in ((8, True), (16, False)):
        q = (
            torch.randn(
                num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16
            )
            / 10.0
        ).clamp(-1, 1)
        indices = torch.randint(
            0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
        )
        indices[:, topk // 2 :] = -1
        ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)
        output = torch.zeros(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
        )
        calls["decode"] = 0
        out_lse = runner.run(q, kv_packed, indices, output, sm_scale, return_lse=True)
        assert (calls["decode"] == 1) == expect_decode
        torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_crossover_cuda_graph(monkeypatch) -> None:
    """Crossover dispatch under CUDA graphs (dsv4, H=128, topk=1024): with an
    injected decode_max_tokens=16 a T=8 capture bakes in decode split-K and a
    T=32 capture bakes in prefill; both replay correctly on fresh data. A T=8
    capture without crossover constants pins the uncalibrated decode default."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla import _sparse_mla_sm120_cpb as cpb_mod
    from flashinfer.mla import _sparse_mla_sm120_plan as plan_mod

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, topk = 128, 1024
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size
    num_splits = sm._decode_dsv4_num_splits(topk)

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)
    sm_scale = d_qk**-0.5

    dev_key = cpb_mod._device_key(device)
    monkeypatch.setattr(cpb_mod, "_maybe_load_disk", lambda: None)

    real_decode = sm.sparse_mla_sm120_decode_dsv4
    calls = {"decode": 0}

    def spy(*args, **kwargs):
        calls["decode"] += 1
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(sm, "sparse_mla_sm120_decode_dsv4", spy)
    runner = sm._SparseMLAPagedAttentionRunner()

    def fresh_inputs(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        q = (
            torch.randn(
                num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16
            )
            / 10.0
        ).clamp(-1, 1)
        indices = torch.randint(
            0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
        )
        indices[:, topk // 2 :] = -1
        return q, indices

    def capture(num_tokens: int):
        # Everything the replay path touches — static buffers, the fresh
        # replay payload, and its eager reference — is allocated BEFORE
        # capture: the captured call performs a small internal allocation
        # whose block would otherwise be recycled into post-capture tensors
        # that g.replay() then overwrites.
        q_s, idx_s = fresh_inputs(num_tokens)
        q_new, idx_new = fresh_inputs(num_tokens)
        ref_out, ref_lse = _ref_sparse_attn(q_new, kv_dequant, idx_new, sm_scale, d_v)
        out_s = torch.zeros(
            num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
        )
        lse_s = torch.zeros(num_tokens, num_heads, dtype=torch.float32, device=device)
        mid_o = torch.empty(
            num_tokens, num_heads, num_splits, d_v, dtype=torch.bfloat16, device=device
        )
        mid_l = torch.empty(
            num_tokens, num_heads, num_splits, dtype=torch.float32, device=device
        )

        def run() -> None:
            runner.run(
                q_s,
                kv_packed,
                idx_s,
                out_s,
                sm_scale,
                out_lse=lse_s,
                mid_out=mid_o,
                mid_lse=mid_l,
            )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(s)
        calls["decode"] = 0
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        return g, q_s, idx_s, out_s, lse_s, q_new, idx_new, ref_out, ref_lse

    def replay_and_check(
        g, q_s, idx_s, out_s, lse_s, q_new, idx_new, ref_out, ref_lse
    ) -> None:
        q_s.copy_(q_new)
        idx_s.copy_(idx_new)
        out_s.zero_()
        lse_s.zero_()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out_s, ref_out, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(lse_s, ref_lse, atol=5e-2, rtol=5e-2)

    # No crossover constants: the decode-first default survives capture.
    monkeypatch.delitem(cpb_mod._crossover, dev_key, raising=False)
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)
    plan_mod._plan_memo.clear()
    res = capture(8)
    assert calls["decode"] == 1
    replay_and_check(*res)

    # Injected crossover decode_max_tokens=16: T=8 -> decode, T=32 -> prefill.
    monkeypatch.setitem(cpb_mod._crossover, dev_key, {"dsv4|128|1024": 16})
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)
    plan_mod._plan_memo.clear()
    res = capture(8)
    assert calls["decode"] == 1
    replay_and_check(*res)
    res = capture(32)
    assert calls["decode"] == 0
    replay_and_check(*res)


def test_sparse_mla_sm120_runner_scratch_follows_routing(monkeypatch) -> None:
    """Runner-internal split-K scratch is allocated only when the call routes
    to a decode kernel, and is cached on the runner (grown on demand): with an
    injected decode_max_tokens=8 crossover, T=16 routes to prefill and leaves
    the scratch untouched, T=4/T=8 decode calls allocate and grow it, and a
    smaller repeat call reuses the grown buffers."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla import _sparse_mla_sm120_cpb as cpb_mod
    from flashinfer.mla import _sparse_mla_sm120_plan as plan_mod

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, topk = 64, 512
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
    sm_scale = d_qk**-0.5

    # Inject the crossover table in-process and invalidate the planner memo.
    dev_key = cpb_mod._device_key(device)
    monkeypatch.setattr(cpb_mod, "_maybe_load_disk", lambda: None)
    monkeypatch.setitem(cpb_mod._crossover, dev_key, {"dsv4|64|512": 8})
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)
    plan_mod._plan_memo.clear()

    runner = sm._SparseMLAPagedAttentionRunner()

    def call(num_tokens: int) -> None:
        q = (
            torch.randn(
                num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16
            )
            / 10.0
        ).clamp(-1, 1)
        indices = torch.randint(
            0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
        )
        indices[:, topk // 2 :] = -1
        ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)
        output = torch.zeros(
            (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
        )
        out_lse = runner.run(q, kv_packed, indices, output, sm_scale, return_lse=True)
        torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)

    call(16)  # past the crossover: prefill-routed, no scratch allocated
    assert runner._mid_out is None
    assert runner._mid_lse is None

    call(4)  # decode-routed: allocates the cached scratch
    assert runner._mid_out is not None
    small = runner._mid_out
    assert small.shape[0] == 4

    call(8)  # decode-routed, larger: the cache grows
    assert runner._mid_out is not small
    grown = runner._mid_out
    assert grown.shape[0] == 8

    call(4)  # fits the grown buffers: reused, not reallocated
    assert runner._mid_out is grown

    call(16)  # prefill-routed again: scratch untouched
    assert runner._mid_out is grown


def test_sparse_mla_sm120_runner_internal_scratch_cuda_graph(monkeypatch) -> None:
    """CUDA graph capture with runner-internal split-K scratch (dsv4, H=128,
    topk=1024, T=8): the warmup calls allocate the cached buffers so capture
    itself performs no scratch allocation, and replay on fresh data matches
    the eager reference."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla import _sparse_mla_sm120_cpb as cpb_mod
    from flashinfer.mla import _sparse_mla_sm120_plan as plan_mod

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, topk = 128, 1024
    d_qk, d_v = 512, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size
    num_tokens = 8

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)
    sm_scale = d_qk**-0.5

    # Pin the uncalibrated decode-first policy.
    dev_key = cpb_mod._device_key(device)
    monkeypatch.setattr(cpb_mod, "_maybe_load_disk", lambda: None)
    monkeypatch.delitem(cpb_mod._crossover, dev_key, raising=False)
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)
    plan_mod._plan_memo.clear()

    runner = sm._SparseMLAPagedAttentionRunner()

    def fresh_inputs() -> tuple[torch.Tensor, torch.Tensor]:
        q = (
            torch.randn(
                num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16
            )
            / 10.0
        ).clamp(-1, 1)
        indices = torch.randint(
            0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
        )
        indices[:, topk // 2 :] = -1
        return q, indices

    # Everything the replay path touches — static buffers, the fresh replay
    # payload, and its eager reference — is allocated BEFORE capture: the
    # captured call performs a small internal allocation whose block would
    # otherwise be recycled into post-capture tensors that g.replay() then
    # overwrites.
    q_s, idx_s = fresh_inputs()
    q_new, idx_new = fresh_inputs()
    ref_out, ref_lse = _ref_sparse_attn(q_new, kv_dequant, idx_new, sm_scale, d_v)
    out_s = torch.zeros(num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device)
    lse_s = torch.zeros(num_tokens, num_heads, dtype=torch.float32, device=device)

    def run() -> None:
        runner.run(q_s, kv_packed, idx_s, out_s, sm_scale, out_lse=lse_s)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    assert runner._mid_out is not None  # scratch allocated pre-capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()

    q_s.copy_(q_new)
    idx_s.copy_(idx_new)
    out_s.zero_()
    lse_s.zero_()
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_s, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_s, ref_lse, atol=5e-2, rtol=5e-2)


# ── Envelope consistency: C++ accepts exactly what the planner claims ─────

# (variant, model_type, num_heads, topk, page_block_size, has_extra)
_ENVELOPE_PROBES = [
    # PREFILL_SWAPAB: DSV3_2 family, H in {64,128}, any whole-tile topk,
    # single cache.
    ("swapab", 0, 64, 2048, 64, False),
    ("swapab", 2, 128, 2048, 64, False),
    ("swapab", 0, 64, 512, 64, False),  # runtime topk: 512 is served
    ("swapab", 0, 64, 1000, 64, False),  # ragged topk (not a whole tile)
    ("swapab", 0, 32, 2048, 64, False),  # H below swapAB
    ("swapab", 0, 64, 2048, 32, False),  # pbs mismatch
    ("swapab", 1, 64, 2048, 64, False),  # DSV4 has no swapAB
    ("swapab", 0, 64, 2048, 64, True),  # dual-cache
    # PREFILL_SG: DSV3_2 family, H in {8,16}, any whole-tile topk.
    ("sg", 0, 8, 2048, 64, False),
    ("sg", 2, 16, 2048, 64, False),
    ("sg", 0, 16, 512, 64, False),  # runtime topk: 512 is served
    ("sg", 0, 16, 63, 64, False),  # ragged topk (not a whole tile)
    ("sg", 0, 32, 2048, 64, False),  # H above SG
    ("sg", 1, 8, 2048, 64, False),  # DSV4 has no SG
    # PREFILL_MG: DSV3_2 family H in {32,64,128} and DSV4 H in {8..128}, any
    # whole-tile topk.
    ("mg", 0, 32, 2048, 64, False),
    ("mg", 2, 128, 2048, 64, False),
    ("mg", 0, 16, 2048, 64, False),  # v32 H=16 is SG territory
    ("mg", 0, 64, 1024, 64, False),  # runtime topk: v32 serves 1024 too
    ("mg", 1, 8, 128, 64, False),
    ("mg", 1, 128, 2048, 64, False),
    ("mg", 1, 64, 192, 64, False),
    ("mg", 1, 64, 384, 64, False),  # runtime topk: between the old pins
    ("mg", 1, 64, 100, 64, False),  # ragged topk (not a whole tile)
    ("mg", 1, 17, 128, 64, False),  # dsv4 H off boundary
    ("mg", 1, 64, 512, 32, False),  # pbs mismatch
    ("mg", 1, 64, 512, 64, True),  # dual-cache must use MG_DUAL
    # PREFILL_MG_DUAL: DSV4 only, any whole-tile topk, extra cache present.
    ("mg_dual", 1, 32, 128, 64, True),
    ("mg_dual", 1, 128, 128, 64, True),
    ("mg_dual", 0, 32, 128, 64, True),  # dual is DSV4-only
    ("mg_dual", 1, 32, 256, 64, True),  # runtime topk: dual serves 256 too
    ("mg_dual", 1, 32, 128, 64, False),  # requires the extra cache
    # GLM53_NOPE: V32 family at any whole-tile topk (swapAB included).
    ("sg", 3, 8, 2176, 64, False),
    ("sg", 3, 16, 2048, 64, False),  # runtime topk: 2048 is served too
    ("mg", 3, 32, 2176, 64, False),
    ("mg", 3, 128, 2176, 64, False),
    ("mg", 3, 64, 2048, 64, False),
    ("swapab", 3, 64, 2176, 64, False),
    ("swapab", 3, 128, 2176, 64, False),
    ("swapab", 3, 32, 2176, 64, False),  # H below swapAB
    ("swapab", 3, 64, 2048, 64, False),  # runtime topk: 2048 is served too
    ("mg_dual", 3, 32, 128, 64, True),  # dual is DSV4-only
    # DOTS3_SWA: SG-only, H in {8,16,32,64}, whole-tile topk >= 513.
    ("sg", 4, 8, 576, 64, False),
    ("sg", 4, 64, 576, 64, False),
    ("sg", 4, 64, 640, 64, False),  # runtime topk: wider than 576 is served
    ("sg", 4, 128, 576, 64, False),  # H above the DOTS3_SWA SG set
    ("sg", 4, 64, 512, 64, False),  # below the 513 sliding-window floor
    ("sg", 4, 64, 576, 32, False),  # pbs mismatch
    ("mg", 4, 32, 576, 64, False),  # DOTS3_SWA is SG-only (no MG)
    ("swapab", 4, 64, 576, 64, False),  # and no swapAB
    ("mg_dual", 4, 64, 576, 64, True),  # dual is DSV4-only
]

_VARIANT_ENUM = {
    "swapab": ("PREFILL_SWAPAB", "prefill_swapab_eligible"),
    "sg": ("PREFILL_SG", "prefill_sg_eligible"),
    "mg": ("PREFILL_MG", "prefill_mg_eligible"),
    ("mg_dual"): ("PREFILL_MG_DUAL", "prefill_mg_dual_eligible"),
}


@pytest.mark.parametrize(
    "variant,model_type,num_heads,topk,page_block_size,has_extra", _ENVELOPE_PROBES
)
def test_sparse_mla_sm120_envelope_consistency(
    variant: str,
    model_type: int,
    num_heads: int,
    topk: int,
    page_block_size: int,
    has_extra: bool,
) -> None:
    """For each probed boundary point, the C++ variant dispatch accepts iff
    the Python envelope predicate claims eligibility."""
    from flashinfer.mla import _sparse_mla_sm120_plan as plan_mod
    from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

    device = torch.device("cuda")
    enum_name, predicate_name = _VARIANT_ENUM[variant]
    variant_id = int(getattr(plan_mod.KernelVariant, enum_name))
    expected = getattr(plan_mod, predicate_name)(
        model_type, num_heads, topk, page_block_size, has_extra
    )

    d_qk = 576 if model_type in (0, 2) else (1088 if model_type == 4 else 512)
    bpt = 656 if model_type in (0, 2, 3) else (1160 if model_type == 4 else 584)
    d_v = 1024 if model_type == 4 else 512
    num_tokens = 2
    kv_cache = torch.zeros(4, page_block_size * bpt, dtype=torch.uint8, device=device)
    q = torch.zeros(num_tokens, num_heads, d_qk, dtype=torch.bfloat16, device=device)
    indices = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    output = torch.zeros(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros(num_tokens, num_heads, dtype=torch.float32, device=device)
    extra_kv = extra_idx = None
    if has_extra:
        extra_kv = torch.zeros(4, 64 * bpt, dtype=torch.uint8, device=device)
        extra_idx = torch.zeros(num_tokens, 64, dtype=torch.int32, device=device)

    module = _get_sparse_mla_sm120_decode_module()

    def call() -> None:
        module.sparse_mla_sm120_paged_attention(
            q,
            kv_cache,
            indices,
            output,
            out_lse,
            d_qk**-0.5,
            model_type,
            variant_id,
            None,
            None,
            extra_kv,
            extra_idx,
            None,
        )

    if expected:
        call()
        torch.cuda.synchronize()
    else:
        with pytest.raises(RuntimeError, match="sparse-MLA"):
            call()
