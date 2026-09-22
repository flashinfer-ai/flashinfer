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

from tests.attention.sparse_mla_test_utils import (
    _assert_has_non_pow2_inline_scales,
    _make_decode_scratch,
    _ref_sparse_attn,
    dequantize_kv_dots3_swa,
    dequantize_kv_dsv3_2,
    dequantize_kv_dsv4,
    dequantize_kv_dsv4_1,
    dequantize_kv_dsv4_1_fp4,
    quantize_kv_dots3_swa,
    quantize_kv_dsv3_2,
    quantize_kv_dsv4,
    quantize_kv_dsv4_1,
    quantize_kv_dsv4_1_fp4,
    quantize_kv_glm53_nope,
    quantize_kv_glm_nsa,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")),
    reason="Sparse-MLA SM120 requires SM12x.",
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
    indices[0] = -1
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

    with pytest.raises(ValueError, match="tightly packed"):
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


def test_sparse_mla_sm120_decode_dsv4_padded_out_lse() -> None:
    """A capacity LSE buffer larger than [num_tokens, num_heads] works through
    the functional entry points: inspect accepts capacity views while execute
    requires the exact [T, H] shape, so the prepared layer slices the buffer
    (wrapper behavior) instead of failing the execute-side check."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size, num_blocks, topk = 64, 64, 640
    num_tokens, num_heads = 8, 64
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

    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)
    padded_lse = torch.full(
        (num_tokens + 3, 128), float("nan"), dtype=torch.float32, device=device
    )
    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        padded_lse,
        sm_scale,
        d_v=d_v,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        padded_lse[:num_tokens, :num_heads], ref_lse, atol=5e-2, rtol=5e-2
    )
    # The padding rows and columns are never written.
    assert torch.isnan(padded_lse[num_tokens:]).all()
    assert torch.isnan(padded_lse[:, num_heads:]).all()

    # The caller-workspace functional entry point slices the same way.
    from flashinfer.mla._sparse_mla_sm120._prepared import functional_run

    workspace = torch.empty(8 << 20, dtype=torch.uint8, device=device)
    padded_lse2 = torch.full(
        (num_tokens + 2, 128), float("nan"), dtype=torch.float32, device=device
    )
    output2 = torch.zeros_like(output)
    result = functional_run(
        q,
        kv_packed,
        indices,
        output2,
        workspace,
        sm_scale,
        lse=padded_lse2,
    )
    assert result is not None and result.shape == (num_tokens, num_heads)
    torch.testing.assert_close(output2, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(
        padded_lse2[:num_tokens, :num_heads], ref_lse, atol=5e-2, rtol=5e-2
    )
    assert torch.isnan(padded_lse2[num_tokens:]).all()
    assert torch.isnan(padded_lse2[:, num_heads:]).all()


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

    with pytest.raises(ValueError, match="indices shape/stride mismatch"):
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
    """Compiled metadata validation rejects heads outside every route."""
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

    with pytest.raises(ValueError, match="unsupported tokens/heads"):
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


@pytest.mark.parametrize("num_tokens,num_heads", [(4, 32), (65, 32), (128, 64)])
def test_sparse_mla_sm120_glm53_nope_compact_rows(
    num_tokens: int, num_heads: int
) -> None:
    """Compact and strided 528B rows match poisoned 656B rows, including replay.

    Shapes cover decode (T=4), prefill MG (T=65, H=32) and prefill swapAB
    (T=128, H=64), with variable lengths and the final cache slot selected.
    """
    torch.manual_seed(5)
    device = torch.device("cuda")
    d_qk = d_v = 512
    page_block_size, num_blocks, topk = 64, 64, 2176
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    packed_656 = quantize_kv_glm53_nope(kv_bf16)  # [nb, pbs, 1, 656]
    packed_528 = packed_656[..., :528].contiguous()
    packed_656[..., 528:].fill_(0xFF)
    sliced_528 = packed_656[..., :528]  # 528-wide view, row stride stays 656

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    indices[:, 0] = s_kv - 1
    counts = torch.arange(num_tokens, device=device, dtype=torch.int32) % 3
    lengths = torch.where(counts == 0, 1, torch.where(counts == 1, 70, topk)).to(
        torch.int32
    )
    indices.masked_fill_(
        torch.arange(topk, device=device)[None, :] >= lengths[:, None], -1
    )
    sm_scale = d_qk**-0.5
    mid_out, mid_lse = _make_decode_scratch(num_tokens, num_heads, topk, d_v, device)
    caches = {
        "padded-656": packed_656,
        "packed-528": packed_528,
        "sliced-528": sliced_528,
    }
    results = {
        name: (
            torch.empty_like(q),
            torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device),
        )
        for name in caches
    }

    def run(name: str) -> None:
        output, out_lse = results[name]
        sparse_mla_sm120_paged_attention(
            q,
            caches[name],
            indices,
            output,
            out_lse,
            sm_scale,
            d_v=d_v,
            kv_scale_format="arbitrary_fp32",
            topk_length=lengths,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    for name in caches:
        run(name)
    ref_out, ref_lse = results["padded-656"]
    for name in ("packed-528", "sliced-528"):
        out, lse = results[name]
        torch.testing.assert_close(out, ref_out, atol=0, rtol=0)
        torch.testing.assert_close(lse, ref_lse, atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run("packed-528")
        run("sliced-528")
    q.mul_(0.75)
    graph.replay()
    run("padded-656")
    for name in ("packed-528", "sliced-528"):
        out, lse = results[name]
        torch.testing.assert_close(out, ref_out, atol=0, rtol=0)
        torch.testing.assert_close(lse, ref_lse, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens,num_heads", [(4, 32), (65, 32), (128, 64)])
def test_sparse_mla_sm120_glm53_nope_masked_rows_ignore_poisoned_slot_zero(
    num_tokens: int, num_heads: int
) -> None:
    """Masked (-1) candidates gather a zero row, never mutable cache slot 0.

    Slot 0 is poisoned with NaNs (values and scales); valid indices exclude
    slot 0. A kernel that clamps masked indices to slot 0 would leak NaN into
    valid outputs through 0 * NaN in the value MMA.
    """
    torch.manual_seed(6)
    device = torch.device("cuda")
    d_qk = d_v = 512
    page_block_size, num_blocks, topk = 64, 64, 2176
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_glm53_nope(kv_bf16)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)[..., :d_qk]
    # Poison slot 0 after computing the reference: NaN values + NaN scales.
    kv_packed.view(-1, 656)[0].fill_(0xFF)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        1, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
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

    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


# ── DeepSeek-V4.1 (DSV4_1) ────────────────────────────────────────────────
# 528B/token: 512B all-FP8 K (rope lanes quantized, no BF16 rope segment) +
# 16B footer of 16 UE8M0 scales over 32-wide groups. Selected explicitly via
# kv_scale_format="ue8m0_g32" (d_qk=512 collides with DSV4; the 528B payload
# collides with GLM53_NOPE).

_DSV4_1_DECODE_CONFIGS = [
    (8, 512),  # dedicated instantiation, the V4.1 indexer topk
    (64, 512),
    (128, 512),
    (24, 512),  # runtime-H instantiation (in-block pad path)
    (64, 500),  # partial tail chunk (runtime topk width)
]


@pytest.mark.parametrize("num_heads,topk", _DSV4_1_DECODE_CONFIGS)
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_mla_sm120_decode_dsv4_1(
    num_heads: int, topk: int, num_tokens: int, with_sink: bool
) -> None:
    """DeepSeek-V4.1 decode (decode-dsv4 tile, BI=64, pair-folded XV)."""
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
    kv_packed = quantize_kv_dsv4_1(kv_bf16)
    kv_dequant = dequantize_kv_dsv4_1(kv_packed)

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
        kv_scale_format="ue8m0_g32",
        attn_sink=attn_sink,
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("num_heads", [16, 64])
def test_sparse_mla_sm120_prefill_dsv4_1(num_heads: int) -> None:
    """DeepSeek-V4.1 prefill: SG-only on the BI=32 producer/consumer tile;
    H=64 rides CTA replication. num_tokens=65 forces the prefill route."""
    torch.manual_seed(5)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    num_tokens, topk = 65, 512
    page_block_size = 64
    num_blocks = 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4_1(kv_bf16)
    kv_dequant = dequantize_kv_dsv4_1(kv_packed)

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
        kv_scale_format="ue8m0_g32",
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "extra_fp4,main_pbs,extra_pbs,num_heads",
    [
        (False, 64, 2, 16),
        (False, 256, 64, 8),
        (False, 61, 53, 16),
        (True, 64, 2, 8),
        (True, 256, 64, 32),
        (True, 61, 53, 16),
    ],
)
@pytest.mark.parametrize("nt", [4, 65])
def test_sparse_mla_sm120_prefill_dsv4_1_dual_replay(
    extra_fp4: bool,
    main_pbs: int,
    extra_pbs: int,
    num_heads: int,
    nt: int,
    verify_symbol: bool = False,
) -> None:
    torch.manual_seed(17)
    device = torch.device("cuda")
    topk, extra_topk, dim = 128, 77, 512
    main_latent = (
        torch.randn(5, main_pbs, 1, dim, device=device, dtype=torch.bfloat16) * 0.3
    )
    extra_latent = (
        torch.randn(
            256 // extra_pbs, extra_pbs, 1, dim, device=device, dtype=torch.bfloat16
        )
        * 0.3
    )
    main = quantize_kv_dsv4_1(main_latent)
    extra = (quantize_kv_dsv4_1_fp4 if extra_fp4 else quantize_kv_dsv4_1)(extra_latent)
    extra_dequant = (dequantize_kv_dsv4_1_fp4 if extra_fp4 else dequantize_kv_dsv4_1)(
        extra
    )
    virtual_kv = torch.cat(
        [dequantize_kv_dsv4_1(main).reshape(-1, dim), extra_dequant.reshape(-1, dim)]
    ).reshape(-1, 1, 1, dim)

    def pitched(cache: torch.Tensor) -> torch.Tensor:
        nb, pbs, _, bpt = cache.shape
        stride = ((pbs * bpt + 511) // 512 + 1) * 512
        storage = torch.empty((nb, stride), device=device, dtype=torch.uint8)
        view = storage[:, : pbs * bpt].view(nb, pbs, 1, bpt)
        view.copy_(cache)
        return view.view(nb, pbs * bpt) if extra_pbs == 64 else view

    main_slots, extra_slots = (
        main_latent.shape[0] * main_pbs,
        extra_latent.shape[0] * extra_pbs,
    )
    main, extra = pitched(main), pitched(extra)
    q = torch.randn(nt, num_heads, dim, device=device, dtype=torch.bfloat16) * 0.3
    idx = torch.randint(0, main_slots, (nt, topk), device=device, dtype=torch.int32)
    exidx = torch.randint(
        0, extra_slots, (nt, extra_topk), device=device, dtype=torch.int32
    )
    lengths = torch.full((nt,), 69, device=device, dtype=torch.int32)
    exlengths = torch.full((nt,), 75, device=device, dtype=torch.int32)
    lengths[0], exlengths[1] = 0, 0
    lengths[2], exlengths[2] = 0, 0
    idx[:, 3:35] = -1
    exidx[:, 1:33] = -1
    idx[:, 69:] = 2**30
    exidx[:, 75:] = 2**30
    idx_next, exidx_next = idx.clone(), exidx.clone()
    idx_next[:, :3] = 7
    exidx_next[:, 33:65] = 13
    sink = torch.randn(num_heads, device=device)
    out = torch.empty_like(q)
    lse_storage = torch.empty(nt, num_heads + 3, device=device)
    lse = lse_storage[:, :num_heads]

    def reference(
        main_idx: torch.Tensor, extra_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mi = main_idx.masked_fill(
            torch.arange(topk, device=device)[None] >= lengths[:, None], -1
        )
        ei = extra_idx.masked_fill(
            torch.arange(extra_topk, device=device)[None] >= exlengths[:, None], -1
        )
        vi = torch.cat([mi, torch.where(ei < 0, ei, ei + main_slots)], dim=-1)
        return _ref_sparse_attn(q, virtual_kv, vi, dim**-0.5, dim, attn_sink=sink)

    refs = [reference(idx, exidx), reference(idx_next, exidx_next)]
    mid_out, mid_lse = _make_decode_scratch(
        nt, num_heads, topk, dim, device, extra_topk=extra_topk
    )

    def run() -> None:
        sparse_mla_sm120_paged_attention(
            q,
            main,
            idx,
            out,
            lse,
            dim**-0.5,
            kv_scale_format="ue8m0_g32",
            topk_length=lengths,
            attn_sink=sink,
            extra_kv_cache=extra,
            extra_indices=exidx,
            extra_topk_length=exlengths,
            extra_fp4=extra_fp4,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    run()
    torch.cuda.synchronize()
    if verify_symbol:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            run()
        schedule = "Dsv41MixedCache" if extra_fp4 else ("Fp8" if nt <= 64 else "Dsv41")
        kind = "Decode" if nt <= 64 else "Prefill"
        assert any(schedule + kind in event.name for event in prof.events())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for iteration, (ref_out, ref_lse) in enumerate(refs):
        if iteration:
            idx.copy_(idx_next)
            exidx.copy_(exidx_next)
        graph.replay()
        # Mixed-cache thresholds match NVFP4; the reference still uses BF16 Q.
        torch.testing.assert_close(
            out, ref_out, atol=5e-2 if extra_fp4 else 1e-2, rtol=5e-2
        )
        relative_rms = (
            out.float() - ref_out.float()
        ).square().mean().sqrt() / ref_out.float().square().mean().sqrt()
        assert relative_rms < 0.05
        lse_tol = 2e-2 if extra_fp4 else 3e-3
        torch.testing.assert_close(lse, ref_lse, atol=lse_tol, rtol=lse_tol)


@pytest.mark.parametrize("nt", [4, 65])
@pytest.mark.parametrize("extra_fp4", [False, True])
def test_sparse_mla_sm120_dsv4_1_kernel_identity(nt: int, extra_fp4: bool) -> None:
    test_sparse_mla_sm120_prefill_dsv4_1_dual_replay(
        extra_fp4, 256, 64, 16, nt, verify_symbol=True
    )


@pytest.mark.parametrize("num_tokens", [4, 65])
def test_sparse_mla_sm120_dsv4_1_dual(num_tokens: int) -> None:
    """Public all-FP8 dual-cache decode and prefill."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads = 64
    topk, extra_topk = 128, 512
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
    main_packed = quantize_kv_dsv4_1(main_bf16)
    extra_packed = quantize_kv_dsv4_1(extra_bf16)
    main_dequant = dequantize_kv_dsv4_1(main_packed)
    extra_dequant = dequantize_kv_dsv4_1(extra_packed)

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
        kv_cache_format="fp8_dsv41",
    )

    torch.testing.assert_close(output.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)


def test_dsv41_fp4_quantize_pack_matches_reference() -> None:
    """The CUDA pack kernel reproduces the FlashMLA V41_FP4 torch trajectory
    bit-exactly, including the scale clamps and the NaN-poison convention."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    nb, bs = 4, 8
    latent = (
        torch.randn(nb, bs, 1, 512, device=device, dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)
    latent[1, 0] *= 1000.0  # large-magnitude group hits the amax/6 <= 448 clamp
    latent[2, 1] = 1e-5  # tiny group hits the 2^-9 scale floor
    latent[3, 2, 0, 0] = float("nan")  # poisons its 16-wide group only

    cache = flashinfer.mla.dsv41_fp4_quantize_pack_sparse_mla_cache(
        latent.squeeze(2), kv_layout="NHD"
    )
    ref = quantize_kv_dsv4_1_fp4(latent)
    assert cache.shape == ref.shape
    torch.testing.assert_close(
        cache.view(torch.uint8), ref.view(torch.uint8), atol=0, rtol=0
    )


def test_dsv41_fp4_quantize_append_matches_pack() -> None:
    """Append-by-slot lands the same bytes as a full pack; a duplicated slot
    resolves to the lowest-index input row deterministically."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    nb, bs = 4, 8
    latent = (
        torch.randn(nb, bs, 1, 512, device=device, dtype=torch.bfloat16) / 10.0
    ).clamp(-1, 1)

    full = flashinfer.mla.dsv41_fp4_quantize_pack_sparse_mla_cache(
        latent.squeeze(2), kv_layout="NHD"
    )
    appended = torch.zeros_like(full)
    # Scatter tokens into arbitrary slots, duplicating one slot: token 5 and
    # token 9 both map to slot 3; token 5 must win. -1 marks padding.
    flat = latent.reshape(-1, 512)
    slot_mapping = torch.randperm(nb * bs, device=device, dtype=torch.int32)
    slot_mapping[9] = slot_mapping[5]
    slot_mapping[12] = -1
    flashinfer.mla.dsv41_fp4_quantize_append_sparse_mla_cache(
        flat, slot_mapping, appended
    )
    slots = slot_mapping.tolist()
    for tok, slot in enumerate(slots):
        if slot < 0 or slot in slots[:tok]:
            continue  # padding, or a duplicate slot this token does not win
        # full packs token tok at position tok; appended stores it at `slot`.
        # Within a page block, entry e's data row is at e*256 and its scale
        # row at bs*256 + e*32 (the [pbs, 1, 288] shape is cosmetic).
        page_s, entry_s = divmod(slot, bs)
        page_t, entry_t = divmod(tok, bs)
        app_page = appended[page_s].reshape(-1)
        full_page = full[page_t].reshape(-1)
        data_s, data_t = entry_s * 256, entry_t * 256
        sc_s, sc_t = bs * 256 + entry_s * 32, bs * 256 + entry_t * 32
        torch.testing.assert_close(
            app_page[data_s : data_s + 256],
            full_page[data_t : data_t + 256],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            app_page[sc_s : sc_s + 32],
            full_page[sc_t : sc_t + 32],
            atol=0,
            rtol=0,
        )


def test_sparse_mla_sm120_decode_dsv4_fp4_extra_requires_dsv4_1() -> None:
    """extra_fp4 is defined only with a DSV4_1 main cache (FlashMLA constraint);
    the binding rejects a DSV4 main cache with a readable error."""
    from flashinfer.mla import _sparse_mla_sm120 as sm

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads, topk, extra_topk = 2, 16, 128, 128
    d_qk, d_v = 512, 512
    pbs = 64
    num_blocks, extra_blocks = 8, (extra_topk + 1) // 2

    main_bf16 = torch.randn(num_blocks, pbs, 1, d_qk, device=device).to(torch.bfloat16)
    extra_bf16 = torch.randn(extra_blocks, 2, 1, d_qk, device=device).to(torch.bfloat16)
    main_packed = quantize_kv_dsv4(main_bf16)  # DSV4 main, NOT DSV4_1
    extra_packed = quantize_kv_dsv4_1_fp4(extra_bf16)
    q = torch.randn(num_tokens, num_heads, d_qk, device=device).to(torch.bfloat16)
    idx = torch.randint(0, num_blocks * pbs, (num_tokens, topk), device=device).int()
    extra_idx = torch.randint(
        0, extra_blocks * 2, (num_tokens, extra_topk), device=device
    ).int()
    output = torch.zeros(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros(num_tokens, num_heads, dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(
        num_tokens, num_heads, topk, d_v, device, extra_topk=extra_topk
    )

    with pytest.raises((ValueError, RuntimeError), match="requires DSV41 dual cache"):
        sm.sparse_mla_sm120_decode_dsv4(
            q,
            main_packed,
            idx,
            mid_out,
            mid_lse,
            output,
            out_lse,
            d_qk**-0.5,
            extra_kv_cache=extra_packed,
            extra_indices=extra_idx,
            model_type=sm._MODEL_TYPE_DSV4,
            extra_fp4=True,
        )


def test_sparse_mla_sm120_decode_dsv4_1_dual_fp4_extra(monkeypatch) -> None:
    """Mixed-cache decode against the dequantized-cache reference."""
    from flashinfer.mla._sparse_mla_sm120 import _api as sm

    def unexpected_cpb(*args, **kwargs):
        raise AssertionError("mixed decode reused FP8 calibration")

    monkeypatch.setattr(sm, "_resolve_cpb", unexpected_cpb)
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, num_heads = 4, 64
    topk, extra_topk = 128, 512
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
    main_packed = quantize_kv_dsv4_1(main_bf16)
    extra_packed = quantize_kv_dsv4_1_fp4(extra_bf16)
    main_dequant = dequantize_kv_dsv4_1(main_packed)
    extra_dequant = dequantize_kv_dsv4_1_fp4(extra_packed)

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

    output = torch.zeros(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros(num_tokens, num_heads, dtype=torch.float32, device=device)
    mid_out, mid_lse = _make_decode_scratch(
        num_tokens, num_heads, topk, d_v, device, extra_topk=extra_topk
    )
    sm.sparse_mla_sm120_decode_dsv4(
        q,
        main_packed,
        main_idx,
        mid_out,
        mid_lse,
        output,
        out_lse,
        sm_scale,
        extra_kv_cache=extra_packed,
        extra_indices=extra_idx,
        model_type=sm._MODEL_TYPE_DSV4_1,
        extra_fp4=True,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("num_tokens", [4, 65])
def test_sparse_mla_sm120_dsv4_1_dual_fp4_extra_trtllm_entry(num_tokens: int) -> None:
    """Public format selection reaches both mixed-cache kernel families."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    num_heads = 64
    topk, extra_topk = 128, 512
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
    main_packed = quantize_kv_dsv4_1(main_bf16)
    extra_packed = quantize_kv_dsv4_1_fp4(extra_bf16)
    main_dequant = dequantize_kv_dsv4_1(main_packed)
    extra_dequant = dequantize_kv_dsv4_1_fp4(extra_packed)

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
        kv_cache_format="fp8_dsv41_fp4_ca",
    )

    torch.testing.assert_close(output.squeeze(1), ref_out, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_decode_dsv4_1_masked_rows_ignore_poisoned_slot_zero() -> None:
    """DSV4_1 decode gathers the shared zero row for masked candidates: slot 0
    is poisoned with 0xFF (NaN FP8 values, +inf UE8M0 scales) and must not leak."""
    torch.manual_seed(8)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size, num_blocks, topk = 64, 64, 512
    num_tokens, num_heads = 16, 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4_1(kv_bf16)
    kv_dequant = dequantize_kv_dsv4_1(kv_packed)
    # Poison slot 0 (block 0 token 0): the 512B data row plus its 16B footer scale.
    flat = kv_packed.view(num_blocks, -1)
    flat[0, :512].fill_(0xFF)
    flat[0, page_block_size * 512 : page_block_size * 512 + 16].fill_(0xFF)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        1, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
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
        kv_scale_format="ue8m0_g32",
        mid_out=mid_out,
        mid_lse=mid_lse,
    )

    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_prefill_dsv4_1_masked_rows_ignore_poisoned_slot_zero() -> (
    None
):
    """The DSV4_1 prefill gather (512B bulk + 16B scale footer read) applies the
    same zero-row masking. num_tokens=128 forces the prefill route."""
    torch.manual_seed(10)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size, num_blocks, topk = 64, 64, 512
    num_tokens, num_heads = 128, 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4_1(kv_bf16)
    kv_dequant = dequantize_kv_dsv4_1(kv_packed)
    flat = kv_packed.view(num_blocks, -1)
    flat[0, :512].fill_(0xFF)
    flat[0, page_block_size * 512 : page_block_size * 512 + 16].fill_(0xFF)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        1, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
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
        kv_scale_format="ue8m0_g32",
    )

    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_decode_dsv4_masked_rows_ignore_poisoned_slot_zero() -> None:
    """The footer-scale decode kernel applies the same zero-row masking."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size, num_blocks, topk = 64, 64, 1024
    num_tokens, num_heads = 16, 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)
    # Poison slot 0 (block 0 token 0): data row plus its footer scale.
    flat = kv_packed.view(num_blocks, -1)
    flat[0, :576].fill_(0xFF)
    flat[0, page_block_size * 576 : page_block_size * 576 + 8].fill_(0xFF)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        1, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
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

    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_prefill_dsv4_masked_rows_ignore_poisoned_slot_zero() -> None:
    """The footer-scale prefill gather (data + scale footer) applies the same
    zero-row masking. num_tokens=128 forces the prefill route."""
    torch.manual_seed(9)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    page_block_size, num_blocks, topk = 64, 64, 1024
    num_tokens, num_heads = 128, 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv4(kv_bf16)
    kv_dequant = dequantize_kv_dsv4(kv_packed)
    # Poison slot 0 (block 0 token 0): data row plus its footer scale.
    flat = kv_packed.view(num_blocks, -1)
    flat[0, :576].fill_(0xFF)
    flat[0, page_block_size * 576 : page_block_size * 576 + 8].fill_(0xFF)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        1, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    indices[:, topk // 2 :] = -1
    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    sparse_mla_sm120_paged_attention(
        q, kv_packed, indices, output, out_lse, sm_scale, d_v=d_v
    )

    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_prefill_dsv3_2_masked_rows_ignore_poisoned_slot_zero() -> (
    None
):
    """The inline-scale prefill path with a real RoPE tail applies the same
    zero-row masking; a masked lane's rope read must not see the poisoned
    slot either. num_tokens=128 forces the prefill route."""
    torch.manual_seed(10)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    page_block_size, num_blocks, topk = 64, 64, 2048
    num_tokens, num_heads = 128, 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_dsv3_2(kv_bf16)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)
    # Poison slot 0: NaN values, NaN scales, NaN rope.
    kv_packed.view(-1, 656)[0].fill_(0xFF)

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        1, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    indices[:, topk // 2 :] = -1
    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    sparse_mla_sm120_paged_attention(
        q, kv_packed, indices, output, out_lse, sm_scale, d_v=d_v
    )

    assert torch.isfinite(output.float()).all()
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("num_heads", [32, 64])
def test_sparse_mla_sm120_prefill_glm53_nope_bounds_stale_padding(
    num_heads: int,
) -> None:
    """Prefill never gathers past topk_length, whatever the padding holds.

    Slots in [topk_length, topk) carry huge in-range-looking garbage rather
    than -1; a kernel reading them would gather a wild gmem address. The
    reference sees the same candidates with -1 padding. num_heads=32 routes
    to MG, 64 to swapAB.
    """
    torch.manual_seed(8)
    device = torch.device("cuda")
    d_qk = d_v = 512
    num_tokens, topk = 65, 2176
    topk_len = topk - 32  # partial last tile
    page_block_size, num_blocks = 64, 64
    s_kv = num_blocks * page_block_size

    kv_bf16 = (
        torch.randn(
            num_blocks, page_block_size, 1, d_qk, device=device, dtype=torch.bfloat16
        )
        / 10.0
    ).clamp(-1, 1)
    kv_packed = quantize_kv_glm53_nope(kv_bf16)
    kv_dequant = dequantize_kv_dsv3_2(kv_packed)[..., :d_qk]

    q = (
        torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16)
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    indices[:, topk_len:] = 1 << 30  # stale garbage, not -1
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
        kv_scale_format="arbitrary_fp32",
        topk_length=topk_length,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def _stale_padding_indices(
    num_tokens: int, topk: int, topk_len: int, s_kv: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indices with garbage (not -1) past topk_len, plus the -1-padded twin
    the reference consumes and the runtime topk_length."""
    indices = torch.randint(
        0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32
    )
    indices[:, topk_len:] = 1 << 30  # stale garbage, not -1
    topk_length = torch.full((num_tokens,), topk_len, dtype=torch.int32, device=device)
    ref_indices = indices.clone()
    ref_indices[:, topk_len:] = -1
    return indices, ref_indices, topk_length


@pytest.mark.parametrize("num_heads,prefill_impl", [(16, None), (32, "mg")])
def test_sparse_mla_sm120_prefill_dsv3_2_bounds_stale_padding(
    num_heads: int, prefill_impl
) -> None:
    """Stale-padding bounds for a rope model on the SG and MG prefill routes.

    The GLM53_NOPE stale-padding case has no rope tail; rope models
    additionally form gmem rope addresses from the raw indices on the math
    side (the QK rope operand loads are real loads, not prefetch hints), so
    stale positive padding must be normalized there as well, not only at the
    IO gather."""
    torch.manual_seed(12)
    device = torch.device("cuda")
    d_qk, d_v = 576, 512
    page_block_size, num_blocks, topk = 64, 64, 2048
    num_tokens = 128
    topk_len = topk - 32  # partial last tile
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
    indices, ref_indices, topk_length = _stale_padding_indices(
        num_tokens, topk, topk_len, s_kv, device
    )
    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, ref_indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    kwargs = {"prefill_impl": prefill_impl} if prefill_impl is not None else {}
    sparse_mla_sm120_paged_attention(
        q,
        kv_packed,
        indices,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        topk_length=topk_length,
        **kwargs,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("dual", [False, True])
def test_sparse_mla_sm120_prefill_dsv4_bounds_stale_padding(dual: bool) -> None:
    """DSv4 MG prefill (single and dual cache) with stale positive padding.

    Covers the footer-scale rope addressing: the QK rope operand and the XV
    rope MMA both re-derive gmem addresses from the raw indices on the math
    side, per cache phase for dual."""
    torch.manual_seed(13)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    num_heads, num_tokens = 64, 128
    topk, topk_len = 1024, 992  # partial last tile
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
    indices, ref_indices, topk_length = _stale_padding_indices(
        num_tokens, topk, topk_len, s_kv, device
    )
    sm_scale = d_qk**-0.5
    kwargs: dict = {}
    if dual:
        extra_topk, extra_len = 512, 480  # partial last extra tile
        extra_bf16 = (
            torch.randn(
                num_blocks,
                page_block_size,
                1,
                d_qk,
                device=device,
                dtype=torch.bfloat16,
            )
            / 10.0
        ).clamp(-1, 1)
        extra_packed = quantize_kv_dsv4(extra_bf16)
        extra_dequant = dequantize_kv_dsv4(extra_packed)
        extra_idx, extra_ref_idx, extra_topk_length = _stale_padding_indices(
            num_tokens, extra_topk, extra_len, s_kv, device
        )
        virtual_kv = torch.cat(
            [kv_dequant.reshape(-1, d_qk), extra_dequant.reshape(-1, d_qk)], dim=0
        ).reshape(-1, 1, 1, d_qk)
        extra_ref_shifted = torch.where(
            extra_ref_idx < 0, extra_ref_idx, extra_ref_idx + s_kv
        )
        ref_idx_all = torch.cat([ref_indices, extra_ref_shifted], dim=-1)
        kwargs = dict(
            extra_kv_cache=extra_packed,
            extra_indices=extra_idx,
            extra_topk_length=extra_topk_length,
        )
        ref_out, ref_lse = _ref_sparse_attn(q, virtual_kv, ref_idx_all, sm_scale, d_v)
    else:
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
        **kwargs,
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_prefill_dots3_swa_bounds_stale_padding() -> None:
    """DOTS3_SWA (SG/BI=32 split producer-consumer path) with stale padding.

    No topk_length is passed: the sliding-window cap (513) itself is the
    runtime bound, so slots in [513, 576) are past the effective length. The
    reference masks them with -1."""
    torch.manual_seed(14)
    device = torch.device("cuda")
    d_qk, d_v, topk, window = 1088, 1024, 576, 513
    num_tokens, num_heads = 65, 32
    page_block_size, num_blocks = 64, 64
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
    indices[:, window:] = 1 << 30  # stale garbage past the window, not -1
    ref_indices = indices.clone()
    ref_indices[:, window:] = -1

    sm_scale = d_qk**-0.5
    ref_out, ref_lse = _ref_sparse_attn(q, kv_dequant, ref_indices, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    sparse_mla_sm120_paged_attention(
        q, kv_packed, indices, output, out_lse, sm_scale, d_v=d_v
    )

    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


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
    """Construction stays JIT-free; a d_v outside the compiled format table is
    rejected on the first run."""
    device = torch.device("cuda")
    for d_v in (512, 1024):
        _SparseMLAPagedAttentionRunner(d_v=d_v, device=device)
    runner = _SparseMLAPagedAttentionRunner(d_v=768, device=device)
    kv_bf16 = torch.zeros(2, 64, 1, 512, device=device, dtype=torch.bfloat16)
    q = torch.zeros(1, 8, 512, device=device, dtype=torch.bfloat16)
    indices = torch.zeros(1, 128, device=device, dtype=torch.int32)
    output = torch.zeros(1, 8, 768, device=device, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="d_v"):
        runner.run(q, quantize_kv_dsv4(kv_bf16), indices, output, 512**-0.5)


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


@pytest.mark.parametrize("page_block_size", [32])
@pytest.mark.parametrize("num_tokens", [1, 128])
def test_sparse_mla_sm120_dsv4_page32(page_block_size: int, num_tokens: int) -> None:
    """DSv4 FP8 single-cache decode (T=1) and MG prefill (T=128) on the
    page-32 instantiation (vLLM's DeepSeek page), dispatched alongside the
    default page-64 kernels."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    topk = 192
    num_heads = 64
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
    scratch = {}
    if num_tokens <= 64:
        mid_out, mid_lse = _make_decode_scratch(
            num_tokens, num_heads, topk, d_v, device
        )
        scratch = {"mid_out": mid_out, "mid_lse": mid_lse}
    sparse_mla_sm120_paged_attention(
        q, kv_packed, indices, output, out_lse, sm_scale, d_v=d_v, **scratch
    )
    torch.testing.assert_close(output, ref_out, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_lse, ref_lse, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize(
    "main_pbs,extra_pbs", [(32, 64), (32, 2), (256, 128), (256, 2), (64, 128)]
)
@pytest.mark.parametrize("num_tokens", [1, 128])
def test_sparse_mla_sm120_dsv4_page32_dual(
    main_pbs: int, extra_pbs: int, num_tokens: int
) -> None:
    """DSv4 dual-cache with a page-32 main cache; extra_pbs=2 exercises the
    XOR extra layout under the page-32 main instantiation."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    d_qk, d_v = 512, 512
    topk, extra_topk = 192, 64
    num_heads = 64

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

    sm_scale = d_qk**-0.5
    virtual_kv = torch.cat(
        [main_dequant.reshape(-1, d_qk), extra_dequant.reshape(-1, d_qk)], dim=0
    ).reshape(-1, 1, 1, d_qk)
    extra_idx_shifted = torch.where(extra_idx < 0, extra_idx, extra_idx + main_s_kv)
    virtual_idx = torch.cat([main_idx, extra_idx_shifted], dim=-1)
    ref_out, ref_lse = _ref_sparse_attn(q, virtual_kv, virtual_idx, sm_scale, d_v)

    output = torch.zeros(
        (num_tokens, num_heads, d_v), dtype=torch.bfloat16, device=device
    )
    out_lse = torch.zeros((num_tokens, num_heads), dtype=torch.float32, device=device)
    scratch = {}
    if num_tokens <= 64:
        mid_out, mid_lse = _make_decode_scratch(
            num_tokens, num_heads, topk + extra_topk, d_v, device
        )
        scratch = {"mid_out": mid_out, "mid_lse": mid_lse}
    sparse_mla_sm120_paged_attention(
        q,
        main_packed,
        main_idx,
        output,
        out_lse,
        sm_scale,
        d_v=d_v,
        extra_kv_cache=extra_packed,
        extra_indices=extra_idx,
        **scratch,
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
    assert torch.isneginf(out_lse).all()


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
            False,
        )

    # topk=1000: 15 whole tiles plus a ragged 40-entry tail.
    with pytest.raises(RuntimeError, match=r"topk % 64 == 0"):
        call(512, 512, 584, 1000, 1, 2)  # DSV4, PREFILL_MG
    # DOTS3_SWA at topk=512: whole tiles, but below the sliding-window floor.
    with pytest.raises(RuntimeError, match=r"topk >= 513"):
        call(1088, 1024, 1160, 512, 4, 1)  # DOTS3_SWA, PREFILL_SG


@pytest.mark.parametrize("num_heads", [64, 128])
def test_sparse_mla_sm120_prefill_impl_mg_matches_swapab(num_heads: int) -> None:
    """Forced MG and forced swapAB agree on identical inputs (and the ref)."""
    case = _make_dsv3_2_prefill_case(num_heads, num_tokens=128)
    out_swapab, lse_swapab = _run_prefill_impl(case, "swapab")
    torch.testing.assert_close(out_swapab, case[5], atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_swapab, case[6], atol=5e-2, rtol=5e-2)
    out_mg, lse_mg = _run_prefill_impl(case, "mg")
    torch.testing.assert_close(out_mg, out_swapab, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_mg, lse_swapab, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(out_mg, case[5], atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_mg, case[6], atol=5e-2, rtol=5e-2)


def test_sparse_mla_sm120_mg_scale_reuse_graph() -> None:
    from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

    torch.manual_seed(723)
    device = torch.device("cuda")
    cache = quantize_kv_dsv3_2(
        torch.randn(4, 64, 1, 576, device=device, dtype=torch.bfloat16) * 0.1
    )
    q = torch.randn(4, 32, 576, device=device, dtype=torch.bfloat16) * 0.1
    indices = torch.randint(256, (4, 128), device=device, dtype=torch.int32)
    indices[:, 15:35] = -1
    lengths = torch.tensor([0, 64, 127, 128], device=device, dtype=torch.int32)
    sink = torch.randn(32, device=device)
    output = torch.empty(4, 32, 512, device=device, dtype=torch.bfloat16)
    lse = torch.empty(4, 32, device=device)
    module = _get_sparse_mla_sm120_decode_module()

    def call() -> None:
        module.sparse_mla_sm120_paged_attention(
            q,
            cache,
            indices,
            output,
            lse,
            576**-0.5,
            0,
            2,
            lengths,
            sink,
            None,
            None,
            None,
            False,
        )

    call()
    torch.cuda.synchronize()
    expected, expected_lse = output.clone(), lse.clone()
    masked = indices.masked_fill(
        torch.arange(128, device=device)[None] >= lengths[:, None], -1
    )
    reference, reference_lse = _ref_sparse_attn(
        q, dequantize_kv_dsv3_2(cache), masked, 576**-0.5, 512, attn_sink=sink
    )
    torch.testing.assert_close(expected, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(expected_lse, reference_lse, atol=5e-2, rtol=5e-2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(3):
        output.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected)
        assert torch.equal(lse, expected_lse)


def test_sparse_mla_sm120_inline_scale_rejects_padded_block_stride() -> None:
    """Large prefill-only calls cannot consume inline-cache page gaps."""
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
    with pytest.raises(ValueError, match="prefill envelope both reject"):
        sparse_mla_sm120_paged_attention(
            q, kv, indices, output, out_lse, sm_scale, d_v=d_v
        )


@pytest.mark.parametrize("num_tokens,num_heads", [(128, 64), (16, 64)])
def test_sparse_mla_sm120_inline_scale_prefill_accepts_padded_rows(
    num_tokens: int, num_heads: int
) -> None:
    """Padded-row inline-scale caches work in both prefill and decode: the
    kernels take the gmem row advance as a runtime stride and read only the
    packed payload at the row start. num_tokens=16 additionally exercises the
    decode-form path (the same runtime-stride addressing)."""
    q, kv_packed, indices, sm_scale, d_v, ref_out, ref_lse = _make_dsv3_2_prefill_case(
        num_heads, num_tokens=num_tokens
    )
    kv = torch.zeros(
        *kv_packed.shape[:-1],
        kv_packed.shape[-1] + 16,
        dtype=kv_packed.dtype,
        device=kv_packed.device,
    )
    kv[..., : kv_packed.shape[-1]] = kv_packed

    output = torch.zeros(
        num_tokens, num_heads, d_v, dtype=torch.bfloat16, device=q.device
    )
    out_lse = torch.zeros(num_tokens, num_heads, dtype=torch.float32, device=q.device)
    mid_out, mid_lse = _make_decode_scratch(
        num_tokens, num_heads, indices.shape[-1], d_v, q.device
    )
    sparse_mla_sm120_paged_attention(
        q,
        kv,
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


def test_sparse_mla_sm120_profile_routing_spy(monkeypatch) -> None:
    """Profile buckets select decode at T=8 and prefill at T=16, both reference-correct."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla._sparse_mla_sm120 import _calibration as cpb_mod

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

    monkeypatch.setattr(
        cpb_mod,
        "get_ordinary_profile",
        lambda request, device: {
            "buckets": {
                str(t): {"variant": 0 if t <= 8 else 2, "cpb": 1}
                for t in cpb_mod._PROFILE_T
            }
        },
    )
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)

    from flashinfer.mla._sparse_mla_sm120 import _prepared as prepared

    real_execute = prepared.PreparedCall.execute
    calls = {"decode": 0}

    def spy(self, *args, **kwargs):
        calls["decode"] += self.plan.inspect()["variant"] == 0
        return real_execute(self, *args, **kwargs)

    monkeypatch.setattr(prepared.PreparedCall, "execute", spy)

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


def test_sparse_mla_sm120_profile_cuda_graph(monkeypatch) -> None:
    """Profile-selected decode and prefill replay correctly on fresh data."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla._sparse_mla_sm120 import _calibration as cpb_mod

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
    monkeypatch.setattr(cpb_mod, "refresh_store", lambda: None)
    monkeypatch.setattr(cpb_mod, "get_ordinary_profile", lambda *args: None)

    from flashinfer.mla._sparse_mla_sm120 import _prepared as prepared

    real_execute = prepared.PreparedCall.execute
    calls = {"decode": 0}

    def spy(self, *args, **kwargs):
        calls["decode"] += self.plan.inspect()["variant"] == 0
        return real_execute(self, *args, **kwargs)

    monkeypatch.setattr(prepared.PreparedCall, "execute", spy)
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
    res = capture(8)
    assert calls["decode"] == 1
    replay_and_check(*res)

    monkeypatch.setattr(
        cpb_mod,
        "get_ordinary_profile",
        lambda request, device: {
            "buckets": {
                str(t): {"variant": 0 if t <= 16 else 2, "cpb": 1}
                for t in cpb_mod._PROFILE_T
            }
        },
    )
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)
    res = capture(8)
    assert calls["decode"] == 1
    replay_and_check(*res)
    res = capture(32)
    assert calls["decode"] == 0
    replay_and_check(*res)


def test_sparse_mla_sm120_runner_scratch_follows_routing(monkeypatch) -> None:
    """Each prepared decode shape owns scratch; prefill needs none and repeats reuse it."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla._sparse_mla_sm120 import _calibration as cpb_mod

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

    monkeypatch.setattr(
        cpb_mod,
        "get_ordinary_profile",
        lambda request, device: {
            "buckets": {
                str(t): {"variant": 0 if t <= 8 else 2, "cpb": 1}
                for t in cpb_mod._PROFILE_T
            }
        },
    )
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)

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

    call(16)
    assert next(iter(runner._prepared_calls.values())).workspace[0][2] == 0
    call(4)
    small = list(runner._prepared_calls.values())[-1]
    assert small.lse.shape[0] == 4
    assert small.mid is None and small.mlse is None
    call(8)
    grown = list(runner._prepared_calls.values())[-1]
    assert grown.mid is None and grown.mlse is None and grown.lse.shape[0] == 8
    arenas = tuple(runner._scratch_arenas)
    call(4)
    call(16)
    assert all(
        now is old for now, old in zip(runner._scratch_arenas, arenas, strict=True)
    )


def test_sparse_mla_sm120_runner_internal_scratch_cuda_graph(monkeypatch) -> None:
    """Capture pins runner-owned split-K scratch; replay matches fresh-input reference."""
    from flashinfer.mla import _sparse_mla_sm120 as sm
    from flashinfer.mla._sparse_mla_sm120 import _calibration as cpb_mod

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
    monkeypatch.setattr(cpb_mod, "refresh_store", lambda: None)
    monkeypatch.setattr(cpb_mod, "get_ordinary_profile", lambda *args: None)
    monkeypatch.delitem(cpb_mod._crossover, dev_key, raising=False)
    monkeypatch.setattr(cpb_mod, "_constants_version", cpb_mod._constants_version + 1)

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

    # Allocate replay inputs and references before capture; capture pins only scratch.
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
    assert next(iter(runner._prepared_calls.values())).mid is None
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    assert next(iter(runner._prepared_calls.values())).mid is not None

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
    ("mg", 1, 64, 512, 32, False),  # runtime DSV4 page
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
    ("sg", 5, 16, 128, 61, False),
    ("sg", 5, 64, 512, 64, True),
    ("sg", 5, 16, 65, 61, False),
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
    from flashinfer.mla._sparse_mla_sm120 import _policy as plan_mod
    from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

    device = torch.device("cuda")
    enum_name, predicate_name = _VARIANT_ENUM[variant]
    variant_id = int(getattr(plan_mod.KernelVariant, enum_name))
    expected = getattr(plan_mod, predicate_name)(
        model_type, num_heads, topk, page_block_size, has_extra
    )

    from flashinfer.mla._sparse_mla_sm120._execution import format_info

    info = format_info(model_type)
    d_qk = info["query_dim"]
    bpt = info["bytes_per_token"]
    d_v = info["value_dim"]
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
            False,
        )

    if expected:
        call()
        torch.cuda.synchronize()
    else:
        with pytest.raises(RuntimeError, match="sparse-MLA"):
            call()


@pytest.mark.parametrize(
    "model,heads,variant,dual,lengths",
    [
        (0, 16, 1, False, True),
        (4, 8, 1, False, True),
        (0, 32, 2, False, True),
        (0, 64, 4, False, True),
        (1, 32, 3, True, True),
        (1, 32, 3, True, False),
    ],
)
@pytest.mark.parametrize("sink_value", [None, 2.0, 1000.0])
def test_sparse_mla_sm120_prefill_empty_effective_kv(
    model, heads, variant, dual, lengths, sink_value
):
    from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

    torch.manual_seed(4484)
    tokens, topk = 6, 576 if model == 4 else 128
    dim = {0: 576, 1: 512, 4: 1088}[model]
    value_dim = 1024 if model == 4 else 512
    quantize = {0: quantize_kv_dsv3_2, 1: quantize_kv_dsv4, 4: quantize_kv_dots3_swa}[
        model
    ]
    dequantize = {
        0: dequantize_kv_dsv3_2,
        1: dequantize_kv_dsv4,
        4: dequantize_kv_dots3_swa,
    }[model]
    cache = quantize(
        torch.randn(2, 64, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    indices = torch.randint(128, (tokens, topk), device="cuda", dtype=torch.int32)
    indices[1] = -1
    lens = (
        torch.full((tokens,), topk, device="cuda", dtype=torch.int32)
        if lengths
        else None
    )
    if lengths:
        lens[0] = 0
    else:
        indices[0] = -1
    extra = cache.clone() if dual else None
    extra_indices = (
        torch.randint(128, (tokens, 128), device="cuda", dtype=torch.int32)
        if dual
        else None
    )
    extra_lens = (
        torch.full((tokens,), 128, device="cuda", dtype=torch.int32)
        if dual and lengths
        else None
    )
    if dual:
        extra_indices[:3] = -1
        indices[3] = -1
    sink = (
        torch.full((heads,), sink_value, device="cuda")
        if sink_value is not None
        else None
    )
    if sink_value == 1000.0:
        sink = torch.linspace(-1000.0, 1000.0, heads, device="cuda")
    output = torch.full(
        (tokens, heads, value_dim), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    lse = torch.full((tokens, heads), float("nan"), device="cuda")
    module = _get_sparse_mla_sm120_decode_module()
    module.sparse_mla_sm120_paged_attention(
        q,
        cache,
        indices,
        output,
        lse,
        dim**-0.5,
        model,
        variant,
        lens,
        sink,
        extra,
        extra_indices,
        extra_lens,
        False,
    )
    virtual = dequantize(cache).reshape(-1, 1, 1, dim)
    masked = indices.clone()
    if lengths:
        masked.masked_fill_(
            torch.arange(topk, device="cuda")[None] >= lens[:, None], -1
        )
    if dual:
        virtual = torch.cat([virtual, dequantize(extra).reshape(-1, 1, 1, dim)])
        masked = torch.cat(
            [
                masked,
                torch.where(extra_indices < 0, extra_indices, extra_indices + 128),
            ],
            -1,
        )
    ref, rlse = _ref_sparse_attn(
        q, virtual, masked, dim**-0.5, value_dim, attn_sink=sink
    )
    assert torch.count_nonzero(output[:2]) == 0
    torch.testing.assert_close(output, ref, atol=0.05, rtol=0.05)
    torch.testing.assert_close(lse, rlse, atol=0.02, rtol=0.02)


def test_sparse_mla_sm120_empty_state_cuda_merge():
    from flashinfer.mla import SparseMLASm120Wrapper

    torch.manual_seed(4484)
    q = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = quantize_kv_dsv4_1(
        torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    idx = torch.zeros(2, 128, device="cuda", dtype=torch.int32)
    idx[0] = -1
    out = torch.full_like(q, float("nan"))
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", compute_precision="fp8"
    )
    lse = wrapper.run(q, cache, idx, out, 512**-0.5, return_lse=True)
    for other in (0, 1):
        merged, merged_lse = flashinfer.merge_state(
            out[:1], lse[:1], out[other : other + 1], lse[other : other + 1]
        )
        torch.testing.assert_close(merged, out[other : other + 1], atol=0, rtol=0)
        torch.testing.assert_close(merged_lse, lse[other : other + 1], atol=0, rtol=0)
