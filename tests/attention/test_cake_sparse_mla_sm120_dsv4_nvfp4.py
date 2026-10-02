# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cake SM120 DeepSeek-V4 NVFP4 sparse-MLA decode (``backend="cake"`` on SM120/SM121).

Every attention case compares against the fp32 reference over the dequantized
NVFP4 operands at the SM120 NVFP4 tolerances (output atol=rtol=5e-2, LSE
atol=rtol=2e-2).
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
from flashinfer.mla._sparse_mla_sm120._cake_dsv4_nvfp4 import (
    _cake_nvfp4_sparse_mla_decode,
    cake_sparse_mla_sm120_dsv4_nvfp4_decode,
    cake_sparse_mla_sm120_dsv4_nvfp4_format_info,
    cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks,
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits,
    cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes,
    cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads,
)
from flashinfer.utils import is_sm12x_supported
from tests.attention.sparse_mla_test_utils import (
    _D_NOPE,
    _D_ROPE,
    _dequantize_nvfp4_cache,
    _dequantize_nvfp4_query,
    _reference_sparse_attention,
)

_D = _D_NOPE + _D_ROPE
_BYTES = 384
_OUT_TOL = dict(atol=5e-2, rtol=5e-2)
_LSE_TOL = dict(atol=2e-2, rtol=2e-2)


def _require_sm120() -> None:
    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("Cake SM120 NVFP4 sparse MLA requires SM12x")


def _latent(num_pages: int, page_size: int) -> torch.Tensor:
    return (
        torch.randn(num_pages, page_size, _D, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)


def _query(num_tokens: int, num_heads: int) -> torch.Tensor:
    return (
        torch.randn(num_tokens, num_heads, _D, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)


def _padded_view(cache_hnd: torch.Tensor, pad_bytes: int) -> torch.Tensor:
    """Copy an HND cache into a pool whose page stride exceeds the page payload."""

    num_pages, _, page_size, _ = cache_hnd.shape
    page_bytes = page_size * _BYTES
    stride = page_bytes + pad_bytes
    storage = torch.full(
        (num_pages * stride + 16,), 0xAB, dtype=torch.uint8, device="cuda"
    )
    base = (-storage.data_ptr()) % 16
    rows = storage[base : base + num_pages * stride].view(num_pages, stride)
    rows[:, :page_bytes] = cache_hnd.reshape(num_pages, page_bytes)
    return rows.as_strided(
        (num_pages, 1, page_size, _BYTES), (stride, page_bytes, _BYTES, 1), base
    )


def _layout_view(cache_hnd: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "HND":
        return cache_hnd
    if layout == "NHD":
        return cache_hnd.permute(0, 2, 1, 3)
    if layout == "3D":
        return cache_hnd.squeeze(1)
    if layout == "padded":
        return _padded_view(cache_hnd, 128)
    raise ValueError(layout)


def _apply_lengths(indices: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
    ref = indices.clone()
    if lengths is not None:
        for token in range(indices.shape[0]):
            ref[token, int(lengths[token].item()) :] = -1
    return ref


def _reference(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    sm_scale: float,
    *,
    main_lengths: torch.Tensor | None = None,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    lse_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    main_dequant = _dequantize_nvfp4_cache(main_cache)
    ref_indices = _apply_lengths(main_indices, main_lengths)
    if extra_cache is not None:
        extra_dequant = _dequantize_nvfp4_cache(extra_cache)
        main_rows = main_dequant.reshape(-1, _D).shape[0]
        kv = torch.cat(
            (main_dequant.reshape(-1, _D), extra_dequant.reshape(-1, _D)), dim=0
        )
        ref_extra = _apply_lengths(extra_indices, extra_lengths)
        ref_indices = torch.cat(
            (ref_indices, torch.where(ref_extra < 0, ref_extra, ref_extra + main_rows)),
            dim=1,
        )
    else:
        kv = main_dequant.reshape(-1, _D)
    output, lse = _reference_sparse_attention(
        _dequantize_nvfp4_query(q),
        kv.reshape(1, -1, 1, _D),
        ref_indices,
        sm_scale,
        attn_sink=attn_sink,
    )
    return output, lse * lse_scale


def test_cake_format_info() -> None:
    info = cake_sparse_mla_sm120_dsv4_nvfp4_format_info()
    assert info["query_dim"] == 512 and info["value_dim"] == 512
    assert info["bytes_per_token"] == 384 and info["chunk_width"] == 64
    assert info["runtime_page"] and info["runtime_extra_page"]
    assert set(info["heads"]) >= {8, 16, 32, 64, 128}
    assert cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads() == info["heads"]
    assert cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(128) == 2
    assert cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(512, 512) == 16
    assert cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(130, 1) == 4


def test_cake_plan_splits_rules() -> None:
    plan = cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits
    # Two chunks never split, whatever the grid.
    assert plan(num_tokens=1, num_heads=8, topk=128, num_sms=188) == (1, 2)
    assert plan(num_tokens=128, num_heads=128, topk=128, num_sms=188) == (1, 2)
    # A lone CTA may go down to one chunk per split.
    assert plan(num_tokens=1, num_heads=16, topk=512, num_sms=188) == (8, 1)
    # Small grids split while they stay within 80 % of the SMs with >= 2 chunks per CTA.
    assert plan(num_tokens=8, num_heads=128, topk=512, num_sms=188) == (2, 4)
    assert plan(num_tokens=8, num_heads=8, topk=512, num_sms=188) == (4, 2)
    # Full grids run unsplit unless the doubled 16-chunk grid ends in a half-full wave.
    assert plan(num_tokens=128, num_heads=128, topk=512, num_sms=188) == (1, 8)
    # Candidate lists beyond the 16-chunk index table always split.
    splits, cpb = plan(
        num_tokens=128, num_heads=128, topk=512, extra_topk=1024, num_sms=188
    )
    assert cpb <= 16 and splits * cpb >= 24
    assert plan(num_tokens=4, num_heads=128, topk=512, num_sms=188, max_splits=1) == (
        1,
        8,
    )
    assert (
        cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(2, 128, 512)
        == 2 * 128 * 8 * 1028 + 2 * 128 * 4 + 48
    )


@pytest.mark.parametrize("topk", [128, 512])
@pytest.mark.parametrize("page_size", [32, 64, 128])
@pytest.mark.parametrize("num_heads", [8, 16, 64, 128])
@pytest.mark.parametrize("with_sink", [False, True])
def test_cake_decode_matches_dequantized_reference(
    topk: int, page_size: int, num_heads: int, with_sink: bool
) -> None:
    """Direct and split epilogues over runtime page sizes with lengths, -1 masks, sink and lse_scale."""

    _require_sm120()
    torch.manual_seed(20261001 + topk + page_size + num_heads + int(with_sink))
    num_tokens = 3
    num_pages = 16 * 64 // page_size
    q = _query(num_tokens, num_heads)
    latent = _latent(num_pages, page_size)
    cache = nvfp4_quantize_pack_sparse_mla_cache(latent)
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    topk_len = (topk * 7) // 10
    indices[:, topk_len - 7 : topk_len] = -1
    indices[0, : topk // 8] = -1
    lengths = torch.tensor(
        [topk_len, topk, topk_len - 1], dtype=torch.int32, device="cuda"
    )
    attn_sink = (
        torch.linspace(-1.0, 1.0, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    sm_scale = _D**-0.5
    lse_scale = 0.5 if with_sink else 1.0
    reference, reference_lse = _reference(
        q,
        cache,
        indices,
        sm_scale,
        main_lengths=lengths,
        attn_sink=attn_sink,
        lse_scale=lse_scale,
    )
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk)
    for num_splits in (None, 1, chunks):
        output, lse = _cake_nvfp4_sparse_mla_decode(
            q,
            cache,
            indices,
            sm_scale,
            topk_length=lengths,
            attn_sink=attn_sink,
            lse_scale=lse_scale,
            num_splits=num_splits,
        )
        torch.testing.assert_close(output, reference, **_OUT_TOL)
        torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)


@pytest.mark.parametrize("num_heads", [8, 128])
@pytest.mark.parametrize("main_page_size", [32, 64])
@pytest.mark.parametrize("extra_page_size,extra_topk", [(2, 128), (32, 512), (64, 512)])
def test_cake_decode_dual_cache_matches_reference(
    num_heads: int, main_page_size: int, extra_page_size: int, extra_topk: int
) -> None:
    """Main and compressed cache sections share one online softmax."""

    _require_sm120()
    torch.manual_seed(20261002 + num_heads + main_page_size + extra_page_size)
    num_tokens, main_topk = 2, 128 if extra_topk == 128 else 512
    main_pages = 8 * 64 // main_page_size
    extra_pages = max(16, (2 * extra_topk + extra_page_size - 1) // extra_page_size)
    q = _query(num_tokens, num_heads)
    main_cache = nvfp4_quantize_pack_sparse_mla_cache(
        _latent(main_pages, main_page_size)
    )
    extra_cache = nvfp4_quantize_pack_sparse_mla_cache(
        _latent(extra_pages, extra_page_size)
    )
    main_indices = torch.randint(
        0,
        main_pages * main_page_size,
        (num_tokens, main_topk),
        dtype=torch.int32,
        device="cuda",
    )
    extra_indices = torch.randint(
        0,
        extra_pages * extra_page_size,
        (num_tokens, extra_topk),
        dtype=torch.int32,
        device="cuda",
    )
    main_lengths = torch.tensor(
        [main_topk - 17, main_topk // 2 + 1], dtype=torch.int32, device="cuda"
    )
    extra_lengths = torch.tensor(
        [extra_topk - 13, max(1, extra_topk - 29)], dtype=torch.int32, device="cuda"
    )
    main_indices[:, 91:96] = -1
    extra_indices[:, 37:43] = -1
    attn_sink = torch.linspace(-1.0, 1.0, num_heads, dtype=torch.float32, device="cuda")
    sm_scale = _D**-0.5
    reference, reference_lse = _reference(
        q,
        main_cache,
        main_indices,
        sm_scale,
        main_lengths=main_lengths,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=attn_sink,
    )
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(main_topk, extra_topk)
    for num_splits in (None, 1, 2, chunks):
        output, lse = _cake_nvfp4_sparse_mla_decode(
            q,
            main_cache,
            main_indices,
            sm_scale,
            topk_length=main_lengths,
            attn_sink=attn_sink,
            extra_kv_cache=extra_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_lengths,
            num_splits=num_splits,
        )
        torch.testing.assert_close(output, reference, **_OUT_TOL)
        torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)


@pytest.mark.parametrize("layout", ["NHD", "3D", "padded"])
def test_cake_decode_cache_layouts_match_hnd(layout: str) -> None:
    """3-D, NHD and padded-page-stride views of one pool give the HND result bit for bit."""

    _require_sm120()
    torch.manual_seed(20261003)
    num_tokens, num_heads, topk, page_size, num_pages = 2, 32, 512, 48, 16
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    sm_scale = _D**-0.5
    reference, reference_lse = _reference(q, cache, indices, sm_scale)
    expected, expected_lse = _cake_nvfp4_sparse_mla_decode(q, cache, indices, sm_scale)
    torch.testing.assert_close(expected, reference, **_OUT_TOL)
    torch.testing.assert_close(expected_lse, reference_lse, **_LSE_TOL)
    view = _layout_view(cache, layout)
    assert view.stride(0) > page_size * _BYTES or layout != "padded"
    output, lse = _cake_nvfp4_sparse_mla_decode(q, view, indices, sm_scale)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
    # The same pool as the extra cache of a dual call.
    extra_indices = torch.randint(
        0, num_pages * page_size, (num_tokens, 128), dtype=torch.int32, device="cuda"
    )
    dual_ref, dual_ref_lse = _reference(
        q, cache, indices, sm_scale, extra_cache=cache, extra_indices=extra_indices
    )
    dual_out, dual_lse = _cake_nvfp4_sparse_mla_decode(
        q, cache, indices, sm_scale, extra_kv_cache=view, extra_indices=extra_indices
    )
    torch.testing.assert_close(dual_out, dual_ref, **_OUT_TOL)
    torch.testing.assert_close(dual_lse, dual_ref_lse, **_LSE_TOL)


@pytest.mark.parametrize("with_sink", [False, True])
def test_cake_decode_empty_rows(with_sink: bool) -> None:
    """Zero lengths and all -1 rows write zeros and -inf / sink-only LSE next to populated rows."""

    _require_sm120()
    torch.manual_seed(20261004 + int(with_sink))
    num_tokens, num_heads, topk = 4, 128, 128
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(4, 64))
    indices = torch.randint(
        0, 4 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    indices[1] = -1
    lengths = torch.tensor([0, topk, 1, 65], dtype=torch.int32, device="cuda")
    attn_sink = (
        torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    sm_scale = _D**-0.5
    reference, reference_lse = _reference(
        q,
        cache,
        indices,
        sm_scale,
        main_lengths=lengths,
        attn_sink=attn_sink,
        lse_scale=2.0,
    )
    for num_splits in (1, 2):
        output, lse = _cake_nvfp4_sparse_mla_decode(
            q,
            cache,
            indices,
            sm_scale,
            topk_length=lengths,
            attn_sink=attn_sink,
            lse_scale=2.0,
            num_splits=num_splits,
        )
        assert torch.count_nonzero(output[:2]) == 0
        if attn_sink is None:
            assert torch.isneginf(lse[:2]).all()
        else:
            torch.testing.assert_close(
                lse[:2],
                (attn_sink * math.log2(math.e) * 2.0).unsqueeze(0).expand(2, -1),
            )
        torch.testing.assert_close(output[2:], reference[2:], **_OUT_TOL)
        torch.testing.assert_close(lse[2:], reference_lse[2:], **_LSE_TOL)


@pytest.mark.parametrize("num_splits", [1, 2, 4])
def test_cake_decode_sink_dominated_rows(num_splits: int) -> None:
    """Sinks above the row's log-sum-exp gate the output on the direct and the split-merge paths alike."""

    _require_sm120()
    torch.manual_seed(20261011 + num_splits)
    num_tokens, num_heads, topk = 2, 16, 256
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(8, 64))
    indices = torch.randint(
        0, 8 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    # Few valid candidates keep the LSE small; sinks from far below to far above it.
    lengths = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
    attn_sink = torch.linspace(-6.0, 6.0, num_heads, dtype=torch.float32, device="cuda")
    sm_scale = _D**-0.5
    reference, reference_lse = _reference(
        q, cache, indices, sm_scale, main_lengths=lengths, attn_sink=attn_sink
    )
    output, lse = _cake_nvfp4_sparse_mla_decode(
        q,
        cache,
        indices,
        sm_scale,
        topk_length=lengths,
        attn_sink=attn_sink,
        num_splits=num_splits,
    )
    torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)
    torch.testing.assert_close(output, reference, **_OUT_TOL)


def test_cake_decode_bitwise_repeatable() -> None:
    _require_sm120()
    torch.manual_seed(20261005)
    num_tokens, num_heads, topk = 8, 64, 512
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(32, 64))
    indices = torch.randint(
        0, 32 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    first = _cake_nvfp4_sparse_mla_decode(q, cache, indices, _D**-0.5)
    for _ in range(3):
        again = _cake_nvfp4_sparse_mla_decode(q, cache, indices, _D**-0.5)
        assert torch.equal(first[0], again[0]) and torch.equal(first[1], again[1])


def test_cake_decode_head_count_48() -> None:
    """A head count outside the hand-written set (three 16-head blocks)."""

    _require_sm120()
    if 48 not in cake_sparse_mla_sm120_dsv4_nvfp4_supported_heads():
        pytest.skip("48-head instance not generated")
    torch.manual_seed(20261006)
    num_tokens, num_heads, topk = 2, 48, 512
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(8, 64))
    indices = torch.randint(
        0, 8 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    reference, reference_lse = _reference(q, cache, indices, _D**-0.5)
    for num_splits in (1, 4):
        output, lse = _cake_nvfp4_sparse_mla_decode(
            q, cache, indices, _D**-0.5, num_splits=num_splits
        )
        torch.testing.assert_close(output, reference, **_OUT_TOL)
        torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)


def test_cake_decode_rejects_bad_inputs() -> None:
    _require_sm120()
    torch.manual_seed(20261007)
    q = _query(2, 16)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(4, 64))
    indices = torch.randint(0, 4 * 64, (2, 512), dtype=torch.int32, device="cuda")
    output = torch.empty_like(q)
    out_lse = torch.empty(2, 16, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="query heads"):
        _cake_nvfp4_sparse_mla_decode(_query(2, 24), cache, indices, 1.0)
    with pytest.raises(ValueError, match="int32"):
        _cake_nvfp4_sparse_mla_decode(q, cache, indices.to(torch.int64), 1.0)
    with pytest.raises(ValueError, match="multiple of 16"):
        _cake_nvfp4_sparse_mla_decode(q, _padded_view(cache, 8), indices, 1.0)
    with pytest.raises(ValueError, match="mid_out"):
        cake_sparse_mla_sm120_dsv4_nvfp4_decode(
            q, cache, indices, output, out_lse, 1.0, num_splits=2
        )
    with pytest.raises(ValueError, match="at most"):
        cake_sparse_mla_sm120_dsv4_nvfp4_decode(
            q,
            cache,
            torch.zeros(2, 2048, dtype=torch.int32, device="cuda"),
            output,
            out_lse,
            1.0,
            num_splits=1,
        )
    with pytest.raises(ValueError, match="provided together"):
        _cake_nvfp4_sparse_mla_decode(q, cache, indices, 1.0, extra_kv_cache=cache)


@pytest.mark.parametrize(
    "num_tokens,num_heads,page_size,topk,extra_page_size",
    [(8, 8, 32, 128, None), (2, 128, 64, 512, 2), (3, 16, 128, 128, 32)],
)
def test_cake_public_api_decode(
    num_tokens: int,
    num_heads: int,
    page_size: int,
    topk: int,
    extra_page_size: int | None,
) -> None:
    """``trtllm_batch_decode_sparse_mla_dsv4(backend="cake", kv_cache_format="nvfp4")`` end to end."""

    _require_sm120()
    torch.manual_seed(20261008 + num_tokens + num_heads + page_size)
    num_pages = 8 * 64 // page_size
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.randint(
        topk // 2, topk + 1, (num_tokens,), dtype=torch.int32, device="cuda"
    )
    extra_cache = extra_indices = extra_lengths = None
    extra_topk = 0
    if extra_page_size is not None:
        extra_topk = 128
        extra_pages = max(16, 256 // extra_page_size)
        extra_cache = nvfp4_quantize_pack_sparse_mla_cache(
            _latent(extra_pages, extra_page_size)
        )
        extra_indices = torch.randint(
            0,
            extra_pages * extra_page_size,
            (num_tokens, extra_topk),
            dtype=torch.int32,
            device="cuda",
        )
        extra_lengths = torch.randint(
            1, extra_topk + 1, (num_tokens,), dtype=torch.int32, device="cuda"
        )
    sm_scale = _D**-0.5
    reference, _ = _reference(
        q,
        cache,
        indices,
        sm_scale,
        main_lengths=lengths,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
    )
    workspace_storage = torch.empty(
        cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(
            num_tokens, num_heads, topk, extra_topk
        )
        + 1,
        dtype=torch.uint8,
        device="cuda",
    )
    workspace = workspace_storage[1:]
    output = torch.empty_like(q)
    returned = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q,
        swa_kv_cache=cache,
        workspace_buffer=workspace,
        sparse_indices=indices,
        compressed_kv_cache=extra_cache,
        swa_topk_lens=lengths,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lengths,
        out=output,
        bmm1_scale=sm_scale,
        backend="cake",
        kv_cache_format="nvfp4",
    )
    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, reference, **_OUT_TOL)
    # 4-D query [T, 1, H, D] with the NHD cache layout.
    output4 = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q.unsqueeze(1),
        swa_kv_cache=cache.permute(0, 2, 1, 3),
        workspace_buffer=workspace,
        sparse_indices=indices,
        compressed_kv_cache=extra_cache.permute(0, 2, 1, 3)
        if extra_cache is not None
        else None,
        swa_topk_lens=lengths,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lengths,
        bmm1_scale=sm_scale,
        kv_layout="NHD",
        backend="cake",
        kv_cache_format="nvfp4",
    )
    torch.testing.assert_close(output4.squeeze(1), output, atol=0, rtol=0)
    with pytest.raises(ValueError, match="workspace"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=cache,
            workspace_buffer=torch.empty(16, dtype=torch.uint8, device="cuda"),
            sparse_indices=indices,
            swa_topk_lens=lengths,
            bmm1_scale=sm_scale,
            backend="cake",
            kv_cache_format="nvfp4",
        )
    with pytest.raises(ValueError, match="kv_cache_format='nvfp4'"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=cache,
            workspace_buffer=workspace,
            sparse_indices=indices,
            swa_topk_lens=lengths,
            bmm1_scale=sm_scale,
            backend="cake",
        )


def test_cake_public_api_cuda_graph() -> None:
    """The public Cake route is replayable in a CUDA graph and bitwise stable across replays."""

    _require_sm120()
    torch.manual_seed(20261009)
    num_tokens, num_heads, topk, page_size = 4, 128, 512, 32
    num_pages = 32
    q = _query(num_tokens, num_heads)
    replay_q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(num_pages, page_size))
    indices = torch.randint(
        0, num_pages * page_size, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.full((num_tokens,), topk - 3, dtype=torch.int32, device="cuda")
    workspace = torch.empty(
        cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes(num_tokens, num_heads, topk),
        dtype=torch.uint8,
        device="cuda",
    )
    output = torch.empty_like(q)

    def run() -> None:
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=cache,
            workspace_buffer=workspace,
            sparse_indices=indices,
            swa_topk_lens=lengths,
            out=output,
            bmm1_scale=_D**-0.5,
            backend="cake",
            kv_cache_format="nvfp4",
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.copy_(replay_q)
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    reference, _ = _reference(replay_q, cache, indices, _D**-0.5, main_lengths=lengths)
    torch.testing.assert_close(first, reference, **_OUT_TOL)
    for _ in range(2):
        output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, first)


@pytest.mark.parametrize("num_tokens", [1, 5])
def test_cake_wrapper_run(num_tokens: int) -> None:
    """``SparseMLASm120Wrapper(kv_cache_format="nvfp4", backend="cake")`` keeps the run() contract."""

    _require_sm120()
    torch.manual_seed(20261010 + num_tokens)
    num_heads, topk, extra_topk = 16, 512, 128
    q = _query(num_tokens, num_heads)
    cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(8, 64))
    extra_cache = nvfp4_quantize_pack_sparse_mla_cache(_latent(64, 2))
    indices = torch.randint(
        0, 8 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    extra_indices = torch.randint(
        0, 64 * 2, (num_tokens, extra_topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.full((num_tokens,), topk - 5, dtype=torch.int32, device="cuda")
    sm_scale = _D**-0.5
    runner = flashinfer.mla.SparseMLASm120Wrapper(
        max_num_tokens=8,
        max_num_heads=num_heads,
        kv_cache_format="nvfp4",
        backend="cake",
        device=q.device,
    )
    output = torch.empty_like(q)
    lse = runner.run(
        q.unsqueeze(1),
        cache,
        indices.unsqueeze(1),
        output.unsqueeze(1),
        sm_scale,
        topk_length=lengths,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices.unsqueeze(1),
        return_lse=True,
        lse_scale=math.log(2),
    )
    reference, reference_lse = _reference(
        q,
        cache,
        indices,
        sm_scale,
        main_lengths=lengths,
        extra_cache=extra_cache,
        extra_indices=extra_indices,
        lse_scale=math.log(2),
    )
    torch.testing.assert_close(output, reference, **_OUT_TOL)
    torch.testing.assert_close(lse, reference_lse, **_LSE_TOL)
    # Caller-owned scratch and LSE buffers.
    chunks = cake_sparse_mla_sm120_dsv4_nvfp4_num_chunks(topk, extra_topk)
    mid_out = torch.empty(
        num_tokens, num_heads, chunks, 512, dtype=torch.bfloat16, device="cuda"
    )
    mid_lse = torch.empty(
        num_tokens, num_heads, chunks, dtype=torch.float32, device="cuda"
    )
    out_lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device="cuda")
    output2 = torch.empty_like(q)
    returned = runner.run(
        q,
        cache,
        indices,
        output2,
        sm_scale,
        topk_length=lengths,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        out_lse=out_lse,
        mid_out=mid_out,
        mid_lse=mid_lse,
        return_lse=True,
        lse_scale=math.log(2),
    )
    assert returned.data_ptr() == out_lse.data_ptr()
    torch.testing.assert_close(output2, output, atol=0, rtol=0)
    torch.testing.assert_close(out_lse, lse, atol=0, rtol=0)
    assert runner.run(q, cache, indices, output2, sm_scale) is None
    with pytest.raises(ValueError, match="max_num_tokens"):
        runner.run(
            _query(9, num_heads),
            cache,
            torch.zeros(9, topk, dtype=torch.int32, device="cuda"),
            _query(9, num_heads),
            sm_scale,
        )
    with pytest.raises(ValueError, match="nvfp4"):
        flashinfer.mla.SparseMLASm120Wrapper(backend="cake")
