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

import inspect
import math

import pytest
import torch

import flashinfer
from flashinfer.mla import (
    nvfp4_quantize_append_sparse_mla_cache,
    nvfp4_quantize_pack_sparse_mla_cache,
)
from flashinfer.mla._core import _nvfp4_sparse_mla_workspace
from flashinfer.mla._sparse_mla_nvfp4_sm120 import (
    _nvfp4_sparse_mla_decode,
    _nvfp4_sparse_mla_prefill,
    _nvfp4_sparse_mla_m16n8k64_candidate_major,
    _nvfp4_sparse_mla_m16n32k64,
)
from flashinfer.utils import is_sm120a_supported


_D_NOPE = 448
_D_ROPE = 64
_PACKED_NOPE_BYTES = 224
_ROPE_BYTES = 128
_DATA_BYTES_PER_TOKEN = 352
_SCALE_BYTES_PER_TOKEN = 32
_BYTES_PER_TOKEN = 384


def test_nvfp4_sparse_mla_reuses_fp8_public_api() -> None:
    """The format selector is additive, keyword-only, and FP8 by default."""
    parameter = inspect.signature(
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4
    ).parameters["kv_cache_format"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == "fp8"

    wrapper_parameter = inspect.signature(
        flashinfer.mla.SparseMLASm120Wrapper.__init__
    ).parameters["kv_cache_format"]
    assert wrapper_parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert wrapper_parameter.default == "fp8"


def _require_sm120() -> None:
    if not is_sm120a_supported(torch.device("cuda")):
        pytest.skip("NVFP4 sparse MLA requires SM120/SM121")


def _split_cache(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cache.shape[1] == 1:
        page_size = cache.shape[2]
    else:
        page_size = cache.shape[1]
    flat = cache.reshape(cache.shape[0], page_size * _BYTES_PER_TOKEN)
    data = flat[:, : page_size * _DATA_BYTES_PER_TOKEN].reshape(
        cache.shape[0], page_size, _DATA_BYTES_PER_TOKEN
    )
    scales = flat[:, page_size * _DATA_BYTES_PER_TOKEN :].reshape(
        cache.shape[0], page_size, _SCALE_BYTES_PER_TOKEN
    )
    return data, scales


def _reference_rows(latent_kv: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rows = latent_kv.reshape(-1, _D_NOPE + _D_ROPE)
    global_scale = torch.ones(1, dtype=torch.float32, device=latent_kv.device)
    packed, scales = flashinfer.nvfp4_kv_quantize(
        rows[:, :_D_NOPE].contiguous(), global_scale
    )
    rope = (
        rows[:, _D_NOPE:]
        .contiguous()
        .view(torch.uint8)
        .reshape(rows.shape[0], _ROPE_BYTES)
    )
    return packed, scales.view(torch.uint8), rope


def _dequantize_linear_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(
        packed.shape[0], -1
    )
    values = lut[codes.long()]
    scale_values = scales.view(torch.float8_e4m3fn).float()
    return values * scale_values.repeat_interleave(16, dim=-1)


def _dequantize_nvfp4_cache(cache: torch.Tensor) -> torch.Tensor:
    data, scales = _split_cache(cache)
    num_pages, page_size = data.shape[:2]
    nope = _dequantize_linear_nvfp4(
        data[..., :_PACKED_NOPE_BYTES].reshape(-1, _PACKED_NOPE_BYTES),
        scales[..., : _D_NOPE // 16].reshape(-1, _D_NOPE // 16),
    )
    rope = (
        data[..., _PACKED_NOPE_BYTES:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, _D_ROPE)
        .float()
    )
    return torch.cat((nope, rope), dim=-1).reshape(
        num_pages, page_size, 1, _D_NOPE + _D_ROPE
    )


def _dequantize_nvfp4_query(q: torch.Tensor) -> torch.Tensor:
    q_flat = q.reshape(-1, _D_NOPE + _D_ROPE)
    global_scale = torch.ones(1, dtype=torch.float32, device=q.device)
    packed, scales = flashinfer.nvfp4_kv_quantize(
        q_flat[:, :_D_NOPE].contiguous(), global_scale
    )
    nope = _dequantize_linear_nvfp4(packed, scales)
    return torch.cat((nope, q_flat[:, _D_NOPE:].float()), dim=-1).reshape_as(q.float())


def _reference_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    *,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, dim = q.shape
    topk = indices.shape[1]
    invalid = indices < 0
    if topk_length is not None:
        positions = torch.arange(topk, device=q.device).unsqueeze(0)
        invalid = invalid | (positions >= topk_length.unsqueeze(1))
    gathered = kv.reshape(-1, dim).index_select(
        0, indices.clamp_min(0).reshape(-1).long()
    )
    gathered = gathered.reshape(num_tokens, topk, dim)
    logits = torch.einsum("thd,tkd->thk", q, gathered) * sm_scale
    logits.masked_fill_(invalid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    safe_lse = torch.where(torch.isneginf(lse), torch.inf, lse)
    weights = torch.exp(logits - safe_lse.unsqueeze(-1))
    output = torch.einsum("thk,tkd->thd", weights, gathered)

    if attn_sink is not None:
        sink = attn_sink.float().unsqueeze(0)
        output *= torch.sigmoid(lse - sink).unsqueeze(-1)
        lse = torch.logaddexp(lse, sink)

    return output.to(torch.bfloat16), lse * math.log2(math.e)


@pytest.mark.parametrize("page_size", [2, 64])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_nvfp4_sparse_mla_full_page_pack(page_size, kv_layout):
    _require_sm120()
    torch.manual_seed(42)
    latent_kv = torch.randn(
        2, page_size, _D_NOPE + _D_ROPE, dtype=torch.bfloat16, device="cuda"
    )

    cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv, kv_layout=kv_layout)
    data, scales = _split_cache(cache)
    packed_ref, scales_ref, rope_ref = _reference_rows(latent_kv)

    assert cache.dtype == torch.uint8
    expected_shape = (
        (2, 1, page_size, _BYTES_PER_TOKEN)
        if kv_layout == "HND"
        else (2, page_size, 1, _BYTES_PER_TOKEN)
    )
    assert cache.shape == expected_shape
    torch.testing.assert_close(
        data[..., :_PACKED_NOPE_BYTES].reshape_as(packed_ref), packed_ref
    )
    torch.testing.assert_close(
        data[..., _PACKED_NOPE_BYTES:].reshape_as(rope_ref), rope_ref
    )
    torch.testing.assert_close(scales[..., :28].reshape_as(scales_ref), scales_ref)
    assert torch.count_nonzero(scales[..., 28:]) == 0


@pytest.mark.parametrize("page_size", [2, 64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_nvfp4_sparse_mla_incremental_append_matches_full_pack(page_size, index_dtype):
    _require_sm120()
    torch.manual_seed(7)
    num_pages = 2
    latent_kv = torch.randn(
        num_pages,
        page_size,
        _D_NOPE + _D_ROPE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    full_cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv)
    append_cache = torch.full_like(full_cache, 0xA5)
    slots = torch.arange(num_pages * page_size, dtype=index_dtype, device="cuda")

    nvfp4_quantize_append_sparse_mla_cache(
        latent_kv.reshape(-1, _D_NOPE + _D_ROPE), slots, append_cache
    )
    torch.testing.assert_close(append_cache, full_cache)


def test_nvfp4_sparse_mla_incremental_append_accepts_3d_cache() -> None:
    """vLLM uses the latent-head-free [pages, page_size, bytes] shorthand."""
    _require_sm120()
    torch.manual_seed(8)
    latent_kv = torch.randn(2, 64, 512, dtype=torch.bfloat16, device="cuda")
    expected = nvfp4_quantize_pack_sparse_mla_cache(latent_kv, kv_layout="NHD").squeeze(
        2
    )
    actual = torch.empty_like(expected)
    slots = torch.arange(128, dtype=torch.int64, device="cuda")
    nvfp4_quantize_append_sparse_mla_cache(latent_kv.reshape(-1, 512), slots, actual)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("page_size", [2, 64])
def test_nvfp4_sparse_mla_pack_and_append_accept_page_stride(page_size: int) -> None:
    """Packed vLLM pools leave padding between logical cache pages."""
    _require_sm120()
    torch.manual_seed(9 + page_size)
    num_pages = 3
    latent_kv = torch.randn(
        num_pages, page_size, 512, dtype=torch.bfloat16, device="cuda"
    )
    expected = nvfp4_quantize_pack_sparse_mla_cache(latent_kv, kv_layout="NHD").squeeze(
        2
    )
    logical_page_bytes = page_size * _BYTES_PER_TOKEN
    page_stride = logical_page_bytes + 3 * _BYTES_PER_TOKEN

    def make_strided_cache() -> torch.Tensor:
        backing = torch.full(
            (num_pages * page_stride,), 0xA5, dtype=torch.uint8, device="cuda"
        )
        return torch.as_strided(
            backing,
            size=(num_pages, page_size, _BYTES_PER_TOKEN),
            stride=(page_stride, _BYTES_PER_TOKEN, 1),
        )

    append_cache = make_strided_cache()
    slots = torch.arange(num_pages * page_size, dtype=torch.int64, device="cuda")
    nvfp4_quantize_append_sparse_mla_cache(
        latent_kv.reshape(-1, 512), slots, append_cache
    )

    # A page is internally opaque but contiguous, so compare its complete byte
    # payload against the independently packed contiguous implementation.
    for page in range(num_pages):
        torch.testing.assert_close(
            append_cache[page].reshape(-1), expected[page].reshape(-1), atol=0, rtol=0
        )


@pytest.mark.parametrize("page_size", [2, 64])
def test_nvfp4_sparse_mla_append_writes_only_selected_slots(page_size):
    _require_sm120()
    torch.manual_seed(11)
    num_pages = 2
    inputs = torch.randn(4, _D_NOPE + _D_ROPE, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor(
        [0, num_pages * page_size - 1, -1, num_pages * page_size],
        dtype=torch.int32,
        device="cuda",
    )
    cache = torch.full(
        (num_pages, 1, page_size, _BYTES_PER_TOKEN),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    )

    nvfp4_quantize_append_sparse_mla_cache(inputs, slots, cache)
    data, scales = _split_cache(cache)
    packed_ref, scales_ref, rope_ref = _reference_rows(inputs)

    for input_idx, slot in enumerate((0, num_pages * page_size - 1)):
        page_idx, entry_idx = divmod(slot, page_size)
        torch.testing.assert_close(
            data[page_idx, entry_idx, :_PACKED_NOPE_BYTES], packed_ref[input_idx]
        )
        torch.testing.assert_close(
            data[page_idx, entry_idx, _PACKED_NOPE_BYTES:], rope_ref[input_idx]
        )
        torch.testing.assert_close(
            scales[page_idx, entry_idx, :28], scales_ref[input_idx]
        )
        assert torch.count_nonzero(scales[page_idx, entry_idx, 28:]) == 0

    if page_size > 2:
        assert torch.all(data[0, 1] == 0xA5)
        assert torch.all(scales[0, 1] == 0xA5)


@pytest.mark.parametrize("misaligned", ["input", "cache_base", "page_stride"])
def test_nvfp4_sparse_mla_append_rejects_misaligned_vector_accesses(
    misaligned: str,
) -> None:
    """Vectorized BF16/uint4 accesses require a 16-byte-aligned cache ABI."""
    _require_sm120()
    latent_kv = torch.empty(1, 512, dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    cache = torch.empty(2, 2, _BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda")
    expected_error = "kv_cache"

    if misaligned == "input":
        input_storage = torch.empty(513, dtype=torch.bfloat16, device="cuda")
        latent_kv = input_storage[1:].view(1, 512)
        expected_error = "latent_kv"
    elif misaligned == "cache_base":
        cache_storage = torch.empty(cache.numel() + 1, dtype=torch.uint8, device="cuda")
        cache = cache_storage[1:].view_as(cache)
    else:
        page_stride = 2 * _BYTES_PER_TOKEN + 1
        cache_storage = torch.empty(
            page_stride + 2 * _BYTES_PER_TOKEN,
            dtype=torch.uint8,
            device="cuda",
        )
        cache = torch.as_strided(
            cache_storage,
            size=(2, 2, _BYTES_PER_TOKEN),
            stride=(page_stride, _BYTES_PER_TOKEN, 1),
        )

    with pytest.raises(RuntimeError, match=rf"{expected_error}.*16"):
        nvfp4_quantize_append_sparse_mla_cache(latent_kv, slots, cache)


def test_nvfp4_sparse_mla_append_rejects_duplicate_valid_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_sm120()
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    latent_kv = torch.empty(2, 512, dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros(2, dtype=torch.int32, device="cuda")
    cache = torch.empty(1, 2, _BYTES_PER_TOKEN, dtype=torch.uint8, device="cuda")

    with pytest.raises(ValueError, match="must be unique"):
        nvfp4_quantize_append_sparse_mla_cache(latent_kv, slots, cache)


def test_nvfp4_sparse_mla_pack_rejects_wrong_dtype():
    _require_sm120()
    latent_kv = torch.empty(1, 2, 512, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="bfloat16"):
        nvfp4_quantize_pack_sparse_mla_cache(latent_kv)


def test_nvfp4_sparse_mla_known_encoding_and_nonfinite_contract():
    _require_sm120()
    latent_kv = torch.zeros(1, 2, 512, dtype=torch.bfloat16, device="cuda")
    latent_kv[0, 0, :16] = torch.tensor(
        [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            2.0,
            -2.0,
            3.0,
            -3.0,
            4.0,
            -4.0,
            6.0,
            -6.0,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    latent_kv[0, 1, :16] = torch.tensor(
        [
            float("nan"),
            float("inf"),
            -float("inf"),
            torch.finfo(torch.bfloat16).max,
            torch.finfo(torch.bfloat16).tiny,
            -torch.finfo(torch.bfloat16).tiny,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            2.5,
            -2.5,
            5.0,
            -5.0,
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )

    cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv)
    data, scales = _split_cache(cache)
    packed_ref, scales_ref, _ = _reference_rows(latent_kv)

    # Low nibble is the earlier element. With scale=1, these are the exact
    # positive/negative E2M1 codes from zero through six.
    expected = torch.tensor(
        [0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7],
        dtype=torch.uint8,
        device="cuda",
    )
    torch.testing.assert_close(data[0, 0, :8], expected)
    assert scales[0, 0, 0].item() == 0x38  # E4M3 encoding of 1.0.

    # NaN/Inf, BF16 max/min-normal, and decision-boundary behavior is defined
    # to match FlashInfer's existing linear NVFP4 KV quantizer byte-for-byte.
    torch.testing.assert_close(
        data[..., :_PACKED_NOPE_BYTES].reshape_as(packed_ref), packed_ref
    )
    torch.testing.assert_close(scales[..., :28].reshape_as(scales_ref), scales_ref)


@pytest.mark.parametrize("iterations", [1, 7])
def test_nvfp4_sparse_mla_m16n32k64_matches_reference(iterations):
    _require_sm120()
    torch.manual_seed(20260831)
    a_bf16 = (torch.randn(16, 64, device="cuda") / 3).to(torch.bfloat16)
    b_bf16 = (torch.randn(32, 64, device="cuda") / 3).to(torch.bfloat16)
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    a, sfa = flashinfer.nvfp4_kv_quantize(a_bf16, global_scale)
    b, sfb = flashinfer.nvfp4_kv_quantize(b_bf16, global_scale)

    output = _nvfp4_sparse_mla_m16n32k64(
        a,
        b,
        sfa.view(torch.float8_e4m3fn),
        sfb.view(torch.float8_e4m3fn),
        iterations=iterations,
    )
    a_dequant = _dequantize_linear_nvfp4(a, sfa)
    b_dequant = _dequantize_linear_nvfp4(b, sfb)
    reference = torch.matmul(a_dequant, b_dequant.T) * iterations
    torch.testing.assert_close(output, reference, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("iterations", [1, 7])
def test_nvfp4_sparse_mla_candidate_major_pv_tile_matches_reference(iterations):
    _require_sm120()
    torch.manual_seed(20260901)
    a_bf16 = (torch.randn(16, 64, device="cuda") / 3).to(torch.bfloat16)
    # Quantize V in the mathematical [N, K] orientation, then repack its raw
    # E2M1 codes into sparse MLA's candidate-major [K, packed-N] cache view.
    b_bf16 = (torch.randn(8, 64, device="cuda") / 3).to(torch.bfloat16)
    global_scale = torch.ones(1, dtype=torch.float32, device="cuda")
    a, sfa = flashinfer.nvfp4_kv_quantize(a_bf16, global_scale)
    b_row_major, sfb = flashinfer.nvfp4_kv_quantize(b_bf16, global_scale)
    b_codes = torch.stack((b_row_major & 0xF, b_row_major >> 4), dim=-1).reshape(8, 64)
    b_codes_candidate_major = b_codes.T.contiguous()
    b_candidate_major = (
        b_codes_candidate_major[:, 0::2] | (b_codes_candidate_major[:, 1::2] << 4)
    ).contiguous()

    output = _nvfp4_sparse_mla_m16n8k64_candidate_major(
        a,
        b_candidate_major,
        sfa.view(torch.float8_e4m3fn),
        sfb.view(torch.float8_e4m3fn),
        iterations=iterations,
    )
    a_dequant = _dequantize_linear_nvfp4(a, sfa)
    b_dequant = _dequantize_linear_nvfp4(b_row_major, sfb)
    reference = torch.matmul(a_dequant, b_dequant.T) * iterations
    torch.testing.assert_close(output, reference, atol=2e-4, rtol=2e-4)


def test_nvfp4_sparse_mla_decode_workspace_has_no_global_vt() -> None:
    """Decode scratch contains only split outputs/LSE, not materialized V^T."""
    _require_sm120()
    num_tokens, num_heads, topk, extra_topk = 2, 128, 128, 128
    num_splits = topk // 64 + extra_topk // 64
    bytes_per_token = (
        num_heads * num_splits * (_D_NOPE + _D_ROPE) * 2
        + num_heads * num_splits * 4
        + num_heads * 4
    )
    workspace = torch.empty(
        num_tokens * bytes_per_token, dtype=torch.uint8, device="cuda"
    )
    chunk_tokens, mid_out, mid_lse, scratch_lse = _nvfp4_sparse_mla_workspace(
        workspace,
        num_tokens=num_tokens,
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        use_prefill=False,
    )

    assert chunk_tokens == num_tokens
    assert mid_out is not None and mid_lse is not None
    assert mid_out.shape == (num_tokens, num_heads, num_splits, 512)
    assert mid_lse.shape == (num_tokens, num_heads, num_splits)
    assert scratch_lse.shape == (num_tokens, num_heads)
    assert mid_out.data_ptr() == workspace.data_ptr()


@pytest.mark.parametrize(
    "topk,chunks_per_block,topk_len", [(128, 2, 111), (512, 6, 389)]
)
@pytest.mark.parametrize("with_sink", [False, True])
def test_nvfp4_sparse_mla_decode_matches_dequantized_reference(
    topk: int, chunks_per_block: int, topk_len: int, with_sink: bool
) -> None:
    """Cover the direct and two-split epilogues with online quantization."""
    _require_sm120()
    torch.manual_seed(20260902 + topk + int(with_sink))
    num_tokens, num_heads = 2, 128
    num_pages, page_size = 16, 64
    kv_bf16 = (
        torch.randn(
            num_pages,
            page_size,
            1,
            _D_NOPE + _D_ROPE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(
            num_tokens,
            num_heads,
            _D_NOPE + _D_ROPE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0,
        num_pages * page_size,
        (num_tokens, topk),
        dtype=torch.int32,
        device="cuda",
    )
    # Exercise both the explicit length and negative-index masks.
    indices[:, topk_len - 7 : topk_len] = -1
    topk_length = torch.full((num_tokens,), topk_len, dtype=torch.int32, device="cuda")
    attn_sink = (
        torch.linspace(-1.0, 1.0, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    sm_scale = (_D_NOPE + _D_ROPE) ** -0.5

    cache = nvfp4_quantize_pack_sparse_mla_cache(kv_bf16.squeeze(2))
    q_dequant = _dequantize_nvfp4_query(q)
    kv_dequant = _dequantize_nvfp4_cache(cache)
    reference, reference_lse = _reference_sparse_attention(
        q_dequant,
        kv_dequant,
        indices,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )
    output, lse = _nvfp4_sparse_mla_decode(
        q,
        cache,
        indices,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        chunks_per_block_override=chunks_per_block,
    )

    # Q/K use the exact dequantized NVFP4 operands above. The remaining delta
    # comes from online P quantization and the candidate-axis V requantization.
    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, reference_lse, atol=2e-2, rtol=2e-2)

    prefill_output, prefill_lse = _nvfp4_sparse_mla_prefill(
        q,
        cache,
        indices,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )
    torch.testing.assert_close(prefill_output, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(prefill_lse, reference_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [2, 65])
def test_nvfp4_sparse_mla_shared_wrapper_matches_reference(num_tokens: int) -> None:
    """The existing SM120 wrapper routes NVFP4 through its independent planner."""
    _require_sm120()
    torch.manual_seed(20260911 + num_tokens)
    num_heads, topk = 16, 128
    kv_bf16 = (
        torch.randn(4, 64, 512, dtype=torch.bfloat16, device="cuda") / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, 4 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    cache = nvfp4_quantize_pack_sparse_mla_cache(kv_bf16)
    output = torch.empty_like(q)
    runner = flashinfer.mla.SparseMLASm120Wrapper(
        max_num_tokens=num_tokens,
        max_num_heads=num_heads,
        kv_cache_format="nvfp4",
        device=q.device,
    )
    lse = runner.run(
        q,
        cache,
        indices,
        output,
        512**-0.5,
        return_lse=True,
    )

    reference, reference_lse = _reference_sparse_attention(
        _dequantize_nvfp4_query(q),
        _dequantize_nvfp4_cache(cache),
        indices,
        512**-0.5,
    )
    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, reference_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [2, 65])
def test_nvfp4_sparse_mla_shared_wrapper_normalizes_singleton_indices(
    num_tokens: int,
) -> None:
    """The shared FP8/NVFP4 ABI accepts [T, 1, topk] for both cache sections."""
    _require_sm120()
    torch.manual_seed(20260912 + num_tokens)
    num_heads, topk = 16, 128
    main_kv = torch.randn(4, 64, 512, dtype=torch.bfloat16, device="cuda") / 10
    extra_kv = torch.randn(64, 2, 512, dtype=torch.bfloat16, device="cuda") / 10
    q = (
        torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device="cuda")
        / 10
    )
    indices = torch.randint(
        0, 4 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    extra_indices = torch.randint(
        0, 64 * 2, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    main_cache = nvfp4_quantize_pack_sparse_mla_cache(main_kv)
    extra_cache = nvfp4_quantize_pack_sparse_mla_cache(extra_kv)
    output_2d = torch.empty_like(q)
    output_3d = torch.empty_like(q)
    runner = flashinfer.mla.SparseMLASm120Wrapper(
        max_num_tokens=num_tokens,
        max_num_heads=num_heads,
        kv_cache_format="nvfp4",
        device=q.device,
    )

    lse_2d = runner.run(
        q,
        main_cache,
        indices,
        output_2d,
        512**-0.5,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        return_lse=True,
    ).clone()
    lse_3d = runner.run(
        q,
        main_cache,
        indices.unsqueeze(1),
        output_3d,
        512**-0.5,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices.unsqueeze(1),
        return_lse=True,
    )

    torch.testing.assert_close(output_3d, output_2d, atol=0, rtol=0)
    torch.testing.assert_close(lse_3d, lse_2d, atol=0, rtol=0)


@pytest.mark.parametrize("extra_page_size,extra_topk", [(2, 128), (64, 512)])
@pytest.mark.parametrize("with_sink", [False, True])
def test_nvfp4_sparse_mla_decode_dual_cache_matches_reference(
    extra_page_size: int, extra_topk: int, with_sink: bool
) -> None:
    """Main and C4A/C128A cache sections share one online softmax."""
    _require_sm120()
    torch.manual_seed(20260904 + extra_page_size + int(with_sink))
    num_tokens, num_heads, main_topk = 2, 128, 128
    main_pages, main_page_size = 8, 64
    extra_pages = max(16, (extra_topk + extra_page_size - 1) // extra_page_size)
    main_bf16 = (
        torch.randn(
            main_pages,
            main_page_size,
            _D_NOPE + _D_ROPE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    extra_bf16 = (
        torch.randn(
            extra_pages,
            extra_page_size,
            _D_NOPE + _D_ROPE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(
            num_tokens,
            num_heads,
            _D_NOPE + _D_ROPE,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
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
    main_lengths = torch.tensor([111, 97], dtype=torch.int32, device="cuda")
    extra_lengths = torch.tensor(
        [extra_topk - 13, max(1, extra_topk - 29)],
        dtype=torch.int32,
        device="cuda",
    )
    main_indices[:, 91:96] = -1
    extra_indices[:, 37:43] = -1
    attn_sink = (
        torch.linspace(-1.0, 1.0, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )
    sm_scale = (_D_NOPE + _D_ROPE) ** -0.5

    main_cache = nvfp4_quantize_pack_sparse_mla_cache(main_bf16)
    extra_cache = nvfp4_quantize_pack_sparse_mla_cache(extra_bf16)
    main_dequant = _dequantize_nvfp4_cache(main_cache)
    extra_dequant = _dequantize_nvfp4_cache(extra_cache)
    q_dequant = _dequantize_nvfp4_query(q)

    ref_main_indices = main_indices.clone()
    ref_extra_indices = extra_indices.clone()
    for token in range(num_tokens):
        ref_main_indices[token, int(main_lengths[token].item()) :] = -1
        ref_extra_indices[token, int(extra_lengths[token].item()) :] = -1
    main_rows = main_pages * main_page_size
    virtual_kv = torch.cat(
        (main_dequant.reshape(-1, 512), extra_dequant.reshape(-1, 512)), dim=0
    ).reshape(1, -1, 1, 512)
    virtual_indices = torch.cat(
        (
            ref_main_indices,
            torch.where(
                ref_extra_indices < 0,
                ref_extra_indices,
                ref_extra_indices + main_rows,
            ),
        ),
        dim=1,
    )
    reference, reference_lse = _reference_sparse_attention(
        q_dequant,
        virtual_kv,
        virtual_indices,
        sm_scale,
        attn_sink=attn_sink,
    )

    total_splits = (main_topk + 63) // 64 + (extra_topk + 63) // 64
    output, lse = _nvfp4_sparse_mla_decode(
        q,
        main_cache,
        main_indices,
        sm_scale,
        topk_length=main_lengths,
        attn_sink=attn_sink,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
        chunks_per_block_override=(total_splits + 1) // 2,
    )

    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, reference_lse, atol=2e-2, rtol=2e-2)

    prefill_output, prefill_lse = _nvfp4_sparse_mla_prefill(
        q,
        main_cache,
        main_indices,
        sm_scale,
        topk_length=main_lengths,
        attn_sink=attn_sink,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_topk_length=extra_lengths,
    )
    torch.testing.assert_close(prefill_output, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(prefill_lse, reference_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_heads", [16, 32, 64])
def test_nvfp4_sparse_mla_supported_head_counts(num_heads: int) -> None:
    """Decode and prefill share the vLLM padded-head dispatch set."""
    _require_sm120()
    torch.manual_seed(20260905 + num_heads)
    num_tokens, topk = 1, 128
    kv_bf16 = (
        torch.randn(4, 64, 512, dtype=torch.bfloat16, device="cuda") / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, 4 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    cache = nvfp4_quantize_pack_sparse_mla_cache(kv_bf16)
    reference, reference_lse = _reference_sparse_attention(
        _dequantize_nvfp4_query(q),
        _dequantize_nvfp4_cache(cache),
        indices,
        512**-0.5,
    )

    decode_output, decode_lse = _nvfp4_sparse_mla_decode(
        q,
        cache,
        indices,
        512**-0.5,
        chunks_per_block_override=2,
    )
    prefill_output, prefill_lse = _nvfp4_sparse_mla_prefill(
        q, cache, indices, 512**-0.5
    )
    torch.testing.assert_close(decode_output, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(decode_lse, reference_lse, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(prefill_output, reference, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(prefill_lse, reference_lse, atol=2e-2, rtol=2e-2)


def test_nvfp4_sparse_mla_public_api_decode_dual_cache() -> None:
    """The DSv4 public dispatcher routes both NVFP4 cache segments."""
    _require_sm120()
    torch.manual_seed(20260906)
    num_tokens, num_heads = 2, 16
    main_topk, extra_topk = 128, 128
    main_pages, main_page_size = 4, 64
    extra_pages, extra_page_size = 64, 2
    main_bf16 = (
        torch.randn(
            main_pages,
            main_page_size,
            512,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    extra_bf16 = (
        torch.randn(
            extra_pages,
            extra_page_size,
            512,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)
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
    main_lengths = torch.tensor([111, 97], dtype=torch.int32, device="cuda")
    extra_lengths = torch.tensor([119, 103], dtype=torch.int32, device="cuda")
    main_cache = nvfp4_quantize_pack_sparse_mla_cache(main_bf16)
    extra_cache = nvfp4_quantize_pack_sparse_mla_cache(extra_bf16)

    main_ref_indices = main_indices.clone()
    extra_ref_indices = extra_indices.clone()
    for token in range(num_tokens):
        main_ref_indices[token, int(main_lengths[token].item()) :] = -1
        extra_ref_indices[token, int(extra_lengths[token].item()) :] = -1
    main_rows = main_pages * main_page_size
    virtual_kv = torch.cat(
        (
            _dequantize_nvfp4_cache(main_cache).reshape(-1, 512),
            _dequantize_nvfp4_cache(extra_cache).reshape(-1, 512),
        ),
        dim=0,
    ).reshape(1, -1, 1, 512)
    virtual_indices = torch.cat(
        (
            main_ref_indices,
            torch.where(
                extra_ref_indices < 0,
                extra_ref_indices,
                extra_ref_indices + main_rows,
            ),
        ),
        dim=1,
    )
    reference, _ = _reference_sparse_attention(
        _dequantize_nvfp4_query(q),
        virtual_kv,
        virtual_indices,
        512**-0.5,
    )
    workspace = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    output = torch.empty_like(q)
    returned = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q,
        swa_kv_cache=main_cache,
        workspace_buffer=workspace,
        sparse_indices=main_indices,
        compressed_kv_cache=extra_cache,
        swa_topk_lens=main_lengths,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lengths,
        out=output,
        bmm1_scale=512**-0.5,
        backend="sparse",
        kv_cache_format="nvfp4",
    )

    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)
    with pytest.raises(ValueError, match="head dim 584"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=main_cache,
            workspace_buffer=workspace,
            sparse_indices=main_indices,
            swa_topk_lens=main_lengths,
            bmm1_scale=512**-0.5,
            backend="sparse",
        )
    with pytest.raises(ValueError, match="workspace"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=main_cache,
            workspace_buffer=torch.empty(1, dtype=torch.uint8, device="cuda"),
            sparse_indices=main_indices,
            swa_topk_lens=main_lengths,
            bmm1_scale=512**-0.5,
            backend="sparse",
            kv_cache_format="nvfp4",
        )


def test_nvfp4_sparse_mla_public_api_prefill_minimal_workspace() -> None:
    """Streaming prefill only reserves the caller-owned final LSE scratch."""
    _require_sm120()
    torch.manual_seed(20260907)
    num_tokens, num_heads, topk = 129, 16, 128
    kv_bf16 = (
        torch.randn(4, 64, 512, dtype=torch.bfloat16, device="cuda") / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)
    indices = torch.randint(
        0, 4 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.full((num_tokens,), 117, dtype=torch.int32, device="cuda")
    cache = nvfp4_quantize_pack_sparse_mla_cache(kv_bf16)
    reference, _ = _reference_sparse_attention(
        _dequantize_nvfp4_query(q),
        _dequantize_nvfp4_cache(cache),
        indices,
        512**-0.5,
        topk_length=lengths,
    )
    bytes_per_token = num_heads * 4
    workspace = torch.empty(
        num_tokens * bytes_per_token, dtype=torch.uint8, device="cuda"
    )
    output = torch.empty_like(q)

    flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q,
        swa_kv_cache=cache,
        workspace_buffer=workspace,
        sparse_indices=indices,
        swa_topk_lens=lengths,
        out=output,
        bmm1_scale=512**-0.5,
        backend="sparse",
        kv_cache_format="nvfp4",
    )
    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)


def test_nvfp4_sparse_mla_public_api_empty_pp_slice() -> None:
    """Empty PP/CUDA-graph metadata is a no-op, not a workspace error."""
    _require_sm120()
    num_heads, topk = 16, 128
    q = torch.empty((0, num_heads, 512), dtype=torch.bfloat16, device="cuda")
    # vLLM prefill metadata retains its singleton q-length dimension.
    indices = torch.empty((0, 1, topk), dtype=torch.int32, device="cuda")
    lengths = torch.empty((0, 1), dtype=torch.int32, device="cuda")
    cache = torch.empty((1, 1, 64, _BYTES_PER_TOKEN), dtype=torch.uint8, device="cuda")
    workspace = torch.empty(1, dtype=torch.uint8, device="cuda")
    output = torch.empty_like(q)

    returned = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=q,
        swa_kv_cache=cache,
        workspace_buffer=workspace,
        sparse_indices=indices,
        swa_topk_lens=lengths,
        out=output,
        bmm1_scale=512**-0.5,
        backend="sparse",
        kv_cache_format="nvfp4",
    )

    assert returned.data_ptr() == output.data_ptr()
    assert returned.shape == (0, num_heads, 512)


@pytest.mark.parametrize("num_tokens", [2, 65])
def test_nvfp4_sparse_mla_public_api_cuda_graph(num_tokens: int) -> None:
    """Both public decode and prefill routes are replayable in a CUDA Graph."""
    _require_sm120()
    torch.manual_seed(20260908 + num_tokens)
    num_heads, topk = 16, 128
    kv_bf16 = (
        torch.randn(4, 64, 512, dtype=torch.bfloat16, device="cuda") / 10.0
    ).clamp(-1, 1)
    q = (
        torch.randn(num_tokens, num_heads, 512, dtype=torch.bfloat16, device="cuda")
        / 10.0
    ).clamp(-1, 1)
    replay_q = (torch.randn_like(q, dtype=torch.bfloat16, device="cuda") / 10.0).clamp(
        -1, 1
    )
    indices = torch.randint(
        0, 4 * 64, (num_tokens, topk), dtype=torch.int32, device="cuda"
    )
    lengths = torch.full((num_tokens,), topk, dtype=torch.int32, device="cuda")
    cache = nvfp4_quantize_pack_sparse_mla_cache(kv_bf16)
    workspace = torch.empty(8 << 20, dtype=torch.uint8, device="cuda")
    output = torch.empty_like(q)

    def run() -> None:
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=cache,
            workspace_buffer=workspace,
            sparse_indices=indices,
            swa_topk_lens=lengths,
            out=output,
            bmm1_scale=512**-0.5,
            backend="sparse",
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
    reference, _ = _reference_sparse_attention(
        _dequantize_nvfp4_query(replay_q),
        _dequantize_nvfp4_cache(cache),
        indices,
        512**-0.5,
        topk_length=lengths,
    )
    torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("with_sink", [False, True])
def test_nvfp4_sparse_mla_decode_zero_topk_length(with_sink: bool) -> None:
    """An empty sparse row produces zero output and sink-aware LSE."""
    _require_sm120()
    torch.manual_seed(20260903)
    num_heads, topk = 128, 128
    kv = torch.randn(1, 64, _D_NOPE + _D_ROPE, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(
        1, num_heads, _D_NOPE + _D_ROPE, dtype=torch.bfloat16, device="cuda"
    )
    cache = nvfp4_quantize_pack_sparse_mla_cache(kv)
    indices = torch.zeros((1, topk), dtype=torch.int32, device="cuda")
    topk_length = torch.zeros(1, dtype=torch.int32, device="cuda")
    attn_sink = (
        torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )

    output, lse = _nvfp4_sparse_mla_decode(
        q,
        cache,
        indices,
        (_D_NOPE + _D_ROPE) ** -0.5,
        topk_length=topk_length,
        attn_sink=attn_sink,
        chunks_per_block_override=2,
    )

    assert torch.count_nonzero(output) == 0
    if attn_sink is None:
        assert torch.all(lse < -1e29)
    else:
        torch.testing.assert_close(lse, attn_sink.unsqueeze(0) * math.log2(math.e))


@pytest.mark.parametrize("with_sink", [False, True])
def test_nvfp4_sparse_mla_invalid_nonempty_chunks_have_zero_probability(
    with_sink: bool,
) -> None:
    """Negative candidates stay masked when topk_length itself is nonzero."""
    _require_sm120()
    torch.manual_seed(20260913 + int(with_sink))
    num_heads, topk = 16, 128
    kv = torch.randn(2, 64, 512, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(1, num_heads, 512, dtype=torch.bfloat16, device="cuda")
    cache = nvfp4_quantize_pack_sparse_mla_cache(kv)
    indices = torch.full((1, topk), -1, dtype=torch.int32, device="cuda")
    attn_sink = (
        torch.linspace(-2.0, 2.0, num_heads, dtype=torch.float32, device="cuda")
        if with_sink
        else None
    )

    for attention in (_nvfp4_sparse_mla_decode, _nvfp4_sparse_mla_prefill):
        output, lse = attention(
            q,
            cache,
            indices,
            512**-0.5,
            attn_sink=attn_sink,
        )
        assert torch.count_nonzero(output) == 0
        if attn_sink is None:
            assert torch.all(lse < -1e29)
        else:
            torch.testing.assert_close(lse, attn_sink.unsqueeze(0) * math.log2(math.e))


@pytest.mark.parametrize("invalid_chunk", ["first", "last"])
def test_nvfp4_sparse_mla_skips_fully_invalid_chunk_in_nonempty_row(
    invalid_chunk: str,
) -> None:
    """An invalid tile must not change the online softmax around a valid tile."""
    _require_sm120()
    torch.manual_seed(20260915 + int(invalid_chunk == "last"))
    num_heads, topk = 16, 128
    kv = (torch.randn(2, 64, 512, dtype=torch.bfloat16, device="cuda") / 10).clamp(
        -1, 1
    )
    q = (
        torch.randn(1, num_heads, 512, dtype=torch.bfloat16, device="cuda") / 10
    ).clamp(-1, 1)
    cache = nvfp4_quantize_pack_sparse_mla_cache(kv)
    indices = torch.randint(0, 128, (1, topk), dtype=torch.int32, device="cuda")
    invalid_slice = slice(0, 64) if invalid_chunk == "first" else slice(64, 128)
    indices[:, invalid_slice] = -1
    reference, reference_lse = _reference_sparse_attention(
        _dequantize_nvfp4_query(q),
        _dequantize_nvfp4_cache(cache),
        indices,
        512**-0.5,
    )

    for attention in (_nvfp4_sparse_mla_decode, _nvfp4_sparse_mla_prefill):
        output, lse = attention(q, cache, indices, 512**-0.5)
        torch.testing.assert_close(output, reference, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(lse, reference_lse, atol=2e-2, rtol=2e-2)
