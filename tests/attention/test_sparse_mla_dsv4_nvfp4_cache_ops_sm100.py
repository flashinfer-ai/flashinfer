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

"""DSV4 NVFP4 cache pack/append on SM100/SM103 and the CAKE NVFP4 gate.

The cache writers share one kernel source with the SM120 path; these tests
pin the bytes they produce on B200/GB300 to the same linear NVFP4 reference
the SM120 tests use (``_reference_rows``), and check that the public DSv4
entry point routes ``backend="cake"`` + ``kv_cache_format="nvfp4"`` into the
CAKE host on CC 10.x while ``backend="sparse"`` still refuses there. The
attention route itself is covered by ``tests/mla/test_cake_dsv4_nvfp4.py``.
"""

import pytest
import torch

import flashinfer
from flashinfer.mla import (
    nvfp4_quantize_append_sparse_mla_cache,
    nvfp4_quantize_pack_sparse_mla_cache,
)
from flashinfer.utils import get_compute_capability
from tests.attention.sparse_mla_test_utils import (
    _BYTES_PER_TOKEN,
    _D_NOPE,
    _D_ROPE,
    _PACKED_NOPE_BYTES,
    _dequantize_nvfp4_cache,
    _reference_rows,
    _split_cache,
)

_D_LATENT = _D_NOPE + _D_ROPE
_NUM_SCALES = _D_NOPE // 16


def _require_sm100_family() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    major, _ = get_compute_capability(torch.device("cuda"))
    if major != 10:
        pytest.skip("DSV4 NVFP4 cache ops on SM100/SM103 require CC 10.x")


def _assert_matches_reference(
    cache: torch.Tensor, latent_kv: torch.Tensor, *, expected_shape
) -> None:
    assert cache.dtype == torch.uint8
    assert tuple(cache.shape) == tuple(expected_shape)
    data, scales = _split_cache(cache)
    packed_ref, scales_ref, rope_ref = _reference_rows(latent_kv)
    torch.testing.assert_close(
        data[..., :_PACKED_NOPE_BYTES].reshape_as(packed_ref),
        packed_ref,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        data[..., _PACKED_NOPE_BYTES:].reshape_as(rope_ref), rope_ref, rtol=0, atol=0
    )
    torch.testing.assert_close(
        scales[..., :_NUM_SCALES].reshape_as(scales_ref), scales_ref, rtol=0, atol=0
    )
    assert torch.count_nonzero(scales[..., _NUM_SCALES:]) == 0


def test_cache_ops_are_declared_for_sm100_family() -> None:
    for fn in (
        nvfp4_quantize_pack_sparse_mla_cache,
        nvfp4_quantize_append_sparse_mla_cache,
    ):
        assert fn.is_compute_capability_supported(100)
        assert fn.is_compute_capability_supported(103)
        assert fn.is_compute_capability_supported(120)
        assert fn.is_compute_capability_supported(121)


@pytest.mark.parametrize("page_size", [2, 64])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_full_page_pack_matches_reference(page_size: int, kv_layout: str) -> None:
    _require_sm100_family()
    torch.manual_seed(100 + page_size)
    num_pages = 3
    latent_kv = torch.randn(
        num_pages, page_size, _D_LATENT, dtype=torch.bfloat16, device="cuda"
    )
    cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv, kv_layout=kv_layout)
    expected_shape = (
        (num_pages, 1, page_size, _BYTES_PER_TOKEN)
        if kv_layout == "HND"
        else (num_pages, page_size, 1, _BYTES_PER_TOKEN)
    )
    _assert_matches_reference(cache, latent_kv, expected_shape=expected_shape)

    # Independent pure-torch decode of the cache. E2M1 x E4M3 is exact, so with
    # the stored per-group scale ``s`` a round-to-nearest value is within half
    # the widest E2M1 step (``s``); when ``s`` rounded below ``amax / 6`` the
    # largest magnitude saturates at ``6 * s`` instead.
    decoded = _dequantize_nvfp4_cache(cache).reshape(-1, _D_LATENT)
    rows = latent_kv.reshape(-1, _D_LATENT).float()
    torch.testing.assert_close(decoded[:, _D_NOPE:], rows[:, _D_NOPE:], rtol=0, atol=0)
    _, scales = _split_cache(cache)
    s = (
        scales[..., :_NUM_SCALES]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(-1, _NUM_SCALES)
    )
    group_amax = rows[:, :_D_NOPE].reshape(-1, _NUM_SCALES, 16).abs().amax(dim=-1)
    bound = torch.maximum(s, group_amax - 6.0 * s).repeat_interleave(16, dim=-1)
    err = (decoded[:, :_D_NOPE] - rows[:, :_D_NOPE]).abs()
    assert torch.all(err <= bound + 1e-6), (err - bound).max().item()


@pytest.mark.parametrize("page_size", [2, 64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_incremental_append_matches_full_pack(page_size: int, index_dtype) -> None:
    _require_sm100_family()
    torch.manual_seed(7)
    num_pages = 2
    latent_kv = torch.randn(
        num_pages, page_size, _D_LATENT, dtype=torch.bfloat16, device="cuda"
    )
    full_cache = nvfp4_quantize_pack_sparse_mla_cache(latent_kv)
    append_cache = torch.full_like(full_cache, 0xA5)
    slots = torch.arange(num_pages * page_size, dtype=index_dtype, device="cuda")

    nvfp4_quantize_append_sparse_mla_cache(
        latent_kv.reshape(-1, _D_LATENT), slots, append_cache
    )
    torch.testing.assert_close(append_cache, full_cache, rtol=0, atol=0)


def test_incremental_append_accepts_3d_cache() -> None:
    """vLLM uses the latent-head-free [pages, page_size, bytes] shorthand."""
    _require_sm100_family()
    torch.manual_seed(8)
    latent_kv = torch.randn(2, 64, _D_LATENT, dtype=torch.bfloat16, device="cuda")
    expected = nvfp4_quantize_pack_sparse_mla_cache(latent_kv, kv_layout="NHD").squeeze(
        2
    )
    actual = torch.empty_like(expected)
    slots = torch.arange(128, dtype=torch.int64, device="cuda")
    nvfp4_quantize_append_sparse_mla_cache(
        latent_kv.reshape(-1, _D_LATENT), slots, actual
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("page_size", [2, 64])
def test_pack_and_append_accept_page_stride(page_size: int) -> None:
    """Packed vLLM pools leave padding between logical cache pages."""
    _require_sm100_family()
    torch.manual_seed(9 + page_size)
    num_pages = 3
    latent_kv = torch.randn(
        num_pages, page_size, _D_LATENT, dtype=torch.bfloat16, device="cuda"
    )
    expected = nvfp4_quantize_pack_sparse_mla_cache(latent_kv, kv_layout="NHD").squeeze(
        2
    )
    logical_page_bytes = page_size * _BYTES_PER_TOKEN
    page_stride = logical_page_bytes + 3 * _BYTES_PER_TOKEN
    backing = torch.full(
        (num_pages * page_stride,), 0xA5, dtype=torch.uint8, device="cuda"
    )
    append_cache = torch.as_strided(
        backing,
        size=(num_pages, page_size, _BYTES_PER_TOKEN),
        stride=(page_stride, _BYTES_PER_TOKEN, 1),
    )
    slots = torch.arange(num_pages * page_size, dtype=torch.int64, device="cuda")
    nvfp4_quantize_append_sparse_mla_cache(
        latent_kv.reshape(-1, _D_LATENT), slots, append_cache
    )
    for page in range(num_pages):
        torch.testing.assert_close(
            append_cache[page].reshape(-1), expected[page].reshape(-1), rtol=0, atol=0
        )
    # The padding between pages is never touched.
    pad = backing.view(num_pages, page_stride)[:, logical_page_bytes:]
    assert torch.all(pad == 0xA5)


@pytest.mark.parametrize("page_size", [2, 64])
def test_append_writes_only_selected_slots(page_size: int) -> None:
    _require_sm100_family()
    torch.manual_seed(11)
    num_pages = 2
    inputs = torch.randn(4, _D_LATENT, dtype=torch.bfloat16, device="cuda")
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
            scales[page_idx, entry_idx, :_NUM_SCALES], scales_ref[input_idx]
        )
        assert torch.count_nonzero(scales[page_idx, entry_idx, _NUM_SCALES:]) == 0
    if page_size > 2:
        assert torch.all(data[0, 1] == 0xA5)
        assert torch.all(scales[0, 1] == 0xA5)


@pytest.mark.parametrize("num_tokens", [4, 257])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_append_duplicate_slots_use_first_row(num_tokens: int, slot_dtype) -> None:
    """Covers both the warp-scan (<=256 rows) and owner-claim (>256) paths."""
    _require_sm100_family()
    torch.manual_seed(20260907 + num_tokens)
    latent_kv = torch.randn(num_tokens, _D_LATENT, dtype=torch.bfloat16, device="cuda")
    slots = torch.full((num_tokens,), -1, dtype=slot_dtype, device="cuda")
    slots[0] = 0
    slots[-1] = 0
    cache = torch.full((1, 2, _BYTES_PER_TOKEN), 0xA5, dtype=torch.uint8, device="cuda")

    nvfp4_quantize_append_sparse_mla_cache(latent_kv, slots, cache)
    expected = nvfp4_quantize_pack_sparse_mla_cache(latent_kv[:1].view(1, 1, _D_LATENT))
    actual_data, actual_scales = _split_cache(cache)
    expected_data, expected_scales = _split_cache(expected)
    torch.testing.assert_close(actual_data[0, 0], expected_data[0, 0], rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scales[0, 0], expected_scales[0, 0], rtol=0, atol=0
    )
    assert torch.all(actual_data[0, 1] == 0xA5)
    assert torch.all(actual_scales[0, 1] == 0xA5)


def test_known_encoding_matches_sm120_contract() -> None:
    """Fixed E2M1/E4M3 bytes: the SM120 contract must hold byte-for-byte here."""
    _require_sm100_family()
    latent_kv = torch.zeros(1, 2, _D_LATENT, dtype=torch.bfloat16, device="cuda")
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
    expected = torch.tensor(
        [0x80, 0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7],
        dtype=torch.uint8,
        device="cuda",
    )
    torch.testing.assert_close(data[0, 0, :8], expected)
    assert scales[0, 0, 0].item() == 0x38  # E4M3 encoding of 1.0.
    _assert_matches_reference(
        cache, latent_kv, expected_shape=(1, 1, 2, _BYTES_PER_TOKEN)
    )


def test_pack_rejects_wrong_dtype() -> None:
    _require_sm100_family()
    latent_kv = torch.empty(1, 2, _D_LATENT, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="bfloat16"):
        nvfp4_quantize_pack_sparse_mla_cache(latent_kv)


# Public DSv4 entry-point gate on CC 10.x.


def _gate_inputs(num_heads: int = 64, topk: int = 512):
    """Minimal valid NVFP4 inputs: one request, one query token."""
    device = torch.device("cuda")
    torch.manual_seed(3)
    swa = nvfp4_quantize_pack_sparse_mla_cache(
        torch.randn(2, 64, _D_LATENT, dtype=torch.bfloat16, device=device)
    )
    compressed = nvfp4_quantize_pack_sparse_mla_cache(
        torch.randn(8, 64, _D_LATENT, dtype=torch.bfloat16, device=device)
    )
    query = torch.randn(1, 1, num_heads, _D_LATENT, dtype=torch.bfloat16, device=device)
    sparse_indices = torch.zeros((1, topk), dtype=torch.int32, device=device)
    sparse_indices[0] = torch.arange(topk, dtype=torch.int32, device=device)
    swa_topk_lens = torch.tensor([topk], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([128], dtype=torch.int32, device=device)
    workspace = torch.empty(
        flashinfer.mla.get_cake_dsv4_workspace_bytes(
            1, num_heads, topk, torch.bfloat16
        ),
        dtype=torch.uint8,
        device=device,
    )
    return dict(
        query=query,
        swa_kv_cache=swa,
        workspace_buffer=workspace,
        sparse_indices=sparse_indices,
        compressed_kv_cache=compressed,
        swa_topk_lens=swa_topk_lens,
        seq_lens=seq_lens,
        bmm1_scale=_D_LATENT**-0.5,
        bmm2_scale=1.0,
        kv_cache_format="nvfp4",
    )


def test_cake_nvfp4_gate_reaches_cake_host() -> None:
    """The public entry runs the CAKE NVFP4 route on CC 10.x (one token, 64 heads)."""
    _require_sm100_family()
    inputs = _gate_inputs()
    out = flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(backend="cake", **inputs)
    torch.cuda.synchronize()
    assert out.shape == inputs["query"].shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    lse = flashinfer.mla.cake_dsv4_nvfp4_lse(inputs["workspace_buffer"], 1, 64)
    assert lse.shape == (1, 64) and torch.isfinite(lse).all()


def test_cake_nvfp4_gate_rejects_combined_lengths() -> None:
    """The NVFP4 main table is not a 128-slot window: sparse_topk_lens is refused."""
    _require_sm100_family()
    inputs = _gate_inputs()
    inputs["sparse_topk_lens"] = inputs.pop("swa_topk_lens")
    with pytest.raises(ValueError, match="requires swa_topk_lens"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(backend="cake", **inputs)


def test_cake_nvfp4_gate_rejects_dense_pools() -> None:
    """An NVFP4 request with a dense BF16 pool is a format mismatch, not a route."""
    _require_sm100_family()
    inputs = _gate_inputs()
    inputs["swa_kv_cache"] = torch.randn(
        2, 1, 64, _D_LATENT, dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(ValueError, match="packed uint8 swa_kv_cache with 384 bytes"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(backend="cake", **inputs)


def test_sparse_backend_still_refuses_sm100_family() -> None:
    _require_sm100_family()
    with pytest.raises(ValueError, match="backend='sparse' requires SM120/SM121"):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            backend="sparse", **_gate_inputs()
        )


def test_auto_backend_nvfp4_still_requires_sparse_or_cake() -> None:
    """``backend="auto"`` resolves to TRTLLM-GEN on CC 10.x, which has no NVFP4 cache."""
    _require_sm100_family()
    with pytest.raises(
        ValueError, match="requires backend='sparse'.*or backend='cake'"
    ):
        flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(
            backend="auto", **_gate_inputs()
        )


@pytest.mark.parametrize(
    "num_heads,expected",
    [
        (128, "nvfp4_h128_prefill_persistent"),
        (64, "nvfp4_h128_prefill_persistent"),
        (32, "nvfp4_h128_prefill_persistent_thin_heads"),
        (16, "nvfp4_h128_prefill_persistent_thin_heads"),
        (8, "nvfp4_h128_prefill_persistent_thin_heads"),
    ],
)
def test_run_cake_dsv4_nvfp4_route_key(num_heads: int, expected: str) -> None:
    """The routing key selects the epilogue form by head count, independent of a GPU."""
    from flashinfer.mla.cake_dsv4 import _route

    for arch in ("sm_100a", "sm_103a"):
        assert (
            _route(
                arch=arch,
                dtype=torch.bfloat16,
                num_heads=num_heads,
                max_q_len=1,
                ragged=False,
                sparse_topk=512,
                batch_size=1,
                compressed_page_size=64,
                num_query_tokens=1,
                kv_cache_format="nvfp4",
            )
            == expected
        )
