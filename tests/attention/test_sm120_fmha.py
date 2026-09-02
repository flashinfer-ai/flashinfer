# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness tests for packed SM120 FP8 FMHA APIs.

The same compiled ragged/paged handles are exercised with different sequence
lengths, batch sizes, total-token counts, and paged block-table widths.

Run on an SM120 (Blackwell GeForce) GPU:
    pytest tests/attention/test_sm120_fmha.py -v
"""

import math

import pytest
import torch

import flashinfer.attention.cute_dsl.sm120_fmha as sm120_fmha
from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available
from flashinfer.utils import get_compute_capability

if not is_cute_dsl_experimental_available():
    pytest.skip("SM120 CuTe DSL dependencies are unavailable", allow_module_level=True)

_device = torch.device("cuda") if torch.cuda.is_available() else None

pytestmark = pytest.mark.skipif(
    _device is None or get_compute_capability(_device) != (12, 0),
    reason="SM120 FMHA requires SM120 GPU (compute capability 12.0)",
)

from flashinfer.attention.cute_dsl.sm120_fmha import (
    sm120_fmha_fp8_paged_prefill,
    sm120_fmha_fp8_ragged_prefill,
)


@pytest.fixture
def clear_cutlass_dsl_version_check_cache():
    sm120_fmha._check_cutlass_dsl_version.cache_clear()
    yield
    sm120_fmha._check_cutlass_dsl_version.cache_clear()


@pytest.mark.parametrize("installed_version", ["4.7.0", "4.8.0"])
def test_cutlass_dsl_runtime_version_check_accepts_supported_versions(
    monkeypatch, installed_version, clear_cutlass_dsl_version_check_cache
):
    monkeypatch.setattr(sm120_fmha, "version", lambda _: installed_version)
    sm120_fmha._check_cutlass_dsl_version()


def test_cutlass_dsl_runtime_version_check_rejects_older_version(
    monkeypatch, clear_cutlass_dsl_version_check_cache
):
    monkeypatch.setattr(sm120_fmha, "version", lambda _: "4.6.1")
    with pytest.raises(RuntimeError, match=r"nvidia-cutlass-dsl>=4\.7\.0.*4\.6\.1"):
        sm120_fmha._check_cutlass_dsl_version()


# ---------------------------------------------------------------------------
# Float32 reference
# ---------------------------------------------------------------------------


def _ref_fmha_and_lse(
    q: torch.Tensor,  # (B, Sq, Hq, D)
    k: torch.Tensor,  # (B, Skv, Hkv, D)
    v: torch.Tensor,  # (B, Skv, Hkv, D)
    sm_scale: float,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 FMHA and log2 LSE with bottom-right-aligned causal masking."""
    B, Sq, Hq, D = q.shape
    _, Skv, Hkv, _ = k.shape

    q_f = q.float().permute(0, 2, 1, 3)  # (B, Hq, Sq, D)
    k_f = k.float().permute(0, 2, 1, 3)  # (B, Hkv, Skv, D)
    v_f = v.float().permute(0, 2, 1, 3)  # (B, Hkv, Skv, D)

    if Hq != Hkv:
        k_f = k_f.repeat_interleave(Hq // Hkv, dim=1)
        v_f = v_f.repeat_interleave(Hq // Hkv, dim=1)

    scores = torch.einsum("bhqd,bhkd->bhqk", q_f, k_f) * sm_scale

    if is_causal:
        q_offset = Skv - Sq
        q_idx = (torch.arange(Sq, device=q.device) + q_offset).view(1, 1, -1, 1)
        k_idx = torch.arange(Skv, device=q.device).view(1, 1, 1, -1)
        scores = scores.masked_fill(k_idx > q_idx, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1).permute(0, 2, 1) * math.log2(math.e)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", attn, v_f)
    return (
        out.permute(0, 2, 1, 3).to(
            q.dtype if q.dtype != torch.float8_e4m3fn else torch.float32
        ),
        lse,
    )


def _ref_fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
) -> torch.Tensor:
    return _ref_fmha_and_lse(q, k, v, sm_scale, is_causal)[0]


def _make_fp8(shape, dtype=torch.float8_e4m3fn, device="cuda"):
    """FP8 tensor with values in [-2, 2] (exactly representable in FP8)."""
    return torch.randint(-2, 3, shape, dtype=torch.float32, device=device).to(dtype)


def _tol(out_dtype):
    return dict(atol=0.2, rtol=0.2)


# ---------------------------------------------------------------------------
# Packed-contiguous ragged correctness and cache reuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_ragged_sequence_lengths_are_runtime(is_causal):
    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_ragged_kernel,
    )

    Hq, Hkv, D = 4, 2, 64
    sm_scale = 1.0 / math.sqrt(D)
    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()

    for q_lens, kv_lens in [
        ([17, 65], [33, 129]),
        ([128], [193]),
        ([7, 31, 97], [11, 64, 131]),
    ]:
        q_parts = [_make_fp8((length, Hq, D)) for length in q_lens]
        k_parts = [_make_fp8((length, Hkv, D)) for length in kv_lens]
        v_parts = [_make_fp8((length, Hkv, D)) for length in kv_lens]
        q = torch.cat(q_parts)
        k = torch.cat(k_parts)
        v = torch.cat(v_parts)
        o = torch.empty_like(q, dtype=torch.float16)
        cu_q = torch.tensor(
            [0, *torch.tensor(q_lens).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        cu_k = torch.tensor(
            [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
            device="cuda",
            dtype=torch.int32,
        )
        sm120_fmha_fp8_ragged_prefill(
            q,
            k,
            v,
            o,
            cu_q,
            cu_k,
            max_seqlen_q=max(q_lens),
            is_causal=is_causal,
            sm_scale=sm_scale,
        )
        refs = [
            _ref_fmha(
                q_part.unsqueeze(0),
                k_part.unsqueeze(0),
                v_part.unsqueeze(0),
                sm_scale,
                is_causal,
            )[0]
            for q_part, k_part, v_part in zip(q_parts, k_parts, v_parts, strict=False)
        ]
        torch.testing.assert_close(o, torch.cat(refs).to(o.dtype), **_tol(o.dtype))

    cache_info = compile_sm120_fmha_fp8_ragged_kernel.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 2


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize(
    "D,q_lens,kv_lens",
    [
        pytest.param(32, [17, 65], [33, 97], id="d32"),
        pytest.param(64, [17, 65], [33, 97], id="d64"),
        pytest.param(128, [17, 65], [33, 97], id="d128"),
        pytest.param(256, [129], [384], id="d256-three-stage"),
    ],
)
def test_sm120_ragged_return_lse(is_causal, D, q_lens, kv_lens):
    Hq, Hkv = 4, 2
    sm_scale = 1.0 / math.sqrt(D)
    q_parts = [_make_fp8((length, Hq, D)) for length in q_lens]
    k_parts = [_make_fp8((length, Hkv, D)) for length in kv_lens]
    v_parts = [_make_fp8((length, Hkv, D)) for length in kv_lens]
    q, k, v = torch.cat(q_parts), torch.cat(k_parts), torch.cat(v_parts)
    o = torch.empty_like(q, dtype=torch.float16)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    cu_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()],
        device=q.device,
        dtype=torch.int32,
    )
    cu_k = torch.tensor(
        [0, *torch.tensor(kv_lens).cumsum(0).tolist()],
        device=q.device,
        dtype=torch.int32,
    )

    sm120_fmha_fp8_ragged_prefill(
        q,
        k,
        v,
        o,
        cu_q,
        cu_k,
        max_seqlen_q=max(q_lens),
        is_causal=is_causal,
        sm_scale=sm_scale,
        lse=lse,
    )

    references = [
        _ref_fmha_and_lse(
            q_part.unsqueeze(0),
            k_part.unsqueeze(0),
            v_part.unsqueeze(0),
            sm_scale,
            is_causal,
        )
        for q_part, k_part, v_part in zip(q_parts, k_parts, v_parts, strict=False)
    ]
    torch.testing.assert_close(
        o,
        torch.cat([reference[0][0] for reference in references]).to(o.dtype),
        **_tol(o.dtype),
    )
    torch.testing.assert_close(
        lse,
        torch.cat([reference[1][0] for reference in references]),
        atol=2e-3,
        rtol=2e-3,
    )


def test_sm120_ragged_empty_kv_return_lse():
    Hq, Hkv, D = 4, 2, 64
    q = _make_fp8((17, Hq, D))
    k = _make_fp8((0, Hkv, D))
    v = _make_fp8((0, Hkv, D))
    o = torch.empty_like(q, dtype=torch.float16)
    lse = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
    cu_q = torch.tensor([0, q.shape[0]], device=q.device, dtype=torch.int32)
    cu_k = torch.tensor([0, 0], device=q.device, dtype=torch.int32)

    sm120_fmha_fp8_ragged_prefill(
        q,
        k,
        v,
        o,
        cu_q,
        cu_k,
        max_seqlen_q=q.shape[0],
        lse=lse,
    )

    torch.testing.assert_close(o, torch.zeros_like(o))
    assert torch.isneginf(lse).all()


# ---------------------------------------------------------------------------
# Paged prefill shared block-table contract
# ---------------------------------------------------------------------------


def _run_paged_case(
    q_lens,
    kv_lens,
    block_table_width,
    is_causal,
    return_lse=False,
):
    B, Hq, Hkv, D = len(q_lens), 4, 2, 64
    page_size = 32
    sm_scale = 1.0 / math.sqrt(D)
    q_parts = [_make_fp8((length, Hq, D)) for length in q_lens]
    k_parts = [_make_fp8((length, Hkv, D)) for length in kv_lens]
    v_parts = [_make_fp8((length, Hkv, D)) for length in kv_lens]
    q = torch.cat(q_parts)
    o = torch.empty_like(q, dtype=torch.float16)
    lse = (
        torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )
    page_counts = [(length + page_size - 1) // page_size for length in kv_lens]
    assert max(page_counts) <= block_table_width
    num_pages = sum(page_counts)
    physical_ids = list(reversed(range(num_pages)))
    block_tables = torch.zeros(B, block_table_width, device="cuda", dtype=torch.int32)
    kv_pool = torch.zeros(num_pages, 2, Hkv, page_size, D, device="cuda", dtype=q.dtype)
    k_pool, v_pool = kv_pool.unbind(dim=1)
    physical_cursor = 0
    for batch_idx, (k_part, v_part, page_count) in enumerate(
        zip(k_parts, v_parts, page_counts, strict=False)
    ):
        for logical_page in range(page_count):
            physical_page = physical_ids[physical_cursor]
            physical_cursor += 1
            block_tables[batch_idx, logical_page] = physical_page
            start = logical_page * page_size
            end = min(start + page_size, k_part.shape[0])
            k_pool[physical_page, :, : end - start].copy_(
                k_part[start:end].transpose(0, 1)
            )
            v_pool[physical_page, :, : end - start].copy_(
                v_part[start:end].transpose(0, 1)
            )

    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    seqlens_kv = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        o,
        block_tables,
        seqlens_kv,
        cu_seqlens_q,
        is_causal=is_causal,
        sm_scale=sm_scale,
        max_seqlen_q=max(q_lens),
        lse=lse,
    )
    references = [
        _ref_fmha_and_lse(
            q_part.unsqueeze(0),
            k_part.unsqueeze(0),
            v_part.unsqueeze(0),
            sm_scale,
            is_causal,
        )
        for q_part, k_part, v_part in zip(q_parts, k_parts, v_parts, strict=False)
    ]
    torch.testing.assert_close(
        o,
        torch.cat([reference[0][0] for reference in references]).to(o.dtype),
        **_tol(o.dtype),
    )
    if lse is not None:
        torch.testing.assert_close(
            lse,
            torch.cat([reference[1][0] for reference in references]),
            atol=2e-3,
            rtol=2e-3,
        )


@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_sequence_lengths_and_m_are_runtime(is_causal):
    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
    )

    compile_sm120_fmha_fp8_paged_kernel.cache_clear()
    _run_paged_case([17, 65], [33, 97], 4, is_causal)
    _run_paged_case([129], [193], 7, is_causal)
    _run_paged_case([7, 31, 97], [11, 64, 131], 6, is_causal)
    cache_info = compile_sm120_fmha_fp8_paged_kernel.cache_info()
    assert cache_info.misses == 1
    assert cache_info.hits == 2


@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_return_lse(is_causal):
    _run_paged_case([17, 65], [33, 97], 4, is_causal, return_lse=True)


def test_sm120_paged_prefill_rejects_dual_plane_page_indices():
    q = _make_fp8((64, 4, 64))
    k_pool = _make_fp8((4, 2, 32, 64))
    v_pool = _make_fp8((4, 2, 32, 64))
    o = torch.empty(64, 4, 64, device="cuda", dtype=torch.float16)
    dual_plane_indices = torch.zeros(1, 2, 4, device="cuda", dtype=torch.int32)
    seqlens_kv = torch.tensor([128], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 64], device="cuda", dtype=torch.int32)

    with pytest.raises(AssertionError, match="block_tables"):
        sm120_fmha_fp8_paged_prefill(
            q,
            k_pool,
            v_pool,
            o,
            dual_plane_indices,
            seqlens_kv,
            cu_seqlens_q,
        )


def test_sm120_paged_prefill_rejects_nhd_pool():
    q = _make_fp8((64, 4, 64))
    # NHD [P, page, Hkv, D] is intentionally unsupported.
    k_pool = _make_fp8((4, 32, 2, 64))
    v_pool = _make_fp8((4, 32, 2, 64))
    o = torch.empty(64, 4, 64, device="cuda", dtype=torch.float16)
    block_tables = torch.arange(4, device="cuda", dtype=torch.int32).unsqueeze(0)
    seqlens_kv = torch.tensor([128], device="cuda", dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 64], device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="cannot implement"):
        sm120_fmha_fp8_paged_prefill(
            q,
            k_pool,
            v_pool,
            o,
            block_tables,
            seqlens_kv,
            cu_seqlens_q,
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_sm120_ragged_unsupported_dtype_raises():
    q = torch.randn(128, 4, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(128, 4, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    o = torch.empty_like(q)
    cu = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    with pytest.raises((KeyError, RuntimeError, ValueError)):
        sm120_fmha_fp8_ragged_prefill(q, k, v, o, cu, cu)


def test_sm120_ragged_unsupported_head_dim_raises():
    q = _make_fp8((128, 4, 96))
    k = _make_fp8((128, 4, 96))
    v = _make_fp8((128, 4, 96))
    o = torch.empty(128, 4, 96, device="cuda", dtype=torch.float16)
    cu = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="cannot implement"):
        sm120_fmha_fp8_ragged_prefill(q, k, v, o, cu, cu)


@pytest.mark.parametrize(
    "make_lse,match",
    [
        (
            lambda q: torch.empty(
                q.shape[0], q.shape[1] + 1, device=q.device, dtype=torch.float32
            ),
            "shape",
        ),
        (
            lambda q: torch.empty(q.shape[:2], device=q.device, dtype=torch.float16),
            "dtype",
        ),
        (
            lambda q: torch.empty(
                q.shape[1], q.shape[0], device=q.device, dtype=torch.float32
            ).transpose(0, 1),
            "contiguous",
        ),
    ],
)
def test_sm120_ragged_invalid_lse_raises(make_lse, match):
    q = _make_fp8((64, 4, 64))
    k = _make_fp8((64, 2, 64))
    v = _make_fp8((64, 2, 64))
    o = torch.empty_like(q, dtype=torch.float16)
    cu = torch.tensor([0, 64], device=q.device, dtype=torch.int32)
    with pytest.raises(ValueError, match=match):
        sm120_fmha_fp8_ragged_prefill(q, k, v, o, cu, cu, lse=make_lse(q))
