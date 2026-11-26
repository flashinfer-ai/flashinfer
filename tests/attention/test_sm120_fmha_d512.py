# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side and SM120 end-to-end tests for head_dim=512 FP8 prefill FMHA.

Host-side tests (tile selection, config/SMEM validation, compile cache keys,
Split-KV planning) run on CPU without a GPU. SM120 GPU tests verify ragged and
paged HND (page_size in {1, 16, 32, 64, 128}) correctness, Gemma 4 GQA shapes,
GQA packing, chunked prefill, empty/all-masked rows, forced Split-KV, and
64-bit Q/O pointer offsets past the signed Int32 2^31 boundary.

Run on an SM120 GPU:
    pytest tests/attention/test_sm120_fmha_d512.py -v
"""

import math
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

import flashinfer
from flashinfer.attention.cute_dsl.sm120_fmha import (
    SM120PrimsBatchPrefillBackend,
    _paged_kv_split_bounds,
    _select_tile_sizes,
    sm120_fmha_fp8_paged_prefill,
    sm120_fmha_fp8_ragged_prefill,
)
from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available
from flashinfer.utils import ceil_div, get_compute_capability


def _has_required_cutlass_dsl() -> bool:
    try:
        installed_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return False
    return Version(installed_version) >= Version("4.7.0")


_HAS_SM120 = (
    torch.cuda.is_available()
    and get_compute_capability(torch.device("cuda")) == (12, 0)
    and is_cute_dsl_experimental_available()
    and _has_required_cutlass_dsl()
)

_requires_sm120 = pytest.mark.skipif(
    not _HAS_SM120,
    reason="requires SM120 GPU (compute capability 12.0) and nvidia-cutlass-dsl>=4.7.0",
)


# ---------------------------------------------------------------------------
# CPU-only unit tests (tile sizing, validation, SMEM budget, compile keys)
# ---------------------------------------------------------------------------


def test_select_tile_sizes_d512_and_legacy():
    for head_dim in (32, 64, 128, 256):
        assert _select_tile_sizes(head_dim) == (128, 128)
    assert _select_tile_sizes(512) == (64, 64)


def test_validate_config_accepts_d512(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    backend = SM120PrimsBatchPrefillBackend(torch.device("cpu"))
    for out_dtype in (torch.float16, torch.bfloat16):
        backend._validate_config(
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            o_dtype=out_dtype,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=512,
            head_dim_vo=512,
        )

    with pytest.raises(ValueError, match="head dimensions"):
        backend._validate_config(
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            o_dtype=torch.bfloat16,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=512,
            head_dim_vo=256,
        )


def test_d512_can_implement_and_smem_budget():
    pytest.importorskip("cutlass.experimental")
    import cutlass

    from flashinfer.cute_dsl.attention.fmha.sm120.fmha_prefill_fp8_tma import (
        SM120FusedMultiHeadAttentionFP8ForwardTMA as Kernel,
    )

    # Valid D=512 ragged config (64x64).
    assert Kernel._can_implement(
        cutlass.Float8E4M3FN,
        cutlass.BFloat16,
        q_shape=(1, 1, 32, 512),
        k_shape=(1, 1, 4, 512),
        kv_tile=64,
        q_tile=64,
    )
    # Unsupported D=512 tile shapes must be rejected.
    for bad_q_tile, bad_kv_tile in ((128, 128), (128, 64), (64, 128)):
        assert not Kernel._can_implement(
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
            q_shape=(1, 1, 32, 512),
            k_shape=(1, 1, 4, 512),
            kv_tile=bad_kv_tile,
            q_tile=bad_q_tile,
        )

    # Valid D=512 paged configs, including page_size=128 > kv_tile=64.
    for page_size in (1, 16, 32, 64, 128):
        assert Kernel.can_implement_paged(
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
            q_shape=(1, 1, 32, 512),
            k_shape=(1, 1, 4, 512),
            kv_tile=64,
            q_tile=64,
            num_tokens_per_page=page_size,
        )
    for bad_page_size in (24, 48, 96):
        assert not Kernel.can_implement_paged(
            cutlass.Float8E4M3FN,
            cutlass.BFloat16,
            q_shape=(1, 1, 32, 512),
            k_shape=(1, 1, 4, 512),
            kv_tile=64,
            q_tile=64,
            num_tokens_per_page=bad_page_size,
        )

    # Verify Split-D attributes and dynamic SMEM budget (81,952 B <= 100,352 B).
    inst = Kernel(
        in_dtype=cutlass.Float8E4M3FN,
        out_dtype=cutlass.BFloat16,
        is_causal=True,
        head_tile=512,
        kv_tile=64,
        q_tile=64,
        use_paged_kv=True,
        num_tokens_per_page=128,
    )
    assert inst.split_d is True
    assert inst.num_warps == 12
    assert inst.num_compute_warps == 8
    assert inst.head_tile_per_warp == 256
    assert inst.kv_pipeline_stages == 2
    assert inst.tma_page_rows == 64
    assert inst.kv_page_chunks == 1
    assert inst.smem_elems == 65536
    assert inst.split_d_scratch_elems == 4096
    dyn_smem_bytes = (
        inst.smem_elems * cutlass.Float8E4M3FN.bytes
        + inst.split_d_scratch_elems * cutlass.Float32.bytes
        + inst.kv_pipeline_stages * 2 * cutlass.Int64.bytes
    )
    assert dyn_smem_bytes == 81952


def test_paged_kv_split_bounds_tile64():
    bounds = _paged_kv_split_bounds(
        q_lens=[16],
        kv_lens=[8192],
        num_qo_heads=32,
        causal=True,
        tile=64,
        num_ctas_per_wave=170,
        fixed_split_tokens=1024,
        max_chunks=16,
        gqa_pack_size=8,
        q_tile=64,
    )
    assert bounds is not None
    assert len(bounds) == 1
    req_bounds = bounds[0]
    assert len(req_bounds) == 9  # 8 chunks -> 9 boundary points
    assert req_bounds[0] == 0
    assert req_bounds[-1] == 128
    assert all(req_bounds[i] < req_bounds[i + 1] for i in range(len(req_bounds) - 1))


def test_d512_compile_cache_keys_distinct(monkeypatch):
    pytest.importorskip("cutlass.experimental")

    from flashinfer.cute_dsl.attention.fmha.sm120.compile import (
        compile_sm120_fmha_fp8_paged_kernel,
        compile_sm120_fmha_fp8_ragged_kernel,
    )
    from flashinfer.jit import cute_dsl_core

    observed = []

    def fake_build_and_load(module_name, kernel_name, compile_fn, **kwargs):
        observed.append((module_name, kernel_name))
        return kernel_name

    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
    compile_sm120_fmha_fp8_paged_kernel.cache_clear()
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: (12, 0)
    )
    monkeypatch.setattr(
        cute_dsl_core, "build_and_load_cute_dsl_kernel", fake_build_and_load
    )

    k_rag_512 = compile_sm120_fmha_fp8_ragged_kernel(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=512,
        is_causal=True,
        kv_tile=64,
        q_tile=64,
        device=torch.device("cpu"),
    )
    k_rag_256 = compile_sm120_fmha_fp8_ragged_kernel(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=256,
        is_causal=True,
        kv_tile=128,
        q_tile=128,
        device=torch.device("cpu"),
    )
    k_pag_512_p128 = compile_sm120_fmha_fp8_paged_kernel(
        in_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=512,
        is_causal=True,
        kv_tile=64,
        q_tile=64,
        num_tokens_per_page=128,
        device=torch.device("cpu"),
    )
    assert "_d512_" in k_rag_512 and "_kt64_qt64" in k_rag_512
    assert k_rag_512 != k_rag_256
    assert "_d512_" in k_pag_512_p128 and "_page128_" in k_pag_512_p128
    assert len(observed) == 3
    compile_sm120_fmha_fp8_ragged_kernel.cache_clear()
    compile_sm120_fmha_fp8_paged_kernel.cache_clear()


# ---------------------------------------------------------------------------
# Reference and data-generation helpers for SM120 GPU tests
# ---------------------------------------------------------------------------


def _make_fp8(shape, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        -2, 3, shape, dtype=torch.float32, device="cuda", generator=g
    ).to(torch.float8_e4m3fn)


def _ref_attention(q_b, k_b, v_b, sm_scale, is_causal, v_scale=1.0):
    """Exact FP32 reference for one request; returns (O, LSE_log2)."""
    sq, hq, d = q_b.shape
    skv, hkv, _ = k_b.shape
    if sq == 0:
        return (
            torch.empty((0, hq, d), dtype=torch.float32, device=q_b.device),
            torch.empty((0, hq), dtype=torch.float32, device=q_b.device),
        )
    if skv == 0:
        return (
            torch.zeros((sq, hq, d), dtype=torch.float32, device=q_b.device),
            torch.full((sq, hq), float("-inf"), dtype=torch.float32, device=q_b.device),
        )
    qf = q_b.float().transpose(0, 1)  # (Hq, sq, D)
    kf = k_b.float().transpose(0, 1)  # (Hkv, skv, D)
    vf = v_b.float().transpose(0, 1)
    if hq != hkv:
        kf = kf.repeat_interleave(hq // hkv, dim=0)
        vf = vf.repeat_interleave(hq // hkv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", qf, kf) * sm_scale
    if is_causal:
        q_offset = skv - sq
        q_pos = torch.arange(sq, device=q_b.device)[:, None] + q_offset
        k_pos = torch.arange(skv, device=q_b.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
    all_masked = torch.isneginf(scores).all(dim=-1)  # (Hq, sq)
    safe_scores = scores.masked_fill(all_masked.unsqueeze(-1), 0.0)
    probs = torch.softmax(safe_scores, dim=-1)
    probs = probs.masked_fill(all_masked.unsqueeze(-1), 0.0)
    out = torch.einsum("hqk,hkd->hqd", probs, vf).transpose(0, 1) * v_scale
    lse = torch.logsumexp(scores, dim=-1).transpose(0, 1) * math.log2(math.e)
    return out, lse


def _make_varlen_paged_kv(k_list, v_list, page_size, seed=0):
    """Pack per-request K/V lists into a shuffled combined HND paged pool."""
    hkv, d = k_list[0].shape[1], k_list[0].shape[2]
    pages_per_req = [ceil_div(k.shape[0], page_size) for k in k_list]
    max_pages = max(max(pages_per_req, default=1), 1)
    total_pages = max(sum(pages_per_req), 1)

    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(total_pages, generator=g).tolist()

    kv_pool = torch.zeros(
        (total_pages, 2, hkv, page_size, d),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    block_tables = torch.zeros(
        (len(k_list), max_pages), dtype=torch.int32, device="cuda"
    )
    page_indptr = [0]
    page_indices = []
    last_page_len = []

    cursor = 0
    for req_idx, (k_b, v_b) in enumerate(zip(k_list, v_list, strict=True)):
        skv = k_b.shape[0]
        n_pages = ceil_div(skv, page_size)
        for p in range(n_pages):
            phys = perm[cursor]
            cursor += 1
            block_tables[req_idx, p] = phys
            page_indices.append(phys)
            tok0 = p * page_size
            tok1 = min(tok0 + page_size, skv)
            kv_pool[phys, 0, :, : tok1 - tok0, :].copy_(k_b[tok0:tok1].transpose(0, 1))
            kv_pool[phys, 1, :, : tok1 - tok0, :].copy_(v_b[tok0:tok1].transpose(0, 1))
        page_indptr.append(len(page_indices))
        last_page_len.append(0 if skv == 0 else ((skv - 1) % page_size) + 1)

    k_pool, v_pool = kv_pool.unbind(dim=1)
    return (
        k_pool,
        v_pool,
        kv_pool,
        block_tables,
        torch.tensor(page_indptr, dtype=torch.int32, device="cuda"),
        torch.tensor(page_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(last_page_len, dtype=torch.int32, device="cuda"),
    )


# ---------------------------------------------------------------------------
# SM120 GPU Correctness Tests
# ---------------------------------------------------------------------------


@_requires_sm120
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("page_size", [None, 16, 128])
def test_sm120_d512_structured_exact_tight_tolerance(is_causal, page_size):
    """Structured low-entropy inputs verify Split-D half-sum & lane mapping tightly."""
    d = 512
    hq, hkv = 8, 2
    sq, skv = 130, 193
    sm_scale = 1.0 / math.sqrt(d)

    # Distinct patterns in D[0:256] and D[256:512] catch any missing/duplicated half-sum.
    q = torch.zeros((sq, hq, d), dtype=torch.float32, device="cuda")
    k = torch.zeros((skv, hkv, d), dtype=torch.float32, device="cuda")
    v = torch.zeros((skv, hkv, d), dtype=torch.float32, device="cuda")
    q[:, :, :256] = 0.25
    q[:, :, 256:] = -0.125
    k[:, :, :256] = (torch.arange(skv, device="cuda") % 3 - 1).float()[
        :, None, None
    ] * 0.25
    k[:, :, 256:] = (torch.arange(skv, device="cuda") % 5 - 2).float()[
        :, None, None
    ] * 0.25
    v[:, :, :256] = ((torch.arange(skv, device="cuda") % 7) - 3).float()[
        :, None, None
    ] * 0.25
    v[:, :, 256:] = ((torch.arange(skv, device="cuda") % 5) - 2).float()[
        :, None, None
    ] * 0.5
    q_f8, k_f8, v_f8 = (
        q.to(torch.float8_e4m3fn),
        k.to(torch.float8_e4m3fn),
        v.to(torch.float8_e4m3fn),
    )

    o = torch.empty((sq, hq, d), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((sq, hq), dtype=torch.float32, device="cuda")
    cu_q = torch.tensor([0, sq], dtype=torch.int32, device="cuda")

    if page_size is None:
        cu_k = torch.tensor([0, skv], dtype=torch.int32, device="cuda")
        sm120_fmha_fp8_ragged_prefill(
            q_f8,
            k_f8,
            v_f8,
            o,
            cu_q,
            cu_k,
            is_causal=is_causal,
            sm_scale=sm_scale,
            lse=lse,
        )
    else:
        k_pool, v_pool, _, bt, _, _, _ = _make_varlen_paged_kv(
            [k_f8], [v_f8], page_size, seed=7
        )
        seqlens_kv = torch.tensor([skv], dtype=torch.int32, device="cuda")
        sm120_fmha_fp8_paged_prefill(
            q_f8,
            k_pool,
            v_pool,
            o,
            bt,
            seqlens_kv,
            cu_q,
            is_causal=is_causal,
            sm_scale=sm_scale,
            lse=lse,
        )

    ref_o, ref_lse = _ref_attention(q_f8, k_f8, v_f8, sm_scale, is_causal)
    torch.testing.assert_close(o.float(), ref_o, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=2e-3, rtol=2e-3)


@_requires_sm120
@pytest.mark.parametrize(
    "hq,hkv",
    [
        (8, 8),  # MHA
        (8, 2),  # Gemma 4 E4B (G=4)
        (16, 2),  # Gemma 4 26B-A4B (G=8)
        (32, 4),  # Gemma 4 31B (G=8)
        (16, 1),  # Gemma 4 12B (G=16)
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_sm120_d512_ragged_gemma4_shapes(hq, hkv, is_causal, out_dtype):
    """Variable-length ragged D=512 across Gemma 4 GQA shapes and non-64-multiple lengths."""
    d = 512
    q_lens = [1, 15, 63, 64, 65, 130]
    kv_lens = [31, 64, 63, 127, 129, 255]
    sm_scale = 1.0 / math.sqrt(d)

    q_list = [_make_fp8((sq, hq, d), seed=10 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=20 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=30 + i) for i, sk in enumerate(kv_lens)]

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    cu_k = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(kv_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    o = torch.empty((q.shape[0], hq, d), dtype=out_dtype, device="cuda")
    lse = torch.empty((q.shape[0], hq), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_ragged_prefill(
        q, k, v, o, cu_q, cu_k, is_causal=is_causal, sm_scale=sm_scale, lse=lse
    )

    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal)
        refs.append(ro)
        lses.append(rl)
    torch.testing.assert_close(o.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, torch.cat(lses), atol=2e-3, rtol=2e-3)


@_requires_sm120
@pytest.mark.parametrize("page_size", [1, 16, 32, 64, 128])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("gqa_pack_size", [1, 8])
def test_sm120_d512_paged_all_page_sizes_and_gqa_pack(
    page_size, is_causal, gqa_pack_size
):
    """Paged HND D=512 across page_size in {1, 16, 32, 64, 128} and GQA packing."""
    d = 512
    hq, hkv = 32, 4
    q_lens = [1, 16, 17, 65, 127]
    kv_lens = [1, 63, 128, 193, 384]
    sm_scale = 1.0 / math.sqrt(d)

    q_list = [_make_fp8((sq, hq, d), seed=100 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=200 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=300 + i) for i, sk in enumerate(kv_lens)]

    q = torch.cat(q_list, dim=0)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    seqlens_kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    k_pool, v_pool, _, bt, _, _, _ = _make_varlen_paged_kv(
        k_list, v_list, page_size, seed=42
    )

    o = torch.empty((q.shape[0], hq, d), dtype=torch.bfloat16, device="cuda")
    lse = torch.empty((q.shape[0], hq), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        o,
        bt,
        seqlens_kv,
        cu_q,
        is_causal=is_causal,
        sm_scale=sm_scale,
        lse=lse,
        gqa_pack_size=gqa_pack_size,
    )

    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal)
        refs.append(ro)
        lses.append(rl)
    torch.testing.assert_close(o.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(lse, torch.cat(lses), atol=2e-3, rtol=2e-3)


@_requires_sm120
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_d512_empty_requests_and_all_masked_rows(is_causal):
    """Verify empty requests (qo=0, kv=0) and causal all-masked rows (qo > kv) yield O=0, LSE=-inf."""
    d = 512
    hq, hkv = 16, 2
    sm_scale = 1.0 / math.sqrt(d)

    # Req 0: qo=0, kv=64; Req 1: qo=17, kv=0; Req 2: qo=80, kv=30 (rows 0..49 all-masked if causal);
    # Req 3: qo=65, kv=130 (normal chunked prefill).
    q_lens = [0, 17, 80, 65]
    kv_lens = [64, 0, 30, 130]
    q_list = [_make_fp8((sq, hq, d), seed=500 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=600 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=700 + i) for i, sk in enumerate(kv_lens)]

    q = torch.cat(q_list, dim=0)
    k = torch.cat(k_list, dim=0)
    v = torch.cat(v_list, dim=0)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    cu_k = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(kv_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )

    o = torch.full(
        (q.shape[0], hq, d), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    lse = torch.full((q.shape[0], hq), float("nan"), dtype=torch.float32, device="cuda")
    sm120_fmha_fp8_ragged_prefill(
        q, k, v, o, cu_q, cu_k, is_causal=is_causal, sm_scale=sm_scale, lse=lse
    )

    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal)
        refs.append(ro)
        lses.append(rl)
    ref_o = torch.cat(refs)
    ref_lse = torch.cat(lses)

    assert not torch.isnan(o).any()
    assert not torch.isnan(lse).any()
    torch.testing.assert_close(o.float(), ref_o, atol=0.2, rtol=0.2)
    # Compare finite LSE entries tightly and verify -inf mask matches identically.
    inf_mask = torch.isneginf(ref_lse)
    assert torch.equal(torch.isneginf(lse), inf_mask)
    if (~inf_mask).any():
        torch.testing.assert_close(
            lse[~inf_mask], ref_lse[~inf_mask], atol=2e-3, rtol=2e-3
        )


@_requires_sm120
@pytest.mark.parametrize("page_size", [1, 16, 64, 128])
@pytest.mark.parametrize("num_kv_chunks", [2, 4, 8])
def test_sm120_d512_paged_split_kv_and_public_wrapper(page_size, num_kv_chunks):
    """Verify forced Split-KV via public BatchPrefillWithPagedKVCacheWrapper and direct merge_states."""
    d = 512
    hq, hkv = 32, 4
    q_lens = [15, 65]
    kv_lens = [512, 1024]
    q_scale, k_scale, v_scale = 0.75, 1.25, 0.5
    sm_scale = (q_scale * k_scale) / math.sqrt(d)

    q_list = [_make_fp8((sq, hq, d), seed=800 + i) for i, sq in enumerate(q_lens)]
    k_list = [_make_fp8((sk, hkv, d), seed=900 + i) for i, sk in enumerate(kv_lens)]
    v_list = [_make_fp8((sk, hkv, d), seed=950 + i) for i, sk in enumerate(kv_lens)]

    q = torch.cat(q_list, dim=0)
    cu_q = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    (
        k_pool,
        v_pool,
        kv_pool,
        bt,
        page_indptr,
        page_indices,
        last_page_len,
    ) = _make_varlen_paged_kv(k_list, v_list, page_size, seed=77)

    # 1. Public BatchPrefillWithPagedKVCacheWrapper with forced fixed_split_size.
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "HND", backend="cute-dsl-prims"
    )
    fixed_split_pages = max(1, ceil_div(max(kv_lens) // num_kv_chunks, page_size))
    wrapper.plan(
        cu_q,
        page_indptr,
        page_indices,
        last_page_len,
        hq,
        hkv,
        d,
        page_size,
        causal=True,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.bfloat16,
        fixed_split_size=fixed_split_pages,
    )
    out, out_lse = wrapper.run_return_lse(
        q,
        kv_pool,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        enable_pdl=True,
    )

    refs, lses = [], []
    for qb, kb, vb in zip(q_list, k_list, v_list, strict=True):
        ro, rl = _ref_attention(qb, kb, vb, sm_scale, is_causal=True, v_scale=v_scale)
        refs.append(ro)
        lses.append(rl)
    ref_o = torch.cat(refs)
    ref_lse = torch.cat(lses)
    torch.testing.assert_close(out.float(), ref_o, atol=0.2, rtol=0.2)
    torch.testing.assert_close(out_lse, ref_lse, atol=2e-3, rtol=2e-3)

    # 2. Direct forced num_kv_chunks with explicit kv_tile_ranges + merge_states.
    seqlens_kv = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    ranges = []
    for c in range(num_kv_chunks):
        for sk in kv_lens:
            n_tiles = ceil_div(sk, 64)
            t0 = (n_tiles * c) // num_kv_chunks
            t1 = (n_tiles * (c + 1)) // num_kv_chunks
            ranges.append([t0, t1])
    kv_tile_ranges = torch.tensor(ranges, dtype=torch.int32, device="cuda")

    total_q = q.shape[0]
    split_o = torch.empty(
        (total_q * num_kv_chunks, hq, d), dtype=torch.bfloat16, device="cuda"
    )
    split_lse = torch.empty(
        (total_q * num_kv_chunks, hq), dtype=torch.float32, device="cuda"
    )
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        split_o,
        bt,
        seqlens_kv,
        cu_q,
        is_causal=True,
        sm_scale=sm_scale,
        v_scale=v_scale,
        lse=split_lse,
        kv_tile_ranges=kv_tile_ranges,
    )
    merged_o, merged_lse = flashinfer.merge_states(
        split_o.view(total_q, num_kv_chunks, hq, d),
        split_lse.view(total_q, num_kv_chunks, hq),
    )
    torch.testing.assert_close(merged_o.float(), ref_o, atol=0.2, rtol=0.2)
    torch.testing.assert_close(merged_lse, ref_lse, atol=2e-3, rtol=2e-3)


@_requires_sm120
def test_sm120_d512_64bit_output_offset_past_int32_boundary():
    """Verify 64-bit O/Q indexing when partial O exceeds 2^31 elements (Int32 overflow)."""
    # total_q = 20_000, num_kv_chunks = 8, Hq = 32, D = 512 ->
    # 20_000 * 8 * 32 * 512 = 2,621,440,000 elements (> 2^31 - 1 = 2,147,483,647).
    d = 512
    hq, hkv = 32, 4
    num_chunks = 8
    page_size = 16
    req_q_len = 8
    req_kv_len = 128
    num_reqs = 2500  # total_q = 20_000
    total_q = num_reqs * req_q_len
    assert total_q * num_chunks * hq * d > (1 << 31)

    sm_scale = 1.0 / math.sqrt(d)
    q = torch.zeros((total_q, hq, d), dtype=torch.float8_e4m3fn, device="cuda")
    # Populate the last 2 requests (crossing the 2^31 element boundary) with non-trivial data.
    tail_reqs = 2
    tail_q = _make_fp8((tail_reqs * req_q_len, hq, d), seed=1234)
    q[-tail_reqs * req_q_len :].copy_(tail_q)

    # Compact shared pool of 8 pages (128 tokens) reused across requests.
    pages_per_req = req_kv_len // page_size
    k_ref = _make_fp8((req_kv_len, hkv, d), seed=2345)
    v_ref = _make_fp8((req_kv_len, hkv, d), seed=3456)
    kv_pool = torch.empty(
        (pages_per_req, 2, hkv, page_size, d),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv_pool[:, 0].copy_(
        k_ref.view(pages_per_req, page_size, hkv, d).permute(0, 2, 1, 3)
    )
    kv_pool[:, 1].copy_(
        v_ref.view(pages_per_req, page_size, hkv, d).permute(0, 2, 1, 3)
    )
    k_pool, v_pool = kv_pool.unbind(dim=1)
    bt = (
        torch.arange(pages_per_req, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(num_reqs, pages_per_req)
        .contiguous()
    )
    seqlens_kv = torch.full((num_reqs,), req_kv_len, dtype=torch.int32, device="cuda")
    cu_q = torch.arange(0, total_q + 1, req_q_len, dtype=torch.int32, device="cuda")

    # 2 KV tiles (128 / 64): put tile 0 in chunk 0, tile 1 in chunk 7 (highest offset > 2^31),
    # and empty ranges in chunks 1..6.
    kv_tile_ranges = torch.zeros(
        (num_chunks * num_reqs, 2), dtype=torch.int32, device="cuda"
    )
    kv_tile_ranges[:num_reqs, 0] = 0
    kv_tile_ranges[:num_reqs, 1] = 1
    kv_tile_ranges[(num_chunks - 1) * num_reqs :, 0] = 1
    kv_tile_ranges[(num_chunks - 1) * num_reqs :, 1] = 2

    split_o = torch.empty(
        (total_q * num_chunks, hq, d), dtype=torch.bfloat16, device="cuda"
    )
    split_lse = torch.empty(
        (total_q * num_chunks, hq), dtype=torch.float32, device="cuda"
    )
    sm120_fmha_fp8_paged_prefill(
        q,
        k_pool,
        v_pool,
        split_o,
        bt,
        seqlens_kv,
        cu_q,
        is_causal=True,
        sm_scale=sm_scale,
        lse=split_lse,
        kv_tile_ranges=kv_tile_ranges,
    )

    # Merge and verify only the tail requests whose split_o offsets exceed 2^31 elements.
    tail_rows = tail_reqs * req_q_len
    tail_split_o = split_o[-tail_rows * num_chunks :].view(tail_rows, num_chunks, hq, d)
    tail_split_lse = split_lse[-tail_rows * num_chunks :].view(
        tail_rows, num_chunks, hq
    )
    merged_o, merged_lse = flashinfer.merge_states(tail_split_o, tail_split_lse)

    refs, lses = [], []
    for r in range(tail_reqs):
        qb = tail_q[r * req_q_len : (r + 1) * req_q_len]
        ro, rl = _ref_attention(qb, k_ref, v_ref, sm_scale, is_causal=True)
        refs.append(ro)
        lses.append(rl)
    torch.testing.assert_close(merged_o.float(), torch.cat(refs), atol=0.2, rtol=0.2)
    torch.testing.assert_close(merged_lse, torch.cat(lses), atol=2e-3, rtol=2e-3)
