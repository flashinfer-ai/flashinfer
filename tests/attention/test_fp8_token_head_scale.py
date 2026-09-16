"""
Copyright (c) 2024 by FlashInfer team.

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

"""End-to-end tests for FP8 K/V cache with per-(token, head) scale.

The scale is a separate float32 tensor mirroring the KV layout minus
``head_dim``:

- ragged/single: NHD ``[kv_len, num_kv_heads]`` / HND ``[num_kv_heads, kv_len]``;
  the scale strides must mirror the KV strides divided by ``head_dim`` (the
  kernel derives the scale strides from the KV strides, as the NVFP4 path
  does)
- paged: NHD ``[num_pages, page_size, num_kv_heads]`` /
  HND ``[num_pages, num_kv_heads, page_size]``; the scale may be a
  non-contiguous view sliced out of an inline slot tensor whose last dim is
  ``head_dim + 16`` (the paged path is the one that supports such views)

The covered paths mirror the NVFP4 FA2 paths: single prefill (standalone,
derived from ``is_float8(kv) and kv_cache_sf is not None``, mirroring the
NVFP4 block-scale derivation), batch prefill ragged/paged, and batch decode
paged (the wrappers take an explicit ``use_token_head_sf=True`` at plan
time). Decode has no CUDA-core kernel: the Python layer routes to the
tensor-core (fa2 prefill) path, which reuses the prefill kernel. The
standalone ``single_decode_with_kv_cache`` is intentionally not covered
(NVFP4 has no standalone decode path either).

Each config computes three outputs from the same ``q``:

- ``o_orig`` -- 16-bit kernel on the original random K/V (pre-quantization)
- ``o_ref``  -- 16-bit kernel on the dequantized K/V
- ``o_fp8``  -- token-head-scale FP8 kernel (the feature under test)

The reference K/V are generated in ``q_dtype`` because the 16-bit FA2 kernel
requires ``DTypeQ == DTypeKV``.

Two cosines are checked:
1. ``cos(o_fp8, o_ref)``: the FP8 kernel must apply the per-(token, head) scale
   correctly, matching the 16-bit kernel on the dequantized values.
2. ``cos(o_orig, o_fp8) > 0.995``: FP8 round-trip accuracy.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


def _cc_major() -> int:
    return get_compute_capability(torch.device("cuda:0"))[0]


def _skip_bf16_on_sm75(q_dtype: torch.dtype):
    if q_dtype == torch.bfloat16 and _cc_major() < 8:
        pytest.skip("bf16 Q is not supported with FP8 token-head scale on SM75")


def quantize_token_head(x_ref: torch.Tensor, dtype: torch.dtype, head_dim: int):
    """Quantize ``x_ref`` per-(token, head) to FP8.

    Returns ``(x_fp8, scale, dequant)`` where ``x_fp8`` has the standard KV
    layout (last dim ``head_dim``), ``scale`` is float32 with shape
    ``x_ref.shape[:-1]`` (the KV layout minus head_dim), and ``dequant`` is
    the 16-bit reference input **in ``x_ref``'s dtype** (so the 16-bit
    reference kernel sees ``DTypeKV == DTypeQ`` when ``x_ref`` is built in
    ``q_dtype``).
    """
    fp8_max = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
    # Scale is float32 (the kernel requirement); compute in float32 so the
    # division is not done in 16-bit.
    scale = (x_ref.abs().to(torch.float32).amax(dim=-1) / fp8_max).clamp(min=1e-12)
    x_fp8 = (x_ref / scale.unsqueeze(-1)).to(dtype)
    dequant = (x_fp8.to(x_ref.dtype) * scale.unsqueeze(-1)).to(x_ref.dtype)
    return x_fp8, scale, dequant


def make_inline_slot(x_ref: torch.Tensor, dtype: torch.dtype, head_dim: int):
    """Pack ``x_ref`` into an inline slot tensor (last dim ``head_dim + 16``).

    The slot layout is ``[head_dim FP8 bytes][float32 scale][pad to 16B]`` —
    the legacy in-slot layout, kept here only as a *source* for the view
    tests: the KV data and the scale are sliced out as non-contiguous views.
    """
    x_fp8, scale, _ = quantize_token_head(x_ref, dtype, head_dim)
    slot = torch.zeros(
        *x_ref.shape[:-1], head_dim + 16, dtype=torch.uint8, device=x_ref.device
    )
    slot[..., :head_dim] = x_fp8.view(torch.uint8)
    slot[..., head_dim : head_dim + 4] = (
        scale.to(torch.float32).unsqueeze(-1).view(torch.uint8)
    )
    return slot


def extract_views(slot: torch.Tensor, dtype: torch.dtype, head_dim: int):
    """Slice the KV data and the scale out of an inline slot tensor.

    Returns ``(kv_view, sf_view)``: ``kv_view`` is the FP8 KV data (last dim
    ``head_dim``) and ``sf_view`` is the float32 scale (the slot layout minus
    head_dim). Both are non-contiguous views sharing the slot's storage.
    """
    kv_view = slot.view(dtype)[..., :head_dim]
    sf_view = slot[..., head_dim : head_dim + 4].view(torch.float32).squeeze(-1)
    return kv_view, sf_view


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _check(
    o_fp8: torch.Tensor,
    o_ref: torch.Tensor,
    o_orig: torch.Tensor,
    name: str,
    correctness_thresh: float = 0.9998,
):
    """Two-part validation:
    - correctness: ``o_fp8`` (token-head-scale kernel) vs ``o_ref`` (16-bit on dequant)
    - quantization: ``o_orig`` (16-bit on original) vs ``o_fp8`` > 0.995

    ``correctness_thresh`` is relaxed for asymmetric head_dim (192).
    """
    cos_correct = _cos(o_fp8, o_ref)
    assert cos_correct > correctness_thresh, (
        f"{name} correctness cos {cos_correct} <= {correctness_thresh}"
    )
    cos_quant = _cos(o_orig, o_fp8)
    assert cos_quant > 0.995, f"{name} quantization cos {cos_quant} <= 0.995"


def _heads(gqa: bool):
    """Return (num_qo_heads, num_kv_heads). MHA: group 1; GQA: group 2."""
    if gqa:
        return 8, 4
    return 4, 4


def _sizes(head_dim: int):
    """Small problem sizes (SM75 has limited memory; head_dim 256 shrinks seq)."""
    if head_dim >= 256:
        return dict(batch_size=2, qo_len=16, kv_len=32)
    return dict(batch_size=3, qo_len=32, kv_len=64)


_WS = 64 * 1024 * 1024  # 64 MiB workspace (kept small for SM75)


def _paged_kv(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    head_dim,
    q_dtype,
    dtype,
    dev,
    kv_layout="HND",
):
    """Build paged indices, the three KV caches (orig/dequant/fp8), and the
    scale tensors.

    ``kv_layout`` selects the cache shape (the K/V head vector is always the
    last dim, so ``quantize_token_head`` is layout-agnostic):
      - HND: ``[num_pages, 2, num_kv_heads, page_size, head_dim]``
      - NHD: ``[num_pages, 2, page_size, num_kv_heads, head_dim]``

    Returns ``(indptr, indices, last_page_len, kv_fp8, kv_deq, kv_orig, kv_sf)``
    where ``kv_sf`` is a ``(k_sf, v_sf)`` tuple of float32 tensors mirroring
    the KV layout minus head_dim (3-D paged).
    """
    indptr = [0]
    indices = []
    last_page_len = []
    for _ in range(batch_size):
        num_pages = (kv_len + page_size - 1) // page_size
        indptr.append(indptr[-1] + num_pages)
        indices.extend(range(indptr[-1] - num_pages, indptr[-1]))
        last_page_len.append(kv_len % page_size or page_size)
    num_pages = indptr[-1]
    indptr = torch.tensor(indptr, dtype=torch.int32).to(dev)
    indices = torch.tensor(indices, dtype=torch.int32).to(dev)
    last_page_len = torch.tensor(last_page_len, dtype=torch.int32).to(dev)

    if kv_layout == "HND":
        inner = (num_pages, num_kv_heads, page_size, head_dim)
    else:  # NHD
        inner = (num_pages, page_size, num_kv_heads, head_dim)
    k_ref = (0.05 * torch.randn(*inner, dtype=q_dtype)).to(dev)
    v_ref = (0.05 * torch.randn(*inner, dtype=q_dtype)).to(dev)
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, dtype, head_dim)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, dtype, head_dim)
    kv_fp8 = torch.cat([k_fp8.unsqueeze(1), v_fp8.unsqueeze(1)], dim=1)
    kv_deq = torch.cat([k_deq.unsqueeze(1), v_deq.unsqueeze(1)], dim=1)
    kv_orig = torch.cat([k_ref.unsqueeze(1), v_ref.unsqueeze(1)], dim=1)
    kv_sf = (k_sf, v_sf)
    return indptr, indices, last_page_len, kv_fp8, kv_deq, kv_orig, kv_sf


# ---------------------------------------------------------------------------
# 1. single prefill (ragged)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_single_prefill_token_head_scale(
    head_dim, dtype, q_dtype, gqa, causal, kv_layout
):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes(head_dim)
    qo_len, kv_len = sz["qo_len"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    # K/V reference in q_dtype: the 16-bit FA2 kernel requires DTypeQ == DTypeKV.
    # NHD: [kv_len, num_kv_heads, head_dim]; HND: [num_kv_heads, kv_len, head_dim].
    # quantize_token_head is layout-agnostic (it reduces over the last dim), so the
    # scale shape follows the KV layout: NHD [kv_len, num_kv_heads] / HND
    # [num_kv_heads, kv_len].
    if kv_layout == "NHD":
        k_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(
            dev
        )
        v_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(
            dev
        )
    else:  # HND
        k_ref = 0.05 * torch.randn(num_kv_heads, kv_len, head_dim, dtype=q_dtype).to(
            dev
        )
        v_ref = 0.05 * torch.randn(num_kv_heads, kv_len, head_dim, dtype=q_dtype).to(
            dev
        )
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, dtype, head_dim)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, dtype, head_dim)

    o_orig = flashinfer.single_prefill_with_kv_cache(
        q, k_ref, v_ref, causal=causal, backend="fa2", kv_layout=kv_layout
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_deq, v_deq, causal=causal, backend="fa2", kv_layout=kv_layout
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q,
        k_fp8,
        v_fp8,
        causal=causal,
        backend="fa2",
        kv_layout=kv_layout,
        kv_cache_sf=(k_sf, v_sf),
    )

    assert o_fp8.shape == (qo_len, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "single_prefill")


# ---------------------------------------------------------------------------
# 2. batch prefill (ragged)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_batch_prefill_ragged_token_head_scale(
    head_dim, dtype, q_dtype, gqa, causal, kv_layout
):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes(head_dim)
    batch_size, qo_len, kv_len = sz["batch_size"], sz["qo_len"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(dev) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(dev) * kv_len
    total_qo = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()

    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    # NHD: [total_kv, num_kv_heads, head_dim]; HND: [num_kv_heads, total_kv, head_dim].
    # quantize_token_head is layout-agnostic (reduces over the last dim), so the scale
    # shape follows the KV layout: NHD [total_kv, num_kv_heads] / HND
    # [num_kv_heads, total_kv].
    if kv_layout == "NHD":
        k_ref = 0.05 * torch.randn(total_kv, num_kv_heads, head_dim, dtype=q_dtype).to(
            dev
        )
        v_ref = 0.05 * torch.randn(total_kv, num_kv_heads, head_dim, dtype=q_dtype).to(
            dev
        )
    else:  # HND
        k_ref = 0.05 * torch.randn(num_kv_heads, total_kv, head_dim, dtype=q_dtype).to(
            dev
        )
        v_ref = 0.05 * torch.randn(num_kv_heads, total_kv, head_dim, dtype=q_dtype).to(
            dev
        )
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, dtype, head_dim)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, dtype, head_dim)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    ref = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend="fa2"
    )
    ref.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_orig = ref.run(q, k_ref, v_ref)
    o_ref = ref.run(q, k_deq, v_deq)

    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend="fa2"
    )
    fi.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=dtype,
        use_token_head_sf=True,
    )
    o_fp8 = fi.run(q, k_fp8, v_fp8, kv_cache_sf=(k_sf, v_sf))

    assert o_fp8.shape == (total_qo, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_prefill_ragged")


# ---------------------------------------------------------------------------
# 3. batch prefill (paged)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_paged_token_head_scale(
    head_dim, dtype, q_dtype, gqa, causal, kv_layout
):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes(head_dim)
    batch_size, qo_len, kv_len = sz["batch_size"], sz["qo_len"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)
    page_size = 16

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(dev) * qo_len
    total_qo = qo_indptr[-1].item()

    (
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        paged_kv_cache,
        k_deq_cache,
        k_orig_cache,
        kv_sf,
    ) = _paged_kv(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        q_dtype,
        dtype,
        dev,
        kv_layout,
    )

    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=q_dtype).to(dev)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend="fa2"
    )
    ref.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_orig = ref.run(q, k_orig_cache)
    o_ref = ref.run(q, k_deq_cache)

    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend="fa2"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=dtype,
        use_token_head_sf=True,
    )
    o_fp8 = fi.run(q, paged_kv_cache, kv_cache_sf=kv_sf)

    assert o_fp8.shape == (total_qo, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_prefill_paged")


# ---------------------------------------------------------------------------
# 4. batch prefill (paged) -- VO-split path (HEAD_DIM_VO >= 512)
#
# Exercises vosplit_compute_pv, which (unlike compute_sfm_v) reads the P
# fragment from p_smem and must apply the per-(token, head) V scale to it
# before the PV MMA. HEAD_DIM_VO = 512 -> NUM_MMA_D_VO = 32 (> 16 and divisible
# by NUM_WARPS_KV), so the kernel takes the VO-split path. The paged cache is a
# single tensor, so head_dim_qk == head_dim_vo == 512.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_batch_prefill_paged_token_head_scale_vosplit(kv_layout, dtype, q_dtype):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    head_dim = 512
    batch_size, qo_len, kv_len = 1, 16, 32
    num_qo_heads = num_kv_heads = 4
    page_size = 16

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(dev) * qo_len
    total_qo = qo_indptr[-1].item()

    (
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        paged_kv_cache,
        k_deq_cache,
        k_orig_cache,
        kv_sf,
    ) = _paged_kv(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        q_dtype,
        dtype,
        dev,
        kv_layout,
    )

    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=q_dtype).to(dev)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend="fa2"
    )
    ref.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_orig = ref.run(q, k_orig_cache)
    o_ref = ref.run(q, k_deq_cache)

    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend="fa2"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=q_dtype,
        kv_data_type=dtype,
        use_token_head_sf=True,
    )
    o_fp8 = fi.run(q, paged_kv_cache, kv_cache_sf=kv_sf)

    assert o_fp8.shape == (total_qo, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_prefill_paged_vosplit")


# ---------------------------------------------------------------------------
# 5. batch decode (paged)
#
# plan(use_token_head_sf=True) forces the tensor-core (fa2 paged-prefill)
# path, the same path exercised by test_batch_prefill_paged_token_head_scale.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
def test_batch_decode_paged_token_head_scale(head_dim, dtype, q_dtype, gqa, kv_layout):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes(head_dim)
    batch_size, kv_len = sz["batch_size"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)
    page_size = 16

    (
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        paged_kv_cache,
        k_deq_cache,
        k_orig_cache,
        kv_sf,
    ) = _paged_kv(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        q_dtype,
        dtype,
        dev,
        kv_layout,
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=q_dtype).to(dev)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout=kv_layout)
    ref.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_orig = ref.run(q, k_orig_cache)
    o_ref = ref.run(q, k_deq_cache)

    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout=kv_layout)
    fi.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=dtype,
        use_token_head_sf=True,
    )
    assert fi.use_tensor_cores  # plan() forces the tensor-core path
    o_fp8 = fi.run(q, paged_kv_cache, kv_cache_sf=kv_sf)

    assert o_fp8.shape == (batch_size, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_decode_paged")


# ---------------------------------------------------------------------------
# 5. extended configs: sliding window, logits soft cap, asymmetric head dims
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_single_prefill_token_head_scale_sliding_window(q_dtype):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    head_dim = 128
    qo_len = kv_len = 32
    window_left = 16
    num_qo_heads = num_kv_heads = 4

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    k_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, torch.float8_e4m3fn, head_dim)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, torch.float8_e4m3fn, head_dim)

    o_orig = flashinfer.single_prefill_with_kv_cache(
        q, k_ref, v_ref, window_left=window_left, backend="fa2"
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_deq, v_deq, window_left=window_left, backend="fa2"
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q,
        k_fp8,
        v_fp8,
        window_left=window_left,
        backend="fa2",
        kv_cache_sf=(k_sf, v_sf),
    )
    _check(o_fp8, o_ref, o_orig, "sliding_window")


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_single_prefill_token_head_scale_logits_soft_cap(q_dtype):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    head_dim = 128
    qo_len = kv_len = 32
    logits_soft_cap = 50.0
    num_qo_heads = num_kv_heads = 4

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    k_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, torch.float8_e4m3fn, head_dim)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, torch.float8_e4m3fn, head_dim)

    o_orig = flashinfer.single_prefill_with_kv_cache(
        q, k_ref, v_ref, logits_soft_cap=logits_soft_cap, backend="fa2"
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_deq, v_deq, logits_soft_cap=logits_soft_cap, backend="fa2"
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q,
        k_fp8,
        v_fp8,
        logits_soft_cap=logits_soft_cap,
        backend="fa2",
        kv_cache_sf=(k_sf, v_sf),
    )
    _check(o_fp8, o_ref, o_orig, "logits_soft_cap")


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_single_prefill_token_head_scale_asym_head_dim(q_dtype):
    # head_dim_qk=192, head_dim_vo=128: K and V carry different head dims, and
    # head_dim_qk is a 16-multiple outside {64,128,256}.
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    head_dim_qk, head_dim_vo = 192, 128
    qo_len = kv_len = 32
    num_qo_heads = num_kv_heads = 4

    q = torch.randn(qo_len, num_qo_heads, head_dim_qk, dtype=q_dtype).to(dev)
    k_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim_qk, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim_vo, dtype=q_dtype).to(dev)
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, torch.float8_e4m3fn, head_dim_qk)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, torch.float8_e4m3fn, head_dim_vo)

    o_orig = flashinfer.single_prefill_with_kv_cache(q, k_ref, v_ref, backend="fa2")
    o_ref = flashinfer.single_prefill_with_kv_cache(q, k_deq, v_deq, backend="fa2")
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q, k_fp8, v_fp8, backend="fa2", kv_cache_sf=(k_sf, v_sf)
    )
    assert o_fp8.shape == (qo_len, num_qo_heads, head_dim_vo), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "asym_head_dim", correctness_thresh=0.998)


# ---------------------------------------------------------------------------
# 6. cta128 (long-q) tiling coverage
#
# The tests above use qo_len<=32 (CTA_TILE_Q=64); these use qo_len=128 to
# cover the CTA_TILE_Q=128 tiling. Only e4m3: the tiling is dtype-independent.
# ---------------------------------------------------------------------------
def _sizes_cta128():
    # kv_len == qo_len so causal masking is well-defined.
    return dict(batch_size=3, qo_len=128, kv_len=128)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_token_head_scale_cta128(head_dim, q_dtype, gqa, causal):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes_cta128()
    qo_len, kv_len = sz["qo_len"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    k_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, torch.float8_e4m3fn, head_dim)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, torch.float8_e4m3fn, head_dim)

    o_orig = flashinfer.single_prefill_with_kv_cache(
        q, k_ref, v_ref, causal=causal, backend="fa2"
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_deq, v_deq, causal=causal, backend="fa2"
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q, k_fp8, v_fp8, causal=causal, backend="fa2", kv_cache_sf=(k_sf, v_sf)
    )

    assert o_fp8.shape == (qo_len, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "single_prefill_cta128")


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_ragged_token_head_scale_cta128(head_dim, q_dtype, gqa, causal):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes_cta128()
    batch_size, qo_len, kv_len = sz["batch_size"], sz["qo_len"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(dev) * qo_len
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(dev) * kv_len
    total_qo = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()

    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    k_ref = 0.05 * torch.randn(total_kv, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(total_kv, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, torch.float8_e4m3fn, head_dim)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, torch.float8_e4m3fn, head_dim)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    ref = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    ref.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_orig = ref.run(q, k_ref, v_ref)
    o_ref = ref.run(q, k_deq, v_deq)

    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    fi.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    o_fp8 = fi.run(q, k_fp8, v_fp8, kv_cache_sf=(k_sf, v_sf))

    assert o_fp8.shape == (total_qo, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_prefill_ragged_cta128")


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_paged_token_head_scale_cta128(head_dim, q_dtype, gqa, causal):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes_cta128()
    batch_size, qo_len, kv_len = sz["batch_size"], sz["qo_len"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)
    page_size = 16

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(dev) * qo_len
    total_qo = qo_indptr[-1].item()

    paged_kv_indptr = [0]
    paged_kv_indices = []
    paged_kv_last_page_len = []
    for _ in range(batch_size):
        num_pages = (kv_len + page_size - 1) // page_size
        paged_kv_indptr.append(paged_kv_indptr[-1] + num_pages)
        paged_kv_indices.extend(
            range(paged_kv_indptr[-1] - num_pages, paged_kv_indptr[-1])
        )
        paged_kv_last_page_len.append(kv_len % page_size or page_size)
    num_pages = paged_kv_indptr[-1]
    paged_kv_indptr = torch.tensor(paged_kv_indptr, dtype=torch.int32).to(dev)
    paged_kv_indices = torch.tensor(paged_kv_indices, dtype=torch.int32).to(dev)
    paged_kv_last_page_len = torch.tensor(paged_kv_last_page_len, dtype=torch.int32).to(
        dev
    )

    k_ref = 0.05 * torch.randn(
        num_pages, num_kv_heads, page_size, head_dim, dtype=q_dtype
    ).to(dev)
    v_ref = 0.05 * torch.randn(
        num_pages, num_kv_heads, page_size, head_dim, dtype=q_dtype
    ).to(dev)
    k_fp8, k_sf, k_deq = quantize_token_head(k_ref, torch.float8_e4m3fn, head_dim)
    v_fp8, v_sf, v_deq = quantize_token_head(v_ref, torch.float8_e4m3fn, head_dim)
    paged_kv_cache = torch.cat([k_fp8.unsqueeze(1), v_fp8.unsqueeze(1)], dim=1)
    k_deq_cache = torch.cat([k_deq.unsqueeze(1), v_deq.unsqueeze(1)], dim=1)
    k_orig_cache = torch.cat([k_ref.unsqueeze(1), v_ref.unsqueeze(1)], dim=1)
    kv_sf = (k_sf, v_sf)

    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=q_dtype).to(dev)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    ref.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_orig = ref.run(q, k_orig_cache)
    o_ref = ref.run(q, k_deq_cache)

    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    o_fp8 = fi.run(q, paged_kv_cache, kv_cache_sf=kv_sf)

    assert o_fp8.shape == (total_qo, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_prefill_paged_cta128")


# ---------------------------------------------------------------------------
# 7. layout views:
#    - paged: the scale (and KV data) may be non-contiguous views sliced out
#      of an inline slot tensor (last dim head_dim + 16)
#    - single/ragged: the scale strides must mirror the KV strides / head_dim;
#      matching strided views are accepted (the kernel derives the scale
#      strides in-kernel), non-mirroring layouts are rejected (section 8)
# ---------------------------------------------------------------------------
def _paged_slot_views(
    batch_size, kv_len, page_size, num_kv_heads, head_dim, q_dtype, dev, kv_layout="HND"
):
    """Build paged indices plus inline-slot views of the KV data and scales.

    Returns ``(indptr, indices, last_page_len, k_view, v_view, k_sf_view,
    v_sf_view, k_deq_cache)``: the views are non-contiguous slices of the slot
    tensors (last dim ``head_dim + 16``) and ``k_deq_cache`` is the 5-D
    dequantized reference cache.
    """
    indptr = [0]
    indices = []
    last_page_len = []
    for _ in range(batch_size):
        num_pages = (kv_len + page_size - 1) // page_size
        indptr.append(indptr[-1] + num_pages)
        indices.extend(range(indptr[-1] - num_pages, indptr[-1]))
        last_page_len.append(kv_len % page_size or page_size)
    num_pages = indptr[-1]
    t = lambda x: torch.tensor(x, dtype=torch.int32).to(dev)
    indptr, indices, last_page_len = t(indptr), t(indices), t(last_page_len)

    inner = (
        (num_pages, num_kv_heads, page_size, head_dim)
        if kv_layout == "HND"
        else (num_pages, page_size, num_kv_heads, head_dim)
    )
    k_ref = 0.05 * torch.randn(*inner, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(*inner, dtype=q_dtype).to(dev)
    k_view, k_sf_view = extract_views(
        make_inline_slot(k_ref, torch.float8_e4m3fn, head_dim),
        torch.float8_e4m3fn,
        head_dim,
    )
    v_view, v_sf_view = extract_views(
        make_inline_slot(v_ref, torch.float8_e4m3fn, head_dim),
        torch.float8_e4m3fn,
        head_dim,
    )
    assert not k_view.is_contiguous() and not k_sf_view.is_contiguous()
    _, _, k_deq = quantize_token_head(k_ref, torch.float8_e4m3fn, head_dim)
    _, _, v_deq = quantize_token_head(v_ref, torch.float8_e4m3fn, head_dim)
    k_deq_cache = torch.cat([k_deq.unsqueeze(1), v_deq.unsqueeze(1)], dim=1)
    return (
        indptr,
        indices,
        last_page_len,
        k_view,
        v_view,
        k_sf_view,
        v_sf_view,
        k_deq_cache,
    )


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_batch_prefill_paged_token_head_scale_view(kv_layout):
    # 3-D (paged) views: the scale view's page/entry strides are slot-sized.
    torch.manual_seed(42)
    dev = "cuda:0"
    head_dim = 128
    batch_size, kv_len = 3, 32
    num_qo_heads = num_kv_heads = 4
    page_size = 16
    q_dtype = torch.float16

    (
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        k_view,
        v_view,
        k_sf_view,
        v_sf_view,
        k_deq_cache,
    ) = _paged_slot_views(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, q_dtype, dev, kv_layout
    )

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32).to(dev) * kv_len
    total_qo = qo_indptr[-1].item()
    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=q_dtype).to(dev)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend="fa2"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=q_dtype,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    o_fp8 = fi.run(q, (k_view, v_view), kv_cache_sf=(k_sf_view, v_sf_view))

    ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, backend="fa2"
    )
    ref.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=False,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_ref = ref.run(q, k_deq_cache)
    _check(o_fp8, o_ref, o_ref, "batch_prefill_paged_view")


def test_batch_decode_paged_token_head_scale_view():
    # The decode tensor-core path reuses the paged-prefill module, so
    # inline-slot views work here too (the decode wrapper's run() only checks
    # the scale shape/dtype, not contiguity).
    torch.manual_seed(42)
    dev = "cuda:0"
    head_dim = 128
    batch_size, kv_len = 2, 32
    num_qo_heads = num_kv_heads = 4
    page_size = 16
    q_dtype = torch.float16

    (
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        k_view,
        v_view,
        k_sf_view,
        v_sf_view,
        k_deq_cache,
    ) = _paged_slot_views(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, q_dtype, dev, "HND"
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=q_dtype).to(dev)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")
    fi.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    o_fp8 = fi.run(q, (k_view, v_view), kv_cache_sf=(k_sf_view, v_sf_view))

    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")
    ref.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_ref = ref.run(q, k_deq_cache)
    _check(o_fp8, o_ref, o_ref, "batch_decode_paged_view")


def test_single_prefill_accepts_matching_strided_views():
    # The mirror contract accepts non-contiguous layouts as long as the scale
    # strides mirror the KV strides divided by head_dim: a [::2] KV view and a
    # [::2] scale view mirror each other, and the kernel derives the scale
    # strides from the KV strides in-kernel (the NVFP4 contract).
    torch.manual_seed(42)
    dev = "cuda:0"
    head_dim = 128
    qo_len = kv_len = 32
    num_qo_heads = num_kv_heads = 4
    q_dtype = torch.float16

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    k_full = 0.05 * torch.randn(2 * kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(
        dev
    )
    v_full = 0.05 * torch.randn(2 * kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(
        dev
    )
    k_full_fp8, k_full_sf, k_full_deq = quantize_token_head(
        k_full, torch.float8_e4m3fn, head_dim
    )
    v_full_fp8, v_full_sf, v_full_deq = quantize_token_head(
        v_full, torch.float8_e4m3fn, head_dim
    )
    # Take every other token: KV and scale both become strided views that
    # mirror each other (scale token stride = KV token stride / head_dim).
    k, v = k_full_fp8[::2], v_full_fp8[::2]
    k_sf, v_sf = k_full_sf[::2], v_full_sf[::2]
    assert not k.is_contiguous() and not k_sf.is_contiguous()
    assert k_sf.stride(0) == k.stride(0) // head_dim  # the mirror contract

    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_full_deq[::2], v_full_deq[::2], backend="fa2"
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q, k, v, backend="fa2", kv_cache_sf=(k_sf, v_sf)
    )
    _check(o_fp8, o_ref, o_ref, "single_prefill_strided_views")


# ---------------------------------------------------------------------------
# 8. negative / validation tests
#
# Pin the plan()/run() error paths so a regression that silently drops a check
# is caught. One small fixed config (head_dim 128, e4m3, fp16, MHA) since these
# exercise validation, not kernel numerics.
# ---------------------------------------------------------------------------
_NEG = dict(head_dim=128, qo_len=16, kv_len=16, batch=2, heads=4, page=16)


def _neg_q(dev, total_rows):
    return torch.randn(
        total_rows, _NEG["heads"], _NEG["head_dim"], dtype=torch.float16
    ).to(dev)


def _neg_sf(total_tokens, dev):
    """A valid (k_sf, v_sf) pair for a ragged KV of ``total_tokens`` rows."""
    shape = (total_tokens, _NEG["heads"])
    k_sf = torch.ones(*shape, dtype=torch.float32, device=dev)
    v_sf = torch.ones(*shape, dtype=torch.float32, device=dev)
    return k_sf, v_sf


def _neg_paged_sf(num_pages, dev):
    """A valid (k_sf, v_sf) pair for an HND paged KV of ``num_pages`` pages."""
    shape = (num_pages, _NEG["heads"], _NEG["page"])
    k_sf = torch.ones(*shape, dtype=torch.float32, device=dev)
    v_sf = torch.ones(*shape, dtype=torch.float32, device=dev)
    return k_sf, v_sf


def _neg_paged_indices(dev):
    """Paged indices for the fixed _NEG config (HND)."""
    indptr = [0]
    indices = []
    last_page_len = []
    for _ in range(_NEG["batch"]):
        num_pages = (_NEG["kv_len"] + _NEG["page"] - 1) // _NEG["page"]
        indptr.append(indptr[-1] + num_pages)
        indices.extend(range(indptr[-1] - num_pages, indptr[-1]))
        last_page_len.append(_NEG["kv_len"] % _NEG["page"] or _NEG["page"])
    t = lambda x: torch.tensor(x, dtype=torch.int32).to(dev)
    return t(indptr), t(indices), t(last_page_len), indptr[-1]


# -- single prefill (ragged) ----------------------------------------------
def test_single_prefill_rejects_kv_dtype_mismatch():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e5m2)
    )
    k_sf, _ = _neg_sf(c["kv_len"], dev)
    with pytest.raises(AssertionError, match="same dtype"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, backend="fa2", kv_cache_sf=(k_sf, None)
        )


def test_single_prefill_rejects_missing_v_sf():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    k_sf, _ = _neg_sf(c["kv_len"], dev)
    with pytest.raises(AssertionError, match="v_sf"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, backend="fa2", kv_cache_sf=(k_sf, None)
        )


def test_single_prefill_rejects_non_float32_sf():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    k_sf, _ = _neg_sf(c["kv_len"], dev)
    v_sf, _ = _neg_sf(c["kv_len"], dev)
    with pytest.raises(AssertionError, match="float32"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, backend="fa2", kv_cache_sf=(k_sf, v_sf.to(torch.float16))
        )


def test_single_prefill_rejects_bad_sf_shape():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    # Transposed (heads, kv_len) instead of (kv_len, heads) for NHD.
    k_sf = torch.ones(c["heads"], c["kv_len"], dtype=torch.float32, device=dev)
    v_sf, _ = _neg_sf(c["kv_len"], dev)
    with pytest.raises(AssertionError, match="shape"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, backend="fa2", kv_cache_sf=(k_sf, v_sf)
        )


def test_single_prefill_rejects_non_mirror_sf():
    # The scale layout must mirror the KV layout (the kernel derives the scale
    # strides from the KV strides, as the NVFP4 path does). An inline-slot view
    # (token stride = slot size, not head_dim) does not mirror a standard-layout
    # KV tensor, so it is rejected here; the paged path is the one that supports
    # such views (section 7).
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k_ref = torch.randn(c["kv_len"], c["heads"], c["head_dim"]).to(dev)
    v_ref = torch.randn(c["kv_len"], c["heads"], c["head_dim"]).to(dev)
    k_slot = make_inline_slot(k_ref, torch.float8_e4m3fn, c["head_dim"])
    v_slot = make_inline_slot(v_ref, torch.float8_e4m3fn, c["head_dim"])
    _, k_sf_view = extract_views(k_slot, torch.float8_e4m3fn, c["head_dim"])
    _, v_sf_view = extract_views(v_slot, torch.float8_e4m3fn, c["head_dim"])
    k = k_ref.to(torch.float8_e4m3fn)
    v = v_ref.to(torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="mirror"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, backend="fa2", kv_cache_sf=(k_sf_view, v_sf_view)
        )


def test_single_prefill_rejects_non_fa2_backend():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    k_sf, v_sf = _neg_sf(c["kv_len"], dev)
    with pytest.raises(ValueError, match="backend='fa2'"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, backend="fa3", kv_cache_sf=(k_sf, v_sf)
        )


# -- batch prefill (ragged) ------------------------------------------------
def test_ragged_prefill_plan_rejects_non_fp8_kv():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    with pytest.raises(AssertionError, match="fp8"):
        fi.plan(
            qo_indptr,
            kv_indptr,
            c["heads"],
            c["heads"],
            c["head_dim"],
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            use_token_head_sf=True,
        )


def test_ragged_prefill_plan_rejects_non_fa2_backend():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa3")
    with pytest.raises(ValueError, match="backend='fa2'"):
        fi.plan(
            qo_indptr,
            kv_indptr,
            c["heads"],
            c["heads"],
            c["head_dim"],
            q_data_type=torch.float16,
            kv_data_type=torch.float8_e4m3fn,
            use_token_head_sf=True,
        )


def test_ragged_prefill_plan_auto_backend_pinned_to_fa2():
    # use_token_head_sf requires the fa2 backend; with backend="auto" the plan
    # must still end up on fa2.
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="auto")
    fi.plan(
        qo_indptr,
        kv_indptr,
        c["heads"],
        c["heads"],
        c["head_dim"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    assert fi._backend == "fa2"


def test_ragged_prefill_run_rejects_v_dtype_mismatch():
    # k is the planned e4m3; v is e5m2 -> run() must reject the dtype mismatch.
    c = _NEG
    dev = "cuda:0"
    total_kv = c["batch"] * c["kv_len"]
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    fi.plan(
        qo_indptr,
        kv_indptr,
        c["heads"],
        c["heads"],
        c["head_dim"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    k = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev).to(torch.float8_e4m3fn)
    v = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev).to(torch.float8_e5m2)
    k_sf, v_sf = _neg_sf(total_kv, dev)
    with pytest.raises(ValueError, match="dtype of v"):
        fi.run(q, k, v, kv_cache_sf=(k_sf, v_sf))


def test_ragged_prefill_run_rejects_legacy_inline_slot():
    # k last dim is head_dim + 16 (the legacy inline slot) -> run() must
    # reject it: the standard layout is head_dim.
    c = _NEG
    dev = "cuda:0"
    total_kv = c["batch"] * c["kv_len"]
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    fi.plan(
        qo_indptr,
        kv_indptr,
        c["heads"],
        c["heads"],
        c["head_dim"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    k = (
        torch.randn(total_kv, c["heads"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev).to(torch.float8_e4m3fn)
    k_sf, v_sf = _neg_sf(total_kv, dev)
    with pytest.raises(ValueError, match="K last dim"):
        fi.run(q, k, v, kv_cache_sf=(k_sf, v_sf))


def test_ragged_prefill_run_rejects_missing_sf():
    c = _NEG
    dev = "cuda:0"
    total_kv = c["batch"] * c["kv_len"]
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    fi.plan(
        qo_indptr,
        kv_indptr,
        c["heads"],
        c["heads"],
        c["head_dim"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    k = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev).to(torch.float8_e4m3fn)
    v = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="both k_sf and v_sf"):
        fi.run(q, k, v)


def test_ragged_prefill_run_rejects_bad_sf_shape():
    c = _NEG
    dev = "cuda:0"
    total_kv = c["batch"] * c["kv_len"]
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    fi.plan(
        qo_indptr,
        kv_indptr,
        c["heads"],
        c["heads"],
        c["head_dim"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    k = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev).to(torch.float8_e4m3fn)
    v = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev).to(torch.float8_e4m3fn)
    k_sf, v_sf = _neg_sf(c["heads"], dev)  # wrong: (heads, ...) not (total_kv, heads)
    with pytest.raises(ValueError, match="must have shape"):
        fi.run(q, k, v, kv_cache_sf=(k_sf, v_sf))


def test_ragged_prefill_run_rejects_non_mirror_sf():
    # Same mirror contract as the single-prefill path: the ragged scale must
    # mirror the KV layout, so an inline-slot view is rejected (paged only).
    c = _NEG
    dev = "cuda:0"
    total_kv = c["batch"] * c["kv_len"]
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    fi.plan(
        qo_indptr,
        kv_indptr,
        c["heads"],
        c["heads"],
        c["head_dim"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    k_ref = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev)
    v_ref = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev)
    k_slot = make_inline_slot(k_ref, torch.float8_e4m3fn, c["head_dim"])
    v_slot = make_inline_slot(v_ref, torch.float8_e4m3fn, c["head_dim"])
    _, k_sf_view = extract_views(k_slot, torch.float8_e4m3fn, c["head_dim"])
    _, v_sf_view = extract_views(v_slot, torch.float8_e4m3fn, c["head_dim"])
    k = k_ref.to(torch.float8_e4m3fn)
    v = v_ref.to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="mirror"):
        fi.run(q, k, v, kv_cache_sf=(k_sf_view, v_sf_view))


# -- batch prefill (paged) -------------------------------------------------
def test_paged_prefill_plan_rejects_non_fp8_kv():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, _num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    with pytest.raises(AssertionError, match="fp8"):
        fi.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            c["heads"],
            c["heads"],
            c["head_dim"],
            c["page"],
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            use_token_head_sf=True,
        )


def test_paged_prefill_plan_rejects_non_fa2_backend():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, _num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa3"
    )
    with pytest.raises(ValueError, match="backend='fa2'"):
        fi.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            c["heads"],
            c["heads"],
            c["head_dim"],
            c["page"],
            q_data_type=torch.float16,
            kv_data_type=torch.float8_e4m3fn,
            use_token_head_sf=True,
        )


def test_paged_prefill_plan_auto_backend_pinned_to_fa2():
    # use_token_head_sf requires the fa2 backend; with backend="auto" the plan
    # must still end up on fa2.
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, _num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="auto"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        c["heads"],
        c["heads"],
        c["head_dim"],
        c["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    assert fi._backend == "fa2"


def test_paged_prefill_run_rejects_v_dtype_mismatch():
    # e5m2 cache vs e4m3 plan -> run() must reject the dtype mismatch.
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        c["heads"],
        c["heads"],
        c["head_dim"],
        c["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    k = (
        torch.randn(num_pages, c["heads"], c["page"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(num_pages, c["heads"], c["page"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e5m2)
    )
    k_sf, v_sf = _neg_paged_sf(num_pages, dev)
    with pytest.raises(ValueError, match="does not match the kv_data_type"):
        fi.run(q, (k, v), kv_cache_sf=(k_sf, v_sf))


def test_paged_prefill_run_rejects_legacy_inline_slot():
    # Cache last dim is head_dim + 16 (the legacy inline slot) -> run() must
    # reject it: the standard layout is head_dim.
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        c["heads"],
        c["heads"],
        c["head_dim"],
        c["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    bad_cache = (
        torch.randn(num_pages, 2, c["heads"], c["page"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    k_sf, v_sf = _neg_paged_sf(num_pages, dev)
    with pytest.raises(ValueError, match="K cache last dim"):
        fi.run(q, bad_cache, kv_cache_sf=(k_sf, v_sf))


def test_paged_prefill_run_rejects_missing_sf():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        c["heads"],
        c["heads"],
        c["head_dim"],
        c["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    paged_kv_cache = (
        torch.randn(num_pages, 2, c["heads"], c["page"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    with pytest.raises(ValueError, match="tuple of float32"):
        fi.run(q, paged_kv_cache)


def test_paged_prefill_run_rejects_bad_sf_shape():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    fi.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        c["heads"],
        c["heads"],
        c["head_dim"],
        c["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    paged_kv_cache = (
        torch.randn(num_pages, 2, c["heads"], c["page"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    # 2-D (ragged) scale instead of 3-D paged.
    k_sf, v_sf = _neg_sf(num_pages * c["page"], dev)
    with pytest.raises(ValueError, match="must have shape"):
        fi.run(q, paged_kv_cache, kv_cache_sf=(k_sf, v_sf))


# -- batch decode (paged) --------------------------------------------------
def test_decode_batch_plan_rejects_non_fa2_backend():
    c = _NEG
    dev = "cuda:0"
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, _num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa3"
    )
    with pytest.raises(ValueError, match="backend='fa2'"):
        fi.plan(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            c["heads"],
            c["heads"],
            c["head_dim"],
            c["page"],
            q_data_type=torch.float16,
            kv_data_type=torch.float8_e4m3fn,
            use_token_head_sf=True,
        )


def test_decode_batch_plan_forces_tensor_cores():
    # The CUDA-core decode path has no token-head-scale kernel; plan() must
    # force the tensor-core (fa2 prefill) path.
    c = _NEG
    dev = "cuda:0"
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, _num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")
    fi.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        c["heads"],
        c["heads"],
        c["head_dim"],
        c["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    assert fi.use_tensor_cores


def test_decode_batch_run_rejects_legacy_inline_slot():
    c = _NEG
    dev = "cuda:0"
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")
    fi.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        c["heads"],
        c["heads"],
        c["head_dim"],
        c["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = torch.randn(c["batch"], c["heads"], c["head_dim"], dtype=torch.float16).to(dev)
    bad_cache = (
        torch.randn(num_pages, 2, c["heads"], c["page"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    k_sf, v_sf = _neg_paged_sf(num_pages, dev)
    with pytest.raises(ValueError, match="K cache last dim"):
        fi.run(q, bad_cache, kv_cache_sf=(k_sf, v_sf))


def test_decode_batch_run_rejects_missing_sf():
    c = _NEG
    dev = "cuda:0"
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")
    fi.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        c["heads"],
        c["heads"],
        c["head_dim"],
        c["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_token_head_sf=True,
    )
    q = torch.randn(c["batch"], c["heads"], c["head_dim"], dtype=torch.float16).to(dev)
    paged_kv_cache = (
        torch.randn(num_pages, 2, c["heads"], c["page"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    with pytest.raises(ValueError, match="tuple of float32"):
        fi.run(q, paged_kv_cache)


# -- custom JIT module: use_token_head_sf must match the built module ------
# The custom module is faked (no compilation) so plan() hits the mismatch check.


def test_paged_prefill_plan_rejects_jit_token_head_sf_mismatch():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, _num_pages = (
        _neg_paged_indices(dev)
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    fi._jit_module = object()
    fi._jit_use_token_head_sf = False
    with pytest.raises(ValueError, match="use_token_head_sf must match"):
        fi.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            c["heads"],
            c["heads"],
            c["head_dim"],
            c["page"],
            q_data_type=torch.float16,
            kv_data_type=torch.float8_e4m3fn,
            use_token_head_sf=True,
        )


def test_ragged_prefill_plan_rejects_jit_token_head_sf_mismatch():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    fi._jit_module = object()
    fi._jit_use_token_head_sf = False
    with pytest.raises(ValueError, match="use_token_head_sf must match"):
        fi.plan(
            qo_indptr,
            kv_indptr,
            c["heads"],
            c["heads"],
            c["head_dim"],
            q_data_type=torch.float16,
            kv_data_type=torch.float8_e4m3fn,
            use_token_head_sf=True,
        )


# ---------------------------------------------------------------------------
# 9. split-KV (flash-decoding) coverage for batch decode
#
# A long kv_len with a small batch drives the (tensor-core) decode path onto
# the split-KV (KV-partitioned) tiling.
# ---------------------------------------------------------------------------
def test_batch_decode_paged_token_head_scale_split_kv():
    torch.manual_seed(42)
    dev = "cuda:0"
    head_dim = 128
    batch_size = 1
    kv_len = 4096
    num_qo_heads = num_kv_heads = 4
    page_size = 16
    dtype = torch.float8_e4m3fn
    q_dtype = torch.float16

    (
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        paged_kv_cache,
        k_deq_cache,
        k_orig_cache,
        kv_sf,
    ) = _paged_kv(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, q_dtype, dtype, dev
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=q_dtype).to(dev)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")
    ref.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=q_dtype,
    )
    o_orig = ref.run(q, k_orig_cache)
    o_ref = ref.run(q, k_deq_cache)

    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, kv_layout="HND")
    fi.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=dtype,
        use_token_head_sf=True,
    )
    o_fp8 = fi.run(q, paged_kv_cache, kv_cache_sf=kv_sf)

    assert o_fp8.shape == (batch_size, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_decode_paged_split_kv")


if __name__ == "__main__":
    # Quick smoke: one config per entry (head_dim 128, e4m3, fp16, MHA) plus a
    # couple of cheap negative checks.
    test_single_prefill_token_head_scale(
        128, torch.float8_e4m3fn, torch.float16, False, False
    )
    test_batch_prefill_ragged_token_head_scale(
        128, torch.float8_e4m3fn, torch.float16, False, False
    )
    test_batch_prefill_paged_token_head_scale(
        128, torch.float8_e4m3fn, torch.float16, False, False, "HND"
    )
    test_batch_decode_paged_token_head_scale(
        128, torch.float8_e4m3fn, torch.float16, False, "HND"
    )
    test_batch_prefill_paged_token_head_scale_view("HND")
    test_single_prefill_accepts_matching_strided_views()
    test_single_prefill_rejects_missing_v_sf()
    test_single_prefill_rejects_non_mirror_sf()
    test_ragged_prefill_run_rejects_v_dtype_mismatch()
    print("ALL PASS")
