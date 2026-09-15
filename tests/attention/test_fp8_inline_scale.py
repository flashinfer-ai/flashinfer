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

"""End-to-end tests for FP8 K/V cache with per-(token, head) inline scale.

Each K/V head vector is stored as a slot: ``[head_dim FP8 bytes][float32
scale][pad to 16B]`` so the slot's last dim is ``head_dim + 16``.

Each config computes three outputs from the same ``q``:

- ``o_orig`` -- 16-bit kernel on the original random K/V (pre-quantization)
- ``o_ref``  -- 16-bit kernel on the dequantized K/V
- ``o_fp8``  -- inline-scale FP8 kernel (the feature under test)

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
        pytest.skip("bf16 Q is not supported with inline FP8 scale on SM75")


def pack_inline_slot(x_ref: torch.Tensor, dtype: torch.dtype, head_dim: int):
    """Quantize ``x_ref`` per-(token, head) to FP8 and pack into a slot tensor.

    Returns ``(slot_fp8_view, dequant, scale)`` where ``slot_fp8_view`` has last
    dim ``head_dim + 16`` (viewed as the FP8 dtype so ``DTypeKV`` is FP8) and
    ``dequant`` is the 16-bit reference input **in ``x_ref``'s dtype** (so the
    16-bit reference kernel sees ``DTypeKV == DTypeQ`` when ``x_ref`` is built
    in ``q_dtype``).
    """
    fp8_max = 448.0 if dtype == torch.float8_e4m3fn else 57344.0
    scale = x_ref.abs().amax(dim=-1, keepdim=True) / fp8_max
    scale = scale.clamp(min=1e-12)
    x_fp8 = (x_ref / scale).to(dtype)
    slot_size = head_dim + 16
    slot = torch.zeros(
        *x_ref.shape[:-1], slot_size, dtype=torch.uint8, device=x_ref.device
    )
    slot[..., :head_dim] = x_fp8.view(torch.uint8)
    # The scale is read by the kernel as float32.
    slot[..., head_dim : head_dim + 4] = scale.to(torch.float32).view(torch.uint8)
    # Dequant in x_ref's dtype so the 16-bit reference kernel sees DTypeKV == DTypeQ.
    dequant = x_fp8.to(x_ref.dtype) * scale
    return slot.view(dtype), dequant, scale


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
    - correctness: ``o_fp8`` (inline-scale kernel) vs ``o_ref`` (16-bit on dequant)
    - quantization: ``o_orig`` (16-bit on original) vs ``o_fp8`` > 0.995

    ``correctness_thresh`` is relaxed for asymmetric head_dim (192), where the
    FP8 path uses a different MMA tiling than the 16-bit path.
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
    """Build paged indices and the three KV caches (orig/dequant/slot).

    ``kv_layout`` selects the cache shape (the K/V head vector is always the
    last dim, so ``pack_inline_slot`` is layout-agnostic):
      - HND: ``[num_pages, 2, num_kv_heads, page_size, *]``
      - NHD: ``[num_pages, 2, page_size, num_kv_heads, *]``

    Returns ``(indptr, indices, last_page_len, kv_slot, kv_deq, kv_orig)``.
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
    k_slot, k_deq, _ = pack_inline_slot(k_ref, dtype, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, dtype, head_dim)
    kv_slot = torch.cat([k_slot.unsqueeze(1), v_slot.unsqueeze(1)], dim=1)
    kv_deq = torch.cat([k_deq.unsqueeze(1), v_deq.unsqueeze(1)], dim=1)
    kv_orig = torch.cat([k_ref.unsqueeze(1), v_ref.unsqueeze(1)], dim=1)
    return indptr, indices, last_page_len, kv_slot, kv_deq, kv_orig


# ---------------------------------------------------------------------------
# 1. single prefill (ragged)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_inline_scale(head_dim, dtype, q_dtype, gqa, causal):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes(head_dim)
    qo_len, kv_len = sz["qo_len"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    # K/V reference in q_dtype: the 16-bit FA2 kernel requires DTypeQ == DTypeKV.
    k_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    k_slot, k_deq, _ = pack_inline_slot(k_ref, dtype, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, dtype, head_dim)

    o_orig = flashinfer.single_prefill_with_kv_cache(
        q, k_ref, v_ref, causal=causal, backend="fa2"
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_deq, v_deq, causal=causal, backend="fa2"
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q, k_slot, v_slot, causal=causal, backend="fa2", use_inline_sf=True
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
def test_batch_prefill_ragged_inline_scale(head_dim, dtype, q_dtype, gqa, causal):
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
    k_ref = 0.05 * torch.randn(total_kv, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(total_kv, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    k_slot, k_deq, _ = pack_inline_slot(k_ref, dtype, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, dtype, head_dim)

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
        kv_data_type=dtype,
        use_inline_sf=True,
    )
    o_fp8 = fi.run(q, k_slot, v_slot)

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
def test_batch_prefill_paged_inline_scale(
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
        use_inline_sf=True,
    )
    o_fp8 = fi.run(q, paged_kv_cache)

    assert o_fp8.shape == (total_qo, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_prefill_paged")


# ---------------------------------------------------------------------------
# 4. single decode (ragged)
#
# use_tensor_cores=True reuses the FA2 prefill kernel (q_len == 1), the same
# inline-scale path exercised by test_single_prefill_inline_scale.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
def test_single_decode_inline_scale(head_dim, dtype, q_dtype, gqa, use_tensor_cores):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    kv_len = _sizes(head_dim)["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)

    q = torch.randn(num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    k_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    k_slot, k_deq, _ = pack_inline_slot(k_ref, dtype, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, dtype, head_dim)

    o_orig = flashinfer.single_decode_with_kv_cache(
        q, k_ref, v_ref, use_tensor_cores=use_tensor_cores
    )
    o_ref = flashinfer.single_decode_with_kv_cache(
        q, k_deq, v_deq, use_tensor_cores=use_tensor_cores
    )
    o_fp8 = flashinfer.single_decode_with_kv_cache(
        q, k_slot, v_slot, use_inline_sf=True, use_tensor_cores=use_tensor_cores
    )

    assert o_fp8.shape == (num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, f"single_decode_tc{int(use_tensor_cores)}")


# ---------------------------------------------------------------------------
# 5. batch decode (paged)
#
# use_tensor_cores=True reuses the FA2 paged-prefill kernel, the same
# inline-scale path exercised by test_batch_prefill_paged_inline_scale.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
def test_batch_decode_paged_inline_scale(
    head_dim, dtype, q_dtype, gqa, use_tensor_cores, kv_layout
):
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
    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, use_tensor_cores=use_tensor_cores
    )
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

    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, kv_layout=kv_layout, use_tensor_cores=use_tensor_cores
    )
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
        use_inline_sf=True,
    )
    o_fp8 = fi.run(q, paged_kv_cache)

    assert o_fp8.shape == (batch_size, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, f"batch_decode_paged_tc{int(use_tensor_cores)}")


# ---------------------------------------------------------------------------
# 6. extended configs: sliding window, logits soft cap, asymmetric head dims
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_single_prefill_inline_scale_sliding_window(q_dtype):
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
    k_slot, k_deq, _ = pack_inline_slot(k_ref, torch.float8_e4m3fn, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, torch.float8_e4m3fn, head_dim)

    o_orig = flashinfer.single_prefill_with_kv_cache(
        q, k_ref, v_ref, window_left=window_left, backend="fa2"
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_deq, v_deq, window_left=window_left, backend="fa2"
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q,
        k_slot,
        v_slot,
        window_left=window_left,
        backend="fa2",
        use_inline_sf=True,
    )
    _check(o_fp8, o_ref, o_orig, "sliding_window")


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_single_prefill_inline_scale_logits_soft_cap(q_dtype):
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
    k_slot, k_deq, _ = pack_inline_slot(k_ref, torch.float8_e4m3fn, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, torch.float8_e4m3fn, head_dim)

    o_orig = flashinfer.single_prefill_with_kv_cache(
        q, k_ref, v_ref, logits_soft_cap=logits_soft_cap, backend="fa2"
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_deq, v_deq, logits_soft_cap=logits_soft_cap, backend="fa2"
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q,
        k_slot,
        v_slot,
        logits_soft_cap=logits_soft_cap,
        backend="fa2",
        use_inline_sf=True,
    )
    _check(o_fp8, o_ref, o_orig, "logits_soft_cap")


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_single_prefill_inline_scale_asym_head_dim(q_dtype):
    # head_dim_qk=192, head_dim_vo=128: K and V carry different slot sizes, and
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
    k_slot, k_deq, _ = pack_inline_slot(k_ref, torch.float8_e4m3fn, head_dim_qk)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, torch.float8_e4m3fn, head_dim_vo)

    o_orig = flashinfer.single_prefill_with_kv_cache(q, k_ref, v_ref, backend="fa2")
    o_ref = flashinfer.single_prefill_with_kv_cache(q, k_deq, v_deq, backend="fa2")
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q, k_slot, v_slot, backend="fa2", use_inline_sf=True
    )
    assert o_fp8.shape == (qo_len, num_qo_heads, head_dim_vo), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "asym_head_dim", correctness_thresh=0.998)


# ---------------------------------------------------------------------------
# 7. cta128 (long-q) tiling coverage
#
# FA2DetermineCtaTileQ (include/flashinfer/utils.cuh) picks CTA_TILE_Q=128 when
# ``packed_qo_len > 64 and head_dim < 256``; the tests above use qo_len<=32
# (CTA_TILE_Q=64). These cases use a long qo_len to cover the CTA_TILE_Q=128
# tiling. Only e4m3 is used: the tiling is independent of the FP8 dtype.
# ---------------------------------------------------------------------------
def _sizes_cta128():
    # qo_len=128 -> packed_qo_len 128 (mha) / 256 (gqa), both > 64 -> CTA_TILE_Q=128.
    # kv_len == qo_len so causal masking is well-defined.
    return dict(batch_size=3, qo_len=128, kv_len=128)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_single_prefill_inline_scale_cta128(head_dim, q_dtype, gqa, causal):
    _skip_bf16_on_sm75(q_dtype)
    torch.manual_seed(42)
    dev = "cuda:0"
    sz = _sizes_cta128()
    qo_len, kv_len = sz["qo_len"], sz["kv_len"]
    num_qo_heads, num_kv_heads = _heads(gqa)

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype).to(dev)
    k_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    v_ref = 0.05 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=q_dtype).to(dev)
    k_slot, k_deq, _ = pack_inline_slot(k_ref, torch.float8_e4m3fn, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, torch.float8_e4m3fn, head_dim)

    o_orig = flashinfer.single_prefill_with_kv_cache(
        q, k_ref, v_ref, causal=causal, backend="fa2"
    )
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k_deq, v_deq, causal=causal, backend="fa2"
    )
    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q, k_slot, v_slot, causal=causal, backend="fa2", use_inline_sf=True
    )

    assert o_fp8.shape == (qo_len, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "single_prefill_cta128")


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_ragged_inline_scale_cta128(head_dim, q_dtype, gqa, causal):
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
    k_slot, k_deq, _ = pack_inline_slot(k_ref, torch.float8_e4m3fn, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, torch.float8_e4m3fn, head_dim)

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
        use_inline_sf=True,
    )
    o_fp8 = fi.run(q, k_slot, v_slot)

    assert o_fp8.shape == (total_qo, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_prefill_ragged_cta128")


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_paged_inline_scale_cta128(head_dim, q_dtype, gqa, causal):
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
    k_slot, k_deq, _ = pack_inline_slot(k_ref, torch.float8_e4m3fn, head_dim)
    v_slot, v_deq, _ = pack_inline_slot(v_ref, torch.float8_e4m3fn, head_dim)
    paged_kv_cache = torch.cat([k_slot.unsqueeze(1), v_slot.unsqueeze(1)], dim=1)
    k_deq_cache = torch.cat([k_deq.unsqueeze(1), v_deq.unsqueeze(1)], dim=1)
    k_orig_cache = torch.cat([k_ref.unsqueeze(1), v_ref.unsqueeze(1)], dim=1)

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
        use_inline_sf=True,
    )
    o_fp8 = fi.run(q, paged_kv_cache)

    assert o_fp8.shape == (total_qo, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_prefill_paged_cta128")


# ---------------------------------------------------------------------------
# 8. negative / validation tests
#
# The inline-scale feature adds host-side validation at plan() and run() time;
# the fix commits (V-dtype check, custom-JIT-module use_inline_sf check) are the
# newest. These tests pin the error paths so a regression that silently drops a
# check is caught. They use one small fixed config (head_dim 128, e4m3, fp16,
# MHA) because they exercise validation, not kernel numerics.
# ---------------------------------------------------------------------------
_NEG = dict(head_dim=128, qo_len=16, kv_len=16, batch=2, heads=4, page=16)


def _neg_q(dev, total_rows):
    return torch.randn(
        total_rows, _NEG["heads"], _NEG["head_dim"], dtype=torch.float16
    ).to(dev)


# -- single prefill (ragged) ----------------------------------------------
def test_single_prefill_rejects_non_fp8_kv():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = torch.randn(
        c["kv_len"], c["heads"], c["head_dim"] + 16, dtype=torch.float16
    ).to(dev)
    v = torch.randn(
        c["kv_len"], c["heads"], c["head_dim"] + 16, dtype=torch.float16
    ).to(dev)
    with pytest.raises(AssertionError, match="fp8"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, use_inline_sf=True, backend="fa2"
        )


def test_single_prefill_rejects_kv_dtype_mismatch():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e5m2)
    )
    with pytest.raises(AssertionError, match="same dtype"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, use_inline_sf=True, backend="fa2"
        )


def test_single_prefill_rejects_bad_k_slot():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    with pytest.raises(AssertionError, match="K slot"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, use_inline_sf=True, backend="fa2"
        )


def test_single_prefill_rejects_non_fa2_backend():
    c = _NEG
    dev = "cuda:0"
    q = _neg_q(dev, c["qo_len"])
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    with pytest.raises(ValueError, match="backend='fa2'"):
        flashinfer.single_prefill_with_kv_cache(
            q, k, v, use_inline_sf=True, backend="fa3"
        )


# -- single decode (ragged) -----------------------------------------------
def test_single_decode_rejects_non_fp8_kv():
    c = _NEG
    dev = "cuda:0"
    q = torch.randn(c["heads"], c["head_dim"], dtype=torch.float16).to(dev)
    k = torch.randn(
        c["kv_len"], c["heads"], c["head_dim"] + 16, dtype=torch.float16
    ).to(dev)
    v = torch.randn(
        c["kv_len"], c["heads"], c["head_dim"] + 16, dtype=torch.float16
    ).to(dev)
    with pytest.raises(AssertionError, match="fp8"):
        flashinfer.single_decode_with_kv_cache(q, k, v, use_inline_sf=True)


def test_single_decode_rejects_bad_kv_slot():
    c = _NEG
    dev = "cuda:0"
    q = torch.randn(c["heads"], c["head_dim"], dtype=torch.float16).to(dev)
    # K slot is head_dim (not head_dim + 16) -> the KV last-dim assert fires.
    k = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    v = (
        torch.randn(c["kv_len"], c["heads"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    with pytest.raises(AssertionError, match="KV last dim"):
        flashinfer.single_decode_with_kv_cache(q, k, v, use_inline_sf=True)


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
            use_inline_sf=True,
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
            use_inline_sf=True,
        )


def test_ragged_prefill_run_rejects_v_dtype_mismatch():
    # k is the planned e4m3; v is e5m2 -> run() must reject the dtype mismatch.
    c = _NEG
    dev = "cuda:0"
    total_kv = c["batch"] * c["kv_len"]
    slot = c["head_dim"] + 16
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
        use_inline_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    k = torch.randn(total_kv, c["heads"], slot).to(dev).to(torch.float8_e4m3fn)
    v = torch.randn(total_kv, c["heads"], slot).to(dev).to(torch.float8_e5m2)
    with pytest.raises(ValueError, match="dtype of v"):
        fi.run(q, k, v)


def test_ragged_prefill_run_rejects_bad_k_slot():
    # k last dim is head_dim (not head_dim + 16) -> run() must reject it.
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
        use_inline_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    k = torch.randn(total_kv, c["heads"], c["head_dim"]).to(dev).to(torch.float8_e4m3fn)
    v = (
        torch.randn(total_kv, c["heads"], c["head_dim"] + 16)
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    with pytest.raises(ValueError, match="K last dim"):
        fi.run(q, k, v)


# -- batch prefill (paged) -------------------------------------------------
def test_paged_prefill_plan_rejects_non_fp8_kv():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, *_rest) = _paged_kv(
        c["batch"],
        c["kv_len"],
        c["page"],
        c["heads"],
        c["head_dim"],
        torch.float16,
        torch.float8_e4m3fn,
        dev,
        "HND",
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
            use_inline_sf=True,
        )


def test_paged_prefill_plan_rejects_non_fa2_backend():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, *_rest) = _paged_kv(
        c["batch"],
        c["kv_len"],
        c["page"],
        c["heads"],
        c["head_dim"],
        torch.float16,
        torch.float8_e4m3fn,
        dev,
        "HND",
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
            use_inline_sf=True,
        )


def test_paged_prefill_run_rejects_v_dtype_mismatch():
    # e5m2 cache vs e4m3 plan; paged K/V share a dtype, so the k-dtype check fires first.
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, kv_slot, *_rest) = (
        _paged_kv(
            c["batch"],
            c["kv_len"],
            c["page"],
            c["heads"],
            c["head_dim"],
            torch.float16,
            torch.float8_e4m3fn,
            dev,
            "HND",
        )
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
        use_inline_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    with pytest.raises(ValueError, match="does not match the kv_data_type"):
        fi.run(q, kv_slot.to(torch.float8_e5m2))


def test_paged_prefill_run_rejects_bad_k_slot():
    # Cache last dim is head_dim (not head_dim + 16) -> run() must reject it.
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, kv_slot, *_rest) = (
        _paged_kv(
            c["batch"],
            c["kv_len"],
            c["page"],
            c["heads"],
            c["head_dim"],
            torch.float16,
            torch.float8_e4m3fn,
            dev,
            "HND",
        )
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
        use_inline_sf=True,
    )
    q = _neg_q(dev, c["batch"] * c["qo_len"])
    bad_cache = (
        torch.randn(kv_slot.shape[0], 2, c["heads"], c["page"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    with pytest.raises(ValueError, match="K cache last dim"):
        fi.run(q, bad_cache)


# -- batch decode (paged) --------------------------------------------------
def test_decode_batch_plan_rejects_non_fa2_backend():
    c = _NEG
    dev = "cuda:0"
    (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, *_rest) = _paged_kv(
        c["batch"],
        c["kv_len"],
        c["page"],
        c["heads"],
        c["head_dim"],
        torch.float16,
        torch.float8_e4m3fn,
        dev,
        "HND",
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
            use_inline_sf=True,
        )


def test_decode_batch_run_rejects_bad_k_slot():
    c = _NEG
    dev = "cuda:0"
    (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, kv_slot, *_rest) = (
        _paged_kv(
            c["batch"],
            c["kv_len"],
            c["page"],
            c["heads"],
            c["head_dim"],
            torch.float16,
            torch.float8_e4m3fn,
            dev,
            "HND",
        )
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, kv_layout="HND", use_tensor_cores=False
    )
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
        use_inline_sf=True,
    )
    q = torch.randn(c["batch"], c["heads"], c["head_dim"], dtype=torch.float16).to(dev)
    bad_cache = (
        torch.randn(kv_slot.shape[0], 2, c["heads"], c["page"], c["head_dim"])
        .to(dev)
        .to(torch.float8_e4m3fn)
    )
    with pytest.raises(ValueError, match="K cache last dim"):
        fi.run(q, bad_cache)


# -- custom JIT module: use_inline_sf must match the built module ----------
# The check reads only _jit_module/_jit_use_inline_sf and fires before plan()
# uses the module, so we fake a non-inline-sf module without compiling.


def test_paged_prefill_plan_rejects_jit_inline_sf_mismatch():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, *_rest) = _paged_kv(
        c["batch"],
        c["kv_len"],
        c["page"],
        c["heads"],
        c["head_dim"],
        torch.float16,
        torch.float8_e4m3fn,
        dev,
        "HND",
    )
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="HND", backend="fa2"
    )
    fi._jit_module = object()
    fi._jit_use_inline_sf = False
    with pytest.raises(ValueError, match="use_inline_sf must match"):
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
            use_inline_sf=True,
        )


def test_ragged_prefill_plan_rejects_jit_inline_sf_mismatch():
    c = _NEG
    dev = "cuda:0"
    qo_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["qo_len"]
    kv_indptr = torch.arange(0, c["batch"] + 1, dtype=torch.int32).to(dev) * c["kv_len"]
    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    fi = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, backend="fa2")
    fi._jit_module = object()
    fi._jit_use_inline_sf = False
    with pytest.raises(ValueError, match="use_inline_sf must match"):
        fi.plan(
            qo_indptr,
            kv_indptr,
            c["heads"],
            c["heads"],
            c["head_dim"],
            q_data_type=torch.float16,
            kv_data_type=torch.float8_e4m3fn,
            use_inline_sf=True,
        )


# ---------------------------------------------------------------------------
# 9. split-KV (flash-decoding) coverage for batch decode
#
# A long kv_len with a small batch drives the decode kernel onto the split-KV
# path (the KV range is partitioned across CTAs and the partial results are
# reduced). This exercises the inline-scale decode kernel under split-KV.
# NOTE: verify offline that split-KV is actually taken for these sizes (e.g.
# via FLASHINFER_LOGLEVEL=3, or by confirming the split-kv workspace is used);
# the numerics check below is valid regardless of which path is taken.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_paged_inline_scale_split_kv(use_tensor_cores):
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
    ) = _paged_kv(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, q_dtype, dtype, dev
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=q_dtype).to(dev)

    ws = torch.empty(_WS, dtype=torch.uint8, device=dev)
    ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, kv_layout="HND", use_tensor_cores=use_tensor_cores
    )
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

    fi = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws, kv_layout="HND", use_tensor_cores=use_tensor_cores
    )
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
        use_inline_sf=True,
    )
    o_fp8 = fi.run(q, paged_kv_cache)

    assert o_fp8.shape == (batch_size, num_qo_heads, head_dim), o_fp8.shape
    _check(o_fp8, o_ref, o_orig, "batch_decode_paged_split_kv")


if __name__ == "__main__":
    # Quick smoke: one config per entry (head_dim 128, e4m3, fp16, MHA) plus a
    # couple of cheap negative checks.
    test_single_prefill_inline_scale(
        128, torch.float8_e4m3fn, torch.float16, False, False
    )
    test_batch_prefill_ragged_inline_scale(
        128, torch.float8_e4m3fn, torch.float16, False, False
    )
    test_batch_prefill_paged_inline_scale(
        128, torch.float8_e4m3fn, torch.float16, False, False, "HND"
    )
    test_single_decode_inline_scale(
        128, torch.float8_e4m3fn, torch.float16, False, False
    )
    test_batch_decode_paged_inline_scale(
        128, torch.float8_e4m3fn, torch.float16, False, False, "HND"
    )
    test_single_prefill_rejects_non_fp8_kv()
    test_ragged_prefill_run_rejects_v_dtype_mismatch()
    print("ALL PASS")
