"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Cache layouts of ``rope_quantize_fp8_append_paged_kv_cache`` that the shape
matrices of test_rope.py do not reach:

* a non-RoPE part that is not a multiple of the RoPE part (partial rotary,
  e.g. rope 32 / no-rope 48), for GQA in both layouts and for MLA;
* a V cache whose strides differ from the K cache's (padded V rows).

The caches and the Q non-RoPE output start as poison. Every case checks that
no byte outside the slots of the appended tokens changed and that the slots
hold the fp32 reference.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.test_helpers.rope_reference import RotaryEmbedding

import flashinfer
from flashinfer.utils import get_compute_capability

_FP8 = torch.float8_e4m3fn
_PAGE_SIZE = 16
_NUM_PAGES = 9
_MAX_POS = 64
# 0x7F is NaN in float8_e4m3fn, so a target byte the kernel misses fails the
# value check as well.
_POISON = 0x7F
_SCALE_Q = 0.5
_SCALE_KV = 2.0


def _skip_unless_fp8():
    if get_compute_capability(torch.device("cuda")) < (8, 9):
        pytest.skip("FP8 requires SM >= 89 (Ada/Hopper)")


def _append_batch():
    """Append 1, 5, 1 and 3 tokens to requests that hold 4, 13, 31 and 0
    tokens, on a shuffled page table that leaves some pages unused: appends
    start mid-page, cross a page boundary and fill the last slot of a page.
    Holds the API's metadata and the (page, slot) each token lands in."""
    kv_lens, append_lens = [4, 13, 31, 0], [1, 5, 1, 3]
    batch_indices, positions, kv_indptr = [], [], [0]
    for b, (before, n) in enumerate(zip(kv_lens, append_lens, strict=True)):
        batch_indices += [b] * n
        positions += range(before, before + n)
        kv_indptr.append(kv_indptr[-1] + -(-(before + n) // _PAGE_SIZE))
    gen = torch.Generator().manual_seed(0)
    kv_indices = torch.randperm(_NUM_PAGES, generator=gen)[: kv_indptr[-1]].tolist()
    pages = [
        kv_indices[kv_indptr[b] + p // _PAGE_SIZE]
        for b, p in zip(batch_indices, positions, strict=True)
    ]
    entries = [p % _PAGE_SIZE for p in positions]

    def cuda(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device="cuda")

    return SimpleNamespace(
        nnz=len(positions),
        batch_indices=cuda(batch_indices),
        positions=cuda(positions),
        kv_indptr=cuda(kv_indptr),
        kv_indices=cuda(kv_indices),
        pages=cuda(pages, torch.int64),
        entries=cuda(entries, torch.int64),
    )


def _randn(gen, *shape):
    return torch.randn(shape, generator=gen, device="cuda").to(torch.bfloat16)


def _poisoned(*shape):
    return torch.full(shape, _POISON, dtype=torch.uint8, device="cuda")


def _rope(head_dim, rope_dim):
    """fp32 reference for ``is_neox=False`` (interleaved): rotates the leading
    ``rope_dim`` features and passes the rest through."""
    return RotaryEmbedding(head_dim, rope_dim, _MAX_POS, 10000, False, torch.float32)


def _slots(cache, hnd, batch):
    """The appended tokens' slots of a uint8 cache storage, one row per token."""
    if hnd:
        return cache[batch.pages, :, batch.entries]
    return cache[batch.pages, batch.entries]


def _slot_mask(store, hnd, batch):
    mask = torch.zeros(store.shape, dtype=torch.bool, device="cuda")
    if hnd:
        mask[batch.pages, :, batch.entries] = True
    else:
        mask[batch.pages, batch.entries] = True
    return mask


def _assert_untouched(store, mask, what):
    stray = (store != _POISON) & ~mask
    assert not stray.any(), (
        f"{what}: {int(stray.sum())} bytes outside the target slots changed"
    )


def _assert_fp8_close(actual, expected, what):
    # FP8 device rounding of an FMA-ordered fp32 value may differ from
    # torch's .to(fp8) by one step; same tolerances as test_rope.py.
    torch.testing.assert_close(
        actual.float(),
        expected.to(_FP8).float(),
        rtol=0.25,
        atol=0.5,
        msg=lambda m: f"{what}: {m}",
    )


def _append(q, k, v, rope_dim, rope, batch, caches, kv_layout="NHD"):
    """Calls the API on rope/no-rope views of full-head q and k, as a fused
    projection gives them. q_nope_out is a view into a poisoned buffer one
    rope_dim wider per head; returns that buffer with the Q outputs."""
    num_qo_heads, head_dim = q.shape[1:]
    no_rope_dim = head_dim - rope_dim
    q_nope_store = _poisoned(batch.nnz, num_qo_heads, no_rope_dim + rope_dim)
    q_rope_out, q_nope_out = flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
        q[..., :rope_dim],
        k[..., :rope_dim],
        q[..., rope_dim:],
        k[..., rope_dim:],
        v,
        rope.cos_sin_cache,
        batch.positions,
        caches,
        batch.kv_indices,
        batch.kv_indptr,
        batch.batch_indices,
        batch.positions,
        is_neox=False,
        quantize_dtype=_FP8,
        quant_scale_q=_SCALE_Q,
        quant_scale_kv=_SCALE_KV,
        page_size=_PAGE_SIZE,
        kv_layout=kv_layout,
        q_nope_out=q_nope_store.view(_FP8)[..., :no_rope_dim],
    )
    torch.cuda.synchronize()
    return q_rope_out, q_nope_out, q_nope_store


def _assert_q(q_ref, rope_dim, q_rope_out, q_nope_out, q_nope_store):
    no_rope_dim = q_ref.shape[-1] - rope_dim
    padding = q_nope_store[..., no_rope_dim:]
    assert (padding == _POISON).all(), (
        f"q_nope_out: {int((padding != _POISON).sum())} bytes past the head changed"
    )
    _assert_fp8_close(q_rope_out, q_ref[..., :rope_dim] * _SCALE_Q, "q_rope_out")
    _assert_fp8_close(q_nope_out, q_ref[..., rope_dim:] * _SCALE_Q, "q_nope_out")


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "rope_dim,no_rope_dim",
    [(64, 64), (32, 48), (64, 96), (64, 32)],
    ids=["64-64", "32-48", "64-96", "64-32"],
)
def test_gqa_append_no_rope_tail(rope_dim, no_rope_dim, kv_layout):
    """The non-RoPE part is processed in rope_dim-wide chunks; when no_rope_dim
    is not a multiple of rope_dim the last chunk is partial and must stay
    inside the head (64-64 is the control)."""
    _skip_unless_fp8()
    hnd = kv_layout == "HND"
    num_qo_heads, num_kv_heads = 8, 2
    head_dim = rope_dim + no_rope_dim
    batch = _append_batch()
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = _randn(gen, batch.nnz, num_qo_heads, head_dim)
    k = _randn(gen, batch.nnz, num_kv_heads, head_dim)
    v = _randn(gen, batch.nnz, num_kv_heads, head_dim)
    if hnd:
        page = (num_kv_heads, _PAGE_SIZE, head_dim)
    else:
        page = (_PAGE_SIZE, num_kv_heads, head_dim)
    k_store = _poisoned(_NUM_PAGES, *page)
    v_store = _poisoned(_NUM_PAGES, *page)
    rope = _rope(head_dim, rope_dim)

    caches = (k_store.view(_FP8), v_store.view(_FP8))
    q_out = _append(q, k, v, rope_dim, rope, batch, caches, kv_layout)

    q_ref, k_ref = rope.forward_native(batch.positions.long(), q, k)
    _assert_q(q_ref, rope_dim, *q_out)
    for name, store, ref in (
        ("k_cache", k_store, k_ref),
        ("v_cache", v_store, v.float()),
    ):
        _assert_untouched(store, _slot_mask(store, hnd, batch), name)
        _assert_fp8_close(_slots(store, hnd, batch).view(_FP8), ref * _SCALE_KV, name)


@pytest.mark.parametrize(
    "rope_dim,no_rope_dim", [(64, 512), (32, 48)], ids=["64-512", "32-48"]
)
def test_mla_append_no_rope_tail(rope_dim, no_rope_dim):
    """MLA shares the chunked non-RoPE path: ckv rows must not spill into the
    next slot when no_rope_dim is not a multiple of rope_dim."""
    _skip_unless_fp8()
    num_qo_heads = 16
    head_dim = rope_dim + no_rope_dim
    batch = _append_batch()
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = _randn(gen, batch.nnz, num_qo_heads, head_dim)
    k = _randn(gen, batch.nnz, head_dim)
    ckv_store = _poisoned(_NUM_PAGES, _PAGE_SIZE, no_rope_dim)
    kpe_store = _poisoned(_NUM_PAGES, _PAGE_SIZE, rope_dim)
    rope = _rope(head_dim, rope_dim)

    caches = (ckv_store.view(_FP8), kpe_store.view(_FP8))
    q_out = _append(q, k, None, rope_dim, rope, batch, caches)

    q_ref, k_ref = rope.forward_native(batch.positions.long(), q, k)
    _assert_q(q_ref, rope_dim, *q_out)
    for name, store, ref in (
        ("ckv_cache", ckv_store, k_ref[:, rope_dim:]),
        ("kpe_cache", kpe_store, k_ref[:, :rope_dim]),
    ):
        _assert_untouched(store, _slot_mask(store, False, batch), name)
        _assert_fp8_close(_slots(store, False, batch).view(_FP8), ref * _SCALE_KV, name)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("v_row_pad", [0, 32], ids=["same-strides", "padded-v-rows"])
def test_gqa_append_v_cache_strides(kv_layout, v_row_pad):
    """A v_cache whose strides differ from k_cache's (each V row padded by
    v_row_pad bytes) is written through its own strides."""
    _skip_unless_fp8()
    hnd = kv_layout == "HND"
    rope_dim, no_rope_dim = 64, 64
    num_qo_heads, num_kv_heads = 8, 2
    head_dim = rope_dim + no_rope_dim
    batch = _append_batch()
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = _randn(gen, batch.nnz, num_qo_heads, head_dim)
    k = _randn(gen, batch.nnz, num_kv_heads, head_dim)
    v = _randn(gen, batch.nnz, num_kv_heads, head_dim)
    if hnd:
        page = (num_kv_heads, _PAGE_SIZE)
    else:
        page = (_PAGE_SIZE, num_kv_heads)
    k_store = _poisoned(_NUM_PAGES, *page, head_dim)
    v_store = _poisoned(_NUM_PAGES, *page, head_dim + v_row_pad)
    v_cache = v_store.view(_FP8)[..., :head_dim]
    rope = _rope(head_dim, rope_dim)

    _append(q, k, v, rope_dim, rope, batch, (k_store.view(_FP8), v_cache), kv_layout)

    _, k_ref = rope.forward_native(batch.positions.long(), q, k)
    _assert_untouched(k_store, _slot_mask(k_store, hnd, batch), "k_cache")
    _assert_fp8_close(
        _slots(k_store, hnd, batch).view(_FP8), k_ref * _SCALE_KV, "k_cache"
    )
    v_mask = _slot_mask(v_store, hnd, batch)
    v_mask[..., head_dim:] = False  # row padding is never a target
    _assert_untouched(v_store, v_mask, "v_cache")
    _assert_fp8_close(
        _slots(v_store[..., :head_dim], hnd, batch).view(_FP8),
        v.float() * _SCALE_KV,
        "v_cache",
    )
