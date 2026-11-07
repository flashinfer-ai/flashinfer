# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Output contract of the plain paged KV append APIs in ``flashinfer.page``:
``append_paged_kv_cache`` and ``append_paged_mla_kv_cache``. The append is a
pure copy, so every check is bitwise, on the raw bytes of the whole cache.
Every group is one ``-k`` selection:

* ``page_append_output_contract``: of a fully poisoned cache, exactly the
  (page, slot) of each appended token changes and holds that token's bytes,
  for both layouts, tuple and combined caches, fp16, bf16, e4m3 and e5m2,
  and every page size and head_dim. uint8 caches, which the binding
  dispatches as fp4, are not covered here.
* ``page_append_strided_inputs``: K and V as strided views (the K and V heads
  of a fused qkv projection, the halves of one ``[nnz, 2, H, D]`` tensor,
  head-major buffers, and a contiguous K with a head-major V), and the source
  buffers are not written.
* ``page_append_strided_cache``: cache views with padded heads, whose padding
  stays poisoned, and cache views stored in the other layout.
* ``page_append_index_dtype``: int64 index tensors for both APIs, and the
  documented index recipe gives the indices the tests build.
* ``page_append_large_batch``: more tokens than one pass of the grid, for
  both APIs.
* ``page_append_mla_output_contract``: the same contract for the MLA latent
  cache, with separate or fused ``[ckv | kpe]`` caches and inputs.
"""

import itertools
from types import SimpleNamespace

import pytest
import torch
from tests.test_helpers.parametrize import pairwise_product_cases, parametrize_product

import flashinfer

_SEED = 3605
# Byte poisons for the two caches, (K, V) or (ckv, kpe). Every case runs once
# per pair, so a stray write cannot match the poison by chance in both runs.
_POISONS = ((0xA5, 0x5A), (0x00, 0xFF))
_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
}
_INDEX_NAMES = (
    "batch_indices",
    "positions",
    "kv_indices",
    "kv_indptr",
    "kv_last_page_len",
)
# Requests as (tokens already cached, tokens appended). At page size 16 they are
# a decode token, a chunk that crosses a page boundary, a token in the last slot
# of a page, a fresh prefill, a request with cached tokens but none appended, and
# an empty request with no pages.
_REQUESTS = ((4, 1), (13, 5), (31, 1), (0, 3), (8, 0), (0, 0))
# 64 requests of 256 tokens: more tokens than one pass of the grid on any GPU.
_LARGE_REQUESTS = tuple((b % 37, 256) for b in range(64))
_CKV_DIM, _KPE_DIM = 512, 64


def _random_bytes(shape, gen):
    """Uniform random bytes. Most cases hold NaNs and subnormals, so a value
    conversion inside the kernel is likely to change some bytes."""
    return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)


def _metadata(page_size, requests=_REQUESTS, num_spare=4):
    """Page table and append indices, built on the host rather than with
    ``get_batch_indices_positions``, with the (page, slot) each appended token
    must land in. The page table is a shuffled subset of the pool, so some
    pages belong to no request, and the last ``page_size + 1`` pages are in no
    table at all, so a write past the highest pool page still lands inside the
    storage that is compared."""
    batch_indices, positions, kv_indptr, last_page_len = [], [], [0], []
    for b, (cached, appended) in enumerate(requests):
        total = cached + appended
        batch_indices += [b] * appended
        positions += range(cached, total)
        kv_indptr.append(kv_indptr[-1] + -(-total // page_size))
        last_page_len.append((total - 1) % page_size + 1 if total else 0)
    num_used = kv_indptr[-1]
    gen = torch.Generator().manual_seed(_SEED)
    kv_indices = torch.randperm(num_used + num_spare, generator=gen)[:num_used].tolist()
    pages = [
        kv_indices[kv_indptr[b] + p // page_size]
        for b, p in zip(batch_indices, positions, strict=True)
    ]

    def cuda(values, dtype=torch.int32):
        return torch.tensor(values, dtype=dtype, device="cuda")

    return SimpleNamespace(
        page_size=page_size,
        nnz=len(positions),
        num_pages=num_used + num_spare + page_size + 1,
        append_indptr=cuda([0, *itertools.accumulate(a for _, a in requests)]),
        batch_indices=cuda(batch_indices),
        positions=cuda(positions),
        kv_indices=cuda(kv_indices),
        kv_indptr=cuda(kv_indptr),
        kv_last_page_len=cuda(last_page_len),
        pages=cuda(pages, torch.int64),
        entries=cuda([p % page_size for p in positions], torch.int64),
    )


def _assert_storage(actual, expected, targets, what):
    """Bitwise comparison of a whole uint8 storage, reporting stray writes
    outside the target slots separately from wrong bytes inside them."""
    bad = actual != expected
    if bad.any():
        stray = int((bad & (targets == 0)).sum())
        wrong = int((bad & (targets != 0)).sum())
        raise AssertionError(
            f"{what}: {stray} bytes outside the target slots changed and "
            f"{wrong} target bytes are wrong"
        )


def _check_append(storages, regions, write, data, append, label):
    """Poison the cache, append, and compare every byte of every storage with
    the poisoned cache plus the expected slot writes.

    ``regions(s)`` gives the two poisoned regions of storages shaped like
    ``storages``, ``write(region, rows)`` stores one row per token into the
    slots the tokens target, ``data`` holds the bytes each region must receive,
    and ``append()`` calls the API under test."""
    targets = [torch.zeros_like(s) for s in storages]
    for region, rows in zip(regions(targets), data, strict=True):
        write(region, torch.ones_like(rows))
    for poisons in _POISONS:
        for region, poison in zip(regions(storages), poisons, strict=True):
            region.fill_(poison)
        expected = [s.clone() for s in storages]
        for region, rows in zip(regions(expected), data, strict=True):
            write(region, rows)
        append()
        torch.cuda.synchronize()
        for i, (s, e, t) in enumerate(zip(storages, expected, targets, strict=True)):
            _assert_storage(s, e, t, f"{label}, poisons {poisons}, storage {i}")


def _assert_unchanged(tensors, before, what):
    for t, b in zip(tensors, before, strict=True):
        assert torch.equal(t, b), f"{what} was written"


# ---------------------------------------------------------------------------
# append_paged_kv_cache
# ---------------------------------------------------------------------------


def _contiguous_inputs(k, v):
    return k, v, [k, v]


def _gqa_case(
    meta,
    kv_layout,
    cache_form,
    dtype,
    num_kv_heads,
    head_dim,
    inputs=_contiguous_inputs,
    index_dtype=torch.int32,
    cache_view="dense",
):
    """Run the contract for one configuration.

    ``inputs(k_bytes, v_bytes)`` returns the byte views passed as K and V and
    the buffers they live in, which must not be written. ``cache_view`` is how a
    tuple cache is laid out in memory: ``dense``, ``padded`` (one extra head on
    each side of the heads the API sees), or ``transposed`` (stored in the other
    layout and passed as a transposed view)."""
    assert cache_form == "tuple" or cache_view == "dense"
    head_bytes = head_dim * dtype.itemsize
    gen = torch.Generator(device="cuda").manual_seed(_SEED)
    k_bytes = _random_bytes((meta.nnz, num_kv_heads, head_bytes), gen)
    v_bytes = _random_bytes((meta.nnz, num_kv_heads, head_bytes), gen)
    k_in, v_in, sources = inputs(k_bytes, v_bytes)
    sources_before = [s.clone() for s in sources]
    indices = [getattr(meta, name).to(index_dtype) for name in _INDEX_NAMES]
    indices_before = [t.clone() for t in indices]

    heads = num_kv_heads + 2 if cache_view == "padded" else num_kv_heads
    stored_layout = kv_layout
    if cache_view == "transposed":
        stored_layout = "HND" if kv_layout == "NHD" else "NHD"
    if stored_layout == "NHD":
        page = (meta.page_size, heads, head_bytes)
    else:
        page = (heads, meta.page_size, head_bytes)
    if cache_form == "combined":  # (num_pages, 2, ...), K and V of a page together
        storages = [
            torch.empty((meta.num_pages, 2, *page), dtype=torch.uint8, device="cuda")
        ]

        def regions(s):
            return s[0][:, 0], s[0][:, 1]

    else:
        storages = [
            torch.empty((meta.num_pages, *page), dtype=torch.uint8, device="cuda")
            for _ in range(2)
        ]

        def regions(s):
            return s[0], s[1]

    def api_view(region):  # the view of a K or V region the API is given
        if cache_view == "padded":
            return region.narrow(2 if kv_layout == "NHD" else 1, 1, num_kv_heads)
        if cache_view == "transposed":
            return region.transpose(1, 2)
        return region

    def write(region, rows):  # rows: [nnz, num_kv_heads, head_bytes]
        view = api_view(region)
        if kv_layout == "NHD":
            view[meta.pages, meta.entries] = rows
        else:
            view[meta.pages, :, meta.entries] = rows

    if cache_form == "combined":
        paged_kv_cache = storages[0].view(dtype)
    else:
        paged_kv_cache = tuple(api_view(r).view(dtype) for r in regions(storages))

    def append():
        flashinfer.append_paged_kv_cache(
            k_in.view(dtype),
            v_in.view(dtype),
            indices[0],
            indices[1],
            paged_kv_cache,
            indices[2],
            indices[3],
            indices[4],
            kv_layout=kv_layout,
        )

    label = (
        f"{kv_layout} {cache_form} {dtype} page_size={meta.page_size} "
        f"head_dim={head_dim} heads={num_kv_heads} indices={index_dtype} "
        f"cache_view={cache_view}"
    )
    _check_append(storages, regions, write, (k_bytes, v_bytes), append, label)
    _assert_unchanged(sources, sources_before, "an append input")
    _assert_unchanged(indices, indices_before, "an index tensor")


def _gqa_id(case):
    kv_layout, cache_form, dtype, page_size, head_dim, num_kv_heads = case
    return f"{kv_layout}-{cache_form}-{dtype}-ps{page_size}-d{head_dim}-h{num_kv_heads}"


@parametrize_product(
    {
        "kv_layout": ["NHD", "HND"],
        "cache_form": ["tuple", "combined"],
        "dtype": list(_DTYPES),
        "page_size": [1, 7, 16, 64],
        "head_dim": [64, 128, 256, 512],
        "num_kv_heads": [1, 4],
    },
    regular=pairwise_product_cases,
    ids=_gqa_id,
)
def test_page_append_output_contract(
    kv_layout, cache_form, dtype, page_size, head_dim, num_kv_heads
):
    """Of a fully poisoned cache, exactly the (page, slot) of each appended
    token changes, and it holds that token's bytes; the inputs are not written."""
    _gqa_case(
        _metadata(page_size),
        kv_layout,
        cache_form,
        _DTYPES[dtype],
        num_kv_heads,
        head_dim,
    )


_NUM_Q_HEADS = 8


def _qkv_inputs(k, v):  # K and V heads of a fused [nnz, Hq + 2 Hk, D] projection
    nnz, heads, head_bytes = k.shape
    gen = torch.Generator(device="cuda").manual_seed(_SEED + 1)
    qkv = _random_bytes((nnz, _NUM_Q_HEADS + 2 * heads, head_bytes), gen)
    qkv[:, _NUM_Q_HEADS : _NUM_Q_HEADS + heads] = k
    qkv[:, _NUM_Q_HEADS + heads :] = v
    return (
        qkv[:, _NUM_Q_HEADS : _NUM_Q_HEADS + heads],
        qkv[:, _NUM_Q_HEADS + heads :],
        [qkv],
    )


def _kv_pair_inputs(k, v):  # kv[:, 0] and kv[:, 1] of one [nnz, 2, H, D] tensor
    kv = torch.stack([k, v], dim=1)
    return kv[:, 0], kv[:, 1], [kv]


def _head_major_inputs(k, v):  # [H, nnz, D] buffers, so the token stride is the smaller
    k_hm, v_hm = k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()
    return k_hm.transpose(0, 1), v_hm.transpose(0, 1), [k_hm, v_hm]


def _mixed_inputs(k, v):  # contiguous K with a head-major V: K and V strides differ
    v_hm = v.transpose(0, 1).contiguous()
    return k, v_hm.transpose(0, 1), [k, v_hm]


_STRIDED_INPUTS = {
    "qkv": _qkv_inputs,
    "kv_pair": _kv_pair_inputs,
    "head_major": _head_major_inputs,
    "mixed": _mixed_inputs,
}


@pytest.mark.parametrize("source", list(_STRIDED_INPUTS))
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_page_append_strided_inputs(kv_layout, source):
    """K and V as strided views land in the cache like contiguous ones, and no
    byte of the source buffers (the q heads included) is written."""
    _gqa_case(
        _metadata(16),
        kv_layout,
        "tuple",
        torch.bfloat16,
        4,
        128,
        inputs=_STRIDED_INPUTS[source],
    )


@pytest.mark.parametrize("cache_view", ["padded", "transposed"])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_page_append_strided_cache(kv_layout, cache_view):
    """Cache views whose strides are not those of a dense cache in the API
    layout: with a padding head on each side of the heads the API sees, only
    the target slots of the visible heads change; stored in the other layout,
    the slots land where the view says."""
    _gqa_case(
        _metadata(16), kv_layout, "tuple", torch.float16, 4, 128, cache_view=cache_view
    )


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_page_append_index_dtype(kv_layout):
    """int64 index tensors work like the int32 ones every other test uses.
    After the first token, the batch indices and positions are not zero, so a
    kernel that read the int64 buffers as int32 words would see zeros and
    shifted values and write other slots."""
    _gqa_case(
        _metadata(16),
        kv_layout,
        "tuple",
        torch.bfloat16,
        4,
        128,
        index_dtype=torch.int64,
    )


def test_page_append_index_dtype_recipe():
    """The recipe in the ``append_paged_kv_cache`` docstring gives the batch
    indices and positions the tests build on the host."""
    meta = _metadata(16)
    seq_lens = flashinfer.get_seq_lens(
        meta.kv_indptr, meta.kv_last_page_len, meta.page_size
    )
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        meta.append_indptr, seq_lens, meta.nnz
    )
    assert torch.equal(batch_indices, meta.batch_indices)
    assert torch.equal(positions, meta.positions)


def test_page_append_large_batch():
    """16384 tokens, so each CTA appends more than one token."""
    _gqa_case(_metadata(16, _LARGE_REQUESTS), "NHD", "tuple", torch.bfloat16, 8, 128)


# ---------------------------------------------------------------------------
# append_paged_mla_kv_cache
# ---------------------------------------------------------------------------


def _mla_case(meta, cache_form, input_form, dtype, index_dtype=torch.int32):
    """The contract for the MLA latent cache: ckv and kpe are two caches, or two
    views of one ``[ckv | kpe]`` row per slot, and the inputs are contiguous or
    two views of one ``[nnz, ckv + kpe]`` latent tensor."""
    size = dtype.itemsize
    ckv_bytes, kpe_bytes = _CKV_DIM * size, _KPE_DIM * size
    gen = torch.Generator(device="cuda").manual_seed(_SEED)
    ckv = _random_bytes((meta.nnz, ckv_bytes), gen)
    kpe = _random_bytes((meta.nnz, kpe_bytes), gen)
    if input_form == "fused":
        latent = torch.cat([ckv, kpe], dim=1)
        ckv_in, kpe_in, sources = latent[:, :ckv_bytes], latent[:, ckv_bytes:], [latent]
    else:
        ckv_in, kpe_in, sources = ckv, kpe, [ckv, kpe]
    sources_before = [s.clone() for s in sources]
    indices = [getattr(meta, name).to(index_dtype) for name in _INDEX_NAMES]
    indices_before = [t.clone() for t in indices]

    rows = (meta.num_pages, meta.page_size)
    if cache_form == "fused":
        storages = [
            torch.empty(
                (*rows, ckv_bytes + kpe_bytes), dtype=torch.uint8, device="cuda"
            )
        ]

        def regions(s):
            return s[0][..., :ckv_bytes], s[0][..., ckv_bytes:]

    else:
        storages = [
            torch.empty((*rows, ckv_bytes), dtype=torch.uint8, device="cuda"),
            torch.empty((*rows, kpe_bytes), dtype=torch.uint8, device="cuda"),
        ]

        def regions(s):
            return s[0], s[1]

    def write(region, data):
        region[meta.pages, meta.entries] = data

    ckv_cache, kpe_cache = (r.view(dtype) for r in regions(storages))

    def append():
        flashinfer.append_paged_mla_kv_cache(
            ckv_in.view(dtype),
            kpe_in.view(dtype),
            indices[0],
            indices[1],
            ckv_cache,
            kpe_cache,
            indices[2],
            indices[3],
            indices[4],
        )

    label = (
        f"MLA {cache_form} cache, {input_form} inputs, {dtype} "
        f"page_size={meta.page_size} indices={index_dtype}"
    )
    _check_append(storages, regions, write, (ckv, kpe), append, label)
    _assert_unchanged(sources, sources_before, "an append input")
    _assert_unchanged(indices, indices_before, "an index tensor")


def _mla_id(case):
    cache_form, input_form, dtype, page_size = case
    return f"{cache_form}_cache-{input_form}_inputs-{dtype}-ps{page_size}"


@parametrize_product(
    {
        "cache_form": ["separate", "fused"],
        "input_form": ["contiguous", "fused"],
        "dtype": list(_DTYPES),
        "page_size": [1, 7, 16, 64],
    },
    regular=pairwise_product_cases,
    ids=_mla_id,
)
def test_page_append_mla_output_contract(cache_form, input_form, dtype, page_size):
    """Of a fully poisoned latent cache, exactly the (page, slot) of each
    appended token changes in ckv and in kpe; the inputs are not written."""
    _mla_case(_metadata(page_size), cache_form, input_form, _DTYPES[dtype])


def test_page_append_large_batch_mla():
    """16384 tokens through the MLA append."""
    _mla_case(_metadata(16, _LARGE_REQUESTS), "separate", "contiguous", torch.bfloat16)


def test_page_append_index_dtype_mla():
    """int64 index tensors through the MLA append, which casts them in its
    own wrapper."""
    _mla_case(
        _metadata(16), "separate", "contiguous", torch.bfloat16, index_dtype=torch.int64
    )
