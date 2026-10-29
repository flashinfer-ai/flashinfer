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

Backend suite of the Cake QSA NVFP4 sparse paged attention package
(``flashinfer.experimental.cake_qsa_nvfp4``): gated sparse paged GQA decode of BF16
queries over an NVFP4 paged KV cache, for SM80, SM90, SM100, SM103, SM107 and SM120.

The reference and the fixtures are those of ``test_qsa_nvfp4.py``, the suite of the
existing implementation: a float64 model of the kernel's arithmetic with its elementwise
tolerance, the ``2^-6`` accuracy limit against exact attention, count-0 / count-1 rows,
unmapped pages, blocks past the cache, rows without a request, NHD and HND caches and the
group sizes 12, 16 and 1.  The split count the reference models is the package's host
split rule (``num_splits``, decided by the device's architecture and SM count).  This suite adds the
host rule itself (formula, measured table, tile and merge caps), a 33-split case, bitwise
repeatability, CUDA-graph replay with and without a caller-owned workspace, the
zero-allocation workspace path, and the ``build.json`` contract of the package.

The suite calls the package's host entry directly; the public route
``flashinfer.sparse.qsa_sparse_paged_attention(..., backend="cake")`` is covered by
``test_cake_qsa_nvfp4_api.py``.  Besides the package it imports nothing of flashinfer: the
cache the kernel reads is filled by ``tests.test_helpers.cake_qsa_nvfp4_reference``, a torch
port of FlashInfer's NVFP4 paged writer (the API file compares the two).
"""

import gc
import hashlib
import json
import math
import re
import sys
import threading
import weakref
from pathlib import Path

import pytest
import torch

from flashinfer.experimental.cake_qsa_nvfp4 import cake_backend, cake_jit
from flashinfer.experimental.cake_qsa_nvfp4.cake_backend import (
    SPLIT_OVERRIDES,
    num_splits,
)
from tests.test_helpers.cake_qsa_nvfp4_reference import write_slots

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent

HEAD_DIM = 256
PAGE_SIZE = 16
WIDTH = 2051  # Qwen QSA selection width: budget 2048 + compress ratio 4 - 1
NUM_REQUESTS, TABLE_WIDTH = 3, 160
NUM_BLOCKS = NUM_REQUESTS * TABLE_WIDTH + 20
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

TILE = 64  # tokens per split tile
# The decodes' launch bounds: the split formula's capacity is this many CTAs per SM, per
# architecture (two for the mma.sync decode of sm_80 / sm_90a / sm_120a, one for the tcgen05
# decode of sm_100a / sm_103a / sm_107a, whose tensor memory and shared-memory pool hold the SM).
CTAS_PER_SM = {
    "sm_80": 2,
    "sm_90a": 2,
    "sm_100a": 1,
    "sm_103a": 1,
    "sm_107a": 1,
    "sm_120a": 2,
}
# The mma.sync decode's shared-memory pool, passed as dynamic shared memory by its binding;
# two CTAs of it fit an SM (the launch bound).  The merge uses static shared memory alone.
DECODE_SMEM_BYTES = 49792
# The tcgen05 decode's pool (four raw stages, 158,848 B, plus its barrier words), passed as
# dynamic shared memory by its binding; one CTA of it holds an SM (the launch bound).
DECODE_TCGEN05_SMEM_BYTES = 159872
# The persistent walk's pool (the tcgen05 pool plus a second Q^T buffer and an epilogue
# exchange region of its own: the raw ring stays live across the pairs a CTA walks).
DECODE_TCGEN05_PERSISTENT_SMEM_BYTES = 192896
MAX_SPLITS = 128  # the merge kernel's split bound: the host rule never passes more
ROWS = (1, 2, 4, 8, 16, 32, 64, 128)  # the decode row counts of the owner's shape list

# The kernel parameter names, in parameter order: the argument plan of each program names
# them, then the three grid arguments.
DECODE_PARAMETERS = (
    "q",
    "k_data",
    "v_data",
    "k_sf",
    "v_sf",
    "indices",
    "block_table",
    "token_to_req",
    "gate",
    "out",
    "partial_out",
    "partial_lse",
    "merge_counters",
    "q_stride_row",
    "q_stride_head",
    "data_stride_block",
    "data_stride_token",
    "data_stride_head",
    "sf_stride_block",
    "sf_stride_token",
    "sf_stride_head",
    "indices_stride_row",
    "table_stride_req",
    "out_stride_row",
    "out_stride_head",
    "score_scale",
    "output_scale",
    "num_rows",
    "num_requests",
    "num_blocks",
    "page_div",
    "table_width",
    "topk",
    "group_size",
    "num_qo_heads",
    "num_tiles",
    "num_splits",
)
MERGE_PARAMETERS = (
    "partial_out",
    "partial_lse",
    "gate",
    "out",
    "out_stride_row",
    "out_stride_head",
    "num_splits",
    "num_qo_heads",
)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {cake_jit.ARCHES}"
        )
    if not (PACKAGE_DIR / "build.json").exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", index)


def _sm_count(device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _splits(rows, num_kv_heads, device) -> int:
    """The split count the package passes for ``rows`` on ``device``: its host split rule
    (the persistent walk's partial slots where the walk serves the call)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return num_splits(
        rows, num_kv_heads, WIDTH, _sm_count(device), cake_jit.device_arch(index)
    )


def _walk(rows, num_kv_heads, device) -> int:
    """The persistent walk's CTA count the package launches for ``rows`` on ``device``, 0
    where the plain cut is kept (every architecture but sm_103a; one-wave grids)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    return cake_backend.select_walk(
        rows,
        num_kv_heads,
        math.ceil(WIDTH / TILE),
        _sm_count(device),
        cake_jit.device_arch(index),
    )


def _walk_segments(rows, num_kv_heads, num_tiles, ctas):
    """Each tile's partial slot under the persistent walk's cut, ``[rows, kv_heads, tiles]``:
    the kernel's slot math restated here rather than imported -- slot ``pair * tiles + tile``
    over ``pairs = rows * kv_heads``, CTA ``c`` of ``G`` walks ``[c S // G, (c + 1) S // G)``
    (``S`` the slot count), the CTA holding slot ``s`` is ``((s + 1) G - 1) // S``, and a
    pair's segment ``k`` is its ``k``-th CTA."""
    pairs = rows * num_kv_heads
    slots = pairs * num_tiles

    def owner(slot):
        return ((slot + 1) * ctas - 1) // slots

    seg = torch.empty(rows, num_kv_heads, num_tiles, dtype=torch.long)
    for pair in range(pairs):
        first = owner(pair * num_tiles)
        for tile in range(num_tiles):
            seg[pair // num_kv_heads, pair % num_kv_heads, tile] = (
                owner(pair * num_tiles + tile) - first
            )
    return seg


def _dequant(data, sf, scale):
    lut = torch.tensor(_E2M1 + [-x for x in _E2M1], device=data.device)
    values = torch.stack((lut[(data & 15).long()], lut[(data >> 4).long()]), -1)
    return values.flatten(-2) * sf.float().repeat_interleave(16, -1) * scale


def _make_cache(num_kv_heads, layout, seed):
    """Quantize random K/V into every slot in the NVFP4 writer's format (the torch port of
    ``tests.test_helpers.cake_qsa_nvfp4_reference``), using one buffer of [data | scales]
    rows per side like a packed vLLM page. Returns the kernel's [blocks, tokens, heads, *]
    views, the global scales and the dequantized cache."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    slots = NUM_BLOCKS * PAGE_SIZE
    k, v = (
        scale * torch.randn(slots, num_kv_heads, HEAD_DIM, device="cuda", generator=gen)
        for scale in (1.0, 3.0)
    )
    k, v = k.bfloat16(), v.bfloat16()
    dims = (num_kv_heads, PAGE_SIZE) if layout == "HND" else (PAGE_SIZE, num_kv_heads)
    row_bytes = HEAD_DIM // 2 + HEAD_DIM // 16
    kv = torch.empty(NUM_BLOCKS, 2, *dims, row_bytes, dtype=torch.uint8, device="cuda")
    data = tuple(kv[:, i, ..., : HEAD_DIM // 2] for i in range(2))
    sf = tuple(
        kv[:, i, ..., HEAD_DIM // 2 :].view(torch.float8_e4m3fn) for i in range(2)
    )
    scales = tuple(x.abs().max().item() / (6 * 448) for x in (k, v))
    write_slots(k, v, data, sf, *scales, kv_layout=layout)
    if layout == "HND":
        data, sf = (tuple(x.transpose(1, 2) for x in pair) for pair in (data, sf))
    deq = tuple(_dequant(*args) for args in zip(data, sf, scales, strict=True))
    return data, sf, scales, deq


def _make_selection(rows, seed):
    gen = torch.Generator().manual_seed(seed)
    table = torch.randperm(NUM_BLOCKS, generator=gen)[: NUM_REQUESTS * TABLE_WIDTH]
    table = table.view(NUM_REQUESTS, TABLE_WIDTH).int()
    table[0, 1::5] = -1  # unmapped pages
    table[1, 2] = NUM_BLOCKS + 5  # a block past the cache
    token_to_req = torch.randint(NUM_REQUESTS, (rows,), generator=gen).int()
    if rows > 3:
        token_to_req[3] = -1  # a row without a request
    # Tokens a little past the table, -1 entries, and counts full, 0, 1, then random.
    # Like vLLM's expand, every entry past a row's count is -1.
    indices = torch.randint(
        TABLE_WIDTH * PAGE_SIZE + 64, (rows, WIDTH + 1), generator=gen
    )
    indices[:, ::7] = -1
    if rows > 2:
        indices[2, 0] = 5  # the count-1 row selects one real token (page 0 is mapped)
    counts = torch.randint(WIDTH + 1, (rows,), generator=gen)
    counts[:3] = torch.tensor([WIDTH, 0, 1])[:rows]
    indices[:, :WIDTH][torch.arange(WIDTH) >= counts[:, None]] = -1
    indices[:, -1] = counts
    return table.cuda(), token_to_req.cuda(), indices.int().cuda()


def _ulp_bf16(x):
    """BF16 spacing at |x|, and 0 at 0."""
    return torch.ldexp(torch.ones_like(x), torch.frexp(x)[1] - 8) * (x != 0)


def _reference(q, cache, selection, gate, num_splits, chunk=64, segments=None):
    """Float64 references, row chunk by row chunk.

    contract: the kernel's arithmetic -- 64-column tiles walked per split with an online
    softmax, P rounded to BF16, LSE merge, BF16 rounding, FP32 gate, BF16 output -- and a
    worst-case bound on how far one-ulp P rounding, FP32 scores (with exp2.approx), FP32
    accumulation and the two output roundings can move the kernel from it.
    exact: softmax attention times the gate, unrounded, and sum_n p_n |v_n| times the gate.
    The cut of a (row, KV head)'s tiles into its ``num_splits`` partials is the plain
    program's strided one (tile ``t`` in split ``t % num_splits``) or, given ``segments``
    (``[rows, kv_heads, tiles]`` of partial slots, :func:`_walk_segments`), the persistent
    walk's contiguous one; within a partial the tiles are walked in order either way.
    """
    k_deq, v_deq = cache[3]
    table, token_to_req, indices = selection
    num_kv_heads = k_deq.shape[2]
    log2_scale = math.log2(math.e) / math.sqrt(HEAD_DIM)
    num_tiles = math.ceil(WIDTH / 64)
    parts = []
    for r0 in range(0, q.shape[0], chunk):
        token = indices[r0 : r0 + chunk, :WIDTH].long()
        req = token_to_req[r0 : r0 + chunk].long()
        page = token.clamp(min=0) // PAGE_SIZE
        block = table[
            req.clamp(0, NUM_REQUESTS - 1)[:, None], page.clamp(max=TABLE_WIDTH - 1)
        ]
        valid = ((req >= 0) & (req < NUM_REQUESTS))[:, None] & (token >= 0)
        valid &= (page < TABLE_WIDTH) & (block >= 0) & (block < NUM_BLOCKS)
        slot = (block.long().clamp(0, NUM_BLOCKS - 1), token.clamp(min=0) % PAGE_SIZE)
        k, v = (
            x[slot].double() for x in (k_deq, v_deq)
        )  # [rows, width, kv_heads, dim]
        qd = q[r0 : r0 + chunk].double().unflatten(1, (num_kv_heads, -1))
        mask = valid[:, None, None]
        s = torch.einsum("rhgd,rwhd->rhgw", qd, k) * log2_scale
        s_abs = (torch.einsum("rhgd,rwhd->rhgw", qd.abs(), k.abs()) * mask).amax(-1)
        s = s.masked_fill(~mask, -1e20)
        e = torch.exp2(s - s.amax(-1, keepdim=True)) * mask
        p = e / e.sum(-1, keepdim=True).clamp(min=1e-300)
        exact = torch.einsum("rhgw,rwhd->rhgd", p, v)
        bound = torch.einsum("rhgw,rwhd->rhgd", p, v.abs())
        seg = None if segments is None else segments[r0 : r0 + chunk].to(s.device)
        splits = []
        for split in range(num_splits):
            m = torch.full(s.shape[:-1], -1e20, dtype=s.dtype, device=s.device)
            l, acc, flip = torch.zeros_like(m), 0.0, 0.0
            if seg is None:
                tiles = range(split, num_tiles, num_splits)
            else:
                tiles = [t for t in range(num_tiles) if (seg[..., t] == split).any()]
            for tile in tiles:
                cols = slice(64 * tile, 64 * tile + 64)
                m_new = torch.maximum(m, s[..., cols].amax(-1))
                if seg is not None:
                    # Only the (row, KV head) pairs whose segment ``split`` holds this tile
                    # take it; the others keep their state (alpha = 1, nothing added).
                    take = (seg[..., tile] == split)[..., None]
                    m_new = torch.where(take, m_new, m)
                pt = torch.exp2(s[..., cols] - m_new[..., None]) * mask[..., cols]
                if seg is not None:
                    # Select 0 rather than multiply by 0: before the pair's first held tile of this
                    # slot ``m`` is still -1e20 and exp2(s - m) is inf (inf * 0 would be NaN and
                    # ``nan_to_num`` would then drop the whole partial).
                    pt = torch.where(take[..., None], pt, 0.0)
                pb = pt.bfloat16().double()
                alpha = torch.exp2(m - m_new)[..., None]
                acc = acc * alpha + torch.einsum("rhgw,rwhd->rhgd", pb, v[:, cols])
                flip = flip * alpha + torch.einsum(
                    "rhgw,rwhd->rhgd", _ulp_bf16(pb), v[:, cols].abs()
                )
                l, m = l * alpha[..., 0] + pt.sum(-1), m_new
            lse = torch.where(l > 0, m + torch.log2(l), -math.inf)[..., None]
            splits.append((acc / l[..., None], flip / l[..., None], lse))
        out, flip, lse = (
            torch.stack(x).nan_to_num(0) for x in zip(*splits, strict=True)
        )
        weight = torch.where(lse > -math.inf, torch.exp2(lse - lse.amax(0)), 0.0)
        weight = (weight / weight.sum(0).clamp(min=1e-300)).nan_to_num(0)
        # Relative error of each P: twice the FP32 score error bound, plus exp2.approx.
        eps_p = 2 * HEAD_DIM * 2**-23 * math.log(2) * s_abs * log2_scale + 2**-21
        parts.append(
            (
                (weight * out).sum(0),
                (weight * flip).sum(0),
                eps_p[..., None],
                exact,
                bound,
            )
        )
    pre, flip, eps_p, exact, bound = (
        torch.cat(x).flatten(1, 2) for x in zip(*parts, strict=True)
    )
    sigma = torch.sigmoid(gate.float())
    rounded = pre.float().bfloat16()
    contract = (rounded.float() * sigma).bfloat16().double()
    sigma = sigma.double()
    tolerance = (
        sigma * (2 * _ulp_bf16(rounded.double()) + flip)
        + sigma * (2 * eps_p + 2 * WIDTH * 2**-23) * bound
        + _ulp_bf16(contract)
    )
    return contract, tolerance, exact * sigma, bound * sigma


def _run(q, cache, selection, gate, out=None, **extra):
    data, sf, (k_scale, v_scale), _ = cache
    table, token_to_req, indices = selection
    return cake_backend.qsa_sparse_paged_attention(
        q,
        data[0],
        data[1],
        indices,
        table,
        token_to_req,
        False,
        out,
        k_scale,
        v_scale,
        output_gate=gate,
        k_sf=sf[0],
        v_sf=sf[1],
        **extra,
    )


def _inputs(rows, num_qo_heads, num_kv_heads, layout, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q, gate = (
        torch.randn(
            rows, num_qo_heads, HEAD_DIM, device="cuda", generator=gen
        ).bfloat16()
        for _ in range(2)
    )
    # q and the cache draw from the same seed on purpose: some q rows then equal a cached
    # K and their attention collapses onto that token, which stresses small probabilities.
    return q, _make_cache(num_kv_heads, layout, seed), _make_selection(rows, seed), gate


def _check_contract(out, q, cache, selection, gate, splits, walk=0):
    segments = (
        _walk_segments(q.shape[0], cache[3][0].shape[2], math.ceil(WIDTH / TILE), walk)
        if walk
        else None
    )
    contract, tolerance, exact, bound = _reference(
        q, cache, selection, gate, splits, segments=segments
    )
    assert ((out - contract).abs() <= tolerance).all()
    # Accuracy limit against exact attention; validated empirically, not a proven bound.
    assert ((out - exact).abs() / bound.clamp(min=1e-12)).max().item() <= 2**-6


# ---------------------------------------------------------------------------
# Device tests: the contract reference, the forms the host rule selects, replay, workspace
# ---------------------------------------------------------------------------


# The eight row counts of the owner's shape list at 24 x 2 heads, the 2048-row case (one
# split, more CTAs than the device holds at once) and the group sizes 16 and 1.  On 132 SMs
# the host rule gives 33, 33, 16, 8, 8, 4, 2, 1 splits for rows 1 .. 128; on 108 SMs 33, 33,
# 27, 13, 6, 3, 4 and 2 (rows 64 and 128 then run the separate merge and, with the 2048-row
# case, exceed one resident wave: on sm_80 they launch the whole-tile decode).
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "rows,num_qo_heads,num_kv_heads",
    [
        (1, 24, 2),
        (2, 24, 2),
        (4, 24, 2),
        (8, 24, 2),
        (16, 24, 2),
        (32, 24, 2),
        (64, 24, 2),
        (128, 24, 2),
        (2048, 24, 2),
        (4, 16, 1),
        (4, 2, 2),
    ],
)
def test_matches_reference(layout, rows, num_qo_heads, num_kv_heads):
    device = _device_or_skip()
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, layout, rows)
    out = _run(q, cache, selection, gate).double()
    _check_contract(
        out,
        q,
        cache,
        selection,
        gate,
        _splits(rows, num_kv_heads, device),
        _walk(rows, num_kv_heads, device),
    )
    if rows > 2:
        assert not out[1].any() and out[2].any()  # count 0; count 1 with a real token


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_thirty_three_splits(layout):
    """Rows 1 and 2 at two KV heads take every tile as its own split (33 splits, 66 and
    132 CTAs) on any device of at least 33 SMs."""
    device = _device_or_skip()
    if _sm_count(device) < math.ceil(WIDTH / TILE):
        pytest.skip("a device of at least 33 SMs is needed for 33 splits")
    assert math.ceil(WIDTH / TILE) == 33
    assert _splits(1, 2, device) == _splits(2, 2, device) == 33
    q, cache, selection, gate = _inputs(2, 24, 2, layout, 33)
    out = _run(q, cache, selection, gate).double()
    _check_contract(out, q, cache, selection, gate, 33)


def test_two_wave_walk_matches_the_plain_cut_and_replays(monkeypatch):
    """On a GB300 (sm_103a, 152 SMs) or an R200 (sm_107a, 212 SMs) rows 128 at two KV heads
    take the persistent walk:
    one CTA per SM walking a contiguous share of the (row, KV head) x tile slots, 2 partial
    slots per pair, then the merge.  The walked call meets the contract of its own cut;
    forced onto the plain cut (one split, 256 CTAs in two waves: the host's walk rule off)
    the same call meets the contract of the strided cut and differs from the walked one by
    the merge's association alone, within the kernel tests' bound; the walked call replays
    bit for bit under CUDA-graph capture.  Skipped where the host walks nothing (every
    other architecture; SM counts whose rows 128 grid is one wave)."""
    device = _device_or_skip()
    rows, num_kv_heads = 128, 2
    walk = _walk(rows, num_kv_heads, device)
    if not walk:
        pytest.skip("the host walks nothing on this device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    assert cake_jit.device_arch(index) in cake_backend.WALK_ARCHES
    assert walk == _sm_count(device) and _splits(rows, num_kv_heads, device) == 2
    q, cache, selection, gate = _inputs(rows, 24, num_kv_heads, "NHD", 1280)
    walked = _run(q, cache, selection, gate)
    _check_contract(walked.double(), q, cache, selection, gate, 2, walk)
    cake_backend._resolve.cache_clear()
    monkeypatch.setattr(cake_backend, "WALK_ARCHES", ())
    try:
        assert _walk(rows, num_kv_heads, device) == 0
        plain_splits = _splits(rows, num_kv_heads, device)
        assert plain_splits == 1
        plain = _run(q, cache, selection, gate)
    finally:
        monkeypatch.undo()
        cake_backend._resolve.cache_clear()
    assert _walk(rows, num_kv_heads, device) == walk
    _check_contract(plain.double(), q, cache, selection, gate, plain_splits)
    torch.testing.assert_close(walked.float(), plain.float(), rtol=2**-6, atol=0.04)
    out = torch.empty_like(q)
    _run(q, cache, selection, gate, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(q, cache, selection, gate, out)
    for dst, src in zip(selection, _make_selection(rows, 1281), strict=True):
        dst.copy_(src)
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.testing.assert_close(out, _run(q, cache, selection, gate), rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 8, 64])
def test_repeated_calls_are_bitwise_equal(rows):
    """The same inputs give the same bits call after call, through the per-call partials
    and the stream's cached counters, and through one caller-owned workspace launched
    three times in a row (the group counters of the in-launch merge alternate
    generations and end where they started)."""
    device = _device_or_skip()
    q, cache, selection, gate = _inputs(rows, 24, 2, "NHD", 100 + rows)
    first = _run(q, cache, selection, gate)
    assert torch.isfinite(first.float()).all()
    # Zeroed: the counters' zero state is the kernel's precondition; that the host zeroes a
    # buffer it has not seen is the last test's subject.
    workspace = torch.zeros(
        cake_backend.qsa_workspace_bytes(rows, 24, 2, _splits(rows, 2, device)),
        dtype=torch.uint8,
        device=device,
    )
    out = torch.full_like(q, math.nan)
    for _ in range(3):
        _run(q, cache, selection, gate, out, workspace_buffer=workspace)
        assert torch.equal(out, first)
        out.fill_(math.nan)
    assert torch.equal(_run(q, cache, selection, gate), first)


def test_stream_counters_are_cached_per_stream():
    """Without a workspace the fused form's counters come from one idle buffer per
    (device, stream): the same call reuses it, a call needing more counters replaces it
    with a larger zeroed one, another stream gets its own, and every call gives the same
    bits."""
    device = _device_or_skip()
    rows, num_qo_heads, num_kv_heads = 1, 24, 2
    index = device.index if device.index is not None else torch.cuda.current_device()
    splits = _splits(rows, num_kv_heads, device)
    if (
        splits == 1
        or cake_backend.select_form(
            rows, num_kv_heads, splits, cake_backend.num_tiles(WIDTH), index
        )
        != "fused"
    ):
        pytest.skip("the fused form is not selected on this device")
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, "NHD", 61)
    expected = _run(q, cache, selection, gate)
    key = (index, torch.cuda.current_stream(index).cuda_stream)
    cake_backend._stream_counters.pop(key, None)
    assert torch.equal(_run(q, cache, selection, gate), expected)
    buffer = cake_backend._stream_counters[key]
    assert buffer.dtype == torch.uint32 and buffer.numel() >= rows * num_kv_heads
    assert torch.equal(_run(q, cache, selection, gate), expected)
    assert cake_backend._stream_counters[key] is buffer
    # Too small for the call: replaced by a zeroed buffer that holds the counters.
    cake_backend._stream_counters[key] = torch.zeros(
        1, dtype=torch.uint32, device=device
    )
    assert torch.equal(_run(q, cache, selection, gate), expected)
    grown = cake_backend._stream_counters[key]
    assert grown.numel() >= rows * num_kv_heads
    # Another stream gets its own buffer and the same bits.
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        other = _run(q, cache, selection, gate)
    stream.synchronize()
    other_key = (index, stream.cuda_stream)
    assert other_key in cake_backend._stream_counters
    assert (
        cake_backend._stream_counters[other_key]
        is not cake_backend._stream_counters[key]
    )
    assert torch.equal(other, expected)


def test_caller_workspace_allocates_nothing():
    device = _device_or_skip()
    rows, num_qo_heads, num_kv_heads = 8, 24, 2
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, "HND", 21)
    splits = _splits(rows, num_kv_heads, device)
    nbytes = cake_backend.qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits)
    # FP32 partial outputs and LSEs per split, one 32-bit group counter per (row, KV head).
    partials = splits * rows * num_qo_heads * (HEAD_DIM + 1) * 4
    assert nbytes >= partials + rows * num_kv_heads * 4
    workspace = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    expected = _run(q, cache, selection, gate)
    out = torch.full_like(q, math.nan)
    # The first call with a buffer records it; every later call reuses its counters.
    _run(q, cache, selection, gate, out, workspace_buffer=workspace)
    assert torch.equal(out, expected)
    out.fill_(math.nan)
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()["allocation.all.allocated"]
    _run(q, cache, selection, gate, out, workspace_buffer=workspace)
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
    assert torch.equal(out, expected)
    with pytest.raises(ValueError):
        _run(q, cache, selection, gate, out, workspace_buffer=workspace[:-1])


def test_caller_workspace_is_carved_once(monkeypatch):
    """A caller workspace is checked and carved as a first use on its first call and
    once more on its second, which builds its plan; every later call that passes the
    same tensor reuses the plan; a call it is too small for is checked afresh and
    refused; a plan whose dtype no longer matches, or a tensor rebound to other bytes,
    is checked again; ``reset_workspace`` drops the plans and the noted sightings; and
    the plan dies with the tensor (its views hold the storage, not it)."""
    device = _device_or_skip()
    rows, num_qo_heads, num_kv_heads = 8, 24, 2
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, "HND", 23)
    splits = _splits(rows, num_kv_heads, device)
    nbytes = cake_backend.qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits)
    checks = [0]
    real_check = cake_backend._check_workspace

    def counted_check(*args):
        checks[0] += 1
        return real_check(*args)

    monkeypatch.setattr(cake_backend, "_check_workspace", counted_check)
    expected = _run(q, cache, selection, gate)
    workspace = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    cake_backend.reset_workspace(workspace)  # no plan or sighting from another test
    out = torch.full_like(q, math.nan)
    for call in range(5):
        out.fill_(math.nan)
        _run(q, cache, selection, gate, out, workspace_buffer=workspace)
        assert torch.equal(out, expected)
        assert checks[0] == min(call + 1, 2)
        if call == 0:
            assert id(workspace) not in cake_backend._workspace_plans
            assert len(cake_backend._seen_once) == 1
    assert not cake_backend._seen_once
    plan = cake_backend._workspace_plans[id(workspace)]
    assert plan.ref() is workspace and len(plan.views) == 1
    partial_out, partial_lse, counters = next(iter(plan.views.values()))
    assert partial_out.data_ptr() >= workspace.data_ptr()
    assert partial_lse.data_ptr() > partial_out.data_ptr()
    assert partial_out._base is not workspace
    assert partial_lse._base is not workspace
    # A call the buffer is too small for (more query heads) is checked and refused.
    wide_q, _, _, wide_gate = _inputs(rows, 32, num_kv_heads, "HND", 24)
    with pytest.raises(ValueError):
        _run(wide_q, cache, selection, wide_gate, workspace_buffer=workspace)
    assert checks[0] == 3 and len(plan.views) == 1
    # A plan that reads another dtype for the same tensor is dropped; the tensor is
    # then a first use again (checked, carved, noted) and gets a new plan on the call
    # after.
    plan.dtype = torch.int8
    out.fill_(math.nan)
    _run(q, cache, selection, gate, out, workspace_buffer=workspace)
    assert torch.equal(out, expected) and checks[0] == 4
    assert id(workspace) not in cake_backend._workspace_plans
    _run(q, cache, selection, gate, out, workspace_buffer=workspace)
    assert checks[0] == 5
    assert cake_backend._workspace_plans[id(workspace)] is not plan
    # A tensor rebound to bytes of another dtype is refused like any other first use.
    rebound = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    _run(q, cache, selection, gate, out, workspace_buffer=rebound)
    rebound.data = torch.zeros(nbytes, dtype=torch.int8, device=device)
    with pytest.raises(ValueError):
        _run(q, cache, selection, gate, out, workspace_buffer=rebound)
    # reset_workspace drops every plan, every noted sighting and the zero-once record
    # of the storage.
    cake_backend.reset_workspace(workspace)
    assert not cake_backend._workspace_plans and not cake_backend._seen_once
    assert workspace.untyped_storage() not in cake_backend._zeroed_counters
    before = checks[0]
    for _ in range(3):
        out.fill_(math.nan)
        _run(q, cache, selection, gate, out, workspace_buffer=workspace)
        assert torch.equal(out, expected)
    assert checks[0] == before + 2
    plan = cake_backend._workspace_plans[id(workspace)]
    # The plan dies with the tensor and releases the buffer.
    ident = id(workspace)
    ref = weakref.ref(workspace)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated()
    del plan, partial_out, partial_lse, counters, workspace
    gc.collect()
    assert ref() is None and ident not in cake_backend._workspace_plans
    assert torch.cuda.memory_allocated() <= held - nbytes


def test_alternating_caller_workspaces_reach_their_plans(monkeypatch):
    """Two workspaces passed turn about are each checked twice (first use, then the
    plan's build on the second sighting) and reuse their plans from their third call
    on."""
    device = _device_or_skip()
    rows, num_qo_heads, num_kv_heads = 4, 24, 2
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, "NHD", 31)
    splits = _splits(rows, num_kv_heads, device)
    nbytes = cake_backend.qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits)
    checks = [0]
    real_check = cake_backend._check_workspace

    def counted_check(*args):
        checks[0] += 1
        return real_check(*args)

    monkeypatch.setattr(cake_backend, "_check_workspace", counted_check)
    expected = _run(q, cache, selection, gate)
    buffers = [torch.zeros(nbytes, dtype=torch.uint8, device=device) for _ in range(2)]
    cake_backend.reset_workspace(buffers[0])  # no plan or sighting from another test
    out = torch.full_like(q, math.nan)
    for call in range(8):
        out.fill_(math.nan)
        _run(q, cache, selection, gate, out, workspace_buffer=buffers[call % 2])
        assert torch.equal(out, expected)
        assert checks[0] == min(call + 1, 4)
    assert all(id(b) in cake_backend._workspace_plans for b in buffers)
    assert not cake_backend._seen_once
    cake_backend.reset_workspace(buffers[0])
    assert not cake_backend._workspace_plans


def test_fresh_caller_workspace_views_build_no_plan(monkeypatch):
    """A caller that passes a new view object of its buffer on every call is checked and
    carved on every call, as before the plans, and never builds one: the view dies
    between calls, so a later view with the same id and bytes is a first use again."""
    device = _device_or_skip()
    rows, num_qo_heads, num_kv_heads = 8, 24, 2
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, "NHD", 37)
    splits = _splits(rows, num_kv_heads, device)
    nbytes = cake_backend.qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits)
    checks = [0]
    real_check = cake_backend._check_workspace

    def counted_check(*args):
        checks[0] += 1
        return real_check(*args)

    monkeypatch.setattr(cake_backend, "_check_workspace", counted_check)
    expected = _run(q, cache, selection, gate)
    buffer = torch.zeros(nbytes + 256, dtype=torch.uint8, device=device)
    cake_backend.reset_workspace(buffer)  # no plan or sighting from another test
    out = torch.full_like(q, math.nan)
    for call in range(6):
        out.fill_(math.nan)
        _run(q, cache, selection, gate, out, workspace_buffer=buffer[:nbytes])
        assert torch.equal(out, expected)
        assert checks[0] == call + 1
        assert not cake_backend._workspace_plans
    assert all(ref() is None for ref in cake_backend._seen_once.values())
    cake_backend.reset_workspace(buffer)
    assert not cake_backend._seen_once


def test_single_split_caller_workspace_is_checked_not_carved(monkeypatch):
    """At one split the call launches without partials: a caller workspace is checked
    (on a first use and on the plan's build) but never carved, a plan hit skips both,
    and a buffer too small for the call is refused as at any other split."""
    device = _device_or_skip()
    rows, num_qo_heads, num_kv_heads = 128, 24, 2
    splits = _splits(rows, num_kv_heads, device)
    if splits != 1:
        pytest.skip(f"{rows} rows take {splits} splits on this device")
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, "NHD", 41)
    nbytes = cake_backend.qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, 1)
    counts = {"check": 0, "carve": 0}
    real_check, real_carve = cake_backend._check_workspace, cake_backend._carve

    def counted_check(*args):
        counts["check"] += 1
        return real_check(*args)

    def counted_carve(*args, **kwargs):
        counts["carve"] += 1
        return real_carve(*args, **kwargs)

    monkeypatch.setattr(cake_backend, "_check_workspace", counted_check)
    monkeypatch.setattr(cake_backend, "_carve", counted_carve)
    expected = _run(q, cache, selection, gate)
    buffer = torch.zeros(nbytes + 256, dtype=torch.uint8, device=device)
    cake_backend.reset_workspace(buffer)  # no plan or sighting from another test
    out = torch.full_like(q, math.nan)
    for _ in range(3):  # a fresh view every call: checked every call, never carved
        out.fill_(math.nan)
        _run(q, cache, selection, gate, out, workspace_buffer=buffer[:nbytes])
        assert torch.equal(out, expected)
    assert counts == {"check": 3, "carve": 0}
    workspace = buffer[:nbytes]  # one tensor object: checked twice, then plan hits
    for _ in range(4):
        out.fill_(math.nan)
        _run(q, cache, selection, gate, out, workspace_buffer=workspace)
        assert torch.equal(out, expected)
    assert counts == {"check": 5, "carve": 0}
    plan = cake_backend._workspace_plans[id(workspace)]
    assert list(plan.views.values()) == [None]
    with pytest.raises(ValueError):
        _run(q, cache, selection, gate, out, workspace_buffer=buffer[: nbytes - 256])
    cake_backend.reset_workspace(buffer)


def test_cuda_graph_replay():
    _device_or_skip()
    q, cache, selection, gate = _inputs(8, 24, 2, "HND", 7)
    out = torch.empty_like(q)
    _run(q, cache, selection, gate, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(q, cache, selection, gate, out)
    for dst, src in zip(selection, _make_selection(8, 8), strict=True):
        dst.copy_(src)
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.testing.assert_close(out, _run(q, cache, selection, gate), rtol=0, atol=0)


def test_cuda_graph_replay_with_caller_workspace():
    """A captured call through a caller-owned workspace replays on new selections and
    queries, three times in a row (both generations of the self-resetting counters), bit
    for bit against eager calls."""
    device = _device_or_skip()
    rows = 8
    q, cache, selection, gate = _inputs(rows, 24, 2, "NHD", 9)
    workspace = torch.zeros(
        cake_backend.qsa_workspace_bytes(rows, 24, 2, _splits(rows, 2, device)),
        dtype=torch.uint8,
        device=device,
    )
    out = torch.empty_like(q)
    # Eager warm-up: the host records the buffer outside capture.
    _run(q, cache, selection, gate, out, workspace_buffer=workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(q, cache, selection, gate, out, workspace_buffer=workspace)
    for seed in (10, 11, 12):
        for dst, src in zip(selection, _make_selection(rows, seed), strict=True):
            dst.copy_(src)
        q.copy_(torch.randn_like(q))
        out.fill_(math.nan)
        graph.replay()
        torch.testing.assert_close(out, _run(q, cache, selection, gate), rtol=0, atol=0)


def test_graph_capture_leaves_the_stream_caches_untouched():
    """A captured call without a workspace takes its partials and counters from the graph's
    private pool: the per-stream scratch and counter tables gain no entry during the capture,
    and the replay is bit for bit the eager result."""
    device = _device_or_skip()
    rows = 8
    if _splits(rows, 2, device) == 1:
        pytest.skip("one split keeps no partials")
    q, cache, selection, gate = _inputs(rows, 24, 2, "NHD", 17)
    out = torch.empty_like(q)
    _run(q, cache, selection, gate, out)
    torch.cuda.synchronize()
    partials_before = {
        key: (id(scratch), frozenset(scratch.views))
        for key, scratch in cake_backend._stream_partials.items()
    }
    counters_before = {
        key: id(buffer) for key, buffer in cake_backend._stream_counters.items()
    }
    views_before = set(cake_backend._stream_counter_views)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(q, cache, selection, gate, out)
    assert {
        key: (id(scratch), frozenset(scratch.views))
        for key, scratch in cake_backend._stream_partials.items()
    } == partials_before
    assert {
        key: id(buffer) for key, buffer in cake_backend._stream_counters.items()
    } == counters_before
    assert set(cake_backend._stream_counter_views) == views_before
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.testing.assert_close(out, _run(q, cache, selection, gate), rtol=0, atol=0)


def test_concurrent_streams_keep_their_scratch():
    """Without a workspace the partials (and the fused form's counters) come from one scratch
    per (device, stream): two streams whose calls interleave on the host without a
    synchronization each get their eager bits, and so do two Python threads enqueuing on
    one stream (the lock orders each call's carve and launches)."""
    device = _device_or_skip()
    rows = 8
    if _splits(rows, 2, device) == 1:
        pytest.skip("one split keeps no partials")
    inputs = [_inputs(rows, 24, 2, "NHD", 40 + i) for i in range(2)]
    expected = [_run(*args) for args in inputs]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream(device) for _ in range(2)]
    outs = [None, None]
    for _ in range(3):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                outs[i] = _run(*inputs[i])
    torch.cuda.synchronize()
    for i, stream in enumerate(streams):
        assert torch.equal(outs[i], expected[i])
        assert (device.index, stream.cuda_stream) in cake_backend._stream_partials
    results = [None, None]
    errors = []
    barrier = threading.Barrier(2)

    def worker(i):
        try:
            torch.cuda.set_device(device)
            barrier.wait(timeout=60)
            for _ in range(20):
                results[i] = _run(*inputs[i])
        except Exception as error:  # reported by the test thread
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    torch.cuda.synchronize()
    assert not errors, errors
    for i in range(2):
        assert torch.equal(results[i], expected[i])


def test_zero_rows_return_the_output_untouched():
    _device_or_skip()
    q, cache, selection, gate = _inputs(4, 24, 2, "NHD", 0)
    table, token_to_req, indices = selection
    out = torch.empty(0, 24, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    result = _run(q[:0], cache, (table, token_to_req[:0], indices[:0]), gate[:0], out)
    assert result is out


def _no_launch(*args, **kwargs):
    raise AssertionError("a launch was reached with invalid inputs")


def test_rejects_invalid_inputs(monkeypatch):
    """Every argument defect raises ``ValueError`` before the host reaches a launch (the
    environment-stream context every launch runs under is replaced by a tripwire)."""
    _device_or_skip()
    q, cache, selection, gate = _inputs(4, 24, 2, "NHD", 0)
    (k_data, v_data), (k_sf, v_sf), (k_scale, v_scale), _ = cache
    table, token_to_req, indices = selection
    args = dict(
        q=q,
        k_cache=k_data,
        v_cache=v_data,
        logical_indices=indices,
        block_table=table,
        token_to_req=token_to_req,
        use_prefill_config=False,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=gate,
        k_sf=k_sf,
        v_sf=v_sf,
    )
    misaligned = torch.empty(q.numel() + 1, dtype=q.dtype, device="cuda")[1:].view(
        q.shape
    )
    wide_group = torch.zeros(4, 48, HEAD_DIM, dtype=q.dtype, device="cuda")
    monkeypatch.setattr(cake_jit, "ffi_stream_context", _no_launch)
    for change in (
        {"q": q.half()},
        {"q": q[..., :128]},
        {"q": misaligned},
        {"q": wide_group, "output_gate": torch.zeros_like(wide_group)},
        {"k_cache": k_data.view(torch.int8)},
        {"v_sf": v_sf[..., :8]},
        {"logical_indices": indices[:, :1]},
        {"logical_indices": indices.long()},
        {"token_to_req": token_to_req[:-1]},
        {"k_scale": None},
        {"output_gate": gate.half()},
        {"workspace_buffer": torch.empty(8, dtype=torch.uint8, device="cuda")},
    ):
        with pytest.raises(ValueError):
            cake_backend.qsa_sparse_paged_attention(**{**args, **change})
    # A workspace is checked at every split count, one included (a selection of one tile
    # takes one split on every device), so a buffer's verdict does not depend on the
    # device's SM count.
    one_tile = torch.full((4, TILE + 1), -1, dtype=torch.int32, device="cuda")
    one_tile[:, -1] = 0
    assert (
        num_splits(
            4, 2, TILE, _sm_count(q.device), cake_jit.device_arch(q.device.index)
        )
        == 1
    )
    nbytes = cake_backend.qsa_workspace_bytes(4, 24, 2, 1)
    for buffer in (
        torch.zeros(nbytes, dtype=torch.uint8),
        torch.zeros(nbytes // 4, dtype=torch.float32, device="cuda"),
        torch.empty(8, dtype=torch.uint8, device="cuda"),
    ):
        with pytest.raises(ValueError):
            cake_backend.qsa_sparse_paged_attention(
                **{**args, "logical_indices": one_tile, "workspace_buffer": buffer}
            )
    # The tripwire is live: a valid call reaches a launch and trips it.
    with pytest.raises(AssertionError, match="a launch was reached"):
        cake_backend.qsa_sparse_paged_attention(
            **{
                **args,
                "logical_indices": one_tile,
                "workspace_buffer": torch.zeros(
                    nbytes, dtype=torch.uint8, device="cuda"
                ),
            }
        )
    monkeypatch.undo()
    cake_backend.qsa_sparse_paged_attention(
        **{
            **args,
            "logical_indices": one_tile,
            "workspace_buffer": torch.zeros(nbytes, dtype=torch.uint8, device="cuda"),
        }
    )


# ---------------------------------------------------------------------------
# CPU tests: the host split rule
# ---------------------------------------------------------------------------


def _formula(rows, num_kv_heads, width, sm_count, arch):
    """The split formula alone: the most splits whose grid ``rows x kv_heads x splits``
    stays within the architecture's CTAs per SM, at least 1 and at most the ``ceil(width /
    64)`` tiles."""
    tiles = math.ceil(width / TILE)
    return max(1, min(tiles, CTAS_PER_SM[arch] * sm_count // (rows * num_kv_heads)))


def test_split_formula_fills_the_device_capacity():
    """``num_splits`` on an SM count the measured table does not name is the formula at the
    architecture's CTAs per SM: 80 SMs (160 CTAs) and 66 SMs (132 CTAs) on sm_80 over the
    shape list's rows, 148 SMs on sm_100a (148 CTAs: the B200, one CTA per SM) and on
    sm_103a, 212 SMs on sm_107a (an R200 board; 208 and 216 on its other board classes),
    the floor (more CTAs than the capacity), one tile, the merge kernel's bound, and the
    sizes it refuses."""
    assert [num_splits(r, 2, WIDTH, 80, "sm_80") for r in ROWS] == [
        33,
        33,
        20,
        10,
        5,
        2,
        1,
        1,
    ]
    assert [num_splits(r, 2, WIDTH, 66, "sm_80") for r in ROWS] == [
        33,
        33,
        16,
        8,
        4,
        2,
        1,
        1,
    ]
    assert [num_splits(r, 2, WIDTH, 148, "sm_100a") for r in ROWS] == [
        33,
        33,
        18,
        9,
        4,
        2,
        1,
        1,
    ]
    # sm_103a: the plain cut at rows 1-64; rows 128 (256 plain CTAs: two waves) take the
    # persistent walk and its 2 partial slots per (row, KV head) replace the formula's 1.
    assert [cake_backend.plain_splits(r, 2, WIDTH, 148, "sm_103a") for r in ROWS] == [
        33,
        33,
        18,
        9,
        4,
        2,
        1,
        1,
    ]
    assert [num_splits(r, 2, WIDTH, 148, "sm_103a") for r in ROWS] == [
        33,
        33,
        18,
        9,
        4,
        2,
        1,
        2,
    ]
    assert [num_splits(r, 2, WIDTH, 152, "sm_103a") for r in ROWS] == [
        33,
        33,
        19,
        9,
        4,
        2,
        1,
        2,
    ]
    # sm_107a: the plain cut at rows 1-64; rows 128 (256 plain CTAs: two waves on 212 and 216
    # SMs) take the persistent walk and its 2 partial slots replace the formula's 1.
    assert [cake_backend.plain_splits(r, 2, WIDTH, 212, "sm_107a") for r in ROWS] == [
        33,
        33,
        26,
        13,
        6,
        3,
        1,
        1,
    ]
    assert [num_splits(r, 2, WIDTH, 212, "sm_107a") for r in ROWS] == [
        33,
        33,
        26,
        13,
        6,
        3,
        1,
        2,
    ]
    assert [num_splits(r, 2, WIDTH, 216, "sm_107a") for r in ROWS] == [
        33,
        33,
        27,
        13,
        6,
        3,
        1,
        2,
    ]
    # sm_120a (the RTX PRO 6000 Blackwell, 188 SMs): the mma.sync decode's two CTAs per SM; the table's rows 4, 8,
    # 16, 32 and 64 (16, 11, 5, 3, 1) replace the formula's 33, 23, 11, 5, 2.
    assert [num_splits(r, 2, WIDTH, 188, "sm_120a") for r in ROWS] == [
        33,
        33,
        16,
        11,
        5,
        3,
        1,
        1,
    ]
    assert [_formula(r, 2, WIDTH, 188, "sm_120a") for r in ROWS] == [
        33,
        33,
        33,
        23,
        11,
        5,
        2,
        1,
    ]
    # At the same SM count the tcgen05 decode's capacity is half the mma.sync decode's.
    assert [num_splits(r, 2, WIDTH, 148, "sm_90a") for r in ROWS] == [
        33,
        33,
        33,
        18,
        9,
        4,
        2,
        1,
    ]
    for sm_count, arch in (
        (80, "sm_80"),
        (66, "sm_80"),
        (148, "sm_100a"),
        (148, "sm_103a"),
        (152, "sm_103a"),
        (212, "sm_107a"),
        (216, "sm_107a"),
        (190, "sm_120a"),
    ):
        assert [
            cake_backend.plain_splits(r, 2, WIDTH, sm_count, arch) for r in ROWS
        ] == [_formula(r, 2, WIDTH, sm_count, arch) for r in ROWS]
        if arch not in cake_backend.WALK_ARCHES:
            assert [num_splits(r, 2, WIDTH, sm_count, arch) for r in ROWS] == [
                _formula(r, 2, WIDTH, sm_count, arch) for r in ROWS
            ]
    assert num_splits(2048, 2, WIDTH, 132, "sm_90a") == 1
    assert num_splits(1, 1, TILE, 132, "sm_90a") == 1
    assert num_splits(1, 1, 200 * TILE, 10**6, "sm_90a") == MAX_SPLITS
    assert num_splits(1, 1, 200 * TILE, 10**6, "sm_100a") == MAX_SPLITS
    for rows, num_kv_heads, sm_count in ((0, 2, 132), (1, 0, 132), (1, 2, 0)):
        with pytest.raises(ValueError):
            num_splits(rows, num_kv_heads, WIDTH, sm_count, "sm_90a")
    with pytest.raises(ValueError, match="arch"):
        num_splits(1, 2, WIDTH, 132, "sm_121a")


def test_host_rule_applies_the_measured_table_within_the_caps():
    """``num_splits``: the formula at two CTAs per SM, replaced by the measured
    ``(sm_count, rows)`` entry where the table has one, never above the tile count or the
    merge kernel's 128 splits."""
    assert SPLIT_OVERRIDES == {
        (132, 4): 16,
        (132, 8): 8,
        (108, 64): 4,
        (108, 128): 2,
        (188, 4): 16,
        (188, 8): 11,
        (188, 16): 5,
        (188, 32): 3,
        (188, 64): 1,
    }
    assert all(
        isinstance(sm, int) and isinstance(rows, int) and 1 <= s <= MAX_SPLITS
        for (sm, rows), s in SPLIT_OVERRIDES.items()
    )
    table_132 = [33, 33, 16, 8, 8, 4, 2, 1]
    table_108 = [33, 33, 27, 13, 6, 3, 4, 2]
    table_188 = [33, 33, 16, 11, 5, 3, 1, 1]
    assert [num_splits(r, 2, WIDTH, 132, "sm_90a") for r in ROWS] == table_132
    assert [num_splits(r, 2, WIDTH, 108, "sm_80") for r in ROWS] == table_108
    assert [num_splits(r, 2, WIDTH, 188, "sm_120a") for r in ROWS] == table_188
    # Away from the table's rows the formula holds; at them the entry does.
    for sm_count, arch in ((132, "sm_90a"), (108, "sm_80"), (188, "sm_120a")):
        for rows in ROWS:
            formula = _formula(rows, 2, WIDTH, sm_count, arch)
            want = SPLIT_OVERRIDES.get((sm_count, rows), formula)
            assert num_splits(rows, 2, WIDTH, sm_count, arch) == want, (sm_count, rows)
    # No tcgen05 device's SM count is in the table: 148 and 152 SMs run the formula on every row.
    assert not any(sm_count in (148, 152) for sm_count, _rows in SPLIT_OVERRIDES)
    # The table never exceeds the tile count ...
    assert num_splits(4, 2, TILE, 132, "sm_90a") == 1
    assert num_splits(4, 2, 10 * TILE, 132, "sm_90a") == 10
    # ... and the formula never exceeds the merge kernel's bound.
    assert num_splits(1, 1, 200 * TILE, 1000, "sm_90a") == MAX_SPLITS


def test_form_rule_takes_the_fused_form_within_the_capacity_and_the_eager_rule(
    monkeypatch,
):
    """``select_form``: never above one split or the device's fused capacity. Within
    it, compute capability 9.0 takes the fused form on every host, eager or captured,
    and so does 12.0 below the tile stream (a call whose CTAs would each walk
    ``STREAM_TILES`` tiles takes the split form: the whole-tile program's);
    10.0 / 10.3 take it only on a host of ``FUSED_HOSTS``, at ``rows <= FUSED_MAX_ROWS``
    and outside capture -- and that list is empty, so they keep the split form on every
    host, eager or captured, and never read the capture state; 10.7 and 8.x are split
    alone.  GPU-free: the device facts are faked per device index."""
    facts = {
        800: ("sm_80", 108),
        900: ("sm_90a", 132),
        1000: ("sm_100a", 148),
        1030: ("sm_103a", 152),
        1070: ("sm_107a", 212),
        1200: ("sm_120a", 188),
    }
    tiles = cake_backend.num_tiles(WIDTH)  # 33: every call below is at the full width

    def form(rows, num_kv_heads, splits, index, capturing=False):
        return cake_backend.select_form(
            rows, num_kv_heads, splits, tiles, index, capturing
        )

    reads = cake_backend._reads_capture_state
    monkeypatch.setattr(cake_backend, "_device_facts", lambda index: facts[index])
    cake_backend.fused_capacity.cache_clear()
    reads.cache_clear()
    try:
        assert cake_backend.FUSED_HOST_RULE_ARCHES == ("sm_100a", "sm_103a")
        assert cake_backend.FUSED_SPLIT_ONLY_ARCHES == ("sm_107a",)
        assert cake_backend.FUSED_HOSTS == ()
        assert cake_backend.FUSED_MAX_ROWS == 4
        capacity = [
            cake_backend.fused_capacity(i) for i in (800, 900, 1000, 1030, 1070, 1200)
        ]
        assert capacity == [0, 264, 148, 152, 212, 376]
        for host in ("aarch64", "x86_64"):
            monkeypatch.setattr(cake_backend, "HOST_MACHINE", host)
            reads.cache_clear()
            # 10.0 / 10.3: the split form at every row on both hosts (no host is listed),
            # eager or captured; no capture-state read.
            for index, splits_at in (
                (1000, (33, 33, 18)),
                (1030, (33, 33, 19)),
            ):
                sm_count = facts[index][1]
                for rows, splits in zip((1, 2, 4), splits_at, strict=True):
                    assert rows * 2 * splits <= sm_count
                    assert form(rows, 2, splits, index) == "split", (host, index, rows)
                    assert form(rows, 2, splits, index, True) == "split"
                    assert reads(rows, index) is False
                # Above FUSED_MAX_ROWS (rows 8 at 9 splits fit the capacity), with one
                # split or above the capacity: split.
                assert form(8, 2, 9, index) == "split"
                assert form(16, 2, 4, index) == "split"
                assert form(4, 2, 1, index) == "split"
                assert form(4, 2, sm_count // 8 + 1, index) == "split"
                assert reads(8, index) is False
                assert reads(16, index) is False
            # 10.7: the split form at every row, eager or captured, on both hosts; no
            # capture-state read.
            for rows, splits in zip((1, 2, 4, 8), (33, 33, 26, 13), strict=True):
                assert rows * 2 * splits <= 212
                assert form(rows, 2, splits, 1070) == "split", (host, rows)
                assert form(rows, 2, splits, 1070, True) == "split"
                assert reads(rows, 1070) is False
            # Compute capability 9.0: the fused form within 264 CTAs, whatever the host
            # or the capture state; above it the split form.
            for capturing in (False, True):
                assert form(16, 2, 8, 900, capturing) == "fused"
                assert form(16, 2, 9, 900, capturing) == "split"
            assert reads(8, 900) is False
            # Compute capability 12.0: the fused form within 376 CTAs (two per SM on
            # 188 SMs) below the tile stream (33 tiles at 4 splits: 9 per CTA), whatever
            # the host or the capture state; above the capacity, or at the stream (33
            # tiles at 3 or 2 splits: 11 / 17 per CTA, the whole-tile program's calls),
            # the split form.
            assert cake_backend.STREAM_TILES == 11
            for capturing in (False, True):
                assert form(32, 2, 4, 1200, capturing) == "fused"
                assert form(48, 2, 4, 1200, capturing) == "split"
                assert form(32, 2, 3, 1200, capturing) == "split"
                assert form(64, 2, 2, 1200, capturing) == "split"
            assert reads(8, 1200) is False
            # Compute capability 8.x: the split form alone.
            assert form(1, 2, 33, 800) == "split"
            assert reads(1, 800) is False
    finally:
        cake_backend.fused_capacity.cache_clear()
        reads.cache_clear()


# ---------------------------------------------------------------------------
# CPU tests: the registry and the build.json contract of the package
# ---------------------------------------------------------------------------


def _manifest() -> dict:
    return json.loads((PACKAGE_DIR / "build.json").read_text())


def test_capabilities_route_to_the_six_libraries():
    assert cake_jit.ARCHES == (
        "sm_80",
        "sm_90a",
        "sm_100a",
        "sm_103a",
        "sm_107a",
        "sm_120a",
    )
    for capability in ((8, 0), (8, 6), (8, 7), (8, 9)):
        assert cake_jit.arch_for_capability(capability) == "sm_80"
    assert cake_jit.arch_for_capability((9, 0)) == "sm_90a"
    assert cake_jit.arch_for_capability((10, 0)) == "sm_100a"
    assert cake_jit.arch_for_capability((10, 3)) == "sm_103a"
    assert cake_jit.arch_for_capability((10, 7)) == "sm_107a"
    assert cake_jit.arch_for_capability((12, 0)) == "sm_120a"
    # Capability-exact: no library is delivered for 12.1.
    for capability in ((7, 5), (12, 1)):
        assert cake_jit.arch_for_capability(capability) is None
    assert set(cake_jit.ARCH_BY_CAPABILITY.values()) == set(cake_jit.ARCHES)
    # The per-architecture launch-bound tables cover every architecture; the fused table every one but sm_80, with
    # the co-resident build's one CTA per SM on the tcgen05 architectures (the eager host rule keeps the split form there).
    assert cake_jit.DECODE_THREADS == {
        "sm_80": 128,
        "sm_90a": 128,
        "sm_100a": 512,
        "sm_103a": 512,
        "sm_107a": 512,
        "sm_120a": 128,
    }
    assert cake_jit.DECODE_CTAS_PER_SM == CTAS_PER_SM
    assert cake_jit.DECODE_FUSED_CTAS_PER_SM == {
        "sm_90a": 2,
        "sm_100a": 1,
        "sm_103a": 1,
        "sm_107a": 1,
        "sm_120a": 2,
    }
    assert cake_backend.FUSED_CTAS_PER_SM == cake_jit.DECODE_FUSED_CTAS_PER_SM


def test_registry_records_carry_the_launch_facts_and_the_argument_plans():
    """Seven programs: the mma.sync plain decode, its whole-tile build delivered for sm_80
    alone (the split form on a multi-wave, long-stream grid), the mma.sync cooperative
    decode that merges its splits in the launch, the tcgen05 plain and cooperative decodes
    of sm_100a / sm_103a / sm_107a, the tcgen05 persistent walk delivered for sm_103a and sm_107a
    (the split form on a two-wave grid, one CTA per SM), and the separate merge; 128
    threads for the mma.sync programs and the merge, 512 for the tcgen05 ones, no cluster;
    the decodes' shared-memory pools as dynamic shared memory, the merge's static; the
    argument plan names every kernel parameter in order and then the grid (the six decodes
    take the same parameters)."""
    assert set(cake_jit.MODULES) == {
        "decode",
        "decode_whole_tile",
        "decode_coop",
        "decode_tcgen05",
        "decode_tcgen05_coop",
        "decode_tcgen05_persistent",
        "merge",
    }
    arches = {
        "decode": ("sm_80", "sm_90a", "sm_120a"),
        "decode_whole_tile": ("sm_80", "sm_120a"),
        "decode_coop": ("sm_80", "sm_90a", "sm_120a"),
        "decode_tcgen05": ("sm_100a", "sm_103a", "sm_107a"),
        "decode_tcgen05_coop": ("sm_100a", "sm_103a", "sm_107a"),
        "decode_tcgen05_persistent": ("sm_103a", "sm_107a"),
        "merge": ("sm_80", "sm_90a", "sm_100a", "sm_103a", "sm_107a", "sm_120a"),
    }
    tcgen05 = ("decode_tcgen05", "decode_tcgen05_coop", "decode_tcgen05_persistent")
    assert cake_jit.SELECT_RULES == ("form", "form_grid")
    assert cake_jit.GRID_BOUNDS == ("multi_wave_long", "long_stream", "two_wave_walk")
    persistent = cake_jit.MODULES["decode_tcgen05_persistent"]
    assert persistent["stage"] == "decode"
    assert persistent["variant"] == "tcgen05_persistent"
    assert persistent["select_rule"] == {
        "kind": "form_grid",
        "value": "split",
        "grid": {"sm_103a": "two_wave_walk", "sm_107a": "two_wave_walk"},
    }
    assert list(persistent["select_rule"]["grid"]) == persistent["arches"]
    assert persistent["grid_rule"] == {"kind": "walk_ctas"}
    assert (
        persistent["workspace_rule"]
        == cake_jit.MODULES["decode_tcgen05"]["workspace_rule"]
    )
    whole_tile = cake_jit.MODULES["decode_whole_tile"]
    assert whole_tile["stage"] == "decode" and whole_tile["variant"] == "whole_tile"
    assert whole_tile["select_rule"] == {
        "kind": "form_grid",
        "value": "split",
        "grid": {"sm_80": "multi_wave_long", "sm_120a": "long_stream"},
    }
    assert list(whole_tile["select_rule"]["grid"]) == whole_tile["arches"]
    for key in ("grid_rule", "workspace_rule", "launch", "arg_plan", "logical"):
        assert whole_tile[key] == cake_jit.MODULES["decode"][key], key
    for name in tcgen05:
        for key in ("grid_rule", "arg_plan", "logical"):
            if key == "grid_rule" and name == "decode_tcgen05_persistent":
                continue  # the walk's grid is its CTA count (asserted above)
            assert cake_jit.MODULES[name][key] == cake_jit.MODULES["decode"][key], (
                name,
                key,
            )
    for name, record in cake_jit.MODULES.items():
        assert re.fullmatch(r"cake_qsa_nvfp4_[0-9a-f]{20}", record["library"]), name
        assert tuple(record["arches"]) == arches[name], name
        assert record["ffi_entry"] == "run"
        assert record["kernel_symbol"] == "kernel_" + record["library"]
        assert isinstance(record["stage"], str) and record["stage"]
        assert record["select_rule"]["kind"] in cake_jit.SELECT_RULES
        assert record["select_rule"]["value"] in cake_jit.FORMS
        assert isinstance(record["grid_rule"]["kind"], str)
        launch = record["launch"]
        assert tuple(launch["block"]) == (
            (512, 1, 1) if name in tcgen05 else (128, 1, 1)
        ), name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["dynamic_smem_bytes"] == (
            0
            if name == "merge"
            else DECODE_TCGEN05_PERSISTENT_SMEM_BYTES
            if name == "decode_tcgen05_persistent"
            else DECODE_TCGEN05_SMEM_BYTES
            if name in tcgen05
            else DECODE_SMEM_BYTES
        ), name
        assert launch["use_pdl"] is False, name
        assert launch["cooperative"] is (
            name in ("decode_coop", "decode_tcgen05_coop")
        ), name
        parameters = MERGE_PARAMETERS if name == "merge" else DECODE_PARAMETERS
        plan = record["arg_plan"]
        assert [key for kind, key in plan if kind != "grid"] == list(parameters), name
        assert [key for kind, key in plan if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        kinds = {key: kind for kind, key in plan}
        for key in parameters:
            if key in ("partial_out", "partial_lse", "merge_counters"):
                assert kinds[key] == "buffer", (name, key)  # nullable pointers
            elif key == "page_div":
                assert kinds[key] == "parameter", name  # the page size as a divisor
        assert record["logical"] == {key: key for key in parameters}, name


def test_logical_index_orders_the_flat_argument_sequence():
    """The host's fast path binds ``LOGICAL_TENSORS`` then ``LOGICAL_SCALARS`` in one flat
    sequence (``LOGICAL_INDEX``); every record's plan names its kernel arguments from that
    vocabulary and ends with the three grid arguments."""
    names = cake_jit.LOGICAL_TENSORS + cake_jit.LOGICAL_SCALARS
    assert list(cake_jit.LOGICAL_INDEX) == list(names)
    assert [cake_jit.LOGICAL_INDEX[name] for name in names] == list(range(len(names)))
    for name, record in cake_jit.MODULES.items():
        plan = record["arg_plan"]
        assert [kind for kind, _ in plan[-3:]] == ["grid"] * 3, name
        assert all(key in cake_jit.LOGICAL_INDEX for _, key in plan[:-3]), name


def test_manifest_holds_one_entry_per_program_and_architecture():
    manifest = _manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == cake_jit.__name__.rsplit(".", 1)[0]
    assert manifest["arch_flags"] == {
        "sm_80": ["-gencode=arch=compute_80,code=sm_80"],
        "sm_90a": ["-gencode=arch=compute_90a,code=sm_90a"],
        "sm_100a": ["-gencode=arch=compute_100a,code=sm_100a"],
        "sm_103a": ["-gencode=arch=compute_103a,code=sm_103a"],
        "sm_107a": ["-gencode=arch=compute_107a,code=sm_107a"],
        "sm_120a": ["-gencode=arch=compute_120a,code=sm_120a"],
    }
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert len(entries) == len(manifest["libraries"])
    expected = {
        cake_jit.library_name(name, arch)
        for name, record in cake_jit.MODULES.items()
        for arch in record["arches"]
    }
    assert set(entries) == expected
    assert len(entries) == sum(
        len(record["arches"]) for record in cake_jit.MODULES.values()
    )
    for name, record in cake_jit.MODULES.items():
        for arch in record["arches"]:
            entry = entries[cake_jit.library_name(name, arch)]
            assert entry["name"] == f"{record['library']}_{arch}"
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["path"] == f"aot/{entry['name']}.so"
            kernel_tu, binding_tu = entry["sources"]
            assert kernel_tu == f"csrc/{record['library']}_kernel.cu"
            assert binding_tu == f"csrc/{record['library']}_binding.cu"
            assert set(entry["sha256"]) == set(entry["sources"])
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert all(isinstance(flag, str) for flag in entry["nvcc_flags"])
            # The kernels' numerics flag; nvcc's default fmad=true is not restated.
            assert "--use_fast_math" in entry["nvcc_flags"]
            assert not any(flag.startswith("--fmad") for flag in entry["nvcc_flags"])
            assert re.fullmatch(r"[0-9]+\.[0-9]+", entry["min_cuda"])


def test_manifest_digests_are_those_of_the_delivered_sources():
    for entry in _manifest()["libraries"]:
        for source in entry["sources"]:
            data = (PACKAGE_DIR / source).read_bytes()
            assert entry["sha256"][source] == hashlib.sha256(data).hexdigest(), (
                entry["name"],
                source,
            )


def test_delivered_sources_carry_the_license_header_and_the_symbols():
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    for record in cake_jit.MODULES.values():
        kernel = (PACKAGE_DIR / "csrc" / f"{record['library']}_kernel.cu").read_text()
        binding = (PACKAGE_DIR / "csrc" / f"{record['library']}_binding.cu").read_text()
        for text in (kernel, binding):
            assert re.search(r"Copyright \(c\) [0-9]{4} by FlashInfer team", text[:800])
        # The unit defines the kernel (a parameter list followed by a body); the binding
        # only declares it.
        symbol = re.escape(record["kernel_symbol"])
        definition = rf"__global__[^;{{}}]*?\b{symbol}\s*\([^;{{}}]*\)\s*\{{"
        assert re.search(definition, kernel), record["kernel_symbol"]
        assert record["kernel_symbol"] in binding
        assert re.search(r"TVM_FFI_DLL_EXPORT_TYPED_FUNC\(\s*run\s*,", binding)
        assert "build_cuda_version" in binding


def test_registry_and_host_pin_no_digest_literal():
    """The digests live in ``build.json`` alone; the registry and the host carry none, so a
    re-export renames nothing by hand."""
    for name in ("cake_jit.py", "cake_backend.py", "__init__.py"):
        text = (PACKAGE_DIR / name).read_text()
        assert re.search(r"\b[0-9a-f]{64}\b", text) is None, name


# ---------------------------------------------------------------------------
# Last on purpose: against a host that skips the zeroing, the launch below faults and the
# CUDA context keeps the error for the rest of the process.
# ---------------------------------------------------------------------------


def test_workspace_at_a_recycled_address_is_zeroed_again():
    """A new workspace the allocator places at the address of a freed one holds whatever
    that address held in between (here the FP32 partials of another call); the host must
    zero its counters as for any buffer it has not seen -- a buffer is new when its storage
    is new, not when its address is.  The fused decode faults on counters that are not
    zero, so this is the suite's last test."""
    device = _device_or_skip()
    rows, num_qo_heads, num_kv_heads = 8, 24, 2
    splits = _splits(rows, num_kv_heads, device)
    if (
        cake_backend.select_form(
            rows, num_kv_heads, splits, cake_backend.num_tiles(WIDTH), device.index
        )
        != "fused"
    ):
        pytest.skip("the split form keeps no counters in the workspace")
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, "NHD", 31)
    nbytes = cake_backend.qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits)
    expected = _run(q, cache, selection, gate)
    out = torch.full_like(q, math.nan)
    # Zeroed: an earlier buffer of this process may have had this address.  The hazard is
    # staged below, where the address is recycled with garbage in it.
    first = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    address = first.data_ptr()
    _run(q, cache, selection, gate, out, workspace_buffer=first)
    assert torch.equal(out, expected)
    del first
    scratch = torch.empty(nbytes, dtype=torch.uint8, device=device)
    scratch.view(torch.float32).fill_(-6.0)
    del scratch
    second = torch.empty(nbytes, dtype=torch.uint8, device=device)
    if second.data_ptr() != address:
        pytest.skip(
            "the allocator did not place the new workspace at the freed address"
        )
    torch.cuda.synchronize()
    assert second[:4].view(torch.float32).item() == -6.0  # the counters hold garbage
    out.fill_(math.nan)
    _run(q, cache, selection, gate, out, workspace_buffer=second)
    assert torch.equal(out, expected)


def test_threads_sharing_a_caller_workspace_raise_nothing():
    """Threads that pass one workspace, while another thread calls ``reset_workspace``
    on it, raise nothing and get bitwise-equal outputs: every change to the plan and
    sighting tables is one dictionary operation that tolerates an entry another thread
    removed.  Calls above one split are serialized by a lock, as a shared stream asks
    of its callers (one call's decode and merge launches must not interleave with
    another's partials); the table changes and the resets are not serialized.  Calls
    at one split launch without partials and run unserialized.  A tiny switch interval
    makes the interpreter change threads between the table operations."""
    device = _device_or_skip()
    num_qo_heads, num_kv_heads, threads, calls = 24, 2, 4, 12
    shapes = [(8, _splits(8, num_kv_heads, device))]
    if _splits(64, num_kv_heads, device) == 1:
        shapes.append((64, 1))
    nbytes = max(
        cake_backend.qsa_workspace_bytes(rows, num_qo_heads, num_kv_heads, splits)
        for rows, splits in shapes
    )
    workspace = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    cake_backend.reset_workspace(workspace)  # no plan or sighting from another test
    cases = []
    for seed, (rows, splits) in enumerate(shapes, start=41):
        q, cache, selection, gate = _inputs(
            rows, num_qo_heads, num_kv_heads, "HND", seed
        )
        cases.append(
            (splits, q, cache, selection, gate, _run(q, cache, selection, gate))
        )
    lock = threading.Lock()
    done = threading.Event()
    errors: list[BaseException] = []
    mismatches = [0]

    def caller():
        try:
            for splits, q, cache, selection, gate, expected in cases:
                out = torch.full_like(q, math.nan)
                for _ in range(calls):
                    out.fill_(math.nan)
                    if splits == 1:
                        _run(q, cache, selection, gate, out, workspace_buffer=workspace)
                    else:
                        with lock:
                            _run(
                                q,
                                cache,
                                selection,
                                gate,
                                out,
                                workspace_buffer=workspace,
                            )
                    if not torch.equal(out, expected):
                        mismatches[0] += 1
        except BaseException as exc:  # every failure is reported below
            errors.append(exc)

    def resetter():
        try:
            while not done.is_set():
                cake_backend.reset_workspace(workspace)
        except BaseException as exc:
            errors.append(exc)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        workers = [threading.Thread(target=caller) for _ in range(threads)]
        reset_thread = threading.Thread(target=resetter)
        reset_thread.start()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        done.set()
        reset_thread.join()
    finally:
        sys.setswitchinterval(interval)
    assert not errors, errors
    assert mismatches[0] == 0
    assert len(cake_backend._seen_once) <= cake_backend._SEEN_ONCE_MAX
    assert len(cake_backend._workspace_plans) <= cake_backend._WORKSPACE_PLANS_MAX
    cake_backend.reset_workspace(workspace)
