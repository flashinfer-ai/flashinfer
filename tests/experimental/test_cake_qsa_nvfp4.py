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
queries over an NVFP4 paged KV cache, for SM80 through SM90.

The reference and the fixtures are those of ``test_qsa_nvfp4.py``, the suite of the
existing implementation: a float64 model of the kernel's arithmetic with its elementwise
tolerance, the ``2^-6`` accuracy limit against exact attention, count-0 / count-1 rows,
unmapped pages, blocks past the cache, rows without a request, NHD and HND caches and the
group sizes 12, 16 and 1.  The split count the reference models is the package's host
split rule (``num_splits``, decided by the device's SM count).  This suite adds the
host rule itself (formula, measured table, tile and merge caps), a 33-split case, bitwise
repeatability, CUDA-graph replay with and without a caller-owned workspace, the
zero-allocation workspace path, and the ``build.json`` contract of the package.

The suite calls the package's host entry directly; the public route
``flashinfer.sparse.qsa_sparse_paged_attention(..., backend="cake")`` is covered by
``test_cake_qsa_nvfp4_api.py``.  Besides the package it imports nothing of flashinfer: the
cache the kernel reads is filled by ``tests.test_helpers.cake_qsa_nvfp4_reference``, a torch
port of FlashInfer's NVFP4 paged writer (the API file compares the two).
"""

import hashlib
import json
import math
import re
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
# The decode's launch bound: the split formula's capacity is this many CTAs per SM.
CTAS_PER_SM = 2
# The decode's shared-memory pool, passed as dynamic shared memory by its binding; two
# CTAs of it fit an SM (the launch bound).  The merge uses static shared memory alone.
DECODE_SMEM_BYTES = 49792
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
    """The split count the package passes for ``rows`` on ``device``: its host split rule."""
    return num_splits(rows, num_kv_heads, WIDTH, _sm_count(device))


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


def _reference(q, cache, selection, gate, num_splits, chunk=64):
    """Float64 references, row chunk by row chunk.

    contract: the kernel's arithmetic -- 64-column tiles walked per split with an online
    softmax, P rounded to BF16, LSE merge, BF16 rounding, FP32 gate, BF16 output -- and a
    worst-case bound on how far one-ulp P rounding, FP32 scores (with exp2.approx), FP32
    accumulation and the two output roundings can move the kernel from it.
    exact: softmax attention times the gate, unrounded, and sum_n p_n |v_n| times the gate.
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
        splits = []
        for split in range(num_splits):
            m = torch.full(s.shape[:-1], -1e20, dtype=s.dtype, device=s.device)
            l, acc, flip = torch.zeros_like(m), 0.0, 0.0
            for tile in range(split, num_tiles, num_splits):
                cols = slice(64 * tile, 64 * tile + 64)
                m_new = torch.maximum(m, s[..., cols].amax(-1))
                pt = torch.exp2(s[..., cols] - m_new[..., None]) * mask[..., cols]
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


def _check_contract(out, q, cache, selection, gate, splits):
    contract, tolerance, exact, bound = _reference(q, cache, selection, gate, splits)
    assert ((out - contract).abs() <= tolerance).all()
    # Accuracy limit against exact attention; validated empirically, not a proven bound.
    assert ((out - exact).abs() / bound.clamp(min=1e-12)).max().item() <= 2**-6


# ---------------------------------------------------------------------------
# Device tests: the contract reference, the forms the host rule selects, replay, workspace
# ---------------------------------------------------------------------------


# The eight row counts of the owner's shape list at 24 x 2 heads, the 2048-row case (one
# split, more CTAs than the device holds at once) and the group sizes 16 and 1.  On 132 SMs
# the host rule gives 33, 33, 16, 8, 8, 4, 2, 1 splits for rows 1 .. 128; on 108 SMs 33, 33,
# 27, 13, 6, 3, 4 and 2 (rows 64 and 128 then run the separate merge).
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
    _check_contract(out, q, cache, selection, gate, _splits(rows, num_kv_heads, device))
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


@pytest.mark.parametrize("rows", [1, 8, 64])
def test_repeated_calls_are_bitwise_equal(rows):
    """The same inputs give the same bits call after call, through fresh per-call scratch
    and through one caller-owned workspace launched three times in a row (the group
    counters of the in-launch merge alternate generations and end where they started)."""
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


def test_zero_rows_return_the_output_untouched():
    _device_or_skip()
    q, cache, selection, gate = _inputs(4, 24, 2, "NHD", 0)
    table, token_to_req, indices = selection
    out = torch.empty(0, 24, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    result = _run(q[:0], cache, (table, token_to_req[:0], indices[:0]), gate[:0], out)
    assert result is out


def test_rejects_invalid_inputs():
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
    assert num_splits(4, 2, TILE, _sm_count(q.device)) == 1
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


def _formula(rows, num_kv_heads, width, sm_count):
    """The split formula alone: the most splits whose grid ``rows x kv_heads x splits``
    stays within two CTAs per SM, at least 1 and at most the ``ceil(width / 64)`` tiles."""
    tiles = math.ceil(width / TILE)
    return max(1, min(tiles, CTAS_PER_SM * sm_count // (rows * num_kv_heads)))


def test_split_formula_fills_the_device_capacity():
    """``num_splits`` on an SM count the measured table does not name is the formula:
    80 SMs (160 CTAs) and 66 SMs (132 CTAs) over the shape list's rows, the floor (more
    CTAs than the capacity), one tile, the merge kernel's bound, and the sizes it refuses."""
    assert [num_splits(r, 2, WIDTH, 80) for r in ROWS] == [33, 33, 20, 10, 5, 2, 1, 1]
    assert [num_splits(r, 2, WIDTH, 66) for r in ROWS] == [33, 33, 16, 8, 4, 2, 1, 1]
    for sm_count in (80, 66):
        assert [num_splits(r, 2, WIDTH, sm_count) for r in ROWS] == [
            _formula(r, 2, WIDTH, sm_count) for r in ROWS
        ]
    assert num_splits(2048, 2, WIDTH, 132) == 1
    assert num_splits(1, 1, TILE, 132) == 1
    assert num_splits(1, 1, 200 * TILE, 10**6) == MAX_SPLITS
    for rows, num_kv_heads, sm_count in ((0, 2, 132), (1, 0, 132), (1, 2, 0)):
        with pytest.raises(ValueError):
            num_splits(rows, num_kv_heads, WIDTH, sm_count)


def test_host_rule_applies_the_measured_table_within_the_caps():
    """``num_splits``: the formula at two CTAs per SM, replaced by the measured
    ``(sm_count, rows)`` entry where the table has one, never above the tile count or the
    merge kernel's 128 splits."""
    assert SPLIT_OVERRIDES == {(132, 4): 16, (132, 8): 8, (108, 64): 4, (108, 128): 2}
    assert all(
        isinstance(sm, int) and isinstance(rows, int) and 1 <= s <= MAX_SPLITS
        for (sm, rows), s in SPLIT_OVERRIDES.items()
    )
    table_132 = [33, 33, 16, 8, 8, 4, 2, 1]
    table_108 = [33, 33, 27, 13, 6, 3, 4, 2]
    assert [num_splits(r, 2, WIDTH, 132) for r in ROWS] == table_132
    assert [num_splits(r, 2, WIDTH, 108) for r in ROWS] == table_108
    # Away from the table's rows the formula holds; at them the entry does.
    for sm_count in (132, 108):
        for rows in ROWS:
            formula = _formula(rows, 2, WIDTH, sm_count)
            want = SPLIT_OVERRIDES.get((sm_count, rows), formula)
            assert num_splits(rows, 2, WIDTH, sm_count) == want, (sm_count, rows)
    # The table never exceeds the tile count ...
    assert num_splits(4, 2, TILE, 132) == 1
    assert num_splits(4, 2, 10 * TILE, 132) == 10
    # ... and the formula never exceeds the merge kernel's bound.
    assert num_splits(1, 1, 200 * TILE, 1000) == MAX_SPLITS


# ---------------------------------------------------------------------------
# CPU tests: the registry and the build.json contract of the package
# ---------------------------------------------------------------------------


def _manifest() -> dict:
    return json.loads((PACKAGE_DIR / "build.json").read_text())


def test_capabilities_route_to_the_two_libraries():
    assert cake_jit.ARCHES == ("sm_80", "sm_90a")
    for capability in ((8, 0), (8, 6), (8, 7), (8, 9)):
        assert cake_jit.arch_for_capability(capability) == "sm_80"
    assert cake_jit.arch_for_capability((9, 0)) == "sm_90a"
    assert cake_jit.arch_for_capability((7, 5)) is None
    assert cake_jit.arch_for_capability((10, 0)) is None
    assert set(cake_jit.ARCH_BY_CAPABILITY.values()) == set(cake_jit.ARCHES)


def test_registry_records_carry_the_launch_facts_and_the_argument_plans():
    """Three programs: the plain decode, the cooperative decode that merges its splits in
    the launch, and the separate merge; 128 threads each, no cluster; the decode's
    shared-memory pool as dynamic shared memory, the merge's static; the argument plan
    names every kernel parameter in order and then the grid."""
    assert set(cake_jit.MODULES) == {"decode", "decode_coop", "merge"}
    for name, record in cake_jit.MODULES.items():
        assert re.fullmatch(r"cake_qsa_nvfp4_[0-9a-f]{20}", record["library"]), name
        assert tuple(record["arches"]) == cake_jit.ARCHES, name
        assert record["ffi_entry"] == "run"
        assert record["kernel_symbol"] == "kernel_" + record["library"]
        assert isinstance(record["stage"], str) and record["stage"]
        assert isinstance(record["select_rule"]["kind"], str)
        assert isinstance(record["grid_rule"]["kind"], str)
        launch = record["launch"]
        assert tuple(launch["block"]) == (128, 1, 1), name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["dynamic_smem_bytes"] == (
            0 if name == "merge" else DECODE_SMEM_BYTES
        ), name
        assert launch["use_pdl"] is False, name
        assert launch["cooperative"] is (name == "decode_coop"), name
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


def test_manifest_holds_one_entry_per_program_and_architecture():
    manifest = _manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == cake_jit.__name__.rsplit(".", 1)[0]
    assert manifest["arch_flags"] == {
        "sm_80": ["-gencode=arch=compute_80,code=sm_80"],
        "sm_90a": ["-gencode=arch=compute_90a,code=sm_90a"],
    }
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert len(entries) == len(manifest["libraries"])
    expected = {
        cake_jit.library_name(name, arch)
        for name, record in cake_jit.MODULES.items()
        for arch in record["arches"]
    }
    assert set(entries) == expected
    assert len(entries) == len(cake_jit.MODULES) * len(cake_jit.ARCHES)
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
    if cake_backend.select_form(rows, num_kv_heads, splits, device.index) != "fused":
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
