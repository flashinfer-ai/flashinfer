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
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.experimental.qsa_nvfp4.backend import _num_splits
from flashinfer.sparse import qsa_sparse_paged_attention
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] not in (8, 9),
    reason="QSA NVFP4 attention runs on SM80 through SM90",
)

HEAD_DIM = 256
PAGE_SIZE = 16
WIDTH = 2051  # Qwen QSA selection width: budget 2048 + compress ratio 4 - 1
NUM_REQUESTS, TABLE_WIDTH = 3, 160
NUM_BLOCKS = NUM_REQUESTS * TABLE_WIDTH + 20
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _dequant(data, sf, scale):
    lut = torch.tensor(_E2M1 + [-x for x in _E2M1], device=data.device)
    values = torch.stack((lut[(data & 15).long()], lut[(data >> 4).long()]), -1)
    return values.flatten(-2) * sf.float().repeat_interleave(16, -1) * scale


def _make_cache(num_kv_heads, layout, seed):
    """Quantize random K/V into every slot with the NVFP4 writer, using one buffer of
    [data | scales] rows per side like a packed vLLM page. Returns the kernel's
    [blocks, tokens, heads, *] views, the global scales and the dequantized cache."""
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
    flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
        k, v, torch.arange(slots, device="cuda"), data, sf, *scales, kv_layout=layout
    )
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


def _run(q, cache, selection, gate, out=None):
    data, sf, (k_scale, v_scale), _ = cache
    table, token_to_req, indices = selection
    return qsa_sparse_paged_attention(
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
        backend="default",
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


# rows x kv_heads of 2, 16, 32, 128 and 4096 give 33, 8, 4, 1 and 1 splits; groups 12, 16, 1.
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "rows,num_qo_heads,num_kv_heads",
    [
        (1, 24, 2),
        (8, 24, 2),
        (16, 24, 2),
        (64, 24, 2),
        (2048, 24, 2),
        (4, 16, 1),
        (4, 2, 2),
    ],
)
def test_matches_reference(layout, rows, num_qo_heads, num_kv_heads):
    q, cache, selection, gate = _inputs(rows, num_qo_heads, num_kv_heads, layout, rows)
    out = _run(q, cache, selection, gate).double()
    major = get_compute_capability(q.device)[0]
    num_splits = _num_splits(rows, num_kv_heads, WIDTH, major)
    contract, tolerance, exact, bound = _reference(
        q, cache, selection, gate, num_splits
    )
    assert ((out - contract).abs() <= tolerance).all()
    # Accuracy limit against exact attention; validated empirically, not a proven bound.
    assert ((out - exact).abs() / bound.clamp(min=1e-12)).max().item() <= 2**-6
    if rows > 2:
        assert not out[1].any() and out[2].any()  # count 0; count 1 with a real token


def test_cuda_graph_replay():
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


def test_rejects_invalid_inputs():
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
    ):
        with pytest.raises(ValueError):
            qsa_sparse_paged_attention(**{**args, **change}, backend="default")
