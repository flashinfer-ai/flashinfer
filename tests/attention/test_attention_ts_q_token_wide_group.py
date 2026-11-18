# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""QToken-KvBlock-Sparse-Attention G16/G32 (two/four page-major membership planes)."""

import pytest
import torch

from flashinfer.attention.prims_ts import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
)
from flashinfer.attention.prims_ts.decode import (
    make_q_token_kv_block_sparse_qo_indptr,
    suggest_q_token_kv_block_sparse_group_size,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="QToken-KvBlock-Sparse-Attention tests require SM100, SM103 or SM107",
)

BLOCK, TOPK, PAGE = 4, 64, 64
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize(x, d):
    blocks = x.float().reshape(*x.shape[:-1], d // 16, 16)
    scales = (blocks.abs().amax(-1) / 6.0).clamp_min(2.0**-9).to(torch.float8_e4m3fn)
    y = blocks / scales.float().unsqueeze(-1)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(y.abs(), mids).to(torch.uint8) | (
        (y < 0).to(torch.uint8) << 3
    )
    code = code.reshape(*x.shape[:-1], d // 2, 2)
    return (code[..., 0] | (code[..., 1] << 4)).contiguous(), scales.reshape(
        *x.shape[:-1], d // 16
    )


def _dequantize(payload, scales, d):
    lut = torch.tensor(E2M1 + tuple(-v for v in E2M1), device=payload.device)
    codes = torch.stack((payload & 15, payload >> 4), -1).reshape(
        *payload.shape[:-1], d
    )
    return lut[codes.long()] * scales.float().repeat_interleave(16, -1)


def _v_interleave(sf):
    p, h, n, _ = sf.shape
    return (
        sf.reshape(p, h, n // 4, 4, 4, 4)
        .permute(0, 1, 2, 4, 5, 3)
        .reshape(p, h, n, 16)
        .contiguous()
    )


@pytest.mark.parametrize("split_kv", [True, False])
@pytest.mark.parametrize(
    ("kv_kind", "head_dim"),
    [("bf16", 128), ("bf16", 256), ("nvfp4", 256)],
)
@pytest.mark.parametrize(
    ("hq", "hkv", "group"),
    [(4, 1, 16), (4, 1, 32), (16, 2, 16), (2, 1, 32), (1, 1, 16)],
)
def test_wide_group_matches_reference(hq, hkv, group, kv_kind, head_dim, split_kv):
    """Every query slot of G16/G32 routes, incl. a short tail route, matches FP32."""
    torch.manual_seed(group * 131 + hq)
    dev = torch.device("cuda")
    d = head_dim
    prefix, m = 1100, 3 * group + 5
    length = prefix + m
    pages = (length + PAGE - 1) // PAGE
    table = torch.randperm(pages, device=dev, dtype=torch.int32).view(1, pages)
    dense = [
        torch.randn(pages, hkv, PAGE, d, device=dev).to(torch.bfloat16)
        for _ in range(2)
    ]
    if kv_kind == "bf16":
        kv, sf = tuple(dense), None
        k_ref, v_ref = (x.float() for x in dense)
    else:
        caches = [_quantize(x, d) for x in dense]
        last = int(table[0, -1])
        for payload, scales in caches:
            payload[last, :, length % PAGE or PAGE :] = 0
            scales[last, :, length % PAGE or PAGE :] = torch.tensor(
                0.0, dtype=torch.float8_e4m3fn
            )
        kv = (caches[0][0], caches[1][0])
        sf = (caches[0][1], _v_interleave(caches[1][1]))
        k_ref, v_ref = (_dequantize(p, s, d) for p, s in caches)
    pos = torch.arange(prefix, length, device=dev, dtype=torch.int32)
    # Half of each selection from a pool shared by the route, half private.
    ids = torch.full((m, TOPK), -1, device=dev, dtype=torch.int32)
    pool = torch.randperm(prefix // BLOCK, device=dev)[: 2 * TOPK]
    for t in range(m):
        bound = int(pos[t] + 1) // BLOCK
        shared = pool[torch.randperm(pool.numel(), device=dev)[: TOPK // 2]]
        rest = torch.randperm(bound, device=dev)
        rest = rest[~torch.isin(rest, shared)][: TOPK - shared.numel()]
        sel = torch.cat([shared, rest])
        ids[t, : sel.numel()] = sel.to(torch.int32)
    q = torch.randn(m, hq, d, device=dev).to(torch.bfloat16)
    t2r = torch.zeros(m, device=dev, dtype=torch.int32)
    qo = make_q_token_kv_block_sparse_qo_indptr(
        torch.tensor([0, m], dtype=torch.int32), m, group_size=group, device=dev
    )
    ws = torch.zeros(
        get_q_token_kv_block_sparse_workspace_size(
            q,
            kv[0],
            table,
            block_topk=TOPK,
            max_seq_len_kv=length,
            out_dtype=torch.bfloat16,
            qo_indptr=qo,
            group_size=group,
            kv_block_size=BLOCK,
            split_kv=split_kv,
        ),
        dtype=torch.uint8,
        device=dev,
    )
    w = QTokenKvBlockSparsePagedTSWrapper()
    w.plan(
        qo.numel() - 1,
        hq,
        hkv,
        d,
        BLOCK,
        PAGE,
        TOPK,
        length,
        device=dev,
        workspace_buffer=ws,
        group_size=group,
        packed_query=True,
        split_kv=split_kv,
        q_dtype=torch.bfloat16,
        kv_dtype=kv[0].dtype,
        out_dtype=torch.bfloat16,
    )
    out = torch.empty_like(q)
    w.run(q, kv, table, ids, t2r, pos, qo_indptr=qo, out=out, kv_scale_factors=sf)
    torch.cuda.synchronize()

    k_all, v_all = (
        x[table[0].long()].permute(0, 2, 1, 3).reshape(pages * PAGE, hkv, d)
        for x in (k_ref, v_ref)
    )
    for t in range(m):
        p_ = int(pos[t])
        live = ids[t][ids[t] >= 0].long()
        toks = (live[:, None] * BLOCK + torch.arange(BLOCK, device=dev)).reshape(-1)
        tail = (p_ + 1) % BLOCK
        if tail:
            toks = torch.cat([toks, torch.arange(p_ + 1 - tail, p_ + 1, device=dev)])
        qq = q[t].float().view(hkv, hq // hkv, d)
        s = torch.einsum("hgd,nhd->hgn", qq, k_all[toks]) * d**-0.5
        ref = torch.einsum("hgn,nhd->hgd", s.softmax(-1), v_all[toks]).reshape(hq, d)
        err = (out[t].float() - ref).norm() / ref.norm()
        assert err < 0.01, (t, t % group, float(err))


def test_wide_group_suggestion_for_long_nvfp4_prefill():
    """Long NVFP4 prefill takes a two/four-plane group where it pays: 4/1 heads
    over a 64K context (G16) or from an empty prefix (G32); 16/2 heads over 64K
    keep G8."""
    kw = dict(nvfp4_kv=True)
    h4 = suggest_q_token_kv_block_sparse_group_size(
        1, 4096, 2051, 4, 1, 148, seq_len_kv=65536 + 4096, **kw
    )
    fresh = suggest_q_token_kv_block_sparse_group_size(
        1, 4096, 2051, 4, 1, 148, seq_len_kv=4096, **kw
    )
    h16 = suggest_q_token_kv_block_sparse_group_size(
        1, 2048, 2051, 16, 2, 148, seq_len_kv=65536 + 2048, **kw
    )
    short = suggest_q_token_kv_block_sparse_group_size(
        1, 4, 2051, 16, 2, 148, seq_len_kv=65536 + 4, **kw
    )
    assert (h4, fresh, h16) == (16, 32, 8) and short <= 4


def test_wide_group_suggestion_needs_small_blocks():
    """G16/G32 read four sparse blocks per membership word in each Keeps lane,
    so blocks wider than 16 tokens keep the suggestion at G8 or below."""
    for block in (32, 64):
        group = suggest_q_token_kv_block_sparse_group_size(
            1, 4096, 2048 + block - 1, 4, 1, 148, kv_block_size=block, nvfp4_kv=True
        )
        assert group <= 8
