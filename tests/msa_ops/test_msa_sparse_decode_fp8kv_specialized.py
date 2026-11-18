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

Tests for the SM103 "specialized" block-sparse MSA decode backend of
msa_sparse_decode_attention (bf16 Q over paged fp8 E4M3 KV): correctness vs
an fp32 reference, backend semantics (auto / specialized / cake, kill
switch, off-allowlist raise), out= handling, and CUDA-graph replay == eager.
"""

import os

import pytest
import torch

import flashinfer.msa_sparse_decode_fp8kv_cuda as spec_mod
from flashinfer.msa_ops import msa_sparse_decode_attention

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the specialized MSA fp8-KV decode kernel is SM103-only",
)

HQ, D, BLOCK, TOPK = 16, 128, 128, 16
SCALE = 0.08838834764831845  # 128 ** -0.5


def _make_inputs(B, SEQ, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    pages = (SEQ + BLOCK - 1) // BLOCK
    total = B * pages
    k = (torch.randn(total, 1, BLOCK, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn(total, 1, BLOCK, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    q = (torch.randn(B, HQ, D, generator=g, device=dev)).to(torch.bfloat16)
    topk = torch.full((1, B, TOPK), -1, dtype=torch.int32, device=dev)
    n_valid = min(TOPK, pages)
    if pages <= TOPK:
        topk[0, :, :n_valid] = torch.arange(pages, dtype=torch.int32, device=dev)
    else:
        picks = torch.argsort(torch.rand(B, pages - 1, generator=g, device=dev), dim=1)[
            :, : TOPK - 1
        ]
        last = torch.full((B, 1), pages - 1, dtype=torch.int64, device=dev)
        topk[0] = torch.cat([picks, last], 1).sort(1).values.to(torch.int32)
    gp = torch.Generator().manual_seed(seed + 7)
    pt = torch.randperm(total, generator=gp).to(torch.int32).to(dev).reshape(B, pages)
    seq_lens = torch.full((B,), SEQ, dtype=torch.int32, device=dev)
    return dict(q=q, k=k, v=v, topk=topk, pt=pt.contiguous(), seq_lens=seq_lens)


def _reference(inp):
    """fp32 block-sparse GQA decode: request b attends its selected blocks
    (ascending ids, -1 padded) gathered through the page table, masked to
    position < seq_lens[b]."""
    dev = inp["q"].device
    B = inp["q"].shape[0]
    kf = inp["k"].reshape(-1, BLOCK, D).to(torch.float32)
    vf = inp["v"].reshape(-1, BLOCK, D).to(torch.float32)
    blk = inp["topk"][0].to(torch.int64)
    blk_c = blk.clamp_min(0)
    page_ids = inp["pt"].to(torch.int64).gather(1, blk_c)
    K = kf[page_ids].reshape(B, TOPK * BLOCK, D)
    V = vf[page_ids].reshape(B, TOPK * BLOCK, D)
    pos = (blk_c.unsqueeze(-1) * BLOCK + torch.arange(BLOCK, device=dev)).reshape(
        B, TOPK * BLOCK
    )
    valid = (blk >= 0).unsqueeze(-1).expand(-1, -1, BLOCK).reshape(B, TOPK * BLOCK) & (
        pos < inp["seq_lens"].to(torch.int64).unsqueeze(1)
    )
    qf = inp["q"].to(torch.float32)
    logits = torch.einsum("bhd,bwd->bhw", qf, K) * SCALE
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("bhw,bwd->bhd", p, V).to(torch.bfloat16)


def _api(inp, out=None, backend="auto"):
    return msa_sparse_decode_attention(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["topk"],
        page_table=inp["pt"],
        seqused_k=inp["seq_lens"],
        seqlen_q=1,
        causal=True,
        softmax_scale=SCALE,
        out=out,
        backend=backend,
    )


def _assert_close(out, gold):
    diff = (out.float() - gold).abs()
    tol = 0.004 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) == 1.0, float(diff.max())


@requires_sm103
@pytest.mark.parametrize(
    "B,SEQ",
    [
        (1, 384),  # fewer visible blocks than TOPK (-1 padding live)
        (8, 8192),  # full top-16 selection
        (64, 384),  # large batch, small kv
        (4, 300),  # partial final page (position mask live)
        (8, 512),  # 4-page regime (general route family)
    ],
)
def test_specialized_correct_and_routed(B, SEQ):
    inp = _make_inputs(B, SEQ, seed=9)
    gold = _reference(inp).float()
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    _assert_close(out, gold)
    # explicit specialized == auto bitwise; out= is honoured and returned
    buf = torch.empty_like(inp["q"])
    out2 = _api(inp, out=buf, backend="specialized")
    assert out2 is buf
    assert torch.equal(out, out2)


@requires_sm103
def test_backend_semantics():
    inp = _make_inputs(4, 300, seed=1)
    gold = _reference(inp).float()
    # backend="cake" (the Cake-generated Blackwell programs, the former
    # arch-dispatch "native" label) never routes to the specialized kernel
    before = dict(spec_mod.DISPATCH_STATS)
    out = _api(inp, backend="cake")
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    _assert_close(out, gold)
    # explicit specialized raises off-allowlist (fp16 q) with reasons
    bad = dict(inp)
    bad["q"] = inp["q"].to(torch.float16)
    with pytest.raises(ValueError, match="fallback_reasons"):
        _api(bad, backend="specialized")
    with pytest.raises(ValueError, match="backend must be one of"):
        _api(inp, backend="bogus")


@requires_sm103
def test_kill_switch_restores_stock():
    inp = _make_inputs(4, 300, seed=2)
    gold = _reference(inp).float()
    os.environ[spec_mod.KILL_SWITCH_ENV] = "1"
    try:
        before = dict(spec_mod.DISPATCH_STATS)
        out = _api(inp)
        assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
        assert spec_mod.LAST_FALLBACK_REASONS[0].startswith("kill_switch")
        _assert_close(out, gold)
        with pytest.raises(ValueError, match="fallback_reasons"):
            _api(inp, backend="specialized")
    finally:
        os.environ.pop(spec_mod.KILL_SWITCH_ENV, None)


@requires_sm103
def test_cuda_graph_replay_matches_eager():
    inp = _make_inputs(8, 8192, seed=4)
    buf = torch.empty_like(inp["q"])
    _api(inp, out=buf)  # eager warmup builds the module before capture
    eager = buf.clone()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _api(inp, out=buf)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        _api(inp, out=buf)
    for _ in range(3):
        buf.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(buf, eager)


@requires_sm103
def test_backend_names_reflect_implementations():
    """Round-3 naming: no arbitrary labels. "cake"/"cutedsl" name real
    implementations; "native" is gone; "auto" only selects."""
    from flashinfer.msa_ops.sparse_decode import MSA_SPARSE_DECODE_BACKENDS

    assert MSA_SPARSE_DECODE_BACKENDS == ("auto", "specialized", "cake", "cutedsl")
    inp = _make_inputs(B=2, SEQ=256, seed=31)
    with pytest.raises(ValueError, match="backend must be one of"):
        _api(inp, backend="native")
    # explicit cutedsl on SM103 raises (it serves SM120/SM121 here)
    with pytest.raises(ValueError, match="cutedsl"):
        _api(inp, backend="cutedsl")
    # explicit cake on SM103 equals what auto selects off-allowlist and
    # the stock Blackwell path
    out_cake = _api(inp, backend="cake")
    gold = _reference(inp)
    diff = (out_cake.float() - gold.float()).abs()
    assert float((diff <= 0.02 + 0.05 * gold.float().abs()).float().mean()) >= 0.99


@requires_sm103
def test_empty_selection_rows_produce_zero_output():
    """A query row whose q2k_indices are all -1 selects nothing; the
    specialized route must return ZERO for that row (the packed backend's
    empty-effective-KV convention), never 0/0 = NaN."""
    inp = _make_inputs(4, 300, seed=47)
    inp["q2k_indices"][0, 2, :] = -1
    gold = _reference(inp)
    out = _api(inp, backend="specialized")
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    row = out.float()[2]
    assert float(row.abs().max()) == 0.0
    # the other rows stay reference-close
    keep = [i for i in range(out.shape[0]) if i != 2]
    diff = (out.float()[keep] - gold.float()[keep]).abs()
    assert (
        float((diff <= 0.02 + 0.05 * gold.float()[keep].abs()).float().mean()) >= 0.99
    )
