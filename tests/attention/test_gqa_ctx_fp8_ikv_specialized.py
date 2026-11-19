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

Tests for the SM103 "specialized" fp8 token-interleaved GQA-context backend
of trtllm_batch_context_with_kv_cache: correctness vs an fp32 reference,
backend-dispatch semantics (auto / explicit / kill switch / allowlist
boundary), and CUDA-graph replay == eager.
"""

import os

import pytest
import torch

import flashinfer.gqa_ctx_fp8_ikv_cuda as spec_mod
from flashinfer.prefill import (
    batch_context_with_kv_cache,
    trtllm_batch_context_with_kv_cache,
)

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the specialized fp8-iKV GQA context kernel is SM103-only",
)

SCALE = 0.08838834764831845  # 128 ** -0.5
PAGE, HQ, D = 128, 16, 128


def _make_inputs(q_lens, kv_lens, seed=0, btw_pad=0, shuffle_pages=False):
    """btw_pad widens the block table past the pages any request needs, with
    arbitrary (valid-pool) ids in the unused columns, and shuffle_pages
    scatters the real page ids -- the serving call form, where tables are as
    wide as the engine's maximum pages per sequence."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    tq = sum(q_lens)
    pprs = [(k + PAGE - 1) // PAGE for k in kv_lens]
    np_, btw = sum(pprs), max(pprs) + btw_pad
    q = (torch.randn(tq, HQ, D, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    kv_buf = (torch.randn(np_, PAGE, 2, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    if btw_pad:
        # garbage ids (any pool slot) where serving engines leave stale columns
        bt = torch.randint(
            0, np_, (len(q_lens), btw), generator=g, dtype=torch.int32, device=dev
        )
    else:
        bt = torch.zeros(len(q_lens), btw, dtype=torch.int32, device=dev)
    if shuffle_pages:
        page_ids = torch.randperm(np_, generator=g, device=dev).to(torch.int32)
    else:
        page_ids = torch.arange(np_, dtype=torch.int32, device=dev)
    s = 0
    for i, ppr in enumerate(pprs):
        bt[i, :ppr] = page_ids[s : s + ppr]
        s += ppr
    cu_q = torch.tensor(
        [0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=dev
    )
    cu_kv = torch.tensor(
        [0] + list(torch.tensor(kv_lens).cumsum(0)), dtype=torch.int32, device=dev
    )
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    return dict(
        q=q,
        kv_buf=kv_buf,
        bt=bt,
        cu_q=cu_q,
        cu_kv=cu_kv,
        seq_lens=seq_lens,
        max_q=max(q_lens),
        max_kv=max(kv_lens),
    )


def _reference(inp):
    """fp32 chunked-prefill causal GQA over the token-interleaved cache."""
    dev = inp["q"].device
    cu = inp["cu_q"].to(torch.int64).tolist()
    outs = []
    for r in range(inp["seq_lens"].shape[0]):
        q_len = cu[r + 1] - cu[r]
        kv_len = int(inp["seq_lens"][r])
        n_pages = (kv_len + PAGE - 1) // PAGE
        flat = inp["kv_buf"][inp["bt"][r, :n_pages].to(torch.int64)]
        k = flat[:, :, 0, :].reshape(-1, D)[:kv_len].to(torch.float32)
        v = flat[:, :, 1, :].reshape(-1, D)[:kv_len].to(torch.float32)
        qf = inp["q"][cu[r] : cu[r + 1]].to(torch.float32)
        sc = torch.einsum("qhd,ld->qhl", qf, k) * SCALE
        pos = torch.arange(q_len, device=dev)
        vis = kv_len - q_len + pos + 1
        mask = torch.arange(kv_len, device=dev).view(1, 1, -1) >= vis.view(-1, 1, 1)
        pr = torch.softmax(sc.masked_fill(mask, float("-inf")), dim=-1)
        outs.append(torch.einsum("qhl,ld->qhd", pr, v))
    return torch.cat(outs, 0).to(torch.bfloat16)


def _api(inp, out, backend="auto"):
    # the generic front is the ONLY surface exposing the specialized
    # backend (vendor-named entries stay stock)
    kv = inp["kv_buf"]
    return batch_context_with_kv_cache(
        query=inp["q"],
        kv_cache=(kv[:, :, 0, :].unsqueeze(1), kv[:, :, 1, :].unsqueeze(1)),
        workspace_buffer=_ws(),
        block_tables=inp["bt"],
        seq_lens=inp["seq_lens"],
        max_q_len=inp["max_q"],
        max_kv_len=inp["max_kv"],
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        batch_size=int(inp["seq_lens"].shape[0]),
        cum_seq_lens_q=inp["cu_q"],
        cum_seq_lens_kv=inp["cu_kv"],
        window_left=-1,
        out=out,
        backend=backend,
    )


_WS = {}


def _ws():
    if "b" not in _WS:
        _WS["b"] = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    return _WS["b"]


def _out_like(inp):
    return torch.empty(inp["q"].shape[0], HQ, D, dtype=torch.bfloat16, device="cuda")


@requires_sm103
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        ([2] * 8, [2] * 8),  # tiny-KV batch (census class B8 q2 kv2)
        ([2] * 64, [2] * 64),  # tiny-KV batch, large B
        ([128], [128]),  # single-page single request
        ([2, 1, 1, 1], [100, 1, 1, 1]),  # ragged, deep-context causal row
        ([1], [3]),  # tiny-cap route
        ([2], [4]),
        ([128] * 10, [128] * 10),  # uniform full-page batch routes
    ],
)
def test_specialized_correct_and_routed(q_lens, kv_lens):
    inp = _make_inputs(q_lens, kv_lens, seed=7)
    gold = _reference(inp).float()
    before = dict(spec_mod.DISPATCH_STATS)
    out = _out_like(inp)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    diff = (out.float() - gold).abs()
    tol = 0.005 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) == 1.0
    # explicit specialized produces the same bits as auto
    out2 = _out_like(inp)
    _api(inp, out2, backend="specialized")
    assert torch.equal(out, out2)


@requires_sm103
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        ([2] * 8, [2] * 8),  # tiny q2/kv2, B<=16: attn_tiny_async<BTW1=false>
        ([2] * 64, [2] * 64),  # tiny q2/kv2, large B, BTW1=false
        ([3] * 4, [3] * 4),  # q3 route
        ([4] * 4, [4] * 4),  # q4 route
        ([6] * 4, [6] * 4),  # q6 route
        ([8] * 4, [8] * 4),  # q8 route
        ([2, 1, 1, 1], [100, 1, 1, 1]),  # ragged deep-context causal row
        ([1], [3]),  # tiny-cap route
        ([128], [128]),  # B=1 q128 (attn_split; table width immaterial but covered)
        ([128] * 7, [128] * 7),  # B>1 uniform q128, wide table: attn_fast U128
        ([128] * 10, [128] * 10),  # B>1 uniform q128, wide table: generic U128
        ([128] * 15, [128] * 15),  # census-row B15 q128 in the serving form
    ],
)
def test_specialized_wide_block_table(q_lens, kv_lens):
    """Serving engines pass block tables as wide as the engine's maximum pages
    per sequence, with arbitrary ids in the unused columns and non-contiguous
    page placement.  That call form selects the BTW1=false tiny variants and
    the wide-table U128 kernels (attn_fast / the generic U128 launch) that a
    minimum-width contiguous table never reaches."""
    inp = _make_inputs(q_lens, kv_lens, seed=29, btw_pad=7, shuffle_pages=True)
    assert inp["bt"].shape[1] >= 8
    gold = _reference(inp).float()
    before = dict(spec_mod.DISPATCH_STATS)
    out = _out_like(inp)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    diff = (out.float() - gold).abs()
    tol = 0.005 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) == 1.0
    # the serving form must be bitwise repeatable too
    out2 = _out_like(inp)
    _api(inp, out2)
    assert torch.equal(out, out2)


def _make_padded_q128_inputs(B, q_len, seed=0):
    """Padded-capacity packing: physical q/out rows say uniform q128
    (q.shape[0] == B*128, max_q_len == 128) while cu_q packs the actual
    shorter requests tightly.  This is the shape that must defeat the
    uniform-q128 speculation per request, not just per batch."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    q = (torch.randn(B * PAGE, HQ, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    kv_buf = (torch.randn(B, PAGE, 2, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    bt = torch.arange(B, dtype=torch.int32, device=dev).view(B, 1)
    cu_q = q_len * torch.arange(B + 1, dtype=torch.int32, device=dev)
    cu_kv = PAGE * torch.arange(B + 1, dtype=torch.int32, device=dev)
    seq_lens = torch.full((B,), PAGE, dtype=torch.int32, device=dev)
    return dict(
        q=q,
        kv_buf=kv_buf,
        bt=bt,
        cu_q=cu_q,
        cu_kv=cu_kv,
        seq_lens=seq_lens,
        max_q=PAGE,
        max_kv=PAGE,
    )


@requires_sm103
@pytest.mark.parametrize(
    "B,q_len", [(2, 127), (7, 125), (10, 125), (10, 127), (15, 125)]
)
def test_u128_padded_capacity_short_requests(B, q_len):
    # Regression: with padded physical capacity and shorter actual requests,
    # the uniform-q128 tail warps used to write a full garbage result to each
    # request's FIRST row (tok fell back to q0), racing the CTA that owns it.
    inp = _make_padded_q128_inputs(B, q_len, seed=17)
    gold = _reference(inp).float()
    nvalid = B * q_len
    before = dict(spec_mod.DISPATCH_STATS)
    out = _out_like(inp)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    valid = out[:nvalid].float()
    diff = (valid - gold).abs()
    tol = 0.005 + 0.05 * gold.abs()
    assert float((diff <= tol).float().mean()) == 1.0
    # the race was also nondeterministic: identical inputs must be bitwise
    # repeatable on the valid rows
    out2 = _out_like(inp)
    _api(inp, out2)
    assert torch.equal(out[:nvalid], out2[:nvalid])


@requires_sm103
def test_u2_large_batch_beyond_chunk_boundary():
    # Regression: uniform q2/kv2 batches larger than 65535 used to be launched
    # in request chunks with q/out pointers advanced by the chunk origin while
    # the kernel's speculation fallback consumed cu_q's ABSOLUTE offsets --
    # applying the origin twice: rows past the boundary were written outside
    # the passed output view and the final logical rows were left untouched.
    B = 65536
    dev = torch.device("cuda")
    q = torch.zeros(B * 2, HQ, D, device=dev).to(torch.float8_e4m3fn)
    kv_buf = torch.empty(1, PAGE, 2, D, device=dev)
    kv_buf[:, :, 0, :] = 0.0  # K = 0 -> uniform softmax
    kv_buf[:, :, 1, :] = 1.0  # V = 1 -> every output element is exactly 1
    kv_buf = kv_buf.to(torch.float8_e4m3fn)
    inp = dict(
        q=q,
        kv_buf=kv_buf,
        bt=torch.zeros(B, 1, dtype=torch.int32, device=dev),
        cu_q=2 * torch.arange(B + 1, dtype=torch.int32, device=dev),
        cu_kv=2 * torch.arange(B + 1, dtype=torch.int32, device=dev),
        seq_lens=torch.full((B,), 2, dtype=torch.int32, device=dev),
        max_q=2,
        max_kv=2,
    )
    before = dict(spec_mod.DISPATCH_STATS)
    out = torch.full((B * 2, HQ, D), -8.0, dtype=torch.bfloat16, device=dev)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1, (
        spec_mod.LAST_FALLBACK_REASONS
    )
    # every logical row (poisoned beforehand) must have been written, exactly
    # within the passed view
    assert float((out.float() - 1.0).abs().max()) <= 0.02


@requires_sm103
def test_allowlist_boundary_multi_page_stays_stock():
    # kv > 128 (multi-page) is outside the allowlist: auto must fall back.
    inp = _make_inputs([4], [512], seed=3)
    before = dict(spec_mod.DISPATCH_STATS)
    out = _out_like(inp)
    _api(inp, out)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    assert "multi_page_regime" in spec_mod.LAST_FALLBACK_REASONS
    gold = _reference(inp).float()
    diff = (out.float() - gold).abs()
    # Sanity bound only: the stock path's own numeric floor at this contract
    # is not this PR's subject.
    assert float((diff <= 0.005 + 0.05 * gold.abs()).float().mean()) >= 0.95
    with pytest.raises(ValueError, match="fallback_reasons"):
        _api(inp, _out_like(inp), backend="specialized")


@requires_sm103
def test_kill_switch_restores_stock():
    inp = _make_inputs([2] * 8, [2] * 8, seed=11)
    os.environ[spec_mod.KILL_SWITCH_ENV] = "1"
    try:
        before = dict(spec_mod.DISPATCH_STATS)
        out = _out_like(inp)
        _api(inp, out)
        assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
        assert spec_mod.LAST_FALLBACK_REASONS[0].startswith("kill_switch")
        gold = _reference(inp).float()
        diff = (out.float() - gold).abs()
        # Sanity bound only (see test_allowlist_boundary note).
        assert float((diff <= 0.005 + 0.05 * gold.abs()).float().mean()) >= 0.95
        with pytest.raises(ValueError, match="fallback_reasons"):
            _api(inp, _out_like(inp), backend="specialized")
    finally:
        os.environ.pop(spec_mod.KILL_SWITCH_ENV, None)


@requires_sm103
def test_cuda_graph_replay_matches_eager():
    inp = _make_inputs([2] * 8, [2] * 8, seed=5)
    out = _out_like(inp)
    _api(inp, out)  # eager warmup also builds the module before capture
    eager = out.clone()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _api(inp, out)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(g):
        _api(inp, out)
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)
    # replay with mutated inputs: the graph must read the live buffers
    kv = inp["kv_buf"]
    kv.copy_((torch.randn_like(kv.to(torch.float32)) * 0.5).to(torch.float8_e4m3fn))
    inp["q"].copy_(
        (torch.randn_like(inp["q"].to(torch.float32)) * 0.5).to(torch.float8_e4m3fn)
    )
    g.replay()
    torch.cuda.synchronize()
    gold2 = _reference(inp).float()
    diff2 = (out.float() - gold2).abs()
    assert float((diff2 <= 0.005 + 0.05 * gold2.abs()).float().mean()) == 1.0


@requires_sm103
def test_generalized_entry_parity_and_dispatch():
    """batch_context_with_kv_cache is a front over NAMED real backends:
    "specialized" links directly to the agent-generated kernel, "trtllm"/
    "cake" are the trtllm-named entry's paths, and "auto" only selects
    among the names (no behavior of its own)."""
    from flashinfer.prefill import (
        BATCH_CONTEXT_BACKENDS,
        batch_context_with_kv_cache,
    )

    assert BATCH_CONTEXT_BACKENDS == ("auto", "specialized", "trtllm", "cake")
    inp = _make_inputs([2] * 8, [2] * 8, seed=11)
    kv = inp["kv_buf"]
    kwargs = dict(
        query=inp["q"],
        kv_cache=(kv[:, :, 0, :].unsqueeze(1), kv[:, :, 1, :].unsqueeze(1)),
        workspace_buffer=_ws(),
        block_tables=inp["bt"],
        seq_lens=inp["seq_lens"],
        max_q_len=inp["max_q"],
        max_kv_len=inp["max_kv"],
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        batch_size=int(inp["seq_lens"].shape[0]),
        cum_seq_lens_q=inp["cu_q"],
        cum_seq_lens_kv=inp["cu_kv"],
        window_left=-1,
    )
    # "auto" selects "specialized" on its allowlist (dispatch stats move)
    before = dict(spec_mod.DISPATCH_STATS)
    gen_out = batch_context_with_kv_cache(out=_out_like(inp), **kwargs)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"] + 1
    # the direct specialized name produces the same bits, without touching
    # the trtllm-named entry's dispatch
    gen_spec = batch_context_with_kv_cache(
        out=_out_like(inp), backend="specialized", **kwargs
    )
    assert torch.equal(gen_spec, gen_out)
    # the vendor-named entry stays STOCK: it has no specialized backend
    with pytest.raises(ValueError, match="trtllm-gen|cake"):
        trtllm_batch_context_with_kv_cache(
            out=_out_like(inp), backend="specialized", **kwargs
        )
    # the named "trtllm" backend equals the entry's trtllm-gen path
    gen_stock = batch_context_with_kv_cache(
        out=_out_like(inp), backend="trtllm", **kwargs
    )
    direct_stock = _api(inp, _out_like(inp), backend="trtllm")
    assert torch.equal(gen_stock, direct_stock)
    # off-allowlist: auto selects "trtllm"; explicit "specialized" raises
    big = _make_inputs([2] * 2, [600] * 2, seed=13)  # multi-page KV
    bkv = big["kv_buf"]
    bkwargs = dict(
        query=big["q"],
        kv_cache=(bkv[:, :, 0, :].unsqueeze(1), bkv[:, :, 1, :].unsqueeze(1)),
        workspace_buffer=_ws(),
        block_tables=big["bt"],
        seq_lens=big["seq_lens"],
        max_q_len=big["max_q"],
        max_kv_len=big["max_kv"],
        bmm1_scale=SCALE,
        bmm2_scale=1.0,
        batch_size=int(big["seq_lens"].shape[0]),
        cum_seq_lens_q=big["cu_q"],
        cum_seq_lens_kv=big["cu_kv"],
        window_left=-1,
    )
    before = dict(spec_mod.DISPATCH_STATS)
    auto_out = batch_context_with_kv_cache(out=_out_like(big), **bkwargs)
    assert spec_mod.DISPATCH_STATS["specialized"] == before["specialized"]
    trt_out = batch_context_with_kv_cache(
        out=_out_like(big), backend="trtllm", **bkwargs
    )
    assert torch.equal(auto_out, trt_out)
    with pytest.raises(ValueError, match="fallback_reasons"):
        batch_context_with_kv_cache(
            out=_out_like(big), backend="specialized", **bkwargs
        )
    # invalid backend raises at the front
    with pytest.raises(ValueError, match="backend must be one of"):
        batch_context_with_kv_cache(out=_out_like(inp), backend="bogus", **kwargs)
