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
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the specialized fp8-iKV GQA context kernel is SM103-only",
)

SCALE = 0.08838834764831845  # 128 ** -0.5
PAGE, HQ, D = 128, 16, 128


def _make_inputs(q_lens, kv_lens, seed=0):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    tq = sum(q_lens)
    pprs = [(k + PAGE - 1) // PAGE for k in kv_lens]
    np_, btw = sum(pprs), max(pprs)
    q = (torch.randn(tq, HQ, D, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    kv_buf = (torch.randn(np_, PAGE, 2, D, generator=g, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    bt = torch.zeros(len(q_lens), btw, dtype=torch.int32, device=dev)
    s = 0
    for i, ppr in enumerate(pprs):
        bt[i, :ppr] = torch.arange(s, s + ppr, dtype=torch.int32, device=dev)
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
    kv = inp["kv_buf"]
    return trtllm_batch_context_with_kv_cache(
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
