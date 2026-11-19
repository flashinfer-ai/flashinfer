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

Tests for fp8_paged_mqa_topk_indexer (SM103): EXACT integer equality vs an
mathematical-score reference (lower index wins exact ties), repeat
determinism, rejection semantics (off-allowlist + kill switch raise), out=
handling, and CUDA-graph replay == eager.
"""

import os

import pytest
import torch

import flashinfer.attn_scores.prefill_indexer as pi
from flashinfer.attn_scores import fp8_paged_mqa_topk_indexer

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the fused fp8 paged-MQA top-k indexer kernel is SM103-only",
)

IH, ID, CR, BTOPK = 4, 128, 4, 512
TOPK, WIDTH, PW, CPAGE = 2048, 2051, 2052, 64
ROW_CHUNK = 2048


def _req_lens(T, B):
    base = T // B
    return [base + (1 if i < T - base * B else 0) for i in range(B)]


def _make_inputs(T, B, seed=0):
    dev = torch.device("cuda")
    lens = _req_lens(T, B)
    cu = [0]
    for ln in lens:
        cu.append(cu[-1] + ln)
    pos = torch.cat([torch.arange(ln, device=dev) for ln in lens]).to(torch.int32)
    visible = ((pos.to(torch.int64) + 1) // CR).to(torch.int32)
    nblk = [ln // CR for ln in lens]
    cpages = [max((n + CPAGE - 1) // CPAGE, 1) for n in nblk]
    ncp, maxcp = sum(cpages), max(cpages)
    table = torch.zeros(B, maxcp, dtype=torch.int32, device=dev)
    s = 0
    for r in range(B):
        table[r, : cpages[r]] = torch.arange(
            s, s + cpages[r], dtype=torch.int32, device=dev
        )
        s += cpages[r]
    g = torch.Generator(device=dev).manual_seed(seed)
    # SIGNED data: the per-head relu in the scoring rule is live code.
    qi = (torch.randn(T, IH, ID, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    ck = (
        (torch.randn(ncp * CPAGE, ID, generator=g, device=dev) * 0.5)
        .to(torch.float8_e4m3fn)
        .reshape(ncp, CPAGE, 1, ID)
    )
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=dev)
    return dict(
        qi=qi,
        ck_cache=ck,
        cpage_table=table,
        cu_seqlens=cu_seqlens,
        pos=pos,
        visible=visible,
    )


def _reference(inp):
    """Exact finite-E4M3 score reference with the lower-index tie-break:
    score visible blocks with sum_h relu(q_h . k_b), take the top-512 by
    (score, -block) order, expand x4, append the causal tail, emit packed
    ascending int32 rows with the count in the last column."""
    dev = inp["qi"].device
    T = inp["qi"].shape[0]
    B = inp["cpage_table"].shape[0]
    cu = inp["cu_seqlens"].to(torch.int64).tolist()
    # E4M3 products are multiples of 2**-18. Even the absolute sum over
    # 4 * 128 products is < 2**27, so binary64 represents every partial
    # sum exactly. Neither GEMM nor head reduction order can create a tie.
    flat_ck = inp["ck_cache"].reshape(-1, ID).to(torch.float64)
    packed = torch.full((T, PW), -1, dtype=torch.int32, device=dev)
    arange_cr = torch.arange(CR, device=dev, dtype=torch.int64)
    for r in range(B):
        t0, t1 = cu[r], cu[r + 1]
        ln = t1 - t0
        nblk = ln // CR
        if nblk > 0:
            n_pages = (nblk + CPAGE - 1) // CPAGE
            rows = (
                inp["cpage_table"][r, :n_pages].to(torch.int64).unsqueeze(1) * CPAGE
                + torch.arange(CPAGE, device=dev).unsqueeze(0)
            ).reshape(-1)[:nblk]
            ck_r = flat_ck[rows]
        for c0 in range(t0, t1, ROW_CHUNK):
            c1 = min(c0 + ROW_CHUNK, t1)
            n = c1 - c0
            pos_c = inp["pos"][c0:c1].to(torch.int64)
            vis_c = inp["visible"][c0:c1].to(torch.int64)
            if nblk > 0:
                qf = inp["qi"][c0:c1].to(torch.float64)
                logits = torch.einsum("thd,bd->thb", qf, ck_r).clamp_min(0).sum(1)
                blk_idx = torch.arange(nblk, device=dev)
                vis_mask = blk_idx.view(1, -1) < vis_c.view(-1, 1)
                logits = logits.masked_fill(~vis_mask, float("-inf"))
                # stable descending sort keeps ascending block order among
                # equal scores — exactly the lower-index-wins tie-break.
                order = torch.argsort(logits, dim=1, descending=True, stable=True)
                for i in range(n):
                    v = int(vis_c[i])
                    sel = order[i, : min(BTOPK, v)]
                    sel, _ = torch.sort(sel)
                    toks = (sel.unsqueeze(1) * CR + arange_cr.view(1, -1)).reshape(-1)
                    tail0 = v * CR
                    tail = torch.arange(tail0, int(pos_c[i]) + 1, device=dev)
                    row = torch.cat([toks, tail])
                    packed[c0 + i, : row.numel()] = row.to(torch.int32)
                    packed[c0 + i, PW - 1] = row.numel()
            else:
                for i in range(n):
                    tail = torch.arange(0, int(pos_c[i]) + 1, device=dev)
                    packed[c0 + i, : tail.numel()] = tail.to(torch.int32)
                    packed[c0 + i, PW - 1] = tail.numel()
    return packed


def _api(inp, out=None, **kw):
    return fp8_paged_mqa_topk_indexer(
        inp["qi"],
        inp["ck_cache"],
        inp["cpage_table"],
        inp["cu_seqlens"],
        inp["pos"],
        inp["visible"],
        out=out,
        **kw,
    )


# The kernel is bitwise deterministic across runs of any given build, and
# dispatch refuses toolkits it was not validated on (single source of truth:
# pi.VALIDATED_CUDA_VERSIONS, applied through pi._toolkit_validation_reasons,
# which also gates on the nvcc the JIT build selects). Functional tests
# therefore run only on validated toolkits, where selection is
# reference-exact; the refusal itself is asserted everywhere else.
VALIDATED_CUDA_VERSIONS = pi.VALIDATED_CUDA_VERSIONS
TOOLKIT_VALIDATED = not pi._toolkit_validation_reasons()
requires_validated_toolkit = pytest.mark.skipif(
    not TOOLKIT_VALIDATED,
    reason="dispatch refuses unvalidated CUDA toolkits "
    f"(cuda={torch.version.cuda}, validated={VALIDATED_CUDA_VERSIONS}); "
    "the refusal is covered by test_unvalidated_toolkit_refusal",
)


def test_cuda_130_admitted_with_workaround():
    """CPU-reachable: CUDA 13.0 is a validated toolkit — served through the
    compiled-in unroll-1 workaround for its ptxas optimizer defect (see the
    kernel source comment), validated exact at a small perf cost on that
    toolkit only — so the exact-membership gate must admit it while still
    rejecting the never-validated neighbors."""
    assert "13.0" in pi.VALIDATED_CUDA_VERSIONS
    assert pi._is_validated_release("13.0")
    assert pi._is_validated_release("13.0.1")  # major.minor extraction
    assert not pi._is_validated_release("12.9")
    assert not pi._is_validated_release("13.00")  # exact membership, no prefixes


@requires_sm103
@requires_validated_toolkit
def test_selection_boundary_tie_and_small_gap():
    inp = _make_inputs(2930, 1)
    t = CR * (BTOPK + 1) - 1  # first row with 513 visible blocks, no tail
    q = torch.zeros_like(inp["qi"], dtype=torch.float32)
    q[t : t + 2, 0, 0] = 1
    q[t + 1, 1, 1] = 2**-6
    k = torch.zeros_like(inp["ck_cache"], dtype=torch.float32).view(-1, ID)
    k[: BTOPK - 1, 0] = 64
    k[BTOPK - 1 : BTOPK + 1, 0] = 32
    k[BTOPK, 1] = 2**-6
    inp["qi"] = q.to(inp["qi"].dtype)
    inp["ck_cache"] = k.view_as(inp["ck_cache"]).to(inp["ck_cache"].dtype)

    # At t the two candidates score 32; at t+1 the higher index wins
    # by 2**-12 (< 1e-5 * 32), and token 2052 is the causal tail.
    expected = torch.full((2, PW), -1, dtype=torch.int32, device=q.device)
    expected[:, :TOPK] = torch.arange(TOPK, device=q.device)
    expected[1, TOPK - CR : TOPK] += CR
    expected[1, TOPK] = t + 1
    expected[0, -1] = TOPK
    expected[1, -1] = TOPK + 1
    for rows in (_reference(inp), _api(inp)):
        torch.testing.assert_close(rows[t : t + 2], expected, rtol=0, atol=0)


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize(
    "T,B",
    [
        (122, 1),  # direct regime (visible <= 512 blocks)
        (949, 2),  # multi-request, still direct
        (2930, 1),  # scored selection regime (nblk > 512)
        (16161, 6),  # multi-request scored regime
    ],
)
def test_selection_contract_and_determinism(T, B):
    inp = _make_inputs(T, B, seed=13)
    gold = _reference(inp)
    out = _api(inp)
    # Compare every column, including block interiors, tail, padding and count.
    torch.testing.assert_close(out, gold, rtol=0, atol=0)
    # bitwise determinism per build, always
    for _ in range(3):
        assert torch.equal(_api(inp), out)
    buf = torch.empty(T, PW, dtype=torch.int32, device="cuda")
    out2 = _api(inp, out=buf)
    assert out2 is buf
    assert torch.equal(out2, out)


@requires_sm103
def test_unvalidated_toolkit_refusal():
    if TOOLKIT_VALIDATED:
        pytest.skip("validated toolkit: dispatch serves; refusal not reachable")
    inp = _make_inputs(122, 1, seed=3)
    with pytest.raises(ValueError, match="unvalidated_toolkit"):
        _api(inp)


@requires_sm103
@requires_validated_toolkit
def test_rejection_semantics():
    inp = _make_inputs(949, 2, seed=3)
    # off-allowlist raises with the fallback reasons (no stock path exists)
    with pytest.raises(ValueError, match="qi_not_Tx4x128_fp8e4m3"):
        _api({**inp, "qi": inp["qi"].to(torch.bfloat16)})
    # only the "specialized" backend exists
    with pytest.raises(ValueError, match="backend must be one of"):
        _api(inp, backend="auto")
    # kill switch: every call raises
    os.environ[pi.KILL_SWITCH_ENV] = "1"
    try:
        with pytest.raises(ValueError, match="kill_switch"):
            _api(inp)
    finally:
        os.environ.pop(pi.KILL_SWITCH_ENV, None)


@requires_sm103
@requires_validated_toolkit
def test_cuda_graph_replay_matches_eager():
    inp = _make_inputs(949, 2, seed=5)
    buf = torch.empty(949, PW, dtype=torch.int32, device="cuda")
    _api(inp, out=buf)  # eager warmup builds the module before capture
    eager = buf.clone()
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())  # order after allocator work
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
