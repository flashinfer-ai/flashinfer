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

import pytest
import torch

import flashinfer
from flashinfer.mla import BatchMLAPagedAttentionWrapper
from flashinfer.mla._batch_mla._decode_fp8_sm90 import (
    _mla_dec_fp8q_sm90_stats,
    _packed_fp8_view,
    _reset_mla_dec_fp8q_sm90_stats,
    _scale_value,
)

DISABLE_ENV = "FLASHINFER_DISABLE_MLA_DECODE_FP8_PAGED_SM90"
NH, CKV, KPE, PAGE, QL = 8, 512, 64, 64, 6
DIMQ = CKV + KPE
SM_SCALE = DIMQ**-0.5
# all-fp8 class gate (the measured verification tolerance)
RTOL, ATOL, MATCHED = 0.12, 0.02, 0.98


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    _reset_mla_dec_fp8q_sm90_stats()
    yield
    _reset_mla_dec_fp8q_sm90_stats()


def _sm90_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")


# --------------------------------------------------------------------------
# GPU-free: structural packed-view reconstruction and scale normalization
# --------------------------------------------------------------------------


def test_packed_view_reconstruction_and_rejection():
    q = torch.zeros((4 * QL, NH, DIMQ), dtype=torch.float8_e4m3fn)
    qn, qp = q[..., :CKV], q[..., CKV:]
    pv = _packed_fp8_view((qn, qp), (CKV, KPE), "query")
    assert pv is not None and tuple(pv.shape) == (4 * QL, NH, DIMQ)
    assert pv.data_ptr() == q.data_ptr()
    # packed tensor passes through
    assert _packed_fp8_view(q, (CKV, KPE), "query") is not None
    # independent (non-adjacent) split pairs are rejected, not copied
    assert _packed_fp8_view(
        (qn.contiguous(), qp.contiguous()), (CKV, KPE), "query"
    ) is None
    # wrong dtype rejected
    qb = torch.zeros((4 * QL, NH, DIMQ), dtype=torch.bfloat16)
    assert _packed_fp8_view(qb, (CKV, KPE), "query") is None


def test_scale_normalization():
    assert _scale_value(None, "x") == (1.0, None)
    assert _scale_value(0.5, "x") == (0.5, None)
    v, r = _scale_value(torch.tensor([0.25]), "x")
    assert v == 0.25 and r is None
    v, r = _scale_value(torch.zeros(2), "x")
    assert v is None and r is not None


def test_stats_report_build_state_and_counters():
    stats = _mla_dec_fp8q_sm90_stats()
    for key in ("plans_seen", "runs_dispatched", "module_loaded", "module_error"):
        assert key in stats


# --------------------------------------------------------------------------
# Device tests (SM90): reference equivalence, kill-switch, fallback, graph
# --------------------------------------------------------------------------


def _make_batch(kv_lens, device, seed=0):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    B = len(kv_lens)
    pprs = [(k + PAGE - 1) // PAGE for k in kv_lens]
    P = sum(pprs)
    pps = max(pprs)
    bt = torch.zeros((B, pps), dtype=torch.int32, device=device)
    s = 0
    indptr = [0]
    indices = []
    for i, ppr in enumerate(pprs):
        bt[i, :ppr] = torch.arange(s, s + ppr, dtype=torch.int32, device=device)
        indices.extend(range(s, s + ppr))
        indptr.append(indptr[-1] + ppr)
        s += ppr

    def rnd(shape):
        x = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
        return (x * 0.5).to(torch.float8_e4m3fn)

    TQ = B * QL
    return dict(
        q=rnd((TQ, NH, DIMQ)),
        kv=rnd((P, PAGE, DIMQ)),
        qo_indptr=torch.arange(0, (B + 1) * QL, QL, dtype=torch.int32,
                               device=device),
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=torch.tensor(indices, dtype=torch.int32, device=device),
        kv_len_arr=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        block_table=bt,
    )


def _reference(batch, q_scale=1.0, kv_scale=1.0):
    """fp32 dequant-then-attend MLA decode reference (query row i of request
    r sees kv positions 0..kv_len_r - QL + i)."""
    device = batch["q"].device
    q = batch["q"].float() * q_scale
    kvf = batch["kv"].float().reshape(-1, DIMQ) * kv_scale
    kv_lens = batch["kv_len_arr"].tolist()
    bt = batch["block_table"]
    B = len(kv_lens)
    out = torch.empty((B * QL, NH, CKV), dtype=torch.bfloat16, device=device)
    off = torch.arange(PAGE, device=device)
    for r in range(B):
        kvl = kv_lens[r]
        npages = (kvl + PAGE - 1) // PAGE
        flat = (bt[r, :npages].long().unsqueeze(1) * PAGE + off).reshape(-1)[:kvl]
        k = kvf[flat]                      # (kvl, 576)
        v = k[:, :CKV]                     # (kvl, 512)
        qr = q[r * QL : (r + 1) * QL]      # (QL, 8, 576)
        sc = torch.einsum("qhd,ld->qhl", qr, k) * SM_SCALE
        vis = (kvl - QL) + torch.arange(QL, device=device) + 1
        mask = torch.arange(kvl, device=device).view(1, 1, kvl) >= vis.view(
            QL, 1, 1
        )
        sc = sc.masked_fill(mask, float("-inf"))
        out[r * QL : (r + 1) * QL] = torch.einsum(
            "qhl,ld->qhd", torch.softmax(sc, -1), v
        ).to(torch.bfloat16)
    return out


def _plan_wrapper(batch, device):
    ws = torch.zeros(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = BatchMLAPagedAttentionWrapper(float_workspace_buffer=ws,
                                      backend="auto")
    w.plan(
        qo_indptr=batch["qo_indptr"],
        kv_indptr=batch["kv_indptr"],
        kv_indices=batch["kv_indices"],
        kv_len_arr=batch["kv_len_arr"],
        num_heads=NH,
        head_dim_ckv=CKV,
        head_dim_kpe=KPE,
        page_size=PAGE,
        causal=True,
        sm_scale=SM_SCALE,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
    )
    return w


def _run(w, batch, q_scale=1.0, kv_scale=1.0, out=None):
    q = batch["q"]
    kv = batch["kv"]
    return w.run(
        query=(q[..., :CKV], q[..., CKV:]),
        kv_cache=(kv[..., :CKV], kv[..., CKV:]),
        q_scale=q_scale,
        ckv_scale=kv_scale,
        kpe_scale=kv_scale,
        out=out,
    )


def _matched_ratio(y, ref):
    yf, rf = y.float(), ref.float()
    ok = (yf - rf).abs() <= (ATOL + RTOL * rf.abs())
    return float(ok.float().mean().item())


@pytest.mark.parametrize(
    "kv_lens",
    [
        [2983],
        [2983, 4097],
        [1500, 658, 760, 341, 863],
    ],
)
def test_specialized_matches_reference_and_stock(kv_lens, monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch(kv_lens, device)
    w = _plan_wrapper(batch, device)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 1, stats

    out = _run(w, batch)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    ref = _reference(batch)
    assert _matched_ratio(out, ref) >= MATCHED

    # determinism: the kernel is atomics-free and bit-stable
    out2 = _run(w, batch)
    assert torch.equal(out, out2)

    # kill-switch: the same call is served by dequant + stock MLA
    monkeypatch.setenv(DISABLE_ENV, "1")
    out_stock = _run(w, batch)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_disabled"] == 1 and stats["runs_dequant_fallback"] == 1, stats
    assert _matched_ratio(out_stock, ref) >= MATCHED
    assert _matched_ratio(out, out_stock) >= MATCHED


def test_scales_are_honored():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983, 4097], device)
    w = _plan_wrapper(batch, device)
    out = _run(w, batch, q_scale=0.5, kv_scale=2.0)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 1
    ref = _reference(batch, q_scale=0.5, kv_scale=2.0)
    assert _matched_ratio(out, ref) >= MATCHED


def test_non_adjacent_views_fall_back_to_dequant_stock():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983, 760], device)
    w = _plan_wrapper(batch, device)
    q = batch["q"]
    kv = batch["kv"]
    out = w.run(
        query=(q[..., :CKV].contiguous(), q[..., CKV:].contiguous()),
        kv_cache=(kv[..., :CKV], kv[..., CKV:]),
        q_scale=1.0,
        ckv_scale=1.0,
        kpe_scale=1.0,
    )
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0
    assert stats["runs_ineligible"] == 1 and stats["runs_dequant_fallback"] == 1
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_graph_capture_replays_bitwise_equal_to_eager():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983, 4097, 1500], device)
    w = _plan_wrapper(batch, device)
    out = torch.empty((3 * QL, NH, CKV), dtype=torch.bfloat16, device=device)
    eager = _run(w, batch).clone()
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 1

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(w, batch, out=out)  # warmup on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        _run(w, batch, out=out)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched_capturing"] == 1, stats

    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)

    # new content through the same buffers replays correctly
    batch["q"].copy_(_make_batch([2983, 4097, 1500], device, seed=7)["q"])
    graph.replay()
    torch.cuda.synchronize()
    eager2 = _run(w, batch)
    assert torch.equal(out, eager2)


def test_ragged_short_kv_b5_workspace_bound_regression():
    """Regression: the plan-time split-workspace bound must cover the
    repaired launcher's B>=5 ONE-PAGE-split path, where the split count can
    reach pages_per_seq. The earlier two-page-only bound
    min(256, ceil(pps/2)) undersized exactly this ragged short-kv B=5 shape
    (pps=24) and run() refused fail-closed with 'workspace too small'."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([1500, 658, 760, 341, 863], device)
    w = _plan_wrapper(batch, device)
    out = _run(w, batch)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 1
    ref = _reference(batch)
    assert _matched_ratio(out, ref) >= MATCHED
