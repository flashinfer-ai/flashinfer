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
    assert (
        _packed_fp8_view((qn.contiguous(), qp.contiguous()), (CKV, KPE), "query")
        is None
    )
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


def _make_batch(kv_lens, device, seed=0, ql=QL):
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

    TQ = B * ql
    return dict(
        q=rnd((TQ, NH, DIMQ)),
        kv=rnd((P, PAGE, DIMQ)),
        qo_indptr=torch.arange(0, (B + 1) * ql, ql, dtype=torch.int32, device=device),
        kv_indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        kv_indices=torch.tensor(indices, dtype=torch.int32, device=device),
        kv_len_arr=torch.tensor(kv_lens, dtype=torch.int32, device=device),
        block_table=bt,
    )


def _reference(batch, q_scale=1.0, kv_scale=1.0):
    """fp32 dequant-then-attend MLA decode reference (query row i of request
    r sees kv positions 0..kv_len_r - q_len + i)."""
    device = batch["q"].device
    q = batch["q"].float() * q_scale
    kvf = batch["kv"].float().reshape(-1, DIMQ) * kv_scale
    kv_lens = batch["kv_len_arr"].tolist()
    bt = batch["block_table"]
    B = len(kv_lens)
    ql = batch["q"].shape[0] // B
    out = torch.empty((B * ql, NH, CKV), dtype=torch.bfloat16, device=device)
    off = torch.arange(PAGE, device=device)
    for r in range(B):
        kvl = kv_lens[r]
        npages = (kvl + PAGE - 1) // PAGE
        flat = (bt[r, :npages].long().unsqueeze(1) * PAGE + off).reshape(-1)[:kvl]
        k = kvf[flat]  # (kvl, 576)
        v = k[:, :CKV]  # (kvl, 512)
        qr = q[r * ql : (r + 1) * ql]  # (ql, 8, 576)
        sc = torch.einsum("qhd,ld->qhl", qr, k) * SM_SCALE
        vis = (kvl - ql) + torch.arange(ql, device=device) + 1
        mask = torch.arange(kvl, device=device).view(1, 1, kvl) >= vis.view(ql, 1, 1)
        sc = sc.masked_fill(mask, float("-inf"))
        p = torch.softmax(sc, -1)
        # A row whose causal window holds zero keys (kvl < ql) is all -inf
        # and softmaxes to NaN; the contract's empty-row semantics (and the
        # stock path's behavior) are exact-zero output rows.
        p = torch.nan_to_num(p, nan=0.0)
        out[r * ql : (r + 1) * ql] = torch.einsum("qhl,ld->qhd", p, v).to(
            torch.bfloat16
        )
    return out


def _plan_wrapper(
    batch, device, fp8q_backend="auto", sm_scale=SM_SCALE, use_profiler=False
):
    ws = torch.zeros(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = BatchMLAPagedAttentionWrapper(
        float_workspace_buffer=ws, backend="auto", fp8q_backend=fp8q_backend
    )
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
        sm_scale=sm_scale,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        use_profiler=use_profiler,
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


SHARED_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"


def test_fp8q_backend_selector_values():
    import pytest as _pytest

    ws = torch.zeros(16, dtype=torch.uint8)
    with _pytest.raises(ValueError):
        BatchMLAPagedAttentionWrapper(
            float_workspace_buffer=ws, backend="auto", fp8q_backend="fa2"
        )


def test_fp8q_backend_stock_never_prepares():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    _reset_mla_dec_fp8q_sm90_stats()
    w = _plan_wrapper(batch, device, fp8q_backend="dequant")
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["plans_selector_dequant"] == 1, stats
    assert stats["plans_prepared"] == 0, stats
    out = _run(w, batch)  # served by dequant + stock
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0, stats
    ref = _reference(batch)
    assert _matched_ratio(out, ref) >= MATCHED


def test_fp8q_backend_specialized_serves_on_contract():
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    _reset_mla_dec_fp8q_sm90_stats()
    w = _plan_wrapper(batch, device, fp8q_backend="specialized")
    out = _run(w, batch)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 1, stats
    ref = _reference(batch)
    assert _matched_ratio(out, ref) >= MATCHED


def test_fp8q_backend_specialized_raises_when_unservable(monkeypatch):
    # an explicitly selected backend is never silently swapped: with the
    # fast path disabled (either kill-switch env), plan must RAISE under
    # fp8q_backend='specialized' instead of dequant-serving
    import pytest as _pytest

    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    _reset_mla_dec_fp8q_sm90_stats()
    monkeypatch.setenv(DISABLE_ENV, "1")
    with _pytest.raises(RuntimeError):
        _plan_wrapper(batch, device, fp8q_backend="specialized")
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["specialized_raises"] == 1, stats


def test_shared_kill_switch_alias(monkeypatch):
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    _reset_mla_dec_fp8q_sm90_stats()
    monkeypatch.setenv(SHARED_DISABLE_ENV, "1")
    w = _plan_wrapper(batch, device)  # auto: alias disables, no raise
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["plans_disabled"] == 1, stats
    out = _run(w, batch)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0, stats
    ref = _reference(batch)
    assert _matched_ratio(out, ref) >= MATCHED


# --------------------------------------------------------------------------
# Review regressions
# --------------------------------------------------------------------------


def test_fp8q_backend_specialized_raises_on_build_failure(monkeypatch):
    """A failed module build/load must raise under fp8q_backend='specialized'
    (no silent substitution) and disclose + dequant-serve under 'auto'."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    import flashinfer.mla._batch_mla._decode_fp8_sm90 as fp8q_mod

    monkeypatch.setattr(fp8q_mod, "_get_module", lambda: None)
    _reset_mla_dec_fp8q_sm90_stats()
    with pytest.raises(RuntimeError, match="failed to build"):
        _plan_wrapper(batch, device, fp8q_backend="specialized")
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["specialized_raises"] == 1, stats

    _reset_mla_dec_fp8q_sm90_stats()
    w = _plan_wrapper(batch, device, fp8q_backend="auto")
    out = _run(w, batch)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0, stats
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_nonpositive_scales_never_reach_the_kernel():
    """q_scale<=0 (or a nonfinite descale) would turn the kernel's -inf score
    masks into NaN through the fused qk scale; such calls must be served by
    the exact dequant path ('auto') or raise ('specialized')."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    w = _plan_wrapper(batch, device)
    out = _run(w, batch, q_scale=0.0)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0, stats
    assert torch.isfinite(out.float()).all()
    assert _matched_ratio(out, _reference(batch, q_scale=0.0)) >= MATCHED

    w2 = _plan_wrapper(batch, device, fp8q_backend="specialized")
    with pytest.raises(RuntimeError, match="finite positive"):
        _run(w2, batch, q_scale=0.0)


@pytest.mark.parametrize("sm_scale", [0.0, -SM_SCALE, float("inf")])
def test_nonpositive_sm_scale_is_ineligible(sm_scale):
    """The kernels reduce max(score) BEFORE applying the fused qk scale,
    which is only order-preserving for a positive sm_scale (and a zero scale
    would turn -inf masks into NaN); such plans stay on the stock path
    ('auto') or raise ('specialized'). What the stock path then does with the
    degenerate scale is its own (upstream) behavior and is not asserted here:
    plan() itself rejects nonfinite scales, and the stock kernels produce NaN
    at sm_scale<=0."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    _reset_mla_dec_fp8q_sm90_stats()
    if sm_scale == float("inf"):
        # The stock plan itself rejects nonfinite scales (pre-existing
        # _MLAPlanArguments validation); the fast path was never prepared.
        with pytest.raises(ValueError, match="finite"):
            _plan_wrapper(batch, device, sm_scale=sm_scale)
        stats = _mla_dec_fp8q_sm90_stats()
        assert stats["plans_prepared"] == 0, stats
    else:
        w = _plan_wrapper(batch, device, sm_scale=sm_scale)
        stats = _mla_dec_fp8q_sm90_stats()
        assert stats["plans_prepared"] == 0, stats
        _run(w, batch)  # served by the stock path; its numerics are its own
        stats = _mla_dec_fp8q_sm90_stats()
        assert stats["runs_dispatched"] == 0, stats

    with pytest.raises(RuntimeError, match="finite positive"):
        _plan_wrapper(batch, device, fp8q_backend="specialized", sm_scale=sm_scale)


def test_profiled_plans_stay_on_stock_path():
    """The specialized dispatch does not receive profiler_buffer, so a
    use_profiler=True plan must not be served by the fast path (which would
    silently produce no profiler records): stock for 'auto', raise for
    'specialized'."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    _reset_mla_dec_fp8q_sm90_stats()
    _plan_wrapper(batch, device, use_profiler=True)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0, stats

    with pytest.raises(RuntimeError, match="use_profiler"):
        _plan_wrapper(batch, device, fp8q_backend="specialized", use_profiler=True)


def test_misaligned_query_views_fall_back():
    """A contiguous offset view can put the packed query base at a
    non-16-byte-aligned address, which the kernel's uint4 query loads do not
    support; such calls are dequant-served ('auto') or raise ('specialized')."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    TQ = batch["q"].shape[0]
    buf = torch.empty(TQ * NH * DIMQ + 16, dtype=torch.float8_e4m3fn, device=device)
    q_mis = buf[1 : 1 + TQ * NH * DIMQ].view(TQ, NH, DIMQ)
    assert q_mis.data_ptr() % 16 != 0
    q_mis.copy_(batch["q"])
    kv = batch["kv"]

    _reset_mla_dec_fp8q_sm90_stats()
    w = _plan_wrapper(batch, device)
    out = w.run(
        query=(q_mis[..., :CKV], q_mis[..., CKV:]),
        kv_cache=(kv[..., :CKV], kv[..., CKV:]),
        q_scale=1.0,
        ckv_scale=1.0,
        kpe_scale=1.0,
    )
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0, stats
    assert stats["runs_dequant_fallback"] >= 1, stats
    assert _matched_ratio(out, _reference(batch)) >= MATCHED

    w2 = _plan_wrapper(batch, device, fp8q_backend="specialized")
    with pytest.raises(RuntimeError, match="aligned"):
        w2.run(
            query=(q_mis[..., :CKV], q_mis[..., CKV:]),
            kv_cache=(kv[..., :CKV], kv[..., CKV:]),
            q_scale=1.0,
            ckv_scale=1.0,
            kpe_scale=1.0,
        )


def test_specialized_path_enforces_run_contract():
    """The specialized return path honors the same planned run contract as
    the stock path: a missing kpe_scale and options the plan does not carry
    (o_scale, BMM scales) are rejected, never silently ignored."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    w = _plan_wrapper(batch, device)
    q, kv = batch["q"], batch["kv"]
    qs = (q[..., :CKV], q[..., CKV:])
    kvs = (kv[..., :CKV], kv[..., CKV:])
    with pytest.raises(ValueError, match="kpe_scale"):
        w.run(query=qs, kv_cache=kvs, q_scale=1.0, ckv_scale=1.0)
    with pytest.raises(ValueError):
        w.run(
            query=qs,
            kv_cache=kvs,
            q_scale=1.0,
            ckv_scale=1.0,
            kpe_scale=1.0,
            o_scale=1.0,
        )
    with pytest.raises(ValueError):
        w.run(query=qs, kv_cache=kvs, q_scale=1.0, bmm1_scale=1.0, bmm2_scale=1.0)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 0


def test_failed_plan_does_not_publish_specialized_state(monkeypatch):
    """A plan the backend rejects must not leave fresh specialized state next
    to the previous backend plan; run() then refuses until the next
    successful plan instead of mixing the two."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    w = _plan_wrapper(batch, device)
    assert w._mla_dec_fp8q_sm90_state is not None

    class _Boom:
        @classmethod
        def plan_from_wrapper(cls, args):
            raise RuntimeError("backend planning failed")

    monkeypatch.setattr(w, "_backend_type", _Boom)
    batch2 = _make_batch([4097], device)
    with pytest.raises(RuntimeError, match="backend planning failed"):
        w.plan(
            qo_indptr=batch2["qo_indptr"],
            kv_indptr=batch2["kv_indptr"],
            kv_indices=batch2["kv_indices"],
            kv_len_arr=batch2["kv_len_arr"],
            num_heads=NH,
            head_dim_ckv=CKV,
            head_dim_kpe=KPE,
            page_size=PAGE,
            causal=True,
            sm_scale=SM_SCALE,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
        )
    assert w._mla_dec_fp8q_sm90_state is None
    assert w._mla_dec_fp8q_sm90_requested is False
    # With the route torn down, q_scale would be silently ignored, so it is
    # rejected up front ...
    with pytest.raises(ValueError, match="q_scale is only supported"):
        _run(w, batch2)
    # ... and without q_scale the call still refuses against the previous
    # (bf16-query) plan instead of mixing the two plans' state.
    with pytest.raises(ValueError, match="does not match the planned"):
        _run(w, batch2, q_scale=None)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 0


def test_dequant_query_preserves_structural_forms():
    """GPU-free: the dequant fallback must keep an adjacent split pair
    adjacent (one packed dequantisation, not two independent copies, so a
    packed-layout plan still accepts it zero-copy) and must accept the
    redundant dual form without crashing on its nested tuple."""
    from flashinfer.mla._batch_mla._contracts import _are_adjacent_last_dim_views
    from flashinfer.mla._batch_mla._decode_fp8_sm90 import _dequant_query

    q = torch.full((QL, NH, DIMQ), 0.5).to(torch.float8_e4m3fn)
    qn, qp = q[..., :CKV], q[..., CKV:]

    dn, dp = _dequant_query((qn, qp), 2.0)
    assert dn.dtype == dp.dtype == torch.bfloat16
    assert _are_adjacent_last_dim_views(dn, dp)
    assert torch.all(dn.float() == 1.0)

    dual = _dequant_query((q, (qn, qp)), 1.0)
    dpk, (dn2, dp2) = dual
    assert dpk.dtype == torch.bfloat16 and tuple(dpk.shape) == (QL, NH, DIMQ)
    assert _are_adjacent_last_dim_views(dn2, dp2)

    # independent split pairs stay independent (and still dequantise)
    in0, in1 = _dequant_query((qn.contiguous(), qp.contiguous()), 1.0)
    assert not _are_adjacent_last_dim_views(in0, in1)


def test_dual_structural_query_dequant_fallback(monkeypatch):
    """The redundant (packed, (nope, pe)) query form must be served by the
    dequant fallback instead of crashing on the nested tuple."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    monkeypatch.setenv(DISABLE_ENV, "1")
    w = _plan_wrapper(batch, device)
    q, kv = batch["q"], batch["kv"]
    out = w.run(
        query=(q, (q[..., :CKV], q[..., CKV:])),
        kv_cache=(kv[..., :CKV], kv[..., CKV:]),
        q_scale=1.0,
        ckv_scale=1.0,
        kpe_scale=1.0,
    )
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 0
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_graph_plan_never_reallocates_prepared_buffers():
    """A CUDA-graph wrapper replan whose geometry outgrows the prepared
    buffers must not reallocate them (a captured graph may hold their
    addresses): it is ineligible under 'auto' and raises under
    'specialized'."""
    _sm90_or_skip()
    from flashinfer.mla._batch_mla import _decode_fp8_sm90 as fp8q_mod
    from flashinfer.mla._batch_mla._contracts import MLAPlanMetadata

    device = torch.device("cuda:0")

    class _W:
        pass

    w = _W()
    w.device = device
    w._use_cuda_graph = True
    w._fp8q_backend = "auto"
    b1 = _make_batch([2983], device)
    md1 = MLAPlanMetadata.csr(
        b1["qo_indptr"], b1["kv_indptr"], b1["kv_indices"], b1["kv_len_arr"]
    )
    fp8q_mod._mla_dec_fp8q_sm90_prepare(
        w, md1, NH, CKV, KPE, PAGE, True, SM_SCALE, "none", False, False, None
    )
    st = w._mla_dec_fp8q_sm90_state
    assert st is not None
    ptrs = {name: t.data_ptr() for name, t in w._mla_dec_fp8q_sm90_bufs.items()}

    b2 = _make_batch([8192], device)
    md2 = MLAPlanMetadata.csr(
        b2["qo_indptr"], b2["kv_indptr"], b2["kv_indices"], b2["kv_len_arr"]
    )
    fp8q_mod._mla_dec_fp8q_sm90_prepare(
        w, md2, NH, CKV, KPE, PAGE, True, SM_SCALE, "none", False, False, None
    )
    assert w._mla_dec_fp8q_sm90_state is None  # ineligible, not reallocated
    for name, t in w._mla_dec_fp8q_sm90_bufs.items():
        assert t.data_ptr() == ptrs[name], name

    w._fp8q_backend = "specialized"
    with pytest.raises(RuntimeError, match="captured buffer addresses"):
        fp8q_mod._mla_dec_fp8q_sm90_prepare(
            w, md2, NH, CKV, KPE, PAGE, True, SM_SCALE, "none", False, False, None
        )


def test_captured_scales_survive_later_eager_scale_changes():
    """The scale writes are recorded inside the capture, so a later eager
    call with different scales cannot retroactively change what a replay
    computes, and eager calls stay correct after replays."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([2983], device)
    w = _plan_wrapper(batch, device)
    out = torch.empty((QL, NH, CKV), dtype=torch.bfloat16, device=device)

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(w, batch, q_scale=0.5, kv_scale=2.0, out=out)  # warmup
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        _run(w, batch, q_scale=0.5, kv_scale=2.0, out=out)

    eager = _run(w, batch, q_scale=1.0, kv_scale=1.0)  # mutates the buffers
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert _matched_ratio(out, _reference(batch, q_scale=0.5, kv_scale=2.0)) >= MATCHED

    # an eager call after a replay must also refill the buffers
    eager2 = _run(w, batch, q_scale=1.0, kv_scale=1.0)
    torch.cuda.synchronize()
    assert torch.equal(eager2, eager)
    assert _matched_ratio(eager2, _reference(batch)) >= MATCHED


@pytest.mark.parametrize("ql", [2, 4, 6])
def test_keyless_causal_rows_match_stock(ql):
    """kv_len < q_len leaves causal query rows with ZERO visible keys. The
    kernels clamp each row's key limit to >= 1 (they would attend such rows
    to key 0 and emit a value), so these plans must decline to the stock
    path, whose empty-row semantics are exact-zero output rows: with a zero
    fp8 query and unit fp8 KV, kv_len=1 must produce [0, ..., 0, 1] per
    channel, never all ones."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([1], device, ql=ql)
    batch["q"] = torch.zeros_like(batch["q"].float()).to(torch.float8_e4m3fn)
    batch["kv"] = torch.ones_like(batch["kv"].float()).to(torch.float8_e4m3fn)
    w = _plan_wrapper(batch, device)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["plans_prepared"] == 0 and stats["plans_ineligible"] == 1, stats

    out = _run(w, batch)
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["runs_dispatched"] == 0 and stats["runs_dequant_fallback"] >= 1
    out = out.float()
    # rows 0..ql-2 have empty causal windows: exact zeros, like stock
    assert torch.all(out[: ql - 1] == 0.0), out[:, 0, 0]
    # the final row sees exactly key 0: uniform softmax over the unit value
    assert torch.all(out[ql - 1 :] == 1.0), out[:, 0, 0]
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_kv_len_equal_to_q_len_stays_on_fast_path():
    """The keyless-row guard declines only kv_len < q_len; the kv_len == q_len
    boundary (every row has >= 1 visible key) stays specialized and exact."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([QL], device)
    w = _plan_wrapper(batch, device)
    assert _mla_dec_fp8q_sm90_stats()["plans_prepared"] == 1
    out = _run(w, batch)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def _graph_wrapper_and_capture(batch, device):
    """Plan a use_cuda_graph wrapper on `batch`, capture one run into a
    caller-owned output buffer, and return (wrapper, graph, out)."""
    ws = torch.zeros(160 * 1024 * 1024, dtype=torch.uint8, device=device)
    B = batch["kv_len_arr"].numel()
    w = BatchMLAPagedAttentionWrapper(
        float_workspace_buffer=ws,
        use_cuda_graph=True,
        qo_indptr=batch["qo_indptr"].clone(),
        kv_indptr=batch["kv_indptr"].clone(),
        kv_indices=batch["kv_indices"].clone(),
        kv_len_arr=batch["kv_len_arr"].clone(),
        backend="auto",
        fp8q_backend="auto",
    )
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
    out = torch.empty((B * QL, NH, CKV), dtype=torch.bfloat16, device=device)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run(w, batch, out=out)  # warmup on the side stream
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        _run(w, batch, out=out)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched_capturing"] == 1
    return w, graph, out


def test_successful_replan_refreshes_captured_metadata():
    """A successful replan at the captured geometry must refresh the SAME
    buffers the capture reads (contents in place, never a rebinding the
    capture cannot see): after replanning with permuted page IDs and a
    shorter kv_len through the same four-page table width, a replay of the
    old capture must compute the NEW plan, bitwise equal to an eager call."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([250], device)  # four pages, identity page order
    w, graph, out = _graph_wrapper_and_capture(batch, device)

    graph.replay()
    torch.cuda.synchronize()
    assert _matched_ratio(out, _reference(batch)) >= MATCHED

    # successful replan: same B/QL/table width, permuted pages, shorter kv
    new_indices = torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=device)
    new_len = torch.tensor([200], dtype=torch.int32, device=device)
    w.plan(
        qo_indptr=batch["qo_indptr"],
        kv_indptr=batch["kv_indptr"],
        kv_indices=new_indices,
        kv_len_arr=new_len,
        num_heads=NH,
        head_dim_ckv=CKV,
        head_dim_kpe=KPE,
        page_size=PAGE,
        causal=True,
        sm_scale=SM_SCALE,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
    )
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["replans_refreshed_captured"] == 1, stats

    batch2 = dict(batch)
    batch2["kv_len_arr"] = new_len
    batch2["block_table"] = new_indices.view(1, 4)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert _matched_ratio(out, _reference(batch2)) >= MATCHED
    eager = _run(w, batch)  # same metadata buffers: bitwise-equal kernel run
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_unrefreshable_replan_leaves_captured_plan_intact():
    """A replan the captured buffers cannot serve (outside the fast-path
    contract at unchanged geometry) leaves their contents untouched: by the
    documented contract it INVALIDATES prior captures, whose replays keep
    computing the PREVIOUS plan's values intact — never a silent mix of old
    page IDs with new lengths."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([250], device)
    w, graph, out = _graph_wrapper_and_capture(batch, device)
    eager_before = _run(w, batch).clone()

    # replan outside the fast-path contract (lse_mode, the route-switch
    # probe) at the captured geometry: not prepared, buffers untouched
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
        lse_mode="basee",
    )
    stats = _mla_dec_fp8q_sm90_stats()
    assert stats["replans_refreshed_captured"] == 0, stats
    assert w._mla_dec_fp8q_sm90_state is None

    # the (contract-invalidated) capture still computes its own plan's
    # values bitwise, not a mix with the rejected-route replan
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_before)


def test_rejected_argument_validation_replan_restores_captured_metadata():
    """The rollback transaction must start before plan-argument validation:
    an unsupported scale_mode replan raises in _MLAPlanArguments AFTER the
    fp8-Q prepare already refreshed the captured buffers in place, so the
    snapshot must be restored — the old capture's metadata stays bitwise
    intact and its replay unchanged (the reviewer's original repro)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([250], device)  # four pages, identity page order
    w, graph, out = _graph_wrapper_and_capture(batch, device)
    eager_before = _run(w, batch).clone()
    bufs = w._mla_dec_fp8q_sm90_bufs
    saved = {
        name: bufs[name].clone() for name in ("block_table", "seq_lens") if name in bufs
    }
    assert saved, bufs.keys()

    with pytest.raises(ValueError):
        w.plan(
            qo_indptr=batch["qo_indptr"],
            kv_indptr=batch["kv_indptr"],
            kv_indices=torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=device),
            kv_len_arr=torch.tensor([200], dtype=torch.int32, device=device),
            num_heads=NH,
            head_dim_ckv=CKV,
            head_dim_kpe=KPE,
            page_size=PAGE,
            causal=True,
            sm_scale=SM_SCALE,
            q_data_type=torch.float8_e4m3fn,
            kv_data_type=torch.float8_e4m3fn,
            scale_mode="invalid-review-probe",
        )
    # the captured metadata buffers are bitwise intact ...
    for name, before in saved.items():
        assert torch.equal(bufs[name], before), name
    # ... and the old capture replays its own plan bitwise
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager_before)


def test_dual_device_smoke():
    """Each device arms its own dynamic-smem opt-in (per-device
    std::call_once): the same warp-group launch class must run correctly on
    a second device, not inherit the first device's one-time opt-in."""
    _sm90_or_skip()
    if torch.cuda.device_count() < 2:
        pytest.skip("two GPUs required")
    if torch.cuda.get_device_capability(1) != (9, 0):
        pytest.skip("second SM90 device required")
    for dev in ("cuda:0", "cuda:1"):
        device = torch.device(dev)
        batch = _make_batch([2983], device)
        w = _plan_wrapper(batch, device)
        out = _run(w, batch)
        torch.cuda.synchronize(device)
        assert _matched_ratio(out, _reference(batch)) >= MATCHED
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 2


@pytest.mark.parametrize(
    "ql,kv_lens",
    [
        (1, [67]),
        (2, [131, 67]),
        (6, [131]),
        (6, [1500, 658, 760, 341, 863]),
    ],
)
def test_partial_page_garbage_rows_never_poison_output(ql, kv_lens):
    """Rows past each sequence's length in a partially-filled page are never
    written by the cache; an fp8 NaN byte there must not reach the output
    (their probabilities are exactly 0, but 0 * NaN = NaN)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch(kv_lens, device, ql=ql)
    ref = _reference(batch)
    kv = batch["kv"]
    s = 0
    for L in kv_lens:
        ppr = (L + PAGE - 1) // PAGE
        tail = L - (ppr - 1) * PAGE
        if tail < PAGE:
            kv[s + ppr - 1, tail:, :] = float("nan")
        s += ppr
    w = _plan_wrapper(batch, device)
    out = _run(w, batch)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert torch.isfinite(out.float()).all()
    assert _matched_ratio(out, ref) >= MATCHED


def test_deep_shallow_splits_use_fp32_pv_accumulation():
    """The MT<=2 split kernels have no nonfinite repair path, so deep splits
    must not use the packed-FP16 PV accumulator: runs of near-max same-sign
    KV rows overflow FP16 beyond ~146 tokens per split, and the fp8 partial
    write then saturates the Inf to 448 where the true balanced average is
    ~0."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([38400], device, ql=1)  # 600 pages, MT=1 split path
    batch["q"] = torch.zeros((1, NH, DIMQ), device=device).to(torch.float8_e4m3fn)
    pages = torch.arange(600, device=device)
    sign = torch.where((pages // 3) % 2 == 0, 448.0, -448.0)
    batch["kv"] = (sign.view(600, 1, 1).expand(batch["kv"].shape).contiguous()).to(
        torch.float8_e4m3fn
    )
    w = _plan_wrapper(batch, device)
    out = _run(w, batch)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert torch.isfinite(out.float()).all()
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_b5_one_page_splits_above_256_are_capped():
    """Regression: the B>=5 one-page-split planner must cap the split count
    at the dispatcher's 256-split workspace bound (a ragged batch whose
    longest sequence nearly fills the page pool otherwise plans ns > 256 and
    run() refuses an eligible call fail-closed)."""
    _sm90_or_skip()
    device = torch.device("cuda:0")
    batch = _make_batch([257 * PAGE, 64, 64, 64, 64], device)
    w = _plan_wrapper(batch, device)
    out = _run(w, batch)
    assert _mla_dec_fp8q_sm90_stats()["runs_dispatched"] == 1
    assert _matched_ratio(out, _reference(batch)) >= MATCHED


def test_empty_splits_never_read_uninitialized_partials():
    """A dense page table wider than the live pages (the padded-capacity
    dense layout; CSR plans reject padding) leaves planned splits empty.
    Their workspace partials are never written, so the combine must skip
    them outright: their weight is exactly 0, but fmaf(0, NaN) would poison
    the reduction."""
    _sm90_or_skip()
    from flashinfer.mla._batch_mla import _decode_fp8_sm90 as fp8q_mod
    from flashinfer.mla._batch_mla._contracts import MLAPlanMetadata

    device = torch.device("cuda:0")
    batch = _make_batch([128 * PAGE], device)  # 128-page pool and table
    batch["kv_len_arr"] = torch.tensor([2983], dtype=torch.int32, device=device)
    ref = _reference(batch)

    class _W:
        pass

    w = _W()
    w.device = device
    w._use_cuda_graph = False
    w._fp8q_backend = "auto"
    md = MLAPlanMetadata.dense(
        cum_seq_lens_q=batch["qo_indptr"],
        block_tables=batch["block_table"],
        seq_lens=batch["kv_len_arr"],
    )
    fp8q_mod._mla_dec_fp8q_sm90_prepare(
        w, md, NH, CKV, KPE, PAGE, True, SM_SCALE, "none", False, False, None
    )
    st = w._mla_dec_fp8q_sm90_state
    assert st is not None and st.pps == 128
    st.workspace.fill_(0x7F)  # e4m3 NaN byte pattern
    out, _ = fp8q_mod._mla_dec_fp8q_sm90_dispatch(
        w, batch["q"], batch["kv"], None, None, False, 1.0, 1.0, None, 1.0, None, None
    )
    assert out is not None
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert _matched_ratio(out, ref) >= MATCHED
