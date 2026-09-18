"""The cuDNN ragged prefill graph is built once per length class with cuDNN's
execute-time shape override and fed the real (batch, max_seq_len) on every call,
so a serving stream of changing shapes builds no new execution plans (one build
is 55-70 ms). These tests pin the cache-shape function, count graph builds across
a stream of shapes, and check numerics against FA2 in both length classes.
"""

import warnings

import pytest
import torch

import flashinfer
from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.cudnn.prefill import (
    _OVERRIDE_CACHE_BATCH,
    _OVERRIDE_CACHE_SEQ_LONG,
    _OVERRIDE_SHORT_SEQ,
    _override_cache_shape,
)
from flashinfer.utils import is_sm100a_supported


B, S128, SL = _OVERRIDE_CACHE_BATCH, _OVERRIDE_SHORT_SEQ, _OVERRIDE_CACHE_SEQ_LONG


@pytest.mark.parametrize(
    "b,s_q,s_kv,expected",
    [
        (3, 4, 4, (B, S128, S128)),
        (64, 128, 128, (B, S128, S128)),
        (64, 129, 128, (B, SL, S128)),
        (64, 4, 4096, (B, S128, SL)),  # short q, long kv: classed separately
        (16, 1000, 4096, (B, SL, SL)),
        (16, 70000, 70000, (B, 131072, 131072)),
        (5000, 1000, 1000, (8192, SL, SL)),
        (7, 1, 256, (B, 1, SL)),  # s_q == 1 is its own class (cuDNN decode kernel)
        (7, 1, 1, (B, 1, S128)),
    ],
)
def test_override_cache_shape(b, s_q, s_kv, expected):
    assert _override_cache_shape(b, s_q, s_kv) == expected


def test_workspace_memo_is_keyed_by_graph_key_not_object_identity():
    # A graph object can be evicted from the FE cache and its id() reused by a
    # later graph with a different declared shape; the memo therefore keys on
    # the graph-cache key (a pure function of the declared shape).
    class FakeGraph:
        def __init__(self, size):
            self.size = size

        def get_workspace_size(self):
            return self.size

    cudnn_prefill._graph_workspace_bytes.clear()
    short_key, long_key = ("short",), ("long",)
    g_short = FakeGraph(0)
    assert cudnn_prefill._graph_workspace_size(g_short, short_key) == 0
    del g_short
    g_long = FakeGraph(1_065_728)  # may well get the same id() as g_short
    assert cudnn_prefill._graph_workspace_size(g_long, long_key) == 1_065_728
    assert (
        cudnn_prefill._graph_workspace_size(FakeGraph(999), short_key) == 0
    )  # memo hit by key
    cudnn_prefill._graph_workspace_bytes.clear()


def _indptr(lens):
    t = torch.zeros(len(lens) + 1, dtype=torch.int32)
    t[1:] = torch.cumsum(torch.tensor(lens, dtype=torch.int32), 0)
    return t


def _cudnn_override_available():
    if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
        return False
    return (
        cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16)
        and cudnn_prefill._cudnn_version_supports_shape_override()
    )


requires_override = pytest.mark.skipif(
    not _cudnn_override_available(),
    reason="needs SM100, cuDNN>=9.24 and cudnn-frontend>=1.29",
)


def _run(
    backend,
    lens,
    hq,
    hkv,
    d,
    causal=True,
    ws=None,
    dvo=None,
    kv_lens=None,
    return_lse=True,
):
    dvo = d if dvo is None else dvo
    indptr = _indptr(lens)
    kv_indptr = _indptr(kv_lens) if kv_lens is not None else indptr
    T, Tk = int(indptr[-1]), int(kv_indptr[-1])
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(T, hq, d, dtype=torch.bfloat16, device="cuda", generator=g)
    k = torch.randn(Tk, hkv, d, dtype=torch.bfloat16, device="cuda", generator=g)
    v = torch.randn(Tk, hkv, dvo, dtype=torch.bfloat16, device="cuda", generator=g)
    ws = (
        ws
        if ws is not None
        else torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    )
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=backend)
    w.plan(
        indptr,
        kv_indptr,
        hq,
        hkv,
        d,
        head_dim_vo=dvo,
        causal=causal,
        q_data_type=torch.bfloat16,
    )
    if not return_lse:
        return w.run(q, k, v), None
    return w.run(q, k, v, return_lse=True)


def _assert_matches_fa2(
    lens,
    hq,
    hkv,
    d,
    causal=True,
    dvo=None,
    kv_lens=None,
    return_lse=True,
    cudnn_ws=None,
):
    kw = dict(dvo=dvo, kv_lens=kv_lens, return_lse=return_lse)
    o_c, lse_c = _run("cudnn", lens, hq, hkv, d, causal, ws=cudnn_ws, **kw)
    o_f, lse_f = _run("fa2", lens, hq, hkv, d, causal, **kw)
    assert o_c.shape == o_f.shape
    torch.testing.assert_close(o_c.float(), o_f.float(), atol=2e-2, rtol=2e-2)
    if not return_lse:
        return
    assert lse_c.shape == lse_f.shape
    finite = torch.isfinite(lse_f)
    assert torch.equal(torch.isfinite(lse_c), finite)
    torch.testing.assert_close(lse_c[finite], lse_f[finite], atol=5e-3, rtol=5e-3)


@requires_override
def test_shape_stream_builds_one_graph_per_length_class():
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    hq, hkv, d = 32, 8, 128
    torch.manual_seed(7)
    _run("cudnn", [50, 166, 7], hq, hkv, d, ws=ws)  # long class, may build
    _run("cudnn", [4] * 9, hq, hkv, d, ws=ws)  # short class, may build
    before = cudnn_prefill._prefill_graph_builds
    for _ in range(12):
        b = int(torch.randint(1, 40, (1,)))
        lens = torch.randint(1, 2048, (b,)).tolist()
        _run("cudnn", lens, hq, hkv, d, ws=ws)
        lens = [int(torch.randint(1, _OVERRIDE_SHORT_SEQ + 1, (1,)))] * int(
            torch.randint(1, 200, (1,))
        )
        _run("cudnn", lens, hq, hkv, d, ws=ws)
    assert cudnn_prefill._prefill_graph_builds == before, (
        "a new (batch, max_len) rebuilt the cuDNN graph"
    )


@requires_override
def test_frost_opt_in_needs_cudnn_frontend_1_30(monkeypatch):
    """With cudnn-frontend's FROST engines opted in, the override graph needs a frontend whose FROST
    execute honors overrides (1.30+); below that the exact-shape graph is used and one warning says why."""
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    monkeypatch.setattr(cudnn_prefill, "_warned_frost_override", False)
    monkeypatch.setattr(cudnn_prefill.cudnn, "__version__", "1.29.0")
    cudnn_prefill._cudnn_frontend_version.cache_clear()
    try:
        with pytest.warns(UserWarning, match="predates 1.30"):
            assert cudnn_prefill._cudnn_supports_shape_override() is False
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the warning is emitted once per process
            assert cudnn_prefill._cudnn_supports_shape_override() is False
        monkeypatch.setattr(cudnn_prefill.cudnn, "__version__", "1.30.0")
        cudnn_prefill._cudnn_frontend_version.cache_clear()
        assert cudnn_prefill._cudnn_supports_shape_override() is True
        monkeypatch.delenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES")
        monkeypatch.setattr(cudnn_prefill.cudnn, "__version__", "1.29.0")
        cudnn_prefill._cudnn_frontend_version.cache_clear()
        assert (
            cudnn_prefill._cudnn_supports_shape_override() is True
        )  # backend plans honor overrides on 1.29
    finally:
        cudnn_prefill._cudnn_frontend_version.cache_clear()


@requires_override
def test_exact_declaration_rebuilds_per_shape(monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUDNN_PREFILL_SHAPE_OVERRIDE", "0")
    hq, hkv, d = 32, 8, 128
    _run("cudnn", [50, 166, 7], hq, hkv, d)
    before = cudnn_prefill._prefill_graph_builds
    _run("cudnn", [50, 166, 7, 9], hq, hkv, d)
    assert cudnn_prefill._prefill_graph_builds == before + 1
    _assert_matches_fa2([50, 166, 7, 9], hq, hkv, d)


@requires_override
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "lens",
    [
        [50, 166, 7],  # long class
        [4] * 13,  # short class (short-row engine)
        [128] * 3 + [1],  # short class upper edge
        [1, 777, 0, 300],  # zero-length row
        [2000, 5, 1500],
    ],
)
def test_override_matches_fa2(lens, causal):
    _assert_matches_fa2(lens, 32, 8, 128, causal)


@requires_override
def test_override_matches_fa2_without_lse():
    _assert_matches_fa2([50, 166, 7], 32, 8, 128, return_lse=False)
    _assert_matches_fa2([4] * 13, 32, 8, 128, return_lse=False)


@requires_override
@pytest.mark.parametrize("q_len", [1, 4, 100])
def test_short_q_long_kv_rows(q_len):
    # Chunked-prefix / verify style: few new query tokens against a long kv
    # chunk (kv_indptr != qo_indptr, non-causal). q_len == 1 lands on cuDNN's
    # decode kernel class, which an override graph cannot cross into.
    hq, hkv, d = 32, 8, 128
    b = 9
    lens, kv_lens = [q_len] * b, [4096, 300, 1, 2048, 777, 4000, 64, 4095, 129]
    # q_len == 1 with GQA: cuDNN's decode kernel writes the ragged Stats (LSE)
    # only for the first head of each kv group (output is correct). That is a
    # cuDNN bug independent of shape override (the exact-shape graph shows the
    # same values), so the LSE is not compared for this class.
    _assert_matches_fa2(
        lens, hq, hkv, d, causal=False, kv_lens=kv_lens, return_lse=(q_len != 1)
    )
    # a stream of such steps builds nothing new
    before = cudnn_prefill._prefill_graph_builds
    for _ in range(4):
        b2 = int(torch.randint(1, 30, (1,)))
        _run(
            "cudnn",
            [q_len] * b2,
            hq,
            hkv,
            d,
            causal=False,
            kv_lens=torch.randint(
                129, 4096, (b2,)
            ).tolist(),  # stays in the long-kv class
            return_lse=(q_len != 1),  # same graph family as the parity call above
        )
    assert cudnn_prefill._prefill_graph_builds == before


@requires_override
def test_small_workspace_falls_back_to_exact_shape():
    # The override graph declared at batch 4096 needs ~1 MiB of workspace for
    # TMA descriptors; a caller with less gets the exact-shape graph (one build
    # per shape, as before) and the same numbers.
    hq, hkv, d = 32, 8, 128
    ws = torch.empty(256 * 1024, dtype=torch.uint8, device="cuda")
    # shapes no other test in this file uses, so the exact graphs are fresh builds
    _assert_matches_fa2(
        [51, 167, 8], hq, hkv, d, cudnn_ws=ws
    )  # may build the override graph too
    before = cudnn_prefill._prefill_graph_builds
    _assert_matches_fa2([51, 167, 8, 10], hq, hkv, d, cudnn_ws=ws)
    _assert_matches_fa2([51, 167, 8, 10, 12], hq, hkv, d, cudnn_ws=ws)
    assert (
        cudnn_prefill._prefill_graph_builds == before + 2
    )  # exact graphs, one per shape


@requires_override
def test_cuda_graph_capture_and_replay():
    hq, hkv, d = 32, 8, 128
    lens = [300, 17, 1200]
    indptr = _indptr(lens).cuda()
    T = int(indptr[-1])
    q = torch.randn(T, hq, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(T, hkv, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(T, hkv, d, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws,
        "NHD",
        backend="cudnn",
        use_cuda_graph=True,
        qo_indptr_buf=indptr.clone(),
        kv_indptr_buf=indptr.clone(),
    )
    w.plan(
        indptr,
        indptr,
        hq,
        hkv,
        d,
        head_dim_vo=d,
        causal=True,
        q_data_type=torch.bfloat16,
    )
    out_eager, lse_eager = w.run(q, k, v, return_lse=True)  # builds outside capture
    out_cap = torch.empty_like(out_eager)
    lse_cap = torch.empty_like(lse_eager)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        w.run(
            q, k, v, return_lse=True, out=out_cap, lse=lse_cap
        )  # warm-up on the capture stream
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        w.run(q, k, v, return_lse=True, out=out_cap, lse=lse_cap)
    out_cap.zero_()
    lse_cap.zero_()
    q.copy_(torch.randn_like(q))  # new inputs, same shapes
    wf = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend="fa2")
    wf.plan(
        indptr,
        indptr,
        hq,
        hkv,
        d,
        head_dim_vo=d,
        causal=True,
        q_data_type=torch.bfloat16,
    )
    out_ref, lse_ref = wf.run(q, k, v, return_lse=True)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_cap.float(), out_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse_cap, lse_ref, atol=5e-3, rtol=5e-3)


@requires_override
def test_override_matches_fa2_mla_dims():
    # DeepSeek-style prefill: 128 MHA heads, d_qk=192, d_vo=128 (V strides differ from Q/K).
    _assert_matches_fa2([700, 33, 1200], 128, 128, 192, dvo=128)


@requires_override
def test_override_handles_strided_views():
    hq, d = 32, 128
    lens = [777, 1, 1500, 96]
    indptr = _indptr(lens)
    T = int(indptr[-1])
    qkv = torch.randn(T, 3, hq, d, dtype=torch.bfloat16, device="cuda")
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    outs = {}
    for backend, (qq, kk, vv) in (
        ("cudnn", (q, k, v)),
        ("fa2", (q.contiguous(), k.contiguous(), v.contiguous())),
    ):
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", backend=backend)
        w.plan(
            indptr,
            indptr,
            hq,
            hq,
            d,
            head_dim_vo=d,
            causal=True,
            q_data_type=torch.bfloat16,
        )
        outs[backend] = w.run(qq, kk, vv, return_lse=True)
    torch.testing.assert_close(
        outs["cudnn"][0].float(), outs["fa2"][0].float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(outs["cudnn"][1], outs["fa2"][1], atol=5e-3, rtol=5e-3)


@requires_override
def test_cache_shape_growth_when_exceeded(monkeypatch):
    # A row longer than the long cache shape moves to a bigger power-of-two cache
    # shape (one more build), and stays correct.
    monkeypatch.setattr(cudnn_prefill, "_OVERRIDE_CACHE_SEQ_LONG", 512)
    hq, hkv, d = 32, 8, 128
    _run("cudnn", [300, 400], hq, hkv, d)
    before = cudnn_prefill._prefill_graph_builds
    _assert_matches_fa2([300, 900], hq, hkv, d)
    assert cudnn_prefill._prefill_graph_builds == before + 1
    _run("cudnn", [1000, 20], hq, hkv, d)  # same grown cache shape (1024): no build
    assert cudnn_prefill._prefill_graph_builds == before + 1
