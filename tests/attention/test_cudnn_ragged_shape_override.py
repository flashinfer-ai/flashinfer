"""The cuDNN ragged prefill graph is built once per length class with cuDNN's
execute-time shape override and fed the real (batch, max_seq_len) on every call,
so a serving stream of changing shapes builds no new execution plans (one build
is 55-70 ms). These tests pin the cache-shape function, count graph builds across
a stream of shapes, and check numerics against FA2 in both length classes.
"""

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
    # Warm the encountered classes, then revisit them in a different order.
    # Small bounded batches and the broad fallback may use different graphs.
    steps = []
    for _ in range(4):
        batch = int(torch.randint(1, 30, (1,)))
        steps.append((batch, torch.randint(129, 4096, (batch,)).tolist()))

    def run_step(step):
        batch, kv = step
        _run(
            "cudnn",
            [q_len] * batch,
            hq,
            hkv,
            d,
            causal=False,
            kv_lens=kv,
            return_lse=(q_len != 1),
        )

    for step in steps:
        run_step(step)
    before = cudnn_prefill._prefill_graph_builds
    for step in reversed(steps):
        run_step(step)
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


@requires_override
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("lse_layout", [None, "NH", "HN"])
@pytest.mark.parametrize("head_dim_qk,num_kv_heads", [(192, 4), (128, 4), (128, 1)])
@pytest.mark.parametrize("small_q", [False, True], ids=["regular_q", "small_q"])
def test_bounded_ragged_replan_capture_and_rebinding(
    causal, lse_layout, head_dim_qk, num_kv_heads, small_q, monkeypatch
):
    """One bounded graph handles different batches, totals, pointers and strides."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 7)):
        pytest.skip("bounded packed cache classes are qualified on SM100/SM107")
    if not cudnn_prefill._cudnn_supports_bounded_ragged(d128=head_dim_qk == 128):
        pytest.skip("requires FE bounded packed overrides")
    from cutlass import cute

    workspace = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, backend="cudnn"
    )
    graph = None
    generator = torch.Generator(device="cuda").manual_seed(733)
    steps = (
        (([63, 2, 0], [2049, 2011, 0]), ([31, 1, 0, 7], [3001, 127, 0, 4096]))
        if small_q
        else (([129, 65, 0], [2049, 2011, 0]), ([241, 1, 0, 128], [3001, 127, 0, 4096]))
    )
    for qlens, klens in steps:
        qo, ko = _indptr(qlens), _indptr(klens)
        q = torch.randn(
            sum(qlens),
            4,
            head_dim_qk,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        k = torch.randn(
            sum(klens),
            num_kv_heads,
            head_dim_qk,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        v = torch.randn(
            sum(klens),
            num_kv_heads,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        out = torch.empty((len(q), 4, 128), device="cuda", dtype=q.dtype)
        lse = (
            None
            if lse_layout is None
            else torch.empty(
                (len(q), 4) if lse_layout == "NH" else (4, len(q)), device="cuda"
            )
        )
        wrapper.plan(
            qo,
            ko,
            4,
            num_kv_heads,
            head_dim_qk,
            head_dim_vo=128,
            causal=causal,
            q_data_type=q.dtype,
        )

        def run():
            return wrapper.run(
                q,
                k,
                v,
                out=out,
                lse=lse,
                return_lse=lse is not None,
                lse_layout=lse_layout or "NH",
                lse_base="ln",
            )

        with monkeypatch.context() as m:
            if graph is not None:
                m.setattr(
                    cute,
                    "compile",
                    lambda *a, **kw: pytest.fail(
                        "new geometry recompiled a bounded plan"
                    ),
                )
            run()
        prepared = wrapper._cudnn_prepared
        cache_b, cache_q, cache_kv = prepared.override_cache
        assert len(qlens) <= cache_b < _OVERRIDE_CACHE_BATCH
        assert max(qlens) <= cache_q < _OVERRIDE_CACHE_SEQ_LONG
        assert max(klens) <= cache_kv < _OVERRIDE_CACHE_SEQ_LONG
        current = prepared.graph
        if graph is not None:
            assert current is graph
        graph = current
        builds = cudnn_prefill._prefill_graph_builds
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            run()
        v.mul_(-0.5)
        out.fill_(float("nan"))
        if lse is not None:
            lse.fill_(float("nan"))
        with monkeypatch.context() as m:
            m.setattr(
                cute,
                "compile",
                lambda *a, **kw: pytest.fail("warm bounded graph compiled again"),
            )
            # Older wrappers stage NH before producing HN. When native HN
            # is available, it must preserve the allocation-free warm path too.
            if lse_layout != "HN" or getattr(
                wrapper._cudnn_prepared, "stats_head_stride", 0
            ):
                m.setattr(
                    torch,
                    "empty",
                    lambda *a, **kw: pytest.fail("warm run allocated a tensor"),
                )
            old_debug = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode("error")
            try:
                run()
                capture.replay()
            finally:
                torch.cuda.set_sync_debug_mode(old_debug)
        assert cudnn_prefill._prefill_graph_builds == builds
        for i, (nq, nk) in enumerate(zip(qlens, klens, strict=True)):
            if not nq:
                continue
            qi = q[qo[i] : qo[i + 1]].double().transpose(0, 1)
            ki = k[ko[i] : ko[i + 1]].double().transpose(0, 1)
            vi = v[ko[i] : ko[i + 1]].double().transpose(0, 1)
            ki = ki.repeat_interleave(4 // num_kv_heads, dim=0)
            vi = vi.repeat_interleave(4 // num_kv_heads, dim=0)
            scores = qi @ ki.transpose(-1, -2) * head_dim_qk**-0.5
            if causal:
                scores.masked_fill_(
                    torch.arange(nk, device="cuda")[None, :]
                    > torch.arange(nq, device="cuda")[:, None] + nk - nq,
                    -float("inf"),
                )
            probs = scores.softmax(-1)
            ref = probs @ vi
            bound = torch.finfo(q.dtype).eps / 2 * (probs @ vi.abs() + ref.abs()) + 2e-5
            error = (out[qo[i] : qo[i + 1]].transpose(0, 1).double() - ref).abs()
            assert torch.all(error <= bound), float((error / bound).max())
            if lse is not None:
                got = lse.T if lse_layout == "NH" else lse
                torch.testing.assert_close(
                    got[:, qo[i] : qo[i + 1]].double(),
                    scores.logsumexp(-1),
                    atol=3e-4,
                    rtol=0,
                )
        capture.reset()


@requires_override
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("head_dim_qk,num_kv_heads", [(192, 4), (128, 1)])
def test_bounded_ragged_prewarm_classes_before_capture(
    return_lse, head_dim_qk, num_kv_heads, monkeypatch
):
    """Warm declared capacities once, then capture new live shapes in any order."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 7)):
        pytest.skip("bounded packed cache classes are qualified on SM100/SM107")
    if not cudnn_prefill._cudnn_supports_bounded_ragged(d128=head_dim_qk == 128):
        pytest.skip("requires FE bounded packed overrides")
    from cutlass import cute

    workspace = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    generator = torch.Generator(device="cuda").manual_seed(734)
    q = torch.randn(
        1024, 4, head_dim_qk, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    k = torch.zeros(8192, num_kv_heads, head_dim_qk, device="cuda", dtype=q.dtype)
    v = torch.ones(8192, num_kv_heads, 128, device="cuda", dtype=q.dtype)
    out = torch.empty(1024, 4, 128, device="cuda", dtype=q.dtype)
    lse = torch.empty(1024, 4, device="cuda") if return_lse else None
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, backend="cudnn"
    )

    total_q = total_kv = 0

    def plan(qlens, klens, q_bound, kv_bound):
        nonlocal total_q, total_kv
        total_q, total_kv = sum(qlens), sum(klens)
        wrapper.plan(
            _indptr(qlens),
            _indptr(klens),
            4,
            num_kv_heads,
            head_dim_qk,
            head_dim_vo=128,
            q_data_type=q.dtype,
            causal=False,
            max_token_per_sequence=q_bound,
            max_sequence_kv=kv_bound,
        )

    def run():
        return wrapper.run(
            q[:total_q],
            k[:total_kv],
            v[:total_kv],
            out=out[:total_q],
            lse=lse[:total_q] if lse is not None else None,
            return_lse=return_lse,
            lse_base="ln",
        )

    # Warm capacity classes using fewer live tokens than their bounds.
    # No bucket size, engine winner or split count is pinned by this test.
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        plan([63], [2049], 128, 4096)
        run()
        plan([129], [2049], 256, 4096)
        run()
        plan([321, 1], [4096, 1], 512, 4096)
        run()
    torch.cuda.current_stream().wait_stream(warm)
    builds = cudnn_prefill._prefill_graph_builds
    captures = []
    try:
        with monkeypatch.context() as m:
            m.setattr(
                cute, "compile", lambda *a, **kw: pytest.fail("warmed class recompiled")
            )
            for qlens, klens, q_bound in (
                ([241], [3001], 256),
                ([400, 0], [4096, 0], 512),
                ([65], [2048], 256),
                ([2], [3001], 2),
                ([7], [2049], 7),
                ([31], [2048], 31),
            ):
                # Keep each capture's metadata owner alive. Different wrappers
                # must also reuse the graph signatures warmed above.
                wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace, backend="cudnn"
                )
                plan(qlens, klens, q_bound, 4096)
                capture = torch.cuda.CUDAGraph()
                captures.append((capture, qlens, klens, wrapper))
                with torch.cuda.graph(capture):
                    run()
            assert cudnn_prefill._prefill_graph_builds == builds
            # All captures survive later replans; each retains its own indptrs.
            v.fill_(2)
            for capture, qlens, klens, _ in captures:
                out.fill_(float("nan"))
                if lse is not None:
                    lse.fill_(float("nan"))
                capture.replay()
                torch.testing.assert_close(
                    out[: sum(qlens)], torch.full_like(out[: sum(qlens)], 2)
                )
                if lse is not None:
                    expected = torch.cat(
                        [
                            torch.full(
                                (nq, 4), float(torch.tensor(nk).log()), device="cuda"
                            )
                            for nq, nk in zip(qlens, klens, strict=True)
                            if nq
                        ]
                    )
                    torch.testing.assert_close(
                        lse[: sum(qlens)], expected, atol=3e-4, rtol=0
                    )
    finally:
        for capture, _, _, _ in captures:
            capture.reset()


@requires_override
@pytest.mark.parametrize("head_dim_qk", [192, 128])
def test_bounded_ragged_older_frontend_keeps_broad_graph(head_dim_qk, monkeypatch):
    """Old FE keeps its existing graph signature, including SDPA kwargs."""
    cudnn = cudnn_prefill.cudnn
    original = cudnn.pygraph.sdpa
    observed = []

    def old_sdpa(self, *args, **kwargs):
        assert "max_total_seq_len_q" not in kwargs
        observed.append(True)
        return original(self, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(cudnn, "__version__", "1.30.0")
        m.setattr(cudnn.pygraph, "sdpa", old_sdpa)
        cudnn_prefill._cudnn_supports_bounded_ragged.cache_clear()
        try:
            assert not cudnn_prefill._cudnn_supports_bounded_ragged(
                d128=head_dim_qk == 128
            )
            # A distinct head count avoids a prior test's graph-cache entry.
            _run("cudnn", [129], 5, 5, head_dim_qk, dvo=128, kv_lens=[2049])
            assert observed
        finally:
            cudnn_prefill._cudnn_supports_bounded_ragged.cache_clear()


@requires_override
@pytest.mark.parametrize("lse_layout", ["NH", "HN"])
@pytest.mark.parametrize("graph_mode", [False, True])
def test_small_workspace_selection_reuse_and_capture(
    lse_layout, graph_mode, monkeypatch
):
    """Smaller workspaces keep valid plans and do not mutate cached captures."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 7)):
        pytest.skip("bounded ragged qualification requires SM100/SM107")
    import flashinfer.prefill as prefill_module

    qlens, klens = [257, 1, 1], [8192] * 3
    qo, ko = _indptr(qlens), _indptr(klens)
    q = torch.zeros(sum(qlens), 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(sum(klens), 4, 128, device="cuda", dtype=q.dtype)
    v = torch.empty_like(k)
    expected = torch.empty_like(q)
    expected_lse = torch.empty(sum(qlens), 16, device="cuda")
    for i, (nq, nk) in enumerate(zip(qlens, klens, strict=True)):
        v[ko[i] : ko[i + 1]].fill_(i + 1)
        expected[qo[i] : qo[i + 1]].fill_(i + 1)
        rows = torch.arange(nk - nq + 1, nk + 1, device="cuda", dtype=torch.float64)
        expected_lse[qo[i] : qo[i + 1]] = rows.log().float()[:, None]
    out = torch.empty_like(q)
    lse = torch.empty_like(
        expected_lse if lse_layout == "NH" else expected_lse.T,
        memory_format=torch.contiguous_format,
    )
    workspaces = [
        torch.empty(mib << 20, device="cuda", dtype=torch.uint8) for mib in (128, 1, 8)
    ]
    graph_options = {}
    if graph_mode:
        monkeypatch.setenv("FLASHINFER_CUDNN_PREFILL_SHAPE_OVERRIDE", "0")
        graph_options = dict(
            use_cuda_graph=True, qo_indptr_buf=qo.cuda(), kv_indptr_buf=ko.cuda()
        )
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspaces[0], backend="cudnn", **graph_options
    )
    captures, graphs = [], []

    def run():
        wrapper.run(
            q,
            k,
            v,
            out=out,
            lse=lse,
            return_lse=True,
            lse_layout=lse_layout,
            lse_base="ln",
        )

    try:
        for workspace in (*workspaces, workspaces[0]):
            wrapper.reset_workspace_buffer(workspace, wrapper._int_workspace_buffer)
            wrapper.plan(qo, ko, 16, 4, 128, q_data_type=q.dtype, causal=True)
            run()
            graph = wrapper._cudnn_prepared.graph
            assert graph.get_workspace_size() <= workspace.numel()
            graphs.append(graph)
            with monkeypatch.context() as m:
                m.setattr(
                    prefill_module,
                    "prepare_cudnn_batch_prefill",
                    lambda *a, **kw: pytest.fail("warm run prepared again"),
                )
                capture = torch.cuda.CUDAGraph()
                with torch.cuda.graph(capture):
                    run()
                captures.append(capture)
            # Earlier captures must still work after selecting another workspace.
            for captured in captures:
                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                captured.replay()
                torch.testing.assert_close(out, expected)
                torch.testing.assert_close(
                    lse if lse_layout == "NH" else lse.T, expected_lse
                )
        assert graphs[-1] is graphs[0]
    finally:
        for capture in captures:
            capture.reset()


@requires_override
@pytest.mark.parametrize("lse_layout", ["NH", "HN"])
@pytest.mark.parametrize("shape_override", [False, True])
def test_graph_query_capacity_replan_and_capture(
    monkeypatch, lse_layout, shape_override
):
    """Graph capacities are cache keys and survive shorter live replans."""
    if not cudnn_prefill._cudnn_supports_bounded_ragged():
        pytest.skip("packed capacity declarations require FE 1.31 native support")
    if shape_override and not cudnn_prefill._cudnn_supports_bounded_ragged(d128=True):
        pytest.skip("bounded D128 requires matching native support")
    monkeypatch.setenv(
        "FLASHINFER_CUDNN_PREFILL_SHAPE_OVERRIDE", str(int(shape_override))
    )
    workspace = torch.empty(128 << 20, device="cuda", dtype=torch.uint8)
    klens = [2048] * 3
    ko = _indptr(klens)
    k = torch.zeros(sum(klens), 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    for i in range(3):
        v[ko[i] : ko[i + 1]].fill_(i + 1)
    owners, captures, graphs = [], [], []
    try:
        for initial in ([128, 1, 1], [128, 128, 128]):
            qo = _indptr(initial)
            wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                backend="cudnn",
                use_cuda_graph=True,
                qo_indptr_buf=qo.cuda(),
                kv_indptr_buf=ko.cuda(),
            )
            q = torch.zeros(sum(initial), 16, 128, device="cuda", dtype=k.dtype)
            out = torch.empty_like(q)
            lse = torch.empty(
                (16, len(q)) if lse_layout == "HN" else (len(q), 16), device="cuda"
            )
            owners.append((wrapper, q, out, lse))

            def plan(lens):
                wrapper.plan(
                    _indptr(lens), ko, 16, 4, 128, q_data_type=q.dtype, causal=True
                )

            def run():
                wrapper.run(
                    q,
                    k,
                    v,
                    out=out,
                    lse=lse,
                    return_lse=True,
                    lse_layout=lse_layout,
                    lse_base="ln",
                )

            plan(initial)
            run()
            prepared = wrapper._cudnn_prepared
            capacity = wrapper._cudnn_plan.metadata.max_total_num_rows
            assert capacity >= sum(initial)
            graphs.append(prepared.graph)
            capture = torch.cuda.CUDAGraph()
            with torch.cuda.graph(capture):
                run()
            captures.append(capture)
            # The captured input/output storage remains at its original capacity;
            # the registered indptr buffers are updated by subsequent plans.
            for lens in ([128, 0, 0], initial):
                plan(lens)
                assert wrapper._cudnn_plan.metadata.max_total_num_rows == capacity
                assert wrapper._cudnn_prepared is prepared
                out.fill_(float("nan"))
                lse.fill_(float("nan"))
                capture.replay()
                offset = 0
                for i, nq in enumerate(lens):
                    active = slice(offset, offset + nq)
                    torch.testing.assert_close(
                        out[active], torch.full_like(out[active], i + 1)
                    )
                    reference = (
                        torch.arange(
                            klens[i] - nq + 1,
                            klens[i] + 1,
                            device="cuda",
                            dtype=torch.float64,
                        )
                        .log()
                        .float()[:, None]
                        .expand(nq, 16)
                    )
                    actual = lse.T[active] if lse_layout == "HN" else lse[active]
                    torch.testing.assert_close(actual, reference, atol=3e-4, rtol=0)
                    offset += nq
                assert torch.isnan(out[offset:]).all()
        # Same B/Q/KV and data strides; only the lifetime Q capacity differs.
        assert graphs[0] is not graphs[1]
        # A later graph selection must not invalidate either old capture.
        v.mul_(2)
        for capture, (_wrapper, _q, out, _lse) in zip(captures, owners, strict=True):
            out.fill_(float("nan"))
            capture.replay()
            assert torch.isfinite(out).all()
            assert float(out[0, 0, 0]) == 2.0
    finally:
        for capture in captures:
            capture.reset()
