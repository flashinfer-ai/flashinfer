"""CUDA-graph re-plan protocol for PagedAttention (experimental).

Contract under test (``PagedAttention(use_cuda_graph=True)``):

1. ``run()`` planned once can be captured into a ``torch.cuda.CUDAGraph``.
2. A later ``plan()`` with a new batch of the SAME capture shapes (batch size,
   table width, host maxes, total query tokens) re-fills the reserved storage;
   replaying the captured graph then computes the new batch — no re-capture.
3. A plan that would change a capture shape is rejected before anything is
   written.
4. A plan that fails midway (here: the causal envelope) leaves the previously
   published plan runnable and the reserved buffers untouched — replay still
   produces the previous batch's answer.
5. Re-plan is sync-free when the host mirrors are supplied.
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import (
    BACKENDS,
    LSE_TOL,
    OUT_TOL,
    _resolve_or_skip,
    make_metadata,
    make_problem,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_SHAPE = dict(
    batch_size=4,
    max_q=32,
    max_kv=256,
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    page_size=16,
    dtype=torch.bfloat16,
)


def _sibling_batch(p, seed):
    """A second batch with the SAME capture shapes as ``p`` (batch size, total
    query tokens, table width, maxes) but different per-request lengths, a
    different page permutation and fresh K/V/q contents — what a serving engine
    replays into one graph bucket."""
    g = torch.Generator().manual_seed(seed)
    dev = torch.device(p["device"])
    b = p["kv_seq_lens_cpu"].shape[0]
    q_lens = p["qo_indptr_cpu"].diff()
    q_lens = q_lens[torch.randperm(b, generator=g)]  # same multiset -> same total
    kv_lens = torch.minimum(
        q_lens + torch.randint(0, p["max_kv_len"] - 1, (b,), generator=g),
        torch.tensor(p["max_kv_len"], dtype=torch.int32),
    ).to(torch.int32)
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    page = p["page_size"]
    pages = (kv_lens + page - 1) // page
    width = p["block_tables"].shape[1]
    assert int(pages.max()) <= width
    pool = p["k_cache"].shape[0]
    perm = torch.randperm(pool, generator=g, dtype=torch.int32)
    bt = torch.zeros(b, width, dtype=torch.int32)
    off = 0
    for i in range(b):
        n = int(pages[i])
        bt[i, :n] = perm[off : off + n]
        off += n
    q = torch.randn_like(p["q"])
    k = torch.randn_like(p["k_cache"])
    v = torch.randn_like(p["v_cache"])
    return dict(
        p,
        q=q,
        k_cache=k,
        v_cache=v,
        k_ref=k,
        v_ref=v,
        qo_indptr=qo_indptr_cpu.to(dev),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(dev),
        kv_seq_lens_cpu=kv_lens,
        block_tables=bt.to(dev),
    )


def _plan(attn, p, backend):
    attn.plan(
        make_metadata(p),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
        backend=backend,
    )


def _reference(p):
    return reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        True,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_capture_replan_replay(backend):
    p1 = make_problem(seed=41, **_SHAPE)
    _resolve_or_skip(p1, backend)
    p2 = _sibling_batch(p1, seed=42)
    dev = torch.device(p1["device"])

    attn = PagedAttention(dev, use_cuda_graph=True)
    # static input/output storage the graph will read and write
    q = p1["q"].clone()
    k = p1["k_cache"].clone()
    v = p1["v_cache"].clone()
    out = torch.empty(
        q.shape[0], p1["num_qo_heads"], p1["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(q.shape[0], p1["num_qo_heads"], dtype=torch.float32, device=dev)

    _plan(attn, p1, backend)
    # warm up on a side stream (module load / graph build), then capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)

    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p1)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)

    # re-plan the sibling batch into the reserved storage (sync-free), swap the
    # tensor CONTENTS the graph reads, replay: the captured graph computes p2
    q.copy_(p2["q"])
    k.copy_(p2["k_cache"])
    v.copy_(p2["v_cache"])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _plan(attn, p2, backend)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    g.replay()
    torch.cuda.synchronize()
    ref_out2, ref_lse2 = _reference(p2)
    torch.testing.assert_close(out.float(), ref_out2, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse2, **LSE_TOL)
    assert not torch.allclose(ref_out, ref_out2)  # the two batches really differ


@pytest.mark.parametrize("backend", BACKENDS)
def test_replan_rejects_capture_shape_drift(backend):
    p1 = make_problem(seed=43, **_SHAPE)
    _resolve_or_skip(p1, backend)
    attn = PagedAttention(torch.device(p1["device"]), use_cuda_graph=True)
    _plan(attn, p1, backend)
    smaller = make_problem(seed=44, **dict(_SHAPE, batch_size=3))
    with pytest.raises(ValueError, match="CUDA graph re-plan: batch_size"):
        _plan(attn, smaller, backend)
    # the published plan is intact
    assert attn.backend is not None
    attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_failed_replan_restores_previous_plan(backend):
    p1 = make_problem(seed=45, **_SHAPE)
    _resolve_or_skip(p1, backend)
    dev = torch.device(p1["device"])
    attn = PagedAttention(dev, use_cuda_graph=True)
    q, k, v = p1["q"].clone(), p1["k_cache"].clone(), p1["v_cache"].clone()
    out = torch.empty(
        q.shape[0], p1["num_qo_heads"], p1["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(q.shape[0], p1["num_qo_heads"], dtype=torch.float32, device=dev)
    _plan(attn, p1, backend)
    attn.run(q, (k, v), out=out, lse=lse)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)

    # a sibling batch that violates the causal envelope on one request
    # (q_len > kv_len): metadata builds fine, plan() must fail AFTER the
    # metadata is accepted and BEFORE anything is published
    bad = _sibling_batch(p1, seed=46)
    kv_bad = bad["kv_seq_lens_cpu"].clone()
    i = int(bad["qo_indptr_cpu"].diff().argmax())
    kv_bad[i] = max(1, int(bad["qo_indptr_cpu"].diff()[i]) - 1)
    bad["kv_seq_lens_cpu"] = kv_bad
    bad["kv_seq_lens"] = kv_bad.to(dev)
    with pytest.raises(ValueError, match="causal masking requires"):
        _plan(attn, bad, backend)

    # and a failure INSIDE the transaction (the backend's own plan raising
    # after the reserved buffers were already overwritten) must roll them back
    good_sibling = _sibling_batch(p1, seed=47)
    active = attn._impl._active
    real_plan = active.plan

    def boom(meta, derived):
        raise RuntimeError("injected backend plan failure")

    active.plan = boom
    try:
        with pytest.raises(RuntimeError, match="injected"):
            _plan(attn, good_sibling, backend)
    finally:
        active.plan = real_plan

    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p1)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
