# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FP8MQATopKWrapper backend='specialized': selection correctness vs an
fp32 reference, multi-request support (which cutedsl refuses), determinism,
and the validated contract's rejection semantics."""

import pytest
import torch

from flashinfer.attn_scores import fp8_paged_mqa_topk_indexer as _functional  # noqa: F401
from flashinfer.attn_scores.mqa_topk import FP8MQATopKWrapper
from flashinfer.attn_scores.prefill_indexer import (
    VALIDATED_CUDA_VERSIONS,  # noqa: F401 - re-exported for debugging
    _toolkit_validation_reasons,
)

requires_sm103 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="specialized backend is SM103-only",
)
requires_validated_toolkit = pytest.mark.skipif(
    bool(_toolkit_validation_reasons()),
    reason="specialized backend refuses unvalidated CUDA toolkits "
    "(same gate as dispatch, torch build + JIT nvcc)",
)

H, D, TOPK, RATIO = 4, 128, 512, 4


def _mk(kv_lens, q_lens, seed):
    """Continuous fp8 K pool, packed prefixes (every non-final kv prefix a
    multiple of 256 tokens), storage padded to a multiple of 64 keys."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    kv_ind, starts = [0], []
    for kv in kv_lens:
        starts.append(kv_ind[-1])
        kv_ind.append(kv_ind[-1] + kv)
    assert all(v % 256 == 0 for v in kv_ind[:-1]), "test case must be aligned"
    live_keys = kv_ind[-1] // RATIO
    n_keys = -(-live_keys // 64) * 64  # pad storage to a page multiple
    k = (torch.randn(n_keys, D, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    t = sum(q_lens)
    q = (torch.randn(t, H, D, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
    qo = [0]
    for ql in q_lens:
        qo.append(qo[-1] + ql)
    return dict(
        k=k,
        q=q,
        qo=torch.tensor(qo, dtype=torch.int32, device=dev),
        kv=torch.tensor(kv_ind, dtype=torch.int32, device=dev),
        starts=starts,
        kv_lens=kv_lens,
        q_lens=q_lens,
        k_scale=torch.ones(n_keys, dtype=torch.float32, device=dev),
        weights=torch.full((t, H), 0.0883883, device=dev),
    )


def _reference_rows(inp):
    """fp32 sum_h relu(q_h . k), top-512 complete visible keys, ascending."""
    kf = inp["k"].to(torch.float32)
    rows = []
    t0 = 0
    for r, (q_len, kv_len) in enumerate(
        zip(inp["q_lens"], inp["kv_lens"], strict=True)
    ):
        key0 = inp["starts"][r] // RATIO
        for i in range(q_len):
            pos = kv_len - q_len + i
            vis = (pos + 1) // RATIO
            row = torch.full((TOPK,), -1, dtype=torch.int32, device="cuda")
            if vis > 0:
                qf = inp["q"][t0 + i].to(torch.float32)
                sc = (qf @ kf[key0 : key0 + vis].T).clamp_min(0).sum(0)
                n = min(vis, TOPK)
                # lower-index tie-breaking: stable sort on (-score, index)
                order = torch.argsort(sc, descending=True, stable=True)[:n]
                row[:n] = order.sort().values.to(torch.int32)
            rows.append(row)
        t0 += q_len
    return torch.stack(rows)


def _wrapper(inp, max_bs):
    w = FP8MQATopKWrapper(
        max_context_length=1 << 15,
        max_chunk_size=1 << 14,
        max_batch_size=max_bs,
        device="cuda",
        backend="specialized",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
    )
    w.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    w.plan(inp["qo"], inp["kv"])
    return w


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize(
    "kv_lens,q_lens",
    [
        ([1024], [1024]),  # direct regime, B=1 (full prefill)
        ([2930], [2930]),  # scored regime (visible > 512), B=1
        ([512, 2816, 300], [512, 2816, 300]),  # multi-request, aligned
    ],
)
def test_selection_matches_reference(kv_lens, q_lens):
    inp = _mk(kv_lens, q_lens, seed=23)
    w = _wrapper(inp, max_bs=len(kv_lens))
    out = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    gold = _reference_rows(inp)
    assert out.dtype == torch.int32 and out.shape == gold.shape
    assert torch.equal(out, gold)
    # deterministic across replays; out= honored
    buf = torch.empty_like(gold)
    out2 = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=buf)
    assert out2 is buf and torch.equal(out2, gold)


@requires_sm103
@requires_validated_toolkit
def test_multi_request_vs_cutedsl_refusal():
    """The specialized backend serves the multi-request batch the cutedsl
    backend's constructor refuses."""
    inp = _mk([512, 2816], [512, 2816], seed=5)
    w = _wrapper(inp, max_bs=2)
    out = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    assert torch.equal(out, _reference_rows(inp))


@requires_sm103
@requires_validated_toolkit
def test_rejection_semantics():
    inp = _mk([1024], [1024], seed=9)
    # constructor rejects unservable configurations loudly
    with pytest.raises(ValueError, match="num_heads=4"):
        FP8MQATopKWrapper(
            max_context_length=4096,
            max_chunk_size=64,
            max_batch_size=1,
            device="cuda",
            backend="specialized",
            num_heads=32,
            compression_ratio=4,
        )
    w = _wrapper(inp, max_bs=1)
    # non-unit k_scale rejected
    bad_scale = inp["k_scale"].clone()
    bad_scale[0] = 2.0
    with pytest.raises(ValueError, match="unit k_scale"):
        w.run(inp["q"], inp["weights"], inp["k"], k_scale=bad_scale)
    # non-uniform weights rejected (selection would not be weight-invariant)
    bad_w = inp["weights"].clone()
    bad_w[0, 0] *= 2
    with pytest.raises(ValueError, match="per-row-constant"):
        w.run(inp["q"], bad_w, inp["k"], k_scale=inp["k_scale"])
    # misaligned multi-request segment rejected at plan
    w2 = FP8MQATopKWrapper(
        max_context_length=4096,
        max_chunk_size=4096,
        max_batch_size=2,
        device="cuda",
        backend="specialized",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
    )
    w2.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    qo = torch.tensor([0, 100, 1024], dtype=torch.int32, device="cuda")
    kv = torch.tensor([0, 100, 1024], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="multiple of 256"):
        w2.plan(qo, kv)
    # chunked prefill (q_len < kv_len) is outside the validated contract
    qo3 = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    kv3 = torch.tensor([0, 1024], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="full-prefill"):
        w2.plan(qo3, kv3)


@requires_sm103
def test_unvalidated_toolkit_refusal():
    if not _toolkit_validation_reasons():
        pytest.skip("validated toolkit: constructor serves")
    with pytest.raises(ValueError, match="validated CUDA toolkit"):
        FP8MQATopKWrapper(
            max_context_length=4096,
            max_chunk_size=64,
            max_batch_size=1,
            device="cuda",
            backend="specialized",
            num_heads=H,
            compression_ratio=4,
        )


@requires_sm103
@requires_validated_toolkit
def test_auto_routes_to_specialized_for_multi_request_full_prefill():
    """backend='auto' is a router: it selects the specialized backend for
    the multi-request full-prefill class it serves (default-visible, no
    opt-in), and reports the resolution."""
    inp = _mk([512, 2816], [512, 2816], seed=41)
    w = FP8MQATopKWrapper(
        max_context_length=1 << 15,
        max_chunk_size=1 << 14,
        max_batch_size=2,
        device="cuda",
        backend="auto",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
    )
    w.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    w.plan(inp["qo"], inp["kv"])
    assert w.resolved_backend == "specialized"
    out = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    assert torch.equal(out, _reference_rows(inp))


def test_packed_to_topk_conversion_matches_sort_reference():
    """The every-4th-column + lane-mask conversion equals the sort-based
    conversion it replaced, on synthetic packed rows built per the packed
    contract (ascending tokens, 4 per complete block, open-block tail,
    -1 padding, count column). CPU-only; no kernel involved."""
    from flashinfer.attn_scores.mqa_topk_specialized import _packed_rows_to_topk
    from flashinfer.attn_scores.prefill_indexer import PACKED_WIDTH

    g = torch.Generator().manual_seed(3)
    rows, visibles = [], []
    for pos in [0, 1, 2, 3, 7, 515, 2047, 2051, 4096, 16383]:
        vis = (pos + 1) // RATIO
        n = min(vis, TOPK)
        sel = (
            torch.sort(torch.randperm(vis, generator=g)[:n]).values
            if n
            else torch.empty(0, dtype=torch.int64)
        )
        toks = (sel.unsqueeze(1) * RATIO + torch.arange(RATIO)).reshape(-1)
        tail = torch.arange(vis * RATIO, pos + 1)
        row = torch.full((PACKED_WIDTH,), -1, dtype=torch.int32)
        valid = torch.cat([toks, tail]).to(torch.int32)
        row[: valid.numel()] = valid
        row[-1] = valid.numel()
        rows.append(row)
        visibles.append(vis)
    packed = torch.stack(rows)
    visible = torch.tensor(visibles, dtype=torch.int32)
    sel_mask = torch.arange(TOPK, dtype=torch.int32).unsqueeze(0) < visible.clamp(
        max=TOPK
    ).unsqueeze(1)
    got = _packed_rows_to_topk(packed, sel_mask)
    # the previous implementation: mask every candidate column, full sort
    toks = packed[:, : PACKED_WIDTH - 1]
    ids = toks >> 2
    sent = torch.iinfo(torch.int32).max
    mask = (toks >= 0) & ((toks & 3) == 0) & (ids < visible.unsqueeze(1))
    cand = torch.where(mask, ids, torch.full_like(ids, sent))
    ref = cand.sort(dim=1).values[:, :TOPK]
    ref = torch.where(ref != sent, ref, torch.full_like(ref, -1))
    assert torch.equal(got, ref)


@requires_sm103
@requires_validated_toolkit
def test_auto_nonuniform_weights_eager_falls_back():
    """A valid common-contract eager call (nonuniform weights) on an
    auto-resolved specialized backend falls back to cutedsl and returns
    cutedsl's result instead of raising."""
    inp = _mk([2304], [2304], seed=81)
    cfg = dict(
        max_context_length=4096,
        max_chunk_size=4096,
        max_batch_size=1,
        device="cuda",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
    )
    w = FP8MQATopKWrapper(backend="auto", **cfg)
    w.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    w.plan(inp["qo"], inp["kv"], num_query_rows=2304)
    assert w.resolved_backend == "specialized"
    bad_w = inp["weights"].clone()
    bad_w[:, 0] = 100.0  # positive but not per-row-constant
    out = w.run(inp["q"], bad_w, inp["k"], k_scale=inp["k_scale"]).clone()
    assert w.resolved_backend == "cutedsl"
    # the fallback served the call with cutedsl's semantics
    wc = FP8MQATopKWrapper(backend="cutedsl", **cfg)
    wc.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    wc.plan(inp["qo"], inp["kv"], num_query_rows=2304)
    ref = wc.run(inp["q"], bad_w, inp["k"], k_scale=inp["k_scale"])
    assert torch.equal(out, ref[: out.shape[0]])
    # the demoted wrapper keeps serving later calls
    out2 = w.run(inp["q"], bad_w, inp["k"], k_scale=inp["k_scale"])
    assert torch.equal(out2[: out.shape[0]], out)


@requires_sm103
@requires_validated_toolkit
def test_auto_refuses_capture_without_declaration():
    """Without declare_specialized_contract, auto refuses to capture the
    specialized backend (a captured graph could not re-check or fall back
    on runtime weights/k_scale updates) and stays usable eagerly."""
    inp = _mk([2304], [2304], seed=81)
    w = FP8MQATopKWrapper(
        backend="auto",
        max_context_length=4096,
        max_chunk_size=4096,
        max_batch_size=1,
        device="cuda",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
    )
    w.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    w.plan(inp["qo"], inp["kv"], num_query_rows=2304)
    assert w.resolved_backend == "specialized"
    gold = _reference_rows(inp)
    out = torch.empty_like(gold)
    w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=out)
    assert torch.equal(out, gold)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="refusing to capture"),
        torch.cuda.graph(graph),
    ):
        w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=out)
    # the refusal is a clean error, not a corrupted router
    out2 = w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    assert torch.equal(out2, gold)


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize(
    "backend_kw",
    [
        dict(backend="specialized"),
        dict(backend="auto", declare_specialized_contract=True),
    ],
)
def test_captured_replay_enforces_contract_on_live_values(backend_kw):
    """A declared (or explicitly specialized) captured graph re-reads the
    live weights/k_scale on every replay: runtime updates that break the
    contract return -1 rows and set repair_error bit 16 instead of the
    stale selection the reviewer reproduced."""
    inp = _mk([2304], [2304], seed=81)
    w = FP8MQATopKWrapper(
        max_context_length=4096,
        max_chunk_size=4096,
        max_batch_size=1,
        device="cuda",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
        **backend_kw,
    )
    w.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    w.plan(inp["qo"], inp["kv"], num_query_rows=2304)
    assert w.resolved_backend == "specialized"
    gold = _reference_rows(inp)
    out = torch.empty_like(gold)
    weights, k_scale = inp["weights"], inp["k_scale"]
    w.run(inp["q"], weights, inp["k"], k_scale=k_scale, out=out)  # warmup
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(inp["q"], weights, inp["k"], k_scale=k_scale, out=out)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, gold)
    assert w.repair_error.item() == 0
    # reviewer repro 1: mutate weights after capture, replay
    weights[:, 0] = 100.0
    graph.replay()
    torch.cuda.synchronize()
    assert bool((out == -1).all()), "violating replay must not keep a selection"
    assert w.repair_error.item() & 16
    # restoring the contract restores the selection on the same graph
    weights.fill_(0.0883883)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, gold)
    # reviewer repro 2: mutate k_scale after capture, replay
    k_scale[:100] = 100.0
    graph.replay()
    torch.cuda.synchronize()
    assert bool((out == -1).all())
    assert w.repair_error.item() & 16


@requires_sm103
def test_auto_exhausted_configuration_raises():
    """auto with a configuration neither backend can serve raises, naming
    both reasons (specialized: wrong head count; cutedsl: multi-request)."""
    if not _toolkit_validation_reasons():
        match = "single-request|auto: no backend"
    else:
        match = "auto: no backend|single-request"
    with pytest.raises(ValueError, match=match):
        FP8MQATopKWrapper(
            max_context_length=4096,
            max_chunk_size=64,
            max_batch_size=6,
            device="cuda",
            backend="auto",
            num_heads=32,
            compression_ratio=4,
        )
