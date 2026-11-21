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
    bool(_toolkit_validation_reasons())
    and not any("nvcc_unknown;" in r for r in _toolkit_validation_reasons()),
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
def test_pool_bindings_cover_active_plan():
    inp = _mk([2304], [2304], seed=51)
    w = _wrapper(inp, max_bs=1)
    other = {**inp, "k": inp["k"].float().neg().to(inp["k"].dtype)}
    gold = [_reference_rows(i) for i in (inp, other)]
    assert not torch.equal(*gold)
    w.prepare_pool(other["k"], k_scale=other["k_scale"])
    for i, ref in zip((inp, other), gold, strict=True):
        got = w.run(i["q"], i["weights"], i["k"], k_scale=i["k_scale"])
        assert torch.equal(got, ref)
    with pytest.raises(ValueError, match="prepared K binding"):
        w.run(inp["q"], inp["weights"], inp["k"].clone(), k_scale=inp["k_scale"])
    small = inp["k"][:64].clone()
    for clear in (False, True):
        if clear:
            w.clear_pools()
        with pytest.raises(ValueError, match="pool too small"):
            w.prepare_pool(small, k_scale=inp["k_scale"][:64])


@requires_sm103
@requires_validated_toolkit
def test_discard_replan_recapture():
    inp = _mk([512], [512], seed=52)
    w = _wrapper(inp, max_bs=1)
    out = torch.empty(512, TOPK, dtype=torch.int32, device="cuda")
    for length in (512, 256):
        prefix = inp["qo"].new_tensor([0, length])
        w.plan(prefix, prefix)

        def run():
            w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=out)

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        out.fill_(-2)
        graph.replay()
        cols = torch.arange(TOPK, device=out.device).expand_as(out)
        rows = torch.arange(512, device=out.device)[:, None]
        ref = torch.where((rows < length) & (cols < (rows + 1) // RATIO), cols, -1)
        assert torch.equal(out, ref.int())
        del graph  # do not replay a graph across replacement of its plan


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


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize("other_current_device", [False, True])
def test_native_wrapper_live_status_and_padding(other_current_device):
    inp = _mk([122], [122], seed=56)
    w = _wrapper(inp, max_bs=1)
    state = w._capacity_plan._state
    dev = inp["k"].device
    if other_current_device and torch.cuda.device_count() < 2:
        pytest.skip("second current device unavailable")
    current = (dev.index + int(other_current_device)) % torch.cuda.device_count()
    q = torch.cat((inp["q"], inp["q"][:1]))
    weights = torch.ones((123, H), device=dev)
    weights[-1].fill_(float("nan"))  # not a live row
    weights[0].view(torch.int32).fill_(1)  # smallest positive subnormal
    weights[1].fill_(float("inf"))
    scale = inp["k_scale"]
    out = torch.empty(123, TOPK, dtype=torch.int32, device=dev)
    ref = _reference_rows(inp)
    calls = []
    for name in ("check", "unpack"):
        fn = state[name]

        def counted(*args, fn=fn, name=name):
            calls.append(name)
            return fn(*args)

        state[name] = counted
    with torch.cuda.device(current):

        def run():
            return w.run(q, weights, inp["k"], k_scale=scale, out=out)

        assert run() is out
        assert calls == ["check", "unpack"]
        assert torch.equal(out[:-1], ref) and (out[-1] == -1).all()
        weights[12].fill_(float("nan"))
        scale[-1] = 0  # padded K row must still be checked
        with pytest.raises(ValueError, match="unit k_scale"):
            run()
        assert state["status"].item() == 3
        scale[-1] = 1
        with pytest.raises(ValueError, match="per-row-constant"):
            run()
        weights[12].fill_(2)
        assert run() is out and torch.equal(out[:-1], ref)
        assert state["status"].item() == w.repair_error.item() == 0

    # Exercise each compatibility gate separately, without repacking views.
    for which in ("weights", "scale", "out"):

        def view(x, name):
            if which != name:
                return x
            padded = x.new_empty((*x.shape, 2))
            padded[..., 0].copy_(x)
            return padded[..., 0]

        dst = view(out, "out")
        dst.fill_(-2)
        calls.clear()
        got = w.run(
            q, view(weights, "weights"), inp["k"], k_scale=view(scale, "scale"), out=dst
        )
        assert got is dst and torch.equal(dst[:-1], ref) and (dst[-1] == -1).all()
        assert not calls


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize("explicit_out", [False, True])
@pytest.mark.parametrize("saved_word", ["out", "repair_error"])
def test_native_wrapper_notifies_saved_tensor_mutation(explicit_out, saved_word):
    class SaveIndex(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, index):
            ctx.save_for_backward(index)
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            _ = ctx.saved_tensors
            return grad, None

    inp = _mk([122], [122], seed=57)
    w = _wrapper(inp, max_bs=1)
    out = (
        torch.empty(122, TOPK, dtype=torch.int32, device=inp["q"].device)
        if explicit_out
        else None
    )

    def run():
        return w.run(
            inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=out
        )

    first = run()
    saved = first if saved_word == "out" else w.repair_error
    value = torch.ones((), device=inp["q"].device, requires_grad=True)
    loss = SaveIndex.apply(value, saved)
    run()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@requires_sm103
@requires_validated_toolkit
def test_wrapper_predicate_views_and_dtype_copy():
    inp = _mk([122], [122], seed=58)
    w = _wrapper(inp, max_bs=1)
    gold = _reference_rows(inp)
    weights = inp["weights"].clone().requires_grad_()
    scale = inp["k_scale"].clone().requires_grad_()
    got = w.run(inp["q"], weights, inp["k"], k_scale=scale)
    assert torch.equal(got, gold)
    assert weights.grad is None and scale.grad is None
    weights = torch._neg_view(-inp["weights"])
    scale = torch._neg_view(-inp["k_scale"])
    out = torch.empty(123, TOPK, dtype=torch.float64, device=inp["q"].device)
    got = w.run(inp["q"], weights, inp["k"], k_scale=scale, out=out)
    assert got is out and torch.equal(got[:122], gold.to(torch.float64))
    assert bool((got[122:] == -1).all())
    with torch.inference_mode():
        inference_out = torch.empty(
            122, TOPK, dtype=torch.int32, device=inp["q"].device
        )
    with pytest.raises(RuntimeError, match="[Ii]nference tensor"):
        w.run(
            inp["q"],
            inp["weights"],
            inp["k"],
            k_scale=inp["k_scale"],
            out=inference_out,
        )


@requires_sm103
@requires_validated_toolkit
def test_native_wrapper_uses_input_current_stream():
    inp = _mk([122], [122], seed=59)
    w = _wrapper(inp, max_bs=1)
    dev = inp["q"].device
    out = torch.empty(122, TOPK, dtype=torch.int32, device=dev)
    w.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=out)
    inp["weights"].zero_()
    stream = torch.cuda.Stream(device=dev)
    stream.wait_stream(torch.cuda.current_stream(dev))
    other = (dev.index + 1) % torch.cuda.device_count()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2000000)
        inp["weights"].fill_(2)
        with torch.cuda.device(other):
            assert (
                w.run(
                    inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=out
                )
                is out
            )
        consumed = out.clone()
        done = stream.record_event()
    torch.cuda.current_stream(dev).wait_event(done)
    assert torch.equal(consumed, _reference_rows(inp))


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


def _assert_weighted_topk(inp, out, weights):
    """The common wrapper permits arbitrary ordering and indices at ties."""
    assert out.dtype == torch.int32
    assert out.shape == (sum(inp["q_lens"]), TOPK)
    kf = inp["k"].double()
    t0 = 0
    for r, (q_len, kv_len) in enumerate(
        zip(inp["q_lens"], inp["kv_lens"], strict=True)
    ):
        key0 = inp["starts"][r] // RATIO
        for i in range(q_len):
            visible = (kv_len - q_len + i + 1) // RATIO
            count = min(visible, TOPK)
            row = out[t0 + i]
            selected = row[row >= 0].long()
            assert selected.numel() == count
            assert bool((row[count:] == -1).all())
            assert selected.unique().numel() == count
            assert bool((selected < visible).all())
            if 0 < count < visible:
                qf = inp["q"][t0 + i].double()
                scores = (qf @ kf[key0 : key0 + visible].T).clamp_min(0)
                scores *= weights[t0 + i].double().unsqueeze(1)
                scores = scores.sum(0) * inp["k_scale"][key0 : key0 + visible].double()
                unselected = torch.ones(visible, dtype=torch.bool, device=row.device)
                unselected[selected] = False
                # These fixtures have exact zero-score boundary ties. Accept
                # any tied set, but never replace a larger score with a smaller one.
                assert scores[unselected].max() <= scores[selected].min()
        t0 += q_len


def test_weighted_topk_oracle_accepts_ties_and_rejects_wrong_selection():
    k = torch.full((TOPK + 1, 1), 2.0)
    k[-2:] = 1.0
    inp = dict(
        k=k,
        q=torch.ones(1, 1, 1),
        k_scale=torch.ones(TOPK + 1),
        q_lens=[1],
        kv_lens=[(TOPK + 1) * RATIO],
        starts=[0],
    )
    weights = torch.ones(1, 1)
    low = torch.arange(TOPK, dtype=torch.int32).unsqueeze(0)
    high = low.clone()
    high[0, -1] = TOPK
    _assert_weighted_topk(inp, low, weights)
    _assert_weighted_topk(inp, high, weights)
    k[-1] = 0.0
    with pytest.raises(AssertionError):
        _assert_weighted_topk(inp, high, weights)
    duplicate = low.clone()
    duplicate[0, -1] = 0
    with pytest.raises(AssertionError):
        _assert_weighted_topk(inp, duplicate, weights)


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
    _assert_weighted_topk(inp, out, bad_w)
    _assert_weighted_topk(inp, ref, bad_w)
    # the demoted wrapper keeps serving later calls
    out2 = w.run(inp["q"], bad_w, inp["k"], k_scale=inp["k_scale"])
    assert w.resolved_backend == "cutedsl"
    _assert_weighted_topk(inp, out2, bad_w)


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
    assert w.repair_error.item() == 16
    k_scale.fill_(1)
    old_error = w.repair_error
    if backend_kw["backend"] == "auto":
        # Demote eagerly with a different weight tensor. Only the router's
        # retirement list now owns the original plan and its status scratch.
        bad = weights.clone()
        bad[:, 0] = 100
        w.run(inp["q"], bad, inp["k"], k_scale=k_scale)
        assert w.resolved_backend == "cutedsl"
    old_error.fill_(4)
    weights[0].zero_()
    graph.replay()
    assert bool((out == -1).all()) and old_error.item() == 20
    weights[0].fill_(1)
    graph.replay()
    assert torch.equal(out, gold) and old_error.item() == 20


@requires_sm103
def test_auto_exhausted_configuration_raises():
    """auto with a configuration neither backend can serve raises, naming
    both reasons (specialized: wrong head count; cutedsl: multi-request)."""
    specialized = (
        "validated CUDA toolkit" if _toolkit_validation_reasons() else "num_heads=4"
    )
    with pytest.raises(
        ValueError,
        match=rf"auto: no backend.*specialized: [^;]*{specialized}.*; cutedsl: .*single-request",
    ):
        FP8MQATopKWrapper(
            max_context_length=4096,
            max_chunk_size=64,
            max_batch_size=6,
            device="cuda",
            backend="auto",
            num_heads=32,
            compression_ratio=4,
        )


def _region_fixture(*, backend="auto", max_bs=1, declared=False, chunk_capacity=8192):
    inp = _mk([8192], [8192], seed=91)
    # Exact quarter-integer scores keep the sampled ranking oracle strict.
    inp["q"] = (inp["q"].float() * 4).round().div(4).to(torch.float8_e4m3fn)
    inp["k"] = (inp["k"].float() * 4).round().div(4).to(torch.float8_e4m3fn)
    inp["weights"].fill_(0.125)
    wrapper = FP8MQATopKWrapper(
        backend=backend,
        max_context_length=8192,
        max_chunk_size=chunk_capacity,
        max_batch_size=max_bs,
        device="cuda",
        num_heads=H,
        compression_ratio=RATIO,
        num_keys=inp["k"].shape[0],
        declare_specialized_contract=declared,
    )
    wrapper.prepare_pool(inp["k"], k_scale=inp["k_scale"])
    return inp, wrapper


def _plan_live(wrapper, length):
    prefix = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    # The row budget deliberately differs from the live prefix.
    wrapper.plan(prefix, prefix, num_query_rows=8192)


def _check_live_prefix(inp, out, length):
    assert out.shape == (8192, TOPK)
    assert bool((out[length:] == -1).all())
    for i in sorted({v for v in (0, 3, length // 2, length - 1) if v < length}):
        sample = dict(
            q=inp["q"][i : i + 1],
            k=inp["k"],
            k_scale=inp["k_scale"],
            q_lens=[1],
            kv_lens=[i + 1],
            starts=[0],
        )
        _assert_weighted_topk(sample, out[i : i + 1], inp["weights"][i : i + 1])


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize(
    "length", [1, 3, 4, 63, 64, 65, 121, 122, 2047, 2048, 4096, 4097, 8192]
)
def test_auto_live_length_performance_region(length):
    inp, wrapper = _region_fixture()
    _plan_live(wrapper, length)
    assert wrapper.resolved_backend == ("specialized" if length <= 4096 else "cutedsl")
    out = wrapper.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    _check_live_prefix(inp, out, length)


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize("backend,max_bs", [("specialized", 1), ("auto", 2)])
def test_b1_performance_region_preserves_other_domains(backend, max_bs):
    inp, wrapper = _region_fixture(backend=backend, max_bs=max_bs)
    _plan_live(wrapper, 8192)
    assert wrapper.resolved_backend == "specialized"
    out = wrapper.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    _check_live_prefix(inp, out, 8192)


@requires_sm103
@requires_validated_toolkit
def test_auto_region_replan_uses_new_prefix_and_preserves_bindings(monkeypatch):
    inp, wrapper = _region_fixture(declared=True)
    second = inp["k"].float().flip(0).to(inp["k"].dtype)
    wrapper.prepare_pool(second, k_scale=inp["k_scale"])
    router = wrapper._capacity_plan
    make = router._make_cutedsl
    replayed = []

    def tracked_replacement():
        replacement = make()
        prepare = replacement.prepare_pool

        def tracked_prepare(k, **kwargs):
            result = prepare(k, **kwargs)
            replayed.append(k.data_ptr())
            return result

        monkeypatch.setattr(replacement, "prepare_pool", tracked_prepare)
        return replacement

    monkeypatch.setattr(router, "_make_cutedsl", tracked_replacement)
    for length, expected in (
        (2304, "specialized"),
        (4096, "specialized"),
        (8192, "cutedsl"),
        (2304, "cutedsl"),
    ):
        _plan_live(wrapper, length)
        assert wrapper.resolved_backend == expected
        if expected == "cutedsl":
            # Check replay before an eager run could lazily prepare a binding.
            assert replayed == [inp["k"].data_ptr(), second.data_ptr()]
        out = torch.empty(8192, TOPK, dtype=torch.int32, device="cuda")
        wrapper.run(inp["q"], inp["weights"], second, k_scale=inp["k_scale"], out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(
                inp["q"], inp["weights"], second, k_scale=inp["k_scale"], out=out
            )
        out.fill_(-7)
        graph.replay()
        _check_live_prefix({**inp, "k": second}, out, length)
        # Replanning explicitly invalidates captures; discard before replan.
        del graph
        torch.cuda.synchronize()


@requires_sm103
@requires_validated_toolkit
def test_outside_region_auto_capture_keeps_live_weight_scale_semantics():
    inp, wrapper = _region_fixture()  # No specialized declaration.
    _plan_live(wrapper, 8192)
    assert wrapper.resolved_backend == "cutedsl"
    out = torch.empty(8192, TOPK, dtype=torch.int32, device="cuda")
    wrapper.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=out)
    before = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"], out=out)
    inp["weights"][:, 0].mul_(3)
    inp["k_scale"][:256].mul_(2)
    graph.replay()
    _check_live_prefix(inp, out, 8192)
    assert not torch.equal(before, out)
    assert wrapper.repair_error.item() == 0


@requires_sm103
@requires_validated_toolkit
@pytest.mark.parametrize("previous_plan", [False, True])
@pytest.mark.parametrize("failure", ["second_binding", "plan"])
def test_auto_region_failed_replacement_is_retryable(
    previous_plan, failure, monkeypatch
):
    inp, wrapper = _region_fixture()
    second = inp["k"].float().flip(0).to(inp["k"].dtype)
    wrapper.prepare_pool(second, k_scale=inp["k_scale"])
    router = wrapper._capacity_plan
    if previous_plan:
        _plan_live(wrapper, 2304)
    old_active = router._active
    wrapper.repair_error.fill_(4)
    make = router._make_cutedsl

    def faulty_replacement():
        replacement = make()
        prepare = replacement.prepare_pool
        calls = 0

        def fail_prepare(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected second binding failure")
            return prepare(*args, **kwargs)

        def fail_plan(*args, **kwargs):
            raise RuntimeError("injected plan failure")

        if failure == "second_binding":
            monkeypatch.setattr(replacement, "prepare_pool", fail_prepare)
        else:
            monkeypatch.setattr(replacement, "plan", fail_plan)
        return replacement

    monkeypatch.setattr(router, "_make_cutedsl", faulty_replacement)
    with pytest.raises(RuntimeError, match="injected"):
        _plan_live(wrapper, 8192)
    assert router._active is old_active
    assert wrapper.repair_error.item() == 4
    assert wrapper.resolved_backend == "specialized"
    assert len(router._bindings) == 2
    if previous_plan:
        assert wrapper.active_shape == (2304, 1)
        out = wrapper.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
        _check_live_prefix(inp, out, 2304)
    monkeypatch.setattr(router, "_make_cutedsl", make)
    _plan_live(wrapper, 8192)
    assert wrapper.resolved_backend == "cutedsl"
    out = wrapper.run(inp["q"], inp["weights"], second, k_scale=inp["k_scale"])
    _check_live_prefix({**inp, "k": second}, out, 8192)


@requires_sm103
@requires_validated_toolkit
def test_auto_region_keeps_specialized_only_capacity():
    # CuTe requires a multiple of 64, but specialized accepts this capacity.
    inp, wrapper = _region_fixture(chunk_capacity=8193)
    _plan_live(wrapper, 8192)
    assert wrapper.resolved_backend == "specialized"
    out = wrapper.run(inp["q"], inp["weights"], inp["k"], k_scale=inp["k_scale"])
    _check_live_prefix(inp, out, 8192)
