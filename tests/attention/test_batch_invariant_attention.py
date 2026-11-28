# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant attention class behind ``batch_invariant=True`` on the three wrappers: bitwise tests only (the mode's
acceptance criterion), the device-free checks of the fingerprint and the cache key, and the "unsupported raises, nothing falls
back" rule."""

import math

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.attention.batch_invariant.wrappers import ATTENTION_ARCHS

needs_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bi.is_supported_device(torch.device("cuda"), ATTENTION_ARCHS),
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

WRAPPERS = (
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t.view(torch.int32)


# ------------------------------------------------------------------------------------------------------ problems
class Problem:
    """A batch of requests with per-request query and KV lengths, as a paged KV cache (NHD / HND, random page table) and as
    the same keys packed along the token dimension (ragged). Every token's key at a given KV position is the same tensor in
    both layouts, so the paged and the ragged wrappers must give the same bits for the same token."""

    def __init__(self, q_lens, kv_lens, Hq, Hkv, D, page, seed=0, device="cuda"):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.q_lens, self.kv_lens, self.Hq, self.Hkv, self.D, self.page = (
            list(q_lens),
            list(kv_lens),
            Hq,
            Hkv,
            D,
            page,
        )
        self.device = torch.device(device)
        total_q = sum(q_lens)
        self.q = (
            (torch.randn(total_q, Hq, D, generator=g) * 1.5)
            .to(torch.bfloat16)
            .to(device)
        )
        self.k_rag = (
            (torch.randn(sum(kv_lens), Hkv, D, generator=g) * 1.5)
            .to(torch.bfloat16)
            .to(device)
        )
        self.v_rag = (
            (torch.randn(sum(kv_lens), Hkv, D, generator=g) * 1.5)
            .to(torch.bfloat16)
            .to(device)
        )
        self.qo_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(q_lens), 0)),
            dtype=torch.int32,
            device=device,
        )
        self.kv_indptr_rag = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(kv_lens), 0)),
            dtype=torch.int32,
            device=device,
        )
        # the paged cache: each request's pages in a random order of a pool with junk pages in between
        n_pages = [(L + page - 1) // page for L in kv_lens]
        pool_pages = sum(n_pages) + 7
        perm = torch.randperm(pool_pages, generator=g).tolist()
        self.kv_indices = torch.tensor(
            perm[: sum(n_pages)], dtype=torch.int32, device=device
        )
        self.kv_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(n_pages), 0)),
            dtype=torch.int32,
            device=device,
        )
        self.kv_last = torch.tensor(
            [
                L - (n - 1) * page if n else 0
                for L, n in zip(kv_lens, n_pages, strict=True)
            ],
            dtype=torch.int32,
            device=device,
        )
        k_pool = (torch.randn(pool_pages, page, Hkv, D, generator=g) * 1.5).to(
            torch.bfloat16
        )  # junk everywhere, then the live keys
        v_pool = (torch.randn(pool_pages, page, Hkv, D, generator=g) * 1.5).to(
            torch.bfloat16
        )
        k_rag, v_rag = self.k_rag.cpu(), self.v_rag.cpu()
        tok = 0
        off = 0
        for r, L in enumerate(kv_lens):
            for j in range(n_pages[r]):
                pg = perm[off + j]
                n = min(page, L - j * page)
                k_pool[pg, :n] = k_rag[tok + j * page : tok + j * page + n]
                v_pool[pg, :n] = v_rag[tok + j * page : tok + j * page + n]
            off += n_pages[r]
            tok += L
        self.k_nhd, self.v_nhd = k_pool.to(device), v_pool.to(device)

    def cache(self, layout):
        if layout == "NHD":
            return self.k_nhd, self.v_nhd
        return self.k_nhd.transpose(1, 2).contiguous(), self.v_nhd.transpose(
            1, 2
        ).contiguous()

    def ragged(self, layout):
        if layout == "NHD":
            return self.k_rag, self.v_rag
        return self.k_rag.transpose(0, 1).contiguous(), self.v_rag.transpose(
            0, 1
        ).contiguous()


def workspace(device, mb=128):
    return torch.empty(mb << 20, dtype=torch.uint8, device=device)


def run_paged_prefill(
    pr: Problem,
    layout="NHD",
    window_left=-1,
    ws=None,
    run_kw=None,
    causal=True,
    **plan_kw,
):
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws if ws is not None else workspace(pr.device), layout, batch_invariant=True
    )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        causal=causal,
        window_left=window_left,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        **plan_kw,
    )
    return w.run(pr.q, pr.cache(layout), **(run_kw or {}))


def run_ragged_prefill(
    pr: Problem,
    layout="NHD",
    window_left=-1,
    ws=None,
    run_kw=None,
    sinks=None,
    causal=True,
    **plan_kw,
):
    """``sinks``: set on the wrapper as ``_sinks``, where the ragged ``run()`` reads them."""
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws if ws is not None else workspace(pr.device), layout, batch_invariant=True
    )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr_rag,
        pr.Hq,
        pr.Hkv,
        pr.D,
        causal=causal,
        window_left=window_left,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        **plan_kw,
    )
    k, v = pr.ragged(layout)
    if sinks is not None:
        w._sinks = sinks
    return w.run(pr.q, k, v, **(run_kw or {}))


def run_decode(
    pr: Problem, layout="NHD", window_left=-1, ws=None, run_kw=None, **plan_kw
):
    """Decode of the LAST token of every request of ``pr`` (q rows picked from the prefill problem)."""
    assert all(ql >= 1 for ql in pr.q_lens)
    last = torch.tensor(
        [int(pr.qo_indptr[i + 1]) - 1 for i in range(len(pr.q_lens))], device=pr.device
    )
    q = pr.q[last].contiguous()
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws if ws is not None else workspace(pr.device), layout, batch_invariant=True
    )
    w.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        window_left=window_left,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        **plan_kw,
    )
    return w.run(q, pr.cache(layout), **(run_kw or {})), last


LSE = {"return_lse": True}


def reference(pr: Problem, window_left=-1, return_lse=False):
    """fp32 causal (bottom-right aligned) attention per request, the yardstick for accuracy (never for bits); with
    ``return_lse`` also the natural-log LSE of the scaled scores, ``(total_q, Hq)``."""
    outs = []
    lses = []
    tok_q = tok_kv = 0
    g = pr.Hq // pr.Hkv
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        q = pr.q[tok_q : tok_q + ql].float()  # (ql, Hq, D)
        k = (
            pr.k_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        )  # (kl, Hq, D)
        v = pr.v_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(pr.D)
        pos = torch.arange(kl - ql, kl, device=pr.device)[:, None]
        key = torch.arange(kl, device=pr.device)[None, :]
        mask = key <= pos
        if window_left >= 0:
            mask &= key >= pos - window_left
        s = s.masked_fill(~mask[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
        if return_lse:
            lses.append(torch.logsumexp(s, dim=-1).t())
        tok_q += ql
        tok_kv += kl
    return (torch.cat(outs), torch.cat(lses)) if return_lse else torch.cat(outs)


def _wrapper_case(kind, pr, metadata_device="cuda", ws=None, **plan_options):
    """Public plan/run arguments for the same one-query-per-request problem."""
    w = getattr(flashinfer, WRAPPERS[kind])(
        ws if ws is not None else workspace(pr.device, mb=8),
        "NHD",
        batch_invariant=True,
    )

    def metadata(t):
        return t.to(metadata_device).clone()

    kwargs = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    if kind == 0:
        plan_args = (
            metadata(pr.kv_indptr),
            metadata(pr.kv_indices),
            metadata(pr.kv_last),
            pr.Hq,
            pr.Hkv,
            pr.D,
            pr.page,
        )
        run_args = (pr.q, pr.cache("NHD"))
    elif kind == 1:
        plan_args = (
            metadata(pr.qo_indptr),
            metadata(pr.kv_indptr),
            metadata(pr.kv_indices),
            metadata(pr.kv_last),
            pr.Hq,
            pr.Hkv,
            pr.D,
            pr.page,
        )
        kwargs["causal"] = True
        run_args = (pr.q, pr.cache("NHD"))
    else:
        plan_args = (
            metadata(pr.qo_indptr),
            metadata(pr.kv_indptr_rag),
            pr.Hq,
            pr.Hkv,
            pr.D,
        )
        kwargs["causal"] = True
        run_args = (pr.q, *pr.ragged("NHD"))
    kwargs.update(plan_options)
    w.plan(*plan_args, **kwargs)
    return w, run_args, plan_args, kwargs


def _saved_forward_options(wrapper):
    names = (
        "_sm_scale",
        "_window_left",
        "_pos_encoding_mode",
        "_logits_soft_cap",
        "_rope_scale",
        "_rope_theta",
        "_causal",
        "_use_fp16_qk_reduction",
    )
    return {name: getattr(wrapper, name) for name in names if hasattr(wrapper, name)}


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_forward_equivalent_scale(kind):
    pr = Problem([1], [17], 8, 2, 128, 16, seed=32)
    w, args, _, _ = _wrapper_case(kind, pr)
    expected = bits(w.run(*args)).clone()
    # These distinct Python floats round to the same binary32 kernel scale.
    equivalent = 128**-0.5
    assert equivalent != 1.0 / math.sqrt(128)
    options = {} if kind == 0 else {"causal": True}
    got = w.forward(*args, sm_scale=equivalent, **options)
    assert torch.equal(bits(got), expected)
    assert torch.equal(bits(w.run(*args)), expected)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
@pytest.mark.parametrize(
    "bad",
    [
        {"sm_scale": 0.5},
        {"pos_encoding_mode": "ROPE_LLAMA"},
        {"window_left": 123},
        {"rope_scale": 2.0},
    ],
    ids=["scale", "rope", "window", "rope-scale"],
)
def test_forward_rejection_preserves_plan(kind, bad):
    pr = Problem([1], [17], 8, 2, 128, 16, seed=33)
    w, args, _, _ = _wrapper_case(kind, pr)
    expected = bits(w.run(*args)).clone()
    saved = _saved_forward_options(w)
    options = {} if kind == 0 else {"causal": True}
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.forward(*args, **(options | bad))
    assert _saved_forward_options(w) == saved
    assert torch.equal(bits(w.run(*args)), expected)


@needs_device
@pytest.mark.parametrize("kind", [1, 2], ids=["paged", "ragged"])
def test_prefill_replan_after_rejected_forward(kind):
    pr = Problem([1], [17], 8, 2, 128, 16, seed=34)
    w, args, plan_args, plan_kw = _wrapper_case(kind, pr)
    expected = bits(w.run(*args)).clone()
    saved = _saved_forward_options(w)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.forward(*args, causal=True, use_fp16_qk_reduction=True)
    assert _saved_forward_options(w) == saved
    assert torch.equal(bits(w.run(*args)), expected)
    w.plan(*plan_args, **plan_kw)
    assert torch.equal(bits(w.run(*args)), expected)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_forward_custom_scale(kind):
    pr = Problem([1], [17], 8, 2, 128, 16, seed=36)
    scale = 0.25
    w, args, _, _ = _wrapper_case(kind, pr, sm_scale=scale)
    expected = bits(w.run(*args)).clone()
    options = {} if kind == 0 else {"causal": True}
    equivalent = math.nextafter(scale, math.inf)
    assert equivalent != scale
    assert torch.equal(bits(w.forward(*args, sm_scale=equivalent, **options)), expected)
    for changed in (None, 0.5):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            w.forward(*args, sm_scale=changed, **options)
        assert torch.equal(bits(w.run(*args)), expected)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
@pytest.mark.parametrize("metadata_device", ["cpu", "cuda"])
def test_metadata_snapshot(kind, metadata_device):
    pr = Problem([1, 1], [17, 33], 8, 2, 128, 16, seed=37)
    w, args, plan_args, _ = _wrapper_case(kind, pr, metadata_device)
    expected = bits(w.run(*args)).clone()
    # Mutating the caller's buffers after plan must not mutate its snapshot.
    for value in plan_args:
        if isinstance(value, torch.Tensor):
            value.zero_()
    assert torch.equal(bits(w.run(*args)), expected)


@needs_device
@pytest.mark.parametrize("kind", [0, 1], ids=["decode", "paged"])
@pytest.mark.parametrize("metadata_device", ["cpu", "cuda"])
def test_metadata_page_index_bounds(kind, metadata_device):
    pr = Problem([1], [17], 8, 2, 128, 16, seed=38)
    w, args, plan_args, plan_kw = _wrapper_case(kind, pr, metadata_device)
    index_arg = 1 if kind == 0 else 2
    bad = list(plan_args)
    bad[index_arg] = plan_args[index_arg].clone()
    bad[index_arg][0] = -1
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.plan(*bad, **plan_kw)
    # The page-pool size is a run-time input; its upper bound is checked then.
    bad[index_arg][0] = pr.k_nhd.shape[0]
    w.plan(*bad, **plan_kw)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.run(*args)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
@pytest.mark.parametrize("metadata_device", ["cpu", "cuda"])
def test_metadata_empty_batch(kind, metadata_device):
    pr = Problem([], [], 8, 2, 128, 16, seed=38)
    w, args, _, _ = _wrapper_case(kind, pr, metadata_device)
    assert w.run(*args).shape == (0, 8, 128)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_output_overlap_and_disjoint_arena(kind):
    pr = Problem([1], [17], 8, 2, 128, 16, seed=39)
    w, args, _, _ = _wrapper_case(kind, pr)
    expected = bits(w.run(*args)).clone()
    q = args[0]
    k, v = args[1] if kind < 2 else args[1:]
    # All inputs and output can occupy distinct aligned ranges in one arena.
    arena = torch.empty(
        q.numel() * 2 + k.numel() + v.numel(), dtype=q.dtype, device=q.device
    )
    offset = 0
    views = []
    for source in (q, k, v):
        view = arena[offset : offset + source.numel()].view(source.shape)
        view.copy_(source)
        views.append(view)
        offset += source.numel()
    out = arena[offset:].view(q.shape)
    q_shared, k_shared, v_shared = views
    shared_args = (q_shared, (k_shared, v_shared)) if kind < 2 else tuple(views)
    assert w.run(*shared_args, out=out) is out
    assert torch.equal(bits(out), expected)
    # A shifted pointer still overlaps K; pointer equality is insufficient.
    start = q.numel() + 8
    overlapping = arena[start : start + q.numel()].view(q.shape)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.run(*shared_args, out=overlapping)
    assert torch.equal(bits(w.run(*shared_args)), expected)
    from flashinfer.attention.batch_invariant import engine

    delegate = w._batch_invariant
    engine_args = (
        delegate._planned,
        q_shared,
        k_shared,
        v_shared,
        delegate._kv_indptr,
        delegate._kv_indices,
        1.0 / math.sqrt(pr.D),
    )
    engine_kw = {"ragged": kind == 2}
    assert engine.run_attention(*engine_args, out=out, **engine_kw) is out
    assert torch.equal(bits(out), expected)
    with pytest.raises(ValueError):
        engine.run_attention(*engine_args, out=overlapping, **engine_kw)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_forward_return_lse_rejection_preserves_plan(kind):
    pr = Problem([1], [17], 8, 2, 128, 16, seed=35)
    w, args, _, _ = _wrapper_case(kind, pr)
    expected = bits(w.run(*args)).clone()
    saved = _saved_forward_options(w)
    options = {} if kind == 0 else {"causal": True}
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.forward_return_lse(*args, window_left=123, **options)
    assert _saved_forward_options(w) == saved
    assert torch.equal(bits(w.run(*args)), expected)


# ------------------------------------------------------------------------------------------------------ device-free
@pytest.mark.parametrize("op", WRAPPERS)
def test_fingerprint_is_a_function_of_the_head_dim(op):
    fps = {D: bi.fingerprint(op, head_dim=D, causal=True) for D in (64, 128, 256)}
    assert len(set(fps.values())) == 3
    for D, fp in fps.items():
        assert fp.startswith("attn-bf16-d") and fp == bi.fingerprint(
            WRAPPERS[0], head_dim=D
        )  # one class for the three wrappers
    assert bi.class_name(op) == "attention.bf16"
    with pytest.raises(TypeError):
        bi.fingerprint(op, head_dim=128, causal=True, page_size=16)


@pytest.mark.parametrize("op", WRAPPERS)
def test_lse_fingerprints_are_pinned(op):
    """The LSE member's receipts, per head dim and base; the class receipts are the output's and do not move with them."""
    from flashinfer.attention.batch_invariant import spec

    assert spec.FINGERPRINTS == {
        64: "attn-bf16-d64-9a05233ec9351b0d",
        128: "attn-bf16-d128-027b0379515b82e2",
        256: "attn-bf16-d256-b8cb59dab4eb6c0f",
    }
    pins = {
        (64, "log2"): "attn-bf16-d64-lse-log2-a8c548ec287e23bd",
        (64, "ln"): "attn-bf16-d64-lse-ln-d94f46b73185b273",
        (128, "log2"): "attn-bf16-d128-lse-log2-c9084818e797cacf",
        (128, "ln"): "attn-bf16-d128-lse-ln-8e4f2608bce73f6c",
        (256, "log2"): "attn-bf16-d256-lse-log2-ad95fa9071f86c5c",
        (256, "ln"): "attn-bf16-d256-lse-ln-54b51dc083581fce",
    }
    assert pins == spec.LSE_FINGERPRINTS
    for (D, base), fp in pins.items():
        assert bi.fingerprint(op, head_dim=D, causal=True, lse_base=base) == fp
        assert bi.fingerprint(op, head_dim=D, causal=True) == spec.FINGERPRINTS[D]
    with pytest.raises(ValueError):
        bi.fingerprint(op, head_dim=128, causal=True, lse_base="log10")
    with pytest.raises(TypeError):
        bi.fingerprint(op, head_dim=128, causal=True, lse_layout="HN")


def test_lse_term_is_declared():
    """``lse2 = add(mul(m, c), lg2(l))`` of the state that normalises the output, ``lse_e = mul(lse2, fp32(ln 2))``; the output
    term contains no ``lg2``."""
    import struct

    from flashinfer.attention.batch_invariant import spec
    from flashinfer.batch_invariant.fp_interface.term import sexpr

    assert struct.pack("<f", spec.LN2) == struct.pack("<f", math.log(2.0))
    assert struct.unpack("<I", struct.pack("<f", spec.LN2))[0] == 0x3F317218
    for D, p, win in ((64, 7, -1), (128, 40, -1), (256, 33, 5)):
        s = spec.AttentionSpec(D, T=16, C=32)
        row = spec.RowSpec(p=p, window_left=win, row=1)
        st = s.state(row, 0)
        m, l = sexpr(st.m), sexpr(st.l)
        lse2 = "(add " + " ".join(sorted([f"(mul {m} (sym c))", f"(op:lg2 {l})"])) + ")"
        assert sexpr(s.lse(row, "log2")) == lse2
        ln = "(mul " + " ".join(sorted([lse2, f"(const {spec.LN2!r})"])) + ")"
        assert sexpr(s.lse(row, "ln")) == ln
        # every output column is normalised by the (m, l) the LSE reads
        other = s.state(row, 3)
        assert other.m is st.m and other.l is st.l
        assert "op:lg2" not in sexpr(s.output(row, 0))
    init = spec.init_state(s.alg)
    assert (
        sexpr(spec.lse_term(s.alg, init, s.c))
        == "(add (mul (const -inf) (sym c)) (op:lg2 (const 0.0)))"
    )


def test_planner_unit_kind_rules():
    """The ``auto`` unit-kind rules of the planner (placements: every kind computes the same bits): a multi-chunk tile of at most
    ``INKERNEL_MAX_ROWS`` rows splits when the fused units would fill under 2 x SM, a wider (prefill) tile only when they would fill
    under 1 / ``WIDE_SPLIT_SM_FRACTION`` of the SMs."""
    from flashinfer.attention.batch_invariant import plan as P

    # 4 requests x 4096 tokens of prefill at g = 4: 32 tokens per tile -> 128 wide tiles per request and kv head, 8 heads = 4096 tiles
    qo = [0, 4096, 8192, 12288, 16384]
    kv = [0, 256, 512, 768, 1024]
    last = [16] * 4
    wide = lambda sm: P.build_plan(qo, kv, last, 32, 8, 128, 16, sm_count=sm)  # noqa: E731
    n_tiles = 4 * 128 * 8
    assert wide(148).num_units == n_tiles
    assert all(
        d[P.F_MODE] == 0 for d in wide(n_tiles).desc
    )  # the tiles fill the SMs once: fused
    assert all(
        d[P.F_MODE] == 0 for d in wide(4 * n_tiles).desc
    )  # exactly 1 / 4 of the SMs: still fused
    assert any(
        d[P.F_MODE] == 1 for d in wide(4 * n_tiles + 1).desc
    )  # under 1 / 4: split
    # decode-sized tiles (one token per request, g = 4 -> 4 rows): split under 2 x SM
    qo_d = list(range(0, 5))
    narrow = lambda sm: P.build_plan(qo_d, kv, last, 32, 8, 128, 16, sm_count=sm)  # noqa: E731
    assert any(d[P.F_MODE] == 1 for d in narrow(32).desc)  # 32 tiles < 64
    assert all(d[P.F_MODE] == 0 for d in narrow(16).desc)  # 32 tiles >= 32


def test_planner_snake_order():
    """``snake_order``: a permutation of the longest-first list whose waves of ``grid`` units alternate direction, and the order
    ``build_plan`` emits follows the launch grid (``grid``: the persistent CTAs, else the SM count)."""
    from flashinfer.attention.batch_invariant import plan as P

    units = [[i, 100 - i] for i in range(23)]  # descending "lengths"
    out = P.snake_order(units, 5)
    assert sorted(map(tuple, out)) == sorted(map(tuple, units))
    for w, start in enumerate(range(0, 23, 5)):
        wave = units[start : start + 5]
        assert out[start : start + 5] == (wave[::-1] if w % 2 else wave), w
    assert P.snake_order(units, 0) == units and P.snake_order([], 4) == []
    qo = [0, 4096, 8192]
    kv = [0, 256, 512]
    last = [16, 16]
    for sm, grid in ((148, 0), (148, 64), (1000, 0)):
        plan = P.build_plan(qo, kv, last, 32, 8, 128, 16, sm_count=sm, grid=grid)
        lengths = [d[P.F_TILE_HI] - d[P.F_TILE_LO] for d in plan.desc]
        g = min(len(plan.desc), grid if grid else sm)
        ref = P.snake_order(sorted(lengths, reverse=True), g)
        assert lengths == ref, (sm, grid)


def test_planner_swaps_unit_kinds():
    """The Swaps-body unit-kind model (``plan.swaps_unit_kinds``; placements: every choice computes the same bits): every unit fused
    on the grid of integral waves when the fused walk reaches the HBM rate, every multi-chunk unit split when there are too few units;
    the per-CTA stream rate differs per head dim; non-Swaps batches keep the 2 x SM rule."""
    from flashinfer.attention.batch_invariant import plan as P

    SM = 212

    def decode(B, L, D, sm=SM, swap=True, page=32, grid_search=True):
        per = -(-L // page)
        qo = list(range(B + 1))
        kv = [b * per for b in range(B + 1)]
        last = [L - (per - 1) * page] * B
        return P.build_plan(
            qo,
            kv,
            last,
            32,
            8,
            D,
            page,
            sm_count=sm,
            small_swap=swap,
            swap_max_rows=8,
            grid_search=grid_search,
        )

    def kinds(plan):
        fused = sum(1 for d in plan.desc if d[P.F_MODE] == 0)
        return fused, plan.num_units - fused, len(plan.groups), plan.grid

    # D = 128, 16-chunk units: 128 units (B = 16) stream at the HBM rate fused on the default grid; 64 (B = 8) do not and split;
    # 256 (B = 32) take two integral waves of 128 CTAs instead of 1.2 waves of 212; 512 (B = 64) five waves of 103
    assert kinds(decode(16, 8192, 128)) == (128, 0, 0, 0)
    assert kinds(decode(8, 8192, 128)) == (0, 1024, 64, 0)
    assert kinds(decode(32, 8192, 128)) == (256, 0, 0, 128)
    assert kinds(decode(64, 8192, 128)) == (512, 0, 0, 103)
    assert kinds(decode(24, 8192, 128)) == (192, 0, 0, 96)
    assert kinds(decode(1, 8192, 128)) == (0, 128, 8, 0)
    # short fused units (two chunks) keep the default grid unless its last wave is nearly empty
    assert kinds(decode(32, 1024, 128)) == (256, 0, 0, 128)
    assert kinds(decode(64, 1024, 128)) == (512, 0, 0, 0)
    assert kinds(decode(256, 1024, 128)) == (2048, 0, 0, 0)
    # two-chunk units: fused from a few units up (the in-kernel fold and the first wave's burst), split at B = 1
    assert kinds(decode(8, 1024, 128)) == (64, 0, 0, 0)
    assert kinds(decode(1, 1024, 128)) == (0, 16, 8, 0)
    # D = 64: a lone CTA streams at about half the D = 128 rate, so 128 units of 16 chunks split, and 512 units take three
    # integral waves of 171 CTAs (above the D = 64 saturating count) rather than a split
    assert kinds(decode(16, 8192, 64)) == (0, 2048, 128, 0)
    assert kinds(decode(8, 1024, 64)) == (64, 0, 0, 0)
    assert kinds(decode(64, 32768, 64)) == (512, 0, 0, 171)
    # off the parts the grid model was fitted on (grid_search False: the engine's rule for every other part) the fused walk takes
    # the full grid or the even-waves grid (no fitted constant) and a part-filled last wave never turns the batch into a split one:
    # on 148 SMs 512 long units take four waves of 128 (74 with the fitted search), 320 units three waves of 107 (a thin third wave
    # on the full grid), 192 units two waves of 96 at every head dim, 64 units their own 64 CTAs; one unit of 16 chunks still splits
    assert kinds(decode(64, 8192, 128, sm=148, grid_search=False)) == (512, 0, 0, 128)
    assert kinds(decode(64, 8192, 128, sm=148)) == (512, 0, 0, 74)
    assert kinds(decode(40, 8192, 128, sm=148, grid_search=False)) == (320, 0, 0, 107)
    assert kinds(decode(40, 2048, 128, sm=148, grid_search=False)) == (320, 0, 0, 107)
    assert kinds(decode(24, 8192, 64, sm=148, grid_search=False)) == (192, 0, 0, 96)
    assert kinds(decode(20, 8192, 128, sm=148, grid_search=False)) == (160, 0, 0, 80)
    assert kinds(decode(8, 8192, 128, sm=148, grid_search=False)) == (64, 0, 0, 0)
    assert kinds(decode(1, 8192, 128, sm=148, grid_search=False)) == (0, 128, 8, 0)
    assert kinds(decode(256, 8192, 256, sm=148, grid_search=False)) == (2048, 0, 0, 147)
    # the grid search terminates on parts of one or two SMs (the integral-wave grid is one CTA there) and never names a grid
    # below one CTA or above the part
    for sm in (1, 2, 3):
        for D in (64, 128, 256):
            for n in (1, 5, 64, 1000):
                grids = P.swaps_fused_grids(n, sm, D)
                assert grids and all(1 <= g <= sm for g in grids), (sm, D, n, grids)
            assert decode(4, 32768, D, sm=sm).num_units > 0
            assert decode(1, 8192, D, sm=sm).num_units > 0
    assert kinds(decode(32, 8192, 64)) == (0, 4096, 256, 0)
    # D = 256: a lone CTA streams faster, so 64 units fuse
    assert kinds(decode(8, 8192, 256)) == (64, 0, 0, 0)
    assert kinds(decode(32, 8192, 256)) == (256, 0, 0, 128)
    # the decision follows the SM count (a launch parameter, never the bits): 128 units on a 1000-SM part split, 512 units on a
    # 148-SM part take seven waves of 74
    assert kinds(decode(16, 8192, 128, sm=1000))[0] == 0
    assert kinds(decode(64, 8192, 128, sm=148)) == (512, 0, 0, 74)
    # the fused grids: the default and every integral-wave grid above the floor, longest-first waves; the chosen one is the cheapest
    assert P.swaps_fused_grids(512, SM, 128) == [212, 171, 128, 103]
    assert P.swaps_fused_grids(128, SM, 128) == [128]
    units = [(64, (16, 4, 4), 4)] * 512
    costs = {
        g: P.swaps_kind_cost(units, [False] * 512, g, SM, 128, 2)
        for g in P.swaps_fused_grids(512, SM, 128)
    }
    assert min(costs, key=costs.get) == 103 and costs[103] < costs[212]
    assert P.swaps_unit_kinds([], SM, 128, 2) == ([], 0)
    # a non-Swaps batch (the row body's replicated placement) keeps the 2 x SM rule: 128 units split
    assert kinds(decode(16, 8192, 128, swap=False)) == (0, 2048, 128, 0)


def test_cache_key_covers_every_kernel_source():
    from flashinfer.batch_invariant.jit import relative_sources

    srcs = relative_sources("batch_invariant_attention")
    for f in (
        "attention/batch_invariant/blocks/compile.py",
        "attention/batch_invariant/blocks/kernel.py",
        "attention/batch_invariant/swaps/kernel.py",
        "attention/batch_invariant/fixup.py",
        "attention/batch_invariant/lse.py",
        "attention/batch_invariant/spec.py",
        "attention/batch_invariant/plan.py",
        "batch_invariant/canon.py",
        "batch_invariant/fp_interface/cute.py",
    ):
        assert f in srcs, f
    assert (
        "attention/batch_invariant/wrappers.py" not in srcs
    )  # the API module stays outside the closure
    # no source of another kernel family: an edit there must not recompile the attention kernels
    other = ("batch_invariant/sampling/", "batch_invariant/quantization/", "fused_moe/")
    assert not [f for f in srcs if f.startswith(other)]


def test_swaps_kernel_identity_carries_its_capacity():
    """The Swaps kernel's name (its cache key, in-process and on disk) differs between the shared-memory capacities its ring is sized
    for, so a kernel compiled for one part is never loaded for another."""
    from flashinfer.attention.batch_invariant.swaps.compile import kernel_name

    a = kernel_name(128, 8, 16, 8, False, cap=228 << 10)
    b = kernel_name(128, 8, 16, 8, False, cap=328 << 10)
    assert a != b and a.endswith("_c228") and b.endswith("_c328")


def test_row_body_kernel_identity_carries_its_capacity():
    """The row body's name carries the shared-memory capacity its smem plan is checked against (the paired placement at D = 128 fits
    the 328 KB parts only), so a kernel compiled for one part is never loaded for another."""
    from flashinfer.attention.batch_invariant.blocks.compile import kernel_name

    a = kernel_name(128, 4, 16, 3, 2, True, False, cap=228 << 10)
    b = kernel_name(128, 4, 16, 3, 2, True, False, cap=328 << 10)
    assert a != b and a.endswith("_q2_c228") and b.endswith("_q2_c328")
    assert kernel_name(128, 4, 16, 4) == "attn_d128_g4_p16_s4_w1"


def test_unsupported_device_raises_before_any_kernel():
    ws = torch.empty(1 << 20, dtype=torch.uint8)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa2", batch_invariant=True
        )


# ------------------------------------------------------------------------------------------------------ on the device
@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_matches_reference(D, Hq, Hkv):
    pr = Problem([300, 1000, 64], [700, 2500, 3000], Hq, Hkv, D, 16, seed=1)
    out = run_paged_prefill(pr)
    ref = reference(pr)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_prefill_equals_decode(D, layout):
    """R4: the last token of every request gives the same bits as a prefill row and as a decode step."""
    pr = Problem([256, 1, 1024, 33], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=2)
    pre = run_paged_prefill(pr, layout)
    dec, last = run_decode(pr, layout)
    assert torch.equal(bits(pre[last]), bits(dec))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("page", [1, 16, 128])
def test_paged_equals_ragged(D, page):
    """R4: the paged and the ragged KV layouts hand the kernel the same keys at the same positions: the same bits."""
    pr = Problem([300, 1000, 64, 2048], [700, 2500, 3000, 2048], 32, 8, D, page, seed=3)
    assert torch.equal(bits(run_paged_prefill(pr)), bits(run_ragged_prefill(pr)))
    assert torch.equal(
        bits(run_paged_prefill(pr, "HND")), bits(run_ragged_prefill(pr, "HND"))
    )


@needs_device
@pytest.mark.parametrize("D", [64, 128])
def test_determinism_and_graph_replay(D):
    pr = Problem([512, 7, 1000], [4096, 1025, 1000], 32, 8, D, 16, seed=4)
    ref = bits(run_paged_prefill(pr)).clone()
    for _ in range(3):
        assert torch.equal(bits(run_paged_prefill(pr)), ref)
    ws = workspace(pr.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out = torch.empty_like(pr.q)
    s = torch.cuda.Stream()
    s.wait_stream(
        torch.cuda.current_stream()
    )  # the plan's device copies precede the side stream's warm-up
    with torch.cuda.stream(s):
        w.run(pr.q, pr.cache("NHD"), out=out)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            w.run(pr.q, pr.cache("NHD"), out=out)
    torch.cuda.synchronize()
    for _ in range(3):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_dependent_launch_chain_reads_the_predecessor(D):
    """R1 under programmatic dependent launch: four decode steps whose query is the previous step's output, enqueued back to back
    on one stream and captured as one CUDA graph, give the bits of the same steps run one at a time with a device synchronisation
    between them. The attention kernel issues its loads only after the preceding grid has completed, so a dependent that is
    scheduled early never reads a predecessor's unfinished output."""
    from flashinfer.utils import device_support_pdl

    pr = Problem([1, 1, 1, 1], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=21)
    if not device_support_pdl(pr.device):
        pytest.skip("this part has no programmatic dependent launch")
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    cache = pr.cache("NHD")
    steps = 4
    # one step at a time, no dependent launch, a synchronisation after each: the yardstick
    q, ref = pr.q.clone(), []
    for _ in range(steps):
        q = w.run(q, cache, enable_pdl=False)
        torch.cuda.synchronize()
        ref.append(bits(q).clone())
    # the same chain back to back as programmatic dependents
    q, got = pr.q.clone(), []
    for _ in range(steps):
        q = w.run(q, cache, enable_pdl=True)
        got.append(q)
    torch.cuda.synchronize()
    for t in range(steps):
        assert torch.equal(bits(got[t]), ref[t]), t
    # and captured as one CUDA graph over fixed buffers (the dependent-launch edges are captured with the kernels)
    bufs = [torch.empty_like(pr.q) for _ in range(steps)]

    def chain():
        for t in range(steps):
            w.run(pr.q if t == 0 else bufs[t - 1], cache, out=bufs[t], enable_pdl=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            chain()
    torch.cuda.synchronize()
    for _ in range(3):
        for b in bufs:
            b.zero_()
        g.replay()
        torch.cuda.synchronize()
        for t in range(steps):
            assert torch.equal(bits(bufs[t]), ref[t]), t


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_batch_permutation_and_subbatch_invariance(D):
    """R2: a request's bits do not depend on its neighbours, its position in the batch or the batch size."""
    q_lens, kv_lens = [300, 1000, 64, 2048, 1], [700, 2500, 3000, 2048, 5000]
    pr = Problem(q_lens, kv_lens, 32, 8, D, 16, seed=5)
    full = bits(run_paged_prefill(pr))
    rows = [
        slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1]))
        for i in range(len(q_lens))
    ]
    # one request alone
    for i in (0, 2, 4):
        single = Problem([q_lens[i]], [kv_lens[i]], 32, 8, D, 16, seed=5)
        single.q = pr.q[rows[i]].clone()
        single.k_rag = pr.k_rag[
            int(pr.kv_indptr_rag[i]) : int(pr.kv_indptr_rag[i + 1])
        ].clone()
        single.v_rag = pr.v_rag[
            int(pr.kv_indptr_rag[i]) : int(pr.kv_indptr_rag[i + 1])
        ].clone()
        assert torch.equal(
            bits(run_ragged_prefill(single)), bits(run_ragged_prefill(pr))[rows[i]]
        ), i
    # a permutation of the requests
    perm = [3, 0, 4, 1, 2]
    pp = Problem(
        [q_lens[i] for i in perm], [kv_lens[i] for i in perm], 32, 8, D, 16, seed=5
    )
    pp.q = torch.cat([pr.q[rows[i]] for i in perm]).contiguous()
    pp.k_rag = torch.cat(
        [
            pr.k_rag[int(pr.kv_indptr_rag[i]) : int(pr.kv_indptr_rag[i + 1])]
            for i in perm
        ]
    ).contiguous()
    pp.v_rag = torch.cat(
        [
            pr.v_rag[int(pr.kv_indptr_rag[i]) : int(pr.kv_indptr_rag[i + 1])]
            for i in perm
        ]
    ).contiguous()
    permuted = bits(run_ragged_prefill(pp))
    ref_rag = bits(run_ragged_prefill(pr))
    assert torch.equal(ref_rag, full)  # paged == ragged on the full batch
    off = 0
    for i in perm:
        n = q_lens[i]
        assert torch.equal(permuted[off : off + n], full[rows[i]]), i
        off += n


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_decode_batch_invariance_and_page_table_junk(D):
    """R2 for decode: the same request's step gives the same bits in a batch of one and in a batch of eight with other requests,
    whatever pages the pool assigns (the page table and the junk beyond each request's keys never reach the bits)."""
    kv = [1024, 4096, 33, 700, 2500, 8192, 1, 1500]
    pr = Problem([1] * 8, kv, 32, 8, D, 16, seed=6)
    out8, _ = run_decode(pr)
    for i in (0, 1, 5, 6):
        one = Problem(
            [1], [kv[i]], 32, 8, D, 16, seed=100 + i
        )  # a different pool and page order
        one.q = pr.q[i : i + 1].clone()
        one.k_rag = pr.k_rag[
            int(pr.kv_indptr_rag[i]) : int(pr.kv_indptr_rag[i + 1])
        ].clone()
        one.v_rag = pr.v_rag[
            int(pr.kv_indptr_rag[i]) : int(pr.kv_indptr_rag[i + 1])
        ].clone()
        one = _repack(one)  # the request's keys in a fresh random page table
        out1, _ = run_decode(one)
        assert torch.equal(bits(out1), bits(out8[i : i + 1])), i


def _repack(p: Problem) -> Problem:
    """Rebuild the paged pool of ``p`` from its ragged keys (after the ragged keys were replaced)."""
    fresh = Problem(
        p.q_lens,
        p.kv_lens,
        p.Hq,
        p.Hkv,
        p.D,
        p.page,
        seed=int(torch.randint(0, 1 << 30, (1,)).item()),
    )
    fresh.q, fresh.k_rag, fresh.v_rag = p.q, p.k_rag, p.v_rag
    # rewrite the pool's live pages from the new ragged keys
    k_pool, v_pool = fresh.k_nhd.clone(), fresh.v_nhd.clone()
    tok = 0
    for r, L in enumerate(p.kv_lens):
        n_pages = int(fresh.kv_indptr[r + 1] - fresh.kv_indptr[r])
        for j in range(n_pages):
            pg = int(fresh.kv_indices[int(fresh.kv_indptr[r]) + j])
            n = min(p.page, L - j * p.page)
            k_pool[pg, :n] = p.k_rag[tok + j * p.page : tok + j * p.page + n]
            v_pool[pg, :n] = p.v_rag[tok + j * p.page : tok + j * p.page + n]
        tok += L
    fresh.k_nhd, fresh.v_nhd = k_pool, v_pool
    return fresh


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_chunked_prefill_equals_full(D):
    """R4 for incremental prefill: a request prefilled in two chunks (the second over the first's cached keys) gives, token for
    token, the bits of the one-shot prefill -- a token's output depends on its position and the keys below it alone, not on how the
    prefill was cut. Two requests cut at different points (one cut inside a 512-key chunk, one at a tile boundary), paged and ragged."""
    kv = [1700, 1000]
    cut = [700, 512]
    full = Problem(kv, kv, 32, 8, D, 16, seed=12)
    out_full = bits(run_paged_prefill(full))
    assert torch.equal(out_full, bits(run_ragged_prefill(full)))
    rows = [slice(int(full.qo_indptr[i]), int(full.qo_indptr[i + 1])) for i in range(2)]
    kv_off = [int(full.kv_indptr_rag[i]) for i in range(2)]
    # chunk 1: the first `cut` tokens of each request, over their own keys only
    c1 = Problem(cut, cut, 32, 8, D, 16, seed=13)
    c1.q = torch.cat([full.q[rows[i]][: cut[i]] for i in range(2)]).contiguous()
    c1.k_rag = torch.cat(
        [full.k_rag[kv_off[i] : kv_off[i] + cut[i]] for i in range(2)]
    ).contiguous()
    c1.v_rag = torch.cat(
        [full.v_rag[kv_off[i] : kv_off[i] + cut[i]] for i in range(2)]
    ).contiguous()
    c1 = _repack(c1)
    # chunk 2: the remaining tokens of each request, over ALL its keys (the first `cut` are the cache)
    c2 = Problem([kv[i] - cut[i] for i in range(2)], kv, 32, 8, D, 16, seed=14)
    c2.q = torch.cat([full.q[rows[i]][cut[i] :] for i in range(2)]).contiguous()
    c2.k_rag, c2.v_rag = full.k_rag.clone(), full.v_rag.clone()
    c2 = _repack(c2)
    for prob, part in ((c1, 0), (c2, 1)):
        out_p = bits(run_paged_prefill(prob))
        out_r = bits(run_ragged_prefill(prob))
        off = 0
        for i in range(2):
            n = cut[i] if part == 0 else kv[i] - cut[i]
            ref = (
                out_full[rows[i]][: cut[i]]
                if part == 0
                else out_full[rows[i]][cut[i] :]
            )
            assert torch.equal(out_p[off : off + n], ref), (D, part, i, "paged")
            assert torch.equal(out_r[off : off + n], ref), (D, part, i, "ragged")
            off += n


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_multi_token_decode_equals_prefill(D):
    """R4 for speculative / multi-token decode: the decode wrapper with ``q_len_per_req = 4`` gives, for the last four tokens of every
    request, the bits of the prefill rows."""
    q_len = 4
    pr = Problem([256, 33, 1024, 7], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=15)
    pre = bits(run_paged_prefill(pr))
    last4 = torch.cat(
        [
            torch.arange(
                int(pr.qo_indptr[i + 1]) - q_len,
                int(pr.qo_indptr[i + 1]),
                device=pr.device,
            )
            for i in range(4)
        ]
    )
    q = pr.q[last4].contiguous()
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_len_per_req=q_len,
    )
    dec = bits(w.run(q, pr.cache("NHD")))
    assert torch.equal(dec, pre[last4])


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_head_subset_invariance(D):
    """R2 across heads: running a subset of the KV heads (whole GQA groups, with their query heads) gives, head for head, the bits of
    the run over all heads -- a head's output depends on its own q / K / V alone, never on how many heads share the launch. Prefill
    (paged and ragged) and decode; a four-group subset and a single group."""
    Hq, Hkv = 32, 8
    g = Hq // Hkv
    pr = Problem([300, 1, 1000, 64], [700, 2500, 2500, 3000], Hq, Hkv, D, 16, seed=16)
    full_pre = bits(run_paged_prefill(pr))
    full_dec, last = run_decode(pr)
    full_dec = bits(full_dec)
    for kv_sel in ([1, 3, 4, 6], [5]):
        q_sel = [j * g + r for j in kv_sel for r in range(g)]
        sub = Problem(
            pr.q_lens, pr.kv_lens, len(q_sel), len(kv_sel), D, 16, seed=17
        )  # another pool and page table
        sub.q = pr.q[:, q_sel].contiguous()
        sub.k_rag = pr.k_rag[:, kv_sel].contiguous()
        sub.v_rag = pr.v_rag[:, kv_sel].contiguous()
        sub = _repack(sub)
        assert torch.equal(bits(run_paged_prefill(sub)), full_pre[:, q_sel]), (
            D,
            kv_sel,
            "paged prefill",
        )
        assert torch.equal(bits(run_ragged_prefill(sub)), full_pre[:, q_sel]), (
            D,
            kv_sel,
            "ragged prefill",
        )
        dec, _ = run_decode(sub)
        assert torch.equal(bits(dec), full_dec[:, q_sel]), (D, kv_sel, "decode")


@needs_device
def test_sm_count_and_grid_do_not_change_bits():
    """R3: the planner's unit kind depends on the SM count and the launch's persistent grid; neither reaches the bits."""
    from flashinfer.attention.batch_invariant import engine

    pr = Problem([256, 1000, 1], [3000, 2500, 8192], 32, 8, 128, 16, seed=7)
    ref = bits(run_paged_prefill(pr))
    cfg = engine.default_config(D=128)
    for sm in (1, 8, 148, 1000):
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            32,
            8,
            128,
            16,
            -1,
            cfg,
            None,
            pr.device,
            sm_count=sm,
        )
        out = engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
            -1,
            "NHD",
            None,
            cfg,
        )
        assert torch.equal(bits(out), ref), sm
    from dataclasses import replace

    for grid in (1, 3, 64):
        c = replace(cfg, grid=grid)
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            32,
            8,
            128,
            16,
            -1,
            c,
            None,
            pr.device,
        )
        out = engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
            -1,
            "NHD",
            None,
            c,
        )
        assert torch.equal(bits(out), ref), grid


@needs_device
@pytest.mark.parametrize("decode", [True, False])
def test_run_attention_runs_the_plan_s_kv_indptr(decode):
    """The plan owns the page-table offsets: the descriptor table carries every request's first page-table entry and page count
    next to its query offset and KV length, the kernels run with the plan's own kv_indptr, and run_attention accepts the array the
    plan was built from or an equal copy (host or device) and rejects one that differs or has another length. The bits do not
    depend on which equal tensor the caller passes."""
    from flashinfer.attention.batch_invariant import engine

    D = 128
    pr = (
        Problem([1] * 6, [900, 1500, 2100, 300, 4096, 777], 32, 8, D, 16, seed=23)
        if decode
        else Problem([200, 33, 1000], [2000, 500, 1000], 32, 8, D, 16, seed=24)
    )
    cfg = engine.default_config(D=D)
    planned = engine.plan_attention(
        pr.qo_indptr, pr.kv_indptr, pr.kv_last, 32, 8, D, 16, -1, cfg, None, pr.device
    )
    assert planned.kv_indptr is not None
    assert torch.equal(planned.kv_indptr, pr.kv_indptr)
    assert planned.kv_indptr.data_ptr() != pr.kv_indptr.data_ptr()  # a snapshot

    def run(plan, kv_indptr):
        return bits(
            engine.run_attention(
                plan,
                pr.q,
                pr.k_nhd,
                pr.v_nhd,
                kv_indptr,
                pr.kv_indices,
                1.0 / math.sqrt(D),
                -1,
                "NHD",
                None,
                cfg,
            )
        )

    ref = run(planned, pr.kv_indptr)
    wrapper = run_decode(pr)[0] if decode else run_paged_prefill(pr)
    assert torch.equal(ref, bits(wrapper))
    assert torch.equal(run(planned, planned.kv_indptr), ref)  # the snapshot itself
    assert torch.equal(run(planned, pr.kv_indptr.clone()), ref)  # an equal copy
    assert torch.equal(run(planned, pr.kv_indptr.cpu()), ref)  # the host array
    from_host = engine.plan_attention(
        pr.qo_indptr.cpu(),
        pr.kv_indptr.cpu(),
        pr.kv_last.cpu(),
        32,
        8,
        D,
        16,
        -1,
        cfg,
        None,
        pr.device,
    )  # a plan built from host arrays (the wrappers' way) runs with the device array
    assert torch.equal(run(from_host, pr.kv_indptr), ref)
    with pytest.raises(ValueError, match="differs from the plan"):
        run(
            planned, pr.kv_indptr + 1
        )  # the same page counts at moved offsets: another page table
    with pytest.raises(ValueError, match="entries at run_attention"):
        run(planned, pr.kv_indptr[:-1])


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_config_agreement(D):
    """R3: every member of the configuration space (ring depth, unit kind, softmax warpgroups, fold placement, small-unit body)
    gives the same bits as the wrapper's pick; at D = 64 the paired placement too (two Q tiles per CTA), on a batch large enough
    to be paired, and at D = 128 on 328 KB parts (pairing forced with an SM count of one: the problem's units would not fill four
    per SM; its rings, the split kind with both fold placements). The whole space at D = 128 / 256 (108 / 36 points); every 7th
    point of the 252 at D = 64 (each distinct kernel is a compile)."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    if D == 64:
        pr = Problem([2048, 2048, 1], [4096, 3000, 1000], 32, 8, D, 16, seed=8)
    else:
        pr = Problem([256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, seed=8)
    ref = bits(run_paged_prefill(pr))
    space = engine.all_valid_configs(D)
    cfgs = list(space[::7] if D == 64 else space)
    q2_sm = (
        None  # the SM count the paired configurations plan with (None: the device's)
    )
    if D == 64:
        two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
        rings = (2, 3) + ((4,) if arch_smem_cap(pr.device) >= (300 << 10) else ())
        cfgs += [replace(two, kv_slots=s) for s in rings]
    elif D == 128 and arch_smem_cap(pr.device) >= (300 << 10):
        two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
        q2_sm = 1
        cfgs += [
            replace(two, kv_slots=s, split_policy=p, merge=m)
            for s in (2, 3)
            for p, m in (("auto", "auto"), ("split", "auto"), ("split", "inkernel"))
        ]
    for cfg in cfgs:
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            32,
            8,
            D,
            16,
            -1,
            cfg,
            None,
            pr.device,
            sm_count=q2_sm if cfg.q2 else None,
        )
        if cfg.q2:
            assert planned.plan.num_launch is not None, (cfg, planned.plan.num_units)
        out = engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
        )
        assert torch.equal(bits(out), ref), cfg


@needs_device
def test_sliding_window_prefill_equals_decode():
    pr = Problem([256, 1, 2000], [1500, 2500, 5000], 32, 8, 128, 16, seed=9)
    pre = run_paged_prefill(pr, window_left=600)
    dec, last = run_decode(pr, window_left=600)
    assert torch.equal(bits(pre[last]), bits(dec))
    assert torch.equal(bits(pre), bits(run_ragged_prefill(pr, window_left=600)))


def _paired_d128(engine):
    """The paired placement's configuration at D = 128 (the wrapper's pick on 328 KB parts)."""
    from dataclasses import replace

    return replace(
        engine.default_config(D=128),
        softmax_wgs=2,
        q2=True,
        kv_slots=engine.Q2_RING[128][True],
    )


@needs_device
def test_paired_d128_sm_count_and_grid_do_not_change_bits():
    """R3 for the paired placement at D = 128: the SM count decides whether the batch pairs at all (below four units per SM the
    one-warpgroup body runs) and the unit kind, the grid how many CTAs walk the pairs; none of it reaches the bits."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    pr = Problem([256, 1000, 1], [3000, 2500, 8192], 32, 8, 128, 16, seed=7)
    if arch_smem_cap(pr.device) < (300 << 10):
        pytest.skip("the paired placement at D = 128 needs a 328 KB part")
    ref = bits(run_paged_prefill(pr))
    cfg = _paired_d128(engine)
    paired = set()
    for sm in (1, 8, 148, 1000):
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            32,
            8,
            128,
            16,
            -1,
            cfg,
            None,
            pr.device,
            sm_count=sm,
        )
        paired.add(planned.plan.num_launch is not None)
        out = engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
            -1,
            "NHD",
            None,
            cfg,
        )
        assert torch.equal(bits(out), ref), sm
    assert paired == {True, False}  # both the paired kernel and the fallback body ran
    for grid in (1, 3, 64):
        c = replace(cfg, grid=grid)
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            32,
            8,
            128,
            16,
            -1,
            c,
            None,
            pr.device,
            sm_count=1,
        )
        assert planned.plan.num_launch is not None
        out = engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
            -1,
            "NHD",
            None,
            c,
        )
        assert torch.equal(bits(out), ref), grid


@needs_device
def test_paired_d128_ragged_window_and_pages():
    """The paired placement at D = 128 over the ragged layout, a sliding window (the union of the two tiles' key ranges) and page
    sizes 1 and 128 gives the one-warpgroup body's bits; tail tiles of at most 32 live rows take the full-row path there."""
    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    if arch_smem_cap(torch.device("cuda")) < (300 << 10):
        pytest.skip("the paired placement at D = 128 needs a 328 KB part")
    cfg = _paired_d128(engine)

    def paired(pr, ragged=False, window_left=-1):
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr_rag if ragged else pr.kv_indptr,
            pr.kv_last,
            pr.Hq,
            pr.Hkv,
            128,
            pr.page,
            window_left,
            cfg,
            None,
            pr.device,
            sm_count=1,
            ragged=ragged,
        )
        assert planned.plan.num_launch is not None
        if ragged:
            k, v = pr.ragged("NHD")
            return engine.run_attention(
                planned,
                pr.q,
                k,
                v,
                pr.kv_indptr_rag,
                torch.zeros(1, dtype=torch.int32, device=pr.device),
                1.0 / math.sqrt(128),
                window_left,
                "NHD",
                None,
                cfg,
                ragged=True,
            )
        return engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
            window_left,
            "NHD",
            None,
            cfg,
        )

    pr = Problem([264, 33, 1, 129], [3000, 2500, 1000, 5000], 32, 8, 128, 16, seed=15)
    ref = bits(run_paged_prefill(pr))
    assert torch.equal(bits(paired(pr)), ref)
    assert torch.equal(bits(paired(pr, ragged=True)), ref)
    pr = Problem([256, 1, 2000], [1500, 2500, 5000], 32, 8, 128, 16, seed=9)
    ref = bits(run_paged_prefill(pr, window_left=600))
    assert torch.equal(bits(paired(pr, window_left=600)), ref)
    for page in (1, 128):
        pr = Problem([256, 1, 700], [3000, 2500, 1000], 32, 8, 128, page, seed=14)
        assert torch.equal(bits(paired(pr)), bits(run_paged_prefill(pr)))


@needs_device
def test_wrapper_default_pairs_by_group_at_d64():
    """The wrappers' default: the paired placement for prefill and ragged batches at D = 64 whose GQA group reaches
    ``engine.Q2_DEFAULT_MIN_GROUP``; the one-warpgroup body for smaller groups, decode, and every batch at D = 128 (where the placement is
    an explicit configuration). The table is the measured-faster body per (layout, group) on SM107."""
    from flashinfer.attention.batch_invariant import engine

    assert engine.Q2_DEFAULT_DIMS == (64,)
    assert engine.Q2_DEFAULT_MIN_GROUP == {64: 4}
    ws = workspace(torch.device("cuda"))
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    for D, Hq, expect in (
        (64, 32, True),
        (64, 16, False),
        (64, 8, False),
        (128, 32, False),
    ):
        pr = Problem([300, 1], [1000, 600], Hq, 8, D, 16, seed=22)
        paged = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        paged.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            Hq,
            8,
            D,
            16,
            causal=True,
            **kw,
        )
        ragged = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        ragged.plan(pr.qo_indptr, pr.kv_indptr_rag, Hq, 8, D, causal=True, **kw)
        dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        dec.plan(pr.kv_indptr, pr.kv_indices, pr.kv_last, Hq, 8, D, 16, **kw)
        assert paged._batch_invariant._config.q2 == expect, (D, Hq)
        assert ragged._batch_invariant._config.q2 == expect, (D, Hq)
        assert not dec._batch_invariant._config.q2, (D, Hq)


# ------------------------------------------------------------------------------------------------------ the LSE member
def _both(res):
    out, lse = res
    assert lse.dtype == torch.float32
    return bits(out), bits(lse)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_lse_matches_reference(D, Hq, Hkv):
    pr = Problem([300, 1000, 64], [700, 2500, 3000], Hq, Hkv, D, 16, seed=1)
    _, ref = reference(pr, return_lse=True)
    tol = 1e-3 * max(1.0, ref.abs().max().item())
    _, ln = run_paged_prefill(pr, run_kw=dict(LSE, lse_base="ln"))
    _, log2 = run_paged_prefill(pr, run_kw=LSE)
    assert ln.shape == log2.shape == (sum(pr.q_lens), Hq)
    assert not torch.isnan(ln).any() and not torch.isnan(log2).any()
    assert (ln - ref).abs().max().item() < tol
    assert (log2 - ref / math.log(2.0)).abs().max().item() < tol / math.log(2.0)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_lse_request_leaves_output_bits(D, kind):
    """The LSE reads the ``(m, l)`` that normalise the output: the output bits are the same without it, with it in base 2
    and (prefill) in base e."""
    pr = Problem([1] * 6, [1024, 4096, 33, 700, 2500, 1], 32, 8, D, 16, seed=40)
    w, args, _, _ = _wrapper_case(kind, pr)
    plain = bits(w.run(*args)).clone()
    out, lse = w.run(*args, return_lse=True)
    assert torch.equal(bits(out), plain)
    out_r, lse_r = w.run_return_lse(*args)
    assert torch.equal(bits(out_r), plain) and torch.equal(bits(lse_r), bits(lse))
    if kind:
        out_e, _ = w.run(*args, return_lse=True, lse_base="ln")
        assert torch.equal(bits(out_e), plain)
    # prefill-shaped units (fused multi-chunk, the paired placement at D = 64)
    big = Problem([2048, 300, 1], [4096, 700, 3000], 32, 8, D, 16, seed=41)
    plain = bits(run_paged_prefill(big))
    for base in ("log2", "ln"):
        out, _ = run_paged_prefill(big, run_kw=dict(LSE, lse_base=base))
        assert torch.equal(bits(out), plain), base


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_lse_prefill_equals_decode_equals_ragged(D, layout):
    """R4 for the LSE: a token's LSE is the same bits as a paged prefill row, a ragged prefill row, a decode step and a row of
    a multi-token decode."""
    pr = Problem([256, 4, 1024, 33], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=2)
    out, lse = _both(run_paged_prefill(pr, layout, run_kw=LSE))
    out_r, lse_r = _both(run_ragged_prefill(pr, layout, run_kw=LSE))
    assert torch.equal(out, out_r) and torch.equal(lse, lse_r)
    (dec, dec_lse), last = run_decode(pr, layout, run_kw=LSE)
    assert torch.equal(bits(dec), out[last]) and torch.equal(bits(dec_lse), lse[last])
    q_len = 4
    last4 = torch.cat(
        [
            torch.arange(
                int(pr.qo_indptr[i + 1]) - q_len,
                int(pr.qo_indptr[i + 1]),
                device=pr.device,
            )
            for i in range(4)
        ]
    )
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), layout, batch_invariant=True
    )
    w.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_len_per_req=q_len,
    )
    mtd, mtd_lse = _both(w.run(pr.q[last4].contiguous(), pr.cache(layout), **LSE))
    assert mtd_lse.shape == (4 * q_len, pr.Hq)
    assert torch.equal(mtd, out[last4]) and torch.equal(mtd_lse, lse[last4])


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_lse_batch_permutation_and_subbatch_invariance(D):
    """R2 for the LSE: a request's LSE bits do not depend on its neighbours, its position in the batch or the batch size."""
    q_lens, kv_lens = [300, 1000, 64, 2048, 1], [700, 2500, 3000, 2048, 5000]
    pr = Problem(q_lens, kv_lens, 32, 8, D, 16, seed=5)
    out, full = _both(run_paged_prefill(pr, run_kw=LSE))
    rows = [
        slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1]))
        for i in range(len(q_lens))
    ]
    kv = [
        slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1]))
        for i in range(len(q_lens))
    ]
    for i in (0, 2, 4):
        single = Problem([q_lens[i]], [kv_lens[i]], 32, 8, D, 16, seed=5)
        single.q = pr.q[rows[i]].clone()
        single.k_rag = pr.k_rag[kv[i]].clone()
        single.v_rag = pr.v_rag[kv[i]].clone()
        o1, l1 = _both(run_ragged_prefill(single, run_kw=LSE))
        assert torch.equal(l1, full[rows[i]]) and torch.equal(o1, out[rows[i]]), i
    perm = [3, 0, 4, 1, 2]
    pp = Problem(
        [q_lens[i] for i in perm], [kv_lens[i] for i in perm], 32, 8, D, 16, seed=5
    )
    pp.q = torch.cat([pr.q[rows[i]] for i in perm]).contiguous()
    pp.k_rag = torch.cat([pr.k_rag[kv[i]] for i in perm]).contiguous()
    pp.v_rag = torch.cat([pr.v_rag[kv[i]] for i in perm]).contiguous()
    _, permuted = _both(run_ragged_prefill(pp, run_kw=LSE))
    off = 0
    for i in perm:
        n = q_lens[i]
        assert torch.equal(permuted[off : off + n], full[rows[i]]), i
        off += n


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_lse_decode_batch_invariance(D):
    """R2 for decode: a request's LSE in a batch of one (a fresh page table) and in a batch of eight is the same bits."""
    kv = [1024, 4096, 33, 700, 2500, 8192, 1, 1500]
    pr = Problem([1] * 8, kv, 32, 8, D, 16, seed=6)
    (_, lse8), _ = run_decode(pr, run_kw=LSE)
    for i in (0, 1, 5, 6):
        one = Problem([1], [kv[i]], 32, 8, D, 16, seed=100 + i)
        one.q = pr.q[i : i + 1].clone()
        one.k_rag = pr.k_rag[
            int(pr.kv_indptr_rag[i]) : int(pr.kv_indptr_rag[i + 1])
        ].clone()
        one.v_rag = pr.v_rag[
            int(pr.kv_indptr_rag[i]) : int(pr.kv_indptr_rag[i + 1])
        ].clone()
        one = _repack(one)
        (_, lse1), _ = run_decode(one, run_kw=LSE)
        assert torch.equal(bits(lse1), bits(lse8[i : i + 1])), i


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_lse_chunked_prefill_equals_full(D):
    """R4 for incremental prefill: the LSE of a token is the same bits whether its request is prefilled in one shot or in two
    chunks (the second over the first's cached keys)."""
    kv = [1700, 1000]
    cut = [700, 512]
    full = Problem(kv, kv, 32, 8, D, 16, seed=12)
    _, lse_full = _both(run_paged_prefill(full, run_kw=LSE))
    rows = [slice(int(full.qo_indptr[i]), int(full.qo_indptr[i + 1])) for i in range(2)]
    kv_off = [int(full.kv_indptr_rag[i]) for i in range(2)]
    c1 = Problem(cut, cut, 32, 8, D, 16, seed=13)
    c1.q = torch.cat([full.q[rows[i]][: cut[i]] for i in range(2)]).contiguous()
    c1.k_rag = torch.cat(
        [full.k_rag[kv_off[i] : kv_off[i] + cut[i]] for i in range(2)]
    ).contiguous()
    c1.v_rag = torch.cat(
        [full.v_rag[kv_off[i] : kv_off[i] + cut[i]] for i in range(2)]
    ).contiguous()
    c1 = _repack(c1)
    c2 = Problem([kv[i] - cut[i] for i in range(2)], kv, 32, 8, D, 16, seed=14)
    c2.q = torch.cat([full.q[rows[i]][cut[i] :] for i in range(2)]).contiguous()
    c2.k_rag, c2.v_rag = full.k_rag.clone(), full.v_rag.clone()
    c2 = _repack(c2)
    for prob, part in ((c1, 0), (c2, 1)):
        _, lse_p = _both(run_paged_prefill(prob, run_kw=LSE))
        _, lse_r = _both(run_ragged_prefill(prob, run_kw=LSE))
        off = 0
        for i in range(2):
            n = cut[i] if part == 0 else kv[i] - cut[i]
            ref = (
                lse_full[rows[i]][: cut[i]]
                if part == 0
                else lse_full[rows[i]][cut[i] :]
            )
            assert torch.equal(lse_p[off : off + n], ref), (D, part, i, "paged")
            assert torch.equal(lse_r[off : off + n], ref), (D, part, i, "ragged")
            off += n


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_lse_config_agreement(D):
    """R3 for the LSE: every kernel site that holds a row's final ``(m, l)`` stores the same bits -- the fused unit's epilogue,
    the fixup kernel, the in-kernel fold, the Swaps body and (D = 64) the paired placement -- over the configuration space on a
    prefill-shaped and a decode-shaped batch, in both bases; the output bits stay the wrapper's."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    if D == 64:
        prefill = Problem([2048, 2048, 1], [4096, 3000, 1000], 32, 8, D, 16, seed=8)
    else:
        prefill = Problem(
            [256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, seed=8
        )
    decode = Problem([1] * 6, [1024, 4096, 33, 700, 2500, 1], 32, 8, D, 16, seed=42)
    space = engine.all_valid_configs(D)
    cfgs = list(space[::7] if D == 64 else space)
    if D == 64:
        two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
        rings = (2, 3) + ((4,) if arch_smem_cap(prefill.device) >= (300 << 10) else ())
        cfgs += [replace(two, kv_slots=s) for s in rings]
    sites = set()
    for pr in (prefill, decode):
        ref = {
            base: _both(run_paged_prefill(pr, run_kw=dict(LSE, lse_base=base)))
            for base in ("log2", "ln")
        }
        for n, cfg in enumerate(cfgs):
            planned = engine.plan_attention(
                pr.qo_indptr,
                pr.kv_indptr,
                pr.kv_last,
                32,
                8,
                D,
                16,
                -1,
                cfg,
                None,
                pr.device,
            )
            plan = planned.plan
            swaps = cfg.small == "swap" and engine.swaps_n_pad(plan) > 0
            body = "swaps" if swaps else "row"
            if any(d[6] == 0 for d in plan.desc):
                sites.add(f"{body}-fused")
            if plan.fixup_groups:
                sites.add(f"{body}-fixup")
            if any(plan.group_inkernel):
                sites.add(f"{body}-inkernel")
            if cfg.q2 and plan.num_launch is not None:
                sites.add("paired")
            for base in ("log2", "ln") if n % 4 == 0 else ("log2",):
                got = engine.run_attention(
                    planned,
                    pr.q,
                    pr.k_nhd,
                    pr.v_nhd,
                    pr.kv_indptr,
                    pr.kv_indices,
                    1.0 / math.sqrt(D),
                    -1,
                    "NHD",
                    None,
                    cfg,
                    return_lse=True,
                    lse_base=base,
                )
                out, lse = _both(got)
                assert torch.equal(out, ref[base][0]), (cfg, base)
                assert torch.equal(lse, ref[base][1]), (cfg, base)
    want = {
        f"{b}-{k}" for b in ("row", "swaps") for k in ("fused", "fixup", "inkernel")
    }
    if D == 64:
        want.add("paired")
    assert sites >= want, want - sites


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_lse_natural_log_is_the_declared_multiply(D):
    """``lse_base="ln"`` is ``mul.rn(lse2, fp32(ln 2))`` of the base-2 bits, formed in the kernel: IEEE ``mul`` on the host
    reproduces it bit for bit."""
    from flashinfer.attention.batch_invariant.spec import LN2

    pr = Problem([300, 1, 1000, 64], [700, 2500, 2500, 3000], 32, 8, D, 16, seed=43)
    ln2 = torch.tensor(LN2, dtype=torch.float32)
    for run in (run_paged_prefill, run_ragged_prefill):
        _, log2 = run(pr, run_kw=LSE)
        _, ln = run(pr, run_kw=dict(LSE, lse_base="ln"))
        assert torch.equal(bits(log2.cpu() * ln2), bits(ln.cpu()))


@needs_device
@pytest.mark.parametrize("kind", [1, 2], ids=["paged", "ragged"])
def test_lse_layouts_and_buffers(kind):
    """``lse_layout="HN"`` is the ``(total_q, Hq)`` bits transposed; a caller's ``lse`` buffer is written in place and returned."""
    pr = Problem([300, 1, 64], [700, 2500, 3000], 32, 8, 128, 16, seed=44)
    w, args, _, _ = _wrapper_case(kind, pr)
    total_q = sum(pr.q_lens)
    for base in ("log2", "ln"):
        out, nh = w.run(*args, return_lse=True, lse_base=base)
        assert nh.shape == (total_q, pr.Hq)
        _, hn = w.run(*args, return_lse=True, lse_base=base, lse_layout="HN")
        assert hn.shape == (pr.Hq, total_q) and hn.is_contiguous()
        assert torch.equal(bits(hn), bits(nh.t().contiguous()))
        buf = torch.full((total_q, pr.Hq), float("nan"), device=pr.device)
        out_b = torch.empty_like(pr.q)
        res = w.run(*args, out=out_b, lse=buf, return_lse=True, lse_base=base)
        assert res[0] is out_b and res[1] is buf
        assert torch.equal(bits(buf), bits(nh)) and torch.equal(bits(out_b), bits(out))
        buf_hn = torch.full((pr.Hq, total_q), float("nan"), device=pr.device)
        res = w.run(*args, lse=buf_hn, return_lse=True, lse_base=base, lse_layout="HN")
        assert res[1] is buf_hn and torch.equal(bits(buf_hn), bits(hn))
    # an lse buffer without return_lse is not used, as on the other backends
    untouched = torch.full((total_q, pr.Hq), float("nan"), device=pr.device)
    res = w.run(*args, lse=untouched)
    assert isinstance(res, torch.Tensor) and bool(torch.isnan(untouched).all())


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_lse_deprecated_forward_and_decode_buffer(kind):
    pr = Problem([1] * 4, [17, 700, 2500, 1], 8, 2, 128, 16, seed=45)
    w, args, _, _ = _wrapper_case(kind, pr)
    out, lse = w.run_return_lse(*args)
    assert lse.shape == (4, 8)
    buf = torch.full((4, 8), float("nan"), device=pr.device)
    res = w.run(*args, lse=buf, return_lse=True)
    assert res[1] is buf and torch.equal(bits(buf), bits(lse))
    options = {} if kind == 0 else {"causal": True}
    out_f, lse_f = w.forward_return_lse(*args, **options)
    assert torch.equal(bits(out_f), bits(out)) and torch.equal(bits(lse_f), bits(lse))


@needs_device
@pytest.mark.parametrize("D", [64, 128])
def test_lse_graph_replay(D):
    """A CUDA graph captured after ``plan()`` replays the output and the LSE (kernel and fixup nodes)."""
    pr = Problem([512, 7, 1000, 1], [4096, 1025, 1000, 6000], 32, 8, D, 16, seed=4)
    ref_out, ref_lse = _both(run_paged_prefill(pr, run_kw=dict(LSE, lse_base="ln")))
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out = torch.empty_like(pr.q)
    lse = torch.empty(pr.q.shape[0], pr.Hq, dtype=torch.float32, device=pr.device)
    kw = dict(out=out, lse=lse, return_lse=True, lse_base="ln")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        w.run(pr.q, pr.cache("NHD"), **kw)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            w.run(pr.q, pr.cache("NHD"), **kw)
    torch.cuda.synchronize()
    for _ in range(3):
        out.zero_()
        lse.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref_out) and torch.equal(bits(lse), ref_lse)


@needs_device
def test_lse_sliding_window():
    pr = Problem([256, 1, 2000], [1500, 2500, 5000], 32, 8, 128, 16, seed=9)
    out, lse = _both(run_paged_prefill(pr, window_left=600, run_kw=LSE))
    (dec, dec_lse), last = run_decode(pr, window_left=600, run_kw=LSE)
    assert torch.equal(bits(dec), out[last]) and torch.equal(bits(dec_lse), lse[last])
    _, lse_r = _both(run_ragged_prefill(pr, window_left=600, run_kw=LSE))
    assert torch.equal(lse_r, lse)
    assert not torch.equal(lse, _both(run_paged_prefill(pr, run_kw=LSE))[1])
    _, ref = reference(pr, window_left=600, return_lse=True)
    _, ln = run_paged_prefill(pr, window_left=600, run_kw=dict(LSE, lse_base="ln"))
    assert (ln - ref).abs().max().item() < 1e-3 * max(1.0, ref.abs().max().item())


@needs_device
def test_lse_rows_are_never_empty():
    """Every row of the class sees at least its own key: a request with more query tokens than keys is refused at ``plan()``
    and a request without query tokens has no LSE row. The term's value at the fold's init state ``(-inf, +0)`` is ``-inf``
    (``lg2.approx.ftz(+0) = -inf``, ``mul.rn(-inf, c) = -inf``), the value a row without keys would take."""
    from flashinfer.batch_invariant.sampling.mufu_device import lg2_device

    probe = lg2_device(torch.tensor([0.0, 1.0]))
    assert torch.equal(bits(probe), bits(torch.tensor([float("-inf"), 0.0])))
    c = torch.tensor(128**-0.5 / math.log(2.0), dtype=torch.float32)
    assert (torch.tensor(float("-inf")) * c + probe[0]).item() == float("-inf")
    pr = Problem([4], [3], 8, 2, 128, 16, seed=46)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        _wrapper_case(1, pr)
    # a request without query tokens between two others: its neighbours' rows are the rows of the batch without it
    with_empty = Problem([5, 0, 1], [40, 9, 700], 8, 2, 128, 16, seed=47)
    out, lse = run_ragged_prefill(with_empty, run_kw=LSE)
    assert lse.shape == (6, 8) and bool(torch.isfinite(lse).all())
    without = Problem([5, 1], [40, 700], 8, 2, 128, 16, seed=47)
    without.q = with_empty.q
    without.k_rag = torch.cat([with_empty.k_rag[:40], with_empty.k_rag[49:]])
    without.v_rag = torch.cat([with_empty.v_rag[:40], with_empty.v_rag[49:]])
    out2, lse2 = run_ragged_prefill(without, run_kw=LSE)
    assert torch.equal(bits(out2), bits(out)) and torch.equal(bits(lse2), bits(lse))
    empty = Problem([], [], 8, 2, 128, 16, seed=38)
    for kind in range(3):
        w, args, _, _ = _wrapper_case(kind, empty)
        out, lse = w.run(*args, return_lse=True)
        assert out.shape == (0, 8, 128) and lse.shape == (0, 8)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_lse_buffer_refusals(kind):
    """A caller's ``lse`` buffer must be a contiguous fp32 tensor of the layout's shape on the wrapper's device whose bytes are
    disjoint from the output, the queries and the KV cache; anything else raises and the plan stays usable."""
    pr = Problem([1] * 4, [17, 700, 2500, 1], 8, 2, 128, 16, seed=48)
    w, args, _, _ = _wrapper_case(kind, pr)
    out, lse = w.run(*args, return_lse=True)
    expected = (bits(out).clone(), bits(lse).clone())
    dev = pr.device
    k = args[1][0] if kind < 2 else args[1]
    out_buf = torch.empty_like(pr.q)
    bad = [
        torch.empty(4, 8, dtype=torch.float64, device=dev),
        torch.empty(4, 8, dtype=torch.bfloat16, device=dev),
        torch.empty(8, 4, dtype=torch.float32, device=dev),
        torch.empty(32, dtype=torch.float32, device=dev),
        torch.empty(4, 16, dtype=torch.float32, device=dev)[:, ::2],
        torch.empty(4, 8, dtype=torch.float32),
        pr.q.view(torch.float32).reshape(-1)[:32].view(4, 8),
        k.view(torch.float32).reshape(-1)[:32].view(4, 8),
    ]
    for buf in bad:
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            w.run(*args, lse=buf, return_lse=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.run(
            *args,
            out=out_buf,
            lse=out_buf.view(torch.float32).reshape(-1)[:32].view(4, 8),
            return_lse=True,
        )
    if kind:
        with pytest.raises(ValueError):
            w.run(*args, return_lse=True, lse_base="log10")
        with pytest.raises(ValueError):
            w.run(*args, return_lse=True, lse_layout="NHD")
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            w.run(
                *args,
                lse=torch.empty(4, 8, dtype=torch.float32, device=dev),
                return_lse=True,
                lse_layout="HN",
            )
    # a strided q is packed into a copy for the kernels: a buffer over the caller's own q storage is refused all the same
    # (the store would change the caller's queries), a disjoint one is written
    wide = torch.zeros(4, 8, 256, dtype=torch.bfloat16, device=dev)
    q_strided = wide[:, :, ::2]
    q_strided.copy_(args[0])
    assert not q_strided.is_contiguous()
    strided_args = (q_strided,) + tuple(args[1:])
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.run(
            *strided_args,
            lse=wide.view(torch.float32).reshape(-1)[:32].view(4, 8),
            return_lse=True,
        )
    assert torch.equal(bits(q_strided), bits(args[0]))
    lse_buf = torch.empty(4, 8, dtype=torch.float32, device=dev)
    got = w.run(*strided_args, lse=lse_buf, return_lse=True)
    assert got[1] is lse_buf
    assert torch.equal(bits(got[0]), expected[0])
    assert torch.equal(bits(got[1]), expected[1])
    # the page table a run reads is the wrapper's own copy of the planned one
    own = w._batch_invariant._kv_indices
    if own.numel() >= 32:
        saved = own.clone()
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            w.run(*args, lse=own.view(torch.float32)[:32].view(4, 8), return_lse=True)
        assert torch.equal(own, saved)
    # a 4-D q that no view flattens is reshaped into a copy: the same rule for the caller's storage
    base = torch.zeros(2, 2, 8, 128, dtype=torch.bfloat16, device=dev)
    q_4d = base.transpose(0, 1)
    q_4d.copy_(args[0].view(2, 2, 8, 128))
    assert q_4d.reshape(-1, 8, 128).data_ptr() != base.data_ptr()
    args_4d = (q_4d,) + tuple(args[1:])
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.run(
            *args_4d,
            lse=base.view(torch.float32).reshape(-1)[:32].view(4, 8),
            return_lse=True,
        )
    assert torch.equal(bits(q_4d.reshape(-1, 8, 128)), bits(args[0]))
    got = w.run(*args_4d, lse=lse_buf, return_lse=True)
    assert got[1] is lse_buf
    assert torch.equal(bits(got[0]), expected[0])
    assert torch.equal(bits(got[1]), expected[1])
    got = w.run(*args, return_lse=True)
    assert torch.equal(bits(got[0]), expected[0])
    assert torch.equal(bits(got[1]), expected[1])
    from flashinfer.attention.batch_invariant import engine

    planned = engine.plan_attention(
        pr.qo_indptr, pr.kv_indptr, pr.kv_last, 8, 2, 128, 16, -1, None, None, dev
    )
    eargs = (planned, pr.q, pr.k_nhd, pr.v_nhd, pr.kv_indptr, pr.kv_indices)
    eargs += (128**-0.5,)
    for buf in bad[:3]:
        with pytest.raises(ValueError):
            engine.run_attention(*eargs, lse=buf, return_lse=True)
    with pytest.raises(ValueError):
        engine.run_attention(*eargs, return_lse=True, lse_base="e")
    # the engine reads the caller's kv_indices at the launch and compares the caller's kv_indptr with the plan's: an LSE
    # buffer over either is refused and the page table keeps its values; so is one over a table of the plan
    ptr_store = torch.zeros(64, dtype=torch.int32, device=dev)
    idx_store = torch.zeros(pr.kv_indices.numel() + 32, dtype=torch.int32, device=dev)
    indptr = ptr_store[: pr.kv_indptr.numel()].copy_(pr.kv_indptr)
    indices = idx_store[: pr.kv_indices.numel()].copy_(pr.kv_indices)
    targs = (planned, pr.q, pr.k_nhd, pr.v_nhd, indptr, indices, 128**-0.5)
    for store in (ptr_store, idx_store, planned.desc.view(-1)):
        saved = store.clone()
        over = store.view(torch.float32)[:32].view(4, 8)
        with pytest.raises(ValueError, match="page table"):
            engine.run_attention(*targs, lse=over, return_lse=True)
        assert torch.equal(store, saved)
    got = engine.run_attention(*targs, return_lse=True)
    assert torch.equal(bits(got[0]), expected[0])
    assert torch.equal(bits(got[1]), expected[1])


# ------------------------------------------------------------------------------------------------------ attention sinks
def sink_logits(pr: Problem, seed=0, spread=2.0):
    """Per-head sink logits around the LSE of the problems' rows, so a sink holds a visible share of its rows' mass."""
    g = torch.Generator(device="cpu").manual_seed(1000 + seed)
    return (8.0 + torch.randn(pr.Hq, generator=g) * spread).to(pr.device)


def sink_reference(pr: Problem, sinks, window_left=-1):
    """The repository's fp32 sink attention (``sink_attention_unified``) for the output and, next to it, the natural-log LSE
    of the scaled scores with the head's sink logit appended, ``(total_q, Hq)``."""
    from tests.attention.test_helpers.sink_attention_reference import (
        sink_attention_unified,
    )

    # the helper returns the query's dtype: fp32 queries (the bf16 values, exactly) keep the reference unrounded
    out = sink_attention_unified(
        pr.q.float(),
        pr.k_rag,
        pr.v_rag,
        sinks,
        window_left,
        True,
        1.0 / math.sqrt(pr.D),
        mode="varlen",
        qo_indptr=pr.qo_indptr,
        kv_indptr=pr.kv_indptr_rag,
    )
    lses = []
    tok_q = tok_kv = 0
    g = pr.Hq // pr.Hkv
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        q = pr.q[tok_q : tok_q + ql].float()
        k = pr.k_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(pr.D)
        pos = torch.arange(kl - ql, kl, device=pr.device)[:, None]
        key = torch.arange(kl, device=pr.device)[None, :]
        mask = key <= pos
        if window_left >= 0:
            mask &= key >= pos - window_left
        s = s.masked_fill(~mask[None], float("-inf"))
        s = torch.cat([s, sinks.float()[:, None, None].expand(-1, ql, 1)], dim=-1)
        lses.append(torch.logsumexp(s, dim=-1).t())
        tok_q += ql
        tok_kv += kl
    assert out.dtype == torch.float32
    return out, torch.cat(lses)


def _multi_token_decode(pr, q_len, layout="NHD", window_left=-1, **run_kw):
    """The decode wrapper over the last ``q_len`` tokens of every request; returns the result and the rows."""
    rows = torch.cat(
        [
            torch.arange(
                int(pr.qo_indptr[i + 1]) - q_len,
                int(pr.qo_indptr[i + 1]),
                device=pr.device,
            )
            for i in range(len(pr.q_lens))
        ]
    )
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), layout, batch_invariant=True
    )
    w.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        window_left=window_left,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_len_per_req=q_len,
    )
    return w.run(pr.q[rows].contiguous(), pr.cache(layout), **run_kw), rows


@pytest.mark.parametrize("op", WRAPPERS)
def test_sink_fingerprints_are_pinned(op):
    """The sink member's receipts (output and LSE, per head dim and base) next to the receipts of the class without sinks,
    which do not move."""
    from flashinfer.attention.batch_invariant import spec

    pins = {
        64: "attn-bf16-d64-sink-a239bf124352c39c",
        128: "attn-bf16-d128-sink-08ebe33f73cbe897",
        256: "attn-bf16-d256-sink-97a726a039ac102d",
    }
    lse_pins = {
        (64, "log2"): "attn-bf16-d64-sink-lse-log2-ce5540cea8c9060a",
        (64, "ln"): "attn-bf16-d64-sink-lse-ln-411e595f84dacdf6",
        (128, "log2"): "attn-bf16-d128-sink-lse-log2-d06a4faff7897362",
        (128, "ln"): "attn-bf16-d128-sink-lse-ln-911ada9a2df50aef",
        (256, "log2"): "attn-bf16-d256-sink-lse-log2-fcdaa847d25ab16a",
        (256, "ln"): "attn-bf16-d256-sink-lse-ln-8bd8ede7bafb8ca0",
    }
    assert pins == spec.SINK_FINGERPRINTS and lse_pins == spec.SINK_LSE_FINGERPRINTS
    c = dict(causal=True)
    for D, fp in pins.items():
        assert bi.fingerprint(op, head_dim=D, sinks=True, **c) == fp
        assert bi.fingerprint(op, head_dim=D, sinks=False, **c) == spec.FINGERPRINTS[D]
        assert bi.fingerprint(op, head_dim=D, **c) == spec.FINGERPRINTS[D]
    for (D, base), fp in lse_pins.items():
        assert bi.fingerprint(op, head_dim=D, lse_base=base, sinks=True, **c) == fp
        assert (
            bi.fingerprint(op, head_dim=D, lse_base=base, **c)
            == spec.LSE_FINGERPRINTS[(D, base)]
        )
    with pytest.raises(ValueError):
        bi.fingerprint(op, head_dim=128, sinks="yes", **c)


def test_sink_kernels_have_their_own_names():
    """The sink step is compiled into kernels of their own: a call without sinks loads the kernels it always did."""
    from flashinfer.attention.batch_invariant.blocks import compile as row
    from flashinfer.attention.batch_invariant.swaps import compile as swaps

    assert row.kernel_name(128, 4, 16, 4) == "attn_d128_g4_p16_s4_w1"
    assert row.kernel_name(128, 4, 16, 4, sinks=True) == "attn_d128_g4_p16_s4_w1_sink"
    plain = swaps.kernel_name(128, 8, 16, 8, False, 228 << 10, True)
    assert swaps.kernel_name(128, 8, 16, 8, False, 228 << 10, True, sinks=True) == (
        plain + "_sink"
    )


def test_sink_term_is_declared():
    """The sink step on the row's final ``(m, l)``: ``m' = fmax(mul(m, c), mul(sink, LOG2E))``, the exponentials of the exact
    differences to ``m'``, ``l' = fma(l, a, b)``; the epilogue's normaliser is ``mul(a, div(1, l'))`` and the LSE
    ``add(m', lg2(l'))``. The terms without sinks contain no sink."""
    import struct

    from flashinfer.attention.batch_invariant import spec
    from flashinfer.batch_invariant.fp_interface.term import sexpr

    assert struct.unpack("<I", struct.pack("<f", spec.LOG2E))[0] == 0x3FB8AA3B

    def comm(op, *xs):
        return f"({op} " + " ".join(sorted(xs)) + ")"

    for D, p, win in ((64, 7, -1), (128, 40, -1), (256, 33, 5)):
        s = spec.AttentionSpec(D, T=16, C=32)
        row = spec.RowSpec(p=p, window_left=win, row=1)
        st = s.state(row, 2)
        m, l, o = sexpr(st.m), sexpr(st.l), sexpr(st.o)
        ms = comm("mul", m, "(sym c)")
        s2 = comm("mul", "(sym sink)", f"(const {spec.LOG2E!r})")
        mx = comm("fmax", ms, s2)
        nm = comm("mul", mx, "(const -1.0)")
        a = f"(ex2 {comm('add', ms, nm)})"
        b = f"(ex2 {comm('add', s2, nm)})"
        ls = "(fma " + " ".join(sorted([l, a])) + f" {b})"
        inv = comm("mul", a, f"(div (const 1.0) {ls})")
        out = f"(cvt bf16 {comm('add', comm('mul', o, inv), '(const 0.0)')})"
        assert sexpr(s.output(row, 2, sinks=True)) == out
        lse2 = comm("add", mx, f"(op:lg2 {ls})")
        assert sexpr(s.lse(row, "log2", sinks=True)) == lse2
        assert sexpr(s.lse(row, "ln", sinks=True)) == comm(
            "mul", lse2, f"(const {spec.LN2!r})"
        )
        for plain in (s.output(row, 2), s.lse(row, "log2"), s.lse(row, "ln")):
            assert "sink" not in sexpr(plain)
        assert s.output(row, 2) is not s.output(row, 2, sinks=True)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_sinks_match_reference(D, Hq, Hkv):
    pr = Problem([300, 1000, 64], [700, 2500, 3000], Hq, Hkv, D, 16, seed=1)
    sinks = sink_logits(pr, seed=D)
    ref, ref_lse = sink_reference(pr, sinks)
    out, ln = run_paged_prefill(pr, run_kw=dict(LSE, lse_base="ln", sinks=sinks))
    _, log2 = run_paged_prefill(pr, run_kw=dict(LSE, sinks=sinks))
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err
    tol = 1e-3 * max(1.0, ref_lse.abs().max().item())
    assert (ln - ref_lse).abs().max().item() < tol
    assert (log2 - ref_lse / math.log(2.0)).abs().max().item() < tol / math.log(2.0)
    # the sinks are in the result: the same batch without them differs, and by more than the tolerance
    plain = run_paged_prefill(pr)
    assert (plain.float() - ref).abs().max().item() > 4 * err


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_sinks_prefill_equals_decode_equals_ragged(D, layout):
    """R4 with sinks: a token's output and LSE are the same bits as a paged prefill row, a ragged prefill row (the wrapper's
    ``_sinks``), a decode step and a row of a multi-token decode."""
    pr = Problem([256, 4, 1024, 33], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=2)
    sinks = sink_logits(pr, seed=D)
    kw = dict(LSE, sinks=sinks)
    out, lse = _both(run_paged_prefill(pr, layout, run_kw=kw))
    out_r, lse_r = _both(run_ragged_prefill(pr, layout, run_kw=LSE, sinks=sinks))
    assert torch.equal(out, out_r) and torch.equal(lse, lse_r)
    (dec, dec_lse), last = run_decode(pr, layout, run_kw=kw)
    assert torch.equal(bits(dec), out[last]) and torch.equal(bits(dec_lse), lse[last])
    res, last4 = _multi_token_decode(pr, 4, layout, **kw)
    mtd, mtd_lse = _both(res)
    assert torch.equal(mtd, out[last4]) and torch.equal(mtd_lse, lse[last4])
    assert not torch.equal(out, bits(run_paged_prefill(pr, layout)))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_no_sinks_is_the_class_without_the_sink_step(D, kind):
    """``sinks=None`` runs the class without sinks: the bits of the wrapper before and after a call with sinks. A sink of
    ``-inf`` adds ``+0`` to the sum and rescales by ``1.0``: the same bits again, output and LSE."""
    pr = Problem([1] * 6, [1024, 4096, 33, 700, 2500, 1], 32, 8, D, 16, seed=40)
    w, args, _, _ = _wrapper_case(kind, pr)
    plain, plain_lse = _both(w.run(*args, return_lse=True))
    plain, plain_lse = plain.clone(), plain_lse.clone()

    def run(sinks):
        if kind == 2:
            w._sinks = sinks
            return _both(w.run(*args, return_lse=True))
        return _both(w.run(*args, return_lse=True, sinks=sinks))

    out, lse = run(sink_logits(pr))
    assert not torch.equal(out, plain) and not torch.equal(lse, plain_lse)
    out, lse = run(None)
    assert torch.equal(out, plain) and torch.equal(lse, plain_lse)
    out, lse = run(torch.full((pr.Hq,), float("-inf"), device=pr.device))
    assert torch.equal(out, plain) and torch.equal(lse, plain_lse)
    if kind == 2:
        w._sinks = None
    assert torch.equal(bits(w.run(*args)), plain)
    # prefill-shaped units (fused multi-chunk, split groups, the paired placement at D = 64)
    big = Problem([2048, 300, 1], [4096, 700, 3000], 32, 8, D, 16, seed=41)
    plain = _both(run_paged_prefill(big, run_kw=LSE))
    off = torch.full((big.Hq,), float("-inf"), device=big.device)
    got = _both(run_paged_prefill(big, run_kw=dict(LSE, sinks=off)))
    assert torch.equal(got[0], plain[0]) and torch.equal(got[1], plain[1])


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sinks_batch_permutation_and_subbatch_invariance(D):
    """R2 with sinks: a request's output and LSE bits do not depend on its neighbours, its position in the batch or the
    batch size, in prefill and in decode."""
    q_lens, kv_lens = [300, 1000, 64, 2048, 1], [700, 2500, 3000, 2048, 5000]
    pr = Problem(q_lens, kv_lens, 32, 8, D, 16, seed=5)
    sinks = sink_logits(pr, seed=D)
    kw = dict(LSE, sinks=sinks)
    out, full = _both(run_paged_prefill(pr, run_kw=kw))
    rows = [
        slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1]))
        for i in range(len(q_lens))
    ]
    kv = [
        slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1]))
        for i in range(len(q_lens))
    ]
    for i in (0, 2, 4):
        single = Problem([q_lens[i]], [kv_lens[i]], 32, 8, D, 16, seed=5)
        single.q = pr.q[rows[i]].clone()
        single.k_rag = pr.k_rag[kv[i]].clone()
        single.v_rag = pr.v_rag[kv[i]].clone()
        o1, l1 = _both(run_ragged_prefill(single, run_kw=LSE, sinks=sinks))
        assert torch.equal(l1, full[rows[i]]) and torch.equal(o1, out[rows[i]]), i
    perm = [3, 0, 4, 1, 2]
    pp = Problem(
        [q_lens[i] for i in perm], [kv_lens[i] for i in perm], 32, 8, D, 16, seed=5
    )
    pp.q = torch.cat([pr.q[rows[i]] for i in perm]).contiguous()
    pp.k_rag = torch.cat([pr.k_rag[kv[i]] for i in perm]).contiguous()
    pp.v_rag = torch.cat([pr.v_rag[kv[i]] for i in perm]).contiguous()
    p_out, p_lse = _both(run_ragged_prefill(pp, run_kw=LSE, sinks=sinks))
    off = 0
    for i in perm:
        n = q_lens[i]
        assert torch.equal(p_lse[off : off + n], full[rows[i]]), i
        assert torch.equal(p_out[off : off + n], out[rows[i]]), i
        off += n
    # decode: a request in a batch of eight and alone (a fresh page table)
    kvd = [1024, 4096, 33, 700, 2500, 8192, 1, 1500]
    dp = Problem([1] * 8, kvd, 32, 8, D, 16, seed=6)
    (out8, lse8), _ = run_decode(dp, run_kw=kw)
    for i in (0, 1, 5, 6):
        one = Problem([1], [kvd[i]], 32, 8, D, 16, seed=100 + i)
        one.q = dp.q[i : i + 1].clone()
        span = slice(int(dp.kv_indptr_rag[i]), int(dp.kv_indptr_rag[i + 1]))
        one.k_rag = dp.k_rag[span].clone()
        one.v_rag = dp.v_rag[span].clone()
        one = _repack(one)
        (out1, lse1), _ = run_decode(one, run_kw=kw)
        assert torch.equal(bits(out1), bits(out8[i : i + 1])), i
        assert torch.equal(bits(lse1), bits(lse8[i : i + 1])), i


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sinks_chunked_prefill_equals_full(D):
    """R4 for incremental prefill with sinks: the step is taken once per row after its last chunk, however the prefill was
    cut."""
    kv = [1700, 1000]
    cut = [700, 512]
    full = Problem(kv, kv, 32, 8, D, 16, seed=12)
    sinks = sink_logits(full, seed=D)
    kw = dict(LSE, sinks=sinks)
    out_full, lse_full = _both(run_paged_prefill(full, run_kw=kw))
    rows = [slice(int(full.qo_indptr[i]), int(full.qo_indptr[i + 1])) for i in range(2)]
    kv_off = [int(full.kv_indptr_rag[i]) for i in range(2)]
    c1 = Problem(cut, cut, 32, 8, D, 16, seed=13)
    c1.q = torch.cat([full.q[rows[i]][: cut[i]] for i in range(2)]).contiguous()
    c1.k_rag = torch.cat(
        [full.k_rag[kv_off[i] : kv_off[i] + cut[i]] for i in range(2)]
    ).contiguous()
    c1.v_rag = torch.cat(
        [full.v_rag[kv_off[i] : kv_off[i] + cut[i]] for i in range(2)]
    ).contiguous()
    c1 = _repack(c1)
    c2 = Problem([kv[i] - cut[i] for i in range(2)], kv, 32, 8, D, 16, seed=14)
    c2.q = torch.cat([full.q[rows[i]][cut[i] :] for i in range(2)]).contiguous()
    c2.k_rag, c2.v_rag = full.k_rag.clone(), full.v_rag.clone()
    c2 = _repack(c2)
    for prob, part in ((c1, 0), (c2, 1)):
        got_p = _both(run_paged_prefill(prob, run_kw=kw))
        got_r = _both(run_ragged_prefill(prob, run_kw=LSE, sinks=sinks))
        off = 0
        for i in range(2):
            n = cut[i] if part == 0 else kv[i] - cut[i]
            sel = slice(0, cut[i]) if part == 0 else slice(cut[i], None)
            for got in (got_p, got_r):
                assert torch.equal(got[0][off : off + n], out_full[rows[i]][sel])
                assert torch.equal(got[1][off : off + n], lse_full[rows[i]][sel])
            off += n


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sinks_config_agreement(D):
    """R3 with sinks: every kernel site that completes a row takes the same sink step -- the fused unit's epilogue, the fixup
    kernel, the in-kernel fold, the Swaps body and (D = 64) the paired placement -- over the configuration space on a
    prefill-shaped and a decode-shaped batch; the partials of split units carry no sink."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    if D == 64:
        prefill = Problem([2048, 2048, 1], [4096, 3000, 1000], 32, 8, D, 16, seed=8)
    else:
        prefill = Problem(
            [256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, seed=8
        )
    decode = Problem([1] * 6, [1024, 4096, 33, 700, 2500, 1], 32, 8, D, 16, seed=42)
    sinks = sink_logits(prefill, seed=D)
    space = engine.all_valid_configs(D)
    cfgs = list(space[::7] if D == 64 else space)
    if D == 64:
        two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
        rings = (2, 3) + ((4,) if arch_smem_cap(prefill.device) >= (300 << 10) else ())
        cfgs += [replace(two, kv_slots=s) for s in rings]
    sites = set()
    for pr in (prefill, decode):
        ref = {
            base: _both(
                run_paged_prefill(pr, run_kw=dict(LSE, lse_base=base, sinks=sinks))
            )
            for base in ("log2", "ln")
        }
        plain = bits(run_paged_prefill(pr))
        assert not torch.equal(ref["log2"][0], plain)
        for n, cfg in enumerate(cfgs):
            planned = engine.plan_attention(
                pr.qo_indptr,
                pr.kv_indptr,
                pr.kv_last,
                32,
                8,
                D,
                16,
                -1,
                cfg,
                None,
                pr.device,
            )
            plan = planned.plan
            swaps = cfg.small == "swap" and engine.swaps_n_pad(plan) > 0
            body = "swaps" if swaps else "row"
            if any(d[6] == 0 for d in plan.desc):
                sites.add(f"{body}-fused")
            if plan.fixup_groups:
                sites.add(f"{body}-fixup")
            if any(plan.group_inkernel):
                sites.add(f"{body}-inkernel")
            if cfg.q2 and plan.num_launch is not None:
                sites.add("paired")
            eargs = (planned, pr.q, pr.k_nhd, pr.v_nhd, pr.kv_indptr, pr.kv_indices)
            eargs += (1.0 / math.sqrt(D), -1, "NHD", None, cfg)
            base = "ln" if n % 4 == 0 else "log2"
            got = engine.run_attention(
                *eargs, return_lse=True, lse_base=base, sinks=sinks
            )
            out, lse = _both(got)
            assert torch.equal(out, ref[base][0]), (cfg, base)
            assert torch.equal(lse, ref[base][1]), (cfg, base)
            if n % 4 == 0:
                # the same plan without sinks, after a launch with them
                assert torch.equal(bits(engine.run_attention(*eargs)), plain), cfg
    want = {
        f"{b}-{k}" for b in ("row", "swaps") for k in ("fused", "fixup", "inkernel")
    }
    if D == 64:
        want.add("paired")
    assert sites >= want, want - sites


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sinks_head_subset_invariance(D):
    """R2 across heads with per-head sinks: a subset of the KV heads (whole GQA groups) with the slice of the sinks gives,
    head for head, the bits of the run over all heads -- the sink is read by the row's query head."""
    Hq, Hkv = 32, 8
    g = Hq // Hkv
    pr = Problem([300, 1, 1000, 64], [700, 2500, 2500, 3000], Hq, Hkv, D, 16, seed=16)
    sinks = sink_logits(pr, seed=D, spread=3.0)
    kw = dict(LSE, sinks=sinks)
    full_pre = _both(run_paged_prefill(pr, run_kw=kw))
    res, _ = run_decode(pr, run_kw=kw)
    full_dec = _both(res)
    for kv_sel in ([1, 3, 4, 6], [5]):
        q_sel = [j * g + r for j in kv_sel for r in range(g)]
        sub = Problem(pr.q_lens, pr.kv_lens, len(q_sel), len(kv_sel), D, 16, seed=17)
        sub.q = pr.q[:, q_sel].contiguous()
        sub.k_rag = pr.k_rag[:, kv_sel].contiguous()
        sub.v_rag = pr.v_rag[:, kv_sel].contiguous()
        sub = _repack(sub)
        sub_sinks = sinks[q_sel].contiguous()
        sub_kw = dict(LSE, sinks=sub_sinks)
        got = [
            _both(run_paged_prefill(sub, run_kw=sub_kw)),
            _both(run_ragged_prefill(sub, run_kw=LSE, sinks=sub_sinks)),
        ]
        for out, lse in got:
            assert torch.equal(out, full_pre[0][:, q_sel]), (D, kv_sel)
            assert torch.equal(lse, full_pre[1][:, q_sel]), (D, kv_sel)
        res, _ = run_decode(sub, run_kw=sub_kw)
        out, lse = _both(res)
        assert torch.equal(out, full_dec[0][:, q_sel]), (D, kv_sel, "decode")
        assert torch.equal(lse, full_dec[1][:, q_sel]), (D, kv_sel, "decode")
    # the sink belongs to the head: permuting the sinks changes the heads whose sink changed, and only those
    swapped = sinks.clone()
    swapped[[0, 1]] = sinks[[1, 0]]
    out, _ = _both(run_paged_prefill(pr, run_kw=dict(LSE, sinks=swapped)))
    assert torch.equal(out[:, 2:], full_pre[0][:, 2:])
    assert not torch.equal(out[:, 0], full_pre[0][:, 0])


@needs_device
def test_sinks_sliding_window_gpt_oss_shape():
    """Head dim 64, 64 query heads over 8 KV heads, a 128-token window and per-head sinks: the sink joins the window's keys
    after its last chunk; prefill (paged and ragged), decode and multi-token decode agree bit for bit and match the fp32
    reference."""
    pr = Problem([700, 4, 300, 128], [700, 2500, 1500, 129], 64, 8, 64, 16, seed=50)
    sinks = sink_logits(pr, seed=50)
    win = 128
    kw = dict(LSE, lse_base="ln", sinks=sinks)
    res = run_paged_prefill(pr, window_left=win, run_kw=kw)
    out, lse = _both(res)
    got_r = _both(
        run_ragged_prefill(
            pr, window_left=win, run_kw=dict(LSE, lse_base="ln"), sinks=sinks
        )
    )
    assert torch.equal(got_r[0], out) and torch.equal(got_r[1], lse)
    _, log2 = _both(
        run_paged_prefill(pr, window_left=win, run_kw=dict(LSE, sinks=sinks))
    )
    (dec, dec_lse), last = run_decode(
        pr, window_left=win, run_kw=dict(LSE, sinks=sinks)
    )
    assert torch.equal(bits(dec), out[last]) and torch.equal(bits(dec_lse), log2[last])
    res4, last4 = _multi_token_decode(pr, 4, window_left=win, **dict(LSE, sinks=sinks))
    mtd, mtd_lse = _both(res4)
    assert torch.equal(mtd, out[last4]) and torch.equal(mtd_lse, log2[last4])
    ref, ref_lse = sink_reference(pr, sinks, window_left=win)
    err = (res[0].float() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err
    assert (res[1] - ref_lse).abs().max().item() < 1e-3 * max(
        1.0, ref_lse.abs().max().item()
    )
    assert not torch.equal(out, bits(run_paged_prefill(pr, run_kw=dict(sinks=sinks))))
    assert not torch.equal(out, bits(run_paged_prefill(pr, window_left=win)))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sinks_lse_natural_log_is_the_declared_multiply(D):
    from flashinfer.attention.batch_invariant.spec import LN2

    pr = Problem([300, 1, 1000, 64], [700, 2500, 2500, 3000], 32, 8, D, 16, seed=43)
    sinks = sink_logits(pr, seed=D)
    ln2 = torch.tensor(LN2, dtype=torch.float32)
    out2, log2 = run_paged_prefill(pr, run_kw=dict(LSE, sinks=sinks))
    oute, ln = run_paged_prefill(pr, run_kw=dict(LSE, lse_base="ln", sinks=sinks))
    assert torch.equal(bits(log2.cpu() * ln2), bits(ln.cpu()))
    assert torch.equal(bits(out2), bits(oute))
    plain = bits(run_paged_prefill(pr, run_kw=dict(sinks=sinks)))
    assert torch.equal(plain, bits(out2))
    _, hn = run_paged_prefill(pr, run_kw=dict(LSE, sinks=sinks, lse_layout="HN"))
    assert torch.equal(bits(hn), bits(log2.t().contiguous()))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sinks_edge_values(D):
    """The step's corner values. A sink far above every score: ``a`` underflows to ``+0``, the outputs are ``+0`` and the
    base-2 LSE is ``mul(sink, LOG2E)`` (``l' = 1``). A sink far below (or ``-inf``): ``b = +0``, the bits of the row without
    sinks. A sink whose scaled logit equals the row's scaled maximum: ``a = b = 1``. ``+inf`` and NaN sinks give NaN on
    their heads only. The step sees the sink's base-2 value ``mul(sink, LOG2E)``: a finite sink whose base-2 value
    overflows binary32 is the sink of that infinity (NaN on its head above, the row without sinks below)."""
    from flashinfer.attention.batch_invariant.spec import LOG2E
    from flashinfer.batch_invariant.sampling.mufu_device import lg2_device

    pr = Problem([300, 1, 700, 1], [700, 2500, 700, 1], 32, 8, D, 16, seed=51)
    plain, plain_lse = _both(run_paged_prefill(pr, run_kw=LSE))
    mixed = sink_logits(pr, seed=D)
    ref, ref_lse = _both(run_paged_prefill(pr, run_kw=dict(LSE, sinks=mixed)))
    sinks = mixed.clone()
    sinks[0] = 1.0e4
    sinks[1] = -1.0e4
    sinks[2] = float("-inf")
    sinks[3] = float("inf")
    sinks[4] = float("nan")
    sinks[5] = 3.0e38
    sinks[6] = -3.0e38
    log2e = torch.tensor(LOG2E, dtype=torch.float32)
    assert bool(torch.isfinite(sinks[5:7]).all())
    assert (sinks[5:7].cpu() * log2e).tolist() == [float("inf"), float("-inf")]
    res = run_paged_prefill(pr, run_kw=dict(LSE, sinks=sinks))
    out, lse = _both(res)
    assert torch.equal(out[:, 7:], ref[:, 7:]) and torch.equal(
        lse[:, 7:], ref_lse[:, 7:]
    )
    assert bool((out[:, 0] == 0).all())  # +0, bit for bit
    big = torch.tensor(1.0e4, dtype=torch.float32) * torch.tensor(
        LOG2E, dtype=torch.float32
    )
    assert bool((res[1][:, 0].cpu() == big).all())
    for h in (1, 2, 6):
        assert torch.equal(out[:, h], plain[:, h])
        assert torch.equal(lse[:, h], plain_lse[:, h])
    for h in (3, 4, 5):
        assert bool(torch.isnan(res[0][:, h].float()).all())
        assert bool(torch.isnan(res[1][:, h]).all())
    (dec, dec_lse), last = run_decode(pr, run_kw=dict(LSE, sinks=sinks))
    assert bool(torch.isnan(dec[:, 5].float()).all())
    assert bool(torch.isnan(dec_lse[:, 5]).all())
    keep = [h for h in range(pr.Hq) if h not in (3, 4, 5)]
    assert torch.equal(bits(dec)[:, keep], out[last][:, keep])
    assert torch.equal(bits(dec_lse)[:, keep], lse[last][:, keep])
    # a row with one key has l = 1 and its base-2 LSE without sinks is its scaled maximum mul(m, c), bit for bit; where a
    # binary32 sink has exactly that scaled logit, the sink step is the merge of two equal maxima
    one = Problem([1], [1], 32, 8, D, 16, seed=52)
    (o1, l1), _ = run_decode(one, run_kw=LSE)
    ms = l1[0].cpu()
    log2e = torch.tensor(LOG2E, dtype=torch.float32)
    guess = ms / log2e
    eq = torch.full((one.Hq,), float("-inf"))
    for step in (0, 1, -1, 2, -2):
        cand = guess
        for _ in range(abs(step)):
            cand = torch.nextafter(cand, torch.tensor(math.copysign(math.inf, step)))
        hit = (cand * log2e == ms) & torch.isinf(eq)
        eq = torch.where(hit, cand, eq)
    hits = torch.isfinite(eq)
    assert int(hits.sum()) >= 4, int(hits.sum())
    (o2, l2), _ = run_decode(one, run_kw=dict(LSE, sinks=eq.to(one.device)))
    lg2_two = lg2_device(torch.tensor([2.0]))[0].cpu()
    half = (o1[0].float() * 0.5).to(torch.bfloat16)
    assert torch.equal(bits(o2[0][hits]), bits(half[hits]))
    assert torch.equal(bits(l2[0].cpu()[hits]), bits((ms + lg2_two)[hits]))
    assert torch.equal(bits(o2[0][~hits]), bits(o1[0][~hits]))


@needs_device
@pytest.mark.parametrize("D", [64, 128])
def test_sinks_graph_replay_reads_the_current_sinks(D):
    """A CUDA graph captured after ``plan()`` holds the sinks tensor, not its values: a replay after an in-place update of
    the sinks is the launch with the new values (kernel and fixup nodes)."""
    pr = Problem([512, 7, 1000, 1], [4096, 1025, 1000, 6000], 32, 8, D, 16, seed=4)
    first, second = sink_logits(pr, seed=1), sink_logits(pr, seed=2)
    refs = [
        _both(run_paged_prefill(pr, run_kw=dict(LSE, lse_base="ln", sinks=s)))
        for s in (first, second)
    ]
    assert not torch.equal(refs[0][0], refs[1][0])
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    out = torch.empty_like(pr.q)
    lse = torch.empty(pr.q.shape[0], pr.Hq, dtype=torch.float32, device=pr.device)
    sinks = first.clone()
    kw = dict(out=out, lse=lse, return_lse=True, lse_base="ln", sinks=sinks)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        w.run(pr.q, pr.cache("NHD"), **kw)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            w.run(pr.q, pr.cache("NHD"), **kw)
    torch.cuda.synchronize()
    for values, ref in ((first, refs[0]), (second, refs[1]), (first, refs[0])):
        sinks.copy_(values)
        out.zero_()
        lse.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref[0]) and torch.equal(bits(lse), ref[1])


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_sinks_refusals(kind):
    """``sinks`` must be a contiguous fp32 ``(num_qo_heads,)`` tensor on the wrapper's device; anything else raises and the
    plan stays usable."""
    pr = Problem([1] * 4, [17, 700, 2500, 1], 8, 2, 128, 16, seed=48)
    w, args, _, _ = _wrapper_case(kind, pr)
    dev = pr.device
    good = sink_logits(pr)

    def run(sinks):
        if kind == 2:
            w._sinks = sinks
            return w.run(*args)
        return w.run(*args, sinks=sinks)

    expected = bits(run(good)).clone()
    bad = [
        good.double(),
        good.to(torch.bfloat16),
        good.to(torch.float16),
        good.to(torch.int32),
        torch.zeros(9, device=dev),
        torch.zeros(2, device=dev),
        torch.zeros(1, 8, device=dev),
        torch.zeros((), device=dev),
        torch.zeros(16, device=dev)[::2],
        good.cpu(),
        good.tolist(),
        1.0,
    ]
    for sinks in bad:
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            run(sinks)
    out_buf = torch.empty_like(pr.q)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        if kind == 2:
            w._sinks = out_buf.view(torch.float32).reshape(-1)[:8]
            w.run(*args, out=out_buf)
        else:
            w.run(*args, out=out_buf, sinks=out_buf.view(torch.float32).reshape(-1)[:8])
    # sinks inside the caller's LSE buffer: refused in both layouts (under "HN" the kernels write a (total_q, Hq) buffer
    # of their own and the transposed copy lands in the caller's), accepted where the buffer is not written (no return_lse)
    for layout in ("NH", "HN") if kind else ("NH",):
        lse_buf = torch.zeros(
            (4, 8) if layout == "NH" else (8, 4), dtype=torch.float32, device=dev
        )
        inside = lse_buf.view(-1)[8:16]
        inside.copy_(good)
        lse_kw = dict(lse=lse_buf, lse_layout=layout) if kind else dict(lse=lse_buf)
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            if kind == 2:
                w._sinks = inside
                w.run(*args, return_lse=True, **lse_kw)
            else:
                w.run(*args, return_lse=True, sinks=inside, **lse_kw)
        assert torch.equal(inside, good)
        if kind == 2:
            w._sinks = inside
            unused = w.run(*args, **lse_kw)
        else:
            unused = w.run(*args, sinks=inside, **lse_kw)
        assert torch.equal(bits(unused), expected)
        assert torch.equal(inside, good)
    assert torch.equal(bits(run(good)), expected)
    from flashinfer.attention.batch_invariant import engine

    planned = engine.plan_attention(
        pr.qo_indptr, pr.kv_indptr, pr.kv_last, 8, 2, 128, 16, -1, None, None, dev
    )
    eargs = (planned, pr.q, pr.k_nhd, pr.v_nhd, pr.kv_indptr, pr.kv_indices)
    eargs += (128**-0.5,)
    for sinks in bad[:10]:
        with pytest.raises(ValueError):
            engine.run_attention(*eargs, sinks=sinks)
    assert torch.equal(bits(engine.run_attention(*eargs, sinks=good)), expected)
    # the LSE buffer counts as written only when the LSE is requested
    lse_buf = torch.zeros(4, 8, dtype=torch.float32, device=dev)
    inside = lse_buf.view(-1)[8:16]
    inside.copy_(good)
    with pytest.raises(ValueError):
        engine.run_attention(*eargs, sinks=inside, lse=lse_buf, return_lse=True)
    unused = engine.run_attention(*eargs, sinks=inside, lse=lse_buf)
    assert torch.equal(bits(unused), expected)
    assert torch.equal(inside, good)


def _run_sinks(kind, w, args, sinks, **kw):
    """``run()`` with ``sinks`` the way each wrapper takes them (``None``: without)."""
    if kind == 2:
        w._sinks = sinks
        return w.run(*args, **kw)
    return w.run(*args, **kw) if sinks is None else w.run(*args, sinks=sinks, **kw)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_buffers_inside_the_workspace_are_refused(kind):
    """A launch stores its split-KV partials in the float workspace buffer: an ``lse`` buffer (either layout), ``sinks``
    or an output whose bytes lie anywhere in that buffer raise, and the plan stays usable."""
    pr = Problem([1] * 4, [17, 700, 2500, 1], 8, 2, 128, 16, seed=48)
    dev = pr.device
    ws = workspace(dev, mb=8)
    w, args, _, _ = _wrapper_case(kind, pr, ws=ws)
    good = sink_logits(pr)
    expected = _both(_run_sinks(kind, w, args, good, return_lse=True))
    expected = (expected[0].clone(), expected[1].clone())
    f32 = ws.view(torch.float32)
    sinks_in = f32[-8:]
    sinks_in.copy_(good)
    lse_bufs = [(f32[:32].view(4, 8), {}), (f32[-64:-32].view(4, 8), {})]
    if kind:
        lse_bufs.append((f32[:32].view(8, 4), dict(lse_layout="HN")))
        lse_bufs.append((f32[-64:-32].view(8, 4), dict(lse_layout="HN")))
    out_in = ws.view(torch.bfloat16)[-4 * 8 * 128 :].view(4, 8, 128)
    for buf, kw in lse_bufs:
        for sinks in (None, good):
            with pytest.raises(bi.BatchInvariantUnsupportedError):
                _run_sinks(kind, w, args, sinks, lse=buf, return_lse=True, **kw)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        _run_sinks(kind, w, args, sinks_in, return_lse=True)
    for sinks in (None, good):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            _run_sinks(kind, w, args, sinks, out=out_in)
    assert torch.equal(sinks_in, good)
    got = _both(_run_sinks(kind, w, args, good, return_lse=True))
    assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])
    from flashinfer.attention.batch_invariant import engine

    planned = engine.plan_attention(
        pr.qo_indptr, pr.kv_indptr, pr.kv_last, 8, 2, 128, 16, -1, None, ws, dev
    )
    storage = engine.plan_storage(planned)
    assert all(any(t is x for x in storage) for t in (ws, planned.ws, planned.scratch))
    eargs = (planned, pr.q, pr.k_nhd, pr.v_nhd, pr.kv_indptr, pr.kv_indices)
    eargs += (128**-0.5,)
    for buf, _ in lse_bufs[:2]:
        with pytest.raises(ValueError, match="workspace"):
            engine.run_attention(*eargs, sinks=good, lse=buf, return_lse=True)
    with pytest.raises(ValueError, match="workspace"):
        engine.run_attention(*eargs, sinks=sinks_in)
    with pytest.raises(ValueError, match="workspace"):
        engine.run_attention(*eargs, sinks=good, out=out_in)
    got = _both(engine.run_attention(*eargs, sinks=good, return_lse=True))
    assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_has_sinks_declaration(kind):
    """``plan(has_sinks=True | False)`` declares which member every ``run()`` of the plan is: a run that departs from it
    raises, ``None`` leaves the choice to each run, and a new ``plan()`` replaces the declaration."""
    pr = Problem([1] * 4, [17, 700, 2500, 1], 8, 2, 128, 16, seed=49)
    good = sink_logits(pr)
    options = {} if kind == 0 else {"causal": True}
    w, args, plan_args, kwargs = _wrapper_case(kind, pr)
    plain = bits(_run_sinks(kind, w, args, None)).clone()
    sunk = bits(_run_sinks(kind, w, args, good)).clone()
    assert not torch.equal(plain, sunk)

    # declared with sinks: a wrapper that was never given any (the ragged ``_sinks`` unset) raises on every entry point
    w, args, plan_args, kwargs = _wrapper_case(kind, pr, has_sinks=True)
    for call in (w.run, w.run_return_lse):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="has_sinks=True"):
            call(*args)
    for call in (w.forward, w.forward_return_lse):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="has_sinks=True"):
            call(*args, **options)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="has_sinks=True"):
        _run_sinks(kind, w, args, None)
    assert torch.equal(bits(_run_sinks(kind, w, args, good)), sunk)
    if kind == 2:  # the deprecated entry points read the same attribute
        assert torch.equal(bits(w.forward(*args, **options)), sunk)

    w.plan(*plan_args, **{**kwargs, "has_sinks": False})
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="has_sinks=False"):
        _run_sinks(kind, w, args, good)
    if kind == 2:
        for call in (w.forward, w.forward_return_lse):
            with pytest.raises(
                bi.BatchInvariantUnsupportedError, match="has_sinks=False"
            ):
                call(*args, **options)
    assert torch.equal(bits(_run_sinks(kind, w, args, None)), plain)
    assert torch.equal(bits(w.forward(*args, **options)), plain)

    kwargs.pop("has_sinks")
    w.plan(*plan_args, **kwargs)
    assert torch.equal(bits(_run_sinks(kind, w, args, good)), sunk)
    assert torch.equal(bits(_run_sinks(kind, w, args, None)), plain)


@needs_device
def test_has_sinks_declaration_under_graph_capture():
    """The declaration is checked in Python at ``run()``: a captured ``run()`` replays as without it."""
    pr = Problem([1] * 4, [17, 700, 2500, 1], 8, 2, 128, 16, seed=50)
    good = sink_logits(pr)
    w, args, _, _ = _wrapper_case(1, pr, has_sinks=True)
    expected = bits(w.run(*args, sinks=good)).clone()
    out = torch.empty_like(pr.q)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(*args, sinks=good, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), expected)


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
@pytest.mark.parametrize("declared", [True, False])
def test_has_sinks_is_refused_without_batch_invariant(kind, declared):
    """The declaration belongs to the batch-invariant plan: the regular wrappers refuse a value instead of ignoring it."""
    pr = Problem([1] * 4, [17, 700, 2500, 1], 8, 2, 128, 16, seed=51)
    w = getattr(flashinfer, WRAPPERS[kind])(workspace(pr.device, mb=64), "NHD")
    kwargs = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    paged = (pr.kv_indptr, pr.kv_indices, pr.kv_last, pr.Hq, pr.Hkv, pr.D, pr.page)
    if kind == 0:
        plan_args = paged
    elif kind == 1:
        plan_args = (pr.qo_indptr, *paged)
    else:
        plan_args = (pr.qo_indptr, pr.kv_indptr_rag, pr.Hq, pr.Hkv, pr.D)
    if kind:
        kwargs["causal"] = True
    with pytest.raises(ValueError, match="has_sinks") as e:
        w.plan(*plan_args, **kwargs, has_sinks=declared)
    assert not isinstance(e.value, bi.BatchInvariantUnsupportedError)
    w.plan(*plan_args, **kwargs, has_sinks=None)


def _cpu_plan(engine, device="cpu", **kw):
    qo = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv = torch.tensor([0, 2, 5], dtype=torch.int32)
    last = torch.tensor([3, 16], dtype=torch.int32)
    planned = engine.plan_attention(
        qo, kv, last, 8, 2, 128, 16, -1, None, None, torch.device(device), **kw
    )
    return planned, kv


def test_engine_has_sinks_declaration():
    """``engine.plan_attention(has_sinks=)`` carries the declaration on the planned record and ``run_attention`` raises
    ``ValueError`` before any launch (device-free: the check precedes every kernel)."""
    from flashinfer.attention.batch_invariant import engine

    q = torch.zeros(2, 8, 128, dtype=torch.bfloat16)
    cache = torch.zeros(5, 16, 2, 128, dtype=torch.bfloat16)
    indices = torch.arange(5, dtype=torch.int32)
    sinks = torch.zeros(8)
    assert _cpu_plan(engine)[0].has_sinks is None
    for declared, given, text in (
        (True, None, "has_sinks=True"),
        (False, sinks, "has_sinks=False"),
    ):
        planned, kv = _cpu_plan(engine, has_sinks=declared)
        assert planned.has_sinks is declared
        with pytest.raises(ValueError, match=text):
            engine.run_attention(
                planned, q, cache, cache, kv, indices, 128**-0.5, sinks=given
            )


@pytest.mark.parametrize("op", WRAPPERS)
def test_has_sinks_is_a_trailing_keyword_of_plan(op):
    import inspect

    cls = getattr(flashinfer, op)
    plan = cls._plan_impl if hasattr(cls, "_plan_impl") else cls.plan
    last = list(inspect.signature(plan).parameters.values())[-1]
    assert last.name == "has_sinks" and last.default is None
    assert last.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_device)])
def test_unused_operand_placeholder_is_one_per_device(device, monkeypatch):
    """The operand a launch without LSE / sinks passes for them is one tensor per device: two plans share it and a
    ``plan()`` allocates none."""
    from flashinfer.attention.batch_invariant import engine

    first, _ = _cpu_plan(engine, device)
    shared = engine.unused_operand(device)
    assert first.lse_unused is shared
    assert shared.shape == (1,) and shared.dtype == torch.float32
    assert shared.device == first.desc.device
    cached = dict(engine._UNUSED_OPERAND)
    made = []
    real_empty = torch.empty

    def empty(*args, **kwargs):
        made.append(real_empty(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(torch, "empty", empty)
    second, _ = _cpu_plan(engine, device)
    monkeypatch.undo()
    assert second.lse_unused is shared
    assert all(t is not shared for t in made)
    # the plan's own one-element operands aside, no tensor like the placeholder was allocated and dropped
    kept = (second.ws, second.scratch)
    assert not any(
        (t.shape, t.dtype, t.device) == (shared.shape, shared.dtype, shared.device)
        for t in made
        if not any(t is k for k in kept)
    )
    assert set(engine._UNUSED_OPERAND) == set(cached)
    assert engine._UNUSED_OPERAND[engine._resolve_device(device)] is shared


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    pr = Problem([64], [700], 32, 8, 128, 16, seed=10)
    ws = workspace(pr.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a window belongs to the causal class: refused with the non-causal one
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            causal=False,
            window_left=8,
            **kw,
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            causal=True,
            logits_soft_cap=30.0,
            **kw,
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            192,
            16,
            causal=True,
            **kw,
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            causal=True,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            causal=True,
            pos_encoding_mode="ROPE_LLAMA",
            **kw,
        )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        32,
        8,
        128,
        16,
        causal=True,
        **kw,
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.run(pr.q, pr.cache("NHD"), k_scale=2.0)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.run(pr.q, pr.cache("NHD"), use_fp16_softmax=True)
    k, v = pr.cache("NHD")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a cache of another page size than the planned 16
        w.run(pr.q, (k[:, :8], v[:, :8]))
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="cute-dsl", batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", use_cuda_graph=True, batch_invariant=True
        )
    # plan() arguments outside the class: typed errors before any plan
    for bad in (
        dict(num_qo_heads=0, num_kv_heads=8),
        dict(sm_scale=float("inf")),
        dict(sm_scale=3e38),
        dict(seq_lens=torch.tensor([700], dtype=torch.int32, device=pr.device)),
        dict(prefix_len_ptr=torch.zeros(1, dtype=torch.int32, device=pr.device)),
    ):
        args = dict(num_qo_heads=32, num_kv_heads=8, causal=True, **kw)
        args.update(bad)
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            w.plan(
                pr.qo_indptr,
                pr.kv_indptr,
                pr.kv_indices,
                pr.kv_last,
                head_dim_qk=128,
                page_size=16,
                **args,
            )
    rag = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, "NHD", batch_invariant=True
    )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a request with a query token and no keys
        rag.plan(
            torch.tensor([0, 1], dtype=torch.int32, device=pr.device),
            torch.tensor([0, 0], dtype=torch.int32, device=pr.device),
            32,
            8,
            128,
            causal=True,
            **kw,
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        rag.plan(
            pr.qo_indptr,
            pr.kv_indptr_rag,
            32,
            8,
            128,
            causal=True,
            v_indptr=pr.kv_indptr_rag,
            **kw,
        )
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a window belongs to the causal class: refused with the non-causal one
        dec.plan(
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            q_len_per_req=2,
            is_causal=False,
            window_left=16,
            **kw,
        )
    # inputs the kernels would read out of bounds or mis-address: typed refusals at plan() / run()
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # qo_indptr not starting at 0
        w.plan(
            pr.qo_indptr + 8,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            causal=True,
            **kw,
        )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # kv_indices shorter than kv_indptr[-1]
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices[:-1],
            pr.kv_last,
            32,
            8,
            128,
            16,
            causal=True,
            **kw,
        )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a scale that rounds to zero in binary32
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            causal=True,
            sm_scale=1e-46,
            **kw,
        )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        32,
        8,
        128,
        16,
        causal=True,
        **kw,
    )
    k, v = pr.cache("NHD")
    top = int(pr.kv_indices.max())
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a cache without the largest planned page
        w.run(pr.q, (k[:top], v[:top]))
    wide = torch.empty(k.shape[:-1] + (2 * 128,), dtype=k.dtype, device=k.device)[
        ..., ::2
    ]
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a non-unit stride along the head dim
        w.run(pr.q, (wide, v))
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # out on another device
        w.run(pr.q, (k, v), out=torch.empty(pr.q.shape, dtype=torch.bfloat16))
    rag.plan(pr.qo_indptr, pr.kv_indptr_rag, 32, 8, 128, causal=True, **kw)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # fewer packed tokens than planned
        rag.run(pr.q, pr.k_rag[:-1], pr.v_rag[:-1])
    with pytest.raises(
        ValueError
    ):  # the functional API: fewer query rows than the plan
        from flashinfer.attention.batch_invariant import engine

        engine.run_attention(
            w._batch_invariant._planned,
            pr.q[:-1].contiguous(),
            k,
            v,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
        )
    assert dec.resolved_backend == "batch_invariant"
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # the deprecated forward()'s sm_scale cannot differ from the plan's
        w.forward(pr.q, (k, v), causal=True, sm_scale=0.5)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # the deprecated forward()'s causal=False
        w.forward(pr.q, (k, v), causal=False)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a positional argument after the cache
        w.run(pr.q, (k, v), 0.5)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # an output aliasing the cache
        w.run(pr.q, (k, v), out=k.flatten()[: pr.q.numel()].view(pr.q.shape))
    off = torch.empty(k.numel() + 1, dtype=k.dtype, device=k.device)[1:].view(k.shape)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a K view that is not 16-byte aligned
        w.run(pr.q, (off, v))
    qoff = torch.empty(pr.q.numel() + 1, dtype=pr.q.dtype, device=pr.q.device)[1:].view(
        pr.q.shape
    )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a q whose base is not 16-byte aligned
        w.run(qoff, (k, v))
    # the functional API refuses the same misuse with ValueError: misaligned operands, an output aliasing the cache
    from flashinfer.attention.batch_invariant import engine as eng

    planned = w._batch_invariant._planned
    padded = torch.empty(
        k.shape[:-1] + (k.shape[-1] + 4,), dtype=k.dtype, device=k.device
    )[..., : k.shape[-1]]
    for bad in (
        dict(q=qoff),
        dict(k_cache=off),
        dict(
            k_cache=padded
        ),  # unit head-dim stride, outer strides not multiples of 8 elements
        dict(out=k.flatten()[: pr.q.numel()].view(pr.q.shape)),
        dict(out=v.flatten()[: pr.q.numel()].view(pr.q.shape)),
    ):
        args = dict(q=pr.q, k_cache=k, v_cache=v, out=None)
        args.update(bad)
        with pytest.raises(ValueError):
            eng.run_attention(
                planned,
                args["q"],
                args["k_cache"],
                args["v_cache"],
                pr.kv_indptr,
                pr.kv_indices,
                1.0 / math.sqrt(128),
                out=args["out"],
            )
    # a plan with no query rows: nothing is launched, the (empty) output comes back
    w.plan(
        torch.tensor([0, 0], dtype=torch.int32, device=pr.device),
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        32,
        8,
        128,
        16,
        causal=True,
        **kw,
    )
    assert w.run(pr.q[:0], (k, v)).shape == (0, 32, 128)
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # a negative kv_indptr entry
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr - 1,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            causal=True,
            **kw,
        )
    # page-table metadata on the CPU is staged to the device (as the other wrappers accept)
    ref = bits(run_paged_prefill(pr))
    w.plan(
        pr.qo_indptr.cpu(),
        pr.kv_indptr.cpu(),
        pr.kv_indices.cpu(),
        pr.kv_last.cpu(),
        32,
        8,
        128,
        16,
        causal=True,
        **kw,
    )
    assert torch.equal(bits(w.run(pr.q, pr.cache("NHD"))), ref)
    # a workspace too small for the split-KV partials plans fused units instead (every chunk of a request in one CTA): the same bits
    # as the split plan of a large workspace (R3), and no partials at all; reset_workspace_buffer reaches the delegate, so the next
    # plan() in a large buffer splits again
    tiny = torch.empty(4096, dtype=torch.uint8, device=pr.device)
    big = Problem([1] * 4, [32768] * 4, 32, 8, 128, 16, seed=11)
    out_big, _ = run_decode(big)
    out_tiny, _ = run_decode(big, ws=tiny)
    assert torch.equal(bits(out_big), bits(out_tiny))
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        tiny, "NHD", batch_invariant=True
    )
    plan_kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    dec.plan(big.kv_indptr, big.kv_indices, big.kv_last, 32, 8, 128, 16, **plan_kw)
    assert dec._batch_invariant._config.split_policy == "fused"
    # workspace_size() reports the split plan's need: a buffer of exactly that size plans split units
    need, _ = dec.workspace_size(
        big.kv_indptr, big.kv_indices, big.kv_last, 32, 8, 128, 16, **plan_kw
    )
    assert need > tiny.numel()
    exact = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(need, dtype=torch.uint8, device=pr.device),
        "NHD",
        batch_invariant=True,
    )
    exact.plan(big.kv_indptr, big.kv_indices, big.kv_last, 32, 8, 128, 16, **plan_kw)
    assert exact._batch_invariant._config.split_policy == "auto"
    dec.reset_workspace_buffer(
        ws, torch.empty(8 << 20, dtype=torch.uint8, device=pr.device)
    )
    dec.plan(big.kv_indptr, big.kv_indices, big.kv_last, 32, 8, 128, 16, **plan_kw)
    assert dec._batch_invariant._config.split_policy == "auto"
    q_last = big.q[[int(big.qo_indptr[i + 1]) - 1 for i in range(4)]].contiguous()
    assert torch.equal(bits(dec.run(q_last, big.cache("NHD"))), bits(out_big))
    # Both prefill forward() APIs refuse fp16 QK reduction after a supported plan.
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        32,
        8,
        128,
        16,
        causal=True,
        **kw,
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.forward(pr.q, (k, v), causal=True, use_fp16_qk_reduction=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        rag.forward(pr.q, pr.k_rag, pr.v_rag, causal=True, use_fp16_qk_reduction=True)


@needs_device
@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory < (60 << 30),
    reason="needs tens of GB of device memory (the large problems and their pools)",
)
def test_output_beyond_2_31_elements():
    """An output of more than 2^31 elements: the kernels' output and workspace offsets are 64-bit. R2 gives the check: the
    last requests of the large batch, run alone, give the same bits (prefill: the row body; decode: the Swaps body)."""
    n, ql, Hq, Hkv, D = (
        4160,
        64,
        64,
        8,
        128,
    )  # 266,240 rows x 64 heads x 128 > 2^31 output elements
    pr = Problem([ql] * n, [ql] * n, Hq, Hkv, D, 16, seed=12)
    out = run_paged_prefill(pr)
    assert out.numel() > 2**31
    tail = 3
    sub = Problem([ql] * tail, [ql] * tail, Hq, Hkv, D, 16, seed=13)
    sub.q = pr.q[-tail * ql :].contiguous()
    sub.k_rag, sub.v_rag = pr.k_rag[-tail * ql :], pr.v_rag[-tail * ql :]
    rows = (-tail * ql, None)
    # the sub-batch's paged cache: the tail requests' pages of the large pool, in the large pool (same page indices)
    sub.k_nhd, sub.v_nhd = pr.k_nhd, pr.v_nhd
    sub.kv_indices = pr.kv_indices[int(pr.kv_indptr[-tail - 1]) :].contiguous()
    sub.kv_indptr = (pr.kv_indptr[-tail - 1 :] - pr.kv_indptr[-tail - 1]).contiguous()
    sub.kv_last = pr.kv_last[-tail:].contiguous()
    assert torch.equal(bits(run_paged_prefill(sub)), bits(out[rows[0] :]))
    del out
    n_dec = 262_200  # 262,200 rows x 64 heads x 128 > 2^31
    dec = Problem([1] * n_dec, [1] * n_dec, Hq, Hkv, D, 1, seed=14)
    out, _ = run_decode(dec)
    assert out.numel() > 2**31
    sub = Problem([1] * tail, [1] * tail, Hq, Hkv, D, 1, seed=15)
    sub.q = dec.q[-tail:].contiguous()
    sub.k_nhd, sub.v_nhd = dec.k_nhd, dec.v_nhd
    sub.kv_indices = dec.kv_indices[int(dec.kv_indptr[-tail - 1]) :].contiguous()
    sub.kv_indptr = (dec.kv_indptr[-tail - 1 :] - dec.kv_indptr[-tail - 1]).contiguous()
    sub.kv_last = dec.kv_last[-tail:].contiguous()
    small, _ = run_decode(sub)
    assert torch.equal(bits(small), bits(out[-tail:]))
    del out, dec
    # the split path past 2^31: many short requests and one long-KV request whose split units write partials to the workspace and
    # fold them in the fixup kernel or in-kernel; the long request alone gives the same bits (R2)
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine

    D = 256
    n_short, long_kv = 2100, 65536
    pr = Problem(
        [ql] * n_short + [ql], [ql] * n_short + [long_kv], Hq, Hkv, D, 16, seed=16
    )
    ws = workspace(pr.device, mb=768)
    alone = Problem([ql], [long_kv], Hq, Hkv, D, 16, seed=17)
    alone.q = pr.q[-ql:].contiguous()
    alone.k_nhd, alone.v_nhd = pr.k_nhd, pr.v_nhd
    alone.kv_indices = pr.kv_indices[int(pr.kv_indptr[-2]) :].contiguous()
    alone.kv_indptr = (pr.kv_indptr[-2:] - pr.kv_indptr[-2]).contiguous()
    alone.kv_last = pr.kv_last[-1:].contiguous()
    for merge in ("fixup", "inkernel"):
        cfg = replace(engine.default_config(D=D), split_policy="split", merge=merge)
        outs, plans = [], []
        for problem in (pr, alone):
            planned = engine.plan_attention(
                problem.qo_indptr,
                problem.kv_indptr,
                problem.kv_last,
                Hq,
                Hkv,
                D,
                16,
                -1,
                cfg,
                ws,
                pr.device,
            )
            plans.append(planned)
            outs.append(
                engine.run_attention(
                    planned,
                    problem.q,
                    problem.k_nhd,
                    problem.v_nhd,
                    problem.kv_indptr,
                    problem.kv_indices,
                    1.0 / math.sqrt(D),
                    -1,
                    "NHD",
                    None,
                    cfg,
                )
            )
        big_out, alone_out = outs
        assert big_out.numel() > 2**31
        assert plans[0].plan.ws_rows > 0, merge  # the large batch's plan splits
        assert torch.equal(bits(big_out[-ql:]), bits(alone_out)), merge
        del outs, big_out, alone_out


@needs_device
@pytest.mark.skipif(
    torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).total_memory < (60 << 30),
    reason="needs tens of GB of device memory (the large problems and their pools)",
)
def test_fixup_beyond_grid_y_limit():
    """More split groups than the hardware's grid.y extent (65,535): the fixup kernel strides over them; the same bits as the wrapper's
    own plan of the batch (R3)."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine

    n, D = (
        8200,
        64,
    )  # 8,200 requests x 8 KV heads = 65,600 two-chunk decode units, every one a split group folded by the fixup
    pr = Problem([1] * n, [520] * n, 32, 8, D, 16, seed=18)
    ref, _ = run_decode(pr)
    cfg = replace(engine.default_config(D=D), split_policy="split", merge="fixup")
    planned = engine.plan_attention(
        pr.qo_indptr, pr.kv_indptr, pr.kv_last, 32, 8, D, 16, -1, cfg, None, pr.device
    )
    assert len(planned.plan.fixup_groups) > 65535
    out = engine.run_attention(
        planned,
        pr.q,
        pr.k_nhd,
        pr.v_nhd,
        pr.kv_indptr,
        pr.kv_indices,
        1.0 / math.sqrt(D),
        -1,
        "NHD",
        None,
        cfg,
    )
    assert torch.equal(bits(out), bits(ref))


@needs_device
@pytest.mark.parametrize("D", [64, 128])
def test_hardware_row_max_equals_the_register_tree(D, monkeypatch):
    """The tile row maximum taken from the TMEM load (the parts that have the instruction) gives the bits of the register
    tree, on finite rows and on rows whose scores hold NaN, +-inf and signed zeros: the maximum is exact and order-free, so
    the placement reaches no value. Patching the gate selects the tree kernel for the comparison."""
    from flashinfer.attention.batch_invariant import engine

    dev = torch.device("cuda")
    if not engine.hw_row_max_enabled(dev):
        pytest.skip("this part has no hardware row maximum")
    pr = Problem([300, 1000, 64], [700, 2500, 3000], 32, 8, D, 16, seed=3)
    pr.q[5, 3, 7] = float("nan")
    pr.q[310, 1, 0] = float("inf")
    pr.q[311, 2, 1] = float("-inf")
    pr.q[1300] = 0.0
    hw = bits(run_paged_prefill(pr))
    monkeypatch.setattr(engine, "hw_row_max_enabled", lambda device=None, D=128: False)
    assert not engine.hw_row_max_enabled(dev)
    tree = bits(run_paged_prefill(pr))
    assert torch.equal(hw, tree)
    assert torch.isnan(tree.view(torch.bfloat16).float()).any()


@needs_device
def test_hardware_row_max_is_a_head_dim_placement():
    """The hardware row maximum is taken at D <= 128 only (the two-stage TMEM plan at D = 256 keeps the register tree), on the
    parts that have the instruction; the D = 256 kernels are the tree kernels on every part, and a non-CUDA device never takes it."""
    from flashinfer.attention.batch_invariant import engine

    dev = torch.device("cuda")
    assert not engine.hw_row_max_enabled(dev, 256)
    assert (
        engine.hw_row_max_enabled(dev, 128)
        == engine.hw_row_max_enabled(dev, 64)
        == engine.hw_row_max_enabled(dev)
    )
    assert engine.hw_row_max_enabled(torch.device("cpu"), 64) is False


# ------------------------------------------------------------------------------------------------------ the non-causal class
def reference_noncausal(pr: Problem):
    """fp32 non-causal attention per request (every token sees every key of its request), the yardstick for accuracy."""
    outs = []
    tok_q = tok_kv = 0
    g = pr.Hq // pr.Hkv
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        q = pr.q[tok_q : tok_q + ql].float()
        k = pr.k_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        v = pr.v_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(pr.D)
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
        tok_q += ql
        tok_kv += kl
    return torch.cat(outs)


def run_decode_noncausal(pr: Problem, q_len: int, layout="NHD", ws=None, **run_kw):
    """The decode wrapper with ``q_len_per_req = q_len`` and ``is_causal=False`` on the LAST ``q_len`` tokens of every request."""
    assert all(ql >= q_len for ql in pr.q_lens)
    rows = torch.cat(
        [
            torch.arange(
                int(pr.qo_indptr[i + 1]) - q_len,
                int(pr.qo_indptr[i + 1]),
                device=pr.device,
            )
            for i in range(len(pr.q_lens))
        ]
    )
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        ws if ws is not None else workspace(pr.device), layout, batch_invariant=True
    )
    w.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_len_per_req=q_len,
        is_causal=False,
    )
    return w.run(pr.q[rows].contiguous(), pr.cache(layout), **run_kw), rows


@pytest.mark.cpu_only
@pytest.mark.parametrize("op", WRAPPERS)
def test_noncausal_fingerprint_is_a_function_of_head_dim_and_mode(op):
    """The mask mode is a class dimension: the non-causal fingerprint differs from the causal one at every head dim and each is
    shared by the three wrappers."""
    for D in (64, 128, 256):
        nc = bi.fingerprint(op, head_dim=D, causal=False)
        c = bi.fingerprint(op, head_dim=D, causal=True)
        assert nc != c and nc.startswith(f"attn-bf16-d{D}-nc-")
        assert c == bi.fingerprint(WRAPPERS[0], head_dim=D, causal=True)
        assert nc == bi.fingerprint(WRAPPERS[0], head_dim=D, causal=False)
    assert (
        len({bi.fingerprint(op, head_dim=D, causal=False) for D in (64, 128, 256)}) == 3
    )
    assert {
        D: bi.fingerprint(op, head_dim=D, causal=False) for D in (64, 128, 256)
    } == {
        64: "attn-bf16-d64-nc-5c64fc377eb66759",
        128: "attn-bf16-d128-nc-421e018610a7f93c",
        256: "attn-bf16-d256-nc-603c24eeb8691c0a",
    }
    # the pinned causal fingerprints are unchanged by the new class
    from flashinfer.attention.batch_invariant.spec import FINGERPRINTS

    for D, fp in FINGERPRINTS.items():
        assert bi.fingerprint(op, head_dim=D, causal=True) == fp


@pytest.mark.cpu_only
@pytest.mark.parametrize("op", WRAPPERS[1:])
def test_prefill_fingerprint_needs_the_mask_mode(op):
    """The prefill wrappers' ``plan()`` defaults to ``causal=False``, the non-causal class: their receipt takes the mode as a
    fixed dimension, so a receipt asked without it (or with ``None``) raises instead of naming the causal class."""
    from flashinfer.attention.batch_invariant.spec import FINGERPRINTS

    for dims in (dict(head_dim=128), dict(head_dim=128, causal=None)):
        with pytest.raises(TypeError, match="causal") as e:
            bi.fingerprint(op, **dims)
        assert "causal=True" in str(e.value) and "causal=False" in str(e.value)
    with pytest.raises(TypeError, match="head_dim"):
        bi.fingerprint(op, causal=True)
    for D, fp in FINGERPRINTS.items():
        assert bi.fingerprint(op, head_dim=D, causal=True) == fp
        assert bi.fingerprint(op, head_dim=D, causal=False) != fp


@pytest.mark.cpu_only
def test_decode_fingerprint_defaults_to_the_causal_class():
    """The decode wrapper's mode is optional: omitted or ``None`` is the causal class (its ``is_causal=None``), never the
    truth value of ``None``."""
    from flashinfer.attention.batch_invariant.spec import FINGERPRINTS

    op = WRAPPERS[0]
    for D, fp in FINGERPRINTS.items():
        assert bi.fingerprint(op, head_dim=D) == fp
        assert bi.fingerprint(op, head_dim=D, causal=None) == fp
        assert bi.fingerprint(op, head_dim=D, causal=True) == fp
        nc = bi.fingerprint(op, head_dim=D, causal=False)
        assert nc != fp and nc == bi.fingerprint(WRAPPERS[1], head_dim=D, causal=False)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_noncausal_matches_reference(D, Hq, Hkv):
    """The non-causal class against fp32 non-causal attention (every token sees every key of its request), paged and ragged; a
    request with more query tokens than keys is a member of the class."""
    pr = Problem([300, 1000, 64, 40], [700, 2500, 3000, 17], Hq, Hkv, D, 16, seed=21)
    ref = reference_noncausal(pr)
    tol = 2e-2 + 1e-2 * ref.abs().max().item()
    out = run_paged_prefill(pr, causal=False)
    assert (out.float() - ref).abs().max().item() < tol
    out = run_ragged_prefill(pr, causal=False)
    assert (out.float() - ref).abs().max().item() < tol


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("page", [1, 16, 128])
def test_noncausal_paged_equals_ragged(D, page):
    """R4: the paged and the ragged KV layouts hand the kernel the same keys: the same bits under the non-causal class."""
    pr = Problem(
        [300, 1000, 64, 2048], [700, 2500, 3000, 2048], 32, 8, D, page, seed=22
    )
    assert torch.equal(
        bits(run_paged_prefill(pr, causal=False)),
        bits(run_ragged_prefill(pr, causal=False)),
    )
    assert torch.equal(
        bits(run_paged_prefill(pr, "HND", causal=False)),
        bits(run_ragged_prefill(pr, "HND", causal=False)),
    )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_noncausal_query_chunks_equal_full(D):
    """Chunked prefill under the non-causal class: a request's query rows run in chunks, each chunk planned over the request's
    FULL key set, give the bits of the one-shot run row for row (a row's visible set is ``[0, kv_len)`` whatever the cut), paged and
    ragged. Two requests cut at different points (inside a 512-key chunk and at a tile boundary); incremental KV growth is not
    this test -- a chunk run over a shorter key set is that chunk's view of those keys, not a prefix of the full run."""
    kv = [1700, 1000]
    cut = [700, 512]
    full = Problem(kv, kv, 32, 8, D, 16, seed=15)
    out_full = bits(run_paged_prefill(full, causal=False))
    assert torch.equal(out_full, bits(run_ragged_prefill(full, causal=False)))
    rows = [slice(int(full.qo_indptr[i]), int(full.qo_indptr[i + 1])) for i in range(2)]
    for part, q_lens in enumerate(([cut[0], cut[1]], [kv[0] - cut[0], kv[1] - cut[1]])):
        prob = Problem(q_lens, kv, 32, 8, D, 16, seed=16 + part)
        prob.q = torch.cat(
            [
                full.q[rows[i]][: cut[i]] if part == 0 else full.q[rows[i]][cut[i] :]
                for i in range(2)
            ]
        ).contiguous()
        prob.k_rag, prob.v_rag = full.k_rag.clone(), full.v_rag.clone()
        prob = _repack(prob)
        out_p = bits(run_paged_prefill(prob, causal=False))
        out_r = bits(run_ragged_prefill(prob, causal=False))
        off = 0
        for i in range(2):
            ref = (
                out_full[rows[i]][: cut[i]]
                if part == 0
                else out_full[rows[i]][cut[i] :]
            )
            n = ref.shape[0]
            assert torch.equal(out_p[off : off + n], ref), (D, part, i, "paged")
            assert torch.equal(out_r[off : off + n], ref), (D, part, i, "ragged")
            off += n


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_noncausal_prefill_equals_multi_token_decode(D, layout):
    """R4: the decode wrapper with ``q_len_per_req = 4`` and ``is_causal=False`` gives, for the last four tokens of every request,
    the bits of the non-causal prefill rows; and every token of a request has the bits of its position-independent class term
    (the first four tokens' rows equal the last four's when their queries are the same)."""
    q_len = 4
    pr = Problem([256, 33, 1024, 7], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=23)
    pre = bits(run_paged_prefill(pr, layout, causal=False))
    dec, rows = run_decode_noncausal(pr, q_len, layout)
    assert torch.equal(bits(dec), pre[rows])
    # the position does not enter the non-causal term: the same query at another position of the same request gives the same bits
    pr2 = Problem(pr.q_lens, pr.kv_lens, 32, 8, D, 16, seed=23)
    for i in range(len(pr.q_lens)):
        a, b = int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])
        pr2.q[a:b] = pr.q[a:b].flip(0)
    pre2 = bits(run_paged_prefill(pr2, layout, causal=False))
    for i in range(len(pr.q_lens)):
        a, b = int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])
        assert torch.equal(pre2[a:b], pre[a:b].flip(0)), i


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_noncausal_batch_permutation_and_subbatch_invariance(D):
    """R2 under the non-causal class: a request's rows are the same bits in a batch of one, in a permuted batch and in a sub-batch;
    paged prefill, ragged prefill and multi-token decode."""
    q_lens, kv_lens = [300, 1, 1000, 64, 4], [700, 2500, 2500, 3000, 4]
    pr = Problem(q_lens, kv_lens, 32, 8, D, 16, seed=24)
    full_p = bits(run_paged_prefill(pr, causal=False))
    full_r = bits(run_ragged_prefill(pr, causal=False))
    assert torch.equal(full_p, full_r)
    for sel in ([2], [4, 0], [3, 1, 2], [1, 0, 4, 3, 2]):
        sub = Problem(
            [q_lens[i] for i in sel], [kv_lens[i] for i in sel], 32, 8, D, 16, seed=25
        )
        for j, i in enumerate(sel):
            qa, qb = int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])
            sa, sb = int(sub.qo_indptr[j]), int(sub.qo_indptr[j + 1])
            sub.q[sa:sb] = pr.q[qa:qb]
            ka, kb = int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1])
            ta, tb = int(sub.kv_indptr_rag[j]), int(sub.kv_indptr_rag[j + 1])
            sub.k_rag[ta:tb] = pr.k_rag[ka:kb]
            sub.v_rag[ta:tb] = pr.v_rag[ka:kb]
        sub = _repack(sub)
        got_p = bits(run_paged_prefill(sub, causal=False))
        got_r = bits(run_ragged_prefill(sub, causal=False))
        for j, i in enumerate(sel):
            qa, qb = int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])
            sa, sb = int(sub.qo_indptr[j]), int(sub.qo_indptr[j + 1])
            assert torch.equal(got_p[sa:sb], full_p[qa:qb]), (D, sel, i, "paged")
            assert torch.equal(got_r[sa:sb], full_r[qa:qb]), (D, sel, i, "ragged")
    # multi-token decode of the whole batch against decode of a sub-batch (every request has at least 4 tokens)
    pr4 = Problem(
        [300, 8, 1000, 64, 4], [700, 2500, 2500, 3000, 4], 32, 8, D, 16, seed=26
    )
    dec, rows = run_decode_noncausal(pr4, 4)
    dec = bits(dec)
    sub = Problem([8, 4], [2500, 4], 32, 8, D, 16, seed=27)
    for j, i in enumerate((1, 4)):
        qa, qb = int(pr4.qo_indptr[i]), int(pr4.qo_indptr[i + 1])
        sa, sb = int(sub.qo_indptr[j]), int(sub.qo_indptr[j + 1])
        sub.q[sa:sb] = pr4.q[qa:qb]
        ka, kb = int(pr4.kv_indptr_rag[i]), int(pr4.kv_indptr_rag[i + 1])
        ta, tb = int(sub.kv_indptr_rag[j]), int(sub.kv_indptr_rag[j + 1])
        sub.k_rag[ta:tb] = pr4.k_rag[ka:kb]
        sub.v_rag[ta:tb] = pr4.v_rag[ka:kb]
    sub = _repack(sub)
    dsub, _ = run_decode_noncausal(sub, 4)
    assert torch.equal(bits(dsub), torch.cat([dec[4:8], dec[16:20]]))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_noncausal_config_agreement(D):
    """R3 under the non-causal class: every member of the configuration space gives the bits of the wrapper's pick (the whole space
    at D = 128 / 256, every 7th point at D = 64), and so does the paired placement: at D = 64, and at D = 128 on 328 KB parts
    (pairing forced with an SM count of one; its rings, the split kind with both fold placements). Under this class the Q tiles
    of a request share one chunk and tile range, so every pair's two tiles walk the same keys."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    if D == 64:
        pr = Problem([2048, 2048, 1], [4096, 3000, 1000], 32, 8, D, 16, seed=28)
    else:
        pr = Problem([256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, seed=28)
    ref = bits(run_paged_prefill(pr, causal=False))
    space = engine.all_valid_configs(D)
    cfgs = list(space[::7] if D == 64 else space)
    q2_sm = (
        None  # the SM count the paired configurations plan with (None: the device's)
    )
    if D == 64:
        two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
        rings = (2, 3) + ((4,) if arch_smem_cap(pr.device) >= (300 << 10) else ())
        cfgs += [replace(two, kv_slots=s) for s in rings]
    elif D == 128 and arch_smem_cap(pr.device) >= (300 << 10):
        two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
        q2_sm = 1
        cfgs += [
            replace(two, kv_slots=s, split_policy=p, merge=m)
            for s in (2, 3)
            for p, m in (("auto", "auto"), ("split", "auto"), ("split", "inkernel"))
        ]
    for cfg in cfgs:
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            32,
            8,
            D,
            16,
            -1,
            cfg,
            None,
            pr.device,
            sm_count=q2_sm if cfg.q2 else None,
            causal=False,
        )
        if cfg.q2:
            assert planned.plan.num_launch is not None, (cfg, planned.plan.num_units)
        out = engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
        )
        assert torch.equal(bits(out), ref), cfg


@needs_device
def test_noncausal_refusals_and_single_token_decode():
    """A window is not a parameter of the non-causal class (prefill and decode); a request with query tokens and no key has no
    visible key and is refused; forward() must agree with the planned mask mode and a rejection preserves the plan; a single-token
    decode step is the causal class whatever ``is_causal`` says (the same bits)."""
    pr = Problem([64, 3], [700, 90], 32, 8, 128, 16, seed=29)
    ws = workspace(pr.device)
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.plan(*args, causal=False, window_left=32, **kw)
    rag = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, "NHD", batch_invariant=True
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        rag.plan(
            pr.qo_indptr,
            pr.kv_indptr_rag,
            32,
            8,
            128,
            causal=False,
            window_left=0,
            **kw,
        )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a request with query tokens and no key: a row without a visible key
        rag.plan(
            pr.qo_indptr,
            torch.tensor([0, 700, 700], dtype=torch.int32, device=pr.device),
            32,
            8,
            128,
            causal=False,
            **kw,
        )
    # forward() must agree with the planned class; a rejection leaves the plan in place
    w.plan(*args, causal=False, **kw)
    expected = bits(w.run(pr.q, pr.cache("NHD"))).clone()
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.forward(pr.q, pr.cache("NHD"), causal=True)
    assert torch.equal(bits(w.run(pr.q, pr.cache("NHD"))), expected)
    assert torch.equal(bits(w.forward(pr.q, pr.cache("NHD"), causal=False)), expected)
    assert torch.equal(bits(w.forward(pr.q, pr.cache("NHD"))), expected)
    assert not torch.equal(bits(run_paged_prefill(pr)), expected)
    # the single-token decode step: is_causal=False and True (and the default) are one class
    last = torch.tensor(
        [int(pr.qo_indptr[i + 1]) - 1 for i in range(2)], device=pr.device
    )
    q1 = pr.q[last].contiguous()
    outs = []
    for is_causal in (None, True, False):
        dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        dec.plan(
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            is_causal=is_causal,
            **kw,
        )
        outs.append(bits(dec.run(q1, pr.cache("NHD"))).clone())
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    assert torch.equal(outs[0], bits(run_paged_prefill(pr))[last])
    # which is also the non-causal class's last row (the last token sees every key in both classes)
    assert torch.equal(outs[0], expected[last])


# ------------------------------------------------------------------------------------------------------ workspace sizing
def _host_delegate(kind):
    """The wrappers' delegate without a device: only its host-side sizing (``workspace_bytes``) is used, which reads the kind and
    the device (the host planner's default SM count on the CPU)."""
    from flashinfer.attention.batch_invariant.wrappers import BatchInvariantAttention

    d = object.__new__(BatchInvariantAttention)
    d.kind = kind
    d.device = torch.device("cpu")
    return d


def _i32(values):
    return torch.tensor(values, dtype=torch.int32)


@pytest.mark.cpu_only
def test_workspace_sizing_raises_what_plan_refuses():
    """The sizing call runs the request and mask-mode checks of ``plan()``: every input the plan refuses is the class's typed error
    here too, never an assertion of the host planner."""
    paged = _host_delegate("prefill")
    one = (
        _i32([0, 100]),
        _i32([0, 2]),
        _i32([16]),
        8,
        8,
        128,
        16,
    )  # 100 tokens, 32 keys
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="window_left 5"):
        paged.workspace_bytes(*one, window_left=5, causal=False)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="100 query tokens"):
        paged.workspace_bytes(*one, causal=True)
    assert (
        paged.workspace_bytes(*one, causal=False) >= 0
    )  # q_len > kv_len is a member of the non-causal class
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="over no key"):
        paged.workspace_bytes(
            _i32([0, 5]), _i32([0, 0]), _i32([0]), 8, 8, 128, 16, causal=False
        )
    ok = (_i32([0, 5]), _i32([0, 1]), _i32([8]))
    for heads_dim_page in ((7, 3, 128, 16), (8, 8, 128, 3), (8, 8, 96, 16)):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            paged.workspace_bytes(*ok, *heads_dim_page)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # one last-page length per request
        paged.workspace_bytes(_i32([0, 5]), _i32([0, 1]), _i32([8, 8]), 8, 8, 128, 16)
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # one KV range per request
        paged.workspace_bytes(_i32([0, 5, 9]), _i32([0, 1]), _i32([8]), 8, 8, 128, 16)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        paged.workspace_bytes(None, _i32([0, 1]), _i32([8]), 8, 8, 128, 16)
    ragged = _host_delegate("ragged")
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="window_left 0"):
        ragged.workspace_bytes(
            _i32([0, 4]), _i32([0, 9]), None, 8, 8, 128, 1, window_left=0, causal=False
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="over no key"):
        ragged.workspace_bytes(
            _i32([0, 4]), _i32([0, 0]), None, 8, 8, 128, 1, causal=False
        )
    # multi-token decode over a request with one key: a member of the non-causal class, outside the causal one
    decode = _host_delegate("decode")
    one_key = (None, _i32([0, 1]), _i32([1]), 16, 8, 128, 16)
    assert decode.workspace_bytes(*one_key, q_len_per_req=4, causal=False) == 0
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="4 query tokens"):
        decode.workspace_bytes(*one_key, q_len_per_req=4, causal=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        decode.workspace_bytes(*one_key, q_len_per_req=0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("D", [64, 128, 256])
def test_workspace_size_is_a_function_of_the_mask_mode(D):
    """The two classes plan different units for the same batch: under the causal class the tokens of the first chunk see one chunk
    (fused units, no partials), under the non-causal class every token sees every chunk of its request, so the split plan of the
    same request holds more partial rows. The sizing follows the mode it is given."""
    from flashinfer.attention.batch_invariant.plan import ws_row
    from flashinfer.attention.batch_invariant.spec import chunk_len

    L = 2 * chunk_len(D)
    # two heads: few enough wide units that the planner's rule splits them at every supported SM count
    request = (_i32([0, L]), _i32([0, L // 16]), _i32([16]), 2, 2, D, 16)
    paged = _host_delegate("prefill")
    causal = paged.workspace_bytes(*request, causal=True)
    noncausal = paged.workspace_bytes(*request, causal=False)
    assert 0 < causal < noncausal
    assert causal % (ws_row(D) * 4) == 0 and noncausal % (ws_row(D) * 4) == 0
    ragged = _host_delegate("ragged")
    packed = (_i32([0, L]), _i32([0, L]), None, 2, 2, D, 1)
    assert ragged.workspace_bytes(*packed, causal=True) == causal
    assert ragged.workspace_bytes(*packed, causal=False) == noncausal


def _exact_workspace(need, device):
    return torch.empty(need, dtype=torch.uint8, device=device)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_prefill_workspace_size_plans_both_classes_without_fallback(D):
    """``workspace_size`` with the mask mode of the plan, then a buffer of exactly that many bytes: the plan of either class keeps its
    split units (no fused fallback) and gives the bits of a large workspace; one byte row less falls back to fused units, the same
    bits. The two classes need different sizes for this request."""
    from flashinfer.attention.batch_invariant.spec import chunk_len

    L = 2 * chunk_len(D)
    # two heads: few enough wide units that the planner's rule splits them at every supported SM count
    pr = Problem([L], [L], 2, 2, D, 16, seed=41)
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 2, 2, D, 16)
    sizer = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device, mb=8), "NHD", batch_invariant=True
    )
    needs = {}
    for causal in (True, False):
        need, int_need = sizer.workspace_size(*args, causal=causal, **kw)
        assert need > 0 and int_need == 0
        needs[causal] = need
        ref = bits(run_paged_prefill(pr, causal=causal))
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _exact_workspace(need, pr.device), "NHD", batch_invariant=True
        )
        w.plan(*args, causal=causal, **kw)
        delegate = w._batch_invariant
        assert delegate._config.split_policy == "auto"
        assert delegate._planned.plan.ws_rows > 0 and delegate._planned.causal == causal
        assert torch.equal(bits(w.run(pr.q, pr.cache("NHD"))), ref)
        small = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _exact_workspace(need - 16, pr.device), "NHD", batch_invariant=True
        )
        small.plan(*args, causal=causal, **kw)
        assert small._batch_invariant._config.split_policy == "fused"
        assert torch.equal(bits(small.run(pr.q, pr.cache("NHD"))), ref)
    assert needs[True] < needs[False]
    # the causal size does not hold the non-causal plan of the same request: sized with the wrong mode, the plan falls back
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _exact_workspace(needs[True], pr.device), "NHD", batch_invariant=True
    )
    w.plan(*args, causal=False, **kw)
    assert w._batch_invariant._config.split_policy == "fused"


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_decode_workspace_size_plans_both_classes_without_fallback(D):
    """The decode wrapper's ``workspace_size`` takes ``q_len_per_req`` and ``is_causal`` as ``plan()`` does: a buffer of exactly the
    reported size plans the split units of the causal (``is_causal=None`` / ``True``) and of the non-causal (``is_causal=False``)
    multi-token decode, with the bits of a large workspace."""
    q_len = 4
    pr = Problem([q_len] * 4, [8192] * 4, 32, 8, D, 16, seed=42)
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    args = (pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, D, 16)
    sizer = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device, mb=8), "NHD", batch_invariant=True
    )
    for is_causal in (None, True, False):
        causal = is_causal is not False
        mode = dict(q_len_per_req=q_len, is_causal=is_causal)
        need, int_need = sizer.workspace_size(*args, **mode, **kw)
        assert need > 0 and int_need == 0
        big = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace(pr.device), "NHD", batch_invariant=True
        )
        big.plan(*args, **mode, **kw)
        ref = bits(big.run(pr.q, pr.cache("NHD"))).clone()
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _exact_workspace(need, pr.device), "NHD", batch_invariant=True
        )
        w.plan(*args, **mode, **kw)
        delegate = w._batch_invariant
        assert delegate._config.split_policy == "auto"
        assert delegate._planned.plan.ws_rows > 0 and delegate._planned.causal == causal
        assert torch.equal(bits(w.run(pr.q, pr.cache("NHD"))), ref)
        prefill = bits(run_paged_prefill(pr, causal=causal))
        assert torch.equal(
            ref, prefill
        )  # every request's four tokens are its whole query block


@needs_device
def test_workspace_size_raises_the_typed_errors_of_plan():
    """Through the wrappers: the sizing call refuses what ``plan()`` refuses with the class's typed error. The paged prefill
    wrapper's ``causal`` defaults to ``False`` in both calls, so a windowed (causal) batch sized without ``causal=True`` is refused
    like the plan it would size; the decode wrapper sizes a non-causal multi-token decode over a one-key request and refuses the
    causal one."""
    pr = Problem([64, 3], [700, 90], 32, 8, 128, 16, seed=43)
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device, mb=8), "NHD", batch_invariant=True
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.workspace_size(*args, window_left=32, **kw)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.plan(*args, window_left=32, **kw)
    float_bytes, int_bytes = w.workspace_size(*args, causal=True, window_left=32, **kw)
    assert float_bytes >= 0 and int_bytes == 0
    more_queries = Problem([40], [17], 32, 8, 128, 16, seed=44)
    margs = (
        more_queries.qo_indptr,
        more_queries.kv_indptr,
        more_queries.kv_indices,
        more_queries.kv_last,
        32,
        8,
        128,
        16,
    )
    assert w.workspace_size(*margs, causal=False, **kw) == (0, 0)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.workspace_size(*margs, causal=True, **kw)
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # query tokens over no key
        w.workspace_size(
            more_queries.qo_indptr,
            torch.zeros(2, dtype=torch.int32, device=pr.device),
            more_queries.kv_indices,
            torch.zeros(1, dtype=torch.int32, device=pr.device),
            32,
            8,
            128,
            16,
            causal=False,
            **kw,
        )
    one_key = Problem([4], [1], 16, 8, 128, 16, seed=45)
    dargs = (one_key.kv_indptr, one_key.kv_indices, one_key.kv_last, 16, 8, 128, 16)
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device, mb=8), "NHD", batch_invariant=True
    )
    assert dec.workspace_size(*dargs, q_len_per_req=4, is_causal=False, **kw) == (0, 0)
    for is_causal in (None, True):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            dec.workspace_size(*dargs, q_len_per_req=4, is_causal=is_causal, **kw)
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            dec.plan(*dargs, q_len_per_req=4, is_causal=is_causal, **kw)
    dec.plan(*dargs, q_len_per_req=4, is_causal=False, **kw)
    out = dec.run(one_key.q, one_key.cache("NHD"))
    ref = reference_noncausal(one_key)
    assert (out.float() - ref).abs().max().item() < 2e-2 + 1e-2 * ref.abs().max().item()
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a window is not a parameter of the non-causal class
        dec.workspace_size(
            *dargs, q_len_per_req=4, is_causal=False, window_left=8, **kw
        )


# ------------------------------------------------------------------------------------------------------ non-causal placements
@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_noncausal_multi_token_decode_on_the_swaps_body(D):
    """Non-causal multi-token decode whose units all have at most ``SWAPS_MAX_ROWS`` live rows (``g x q_len = 2 x 4``) runs the
    Swaps-AB kernel of the non-causal class: the bits of the non-causal prefill rows (the row body) for the same tokens, of the
    row body's replicated placement of the same decode batch, and the fp32 reference within the class's tolerance. Requests of
    one and of several chunks, and one with fewer keys than query tokens."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine

    assert D in engine.SWAPS_DIMS
    q_len, Hq, Hkv = 4, 16, 8
    assert (Hq // Hkv) * q_len <= engine.SWAPS_MAX_ROWS
    pr = Problem(
        [256, 33, 1024, 7, 4], [1500, 2500, 1024, 4096, 3], Hq, Hkv, D, 16, seed=51
    )
    rows = torch.cat(
        [
            torch.arange(
                int(pr.qo_indptr[i + 1]) - q_len,
                int(pr.qo_indptr[i + 1]),
                device=pr.device,
            )
            for i in range(len(pr.q_lens))
        ]
    )
    q = pr.q[rows].contiguous()
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        Hq,
        Hkv,
        D,
        16,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_len_per_req=q_len,
        is_causal=False,
    )
    delegate = w._batch_invariant
    plan = delegate._planned.plan
    assert delegate._planned.causal is False
    assert delegate._config.small == "swap"
    assert max(d[2] for d in plan.desc) <= engine.SWAPS_MAX_ROWS
    assert engine.swaps_n_pad(plan) > 0  # the plan runs the Swaps-AB kernel
    assert plan.groups  # the multi-chunk requests are split
    dec = w.run(q, pr.cache("NHD"))
    pre = run_paged_prefill(pr, causal=False)
    assert torch.equal(bits(dec), bits(pre)[rows])
    ref = reference_noncausal(pr)[rows]
    err = (dec.float() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err
    # the same decode batch through the engine on the row body (small = "rep": the replicated-row placement of small units)
    # and on the Swaps body again: the same bits
    qo = torch.arange(len(pr.q_lens) + 1, dtype=torch.int32, device=pr.device) * q_len
    for small in ("rep", "swap"):
        cfg = replace(engine.default_config(D=D), small=small)
        planned = engine.plan_attention(
            qo,
            pr.kv_indptr,
            pr.kv_last,
            Hq,
            Hkv,
            D,
            16,
            -1,
            cfg,
            None,
            pr.device,
            causal=False,
        )
        out = engine.run_attention(
            planned,
            q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
        )
        assert torch.equal(bits(out), bits(dec)), small


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_noncausal_more_queries_than_keys_split_under_both_merges(D):
    """Non-causal requests of several chunks with more query tokens than keys (and one short query block over more chunks), with
    every multi-chunk unit split and the chunk fold placed in the fixup kernel, in-kernel, or by the planner's rule, and with
    fused units: the bits of the wrapper's own plan, and the fp32 reference within the class's tolerance."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.spec import chunk_len

    C = chunk_len(D)
    q_lens = [1500, 700, 20]
    kv_lens = [2 * C + 37, C + 150, 3 * C + 5]
    assert q_lens[0] > kv_lens[0] > 2 * C and q_lens[1] > kv_lens[1] > C
    Hq = Hkv = 8
    pr = Problem(q_lens, kv_lens, Hq, Hkv, D, 16, seed=52)
    out = run_paged_prefill(pr, causal=False)
    ref = reference_noncausal(pr)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err
    assert torch.equal(bits(out), bits(run_ragged_prefill(pr, causal=False)))
    for policy, merge in (
        ("split", "fixup"),
        ("split", "inkernel"),
        ("split", "auto"),
        ("fused", "auto"),
    ):
        cfg = replace(engine.default_config(D=D), split_policy=policy, merge=merge)
        assert cfg.well_formed(D)
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            Hq,
            Hkv,
            D,
            16,
            -1,
            cfg,
            None,
            pr.device,
            causal=False,
        )
        plan = planned.plan
        if policy == "fused":
            assert not plan.groups and plan.ws_rows == 0
        else:
            # every (Q tile, KV head) of the three requests is a multi-chunk unit: one split group each
            tiles = sum(-(-ql // plan.tokens_per_tile) for ql in q_lens) * Hkv
            assert len(plan.groups) == tiles and plan.ws_rows > 0
            assert (
                max(gr[3] for gr in plan.groups) == 4
            )  # the short block's four chunks
            assert (
                min(gr[1] for gr in plan.groups)
                <= 32
                < max(gr[1] for gr in plan.groups)
            )
            if merge == "fixup":
                assert len(plan.fixup_groups) == len(plan.groups)
            elif merge == "inkernel":
                assert all(plan.group_inkernel) and not plan.fixup_groups
        got = engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
        )
        assert torch.equal(bits(got), bits(out)), (policy, merge)


# ------------------------------------------------------------------------------------------------------ the non-causal members
NC_SINKS = pytest.mark.parametrize("with_sinks", [False, True], ids=["plain", "sinks"])


def reference_noncausal_members(pr: Problem, sinks=None):
    """fp32 non-causal attention per request with the natural-log LSE of the scaled scores, ``(total_q, Hq)``; with ``sinks``
    every row's softmax has the head's sink logit as one more key of zero value."""
    outs, lses = [], []
    tok_q = tok_kv = 0
    g = pr.Hq // pr.Hkv
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        q = pr.q[tok_q : tok_q + ql].float()
        k = pr.k_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        v = pr.v_rag[tok_kv : tok_kv + kl].float().repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(pr.D)
        if sinks is not None:
            s = torch.cat([s, sinks.float()[:, None, None].expand(-1, ql, 1)], dim=-1)
        p = torch.softmax(s, dim=-1)[..., :kl]
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
        lses.append(torch.logsumexp(s, dim=-1).t())
        tok_q += ql
        tok_kv += kl
    return torch.cat(outs), torch.cat(lses)


def _nc_run_kw(pr, with_sinks, seed=0, **kw):
    """Run-time arguments of a non-causal call with the LSE and, when asked, sinks; the sinks next to them."""
    sinks = sink_logits(pr, seed=seed) if with_sinks else None
    kw = dict(LSE, **kw)
    if with_sinks:
        kw["sinks"] = sinks
    return kw, sinks


def _nc_ragged(pr, kw, layout="NHD"):
    """The ragged wrapper under the non-causal class with the run-time arguments ``kw`` (its sinks go to ``_sinks``)."""
    kw = dict(kw)
    sinks = kw.pop("sinks", None)
    return run_ragged_prefill(pr, layout, causal=False, run_kw=kw, sinks=sinks)


@pytest.mark.cpu_only
@pytest.mark.parametrize("op", WRAPPERS)
def test_noncausal_member_fingerprints_are_pinned(op):
    """The LSE and sink members of the non-causal class have receipts of their own, the mode before the member in the
    name; the receipts of the causal class and of its members do not move."""
    from flashinfer.attention.batch_invariant import spec

    nc = {
        64: "attn-bf16-d64-nc-5c64fc377eb66759",
        128: "attn-bf16-d128-nc-421e018610a7f93c",
        256: "attn-bf16-d256-nc-603c24eeb8691c0a",
    }
    nc_lse = {
        (64, "log2"): "attn-bf16-d64-nc-lse-log2-f287f9e1fd18384f",
        (64, "ln"): "attn-bf16-d64-nc-lse-ln-f1552997d639d2ce",
        (128, "log2"): "attn-bf16-d128-nc-lse-log2-66ad5536ca59cb76",
        (128, "ln"): "attn-bf16-d128-nc-lse-ln-8953d00478ded858",
        (256, "log2"): "attn-bf16-d256-nc-lse-log2-9bfafd1d8cbeb36d",
        (256, "ln"): "attn-bf16-d256-nc-lse-ln-67f2e98028fae0ef",
    }
    nc_sink = {
        64: "attn-bf16-d64-nc-sink-e2b883b827085097",
        128: "attn-bf16-d128-nc-sink-f380a0e936c103f3",
        256: "attn-bf16-d256-nc-sink-f0a0bf2c7f86e1b7",
    }
    nc_sink_lse = {
        (64, "log2"): "attn-bf16-d64-nc-sink-lse-log2-e973b28b732c6827",
        (64, "ln"): "attn-bf16-d64-nc-sink-lse-ln-2a3b2f4f2e221fe6",
        (128, "log2"): "attn-bf16-d128-nc-sink-lse-log2-760246bf1153b0c1",
        (128, "ln"): "attn-bf16-d128-nc-sink-lse-ln-43249ae085aa93a0",
        (256, "log2"): "attn-bf16-d256-nc-sink-lse-log2-edc47c7137f23613",
        (256, "ln"): "attn-bf16-d256-nc-sink-lse-ln-5e999137d1b8c7e3",
    }
    assert nc == spec.NC_FINGERPRINTS and nc_lse == spec.NC_LSE_FINGERPRINTS
    assert nc_sink == spec.NC_SINK_FINGERPRINTS
    assert nc_sink_lse == spec.NC_SINK_LSE_FINGERPRINTS
    for D in (64, 128, 256):
        assert bi.fingerprint(op, head_dim=D, causal=False) == nc[D]
        assert bi.fingerprint(op, head_dim=D, causal=False, sinks=True) == nc_sink[D]
        assert bi.fingerprint(op, head_dim=D, causal=True) == spec.FINGERPRINTS[D]
        assert (
            bi.fingerprint(op, head_dim=D, causal=True, sinks=True)
            == spec.SINK_FINGERPRINTS[D]
        )
        for base in ("log2", "ln"):
            got = bi.fingerprint(op, head_dim=D, causal=False, lse_base=base)
            assert got == nc_lse[(D, base)]
            got = bi.fingerprint(
                op, head_dim=D, causal=False, lse_base=base, sinks=True
            )
            assert got == nc_sink_lse[(D, base)]
            got = bi.fingerprint(op, head_dim=D, causal=True, lse_base=base)
            assert got == spec.LSE_FINGERPRINTS[(D, base)]
            got = bi.fingerprint(op, head_dim=D, causal=True, lse_base=base, sinks=True)
            assert got == spec.SINK_LSE_FINGERPRINTS[(D, base)]
    every = (
        list(nc.values())
        + list(nc_lse.values())
        + list(nc_sink.values())
        + list(nc_sink_lse.values())
        + list(spec.FINGERPRINTS.values())
        + list(spec.LSE_FINGERPRINTS.values())
        + list(spec.SINK_FINGERPRINTS.values())
        + list(spec.SINK_LSE_FINGERPRINTS.values())
    )
    assert len(set(every)) == 36
    if op != WRAPPERS[0]:  # the prefill wrappers' member receipts need the mode too
        for dims in (dict(lse_base="ln"), dict(sinks=True)):
            with pytest.raises(TypeError, match="causal"):
                bi.fingerprint(op, head_dim=128, **dims)


@pytest.mark.cpu_only
def test_noncausal_member_kernels_have_their_own_names():
    """A non-causal kernel with sinks is an object of its own: the mode, then the member, in the row body, the Swaps body
    and the fixup."""
    from flashinfer.attention.batch_invariant import fixup
    from flashinfer.attention.batch_invariant.blocks import compile as row
    from flashinfer.attention.batch_invariant.swaps import compile as swaps

    base = row.kernel_name(128, 4, 16, 4)
    assert row.kernel_name(128, 4, 16, 4, causal=False) == base + "_nc"
    assert row.kernel_name(128, 4, 16, 4, sinks=True) == base + "_sink"
    assert row.kernel_name(128, 4, 16, 4, causal=False, sinks=True) == base + "_nc_sink"
    args = (128, 8, 16, 8, False, 228 << 10, True)
    base = swaps.kernel_name(*args)
    assert swaps.kernel_name(*args, causal=False) == base + "_nc"
    assert swaps.kernel_name(*args, causal=False, sinks=True) == base + "_nc_sink"
    base = fixup.kernel_name(128, 4)
    assert fixup.kernel_name(128, 4, causal=False) == base + "_nc"
    assert fixup.kernel_name(128, 4, sinks=True) == base + "_sink"
    assert fixup.kernel_name(128, 4, causal=False, sinks=True) == base + "_nc_sink"


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ["prefill", "ragged", "decode"])
@pytest.mark.parametrize("causal", [True, False])
def test_mask_mode_reaches_the_plan_and_every_kernel_lookup(monkeypatch, kind, causal):
    """Device-free: a plan through the wrappers' delegate records its mask mode, and a run with the LSE, with sinks and
    with both asks for the kernels of that mode and member -- ``_nc`` under ``causal=False``, ``_nc_sink`` with sinks, in
    the row body or the Swaps body and in the fixup -- whatever else the call carries. The kernels are stubbed."""
    from flashinfer import utils
    from flashinfer.attention.batch_invariant import engine, fixup
    from flashinfer.attention.batch_invariant.blocks import compile as row
    from flashinfer.attention.batch_invariant.blocks import operands as row_operands
    from flashinfer.attention.batch_invariant.swaps import compile as swaps
    from flashinfer.attention.batch_invariant.swaps import operands as swaps_operands
    from flashinfer.attention.batch_invariant.wrappers import BatchInvariantAttention

    names = []

    def stub(module):
        def get(*args, **kw):
            names.append(module.kernel_name(*args, **kw))
            return lambda *operands: None

        return get

    monkeypatch.setattr(row, "get_compiled_attention", stub(row))
    monkeypatch.setattr(swaps, "get_compiled_swaps", stub(swaps))
    monkeypatch.setattr(fixup, "get_fixup", stub(fixup))
    monkeypatch.setattr(engine, "device_sm_count", lambda device: 204)
    monkeypatch.setattr(engine, "hw_row_max_enabled", lambda device=None, D=128: False)
    monkeypatch.setattr(row_operands, "arch_smem_cap", lambda device=None: 228 << 10)
    monkeypatch.setattr(swaps_operands, "arch_smem_cap", lambda device=None: 228 << 10)
    monkeypatch.setattr(utils, "device_support_pdl", lambda device: False)

    D, Hq, Hkv, page, q_len = 256, 4, 2, 16, 3
    kv_lens = [5000, 40, 2100]
    q_lens = [q_len] * 3 if kind == "decode" else [700, 3, 2100]
    d = object.__new__(BatchInvariantAttention)
    d.kind, d.kv_layout, d.device = kind, "NHD", torch.device("cpu")
    d.workspace = torch.empty(64 << 20, dtype=torch.uint8)
    d._planned = None
    n_pages = [-(-L // page) for L in kv_lens]
    paged = kind != "ragged"
    kv_indptr = _i32(
        [0] + list(torch.cumsum(torch.tensor(n_pages if paged else kv_lens), 0))
    )
    d.plan(
        qo_indptr=None
        if kind == "decode"
        else _i32([0] + list(torch.cumsum(torch.tensor(q_lens), 0))),
        kv_indptr=kv_indptr,
        kv_indices=torch.arange(sum(n_pages), dtype=torch.int32) if paged else None,
        kv_last_page_len=_i32(
            [L - (n - 1) * page for L, n in zip(kv_lens, n_pages, strict=True)]
        )
        if paged
        else None,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim_qk=D,
        head_dim_vo=D,
        page_size=page,
        causal=causal,
        window_left=-1,
        logits_soft_cap=None,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=None,
        q_len_per_req=q_len,
    )
    assert d._causal is causal and d._planned.causal is causal
    plan = d._planned.plan
    if (
        kind == "decode"
    ):  # split units: the fold that completes their rows is a kernel lookup too
        assert plan.groups and engine.swaps_n_pad(plan) > 0
    else:
        assert engine.swaps_n_pad(plan) == 0  # the row body
    q = torch.zeros(sum(q_lens), Hq, D, dtype=torch.bfloat16)
    if paged:
        k = torch.zeros(sum(n_pages), page, Hkv, D, dtype=torch.bfloat16)
    else:
        k = torch.zeros(sum(kv_lens), Hkv, D, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    sinks = torch.zeros(Hq)
    mode = "" if causal else "_nc"
    for given, return_lse in ((None, True), (sinks, False), (sinks, True)):
        names.clear()
        d.run(
            q,
            k,
            v,
            out=None,
            return_lse=return_lse,
            lse=None,
            window_left=None,
            sinks=given,
        )
        want = mode + ("" if given is None else "_sink")
        assert names and all(n.endswith(want) for n in names), (names, want)
        assert all(("_nc" in n) is (not causal) for n in names), names
        assert all(("_sink" in n) is (given is not None) for n in names), names
        assert any(n.startswith("attn_fixup_") for n in names) is bool(
            plan.fixup_groups
        )
    # the engine's own entry points carry the mode the same way
    planned = engine.plan_attention(
        _i32([0, 3, 6]),
        _i32([0, 300, 900]),
        None,
        Hq,
        Hkv,
        D,
        1,
        -1,
        None,
        None,
        "cpu",
        ragged=True,
        causal=causal,
    )
    assert planned.causal is causal
    assert engine.plan_storage(planned) and engine.plan_tables(planned)


@pytest.mark.cpu_only
def test_noncausal_member_terms_are_the_members_of_the_final_state():
    """Under the non-causal class the LSE and the sink step read the final ``(m, l)`` of the fold over ``[0, kv_len)``: the
    terms do not depend on the row's position, and they are those of the causal row at the last position."""
    from flashinfer.attention.batch_invariant import spec
    from flashinfer.batch_invariant.fp_interface.term import sexpr

    for D in (64, 128, 256):
        s = spec.AttentionSpec(D, T=16, C=32)
        rows = [spec.RowSpec(p=p, row=1, causal=False, kv_len=45) for p in (0, 20, 44)]
        last = spec.RowSpec(p=44, row=1)
        for base in ("log2", "ln"):
            for sinks in (False, True):
                want = sexpr(s.lse(last, base, sinks=sinks))
                assert {sexpr(s.lse(r, base, sinks=sinks)) for r in rows} == {want}
        want = sexpr(s.output(last, 2, sinks=True))
        assert {sexpr(s.output(r, 2, sinks=True)) for r in rows} == {want}
        short = spec.RowSpec(p=20, row=1, causal=False, kv_len=21)
        assert sexpr(s.lse(short, "log2")) != sexpr(s.lse(rows[1], "log2"))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@NC_SINKS
def test_noncausal_members_match_reference(D, with_sinks):
    """The LSE (both bases) and, with sinks, the output of the non-causal class against fp32 non-causal attention; with sinks
    also against the repository's sink reference under a full mask. Requesting the LSE leaves the output bits; a request with
    more query tokens than keys is a member."""
    pr = Problem([300, 1000, 64, 40], [700, 2500, 3000, 17], 32, 8, D, 16, seed=61)
    kw, sinks = _nc_run_kw(pr, with_sinks, seed=D)
    ref, ref_lse = reference_noncausal_members(pr, sinks)
    out, log2 = run_paged_prefill(pr, causal=False, run_kw=kw)
    out_ln, ln = run_paged_prefill(pr, causal=False, run_kw=dict(kw, lse_base="ln"))
    assert log2.shape == ln.shape == (sum(pr.q_lens), pr.Hq)
    assert not torch.isnan(ln).any() and not torch.isinf(ln).any()
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err
    tol = 1e-3 * max(1.0, ref_lse.abs().max().item())
    assert (ln - ref_lse).abs().max().item() < tol
    assert (log2 - ref_lse / math.log(2.0)).abs().max().item() < tol / math.log(2.0)
    no_lse = {k: v for k, v in kw.items() if k != "return_lse"}
    plain = bits(run_paged_prefill(pr, causal=False, run_kw=no_lse))
    assert torch.equal(bits(out), plain) and torch.equal(bits(out_ln), plain)
    got_r = _nc_ragged(pr, kw)
    assert torch.equal(bits(got_r[0]), plain) and torch.equal(
        bits(got_r[1]), bits(log2)
    )
    if with_sinks:
        from tests.attention.test_helpers.sink_attention_reference import (
            sink_attention_unified,
        )

        repo = sink_attention_unified(
            pr.q.float(),
            pr.k_rag,
            pr.v_rag,
            sinks,
            -1,
            False,
            1.0 / math.sqrt(pr.D),
            mode="varlen",
            qo_indptr=pr.qo_indptr,
            kv_indptr=pr.kv_indptr_rag,
        )
        err = (out.float() - repo.float()).abs().max().item()
        assert err < 2e-2 + 1e-2 * repo.abs().max().item(), err
        without = run_paged_prefill(pr, causal=False)
        assert (without.float() - ref).abs().max().item() > 4 * err
    # the members of the causal class are other terms: the same tokens, other bits (the last row of a request aside)
    causal = Problem([300, 1000, 64], [700, 2500, 3000], 32, 8, D, 16, seed=61)
    ckw, _ = _nc_run_kw(causal, with_sinks, seed=D)
    a = _both(run_paged_prefill(causal, causal=True, run_kw=ckw))
    b = _both(run_paged_prefill(causal, causal=False, run_kw=ckw))
    assert not torch.equal(a[0], b[0]) and not torch.equal(a[1], b[1])
    last = [int(x) - 1 for x in causal.qo_indptr[1:]]
    assert torch.equal(a[0][last], b[0][last]) and torch.equal(a[1][last], b[1][last])


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@NC_SINKS
def test_noncausal_members_prefill_equals_multi_token_decode(D, layout, with_sinks):
    """R4 for the members of the non-causal class: paged prefill, ragged prefill and multi-token decode (``is_causal=False``)
    give a token the same output and LSE bits; the position does not enter (the same query elsewhere in its request has the
    same LSE); a sink of ``-inf`` gives the bits of the call without sinks."""
    q_len = 4
    pr = Problem([256, 33, 1024, 7], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=62)
    kw, sinks = _nc_run_kw(pr, with_sinks, seed=D)
    out, lse = _both(run_paged_prefill(pr, layout, causal=False, run_kw=kw))
    out_r, lse_r = _both(_nc_ragged(pr, kw, layout))
    assert torch.equal(out, out_r) and torch.equal(lse, lse_r)
    res, rows = run_decode_noncausal(pr, q_len, layout, **kw)
    dec, dec_lse = _both(res)
    assert torch.equal(dec, out[rows]) and torch.equal(dec_lse, lse[rows])
    _, ln = _both(
        run_paged_prefill(pr, layout, causal=False, run_kw=dict(kw, lse_base="ln"))
    )
    _, ln_r = _both(_nc_ragged(pr, dict(kw, lse_base="ln"), layout))
    assert torch.equal(ln, ln_r) and not torch.equal(ln, lse)
    pr2 = Problem(pr.q_lens, pr.kv_lens, 32, 8, D, 16, seed=62)
    for i in range(len(pr.q_lens)):
        a, b = int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])
        pr2.q[a:b] = pr.q[a:b].flip(0)
    out2, lse2 = _both(run_paged_prefill(pr2, layout, causal=False, run_kw=kw))
    for i in range(len(pr.q_lens)):
        a, b = int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])
        assert torch.equal(out2[a:b], out[a:b].flip(0)), i
        assert torch.equal(lse2[a:b], lse[a:b].flip(0)), i
    if with_sinks:
        plain = _both(run_paged_prefill(pr, layout, causal=False, run_kw=LSE))
        assert not torch.equal(out, plain[0]) and not torch.equal(lse, plain[1])
        off = torch.full((pr.Hq,), float("-inf"), device=pr.device)
        got = _both(
            run_paged_prefill(pr, layout, causal=False, run_kw=dict(LSE, sinks=off))
        )
        assert torch.equal(got[0], plain[0]) and torch.equal(got[1], plain[1])
        res, rows = run_decode_noncausal(pr, q_len, layout, **dict(LSE, sinks=off))
        got = _both(res)
        assert torch.equal(got[0], plain[0][rows]) and torch.equal(
            got[1], plain[1][rows]
        )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@NC_SINKS
def test_noncausal_members_batch_permutation_and_subbatch_invariance(D, with_sinks):
    """R2 for the members of the non-causal class: a request's output and LSE rows are the same bits in a batch of one, in a
    permuted batch and in a sub-batch; paged prefill, ragged prefill and multi-token decode."""
    q_lens, kv_lens = [300, 1, 1000, 64, 4], [700, 2500, 2500, 3000, 4]
    pr = Problem(q_lens, kv_lens, 32, 8, D, 16, seed=63)
    kw, sinks = _nc_run_kw(pr, with_sinks, seed=D)
    full_p = _both(run_paged_prefill(pr, causal=False, run_kw=kw))
    full_r = _both(_nc_ragged(pr, kw))
    assert torch.equal(full_p[0], full_r[0]) and torch.equal(full_p[1], full_r[1])

    def sub_batch(src, sel, seed):
        sub = Problem(
            [src.q_lens[i] for i in sel],
            [src.kv_lens[i] for i in sel],
            32,
            8,
            D,
            16,
            seed=seed,
        )
        for j, i in enumerate(sel):
            qa, qb = int(src.qo_indptr[i]), int(src.qo_indptr[i + 1])
            sa, sb = int(sub.qo_indptr[j]), int(sub.qo_indptr[j + 1])
            sub.q[sa:sb] = src.q[qa:qb]
            ka, kb = int(src.kv_indptr_rag[i]), int(src.kv_indptr_rag[i + 1])
            ta, tb = int(sub.kv_indptr_rag[j]), int(sub.kv_indptr_rag[j + 1])
            sub.k_rag[ta:tb] = src.k_rag[ka:kb]
            sub.v_rag[ta:tb] = src.v_rag[ka:kb]
        return _repack(sub)

    for sel in ([2], [4, 0], [3, 1, 2], [1, 0, 4, 3, 2]):
        sub = sub_batch(pr, sel, 64)
        got_p = _both(run_paged_prefill(sub, causal=False, run_kw=kw))
        got_r = _both(_nc_ragged(sub, kw))
        for j, i in enumerate(sel):
            qa, qb = int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])
            sa, sb = int(sub.qo_indptr[j]), int(sub.qo_indptr[j + 1])
            for m in (0, 1):
                assert torch.equal(got_p[m][sa:sb], full_p[m][qa:qb]), (sel, i, m)
                assert torch.equal(got_r[m][sa:sb], full_r[m][qa:qb]), (sel, i, m)
    pr4 = Problem(
        [300, 8, 1000, 64, 4], [700, 2500, 2500, 3000, 4], 32, 8, D, 16, seed=65
    )
    res, _ = run_decode_noncausal(pr4, 4, **kw)
    dec = _both(res)
    sub = sub_batch(pr4, (1, 4), 66)
    res, _ = run_decode_noncausal(sub, 4, **kw)
    dsub = _both(res)
    for m in (0, 1):
        assert torch.equal(dsub[m], torch.cat([dec[m][4:8], dec[m][16:20]])), m


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_noncausal_members_config_agreement(D):
    """R3 for the members of the non-causal class: every kernel site that holds a row's final ``(m, l)`` -- the fused unit's
    epilogue, the fixup kernel, the in-kernel fold, the Swaps body and (D = 64) the paired placement -- stores the same LSE
    bits and takes the same sink step, over the configuration space on a prefill-shaped and a small-unit batch. Every
    configuration runs the LSE alone, sinks alone and sinks with the LSE."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    if D == 64:
        prefill = Problem([2048, 2048, 1], [4096, 3000, 1000], 32, 8, D, 16, seed=67)
    else:
        prefill = Problem(
            [256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, seed=67
        )
    small = Problem([1] * 6, [1024, 4096, 33, 700, 2500, 1], 32, 8, D, 16, seed=68)
    space = engine.all_valid_configs(D)
    cfgs = list(space[::7] if D == 64 else space)
    if D == 64:
        two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
        rings = (2, 3) + ((4,) if arch_smem_cap(prefill.device) >= (300 << 10) else ())
        cfgs += [replace(two, kv_slots=s) for s in rings]
    sites = set()
    for pr in (prefill, small):
        sinks = sink_logits(pr, seed=D)
        ref = {
            (base, s is not None): _both(
                run_paged_prefill(
                    pr,
                    causal=False,
                    run_kw=dict(
                        LSE, lse_base=base, **({} if s is None else {"sinks": s})
                    ),
                )
            )
            for base in ("log2", "ln")
            for s in (None, sinks)
        }
        assert not torch.equal(ref[("log2", True)][1], ref[("log2", False)][1])
        sink_out = bits(run_paged_prefill(pr, causal=False, run_kw=dict(sinks=sinks)))
        for n, cfg in enumerate(cfgs):
            planned = engine.plan_attention(
                pr.qo_indptr,
                pr.kv_indptr,
                pr.kv_last,
                32,
                8,
                D,
                16,
                -1,
                cfg,
                None,
                pr.device,
                causal=False,
            )
            plan = planned.plan
            swaps = cfg.small == "swap" and engine.swaps_n_pad(plan) > 0
            body = "swaps" if swaps else "row"
            if any(d[6] == 0 for d in plan.desc):
                sites.add(f"{body}-fused")
            if plan.fixup_groups:
                sites.add(f"{body}-fixup")
            if any(plan.group_inkernel):
                sites.add(f"{body}-inkernel")
            if cfg.q2 and plan.num_launch is not None:
                sites.add("paired")
            base = ("log2", "ln")[n % 2]
            for s, with_lse in ((None, True), (sinks, False), (sinks, True)):
                got = engine.run_attention(
                    planned,
                    pr.q,
                    pr.k_nhd,
                    pr.v_nhd,
                    pr.kv_indptr,
                    pr.kv_indices,
                    1.0 / math.sqrt(D),
                    -1,
                    "NHD",
                    None,
                    cfg,
                    return_lse=with_lse,
                    lse_base=base,
                    sinks=s,
                )
                want = ref[(base, s is not None)]
                what = (cfg, base, s is not None, with_lse)
                if with_lse:
                    out, lse = _both(got)
                    assert torch.equal(lse, want[1]), what
                else:
                    out = bits(got)
                    assert torch.equal(out, sink_out), what
                assert torch.equal(out, want[0]), what
    want_sites = {
        f"{b}-{k}" for b in ("row", "swaps") for k in ("fused", "fixup", "inkernel")
    }
    if D == 64:
        want_sites.add("paired")
    assert sites >= want_sites, want_sites - sites


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_noncausal_members_large_batch_equals_single_requests(D):
    """A large non-causal batch of many requests of several chunks gives every request the output and LSE bits of its
    single-request run, with sinks, under the wrapper's plan and with every request split and its rows completed by the
    fixup kernel and by the in-kernel fold; the plans carry the non-causal mode and every row sees its request's whole key
    range (the last row's bits are the causal last row's)."""
    import random

    rng = random.Random(72 + D)
    n = 40
    kv_lens = [rng.choice([40, 513, 1100, 2100, 3300, 5000]) for _ in range(n)]
    q_lens = [rng.choice([1, 3, 130, 600]) for _ in range(n)]
    pr = Problem(q_lens, kv_lens, 8, 2, D, 16, seed=72)
    sinks = sink_logits(pr, seed=D)
    kw = dict(LSE, sinks=sinks)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device, mb=512), "NHD", batch_invariant=True
    )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        D,
        16,
        causal=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        has_sinks=True,
    )
    assert w._batch_invariant._planned.causal is False
    full = _both(w.run(pr.q, pr.cache("NHD"), **kw))
    # the same batch with every multi-chunk request split, its rows completed by the fixup kernel and by the in-kernel fold
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine

    for merge in ("fixup", "inkernel"):
        cfg = replace(engine.default_config(D=D), split_policy="split", merge=merge)
        planned = engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            pr.Hq,
            pr.Hkv,
            D,
            16,
            -1,
            cfg,
            None,
            pr.device,
            causal=False,
            has_sinks=True,
        )
        plan = planned.plan
        assert planned.causal is False and len(plan.groups) >= n
        if merge == "fixup":
            assert len(plan.fixup_groups) == len(plan.groups)
        else:
            assert any(plan.group_inkernel)
        got = engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
            return_lse=True,
            sinks=sinks,
        )
        got = _both(got)
        assert torch.equal(got[0], full[0]) and torch.equal(got[1], full[1]), merge
    got_r = _both(_nc_ragged(pr, kw))
    assert torch.equal(got_r[0], full[0]) and torch.equal(got_r[1], full[1])
    for i in range(0, n, 3):
        one = Problem([q_lens[i]], [kv_lens[i]], 8, 2, D, 16, seed=73)
        qa, qb = int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])
        ka, kb = int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1])
        one.q[:] = pr.q[qa:qb]
        one.k_rag[:] = pr.k_rag[ka:kb]
        one.v_rag[:] = pr.v_rag[ka:kb]
        one = _repack(one)
        got = _both(run_paged_prefill(one, causal=False, run_kw=kw))
        assert torch.equal(got[0], full[0][qa:qb]), i
        assert torch.equal(got[1], full[1][qa:qb]), i
        if q_lens[i] <= kv_lens[i]:
            last = _both(run_paged_prefill(one, causal=True, run_kw=kw))
            assert torch.equal(last[0][-1], got[0][-1]), i
            assert torch.equal(last[1][-1], got[1][-1]), i
            if q_lens[i] > 1:
                assert not torch.equal(last[1][0], got[1][0]), i


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_noncausal_members_on_the_swaps_body(D):
    """Non-causal multi-token decode on the Swaps-AB kernel with the LSE and sinks: the bits of the non-causal prefill rows
    (the row body), including a request with fewer keys than query tokens."""
    from flashinfer.attention.batch_invariant import engine

    q_len, Hq, Hkv = 4, 16, 8
    pr = Problem(
        [256, 33, 1024, 7, 4], [1500, 2500, 1024, 4096, 3], Hq, Hkv, D, 16, seed=69
    )
    sinks = sink_logits(pr, seed=D)
    for kw in (LSE, dict(LSE, sinks=sinks)):
        rows = torch.cat(
            [
                torch.arange(int(e) - q_len, int(e), device=pr.device)
                for e in pr.qo_indptr[1:]
            ]
        )
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace(pr.device), "NHD", batch_invariant=True
        )
        w.plan(
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            Hq,
            Hkv,
            D,
            16,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            q_len_per_req=q_len,
            is_causal=False,
            has_sinks="sinks" in kw,
        )
        planned = w._batch_invariant._planned
        assert planned.causal is False and engine.swaps_n_pad(planned.plan) > 0
        dec = _both(w.run(pr.q[rows].contiguous(), pr.cache("NHD"), **kw))
        pre = _both(run_paged_prefill(pr, causal=False, run_kw=kw))
        assert torch.equal(dec[0], pre[0][rows]) and torch.equal(dec[1], pre[1][rows])
        assert not torch.isinf(pre[1].view(torch.float32)).any()


@needs_device
def test_noncausal_member_refusals():
    """Under the non-causal class: a window stays refused whatever the members asked for; a request with query tokens and no
    key is refused, so no stored LSE is that of an empty row; ``has_sinks`` and the aliasing checks of the LSE buffer and of
    the sinks behave as under the causal class; a refusal leaves the plan usable."""
    pr = Problem([64, 3], [700, 90], 8, 2, 128, 16, seed=70)
    dev = pr.device
    good = sink_logits(pr)
    U = bi.BatchInvariantUnsupportedError
    with pytest.raises(U, match="window_left"):
        run_paged_prefill(pr, causal=False, window_left=64, has_sinks=True)
    with pytest.raises(U, match="window_left"):
        run_ragged_prefill(pr, causal=False, window_left=64, run_kw=LSE)
    empty = Problem([4, 3], [0, 90], 8, 2, 128, 16, seed=71)
    with pytest.raises(U, match="no key"):
        run_paged_prefill(empty, causal=False, run_kw=LSE)
    with pytest.raises(U, match="no key"):
        run_ragged_prefill(empty, causal=False, run_kw=LSE, sinks=sink_logits(empty))

    def paged(**plan_kw):
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace(dev), "NHD", batch_invariant=True
        )
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            pr.Hq,
            pr.Hkv,
            pr.D,
            pr.page,
            causal=False,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            **plan_kw,
        )
        return w

    args = (pr.q, pr.cache("NHD"))
    w = paged()
    plain = _both(w.run(*args, return_lse=True))
    sunk = _both(w.run(*args, return_lse=True, sinks=good))
    assert not torch.equal(plain[0], sunk[0]) and not torch.equal(plain[1], sunk[1])
    w = paged(has_sinks=True)
    with pytest.raises(U, match="has_sinks=True"):
        w.run(*args)
    with pytest.raises(U, match="has_sinks=True"):
        w.run(*args, return_lse=True)
    assert torch.equal(bits(w.run(*args, sinks=good)), sunk[0])
    w = paged(has_sinks=False)
    with pytest.raises(U, match="has_sinks=False"):
        w.run(*args, sinks=good)
    assert torch.equal(bits(w.run(*args)), plain[0])
    res, rows = run_decode_noncausal(pr, 3, **LSE)
    assert torch.equal(bits(res[1]), plain[1][rows])
    d = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(dev), "NHD", batch_invariant=True
    )
    d.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_len_per_req=3,
        is_causal=False,
        has_sinks=True,
    )
    with pytest.raises(U, match="has_sinks=True"):
        d.run(pr.q[rows].contiguous(), pr.cache("NHD"))
    got = d.run(pr.q[rows].contiguous(), pr.cache("NHD"), sinks=good)
    assert torch.equal(bits(got), sunk[0][rows])

    w = paged()
    n = sum(pr.q_lens)
    k = args[1][0]
    out_buf = torch.empty_like(pr.q)
    for buf in (
        torch.empty(n, pr.Hq, dtype=torch.float64, device=dev),
        torch.empty(pr.Hq, n, dtype=torch.float32, device=dev),
        pr.q.view(torch.float32).reshape(-1)[: n * pr.Hq].view(n, pr.Hq),
        k.view(torch.float32).reshape(-1)[: n * pr.Hq].view(n, pr.Hq),
        w._batch_invariant.workspace.view(torch.float32)[: n * pr.Hq].view(n, pr.Hq),
    ):
        with pytest.raises(U):
            w.run(*args, lse=buf, return_lse=True)
    with pytest.raises(U):
        alias = out_buf.view(torch.float32).reshape(-1)[: n * pr.Hq].view(n, pr.Hq)
        w.run(*args, out=out_buf, lse=alias, return_lse=True)
    with pytest.raises(U):
        w.run(*args, out=out_buf, sinks=out_buf.view(torch.float32).reshape(-1)[:8])
    lse_buf = torch.zeros(n, pr.Hq, dtype=torch.float32, device=dev)
    inside = lse_buf.view(-1)[8:16]
    inside.copy_(good)
    with pytest.raises(U):
        w.run(*args, return_lse=True, lse=lse_buf, sinks=inside)
    assert torch.equal(inside, good)
    for bad in (good.double(), torch.zeros(9, device=dev), good.cpu()):
        with pytest.raises(U):
            w.run(*args, sinks=bad)
    got = _both(w.run(*args, return_lse=True, lse=lse_buf, sinks=good))
    assert torch.equal(got[0], sunk[0]) and torch.equal(got[1], sunk[1])
    assert torch.equal(bits(lse_buf), sunk[1])
