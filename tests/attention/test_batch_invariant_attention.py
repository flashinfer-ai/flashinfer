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
    pr: Problem, layout="NHD", window_left=-1, ws=None, run_kw=None, **plan_kw
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
        causal=True,
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
        causal=True,
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
    fps = {D: bi.fingerprint(op, head_dim=D) for D in (64, 128, 256)}
    assert len(set(fps.values())) == 3
    for D, fp in fps.items():
        assert fp.startswith("attn-bf16-d") and fp == bi.fingerprint(
            WRAPPERS[0], head_dim=D
        )  # one class for the three wrappers
    assert bi.class_name(op) == "attention.bf16"
    with pytest.raises(TypeError):
        bi.fingerprint(op, head_dim=128, page_size=16)


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
        assert bi.fingerprint(op, head_dim=D, lse_base=base) == fp
        assert bi.fingerprint(op, head_dim=D) == spec.FINGERPRINTS[D]
    with pytest.raises(ValueError):
        bi.fingerprint(op, head_dim=128, lse_base="log10")
    with pytest.raises(TypeError):
        bi.fingerprint(op, head_dim=128, lse_layout="HN")


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
        "batch_invariant/sampling/cute_ext.py",
        "attention/batch_invariant/spec.py",
        "attention/batch_invariant/plan.py",
        "batch_invariant/canon.py",
        "batch_invariant/fp_interface/cute.py",
    ):
        assert f in srcs, f
    assert (
        "attention/batch_invariant/wrappers.py" not in srcs
    )  # the API module stays outside the closure


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
    for D, fp in pins.items():
        assert bi.fingerprint(op, head_dim=D, sinks=True) == fp
        assert bi.fingerprint(op, head_dim=D, sinks=False) == spec.FINGERPRINTS[D]
        assert bi.fingerprint(op, head_dim=D) == spec.FINGERPRINTS[D]
    for (D, base), fp in lse_pins.items():
        assert bi.fingerprint(op, head_dim=D, lse_base=base, sinks=True) == fp
        assert (
            bi.fingerprint(op, head_dim=D, lse_base=base)
            == spec.LSE_FINGERPRINTS[(D, base)]
        )
    with pytest.raises(ValueError):
        bi.fingerprint(op, head_dim=128, sinks="yes")


def test_sink_kernels_have_their_own_names():
    """The sink step is compiled into kernels of their own: a call without sinks loads the kernels it always did."""
    from flashinfer.attention.batch_invariant.blocks import compile as row
    from flashinfer.attention.batch_invariant.swaps import compile as swaps

    assert row.kernel_name(128, 4, 16, 4) == "attn_d128_g4_p16_s4_w1"
    assert row.kernel_name(128, 4, 16, 4, sinks=True) == "attn_d128_g4_p16_s4_w1_sink"
    plain = swaps.kernel_name(128, 8, 16, 8, False, 228 << 10, True)
    assert swaps.kernel_name(128, 8, 16, 8, False, 228 << 10, True, True) == (
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
            causal=False,
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
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            ws, "NHD", use_cuda_graph=True, batch_invariant=True
        )
    # plan() arguments outside the class: typed errors before any plan
    for bad in (
        dict(num_qo_heads=0, num_kv_heads=8),
        dict(sm_scale=float("inf")),
        dict(sm_scale=3e38),
        # seq_lens that contradict the page table (the request has 700 keys)
        dict(seq_lens=torch.tensor([699], dtype=torch.int32, device=pr.device)),
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
    with pytest.raises(bi.BatchInvariantUnsupportedError):
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


def test_fixed_launch_kernels_have_their_own_names():
    """The kernels of a fixed launch (``use_cuda_graph=True``) read their unit and group counts from the plan's table and are
    compiled under their own names: every other call loads the kernels it always did."""
    from flashinfer.attention.batch_invariant.blocks import compile as row
    from flashinfer.attention.batch_invariant.swaps import compile as swaps

    assert row.kernel_name(128, 4, 16, 4, fixed=True) == "attn_d128_g4_p16_s4_w1_cg"
    assert row.kernel_name(128, 4, 16, 4, sinks=True, fixed=True) == (
        "attn_d128_g4_p16_s4_w1_sink_cg"
    )
    plain = swaps.kernel_name(128, 8, 16, 8, False, 228 << 10, True)
    assert swaps.kernel_name(128, 8, 16, 8, False, 228 << 10, True, fixed=True) == (
        plain + "_cg"
    )
    assert swaps.kernel_name(128, 8, 16, 8, False, 228 << 10, True, True, True) == (
        plain + "_sink_cg"
    )


def test_planner_fixed_capacity_budget():
    """The planner held to a fixed capacity (``build_plan(unit_budget=, row_budget=)``, a launch captured in a CUDA graph):
    within the budgets the plan is the unbudgeted plan; beyond them the split tiles are fused longest class first until the
    rest fits (placements: the row work -- which rows over which chunks -- is the same multiset either way); a request
    without keys gets no unit under ``null_requests``."""
    from flashinfer.attention.batch_invariant import plan as P

    def row_work(plan):
        # (request, kv head, first token, rows, chunk) of every unit, a split or fused unit alike
        return sorted(
            (d[P.F_REQ], d[P.F_KVHEAD], d[P.F_QTOK0], d[P.F_ROWS], c)
            for d in plan.desc
            for c in range(d[P.F_CHUNK_LO], d[P.F_CHUNK_HI])
        )

    kv = [0, 256, 320, 330]  # 4096, 1024 and 160 keys at page 16
    last = [16, 16, 16]
    qo = [0, 1, 2, 3]
    for D, heads, swap in (
        (128, (32, 8), True),
        (256, (32, 8), False),
        (64, (8, 8), True),
    ):
        kw = dict(sm_count=148, small_swap=swap, swap_max_rows=8)
        free = P.build_plan(qo, kv, last, *heads, D, 16, split_policy="split", **kw)
        fused = P.build_plan(qo, kv, last, *heads, D, 16, split_policy="fused", **kw)
        assert free.num_units > fused.num_units == 24 and not fused.groups
        roomy = P.build_plan(
            qo,
            kv,
            last,
            *heads,
            D,
            16,
            split_policy="split",
            unit_budget=free.num_units,
            row_budget=free.ws_rows,
            **kw,
        )
        assert roomy.desc == free.desc and roomy.groups == free.groups
        assert roomy.group_inkernel == free.group_inkernel
        # one unit or one partial row short of the whole plan: the shortest split class stays fused, the longer ones split
        for budget in (
            dict(unit_budget=free.num_units - 1),
            dict(row_budget=free.ws_rows - 1),
        ):
            part = P.build_plan(
                qo, kv, last, *heads, D, 16, split_policy="split", **budget, **kw
            )
            assert fused.num_units < part.num_units < free.num_units
            assert part.ws_rows < free.ws_rows and row_work(part) == row_work(free)
            longest = max(d[P.F_KVLEN] for d in part.desc)
            assert all(
                (d[P.F_MODE] == 1) == (d[P.F_KVLEN] == longest) for d in part.desc
            )
        # no room for the longest class: everything fused, whatever shorter class would still fit
        for budget in (dict(unit_budget=fused.num_units + 8), dict(row_budget=0)):
            none = P.build_plan(
                qo, kv, last, *heads, D, 16, split_policy="split", **budget, **kw
            )
            assert none.desc == fused.desc and not none.groups and none.ws_rows == 0
    assert P.budget_unit_kinds([True, False], [3, 9], [1, 3], [4, 4]) == [True, False]
    assert P.budget_unit_kinds([False, False], [9, 9], [3, 3], [4, 4], 5) == [True] * 2
    # a request without keys: refused by the planner, no unit under null_requests
    with pytest.raises(AssertionError):
        P.build_plan(qo, [0, 256, 256, 266], last, 32, 8, 128, 16)
    null = P.build_plan(
        qo, [0, 256, 256, 266], last, 32, 8, 128, 16, null_requests=True
    )
    assert sorted({d[P.F_REQ] for d in null.desc}) == [0, 2]


def test_planner_decode_arrays_equal_the_reference():
    """``plan.build_array_plan`` / ``build_decode_plan`` (the array planner a ``use_cuda_graph=True`` replan runs) give the
    tables of ``plan.build_plan`` entry for entry: random decode and prefill batches over head dims, GQA groups, tokens per
    request, windows, unit kinds, fold placements, SM counts, grids, the grid search, budgets, null requests and requests
    without query tokens; ``plan.pair_units_array`` gives ``engine.pair_units``'s table."""
    import random

    import numpy as np

    from flashinfer.attention.batch_invariant import plan as P

    rng = random.Random(7)
    kinds = set()
    rewritten = 0
    for case in range(400):
        D = rng.choice([64, 128, 256])
        Hq, Hkv = rng.choice([(32, 8), (8, 8), (64, 8), (24, 2), (128, 1), (4, 4)])
        q_len = rng.choice([1, 1, 1, 2, 4, 9, 40, 200])
        batch = rng.choice([1, 2, 3, 7, 30])
        top = rng.choice([q_len, 300, 1100, 5000, 20000])
        kv_lens = [
            0 if rng.random() < 0.15 else rng.randint(q_len, max(q_len, top))
            for _ in range(batch)
        ]
        if rng.random() < 0.3:
            kv_lens = [kv_lens[0]] * batch
        # a decode batch (one query length, requests without keys null) or a prefill batch (a query length per request,
        # zero included)
        q_lens = [q_len] * batch
        null = True
        if case % 2:
            null = False
            q_lens = [
                0 if L == 0 or rng.random() < 0.2 else rng.randint(1, min(L, 3 * q_len))
                for L in kv_lens
            ]
        sm = rng.choice([1, 8, 148, 224])
        opts = dict(
            window_left=rng.choice([-1, -1, 0, 100, 3000]),
            split_policy=rng.choice(["auto", "auto", "split", "fused"]),
            sm_count=sm,
            merge=rng.choice(["auto", "auto", "fixup", "inkernel"]),
            small_swap=rng.random() < 0.8,
            swap_max_rows=8,
            grid=rng.choice([0, 0, 1, 5, sm]),
            grid_search=rng.random() < 0.5,
            unit_budget=rng.choice([0, 1, 64, 4 * sm, 10**9]),
            row_budget=rng.choice([-1, 0, 100, 5000]),
        )
        page = 16
        pages = [-(-L // page) for L in kv_lens]
        indptr = [0]
        for n in pages:
            indptr.append(indptr[-1] + n)
        last = [
            L - (n - 1) * page if n else 0 for L, n in zip(kv_lens, pages, strict=True)
        ]
        qo = [0]
        for n in q_lens:
            qo.append(qo[-1] + n)
        ref = P.build_plan(
            qo, indptr, last, Hq, Hkv, D, page, null_requests=null, **opts
        )
        if null:
            got = P.build_decode_plan(
                np.array(kv_lens), q_len, Hq, Hkv, D, kv_indptr=np.array(indptr), **opts
            )
        else:
            got = P.build_array_plan(
                np.array(q_lens),
                np.array(kv_lens),
                Hq,
                Hkv,
                D,
                kv_indptr=np.array(indptr),
                **opts,
            )
        # the next step of a decode loop, planned from this plan: the reference's tables again, whether or not the batch
        # kept its shape (every Q tile on the same key tiles) and only its KV lengths were rewritten
        grown = [L + 1 if L else 0 for L in kv_lens]
        pages = [-(-L // page) for L in grown]
        grown_indptr = [0]
        for n in pages:
            grown_indptr.append(grown_indptr[-1] + n)
        grown_last = [
            L - (n - 1) * page if n else 0 for L, n in zip(grown, pages, strict=True)
        ]
        ref_next = P.build_plan(
            qo, grown_indptr, grown_last, Hq, Hkv, D, page, null_requests=null, **opts
        )
        got_next = P.build_array_plan(
            np.array(q_lens),
            np.array(grown),
            Hq,
            Hkv,
            D,
            null_requests=null,
            previous=got,
            kv_indptr=np.array(grown_indptr),
            **opts,
        )
        assert got_next.desc == ref_next.desc and got_next.groups == ref_next.groups
        assert got_next.group_inkernel == ref_next.group_inkernel
        assert (got_next.ws_rows, got_next.grid) == (ref_next.ws_rows, ref_next.grid)
        rewritten += got_next.shape is got.shape
        # the paired placement's table over arrays, and the table of units alone in their pairs
        from flashinfer.attention.batch_invariant import engine

        for grid in (0, 3, sm):
            assert P.pair_units_array(got.desc_array, grid).tolist() == (
                engine.pair_units(ref.desc, grid)
            ), (case, grid)
        lone = P.lone_units_array(got.desc_array).tolist()
        assert lone[0::2] == ref.desc
        assert all(
            b[P.F_ROWS] == 0 and b[P.F_TILE_LO] == b[P.F_TILE_HI] and b[P.F_GROUP] == -1
            for b in lone[1::2]
        )
        assert got.desc == ref.desc, (case, kv_lens, opts)
        assert got.groups == ref.groups and got.group_inkernel == ref.group_inkernel
        assert got.fixup_groups == ref.fixup_groups and got.ws_rows == ref.ws_rows
        assert (got.num_units, got.num_groups, got.num_fixup_groups) == (
            ref.num_units,
            ref.num_groups,
            ref.num_fixup_groups,
        )
        assert (got.tokens_per_tile, got.g, got.ws_row, got.grid) == (
            ref.tokens_per_tile,
            ref.g,
            ref.ws_row,
            ref.grid,
        )
        kinds.add((bool(ref.groups), bool(ref.fixup_groups), any(ref.group_inkernel)))
    # fused plans and split plans of both fold placements
    assert (False, False, False) in kinds
    assert any(k[1] for k in kinds) and any(k[2] for k in kinds)
    assert 100 < rewritten < 400  # both ways of planning the next step were taken


def _prefix_tables(pr: Problem, kv_lens, reqs=None):
    """The page table of the first ``kv_lens[b]`` keys of every request of ``pr`` (0: a request without pages); ``reqs``:
    slot ``b`` holds request ``reqs[b]`` of ``pr`` instead of request ``b``."""
    indices = pr.kv_indices.cpu().tolist()
    starts = pr.kv_indptr.cpu().tolist()
    indptr, idx, last = [0], [], []
    for slot, L in enumerate(kv_lens):
        b = reqs[slot] if reqs is not None else slot
        assert pr.kv_lens[b] >= L
        n = -(-L // pr.page)
        idx += indices[starts[b] : starts[b] + n]
        indptr.append(indptr[-1] + n)
        last.append(L - (n - 1) * pr.page if n else 0)
    mk = lambda x: torch.tensor(x, dtype=torch.int32, device=pr.device)  # noqa: E731
    return mk(indptr), mk(idx), mk(last)


def _sink_only_lse(sinks, base="log2"):
    """The LSE of a row without keys under sinks, per head: the sink step on the init state, ``add(mul(sink, LOG2E), +0)``
    in base 2 and that times ``LN2`` in base e, in binary32 on the host (finite sinks)."""
    import numpy as np

    from flashinfer.attention.batch_invariant import spec

    s2 = sinks.detach().cpu().numpy().astype(np.float32) * np.float32(spec.LOG2E)
    lse = s2 + np.float32(0.0)
    if base == "ln":
        lse = lse * np.float32(spec.LN2)
    return torch.from_numpy(lse.astype(np.float32)).to(sinks.device)


def _eager_decode(
    pr: Problem, q, kv_lens, q_len, window_left=-1, run_kw=None, reqs=None
):
    """An eager batch-invariant decode of the requests with keys (``kv_lens[b] > 0``): ``(live rows, result)``, the result
    ``None`` when no request has keys."""
    live = [b for b, L in enumerate(kv_lens) if L]
    rows = torch.tensor(
        [b * q_len + t for b in live for t in range(q_len)],
        dtype=torch.long,
        device=pr.device,
    )
    indptr, idx, last = _prefix_tables(pr, kv_lens, reqs)
    if not live:
        return rows, None
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    keep = torch.tensor(live, dtype=torch.long, device=pr.device)
    spans = (indptr[1:] - indptr[:-1])[keep]
    w.plan(
        torch.cat([spans.new_zeros(1), torch.cumsum(spans, 0)]).to(torch.int32),
        idx,
        last[keep],
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        window_left=window_left,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        q_len_per_req=q_len,
    )
    return rows, w.run(q[rows].contiguous(), pr.cache("NHD"), **(run_kw or {}))


class _GraphDecode:
    """A ``use_cuda_graph=True`` batch-invariant decode wrapper over persistent buffers: ``run()`` is captured once, after
    the first plan, and every later step is a replan (``plan`` / ``fast_decode_plan`` / the ``begin_forward`` partial of
    serving engines) followed by a replay of that one graph."""

    def __init__(
        self,
        pr: Problem,
        q_len,
        window_left=-1,
        lse=False,
        sinks=None,
        batch=None,
        ws=None,
    ):
        dev = pr.device
        self.pr, self.q_len, self.window_left = pr, q_len, window_left
        batch = batch or len(pr.kv_lens)
        self.indptr_buf = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
        self.indices_buf = torch.zeros(
            pr.kv_indices.numel() + 5, dtype=torch.int32, device=dev
        )
        self.last_buf = torch.zeros(batch, dtype=torch.int32, device=dev)
        self.w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws if ws is not None else workspace(dev),
            "NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=self.indptr_buf,
            paged_kv_indices_buffer=self.indices_buf,
            paged_kv_last_page_len_buffer=self.last_buf,
            batch_invariant=True,
        )
        g = torch.Generator(device="cpu").manual_seed(77)
        self.q = (
            (torch.randn(batch * q_len, pr.Hq, pr.D, generator=g) * 1.5)
            .to(torch.bfloat16)
            .to(dev)
        )
        self.out = torch.empty_like(self.q)
        self.run_kw = dict(out=self.out)
        self.lse = None
        if lse:
            self.lse = torch.empty(
                self.q.shape[0], pr.Hq, dtype=torch.float32, device=dev
            )
            self.run_kw.update(lse=self.lse, return_lse=True)
        if sinks is not None:
            self.run_kw.update(sinks=sinks)
        self.graph = None

    def plan(self, kv_lens, how="plan", reqs=None, **kw):
        pr = self.pr
        indptr, idx, last = _prefix_tables(pr, kv_lens, reqs)
        args = (indptr, idx, last, pr.Hq, pr.Hkv, pr.D, pr.page)
        opts = dict(
            window_left=self.window_left,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            q_len_per_req=self.q_len,
        )
        opts.update(kw)
        if how == "plan":
            self.w.plan(*args, **opts)
        elif how == "fast":
            flashinfer.decode.fast_decode_plan(
                self.w,
                *args,
                **dict(
                    dict(
                        fixed_split_size=-1,
                        disable_split_kv=False,
                        global_override_indptr_cpu=indptr.cpu(),
                    ),
                    **opts,
                ),
            )
        else:  # the serving engines' replacement of the wrapper's planner
            import functools

            self.w.begin_forward = functools.partial(
                flashinfer.decode.fast_decode_plan, self.w
            )
            self.w.begin_forward(*args, **dict(dict(non_blocking=True), **opts))

    def replay(self):
        cache = self.pr.cache("NHD")
        if self.graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.w.run(self.q, cache, **self.run_kw)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=s):
                    self.w.run(self.q, cache, **self.run_kw)
            torch.cuda.synchronize()
        self.out.fill_(3.0)  # every row is rewritten by the replay
        if self.lse is not None:
            self.lse.fill_(3.0)
        self.graph.replay()
        torch.cuda.synchronize()
        return bits(self.out).clone(), (
            bits(self.lse).clone() if self.lse is not None else None
        )

    def check(self, kv_lens, how="plan", run_kw=None, reqs=None, **kw):
        """Replan to ``kv_lens``, replay, and compare with an eager wrapper: the requests with keys bit for bit, the others
        zeros (LSE ``-inf``, or with sinks the LSE of the sink alone)."""
        self.plan(kv_lens, how, reqs, **kw)
        out, lse = self.replay()
        rows, ref = _eager_decode(
            self.pr, self.q, kv_lens, self.q_len, self.window_left, run_kw, reqs
        )
        ref_out, ref_lse = ref if isinstance(ref, tuple) else (ref, None)
        null = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
        null[rows] = False
        assert ref is None or torch.equal(out[rows], bits(ref_out)), kv_lens
        assert not out[null].any(), kv_lens
        if lse is not None:
            assert ref is None or torch.equal(lse[rows], bits(ref_lse)), kv_lens
            sinks = self.run_kw.get("sinks")
            if sinks is None:
                assert bool((self.lse[null] == float("-inf")).all()), kv_lens
            else:
                want = _sink_only_lse(sinks).expand_as(self.lse)[null]
                assert torch.equal(bits(self.lse[null]), bits(want)), kv_lens
        return self.w._batch_invariant._planned.plan


GRAPH_STEPS = (
    # (KV lengths, replan entry point): short, long over several 512-key chunks, lengths around the chunk and tile
    # boundaries, short again
    ([300, 40, 129], "plan"),
    ([3000, 1500, 700], "fast"),
    ([513, 2049, 64], "partial"),
    ([5120, 5120, 5120], "fast"),
    ([1024, 512, 4097], "partial"),
    ([8, 4, 77], "plan"),
    ([3000, 1500, 700], "fast"),
)


@needs_device
@pytest.mark.parametrize(
    "D,Hq,Hkv,q_len,page",
    [
        (64, 32, 8, 1, 16),
        (128, 32, 8, 1, 16),
        (256, 32, 8, 1, 16),
        (128, 8, 8, 1, 1),
        (128, 64, 8, 1, 16),
        (64, 8, 8, 2, 16),
        (128, 32, 8, 4, 16),
        (256, 32, 8, 3, 1),
        (128, 16, 8, 1, 1),
        (64, 16, 8, 1, 2),
        (256, 16, 8, 1, 4),
        (128, 32, 8, 4, 1),
    ],
)
def test_cuda_graph_decode_replan_equals_eager(D, Hq, Hkv, q_len, page):
    """``use_cuda_graph=True`` on the decode wrapper: one capture of ``run()``, then replans with changing KV lengths (through
    ``plan``, ``fast_decode_plan`` and the ``begin_forward`` partial) each followed by a replay, give the bits of an eager
    wrapper on the same batch -- the Swaps body (at most 8 live rows per unit) and the row body, pages of 1 to 16 (the pool's
    page table is a random permutation with junk pages in between: token-pool indices; pages of at most 4 keys take the split
    placement of the K / V boxes in both bodies)."""
    pr = Problem([1, 1, 1], [5120, 5120, 5120], Hq, Hkv, D, page, seed=11)
    gd = _GraphDecode(pr, q_len)
    fixed = None
    for kv_lens, how in GRAPH_STEPS:
        plan = gd.check(kv_lens, how)
        launch = gd.w._batch_invariant._fixed
        assert fixed is None or launch is fixed  # one launch for the wrapper's lifetime
        fixed = launch
        assert plan.num_units <= fixed.unit_cap and len(plan.groups) <= fixed.group_cap
        assert len(plan.fixup_groups) <= fixed.fixup_cap
    assert gd.w.is_cuda_graph_enabled


@needs_device
@pytest.mark.parametrize("D,q_len", [(64, 1), (128, 1), (256, 1), (128, 4), (256, 4)])
def test_cuda_graph_fixed_launch_unit_kinds_and_budget(D, q_len):
    """The fixed launch under every unit kind and fold placement, plan after plan on one captured graph: all fused, all split
    (in-kernel folds, the fixup kernel, both), the planner's choice, a capacity that the split plan exhausts (it fuses), and
    a plan walked by fewer CTAs than the launch grid; the fixup node replays with and without groups to fold. Every replay
    equals the eager engine."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant import plan as P

    pr = Problem([1, 1, 1], [3000, 3000, 3000], 32, 8, D, 16, seed=12)
    dev = pr.device
    k, v = pr.cache("NHD")
    q = pr.q[:1].repeat(3 * q_len, 1, 1).contiguous()
    q[1::2] = pr.q[1]
    sm_scale = 1.0 / math.sqrt(D)
    qo = torch.arange(4, dtype=torch.int32) * q_len
    cfg = engine.default_config(D=D)
    steps = (
        ([3000, 1500, 700], "split", "auto"),
        ([300, 40, 129], "split", "fixup"),
        ([3000, 3000, 3000], "fused", "auto"),
        ([513, 2049, 64], "split", "inkernel"),
        ([1024, 512, 2000], "auto", "auto"),
        ([3000, 1500, 700], "split", "fixup"),
        ([8, 4, 77], "auto", "auto"),
    )
    for unit_cap, walkers in ((None, None), (64, None), (None, 7)):
        fixed = engine.FixedLaunch(
            3, q_len, 32, 8, D, 16, cfg, workspace(dev), dev, unit_cap=unit_cap
        )
        if walkers and not fixed.n_pad:
            continue  # the row body walks its units on the launch grid
        if walkers:
            # plans for a 7-SM part on this part's launch grid: 7 CTAs walk the units in waves, the others idle
            assert fixed.grid > walkers
            fixed.sm_count = walkers
        indptr_buf = torch.zeros(4, dtype=torch.int32, device=dev)
        indices_buf = torch.zeros(pr.kv_indices.numel(), dtype=torch.int32, device=dev)
        out = torch.empty_like(q)
        graph = None
        kinds = set()
        for kv_lens, policy, merge in steps:
            step_cfg = replace(cfg, split_policy=policy, merge=merge)
            indptr, idx, last = _prefix_tables(pr, kv_lens)
            planned = engine.plan_attention(
                qo, indptr, last, 32, 8, D, 16, config=step_cfg, fixed=fixed
            )
            indptr_buf.copy_(indptr)
            indices_buf[: idx.numel()].copy_(idx)
            plan = planned.plan
            assert plan.num_units <= fixed.unit_cap
            if unit_cap and (kv_lens, policy) == ([3000, 1500, 700], "split"):
                # room for the six chunks of the longest request's units only: the others stay fused
                assert plan.num_units == 64 and len(plan.groups) == 8
            kinds.add(
                (
                    any(d[P.F_MODE] == 1 for d in plan.desc),
                    bool(plan.fixup_groups),
                    any(plan.group_inkernel),
                )
            )
            run = lambda: engine.run_attention(  # noqa: E731
                planned, q, k, v, indptr_buf, indices_buf, sm_scale, out=out
            )
            if graph is None:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=s):
                        run()
                torch.cuda.synchronize()
            out.fill_(3.0)
            graph.replay()
            torch.cuda.synchronize()
            ref = engine.attention_paged(
                q, k, v, qo.to(dev), indptr, idx, last, config=step_cfg
            )
            assert torch.equal(bits(out), bits(ref)), (unit_cap, kv_lens, policy, merge)
            if walkers:
                assert 1 <= int(fixed.counts[P.N_WALK]) <= walkers < plan.num_units
        if unit_cap is None:
            # split and fused plans, fixup and in-kernel folds, all through the one graph
            assert {kd[0] for kd in kinds} == {True, False}
            assert any(kd[1] for kd in kinds) and any(kd[2] for kd in kinds)
        else:
            assert fixed.unit_cap == 64 and any(kd[0] for kd in kinds)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_cuda_graph_split_budget_exhausted(D, monkeypatch):
    """A batch whose split plan exceeds the wrapper's unit capacity is planned fused (the same bits), and the next, shorter
    batch splits again -- on the same captured graph. The capacity is lowered to one unit per SM so that a small batch
    exhausts it."""
    from flashinfer.attention.batch_invariant import engine

    monkeypatch.setattr(engine, "GRAPH_UNIT_WAVES", 1)
    pr = Problem([1, 1], [16000, 16000], 32, 8, D, 16, seed=13)
    gd = _GraphDecode(pr, 1)
    gd.check([700, 300], "plan")
    fixed = gd.w._batch_invariant._fixed
    assert fixed.unit_cap == engine.device_sm_count(pr.device)
    long = gd.check([16000, 15000], "fast")
    # the eager planner is free to split these 16 units into their 32 and 30 chunks; the fixed launch has no room for the
    # longer class, so every unit stays fused
    free = engine.plan_attention(
        torch.arange(3, dtype=torch.int32),
        *_prefix_tables(pr, [16000, 15000])[::2],
        32,
        8,
        D,
        16,
        device=pr.device,
    ).plan
    if free.groups:
        assert free.num_units > fixed.unit_cap
        assert long.num_units == 16 and not long.groups
    gd.check([3000, 1500], "fast")
    gd.check([16000, 2000], "partial")
    assert gd.check([16000, 16000], "plan").num_units == 16


@needs_device
@pytest.mark.parametrize("D,q_len", [(64, 1), (128, 1), (256, 1), (128, 3)])
@pytest.mark.parametrize("lse", [False, True])
def test_cuda_graph_null_requests(D, q_len, lse):
    """Padded slots of a captured batch -- requests without pages -- are null requests under ``use_cuda_graph=True``: their
    output rows are zeros (LSE ``-inf``) and the other requests keep the bits of an eager call on the live sub-batch; the
    slots change from replan to replan. A request with fewer keys than query tokens is still refused."""
    pr = Problem([1] * 4, [2100, 1300, 1300, 2100], 32, 8, D, 16, seed=14)
    gd = _GraphDecode(pr, q_len, lse=lse)
    kw = dict(run_kw=LSE) if lse else {}
    for kv_lens, how in (
        ([700, 0, 1300, 0], "plan"),
        ([0, 50, 0, 2100], "fast"),
        ([0, 0, 0, 0], "fast"),
        ([2100, 1300, 1300, 2100], "partial"),
        ([0, 0, 0, 3], "plan"),
    ):
        gd.check(kv_lens, how, **kw)
    if q_len > 1:
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            gd.plan([700, q_len - 1, 0, 9])


@needs_device
@pytest.mark.parametrize("D,q_len", [(64, 1), (128, 1), (256, 1), (128, 4)])
def test_cuda_graph_lse_and_sinks_replay(D, q_len):
    """The LSE member and the sink member under replan and replay: the caller's LSE buffer and sinks tensor are captured by
    address, the sinks are read in place at every replay, and every replay equals the eager wrapper's ``(out, lse)``; a null
    request's rows are zeros and ``-inf`` with sinks too."""
    pr = Problem([1] * 3, [5120, 5120, 5120], 32, 8, D, 16, seed=15)
    first, second = sink_logits(pr, seed=3), sink_logits(pr, seed=4)
    sinks = first.clone()
    gd = _GraphDecode(pr, q_len, lse=True, sinks=sinks)
    for kv_lens, how, values in (
        ([300, 40, 129], "plan", first),
        ([3000, 1500, 700], "fast", first),
        ([3000, 1500, 700], "fast", second),
        ([513, 0, 64], "partial", second),
        ([5120, 5120, 5120], "fast", first),
    ):
        sinks.copy_(values)
        gd.check(kv_lens, how, run_kw=dict(LSE, sinks=values))


@needs_device
def test_cuda_graph_sliding_window_and_serving_arguments():
    """A sliding window under replan and replay, and the arguments serving engines pass on every plan: ``seq_lens`` equal to
    the page table's KV lengths, ``fixed_split_size`` (-1 or a positive value: accepted and not used) and
    ``disable_split_kv`` (either value: the plan is the default plan). None of them changes a bit."""
    pr = Problem([1] * 3, [5120, 5120, 5120], 32, 8, 128, 16, seed=16)
    gd = _GraphDecode(pr, 1, window_left=700)
    seq = lambda lens: torch.tensor(lens, dtype=torch.int32)  # noqa: E731
    gd.check([300, 40, 129], seq_lens=seq([300, 40, 129]), fixed_split_size=-1)
    gd.check([3000, 1500, 700], "fast")
    split = gd.check([5120, 5120, 2049], fixed_split_size=2048)
    tables = (split.desc_array.copy(), split.groups_array.copy())
    same = gd.check([5120, 5120, 2049], fixed_split_size=2048, disable_split_kv=True)
    assert same.num_groups == split.num_groups and same.num_units == split.num_units
    assert (same.desc_array == tables[0]).all()
    assert (same.groups_array == tables[1]).all()
    gd.check([5120, 800, 2049], "fast", fixed_split_size=2048, disable_split_kv=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gd.plan([300, 40, 129], seq_lens=seq([300, 41, 129]))
    # the same arguments on an eager wrapper
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    tables = _prefix_tables(pr, [3000, 1500, 700])
    args = (*tables, 32, 8, 128, 16)
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    q = gd.q
    w.plan(*args, **kw)
    ref = bits(w.run(q, pr.cache("NHD"))).clone()
    w.plan(
        *args,
        seq_lens=seq([3000, 1500, 700]),
        fixed_split_size=-1,
        disable_split_kv=True,
        **kw,
    )
    assert torch.equal(bits(w.run(q, pr.cache("NHD"))), ref)
    flashinfer.decode.fast_decode_plan(w, *args, fixed_split_size=-1, **kw)
    assert torch.equal(bits(w.run(q, pr.cache("NHD"))), ref)
    w.plan(*args, fixed_split_size=2048, **kw)
    assert torch.equal(bits(w.run(q, pr.cache("NHD"))), ref)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.plan(*args, seq_lens=seq([3000, 1500, 701]), **kw)


@needs_device
def test_cuda_graph_frozen_shape_violations():
    """What a ``use_cuda_graph=True`` wrapper fixes for its lifetime: the batch size, ``q_len_per_req`` and the size of the
    page-index buffer raise ``ValueError`` as on the other backends; the head counts, head dim, page size, window, ``sm_scale``
    and -- from the first ``run()`` -- the LSE mode and the presence of sinks raise the class's typed error. A refused plan
    leaves the wrapper on its last plan. The ragged prefill wrapper still refuses the flag."""
    pr = Problem([1] * 3, [3000, 3000, 3000], 32, 8, 128, 16, seed=17)
    dev = pr.device
    gd = _GraphDecode(pr, 1)
    gd.check([300, 40, 129])
    ref = gd.replay()[0]
    indptr, idx, last = _prefix_tables(pr, [3000, 1500, 700])
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    too_many = torch.zeros(gd.indices_buf.numel() + 1, dtype=torch.int32, device=dev)
    for bad_args, bad_kw in (
        ((indptr[:3], idx, last[:2], 32, 8, 128, 16), {}),  # batch size
        ((indptr, too_many, last, 32, 8, 128, 16), {}),  # more indices than the buffer
        ((indptr, idx, last, 32, 8, 128, 16), dict(q_len_per_req=2)),
    ):
        for planner in (
            gd.w.plan,
            lambda *a, **k: flashinfer.decode.fast_decode_plan(gd.w, *a, **k),
        ):
            with pytest.raises(ValueError) as err:
                planner(*bad_args, **kw, **bad_kw)
            assert not isinstance(err.value, bi.BatchInvariantUnsupportedError)
    for bad_args, bad_kw in (
        ((indptr, idx, last, 32, 8, 128, 16), dict(window_left=64)),
        ((indptr, idx, last, 32, 8, 128, 16), dict(sm_scale=0.5)),
        ((indptr, idx, last, 16, 8, 128, 16), {}),
        ((indptr, idx, last, 32, 4, 128, 16), {}),
        ((indptr, idx, last, 32, 8, 64, 16), {}),
        ((indptr, idx, last, 32, 8, 128, 8), {}),
    ):
        for planner in (
            gd.w.plan,
            lambda *a, **k: flashinfer.decode.fast_decode_plan(gd.w, *a, **k),
        ):
            with pytest.raises(bi.BatchInvariantUnsupportedError):
                planner(*bad_args, **kw, **bad_kw)
    assert torch.equal(gd.replay()[0], ref)  # the refused plans changed nothing
    gd.check([3000, 1500, 700], "fast")
    cache = pr.cache("NHD")
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gd.w.run(gd.q, cache, return_lse=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gd.w.run(gd.q, cache, sinks=sink_logits(pr))
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gd.w.reset_workspace_buffer(workspace(dev), gd.w._int_workspace_buffer)
    # the constructor: the page-table buffers are required, int32 and on the device
    ws = workspace(dev)
    bufs = dict(
        paged_kv_indptr_buffer=gd.indptr_buf,
        paged_kv_indices_buffer=gd.indices_buf,
        paged_kv_last_page_len_buffer=gd.last_buf,
    )
    for name, other in (
        ("paged_kv_indptr_buffer", None),
        ("paged_kv_indices_buffer", gd.indices_buf.to(torch.int64)),
        ("paged_kv_last_page_len_buffer", gd.last_buf.cpu()),
    ):
        with pytest.raises(ValueError):
            flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                ws,
                "NHD",
                use_cuda_graph=True,
                batch_invariant=True,
                **dict(bufs, **{name: other}),
            )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            ws, "NHD", use_cuda_graph=True, batch_invariant=True
        )


@needs_device
@pytest.mark.parametrize("D,q_len", [(128, 1), (256, 4)])
def test_cuda_graph_replans_without_synchronization(D, q_len):
    """A decode loop as serving engines run it: replan and replay step after step with no synchronisation in between (the
    plan's table copies are asynchronous and stream-ordered with the replays). Every step's output, cloned on the stream,
    equals the eager wrapper's."""
    pr = Problem([1] * 4, [4200] * 4, 32, 8, D, 16, seed=18)
    gd = _GraphDecode(pr, q_len)
    gd.check([300, 40, 129, 2000])
    g = torch.Generator(device="cpu").manual_seed(5)
    steps = [
        [int(x) for x in torch.randint(q_len, 4200, (4,), generator=g)]
        for _ in range(24)
    ]
    steps[5][2] = steps[11][0] = steps[11][3] = 0  # padded slots come and go
    outs = []
    for i, kv_lens in enumerate(steps):
        gd.plan(kv_lens, ("fast", "partial", "plan")[i % 3])
        gd.out.fill_(3.0)
        gd.graph.replay()
        outs.append(gd.out.clone())
    torch.cuda.synchronize()
    for kv_lens, out in zip(steps, outs, strict=True):
        rows, ref = _eager_decode(pr, gd.q, kv_lens, q_len)
        null = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
        null[rows] = False
        assert torch.equal(bits(out[rows]), bits(ref)), kv_lens
        assert not bits(out[null]).any(), kv_lens


def test_fixed_launch_prefill_capacity_rule():
    """The unit capacity of a prefill wrapper's fixed launch comes from its request slots and its row capacity alone: any
    split of at most ``max_rows`` query tokens over the slots has at most ``ceil(max_rows / tokens per tile) + slots - 1``
    Q tiles per KV head (every request rounds up at most once)."""
    import random

    from flashinfer.attention.batch_invariant import plan as P

    rng = random.Random(3)
    for Hq, Hkv in ((32, 8), (8, 8), (64, 8), (24, 2)):
        tpt = 128 // (Hq // Hkv)
        for slots, max_rows in ((1, 700), (4, 700), (8, 4096), (16, 33)):
            bound = Hkv * (-(-max_rows // tpt) + slots - 1)
            for trial in range(40):
                cuts = sorted(rng.randint(0, max_rows) for _ in range(slots - 1))
                q_lens = [
                    b - a for a, b in zip([0] + cuts, cuts + [max_rows], strict=False)
                ]
                if trial == 0 and max_rows > (slots - 1) * (tpt + 1):
                    # every request but one a tile and a token: each of them rounds up
                    q_lens = [tpt + 1] * (slots - 1)
                    q_lens.append(max_rows - sum(q_lens))
                if trial % 5 == 4:
                    q_lens[rng.randrange(slots)] = 0
                qo = [0]
                for n in q_lens:
                    qo.append(qo[-1] + n)
                plan = P.build_plan(
                    qo,
                    [4800 * i for i in range(slots + 1)],
                    [16] * slots,
                    Hq,
                    Hkv,
                    128,
                    16,
                    split_policy="fused",
                )
                assert plan.num_units <= bound, (Hq, Hkv, slots, max_rows, q_lens)
                if trial == 0 and slots > 1 and max_rows > (slots - 1) * (tpt + 1):
                    assert plan.num_units >= Hkv * 2 * (slots - 1)


def _eager_prefill(pr: Problem, q, q_lens, kv_lens, window_left=-1, run_kw=None):
    """An eager batch-invariant paged prefill of the first ``sum(q_lens)`` rows of ``q`` over the first ``kv_lens[b]`` keys
    of every request of ``pr``; ``None`` when no request has a query token."""
    total = sum(q_lens)
    if not total:
        return None
    indptr, idx, last = _prefix_tables(pr, kv_lens)
    qo = torch.tensor(
        [sum(q_lens[:b]) for b in range(len(q_lens) + 1)],
        dtype=torch.int32,
        device=pr.device,
    )
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(
        qo,
        indptr,
        idx,
        last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        causal=True,
        window_left=window_left,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    return w.run(q[:total].contiguous(), pr.cache("NHD"), **(run_kw or {}))


class _GraphPrefill:
    """A ``use_cuda_graph=True`` batch-invariant paged prefill wrapper over persistent buffers with ``max_rows`` query rows
    of capacity: ``run()`` is captured once on capacity-shaped ``q`` / ``out`` (/ ``lse``), and every later step is a
    ``plan()`` with its own query and KV lengths per request slot followed by a replay of that one graph."""

    JUNK = 3.0

    def __init__(
        self, pr: Problem, max_rows, window_left=-1, lse=None, sinks=None, declare=True
    ):
        dev = pr.device
        self.pr, self.max_rows, self.window_left = pr, max_rows, window_left
        slots = len(pr.kv_lens)
        mk = lambda n: torch.zeros(n, dtype=torch.int32, device=dev)  # noqa: E731
        self.qo_buf, self.indptr_buf, self.last_buf = (
            mk(slots + 1),
            mk(slots + 1),
            mk(slots),
        )
        self.indices_buf = mk(pr.kv_indices.numel() + 5)
        self.w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace(dev),
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=self.qo_buf,
            paged_kv_indptr_buf=self.indptr_buf,
            paged_kv_indices_buf=self.indices_buf,
            paged_kv_last_page_len_buf=self.last_buf,
            max_total_num_rows=max_rows if declare else None,
            batch_invariant=True,
        )
        g = torch.Generator(device="cpu").manual_seed(78)
        self.q = (
            (torch.randn(max_rows, pr.Hq, pr.D, generator=g) * 1.5)
            .to(torch.bfloat16)
            .to(dev)
        )
        self.out = torch.empty_like(self.q)
        self.run_kw = dict(out=self.out)
        self.lse = None
        self.hn = False
        if lse is not None:
            self.hn = lse.get("lse_layout") == "HN"
            shape = (pr.Hq, max_rows) if self.hn else (max_rows, pr.Hq)
            self.lse = torch.empty(shape, dtype=torch.float32, device=dev)
            self.run_kw.update(lse=self.lse, return_lse=True, **lse)
        if sinks is not None:
            self.run_kw.update(sinks=sinks)
        self.graph = None

    def plan(self, q_lens, kv_lens, **kw):
        pr = self.pr
        indptr, idx, last = _prefix_tables(pr, kv_lens)
        qo = torch.tensor(
            [sum(q_lens[:b]) for b in range(len(q_lens) + 1)],
            dtype=torch.int32,
            device=pr.device,
        )
        opts = dict(
            causal=True,
            window_left=self.window_left,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        opts.update(kw)
        self.w.plan(qo, indptr, idx, last, pr.Hq, pr.Hkv, pr.D, pr.page, **opts)

    def launch(self):
        cache = self.pr.cache("NHD")
        if self.graph is None:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self.w.run(self.q, cache, **self.run_kw)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=s):
                    self.w.run(self.q, cache, **self.run_kw)
            torch.cuda.synchronize()
        self.out.fill_(self.JUNK)
        if self.lse is not None:
            self.lse.fill_(self.JUNK)
        self.graph.replay()

    def check(self, q_lens, kv_lens, run_kw=None, **kw):
        """Replan, replay, and compare with an eager wrapper: the planned rows bit for bit, the rows beyond untouched."""
        self.plan(q_lens, kv_lens, **kw)
        self.launch()
        torch.cuda.synchronize()
        total = sum(q_lens)
        ref = _eager_prefill(self.pr, self.q, q_lens, kv_lens, self.window_left, run_kw)
        ref_out, ref_lse = ref if isinstance(ref, tuple) else (ref, None)
        if ref is not None:
            assert torch.equal(bits(self.out[:total]), bits(ref_out)), (q_lens, kv_lens)
        assert bool((self.out[total:] == self.JUNK).all()), (q_lens, kv_lens)
        if self.lse is not None:
            live = self.lse[:, :total] if self.hn else self.lse[:total]
            if ref is not None:
                assert torch.equal(bits(live), bits(ref_lse)), (q_lens, kv_lens)
            if not self.hn:
                assert bool((self.lse[total:] == self.JUNK).all()), (q_lens, kv_lens)
        return self.w._batch_invariant._planned.plan


PREFILL_GRAPH_STEPS = (
    # (query lengths, KV lengths) per request slot: a full bucket, slots without rows, prefixes growing across 512-key
    # chunks, Q tiles of 128 rows and more, a nearly empty batch, an empty one, the full bucket again
    ([300, 200, 150, 50], [300, 1200, 150, 2600]),
    ([5, 0, 130, 0], [600, 0, 2900, 0]),
    ([129, 1, 0, 33], [129, 3000, 77, 545]),
    ([0, 0, 1, 0], [0, 0, 1, 0]),
    ([0, 700, 0, 0], [0, 3000, 0, 0]),
    ([0, 0, 0, 0], [0, 40, 0, 0]),
    ([257, 64, 250, 129], [1025, 512, 3000, 513]),
    ([300, 200, 150, 50], [300, 1200, 150, 2600]),
)


@needs_device
@pytest.mark.parametrize(
    "D,Hq,Hkv,page",
    [
        (64, 32, 8, 16),
        (128, 32, 8, 16),
        (256, 32, 8, 16),
        (128, 8, 8, 1),
        (64, 8, 8, 16),
        (128, 64, 8, 16),
        (128, 16, 8, 1),
        (64, 16, 8, 2),
        (256, 16, 8, 4),
    ],
)
def test_cuda_graph_prefill_replan_equals_eager(D, Hq, Hkv, page):
    """``use_cuda_graph=True`` on the paged prefill wrapper: one capture of ``run()`` over capacity-shaped ``q`` / ``out``,
    then plans that change every slot's query and KV lengths (zero-length slots, prefixes growing across chunks, Q tiles
    of 128 rows and more, a full and a nearly empty bucket) each followed by a replay, give the bits of an eager wrapper
    on the same batch in the planned rows and leave the rows beyond untouched. The page table is a random permutation of the
    pool (token-pool indices at page size 1); pages of at most 4 keys take the split placement of the K / V boxes."""
    pr = Problem([1] * 4, [3000] * 4, Hq, Hkv, D, page, seed=21)
    gp = _GraphPrefill(pr, 700)
    for q_lens, kv_lens in PREFILL_GRAPH_STEPS:
        plan = gp.check(q_lens, kv_lens)
        fixed = gp.w._batch_invariant._fixed
        assert plan.num_units <= fixed.unit_cap and plan.num_groups <= fixed.group_cap
        assert plan.num_fixup_groups <= fixed.fixup_cap
    assert fixed.n_pad == 0 and gp.w.is_cuda_graph_enabled


@needs_device
def test_cuda_graph_prefill_paired_placement():
    """A prefill wrapper whose default is the paired placement (D = 64, GQA group 4) keeps the paired kernel for its
    lifetime under ``use_cuda_graph=True``: a batch with the units for it runs real pairs, a smaller one every unit alone in
    its pair, on the one captured graph, with the bits of the eager wrapper (which switches kernels instead)."""
    from flashinfer.attention.batch_invariant import engine

    pr = Problem([1, 1], [5000, 5000], 32, 8, 64, 16, seed=22)
    gp = _GraphPrefill(pr, 4096)
    fixed = None
    seen = set()
    for q_lens, kv_lens in (
        ([4000, 96], [4100, 600]),
        ([40, 0], [40, 0]),
        ([2048, 2048], [5000, 2048]),
        ([300, 7], [1300, 4000]),
        ([4096, 0], [5000, 0]),
    ):
        plan = gp.check(q_lens, kv_lens)
        fixed = gp.w._batch_invariant._fixed
        if not fixed.config.q2:
            pytest.skip("the paired placement is not this part's default")
        paired = plan.num_units >= engine.Q2_MIN_UNITS_PER_SM[64] * fixed.sm_count
        assert paired == (plan.num_launch < plan.num_units)
        assert paired or plan.num_launch == plan.num_units
        seen.add(paired)
    assert seen == {True, False}


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_cuda_graph_prefill_fixed_launch_unit_kinds_and_budget(D):
    """The fixed launch of a prefill geometry under every unit kind and fold placement, plan after plan on one captured
    graph: fused, split with in-kernel folds and with the fixup kernel (more groups than its fixed grid holds: the blocks
    stride), the planner's choice, and a capacity the split plan exhausts."""
    from dataclasses import replace

    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant import plan as P

    pr = Problem([1] * 3, [3000] * 3, 32, 8, D, 16, seed=23)
    dev = pr.device
    k, v = pr.cache("NHD")
    max_rows = 400
    g = torch.Generator(device="cpu").manual_seed(5)
    q = (torch.randn(max_rows, 32, D, generator=g) * 1.5).to(torch.bfloat16).to(dev)
    sm_scale = 1.0 / math.sqrt(D)
    cfg = engine.default_config(D=D)
    steps = (
        ([200, 100, 100], [3000, 1500, 700], "split", "auto"),
        ([40, 0, 9], [300, 0, 129], "split", "fixup"),
        ([100, 200, 100], [3000, 3000, 3000], "fused", "auto"),
        ([3, 300, 1], [513, 2049, 64], "split", "inkernel"),
        ([64, 65, 0], [1024, 512, 0], "auto", "auto"),
        ([200, 100, 100], [3000, 1500, 700], "split", "fixup"),
        ([1, 1, 1], [8, 4, 77], "auto", "auto"),
    )
    for unit_cap in (None, 150):
        fixed = engine.FixedLaunch(
            3,
            None,
            32,
            8,
            D,
            16,
            cfg,
            workspace(dev),
            dev,
            unit_cap=unit_cap,
            max_rows=max_rows,
        )
        indptr_buf = torch.zeros(4, dtype=torch.int32, device=dev)
        indices_buf = torch.zeros(pr.kv_indices.numel(), dtype=torch.int32, device=dev)
        out = torch.empty_like(q)
        graph = None
        kinds = set()
        for q_lens, kv_lens, policy, merge in steps:
            step_cfg = replace(cfg, split_policy=policy, merge=merge)
            indptr, idx, last = _prefix_tables(pr, kv_lens)
            qo = torch.tensor(
                [sum(q_lens[:b]) for b in range(4)], dtype=torch.int32, device=dev
            )
            planned = engine.plan_attention(
                qo, indptr, last, 32, 8, D, 16, config=step_cfg, fixed=fixed
            )
            indptr_buf.copy_(indptr)
            indices_buf[: idx.numel()].copy_(idx)
            plan = planned.plan
            assert plan.num_units <= fixed.unit_cap
            kinds.add(
                (
                    any(d[P.F_MODE] == 1 for d in plan.desc),
                    plan.num_fixup_groups > fixed.fixup_grid_y,
                    any(plan.group_inkernel),
                )
            )
            run = lambda: engine.run_attention(  # noqa: E731
                planned, q, k, v, indptr_buf, indices_buf, sm_scale, out=out
            )
            if graph is None:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=s):
                        run()
                torch.cuda.synchronize()
            out.fill_(3.0)
            graph.replay()
            torch.cuda.synchronize()
            total = sum(q_lens)
            ref = engine.attention_paged(
                q[:total].contiguous(), k, v, qo, indptr, idx, last, config=step_cfg
            )
            assert torch.equal(bits(out[:total]), bits(ref)), (
                unit_cap,
                q_lens,
                policy,
                merge,
            )
            assert bool((out[total:] == 3.0).all())
        if unit_cap is None:
            assert {kd[0] for kd in kinds} == {True, False}
            assert any(kd[1] for kd in kinds) and any(kd[2] for kd in kinds)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NH", "HN"])
def test_cuda_graph_prefill_lse_and_sinks_replay(D, layout):
    """The LSE member (base e, both layouts) and the sink member of a prefill wrapper under replan and replay: the caller's
    LSE buffer and sinks tensor are captured by address, the sinks are read in place, and the planned rows equal the eager
    wrapper's ``(out, lse)``."""
    pr = Problem([1] * 3, [3000] * 3, 32, 8, D, 16, seed=24)
    first, second = sink_logits(pr, seed=5), sink_logits(pr, seed=6)
    sinks = first.clone()
    lse = dict(lse_base="ln", lse_layout=layout)
    gp = _GraphPrefill(pr, 500, lse=lse, sinks=sinks)
    for q_lens, kv_lens, values in (
        ([200, 250, 50], [300, 1200, 150], first),
        ([5, 0, 130], [600, 0, 2900], first),
        ([5, 0, 130], [600, 0, 2900], second),
        ([129, 1, 33], [129, 3000, 545], second),
        ([200, 250, 50], [3000, 2600, 150], first),
    ):
        sinks.copy_(values)
        gp.check(q_lens, kv_lens, run_kw=dict(LSE, sinks=values, **lse))


@needs_device
@pytest.mark.parametrize("D", [64, 128])
def test_cuda_graph_prefill_uniform_window_and_chunked(D):
    """Verify-style batches (the same query length in every slot) with a sliding window, and a prefill fed in pieces
    through one captured graph: the pieces' rows, replay after replay, are the rows of the one-shot eager prefill."""
    pr = Problem([1] * 3, [3000] * 3, 32, 8, D, 16, seed=25)
    gp = _GraphPrefill(pr, 24, window_left=700)
    for kv_lens in ([300, 40, 129], [3000, 1500, 700], [513, 2049, 64], [8, 3000, 77]):
        gp.check([8, 8, 8], kv_lens, uniform_q_len=8)
    one = Problem([600], [600], 32, 8, D, 16, seed=26)
    ref = bits(run_paged_prefill(one)).clone()
    chunked = _GraphPrefill(one, 200)
    for piece in range(3):
        chunked.q.copy_(one.q[200 * piece : 200 * (piece + 1)])
        chunked.plan([200], [200 * (piece + 1)])
        chunked.launch()
        torch.cuda.synchronize()
        assert torch.equal(bits(chunked.out), ref[200 * piece : 200 * (piece + 1)])


@needs_device
def test_cuda_graph_prefill_replans_without_synchronization():
    """A sequence of prefill plans and replays with no synchronisation in between: every step's planned rows, cloned on the
    stream, equal the eager wrapper's."""
    pr = Problem([1] * 4, [3000] * 4, 32, 8, 128, 16, seed=27)
    gp = _GraphPrefill(pr, 600)
    gp.check([300, 100, 150, 50], [300, 1200, 150, 2600])
    rng = torch.Generator(device="cpu").manual_seed(9)
    steps = []
    for _ in range(16):
        kv = [int(x) for x in torch.randint(0, 3000, (4,), generator=rng)]
        ql = [
            min(L, int(x))
            for L, x in zip(
                kv, torch.randint(0, 150, (4,), generator=rng), strict=False
            )
        ]
        steps.append((ql, kv))
    outs = []
    for q_lens, kv_lens in steps:
        gp.plan(q_lens, kv_lens)
        gp.launch()
        outs.append(gp.out.clone())
    torch.cuda.synchronize()
    for (q_lens, kv_lens), out in zip(steps, outs, strict=True):
        total = sum(q_lens)
        ref = _eager_prefill(pr, gp.q, q_lens, kv_lens)
        if ref is not None:
            assert torch.equal(bits(out[:total]), bits(ref)), (q_lens, kv_lens)
        assert bool((out[total:] == gp.JUNK).all())


@needs_device
def test_cuda_graph_prefill_frozen_shape_violations():
    """What a ``use_cuda_graph=True`` prefill wrapper fixes: the request slots, the row capacity (``max_total_num_rows``, or
    the first plan's rows) and the size of the page-index buffer raise ``ValueError`` as on the other backends; head counts,
    head dim, page size, window, ``sm_scale`` and the LSE / sinks of the first ``run()`` raise the class's typed error, as
    do a request with fewer keys than query tokens (with no key at all, too) and the plan arguments outside the class. A
    refused plan leaves the wrapper on its last plan."""
    pr = Problem([1] * 3, [3000] * 3, 32, 8, 128, 16, seed=28)
    dev = pr.device
    gp = _GraphPrefill(pr, 400)
    gp.check([200, 100, 100], [300, 1200, 150])
    ref = bits(gp.out).clone()
    indptr, idx, last = _prefix_tables(pr, [3000, 1500, 700])
    mk = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)  # noqa: E731
    qo = mk([0, 100, 200, 300])
    kw = dict(causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    too_many = torch.zeros(gp.indices_buf.numel() + 1, dtype=torch.int32, device=dev)
    for bad_args in (
        (qo[:3], indptr[:3], idx, last[:2], 32, 8, 128, 16),  # request slots
        (
            mk([0, 200, 300, 401]),
            indptr,
            idx,
            last,
            32,
            8,
            128,
            16,
        ),  # rows over capacity
        (qo, indptr, too_many, last, 32, 8, 128, 16),  # more indices than the buffer
    ):
        with pytest.raises(ValueError) as err:
            gp.w.plan(*bad_args, **kw)
        assert not isinstance(err.value, bi.BatchInvariantUnsupportedError)
    short = _prefix_tables(pr, [3000, 99, 700])
    nokeys = _prefix_tables(pr, [3000, 0, 700])
    for bad_args, bad_kw in (
        ((qo, indptr, idx, last, 32, 8, 128, 16), dict(window_left=64)),
        ((qo, indptr, idx, last, 32, 8, 128, 16), dict(sm_scale=0.5)),
        ((qo, indptr, idx, last, 16, 8, 128, 16), {}),
        ((qo, indptr, idx, last, 32, 8, 64, 16), {}),
        ((qo, *short, 32, 8, 128, 16), {}),
        ((qo, *nokeys, 32, 8, 128, 16), {}),
        ((qo, indptr, idx, last, 32, 8, 128, 16), dict(causal=False)),
        ((qo, indptr, idx, last, 32, 8, 128, 16), dict(logits_soft_cap=30.0)),
        (
            (qo, indptr, idx, last, 32, 8, 128, 16),
            dict(custom_mask=torch.ones(10, dtype=torch.bool, device=dev)),
        ),
        (
            (qo, indptr, idx, last, 32, 8, 128, 16),
            dict(prefix_len_ptr=torch.zeros(1, dtype=torch.int32, device=dev)),
        ),
    ):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            gp.w.plan(*bad_args, **dict(kw, **bad_kw))
    gp.launch()
    torch.cuda.synchronize()
    assert torch.equal(bits(gp.out), ref)  # the refused plans changed nothing
    # the arguments serving engines pass on every plan
    gp.check(
        [100, 100, 100],
        [3000, 1500, 700],
        fixed_split_size=4096,
        non_blocking=True,
        custom_mask=None,
        prefix_len_ptr=None,
        token_pos_in_items_ptr=None,
        token_pos_in_items_len=0,
        max_item_len_ptr=None,
    )
    plain = gp.check([100, 100, 100], [3000, 1500, 700])
    tables = (plain.desc_array.copy(), plain.groups_array.copy(), plain.num_groups)
    same = gp.check([100, 100, 100], [3000, 1500, 700], disable_split_kv=True)
    assert same.num_groups == tables[2] and (same.desc_array == tables[0]).all()
    assert (same.groups_array == tables[1]).all()
    cache = pr.cache("NHD")
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gp.w.run(gp.q, cache, return_lse=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gp.w.run(gp.q, cache, sinks=sink_logits(pr))
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gp.w.run(gp.q[:299], cache)  # fewer rows than planned
    # a slot with query rows and no keys is refused, by name; one with neither is padding
    with pytest.raises(
        bi.BatchInvariantUnsupportedError,
        match="request 1 with 7 query tokens over 0 keys",
    ):
        gp.plan([100, 7, 30], [300, 0, 129])
    gp.check([100, 0, 30], [300, 0, 129])
    # without max_total_num_rows the first plan's rows are the capacity
    first = _GraphPrefill(pr, 400, declare=False)
    first.check([100, 100, 50], [300, 1200, 150])
    first.check([250, 0, 0], [3000, 0, 0])
    with pytest.raises(ValueError):
        first.plan([251, 0, 0], [3000, 0, 0])
    # the constructor: the buffers are required, int32 and on the device
    ws = workspace(dev)
    bufs = dict(
        qo_indptr_buf=gp.qo_buf,
        paged_kv_indptr_buf=gp.indptr_buf,
        paged_kv_indices_buf=gp.indices_buf,
        paged_kv_last_page_len_buf=gp.last_buf,
    )
    for name, other in (
        ("qo_indptr_buf", None),
        ("paged_kv_indices_buf", gp.indices_buf.to(torch.int64)),
        ("qo_indptr_buf", gp.qo_buf.cpu()),
    ):
        with pytest.raises(ValueError):
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                ws,
                "NHD",
                use_cuda_graph=True,
                batch_invariant=True,
                **dict(bufs, **{name: other}),
            )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_cuda_graph_decode_loop_token_by_token(D):
    """A decode loop on one captured graph: every step grows every request by one key, replans (``plan()`` and
    ``fast_decode_plan`` in turn) and replays -- 620 steps, across the 128-key tile and the 512-key chunk boundaries one
    token at a time -- and every step's output equals an eager wrapper's."""
    pr = Problem([1] * 3, [1000] * 3, 32, 8, D, 16, seed=31)
    gd = _GraphDecode(pr, 1)
    start = [1, 60, 300]
    gd.check(start)
    eager = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    cache = pr.cache("NHD")
    kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    for step in range(1, 621):
        kv_lens = [L + step for L in start]
        gd.plan(kv_lens, "fast" if step % 2 else "plan")
        gd.out.fill_(3.0)
        gd.graph.replay()
        eager.plan(*_prefix_tables(pr, kv_lens), 32, 8, D, 16, **kw)
        ref = eager.run(gd.q, cache)
        assert torch.equal(bits(gd.out), bits(ref)), (step, kv_lens)


@needs_device
@pytest.mark.parametrize("D", [64, 128])
def test_cuda_graph_decode_bucket_churn(D):
    """The slots of a captured batch change hands: a request finishes and a new short one takes its slot, and a batch
    smaller than the bucket is padded with idle slots of one key (the padding of serving engines that give padded slots a
    length of 1). Every step equals an eager wrapper on the same page table."""
    pr = Problem([1] * 6, [900, 300, 2000, 40, 600, 1500], 32, 8, D, 16, seed=32)
    gd = _GraphDecode(pr, 1, batch=4)
    for reqs, kv_lens, how in (
        ([0, 1, 2, 3], [500, 200, 1500, 39], "plan"),
        ([0, 1, 2, 3], [501, 201, 1501, 40], "fast"),
        (
            [0, 1, 2, 4],
            [502, 202, 1502, 1],
            "fast",
        ),  # request 3 finished: request 4 starts in its slot
        ([0, 1, 2, 4], [503, 203, 1503, 2], "partial"),
        (
            [0, 5, 2, 4],
            [504, 7, 1504, 3],
            "fast",
        ),  # request 1 finished: request 5 starts
        (
            [0, 5, 0, 4],
            [505, 8, 1, 4],
            "fast",
        ),  # request 2 finished, nothing waiting: an idle slot of one key
        ([5, 5, 0, 0], [9, 1, 1, 1], "plan"),  # one live request, three idle slots
        ([1, 3, 5, 4], [300, 40, 1500, 600], "fast"),  # a full bucket of other requests
    ):
        gd.check(kv_lens, how, reqs=reqs)


@needs_device
def test_cuda_graph_capture_and_replay_streams():
    """Capture on a stream of the caller's (``torch.cuda.graph`` with its default error mode, with and without an explicit
    stream), plans issued from the default stream, replays from the default stream or from another one: the plan's table
    copies are stream-ordered work of the planning stream, so a replay on another stream follows a ``wait_stream`` of it
    (and the next plan a ``wait_stream`` of the replay's), with no host synchronisation; every step equals the eager
    wrapper."""
    pr = Problem([1] * 3, [3000] * 3, 32, 8, 128, 16, seed=33)
    cache = pr.cache("NHD")
    g = torch.Generator(device="cpu").manual_seed(11)
    steps = [
        [int(x) for x in torch.randint(1, 3000, (3,), generator=g)] for _ in range(12)
    ]
    for explicit in (True, False):
        gd = _GraphDecode(pr, 1)
        gd.plan([300, 40, 129])
        own = torch.cuda.Stream()
        own.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(own):
            gd.w.run(gd.q, cache, **gd.run_kw)
            gd.graph = torch.cuda.CUDAGraph()
            if explicit:
                with torch.cuda.graph(gd.graph, stream=own):
                    gd.w.run(gd.q, cache, **gd.run_kw)
            else:
                with torch.cuda.graph(gd.graph):
                    gd.w.run(gd.q, cache, **gd.run_kw)
        torch.cuda.synchronize()
        outs = []
        for i, kv_lens in enumerate(steps):
            default = torch.cuda.current_stream()
            gd.plan(kv_lens, ("fast", "plan")[i % 2])  # on the default stream
            if i % 3 == 2:
                gd.graph.replay()  # on the default stream
                outs.append(gd.out.clone())
            else:
                own.wait_stream(default)
                with torch.cuda.stream(own):
                    gd.graph.replay()
                    outs.append(gd.out.clone())
                default.wait_stream(own)
        torch.cuda.synchronize()
        for kv_lens, out in zip(steps, outs, strict=True):
            _, ref = _eager_decode(pr, gd.q, kv_lens, 1)
            assert torch.equal(bits(out), bits(ref)), (explicit, kv_lens)


@needs_device
def test_cuda_graph_wrappers_share_one_workspace():
    """Several ``use_cuda_graph=True`` wrappers on ONE float workspace, as serving engines build them: a wrapper per batch
    bucket and two wrappers of one bucket, each captured once, then replanned and replayed in an interleaved order. The
    workspace holds only the partials of the launch that is running, so every replay equals the eager wrapper."""
    dev = torch.device("cuda")
    ws = workspace(dev)
    wrappers = []
    for i, batch in enumerate((1, 2, 2, 5)):
        pr = Problem([1] * batch, [5000] * batch, 32, 8, 128, 16, seed=40 + i)
        gd = _GraphDecode(pr, 1, ws=ws)
        gd.check([40 + 300 * b for b in range(batch)])
        wrappers.append(gd)
    assert len({gd.w._batch_invariant._fixed.ws.data_ptr() for gd in wrappers}) == 1
    g = torch.Generator(device="cpu").manual_seed(13)
    pending = []
    for step in range(24):
        gd = wrappers[int(torch.randint(0, len(wrappers), (1,), generator=g))]
        batch = len(gd.pr.kv_lens)
        kv_lens = [int(x) for x in torch.randint(1, 5000, (batch,), generator=g)]
        gd.plan(kv_lens, ("fast", "plan", "partial")[step % 3])
        if (
            step % 4 == 3
        ):  # a replan that is never replayed, then another wrapper's turn
            continue
        gd.out.fill_(3.0)
        gd.graph.replay()
        pending.append((gd, kv_lens, gd.out.clone()))
    torch.cuda.synchronize()
    for gd, kv_lens, out in pending:
        _, ref = _eager_decode(gd.pr, gd.q, kv_lens, 1)
        assert torch.equal(bits(out), bits(ref)), kv_lens


@needs_device
def test_cuda_graph_plan_argument_spellings():
    """The spellings serving engines use for a replan: ``data_type`` with or without ``q_data_type`` / ``kv_data_type``, as
    torch dtypes or strings, and ``non_blocking`` True or False, through ``plan()`` and ``fast_decode_plan``."""
    pr = Problem([1] * 3, [3000] * 3, 32, 8, 128, 16, seed=34)
    gd = _GraphDecode(pr, 1)
    gd.check([300, 40, 129])
    none = dict(q_data_type=None, kv_data_type=None)
    for i, spelling in enumerate(
        (
            dict(none, data_type=torch.bfloat16),
            dict(none, data_type="bfloat16"),
            dict(
                data_type=torch.bfloat16, q_data_type=torch.bfloat16, kv_data_type=None
            ),
            dict(data_type="bfloat16", q_data_type="bfloat16", kv_data_type=None),
            dict(q_data_type="bfloat16", kv_data_type="bfloat16"),
        )
    ):
        for how in ("plan", "fast", "partial"):
            for non_blocking in (True, False):
                gd.check(
                    [3000 - 37 * i, 1500 + i, 700],
                    how,
                    non_blocking=non_blocking,
                    **spelling,
                )
    for how in ("plan", "fast"):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            gd.plan([300, 40, 129], how, **dict(none, data_type=torch.float16))


@needs_device
def test_cuda_graph_sinks_declaration_and_neutral_arguments():
    """``use_cuda_graph=True`` with the plan-time sinks declaration and the neutral serving arguments: the first
    ``has_sinks`` given is the wrapper's (a fast replan, which has no such argument, keeps it; another value raises; a run
    that departs from it raises), a declaration after a first ``run()`` must agree with that run, unit scales are accepted
    at ``run()``, and ``_logits_soft_cap`` reads 0.0 after ``plan()`` and after ``fast_decode_plan``."""
    pr = Problem([1] * 3, [3000] * 3, 32, 8, 128, 16, seed=35)
    cache = pr.cache("NHD")
    sinks = sink_logits(pr, seed=7)
    with_sinks = dict(run_kw=dict(sinks=sinks))
    gd = _GraphDecode(pr, 1, sinks=sinks)
    gd.check([300, 40, 129], has_sinks=True, **with_sinks)
    assert gd.w._logits_soft_cap == 0.0
    for how in ("fast", "partial", "plan"):
        gd.check([3000, 1500, 700], how, **with_sinks)
        assert gd.w._batch_invariant._planned.has_sinks is True
        assert gd.w._logits_soft_cap == 0.0
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gd.plan([300, 40, 129], has_sinks=False)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        gd.w.run(gd.q, cache)
    gd.check([513, 2049, 64], "fast", **with_sinks)
    plain = _GraphDecode(pr, 1)
    plain.check([300, 40, 129], has_sinks=False)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        plain.w.run(plain.q, cache, sinks=sinks)
    late = _GraphDecode(pr, 1)
    late.check([300, 40, 129])
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        late.plan([300, 40, 129], has_sinks=True)
    late.check([3000, 1500, 700], has_sinks=False)
    unit = _GraphDecode(pr, 1)
    unit.run_kw.update(q_scale=1.0, k_scale=1.0, v_scale=1.0)
    unit.check([300, 40, 129])
    unit.check([3000, 1500, 700], "fast")
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        unit.w.run(unit.q, cache, k_scale=0.5)


@needs_device
@pytest.mark.parametrize("q_len", [1, 3])
@pytest.mark.parametrize("with_sinks", [False, True])
def test_eager_decode_requests_without_keys(q_len, with_sinks):
    """An eager decode batch with requests without keys (no pages: padded slots): they are skipped, the rows of the
    requests with keys have the bits of the batch of those requests alone, and the skipped rows hold what the class
    gives on the init state: zeros and an LSE of ``-inf``, or with sinks zeros and the LSE of the sink alone. A request
    with some keys but fewer than its query tokens is refused, and a batch without any key gives the fills alone."""
    pr = Problem([1] * 4, [3000] * 4, 32, 8, 128, 16, seed=38)
    cache = pr.cache("NHD")
    g = torch.Generator(device="cpu").manual_seed(5)
    q = (torch.randn(4 * q_len, pr.Hq, pr.D, generator=g) * 1.5).to(torch.bfloat16)
    q = q.to(pr.device)
    sinks = sink_logits(pr, seed=9) if with_sinks else None
    run_kw = dict(return_lse=True, **(dict(sinks=sinks) if with_sinks else {}))
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    opts = dict(
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, q_len_per_req=q_len
    )
    for kv_lens in ([300, 0, 129, 2049], [0, 700, 0, 0], [0, 0, 0, 0]):
        w.plan(*_prefix_tables(pr, kv_lens), pr.Hq, pr.Hkv, pr.D, pr.page, **opts)
        assert (w._batch_invariant._planned.null_rows is not None) == (0 in kv_lens)
        out, lse = w.run(q, cache, **run_kw)
        rows, ref = _eager_decode(pr, q, kv_lens, q_len, run_kw=run_kw)
        null = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
        null[rows] = False
        if ref is not None:
            assert torch.equal(bits(out[rows]), bits(ref[0])), kv_lens
            assert torch.equal(bits(lse[rows]), bits(ref[1])), kv_lens
        assert not bits(out[null]).any(), kv_lens
        if with_sinks:
            want = _sink_only_lse(sinks).expand_as(lse)[null]
            assert torch.equal(bits(lse[null]), bits(want)), kv_lens
        else:
            assert bool((lse[null] == float("-inf")).all()), kv_lens
    # a plan without such a request records nothing to fill
    w.plan(
        *_prefix_tables(pr, [300, 40, 129, 2049]), pr.Hq, pr.Hkv, pr.D, pr.page, **opts
    )
    assert w._batch_invariant._planned.null_rows is None
    if q_len > 1:
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="request 1 with"):
            w.plan(
                *_prefix_tables(pr, [300, q_len - 1, 129, 64]),
                pr.Hq,
                pr.Hkv,
                pr.D,
                pr.page,
                **opts,
            )


@needs_device
def test_prefill_empty_row_lse_bases_under_sinks_follow_the_sink_step():
    """The null-row values in both LSE bases against the declared step on the host, and where the sink's base-2 value is
    not finite: NaN for the output and the LSE, as the step gives on a row with keys."""
    from flashinfer.attention.batch_invariant import engine

    dev = torch.device("cuda")
    sinks = torch.tensor([0.5, -3.25, 7.0, -0.0, 1e-30, 250.0], device=dev)
    for base in ("log2", "ln"):
        o, lse = engine.null_row_values(sinks, engine.LSE_MODES[base])
        assert not bits(o).any()
        assert torch.equal(bits(lse), bits(_sink_only_lse(sinks, base)))
    o, lse = engine.null_row_values(
        torch.tensor([float("inf"), float("-inf"), float("nan"), 3e38], device=dev), 1
    )
    assert bool(torch.isnan(o).all()) and bool(torch.isnan(lse).all())
    assert engine.null_row_values(None, 1) == (0.0, float("-inf"))


def test_attention_cuda_graph_wrappers_flag_lists_the_wrappers_that_take_it():
    """``flashinfer.batch_invariant.ATTENTION_CUDA_GRAPH_WRAPPERS`` names exactly the attention wrappers whose constructor
    takes ``use_cuda_graph=True`` with ``batch_invariant=True``; the others refuse the flag with the typed error."""
    assert "ATTENTION_CUDA_GRAPH_WRAPPERS" in bi.__all__
    assert set(bi.ATTENTION_CUDA_GRAPH_WRAPPERS) == {
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
    }
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws = torch.empty(1 << 20, dtype=torch.uint8, device=dev)
    buf = lambda n: torch.zeros(n, dtype=torch.int32, device=dev)  # noqa: E731
    buffers = {
        "BatchDecodeWithPagedKVCacheWrapper": dict(
            paged_kv_indptr_buffer=buf(3),
            paged_kv_indices_buffer=buf(8),
            paged_kv_last_page_len_buffer=buf(2),
        ),
        "BatchPrefillWithPagedKVCacheWrapper": dict(
            qo_indptr_buf=buf(3),
            paged_kv_indptr_buf=buf(3),
            paged_kv_indices_buf=buf(8),
            paged_kv_last_page_len_buf=buf(2),
        ),
        "BatchPrefillWithRaggedKVCacheWrapper": dict(
            qo_indptr_buf=buf(3), kv_indptr_buf=buf(3)
        ),
    }
    for name, kw in buffers.items():
        cls = getattr(flashinfer, name)
        if name in bi.ATTENTION_CUDA_GRAPH_WRAPPERS:
            w = cls(ws, "NHD", use_cuda_graph=True, batch_invariant=True, **kw)
            assert w.is_cuda_graph_enabled and w._batch_invariant.graph
        else:
            with pytest.raises(bi.BatchInvariantUnsupportedError):
                cls(ws, "NHD", use_cuda_graph=True, batch_invariant=True, **kw)


@needs_device
@pytest.mark.parametrize("kind", ["decode", "prefill"])
def test_cuda_graph_wrappers_across_inference_mode(kind):
    """A ``use_cuda_graph=True`` wrapper may be built, planned, captured and replanned on either side of
    ``torch.inference_mode()`` (and under ``torch.no_grad()``): the fixed launch's tables and the page-table buffers are
    written in place at every plan, whichever side created them, and the replays keep the eager bits."""
    pr = Problem([1] * 3, [3000] * 3, 32, 8, 128, 16, seed=37)

    def make():
        return _GraphDecode(pr, 1) if kind == "decode" else _GraphPrefill(pr, 400)

    def step(g, i):
        if (
            kind == "decode"
        ):  # the last one leaves a request without keys: its rows are filled ahead of the kernels
            g.check(([300, 40, 129], [3000, 1500, 700], [513, 0, 64])[i])
        else:
            g.check(
                *(
                    ([100, 7, 30], [300, 40, 129]),
                    ([0, 390, 10], [0, 3000, 700]),
                    ([64, 64, 64], [513, 2049, 64]),
                )[i]
            )

    with torch.inference_mode():
        inside = (
            make()
        )  # the wrapper and its page-table buffers are created under inference mode
    # the test's own in-place fills need normal tensors on this side
    inside.q = inside.q.clone()
    inside.out = torch.empty_like(inside.q)
    inside.run_kw["out"] = inside.out
    step(
        inside, 0
    )  # the first plan (the fixed launch's tables) and the capture outside
    step(inside, 1)
    with torch.inference_mode():
        step(inside, 2)
    with torch.no_grad():
        step(inside, 0)
    outside = make()
    with torch.inference_mode():
        step(outside, 0)  # the first plan and the capture inside
        step(outside, 1)
    step(outside, 2)
    with torch.no_grad():
        step(outside, 1)
    step(outside, 0)


@needs_device
def test_cuda_graph_device_scale_tensors():
    """Unit scales as device tensors under ``use_cuda_graph=True``: a tensor resolved by the warm-up ``run()`` is served
    from memory inside the capture and the graph replays after replans; a tensor first seen inside a capture raises the
    typed error instead of reading the device."""
    pr = Problem([1] * 3, [3000] * 3, 32, 8, 128, 16, seed=36)
    cache = pr.cache("NHD")
    one = torch.ones((), device=pr.device)
    gd = _GraphDecode(pr, 1)
    gd.run_kw.update(q_scale=one, k_scale=one, v_scale=one)
    gd.check([300, 40, 129])  # warm-up, then capture
    gd.check([3000, 1500, 700], "fast")
    cold = _GraphDecode(pr, 1)
    cold.plan([300, 40, 129])
    cold.w.run(cold.q, cache, out=cold.out)  # a warm-up without the scale
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="capture"):
            cold.w.run(
                cold.q, cache, out=cold.out, k_scale=torch.ones((), device=pr.device)
            )
        cold.w.run(cold.q, cache, out=cold.out, k_scale=one)


# ------------------------------------------------------------------------------------------------------ small pages
def test_split_placement_rule_and_kernel_names():
    """The split placement of the K / V boxes is a function of the page size alone (pages of at most 4 keys of a paged pool,
    never the ragged pool) and part of a kernel's name, so a kernel built for one placement is never loaded for the other."""
    from flashinfer.attention.batch_invariant.blocks import compile as row
    from flashinfer.attention.batch_invariant.blocks.operands import split_pages
    from flashinfer.attention.batch_invariant.swaps import compile as swaps

    assert [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if split_pages(p)] == [1, 2, 4]
    assert not any(split_pages(p, True) for p in (1, 2, 4, 8, 16, 32, 64, 128))
    assert row.kernel_name(128, 2, 1, 4) == "attn_d128_g2_p1_s4_w1_sp"
    assert row.kernel_name(128, 2, 4, 4, fixed=True) == "attn_d128_g2_p4_s4_w1_sp_cg"
    assert row.kernel_name(128, 2, 8, 4) == "attn_d128_g2_p8_s4_w1"
    assert row.kernel_name(128, 2, 128, 4, ragged=True) == "attn_d128_g2_p128_s4_w1_rg"
    assert swaps.kernel_name(128, 2, 1, 8) == "attn_swaps_d128_g2_p1_n8_sp"
    assert (
        swaps.kernel_name(128, 2, 2, 8, fixed=True) == "attn_swaps_d128_g2_p2_n8_sp_cg"
    )
    assert swaps.kernel_name(128, 2, 8, 8) == "attn_swaps_d128_g2_p8_n8"
    # one name per (placement, fixed launch, sinks) combination, in both bodies
    combos = [(p, f, s) for p in (1, 8) for f in (False, True) for s in (False, True)]
    assert (
        len({row.kernel_name(128, 2, p, 4, sinks=s, fixed=f) for p, f, s in combos})
        == 8
    )
    assert (
        len({swaps.kernel_name(128, 2, p, 8, sinks=s, fixed=f) for p, f, s in combos})
        == 8
    )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("page", [1, 2, 4])
def test_small_pages_scattered_prefill_decode_chunked(D, page):
    """Pages of 1, 2 and 4 keys with scattered page indices (a token pool: the page table is a random permutation of a pool with
    junk pages in between), two query heads per KV head: the paged prefill gives the bits of the ragged prefill (whose K / V
    load is one box per half: the split placement delivers the same bytes), in both pool layouts; the decode step of a
    request's last token and a chunked prefill of its last tokens give that prefill's rows; and a request alone, in another
    pool and page order, gives the bits it has in the batch."""
    q_lens = kv_lens = [700, 1300, 257, 129, 31]
    pr = Problem(q_lens, kv_lens, 16, 8, D, page, seed=41)
    full = bits(run_paged_prefill(pr))
    assert torch.equal(full, bits(run_ragged_prefill(pr)))
    assert torch.equal(full, bits(run_paged_prefill(pr, "HND")))
    rows = [
        slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1]))
        for i in range(len(q_lens))
    ]
    kv_rows = [
        slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1]))
        for i in range(len(q_lens))
    ]
    # decode: every request's last token over all its keys (the Swaps body: two live rows per unit)
    for layout in ("NHD", "HND"):
        dec, last = run_decode(pr, layout)
        assert torch.equal(bits(dec), full[last]), layout
    # chunked prefill: the last tokens of every request over all its keys
    tail = [200, 513, 129, 1, 31]
    ch = Problem(tail, kv_lens, 16, 8, D, page, seed=42)
    ch.q = torch.cat(
        [pr.q[rows[i]][q_lens[i] - tail[i] :] for i in range(len(tail))]
    ).contiguous()
    ch.k_rag, ch.v_rag = pr.k_rag.clone(), pr.v_rag.clone()
    ch = _repack(ch)
    out_ch = bits(run_paged_prefill(ch))
    off = 0
    for i, n in enumerate(tail):
        assert torch.equal(out_ch[off : off + n], full[rows[i]][q_lens[i] - n :]), i
        off += n
    # a request alone, in a fresh pool and page order, and the batch in another order
    for i in (1, 3):
        one = Problem([q_lens[i]], [kv_lens[i]], 16, 8, D, page, seed=43 + i)
        one.q = pr.q[rows[i]].clone()
        one.k_rag, one.v_rag = (
            pr.k_rag[kv_rows[i]].clone(),
            pr.v_rag[kv_rows[i]].clone(),
        )
        one = _repack(one)
        assert torch.equal(bits(run_paged_prefill(one)), full[rows[i]]), i
        dec1, _ = run_decode(one)
        assert torch.equal(bits(dec1), full[rows[i]][-1:]), i
    perm = [4, 2, 0, 3, 1]
    pp = Problem(
        [q_lens[i] for i in perm], [kv_lens[i] for i in perm], 16, 8, D, page, seed=47
    )
    pp.q = torch.cat([pr.q[rows[i]] for i in perm]).contiguous()
    pp.k_rag = torch.cat([pr.k_rag[kv_rows[i]] for i in perm]).contiguous()
    pp.v_rag = torch.cat([pr.v_rag[kv_rows[i]] for i in perm]).contiguous()
    pp = _repack(pp)
    permuted = bits(run_paged_prefill(pp))
    off = 0
    for i in perm:
        assert torch.equal(permuted[off : off + q_lens[i]], full[rows[i]]), i
        off += q_lens[i]
