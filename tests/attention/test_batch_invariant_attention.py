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


def run_paged_prefill(pr: Problem, layout="NHD", window_left=-1, ws=None, **plan_kw):
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
    return w.run(pr.q, pr.cache(layout))


def run_ragged_prefill(pr: Problem, layout="NHD", window_left=-1, ws=None, **plan_kw):
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
    return w.run(pr.q, k, v)


def run_decode(pr: Problem, layout="NHD", window_left=-1, ws=None, **plan_kw):
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
    return w.run(q, pr.cache(layout)), last


def reference(pr: Problem, window_left=-1):
    """fp32 causal (bottom-right aligned) attention per request, the yardstick for accuracy (never for bits)."""
    outs = []
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
        tok_q += ql
        tok_kv += kl
    return torch.cat(outs)


def _wrapper_case(kind, pr, metadata_device="cuda", **plan_options):
    """Public plan/run arguments for the same one-query-per-request problem."""
    w = getattr(flashinfer, WRAPPERS[kind])(
        workspace(pr.device, mb=8), "NHD", batch_invariant=True
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


def test_cache_key_covers_every_kernel_source():
    from flashinfer.batch_invariant.jit import relative_sources

    srcs = relative_sources("batch_invariant_attention")
    for f in (
        "attention/batch_invariant/blocks/compile.py",
        "attention/batch_invariant/blocks/kernel.py",
        "attention/batch_invariant/swaps/kernel.py",
        "attention/batch_invariant/fixup.py",
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
@pytest.mark.parametrize("D", [64, 128, 256])
def test_config_agreement(D):
    """R3: every member of the configuration space (ring depth, unit kind, softmax warpgroups, fold placement, small-unit body)
    gives the same bits as the wrapper's pick; at D = 64 the paired placement too (two Q tiles per CTA), on a batch large enough
    to be paired. The whole space at D = 128 / 256 (108 / 36 points); every 7th point of the 252 at D = 64 (each distinct kernel
    is a compile)."""
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
    if D == 64:
        two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
        rings = (2, 3) + ((4,) if arch_smem_cap(pr.device) >= (300 << 10) else ())
        cfgs += [replace(two, kv_slots=s) for s in rings]
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
        w.run(pr.q, pr.cache("NHD"), return_lse=True)
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
