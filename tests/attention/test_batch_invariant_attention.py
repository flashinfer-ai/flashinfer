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
def test_config_agreement():
    """R3: every member of the configuration space (ring depth, unit kind, softmax warpgroups, fold placement, small-unit body)
    gives the same bits."""
    from flashinfer.attention.batch_invariant import engine

    pr = Problem([256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, 128, 16, seed=8)
    ref = bits(run_paged_prefill(pr))
    for cfg in engine.all_valid_configs(128)[::7]:
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
        flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="cute-dsl", batch_invariant=True
        )
    # a workspace too small for the split partials names the needed size instead of launching anything
    tiny = torch.empty(4096, dtype=torch.uint8, device=pr.device)
    big = Problem([1] * 4, [32768] * 4, 32, 8, 128, 16, seed=11)
    wd = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        tiny, "NHD", batch_invariant=True
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="bytes"):
        wd.plan(big.kv_indptr, big.kv_indices, big.kv_last, 32, 8, 128, 16, **kw)
