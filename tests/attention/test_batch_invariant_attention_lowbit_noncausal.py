# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The low-bit KV classes of the batch-invariant attention under the non-causal mask mode: the fp8-KV classes
(``attention.e4m3kv`` / ``attention.e5m2kv``), the NVFP4-KV class (``attention.nvfp4kv``) and the token-head-scale classes
(``attention.e4m3kv_th`` / ``attention.e5m2kv_th``) with ``causal=False`` on the prefill wrappers and ``is_causal=False`` with
``q_len_per_req > 1`` on the decode wrapper. Every row then sees every key of its request.

Bitwise tests only. The oracle of a class is the non-causal bf16 class over the class's exact leaves (the fp8 values widened to
bf16, the NVFP4 products), which ``test_batch_invariant_attention.py`` tests on its own; next to it the invariances of the mode
with the classes' own scales: prefill == multi-token decode, paged == ragged, batch / permutation / sub-batch invariance, query
chunks over a request's full key set, the configuration space, and for the token-head-scale classes the specification
evaluated on the CPU. The device-free part pins the fifteen non-causal fingerprints.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.attention.batch_invariant import engine, spec
from tests.attention.test_batch_invariant_attention import (
    WRAPPERS,
    Problem,
    bits,
    reference_noncausal,
    workspace,
)
from tests.attention.test_batch_invariant_attention import (
    run_paged_prefill as bf16_paged_prefill,
)
from tests.attention.test_batch_invariant_attention import (
    run_ragged_prefill as bf16_ragged_prefill,
)
from tests.attention.test_batch_invariant_attention_lowbit_kv import (
    E4,
    E5,
    LOWBIT_FINGERPRINTS,
    fp4_problem,
    needs_device,
    nvfp4_dequant,
    swizzle_v_sf,
)
from tests.attention.test_batch_invariant_attention_lowbit_kv import (
    problem as fp8_problem,
)
from tests.attention.test_batch_invariant_attention_token_head_sf import (
    SM_EXACT,
    TOKEN_HEAD,
    ExactModel,
    exact_problem,
)
from tests.attention.test_batch_invariant_attention_token_head_sf import (
    problem as th_problem,
)

BF16 = torch.bfloat16
KINDS = ("e4m3", "e5m2", "nvfp4", "e4m3th", "e5m2th")
FP8 = {"e4m3": E4, "e5m2": E5}
#: The key scale of the token-head-scale classes' bf16 oracle: one power of two for every key (the scaled scores are the scores
#: times it exactly, which the bf16 class folds into ``sm_scale``), with every value scale 1.
TH_K = 0.125
KIND = pytest.mark.parametrize("kind", KINDS)
HEAD_DIMS = pytest.mark.parametrize("D", [64, 128, 256])

#: The fingerprints of the fifteen non-causal low-bit-KV classes.
NONCAUSAL_FINGERPRINTS = {
    ("e4m3", 64): "attn-e4m3kv-d64-nc-39d0b2db334e635e",
    ("e5m2", 64): "attn-e5m2kv-d64-nc-8a5f3644cb7c316f",
    ("nvfp4", 64): "attn-nvfp4kv-d64-nc-3c92422193e97a4a",
    ("e4m3th", 64): "attn-e4m3kv_th-d64-nc-ba11d98efdfcf97c",
    ("e5m2th", 64): "attn-e5m2kv_th-d64-nc-6640799e7b6d94b6",
    ("e4m3", 128): "attn-e4m3kv-d128-nc-03eda5fe914defe7",
    ("e5m2", 128): "attn-e5m2kv-d128-nc-0330e3838aa2aeb3",
    ("nvfp4", 128): "attn-nvfp4kv-d128-nc-57c79f51f50d12dd",
    ("e4m3th", 128): "attn-e4m3kv_th-d128-nc-4333b85ec4442e2a",
    ("e5m2th", 128): "attn-e5m2kv_th-d128-nc-d758818aaee37d7c",
    ("e4m3", 256): "attn-e4m3kv-d256-nc-541a6edad9700431",
    ("e5m2", 256): "attn-e5m2kv-d256-nc-18fd4f52bf5fcf44",
    ("nvfp4", 256): "attn-nvfp4kv-d256-nc-39cdbac973e71d96",
    ("e4m3th", 256): "attn-e4m3kv_th-d256-nc-85a1c0b8b840c7b6",
    ("e5m2th", 256): "attn-e5m2kv_th-d256-nc-138e6f0dffef9619",
}


# ------------------------------------------------------------------------------------------------------ problems
def _junk(shape, dtype, gen):
    """Finite junk of a cache or scale dtype (the classes are stated for finite values: the non-finite fp8 codes are replaced)."""
    if dtype == torch.float32:
        return torch.randn(shape, generator=gen)
    b = torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)
    if dtype == E5:
        b[(b & 0x7C) == 0x7C] = 0x35
    else:  # e4m3 values, or the bytes of an NVFP4 cache and of its e4m3 block scales
        b[(b & 0x7F) == 0x7F] = 0x35
    return b.view(dtype)


class Case:
    """One problem of a low-bit class: the request geometry and the queries (``pr``, a bf16 ``Problem``), the cache and scale
    tensors by name -- ``k`` / ``v`` and, for the NVFP4 and the token-head-scale classes, ``ksf`` / ``vsf`` -- as the paged
    pools (``pool``, NHD) and packed along the tokens (``rag``), and the per-tensor scales of the classes that have them."""

    def __init__(self, kind, pr, pool, rag, k_scale=None, v_scale=None):
        self.kind, self.pr, self.pool, self.rag = kind, pr, pool, rag
        self.k_scale, self.v_scale = k_scale, v_scale
        self.th = kind.endswith("th")

    @classmethod
    def make(cls, kind, q_lens, kv_lens, D, seed, page=16, Hq=32, Hkv=8):
        if kind == "nvfp4":
            x = fp4_problem(q_lens, kv_lens, Hq, Hkv, D, page, seed)
            pool = dict(k=x.k_nhd, v=x.v_nhd, ksf=x.k_sf, vsf=x.v_sf)
            rag = dict(k=x.k_rag, v=x.v_rag, ksf=x.k_sf_rag, vsf=x.v_sf_rag)
            return cls(kind, x.pr, pool, rag, x.k_scale, x.v_scale)
        if kind.endswith("th"):
            return cls.of_token_head(
                kind, th_problem(q_lens, kv_lens, Hq, Hkv, D, page, seed, FP8[kind[:4]])
            )
        x = fp8_problem(q_lens, kv_lens, Hq, Hkv, D, page, seed, FP8[kind])
        pool, rag = dict(k=x.k_nhd, v=x.v_nhd), dict(k=x.k_rag, v=x.v_rag)
        return cls(kind, x.pr, pool, rag, x.k_scale, x.v_scale)

    @classmethod
    def of_token_head(cls, kind, tp):
        pool = dict(k=tp.k_nhd, v=tp.v_nhd, ksf=tp.ksf_nhd, vsf=tp.vsf_nhd)
        rag = dict(k=tp.k_rag, v=tp.v_rag, ksf=tp.ksf_rag, vsf=tp.vsf_rag)
        return cls(kind, tp.pr, pool, rag)

    def derive(self, q_lens, kv_lens, q, rag, seed):
        """A problem of this class over given query rows and ragged tensors (a sub-batch, a permutation, a query chunk): a
        fresh page table, and pools with finite junk around the live keys."""
        old = self.pr
        pr = Problem(q_lens, kv_lens, old.Hq, old.Hkv, old.D, old.page, seed=seed)
        pr.q = q.contiguous()
        gen = torch.Generator().manual_seed(seed)
        pool = {}
        for name, r in rag.items():
            t = _junk(
                (pr.k_nhd.shape[0], pr.page) + tuple(r.shape[1:]), r.dtype, gen
            ).to(pr.device)
            tok = 0
            for i, L in enumerate(kv_lens):
                pages = pr.kv_indices[
                    int(pr.kv_indptr[i]) : int(pr.kv_indptr[i + 1])
                ].tolist()
                for j, pg in enumerate(pages):
                    n = min(pr.page, L - j * pr.page)
                    t[pg, :n] = r[tok + j * pr.page : tok + j * pr.page + n]
                tok += L
            pool[name] = t
        rag = {n: r.contiguous() for n, r in rag.items()}
        return Case(self.kind, pr, pool, rag, self.k_scale, self.v_scale)

    # -------------------------------------------------------------------------------------------------- the wrappers
    def plan_kw(self):
        kv = torch.uint8 if self.kind == "nvfp4" else FP8[self.kind[:4]]
        kw = dict(q_data_type=BF16, kv_data_type=kv)
        if self.th:
            kw["use_token_head_sf"] = True
        return kw

    def run_kw(self, t, unit=False):
        """The ``run()`` scale keywords over the tensors ``t``; ``unit``: the scales of the bf16 oracle (``oracle``)."""
        kw = {}
        if "ksf" in t:
            kw["kv_cache_sf"] = (t["ksf"], t["vsf"])
            if self.th and unit:
                kw["kv_cache_sf"] = (
                    torch.full_like(t["ksf"], TH_K),
                    torch.ones_like(t["vsf"]),
                )
        if not self.th:
            kw.update(k_scale=self.k_scale, v_scale=1.0 if unit else self.v_scale)
        return kw

    @staticmethod
    def _views(src, layout, axes):
        if layout == "NHD":
            return src
        return {n: t.transpose(*axes).contiguous() for n, t in src.items()}

    def paged(self, layout="NHD", causal=False, unit=False, **plan):
        pr = self.pr
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace(pr.device), layout, batch_invariant=True
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
            **self.plan_kw(),
            **plan,
        )
        t = self._views(self.pool, layout, (1, 2))
        return w.run(pr.q, (t["k"], t["v"]), **self.run_kw(t, unit))

    def ragged(self, layout="NHD", causal=False, unit=False, **plan):
        pr = self.pr
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace(pr.device), layout, batch_invariant=True
        )
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr_rag,
            pr.Hq,
            pr.Hkv,
            pr.D,
            causal=causal,
            **self.plan_kw(),
            **plan,
        )
        t = self._views(self.rag, layout, (0, 1))
        return w.run(pr.q, t["k"], t["v"], **self.run_kw(t, unit))

    def decode(self, q_len, layout="NHD", is_causal=False, unit=False, **plan):
        """The decode wrapper on the LAST ``q_len`` tokens of every request; returns the output and those query rows."""
        pr = self.pr
        rows = last_rows(pr, q_len)
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
            q_len_per_req=q_len,
            is_causal=is_causal,
            **self.plan_kw(),
            **plan,
        )
        t = self._views(self.pool, layout, (1, 2))
        out = w.run(pr.q[rows].contiguous(), (t["k"], t["v"]), **self.run_kw(t, unit))
        return out, rows

    # -------------------------------------------------------------------------------------------------- the oracles
    def exact(self, t):
        """The class's K / V leaves over the tensors ``t`` as bf16 values: the fp8 values widened, the NVFP4 products."""
        if self.kind == "nvfp4":
            k, v = nvfp4_dequant(t["k"], t["ksf"]), nvfp4_dequant(t["v"], t["vsf"])
            assert torch.equal(k.to(BF16).float(), k) and torch.equal(
                v.to(BF16).float(), v
            )
            return k.to(BF16), v.to(BF16)
        return t["k"].to(BF16), t["v"].to(BF16)

    def oracle(self):
        """The bf16 problem over the exact leaves and the ``sm_scale`` with which the bf16 class gives this class's bits under
        the ``unit`` scales: the value scales 1 and the key scale -- the per-tensor one, or ``TH_K`` for every key -- folded
        into ``sm_scale`` (the host's double product, as the class forms its constant)."""
        w = Problem.__new__(Problem)
        w.__dict__.update(self.pr.__dict__)
        w.k_nhd, w.v_nhd = self.exact(self.pool)
        w.k_rag, w.v_rag = self.exact(self.rag)
        ks = TH_K if self.th else self.k_scale
        return w, (1.0 / math.sqrt(self.pr.D)) * ks

    def dequantized(self):
        """The bf16 problem's geometry over the fp32 values the cache stands for (the scales applied): the accuracy yardstick."""
        r = self.rag
        if self.th:
            k = r["k"].float() * r["ksf"][..., None]
            v = r["v"].float() * r["vsf"][..., None]
        elif self.kind == "nvfp4":
            k = nvfp4_dequant(r["k"], r["ksf"]) * self.k_scale
            v = nvfp4_dequant(r["v"], r["vsf"]) * self.v_scale
        else:
            k, v = r["k"].float() * self.k_scale, r["v"].float() * self.v_scale
        w = Problem.__new__(Problem)
        w.__dict__.update(self.pr.__dict__)
        w.k_rag, w.v_rag = k, v
        return w


def last_rows(pr: Problem, q_len: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(
                int(pr.qo_indptr[i + 1]) - q_len,
                int(pr.qo_indptr[i + 1]),
                device=pr.device,
            )
            for i in range(len(pr.q_lens))
        ]
    )


def q_rows(pr: Problem, i: int) -> slice:
    return slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1]))


def kv_rows(pr: Problem, i: int) -> slice:
    return slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1]))


# ------------------------------------------------------------------------------------------------------ device-free
@pytest.mark.parametrize("op", WRAPPERS)
def test_noncausal_fingerprints(op):
    """Every low-bit class has both mask modes: the non-causal fingerprint of a (format, head dim) is its own, shared by the
    three wrappers and the recorded one; the causal fingerprints are where they were, and the decode wrapper's receipt
    without a mode is the causal class's."""
    causal_recorded = {**LOWBIT_FINGERPRINTS, **TOKEN_HEAD}
    for (kind, D), want in NONCAUSAL_FINGERPRINTS.items():
        nc = bi.fingerprint(op, head_dim=D, causal=False, kv_dtype=kind)
        c = bi.fingerprint(op, head_dim=D, causal=True, kv_dtype=kind)
        assert nc == want == spec.spec_fingerprint(D, causal=False, kv_fmt=kind)
        assert c == causal_recorded[(kind, D)] and nc != c
        tag = spec.class_name(kind).split(".")[1]
        assert nc.startswith(f"attn-{tag}-d{D}-nc-") and c.startswith(
            f"attn-{tag}-d{D}-"
        )
        assert nc == bi.fingerprint(
            WRAPPERS[0], head_dim=D, causal=False, kv_dtype=kind
        )
        assert c == bi.fingerprint(WRAPPERS[0], head_dim=D, kv_dtype=kind)
        assert nc != bi.fingerprint(op, head_dim=D, causal=False)
        if kind.endswith("th"):
            assert nc == bi.fingerprint(
                op,
                head_dim=D,
                causal=False,
                kv_dtype=FP8[kind[:4]],
                use_token_head_sf=True,
            )
    values = set(NONCAUSAL_FINGERPRINTS.values())
    assert len(values) == 15 and not values & set(causal_recorded.values())
    with pytest.raises(ValueError):
        spec.spec_fingerprint(128, window_left=8, causal=False, kv_fmt="e4m3")
    if (
        op != WRAPPERS[0]
    ):  # the prefill wrappers' receipt has no default mode, whatever the format
        with pytest.raises(TypeError, match="causal"):
            bi.fingerprint(op, head_dim=128, kv_dtype=E4)


@KIND
def test_noncausal_term_is_the_class_term_over_the_whole_key_range(kind):
    """A non-causal row of a ``kv_len``-key request has the term of the causal row at the request's last position (no
    window), whatever its own position: the mask mode moves the visible range and nothing else of a class -- the leaves, the
    scales per key, the chain. The token-head-scale term names the scales of exactly the request's keys."""
    from flashinfer.batch_invariant.fp_interface.term import sexpr

    s = spec.AttentionSpec(64, T=16, C=32, kv_fmt=kind)
    last = sexpr(s.output(spec.RowSpec(p=40, row=1), 17))
    for p in (0, 5, 40):
        row = spec.RowSpec(p=p, row=1, causal=False, kv_len=41)
        assert sexpr(s.output(row, 17)) == last
    shorter = sexpr(s.output(spec.RowSpec(p=5, row=1, causal=False, kv_len=36), 17))
    assert shorter != last and shorter == sexpr(s.output(spec.RowSpec(p=35, row=1), 17))
    assert sexpr(s.output(spec.RowSpec(p=5, row=1), 17)) != last
    if kind.endswith("th"):
        for leaf in ("sfk", "sfvt"):
            named = [k for k in range(48) if f"({leaf} {k})" in last]
            assert named == list(range(41)), leaf
    with pytest.raises(AssertionError):
        spec.RowSpec(p=5, window_left=3, causal=False, kv_len=41)


def test_noncausal_kernel_identity():
    """The mask mode and the cache format are both in a kernel's name: four objects for one geometry."""
    pytest.importorskip("cutlass")
    from flashinfer.attention.batch_invariant.blocks.compile import kernel_name

    names = {
        kernel_name(128, 4, 16, 4, kv_fmt=fmt, causal=causal)
        for fmt in ("bf16", "e4m3")
        for causal in (True, False)
    }
    assert len(names) == 4
    assert kernel_name(128, 4, 16, 4, kv_fmt="e4m3", causal=False).endswith(
        "_kve4m3_nc"
    )
    assert kernel_name(
        128, 4, 16, 4, kv_fmt="nvfp4", v_sf_swizzled=True, causal=False
    ).endswith("_kvnvfp4_vsfswz_nc")
    assert kernel_name(128, 4, 16, 4, causal=False).endswith("_nc")
    assert "kv" not in kernel_name(128, 4, 16, 4, causal=False)


# ------------------------------------------------------------------------------------------------------ on the device
@needs_device
@HEAD_DIMS
@KIND
def test_noncausal_matches_reference(D, kind):
    """Against fp32 non-causal attention over the dequantized cache (the yardstick for accuracy, never for bits), paged and
    ragged; a request with more query tokens than keys is a member of the classes."""
    c = Case.make(kind, [300, 1000, 64, 40], [700, 2500, 3000, 17], D, 31)
    ref = reference_noncausal(c.dequantized())
    tol = 2e-2 + 1e-2 * ref.abs().max().item()
    assert (c.paged().float() - ref).abs().max().item() < tol
    assert (c.ragged().float() - ref).abs().max().item() < tol


@needs_device
@HEAD_DIMS
@KIND
def test_noncausal_bf16_class_over_the_exact_leaves_is_the_placement_oracle(D, kind):
    """The conversion stage, the per-key scales and the scaled epilogue are placements and leaves that do not depend on the
    mask mode: with the value scales 1 and the key scale folded into ``sm_scale``, every low-bit class under the non-causal
    mode gives, bit for bit, the non-causal bf16 class over its exact leaves -- paged prefill, ragged prefill and multi-token
    decode, both KV layouts. The oracle differs from the causal class's output on the same problem."""
    c = Case.make(kind, [256, 33, 1024, 7], [1500, 2500, 1024, 4096], D, 32)
    w, sm = c.oracle()
    want = bits(bf16_paged_prefill(w, causal=False, sm_scale=sm))
    assert torch.equal(want, bits(bf16_ragged_prefill(w, causal=False, sm_scale=sm)))
    assert torch.equal(bits(c.paged(unit=True)), want)
    assert torch.equal(bits(c.ragged(unit=True)), want)
    assert torch.equal(bits(c.paged("HND", unit=True)), want)
    assert torch.equal(bits(c.ragged("HND", unit=True)), want)
    dec, rows = c.decode(4, unit=True)
    assert torch.equal(bits(dec), want[rows])
    assert not torch.equal(bits(c.paged(causal=True, unit=True)), want)


@needs_device
@HEAD_DIMS
@KIND
def test_noncausal_prefill_decode_paged_ragged_agree(D, kind):
    """R4 with the classes' own scales (per-tensor, block, per-(token, head)): the paged and the ragged wrappers and the
    decode wrapper with ``q_len_per_req = 4`` and ``is_causal=False`` give a token the same bits, in both KV layouts; a second
    run reproduces them (R1); and the position does not enter the non-causal term: the same query at another position of its
    request gives the same bits."""
    c = Case.make(kind, [256, 33, 1024, 7], [1500, 2500, 1024, 4096], D, 33)
    pr = c.pr
    pre = bits(c.paged())
    assert torch.equal(pre, bits(c.paged()))
    assert torch.equal(pre, bits(c.ragged()))
    assert torch.equal(pre, bits(c.paged("HND")))
    assert torch.equal(pre, bits(c.ragged("HND")))
    for layout in ("NHD", "HND"):
        dec, rows = c.decode(4, layout)
        assert torch.equal(bits(dec), pre[rows]), layout
    assert not torch.equal(bits(c.paged(unit=True)), pre)  # the scales are inputs
    q = pr.q.clone()
    for i in range(len(pr.q_lens)):
        q[q_rows(pr, i)] = pr.q[q_rows(pr, i)].flip(0)
    flipped = c.derive(pr.q_lens, pr.kv_lens, q, c.rag, seed=34)
    out = bits(flipped.paged())
    for i in range(len(pr.q_lens)):
        assert torch.equal(out[q_rows(pr, i)], pre[q_rows(pr, i)].flip(0)), i


@needs_device
@KIND
@pytest.mark.parametrize("page", [1, 128])
def test_noncausal_page_sizes(kind, page):
    """The page size is a geometry of the same keys: one-token pages and 128-token pages against the ragged pool."""
    c = Case.make(kind, [300, 64, 2048], [700, 3000, 2048], 128, 35, page=page)
    pre = bits(c.paged())
    assert torch.equal(pre, bits(c.ragged()))
    assert torch.equal(bits(c.paged("HND")), pre)
    dec, rows = c.decode(4)
    assert torch.equal(bits(dec), pre[rows])


@needs_device
@HEAD_DIMS
@KIND
def test_noncausal_batch_permutation_and_subbatch_invariance(D, kind):
    """R2: a request's rows are the same bits in a batch of one, in a permuted batch and in a sub-batch (another page table,
    other junk around the live keys); paged prefill, ragged prefill and multi-token decode."""
    q_lens, kv_lens = [300, 8, 1000, 64, 4], [700, 2500, 2500, 3000, 4]
    c = Case.make(kind, q_lens, kv_lens, D, 36)
    pr = c.pr
    full = bits(c.paged())
    assert torch.equal(full, bits(c.ragged()))
    dec_full, _ = c.decode(4)
    dec_full = bits(dec_full)
    for n, sel in enumerate(([2], [4, 0], [3, 1, 2], [1, 0, 4, 3, 2])):
        sub = c.derive(
            [q_lens[i] for i in sel],
            [kv_lens[i] for i in sel],
            torch.cat([pr.q[q_rows(pr, i)] for i in sel]),
            {
                name: torch.cat([r[kv_rows(pr, i)] for i in sel])
                for name, r in c.rag.items()
            },
            seed=37 + n,
        )
        got_p, got_r = bits(sub.paged()), bits(sub.ragged())
        got_d, _ = sub.decode(4)
        got_d = bits(got_d)
        for j, i in enumerate(sel):
            want = full[q_rows(pr, i)]
            assert torch.equal(got_p[q_rows(sub.pr, j)], want), (sel, i, "paged")
            assert torch.equal(got_r[q_rows(sub.pr, j)], want), (sel, i, "ragged")
            assert torch.equal(got_d[4 * j : 4 * j + 4], dec_full[4 * i : 4 * i + 4]), (
                sel,
                i,
                "decode",
            )


@needs_device
@HEAD_DIMS
@KIND
def test_noncausal_query_chunks_equal_full(D, kind):
    """Chunked prefill under the non-causal mode: a request's query rows run in chunks, each chunk planned over the request's
    FULL key set, give the bits of the one-shot run row for row, paged and ragged. Two requests cut at different points
    (inside a 512-key chunk and at a tile boundary). A chunk over a shorter key set is another term: its bits differ."""
    kv, cut = [1700, 1000], [700, 512]
    c = Case.make(kind, kv, kv, D, 41)
    pr = c.pr
    full = bits(c.paged())
    assert torch.equal(full, bits(c.ragged()))
    for part in range(2):
        take = [
            slice(0, cut[i]) if part == 0 else slice(cut[i], kv[i]) for i in range(2)
        ]
        q_lens = [cut[i] if part == 0 else kv[i] - cut[i] for i in range(2)]
        chunk = c.derive(
            q_lens,
            kv,
            torch.cat([pr.q[q_rows(pr, i)][take[i]] for i in range(2)]),
            c.rag,
            seed=42 + part,
        )
        want = torch.cat([full[q_rows(pr, i)][take[i]] for i in range(2)])
        assert torch.equal(bits(chunk.paged()), want), (part, "paged")
        assert torch.equal(bits(chunk.ragged()), want), (part, "ragged")
    # the first chunk over the keys cached so far (the causal class's chunking) is not the one-shot non-causal run
    short = c.derive(
        cut,
        cut,
        torch.cat([pr.q[q_rows(pr, i)][: cut[i]] for i in range(2)]),
        {
            name: torch.cat([r[kv_rows(pr, i)][: cut[i]] for i in range(2)])
            for name, r in c.rag.items()
        },
        seed=44,
    )
    want = torch.cat([full[q_rows(pr, i)][: cut[i]] for i in range(2)])
    assert not torch.equal(bits(short.paged()), want)


@needs_device
@KIND
def test_noncausal_config_agreement(kind):
    """R3: every member of the low-bit configuration space (ring depth, unit kind, fold placement) and every SM count the
    planner may be told give the wrapper's bits under the non-causal mode; the single-call engine entry is the same class."""
    D = 128
    c = Case.make(kind, [256, 1, 1000, 9], [3000, 2500, 1000, 5000], D, 45)
    pr = c.pr
    ref = bits(c.paged())
    fmt = kind
    scales = c.run_kw(c.pool)

    def run(cfg, **kw):
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
            kv_fmt=fmt,
            causal=False,
            **kw,
        )
        return engine.run_attention(
            planned,
            pr.q,
            c.pool["k"],
            c.pool["v"],
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
            **scales,
        )

    for cfg in engine.all_valid_configs(D, lowbit=True):
        assert torch.equal(bits(run(cfg)), ref), cfg
    for sm in (1, 8, 1000):
        assert torch.equal(bits(run(engine.default_config(D, fmt), sm_count=sm)), ref)
    single = engine.attention_paged(
        pr.q,
        c.pool["k"],
        c.pool["v"],
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        causal=False,
        use_token_head_sf=c.th,
        **scales,
    )
    assert torch.equal(bits(single), ref)


def model_output_noncausal(tp, req, row, head, d, kv_fmt, sm_scale):
    """The specification's value of output element ``(row of request req, query head, d)`` under the non-causal mode, as
    bf16 bits (``test_batch_invariant_attention_token_head_sf.model_output`` with the non-causal row)."""
    pr = tp.pr
    g = pr.Hq // pr.Hkv
    q0, k0 = int(pr.qo_indptr[req]), int(pr.kv_indptr_rag[req])
    kl = pr.kv_lens[req]
    kh = head // g
    model = ExactModel(
        pr.q[q0 + row, head].float().tolist(),
        tp.k_rag[k0 : k0 + kl, kh].float().tolist(),
        tp.v_rag[k0 : k0 + kl, kh].float().tolist(),
        tp.ksf_rag[k0 : k0 + kl, kh].tolist(),
        tp.vsf_rag[k0 : k0 + kl, kh].tolist(),
        spec.scale_log2(sm_scale),
    )
    val = spec.AttentionSpec(pr.D, alg=model, kv_fmt=kv_fmt).output(
        spec.RowSpec(p=row, row=0, causal=False, kv_len=kl), d
    )
    return torch.tensor(val, dtype=torch.float32).to(BF16).view(torch.int16).item()


@needs_device
@pytest.mark.parametrize("kind", ["e4m3th", "e5m2th"])
def test_noncausal_token_head_specification_evaluated_on_the_cpu(kind):
    """The token-head-scale kernels compute the non-causal term: on a problem whose opaque steps are exact, the output equals
    the specification evaluated on the CPU bit for bit -- rows of requests of one tile, of several tiles, of two chunks, and
    of a request with more query tokens than keys; negative, zero and fractional key and value scales; the three wrappers."""
    q_lens, kv_lens = [70, 3, 9, 40], [300, 131, 640, 17]
    tp = exact_problem(q_lens, kv_lens, 8, 4, 64, 16, 501, FP8[kind[:4]])
    c = Case.of_token_head(kind, tp)
    pr = c.pr
    got = bits(c.paged(sm_scale=SM_EXACT))
    assert torch.equal(got, bits(c.ragged("HND", sm_scale=SM_EXACT)))
    dec, rows = c.decode(3, sm_scale=SM_EXACT)
    assert torch.equal(bits(dec), got[rows])
    picks = [  # (request, row, query head, d)
        (0, 0, 0, 0),
        (0, 26, 3, 63),
        (0, 69, 5, 17),
        (1, 0, 7, 1),
        (1, 2, 2, 32),
        (2, 0, 1, 5),
        (2, 8, 6, 2),
        (3, 0, 4, 33),
        (3, 16, 0, 40),
        (3, 39, 3, 9),
    ]
    for req, row, head, d in picks:
        want = model_output_noncausal(tp, req, row, head, d, kind, SM_EXACT)
        at = (req, row, head, d)
        assert got[int(pr.qo_indptr[req]) + row, head, d].item() == want, at


@needs_device
def test_noncausal_nvfp4_swizzled_v_scales_are_the_linear_cache():
    """The V block-scale layout is an address of the same scale bytes under either mask mode."""
    c = Case.make("nvfp4", [300, 64, 9], [700, 3000, 100], 128, 46)
    lin = bits(c.paged())
    swz = dict(c.pool, vsf=swizzle_v_sf(c.pool["vsf"]))
    sw = Case(c.kind, c.pr, swz, c.rag, c.k_scale, c.v_scale)
    assert torch.equal(bits(sw.paged(v_sf_swizzled=True)), lin)
    dec, rows = sw.decode(4, v_sf_swizzled=True)
    assert torch.equal(bits(dec), lin[rows])


@needs_device
@KIND
def test_noncausal_refusals_and_single_token_decode(kind):
    """What the mode refuses for the bf16 class it refuses for every low-bit class, by the same typed error: a window, a
    request with query tokens and no key, a ``forward()`` whose ``causal`` is not the planned one (the plan stays in place).
    A single-token decode step is the causal class whatever ``is_causal`` says."""
    c = Case.make(kind, [64, 3], [700, 90], 128, 47)
    pr = c.pr
    kw = c.plan_kw()
    ws = workspace(pr.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="window_left"):
        w.plan(*args, causal=False, window_left=32, **kw)
    rag = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, "NHD", batch_invariant=True
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="window_left"):
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
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="no key"):
        rag.plan(
            pr.qo_indptr,
            torch.tensor([0, 700, 700], dtype=torch.int32, device=pr.device),
            32,
            8,
            128,
            causal=False,
            **kw,
        )
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="window_left"):
        dec.plan(*args[1:], q_len_per_req=2, is_causal=False, window_left=8, **kw)
    w.plan(*args, causal=False, **kw)
    cache = (c.pool["k"], c.pool["v"])
    scales = c.run_kw(c.pool)
    expected = bits(w.run(pr.q, cache, **scales)).clone()
    per_tensor = {n: x for n, x in scales.items() if n != "kv_cache_sf"}
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="causal"):
        w.forward(pr.q, cache, causal=True, **per_tensor)
    assert torch.equal(bits(w.run(pr.q, cache, **scales)), expected)
    causal = bits(c.paged(causal=True))
    assert not torch.equal(causal, expected)
    # the single-token decode step: one class whatever is_causal says, the causal class's last row -- which is also the
    # non-causal class's (the last token sees every key in both)
    last = last_rows(pr, 1)
    outs = [
        bits(c.decode(1, is_causal=flag)[0]).clone() for flag in (None, True, False)
    ]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    assert torch.equal(outs[0], causal[last]) and torch.equal(outs[0], expected[last])
