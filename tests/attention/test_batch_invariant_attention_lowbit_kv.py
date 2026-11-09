# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8-KV classes of the batch-invariant attention (``attention.e4m3kv`` / ``attention.e5m2kv``): bitwise tests only (the
mode's acceptance criterion), the device-free checks of the fingerprints and the configuration space, and the "unsupported
raises, nothing falls back" rule for what the classes do not cover (per-(token, head) scales, NVFP4 caches, q / o scales).

The caches are quantized per tensor as a serving engine's fp8 KV cache is written (``append_paged_kv_cache`` on an fp8 pool: the
elementwise ``cvt.rn`` of ``x / scale``): the paged pool and the packed (ragged) copy of a problem are quantized with the same
scale, so a key has the same fp8 bytes in both layouts.
"""

import math
from dataclasses import replace

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.attention.batch_invariant import engine, spec
from flashinfer.attention.batch_invariant.wrappers import ATTENTION_ARCHS
from tests.attention.test_batch_invariant_attention import (
    WRAPPERS,
    Problem,
    bits,
    workspace,
)

needs_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bi.is_supported_device(torch.device("cuda"), ATTENTION_ARCHS),
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
FMT_OF = {E4: "e4m3", E5: "e5m2"}
FP8_MAX = {E4: 448.0, E5: 57344.0}


# ------------------------------------------------------------------------------------------------------ problems
class LowbitProblem:
    """``Problem`` with its K / V quantized per tensor to ``dtype``: the paged pool (junk pages included) and the ragged copy
    share the scale, so the live keys have the same bytes in both layouts; ``k_scale`` / ``v_scale`` dequantize them."""

    def __init__(self, pr: Problem, dtype, scale_jitter: float = 1.0):
        self.pr, self.dtype = pr, dtype
        self.k_scale = (
            pr.k_nhd.float().abs().amax().item() / FP8_MAX[dtype] * scale_jitter
        )
        self.v_scale = (
            pr.v_nhd.float().abs().amax().item() / FP8_MAX[dtype] / scale_jitter
        )
        self.k_nhd = (pr.k_nhd.float() / self.k_scale).to(dtype)
        self.v_nhd = (pr.v_nhd.float() / self.v_scale).to(dtype)
        self.k_rag = (pr.k_rag.float() / self.k_scale).to(dtype)
        self.v_rag = (pr.v_rag.float() / self.v_scale).to(dtype)

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

    def widened(self):
        """The caches as the bf16 values the class's leaves are (the exact widening), for the bf16-class oracle."""
        w = Problem.__new__(Problem)
        w.__dict__.update(self.pr.__dict__)
        w.k_nhd, w.v_nhd = self.k_nhd.to(torch.bfloat16), self.v_nhd.to(torch.bfloat16)
        w.k_rag, w.v_rag = self.k_rag.to(torch.bfloat16), self.v_rag.to(torch.bfloat16)
        return w


def problem(q_lens, kv_lens, Hq, Hkv, D, page, seed, dtype, jitter=1.0):
    return LowbitProblem(
        Problem(q_lens, kv_lens, Hq, Hkv, D, page, seed=seed), dtype, jitter
    )


def plan_kw(lp: LowbitProblem):
    return dict(q_data_type=torch.bfloat16, kv_data_type=lp.dtype)


def run_paged_prefill(lp, layout="NHD", window_left=-1, ws=None, scales=True):
    pr = lp.pr
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
        **plan_kw(lp),
    )
    kw = dict(k_scale=lp.k_scale, v_scale=lp.v_scale) if scales else {}
    return w.run(pr.q, lp.cache(layout), **kw)


def run_ragged_prefill(lp, layout="NHD", window_left=-1, ws=None):
    pr = lp.pr
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
        **plan_kw(lp),
    )
    k, v = lp.ragged(layout)
    return w.run(pr.q, k, v, k_scale=lp.k_scale, v_scale=lp.v_scale)


def run_decode(lp, layout="NHD", window_left=-1, ws=None):
    """Decode of the LAST token of every request (q rows picked from the prefill problem)."""
    pr = lp.pr
    last = torch.tensor(
        [int(pr.qo_indptr[i + 1]) - 1 for i in range(len(pr.q_lens))], device=pr.device
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
        window_left=window_left,
        **plan_kw(lp),
    )
    return w.run(
        pr.q[last].contiguous(),
        lp.cache(layout),
        k_scale=lp.k_scale,
        v_scale=lp.v_scale,
    ), last


def reference(lp: LowbitProblem, window_left=-1):
    """fp32 causal attention over the dequantized cache (``fp8 * scale`` exactly in fp32): the yardstick for accuracy, never for
    bits."""
    pr = lp.pr
    outs = []
    tok_q = tok_kv = 0
    g = pr.Hq // pr.Hkv
    k_all = lp.k_rag.float() * lp.k_scale
    v_all = lp.v_rag.float() * lp.v_scale
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        q = pr.q[tok_q : tok_q + ql].float()
        k = k_all[tok_kv : tok_kv + kl].repeat_interleave(g, dim=1)
        v = v_all[tok_kv : tok_kv + kl].repeat_interleave(g, dim=1)
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
def test_fingerprints_per_head_dim_and_kv_format(op):
    """One fingerprint per (head dim, KV format); the bf16 class's digests are the recorded ones and do not see the new dimension."""
    for D in (64, 128, 256):
        bf16 = bi.fingerprint(op, head_dim=D)
        assert (
            bf16
            == spec.FINGERPRINTS[D]
            == bi.fingerprint(op, head_dim=D, kv_dtype=torch.bfloat16)
        )
        e4 = bi.fingerprint(op, head_dim=D, kv_dtype=E4)
        e5 = bi.fingerprint(op, head_dim=D, kv_dtype=E5)
        assert e4.startswith(f"attn-e4m3kv-d{D}-") and e5.startswith(
            f"attn-e5m2kv-d{D}-"
        )
        assert len({bf16, e4, e5}) == 3
        assert (
            e4
            == bi.fingerprint(op, head_dim=D, kv_dtype="e4m3")
            == bi.fingerprint(WRAPPERS[0], head_dim=D, kv_dtype=E4)
        )
    assert (
        len({bi.fingerprint(op, head_dim=D, kv_dtype=E4) for D in (64, 128, 256)}) == 3
    )
    with pytest.raises(ValueError):
        bi.fingerprint(op, head_dim=128, kv_dtype=torch.float16)
    with pytest.raises(ValueError):
        bi.fingerprint(op, head_dim=128, kv_dtype=torch.uint8)
    assert bi.class_name(op) == "attention.bf16"
    assert (
        spec.class_name("e4m3") == "attention.e4m3kv"
        and spec.class_name("e5m2") == "attention.e5m2kv"
    )


def test_spec_terms_of_the_fp8_classes():
    """The fp8-KV term is the bf16 term with every K / V leaf widened and the value scale in the epilogue; the two formats differ."""
    from flashinfer.batch_invariant.fp_interface.term import sexpr

    row = spec.RowSpec(p=40, window_left=-1, row=0)
    bf, e4, e5 = (
        sexpr(spec.AttentionSpec(64, T=16, C=32, kv_fmt=f).output(row, 0))
        for f in ("bf16", "e4m3", "e5m2")
    )
    assert "widen" not in bf and "v_scale" not in bf
    assert (
        "(widen e4m3 (b " in e4
        and "(widen e4m3 (op:v_slice " in e4
        and "(sym v_scale)" in e4
    )
    assert e4.replace("e4m3", "e5m2") == e5 and e4 != e5
    # the bf16 term is the fp8 term with the widenings erased and the value scale removed
    stripped = (
        e4.replace("(widen e4m3 ", "")
        .replace("(sym v_scale)", "")
        .replace("(mul  ", "(mul ")
    )
    assert len(stripped) < len(e4)
    # the key scale is not a term node: it is inside the one constant c (formed once on the host)
    assert spec.scale_log2(0.5, 2.0) == spec.scale_log2(1.0)
    assert spec.scale_log2(1.0 / 8, 3.0) == spec.to_f32(3.0 / 8 / math.log(2.0))


def test_lowbit_configuration_space_and_kernel_identity():
    """The fp8-KV kernels' space: one softmax warpgroup, the row body for small units; their names carry the format."""
    from flashinfer.attention.batch_invariant.blocks.compile import kernel_name

    for D, n in ((64, 63), (128, 27), (256, 9)):
        space = engine.all_valid_configs(D, lowbit=True)
        assert len(space) == n and all(c.well_formed(D, lowbit=True) for c in space)
        assert all(c.softmax_wgs == 1 and c.small == "rep" and not c.q2 for c in space)
        assert all(c in engine.all_valid_configs(D) for c in space)
    assert not engine.default_config(128).well_formed(128, lowbit=True)
    assert engine.default_config(128, "e4m3").well_formed(128, lowbit=True)
    assert engine.default_config(128, "e4m3").small == "rep"
    a = kernel_name(128, 4, 16, 4)
    b = kernel_name(128, 4, 16, 4, kv_fmt="e4m3")
    c = kernel_name(128, 4, 16, 4, kv_fmt="e5m2")
    assert (
        a != b != c
        and b.endswith("_kve4m3")
        and c.endswith("_kve5m2")
        and "kv" not in a
    )


def test_cache_key_covers_the_conversion_stage():
    from flashinfer.batch_invariant.jit import relative_sources

    assert "attention/batch_invariant/blocks/convert.py" in relative_sources(
        "batch_invariant_attention"
    )


# ------------------------------------------------------------------------------------------------------ on the device
@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [E4, E5], ids=["e4m3", "e5m2"])
def test_matches_reference(D, dtype):
    lp = problem([300, 1000, 64], [700, 2500, 3000], 32, 8, D, 16, 1, dtype)
    out = run_paged_prefill(lp)
    ref = reference(lp)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [E4, E5], ids=["e4m3", "e5m2"])
def test_bf16_class_over_the_widened_cache_is_the_placement_oracle(D, dtype):
    """The conversion stage is a placement of the exact ``widen`` leaves: with ``v_scale = 1`` (``mul(x, 1)`` is the identity) the
    fp8-KV class gives, bit for bit, the bf16 class run over the exactly widened cache with ``k_scale`` folded into ``sm_scale`` --
    prefill (paged and ragged) and decode. A nontrivial ``k_scale`` is folded on both sides the same way (the host's double product)."""
    lp = problem(
        [256, 1, 1000, 33], [1500, 2500, 1000, 4096], 32, 8, D, 16, 2, dtype, jitter=1.7
    )
    pr, w = lp.pr, lp.widened()
    sm = (1.0 / math.sqrt(D)) * lp.k_scale
    wp = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    wp.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        D,
        16,
        causal=True,
        **plan_kw(lp),
    )
    got = wp.run(pr.q, lp.cache("NHD"), k_scale=lp.k_scale, v_scale=1.0)
    wb = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    wb.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        D,
        16,
        causal=True,
        sm_scale=sm,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    want = wb.run(pr.q, w.cache("NHD"))
    assert torch.equal(bits(got), bits(want))
    # a tensor-valued k_scale (as the wrappers accept) is the same input
    got_t = wp.run(
        pr.q,
        lp.cache("NHD"),
        k_scale=torch.tensor([lp.k_scale], dtype=torch.float32, device=pr.device),
        v_scale=1.0,
    )
    assert torch.equal(bits(got_t), bits(want))
    wr = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    wr.plan(
        pr.qo_indptr, pr.kv_indptr_rag, pr.Hq, pr.Hkv, D, causal=True, **plan_kw(lp)
    )
    assert torch.equal(
        bits(wr.run(pr.q, *lp.ragged("NHD"), k_scale=lp.k_scale, v_scale=1.0)),
        bits(want),
    )
    last = torch.tensor(
        [int(pr.qo_indptr[i + 1]) - 1 for i in range(4)], device=pr.device
    )
    wd = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    wd.plan(
        pr.kv_indptr, pr.kv_indices, pr.kv_last, pr.Hq, pr.Hkv, D, 16, **plan_kw(lp)
    )
    dec = wd.run(
        pr.q[last].contiguous(), lp.cache("NHD"), k_scale=lp.k_scale, v_scale=1.0
    )
    assert torch.equal(bits(dec), bits(want[last]))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_prefill_equals_decode(D, layout):
    """R4: the last token of every request gives the same bits as a prefill row and as a decode step (both scales nontrivial)."""
    lp = problem(
        [256, 1, 1024, 33], [1500, 2500, 1024, 4096], 32, 8, D, 16, 2, E4, jitter=1.3
    )
    pre = run_paged_prefill(lp, layout)
    dec, last = run_decode(lp, layout)
    assert torch.equal(bits(pre[last]), bits(dec))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("page", [1, 16, 128])
def test_paged_equals_ragged(D, page):
    """R4: the paged and the ragged fp8 caches hand the kernel the same bytes at the same positions: the same bits."""
    lp = problem([300, 1000, 64, 2048], [700, 2500, 3000, 2048], 32, 8, D, page, 3, E5)
    assert torch.equal(bits(run_paged_prefill(lp)), bits(run_ragged_prefill(lp)))
    assert torch.equal(
        bits(run_paged_prefill(lp, "HND")), bits(run_ragged_prefill(lp, "HND"))
    )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_batch_permutation_and_subbatch_invariance(D):
    """R2: a request's bits do not depend on its neighbours, its position in the batch or the batch size."""
    q_lens, kv_lens = [300, 1000, 64, 2048, 1], [700, 2500, 3000, 2048, 5000]
    lp = problem(q_lens, kv_lens, 32, 8, D, 16, 5, E4)
    pr = lp.pr
    full = bits(run_paged_prefill(lp))
    ref_rag = bits(run_ragged_prefill(lp))
    assert torch.equal(ref_rag, full)
    rows = [
        slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1]))
        for i in range(len(q_lens))
    ]
    kv_rows = [
        slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1]))
        for i in range(len(q_lens))
    ]

    def sub(sel):
        s = LowbitProblem.__new__(LowbitProblem)
        s.pr = Problem(
            [q_lens[i] for i in sel], [kv_lens[i] for i in sel], 32, 8, D, 16, seed=5
        )
        s.pr.q = torch.cat([pr.q[rows[i]] for i in sel]).contiguous()
        s.dtype, s.k_scale, s.v_scale = lp.dtype, lp.k_scale, lp.v_scale
        s.k_rag = torch.cat([lp.k_rag[kv_rows[i]] for i in sel]).contiguous()
        s.v_rag = torch.cat([lp.v_rag[kv_rows[i]] for i in sel]).contiguous()
        return s

    for i in (0, 2, 4):
        assert torch.equal(bits(run_ragged_prefill(sub([i]))), full[rows[i]]), i
    perm = [3, 0, 4, 1, 2]
    permuted = bits(run_ragged_prefill(sub(perm)))
    off = 0
    for i in perm:
        n = q_lens[i]
        assert torch.equal(permuted[off : off + n], full[rows[i]]), i
        off += n


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_decode_batch_invariance(D):
    """R2 for decode: a request's step gives the same bits alone and in a batch of eight, whatever pages the pool assigns."""
    kv = [1024, 4096, 33, 700, 2500, 8192, 1, 1500]
    lp = problem([1] * 8, kv, 32, 8, D, 16, 6, E4)
    pr = lp.pr
    out8, _ = run_decode(lp)
    for i in (0, 1, 5, 6):
        one = LowbitProblem.__new__(LowbitProblem)
        one.pr = Problem(
            [1], [kv[i]], 32, 8, D, 16, seed=100 + i
        )  # another pool and page order
        one.pr.q = pr.q[i : i + 1].clone()
        one.dtype, one.k_scale, one.v_scale = lp.dtype, lp.k_scale, lp.v_scale
        sl = slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1]))
        one.k_rag, one.v_rag = lp.k_rag[sl].clone(), lp.v_rag[sl].clone()
        # the request's keys in the fresh pool's pages (the junk of the other pages stays the fresh pool's random bytes)
        k_pool = (one.pr.k_nhd.float() / lp.k_scale).to(lp.dtype).clone()
        v_pool = (one.pr.v_nhd.float() / lp.v_scale).to(lp.dtype).clone()
        n_pages = int(one.pr.kv_indptr[1])
        for j in range(n_pages):
            pg = int(one.pr.kv_indices[j])
            n = min(16, kv[i] - j * 16)
            k_pool[pg, :n] = one.k_rag[j * 16 : j * 16 + n]
            v_pool[pg, :n] = one.v_rag[j * 16 : j * 16 + n]
        one.k_nhd, one.v_nhd = k_pool, v_pool
        out1, _ = run_decode(one)
        assert torch.equal(bits(out1), bits(out8[i : i + 1])), i


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [E4, E5], ids=["e4m3", "e5m2"])
def test_config_agreement(D, dtype):
    """R3: every member of the fp8-KV classes' configuration space (ring depth, unit kind, fold placement) gives the wrapper's bits."""
    lp = problem(
        [256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, 8, dtype, jitter=1.3
    )
    pr = lp.pr
    ref = bits(run_paged_prefill(lp))
    fmt = FMT_OF[dtype]
    for cfg in engine.all_valid_configs(D, lowbit=True):
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
        )
        out = engine.run_attention(
            planned,
            pr.q,
            lp.k_nhd,
            lp.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
            k_scale=lp.k_scale,
            v_scale=lp.v_scale,
        )
        assert torch.equal(bits(out), ref), cfg
    for sm in (1, 8, 1000):
        cfg = engine.default_config(D, fmt)
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
            sm_count=sm,
            kv_fmt=fmt,
        )
        out = engine.run_attention(
            planned,
            pr.q,
            lp.k_nhd,
            lp.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
            k_scale=lp.k_scale,
            v_scale=lp.v_scale,
        )
        assert torch.equal(bits(out), ref), sm


@needs_device
def test_scales_are_inputs_of_the_chain():
    """The value scale multiplies once before the rounding and the key scale enters through ``c``: changing either changes the bits;
    a Python float and a 1-element tensor are the same input; the same scale through the three wrappers gives the same bits."""
    lp = problem([64, 1], [700, 2500], 8, 2, 128, 16, 20, E4)
    pr = lp.pr
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        8,
        2,
        128,
        16,
        causal=True,
        **plan_kw(lp),
    )
    base = bits(
        w.run(pr.q, lp.cache("NHD"), k_scale=lp.k_scale, v_scale=lp.v_scale)
    ).clone()
    assert not torch.equal(
        base,
        bits(w.run(pr.q, lp.cache("NHD"), k_scale=lp.k_scale, v_scale=lp.v_scale * 2)),
    )
    assert not torch.equal(
        base,
        bits(w.run(pr.q, lp.cache("NHD"), k_scale=lp.k_scale * 2, v_scale=lp.v_scale)),
    )
    dev = pr.device
    as_t = w.run(
        pr.q,
        lp.cache("NHD"),
        k_scale=torch.tensor(lp.k_scale, device=dev),
        v_scale=torch.tensor([lp.v_scale], device=dev),
    )
    assert torch.equal(base, bits(as_t))
    # v_scale = 2^k is exact on the fp32 row: doubling the scale doubles every output (bf16 rounding commutes with a power of two)
    twice = w.run(pr.q, lp.cache("NHD"), k_scale=lp.k_scale, v_scale=lp.v_scale * 2)
    assert torch.equal(
        bits(twice), bits((base.view(torch.bfloat16).float() * 2).to(torch.bfloat16))
    )
    # None is 1.0
    assert torch.equal(
        bits(w.run(pr.q, lp.cache("NHD"))),
        bits(w.run(pr.q, lp.cache("NHD"), k_scale=1.0, v_scale=1.0)),
    )


@needs_device
def test_sliding_window_prefill_equals_decode():
    lp = problem([256, 1, 2000], [1500, 2500, 5000], 32, 8, 128, 16, 9, E5)
    pre = run_paged_prefill(lp, window_left=600)
    dec, last = run_decode(lp, window_left=600)
    assert torch.equal(bits(pre[last]), bits(dec))
    assert torch.equal(bits(pre), bits(run_ragged_prefill(lp, window_left=600)))


@needs_device
def test_determinism_and_graph_replay():
    lp = problem([512, 7, 1000], [4096, 1025, 1000], 32, 8, 128, 16, 4, E4)
    pr = lp.pr
    ref = bits(run_paged_prefill(lp)).clone()
    for _ in range(2):
        assert torch.equal(bits(run_paged_prefill(lp)), ref)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
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
        **plan_kw(lp),
    )
    out = torch.empty_like(pr.q)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        w.run(pr.q, lp.cache("NHD"), out=out, k_scale=lp.k_scale, v_scale=lp.v_scale)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            w.run(
                pr.q, lp.cache("NHD"), out=out, k_scale=lp.k_scale, v_scale=lp.v_scale
            )
    torch.cuda.synchronize()
    for _ in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)


@needs_device
def test_unsupported_inputs_raise_and_nothing_falls_back():
    lp = problem([64], [700], 32, 8, 128, 16, 10, E4)
    pr = lp.pr
    ws = workspace(pr.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    plan_args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    # plan(): what the classes do not cover
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="use_token_head_sf"
    ):  # per-(token, head) scales
        w.plan(*plan_args, causal=True, use_token_head_sf=True, **plan_kw(lp))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="NVFP4"
    ):  # an NVFP4 cache
        w.plan(
            *plan_args,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
        )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # fp16 queries over an fp8 cache
        w.plan(*plan_args, causal=True, q_data_type=torch.float16, kv_data_type=E4)
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # an fp8 query
        w.plan(*plan_args, causal=True, q_data_type=E4, kv_data_type=E4)
    # run(): the planned class is e4m3
    w.plan(*plan_args, causal=True, **plan_kw(lp))
    k, v = lp.cache("NHD")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # the bf16 cache of the same problem
        w.run(pr.q, pr.cache("NHD"), k_scale=lp.k_scale)
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # the other fp8 format
        w.run(pr.q, (k.view(E5), v.view(E5)), k_scale=lp.k_scale)
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # q_scale
        w.run(pr.q, (k, v), q_scale=1.0, k_scale=lp.k_scale)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="kv_cache_sf"
    ):  # block / per-token scales
        w.run(
            pr.q,
            (k, v),
            kv_cache_sf=(
                torch.ones(1, device=pr.device),
                torch.ones(1, device=pr.device),
            ),
        )
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            w.run(pr.q, (k, v), k_scale=bad)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        w.run(pr.q, (k, v), v_scale=float("inf"))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a scale tensor of more than one element
        w.run(pr.q, (k, v), k_scale=torch.ones(2, device=pr.device))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a 16-byte misaligned fp8 cache
        off = torch.empty(k.numel() + 8, dtype=E4, device=pr.device)[8:].view(k.shape)
        w.run(pr.q, (off, v))
    ref = bits(w.run(pr.q, (k, v), k_scale=lp.k_scale, v_scale=lp.v_scale))
    assert torch.equal(
        bits(w.run(pr.q, (k, v), k_scale=lp.k_scale, v_scale=lp.v_scale)), ref
    )  # the plan survived
    # the bf16 class refuses the scales (its chain has none)
    wb = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    wb.plan(
        *plan_args, causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        wb.run(pr.q, pr.cache("NHD"), k_scale=1.0)
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        wb.run(pr.q, pr.cache("NHD"), v_scale=1.0)
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # an fp8 cache over a bf16 plan
        wb.run(pr.q, (k, v))
    # the functional API: a plan of one format cannot run another; scales with a bf16 cache raise
    planned = engine.plan_attention(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_last,
        32,
        8,
        128,
        16,
        -1,
        None,
        None,
        pr.device,
        kv_fmt="e4m3",
    )
    with pytest.raises(ValueError):
        engine.run_attention(
            planned,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
        )
    with pytest.raises(ValueError):
        engine.run_attention(
            planned,
            pr.q,
            k,
            v.view(E5),
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
        )
    bplan = engine.plan_attention(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_last,
        32,
        8,
        128,
        16,
        -1,
        None,
        None,
        pr.device,
    )
    with pytest.raises(ValueError):
        engine.run_attention(
            bplan,
            pr.q,
            pr.k_nhd,
            pr.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
            k_scale=2.0,
        )
    with pytest.raises(
        AssertionError
    ):  # a bf16-class configuration (the Swaps body) is not a member of the fp8-KV space
        engine.plan_attention(
            pr.qo_indptr,
            pr.kv_indptr,
            pr.kv_last,
            32,
            8,
            128,
            16,
            -1,
            replace(engine.default_config(128), small="swap"),
            None,
            pr.device,
            kv_fmt="e4m3",
        )
    # decode and ragged take the same plan-time refusals
    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="use_token_head_sf"):
        dec.plan(
            pr.kv_indptr,
            pr.kv_indices,
            pr.kv_last,
            32,
            8,
            128,
            16,
            use_token_head_sf=True,
            **plan_kw(lp),
        )
    rag = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        ws, "NHD", batch_invariant=True
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="NVFP4"):
        rag.plan(
            pr.qo_indptr,
            pr.kv_indptr_rag,
            32,
            8,
            128,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
        )
    assert w.resolved_backend == "batch_invariant"
