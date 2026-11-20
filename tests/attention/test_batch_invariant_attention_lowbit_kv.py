# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8-KV classes of the batch-invariant attention (``attention.e4m3kv`` / ``attention.e5m2kv``): bitwise tests only (the
mode's acceptance criterion), the device-free checks of the fingerprints and the configuration space, and the "unsupported
raises, nothing falls back" rule for what the classes do not cover (per-(token, head) scales, scaled queries and outputs); then
the NVFP4-KV class (``attention.nvfp4kv``) with its two V block-scale layouts, every code of the formats, and the argument lists
serving engines call the wrappers with.

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
from tests.attention.test_batch_invariant_attention import (
    run_paged_prefill as bf16_paged_prefill,
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
        # binary32 scale values (as a serving engine's fp32 scale tensors hold them): a Python float that is a binary32 value and a
        # 1-element fp32 tensor are then the same input of the class
        self.k_scale = spec.to_f32(
            pr.k_nhd.float().abs().amax().item() / FP8_MAX[dtype] * scale_jitter
        )
        self.v_scale = spec.to_f32(
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
        bi.fingerprint(op, head_dim=128, kv_dtype=torch.int8)
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
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # an fp16 cache
        w.plan(
            *plan_args,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float16,
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
    for bad in (0.5, 2.0, torch.tensor([0.5], device=pr.device), "1.0"):
        with pytest.raises(
            bi.BatchInvariantUnsupportedError, match="q_scale"
        ):  # a scaled query
            w.run(pr.q, (k, v), q_scale=bad, k_scale=lp.k_scale, v_scale=lp.v_scale)
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
    # a key scale beyond binary32, and one whose constant c = fp32(sm_scale * k_scale * log2 e) is subnormal
    for bad in (1e39, 1e-44, torch.tensor(1e-44, dtype=torch.float64)):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="k_scale"):
            w.run(pr.q, (k, v), k_scale=bad)
    # a value scale that is not finite, or is zero in binary32
    for bad in (float("inf"), float("nan"), 1e39, 0.0, 1e-50):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="v_scale"):
            w.run(pr.q, (k, v), k_scale=lp.k_scale, v_scale=bad)
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
    # q_scale of 1.0 (a number or a 1-element tensor, on the host or the device) is the unscaled query: the bits of no q_scale
    for unit in (1.0, 1, torch.ones(1), torch.tensor(1.0, device=pr.device)):
        got = w.run(pr.q, (k, v), q_scale=unit, k_scale=lp.k_scale, v_scale=lp.v_scale)
        assert torch.equal(bits(got), ref)
    # the bf16 class has no scale in its chain: every scale is accepted at None / 1.0 (the same bits) and refused otherwise
    wb = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    wb.plan(
        *plan_args, causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16
    )
    bref = bits(wb.run(pr.q, pr.cache("NHD"))).clone()
    for name in ("q_scale", "k_scale", "v_scale"):
        for bad in (2.0, 0.999, torch.tensor([2.0])):
            with pytest.raises(bi.BatchInvariantUnsupportedError, match=name):
                wb.run(pr.q, pr.cache("NHD"), **{name: bad})
    unit = torch.ones(1, device=pr.device)
    assert torch.equal(
        bits(wb.run(pr.q, pr.cache("NHD"), q_scale=1.0, k_scale=1.0, v_scale=unit)),
        bref,
    )
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # an fp8 cache over a bf16 plan
        wb.run(pr.q, (k, v))
    # an sm_scale whose constant is subnormal in binary32 is refused at plan(), for every class
    for kv in (torch.bfloat16, E4):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="sm_scale"):
            wb.plan(
                *plan_args,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=kv,
                sm_scale=1e-40,
            )
    assert torch.equal(bits(wb.run(pr.q, pr.cache("NHD"))), bref)  # the plan survived
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
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="use_token_head_sf"):
        rag.plan(
            pr.qo_indptr,
            pr.kv_indptr_rag,
            32,
            8,
            128,
            causal=True,
            use_token_head_sf=True,
            q_data_type=torch.bfloat16,
            kv_data_type=E4,
        )


# ------------------------------------------------------------------------------------------------------ NVFP4
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
)


def nvfp4_dequant(x: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """The class's ``dq4`` leaves in fp32: the e2m1 nibbles (the low nibble is the even element) times the e4m3 block scale of
    their 16-element block; every product is exact (and a bf16 value)."""
    lut = E2M1_VALUES.to(x.device)
    lo, hi = lut[(x & 0xF).long()], lut[(x >> 4).long()]
    vals = torch.stack([lo, hi], dim=-1).reshape(*x.shape[:-1], x.shape[-1] * 2)
    scales = sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return vals * scales


class Fp4Problem:
    """``Problem``'s geometry with random NVFP4 pools (``tests.quantization.test_helpers.utils_fp4.create_nvfp4_kv``: packed nibbles and e4m3
    block scales of the cache's geometry, as FlashInfer's NVFP4 append kernel lays them out) and per-tensor scales; the ragged
    copies are gathered through the page table (the same bytes)."""

    def __init__(self, pr: Problem, k_scale=0.75, v_scale=1.5, seed=0):
        from tests.quantization.test_helpers.utils_fp4 import create_nvfp4_kv

        self.pr, self.k_scale, self.v_scale = pr, k_scale, v_scale
        torch.manual_seed(seed)
        D, dev = pr.D, pr.device
        self.k_nhd, self.k_sf, _ = create_nvfp4_kv(pr.k_nhd.shape[:-1] + (D // 2,), dev)
        self.v_nhd, self.v_sf, _ = create_nvfp4_kv(pr.v_nhd.shape[:-1] + (D // 2,), dev)
        self.k_rag, self.v_rag = self._gather(self.k_nhd), self._gather(self.v_nhd)
        self.k_sf_rag, self.v_sf_rag = self._gather(self.k_sf), self._gather(self.v_sf)

    def _gather(self, pool):
        pr = self.pr
        outs = []
        for r in range(len(pr.kv_lens)):
            pages = pr.kv_indices[int(pr.kv_indptr[r]) : int(pr.kv_indptr[r + 1])]
            outs.append(
                pool[pages].reshape(-1, pr.Hkv, pool.shape[-1])[: pr.kv_lens[r]]
            )
        return torch.cat(outs).contiguous()

    def cache(self, layout):
        if layout == "NHD":
            return self.k_nhd, self.v_nhd
        return self.k_nhd.transpose(1, 2).contiguous(), self.v_nhd.transpose(
            1, 2
        ).contiguous()

    def sf(self, layout):
        if layout == "NHD":
            return self.k_sf, self.v_sf
        return self.k_sf.transpose(1, 2).contiguous(), self.v_sf.transpose(
            1, 2
        ).contiguous()

    def ragged(self, layout):
        if layout == "NHD":
            return self.k_rag, self.v_rag
        return self.k_rag.transpose(0, 1).contiguous(), self.v_rag.transpose(
            0, 1
        ).contiguous()

    def sf_ragged(self, layout):
        if layout == "NHD":
            return self.k_sf_rag, self.v_sf_rag
        return self.k_sf_rag.transpose(0, 1).contiguous(), self.v_sf_rag.transpose(
            0, 1
        ).contiguous()

    def dq_rag(self):
        return nvfp4_dequant(self.k_rag, self.k_sf_rag), nvfp4_dequant(
            self.v_rag, self.v_sf_rag
        )


def fp4_problem(q_lens, kv_lens, Hq, Hkv, D, page, seed):
    return Fp4Problem(Problem(q_lens, kv_lens, Hq, Hkv, D, page, seed=seed), seed=seed)


def fp4_run_paged_prefill(fp, layout="NHD", window_left=-1, sf=None, **plan_kw):
    """``sf``: the ``kv_cache_sf`` to pass instead of the problem's linear scales; ``plan_kw``: extra ``plan()`` keywords."""
    pr = fp.pr
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
        causal=True,
        window_left=window_left,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        **plan_kw,
    )
    return w.run(
        pr.q,
        fp.cache(layout),
        kv_cache_sf=fp.sf(layout) if sf is None else sf,
        k_scale=fp.k_scale,
        v_scale=fp.v_scale,
    )


def fp4_run_ragged_prefill(fp, layout="NHD", window_left=-1):
    pr = fp.pr
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace(pr.device), layout, batch_invariant=True
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
        kv_data_type=torch.uint8,
    )
    k, v = fp.ragged(layout)
    return w.run(
        pr.q,
        k,
        v,
        kv_cache_sf=fp.sf_ragged(layout),
        k_scale=fp.k_scale,
        v_scale=fp.v_scale,
    )


def fp4_run_decode(fp, layout="NHD", window_left=-1, sf=None, **plan_kw):
    pr = fp.pr
    last = torch.tensor(
        [int(pr.qo_indptr[i + 1]) - 1 for i in range(len(pr.q_lens))], device=pr.device
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
        kv_data_type=torch.uint8,
        **plan_kw,
    )
    return w.run(
        pr.q[last].contiguous(),
        fp.cache(layout),
        kv_cache_sf=fp.sf(layout) if sf is None else sf,
        k_scale=fp.k_scale,
        v_scale=fp.v_scale,
    ), last


@pytest.mark.parametrize("op", WRAPPERS)
def test_nvfp4_fingerprint(op):
    for D in (64, 128, 256):
        f4 = bi.fingerprint(op, head_dim=D, kv_dtype=torch.uint8)
        assert f4.startswith(f"attn-nvfp4kv-d{D}-") and f4 == bi.fingerprint(
            op, head_dim=D, kv_dtype="nvfp4"
        )
        assert f4 != bi.fingerprint(
            op, head_dim=D, kv_dtype=E4
        ) and f4 != bi.fingerprint(op, head_dim=D)
    assert spec.class_name("nvfp4") == "attention.nvfp4kv"
    from flashinfer.batch_invariant.fp_interface.term import sexpr

    sx = sexpr(
        spec.AttentionSpec(64, T=16, C=32, kv_fmt="nvfp4").output(
            spec.RowSpec(p=40, row=0), 17
        )
    )
    assert (
        "(dq4 (b " in sx
        and "(sfb " in sx
        and "(dq4 (op:v_slice " in sx
        and "(sfv " in sx
        and "widen" not in sx
    )
    assert all(
        c.well_formed(D, lowbit=True)
        for D in (64, 128, 256)
        for c in engine.all_valid_configs(D, lowbit=True)
    )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_nvfp4_matches_reference_and_the_bf16_class_oracle(D):
    """The NVFP4-KV class against the fp32 reference over the dequantized cache, and -- the dq4 leaves being exact bf16 values --
    bit for bit against the bf16 class over those values with ``k_scale`` folded and ``v_scale = 1``."""
    fp = fp4_problem([300, 1, 1000, 64], [700, 2500, 1000, 3000], 32, 8, D, 16, 1)
    pr = fp.pr
    out = fp4_run_paged_prefill(fp)
    k_dq, v_dq = fp.dq_rag()
    assert torch.equal(k_dq.to(torch.bfloat16).float(), k_dq) and torch.equal(
        v_dq.to(torch.bfloat16).float(), v_dq
    )
    lp = LowbitProblem.__new__(LowbitProblem)
    lp.pr, lp.k_scale, lp.v_scale = pr, fp.k_scale, fp.v_scale
    lp.k_rag, lp.v_rag = (
        k_dq,
        v_dq,
    )  # fp32 values: reference() multiplies by the per-tensor scales
    ref = reference(lp)
    err = (out.float() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err
    w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    w.plan(
        pr.qo_indptr,
        pr.kv_indptr_rag,
        pr.Hq,
        pr.Hkv,
        D,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
    )
    got = w.run(
        pr.q,
        fp.k_rag,
        fp.v_rag,
        kv_cache_sf=(fp.k_sf_rag, fp.v_sf_rag),
        k_scale=fp.k_scale,
        v_scale=1.0,
    )
    wb = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    wb.plan(
        pr.qo_indptr,
        pr.kv_indptr_rag,
        pr.Hq,
        pr.Hkv,
        D,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=(1.0 / math.sqrt(D)) * fp.k_scale,
    )
    want = wb.run(
        pr.q, k_dq.to(torch.bfloat16).contiguous(), v_dq.to(torch.bfloat16).contiguous()
    )
    assert torch.equal(bits(got), bits(want))


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_nvfp4_prefill_decode_paged_ragged_agree(D, layout):
    """R4 for the NVFP4-KV class: prefill rows == decode rows, paged == ragged, in both cache layouts; the 5-D kv_cache_sf form
    is the tuple form."""
    fp = fp4_problem([256, 1, 1024, 33], [1500, 2500, 1024, 4096], 32, 8, D, 16, 2)
    pre = fp4_run_paged_prefill(fp, layout)
    dec, last = fp4_run_decode(fp, layout)
    assert torch.equal(bits(pre[last]), bits(dec))
    assert torch.equal(bits(pre), bits(fp4_run_ragged_prefill(fp, layout)))
    pr = fp.pr
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
        D,
        16,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
    )
    k_sf, v_sf = fp.sf(layout)
    five = torch.stack([k_sf, v_sf], dim=1)
    assert torch.equal(
        bits(
            w.run(
                pr.q,
                fp.cache(layout),
                kv_cache_sf=five,
                k_scale=fp.k_scale,
                v_scale=fp.v_scale,
            )
        ),
        bits(pre),
    )
    # e4m3-typed scale tensors are the same bytes
    e4 = (k_sf.view(torch.float8_e4m3fn), v_sf.view(torch.float8_e4m3fn))
    assert torch.equal(
        bits(
            w.run(
                pr.q,
                fp.cache(layout),
                kv_cache_sf=e4,
                k_scale=fp.k_scale,
                v_scale=fp.v_scale,
            )
        ),
        bits(pre),
    )


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_nvfp4_config_agreement_and_batch_invariance(D):
    fp = fp4_problem([256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, 8)
    pr = fp.pr
    ref = bits(fp4_run_paged_prefill(fp))
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
            kv_fmt="nvfp4",
        )
        out = engine.run_attention(
            planned,
            pr.q,
            fp.k_nhd,
            fp.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
            k_scale=fp.k_scale,
            v_scale=fp.v_scale,
            kv_cache_sf=(fp.k_sf, fp.v_sf),
        )
        assert torch.equal(bits(out), ref), cfg
    # a request alone (ragged) gives its rows of the batch
    rows = [slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])) for i in range(4)]
    kv_rows = [
        slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1])) for i in range(4)
    ]
    full = bits(fp4_run_ragged_prefill(fp))
    assert torch.equal(full, ref)
    for i in (0, 3):
        one = Fp4Problem.__new__(Fp4Problem)
        one.pr = Problem([pr.q_lens[i]], [pr.kv_lens[i]], 32, 8, D, 16, seed=30 + i)
        one.pr.q = pr.q[rows[i]].clone()
        one.k_scale, one.v_scale = fp.k_scale, fp.v_scale
        one.k_rag, one.v_rag = (
            fp.k_rag[kv_rows[i]].clone(),
            fp.v_rag[kv_rows[i]].clone(),
        )
        one.k_sf_rag, one.v_sf_rag = (
            fp.k_sf_rag[kv_rows[i]].clone(),
            fp.v_sf_rag[kv_rows[i]].clone(),
        )
        assert torch.equal(bits(fp4_run_ragged_prefill(one)), full[rows[i]]), i


@needs_device
def test_nvfp4_refusals():
    fp = fp4_problem([64], [700], 32, 8, 128, 16, 10)
    pr = fp.pr
    ws = workspace(pr.device)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    plan_args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="use_token_head_sf"):
        w.plan(
            *plan_args,
            causal=True,
            use_token_head_sf=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
        )
    w.plan(
        *plan_args, causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.uint8
    )
    k, v = fp.cache("NHD")
    with pytest.raises(
        bi.BatchInvariantUnsupportedError, match="kv_cache_sf"
    ):  # no block scales
        w.run(pr.q, (k, v), k_scale=fp.k_scale)
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # scales of the wrong width
        w.run(pr.q, (k, v), kv_cache_sf=(fp.k_sf[..., :4], fp.v_sf[..., :4]))
    with pytest.raises(bi.BatchInvariantUnsupportedError):  # scales of the wrong dtype
        w.run(pr.q, (k, v), kv_cache_sf=(fp.k_sf.float(), fp.v_sf.float()))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # one scale tensor of the tuple missing
        w.run(pr.q, (k, v), kv_cache_sf=(fp.k_sf,))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # an fp8 cache over the NVFP4 plan
        w.run(pr.q, (k.view(E4), v.view(E4)), kv_cache_sf=(fp.k_sf, fp.v_sf))
    with pytest.raises(
        bi.BatchInvariantUnsupportedError
    ):  # a bf16 cache of the full width
        w.run(pr.q, pr.cache("NHD"), kv_cache_sf=(fp.k_sf, fp.v_sf))
    ref = bits(
        w.run(
            pr.q,
            (k, v),
            kv_cache_sf=(fp.k_sf, fp.v_sf),
            k_scale=fp.k_scale,
            v_scale=fp.v_scale,
        )
    )
    assert torch.equal(
        bits(
            w.run(
                pr.q,
                (k, v),
                kv_cache_sf=(fp.k_sf, fp.v_sf),
                k_scale=fp.k_scale,
                v_scale=fp.v_scale,
            )
        ),
        ref,
    )
    # the fp8 and bf16 classes refuse block scales
    w8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    w8.plan(*plan_args, causal=True, q_data_type=torch.bfloat16, kv_data_type=E4)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="kv_cache_sf"):
        w8.run(pr.q, (pr.k_nhd.to(E4), pr.v_nhd.to(E4)), kv_cache_sf=(fp.k_sf, fp.v_sf))
    with pytest.raises(ValueError):  # the functional API: an NVFP4 cache without scales
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
            kv_fmt="nvfp4",
        )
        engine.run_attention(
            planned,
            pr.q,
            fp.k_nhd,
            fp.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(128),
        )


# ------------------------------------------------------------------------------------------------------ scale edge cases
def test_scale_constants_outside_binary32_are_refused():
    """``engine.check_sm_scale`` / ``engine.check_v_scale`` are the one place the scales of the chain are checked: the constant
    ``c = fp32(sm_scale * k_scale * log2 e)`` is a normal binary32 number, the key scale a binary32 number, the value scale
    finite and nonzero in binary32."""
    sm = 1.0 / math.sqrt(128)
    assert engine.check_sm_scale(sm, 0.25) == spec.scale_log2(sm, 0.25)
    assert engine.check_sm_scale(1e-37) == spec.scale_log2(1e-37) >= spec.FLT_MIN
    for ks in (0.0, -1.0, float("inf"), float("nan"), 1e39, 2 * spec.FLT_MAX):
        with pytest.raises(ValueError, match="k_scale"):
            engine.check_sm_scale(sm, ks)
    # a key scale beyond binary32 is refused even when sm_scale brings the constant back into range
    assert spec.scale_log2(1e-10, 1e39) < spec.FLT_MAX
    with pytest.raises(ValueError, match="k_scale"):
        engine.check_sm_scale(1e-10, 1e39)
    for s, ks in (
        (sm, 1e-60),  # c rounds to zero
        (sm, 1e-44),  # c subnormal
        (1e-40, 1.0),  # c subnormal, the bf16 class
        (1e25, 1e25),  # c beyond binary32
        (float("inf"), 1.0),
        (float("nan"), 1.0),
        (0.0, 1.0),
        (-sm, 1.0),
    ):
        with pytest.raises(ValueError):
            engine.check_sm_scale(s, ks)
    for v in (float("inf"), -float("inf"), float("nan"), 1e39, 0.0, -0.0, 1e-50):
        with pytest.raises(ValueError, match="v_scale"):
            engine.check_v_scale(v)
    assert engine.check_v_scale(-1.5) == -1.5  # a signed scale is an input of the chain
    assert engine.check_v_scale(0.1) == spec.to_f32(0.1)


# ------------------------------------------------------------------------------------------------------ plan dtype arguments
@needs_device
def test_dtype_arguments_raise_the_typed_error_and_name_the_planned_class():
    """A ``kv_data_type`` that names no torch dtype (``"nvfp4"``) raises the typed error, not the lookup's ``AttributeError``; the
    refusal of a query dtype names the class the KV dtype plans, and the family while the KV dtype itself is not one of them."""
    lp = problem([64], [700], 32, 8, 128, 16, 10, E4)
    pr = lp.pr
    ws = workspace(pr.device)
    paged = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)

    def plans():
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        yield lambda **kw: w.plan(*paged, causal=True, **kw)
        d = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        yield lambda **kw: d.plan(*paged[1:], **kw)
        r = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        yield lambda **kw: r.plan(
            pr.qo_indptr, pr.kv_indptr_rag, 32, 8, 128, causal=True, **kw
        )

    for plan in plans():
        for bad in ("nvfp4", "fp8", "nn"):
            with pytest.raises(bi.BatchInvariantUnsupportedError, match="kv_data_type"):
                plan(q_data_type=torch.bfloat16, kv_data_type=bad)
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="q_data_type"):
            plan(q_data_type="bf16", kv_data_type=E4)
        for kv, cls in (
            (E4, "attention.e4m3kv"),
            (E5, "attention.e5m2kv"),
            (torch.uint8, "attention.nvfp4kv"),
            (torch.bfloat16, "attention.bf16"),
        ):
            with pytest.raises(bi.BatchInvariantUnsupportedError) as e:
                plan(q_data_type=E4, kv_data_type=kv)
            assert f"`{cls}`" in str(e.value) and "q_data_type" in str(e.value)
        with pytest.raises(bi.BatchInvariantUnsupportedError) as e:
            plan(q_data_type=torch.bfloat16, kv_data_type=torch.float16)
        assert "`attention`" in str(e.value) and "attention.bf16" not in str(e.value)
    # dtype names are still dtypes
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    w.plan(*paged, causal=True, q_data_type="bfloat16", kv_data_type="float8_e4m3fn")
    assert torch.equal(
        bits(w.run(pr.q, lp.cache("NHD"), k_scale=lp.k_scale, v_scale=lp.v_scale)),
        bits(run_paged_prefill(lp)),
    )


# ------------------------------------------------------------------------------------------------------ recorded receipts
#: The fingerprints of the nine low-bit-KV classes. A fingerprint names a class: a change of placement or of addressing (the
#: layout the block scales are read in, the argument forms a wrapper accepts) must leave every one of them where it is.
LOWBIT_FINGERPRINTS = {
    ("e4m3", 64): "attn-e4m3kv-d64-a00dd1765f9182b4",
    ("e5m2", 64): "attn-e5m2kv-d64-a8fdf71de8b7c88c",
    ("nvfp4", 64): "attn-nvfp4kv-d64-d73d0adce1e215f9",
    ("e4m3", 128): "attn-e4m3kv-d128-a961e59f805ec08c",
    ("e5m2", 128): "attn-e5m2kv-d128-ee18bcf735786c12",
    ("nvfp4", 128): "attn-nvfp4kv-d128-dd667c5f850bd4fe",
    ("e4m3", 256): "attn-e4m3kv-d256-7fedccc342c5f71a",
    ("e5m2", 256): "attn-e5m2kv-d256-7e342886ee623ba2",
    ("nvfp4", 256): "attn-nvfp4kv-d256-ac5d34ddab6116f6",
}


@pytest.mark.parametrize("op", WRAPPERS)
def test_lowbit_fingerprints_are_the_recorded_ones(op):
    for (fmt, D), want in LOWBIT_FINGERPRINTS.items():
        assert bi.fingerprint(op, head_dim=D, kv_dtype=fmt) == want, (fmt, D)


# ------------------------------------------------------------------------------------------------------ the V block-scale layout
def swizzle_v_sf(sf, layout="NHD", live=None, seed=0):
    """Linear NVFP4 block scales of a paged cache in the 4-token swizzle a serving engine's cache writer stores V scales in for
    trtllm-gen: the scale of block ``s`` (of ``S``) of the token at page offset ``t`` is written at token
    ``(t // 4) * 4 + s // (S / 4)``, block ``(s % (S / 4)) * 4 + t % 4`` of the same page and head. ``live`` (``(pages, page)``
    bool): only the scales of these tokens are written, as a writer appends them; every other byte is stale (a finite code)."""
    x = sf if layout == "HND" else sf.transpose(1, 2)  # (pages, Hkv, page, S)
    P, H, N, S = x.shape
    per = S // 4
    t = torch.arange(N, device=x.device)[:, None]
    s = torch.arange(S, device=x.device)[None, :]
    ts, ss = (t // 4) * 4 + s // per, (s % per) * 4 + t % 4
    out = torch.empty_like(x)
    out[:, :, ts, ss] = x
    if live is not None:
        written = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        written[:, :, ts, ss] = live[:, None, :, None].expand(P, H, N, S)
        g = torch.Generator().manual_seed(seed)
        stale = torch.randint(0, 0x7F, x.shape, generator=g, dtype=torch.uint8)
        out = torch.where(written, out, stale.to(x.device))
    return (out if layout == "HND" else out.transpose(1, 2)).contiguous()


def live_tokens(pr: Problem):
    """``(pool pages, page)`` bool: the tokens of the pool a request holds (the tail of a last page and the junk pages are not)."""
    live = torch.zeros(pr.k_nhd.shape[0], pr.page, dtype=torch.bool, device=pr.device)
    for r in range(len(pr.kv_lens)):
        pages = pr.kv_indices[int(pr.kv_indptr[r]) : int(pr.kv_indptr[r + 1])].long()
        if len(pages):
            live[pages] = True
            live[pages[-1], int(pr.kv_last[r]) :] = False
    return live


def test_v_scale_layout_addressing_and_kernel_identity():
    """Device-free: the conversion stage's swizzle address is the writer's statement and a permutation inside each group of four
    tokens; the swizzled kernel is another kernel (name) of the same class (fingerprint); the layouts a configuration cannot
    express are refused."""
    from flashinfer.attention.batch_invariant.blocks.compile import kernel_name
    from flashinfer.attention.batch_invariant.blocks.convert import sf_swizzle_coord

    for n_blk in (4, 8, 16):  # D = 64 / 128 / 256
        per = n_blk // 4
        seen = set()
        for tok in range(16):
            for blk in range(n_blk):
                b, t = sf_swizzle_coord(n_blk, blk, tok)
                assert (b, t) == (
                    (blk % per) * 4 + tok % 4,
                    (tok // 4) * 4 + blk // per,
                )
                assert (
                    t // 4 == tok // 4 and 0 <= b < n_blk
                )  # inside the token's group of four
                seen.add((b, t))
        assert len(seen) == 16 * n_blk
    # the test's writer (an independent statement on tensors) agrees with it
    sf = (torch.arange(2 * 8 * 3 * 8) % 251).to(torch.uint8).reshape(2, 8, 3, 8)
    sw = swizzle_v_sf(sf)
    for tok in range(8):
        for blk in range(8):
            b, t = sf_swizzle_coord(8, blk, tok)
            assert torch.equal(sw[:, t, :, b], sf[:, tok, :, blk])
    assert torch.equal(
        swizzle_v_sf(sf.transpose(1, 2).contiguous(), "HND"),
        sw.transpose(1, 2).contiguous(),
    )
    lin = kernel_name(128, 4, 16, 4, kv_fmt="nvfp4")
    assert kernel_name(128, 4, 16, 4, kv_fmt="nvfp4", v_sf_swizzled=True) == (
        lin + "_vsfswz"
    )
    engine.check_v_sf_layout(False, "bf16", False, 1)
    engine.check_v_sf_layout(False, "nvfp4", True, 1)
    engine.check_v_sf_layout(True, "nvfp4", False, 4)
    engine.check_v_sf_layout(True, "nvfp4", False, 128)
    for args in (
        (True, "nvfp4", False, 1),  # a page of one token has no group of four
        (True, "nvfp4", False, 2),
        (True, "nvfp4", True, 1),  # the ragged pool has no pages
        (True, "e4m3", False, 16),  # no block scales
        (True, "bf16", False, 16),
        (1, "nvfp4", False, 16),  # the layout is stated as a bool
        ("swizzled", "nvfp4", False, 16),
        (None, "nvfp4", False, 16),
    ):
        with pytest.raises(ValueError, match="v_sf_swizzled"):
            engine.check_v_sf_layout(*args)


@needs_device
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    "D,page", [(64, 4), (64, 16), (128, 16), (128, 128), (256, 4), (256, 16)]
)
def test_nvfp4_swizzled_v_scales_are_the_linear_cache(D, page, layout):
    """One quantized cache with its V block scales stored in both layouts: (linear scales, the default option) and (swizzled
    scales, ``plan(v_sf_swizzled=True)``) give the same bits, for paged prefill and paged decode, in both cache layouts. The KV
    lengths leave page tails that are not multiples of four tokens: the scales of a tail's tokens lie in the rows of their group
    of four, among stale bytes of the tokens not appended yet."""
    fp = fp4_problem([256, 1, 33], [1501, 2503, 770], 32, 8, D, page, 3)
    pr = fp.pr
    assert any(int(n) % 4 for n in pr.kv_last)
    ref = bits(fp4_run_paged_prefill(fp, layout)).clone()
    k_sf, v_sf = fp.sf(layout)
    v_swz = swizzle_v_sf(v_sf, layout, live=live_tokens(pr), seed=D + page)
    assert not torch.equal(v_swz, v_sf)
    got = fp4_run_paged_prefill(fp, layout, sf=(k_sf, v_swz), v_sf_swizzled=True)
    assert torch.equal(bits(got), ref)
    # the single 5-D form of the paged wrappers
    five = torch.stack([k_sf, v_swz], dim=1)
    got = fp4_run_paged_prefill(fp, layout, sf=five, v_sf_swizzled=True)
    assert torch.equal(bits(got), ref)
    dec, last = fp4_run_decode(fp, layout, sf=(k_sf, v_swz), v_sf_swizzled=True)
    assert torch.equal(bits(dec), ref[last])
    assert torch.equal(bits(fp4_run_decode(fp, layout)[0]), ref[last])


@needs_device
def test_nvfp4_v_scale_layout_is_stated_not_detected():
    """The option is live: scales read in the other layout give other bits with no error (the bytes cannot tell the layout), so
    the caller states it. The K block scales are linear under both values."""
    fp = fp4_problem([64, 1], [700, 2500], 32, 8, 128, 16, 5)
    k_sf, v_sf = fp.sf("NHD")
    v_swz = swizzle_v_sf(v_sf)
    ref = bits(fp4_run_paged_prefill(fp)).clone()
    assert torch.equal(
        bits(fp4_run_paged_prefill(fp, sf=(k_sf, v_swz), v_sf_swizzled=True)), ref
    )
    wrong_option = bits(fp4_run_paged_prefill(fp, v_sf_swizzled=True))
    wrong_data = bits(fp4_run_paged_prefill(fp, sf=(k_sf, v_swz)))
    assert (wrong_option != ref).float().mean() > 0.5
    assert (wrong_data != ref).float().mean() > 0.5
    # K scales stored swizzled are not what the option describes
    k_swz = swizzle_v_sf(k_sf)
    for kw in ({}, {"v_sf_swizzled": True}):
        sf = (k_swz, v_swz if kw else v_sf)
        assert not torch.equal(bits(fp4_run_paged_prefill(fp, sf=sf, **kw)), ref)


@needs_device
def test_nvfp4_v_scale_layout_refusals():
    """What the swizzle cannot express is refused with the typed error at ``plan()``: a page that is not a multiple of four
    tokens (linear scales serve those pages), a cache without block scales, a value that is not a bool; without
    ``batch_invariant=True`` the option is not read and ``True`` raises."""
    import types

    ws = workspace(torch.device("cuda"))
    for page in (1, 2):
        fp = fp4_problem([40, 1], [300, 1001], 8, 2, 64, page, 6)
        pr = fp.pr
        args = (pr.kv_indptr, pr.kv_indices, pr.kv_last, 8, 2, 64, page)
        kw = dict(q_data_type=torch.bfloat16, kv_data_type=torch.uint8)
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="v_sf_swizzled"):
            w.plan(pr.qo_indptr, *args, causal=True, v_sf_swizzled=True, **kw)
        d = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="v_sf_swizzled"):
            d.plan(*args, v_sf_swizzled=True, **kw)
        # linear scales serve the page: the paged wrapper gives the ragged wrapper's bits
        assert torch.equal(
            bits(fp4_run_paged_prefill(fp)), bits(fp4_run_ragged_prefill(fp))
        )
    fp = fp4_problem([64], [700], 32, 8, 128, 16, 10)
    pr = fp.pr
    plan_args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    for kv in (E4, E5, torch.bfloat16):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="v_sf_swizzled"):
            w.plan(
                *plan_args,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=kv,
                v_sf_swizzled=True,
            )
    for bad in (1, "swizzled", None):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="v_sf_swizzled"):
            w.plan(
                *plan_args,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.uint8,
                v_sf_swizzled=bad,
            )
    # the delegate of the ragged wrapper (its pool has no pages)
    r = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="v_sf_swizzled"):
        r._batch_invariant.plan(
            qo_indptr=pr.qo_indptr,
            kv_indptr=pr.kv_indptr_rag,
            kv_indices=None,
            kv_last_page_len=None,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim_qk=128,
            head_dim_vo=None,
            page_size=1,
            causal=True,
            window_left=-1,
            logits_soft_cap=None,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
            o_data_type=None,
            sm_scale=None,
            v_sf_swizzled=True,
        )
    # the regular wrappers do not read the option: the check precedes everything else in their plan()
    regular = types.SimpleNamespace(_batch_invariant=None)
    paged_plan = flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan
    with pytest.raises(ValueError, match="v_sf_swizzled"):
        getattr(paged_plan, "__wrapped__", paged_plan)(
            regular, *plan_args, causal=True, v_sf_swizzled=True
        )
    with pytest.raises(ValueError, match="v_sf_swizzled"):
        flashinfer.BatchDecodeWithPagedKVCacheWrapper._plan_impl(
            regular, *plan_args[1:], v_sf_swizzled=True
        )


# ------------------------------------------------------------------------------------------------------ every code of the formats
def fp8_value(code: int, fmt: str) -> float:
    """One fp8 byte decoded from the format's definition (no torch conversion)."""
    sign = -1.0 if code & 0x80 else 1.0
    if fmt == "e4m3":
        e, m = (code >> 3) & 0xF, code & 7
        if e == 15 and m == 7:
            return float("nan")
        return sign * (m / 8 * 2.0**-6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    e, m = (code >> 2) & 0x1F, code & 3
    if e == 31:
        return sign * float("inf") if m == 0 else float("nan")
    return sign * (m / 4 * 2.0**-14 if e == 0 else (1 + m / 4) * 2.0 ** (e - 15))


def widened_problem(pr: Problem, k_nhd, v_nhd) -> Problem:
    """``pr`` with its paged pools replaced by the bf16 values ``k_nhd`` / ``v_nhd`` (the bf16-class oracle of a low-bit cache)."""
    w = Problem.__new__(Problem)
    w.__dict__.update(pr.__dict__)
    w.k_nhd, w.v_nhd = k_nhd.to(torch.bfloat16), v_nhd.to(torch.bfloat16)
    return w


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("dtype", [E4, E5], ids=["e4m3", "e5m2"])
def test_every_finite_fp8_code_is_widened_exactly(D, dtype):
    """Caches drawn from every finite code of the format (both zeros, the subnormals, the largest magnitudes): the class's bits
    are the bf16 class's over the widened cache with ``k_scale`` folded into ``sm_scale``. The non-finite codes (e4m3 NaN,
    e5m2 infinities and NaNs) are outside the class, as non-finite values are outside the bf16 class: they are widened like any
    code and propagate as the fp32 arithmetic propagates them; nothing on the device checks for them."""
    fmt = FMT_OF[dtype]
    values = torch.tensor([fp8_value(c, fmt) for c in range(256)])
    finite = torch.isfinite(values)
    codes = torch.arange(256, dtype=torch.uint8)
    # torch's widening (the oracle's) is the format's definition, on the host and on the device
    for dev in ("cpu", "cuda"):
        w = codes.to(dev).view(dtype).to(torch.bfloat16).float().cpu()
        assert torch.equal(w[finite], values[finite])
        assert not torch.isfinite(w[~finite]).any()
    assert int(finite.sum()) == (254 if fmt == "e4m3" else 248)
    fin = codes[finite]
    pr = Problem([40, 1, 300], [1100, 2500, 300], 8, 2, D, 16, seed=D)
    g = torch.Generator().manual_seed(D)
    shape = pr.k_nhd.shape

    def pool():
        x = fin[torch.randint(0, len(fin), shape, generator=g)]
        assert len(torch.unique(x)) == len(fin)  # every finite code is in the cache
        return x.to(pr.device).view(dtype)

    lp = LowbitProblem.__new__(LowbitProblem)
    lp.pr, lp.dtype, lp.v_scale = pr, dtype, 1.0
    lp.k_nhd, lp.v_nhd = pool(), pool()
    oracle = widened_problem(pr, lp.k_nhd, lp.v_nhd)
    sm = 1.0 / math.sqrt(D)
    for k_scale in (1.0, 4.0 / FP8_MAX[dtype]):
        lp.k_scale = k_scale
        got = run_paged_prefill(lp)
        assert torch.isfinite(got.float()).all()
        want = bf16_paged_prefill(oracle, sm_scale=sm * k_scale)
        assert torch.equal(bits(got), bits(want)), k_scale
    dec, last = run_decode(lp)
    assert torch.equal(bits(dec), bits(got[last]))


def signed_fp4_problem(q_lens, kv_lens, Hq, Hkv, D, page, seed, sf_codes, k_scale):
    """An NVFP4 problem over every nibble (the eight negative e2m1 codes included: ``create_nvfp4_kv`` clears the sign bits)
    with block scales drawn from ``sf_codes`` (e4m3 bytes)."""
    fp = Fp4Problem.__new__(Fp4Problem)
    pr = Problem(q_lens, kv_lens, Hq, Hkv, D, page, seed=seed)
    g = torch.Generator().manual_seed(seed)
    lead = tuple(pr.k_nhd.shape[:-1])
    codes = torch.tensor(sf_codes, dtype=torch.uint8)

    def data():
        x = torch.randint(0, 256, lead + (D // 2,), generator=g, dtype=torch.uint8)
        return x.to(pr.device)

    def scales():
        idx = torch.randint(0, len(codes), lead + (D // 16,), generator=g)
        return codes[idx].to(pr.device)

    fp.pr, fp.k_scale, fp.v_scale = pr, k_scale, 1.0
    fp.k_nhd, fp.v_nhd, fp.k_sf, fp.v_sf = data(), data(), scales(), scales()
    fp.k_rag, fp.v_rag = fp._gather(fp.k_nhd), fp._gather(fp.v_nhd)
    fp.k_sf_rag, fp.v_sf_rag = fp._gather(fp.k_sf), fp._gather(fp.v_sf)
    return fp


E4M3_FINITE_CODES = [c for c in range(256) if c not in (0x7F, 0xFF)]


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize(
    "name,sf_codes,k_scale",
    [
        ("moderate", [56, 48, 40, 32, 60, 44], 1.0),
        ("every_finite_scale_code", E4M3_FINITE_CODES, 1.0 / 256),
        ("extreme", [0x7E, 0xFE, 0x01, 0x81, 0x00, 0x80, 0x08, 0x07], 1.0 / 256),
    ],
)
def test_nvfp4_signed_values_and_every_scale_code(D, name, sf_codes, k_scale):
    """Negative e2m1 values and block scales over the whole finite e4m3 range (negative scales, both zeros, the smallest subnormal
    and the largest magnitude): every product is an exact bf16 value, and the class's bits are the bf16 class's over the products
    with ``k_scale`` folded into ``sm_scale``; decode rows are prefill rows, the ragged pool the paged one."""
    fp = signed_fp4_problem(
        [40, 1, 300], [1100, 2500, 300], 8, 2, D, 16, 100 + D, sf_codes, k_scale
    )
    pr = fp.pr
    assert bool((fp.k_nhd & 0x88).any()) and bool((fp.v_nhd & 0x88).any())
    k_dq, v_dq = nvfp4_dequant(fp.k_nhd, fp.k_sf), nvfp4_dequant(fp.v_nhd, fp.v_sf)
    for x in (k_dq, v_dq):
        assert torch.equal(x.to(torch.bfloat16).float(), x)  # exact bf16 leaves
    assert bool((k_dq < 0).any()) and bool((v_dq < 0).any())
    got = fp4_run_paged_prefill(fp)
    assert torch.isfinite(got.float()).all()
    want = bf16_paged_prefill(
        widened_problem(pr, k_dq, v_dq), sm_scale=(1.0 / math.sqrt(D)) * k_scale
    )
    assert torch.equal(bits(got), bits(want))
    dec, last = fp4_run_decode(fp)
    assert torch.equal(bits(dec), bits(got[last]))
    assert torch.equal(bits(fp4_run_ragged_prefill(fp)), bits(got))


# ------------------------------------------------------------------------------------------------------ serving call shapes
@needs_device
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_serving_call_shape_fp8(layout):
    """The argument list a serving engine passes on every call: K and V as the two halves of one cache tensor, ``q_scale`` 1.0,
    the K / V scales as 1-element tensors, the output buffer, and ``None`` for the features that are off. The bits are the
    minimal call's."""
    lp = problem([64, 1, 300], [700, 2500, 300], 32, 8, 128, 16, 21, E4)
    pr = lp.pr
    ref = bits(run_paged_prefill(lp, layout)).clone()
    k, v = lp.cache(layout)
    cache = torch.stack(
        [k, v], dim=1
    )  # (pages, 2, ...): one allocation, K and V its halves
    ks = torch.tensor([lp.k_scale], dtype=torch.float32, device=pr.device)
    vs = torch.tensor(lp.v_scale, dtype=torch.float32)
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
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=E4,
        o_data_type=torch.bfloat16,
    )
    out = torch.empty_like(pr.q)
    ret = w.run(
        pr.q, cache, q_scale=1.0, k_scale=ks, v_scale=vs, out=out, kv_cache_sf=None
    )
    assert ret is out and torch.equal(bits(out), ref)
    last = torch.tensor(
        [int(pr.qo_indptr[i + 1]) - 1 for i in range(len(pr.q_lens))], device=pr.device
    )
    d = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), layout, batch_invariant=True
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
        kv_data_type=E4,
        o_data_type=torch.bfloat16,
    )
    dout = torch.empty_like(pr.q[last])
    d.run(
        pr.q[last].contiguous(),
        cache,
        q_scale=torch.ones(1, device=pr.device),
        k_scale=ks,
        v_scale=vs,
        out=dout,
        kv_cache_sf=None,
        sinks=None,
    )
    assert torch.equal(bits(dout), ref[last])


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_serving_page_layout_nvfp4(D):
    """A serving engine's NVFP4 page: one byte tensor per block, ``[K data | K scales | V data | V scales]`` (HND), read through
    four strided views of the one allocation, with the V scales in the 4-token swizzle its writer stores. The bits are those of
    the separate contiguous tensors with linear scales."""
    fp = fp4_problem([64, 1, 300], [700, 2501, 303], 32, 8, D, 16, 22)
    pr = fp.pr
    ref = bits(fp4_run_paged_prefill(fp)).clone()
    H, N, row = pr.Hkv, pr.page, D // 2 + D // 16
    pages = fp.k_nhd.shape[0]
    cache = torch.zeros(pages, 2 * H, N, row, dtype=torch.uint8, device=pr.device)
    side = (
        H * N * row
    )  # bytes of the K (or V) half of a page: the data of all heads, then their scales

    def views(half):
        base = cache.storage_offset() + half * side
        data = torch.as_strided(
            cache,
            (pages, H, N, D // 2),
            (cache.stride(0), N * (D // 2), D // 2, 1),
            base,
        )
        scales = torch.as_strided(
            cache,
            (pages, H, N, D // 16),
            (cache.stride(0), N * (D // 16), D // 16, 1),
            base + H * N * (D // 2),
        )
        return data, scales

    (k, k_sf), (v, v_sf) = views(0), views(1)
    k_hnd, v_hnd = fp.cache("HND")
    ksf_hnd, vsf_hnd = fp.sf("HND")
    k.copy_(k_hnd)
    v.copy_(v_hnd)
    k_sf.copy_(ksf_hnd)
    v_sf.copy_(swizzle_v_sf(vsf_hnd, "HND", live=live_tokens(pr), seed=D))
    e4 = torch.float8_e4m3fn
    got = fp4_run_paged_prefill(
        fp, "HND", sf=(k_sf.view(e4), v_sf.view(e4)), v_sf_swizzled=True
    )
    assert torch.equal(bits(got), ref)  # the scale views alone, next to contiguous data
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "HND", batch_invariant=True
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
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.uint8,
        v_sf_swizzled=True,
    )
    out = torch.empty_like(pr.q)
    w.run(
        pr.q,
        (k, v),
        q_scale=1.0,
        k_scale=torch.tensor([fp.k_scale], device=pr.device),
        v_scale=torch.tensor([fp.v_scale], device=pr.device),
        out=out,
        kv_cache_sf=(k_sf.view(e4), v_sf.view(e4)),
    )
    assert torch.equal(bits(out), ref)
