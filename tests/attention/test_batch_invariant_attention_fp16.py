# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp16 batch-invariant attention class (``attention.fp16``: binary16 q / K / V / output behind ``batch_invariant=True`` on the
three wrappers): the bf16 suite's bitwise invariances on binary16 operands, the device-free checks of the class dimension (the
fingerprints, the term, the kernel names), the refusal of mixed formats, and the behaviour at the edges of the binary16 range
(the output rounding's overflow to infinity, the gradual underflow of the probabilities)."""

import math
from dataclasses import replace

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
F16 = torch.float16
F16_MAX = 65504.0
#: The fp16 class's fingerprints, and the bf16 class's (unchanged by the operand-format dimension).
FP16_FINGERPRINTS = {
    64: "attn-fp16-d64-94871aafae9b4e51",
    128: "attn-fp16-d128-f00b125b7722e683",
    256: "attn-fp16-d256-79c1eec69fd60b8a",
}
BF16_FINGERPRINTS = {
    64: "attn-bf16-d64-9a05233ec9351b0d",
    128: "attn-bf16-d128-027b0379515b82e2",
    256: "attn-bf16-d256-b8cb59dab4eb6c0f",
}


def bits(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype in (torch.float16, torch.bfloat16), t.dtype
    return t.view(torch.int16)


# ------------------------------------------------------------------------------------------------------ problems
class Problem:
    """A batch of requests with per-request query and KV lengths in ``dtype`` (binary16 unless stated), as a paged KV cache (NHD /
    HND, random page table with junk pages) and as the same keys packed along the token dimension (ragged). The values are drawn
    in fp32 and rounded to ``dtype``, so binary16 operands use all eleven significand bits."""

    def __init__(
        self, q_lens, kv_lens, Hq, Hkv, D, page, seed=0, device="cuda", dtype=F16
    ):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.q_lens, self.kv_lens, self.Hq, self.Hkv, self.D, self.page = (
            list(q_lens),
            list(kv_lens),
            Hq,
            Hkv,
            D,
            page,
        )
        self.dtype = dtype
        self.device = torch.device(device)
        self.q = (torch.randn(sum(q_lens), Hq, D, generator=g) * 1.5).to(dtype)
        self.k_rag = (torch.randn(sum(kv_lens), Hkv, D, generator=g) * 1.5).to(dtype)
        self.v_rag = (torch.randn(sum(kv_lens), Hkv, D, generator=g) * 1.5).to(dtype)
        self._seed_pool = int(torch.randint(0, 1 << 30, (1,), generator=g).item())
        self.build()

    def build(self):
        """(Re)build the device tensors and the paged pool from the host ``q`` / ``k_rag`` / ``v_rag``."""
        device, page, dtype = self.device, self.page, self.dtype
        q_lens, kv_lens, Hkv, D = self.q_lens, self.kv_lens, self.Hkv, self.D
        g = torch.Generator(device="cpu").manual_seed(self._seed_pool)
        self.q = self.q.to(device).contiguous()
        k_rag, v_rag = self.k_rag.cpu(), self.v_rag.cpu()
        self.k_rag, self.v_rag = (
            k_rag.to(device).contiguous(),
            v_rag.to(device).contiguous(),
        )

        def indptr(lens):
            return torch.tensor(
                [0] + list(torch.cumsum(torch.tensor(lens), 0)),
                dtype=torch.int32,
                device=device,
            )

        self.qo_indptr = indptr(q_lens)
        self.kv_indptr_rag = indptr(kv_lens)
        n_pages = [(L + page - 1) // page for L in kv_lens]
        pool_pages = sum(n_pages) + 7
        perm = torch.randperm(pool_pages, generator=g).tolist()
        self.kv_indices = torch.tensor(
            perm[: sum(n_pages)], dtype=torch.int32, device=device
        )
        self.kv_indptr = indptr(n_pages)
        self.kv_last = torch.tensor(
            [
                L - (n - 1) * page if n else 0
                for L, n in zip(kv_lens, n_pages, strict=True)
            ],
            dtype=torch.int32,
            device=device,
        )
        # junk everywhere, then the live keys
        k_pool = (torch.randn(pool_pages, page, Hkv, D, generator=g) * 1.5).to(dtype)
        v_pool = (torch.randn(pool_pages, page, Hkv, D, generator=g) * 1.5).to(dtype)
        tok = off = 0
        for r, L in enumerate(kv_lens):
            for j in range(n_pages[r]):
                pg = perm[off + j]
                n = min(page, L - j * page)
                k_pool[pg, :n] = k_rag[tok + j * page : tok + j * page + n]
                v_pool[pg, :n] = v_rag[tok + j * page : tok + j * page + n]
            off += n_pages[r]
            tok += L
        self.k_nhd, self.v_nhd = k_pool.to(device), v_pool.to(device)
        return self

    def rows(self, i):
        return slice(int(self.qo_indptr[i]), int(self.qo_indptr[i + 1]))

    def keys(self, i):
        return slice(int(self.kv_indptr_rag[i]), int(self.kv_indptr_rag[i + 1]))

    def cache(self, layout="NHD"):
        if layout == "NHD":
            return self.k_nhd, self.v_nhd
        return self.k_nhd.transpose(1, 2).contiguous(), self.v_nhd.transpose(
            1, 2
        ).contiguous()

    def ragged(self, layout="NHD"):
        if layout == "NHD":
            return self.k_rag, self.v_rag
        return self.k_rag.transpose(0, 1).contiguous(), self.v_rag.transpose(
            0, 1
        ).contiguous()


def derived(pr: Problem, q_lens, kv_lens, q, k_rag, v_rag, seed, Hq=None, Hkv=None):
    """A problem over given operands (a sub-batch, a permutation, a chunk of ``pr``) in a fresh random page table."""
    new = Problem(
        q_lens, kv_lens, Hq or pr.Hq, Hkv or pr.Hkv, pr.D, pr.page, seed, dtype=pr.dtype
    )
    new.q, new.k_rag, new.v_rag = q.clone(), k_rag.clone(), v_rag.clone()
    return new.build()


def workspace(device, mb=128):
    return torch.empty(mb << 20, dtype=torch.uint8, device=device)


def plan_dtypes(pr: Problem, plan_kw):
    kw = dict(q_data_type=pr.dtype, kv_data_type=pr.dtype)
    kw.update(plan_kw)
    return kw


def run_paged_prefill(pr: Problem, layout="NHD", window_left=-1, **plan_kw):
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
        **plan_dtypes(pr, plan_kw),
    )
    return w.run(pr.q, pr.cache(layout))


def run_ragged_prefill(pr: Problem, layout="NHD", window_left=-1, **plan_kw):
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
        **plan_dtypes(pr, plan_kw),
    )
    k, v = pr.ragged(layout)
    return w.run(pr.q, k, v)


def run_decode(pr: Problem, layout="NHD", window_left=-1, q_len=1, **plan_kw):
    """Decode of the last ``q_len`` tokens of every request of ``pr`` (q rows picked from the prefill problem)."""
    assert all(ql >= q_len for ql in pr.q_lens)
    last = torch.cat(
        [
            torch.arange(
                int(pr.qo_indptr[i + 1]) - q_len,
                int(pr.qo_indptr[i + 1]),
                device=pr.device,
            )
            for i in range(len(pr.q_lens))
        ]
    )
    q = pr.q[last].contiguous()
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), layout, batch_invariant=True
    )
    if q_len != 1:
        plan_kw = dict(plan_kw, q_len_per_req=q_len)
    w.plan(
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        window_left=window_left,
        **plan_dtypes(pr, plan_kw),
    )
    return w.run(q, pr.cache(layout)), last


def reference(pr: Problem, window_left=-1, sm_scale=None, with_scale=False):
    """Causal (bottom-right aligned) attention per request in binary64 over the operands' exact values: the yardstick for accuracy
    (never for bits). ``with_scale`` also returns ``sum_k p_k |v_k|`` (the magnitude the relative rounding errors of an output
    element scale with) and ``sum_k |v_k|`` over the visible keys (what the absolute rounding error of the subnormal weights
    scales with)."""
    outs, mags, sums = [], [], []
    tok_q = tok_kv = 0
    g = pr.Hq // pr.Hkv
    scale = 1.0 / math.sqrt(pr.D) if sm_scale is None else sm_scale
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        q = pr.q[tok_q : tok_q + ql].double()
        k = pr.k_rag[tok_kv : tok_kv + kl].double().repeat_interleave(g, dim=1)
        v = pr.v_rag[tok_kv : tok_kv + kl].double().repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) * scale
        pos = torch.arange(kl - ql, kl, device=pr.device)[:, None]
        key = torch.arange(kl, device=pr.device)[None, :]
        mask = key <= pos
        if window_left >= 0:
            mask &= key >= pos - window_left
        s = s.masked_fill(~mask[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
        mags.append(torch.einsum("hqk,khd->qhd", p, v.abs()))
        sums.append(torch.einsum("qk,khd->qhd", mask.double(), v.abs()))
        tok_q += ql
        tok_kv += kl
    if with_scale:
        return torch.cat(outs), torch.cat(mags), torch.cat(sums)
    return torch.cat(outs)


# ------------------------------------------------------------------------------------------------------ device-free
@pytest.mark.parametrize("op", WRAPPERS)
def test_fingerprint_operand_format_dimension(op):
    """The operand dtype is an optional dimension of the receipt: omitted (or bf16) it is the bf16 class's recorded digest, fp16
    names the fp16 class; the three wrappers share both."""
    # the mask mode is a fixed dimension of the prefill wrappers' receipt and optional for the decode wrapper
    kw = {} if op.startswith("BatchDecode") else {"causal": True}
    for D in (64, 128, 256):
        assert bi.fingerprint(op, head_dim=D, **kw) == BF16_FINGERPRINTS[D]
        assert (
            bi.fingerprint(op, head_dim=D, dtype=torch.bfloat16, **kw)
            == (BF16_FINGERPRINTS[D])
        )
        assert (
            bi.fingerprint(op, head_dim=D, dtype="bf16", **kw) == BF16_FINGERPRINTS[D]
        )
        for dt in (torch.float16, "fp16", "f16"):
            assert (
                bi.fingerprint(op, head_dim=D, dtype=dt, **kw) == FP16_FINGERPRINTS[D]
            )
    for dt in (torch.float32, torch.float8_e4m3fn, "half"):
        with pytest.raises(ValueError):
            bi.fingerprint(op, head_dim=128, dtype=dt, **kw)
    with pytest.raises(TypeError):
        bi.fingerprint(op, head_dim=128, dtype=torch.float16, page_size=16, **kw)


def test_fp16_term_names_its_format_everywhere():
    """The fp16 class's term: the binary16 step in every score and PV slice, ``cvt f16`` on every probability and on the output;
    nothing of the bf16 class's primitives. The bf16 class's term is untouched by the dimension."""
    from flashinfer.attention.batch_invariant import spec
    from flashinfer.batch_invariant.fp_interface.term import sexpr

    row = spec.RowSpec(p=40, window_left=-1)
    f16 = sexpr(spec.AttentionSpec(64, T=16, C=32, fmt="f16").output(row, 0))
    bf16 = sexpr(spec.AttentionSpec(64, T=16, C=32).output(row, 0))
    assert f16.startswith("(cvt f16 ") and bf16.startswith("(cvt bf16 ")
    assert "(mma16 " not in f16 and "bf16" not in f16
    assert "mma16f16" not in bf16 and "f16 " not in bf16.replace("bf16 ", "")
    # the two terms are the same tree up to the step's name and the format of the roundings
    assert f16.replace("(mma16f16 ", "(mma16 ").replace("(cvt f16 ", "(cvt bf16 ") == (
        bf16
    )
    # 41 visible keys: every one is a probability rounded to the class's format (the masked keys of the last tile carry +0)
    assert f16.count("(cvt f16 (ex2 ") == bf16.count("(cvt bf16 (ex2 ") > 0
    assert spec.class_name("f16") == "attention.fp16"
    assert spec.class_name() == "attention.bf16"
    for D in (64, 128, 256):
        assert spec.spec_fingerprint(D, fmt="f16") == FP16_FINGERPRINTS[D]
        assert spec.spec_fingerprint(D) == spec.FINGERPRINTS[D] == BF16_FINGERPRINTS[D]


def test_fp16_kernel_identity_carries_the_format():
    """Every kernel of the fp16 class has its own name (the cache key, in-process and on disk); the bf16 kernels keep theirs."""
    from flashinfer.attention.batch_invariant.blocks.compile import kernel_name as row
    from flashinfer.attention.batch_invariant.swaps.compile import kernel_name as swaps

    assert row(128, 4, 16, 4) == "attn_d128_g4_p16_s4_w1"
    assert row(128, 4, 16, 4, fmt="f16") == "attn_d128_g4_p16_s4_w1_f16"
    assert row(128, 4, 16, 3, 2, True, False, cap=328 << 10, fmt="f16").endswith(
        "_q2_c328_f16"
    )
    assert swaps(128, 8, 16, 8, False, cap=228 << 10).endswith("_c228")
    assert swaps(128, 8, 16, 8, False, cap=228 << 10, fmt="f16").endswith("_c228_f16")


def test_binary16_rounding_model():
    """The two roundings the fp16 class declares, on the CPU model of ``cvt.rn.f16.f32``: the output overflows to infinity from
    65520 (half an ulp above the largest binary16 number), and a probability underflows gradually (11 bits down to ``2^-14``, a
    multiple of ``2^-24`` below, ``+0`` at ``2^-25`` and under)."""
    from flashinfer.batch_invariant.fp_interface import exact

    def cvt(x):
        return exact.to_format(exact.to_f32(x), exact.F16)

    assert cvt(F16_MAX) == F16_MAX
    assert cvt(65519.996) == F16_MAX
    assert cvt(65520.0) == math.inf and cvt(-65520.0) == -math.inf
    assert cvt(1.0) == 1.0 and cvt(1.0 + 2.0**-11) == 1.0
    assert cvt(1.0 + 2.0**-11 + 2.0**-20) == 1.0 + 2.0**-10
    assert cvt(2.0**-24) == 2.0**-24
    assert cvt(2.0**-25) == 0.0  # the tie rounds to the even neighbour, +0
    assert cvt(2.0**-25 * 1.001) == 2.0**-24
    assert cvt(2.0**-26) == 0.0
    assert cvt(3.0 * 2.0**-20 + 2.0**-26) == 3.0 * 2.0**-20  # subnormal: 2^-24 grid


# ------------------------------------------------------------------------------------------------------ the class on the device
@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_matches_reference(D, Hq, Hkv):
    pr = Problem([300, 1000, 64], [700, 2500, 3000], Hq, Hkv, D, 16, seed=1)
    out = run_paged_prefill(pr)
    assert out.dtype == F16
    ref = reference(pr)
    err = (out.double() - ref).abs().max().item()
    # binary16 probabilities and output: an order of magnitude inside the bf16 class's bound
    assert err < 2e-3 + 2e-3 * ref.abs().max().item(), err


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_prefill_equals_decode(D, layout):
    """R4: the last token of every request gives the same bits as a prefill row and as a decode step (the Swaps decode body
    against the row body)."""
    pr = Problem([256, 1, 1024, 33], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=2)
    pre = run_paged_prefill(pr, layout)
    dec, last = run_decode(pr, layout)
    assert dec.dtype == F16
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
    """R1: repeated runs and the replays of a captured ``run()`` give the same bits."""
    pr = Problem([512, 7, 1000], [4096, 1025, 1000], 32, 8, D, 16, seed=4)
    ref = bits(run_paged_prefill(pr)).clone()
    for _ in range(3):
        assert torch.equal(bits(run_paged_prefill(pr)), ref)
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
        q_data_type=F16,
        kv_data_type=F16,
    )
    out = torch.empty_like(pr.q)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
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
    assert torch.equal(bits(run_ragged_prefill(pr)), full)
    for i in (0, 2, 4):  # one request alone
        single = derived(
            pr,
            [q_lens[i]],
            [kv_lens[i]],
            pr.q[pr.rows(i)],
            pr.k_rag[pr.keys(i)],
            pr.v_rag[pr.keys(i)],
            seed=50 + i,
        )
        assert torch.equal(bits(run_ragged_prefill(single)), full[pr.rows(i)]), i
        assert torch.equal(bits(run_paged_prefill(single)), full[pr.rows(i)]), i
    perm = [3, 0, 4, 1, 2]
    pp = derived(
        pr,
        [q_lens[i] for i in perm],
        [kv_lens[i] for i in perm],
        torch.cat([pr.q[pr.rows(i)] for i in perm]),
        torch.cat([pr.k_rag[pr.keys(i)] for i in perm]),
        torch.cat([pr.v_rag[pr.keys(i)] for i in perm]),
        seed=60,
    )
    permuted = bits(run_paged_prefill(pp))
    off = 0
    for i in perm:
        n = q_lens[i]
        assert torch.equal(permuted[off : off + n], full[pr.rows(i)]), i
        off += n


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_decode_batch_invariance_and_page_table_junk(D):
    """R2 for decode: a request's step gives the same bits in a batch of one and in a batch of eight, whatever pages the pool
    assigns (the page table and the junk beyond each request's keys never reach the bits)."""
    kv = [1024, 4096, 33, 700, 2500, 8192, 1, 1500]
    pr = Problem([1] * 8, kv, 32, 8, D, 16, seed=6)
    out8, _ = run_decode(pr)
    for i in (0, 1, 5, 6):
        one = derived(
            pr,
            [1],
            [kv[i]],
            pr.q[i : i + 1],
            pr.k_rag[pr.keys(i)],
            pr.v_rag[pr.keys(i)],
            seed=100 + i,
        )
        out1, _ = run_decode(one)
        assert torch.equal(bits(out1), bits(out8[i : i + 1])), i


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_chunked_prefill_equals_full(D):
    """R4 for incremental prefill: a request prefilled in two chunks (the second over the first's cached keys) gives, token for
    token, the bits of the one-shot prefill. Two requests cut at different points (inside a 512-key chunk, at a tile boundary),
    paged and ragged."""
    kv = [1700, 1000]
    cut = [700, 512]
    full = Problem(kv, kv, 32, 8, D, 16, seed=12)
    out_full = bits(run_paged_prefill(full))
    assert torch.equal(out_full, bits(run_ragged_prefill(full)))
    rows = [full.rows(i) for i in range(2)]
    keys = [full.keys(i) for i in range(2)]
    c1 = derived(
        full,
        cut,
        cut,
        torch.cat([full.q[rows[i]][: cut[i]] for i in range(2)]),
        torch.cat([full.k_rag[keys[i]][: cut[i]] for i in range(2)]),
        torch.cat([full.v_rag[keys[i]][: cut[i]] for i in range(2)]),
        seed=13,
    )
    c2 = derived(
        full,
        [kv[i] - cut[i] for i in range(2)],
        kv,
        torch.cat([full.q[rows[i]][cut[i] :] for i in range(2)]),
        full.k_rag,
        full.v_rag,
        seed=14,
    )
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
    """R4 for multi-token decode: ``q_len_per_req = 4`` gives, for the last four tokens of every request, the prefill rows' bits."""
    pr = Problem([256, 33, 1024, 7], [1500, 2500, 1024, 4096], 32, 8, D, 16, seed=15)
    pre = bits(run_paged_prefill(pr))
    dec, last4 = run_decode(pr, q_len=4)
    assert torch.equal(bits(dec), pre[last4])


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_head_subset_invariance(D):
    """R2 across heads: a subset of the KV heads (whole GQA groups) gives, head for head, the bits of the run over all heads."""
    Hq, Hkv = 32, 8
    g = Hq // Hkv
    pr = Problem([300, 1, 1000, 64], [700, 2500, 2500, 3000], Hq, Hkv, D, 16, seed=16)
    full_pre = bits(run_paged_prefill(pr))
    full_dec, _ = run_decode(pr)
    for kv_sel in ([1, 3, 4, 6], [5]):
        q_sel = [j * g + r for j in kv_sel for r in range(g)]
        sub = derived(
            pr,
            pr.q_lens,
            pr.kv_lens,
            pr.q[:, q_sel],
            pr.k_rag[:, kv_sel],
            pr.v_rag[:, kv_sel],
            seed=17,
            Hq=len(q_sel),
            Hkv=len(kv_sel),
        )
        assert torch.equal(bits(run_paged_prefill(sub)), full_pre[:, q_sel]), kv_sel
        assert torch.equal(bits(run_ragged_prefill(sub)), full_pre[:, q_sel]), kv_sel
        dec, _ = run_decode(sub)
        assert torch.equal(bits(dec), bits(full_dec)[:, q_sel]), kv_sel


def _engine_run(pr, cfg, sm_count=None, window_left=-1):
    from flashinfer.attention.batch_invariant import engine

    planned = engine.plan_attention(
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_last,
        pr.Hq,
        pr.Hkv,
        pr.D,
        pr.page,
        window_left,
        cfg,
        None,
        pr.device,
        sm_count=sm_count,
    )
    out = engine.run_attention(
        planned,
        pr.q,
        pr.k_nhd,
        pr.v_nhd,
        pr.kv_indptr,
        pr.kv_indices,
        1.0 / math.sqrt(pr.D),
        window_left,
        "NHD",
        None,
        cfg,
    )
    return planned, out


@needs_device
def test_sm_count_and_grid_do_not_change_bits():
    """R3: the planner's unit kind (fused, split with the fixup kernel, split with the in-kernel fold) depends on the SM count and
    the launch's persistent grid; neither reaches the bits."""
    from flashinfer.attention.batch_invariant import engine

    pr = Problem([256, 1000, 1], [3000, 2500, 8192], 32, 8, 128, 16, seed=7)
    ref = bits(run_paged_prefill(pr))
    cfg = engine.default_config(D=128)
    kinds = set()
    for sm in (1, 8, 148, 1000):
        planned, out = _engine_run(pr, cfg, sm_count=sm)
        kinds.add(bool(planned.plan.groups))
        assert torch.equal(bits(out), ref), sm
    assert kinds == {False, True}  # both the fused and the split units ran
    for grid in (1, 3, 64):
        _, out = _engine_run(pr, replace(cfg, grid=grid))
        assert torch.equal(bits(out), ref), grid


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_config_agreement(D):
    """R3: members of the configuration space (ring depth, unit kind, softmax warpgroups, fold placement, small-unit body) give
    the bits of the wrapper's pick, and so does the paired placement (two Q tiles per CTA) at D = 64, and at D = 128 on 328 KB
    parts. A stride through the space (each distinct kernel is a compile, and the bf16 suite walks the whole space of the same
    kernels): every 5th point at D = 128 / 256, every 13th at D = 64, with every split / fold placement of the paired body."""
    from flashinfer.attention.batch_invariant import engine
    from flashinfer.attention.batch_invariant.swaps.operands import arch_smem_cap

    if D == 64:
        pr = Problem([2048, 2048, 1], [4096, 3000, 1000], 32, 8, D, 16, seed=8)
    else:
        pr = Problem([256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, seed=8)
    ref = bits(run_paged_prefill(pr))
    space = engine.all_valid_configs(D)
    cfgs = list(space[::13] if D == 64 else space[::5])
    assert {c.split_policy for c in cfgs} >= {"split", "fused"}
    assert {c.small for c in cfgs} >= {"swap"} and {c.softmax_wgs for c in cfgs} == {
        1,
        2,
    }
    q2_sm = None
    two = replace(engine.default_config(D=D), softmax_wgs=2, q2=True)
    if D == 64:
        rings = (2, 3) + ((4,) if arch_smem_cap(pr.device) >= (300 << 10) else ())
        cfgs += [
            replace(two, kv_slots=s, split_policy=p, merge=m)
            for s in rings
            for p, m in (("auto", "auto"), ("split", "auto"), ("split", "inkernel"))
        ]
    elif D == 128 and arch_smem_cap(pr.device) >= (300 << 10):
        q2_sm = 1
        cfgs += [
            replace(two, kv_slots=s, split_policy=p, merge=m)
            for s in (2, 3)
            for p, m in (("auto", "auto"), ("split", "auto"), ("split", "inkernel"))
        ]
    for cfg in cfgs:
        planned, out = _engine_run(
            pr, cfg, sm_count=(q2_sm if D == 128 else None) if cfg.q2 else None
        )
        if cfg.q2 and (D == 128 or cfg.split_policy == "auto"):
            assert planned.plan.num_launch is not None, (cfg, planned.plan.num_units)
        assert torch.equal(bits(out), ref), cfg


@needs_device
def test_wrapper_default_pairs_at_d64():
    """The wrappers' prefill pick at D = 64 from GQA group 4 up is the paired placement: its fp16 build gives the one-warpgroup
    body's bits (paged and ragged), on a batch whose units pair."""
    from flashinfer.attention.batch_invariant import engine

    pr = Problem([2048, 2048, 700], [4096, 3000, 1000], 32, 8, 64, 16, seed=18)
    paged = bits(run_paged_prefill(pr))
    assert torch.equal(paged, bits(run_ragged_prefill(pr)))
    _, one = _engine_run(pr, engine.default_config(D=64))
    assert torch.equal(bits(one), paged)


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_sliding_window_prefill_equals_decode(D):
    pr = Problem([256, 1, 2000], [1500, 2500, 5000], 32, 8, D, 16, seed=9)
    pre = run_paged_prefill(pr, window_left=600)
    dec, last = run_decode(pr, window_left=600)
    assert torch.equal(bits(pre[last]), bits(dec))
    assert torch.equal(bits(pre), bits(run_ragged_prefill(pr, window_left=600)))
    ref = reference(pr, window_left=600)
    err = (pre.double() - ref).abs().max().item()
    assert err < 2e-3 + 2e-3 * ref.abs().max().item(), err


@needs_device
def test_fp16_and_bf16_are_different_classes_on_shared_values():
    """Operands representable in both formats (three significand bits) run through both classes: each is close to the reference,
    and the two differ in bits somewhere (binary16 probabilities and output against bf16 ones) -- the reason the format is a class
    dimension and not a placement."""
    pr = Problem([64, 300], [700, 2500], 32, 8, 128, 16, seed=19)
    for name in ("q", "k_rag", "v_rag"):
        t = getattr(pr, name).float()
        setattr(pr, name, (torch.round(t * 4) / 4).to(F16))
    pr.build()
    pb = Problem(pr.q_lens, pr.kv_lens, 32, 8, 128, 16, seed=19, dtype=torch.bfloat16)
    pb.q, pb.k_rag, pb.v_rag = (
        pr.q.to(torch.bfloat16),
        pr.k_rag.to(torch.bfloat16),
        pr.v_rag.to(torch.bfloat16),
    )
    pb.build()
    assert torch.equal(pb.q.float(), pr.q.float())
    assert torch.equal(pb.v_rag.float(), pr.v_rag.float())
    o16, ob = run_paged_prefill(pr), run_paged_prefill(pb)
    assert o16.dtype == F16 and ob.dtype == torch.bfloat16
    ref = reference(pr)
    e16 = (o16.double() - ref).abs().max().item()
    eb = (ob.double() - ref).abs().max().item()
    assert e16 < 4e-3 and eb < 4e-2, (e16, eb)
    assert e16 < eb  # the binary16 roundings are the finer ones
    assert not torch.equal(o16.float(), ob.float())


# ------------------------------------------------------------------------------------------------------ the binary16 range
def _two_level_problem(D, n_low, delta, v_top, v_low, sm_scale_c, q_len=1):
    """One request of ``1 + n_low`` keys and its last ``q_len`` query tokens, one KV head with two query heads: key 0 scores 0
    and the other keys score ``-delta`` (``q = e_0``, ``k_0 = 0``, ``k_j = -delta e_0``), so with ``sm_scale = sm_scale_c ln 2`` the
    probabilities are 1 and ``2^(-delta sm_scale_c)`` up to the last bit of the constant. V is ``v_top`` for key 0 and ``v_low`` for
    the others in every dim. Returns the problem and its ``sm_scale``."""
    L = 1 + n_low
    pr = Problem([q_len], [L], 2, 1, D, 16, seed=21)
    q = torch.zeros(q_len, 2, D)
    q[..., 0] = 1.0
    k = torch.zeros(L, 1, D)
    k[1:, :, 0] = -delta
    v = torch.full((L, 1, D), float(v_low))
    v[0] = float(v_top)
    pr.q, pr.k_rag, pr.v_rag = q.to(F16), k.to(F16), v.to(F16)
    assert torch.equal(pr.k_rag.float(), k) and torch.equal(pr.v_rag.float(), v)
    return pr.build(), sm_scale_c * math.log(2.0)


def _all_paths(pr, sm_scale):
    """The token through ragged prefill, paged prefill and decode: one result, checked equal in bits."""
    out = run_ragged_prefill(pr, sm_scale=sm_scale)
    assert torch.equal(bits(out), bits(run_paged_prefill(pr, sm_scale=sm_scale)))
    dec, last = run_decode(pr, sm_scale=sm_scale)
    assert torch.equal(bits(dec), bits(out[last]))
    return out


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_one_hot_rows_return_the_value_bits(D):
    """Scores far apart (operands far up the binary16 range: the fp32 score accumulation holds them) make every row one-hot:
    the output is the selected key's V, bit for bit, including the largest binary16 number, subnormals and signed values; nothing
    overflows on the way. ``sm_scale = ln 2`` makes the class's constant ``c`` exactly 1, so the winner's exponent argument
    ``fma(s, c, mul(m, nc))`` is exactly zero and its probability exactly 1 (with another scale the argument is the rounding error of
    ``m c``, the weight is ``2^error`` and the output is V to within one binary16 ulp)."""
    L = 300
    pr = Problem([1, L], [L, L], 8, 2, D, 16, seed=22)
    g = torch.Generator().manual_seed(23)
    # request 0: one token at the end; request 1: every position. Key j's direction is e_(j mod D) scaled by a strictly increasing
    # magnitude, the queries are large along every axis: the newest visible key on the query's best axis wins by a wide margin
    k = torch.zeros(2 * L, 2, D)
    for j in range(L):
        k[j, :, j % D] = k[L + j, :, j % D] = 100.0 + 50.0 * j
    q = torch.full((1 + L, 8, D), 200.0)
    v = torch.randn(2 * L, 2, D, generator=g) * 100.0
    v[5] = F16_MAX
    v[L + 7] = -F16_MAX
    v[L + 9] = 2.0**-24  # the smallest binary16 subnormal
    v[L + 11] = -3.0 * 2.0**-20
    v[L - 1, :, : D // 2] = F16_MAX
    pr.q, pr.k_rag, pr.v_rag = q.to(F16), k.to(F16), v.to(F16)
    pr.build()
    out = _all_paths(pr, math.log(2.0))
    assert torch.isfinite(out.float()).all()
    # the winner of position p: the key with the largest 200 * (100 + 50 j) over j <= p, i.e. j = p
    want = torch.cat(
        [
            pr.v_rag[L - 1 : L],
            pr.v_rag[L : 2 * L],
        ]
    ).repeat_interleave(4, dim=1)
    assert torch.equal(bits(out), bits(want))
    assert (out[0, :, : D // 2].float() == F16_MAX).all()
    # the default scale: the same rows to within two binary16 ulps of the selected V (the weight's rounding, then the output's)
    out = run_ragged_prefill(pr).double()
    ulp = torch.pow(
        2.0, torch.floor(torch.log2(want.double().abs().clamp_min(2.0**-14))) - 10
    )
    finite = torch.isfinite(out)
    assert ((out - want.double()).abs()[finite] <= 2.0 * ulp[finite]).all()
    # a non-finite element can only be the overflow of a value within an ulp of the top of the range
    assert (want.double().abs()[~finite] >= F16_MAX - 32.0).all()


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_output_rounding_overflows_to_infinity(D):
    """The class's pinned overflow: every value is the largest binary16 number, so every exact convex combination is 65504, but
    the PV product takes the probabilities rounded to binary16 while the row sum adds the unrounded ones. With keys at
    probability ``0.5 (1 + 0.55 * 2^-10)`` (rounded up to ``0.5 (1 + 2^-10)``) next to one key at probability 1, the accumulator is
    about ``65504 * 1.0004`` times the row sum: above 65520, which ``cvt.rn.f16.f32`` rounds to infinity. Prefill and decode agree
    on the infinities; the mirrored values give ``-inf``; half the magnitude gives the finite, correctly rounded result."""
    c = 1.0 - math.log2(1.0 + 0.55 * 2.0**-10)
    # one tile and one token (the Swaps body); one tile and 40 tokens (the row body: every token sees at least 88 keys); four
    # tiles of one chunk and 64 tokens. One chunk on purpose: a chunk's probabilities are relative to its own maximum, so a later
    # chunk of equal scores has weights of exactly 1 and its share of the row carries no rounding (the merge scales it in fp32)
    for n_low, q_len in ((127, 1), (127, 40), (500, 64)):
        for sign in (1.0, -1.0):
            pr, sm = _two_level_problem(
                D, n_low, 1.0, sign * F16_MAX, sign * F16_MAX, c, q_len
            )
            out = _all_paths(pr, sm)
            assert torch.isinf(out.float()).all(), (n_low, q_len, sign)
            assert (out.float() * sign > 0).all(), (n_low, q_len, sign)
    # the same weights over values with headroom: finite, one binary16 ulp above the exact value (the same upward bias, visible
    # below the top of the range)
    for n_low, q_len in ((127, 1), (127, 40), (500, 64)):
        pr, sm = _two_level_problem(D, n_low, 1.0, 32752.0, 32752.0, c, q_len)
        out = _all_paths(pr, sm)
        assert (out.float() == 32768.0).all(), (n_low, q_len)
    # one visible key: the probability is exactly 1 and the largest binary16 number passes through
    pr, sm = _two_level_problem(D, 0, 1.0, F16_MAX, F16_MAX, c)
    assert (_all_paths(pr, sm).float() == F16_MAX).all()


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
def test_probabilities_underflow_gradually(D):
    """The binary16 rounding of the probabilities: a weight of ``2^-26`` (below half the smallest subnormal) is ``+0`` in the PV
    product, so a value of 65504 behind it leaves the output at the dominant key's 1.0 exactly (the fp32 reference, and the bf16
    class's wider-range weights, would move it by one binary16 ulp); a weight of ``2^-20`` is a binary16 subnormal and is kept."""
    pr, sm = _two_level_problem(D, 1, 26.0, 1.0, F16_MAX, 1.0)
    out = _all_paths(pr, sm)
    assert (out.float() == 1.0).all(), out.float().flatten()[:4]
    exact_ref = reference(pr, sm_scale=sm)
    assert (exact_ref - 1.0).abs().min().item() > 0.9 * 2.0**-10  # the dropped mass
    pr, sm = _two_level_problem(D, 1, 20.0, 1.0, F16_MAX, 1.0)
    out = _all_paths(pr, sm)
    want = (1.0 + F16_MAX * 2.0**-20) / (1.0 + 2.0**-20)
    assert (out.double() - want).abs().max().item() <= 2.0**-10, out.float().flatten()[
        :4
    ]
    assert (out.float() > 1.03).all()


@needs_device
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("Hq", [2, 64], ids=["swaps", "row_body"])
def test_subnormal_operands_reach_the_scores(D, Hq):
    """A binary16 subnormal in the query (or in the key) is an operand like any other: the score MMA multiplies it, it is not
    flushed to zero. Two keys, the first scoring ``2^-20 * 2^15 = 2^-5`` and the second 0; with ``c = 32`` the weights are 1 and
    1/2 and the output is ``3 / 1.5 = 2`` (a flushed operand would give equal weights and 1.5)."""
    for q0, k0 in ((2.0**-20, 32768.0), (32768.0, 2.0**-20)):
        pr = Problem([1], [2], Hq, 1, D, 16, seed=28)
        q = torch.zeros(1, Hq, D)
        q[..., 0] = q0
        k = torch.zeros(2, 1, D)
        k[0, :, 0] = k0
        v = torch.zeros(2, 1, D)
        v[0] = 3.0
        pr.q, pr.k_rag, pr.v_rag = q.to(F16), k.to(F16), v.to(F16)
        assert float(pr.q[0, 0, 0]) == q0 and float(pr.k_rag[0, 0, 0]) == k0
        pr.build()
        out = _all_paths(pr, 32.0 * math.log(2.0))
        assert (out.float() == 2.0).all(), (q0, k0, out.float().flatten()[:4])


@needs_device
def test_large_operands_match_the_reference():
    """Operands spread over the binary16 exponent range (queries and keys up to a few hundred, values from subnormals up to tens of
    thousands, peaked rows): the class stays within its stated bound of the reference, and the bound's absolute term (the
    gradual underflow of the weights) is what large values behind small weights cost."""
    pr = Problem([200, 1], [900, 1500], 16, 4, 128, 16, seed=24)
    g = torch.Generator().manual_seed(25)
    n = pr.v_rag.numel()
    mag = torch.pow(2.0, torch.randint(-24, 15, (n,), generator=g).float())
    v = (torch.randn(n, generator=g) * mag).reshape(pr.v_rag.shape)
    pr.v_rag = v.clamp(-6.0e4, 6.0e4).to(F16)
    pr.q = (pr.q.float().cpu() * 4.0).to(F16)
    pr.k_rag = (pr.k_rag.float().cpu() * 0.5).to(F16)
    pr.build()
    out = run_paged_prefill(pr)
    assert torch.equal(bits(out), bits(run_ragged_prefill(pr)))
    dec, last = run_decode(pr)
    assert torch.equal(bits(dec), bits(out[last]))
    assert torch.isfinite(out.float()).all()
    ref, mag, vsum = reference(pr, with_scale=True)
    assert mag.max().item() > 1.0e4 and ref.abs().max().item() > 1.0e4
    # the class's error bound: the binary16 roundings of the normal weights and of the output are relative (2^-11 of the magnitude
    # the element's terms sum to), the rounding of a weight below 2^-14 is absolute (at most 2^-25 per key, times its value)
    err = (out.double() - ref).abs()
    assert (err <= 1.5e-3 * mag + 2.0**-25 * vsum + 2.0**-24).all(), (
        (err - 1.5e-3 * mag - 2.0**-25 * vsum).max().item()
    )
    # the absolute term is live on these values (small weights over large values): the relative term alone does not hold
    assert (err > 1.5e-3 * mag + 2.0**-24).any()


# ------------------------------------------------------------------------------------------------------ the gate
@needs_device
def test_wrapper_dtype_defaults_select_the_class():
    """The wrappers' own dtype defaults: ``q_data_type`` defaults to fp16 and the KV cache and the output follow the query, so a
    plan without dtype arguments runs the fp16 class; a bf16 query alone runs the bf16 class; one wrapper can be replanned from
    one class to the other."""
    pr = Problem([64, 1], [700, 300], 32, 8, 128, 16, seed=26)
    ref = bits(run_paged_prefill(pr))
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    w.plan(*args, causal=True)
    assert w._batch_invariant.CLASS == "attention.fp16"
    assert torch.equal(bits(w.run(pr.q, pr.cache())), ref)
    w.plan(*args, causal=True, q_data_type="float16", o_data_type=torch.float16)
    assert torch.equal(bits(w.run(pr.q, pr.cache())), ref)
    out = torch.empty_like(pr.q)
    assert w.run(pr.q, pr.cache(), out=out) is out and torch.equal(bits(out), ref)
    # the same wrapper replanned for bf16 operands runs the bf16 class and refuses the fp16 tensors
    w.plan(*args, causal=True, q_data_type=torch.bfloat16)
    assert w._batch_invariant.CLASS == "attention.bf16"
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="attention.bf16"):
        w.run(pr.q, pr.cache())
    k, v = pr.cache()
    ob = w.run(pr.q.to(torch.bfloat16), (k.to(torch.bfloat16), v.to(torch.bfloat16)))
    assert ob.dtype == torch.bfloat16
    wd = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace(pr.device), "NHD", batch_invariant=True
    )
    wd.plan(pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    assert wd._batch_invariant.CLASS == "attention.fp16"
    last = torch.tensor([63, 64], device=pr.device)
    assert torch.equal(bits(wd.run(pr.q[last].contiguous(), pr.cache())), ref[last])


@needs_device
@pytest.mark.parametrize("kind", range(3), ids=["decode", "paged", "ragged"])
def test_mixed_formats_raise_and_nothing_falls_back(kind):
    """Mixed fp16 / bf16 operands are in neither class: every mix raises the typed error at ``plan()``, a tensor of the other
    format raises at ``run()``, and the plan survives the refusals."""
    pr = Problem([1, 1], [700, 300], 32, 8, 128, 16, seed=27)
    w = getattr(flashinfer, WRAPPERS[kind])(
        workspace(pr.device, mb=8), "NHD", batch_invariant=True
    )
    if kind == 0:
        args, kw = (pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16), {}
        cache = (pr.cache(),)
    elif kind == 1:
        args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
        kw, cache = dict(causal=True), (pr.cache(),)
    else:
        args, kw = (pr.qo_indptr, pr.kv_indptr_rag, 32, 8, 128), dict(causal=True)
        cache = pr.ragged()
    bf, f16 = torch.bfloat16, torch.float16
    for q_dt, kv_dt, o_dt in (
        (f16, bf, None),
        (bf, f16, None),
        (f16, None, bf),
        (bf, None, f16),
        (f16, f16, bf),
        (bf, bf, f16),
        ("float16", "bfloat16", None),
        (torch.float32, None, None),
        (f16, torch.float32, None),
        (f16, torch.float8_e4m3fn, None),
        (f16, None, torch.float32),
    ):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            w.plan(*args, q_data_type=q_dt, kv_data_type=kv_dt, o_data_type=o_dt, **kw)
    if (
        kind == 0
    ):  # the decode wrapper takes an absent query dtype: the bf16 class, which an fp16 cache is outside of
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            w.plan(*args, q_data_type=None, kv_data_type=f16, **kw)
    w.plan(*args, q_data_type=f16, kv_data_type=f16, o_data_type=f16, **kw)
    ref = bits(w.run(pr.q, *cache))

    def other(t):
        return t.to(bf)

    def cast(c, i):
        if kind == 2:
            return tuple(other(t) if j == i else t for j, t in enumerate(c))
        return (tuple(other(t) if j == i else t for j, t in enumerate(c[0])),)

    with pytest.raises(bi.BatchInvariantUnsupportedError, match="attention.fp16"):
        w.run(other(pr.q), *cache)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="attention.fp16"):
        w.run(pr.q, *cast(cache, 0))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="attention.fp16"):
        w.run(pr.q, *cast(cache, 1))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="fp16"):
        w.run(pr.q, *cache, out=torch.empty(pr.q.shape, dtype=bf, device=pr.device))
    # the class's other refusals carry the fp16 class's name
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="attention.fp16"):
        w.plan(*args, q_data_type=f16, kv_data_type=f16, logits_soft_cap=30.0, **kw)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="attention.fp16"):
        w.plan(*args, q_data_type=f16, pos_encoding_mode="ROPE_LLAMA", **kw)
    assert torch.equal(bits(w.run(pr.q, *cache)), ref)
