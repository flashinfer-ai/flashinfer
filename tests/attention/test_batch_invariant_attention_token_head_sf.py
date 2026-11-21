# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The token-head-scale classes of the batch-invariant attention (``attention.e4m3kv_th`` / ``attention.e5m2kv_th``): an fp8 KV
cache with one float32 scale per (token, KV head) for K and for V, as ``plan(use_token_head_sf=True)`` /
``run(kv_cache_sf=(k_sf, v_sf))`` take it. Bitwise tests (the mode's acceptance criterion), the device-free checks of the
term and the fingerprints, an accuracy check against an fp64 reference at extreme scales, and the "unsupported raises,
nothing falls back" rule.

The class keeps K and V exact and applies each scale once per key: the score of key ``j`` is ``mul(s_j, ks_j)`` before the
softmax step, the PV operand of key ``j`` is ``cvt_bf16(mul(p_j, vs_j))``, the normaliser adds the unscaled ``p_j``. Its
bitwise oracles are

* the specification itself, evaluated on the CPU with exact binary32 arithmetic (``ExactModel``) on problems whose opaque
  steps (the tensor-core K-steps, ``ex2``) are exact;
* the per-tensor fp8 class where the two classes coincide: every scale equal to 1, or a power of two shared by all keys.

The caches are quantized per (token, KV head) row (``scale = amax / fp8_max`` in float32, ``x / scale`` rounded to fp8), a
function of the row alone: a key has the same fp8 bytes and the same scale in the paged pool, in the packed (ragged) copy and
in any sub-batch.
"""

import math
import re

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.attention.batch_invariant import engine, spec
from flashinfer.attention.batch_invariant.wrappers import ATTENTION_ARCHS
from flashinfer.batch_invariant.fp_interface import exact
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
BF16 = torch.bfloat16
FMT_OF = {E4: "e4m3th", E5: "e5m2th"}
FP8_MAX = {E4: 448.0, E5: 57344.0}
DTYPES = pytest.mark.parametrize("dtype", [E4, E5], ids=["e4m3", "e5m2"])
HEAD_DIMS = pytest.mark.parametrize("D", [64, 128, 256])


# ------------------------------------------------------------------------------------------------------ problems
def quantize(x: torch.Tensor, dtype):
    """``x`` (``[..., D]``) quantized per row: ``(fp8, scale)`` with the float32 scale of shape ``x.shape[:-1]``."""
    sf = (x.float().abs().amax(dim=-1) / FP8_MAX[dtype]).clamp(min=1e-12)
    return (x.float() / sf[..., None]).to(dtype), sf


class THProblem:
    """``Problem`` with its K / V quantized per (token, KV head): the paged pool (junk pages included) and the ragged copy
    hold the same bytes and scales for the live keys."""

    def __init__(self, pr: Problem, dtype):
        self.pr, self.dtype = pr, dtype
        self.k_nhd, self.ksf_nhd = quantize(pr.k_nhd, dtype)
        self.v_nhd, self.vsf_nhd = quantize(pr.v_nhd, dtype)
        self.k_rag, self.ksf_rag = quantize(pr.k_rag, dtype)
        self.v_rag, self.vsf_rag = quantize(pr.v_rag, dtype)

    def cache(self, layout):
        if layout == "NHD":
            return self.k_nhd, self.v_nhd
        return (
            self.k_nhd.transpose(1, 2).contiguous(),
            self.v_nhd.transpose(1, 2).contiguous(),
        )

    def sf(self, layout):
        if layout == "NHD":
            return self.ksf_nhd, self.vsf_nhd
        return (
            self.ksf_nhd.transpose(1, 2).contiguous(),
            self.vsf_nhd.transpose(1, 2).contiguous(),
        )

    def ragged(self, layout):
        if layout == "NHD":
            return self.k_rag, self.v_rag
        return (
            self.k_rag.transpose(0, 1).contiguous(),
            self.v_rag.transpose(0, 1).contiguous(),
        )

    def sf_ragged(self, layout):
        if layout == "NHD":
            return self.ksf_rag, self.vsf_rag
        return (
            self.ksf_rag.transpose(0, 1).contiguous(),
            self.vsf_rag.transpose(0, 1).contiguous(),
        )

    def repack(self, seed=None):
        """Rebuild the paged pool (another page order, fresh junk) from the ragged bytes and scales."""
        pr = self.pr
        fresh = Problem(
            pr.q_lens,
            pr.kv_lens,
            pr.Hq,
            pr.Hkv,
            pr.D,
            pr.page,
            seed=int(torch.randint(0, 1 << 30, (1,)).item()) if seed is None else seed,
        )
        fresh.q, fresh.k_rag, fresh.v_rag = pr.q, pr.k_rag, pr.v_rag
        self.pr = fresh
        self.k_nhd, self.ksf_nhd = quantize(fresh.k_nhd, self.dtype)
        self.v_nhd, self.vsf_nhd = quantize(fresh.v_nhd, self.dtype)
        tok = 0
        for r, L in enumerate(pr.kv_lens):
            n_pages = int(fresh.kv_indptr[r + 1] - fresh.kv_indptr[r])
            for j in range(n_pages):
                pg = int(fresh.kv_indices[int(fresh.kv_indptr[r]) + j])
                n = min(pr.page, L - j * pr.page)
                sl = slice(tok + j * pr.page, tok + j * pr.page + n)
                self.k_nhd[pg, :n], self.ksf_nhd[pg, :n] = (
                    self.k_rag[sl],
                    self.ksf_rag[sl],
                )
                self.v_nhd[pg, :n], self.vsf_nhd[pg, :n] = (
                    self.v_rag[sl],
                    self.vsf_rag[sl],
                )
            tok += L
        return self


def problem(q_lens, kv_lens, Hq, Hkv, D, page, seed, dtype) -> THProblem:
    return THProblem(Problem(q_lens, kv_lens, Hq, Hkv, D, page, seed=seed), dtype)


def subproblem(tp: THProblem, q_lens, kv_lens, q, k, ksf, v, vsf, seed=77) -> THProblem:
    """A problem over given query rows and ragged fp8 keys / scales (a sub-batch, a permutation, a chunk of a prefill, or
    hand-made data)."""
    pr = tp.pr
    s = THProblem.__new__(THProblem)
    s.pr = Problem(q_lens, kv_lens, pr.Hq, pr.Hkv, pr.D, pr.page, seed=seed)
    s.pr.q = q.contiguous()
    s.dtype = tp.dtype
    s.k_rag, s.ksf_rag = k.contiguous(), ksf.contiguous()
    s.v_rag, s.vsf_rag = v.contiguous(), vsf.contiguous()
    return s.repack(seed)


def plan_kw(tp: THProblem):
    return dict(q_data_type=BF16, kv_data_type=tp.dtype, use_token_head_sf=True)


def paged_prefill_wrapper(tp, layout="NHD", window_left=-1, ws=None, **kw):
    pr = tp.pr
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
        **plan_kw(tp),
        **kw,
    )
    return w


def run_paged_prefill(tp, layout="NHD", window_left=-1, ws=None, sf=None, **kw):
    w = paged_prefill_wrapper(tp, layout, window_left, ws, **kw)
    return w.run(
        tp.pr.q, tp.cache(layout), kv_cache_sf=tp.sf(layout) if sf is None else sf
    )


def ragged_prefill_wrapper(tp, layout="NHD", window_left=-1, ws=None, **kw):
    pr = tp.pr
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
        **plan_kw(tp),
        **kw,
    )
    return w


def run_ragged_prefill(tp, layout="NHD", window_left=-1, ws=None, sf=None, **kw):
    w = ragged_prefill_wrapper(tp, layout, window_left, ws, **kw)
    k, v = tp.ragged(layout)
    return w.run(tp.pr.q, k, v, kv_cache_sf=tp.sf_ragged(layout) if sf is None else sf)


def last_rows(pr: Problem) -> torch.Tensor:
    return torch.tensor(
        [int(pr.qo_indptr[i + 1]) - 1 for i in range(len(pr.q_lens))], device=pr.device
    )


def decode_wrapper(tp, layout="NHD", window_left=-1, ws=None, **kw):
    pr = tp.pr
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
        **plan_kw(tp),
        **kw,
    )
    return w


def run_decode(tp, layout="NHD", window_left=-1, ws=None, sf=None, **kw):
    """Decode of the LAST token of every request (q rows picked from the prefill problem)."""
    last = last_rows(tp.pr)
    w = decode_wrapper(tp, layout, window_left, ws, **kw)
    out = w.run(
        tp.pr.q[last].contiguous(),
        tp.cache(layout),
        kv_cache_sf=tp.sf(layout) if sf is None else sf,
    )
    return out, last


def per_tensor(tp: THProblem, kind, k_scale, v_scale, window_left=-1, **kw):
    """The per-tensor fp8 class over the same fp8 bytes with the scalars ``k_scale`` / ``v_scale``."""
    pr = tp.pr
    dt = dict(q_data_type=BF16, kv_data_type=tp.dtype, **kw)
    ws = workspace(pr.device)
    if kind == "ragged":
        w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(
            pr.qo_indptr,
            pr.kv_indptr_rag,
            pr.Hq,
            pr.Hkv,
            pr.D,
            causal=True,
            window_left=window_left,
            **dt,
        )
        return w.run(pr.q, tp.k_rag, tp.v_rag, k_scale=k_scale, v_scale=v_scale)
    paged = (pr.kv_indptr, pr.kv_indices, pr.kv_last, pr.Hq, pr.Hkv, pr.D, pr.page)
    if kind == "decode":
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(*paged, window_left=window_left, **dt)
        q = pr.q[last_rows(pr)].contiguous()
    else:
        w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", batch_invariant=True
        )
        w.plan(pr.qo_indptr, *paged, causal=True, window_left=window_left, **dt)
        q = pr.q
    return w.run(q, tp.cache("NHD"), k_scale=k_scale, v_scale=v_scale)


def reference64(tp: THProblem, window_left=-1, sm_scale=None, parts=False):
    """fp64 causal attention over the dequantized cache (``fp8 * scale`` in fp64): the yardstick for accuracy, never for bits.
    With ``parts``, also returns per output element the sum of the keys' absolute contributions ``sum_j p_j |vs_j v_j|`` (the
    scale of the element's rounding errors) and the part of it that comes from keys whose weight relative to the row's
    maximum is below ``2^-120``: the kernels' ``ex2`` flushes probabilities below ``2^-126`` to zero, so those keys add
    nothing whatever their value scale."""
    pr = tp.pr
    outs, mags, lost = [], [], []
    tok_q = tok_kv = 0
    g = pr.Hq // pr.Hkv
    k_all = tp.k_rag.double() * tp.ksf_rag.double()[..., None]
    v_all = tp.v_rag.double() * tp.vsf_rag.double()[..., None]
    sm = 1.0 / math.sqrt(pr.D) if sm_scale is None else sm_scale
    for ql, kl in zip(pr.q_lens, pr.kv_lens, strict=True):
        q = pr.q[tok_q : tok_q + ql].double()
        k = k_all[tok_kv : tok_kv + kl].repeat_interleave(g, dim=1)
        v = v_all[tok_kv : tok_kv + kl].repeat_interleave(g, dim=1)
        s = torch.einsum("qhd,khd->hqk", q, k) * sm
        pos = torch.arange(kl - ql, kl, device=pr.device)[:, None]
        key = torch.arange(kl, device=pr.device)[None, :]
        mask = key <= pos
        if window_left >= 0:
            mask &= key >= pos - window_left
        s = s.masked_fill(~mask[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v))
        if parts:
            mags.append(torch.einsum("hqk,khd->qhd", p, v.abs()))
            low = (s - s.amax(-1, keepdim=True)) < -120 * math.log(2.0)
            lost.append(torch.einsum("hqk,khd->qhd", p * low, v.abs()))
        tok_q += ql
        tok_kv += kl
    if parts:
        return torch.cat(outs), torch.cat(mags), torch.cat(lost)
    return torch.cat(outs)


# --------------------------------------------------------------------------- the specification evaluated on the CPU
class ExactModel:
    """An ``fp`` backend that EVALUATES the class's specification (``spec.AttentionSpec(..., alg=ExactModel(...))``) for one
    query row over the keys of its KV head: every ``.rn`` primitive is the exact binary32 operation of
    ``fp_interface.exact`` (one round-to-nearest-even of the exact result), ``cvt_bf16`` the exact rounding to bf16. The two
    opaque primitives are evaluated only where they are exact, and anything else raises: a tensor-core K-step must have
    every partial sum of its 16 products and its accumulator representable in binary32 whatever the order of the additions,
    and ``ex2`` takes ``-inf`` or an integer (``2^n``, flushed to ``+0`` below ``2^-126``). On such inputs the value this
    model returns is the class's output bit for bit, with no reference to a kernel."""

    def __init__(self, q, k, v, ksf, vsf, c, v_scale=1.0):
        self.q, self.k, self.v = q, k, v  # [D], [L][D], [L][D] Python floats
        self.ksf, self.vsf, self.c, self.v_scale = ksf, vsf, c, v_scale
        self._steps = {}

    def const(self, x):
        return float(x)

    def sym(self, name):
        return {"c": self.c, "nc": -self.c, "v_scale": self.v_scale}[name]

    add = staticmethod(exact.add)
    mul = staticmethod(exact.mul)
    fma = staticmethod(exact.fma)
    div = staticmethod(exact.div)
    fmaxc = staticmethod(exact.fmaxc)

    def cvt_out(self, x, fmt):
        assert fmt == "bf16"
        return exact.to_format(x, exact.BF16)

    def ex2(self, x):
        if x == -math.inf:
            return 0.0
        if x != int(x):
            raise ValueError(f"ex2({x}): the model evaluates ex2 at integers only")
        return 0.0 if x < -126 else 2.0 ** int(x)

    @staticmethod
    def _exact_sum(terms):
        """The sum of ``terms`` when every partial sum, in any order, is a binary32 number: all terms are multiples of a
        common power of two ``q`` and the sum of their magnitudes is below ``2^24 q``."""
        nz = [t for t in terms if t != 0.0]
        if not nz:
            return 0.0
        quantum = math.inf
        for t in nz:
            m, e = math.frexp(abs(t))
            n = int(m * (1 << 53))
            quantum = min(quantum, math.ldexp(1.0, e - 53 + (n & -n).bit_length() - 1))
        if math.fsum(abs(t) for t in nz) >= quantum * (1 << 24):
            raise ValueError(
                "a tensor-core step of this problem is not exact in binary32"
            )
        return math.fsum(nz)

    def op(self, name, *a):
        if name == "a_rows":  # (row, t): the query's elements of K-step t
            return tuple(self.q[16 * a[1] : 16 * a[1] + 16])
        if name == "b_cols":  # (t, key)
            return tuple(self.k[a[1]][16 * a[0] : 16 * a[0] + 16])
        if (
            name == "v_slice"
        ):  # (vt, d): column d of the 16 keys of PV step vt (zero past the cache)
            keys = range(16 * a[0], 16 * a[0] + 16)
            return tuple(self.v[j][a[1]] if j < len(self.v) else 0.0 for j in keys)
        if name == "kv_widen":
            return a[1]
        if name in ("ab_kslice", "pv_kslice"):
            return (tuple(a[0]), tuple(a[1]))
        if name == "p_slice":
            return tuple(a)
        if name == "sfk_tok":
            return self.ksf[a[0]]
        if name == "sfv_tok":
            return self.vsf[a[0]]
        if name == spec.MMA_OP:
            acc, (x, y) = a
            key = (acc, x, y)
            if key not in self._steps:
                self._steps[key] = self._exact_sum(
                    [acc] + [xi * yi for xi, yi in zip(x, y, strict=True)]
                )
            return self._steps[key]
        raise KeyError(name)


def model_output(
    tp: THProblem, req, row, head, d, kv_fmt, sm_scale, window_left=-1, v_scale=1.0
):
    """The specification's value of output element ``(row of request req, query head, d)`` of ``tp`` for the class
    ``kv_fmt``, as bf16 bits (an int16)."""
    pr = tp.pr
    g = pr.Hq // pr.Hkv
    q0, k0 = int(pr.qo_indptr[req]), int(pr.kv_indptr_rag[req])
    kl, ql = pr.kv_lens[req], pr.q_lens[req]
    kh = head // g
    model = ExactModel(
        pr.q[q0 + row, head].float().tolist(),
        tp.k_rag[k0 : k0 + kl, kh].float().tolist(),
        tp.v_rag[k0 : k0 + kl, kh].float().tolist(),
        tp.ksf_rag[k0 : k0 + kl, kh].tolist(),
        tp.vsf_rag[k0 : k0 + kl, kh].tolist(),
        spec.scale_log2(sm_scale),
        v_scale,
    )
    val = spec.AttentionSpec(pr.D, alg=model, kv_fmt=kv_fmt).output(
        spec.RowSpec(p=kl - ql + row, window_left=window_left, row=0), d
    )
    return torch.tensor(val, dtype=torch.float32).to(BF16).view(torch.int16).item()


SM_EXACT = math.log(
    2.0
)  # sm_scale with c = fp32(sm_scale * log2 e) = 1: integer scores give integer ex2 arguments
K_SCALES = (1.0, 1.0, -1.0, 2.0, -2.0, 0.0, 0.5, 0.25, -0.5)
V_SCALES = (1.0, 0.5, 2.0, -1.0, 0.0, 1.5, 0.75, 1.25, -1.75, 3.0, 0.625)


def exact_problem(q_lens, kv_lens, Hq, Hkv, D, page, seed, dtype, v_scales=V_SCALES):
    """A problem on which the opaque steps are exact: one-hot queries (a score is one key element), small integer keys and
    values, key scales that keep the scaled scores integers within [-2, 2] (so the probabilities are powers of two within
    2^-4 of the maximum) and value scales of at most three significant bits -- or, with ``v_scales=None`` on short rows,
    random 24-bit ones in [0.5, 1.5) of either sign: the PV operand's rounding to bf16 is then exercised."""
    tp = problem(q_lens, kv_lens, Hq, Hkv, D, page, seed, dtype)
    dev = tp.pr.device
    g = torch.Generator().manual_seed(seed)
    total_q, total_kv = sum(q_lens), sum(kv_lens)
    q = torch.zeros(total_q, Hq, D)
    q.scatter_(2, torch.randint(0, D, (total_q, Hq, 1), generator=g), 1.0)
    ks = torch.tensor(K_SCALES)[
        torch.randint(0, len(K_SCALES), (total_kv, Hkv), generator=g)
    ]
    k = torch.randint(-1, 2, (total_kv, Hkv, D), generator=g).float()
    k = (
        k
        / torch.where(ks.abs() < 1, ks.abs().clamp(min=0.25), torch.ones(()))[..., None]
    )
    v = torch.randint(-2, 3, (total_kv, Hkv, D), generator=g).float()
    if v_scales is None:
        vs = (torch.rand(total_kv, Hkv, generator=g) + 0.5) * (
            1.0 - 2.0 * torch.randint(0, 2, (total_kv, Hkv), generator=g)
        )
    else:
        vs = torch.tensor(v_scales)[
            torch.randint(0, len(v_scales), (total_kv, Hkv), generator=g)
        ]
    return subproblem(
        tp,
        q_lens,
        kv_lens,
        q.to(BF16).to(dev),
        k.to(dtype).to(dev),
        ks.to(dev),
        v.to(dtype).to(dev),
        vs.to(dev),
        seed=seed + 1,
    )


def strip_scale(sx: str, leaf: str) -> str:
    """``sx`` with every ``(mul X (<leaf> n))`` replaced by ``X``."""
    pat = re.compile(r" \(" + leaf + r"(?: \d+)?\)\)")
    while True:
        m = pat.search(sx)
        if m is None:
            return sx
        depth, i = 1, m.start()  # walk back to the "(" of this mul
        while depth:
            i -= 1
            depth += {")": 1, "(": -1}.get(sx[i], 0)
        assert sx.startswith("(mul ", i), sx[i : i + 20]
        sx = sx[:i] + sx[i + 5 : m.start()] + sx[m.end() :]


# ------------------------------------------------------------------------------------------------------ device-free
RECORDED = {  # the digests of the classes that existed before the token-head-scale classes: unchanged
    ("bf16", 64): "attn-bf16-d64-9a05233ec9351b0d",
    ("bf16", 128): "attn-bf16-d128-027b0379515b82e2",
    ("bf16", 256): "attn-bf16-d256-b8cb59dab4eb6c0f",
    ("e4m3", 64): "attn-e4m3kv-d64-a00dd1765f9182b4",
    ("e4m3", 128): "attn-e4m3kv-d128-a961e59f805ec08c",
    ("e4m3", 256): "attn-e4m3kv-d256-7fedccc342c5f71a",
    ("e5m2", 64): "attn-e5m2kv-d64-a8fdf71de8b7c88c",
    ("e5m2", 128): "attn-e5m2kv-d128-ee18bcf735786c12",
    ("e5m2", 256): "attn-e5m2kv-d256-7e342886ee623ba2",
    ("nvfp4", 64): "attn-nvfp4kv-d64-d73d0adce1e215f9",
    ("nvfp4", 128): "attn-nvfp4kv-d128-dd667c5f850bd4fe",
    ("nvfp4", 256): "attn-nvfp4kv-d256-ac5d34ddab6116f6",
}


TOKEN_HEAD = {
    ("e4m3th", 64): "attn-e4m3kv_th-d64-908c71010bb5f2ca",
    ("e4m3th", 128): "attn-e4m3kv_th-d128-b8295843d4f279df",
    ("e4m3th", 256): "attn-e4m3kv_th-d256-98230582d54343b8",
    ("e5m2th", 64): "attn-e5m2kv_th-d64-e43c08dc89c2932c",
    ("e5m2th", 128): "attn-e5m2kv_th-d128-323f8ddcb4211475",
    ("e5m2th", 256): "attn-e5m2kv_th-d256-1d7db487bb80abfb",
}


@pytest.mark.parametrize("op", WRAPPERS)
def test_fingerprints(op):
    """One fingerprint per (head dim, fp8 format) of the token-head-scale classes, shared by the three wrappers; the bf16,
    per-tensor fp8 and NVFP4 digests are the recorded ones."""
    for (fmt, D), want in RECORDED.items():
        assert spec.spec_fingerprint(D, kv_fmt=fmt) == want
        assert bi.fingerprint(op, head_dim=D, causal=True, kv_dtype=fmt) == want
        assert (
            bi.fingerprint(
                op, head_dim=D, causal=True, kv_dtype=fmt, use_token_head_sf=False
            )
            == want
        )
    for D in (64, 128, 256):
        assert bi.fingerprint(op, head_dim=D, causal=True) == RECORDED[("bf16", D)]
        assert (
            bi.fingerprint(op, head_dim=D, causal=True, kv_dtype=E4)
            == RECORDED[("e4m3", D)]
        )
        assert (
            bi.fingerprint(op, head_dim=D, causal=True, kv_dtype=E5)
            == RECORDED[("e5m2", D)]
        )
        assert (
            bi.fingerprint(op, head_dim=D, causal=True, kv_dtype=torch.uint8)
            == RECORDED[("nvfp4", D)]
        )
        for dtype in (E4, E5):
            fmt = FMT_OF[dtype]
            th = bi.fingerprint(
                op, head_dim=D, causal=True, kv_dtype=dtype, use_token_head_sf=True
            )
            assert th == TOKEN_HEAD[(fmt, D)] == spec.spec_fingerprint(D, kv_fmt=fmt)
            assert th == bi.fingerprint(op, head_dim=D, causal=True, kv_dtype=fmt)
            assert th == bi.fingerprint(
                op, head_dim=D, causal=True, kv_dtype=fmt[:4], use_token_head_sf=True
            )
            assert th.startswith(f"attn-{fmt[:4]}kv_th-d{D}-")
    assert len(set(RECORDED.values()) | set(TOKEN_HEAD.values())) == 18
    for bad in (None, BF16, torch.uint8, "nvfp4", torch.float16):
        with pytest.raises(ValueError):
            bi.fingerprint(
                op, head_dim=128, causal=True, kv_dtype=bad, use_token_head_sf=True
            )
    with pytest.raises(TypeError):
        bi.fingerprint(op, head_dim=128, causal=True, kv_dtype=E4, token_head=True)
    assert bi.class_name(op) == "attention.bf16"
    assert spec.class_name("e4m3th") == "attention.e4m3kv_th"
    assert spec.class_name("e5m2th") == "attention.e5m2kv_th"
    assert spec.class_name("e4m3") == "attention.e4m3kv"
    assert spec.class_name("nvfp4") == "attention.nvfp4kv"
    assert spec.class_name() == "attention.bf16"


def test_term_scales_each_key_once():
    """The class's term is the per-tensor fp8 class's (exactly widened leaves) with the per-tensor scale removed from the
    epilogue and, for every VISIBLE key ``j``, one ``mul`` of its score by ``(sfk j)`` and one ``mul`` of its probability by
    ``(sfvt j)`` inside the operand's ``cvt bf16``; the row sum and the normaliser hold no scale, and a masked key has none."""
    from flashinfer.batch_invariant.fp_interface.term import sexpr

    for D, T, C, p, win in (
        (64, 16, 32, 40, -1),
        (128, 16, 32, 33, 5),
        (64, 32, 64, 40, 20),
    ):
        row = spec.RowSpec(p=p, window_left=win, row=1)
        bf, th4, th5, e4 = (
            sexpr(spec.AttentionSpec(D, T=T, C=C, kv_fmt=f).output(row, 2))
            for f in ("bf16", "e4m3th", "e5m2th", "e4m3")
        )
        assert th4.replace("e4m3", "e5m2") == th5 and th4 != th5
        assert th4 != e4 and th4 != bf and "v_scale" not in th4
        assert "(widen e4m3 (b " in th4 and "(widen e4m3 (op:v_slice " in th4
        # the scales named by the term are the visible keys', nothing else
        visible = set(range(max(0, p - win) if win >= 0 else 0, p + 1))
        assert {int(n) for n in re.findall(r"\(sfk (\d+)\)", th4)} == visible
        assert {int(n) for n in re.findall(r"\(sfvt (\d+)\)", th4)} == visible
        # a key scale multiplies the score of its own key; a value scale sits inside the operand's rounding
        own = re.findall(r"\(widen e4m3 \(b \d+ (\d+)\)\)\)\) \(sfk (\d+)\)", th4)
        assert own and all(a == b for a, b in own)
        assert len(own) == th4.count("(sfk ")
        operands = re.findall(r"\(cvt bf16 \(mul \(ex2 ", th4)
        assert len(operands) == th4.count("(sfvt ") > 0
        # without the scales it is the per-tensor term without its value scale
        assert strip_scale(strip_scale(th4, "sfk"), "sfvt") == strip_scale(
            e4, "sym v_scale"
        )
    # c is the bf16 class's constant: no key scale is folded
    assert spec.scale_log2(0.125) == spec.to_f32(0.125 / math.log(2.0))
    assert spec.scale_log2(SM_EXACT) == 1.0


def test_configuration_space_and_kernel_identity():
    """The token-head-scale kernels take the low-bit space (one softmax warpgroup, the row body for small units); their names
    carry the variant and the existing kernels keep theirs."""
    from flashinfer.attention.batch_invariant.blocks.compile import kernel_name

    for D in (64, 128, 256):
        for fmt in ("e4m3th", "e5m2th"):
            cfg = engine.default_config(D, fmt)
            assert cfg.well_formed(D, lowbit=True) and cfg.small == "rep"
            assert cfg == engine.default_config(D, "e4m3")
    assert kernel_name(128, 4, 16, 4) == "attn_d128_g4_p16_s4_w1"
    assert kernel_name(128, 4, 16, 4, kv_fmt="e4m3") == "attn_d128_g4_p16_s4_w1_kve4m3"
    assert kernel_name(128, 4, 16, 4, kv_fmt="e5m2") == "attn_d128_g4_p16_s4_w1_kve5m2"
    assert (
        kernel_name(128, 4, 16, 4, kv_fmt="nvfp4") == "attn_d128_g4_p16_s4_w1_kvnvfp4"
    )
    assert (
        kernel_name(64, 4, 16, 4, 2, True, False) == "attn_d64_g4_p16_s4_w2_q2"
        and kernel_name(128, 4, 128, 4, ragged=True) == "attn_d128_g4_p128_s4_w1_rg"
    )
    th4 = kernel_name(128, 4, 16, 4, kv_fmt="e4m3th")
    th5 = kernel_name(128, 4, 16, 4, kv_fmt="e5m2th")
    assert th4 == "attn_d128_g4_p16_s4_w1_kve4m3th" and th5.endswith("_kve5m2th")
    assert (
        kernel_name(128, 4, 128, 4, ragged=True, kv_fmt="e4m3th")
        == "attn_d128_g4_p128_s4_w1_rg_kve4m3th"
    )
    assert engine.kv_format(E4, True) == "e4m3th" and engine.kv_format(E5) == "e5m2"
    for dt in (BF16, torch.uint8):
        with pytest.raises(ValueError):
            engine.kv_format(dt, True)


def test_exact_model_evaluates_the_specification():
    """The CPU evaluation of the specification on a hand-sized row: two visible keys with scores 1 and 1 scaled by 2 and -1,
    value scales 1.5 and 0.75; a masked key's scales (NaN here) are never read; with every scale 1 the value is the
    per-tensor class's; inputs on which an opaque step would round are refused."""
    D = 64
    q = [1.0] + [0.0] * (D - 1)
    k = [[1.0] + [0.0] * (D - 1), [1.0] + [0.0] * (D - 1), [1.0] * D]
    v = [[2.0] * D, [-1.0] * D, [1.0] * D]
    ks, vs = [2.0, -1.0, math.nan], [1.5, 0.75, math.nan]

    def out(kv_fmt, ks, vs, p, d=3):
        model = ExactModel(q, k, v, ks, vs, 1.0)
        return spec.AttentionSpec(D, alg=model, kv_fmt=kv_fmt).output(
            spec.RowSpec(p=p), d
        )

    # s' = (2, -1): p = (1, 2^-3), l = 1.125, O = 1.5 * 2 - 0.75 * 2^-3 (both operands are bf16 values)
    want = exact.to_format(
        exact.add(exact.mul(3.0 - 0.09375, exact.div(1.0, 1.125)), 0.0), exact.BF16
    )
    assert out("e4m3th", ks, vs, 1) == want
    assert abs(want - 2.90625 / 1.125) < 1e-2
    one = [1.0] * 3
    assert out("e4m3th", one, one, 2) == out("e4m3", one, one, 2)
    with pytest.raises(ValueError):  # a non-integer exponent
        out("e4m3th", [0.3, 1.0, 1.0], one, 1)
    with pytest.raises(ValueError):  # a step whose partial sums need more than 24 bits
        ExactModel._exact_sum([1.0, 2.0**-30])


def test_cache_key_covers_the_conversion_stage():
    from flashinfer.batch_invariant.jit import relative_sources

    src = relative_sources("batch_invariant_attention")
    assert "attention/batch_invariant/blocks/convert.py" in src
    assert "attention/batch_invariant/blocks/softmax.py" in src


# ------------------------------------------------------------------------------------------------------ on the device
@needs_device
@HEAD_DIMS
@DTYPES
def test_matches_reference(D, dtype):
    tp = problem([300, 1000, 64], [700, 2500, 3000], 32, 8, D, 16, 1, dtype)
    out = run_paged_prefill(tp)
    ref = reference64(tp)
    err = (out.double() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err


@needs_device
@DTYPES
@pytest.mark.parametrize("D", [64, 128])
def test_specification_evaluated_on_the_cpu_is_the_bitwise_oracle(D, dtype):
    """The kernels compute the term: on problems whose opaque steps are exact, the output equals the specification evaluated
    on the CPU (``ExactModel``) bit for bit -- rows across tile and chunk boundaries, with and without a window, with
    negative, zero and fractional key and value scales, through the three wrappers. The same model reproduces the
    per-tensor fp8 class on these problems, which validates the model against a class that existed before."""
    q_lens, kv_lens = [70, 3, 640], [300, 131, 640]
    tp = exact_problem(q_lens, kv_lens, 8, 4, D, 16, 300 + D, dtype)
    pr = tp.pr
    fmt = FMT_OF[dtype]
    sm = dict(sm_scale=SM_EXACT)
    picks = [  # (request, row, query head, d)
        (0, 0, 0, 0),
        (0, 26, 3, D - 1),
        (0, 69, 5, 17),
        (1, 0, 7, 1),
        (1, 2, 2, D // 2),
        (2, 0, 1, 5),
        (2, 127, 4, 33),
        (2, 128, 6, 2),
        (2, 511, 0, 63),
        (2, 512, 3, 40),
        (2, 639, 7, 9),
    ]
    for window_left in (-1, 150):
        pre = run_paged_prefill(tp, window_left=window_left, **sm)
        rag = run_ragged_prefill(tp, "HND", window_left=window_left, **sm)
        assert torch.equal(bits(pre), bits(rag))
        dec, last = run_decode(tp, window_left=window_left, **sm)
        assert torch.equal(bits(dec), bits(pre)[last])
        got = bits(pre)
        for req, row, head, d in picks:
            want = model_output(tp, req, row, head, d, fmt, SM_EXACT, window_left)
            at = (window_left, req, row, head, d)
            assert got[int(pr.qo_indptr[req]) + row, head, d].item() == want, at
    # the model against the per-tensor class (v_scale = 0.75 multiplies its epilogue)
    pt = bits(per_tensor(tp, "prefill", 1.0, 0.75, **sm))
    for req, row, head, d in picks[::2]:
        want = model_output(tp, req, row, head, d, fmt[:4], SM_EXACT, v_scale=0.75)
        assert pt[int(pr.qo_indptr[req]) + row, head, d].item() == want


@needs_device
@DTYPES
@pytest.mark.parametrize("shift", [0, -125])
def test_operand_rounding_against_the_cpu_specification(dtype, shift):
    """Short rows with 24-bit value scales: the PV operand ``cvt_bf16(mul(p_j, vs_j))`` rounds (ties included: half of the
    scales are bf16 midpoints) and the output still equals the CPU evaluation of the specification bit for bit. With the
    scales moved down by ``2^-125`` the operands are bf16 subnormals (and zeros): the rounding the two operations define
    there is the kernels' too."""
    q_lens, kv_lens = [9, 14, 1, 16], [9, 14, 5, 16]
    tp = exact_problem(q_lens, kv_lens, 4, 2, 64, 16, 77, dtype, v_scales=None)
    half = tp.vsf_rag.clone().view(torch.int32)
    half[::2] = (half[::2] & ~0xFFFF) | 0x8000
    tp.vsf_rag = half.view(torch.float32) * 2.0**shift
    tp.repack(seed=78)
    pr = tp.pr
    got = bits(run_ragged_prefill(tp, sm_scale=SM_EXACT))
    assert torch.equal(got, bits(run_paged_prefill(tp, sm_scale=SM_EXACT)))
    assert (got != 0).any()
    n = 0
    for req in range(4):
        for row in range(q_lens[req]):
            head, d = (row + req) % 4, (7 * row + req) % 64
            want = model_output(tp, req, row, head, d, FMT_OF[dtype], SM_EXACT)
            at = (req, row, head, d)
            assert got[int(pr.qo_indptr[req]) + row, head, d].item() == want, at
            n += 1
    assert n == 40


@needs_device
@HEAD_DIMS
@DTYPES
def test_unit_and_uniform_power_of_two_scales_are_the_per_tensor_class(D, dtype):
    """Where the two classes coincide. With every scale equal to 1 both multiplies are identities: the output is the
    per-tensor fp8 class's with ``k_scale = v_scale = 1``. With one power of two for every key scale the scaled scores are
    the scores times that power exactly, which the per-tensor class folds into ``c``; with one power of two for every value
    scale the operands, the accumulator and the merges scale exactly, which the per-tensor class applies in its epilogue.
    Bit for bit, for the three wrappers."""
    tp = problem([256, 1, 1000, 33], [1500, 2500, 1000, 4096], 32, 8, D, 16, 2, dtype)
    one_k, one_v = torch.ones_like(tp.ksf_nhd), torch.ones_like(tp.vsf_nhd)
    one_kr, one_vr = torch.ones_like(tp.ksf_rag), torch.ones_like(tp.vsf_rag)
    for ks, vs in (
        (1.0, 1.0),
        (0.25, 1.0),
        (1.0, 8.0),
        (2.0**-7, 2.0**5),
        (64.0, 2.0**-9),
    ):
        want = bits(per_tensor(tp, "prefill", ks, vs))
        assert torch.equal(bits(per_tensor(tp, "ragged", ks, vs)), want)
        sf = (one_k * ks, one_v * vs)
        assert torch.equal(bits(run_paged_prefill(tp, sf=sf)), want), (ks, vs)
        sfr = (one_kr * ks, one_vr * vs)
        assert torch.equal(bits(run_ragged_prefill(tp, sf=sfr)), want), (ks, vs)
        dec, last = run_decode(tp, sf=sf)
        assert torch.equal(bits(dec), want[last]), (ks, vs)
        assert torch.equal(bits(dec), bits(per_tensor(tp, "decode", ks, vs)))
    # sm_scale is the per-tensor class's too
    want = bits(per_tensor(tp, "prefill", 1.0, 1.0, sm_scale=0.11))
    got = run_paged_prefill(tp, sf=(one_k, one_v), sm_scale=0.11)
    assert torch.equal(bits(got), want)


def extreme_problem(dtype):
    """Scales no quantizer would write, inside the class's statement: value scales spread over fifty decades per token, one
    request moved as a whole towards each end, key scales spread over a factor of 256, random signs on both."""
    q_lens, kv_lens = [200, 150, 150, 1], [700, 400, 150, 900]
    tp = problem(q_lens, kv_lens, 8, 2, 128, 16, 90, dtype)
    g = torch.Generator().manual_seed(91)
    dev = tp.pr.device
    n = tp.vsf_rag.shape
    tp.vsf_rag = tp.vsf_rag * (10.0 ** ((torch.rand(n, generator=g) - 0.5) * 50)).to(
        dev
    )
    tp.vsf_rag[700:1100] *= 1e-10
    tp.vsf_rag[1100:1250] *= 1e8
    tp.vsf_rag *= 1.0 - 2.0 * (torch.rand(n, generator=g) < 0.3).float().to(dev)
    tp.ksf_rag = tp.ksf_rag * (2.0 ** ((torch.rand(n, generator=g) - 0.5) * 8)).to(dev)
    tp.ksf_rag *= 1.0 - 2.0 * (torch.rand(n, generator=g) < 0.2).float().to(dev)
    return tp.repack(seed=92)


#: The accuracy the tests hold the classes to at extreme scales, relative to ``sum_j p_j |vs_j v_j|`` per output element.
EXTREME_TOL = 2e-2


def extreme_error(tp: THProblem, out, rows=None):
    """The largest error of ``out`` against the fp64 reference in units of the element's ``sum_j p_j |vs_j v_j|``, after
    the two absolute terms of the class's statement: the contributions of keys whose probability the kernels flush to zero
    (below ``2^-126`` of the row's maximum), and half a bf16 subnormal step per operand."""
    ref, mag, lost = reference64(tp, parts=True)
    assert ref.isfinite().all() and ref.abs().max() < 3e38
    if rows is not None:
        ref, mag, lost = ref[rows], mag[rows], lost[rows]
    floor = lost + max(tp.pr.kv_lens) * 2.0**-134 * FP8_MAX[tp.dtype]
    excess = ((out.double() - ref).abs() - floor).clamp(min=0)
    return (excess / mag.clamp(min=1e-300)).max().item()


@needs_device
@DTYPES
def test_accuracy_at_extreme_scales(dtype):
    """Extreme but finite scales against an fp64 reference, for the three wrappers: every output element is within
    ``EXTREME_TOL`` of the sum of its keys' absolute contributions ``sum_j p_j |vs_j v_j|`` -- the bf16 rounding of each
    operand and of the output and the fp32 scores, relative to what was added -- once the contributions of the keys the
    softmax flushes (a probability below ``2^-126`` is ``+0`` whatever the key's value scale) are set aside. The scales
    stay inside the class's statement: scaled scores and operands finite."""
    tp = extreme_problem(dtype)
    assert tp.vsf_rag.abs().max() > 1e20 and tp.vsf_rag.abs().min() < 1e-25
    for out in (run_paged_prefill(tp), run_ragged_prefill(tp, "HND")):
        assert out.isfinite().all()
        assert extreme_error(tp, out) < EXTREME_TOL
    dec, last = run_decode(tp)
    assert extreme_error(tp, dec, last) < EXTREME_TOL
    # value scales alone, key scales alone: nothing is flushed that matters, the plain relative bound holds
    for name in ("ksf_rag", "vsf_rag"):
        one = extreme_problem(dtype)
        setattr(one, name, quantize(getattr(one.pr, name[0] + "_rag"), dtype)[1])
        one.repack(seed=93)
        ref, mag, _ = reference64(one, parts=True)
        out = run_paged_prefill(one)
        rel = ((out.double() - ref).abs() / mag.clamp(min=1e-300)).max().item()
        assert rel < EXTREME_TOL, (name, rel)


@needs_device
def test_flushed_probability_drops_the_key_whatever_its_value_scale():
    """The floor of the softmax weights, which per-token value scales make visible: ``ex2`` returns ``+0`` below ``2^-126``,
    so a key whose scaled score is more than 126 / c below the row's maximum has the operand ``+0`` and adds nothing --
    even when its value scale is so large that, in exact arithmetic, it would dominate the row. Here key 0 outweighs every
    other key by about ``2^196`` and carries the value scale ``2^-125``; the others carry ``2^120``. The output is the
    one with the other keys' value scales set to zero, bit for bit, and it is key 0's scaled value; the fp64 reference,
    which keeps weights down to ``2^-1074``, is larger by orders of magnitude."""
    L, D, H = 300, 128, 2
    tp = problem([L], [L], H, H, D, 16, 96, E4)
    dev = tp.pr.device
    g = torch.Generator().manual_seed(97)
    q = (1.0 + torch.rand(L, H, D, generator=g)).to(BF16)
    k = torch.zeros(L, H, D)
    k[0] = 8.0
    v = torch.randint(1, 201, (L, H, D), generator=g).float()
    ks = torch.ones(L, H)
    vs = torch.full((L, H), 2.0**120)
    vs[0] = 2.0**-125
    tp = subproblem(
        tp,
        [L],
        [L],
        q.to(dev),
        k.to(E4).to(dev),
        ks.to(dev),
        v.to(E4).to(dev),
        vs.to(dev),
        seed=98,
    )
    out = run_paged_prefill(tp)
    assert torch.equal(bits(out), bits(run_ragged_prefill(tp)))
    ref, mag, lost = reference64(tp, parts=True)
    assert (
        lost[1:] > 0.99 * mag[1:]
    ).all()  # every key but key 0 is below the floor, and they carry the reference
    assert (ref[1:].abs().amax(-1) > 2.0**20 * out[1:].double().abs().amax(-1)).all()
    zeroed = tp.vsf_rag.clone()
    zeroed[1:] = 0.0
    tp.vsf_rag = zeroed
    tp.repack(seed=99)
    assert torch.equal(bits(run_paged_prefill(tp)), bits(out))
    key0 = (tp.v_rag[0].float() * 2.0**-125 + 0.0).to(
        BF16
    )  # p_0 = 1, l = 1: the row is key 0's operand times its value
    assert torch.equal(bits(out[1:]), bits(key0.expand(L - 1, H, D)))


@needs_device
@DTYPES
def test_ends_of_the_scale_range(dtype):
    """Value scales of ``2^100`` keep every operand finite (``p_j <= 1``) and give exactly ``2^100`` times the unit-scale
    output. Value scales of ``2^-120`` put most operands in the bf16 subnormal range or at zero: the output stays within
    the operands' rounding of the fp64 reference. A zero key scale on every key gives the softmax of a constant row."""
    tp = problem([120, 1], [400, 700], 8, 2, 128, 16, 95, dtype)
    vsf_rag = tp.vsf_rag
    one_k, one_v = torch.ones_like(tp.ksf_nhd), torch.ones_like(tp.vsf_nhd)
    base = run_paged_prefill(tp, sf=(tp.ksf_nhd, one_v))
    big = run_paged_prefill(tp, sf=(tp.ksf_nhd, one_v * 2.0**100))
    assert big.isfinite().all()
    assert torch.equal(bits(big), bits((base.float() * 2.0**100).to(BF16)))
    tiny = run_paged_prefill(tp, sf=(tp.ksf_nhd, one_v * 2.0**-120))
    tp.vsf_rag = torch.full_like(vsf_rag, 2.0**-120)
    assert tiny.isfinite().all() and (tiny != 0).any()
    # an operand is within 2^-9 of itself (normal) or half a subnormal step 2^-134 of p_j vs_j: extreme_error's floor
    assert extreme_error(tp, tiny) < EXTREME_TOL
    flat = run_paged_prefill(tp, sf=(one_k * 0.0, tp.vsf_nhd))
    tp.vsf_rag, tp.ksf_rag = vsf_rag, torch.zeros_like(tp.ksf_rag)
    ref0 = reference64(tp)
    err = (flat.double() - ref0).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref0.abs().max().item(), err


@needs_device
@HEAD_DIMS
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_prefill_equals_decode(D, layout):
    """R4: the last token of every request gives the same bits as a prefill row and as a decode step."""
    tp = problem([256, 1, 1024, 33], [1500, 2500, 1024, 4096], 32, 8, D, 16, 2, E4)
    pre = run_paged_prefill(tp, layout)
    dec, last = run_decode(tp, layout)
    assert torch.equal(bits(pre[last]), bits(dec))


@needs_device
@HEAD_DIMS
@pytest.mark.parametrize("page", [1, 16, 128])
def test_paged_equals_ragged(D, page):
    """R4: the paged and the ragged caches hand the kernel the same bytes and the same scales at the same positions, in both
    layouts and at every page size."""
    tp = problem([300, 1000, 64, 2048], [700, 2500, 3000, 2048], 32, 8, D, page, 3, E5)
    ref = bits(run_paged_prefill(tp))
    assert torch.equal(ref, bits(run_ragged_prefill(tp)))
    assert torch.equal(ref, bits(run_paged_prefill(tp, "HND")))
    assert torch.equal(ref, bits(run_ragged_prefill(tp, "HND")))


@needs_device
@pytest.mark.parametrize("page", [2, 4, 8, 32, 64])
def test_page_sizes(page):
    tp = problem([300, 77], [1300, 2100], 8, 2, 128, page, 30 + page, E4)
    ref = bits(run_ragged_prefill(tp))
    assert torch.equal(ref, bits(run_paged_prefill(tp)))
    assert torch.equal(ref, bits(run_paged_prefill(tp, "HND")))
    dec, last = run_decode(tp, "HND")
    assert torch.equal(ref[last], bits(dec))


def inline_slots(x8: torch.Tensor, sf: torch.Tensor):
    """The fp8 rows and their scales packed into one slot tensor (last dim ``D + 16``: the fp8 bytes, the float32 scale, padding)
    and sliced back out as non-contiguous views sharing the slot's storage."""
    D = x8.shape[-1]
    slot = torch.zeros(*x8.shape[:-1], D + 16, dtype=torch.uint8, device=x8.device)
    slot[..., :D] = x8.view(torch.uint8)
    slot[..., D : D + 4] = sf.unsqueeze(-1).view(torch.uint8)
    slot[..., D + 4 :] = (
        0xFF  # padding that would be a NaN if it were ever read as a scale or a value
    )
    return slot.view(x8.dtype)[..., :D], slot[..., D : D + 4].view(
        torch.float32
    ).squeeze(-1)


@needs_device
@HEAD_DIMS
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_noncontiguous_scale_views(D, layout):
    """The scales are read with their own strides: views sliced out of an inline slot tensor (last dim ``head_dim + 16``, the
    cache itself a view of the same slots), one channel of a wider float tensor, and a transposed (not materialised) layout
    give the bits of the contiguous tensors -- prefill and decode, paged and ragged."""
    tp = problem([300, 1, 513], [1100, 2100, 513], 16, 4, D, 16, 40, E4)
    pr = tp.pr
    ref = bits(run_paged_prefill(tp, layout))
    (k, v), (ksf, vsf) = tp.cache(layout), tp.sf(layout)
    kq, ksq = inline_slots(k, ksf)
    vq, vsq = inline_slots(v, vsf)
    assert not ksq.is_contiguous() and not kq.is_contiguous()
    assert ksq.stride(-1) == (D + 16) // 4
    w = paged_prefill_wrapper(tp, layout)
    assert torch.equal(bits(w.run(pr.q, (kq, vq), kv_cache_sf=(ksq, vsq))), ref)
    # contiguous caches with slot-view scales, and the reverse
    assert torch.equal(bits(w.run(pr.q, (k, v), kv_cache_sf=(ksq, vsq))), ref)
    assert torch.equal(bits(w.run(pr.q, (kq, vq), kv_cache_sf=(ksf, vsf))), ref)
    # one channel of a wider tensor
    wide_k = torch.full((*ksf.shape, 3), float("nan"), device=pr.device)
    wide_v = torch.full((*vsf.shape, 3), float("nan"), device=pr.device)
    wide_k[..., 1], wide_v[..., 2] = ksf, vsf
    assert torch.equal(
        bits(w.run(pr.q, (k, v), kv_cache_sf=(wide_k[..., 1], wide_v[..., 2]))), ref
    )
    # the other layout's scale tensor, transposed without a copy
    other = tp.sf("HND" if layout == "NHD" else "NHD")
    views = tuple(t.transpose(1, 2) for t in other)
    assert not views[0].is_contiguous()
    assert torch.equal(bits(w.run(pr.q, (k, v), kv_cache_sf=views)), ref)
    last = last_rows(pr)
    wd = decode_wrapper(tp, layout)
    dec = wd.run(pr.q[last].contiguous(), (kq, vq), kv_cache_sf=(ksq, vsq))
    assert torch.equal(bits(dec), ref[last])
    # ragged: slot views and a transposed scale tensor
    (kr, vr), (ksr, vsr) = tp.ragged(layout), tp.sf_ragged(layout)
    krq, ksrq = inline_slots(kr, ksr)
    vrq, vsrq = inline_slots(vr, vsr)
    wr = ragged_prefill_wrapper(tp, layout)
    assert torch.equal(bits(wr.run(pr.q, krq, vrq, kv_cache_sf=(ksrq, vsrq))), ref)
    other = tp.sf_ragged("HND" if layout == "NHD" else "NHD")
    assert torch.equal(
        bits(wr.run(pr.q, kr, vr, kv_cache_sf=tuple(t.transpose(0, 1) for t in other))),
        ref,
    )


@needs_device
def test_expanded_scale_tensor():
    """A scale tensor with zero strides (one value expanded over the cache) is the materialised tensor."""
    tp = problem([200, 1], [900, 700], 8, 2, 128, 16, 41, E4)
    pr = tp.pr
    ksf = torch.full_like(tp.ksf_nhd, 0.004)
    vsf = torch.full_like(tp.vsf_nhd, 0.007)
    ref = bits(run_paged_prefill(tp, sf=(ksf, vsf)))
    ke = torch.tensor(0.004, device=pr.device).expand(ksf.shape)
    ve = torch.tensor(0.007, device=pr.device).expand(vsf.shape)
    assert ke.stride() == (0, 0, 0)
    assert torch.equal(bits(run_paged_prefill(tp, sf=(ke, ve))), ref)
    dec, last = run_decode(tp, sf=(ke, ve))
    assert torch.equal(bits(dec), ref[last])


@needs_device
@HEAD_DIMS
def test_batch_permutation_and_subbatch_invariance(D):
    """R2: a request's bits do not depend on its neighbours, its position in the batch, the batch size or the pool's pages."""
    q_lens, kv_lens = [300, 1000, 64, 2048, 1], [700, 2500, 3000, 2048, 5000]
    tp = problem(q_lens, kv_lens, 32, 8, D, 16, 5, E4)
    pr = tp.pr
    full = bits(run_paged_prefill(tp))
    assert torch.equal(bits(run_ragged_prefill(tp)), full)
    rows = [
        slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1]))
        for i in range(len(q_lens))
    ]
    kv_rows = [
        slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1]))
        for i in range(len(q_lens))
    ]

    def sub(sel):
        cat = lambda t, r: torch.cat([t[r[i]] for i in sel])  # noqa: E731
        return subproblem(
            tp,
            [q_lens[i] for i in sel],
            [kv_lens[i] for i in sel],
            cat(pr.q, rows),
            cat(tp.k_rag, kv_rows),
            cat(tp.ksf_rag, kv_rows),
            cat(tp.v_rag, kv_rows),
            cat(tp.vsf_rag, kv_rows),
            seed=50 + len(sel) + sel[0],
        )

    for i in (0, 2, 4):
        s = sub([i])
        assert torch.equal(bits(run_ragged_prefill(s)), full[rows[i]]), i
        assert torch.equal(bits(run_paged_prefill(s)), full[rows[i]]), i
    perm = [3, 0, 4, 1, 2]
    s = sub(perm)
    for out in (bits(run_ragged_prefill(s)), bits(run_paged_prefill(s, "HND"))):
        off = 0
        for i in perm:
            n = q_lens[i]
            assert torch.equal(out[off : off + n], full[rows[i]]), i
            off += n


@needs_device
@HEAD_DIMS
def test_decode_batch_invariance(D):
    """R2 for decode: a request's step gives the same bits alone and in a batch of eight, whatever pages the pool assigns and
    whatever the junk (bytes and scales) of the other pages."""
    kv = [1024, 4096, 33, 700, 2500, 8192, 1, 1500]
    tp = problem([1] * 8, kv, 32, 8, D, 16, 6, E4)
    pr = tp.pr
    out8, _ = run_decode(tp)
    assert torch.equal(bits(out8), bits(run_ragged_prefill(tp)))
    for i in (0, 1, 5, 6):
        sl = slice(int(pr.kv_indptr_rag[i]), int(pr.kv_indptr_rag[i + 1]))
        one = subproblem(
            tp,
            [1],
            [kv[i]],
            pr.q[i : i + 1],
            tp.k_rag[sl],
            tp.ksf_rag[sl],
            tp.v_rag[sl],
            tp.vsf_rag[sl],
            seed=100 + i,
        )
        out1, _ = run_decode(one)
        assert torch.equal(bits(out1), bits(out8[i : i + 1])), i


@needs_device
@HEAD_DIMS
def test_chunked_prefill_equals_full(D):
    """R4 for incremental prefill: a request prefilled in two chunks (the second over the first's cached keys and scales)
    gives, token for token, the bits of the one-shot prefill; paged and ragged."""
    kv = [1700, 1000]
    cut = [700, 512]
    tp = problem(kv, kv, 32, 8, D, 16, 12, E5)
    pr = tp.pr
    out_full = bits(run_paged_prefill(tp))
    assert torch.equal(out_full, bits(run_ragged_prefill(tp)))
    rows = [slice(int(pr.qo_indptr[i]), int(pr.qo_indptr[i + 1])) for i in range(2)]
    off = [int(pr.kv_indptr_rag[i]) for i in range(2)]
    head = [slice(off[i], off[i] + cut[i]) for i in range(2)]
    cat = lambda t, r: torch.cat([t[r[i]] for i in range(2)])  # noqa: E731
    c1 = subproblem(
        tp,
        cut,
        cut,
        torch.cat([pr.q[rows[i]][: cut[i]] for i in range(2)]),
        cat(tp.k_rag, head),
        cat(tp.ksf_rag, head),
        cat(tp.v_rag, head),
        cat(tp.vsf_rag, head),
        seed=13,
    )
    c2 = subproblem(
        tp,
        [kv[i] - cut[i] for i in range(2)],
        kv,
        torch.cat([pr.q[rows[i]][cut[i] :] for i in range(2)]),
        tp.k_rag,
        tp.ksf_rag,
        tp.v_rag,
        tp.vsf_rag,
        seed=14,
    )
    for prob, part in ((c1, 0), (c2, 1)):
        out_p = bits(run_paged_prefill(prob))
        out_r = bits(run_ragged_prefill(prob))
        o = 0
        for i in range(2):
            n = cut[i] if part == 0 else kv[i] - cut[i]
            ref = (
                out_full[rows[i]][: cut[i]]
                if part == 0
                else out_full[rows[i]][cut[i] :]
            )
            assert torch.equal(out_p[o : o + n], ref), (D, part, i, "paged")
            assert torch.equal(out_r[o : o + n], ref), (D, part, i, "ragged")
            o += n


@needs_device
@HEAD_DIMS
def test_multi_token_decode_equals_prefill(D):
    """R4 for multi-token decode: ``q_len_per_req = 4`` gives the last four tokens of every request the prefill's bits."""
    kv = [1500, 4, 2500, 1028]
    tp = problem(kv, kv, 32, 8, D, 16, 15, E4)
    pr = tp.pr
    pre = bits(run_paged_prefill(tp))
    rows = torch.cat(
        [
            torch.arange(int(pr.qo_indptr[i + 1]) - 4, int(pr.qo_indptr[i + 1]))
            for i in range(4)
        ]
    ).to(pr.device)
    w = decode_wrapper(tp, q_len_per_req=4)
    out = w.run(pr.q[rows].contiguous(), tp.cache("NHD"), kv_cache_sf=tp.sf("NHD"))
    assert torch.equal(bits(out), pre[rows])


@needs_device
@DTYPES
def test_sliding_window(dtype):
    """``window_left`` masks after the key scaling: prefill == decode == ragged under a window, the window changes the bits,
    the fp64 reference agrees, and with unit scales the windowed output is the per-tensor class's."""
    tp = problem([256, 1, 2000], [1500, 2500, 5000], 32, 8, 128, 16, 9, dtype)
    pre = run_paged_prefill(tp, window_left=600)
    dec, last = run_decode(tp, window_left=600)
    assert torch.equal(bits(pre)[last], bits(dec))
    assert torch.equal(bits(pre), bits(run_ragged_prefill(tp, window_left=600)))
    assert not torch.equal(bits(pre), bits(run_paged_prefill(tp)))
    ref = reference64(tp, window_left=600)
    err = (pre.double() - ref).abs().max().item()
    assert err < 2e-2 + 1e-2 * ref.abs().max().item(), err
    one = (torch.ones_like(tp.ksf_nhd), torch.ones_like(tp.vsf_nhd))
    assert torch.equal(
        bits(run_paged_prefill(tp, window_left=600, sf=one)),
        bits(per_tensor(tp, "prefill", 1.0, 1.0, window_left=600)),
    )


@needs_device
@HEAD_DIMS
@DTYPES
def test_config_agreement(D, dtype):
    """R3: every member of the low-bit configuration space (ring depth, unit kind, fold placement) and every SM count the
    planner may be told give the wrapper's bits."""
    tp = problem([256, 1, 1000, 1], [3000, 2500, 1000, 5000], 32, 8, D, 16, 8, dtype)
    pr = tp.pr
    ref = bits(run_paged_prefill(tp))
    fmt = FMT_OF[dtype]

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
            **kw,
        )
        return engine.run_attention(
            planned,
            pr.q,
            tp.k_nhd,
            tp.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            1.0 / math.sqrt(D),
            -1,
            "NHD",
            None,
            cfg,
            kv_cache_sf=(tp.ksf_nhd, tp.vsf_nhd),
        )

    for cfg in engine.all_valid_configs(D, lowbit=True):
        assert torch.equal(bits(run(cfg)), ref), cfg
    for sm in (1, 8, 1000):
        assert torch.equal(bits(run(engine.default_config(D, fmt), sm_count=sm)), ref)
    single = engine.attention_paged(
        pr.q,
        tp.k_nhd,
        tp.v_nhd,
        pr.qo_indptr,
        pr.kv_indptr,
        pr.kv_indices,
        pr.kv_last,
        kv_cache_sf=(tp.ksf_nhd, tp.vsf_nhd),
        use_token_head_sf=True,
    )
    assert torch.equal(bits(single), ref)


@needs_device
@pytest.mark.parametrize("window_left", [-1, 300])
def test_scales_are_inputs_seen_by_the_rows_that_see_the_token(window_left):
    """A token's scales are inputs of exactly the rows whose mask holds the token: changing the K scale or the V scale of one
    (token, KV head) -- to another finite value, to zero, or to a NaN or an infinity -- changes rows of that request and head
    group at or after the token (and within the window), and no other row: not the earlier tokens of the request (which
    share its tile), not the other head groups, not the other requests. A NaN scale, or an infinite value scale, on a
    visible key is outside the class's statement and makes exactly the rows that see it non-finite."""
    q_lens, kv_lens = [900, 700], [900, 700]
    Hq, Hkv, D = 8, 2, 128
    g = Hq // Hkv
    tp = problem(q_lens, kv_lens, Hq, Hkv, D, 16, 60, E4)
    base = bits(run_ragged_prefill(tp, window_left=window_left)).clone()
    tok, head = 411, 1  # a token of request 0 (inside a tile, not at a boundary)
    hi = q_lens[0] if window_left < 0 else min(q_lens[0], tok + window_left + 1)
    for name in ("ksf_rag", "vsf_rag"):
        saved = getattr(tp, name).clone()
        for value in (None, -1.0, 0.0, math.nan, math.inf):
            setattr(tp, name, saved.clone())
            if value is None:
                getattr(tp, name)[tok, head] *= 1.75
            else:
                getattr(tp, name)[tok, head] = value
            tp.repack(seed=61)
            at = (name, value)
            for out in (
                run_ragged_prefill(tp, window_left=window_left),
                run_paged_prefill(tp, window_left=window_left),
            ):
                changed = (bits(out) != base).any(dim=-1)  # (total_q, Hq)
                sees = torch.zeros_like(changed)
                sees[tok:hi, head * g : (head + 1) * g] = True
                assert not (changed & ~sees).any(), at
                assert changed[tok, head * g : (head + 1) * g].all(), at
                assert out[~sees].isfinite().all(), at
                if value is None or value == -1.0:
                    # every row that sees the token moves (its softmax weight on the token is far above the bf16 rounding)
                    assert changed[sees].float().mean() > 0.9, at
                    assert out.isfinite().all(), at
                elif math.isnan(value) or name == "vsf_rag" and math.isinf(value):
                    assert not out[sees].isfinite().all(-1).any(), at
        setattr(tp, name, saved)
    tp.repack(seed=62)
    assert torch.equal(bits(run_paged_prefill(tp, window_left=window_left)), base)


@needs_device
@pytest.mark.parametrize("kind", ["prefill", "decode", "ragged"])
def test_junk_scales_and_bytes_never_reach_a_row(kind):
    """Keys past a request's length -- the rest of its last page, the pages no request references, the packed tokens past the
    planned total -- are never visible: NaN / infinite scales and NaN bytes there leave every bit unchanged."""
    tp = problem([300, 1, 77], [700, 2501, 1029], 8, 2, 128, 16, 70, E4)
    pr = tp.pr
    dev = pr.device
    if kind == "ragged":
        ref = bits(run_ragged_prefill(tp))
        pad = 300
        k = torch.cat(
            [tp.k_rag, torch.full((pad, 2, 128), float("nan"), device=dev).to(E4)]
        )
        v = torch.cat(
            [tp.v_rag, torch.full((pad, 2, 128), float("nan"), device=dev).to(E4)]
        )
        assert k[-1].float().isnan().all()
        ksf = torch.cat([tp.ksf_rag, torch.full((pad, 2), float("nan"), device=dev)])
        vsf = torch.cat([tp.vsf_rag, torch.full((pad, 2), float("inf"), device=dev)])
        w = ragged_prefill_wrapper(tp)
        assert torch.equal(bits(w.run(pr.q, k, v, kv_cache_sf=(ksf, vsf))), ref)
        return
    live = torch.zeros(tp.ksf_nhd.shape[:2], dtype=torch.bool, device=dev)
    for r, L in enumerate(pr.kv_lens):
        for j in range(int(pr.kv_indptr[r + 1] - pr.kv_indptr[r])):
            pg = int(pr.kv_indices[int(pr.kv_indptr[r]) + j])
            live[pg, : min(16, L - j * 16)] = True
    assert (~live).any()
    run = run_paged_prefill if kind == "prefill" else lambda t: run_decode(t)[0]
    ref = bits(run(tp))
    tp.ksf_nhd[~live] = float("nan")
    tp.vsf_nhd[~live] = float("inf")
    tp.k_nhd.view(torch.uint8)[~live] = 0x7F  # the e4m3 NaN code
    tp.v_nhd.view(torch.uint8)[~live] = 0x7F
    assert torch.equal(bits(run(tp)), ref)
    tp.ksf_nhd[~live] = float("-inf")
    tp.vsf_nhd[~live] = float("nan")
    assert torch.equal(bits(run(tp)), ref)


@needs_device
def test_determinism_and_graph_replay():
    """A captured ``run()`` replays the planned batch; the scale tensors are read at every replay, like the cache."""
    tp = problem([512, 7, 1000], [4096, 1025, 1000], 32, 8, 128, 16, 4, E4)
    pr = tp.pr
    ref = bits(run_paged_prefill(tp)).clone()
    for _ in range(2):
        assert torch.equal(bits(run_paged_prefill(tp)), ref)
    w = paged_prefill_wrapper(tp)
    out = torch.empty_like(pr.q)
    ksf, vsf = tp.ksf_nhd.clone(), tp.vsf_nhd.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        w.run(pr.q, tp.cache("NHD"), out=out, kv_cache_sf=(ksf, vsf))
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            w.run(pr.q, tp.cache("NHD"), out=out, kv_cache_sf=(ksf, vsf))
    torch.cuda.synchronize()
    for _ in range(2):
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(bits(out), ref)
    # new scale values in the captured tensors: the replay computes with them
    ksf *= 0.5
    vsf *= 3.0
    want = bits(run_paged_prefill(tp, sf=(ksf, vsf))).clone()
    assert not torch.equal(want, ref)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(out), want)
    # decode
    last = last_rows(pr)
    wd = decode_wrapper(tp)
    qd = pr.q[last].contiguous()
    outd = torch.empty_like(qd)
    with torch.cuda.stream(s):
        wd.run(qd, tp.cache("NHD"), out=outd, kv_cache_sf=(ksf, vsf))
        gd = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gd, stream=s):
            wd.run(qd, tp.cache("NHD"), out=outd, kv_cache_sf=(ksf, vsf))
    torch.cuda.synchronize()
    outd.zero_()
    gd.replay()
    torch.cuda.synchronize()
    assert torch.equal(bits(outd), want[last])


# ------------------------------------------------------------------------------------------------------ refusals
def _unsupported(match=None):
    return pytest.raises(bi.BatchInvariantUnsupportedError, match=match)


@needs_device
def test_plan_refusals():
    """``use_token_head_sf`` is a class of fp8 caches with bf16 queries: every other combination raises at ``plan()`` for the
    three wrappers, and nothing is planned."""
    tp = problem([64], [700], 32, 8, 128, 16, 10, E4)
    pr = tp.pr
    ws = workspace(pr.device)
    paged = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    planners = {
        "prefill": (
            flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                ws, "NHD", batch_invariant=True
            ),
            paged,
            dict(causal=True),
        ),
        "decode": (
            flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                ws, "NHD", batch_invariant=True
            ),
            (pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16),
            {},
        ),
        "ragged": (
            flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                ws, "NHD", batch_invariant=True
            ),
            (pr.qo_indptr, pr.kv_indptr_rag, 32, 8, 128),
            dict(causal=True),
        ),
    }
    for kind, (w, args, kw) in planners.items():
        for q_dt, kv_dt in (
            (BF16, BF16),  # a bf16 cache has no scales
            (BF16, torch.uint8),  # NVFP4 has block scales, not token-head scales
            (BF16, torch.float16),
            (
                torch.float16,
                E4,
            ),  # the regular path's fp16 queries are not a class of the mode
            (E4, E4),
        ):
            with _unsupported():
                w.plan(
                    *args,
                    **kw,
                    use_token_head_sf=True,
                    q_data_type=q_dt,
                    kv_data_type=kv_dt,
                )
            with pytest.raises(RuntimeError, match="plan"):
                w.run(pr.q, pr.k_rag, pr.v_rag) if kind == "ragged" else w.run(
                    pr.q, pr.cache("NHD")
                )
        with _unsupported("use_token_head_sf"):
            w.plan(
                *args, **kw, use_token_head_sf=True, q_data_type=BF16, kv_data_type=BF16
            )
        with _unsupported("use_token_head_sf"):
            w.plan(
                *args,
                **kw,
                use_token_head_sf=True,
                q_data_type=BF16,
                kv_data_type=torch.uint8,
            )
        with _unsupported():
            w.plan(
                *args,
                **kw,
                use_token_head_sf=True,
                q_data_type=BF16,
                kv_data_type=E4,
                o_data_type=torch.float16,
            )
        if kind != "ragged":  # the swizzled V block-scale layout is NVFP4's
            with _unsupported("v_sf_swizzled"):
                w.plan(
                    *args,
                    **kw,
                    use_token_head_sf=True,
                    v_sf_swizzled=True,
                    q_data_type=BF16,
                    kv_data_type=E4,
                )
        with _unsupported():  # an unknown dtype argument is a typed refusal too
            w.plan(
                *args,
                **kw,
                use_token_head_sf=True,
                q_data_type=BF16,
                kv_data_type="float8",
            )
        # the supported plan names the class
        w.plan(*args, **kw, use_token_head_sf=True, q_data_type=BF16, kv_data_type=E4)
        assert w._batch_invariant.CLASS == "attention.e4m3kv_th", kind


@needs_device
@pytest.mark.parametrize("kind", ["prefill", "decode", "ragged"])
def test_run_refusals(kind):
    """``run()`` of a token-head-scale plan takes the fp8 caches with ``kv_cache_sf=(k_sf, v_sf)``: float32, on the device, of
    the cache's shape minus the head dim, disjoint from ``out``. Anything else raises the typed error; nothing falls back, and
    the plan still runs afterwards."""
    tp = problem([1] if kind == "decode" else [64], [700], 32, 8, 128, 16, 10, E4)
    pr = tp.pr
    dev = pr.device
    if kind == "ragged":
        w = ragged_prefill_wrapper(tp)
        k, v = tp.ragged("NHD")
        ksf, vsf = tp.sf_ragged("NHD")
        run = lambda *a, **kw: w.run(pr.q, k, v, *a, **kw)  # noqa: E731
        run_kv = lambda k_, v_, **kw: w.run(pr.q, k_, v_, **kw)  # noqa: E731
    else:
        w = paged_prefill_wrapper(tp) if kind == "prefill" else decode_wrapper(tp)
        k, v = tp.cache("NHD")
        ksf, vsf = tp.sf("NHD")
        run = lambda *a, **kw: w.run(pr.q, (k, v), *a, **kw)  # noqa: E731
        run_kv = lambda k_, v_, **kw: w.run(pr.q, (k_, v_), **kw)  # noqa: E731
    sf = (ksf, vsf)
    ref = bits(run(kv_cache_sf=sf)).clone()
    with _unsupported("kv_cache_sf"):  # no scales
        run()
    with _unsupported("kv_cache_sf"):  # one tensor, three tensors, a stacked tensor
        run(kv_cache_sf=ksf)
    with _unsupported("kv_cache_sf"):
        run(kv_cache_sf=(ksf,))
    with _unsupported("kv_cache_sf"):
        run(kv_cache_sf=(ksf, vsf, vsf))
    with _unsupported("kv_cache_sf"):
        run(kv_cache_sf=torch.stack(sf, dim=1))
    with _unsupported("kv_cache_sf"):
        run(kv_cache_sf=(ksf, None))
    with _unsupported("kv_cache_sf"):
        run(kv_cache_sf=(ksf, 1.0))
    for bad in (torch.float64, torch.float16, BF16, torch.int32):  # dtype
        with _unsupported("float32"):
            run(kv_cache_sf=(ksf.to(bad), vsf))
        with _unsupported("float32"):
            run(kv_cache_sf=(ksf, vsf.to(bad)))
    with _unsupported("float32"):  # NVFP4-style block scales
        run(kv_cache_sf=(torch.zeros(*k.shape[:-1], 8, dtype=E4, device=dev),) * 2)
    for bad in (  # shape
        ksf[..., None],
        ksf[:-1],
        ksf[..., :-1],
        ksf.transpose(-1, -2),
        ksf.flatten(),
        ksf[None],
        k.float(),
    ):
        with _unsupported("shape"):
            run(kv_cache_sf=(bad, vsf))
        with _unsupported("shape"):
            run(kv_cache_sf=(ksf, bad))
    with _unsupported():  # device
        run(kv_cache_sf=(ksf.cpu(), vsf))
    # the per-tensor scales are not operands of the class: unit values (what serving engines pass; host numbers or host
    # tensors) are accepted and change nothing, any other value raises
    unit = run(kv_cache_sf=sf, q_scale=1.0, k_scale=1.0, v_scale=torch.ones(1))
    assert torch.equal(bits(unit), ref)
    with _unsupported("k_scale"):
        run(kv_cache_sf=sf, k_scale=2.0)
    with _unsupported("v_scale"):
        run(kv_cache_sf=sf, v_scale=2.0)
    with _unsupported("k_scale"):
        run(kv_cache_sf=sf, k_scale=torch.tensor([0.5]), v_scale=1.0)
    with _unsupported("q_scale"):
        run(kv_cache_sf=sf, q_scale=0.5)
    with _unsupported():  # the other fp8 format, a bf16 cache, an NVFP4 cache
        run_kv(k.view(E5), v.view(E5), kv_cache_sf=sf)
    with _unsupported():
        run_kv(k.float().to(BF16), v.float().to(BF16), kv_cache_sf=sf)
    with _unsupported():
        run_kv(k.view(torch.uint8), v.view(torch.uint8), kv_cache_sf=sf)
    with _unsupported():
        run_kv(k, v.view(E5), kv_cache_sf=sf)
    with _unsupported():
        run(kv_cache_sf=sf, return_lse=True)
    # an output that shares bytes with a scale tensor
    arena = torch.zeros(pr.q.numel() // 2 + ksf.numel() + 64, device=dev)
    out = arena[: pr.q.numel() // 2].view(BF16).view(pr.q.shape)
    inside = arena[8 : 8 + ksf.numel()].view(ksf.shape)
    inside.copy_(ksf)
    with _unsupported("alias"):
        run(kv_cache_sf=(inside, vsf), out=out)
    with _unsupported("alias"):
        run(kv_cache_sf=(ksf, inside), out=out)
    # a disjoint slice of the same allocation is fine
    after = arena[pr.q.numel() // 2 + 16 : pr.q.numel() // 2 + 16 + ksf.numel()].view(
        ksf.shape
    )
    after.copy_(ksf)
    assert torch.equal(bits(run(kv_cache_sf=(after, vsf), out=out)), ref)
    assert torch.equal(bits(run(kv_cache_sf=sf)), ref)


@needs_device
def test_other_classes_refuse_token_head_scales():
    """Float32 (token, head) scales on a plan without ``use_token_head_sf`` raise: the per-tensor fp8 classes, the bf16 class
    and the NVFP4 class do not take them; the engine entry points raise ``ValueError`` for the same inputs."""
    tp = problem([64], [700], 32, 8, 128, 16, 11, E4)
    pr = tp.pr
    ws = workspace(pr.device)
    args = (pr.qo_indptr, pr.kv_indptr, pr.kv_indices, pr.kv_last, 32, 8, 128, 16)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", batch_invariant=True)
    w.plan(*args, causal=True, q_data_type=BF16, kv_data_type=E4)
    with _unsupported("kv_cache_sf"):
        w.run(pr.q, tp.cache("NHD"), kv_cache_sf=tp.sf("NHD"))
    w.plan(*args, causal=True, q_data_type=BF16, kv_data_type=BF16)
    with _unsupported("kv_cache_sf"):
        w.run(pr.q, pr.cache("NHD"), kv_cache_sf=tp.sf("NHD"))
    w.plan(*args, causal=True, q_data_type=BF16, kv_data_type=torch.uint8)
    with _unsupported():
        w.run(
            pr.q,
            (
                tp.k_nhd.view(torch.uint8)[..., :64],
                tp.v_nhd.view(torch.uint8)[..., :64],
            ),
            kv_cache_sf=tp.sf("NHD"),
        )
    sm = 1.0 / math.sqrt(128)
    planned = engine.plan_attention(
        pr.qo_indptr, pr.kv_indptr, pr.kv_last, 32, 8, 128, 16, kv_fmt="e4m3th"
    )
    run = lambda k, v, **kw: engine.run_attention(  # noqa: E731
        planned, pr.q, k, v, pr.kv_indptr, pr.kv_indices, sm, **kw
    )
    sf = tp.sf("NHD")
    ref = bits(run_paged_prefill(tp))
    assert torch.equal(bits(run(tp.k_nhd, tp.v_nhd, kv_cache_sf=sf)), ref)
    unit = run(tp.k_nhd, tp.v_nhd, kv_cache_sf=sf, k_scale=1.0, v_scale=1.0)
    assert torch.equal(bits(unit), ref)
    for kw in (
        {},
        dict(kv_cache_sf=sf[0]),
        dict(kv_cache_sf=sf, k_scale=2.0),
        dict(kv_cache_sf=sf, v_scale=0.5),
        dict(kv_cache_sf=(sf[0].double(), sf[1])),
        dict(kv_cache_sf=(sf[0], sf[1][:-1])),
        dict(kv_cache_sf=(sf[0].cpu(), sf[1])),
    ):
        with pytest.raises(ValueError):
            run(tp.k_nhd, tp.v_nhd, **kw)
    with pytest.raises(ValueError):  # a bf16 cache over a token-head-scale plan
        run(pr.k_nhd, pr.v_nhd, kv_cache_sf=sf)
    with pytest.raises(ValueError):  # the other fp8 format
        run(tp.k_nhd.view(E5), tp.v_nhd.view(E5), kv_cache_sf=sf)
    per_tensor = engine.plan_attention(
        pr.qo_indptr, pr.kv_indptr, pr.kv_last, 32, 8, 128, 16, kv_fmt="e4m3"
    )
    with pytest.raises(ValueError):
        engine.run_attention(
            per_tensor,
            pr.q,
            tp.k_nhd,
            tp.v_nhd,
            pr.kv_indptr,
            pr.kv_indices,
            sm,
            kv_cache_sf=sf,
        )
    with pytest.raises(ValueError):  # a per-tensor run over a token-head-scale plan
        run(tp.k_nhd, tp.v_nhd, k_scale=0.01)
