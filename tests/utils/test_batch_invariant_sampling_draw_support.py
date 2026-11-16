# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``flashinfer.sampling.top_k_sampling_from_probs`` / ``top_p_sampling_from_probs`` / ``min_p_sampling_from_probs`` /
``top_k_top_p_sampling_from_probs`` / ``top_k_top_p_sampling_from_logits(..., batch_invariant=True)``: the truncated
draw -- the support class ``support.row.v1``'s kept set, then the draw class ``sample.icdf.v1`` over the kept weights, in
one launch (``draw_support_kernel.py``; the generic entry ``draw_support_api.draw_support``).

Device-free: the pinned receipts (the two classes' fingerprints joined; the dimensions move them, the parameter values do
not), the exact model is the support model then the draw model over the masked row, the error contract, the kernel's PTX
(only the declared forms). On an SM100 / SM103 / SM107 device with the CuTe DSL: tokens, validity and the kept mass bitwise
against the exact model for probability and logits inputs (with the drawn token's log-probability under the full
distribution and under the kept set), f32 / bf16, every filter set and both orders, across single-CTA / cluster / staged
members and the dispatcher's picks; the composition equals the support's mask then the plain draw over the masked weights,
the five public entries equal the generic entry (the logits entry with its temperature and its two log-probabilities), the
token's full-distribution log-probability is ``log_probs`` at the token and the kept one the support's renormalised
log-probability there (R4); the index gather; the
generator-only call shape (the pair reserved as the operator's own path reserves it: 32 per row for the rejection
samplers, one for min-p) and the key forms; batch invariance the way the operator's kernels were probed (a row alone == at
slot 5 of 64 == at slot 5 of 8; a permutation moves no token) and sub-batches; degenerate parameters and invalid rows;
the drawn distribution against the kept set's renormalisation; SM-count independence where the picks differ;
determinism, CUDA-graph replay and PDL; ``batch_invariant=False`` is the operator's kernel, byte for byte; 3-D inputs,
views, M = 0; the big vocabulary.
"""

import os

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.batch_invariant.sampling import draw_reference as DR
from flashinfer.batch_invariant.sampling import schedule as C
from flashinfer.batch_invariant.sampling import stream
from flashinfer.batch_invariant.sampling import support_schedule as DS
from flashinfer.batch_invariant.sampling import support_spec as SS
from flashinfer.batch_invariant.sampling.draw_api import (
    sampling_from_probs_batch_invariant,
)
from flashinfer.batch_invariant.sampling.draw_support_api import (
    CLASS,
    draw_support,
    min_p_sampling_from_probs_batch_invariant,
    top_k_sampling_from_probs_batch_invariant,
    top_k_top_p_sampling_from_logits_batch_invariant,
    top_k_top_p_sampling_from_probs_batch_invariant,
    top_p_sampling_from_probs_batch_invariant,
)
from flashinfer.batch_invariant.sampling.schedule import device_sm_count
from flashinfer.batch_invariant.sampling.support_api import (
    SUPPORT_ARCHS,
    sampling_support,
)
from flashinfer.batch_invariant.sampling.support_reference import support_row
from flashinfer.batch_invariant.sampling.support_spec import Filters

from .batch_invariant_sampling_common import (
    DS as _DS,
    PDL as _PDL,
    SM_COUNTS,
    bad_ptx_ops,
    compile_device_free,
    device_marker,
    device_supported,
    device_tests_selected,
    logits,
    precompile,
)

DT = [torch.float32, torch.bfloat16]
#: One block, a partial block (an odd vocabulary), an odd long one, a long one.
NS = [64, 1000, 4097, 32768]
N_BIG = 152064
SETS = {
    "k": Filters(top_k=1),
    "p": Filters(top_p=1.0),
    "m": Filters(min_p=0.0),
    "kp": Filters(top_k=1, top_p=1.0),
    "kpj": Filters(top_k=1, top_p=1.0, order="joint"),
    "kpm": Filters(top_k=1, top_p=1.0, min_p=0.0),
}
SETS_BY_N = {
    64: ("k", "kpm"),
    1000: ("kp", "m"),
    4097: ("kpj", "p"),
    32768: ("kpm", "k"),
}
#: ``(N, M)`` at which the SM counts split the support dispatcher's pick (the support suite's).
SM_SPLIT_ROWS = ((152064, 180),)
ENTRIES = (
    "top_k_sampling_from_probs",
    "top_p_sampling_from_probs",
    "min_p_sampling_from_probs",
    "top_k_top_p_sampling_from_probs",
    "top_k_top_p_sampling_from_logits",
)
_supported = device_supported(SUPPORT_ARCHS)
needs_device = device_marker(SUPPORT_ARCHS)


# ------------------------------------------------------------------------------------------------ helpers
def _configs(N, eb, k=3):
    """Members with a share of at most 256 elements per thread whose staged tile fits the budget (the support kernel's
    space): a single-CTA, a cluster, a staged, a wide single-CTA and a re-read member."""
    cfgs = [
        c
        for c in C.all_valid_configs(N, eb)
        if c.elems_per_thread(N) <= 256
        and c.rows_per_cta == 1
        and c.blocks_per_cta(N) * C.BLOCK * eb <= C.SMEM_BUDGET
    ]
    picks = (
        cfgs[:1]
        + [c for c in cfgs if c.cluster > 1][:1]
        + [c for c in cfgs if c.stage == "smem"][:1]
        + [c for c in cfgs if c.threads_per_row >= 512 and c.cluster == 1][:1]
        + [c for c in cfgs if c.stage == "reread"][:1]
    )
    return list(dict.fromkeys(picks))[:k]


def weights(M, N, dtype, seed=0, device="cuda"):
    """Weights with zeros (every seventh element), ties at the top of row 0, an all-zero row 5 and a one-hot row 7."""
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(M, N, generator=g) ** 3
    w[:, ::7] = 0.0
    w[0, 1] = w[0, 2] = w[0].max() * 1.5
    if M > 5:
        w[5] = 0.0
    if M > 7:
        w[7] = 0.0
        w[7, 11] = 1.0
    return w.to(dtype).to(device)


def tied_logits(M, N, dtype, seed=0):
    """The shared helper's adversarial logits with a three-way tie at the top of row 1."""
    x = logits(M, N, dtype, seed=seed).float()
    x[1, :3] = x[1].max() + 1.0
    return x.to(dtype).cuda()


def keys(M, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(-(2**62), 2**62, (M,), generator=g).cuda(),
        torch.randint(0, 2**40, (M,), generator=g).cuda(),
    )


def params(fset, M, N, seed=0):
    """Per-row parameters of the filter set (``k = 1`` / ``N`` / ``0`` and ``p = 0`` / ``1`` among them) as the entry's
    keyword arguments and as the model's per-row :class:`Filters`."""
    g = torch.Generator().manual_seed(seed)
    ks = torch.randint(1, max(2, N // 2), (M,), generator=g)
    ks[0] = 1
    if M > 2:
        ks[2] = N
    if M > 3:
        ks[3] = 0
    ps = torch.rand(M, generator=g)
    if M > 1:
        ps[1] = 0.0
    if M > 2:
        ps[2] = 1.0
    ms = torch.rand(M, generator=g) * 0.5
    f = SETS[fset]
    kw = {"order": f.order}
    if f.top_k is not None:
        kw["top_k"] = ks.to(torch.int32).cuda()
    if f.top_p is not None:
        kw["top_p"] = ps.cuda()
    if f.min_p is not None:
        kw["min_p"] = ms.cuda()
    filters = [
        Filters(
            top_k=int(ks[i]) if f.top_k is not None else None,
            top_p=float(ps[i]) if f.top_p is not None else None,
            min_p=float(ms[i]) if f.min_p is not None else None,
            order=f.order,
        )
        for i in range(M)
    ]
    return kw, filters


def probs_draw(w, s, o, **kw):
    """The generic entry on weights with the public entries' PDL choice."""
    return draw_support(w, s, o, enable_pdl=_PDL, **kw)


def logits_draw(x, s, o, **kw):
    return draw_support(x, s, o, from_logits=True, enable_pdl=_PDL, **kw)


def _bits(t):
    return torch.nan_to_num(t.cpu(), 0.0).view(torch.int32)


def _same_nan(a, b, what):
    a, b = a.cpu(), b.cpu()
    assert torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(
        _bits(a), _bits(b)
    ), what


def _same(r, ref, lp=False, label="", lpk=False):
    assert torch.equal(r["samples"].cpu().long(), ref["samples"]), f"{label} token"
    assert torch.equal(r["valid"].cpu(), ref["valid"]), f"{label} valid"
    assert torch.equal(_bits(r["mass"]), _bits(ref["mass"])), f"{label} mass"
    if lp:
        _same_nan(r["logp"], ref["logprobs"], f"{label} logprob (full distribution)")
    if lpk:
        _same_nan(r["kept_logp"], ref["kept_logprobs"], f"{label} kept logprob")


# ------------------------------------------------------------------------------------------------ device-free
def test_fingerprints_are_pinned():
    """The receipts: the five entries name the composition and their fingerprints join the draw class's (the tree the
    kept mass and the prefixes are read from) with the support class's (the kept set); the dimensions move them (the
    vocabulary, the dtype, the order, the weight form), the parameter values do not."""
    pins = {
        "top_k_sampling_from_probs": "sample-icdf-f2947e7df54776f2+support-row-4a26c16577a204b6",
        "top_p_sampling_from_probs": "sample-icdf-f2947e7df54776f2+support-row-475956ca58f276bb",
        "min_p_sampling_from_probs": "sample-icdf-f2947e7df54776f2+support-row-d6605a499dd10213",
        "top_k_top_p_sampling_from_probs": "sample-icdf-f2947e7df54776f2+support-row-a18e33002ad6e1d6",
        "top_k_top_p_sampling_from_logits": "sample-icdf-5d0ecdc7569e9635+support-row-37d93a3dcdf2723b",
    }
    for op in ENTRIES:
        assert bi.class_name(op) == CLASS == "sample.icdf.v1+support.row.v1"
        assert bi.fingerprint(op, N=1000, dtype="bf16") == pins[op]
    assert (
        bi.fingerprint(
            "top_k_top_p_sampling_from_probs",
            N=1000,
            dtype="bf16",
            filter_apply_order="joint",
        )
        == "sample-icdf-f2947e7df54776f2+support-row-33c2845857c90a47"
    )
    draw_fp = bi.fingerprint("sampling_from_probs", N=1000, dtype="bf16")
    sup_fp = SS.spec_fingerprint(Filters(top_k=1), 1000, False, "bf16")
    assert bi.fingerprint("top_k_sampling_from_probs", N=1000, dtype="bf16") == (
        f"{draw_fp}+{sup_fp}"
    )
    assert bi.fingerprint(
        "top_k_top_p_sampling_from_logits", N=1000, dtype="bf16"
    ).startswith(bi.fingerprint("sampling_from_logits", N=1000, dtype="bf16") + "+")
    # a temperature in the scale: both halves embed `c`, so another receipt (the entry's `temperature=` keyword)
    fl_t = bi.fingerprint(
        "top_k_top_p_sampling_from_logits", N=1000, dtype="bf16", temperature=True
    )
    assert fl_t == "sample-icdf-ce7d921bf6afce4b+support-row-2039ae06e93b6c1e"
    assert fl_t != pins["top_k_top_p_sampling_from_logits"]
    assert fl_t.endswith(
        "+" + SS.spec_fingerprint(Filters(top_k=1, top_p=1.0), 1000, True, "bf16", True)
    )
    assert (
        bi.fingerprint(
            "top_k_top_p_sampling_from_logits", N=1000, dtype="bf16", temperature=False
        )
        == pins["top_k_top_p_sampling_from_logits"]
    )
    k_bf16 = bi.fingerprint("top_k_sampling_from_probs", N=1000, dtype="bf16")
    assert k_bf16 != bi.fingerprint("top_k_sampling_from_probs", N=1000, dtype="f32")
    assert k_bf16 != bi.fingerprint("top_k_sampling_from_probs", N=1001, dtype="bf16")
    assert k_bf16 != bi.fingerprint("top_p_sampling_from_probs", N=1000, dtype="bf16")
    assert sup_fp == SS.spec_fingerprint(Filters(top_k=7), 1000, False, "bf16")
    with pytest.raises(TypeError):
        bi.fingerprint("top_k_sampling_from_probs", N=1000)
    with pytest.raises(ValueError):
        bi.fingerprint("top_k_sampling_from_probs", N=1000, dtype="f64")
    for op in ("top_k_renorm_probs", "top_p_renorm_probs", "top_k_mask_logits"):
        assert (
            op not in bi.CLASSES
        )  # the renormalisation entries carry no flag in this release


def test_model_is_the_support_then_the_draw_over_the_kept_weights():
    """The exact model composes the two classes' models: the support row's mask, then the draw row over the masked
    weights at the row's uniform; the drawn token is kept; nothing kept (``k = 0``, an all-zero row) is invalid."""
    N, M = 1000, 6
    w = weights(M, N, torch.float32, device="cpu")
    filters = [
        Filters(top_k=5),
        Filters(top_p=0.5),
        Filters(min_p=0.1),
        Filters(top_k=0),
        Filters(top_k=3, top_p=0.9, order="joint"),
        Filters(top_k=N),
    ]
    seeds, offs = list(range(M)), [7] * M
    ref = DR.draw_support_reference(w, seeds, offs, filters)
    for r in range(M):
        res = support_row(w[r], N, filters[r])
        wk = torch.where(res["mask"], w[r], torch.zeros_like(w[r]))
        u = stream.uniform32(stream.words(seeds[r], offs[r], 0)[0])
        tok, ok, mass, _ = DR.draw_row(wk, N, u)
        assert int(ref["samples"][r]) == tok and bool(ref["valid"][r]) == ok
        assert ref["mass"][r].item() == torch.tensor(mass, dtype=torch.float32).item()
        assert ref["u"][r].item() == torch.tensor(u, dtype=torch.float32).item()
        if ok:
            assert bool(res["mask"][tok])
    assert not bool(ref["valid"][3]) and int(ref["samples"][3]) == -1
    assert not bool(ref["valid"][5]) and int(ref["samples"][5]) == -1
    assert bool(ref["valid"][[0, 1, 2, 4]].all())
    # k = N keeps everything: the plain draw's model
    plain = DR.draw_reference(w[5:6], seeds[5:6], offs[5:6])
    assert int(plain["samples"][0]) == -1  # the all-zero row is invalid there too
    plain = DR.draw_reference(w[0:1], seeds[0:1], offs[0:1])
    full = DR.draw_support_reference(w[0:1], seeds[0:1], offs[0:1], [Filters(top_k=N)])
    assert int(plain["samples"][0]) == int(full["samples"][0])


def test_unsupported_inputs_raise_before_any_kernel():
    x = torch.rand(4, 64, dtype=torch.bfloat16)
    s = torch.zeros(4, dtype=torch.int64)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        flashinfer.sampling.top_k_sampling_from_probs(
            x, 5, seed=s, offset=s, batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="sample.icdf"):
        flashinfer.sampling.top_p_sampling_from_probs(
            x, 0.5, seed=s, offset=s, batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        flashinfer.sampling.min_p_sampling_from_probs(
            x, 0.1, seed=s, offset=s, batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        flashinfer.sampling.top_k_top_p_sampling_from_probs(
            x, 5, 0.5, seed=s, offset=s, batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        flashinfer.sampling.top_k_top_p_sampling_from_logits(
            x, 5, 0.5, seed=s, offset=s, batch_invariant=True
        )
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="sample.icdf"):
        draw_support(x.double(), s, s, top_k=5)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="support.row"):
        top_k_sampling_from_probs_batch_invariant(x.double(), 5, seed=s, offset=s)
    with pytest.raises(ValueError, match="Invalid filter_apply_order"):
        top_k_top_p_sampling_from_probs_batch_invariant(
            x, 5, 0.5, filter_apply_order="p_first", seed=s, offset=s
        )
    with pytest.raises(ValueError, match="Invalid filter_apply_order"):
        top_k_top_p_sampling_from_logits_batch_invariant(
            x, 5, 0.5, filter_apply_order="p_first", seed=s, offset=s
        )


# The admitted PTX forms (the row kernels' suite); the cases: probabilities under top-k + top-p on a single CTA (the mass
# selection with its early finish), logits under all three filters on a cluster with a per-row temperature, the gather and
# the token's log-probability.
IR_CASES = [
    # (dtype, N, config, logits, filters, per-row temperature, gather, full log-probability, kept log-probability)
    ("f32", 4096, (128, 1, 4), False, "kp", False, False, False, False),
    ("bf16", 32768, (64, 1, 8, "regs", 4), True, "kpm", True, True, True, True),
    # top-k alone on logits with the full log-probability: the row sum tree only for it, no fixed-point mass, no division
    ("f32", 4096, (128, 1, 4), True, "k", False, False, True, False),
]


@pytest.mark.parametrize(
    "dtype,N,cfg,from_logits,fset,trow,idx,lp,lpk", IR_CASES, ids=lambda v: str(v)
)
def test_draw_support_ir_uses_only_declared_forms(
    tmp_path, dtype, N, cfg, from_logits, fset, trow, idx, lp, lpk
):
    """Compile one kernel per code path with no GPU visible and check the PTX: `.rn` arithmetic only, `ex2.approx` iff
    logits and `lg2.approx` iff the log-probability, the declared `cvt.rzi.s64.f32` iff top-p, the divisions declared
    (`div(2^44, S)` for top-p, the per-row temperature's), the shared-memory integer adds of the histograms, the cluster
    forms only in cluster members, no fast-math anywhere."""
    pytest.importorskip("cutlass")
    f = SETS[fset]
    code = (
        "from flashinfer.batch_invariant.sampling.schedule import LSEConfig\n"
        "from flashinfer.batch_invariant.sampling.draw_support_kernel import get_compiled_kernel\n"
        "from flashinfer.batch_invariant.sampling.support_spec import Filters\n"
        f"get_compiled_kernel({dtype!r}, {N}, LSEConfig{cfg}, {from_logits}, Filters({f.top_k!r}, {f.top_p!r}, {f.min_p!r}, {f.order!r}), "
        f"{trow}, {idx}, {lp}, False, 'mass', {lpk})\n"
    )
    _, ptx = compile_device_free(tmp_path, code)
    bad = bad_ptx_ops(ptx)
    assert not bad, f"PTX contains non-admitted floating-point forms: {bad}"
    assert "ftz" not in ptx.replace("ex2.approx.ftz.f32", "").replace(
        "lg2.approx.ftz.f32", ""
    )
    assert ("ex2.approx.ftz.f32" in ptx) == from_logits
    assert ("lg2.approx.ftz.f32" in ptx) == (lp or lpk)
    assert ("cvt.rzi.s64.f32" in ptx) == (f.top_p is not None)
    divs = (1 if trow else 0) + (1 if f.top_p is not None else 0)
    assert ptx.count("div.rn.f32") == divs, (ptx.count("div.rn.f32"), divs)
    assert "red.shared.add.u32" in ptx
    Cc = cfg[4] if len(cfg) > 4 else 1
    assert ("barrier.cluster" in ptx) == (Cc > 1) and (
        "ld.shared::cluster.b32" in ptx
    ) == (Cc > 1)


# ------------------------------------------------------------------------------------------------ the precompile
_WORKER_CODE = r"""
import ast, sys
from flashinfer.batch_invariant.sampling.schedule import LSEConfig
from flashinfer.batch_invariant.sampling.draw_support_kernel import get_compiled_kernel
from flashinfer.batch_invariant.sampling.support_spec import Filters
bad = []
for s in ast.literal_eval(sys.argv[1]):
    try:
        ds, N, cfg, from_logits, fset, trow, idx, lp, pdl, *rest = s
        get_compiled_kernel(ds, N, LSEConfig(*cfg), from_logits, Filters(**dict(fset)), trow, idx, lp, pdl, *rest)
    except Exception as e:  # noqa: BLE001
        bad.append((s, repr(e)[:300]))
print("BAD", repr(bad))
"""


def _cfg_tuple(c):
    return (c.threads_per_row, c.rows_per_cta, c.vec_elems, c.stage, c.cluster)


def _fset(f: Filters):
    # a tuple of items (the specs are de-duplicated by hashing; the worker rebuilds the Filters)
    return (
        ("top_k", f.top_k),
        ("top_p", f.top_p),
        ("min_p", f.min_p),
        ("order", f.order),
    )


def _spec(
    ds, N, cfg, from_logits, fset, trow, idx=False, lp=False, pdl=_PDL, lpk=False
):
    return (
        ds,
        N,
        _cfg_tuple(cfg),
        from_logits,
        _fset(SETS[fset]),
        trow,
        idx,
        lp,
        pdl,
        "mass",
        lpk,
    )


def _d(N, M, eb, sm):
    return DS.support_config(N, M, eb, sm)


def _suite_specs(sm):
    """Every kernel the device tests below may ask for."""
    specs = []
    for dtype in DT:
        ds, eb = _DS[dtype], dtype.itemsize
        for N in NS:
            for fset in SETS_BY_N[N]:
                for c in _configs(N, eb):
                    specs.append(_spec(ds, N, c, False, fset, False))
                    specs.append(_spec(ds, N, c, True, fset, True, lp=True, lpk=True))
                d12 = _d(N, 12, eb, sm)
                specs.append(_spec(ds, N, d12, False, fset, False))
                specs.append(_spec(ds, N, d12, True, fset, True, lp=True, lpk=True))
    # the composition / public-entries test and the degenerate parameters (N = 1000)
    for fset in ("k", "p", "m", "kp", "kpj"):
        specs.append(_spec("f32", 1000, _d(1000, 6, 4, sm), False, fset, False))
    for fset in ("kp", "kpj"):
        d6 = _d(1000, 6, 2, sm)
        specs.append(_spec("bf16", 1000, d6, True, fset, False))
        specs.append(_spec("bf16", 1000, d6, True, fset, False, lp=True, lpk=True))
        specs.append(_spec("bf16", 1000, d6, True, fset, False, lpk=True))
        specs.append(_spec("bf16", 1000, d6, True, fset, True))
    for fset in ("k", "p", "m"):
        specs.append(_spec("f32", 1000, _d(1000, 8, 4, sm), False, fset, False))
    # the index gather (probabilities; logits with the log-probability)
    specs.append(_spec("f32", 1000, _d(1000, 6, 4, sm), False, "kp", False, idx=True))
    specs.append(
        _spec("f32", 1000, _d(1000, 6, 4, sm), True, "kp", False, idx=True, lp=True)
    )
    # the generator-only call shape (one row)
    for fset in ("k", "p", "m", "kp"):
        specs.append(_spec("f32", 1000, _d(1000, 1, 4, sm), False, fset, False))
    specs.append(_spec("bf16", 1000, _d(1000, 1, 2, sm), True, "kp", False))
    # batch invariance: sub-batches of 4096 x 64, and the operator's probe at 32000 rows
    for M in (64, 1, 3, 8, 16):
        specs.append(_spec("f32", 4096, _d(4096, M, 4, sm), False, "kpm", False))
    for M in (64, 8, 1):
        for fset in ("k", "p", "m", "kp"):
            specs.append(_spec("f32", 32000, _d(32000, M, 4, sm), False, fset, False))
        specs.append(_spec("f32", 32000, _d(32000, M, 4, sm), True, "kp", False))
    # the drawn distribution (one row repeated)
    specs.append(_spec("f32", 1000, _d(1000, 4096, 4, sm), False, "kp", False))
    # determinism / graph replay / PDL on and off
    for pdl in (True, False):
        specs.append(
            _spec("bf16", 4096, _d(4096, 5, 2, sm), False, "k", False, pdl=pdl)
        )
        specs.append(
            _spec("bf16", 4096, _d(4096, 5, 2, sm), True, "kp", False, pdl=pdl)
        )
    # the SM-count test's picks
    for N, M in SM_SPLIT_ROWS:
        for smc in SM_COUNTS:
            specs.append(
                _spec("bf16", N, DS.support_config(N, M, 2, smc), False, "kp", False)
            )
    # the big vocabulary (logits, every filter, the log-probability)
    for c in _configs(N_BIG, 2, 2):
        specs.append(_spec("bf16", N_BIG, c, True, "kpm", True, lp=True))
    specs.append(_spec("bf16", N_BIG, _d(N_BIG, 2, 2, sm), True, "kpm", True, lp=True))
    # views / 3-D inputs / empty batches
    specs.append(_spec("bf16", 4096, _d(4096, 6, 2, sm), False, "k", False))
    specs.append(_spec("bf16", 4096, _d(4096, 6, 2, sm), True, "kp", False))
    return list(dict.fromkeys(specs))


@pytest.fixture(scope="session", autouse=True)
def _precompile_draw_support_kernels(request):
    """Precompile the device tests' kernels on worker processes once per session: a no-op without a supported device,
    under ``BI_DRAW_SUPPORT_NO_PRECOMPILE=1``, and when no device test is selected."""
    if (
        not _supported
        or os.environ.get("BI_DRAW_SUPPORT_NO_PRECOMPILE") == "1"
        or not device_tests_selected(request.session, needs_device.kwargs["reason"])
    ):
        yield
        return
    precompile(
        _suite_specs(device_sm_count(torch.device("cuda"))),
        _WORKER_CODE,
        "batch_invariant_sampling_draw_support",
        "[batch_invariant.sampling.draw_support]",
    )
    yield


# ------------------------------------------------------------------------------------------------ device
@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["f32", "bf16"])
@pytest.mark.parametrize("N", NS)
def test_probs_match_exact_model_across_configs(dtype, N):
    """Token, validity and the kept mass equal the exact model bit for bit for every filter set of the vocabulary, across
    single-CTA / cluster / staged / wide members and the dispatcher's pick (per-row parameters with the degenerate
    values among them)."""
    M = 12
    w = weights(M, N, dtype)
    s, o = keys(M)
    for fset in SETS_BY_N[N]:
        kw, filters = params(fset, M, N)
        ref = DR.draw_support_reference(w, s.tolist(), o.tolist(), filters)
        for cfg in _configs(N, dtype.itemsize) + [None]:
            r = probs_draw(w, s, o, config=cfg, **kw)
            _same(r, ref, label=f"{fset} {cfg.describe() if cfg else 'default'}")


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["f32", "bf16"])
@pytest.mark.parametrize("N", NS)
def test_logits_match_exact_model_and_logprob(dtype, N):
    """Logits with per-row temperatures and ties: token, validity, the kept mass and the drawn token's two
    log-probabilities (under the full distribution, under the kept set) equal the exact model bit for bit across
    configs; the full one is ``log_probs`` at the drawn token, bit for bit (R4)."""
    M = 12
    x = tied_logits(M, N, dtype, seed=4)
    s, o = keys(M, 3)
    temps = torch.rand(M, device="cuda") + 0.5
    for fset in SETS_BY_N[N]:
        kw, filters = params(fset, M, N, seed=1)
        ref = DR.draw_support_reference(
            x, s.tolist(), o.tolist(), filters, from_logits=True, temperature=temps
        )
        for cfg in _configs(N, dtype.itemsize) + [None]:
            r = logits_draw(
                x,
                s,
                o,
                temperature=temps,
                return_logprob=True,
                return_kept_logprob=True,
                config=cfg,
                **kw,
            )
            _same(
                r,
                ref,
                lp=True,
                lpk=True,
                label=f"{fset} {cfg.describe() if cfg else 'default'}",
            )
        ok = r["valid"]
        full = flashinfer.sampling.log_probs(x, r["samples"].clamp(min=0), temps)
        assert torch.equal(_bits(r["logp"][ok]), _bits(full[ok])), (
            f"{fset}: the full-distribution log-probability is not log_probs at the token"
        )


@needs_device
def test_support_then_draw_and_the_public_entries():
    """The composition equals the support class's mask then the plain draw over the masked weights (token, validity
    and the kept mass bit for bit); the five public entries are the generic entry (scalars and per-row parameters, both
    orders, ``deterministic`` and ``top_k_max`` ignored); a drawn token is always kept; for logits the drawn token's
    log-probability is the support's renormalised one at the token (R4); ``check_nan`` is the operator's host check;
    the error contract of the parameters."""
    N, M = 1000, 6
    w = weights(M, N, torch.float32)
    s, o = keys(M, 4)
    ks = torch.tensor([1, 5, 0, N, 50, 7], dtype=torch.int32, device="cuda")
    sup = sampling_support(w, top_k=ks)
    wk = torch.where(sup.mask, w, torch.zeros_like(w))
    tok3, ok3 = sampling_from_probs_batch_invariant(
        wk, seed=s, offset=o, return_valid=True
    )
    r = probs_draw(w, s, o, top_k=ks)
    assert torch.equal(r["samples"], tok3) and torch.equal(r["valid"], ok3)
    assert torch.equal(r["mass"], sup.mass)
    assert not bool(r["valid"][2]) and int(r["samples"][2]) == -1  # k = 0 keeps nothing
    assert bool(r["valid"][0]) and bool(r["valid"][3])  # k = 1 / k = N draw
    assert not bool(r["valid"][5])  # the all-zero row
    tk, vk = flashinfer.sampling.top_k_sampling_from_probs(
        w, ks, seed=s, offset=o, return_valid=True, batch_invariant=True
    )
    assert torch.equal(tk, r["samples"]) and torch.equal(vk, r["valid"])
    assert torch.equal(
        top_k_sampling_from_probs_batch_invariant(w, ks, seed=s, offset=o), r["samples"]
    )
    assert torch.equal(
        flashinfer.sampling.top_k_sampling_from_probs(
            w, 7, seed=s, offset=o, deterministic=False, batch_invariant=True
        ),
        probs_draw(w, s, o, top_k=7)["samples"],
    )
    ps = torch.tensor([0.9, 0.5, 0.0, 1.0, 0.3, 0.99], device="cuda")
    assert torch.equal(
        flashinfer.sampling.top_p_sampling_from_probs(
            w, ps, seed=s, offset=o, batch_invariant=True
        ),
        probs_draw(w, s, o, top_p=ps)["samples"],
    )
    assert torch.equal(
        top_p_sampling_from_probs_batch_invariant(w, 0.8, seed=s, offset=o),
        probs_draw(w, s, o, top_p=0.8)["samples"],
    )
    ms = torch.rand(M, device="cuda") * 0.5
    assert torch.equal(
        flashinfer.sampling.min_p_sampling_from_probs(
            w, ms, seed=s, offset=o, batch_invariant=True
        ),
        probs_draw(w, s, o, min_p=ms)["samples"],
    )
    assert torch.equal(
        min_p_sampling_from_probs_batch_invariant(w, 0.1, seed=s, offset=o),
        probs_draw(w, s, o, min_p=0.1)["samples"],
    )
    for order in ("top_k_first", "joint"):
        r2 = probs_draw(w, s, o, top_k=ks, top_p=ps, order=order)
        assert torch.equal(
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                w,
                ks,
                ps,
                filter_apply_order=order,
                seed=s,
                offset=o,
                top_k_max=64,
                batch_invariant=True,
            ),
            r2["samples"],
        )
        assert torch.equal(
            top_k_top_p_sampling_from_probs_batch_invariant(
                w, ks, ps, filter_apply_order=order, seed=s, offset=o
            ),
            r2["samples"],
        )
        sup2 = sampling_support(w, top_k=ks, top_p=ps, order=order)
        assert torch.equal(r2["mass"], sup2.mass)
        for i in range(M):
            if bool(r2["valid"][i]):
                assert bool(sup2.mask[i, int(r2["samples"][i])])
    # logits: the kept log-probability is the support's renormalised one at the token and the full one is log_probs
    # there (R4); the public entry returns (samples[, logp][, kept_logp]) and takes the temperature the classes take
    x = tied_logits(M, N, torch.bfloat16, seed=2)
    temps = torch.rand(M, device="cuda") + 0.5
    for order in ("top_k_first", "joint"):
        rl = logits_draw(
            x,
            s,
            o,
            top_k=ks,
            top_p=ps,
            order=order,
            return_logprob=True,
            return_kept_logprob=True,
        )
        assert torch.equal(
            flashinfer.sampling.top_k_top_p_sampling_from_logits(
                x,
                ks,
                ps,
                filter_apply_order=order,
                seed=s,
                offset=o,
                batch_invariant=True,
            ),
            rl["samples"],
        )
        tk, lpf, lpk = flashinfer.sampling.top_k_top_p_sampling_from_logits(
            x,
            ks,
            ps,
            filter_apply_order=order,
            seed=s,
            offset=o,
            batch_invariant=True,
            return_logprob=True,
            return_kept_logprob=True,
        )
        assert torch.equal(tk, rl["samples"])
        _same_nan(lpf, rl["logp"], "public logp")
        _same_nan(lpk, rl["kept_logp"], "public kept_logp")
        tk2, lpk2 = top_k_top_p_sampling_from_logits_batch_invariant(
            x,
            ks,
            ps,
            filter_apply_order=order,
            seed=s,
            offset=o,
            return_kept_logprob=True,
        )
        assert torch.equal(tk2, rl["samples"])
        _same_nan(lpk2, rl["kept_logp"], "entry kept_logp")
        rt = logits_draw(x, s, o, top_k=ks, top_p=ps, order=order, temperature=temps)
        assert torch.equal(
            flashinfer.sampling.top_k_top_p_sampling_from_logits(
                x,
                ks,
                ps,
                filter_apply_order=order,
                seed=s,
                offset=o,
                batch_invariant=True,
                temperature=temps,
            ),
            rt["samples"],
        )
        supl = sampling_support(
            x, top_k=ks, top_p=ps, order=order, from_logits=True, return_logprobs=True
        )
        assert torch.equal(rl["mass"], supl.mass)
        full = flashinfer.sampling.log_probs(x, rl["samples"].clamp(min=0))
        for i in range(M):
            if bool(rl["valid"][i]):
                t = int(rl["samples"][i])
                assert bool(supl.mask[i, t])
                assert rl["kept_logp"][i].view(torch.int32).item() == (
                    supl.logprobs[i, t].view(torch.int32).item()
                )
                assert rl["logp"][i].view(torch.int32).item() == (
                    full[i].view(torch.int32).item()
                )
            else:
                assert bool(torch.isnan(rl["logp"][i])) and bool(
                    torch.isnan(rl["kept_logp"][i])
                )
    bad = w.clone()
    bad[1, 3] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        flashinfer.sampling.top_k_sampling_from_probs(
            bad, 5, seed=s, offset=o, check_nan=True, batch_invariant=True
        )
    with pytest.raises(ValueError, match="NaN"):
        flashinfer.sampling.top_k_top_p_sampling_from_probs(
            bad, 5, 0.9, seed=s, offset=o, check_nan=True, batch_invariant=True
        )
    with pytest.raises(ValueError, match="NaN"):
        flashinfer.sampling.top_k_top_p_sampling_from_logits(
            bad, 5, 0.9, seed=s, offset=o, check_nan=True, batch_invariant=True
        )
    with pytest.raises(ValueError, match="at least one"):
        probs_draw(w, s, o)
    with pytest.raises(ValueError, match="log-probability"):
        probs_draw(w, s, o, top_k=5, return_logprob=True)
    with pytest.raises(ValueError, match="log-probability"):
        probs_draw(w, s, o, top_k=5, return_kept_logprob=True)
    with pytest.raises(ValueError, match="Invalid filter_apply_order"):
        flashinfer.sampling.top_k_top_p_sampling_from_probs(
            w,
            5,
            0.9,
            filter_apply_order="p_first",
            seed=s,
            offset=o,
            batch_invariant=True,
        )
    with pytest.raises(ValueError, match="top_k"):
        probs_draw(w, s, o, top_k=torch.ones(5, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="one row per CTA"):
        probs_draw(w, s, o, top_k=5, config=C.LSEConfig(32, 4, 4))


@needs_device
def test_index_gather_and_batch_invariance():
    """``indices[i]`` selects the source row of output ``i`` (the output takes the indices' dtype on the public entries;
    an index outside the source rows makes its output invalid); logits with the log-probability through the gather. R2:
    rows alone equal their rows in the batch, a permuted batch permutes the tokens, sub-batches agree."""
    N = 1000
    w = weights(8, N, torch.float32, seed=2)
    idx = torch.tensor([3, 0, 0, 7, 1, 3], device="cuda")
    s, o = keys(6, 7)
    ks = torch.tensor([2, 9, 9, 4, 1, 2], dtype=torch.int32, device="cuda")
    ps = torch.tensor([0.5, 0.9, 0.9, 0.7, 1.0, 0.5], device="cuda")
    filters = [Filters(top_k=int(ks[i]), top_p=float(ps[i])) for i in range(6)]
    r = probs_draw(w, s, o, top_k=ks, top_p=ps, indices=idx)
    ref = DR.draw_support_reference(w[idx].cpu(), s.tolist(), o.tolist(), filters)
    _same(r, ref)
    assert bool(r["valid"].all())
    t32 = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        w, ks, ps, indices=idx.to(torch.int32), seed=s, offset=o, batch_invariant=True
    )
    assert t32.dtype == torch.int32 and torch.equal(t32, r["samples"])
    t64 = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        w, ks, ps, indices=idx, seed=s, offset=o, batch_invariant=True
    )
    assert t64.dtype == torch.int64 and torch.equal(t64, r["samples"].long())
    bad = idx.clone()
    bad[4] = 99
    rb = probs_draw(w, s, o, top_k=ks, top_p=ps, indices=bad)
    assert not bool(rb["valid"][4]) and int(rb["samples"][4]) == -1
    assert torch.equal(rb["samples"][:4], r["samples"][:4])
    x = tied_logits(8, N, torch.float32, seed=3)
    rl = logits_draw(x, s, o, top_k=ks, top_p=ps, indices=idx, return_logprob=True)
    refl = DR.draw_support_reference(
        x[idx].cpu(), s.tolist(), o.tolist(), filters, from_logits=True
    )
    _same(rl, refl, lp=True)
    # R2
    N, M = 4096, 64
    w = weights(M, N, torch.float32, seed=5)
    s, o = keys(M, 9)
    kw, _ = params("kpm", M, N, seed=2)
    full = probs_draw(w, s, o, **kw)["samples"]

    def sub(sel):
        return probs_draw(
            w[sel].contiguous(),
            s[sel],
            o[sel],
            top_k=kw["top_k"][sel],
            top_p=kw["top_p"][sel],
            min_p=kw["min_p"][sel],
            order=kw["order"],
        )["samples"]

    for i in (0, 17, 63):
        assert torch.equal(sub(slice(i, i + 1)), full[i : i + 1])
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(1)).cuda()
    assert torch.equal(sub(perm), full[perm])
    for k in (3, 8, 16):
        sel = torch.randperm(M, generator=torch.Generator().manual_seed(k))[:k].cuda()
        assert torch.equal(sub(sel), full[sel]), k


@needs_device
def test_batch_invariance_the_way_the_operator_was_probed():
    """The probe that found the operator's samplers slot-dependent, run on the flag for every entry: row 5's token alone
    == at slot 5 of a batch of 64 == at slot 5 of a batch of 8 (same per-row seed / offset); a batch permutation moves no
    token. The operator's deterministic path's count of moved tokens is printed for contrast, not asserted."""
    torch.manual_seed(0)
    N, M = 32000, 64
    probs = torch.softmax(torch.randn(M, N, device="cuda") * 2, dim=-1)
    lg = torch.randn(M, N, device="cuda") * 2
    seed = torch.full((M,), 1234, dtype=torch.int64, device="cuda")
    off = torch.full((M,), 7, dtype=torch.int64, device="cuda")
    perm = torch.roll(torch.arange(M, device="cuda"), 1)
    for fn, x, args in (
        (flashinfer.sampling.top_k_sampling_from_probs, probs, (40,)),
        (flashinfer.sampling.top_p_sampling_from_probs, probs, (0.9,)),
        (flashinfer.sampling.min_p_sampling_from_probs, probs, (0.05,)),
        (flashinfer.sampling.top_k_top_p_sampling_from_probs, probs, (40, 0.9)),
        (flashinfer.sampling.top_k_top_p_sampling_from_logits, lg, (40, 0.9)),
    ):
        a = fn(x, *args, seed=seed, offset=off, batch_invariant=True)
        assert torch.equal(a, fn(x, *args, seed=seed, offset=off, batch_invariant=True))
        alone = fn(
            x[5:6].clone(), *args, seed=seed[:1], offset=off[:1], batch_invariant=True
        )
        a8 = fn(x[:8], *args, seed=seed[:8], offset=off[:8], batch_invariant=True)
        assert int(alone[0]) == int(a[5]) == int(a8[5])
        ap = fn(x[perm], *args, seed=seed[perm], offset=off[perm], batch_invariant=True)
        assert torch.equal(ap, a[perm]), "a batch permutation moved a token"
        d = fn(x, *args, seed=seed, offset=off, deterministic=True)
        dp = fn(x[perm], *args, seed=seed[perm], offset=off[perm], deterministic=True)
        print(
            f"[probe] {fn.__name__}: the operator's deterministic path moves {int((dp != d[perm]).sum())} of {M} "
            "tokens under the permutation; the flag moves 0",
            flush=True,
        )


@needs_device
def test_generator_only_call_shape_and_key_forms():
    """One output row: a call with only a ``generator`` reserves ``(seed, offset)`` exactly as the operator's path does
    (32 per row for the rejection samplers, one for min-p), so a fresh generator seeded alike reproduces the draw through
    the exact model; a scalar key and a length-one tensor are that row's key; a seed without an offset takes the
    generator's pair; uint64 keys are read by their bits. Several rows need one key per output row: a shared key is
    refused without advancing the generator; per-row tensors are the caller's."""
    N, M = 1000, 6
    p = weights(M, N, torch.float32, seed=4)
    x = tied_logits(M, N, torch.bfloat16)
    dev = torch.device("cuda", torch.cuda.current_device())
    for fn, inp, args, f, inc, fl in (
        (
            flashinfer.sampling.top_k_sampling_from_probs,
            p,
            (5,),
            Filters(top_k=5),
            32,
            False,
        ),
        (
            flashinfer.sampling.top_p_sampling_from_probs,
            p,
            (0.7,),
            Filters(top_p=0.7),
            32,
            False,
        ),
        (
            flashinfer.sampling.min_p_sampling_from_probs,
            p,
            (0.1,),
            Filters(min_p=0.1),
            1,
            False,
        ),
        (
            flashinfer.sampling.top_k_top_p_sampling_from_probs,
            p,
            (5, 0.7),
            Filters(top_k=5, top_p=0.7),
            32,
            False,
        ),
        (
            flashinfer.sampling.top_k_top_p_sampling_from_logits,
            x,
            (5, 0.7),
            Filters(top_k=5, top_p=0.7),
            32,
            True,
        ),
    ):
        one = inp[1:2].clone()
        g = torch.Generator(device="cuda").manual_seed(11)
        tok = fn(one, *args, generator=g, batch_invariant=True)
        g2 = torch.Generator(device="cuda").manual_seed(11)
        seed, offset = flashinfer.sampling.get_seed_and_offset(inc, g2, dev)
        ref = DR.draw_support_reference(one, [seed], [offset], [f], from_logits=fl)
        assert torch.equal(tok.cpu().long(), ref["samples"]) and bool(
            ref["valid"].all()
        ), fn.__name__
        assert torch.equal(
            fn(one, *args, seed=seed, offset=offset, batch_invariant=True), tok
        )
        assert torch.equal(
            fn(
                one,
                *args,
                seed=torch.tensor([seed], device="cuda"),
                offset=torch.tensor([offset], device="cuda"),
                batch_invariant=True,
            ),
            tok,
        )
        g3 = torch.Generator(device="cuda").manual_seed(11)
        assert torch.equal(
            fn(one, *args, seed=5, generator=g3, batch_invariant=True), tok
        )
        g_bad = torch.Generator(device="cuda").manual_seed(1)
        state = g_bad.get_state()
        for kw in (
            dict(generator=g_bad),
            dict(seed=3, offset=4),
            dict(
                seed=torch.tensor([3], device="cuda"),
                offset=torch.tensor([4], device="cuda"),
            ),
        ):
            with pytest.raises(ValueError, match="one key per output row"):
                fn(inp, *args, batch_invariant=True, **kw)
        assert torch.equal(g_bad.get_state(), state), (
            "a refused call must not advance the generator"
        )
        s = torch.full((M,), 3, dtype=torch.int64, device="cuda")
        o = torch.arange(M, device="cuda")
        got = fn(inp, *args, seed=s, offset=o, batch_invariant=True)
        ref = DR.draw_support_reference(
            inp, s.tolist(), o.tolist(), [f] * M, from_logits=fl
        )
        assert torch.equal(got.cpu().long(), ref["samples"]), fn.__name__
    big = 2**63 + 5
    su = torch.full((M,), big, dtype=torch.uint64, device="cuda")
    ou = torch.full((M,), 2**64 - 3, dtype=torch.uint64, device="cuda")
    tok = flashinfer.sampling.top_k_sampling_from_probs(
        p, 5, seed=su, offset=ou, batch_invariant=True
    )
    ref = DR.draw_support_reference(
        p, [big] * M, [2**64 - 3] * M, [Filters(top_k=5)] * M
    )
    assert torch.equal(tok.cpu().long(), ref["samples"])
    with pytest.raises(ValueError, match="length"):
        flashinfer.sampling.top_k_sampling_from_probs(
            p,
            5,
            seed=torch.zeros(5, dtype=torch.int64, device="cuda"),
            offset=ou,
            batch_invariant=True,
        )
    with pytest.raises(ValueError, match="both"):
        flashinfer.sampling.top_k_sampling_from_probs(
            p, 5, seed=su, offset=3, batch_invariant=True
        )


@needs_device
def test_degenerate_parameters_and_invalid_rows():
    """``k <= 0`` keeps nothing (invalid), ``k >= N`` everything (the plain draw); ``p >= 1`` every positive weight (the
    plain draw); ``min_p = 1`` the maximum (ties), ``min_p > 1`` nothing, ``min_p = 0`` every positive weight; an all-zero
    row and a row with an infinite weight are invalid; every case the exact model's; ``return_valid`` reports them."""
    N, M = 1000, 8
    w = weights(M, N, torch.float32, seed=6)
    w[6] = 0.0
    w[6, 10] = float("inf")
    s, o = keys(M, 5)
    ks = torch.tensor([0, -3, N, 2 * N, 1, 5, 5, 5], dtype=torch.int32, device="cuda")
    r = probs_draw(w, s, o, top_k=ks)
    _same(
        r,
        DR.draw_support_reference(
            w, s.tolist(), o.tolist(), [Filters(top_k=int(k)) for k in ks]
        ),
    )
    want = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1], dtype=torch.bool, device="cuda")
    assert torch.equal(r["valid"], want)
    assert torch.equal(
        r["samples"][2:4],
        sampling_from_probs_batch_invariant(w[2:4], seed=s[2:4], offset=o[2:4]),
    )
    assert int(r["samples"][4]) == int(w[4].argmax())
    assert int(r["samples"][7]) == 11  # the one-hot row
    tk, vk = flashinfer.sampling.top_k_sampling_from_probs(
        w, ks, seed=s, offset=o, return_valid=True, batch_invariant=True
    )
    assert torch.equal(vk, want) and torch.equal(tk, r["samples"])
    ps = torch.tensor([0.0, 1.0, 1.5, 0.5, 0.999, 0.5, 0.5, 1e-9], device="cuda")
    r = probs_draw(w, s, o, top_p=ps)
    _same(
        r,
        DR.draw_support_reference(
            w, s.tolist(), o.tolist(), [Filters(top_p=float(p)) for p in ps]
        ),
    )
    assert torch.equal(
        r["samples"][1:3],
        sampling_from_probs_batch_invariant(w[1:3], seed=s[1:3], offset=o[1:3]),
    )
    assert not bool(r["valid"][5]) and not bool(r["valid"][6])
    ms = torch.tensor([0.0, 1.0, 1.5, 0.3, 0.0, 0.0, 0.0, 0.5], device="cuda")
    r = probs_draw(w, s, o, min_p=ms)
    _same(
        r,
        DR.draw_support_reference(
            w, s.tolist(), o.tolist(), [Filters(min_p=float(m)) for m in ms]
        ),
    )
    assert int(r["samples"][1]) == int(w[1].argmax())  # min_p = 1: the maximum alone
    assert not bool(r["valid"][2])  # min_p > 1 keeps nothing
    assert torch.equal(
        r["samples"][0:1],
        sampling_from_probs_batch_invariant(w[0:1], seed=s[:1], offset=o[:1]),
    )
    assert not bool(r["valid"][5]) and not bool(r["valid"][6])


@needs_device
def test_distribution_follows_the_kept_set():
    """4096 draws of one row (distinct offsets) never leave the kept set and their frequencies match the kept set's
    renormalisation ``p_j / mass`` within sampling error."""
    N, M = 1000, 4096
    row = weights(1, N, torch.float32, seed=7)[0]
    w = row.repeat(M, 1).contiguous()
    s = torch.full((M,), 12345, dtype=torch.int64, device="cuda")
    o = torch.arange(M, device="cuda", dtype=torch.int64)
    sup = sampling_support(w[:1], top_k=5, top_p=0.9)
    kept = sup.mask[0]
    p = torch.where(kept, row, torch.zeros_like(row))
    p = (p / p.sum()).cpu()
    tok = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        w, 5, 0.9, seed=s, offset=o, batch_invariant=True
    ).cpu()
    assert bool(kept.cpu()[tok.long()].all())
    freq = torch.bincount(tok.long(), minlength=N).float() / M
    tol = 4 * torch.sqrt(p * (1 - p) / M) + 1e-3
    assert bool(((freq - p).abs() <= tol).all())
    assert len(set(tok.tolist())) >= 2


@needs_device
def test_big_vocabulary():
    """152064 (Qwen's vocabulary) in bf16 logits under all three filters with per-row temperatures: the model's bits
    across two members and the dispatcher's pick."""
    M, N = 2, N_BIG
    x = tied_logits(M, N, torch.bfloat16, seed=1)
    s, o = keys(M, 2)
    temps = torch.rand(M, device="cuda") + 0.5
    kw, filters = params("kpm", M, N, seed=3)
    ref = DR.draw_support_reference(
        x, s.tolist(), o.tolist(), filters, from_logits=True, temperature=temps
    )
    for cfg in _configs(N, 2, 2) + [None]:
        r = logits_draw(
            x, s, o, temperature=temps, return_logprob=True, config=cfg, **kw
        )
        _same(r, ref, lp=True, label=cfg.describe() if cfg else "default")


@needs_device
def test_sm_count_independence():
    """Where the SM counts split the dispatcher's pick the tokens and the kept mass are the same bits."""
    for N, M in SM_SPLIT_ROWS:
        w = weights(M, N, torch.bfloat16, seed=3)
        s, o = keys(M, 6)
        kw, _ = params("kp", M, N, seed=4)
        picks = {DS.support_config(N, M, 2, sm).describe() for sm in SM_COUNTS}
        assert len(picks) >= 2, picks
        outs = [probs_draw(w, s, o, sm_count=sm, **kw) for sm in SM_COUNTS]
        for r in outs[1:]:
            assert torch.equal(r["samples"], outs[0]["samples"])
            assert torch.equal(r["mass"], outs[0]["mass"])


@needs_device
def test_determinism_graph_replay_and_pdl():
    """R1: repeated launches, PDL off and on, and CUDA-graph replay give the same tokens."""
    N, M = 4096, 5
    p = weights(M, N, torch.bfloat16)
    x = tied_logits(M, N, torch.bfloat16)
    s, o = keys(M, 7)

    def fk():
        return flashinfer.sampling.top_k_sampling_from_probs(
            p, 7, seed=s, offset=o, batch_invariant=True
        )

    def fx():
        return flashinfer.sampling.top_k_top_p_sampling_from_logits(
            x, 7, 0.8, seed=s, offset=o, batch_invariant=True
        )

    ref_p, ref_x = fk(), fx()
    for _ in range(3):
        assert torch.equal(fk(), ref_p) and torch.equal(fx(), ref_x)
    for pdl in (False, True):
        assert torch.equal(
            draw_support(p, s, o, top_k=7, enable_pdl=pdl)["samples"], ref_p
        )
        assert torch.equal(
            draw_support(x, s, o, from_logits=True, top_k=7, top_p=0.8, enable_pdl=pdl)[
                "samples"
            ],
            ref_x,
        )
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        tp, tx = fk(), fx()
        torch.cuda.current_stream().synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            tp, tx = fk(), fx()
    torch.cuda.synchronize()
    tp.fill_(-7)
    tx.fill_(-7)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(tp, ref_p) and torch.equal(tx, ref_x)


@needs_device
def test_preallocated_outputs():
    """Caller-owned output buffers on the public entries (``out=`` on every truncated sampler; ``logprob_out=`` /
    ``kept_logprob_out=`` on the logits one, implying their ``return_*``): written in place and returned as given, bit
    for bit the allocating call; the contract (dtype, shape) refused rather than copied; a log-probability buffer on
    weights refused; the default paths refuse the buffers."""
    N, M = 1000, 6
    w = weights(M, N, torch.float32)
    s, o = keys(M, 4)
    ks = torch.tensor([1, 5, 0, N, 50, 7], dtype=torch.int32, device="cuda")
    ref = flashinfer.sampling.top_k_sampling_from_probs(
        w, ks, seed=s, offset=o, batch_invariant=True
    )
    ob = torch.empty(M, dtype=torch.int32, device="cuda")
    r = flashinfer.sampling.top_k_sampling_from_probs(
        w, ks, seed=s, offset=o, batch_invariant=True, out=ob
    )
    assert r is ob and torch.equal(ob, ref)
    r2, v2 = flashinfer.sampling.top_k_sampling_from_probs(
        w, ks, seed=s, offset=o, return_valid=True, batch_invariant=True, out=ob
    )
    assert r2 is ob and v2.dtype == torch.bool and torch.equal(ob, ref)
    ps = torch.tensor([0.9, 0.5, 0.0, 1.0, 0.3, 0.99], device="cuda")
    x = tied_logits(M, N, torch.bfloat16, seed=2)
    ref_t, ref_lp, ref_lpk = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        x,
        ks,
        ps,
        seed=s,
        offset=o,
        batch_invariant=True,
        return_logprob=True,
        return_kept_logprob=True,
    )
    tb = torch.empty(M, dtype=torch.int32, device="cuda")
    lb = torch.empty(M, dtype=torch.float32, device="cuda")
    kb = torch.empty(M, dtype=torch.float32, device="cuda")
    t2, l2, k2 = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        x,
        ks,
        ps,
        seed=s,
        offset=o,
        batch_invariant=True,
        out=tb,
        logprob_out=lb,
        kept_logprob_out=kb,
    )
    assert t2 is tb and l2 is lb and k2 is kb and torch.equal(tb, ref_t)
    _same_nan(lb, ref_lp, "logprob_out")
    _same_nan(kb, ref_lpk, "kept_logprob_out")
    # the contract: refused, never copied; a log-probability buffer on weights; the default paths
    with pytest.raises(ValueError, match="dtype"):
        flashinfer.sampling.top_k_sampling_from_probs(
            w, ks, seed=s, offset=o, batch_invariant=True, out=lb
        )
    with pytest.raises(ValueError, match="shape"):
        flashinfer.sampling.top_k_top_p_sampling_from_logits(
            x,
            ks,
            ps,
            seed=s,
            offset=o,
            batch_invariant=True,
            out=torch.empty(M + 1, dtype=torch.int32, device="cuda"),
        )
    with pytest.raises(ValueError, match="log-probability"):
        probs_draw(w, s, o, top_k=5, logprob_out=lb)
    with pytest.raises(ValueError, match="batch_invariant=True"):
        flashinfer.sampling.top_k_sampling_from_probs(w, 7, seed=s, offset=o, out=ob)
    with pytest.raises(ValueError, match="batch_invariant=True"):
        flashinfer.sampling.top_k_top_p_sampling_from_logits(
            x, ks, ps, seed=s, offset=o, out=tb
        )
    # overlap: an output may not overlap the input's storage, nor another output (a dtype view included)
    with pytest.raises(ValueError, match="overlap"):
        flashinfer.sampling.top_k_top_p_sampling_from_logits(
            x,
            ks,
            ps,
            seed=s,
            offset=o,
            batch_invariant=True,
            out=x.view(-1)[: 2 * M].view(torch.int32),
        )
    with pytest.raises(ValueError, match="overlap"):
        flashinfer.sampling.top_k_top_p_sampling_from_logits(
            x,
            ks,
            ps,
            seed=s,
            offset=o,
            batch_invariant=True,
            out=lb.view(torch.int32),
            logprob_out=lb,
        )
    with pytest.raises(ValueError, match="overlap"):
        flashinfer.sampling.top_k_sampling_from_probs(
            w, ks, seed=s, offset=o, batch_invariant=True, out=ks
        )


@needs_device
def test_default_path_is_the_operator_kernel(monkeypatch):
    """``batch_invariant=False`` (the default) never enters the classes and is byte for byte the operator's own kernel
    (the registered custom op the unmodified entries called), for every entry."""
    import flashinfer.batch_invariant.sampling.draw_support_api as dsapi

    def boom(*a, **k):
        raise AssertionError("the batch-invariant classes ran without the flag")

    for op in ENTRIES:
        monkeypatch.setattr(dsapi, f"{op}_batch_invariant", boom)
    torch.manual_seed(0)
    N, M = 32000, 16
    probs = torch.softmax(torch.randn(M, N, device="cuda") * 2, dim=-1)
    lg = torch.randn(M, N, device="cuda") * 2
    seed = torch.full((M,), 1234, dtype=torch.int64, device="cuda")
    off = torch.full((M,), 7, dtype=torch.int64, device="cuda")
    mod = flashinfer.sampling.get_sampling_module()
    tts = flashinfer.sampling._to_tensor_scalar_tuple
    a, v = flashinfer.sampling.top_k_sampling_from_probs(
        probs, 40, seed=seed, offset=off, return_valid=True
    )
    b, w_ = mod.top_k_sampling_from_probs(
        probs, None, *tts(40), True, None, seed, off, True
    )
    assert torch.equal(a, b) and torch.equal(v, w_)
    assert torch.equal(
        flashinfer.sampling.top_p_sampling_from_probs(
            probs, 0.9, seed=seed, offset=off
        ),
        mod.top_p_sampling_from_probs(
            probs, None, *tts(0.9), True, None, seed, off, False
        ),
    )
    assert torch.equal(
        flashinfer.sampling.min_p_sampling_from_probs(
            probs, 0.05, seed=seed, offset=off
        ),
        mod.min_p_sampling_from_probs(
            probs, None, *tts(0.05), True, None, seed, off, False
        ),
    )
    assert torch.equal(
        flashinfer.sampling.top_k_top_p_sampling_from_probs(
            probs, 40, 0.9, filter_apply_order="joint", seed=seed, offset=off
        ),
        mod.top_k_top_p_sampling_from_probs(
            probs, None, *tts(40), *tts(0.9), True, None, seed, off, False
        ),
    )
    # the operator's own compositions (top_k_first, the logits entry) run without entering the classes
    t = flashinfer.sampling.top_k_top_p_sampling_from_probs(
        probs, 40, 0.9, seed=seed, offset=off
    )
    tl = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        lg, 40, 0.9, seed=seed, offset=off
    )
    assert t.shape == (M,) and tl.shape == (M,)
    # the logits entry's temperature and log-probabilities exist under the flag only: the default path refuses them
    for kw in (
        {"temperature": 0.7},
        {"return_logprob": True},
        {"return_kept_logprob": True},
    ):
        with pytest.raises(ValueError, match="batch_invariant=True"):
            flashinfer.sampling.top_k_top_p_sampling_from_logits(
                lg, 40, 0.9, seed=seed, offset=off, **kw
            )
    for fn, x, args in (
        (flashinfer.sampling.top_k_sampling_from_probs, probs, (40,)),
        (flashinfer.sampling.top_p_sampling_from_probs, probs, (0.9,)),
        (flashinfer.sampling.min_p_sampling_from_probs, probs, (0.05,)),
        (flashinfer.sampling.top_k_top_p_sampling_from_probs, probs, (40, 0.9)),
        (flashinfer.sampling.top_k_top_p_sampling_from_logits, lg, (40, 0.9)),
    ):
        with pytest.raises(AssertionError, match="without the flag"):
            fn(x, *args, seed=seed, offset=off, batch_invariant=True)


@needs_device
def test_shapes_views_and_empty_batches():
    """3-D inputs (tokens of the leading shape; a per-row parameter of that shape), a view into a wider tensor and a
    strided view (copied: a placement), M = 0, a gather from an empty source, the unsupported dtypes."""
    N = 4096
    s, o = keys(6, 8)
    x3 = weights(6, N, torch.bfloat16).view(2, 3, N)
    t3, v3 = flashinfer.sampling.top_k_sampling_from_probs(
        x3, 7, seed=s, offset=o, return_valid=True, batch_invariant=True
    )
    assert t3.shape == (2, 3) and t3.dtype == torch.int32
    assert v3.shape == (2, 3) and v3.dtype == torch.bool
    flat = flashinfer.sampling.top_k_sampling_from_probs(
        x3.view(6, N), 7, seed=s, offset=o, batch_invariant=True
    )
    assert torch.equal(t3.view(6), flat)
    ks = torch.tensor([[3, 7, 9], [1, N, 2]], dtype=torch.int32, device="cuda")
    assert torch.equal(
        flashinfer.sampling.top_k_sampling_from_probs(
            x3, ks, seed=s, offset=o, batch_invariant=True
        ).view(6),
        flashinfer.sampling.top_k_sampling_from_probs(
            x3.view(6, N), ks.view(6), seed=s, offset=o, batch_invariant=True
        ),
    )
    xl = tied_logits(6, N, torch.bfloat16).view(2, 3, N)
    assert torch.equal(
        flashinfer.sampling.top_k_top_p_sampling_from_logits(
            xl, 7, 0.8, seed=s, offset=o, batch_invariant=True
        ).view(6),
        flashinfer.sampling.top_k_top_p_sampling_from_logits(
            xl.view(6, N), 7, 0.8, seed=s, offset=o, batch_invariant=True
        ),
    )
    wide = weights(6, 2 * N, torch.bfloat16, seed=1)
    for v in (wide[:, :N], wide[:, ::2]):
        assert torch.equal(
            flashinfer.sampling.top_k_sampling_from_probs(
                v, 7, seed=s, offset=o, batch_invariant=True
            ),
            flashinfer.sampling.top_k_sampling_from_probs(
                v.contiguous(), 7, seed=s, offset=o, batch_invariant=True
            ),
        )
    x0 = torch.empty(0, N, device="cuda", dtype=torch.bfloat16)
    e = torch.empty(0, dtype=torch.int64, device="cuda")
    t0, v0 = flashinfer.sampling.top_k_sampling_from_probs(
        x0, 7, seed=e, offset=e, return_valid=True, batch_invariant=True
    )
    assert t0.shape == (0,) and v0.shape == (0,)
    assert flashinfer.sampling.top_k_top_p_sampling_from_logits(
        x0, 7, 0.8, seed=e, offset=e, batch_invariant=True
    ).shape == (0,)
    r = probs_draw(
        x0, s[:2], o[:2], top_k=7, indices=torch.tensor([0, 1], device="cuda")
    )
    assert torch.equal(
        r["samples"], torch.tensor([-1, -1], dtype=torch.int32, device="cuda")
    ) and not bool(r["valid"].any())
    for dt in (torch.float64, torch.int32):
        with pytest.raises(bi.BatchInvariantUnsupportedError):
            flashinfer.sampling.top_k_sampling_from_probs(
                torch.zeros(2, N, device="cuda", dtype=dt),
                7,
                seed=s[:2],
                offset=o[:2],
                batch_invariant=True,
            )
