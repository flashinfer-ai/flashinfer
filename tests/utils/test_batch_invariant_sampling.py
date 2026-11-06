# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""``flashinfer.sampling.softmax(logits, temperature, batch_invariant=True)``, ``flashinfer.sampling.log_probs`` and
``flashinfer.sampling.entropy``: the class ``lse.row`` (the row maximum and the sum of ``ex2.approx`` weights as fixed
trees over the vocabulary, the probabilities / token log-probabilities / log-sum-exp as fixed chains over them) and its
entropy member (a third tree with the sum's shape, accumulated in the same pass, and the entropy chain).

Device-free: the pinned fingerprints, every placement of the configuration space traces to the canonical trees, the
dispatcher's membership in the enumerated space and the device check. On an SM100 / SM103 / SM107 device with the CuTe
DSL: bit equality with the exact CPU model of the specification (``sampling/reference.py``: the trees with single-rounding
CPU ops and the two MUFU primitives tabulated on the device; the Fractions evaluation of the symbolic spec on short
rows) for bf16 / f16 / fp32 logits, a few vocabularies (an odd one and a long one among them) and the three temperature
forms; configuration agreement over single-CTA, cluster and staged members; SM-count independence; the batch-invariance
guarantee itself (a row alone equals the row inside a batch, inside another batch, permuted, in sub-batches);
determinism including CUDA-graph replay and PDL; the shared dataflow of the two entries; masked / NaN / infinite rows;
views, 3-D inputs, M = 0 and the error contract. Every bitwise assertion is against the class's own specification; the
one closeness check against torch is a sanity tolerance, not a promise.
"""

import pytest
import torch

import flashinfer
from flashinfer import batch_invariant as bi
from flashinfer.batch_invariant import canon
from flashinfer.batch_invariant.sampling import reference as R
from flashinfer.batch_invariant.sampling import schedule as C
from flashinfer.batch_invariant.sampling import spec as S
from flashinfer.batch_invariant.sampling.api import (
    LSE_ROW_ARCHS,
    entropy_batch_invariant,
    log_probs_batch_invariant,
    lse_row,
    softmax_batch_invariant,
)

DT = [torch.bfloat16, torch.float16, torch.float32]
#: One block, a partial block (an odd vocabulary), a long odd vocabulary, a long one.
NS = [64, 1000, 4097, 32000]
N_BIG = 152064
SM_COUNTS = [148, 152, 204, 224]  # B200 / B300 (SM100 / SM103), Rubin (SM107) parts
_supported = torch.cuda.is_available() and bi.is_supported_device(
    torch.device("cuda"), LSE_ROW_ARCHS
)
needs_device = pytest.mark.skipif(
    not _supported,
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL",
)


def logits(M, N, dtype, seed=0, scale=2.0**-10):
    """Rows that make reduction-order differences visible: magnitudes spread over about 40 binades (12 for f16) with
    random signs, near-cancelling pairs, exact +0 and -0, subnormals and a few outliers, scaled into a logits-like
    range (an exponent span of about 2^40 around 2^-10). Every value is finite."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    exp_span = 20.0 if dtype != torch.float16 else 6.0
    e = (torch.rand(M, N, generator=g) * 2 - 1) * exp_span
    mant = torch.rand(M, N, generator=g) + 1.0
    sign = torch.where(torch.rand(M, N, generator=g) < 0.5, -1.0, 1.0)
    x = sign * mant * torch.exp2(e)
    pairs = torch.rand(M, N // 2, generator=g) < 0.15
    idx = torch.arange(0, N - 1, 2)
    x[:, idx + 1] = torch.where(
        pairs,
        -x[:, idx] * (1 + 1e-3 * torch.randn(M, N // 2, generator=g)),
        x[:, idx + 1],
    )
    r = torch.rand(M, N, generator=g)
    outlier = 2.0**12 if dtype != torch.float16 else 2.0**4
    x = torch.where(r < 0.01, x * outlier, x)
    x = torch.where((r >= 0.01) & (r < 0.02), torch.zeros_like(x), x)
    x = torch.where((r >= 0.02) & (r < 0.03), -torch.zeros_like(x), x)
    x = torch.where((r >= 0.03) & (r < 0.035), x * 2.0**-130, x)
    x = (x.to(dtype).float() * scale).to(dtype)
    assert torch.isfinite(x).all()
    return x.cuda()


def bits(t):
    t = t.detach().contiguous()
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def same(a, b) -> bool:
    return a.shape == b.shape and torch.equal(bits(a.cpu()), bits(b.cpu()))


def ids_for(x, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, x.shape[-1], x.shape[:-1], generator=g).cuda()


def run_all(x, temp=None, ids=None, config=None, enable_pdl=None, full=False):
    """The public entries (``probs``, ``logp``, ``lse2``, ``entropy``, and ``entropy2`` from the fused
    ``log_probs(return_lse, return_entropy)``) and, with ``full``, the generic entry's row log-softmax (``logprobs``)
    and log-sum-exp (``lse``)."""
    if ids is None:
        ids = ids_for(x)
    out = {
        "probs": flashinfer.sampling.softmax(x, temp, enable_pdl, batch_invariant=True)
    }
    out["logp"], out["lse2"], out["entropy2"] = flashinfer.sampling.log_probs(
        x, ids, temp, return_lse=True, return_entropy=True, enable_pdl=enable_pdl
    )
    out["entropy"] = flashinfer.sampling.entropy(x, temp, enable_pdl=enable_pdl)
    if config is not None:
        out["probs"] = softmax_batch_invariant(x, temp, enable_pdl, config=config)
        out["logp"], out["lse2"], out["entropy2"] = log_probs_batch_invariant(
            x,
            ids,
            temp,
            enable_pdl,
            return_lse=True,
            return_entropy=True,
            config=config,
        )
        out["entropy"] = entropy_batch_invariant(x, temp, enable_pdl, config=config)
    if full:
        pdl = bool(enable_pdl)
        out["logprobs"] = lse_row(
            S.LOG_SOFTMAX, x, temp, enable_pdl=pdl, config=config
        )["logprobs"]
        out["lse"] = lse_row(S.LSE, x, temp, enable_pdl=pdl, config=config)["lse"]
    return out


def check_ref(got, x, temp=None, ids=None, rows=None):
    """Every output equals the exact CPU model of the class, bit for bit."""
    if ids is None:
        ids = ids_for(x)
    ref = R.lse_reference(x, temp, ids, rows=rows)
    sel = (lambda t: t) if rows is None else (lambda t: t[list(rows)])
    for k, rk in (
        ("probs", "probs"),
        ("logp", "logp"),
        ("lse2", "lse"),
        ("logprobs", "logprobs"),
        ("lse", "lse"),
        ("entropy", "entropy"),
        ("entropy2", "entropy"),
    ):
        if k in got:
            assert same(sel(got[k]), ref[rk]), f"{k} is not the canonical dataflow"


# ----------------------------------------------------------------------------------------------- device-free


def test_fingerprints_are_pinned():
    """The receipts: one specification per (vocabulary, input dtype, temperature form); the two entries run one class;
    the hash is structural over the term DAG (``sterm.fingerprint_dag``)."""
    assert (
        bi.class_name("softmax") == "lse.row"
        and bi.class_name("log_probs") == "lse.row"
    )
    assert (
        bi.fingerprint("softmax", N=300, dtype="bf16", temperature=False)
        == "lse-row-89aa36214ccfeb63"
    )
    assert (
        bi.fingerprint("softmax", N=300, dtype=torch.bfloat16, temperature=True)
        == "lse-row-02dd3fe1ed41200e"
    )
    assert (
        bi.fingerprint("log_probs", N=300, dtype="f32", temperature=False)
        == "lse-row-209c8fa9070f30ed"
    )
    assert bi.fingerprint("log_probs", N=300, dtype="bf16", temperature=False) == (
        bi.fingerprint("softmax", N=300, dtype="bf16", temperature=False)
    )
    assert S.spec_fingerprint(301, "bf16") != S.spec_fingerprint(300, "bf16")
    # the entropy member: its own fingerprint (the W tree and the entropy chain embed the class's trees), the class's
    # fingerprints above unchanged by it; the input dtype does not enter (the leaves are exact widenings)
    assert bi.class_name("entropy") == "lse.row"
    assert (
        bi.fingerprint("entropy", N=300, dtype="bf16", temperature=False)
        == "lse-row-entropy-ddd184ec201e4d14"
    )
    assert (
        bi.fingerprint("entropy", N=300, dtype=torch.float32, temperature=True)
        == "lse-row-entropy-37c03b3db57c2f46"
    )
    assert bi.fingerprint("entropy", N=300, dtype="f16", temperature=False) == (
        S.entropy_fingerprint(300)
    )
    assert S.entropy_fingerprint(301) != S.entropy_fingerprint(300)
    assert S.ENTROPY.name == "h" and S.LOGP_LSE_ENTROPY.name == "sth"
    assert not S.ENTROPY.row_output and S.ENTROPY.needs_log
    # the hash is linear in the term DAG: a production vocabulary is a receipt, not a cost
    assert (
        bi.fingerprint("log_probs", N=32000, dtype="bf16", temperature=False)
        == "lse-row-c5477bcc11e0e90c"
    )
    with pytest.raises(TypeError):
        bi.fingerprint("softmax", N=300)


@pytest.mark.parametrize("N", [8, 300, 4097])
def test_every_placement_traces_to_the_canonical_trees(N):
    """The engine coupling: every member of the configuration space (single-CTA, cluster and staged) executes the
    canonical maximum and sum trees in every unit (``canon.verify`` on the symbolic terms)."""
    sp = S.LSESpec(N)
    cfgs = C.all_valid_configs(N, 2)
    assert cfgs
    rs, rw = sp.sum_reduction(0), sp.w_reduction(0)
    refs, refw = rs.canonical(sp.alg), rw.canonical(sp.alg)
    nm, c = sp.alg.neg(sp.m(0)), sp.c()

    class PairAlg:
        """The kernel's lockstep backend over the term algebra: every value is the pair (s, W) and the sum digits'
        const / add act componentwise (the digits call nothing else)."""

        def const(self, v):
            return (sp.alg.const(v), sp.alg.const(v))

        def add(self, a, b):
            return (sp.alg.add(a[0], b[0]), sp.alg.add(a[1], b[1]))

    def pair_leaf(fp, idx):
        i = S.flat_index(idx)
        if i >= N:
            return None
        z = S.shifted(sp.alg, sp.leaf(0, i), nm, c)
        e = S.weight(sp.alg, z)
        return (e, S.wprod(sp.alg, e, z))

    for cfg in cfgs:
        pl = cfg.placement(N)
        canon.verify(canon.plan(sp.max_reduction(0), pl), sp.alg)
        ps = canon.plan(rs, pl)
        canon.verify(ps, sp.alg)
        pw = canon.plan(rw, pl)
        assert canon.verify(pw, sp.alg) is refw
        # the entropy member's tree has exactly the sum's step list under every placement, so the kernel runs the sum
        # plan once over (s, W) pairs: traced on the term algebra, both components are the canonical terms (every unit)
        assert pw.steps == ps.steps
        got = canon.trace(ps, PairAlg(), pair_leaf)
        assert all(v[0] is refs and v[1] is refw for v in got.values()), cfg.describe()


def test_dispatcher_picks_are_members_of_the_space():
    """R3 membership: every configuration the closed-form dispatcher returns, for both regimes, both output kinds,
    both element widths and the SM counts of the SM100 / SM103 / SM107 parts, is a member of the enumerated space."""
    for N in (64, 1000, 4097, 32000, 152064):
        for eb in (2, 4):
            space = set(C.all_valid_configs(N, eb))
            for M in (1, 4, 37, 256, 4096):
                for sm in SM_COUNTS:
                    for row_output in (True, False):
                        cfg = C.default_config(N, M, eb, sm, row_output)
                        assert cfg in space, (N, eb, M, sm, row_output, cfg.describe())


def test_unsupported_inputs_raise_before_any_kernel():
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse.row"):
        softmax_batch_invariant(x)
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse.row"):
        log_probs_batch_invariant(x, torch.zeros(4, dtype=torch.int64))
    with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse.row"):
        entropy_batch_invariant(x)
    with pytest.raises(ValueError, match="batch_invariant"):
        flashinfer.sampling.log_probs(
            x, torch.zeros(4, dtype=torch.int64), batch_invariant=False
        )
    with pytest.raises(ValueError, match="batch_invariant"):
        flashinfer.sampling.entropy(x, batch_invariant=False)
    with pytest.raises(ValueError, match="CUDA"):
        flashinfer.sampling.entropy(x)


def test_entropy_select_in_the_exact_model():
    """``ZSEL(e, p) = +0 if e == 0 else p``: the -inf logit's NaN product becomes +0, both zero signs select, a NaN weight
    passes (the leaf's product is NaN then, so NaN rows stay NaN); the entropy chain evaluated with correctly rounded
    stand-ins for the two MUFU primitives is -(p ln p) to fp32 accuracy, exactly 0 on a one-hot row, NaN on an all-masked
    row (device-free)."""
    import math

    from flashinfer.batch_invariant.fp_interface import exact as fp32
    from flashinfer.batch_invariant.sampling import sterm

    nan, inf = float("nan"), float("inf")
    assert (
        R.zsel_exact(0.0, nan) == 0.0 and math.copysign(1.0, R.zsel_exact(0.0, nan)) > 0
    )
    assert math.copysign(1.0, R.zsel_exact(-0.0, -0.0)) > 0
    assert R.zsel_exact(1.0, -2.5) == -2.5 and R.zsel_exact(nan, 1.0) == 1.0
    assert math.isnan(R.zsel_exact(0.5, nan))
    e = torch.tensor([0.0, -0.0, 1.0, 0.25, nan, 0.0], dtype=torch.float32)
    z = torch.tensor([-inf, -140.0, 0.0, -2.0, -1.0, -3.0], dtype=torch.float32)
    w = R.row_w(e, z)
    assert (
        w[0] == 0 and w[1] == 0 and not torch.signbit(w[0]) and not torch.signbit(w[1])
    )
    assert w[2] == 0 and w[3] == -0.5 and torch.isnan(w[4]) and w[5] == 0

    def ex2(v):
        return nan if v != v else (fp32.to_f32(2.0**v) if v > -1000 else 0.0)

    def lg2(v):
        return (
            nan
            if (v != v or v < 0)
            else (-inf if v == 0 else fp32.to_f32(math.log2(v)))
        )

    syms = {"ex2": ex2, "op:lg2": lg2, "op:zsel": R.zsel_exact}
    x = (torch.randn(1, 40, generator=torch.Generator().manual_seed(0)) * 3).float()
    H = sterm.evaluate(S.LSESpec(40).entropy(0), lambda r, i: float(x[r, i]), {}, syms)
    p = torch.softmax(x.double(), -1)
    assert abs(H - float(-(p * p.log()).sum())) < 2e-6 * max(1.0, abs(H))
    x1 = torch.full((1, 40), -inf)
    x1[0, 7] = 2.0
    assert (
        sterm.evaluate(S.LSESpec(40).entropy(0), lambda r, i: float(x1[r, i]), {}, syms)
        == 0.0
    )
    x1[0, 7] = -inf
    assert math.isnan(
        sterm.evaluate(S.LSESpec(40).entropy(0), lambda r, i: float(x1[r, i]), {}, syms)
    )


# ----------------------------------------------------------------------------------------------- on the device


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["bf16", "f16", "f32"])
@pytest.mark.parametrize("N", NS)
def test_matches_exact_model(dtype, N):
    """Every output of both entries equals the exact CPU model of the class, under the three temperature forms."""
    x = logits(4, N, dtype)
    ids = ids_for(x)
    full = (dtype, N) in ((torch.bfloat16, 1000), (torch.float32, 64))
    check_ref(run_all(x, ids=ids, full=full), x, ids=ids)
    check_ref(run_all(x, 0.37, ids, full=full), x, 0.37, ids)
    temps = (torch.rand(4, device="cuda") * 1.5 + 0.25).to(dtype)
    check_ref(run_all(x, temps, ids, full=full), x, temps, ids)


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["bf16", "f16", "f32"])
@pytest.mark.parametrize("N", [64, 300])
def test_symbolic_oracle(dtype, N):
    """The vectorised exact model equals the Fractions evaluation of the symbolic spec, and so does the device."""
    x = logits(3, N, dtype)
    ids = torch.tensor([0, N - 1, N // 2], device="cuda")
    temps = torch.tensor([1.0, 0.5, 1.7], device="cuda")
    for temp in (None, temps):
        ex = R.exact_rows(x, temp, rows=range(3), token_ids=ids)
        ref = R.lse_reference(x, temp, ids)
        got = run_all(x, temp, ids)
        for k in ("probs", "logprobs", "lse", "logp", "m", "s", "W", "entropy"):
            assert same(ref[k], ex[k]), f"vectorised model {k} != symbolic spec"
        assert same(got["probs"], ex["probs"]) and same(got["logp"], ex["logp"])
        assert same(got["lse2"], ex["lse"])
        assert same(got["entropy"], ex["entropy"]) and same(
            got["entropy2"], ex["entropy"]
        )


def _config_sample(N, dtype):
    """Single-CTA (register and re-read), cluster and staged members: the whole space for bf16 at one vocabulary,
    one member of each kind elsewhere."""
    cfgs = C.all_valid_configs(N, dtype.itemsize)
    if dtype == torch.bfloat16 and N == 1000:
        return cfgs
    out = []
    for stage, cluster in (
        ("regs", 1),
        ("reread", 1),
        ("regs", 4),
        ("smem", 1),
        ("smem", 4),
    ):
        for c in cfgs:
            if c.stage == stage and c.cluster == cluster and c not in out:
                out.append(c)
                break
    return out


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["bf16", "f16", "f32"])
@pytest.mark.parametrize("N", [1000, 4097])
def test_configuration_agreement(dtype, N):
    """R3: every configuration gives the bits of the dispatcher's pick, for the softmax row and the token log-probabilities."""
    x = logits(37, N, dtype)
    ids = ids_for(x)
    temps = torch.rand(37, device="cuda") * 1.5 + 0.25
    ref = run_all(x, temps, ids)
    cfgs = _config_sample(N, dtype)
    assert cfgs
    for cfg in cfgs:
        got = run_all(x, temps, ids, config=cfg)
        for k in ref:
            assert same(got[k], ref[k]), f"{k}: config {cfg.describe()} differs (N={N})"


@needs_device
def test_sm_count_independence():
    """The dispatcher's pick depends on the SM count; the bits do not."""
    x = logits(37, 4097, torch.bfloat16)
    ids = ids_for(x)
    ref = softmax_batch_invariant(x), log_probs_batch_invariant(x, ids, 0.9)
    picks = set()
    for sm in SM_COUNTS:
        picks.add(C.default_config(4097, 37, 2, sm).describe())
        assert same(softmax_batch_invariant(x, sm_count=sm), ref[0])
        assert same(log_probs_batch_invariant(x, ids, 0.9, sm_count=sm), ref[1])
    x = logits(512, 32000, torch.bfloat16, seed=3)
    ref = softmax_batch_invariant(x)
    for sm in SM_COUNTS:
        assert same(softmax_batch_invariant(x, sm_count=sm), ref)


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["bf16", "f16", "f32"])
@pytest.mark.parametrize("N", NS)
def test_batch_invariance(dtype, N):
    """R2: a row alone equals the row inside a batch of 64 and inside another batch of 13, every output."""
    xb = logits(64, N, dtype)
    ids = ids_for(xb)
    full = run_all(xb, ids=ids)
    for i in (0, 17, 63):
        one = run_all(xb[i : i + 1].clone(), ids=ids[i : i + 1])
        for k in full:
            assert same(one[k][0], full[k][i]), (
                f"{k}: row {i} differs between M=1 and M=64"
            )
    xo = logits(13, N, dtype, seed=5)
    xo[4] = xb[17]
    io = ids_for(xo, seed=7)
    io[4] = ids[17]
    other = run_all(xo, ids=io)
    for k in full:
        assert same(other[k][4], full[k][17]), f"{k}: row differs in another batch"


@needs_device
def test_permutation_subbatch_and_regime_invariance():
    """R2 across the dispatcher's regimes: a batch of 512 rows (the throughput regime) against the same rows permuted and
    in sub-batches of 1 / 3 / 16 / 100 (the latency regime, other picks)."""
    x = logits(512, 4096, torch.bfloat16)
    ids = ids_for(x)
    full = run_all(x, 0.8, ids)
    perm = torch.randperm(512, generator=torch.Generator().manual_seed(1)).cuda()
    got = run_all(x[perm].contiguous(), 0.8, ids[perm])
    for k in full:
        assert same(got[k], full[k][perm]), k
    for k in (1, 3, 16, 100):
        idx = torch.randperm(512, generator=torch.Generator().manual_seed(k))[:k].cuda()
        got = run_all(x[idx].contiguous(), 0.8, ids[idx])
        for key in full:
            assert same(got[key], full[key][idx]), (key, k)


@needs_device
def test_big_vocabulary():
    """A long row (one CTA per 16 KB share in the latency regime): the exact model, the cluster, staged and re-read
    members and batch invariance against a batch of 512."""
    x = logits(2, N_BIG, torch.bfloat16)
    ids = torch.tensor([N_BIG - 1, 12345], device="cuda")
    got = run_all(x, ids=ids)
    check_ref(got, x, ids=ids)
    for cfg in (
        C.LSEConfig(256, 1, 8, "regs", 8),
        C.LSEConfig(512, 1, 8, "smem", 4),
        C.LSEConfig(1024, 1, 8, "reread"),
    ):
        g2 = run_all(x, ids=ids, config=cfg)
        for k in got:
            assert same(g2[k], got[k]), (k, cfg.describe())
    xb = logits(512, N_BIG, torch.bfloat16, seed=2)
    xb[100] = x[0]
    big = run_all(xb, ids=torch.full((512,), N_BIG - 1, device="cuda"))
    for k in big:
        assert same(big[k][100], got[k][0]), f"{k}: batch invariance at N={N_BIG}"


@needs_device
def test_temperature_forms():
    """``None``, a scalar, a per-row tensor of f16 / bf16 / f32 / f64 (f64 rounded to f32, declared); the invalid forms."""
    x = logits(6, 2048, torch.bfloat16)
    assert same(softmax_batch_invariant(x), softmax_batch_invariant(x, 1.0))
    assert same(
        softmax_batch_invariant(x),
        softmax_batch_invariant(x, torch.ones(6, device="cuda")),
    )
    base = torch.rand(6, device="cuda", dtype=torch.float64) * 2 + 0.1
    for dt in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        t = base.to(dt)
        assert same(softmax_batch_invariant(x, t), R.lse_reference(x, t)["probs"])
    assert same(
        softmax_batch_invariant(x, base),
        R.lse_reference(x, base.to(torch.float32))["probs"],
    )
    assert same(softmax_batch_invariant(x, 0.5), R.lse_reference(x, 0.5)["probs"])
    with pytest.raises(ValueError):
        softmax_batch_invariant(x, 0.0)
    with pytest.raises(ValueError):
        softmax_batch_invariant(x, torch.ones(5, device="cuda"))
    with pytest.raises(ValueError):
        flashinfer.sampling.softmax(
            x, torch.ones(5, device="cuda"), batch_invariant=True
        )


@needs_device
def test_shared_dataflow_of_the_entries():
    """R4 by construction: ``log_probs`` is the gathered log-softmax row of the class, its ``lse`` is the class's
    log-sum-exp, and an id outside the vocabulary gives NaN."""
    x = logits(16, 8192, torch.bfloat16)
    ids = ids_for(x)
    temps = torch.rand(16, device="cuda") + 0.3
    lp_row = lse_row(S.LOG_SOFTMAX, x, temps)["logprobs"]
    lp, lse = flashinfer.sampling.log_probs(x, ids, temps, return_lse=True)
    assert same(lp, lp_row[torch.arange(16, device="cuda"), ids])
    assert same(lse, lse_row(S.LSE, x, temps)["lse"])
    assert same(lp, flashinfer.sampling.log_probs(x, ids, temps))
    bad = torch.tensor([-1, 8192] + [1] * 14, device="cuda")
    lpb = flashinfer.sampling.log_probs(x, bad, temps)
    assert torch.isnan(lpb[:2]).all()
    assert same(lpb[2:], lp_row[torch.arange(2, 16, device="cuda"), bad[2:]])
    assert same(flashinfer.sampling.log_probs(x, ids.to(torch.int32), temps), lp)
    # the log-softmax entry is the class's row; K ids per row are the row gathered K times (one launch), NaN per id
    assert same(flashinfer.sampling.log_softmax(x, temps), lp_row)
    with pytest.raises(ValueError):
        flashinfer.sampling.log_softmax(x, temps, batch_invariant=False)
    ar = torch.arange(16, device="cuda")
    ids3 = torch.stack([ids, bad, ids.flip(0)], dim=1)
    lp3, lse3 = flashinfer.sampling.log_probs(x, ids3, temps, return_lse=True)
    assert lp3.shape == (16, 3) and lp3.dtype == torch.float32 and same(lse3, lse)
    assert same(lp3[:, 0], lp) and same(lp3[:, 2], lp_row[ar, ids.flip(0)])
    assert torch.isnan(lp3[:2, 1]).all() and same(lp3[2:, 1], lpb[2:])
    ids40 = torch.randint(0, 8192, (16, 40), device="cuda")  # two lane rounds of ids
    lp40 = flashinfer.sampling.log_probs(x, ids40, temps)
    assert lp40.shape == (16, 40) and same(lp40, lp_row[ar[:, None], ids40])
    assert same(flashinfer.sampling.log_probs(x, ids40.to(torch.int32), temps), lp40)
    assert same(flashinfer.sampling.log_probs(x, ids[:, None], temps), lp[:, None])
    # the entropy member: alone, with the token log-probability, with both, with K ids per row -- the same bits
    h = flashinfer.sampling.entropy(x, temps)
    assert h.shape == (16,) and h.dtype == torch.float32
    lp_h, h2 = flashinfer.sampling.log_probs(x, ids, temps, return_entropy=True)
    lp_h3, lse_h3, h3 = flashinfer.sampling.log_probs(
        x, ids, temps, return_lse=True, return_entropy=True
    )
    lp40h, h40 = flashinfer.sampling.log_probs(x, ids40, temps, return_entropy=True)
    assert (
        same(lp_h, lp) and same(lp_h3, lp) and same(lse_h3, lse) and same(lp40h, lp40)
    )
    assert same(h2, h) and same(h3, h) and same(h40, h)
    assert same(entropy_batch_invariant(x, temps), h)
    # softmax in fp32 is the fp32 input's class: the probabilities before the output conversion
    p32 = flashinfer.sampling.softmax(
        x, temps, batch_invariant=True, out_dtype=torch.float32
    )
    assert p32.dtype == torch.float32 and same(
        p32, flashinfer.sampling.softmax(x.float(), temps, batch_invariant=True)
    )
    assert same(
        flashinfer.sampling.softmax(
            x, temps, batch_invariant=True, out_dtype=torch.bfloat16
        ),
        flashinfer.sampling.softmax(x, temps, batch_invariant=True),
    )
    with pytest.raises(bi.BatchInvariantUnsupportedError):
        flashinfer.sampling.softmax(
            x, temps, batch_invariant=True, out_dtype=torch.float16
        )
    with pytest.raises(ValueError):
        flashinfer.sampling.softmax(x, temps, out_dtype=torch.float32)


@needs_device
def test_masked_nan_and_infinite_rows():
    """The declared corners: a masked row (``-inf`` logits), an all-masked row, a NaN logit, an infinite logit."""
    N = 1000
    x = logits(4, N, torch.float32)
    x[0, 10:] = float("-inf")
    x[1, :] = float("-inf")
    x[2, 5] = float("nan")
    x[3, 7] = float("inf")
    got = run_all(x, full=True)
    ref = R.lse_reference(x, None, ids_for(x))
    assert same(got["probs"][0], ref["probs"][0]) and bool(
        (got["probs"][0, 10:] == 0).all()
    )
    assert same(got["lse"][0], ref["lse"][0])
    assert same(got["logprobs"][0, :10], ref["logprobs"][0, :10])
    assert bool(torch.isinf(got["logprobs"][0, 10:]).all())
    for r in (1, 2, 3):
        assert bool(torch.isnan(got["probs"][r]).all())
        assert bool(torch.isnan(got["lse"][r])) and bool(
            torch.isnan(got["logprobs"][r]).all()
        )
        assert bool(torch.isnan(got["entropy"][r]))
    assert same(got["entropy"][0], ref["entropy"][0]) and float(got["entropy"][0]) > 0


@needs_device
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_entropy_select_rows(dtype):
    """The rows that exercise the member's select: a one-hot row (every other logit -inf) is exactly 0; a finite logit
    so far below the maximum that its weight flushes to +0, a masked half and two equal tokens (ln 2) equal the exact
    model; every member of a configuration sample agrees."""
    N = 4097
    x = torch.full((5, N), float("-inf"), device="cuda")
    x[0, 100] = 3.0
    x[1] = logits(1, N, torch.float32)[0]
    x[1, 7] = float(x[1].max()) - 200.0
    x[2] = logits(1, N, torch.float32, seed=3)[0]
    x[2, 2000:] = float("-inf")
    x[3, 5] = 1.0
    x[3, N - 1] = 1.0
    x[4] = logits(1, N, torch.float32, seed=4)[0]
    xd = x.to(dtype)
    got = flashinfer.sampling.entropy(xd)
    assert same(got, R.lse_reference(xd)["entropy"])
    assert float(got[0]) == 0.0 and not torch.signbit(got[0])
    assert abs(float(got[3]) - 0.6931471805599453) < 2e-7
    assert torch.isfinite(got).all()
    for cfg in _config_sample(N, dtype):
        assert same(entropy_batch_invariant(xd, config=cfg), got), cfg.describe()


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["bf16", "f16", "f32"])
def test_entropy_sanity_against_fp64(dtype):
    """A tolerance check only: the member against -(p ln p) in fp64 and against verl's ``entropy_from_logits``
    formula ``logsumexp(x) - (softmax(x) * x).sum`` in fp32, with and without a temperature."""
    N = 32000
    x = (torch.randn(8, N, device="cuda") * 4).to(dtype)
    for temp in (None, 0.7, torch.rand(8, device="cuda") + 0.5):
        ours = flashinfer.sampling.entropy(x, temp)
        xs = x.double()
        if temp is not None:
            xs = xs / (
                temp.double()[:, None] if isinstance(temp, torch.Tensor) else temp
            )
        p = torch.softmax(xs, -1)
        assert torch.allclose(
            ours.double(), -(p * torch.log(p)).sum(-1), rtol=2e-5, atol=2e-5
        )
        xf = xs.float()
        verl = torch.logsumexp(xf, -1) - (torch.softmax(xf, -1) * xf).sum(-1)
        assert torch.allclose(ours, verl, rtol=1e-4, atol=1e-4)


@needs_device
def test_determinism_graph_replay_and_pdl():
    """R1: repeated launches, PDL on, and CUDA-graph replay give the same bits."""
    x = logits(5, 4096, torch.bfloat16)
    ids = ids_for(x)
    temps = torch.rand(5, device="cuda") + 0.5
    ref = run_all(x, temps, ids)
    for _ in range(3):
        got = run_all(x, temps, ids)
        assert all(same(got[k], ref[k]) for k in ref)
    got = run_all(x, temps, ids, enable_pdl=True)
    assert all(same(got[k], ref[k]) for k in ref)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        p = softmax_batch_invariant(x, temps)
        lp, lse, h = log_probs_batch_invariant(
            x, ids, temps, return_lse=True, return_entropy=True
        )
        torch.cuda.current_stream().synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            p = softmax_batch_invariant(x, temps)
            lp, lse, h = log_probs_batch_invariant(
                x, ids, temps, return_lse=True, return_entropy=True
            )
    torch.cuda.synchronize()
    p.zero_()
    lp.zero_()
    lse.zero_()
    h.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert same(p, ref["probs"]) and same(lp, ref["logp"]) and same(lse, ref["lse2"])
    assert same(h, ref["entropy"])


@needs_device
def test_shapes_views_and_empty_batches():
    """3-D logits, a view into a wider tensor (read in place when its row stride is 16-byte aligned, copied otherwise:
    a placement), the output shapes and dtypes, K ids per row in 3-D, M = 0."""
    N = 4096
    x = logits(8, 2 * N, torch.bfloat16)[:, :N]
    assert same(softmax_batch_invariant(x), softmax_batch_invariant(x.contiguous()))
    ids8 = ids_for(x)
    assert same(
        log_probs_batch_invariant(x, ids8),
        log_probs_batch_invariant(x.contiguous(), ids8),
    )
    xo = logits(8, N + 8, torch.bfloat16)[
        :, :N
    ]  # an odd pad: a 16-byte-aligned stride still, in place
    assert same(softmax_batch_invariant(xo), softmax_batch_invariant(xo.contiguous()))
    x3 = logits(6, N, torch.bfloat16).view(2, 3, N)
    ids3 = torch.zeros(2, 3, dtype=torch.long, device="cuda")
    p3 = softmax_batch_invariant(x3)
    assert p3.shape == (2, 3, N) and p3.dtype == torch.bfloat16
    assert same(p3.view(6, N), softmax_batch_invariant(x3.view(6, N)))
    lp3, lse3 = log_probs_batch_invariant(x3, ids3, return_lse=True)
    assert lp3.shape == (2, 3) and lse3.shape == (2, 3) and lp3.dtype == torch.float32
    assert same(lp3.view(6), log_probs_batch_invariant(x3.view(6, N), ids3.view(6)))
    ids3k = torch.randint(0, N, (2, 3, 5), device="cuda")
    lp3k = log_probs_batch_invariant(x3, ids3k)
    assert lp3k.shape == (2, 3, 5)
    assert same(
        lp3k.view(6, 5), log_probs_batch_invariant(x3.view(6, N), ids3k.view(6, 5))
    )
    assert flashinfer.sampling.log_softmax(x3).shape == (2, 3, N)
    with pytest.raises(ValueError):
        log_probs_batch_invariant(x3, ids3.view(6))
    with pytest.raises(ValueError):
        log_probs_batch_invariant(
            x3, torch.zeros(2, 3, 0, dtype=torch.long, device="cuda")
        )
    assert flashinfer.sampling.entropy(x3).shape == (2, 3)
    assert same(
        flashinfer.sampling.entropy(x3).view(6),
        flashinfer.sampling.entropy(x3.view(6, N)),
    )
    x0 = torch.empty(0, N, device="cuda", dtype=torch.bfloat16)
    assert softmax_batch_invariant(x0).shape == (0, N)
    lp0, lse0, h0 = log_probs_batch_invariant(
        x0,
        torch.empty(0, dtype=torch.long, device="cuda"),
        return_lse=True,
        return_entropy=True,
    )
    assert lp0.shape == (0,) and lse0.shape == (0,) and h0.shape == (0,)
    assert flashinfer.sampling.entropy(x0).shape == (0,)
    for dt in (torch.float64, torch.int32):
        with pytest.raises(bi.BatchInvariantUnsupportedError, match="lse.row"):
            softmax_batch_invariant(torch.zeros(2, N, device="cuda", dtype=dt))


@needs_device
def test_default_path_does_not_enter_the_class(monkeypatch):
    """``softmax`` without the flag runs the operator's own kernels: the class is never imported on that path."""
    import flashinfer.batch_invariant.sampling.api as sapi

    def boom(*a, **k):
        raise AssertionError("the batch-invariant class ran without the flag")

    monkeypatch.setattr(sapi, "softmax_batch_invariant", boom)
    x = logits(4, 1000, torch.float32)
    p = flashinfer.sampling.softmax(x)
    assert p.shape == x.shape
    with pytest.raises(AssertionError, match="without the flag"):
        flashinfer.sampling.softmax(x, batch_invariant=True)


@needs_device
@pytest.mark.parametrize("dtype", DT, ids=["bf16", "f16", "f32"])
def test_sanity_against_torch(dtype):
    """A tolerance check only (nothing is promised relative to another implementation): the class's probabilities and
    log-probabilities are close to torch's fp32 softmax / log-softmax of the scaled rows."""
    N = 32000
    x = (torch.randn(8, N, device="cuda") * 3).to(dtype)
    ids = ids_for(x)
    temps = torch.rand(8, device="cuda") + 0.5
    p = flashinfer.sampling.softmax(x, temps, batch_invariant=True).float()
    lp, lse = flashinfer.sampling.log_probs(x, ids, temps, return_lse=True)
    xs = x.float() / temps[:, None]
    tol = {torch.bfloat16: 1e-2, torch.float16: 2e-3, torch.float32: 2e-6}[dtype]
    pt = torch.softmax(xs, dim=-1)
    assert torch.allclose(p, pt, rtol=tol, atol=tol * pt.max().item())
    assert torch.allclose(
        p.sum(-1),
        torch.ones(8, device="cuda"),
        atol=5e-3 if dtype != torch.float32 else 1e-5,
    )
    lpt = torch.log_softmax(xs, dim=-1)
    assert torch.allclose(
        lp, lpt[torch.arange(8, device="cuda"), ids], rtol=1e-5, atol=2e-5
    )
    assert torch.allclose(lse, torch.logsumexp(xs, dim=-1), rtol=1e-6, atol=2e-5)
