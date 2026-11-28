# Copyright (c) 2026 by FlashInfer team.
"""Experimental tcgen05 / UMMA backend of the KDA output-only (frozen-state) decode.

Re-runs every case of ``tests/kda/test_kda_output_only.py`` (its fp32 references,
the vLLM RecoverSSM drop-in contract and the ``recurrent_kda`` frozen-mode API) with
the UMMA backend forced: the base module's entry points are swapped for wrappers that
pass ``backend="umma"`` (autouse fixture below), so no reference or case is copied.
UMMA-only tests (WY parity across T = 1..16 and every gate mode, determinism, the
``backend="auto"`` experimental gate) follow.
"""

import functools

import pytest
import torch
import torch.nn.functional as F

from tests.kda import test_kda_output_only as _base
from tests.kda.test_kda_output_only import *  # noqa: F401,F403  (re-collected below)

if _base._KDA_OUTPUT_ONLY_AVAILABLE:
    import flashinfer.kda_decode as _kda_decode
    import flashinfer.kda_kernels.kda_decode_wy_output_only as _oo
    from flashinfer.experimental.kda_output_only_umma import launch as _launch
    from flashinfer.experimental.kda_output_only_umma import (
        kda_oo_umma_unavailable_reason,
    )

    _WY_DENSE = _oo.kda_wy_output_only
    _WY_DROPIN = _oo.kda_recoverssm_verify


def _forced(fn, *args, **kwargs):
    kwargs["backend"] = "umma"
    return fn(*args, **kwargs)


def _launches() -> int:
    return _launch._LAUNCHES


@pytest.fixture(autouse=True)
def _umma_backend(request, monkeypatch):
    """Route the base tests' entry points to ``backend="umma"``; skip off SM10x.
    Base cases parametrized over another backend run once (as their ``"wy"`` case)."""
    if not torch.cuda.is_available() or not _base._KDA_OUTPUT_ONLY_AVAILABLE:
        pytest.skip("needs CUDA and the CuTe-DSL KDA output-only kernels")
    why = kda_oo_umma_unavailable_reason(torch.device("cuda"), 1)
    if why is not None:
        pytest.skip(f"KDA output-only UMMA backend unavailable: {why}")
    params = getattr(getattr(request.node, "callspec", None), "params", {})
    if params.get("backend", "wy") != "wy":
        pytest.skip("the UMMA re-run keeps one case per backend parametrization")
    dense = functools.partial(_forced, _WY_DENSE)
    dropin = functools.partial(_forced, _WY_DROPIN)
    monkeypatch.setattr(_base, "kda_output_only_decode", dense)
    monkeypatch.setattr(_oo, "kda_wy_output_only", dense)
    monkeypatch.setattr(_oo, "kda_recoverssm_verify", dropin)
    monkeypatch.setattr(_kda_decode, "_run_kda_output_only", dense)
    n0 = _launches()
    yield
    if request.node.originalname not in _NO_LAUNCH_EXPECTED:
        assert _launches() > n0, "the UMMA backend did not run"


# Base tests that raise before any launch (contract checks).
_NO_LAUNCH_EXPECTED = {
    "test_frozen_mode_public_api_errors",
    "test_frozen_mode_shape_validation",
    "test_umma_auto_needs_experimental_opt_in",
    "test_umma_backend_rejects_bad_values",
    "test_dropin_rejects_misaligned_token_stride",
    "test_dense_rejects_unknown_backend",
    "test_dropin_rejects_negative_null_min",
    "test_dropin_explicit_umma_raises_on_empty_batch_off_sm10x",
    "test_umma_persistent_policy_off_when_min_blocks_forced",
}


def _inputs(B, T, H, HV, gate, seed):
    g_ = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"

    def rn(*shape):
        return torch.randn(*shape, generator=g_, device=dev)

    q = rn(B, T, H, 128).bfloat16()
    k = rn(B, T, H, 128).bfloat16()
    v = rn(B, T, HV, 128).bfloat16()
    g = rn(B, T, HV, 128)
    g = (F.logsigmoid(g) if gate == "pre" else g).bfloat16()
    beta = torch.rand(B, T, HV, generator=g_, device=dev).bfloat16()
    pool = rn(B + 3, HV, 128, 128).mul(0.1).bfloat16()
    idx = torch.randperm(B + 3, generator=g_, device=dev)[:B].int()
    A_log = rn(H) * 0.3
    dt_bias = rn(H * 128) * 0.1
    kw = dict(
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=gate != "pre",
        lower_bound=-5.0 if gate == "lb" else None,
    )
    return (q, k, v, g, beta, pool, idx), kw


@pytest.mark.parametrize("T", list(range(1, 17)))
@pytest.mark.parametrize("gate", ["pre", "lb", "sp"])
@pytest.mark.parametrize("H,HV", [(4, 4), (2, 4)])
def test_umma_matches_wy(T, gate, H, HV):
    """UMMA vs the WY kernel and the fp32 grouped recurrence on identical inputs:
    outputs and corrections within bf16 rounding of the fp32 reference and no less
    accurate than WY, the kg cache within bf16 rounding of WY; the pool is never
    written."""
    B = 5
    args, kw = _inputs(B, T, H, HV, gate, seed=100 + T)
    pool_before = args[5].clone()
    ref = _WY_DENSE(*args, **kw, emit_corrections=True, backend="recurrent")
    wy = _WY_DENSE(*args, **kw, emit_corrections=True, backend="wy")
    out = _WY_DENSE(*args, **kw, emit_corrections=True, backend="umma")
    torch.cuda.synchronize()
    assert torch.equal(args[5], pool_before)
    _check_vs_reference(out[:2], wy[:2], ref[:2])
    torch.testing.assert_close(out[2].float(), wy[2].float(), atol=2e-3, rtol=1e-2)


def _check_vs_reference(got, wy, ref):
    """Scale-invariant accuracy check of bf16 outputs against an fp32 reference:
    element-wise within 1 bf16 ulp of the tensor's max magnitude (+1 % relative),
    and the worst error no larger than WY's (25 % slack) or that ulp."""
    for g_, w_, r_ in zip(got, wy, ref, strict=True):
        g_, w_, r_ = g_.float(), w_.float(), r_.float()
        ulp = 2.0**-7 * r_.abs().max().item()  # bf16 ulp at the tensor's max magnitude
        torch.testing.assert_close(g_, r_, atol=ulp, rtol=1e-2)
        err_g = (g_ - r_).abs().max().item()
        err_w = (w_ - r_).abs().max().item()
        assert err_g <= max(ulp, 1.25 * err_w), (err_g, err_w, ulp)


@pytest.mark.parametrize("T", [1, 4, 8, 16])
def test_umma_bitwise_deterministic(T):
    """Repeated launches give bitwise identical outputs (B large enough for
    several waves)."""
    args, kw = _inputs(600, T, 12, 12, "lb", seed=7)
    first = _WY_DENSE(*args, **kw, emit_corrections=True, backend="umma")
    for _ in range(5):
        again = _WY_DENSE(*args, **kw, emit_corrections=True, backend="umma")
        for a, b in zip(first, again, strict=True):
            assert torch.equal(a, b)


@pytest.mark.parametrize(
    "spec_len,kg_pad",
    [(1, 0), (3, 0), (8, 0), (12, 0), (16, 0), (4, 1), (4, 2), (8, 4), (16, 6)],
)
def test_umma_dropin_matches_wy(spec_len, kg_pad):
    """Drop-in contract (ragged lengths incl. empty sequences, null slots, padded
    pool, sentinel caches) bitwise-equal on untouched regions and within bf16 /
    fp32 rounding elsewhere, vs the WY kernel. ``kg_pad`` pads the kg cache's
    position and block strides (strides not multiples of 8 elements take the
    kernel's 2-byte store path)."""
    torch.manual_seed(spec_len)
    dev = "cuda"
    B, H = 7, 4
    K = V = 128
    lens = torch.randint(0, spec_len + 1, (B,))
    lens[1] = spec_len
    qsl = torch.zeros(B + 1, dtype=torch.int32)
    qsl[1:] = lens.cumsum(0)
    tot = int(qsl[-1])
    qsl = qsl.to(dev)
    q = torch.randn(1, tot, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn_like(q)
    v = torch.randn(1, tot, H, V, dtype=torch.bfloat16, device=dev)
    g = torch.randn(1, tot, H, K, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(1, tot, H, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(H, dtype=torch.float32, device=dev) * 0.3
    dt = torch.randn(H * K, dtype=torch.float32, device=dev) * 0.1
    nblk = B + 2
    storage = torch.randn(nblk, H * V * K + 64, dtype=torch.bfloat16, device=dev) * 0.1
    state = storage[:, : H * V * K].view(nblk, H, V, K)
    sidx = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
    sidx[0] = 0
    sidx[3] = -2
    res = {}
    for be in ("wy", "umma"):
        corr = torch.full((nblk, H, spec_len, V), 7.0, device=dev)
        row = 2 * K + kg_pad
        kg_buf = torch.full(
            (nblk, H * spec_len * row + 3 * kg_pad),
            7.0,
            dtype=torch.bfloat16,
            device=dev,
        )
        kg = kg_buf[:, : H * spec_len * row].view(nblk, H, spec_len, row)[..., : 2 * K]
        out = torch.full((1, tot, H, V), 5.0, dtype=torch.bfloat16, device=dev)
        _WY_DROPIN(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt,
            -5.0,
            state,
            corr,
            kg,
            qsl,
            sidx,
            spec_len,
            out=out,
            backend=be,
        )
        res[be] = (out, corr, kg, kg_buf)
    torch.cuda.synchronize()
    (o0, c0, k0, b0), (o1, c1, k1, b1) = res["wy"], res["umma"]
    assert torch.equal(b0, b1)  # padding of the kg buffer untouched by both
    assert torch.equal(c0 == 7.0, c1 == 7.0) and torch.equal(k0 == 7.0, k1 == 7.0)
    assert torch.equal(o0 == 5.0, o1 == 5.0)
    torch.testing.assert_close(o1.float(), o0.float(), atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(c1, c0, atol=3e-2, rtol=1e-2)
    assert torch.equal(k1, k0)


def test_umma_auto_needs_experimental_opt_in(monkeypatch):
    """``backend="auto"`` picks the experimental kernel only with
    FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1, at T <= 8, and never where the
    dense dispatch would run the grouped recurrence (small batch)."""
    args, kw = _inputs(64, 8, 4, 4, "lb", seed=3)
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    n0 = _launches()
    _WY_DENSE(*args, **kw, backend="auto")
    assert _launches() == n0
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    ref = _WY_DENSE(*args, **kw, backend="umma")
    n0 = _launches()
    out = _WY_DENSE(*args, **kw, backend="auto")
    assert _launches() == n0 + 1
    assert torch.equal(out, ref)
    # T = 16: the WY kernel stays the automatic choice
    args16, kw16 = _inputs(64, 16, 4, 4, "lb", seed=4)
    n0 = _launches()
    _WY_DENSE(*args16, **kw16, backend="auto")
    assert _launches() == n0
    # small batch at T = 1: the grouped recurrence stays the automatic choice
    args1, kw1 = _inputs(2, 1, 4, 4, "pre", seed=5)
    n0 = _launches()
    _WY_DENSE(*args1, **kw1, backend="auto")
    assert _launches() == n0


def test_umma_backend_rejects_bad_values():
    """The drop-in validates its backend argument."""
    with pytest.raises(ValueError, match="backend must be"):
        _WY_DROPIN(*([None] * 13), 4, backend="bogus")


@pytest.mark.parametrize("T", [1, 8, 16])
def test_umma_cuda_graph_replay(T):
    """A captured UMMA launch replays with new input values (in place) and matches
    eager launches on the same values."""
    args, kw = _inputs(33, T, 4, 4, "lb", seed=T)
    _WY_DENSE(*args, **kw, emit_corrections=True, backend="umma")  # compile outside
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s):
        outs = _WY_DENSE(*args, **kw, emit_corrections=True, backend="umma")
    torch.cuda.current_stream().wait_stream(s)
    for seed in range(3):
        fresh, _ = _inputs(33, T, 4, 4, "lb", seed=50 + seed)
        for dst, src in zip(args, fresh, strict=True):
            dst.copy_(src)
        graph.replay()
        ref = _WY_DENSE(*args, **kw, emit_corrections=True, backend="umma")
        torch.cuda.synchronize()
        for a, b in zip(outs, ref, strict=True):
            assert torch.equal(a, b)


def _dropin_case(B, spec_len, H, seed, g_pad=0):
    """Ragged drop-in inputs (empty sequences, null / zero slots); ``g_pad`` pads the
    gate's token stride by that many elements."""
    gen = torch.Generator().manual_seed(seed)
    dev = "cuda"
    K = V = 128
    lens = torch.randint(0, spec_len + 1, (B,), generator=gen)
    lens[1] = spec_len
    qsl = torch.zeros(B + 1, dtype=torch.int32)
    qsl[1:] = lens.cumsum(0)
    tot = int(qsl[-1])
    rn = lambda *s_: torch.randn(*s_, generator=gen).to(dev)  # noqa: E731
    q = rn(1, tot, H, K).bfloat16()
    k = rn(1, tot, H, K).bfloat16()
    v = rn(1, tot, H, V).bfloat16()
    g_buf = rn(1, tot, H * K + g_pad).bfloat16()
    g = g_buf[..., : H * K].view(1, tot, H, K)
    beta = rn(1, tot, H).bfloat16()
    nblk = B + 2
    state = (rn(nblk, H, V, K) * 0.1).bfloat16()
    sidx = torch.randperm(nblk - 1, generator=gen)[:B].to(torch.int32) + 1
    sidx[0] = 0
    sidx[3] = -1
    rest = (rn(H) * 0.3, rn(H * K) * 0.1, -5.0)
    return (q, k, v, g, beta), rest, state, qsl.to(dev), sidx.to(dev), nblk


def _run_dropin(case, spec_len, H):
    (q, k, v, g, beta), (A_log, dt, lb), state, qsl, sidx, nblk = case
    corr = torch.full((nblk, H, spec_len, 128), 7.0, device="cuda")
    kg = torch.full((nblk, H, spec_len, 256), 7.0, dtype=torch.bfloat16, device="cuda")
    out = _WY_DROPIN(
        q, k, v, g, beta, A_log, dt, lb, state, corr, kg, qsl, sidx, spec_len,
        backend="umma",
    )  # fmt: skip
    return out, corr, kg


@pytest.mark.parametrize(
    "mode,T",
    [
        ("dense", 1),
        ("dense", 2),
        ("dense", 12),
        ("dense", 16),
        ("dropin", 3),
        ("dropin", 9),
        ("dropin", 16),
    ],
)
def test_umma_persistent_matches_one_item_per_cta(monkeypatch, mode, T):
    """The persistent grid (several work items per CTA: phase flips, buffer reuse,
    the next item's state TMA behind the readout), with and without the next-item
    input prefetch, is bitwise equal to one item per CTA, with more items than
    resident CTAs."""
    res = {}
    for grid in ((True, True), (True, False), (False, None)):
        monkeypatch.setattr(_launch, "_PERSISTENT", grid[0])
        monkeypatch.setattr(_launch, "_PREFETCH_NEXT", grid[1])
        if mode == "dense":
            args, kw = _inputs(250, T, 4, 8, "sp", seed=T)
            res[grid] = _WY_DENSE(*args, **kw, emit_corrections=True, backend="umma")
        else:
            res[grid] = _run_dropin(_dropin_case(420, T, 4, seed=T), T, 4)
    torch.cuda.synchronize()
    for grid in ((True, True), (True, False)):
        for a, b in zip(res[grid], res[(False, None)], strict=True):
            assert torch.equal(a, b)


def test_umma_persistent_policy_off_when_min_blocks_forced(monkeypatch):
    """The persistent grid is sized from the build's occupancy bound; where the launch
    forces min_blocks_per_mp=1 (SM107) that bound says nothing about residency, so the
    automatic policy keeps one item per CTA there at every T."""
    monkeypatch.setattr(_launch, "_PERSISTENT", None)
    for T in range(1, 17):
        for dropin in (False, True):
            assert _launch._use_persistent(T, None, dropin) == (
                T > 8 or (dropin and T <= 4)
            )
            assert not _launch._use_persistent(T, 1, dropin)


@pytest.mark.parametrize("T", [1, 5, 16])
def test_umma_dropin_unaligned_gate_stride(T):
    """A gate token stride that is not a multiple of 8 elements takes the 2-byte gate
    load build; bitwise equal to the 16-byte build on the same values."""
    padded = _dropin_case(40, T, 4, seed=3 + T, g_pad=4)
    aligned = list(padded)
    aligned[0] = (*padded[0][:3], padded[0][3].contiguous(), padded[0][4])
    a = _run_dropin(padded, T, 4)
    b = _run_dropin(tuple(aligned), T, 4)
    torch.cuda.synchronize()
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


def test_umma_dense_negative_slot_reads_zero_state():
    """Dense mode: a negative ``initial_state_indices`` entry reads a zero state on
    the UMMA backend, as on the WY kernel (so automatic selection cannot change the
    output of those rows)."""
    B, T = 6, 4
    args, kw = _inputs(B, T, 4, 4, "lb", seed=21)
    q, k, v, g, beta, pool, idx = args
    idx[1] = -1
    idx[4] = -3
    zero_slot = pool.shape[0]
    pool_z = torch.cat([pool, torch.zeros_like(pool[:1])])
    idx_z = torch.where(idx < 0, torch.full_like(idx, zero_slot), idx)
    out = _WY_DENSE(q, k, v, g, beta, pool, idx, **kw, backend="umma")
    ref = _WY_DENSE(q, k, v, g, beta, pool_z, idx_z, **kw, backend="umma")
    wy = _WY_DENSE(q, k, v, g, beta, pool, idx, **kw, backend="wy")
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    torch.testing.assert_close(out.float(), wy.float(), atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("be", ["wy", "umma"])
@pytest.mark.parametrize("which", ["q", "k", "v"])
def test_dropin_rejects_misaligned_token_stride(be, which):
    """q / k / v token strides that are not multiples of 8 elements raise (both
    kernels read token rows with 16-byte vector or bulk copies)."""
    dev = "cuda"
    H, K, T, tot = 4, 128, 4, 6
    ten = {}
    for name in ("q", "k", "v"):
        pad = 2 if name == which else 0
        ten[name] = torch.randn(1, tot, H * K + pad, dtype=torch.bfloat16, device=dev)[
            ..., : H * K
        ].view(1, tot, H, K)
    g = torch.randn(1, tot, H, K, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(1, tot, H, dtype=torch.bfloat16, device=dev)
    state = torch.zeros(4, H, K, K, dtype=torch.bfloat16, device=dev)
    corr = torch.zeros(4, H, T, K, device=dev)
    kg = torch.zeros(4, H, T, 2 * K, dtype=torch.bfloat16, device=dev)
    qsl = torch.tensor([0, 4, 6], dtype=torch.int32, device=dev)
    sidx = torch.tensor([1, 2], dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="token strides must be 16B-aligned"):
        _WY_DROPIN(
            ten["q"],
            ten["k"],
            ten["v"],
            g,
            beta,
            torch.zeros(H, device=dev),
            torch.zeros(H * K, device=dev),
            -5.0,
            state,
            corr,
            kg,
            qsl,
            sidx,
            T,
            backend=be,
        )


@pytest.mark.parametrize("be", ["UMMA", "bogus", None])
def test_dense_rejects_unknown_backend(be):
    """``kda_wy_output_only`` validates its backend argument (a typo used to run the
    WY kernel silently)."""
    args, kw = _inputs(2, 4, 4, 4, "lb", seed=1)
    with pytest.raises(ValueError, match="backend must be"):
        _WY_DENSE(*args, **kw, backend=be)


def _dropin_inputs(tot, T=4, H=4, nblk=4):
    """Minimal packed drop-in inputs: two sequences, ``tot`` tokens in total."""
    dev = "cuda"
    K = 128
    q = torch.randn(1, tot, H, K, dtype=torch.bfloat16, device=dev)
    args = (
        q,
        torch.randn_like(q),
        torch.randn_like(q),
        torch.randn_like(q),
        torch.randn(1, tot, H, dtype=torch.bfloat16, device=dev),
        torch.zeros(H, device=dev),
        torch.zeros(H * K, device=dev),
        -5.0,
        torch.zeros(nblk, H, K, K, dtype=torch.bfloat16, device=dev),
        torch.zeros(nblk, H, T, K, device=dev),
        torch.zeros(nblk, H, T, 2 * K, dtype=torch.bfloat16, device=dev),
        torch.tensor([0, min(tot, T), tot], dtype=torch.int32, device=dev),
        torch.tensor([1, 2], dtype=torch.int32, device=dev),
        T,
    )
    return args


@pytest.mark.parametrize("be", ["wy", "umma"])
def test_dropin_rejects_negative_null_min(be):
    """``null_min < 0`` would let negative slots index memory before the cache
    buffers (the UMMA kernel wrote there silently, the WY kernel faulted): every
    backend rejects it."""
    with pytest.raises(ValueError, match="null_min must be >= 0"):
        _WY_DROPIN(*_dropin_inputs(6), backend=be, null_min=-1)


def test_dropin_explicit_umma_raises_on_empty_batch_off_sm10x(monkeypatch):
    """An explicit ``backend="umma"`` raises on an unsupported device even when the
    packed batch is empty (the support check runs before the early return)."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    with pytest.raises(ValueError, match="backend='umma' unavailable"):
        _WY_DROPIN(*_dropin_inputs(0), backend="umma")


def test_umma_experimental_warning_names_the_caller(monkeypatch):
    """The ExperimentalWarning points at the user's call site (this file) from every
    entry: dense, drop-in, and recurrent_kda(disable_state_update=True) through the
    automatic route."""
    import warnings

    import flashinfer.api_logging as _api_logging

    def check(call):
        monkeypatch.setattr(_api_logging, "_WARNED_EXPERIMENTAL_BACKENDS", set())
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            call()
        hits = [
            w for w in rec if issubclass(w.category, _api_logging.ExperimentalWarning)
        ]
        assert hits, "no ExperimentalWarning"
        assert all(w.filename == __file__ for w in hits), [
            (w.filename, w.lineno) for w in hits
        ]

    args, kw = _inputs(64, 4, 4, 4, "lb", seed=11)
    check(lambda: _WY_DENSE(*args, **kw, backend="umma"))
    check(lambda: _WY_DROPIN(*_dropin_inputs(6), backend="umma"))
    # recurrent_kda -> kda_decode -> kda_wy_output_only(backend="auto") -> UMMA
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    monkeypatch.setattr(_kda_decode, "_run_kda_output_only", _WY_DENSE)
    q, k, v, g, beta, pool, idx = args
    check(
        lambda: _kda_decode.recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=idx,
            disable_state_update=True,
            use_qk_l2norm_in_kernel=True,
            **kw,
        )
    )
