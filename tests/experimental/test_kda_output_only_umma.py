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
    """UMMA vs the WY kernel on identical inputs: outputs, corrections and the kg
    cache (emit mode) within bf16 rounding; the pool is never written."""
    B = 5
    args, kw = _inputs(B, T, H, HV, gate, seed=100 + T)
    pool_before = args[5].clone()
    ref = _WY_DENSE(*args, **kw, emit_corrections=True, backend="wy")
    out = _WY_DENSE(*args, **kw, emit_corrections=True, backend="umma")
    torch.cuda.synchronize()
    assert torch.equal(args[5], pool_before)
    # absolute tolerances scale with the output magnitude once it exceeds 1 (both
    # kernels round to bf16; at the default scale |out| < 1 and they are unchanged)
    for got, want, atol in zip(out, ref, (2e-3, 3e-2, 2e-3), strict=True):
        mag = max(1.0, want.float().abs().max().item())
        torch.testing.assert_close(
            got.float(), want.float(), atol=atol * mag, rtol=1e-2
        )


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
