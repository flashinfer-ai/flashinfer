"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---------------------------------------------------------------------------
Correctness tests for the Rubin (sm_107a) build of the UMMA MTP ReplaySSM kernel
(flashinfer/gdn_kernels/gdn_replay_mtp_umma_rubin.py).

1. Every test of tests/gdn/test_mtp_umma.py, re-run against
   ``gated_delta_rule_mtp_ucache_flush_umma_rubin`` (same fp32 oracle, cases and
   tolerances; the base file's kernel entry is swapped by an autouse fixture).
2. What only the Rubin build has: bit-identity with the base kernel, the fp32 gate-parameter
   cache across streams and in-place updates, and the persistent kernel (sm_107a):
   bit-identity with the non-persistent launch at T = 1-8 on any row mix and grid, the automatic
   choice, the dispatch rule and ``flush_hint``, a caller-owned claim counter, and concurrent
   replays of separately captured CUDA graphs; uGPU buffers (``ugpu_ucache_buffers``: one
   launch per die) bit-identical to the 4-D buffers, eager and captured; the routing of
   ``gated_delta_rule_mtp_ucache_flush`` to this build.

Run:  pytest tests/gdn/test_mtp_umma_rubin.py -q
"""

from __future__ import annotations

import importlib

import pytest
import torch

from . import test_mtp_umma as _base
from .test_mtp_umma import *  # noqa: F401,F403  (re-collects every base test here)
from .test_mtp_umma import DEV, RING, SCALE, W, _make_case

_RUBIN = (
    "flashinfer.gdn_kernels.gdn_replay_mtp_umma_rubin",
    "gated_delta_rule_mtp_ucache_flush_umma_rubin",
)
_RUBIN_CC = ((10, 0), (10, 7))  # archs the Rubin build runs on


def _rubin_mod():
    return importlib.import_module(_RUBIN[0])


def _umma():
    return getattr(_rubin_mod(), _RUBIN[1])


def _skip_if_no_rubin_build():
    from flashinfer.utils import get_compute_capability

    _base._skip_if_not_sm100()
    if get_compute_capability(torch.device("cuda")) not in _RUBIN_CC:
        pytest.skip("the Rubin build runs on sm_107a and sm_100a only")


@pytest.fixture(autouse=True)
def _rubin_entry(monkeypatch):
    """Point the base file's tests (collected here by the star import) at the Rubin build."""
    if torch.cuda.is_available():
        from flashinfer.utils import get_compute_capability

        cc = get_compute_capability(torch.device("cuda"))
        if cc[0] == 10 and cc not in _RUBIN_CC:
            pytest.skip("the Rubin build runs on sm_107a and sm_100a only")
    monkeypatch.setattr(_base, "_ENTRY", _RUBIN)


@pytest.mark.parametrize("T", [1, 3, 4, 5, 8])
@pytest.mark.parametrize("hpc", [1, 2, 4])
def test_bit_identical_to_base_kernel(T, hpc):
    """Round to nearest: outputs, state pool and rings bit-identical to the base kernel on a mixed
    verify / flush / padded batch (wrapped windows)."""
    _skip_if_no_rubin_build()
    from flashinfer.gdn_kernels.gdn_replay_mtp_umma import (
        gated_delta_rule_mtp_ucache_flush_umma as base_fn,
    )

    case, fm = _pz_case(T, 500 + 11 * T + hpc, "FVFFPVVFPFFFVVPF")
    ref = _pz_call(case, fm, fn=base_fn, heads_per_cta=hpc)
    got = _pz_call(case, fm, heads_per_cta=hpc)
    for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
        assert torch.equal(x, y), name


def test_persistent_rejected_off_sm107():
    """persistent=True is sm_107a only: elsewhere a clear ValueError, not a silent fallback."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        pytest.skip("sm_107a runs the persistent kernel")
    case, fm = _pz_case(8, 71, "FVVPF")
    with pytest.raises(ValueError, match="sm_107a"):
        _pz_call(case, fm, persistent=True)


# ---------------------------------------------------------------------------
# persistent kernel (opt-in; sm_107a): CTAs loop over (key head, row) items claimed from a
# counter. It must be bit-identical to the non-persistent launch on any mix and order of
# verify (V), flush (F) and padded (P) rows, at any grid size.
_PZ_PATTERNS = (
    "FVFFPVVFPFFFVVPFVF",  # every transition: V->F, F->V, F->F, V->V, a padded row between
    "FFFFFFFFFFFF",  # every row flushes
    "VVVVVVVVVVVF",  # one flush row, the last item
    "FPVVVVPVVVVV",  # one flush row, the first item
)


def _skip_if_no_persistent():
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    if get_compute_capability(torch.device("cuda")) != (10, 7):
        pytest.skip("the persistent kernel is enabled on sm_107a only")


def _pz_case(T, seed, pattern):
    import random

    fm = W - T + 1
    rng = random.Random(seed)
    hist = [rng.randint(fm, W) if c == "F" else rng.randint(0, fm - 1) for c in pattern]
    bases = [rng.randrange(RING) for _ in pattern]
    case = list(_make_case(len(pattern), T, hist, seed, bases))
    case[13] = torch.tensor(
        [-1 if c == "P" else r for r, c in enumerate(pattern)],
        dtype=torch.int32,
        device=DEV,
    )
    return case, fm


def _pz_bits(t):
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _pz_call(case, fm, fn=None, **kw):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)
    kw.setdefault("heads_per_cta", 4)
    (fn or _umma())(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=pool,
        initial_state_indices=idx,
        k_cache=kc,
        u_cache=uc,
        g_cache=gc,
        hist_len=hl,
        cache_base=cb,
        scale=SCALE,
        flush_min=fm,
        restart_hist_on_flush=False,
        output=out,
        **kw,
    )
    torch.cuda.synchronize()
    return [_pz_bits(x) for x in (out, pool, kc, uc, gc)]


def test_bf16_gate_params_two_streams():
    """bf16 A_log / dt_bias are converted to fp32 once and cached: a launch on another stream (e.g. the
    other die's stream of a die-localized split) must not read the copy before the converting stream
    has written it. Stream 0 is held back, so an unsynchronized read on stream 1 sees no copy yet."""
    _skip_if_no_rubin_build()
    T, B = 4, 8
    fm = W - T + 1
    hist = [fm, 3, 0, fm, 7, fm, 1, 9]
    case = _make_case(B, T, hist, seed=55, bases=[3, 30, 0, 17, 9, 28, 5, 12])
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    fn = _umma()
    ss = [torch.cuda.Stream(), torch.cuda.Stream()]

    def run(A_, dt_, delay):
        bufs = [x.clone() for x in (pool, kc, uc, gc)]
        ys = []
        torch.cuda.synchronize()
        for d, s in enumerate(ss):
            r = slice(4 * d, 4 * d + 4)
            with torch.cuda.stream(s):
                if d == 0 and delay:
                    torch.cuda._sleep(20_000_000)
                y = fn(
                    A_,
                    a[r],
                    dt_,
                    q=q[r],
                    k=k[r],
                    v=v[r],
                    b=b[r],
                    initial_state_source=bufs[0],
                    initial_state_indices=idx[r],
                    k_cache=bufs[1],
                    u_cache=bufs[2],
                    g_cache=bufs[3],
                    hist_len=hl[r],
                    cache_base=cb[r],
                    scale=SCALE,
                    flush_min=fm,
                    restart_hist_on_flush=False,
                )
                ys.append(y)
        torch.cuda.synchronize()
        return [_pz_bits(x) for x in [torch.cat(ys)] + bufs]

    ref = run(A_log.float(), dt_bias.float(), False)
    # fresh bf16 objects: their first conversion is enqueued on the held-back stream 0
    A_, dt_ = A_log.clone(), dt_bias.clone()
    got = run(A_, dt_, True)
    for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
        assert torch.equal(x, y), name
    # warm cache, both streams already synchronized with the conversion
    got = run(A_, dt_, True)
    for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
        assert torch.equal(x, y), ("warm", name)


def test_bf16_gate_params_inplace_update():
    """The fp32 copy of a bf16 A_log / dt_bias is re-made after an in-place update of the same
    object (e.g. a weight reload), not served stale."""
    _skip_if_no_rubin_build()
    T, B = 4, 8
    fm = W - T + 1
    hist = [fm, 3, 0, fm, 7, fm, 1, 9]
    case = _make_case(B, T, hist, seed=56, bases=[3, 30, 0, 17, 9, 28, 5, 12])
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    fn = _umma()

    def run(A_, dt_):
        bufs = [x.clone() for x in (pool, kc, uc, gc)]
        y = fn(
            A_,
            a,
            dt_,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bufs[0],
            initial_state_indices=idx,
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
        )
        torch.cuda.synchronize()
        return [_pz_bits(x) for x in [y] + bufs]

    A_, dt_ = A_log.clone(), dt_bias.clone()
    run(A_, dt_)  # caches A_ / dt_
    A_new = (A_log.float() + 0.25).to(A_log.dtype)
    dt_new = (dt_bias.float() - 0.25).to(dt_bias.dtype)
    ref = run(A_new.float(), dt_new.float())
    A_.copy_(A_new)
    dt_.copy_(dt_new)
    got = run(A_, dt_)
    for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
        assert torch.equal(x, y), name
    # inference-mode tensors (no version counter) are cached by identity
    with torch.inference_mode():
        A_i, dt_i = A_new.clone(), dt_new.clone()
    got = run(A_i, dt_i)
    got2 = run(A_i, dt_i)
    for name, x, y, z in zip(("y", "pool", "k", "u", "g"), ref, got, got2, strict=True):
        assert torch.equal(x, y) and torch.equal(x, z), ("inference", name)


@pytest.mark.parametrize("arena", [True, False])
def test_persistent_graphs_concurrent(arena, monkeypatch):
    """Two persistent launches captured in two separate CUDA graphs (torch.cuda.graph defaults: one
    capture stream) and replayed concurrently on two streams must not share a claim counter (the
    captured counters come from the eagerly zeroed arena, or without it from a memset node)."""
    _skip_if_no_persistent()
    M = _rubin_mod()

    if not arena:
        monkeypatch.setattr(M, "_PZ_ARENA_SLOTS", 0)
        monkeypatch.setattr(M, "_PZ_ARENA", {})

    case, fm = _pz_case(8, 91, "FVVVPVFVVFVVVVVVFVPVVFVV")
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    n = q.shape[0] // 2
    parts = [slice(0, n), slice(n, 2 * n)]
    bufs = [x.clone() for x in (pool, kc, uc, gc)]
    outs = [
        torch.full(v[r].shape, -7.0, dtype=torch.bfloat16, device=DEV) for r in parts
    ]
    fn = _umma()

    def call(d):
        r = parts[d]
        fn(
            A_log,
            a[r],
            dt_bias,
            q=q[r],
            k=k[r],
            v=v[r],
            b=b[r],
            initial_state_source=bufs[0],
            initial_state_indices=idx[r],
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl[r],
            cache_base=cb[r],
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            heads_per_cta=4,
            output=outs[d],
            persistent=True,
            num_sms=8,
        )

    ref = _pz_call(case, fm)
    call(0)  # eager warm-up (compile, fp32 parameter cache)
    call(1)
    torch.cuda.synchronize()
    graphs = []
    for d in range(2):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call(d)
        graphs.append(g)
    ss = [torch.cuda.Stream(), torch.cuda.Stream()]
    for _ in range(6):
        for x, x0 in zip(bufs, (pool, kc, uc, gc), strict=True):
            x.copy_(x0)
        torch.cuda.synchronize()
        for g, s in zip(graphs, ss, strict=True):
            with torch.cuda.stream(s):
                g.replay()
        torch.cuda.synchronize()
        got = [_pz_bits(x) for x in [torch.cat(outs)] + bufs]
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), name
    assert all(c.tolist() == [0, 0] for c in M._PZ_CTR.values())
    assert all(
        int(c.abs().sum()) == 0 for a in M._PZ_ARENA.values() for c in a["chunks"]
    )
    assert bool(M._PZ_ARENA) == arena


@pytest.mark.parametrize("T", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("mode", ["none", "philox5", "lcg"])
def test_persistent_bit_identical(T, mode):
    _skip_if_no_persistent()
    sr = {}
    if mode != "none":
        sr = dict(
            stochastic_rounding=mode.rstrip("5"),
            rand_seed=torch.tensor([1234], dtype=torch.int64, device=DEV),
        )
        if mode == "philox5":
            sr["philox_rounds"] = 5
    for i, pattern in enumerate(_PZ_PATTERNS):
        case, fm = _pz_case(T, 300 + 7 * T + i, pattern)
        ref = _pz_call(case, fm, **sr)
        for grid in (1, 3, None):
            got = _pz_call(case, fm, persistent=True, grid_override=grid, **sr)
            for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
                assert torch.equal(x, y), (pattern, grid, name)


def test_persistent_dispatch_rule(monkeypatch):
    """persistent=True runs the non-persistent kernel where the measured rule says so (Philox,
    T = 8, few items per CTA slot), the persistent one otherwise or with grid_override; same bits."""
    _skip_if_no_persistent()
    M = _rubin_mod()

    seen = []
    real = M._get_kernel

    def spy(*args, **kw):
        seen.append(args[9] if len(args) > 9 else kw.get("persistent", False))
        return real(*args, **kw)

    monkeypatch.setattr(M, "_get_kernel", spy)
    case, fm = _pz_case(8, 79, "FVVPFVVFVVVF")
    nkh = case[0].shape[2]
    sr = dict(
        stochastic_rounding="philox",
        rand_seed=torch.tensor([99], dtype=torch.int64, device=DEV),
        philox_rounds=5,
    )
    ref = _pz_call(case, fm, **sr)
    B = case[0].shape[0]
    lim = M._PZ_PLAIN_MAX_ITEMS["some"][("philox", 8)] * M._CTA_SLOTS_PER_SM
    few = -(-nkh * B // lim)  # the fewest SMs at which the rule still applies
    for kw, want in (
        (dict(num_sms=few), False),
        (dict(num_sms=few - 1), True)
        if few > 1
        else (dict(num_sms=few, grid_override=3), True),
        (dict(num_sms=few, grid_override=3), True),
    ):
        seen.clear()
        got = _pz_call(case, fm, persistent=True, **kw, **sr)
        assert seen == [want], (kw, seen)
        assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True)), kw
    seen.clear()
    _pz_call(case, fm, persistent=True, num_sms=few)  # round to nearest: no rule
    assert seen == [True]


def test_persistent_flush_hint(monkeypatch):
    """flush_hint: None / "some" use the dispatch rule's flush threshold (non-persistent kernel at
    <= 10 items per CTA slot), "none" / "flush_first" the lower one (<= 5): between the two only
    they run the persistent kernel; same bits either way, also from a captured CUDA graph (the
    hint is read at capture); anything else is a ValueError."""
    _skip_if_no_persistent()
    M = _rubin_mod()

    seen = []
    real = M._get_kernel

    def spy(*args, **kw):
        seen.append(args[9] if len(args) > 9 else kw.get("persistent", False))
        return real(*args, **kw)

    monkeypatch.setattr(M, "_get_kernel", spy)
    case, fm = _pz_case(8, 83, "FFFVVPVVVFVV")
    sr = dict(
        stochastic_rounding="philox",
        rand_seed=torch.tensor([7], dtype=torch.int64, device=DEV),
        philox_rounds=5,
    )
    ref = _pz_call(case, fm, **sr)
    n_items = case[0].shape[2] * case[0].shape[0]
    slots = M._CTA_SLOTS_PER_SM
    hi, lo = (M._PZ_PLAIN_MAX_ITEMS[k][("philox", 8)] for k in ("some", "few"))
    few = -(-n_items // (hi * slots))  # the fewest SMs at which the "some" rule applies
    more = -(-n_items // (lo * slots))  # the fewest at which the "few" rule applies too
    assert few < more
    assert M._pz_runs_plain(8, "philox", n_items, few)
    assert not M._pz_runs_plain(8, "philox", n_items, few, "none")
    assert not M._pz_runs_plain(8, "philox", n_items, more - 1, "flush_first")
    assert M._pz_runs_plain(8, "philox", n_items, more, "flush_first")
    for nsm, hint, want in (
        (few, None, False),
        (few, "some", False),
        (few, "none", True),
        (few, "flush_first", True),
        (more, "none", False),
        (more, "flush_first", False),
    ):
        seen.clear()
        got = _pz_call(case, fm, persistent=True, num_sms=nsm, flush_hint=hint, **sr)
        assert seen == [want], (nsm, hint, seen)
        assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True)), hint
    # outside the rule (round to nearest) every hint runs the persistent kernel
    for hint in (None, "some", "none", "flush_first"):
        seen.clear()
        _pz_call(case, fm, persistent=True, num_sms=few, flush_hint=hint)
        assert seen == [True], hint
    # under graph capture: the kernel chosen at capture is replayed
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    bufs = [x.clone() for x in (pool, kc, uc, gc)]
    out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)

    def call():
        _umma()(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bufs[0],
            initial_state_indices=idx,
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            heads_per_cta=4,
            output=out,
            persistent=True,
            num_sms=few,
            flush_hint="none",
            **sr,
        )

    call()
    torch.cuda.synchronize()
    seen.clear()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    assert seen == [True]
    for _ in range(3):
        for x, x0 in zip(bufs, (pool, kc, uc, gc), strict=True):
            x.copy_(x0)
        g.replay()
        torch.cuda.synchronize()
        got = [_pz_bits(x) for x in [out] + bufs]
        assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))
    for bad in ("all", "None", 0):
        with pytest.raises(ValueError):
            _pz_call(case, fm, persistent=True, num_sms=few, flush_hint=bad)
    with pytest.raises(ValueError):
        _pz_call(case, fm, flush_hint="mixed")  # validated without persistent too


def test_persistent_auto(monkeypatch):
    """persistent=None (the default): the persistent kernel at T in _PZ_AUTO_T (2-8) with all value
    heads per CTA when num_sms names fewer SMs than the device (a die-localized launch) and there are
    enough items per CTA slot; the non-persistent kernel otherwise. Same bits either way."""
    _skip_if_no_persistent()
    M = _rubin_mod()
    seen = []
    real = M._get_kernel

    def spy(*args, **kw):
        seen.append(args[9] if len(args) > 9 else kw.get("persistent", False))
        return real(*args, **kw)

    monkeypatch.setattr(M, "_get_kernel", spy)
    monkeypatch.setattr(M, "_PZ_AUTO_MIN_ITEMS", {"none": 0, "some": 0})
    nsm = torch.cuda.get_device_properties(DEV).multi_processor_count
    for T, pat in ((1, "FVVPF"), (2, "VFVVP"), (4, "FVVPFV"), (8, "VFVVF")):
        case, fm = _pz_case(T, 400 + T, pat)
        ref = _pz_call(case, fm, persistent=False)
        auto = T in M._PZ_AUTO_T
        for kw, want in (
            (dict(), False),  # no num_sms: a whole-device launch
            (dict(num_sms=nsm), False),
            (dict(num_sms=nsm // 2), auto),
            (dict(num_sms=nsm // 2, heads_per_cta=2), False),
            (dict(num_sms=nsm // 2, grid_override=3), False),
        ):
            seen.clear()
            got = _pz_call(case, fm, **kw)
            assert seen == [want], (T, kw, seen)
            assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True)), (
                T,
                kw,
            )
    # too few items per CTA slot: non-persistent, unless the launch has no flush rows
    monkeypatch.setattr(M, "_PZ_AUTO_MIN_ITEMS", {"none": 0, "some": 10**6})
    case, fm = _pz_case(4, 404, "VVVPVV")
    for hint, want in (
        (None, False),
        ("some", False),
        ("flush_first", False),
        ("none", True),
    ):
        seen.clear()
        _pz_call(case, fm, num_sms=nsm // 2, flush_hint=hint)
        assert seen == [want], (hint, seen)


def test_persistent_api():
    """Scope checks, and a caller-owned claim counter (zeroed again by the launch's last CTA)."""
    _skip_if_no_persistent()
    case, fm = _pz_case(8, 77, "FVVPF")
    ref = _pz_call(case, fm)
    ctr = torch.zeros(2, dtype=torch.int32, device=DEV)
    got = _pz_call(case, fm, persistent=True, grid_override=2, persistent_counter=ctr)
    assert all(torch.equal(x, y) for x, y in zip(ref, got, strict=True))
    assert ctr.tolist() == [0, 0]
    with pytest.raises(ValueError):
        _pz_call(case, fm, persistent=True, persistent_counter=ctr.to(torch.int64))
    with pytest.raises(
        ValueError
    ):  # the persistent kernel runs all 4 value heads per CTA
        _pz_call(case, fm, persistent=True, heads_per_cta=2)


# ---------------------------------------------------------------------------
# uGPU buffers ([pool, dies, heads / dies, ...], see ugpu_ucache_buffers): one launch per die's
# key heads, bitwise the same as the 4-D buffers in every rounding mode. Without two locality
# domains (older drivers), and for small eager calls, the per-die launches run back to back on
# the current stream; otherwise (Rubin, B200 on recent drivers) on the dies' green contexts.
def _sr_kw(mode):
    if mode == "none":
        return {}
    kw = dict(
        stochastic_rounding=mode.rstrip("5"),
        rand_seed=torch.tensor([1234], dtype=torch.int64, device=DEV),
    )
    if mode == "philox5":
        kw["philox_rounds"] = 5
    return kw


def _ugpu_split(x, n, layout):
    """[pool, heads, ...] -> [pool, n, heads / n, ...]: a view of x ("view"), or a copy whose
    parts are separate dense blocks, as ugpu_ucache_buffers allocates them ("parts")."""
    v5 = x.view(x.shape[0], n, x.shape[1] // n, *x.shape[2:])
    if layout == "view":
        return v5
    parts = torch.empty(
        (n, x.shape[0], x.shape[1] // n, *x.shape[2:]), dtype=x.dtype, device=x.device
    ).transpose(0, 1)
    parts.copy_(v5)
    return parts


def _ugpu_logical(x):
    return x.reshape(x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:])


def _ugpu_call(case, fm, bufs=None, layout="parts", **kw):
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    if bufs is None:
        bufs = [_ugpu_split(x, 2, layout) for x in (pool, kc, uc, gc)]
    else:
        for dst, src in zip(bufs, (pool, kc, uc, gc), strict=True):
            dst.copy_(_ugpu_split(src, dst.shape[1], "view"))
    out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)
    kw.setdefault("heads_per_cta", 4)
    _umma()(
        A_log,
        a,
        dt_bias,
        q=q,
        k=k,
        v=v,
        b=b,
        initial_state_source=bufs[0],
        initial_state_indices=idx,
        k_cache=bufs[1],
        u_cache=bufs[2],
        g_cache=bufs[3],
        hist_len=hl,
        cache_base=cb,
        scale=SCALE,
        flush_min=fm,
        restart_hist_on_flush=False,
        output=out,
        **kw,
    )
    torch.cuda.synchronize()
    return [_pz_bits(x) for x in [out] + [_ugpu_logical(y) for y in bufs]]


def _ugpu_pattern(B, seed):
    import random

    rng = random.Random(seed)
    return "".join(rng.choice("VVVVFFP") for _ in range(B))


@pytest.mark.parametrize("T", [1, 4, 8])
@pytest.mark.parametrize("mode", ["none", "philox5", "lcg"])
@pytest.mark.parametrize("layout", ["view", "parts"])
def test_ugpu_buffers_bit_identical(T, mode, layout):
    """5-D uGPU buffers: same bits as the 4-D buffers (stochastic rounding included: its ids use
    the global value head), for both kernels and every heads-per-CTA."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    sr = _sr_kw(mode)
    case, fm = _pz_case(T, 700 + 7 * T, _PZ_PATTERNS[0])
    ref = _pz_call(case, fm, **sr)
    runs = [dict(heads_per_cta=h) for h in (1, 2, 4)]
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        runs += [dict(persistent=True), dict(persistent=True, grid_override=3)]
    for kw in runs:
        got = _ugpu_call(case, fm, layout=layout, **kw, **sr)
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), (kw, name)


def test_ugpu_buffers_graph():
    """A captured uGPU call (on Rubin: the per-die green-context launches) replays to the same bits
    as the 4-D buffers, with a per-die caller-owned claim counter where persistent."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    T, B = 4, 48
    case, fm = _pz_case(T, 811, _ugpu_pattern(B, 811))
    ref = _pz_call(case, fm)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    bufs = [_ugpu_split(x.clone(), 2, "parts") for x in (pool, kc, uc, gc)]
    out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)
    kw = {}
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        kw = dict(
            persistent=True,
            persistent_counter=torch.zeros(2, 2, dtype=torch.int32, device=DEV),
        )

    def call():
        _umma()(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bufs[0],
            initial_state_indices=idx,
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            output=out,
            **kw,
        )

    call()  # eager warm-up: compile, green-context streams, fp32 parameter cache
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for _ in range(3):
        for x, x0 in zip(bufs, (pool, kc, uc, gc), strict=True):
            x.copy_(_ugpu_split(x0, 2, "view"))
        out.fill_(-7.0)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        got = [_pz_bits(x) for x in [out] + [_ugpu_logical(y) for y in bufs]]
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), name
    if kw:
        assert kw["persistent_counter"].tolist() == [[0, 0], [0, 0]]


def test_ugpu_ucache_buffers():
    """Buffers from ugpu_ucache_buffers (each die's part in its own memory) and an eager call large
    enough for the per-die green-context launches: same bits as the 4-D buffers. Needs a device
    with two locality domains (Rubin; B200 with a driver that exposes them)."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    M = _rubin_mod()
    if M._locality_domains(torch.cuda.current_device()) != 2:
        pytest.skip("needs a device with two locality domains")
    T, B = 8, 192  # B x HV = 12288: above the eager fork / join threshold
    case, fm = _pz_case(T, 905, _ugpu_pattern(B, 905))
    ref = _pz_call(case, fm)
    runs = [{}, dict(persistent=False)]
    if get_compute_capability(torch.device("cuda")) == (10, 7):
        runs.append(dict(persistent=True))
    for kw in runs:
        bufs = list(M.ugpu_ucache_buffers(B, 16, 64, device=DEV))
        assert [tuple(x.shape) for x in bufs] == [
            (B, 2, 32, 128, 128),
            (B, 2, 8, 32, 128),
            (B, 2, 32, 32, 128),
            (B, 2, 32, 32),
        ]
        got = _ugpu_call(case, fm, bufs=bufs, **kw)
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), (kw, name)


def test_ugpu_buffers_rejected():
    """Mismatched 5-D buffers: a ValueError naming the expected layout."""
    _skip_if_no_rubin_build()
    case, fm = _pz_case(4, 61, "FVVPF")
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    bufs = [_ugpu_split(x, 2, "view") for x in (pool, kc, uc, gc)]
    bad = [bufs[0], kc, bufs[2], bufs[3]]  # a 4-D k ring next to a 5-D pool
    with pytest.raises(ValueError, match="uGPU buffers"):
        _umma()(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bad[0],
            initial_state_indices=idx,
            k_cache=bad[1],
            u_cache=bad[2],
            g_cache=bad[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
        )


def test_dispatcher_routes_rubin_build(monkeypatch):
    """gated_delta_rule_mtp_ucache_flush (fp16-state arm, backend "auto"): 4-D buffers run the Rubin
    build on sm_107a and the base UMMA kernel elsewhere; uGPU buffers always run the Rubin build.
    Same bits as the Rubin build called directly."""
    from flashinfer.utils import get_compute_capability

    _skip_if_no_rubin_build()
    mod = _base._flashinfer_fp16_state()
    calls = []

    def spy(name, real):
        def wrapped(*args, **kw):
            calls.append(name)
            return real(*args, **kw)

        return wrapped

    for name in ("_UMMA", "_UMMA_RUBIN"):
        monkeypatch.setattr(mod, name, spy(name, getattr(mod, name)))

    def run(case, fm, ugpu):
        q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
            x.clone() for x in case
        ]
        bufs = [pool, kc, uc, gc]
        if ugpu:
            bufs = [_ugpu_split(x, 2, "parts") for x in bufs]
        out = torch.full(v.shape, -7.0, dtype=torch.bfloat16, device=DEV)  # padded rows
        y = mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=bufs[0],
            initial_state_indices=idx,
            k_cache=bufs[1],
            u_cache=bufs[2],
            g_cache=bufs[3],
            hist_len=hl,
            cache_base=cb,
            scale=SCALE,
            flush_min=fm,
            restart_hist_on_flush=False,
            output=out,
        )
        torch.cuda.synchronize()
        assert y is out
        if ugpu:
            bufs = [_ugpu_logical(x) for x in bufs]
        return [_pz_bits(x) for x in [y] + bufs]

    case, fm = _pz_case(4, 123, _PZ_PATTERNS[0])
    ref = _pz_call(case, fm, heads_per_cta=None)
    on_rubin = get_compute_capability(torch.device("cuda")) == (10, 7)
    for ugpu, want in (
        (False, "_UMMA_RUBIN" if on_rubin else "_UMMA"),
        (True, "_UMMA_RUBIN"),
    ):
        calls.clear()
        got = run(case, fm, ugpu)
        assert calls == [want], (ugpu, calls)
        for name, x, y in zip(("y", "pool", "k", "u", "g"), ref, got, strict=True):
            assert torch.equal(x, y), (ugpu, name)
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = case
    with pytest.raises(ValueError, match="uGPU buffers"):
        mod.gated_delta_rule_mtp_ucache_flush(
            A_log,
            a,
            dt_bias,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=_ugpu_split(pool.clone(), 2, "view"),
            k_cache=kc,
            u_cache=uc,
            g_cache=gc,
            hist_len=hl,
            cache_base=cb,
            restart_hist_on_flush=False,
            backend="hmma",
        )


def test_dispatcher_ugpu_cursor_checks():
    """gated_delta_rule_mtp_ucache_flush checks the ring cursors of a standalone call
    (restart_hist_on_flush=True) before handing uGPU buffers to the Rubin build, as for 4-D buffers:
    an out-of-range hist_len raises instead of reaching the kernel."""
    _skip_if_no_rubin_build()
    mod = _base._flashinfer_fp16_state()
    case, fm = _pz_case(4, 131, "FVVPF")
    q, k, v, a, b, A_log, dt_bias, pool, kc, uc, gc, hl, cb, idx = [
        x.clone() for x in case
    ]
    bufs = [_ugpu_split(x, 2, "parts") for x in (pool, kc, uc, gc)]
    hl[1] = W + 4  # beyond the 16-slot window
    for ugpu in (False, True):
        b4 = bufs if ugpu else [pool, kc, uc, gc]
        with pytest.raises(AssertionError, match="hist_len out of legal range"):
            mod.gated_delta_rule_mtp_ucache_flush(
                A_log,
                a,
                dt_bias,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=b4[0],
                initial_state_indices=idx,
                k_cache=b4[1],
                u_cache=b4[2],
                g_cache=b4[3],
                hist_len=hl,
                cache_base=cb,
                scale=SCALE,
                flush_min=fm,
                restart_hist_on_flush=True,
            )
