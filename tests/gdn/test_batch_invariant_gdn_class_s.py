# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant GDN class ``gdn.bf16.s16`` behind ``chunk_gated_delta_rule(batch_invariant=True)``.

Every comparison between call shapes is ``torch.equal``: decode, multi-token steps, prefill and prefill in pieces give
the same rows and leave the same slot, whatever the batch, the slots and the placement. The inputs are integer-derived
on the host, so ``FLASHINFER_GDN_S16_HASH_DIR`` records hashes that compare across processes, devices and hosts.
"""

import hashlib
import json
import math
import os

import numpy as np
import pytest
import torch

import flashinfer
from flashinfer.batch_invariant import (
    CLASSES,
    BatchInvariantUnsupportedError,
    class_name,
    fingerprint,
)
from flashinfer.batch_invariant.gdn_s16 import pool as s16_pool
from flashinfer.batch_invariant.gdn_s16 import spec
from flashinfer.batch_invariant.gdn_s16.pool import get_pool
from flashinfer.utils import get_compute_capability

D = 128
CHUNK = spec.CHUNK
HEADS = [(16, 48), (16, 32), (4, 4)]
BF = torch.bfloat16
_HASHES = {}


def _supported():
    if not torch.cuda.is_available():
        return False
    return tuple(get_compute_capability(torch.device("cuda"))) in spec.GDN_ARCHS


needs_device = pytest.mark.skipif(
    not _supported(), reason="the class gdn.bf16.s16 needs SM100 / SM103 / SM107"
)


@pytest.fixture(scope="module", autouse=True)
def _hash_file():
    yield
    root = os.environ.get("FLASHINFER_GDN_S16_HASH_DIR")
    if not root or not _HASHES:
        return
    path = os.path.join(root, "gdn_s16_hashes.json")
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        diff = {k for k in _HASHES if k in saved and saved[k] != _HASHES[k]}
        assert not diff, (
            f"output hashes differ from the recorded process: {sorted(diff)}"
        )
        saved.update(_HASHES)
        with open(path, "w") as f:
            json.dump(saved, f, indent=1, sort_keys=True)
    else:
        os.makedirs(root, exist_ok=True)
        with open(path, "w") as f:
            json.dump(_HASHES, f, indent=1, sort_keys=True)


def _record(key, *tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    _HASHES[key] = h.hexdigest()


def _ints(n, seed):
    with np.errstate(over="ignore"):
        x = np.arange(n, dtype=np.uint64) + np.uint64(
            (seed * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        )
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _normal(shape, seed):
    """Approximately N(0, 1) without libm or a torch RNG: four 16-bit integers and one exact fp32 division."""
    x = _ints(int(np.prod(shape)), seed)
    s = sum(
        ((x >> np.uint64(16 * i)) & np.uint64(0xFFFF)).astype(np.int64)
        for i in range(4)
    )
    s = (s - 131070).astype(np.float32) / np.float32(37837.0)
    return torch.from_numpy(s.reshape(shape))


def _inputs(L, HQ, HV, seed, steep=False):
    """One sequence: packed rows ``(q, k, v, a, b)``, the gate parameters and a nonzero bf16 state."""

    def r(i, *shape):
        return _normal(shape, seed * 16 + i)

    a = r(3, L, HV) * 2.0 if steep else r(3, L, HV) - 6.0
    rows = tuple(
        t.to("cuda", BF).contiguous()
        for t in (r(0, L, HQ, D), r(1, L, HQ, D), r(2, L, HV, D), a, r(4, L, HV))
    )
    par = ((r(5, HV) * 0.5).cuda(), (r(6, HV) * 0.125).cuda())
    return rows, par, (r(7, HV, D, D) * 0.125).to("cuda", BF)


def _cut(x, a, b):
    return tuple(t[a:b].contiguous() for t in x)


def _cat(xs):
    return tuple(torch.cat(ts).contiguous() for ts in zip(*xs, strict=True))


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _offsets(lens):
    return [0] + [int(c) for c in np.cumsum(lens)]


def _bi(pool, slots, x, par, cu=None, prefix=None, **kwargs):
    q, k, v, a, b = x
    T = q.shape[0]
    if cu is None:
        cu = list(range(T + 1)) if len(slots) == T and T > 1 else [0, T]
    args = dict(
        initial_state=pool,
        state_indices=_i32(slots),
        cu_seqlens=_i32(cu),
        prefix_lens=None if prefix is None else _i32(prefix),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        A_log=par[0],
        dt_bias=par[1],
        batch_invariant=True,
    )
    args.update(kwargs)
    return flashinfer.chunk_gated_delta_rule(q, k, v, a, b, **args)


def _pool(HV, slots=4, s0=None):
    pool = torch.zeros(slots, HV, D, D, dtype=BF, device="cuda")
    if s0 is not None:
        pool[:] = s0
    return pool


def _slot(pool, HQ, s):
    """The slot as the class sees it: state, open-row count and the open chunk's rows (bank resolved)."""
    gp = get_pool(pool, HQ)
    c = int(gp.cnt[s, 0, 0])
    n, bank = c & 0xFF, (c >> 8) & 1
    assert bool((gp.cnt[s] == c).all())
    return (
        n,
        pool[s].clone(),
        gp.hist_k[s, bank, :, :n].clone(),
        gp.hist_u[s, bank, :, :n].clone(),
        gp.hist_l[s, bank, :, :n].clone(),
    )


def _slot_eq(a, b):
    return a[0] == b[0] and all(
        torch.equal(x, y) for x, y in zip(a[1:], b[1:], strict=True)
    )


def _raw(pool, HQ):
    gp = get_pool(pool, HQ)
    return [t.clone() for t in (pool, gp.hist_k, gp.hist_u, gp.hist_l, gp.cnt)]


def _prefill(x, par, s0, HQ, L=None):
    """One prefill call of the whole sequence from ``s0`` on a fresh pool: rows and the slot."""
    pool = _pool(s0.shape[0], 2, s0)
    out = _bi(pool, [1], _cut(x, 0, L or x[0].shape[0]), par, prefix=[0])
    return out, _slot(pool, HQ, 1)


def _steps(x, par, s0, HQ, T):
    """The sequence in calls of ``T`` tokens; ``T = 1`` is the decode step (no ``prefix_lens``)."""
    L = x[0].shape[0]
    pool = _pool(s0.shape[0], 2, s0)
    rows = []
    for t0 in range(0, L, T):
        piece = _cut(x, t0, min(L, t0 + T))
        rows.append(_bi(pool, [1], piece, par, prefix=None if T == 1 else [t0]))
    return torch.cat(rows), _slot(pool, HQ, 1)


@needs_device
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("steep", [False, True])
@pytest.mark.parametrize("L", [1, 15, 16, 17, 64, 200, 1023])
def test_prefill_decode_pieces_and_steps_are_one_term(heads, steep, L, request):
    if L > 200 and not request.config.getoption("--full", default=False):
        pytest.skip("1023 stepped tokens run with --full")
    HQ, HV = heads
    x, par, s0 = _inputs(L, HQ, HV, 100 + L, steep)
    ref, slot = _prefill(x, par, s0, HQ)
    assert bool(torch.isfinite(ref.float()).all())
    for T in (1, 2, 4, 5, 16, 32):
        out, got = _steps(x, par, s0, HQ, T)
        assert torch.equal(out, ref), f"calls of {T} tokens"
        assert _slot_eq(got, slot), f"slot after calls of {T} tokens"
    cut = max(1, (37 * L) // 100)
    pool = _pool(HV, 2, s0)
    rows = [_bi(pool, [1], _cut(x, 0, cut), par, prefix=[0])]
    if cut < L:
        rows.append(_bi(pool, [1], _cut(x, cut, L), par, prefix=[cut]))
    assert torch.equal(torch.cat(rows), ref)
    assert _slot_eq(_slot(pool, HQ, 1), slot)
    assert slot[0] == spec.phase(L)
    _record(f"one_term/{HQ}x{HV}/{int(steep)}/{L}", ref, *slot[1:])


@needs_device
@pytest.mark.parametrize("heads", [(16, 48), (4, 4)])
def test_batch_and_slots_do_not_enter_the_bits(heads):
    HQ, HV = heads
    lens = [1 + (i * 37) % 90 for i in range(12)]
    seqs = [_inputs(n, HQ, HV, 300 + i, steep=i % 2 == 0) for i, n in enumerate(lens)]
    par = seqs[0][1]
    refs = [_prefill(x, par, s0, HQ) for x, _, s0 in seqs]
    nxt = [_inputs(1, HQ, HV, 400 + i)[0] for i in range(12)]
    dec = []
    for (x, _, s0), y in zip(seqs, nxt, strict=True):
        pool = _pool(HV, 1, s0)
        _bi(pool, [0], x, par, prefix=[0])
        dec.append((_bi(pool, [0], y, par), _slot(pool, HQ, 0)))

    def run(order, slots, size, poison=False):
        pool = _pool(HV, size)
        for i, s in zip(order, slots, strict=True):
            pool[s] = seqs[i][2]
        if poison:
            free = sorted(set(range(size)) - set(slots))
            pool[free] = float("nan")
        cu = _offsets([lens[i] for i in order])
        out = _bi(pool, slots, _cat([seqs[i][0] for i in order]), par, cu=cu)
        for j, (i, s) in enumerate(zip(order, slots, strict=True)):
            assert torch.equal(out[cu[j] : cu[j + 1]], refs[i][0]), (order, i)
            assert _slot_eq(_slot(pool, HQ, s), refs[i][1]), (order, i)
        step = _bi(pool, slots, _cat([nxt[i] for i in order]), par)
        for j, (i, s) in enumerate(zip(order, slots, strict=True)):
            assert torch.equal(step[j : j + 1], dec[i][0]), (order, i)
            assert _slot_eq(_slot(pool, HQ, s), dec[i][1]), (order, i)

    run(list(range(12)), list(range(12)), 12)
    perm = [int(i) for i in np.argsort(_ints(12, 7))]
    run(perm, [int(s) for s in np.argsort(_ints(40, 9))[:12]], 40, poison=True)
    run([5], [3], 4)
    run([11, 0, 7], [2, 0, 1], 3)
    _record(f"batch/{HQ}x{HV}", *[r[0] for r in refs], *[d[0] for d in dec])


@needs_device
def test_neighbour_rows_do_not_enter_the_bits():
    HQ, HV = 16, 48
    a, par, sa = _inputs(70, HQ, HV, 11)
    b, _, sb = _inputs(70, HQ, HV, 12)
    ref, slot = _prefill(a, par, sa, HQ)
    bad = tuple(t.clone() for t in b)
    for t, val in zip(
        bad, (float("nan"), 3e4, float("inf"), float("nan"), -3e4), strict=True
    ):
        t[1::2] = val
    pool = _pool(HV, 3)
    pool[0], pool[1], pool[2] = float("nan"), sa, sb
    out = _bi(pool, [2, 1, 0], _cat([bad, a, bad]), par, cu=[0, 70, 140, 210])
    assert torch.equal(out[70:140], ref) and _slot_eq(_slot(pool, HQ, 1), slot)


@needs_device
@pytest.mark.parametrize("tiles", [1, 2, 4, 8])
def test_placements_agree(tiles, monkeypatch):
    HQ, HV = 16, 32
    for lens in ([200, 33, 16, 1, 47], [20, 3, 31]):
        seqs = [_inputs(n, HQ, HV, 500 + n) for n in lens]
        par = seqs[0][1]
        pool = _pool(HV, len(lens))
        for i, (_, _, s0) in enumerate(seqs):
            pool[i] = s0
        gp = get_pool(pool, HQ)
        assert hasattr(gp, "num_tiles")
        monkeypatch.setattr(gp, "num_tiles", lambda n: tiles)
        cu = _offsets(lens)
        out = _bi(
            pool,
            list(range(len(lens))),
            _cat([s[0] for s in seqs]),
            par,
            cu=cu,
            cu_seqlens_cpu=cu,
        )
        for i, (x, _, s0) in enumerate(seqs):
            ref, slot = _prefill(x, par, s0, HQ)
            assert torch.equal(out[cu[i] : cu[i + 1]], ref), (lens, i)
            assert _slot_eq(_slot(pool, HQ, i), slot), (lens, i)


@needs_device
def test_short_and_long_run_kernels_agree():
    """A 20-token sequence alone runs the short-call kernel; beside a 40-token one, the long-run kernel."""
    HQ, HV = 16, 48
    assert hasattr(s16_pool, "CHAIN_MIN_TOKENS") and s16_pool.CHAIN_MIN_TOKENS == 32
    x, par, s0 = _inputs(20, HQ, HV, 21)
    y, _, s1 = _inputs(40, HQ, HV, 22)
    ref, slot = _prefill(x, par, s0, HQ)
    pool = _pool(HV, 2)
    pool[0], pool[1] = s0, s1
    out = _bi(
        pool, [0, 1], _cat([x, y]), par, cu=[0, 20, 60], cu_seqlens_cpu=[0, 20, 60]
    )
    assert torch.equal(out[:20], ref) and _slot_eq(_slot(pool, HQ, 0), slot)
    # continuing an open chunk with 40 more tokens crosses the kernels the other way
    z, _, _ = _inputs(60, HQ, HV, 23)
    whole, wslot = _prefill(z, par, s0, HQ)
    pool = _pool(HV, 1, s0)
    rows = [_bi(pool, [0], _cut(z, 0, 20), par, prefix=[0])]
    rows.append(_bi(pool, [0], _cut(z, 20, 60), par, prefix=[20]))
    assert torch.equal(torch.cat(rows), whole) and _slot_eq(_slot(pool, HQ, 0), wslot)


@needs_device
@pytest.mark.parametrize("shape", ["decode", "steps4", "prefill48"])
def test_cuda_graph_equals_eager(shape):
    HQ, HV, B = 16, 32, 3
    T = {"decode": 1, "steps4": 4, "prefill48": 48}[shape]
    rounds = 1 if shape == "prefill48" else 20
    seqs = [_inputs(T * rounds, HQ, HV, 600 + i) for i in range(B)]
    par = seqs[0][1]
    slots = [2, 0, 3]

    def load(pool):
        for s, (_, _, s0) in zip(slots, seqs, strict=True):
            pool[s] = s0

    eager, graphed = _pool(HV, 4), _pool(HV, 4)
    load(eager)
    load(graphed)
    cu = list(range(0, B * T + 1, T))
    prefix = _i32([0] * B)
    buf = tuple(torch.empty_like(t) for t in _cat([_cut(s[0], 0, T) for s in seqs]))
    out = torch.empty(B * T, HV, D, dtype=BF, device="cuda")
    kw = dict(
        initial_state=graphed,
        state_indices=_i32(slots),
        cu_seqlens=_i32(cu),
        prefix_lens=None if shape == "decode" else prefix,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        A_log=par[0],
        dt_bias=par[1],
        output=out,
        batch_invariant=True,
    )
    # the module is loaded before the capture
    _bi(_pool(HV, 4), slots, _cat([_cut(s[0], 0, T) for s in seqs]), par, cu=cu)
    side = _raw(graphed, HQ)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flashinfer.chunk_gated_delta_rule(*buf, **kw)
    gp = get_pool(graphed, HQ)
    for dst, src in zip(
        (graphed, gp.hist_k, gp.hist_u, gp.hist_l, gp.cnt), side, strict=True
    ):
        dst.copy_(src)
    for r in range(rounds):
        x = _cat([_cut(s[0], r * T, (r + 1) * T) for s in seqs])
        for dst, src in zip(buf, x, strict=True):
            dst.copy_(src)
        prefix.fill_(r * T)
        graph.replay()
        want = _bi(
            eager,
            slots,
            x,
            par,
            cu=cu,
            prefix=None if shape == "decode" else [r * T] * B,
        )
        assert torch.equal(out, want), (shape, r)
    for s in slots:
        assert _slot_eq(_slot(graphed, HQ, s), _slot(eager, HQ, s))


def _bits(t):
    return t.contiguous().view(torch.int16)


@needs_device
@pytest.mark.parametrize("field", [2, 1, 3])
def test_nonfinite_own_inputs_are_one_term(field):
    HQ = HV = 4
    x, par, s0 = _inputs(40, HQ, HV, 950 + field)
    x = tuple(t.clone() for t in x)
    x[field][3] = float("nan")
    x[field][21] = float("inf")
    ref, ref_slot = _prefill(x, par, s0, HQ)
    assert bool(torch.isnan(ref[3].float()).any())
    for T in (1, 5):
        out, slot = _steps(x, par, s0, HQ, T)
        assert torch.equal(_bits(out), _bits(ref))
        assert slot[0] == ref_slot[0]
        for a, b in zip(slot[1:], ref_slot[1:], strict=True):
            assert torch.equal(
                a.contiguous().view(-1).view(torch.uint8),
                b.contiguous().view(-1).view(torch.uint8),
            )


@needs_device
def test_capture_on_an_unused_pool_raises():
    HQ, HV = 4, 4
    x, par, _ = _inputs(2, HQ, HV, 960)
    pool = _pool(HV, 2)
    slots, out = _i32([0, 1]), torch.empty_like(x[2])
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(BatchInvariantUnsupportedError), torch.cuda.graph(graph):
        flashinfer.chunk_gated_delta_rule(
            *x,
            initial_state=pool,
            state_indices=slots,
            output=out,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            beta_is_logit=True,
            A_log=par[0],
            dt_bias=par[1],
            batch_invariant=True,
        )


@needs_device
def test_offsets_outside_the_rows_return_nan():
    HQ = HV = 4
    x, par, s0 = _inputs(14, HQ, HV, 970)
    big = torch.zeros(14, HV, D, dtype=BF, device="cuda")
    pool = _pool(HV, 3, s0)
    before = _raw(pool, HQ)
    out = _bi(pool, [1], _cut(x, 0, 10), par, cu=[0, 14], prefix=[0], output=big[:10])
    assert bool(torch.isnan(out.float()).all())
    assert not bool(big[10:].any())
    for a, b in zip(before, _raw(pool, HQ), strict=True):
        assert torch.equal(a, b)
    # an empty sequence breaks the bound on the longest one: NaN rows, never another length's rows
    ref, _ = _prefill(x, par, s0, HQ, 10)
    out = _bi(pool, [0, 1, 2], _cut(x, 0, 10), par, cu=[0, 0, 0, 10])
    assert bool(torch.isnan(out.float()).all()) or torch.equal(out, ref)


@needs_device
def test_wrong_prefix_returns_nan_and_keeps_the_slot():
    HQ, HV = 16, 48
    x, par, s0 = _inputs(60, HQ, HV, 31)
    for n in (5, 40):
        pool = _pool(HV, 2, s0)
        _bi(pool, [1], _cut(x, 0, 21), par, prefix=[0])
        before = _raw(pool, HQ)
        out = _bi(pool, [1], _cut(x, 21, 21 + n), par, prefix=[20])
        assert bool(torch.isnan(out.float()).all())
        for a, b in zip(before, _raw(pool, HQ), strict=True):
            assert torch.equal(a, b)
        good = _bi(pool, [1], _cut(x, 21, 21 + n), par, prefix=[21])
        assert torch.equal(good, _prefill(x, par, s0, HQ, 21 + n)[0][21:])


@needs_device
@pytest.mark.parametrize("n", [1, 7, 40])
def test_slot_outside_the_pool_returns_nan(n):
    HQ, HV = 4, 4
    x, par, s0 = _inputs(n, HQ, HV, 41)
    pool = _pool(HV, 3, s0)
    ref = _bi(_pool(HV, 3, s0), [1], x, par, prefix=[0])
    before = _raw(pool, HQ)
    cu = [0, n, 2 * n, 3 * n]
    out = _bi(pool, [-1, 1, 3], _cat([x, x, x]), par, cu=cu, prefix=[0, 0, 0])
    assert bool(torch.isnan(out[:n].float()).all())
    assert bool(torch.isnan(out[2 * n :].float()).all())
    assert torch.equal(out[n : 2 * n], ref)
    after = _raw(pool, HQ)
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
    if n == 1:
        pool = _pool(HV, 3, s0)
        step = _bi(pool, [3, 1, -1], _cat([x, x, x]), par)
        assert bool(torch.isnan(step[0].float()).all())
        assert bool(torch.isnan(step[2].float()).all())
        assert torch.equal(step[1:2], ref)


@needs_device
def test_ragged_prefill_then_decode():
    HQ, HV = 16, 32
    lens = [37, 64, 5, 129]
    more = 20
    seqs = [_inputs(n + more, HQ, HV, 700 + n) for n in lens]
    par = seqs[0][1]
    pool = _pool(HV, 4)
    for i, (_, _, s0) in enumerate(seqs):
        pool[i] = s0
    cu = _offsets(lens)
    slots = [0, 1, 2, 3]
    rows = [[] for _ in lens]
    out = _bi(
        pool,
        slots,
        _cat([_cut(s[0], 0, n) for s, n in zip(seqs, lens, strict=True)]),
        par,
        cu=cu,
    )
    for i in range(4):
        rows[i].append(out[cu[i] : cu[i + 1]])
    for t in range(more):
        x = _cat(
            [_cut(s[0], n + t, n + t + 1) for s, n in zip(seqs, lens, strict=True)]
        )
        step = _bi(pool, slots, x, par)
        for i in range(4):
            rows[i].append(step[i : i + 1])
    for i, (x, _, s0) in enumerate(seqs):
        ref, slot = _prefill(x, par, s0, HQ)
        assert torch.equal(torch.cat(rows[i]), ref), i
        assert _slot_eq(_slot(pool, HQ, i), slot), i


@needs_device
def test_pool_views():
    HQ, HV = 4, 4
    x, par, s0 = _inputs(30, HQ, HV, 51)
    ref, slot = _prefill(x, par, s0, HQ)
    pool = _pool(HV, 6, s0)
    assert get_pool(pool[:3], HQ) is get_pool(pool, HQ)
    assert get_pool(pool[2:], HQ) is not get_pool(pool, HQ)
    rows = [_bi(pool, [1], _cut(x, 0, 20), par, prefix=[0])]
    rows += [_bi(pool[:3], [1], _cut(x, p, p + 1), par) for p in range(20, 30)]
    assert torch.equal(torch.cat(rows), ref) and _slot_eq(_slot(pool, HQ, 1), slot)
    # the view at an offset is another pool: its slot 0 holds no open rows of slot 2
    tail = pool[2:]
    _bi(pool, [2], _cut(x, 0, 5), par, prefix=[0])
    assert _slot(tail, HQ, 0)[0] == 0 and _slot(pool, HQ, 2)[0] == 5


@needs_device
def test_replaced_state_starts_a_new_chunk():
    HQ, HV = 16, 32
    x, par, s0 = _inputs(20, HQ, HV, 61)
    y, _, s1 = _inputs(45, HQ, HV, 62)
    pool = _pool(HV, 2, s0)
    _bi(pool, [1], x, par, prefix=[0])
    pool[1] = s1
    out = _bi(pool, [1], y, par)
    ref, slot = _prefill(y, par, s1, HQ)
    assert torch.equal(out, ref) and _slot_eq(_slot(pool, HQ, 1), slot)


@needs_device
@pytest.mark.parametrize("n", [16, 48, 160])
def test_state_at_a_chunk_boundary_is_a_resume_point(n):
    HQ, HV = 16, 48
    x, par, s0 = _inputs(n + 37, HQ, HV, 71)
    ref, slot = _prefill(x, par, s0, HQ)
    pool = _pool(HV, 2, s0)
    _bi(pool, [1], _cut(x, 0, n), par, prefix=[0])
    assert _slot(pool, HQ, 1)[0] == 0
    fresh = _pool(HV, 3)
    fresh[2] = pool[1]
    out = _bi(fresh, [2], _cut(x, n, n + 37), par, prefix=[0])
    assert torch.equal(out, ref[n:]) and _slot_eq(_slot(fresh, HQ, 2), slot)


def _torch_reference(x, par, s0, scale=None):
    """The term in torch with the kernels' operation and rounding points (plain matmuls for the dots)."""

    def bf(t):
        return t.to(BF).float()

    q, k, v, a, b = (t.float() for t in x)
    T, H = q.shape[:2]
    HV = v.shape[1]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    kn = bf(k * (1.0 / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)))
    qn = bf(q * ((1.0 / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)) * scale))
    kn = kn.repeat_interleave(HV // H, dim=1).permute(1, 0, 2)
    qn = qn.repeat_interleave(HV // H, dim=1).permute(1, 0, 2)
    sp = a + par[1]
    g = (-torch.exp(par[0]) * torch.where(sp > 20.0, sp, torch.log1p(torch.exp(sp)))).T
    beta = torch.sigmoid(b).T
    vv = v.permute(1, 0, 2)
    S = s0.float()
    out = torch.empty(HV, T, D, device=q.device)
    for t0 in range(0, T, CHUNK):
        m = min(CHUNK, T - t0)
        Kc, Qc = kn[:, t0 : t0 + m], qn[:, t0 : t0 + m]
        L = torch.cumsum(g[:, t0 : t0 + m], dim=-1)
        Aco, Bco = S @ Kc.transpose(-1, -2), S @ Qc.transpose(-1, -2)
        W = torch.exp(L.unsqueeze(1) - L.unsqueeze(2))
        cG, cE = W * (Kc @ Kc.transpose(-1, -2)), W * (Kc @ Qc.transpose(-1, -2))
        dd = torch.exp(L)
        U = torch.zeros(HV, m, D, device=q.device)
        for i in range(m):
            pred = dd[:, i, None] * Aco[..., i]
            if i:
                pred = pred + torch.einsum("hj,hjv->hv", cG[:, :i, i], U[:, :i])
            U[:, i] = beta[:, t0 + i, None] * (vv[:, t0 + i] - pred)
            o = dd[:, i, None] * Bco[..., i]
            o = o + torch.einsum("hj,hjv->hv", cE[:, : i + 1, i], U[:, : i + 1])
            out[:, t0 + i] = bf(o)
        if m == CHUNK:
            Ut = bf(torch.exp(L[:, -1:] - L).unsqueeze(-1) * U)
            S = bf(torch.exp(L[:, -1])[:, None, None] * S + Ut.transpose(-1, -2) @ Kc)
    return out.permute(1, 0, 2)


@needs_device
@pytest.mark.parametrize("heads", [(16, 48), (4, 4)])
@pytest.mark.parametrize("steep", [False, True])
def test_agrees_with_the_torch_reference(heads, steep):
    HQ, HV = heads
    x, par, s0 = _inputs(100, HQ, HV, 81, steep)
    out, _ = _prefill(x, par, s0, HQ)
    ref = _torch_reference(x, par, s0)
    err = (out.float() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert float(err) < 1e-3, float(err)


@needs_device
def test_other_routes_are_other_terms():
    """Today's dispatch on the same fused-gate inputs does not compute the class's bits."""
    HQ, HV = 16, 32
    x, par, s0 = _inputs(130, HQ, HV, 91)
    ref, _ = _prefill(x, par, s0, HQ)
    q, k, v, a, b = x
    out = flashinfer.chunk_gated_delta_rule(
        q,
        k,
        v,
        a,
        b,
        initial_state=s0[None].float(),
        cu_seqlens=_i32([0, 130]),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        A_log=par[0],
        dt_bias=par[1],
    )
    out = out[0] if isinstance(out, tuple) else out
    assert not torch.equal(out.to(BF), ref)


def test_fingerprint_and_registry():
    assert class_name("chunk_gated_delta_rule") == spec.CLASS == "gdn.bf16.s16"
    assert CLASSES["chunk_gated_delta_rule"].name == "gdn.bf16.s16"
    fp = fingerprint("chunk_gated_delta_rule")
    assert fp.startswith("gdn-bf16-s16-") and fp == spec.spec_fingerprint()


def test_spec_arithmetic():
    assert CHUNK == 16
    for count in range(CHUNK):
        for n in (1, 2, 15, 16, 17, 63, 64, 200):
            whole, tail = spec.extend_split(count, n)
            assert whole % CHUNK == 0 and 0 <= tail < CHUNK
            assert whole + tail == count + n
        assert spec.commits(count) == (count == CHUNK - 1)
    for p in (0, 1, 15, 16, 17, 1023):
        assert spec.boundary(p) + spec.phase(p) == p and spec.boundary(p) % CHUNK == 0


@needs_device
def test_unsupported_raises():
    HQ, HV = 4, 8
    x, par, _ = _inputs(8, HQ, HV, 0)
    q, k, v, a, b = x
    pool = _pool(HV, 2)
    before = pool.clone()
    used = _pool(HV, 2)
    _bi(used, [0], _inputs(8, 8, HV, 1)[0], par)
    bad = [
        dict(backend="flashinfer"),
        dict(use_cp=False),
        dict(use_cp=True),
        dict(use_gate_in_kernel=False),
        dict(beta_is_logit=False),
        dict(use_qk_l2norm_in_kernel=False),
        dict(A_log=None),
        dict(dt_bias=None),
        dict(A_log=par[0].to(BF)),
        dict(dt_bias=par[1][:4]),
        dict(initial_state=None),
        dict(initial_state=pool.float()),
        dict(initial_state=pool[:, :4]),
        dict(initial_state=used),
        dict(output_state=torch.empty_like(pool)),
        dict(state_indices=torch.zeros(1, dtype=torch.int64, device="cuda")),
        dict(state_indices=_i32([0, 1])),
        dict(checkpoint_every_n_tokens=64),
        dict(state_checkpoints=torch.empty(1, HV, D, D, device="cuda")),
        dict(checkpoint_state_indices=_i32([0])),
        dict(output_state_indices=_i32([0])),
        dict(output=torch.empty(8, HV, D, device="cuda")),
        dict(output=torch.empty(4, HV, D, dtype=BF, device="cuda")),
        dict(prefix_lens=torch.zeros(1, device="cuda")),
    ]
    for kwargs in bad:
        with pytest.raises(BatchInvariantUnsupportedError):
            _bi(pool, [0], x, par, **kwargs)
    for change in (
        dict(q=q.float()),
        dict(q=q[:, :3], k=k[:, :3]),
        dict(a=a.float()),
        dict(b=b.float()),
        dict(b=None),
        dict(v=v[:, :, :64]),
        dict(q=q[:, :, ::2], k=k[:, :, ::2], v=v[:, :, ::2]),
        dict(q=torch.cat([q, q])[::2]),
        dict(v=torch.cat([v, v])[::2]),
    ):
        y = dict(q=q, k=k, v=v, a=a, b=b)
        y.update(change)
        with pytest.raises(BatchInvariantUnsupportedError):
            _bi(pool, [0], tuple(y[n] for n in ("q", "k", "v", "a", "b")), par)
    assert torch.equal(pool, before)
    assert issubclass(BatchInvariantUnsupportedError, ValueError)


def test_unsupported_device_raises():
    q = torch.zeros(2, 4, D, dtype=BF)
    v = torch.zeros(2, 8, D, dtype=BF)
    g = torch.zeros(2, 8, dtype=BF)
    with pytest.raises(BatchInvariantUnsupportedError):
        flashinfer.chunk_gated_delta_rule(
            q,
            q,
            v,
            g,
            g,
            initial_state=torch.zeros(1, 8, D, D, dtype=BF),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            beta_is_logit=True,
            A_log=torch.zeros(8),
            dt_bias=torch.zeros(8),
            batch_invariant=True,
        )
