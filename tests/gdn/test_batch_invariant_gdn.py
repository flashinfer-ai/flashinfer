# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant GDN class ``gdn.bf16.c64`` behind ``chunk_gated_delta_rule(batch_invariant=True)``.

Every comparison is ``torch.equal``: decode, multi-token steps, prefill and prefill in pieces give the rows and the
boundary states of the one-call prefill kernel without context parallelism, whatever the batch and the slots.
``FLASHINFER_GDN_BI_HASH_DIR`` records the output hashes in one process and asserts them in another.
"""

import hashlib
import json
import os

import pytest
import torch

import flashinfer
from flashinfer.batch_invariant import (
    BatchInvariantUnsupportedError,
    class_name,
    fingerprint,
)
from flashinfer.batch_invariant.gdn import spec
from flashinfer.batch_invariant.gdn.pool import get_pool
from flashinfer.utils import get_compute_capability

D = 128
CHUNK = spec.CHUNK
HEADS = [(16, 32), (16, 48), (4, 8)]
_HASHES = {}


def _supported():
    if not torch.cuda.is_available():
        return False
    if int((torch.version.cuda or "0").split(".")[0]) < 13:
        return False
    return tuple(get_compute_capability(torch.device("cuda"))) in spec.GDN_ARCHS


needs_device = pytest.mark.skipif(
    not _supported(),
    reason="the class gdn.bf16.c64 needs SM100 / SM103 / SM107 and CUDA 13",
)


@pytest.fixture(scope="module", autouse=True)
def _hash_file():
    yield
    root = os.environ.get("FLASHINFER_GDN_BI_HASH_DIR")
    if not root or not _HASHES:
        return
    path = os.path.join(root, "gdn_bi_hashes.json")
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        diff = {k for k in _HASHES if k in saved and saved[k] != _HASHES[k]}
        assert not diff, (
            f"output hashes differ from the recorded process: {sorted(diff)}"
        )
    else:
        os.makedirs(root, exist_ok=True)
        with open(path, "w") as f:
            json.dump(_HASHES, f, indent=1, sort_keys=True)


def _record(key, *tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    _HASHES[key] = h.hexdigest()


def _inputs(L, HQ, HV, seed, steep=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    q, k, v = (randn(L, h, D).to(torch.bfloat16) for h in (HQ, HQ, HV))
    g = -torch.nn.functional.softplus(randn(L, HV) - (0.0 if steep else 6.0))
    beta = torch.sigmoid(randn(L, HV))
    s0 = randn(HV, D, D) * 0.1
    return (q, k, v, g, beta), s0


def _cut(x, a, b):
    return tuple(t[a:b] for t in x)


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _reference(x, s0):
    """The one-call prefill without context parallelism: the class's term."""
    q, k, v, g, beta = x
    return flashinfer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=s0[None].clone(),
        output_final_state=True,
        cu_seqlens=_i32([0, q.shape[0]]),
        use_qk_l2norm_in_kernel=True,
        backend="flashinfer",
        use_cp=False,
        gate_domain="log",
    )


def _bi(pool, slots, x, cu=None, prefix=None, **kwargs):
    q, k, v, g, beta = x
    args = dict(
        initial_state=pool,
        state_indices=_i32(slots),
        cu_seqlens=None if cu is None else _i32(cu),
        use_qk_l2norm_in_kernel=True,
        gate_domain="log",
        prefix_lens=None if prefix is None else _i32(prefix),
        batch_invariant=True,
    )
    args.update(kwargs)
    return flashinfer.chunk_gated_delta_rule(q, k, v, g, beta, **args)


def _pool(HV, slots=6):
    return torch.zeros(slots, HV, D, D, dtype=torch.float32, device="cuda")


def _decode(pool, slot, x, a, b):
    return [_bi(pool, [slot], _cut(x, p, p + 1)) for p in range(a, b)]


@needs_device
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("steep", [False, True])
@pytest.mark.parametrize(
    "L,n0", [(1, 0), (64, 0), (200, 0), (200, 63), (200, 64), (200, 65), (330, 190)]
)
def test_prefill_then_decode_is_one_call_prefill(heads, steep, L, n0):
    HQ, HV = heads
    x, s0 = _inputs(L, HQ, HV, 7 * L + n0, steep)
    ref, ref_state = _reference(x, s0)
    pool = _pool(HV)
    pool[3] = s0
    rows = [_bi(pool, [3], _cut(x, 0, max(n0, 1)), prefix=[0])]
    rows += _decode(pool, 3, x, max(n0, 1), L)
    out = torch.cat(rows)
    assert torch.equal(out, ref)
    if L % CHUNK == 0:
        assert torch.equal(pool[3], ref_state[0])
    else:
        b = spec.boundary(L)
        expect = _reference(_cut(x, 0, b), s0)[1][0] if b else s0
        assert torch.equal(pool[3], expect)
    _record(f"split-{HQ}-{HV}-{int(steep)}-{L}-{n0}", out, pool[3])


@needs_device
@pytest.mark.parametrize("heads", HEADS[:2])
@pytest.mark.parametrize("L", [1023, 4096])
def test_long_decode_from_zero(heads, L, request):
    if L > 1023 and not request.config.getoption("--full", default=False):
        pytest.skip("4096 stepped tokens run with --full")
    HQ, HV = heads
    x, s0 = _inputs(L, HQ, HV, L, steep=False)
    ref, ref_state = _reference(x, s0)
    pool = _pool(HV, 2)
    pool[1] = s0
    out = torch.cat(
        [_bi(pool, [1], _cut(x, 0, 1), prefix=[0])] + _decode(pool, 1, x, 1, L)
    )
    assert torch.equal(out, ref)
    if L % CHUNK == 0:
        assert torch.equal(pool[1], ref_state[0])
    _record(f"long-{HQ}-{HV}-{L}", out, pool[1])


@needs_device
@pytest.mark.parametrize("heads", HEADS)
def test_batch_and_slots_do_not_enter_the_bits(heads):
    HQ, HV = heads
    lens = [130, 1, 64, 77, 200, 63]
    slots = [5, 0, 9, 2, 7, 4]
    steps = 70
    seqs = [_inputs(n + steps, HQ, HV, 1000 + n) for n in lens]
    refs = [_reference(x, s0) for x, s0 in seqs]
    pool = _pool(HV, 10)
    for s, (_, s0) in zip(slots, seqs, strict=True):
        pool[s] = s0
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    packed = tuple(
        torch.cat([x[j][:n] for (x, _), n in zip(seqs, lens, strict=True)])
        for j in range(5)
    )
    out = _bi(pool, slots, packed, cu=cu)
    for i, n in enumerate(lens):
        assert torch.equal(out[cu[i] : cu[i + 1]], refs[i][0][:n])
    rows = []
    for t in range(steps):
        tok = tuple(
            torch.cat(
                [x[j][n + t : n + t + 1] for (x, _), n in zip(seqs, lens, strict=True)]
            )
            for j in range(5)
        )
        rows.append(_bi(pool, slots, tok, cu=list(range(len(lens) + 1))))
    rows = torch.stack(rows)
    for i, n in enumerate(lens):
        assert torch.equal(rows[:, i], refs[i][0][n : n + steps])
        b = spec.boundary(n + steps)
        x, s0 = seqs[i]
        assert torch.equal(pool[slots[i]], _reference(_cut(x, 0, b), s0)[1][0])
    # the same request alone, on another slot of another pool
    x, s0 = seqs[3]
    alone = _pool(HV, 3)
    alone[1] = s0
    solo = torch.cat(
        [_bi(alone, [1], _cut(x, 0, lens[3]))]
        + _decode(alone, 1, x, lens[3], lens[3] + steps)
    )
    assert torch.equal(solo, refs[3][0])
    _record(f"batch-{HQ}-{HV}", out, rows)


@needs_device
@pytest.mark.parametrize("heads", HEADS[:2])
@pytest.mark.parametrize(
    "pieces", [[64, 128, 9], [192, 64], [37, 100, 64, 70], [5, 3, 9, 300]]
)
def test_prefill_in_pieces(heads, pieces):
    HQ, HV = heads
    L = sum(pieces)
    x, s0 = _inputs(L, HQ, HV, L + len(pieces))
    ref, _ = _reference(x, s0)
    aligned = all(n % CHUNK == 0 for n in pieces[:-1])
    pool = _pool(HV)
    pool[2] = s0
    rows, pos = [], 0
    for n in pieces:
        # pieces cut at multiples of 64 need no positions; pieces cut anywhere carry prefix_lens
        prefix = None if aligned and n > 1 else [pos]
        rows.append(_bi(pool, [2], _cut(x, pos, pos + n), prefix=prefix))
        pos += n
    assert torch.equal(torch.cat(rows), ref)
    # the slot is decode-ready after any pieces
    more, _ = _inputs(3, HQ, HV, 99)
    longer = tuple(torch.cat([a, b]) for a, b in zip(x, more, strict=True))
    tail = torch.cat(_decode(pool, 2, longer, L, L + 3))
    assert torch.equal(tail, _reference(longer, s0)[0][L:])


@needs_device
@pytest.mark.parametrize("heads", HEADS[:2])
@pytest.mark.parametrize("T", [2, 4, 8, 16, 32])
def test_multi_token_steps(heads, T):
    HQ, HV = heads
    starts = [0, 64 - T // 2, 31, 60]
    total = 2 * T + 3
    seqs = [_inputs(s + total, HQ, HV, 31 * T + s) for s in starts]
    refs = [_reference(x, s0)[0] for x, s0 in seqs]
    pool = _pool(HV, 8)
    slots = [6, 1, 3, 0]
    for i, (s, (x, s0)) in enumerate(zip(starts, seqs, strict=True)):
        pool[slots[i]] = s0
        if s:
            assert torch.equal(_bi(pool, [slots[i]], _cut(x, 0, s)), refs[i][:s])
    pos = list(starts)
    for _ in range(2):
        packed = tuple(
            torch.cat([x[j][p : p + T] for (x, _), p in zip(seqs, pos, strict=True)])
            for j in range(5)
        )
        out = _bi(pool, slots, packed, cu=[T * i for i in range(5)], prefix=pos)
        for i, p in enumerate(pos):
            assert torch.equal(out[T * i : T * (i + 1)], refs[i][p : p + T])
        pos = [p + T for p in pos]
    tok = tuple(
        torch.cat([x[j][p : p + 1] for (x, _), p in zip(seqs, pos, strict=True)])
        for j in range(5)
    )
    out = _bi(pool, slots, tok, cu=list(range(5)))
    for i, p in enumerate(pos):
        assert torch.equal(out[i], refs[i][p])
    _record(f"multi-{HQ}-{HV}-{T}", out)


@needs_device
@pytest.mark.parametrize("T", [2, 8, 16, 32, 63])
def test_placements_agree(T):
    """One span launch of T tokens and T single steps are one term, rows and pool."""
    HQ, HV = 16, 32
    starts = [0, 64 - T // 2, 31]
    seqs = [_inputs(s + 2 * T, HQ, HV, 17 * T + s) for s in starts]
    refs = [_reference(x, s0)[0] for x, s0 in seqs]
    results = []
    for span in (True, False):
        pool = _pool(HV, 4)
        for i, (s, (x, s0)) in enumerate(zip(starts, seqs, strict=True)):
            pool[i] = s0
            if s:
                _bi(pool, [i], _cut(x, 0, s))
        outs = []
        for step in range(2):
            pos = [s + step * T for s in starts]
            packed = tuple(
                torch.cat(
                    [x[j][p : p + T] for (x, _), p in zip(seqs, pos, strict=True)]
                )
                for j in range(5)
            )
            if span:
                o = _bi(pool, [0, 1, 2], packed, cu=[0, T, 2 * T, 3 * T], prefix=pos)
            else:
                o = torch.empty_like(packed[2])
                for t in range(T):
                    tok = tuple(x[t::T].contiguous() for x in packed)
                    o[t::T] = _bi(pool, [0, 1, 2], tok, cu=[0, 1, 2, 3])
            outs.append(o)
            for i, p in enumerate(pos):
                assert torch.equal(o[T * i : T * (i + 1)], refs[i][p : p + T])
        results.append((torch.cat(outs), pool.clone()))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


@needs_device
def test_cuda_graph_step_equals_eager():
    HQ, HV = 16, 32
    B, steps = 3, 140
    seqs = [_inputs(steps, HQ, HV, 500 + i) for i in range(B)]
    refs = [_reference(x, s0)[0] for x, s0 in seqs]
    pool = _pool(HV, 4)
    slots = _i32([2, 0, 3])
    for i, (_, s0) in enumerate(seqs):
        pool[int(slots[i])] = s0
    bufs = tuple(torch.empty_like(seqs[0][0][j][:B]) for j in range(5))
    out = torch.empty(B, HV, D, dtype=torch.bfloat16, device="cuda")
    cu = _i32(list(range(B + 1)))

    def load(t):
        for j in range(5):
            bufs[j].copy_(torch.cat([x[j][t : t + 1] for x, _ in seqs]))

    def step():
        q, k, v, g, beta = bufs
        flashinfer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            state_indices=slots,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
            output=out,
            batch_invariant=True,
        )

    load(0)
    step()
    for i in range(B):
        assert torch.equal(out[i], refs[i][0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    load(1)
    with torch.cuda.graph(graph):
        step()
    for t in range(1, steps):
        load(t)
        graph.replay()
        for i in range(B):
            assert torch.equal(out[i], refs[i][t]), (t, i)


@needs_device
def test_step_without_rebuild_returns_nan():
    """A prefill leaves its open tail's cached rows to the next call; a step launched past that returns NaN."""
    from flashinfer.batch_invariant.gdn import pool as gdn_pool

    HQ, HV = 4, 8
    x, s0 = _inputs(110, HQ, HV, 11)
    ref, _ = _reference(x, s0)
    pool = _pool(HV, 2)
    pool[0] = s0
    assert torch.equal(_bi(pool, [0], _cut(x, 0, 100)), ref[:100])
    gp = get_pool(pool, HQ)
    assert gp.pending == {0: 36}
    q, k, v, g, beta = _cut(x, 100, 101)
    out = torch.empty_like(v)
    gdn_pool._module().gdn_batch_invariant_decode_step(
        q, k, v, g, beta, _i32([0]), out, pool, *gp._side(), D**-0.5
    )
    assert torch.isnan(out.float()).all()
    assert torch.equal(_bi(pool, [0], _cut(x, 100, 101)), ref[100:101])
    assert not gp.pending
    assert torch.equal(_bi(pool, [0], _cut(x, 101, 110), prefix=[101]), ref[101:110])


@needs_device
def test_wrong_prefix_returns_nan_and_keeps_the_slot():
    HQ, HV = 4, 8
    x, s0 = _inputs(40, HQ, HV, 11)
    ref, _ = _reference(x, s0)
    pool = _pool(HV, 2)
    pool[0] = s0
    assert torch.equal(_bi(pool, [0], _cut(x, 0, 30)), ref[:30])
    before = pool.clone()
    for wrong in (29, 31, 95):
        out = _bi(pool, [0], _cut(x, 30, 33), prefix=[wrong])
        assert torch.isnan(out.float()).all()
    assert torch.equal(pool, before)
    assert torch.equal(_bi(pool, [0], _cut(x, 30, 33), prefix=[30]), ref[30:33])
    assert torch.equal(_bi(pool, [0], _cut(x, 33, 34)), ref[33:34])


@needs_device
def test_ragged_prefill_then_decode():
    """Whole chunks and open tails in one call: rows, boundary states and the steps after it."""
    HQ, HV = 16, 32
    lens = [200, 64, 37, 1061, 128, 5]
    seqs = [_inputs(n + 70, HQ, HV, 900 + i) for i, n in enumerate(lens)]
    refs = [_reference(x, s0)[0] for x, s0 in seqs]
    pool = _pool(HV, 8)
    slots = [5, 0, 7, 2, 3, 6]
    for s, (_, s0) in zip(slots, seqs, strict=True):
        pool[s] = s0
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    packed = tuple(
        torch.cat([x[j][:n] for (x, _), n in zip(seqs, lens, strict=True)])
        for j in range(5)
    )
    out = _bi(pool, slots, packed, cu=cu)
    for i, n in enumerate(lens):
        assert torch.equal(out[cu[i] : cu[i + 1]], refs[i][:n]), i
        b = n // CHUNK * CHUNK
        want = _reference(_cut(seqs[i][0], 0, b), seqs[i][1])[1][0] if b else seqs[i][1]
        assert torch.equal(pool[slots[i]], want), i
    rows = []
    for t in range(70):
        tok = tuple(
            torch.cat(
                [x[j][n + t : n + t + 1] for (x, _), n in zip(seqs, lens, strict=True)]
            )
            for j in range(5)
        )
        rows.append(_bi(pool, slots, tok, cu=list(range(len(lens) + 1))))
    rows = torch.stack(rows)
    for i, n in enumerate(lens):
        assert torch.equal(rows[:, i], refs[i][n : n + 70]), i
    _record("ragged-then-decode", torch.cat([out, rows.flatten(0, 1)]))


@needs_device
def test_crossing_spans_on_a_large_grid():
    """More CTAs than fit the device at once, runs that commit a chunk, v-heads sharing a key head."""
    HQ, HV = 16, 48
    B, start, T = 24, 40, 40
    seqs = [_inputs(start + T, HQ, HV, 300 + i) for i in range(B)]
    refs = [_reference(x, s0)[0] for x, s0 in seqs]
    pool = _pool(HV, B)
    for i, (_, s0) in enumerate(seqs):
        pool[i] = s0
    slots = list(range(B))

    def cat(a, b):
        return tuple(torch.cat([x[j][a:b] for x, _ in seqs]) for j in range(5))

    out = _bi(
        pool, slots, cat(0, start), cu=[start * i for i in range(B + 1)], prefix=[0] * B
    )
    for i in range(B):
        assert torch.equal(out[start * i : start * (i + 1)], refs[i][:start]), i
    out = flashinfer.chunk_gated_delta_rule(
        *cat(start, start + T),
        initial_state=pool,
        state_indices=_i32(slots),
        cu_seqlens=_i32([T * i for i in range(B + 1)]),
        cu_seqlens_cpu=[T * i for i in range(B + 1)],
        prefix_lens=torch.full((B,), start, dtype=torch.int64, device="cuda"),
        use_qk_l2norm_in_kernel=True,
        gate_domain="log",
        batch_invariant=True,
    )
    for i in range(B):
        assert torch.equal(out[T * i : T * (i + 1)], refs[i][start:]), i
        want = _reference(_cut(seqs[i][0], 0, CHUNK), seqs[i][1])[1][0]
        assert torch.equal(pool[i], want), i


@needs_device
def test_calls_after_a_ragged_prefill():
    """The tail a prefill leaves is built by whatever call comes next: a span, a mixed call, another prefill."""
    HQ, HV = 4, 8
    x, s0 = _inputs(400, HQ, HV, 21)
    ref, _ = _reference(x, s0)
    pool = _pool(HV, 3)
    for slot in range(3):
        pool[slot] = s0
        assert torch.equal(_bi(pool, [slot], _cut(x, 0, 101)), ref[:101])
    gp = get_pool(pool, HQ)
    assert sorted(gp.pending) == [0, 1, 2]
    assert torch.equal(_bi(pool, [0], _cut(x, 101, 110), prefix=[101]), ref[101:110])
    assert not gp.pending
    with pytest.raises(BatchInvariantUnsupportedError):
        _bi(pool, [1], _cut(x, 101, 300), prefix=[100])
    assert torch.equal(_bi(pool, [1], _cut(x, 101, 300), prefix=[101]), ref[101:300])
    assert torch.equal(_bi(pool, [1], _cut(x, 300, 301)), ref[300:301])
    pool[2] = s0
    assert torch.equal(_bi(pool, [2], _cut(x, 0, 70)), ref[:70])
    assert torch.equal(_bi(pool, [2], _cut(x, 70, 71)), ref[70:71])


@needs_device
def test_capture_with_a_pending_tail_raises():
    HQ, HV = 4, 8
    x, s0 = _inputs(80, HQ, HV, 23)
    pool = _pool(HV, 1)
    pool[0] = s0
    _bi(pool, [0], _cut(x, 0, 70))
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError), torch.cuda.graph(graph):
        _bi(pool, [0], _cut(x, 70, 71))
    torch.cuda.synchronize()
    assert torch.equal(_bi(pool, [0], _cut(x, 70, 71)), _reference(x, s0)[0][70:71])


@needs_device
def test_span_as_cuda_graph():
    """A multi-token step on caller-owned buffers reads nothing on the host, so it captures."""
    HQ, HV = 16, 32
    B, T, calls = 3, 5, 30
    seqs = [_inputs(T * calls, HQ, HV, 700 + i) for i in range(B)]
    refs = [_reference(x, s0)[0] for x, s0 in seqs]
    pool = _pool(HV, B)
    for i, (_, s0) in enumerate(seqs):
        pool[i] = s0
    bufs = tuple(
        torch.empty((B * T, *x.shape[1:]), dtype=x.dtype, device="cuda")
        for x in seqs[0][0]
    )
    out = torch.empty((B * T, HV, D), dtype=torch.bfloat16, device="cuda")
    cu = _i32([T * i for i in range(B + 1)])
    prefix = torch.zeros(B, dtype=torch.int32, device="cuda")
    slots = _i32(list(range(B)))

    def call():
        q, k, v, g, beta = bufs
        flashinfer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=pool,
            state_indices=slots,
            cu_seqlens=cu,
            prefix_lens=prefix,
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
            output=out,
            batch_invariant=True,
        )

    def load(c):
        for j, buf in enumerate(bufs):
            buf.copy_(torch.cat([x[j][T * c : T * (c + 1)] for x, _ in seqs]))
        prefix.fill_(T * c)

    load(0)
    call()
    for i in range(B):
        assert torch.equal(out[T * i : T * (i + 1)], refs[i][:T])
    graph = torch.cuda.CUDAGraph()
    load(1)
    with torch.cuda.graph(graph):
        call()
    for c in range(1, calls):
        load(c)
        graph.replay()
        for i in range(B):
            assert torch.equal(
                out[T * i : T * (i + 1)], refs[i][T * c : T * (c + 1)]
            ), (c, i)


@needs_device
def test_pool_view_keeps_its_state():
    HQ, HV = 4, 8
    x, s0 = _inputs(20, HQ, HV, 5)
    ref, _ = _reference(x, s0)
    buf = _pool(HV, 3)
    buf[1] = s0
    rows = [_bi(buf[:], [1], _cut(x, 0, 7))]
    rows += [_bi(buf.view(3, HV, D, D), [1], _cut(x, p, p + 1)) for p in range(7, 20)]
    assert torch.equal(torch.cat(rows), ref)


@needs_device
def test_replaced_state_starts_a_new_chunk():
    HQ, HV = 4, 8
    x, s0 = _inputs(90, HQ, HV, 1)
    y, t0 = _inputs(90, HQ, HV, 2)
    pool = _pool(HV, 2)
    pool[0] = s0
    _bi(pool, [0], _cut(x, 0, 80))
    pool[0] = t0
    out = torch.cat(
        [_bi(pool, [0], _cut(y, 0, 1), prefix=[0])] + _decode(pool, 0, y, 1, 90)
    )
    assert torch.equal(out, _reference(y, t0)[0])


@needs_device
def test_other_routes_are_other_terms():
    """Today's decode kernel does not compute the prefill's bits: what the class is for."""
    HQ, HV = 16, 32
    x, s0 = _inputs(65, HQ, HV, 3)
    ref, _ = _reference(x, s0)
    q, k, v, g, beta = x
    state = s0[None].clone()
    rows = []
    for p in range(65):
        o, state = flashinfer.chunk_gated_delta_rule(
            q[p : p + 1],
            k[p : p + 1],
            v[p : p + 1],
            g[p : p + 1],
            beta[p : p + 1],
            initial_state=state,
            output_final_state=True,
            cu_seqlens=_i32([0, 1]),
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
        )
        rows.append(o)
    assert not torch.equal(torch.cat(rows), ref)


def test_fingerprint_and_registry():
    assert class_name("chunk_gated_delta_rule") == spec.CLASS
    fp = fingerprint("chunk_gated_delta_rule")
    assert fp.startswith("gdn-bf16-c64-") and fp == spec.spec_fingerprint()


def test_spec_arithmetic():
    for count in range(CHUNK):
        for n in (1, 2, 15, 16, 63, 64, 65, 200):
            whole, tail = spec.extend_split(count, n)
            assert whole % CHUNK == 0 and 0 <= tail < CHUNK
            assert whole + tail == count + n
        assert spec.commits(count) == (spec.next_count(count) == 0)
    assert spec.uses_step_kernel(63) and not spec.uses_step_kernel(64)


@needs_device
def test_unsupported_raises():
    HQ, HV = 4, 8
    x, s0 = _inputs(8, HQ, HV, 0)
    q, k, v, g, beta = x
    pool = _pool(HV, 2)
    before = pool.clone()
    bad = [
        dict(backend="flashinfer"),
        dict(use_cp=False),
        dict(use_cp=True),
        dict(gate_domain="linear"),
        dict(use_qk_l2norm_in_kernel=False),
        dict(initial_state=None),
        dict(initial_state=pool.to(torch.bfloat16)),
        dict(initial_state=pool[:, :4]),
        dict(output_state=torch.empty_like(pool)),
        dict(state_indices=torch.zeros(1, dtype=torch.int64, device="cuda")),
        dict(state_indices=_i32([0, 1])),
        dict(checkpoint_every_n_tokens=64),
        dict(state_checkpoints=torch.empty(1, HV, D, D, device="cuda")),
        dict(output_state_indices=_i32([0])),
        dict(use_gate_in_kernel=True),
        dict(beta_is_logit=True),
        dict(A_log=torch.zeros(HV, device="cuda")),
        dict(output=torch.empty(8, HV, D, device="cuda")),
        dict(prefix_lens=torch.zeros(1, device="cuda")),
    ]
    for kwargs in bad:
        with pytest.raises(BatchInvariantUnsupportedError):
            _bi(pool, [0], x, **kwargs)
    for change in (
        dict(q=q.float()),
        dict(q=q[:, :3], k=k[:, :3]),
        dict(g=g.to(torch.bfloat16)),
        dict(beta=None),
        dict(v=v[:, :, :64]),
    ):
        y = dict(q=q, k=k, v=v, g=g, beta=beta)
        y.update(change)
        with pytest.raises(BatchInvariantUnsupportedError):
            _bi(pool, [0], tuple(y[n] for n in ("q", "k", "v", "g", "beta")))
    assert torch.equal(pool, before)
    assert issubclass(BatchInvariantUnsupportedError, ValueError)


def test_unsupported_device_raises():
    q = torch.zeros(2, 4, D, dtype=torch.bfloat16)
    v = torch.zeros(2, 8, D, dtype=torch.bfloat16)
    g = torch.zeros(2, 8)
    with pytest.raises(BatchInvariantUnsupportedError):
        flashinfer.chunk_gated_delta_rule(
            q,
            q,
            v,
            g,
            g,
            initial_state=torch.zeros(1, 8, D, D),
            use_qk_l2norm_in_kernel=True,
            gate_domain="log",
            batch_invariant=True,
        )


@needs_device
@pytest.mark.parametrize("heads", [(16, 16), (16, 32)])
def test_equal_heads_and_uneven_calls(heads):
    """HV == HQ; unequal token counts per sequence on the step placement; a tail appended to step-built rows."""
    HQ, HV = heads
    lens = [3, 9, 1, 14]
    more = [5, 2, 11, 7]
    big = [20, 17, 40, 16]
    seqs = [
        _inputs(a + b + c + 2, HQ, HV, 40 + a)
        for a, b, c in zip(lens, more, big, strict=True)
    ]
    refs = [_reference(x, s0)[0] for x, s0 in seqs]
    pool = _pool(HV, 5)
    slots = [4, 0, 2, 1]
    for s, (_, s0) in zip(slots, seqs, strict=True):
        pool[s] = s0
    pos = [0] * 4
    for step in (lens, more, big):
        packed = tuple(
            torch.cat(
                [x[j][p : p + n] for (x, _), p, n in zip(seqs, pos, step, strict=True)]
            )
            for j in range(5)
        )
        cu = [0]
        for n in step:
            cu.append(cu[-1] + n)
        out = _bi(pool, slots, packed, cu=cu, prefix=pos)
        for i, (p, n) in enumerate(zip(pos, step, strict=True)):
            assert torch.equal(out[cu[i] : cu[i + 1]], refs[i][p : p + n])
        pos = [p + n for p, n in zip(pos, step, strict=True)]
    tok = tuple(
        torch.cat([x[j][p : p + 1] for (x, _), p in zip(seqs, pos, strict=True)])
        for j in range(5)
    )
    out = _bi(pool, slots, tok, cu=list(range(5)))
    for i, p in enumerate(pos):
        assert torch.equal(out[i], refs[i][p])


@needs_device
def test_many_ragged_sequences():
    """More than 32 sequences with open tails: the indexed packing of the prefill placement."""
    HQ, HV = 4, 8
    lens = [64 + 3 * i + 17 for i in range(40)]
    seqs = [_inputs(n + 1, HQ, HV, 300 + i) for i, n in enumerate(lens)]
    pool = _pool(HV, 40)
    for i, (_, s0) in enumerate(seqs):
        pool[i] = s0
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    packed = tuple(
        torch.cat([x[j][:n] for (x, _), n in zip(seqs, lens, strict=True)])
        for j in range(5)
    )
    out = _bi(pool, list(range(40)), packed, cu=cu)
    tok = tuple(
        torch.cat([x[j][n : n + 1] for (x, _), n in zip(seqs, lens, strict=True)])
        for j in range(5)
    )
    step = _bi(pool, list(range(40)), tok, cu=list(range(41)))
    for i, n in enumerate(lens):
        ref = _reference(*seqs[i])[0]
        assert torch.equal(out[cu[i] : cu[i + 1]], ref[:n])
        assert torch.equal(step[i], ref[n])


@needs_device
def test_views_of_the_pool_share_its_state():
    """Detached, inference-mode and shorter views of one buffer are the same pool."""
    HQ, HV = 4, 8
    x, s0 = _inputs(30, HQ, HV, 8)
    ref, _ = _reference(x, s0)
    buf = _pool(HV, 4)
    buf[1] = s0
    rows = [_bi(buf, [1], _cut(x, 0, 5))]
    for p in range(5, 30):
        if p % 3 == 0:
            rows.append(_bi(buf.detach(), [1], _cut(x, p, p + 1)))
        elif p % 3 == 1:
            with torch.inference_mode():
                rows.append(_bi(buf[:], [1], _cut(x, p, p + 1)).clone())
        else:
            rows.append(_bi(buf[:2], [1], _cut(x, p, p + 1)))
    assert torch.equal(torch.cat(rows), ref)


@needs_device
def test_slot_outside_the_pool_returns_nan():
    HQ, HV = 4, 8
    x, _ = _inputs(2, HQ, HV, 4)
    pool = _pool(HV, 2)
    _bi(pool, [0, 1], x, cu=[0, 1, 2], prefix=[0, 0])
    out = _bi(pool, [1, 7], x, cu=[0, 1, 2])
    assert torch.isnan(out[1].float()).all() and not torch.isnan(out[0].float()).any()
    y, _ = _inputs(6, HQ, HV, 5)
    out = _bi(pool, [9, 0], y, cu=[0, 3, 6], prefix=[0, 1])
    assert torch.isnan(out[:3].float()).all() and not torch.isnan(out[3:].float()).any()
