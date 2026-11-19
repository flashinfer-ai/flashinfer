# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The multi-token verify (speculative decode, MTP) of the batch-invariant KDA class ``kda.bf16.c16``.

Every comparison is bitwise. A verify of ``T`` tokens is checked against the pinned prefill of the tokens the request
had kept plus the drafts (the class's term), against ``T`` single decode steps on a twin pool, and its settlement with
``a`` rows kept against the twin after ``a`` decode steps: the pool, the open-row counts and every cached row the class
reads. ``FLASHINFER_KDA_BI_HASH_DIR`` records the hashes of one run and asserts them in another (across devices);
``--digest`` prints the hashes of a small case set for the fresh-process test.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.batch_invariant as bi
from flashinfer.batch_invariant import BatchInvariantUnsupportedError
from flashinfer.batch_invariant.kda import pool as bi_pool
from flashinfer.batch_invariant.kda import spec
from flashinfer.batch_invariant.kda.api import (
    recurrent_kda_batch_invariant_accept,
    recurrent_kda_batch_invariant_reset,
    recurrent_kda_batch_invariant_verify,
)
from flashinfer.kda import recurrent_kda
from tests.kda.test_batch_invariant_recurrent_kda import (
    GATES,
    LB,
    SEED,
    D,
    _check_hash,
    _fused,
    _hash,
    bi_decode,
    bi_prefill,
    make_seq,
    needs_device,
    new_pool,
    one_shot,
    params,
    same_bits,
    sub,
    token,
)

KEYS = ("q", "k", "v", "g", "beta")
HEADS = [
    int(x)
    for x in os.environ.get("FLASHINFER_KDA_BI_VERIFY_HEADS", "64,32,16,8").split(",")
]
#: Tokens per verify: one, the MTP sizes and a whole chunk (the most a verify takes).
SIZES = [1, 2, 6, 9, 16]
#: Start positions on both sides of the first three boundaries.
STARTS = [0, 3, 11, 15, 16, 21, 37, 47]
SLOTS = [4, 0, 9, 2, 7, 3, 1, 5]


def ints(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def kept_counts(T, n, r):
    """The rows each of ``n`` requests keeps in round ``r``: all, one and none in turn, else a count of its own."""
    return [(T, 1, 0, (i * 5 + r * 3) % (T + 1))[(i + r) % 4] for i in range(n)]


class Requests:
    """``n`` requests with their true tokens and a second stream the rejected drafts come from; ``kept`` is what each
    request has consumed so far (the tokens its slot's state stands for)."""

    def __init__(self, h, n, length, seed, starts):
        self.h, self.n = h, n
        gate = lambda i: GATES["long" if i % 2 else "steep"]  # noqa: E731
        self.true = [make_seq(length, h, seed + i, gate(i)) for i in range(n)]
        self.alt = [make_seq(length, h, seed + 500 + i, gate(i)) for i in range(n)]
        self.pos = list(starts[:n])

    def drafts(self, T, kept):
        """``T`` tokens per request from its position: the first ``kept[i]`` are its true ones, the rest are not."""
        out = []
        for s, alt, p, a in zip(self.true, self.alt, self.pos, kept, strict=True):
            out.append(
                {
                    key: torch.cat([s[key][p : p + a], alt[key][p + a : p + T]], 0)
                    for key in KEYS
                }
            )
        return out

    def stacked(self, drafts):
        return {key: torch.stack([d[key] for d in drafts], 0) for key in KEYS}

    def reference(self, i, draft, p):
        """The term of request ``i``'s verify: the pinned prefill of its kept tokens and the drafts, the drafts' rows."""
        s, T = self.true[i], draft["q"].shape[0]
        # two more tokens behind the drafts: a one-token call would be a decode step of the operator, not a prefill
        full = {
            key: torch.cat([s[key][: self.pos[i]], draft[key], self.alt[i][key][:2]], 0)
            for key in KEYS
        }
        full["s0"] = s["s0"]
        return one_shot(full, p)[0][self.pos[i] : self.pos[i] + T]


def start_pools(req, slots, p):
    """The pool and a twin, each prefilled to the requests' start positions."""
    pools = []
    for _ in range(2):
        pool = new_pool(req.true, slots)
        live = [i for i in range(req.n) if req.pos[i]]
        if live:
            bi_prefill(
                [sub(req.true[i], 0, req.pos[i]) for i in live],
                pool,
                [slots[i] for i in live],
                p,
            )
        pools.append(pool)
    return pools


def twin_steps(req, twin, slots, p, counts):
    """Request ``i`` advances ``counts[i]`` single decode steps on the twin; returns its rows per request."""
    rows = [[] for _ in range(req.n)]
    for t in range(max(counts) if counts else 0):
        live = [i for i in range(req.n) if t < counts[i]]
        toks = {
            key: torch.cat(
                [req.true[i][key][req.pos[i] + t : req.pos[i] + t + 1] for i in live], 0
            )
            for key in KEYS
        }
        out = bi_decode(toks, twin, [slots[i] for i in live], p)
        for j, i in enumerate(live):
            rows[i].append(out[j])
    return rows


def assert_same_state(pool, twin, slots, where=""):
    """The class's state of ``slots`` on ``pool`` is the twin's: the boundary states, their bf16 copies, the stamps,
    the open-row counts and what the class reads at those counts (the decay prefix and the cached rows below them)."""
    a, b = bi_pool.get_pool(pool), bi_pool.get_pool(twin)
    idx = torch.tensor(slots, dtype=torch.int64, device="cuda")
    pick = lambda t: t.index_select(0, idx)  # noqa: E731
    assert same_bits(pick(pool), pick(twin)), f"pool {where}"
    assert same_bits(pick(a.cnt), pick(b.cnt)), f"counts {where}"
    assert same_bits(pick(a.stamp), pick(b.stamp)), f"stamp {where}"
    assert same_bits(pick(a.state_bf16), pick(b.state_bf16)), f"bf16 copy {where}"
    cnt = pick(a.cnt)
    assert bool((cnt >= 0).all()) and bool((cnt == cnt[:, :1]).all())
    open_rows = cnt > 0
    assert same_bits(pick(a.prefix)[open_rows], pick(b.prefix)[open_rows]), (
        f"prefix {where}"
    )
    live = torch.arange(spec.CHUNK, device="cuda")[None, None, :] < cnt[:, :, None]
    for name in ("k_inv_rows", "k_inv_f32_rows", "rhs_rows", "update_rows", "l_rows"):
        x, y = pick(getattr(a, name)), pick(getattr(b, name))
        assert same_bits(x[live], y[live]), f"{name} {where}"


def run_rounds(h, n, T, rounds, p, *, call, seed=SEED + 4000, check_twin=True):
    """``rounds`` verifies of ``T`` tokens per request, each settling the one before with its kept counts; returns
    every verify's rows. ``call(tokens, pool, slots, accepted)`` is the entry under test."""
    length = max(STARTS[:n]) + rounds * T + 24
    req = Requests(h, n, length, seed, STARTS)
    slots = SLOTS[:n]
    pool, twin = start_pools(req, slots, p)
    outs, last = [], None
    for r in range(rounds):
        kept = kept_counts(T, n, r)
        drafts = req.drafts(T, kept)
        out = call(req.stacked(drafts), pool, slots, last)
        for i in range(n):
            ref = req.reference(i, drafts[i], p)
            assert same_bits(out[i], ref), f"round {r}, request {i} at {req.pos[i]}"
        steps = twin_steps(req, twin, slots, p, kept)
        for i in range(n):
            for t, row in enumerate(steps[i]):
                assert same_bits(out[i, t], row), f"round {r}, request {i}, step {t}"
        outs.append(out)
        req.pos = [q0 + a for q0, a in zip(req.pos, kept, strict=True)]
        last = ints(kept)
    recurrent_kda_batch_invariant_accept(
        pool,
        last,
        ssm_state_indices=ints(slots),
    )
    if check_twin:
        assert_same_state(pool, twin, slots, "after the last settlement")
    # the requests go on as single decode steps across the next boundary
    for t in range(18):
        toks = {
            key: torch.cat(
                [
                    s[key][q0 + t : q0 + t + 1]
                    for s, q0 in zip(req.true, req.pos, strict=True)
                ],
                0,
            )
            for key in KEYS
        }
        a, b = bi_decode(toks, pool, slots, p), bi_decode(toks, twin, slots, p)
        assert same_bits(a, b), f"decode step {t} after the verifies"
    assert_same_state(pool, twin, slots, "after the decode steps")
    return torch.cat(outs, 0), pool


def call_api(p):
    def call(tokens, pool, slots, accepted):
        return recurrent_kda_batch_invariant_verify(
            *(tokens[key] for key in KEYS),
            A_log=p["A_log"],
            dt_bias=p["dt_bias"],
            initial_state=pool,
            lower_bound=LB,
            ssm_state_indices=slots if isinstance(slots, torch.Tensor) else ints(slots),
            num_accepted_tokens=accepted,
        )

    return call


def call_operator(p, packed):
    """``recurrent_kda(num_spec_tokens=)`` in its ``[N, 1 + S]`` or its packed ``cu_seqlens`` form."""

    def call(tokens, pool, slots, accepted):
        N, T = tokens["q"].shape[:2]
        if T == 1:
            return call_api(p)(tokens, pool, slots, accepted)
        if packed:
            args = [
                tokens[key].reshape(1, N * T, *tokens[key].shape[2:]) for key in KEYS
            ]
            extra = dict(
                cu_seqlens=torch.arange(N + 1, dtype=torch.int32, device="cuda") * T
            )
        else:
            args, extra = [tokens[key] for key in KEYS], {}
        out, st = recurrent_kda(
            *args,
            initial_state=pool,
            ssm_state_indices=ints(slots),
            num_spec_tokens=T - 1,
            num_accepted_tokens=accepted,
            output_final_state=True,
            batch_invariant=True,
            **_fused(p),
            **extra,
        )
        assert st is pool
        return out.reshape(N, T, -1, D)

    return call


# -- device-free ---------------------------------------------------------------------------------------------------


def test_verify_adds_no_fingerprint_source():
    """The verify's kernels hold no row arithmetic and are not sources of the class: its fingerprint is the one of
    the class without the verify."""
    sources = spec.TERM_SOURCES + spec.TERM_INCLUDES + spec.TERM_CSRC
    assert sources == (
        "kda_kernels/kda_chunked_bt16.py",
        "jit/kda_batch_invariant.py",
        "flashinfer/kda/batch_invariant_decode.cuh",
        "kda_batch_invariant.cu",
    )
    assert bi.class_name("recurrent_kda") == "kda.bf16.c16"


@pytest.mark.parametrize("count", range(spec.CHUNK))
def test_verify_split_arithmetic(count):
    """What a verify of ``n`` rows on a slot with ``count`` open rows leaves and saves, and where ``a`` kept rows
    put the slot: the arithmetic the Lean cases pin."""
    for n in range(1, spec.CHUNK + 1):
        after, saved = spec.verify_split(count, n)
        assert after == (count + n) % spec.CHUNK
        crossed = count + n >= spec.CHUNK
        assert (saved >= 0) == crossed
        if crossed:
            assert saved == min(count, count + n - spec.CHUNK)
        for a in range(n + 1):
            assert spec.accepted_count(count, a) == (count + a) % spec.CHUNK
            assert spec.accepted_commits(count, a) == (count + a) // spec.CHUNK


# -- device --------------------------------------------------------------------------------------------------------


@needs_device
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("T", SIZES)
def test_verify_equals_decode_steps_and_prefill(h, T):
    """Six requests at positions around the chunk boundaries, five rounds with every kind of settlement (all rows
    kept, one, none, a count of its own): each verify's rows are the prefill's and the decode steps', and the state
    after the last settlement is the state of the kept tokens fed as single decode steps."""
    p = params(h)
    n = 6 if h <= 32 else 4
    out, _ = run_rounds(h, n, T, 5, p, call=call_api(p))
    _check_hash(f"verify_T{T}_h{h}", out)


@needs_device
@pytest.mark.parametrize("n", [1, 3, 8])
@pytest.mark.parametrize("T", [2, 6, 9])
def test_verify_rows_do_not_depend_on_the_batch(n, T):
    """The first request's rows are the same bits alone, among three and among eight (other slots, other neighbours)."""
    h = 16
    p = params(h)
    out, _ = run_rounds(h, n, T, 4, p, call=call_api(p))
    alone, _ = run_rounds(h, 1, T, 4, p, call=call_api(p))
    assert same_bits(out.view(4, n, T, h, D)[:, 0], alone.view(4, 1, T, h, D)[:, 0])


@needs_device
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("T", [2, 6, 16])
def test_operator_spelling(packed, T):
    """``recurrent_kda(num_spec_tokens=T - 1, num_accepted_tokens=, batch_invariant=True)`` is the same call."""
    h = 16
    p = params(h)
    out, _ = run_rounds(h, 5, T, 4, p, call=call_operator(p, packed))
    ref, _ = run_rounds(h, 5, T, 4, p, call=call_api(p))
    assert same_bits(out, ref)


@needs_device
def test_rerun_gives_the_same_bits():
    h, T = 16, 6
    p = params(h)
    runs = [run_rounds(h, 6, T, 5, p, call=call_api(p)) for _ in range(3)]
    for out, pool in runs[1:]:
        assert same_bits(out, runs[0][0]) and same_bits(pool, runs[0][1])


@needs_device
def test_explicit_accept_then_other_calls():
    """Flow with the settlement as its own call: verify, accept, then a decode step, a multi-token step and a prefill
    continue from the kept position; an unsettled verify counts as kept in full."""
    from flashinfer.batch_invariant.kda.api import recurrent_kda_batch_invariant_extend

    h, T = 16, 6
    p = params(h)
    req = Requests(h, 4, 200, SEED + 4100, [5, 13, 16, 30])
    slots = [3, 0, 2, 1]
    pool, twin = start_pools(req, slots, p)
    ids = ints(slots)
    verify = call_api(p)
    for r, kept in enumerate([[2, 6, 0, 3], [6, 1, 4, 5], [3, 3, 3, 3]]):
        drafts = req.drafts(T, kept)
        out = verify(req.stacked(drafts), pool, slots, None)
        for i in range(4):
            assert same_bits(out[i], req.reference(i, drafts[i], p)), f"round {r}"
        recurrent_kda_batch_invariant_accept(
            pool,
            ints(kept),
            ssm_state_indices=ids,
        )
        twin_steps(req, twin, slots, p, kept)
        req.pos = [q0 + a for q0, a in zip(req.pos, kept, strict=True)]
        assert_same_state(pool, twin, slots, f"round {r}")
        # a second settlement of the same verify changes nothing
        recurrent_kda_batch_invariant_accept(
            pool,
            ints([0, 0, 0, 0]),
            ssm_state_indices=ids,
        )
        assert_same_state(pool, twin, slots, f"round {r}, settled twice")
    # an unsettled verify of true tokens, then a multi-token step: the slot is T tokens ahead
    drafts = req.drafts(T, [T] * 4)
    verify(req.stacked(drafts), pool, slots, None)
    twin_steps(req, twin, slots, p, [T] * 4)
    req.pos = [q0 + T for q0 in req.pos]
    refs = [one_shot(s, p)[0] for s in req.true]
    stack = (
        lambda key: torch.stack(  # noqa: E731
            [s[key][q0 : q0 + 5] for s, q0 in zip(req.true, req.pos, strict=True)], 0
        )
    )
    out = recurrent_kda_batch_invariant_extend(
        *(stack(key) for key in KEYS),
        A_log=p["A_log"],
        dt_bias=p["dt_bias"],
        initial_state=pool,
        lower_bound=LB,
        ssm_state_indices=ids,
    )
    for i in range(4):
        assert same_bits(out[i], refs[i][req.pos[i] : req.pos[i] + 5])
    # a prefill takes the slot as a boundary and drops the verify: a late settlement is a no-op
    s = req.true[0]
    pool[slots[0]].copy_(s["s0"])
    rows = bi_prefill([sub(s, 0, 40)], pool, [slots[0]], p)
    assert same_bits(rows, refs[0][:40])
    recurrent_kda_batch_invariant_accept(
        pool,
        ints([0]),
        ssm_state_indices=ints([slots[0]]),
    )
    assert same_bits(bi_decode(token(s, 40), pool, [slots[0]], p)[0], refs[0][40])


@needs_device
@pytest.mark.parametrize("T", [3, 6, 16])
def test_graph_replay(T):
    """Settlement and verify captured once as a CUDA graph (nothing is read back) and replayed across chunk
    boundaries with changing kept counts: the rows and the final state are the eager ones."""
    h, n, rounds = 16, 4, 9
    p = params(h)
    req = Requests(h, n, 60 + rounds * T, SEED + 4200, [0, 7, 15, 26])
    slots = [2, 5, 0, 3]
    pool, twin = start_pools(req, slots, p)
    ids = ints(slots)
    kept = kept_counts(T, n, 0)
    drafts = req.drafts(T, kept)
    static = {key: t.clone() for key, t in req.stacked(drafts).items()}
    accepted = ints([T] * n)
    out_buf = torch.empty(n, T, h, D, dtype=torch.bfloat16, device="cuda")

    def step():
        return recurrent_kda_batch_invariant_verify(
            *(static[key] for key in KEYS),
            A_log=p["A_log"],
            dt_bias=p["dt_bias"],
            initial_state=pool,
            lower_bound=LB,
            ssm_state_indices=ids,
            num_accepted_tokens=accepted,
            output=out_buf,
        )

    def check(out, r):
        for i in range(n):
            ref = req.reference(i, drafts[i], p)
            assert same_bits(out[i], ref), f"round {r}, request {i}"
        twin_steps(req, twin, slots, p, kept)
        req.pos = [q0 + a for q0, a in zip(req.pos, kept, strict=True)]
        accepted.copy_(ints(kept))

    check(step(), 0)  # eager: warms the kernels and allocates the verify record
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s_), torch.cuda.graph(g, stream=s_):
        out = step()
    torch.cuda.current_stream().wait_stream(s_)
    torch.cuda.synchronize()
    assert out is out_buf
    for r in range(1, rounds):
        kept = kept_counts(T, n, r)
        drafts = req.drafts(T, kept)
        for key, t in req.stacked(drafts).items():
            static[key].copy_(t)
        g.replay()
        torch.cuda.synchronize()
        check(out, r)
    recurrent_kda_batch_invariant_accept(
        pool,
        accepted,
        ssm_state_indices=ids,
    )
    assert_same_state(pool, twin, slots, "after the replays")


@needs_device
def test_first_verify_inside_a_capture_is_refused():
    h = 16
    p = params(h)
    req = Requests(h, 2, 80, SEED + 4300, [4, 9])
    pool, _ = start_pools(req, [0, 1], p)
    verify = call_api(p)
    ids = ints([0, 1])
    small = req.stacked(req.drafts(3, [3, 3]))
    large = req.stacked(req.drafts(7, [7, 7]))

    def capture(tokens):
        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.stream(s_), torch.cuda.graph(g, stream=s_):
                verify(tokens, pool, ids, None)
        finally:
            torch.cuda.current_stream().wait_stream(s_)
            torch.cuda.synchronize()

    with pytest.raises(BatchInvariantUnsupportedError, match="eager verify"):
        capture(small)
    verify(small, pool, [0, 1], None)
    recurrent_kda_batch_invariant_accept(
        pool,
        ints([0, 0]),
    )
    with pytest.raises(BatchInvariantUnsupportedError, match="eager verify"):
        capture(large)


@needs_device
def test_record_grows_with_a_verify_pending():
    """A longer verify than any before grows the record; the pending shorter verify is still settled from it."""
    h = 16
    p = params(h)
    req = Requests(h, 3, 120, SEED + 4400, [2, 14, 20])
    slots = [1, 2, 0]
    pool, twin = start_pools(req, slots, p)
    verify = call_api(p)
    last = None
    for r, (T, kept) in enumerate([(3, [1, 3, 2]), (9, [4, 0, 9]), (5, [5, 2, 1])]):
        drafts = req.drafts(T, kept)
        out = verify(req.stacked(drafts), pool, slots, last)
        for i in range(3):
            assert same_bits(out[i], req.reference(i, drafts[i], p)), f"round {r}"
        twin_steps(req, twin, slots, p, kept)
        req.pos = [q0 + a for q0, a in zip(req.pos, kept, strict=True)]
        last = ints(kept)
    recurrent_kda_batch_invariant_accept(
        pool,
        last,
        ssm_state_indices=ints(slots),
    )
    assert_same_state(pool, twin, slots)


@needs_device
def test_kept_counts_outside_the_verify_are_clamped():
    """A count above the verify's tokens keeps all of them, a negative one none."""
    h, T = 16, 6
    p = params(h)
    req = Requests(h, 2, 80, SEED + 4500, [12, 12])
    slots = [0, 1]
    pool, twin = start_pools(req, slots, p)
    drafts = req.drafts(T, [T, 0])
    call_api(p)(req.stacked(drafts), pool, slots, None)
    recurrent_kda_batch_invariant_accept(
        pool,
        ints([T + 9, -3]),
    )
    twin_steps(req, twin, slots, p, [T, 0])
    assert_same_state(pool, twin, slots)


@needs_device
def test_replaced_state_and_reset_drop_the_verify():
    """A slot whose state the caller replaced after a verify (a new request), and a slot that was reset: the late
    settlement leaves both alone and the new request decodes from its own state."""
    h, T = 16, 6
    p = params(h)
    req = Requests(h, 2, 80, SEED + 4600, [13, 13])
    slots = [0, 1]
    pool, _ = start_pools(req, slots, p)
    call_api(p)(req.stacked(req.drafts(T, [2, 2])), pool, slots, None)
    fresh = [make_seq(40, h, SEED + 4700 + i, GATES["steep"]) for i in range(2)]
    refs = [one_shot(s, p)[0] for s in fresh]
    pool[0].copy_(fresh[0]["s0"])
    pool[1].copy_(fresh[1]["s0"])
    recurrent_kda_batch_invariant_reset(pool, ints([1]))
    recurrent_kda_batch_invariant_accept(
        pool,
        ints([2, 2]),
    )
    for t in range(20):
        toks = {key: torch.cat([s[key][t : t + 1] for s in fresh], 0) for key in KEYS}
        rows = bi_decode(toks, pool, slots, p)
        assert same_bits(rows[0], refs[0][t]) and same_bits(rows[1], refs[1][t])


@needs_device
def test_verify_into_a_replaced_state():
    """A verify as the first call of a new request in a slot with open rows of the previous one, across a boundary."""
    h, T = 16, 16
    p = params(h)
    req = Requests(h, 2, 90, SEED + 4800, [9, 9])
    slots = [0, 1]
    pool, _ = start_pools(req, slots, p)
    fresh = Requests(h, 2, 90, SEED + 4900, [0, 0])
    twin = new_pool(fresh.true, slots)
    pool[0].copy_(fresh.true[0]["s0"])
    pool[1].copy_(fresh.true[1]["s0"])
    kept = [3, 14]
    drafts = fresh.drafts(T, kept)
    out = call_api(p)(fresh.stacked(drafts), pool, slots, None)
    for i in range(2):
        assert same_bits(out[i], fresh.reference(i, drafts[i], p))
    recurrent_kda_batch_invariant_accept(
        pool,
        ints(kept),
    )
    twin_steps(fresh, twin, slots, p, kept)
    assert_same_state(pool, twin, slots)


@needs_device
@pytest.mark.parametrize("bad", [-1, 100000])
def test_slot_outside_the_pool(bad):
    """A padded row (a slot outside the pool) gets NaN rows; the other requests' rows and states are untouched by it."""
    h, T = 16, 6
    p = params(h)
    req = Requests(h, 3, 80, SEED + 5000, [5, 14, 16])
    pool, twin = start_pools(req, [0, 1, 2], p)
    kept = [2, 6, 4]
    drafts = req.drafts(T, kept)
    out = call_api(p)(req.stacked(drafts), pool, [0, bad, 2], None)
    assert bool(torch.isnan(out[1].float()).all())
    for i in (0, 2):
        assert same_bits(out[i], req.reference(i, drafts[i], p))
    recurrent_kda_batch_invariant_accept(
        pool,
        ints(kept),
        ssm_state_indices=ints([0, bad, 2]),
    )
    twin_steps(req, twin, [0, 1, 2], p, [2, 0, 4])
    assert_same_state(pool, twin, [0, 2])
    assert same_bits(pool[1], twin[1])


@needs_device
def test_unsupported_verify_inputs_raise_and_nothing_runs():
    h = 16
    p = params(h)
    req = Requests(h, 2, 80, SEED + 5100, [5, 9])
    pool, twin = start_pools(req, [0, 1], p)
    tok = req.stacked(req.drafts(4, [4, 4]))
    args = [tok[key] for key in KEYS]
    base = dict(initial_state=pool, batch_invariant=True, **_fused(p))
    two = torch.zeros(2, 4, dtype=torch.int32, device="cuda")
    bad = [
        dict(num_spec_tokens=3, ssm_state_indices=two),
        dict(num_spec_tokens=2),
        dict(num_spec_tokens=0, num_accepted_tokens=ints([1, 1])),
        dict(num_spec_tokens=True),
        dict(num_spec_tokens=3, num_accepted_tokens=ints([1])),
        dict(num_spec_tokens=3, num_accepted_tokens=ints([1, 1]).long()),
        dict(num_spec_tokens=3, num_accepted_tokens=[1, 1]),
        dict(num_accepted_tokens=ints([1, 1])),
        dict(num_spec_tokens=3, initial_state=pool.to(torch.bfloat16)),
        dict(num_spec_tokens=3, initial_state=None),
        dict(num_spec_tokens=3, backend="cute-dsl"),
        dict(num_spec_tokens=3, disable_state_update=True),
        dict(
            num_spec_tokens=3,
            output=torch.empty(2, 4, h, D, dtype=torch.float32, device="cuda"),
        ),
        dict(num_spec_tokens=3, output=pool.view(torch.bfloat16)[:1, :8]),
    ]
    for kw in bad:
        with pytest.raises(BatchInvariantUnsupportedError):
            recurrent_kda(*args, **{**base, **kw})
    long = req.stacked(req.drafts(17, [17, 17]))
    with pytest.raises(BatchInvariantUnsupportedError, match="at most 16"):
        call_api(p)(long, pool, [0, 1], None)
    with pytest.raises(BatchInvariantUnsupportedError):
        recurrent_kda_batch_invariant_accept(
            pool,
            ints([1]),
            ssm_state_indices=ints([0, 1]),
        )
    assert_same_state(pool, twin, [0, 1])


def _digest_cases():
    out = {}
    for h, T in ((16, 6), (64, 9)):
        p = params(h)
        rows, pool = run_rounds(h, 4, T, 4, p, call=call_api(p), check_twin=False)
        out[f"verify_rows_h{h}_T{T}"] = _hash(rows)
        out[f"verify_state_h{h}_T{T}"] = _hash(pool)
    return out


@needs_device
def test_verify_repeats_across_fresh_processes():
    runs = []
    for _ in range(2):
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.kda.test_batch_invariant_kda_verify",
                "--digest",
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
        )
        runs.append(json.loads(done.stdout.strip().splitlines()[-1]))
    here = _digest_cases()
    assert len(here) == 4 and runs[0] == here and runs[1] == here
    d = os.environ.get("FLASHINFER_KDA_BI_HASH_DIR")
    for name, value in here.items() if d else ():
        f = Path(d) / f"{name}.sha256"
        if f.exists():
            assert f.read_text().strip() == value, (
                f"{name}: differs from the recorded run"
            )
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(value + "\n")


if __name__ == "__main__" and "--digest" in sys.argv:
    print(json.dumps(_digest_cases(), sort_keys=True))
