# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant KDA class ``kda.bf16.c16`` behind ``recurrent_kda(batch_invariant=True)`` (SM100 / SM103 / SM107).

Every comparison is ``torch.equal``; nothing uses a tolerance. The reference term is the pinned chunked prefill of the
whole sequence in one call. Set ``FLASHINFER_KDA_BI_HASH_DIR`` to a directory to record the output hashes in one process
and assert them in another.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

import flashinfer.batch_invariant as bi
from flashinfer import kda_prefill_cute
from flashinfer.batch_invariant import BatchInvariantUnsupportedError
from flashinfer.batch_invariant.kda import pool as bi_pool
from flashinfer.batch_invariant.kda.spec import CHUNK, KDA_ARCHS, ROUTE
from flashinfer.kda import recurrent_kda

needs_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bi.is_supported_device(torch.device("cuda"), KDA_ARCHS)
    or not kda_prefill_cute._is_cute_dsl_kda_runtime_available(),
    reason="needs a CUDA device of SM100 / SM103 with the CuTe DSL KDA prefill",
)

D = 128
L = 4096
SEED = 1234
LB = -5.0
HEADS = [16, 32]
GATES = {
    "steep": 0.0,
    "long": -6.0,
}  # raw-gate offset; g_eff = LB * sigmoid(exp(A_log) * (g + dt_bias))
N0_LIST = [1, 15, 16, 17, 1000, 4079]
BATCH_POSITIONS = [1, 15, 16, 17, 1000, 2063]
SCATTERED = [9, 2, 5, 0, 7, 3]


def params(h, seed=SEED):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed + 7)
    return dict(
        A_log=(0.1 * torch.randn(h, device=dev, generator=gen)).float(),
        dt_bias=(0.1 * torch.randn(h, D, device=dev, generator=gen)).float(),
        lower_bound=LB,
    )


def make_seq(length, h, seed, gate_offset):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    f32 = dict(device=dev, generator=gen, dtype=torch.float32)
    return dict(
        q=torch.randn(length, h, D, **f32).to(torch.bfloat16),
        k=torch.randn(length, h, D, **f32).to(torch.bfloat16),
        v=torch.randn(length, h, D, **f32).to(torch.bfloat16),
        g=(torch.randn(length, h, D, **f32) + gate_offset).to(torch.bfloat16),
        beta=torch.randn(length, h, **f32).to(torch.bfloat16),
        s0=0.05 * torch.randn(h, D, D, **f32),
    )


def sub(s, a, b):
    return {key: (s[key][a:b] if key != "s0" else s[key]) for key in s}


def _fused(p):
    return dict(
        A_log=p["A_log"],
        dt_bias=p["dt_bias"],
        lower_bound=p["lower_bound"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
    )


def one_shot(s, p, n=None):
    """The class's term: the pinned prefill of the first ``n`` tokens in one call; ``(rows, final fp32 state)``."""
    n = s["q"].shape[0] if n is None else n
    cu = torch.tensor([0, n], dtype=torch.int32, device=s["q"].device)
    with kda_prefill_cute.force_route(ROUTE):
        out, st = recurrent_kda(
            s["q"][None, :n],
            s["k"][None, :n],
            s["v"][None, :n],
            s["g"][None, :n],
            s["beta"][None, :n],
            initial_state=s["s0"][None].clone(),
            output_final_state=True,
            cu_seqlens=cu,
            backend="cute-dsl",
            **_fused(p),
        )[:2]
    torch.cuda.synchronize()
    return out[0], st[0]


def new_pool(seqs, slots):
    """An fp32 pool of ``slots`` with each sequence's initial state in its slot."""
    h = seqs[0]["s0"].shape[0]
    pool = torch.zeros(
        len(slots) + max(slots) + 1, h, D, D, dtype=torch.float32, device="cuda"
    )
    for s, slot in zip(seqs, slots, strict=True):
        pool[slot].copy_(s["s0"])
    return pool


def bi_prefill(pieces, pool, slots, p, **kw):
    """One packed prefill call of ``pieces`` (each ``q/k/v/g/beta`` of its own length) into ``slots``."""
    dev = pool.device
    lens = [x["q"].shape[0] for x in pieces]
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64, device=dev
    )
    cat = {
        key: torch.cat([x[key] for x in pieces], 0)[None]
        for key in ("q", "k", "v", "g", "beta")
    }
    out, st = recurrent_kda(
        cat["q"],
        cat["k"],
        cat["v"],
        cat["g"],
        cat["beta"],
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=torch.tensor(slots, dtype=torch.int32, device=dev),
        batch_invariant=True,
        **_fused(p),
        **kw,
    )
    assert st is pool
    return out[0]


def bi_decode(tokens, pool, slots, p, packed=True, **kw):
    """One decode step: ``tokens`` ``q/k/v/g/beta`` of shape ``[B, ...]`` for ``slots``; returns ``[B, HV, 128]``."""
    dev = pool.device
    B = tokens["q"].shape[0]
    ids = torch.tensor(slots, dtype=torch.int32, device=dev)
    if packed:
        args = {key: tokens[key][None] for key in ("q", "k", "v", "g", "beta")}
        extra = dict(cu_seqlens=torch.arange(B + 1, dtype=torch.int64, device=dev))
    else:
        args = {key: tokens[key][:, None] for key in ("q", "k", "v", "g", "beta")}
        extra = {}
    out, _ = recurrent_kda(
        args["q"],
        args["k"],
        args["v"],
        args["g"],
        args["beta"],
        initial_state=pool,
        ssm_state_indices=ids,
        batch_invariant=True,
        **_fused(p),
        **extra,
        **kw,
    )
    return out.reshape(B, -1, D)


def token(s, pos):
    return {key: s[key][pos : pos + 1] for key in ("q", "k", "v", "g", "beta")}


def tokens(seqs, positions):
    return {
        key: torch.cat(
            [s[key][pos : pos + 1] for s, pos in zip(seqs, positions, strict=True)], 0
        )
        for key in ("q", "k", "v", "g", "beta")
    }


def decode_all(s, pool, slot, p, start, stop):
    return torch.cat(
        [bi_decode(token(s, pos), pool, [slot], p) for pos in range(start, stop)], 0
    )


def _hash(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _check_hash(name, t):
    d = os.environ.get("FLASHINFER_KDA_BI_HASH_DIR")
    if not d:
        return
    f = Path(d) / f"{name}.sha256"
    h = _hash(t)
    if f.exists():
        assert f.read_text().strip() == h, f"{name}: hash differs from the recorded run"
    else:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(h + "\n")


# -- device-free ---------------------------------------------------------------------------------------------------


def test_class_is_registered():
    assert "recurrent_kda" in bi.CLASSES
    assert bi.class_name("recurrent_kda") == "kda.bf16.c16"
    fp = bi.fingerprint("recurrent_kda")
    assert fp.startswith("kda-bf16-c16-engine-") and fp == bi.fingerprint(
        "recurrent_kda"
    )


def test_unsupported_device_raises_before_any_kernel():
    h = 16
    x = torch.zeros(1, 4, h, D, dtype=torch.bfloat16)
    with pytest.raises(BatchInvariantUnsupportedError):
        recurrent_kda(
            x,
            x,
            x,
            x,
            torch.zeros(1, 4, h, dtype=torch.bfloat16),
            A_log=torch.zeros(h),
            dt_bias=torch.zeros(h, D),
            lower_bound=LB,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            beta_is_logit=True,
            batch_invariant=True,
        )


# -- the class -------------------------------------------------------------------------------------------------------


@needs_device
@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("n0", N0_LIST)
def test_prefill_then_decode_equals_one_shot(h, gate, n0):
    p = params(h)
    s = make_seq(L, h, SEED + n0, GATES[gate])
    ref, ref_state = one_shot(s, p)
    pool = new_pool([s], [0])
    rows0 = bi_prefill([sub(s, 0, n0)], pool, [0], p)
    rows = decode_all(s, pool, 0, p, n0, L)
    got = torch.cat([rows0, rows], 0)
    assert torch.equal(got, ref)
    assert torch.equal(
        pool[0], ref_state
    )  # L is a multiple of 16: the slot holds the final state
    _check_hash(f"prefill_then_decode_h{h}_{gate}_n{n0}", got)


@needs_device
@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEADS)
def test_packed_requests_at_mixed_positions(h, gate):
    p = params(h)
    seqs = [
        make_seq(2200, h, SEED + 100 + i, GATES[gate])
        for i in range(len(BATCH_POSITIONS))
    ]
    refs = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, SCATTERED)
    pre = bi_prefill(
        [sub(s, 0, n0) for s, n0 in zip(seqs, BATCH_POSITIONS, strict=True)],
        pool,
        SCATTERED,
        p,
    )
    off = 0
    for n0, ref in zip(BATCH_POSITIONS, refs, strict=True):
        assert torch.equal(pre[off : off + n0], ref[:n0])
        off += n0
    steps = 40
    got = []
    for t in range(steps):
        got.append(
            bi_decode(
                tokens(seqs, [n0 + t for n0 in BATCH_POSITIONS]),
                pool,
                SCATTERED,
                p,
                packed=(t % 2 == 0),
            )
        )
    got = torch.stack(got, 1)  # [B, steps, HV, D]
    for i, (n0, ref) in enumerate(zip(BATCH_POSITIONS, refs, strict=True)):
        assert torch.equal(got[i], ref[n0 : n0 + steps]), (
            f"request {i} at position {n0}"
        )
    # the same requests decoded alone give the same bits, and a slot's committed state is the prefix prefill's
    for i, (s, n0) in enumerate(zip(seqs, BATCH_POSITIONS, strict=True)):
        alone = new_pool([s], [0])
        bi_prefill([sub(s, 0, n0)], alone, [0], p)
        rows = decode_all(s, alone, 0, p, n0, n0 + steps)
        assert torch.equal(rows, got[i])
        b = CHUNK * ((n0 + steps) // CHUNK)
        assert torch.equal(pool[SCATTERED[i]], one_shot(s, p, b)[1])
    _check_hash(f"packed_mixed_h{h}_{gate}", got)


@needs_device
def test_chunked_prefill_cut_at_chunk_multiples_equals_one_shot():
    h = 16
    p = params(h)
    s = make_seq(3000, h, SEED + 9, GATES["steep"])
    ref, _ = one_shot(s, p)
    pool = new_pool([s], [0])
    cuts = [
        0,
        1024,
        2064,
        2071,
    ]  # cuts at multiples of 16, the last piece off a boundary
    rows = [
        bi_prefill([sub(s, a, b)], pool, [0], p)
        for a, b in zip(cuts, cuts[1:], strict=False)
    ]
    rows.append(decode_all(s, pool, 0, p, 2071, 3000))
    assert torch.equal(torch.cat(rows, 0), ref)


@needs_device
def test_fixed_layout_prefill_and_output_buffer():
    h = 16
    p = params(h)
    seqs = [make_seq(300, h, SEED + 20 + i, GATES["long"]) for i in range(3)]
    refs = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, [0, 1, 2])
    cat = {
        key: torch.stack([s[key][:200] for s in seqs], 0)
        for key in ("q", "k", "v", "g", "beta")
    }
    out = torch.empty_like(cat["v"])
    got, st = recurrent_kda(
        cat["q"],
        cat["k"],
        cat["v"],
        cat["g"],
        cat["beta"],
        initial_state=pool,
        output_final_state=True,
        output=out,
        batch_invariant=True,
        **_fused(p),
    )
    assert got is out and st is pool
    for i, ref in enumerate(refs):
        assert torch.equal(got[i], ref[:200])
    tok = tokens(seqs, [200, 200, 200])
    out1 = torch.empty(3, 1, h, D, dtype=torch.bfloat16, device="cuda")
    got1, st1 = recurrent_kda(
        tok["q"][:, None],
        tok["k"][:, None],
        tok["v"][:, None],
        tok["g"][:, None],
        tok["beta"][:, None],
        initial_state=pool,
        output=out1,
        batch_invariant=True,
        **_fused(p),
    )
    assert got1 is out1 and st1 is None
    for i, ref in enumerate(refs):
        assert torch.equal(got1[i, 0], ref[200])


@needs_device
@pytest.mark.parametrize("h", HEADS)
def test_graph_replay_equals_eager(h):
    p = params(h)
    seqs = [make_seq(400, h, SEED + 50 + i, GATES["steep"]) for i in range(4)]
    n0 = [7, 16, 100, 301]
    refs = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, [3, 1, 0, 2])
    bi_prefill(
        [sub(s, 0, n) for s, n in zip(seqs, n0, strict=True)], pool, [3, 1, 0, 2], p
    )
    dev = pool.device
    ids = torch.tensor([3, 1, 0, 2], dtype=torch.int32, device=dev)
    cu = torch.arange(5, dtype=torch.int64, device=dev)
    static = {key: t[None].clone() for key, t in tokens(seqs, n0).items()}

    def step():
        return recurrent_kda(
            static["q"],
            static["k"],
            static["v"],
            static["g"],
            static["beta"],
            initial_state=pool,
            ssm_state_indices=ids,
            cu_seqlens=cu,
            batch_invariant=True,
            **_fused(p),
        )[0]

    first = step()  # eager (warms the kernels and the class's buffers); token n0
    for i, ref in enumerate(refs):
        assert torch.equal(first[0, i], ref[n0[i]])
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s_), torch.cuda.graph(g, stream=s_):
        out = step()
    torch.cuda.current_stream().wait_stream(s_)
    torch.cuda.synchronize()
    for t in range(1, 40):
        tok = tokens(seqs, [n + t for n in n0])
        for key in static:
            static[key].copy_(tok[key][None])
        g.replay()
        torch.cuda.synchronize()
        for i, ref in enumerate(refs):
            assert torch.equal(out[0, i], ref[n0[i] + t]), f"step {t}, request {i}"


@needs_device
@pytest.mark.parametrize("fill", [0.0, 1e4, -3.5])
def test_cached_rows_past_the_count_are_inert(fill):
    h = 16
    p = params(h)
    s = make_seq(200, h, SEED + 3, GATES["steep"])
    ref, _ = one_shot(s, p)
    pool = new_pool([s], [0])
    bi_prefill([sub(s, 0, 37)], pool, [0], p)
    ep = bi_pool.get_pool(pool)
    for rows_buf in (
        ep.k_inv_rows,
        ep.k_inv_f32_rows,
        ep.rhs_rows,
        ep.update_rows,
        ep.l_rows,
    ):
        rows_buf[0, :, 5:].fill_(fill)  # cached rows past the live count (37 % 16 = 5)
    rows = decode_all(s, pool, 0, p, 37, 80)
    assert torch.equal(rows, ref[37:80])


@needs_device
@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("length", [1, 16, 50, 1023, 4096])
def test_decode_lengths_from_a_nonzero_state(h, gate, length):
    """Decode alone from a loaded state, across every chunk phase and boundary: rows and the committed boundary state."""
    p = params(h)
    # a one-token reference call would take the operator's decode kernel: the reference always has a second token
    s = make_seq(length + 1, h, SEED + 200 + length, GATES[gate])
    ref, _ = one_shot(s, p)
    pool = new_pool([s], [0])
    rows = decode_all(s, pool, 0, p, 0, length)
    assert torch.equal(rows, ref[:length])
    b = 16 * (length // 16)
    if b:
        _, st = one_shot(sub(s, 0, b), p)
        assert torch.equal(pool[0], st)
    else:
        assert torch.equal(pool[0], s["s0"])
    _check_hash(f"decode_len{length}_h{h}_{gate}", rows)


@needs_device
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("T", [2, 5, 16, 23, 31, 32, 40])
def test_multi_token_steps_equal_one_shot(h, T):
    """Several tokens appended to open chunks at mixed phases, then single-token decode again: fewer than 32 tokens
    in one launch of the step kernel (which commits the chunks it completes), 32 and more with the step kernel to the
    boundary, the prefill kernel over the whole chunks and the step kernel for the tail."""
    from flashinfer.batch_invariant.kda.api import recurrent_kda_batch_invariant_extend

    p = params(h)
    n0 = [0, 3, 15, 16, 21, 37]
    slots = [4, 0, 5, 2, 1, 3]
    seqs = [
        make_seq(160, h, SEED + 300 + i, GATES["long" if i % 2 else "steep"])
        for i in range(len(n0))
    ]
    refs = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, slots)
    bi_prefill(
        [sub(s, 0, n) for s, n in zip(seqs, n0, strict=True) if n],
        pool,
        [sl for sl, n in zip(slots, n0, strict=True) if n],
        p,
    )
    ids = torch.tensor(slots, dtype=torch.int32, device="cuda")
    pos = list(n0)
    for _ in range(2):
        stack = lambda key: torch.stack(
            [s[key][q0 : q0 + T] for s, q0 in zip(seqs, pos, strict=True)], 0
        )  # noqa: E731
        out = recurrent_kda_batch_invariant_extend(
            stack("q"),
            stack("k"),
            stack("v"),
            stack("g"),
            stack("beta"),
            A_log=p["A_log"],
            dt_bias=p["dt_bias"],
            initial_state=pool,
            lower_bound=LB,
            ssm_state_indices=ids,
        )
        for i, ref in enumerate(refs):
            assert torch.equal(out[i], ref[pos[i] : pos[i] + T]), (
                f"request {i} at {pos[i]}"
            )
        pos = [q0 + T for q0 in pos]
        rows = bi_decode(tokens(seqs, pos), pool, slots, p)
        for i, ref in enumerate(refs):
            assert torch.equal(rows[i], ref[pos[i]]), f"request {i} decode at {pos[i]}"
        pos = [q0 + 1 for q0 in pos]


@needs_device
def test_multi_token_step_replayed_as_a_graph():
    """A multi-token step on the default slots reads nothing on the host: captured once, replayed across chunk
    boundaries, its rows are the one-shot prefill's."""
    from flashinfer.batch_invariant.kda.api import recurrent_kda_batch_invariant_extend

    h, T = 16, 3
    p = params(h)
    seqs = [make_seq(120, h, SEED + 700 + i, GATES["long"]) for i in range(3)]
    n0 = [5, 16, 30]
    refs = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, [0, 1, 2])
    bi_prefill(
        [sub(s, 0, n) for s, n in zip(seqs, n0, strict=True)], pool, [0, 1, 2], p
    )
    grab = lambda pos: {
        key: torch.stack(
            [s[key][q0 : q0 + T] for s, q0 in zip(seqs, pos, strict=True)], 0
        )
        for key in ("q", "k", "v", "g", "beta")
    }  # noqa: E731
    static = {key: t.clone() for key, t in grab(n0).items()}

    def step():
        return recurrent_kda_batch_invariant_extend(
            static["q"],
            static["k"],
            static["v"],
            static["g"],
            static["beta"],
            A_log=p["A_log"],
            dt_bias=p["dt_bias"],
            initial_state=pool,
            lower_bound=LB,
        )

    first = step()
    for i, ref in enumerate(refs):
        assert torch.equal(first[i], ref[n0[i] : n0[i] + T])
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s_), torch.cuda.graph(g, stream=s_):
        out = step()
    torch.cuda.current_stream().wait_stream(s_)
    torch.cuda.synchronize()
    for r in range(1, 12):
        pos = [n + r * T for n in n0]
        for key, t in grab(pos).items():
            static[key].copy_(t)
        g.replay()
        torch.cuda.synchronize()
        for i, ref in enumerate(refs):
            assert torch.equal(out[i], ref[pos[i] : pos[i] + T]), (
                f"round {r}, request {i}"
            )


@needs_device
def test_multi_token_steps_after_a_replaced_state():
    """A slot with open rows whose state the caller replaced: the multi-token append starts the new request at 0."""
    from flashinfer.batch_invariant.kda.api import recurrent_kda_batch_invariant_extend

    h = 16
    p = params(h)
    old = make_seq(40, h, SEED + 80, GATES["steep"])
    new = make_seq(64, h, SEED + 81, GATES["long"])
    ref_new, _ = one_shot(new, p)
    pool = new_pool([old], [0])
    bi_prefill([sub(old, 0, 21)], pool, [0], p)  # 5 open rows
    pool[0].copy_(new["s0"])
    out = recurrent_kda_batch_invariant_extend(
        *(new[key][None, :30] for key in ("q", "k", "v", "g", "beta")),
        A_log=p["A_log"],
        dt_bias=p["dt_bias"],
        initial_state=pool,
        lower_bound=LB,
    )
    assert torch.equal(out[0], ref_new[:30])
    rows = decode_all(new, pool, 0, p, 30, 50)
    assert torch.equal(rows, ref_new[30:50])


@needs_device
def test_explicit_reset_of_a_recycled_slot():
    """A slot recycled after a request shorter than 16 tokens (its boundary state never moved) is reset explicitly."""
    from flashinfer.batch_invariant.kda.api import recurrent_kda_batch_invariant_reset

    h = 16
    p = params(h)
    old = make_seq(9, h, SEED + 82, GATES["steep"])
    new = make_seq(40, h, SEED + 83, GATES["steep"])
    new["s0"] = old[
        "s0"
    ]  # the same loaded state: the stamp cannot tell the requests apart
    ref_new, _ = one_shot(new, p)
    pool = new_pool([old], [0])
    decode_all(old, pool, 0, p, 0, 9)
    recurrent_kda_batch_invariant_reset(
        pool, torch.zeros(1, dtype=torch.int32, device="cuda")
    )
    rows = decode_all(new, pool, 0, p, 0, 40)
    assert torch.equal(rows, ref_new)


@needs_device
def test_pool_aliases_share_one_state():
    """A ``detach()`` alias of the pool on every call (no view relation) keeps the slot's open chunk."""
    h = 16
    p = params(h)
    s = make_seq(60, h, SEED + 84, GATES["long"])
    ref, _ = one_shot(s, p)
    pool = new_pool([s], [0])
    rows = torch.cat(
        [bi_decode(token(s, pos), pool.detach(), [0], p) for pos in range(40)], 0
    )
    assert torch.equal(rows, ref[:40])
    assert (
        len(
            {
                bi_pool.pool_key(pool),
                bi_pool.pool_key(pool.detach()),
                bi_pool.pool_key(pool[:]),
            }
        )
        == 1
    )


@needs_device
def test_first_use_inside_a_graph_capture_is_refused():
    h = 16
    p = params(h)
    s = make_seq(4, h, SEED + 85, GATES["steep"])
    pool = new_pool([s], [0])
    args = [s[key][None, :1] for key in ("q", "k", "v", "g", "beta")]
    ids = torch.zeros(1, dtype=torch.int32, device="cuda")
    cu = torch.arange(2, dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph()
    with (
        pytest.raises(BatchInvariantUnsupportedError, match="capture"),
        torch.cuda.graph(g),
    ):
        recurrent_kda(
            *args,
            initial_state=pool,
            ssm_state_indices=ids,
            cu_seqlens=cu,
            batch_invariant=True,
            **_fused(p),
        )


@needs_device
@pytest.mark.parametrize("slots", [[0, 0], [0, 7], [-1, 0]])
def test_prefill_refuses_bad_slots(slots):
    h = 16
    p = params(h)
    seqs = [make_seq(20, h, SEED + 86 + i, GATES["steep"]) for i in range(2)]
    pool = torch.zeros(4, h, D, D, device="cuda")
    with pytest.raises(BatchInvariantUnsupportedError, match="slot"):
        bi_prefill(seqs, pool, slots, p)


@needs_device
def test_other_lower_bound_and_head_count():
    """A gate lower bound that is not a dyadic number, on 8 heads: prefill, tail and decode against the one-shot."""
    h = 8
    p = dict(params(h), lower_bound=-3.7)
    s = make_seq(70, h, SEED + 90, GATES["long"])
    ref, _ = one_shot(s, p)
    pool = new_pool([s], [0])
    pre = bi_prefill([sub(s, 0, 37)], pool, [0], p)
    rows = decode_all(s, pool, 0, p, 37, 70)
    assert torch.equal(torch.cat([pre, rows], 0), ref)


@needs_device
def test_slot_reuse_with_a_replaced_state():
    """A slot whose state the caller replaced starts a new chunk: a new request decoded from its loaded state, and a new
    request prefilled over a slot that had a partial chunk."""
    h = 16
    p = params(h)
    old = make_seq(60, h, SEED + 70, GATES["steep"])
    new = make_seq(60, h, SEED + 71, GATES["long"])
    ref_new, _ = one_shot(new, p)
    pool = new_pool([old], [0])
    bi_prefill([sub(old, 0, 21)], pool, [0], p)
    decode_all(old, pool, 0, p, 21, 27)  # slot 0 holds 11 history rows
    pool[0].copy_(new["s0"])  # the caller loads another request's state
    rows = decode_all(new, pool, 0, p, 0, 40)
    assert torch.equal(rows, ref_new[:40])
    pool[0].copy_(new["s0"])
    pre = bi_prefill([sub(new, 0, 23)], pool, [0], p)
    rows = decode_all(new, pool, 0, p, 23, 60)
    assert torch.equal(torch.cat([pre, rows], 0), ref_new)


@needs_device
def test_negative_controls():
    """The operator's unpinned route (the two-kernel decomposition at a small sequence count) and its recurrent
    decode kernel are other terms; the class gives the pinned prefill's row."""
    h = 16
    p = params(h)
    s = make_seq(L, h, SEED + 5, GATES["steep"])
    ref, _ = one_shot(s, p)
    cu = torch.tensor([0, L], dtype=torch.int32, device="cuda")
    call = dict(
        initial_state=s["s0"][None].clone(),
        output_final_state=True,
        cu_seqlens=cu,
        backend="cute-dsl",
        **_fused(p),
    )
    seq = (s["q"][None], s["k"][None], s["v"][None], s["g"][None], s["beta"][None])
    with kda_prefill_cute.force_route("decomp"):
        other = recurrent_kda(*seq, **call)[0][0]
    today = recurrent_kda(
        s["q"][None, :1],
        s["k"][None, :1],
        s["v"][None, :1],
        s["g"][None, :1],
        s["beta"][None, :1],
        initial_state=s["s0"][None].clone(),
        output_final_state=True,
        **_fused(p),
    )[0][0, 0]
    if torch.equal(other, ref) or torch.equal(today, ref[0]):
        pytest.skip(
            "the other terms agree with the pinned prefill on this input; the control is void here"
        )
    unpinned = recurrent_kda(*seq, **{**call, "initial_state": s["s0"][None].clone()})[
        0
    ][0]
    assert torch.equal(
        unpinned, other
    )  # one sequence at 16 heads takes the decomposition on these parts
    pool = new_pool([s], [0])
    assert torch.equal(bi_decode(token(s, 0), pool, [0], p)[0], ref[0])


@needs_device
def test_pool_passed_as_a_fresh_view_each_call():
    """A framework hands the class a new view of its long-lived buffer on every call; the class's state survives."""
    h = 16
    p = params(h)
    s = make_seq(120, h, SEED + 80, GATES["steep"])
    ref, _ = one_shot(s, p)
    big = torch.zeros(2, 4, h, D, D, dtype=torch.float32, device="cuda")
    big[1, 2].copy_(s["s0"])
    bi_prefill([sub(s, 0, 21)], big[1], [2], p)
    rows = [bi_decode(token(s, pos), big[1], [2], p) for pos in range(21, 120)]
    assert torch.equal(torch.cat(rows, 0), ref[21:])


@needs_device
def test_unsupported_inputs_raise_and_nothing_runs():
    h = 16
    p = params(h)
    s = make_seq(8, h, SEED + 1, GATES["steep"])
    pool = new_pool([s], [0])
    before = pool.clone()
    base = dict(initial_state=pool, batch_invariant=True)
    tok = token(s, 0)
    args = (
        tok["q"][None],
        tok["k"][None],
        tok["v"][None],
        tok["g"][None],
        tok["beta"][None],
    )
    cu = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    bad = [
        dict(backend="cute-dsl"),
        dict(num_spec_tokens=1),
        dict(lower_bound=None),
        dict(lower_bound=0.0),
        dict(use_gate_in_kernel=False),
        dict(beta_is_logit=False),
        dict(A_log=None),
        dict(initial_state=pool.to(torch.bfloat16)),
        dict(ssm_state_indices=torch.zeros(1, dtype=torch.int64, device="cuda")),
        dict(seq_order=torch.zeros(1, dtype=torch.int32, device="cuda")),
        dict(disable_state_update=True),
    ]
    for kw in bad:
        full = {**_fused(p), **base, "cu_seqlens": cu, **kw}
        with pytest.raises(BatchInvariantUnsupportedError):
            recurrent_kda(*args, **full)
    g16 = make_seq(8, 16, SEED + 2, 0.0)
    with pytest.raises(BatchInvariantUnsupportedError):  # GQA
        recurrent_kda(
            torch.cat([tok["q"], tok["q"]], 1)[None],
            torch.cat([tok["k"], tok["k"]], 1)[None],
            tok["v"][None],
            tok["g"][None],
            tok["beta"][None],
            cu_seqlens=cu,
            **_fused(p),
            **base,
        )
    with pytest.raises(BatchInvariantUnsupportedError):  # 12 heads
        t12 = {key: g16[key][:1, :12] for key in ("q", "k", "v", "g", "beta")}
        recurrent_kda(
            t12["q"][None],
            t12["k"][None],
            t12["v"][None],
            t12["g"][None],
            t12["beta"][None],
            initial_state=pool[:, :12].contiguous(),
            batch_invariant=True,
            cu_seqlens=cu,
            A_log=p["A_log"][:12],
            dt_bias=p["dt_bias"][:12],
            lower_bound=LB,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            beta_is_logit=True,
        )
    with pytest.raises(BatchInvariantUnsupportedError):  # decode without a pool
        recurrent_kda(*args, cu_seqlens=cu, batch_invariant=True, **_fused(p))
    with pytest.raises(BatchInvariantUnsupportedError):  # fp16 activations
        recurrent_kda(
            *(a.to(torch.float16) for a in args), cu_seqlens=cu, **_fused(p), **base
        )
    assert torch.equal(pool, before)
    assert (
        bi_pool.pool_key(pool) not in bi_pool._POOLS
        or bi_pool._POOLS[bi_pool.pool_key(pool)][0].cnt.sum().item() == 0
    )
