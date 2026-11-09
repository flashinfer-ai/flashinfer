"""Bitwise tests for the pooled serving form of the exact KDA decode (SM100/SM103 only).

The pool must return, step for step, the rows of the per-request form (``kda_exact_decode``) and of the one-shot
pinned prefill, with contiguous and scattered slots, through eager steps and through a captured CUDA graph, and
with arbitrary finite data in the padding rows of a slot's 16-row history.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flashinfer.kda_exact_decode import (
    CHUNK,
    KdaExactDecodePool,
    KdaLayerParams,
    kda_exact_decode,
    kda_exact_prefill,
)

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "_kda_replay_test", Path(__file__).with_name("test_exact_decode_replay.py")
)
_rep = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rep)
BATCH_POSITIONS, GATES, HEAD_CONFIGS, L, SEED = (
    _rep.BATCH_POSITIONS,
    _rep.GATES,
    _rep.HEAD_CONFIGS,
    _rep.L,
    _rep.SEED,
)
make_params, make_seq, one_shot = _rep.make_params, _rep.make_seq, _rep.one_shot
_skip_unless_supported, _check_hash = _rep._skip_unless_supported, _rep._check_hash


def _prefilled(seqs, params):
    return [
        kda_exact_prefill(
            s["q"][:p0],
            s["k"][:p0],
            s["v"][:p0],
            s["g"][:p0],
            s["beta"][:p0],
            s["s0"].clone(),
            params,
        )[1]
        for s, p0 in zip(seqs, BATCH_POSITIONS, strict=True)
    ]


def _tokens(seqs, positions):
    return tuple(
        torch.stack([s[key][p] for s, p in zip(seqs, positions, strict=True)])
        for key in ("q", "k", "v", "g", "beta")
    )


SCATTERED = [9, 2, 5, 0, 7, 3]


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEAD_CONFIGS)
def test_pool_matches_per_request_and_one_shot(h, gate):
    _skip_unless_supported()
    params = make_params(h)
    n = len(BATCH_POSITIONS)
    seqs = [make_seq(L, h, SEED + 100 + i, GATES[gate]) for i in range(n)]
    refs = [one_shot(s, params)[0] for s in seqs]
    v0 = _prefilled(seqs, params)
    dev = torch.device("cuda")
    pool_c = KdaExactDecodePool(n, h, h, params, dev)
    pool_s = KdaExactDecodePool(12, h, h, params, dev)
    for i, st in enumerate(_prefilled(seqs, params)):
        pool_c.load(i, st)
    for i, st in enumerate(_prefilled(seqs, params)):
        pool_s.load(SCATTERED[i], st)
    ids_c = torch.arange(n, dtype=torch.int32, device=dev)
    ids_s = torch.tensor(SCATTERED, dtype=torch.int32, device=dev)
    steps = 2 * CHUNK + 3
    got = []
    for _ in range(steps):
        pos = [st.position for st in v0]
        q_t, k_t, v_t, g_t, b_t = _tokens(seqs, pos)
        r0 = kda_exact_decode(q_t, k_t, v_t, g_t, b_t, v0, params)
        rc = pool_c.step(ids_c, q_t, k_t, v_t, g_t, b_t, contiguous=True)
        rs = pool_s.step(ids_s, q_t, k_t, v_t, g_t, b_t, contiguous=False)
        pool_c.advance_positions(range(n))
        pool_s.advance_positions(SCATTERED)
        torch.cuda.synchronize()
        assert torch.equal(rc, r0), "contiguous pool differs from the per-request form"
        assert torch.equal(rs, r0), "scattered pool differs from the per-request form"
        got.append(r0)
    for i, p0 in enumerate(BATCH_POSITIONS):
        assert torch.equal(torch.stack([g_[i] for g_ in got]), refs[i][p0 : p0 + steps])
        assert torch.equal(pool_c.current_state(i), v0[i].current_state)
        assert torch.equal(pool_s.current_state(SCATTERED[i]), v0[i].current_state)
        assert pool_c.position[i] == v0[i].position
    # export -> load round trip, then one scattered step at B = 1
    st = pool_s.export(SCATTERED[2])
    assert torch.equal(st.current_state, v0[2].current_state)
    pool_c.load(0, st)
    pos = v0[2].position
    q_t, k_t, v_t, g_t, b_t = _tokens(seqs[2:3], [pos])
    r_single = pool_c.step(
        torch.zeros(1, dtype=torch.int32, device=dev), q_t, k_t, v_t, g_t, b_t
    )
    r_scattered = pool_s.step(
        torch.tensor([SCATTERED[2]], dtype=torch.int32, device=dev),
        q_t,
        k_t,
        v_t,
        g_t,
        b_t,
        contiguous=False,
    )
    torch.cuda.synchronize()
    assert torch.equal(r_single, refs[2][pos : pos + 1])
    assert torch.equal(r_scattered, refs[2][pos : pos + 1])
    _check_hash(f"pool_h{h}_{gate}", torch.stack(got))


@pytest.mark.parametrize("h", HEAD_CONFIGS)
def test_pool_graph_matches_eager(h):
    _skip_unless_supported()
    params = make_params(h)
    n = len(BATCH_POSITIONS)
    seqs = [make_seq(L, h, SEED + 300 + i, GATES["long"]) for i in range(n)]
    dev = torch.device("cuda")
    eager = KdaExactDecodePool(n, h, h, params, dev)
    graphed = KdaExactDecodePool(n, h, h, params, dev)
    for i, st in enumerate(_prefilled(seqs, params)):
        eager.load(i, st)
    for i, st in enumerate(_prefilled(seqs, params)):
        graphed.load(i, st)
    ids = torch.arange(n, dtype=torch.int32, device=dev)
    pos = list(BATCH_POSITIONS)
    bufs = [t.clone() for t in _tokens(seqs, pos)]
    g = graphed.capture_step(ids, *bufs)
    steps = 2 * CHUNK + 3
    for _ in range(steps):
        toks = _tokens(seqs, pos)
        for b, t in zip(bufs, toks, strict=True):
            b.copy_(t)
        g.replay()
        r_eager = eager.step(ids, *toks)
        torch.cuda.synchronize()
        assert torch.equal(g.rows, r_eager)
        pos = [p + 1 for p in pos]
    assert torch.equal(graphed.count, eager.count)
    assert torch.equal(graphed.state[:n], eager.state[:n])


@pytest.mark.parametrize("fill", [0.0, 1e4, -3.5])
def test_pool_padding_rows_are_inert(fill):
    """Rows past a slot's live count never reach the returned rows or the committed boundary state."""
    _skip_unless_supported()
    h = 16
    params = make_params(h)
    n = len(BATCH_POSITIONS)
    seqs = [make_seq(L, h, SEED + 500 + i, GATES["long"]) for i in range(n)]
    dev = torch.device("cuda")
    clean = KdaExactDecodePool(n, h, h, params, dev)
    dirty = KdaExactDecodePool(n, h, h, params, dev)
    for buf in (
        dirty.hist_q,
        dirty.hist_k,
        dirty.hist_v,
        dirty.hist_g,
        dirty.hist_beta,
    ):
        buf.fill_(fill)
    for i, st in enumerate(_prefilled(seqs, params)):
        clean.load(i, st)
    for i, st in enumerate(_prefilled(seqs, params)):
        dirty.load(i, st)
    ids = torch.arange(n, dtype=torch.int32, device=dev)
    pos = list(BATCH_POSITIONS)
    for _ in range(CHUNK + 2):
        toks = _tokens(seqs, pos)
        a = clean.step(ids, *toks)
        b = dirty.step(ids, *toks)
        torch.cuda.synchronize()
        assert torch.equal(a, b)
        pos = [p + 1 for p in pos]
    assert torch.equal(clean.state[:n], dirty.state[:n])


def test_pool_rejects_unsupported_contracts():
    _skip_unless_supported()
    h = 16
    params = make_params(h)
    dev = torch.device("cuda")
    with pytest.raises(NotImplementedError):  # GQA
        KdaExactDecodePool(2, 32, h, params, dev)
    p12 = KdaLayerParams(
        A_log=params.A_log[:12],
        dt_bias=params.dt_bias[:12],
        lower_bound=params.lower_bound,
    )
    with pytest.raises(NotImplementedError):  # non-power-of-two head count
        KdaExactDecodePool(2, 12, 12, p12, dev)
    pool = KdaExactDecodePool(2, h, h, params, dev)
    s = make_seq(4, h, SEED + 17, GATES["long"])
    q_t, k_t, v_t, g_t, b_t = (s[key][:1] for key in ("q", "k", "v", "g", "beta"))
    ids = torch.zeros(1, dtype=torch.int32, device=dev)
    with pytest.raises(ValueError):
        pool.step(ids[:0], q_t[:0], k_t[:0], v_t[:0], g_t[:0], b_t[:0])
    s32 = make_seq(4, 32, SEED + 17, GATES["long"])
    with pytest.raises(ValueError):  # token heads differ from the pool's
        pool.step(ids, *(s32[key][:1] for key in ("q", "k", "v", "g", "beta")))
    with pytest.raises(ValueError):
        pool.step(ids.long(), q_t, k_t, v_t, g_t, b_t)
    pool.step(ids, q_t, k_t, v_t, g_t, b_t)
    with pytest.raises(RuntimeError):  # position not advanced
        pool.current_state(0)
