# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant KDA class ``kda.bf16.s16`` behind ``recurrent_kda(batch_invariant=True)``.

Every comparison is ``torch.equal`` except the torch reference test. The oracle is the class
itself: the prefill of the whole sequence in one call. Inputs come from numpy on the host, so the
bits do not depend on the device. Set ``FLASHINFER_KDA_CLASS_S_HASH_DIR`` to record the hashes in
one process and assert them in another.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import flashinfer.batch_invariant as bi
from flashinfer.batch_invariant import BatchInvariantUnsupportedError
from flashinfer.batch_invariant.kda_class_s import pool as s_pool
from flashinfer.batch_invariant.kda_class_s.api import (
    recurrent_kda_class_s_extend,
    recurrent_kda_class_s_reset,
)
from flashinfer.batch_invariant.kda_class_s.reference import kda_class_s_reference
from flashinfer.batch_invariant.kda_class_s.spec import CHUNK, CLASS, KDA_ARCHS
from flashinfer.batch_invariant.support import compute_capability
from flashinfer.kda import recurrent_kda


def _device_ok():
    if not torch.cuda.is_available():
        return False
    return compute_capability(torch.device("cuda")) in KDA_ARCHS


pytestmark = pytest.mark.skipif(
    not _device_ok(), reason="needs a CUDA device of SM100 / SM103 / SM107"
)

DEV = "cuda"
D = 128
SEED = 1234
LB = -5.0
HEADS = [4, 16]
GATES = {"steep": (-6.0, 0.0), "long": (-12.0, -4.0)}
KEYS = ("q", "k", "v", "g", "beta")
RESULT_KEYS = ("rows", "state", "n", "key", "u", "prefix")
POISONS = {"nan": float("nan"), "inf": float("inf"), "big": 3e4}


def _bf16(a):
    return torch.from_numpy(a).to(torch.bfloat16).to(DEV)


def params(h, lower_bound=LB, scale=None):
    rng = np.random.default_rng(SEED + 1000 + h)
    a_log = rng.standard_normal(h, dtype=np.float32) * np.float32(0.3)
    dt_bias = rng.standard_normal((h, D), dtype=np.float32) * np.float32(0.5)
    return dict(
        A_log=torch.from_numpy(a_log).to(DEV),
        dt_bias=torch.from_numpy(dt_bias).to(DEV),
        lower_bound=lower_bound,
        scale=scale,
    )


def make_seq(length, h, seed, gate):
    """One sequence: ``q/k/v/g`` ``[length, h, 128]``, ``beta`` ``[length, h]``, state ``s0``."""
    g_off, b_off = GATES[gate]
    rng = np.random.default_rng(seed)

    def rn(*shape):
        return rng.standard_normal(shape, dtype=np.float32)

    return dict(
        q=_bf16(rn(length, h, D)),
        k=_bf16(rn(length, h, D)),
        v=_bf16(rn(length, h, D)),
        g=_bf16(rn(length, h, D) + np.float32(g_off)),
        beta=_bf16(rn(length, h) + np.float32(b_off)),
        s0=_bf16(rn(h, D, D) * np.float32(0.1)),
    )


def sub(s, a, b):
    return {key: (s[key][a:b] if key != "s0" else s[key]) for key in s}


def batch(seqs, a=0, b=None):
    return {key: torch.stack([s[key][a:b] for s in seqs], 0) for key in KEYS}


def cut(x, a, b):
    return {key: x[key][:, a:b] for key in KEYS}


def tokens(seqs, positions):
    return {
        key: torch.stack(
            [s[key][pos : pos + 1] for s, pos in zip(seqs, positions, strict=True)], 0
        )
        for key in KEYS
    }


def ids(slots):
    return torch.tensor(slots, dtype=torch.int32, device=DEV)


def _kw(p):
    return dict(
        A_log=p["A_log"],
        dt_bias=p["dt_bias"],
        lower_bound=p["lower_bound"],
        scale=p["scale"],
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        batch_invariant=True,
    )


def new_pool(seqs, slots, size=None):
    """A bf16 pool with each sequence's initial state in its slot; the last slot is a spare."""
    h = seqs[0]["s0"].shape[0]
    size = max(slots) + 2 if size is None else size
    pool = torch.zeros(size, h, D, D, dtype=torch.bfloat16, device=DEV)
    for s, slot in zip(seqs, slots, strict=True):
        pool[slot].copy_(s["s0"])
    return pool


def kda(x, pool, p, **kw):
    """``recurrent_kda`` on ``[B, T, ...]`` inputs: decode for ``T == 1``, else a fresh prefill."""
    return recurrent_kda(*(x[key] for key in KEYS), initial_state=pool, **_kw(p), **kw)[
        0
    ]


def ext(x, pool, p, **kw):
    return recurrent_kda_class_s_extend(
        *(x[key] for key in KEYS),
        A_log=p["A_log"],
        dt_bias=p["dt_bias"],
        initial_state=pool,
        lower_bound=p["lower_bound"],
        scale=p["scale"],
        **kw,
    )


def prefill(pieces, pool, slots, p, companion=False, **kw):
    """One packed call of ragged ``pieces`` into ``slots``; returns the rows of each piece.

    ``companion`` adds a 2-token sequence in the spare slot so that one-token pieces take the
    prefill kernel too."""
    pieces, slots = list(pieces), list(slots)
    n = len(pieces)
    if companion:
        spare = pool.shape[0] - 1
        assert spare not in slots
        pieces.append(make_seq(2, pool.shape[1], SEED + 4242, "steep"))
        slots.append(spare)
    lens = [x["q"].shape[0] for x in pieces]
    cu = torch.tensor(np.cumsum([0] + lens), dtype=torch.int32, device=DEV)
    cat = {key: torch.cat([x[key] for x in pieces], 0)[None] for key in KEYS}
    out, st = recurrent_kda(
        *(cat[key] for key in KEYS),
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=ids(slots),
        **_kw(p),
        **kw,
    )
    assert st is pool
    return list(torch.split(out[0], lens, 0))[:n]


def decode_all(s, pool, slot, p, start, stop):
    return torch.cat(
        [
            kda(tokens([s], [pos]), pool, p, ssm_state_indices=ids([slot]))[0]
            for pos in range(start, stop)
        ],
        0,
    )


def one_shot(s, p, n=None):
    """The oracle: rows and final state of the first ``n`` tokens prefilled in one call."""
    n = s["q"].shape[0] if n is None else n
    pool = new_pool([s], [0])
    rows = prefill([sub(s, 0, n)], pool, [0], p, companion=(n == 1))[0]
    return rows, pool[0].clone()


def side(pool, slots):
    """The open chunk of each slot through the pool's accessor: row count and row bits."""
    sp = s_pool.get_pool(pool)
    out = {key: [] for key in RESULT_KEYS[2:]}
    for slot in slots:
        n, key_rows, u_rows, prefix_rows = sp.open_chunk(slot)
        h = pool.shape[1]
        for t in (key_rows, u_rows, prefix_rows):
            assert tuple(t.shape) == (h, n, D)
        out["n"].append(torch.tensor([n], dtype=torch.int32))
        out["key"].append(key_rows.contiguous().view(torch.int16))
        out["u"].append(u_rows.contiguous().view(torch.int32))
        out["prefix"].append(prefix_rows.contiguous().view(torch.int32))
    return out


def open_rows(pool, slot):
    return s_pool.get_pool(pool).open_chunk(slot)[0]


def result(rows, pool, slots):
    return dict(
        rows=list(rows), state=pool[ids(slots).long()].clone(), **side(pool, slots)
    )


def assert_same(a, b, what, keep=None):
    assert len(a["rows"]) == len(b["rows"])
    for i in range(len(a["rows"])) if keep is None else keep:
        for key in RESULT_KEYS:
            assert torch.equal(a[key][i], b[key][i]), f"{what}: {key} of request {i}"


def _bits(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    elif t.dtype == torch.float32:
        t = t.view(torch.int32)
    return t.numpy().tobytes()


def _check_hash(name, t):
    d = os.environ.get("FLASHINFER_KDA_CLASS_S_HASH_DIR")
    if not d:
        return
    f = Path(d) / name
    h = hashlib.sha256(_bits(t)).hexdigest()
    if f.exists():
        assert f.read_text().strip() == h, f"{name}: hash differs from the recorded run"
    else:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(h + "\n")


def _check_result_hash(name, res):
    _check_hash(f"{name}_rows", torch.cat([r.reshape(-1) for r in res["rows"]]))
    for key in RESULT_KEYS[1:]:
        _check_hash(f"{name}_{key}", torch.cat([t.reshape(-1).cpu() for t in res[key]]))


def _groups(n, sizes=(1, 2, 3, 8, 31)):
    out, i, j = [], 0, 0
    while i < n:
        m = min(sizes[j % len(sizes)], n - i)
        out.append(list(range(i, i + m)))
        i, j = i + m, j + 1
    return out


def _all_ways(
    seqs, p, cuts_any=(5, 23, 71, 300, 1001, 2049), cuts_16=(16, 48, 160, 1008, 2048)
):
    """The sequences prefilled whole, decoded token by token, and prefilled in pieces cut anywhere
    and at multiples of 16."""
    T = seqs[0]["q"].shape[0]
    slots = list(range(len(seqs)))
    x = batch(seqs)
    res = {}
    pool = new_pool(seqs, slots)
    res["whole"] = result(
        prefill(seqs, pool, slots, p, companion=(T == 1)), pool, slots
    )
    pool = new_pool(seqs, slots)
    out = torch.cat([kda(cut(x, t, t + 1), pool, p) for t in range(T)], 1)
    res["stepped"] = result(out, pool, slots)
    pool = new_pool(seqs, slots)
    edges = [0] + [c for c in cuts_any if c < T] + [T]
    parts = [
        kda(cut(x, a, b), pool, p) if a == 0 and b > 1 else ext(cut(x, a, b), pool, p)
        for a, b in zip(edges, edges[1:], strict=False)
    ]
    res["split"] = result(torch.cat(parts, 1), pool, slots)
    pool = new_pool(seqs, slots)
    edges = [0] + [c for c in cuts_16 if c < T] + [T]
    parts = [kda(cut(x, a, b), pool, p) for a, b in zip(edges, edges[1:], strict=False)]
    res["split16"] = result(torch.cat(parts, 1), pool, slots)
    for name in ("stepped", "split", "split16"):
        assert_same(res["whole"], res[name], name)
    assert all(int(n) == T % CHUNK for n in res["whole"]["n"])
    return res["whole"]


def test_class_is_registered():
    assert "recurrent_kda" in bi.CLASSES
    assert CLASS == "kda.bf16.s16" and bi.class_name("recurrent_kda") == CLASS
    fp = bi.fingerprint("recurrent_kda")
    assert fp.startswith("kda-bf16-s16-")
    assert fp == bi.fingerprint("recurrent_kda", kda_class="kda.bf16.s16")
    with pytest.raises(ValueError, match="kda.bf16.c16"):
        bi.fingerprint("recurrent_kda", kda_class="kda.bf16.c16")


def test_inputs_are_the_recorded_ones():
    s = make_seq(40, 4, SEED, "long")
    p = params(4)
    for key in (*KEYS, "s0"):
        _check_hash(f"inputs_{key}", s[key])
    _check_hash("inputs_A_log", p["A_log"])
    _check_hash("inputs_dt_bias", p["dt_bias"])


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("length", [1, 16, 50, 200, 1023, 4096])
def test_prefill_decode_and_split_prefill_agree(h, gate, length):
    p = params(h)
    nseq = 1 if length == 4096 else 2
    seqs = [make_seq(length, h, SEED + 31 * length + h + i, gate) for i in range(nseq)]
    whole = _all_ways(seqs, p)
    assert bool(torch.isfinite(torch.stack(whole["rows"]).float()).all())
    _check_result_hash(f"all_ways_T{length}_h{h}_{gate}", whole)


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("n0", [1, 7, 16, 17, 40])
def test_prefill_then_decode_equals_one_shot(h, gate, n0):
    p = params(h)
    T = 100
    seqs = [make_seq(T, h, SEED + 500 + 7 * n0 + h + i, gate) for i in range(2)]
    slots = [0, 1]
    pool = new_pool(seqs, slots)
    ref = result(prefill(seqs, pool, slots, p), pool, slots)
    pool = new_pool(seqs, slots)
    pre = prefill([sub(s, 0, n0) for s in seqs], pool, slots, p, companion=(n0 == 1))
    x = batch(seqs)
    dec = torch.cat([kda(cut(x, t, t + 1), pool, p) for t in range(n0, T)], 1)
    got = result(torch.cat([torch.stack(pre), dec], 1), pool, slots)
    assert_same(ref, got, f"prefill {n0} then decode")
    _check_result_hash(f"prefill_then_decode_n{n0}_h{h}_{gate}", got)


def _decode_variant(seqs, n0, p, groups, slots=None, size=None, poison=None, steps=2):
    """History prefilled into a fresh pool, then ``steps`` decode steps, one call per group."""
    n = len(seqs)
    slots = list(range(n)) if slots is None else slots
    poison = poison or {}
    pool = new_pool(seqs, slots, size)
    have = [i for i in range(n) if n0[i]]
    prefill([sub(seqs[i], 0, n0[i]) for i in have], pool, [slots[i] for i in have], p)
    rows = [[None] * steps for _ in range(n)]
    for t in range(steps):
        for grp in groups:
            tok = tokens([seqs[i] for i in grp], [n0[i] + t for i in grp])
            tok = {key: val.clone() for key, val in tok.items()}
            for j, i in enumerate(grp):
                if i in poison:
                    for key in KEYS:
                        tok[key][j] = poison[i]
            out = kda(tok, pool, p, ssm_state_indices=ids([slots[i] for i in grp]))
            for j, i in enumerate(grp):
                rows[i][t] = out[j, 0]
    return result([torch.stack(r) for r in rows], pool, slots)


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEADS)
def test_decode_batch_and_position_invariance(h, gate):
    n, steps = 64, 2
    p = params(h)
    n0 = [(i * 37) % 97 for i in range(n)]
    assert {x % CHUNK for x in n0} == set(range(CHUNK))
    seqs = [make_seq(n0[i] + steps, h, SEED + 3000 + i, gate) for i in range(n)]
    everyone = list(range(n))
    base = _decode_variant(seqs, n0, p, [everyone])
    pool = new_pool(seqs, everyone)
    whole = prefill(seqs, pool, everyone, p)
    oracle = result([r[m:] for r, m in zip(whole, n0, strict=True)], pool, everyone)
    assert_same(oracle, base, "decode after history against the whole prefill")
    rng = np.random.default_rng(5)
    order = [int(i) for i in rng.permutation(n)]
    far = [int(i) for i in rng.permutation(255)[:n]]
    variants = {
        "sub-batches": dict(groups=_groups(n)),
        "singles": dict(groups=[[i] for i in everyone]),
        "permuted": dict(groups=[order]),
        "relocated": dict(groups=[everyone], slots=far, size=256),
        "relocated permuted": dict(groups=[order], slots=far, size=256),
    }
    for name, kw in variants.items():
        assert_same(base, _decode_variant(seqs, n0, p, **kw), name)
    for name, value in POISONS.items():
        for parity in (0, 1):
            bad = {i: value for i in everyone if i % 2 == parity}
            got = _decode_variant(seqs, n0, p, [everyone], poison=bad)
            keep = [i for i in everyone if i not in bad]
            assert_same(base, got, f"{name} neighbours", keep=keep)
    _check_result_hash(f"decode_invariance_h{h}_{gate}", base)


def _prefill_variant(seqs, p, groups, slots=None, size=None, poison=None):
    """Ragged sequences prefilled into a fresh pool with one packed call per group."""
    n = len(seqs)
    slots = list(range(n)) if slots is None else slots
    poison = poison or {}
    pool = new_pool(seqs, slots, size)
    rows = [None] * n
    for grp in groups:
        pieces = []
        for i in grp:
            s = seqs[i]
            if i in poison:
                s = {key: torch.full_like(s[key], poison[i]) for key in KEYS}
            pieces.append(s)
        out = prefill(pieces, pool, [slots[i] for i in grp], p)
        for j, i in enumerate(grp):
            rows[i] = out[j]
    return result(rows, pool, slots)


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEADS)
def test_prefill_batch_and_position_invariance(h, gate):
    n = 40
    p = params(h)
    rng = np.random.default_rng(77)
    lens = [int(x) for x in rng.integers(1, 201, size=n)]
    lens[:8] = [1, 200, 16, 1, 17, 2, 1023, 515]
    seqs = [make_seq(lens[i], h, SEED + 5000 + i, gate) for i in range(n)]
    everyone = list(range(n))
    base = _prefill_variant(seqs, p, [everyone])
    for i in (0, 1, 2, 6, 9):
        rows, state = one_shot(seqs[i], p)
        assert torch.equal(rows, base["rows"][i]) and torch.equal(
            state, base["state"][i]
        )
    order = [int(i) for i in rng.permutation(n)]
    far = [int(i) for i in rng.permutation(127)[:n]]
    variants = {
        "sub-batches": dict(groups=_groups(n)),
        "sub-batches reversed": dict(groups=_groups(n, sizes=(31, 8, 1))[::-1]),
        "permuted": dict(groups=[order]),
        "relocated": dict(groups=[everyone], slots=far, size=128),
        "relocated permuted": dict(groups=[order], slots=far, size=128),
    }
    for name, kw in variants.items():
        assert_same(base, _prefill_variant(seqs, p, **kw), name)
    for name, value in POISONS.items():
        for parity in (0, 1):
            bad = {i: value for i in everyone if i % 2 == parity}
            got = _prefill_variant(seqs, p, [everyone], poison=bad)
            keep = [i for i in everyone if i not in bad]
            assert_same(base, got, f"{name} neighbours", keep=keep)
    _check_result_hash(f"prefill_invariance_h{h}_{gate}", base)


def test_repeated_calls_give_equal_bits():
    h, n = 16, 24
    p = params(h)
    n0 = [(i * 11) % 53 + 1 for i in range(n)]
    seqs = [make_seq(n0[i] + 2, h, SEED + 6000 + i, "steep") for i in range(n)]
    everyone = list(range(n))
    first = _prefill_variant(seqs, p, [everyone])
    for _ in range(3):
        assert_same(first, _prefill_variant(seqs, p, [everyone]), "prefill repeat")
    first = _decode_variant(seqs, n0, p, [everyone])
    for _ in range(3):
        assert_same(first, _decode_variant(seqs, n0, p, [everyone]), "decode repeat")


@pytest.mark.parametrize("h", HEADS)
def test_packed_decode_form_equals_the_batched_form(h):
    n, steps = 6, 20
    p = params(h)
    n0 = [0, 3, 15, 16, 21, 37]
    slots = [4, 0, 5, 2, 1, 3]
    seqs = [make_seq(n0[i] + steps, h, SEED + 6500 + i, "long") for i in range(n)]
    got = {}
    for form in ("batched", "packed"):
        pool = new_pool(seqs, slots)
        prefill(
            [sub(s, 0, m) for s, m in zip(seqs, n0, strict=True) if m],
            pool,
            [sl for sl, m in zip(slots, n0, strict=True) if m],
            p,
        )
        rows = []
        for t in range(steps):
            tok = tokens(seqs, [m + t for m in n0])
            if form == "packed":
                tok = {
                    key: val.transpose(0, 1).contiguous() for key, val in tok.items()
                }
                cu = torch.arange(n + 1, dtype=torch.int64, device=DEV)
                out = kda(tok, pool, p, cu_seqlens=cu, ssm_state_indices=ids(slots))
                assert out.shape == (1, n, h, D)
                rows.append(out[0])
            else:
                rows.append(kda(tok, pool, p, ssm_state_indices=ids(slots))[:, 0])
        got[form] = result(torch.stack(rows, 1), pool, slots)
    assert_same(got["batched"], got["packed"], "packed decode")
    for i, s in enumerate(seqs):
        assert torch.equal(got["packed"]["rows"][i], one_shot(s, p)[0][n0[i] :])
    _check_result_hash(f"packed_decode_h{h}", got["packed"])


def test_output_buffer_and_final_state():
    h = 4
    p = params(h)
    seqs = [make_seq(60, h, SEED + 7000 + i, "long") for i in range(3)]
    refs = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, [0, 1, 2])
    x = batch(seqs)
    out = torch.empty_like(x["v"][:, :21])
    got, st = recurrent_kda(
        *(cut(x, 0, 21)[key] for key in KEYS),
        initial_state=pool,
        output_final_state=True,
        output=out,
        **_kw(p),
    )
    assert got is out and st is pool
    out1 = torch.empty(3, 1, h, D, dtype=torch.bfloat16, device=DEV)
    got1, st1 = recurrent_kda(
        *(cut(x, 21, 22)[key] for key in KEYS),
        initial_state=pool,
        output=out1,
        **_kw(p),
    )
    assert got1 is out1 and st1 is None
    out2 = torch.empty_like(x["v"][:, 22:60])
    got2 = ext(cut(x, 22, 60), pool, p, output=out2)
    assert got2 is out2
    for i, ref in enumerate(refs):
        assert torch.equal(torch.cat([out[i], out1[i], out2[i]], 0), ref)
    with pytest.raises(BatchInvariantUnsupportedError):
        kda(cut(x, 0, 1), pool, p, output=out1.float())
    with pytest.raises(BatchInvariantUnsupportedError):
        kda(cut(x, 0, 1), pool, p, output=out1[:2])


def test_class_name_selects_the_same_class():
    h = 4
    p = params(h)
    s = make_seq(30, h, SEED + 7100, "steep")
    ref, ref_state = one_shot(s, p)
    x = batch([s])
    pool = new_pool([s], [0])
    kw = {**_kw(p), "batch_invariant": "kda.bf16.s16"}
    out = recurrent_kda(*(x[key] for key in KEYS), initial_state=pool, **kw)[0]
    assert torch.equal(out[0], ref) and torch.equal(pool[0], ref_state)
    before = pool.clone()
    for name in ("kda.bf16.c16", "kda", "true"):
        kw = {**_kw(p), "batch_invariant": name}
        with pytest.raises(ValueError, match="batch_invariant"):
            recurrent_kda(*(x[key] for key in KEYS), initial_state=pool, **kw)
    assert torch.equal(pool, before)


def _capture(step):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph, out


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEADS)
def test_graph_replay_equals_eager(h, gate):
    p = params(h)
    n0 = [7, 16, 100, 301]
    slots = [3, 1, 0, 2]
    steps = 41
    seqs = [make_seq(m + steps, h, SEED + 8000 + i, gate) for i, m in enumerate(n0)]

    def history():
        pool = new_pool(seqs, slots)
        prefill([sub(s, 0, m) for s, m in zip(seqs, n0, strict=True)], pool, slots, p)
        return pool

    idx = ids(slots)
    eager_pool = history()
    eager = [
        kda(tokens(seqs, [m + t for m in n0]), eager_pool, p, ssm_state_indices=idx)
        for t in range(steps)
    ]
    for i, s in enumerate(seqs):
        ref = one_shot(s, p)[0]
        assert torch.equal(torch.cat([e[i] for e in eager], 0), ref[n0[i] :])
    pool = history()
    static = {key: val.clone() for key, val in tokens(seqs, n0).items()}

    def step():
        return kda(static, pool, p, ssm_state_indices=idx)

    assert torch.equal(step(), eager[0])
    graph, out = _capture(step)
    for t in range(1, steps):
        for key, val in tokens(seqs, [m + t for m in n0]).items():
            static[key].copy_(val)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager[t]), f"step {t}"
    assert_same(result(out, eager_pool, slots), result(out, pool, slots), "graph state")
    _check_hash(f"graph_last_row_h{h}_{gate}", out)


@pytest.mark.parametrize("T", [3, 20])
def test_extend_on_default_slots_replayed_as_a_graph(T):
    h = 4
    p = params(h)
    n0 = [5, 16, 30]
    rounds = 12
    seqs = [
        make_seq(m + rounds * T, h, SEED + 8200 + i, "long") for i, m in enumerate(n0)
    ]
    refs = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, [0, 1, 2])
    prefill([sub(s, 0, m) for s, m in zip(seqs, n0, strict=True)], pool, [0, 1, 2], p)

    def grab(r):
        return {
            key: torch.stack(
                [
                    s[key][m + r * T : m + (r + 1) * T]
                    for s, m in zip(seqs, n0, strict=True)
                ]
            )
            for key in KEYS
        }

    static = {key: val.clone() for key, val in grab(0).items()}

    def step():
        return ext(static, pool, p)

    first = step()
    for i, ref in enumerate(refs):
        assert torch.equal(first[i], ref[n0[i] : n0[i] + T])
    graph, out = _capture(step)
    for r in range(1, rounds):
        for key, val in grab(r).items():
            static[key].copy_(val)
        graph.replay()
        torch.cuda.synchronize()
        for i, ref in enumerate(refs):
            a = n0[i] + r * T
            assert torch.equal(out[i], ref[a : a + T]), f"round {r}, request {i}"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_first_use_inside_a_graph_capture_is_refused():
    h = 4
    p = params(h)
    s = make_seq(4, h, SEED + 8500, "steep")
    pool = new_pool([s], [0])
    tok, idx = tokens([s], [0]), ids([0])
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(BatchInvariantUnsupportedError, match="capture"),
        torch.cuda.graph(graph),
    ):
        kda(tok, pool, p, ssm_state_indices=idx)
    torch.cuda.synchronize()
    assert torch.equal(pool[0], s["s0"])
    assert torch.equal(decode_all(s, pool, 0, p, 0, 4), one_shot(s, p)[0])


@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("T", [2, 5, 16, 23, 40])
def test_multi_token_steps_at_mixed_phases(h, T):
    p = params(h)
    n0 = [0, 3, 15, 16, 21, 37]
    slots = [4, 0, 5, 2, 1, 3]
    total = max(n0) + 2 * T + 2
    seqs = [
        make_seq(total, h, SEED + 8600 + i, "long" if i % 2 else "steep")
        for i in range(len(n0))
    ]
    refs = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, slots)
    prefill(
        [sub(s, 0, m) for s, m in zip(seqs, n0, strict=True) if m],
        pool,
        [sl for sl, m in zip(slots, n0, strict=True) if m],
        p,
    )
    pos = list(n0)
    outs = []
    for _ in range(2):
        x = {
            key: torch.stack(
                [s[key][a : a + T] for s, a in zip(seqs, pos, strict=True)]
            )
            for key in KEYS
        }
        out = ext(x, pool, p, ssm_state_indices=ids(slots))
        for i, ref in enumerate(refs):
            assert torch.equal(out[i], ref[pos[i] : pos[i] + T]), f"request {i}"
        outs.append(out)
        pos = [a + T for a in pos]
        rows = kda(tokens(seqs, pos), pool, p, ssm_state_indices=ids(slots))
        for i, ref in enumerate(refs):
            assert torch.equal(rows[i, 0], ref[pos[i]]), f"request {i} decode"
        pos = [a + 1 for a in pos]
    _check_hash(f"extend_T{T}_h{h}", torch.cat(outs, 0))


def test_slot_reuse_with_a_replaced_state():
    h = 4
    p = params(h)
    old = make_seq(60, h, SEED + 9000, "steep")
    new = make_seq(60, h, SEED + 9001, "long")
    ref_new, _ = one_shot(new, p)
    pool = new_pool([old], [0])
    prefill([sub(old, 0, 21)], pool, [0], p)
    decode_all(old, pool, 0, p, 21, 27)
    assert open_rows(pool, 0) == 11
    pool[0].copy_(new["s0"])
    assert torch.equal(decode_all(new, pool, 0, p, 0, 40), ref_new[:40])
    pool[0].copy_(new["s0"])
    pre = prefill([sub(new, 0, 23)], pool, [0], p)[0]
    rows = decode_all(new, pool, 0, p, 23, 60)
    assert torch.equal(torch.cat([pre, rows], 0), ref_new)
    pool[0].copy_(new["s0"])
    out = ext(batch([new], 0, 30), pool, p)
    assert torch.equal(out[0], ref_new[:30])
    assert torch.equal(decode_all(new, pool, 0, p, 30, 60), ref_new[30:])


def test_slot_reuse_from_a_zero_state():
    h = 4
    p = params(h)
    zero = torch.zeros(h, D, D, dtype=torch.bfloat16, device=DEV)
    old = dict(make_seq(53, h, SEED + 9050, "long"), s0=zero)
    new = dict(make_seq(53, h, SEED + 9051, "steep"), s0=zero)
    ref_old, _ = one_shot(old, p)
    ref_new, ref_state = one_shot(new, p)
    pool = new_pool([old], [0])
    assert torch.equal(decode_all(old, pool, 0, p, 0, 53), ref_old)
    pool[0].zero_()
    assert torch.equal(decode_all(new, pool, 0, p, 0, 53), ref_new)
    assert torch.equal(pool[0], ref_state)
    _check_hash("zero_state_rows", ref_new)


def test_reset_of_a_slot_recycled_after_a_short_request():
    h = 4
    p = params(h)
    old = make_seq(9, h, SEED + 9100, "steep")
    new = dict(make_seq(40, h, SEED + 9101, "steep"), s0=old["s0"])
    ref_new, _ = one_shot(new, p)
    pool = new_pool([old], [0])
    decode_all(old, pool, 0, p, 0, 9)
    assert torch.equal(pool[0], old["s0"])
    # the stamp cannot see this reuse: without a reset the old rows stay open
    assert open_rows(pool, 0) == 9
    assert not torch.equal(decode_all(new, pool, 0, p, 0, 3), ref_new[:3])
    pool[0].copy_(old["s0"])
    recurrent_kda_class_s_reset(pool, ids([0]))
    assert open_rows(pool, 0) == 0
    assert torch.equal(decode_all(new, pool, 0, p, 0, 40), ref_new)
    pool = new_pool([old], [0])
    decode_all(old, pool, 0, p, 0, 9)
    pre = prefill([sub(new, 0, 2)], pool, [0], p)[0]
    rows = decode_all(new, pool, 0, p, 2, 40)
    assert torch.equal(torch.cat([pre, rows], 0), ref_new)


def test_pool_aliases_share_one_side_state():
    h = 4
    p = params(h)
    s = make_seq(60, h, SEED + 9200, "long")
    ref, _ = one_shot(s, p)
    pool = new_pool([s], [0])
    aliases = (pool, pool.view_as(pool), pool.detach(), pool[:])
    rows = [
        kda(tokens([s], [t]), aliases[t % 4], p, ssm_state_indices=ids([0]))[0]
        for t in range(40)
    ]
    assert torch.equal(torch.cat(rows, 0), ref[:40])
    assert all(s_pool.get_pool(a) is s_pool.get_pool(pool) for a in aliases)
    assert s_pool.get_pool(pool.clone()) is not s_pool.get_pool(pool)


def test_pool_passed_as_a_fresh_view_each_call():
    h = 4
    p = params(h)
    s = make_seq(120, h, SEED + 9300, "steep")
    ref, _ = one_shot(s, p)
    big = torch.zeros(2, 4, h, D, D, dtype=torch.bfloat16, device=DEV)
    big[1, 2].copy_(s["s0"])
    pre = prefill([sub(s, 0, 21)], big[1], [2], p)[0]
    rows = decode_all(s, big[1], 2, p, 21, 120)
    assert torch.equal(torch.cat([pre, rows], 0), ref)


def _poison_dead_rows(pool, fill):
    sp = s_pool.get_pool(pool)
    for slot in range(pool.shape[0]):
        n = sp.open_chunk(slot)[0]
        for t in (sp.key_rows, sp.u_rows, sp.prefix_rows):
            assert t.shape[0] == pool.shape[0] and t.shape[-2:] == (CHUNK, D)
            t[slot, ..., n:, :] = fill


@pytest.mark.parametrize("fill", [0.0, 1e4, float("nan")])
@pytest.mark.parametrize("path", ["decode", "extend"])
def test_side_rows_past_the_count_are_inert(path, fill):
    h = 4
    p = params(h)
    s = make_seq(90, h, SEED + 9400, "steep")
    ref, ref_state = one_shot(s, p)
    pool = new_pool([s], [0])
    prefill([sub(s, 0, 37)], pool, [0], p)
    rows = []
    step = 1 if path == "decode" else 7
    for a in range(37, 86, step):
        _poison_dead_rows(pool, fill)
        if path == "decode":
            rows.append(decode_all(s, pool, 0, p, a, a + 1))
        else:
            rows.append(ext(batch([s], a, a + step), pool, p)[0])
    rows = torch.cat(rows, 0)
    assert torch.equal(rows, ref[37 : 37 + rows.shape[0]])
    assert torch.equal(pool[0], one_shot(s, p, 37 + rows.shape[0])[1])


BAD_SLOTS = [[0, 0], [0, 7], [-1, 0]]


@pytest.mark.parametrize("slots", BAD_SLOTS)
@pytest.mark.parametrize("form", ["prefill", "extend", "reset"])
def test_bad_slots_are_refused(form, slots):
    h = 4
    p = params(h)
    seqs = [make_seq(20, h, SEED + 9500 + i, "steep") for i in range(2)]
    pool = new_pool(seqs, [0, 1], size=4)
    before = pool.clone()
    with pytest.raises(BatchInvariantUnsupportedError, match="slot"):
        if form == "prefill":
            prefill(seqs, pool, slots, p)
        elif form == "extend":
            ext(batch(seqs, 0, 4), pool, p, ssm_state_indices=ids(slots))
        else:
            recurrent_kda_class_s_reset(pool, ids(slots))
    assert torch.equal(pool, before)
    assert all(open_rows(pool, slot) == 0 for slot in range(4))


def _to_cpu(x, kw):
    for d in (x, kw):
        for key, val in d.items():
            if isinstance(val, torch.Tensor):
                d[key] = val.cpu()


def _gqa(x, kw):
    for key in ("q", "k"):
        x[key] = torch.cat([x[key], x[key]], 2)


def _other_heads(x, kw):
    kw["initial_state"] = kw["initial_state"][:, :3].contiguous()


def _strided_pool(x, kw):
    pool = kw["initial_state"]
    big = torch.zeros(2 * pool.shape[0], *pool.shape[1:], dtype=pool.dtype, device=DEV)
    big[::2].copy_(pool)
    kw["initial_state"] = big[::2]


def _cast_inputs(dtype):
    def apply(x, kw):
        for key in KEYS:
            x[key] = x[key].to(dtype)

    return apply


def _set(**changes):
    def apply(x, kw):
        for key, val in changes.items():
            kw[key] = val(kw) if callable(val) else val

    return apply


BAD_INPUTS = {
    "fp32_pool": _set(initial_state=lambda kw: kw["initial_state"].float()),
    "fp16_pool": _set(initial_state=lambda kw: kw["initial_state"].half()),
    "no_pool": _set(initial_state=None),
    "fp32_inputs": _cast_inputs(torch.float32),
    "fp16_inputs": _cast_inputs(torch.float16),
    "gqa": _gqa,
    "no_A_log": _set(A_log=None),
    "no_dt_bias": _set(dt_bias=None),
    "fp64_A_log": _set(A_log=lambda kw: kw["A_log"].double()),
    "lower_bound_-10": _set(lower_bound=-10.0),
    "lower_bound_0.5": _set(lower_bound=0.5),
    "lower_bound_0": _set(lower_bound=0.0),
    "lower_bound_none": _set(lower_bound=None),
    "lower_bound_nan": _set(lower_bound=float("nan")),
    "strided_pool": _strided_pool,
    "pool_of_other_heads": _other_heads,
    "cpu": _to_cpu,
    "int64_slots": _set(
        ssm_state_indices=lambda kw: torch.zeros(1, dtype=torch.int64, device=DEV)
    ),
    "two_slots_for_one": _set(ssm_state_indices=lambda kw: ids([0, 1])),
    "backend_cute_dsl": _set(backend="cute-dsl"),
    "num_spec_tokens": _set(num_spec_tokens=2),
    "no_qk_l2norm": _set(use_qk_l2norm_in_kernel=False),
    "no_gate_in_kernel": _set(use_gate_in_kernel=False),
    "beta_not_logit": _set(beta_is_logit=False),
    "disable_state_update": _set(disable_state_update=True),
}
OPERATOR_ONLY = (
    "no_pool",
    "backend_cute_dsl",
    "num_spec_tokens",
    "no_qk_l2norm",
    "no_gate_in_kernel",
    "beta_not_logit",
    "disable_state_update",
)


BAD_CALLS = [
    (form, name)
    for form in ("decode", "prefill", "extend")
    for name in BAD_INPUTS
    if form != "extend" or name not in OPERATOR_ONLY
]


@pytest.mark.parametrize("form,name", BAD_CALLS)
def test_unsupported_inputs_raise_and_nothing_runs(form, name):
    h = 4
    p = params(h)
    seqs = [make_seq(8, h, SEED + 9600 + i, "steep") for i in range(2)]
    pool = new_pool(seqs, [0, 1])
    before = pool.clone()
    x = batch(seqs[:1], 0, 1 if form == "decode" else 8)
    kw = dict(_kw(p), initial_state=pool)
    if form == "extend":
        for key in list(kw):
            if key.startswith("use_") or key in ("beta_is_logit", "batch_invariant"):
                del kw[key]
    BAD_INPUTS[name](x, kw)
    given = kw["initial_state"]
    given_before = None if given is None else given.clone()
    fn = recurrent_kda_class_s_extend if form == "extend" else recurrent_kda
    with pytest.raises(BatchInvariantUnsupportedError, match="kda.bf16.s16"):
        fn(*(x[key] for key in KEYS), **kw)
    assert torch.equal(pool, before)
    if given is not None:
        assert torch.equal(given, given_before)
    assert all(open_rows(pool, slot) == 0 for slot in range(pool.shape[0]))
    assert torch.equal(decode_all(seqs[0], pool, 0, p, 0, 8), one_shot(seqs[0], p)[0])


def test_more_sequences_than_slots_is_refused():
    h = 4
    p = params(h)
    seqs = [make_seq(8, h, SEED + 9700 + i, "steep") for i in range(3)]
    pool = new_pool(seqs[:1], [0], size=2)
    for a, b in ((0, 1), (0, 8)):
        with pytest.raises(BatchInvariantUnsupportedError, match="slots"):
            kda(batch(seqs, a, b), pool, p)
        with pytest.raises(BatchInvariantUnsupportedError, match="slots"):
            ext(batch(seqs, a, b), pool, p)


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("length", [37, 200])
def test_torch_reference_agrees_to_tolerance(length, gate):
    h = 4
    p = params(h)
    seqs = [make_seq(length, h, SEED + 9800 + length + i, gate) for i in range(2)]
    pool = new_pool(seqs, [0, 1])
    rows = torch.stack(prefill(seqs, pool, [0, 1], p)).float()
    x = batch(seqs)
    s0 = torch.stack([s["s0"] for s in seqs])
    ref, ref_state = kda_class_s_reference(
        *(x[key] for key in KEYS), p["A_log"], p["dt_bias"], s0, lower_bound=LB
    )
    assert bool(torch.isfinite(rows).all())
    assert float((rows - ref).norm() / ref.norm()) < 1e-2
    state = pool[:2].float()
    assert float((state - ref_state).norm() / ref_state.norm()) < 1e-2


@pytest.mark.parametrize(
    "h,lower_bound,scale",
    [
        (1, -2.5, 0.25),
        (3, -2.5, 0.25),
        (32, -2.5, 0.25),
        (4, -2.5, None),
        (4, LB, 0.25),
        (4, -0.37, 1.0),
        (3, LB, None),
    ],
)
def test_other_parameters(h, lower_bound, scale):
    p = params(h, lower_bound=lower_bound, scale=scale)
    gate = "long" if h % 2 else "steep"
    seqs = [make_seq(50, h, SEED + 9900 + h + i, gate) for i in range(2)]
    whole = _all_ways(seqs, p)
    x = batch(seqs)
    s0 = torch.stack([s["s0"] for s in seqs])
    ref, _ = kda_class_s_reference(
        *(x[key] for key in KEYS),
        p["A_log"],
        p["dt_bias"],
        s0,
        scale=scale,
        lower_bound=lower_bound,
    )
    rows = torch.stack(whole["rows"]).float()
    assert float((rows - ref).norm() / ref.norm()) < 1e-2
    _check_result_hash(f"other_h{h}_lb{lower_bound}_scale{scale}", whole)


def test_parameters_are_inputs_of_the_term():
    h = 4
    s = make_seq(40, h, SEED + 9950, "steep")
    base = one_shot(s, params(h))[0]
    assert not torch.equal(one_shot(s, params(h, lower_bound=-2.5))[0], base)
    assert not torch.equal(one_shot(s, params(h, scale=0.25))[0], base)
    assert torch.equal(one_shot(s, params(h, scale=D**-0.5))[0], base)


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("path", ["prefill", "decode", "extend"])
def test_negative_control_one_changed_element(path, key):
    h, T, t = 4, 40, 21
    p = params(h)
    s = make_seq(T, h, SEED + 9960, "steep")

    def run(seq):
        if path == "prefill":
            return one_shot(seq, p)[0]
        pool = new_pool([seq], [0])
        if path == "decode":
            return decode_all(seq, pool, 0, p, 0, T)
        pre = prefill([sub(seq, 0, 5)], pool, [0], p)[0]
        return torch.cat([pre, ext(batch([seq], 5, T), pool, p)[0]], 0)

    base = run(s)
    changed = dict(s, **{key: s[key].clone()})
    changed[key][t, 0].flatten()[0] = 5.0
    assert not torch.equal(changed[key], s[key])
    got = run(changed)
    assert torch.equal(got[:t], base[:t])
    assert torch.equal(got[t:, 1:], base[t:, 1:])
    assert not torch.equal(got[t, 0], base[t, 0])
    assert not torch.equal(got, base)


def test_prefill_with_int64_cu_seqlens():
    h = 4
    p = params(h)
    seqs = [make_seq(n, h, SEED + 9970 + n, "steep") for n in (37, 5, 64)]
    want = [one_shot(s, p)[0] for s in seqs]
    pool = new_pool(seqs, [0, 1, 2])
    lens = [s["q"].shape[0] for s in seqs]
    cu = torch.tensor(np.cumsum([0] + lens), dtype=torch.int64, device=DEV)
    cat = {key: torch.cat([s[key] for s in seqs], 0)[None] for key in KEYS}
    out = kda(cat, pool, p, cu_seqlens=cu, ssm_state_indices=ids([0, 1, 2]))
    for got, ref in zip(torch.split(out[0], lens, 0), want, strict=True):
        assert torch.equal(got, ref)


def test_non_contiguous_activations():
    h, T = 4, 40
    p = params(h)
    s = make_seq(T, h, SEED + 9975, "long")
    ref, ref_state = one_shot(s, p)
    x = batch([s])
    wide = {key: torch.repeat_interleave(x[key], 2, dim=1)[:, ::2] for key in KEYS}
    assert not wide["q"].is_contiguous()
    pool = new_pool([s], [0])
    assert torch.equal(kda(wide, pool, p)[0], ref)
    assert torch.equal(pool[0], ref_state)


def test_another_slice_of_a_pool_is_refused():
    h = 4
    p = params(h)
    s = make_seq(20, h, SEED + 9980, "steep")
    pool = new_pool([s], [0], size=4)
    prefill([sub(s, 0, 19)], pool, [0], p)
    before = pool.clone()
    for other in (pool[:2], pool[1:]):
        with pytest.raises(BatchInvariantUnsupportedError, match="slice"):
            kda(tokens([s], [19]), other, p, ssm_state_indices=ids([0]))
    assert torch.equal(pool, before)
    got = kda(tokens([s], [19]), pool, p, ssm_state_indices=ids([0]))[0]
    assert torch.equal(got, one_shot(s, p)[0][19:])


def test_decode_slot_outside_the_pool_is_skipped():
    h = 4
    p = params(h)
    seqs = [make_seq(20, h, SEED + 9985 + i, "steep") for i in range(2)]
    ref = one_shot(seqs[0], p)[0]
    for bad in (99, -1, 3):
        pool = new_pool(seqs, [0, 1], size=3)
        prefill([sub(seqs[0], 0, 19)], pool, [0], p)
        sp = s_pool.get_pool(pool)
        live = (pool, sp.key_rows, sp.u_rows, sp.prefix_rows, sp.cnt, sp.stamp)
        before = [t[1:].clone() for t in live]
        out = kda(tokens(seqs, [19, 0]), pool, p, ssm_state_indices=ids([0, bad]))
        torch.cuda.synchronize()
        assert torch.equal(out[0], ref[19:])
        for t, old in zip(live, before, strict=True):
            assert torch.equal(t[1:], old)


def test_restored_snapshot_needs_a_reset():
    h = 4
    p = params(h)
    s = make_seq(40, h, SEED + 9990, "long")
    pool = new_pool([s], [0])
    prefill([sub(s, 0, 20)], pool, [0], p)
    snapshot = pool[0].clone()
    decode_all(s, pool, 0, p, 20, 25)
    assert open_rows(pool, 0) == 9
    fresh = dict(sub(s, 25, 40), s0=snapshot)
    want = one_shot(fresh, p)[0]
    pool[0].copy_(snapshot)
    stale = kda(tokens([s], [25]), pool, p, ssm_state_indices=ids([0]))[0]
    assert not torch.equal(stale, want[:1])
    pool[0].copy_(snapshot)
    recurrent_kda_class_s_reset(pool, ids([0]))
    assert open_rows(pool, 0) == 0
    assert torch.equal(decode_all(s, pool, 0, p, 25, 40), want)


def test_output_sharing_memory_with_an_input_is_refused():
    h = 4
    p = params(h)
    s = make_seq(20, h, SEED + 9995, "steep")
    pool = new_pool([s], [0])
    x = batch([s])
    with pytest.raises(BatchInvariantUnsupportedError, match="shares memory"):
        kda(x, pool, p, output=x["v"])
    with pytest.raises(BatchInvariantUnsupportedError, match="shares memory"):
        ext(x, pool, p, output=x["v"])
    assert torch.equal(pool[0], s["s0"])


def test_host_read_forms_are_refused_inside_a_capture():
    h = 4
    p = params(h)
    s = make_seq(20, h, SEED + 9997, "steep")
    pool = new_pool([s], [0])
    kda(tokens([s], [0]), pool, p, ssm_state_indices=ids([0]))
    cu = torch.tensor([0, 19], dtype=torch.int32, device=DEV)
    x, idx = batch([s], 1, 20), ids([0])
    packed = {key: x[key].reshape(1, 19, *x[key].shape[2:]) for key in KEYS}
    for call in (
        lambda: kda(packed, pool, p, cu_seqlens=cu, ssm_state_indices=idx),
        lambda: kda(x, pool, p, ssm_state_indices=idx),
        lambda: ext(x, pool, p, ssm_state_indices=idx),
        lambda: recurrent_kda_class_s_reset(pool, idx),
    ):
        graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(BatchInvariantUnsupportedError, match="capture"),
            torch.cuda.graph(graph),
        ):
            call()
        torch.cuda.synchronize()
    assert open_rows(pool, 0) == 1
    assert torch.equal(decode_all(s, pool, 0, p, 1, 20), one_shot(s, p)[0][1:])
