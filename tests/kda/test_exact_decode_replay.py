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

Bitwise tests for the replay-based exact KDA decode (SM100/SM103 only).

Every comparison is ``torch.equal``; nothing uses a tolerance. Set
``KDA_EXACT_HASH_DIR`` to a directory to record output hashes on the first run
and assert them in a second process.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch

from flashinfer import kda_prefill_cute
from flashinfer.kda import recurrent_kda
from flashinfer.kda_exact_decode import (
    CHUNK,
    ROUTE,
    KdaExactDecodeState,
    KdaLayerParams,
    kda_exact_accept,
    kda_exact_decode,
    kda_exact_prefill,
    kda_exact_verify,
)
from flashinfer.utils import get_compute_capability

D = 128
L = 4096
SEED = 1234
LB = -5.0
HEAD_CONFIGS = [16, 32]
GATES = {
    "steep": 0.0,
    "long": -6.0,
}  # raw-gate offset; g_eff = LB * sigmoid(exp(A_log) * (g + dt_bias))
N0_LIST = [0, 1, 15, 16, 17, 1000, 4079]
BATCH_POSITIONS = [0, 1, 15, 16, 17, 1000]


def _skip_unless_supported():
    dev = torch.device("cuda")
    if get_compute_capability(dev) not in ((10, 0), (10, 3)):
        pytest.skip(
            "exact KDA decode is pinned to the SM100/SM103 cute-dsl prefill kernel"
        )
    if not kda_prefill_cute._is_cute_dsl_kda_runtime_available():
        pytest.skip("the cute-dsl KDA prefill runtime is not available")


def kda_exact_prefill_then_decode(q, k, v, g, beta, initial_state, n0, params):
    """Prefill ``[0, n0)`` then decode ``[n0, L)`` one token at a time."""
    L = q.shape[0]
    out0, st = kda_exact_prefill(
        q[:n0], k[:n0], v[:n0], g[:n0], beta[:n0], initial_state, params
    )
    rows = [out0]
    for p in range(n0, L):
        rows.append(
            kda_exact_decode(
                q[p : p + 1],
                k[p : p + 1],
                v[p : p + 1],
                g[p : p + 1],
                beta[p : p + 1],
                [st],
                params,
            )
        )
    return torch.cat(rows, 0), st


def make_params(h, seed=SEED):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed + 7)
    return KdaLayerParams(
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


def one_shot(s, params):
    """The pinned prefill of the whole sequence (the reference term)."""
    dev = s["q"].device
    cu = torch.tensor([0, s["q"].shape[0]], dtype=torch.int32, device=dev)
    with kda_prefill_cute.force_route(
        ROUTE
    ):  # the class's reference term is the pinned route
        res = recurrent_kda(
            s["q"][None],
            s["k"][None],
            s["v"][None],
            s["g"][None],
            s["beta"][None],
            A_log=params.A_log,
            dt_bias=params.dt_bias,
            use_gate_in_kernel=True,
            beta_is_logit=True,
            lower_bound=LB,
            initial_state=s["s0"][None].clone(),
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            backend="cute-dsl",
        )
    torch.cuda.synchronize()
    return res[0][0], res[1][0]


def _hash(t):
    return hashlib.sha256(
        t.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def _check_hash(name, t):
    d = os.environ.get("KDA_EXACT_HASH_DIR")
    if not d:
        return
    p = Path(d) / f"{name}.sha256"
    h = _hash(t)
    if p.exists():
        assert p.read_text().strip() == h, f"{name}: cross-process hash mismatch"
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(h)


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEAD_CONFIGS)
@pytest.mark.parametrize("n0", N0_LIST)
def test_prefill_then_decode_equals_one_shot(h, gate, n0):
    _skip_unless_supported()
    params = make_params(h)
    s = make_seq(L, h, SEED + h, GATES[gate])
    o_ref, st_ref = one_shot(s, params)
    o, st = kda_exact_prefill_then_decode(
        s["q"], s["k"], s["v"], s["g"], s["beta"], s["s0"].clone(), n0, params
    )
    torch.cuda.synchronize()
    assert torch.equal(o, o_ref), (
        f"decoded rows differ from the one-shot prefill (n0={n0})"
    )
    assert torch.equal(st.current_state, st_ref), (
        "final state differs from the one-shot prefill"
    )
    assert st.position == L
    _check_hash(f"prefill_then_decode_h{h}_{gate}_n0{n0}", o)


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", HEAD_CONFIGS)
def test_packed_requests_at_mixed_positions(h, gate):
    """Six requests at different chunk phases decoded in one packed call; each equals its own
    one-shot prefill, and equals the same request decoded alone."""
    _skip_unless_supported()
    params = make_params(h)
    seqs = [
        make_seq(L, h, SEED + 100 + i, GATES[gate]) for i in range(len(BATCH_POSITIONS))
    ]
    refs = [one_shot(s, params) for s in seqs]
    states = []
    for s, p0 in zip(seqs, BATCH_POSITIONS, strict=True):
        _, st = kda_exact_prefill(
            s["q"][:p0],
            s["k"][:p0],
            s["v"][:p0],
            s["g"][:p0],
            s["beta"][:p0],
            s["s0"].clone(),
            params,
        )
        states.append(st)
    steps = 2 * CHUNK + 3
    got = [[] for _ in seqs]
    for _t in range(steps):
        idx = [st.position for st in states]
        q_t = torch.stack([s["q"][p] for s, p in zip(seqs, idx, strict=True)])
        k_t = torch.stack([s["k"][p] for s, p in zip(seqs, idx, strict=True)])
        v_t = torch.stack([s["v"][p] for s, p in zip(seqs, idx, strict=True)])
        g_t = torch.stack([s["g"][p] for s, p in zip(seqs, idx, strict=True)])
        b_t = torch.stack([s["beta"][p] for s, p in zip(seqs, idx, strict=True)])
        rows = kda_exact_decode(q_t, k_t, v_t, g_t, b_t, states, params)
        for i in range(len(seqs)):
            got[i].append(rows[i])
    torch.cuda.synchronize()
    for i, (s, p0) in enumerate(zip(seqs, BATCH_POSITIONS, strict=True)):
        o_ref = refs[i][0]
        batched = torch.stack(got[i])
        assert torch.equal(batched, o_ref[p0 : p0 + steps]), (
            f"request {i} (start {p0}) differs from its one-shot prefill"
        )
        # the same request decoded alone
        _, st_alone = kda_exact_prefill(
            s["q"][:p0],
            s["k"][:p0],
            s["v"][:p0],
            s["g"][:p0],
            s["beta"][:p0],
            s["s0"].clone(),
            params,
        )
        alone = []
        for _t in range(steps):
            p = st_alone.position
            alone.append(
                kda_exact_decode(
                    s["q"][p : p + 1],
                    s["k"][p : p + 1],
                    s["v"][p : p + 1],
                    s["g"][p : p + 1],
                    s["beta"][p : p + 1],
                    [st_alone],
                    params,
                )[0]
            )
        torch.cuda.synchronize()
        assert torch.equal(torch.stack(alone), batched), (
            f"request {i}: alone vs batched differ"
        )
        assert torch.equal(st_alone.current_state, states[i].current_state)
    _check_hash(f"packed_h{h}_{gate}", torch.cat([torch.stack(g_) for g_ in got]))


@pytest.mark.parametrize("h", HEAD_CONFIGS)
def test_repeat_is_bitwise(h):
    _skip_unless_supported()
    params = make_params(h)
    s = make_seq(600, h, SEED + 5, GATES["long"])
    o1, st1 = kda_exact_prefill_then_decode(
        s["q"], s["k"], s["v"], s["g"], s["beta"], s["s0"].clone(), 37, params
    )
    o2, st2 = kda_exact_prefill_then_decode(
        s["q"], s["k"], s["v"], s["g"], s["beta"], s["s0"].clone(), 37, params
    )
    torch.cuda.synchronize()
    assert torch.equal(o1, o2) and torch.equal(st1.current_state, st2.current_state)
    _check_hash(f"repeat_h{h}", o1)


def test_negative_controls():
    _skip_unless_supported()
    h = 16
    params = make_params(h)
    s = make_seq(200, h, SEED + 9, GATES["long"])
    with pytest.raises(ValueError):
        KdaExactDecodeState(boundary_state=s["s0"].to(torch.bfloat16))
    with pytest.raises(ValueError):
        KdaExactDecodeState(boundary_state=s["s0"].clone(), position=37)
    # resuming from the state saved mid-chunk, without the history, is not exact
    o_ref, _ = one_shot(s, params)
    cu = torch.tensor([0, 37], dtype=torch.int32, device=s["q"].device)
    with kda_prefill_cute.force_route(ROUTE):
        res = recurrent_kda(
            s["q"][:37][None],
            s["k"][:37][None],
            s["v"][:37][None],
            s["g"][:37][None],
            s["beta"][:37][None],
            A_log=params.A_log,
            dt_bias=params.dt_bias,
            use_gate_in_kernel=True,
            beta_is_logit=True,
            lower_bound=LB,
            initial_state=s["s0"][None].clone(),
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            backend="cute-dsl",
        )
    st37 = res[1][0]
    # the kernel's own runtime route rule is a non-member: unpinned, one sequence of 16 heads takes the "decomp" form
    # (n_seq * heads * 2 <= sm_count) while the class pins "engine"; the two forms differ bitwise
    s1 = make_seq(300, h, SEED + 300, GATES["long"])
    cu1 = torch.tensor([0, 300], dtype=torch.int32, device=s["q"].device)

    def call1():
        return recurrent_kda(
            s1["q"][None],
            s1["k"][None],
            s1["v"][None],
            s1["g"][None],
            s1["beta"][None],
            A_log=params.A_log,
            dt_bias=params.dt_bias,
            use_gate_in_kernel=True,
            beta_is_logit=True,
            lower_bound=LB,
            initial_state=s1["s0"][None].clone(),
            output_final_state=True,
            cu_seqlens=cu1,
            use_qk_l2norm_in_kernel=True,
            backend="cute-dsl",
        )

    o_auto = call1()[0]
    with kda_prefill_cute.force_route(ROUTE):
        o_pin = call1()[0]
    with kda_prefill_cute.force_route("decomp"):
        o_decomp = call1()[0]
    torch.cuda.synchronize()
    assert torch.equal(o_auto, o_decomp), (
        "one sequence of 16 heads takes decomp unpinned"
    )
    if torch.equal(o_decomp, o_pin):
        pytest.skip(
            "the two cute-dsl routes agree bitwise on this device; nothing to pin"
        )
    wrong = KdaExactDecodeState(
        boundary_state=st37.clone(), position=32
    )  # pretend 37 is a boundary
    rows = []
    for p in range(37, 60):
        rows.append(
            kda_exact_decode(
                s["q"][p : p + 1],
                s["k"][p : p + 1],
                s["v"][p : p + 1],
                s["g"][p : p + 1],
                s["beta"][p : p + 1],
                [wrong],
                params,
            )[0]
        )
    torch.cuda.synchronize()
    assert not torch.equal(torch.stack(rows), o_ref[37:60]), (
        "the negative control must differ"
    )


@pytest.mark.parametrize("gate", list(GATES))
@pytest.mark.parametrize("h", [16])
def test_verify_rounds_with_partial_acceptance(h, gate):
    """MTP verify (T = 1 + S drafts) with partial acceptance per round equals the one-shot prefill of the accepted
    sequence: every draft row is the prefill row of that position given the accepted prefix, and rollback is exact."""
    from flashinfer.kda_exact_decode import kda_exact_verify

    _skip_unless_supported()
    params = make_params(h)
    S = 4
    T = 1 + S
    rounds = 12
    # Two requests. Draft tokens come from a long pool; accepted tokens form the true sequence.
    pool = [
        make_seq(rounds * T + 100, h, SEED + 200 + i, GATES[gate]) for i in range(2)
    ]
    gen = torch.Generator(device="cuda").manual_seed(77)
    accepts = torch.randint(1, T + 1, (rounds, 2), generator=gen, device="cuda")
    accepts[0] = T  # round 0: every token accepted (fresh prompt)
    states = [KdaExactDecodeState(boundary_state=p["s0"].clone()) for p in pool]
    cursor = [0, 0]  # accepted length so far per request
    prev_acc = None
    for r in range(rounds):
        # round r drafts tokens [cursor, cursor + T) of each request's true sequence
        toks = {
            key: torch.stack(
                [pool[i][key][cursor[i] : cursor[i] + T] for i in range(2)]
            )
            for key in ("q", "k", "v", "g", "beta")
        }
        out = kda_exact_verify(
            toks["q"],
            toks["k"],
            toks["v"],
            toks["g"],
            toks["beta"],
            states,
            params,
            num_accepted_tokens=prev_acc,
        )
        torch.cuda.synchronize()
        for i in range(2):
            # the reference: a one-shot prefill of the accepted prefix + these T drafts
            Lr = cursor[i] + T
            ref, _ = one_shot(
                {
                    key: (pool[i][key][:Lr] if key != "s0" else pool[i]["s0"])
                    for key in pool[i]
                },
                params,
            )
            assert torch.equal(out[i], ref[cursor[i] : Lr]), (
                f"round {r} request {i}: draft rows differ"
            )
        prev_acc = accepts[r].to(torch.int32)
        for i in range(2):
            cursor[i] += int(accepts[r, i])
    # close the last draft and compare the final state with the one-shot prefill of the accepted sequence
    from flashinfer.kda_exact_decode import kda_exact_accept

    kda_exact_accept(states, prev_acc, params)
    torch.cuda.synchronize()
    for i in range(2):
        _, st_ref = one_shot(
            {
                key: (pool[i][key][: cursor[i]] if key != "s0" else pool[i]["s0"])
                for key in pool[i]
            },
            params,
        )
        assert torch.equal(states[i].current_state, st_ref), (
            f"request {i}: state after acceptance differs"
        )
        assert states[i].position == cursor[i]


def test_rejects_unsupported_contracts():
    _skip_unless_supported()
    h = 16
    params = make_params(h)
    s = make_seq(40, h, SEED + 11, GATES["long"])
    with pytest.raises(ValueError):
        KdaLayerParams(A_log=params.A_log, dt_bias=params.dt_bias, lower_bound=0.0)
    with pytest.raises(ValueError):
        KdaLayerParams(A_log=params.A_log, dt_bias=params.dt_bias, lower_bound=None)
    # GQA (HV != H) has no pinned kernel form
    with pytest.raises(NotImplementedError):
        kda_exact_prefill(
            s["q"][:, :8], s["k"][:, :8], s["v"], s["g"], s["beta"], s["s0"], params
        )
    st = KdaExactDecodeState(boundary_state=s["s0"].clone())
    with pytest.raises(NotImplementedError):
        kda_exact_decode(
            s["q"][:1, :8],
            s["k"][:1, :8],
            s["v"][:1],
            s["g"][:1],
            s["beta"][:1],
            [st],
            params,
        )
    # a state built with explicit history resumes mid-chunk
    o_ref, _ = one_shot(s, params)
    _, st = kda_exact_prefill(
        s["q"][:21],
        s["k"][:21],
        s["v"][:21],
        s["g"][:21],
        s["beta"][:21],
        s["s0"].clone(),
        params,
    )
    copy = KdaExactDecodeState(
        boundary_state=st.boundary_state.clone(),
        position=21,
        hist_q=st.hist_q.clone(),
        hist_k=st.hist_k.clone(),
        hist_v=st.hist_v.clone(),
        hist_g=st.hist_g.clone(),
        hist_beta=st.hist_beta.clone(),
    )
    row = kda_exact_decode(
        s["q"][21:22],
        s["k"][21:22],
        s["v"][21:22],
        s["g"][21:22],
        s["beta"][21:22],
        [copy],
        params,
    )
    torch.cuda.synchronize()
    assert torch.equal(row[0], o_ref[21])


def test_verify_rejects_bad_acceptance():
    _skip_unless_supported()
    h = 16
    params = make_params(h)
    s = make_seq(20, h, SEED + 13, GATES["long"])
    T = 3
    states = [KdaExactDecodeState(boundary_state=s["s0"].clone()) for _ in range(2)]
    toks = {
        key: torch.stack([s[key][:T], s[key][T : 2 * T]])
        for key in ("q", "k", "v", "g", "beta")
    }
    with pytest.raises(ValueError):
        kda_exact_accept(
            states, torch.tensor([1, 1], dtype=torch.int32, device="cuda"), params
        )
    kda_exact_verify(
        toks["q"], toks["k"], toks["v"], toks["g"], toks["beta"], states, params
    )
    with pytest.raises(ValueError):  # open draft, no acceptance given
        kda_exact_verify(
            toks["q"], toks["k"], toks["v"], toks["g"], toks["beta"], states, params
        )
    for bad in ([0, 1], [1, T + 1]):
        with pytest.raises(ValueError):
            kda_exact_verify(
                toks["q"],
                toks["k"],
                toks["v"],
                toks["g"],
                toks["beta"],
                states,
                params,
                num_accepted_tokens=torch.tensor(bad, dtype=torch.int32, device="cuda"),
            )
    assert all(st._snap is not None for st in states), (
        "a rejected call must leave every request untouched"
    )
    kda_exact_accept(
        states, torch.tensor([T, 1], dtype=torch.int32, device="cuda"), params
    )
    assert [st.position for st in states] == [T, 1]
