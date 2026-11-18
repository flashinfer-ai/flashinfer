# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The stateless call of the KDA class: ``recurrent_kda(batch_invariant=True)`` without ``initial_state``, the form a
trainer's forward pass takes (SM100 / SM103 / SM107).

Every comparison is bitwise. The references are the pool calls of the class: a prefill of the sequence alone into a
pool, a chunked prefill, and a prefill followed by decode steps.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer.batch_invariant as bi
from flashinfer import kda_prefill_cute
from flashinfer.batch_invariant import BatchInvariantUnsupportedError
from flashinfer.batch_invariant.kda import pool as bi_pool
from flashinfer.batch_invariant.kda.spec import KDA_ARCHS
from flashinfer.kda import recurrent_kda

needs_device = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not bi.is_supported_device(torch.device("cuda"), KDA_ARCHS)
    or not kda_prefill_cute._is_cute_dsl_kda_runtime_available(),
    reason="needs a CUDA device of SM100 / SM103 / SM107 with the CuTe DSL KDA prefill",
)

D = 128
LB = -5.0
HEADS = [96, 12, 8]
KEYS = ("q", "k", "v", "g", "beta")
RAGGED = [1, 15, 16, 17, 33, 250, 64, 5, 129, 1000, 2, 47]


def params(h, seed=1234):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed + 7)
    return dict(
        A_log=(0.1 * torch.randn(h, device=dev, generator=gen)).float(),
        dt_bias=(0.1 * torch.randn(h, D, device=dev, generator=gen)).float(),
        lower_bound=LB,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        beta_is_logit=True,
        batch_invariant=True,
    )


def make_seq(length, h, seed, gate_offset=0.0):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    f32 = dict(device=dev, generator=gen, dtype=torch.float32)
    return dict(
        q=torch.randn(length, h, D, **f32).to(torch.bfloat16),
        k=torch.randn(length, h, D, **f32).to(torch.bfloat16),
        v=torch.randn(length, h, D, **f32).to(torch.bfloat16),
        g=(torch.randn(length, h, D, **f32) + gate_offset).to(torch.bfloat16),
        beta=torch.randn(length, h, **f32).to(torch.bfloat16),
    )


def make_seqs(lens, h, seed=100, gate_offset=0.0):
    return [make_seq(n, h, seed + i, gate_offset) for i, n in enumerate(lens)]


def packed(seqs):
    lens = [s["q"].shape[0] for s in seqs]
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int64, device="cuda"
    )
    return [torch.cat([s[key] for s in seqs], 0)[None] for key in KEYS], cu


def same_bits(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))


def stateless(seqs, p, **kw):
    """One stateless call of the packed sequences; the rows of each sequence."""
    x, cu = packed(seqs)
    out, state = recurrent_kda(*x, cu_seqlens=cu, **p, **kw)
    torch.cuda.synchronize()
    assert state is None
    return list(out[0].split([s["q"].shape[0] for s in seqs]))


def pool_prefill(s, p, cuts=()):
    """The sequence alone in a one-slot pool, prefilled in pieces (each longer than one token) cut at ``cuts``."""
    h = s["q"].shape[1]
    pool = torch.zeros(1, h, D, D, device="cuda")
    rows, edges = [], [0, *cuts, s["q"].shape[0]]
    for a, b in zip(edges, edges[1:], strict=False):
        x = [s[key][None, a:b].contiguous() for key in KEYS]
        cu = torch.tensor([0, b - a], dtype=torch.int64, device="cuda")
        rows.append(recurrent_kda(*x, initial_state=pool, cu_seqlens=cu, **p)[0][0])
    torch.cuda.synchronize()
    return torch.cat(rows, 0)


def pool_prefill_then_decode(s, p, n0):
    """The first ``n0`` tokens prefilled into a pool (none when ``n0`` is 0), the rest as ``[1, 1]`` decode steps."""
    h, n = s["q"].shape[1], s["q"].shape[0]
    pool = torch.zeros(1, h, D, D, device="cuda")
    rows = []
    if n0 > 1:
        x = [s[key][None, :n0].contiguous() for key in KEYS]
        cu = torch.tensor([0, n0], dtype=torch.int64, device="cuda")
        rows.append(recurrent_kda(*x, initial_state=pool, cu_seqlens=cu, **p)[0][0])
    else:
        n0 = 0
    for t in range(n0, n):
        x = [s[key][None, t : t + 1].contiguous() for key in KEYS]
        rows.append(recurrent_kda(*x, initial_state=pool, **p)[0][0])
    torch.cuda.synchronize()
    return torch.cat(rows, 0)


@needs_device
@pytest.mark.parametrize("h", HEADS)
def test_ragged_batch_equals_pool_prefill_of_each_sequence(h):
    p = params(h)
    seqs = make_seqs(RAGGED, h)
    got = stateless(seqs, p)
    for s, rows in zip(seqs, got, strict=True):
        n = s["q"].shape[0]
        ref = pool_prefill(s, p) if n > 1 else pool_prefill_then_decode(s, p, 0)
        assert same_bits(rows, ref), f"h={h} length={n}"
        assert bool(torch.isfinite(rows.float()).all())


@needs_device
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("gate", [0.0, -6.0])
def test_equals_one_packed_pool_prefill(h, gate):
    p = params(h)
    seqs = make_seqs([300, 17, 64, 31, 2, 513], h, seed=300, gate_offset=gate)
    x, cu = packed(seqs)
    pool = torch.zeros(len(seqs), h, D, D, device="cuda")
    ref = recurrent_kda(*x, initial_state=pool, cu_seqlens=cu, **p)[0]
    out = recurrent_kda(*x, cu_seqlens=cu, **p)[0]
    torch.cuda.synchronize()
    assert same_bits(out, ref)


@needs_device
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize(
    "length,cuts",
    [
        (200, (16,)),
        (200, (64, 128)),
        (203, (48, 96, 192)),
        (1000, (512,)),
        (77, (16, 32, 48, 64)),
    ],
)
def test_equals_chunked_prefill_cut_at_chunk_multiples(h, length, cuts):
    p = params(h)
    s = make_seq(length, h, seed=500 + length)
    got = stateless([make_seq(40, h, 1), s, make_seq(7, h, 2)], p)[1]
    assert same_bits(got, pool_prefill(s, p, cuts))


@needs_device
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("length,cut", [(200, 7), (200, 100), (203, 33), (64, 31)])
def test_equals_prefill_cut_anywhere_then_continued(h, length, cut):
    """A cut off a multiple of 16 continues as decode steps (a prefill would drop the open rows)."""
    p = params(h)
    s = make_seq(length, h, seed=700 + length + cut)
    got = stateless([s, make_seq(90, h, 3)], p)[0]
    assert same_bits(got, pool_prefill_then_decode(s, p, cut))


@needs_device
@pytest.mark.parametrize("h", HEADS)
@pytest.mark.parametrize("n0", [0, 2, 15, 16, 17, 40])
def test_equals_prefill_then_decode_steps(h, n0):
    p = params(h)
    s = make_seq(n0 + 37, h, seed=900 + n0)
    got = stateless([make_seq(5, h, 4), make_seq(130, h, 5), s], p)[2]
    assert same_bits(got, pool_prefill_then_decode(s, p, n0))


@needs_device
@pytest.mark.parametrize("h", HEADS)
def test_rows_do_not_depend_on_the_batch(h):
    p = params(h)
    seqs = make_seqs([129, 16, 3, 700, 33, 1, 64, 255], h, seed=1100)
    alone = [stateless([s], p)[0] for s in seqs]
    for order in (
        [0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0],
        [3, 0, 5],
        [5, 3],
    ):
        got = stateless([seqs[i] for i in order], p)
        for i, rows in zip(order, got, strict=True):
            assert same_bits(rows, alone[i]), f"h={h} order={order} seq={i}"
    for copies in (2, 16, 64):
        got = stateless([seqs[4]] * copies, p)
        assert all(same_bits(rows, alone[4]) for rows in got)


@needs_device
@pytest.mark.parametrize("h", HEADS)
def test_fixed_layout_equals_packed(h):
    p = params(h)
    seqs = make_seqs([50, 50, 50, 50, 50], h, seed=1300)
    ref = stateless(seqs, p)
    x = [torch.stack([s[key] for s in seqs]) for key in KEYS]
    out, state = recurrent_kda(*x, **p)
    torch.cuda.synchronize()
    assert state is None
    for i in range(len(seqs)):
        assert same_bits(out[i], ref[i])
    one = [t[:, :1].contiguous() for t in x]
    first = recurrent_kda(*one, **p)[0]
    torch.cuda.synchronize()
    assert same_bits(first, out[:, :1])


@needs_device
def test_one_token_sequences_are_first_tokens():
    """With a pool, ``cu_seqlens`` of one token per sequence is a decode step; without one it is N whole sequences."""
    h = 8
    p = params(h)
    seqs = make_seqs([1, 1, 1, 1], h, seed=1500)
    got = stateless(seqs, p)
    for s, rows in zip(seqs, got, strict=True):
        assert same_bits(rows, pool_prefill_then_decode(s, p, 0))


@needs_device
@pytest.mark.parametrize("h", HEADS)
def test_repeats_and_output_buffer(h):
    p = params(h)
    seqs = make_seqs([77, 16, 300, 9], h, seed=1700)
    x, cu = packed(seqs)
    ref = recurrent_kda(*x, cu_seqlens=cu, **p)[0]
    buf = torch.full_like(x[2], float("nan"))
    for _ in range(3):
        out, state = recurrent_kda(*x, cu_seqlens=cu, output=buf, **p)
        torch.cuda.synchronize()
        assert out is buf and state is None
        assert same_bits(buf, ref)
        buf.fill_(float("nan"))
    cu32 = cu.to(torch.int32)
    assert same_bits(recurrent_kda(*x, cu_seqlens=cu32, **p)[0], ref)
    odd = torch.empty(x[2].numel() + 1, dtype=torch.bfloat16, device="cuda")[1:]
    odd = odd.view(x[2].shape)
    assert odd.data_ptr() % 16
    assert recurrent_kda(*x, cu_seqlens=cu, output=odd, **p)[0] is odd
    torch.cuda.synchronize()
    assert same_bits(odd, ref)


@needs_device
@pytest.mark.parametrize("layout", ["packed", "fixed"])
def test_no_pool_state_is_created_and_a_repeated_call_allocates_nothing(layout):
    h = 12
    p = params(h)
    pools, retired = dict(bi_pool._POOLS), dict(bi_pool._RETIRED)
    if layout == "packed":
        x, cu = packed(make_seqs([100, 33, 7], h, seed=1900))
        kw = dict(cu_seqlens=cu)
    else:
        seqs = make_seqs([40, 40, 40], h, seed=1900)
        x, kw = [torch.stack([s[key] for s in seqs]) for key in KEYS], {}
    ref = recurrent_kda(*x, **kw, **p)[0]
    buf = torch.empty_like(x[2])
    recurrent_kda(*x, output=buf, **kw, **p)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    version = buf._version
    out, state = recurrent_kda(*x, output=buf, **kw, **p)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == before
    assert torch.cuda.memory_allocated() == before
    assert out is buf and state is None and buf._version > version
    assert same_bits(buf, ref)
    assert dict(bi_pool._POOLS) == pools and dict(bi_pool._RETIRED) == retired


@needs_device
@pytest.mark.parametrize("h", [1, 3, 5, 9, 10])
def test_head_counts_off_a_multiple_of_eight(h):
    p = params(h)
    seqs = make_seqs([70, 16, 5, 33], h, seed=3100)
    for s, rows in zip(seqs, stateless(seqs, p), strict=True):
        assert same_bits(rows, pool_prefill(s, p))


@needs_device
def test_many_sequences_in_one_call():
    h = 8
    p = params(h)
    lens = [1 + (37 * i) % 61 for i in range(300)]
    seqs = make_seqs(lens, h, seed=3300)
    got = stateless(seqs, p)
    for i in (0, 1, 17, 150, 298, 299):
        ref = stateless([seqs[i]], p)[0]
        assert same_bits(got[i], ref), f"sequence {i} of length {lens[i]}"
    x, cu = packed(seqs)
    pool = torch.zeros(len(seqs), h, D, D, device="cuda")
    ref = recurrent_kda(*x, initial_state=pool, cu_seqlens=cu, **p)[0]
    torch.cuda.synchronize()
    assert same_bits(torch.cat(got, 0), ref[0])


@needs_device
def test_scale_and_parameter_forms():
    h = 8
    p = params(h)
    s = make_seq(90, h, seed=3500)
    x, cu = packed([s])
    pool = torch.zeros(1, h, D, D, device="cuda")
    ref = recurrent_kda(*x, initial_state=pool, cu_seqlens=cu, scale=0.25, **p)[0]
    assert same_bits(recurrent_kda(*x, cu_seqlens=cu, scale=0.25, **p)[0], ref)
    assert not same_bits(recurrent_kda(*x, cu_seqlens=cu, **p)[0], ref)
    ref = recurrent_kda(*x, cu_seqlens=cu, **p)[0]
    flat = dict(p, dt_bias=p["dt_bias"].reshape(-1))
    assert same_bits(recurrent_kda(*x, cu_seqlens=cu, **flat)[0], ref)
    wide = torch.stack([p["A_log"], p["A_log"]], 1)[:, 0]
    assert not wide.is_contiguous()
    assert same_bits(recurrent_kda(*x, cu_seqlens=cu, **dict(p, A_log=wide))[0], ref)
    with torch.inference_mode():
        buf = torch.empty_like(x[2])
        assert same_bits(recurrent_kda(*x, cu_seqlens=cu, output=buf, **p)[0], ref)
    with torch.cuda.stream(torch.cuda.Stream()):
        got = recurrent_kda(*x, cu_seqlens=cu, **p)[0]
        torch.cuda.synchronize()
    assert same_bits(got, ref)
    with pytest.raises(BatchInvariantUnsupportedError, match="scale"):
        recurrent_kda(*x, cu_seqlens=cu, scale=float("nan"), **p)
    torch.cuda.synchronize()


@needs_device
def test_rows_carry_no_grad_fn():
    h = 8
    p = params(h)
    seqs = make_seqs([40, 21], h, seed=2100)
    x, cu = packed(seqs)
    ref = recurrent_kda(*x, cu_seqlens=cu, **p)[0]
    leaves = [t.clone().requires_grad_(True) for t in x]
    p_grad = dict(p)
    p_grad["A_log"] = p["A_log"].clone().requires_grad_(True)
    p_grad["dt_bias"] = p["dt_bias"].clone().requires_grad_(True)
    out = recurrent_kda(*leaves, cu_seqlens=cu, **p_grad)[0]
    torch.cuda.synchronize()
    assert out.grad_fn is None and not out.requires_grad
    assert same_bits(out, ref)
    with torch.no_grad():
        assert same_bits(recurrent_kda(*leaves, cu_seqlens=cu, **p_grad)[0], ref)


@needs_device
def test_strided_and_misaligned_inputs():
    h = 12
    p = params(h)
    seqs = make_seqs([45, 18], h, seed=2300)
    x, cu = packed(seqs)
    ref = recurrent_kda(*x, cu_seqlens=cu, **p)[0]
    wide = [torch.cat([t, t], dim=2)[:, :, :h] for t in x]
    assert not wide[0].is_contiguous()
    assert same_bits(recurrent_kda(*wide, cu_seqlens=cu, **p)[0], ref)
    shifted = [torch.cat([t[:, :1], t], dim=1)[:, 1:] for t in x]
    assert shifted[4].data_ptr() % 16
    assert same_bits(recurrent_kda(*shifted, cu_seqlens=cu, **p)[0], ref)
    torch.cuda.synchronize()


@needs_device
def test_pool_arguments_are_refused_without_a_pool():
    h = 8
    p = params(h)
    seqs = make_seqs([40, 21], h, seed=2500)
    x, cu = packed(seqs)
    slots = torch.arange(2, dtype=torch.int32, device="cuda")
    like = torch.empty_like(x[2])
    cases = [
        (dict(ssm_state_indices=slots), "ssm_state_indices without initial_state"),
        (dict(output_final_state=True), "output_final_state without initial_state"),
        (dict(output=like.float()), "an output of another shape, dtype or layout"),
        (dict(output=like[:, :-1]), "an output of another shape, dtype or layout"),
        (
            dict(output=torch.cat([like, like], 2)[:, :, :h]),
            "an output of another shape, dtype or layout",
        ),
        (dict(output=like.requires_grad_(True)), "an output that requires grad"),
        (dict(output=x[2]), "shares memory with an input"),
    ]
    for kw, message in cases:
        with pytest.raises(BatchInvariantUnsupportedError, match=message):
            recurrent_kda(*x, cu_seqlens=cu, **p, **kw)
    for bad in ([0, 40, 40, 61], [0, 50, 40, 61], [1, 40, 61], [0, 40, 60]):
        bad_cu = torch.tensor(bad, dtype=torch.int64, device="cuda")
        with pytest.raises(BatchInvariantUnsupportedError, match="cu_seqlens"):
            recurrent_kda(*x, cu_seqlens=bad_cu, **p)
    two = [torch.cat([t, t], 0) for t in x]
    with pytest.raises(BatchInvariantUnsupportedError, match="cu_seqlens with a batch"):
        recurrent_kda(*two, cu_seqlens=cu, **p)
    with pytest.raises(BatchInvariantUnsupportedError, match="11 heads"):
        xo, cuo = packed(make_seqs([20], 11, seed=1))
        recurrent_kda(*xo, cu_seqlens=cuo, **params(11))
    torch.cuda.synchronize()


@needs_device
def test_refused_inside_a_graph_capture():
    h = 8
    p = params(h)
    seqs = make_seqs([40], h, seed=2700)
    x, cu = packed(seqs)
    buf = torch.empty_like(x[2])
    recurrent_kda(*x, cu_seqlens=cu, output=buf, **p)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(BatchInvariantUnsupportedError), torch.cuda.graph(graph):
        recurrent_kda(*x, cu_seqlens=cu, output=buf, **p)


@needs_device
def test_stateless_call_leaves_pool_calls_alone():
    """A stateless call between the decode steps of a pool changes neither the pool nor the steps' rows."""
    h = 8
    p = params(h)
    s = make_seq(60, h, seed=2900)
    ref = pool_prefill_then_decode(s, p, 20)
    pool = torch.zeros(1, h, D, D, device="cuda")
    x = [s[key][None, :20].contiguous() for key in KEYS]
    cu = torch.tensor([0, 20], dtype=torch.int64, device="cuda")
    rows = [recurrent_kda(*x, initial_state=pool, cu_seqlens=cu, **p)[0][0]]
    for t in range(20, 60):
        assert same_bits(stateless([s, make_seq(33, h, 6)], p)[0], ref)
        one = [s[key][None, t : t + 1].contiguous() for key in KEYS]
        rows.append(recurrent_kda(*one, initial_state=pool, **p)[0][0])
    torch.cuda.synchronize()
    assert same_bits(torch.cat(rows, 0), ref)
    assert same_bits(stateless([s], p)[0], ref)
