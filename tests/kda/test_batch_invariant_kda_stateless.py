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


def chunk_counts(lens):
    return [-(-n // 16) for n in lens]


def checkpointed(seqs, p, **kw):
    x, cu = packed(seqs)
    out, state, ck = recurrent_kda(
        *x, cu_seqlens=cu, checkpoint_every_n_tokens=16, **p, **kw
    )
    torch.cuda.synchronize()
    assert state is None
    return out, ck


@needs_device
@pytest.mark.parametrize(
    "h,lens", [(96, [33, 1, 16, 49]), (12, [33, 1, 16, 250, 15, 64]), (8, [17, 130])]
)
def test_checkpoints_are_the_pool_state_at_every_chunk_boundary(h, lens):
    p = params(h)
    seqs = make_seqs(lens, h, seed=2700)
    rows = torch.cat(stateless(seqs, p), 0)
    total = sum(chunk_counts(lens))
    f32 = torch.full((total, h, D, D), float("nan"), device="cuda")
    out, ck = checkpointed(seqs, p, state_checkpoints=f32)
    assert ck is f32 and same_bits(out[0], rows)
    row = 0
    for s, n in zip(seqs, lens, strict=False):
        for j in range(-(-n // 16)):
            pool = torch.zeros(1, h, D, D, device="cuda")
            if j:
                x = [s[key][None, : 16 * j].contiguous() for key in KEYS]
                cu = torch.tensor([0, 16 * j], dtype=torch.int64, device="cuda")
                recurrent_kda(*x, initial_state=pool, cu_seqlens=cu, **p)
            assert same_bits(pool[0], f32[row]), (n, j)
            row += 1
    out, ck = checkpointed(seqs, p)
    assert ck.dtype == torch.bfloat16 and ck.shape == f32.shape
    assert same_bits(out[0], rows) and same_bits(ck, f32.to(torch.bfloat16))


@needs_device
def test_checkpoints_do_not_depend_on_the_batch():
    h = 8
    p = params(h)
    lens = [40, 7, 64, 100]
    seqs = make_seqs(lens, h, seed=2800)
    _, ck = checkpointed(seqs, p)
    parts = ck.split(chunk_counts(lens))
    for i in (0, 2, 3):
        assert same_bits(checkpointed([seqs[i]], p)[1], parts[i])
    _, rev = checkpointed(seqs[::-1], p)
    for a, b in zip(rev.split(chunk_counts(lens[::-1])), parts[::-1], strict=False):
        assert same_bits(a, b)
    same = make_seqs([48, 48, 48], h, seed=2810)
    x = [torch.stack([s[key] for s in same]) for key in KEYS]
    out, state, fixed = recurrent_kda(*x, checkpoint_every_n_tokens=16, **p)
    assert state is None and same_bits(fixed, checkpointed(same, p)[1])


@needs_device
def test_checkpointed_repeat_allocates_nothing_and_writes_only_its_rows():
    h = 12
    p = params(h)
    lens = [100, 33, 7]
    x, cu = packed(make_seqs(lens, h, seed=2900))
    counts = chunk_counts(lens)
    total = sum(counts)
    starts = torch.tensor(
        [0] + list(torch.tensor(counts).cumsum(0)), dtype=torch.int64, device="cuda"
    )
    ref_rows, _, ref = recurrent_kda(
        *x, cu_seqlens=cu, checkpoint_every_n_tokens=16, **p
    )
    buf = torch.empty_like(x[2])
    ck = torch.full((total + 2, h, D, D), 7.0, dtype=torch.bfloat16, device="cuda")
    kw = dict(
        cu_seqlens=cu,
        output=buf,
        state_checkpoints=ck,
        checkpoint_cu_starts=starts,
        checkpoint_every_n_tokens=16,
    )
    recurrent_kda(*x, **kw, **p)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    version = ck._version
    out, state, got = recurrent_kda(*x, **kw, **p)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() == before
    assert out is buf and state is None and got is ck and ck._version > version
    assert same_bits(buf, ref_rows) and same_bits(ck[:total], ref)
    assert bool((ck[total:] == 7.0).all())


@needs_device
def test_checkpoint_arguments_are_refused_outside_their_form():
    h = 8
    p = params(h)
    lens = [40, 21]
    x, cu = packed(make_seqs(lens, h, seed=3000))
    total = sum(chunk_counts(lens))
    ck = torch.empty(total, h, D, D, dtype=torch.bfloat16, device="cuda")
    starts = torch.tensor([0, 3, 5], dtype=torch.int64, device="cuda")
    shared = torch.empty(total * h * D * D, dtype=torch.bfloat16, device="cuda")
    cases = [
        (dict(checkpoint_every_n_tokens=32), "checkpoint_every_n_tokens=32"),
        (dict(checkpoint_every_n_tokens=64), "checkpoint_every_n_tokens=64"),
        (dict(checkpoint_every_n_tokens=8), "checkpoint_every_n_tokens=8"),
        (dict(checkpoint_every_n_tokens=16.0), "checkpoint_every_n_tokens=16.0"),
        (
            dict(checkpoint_every_n_tokens=16, state_checkpoints=ck.cpu()),
            "state_checkpoints of another shape",
        ),
        (
            dict(checkpoint_every_n_tokens=16, state_checkpoints=ck[:, 0]),
            "state_checkpoints of another shape",
        ),
        (
            dict(checkpoint_every_n_tokens=16, checkpoint_cu_starts=starts[:-1]),
            "checkpoint_cu_starts",
        ),
        (
            dict(checkpoint_every_n_tokens=16, checkpoint_cu_starts=starts.cpu()),
            "checkpoint_cu_starts",
        ),
        (dict(state_checkpoints=ck), "without checkpoint_every_n_tokens"),
        (dict(checkpoint_cu_starts=starts), "without checkpoint_every_n_tokens"),
        (
            dict(checkpoint_every_n_tokens=16, state_checkpoints=ck.double()),
            "state_checkpoints of another shape",
        ),
        (
            dict(checkpoint_every_n_tokens=16, state_checkpoints=ck[:-1]),
            "state_checkpoints of another shape",
        ),
        (
            dict(checkpoint_every_n_tokens=16, state_checkpoints=ck[:, :4]),
            "state_checkpoints of another shape",
        ),
        (
            dict(
                checkpoint_every_n_tokens=16,
                state_checkpoints=ck.clone().requires_grad_(True),
            ),
            "state_checkpoints of another shape",
        ),
        (
            dict(
                checkpoint_every_n_tokens=16,
                state_checkpoints=shared.view(total, h, D, D),
                output=shared[: x[2].numel()].view(x[2].shape),
            ),
            "state_checkpoints that share memory",
        ),
        (
            dict(checkpoint_every_n_tokens=16, checkpoint_cu_starts=starts + 1),
            "checkpoint_cu_starts",
        ),
        (
            dict(checkpoint_every_n_tokens=16, checkpoint_cu_starts=starts.int()),
            "checkpoint_cu_starts",
        ),
        (
            dict(
                checkpoint_every_n_tokens=16,
                checkpoint_state_indices=torch.arange(
                    total, dtype=torch.int32, device="cuda"
                ),
            ),
            "checkpoint_state_indices",
        ),
    ]
    for kw, message in cases:
        with pytest.raises(BatchInvariantUnsupportedError, match=message):
            recurrent_kda(*x, cu_seqlens=cu, **p, **kw)
    v = shared[-x[2].numel() :].view(x[2].shape).copy_(x[2])
    with pytest.raises(BatchInvariantUnsupportedError, match="share memory"):
        recurrent_kda(
            x[0],
            x[1],
            v,
            *x[3:],
            cu_seqlens=cu,
            checkpoint_every_n_tokens=16,
            state_checkpoints=shared.view(total, h, D, D),
            **p,
        )
    good = dict(checkpoint_every_n_tokens=16, checkpoint_cu_starts=starts)
    for _ in range(2):
        recurrent_kda(*x, cu_seqlens=cu, **p, **good)
    starts[1] += 1
    with pytest.raises(BatchInvariantUnsupportedError, match="checkpoint_cu_starts"):
        recurrent_kda(*x, cu_seqlens=cu, **p, **good)
    starts[1] -= 1
    pool = torch.zeros(2, h, D, D, device="cuda")
    slots = torch.arange(2, dtype=torch.int32, device="cuda")
    for kw in (
        dict(checkpoint_every_n_tokens=16),
        dict(state_checkpoints=ck),
        dict(checkpoint_cu_starts=starts),
    ):
        with pytest.raises(BatchInvariantUnsupportedError, match="with a state pool"):
            recurrent_kda(
                *x,
                cu_seqlens=cu,
                initial_state=pool,
                ssm_state_indices=slots,
                **p,
                **kw,
            )
    torch.cuda.synchronize()
    assert not bool(pool.any())


@needs_device
def test_fixed_layout_on_a_second_stream_while_the_first_is_busy():
    h = 8
    p = params(h)
    busy = torch.randn(6000, 6000, device="cuda")
    warm = make_seqs([33, 33], h, seed=3190)
    recurrent_kda(*[torch.stack([s[key] for s in warm]) for key in KEYS], **p)
    for shape in ([9, 21], [13, 50], [5, 37], [11, 29]):
        seqs = make_seqs([shape[1]] * shape[0], h, seed=3200 + shape[0])
        x = [torch.stack([s[key] for s in seqs]) for key in KEYS]
        ref = torch.stack(stateless(seqs, p))
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(first):
            for _ in range(12):
                busy @ busy
            a = recurrent_kda(*x, **p)[0]
        with torch.cuda.stream(second):
            b = recurrent_kda(*x, **p)[0]
        torch.cuda.synchronize()
        assert same_bits(a, ref) and same_bits(b, ref), shape


@needs_device
def test_buffers_sharing_memory_with_any_argument_are_refused():
    h = 8
    p = params(h)
    lens = [40, 20, 70]
    x, cu = packed(make_seqs(lens, h, seed=3300))
    total = sum(chunk_counts(lens))
    shares = pytest.raises(BatchInvariantUnsupportedError, match="shares? memory")

    def call(*a, **kw):
        return recurrent_kda(*(a or x), cu_seqlens=cu, **{**p, **kw})

    rows = x[2].numel() // h // D
    big = torch.zeros(rows * h * D, dtype=torch.float32, device="cuda")
    bias = big[: h * D].view(h, D).copy_(p["dt_bias"])
    kept = bias.clone()
    with shares:
        call(
            dt_bias=bias,
            output=big.view(torch.bfloat16)[: x[2].numel()].view(x[2].shape),
        )
    with shares:
        call(
            A_log=big[:h],
            output=big.view(torch.bfloat16)[: x[2].numel()].view(x[2].shape),
        )
    assert torch.equal(bias, kept)
    ck = torch.full((total, h, D, D), 3.0, dtype=torch.bfloat16, device="cuda")
    starts = ck.view(-1).view(torch.int64)[: len(lens) + 1]
    starts.copy_(torch.tensor([0, 3, 5, 10], device="cuda"))
    with shares:
        call(
            checkpoint_every_n_tokens=16,
            state_checkpoints=ck,
            checkpoint_cu_starts=starts,
        )
    assert starts.tolist() == [0, 3, 5, 10]
    with shares:
        call(
            checkpoint_every_n_tokens=16,
            state_checkpoints=ck,
            A_log=ck.view(-1).view(torch.float32)[:h],
        )
    spaced = torch.zeros(2, 130, h, D, dtype=torch.bfloat16, device="cuda")
    v = spaced[None, :, 0]
    between = spaced.view(-1)[h * D : h * D + h * D * D].view(1, h, D, D)
    two = [t[:, :2].contiguous() for t in x]
    two[2] = v
    with shares:
        recurrent_kda(
            *two,
            cu_seqlens=torch.tensor([0, 2], device="cuda"),
            checkpoint_every_n_tokens=16,
            state_checkpoints=between,
            **p,
        )
    torch.cuda.synchronize()
    assert not bool(spaced.any())


@needs_device
def test_changed_starts_too_many_sequences_and_inference_buffers_are_refused():
    h = 8
    p = params(h)
    lens = [40, 20, 70]
    x, cu = packed(make_seqs(lens, h, seed=3400))
    total = sum(chunk_counts(lens))
    starts = torch.tensor([0, 3, 5, 10], dtype=torch.int64, device="cuda")
    ck = torch.full((total, h, D, D), 3.0, dtype=torch.bfloat16, device="cuda")
    kw = dict(
        checkpoint_every_n_tokens=16, state_checkpoints=ck, checkpoint_cu_starts=starts
    )
    recurrent_kda(*x, cu_seqlens=cu, **p, **kw)
    torch.cuda.synchronize()
    version, kept = starts._version, ck.clone()
    starts.data[1] = 30
    starts.data[2] = 32
    assert starts._version == version
    with pytest.raises(BatchInvariantUnsupportedError, match="checkpoint_cu_starts"):
        recurrent_kda(*x, cu_seqlens=cu, **p, **kw)
    starts.data.copy_(torch.tensor([0, 3, 5, 10], device="cuda"))

    with torch.inference_mode():
        frozen = torch.empty(x[2].numel() + 1, dtype=torch.bfloat16, device="cuda")
        frozen_ck = torch.empty_like(ck)
    odd = frozen[1:].view(x[2].shape)
    assert odd.data_ptr() % 16
    ck.fill_(3.0)
    with pytest.raises(BatchInvariantUnsupportedError, match="inference tensor"):
        recurrent_kda(*x, cu_seqlens=cu, output=odd, **p, **kw)
    with pytest.raises(BatchInvariantUnsupportedError, match="inference tensor"):
        recurrent_kda(*x, cu_seqlens=cu, **p, **dict(kw, state_checkpoints=frozen_ck))
    torch.cuda.synchronize()
    assert bool((ck == 3.0).all())
    with torch.inference_mode():
        out, _, got = recurrent_kda(
            *x, cu_seqlens=cu, output=odd, **p, **dict(kw, state_checkpoints=frozen_ck)
        )
        torch.cuda.synchronize()
        assert same_bits(got, kept) and out is odd

    p1 = params(1)
    many = 65536
    one = [
        torch.zeros(many, 1, 1, D, dtype=torch.bfloat16, device="cuda")
        for _ in range(4)
    ] + [torch.zeros(many, 1, 1, dtype=torch.bfloat16, device="cuda")]
    with pytest.raises(BatchInvariantUnsupportedError, match="65536 sequences"):
        recurrent_kda(*one, **p1)
    flat = [t.view(1, many, *t.shape[2:]) for t in one]
    offsets = torch.arange(many + 1, dtype=torch.int64, device="cuda")
    with pytest.raises(BatchInvariantUnsupportedError, match="65536 sequences"):
        recurrent_kda(*flat, cu_seqlens=offsets, **p1)


@needs_device
def test_checkpoints_feed_the_cudnn_kda_backward():
    """The cuDNN frontend's KDA backward takes the series in place of its own forward's and returns its gradients."""
    pytest.importorskip("cudnn")
    try:
        from cudnn.linear_attention.ops.kda import kimi_delta_attention

        bwd_op = torch.ops.cudnn.kimi_delta_attention_bwd
    except (ImportError, AttributeError) as e:
        pytest.skip(f"this cuDNN frontend has no KDA backward: {e}")
    h = 8
    p = params(h)
    lens = [33, 1, 16, 250, 15, 64]
    x, cu = packed(make_seqs(lens, h, seed=3100))
    _, _, ours = recurrent_kda(*x, cu_seqlens=cu, checkpoint_every_n_tokens=16, **p)
    q, k, v, g, beta = (t[0] for t in x)
    kw = dict(
        use_qk_l2norm_in_kernel=True,
        batch_invariant=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        gate_lower_bound=LB,
        a_log=p["A_log"],
        dt_bias=p["dt_bias"],
    )
    try:
        rows, _, theirs = kimi_delta_attention(
            q, k, v, g, beta, cu, scale=D**-0.5, checkpoint_every_n_tokens=16, **kw
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"the cuDNN KDA engine does not run here: {e}")
    m = ours.shape[0]
    assert theirs.dtype == ours.dtype and theirs.shape[0] >= m
    assert float((theirs[:m].float() - ours.float()).abs().max()) <= 1e-2
    assert float((theirs[:m].float() - ours.roll(1, 0).float()).abs().max()) > 1e-1
    pad = theirs.new_zeros(theirs.shape[0] - m, *ours.shape[1:])
    gen = torch.Generator(device="cuda").manual_seed(5)
    d_out = torch.randn(rows.shape, device="cuda", generator=gen).to(rows.dtype)

    def backward(checkpoints):
        return bwd_op(
            d_out,
            q,
            k,
            v,
            g,
            beta,
            cu,
            D**-0.5,
            state_checkpoints=checkpoints,
            checkpoint_every_n_tokens=16,
            **kw,
        )

    def distances(checkpoints):
        pairs = zip(backward(theirs), backward(checkpoints), strict=False)
        return [
            float((a.float() - b.float()).norm()) / float(a.float().norm())
            for a, b in pairs
            if a.numel()
        ]

    got = distances(torch.cat([ours, pad]))
    print("KDA_BWD relative distances", got)
    assert all(d <= 1e-3 for d in got)
    wrong = distances(torch.cat([ours.roll(1, 0), pad]))
    print("KDA_BWD relative distances, rows shifted by one", wrong)
    assert max(wrong) > 1e-2
