# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The stateless call of the GDN class ``gdn.bf16.c64``: ``chunk_gated_delta_rule(batch_invariant=True)`` without
``initial_state``, the form of a trainer's forward pass.

Every comparison is ``torch.equal`` against the pool calls of the class (prefill, chunked prefill and decode).
"""

import gc

import pytest
import torch

import flashinfer
from flashinfer.batch_invariant import BatchInvariantUnsupportedError
from flashinfer.batch_invariant.gdn import pool as gdn_pool
from flashinfer.batch_invariant.gdn import training_state_rows, training_state_slots
from flashinfer.batch_invariant.gdn import spec
from flashinfer.utils import get_compute_capability

D = 128
CHUNK = spec.CHUNK
HEADS = [(16, 48), (4, 8), (16, 16)]
SCALE = D**-0.5


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


def _inputs(L, HQ, HV, seed, steep=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    q, k, v = (randn(L, h, D).to(torch.bfloat16) for h in (HQ, HQ, HV))
    g = -torch.nn.functional.softplus(randn(L, HV) - (0.0 if steep else 6.0))
    beta = torch.sigmoid(randn(L, HV))
    return q, k, v, g, beta


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def _offsets(lens):
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)
    return offs


def _cut(x, a, b):
    return tuple(t[a:b].contiguous() for t in x)


def _gather(x, offs, order):
    return tuple(torch.cat([t[offs[i] : offs[i + 1]] for i in order]) for t in x)


_KW = dict(use_qk_l2norm_in_kernel=True, gate_domain="log", batch_invariant=True)


def _stateless(x, lens, **kwargs):
    return flashinfer.chunk_gated_delta_rule(
        *x, cu_seqlens=_i32(_offsets(lens)), **_KW, **kwargs
    )


def _infer(x, slots, lens=None, prefix=None, pool=None):
    HV = x[2].shape[1]
    if pool is None:
        pool = torch.zeros(max(slots) + 1, HV, D, D, device="cuda")
    return flashinfer.chunk_gated_delta_rule(
        *x,
        initial_state=pool,
        state_indices=_i32(slots),
        cu_seqlens=None if lens is None else _i32(_offsets(lens)),
        prefix_lens=None if prefix is None else _i32(prefix),
        **_KW,
    )


def _recurrence(x, lens):
    """The class's recurrence token by token in fp64 torch."""
    q, k, v, g, beta = (t.double() for t in x)
    HV = v.shape[1]
    rep = HV // q.shape[1]

    def normalize(t):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6)

    qn = normalize(q).repeat_interleave(rep, 1)
    kn = normalize(k).repeat_interleave(rep, 1)
    rows = []
    offs = _offsets(lens)
    for a, b in zip(offs, offs[1:], strict=False):
        S = torch.zeros(HV, D, D, dtype=torch.float64, device=q.device)
        for t in range(a, b):
            S = g[t].exp()[:, None, None] * S
            err = v[t] - (S @ kn[t][:, :, None])[..., 0]
            S = S + (beta[t][:, None] * err)[:, :, None] * kn[t][:, None, :]
            rows.append(SCALE * (S @ qn[t][:, :, None])[..., 0])
    return torch.stack(rows)


@needs_device
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("steep", [False, True])
@pytest.mark.parametrize(
    "lens",
    [[1], [63], [64], [65], [200], [7, 64, 130, 1, 129, 320], [1000, 3, 128, 191]],
)
def test_stateless_rows_are_the_pool_prefill_rows(heads, steep, lens):
    HQ, HV = heads
    offs = _offsets(lens)
    x = _inputs(offs[-1], HQ, HV, 11, steep)
    out = _stateless(x, lens)
    assert torch.equal(out, _infer(x, list(range(len(lens))), lens))
    for i in range(len(lens)):
        alone = _infer(_cut(x, offs[i], offs[i + 1]), [0], [lens[i]])
        assert torch.equal(out[offs[i] : offs[i + 1]], alone)


@needs_device
@pytest.mark.parametrize("heads", HEADS[:2])
def test_batch_and_position_do_not_enter_the_bits(heads):
    HQ, HV = heads
    lens = [150, 64, 9, 257, 65, 128]
    offs = _offsets(lens)
    x = _inputs(offs[-1], HQ, HV, 3)
    out = _stateless(x, lens)
    for order in ([5, 4, 3, 2, 1, 0], [3], [2, 0], [1, 5, 3]):
        sub = [lens[i] for i in order]
        o = _stateless(_gather(x, offs, order), sub)
        so = _offsets(sub)
        for j, i in enumerate(order):
            assert torch.equal(o[so[j] : so[j + 1]], out[offs[i] : offs[i + 1]])


@needs_device
@pytest.mark.parametrize("heads", HEADS[:2])
@pytest.mark.parametrize("L,first", [(150, 70), (129, 64), (200, 1), (90, 89)])
def test_stateless_rows_are_chunked_prefill_plus_decode(heads, L, first):
    HQ, HV = heads
    x = _inputs(L, HQ, HV, 5)
    out = _stateless(x, [L])
    pool = torch.zeros(2, HV, D, D, device="cuda")
    pieces = [_infer(_cut(x, 0, first), [1], [first], prefix=[0], pool=pool)]
    mid = min(L, first + 37)
    if mid > first:
        pieces.append(
            _infer(_cut(x, first, mid), [1], [mid - first], prefix=[first], pool=pool)
        )
    for p in range(mid, L):
        pieces.append(_infer(_cut(x, p, p + 1), [1], pool=pool))
    assert torch.equal(out, torch.cat(pieces))


@needs_device
def test_rows_follow_the_recurrence():
    lens = [130, 64, 7, 70]
    x = _inputs(sum(lens), 4, 8, 0)
    out = _stateless(x, lens).double()
    ref = _recurrence(x, lens)
    assert ((out - ref).norm() / ref.norm()).item() < 1e-2


@needs_device
def test_single_sequence_without_offsets():
    x = _inputs(200, 4, 8, 2)
    out = flashinfer.chunk_gated_delta_rule(*x, **_KW)
    assert torch.equal(out, _stateless(x, [200]))
    host = flashinfer.chunk_gated_delta_rule(*x, cu_seqlens_cpu=[0, 70, 200], **_KW)
    assert torch.equal(host, _stateless(x, [70, 130]))


@needs_device
def test_inputs_that_require_grad_give_plain_rows():
    lens = [100, 65]
    x = _inputs(sum(lens), 4, 8, 4)
    ref = _stateless(x, lens)
    xg = [t.clone().requires_grad_(True) for t in x]
    out = _stateless(xg, lens)
    assert out.grad_fn is None and not out.requires_grad
    assert torch.equal(out, ref)
    with torch.no_grad():
        assert torch.equal(_stateless(xg, lens), ref)


@needs_device
def test_output_buffer_no_pool_and_nothing_kept():
    HQ, HV = 4, 8
    lens = [100, 65, 300]
    x = _inputs(sum(lens), HQ, HV, 6)
    cu = _i32(_offsets(lens))
    ref = _stateless(x, lens)
    out = torch.full_like(x[2], float("nan"))
    gc.collect()
    pools = set(gdn_pool._POOLS)
    got = flashinfer.chunk_gated_delta_rule(*x, cu_seqlens=cu, output=out, **_KW)
    assert got is out and torch.equal(out, ref)
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    flashinfer.chunk_gated_delta_rule(*x, cu_seqlens=cu, output=out, **_KW)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= held
    transient = torch.cuda.max_memory_allocated() - held
    assert transient <= 2 * (x[0].nbytes + x[1].nbytes) + (1 << 20)
    assert set(gdn_pool._POOLS) <= pools
    assert torch.equal(out, ref)
    state, _ = gdn_pool._ZERO[(out.device, HV)]
    assert torch.count_nonzero(state) == 0


@needs_device
def test_more_sequences_than_the_cached_index_rows():
    HQ, HV = 4, 8
    for lens in ([3, 70, 1] * 34, [5, 64]):
        x = _inputs(sum(lens), HQ, HV, 9)
        out = _stateless(x, lens)
        assert torch.equal(out, _infer(x, list(range(len(lens))), lens))


@needs_device
def test_a_bad_sequence_stays_local():
    HQ, HV = 4, 8
    lens = [70, 130, 64]
    offs = _offsets(lens)
    x = _inputs(offs[-1], HQ, HV, 8)
    ref = _stateless(x, lens)
    bad = tuple(t.clone() for t in x)
    bad[2][offs[1] + 3] = float("nan")
    out = _stateless(bad, lens)
    assert torch.equal(out[: offs[1]], ref[: offs[1]])
    assert torch.equal(out[offs[2] :], ref[offs[2] :])
    assert out[offs[1] + 3 : offs[2]].isnan().any()


@needs_device
def test_unsupported_arguments_raise():
    HQ, HV = 4, 8
    x = _inputs(8, HQ, HV, 0)
    pool = torch.zeros(1, HV, D, D, device="cuda")
    bad = [
        dict(state_indices=_i32([0])),
        dict(prefix_lens=_i32([0])),
        dict(output_state=pool),
        dict(output_final_state=True),
        dict(output=torch.empty(8, HV, D, device="cuda")),
        dict(output=torch.empty_like(x[2]).requires_grad_(True)),
        dict(cu_seqlens=_i32([0, 4, 4, 8])),
        dict(cu_seqlens=_i32([0, 9])),
        dict(cu_seqlens=_i32([0, 4, 8]), cu_seqlens_cpu=[0, 8]),
    ]
    for kwargs in bad:
        with pytest.raises(BatchInvariantUnsupportedError):
            flashinfer.chunk_gated_delta_rule(*x, **_KW, **kwargs)
    assert torch.count_nonzero(pool) == 0


def _with_states(x, lens, extra=0, **kwargs):
    HV = x[2].shape[1]
    states = torch.full(
        (training_state_slots(lens) + extra, HV, D, D), 7.0, device="cuda"
    )
    return _stateless(x, lens, train_states=states, **kwargs), states


def test_training_state_layout():
    assert [training_state_slots([n]) for n in (1, 64, 65, 128, 129)] == [1, 1, 2, 2, 3]
    assert training_state_slots([64, 65, 200, 1]) == 8
    assert training_state_rows([64, 65, 200, 1]) == [[0], [1, 2], [3, 4, 5, 6], [7]]


@needs_device
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("steep", [False, True])
@pytest.mark.parametrize("lens", [[65], [128], [200], [7, 64, 130, 1, 129, 320]])
def test_train_states_are_the_pool_states_each_chunk_starts_from(heads, steep, lens):
    HQ, HV = heads
    offs = _offsets(lens)
    x = _inputs(offs[-1], HQ, HV, 21, steep)
    out, states = _with_states(x, lens, extra=1)
    assert torch.equal(out, _stateless(x, lens))
    rows = training_state_rows(lens)
    assert sum(len(r) for r in rows) == training_state_slots(lens)
    assert bool((states[-1] == 7.0).all())
    for i, seq_rows in enumerate(rows):
        for j, row in enumerate(seq_rows):
            n = CHUNK * j
            pool = torch.zeros(1, HV, D, D, device="cuda")
            if n:
                _infer(_cut(x, offs[i], offs[i] + n), [0], [n], pool=pool)
            assert torch.equal(states[row], pool[0])


@needs_device
def test_train_states_do_not_depend_on_the_batch():
    lens = [150, 64, 9, 257, 65, 128]
    offs = _offsets(lens)
    x = _inputs(offs[-1], 4, 8, 5)
    _, states = _with_states(x, lens)
    rows = training_state_rows(lens)
    for order in ([5, 4, 3, 2, 1, 0], [3], [2, 0]):
        sub = [lens[i] for i in order]
        _, st = _with_states(_gather(x, offs, order), sub)
        for sub_rows, i in zip(training_state_rows(sub), order, strict=True):
            assert torch.equal(st[sub_rows], states[rows[i]])


@needs_device
def test_a_chunk_follows_from_its_stored_state():
    lens = [200, 130]
    offs = _offsets(lens)
    x = _inputs(offs[-1], 4, 8, 9)
    out, states = _with_states(x, lens)
    q, k, v, g, beta = (t.double() for t in x)

    def normalize(t):
        return t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6)

    qn, kn = (normalize(t).repeat_interleave(2, 1) for t in (q, k))
    for i, seq_rows in enumerate(training_state_rows(lens)):
        for j, row in enumerate(seq_rows):
            S = states[row].double()
            a = offs[i] + CHUNK * j
            for t in range(a, min(a + CHUNK, offs[i + 1])):
                S = g[t].exp()[:, None, None] * S
                err = v[t] - (S @ kn[t][:, :, None])[..., 0]
                S = S + (beta[t][:, None] * err)[:, :, None] * kn[t][:, None, :]
                ref = SCALE * (S @ qn[t][:, :, None])[..., 0]
                assert (out[t].double() - ref).abs().max().item() < 1e-2


@needs_device
def test_train_states_single_chunk_and_version():
    x = _inputs(40, 4, 8, 2)
    out, states = _with_states(x, [40])
    assert torch.equal(out, _stateless(x, [40]))
    assert bool((states == 0.0).all())
    x = _inputs(130, 4, 8, 2)
    states = torch.zeros(3, 8, D, D, device="cuda")
    before = states._version
    _stateless(x, [130], train_states=states)
    assert states._version > before


@needs_device
@pytest.mark.parametrize("lens", [[64], [64, 64], [1, 64, 3]])
def test_single_chunk_sequences_store_their_zero_state(lens):
    x = _inputs(sum(lens), 4, 8, 4)
    out, states = _with_states(x, lens, extra=2)
    assert torch.equal(out, _stateless(x, lens))
    assert bool((states[: len(lens)] == 0.0).all())
    assert bool((states[len(lens) :] == 7.0).all())


@needs_device
def test_train_states_with_host_offsets_and_an_output_buffer():
    lens = [130, 64, 200]
    offs = _offsets(lens)
    x = _inputs(offs[-1], 4, 8, 6)
    ref, ref_states = _with_states(x, lens)
    out = torch.empty_like(x[2])
    states = torch.zeros_like(ref_states)
    got = flashinfer.chunk_gated_delta_rule(
        *x, cu_seqlens_cpu=offs, output=out, train_states=states, **_KW
    )
    assert got is out and torch.equal(out, ref) and torch.equal(states, ref_states)


@needs_device
def test_train_states_unsupported_arguments_raise():
    x = _inputs(130, 4, 8, 1)
    good = torch.zeros(3, 8, D, D, device="cuda")
    for bad in (
        torch.zeros(2, 8, D, D, device="cuda"),
        torch.zeros(3, 4, D, D, device="cuda"),
        torch.zeros(3, 8, D, D, device="cuda", dtype=torch.bfloat16),
        torch.zeros(3, 8, D, D),
        torch.zeros(3, 8, D, D, device="cuda", requires_grad=True),
        torch.zeros(3, 8, D, 2 * D, device="cuda")[..., ::2],
    ):
        with pytest.raises(BatchInvariantUnsupportedError):
            _stateless(x, [130], train_states=bad)
    with pytest.raises(BatchInvariantUnsupportedError):
        flashinfer.chunk_gated_delta_rule(
            *x,
            initial_state=torch.zeros(1, 8, D, D, device="cuda"),
            state_indices=_i32([0]),
            train_states=good,
            **_KW,
        )
    with pytest.raises(ValueError):
        flashinfer.chunk_gated_delta_rule(
            *x, use_qk_l2norm_in_kernel=True, gate_domain="log", train_states=good
        )
    with pytest.raises(BatchInvariantUnsupportedError):
        _stateless(x, [130], train_states=[good])
    with torch.inference_mode():
        made_in_inference_mode = torch.zeros(3, 8, D, D, device="cuda")
    with pytest.raises(BatchInvariantUnsupportedError):
        _stateless(x, [130], train_states=made_in_inference_mode)
    pool = torch.zeros(3, 8, D, D, device="cuda")
    _infer(x, [0], [130], pool=pool)
    with pytest.raises(BatchInvariantUnsupportedError):
        _stateless(x, [130], train_states=pool)


@needs_device
def test_train_states_inside_a_pool_are_refused_and_the_pool_is_kept():
    lens = [130, 70]
    x = _inputs(200, 4, 8, 11)
    big = torch.zeros(8, 8, D, D, device="cuda")
    pool = big[1:7]
    _infer(x, [0, 1], lens, pool=pool)
    kept = big.clone()
    for view in (pool[1:], big[2:], big[:5]):
        with pytest.raises(BatchInvariantUnsupportedError, match="state pool"):
            _stateless(x, lens, train_states=view)
    assert torch.equal(big, kept)
    free = torch.zeros(5, 8, D, D, device="cuda")
    _stateless(x, lens, train_states=free)


@needs_device
def test_train_states_misaligned_or_sharing_memory_are_refused():
    lens = [130, 70]
    x = _inputs(200, 4, 8, 12)
    flat = torch.zeros(5 * 8 * D * D + 4, device="cuda")
    for shift in (1, 2, 3):
        with pytest.raises(BatchInvariantUnsupportedError, match="16-byte aligned"):
            _stateless(
                x,
                lens,
                train_states=flat[shift : shift + 5 * 8 * D * D].view(5, 8, D, D),
            )
    states = torch.zeros(5, 8, D, D, device="cuda")
    inside = states.view(-1).view(torch.bfloat16)[: 200 * 8 * D].view(200, 8, D)
    with pytest.raises(BatchInvariantUnsupportedError, match="shares memory"):
        _stateless(x, lens, train_states=states, output=inside)
    v_inside = states.view(-1).view(torch.bfloat16)[-200 * 8 * D :].view(200, 8, D)
    with pytest.raises(BatchInvariantUnsupportedError, match="shares memory"):
        _stateless((x[0], x[1], v_inside, x[3], x[4]), lens, train_states=states)
    shifted = torch.zeros(6, 8, D, D, device="cuda")[1:]
    _stateless(x, lens, train_states=shifted)


def _infer_out(x, lens, out):
    pool = torch.zeros(len(lens), x[2].shape[1], D, D, device="cuda")
    return flashinfer.chunk_gated_delta_rule(
        *x,
        initial_state=pool,
        state_indices=_i32(list(range(len(lens)))),
        cu_seqlens=_i32(_offsets(lens)),
        output=out,
        **_KW,
    )


def _inside(holder, like, at=0):
    flat = holder.view(-1).view(torch.uint8)[at : at + like.nbytes]
    return flat.view(like.dtype).view(like.shape).copy_(like)


@needs_device
def test_an_output_sharing_memory_is_refused_before_anything_is_written():
    lens = [130, 70]
    x = _inputs(200, 8, 8, 14)
    ref = _stateless(x, lens)
    cu = _i32(_offsets(lens))
    for i in range(5):
        for at in (0, 4096):
            holder = torch.zeros(264, 8, D, dtype=torch.bfloat16, device="cuda")
            xi = list(x)
            xi[i] = _inside(holder, x[i], at)
            with pytest.raises(BatchInvariantUnsupportedError, match="shares memory"):
                _stateless(xi, lens, output=holder[:200])
            assert torch.equal(xi[i], x[i])
    holder = torch.zeros(264, 8, D, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(BatchInvariantUnsupportedError, match="shares memory"):
        flashinfer.chunk_gated_delta_rule(
            *x, cu_seqlens=_inside(holder, cu, 64), output=holder[:200], **_KW
        )
    pool = torch.zeros(4, 8, D, D, device="cuda")
    _infer(x, [0, 1], lens, pool=pool)
    in_pool = _inside(pool, x[2], 1 << 16)
    kept = pool.clone()
    with pytest.raises(BatchInvariantUnsupportedError, match="state pool"):
        _stateless(x, lens, output=in_pool)
    assert torch.equal(pool, kept)
    with torch.inference_mode():
        made_in_inference_mode = torch.empty_like(x[2])
    with pytest.raises(BatchInvariantUnsupportedError, match="inference_mode"):
        _stateless(x, lens, output=made_in_inference_mode)
    out = torch.empty_like(x[2])
    version = out._version
    assert torch.equal(_stateless(x, lens, output=out), ref) and out._version > version
    states = torch.zeros(5, 8, D, D, device="cuda")
    version = out._version
    _stateless(x, lens, output=out, train_states=states)
    assert out._version > version
    arena = torch.zeros(3 * x[0].numel(), dtype=torch.bfloat16, device="cuda")
    shared = [_inside(arena, t, i * t.nbytes) for i, t in enumerate(x[:2])]
    tail = arena[2 * x[0].numel() :].view(200, 8, D)
    got = _stateless((shared[0], shared[1], *x[2:]), lens, output=tail)
    assert got is tail and torch.equal(got, ref)
    with torch.inference_mode():
        inside = _stateless(x, lens, output=torch.empty_like(x[2]))
    assert torch.equal(inside, ref)
    assert torch.equal(_infer_out(x, lens, made_in_inference_mode), ref)


@needs_device
def test_rows_follow_the_host_offsets_when_given():
    host = [0, 130, 200, 264]
    x = _inputs(264, 4, 8, 15)
    ref = _stateless(x, [130, 70, 64])
    for device in ([0, 64, 200, 264], [0, 100, 150, 264], [0, 1, 2, 264]):
        big = torch.full((328, 8, D), 5.0, dtype=torch.bfloat16, device="cuda")
        got = flashinfer.chunk_gated_delta_rule(
            *x, cu_seqlens=_i32(device), cu_seqlens_cpu=host, output=big[32:296], **_KW
        )
        assert torch.equal(got, ref)
        assert bool((big[:32] == 5).all()) and bool((big[296:] == 5).all())
    strided = _i32([0, 9, 130, 9, 200, 9, 264])[::2]
    assert torch.equal(
        flashinfer.chunk_gated_delta_rule(*x, cu_seqlens=strided, **_KW), ref
    )
    for bad in (
        [0, 129.9, 200, 264],
        torch.tensor([0.0, 130.0, 200.0, 264.0]),
        [[0, 264]],
        [False, 264],
    ):
        with pytest.raises(BatchInvariantUnsupportedError, match="1-D integers"):
            flashinfer.chunk_gated_delta_rule(*x, cu_seqlens_cpu=bad, **_KW)
    with pytest.raises(BatchInvariantUnsupportedError, match="1-D integers"):
        flashinfer.chunk_gated_delta_rule(
            *x, cu_seqlens=torch.tensor([0.0, 264.0], device="cuda"), **_KW
        )
    with pytest.raises(BatchInvariantUnsupportedError, match="malformed"):
        flashinfer.chunk_gated_delta_rule(*x, cu_seqlens=host, **_KW)
    with pytest.raises(BatchInvariantUnsupportedError, match="different lengths"):
        flashinfer.chunk_gated_delta_rule(
            *x, cu_seqlens=_i32([0, 264]), cu_seqlens_cpu=host, **_KW
        )
    for key, same in (
        ("cu_seqlens_cpu", torch.tensor(host)),
        ("cu_seqlens_cpu", torch.tensor(host, dtype=torch.int16)),
        ("cu_seqlens_cpu", tuple(host)),
        ("cu_seqlens", _i32([9] + host)[1:]),
        ("cu_seqlens", torch.tensor(host, device="cuda")),
    ):
        got = flashinfer.chunk_gated_delta_rule(*x, **{key: same}, **_KW)
        assert torch.equal(got, ref)


@needs_device
def test_a_float64_default_dtype_does_not_enter_the_call():
    lens = [130, 70]
    x = _inputs(200, 2, 38, 19)
    ref, ref_states = _with_states(x, lens)
    gdn_pool._ZERO.pop((x[2].device, 38), None)
    torch.set_default_dtype(torch.float64)
    try:
        got = _stateless(x, lens)
        states = torch.zeros_like(ref_states)
        with_states = _stateless(x, lens, train_states=states)
    finally:
        torch.set_default_dtype(torch.float32)
    assert gdn_pool._ZERO[(x[2].device, 38)][0].dtype == torch.float32
    assert torch.equal(got, ref) and torch.equal(with_states, ref)
    assert torch.equal(states, ref_states)
    assert torch.equal(_stateless(x, lens), ref)


@needs_device
def test_repeated_host_offsets_copy_nothing():
    a, b = [0, 130, 200, 264], [0, 64, 200, 264]
    x = _inputs(264, 4, 8, 18)
    refs = {
        tuple(o): _with_states(x, [q - p for p, q in zip(o, o[1:], strict=False)])
        for o in (a, b)
    }
    out = torch.empty_like(x[2])
    states = torch.empty_like(refs[tuple(a)][1])
    for offs in (a, b, a, b):
        got = flashinfer.chunk_gated_delta_rule(
            *x, cu_seqlens_cpu=offs, output=out, **_KW
        )
        assert torch.equal(got, refs[tuple(offs)][0])
        flashinfer.chunk_gated_delta_rule(
            *x, cu_seqlens_cpu=offs, output=out, train_states=states, **_KW
        )
        assert torch.equal(out, refs[tuple(offs)][0])
        want = refs[tuple(offs)][1]
        assert torch.equal(states[: want.shape[0]], want)
    torch.cuda.synchronize()
    held, kept = torch.cuda.memory_allocated(), dict(gdn_pool._INDEX)
    assert sum(key[3] == tuple(a) for key in kept) == 2
    for offs in (a, b):
        flashinfer.chunk_gated_delta_rule(*x, cu_seqlens_cpu=offs, output=out, **_KW)
        flashinfer.chunk_gated_delta_rule(
            *x, cu_seqlens_cpu=offs, output=out, train_states=states, **_KW
        )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == held
    assert all(gdn_pool._INDEX[key] is entry for key, entry in kept.items())
    for n in range(1, 40):
        flashinfer.chunk_gated_delta_rule(*_cut(x, 0, n), cu_seqlens_cpu=[0, n], **_KW)
    assert len(gdn_pool._INDEX) <= 16
    assert not any(key[3] == tuple(a) for key in gdn_pool._INDEX)
    flashinfer.chunk_gated_delta_rule(
        *x, cu_seqlens_cpu=a, output=out, train_states=states, **_KW
    )
    assert torch.equal(out, refs[tuple(a)][0])
    assert torch.equal(states[:6], refs[tuple(a)][1])
    got = flashinfer.chunk_gated_delta_rule(*x, cu_seqlens_cpu=iter(a), **_KW)
    assert torch.equal(got, refs[tuple(a)][0])


@needs_device
def test_replaced_index_rows_are_kept():
    x = _inputs(4 * 9001, 2, 34, 16)
    key = (x[2].device, 34)
    gdn_pool._ZERO.pop(key, None)
    small = _stateless(_cut(x, 0, 12), [4, 4, 4])
    first = gdn_pool._ZERO[key][1]
    retired = len(gdn_pool._RETIRED)
    out = _stateless(_cut(x, 0, 36000), [4] * 9000)
    assert first.numel() == 64 and gdn_pool._ZERO[key][1].numel() == 18000
    assert len(gdn_pool._RETIRED) == retired + 1 and gdn_pool._RETIRED[-1] is first
    assert torch.equal(out[:12], small) and bool((first == -1).all())
    _stateless(x, [4] * 9001)
    assert len(gdn_pool._RETIRED) == retired + 1


@needs_device
def test_train_states_follow_the_host_offsets():
    host, device = [0, 130, 200, 264], [0, 64, 200, 264]
    x = _inputs(264, 4, 8, 13)
    ref, ref_states = _with_states(x, [130, 70, 64])
    states = torch.full_like(ref_states, float("nan"))
    got = flashinfer.chunk_gated_delta_rule(
        *x, cu_seqlens=_i32(device), cu_seqlens_cpu=host, train_states=states, **_KW
    )
    assert torch.equal(got, ref) and torch.equal(states, ref_states)


@needs_device
def test_train_states_are_the_series_the_cudnn_backward_takes():
    pytest.importorskip("cudnn")
    gdn = pytest.importorskip("cudnn.linear_attention.ops.gdn")
    if not hasattr(gdn, "gdn_bwd"):
        pytest.skip("this cuDNN frontend has no GDN backward")
    lens = [1, 64, 65, 130, 200]
    offs = _offsets(lens)
    x = _inputs(offs[-1], 4, 8, 13)
    dO = torch.randn(offs[-1], 8, D, device="cuda").to(torch.bfloat16)
    _, states = _with_states(x, lens)
    kw = dict(use_qk_l2norm_in_kernel=True, gate_domain="log")
    # The third output is the checkpoint series. cuDNN Frontend 1.33 appends
    # more outputs after it, so index instead of unpacking.
    own = torch.ops.cudnn.gated_delta_net_fwd(
        *x, _i32(offs), SCALE, checkpoint_every_n_tokens=CHUNK, **kw
    )[2]
    used = states.shape[0]
    assert own.shape[0] >= used and tuple(own.shape[1:]) == tuple(states.shape[1:])
    lead = [rows[0] for rows in training_state_rows(lens)]
    assert bool((states[lead] == 0).all()) and bool((own[lead] == 0).all())
    assert (own[:used].float() - states).abs().max() < 2e-2 * states.abs().max()

    def backward(series):
        return torch.ops.cudnn.gated_delta_net_bwd(
            dO,
            *x,
            _i32(offs),
            SCALE,
            state_checkpoints=series,
            checkpoint_every_n_tokens=CHUNK if series is not None else 0,
            **kw,
        )[:5]

    ref = backward(None)
    for got, want in zip(backward(states.to(x[0].dtype)), ref, strict=True):
        assert (got.float() - want.float()).norm() <= 2e-2 * want.float().norm()
