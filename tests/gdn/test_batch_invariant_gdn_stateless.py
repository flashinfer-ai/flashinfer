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
