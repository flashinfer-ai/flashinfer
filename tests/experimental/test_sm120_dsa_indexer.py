# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Correctness tests of the experimental SM120 DSA indexer top-k backend
# (dsa_indexer_topk(..., backend="sm120")), ragged (prefill) and paged
# (decode) forms, FP8 and MXFP4.
#
# The inputs are small integers and powers of two, so every product and sum
# of the score is exact in FP32 whatever the accumulation order: the FP64
# reference gives the kernels' score bits, and the selection (score
# descending, then the larger key id) is compared exactly, ties included.

import warnings

import pytest
import torch

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.dsa_indexer import dsa_indexer_topk
from flashinfer.experimental import sm120_dsa_indexer as sm120
from flashinfer.utils import get_compute_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] != 12,
    reason="backend='sm120' requires an SM12x GPU",
)

_D = 128
# e2m1 code -> value (bit 3: sign).
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@pytest.fixture(autouse=True)
def _quiet_experimental():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


class _Keys:
    """Exact-valued keys or queries of one format: ``data`` in the kernel's
    layout, ``scale`` its scales and ``value`` the dequantized FP64 values."""

    def __init__(self, data, scale, value):
        self.data, self.scale, self.value = data, scale, value


def _fp8(shape, gen, dev, with_scale, zero=None, crowd=False):
    # Integers in [-7, 7] are exact in e4m3; per-row scales are powers of two.
    # crowd: 5 to 7 at scale 1 (scores within ~2% of each other).
    lo, hi = (5, 8) if crowd else (-7, 8)
    v = torch.randint(lo, hi, (*shape, _D), generator=gen, device=dev).double()
    if zero is not None:
        v[zero] = 0
    scale = None
    if with_scale:
        scale = torch.exp2(
            torch.randint(
                0 if crowd else -2, 1 if crowd else 2, shape, generator=gen, device=dev
            ).double()
        )
    value = v * scale[..., None] if with_scale else v
    data = v.to(torch.float8_e4m3fn)
    return _Keys(data, scale.float() if with_scale else None, value)


def _fp4(shape, gen, dev, zero=None, crowd=False):
    # Random e2m1 codes, block scales 2^0 or 2^1 (ue8m0 127 or 128). crowd:
    # codes 5 to 7 (3, 4, 6) at scale 1.
    code = torch.randint(
        5 if crowd else 0, 8 if crowd else 16, (*shape, _D), generator=gen, device=dev
    )
    if zero is not None:
        code[zero] = 0
    exp = torch.randint(
        127, 128 if crowd else 129, (*shape, 4), generator=gen, device=dev
    )
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=dev)
    mag = lut[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    blk = torch.exp2(exp.double() - 127).repeat_interleave(32, dim=-1)
    packed = (code[..., 0::2] | (code[..., 1::2] << 4)).to(torch.uint8)
    sf = (exp[..., 0] | exp[..., 1] << 8 | exp[..., 2] << 16 | exp[..., 3] << 24).to(
        torch.int64
    )
    sf = torch.where(sf >= 2**31, sf - 2**32, sf).to(torch.int32)
    return _Keys(packed, sf, mag * blk)


def _keys(fp4, shape, gen, dev, zero_frac=0.0, crowd=False):
    # All-zero keys score 0: long runs of ties.
    z = torch.rand(shape, generator=gen, device=dev) < zero_frac if zero_frac else None
    if fp4:
        return _fp4(shape, gen, dev, z, crowd)
    return _fp8(shape, gen, dev, True, z, crowd)


def _queries(fp4, shape, gen, dev):
    return _fp4(shape, gen, dev) if fp4 else _fp8(shape, gen, dev, False)


def _weights(shape, gen, dev):
    return torch.randint(-2, 3, shape, generator=gen, device=dev).float() * 0.5


def _scores(qv, w, kv):
    """FP64 scores of query rows qv [R, H, D] (weights w [R, H]) against keys
    kv [N, D]: [R, N]."""
    dots = torch.einsum("rhd,nd->rhn", qv, kv).clamp_min(0)
    return torch.einsum("rhn,rh->rn", dots, w.double())


def _check_row(idx, sc, row_scores, top_k, sorted_ids, what):
    """The row's exact top min(top_k, n) of row_scores [n] (score descending,
    then the larger id), then -1."""
    n = row_scores.numel()
    want = min(top_k, n)
    ids = torch.arange(n, device=row_scores.device)
    order = torch.argsort(ids, descending=True, stable=True)
    order = order[torch.argsort(row_scores[order], descending=True, stable=True)]
    ref = torch.sort(order[:want]).values.int()
    got = idx[:want]
    if sorted_ids:
        assert torch.equal(got, ref), f"{what}: ids"
    else:
        assert torch.equal(torch.sort(got).values, ref), f"{what}: ids"
    assert bool((idx[want:] == -1).all()), f"{what}: padding"
    assert torch.equal(sc[:want], row_scores[got.long()].float()), f"{what}: scores"
    assert bool((sc[want:] == -torch.inf).all()), f"{what}: padding scores"


# ---------------------------------------------------------------------------
# Ragged (prefill) form
# ---------------------------------------------------------------------------


def _visible(lq, lk, u, offset, ratio):
    if offset is None:
        offset = lk - lq if ratio == 1 else 0
    return max(0, min(lk, (offset + u + 1) // ratio))


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", [(64, 512), (32, 2048)])
@pytest.mark.parametrize(
    "seg_q,seg_k,offsets,ratio,zero_frac",
    [
        ([300, 1, 77], [9000, 5000, 77], None, 1, 0.0),
        ([200], [30000], None, 1, 0.9),
        ([64, 50], [4000, 20000], [100, -7], 1, 0.0),
        ([128], [3000], [11000], 4, 0.0),
        # Rows past the resident route's 16K keys: the sampled-threshold scan.
        ([96], [120000], None, 1, 0.0),
        # Both routes in one call: short and long segments, a query block
        # straddling them; rows crossing 16K keys within a segment.
        ([6, 64, 33], [5000, 40000, 2000], None, 1, 0.0),
        ([64], [30000], [16384 - 32], 1, 0.0),
        # Resident rows whose k-th score is one value tied past the list
        # (mostly zero keys).
        ([300], [12000], None, 1, 0.95),
        # A fresh chunk: rows of 1-2048 keys, the tail's from registers (and
        # with ties: mostly zero keys).
        ([2048], [2048], None, 1, 0.0),
        ([2048], [2048], None, 1, 0.9),
    ],
)
def test_ragged(fp4, heads, top_k, seg_q, seg_k, offsets, ratio, zero_frac):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(sum(seg_q) + heads + top_k)
    T, Tkv = sum(seg_q), sum(seg_k)
    q = _queries(fp4, (T, heads), gen, dev)
    k = _keys(fp4, (Tkv,), gen, dev, zero_frac)
    w = _weights((T, heads), gen, dev)
    cu_q = torch.tensor([0, *torch.tensor(seg_q).cumsum(0).tolist()], dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(seg_k).cumsum(0).tolist()], dtype=torch.int32)
    offs = (
        None
        if offsets is None
        else torch.tensor(offsets, dtype=torch.int64, device=dev)
    )
    idx, sc = dsa_indexer_topk(
        q.data,
        k.data,
        w,
        cu_q.to(dev),
        cu_k.to(dev),
        top_k=top_k,
        backend="sm120",
        k_scale=k.scale,
        q_scale=q.scale,
        q_causal_offsets=offs,
        ratio=ratio,
    )
    for s, (lq, lk) in enumerate(zip(seg_q, seg_k, strict=True)):
        q0, k0 = int(cu_q[s]), int(cu_k[s])
        full = _scores(q.value[q0 : q0 + lq], w[q0 : q0 + lq], k.value[k0 : k0 + lk])
        for u in range(lq):
            n = _visible(lq, lk, u, None if offsets is None else offsets[s], ratio)
            _check_row(
                idx[q0 + u],
                sc[q0 + u],
                full[u, :n],
                top_k,
                True,
                f"segment {s} row {u}",
            )


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", [(64, 512), (32, 2048)])
@pytest.mark.parametrize(
    "seg_q,seg_k",
    [
        ([300, 1, 77], [9000, 5000, 77]),  # resident, short rows
        ([64, 50], [4000, 20000]),  # resident, long rows
        ([96], [120000]),  # sampled-threshold scan
    ],
)
def test_ragged_indices_only(fp4, heads, top_k, seg_q, seg_k):
    """return_scores=False: the same indices, no scores."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(sum(seg_k) + heads)
    T, Tkv = sum(seg_q), sum(seg_k)
    q = _queries(fp4, (T, heads), gen, dev)
    k = _keys(fp4, (Tkv,), gen, dev)
    w = _weights((T, heads), gen, dev)
    cu_q = torch.tensor([0, *torch.tensor(seg_q).cumsum(0).tolist()], dtype=torch.int32)
    cu_k = torch.tensor([0, *torch.tensor(seg_k).cumsum(0).tolist()], dtype=torch.int32)

    def run(**kw):
        return dsa_indexer_topk(
            q.data,
            k.data,
            w,
            cu_q.to(dev),
            cu_k.to(dev),
            top_k=top_k,
            backend="sm120",
            k_scale=k.scale,
            q_scale=q.scale,
            **kw,
        )

    idx, _ = run()
    idx2, sc2 = run(return_scores=False)
    assert sc2 is None
    assert torch.equal(idx, idx2)
    with pytest.raises(ValueError, match="return_scores"):
        run(scores=torch.empty(T, top_k, device=dev), return_scores=False)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", [(64, 512), (32, 2048)])
def test_ragged_cuda_graph(fp4, heads, top_k, monkeypatch):
    """One capture replays over new segment boundaries of the same buffers:
    each query block takes its route (resident or scan) on the device. Scan
    chunks of 64 rows: several chunks per call."""
    from flashinfer.experimental.sm120_dsa_indexer import _api

    monkeypatch.setattr(_api, "_SCAN_ROWS", 64)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(heads + top_k)
    T, Tkv = 200, 60000
    q = _queries(fp4, (T, heads), gen, dev)
    k = _keys(fp4, (Tkv,), gen, dev)
    w = _weights((T, heads), gen, dev)
    cu_q = torch.zeros(4, dtype=torch.int32, device=dev)
    cu_k = torch.zeros(4, dtype=torch.int32, device=dev)
    idx = torch.empty(T, top_k, dtype=torch.int32, device=dev)
    sc = torch.empty(T, top_k, dtype=torch.float32, device=dev)

    def bounds(seg_q, seg_k):
        cu_q.copy_(torch.tensor([0, *torch.tensor(seg_q).cumsum(0).tolist()]))
        cu_k.copy_(torch.tensor([0, *torch.tensor(seg_k).cumsum(0).tolist()]))

    def run():
        dsa_indexer_topk(
            q.data,
            k.data,
            w,
            cu_q,
            cu_k,
            top_k=top_k,
            backend="sm120",
            k_scale=k.scale,
            q_scale=q.scale,
            indices=idx,
            scores=sc,
        )

    bounds([100, 50, 50], [40000, 15000, 5000])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for seg_q, seg_k in (
        ([100, 50, 50], [40000, 15000, 5000]),
        ([10, 180, 10], [2000, 50000, 8000]),
        ([190, 5, 5], [58000, 1000, 1000]),
        ([60, 70, 70], [3000, 9000, 48000]),
    ):
        bounds(seg_q, seg_k)
        graph.replay()
        torch.cuda.synchronize()
        q0 = k0 = 0
        for s_, (lq, lk) in enumerate(zip(seg_q, seg_k, strict=True)):
            full = _scores(
                q.value[q0 : q0 + lq], w[q0 : q0 + lq], k.value[k0 : k0 + lk]
            )
            for u in range(lq):
                n = _visible(lq, lk, u, None, 1)
                _check_row(
                    idx[q0 + u], sc[q0 + u], full[u, :n], top_k, True, f"{s_}/{u}"
                )
            q0, k0 = q0 + lq, k0 + lk


def test_ragged_scratch_bound():
    """The scratch is bounded by the rows, not the keys: a 16K-row call of
    long rows (the scan, in chunks) and short rows (resident) stays within
    ragged_scratch_bytes."""
    from flashinfer.experimental.sm120_dsa_indexer import ragged_scratch_bytes

    dev = torch.device("cuda")
    T, C, heads, top_k = 16384, 16384, 32, 2048
    q = torch.zeros(T, heads, _D, device=dev).to(torch.float8_e4m3fn)
    k = torch.zeros(C + T, _D, device=dev).to(torch.float8_e4m3fn)
    k_scale = torch.ones(C + T, device=dev)
    w = torch.ones(T, heads, device=dev)
    cu_q = torch.tensor([0, T], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, C + T], dtype=torch.int32, device=dev)
    idx = torch.empty(T, top_k, dtype=torch.int32, device=dev)
    offs = torch.tensor([C - 4096], dtype=torch.int64, device=dev)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    dsa_indexer_topk(
        q,
        k,
        w,
        cu_q,
        cu_k,
        top_k=top_k,
        backend="sm120",
        k_scale=k_scale,
        q_causal_offsets=offs,
        indices=idx,
        return_scores=False,
    )
    torch.cuda.synchronize()
    used = torch.cuda.max_memory_allocated() - base
    bound = ragged_scratch_bytes(T, top_k, heads, dev)
    # The caching allocator rounds each of the ~20 tensors up to 512 bytes.
    assert used <= bound + 20 * 512, (used, bound)
    assert bound < 1 << 30


def _ragged_call(fp4, heads, top_k, seg_q, seg_k, seed, **kw):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(seed)
    T, Tkv = sum(seg_q), sum(seg_k)
    q = _queries(fp4, (T, heads), gen, dev)
    k = _keys(fp4, (Tkv,), gen, dev)
    w = _weights((T, heads), gen, dev)
    cu = [
        torch.tensor(
            [0, *torch.tensor(x).cumsum(0).tolist()], dtype=torch.int32, device=dev
        )
        for x in (seg_q, seg_k)
    ]
    idx, sc = dsa_indexer_topk(
        q.data,
        k.data,
        w,
        cu[0],
        cu[1],
        top_k=top_k,
        backend="sm120",
        k_scale=k.scale,
        q_scale=q.scale,
        **kw,
    )
    return q, k, w, idx, sc


@pytest.mark.parametrize("top_k", [1, 1000, 4096])
def test_ragged_top_k(top_k):
    seg_q, seg_k = [64, 32], [5000, 20000]
    q, k, w, idx, sc = _ragged_call(False, 32, top_k, seg_q, seg_k, top_k)
    q0 = k0 = 0
    for s_, (lq, lk) in enumerate(zip(seg_q, seg_k, strict=True)):
        full = _scores(q.value[q0 : q0 + lq], w[q0 : q0 + lq], k.value[k0 : k0 + lk])
        for u in range(lq):
            n = _visible(lq, lk, u, None, 1)
            _check_row(idx[q0 + u], sc[q0 + u], full[u, :n], top_k, True, f"{s_}/{u}")
        q0, k0 = q0 + lq, k0 + lk


def test_ragged_max_seqlen_k():
    """A bound on the segments' keys changes the launches, not the result; a
    bound of 0 (or no keys at all) leaves every row padding."""
    seg_q, seg_k = [64, 32], [5000, 20000]
    _, _, _, idx, sc = _ragged_call(False, 64, 512, seg_q, seg_k, 3)
    _, _, _, idx2, sc2 = _ragged_call(
        False, 64, 512, seg_q, seg_k, 3, max_seqlen_k=20000
    )
    assert torch.equal(idx, idx2) and torch.equal(sc, sc2)
    for kw, seg_k0 in (({"max_seqlen_k": 0}, seg_k), ({}, [0, 0])):
        _, _, _, idx, sc = _ragged_call(False, 64, 512, seg_q, seg_k0, 3, **kw)
        assert bool((idx == -1).all()) and bool((sc == -torch.inf).all())


def test_ragged_validation():
    dev = torch.device("cuda")
    q = torch.zeros(4, 32, _D, device=dev).to(torch.float8_e4m3fn)
    k = torch.zeros(64, _D, device=dev).to(torch.float8_e4m3fn)
    w = torch.zeros(4, 32, device=dev)
    ks = torch.ones(64, device=dev)
    cu_q = torch.tensor([0, 4], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, 64], dtype=torch.int32, device=dev)

    def call(**kw):
        args = dict(top_k=16, backend="sm120", k_scale=ks)
        args.update(kw)
        cq = args.pop("cu_q", cu_q)
        return dsa_indexer_topk(args.pop("q", q), k, w, cq, cu_k, **args)

    with pytest.raises(ValueError, match="top_k"):
        call(top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        call(top_k=4097)
    with pytest.raises(ValueError, match="heads"):
        call(q=torch.zeros(4, 16, _D, device=dev).to(torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="ratio"):
        call(ratio=0)
    with pytest.raises(ValueError, match="softmax_scale"):
        call(softmax_scale=0.1)
    with pytest.raises(ValueError, match="k_scale"):
        call(k_scale=None)
    with pytest.raises(ValueError, match="cu_seqlens"):
        call(cu_q=None)


# ---------------------------------------------------------------------------
# Paged (decode) form
# ---------------------------------------------------------------------------


class _Paged:
    """A paged indexer cache ([num_pages, P, row + 4] bytes: P key rows, then
    their P scales) holding each request's keys at shuffled pages."""

    def __init__(self, fp4, B, max_len, P, gen, dev, zero_frac=0.0, crowd=False):
        row = _D // 2 if fp4 else _D
        blocks = (max_len + P - 1) // P
        num_pages = B * blocks + 3
        self.keys = _keys(fp4, (num_pages * P,), gen, dev, zero_frac, crowd)
        cache = torch.zeros(num_pages, P * (row + 4), dtype=torch.uint8, device=dev)
        cache[:, : P * row] = self.keys.data.view(torch.uint8).reshape(
            num_pages, P * row
        )
        cache[:, P * row :] = self.keys.scale.view(torch.uint8).reshape(
            num_pages, P * 4
        )
        self.cache = cache.view(num_pages, P, row + 4)
        perm = torch.randperm(num_pages, generator=gen, device=dev)[: B * blocks]
        self.table = perm.int().view(B, blocks)
        self.P = P

    def request_keys(self, b, n):
        """FP64 values of request b's first n keys."""
        pages = self.table[b, : (n + self.P - 1) // self.P].long()
        rows = (
            pages[:, None] * self.P + torch.arange(self.P, device=pages.device)
        ).reshape(-1)
        return self.keys.value[rows[:n]]


def _decode_inputs(fp4, B, next_n, heads, lens, gen, dev):
    q = _queries(fp4, (B, next_n, heads), gen, dev)
    w = _weights((B * next_n, heads), gen, dev)
    # Row j of a request sees one key more than row j - 1 (the MTP layout).
    ctx = torch.tensor(
        [[max(0, L - next_n + 1 + j) for j in range(next_n)] for L in lens],
        dtype=torch.int32,
        device=dev,
    )
    return q, w, ctx


def _run_decode(q, w, ctx, paged, top_k, **kw):
    return dsa_indexer_topk(
        q.data,
        paged.cache,
        w,
        None,
        None,
        top_k=top_k,
        backend="sm120",
        q_scale=q.scale,
        block_table=paged.table,
        context_lens=ctx,
        **kw,
    )


def _check_decode(idx, sc, q, w, ctx, paged, top_k):
    B, next_n = ctx.shape
    for b in range(B):
        n_max = int(ctx[b].max())
        keys = paged.request_keys(b, n_max)
        rows = slice(b * next_n, (b + 1) * next_n)
        full = _scores(q.value[b], w[rows], keys)
        for j in range(next_n):
            r = b * next_n + j
            n = int(ctx[b, j])
            _check_row(idx[r], sc[r], full[j, :n], top_k, False, f"request {b} row {j}")


# (heads, top_k) as served: 64 heads keep 512, 32 heads 2048.
_HK = [(64, 512), (32, 2048)]


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", _HK)
@pytest.mark.parametrize(
    "B,next_n,lens,P,zero_frac",
    [
        # Rows longer than one CTA selects (several CTAs each), and short.
        (1, 1, [70000], 64, 0.0),
        (1, 2, [1500], 64, 0.0),
        (1, 2, [70000], 64, 0.0),
        # Long rows of near-equal scores (crowded bins), beside a short one.
        (1, 1, [100000], 64, 0.98),
        (2, 1, [90000, 20000], 64, 0.0),
        # Several requests, long and short rows mixed.
        (8, 2, [5000, 20000, 9000, 70000, 100, 30000, 8193, 64000], 64, 0.0),
        (4, 2, [40000, 3000, 0, 12000], 32, 0.98),
        # More requests than SMs leave free: CTAs span several requests.
        (37, 2, list(range(1000, 38000, 1000)), 64, 0.0),
        (200, 1, [(i * 7919) % 20000 for i in range(200)], 64, 0.0),
        # More long rows than SMs: a long-row CTA takes several rows.
        (100, 2, [17000] * 100, 64, 0.0),
    ],
)
def test_paged(fp4, heads, top_k, B, next_n, lens, P, zero_frac):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(B * 131 + next_n + heads + top_k)
    paged = _Paged(fp4, B, max(lens), P, gen, dev, zero_frac)
    q, w, ctx = _decode_inputs(fp4, B, next_n, heads, lens, gen, dev)
    idx, sc = _run_decode(q, w, ctx, paged, top_k)
    _check_decode(idx, sc, q, w, ctx, paged, top_k)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", _HK)
@pytest.mark.parametrize("next_n", [3, 4, 6])
def test_paged_verify(fp4, heads, top_k, next_n):
    """Speculative-decoding verify: next_n draft rows per request (units of
    one or two rows sharing the request's block table row)."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(next_n * 17 + heads)
    lens = [90000, 3000, 200, 40000]
    paged = _Paged(fp4, len(lens), max(lens), 64, gen, dev)
    q, w, ctx = _decode_inputs(fp4, len(lens), next_n, heads, lens, gen, dev)
    idx, sc = _run_decode(q, w, ctx, paged, top_k)
    _check_decode(idx, sc, q, w, ctx, paged, top_k)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", _HK)
@pytest.mark.parametrize("next_n", [1, 2])
@pytest.mark.parametrize(
    "lens",
    [
        [512, 300, 1, 0],
        [2048, 1500, 129, 2000],
        [2000, 60000, 3000, 511],
        [9216, 5000, 2700, 8193],
        [8000, 6000, 3000, 4500],
    ],
    ids=["short", "within", "mixed", "lean", "bins"],
)
def test_paged_within_top_k(fp4, heads, top_k, next_n, lens):
    """Rows of at most top_k keys (every key selected) beside rows of a few
    thousand keys, alone or beside longer requests."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(next_n + heads + len(lens))
    paged = _Paged(fp4, len(lens), max(lens), 64, gen, dev)
    q, w, ctx = _decode_inputs(fp4, len(lens), next_n, heads, lens, gen, dev)
    idx, sc = _run_decode(q, w, ctx, paged, top_k)
    _check_decode(idx, sc, q, w, ctx, paged, top_k)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", _HK)
@pytest.mark.parametrize(
    "lens",
    [[2500, 2300, 2560, 1000], [9000, 7000, 4000, 2600], [2500, 40000, 2300, 9000]],
    ids=["short", "lean", "mixed"],
)
def test_paged_crowded_scores(fp4, heads, top_k, lens):
    """Scores within ~2% of each other beside a few zero scores: the
    selection's histogram bins (over the whole key range) crowd and narrow."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(heads + top_k)
    paged = _Paged(fp4, len(lens), max(lens), 64, gen, dev, 0.002, crowd=True)
    q, w, ctx = _decode_inputs(fp4, len(lens), 2, heads, lens, gen, dev)
    # Positive queries and weights: every head's dot product counts.
    if fp4:
        q.data = q.data & 0x77
    else:
        q.data = q.data.float().abs().to(torch.float8_e4m3fn)
    q.value = q.value.abs()
    w = w.abs() + 0.5
    idx, sc = _run_decode(q, w, ctx, paged, top_k)
    _check_decode(idx, sc, q, w, ctx, paged, top_k)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
def test_paged_long_rows(fp4):
    """Several long rows (each over several selection CTAs) in one batch."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(11)
    lens = [140000, 80000, 3000, 140000, 100, 90000, 75000, 20000]
    paged = _Paged(fp4, 8, max(lens), 64, gen, dev)
    q, w, ctx = _decode_inputs(fp4, 8, 2, 32, lens, gen, dev)
    idx, sc = _run_decode(q, w, ctx, paged, 2048)
    _check_decode(idx, sc, q, w, ctx, paged, 2048)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", _HK)
@pytest.mark.parametrize(
    "B,next_n,lens",
    [
        (1, 1, [70000]),
        (1, 2, [1500]),
        (8, 2, [5000, 20000, 9000, 70000, 100, 30000, 8193, 64000]),
        (37, 2, list(range(1000, 38000, 1000))),
        (4, 6, [90000, 3000, 200, 40000]),
    ],
)
def test_paged_indices_only(fp4, heads, top_k, B, next_n, lens):
    """return_scores=False: the same index sets, no scores."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(B * 7 + next_n + heads)
    paged = _Paged(fp4, B, max(lens), 64, gen, dev)
    q, w, ctx = _decode_inputs(fp4, B, next_n, heads, lens, gen, dev)
    idx, _ = _run_decode(q, w, ctx, paged, top_k)
    idx2, sc2 = _run_decode(q, w, ctx, paged, top_k, return_scores=False)
    assert sc2 is None
    assert torch.equal(torch.sort(idx, 1).values, torch.sort(idx2, 1).values)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
def test_paged_no_pdl(fp4):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7)
    lens = [30000, 2000]
    paged = _Paged(fp4, 2, max(lens), 64, gen, dev)
    q, w, ctx = _decode_inputs(fp4, 2, 2, 32, lens, gen, dev)
    idx, sc = _run_decode(q, w, ctx, paged, 2048, enable_pdl=False)
    _check_decode(idx, sc, q, w, ctx, paged, 2048)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize("heads,top_k", _HK)
def test_paged_cuda_graph(fp4, heads, top_k):
    """One capture replays over new contents and lengths of the same buffers:
    the kernel plans its split from the lengths on the device."""
    dev = torch.device("cuda")
    B, next_n, max_len = 8, 2, 70000
    gen = torch.Generator(device=dev).manual_seed(heads + top_k)
    paged = _Paged(fp4, B, max_len, 64, gen, dev)
    q, w, ctx = _decode_inputs(fp4, B, next_n, heads, [max_len] * B, gen, dev)
    idx = torch.empty(B * next_n, top_k, dtype=torch.int32, device=dev)
    sc = torch.empty(B * next_n, top_k, dtype=torch.float32, device=dev)

    def run():
        _run_decode(q, w, ctx, paged, top_k, indices=idx, scores=sc)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lo, hi in ((1, max_len), (1, 2000), (60000, max_len), (0, 1)):
        q2 = _queries(fp4, (B, next_n, heads), gen, dev)
        q.data.copy_(q2.data)
        if fp4:
            q.scale.copy_(q2.scale)
        q.value = q2.value
        Lb = torch.randint(
            lo, hi + 1, (B, 1), generator=gen, device=dev, dtype=torch.int32
        )
        ctx.copy_((Lb - next_n + 1 + torch.arange(next_n, device=dev)).clamp_min(0))
        graph.replay()
        torch.cuda.synchronize()
        _check_decode(idx, sc, q, w, ctx, paged, top_k)


@pytest.mark.parametrize("fp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize(
    "B,next_n", [(4, 1), (8, 2)], ids=["one-row-units", "two-row-units"]
)
def test_paged_cuda_graph_long_short(fp4, B, next_n):
    """Replays alternating long, short and mixed rows: a row's long-row state
    (double-buffered by its own epoch) must not carry over a short call."""
    dev = torch.device("cuda")
    heads, top_k, max_len = 32, 2048, 70000
    gen = torch.Generator(device=dev).manual_seed(B * 3 + next_n)
    paged = _Paged(fp4, B, max_len, 64, gen, dev)
    q, w, ctx = _decode_inputs(fp4, B, next_n, heads, [max_len] * B, gen, dev)
    idx = torch.empty(B * next_n, top_k, dtype=torch.int32, device=dev)
    sc = torch.empty(B * next_n, top_k, dtype=torch.float32, device=dev)

    def run():
        _run_decode(q, w, ctx, paged, top_k, indices=idx, scores=sc)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    ranges = [(60000, max_len), (1, 2000), (60000, max_len), (1, max_len), (1, 2000)]
    ranges += [(17000, max_len), (60000, max_len)]
    for lo, hi in ranges:
        Lb = torch.randint(
            lo, hi + 1, (B, 1), generator=gen, device=dev, dtype=torch.int32
        )
        ctx.copy_((Lb - next_n + 1 + torch.arange(next_n, device=dev)).clamp_min(0))
        graph.replay()
        torch.cuda.synchronize()
        _check_decode(idx, sc, q, w, ctx, paged, top_k)


def test_paged_validation():
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    paged = _Paged(False, 1, 1000, 64, gen, dev)
    q, w, ctx = _decode_inputs(False, 1, 17, 32, [1000], gen, dev)
    with pytest.raises(ValueError, match="next_n"):
        _run_decode(q, w, ctx, paged, 512)
    with pytest.raises(ValueError, match="block_table and context_lens"):
        dsa_indexer_topk(
            q.data, paged.cache, w, None, None, backend="sm120", context_lens=ctx
        )
    q, w, ctx = _decode_inputs(False, 1, 1, 32, [1000], gen, dev)
    with pytest.raises(ValueError, match="ragged form only"):
        _run_decode(q, w, ctx, paged, 512, ratio=2)
    with pytest.raises(ValueError, match="top_k"):
        _run_decode(q, w, ctx, paged, 0)
    with pytest.raises(ValueError, match="capacity"):
        sm120.dsa_indexer_topk_paged(
            q.data,
            paged.cache,
            w,
            ctx,
            paged.table,
            max_context_len=paged.table.shape[1] * paged.P + 1,
        )


@pytest.mark.parametrize("top_k", [1, 4096])
def test_paged_top_k(top_k):
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(top_k)
    lens = [5000, 20000]
    paged = _Paged(False, 2, max(lens), 64, gen, dev)
    q, w, ctx = _decode_inputs(False, 2, 2, 32, lens, gen, dev)
    idx, sc = _run_decode(q, w, ctx, paged, top_k)
    _check_decode(idx, sc, q, w, ctx, paged, top_k)
