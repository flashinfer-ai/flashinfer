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

Paged recurrent-state GDN prefill and decode (``flashinfer.gdn_paged``) and the
slot arguments of ``chunk_gated_delta_rule`` they use (``output_state_indices``,
``checkpoint_state_indices``, ``prefix_lens``).

The oracle is ``chunk_gated_delta_rule`` without those arguments, run on each
sequence's whole token stream from its first token: its 64-token checkpoints
give the state at every page boundary and its final state the state at the
end.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.gdn_paged import (
    GDNPagedDecodeSlots,
    GDNPagedPrefillSlots,
    chunk_gated_delta_rule_paged,
    gated_delta_rule_decode_paged,
    gdn_paged_decode_slots,
    gdn_paged_prefill_slots,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import get_compute_capability

from .conftest import gen_qkv

HEAD_SIZE = 128
CHUNK = 64


def _skip_if_unsupported():
    major, _ = get_compute_capability(torch.device("cuda"))
    if major not in (8, 9, 10, 11, 12):
        pytest.skip("GDN prefill needs SM8x, SM90, SM100/SM103/SM107, SM110 or SM12x")
    if major in (10, 11):
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip("SM100/SM110 GDN prefill requires CUDA 13+")


def _cu(lens: list[int], device) -> torch.Tensor:
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32, device=device)


class Streams:
    """Whole token streams of a batch, one per sequence, and their gates."""

    def __init__(self, total_lens, num_q_heads, num_k_heads, num_v_heads, dtype):
        self.total_lens = list(total_lens)
        self.num_sab_heads = max(num_q_heads, num_v_heads)
        self.dtype = dtype
        total = sum(total_lens)
        q, k, v = gen_qkv(
            [total], num_q_heads, num_k_heads, num_v_heads, HEAD_SIZE, dtype
        )
        self.q = q.cuda()
        self.k = F.normalize(k.cuda().float(), p=2.0, dim=-1).to(dtype)
        self.v = v.cuda()
        # Gates in (0.9, 1) keep long streams in a well-conditioned range.
        self.g = 0.9 + 0.1 * torch.rand(total, self.num_sab_heads, device="cuda")
        self.beta = torch.rand(total, self.num_sab_heads, device="cuda")
        self.starts = _cu(self.total_lens, "cpu").tolist()

    def slices(self, lo: list[int], hi: list[int]):
        """Pack tokens ``[lo[b], hi[b])`` of every stream."""
        rows = torch.cat(
            [
                torch.arange(self.starts[b] + lo[b], self.starts[b] + hi[b])
                for b in range(len(lo))
            ]
        ).cuda()
        return (
            self.q[rows].contiguous(),
            self.k[rows].contiguous(),
            self.v[rows].contiguous(),
            self.g[rows].contiguous(),
            self.beta[rows].contiguous(),
        )


def _oracle(streams: Streams, lens: list[int], state_dtype=torch.float32):
    """Run every stream from token 0 for ``lens[b]`` tokens.

    Returns the packed output, the final states and, per sequence, the state
    after every 64th token.
    """
    num_seqs = len(lens)
    q, k, v, g, beta = streams.slices([0] * num_seqs, lens)
    counts = [n // CHUNK for n in lens]
    total_ckpts = max(sum(counts), 1)
    ckpts = torch.zeros(
        total_ckpts,
        streams.num_sab_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        dtype=state_dtype,
        device="cuda",
    )
    final = torch.zeros(
        num_seqs,
        streams.num_sab_heads,
        HEAD_SIZE,
        HEAD_SIZE,
        dtype=state_dtype,
        device="cuda",
    )
    out, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(HEAD_SIZE),
        None,
        True,
        _cu(lens, "cuda").long(),
        output_state=final,
        state_checkpoints=ckpts if sum(counts) else None,
        checkpoint_cu_starts=_cu(counts, "cuda").long() if sum(counts) else None,
        checkpoint_every_n_tokens=CHUNK if sum(counts) else 0,
        use_cp=False,
    )
    starts = _cu(counts, "cpu").tolist()
    per_seq = [ckpts[starts[b] : starts[b + 1]] for b in range(num_seqs)]
    return out, final, per_seq


class PagedBatch:
    """A page table over a shuffled pool, sized for ``total_lens`` tokens."""

    def __init__(
        self, total_lens, page_size, num_sab_heads, state_dtype, spare=7, row_pad=0
    ):
        self.page_size = page_size
        self.total_lens = list(total_lens)
        self.num_pages = [max(1, math.ceil(n / page_size)) for n in total_lens]
        pool_pages = sum(self.num_pages) + spare
        ids = list(range(pool_pages))
        random.shuffle(ids)
        self.pages = []
        it = iter(ids)
        for n in self.num_pages:
            self.pages.append([next(it) for _ in range(n)])
        # Finite, page-distinct filler: a page read by mistake corrupts the
        # result, and a page written by mistake no longer equals its copy.
        self.guarded = GuardedPool(pool_pages, num_sab_heads, state_dtype, row_pad)
        self.pool = self.guarded.pool
        self.pool.copy_(
            torch.randn(pool_pages, num_sab_heads, HEAD_SIZE, HEAD_SIZE, device="cuda")
            + torch.arange(pool_pages, device="cuda").view(-1, 1, 1, 1)
        )

    def tables(self, index_dtype=torch.int32):
        indices = torch.tensor(
            [p for pages in self.pages for p in pages], dtype=index_dtype, device="cuda"
        )
        indptr = _cu(self.num_pages, "cuda").to(index_dtype)
        last = torch.tensor(
            [
                n - (m - 1) * self.page_size
                for n, m in zip(self.total_lens, self.num_pages, strict=True)
            ],
            dtype=index_dtype,
            device="cuda",
        )
        return indices, indptr, last

    def page_of(self, b: int, token: int) -> int:
        return self.pages[b][token // self.page_size]


def _assert_state(actual, expected, *, msg):
    tol = 1e-2 if actual.dtype == torch.float32 else 3e-2
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=tol,
        rtol=tol,
        msg=lambda m: f"{msg}: {m}",
    )


GUARD = 1.0e4


class GuardedPool:
    """``rows`` state rows between two guard rows. ``row_pad`` extra elements
    after each row make the row stride larger than ``H * V * K``. A write
    through a ``-1`` row or past a row's ``[H, V, K]`` block hits a guard."""

    def __init__(self, rows, num_heads, dtype, row_pad=0):
        self.row = num_heads * HEAD_SIZE * HEAD_SIZE
        self.storage = torch.full(
            (rows + 2, self.row + row_pad), GUARD, dtype=dtype, device="cuda"
        )
        self.pool = self.storage[1:-1, : self.row].unflatten(
            1, (num_heads, HEAD_SIZE, HEAD_SIZE)
        )

    def assert_guards(self):
        for name, part in (
            ("row -1", self.storage[0]),
            ("the row past the end", self.storage[-1]),
            ("the row padding", self.storage[1:-1, self.row :]),
        ):
            assert torch.equal(part, torch.full_like(part, GUARD)), f"wrote {name}"


def _guarded_output(num_tokens, num_heads, dtype):
    """An output buffer between ``CHUNK`` guard rows on each side: a sequence
    chunked from before its first token must not store the rows before it,
    and the last sequence must not store past its last token."""
    buf = torch.full(
        (CHUNK + num_tokens + CHUNK, num_heads, HEAD_SIZE),
        GUARD,
        dtype=dtype,
        device="cuda",
    )
    return buf, buf[CHUNK:-CHUNK]


def _assert_output_guard(buf):
    for name, part in (("before row 0", buf[:CHUNK]), ("past the end", buf[-CHUNK:])):
        assert torch.equal(part, torch.full_like(part, GUARD)), f"stored {name}"


def _run_paged_prefill_case(
    prefix_lens,
    new_lens,
    page_size,
    heads,
    dtype,
    state_dtype,
    index_dtype=torch.int32,
    row_pad=0,
    backend="flashinfer",
    max_seqlen=None,
):
    num_q_heads, num_k_heads, num_v_heads = heads
    totals = [p + n for p, n in zip(prefix_lens, new_lens, strict=True)]
    streams = Streams(totals, num_q_heads, num_k_heads, num_v_heads, dtype)
    batch = PagedBatch(
        totals, page_size, streams.num_sab_heads, state_dtype, row_pad=row_pad
    )
    num_seqs = len(totals)

    # Seed every prefix page with its true state; the call must only read the
    # page that holds the last prefix token.
    _, prefix_final, prefix_ckpts = _oracle(streams, prefix_lens, state_dtype)
    for b, p in enumerate(prefix_lens):
        for j in range(p // page_size):
            batch.pool[batch.pages[b][j]] = prefix_ckpts[b][
                (j + 1) * page_size // CHUNK - 1
            ]
        if p % page_size:
            batch.pool[batch.page_of(b, p - 1)] = prefix_final[b]
    pool_before = batch.pool.clone()

    q, k, v, g, beta = streams.slices(prefix_lens, totals)
    indices, indptr, last = batch.tables(index_dtype)
    guard, out = _guarded_output(sum(new_lens), streams.num_sab_heads, dtype)
    chunk_gated_delta_rule_paged(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(HEAD_SIZE),
        cu_seqlens=_cu(new_lens, "cuda").to(index_dtype),
        state_pool=batch.pool,
        page_indices=indices,
        page_indptr=indptr,
        page_size=page_size,
        last_page_len=last,
        output=out,
        backend=backend,
        max_seqlen=max_seqlen,
    )
    torch.cuda.synchronize()
    _assert_output_guard(guard)
    batch.guarded.assert_guards()

    ref_out, ref_final, ref_ckpts = _oracle(streams, totals, state_dtype)
    ref_rows = torch.cat(
        [
            torch.arange(start + p, start + t)
            for start, p, t in zip(
                _cu(totals, "cpu").tolist()[:-1], prefix_lens, totals, strict=True
            )
        ]
    ).cuda()
    torch.testing.assert_close(
        out.float(), ref_out[ref_rows].float(), atol=1e-2, rtol=1e-2
    )

    written = set()
    for b in range(num_seqs):
        p, t = prefix_lens[b], totals[b]
        if t == p:
            continue
        for j in range(p // page_size, batch.num_pages[b]):
            page = batch.pages[b][j]
            written.add(page)
            end = min((j + 1) * page_size, t)
            expected = ref_final[b] if end == t else ref_ckpts[b][end // CHUNK - 1]
            _assert_state(batch.pool[page], expected, msg=f"seq {b} page {j}")
    for page in range(batch.pool.size(0)):
        if page not in written:
            assert torch.equal(batch.pool[page], pool_before[page]), (
                f"page {page} was not owned by any new token but changed"
            )


@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [(4, 4, 4), (2, 2, 4), (4, 1, 1)])
def test_paged_prefill_ragged(page_size, state_dtype, dtype, heads):
    """Ragged batch mixing no prefix, page-aligned, chunk-aligned and unaligned
    prefixes with new spans that stop short of, on and past page boundaries."""
    _skip_if_unsupported()
    if dtype == torch.float16 and state_dtype == torch.bfloat16:
        pytest.skip("covered by the bfloat16 inputs")
    random.seed(page_size)
    torch.manual_seed(page_size)
    L = page_size
    prefix_lens = [0, 0, L, 2 * L, CHUNK, L + 2 * CHUNK, 37, L + 5, 3 * L - 1, L - 1]
    new_lens = [1, 3 * L + 17, L, 5, 2 * L, L + 7, L - 37, 2 * L + 63, 1, 1]
    _run_paged_prefill_case(prefix_lens, new_lens, L, heads, dtype, state_dtype)


def _skip_unless_cake_cp(*, needs_nvcc=True):
    major, minor = get_compute_capability(torch.device("cuda"))
    if (major, minor) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("the Cake GDN CP route needs SM100, SM103 or SM107")
    cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
    if cuda_major < 13:
        pytest.skip("the Cake GDN CP route needs CUDA 13+")
    if needs_nvcc and (major, minor) == (10, 7):
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least

        # The route rejects sm_107a below a CUDA 13.4 nvcc.
        if not is_cuda_version_at_least("13.4"):
            pytest.skip("the Cake GDN CP route needs CUDA 13.4+ on SM107")


@pytest.mark.parametrize("page_size", [64, 128, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [(4, 4, 4), (2, 2, 4), (4, 1, 1), (2, 2, 8)])
@pytest.mark.parametrize("max_seqlen_hint", [False, True])
def test_paged_prefill_cake_cp_ragged(page_size, dtype, heads, max_seqlen_hint):
    """backend="cake_gdn": a ragged batch with 64-aligned prefixes (none,
    page-aligned, chunk-aligned) and spans that stop short of, on and past
    page boundaries, long enough to split across CP chunks."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    random.seed(page_size)
    torch.manual_seed(page_size)
    L = page_size
    prefix_lens = [0, 0, L, 2 * L, CHUNK, L + 2 * CHUNK, 3 * CHUNK, 0]
    new_lens = [1, 3 * L + 17, L, 5, 2 * L, L + 7, 4096 + 37, 0]
    _run_paged_prefill_case(
        prefix_lens,
        new_lens,
        L,
        heads,
        dtype,
        torch.float32,
        backend="cake_gdn",
        max_seqlen=max(new_lens) if max_seqlen_hint else None,
    )


@pytest.mark.parametrize("page_size", [1024, 4096])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("prefix", [0, 64, 1024, 1088])
def test_paged_prefill_cake_cp_long_padded_pool(index_dtype, prefix, page_size):
    """One long sequence whose CP chunks end on and between page boundaries,
    over a pool with padded rows."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    random.seed(prefix)
    torch.manual_seed(prefix)
    _run_paged_prefill_case(
        [prefix],
        [16384 - 64],
        page_size,
        (2, 2, 4),
        torch.bfloat16,
        torch.float32,
        index_dtype=index_dtype,
        row_pad=4,
        backend="cake_gdn",
        max_seqlen=16384,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_prefill_cake_cp_qk_l2norm_in_kernel(dtype):
    """use_qk_l2norm_in_kernel=True normalizes q and k before the split; the
    output and every written page match the serial route."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    torch.manual_seed(5)
    random.seed(5)
    L = 256
    prefix_lens, new_lens = [0, L, 3 * CHUNK], [3 * L + 17, 2 * L, 1000]
    totals = [p + n for p, n in zip(prefix_lens, new_lens, strict=True)]
    streams = Streams(totals, 2, 2, 4, dtype)
    batch = PagedBatch(totals, L, streams.num_sab_heads, torch.float32)
    q, k, v, g, beta = streams.slices(prefix_lens, totals)
    q = q * 3.0
    k = k * 0.5
    indices, indptr, last = batch.tables()
    results = []
    for backend in ("flashinfer", "cake_gdn"):
        pool = batch.pool.clone()
        out = chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(HEAD_SIZE),
            cu_seqlens=_cu(new_lens, "cuda"),
            state_pool=pool,
            page_indices=indices,
            page_indptr=indptr,
            page_size=L,
            last_page_len=last,
            use_qk_l2norm_in_kernel=True,
            backend=backend,
            max_seqlen=max(new_lens),
        )
        results.append((out, pool))
    torch.testing.assert_close(
        results[1][0].float(), results[0][0].float(), atol=1e-2, rtol=1e-2
    )
    _assert_state(results[1][1], results[0][1], msg="pool")


def test_paged_prefill_cake_cp_all_empty_spans():
    """A batch whose sequences have no new token leaves the pool unchanged."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    L = 128
    pool = torch.randn(6, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    before = pool.clone()
    q = torch.zeros(0, 4, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    out = chunk_gated_delta_rule_paged(
        q,
        q,
        q,
        torch.zeros(0, 4, device="cuda"),
        torch.zeros(0, 4, device="cuda"),
        cu_seqlens=torch.zeros(3, dtype=torch.int32, device="cuda"),
        state_pool=pool,
        page_indices=torch.tensor([0, 3, 4], dtype=torch.int32, device="cuda"),
        page_indptr=torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda"),
        page_size=L,
        last_page_len=torch.tensor([64, L], dtype=torch.int32, device="cuda"),
        backend="cake_gdn",
    )
    torch.cuda.synchronize()
    assert out.shape == (0, 4, HEAD_SIZE)
    assert torch.equal(pool, before)


def test_paged_prefill_cake_cp_rejects_unsupported_pools():
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    q = torch.zeros(64, 2, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    pool = torch.zeros(4, 2, HEAD_SIZE, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    one = torch.ones(1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="float32 state pool"):
        chunk_gated_delta_rule_paged(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32, device="cuda"),
            state_pool=pool,
            page_indices=one,
            page_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            page_size=64,
            last_page_len=torch.tensor([64], dtype=torch.int32, device="cuda"),
            backend="cake_gdn",
        )


def _cake_cp_backend():
    from flashinfer.gdn_kernels.blackwell import cake_gdn_cp_backend

    return cake_gdn_cp_backend


def _record_cake_cp_plans(monkeypatch):
    """The paged plans of the Cake CP calls of the test, in call order."""
    gdn_cp = _cake_cp_backend()
    plans = []
    paged_plan = gdn_cp._paged_plan

    def recording_plan(*args, **kwargs):
        plans.append(paged_plan(*args, **kwargs))
        return plans[-1]

    monkeypatch.setattr(gdn_cp, "_paged_plan", recording_plan)
    return plans


def _skip_unless_cake_cp_kernels(*names):
    gdn_cp = _cake_cp_backend()
    arch = gdn_cp._arch_for(torch.device("cuda"))
    for name in names:
        if not gdn_cp.has_gdn_cp_kernel(name, arch):
            pytest.skip(f"{arch} ships no {name}")
    return arch


def _run_cake_cp_against_flashinfer(
    prefix_lens,
    new_lens,
    page_size,
    heads,
    dtype,
    *,
    max_seqlen=None,
    gates=None,
    seed=0,
):
    """Run both backends from one pool: the Cake CP output and every pool row
    match the serial route, and the Cake call writes no guard."""
    random.seed(seed)
    torch.manual_seed(seed)
    num_q_heads, num_k_heads, num_v_heads = heads
    num_sab_heads = max(num_q_heads, num_v_heads)
    total = sum(new_lens)
    totals = [p + n for p, n in zip(prefix_lens, new_lens, strict=True)]

    def uniform(num_heads):
        return (0.5 * torch.rand(total, num_heads, HEAD_SIZE, device="cuda") - 0.25).to(
            dtype
        )

    q, v = uniform(num_q_heads), uniform(num_v_heads)
    k = F.normalize(uniform(num_k_heads).float(), p=2.0, dim=-1).to(dtype)
    g = 0.9 + 0.1 * torch.rand(total, num_sab_heads, device="cuda")
    beta = torch.rand(total, num_sab_heads, device="cuda")
    kwargs = {}
    if gates is not None:
        g, beta, kwargs = gates(g, beta)
    batch = PagedBatch(totals, page_size, num_sab_heads, torch.float32)
    # Page-distinct initial states of the scale of real ones: ungated, the
    # filler's page-id offsets would persist and dominate the outputs.
    num_pages = batch.pool.size(0)
    batch.pool.copy_(
        0.1 * torch.randn_like(batch.pool)
        + torch.arange(num_pages, device="cuda").view(-1, 1, 1, 1) / num_pages
    )
    indices, indptr, last = batch.tables()
    reference_pool = batch.pool.clone()
    outputs = {}
    for backend, pool in (("flashinfer", reference_pool), ("cake_gdn", batch.pool)):
        guard, out = _guarded_output(total, num_sab_heads, dtype)
        chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(HEAD_SIZE),
            cu_seqlens=_cu(new_lens, "cuda"),
            state_pool=pool,
            page_indices=indices,
            page_indptr=indptr,
            page_size=page_size,
            last_page_len=last,
            output=out,
            backend=backend,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        torch.cuda.synchronize()
        _assert_output_guard(guard)
        outputs[backend] = out
    batch.guarded.assert_guards()
    torch.testing.assert_close(
        outputs["cake_gdn"].float(),
        outputs["flashinfer"].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    _assert_state(batch.pool, reference_pool, msg="pool")


def _fp64_gates(g, beta, total, num_heads, kwargs):
    """The alpha and beta a gate convention means, in float64."""
    ones = torch.ones(total, num_heads, dtype=torch.float64, device="cuda")
    if kwargs.get("use_gate_in_kernel"):
        A_log = kwargs["A_log"].double()
        dt_bias = kwargs["dt_bias"].double()
        alpha = torch.exp(-A_log.exp() * F.softplus(g.double() + dt_bias))
    elif g is None:
        alpha = ones
    elif kwargs.get("gate_domain") == "log":
        alpha = g.double().exp()
    else:
        alpha = g.double()
    if beta is None:
        beta = ones
    elif kwargs.get("beta_is_logit"):
        beta = torch.sigmoid(beta.double())
    else:
        beta = beta.double()
    return alpha, beta


def _fp64_paged_reference(
    q, k, v, alpha, beta, scale, prefix_lens, new_lens, batch, initial_pool
):
    """Serial float64 recurrence of every sequence from its initial pool row
    (the zero state without a prefix). Returns the output and the state of
    every pool row the call writes: each page boundary it crosses and the
    last page."""
    num_heads = alpha.size(1)
    page_size = batch.page_size
    out = torch.zeros(
        q.size(0), num_heads, HEAD_SIZE, dtype=torch.float64, device="cuda"
    )
    rows = {}
    start = 0
    for b, (prefix, num_new) in enumerate(zip(prefix_lens, new_lens, strict=True)):
        if prefix:
            row = initial_pool[batch.page_of(b, prefix - 1)]
            state = row.double().transpose(-1, -2).clone()  # [H, K, V]
        else:
            state = torch.zeros(
                num_heads, HEAD_SIZE, HEAD_SIZE, dtype=torch.float64, device="cuda"
            )
        tokens = slice(start, start + num_new)
        qs, ks, vs = (
            x[tokens].double().repeat_interleave(num_heads // x.size(1), dim=1)
            for x in (q, k, v)
        )
        for i in range(num_new):
            end = prefix + i + 1
            state = alpha[start + i].view(-1, 1, 1) * state
            delta = beta[start + i].view(-1, 1) * (
                vs[i] - torch.einsum("hk,hkv->hv", ks[i], state)
            )
            state = state + ks[i].unsqueeze(-1) * delta.unsqueeze(-2)
            out[start + i] = scale * torch.einsum("hk,hkv->hv", qs[i], state)
            if end % page_size == 0 or i == num_new - 1:
                rows[batch.page_of(b, end - 1)] = state.transpose(-1, -2).clone()
        start += num_new
    return out, rows


def _run_cake_cp_and_flashinfer_against_fp64(
    prefix_lens, new_lens, page_size, heads, dtype, *, gates=None, seed=0
):
    """Run both backends from one pool against a float64 reference: the
    output and every pool row each call writes match it, each call leaves the
    other rows and the guards untouched."""
    random.seed(seed)
    torch.manual_seed(seed)
    num_q_heads, num_k_heads, num_v_heads = heads
    num_sab_heads = max(num_q_heads, num_v_heads)
    total = sum(new_lens)
    totals = [p + n for p, n in zip(prefix_lens, new_lens, strict=True)]

    def uniform(num_heads):
        return (0.5 * torch.rand(total, num_heads, HEAD_SIZE, device="cuda") - 0.25).to(
            dtype
        )

    q, v = uniform(num_q_heads), uniform(num_v_heads)
    k = F.normalize(uniform(num_k_heads).float(), p=2.0, dim=-1).to(dtype)
    g = 0.9 + 0.1 * torch.rand(total, num_sab_heads, device="cuda")
    beta = torch.rand(total, num_sab_heads, device="cuda")
    kwargs = {}
    if gates is not None:
        g, beta, kwargs = gates(g, beta)
    batch = PagedBatch(totals, page_size, num_sab_heads, torch.float32)
    num_pages = batch.pool.size(0)
    batch.pool.copy_(
        0.1 * torch.randn_like(batch.pool)
        + torch.arange(num_pages, device="cuda").view(-1, 1, 1, 1) / num_pages
    )
    initial_pool = batch.pool.clone()
    indices, indptr, last = batch.tables()
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    alpha, beta_fp64 = _fp64_gates(g, beta, total, num_sab_heads, kwargs)
    reference_out, reference_rows = _fp64_paged_reference(
        q, k, v, alpha, beta_fp64, scale, prefix_lens, new_lens, batch, initial_pool
    )
    written = sorted(reference_rows)
    untouched = [r for r in range(num_pages) if r not in reference_rows]
    reference_pool = torch.stack([reference_rows[r] for r in written])
    flashinfer_pool = GuardedPool(num_pages, num_sab_heads, torch.float32)
    flashinfer_pool.pool.copy_(initial_pool)
    for backend, guarded in (
        ("flashinfer", flashinfer_pool),
        ("cake_gdn", batch.guarded),
    ):
        guard, out = _guarded_output(total, num_sab_heads, dtype)
        chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            scale,
            cu_seqlens=_cu(new_lens, "cuda"),
            state_pool=guarded.pool,
            page_indices=indices,
            page_indptr=indptr,
            page_size=page_size,
            last_page_len=last,
            output=out,
            backend=backend,
            **kwargs,
        )
        torch.cuda.synchronize()
        _assert_output_guard(guard)
        guarded.assert_guards()
        assert torch.equal(guarded.pool[untouched], initial_pool[untouched]), backend
        torch.testing.assert_close(
            out.double(),
            reference_out,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m, backend=backend: f"{backend} output: {m}",
        )
        _assert_state(guarded.pool[written], reference_pool, msg=f"{backend} pool")


def _log_gates(g, beta):
    return g.log(), beta, {"gate_domain": "log"}


def _gates_in_kernel(g, beta):
    heads = g.size(1)
    A_log = -3.0 + 0.1 * torch.randn(heads, device="cuda")
    dt_bias = 0.1 * torch.randn(heads, device="cuda")
    kwargs = {"use_gate_in_kernel": True, "A_log": A_log, "dt_bias": dt_bias}
    return torch.randn_like(g), beta, kwargs


def _beta_logits(g, beta):
    return g, torch.logit(beta.clamp(1e-3, 1 - 1e-3)), {"beta_is_logit": True}


@pytest.mark.parametrize(
    "gates",
    [
        _log_gates,
        _gates_in_kernel,
        _beta_logits,
        lambda g, beta: (None, beta, {}),
        lambda g, beta: (g, None, {}),
        lambda g, beta: (None, None, {}),
    ],
    ids=["log_g", "gate_in_kernel", "beta_logit", "no_g", "no_beta", "no_gates"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_prefill_cake_cp_gate_conventions(gates, dtype):
    """Every gate convention of the paged API gives the output and pages of
    the recurrence it means, on both backends."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    # Against float64, not each other: without gates the states do not
    # contract, so the two backends' independent rounding errors add.
    _run_cake_cp_and_flashinfer_against_fp64(
        [0, 128, 64], [300, 273, 129], 128, (2, 2, 4), dtype, gates=gates
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"gate_domain": "exp"}, "gate_domain must be"),
        ({"A_log": "heads"}, "A_log and dt_bias require use_gate_in_kernel"),
        ({"use_gate_in_kernel": True, "g": None}, "requires g, A_log and dt_bias"),
        ({"beta_is_logit": True, "beta": None}, "beta_is_logit requires beta"),
        ({"gate_domain": "log", "g": "short"}, "g must have shape"),
    ],
    ids=["domain", "A_log_alone", "gate_in_kernel_no_g", "logit_no_beta", "g_shape"],
)
def test_paged_prefill_cake_cp_validates_gates(kwargs, match):
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    heads = 4
    q = torch.zeros(64, heads, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    pool = torch.zeros(2, heads, HEAD_SIZE, HEAD_SIZE, device="cuda")
    args = {
        "g": torch.ones(64, heads, device="cuda"),
        "beta": torch.ones(64, heads, device="cuda"),
        **kwargs,
    }
    if args.get("A_log") == "heads":
        args["A_log"] = torch.zeros(heads, device="cuda")
    if isinstance(args.get("g"), str):
        args["g"] = torch.zeros(32, heads, device="cuda")
    if args.get("use_gate_in_kernel"):
        args.update(
            A_log=torch.zeros(heads, device="cuda"),
            dt_bias=torch.zeros(heads, device="cuda"),
        )
    with pytest.raises(ValueError, match=match):
        chunk_gated_delta_rule_paged(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32, device="cuda"),
            state_pool=pool,
            page_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
            page_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            page_size=64,
            last_page_len=torch.tensor([64], dtype=torch.int32, device="cuda"),
            backend="cake_gdn",
            **args,
        )


@pytest.mark.parametrize("page_size", [128, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [(4, 4, 4), (2, 2, 4)])
def test_paged_prefill_cake_cp_equal_lengths_full_block(
    monkeypatch, page_size, dtype, heads
):
    """Equal 128-aligned spans under the host bound run the full-block stage-4
    program; the fp16 equal-head plan maps to it too."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    kernel = f"cp_prefill{'_bf16' if dtype == torch.bfloat16 else ''}_paged"
    _skip_unless_cake_cp_kernels(kernel)
    gdn_cp = _cake_cp_backend()
    q = torch.empty(2048, heads[0], HEAD_SIZE, dtype=dtype, device="cuda")
    k = torch.empty(2048, heads[1], HEAD_SIZE, dtype=dtype, device="cuda")
    v = torch.empty(2048, heads[2], HEAD_SIZE, dtype=dtype, device="cuda")
    unpaged = gdn_cp._build_plan(
        q, k, v, num_seqs=2, max_seqlen=1024, cu_seqlens_dtype=torch.int32
    )
    if dtype == torch.float16 and heads[0] == heads[2]:
        assert unpaged.prefill_kernel.startswith("cp_prefill_equal_head")
    else:
        assert unpaged.prefill_kernel == kernel.removesuffix("_paged")
    plans = _record_cake_cp_plans(monkeypatch)
    _run_cake_cp_against_flashinfer(
        [64, page_size], [1024, 1024], page_size, heads, dtype, max_seqlen=1024
    )
    assert [plan.prefill_kernel for plan in plans] == [kernel]


@pytest.mark.parametrize(
    "dtype, new_lens, cp_chunk_len",
    [
        (torch.bfloat16, [128], 64),
        (torch.bfloat16, [256, 256], 64),
        (torch.float16, [384], 192),
    ],
    ids=["bf16_128", "bf16_2x256", "fp16_384"],
)
def test_paged_prefill_cake_cp_equal_lengths_partial_pair_chunks(
    monkeypatch, dtype, new_lens, cp_chunk_len
):
    """Equal 128-aligned spans whose CP chunk is not whole 128-token pairs run
    the generic stage-4 program: the full-block one reads the gates of whole
    pairs, past the end of a chunk that ends inside one."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    plans = _record_cake_cp_plans(monkeypatch)
    _run_cake_cp_against_flashinfer(
        [0, 128][: len(new_lens)],
        new_lens,
        128,
        (2, 2, 4),
        dtype,
        max_seqlen=max(new_lens),
    )
    suffix = "_bf16" if dtype == torch.bfloat16 else ""
    assert [(plan.cp_chunk_len, plan.prefill_kernel) for plan in plans] == [
        (cp_chunk_len, f"cp_prefill_generic{suffix}_paged")
    ]


@pytest.mark.parametrize(
    "heads, prefix_lens, new_lens, page_size, fixup, cp_chunk_len",
    [
        ((4, 4, 16), [64], [8128], 1024, "state_fixup_utcmma64_paged", None),
        (
            (16, 16, 64),
            [64, 1024],
            [2048, 2048],
            1024,
            "state_fixup_utcmma128_paged",
            None,
        ),
        ((16, 16, 64), [64], [65536], 4096, "state_fixup_utcmma64_paged", 4096),
    ],
    ids=["utcmma64", "utcmma128", "utcmma64_chunk4096"],
)
def test_paged_prefill_cake_cp_utcmma_fixups(
    monkeypatch, heads, prefix_lens, new_lens, page_size, fixup, cp_chunk_len
):
    """Batches with more parallel states than the SIMT fixup fills run the
    paged UTCMMA fixups; the one long GB300 sequence takes the 4096-token CP
    chunk."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    arch = _skip_unless_cake_cp_kernels(fixup)
    if arch == "sm_107a":
        pytest.skip("sm_107a runs the SIMT fixups only")
    if cp_chunk_len is not None and arch != "sm_103a":
        pytest.skip("the 4096-token CP chunk override is sm_103a only")
    plans = _record_cake_cp_plans(monkeypatch)
    _run_cake_cp_against_flashinfer(
        prefix_lens,
        new_lens,
        page_size,
        heads,
        torch.bfloat16,
        max_seqlen=max(new_lens),
    )
    assert [plan.fixup_kernel for plan in plans] == [fixup]
    if cp_chunk_len is not None:
        assert plans[0].cp_chunk_len == cp_chunk_len


def test_paged_prefill_cake_cp_sm107_three_cta_fixup(monkeypatch):
    """On sm_107a, equal sequences whose 32 row CTAs per state exceed two
    CTAs per SM run the paged 3-CTA-per-SM SIMT fixup."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    fixup = "state_fixup_simt_row4_mb3_paged"
    if _skip_unless_cake_cp_kernels(fixup) != "sm_107a":
        pytest.skip("the 3-CTA-per-SM SIMT fixup is sm_107a only")
    plans = _record_cake_cp_plans(monkeypatch)
    _run_cake_cp_against_flashinfer(
        [64, 1024],
        [2048, 2048],
        1024,
        (16, 16, 64),
        torch.bfloat16,
        max_seqlen=2048,
    )
    assert [plan.fixup_kernel for plan in plans] == [fixup]


@pytest.mark.parametrize(
    "prefix_lens, new_lens, page_size, heads, fixup",
    [
        ([32, 0], [96, 128], 64, (2, 2, 4), None),
        ([0, 32], [128, 100], 64, (2, 2, 4), None),
        ([32, 64], [2048, 2048], 1024, (16, 16, 64), "state_fixup_utcmma128_paged"),
    ],
    ids=["page_end", "mid_page", "utcmma128"],
)
def test_paged_prefill_cake_cp_misaligned_prefix_poisons_its_rows(
    monkeypatch, prefix_lens, new_lens, page_size, heads, fixup
):
    """A prefix that is not 64-aligned fills every row its sequence writes
    with NaN; the aligned sequence of the batch is unaffected.  The SIMT and
    the UTCMMA paged fixups both do so."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    if fixup is not None:
        if _skip_unless_cake_cp_kernels(fixup) == "sm_107a":
            pytest.skip("sm_107a runs the SIMT fixups only")
        plans = _record_cake_cp_plans(monkeypatch)
    torch.manual_seed(11)
    random.seed(11)
    L = page_size
    dtype = torch.bfloat16
    totals = [p + n for p, n in zip(prefix_lens, new_lens, strict=True)]
    streams = Streams(totals, *heads, dtype)
    batch = PagedBatch(totals, L, streams.num_sab_heads, torch.float32)
    _, prefix_final, _ = _oracle(streams, prefix_lens)
    for b, p in enumerate(prefix_lens):
        if p:
            batch.pool[batch.page_of(b, p - 1)] = prefix_final[b]
    pool_before = batch.pool.clone()
    q, k, v, g, beta = streams.slices(prefix_lens, totals)
    indices, indptr, last = batch.tables()
    guard, out = _guarded_output(sum(new_lens), streams.num_sab_heads, dtype)
    chunk_gated_delta_rule_paged(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(HEAD_SIZE),
        cu_seqlens=_cu(new_lens, "cuda"),
        state_pool=batch.pool,
        page_indices=indices,
        page_indptr=indptr,
        page_size=L,
        last_page_len=last,
        output=out,
        backend="cake_gdn",
        # Equal lengths under their own bound pick the UTCMMA fixup.
        max_seqlen=None if fixup is None else max(new_lens),
    )
    torch.cuda.synchronize()
    _assert_output_guard(guard)
    batch.guarded.assert_guards()
    if fixup is not None:
        assert [plan.fixup_kernel for plan in plans] == [fixup]

    ref_out, ref_final, ref_ckpts = _oracle(streams, totals)
    good = next(b for b, p in enumerate(prefix_lens) if p % CHUNK == 0)
    new_start = _cu(new_lens, "cpu").tolist()[good]
    ref_start = _cu(totals, "cpu").tolist()[good] + prefix_lens[good]
    torch.testing.assert_close(
        out[new_start : new_start + new_lens[good]].float(),
        ref_out[ref_start : ref_start + new_lens[good]].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    written = set()
    for b, (p, t) in enumerate(zip(prefix_lens, totals, strict=True)):
        for j in range(p // L, batch.num_pages[b]):
            page = batch.pages[b][j]
            written.add(page)
            if b != good:
                assert torch.isnan(batch.pool[page]).all(), f"seq {b} page {j}"
                continue
            end = min((j + 1) * L, t)
            expected = ref_final[b] if end == t else ref_ckpts[b][end // CHUNK - 1]
            _assert_state(batch.pool[page], expected, msg=f"seq {b} page {j}")
    for page in range(batch.pool.size(0)):
        if page not in written:
            assert torch.equal(batch.pool[page], pool_before[page]), page


def _unaligned_pool(case, heads):
    row = heads * HEAD_SIZE * HEAD_SIZE
    if case == "row_pad_1":
        return GuardedPool(4, heads, torch.float32, row_pad=1).pool
    if case == "row_pad_6":
        return GuardedPool(4, heads, torch.float32, row_pad=6).pool
    flat = torch.zeros(4 * row + 1, device="cuda")
    return flat[1:].view(4, heads, HEAD_SIZE, HEAD_SIZE)


@pytest.mark.parametrize(
    "case, match",
    [
        ("row_pad_1", "row stride of whole 16-byte groups"),
        ("row_pad_6", "row stride of whole 16-byte groups"),
        ("base_plus_one_float", "16-byte aligned"),
    ],
)
def test_paged_prefill_cake_cp_rejects_unaligned_pools(case, match):
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    q = torch.zeros(64, 2, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match=match):
        chunk_gated_delta_rule_paged(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32, device="cuda"),
            state_pool=_unaligned_pool(case, 2),
            page_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
            page_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            page_size=64,
            last_page_len=torch.tensor([64], dtype=torch.int32, device="cuda"),
            backend="cake_gdn",
        )


@pytest.mark.parametrize("page_size", [96, 0])
def test_paged_prefill_cake_cp_rejects_page_size(page_size):
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    q = torch.zeros(64, 2, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    pool = torch.zeros(4, 2, HEAD_SIZE, HEAD_SIZE, device="cuda")
    reason = _cake_cp_backend().gdn_cp_paged_rejection_reason(
        q, q, q, pool, page_size=page_size
    )
    assert reason is not None and "positive multiple of 64" in reason
    with pytest.raises(ValueError, match="positive multiple of 64"):
        chunk_gated_delta_rule_paged(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 64], dtype=torch.int32, device="cuda"),
            state_pool=pool,
            page_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
            page_indptr=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            page_size=page_size,
            last_page_len=torch.tensor([64], dtype=torch.int32, device="cuda"),
            backend="cake_gdn",
        )


@pytest.mark.parametrize(
    "case, match",
    [
        ("k_tokens", "same token count"),
        ("v_tokens", "same token count"),
        ("k_head_size", "head size 128"),
        ("v_head_size", "head size 128"),
        ("pool_on_cpu", "on one device"),
    ],
)
def test_cake_cp_paged_rejection_reason(case, match):
    _skip_if_unsupported()
    _skip_unless_cake_cp()

    def tokens(n, size=HEAD_SIZE):
        return torch.zeros(n, 2, size, dtype=torch.bfloat16, device="cuda")

    q, k, v = tokens(128), tokens(128), tokens(128)
    pool = torch.zeros(4, 2, HEAD_SIZE, HEAD_SIZE, device="cuda")
    if case == "k_tokens":
        k = tokens(64)
    elif case == "v_tokens":
        v = tokens(192)
    elif case == "k_head_size":
        k = tokens(128, 64)
    elif case == "v_head_size":
        v = tokens(128, 256)
    else:
        pool = pool.cpu()
    reason = _cake_cp_backend().gdn_cp_paged_rejection_reason(
        q, k, v, pool, page_size=64
    )
    assert reason is not None and match in reason, reason


@pytest.mark.parametrize(
    "arch, nvcc, reason",
    [
        ("sm_107a", (13, 3), "requires CUDA 13.4 or newer on sm_107a"),
        ("sm_107a", (13, 4), None),
        ("sm_103a", (13, 0), None),
        ("sm_103a", (12, 8), None),
        ("sm_103a", (12, 6), "requires a CUDA 12.8 or newer toolkit"),
        ("sm_107a", (12, 6), "requires a CUDA 12.8 or newer toolkit"),
    ],
)
def test_cake_cp_paged_rejection_reason_needs_cuda_13_4_on_sm107(
    monkeypatch, arch, nvcc, reason
):
    """The paged programs compile with the installed nvcc: it needs CUDA 12.8,
    and 13.4 for the sm_107a programs."""
    _skip_if_unsupported()
    # The nvcc version is patched, so the check runs with any installed nvcc.
    _skip_unless_cake_cp(needs_nvcc=False)
    gdn_cp = _cake_cp_backend()

    def at_least(version):
        assert version in ("12.8", "13.4")
        return nvcc >= tuple(int(part) for part in version.split("."))

    monkeypatch.setattr(gdn_cp, "_arch_for", lambda _device: arch)
    monkeypatch.setattr(gdn_cp, "is_cuda_version_at_least", at_least)
    q = torch.zeros(128, 2, HEAD_SIZE, dtype=torch.bfloat16, device="cuda")
    pool = torch.zeros(4, 2, HEAD_SIZE, HEAD_SIZE, device="cuda")
    assert gdn_cp.gdn_cp_paged_rejection_reason(q, q, q, pool, page_size=64) == reason


def _misaligned(shape, dtype):
    """A contiguous tensor 2 bytes past a 16-byte boundary."""
    flat = torch.zeros(math.prod(shape) + 1, dtype=dtype, device="cuda")
    return flat[1:].view(shape)


@pytest.mark.parametrize(
    "case, match",
    [
        ("q", "16-byte aligned q/k/v"),
        ("k", "16-byte aligned q/k/v"),
        ("v", "16-byte aligned q/k/v"),
        ("output", "output must be 16-byte aligned"),
        ("one_row_pool", "row stride of whole 16-byte groups"),
        ("grid_y", "at most 1 sequences"),
        ("tensor_map_offset", "tensor-map offsets"),
        ("max_seqlen_below_mean", "cannot be smaller"),
    ],
)
def test_paged_prefill_cake_cp_rejects_before_any_launch(monkeypatch, case, match):
    """What stage 4 would reject only after the fixup has written the pool
    (misaligned q/k/v or output, a one-row pool whose row stride is not whole
    16-byte groups), grids past the launch or 32-bit tensor-map offset
    limits, and a ``max_seqlen`` below ``ceil(total / num_seqs)`` raise
    before any launch: the pool is unchanged."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    gdn_cp = _cake_cp_backend()
    torch.manual_seed(3)
    heads = 2
    new_lens = [128] if case == "one_row_pool" else [128, 128]
    num_seqs = len(new_lens)
    shape = (sum(new_lens), heads, HEAD_SIZE)
    qkv = {
        name: torch.randn(shape, device="cuda").to(torch.bfloat16)
        for name in ("q", "k", "v")
    }
    output = None
    if case in qkv:
        qkv[case] = _misaligned(shape, torch.bfloat16).copy_(qkv[case])
    elif case == "output":
        output = _misaligned(shape, torch.bfloat16)
    elif case == "grid_y":
        monkeypatch.setattr(gdn_cp, "_MAX_GRID_Y", 1)
    elif case == "tensor_map_offset":
        monkeypatch.setattr(gdn_cp, "_MAX_TENSOR_MAP_OFFSET", 0)
    row = heads * HEAD_SIZE * HEAD_SIZE
    pool = torch.empty_strided(
        (num_seqs, heads, HEAD_SIZE, HEAD_SIZE),
        (row + (case == "one_row_pool"), HEAD_SIZE * HEAD_SIZE, HEAD_SIZE, 1),
        device="cuda",
    )
    pool.copy_(torch.randn(pool.shape, device="cuda"))
    before = pool.clone()
    with pytest.raises(ValueError, match=match):
        chunk_gated_delta_rule_paged(
            qkv["q"],
            qkv["k"],
            qkv["v"],
            cu_seqlens=_cu(new_lens, "cuda"),
            state_pool=pool,
            page_indices=torch.arange(num_seqs, dtype=torch.int32, device="cuda"),
            page_indptr=torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda"),
            page_size=128,
            last_page_len=torch.full(
                (num_seqs,), 128, dtype=torch.int32, device="cuda"
            ),
            output=output,
            backend="cake_gdn",
            max_seqlen=127 if case == "max_seqlen_below_mean" else None,
        )
    torch.cuda.synchronize()
    assert torch.equal(pool, before)


_INVALID_ROW_CHILD = """
import sys
import torch
from flashinfer.gdn_paged import chunk_gated_delta_rule_paged, gdn_paged_prefill_slots

case = sys.argv[1]
pool = torch.zeros(4, 4, 128, 128, device="cuda")
q = torch.zeros(128, 4, 128, dtype=torch.bfloat16, device="cuda")
pages = {
    "valid": [0, 1],
    "duplicate": [0, 0],
    "out_of_pool": [0, 4],
    "overlapping_starts": [0, 1],
    "misaligned_prefix": [0, 1],
    "ragged": [0, 1, 2],
    "short_max_seqlen": [0, 1, 2],
}[case]
# Two sequences of one page each, except "ragged" and "short_max_seqlen": 32
# and 96 new tokens (seq 1 over two pages); "short_max_seqlen" passes
# max_seqlen=64, which meets the host bound ceil(128 / 2) but is less than
# seq 1's new tokens.
short = case in ("ragged", "short_max_seqlen")
cu_seqlens = torch.tensor(
    [0, 32, 128] if short else [0, 64, 128], dtype=torch.int32, device="cuda"
)
page_indices = torch.tensor(pages, dtype=torch.int32, device="cuda")
page_indptr = torch.tensor(
    [0, 1, 3] if short else [0, 1, 2], dtype=torch.int32, device="cuda"
)
last_page_len = torch.tensor(
    [32, 32] if short else [64, 64], dtype=torch.int32, device="cuda"
)
slots = gdn_paged_prefill_slots(
    cu_seqlens, page_indices, page_indptr, 64, last_page_len
)
if case == "overlapping_starts":
    # Both sequences write their page-end checkpoint to entry 0 (row 0).
    starts = torch.zeros_like(slots.checkpoint_cu_starts)
    slots = slots._replace(checkpoint_cu_starts=starts)
if case == "misaligned_prefix":
    # Rows and checkpoint ranges stay valid; only the prefix is not 64-aligned.
    slots = slots._replace(prefix_lens=torch.full_like(slots.prefix_lens, 32))
chunk_gated_delta_rule_paged(
    q,
    q,
    q,
    cu_seqlens=cu_seqlens,
    state_pool=pool,
    page_indices=page_indices,
    page_indptr=page_indptr,
    page_size=64,
    last_page_len=last_page_len,
    slots=slots,
    backend="cake_gdn",
    max_seqlen=64 if case == "short_max_seqlen" else None,
)
torch.cuda.synchronize()
"""


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "duplicate",
        "out_of_pool",
        "overlapping_starts",
        "misaligned_prefix",
        "ragged",
        "short_max_seqlen",
    ],
)
def test_paged_prefill_cake_cp_invalid_rows_fail_in_isolated_process(case):
    """With FLASHINFER_CAKE_GDN_VALIDATE_SLOTS=1 a row written by two sequences
    (through duplicate page ids or overlapping checkpoint ranges) or past the
    pool, a prefix that is not 64-aligned, or a sequence with more new tokens
    than ``max_seqlen`` fails a device assert (in a child: it poisons CUDA),
    and distinct rows pass, also for a ragged batch without the hint."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])]
    )
    env["FLASHINFER_CAKE_GDN_VALIDATE_SLOTS"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", _INVALID_ROW_CHILD, case],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    combined = completed.stdout + completed.stderr
    if case in ("valid", "ragged"):
        assert completed.returncode == 0, combined
        return
    assert completed.returncode != 0, combined
    assert "_assert_async_cuda_kernel" in combined, combined


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_paged_prefill_unaligned_first_sequence_padded_pool(state_dtype):
    """The first sequence has an unaligned prefix, so its chunking starts before
    output row 0, and the pool rows are padded (row stride > H * V * K)."""
    _skip_if_unsupported()
    random.seed(8)
    torch.manual_seed(8)
    L = 128
    _run_paged_prefill_case(
        [37, 0, L + 5],
        [100, 64, 30],
        L,
        (4, 4, 4),
        torch.bfloat16,
        state_dtype,
        row_pad=96,
    )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_paged_prefill_empty_and_boundary_spans(index_dtype):
    """Zero-length spans leave the pool alone; spans ending exactly on a page
    boundary store the final state as that page's checkpoint."""
    _skip_if_unsupported()
    random.seed(1)
    torch.manual_seed(1)
    L = 128
    prefix_lens = [L, 70, 0, L, 2 * L - 64]
    new_lens = [0, 0, L, L, 64]
    _run_paged_prefill_case(
        prefix_lens,
        new_lens,
        L,
        (4, 4, 4),
        torch.bfloat16,
        torch.float32,
        index_dtype=index_dtype,
    )


def test_prefix_reuse_does_not_write_shared_pages():
    """Sequence B reuses the two full pages of sequence A; its suffix starts
    from A's page-boundary state and writes only B's own pages."""
    _skip_if_unsupported()
    random.seed(2)
    torch.manual_seed(2)
    L = 256
    a_len, shared_pages, b_suffix = 2 * L + 100, 2, L + 33
    streams = Streams([a_len], 4, 4, 4, torch.bfloat16)
    # B's stream = A's first two pages + its own suffix tokens.
    b_streams = Streams([shared_pages * L + b_suffix], 4, 4, 4, torch.bfloat16)
    for name in ("q", "k", "v", "g", "beta"):
        getattr(b_streams, name)[: shared_pages * L] = getattr(streams, name)[
            : shared_pages * L
        ]

    pool = torch.full((16, 4, HEAD_SIZE, HEAD_SIZE), 7.0, device="cuda")
    a_pages = [11, 3, 8]
    b_pages = a_pages[:shared_pages] + [5, 14]
    scale = 1.0 / math.sqrt(HEAD_SIZE)

    def run(s: Streams, pages, prefix, new):
        q, k, v, g, beta = s.slices([prefix], [prefix + new])
        total = prefix + new
        return chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            scale,
            cu_seqlens=_cu([new], "cuda"),
            state_pool=pool,
            page_indices=torch.tensor(pages, dtype=torch.int32, device="cuda"),
            page_indptr=_cu([len(pages)], "cuda"),
            page_size=L,
            last_page_len=torch.tensor(
                [total - (len(pages) - 1) * L], dtype=torch.int32, device="cuda"
            ),
        )

    run(streams, a_pages, 0, a_len)
    a_after = pool.clone()
    b_out = run(b_streams, b_pages, shared_pages * L, b_suffix)
    torch.cuda.synchronize()

    for page in a_pages:
        assert torch.equal(pool[page], a_after[page]), f"shared/A page {page} changed"
    ref_out, ref_final, ref_ckpts = _oracle(b_streams, [shared_pages * L + b_suffix])
    torch.testing.assert_close(
        b_out.float(), ref_out[shared_pages * L :].float(), atol=1e-2, rtol=1e-2
    )
    _assert_state(pool[5], ref_ckpts[0][3 * L // CHUNK - 1], msg="B page 2")
    _assert_state(pool[14], ref_final[0], msg="B page 3")


def test_chunked_prefill_continuation_at_unaligned_offsets():
    """The packing of issue #2873's discussion: two requests of 1040 and 1050
    tokens served by 1024-token prefill chunks, [1024], [16 + 1008], [42], so
    the second request continues from token 1008 and crosses its page boundary
    at token 1024, 16 tokens into the third chunk."""
    _skip_if_unsupported()
    random.seed(3)
    torch.manual_seed(3)
    L = 1024
    streams = Streams([1040, 1050], 4, 4, 4, torch.bfloat16)
    batch = PagedBatch([1040, 1050], L, 4, torch.float32)
    pool_before = batch.pool.clone()
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    # Outputs are causal: the whole-stream oracle gives every chunk's rows.
    ref_out, ref_final, ref_ckpts = _oracle(streams, [1040, 1050])
    done = [0, 0]
    for chunk in ([1024, 0], [16, 1008], [0, 42]):
        new_done = [d + c for d, c in zip(done, chunk, strict=True)]
        q, k, v, g, beta = streams.slices(done, new_done)
        pages = [math.ceil(n / L) if n else 1 for n in new_done]
        used = [batch.pages[b][: pages[b]] for b in range(2)]
        guard, out = _guarded_output(sum(chunk), 4, torch.bfloat16)
        chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            scale,
            cu_seqlens=_cu(chunk, "cuda"),
            state_pool=batch.pool,
            page_indices=torch.tensor(
                used[0] + used[1], dtype=torch.int32, device="cuda"
            ),
            page_indptr=_cu(pages, "cuda"),
            page_size=L,
            last_page_len=torch.tensor(
                [max(n - (m - 1) * L, 1) for n, m in zip(new_done, pages, strict=True)],
                dtype=torch.int32,
                device="cuda",
            ),
            output=out,
        )
        # The last call's sequence 1 continues from token 1008 right after
        # an empty sequence: its chunking starts 48 rows before row 0.
        torch.cuda.synchronize()
        _assert_output_guard(guard)
        ref_rows = torch.cat(
            [
                torch.arange(streams.starts[b] + done[b], streams.starts[b] + n)
                for b, n in enumerate(new_done)
            ]
        ).cuda()
        torch.testing.assert_close(
            out.float(), ref_out[ref_rows].float(), atol=1e-2, rtol=1e-2
        )
        done = new_done
    batch.guarded.assert_guards()
    used = {page for pages in batch.pages for page in pages}
    for page in range(batch.pool.size(0)):
        if page not in used:
            assert torch.equal(batch.pool[page], pool_before[page]), (
                f"page {page} belongs to no request but changed"
            )
    for b in range(2):
        _assert_state(
            batch.pool[batch.pages[b][0]],
            ref_ckpts[b][L // CHUNK - 1],
            msg=f"seq {b} state at token 1024",
        )
        _assert_state(batch.pool[batch.pages[b][1]], ref_final[b], msg=f"seq {b} end")


# Index dtypes of the slot-argument tests: all int32, all int64, and a mix.
_INDEX_DTYPES = {
    "int32": dict.fromkeys(("cu", "read", "write", "ckpt", "starts", "prefix"), 32),
    "int64": dict.fromkeys(("cu", "read", "write", "ckpt", "starts", "prefix"), 64),
    "mixed": {
        "cu": 32,
        "read": 32,
        "write": 64,
        "ckpt": 64,
        "starts": 64,
        "prefix": 64,
    },
}


def _idx(values, bits):
    dtype = torch.int32 if bits == 32 else torch.int64
    if isinstance(values, torch.Tensor):
        return values.to(dtype)
    return torch.tensor(values, dtype=dtype, device="cuda")


@pytest.mark.parametrize("full_starts", [False, True])
@pytest.mark.parametrize("index_dtypes", list(_INDEX_DTYPES))
def test_slot_arguments_match_packed_api(index_dtypes, full_starts):
    """The low-level slot arguments against the packed API: -1 read rows start
    from zero, -1 write rows and -1 checkpoint rows are skipped. With
    checkpoint rows, ``checkpoint_cu_starts`` may have ``num_seqs`` or
    ``num_seqs + 1`` entries."""
    _skip_if_unsupported()
    bits = _INDEX_DTYPES[index_dtypes]
    random.seed(4)
    torch.manual_seed(4)
    lens = [130, 64, 200]
    streams = Streams(lens, 4, 4, 4, torch.bfloat16)
    q, k, v, g, beta = streams.slices([0, 0, 0], lens)
    cu = _idx(_cu(lens, "cuda"), bits["cu"])
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    ref_out, ref_final, ref_ckpts = _oracle(streams, lens)

    guarded = GuardedPool(12, 4, torch.float32)
    pool = guarded.pool
    pool.copy_(torch.randn_like(pool))
    before = pool.clone()
    read = _idx([-1, -1, -1], bits["read"])
    write = _idx([9, -1, 2], bits["write"])
    # 64-token checkpoints: 2 + 1 + 3; route seq 2's second one to -1.
    ckpt_rows = _idx([4, 6, 0, 1, -1, 10], bits["ckpt"])
    out, _ = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale,
        pool,
        True,
        cu,
        output_state=pool,
        state_checkpoints=pool,
        checkpoint_cu_starts=_idx(
            [0, 2, 3, 6] if full_starts else [0, 2, 3], bits["starts"]
        ),
        checkpoint_every_n_tokens=CHUNK,
        state_indices=read,
        output_state_indices=write,
        checkpoint_state_indices=ckpt_rows,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)
    guarded.assert_guards()
    expected = {
        9: ref_final[0],
        2: ref_final[2],
        4: ref_ckpts[0][0],
        6: ref_ckpts[0][1],
        0: ref_ckpts[1][0],
        1: ref_ckpts[2][0],
        10: ref_ckpts[2][2],
    }
    for row in range(pool.size(0)):
        if row in expected:
            _assert_state(pool[row], expected[row], msg=f"row {row}")
        else:
            assert torch.equal(pool[row], before[row]), f"row {row} changed"


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_output_state_indices_without_checkpoints(state_dtype, index_dtype):
    """Separate read and write rows with no checkpoints: a -1 read row starts
    from zero, a -1 write row is skipped, read rows stay unchanged, and a
    zero-length sequence writes nothing to its write row."""
    _skip_if_unsupported()
    random.seed(7)
    torch.manual_seed(7)
    prefix = [0, 70, 128, 50]
    new = [130, 64, 200, 0]
    totals = [p + n for p, n in zip(prefix, new, strict=True)]
    streams = Streams(totals, 4, 4, 4, torch.bfloat16)
    _, prefix_final, _ = _oracle(streams, prefix, state_dtype)
    ref_out, ref_final, _ = _oracle(streams, totals, state_dtype)
    q, k, v, g, beta = streams.slices(prefix, totals)

    guarded = GuardedPool(12, 4, state_dtype)
    pool = guarded.pool
    pool.copy_(torch.randn_like(pool))
    read_rows = [-1, 3, 5, 8]
    write_rows = [1, 7, -1, 10]
    for b, row in enumerate(read_rows):
        if row >= 0:
            pool[row] = prefix_final[b]
    before = pool.clone()
    guard, out = _guarded_output(sum(new), 4, torch.bfloat16)
    chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(HEAD_SIZE),
        pool,
        True,
        _cu(new, "cuda").to(index_dtype),
        output_state=pool,
        state_indices=torch.tensor(read_rows, dtype=index_dtype, device="cuda"),
        output_state_indices=torch.tensor(write_rows, dtype=index_dtype, device="cuda"),
        output=out,
    )
    torch.cuda.synchronize()
    _assert_output_guard(guard)
    guarded.assert_guards()
    ref_rows = torch.cat(
        [
            torch.arange(s + p, s + t)
            for s, p, t in zip(streams.starts[:-1], prefix, totals, strict=True)
        ]
    ).cuda()
    torch.testing.assert_close(
        out.float(), ref_out[ref_rows].float(), atol=1e-2, rtol=1e-2
    )
    expected = {1: ref_final[0], 7: ref_final[1]}
    for row in range(pool.size(0)):
        if row in expected:
            _assert_state(pool[row], expected[row], msg=f"row {row}")
        else:
            assert torch.equal(pool[row], before[row]), f"row {row} changed"


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_one_selected_state_per_sequence(state_dtype):
    """One chunk-aligned intermediate state per sequence (or none) goes straight
    to its own row: every other 64-token checkpoint entry is -1, so nothing
    else is stored and no gather follows. The call passes no initial or final
    state, so the checkpoint buffer alone sets the state dtype."""
    _skip_if_unsupported()
    random.seed(6)
    torch.manual_seed(6)
    lens = [300, 130, 64, 257]
    # Sequence-local token counts (multiples of 64); -1 = no state wanted.
    positions = [192, -1, 64, 256]
    dest_rows = [5, 3, 0, 6]
    streams = Streams(lens, 4, 4, 4, torch.bfloat16)
    q, k, v, g, beta = streams.slices([0] * len(lens), lens)
    cu = _cu(lens, "cuda")
    ref_out, _, ref_ckpts = _oracle(streams, lens)

    # Built on the device from the request's positions.
    counts = (cu[1:] - cu[:-1]) // CHUNK
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.tensor(positions, dtype=torch.int32, device="cuda")
    rows = torch.tensor(dest_rows, dtype=torch.int32, device="cuda")
    entries = torch.full(
        (sum(n // CHUNK for n in lens),), -1, dtype=torch.int32, device="cuda"
    )
    wanted = pos > 0
    entries[(starts + pos // CHUNK - 1)[wanted]] = rows[wanted]

    guarded = GuardedPool(8, 4, state_dtype, row_pad=64)
    states = guarded.pool
    states.copy_(torch.randn_like(states))
    before = states.clone()
    out = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(HEAD_SIZE),
        cu_seqlens=cu,
        state_checkpoints=states,
        checkpoint_cu_starts=starts.to(torch.int32),
        checkpoint_every_n_tokens=CHUNK,
        checkpoint_state_indices=entries,
    )
    torch.cuda.synchronize()
    guarded.assert_guards()
    torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)
    expected = {
        row: ref_ckpts[b][p // CHUNK - 1]
        for b, (p, row) in enumerate(zip(positions, dest_rows, strict=True))
        if p > 0
    }
    for row in range(states.size(0)):
        if row in expected:
            _assert_state(states[row], expected[row], msg=f"row {row}")
        else:
            assert torch.equal(states[row], before[row]), f"row {row} changed"


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_packed_checkpoints_without_state(state_dtype):
    """The packed API with checkpoints only, no initial or final state: the
    checkpoint buffer sets the state dtype."""
    _skip_if_unsupported()
    random.seed(9)
    torch.manual_seed(9)
    lens = [130, 64, 257]
    streams = Streams(lens, 4, 4, 4, torch.bfloat16)
    q, k, v, g, beta = streams.slices([0] * len(lens), lens)
    ref_out, _, ref_ckpts = _oracle(streams, lens, state_dtype)
    counts = [n // CHUNK for n in lens]
    ckpts = torch.zeros(
        sum(counts), 4, HEAD_SIZE, HEAD_SIZE, dtype=state_dtype, device="cuda"
    )
    out = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(HEAD_SIZE),
        cu_seqlens=_cu(lens, "cuda"),
        state_checkpoints=ckpts,
        checkpoint_cu_starts=_cu(counts, "cuda"),
        checkpoint_every_n_tokens=CHUNK,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)
    for i, expected in enumerate(torch.cat(ref_ckpts)):
        _assert_state(ckpts[i], expected, msg=f"checkpoint {i}")


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_rows_deterministic_under_store_load(state_dtype):
    """Checkpoints into pool rows are bit-identical in repeated identical calls
    and match the packed checkpoints. One pool row per 64-token checkpoint (the
    paged API with 64-token pages), and one selected row per sequence three
    blocks before its end, on 64 sequences x 32 heads: every SM stores
    checkpoints while the other math warp group still reads its K stage. On
    SM120 the release of that stage once overtook the read, and the next TMA
    fill corrupted a 16 x 16 tile of the state in about half of the calls."""
    _skip_if_unsupported()
    torch.manual_seed(13)
    num_seqs, seq_len = 64, 512
    num_q_heads, num_k_heads, num_v_heads = 16, 16, 32
    heads = max(num_q_heads, num_v_heads)
    lens = [seq_len] * num_seqs
    total = sum(lens)
    q = torch.randn(total, num_q_heads, HEAD_SIZE, device="cuda").to(torch.bfloat16)
    k = F.normalize(torch.randn(total, num_k_heads, HEAD_SIZE, device="cuda"), dim=-1)
    k = k.to(torch.bfloat16)
    v = torch.randn(total, num_v_heads, HEAD_SIZE, device="cuda").to(torch.bfloat16)
    q, v = q * 0.1, v * 0.1
    g = 0.9 + 0.1 * torch.rand(total, heads, device="cuda")
    beta = torch.rand(total, heads, device="cuda")
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    cu = _cu(lens, "cuda")
    counts = [n // CHUNK for n in lens]
    starts = _cu(counts, "cuda")
    num_ckpts = sum(counts)
    packed = torch.zeros(
        num_ckpts, heads, HEAD_SIZE, HEAD_SIZE, dtype=state_dtype, device="cuda"
    )
    ref_out = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale,
        cu_seqlens=cu,
        state_checkpoints=packed,
        checkpoint_cu_starts=starts,
        checkpoint_every_n_tokens=CHUNK,
    )
    selected = starts[:-1].long() + (counts[0] - 3)
    selected_entries = torch.full((num_ckpts,), -1, dtype=torch.int32, device="cuda")
    selected_entries[selected] = torch.arange(
        num_seqs, dtype=torch.int32, device="cuda"
    )
    cases = {
        "every": (torch.arange(num_ckpts, dtype=torch.int32, device="cuda"), packed),
        "selected": (selected_entries, packed[selected]),
    }
    for name, (entries, expected) in cases.items():
        rows = torch.empty_like(expected)
        first = None
        for call in range(8):
            rows.zero_()
            out = chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                scale,
                cu_seqlens=cu,
                state_checkpoints=rows,
                checkpoint_cu_starts=starts[:-1],
                checkpoint_every_n_tokens=CHUNK,
                checkpoint_state_indices=entries,
            )
            torch.cuda.synchronize()
            if first is None:
                first = (out.clone(), rows.clone())
                torch.testing.assert_close(
                    out.float(), ref_out.float(), atol=1e-2, rtol=1e-2
                )
                for lo in range(0, rows.size(0), num_seqs):
                    hi = lo + num_seqs
                    _assert_state(rows[lo:hi], expected[lo:hi], msg=f"{name} {lo}")
                continue
            assert torch.equal(out, first[0]), f"{name}, call {call}: output"
            assert torch.equal(rows, first[1]), f"{name}, call {call}: checkpoints"


@pytest.mark.parametrize("index_dtypes", list(_INDEX_DTYPES))
def test_prefix_lens_absolute_checkpoints(index_dtypes):
    """``prefix_lens`` moves the checkpoint grid to absolute positions; an
    unaligned prefix is chunked from its 64-aligned absolute position."""
    _skip_if_unsupported()
    bits = _INDEX_DTYPES[index_dtypes]
    random.seed(5)
    torch.manual_seed(5)
    prefix = [100, 64, 0]
    new = [200, 192, 130]
    totals = [p + n for p, n in zip(prefix, new, strict=True)]
    streams = Streams(totals, 4, 4, 4, torch.bfloat16)
    _, prefix_final, _ = _oracle(streams, prefix)
    ref_out, ref_final, ref_ckpts = _oracle(streams, totals)
    q, k, v, g, beta = streams.slices(prefix, totals)
    N = 128
    counts = [(p + n) // N - p // N for p, n in zip(prefix, new, strict=True)]
    ckpts = torch.zeros(sum(counts), 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    init = prefix_final.clone()
    final = torch.zeros_like(init)
    # Sequence 0 (prefix 100) is chunked from absolute position 64, 36 rows
    # before output row 0.
    guard, out = _guarded_output(sum(new), 4, torch.bfloat16)
    out, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(HEAD_SIZE),
        init,
        True,
        _idx(_cu(new, "cuda"), bits["cu"]),
        output_state=final,
        state_checkpoints=ckpts,
        checkpoint_cu_starts=_idx(_cu(counts, "cuda"), bits["starts"]),
        checkpoint_every_n_tokens=N,
        prefix_lens=_idx(prefix, bits["prefix"]),
        output=out,
    )
    torch.cuda.synchronize()
    _assert_output_guard(guard)
    ref_rows = torch.cat(
        [
            torch.arange(s + p, s + t)
            for s, p, t in zip(
                _cu(totals, "cpu").tolist()[:-1], prefix, totals, strict=True
            )
        ]
    ).cuda()
    torch.testing.assert_close(
        out.float(), ref_out[ref_rows].float(), atol=1e-2, rtol=1e-2
    )
    row = 0
    for b in range(3):
        _assert_state(final[b], ref_final[b], msg=f"seq {b} final")
        first = prefix[b] // N + 1
        for j in range(counts[b]):
            position = (first + j) * N
            _assert_state(
                ckpts[row],
                ref_ckpts[b][position // CHUNK - 1],
                msg=f"seq {b} @{position}",
            )
            row += 1


@pytest.mark.parametrize("omit", ["g", "beta", "both"])
def test_prefix_lens_without_gates(omit):
    """``prefix_lens`` with ``g`` or ``beta`` omitted: the identity rows that
    pad the first chunk of an unaligned prefix still continue the
    whole-stream result."""
    _skip_if_unsupported()
    random.seed(13)
    torch.manual_seed(13)
    prefix, new = [100, 37], [200, 91]
    totals = [p + n for p, n in zip(prefix, new, strict=True)]
    streams = Streams(totals, 4, 4, 4, torch.bfloat16)

    def run(lo, hi, initial_state=None, prefix_lens=None):
        q, k, v, g, beta = streams.slices(lo, hi)
        if omit in ("g", "both"):
            g = None
        if omit in ("beta", "both"):
            beta = None
        base = prefix_lens or [0] * len(lo)
        counts = [
            (p + h - l) // CHUNK - p // CHUNK
            for p, l, h in zip(base, lo, hi, strict=True)
        ]
        ckpts = torch.zeros(sum(counts), 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
        out, final = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(HEAD_SIZE),
            initial_state,
            True,
            _cu([h - l for l, h in zip(lo, hi, strict=True)], "cuda"),
            state_checkpoints=ckpts,
            checkpoint_cu_starts=_cu(counts, "cuda"),
            checkpoint_every_n_tokens=CHUNK,
            prefix_lens=None
            if prefix_lens is None
            else torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
        )
        starts = _cu(counts, "cpu").tolist()
        return out, final, [ckpts[starts[b] : starts[b + 1]] for b in range(len(lo))]

    ref_out, ref_final, ref_ckpts = run([0, 0], totals)
    _, prefix_final, _ = run([0, 0], prefix)
    out, final, ckpts = run(prefix, totals, prefix_final, prefix)
    torch.cuda.synchronize()
    ref_rows = torch.cat(
        [
            torch.arange(s + p, s + t)
            for s, p, t in zip(streams.starts[:-1], prefix, totals, strict=True)
        ]
    ).cuda()
    torch.testing.assert_close(
        out.float(), ref_out[ref_rows].float(), atol=1e-2, rtol=1e-2
    )
    for b in range(len(prefix)):
        _assert_state(final[b], ref_final[b], msg=f"seq {b} final")
        for j, ckpt in enumerate(ckpts[b]):
            position = (prefix[b] // CHUNK + 1 + j) * CHUNK
            _assert_state(
                ckpt, ref_ckpts[b][position // CHUNK - 1], msg=f"seq {b} @{position}"
            )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"backend": "cake_gdn"}, "native non-CP"),
        ({"backend": "cudnn"}, "native non-CP"),
        ({"use_cp": True}, "context parallelism"),
    ],
)
def test_slot_arguments_reject_other_routes(kwargs, match):
    _skip_if_unsupported()
    q, k, v = gen_qkv([64], 4, 4, 4, HEAD_SIZE, torch.bfloat16)
    pool = torch.zeros(2, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    idx = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match=match):
        chunk_gated_delta_rule(
            q.cuda(),
            k.cuda(),
            v.cuda(),
            None,
            None,
            None,
            pool,
            True,
            _cu([64], "cuda"),
            output_state=pool,
            state_indices=idx,
            output_state_indices=idx,
            **kwargs,
        )


def _strided(x: torch.Tensor) -> torch.Tensor:
    """``x`` as column 0 of a ``[n, 2]`` table (stride 2)."""
    return torch.stack([x, x], 1)[:, 0]


@pytest.mark.parametrize(
    "case, match",
    [
        ("write_without_read", "output_state_indices requires state_indices"),
        ("write_without_output_state", "explicit output_state pool"),
        ("write_without_final_state", "explicit output_state pool"),
        ("checkpoint_rows_without_every_n", "checkpoint_every_n_tokens > 0"),
        ("prefix_without_every_n", "checkpoint_every_n_tokens > 0"),
        ("float_rows", "must be int32 or int64"),
        ("2d_rows", "must be 1D"),
        ("long_rows", "must be 1D of 2 entries"),
        ("cpu_rows", "must be on cuda"),
        ("strided_rows", "output_state_indices must be contiguous"),
        ("starts_length", "checkpoint_cu_starts must have 2 or 3 elements"),
        ("sparse_checkpoint_rows", "pool rows must be dense"),
    ],
)
def test_slot_arguments_reject_bad_inputs(case, match):
    """Every check fires on the host, before any launch."""
    _skip_if_unsupported()
    q, k, v = gen_qkv([64, 64], 4, 4, 4, HEAD_SIZE, torch.bfloat16)
    pool = torch.zeros(4, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    rows = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    kwargs = dict(
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=_cu([64, 64], "cuda"),
        output_state=pool,
        state_indices=rows,
        output_state_indices=rows,
    )
    if case == "write_without_read":
        del kwargs["state_indices"]
    elif case == "write_without_output_state":
        del kwargs["output_state"]
    elif case == "write_without_final_state":
        kwargs["output_final_state"] = False
    elif case == "checkpoint_rows_without_every_n":
        del kwargs["output_state_indices"]
        kwargs["checkpoint_state_indices"] = rows
    elif case == "prefix_without_every_n":
        del kwargs["output_state_indices"]
        kwargs["prefix_lens"] = rows
    elif case == "float_rows":
        kwargs["output_state_indices"] = rows.float()
    elif case == "2d_rows":
        kwargs["output_state_indices"] = rows.view(1, 2)
    elif case == "long_rows":
        kwargs["output_state_indices"] = torch.cat([rows, rows[:1]])
    elif case == "cpu_rows":
        kwargs["output_state_indices"] = rows.cpu()
    elif case == "strided_rows":
        kwargs["output_state_indices"] = _strided(rows)
    else:
        # One 64-token checkpoint per sequence, routed to pool rows.
        del kwargs["output_state_indices"]
        checkpoints = pool
        starts = rows[:1]
        if case == "sparse_checkpoint_rows":
            # Rows of [H, V, K + 8]: each [H, V, K] row is not one dense block.
            checkpoints = torch.zeros(4, 4, HEAD_SIZE, HEAD_SIZE + 8, device="cuda")[
                ..., :HEAD_SIZE
            ]
            starts = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        kwargs.update(
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=starts,
            checkpoint_every_n_tokens=CHUNK,
            checkpoint_state_indices=rows,
        )
    with pytest.raises((ValueError, RuntimeError), match=match):
        chunk_gated_delta_rule(q.cuda(), k.cuda(), v.cuda(), **kwargs)


def test_slot_calls_are_not_traced():
    """``chunk_gated_delta_rule``'s trace has no slot arguments: a slot call
    gets no template, so it is not dumped and Apply keeps the original call."""
    from flashinfer.trace.templates.gdn import gdn_prefill_trace

    rows = torch.zeros(1, dtype=torch.int32)
    for name in ("output_state_indices", "checkpoint_state_indices", "prefix_lens"):
        assert gdn_prefill_trace(save_dir=None, name=None, **{name: rows}) is None
    assert gdn_prefill_trace(save_dir=None, name=None) is not None


def test_checkpoint_key_range():
    """The 32-bit checkpoint key of the SM8x/SM90/SM120 slot paths."""
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
        check_checkpoint_key_range,
    )

    # (entries + 2) * N + tokens must stay below 2**31.
    check_checkpoint_key_range(2**31 // CHUNK - 3, CHUNK, CHUNK - 1)
    with pytest.raises(RuntimeError, match="exceeds 32 bits"):
        check_checkpoint_key_range(2**31 // CHUNK - 3, CHUNK, CHUNK)


def _page_tables(values: dict, strided=None):
    """int32 tensors of ``values``; the one named ``strided`` is column 0 of a
    ``[n, 2]`` table (stride 2), as a view into a caller's metadata would be."""
    tables = {}
    for name, x in values.items():
        if name == strided:
            table = torch.tensor([[v, 0] for v in x], dtype=torch.int32, device="cuda")
            tables[name] = table[:, 0]
        else:
            tables[name] = torch.tensor(x, dtype=torch.int32, device="cuda")
    return tables


@pytest.mark.parametrize(
    "page_size, strided, match",
    [
        (100, None, "multiple of 64"),
        (128.0, None, "Python int"),
        (128, "page_indices", "page_indices must be contiguous"),
        (128, "page_indptr", "page_indptr must be contiguous"),
        (128, "last_page_len", "last_page_len must be contiguous"),
        (128, "cu_seqlens", "cu_seqlens must be a contiguous"),
    ],
)
def test_paged_prefill_rejects_bad_arguments(page_size, strided, match):
    q, k, v = gen_qkv([16], 4, 4, 4, HEAD_SIZE, torch.bfloat16)
    pool = torch.zeros(4, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    # Two sequences of 8 new tokens, on pages [0, 1] and [2].
    tables = _page_tables(
        {
            "cu_seqlens": [0, 8, 16],
            "page_indices": [0, 1, 2],
            "page_indptr": [0, 2, 3],
            "last_page_len": [8, 8],
        },
        strided,
    )
    with pytest.raises(ValueError, match=match):
        chunk_gated_delta_rule_paged(
            q.cuda(),
            k.cuda(),
            v.cuda(),
            state_pool=pool,
            page_size=page_size,
            **tables,
        )


def test_paged_prefill_rejects_cu_seqlens_off_device():
    q, k, v = gen_qkv([16], 4, 4, 4, HEAD_SIZE, torch.bfloat16)
    pool = torch.zeros(4, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    tables = _page_tables(
        {"page_indices": [0, 1, 2], "page_indptr": [0, 2, 3], "last_page_len": [8, 8]}
    )
    cu = _cu([8, 8], "cpu")
    with pytest.raises(ValueError, match="cu_seqlens must be a CUDA tensor"):
        gdn_paged_prefill_slots(cu, page_size=128, **tables)
    with pytest.raises(ValueError, match="cu_seqlens must be on cuda"):
        chunk_gated_delta_rule_paged(
            q.cuda(),
            k.cuda(),
            v.cuda(),
            cu_seqlens=cu,
            state_pool=pool,
            page_size=128,
            **tables,
        )


# ── decode ────────────────────────────────────────────────────────────────────


def _decode_inputs(batch, heads, dtype=torch.bfloat16):
    num_q_heads, num_k_heads, num_v_heads = heads
    q = torch.randn(batch, 1, num_q_heads, HEAD_SIZE, dtype=dtype, device="cuda")
    k = torch.randn(batch, 1, num_k_heads, HEAD_SIZE, dtype=dtype, device="cuda")
    v = torch.randn(batch, 1, num_v_heads, HEAD_SIZE, dtype=dtype, device="cuda")
    A_log = torch.rand(num_v_heads, device="cuda") - 2.0
    a = torch.randn(batch, 1, num_v_heads, dtype=dtype, device="cuda")
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device="cuda")
    b = torch.randn(batch, 1, num_v_heads, dtype=dtype, device="cuda")
    return q, k, v, A_log, a, dt_bias, b


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads", [(4, 4, 8), (8, 8, 8)])
def test_paged_decode_reads_previous_page_on_a_new_page(state_dtype, heads):
    """A token that opens a new page reads the previous (possibly shared) page
    and writes the new one; other tokens update their last page in place."""
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 9:
        pytest.skip("GDN decode needs SM90+")
    torch.manual_seed(6)
    page_size = 64
    # last_page_len per sequence: new page, mid page, full page, new page.
    pages = [[3, 9], [5], [7, 0, 2], [1, 4]]
    last = [1, 17, page_size, 1]
    num_v_heads = heads[2]
    pool = torch.randn(12, num_v_heads, HEAD_SIZE, HEAD_SIZE, device="cuda").to(
        state_dtype
    )
    before = pool.clone()
    q, k, v, A_log, a, dt_bias, b = _decode_inputs(len(pages), heads)

    read = [p[-2] if n == 1 else p[-1] for p, n in zip(pages, last, strict=True)]
    write = [p[-1] for p in pages]
    expected_state = before[read].clone()
    expected_out, _ = (
        gated_delta_rule_decode_pretranspose(
            q, k, v, expected_state, A_log, a, dt_bias, b, use_qk_l2norm=True
        )
        if state_dtype == torch.float32
        else gated_delta_rule_decode_pretranspose(
            q,
            k,
            v,
            None,
            A_log,
            a,
            dt_bias,
            b,
            use_qk_l2norm=True,
            initial_state=expected_state,
            initial_state_indices=torch.arange(
                len(pages), dtype=torch.int32, device="cuda"
            ),
        )
    )

    out = gated_delta_rule_decode_paged(
        q,
        k,
        v,
        A_log,
        a,
        dt_bias,
        b,
        use_qk_l2norm=True,
        state_pool=pool,
        page_indices=torch.tensor(
            [p for ps in pages for p in ps], dtype=torch.int32, device="cuda"
        ),
        page_indptr=_cu([len(p) for p in pages], "cuda"),
        last_page_len=torch.tensor(last, dtype=torch.int32, device="cuda"),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected_out.float(), atol=1e-2, rtol=1e-2)
    for i, row in enumerate(write):
        assert torch.equal(pool[row], expected_state[i]), f"seq {i} write page"
    for row in range(pool.size(0)):
        if row not in write:
            assert torch.equal(pool[row], before[row]), f"page {row} changed"


@pytest.mark.parametrize("strided", ["page_indices", "page_indptr", "last_page_len"])
def test_paged_decode_rejects_strided_page_tables(strided):
    q, k, v, A_log, a, dt_bias, b = _decode_inputs(2, (4, 4, 4))
    pool = torch.zeros(4, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    tables = _page_tables(
        {"page_indices": [0, 1, 2], "page_indptr": [0, 2, 3], "last_page_len": [1, 5]},
        strided,
    )
    with pytest.raises(ValueError, match=f"{strided} must be contiguous"):
        gated_delta_rule_decode_paged(
            q, k, v, A_log, a, dt_bias, b, state_pool=pool, **tables
        )


def test_paged_prefill_then_decode_end_to_end():
    """Prefill on pages, then decode across a page boundary, against one
    prefill of the whole stream with the same (raw) gates."""
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 9:
        pytest.skip("GDN decode needs SM90+")
    _skip_if_unsupported()
    torch.manual_seed(7)
    page_size, prompt, steps, H = 64, 126, 4, 4
    total = prompt + steps
    dtype = torch.bfloat16
    q, k, v = gen_qkv([total], H, H, H, HEAD_SIZE, dtype)
    q, v = q.cuda(), v.cuda()
    k = F.normalize(k.cuda().float(), dim=-1).to(dtype)
    A_log = torch.rand(H, device="cuda") - 2.0
    dt_bias = torch.randn(H, device="cuda")
    a = torch.randn(total, H, dtype=dtype, device="cuda")
    b = torch.randn(total, H, dtype=dtype, device="cuda")
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    gate_kwargs = dict(
        use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, beta_is_logit=True
    )

    pool = torch.zeros(8, H, HEAD_SIZE, HEAD_SIZE, device="cuda")
    pages = [6, 1, 3]
    chunk_gated_delta_rule_paged(
        q[:prompt],
        k[:prompt],
        v[:prompt],
        a[:prompt],
        b[:prompt],
        scale,
        cu_seqlens=_cu([prompt], "cuda"),
        state_pool=pool,
        page_indices=torch.tensor(pages[:2], dtype=torch.int32, device="cuda"),
        page_indptr=_cu([2], "cuda"),
        page_size=page_size,
        last_page_len=torch.tensor(
            [prompt - page_size], dtype=torch.int32, device="cuda"
        ),
        **gate_kwargs,
    )
    outs = []
    for t in range(prompt, total):
        n_pages = math.ceil((t + 1) / page_size)
        outs.append(
            gated_delta_rule_decode_paged(
                q[t].view(1, 1, H, HEAD_SIZE),
                k[t].view(1, 1, H, HEAD_SIZE),
                v[t].view(1, 1, H, HEAD_SIZE),
                A_log,
                a[t].view(1, 1, H),
                dt_bias,
                b[t].view(1, 1, H),
                scale,
                use_qk_l2norm=False,
                state_pool=pool,
                page_indices=torch.tensor(
                    pages[:n_pages], dtype=torch.int32, device="cuda"
                ),
                page_indptr=_cu([n_pages], "cuda"),
                last_page_len=torch.tensor(
                    [t + 1 - (n_pages - 1) * page_size],
                    dtype=torch.int32,
                    device="cuda",
                ),
            ).view(H, HEAD_SIZE)
        )
    torch.cuda.synchronize()

    ckpts = torch.zeros(2, H, HEAD_SIZE, HEAD_SIZE, device="cuda")
    ref_out, ref_final = chunk_gated_delta_rule(
        q,
        k,
        v,
        a,
        b,
        scale,
        None,
        True,
        _cu([total], "cuda"),
        output_state=torch.zeros(1, H, HEAD_SIZE, HEAD_SIZE, device="cuda"),
        state_checkpoints=ckpts,
        checkpoint_cu_starts=_cu([2], "cuda"),
        checkpoint_every_n_tokens=page_size,
        use_cp=False,
        **gate_kwargs,
    )
    # Strong decay keeps these outputs small: compare the relative error.
    got, ref = torch.stack(outs).float(), ref_out[prompt:].float()
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 2e-2, f"decode outputs: relative error {rel:.3e}"
    _assert_state(pool[6], ckpts[0], msg="page 0")
    _assert_state(pool[1], ckpts[1], msg="page 1 (closed by decode)")
    _assert_state(pool[3], ref_final[0], msg="page 2")


def _capture(fn):
    """Warm ``fn`` up on a side stream, then capture it in a CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(warmup=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(warmup=False)
    return graph


@pytest.mark.parametrize("backend", ["flashinfer", "cake_gdn"])
def test_paged_prefill_cuda_graph_reads_live_page_tables(backend):
    """A captured paged prefill derives its rows from the page table at replay
    (no host synchronization): the same graph serves different prefixes,
    including an unaligned one (64-aligned for the Cake CP route), different
    page ids, and different new lengths of the same total."""
    _skip_if_unsupported()
    if backend == "cake_gdn":
        _skip_unless_cake_cp()
    torch.manual_seed(9)
    random.seed(9)
    L = 128
    new_lens = [70, 130]
    streams = Streams(new_lens, 4, 4, 4, torch.bfloat16)
    q, k, v, g, beta = streams.slices([0, 0], new_lens)
    cu = _cu(new_lens, "cuda")
    pool = torch.randn(16, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    scratch = pool.clone()
    indices = torch.empty(5, dtype=torch.int32, device="cuda")
    indptr = torch.empty(3, dtype=torch.int32, device="cuda")
    last = torch.empty(2, dtype=torch.int32, device="cuda")
    guard, out = _guarded_output(sum(new_lens), 4, q.dtype)
    # (page ids, page counts, last_page_len, new lengths): prefixes 0 / 256,
    # then 100 / 0 (128 / 0 for the Cake CP route), then new lengths 130 / 70
    # after prefixes 64 / 128.
    steps = [
        ([4, 7, 1, 12, 3], [1, 4], [70, 2], new_lens),
        (
            [6, 9, 2, 11, 15],
            [2, 2],
            [42 if backend == "flashinfer" else 70, 2],
            new_lens,
        ),
        ([5, 8, 0, 10, 13], [2, 2], [66, 70], [130, 70]),
    ]

    def call(state_pool, output):
        return chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(HEAD_SIZE),
            cu_seqlens=cu,
            state_pool=state_pool,
            page_indices=indices,
            page_indptr=indptr,
            page_size=L,
            last_page_len=last,
            output=output,
            backend=backend,
        )

    def set_tables(step):
        page_ids, counts, last_lens, lens = step
        indices.copy_(torch.tensor(page_ids, dtype=torch.int32))
        indptr.copy_(_cu(counts, "cpu"))
        last.copy_(torch.tensor(last_lens, dtype=torch.int32))
        cu.copy_(_cu(lens, "cpu"))

    set_tables(steps[0])
    graph = _capture(lambda warmup: call(scratch if warmup else pool, out))
    expected_out = torch.empty_like(out)
    for i, step in enumerate(steps):
        set_tables(step)
        expected_pool = pool.clone()
        call(expected_pool, expected_out)
        # A row the replay does not store stays NaN and fails the comparison.
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_output_guard(guard)
        # A stale page table reads another page's (unrelated) state.
        torch.testing.assert_close(
            out, expected_out, atol=1e-3, rtol=1e-3, msg=f"step {i}: output"
        )
        torch.testing.assert_close(
            pool, expected_pool, atol=1e-3, rtol=1e-3, msg=f"step {i}: pool"
        )


def test_paged_prefill_cake_cp_cuda_graph_interleaved_plans():
    """Two captured Cake CP calls with different plans (batch shapes, heads and
    page sizes) replay in turn, A, B, A, each matching an eager call of its
    own inputs: neither graph depends on the other's workspaces."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    torch.manual_seed(12)
    random.seed(12)

    def setup(new_lens, heads, page_size, page_ids, counts, last_lens):
        streams = Streams(new_lens, *heads, torch.bfloat16)
        q, k, v, g, beta = streams.slices([0] * len(new_lens), new_lens)
        cu = _cu(new_lens, "cuda")
        pool = torch.randn(
            16, streams.num_sab_heads, HEAD_SIZE, HEAD_SIZE, device="cuda"
        )
        indices = torch.tensor(page_ids, dtype=torch.int32, device="cuda")
        indptr = _cu(counts, "cuda")
        last = torch.tensor(last_lens, dtype=torch.int32, device="cuda")
        guard, out = _guarded_output(sum(new_lens), streams.num_sab_heads, q.dtype)

        def call(state_pool, output):
            return chunk_gated_delta_rule_paged(
                q,
                k,
                v,
                g,
                beta,
                1.0 / math.sqrt(HEAD_SIZE),
                cu_seqlens=cu,
                state_pool=state_pool,
                page_indices=indices,
                page_indptr=indptr,
                page_size=page_size,
                last_page_len=last,
                output=output,
                backend="cake_gdn",
            )

        scratch = pool.clone()
        graph = _capture(lambda warmup: call(scratch if warmup else pool, out))
        return graph, call, pool, guard, out

    # A: two sequences after prefixes 128 / 0 on 128-token pages.
    a = setup([70, 130], (4, 4, 4), 128, [4, 7, 1, 12], [2, 2], [70, 2])
    # B: one sequence after a 256-token prefix on 64-token pages.
    b = setup([256], (2, 2, 4), 64, [3, 9, 11, 2, 6, 14, 0, 8], [8], [64])
    for i, (graph, call, pool, guard, out) in enumerate((a, b, a)):
        expected_pool = pool.clone()
        expected_out = torch.empty_like(out)
        call(expected_pool, expected_out)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_output_guard(guard)
        torch.testing.assert_close(
            out, expected_out, atol=1e-3, rtol=1e-3, msg=f"replay {i}: output"
        )
        torch.testing.assert_close(
            pool, expected_pool, atol=1e-3, rtol=1e-3, msg=f"replay {i}: pool"
        )


def _count_cake_calls(monkeypatch):
    """Count the calls of the Cake entry of ``chunk_gated_delta_rule_paged``."""
    import flashinfer.gdn_paged as gdn_paged

    calls = []
    entry = gdn_paged._chunk_gated_delta_rule_paged_cake_cp

    def counting(*args, **kwargs):
        calls.append(None)
        return entry(*args, **kwargs)

    monkeypatch.setattr(gdn_paged, "_chunk_gated_delta_rule_paged_cake_cp", counting)
    return calls


def _auto_inputs(total, heads, dtype):
    num_q_heads, num_k_heads, num_v_heads = heads
    num_sab_heads = max(num_q_heads, num_v_heads)

    def uniform(num_heads):
        return (0.5 * torch.rand(total, num_heads, HEAD_SIZE, device="cuda") - 0.25).to(
            dtype
        )

    q, v = uniform(num_q_heads), uniform(num_v_heads)
    k = F.normalize(uniform(num_k_heads).float(), p=2.0, dim=-1).to(dtype)
    g = 0.9 + 0.1 * torch.rand(total, num_sab_heads, device="cuda")
    beta = torch.rand(total, num_sab_heads, device="cuda")
    return q, k, v, g, beta


def _run_auto_against(
    monkeypatch,
    backend,
    new_lens,
    heads,
    *,
    max_seqlen=None,
    aligned_prefixes=True,
    state_dtype=torch.float32,
    page_size=1024,
    seed=0,
):
    """``backend="auto"`` takes ``backend``'s route and returns its exact
    output and pool, from one starting pool."""
    random.seed(seed)
    torch.manual_seed(seed)
    calls = _count_cake_calls(monkeypatch)
    num_sab_heads = max(heads[0], heads[2])
    total = sum(new_lens)
    # 64-aligned prefixes: none, one page, one chunk.
    prefix_lens = [(0, page_size, CHUNK)[b % 3] for b in range(len(new_lens))]
    totals = [p + n for p, n in zip(prefix_lens, new_lens, strict=True)]
    q, k, v, g, beta = _auto_inputs(total, heads, torch.bfloat16)
    batch = PagedBatch(totals, page_size, num_sab_heads, state_dtype)
    batch.pool.copy_(0.1 * torch.randn_like(batch.pool))
    indices, indptr, last = batch.tables()
    results = {}
    for name in (backend, "auto"):
        pool = batch.pool.clone()
        calls.clear()
        out = chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(HEAD_SIZE),
            cu_seqlens=_cu(new_lens, "cuda"),
            state_pool=pool,
            page_indices=indices,
            page_indptr=indptr,
            page_size=page_size,
            last_page_len=last,
            backend=name,
            max_seqlen=max_seqlen,
            aligned_prefixes=aligned_prefixes,
        )
        torch.cuda.synchronize()
        results[name] = (out, pool, len(calls))
    assert results["auto"][2] == (1 if backend == "cake_gdn" else 0), (
        f'backend="auto" did not take the {backend} route'
    )
    assert torch.equal(results["auto"][0], results[backend][0]), "output"
    assert torch.equal(results["auto"][1], results[backend][1]), "pool"


@pytest.mark.parametrize(
    ("new_lens", "heads", "max_seqlen", "aligned_prefixes", "backend"),
    [
        # 2 x 4 states and a longest sequence of about 4128 tokens: in the band.
        ([4096, 4160], (2, 2, 4), None, True, "cake_gdn"),
        # The same batch without the caller's aligned-prefix promise.
        ([4096, 4160], (2, 2, 4), None, False, "flashinfer"),
        # A hint below ceil(total / num_seqs), which the Cake route rejects.
        ([4096, 4160], (2, 2, 4), 4096, True, "flashinfer"),
        # The hint lifts a balanced batch of 1024-token sequences into the band.
        ([1024, 1024], (2, 2, 4), 2048, True, "cake_gdn"),
        # Any integer type is a hint, for "auto" and for "cake_gdn" alike.
        ([1024, 1024], (2, 2, 4), np.int64(2048), True, "cake_gdn"),
        # Without the hint the mean estimate (1024) keeps it out.
        ([1024, 1024], (2, 2, 4), None, True, "flashinfer"),
        # Long sequences, but 16 x 4 = 64 states run in parallel.
        ([4096] * 16, (2, 2, 4), None, True, "flashinfer"),
    ],
)
def test_paged_prefill_auto_backend_choice(
    monkeypatch, new_lens, heads, max_seqlen, aligned_prefixes, backend
):
    _skip_if_unsupported()
    if backend == "cake_gdn":
        _skip_unless_cake_cp()
    _run_auto_against(
        monkeypatch,
        backend,
        new_lens,
        heads,
        max_seqlen=max_seqlen,
        aligned_prefixes=aligned_prefixes,
        seed=11,
    )


@pytest.mark.parametrize("case", ["bf16_pool", "no_cake_programs"])
def test_paged_prefill_auto_falls_back_when_cake_rejects(monkeypatch, case):
    """In the band, a call the Cake route rejects runs "flashinfer" instead of
    raising."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    state_dtype = torch.float32
    if case == "bf16_pool":
        state_dtype = torch.bfloat16
    else:
        import flashinfer.gdn_kernels.blackwell as blackwell

        monkeypatch.setattr(
            blackwell, "_chunk_gated_delta_rule_gdn_cp_paged_sm100", None
        )
    _run_auto_against(
        monkeypatch,
        "flashinfer",
        [4096, 4160],
        (2, 2, 4),
        state_dtype=state_dtype,
        seed=12,
    )


def test_paged_prefill_auto_cuda_graph(monkeypatch):
    """A captured backend="auto" call in the band runs the Cake route and
    reads the live page table at replay, matching an eager call."""
    _skip_if_unsupported()
    _skip_unless_cake_cp()
    calls = _count_cake_calls(monkeypatch)
    torch.manual_seed(13)
    random.seed(13)
    L = 1024
    new_lens = [4096, 4160]
    heads = (2, 2, 4)
    q, k, v, g, beta = _auto_inputs(sum(new_lens), heads, torch.bfloat16)
    cu = _cu(new_lens, "cuda")
    pool = 0.1 * torch.randn(24, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    scratch = pool.clone()
    indices = torch.zeros(10, dtype=torch.int32, device="cuda")
    indptr = torch.empty(3, dtype=torch.int32, device="cuda")
    last = torch.empty(2, dtype=torch.int32, device="cuda")
    guard, out = _guarded_output(sum(new_lens), 4, q.dtype)
    # (page ids, page counts, last_page_len): prefixes 0 / 0, then 1024 / 64.
    steps = [
        ([4, 7, 1, 12, 3, 20, 9, 15, 6, 0], [4, 5], [L, 64]),
        ([5, 8, 2, 11, 13, 21, 10, 16, 17, 18], [5, 5], [L, 128]),
    ]

    def call(state_pool, output):
        return chunk_gated_delta_rule_paged(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(HEAD_SIZE),
            cu_seqlens=cu,
            state_pool=state_pool,
            page_indices=indices,
            page_indptr=indptr,
            page_size=L,
            last_page_len=last,
            output=output,
            backend="auto",
            aligned_prefixes=True,
        )

    def set_tables(step):
        page_ids, counts, last_lens = step
        indices.copy_(torch.tensor(page_ids, dtype=torch.int32))
        indptr.copy_(_cu(counts, "cpu"))
        last.copy_(torch.tensor(last_lens, dtype=torch.int32))

    set_tables(steps[0])
    graph = _capture(lambda warmup: call(scratch if warmup else pool, out))
    assert len(calls) == 2, 'backend="auto" did not capture the Cake route'
    expected_out = torch.empty_like(out)
    for i, step in enumerate(steps):
        set_tables(step)
        expected_pool = pool.clone()
        call(expected_pool, expected_out)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_output_guard(guard)
        torch.testing.assert_close(
            out, expected_out, atol=1e-3, rtol=1e-3, msg=f"step {i}: output"
        )
        torch.testing.assert_close(
            pool, expected_pool, atol=1e-3, rtol=1e-3, msg=f"step {i}: pool"
        )


def test_paged_decode_cuda_graph_reads_live_page_tables():
    """A captured paged decode reads the page table at replay: the same graph
    serves a token that opens a new page and one that continues a page."""
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 9:
        pytest.skip("GDN decode needs SM90+")
    torch.manual_seed(10)
    heads = (4, 4, 8)
    pool = torch.randn(12, heads[2], HEAD_SIZE, HEAD_SIZE, device="cuda")
    scratch = pool.clone()
    q, k, v, A_log, a, dt_bias, b = _decode_inputs(2, heads)
    indices = torch.empty(4, dtype=torch.int32, device="cuda")
    indptr = _cu([2, 2], "cuda")
    last = torch.empty(2, dtype=torch.int32, device="cuda")
    out = torch.empty(2, 1, heads[2], HEAD_SIZE, dtype=q.dtype, device="cuda")
    steps = [([3, 9, 5, 0], [1, 1]), ([3, 9, 6, 2], [2, 1])]

    def call(state_pool):
        return gated_delta_rule_decode_paged(
            q,
            k,
            v,
            A_log,
            a,
            dt_bias,
            b,
            output=out,
            state_pool=state_pool,
            page_indices=indices,
            page_indptr=indptr,
            last_page_len=last,
        )

    indices.copy_(torch.tensor(steps[0][0], dtype=torch.int32))
    last.copy_(torch.tensor(steps[0][1], dtype=torch.int32))
    graph = _capture(lambda warmup: call(scratch if warmup else pool))
    for i, (page_ids, last_lens) in enumerate(steps):
        indices.copy_(torch.tensor(page_ids, dtype=torch.int32))
        last.copy_(torch.tensor(last_lens, dtype=torch.int32))
        expected_pool = pool.clone()
        expected_out = call(expected_pool).clone()
        graph.replay()
        torch.cuda.synchronize()
        # A stale page table reads another page's (unrelated) state.
        torch.testing.assert_close(out, expected_out, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            pool, expected_pool, atol=1e-3, rtol=1e-3, msg=f"step {i}: pool"
        )


# ── slot helpers ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_paged_prefill_slots_rows(index_dtype):
    """The prefill rows, case by case (page_size 128)."""
    L = 128
    # (prefix, new tokens, pages)
    cases = [
        (0, 300, [7, 2, 9]),  # no prefix, ends inside page 2
        (100, 200, [4, 11, 0]),  # mid-page prefix crossing two boundaries
        (256, 128, [5, 6, 8]),  # page-aligned prefix, ends on a boundary
        (129, 0, [1, 3]),  # no new token
        (129, 10, [12, 13]),  # prefix one token into page 1
        (0, 5, [10]),  # one partial page
    ]
    pages = [c[2] for c in cases]
    totals = [p + n for p, n, _ in cases]
    slots = gdn_paged_prefill_slots(
        _cu([n for _, n, _ in cases], "cuda").to(index_dtype),
        torch.tensor(sum(pages, []), dtype=index_dtype, device="cuda"),
        _cu([len(ps) for ps in pages], "cuda").to(index_dtype),
        L,
        torch.tensor(
            [t - (len(ps) - 1) * L for t, ps in zip(totals, pages, strict=True)],
            dtype=index_dtype,
            device="cuda",
        ),
    )
    expected = {
        "state_indices": [-1, 4, 6, 3, 13, -1],
        "output_state_indices": [9, 0, -1, -1, 13, 10],
        "checkpoint_cu_starts": [0, 3, 8, 10, 12, 13],
        "prefix_lens": [0, 100, 256, 129, 129, 0],
    }
    for name, values in expected.items():
        tensor = getattr(slots, name)
        assert tensor.dtype == torch.int32, name
        assert tensor.tolist() == values, name


def test_paged_decode_slots_rows():
    pages = [[3, 9], [5], [7, 0, 2], [1, 4]]
    slots = gdn_paged_decode_slots(
        torch.tensor(sum(pages, []), dtype=torch.int64, device="cuda"),
        _cu([len(p) for p in pages], "cuda"),
        torch.tensor([1, 17, 64, 1], dtype=torch.int32, device="cuda"),
    )
    assert slots.initial_state_indices.tolist() == [3, 5, 2, 1]
    assert slots.output_state_indices.tolist() == [9, 5, 2, 4]


def test_paged_prefill_slots_reused_across_layers():
    """Rows computed once serve every layer: each layer's call with the shared
    slots matches the call that derives them itself, bit for bit."""
    _skip_if_unsupported()
    torch.manual_seed(11)
    random.seed(11)
    L = 128
    prefix, new = [0, 70, 256], [100, 130, 64]
    totals = [p + n for p, n in zip(prefix, new, strict=True)]
    batch = PagedBatch(totals, L, 4, torch.float32)
    indices, indptr, last = batch.tables()
    cu = _cu(new, "cuda")
    slots = gdn_paged_prefill_slots(cu, indices, indptr, L, last)
    for layer in range(2):
        streams = Streams(totals, 4, 4, 4, torch.bfloat16)
        q, k, v, g, beta = streams.slices(prefix, totals)
        pools = [batch.pool.clone() + layer, batch.pool.clone() + layer]
        outs = [
            chunk_gated_delta_rule_paged(
                q,
                k,
                v,
                g,
                beta,
                cu_seqlens=cu,
                state_pool=pool,
                page_indices=indices,
                page_indptr=indptr,
                page_size=L,
                last_page_len=last,
                slots=shared,
            )
            for pool, shared in zip(pools, (slots, None), strict=True)
        ]
        torch.cuda.synchronize()
        assert torch.equal(outs[0], outs[1]), f"layer {layer} output"
        assert torch.equal(pools[0], pools[1]), f"layer {layer} pool"


def test_paged_decode_slots_reused_across_layers():
    """Decode rows computed once serve every layer, bit for bit."""
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 9:
        pytest.skip("GDN decode needs SM90+")
    torch.manual_seed(12)
    pages = [[3, 9], [5], [7, 0, 2], [1, 4]]
    indices = torch.tensor(sum(pages, []), dtype=torch.int32, device="cuda")
    indptr = _cu([len(p) for p in pages], "cuda")
    last = torch.tensor([1, 17, 64, 1], dtype=torch.int32, device="cuda")
    slots = gdn_paged_decode_slots(indices, indptr, last)
    for layer in range(2):
        q, k, v, A_log, a, dt_bias, b = _decode_inputs(len(pages), (4, 4, 8))
        pool = torch.randn(12, 8, HEAD_SIZE, HEAD_SIZE, device="cuda").to(
            torch.bfloat16
        )
        pools = [pool.clone(), pool.clone()]
        outs = [
            gated_delta_rule_decode_paged(
                q,
                k,
                v,
                A_log,
                a,
                dt_bias,
                b,
                use_qk_l2norm=True,
                state_pool=state_pool,
                page_indices=indices,
                page_indptr=indptr,
                last_page_len=last,
                slots=shared,
            )
            for state_pool, shared in zip(pools, (slots, None), strict=True)
        ]
        torch.cuda.synchronize()
        assert torch.equal(outs[0], outs[1]), f"layer {layer} output"
        assert torch.equal(pools[0], pools[1]), f"layer {layer} pool"


@pytest.mark.parametrize("bad", ["int64", "length", "strided", "cpu"])
@pytest.mark.parametrize("entry", ["prefill", "decode"])
def test_paged_rejects_bad_slots(entry, bad):
    def corrupt(x: torch.Tensor) -> torch.Tensor:
        if bad == "int64":
            return x.long()
        if bad == "length":
            return torch.cat([x, x[:1]])
        if bad == "strided":
            return _strided(x)
        return x.cpu()

    rows = torch.zeros(2, dtype=torch.int32, device="cuda")
    pool = torch.zeros(4, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    match = r"slots\.\w+ must be a contiguous int32 tensor of shape \(2,\)"
    if entry == "prefill":
        q, k, v = gen_qkv([16], 4, 4, 4, HEAD_SIZE, torch.bfloat16)
        tables = _page_tables(
            {
                "cu_seqlens": [0, 8, 16],
                "page_indices": [0, 1, 2],
                "page_indptr": [0, 2, 3],
                "last_page_len": [8, 8],
            }
        )
        slots = GDNPagedPrefillSlots(rows, rows, rows, rows)
        slots = slots._replace(prefix_lens=corrupt(rows))
        with pytest.raises(ValueError, match=match):
            chunk_gated_delta_rule_paged(
                q.cuda(),
                k.cuda(),
                v.cuda(),
                state_pool=pool,
                page_size=128,
                slots=slots,
                **tables,
            )
    else:
        q, k, v, A_log, a, dt_bias, b = _decode_inputs(2, (4, 4, 4))
        tables = _page_tables(
            {
                "page_indices": [0, 1, 2],
                "page_indptr": [0, 2, 3],
                "last_page_len": [1, 5],
            }
        )
        slots = GDNPagedDecodeSlots(rows, corrupt(rows))
        with pytest.raises(ValueError, match=match):
            gated_delta_rule_decode_paged(
                q, k, v, A_log, a, dt_bias, b, state_pool=pool, slots=slots, **tables
            )


# ── SM120 state rows as element pairs ─────────────────────────────────────────
#
# On SM120 a slot call (``output_state_indices``, ``checkpoint_state_indices``
# or ``prefix_lens``) reads the initial state and writes the final state and
# the pool-row checkpoints of dense [H, V, K] rows as 2-element vector accesses
# when every row is aligned to two elements or more (``checkpoint_row_align``
# of the tensor, part of the compile key), and element by element otherwise.
# Both give the same bits.

_FP8_STATE_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
# Relative error over a tensor of fp8 states (3 and 2 mantissa bits) and of
# the outputs that read them.
_FP8_REL_TOL = {torch.float8_e4m3fn: 0.1, torch.float8_e5m2: 0.2}


def _skip_unless_sm120():
    _skip_if_unsupported()
    if get_compute_capability(torch.device("cuda"))[0] != 12:
        pytest.skip(
            "covers the SM120 choice of element-pair or element-wise state rows"
        )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def _guard_value(dtype):
    """``GUARD`` where the dtype holds it (float8_e4m3fn tops out at 448)."""
    return 256.0 if dtype.itemsize == 1 else GUARD


class AlignedPool:
    """``rows`` state rows ``H * V * K + row_pad`` elements apart from a
    16-byte aligned base: ``row_pad`` sets the alignment of every row
    (``checkpoint_row_align``), 1, 2, 4, ... elements, or 16 bytes with
    ``row_pad = 0``. The row padding and at least one row of elements before
    and after the rows hold ``guard``."""

    def __init__(self, rows, num_heads, dtype, row_pad, guard):
        self.rows = rows
        self.row = num_heads * HEAD_SIZE * HEAD_SIZE
        self.stride = self.row + row_pad
        self.guard = guard
        per_16_bytes = 16 // dtype.itemsize
        self.lead = -(-self.stride // per_16_bytes) * per_16_bytes
        self.end = self.lead + rows * self.stride
        self.storage = torch.full((self.end + self.stride,), guard, device="cuda").to(
            dtype
        )
        self.pool = self.storage.as_strided(
            (rows, num_heads, HEAD_SIZE, HEAD_SIZE),
            (self.stride, HEAD_SIZE * HEAD_SIZE, HEAD_SIZE, 1),
            self.lead,
        )

    def assert_guards(self):
        rows = self.storage[self.lead : self.end].view(self.rows, self.stride)
        for name, part in (
            ("before row 0", self.storage[: self.lead]),
            ("the row padding", rows[:, self.row :]),
            ("past the last row", self.storage[self.end :]),
        ):
            full = torch.full(part.shape, self.guard, device="cuda").to(part.dtype)
            assert _same_bits(part, full), f"wrote {name}"


def _check_against_oracle(state_dtype, actual, expected, *, msg, output=False):
    """Outputs at 1e-2 and states as ``_assert_state``, like the tests above.
    With an fp8 state both are checked by their relative error over the
    tensor: one rounding step of a 2- or 3-bit mantissa is larger than an
    element-wise tolerance that still tells a wrong row from the right one."""
    if state_dtype in _FP8_REL_TOL:
        a, e = actual.float(), expected.float()
        rel = ((a - e).norm() / e.norm()).item()
        assert rel < _FP8_REL_TOL[state_dtype], f"{msg}: relative error {rel:.3e}"
    elif output:
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m: f"{msg}: {m}",
        )
    else:
        _assert_state(actual, expected, msg=msg)


# Sequence 0 starts from zero (no prefix, read row -1), sequence 1 reads and
# writes row 3 (a partial page updated in place), sequence 2 skips its final
# state, sequence 3 has no new token (its write row 10 keeps its contents)
# and sequence 4 ends on a 64-token boundary.
_PAIR_PREFIX = [0, 70, 128, 50, 37]
_PAIR_NEW = [130, 64, 200, 0, 91]
_PAIR_READ = [-1, 3, 5, 8, 2]
_PAIR_WRITE = [1, 3, -1, 10, 6]
# 64-token checkpoints at absolute positions, 2 + 1 + 3 + 0 + 2 entries: the
# first block of sequences 0, 2 and 4 stores one (sequence 1's is skipped),
# the block runs of sequences 0 and 2 store the next ones, and the last block
# of sequence 4 stores the one at token 128.
_PAIR_CKPT = [0, 4, -1, 7, 9, 11, 12, 13]


@pytest.mark.parametrize(
    "state_dtype",
    [torch.float32, torch.bfloat16, torch.float16, *_FP8_STATE_DTYPES],
)
def test_sm120_slot_state_rows_at_every_alignment(state_dtype):
    """Slot calls on pools whose rows are aligned to every power of two from
    one element to 16 bytes, in one process: element pairs and element-wise
    access give the same bits for the initial state, the final state and the
    pool-row checkpoints, also with initial, final and checkpoint pools of
    different alignments. The last call returns to the first call's pools
    (each alignment compiles its own kernel) and reproduces its result.
    Then ``output_state_indices`` alone, at 16 bytes and at one element."""
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
        checkpoint_row_align,
    )

    _skip_unless_sm120()
    random.seed(14)
    torch.manual_seed(14)
    prefix, new = _PAIR_PREFIX, _PAIR_NEW
    totals = [p + n for p, n in zip(prefix, new, strict=True)]
    streams = Streams(totals, 4, 4, 4, torch.bfloat16)
    _, prefix_final, _ = _oracle(streams, prefix, state_dtype)
    ref_out, ref_final, ref_ckpts = _oracle(streams, totals, state_dtype)
    q, k, v, g, beta = streams.slices(prefix, totals)
    counts = [(p + n) // CHUNK - p // CHUNK for p, n in zip(prefix, new, strict=True)]
    # Rows written: final states (row -> sequence) and checkpoints (row ->
    # (sequence, absolute position)).
    finals = {row: b for b, row in enumerate(_PAIR_WRITE) if row >= 0 and new[b]}
    entries = iter(_PAIR_CKPT)
    ckpts = {}
    for b, p in enumerate(prefix):
        for j in range(counts[b]):
            row = next(entries)
            if row >= 0:
                ckpts[row] = (b, (p // CHUNK + 1 + j) * CHUNK)

    num_rows = 16
    content = torch.randn(num_rows, 4, HEAD_SIZE, HEAD_SIZE, device="cuda").to(
        state_dtype
    )
    for b, row in enumerate(_PAIR_READ):
        if row >= 0:
            content[row] = prefix_final[b]
    guard = _guard_value(state_dtype)
    max_align = 16 // state_dtype.itemsize

    def call(aligns, one_pool, checkpoints=True):
        """Initial, final and checkpoint pools with rows aligned to
        ``aligns`` (one pool for all three: ``one_pool``)."""

        def make_pool(align):
            pool = AlignedPool(
                num_rows, 4, state_dtype, 0 if align == max_align else align, guard
            )
            pool.pool.copy_(content)
            assert checkpoint_row_align(pool.pool) == align
            return pool

        if one_pool:
            init = out_pool = ckpt_pool = make_pool(aligns[0])
        else:
            init, out_pool, ckpt_pool = (make_pool(a) for a in aligns)
        kwargs = {}
        if checkpoints:
            kwargs = dict(
                state_checkpoints=ckpt_pool.pool,
                checkpoint_cu_starts=_cu(counts, "cuda"),
                checkpoint_every_n_tokens=CHUNK,
                checkpoint_state_indices=_idx(_PAIR_CKPT, 32),
                prefix_lens=_idx(prefix, 32),
            )
        guarded_out, out = _guarded_output(sum(new), 4, torch.bfloat16)
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(HEAD_SIZE),
            init.pool,
            True,
            _cu(new, "cuda"),
            output_state=out_pool.pool,
            state_indices=_idx(_PAIR_READ, 32),
            output_state_indices=_idx(_PAIR_WRITE, 32),
            output=out,
            **kwargs,
        )
        torch.cuda.synchronize()
        label = f"{state_dtype} rows aligned to {aligns}" + (
            " in one pool" if one_pool else ""
        )
        _assert_output_guard(guarded_out)
        written = {}
        for pool, rows in (
            (init, ()),
            (out_pool, finals),
            (ckpt_pool, ckpts if checkpoints else ()),
        ):
            written.setdefault(id(pool), (pool, set()))[1].update(rows)
        for pool, rows in written.values():
            pool.assert_guards()
            for row in range(num_rows):
                if row not in rows:
                    assert _same_bits(pool.pool[row], content[row]), (
                        f"{label}: row {row} changed"
                    )
        # Every row holds its content, a final state or a checkpoint.
        merged = content.clone()
        for row in finals:
            merged[row] = out_pool.pool[row]
        if checkpoints:
            for row in ckpts:
                merged[row] = ckpt_pool.pool[row]
        return out, merged, label

    # Row alignments in elements: 16 bytes, then 1, 2, 4, ... below it.
    aligns = [max_align] + [1 << i for i in range(max_align.bit_length() - 1)]
    n = len(aligns)
    layouts = (
        [((max_align,) * 3, True), ((1,) * 3, True)]
        # Each alignment once as the initial, final and checkpoint pool's.
        + [(tuple(aligns[(i + j) % n] for j in range(3)), False) for i in range(n)]
        + [((max_align,) * 3, True)]
    )
    first = None
    for layout, one_pool in layouts:
        out, merged, label = call(layout, one_pool)
        if first is None:
            first = out, merged
            continue
        assert _same_bits(out, first[0]), f"{label}: output"
        assert _same_bits(merged, first[1]), f"{label}: states"

    ref_rows = torch.cat(
        [
            torch.arange(s + p, s + t)
            for s, p, t in zip(streams.starts[:-1], prefix, totals, strict=True)
        ]
    ).cuda()
    out, merged = first
    _check_against_oracle(
        state_dtype, out, ref_out[ref_rows], msg="output", output=True
    )
    for row, b in finals.items():
        _check_against_oracle(
            state_dtype, merged[row], ref_final[b], msg=f"seq {b} final"
        )
    for row, (b, position) in ckpts.items():
        _check_against_oracle(
            state_dtype,
            merged[row],
            ref_ckpts[b][position // CHUNK - 1],
            msg=f"seq {b} @{position}",
        )

    # output_state_indices alone (no checkpoints, no prefix_lens).
    (out, merged, _), (out_1, merged_1, _) = (
        call((align,) * 3, True, checkpoints=False) for align in (max_align, 1)
    )
    assert _same_bits(out_1, out), "output_state_indices alone: output"
    assert _same_bits(merged_1, merged), "output_state_indices alone: states"
    _check_against_oracle(
        state_dtype, out, ref_out[ref_rows], msg="no checkpoints: output", output=True
    )
    for row, b in finals.items():
        _check_against_oracle(
            state_dtype, merged[row], ref_final[b], msg=f"no checkpoints: seq {b}"
        )


@pytest.mark.parametrize(
    "state_dtype", [torch.bfloat16, torch.float16, *_FP8_STATE_DTYPES]
)
def test_sm120_packed_slot_states_match_pool_rows(state_dtype):
    """``prefix_lens`` with packed per-sequence initial and final states of a
    2- or 1-byte dtype (no ``state_indices``), whose rows SM120 reads and
    writes as element pairs: against the oracle, and bit for bit against the
    same call on pool rows aligned to one element (element by element).
    Sequence 2 has no prefix; sequence 3 has no new token and keeps its final
    state."""
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import (
        checkpoint_row_align,
    )

    _skip_unless_sm120()
    random.seed(15)
    torch.manual_seed(15)
    prefix = [100, 64, 0, 37]
    new = [200, 192, 130, 0]
    totals = [p + n for p, n in zip(prefix, new, strict=True)]
    num_seqs = len(prefix)
    streams = Streams(totals, 4, 4, 4, torch.bfloat16)
    _, prefix_final, _ = _oracle(streams, prefix, state_dtype)
    ref_out, ref_final, ref_ckpts = _oracle(streams, totals, state_dtype)
    q, k, v, g, beta = streams.slices(prefix, totals)
    N = 128
    counts = [(p + n) // N - p // N for p, n in zip(prefix, new, strict=True)]
    filler = torch.randn(num_seqs, 4, HEAD_SIZE, HEAD_SIZE, device="cuda").to(
        state_dtype
    )
    guard = _guard_value(state_dtype)
    pool_rows = [4, 0, 2, 5]

    def call(packed):
        rows = list(range(num_seqs)) if packed else pool_rows
        num_rows, row_pad = (num_seqs, 0) if packed else (6, 1)
        init = AlignedPool(num_rows, 4, state_dtype, row_pad, guard)
        final = AlignedPool(num_rows, 4, state_dtype, row_pad, guard)
        for b, row in enumerate(rows):
            init.pool[row] = prefix_final[b]
            final.pool[row] = filler[b]
        assert init.pool.is_contiguous() == packed
        assert checkpoint_row_align(init.pool) == (
            16 // state_dtype.itemsize if packed else 1
        )
        ckpts = torch.zeros(sum(counts), 4, HEAD_SIZE, HEAD_SIZE, device="cuda").to(
            state_dtype
        )
        guarded_out, out = _guarded_output(sum(new), 4, torch.bfloat16)
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            1.0 / math.sqrt(HEAD_SIZE),
            init.pool,
            True,
            _cu(new, "cuda"),
            output_state=final.pool,
            state_checkpoints=ckpts,
            checkpoint_cu_starts=_cu(counts, "cuda"),
            checkpoint_every_n_tokens=N,
            state_indices=None if packed else _idx(pool_rows, 32),
            prefix_lens=_idx(prefix, 32),
            output=out,
        )
        torch.cuda.synchronize()
        _assert_output_guard(guarded_out)
        init.assert_guards()
        final.assert_guards()
        for b, row in enumerate(rows):
            assert _same_bits(init.pool[row], prefix_final[b]), f"seq {b} init"
        return out, [final.pool[row].clone() for row in rows], ckpts

    (out, finals, ckpts), (out_1, finals_1, ckpts_1) = call(True), call(False)
    assert _same_bits(out_1, out), "output"
    for b in range(num_seqs):
        assert _same_bits(finals_1[b], finals[b]), f"seq {b} final"
    assert _same_bits(ckpts_1, ckpts), "checkpoints"

    ref_rows = torch.cat(
        [
            torch.arange(s + p, s + t)
            for s, p, t in zip(streams.starts[:-1], prefix, totals, strict=True)
        ]
    ).cuda()
    _check_against_oracle(
        state_dtype, out, ref_out[ref_rows], msg="output", output=True
    )
    for b in range(num_seqs):
        if new[b]:
            _check_against_oracle(
                state_dtype, finals[b], ref_final[b], msg=f"seq {b} final"
            )
        else:
            assert _same_bits(finals[b], filler[b]), f"seq {b} final changed"
    row = 0
    for b in range(num_seqs):
        for j in range(counts[b]):
            position = (prefix[b] // N + 1 + j) * N
            _check_against_oracle(
                state_dtype,
                ckpts[row],
                ref_ckpts[b][position // CHUNK - 1],
                msg=f"seq {b} @{position}",
            )
            row += 1
