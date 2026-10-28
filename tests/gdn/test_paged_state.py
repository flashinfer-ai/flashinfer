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
import random

import pytest
import torch
import torch.nn.functional as F

from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.gdn_paged import (
    chunk_gated_delta_rule_paged,
    gated_delta_rule_decode_paged,
)
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import get_compute_capability

from .conftest import gen_qkv

HEAD_SIZE = 128
CHUNK = 64


def _skip_if_unsupported():
    major, _ = get_compute_capability(torch.device("cuda"))
    if major not in (8, 9, 10, 12):
        pytest.skip("GDN prefill needs SM8x, SM90, SM100/SM103/SM107 or SM12x")
    if major == 10:
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip("SM100 GDN prefill requires CUDA 13+")


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

    def __init__(self, total_lens, page_len, num_sab_heads, state_dtype, spare=7):
        self.page_len = page_len
        self.total_lens = list(total_lens)
        self.num_pages = [max(1, math.ceil(n / page_len)) for n in total_lens]
        pool_pages = sum(self.num_pages) + spare
        ids = list(range(pool_pages))
        random.shuffle(ids)
        self.pages = []
        it = iter(ids)
        for n in self.num_pages:
            self.pages.append([next(it) for _ in range(n)])
        # Finite, page-distinct filler: a page read by mistake corrupts the
        # result, and a page written by mistake no longer equals its copy.
        self.pool = torch.randn(
            pool_pages, num_sab_heads, HEAD_SIZE, HEAD_SIZE, device="cuda"
        ).to(state_dtype) + torch.arange(pool_pages, device="cuda").view(
            -1, 1, 1, 1
        ).to(state_dtype)

    def tables(self, index_dtype=torch.int32):
        indices = torch.tensor(
            [p for pages in self.pages for p in pages], dtype=index_dtype, device="cuda"
        )
        indptr = _cu(self.num_pages, "cuda").to(index_dtype)
        last = torch.tensor(
            [
                n - (m - 1) * self.page_len
                for n, m in zip(self.total_lens, self.num_pages, strict=True)
            ],
            dtype=index_dtype,
            device="cuda",
        )
        return indices, indptr, last

    def page_of(self, b: int, token: int) -> int:
        return self.pages[b][token // self.page_len]


def _assert_state(actual, expected, *, msg):
    tol = 1e-2 if actual.dtype == torch.float32 else 3e-2
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=tol,
        rtol=tol,
        msg=lambda m: f"{msg}: {m}",
    )


def _run_paged_prefill_case(
    prefix_lens,
    new_lens,
    page_len,
    heads,
    dtype,
    state_dtype,
    index_dtype=torch.int32,
):
    num_q_heads, num_k_heads, num_v_heads = heads
    totals = [p + n for p, n in zip(prefix_lens, new_lens, strict=True)]
    streams = Streams(totals, num_q_heads, num_k_heads, num_v_heads, dtype)
    batch = PagedBatch(totals, page_len, streams.num_sab_heads, state_dtype)
    num_seqs = len(totals)

    # Seed every prefix page with its true state; the call must only read the
    # page that holds the last prefix token.
    _, prefix_final, prefix_ckpts = _oracle(streams, prefix_lens, state_dtype)
    for b, p in enumerate(prefix_lens):
        for j in range(p // page_len):
            batch.pool[batch.pages[b][j]] = prefix_ckpts[b][
                (j + 1) * page_len // CHUNK - 1
            ]
        if p % page_len:
            batch.pool[batch.page_of(b, p - 1)] = prefix_final[b]
    pool_before = batch.pool.clone()

    q, k, v, g, beta = streams.slices(prefix_lens, totals)
    indices, indptr, last = batch.tables(index_dtype)
    out = chunk_gated_delta_rule_paged(
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
        page_len=page_len,
        last_page_len=last,
    )
    torch.cuda.synchronize()

    ref_out, ref_final, ref_ckpts = _oracle(streams, totals, state_dtype)
    ref_rows = torch.cat(
        [
            torch.arange(start + p, start + t)
            for start, p, t in zip(
                _cu(totals, "cpu").tolist(), prefix_lens, totals, strict=True
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
        for j in range(p // page_len, batch.num_pages[b]):
            page = batch.pages[b][j]
            written.add(page)
            end = min((j + 1) * page_len, t)
            expected = ref_final[b] if end == t else ref_ckpts[b][end // CHUNK - 1]
            _assert_state(batch.pool[page], expected, msg=f"seq {b} page {j}")
    for page in range(batch.pool.size(0)):
        if page not in written:
            assert torch.equal(batch.pool[page], pool_before[page]), (
                f"page {page} was not owned by any new token but changed"
            )


@pytest.mark.parametrize("page_len", [64, 128, 256])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [(4, 4, 4), (2, 2, 4), (4, 1, 1)])
def test_paged_prefill_ragged(page_len, state_dtype, dtype, heads):
    """Ragged batch mixing no prefix, page-aligned, chunk-aligned and unaligned
    prefixes with new spans that stop short of, on and past page boundaries."""
    _skip_if_unsupported()
    if dtype == torch.float16 and state_dtype == torch.bfloat16:
        pytest.skip("covered by the bfloat16 inputs")
    random.seed(page_len)
    torch.manual_seed(page_len)
    L = page_len
    prefix_lens = [0, 0, L, 2 * L, CHUNK, L + 2 * CHUNK, 37, L + 5, 3 * L - 1, L - 1]
    new_lens = [1, 3 * L + 17, L, 5, 2 * L, L + 7, L - 37, 2 * L + 63, 1, 1]
    _run_paged_prefill_case(prefix_lens, new_lens, L, heads, dtype, state_dtype)


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
            page_len=L,
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
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    done = [0, 0]
    for chunk in ([1024, 0], [16, 1008], [0, 42]):
        new_done = [d + c for d, c in zip(done, chunk, strict=True)]
        q, k, v, g, beta = streams.slices(done, new_done)
        pages = [math.ceil(n / L) if n else 1 for n in new_done]
        used = [batch.pages[b][: pages[b]] for b in range(2)]
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
            page_len=L,
            last_page_len=torch.tensor(
                [max(n - (m - 1) * L, 1) for n, m in zip(new_done, pages, strict=True)],
                dtype=torch.int32,
                device="cuda",
            ),
        )
        done = new_done
    torch.cuda.synchronize()
    _, ref_final, ref_ckpts = _oracle(streams, [1040, 1050])
    for b in range(2):
        _assert_state(
            batch.pool[batch.pages[b][0]],
            ref_ckpts[b][L // CHUNK - 1],
            msg=f"seq {b} state at token 1024",
        )
        _assert_state(batch.pool[batch.pages[b][1]], ref_final[b], msg=f"seq {b} end")


def test_slot_arguments_match_packed_api():
    """The low-level slot arguments against the packed API: -1 read rows start
    from zero, -1 write rows and -1 checkpoint rows are skipped."""
    _skip_if_unsupported()
    random.seed(4)
    torch.manual_seed(4)
    lens = [130, 64, 200]
    streams = Streams(lens, 4, 4, 4, torch.bfloat16)
    q, k, v, g, beta = streams.slices([0, 0, 0], lens)
    cu = _cu(lens, "cuda")
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    ref_out, ref_final, ref_ckpts = _oracle(streams, lens)

    pool = torch.randn(12, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    before = pool.clone()
    read = torch.tensor([-1, -1, -1], dtype=torch.int32, device="cuda")
    write = torch.tensor([9, -1, 2], dtype=torch.int32, device="cuda")
    # 64-token checkpoints: 2 + 1 + 3; route seq 2's second one to -1.
    ckpt_rows = torch.tensor([4, 6, 0, 1, -1, 10], dtype=torch.int32, device="cuda")
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
        checkpoint_cu_starts=torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda"),
        checkpoint_every_n_tokens=CHUNK,
        state_indices=read,
        output_state_indices=write,
        checkpoint_state_indices=ckpt_rows,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)
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


def test_one_selected_state_per_sequence():
    """One chunk-aligned intermediate state per sequence (or none) goes straight
    to its own row: every other 64-token checkpoint entry is -1, so nothing
    else is stored and no gather follows."""
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

    states = torch.randn(8, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
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


def test_prefix_lens_absolute_checkpoints():
    """``prefix_lens`` moves the checkpoint grid to absolute positions; an
    unaligned prefix is chunked from its 64-aligned absolute position."""
    _skip_if_unsupported()
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
    out, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        1.0 / math.sqrt(HEAD_SIZE),
        init,
        True,
        _cu(new, "cuda"),
        output_state=final,
        state_checkpoints=ckpts,
        checkpoint_cu_starts=_cu(counts, "cuda"),
        checkpoint_every_n_tokens=N,
        prefix_lens=torch.tensor(prefix, dtype=torch.int32, device="cuda"),
    )
    torch.cuda.synchronize()
    ref_rows = torch.cat(
        [
            torch.arange(s + p, s + t)
            for s, p, t in zip(_cu(totals, "cpu").tolist(), prefix, totals, strict=True)
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


def test_paged_prefill_rejects_bad_page_len():
    q, k, v = gen_qkv([8], 4, 4, 4, HEAD_SIZE, torch.bfloat16)
    pool = torch.zeros(2, 4, HEAD_SIZE, HEAD_SIZE, device="cuda")
    one = torch.ones(1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="multiple of 64"):
        chunk_gated_delta_rule_paged(
            q.cuda(),
            k.cuda(),
            v.cuda(),
            cu_seqlens=_cu([8], "cuda"),
            state_pool=pool,
            page_indices=one - 1,
            page_indptr=_cu([1], "cuda"),
            page_len=100,
            last_page_len=one * 8,
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
    page_len = 64
    # last_page_len per sequence: new page, mid page, full page, new page.
    pages = [[3, 9], [5], [7, 0, 2], [1, 4]]
    last = [1, 17, page_len, 1]
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


def test_paged_prefill_then_decode_end_to_end():
    """Prefill on pages, then decode across a page boundary, against one
    prefill of the whole stream with the same (raw) gates."""
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 9:
        pytest.skip("GDN decode needs SM90+")
    _skip_if_unsupported()
    torch.manual_seed(7)
    page_len, prompt, steps, H = 64, 126, 4, 4
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
        page_len=page_len,
        last_page_len=torch.tensor(
            [prompt - page_len], dtype=torch.int32, device="cuda"
        ),
        **gate_kwargs,
    )
    outs = []
    for t in range(prompt, total):
        n_pages = math.ceil((t + 1) / page_len)
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
                    [t + 1 - (n_pages - 1) * page_len], dtype=torch.int32, device="cuda"
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
        checkpoint_every_n_tokens=page_len,
        use_cp=False,
        **gate_kwargs,
    )
    torch.testing.assert_close(
        torch.stack(outs).float(), ref_out[prompt:].float(), atol=2e-2, rtol=2e-2
    )
    _assert_state(pool[6], ckpts[0], msg="page 0")
    _assert_state(pool[1], ckpts[1], msg="page 1 (closed by decode)")
    _assert_state(pool[3], ref_final[0], msg="page 2")
