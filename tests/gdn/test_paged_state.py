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


def test_paged_prefill_cuda_graph_reads_live_page_tables():
    """A captured paged prefill derives its rows from the page table at replay
    (no host synchronization): the same graph serves different prefixes,
    including an unaligned one, and different page ids."""
    _skip_if_unsupported()
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
    # (page ids, page counts, last_page_len): prefixes 0 / 256, then 100 / 0.
    steps = [
        ([4, 7, 1, 12, 3], [1, 4], [70, 2]),
        ([6, 9, 2, 11, 15], [2, 2], [42, 2]),
    ]

    def call(state_pool):
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
            output=out,
        )

    def set_tables(step):
        page_ids, counts, last_lens = step
        indices.copy_(torch.tensor(page_ids, dtype=torch.int32))
        indptr.copy_(_cu(counts, "cpu"))
        last.copy_(torch.tensor(last_lens, dtype=torch.int32))

    set_tables(steps[0])
    graph = _capture(lambda warmup: call(scratch if warmup else pool))
    for i, step in enumerate(steps):
        set_tables(step)
        expected_pool = pool.clone()
        expected_out = call(expected_pool).clone()
        graph.replay()
        torch.cuda.synchronize()
        _assert_output_guard(guard)
        # A stale page table reads another page's (unrelated) state.
        torch.testing.assert_close(out, expected_out, atol=1e-3, rtol=1e-3)
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
