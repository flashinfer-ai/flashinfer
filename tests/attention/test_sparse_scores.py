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
"""

import pytest
import torch

import flashinfer

DEV = "cuda:0"

requires_cuda_sm80 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(DEV)[0] < 8,
    reason="the scorer's tensor-core path starts at sm_80",
)


def _scores_case(
    head_dim=128, num_heads=4, rows=8, pages=6, page_size=8, dtype=torch.bfloat16
):
    """An aligned scorer case, plus what it takes to re-run it from a view."""
    torch.manual_seed(0)
    q = torch.randn(rows, num_heads, head_dim, dtype=dtype, device=DEV)
    k_cache = torch.randn(pages, page_size, head_dim, dtype=dtype, device=DEV)
    page_table = torch.arange(pages, dtype=torch.int32, device=DEV).reshape(1, pages)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=DEV)
    positions = torch.arange(rows, dtype=torch.int32, device=DEV) + pages * page_size
    seq_lens = torch.full((1,), pages * page_size, dtype=torch.int32, device=DEV)
    return q, k_cache, page_table, token_to_req, positions, seq_lens


def _run_scores(q, k_cache, page_table, token_to_req, positions, seq_lens):
    return flashinfer.sparse_paged_scores(
        q, k_cache, page_table, token_to_req, positions, seq_lens, 1, q.shape[2] ** 0.5
    )


@requires_cuda_sm80
def test_scores_accept_a_query_view_with_a_storage_offset():
    """A view onto the middle of a tensor keeps its strides and loses its
    alignment, which the staged 128-bit loads cannot assume away."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    want, want_visible = _run_scores(q, k_cache, table, t2r, pos, lens)

    head_dim = q.shape[2]
    # Pad by a whole vector so every stride stays a multiple of it, and take the
    # slice one element in. Padding by one instead would misalign the strides
    # too, and the staging already refuses those -- the base would never be
    # reached.
    backing = torch.empty(
        q.shape[0], q.shape[1], head_dim + 8, dtype=q.dtype, device=DEV
    )
    backing[..., 1 : 1 + head_dim] = q
    offset_q = backing[..., 1 : 1 + head_dim]
    assert offset_q.storage_offset() % 8 != 0
    assert all(stride % 8 == 0 for stride in offset_q.stride()[:-1])
    got, got_visible = _run_scores(offset_q, k_cache, table, t2r, pos, lens)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_visible, want_visible)


@requires_cuda_sm80
def test_scores_accept_a_cache_view_with_a_storage_offset():
    """The same for the cache, which is staged through cp_async."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    want, want_visible = _run_scores(q, k_cache, table, t2r, pos, lens)

    pages, page_size, head_dim = k_cache.shape
    # Same shape of view: aligned page and entry strides, misaligned base.
    backing = torch.empty(
        pages, page_size, head_dim + 8, dtype=k_cache.dtype, device=DEV
    )
    backing[..., 1 : 1 + head_dim] = k_cache
    offset_cache = backing[..., 1 : 1 + head_dim]
    assert offset_cache.storage_offset() % 8 != 0
    assert all(stride % 8 == 0 for stride in offset_cache.stride()[:-1])
    got, got_visible = _run_scores(q, offset_cache, table, t2r, pos, lens)
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_visible, want_visible)


def _reference(
    q,
    k_cache,
    page_table,
    token_to_req,
    positions,
    seq_lens,
    compress_ratio,
    divisor,
    num_columns,
):
    """What the kernel should produce, in plain torch and float32.

    Independent of the kernel: it walks the page table itself and never calls
    into flashinfer, so a shared mistake cannot make both agree.
    """
    rows = q.shape[0]
    pages, page_size = k_cache.shape[0], k_cache.shape[1]
    logits = torch.full((rows, num_columns), float("-inf"), device=q.device)
    visible = torch.zeros(rows, dtype=page_table.dtype, device=q.device)
    keys = k_cache.float()
    qf = q.float()
    for row in range(rows):
        req = int(token_to_req[row])
        if req < 0 or req >= page_table.shape[0]:
            continue
        seen = min(
            (int(positions[row]) + 1) // compress_ratio,
            int(seq_lens[req]) // compress_ratio,
        )
        visible[row] = min(seen, num_columns)
        for col in range(min(seen, num_columns)):
            page = int(page_table[req, col // page_size])
            if page < 0 or page >= pages:
                continue  # unmapped: the kernel leaves -inf here
            k = keys[page, col % page_size]
            logits[row, col] = torch.clamp(qf[row] @ k, min=0).sum() / divisor
    return logits, visible


@requires_cuda_sm80
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128, 192, 256])
@pytest.mark.parametrize("num_heads", [1, 4, 8, 9, 16])
def test_scores_match_a_torch_reference(dtype, head_dim, num_heads):
    """Eight heads and nine straddle the two multiply widths: up to eight the
    tile is m16n8k16 fed by an ldmatrix.x2, above it m16n16k16 fed by an x4 with
    the unused heads zeroed. Sixteen heads at head_dim 256 is also the one shape
    whose query fragments do not fit in registers, so it reloads them per
    multiply instead of holding them across the column tiles."""
    q, k_cache, table, t2r, pos, lens = _scores_case(
        head_dim=head_dim, num_heads=num_heads, dtype=dtype
    )
    divisor = head_dim**0.5
    columns = table.shape[1] * k_cache.shape[1]
    got, got_visible = flashinfer.sparse_paged_scores(
        q, k_cache, table, t2r, pos, lens, 1, divisor
    )
    want, want_visible = _reference(
        q, k_cache, table, t2r, pos, lens, 1, divisor, columns
    )
    torch.testing.assert_close(got_visible, want_visible)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@requires_cuda_sm80
@pytest.mark.parametrize("rows", [1, 96])
def test_scores_match_a_torch_reference_across_launch_shapes(rows):
    """One row does not fill the device and takes the deeper staging shape;
    many rows take the wider one."""
    q, k_cache, table, t2r, pos, lens = _scores_case(rows=rows, pages=8)
    divisor = q.shape[2] ** 0.5
    columns = table.shape[1] * k_cache.shape[1]
    got, got_visible = flashinfer.sparse_paged_scores(
        q, k_cache, table, t2r, pos, lens, 1, divisor
    )
    want, want_visible = _reference(
        q, k_cache, table, t2r, pos, lens, 1, divisor, columns
    )
    torch.testing.assert_close(got_visible, want_visible)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@requires_cuda_sm80
def test_scores_agree_between_a_block_per_tile_and_a_block_per_eight():
    """Several waves of blocks switch the launch from one column tile per block
    to eight, which stages a feature slice at a time and runs the copies for the
    next tile while the current one multiplies. A single row is one wave at
    most, so scoring the rows one at a time takes the other shape; the two have
    to agree column for column.

    The row count is taken from the device so the two sides really do land on
    opposite sides of the switch: the kernel turns over at four waves, and a
    block never holds more than 32 warps, so eight rows per SM is past it on any
    part.

    The reference the single-tile shape is checked against is the torch one
    above; this pins the pipelined shape to it without paying for a python loop
    over every column of every row.
    """
    pages = 64
    rows = 8 * torch.cuda.get_device_properties(DEV).multi_processor_count
    q, k_cache, table, t2r, pos, lens = _scores_case(rows=rows, pages=pages)
    divisor = q.shape[2] ** 0.5
    columns = table.shape[1] * k_cache.shape[1]
    got, got_visible = flashinfer.sparse_paged_scores(
        q, k_cache, table, t2r, pos, lens, 1, divisor
    )
    assert got.shape == (rows, columns)
    for row in range(rows):
        one, one_visible = flashinfer.sparse_paged_scores(
            q[row : row + 1],
            k_cache,
            table,
            t2r[row : row + 1],
            pos[row : row + 1],
            lens,
            1,
            divisor,
        )
        torch.testing.assert_close(got_visible[row : row + 1], one_visible)
        torch.testing.assert_close(got[row : row + 1], one)


@requires_cuda_sm80
def test_scores_read_a_cache_with_no_pages_at_all():
    """A column with no page carries the first page's address so the staging
    loop needs no predicate. With no page there is no first one to carry, so the
    staging has to be skipped outright rather than read the empty tensor."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    empty = k_cache[:0]
    divisor = q.shape[2] ** 0.5
    columns = table.shape[1] * k_cache.shape[1]
    got, got_visible = flashinfer.sparse_paged_scores(
        q, empty, table, t2r, pos, lens, 1, divisor, num_columns=columns
    )
    want, want_visible = _reference(
        q, empty, table, t2r, pos, lens, 1, divisor, columns
    )
    torch.testing.assert_close(got_visible, want_visible)
    torch.testing.assert_close(got, want)
    assert torch.isinf(got).all()


@requires_cuda_sm80
def test_scores_leave_an_unmapped_page_unselectable():
    q, k_cache, table, t2r, pos, lens = _scores_case()
    table = table.clone()
    table[0, 1] = -1
    divisor = q.shape[2] ** 0.5
    columns = table.shape[1] * k_cache.shape[1]
    got, _ = flashinfer.sparse_paged_scores(
        q, k_cache, table, t2r, pos, lens, 1, divisor
    )
    want, _ = _reference(q, k_cache, table, t2r, pos, lens, 1, divisor, columns)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    page_size = k_cache.shape[1]
    assert torch.isinf(got[:, page_size : 2 * page_size]).all()


@requires_cuda_sm80
def test_scores_empty_a_row_whose_request_is_invalid():
    q, k_cache, table, t2r, pos, lens = _scores_case()
    t2r = t2r.clone()
    t2r[0] = -1
    divisor = q.shape[2] ** 0.5
    _, got_visible = flashinfer.sparse_paged_scores(
        q, k_cache, table, t2r, pos, lens, 1, divisor
    )
    assert int(got_visible[0]) == 0


@requires_cuda_sm80
def test_scores_stop_at_num_columns():
    """A width narrower than the query can see caps both the logits and the
    count the caller bounds its top-k by."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    divisor = q.shape[2] ** 0.5
    narrow = 8
    got, got_visible = flashinfer.sparse_paged_scores(
        q, k_cache, table, t2r, pos, lens, 1, divisor, num_columns=narrow
    )
    want, want_visible = _reference(
        q, k_cache, table, t2r, pos, lens, 1, divisor, narrow
    )
    assert got.shape[1] == narrow
    assert int(got_visible.max()) <= narrow
    torch.testing.assert_close(got_visible, want_visible)
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


@requires_cuda_sm80
def test_scores_reject_a_logits_width_that_contradicts_num_columns():
    """The kernel takes the width from the tensor it writes, so a wider logits
    would score more than was asked for and report a count past it."""
    q, k_cache, table, t2r, pos, lens = _scores_case()
    logits = torch.empty(q.shape[0], 12, dtype=torch.float32, device=DEV)
    with pytest.raises(ValueError, match="columns wide"):
        flashinfer.sparse_paged_scores(
            q,
            k_cache,
            table,
            t2r,
            pos,
            lens,
            1,
            q.shape[2] ** 0.5,
            num_columns=8,
            logits=logits,
        )


@requires_cuda_sm80
def test_scores_leave_the_columns_past_a_row_alone():
    """Only what a row can see is written.

    The count says how far to read, so the columns past it have to keep
    whatever the caller left there -- a top-k bounded by the count never looks
    at them, and a caller reusing the buffer would otherwise read this row's
    logits as the next row's.
    """
    q, k_cache, table, t2r, pos, lens = _scores_case(rows=8, pages=6, page_size=8)
    divisor = q.shape[2] ** 0.5
    columns = table.shape[1] * k_cache.shape[1]
    # Row i sees i entries, so every row but the last has untouched columns.
    pos = torch.arange(q.shape[0], dtype=pos.dtype, device=DEV)
    sentinel = -12345.0
    logits = torch.full(
        (q.shape[0], columns), sentinel, dtype=torch.float32, device=DEV
    )
    got, got_visible = flashinfer.sparse_paged_scores(
        q, k_cache, table, t2r, pos, lens, 1, divisor, logits=logits
    )
    want, want_visible = _reference(
        q, k_cache, table, t2r, pos, lens, 1, divisor, columns
    )
    torch.testing.assert_close(got_visible, want_visible)
    assert int(got_visible.min()) < columns, "a row must have untouched columns"
    for row in range(q.shape[0]):
        seen = int(got_visible[row])
        assert (got[row, seen:] == sentinel).all(), f"row {row} wrote past its count"
        torch.testing.assert_close(
            got[row, :seen], want[row, :seen], rtol=2e-2, atol=2e-2
        )
