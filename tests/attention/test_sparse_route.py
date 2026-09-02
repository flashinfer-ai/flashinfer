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
    torch.cuda.get_device_capability()[0] < 8,
    reason="the FA2 large-head path starts at sm_80",
)


def _expand_reference(
    block_indices, query_positions, sequence_lengths, token_to_req, compress_ratio
):
    """Expand the selection one row at a time, in plain Python.

    The index tensors may be int64 while the kernel works in int32, so a value
    outside that range is not an index at all. Python has no such range, so the
    rule is written out here rather than inherited.
    """
    int32_max = 2147483647

    def in_range(value):
        return 0 <= value <= int32_max

    rows, block_topk = block_indices.shape
    width = block_topk * compress_ratio + compress_ratio - 1
    num_requests = sequence_lengths.shape[0]
    out = torch.full((rows, width), -1, dtype=block_indices.dtype, device=DEV)
    blocks = block_indices.tolist()
    positions = query_positions.tolist()
    lengths = sequence_lengths.tolist()
    requests = token_to_req.tolist()
    for row in range(rows):
        request = requests[row]
        request_ok = in_range(request) and request < num_requests
        raw_seq = lengths[request] if request_ok else 0
        seq = raw_seq if in_range(raw_seq) else 0
        position = positions[row] if in_range(positions[row]) else -1
        past = min((position + 1) // compress_ratio, seq // compress_ratio)
        complete = min(past, block_topk)
        route = []
        for rank in range(complete):
            block = blocks[row][rank]
            # The whole block or none of it: the block the query sits in is
            # partly ahead of it, and the tail below appends its seen half, so
            # expanding it here would route those tokens twice.
            base = block * compress_ratio if in_range(block) and block < past else None
            route.extend(
                None if base is None else base + offset
                for offset in range(compress_ratio)
            )
        tail_start = ((position + 1) // compress_ratio) * compress_ratio
        tail = min((position + 1) - tail_start, compress_ratio - 1)
        route.extend(tail_start + offset for offset in range(tail))
        for column, token in enumerate(route[:width]):
            if token is not None and 0 <= token <= position and token < seq:
                out[row, column] = token
    return out


def _route_reference(
    logical, token_to_req, block_table, page_size, num_slots, valid_rows
):
    """Map the logical route to slots and pack validity, one row at a time."""
    rows, width = logical.shape
    nbytes = -(-width // 8)
    route = torch.zeros((rows, width), dtype=logical.dtype, device=DEV)
    mask = torch.zeros(rows * nbytes, dtype=torch.uint8, device=DEV)
    table = block_table.tolist()
    requests = token_to_req.tolist()
    tokens = logical.tolist()
    table_width = block_table.shape[1]
    for row in range(rows):
        if row >= valid_rows:
            continue
        request = requests[row]
        if not 0 <= request < block_table.shape[0]:
            continue
        for column in range(width):
            token = tokens[row][column]
            if token < 0:
                continue
            page = token // page_size
            if page >= table_width:
                continue
            mapped = table[request][page]
            if mapped < 0:
                continue
            slot = mapped * page_size + token % page_size
            if slot >= num_slots:
                continue
            route[row, column] = slot
            mask[row * nbytes + column // 8] |= 1 << (column % 8)
    return route, mask


def _make_case(
    rows, block_topk, compress_ratio, seq_len, num_requests, page_size, seed
):
    g = torch.Generator(device=DEV).manual_seed(seed)
    blocks = torch.randint(
        0,
        max(1, seq_len // compress_ratio),
        (rows, block_topk),
        dtype=torch.int32,
        device=DEV,
        generator=g,
    )
    positions = torch.randint(
        0, seq_len, (rows,), dtype=torch.int32, device=DEV, generator=g
    )
    lengths = torch.full((num_requests,), seq_len, dtype=torch.int32, device=DEV)
    token_to_req = torch.randint(
        0, num_requests, (rows,), dtype=torch.int32, device=DEV, generator=g
    )
    pages_per_request = (seq_len + page_size - 1) // page_size
    pages = pages_per_request * num_requests
    table = (
        torch.randperm(pages, device=DEV, generator=g)
        .reshape(num_requests, pages_per_request)
        .contiguous()
        .to(torch.int32)
    )
    # unmapped pages the validity has to catch
    table[:, ::7] = -1
    return blocks, positions, lengths, token_to_req, table, pages * page_size


@pytest.mark.parametrize("compress_ratio", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [1, 7, 64, 300])
@pytest.mark.parametrize("block_topk", [1, 16, 128])
def test_expand_block_route(compress_ratio, rows, block_topk):
    blocks, positions, lengths, token_to_req, _, _ = _make_case(
        rows, block_topk, compress_ratio, 512, 4, 64, seed=rows + block_topk
    )
    out = flashinfer.expand_block_route(
        blocks, positions, lengths, token_to_req, compress_ratio
    )
    expected = _expand_reference(
        blocks, positions, lengths, token_to_req, compress_ratio
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_expand_block_route_empties_a_row_without_a_request():
    blocks = torch.zeros(2, 4, dtype=torch.int32, device=DEV)
    positions = torch.tensor([100, 100], dtype=torch.int32, device=DEV)
    lengths = torch.tensor([512], dtype=torch.int32, device=DEV)
    token_to_req = torch.tensor([0, -1], dtype=torch.int32, device=DEV)
    out = flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, 4)
    assert int((out[1] >= 0).sum()) == 0
    assert int((out[0] >= 0).sum()) > 0


def test_expand_block_route_drops_a_block_the_query_has_not_reached():
    """The block ids are the caller's. A block past the query expands into
    tokens the query cannot see -- at a ratio of four, a query at position 3
    selecting block 2 would route tokens 8 through 11 -- and the sequence
    length alone does not bound them."""
    ratio = 4
    blocks = torch.tensor([[2, 0, 0, 0]], dtype=torch.int32, device=DEV)
    positions = torch.tensor([3], dtype=torch.int32, device=DEV)
    lengths = torch.tensor([512], dtype=torch.int32, device=DEV)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=DEV)
    out = flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, ratio)
    assert int(out.max()) <= 3, f"routed a token past the query: {out.tolist()}"
    assert out[0, :ratio].tolist() == [-1] * ratio
    expected = _expand_reference(blocks, positions, lengths, token_to_req, ratio)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_expand_block_route_does_not_repeat_the_block_the_query_sits_in():
    """The block a query sits in is half behind it, and the tail is what
    appends that half. Keeping the seen tokens of that block during expansion
    instead of dropping the block whole would route them twice -- at a ratio of
    four and a query at position 9, block 2 covers tokens 8 to 11 and the tail
    already carries 8 and 9."""
    ratio = 4
    # One column, so the only expansion is the offending block.
    blocks = torch.tensor([[2]], dtype=torch.int32, device=DEV)
    positions = torch.tensor([9], dtype=torch.int32, device=DEV)
    lengths = torch.tensor([512], dtype=torch.int32, device=DEV)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=DEV)
    out = flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, ratio)
    routed = [t for t in out[0].tolist() if t >= 0]
    assert len(routed) == len(set(routed)), f"a token is routed twice: {routed}"
    assert routed == [8, 9], routed
    assert out[0, :ratio].tolist() == [-1] * ratio
    expected = _expand_reference(blocks, positions, lengths, token_to_req, ratio)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("compress_ratio", [2, 4])
@pytest.mark.parametrize("rows", [1, 9, 128])
@pytest.mark.parametrize("page_size", [1, 16, 64])
def test_qsa_route_from_blocks(compress_ratio, rows, page_size):
    block_topk = 32
    blocks, positions, lengths, token_to_req, table, num_slots = _make_case(
        rows, block_topk, compress_ratio, 1024, 4, page_size, seed=rows + page_size
    )
    width = block_topk * compress_ratio + compress_ratio - 1
    nbytes = -(-width // 8)
    logical = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    route = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(rows * nbytes, dtype=torch.uint8, device=DEV)

    flashinfer.qsa_route_from_blocks(
        blocks,
        positions,
        lengths,
        token_to_req,
        table,
        logical,
        route,
        mask,
        compress_ratio,
        page_size,
        num_slots,
    )

    expected_logical = _expand_reference(
        blocks, positions, lengths, token_to_req, compress_ratio
    )
    torch.testing.assert_close(logical, expected_logical, rtol=0, atol=0)
    expected_route, expected_mask = _route_reference(
        expected_logical, token_to_req, table, page_size, num_slots, rows
    )
    torch.testing.assert_close(route, expected_route, rtol=0, atol=0)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)


@pytest.mark.parametrize("rows,valid_rows", [(1, 1), (16, 16), (128, 100), (64, 0)])
@pytest.mark.parametrize("page_size", [1, 64])
def test_qsa_route_from_logical(rows, valid_rows, page_size):
    """Padding rows must come out fully masked, whatever the logical route holds."""
    block_topk, compress_ratio = 32, 4
    blocks, positions, lengths, token_to_req, table, num_slots = _make_case(
        rows, block_topk, compress_ratio, 1024, 4, page_size, seed=rows
    )
    logical = _expand_reference(
        blocks, positions, lengths, token_to_req, compress_ratio
    )
    width = logical.shape[1]
    nbytes = -(-width // 8)
    route = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(rows * nbytes, dtype=torch.uint8, device=DEV)

    flashinfer.qsa_route_from_logical(
        logical, token_to_req, table, route, mask, valid_rows, page_size, num_slots
    )
    expected_route, expected_mask = _route_reference(
        logical, token_to_req, table, page_size, num_slots, valid_rows
    )
    torch.testing.assert_close(route, expected_route, rtol=0, atol=0)
    torch.testing.assert_close(mask, expected_mask, rtol=0, atol=0)


def test_qsa_route_never_points_outside_the_cache():
    """An entry the mask clears must still hold an in-range slot."""
    rows, block_topk, compress_ratio, page_size = 32, 16, 4, 16
    blocks, positions, lengths, token_to_req, table, num_slots = _make_case(
        rows, block_topk, compress_ratio, 256, 2, page_size, seed=7
    )
    # every page unmapped: nothing is valid, and every slot must still be legal
    table.fill_(-1)
    width = block_topk * compress_ratio + compress_ratio - 1
    nbytes = -(-width // 8)
    logical = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    route = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(rows * nbytes, dtype=torch.uint8, device=DEV)
    flashinfer.qsa_route_from_blocks(
        blocks,
        positions,
        lengths,
        token_to_req,
        table,
        logical,
        route,
        mask,
        compress_ratio,
        page_size,
        num_slots,
    )
    assert int(mask.sum()) == 0
    assert int(route.min()) >= 0
    assert int(route.max()) < num_slots


def test_qsa_route_rejects_a_mapped_page_past_the_cache():
    """An unmapped page is not the only way out of range.

    A page the table maps is trusted for its id but not for where that id
    lands: a mapping whose first slot is already at the end of the cache has
    to clear as well, or the route walks past it holding a legal-looking id.
    """
    rows, block_topk, compress_ratio, page_size = 32, 16, 4, 16
    blocks, positions, lengths, token_to_req, table, num_slots = _make_case(
        rows, block_topk, compress_ratio, 256, 2, page_size, seed=7
    )
    # Mapped, and one page past the last the cache holds.
    table.fill_(num_slots // page_size)
    width = block_topk * compress_ratio + compress_ratio - 1
    nbytes = -(-width // 8)
    logical = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    route = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(rows * nbytes, dtype=torch.uint8, device=DEV)
    flashinfer.qsa_route_from_blocks(
        blocks,
        positions,
        lengths,
        token_to_req,
        table,
        logical,
        route,
        mask,
        compress_ratio,
        page_size,
        num_slots,
    )
    assert int(mask.sum()) == 0
    assert int(route.min()) >= 0
    assert int(route.max()) < num_slots


def test_route_ops_reject_bad_arguments():
    blocks = torch.zeros(4, 8, dtype=torch.int32, device=DEV)
    positions = torch.zeros(4, dtype=torch.int32, device=DEV)
    lengths = torch.full((1,), 64, dtype=torch.int32, device=DEV)
    token_to_req = torch.zeros(4, dtype=torch.int32, device=DEV)

    with pytest.raises(ValueError, match="compress_ratio"):
        flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, 0)
    with pytest.raises(ValueError, match="2D"):
        flashinfer.expand_block_route(blocks[0], positions, lengths, token_to_req, 4)
    with pytest.raises(ValueError, match="shape"):
        flashinfer.expand_block_route(
            blocks,
            positions,
            lengths,
            token_to_req,
            4,
            out=torch.zeros(4, 8, dtype=torch.int32, device=DEV),
        )
    # a ratio the dispatch has no kernel for
    with pytest.raises(Exception, match="compress_ratio"):
        flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, 3)


# Values that do not fit an int32. The index tensors may be int64 and the
# kernels narrow, so a wrapped one must not become a valid position, request or
# block: 2^32 exactly wraps to zero, which is a real block and a real request.
_PAST_INT32 = [2147483648, 4294967296, 9223372036854775807, -2147483649]


@pytest.mark.parametrize("bad", _PAST_INT32)
@pytest.mark.parametrize("field", ["positions", "token_to_req", "blocks", "lengths"])
def test_expand_block_route_does_not_wrap_an_index_that_is_not_an_int32(field, bad):
    ratio = 4
    # Distinct blocks, so a repeat in the output can only come from the kernel
    # and not from a selector that named the same block twice.
    blocks = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=DEV)
    positions = torch.tensor([31], dtype=torch.int64, device=DEV)
    lengths = torch.tensor([512], dtype=torch.int64, device=DEV)
    token_to_req = torch.zeros(1, dtype=torch.int64, device=DEV)
    field_map = {
        "positions": positions,
        "token_to_req": token_to_req,
        "lengths": lengths,
    }
    if field == "blocks":
        blocks[0, 0] = bad  # rank 0; ranks 1..3 stay valid
    else:
        field_map[field][0] = bad

    out = flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, ratio)
    torch.cuda.synchronize()
    routed = [t for t in out[0].tolist() if t >= 0]
    assert len(routed) == len(set(routed)), f"a token is routed twice: {routed}"
    if field == "blocks":
        # The rank that named it contributes nothing; the rest still route.
        assert out[0, :ratio].tolist() == [-1] * ratio
    else:
        # No request, no position and no length the row can resolve, so it
        # routes nothing at all.
        assert routed == [], routed
    expected = _expand_reference(blocks, positions, lengths, token_to_req, ratio)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
