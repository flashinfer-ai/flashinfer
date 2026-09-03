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
        route = []
        # Every rank gets its columns; whether the block it names is usable is
        # decided per block, not by truncating the rank range.
        for rank in range(block_topk):
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


@pytest.mark.parametrize("field", ["positions", "lengths"])
def test_expand_block_route_takes_a_value_at_the_top_of_the_int32_range(field):
    """INT32_MAX is inside the range the kernels accept, so what it exercises is
    the arithmetic after the cast: the block bound and the tail both come from
    position + 1, and adding one to INT32_MAX in int32 is signed overflow."""
    ratio = 4
    blocks = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=DEV)
    positions = torch.tensor([31], dtype=torch.int64, device=DEV)
    lengths = torch.tensor([512], dtype=torch.int64, device=DEV)
    token_to_req = torch.zeros(1, dtype=torch.int64, device=DEV)
    if field == "positions":
        positions[0] = 2147483647
    else:
        lengths[0] = 2147483647
    out = flashinfer.expand_block_route(blocks, positions, lengths, token_to_req, ratio)
    torch.cuda.synchronize()
    routed = [t for t in out[0].tolist() if t >= 0]
    assert len(routed) == len(set(routed)), routed
    expected = _expand_reference(blocks, positions, lengths, token_to_req, ratio)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("path", ["logical", "blocks"])
def test_qsa_route_does_not_let_a_slot_wrap_into_range(path):
    """The physical slot is page * page_size + entry. Both factors fit a uint32
    while their product does not, and a product computed in 32 bits wraps to a
    small number that passes the num_slots bound -- page 65536 of a 65536-entry
    page is exactly 2^32, which would arrive as slot 0.

    The page itself is inside the slot space here, so the bound that precedes
    the multiply lets it through and the width of the multiply is what decides
    the case. That needs the largest slot space a route can be given, which is
    an int64 one.

    Both paths, because the fused kernel carries its own copy of the multiply:
    narrowing one of them is not caught by exercising the other.
    """
    page_size, num_slots = 65536, 4294967295
    big_page = 65536  # times the page size, exactly 2^32
    if path == "logical":
        width = 8
        logical = torch.arange(width, dtype=torch.int64, device=DEV).reshape(1, width)
        token_to_req = torch.zeros(1, dtype=torch.int64, device=DEV)
        table = torch.tensor([[big_page]], dtype=torch.int64, device=DEV)
        route = torch.empty(1, width, dtype=torch.int64, device=DEV)
        mask = torch.empty(-(-width // 8), dtype=torch.uint8, device=DEV)
        flashinfer.qsa_route_from_logical(
            logical, token_to_req, table, route, mask, 1, page_size, num_slots
        )
    else:
        ratio = 1
        width = 4 * ratio + ratio - 1
        blocks = torch.zeros(1, 4, dtype=torch.int64, device=DEV)
        positions = torch.tensor([7], dtype=torch.int64, device=DEV)
        lengths = torch.tensor([8], dtype=torch.int64, device=DEV)
        token_to_req = torch.zeros(1, dtype=torch.int64, device=DEV)
        table = torch.tensor([[big_page]], dtype=torch.int64, device=DEV)
        logical = torch.empty(1, width, dtype=torch.int64, device=DEV)
        route = torch.empty(1, width, dtype=torch.int64, device=DEV)
        mask = torch.empty(-(-width // 8), dtype=torch.uint8, device=DEV)
        flashinfer.qsa_route_from_blocks(
            blocks,
            positions,
            lengths,
            token_to_req,
            table,
            logical,
            route,
            mask,
            ratio,
            page_size,
            num_slots,
        )
    torch.cuda.synchronize()
    assert int(mask[0]) == 0, (
        f"a slot past num_slots was accepted: route={route.tolist()} mask={mask.tolist()}"
    )


def test_qsa_route_from_logical_refuses_a_slot_space_its_route_cannot_hold():
    """A slot is written into the route's own dtype. An int32 route cannot hold
    2^31, which would come back out as a negative index with its mask bit set,
    so a slot space that large has to be refused rather than truncated."""
    width = 8
    logical = torch.arange(width, dtype=torch.int32, device=DEV).reshape(1, width)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=DEV)
    table = torch.zeros(1, 1, dtype=torch.int32, device=DEV)
    route = torch.empty(1, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(1, dtype=torch.uint8, device=DEV)
    # A count of 2^31 is slots 0 .. INT32_MAX, which the route holds; one more
    # is not.
    flashinfer.qsa_route_from_logical(
        logical, token_to_req, table, route, mask, 1, 65536, 2147483648
    )
    with pytest.raises(Exception, match="fit the route dtype"):
        flashinfer.qsa_route_from_logical(
            logical, token_to_req, table, route, mask, 1, 65536, 2147483649
        )


def test_qsa_route_from_blocks_bounds_the_ratio_before_it_derives_a_width():
    """The route width is block_topk * compress_ratio + compress_ratio - 1. A
    ratio near the top of int64 overflows that product before any check on the
    ratio could reject it, so the bound has to come first."""
    blocks = torch.zeros(1, 4, dtype=torch.int32, device=DEV)
    positions = torch.zeros(1, dtype=torch.int32, device=DEV)
    lengths = torch.tensor([16], dtype=torch.int32, device=DEV)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=DEV)
    table = torch.zeros(1, 1, dtype=torch.int32, device=DEV)
    logical = torch.empty(1, 8, dtype=torch.int32, device=DEV)
    route = torch.empty(1, 8, dtype=torch.int32, device=DEV)
    mask = torch.empty(1, dtype=torch.uint8, device=DEV)
    with pytest.raises(Exception, match="compress_ratio must fit in 32 bits"):
        flashinfer.qsa_route_from_blocks(
            blocks,
            positions,
            lengths,
            token_to_req,
            table,
            logical,
            route,
            mask,
            9223372036854775807,
            16,
            256,
        )


@pytest.mark.parametrize("bad", _PAST_INT32)
@pytest.mark.parametrize("field", ["token_to_req", "logical", "page_table"])
def test_qsa_route_from_logical_does_not_wrap_an_index_that_is_not_an_int32(field, bad):
    """The slot kernel narrows the request and the logical token just as the
    expansion does, so neither may come back as request or token zero. The page
    id is not narrowed at all -- it is bounded against the slot space in IdType
    and then widened -- so what this asks of it is that a value it cannot
    address is refused rather than folded into one it can."""
    width, page_size, num_slots = 8, 16, 4096
    logical = torch.arange(width, dtype=torch.int64, device=DEV).reshape(1, width)
    token_to_req = torch.zeros(1, dtype=torch.int64, device=DEV)
    table = torch.zeros(1, 4, dtype=torch.int64, device=DEV)
    if field == "token_to_req":
        token_to_req[0] = bad
    elif field == "logical":
        logical[0, 0] = bad
    else:
        table[0, 0] = bad

    route = torch.empty(1, width, dtype=torch.int64, device=DEV)
    mask = torch.empty(1, dtype=torch.uint8, device=DEV)
    flashinfer.qsa_route_from_logical(
        logical, token_to_req, table, route, mask, 1, page_size, num_slots
    )
    torch.cuda.synchronize()
    live = [c for c in range(width) if (int(mask[0]) >> c) & 1]
    if field == "logical":
        # Only that column is unusable; the rest of the row still routes.
        assert 0 not in live, f"a wrapped token was routed: {route[0].tolist()}"
    else:
        # No request, or no page under it, so the whole row is masked.
        assert live == [], f"a wrapped index was routed: {route[0].tolist()}"


@pytest.mark.parametrize("scalar", ["page_size", "num_slots", "compress_ratio"])
def test_qsa_route_rejects_a_host_scalar_that_is_not_a_uint32(scalar):
    """page_size, num_slots and compress_ratio are narrowed to uint32 for the
    kernel, so 2^32 arrives as zero -- a page size or a compression ratio to
    divide by."""
    blocks = torch.zeros(1, 4, dtype=torch.int32, device=DEV)
    positions = torch.zeros(1, dtype=torch.int32, device=DEV)
    lengths = torch.tensor([16], dtype=torch.int32, device=DEV)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=DEV)
    table = torch.zeros(1, 1, dtype=torch.int32, device=DEV)
    args = dict(compress_ratio=1, page_size=16, num_slots=256)
    args[scalar] = 4294967296
    # Sized for the ratio the binding will accept, not the one under test: a
    # width derived from 2^32 is 21 billion columns and the allocation is what
    # would fail rather than the check.
    width = 4 * 1 + 1 - 1
    logical = torch.empty(1, width, dtype=torch.int32, device=DEV)
    route = torch.empty(1, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(-(-width // 8), dtype=torch.uint8, device=DEV)
    with pytest.raises(Exception, match="must fit in 32 bits"):
        flashinfer.qsa_route_from_blocks(
            blocks,
            positions,
            lengths,
            token_to_req,
            table,
            logical,
            route,
            mask,
            args["compress_ratio"],
            args["page_size"],
            args["num_slots"],
        )


@pytest.mark.parametrize("bad", _PAST_INT32)
@pytest.mark.parametrize("field", ["token_to_req", "blocks", "page_table"])
def test_qsa_route_from_blocks_does_not_wrap_an_index_that_is_not_an_int32(field, bad):
    """The fused kernel expands and resolves in one pass, with its own copy of
    every guard, so the same values are injected here and not only into the
    standalone slot kernel.

    Removing those copies does not fail this, for different reasons per guard.
    The block bound is genuinely redundant: block < past_blocks is applied in
    IdType and past_blocks is an int32 count. The request bound is not -- it is
    compared against num_requests, which is a tensor extent with no int32 cap of
    its own, so a large enough request count would let a wrapped id through.
    This case cannot show that, since building a request table of two billion
    rows is not a test. What the boundary test above pins is the query position,
    which nothing else bounds.
    """
    ratio, page_size, num_slots = 4, 16, 4096
    blocks = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=DEV)
    positions = torch.tensor([31], dtype=torch.int64, device=DEV)
    lengths = torch.tensor([512], dtype=torch.int64, device=DEV)
    token_to_req = torch.zeros(1, dtype=torch.int64, device=DEV)
    table = torch.zeros(1, 32, dtype=torch.int64, device=DEV)
    if field == "token_to_req":
        token_to_req[0] = bad
    elif field == "blocks":
        blocks[0, 0] = bad
    else:
        table[0, 0] = bad

    width = 4 * ratio + ratio - 1
    logical = torch.empty(1, width, dtype=torch.int64, device=DEV)
    route = torch.empty(1, width, dtype=torch.int64, device=DEV)
    mask = torch.empty(-(-width // 8), dtype=torch.uint8, device=DEV)
    flashinfer.qsa_route_from_blocks(
        blocks,
        positions,
        lengths,
        token_to_req,
        table,
        logical,
        route,
        mask,
        ratio,
        page_size,
        num_slots,
    )
    torch.cuda.synchronize()
    routed = [t for t in logical[0].tolist() if t >= 0]
    assert len(routed) == len(set(routed)), routed
    live = [
        c
        for c in range(width)
        if (int(mask[-(-width // 8) * 0 + c // 8]) >> (c % 8)) & 1
    ]
    if field == "token_to_req":
        assert live == [], f"a wrapped request routed: {route[0].tolist()}"
    else:
        # Only the rank or the page that carried it goes; the rest still route.
        assert all(t <= 31 for t in routed), routed
        assert 0 not in live, f"a wrapped index routed: {route[0].tolist()}"


@pytest.mark.parametrize("path", ["logical", "blocks"])
def test_qsa_route_keeps_a_page_an_int64_route_can_hold(path):
    """A route of int64 is allowed a slot space past what an int32 holds, so a
    page that lands inside it is a page the caller is entitled to. Bounding the
    page id at INT32_MAX instead of at the slot space would mask it: page 2^31
    at a page size of one is slot 2^31, inside a space of 2^31 + 1."""
    page_size, num_slots = 1, 2147483649
    big_page = 2147483648
    if path == "logical":
        width = 4
        logical = torch.zeros(1, width, dtype=torch.int64, device=DEV)
        token_to_req = torch.zeros(1, dtype=torch.int64, device=DEV)
        table = torch.full((1, 1), big_page, dtype=torch.int64, device=DEV)
        route = torch.empty(1, width, dtype=torch.int64, device=DEV)
        mask = torch.empty(1, dtype=torch.uint8, device=DEV)
        flashinfer.qsa_route_from_logical(
            logical, token_to_req, table, route, mask, 1, page_size, num_slots
        )
    else:
        ratio = 1
        width = 4 * ratio + ratio - 1
        blocks = torch.zeros(1, 4, dtype=torch.int64, device=DEV)
        positions = torch.tensor([7], dtype=torch.int64, device=DEV)
        lengths = torch.tensor([8], dtype=torch.int64, device=DEV)
        token_to_req = torch.zeros(1, dtype=torch.int64, device=DEV)
        table = torch.full((1, 8), big_page, dtype=torch.int64, device=DEV)
        logical = torch.empty(1, width, dtype=torch.int64, device=DEV)
        route = torch.empty(1, width, dtype=torch.int64, device=DEV)
        mask = torch.empty(-(-width // 8), dtype=torch.uint8, device=DEV)
        flashinfer.qsa_route_from_blocks(
            blocks,
            positions,
            lengths,
            token_to_req,
            table,
            logical,
            route,
            mask,
            ratio,
            page_size,
            num_slots,
        )
    torch.cuda.synchronize()
    assert int(mask[0]) & 1, "a page inside the slot space was masked out"
    assert int(route[0, 0]) == big_page, route[0, 0].item()


@pytest.mark.parametrize("path", ["logical", "blocks"])
def test_expand_block_route_reads_every_rank_the_selector_filled(path):
    """The selector's ranks are its own business. A valid block sitting at a
    rank the query happens to be short of is still a valid block: with a ratio
    of four and a query at position 7, two blocks are behind the query, and a
    selection of [-1, -1, 0, 1] has both of them at ranks 2 and 3. Sizing the
    expanded region by the query rather than by the rank count dropped them and
    reinterpreted their columns as tail columns."""
    ratio = 4
    blocks = torch.tensor([[-1, -1, 0, 1]], dtype=torch.int32, device=DEV)
    positions = torch.tensor([7], dtype=torch.int32, device=DEV)
    lengths = torch.tensor([16], dtype=torch.int32, device=DEV)
    token_to_req = torch.zeros(1, dtype=torch.int32, device=DEV)
    if path == "logical":
        out = flashinfer.expand_block_route(
            blocks, positions, lengths, token_to_req, ratio
        )
    else:
        page_size, width = 4, 4 * ratio + ratio - 1
        table = torch.arange(4, dtype=torch.int32, device=DEV).reshape(1, 4)
        out = torch.empty(1, width, dtype=torch.int32, device=DEV)
        route = torch.empty(1, width, dtype=torch.int32, device=DEV)
        mask = torch.empty(-(-width // 8), dtype=torch.uint8, device=DEV)
        flashinfer.qsa_route_from_blocks(
            blocks,
            positions,
            lengths,
            token_to_req,
            table,
            out,
            route,
            mask,
            ratio,
            page_size,
            16,
        )
    torch.cuda.synchronize()
    routed = sorted(t for t in out[0].tolist() if t >= 0)
    assert routed == [0, 1, 2, 3, 4, 5, 6, 7], routed
    expected = _expand_reference(blocks, positions, lengths, token_to_req, ratio)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("path", ["expand", "logical", "blocks"])
def test_a_route_covers_more_rows_than_one_grid_can_hold(path):
    """Rows go on the grid's y dimension, which stops at 65535 on every compute
    capability, and a row is a query token. A long-context step goes past that,
    so the rows a grid cannot cover are walked by a stride rather than dropped
    or refused.

    Every output each entry point writes is compared against the reference, not
    just the first: a stride that loses rows in the physical route or in the
    packed mask would not show in the logical one.
    """
    rows, ratio, topk = 70000, 4, 2
    assert rows > 65535
    blocks = torch.zeros(rows, topk, dtype=torch.int32, device=DEV)
    positions = torch.full((rows,), 31, dtype=torch.int32, device=DEV)
    lengths = torch.tensor([64], dtype=torch.int32, device=DEV)
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=DEV)
    width = topk * ratio + ratio - 1
    page_size, pages = 8, 8
    num_slots = pages * page_size
    table = torch.arange(pages, dtype=torch.int32, device=DEV).reshape(1, pages)

    want_logical = _expand_reference(blocks, positions, lengths, token_to_req, ratio)
    want_route, want_mask = _route_reference(
        want_logical, token_to_req, table, page_size, num_slots, rows
    )

    if path == "expand":
        got = flashinfer.expand_block_route(
            blocks, positions, lengths, token_to_req, ratio
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(got, want_logical, rtol=0, atol=0)
        # A row that the stride skipped would be left as -1 rather than routed.
        assert int((got[-1] >= 0).sum()) > 0, got[-1].tolist()
        return

    route = torch.empty(rows, width, dtype=torch.int32, device=DEV)
    mask = torch.empty(rows * (-(-width // 8)), dtype=torch.uint8, device=DEV)
    if path == "blocks":
        logical = torch.empty(rows, width, dtype=torch.int32, device=DEV)
        flashinfer.qsa_route_from_blocks(
            blocks,
            positions,
            lengths,
            token_to_req,
            table,
            logical,
            route,
            mask,
            ratio,
            page_size,
            num_slots,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(logical, want_logical, rtol=0, atol=0)
    else:
        flashinfer.qsa_route_from_logical(
            want_logical,
            token_to_req,
            table,
            route,
            mask,
            rows,
            page_size,
            num_slots,
        )
        torch.cuda.synchronize()

    torch.testing.assert_close(route, want_route, rtol=0, atol=0)
    torch.testing.assert_close(mask, want_mask, rtol=0, atol=0)
    # The last row really was written: its first mask byte carries the columns
    # the expansion filled. The byte after it covers columns 8 to 10, which this
    # selection leaves empty, so it is zero for every row.
    nbytes = -(-width // 8)
    assert int(mask[(rows - 1) * nbytes]) != 0, mask[(rows - 1) * nbytes].item()
