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

COMPRESS_RATIO = 4
TOKEN_TOPK = 32
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
ROUTE_WIDTH = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
NUM_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 16


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def _batch(device, rows, num_requests, pages_per_request, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    pages = num_requests * pages_per_request
    page_table = (
        torch.randperm(pages, device=device, generator=generator)
        .reshape(num_requests, pages_per_request)
        .contiguous()
        .to(torch.int32)
    )
    k_compressed = torch.randn(
        pages,
        PAGE_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q = torch.randn(
        rows,
        NUM_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    per_request = max(1, rows // num_requests)
    token_to_req = (
        (torch.arange(rows, device=device, dtype=torch.int32) // per_request)
        .clamp(max=num_requests - 1)
        .contiguous()
    )
    # Positions inside each request, and lengths that cover them.
    positions = torch.arange(rows, device=device, dtype=torch.int32) % (
        pages_per_request * PAGE_SIZE * COMPRESS_RATIO
    )
    lengths = torch.full(
        (num_requests,),
        pages_per_request * PAGE_SIZE,
        dtype=torch.int32,
        device=device,
    )
    return q, k_compressed, page_table, token_to_req, positions, lengths


def _oracle(q, k_compressed, page_table, token_to_req, positions, lengths, columns):
    """Score, select and expand in plain torch.

    The score is the one the kernel documents -- the summed positive dot over
    heads, divided by sqrt(head_dim) -- and the selection is the top blocks of
    it. The expansion writes each selected block's tokens, then the seen tail of
    the block the query sits in, and ``-1`` for the rest.
    """
    rows = q.size(0)
    device = q.device
    route = torch.full((rows, ROUTE_WIDTH), -1, dtype=torch.int32, device=device)
    flat_keys = k_compressed.reshape(-1, HEAD_DIM).to(torch.float64)
    for row in range(rows):
        request = int(token_to_req[row])
        position = int(positions[row])
        # What the query can see: the blocks it has entirely behind it, capped
        # by the request's length and by the column width. The scorer scores
        # exactly these and reports the count, and the top-k ranks only them.
        past_blocks = min(
            (position + 1) // COMPRESS_RATIO, int(lengths[request]) // COMPRESS_RATIO
        )
        visible = min(past_blocks, columns)
        if visible > 0:
            # Gather this request's compressed columns through its page table.
            entries = []
            for column in range(visible):
                page = int(page_table[request, column // PAGE_SIZE])
                entries.append(page * PAGE_SIZE + column % PAGE_SIZE)
            keys = flat_keys[entries]
            query = q[row].to(torch.float64)
            scores = (keys @ query.T).clamp(min=0).sum(dim=1) / (HEAD_DIM**0.5)
            order = sorted(range(visible), key=lambda c: (-scores[c].item(), c))
            for rank, block in enumerate(order[:BLOCK_TOPK]):
                if block >= past_blocks:
                    continue  # a block the query has not passed is dropped whole
                for offset in range(COMPRESS_RATIO):
                    route[row, rank * COMPRESS_RATIO + offset] = (
                        block * COMPRESS_RATIO + offset
                    )
        # The tail is the query's own block, seen or not seen has nothing to do
        # with the selection above.
        tail_start = (position + 1) // COMPRESS_RATIO * COMPRESS_RATIO
        for index, token in enumerate(range(tail_start, position + 1)):
            route[row, BLOCK_TOPK * COMPRESS_RATIO + index] = token
    return route


def _selection(device, rows, columns, **kwargs):
    return flashinfer.QSASelection(
        max_rows=rows,
        max_columns=columns,
        compress_ratio=COMPRESS_RATIO,
        token_topk=TOKEN_TOPK,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        device=device,
        **kwargs,
    )


def _run(selection, batch, rows, columns, workspace=None):
    q, k_compressed, page_table, token_to_req, positions, lengths = batch
    device = q.device
    if workspace is None:
        workspace = torch.zeros(
            selection.workspace_size(), dtype=torch.uint8, device=device
        )
    route = torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)
    selection.run(
        q,
        k_compressed,
        page_table,
        token_to_req,
        positions,
        lengths,
        out_route=route,
        workspace=workspace,
    )
    return route


EXPANDED = BLOCK_TOPK * COMPRESS_RATIO


def _assert_same_route(got, want):
    """Same selection and same tail, with the ranks left free.

    Which rank a selected block lands on is not a contract: the top-k does not
    promise an order -- the CUB backend says outright that it returns unsorted
    results -- and the attention that reads the route treats it as a set with a
    mask. So the expanded region is compared as the set of tokens it carries,
    and the causal tail, whose column positions do mean something, exactly.
    """
    assert got.shape == want.shape
    for row in range(got.size(0)):
        got_blocks = got[row, :EXPANDED]
        want_blocks = want[row, :EXPANDED]
        torch.testing.assert_close(
            got_blocks.sort().values, want_blocks.sort().values, rtol=0, atol=0
        )
        torch.testing.assert_close(
            got[row, EXPANDED:], want[row, EXPANDED:], rtol=0, atol=0
        )


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_route_matches_a_plain_torch_oracle(device, seed):
    """Scores, selection and expansion, end to end, against torch."""
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=seed)
    selection = _selection(device, rows, columns)
    route = _run(selection, batch, rows, columns)

    q, k_compressed, page_table, token_to_req, positions, lengths = batch
    expected = _oracle(
        q, k_compressed, page_table, token_to_req, positions, lengths, columns
    )
    _assert_same_route(route, expected)


def test_the_padding_stays_minus_one(device):
    """Every position no token reaches is ``-1``, and the width is fixed."""
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=2)
    selection = _selection(device, rows, columns)
    route = _run(selection, batch, rows, columns)

    assert route.shape == (rows, ROUTE_WIDTH)
    assert route.dtype == torch.int32
    # Nothing between -1 and zero, and nothing above the request's length.
    assert bool(((route == -1) | (route >= 0)).all())


def test_running_allocates_nothing(device):
    """The whole selection, with the caller holding every buffer."""
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=3)
    selection = _selection(device, rows, columns)
    workspace = torch.zeros(
        selection.workspace_size(), dtype=torch.uint8, device=device
    )
    _run(selection, batch, rows, columns, workspace=workspace)
    torch.cuda.synchronize()

    route = torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)
    q, k_compressed, page_table, token_to_req, positions, lengths = batch
    before = torch.cuda.memory_allocated()
    selection.run(
        q,
        k_compressed,
        page_table,
        token_to_req,
        positions,
        lengths,
        out_route=route,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


def test_a_batch_wider_than_the_score_budget_is_chunked(device):
    """The chunk is the budget's doing, and the answer does not change."""
    rows, columns = 16, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=4)
    whole = _selection(device, rows, columns)
    chunked = _selection(device, rows, columns, score_budget_bytes=4 * columns * 4)
    assert chunked.rows_per_chunk < whole.rows_per_chunk

    torch.testing.assert_close(
        _run(chunked, batch, rows, columns),
        _run(whole, batch, rows, columns),
        rtol=0,
        atol=0,
    )


def test_two_selections_run_side_by_side_on_one_arena(device):
    """Different slices, two streams, and neither disturbs the other."""
    rows, columns = 8, 64
    first = _batch(device, rows, num_requests=2, pages_per_request=4, seed=5)
    second = _batch(device, rows, num_requests=2, pages_per_request=4, seed=6)
    selections = [_selection(device, rows, columns) for _ in range(2)]
    size = selections[0].workspace_size()
    arena = torch.zeros(2 * size, dtype=torch.uint8, device=device)
    slices = [arena[:size], arena[size:]]

    alone = [
        _run(selections[0], first, rows, columns),
        _run(selections[1], second, rows, columns),
    ]

    routes = [
        torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)
        for _ in range(2)
    ]
    streams = [torch.cuda.Stream() for _ in range(2)]
    for stream, selection, batch, route, slot in zip(
        streams, selections, (first, second), routes, slices, strict=True
    ):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            q, k_compressed, page_table, token_to_req, positions, lengths = batch
            selection.run(
                q,
                k_compressed,
                page_table,
                token_to_req,
                positions,
                lengths,
                out_route=route,
                workspace=slot,
            )
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    for got, want in zip(routes, alone, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_two_graphs_replay_from_their_own_slices(device):
    """Captured separately, replayed together, each with its own scratch."""
    rows, columns = 8, 64
    batches = [
        _batch(device, rows, num_requests=2, pages_per_request=4, seed=seed)
        for seed in (7, 8)
    ]
    selections = [_selection(device, rows, columns) for _ in range(2)]
    size = selections[0].workspace_size()
    arena = torch.zeros(2 * size, dtype=torch.uint8, device=device)
    routes = [
        torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)
        for _ in range(2)
    ]

    def call(index):
        q, k_compressed, page_table, token_to_req, positions, lengths = batches[index]
        selections[index].run(
            q,
            k_compressed,
            page_table,
            token_to_req,
            positions,
            lengths,
            out_route=routes[index],
            workspace=arena[index * size : (index + 1) * size],
        )

    expected = []
    for index in range(2):
        call(index)
        torch.cuda.synchronize()
        expected.append(routes[index].clone())

    graphs = []
    for index in range(2):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                call(index)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(index)
        graphs.append(graph)

    for route in routes:
        route.zero_()
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(routes, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_a_workspace_that_is_too_small_or_misplaced_is_refused(device):
    """Checked before anything runs, not by a fault inside a kernel."""
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=9)
    selection = _selection(device, rows, columns)
    size = selection.workspace_size()
    q, k_compressed, page_table, token_to_req, positions, lengths = batch
    route = torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)

    def call(workspace):
        selection.run(
            q,
            k_compressed,
            page_table,
            token_to_req,
            positions,
            lengths,
            out_route=route,
            workspace=workspace,
        )

    with pytest.raises(ValueError, match="workspace needs"):
        call(torch.zeros(size - 1, dtype=torch.uint8, device=device))
    with pytest.raises(ValueError, match="aligned"):
        call(torch.zeros(size + 16, dtype=torch.uint8, device=device)[1:])
    with pytest.raises(ValueError, match="contiguous uint8"):
        call(torch.zeros(size, dtype=torch.int32, device=device))


def test_a_route_of_the_wrong_shape_is_refused(device):
    """The route is indices and nothing else -- no trailing count column."""
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=10)
    selection = _selection(device, rows, columns)
    workspace = torch.zeros(
        selection.workspace_size(), dtype=torch.uint8, device=device
    )
    q, k_compressed, page_table, token_to_req, positions, lengths = batch

    with pytest.raises(ValueError, match="out_route must be"):
        selection.run(
            q,
            k_compressed,
            page_table,
            token_to_req,
            positions,
            lengths,
            out_route=torch.empty(
                rows, ROUTE_WIDTH + 1, dtype=torch.int32, device=device
            ),
            workspace=workspace,
        )


def test_running_makes_no_tensor_at_all(device, monkeypatch):
    """Not even a CPU one.

    ``torch.cuda.memory_allocated`` would not notice a host tensor made to ask
    a dtype its size, so the factories themselves are watched.
    """
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=12)
    selection = _selection(device, rows, columns)
    workspace = torch.zeros(
        selection.workspace_size(), dtype=torch.uint8, device=device
    )
    route = torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)
    q, k_compressed, page_table, token_to_req, positions, lengths = batch

    def refuse(name):
        def factory(*args, **kwargs):
            raise AssertionError(f"run() called torch.{name}")

        return factory

    for name in ("empty", "zeros", "full", "empty_strided"):
        monkeypatch.setattr(torch, name, refuse(name))
    selection.run(
        q,
        k_compressed,
        page_table,
        token_to_req,
        positions,
        lengths,
        out_route=route,
        workspace=workspace,
    )
    torch.cuda.synchronize()


def test_a_workspace_full_of_rubbish_still_gives_the_right_route(device):
    """The API owns the scratch's initial state, not the caller.

    The radix top-k reads its counters before writing them on the first round.
    A caller who allocated with ``torch.empty`` would otherwise get a different
    answer than one who used ``torch.zeros``.
    """
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=13)
    selection = _selection(device, rows, columns)
    clean = torch.zeros(selection.workspace_size(), dtype=torch.uint8, device=device)
    expected = _run(selection, batch, rows, columns, workspace=clean)

    poisoned = torch.full(
        (selection.workspace_size(),), 0xA5, dtype=torch.uint8, device=device
    )
    got = _run(selection, batch, rows, columns, workspace=poisoned)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    # And reused across calls, which is what a server does.
    again = _run(selection, batch, rows, columns, workspace=poisoned)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)


def test_two_graphs_replay_on_two_streams(device):
    """Each graph on its own stream, each with its own slice."""
    rows, columns = 8, 64
    batches = [
        _batch(device, rows, num_requests=2, pages_per_request=4, seed=seed)
        for seed in (14, 15)
    ]
    selections = [_selection(device, rows, columns) for _ in range(2)]
    size = selections[0].workspace_size()
    arena = torch.zeros(2 * size, dtype=torch.uint8, device=device)
    routes = [
        torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)
        for _ in range(2)
    ]

    def call(index):
        q, k_compressed, page_table, token_to_req, positions, lengths = batches[index]
        selections[index].run(
            q,
            k_compressed,
            page_table,
            token_to_req,
            positions,
            lengths,
            out_route=routes[index],
            workspace=arena[index * size : (index + 1) * size],
        )

    oracles = []
    for index in range(2):
        call(index)
        torch.cuda.synchronize()
        q, k_compressed, page_table, token_to_req, positions, lengths = batches[index]
        oracles.append(
            _oracle(
                q, k_compressed, page_table, token_to_req, positions, lengths, columns
            )
        )

    graphs = []
    for index in range(2):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                call(index)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(index)
        graphs.append(graph)

    for route in routes:
        route.zero_()
    streams = [torch.cuda.Stream() for _ in range(2)]
    for stream, graph in zip(streams, graphs, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    for route, oracle in zip(routes, oracles, strict=True):
        _assert_same_route(route, oracle)


@pytest.mark.parametrize(
    "position",
    [0, COMPRESS_RATIO - 1, COMPRESS_RATIO, 2 * COMPRESS_RATIO - 1, 37, 63],
)
def test_the_positions_around_a_block_boundary(device, position):
    """Where the causal tail starts and how many blocks are behind it."""
    rows, columns = 4, 64
    batch = _batch(device, rows, num_requests=1, pages_per_request=4, seed=16)
    q, k_compressed, page_table, token_to_req, _positions, lengths = batch
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), position, dtype=torch.int32, device=device)
    lengths = torch.full((1,), 64, dtype=torch.int32, device=device)
    batch = (q, k_compressed, page_table, token_to_req, positions, lengths)

    selection = _selection(device, rows, columns)
    route = _run(selection, batch, rows, columns)
    expected = _oracle(
        q, k_compressed, page_table, token_to_req, positions, lengths, columns
    )
    _assert_same_route(route, expected)


@pytest.mark.parametrize("length", [0, 4, 8, 32, 64])
def test_the_visible_count_from_none_to_more_than_k(device, length):
    """A sequence with nothing behind the query, and one with plenty."""
    rows, columns = 4, 64
    batch = _batch(device, rows, num_requests=1, pages_per_request=4, seed=17)
    q, k_compressed, page_table, _t2r, _pos, _lens = batch
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), 63, dtype=torch.int32, device=device)
    lengths = torch.full((1,), length, dtype=torch.int32, device=device)
    batch = (q, k_compressed, page_table, token_to_req, positions, lengths)

    selection = _selection(device, rows, columns)
    route = _run(selection, batch, rows, columns)
    expected = _oracle(
        q, k_compressed, page_table, token_to_req, positions, lengths, columns
    )
    _assert_same_route(route, expected)


def test_the_same_input_gives_the_same_route_every_time(device):
    """Deterministic, element for element, not merely the same set."""
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=18)
    selection = _selection(device, rows, columns)
    workspace = torch.zeros(
        selection.workspace_size(), dtype=torch.uint8, device=device
    )
    first = _run(selection, batch, rows, columns, workspace=workspace).clone()
    for _ in range(4):
        again = _run(selection, batch, rows, columns, workspace=workspace)
        torch.testing.assert_close(again, first, rtol=0, atol=0)


def test_an_unmapped_page_is_not_read(device):
    """A block table entry of ``-1`` names no page.

    The scorer is told how much of the table is real through the sequence
    length; a page past it must not be reached whatever the entry says.
    """
    rows, columns = 4, 64
    batch = _batch(device, rows, num_requests=1, pages_per_request=4, seed=19)
    q, k_compressed, page_table, _t2r, _pos, _lens = batch
    token_to_req = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.full((rows,), 63, dtype=torch.int32, device=device)
    # Two pages of real cache; the rest of the table is unmapped.
    lengths = torch.full((1,), 2 * PAGE_SIZE, dtype=torch.int32, device=device)
    table = page_table.clone()
    table[0, 2:] = -1
    batch = (q, k_compressed, table, token_to_req, positions, lengths)

    selection = _selection(device, rows, columns)
    route = _run(selection, batch, rows, columns)
    expected = _oracle(
        q, k_compressed, table, token_to_req, positions, lengths, columns
    )
    _assert_same_route(route, expected)


@pytest.mark.parametrize("columns", [2048, 1 << 16])
def test_planning_allocates_nothing(device, columns, monkeypatch):
    """Sizing the workspace must not cost anything at all.

    Both backends answer from the shape they were prepared for, so planning
    makes no tensor -- not a score buffer, and not a row of one. The factories
    are watched rather than the allocator's counters, which would miss a host
    tensor and could hide behind a cached block.
    """
    warm = _selection(device, 8, 64)
    assert warm.workspace_size() > 0
    torch.cuda.synchronize()

    def refuse(name):
        def factory(*args, **kwargs):
            raise AssertionError(f"planning called torch.{name}")

        return factory

    for name in ("empty", "zeros", "full", "empty_strided", "randn"):
        monkeypatch.setattr(torch, name, refuse(name))
    selection = _selection(device, 2048, columns)
    assert selection.workspace_size() > 0


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"num_heads": 17}, "query heads"),
        ({"head_dim": 96}, "head dimensions"),
        ({"token_topk": 0}, "token_topk"),
        ({"compress_ratio": 0}, "compress_ratio"),
        ({"score_budget_bytes": 0}, "score_budget_bytes"),
        ({"token_topk": 30}, "whole number of blocks"),
    ],
)
def test_a_shape_the_scorer_has_no_kernel_for_is_refused(device, overrides, message):
    """Refused while planning, not by a kernel that was never built."""
    arguments = {
        "max_rows": 8,
        "max_columns": 64,
        "compress_ratio": COMPRESS_RATIO,
        "token_topk": TOKEN_TOPK,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "device": device,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        flashinfer.QSASelection(**arguments)


def test_the_index_side_has_to_be_int32(device):
    """The visible counts are int32, and the scorer wants them to match."""
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=20)
    q, k_compressed, page_table, token_to_req, positions, lengths = batch
    selection = _selection(device, rows, columns)
    workspace = torch.zeros(
        selection.workspace_size(), dtype=torch.uint8, device=device
    )
    route = torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="int32"):
        selection.run(
            q,
            k_compressed,
            page_table.to(torch.int64),
            token_to_req.to(torch.int64),
            positions.to(torch.int64),
            lengths.to(torch.int64),
            out_route=route,
            workspace=workspace,
        )


def test_a_float32_query_is_refused(device):
    """The scorer serves half and bfloat16."""
    rows, columns = 8, 64
    batch = _batch(device, rows, num_requests=2, pages_per_request=4, seed=21)
    q, k_compressed, page_table, token_to_req, positions, lengths = batch
    selection = _selection(device, rows, columns)
    workspace = torch.zeros(
        selection.workspace_size(), dtype=torch.uint8, device=device
    )
    route = torch.empty(rows, ROUTE_WIDTH, dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="q must be one of"):
        selection.run(
            q.float(),
            k_compressed.float(),
            page_table,
            token_to_req,
            positions,
            lengths,
            out_route=route,
            workspace=workspace,
        )
