"""Tests for :func:`flashinfer.qsa_pre_indexer`."""

import pytest
import torch

import flashinfer

DTYPES = [torch.bfloat16, torch.float16]


def _skip_unless_cuda():
    if not torch.cuda.is_available():
        pytest.skip("qsa_pre_indexer runs on CUDA")


def _pair_major_table(max_pos, head_dim, device, dtype, seed=0):
    """The rotary table the kernel reads: cosine and sine interleaved per pair."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.rand(max_pos, head_dim // 4, generator=gen) * 6.0
    table = torch.zeros(max_pos, head_dim // 2)
    table[:, 0::2] = torch.cos(theta)
    table[:, 1::2] = torch.sin(theta)
    return table.to(device=device, dtype=dtype)


def _axis_of(pair, mrope_h, mrope_w, mrope):
    if not mrope:
        return 0
    mod = pair % 3
    if mod == 1 and pair <= 3 * mrope_h:
        return 1
    if mod == 2 and pair <= 3 * mrope_w:
        return 2
    return 0


def _norm_rope_row(row, weight, table, pos, mrope_h, mrope_w, mrope, eps, dtype):
    """One row through the norm and the partial rope, in float64."""
    head_dim = row.numel()
    quarter = head_dim // 4
    v = row.double()
    rrms = torch.rsqrt(v.square().mean() + eps)
    # Rounded to the cache dtype before the rotation reads it, as the kernel does.
    v = (v * rrms * (weight.double() + 1.0)).to(dtype).double()
    out = v.clone()
    for i in range(quarter):
        axis = _axis_of(i, mrope_h, mrope_w, mrope)
        p = pos[axis]
        c = table[p, 2 * i].double()
        s = table[p, 2 * i + 1].double()
        lo, hi = v[i], v[i + quarter]
        out[i] = lo * c - hi * s
        out[i + quarter] = hi * c + lo * s
    return out


def _reference(
    q,
    k,
    positions,
    table,
    q_weight,
    k_weight,
    eps,
    state_cache,
    state_slots,
    state_block_table,
    query_start_loc,
    logical_positions,
    compressed_cache,
    compressed_slots,
    work_metadata,
    compress_ratio,
    num_heads,
    head_dim,
    mrope_h,
    mrope_w,
    mrope_q,
    mrope_k,
    cache_pos,
    dtype,
):
    """What the kernel should produce, computed row by row on the host."""
    num_tokens = q.shape[0]
    state_size = state_cache.shape[1]
    comp_page = compressed_cache.shape[1]
    q_out = torch.zeros(num_tokens, num_heads, head_dim, dtype=dtype, device=q.device)
    state = state_cache.clone()
    compressed = compressed_cache.clone()

    def coords(token):
        if positions.ndim == 2:
            return [int(positions[a, token]) for a in range(3)]
        p = int(positions[token])
        return [p, p, p]

    for token in range(num_tokens):
        pos = coords(token)
        for head in range(num_heads):
            row = q[token, head * head_dim : (head + 1) * head_dim]
            out = _norm_rope_row(
                row, q_weight, table, pos, mrope_h, mrope_w, mrope_q, eps, dtype
            )
            q_out[token, head] = out.to(dtype)

    # The compression reads the ring as the previous step left it.
    history = state_cache.clone()
    for pid in range(work_metadata.shape[0]):
        request = int(work_metadata[pid, 0])
        work_in_request = int(work_metadata[pid, 1])
        if request < 0:
            continue
        start = int(query_start_loc[request])
        end = int(query_start_loc[request + 1])
        if start < 0 or end > num_tokens or end <= start:
            continue
        query_len = end - start
        chunk_end = int(logical_positions[end - 1])
        chunk_start = chunk_end - query_len + 1
        num_groups = (chunk_end + 1) // compress_ratio - chunk_start // compress_ratio
        block = int(state_block_table[request, 0])

        if work_in_request < num_groups:
            first_boundary = (
                (chunk_start + compress_ratio) // compress_ratio
            ) * compress_ratio - 1
            end_position = first_boundary + work_in_request * compress_ratio
            boundary_token = start + end_position - chunk_start
            slot = (
                int(compressed_slots[boundary_token])
                if start <= boundary_token < min(end, num_tokens)
                else -1
            )
            if 0 <= slot < compressed.shape[0] * comp_page:
                acc = torch.zeros(head_dim, dtype=torch.float64, device=q.device)
                for g in range(compress_ratio):
                    source_position = end_position - (compress_ratio - 1) + g
                    if source_position >= chunk_start:
                        source_token = start + source_position - chunk_start
                        if start <= source_token < min(end, num_tokens):
                            acc += k[source_token].double()
                    elif 0 <= block < state.shape[0]:
                        acc += history[
                            block, source_position % state_size, 0, :head_dim
                        ].double()
                acc = (acc / compress_ratio).to(dtype).double()

                first_position = end_position - (compress_ratio - 1)
                pos = [first_position] * 3
                if cache_pos:
                    first_token = start + first_position - chunk_start
                    if first_position >= chunk_start and start <= first_token < min(
                        end, num_tokens
                    ):
                        pos = coords(first_token)
                    elif first_position < chunk_start and 0 <= block < state.shape[0]:
                        tail = history[
                            block, first_position % state_size, 0, head_dim:
                        ].view(torch.int64)
                        pos = [int(tail[0]), int(tail[1]), int(tail[2])]
                    else:
                        pos = [0, 0, 0]
                out = _norm_rope_row(
                    acc, k_weight, table, pos, mrope_h, mrope_w, mrope_k, eps, dtype
                )
                compressed[slot // comp_page, slot % comp_page, 0] = out.to(dtype)

        if work_in_request == 0:
            rows = min(query_len, state_size)
            for offset in range(rows):
                token = end - rows + offset
                if not (start <= token < min(end, num_tokens)):
                    continue
                slot = int(state_slots[token])
                if not (0 <= slot < state.shape[0] * state_size):
                    continue
                state[slot // state_size, slot % state_size, 0, :head_dim] = k[token]
                if cache_pos:
                    pos = coords(token)
                    tail = state[slot // state_size, slot % state_size, 0, head_dim:]
                    tail.view(torch.int64)[:3] = torch.tensor(
                        pos, dtype=torch.int64, device=q.device
                    )
    return q_out, state, compressed


def _case(
    *,
    num_tokens,
    num_heads=4,
    head_dim=128,
    compress_ratio=4,
    state_size=8,
    comp_page=4,
    dtype=torch.bfloat16,
    mrope=False,
    mrope_k=None,
    cache_pos=False,
    num_requests=1,
    seed=0,
    history=False,
    starts=None,
):
    device = torch.device("cuda")
    torch.manual_seed(seed)
    mrope_k = mrope if mrope_k is None else mrope_k
    mrope_h, mrope_w = 11, 11
    eps = 1e-6
    max_pos = 512

    q = torch.randn(num_tokens, num_heads * head_dim, dtype=dtype, device=device)
    k = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.2
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.2
    table = _pair_major_table(max_pos, head_dim, device, dtype, seed)

    # An even split unless the caller wants a specific one; every tensor below
    # is derived from these offsets, so a caller-supplied split stays consistent
    # with the slots and the work list rather than only moving the boundaries.
    if starts is None:
        per = num_tokens // num_requests
        starts = [i * per for i in range(num_requests)] + [num_tokens]
    else:
        assert len(starts) == num_requests + 1 and starts[-1] == num_tokens
    per = max(starts[i + 1] - starts[i] for i in range(num_requests))
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=device)

    # A prior chunk sits behind this one whenever history is asked for, which is
    # what makes a group reach back into the ring.
    offset = state_size if history else 0
    logical = torch.cat(
        [
            torch.arange(offset, offset + starts[i + 1] - starts[i], dtype=torch.int64)
            for i in range(num_requests)
        ]
    ).to(device)
    positions_1d = logical.clone()
    positions = (
        torch.stack([positions_1d, positions_1d // 7, positions_1d // 13])
        if mrope
        else positions_1d
    )

    width = head_dim + (12 if cache_pos else 0)
    state_cache = torch.randn(
        num_requests, state_size, 1, width, dtype=dtype, device=device
    )
    if cache_pos:
        # Plausible coordinates for the rows a group may reach back into.
        for b in range(num_requests):
            for r in range(state_size):
                state_cache[b, r, 0, head_dim:].view(torch.int64)[:3] = torch.tensor(
                    [r, r // 7, r // 13], dtype=torch.int64, device=device
                )
    state_block_table = torch.arange(
        num_requests, dtype=torch.int32, device=device
    ).reshape(num_requests, 1)

    token_to_req = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    for r in range(num_requests):
        token_to_req[starts[r] : starts[r + 1]] = r

    state_slots = token_to_req * state_size + logical % state_size

    # Each request owns its own run of compressed slots, so two of them never
    # write the same row and the order they run in cannot matter.
    slots_per_request = (offset + per) // compress_ratio + 1
    comp_blocks = max(
        1, (num_requests * slots_per_request + comp_page - 1) // comp_page
    )
    compressed_cache = torch.zeros(
        comp_blocks, comp_page, 1, head_dim, dtype=dtype, device=device
    )
    compressed_slots = (
        token_to_req * slots_per_request + logical // compress_ratio
    ).clamp(max=comp_blocks * comp_page - 1)

    work = []
    for r in range(num_requests):
        length = starts[r + 1] - starts[r]
        if length == 0:
            # The scheduler still emits a group for a request it gave nothing
            # to; there is no last token to read the chunk's end from.
            work.append([r, 0])
            continue
        chunk_end = int(logical[starts[r + 1] - 1])
        chunk_start = chunk_end - length + 1
        groups = (chunk_end + 1) // compress_ratio - chunk_start // compress_ratio
        for g in range(max(groups, 1)):
            work.append([r, g])
    work_metadata = torch.tensor(work, dtype=torch.int32, device=device)

    return dict(
        q=q,
        k=k,
        positions=positions,
        table=table,
        q_weight=q_weight,
        k_weight=k_weight,
        eps=eps,
        state_cache=state_cache,
        state_slots=state_slots,
        state_block_table=state_block_table,
        query_start_loc=query_start_loc,
        logical_positions=logical,
        compressed_cache=compressed_cache,
        compressed_slots=compressed_slots,
        work_metadata=work_metadata,
        compress_ratio=compress_ratio,
        num_heads=num_heads,
        head_dim=head_dim,
        mrope_h=mrope_h,
        mrope_w=mrope_w,
        mrope_q=mrope,
        mrope_k=mrope_k,
        cache_pos=cache_pos,
        dtype=dtype,
    )


def _run(case):
    q_out = torch.zeros(
        case["q"].shape[0],
        case["num_heads"],
        case["head_dim"],
        dtype=case["dtype"],
        device=case["q"].device,
    )
    state = case["state_cache"].clone()
    compressed = case["compressed_cache"].clone()
    flashinfer.qsa_pre_indexer(
        case["q"],
        case["k"],
        case["positions"],
        case["table"],
        case["q_weight"],
        case["k_weight"],
        case["eps"],
        q_out,
        state,
        case["state_slots"],
        case["state_block_table"],
        case["query_start_loc"],
        case["logical_positions"],
        compressed,
        case["compressed_slots"],
        case["work_metadata"],
        case["compress_ratio"],
        mrope_h=case["mrope_h"],
        mrope_w=case["mrope_w"],
        is_k_mrope=case["mrope_k"],
        cache_has_rope_pos=case["cache_pos"],
    )
    return q_out, state, compressed


def _assert_matches(case):
    q_out, state, compressed = _run(case)
    want_q, want_state, want_compressed = _reference(**case)
    tol = dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(q_out.float(), want_q.float(), **tol)
    torch.testing.assert_close(state.float(), want_state.float(), **tol)
    torch.testing.assert_close(compressed.float(), want_compressed.float(), **tol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("num_heads", [1, 4, 8])
def test_query_path(dtype, head_dim, num_heads):
    _skip_unless_cuda()
    _assert_matches(
        _case(num_tokens=16, num_heads=num_heads, head_dim=head_dim, dtype=dtype)
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 7, 8, 9, 33, 64, 129])
def test_token_counts(num_tokens):
    """Odd counts land on the tail warp, which carries the validity predicate."""
    _skip_unless_cuda()
    _assert_matches(_case(num_tokens=num_tokens))


@pytest.mark.parametrize("compress_ratio", [2, 4, 8])
@pytest.mark.parametrize("history", [False, True])
def test_compression(compress_ratio, history):
    _skip_unless_cuda()
    _assert_matches(
        _case(num_tokens=32, compress_ratio=compress_ratio, history=history)
    )


@pytest.mark.parametrize("cache_pos", [False, True])
def test_mrope(cache_pos):
    """Three-axis positions, and the coordinates a group reads out of the ring."""
    _skip_unless_cuda()
    _assert_matches(_case(num_tokens=32, mrope=True, cache_pos=cache_pos, history=True))


def test_key_mrope_with_one_axis_positions():
    """A three-axis key over a single-axis position tensor is a supported pair."""
    _skip_unless_cuda()
    _assert_matches(_case(num_tokens=16, mrope=False, mrope_k=True))


@pytest.mark.parametrize("num_requests", [2, 4])
def test_multiple_requests(num_requests):
    _skip_unless_cuda()
    _assert_matches(_case(num_tokens=32, num_requests=num_requests, history=True))


def test_non_power_of_two_divisors():
    """The general path: no shift stands in for these divisions."""
    _skip_unless_cuda()
    _assert_matches(_case(num_tokens=24, compress_ratio=3, state_size=6, comp_page=3))


def test_dead_work_item_is_skipped():
    _skip_unless_cuda()
    case = _case(num_tokens=16)
    case["work_metadata"][0, 0] = -1
    _assert_matches(case)


def test_no_tokens_is_a_no_op():
    _skip_unless_cuda()
    case = _case(num_tokens=8)
    empty = case["q"][:0]
    before = case["compressed_cache"].clone()
    flashinfer.qsa_pre_indexer(
        empty,
        case["k"][:0],
        case["positions"][:0]
        if case["positions"].ndim == 1
        else case["positions"][:, :0],
        case["table"],
        case["q_weight"],
        case["k_weight"],
        case["eps"],
        torch.zeros(
            0,
            case["num_heads"],
            case["head_dim"],
            dtype=case["dtype"],
            device=empty.device,
        ),
        case["state_cache"],
        case["state_slots"][:0],
        case["state_block_table"],
        case["query_start_loc"],
        case["logical_positions"][:0],
        case["compressed_cache"],
        case["compressed_slots"][:0],
        case["work_metadata"],
        case["compress_ratio"],
    )
    torch.testing.assert_close(case["compressed_cache"], before)


def test_rejects_three_axis_positions_with_one_axis_key():
    _skip_unless_cuda()
    case = _case(num_tokens=8, mrope=True)
    case["mrope_k"] = False
    with pytest.raises(ValueError, match="three-axis"):
        _run(case)


def test_rejects_unsupported_head_dim():
    _skip_unless_cuda()
    case = _case(num_tokens=8, head_dim=128)
    case["head_dim"] = 64
    case["num_heads"] = 1
    with pytest.raises(ValueError, match="head_dim"):
        _run(case)


def _expect_rejected(case, match):
    with pytest.raises(Exception, match=match):
        _run(case)


def test_rejects_a_rotary_table_of_the_wrong_width():
    """A row of any other width silently shifts every position."""
    _skip_unless_cuda()
    case = _case(num_tokens=8)
    case["table"] = case["table"][:, : case["head_dim"] // 4].contiguous()
    _expect_rejected(case, "pair-major")


def test_rejects_a_second_kv_head():
    _skip_unless_cuda()
    case = _case(num_tokens=8)
    case["state_cache"] = case["state_cache"].repeat(1, 1, 2, 1)
    _expect_rejected(case, "one KV head")


def test_rejects_a_ring_too_narrow_for_its_coordinates():
    """With cache_has_rope_pos the row is followed by three int64."""
    _skip_unless_cuda()
    case = _case(num_tokens=8, cache_pos=True)
    case["state_cache"] = case["state_cache"][..., : case["head_dim"] + 4].contiguous()
    _expect_rejected(case, "coordinates after it")


def test_rejects_a_short_query_start_loc():
    _skip_unless_cuda()
    case = _case(num_tokens=8)
    case["query_start_loc"] = case["query_start_loc"][:-1].contiguous()
    _expect_rejected(case, "one entry past")


def _constant_row_out(fill, num_tokens=8, head_dim=128, dtype=torch.bfloat16):
    """A row of one value, normalised with a zero affine, and what came out."""
    case = _case(num_tokens=num_tokens, num_heads=1, head_dim=head_dim, dtype=dtype)
    case["q"] = torch.full_like(case["q"], fill)
    case["q_weight"] = torch.zeros_like(case["q_weight"])
    q_out, _, _ = _run(case)
    return q_out[0, 0].float()


@pytest.mark.parametrize("fill", [1e-2, 1.0, 1e6, 1e12, 1e18])
def test_a_constant_row_normalises_to_one(fill):
    """Across the range the norm is exact: every element is +/-1 before it turns.

    The rotation mixes an element with its partner, so the largest an element
    can come out is sqrt(2); what matters is that none of it is lost.
    """
    _skip_unless_cuda()
    out = _constant_row_out(fill)
    assert torch.isfinite(out).all()
    magnitude = out.abs().max().item()
    assert 0.9 < magnitude < 1.5, magnitude


def test_a_row_that_overflows_the_sum_of_squares_comes_out_zero():
    """Documented, not desired: the sum of squares is accumulated in float32.

    A row whose RMS passes about 1.6e18 squares to infinity there, the
    reciprocal square root of which is zero, so the row is lost rather than
    normalised. Real activations are nowhere near this; the point of pinning it
    is that the failure is silent, and a caller that does get here should know
    it reads zeros rather than a normalised row.
    """
    _skip_unless_cuda()
    assert _constant_row_out(1e18).abs().max().item() > 0.9
    assert _constant_row_out(1e19).abs().max().item() == 0.0


def test_eps_bounds_the_gain_on_a_vanishing_row():
    """Below an RMS of about 1e-3 the epsilon is the whole denominator."""
    _skip_unless_cuda()
    out = _constant_row_out(1e-20)
    # rsqrt(eps) is 1e3, so the row is scaled by that rather than to one.
    assert 0.9e-17 < out.abs().max().item() < 2e-17


@pytest.mark.parametrize("fill", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_input_propagates(fill):
    """It arrives as a NaN rather than as a plausible number."""
    _skip_unless_cuda()
    assert torch.isnan(_constant_row_out(fill)).all()


def test_an_affine_of_minus_one_zeroes_the_row():
    """The weight is the offset from one, so -1 is a zero gain."""
    _skip_unless_cuda()
    case = _case(num_tokens=8, num_heads=1)
    case["q_weight"] = torch.full_like(case["q_weight"], -1.0)
    q_out, _, _ = _run(case)
    assert q_out.abs().max().item() == 0.0


def test_runs_on_a_non_default_stream():
    """The launch takes the caller's stream, not whatever stream zero is doing."""
    _skip_unless_cuda()
    case = _case(num_tokens=32, history=True)
    expected = _run(case)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        got = _run(case)
    stream.synchronize()
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b)


def test_replays_inside_a_cuda_graph():
    """Captured and replayed, with the inputs changed between replays.

    A launch that allocated, synchronised or read a host value would either
    fail to capture or replay the first call's answer for the second.
    """
    _skip_unless_cuda()
    case = _case(num_tokens=32, history=True)
    q_out = torch.zeros(
        case["q"].shape[0],
        case["num_heads"],
        case["head_dim"],
        dtype=case["dtype"],
        device=case["q"].device,
    )
    state = case["state_cache"].clone()
    compressed = case["compressed_cache"].clone()

    def launch():
        flashinfer.qsa_pre_indexer(
            case["q"],
            case["k"],
            case["positions"],
            case["table"],
            case["q_weight"],
            case["k_weight"],
            case["eps"],
            q_out,
            state,
            case["state_slots"],
            case["state_block_table"],
            case["query_start_loc"],
            case["logical_positions"],
            compressed,
            case["compressed_slots"],
            case["work_metadata"],
            case["compress_ratio"],
            mrope_h=case["mrope_h"],
            mrope_w=case["mrope_w"],
            is_k_mrope=case["mrope_k"],
            cache_has_rope_pos=case["cache_pos"],
        )

    # Warm up on a side stream, which is what capture requires.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            launch()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    for _ in range(2):
        state.copy_(case["state_cache"])
        compressed.copy_(case["compressed_cache"])
        graph.replay()
        torch.cuda.synchronize()
        want_q, want_state, want_compressed = _reference(**case)
        tol = dict(rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(q_out.float(), want_q.float(), **tol)
        torch.testing.assert_close(state.float(), want_state.float(), **tol)
        torch.testing.assert_close(compressed.float(), want_compressed.float(), **tol)

    # A different query through the same graph has to give a different answer.
    case["q"].normal_()
    state.copy_(case["state_cache"])
    compressed.copy_(case["compressed_cache"])
    graph.replay()
    torch.cuda.synchronize()
    want_q, _, _ = _reference(**case)
    torch.testing.assert_close(q_out.float(), want_q.float(), rtol=2e-2, atol=2e-2)


def test_a_request_with_no_tokens_is_skipped():
    """A work item may name a request the step gave nothing to.

    Its end is also its start, so there is no last token to read the chunk's
    end from; the group loop has to stop before that read rather than index
    behind the tensor.
    """
    _skip_unless_cuda()
    # The whole case is built from this split, so the slots and the work list
    # describe the collapsed request rather than an even one; the work list
    # still names it, which is what reaches the guard.
    case = _case(num_tokens=16, num_requests=2, starts=[0, 0, 16])
    assert (case["work_metadata"][:, 0] == 0).any()
    _assert_matches(case)


@pytest.mark.parametrize("bad", ["past_the_end", "reversed", "negative", "wraps"])
def test_a_request_whose_offsets_leave_the_token_axis_is_skipped(bad):
    """Only the length of query_start_loc is checked on the host, so a value
    that runs off the token axis would put logical_positions[end - 1] outside
    the tensor."""
    _skip_unless_cuda()
    case = _case(num_tokens=16, num_requests=1)
    offsets = case["query_start_loc"].clone()
    if bad == "past_the_end":
        offsets[1] = 64
    elif bad == "reversed":
        offsets[0] = 8
        offsets[1] = 4
    elif bad == "wraps":
        # The difference of these two overflows int32 and comes out as 1, so a
        # guard that subtracts before it bounds the ends lets them through.
        offsets[0] = 2147483647
        offsets[1] = -2147483648
    else:
        offsets[0] = -8
    case["query_start_loc"] = offsets
    _assert_matches(case)


@pytest.mark.parametrize("short", ["k", "positions"])
def test_rejects_a_short_token_axis(short):
    """Both are walked by a token stride, so a short axis reads past its end."""
    _skip_unless_cuda()
    case = _case(num_tokens=16)
    if short == "k":
        case["k"] = case["k"][:-1].contiguous()
    else:
        case["positions"] = case["positions"][:-1].contiguous()
    with pytest.raises(Exception, match="per token"):
        _run(case)


def test_a_work_item_naming_a_request_that_does_not_exist_is_skipped():
    """The work list is the caller's, and the request it names indexes both
    ends of a prefix-sum entry. One past the last request reads off the table."""
    _skip_unless_cuda()
    case = _case(num_tokens=16, num_requests=1)
    case["work_metadata"] = case["work_metadata"].clone()
    case["work_metadata"][0, 0] = 1  # only request 0 exists
    q_out, _, _ = _run(case)
    torch.cuda.synchronize()
    assert torch.isfinite(q_out.float()).all()


def test_rejects_an_empty_ring():
    """The ring is indexed modulo its size, so an empty one addresses off the
    front of the cache rather than reading nothing."""
    _skip_unless_cuda()
    case = _case(num_tokens=16)
    case["state_cache"] = case["state_cache"][:, :0].contiguous()
    with pytest.raises(Exception, match="at least one row"):
        _run(case)


def test_rejects_an_empty_compressed_page():
    _skip_unless_cuda()
    case = _case(num_tokens=16)
    case["compressed_cache"] = case["compressed_cache"][:, :0].contiguous()
    with pytest.raises(Exception, match="at least one row"):
        _run(case)


@pytest.mark.parametrize("state_size,compress_ratio", [(7, 4), (6, 4), (5, 2)])
def test_a_group_reaching_into_an_unaligned_ring(state_size, compress_ratio):
    """A ring whose size is not a multiple of the ratio makes the wrap land
    inside a group, which is the case the aligned sizes never produce."""
    _skip_unless_cuda()
    _assert_matches(
        _case(
            num_tokens=32,
            state_size=state_size,
            compress_ratio=compress_ratio,
            history=True,
            cache_pos=True,
        )
    )
