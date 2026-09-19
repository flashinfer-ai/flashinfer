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

import gc
import weakref

import pytest
import torch

import flashinfer

NUM_QO_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 16
PAGES = 32
NUM_SLOTS = PAGES * PAGE_SIZE
ROUTE_WIDTH = 35
BUCKETS = (4, 16)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def _arguments(**kwargs):
    arguments = {
        "max_rows": max(BUCKETS),
        "num_qo_heads": NUM_QO_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "route_width": ROUTE_WIDTH,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "kv_cache_format": "dense",
    }
    arguments.update(kwargs)
    return arguments


def _workspace(device, **kwargs):
    """Exactly what this geometry asks for, as the caller allocates it.

    Two buffers: one nobody else writes, one the caller may share with
    everything else its step runs.
    """
    arguments = _arguments(**kwargs)
    persistent_bytes, transient_bytes = flashinfer.QSAAttention.workspace_bytes(
        device=device, **arguments
    )
    persistent = torch.zeros(persistent_bytes, dtype=torch.uint8, device=device)
    transient = torch.zeros(transient_bytes, dtype=torch.uint8, device=device)
    return persistent, transient, arguments


def _attention(device, num_slots=NUM_SLOTS, **kwargs):
    persistent, transient, arguments = _workspace(device, **kwargs)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    attention.plan_cache(num_slots, PAGE_SIZE)
    # The buffers are the caller's; keep them alive as long as the views are.
    attention._test_workspace = (persistent, transient)
    return attention


def _batch(device, rows, num_requests, seed, gate=True):
    generator = torch.Generator(device=device).manual_seed(seed)
    pages_per_request = PAGES // num_requests
    block_table = (
        torch.randperm(PAGES, device=device, generator=generator)
        .reshape(num_requests, pages_per_request)
        .contiguous()
        .to(torch.int32)
    )
    k = torch.randn(
        PAGES,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        PAGES,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    q = torch.randn(
        rows,
        NUM_QO_HEADS,
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
    # A logical route with a live prefix and -1 padding, inside each request.
    live = pages_per_request * PAGE_SIZE
    route = torch.full((rows, ROUTE_WIDTH), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        count = min(live, ROUTE_WIDTH - row % 3)
        route[row, :count] = torch.randperm(live, device=device, generator=generator)[
            :count
        ].to(torch.int32)
    gates = (
        torch.randn(
            rows,
            NUM_QO_HEADS * HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        if gate
        else None
    )
    return q, k, v, route, block_table, token_to_req, gates


def _oracle(q, k, v, route, block_table, token_to_req, gates):
    """Attention over exactly the route's live entries, in float64.

    Each route entry names a logical token; the block table turns its page into
    a physical one, and an entry that names no token -- ``-1``, a page past the
    table, an unmapped page, a slot past the cache -- contributes nothing. The
    gate is applied the way the kernel applies it: the output rounded to the
    buffer's dtype, the logistic at float, one store.
    """
    return _oracle_from_values(
        q,
        k.reshape(-1, NUM_KV_HEADS, HEAD_DIM).to(torch.float64),
        v.reshape(-1, NUM_KV_HEADS, HEAD_DIM).to(torch.float64),
        route,
        block_table,
        token_to_req,
        gates,
    )


def _oracle_from_values(q, flat_k, flat_v, route, block_table, token_to_req, gates):
    """The same, over keys and values already decoded to float64 by slot."""
    rows = q.size(0)
    device = q.device
    out = torch.zeros(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float64, device=device)
    scale = HEAD_DIM**-0.5
    for row in range(rows):
        request = int(token_to_req[row])
        slots = []
        for column in range(route.size(1)):
            token = int(route[row, column])
            if token < 0:
                continue
            page = token // PAGE_SIZE
            if page >= block_table.size(1):
                continue
            physical = int(block_table[request, page])
            if physical < 0:
                continue
            slot = physical * PAGE_SIZE + token % PAGE_SIZE
            if slot >= NUM_SLOTS:
                continue
            slots.append(slot)
        if not slots:
            continue
        keys = flat_k[slots, 0]
        values = flat_v[slots, 0]
        for head in range(NUM_QO_HEADS):
            logits = (keys @ q[row, head].to(torch.float64)) * scale
            weights = torch.softmax(logits, dim=0)
            out[row, head] = weights @ values
    rounded = out.to(torch.bfloat16)
    if gates is None:
        return rounded
    gate = gates.unflatten(1, (NUM_QO_HEADS, HEAD_DIM))
    return (rounded.to(torch.float32) * torch.sigmoid(gate.to(torch.float32))).to(
        torch.bfloat16
    )


#: Below this the relative error says nothing. A bfloat16 has eight mantissa
#: bits, so one step of rounding on a value of 0.01 is already six percent; the
#: absolute bound covers those elements instead.
_RELATIVE_FLOOR = 5e-2
#: A numerical tolerance, proportional to the output, because the error is: a
#: batch whose values are twice as large rounds at twice the spacing. It is
#: not a count of ulps -- a bfloat16's spacing changes with the binade, so the
#: distance between two representable values is not a fixed fraction of
#: either. The measured error over these shapes is 0.0084 to 0.0104 of the
#: largest output; this is 0.0156.
_ABSOLUTE_PER_UNIT = 4 * 2**-8
_MAX_RELATIVE = 3e-2


def _assert_within(got, want, message=""):
    """The two bounds, with the absolute one scaled to the output's own size."""
    absolute, relative = _errors(got, want)
    peak = max(1.0, float(want.to(torch.float64).abs().max()))
    allowed = _ABSOLUTE_PER_UNIT * peak
    assert absolute <= allowed, f"max abs {absolute} over {allowed} {message}"
    assert relative <= _MAX_RELATIVE, f"max rel {relative} {message}"


def _errors(got, want):
    """Max absolute error, and max relative error where it means something."""
    a, b = got.to(torch.float64), want.to(torch.float64)
    difference = (a - b).abs()
    absolute = difference.max().item()
    significant = b.abs() >= _RELATIVE_FLOOR
    relative = (
        (difference[significant] / b.abs()[significant]).max().item()
        if bool(significant.any())
        else 0.0
    )
    return absolute, relative


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_the_output_matches_a_plain_torch_oracle(device, seed):
    """Route in, gated attention out, against torch."""
    rows = 16
    attention = _attention(device)
    q, k, v, route, block_table, token_to_req, gates = _batch(
        device, rows, 2, seed=seed
    )
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
        out=out,
    )
    expected = _oracle(q, k, v, route, block_table, token_to_req, gates)
    _assert_within(out, expected)


def test_a_batch_shorter_than_its_bucket_is_padded_and_trimmed(device):
    """The plan runs a full bucket; the caller gets its own rows."""
    attention = _attention(device)
    rows = 3
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 1, seed=2)
    assert attention.bucket_for(rows) == attention.row_buckets[0] > rows
    out = torch.full(
        (rows, NUM_QO_HEADS, HEAD_DIM), 7.0, dtype=torch.bfloat16, device=device
    )
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
        out=out,
    )
    expected = _oracle(q, k, v, route, block_table, token_to_req, gates)
    _assert_within(out, expected)


def test_running_allocates_nothing(device, monkeypatch):
    """Not a tensor, not a plan."""
    rows = 16
    attention = _attention(device)
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=3)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
        out=out,
    )
    torch.cuda.synchronize()

    def refuse(name):
        def factory(*args, **kwargs):
            raise AssertionError(f"run() called torch.{name}")

        return factory

    for name in ("empty", "zeros", "full", "empty_strided"):
        monkeypatch.setattr(torch, name, refuse(name))
    monkeypatch.setattr(
        flashinfer.sparse.BlockSparseAttentionWrapper,
        "plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("run() planned")),
    )
    before = torch.cuda.memory_allocated()
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
        out=out,
    )
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


@pytest.mark.parametrize(
    "value,error,message",
    [
        ("tensor", TypeError, "host float"),
        (True, TypeError, "real number"),
        ("two", TypeError, "real number"),
        (float("nan"), ValueError, "finite"),
        (float("inf"), ValueError, "finite"),
        (0.0, ValueError, "positive"),
        (-1.0, ValueError, "positive"),
    ],
)
def test_a_scale_that_is_not_a_positive_number_is_refused(
    device, value, error, message
):
    """Each of these reaches the kernel as a multiplier if it is not caught."""
    rows = 16
    attention = _attention(
        device, kv_data_type=torch.float8_e4m3fn, kv_cache_format="fp8_e4m3"
    )
    q, k, _v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=4)
    k8 = k.float().to(torch.float8_e4m3fn)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    scale = torch.tensor(2.0, device=device) if value == "tensor" else value
    with pytest.raises(error, match=message):
        attention.run(
            q,
            k8,
            k8,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            k_scale=scale,
            v_scale=1.0,
            output_gate=gates,
            out=out,
        )


def test_a_quantized_cache_needs_both_global_scales(device):
    """The values are stored relative to them; leaving one out reads it as one."""
    rows = 16
    attention = _attention(
        device, kv_data_type=torch.float8_e4m3fn, kv_cache_format="fp8_e4m3"
    )
    q, k, _v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=5)
    k8 = k.float().to(torch.float8_e4m3fn)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    for missing in ("k_scale", "v_scale"):
        scales = {"k_scale": 2.0, "v_scale": 2.0}
        scales[missing] = None
        with pytest.raises(ValueError, match=missing):
            attention.run(
                q,
                k8,
                k8,
                route=route,
                block_table=block_table,
                token_to_req=token_to_req,
                output_gate=gates,
                out=out,
                **scales,
            )


def test_a_route_with_a_count_column_is_refused(device):
    """The route is indices; a trailing count is a different width."""
    attention = _attention(device)
    rows = 4
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 1, seed=5)
    packed = torch.cat(
        [route, torch.zeros(rows, 1, dtype=torch.int32, device=device)], dim=1
    )
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="count column"):
        attention.run(
            q,
            k,
            v,
            route=packed,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gates,
            out=out,
        )


def test_two_instances_replay_two_graphs_on_two_streams(device):
    """Separate arenas, separate plans, separate answers."""
    rows = 16
    instances = [_attention(device) for _ in range(2)]
    batches = [_batch(device, rows, 2, seed=seed) for seed in (7, 8)]
    outs = [
        torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        for _ in range(2)
    ]

    def call(index):
        q, k, v, route, block_table, token_to_req, gates = batches[index]
        instances[index].run(
            q,
            k,
            v,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gates,
            out=outs[index],
        )

    expected = []
    for index in range(2):
        call(index)
        torch.cuda.synchronize()
        expected.append(outs[index].clone())

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

    for out in outs:
        out.zero_()
    streams = [torch.cuda.Stream() for _ in range(2)]
    for stream, graph in zip(streams, graphs, strict=True):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    for got, want in zip(outs, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_a_workspace_that_is_too_small_or_misplaced_is_refused(device):
    """Each buffer has to be one this geometry fits in.

    The caller is told two numbers and hands over that many bytes; everything
    inside is cut from them. A buffer a byte short, or one that does not start
    on the alignment the planner wants, would have the plans reading memory
    nobody set aside.
    """
    arguments = _arguments()
    persistent_bytes, transient_bytes = flashinfer.QSAAttention.workspace_bytes(
        device=device, **arguments
    )
    with pytest.raises(ValueError, match="needs"):
        flashinfer.QSAAttention(
            torch.zeros(persistent_bytes - 1, dtype=torch.uint8, device=device),
            **arguments,
        )
    with pytest.raises(ValueError, match="aligned"):
        flashinfer.QSAAttention(
            torch.zeros(persistent_bytes + 16, dtype=torch.uint8, device=device)[1:],
            **arguments,
        )
    with pytest.raises(ValueError, match="raw bytes"):
        flashinfer.QSAAttention(
            torch.zeros(persistent_bytes, dtype=torch.int32, device=device),
            **arguments,
        )

    attention = flashinfer.QSAAttention(
        torch.zeros(persistent_bytes, dtype=torch.uint8, device=device), **arguments
    )
    with pytest.raises(ValueError, match="needs"):
        attention.bind_transient_workspace(
            torch.zeros(transient_bytes - 1, dtype=torch.uint8, device=device)
        )


def test_planning_before_the_scratch_is_bound_is_refused(device):
    """The plans run out of the float workspace the scratch carries."""
    persistent, _transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    with pytest.raises(RuntimeError, match="bind_transient_workspace"):
        attention.plan_cache(NUM_SLOTS, PAGE_SIZE)


def test_running_before_the_cache_is_planned_is_refused(device):
    """A plan is for a cache of a size, and there is no cache yet."""
    persistent, transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    rows = 4
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 1, seed=9)
    with pytest.raises(RuntimeError, match="plan_cache"):
        attention.run(
            q,
            k,
            v,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gates,
        )


def test_a_cache_that_is_replaced_is_replanned_in_the_same_workspace(device):
    """The minimal cache a memory profile runs against, then the real one.

    Both are planned in the buffer the caller reserved once, so the second
    cache costs nothing: what changes is the schedule, not the room it needs.
    """
    persistent, transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    attention.plan_cache(PAGE_SIZE, PAGE_SIZE)
    assert attention.num_slots == PAGE_SIZE

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    torch.cuda.synchronize()

    assert attention.num_slots == NUM_SLOTS
    assert torch.cuda.memory_allocated() == before, "replanning allocated"

    rows = 4
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 1, seed=3)
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
    )


def test_a_cache_the_pages_do_not_divide_is_refused(device):
    """The cache holds whole pages, so a slot count between two has no plan."""
    persistent, transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    with pytest.raises(ValueError, match="multiple of page_size"):
        attention.plan_cache(NUM_SLOTS + 1, PAGE_SIZE)


def test_the_same_input_gives_the_same_output_every_time(device):
    """Element for element, over repeated calls."""
    rows = 16
    attention = _attention(device)
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=10)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    def once():
        attention.run(
            q,
            k,
            v,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gates,
            out=out,
        )
        torch.cuda.synchronize()
        return out.clone()

    first = once()
    for _ in range(4):
        torch.testing.assert_close(once(), first, rtol=0, atol=0)


def test_the_gate_is_not_optional(device):
    """The model this serves gates its output; serving it ungated was the bug.

    Sparse attention without a gate is what
    :class:`BlockSparseAttentionWrapper` is for. Here it is a missing argument,
    caught before anything runs.
    """
    rows = 16
    attention = _attention(device)
    q, k, v, route, block_table, token_to_req, _gates = _batch(device, rows, 2, seed=11)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    with pytest.raises(TypeError, match="output_gate"):
        attention.run(
            q,
            k,
            v,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            out=out,
        )


def test_the_gate_is_exactly_what_the_kernel_applies(device):
    """The output is the attention rounded, times the logistic at float."""
    rows = 16
    attention = _attention(device)
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=11)
    ones = torch.full_like(gates, 40.0)  # sigmoid(40) is one to the bit
    ungated = torch.empty(
        rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=ones,
        out=ungated,
    )
    gated = torch.empty_like(ungated)
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
        out=gated,
    )
    expected = (
        ungated.to(torch.float32)
        * torch.sigmoid(gates.unflatten(1, (NUM_QO_HEADS, HEAD_DIM)).to(torch.float32))
    ).to(torch.bfloat16)
    torch.testing.assert_close(gated, expected, rtol=0, atol=0)


# --- packed NVFP4 ---------------------------------------------------------
#
# The cache below is the one FlashInfer's own writer produces, and the oracle
# decodes those same bytes in torch rather than asking the reader what they
# mean. Random bytes would not do: a scale plane of arbitrary values can hold
# encodings that make the whole thing NaN, and an output that is NaN agrees
# with nothing while appearing to differ from everything.

#: What a four-bit e2m1 field decodes to, by its three magnitude bits.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _decode_nvfp4(data, planes, global_scale, device):
    """Unpack a packed cache in torch, independently of the reader.

    Two e2m1 values to a byte, low nibble first; one e4m3 block scale per
    sixteen values; one host scale over the whole tensor.
    """
    magnitudes = torch.tensor(_E2M1, dtype=torch.float64, device=device)
    packed = data.to(torch.int64)
    low, high = packed & 0xF, (packed >> 4) & 0xF
    fields = torch.stack((low, high), dim=-1).reshape(*data.shape[:-1], -1)
    values = magnitudes[fields & 0x7] * torch.where((fields & 0x8) != 0, -1.0, 1.0)
    scales = planes.float().to(torch.float64).repeat_interleave(16, dim=-1)
    return values * scales * global_scale


def _nvfp4_cache(device, seed, k_scale, v_scale):
    """A packed cache the library's own writer produced."""
    generator = torch.Generator(device=device).manual_seed(seed)
    data_dim, scale_dim = HEAD_DIM // 2, HEAD_DIM // 16
    k_data = torch.zeros(
        PAGES, NUM_KV_HEADS, PAGE_SIZE, data_dim, dtype=torch.uint8, device=device
    )
    v_data = torch.zeros_like(k_data)
    k_sf = torch.zeros(
        PAGES,
        NUM_KV_HEADS,
        PAGE_SIZE,
        scale_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    v_sf = torch.zeros_like(k_sf)
    entries = PAGES * PAGE_SIZE
    keys = torch.randn(
        entries,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    values = torch.randn(
        entries,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    slots = torch.arange(entries, dtype=torch.int32, device=device)
    flashinfer.nvfp4_quantize_append_paged_kv_cache_with_slot_mapping(
        keys,
        values,
        slots,
        (k_data, v_data),
        (k_sf, v_sf),
        k_scale,
        v_scale,
        kv_layout="HND",
    )
    torch.cuda.synchronize()
    return k_data, v_data, k_sf, v_sf


def _nvfp4_attention(device):
    return _attention(
        device,
        kv_data_type=torch.uint8,
        kv_cache_format="nvfp4",
        kv_layout="HND",
        max_rows=16,
    )


def _nvfp4_case(device, seed, k_scale=1.0, v_scale=1.0, swap=False, double=None):
    """Run the packed path, and decode the same bytes in torch beside it."""
    rows = 16
    q, _k, _v, route, block_table, token_to_req, gates = _batch(
        device, rows, 2, seed=seed
    )
    k_data, v_data, k_sf, v_sf = _nvfp4_cache(device, seed, k_scale, v_scale)
    read_k = k_scale * 2 if double == "k_scale" else k_scale
    read_v = v_scale * 2 if double == "v_scale" else v_scale
    planes = (v_sf, k_sf) if swap else (k_sf, v_sf)

    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    _nvfp4_attention(device).run(
        q,
        k_data,
        v_data,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        k_sf=planes[0],
        v_sf=planes[1],
        k_scale=read_k,
        v_scale=read_v,
        output_gate=gates,
        out=out,
    )
    torch.cuda.synchronize()

    keys = _decode_nvfp4(k_data, planes[0], read_k, device)
    values = _decode_nvfp4(v_data, planes[1], read_v, device)
    # HND to the flat (slot, head, dim) the oracle indexes.
    keys = keys.permute(0, 2, 1, 3).reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    values = values.permute(0, 2, 1, 3).reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    expected = _oracle_from_values(
        q, keys, values, route, block_table, token_to_req, gates
    )
    return out, expected


def test_an_interleaved_cache_reads_the_same_as_separate_planes(device):
    """The layout this library's own writer produces, read back.

    A deployment that stores K and V per head slot as one allocation --
    ``[fp4 data | e4m3 block scales]`` per entry, which is what keeps the two
    planes in one page and one KV cache spec -- hands over two strided views
    of it. ``nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`` takes
    exactly that, so the reader has to as well; demanding contiguity here
    would refuse a cache the writer just filled.
    """
    rows = 16
    q, _k, _v, route, block_table, token_to_req, gates = _batch(
        device, rows, 2, seed=34
    )
    k_data, v_data, k_sf, v_sf = _nvfp4_cache(device, 34, 1.5, 0.75)
    data_dim, scale_dim = HEAD_DIM // 2, HEAD_DIM // 16

    # One allocation per slot, data then scales, as the deployment lays it out.
    full = torch.zeros(
        PAGES,
        2 * NUM_KV_HEADS,
        PAGE_SIZE,
        data_dim + scale_dim,
        dtype=torch.uint8,
        device=device,
    )
    full[:, 0::2, :, :data_dim] = k_data
    full[:, 1::2, :, :data_dim] = v_data
    full[:, 0::2, :, data_dim:] = k_sf.view(torch.uint8)
    full[:, 1::2, :, data_dim:] = v_sf.view(torch.uint8)
    k_slot, v_slot = full[:, 0::2], full[:, 1::2]
    interleaved = (
        k_slot[..., :data_dim],
        v_slot[..., :data_dim],
        k_slot[..., data_dim:].view(torch.float8_e4m3fn),
        v_slot[..., data_dim:].view(torch.float8_e4m3fn),
    )
    assert not interleaved[0].is_contiguous(), "the views under test are packed"
    assert interleaved[0].stride(-1) == 1

    outs = []
    for planes in ((k_data, v_data, k_sf, v_sf), interleaved):
        out = torch.empty(
            rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        _nvfp4_attention(device).run(
            q,
            planes[0],
            planes[1],
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            k_sf=planes[2],
            v_sf=planes[3],
            k_scale=1.5,
            v_scale=0.75,
            output_gate=gates,
            out=out,
        )
        outs.append(out)
    torch.cuda.synchronize()
    torch.testing.assert_close(outs[1], outs[0], rtol=0, atol=0)


def test_a_cache_plane_strided_in_its_last_dimension_is_refused(device):
    """Strides the kernel reads, yes; a gap between values, no."""
    rows = 16
    q, _k, _v, route, block_table, token_to_req, gates = _batch(
        device, rows, 2, seed=35
    )
    k_data, v_data, k_sf, v_sf = _nvfp4_cache(device, 35, 1.0, 1.0)
    doubled = torch.zeros(
        *k_data.shape[:-1], k_data.shape[-1] * 2, dtype=torch.uint8, device=device
    )
    doubled[..., ::2] = k_data
    strided = doubled[..., ::2]
    assert strided.stride(-1) == 2
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="innermost dimension"):
        _nvfp4_attention(device).run(
            q,
            strided,
            v_data,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            k_sf=k_sf,
            v_sf=v_sf,
            k_scale=1.0,
            v_scale=1.0,
            output_gate=gates,
            out=out,
        )


def test_the_packed_path_matches_a_torch_decode_of_the_same_bytes(device):
    """The reader against an independent decoder, not against itself."""
    out, expected = _nvfp4_case(device, seed=30, k_scale=1.5, v_scale=0.75)
    assert bool(torch.isfinite(out.to(torch.float32)).all()), "the output is not finite"
    _assert_within(out, expected)


def test_swapping_the_scale_planes_breaks_the_answer(device):
    """K's scales are K's: reading V's in their place has to be wrong."""
    straight, expected = _nvfp4_case(device, seed=31, k_scale=1.5, v_scale=0.75)
    _assert_within(straight, expected)
    swapped, _ = _nvfp4_case(device, seed=31, k_scale=1.5, v_scale=0.75, swap=True)
    assert bool(torch.isfinite(swapped.to(torch.float32)).all())
    assert not torch.equal(straight, swapped), "the planes were interchangeable"


@pytest.mark.parametrize("name", ["k_scale", "v_scale"])
def test_a_global_scale_is_applied_exactly_once(device, name):
    """Doubling it has to give what a doubled decode gives, and nothing else.

    A scale left out, or folded in twice, agrees with neither the plain decode
    nor the doubled one.
    """
    out, expected = _nvfp4_case(device, seed=32, k_scale=1.5, v_scale=0.75, double=name)
    assert bool(torch.isfinite(out.to(torch.float32)).all())
    _assert_within(out, expected)

    plain, _plain = _nvfp4_case(device, seed=32, k_scale=1.5, v_scale=0.75)
    assert not torch.equal(out, plain), f"{name} was ignored"


def test_a_packed_cache_without_its_scales_is_refused(device):
    """The format says the planes are there, so leaving them out is an error."""
    attention = _nvfp4_attention(device)
    rows = 16
    q, _k, _v, route, block_table, token_to_req, gates = _batch(
        device, rows, 2, seed=33
    )
    k_data, _v_data, k_sf, _v_sf = _nvfp4_cache(device, 33, 1.0, 1.0)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="scale plane"):
        attention.run(
            q,
            k_data,
            k_data,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            k_sf=k_sf,
            output_gate=gates,
            out=out,
        )


# --- FP8 ------------------------------------------------------------------


def test_the_fp8_path_matches_a_torch_dequant(device):
    """e4m3 values with one host scale each, against a torch dequantisation."""
    rows = 16
    attention = _attention(
        device, kv_data_type=torch.float8_e4m3fn, kv_cache_format="fp8_e4m3"
    )
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=40)
    k_scale, v_scale = 2.0, 0.5
    k8 = (k.float() / k_scale).to(torch.float8_e4m3fn)
    v8 = (v.float() / v_scale).to(torch.float8_e4m3fn)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attention.run(
        q,
        k8,
        v8,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        k_scale=k_scale,
        v_scale=v_scale,
        output_gate=gates,
        out=out,
    )
    torch.cuda.synchronize()
    assert bool(torch.isfinite(out.to(torch.float32)).all())

    keys = (k8.float().to(torch.float64) * k_scale).reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    values = (v8.float().to(torch.float64) * v_scale).reshape(
        -1, NUM_KV_HEADS, HEAD_DIM
    )
    expected = _oracle_from_values(
        q, keys, values, route, block_table, token_to_req, gates
    )
    _assert_within(out, expected)


def test_an_fp8_cache_may_not_carry_scale_planes(device):
    """Block scales belong to nvfp4; an FP8 cache has one scale per tensor."""
    rows = 16
    attention = _attention(
        device, kv_data_type=torch.float8_e4m3fn, kv_cache_format="fp8_e4m3"
    )
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=41)
    k8 = k.float().to(torch.float8_e4m3fn)
    planes = torch.zeros(
        PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM // 16, dtype=torch.uint8, device=device
    )
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="no scale planes"):
        attention.run(
            q,
            k8,
            k8,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            k_sf=planes,
            v_sf=planes,
            k_scale=1.0,
            v_scale=1.0,
            output_gate=gates,
            out=out,
        )


def test_a_dense_cache_has_no_global_scale(device):
    """Naming the format means the arguments that go with it are checked."""
    rows = 16
    attention = _attention(device)
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=42)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="no global scale"):
        attention.run(
            q,
            k,
            v,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            k_scale=2.0,
            output_gate=gates,
            out=out,
        )


def test_a_format_the_dtype_contradicts_is_refused(device):
    """The format names what the bytes are; the dtype has to agree."""
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    common = {
        "max_rows": max(BUCKETS),
        "num_qo_heads": NUM_QO_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "route_width": ROUTE_WIDTH,
        "q_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
    }
    with pytest.raises(ValueError, match="packed NVFP4"):
        flashinfer.QSAAttention(
            workspace, kv_data_type=torch.bfloat16, kv_cache_format="nvfp4", **common
        )
    with pytest.raises(ValueError, match="not a dense cache"):
        flashinfer.QSAAttention(
            workspace, kv_data_type=torch.uint8, kv_cache_format="dense", **common
        )
    with pytest.raises(ValueError, match="kv_cache_format"):
        flashinfer.QSAAttention(
            workspace, kv_data_type=torch.bfloat16, kv_cache_format="fp4", **common
        )


def test_replanning_after_a_run_is_refused(device):
    """Once a step has run, a graph may be holding the schedule it ran.

    Replacing a plan then would leave the capture replaying byte offsets into
    an arena that has been written over. Before the first run there is nothing
    holding them, which is why a cache may still be replaced up to that point.
    """
    attention = _attention(device)
    rows = 4
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 1, seed=5)
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
    )
    # The same cache is what every other layer of a rank asks for, and it
    # changes nothing, so it stays allowed.
    attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    with pytest.raises(RuntimeError, match="has run"):
        attention.plan_cache(NUM_SLOTS * 2, PAGE_SIZE)


def test_the_workspace_size_does_not_depend_on_the_page_size(device):
    """Which is why the page size is not part of the geometry.

    A caller sizes its workspace before the cache is allocated, and how many
    slots a page holds is decided when it is. That is only sound if the answer
    does not move with it: the planner lays out one entry per route element and
    the page size says how those elements are addressed, not how many there are.
    """
    arguments = _arguments()
    sizes = {
        page: flashinfer.QSAAttention.workspace_bytes(device=device, **arguments)
        for page in (16, 64, 256, 4096)
    }
    assert len(set(sizes.values())) == 1, sizes


def test_planning_the_cache_it_already_has_does_nothing(device):
    """Every layer of a rank binds the same cache and asks for the same plan.

    They share one runtime, so the second of them through must not rebuild
    what the first built -- rebuilding would replace the wrappers a captured
    graph is about to hold, once per layer, for no change at all.
    """
    persistent, transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    first = dict(attention._wrappers)
    staging = attention._staging

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(48):
        attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() == before, "a repeat ask allocated"
    assert attention._staging is staging, "a repeat ask took another staging buffer"
    for rows, wrapper in first.items():
        assert attention._wrappers[rows] is wrapper, f"rung {rows} was rebuilt"


def test_the_cache_is_replaced_exactly_once(device):
    """A memory profile's minimal cache, then the real one, and no more.

    The rank's layers all bind in one pass, so the replacement happens on the
    first of them and the other forty-seven find it already done.
    """
    persistent, transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    attention.plan_cache(PAGE_SIZE, PAGE_SIZE)
    profiling = dict(attention._wrappers)

    attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    production = dict(attention._wrappers)
    assert all(production[rows] is not profiling[rows] for rows in production)

    once = dict(attention._wrappers)
    for _ in range(47):
        attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    assert all(attention._wrappers[rows] is once[rows] for rows in once)


def test_replanning_back_and_forth_does_not_accumulate(device):
    """What a worker torn down and rebuilt, or scaled, would do repeatedly.

    Each replacement drops the plans before it, so the memory the wrappers keep
    outside the workspace has to come back. One allocated delta cannot show
    that; ten alternations can.
    """
    persistent, transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    attention.plan_cache(PAGE_SIZE, PAGE_SIZE)
    attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    gc.collect()
    torch.cuda.synchronize()
    settled = torch.cuda.memory_allocated()
    staging = attention._staging

    for index in range(10):
        attention.plan_cache(PAGE_SIZE if index % 2 else NUM_SLOTS, PAGE_SIZE)
        attention.plan_cache(NUM_SLOTS if index % 2 else PAGE_SIZE, PAGE_SIZE)
    gc.collect()
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated() == settled, "twenty replans left memory behind"
    assert attention._staging is staging, "a replan took another staging buffer"


def test_the_plans_a_replan_replaces_are_released(device):
    """Not just "the total is the same" -- the old objects have to be gone."""
    persistent, transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    attention.plan_cache(PAGE_SIZE, PAGE_SIZE)
    dead = [weakref.ref(wrapper) for wrapper in attention._wrappers.values()]

    attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    gc.collect()

    assert all(reference() is None for reference in dead), (
        "a replaced plan is still alive"
    )


def test_the_pinned_staging_buffer_is_taken_once(device):
    """Host memory, and the planner is done with it before it returns.

    One per attention rather than one per plan: the plans are built one after
    another, so eight megabytes of pinned memory each would be that many times
    over for a buffer only one of them is using at a time.
    """
    persistent, transient, arguments = _workspace(device)
    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    assert attention._staging is None, "staging was taken before there was a plan"

    attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    staging = attention._staging
    assert staging.is_pinned() and staging.device.type == "cpu"
    assert staging.numel() == max(attention._plan_bytes)

    attention.plan_cache(PAGE_SIZE, PAGE_SIZE)
    assert attention._staging is staging


# --- what planning may read ------------------------------------------------


def test_planning_initialises_the_scratch_it_plans_against(device, monkeypatch):
    """The transient buffer is the caller's, and it arrives dirty.

    It is scratch every consumer of a step reuses from its first byte, so what
    is in the route and the mask when :meth:`plan_cache` runs is whatever the
    last tenant left. The planner reads both. A leftover negative is refused
    outright -- ``indices must be non-negative`` -- and a leftover that happens
    to be in range is worse: it plans a sparsity pattern no step will use, and
    says nothing. The schedule comes from the row pointers, which are fixed, so
    planning has to start from a route and a mask this object put there.

    Poisoned with 0xA5 rather than zeroed: every byte is nonzero and every
    int32 word is positive and in range, which is exactly the case that used to
    pass silently.
    """
    persistent, transient, arguments = _workspace(device)
    persistent.fill_(0xA5)
    transient.fill_(0xA5)

    attention = flashinfer.QSAAttention(persistent, **arguments)
    attention.bind_transient_workspace(transient)
    assert int(attention._route_base.max()) != 0, "the poison did not reach the route"

    seen = []
    original = flashinfer.sparse.BlockSparseAttentionWrapper.plan

    def spy(self, indptr, indices, *args, **kwargs):
        packed_mask = kwargs.get("packed_mask")
        seen.append(
            (
                indices.clone(),
                None if packed_mask is None else packed_mask.clone(),
            )
        )
        return original(self, indptr, indices, *args, **kwargs)

    monkeypatch.setattr(flashinfer.sparse.BlockSparseAttentionWrapper, "plan", spy)
    attention.plan_cache(NUM_SLOTS, PAGE_SIZE)
    monkeypatch.undo()

    ladder = attention.row_buckets
    assert len(seen) == len(ladder), f"{len(seen)} plans for {len(ladder)} buckets"
    for indices, packed_mask in seen:
        assert indices.numel()
        assert int(indices.min()) == 0 and int(indices.max()) == 0, (
            "the planner was handed a route this object did not write"
        )
        assert packed_mask is not None and packed_mask.numel()
        assert int(packed_mask.max()) == 0, (
            "the planner was handed a mask this object did not write"
        )

    # And the plans it built are the plans a clean workspace builds: same
    # inputs, same answer, bit for bit.
    rows = 16
    q, k, v, route, block_table, token_to_req, gates = _batch(device, rows, 2, seed=3)
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    attention.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
        out=out,
    )

    clean = _attention(device)
    expected = torch.empty_like(out)
    clean.run(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gates,
        out=expected,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# --- nothing is launched on a bad call ------------------------------------


def _poisoned(monkeypatch):
    """Make every kernel this path can reach fail loudly if it is reached."""
    import flashinfer.qsa_attention as module

    def refuse(*args, **kwargs):
        raise AssertionError("a kernel was launched on an input that is refused")

    monkeypatch.setattr(module, "qsa_route_from_logical", refuse)
    monkeypatch.setattr(module, "qsa_output_gate", refuse)
    monkeypatch.setattr(flashinfer.sparse.BlockSparseAttentionWrapper, "run", refuse)


@pytest.mark.parametrize(
    "case",
    [
        "short_cache",
        "wrong_cache_shape",
        "wrong_plane_shape",
        "one_plane",
        "missing_scale",
        "bad_scale",
        "wide_route",
        "wrong_gate",
        "wrong_out",
    ],
)
def test_nothing_is_launched_when_the_call_is_refused(device, monkeypatch, case):
    """The checks come before the kernels, not instead of a fault inside one.

    A cache shorter than the route can address, a scale plane of the wrong
    shape, a missing global scale: each of these would be read as a pointer or
    a multiplier. With every kernel poisoned, the refusal has to arrive first.
    """
    rows = 16
    packed = case in ("wrong_plane_shape", "one_plane")
    if packed:
        attention = _nvfp4_attention(device)
        k_data, v_data, k_sf, v_sf = _nvfp4_cache(device, 50, 1.0, 1.0)
        scales = {"k_scale": 1.0, "v_scale": 1.0}
        planes = {"k_sf": k_sf, "v_sf": v_sf}
    else:
        attention = _attention(device)
        _q, k_data, v_data, _r, _bt, _t2r, _g = _batch(device, rows, 2, seed=50)
        scales, planes = {}, {}
    q, _k, _v, route, block_table, token_to_req, gates = _batch(
        device, rows, 2, seed=50
    )
    out = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    if case == "short_cache":
        k_data = k_data[: PAGES // 2]
        v_data = v_data[: PAGES // 2]
    elif case == "wrong_cache_shape":
        k_data = k_data[..., : HEAD_DIM // 2].contiguous()
        v_data = v_data[..., : HEAD_DIM // 2].contiguous()
    elif case == "wrong_plane_shape":
        planes["k_sf"] = planes["k_sf"][..., :1].contiguous()
    elif case == "one_plane":
        planes["v_sf"] = None
    elif case == "missing_scale":
        attention = _attention(
            device, kv_data_type=torch.float8_e4m3fn, kv_cache_format="fp8_e4m3"
        )
        k_data = v_data = k_data.float().to(torch.float8_e4m3fn)
        scales = {"k_scale": 2.0}
    elif case == "bad_scale":
        attention = _attention(
            device, kv_data_type=torch.float8_e4m3fn, kv_cache_format="fp8_e4m3"
        )
        k_data = v_data = k_data.float().to(torch.float8_e4m3fn)
        scales = {"k_scale": 0.0, "v_scale": 1.0}
    elif case == "wide_route":
        route = torch.zeros(rows, ROUTE_WIDTH + 1, dtype=torch.int32, device=device)
    elif case == "wrong_gate":
        gates = gates[:, : NUM_QO_HEADS * HEAD_DIM // 2].contiguous()
    elif case == "wrong_out":
        out = torch.empty(
            rows, NUM_QO_HEADS, HEAD_DIM // 2, dtype=torch.bfloat16, device=device
        )

    _poisoned(monkeypatch)
    with pytest.raises((ValueError, TypeError)):
        attention.run(
            q,
            k_data,
            v_data,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gates,
            out=out,
            **planes,
            **scales,
        )


def test_an_fp8_cache_has_to_carry_its_type(device):
    """A uint8 view of e4m3 bytes is a view away; the format stays in the type.

    Two formats arrive as raw bytes -- e4m3 values and packed NVFP4 -- and a
    uint8 cache cannot say which it is. So FP8 is handed over typed.
    """
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        flashinfer.QSAAttention(
            workspace,
            max_rows=max(BUCKETS),
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            route_width=ROUTE_WIDTH,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.uint8,
            o_data_type=torch.bfloat16,
            kv_cache_format="fp8_e4m3",
        )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"num_qo_heads": 3, "num_kv_heads": 2}, "whole number of query heads"),
        ({"num_kv_heads": 0}, "positive"),
        (
            {"kv_cache_format": "nvfp4", "kv_data_type": torch.uint8, "head_dim": 100},
            "multiple of sixteen",
        ),
    ],
)
def test_a_geometry_the_route_cannot_serve_is_refused(device, overrides, message):
    """Caught while the layer is built, not by a kernel that was never made."""
    arguments = {
        "max_rows": max(BUCKETS),
        "num_qo_heads": NUM_QO_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "route_width": ROUTE_WIDTH,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "kv_cache_format": "dense",
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        flashinfer.QSAAttention(
            torch.empty(1024 * 1024, dtype=torch.uint8, device=device), **arguments
        )


@pytest.mark.parametrize(
    "workspace,message",
    [
        ("cpu", "CUDA"),
        ("int32", "raw bytes"),
        ("strided", "contiguous"),
    ],
)
def test_a_workspace_the_kernels_cannot_use_is_refused(device, workspace, message):
    """It is handed straight to the planner, so its shape is checked here."""
    buffers = {
        "cpu": torch.empty(1024, dtype=torch.uint8),
        "int32": torch.empty(1024, dtype=torch.int32, device=device),
        "strided": torch.empty(2048, dtype=torch.uint8, device=device)[::2],
    }
    with pytest.raises(ValueError, match=message):
        flashinfer.QSAAttention(
            buffers[workspace],
            max_rows=max(BUCKETS),
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            route_width=ROUTE_WIDTH,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            o_data_type=torch.bfloat16,
            kv_cache_format="dense",
        )


def test_two_bucket_graphs_replay_against_each_other(device):
    """The route and the mask are one allocation now, viewed per bucket.

    Two graphs capture the same pointers, so replaying one has to leave the
    other's answer intact -- and it does, because each run rewrites the rows it
    is about to read before it launches. What is not supported, and what this
    does not do, is replaying two of them at once on one slot: they would be
    writing the same bytes.
    """
    buckets = (4, 16)
    attention = _attention(
        device,
        kv_data_type=torch.bfloat16,
        kv_cache_format="dense",
        max_rows=max(buckets),
    )

    graphs, outs, expected = {}, {}, {}
    for rows in buckets:
        q, k, v, route, block_table, token_to_req, gates = _batch(
            device, rows, 2, seed=60 + rows
        )
        out = torch.empty(
            rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        held = (q, k, v, route, block_table, token_to_req, gates, out)
        call = lambda held=held: attention.run(
            held[0],
            held[1],
            held[2],
            route=held[3],
            block_table=held[4],
            token_to_req=held[5],
            output_gate=held[6],
            out=held[7],
        )
        call()
        torch.cuda.synchronize()
        expected[rows] = out.clone()
        outs[rows] = held

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                call()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[rows] = graph

    # Alternate. Each replay has to give that bucket's answer, however many
    # times the other one has run in between.
    for _ in range(4):
        for rows in buckets:
            outs[rows][-1].zero_()
            graphs[rows].replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outs[rows][-1], expected[rows], rtol=0, atol=0)
    for rows in buckets:
        torch.testing.assert_close(outs[rows][-1], expected[rows], rtol=0, atol=0)


def test_the_route_and_mask_are_one_allocation(device):
    """What makes a longer ladder affordable, asserted rather than assumed."""
    attention = _attention(
        device,
        kv_data_type=torch.bfloat16,
        kv_cache_format="dense",
        max_rows=2048,
    )
    ladder = attention.row_buckets
    assert len(ladder) > 1, "one rung proves nothing about sharing"
    base = attention._route_base.data_ptr()
    mask_base = attention._mask_base.data_ptr()
    for rows in ladder:
        assert attention._route[rows].data_ptr() == base
        assert attention._route[rows].shape[0] == rows
        assert attention._mask[rows].data_ptr() == mask_base
