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

# The step this file drives, end to end and inside FlashInfer alone:
#
#   scores -> top-k -> route -> sparse attention -> gate -> a projection
#
# The oracle beside it is plain torch: it scores, selects, gathers and attends
# in float64 and applies the same projection, so the comparison is against an
# implementation that shares no code with the kernels. Nothing here imports a
# serving stack; the projection is a fixed synthetic head, which is what makes
# the token comparison below mean anything -- and it is not a model, so this is
# not a claim about one.

COMPRESS_RATIO = 4
TOKEN_TOPK = 32
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
ROUTE_WIDTH = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
NUM_QO_HEADS = 4
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 16
PAGES = 16
NUM_SLOTS = PAGES * PAGE_SIZE
COLUMNS = 64
ROWS = 8
VOCAB = 64

#: A numerical tolerance, proportional to the output, because the error is:
#: a batch whose values are twice as large rounds at twice the spacing. It is
#: not a count of ulps -- a bfloat16's spacing changes with the binade, so the
#: distance between two representable values is not a fixed fraction of
#: either. The measured error over these shapes is 0.0084 to 0.0104 of the
#: largest output; this is 0.0156.
_ABSOLUTE_PER_UNIT = 4 * 2**-8
_MAX_RELATIVE = 3e-2
_RELATIVE_FLOOR = 5e-2


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def _world(device, seed):
    """One request's cache, queries, positions and a fixed projection head."""
    generator = torch.Generator(device=device).manual_seed(seed)
    block_table = (
        torch.randperm(PAGES, device=device, generator=generator)
        .reshape(1, PAGES)
        .contiguous()
        .to(torch.int32)
    )
    compressed = torch.randn(
        PAGES,
        PAGE_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
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
        ROWS,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gate = torch.randn(
        ROWS,
        NUM_QO_HEADS * HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    token_to_req = torch.zeros(ROWS, dtype=torch.int32, device=device)
    positions = torch.full(
        (ROWS,), PAGES * PAGE_SIZE - 1, dtype=torch.int32, device=device
    )
    lengths = torch.full((1,), PAGES * PAGE_SIZE, dtype=torch.int32, device=device)
    head = torch.randn(
        NUM_QO_HEADS * HEAD_DIM,
        VOCAB,
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    return (
        q,
        compressed,
        k,
        v,
        block_table,
        token_to_req,
        positions,
        lengths,
        gate,
        head,
    )


def _config(**overrides):
    """One deployment's QSA, as the caller describes it."""
    arguments = {
        "num_qo_heads": NUM_QO_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "max_rows": ROWS,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "kv_cache_format": "dense",
        "max_columns": COLUMNS,
        "compress_ratio": COMPRESS_RATIO,
        "token_topk": TOKEN_TOPK,
        "index_num_heads": NUM_QO_HEADS,
        "index_head_dim": HEAD_DIM,
    }
    arguments.update(overrides)
    return flashinfer.QSAConfig(**arguments)


def _runtime(device, config=None, num_slots=NUM_SLOTS, page_size=PAGE_SIZE):
    """Allocate what it asks for, build it, plan it. The caller's whole part."""
    config = config or _config()
    need = flashinfer.QSA.workspace_requirements(config, device=device)
    persistent = torch.zeros(need.persistent_bytes, dtype=torch.uint8, device=device)
    transient = torch.zeros(need.transient_bytes, dtype=torch.uint8, device=device)
    runtime = flashinfer.QSA(config, persistent)
    runtime.bind_transient_workspace(transient)
    runtime.plan_cache(num_slots, page_size)
    # The buffers are the caller's; keep them alive as long as the runtime.
    runtime._test_workspace = (persistent, transient)
    return runtime


def _flashinfer_step(device, world, runtime=None):
    """Selection and attention, both from the library, nothing else."""
    q, compressed, k, v, block_table, token_to_req, positions, lengths, gate, _ = world
    runtime = runtime or _runtime(device)
    # The route is the caller's: a layer reusing it across speculative steps
    # reuses its own, so it cannot come out of a buffer layers share.
    route = torch.empty(ROWS, ROUTE_WIDTH, dtype=torch.int32, device=device)
    runtime.run_selection(
        q,
        compressed,
        block_table,
        token_to_req,
        positions,
        lengths,
        out_route=route,
    )
    out = torch.empty(ROWS, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    runtime.run_attention(
        q,
        k,
        v,
        route=route,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gate,
        out=out,
    )
    torch.cuda.synchronize()
    return route, out


def _torch_step(device, world):
    """The same thing in float64, sharing no code with the kernels."""
    q, compressed, k, v, block_table, token_to_req, positions, lengths, gate, _ = world
    flat_compressed = compressed.reshape(-1, HEAD_DIM).to(torch.float64)
    flat_k = k.reshape(-1, NUM_KV_HEADS, HEAD_DIM).to(torch.float64)
    flat_v = v.reshape(-1, NUM_KV_HEADS, HEAD_DIM).to(torch.float64)

    route = torch.full((ROWS, ROUTE_WIDTH), -1, dtype=torch.int32, device=device)
    out = torch.zeros(ROWS, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float64, device=device)
    for row in range(ROWS):
        request = int(token_to_req[row])
        position = int(positions[row])
        past_blocks = min(
            (position + 1) // COMPRESS_RATIO,
            int(lengths[request]) // COMPRESS_RATIO,
            COLUMNS,
        )
        # Score the blocks the query has entirely behind it.
        if past_blocks:
            entries = [
                int(block_table[request, column // PAGE_SIZE]) * PAGE_SIZE
                + column % PAGE_SIZE
                for column in range(past_blocks)
            ]
            keys = flat_compressed[entries]
            scores = (keys @ q[row].to(torch.float64).T).clamp(min=0).sum(dim=1) / (
                HEAD_DIM**0.5
            )
            order = sorted(range(past_blocks), key=lambda c: (-scores[c].item(), c))
            for rank, block in enumerate(order[:BLOCK_TOPK]):
                for offset in range(COMPRESS_RATIO):
                    route[row, rank * COMPRESS_RATIO + offset] = (
                        block * COMPRESS_RATIO + offset
                    )
        tail_start = (position + 1) // COMPRESS_RATIO * COMPRESS_RATIO
        for index, token in enumerate(range(tail_start, position + 1)):
            route[row, BLOCK_TOPK * COMPRESS_RATIO + index] = token

        # Attend over exactly the tokens that route names.
        slots = []
        for column in range(ROUTE_WIDTH):
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
            if slot < NUM_SLOTS:
                slots.append(slot)
        if not slots:
            continue
        keys, values = flat_k[slots, 0], flat_v[slots, 0]
        for head in range(NUM_QO_HEADS):
            logits = (keys @ q[row, head].to(torch.float64)) * (HEAD_DIM**-0.5)
            out[row, head] = torch.softmax(logits, dim=0) @ values

    rounded = out.to(torch.bfloat16)
    gates = gate.unflatten(1, (NUM_QO_HEADS, HEAD_DIM))
    gated = (rounded.to(torch.float32) * torch.sigmoid(gates.to(torch.float32))).to(
        torch.bfloat16
    )
    return route, gated


def _errors(got, want):
    a, b = got.to(torch.float64), want.to(torch.float64)
    difference = (a - b).abs()
    significant = b.abs() >= _RELATIVE_FLOOR
    relative = (
        (difference[significant] / b.abs()[significant]).max().item()
        if bool(significant.any())
        else 0.0
    )
    return difference.max().item(), relative


def _assert_within(got, want, message=""):
    """The two bounds, with the absolute one scaled to the output's own size."""
    absolute, relative = _errors(got, want)
    peak = max(1.0, float(want.to(torch.float64).abs().max()))
    allowed = _ABSOLUTE_PER_UNIT * peak
    assert absolute <= allowed, f"max abs {absolute} over {allowed} {message}"
    assert relative <= _MAX_RELATIVE, f"max rel {relative} {message}"


def _logits(out, head):
    return out.reshape(ROWS, -1).to(torch.float32) @ head


def test_the_whole_step_matches_a_plain_torch_oracle(device):
    """Scores, selection, route, attention and gate, against float64 torch."""
    world = _world(device, seed=1)
    route, out = _flashinfer_step(device, world)
    want_route, want_out = _torch_step(device, world)

    # The route, as a set per row: the top-k promises no order, and the
    # attention that reads it treats the expanded region as a set with a mask.
    expanded = BLOCK_TOPK * COMPRESS_RATIO
    for row in range(ROWS):
        torch.testing.assert_close(
            route[row, :expanded].sort().values,
            want_route[row, :expanded].sort().values,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            route[row, expanded:], want_route[row, expanded:], rtol=0, atol=0
        )

    assert bool(torch.isfinite(out.to(torch.float32)).all())
    _assert_within(out, want_out)


def test_the_first_token_agrees_through_a_fixed_head(device):
    """What the step decides, not only what it computes.

    A fixed synthetic projection turns the output into logits, and those into a
    choice. This is not a model and says nothing about one; it is a way of
    asking whether the numeric difference is small enough to leave the decision
    alone.
    """
    world = _world(device, seed=2)
    head = world[-1]
    _route, out = _flashinfer_step(device, world)
    _want_route, want_out = _torch_step(device, world)

    got_logits, want_logits = _logits(out, head), _logits(want_out, head)
    assert torch.equal(got_logits.argmax(dim=1), want_logits.argmax(dim=1)), (
        "the chosen token moved"
    )
    # The top five, allowing the boundary to move. Two candidates whose logits
    # are a hundredth apart can trade the fifth place on a difference the size
    # of one bfloat16 step, and that is not the step deciding anything
    # differently. So a candidate that is in one list and not the other has to
    # be one that close to the cut.
    got_top, want_top = got_logits.topk(5, dim=1), want_logits.topk(5, dim=1)
    for row in range(ROWS):
        mine = set(got_top.indices[row].tolist())
        theirs = set(want_top.indices[row].tolist())
        cut = float(want_top.values[row, -1])
        for candidate in mine ^ theirs:
            gap = abs(float(want_logits[row, candidate]) - cut)
            assert gap < 1e-2, (
                f"row {row} candidate {candidate} entered or left the top five "
                f"from {gap} away"
            )

    got_logprobs = torch.log_softmax(got_logits.to(torch.float64), dim=1)
    want_logprobs = torch.log_softmax(want_logits.to(torch.float64), dim=1)
    chosen = want_logits.argmax(dim=1)
    difference = (
        got_logprobs.gather(1, chosen[:, None])
        - want_logprobs.gather(1, chosen[:, None])
    ).abs()
    assert float(difference.max()) < 1e-2, f"logprob moved by {float(difference.max())}"


def test_dropping_the_gate_is_caught(device):
    """The first negative control: the defect this whole API exists to stop."""
    world = _world(device, seed=3)
    _route, gated = _flashinfer_step(device, world)
    _want_route, want_out = _torch_step(device, world)
    gate = world[8].unflatten(1, (NUM_QO_HEADS, HEAD_DIM))
    # What the route would have produced without the gate: the oracle's output
    # divided by the logistic it was multiplied by.
    ungated = (want_out.to(torch.float64) / torch.sigmoid(gate.to(torch.float64))).to(
        torch.bfloat16
    )
    absolute, _relative = _errors(ungated, want_out)
    peak = max(1.0, float(want_out.to(torch.float64).abs().max()))
    assert absolute > _ABSOLUTE_PER_UNIT * peak, (
        "the gate makes no difference to this batch"
    )
    _assert_within(gated, want_out)


def test_a_count_column_in_the_route_is_caught(device):
    """The second: a route one column too wide is refused, not trimmed."""
    world = _world(device, seed=4)
    q, _compressed, k, v, block_table, token_to_req, _pos, _lens, gate, _head = world
    runtime = _runtime(device)
    packed = torch.zeros(ROWS, ROUTE_WIDTH + 1, dtype=torch.int32, device=device)
    out = torch.empty(ROWS, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="count column"):
        runtime.run_attention(
            q,
            k,
            v,
            route=packed,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gate,
            out=out,
        )


def test_the_wrong_page_size_is_caught(device):
    """The third: a route of KV entries read as page ids goes somewhere else.

    It never gets as far as reading it wrongly. A page size of one over the
    same number of slots describes a different cache -- two hundred and fifty
    six pages of one entry, not sixteen of sixteen -- and the cache handed over
    does not have that shape, which is checked before anything is launched. The
    right page size gives the oracle's answer.
    """
    world = _world(device, seed=5)
    q, compressed, k, v, block_table, token_to_req, positions, lengths, gate, _ = world
    selection = flashinfer.QSASelection(
        max_rows=ROWS,
        max_columns=COLUMNS,
        compress_ratio=COMPRESS_RATIO,
        token_topk=TOKEN_TOPK,
        num_heads=NUM_QO_HEADS,
        head_dim=HEAD_DIM,
        device=device,
    )
    route = torch.empty(ROWS, ROUTE_WIDTH, dtype=torch.int32, device=device)
    selection.run(
        q,
        compressed,
        block_table,
        token_to_req,
        positions,
        lengths,
        out_route=route,
        workspace=torch.zeros(
            selection.workspace_size(), dtype=torch.uint8, device=device
        ),
    )

    def attend(page_size):
        runtime = _runtime(device, page_size=page_size)
        out = torch.empty(
            ROWS, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        runtime.run_attention(
            q,
            k,
            v,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gate,
            out=out,
        )
        torch.cuda.synchronize()
        return out

    with pytest.raises(ValueError, match="k_data must be"):
        attend(1)

    _want_route, want_out = _torch_step(device, world)
    _assert_within(attend(PAGE_SIZE), want_out)


def test_one_selection_workspace_serves_two_selections_in_turn(device):
    """Taking turns is fine: the scratch does not outlive a call.

    This is not the alias negative control -- it is the reason the control has
    to live where the regions are handed out. A selection cannot see the other
    selections, and a region it is given and done with is a region another may
    have next.
    """
    first, second = _world(device, seed=6), _world(device, seed=7)
    selections = [
        flashinfer.QSASelection(
            max_rows=ROWS,
            max_columns=COLUMNS,
            compress_ratio=COMPRESS_RATIO,
            token_topk=TOKEN_TOPK,
            num_heads=NUM_QO_HEADS,
            head_dim=HEAD_DIM,
            device=device,
        )
        for _ in range(2)
    ]
    shared = torch.zeros(
        selections[0].workspace_size(), dtype=torch.uint8, device=device
    )

    def select(selection, world, workspace):
        q, compressed, _k, _v, block_table, token_to_req, positions, lengths, _g, _h = (
            world
        )
        route = torch.empty(ROWS, ROUTE_WIDTH, dtype=torch.int32, device=device)
        selection.run(
            q,
            compressed,
            block_table,
            token_to_req,
            positions,
            lengths,
            out_route=route,
            workspace=workspace,
        )
        torch.cuda.synchronize()
        return route

    alone = [
        select(
            selections[index],
            world,
            torch.zeros(
                selections[index].workspace_size(), dtype=torch.uint8, device=device
            ),
        )
        for index, world in enumerate((first, second))
    ]
    together = [
        select(selections[0], first, shared),
        select(selections[1], second, shared),
    ]
    for got, want in zip(together, alone, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert not torch.equal(alone[0], alone[1]), "the two batches select the same route"


def test_the_runtime_allocates_nothing_the_caller_did_not_reserve(device):
    """Its buffers are views of the one buffer it was handed, not its own.

    What the caller reserves is what the memory profile counted; anything the
    runtime took for itself on top of that is memory nobody budgeted. Measured
    across construction, planning and a step, because each of the three used to
    allocate: the padded query and output, the route and mask it expands into,
    and the row pointers all came out of the allocator before.
    """
    world = _world(device, seed=11)
    config = _config()
    need = flashinfer.QSA.workspace_requirements(config, device=device)
    persistent = torch.zeros(need.persistent_bytes, dtype=torch.uint8, device=device)
    transient = torch.zeros(need.transient_bytes, dtype=torch.uint8, device=device)

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    runtime = flashinfer.QSA(config, persistent)
    runtime.bind_transient_workspace(transient)
    torch.cuda.synchronize()
    construction = torch.cuda.memory_allocated() - before

    runtime.plan_cache(NUM_SLOTS, PAGE_SIZE)
    _route, _out = _flashinfer_step(device, world, runtime=runtime)
    torch.cuda.synchronize()

    # The block-sparse wrapper keeps its own plan metadata, which is small and
    # fixed per rung; what must be zero is this object taking buffers for
    # itself while holding a buffer the caller gave it for exactly that.
    assert construction == 0, f"construction allocated {construction} bytes"


def test_the_size_query_allocates_nothing(device):
    """It is asked once per layer while the caller's memory can still grow."""
    config = _config()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()

    sizes = [
        tuple(flashinfer.QSA.workspace_requirements(config, device=device))
        for _ in range(8)
    ]
    torch.cuda.synchronize()

    assert len(set(sizes)) == 1, "the same configuration sized differently"
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.memory_reserved() == reserved
    assert torch.cuda.max_memory_allocated() <= allocated


def test_a_route_belongs_to_its_layer_and_survives_another_layers_step(device):
    """The route is the caller's, which is what makes reuse across steps work.

    A speculative decoder selects once on the target-aligned step and attends
    with that route again on the steps after it. The layers of a rank share one
    runtime and run one after another inside a step, so a route kept inside the
    runtime would be the previous layer's by the time the next step reused it.
    """
    world = _world(device, seed=12)
    q, compressed, k, v, block_table, token_to_req, positions, lengths, gate, _ = world
    runtime = _runtime(device)

    mine = torch.empty(ROWS, ROUTE_WIDTH, dtype=torch.int32, device=device)
    runtime.run_selection(
        q, compressed, block_table, token_to_req, positions, lengths, out_route=mine
    )
    kept = mine.clone()

    # Another layer of the same rank, on the same runtime, with its own route.
    theirs = torch.empty(ROWS, ROUTE_WIDTH, dtype=torch.int32, device=device)
    runtime.run_selection(
        q.flip(0).contiguous(),
        compressed,
        block_table,
        token_to_req,
        positions,
        lengths,
        out_route=theirs,
    )
    torch.cuda.synchronize()
    assert torch.equal(mine, kept), "another layer's selection moved this route"

    # And the reuse itself: attending with the route kept from before.
    out = torch.empty(ROWS, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    runtime.run_attention(
        q,
        k,
        v,
        route=mine,
        block_table=block_table,
        token_to_req=token_to_req,
        output_gate=gate,
        out=out,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()


# --- the scratch is shared, and the plans are not -------------------------


def _poison(runtime):
    """What another consumer of the step does to the scratch it also uses.

    The caller's workspace hands every consumer views from its first byte, so
    between two QSA calls the whole transient buffer may hold anything at all.
    """
    persistent, transient = runtime._test_workspace
    transient.fill_(0xA5)
    return persistent


def _stamp(persistent):
    return int(persistent.to(torch.int64).sum().item())


def test_a_step_survives_the_scratch_being_overwritten(device):
    """Every transient byte is rewritten before it is read, or this fails.

    A buffer initialised once and reused -- a padded query's tail, a route's
    padding rows, a mask -- looks correct for as long as nothing else touches
    the scratch, and the shared workspace touches it constantly.
    """
    world = _world(device, seed=21)
    runtime = _runtime(device)
    _route, clean = _flashinfer_step(device, world, runtime=runtime)
    clean = clean.clone()

    _poison(runtime)
    _route, poisoned = _flashinfer_step(device, world, runtime=runtime)

    torch.testing.assert_close(poisoned, clean, rtol=0, atol=0)


def test_the_plans_are_not_in_the_scratch(device):
    """The failure this split exists to stop, as the boot found it.

    A step ran, another consumer wrote over the workspace they share, and the
    next step read a schedule that was no longer there. What has to be true is
    that the bytes the plans live in are not bytes anybody else writes.
    """
    q, compressed, k, v, block_table, token_to_req, positions, lengths, gate, _ = (
        _world(device, seed=22)
    )
    runtime = _runtime(device)
    # The caller's own buffers, taken once, so what is measured below is the
    # step and not the test allocating a route to hand it.
    route = torch.empty(ROWS, ROUTE_WIDTH, dtype=torch.int32, device=device)
    out = torch.empty(ROWS, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    def step():
        runtime.run_selection(
            q,
            compressed,
            block_table,
            token_to_req,
            positions,
            lengths,
            out_route=route,
        )
        runtime.run_attention(
            q,
            k,
            v,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gate,
            out=out,
        )

    step()
    torch.cuda.synchronize()
    first = out.clone()

    persistent = _poison(runtime)
    before = _stamp(persistent)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()

    step()
    torch.cuda.synchronize()

    assert _stamp(persistent) == before, "the plans were overwritten"
    assert torch.cuda.memory_allocated() == allocated, "the step allocated"
    torch.testing.assert_close(out, first, rtol=0, atol=0)


def test_a_captured_step_replays_after_the_scratch_is_overwritten(device):
    """A graph replays pointers, and the scratch it points at is shared.

    Overwriting the scratch between capture and replay is what the step after
    a captured one does; the replay has to rewrite it and come out the same as
    running eagerly.
    """
    world = _world(device, seed=23)
    q, compressed, k, v, block_table, token_to_req, positions, lengths, gate, _ = world
    runtime = _runtime(device)
    route = torch.empty(ROWS, ROUTE_WIDTH, dtype=torch.int32, device=device)
    out = torch.empty(ROWS, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    def step():
        runtime.run_selection(
            q,
            compressed,
            block_table,
            token_to_req,
            positions,
            lengths,
            out_route=route,
        )
        runtime.run_attention(
            q,
            k,
            v,
            route=route,
            block_table=block_table,
            token_to_req=token_to_req,
            output_gate=gate,
            out=out,
        )

    step()
    torch.cuda.synchronize()
    expected = out.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    _poison(runtime)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "region",
    ["padded_q", "padded_out", "route", "mask", "float_workspace", "selection"],
)
def test_every_transient_region_is_rewritten_before_it_is_read(device, region):
    """One region at a time, so a survivor cannot hide behind the others.

    Filling the whole scratch at once says the step is safe; filling one region
    says which. A region that is written at construction and only read
    afterwards passes the first and fails this.
    """
    world = _world(device, seed=24)
    runtime = _runtime(device)
    _route, clean = _flashinfer_step(device, world, runtime=runtime)
    clean = clean.clone()

    attention = runtime._attention
    views = {
        "padded_q": attention._padded_q,
        "padded_out": attention._padded_out,
        "route": attention._route_base,
        "mask": attention._mask_base,
        "float_workspace": attention._float_workspace,
        "selection": runtime._selection._bound_workspace,
    }
    view = views[region]
    if view.numel() == 0:
        pytest.skip(f"{region} is empty for this geometry")
    view.view(torch.uint8).fill_(0xA5)

    _route, poisoned = _flashinfer_step(device, world, runtime=runtime)
    torch.testing.assert_close(poisoned, clean, rtol=0, atol=0)
