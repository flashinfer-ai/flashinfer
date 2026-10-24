"""Prepared CSA compressor numerical, replay, and 64-bit addressing contracts."""

from __future__ import annotations

import pytest
import torch

from b12x.attention import mla_compress as op
from b12x.attention.mla_compress import _impl
from b12x.attention.mla_compress.reference import streaming_reference
from b12x.preparation import PreparedCall, PreparationSession
from b12x.testing.reference.helpers import require_b12x


def _device():
    device = require_b12x()
    pytest.importorskip("cutlass")
    pytest.importorskip("cuda.bindings.driver")
    return device


def _prepared(ratio=2, states=8, *, sparse_pending=False):
    device = _device()
    caps = op.Caps(
        device=device, max_tokens=12, max_requests=4, max_states=states, ratio=ratio
    )
    declaration = op.plan(caps)
    owned = dict(
        values=torch.empty(
            (12, 512),
            dtype=torch.float32 if ratio == 2 else torch.bfloat16,
            device=device,
        ),
        weight=torch.linspace(-1.1, 1.3, 512, device=device),
        query_start_loc=torch.zeros(5, dtype=torch.int32, device=device),
        positions=torch.zeros(4, dtype=torch.int64, device=device),
        state_ids=torch.full((4,), -1, dtype=torch.int64, device=device),
        destination_slots=torch.arange(12, dtype=torch.int64, device=device) + 2**35,
        live_counts=torch.zeros(2, dtype=torch.int32, device=device),
        out=torch.empty((12, 512), dtype=torch.bfloat16, device=device),
        emitted=torch.empty(12, dtype=torch.bool, device=device),
        emitted_slots=torch.empty(12, dtype=torch.int64, device=device),
    )
    if ratio == 2:
        owned.update(
            gates=torch.empty((12, 512), dtype=torch.float32, device=device),
            pending_values=torch.empty(
                (states, 512), dtype=torch.float32, device=device
            ),
            pending_gates=torch.empty(
                (states, 512), dtype=torch.float32, device=device
            ),
            pending_position=(
                torch.empty((states,), dtype=torch.int64, device=device)
                if sparse_pending
                else torch.full((states,), -1, dtype=torch.int64, device=device)
            ),
        )

    def prepare_call(state):
        # Prime an actual pair, then restore only the caller-owned rows touched.
        snapshots = {
            name: owned[name].clone()
            for name in (
                "values",
                "gates",
                "query_start_loc",
                "positions",
                "state_ids",
                "live_counts",
                "out",
                "emitted",
                "emitted_slots",
            )
            if name in owned
        }
        pending = None
        if ratio == 2:
            pending = (
                owned["pending_values"][0].clone(),
                owned["pending_gates"][0].clone(),
                owned["pending_position"][0].clone(),
            )
        owned["values"].fill_(1)
        owned["query_start_loc"][:] = torch.tensor(
            [0, 2, 2, 2, 2], dtype=torch.int32, device=device
        )
        owned["positions"][:] = torch.tensor(
            [0, -1, -1, -1], dtype=torch.int64, device=device
        )
        owned["state_ids"][:] = torch.tensor(
            [0, -1, -1, -1], dtype=torch.int64, device=device
        )
        owned["live_counts"][:] = torch.tensor([2, 1], dtype=torch.int32, device=device)
        if ratio == 2:
            owned["gates"].zero_()
        binding = state.bind(**owned)

        def restore():
            for name, value in snapshots.items():
                owned[name].copy_(value)
            if pending is not None:
                owned["pending_values"][0].copy_(pending[0])
                owned["pending_gates"][0].copy_(pending[1])
                owned["pending_position"][0].copy_(pending[2])

        return PreparedCall(run=lambda: state.run(binding), restore=restore)

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    result = session.prepare(
        (declaration.request(name="mla-compress", prepare_call=prepare_call),)
    )
    return session, result, result.plans["mla-compress"], owned


def _prepare(binding, lengths, positions, ids, seed):
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    n = starts[-1]
    binding.query_start_loc.copy_(
        torch.tensor(
            starts + [n] * (5 - len(starts)),
            dtype=torch.int32,
            device=binding.values.device,
        )
    )
    binding.positions.copy_(
        torch.tensor(
            positions + [-1] * (4 - len(positions)),
            dtype=torch.int64,
            device=binding.values.device,
        )
    )
    binding.state_ids.copy_(
        torch.tensor(
            ids + [-1] * (4 - len(ids)), dtype=torch.int64, device=binding.values.device
        )
    )
    binding.live_counts.copy_(
        torch.tensor([n, len(lengths)], dtype=torch.int32, device=binding.values.device)
    )
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((12, 512), generator=generator).to(binding.values.dtype)
    binding.values.copy_(values)
    gates = None
    if binding.gates is not None:
        gates = torch.randn((12, 512), generator=generator) * 30
        gates[:, ::2] *= -1
        binding.gates.copy_(gates)
    return dict(
        values=values,
        gates=gates,
        weight=binding.weight.cpu(),
        starts=starts,
        positions=positions,
        state_ids=ids,
        slots=binding.destination_slots.cpu(),
        ratio=binding._state.caps.ratio,
    )


def _assert_result(binding, expected):
    torch.testing.assert_close(binding.out.cpu(), expected[0], rtol=0.012, atol=0.016)
    torch.testing.assert_close(binding.emitted.cpu(), expected[1], rtol=0, atol=0)
    torch.testing.assert_close(binding.emitted_slots.cpu(), expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1, 2])
def test_prepared_compressor_replays_dynamic_counts_against_streaming_oracle(ratio):
    session, result, plan, owned = _prepared(ratio)
    graph = None
    try:
        binding = op.bind(plan, **owned)
        _prepare(binding, [2], [0], [1], 4)
        op.run(binding)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            op.run(binding)
        state = {}
        for seed, (lengths, positions, ids) in enumerate(
            (([3, 1], [0, 0], [4, 2]), ([1, 2, 1], [1, 3, 7], [2, 4, -1]), ([], [], []))
        ):
            kwargs = _prepare(binding, lengths, positions, ids, seed + 20)
            expected = streaming_reference(**kwargs, state=state)
            binding.out.fill_(float("nan"))
            binding.emitted.fill_(True)
            binding.emitted_slots.fill_(42)
            graph.replay()
            torch.cuda.synchronize(binding.values.device)
            _assert_result(binding, expected)
    finally:
        if graph is not None:
            graph.reset()
        result.close()
        session.close()


@pytest.mark.parametrize("ratio", [1, 2])
def test_prepared_compressor_compiles_through_prepared_plan(ratio):
    session, result, plan, owned = _prepared(ratio)
    try:
        binding = op.bind(plan, **owned)
        kwargs = _prepare(binding, [2], [0], [0], 91)
        expected = streaming_reference(**kwargs, state={})
        compiled_run = torch.compile(lambda: op.run(binding), fullgraph=True)
        compiled_run()
        torch.cuda.synchronize(binding.values.device)
        _assert_result(binding, expected)
    finally:
        result.close()
        session.close()


def test_prepared_compressor_high_state_id_uses_int64_offsets():
    device = _device()
    state_id = 2**31 // 512 + 1
    # The large state arrays are virtual/uninitialized; only the addressed row
    # participates in this probe, so it never clones or initializes the pool.
    session, result, plan, owned = _prepared(states=state_id + 2, sparse_pending=True)
    try:
        binding = op.bind(plan, **owned)
        binding.pending_position[state_id].fill_(-1)
        state = {}
        for seed, position in enumerate((0, 1)):
            kwargs = _prepare(binding, [1], [position], [state_id], seed + 50)
            expected = streaming_reference(**kwargs, state=state)
            op.run(binding)
            _assert_result(binding, expected)
        assert binding.pending_position[state_id].item() == -1
    finally:
        result.close()
        session.close()
