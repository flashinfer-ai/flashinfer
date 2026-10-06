"""Prepared Engram native hashing and lookup numerical/capture contracts."""

from __future__ import annotations

import pytest
import torch

from b12x.preparation import PreparedCall, PreparationSession
from b12x.sequence import engram
from b12x.sequence.engram.reference import hash_reference
from b12x.sequence.engram import _impl
from b12x.testing.reference.helpers import require_b12x


@pytest.mark.parametrize("resident_scales", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_disk_lookup_transaction_recovers_after_read_failure(
    resident_scales,
    prefetch,
    monkeypatch,
):
    """Host lifecycle coverage for synchronous reads and explicit close."""
    import sys
    from contextlib import contextmanager
    from types import SimpleNamespace
    from b12x.sequence._shared import disk_table
    from b12x.sequence.engram import _disk

    class Cache:
        def __init__(self, **kwargs):
            self.weight = torch.zeros((72, 256), dtype=torch.uint8)
            self.scale = torch.zeros((72, 8), dtype=torch.uint8)
            self.active = False
            self.rows = None
            self.fail_read = False
            self.closed = False

        @contextmanager
        def transaction(self):
            assert not self.active and not self.closed
            self.active = True
            try:
                yield
            finally:
                self.active = False

        def read_rows(self, ids, count):
            assert self.active
            if self.fail_read:
                raise OSError("injected read failure")
            self.rows = ids.view(-1)[:count].clone()

        def close(self):
            assert not self.active
            self.closed = True

    scale_closed = []

    def mapped_scales(shape, dtype, device):
        host = torch.zeros(shape, dtype=dtype)
        return SimpleNamespace(
            host_view=host,
            device_view=host,
            close=lambda: scale_closed.append(True),
        )

    monkeypatch.setattr(disk_table, "DiskRowCache", Cache)
    monkeypatch.setattr(disk_table, "MappedHostAllocation", mapped_scales)
    monkeypatch.setattr(_disk, "_require_disk_eager", lambda _: None)
    state = object.__new__(_impl._State)
    for name, value in dict(
        caps=SimpleNamespace(device=torch.device("cuda:0"), max_tokens=3),
        table_rows=6,
        shard_start=0,
        shard_end=6,
        shard_rows=6,
        compact_rows=True,
        resident_scales=resident_scales,
    ).items():
        object.__setattr__(state, name, value)
    table = engram.DiskTable(state, resident_scales=resident_scales, prefetch=prefetch)
    cache = table._cache
    ids = torch.arange(72).reshape(3, 24)
    out = torch.full((3,), -1, dtype=torch.int64)
    binding = engram.LookupBinding(
        plan=SimpleNamespace(handle=1),
        _state=state,
        disk_table=table,
        weight=table.weight,
        scale_bytes=table.scale_bytes,
        hash_ids=ids,
        num_tokens=torch.tensor([2], dtype=torch.int32),
        out=out,
    )

    def lookup(handle, weight, scales, hash_ids, num_tokens, result, count, clear_tail):
        assert cache.active and clear_tail is False
        result[:count].copy_(cache.rows.view(count, 24)[:, 0])

    monkeypatch.setitem(
        sys.modules,
        "b12x.sequence.engram._kernels",
        SimpleNamespace(lookup_op=lookup),
    )
    try:
        table.prefetch(binding, token_count=2)
        assert cache.rows is None and not table.prefetch_pending
        engram.run_lookup(binding, token_count=2, clear_tail=False)
        torch.testing.assert_close(out, torch.tensor([0, 24, -1]))
        assert not cache.active and not table.prefetch_pending

        cache.fail_read = True
        with pytest.raises(OSError, match="injected"):
            engram.run_lookup(binding, token_count=2, clear_tail=False)
        assert not cache.active and not table.prefetch_pending
        table.abort_prefetch()
        cache.fail_read = False
        ids.add_(1)
        engram.run_lookup(binding, token_count=2, clear_tail=False)
        torch.testing.assert_close(out, torch.tensor([1, 25, -1]))
        assert not cache.active
    finally:
        table.close()
    assert cache.closed and bool(scale_closed) == resident_scales
    with pytest.raises(RuntimeError, match="closed"):
        engram.run_lookup(binding, token_count=2, clear_tail=False)


def _declaration(device, *, tokens=7, rank=0, tp=2, base=101, invocation=None):
    geometry = engram.build_geometry(base_table_size=base, compressed_vocab_size=32)
    return engram.plan(
        engram.Caps(
            device=device,
            max_tokens=tokens,
            max_seqs=3,
            max_requests=4,
            vocab_size=32,
            layer_id=1,
            tp_rank=rank,
            tp_size=tp,
        ),
        token_map=list(range(32)),
        geometry=geometry,
        invocation={} if invocation is None else invocation,
    )


def _hash_inputs(caps):
    device = caps.device
    return dict(
        scratch=torch.empty((4096,), dtype=torch.uint8, device=device),
        token_ids=torch.zeros(caps.max_tokens, dtype=torch.int64, device=device),
        token_mask=torch.ones(caps.max_tokens, dtype=torch.bool, device=device),
        query_start_loc=torch.tensor(
            [0, caps.max_tokens, caps.max_tokens, caps.max_tokens],
            dtype=torch.int32,
            device=device,
        ),
        request_slots=torch.tensor([2, 0, 3], dtype=torch.int32, device=device),
        committed_history=torch.tensor(
            [[-1, -1, -1], [4, 5, 6], [7, -1, 9], [10, 11, 12]],
            dtype=torch.int64,
            device=device,
        ),
        num_seqs=torch.tensor([1], dtype=torch.int32, device=device),
        num_tokens=torch.tensor([caps.max_tokens], dtype=torch.int32, device=device),
        hash_ids=torch.empty((caps.max_tokens, 24), dtype=torch.int64, device=device),
    )


def _prepared_hash(device, *, tokens=7):
    declaration = _declaration(device, tokens=tokens)
    inputs = _hash_inputs(
        engram.Caps(
            device=device,
            max_tokens=tokens,
            max_seqs=3,
            max_requests=4,
            vocab_size=32,
            layer_id=1,
            tp_rank=0,
            tp_size=2,
        )
    )

    # The private binder is intentionally used only by the pre-publication callback.
    def prepare_call(state):
        spec = state.scratch_specs()[0]
        inputs["scratch"] = torch.empty(
            spec.shape, dtype=spec.dtype, device=spec.device
        )
        before = inputs["hash_ids"].clone()
        binding = _impl._bind_state(state, **inputs)
        return PreparedCall(
            run=lambda: state.run(binding, tokens),
            restore=lambda: inputs["hash_ids"].copy_(before),
        )

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    result = session.prepare(
        (declaration.request(name="engram-hash", prepare_call=prepare_call),)
    )
    return session, result, declaration, inputs


def test_prepared_hash_replay_matches_signed_reference_and_preserves_history():
    device = require_b12x()
    session, result, plan, inputs = _prepared_hash(device)
    try:
        binding = engram.bind(plan, **inputs)
        history = binding.committed_history.clone()
        cases = [
            (
                [1, 13, 14, 15, 16, 17, 18],
                [True, True, False, True, True, True, True],
                [0, 3, 3, 7],
                [2, 0, 3],
            ),
            ([19, 1, 20, 21, 22], [True] * 5, [0, 1, 4, 5], [3, 2, 0]),
        ]
        launch = torch.compile(lambda: engram.run(binding), fullgraph=True)
        launch()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            launch()
        for tokens, mask, starts, slots in cases:
            count = len(tokens)
            binding.token_ids.zero_()
            binding.token_ids[:count].copy_(torch.tensor(tokens, device=device))
            binding.token_mask.zero_()
            binding.token_mask[:count].copy_(torch.tensor(mask, device=device))
            binding.query_start_loc.copy_(
                torch.tensor(starts, dtype=torch.int32, device=device)
            )
            binding.request_slots.copy_(
                torch.tensor(slots, dtype=torch.int32, device=device)
            )
            binding.num_seqs.fill_(len(slots))
            binding.num_tokens.fill_(count)
            graph.replay()
            expected = hash_reference(
                tokens,
                mask,
                starts,
                slots,
                history.cpu().tolist(),
                list(range(32)),
                binding._state.geometry,
                1,
            )
            torch.testing.assert_close(
                binding.hash_ids[:count].cpu(), expected, rtol=0, atol=0
            )
            assert binding.hash_ids[count:].eq(-1).all()
            torch.testing.assert_close(
                binding.committed_history, history, rtol=0, atol=0
            )
    finally:
        del graph
        result.close()
        session.close()


def test_prepared_lookup_accepts_e8m0_bytes_and_exact_row_shards():
    device = require_b12x()
    declaration = _declaration(
        device,
        tokens=2,
        rank=1,
        tp=3,
        invocation={"operation": "lookup", "compact_rows": False},
    )
    weight = torch.empty(
        ((declaration.query.table_rows + 2) // 3, 256),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    scales = torch.empty((weight.shape[0], 8), dtype=torch.uint8, device=device)
    ids = torch.full((2, 24), -1, dtype=torch.int64, device=device)
    out = torch.empty((2, 6144), dtype=torch.bfloat16, device=device)
    count = torch.tensor([1], dtype=torch.int32, device=device)
    row = 1
    weight.zero_()
    weight[row].fill_(2)
    scale = torch.tensor(
        [0, 1, 126, 127, 128, 253, 255, 127], dtype=torch.uint8, device=device
    )
    scales.zero_()
    scales[row].copy_(scale)
    shard_rows = weight.shape[0]
    ids[0, ::3] = shard_rows + row

    def prepare_call(state):
        before = out.clone()
        binding = _impl._bind_lookup_state(
            state,
            weight=weight,
            scales=scales.view(torch.float8_e8m0fnu),
            hash_ids=ids,
            num_tokens=count,
            out=out,
        )
        return PreparedCall(
            run=lambda: state.run_lookup(binding, 1, clear_tail=True),
            restore=lambda: out.copy_(before),
        )

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    result = session.prepare(
        (declaration.request(name="engram-lookup", prepare_call=prepare_call),)
    )
    try:
        binding = engram.bind_lookup(
            declaration,
            weight=weight,
            scales=scales.view(torch.float8_e8m0fnu),
            hash_ids=ids,
            num_tokens=count,
            out=out,
        )
        torch.compile(
            lambda: engram.run_lookup(binding, token_count=1), fullgraph=True
        )()
        expected = (
            2 * scale.view(torch.float8_e8m0fnu).float().repeat_interleave(32)
        ).to(torch.bfloat16)
        torch.testing.assert_close(
            out[0].view(24, 256)[::3],
            expected.expand(8, -1),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        assert torch.count_nonzero(out[0].view(24, 256)[1::3]) == 0
    finally:
        result.close()
        session.close()


def test_run_lookups_overlaps_disk_reads_and_matches_sequential(tmp_path):
    """Batched lookups over two disk tables equal the tables read one by one."""
    from contextlib import ExitStack

    device = require_b12x()
    plan = _declaration(
        device,
        tokens=3,
        rank=1,
        tp=3,
        invocation={
            "operation": "lookup",
            "compact_rows": True,
            "resident_scales": False,
        },
    )
    rows = plan.query.table_rows
    tables_data = []
    for k in range(2):
        weights = (
            torch.arange(rows * 256, dtype=torch.float32)
            .add(k * 5)
            .remainder(15)
            .sub(7)
        )
        weights = weights.reshape(rows, 256).to(torch.float8_e4m3fn)
        scales = (
            torch.arange(rows * 8)
            .add(k)
            .remainder(5)
            .add(125)
            .to(torch.uint8)
            .reshape(rows, 8)
        )
        paths = [tmp_path / f"weights{k}.bin", tmp_path / f"scales{k}.bin"]
        for path, data in zip(paths, (weights, scales)):
            path.write_bytes(bytes(4093) + data.view(torch.uint8).numpy().tobytes())
        tables_data.append((weights, scales, paths))
    ids = [torch.full((3, 24), -1, dtype=torch.int64, device=device) for _ in range(2)]
    count = torch.tensor([3], dtype=torch.int32, device=device)
    outs = [
        torch.full((3, 6144), 73, dtype=torch.bfloat16, device=device) for _ in range(2)
    ]
    states = []
    with ExitStack() as resources:
        session = resources.enter_context(
            PreparationSession(device=device, autotune=False, compile_workers=2)
        )

        def prepare_call(state):
            states.append(state)
            return PreparedCall(run=lambda: None)

        resources.enter_context(
            session.prepare(
                (plan.request(name="engram-disk-batch", prepare_call=prepare_call),)
            )
        )
        tables = []
        for weights, scales, paths in tables_data:
            table = engram.DiskTable(states[-1], queue_depth=4)
            resources.callback(table.close)
            for scale, path in zip((False, True), paths):
                table.add_shard(0, str(path), 4093, scale=scale)
            tables.append(table)
        bindings = [
            engram.bind_lookup(plan, disk_table=t, hash_ids=i, num_tokens=count, out=o)
            for t, i, o in zip(tables, ids, outs)
        ]
        start, end = tables[0].state.shard_start, min(tables[0].state.shard_end, rows)
        cases = (
            torch.tensor(
                [start, end - 1, start, start - 1, end, -1, start + 3, start + 17],
                device=device,
            )
            .repeat(9)
            .reshape(3, 24)
        )
        for live in (3, 1, 2):
            for k in range(2):
                ids[k].copy_(cases.roll(k + live, dims=1))
                outs[k].fill_(73)
            count.fill_(live)
            engram.run_lookups(bindings, [live, live])
            torch.cuda.synchronize(device)
            for k, (weights, scales, _) in enumerate(tables_data):
                expected = torch.zeros((3, 24, 256), dtype=torch.bfloat16)
                cpu_ids = ids[k].cpu()
                valid = (cpu_ids[:live] >= start) & (cpu_ids[:live] < end)
                selected = cpu_ids[:live][valid]
                expected[:live][valid] = (
                    weights.float()[selected]
                    * scales[selected]
                    .view(torch.float8_e8m0fnu)
                    .float()
                    .repeat_interleave(32, dim=1)
                ).to(torch.bfloat16)
                torch.testing.assert_close(
                    outs[k].cpu(), expected.flatten(1), rtol=0, atol=0
                )


@torch.inference_mode()
@pytest.mark.parametrize("resident_scales", [False, True])
def test_disk_lookup_matches_varied_rows_with_graph_replay_and_stream_reuse(
    resident_scales,
    tmp_path,
    monkeypatch,
):
    from contextlib import ExitStack
    from b12x._lib.runtime_control import kernel_resolution_guard

    device = require_b12x()
    plan = _declaration(
        device,
        tokens=3,
        rank=1,
        tp=3,
        invocation={
            "operation": "lookup",
            "compact_rows": True,
            "resident_scales": resident_scales,
        },
    )
    rows = plan.query.table_rows
    weights = (
        torch.arange(rows * 256, dtype=torch.float32)
        .remainder(15)
        .sub(7)
        .reshape(rows, 256)
        .to(torch.float8_e4m3fn)
    )
    scales = (
        torch.arange(rows * 8).remainder(5).add(125).to(torch.uint8).reshape(rows, 8)
    )
    paths = [tmp_path / "weights.bin", tmp_path / "scales.bin"]
    for path, data in zip(paths, (weights, scales)):
        path.write_bytes(bytes(4093) + data.view(torch.uint8).numpy().tobytes())
    ids = torch.full((3, 24), -1, dtype=torch.int64, device=device)
    count = torch.tensor([3], dtype=torch.int32, device=device)
    out = torch.full((3, 6144), 73, dtype=torch.bfloat16, device=device)
    tables = []
    with ExitStack() as resources:
        session = resources.enter_context(
            PreparationSession(device=device, autotune=False, compile_workers=2)
        )

        def prepare_call(state):
            table = engram.DiskTable(
                state, queue_depth=2, resident_scales=resident_scales
            )
            resources.callback(table.close)
            for scale, path in zip((False, True), paths):
                table.add_shard(0, str(path), 4093, scale=scale)
            tables.append(table)
            trial = _impl._bind_lookup_state(
                state, disk_table=table, hash_ids=ids, num_tokens=count, out=out
            )

            def run():
                with table._cache.transaction():
                    table._cache.read_rows(ids, 72)
                    state.run_lookup(trial, 3, clear_tail=True)

            return PreparedCall(run=run)

        resources.enter_context(
            session.prepare(
                (plan.request(name="engram-disk", prepare_call=prepare_call),)
            )
        )
        table = tables[-1]
        binding = engram.bind_lookup(
            plan, disk_table=table, hash_ids=ids, num_tokens=count, out=out
        )
        start, end = table.state.shard_start, min(table.state.shard_end, rows)
        cases = (
            torch.tensor(
                [start, end - 1, start, start - 1, end, -1, start + 3, start + 17],
                device=device,
            )
            .repeat(9)
            .reshape(3, 24)
        )
        ids.copy_(cases)
        engram.run_lookup(binding)
        consumed = torch.empty_like(out)
        graph = torch.cuda.CUDAGraph()
        resources.callback(graph.reset)
        with session.capture(), torch.cuda.graph(graph):
            torch.mul(out, 2, out=consumed)
        pointers = out.data_ptr(), table.weight.data_ptr(), table.scale_bytes.data_ptr()
        streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        for iteration, live in enumerate([3, 1, 0, 2, 3]):
            with (
                torch.cuda.stream(streams[iteration % 2]),
                kernel_resolution_guard("disk lookup reuses prepared programs"),
            ):
                count.fill_(live)
                ids.copy_(cases.roll(iteration, dims=1))
                out.fill_(73)
                engram.run_lookup(binding, token_count=live)
                graph.replay()
                streams[iteration % 2].synchronize()
            expected = torch.zeros((3, 24, 256), dtype=torch.bfloat16)
            cpu_ids = ids.cpu()
            valid = (cpu_ids[:live] >= start) & (cpu_ids[:live] < end)
            selected = cpu_ids[:live][valid]
            expected[:live][valid] = (
                weights.float()[selected]
                * scales[selected]
                .view(torch.float8_e8m0fnu)
                .float()
                .repeat_interleave(32, dim=1)
            ).to(torch.bfloat16)
            torch.testing.assert_close(out.cpu(), expected.flatten(1), rtol=0, atol=0)
            torch.testing.assert_close(
                consumed.cpu(), expected.flatten(1) * 2, rtol=0, atol=0
            )
            assert pointers == (
                out.data_ptr(),
                table.weight.data_ptr(),
                table.scale_bytes.data_ptr(),
            )
        with monkeypatch.context() as patch:
            patch.setattr(
                table._cache,
                "read_rows",
                lambda *a: pytest.fail("consumer replay cannot read disk"),
            )
            graph.replay()
            torch.cuda.synchronize(device)
        if table._cache._gds:
            assert table._cache.weight_host is None
            assert (table._scale_owner is not None) == resident_scales
