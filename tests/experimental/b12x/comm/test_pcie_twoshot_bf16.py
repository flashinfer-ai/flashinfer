"""Correctness for the single-rounding BF16 PCIe two-shot collectives.

Run with torchrun on 4 GPUs:

    NCCL_ALGO=Ring python -m torch.distributed.run --nproc-per-node=4 \
        tests/comm/test_pcie_twoshot_bf16.py

Every all-reduce is checked against the exact fp32 sum of the gathered
inputs: the kernel accumulates in fp32 in a fixed rank order and rounds once,
so each output element must lie within one bf16 rounding of the exact sum,
same-input eager calls must be bitwise identical, and graph replay must consume
mutated live inputs.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.comm.pcie import PCIeTwoShotBF16
from b12x.comm.pcie._twoshot_bf16_cute import (
    get_twoshot_bf16_allreduce_launcher,
    get_twoshot_bf16_launcher,
)
from b12x.comm.pcie.pcie_twoshot_bf16 import _make_layout

ROW_ELEMS = int(os.getenv("B12X_TEST_TWOSHOT_BF16_ROW_ELEMS", "4096"))
MAX_ROWS = int(os.getenv("B12X_TEST_TWOSHOT_BF16_MAX_ROWS", "512"))
ROWS = (8, 16, 32, 64, 96, 128, 192, 256)


@pytest.mark.parametrize("shape", ((128, 1024), (32, 4096), (16, 8192), (131072,)))
def test_prepared_all_reduce_counts_native_rows(monkeypatch, shape) -> None:
    """Prepared all-reduce accepts the same contiguous layouts as the public API."""
    import b12x.comm.pcie.pcie_twoshot_bf16 as twoshot_bf16

    payload = torch.empty(shape, dtype=torch.bfloat16)
    out = torch.empty_like(payload)
    launches = []
    layout = _make_layout(32, 4096, 4)
    runtime = SimpleNamespace(
        world_size=4,
        rank=0,
        row_elems=4096,
        device=torch.device("cpu"),
        _pack_stride=layout.pack_stride,
        _reduced_offset=layout.reduced_offset,
        _slot_bytes=layout.slot_bytes,
        _capture_context_depth=0,
        _device_slot_selection=False,
        _device_slot_bias=0,
        _slot=0,
        _staging_ptrs=((1, 2, 3, 4), (5, 6, 7, 8)),
        _signal_ptrs=(9, 10, 11, 12),
    )
    state = SimpleNamespace(
        query=SimpleNamespace(
            call={"operation": "all_reduce", "threads": 512, "block_limit": 64}
        ),
        launcher=lambda *_: lambda *args: launches.append(args),
    )
    monkeypatch.setattr(twoshot_bf16, "_is_current_stream_capturing", lambda _: False)
    PCIeTwoShotBF16._launch_prepared(
        runtime, payload, None, out, state=state, threads=512, block_limit=64
    )
    assert len(launches) == 1
    assert launches[0][0] == payload.data_ptr()
    assert launches[0][3] == out.data_ptr()
    assert launches[0][-2] == 8  # 32 native rows split across four ranks.


def test_layout_is_tp4_only_and_scales_with_capacity() -> None:
    for world_size in (2, 8):
        with pytest.raises(ValueError, match="supports only world size 4"):
            _make_layout(64, ROW_ELEMS, world_size)

    base = _make_layout(64, ROW_ELEMS, 4)
    assert base.pack_stride > 0 and base.slot_bytes > 0
    assert base.reduced_offset > 0 and base.slab_bytes >= 2 * base.slot_bytes
    taller = _make_layout(128, ROW_ELEMS, 4)
    assert taller.slab_bytes > base.slab_bytes
    wider = _make_layout(64, 2 * ROW_ELEMS, 4)
    assert wider.slab_bytes > base.slab_bytes


@pytest.mark.parametrize("world_size", (2, 8))
def test_private_launchers_reject_unsupported_world_sizes(world_size: int) -> None:
    with pytest.raises(ValueError, match="require world size 4"):
        get_twoshot_bf16_launcher(
            "reduce_scatter",
            world_size,
            0,
            False,
            0,
            512,
            ROW_ELEMS,
            0,
        )
    with pytest.raises(ValueError, match="require world size 4"):
        get_twoshot_bf16_allreduce_launcher(
            world_size,
            0,
            False,
            0,
            512,
            ROW_ELEMS,
            0,
        )


def test_graph_capture_requires_caller_owned_outputs(monkeypatch) -> None:
    import b12x.comm.pcie.pcie_twoshot_bf16 as twoshot_bf16

    runtime = object.__new__(PCIeTwoShotBF16)
    runtime.rank = 0
    runtime.world_size = 4
    runtime.device = torch.device("cpu")
    runtime.max_rows = 4
    runtime.row_elems = 8
    runtime._closed = False
    monkeypatch.setattr(twoshot_bf16, "_device_guard", lambda _device: nullcontext())
    monkeypatch.setattr(
        twoshot_bf16,
        "_is_current_stream_capturing",
        lambda _device: True,
    )

    payload = torch.empty((4, 8), dtype=torch.bfloat16)
    shard = torch.empty((1, 8), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="caller-owned preallocated output"):
        runtime.reduce_scatter(payload)
    with pytest.raises(RuntimeError, match="caller-owned preallocated output"):
        runtime.all_gather(shard)
    with pytest.raises(RuntimeError, match="caller-owned preallocated output"):
        runtime.all_reduce(payload)


def _payload(seed: int, rows: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, ROW_ELEMS, generator=gen, dtype=torch.float32) * 3.0
    return x.to(device=device, dtype=torch.bfloat16)


def _exact_sum(x: torch.Tensor, world: int) -> torch.Tensor:
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    return sum(g.float() for g in gathered)


def _assert_one_rounding(out: torch.Tensor, exact: torch.Tensor) -> None:
    err = (out.float() - exact).abs()
    bound = exact.abs() * 2.0**-8 + 1e-5
    worst = (err - bound).max().item()
    assert bool((err <= bound).all()), (
        f"output exceeds one bf16 rounding by {worst:.3e}"
    )


def _check_all_reduce(
    pool: PCIeTwoShotBF16, rank: int, world: int, rows: int, step: int
) -> None:
    x = _payload(1000 * step + 7 * rows + rank, rows, pool.device)
    assert pool.accepts(x)
    exact = _exact_sum(x, world)
    out = pool.all_reduce(x)
    torch.cuda.synchronize()
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    _assert_one_rounding(out, exact)
    again = pool.all_reduce(x)
    torch.cuda.synchronize()
    assert torch.equal(out, again), "two-shot all-reduce must be deterministic"
    # The bf16 NCCL ring rounds after every hop; the two-shot rounds once.
    ref = x.clone()
    dist.all_reduce(ref)
    assert (out.float() - exact).abs().max() <= (ref.float() - exact).abs().max() + 1e-5


def _check_graph_capture(pool: PCIeTwoShotBF16, rank: int, world: int) -> None:
    rows = min(64, MAX_ROWS)
    rows -= rows % world
    assert rows > 0
    local_rows = rows // world
    all_reduce_input = _payload(4242 + rank, rows, pool.device)
    reduce_scatter_input = _payload(4342 + rank, rows, pool.device)
    all_reduce_out = torch.empty_like(all_reduce_input)
    reduce_scatter_out = torch.empty(
        local_rows,
        ROW_ELEMS,
        dtype=torch.bfloat16,
        device=pool.device,
    )
    all_gather_out = torch.empty_like(reduce_scatter_input)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with pool.capture(operations=("reduce_scatter", "all_gather")):
        with torch.cuda.stream(stream):
            for _ in range(3):
                pool.all_reduce(all_reduce_input, out=all_reduce_out)
                pool.reduce_scatter(
                    reduce_scatter_input,
                    out=reduce_scatter_out,
                )
                pool.all_gather(reduce_scatter_out, out=all_gather_out)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            assert (
                pool.all_reduce(all_reduce_input, out=all_reduce_out) is all_reduce_out
            )
            assert (
                pool.reduce_scatter(
                    reduce_scatter_input,
                    out=reduce_scatter_out,
                )
                is reduce_scatter_out
            )
            assert (
                pool.all_gather(reduce_scatter_out, out=all_gather_out)
                is all_gather_out
            )
    torch.cuda.synchronize()
    dist.barrier()
    output_addresses = (
        all_reduce_out.data_ptr(),
        reduce_scatter_out.data_ptr(),
        all_gather_out.data_ptr(),
    )
    allocator_reports_request_count = torch.cuda.get_allocator_backend() == "native"
    for replay_step in range(3):
        all_reduce_input.copy_(
            _payload(4442 + 100 * replay_step + rank, rows, pool.device)
        )
        reduce_scatter_input.copy_(
            _payload(4542 + 100 * replay_step + rank, rows, pool.device)
        )
        exact_all_reduce = _exact_sum(all_reduce_input, world)
        exact_reduce_scatter = _exact_sum(reduce_scatter_input, world)
        all_reduce_out.fill_(float("nan"))
        reduce_scatter_out.fill_(float("nan"))
        all_gather_out.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier()
        allocated_before_replay = torch.cuda.memory_allocated(pool.device)
        allocation_count_before_replay = (
            int(torch.cuda.memory_stats(pool.device)["allocation.all.allocated"])
            if allocator_reports_request_count
            else None
        )
        graph.replay()
        torch.cuda.synchronize()
        if allocation_count_before_replay is not None:
            assert (
                int(torch.cuda.memory_stats(pool.device)["allocation.all.allocated"])
                == allocation_count_before_replay
            )
        assert torch.cuda.memory_allocated(pool.device) == allocated_before_replay
        dist.barrier()
        assert (
            all_reduce_out.data_ptr(),
            reduce_scatter_out.data_ptr(),
            all_gather_out.data_ptr(),
        ) == output_addresses
        _assert_one_rounding(all_reduce_out, exact_all_reduce)
        expected_shard = exact_reduce_scatter[
            rank * local_rows : (rank + 1) * local_rows
        ]
        _assert_one_rounding(reduce_scatter_out, expected_shard)
        reference_shards = [torch.empty_like(reduce_scatter_out) for _ in range(world)]
        dist.all_gather(reference_shards, reduce_scatter_out)
        assert torch.equal(all_gather_out, torch.cat(reference_shards, dim=0))
        for source_rank in range(world):
            source_rows = slice(
                source_rank * local_rows,
                (source_rank + 1) * local_rows,
            )
            _assert_one_rounding(
                all_gather_out[source_rows],
                exact_reduce_scatter[source_rows],
            )


def _check_live_rows_reuse_prepared_launchers(
    pool: PCIeTwoShotBF16,
    rank: int,
    world: int,
) -> None:
    live_rows = tuple(
        rows
        for rows in (world, 2 * world, min(64, MAX_ROWS), MAX_ROWS)
        if rows > 0 and rows <= MAX_ROWS and rows % world == 0
    )
    cases = []
    for index, rows in enumerate(dict.fromkeys(live_rows)):
        payload = _payload(7000 + 100 * index + rank, rows, pool.device)
        local_rows = rows // world
        cases.append(
            (
                payload,
                _exact_sum(payload, world),
                torch.empty_like(payload),
                torch.empty(
                    local_rows,
                    ROW_ELEMS,
                    dtype=torch.bfloat16,
                    device=pool.device,
                ),
                torch.empty_like(payload),
            )
        )

    warm_payload, _, warm_all_reduce, warm_shard, warm_gather = cases[-1]
    pool.all_reduce(warm_payload, out=warm_all_reduce)
    pool.reduce_scatter(warm_payload, out=warm_shard)
    pool.all_gather(warm_shard, out=warm_gather)
    torch.cuda.synchronize()
    dist.barrier()

    with kernel_resolution_guard(
        "PCIeTwoShotBF16 live rows must reuse warmed launcher geometry"
    ):
        for payload, exact, all_reduce_out, shard, gathered in cases:
            local_rows = payload.shape[0] // world
            pool.all_reduce(payload, out=all_reduce_out)
            pool.reduce_scatter(payload, out=shard)
            pool.all_gather(shard, out=gathered)
            torch.cuda.synchronize()
            dist.barrier()
            _assert_one_rounding(all_reduce_out, exact)
            expected_shard = exact[rank * local_rows : (rank + 1) * local_rows]
            _assert_one_rounding(shard, expected_shard)
            reference_shards = [torch.empty_like(shard) for _ in range(world)]
            dist.all_gather(reference_shards, shard)
            assert torch.equal(gathered, torch.cat(reference_shards, dim=0))


def _check_rejects_divergent_graph_slot_bias(pool: PCIeTwoShotBF16, rank: int) -> None:
    if rank == 0:
        pool._slot += 1
    try:
        with (
            pytest.raises(
                RuntimeError,
                match="graph slot selection contract differs across ranks",
            ),
            pool.capture(),
        ):
            pass
    finally:
        if rank == 0:
            pool._slot -= 1
    dist.barrier()


def _check_reduce_scatter_all_gather(
    pool: PCIeTwoShotBF16, rank: int, world: int
) -> None:
    rows = MAX_ROWS
    assert rows > 0 and rows % world == 0
    local_rows = rows // world
    payload = _payload(5300 + rank, rows, pool.device)
    exact = _exact_sum(payload, world)

    shard = torch.empty(
        local_rows,
        ROW_ELEMS,
        dtype=torch.bfloat16,
        device=pool.device,
    )
    returned_shard = pool.reduce_scatter(payload, out=shard)
    torch.cuda.synchronize()
    assert returned_shard is shard
    expected_shard = exact[rank * local_rows : (rank + 1) * local_rows]
    _assert_one_rounding(shard, expected_shard)

    gathered = torch.empty_like(payload)
    returned_gather = pool.all_gather(shard, out=gathered)
    torch.cuda.synchronize()
    assert returned_gather is gathered
    reference_shards = [torch.empty_like(shard) for _ in range(world)]
    dist.all_gather(reference_shards, shard)
    assert torch.equal(gathered, torch.cat(reference_shards, dim=0))


def _check_rejects_unsupported(pool: PCIeTwoShotBF16, world: int) -> None:
    device = pool.device
    assert not pool.accepts(
        torch.empty(MAX_ROWS + world, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    )
    assert not pool.accepts(
        torch.empty(world, ROW_ELEMS, dtype=torch.float16, device=device)
    )
    assert not pool.accepts(
        torch.empty(world, ROW_ELEMS + 8, dtype=torch.bfloat16, device=device)
    )
    if world > 1:
        assert not pool.accepts(
            torch.empty(world + 1, ROW_ELEMS, dtype=torch.bfloat16, device=device)
        )
    storage = torch.empty(
        world * ROW_ELEMS + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    misaligned = storage[1:].view(world, ROW_ELEMS)
    assert misaligned.is_contiguous()
    assert misaligned.data_ptr() % 16 != 0
    assert not pool.accepts(misaligned)


def _check_rejects_invalid_outputs(pool: PCIeTwoShotBF16, world: int) -> None:
    x = _payload(6100 + pool.rank, world, pool.device)
    wrong_shape = torch.empty(
        x.numel(),
        dtype=torch.bfloat16,
        device=pool.device,
    )
    with pytest.raises(ValueError, match="output shape"):
        pool.all_reduce(x, out=wrong_shape)

    storage = torch.empty(
        x.numel() + 1,
        dtype=torch.bfloat16,
        device=pool.device,
    )
    misaligned = storage[1:].view_as(x)
    with pytest.raises(ValueError, match="16-byte aligned"):
        pool.all_reduce(x, out=misaligned)

    local_rows = x.shape[0] // world
    with pytest.raises(TypeError, match="output must be bfloat16"):
        pool.reduce_scatter(
            x,
            out=torch.empty(
                local_rows,
                ROW_ELEMS,
                dtype=torch.float16,
                device=pool.device,
            ),
        )

    shard = _payload(6200 + pool.rank, local_rows, pool.device)
    with pytest.raises(ValueError, match="output shape"):
        pool.all_gather(shard, out=x[:local_rows])

    overlap_storage = torch.empty(
        x.numel() + 8,
        dtype=torch.bfloat16,
        device=pool.device,
    )
    overlap_input = overlap_storage[: x.numel()].view_as(x)
    overlap_output = overlap_storage[8:].view_as(x)
    with pytest.raises(ValueError, match="output must not overlap input"):
        pool.all_reduce(overlap_input, out=overlap_output)

    with pytest.raises(ValueError, match="output must not overlap payload"):
        pool.reduce_scatter(x, out=x[:local_rows])

    gather_storage = torch.empty_like(x)
    gather_payload = gather_storage[:local_rows]
    with pytest.raises(ValueError, match="output must not overlap payload"):
        pool.all_gather(gather_payload, out=gather_storage)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # The error comparison below is specifically against NCCL's BF16 ring.
    os.environ["NCCL_ALGO"] = "Ring"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    pool = PCIeTwoShotBF16.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_rows=MAX_ROWS,
        row_elems=ROW_ELEMS,
    )
    pool.prepare_graph()
    _check_rejects_unsupported(pool, world)
    _check_rejects_invalid_outputs(pool, world)
    _check_reduce_scatter_all_gather(pool, rank, world)
    executed_rows = []
    for step in range(3):  # exercises the double-buffered staging slots
        for rows in ROWS:
            if rows % world == 0 and rows <= MAX_ROWS:
                _check_all_reduce(pool, rank, world, rows, step)
                if rows not in executed_rows:
                    executed_rows.append(rows)
    _check_all_reduce(pool, rank, world, MAX_ROWS, step=3)
    if MAX_ROWS not in executed_rows:
        executed_rows.append(MAX_ROWS)
    _check_rejects_divergent_graph_slot_bias(pool, rank)
    _check_live_rows_reuse_prepared_launchers(pool, rank, world)
    _check_graph_capture(pool, rank, world)
    dist.barrier()
    if rank == 0:
        print(
            "pcie_twoshot_bf16 correctness OK "
            f"({world} ranks, all_reduce_rows={tuple(executed_rows)}, "
            f"workspace_max_rows={MAX_ROWS})"
        )
    pool.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
