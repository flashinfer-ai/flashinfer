"""Correctness + micro-benchmark for PCIe two-shot fp8 SP collectives.

Run with torchrun on 2, 4 or 8 GPUs:

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
        --nproc-per-node=8 tests/distributed/test_pcie_twoshot.py
"""

import ctypes
import os
import time

import pytest
import torch
import torch.distributed as dist

from b12x.comm.pcie.pcie_oneshot import (
    PCIeOneshotAllReduce,
    _OwnedSharedBuffer,
    _RETAINED_FAILED_IPC_EXPORTS,
)
from b12x.comm.pcie.pcie_twoshot import (
    PCIeTwoShotSP,
    _make_layout,
    quantize_per_row,
)

ROWS = int(os.getenv("B12X_TEST_TWOSHOT_ROWS", "4096"))
ROW_ELEMS = int(os.getenv("B12X_TEST_TWOSHOT_ROW_ELEMS", "6144"))


def test_factory_retains_twoshot_ipc_ownership_when_verdict_fails(
    monkeypatch,
) -> None:
    events: list[tuple[str, int]] = []

    class FakeIPC:
        def cudaIpcCloseMemHandle(self, ptr):
            events.append(("close", ptr))

        def cudaFree(self, ptr):
            events.append(("free", ptr))

    ipc = FakeIPC()
    group = object()
    shared = _OwnedSharedBuffer(
        local_ptr=1000,
        peer_ptrs=(1000, 2000),
        remote_ptrs=(2000,),
    )
    preallocation_round = 0

    def fake_preallocation(*, owner, exchange_group, setup):
        nonlocal preallocation_round
        assert exchange_group is group
        preallocation_round += 1
        if preallocation_round == 1:
            return torch.device("cuda:0"), 8, 16
        assert preallocation_round == 2
        return ipc, _make_layout(8, 16, 2)

    verdict_round = 0

    def exchange(local_status, exchange_group):
        nonlocal verdict_round
        assert exchange_group is group
        verdict_round += 1
        if verdict_round == 1:
            raise RuntimeError("injected twoshot runtime verdict exchange failure")
        return [local_status, ()]

    monkeypatch.setattr(dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot._run_collective_preallocation_setup",
        fake_preallocation,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot._require_full_grid_residency",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_twoshot._require_collective_contract",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        PCIeOneshotAllReduce,
        "_allocate_shared_buffer",
        classmethod(lambda cls, *args, **kwargs: shared),
    )
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_oneshot._broadcast_gather_object",
        exchange,
    )

    with pytest.raises(RuntimeError, match="must be coordinated by every rank") as exc:
        PCIeTwoShotSP.from_exchange_group(
            exchange_group=group,
            device="cuda:0",
            max_rows=8,
            row_elems=16,
        )

    retained = exc.value.retryable_setup
    assert retained.local_ptr == 1000
    assert retained.remote_ptrs == [2000]
    assert retained.local_cleanup is None
    assert events == []
    assert _RETAINED_FAILED_IPC_EXPORTS[retained.key] is retained

    retained.retry()
    assert events == [("close", 2000), ("free", 1000)]
    assert _RETAINED_FAILED_IPC_EXPORTS == {}


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _local_staging_words(
    pool: PCIeTwoShotSP, stream: torch.cuda.Stream
) -> tuple[int, int]:
    assert len(pool._owned_buffers) == 1
    max_rows_per_rank = pool.max_rows // pool.world_size
    pack_stride = _align_up(max_rows_per_rank * (pool.row_elems // 16), 16)
    payload_bytes = pool.world_size * pack_stride * 16
    scale_offset = _align_up(payload_bytes, 256)
    scale_stride = _align_up(max_rows_per_rank, 64)
    slot_bytes = _align_up(
        scale_offset + pool.world_size * scale_stride * 4,
        256,
    )
    signal_bytes = (
        pool._staging_ptrs[0][pool.rank] - pool._signal_ptrs[pool.rank]
    )
    remote_source = (pool.rank + 1) % pool.world_size
    source_offset = remote_source * pack_stride * 16
    words = (ctypes.c_uint64(), ctypes.c_uint64())
    local_ptr = pool._owned_buffers[0].local_ptr
    for word, offset in zip(
        words,
        (
            signal_bytes + source_offset,
            signal_bytes + slot_bytes + source_offset,
        ),
        strict=True,
    ):
        pool._ipc.cudaMemcpyAsync(
            ctypes.addressof(word),
            local_ptr + offset,
            ctypes.sizeof(word),
            int(stream.cuda_stream),
        )
    stream.synchronize()
    return words[0].value, words[1].value


def _assert_alternating_slots(snapshots: list[tuple[int, int]]) -> None:
    changed_slots = [
        {
            slot
            for slot, (before, after) in enumerate(
                zip(snapshots[index], snapshots[index + 1], strict=True)
            )
            if before != after
        }
        for index in range(len(snapshots) - 1)
    ]
    assert all(len(changed) == 1 for changed in changed_slots)
    assert all(
        changed_slots[index] != changed_slots[index + 1]
        for index in range(len(changed_slots) - 1)
    )


def _partial(seed: int, rows: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(rows, ROW_ELEMS, generator=gen, dtype=torch.float32)
    return x.to(device=device, dtype=torch.bfloat16)


def _check_reduce_scatter(
    pool: PCIeTwoShotSP, rank: int, world: int, step: int
) -> None:
    device = pool.device
    payloads = []
    scales = []
    for r in range(world):
        q, s = quantize_per_row(_partial(1000 * step + r, ROWS, device))
        payloads.append(q)
        scales.append(s)

    if step == 0:
        wrong_device = (rank + 1) % world
        torch.cuda.set_device(wrong_device)
    out = pool.reduce_scatter_fp8(payloads[rank], scales[rank])
    if step == 0:
        assert torch.cuda.current_device() == wrong_device
        torch.cuda.set_device(rank)

    rows_per_rank = ROWS // world
    lo, hi = rank * rows_per_rank, (rank + 1) * rows_per_rank
    ref = torch.zeros(rows_per_rank, ROW_ELEMS, dtype=torch.float32, device=device)
    for r in range(world):
        ref += payloads[r][lo:hi].float() * scales[r][lo:hi, None]
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def _check_all_gather(pool: PCIeTwoShotSP, rank: int, world: int, step: int) -> None:
    device = pool.device
    rows_per_rank = ROWS // world
    shards = []
    scales = []
    for r in range(world):
        q, s = quantize_per_row(_partial(5000 * step + r, rows_per_rank, device))
        shards.append(q)
        scales.append(s)

    out = pool.all_gather_fp8(shards[rank], scales[rank])

    ref = torch.cat(
        [
            (shards[r].float() * scales[r][:, None]).to(torch.bfloat16)
            for r in range(world)
        ]
    )
    assert torch.equal(out, ref), "all_gather must be exact (dequant only)"


def _check_graph_capture(pool: PCIeTwoShotSP, rank: int, world: int) -> None:
    device = pool.device
    rows_per_rank = ROWS // world
    q_in = torch.zeros(ROWS, ROW_ELEMS, dtype=torch.float8_e4m3fn, device=device)
    s_in = torch.zeros(ROWS, dtype=torch.float32, device=device)
    rs_out = torch.empty(rows_per_rank, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    ag_q = torch.zeros(
        rows_per_rank, ROW_ELEMS, dtype=torch.float8_e4m3fn, device=device
    )
    ag_s = torch.zeros(rows_per_rank, dtype=torch.float32, device=device)
    ag_out = torch.empty(ROWS, ROW_ELEMS, dtype=torch.bfloat16, device=device)

    graph = torch.cuda.CUDAGraph()
    # Warmup on a side stream, then capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        pool.reduce_scatter_fp8(q_in, s_in, rs_out, block_limit=7)
        pool.all_gather_fp8(ag_q, ag_s, ag_out, block_limit=3)
        # Leave the host selector on odd parity before first capture. The
        # graph specialization must seed that exact next slot, not assume 0.
        pool.reduce_scatter_fp8(q_in, s_in, rs_out, block_limit=7)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()

    with pool.capture(), torch.cuda.graph(graph):
        # Deliberately use different grid sizes: slot selection is driven by
        # one channel-wide epoch, never by per-CTA barrier-counter parity.
        pool.reduce_scatter_fp8(q_in, s_in, rs_out, block_limit=7)
        pool.all_gather_fp8(ag_q, ag_s, ag_out, block_limit=3)

    for step in (11, 12):
        payloads, scales, shards, sscales = [], [], [], []
        for r in range(world):
            q, s = quantize_per_row(_partial(7000 * step + r, ROWS, device))
            payloads.append(q)
            scales.append(s)
            qs, ss = quantize_per_row(_partial(9000 * step + r, ROWS // world, device))
            shards.append(qs)
            sscales.append(ss)
        q_in.copy_(payloads[rank])
        s_in.copy_(scales[rank])
        ag_q.copy_(shards[rank])
        ag_s.copy_(sscales[rank])
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()

        rows_per_rank = ROWS // world
        lo, hi = rank * rows_per_rank, (rank + 1) * rows_per_rank
        ref = torch.zeros(rows_per_rank, ROW_ELEMS, dtype=torch.float32, device=device)
        for r in range(world):
            ref += payloads[r][lo:hi].float() * scales[r][lo:hi, None]
        torch.testing.assert_close(rs_out.float(), ref, rtol=2e-2, atol=2e-2)
        ag_ref = torch.cat(
            [
                (shards[r].float() * sscales[r][:, None]).to(torch.bfloat16)
                for r in range(world)
            ]
        )
        assert torch.equal(ag_out, ag_ref)

    # Capture a single collective so adjacent replays must alternate the
    # device-selected staging slot. The two-op graph above has even parity.
    odd_graph = torch.cuda.CUDAGraph()
    with pool.capture(operations=("all_gather",)), torch.cuda.graph(odd_graph):
        pool.all_gather_fp8(ag_q, ag_s, ag_out)

    for value in (11, 12):
        ag_q.fill_(float(value + rank))
        ag_s.fill_(1.0)
        pool.all_gather_fp8(ag_q, ag_s, ag_out)
    torch.cuda.synchronize()
    snapshots = [_local_staging_words(pool, torch.cuda.current_stream(device))]

    for value in (1, 2):
        ag_q.fill_(float(value + rank))
        ag_s.fill_(1.0)
        dist.barrier()
        odd_graph.replay()
        torch.cuda.synchronize()
        snapshots.append(_local_staging_words(pool, torch.cuda.current_stream(device)))
        for source_rank in range(world):
            lo = source_rank * rows_per_rank
            hi = lo + rows_per_rank
            assert torch.all(ag_out[lo:hi] == float(value + source_rank))

    _assert_alternating_slots(snapshots)

    rs_graph = torch.cuda.CUDAGraph()
    with pool.capture(operations=("reduce_scatter",)), torch.cuda.graph(rs_graph):
        pool.reduce_scatter_fp8(q_in, s_in, rs_out)

    for value in (21, 22):
        q_in.fill_(float(value + rank))
        s_in.fill_(1.0)
        pool.reduce_scatter_fp8(q_in, s_in, rs_out)
    torch.cuda.synchronize()
    snapshots = [_local_staging_words(pool, torch.cuda.current_stream(device))]

    rank_sum = world * (world - 1) // 2
    for value in (3, 4):
        q_in.fill_(float(value + rank))
        s_in.fill_(1.0)
        dist.barrier()
        rs_graph.replay()
        torch.cuda.synchronize()
        snapshots.append(_local_staging_words(pool, torch.cuda.current_stream(device)))
        assert torch.all(rs_out == float(world * value + rank_sum))

    _assert_alternating_slots(snapshots)


def _check_queued_graph_replay(
    pool: PCIeTwoShotSP,
    rank: int,
    world: int,
    operation: str,
) -> None:
    """Stress one collective graph under intentional host-rank skew."""

    device = pool.device
    graph_rows = min(64, pool.max_rows)
    rows_per_rank = graph_rows // world
    input_rows = graph_rows if operation == "reduce_scatter" else rows_per_rank
    output_rows = rows_per_rank if operation == "reduce_scatter" else graph_rows
    q_in = torch.empty(
        input_rows, ROW_ELEMS, dtype=torch.float8_e4m3fn, device=device
    )
    s_in = torch.empty(input_rows, dtype=torch.float32, device=device)
    out = torch.empty(
        output_rows, ROW_ELEMS, dtype=torch.bfloat16, device=device
    )
    snapshots = [torch.empty_like(out) for _ in range(3)]

    payload_steps: list[list[torch.Tensor]] = []
    scale_steps: list[list[torch.Tensor]] = []
    for step in range(3):
        payloads, scales = [], []
        for source_rank in range(world):
            q, s = quantize_per_row(
                _partial(
                    11000 + step * 100 + source_rank,
                    input_rows,
                    device,
                )
            )
            payloads.append(q)
            scales.append(s)
        payload_steps.append(payloads)
        scale_steps.append(scales)

    q_in.copy_(payload_steps[0][rank])
    s_in.copy_(scale_steps[0][rank])
    # Keep identical block ids across ranks while making rank zero consume
    # phase two much more slowly on the GPU.  Fast peers can then enter the
    # following replay's remote push before rank zero has drained its prior
    # slot, deterministically exercising the double-slot graph protocol.
    rank_threads = 32 if rank == 0 else 512
    if operation == "reduce_scatter":

        def launch():
            return pool.reduce_scatter_fp8(
                q_in,
                s_in,
                out,
                threads=rank_threads,
                block_limit=3,
            )

    else:

        def launch():
            return pool.all_gather_fp8(
                q_in,
                s_in,
                out,
                threads=rank_threads,
                block_limit=3,
            )
    launch()
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with pool.capture(
        operations=(operation,), threads=rank_threads
    ), torch.cuda.graph(graph):
        launch()

    # Queue three replays without a cross-rank host barrier. Rank zero pauses
    # immediately after its first enqueue, allowing peers to begin the next
    # remote push while its prior consumer may still be draining slot zero.
    for step in range(3):
        q_in.copy_(payload_steps[step][rank])
        s_in.copy_(scale_steps[step][rank])
        graph.replay()
        snapshots[step].copy_(out)
        if step == 0 and rank == 0:
            time.sleep(0.02)
    torch.cuda.synchronize()
    dist.barrier()

    for step in range(3):
        if operation == "reduce_scatter":
            lo = rank * rows_per_rank
            hi = (rank + 1) * rows_per_rank
            ref = torch.zeros(
                rows_per_rank,
                ROW_ELEMS,
                dtype=torch.float32,
                device=device,
            )
            for source_rank in range(world):
                ref += (
                    payload_steps[step][source_rank][lo:hi].float()
                    * scale_steps[step][source_rank][lo:hi, None]
                )
            torch.testing.assert_close(
                snapshots[step].float(), ref, rtol=2e-2, atol=2e-2
            )
        else:
            ref = torch.cat(
                [
                    (
                        payload_steps[step][source_rank].float()
                        * scale_steps[step][source_rank][:, None]
                    ).to(torch.bfloat16)
                    for source_rank in range(world)
                ]
            )
            assert torch.equal(snapshots[step], ref)


def _bench(pool: PCIeTwoShotSP, rank: int, world: int) -> None:
    device = pool.device
    rows_per_rank = ROWS // world
    x = torch.randn(ROWS, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    q, s = quantize_per_row(x)
    qs, ss = quantize_per_row(x[:rows_per_rank])
    rs_out = torch.empty(rows_per_rank, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    ag_out = torch.empty(ROWS, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    nccl_rs_out = torch.empty(
        rows_per_rank, ROW_ELEMS, dtype=torch.bfloat16, device=device
    )
    nccl_ag_out = torch.empty(ROWS, ROW_ELEMS, dtype=torch.bfloat16, device=device)
    shard_bf16 = x[:rows_per_rank].contiguous()

    def timeit(fn, iters=30) -> float:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / iters * 1e6

    results = {
        "b12x rs_fp8": timeit(lambda: pool.reduce_scatter_fp8(q, s, rs_out)),
        "b12x ag_fp8": timeit(lambda: pool.all_gather_fp8(qs, ss, ag_out)),
        "nccl rs bf16": timeit(lambda: dist.reduce_scatter_tensor(nccl_rs_out, x)),
        "nccl ag bf16": timeit(
            lambda: dist.all_gather_into_tensor(nccl_ag_out, shard_bf16)
        ),
    }
    if rank == 0:
        payload_mb = ROWS * ROW_ELEMS / 1e6
        print(
            f"[{world} ranks, {ROWS}x{ROW_ELEMS}, payload {payload_mb:.0f} MB bf16-equiv]"
        )
        for name, us in results.items():
            print(f"  {name:14s} {us:9.1f} us")


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    pool = PCIeTwoShotSP.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_rows=ROWS,
        row_elems=ROW_ELEMS,
    )

    for step in range(4):  # exercises double-buffer slot alternation
        _check_reduce_scatter(pool, rank, world, step)
        _check_all_gather(pool, rank, world, step)
    _check_graph_capture(pool, rank, world)
    _check_queued_graph_replay(pool, rank, world, "reduce_scatter")
    _check_queued_graph_replay(pool, rank, world, "all_gather")
    dist.barrier()
    if rank == 0:
        print("pcie_twoshot correctness OK")
    if os.getenv("B12X_TEST_TWOSHOT_SKIP_BENCH", "0") != "1":
        _bench(pool, rank, world)

    pool.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
