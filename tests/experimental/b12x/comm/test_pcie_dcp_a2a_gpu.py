from __future__ import annotations

import ctypes
import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import b12x.comm.pcie.pcie_dcp_a2a as pcie_dcp_a2a
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce
from b12x.comm.pcie.pcie_dcp_a2a import (
    PCIeDCPA2A,
    PCIeDCPA2APool,
    _staging_layout,
    lse_reduce_scatter_reference,
)


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_A2A_TEST") != "1",
    reason="set B12X_RUN_PCIE_DCP_A2A_TEST=1 to run PCIe DCP A2A GPU tests",
)

TOTAL_HEADS = 16
HEAD_DIM = 512
QUERY_HEAD_DIM = 576
MAX_BATCH = 64
TEST_BATCHES = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rank_inputs(
    step: int,
    source_rank: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(10000 * step + source_rank)
    output = torch.randn(
        batch,
        TOTAL_HEADS,
        HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    lse = torch.randn(
        batch,
        TOTAL_HEADS,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    if batch > 0:
        lse[0, 0] = -torch.inf
        lse[0, 1] = torch.nan
        if source_rank == 0:
            lse[0, 2] = -torch.inf
    return output, lse


def _rank_query(
    step: int,
    source_rank: int,
    world_size: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20000 * step + source_rank)
    return torch.randn(
        batch,
        TOTAL_HEADS // world_size,
        QUERY_HEAD_DIM,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)


def _reference(
    step: int,
    rank: int,
    world_size: int,
    batch: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    inputs = [
        _rank_inputs(step, source_rank, batch, dtype, device)
        for source_rank in range(world_size)
    ]
    return lse_reduce_scatter_reference(
        torch.stack([item[0] for item in inputs]),
        torch.stack([item[1] for item in inputs]),
        rank,
    )


def _local_staging_words(channel, stream: torch.cuda.Stream) -> tuple[int, int]:
    assert channel._ipc is not None
    assert len(channel._owned_buffers) == 1
    layout = _staging_layout(
        signal_bytes=pcie_dcp_a2a._SIGNAL_BYTES,
        world_size=channel.world_size,
        max_batch_size=channel.max_batch_size,
        total_heads=channel.total_heads,
        head_dim=channel.head_dim,
        query_head_dim=channel.query_head_dim,
    )
    words = (ctypes.c_uint16(), ctypes.c_uint16())
    local_ptr = channel._owned_buffers[0].local_ptr
    for word, offset in zip(
        words,
        (layout.staging0_offset, layout.staging1_offset),
        strict=True,
    ):
        channel._ipc.cudaMemcpyAsync(
            ctypes.addressof(word),
            local_ptr + offset,
            ctypes.sizeof(word),
            int(stream.cuda_stream),
        )
    stream.synchronize()
    return words[0].value, words[1].value


def _check_eager(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    for dtype in (torch.bfloat16, torch.float16):
        for step, batch in enumerate(TEST_BATCHES, start=1):
            local_q = _rank_query(
                step + 100,
                rank,
                world_size,
                batch,
                dtype,
                device,
            )
            wrong_device = (rank + 1) % world_size
            guard_gather = dtype == torch.bfloat16 and step == 1
            if guard_gather:
                torch.cuda.set_device(wrong_device)
            gathered_q = pool.all_gather_heads(
                local_q, channel_id="eager:dcp"
            )
            if guard_gather:
                assert torch.cuda.current_device() == wrong_device
                torch.cuda.set_device(rank)
            expected_q = torch.cat(
                [
                    _rank_query(
                        step + 100,
                        source,
                        world_size,
                        batch,
                        dtype,
                        device,
                    )
                    for source in range(world_size)
                ],
                dim=1,
            )
            torch.testing.assert_close(gathered_q, expected_q, rtol=0, atol=0)

            partial_output, partial_lse = _rank_inputs(step, rank, batch, dtype, device)
            guard_reduce = dtype == torch.bfloat16 and step == 2
            if guard_reduce:
                torch.cuda.set_device(wrong_device)
            out = pool.lse_reduce_scatter(
                partial_output, partial_lse, channel_id="eager:dcp"
            )
            if guard_reduce:
                assert torch.cuda.current_device() == wrong_device
                torch.cuda.set_device(rank)
            torch.cuda.synchronize(device)
            expected = _reference(
                step,
                rank,
                world_size,
                batch,
                dtype,
                device,
            )
            torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

            input_storage = torch.empty(
                TOTAL_HEADS,
                MAX_BATCH,
                HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            head_major_input = input_storage.transpose(0, 1)[:batch]
            head_major_input.copy_(partial_output)
            output_storage = torch.empty(
                TOTAL_HEADS // world_size,
                MAX_BATCH,
                HEAD_DIM,
                dtype=dtype,
                device=device,
            )
            head_major_output = output_storage.transpose(0, 1)[:batch]
            actual = pool.lse_reduce_scatter(
                head_major_input,
                partial_lse,
                out=head_major_output,
                channel_id="eager:dcp",
            )
            torch.cuda.synchronize(device)
            assert actual is head_major_output
            assert actual.movedim(0, 1).stride(0) == MAX_BATCH * HEAD_DIM
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    for step, batch in enumerate(TEST_BATCHES, start=100):
        local_q = _rank_query(
            step,
            rank,
            world_size,
            batch,
            torch.float8_e4m3fn,
            device,
        )
        gathered_q = pool.all_gather_heads(local_q, channel_id="eager:dcp")
        expected_q = torch.cat(
            [
                _rank_query(
                    step,
                    source,
                    world_size,
                    batch,
                    torch.float8_e4m3fn,
                    device,
                )
                for source in range(world_size)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            gathered_q.view(torch.uint8),
            expected_q.view(torch.uint8),
            rtol=0,
            atol=0,
        )


def _check_eager_adjacency(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    channel = pool.for_stream(channel_id="eager:dcp")
    first_query = _rank_query(700, rank, world_size, MAX_BATCH, torch.bfloat16, device)
    second_query = _rank_query(701, rank, world_size, MAX_BATCH, torch.bfloat16, device)
    partial_output, partial_lse = _rank_inputs(702, rank, 1, torch.bfloat16, device)
    first_gather = torch.empty(
        MAX_BATCH,
        TOTAL_HEADS,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    second_gather = torch.empty_like(first_gather)
    reduced = torch.empty(
        1,
        TOTAL_HEADS // world_size,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    # Issue large-grid AG -> small-grid RS -> large-grid AG without a host
    # synchronization so adjacent eager executions exercise slot advancement.
    channel.all_gather_heads(first_query, first_gather)
    channel.lse_reduce_scatter(partial_output, partial_lse, reduced)
    channel.all_gather_heads(second_query, second_gather)
    torch.cuda.synchronize(device)

    expected_first = torch.cat(
        [
            _rank_query(700, source, world_size, MAX_BATCH, torch.bfloat16, device)
            for source in range(world_size)
        ],
        dim=1,
    )
    expected_second = torch.cat(
        [
            _rank_query(701, source, world_size, MAX_BATCH, torch.bfloat16, device)
            for source in range(world_size)
        ],
        dim=1,
    )
    torch.testing.assert_close(first_gather, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second_gather, expected_second, rtol=0, atol=0)
    torch.testing.assert_close(
        reduced,
        _reference(702, rank, world_size, 1, torch.bfloat16, device),
        rtol=2e-2,
        atol=2e-2,
    )


def _check_graph(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    stream = torch.cuda.Stream(device=device)
    pool.prepare_channels(("graph",))
    channel = pool.for_stream(stream, channel_id="graph")
    layers = 7
    input_storages = [
        torch.empty(
            TOTAL_HEADS,
            MAX_BATCH,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(layers)
    ]
    inputs = [storage.transpose(0, 1) for storage in input_storages]
    lses = [
        torch.empty(MAX_BATCH, TOTAL_HEADS, dtype=torch.float32, device=device)
        for _ in range(layers)
    ]
    output_storages = [
        torch.empty(
            TOTAL_HEADS // world_size,
            MAX_BATCH,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(layers)
    ]
    outputs = [storage.transpose(0, 1) for storage in output_storages]
    assert all(tensor.stride(1) == MAX_BATCH * HEAD_DIM for tensor in inputs)
    assert all(tensor.stride(1) == MAX_BATCH * HEAD_DIM for tensor in outputs)
    local_queries = [
        torch.empty(
            1,
            TOTAL_HEADS // world_size,
            QUERY_HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        for _ in range(layers)
    ]
    gathered_queries = [
        torch.empty(
            1,
            TOTAL_HEADS,
            QUERY_HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        for _ in range(layers)
    ]

    with torch.cuda.stream(stream):
        channel.all_gather_heads(local_queries[0], gathered_queries[0])
        channel.lse_reduce_scatter(inputs[0], lses[0], outputs[0])
    stream.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with pool.capture(
        stream, channel_id="graph:layer-stack"
    ) as graph_channel, torch.cuda.graph(graph, stream=stream):
        for layer in range(layers):
            graph_channel.all_gather_heads(
                local_queries[layer], gathered_queries[layer]
            )
            graph_channel.lse_reduce_scatter(
                inputs[layer], lses[layer], outputs[layer]
            )
    stream.synchronize()

    replay_count = int(os.getenv("B12X_PCIE_DCP_GRAPH_REPLAYS", "8"))
    if replay_count <= 0:
        raise ValueError("B12X_PCIE_DCP_GRAPH_REPLAYS must be positive")
    for replay in range(replay_count):
        expected = []
        expected_queries = []
        for layer in range(layers):
            step = 1000 * replay + layer + 100
            partial_output, partial_lse = _rank_inputs(
                step,
                rank,
                MAX_BATCH,
                torch.bfloat16,
                device,
            )
            inputs[layer].copy_(partial_output)
            lses[layer].copy_(partial_lse)
            local_queries[layer].copy_(
                _rank_query(
                    step,
                    rank,
                    world_size,
                    1,
                    torch.float8_e4m3fn,
                    device,
                )
            )
            expected_queries.append(
                torch.cat(
                    [
                        _rank_query(
                            step,
                            source,
                            world_size,
                            1,
                            torch.float8_e4m3fn,
                            device,
                        )
                        for source in range(world_size)
                    ],
                    dim=1,
                )
            )
            expected.append(
                _reference(
                    step,
                    rank,
                    world_size,
                    MAX_BATCH,
                    torch.bfloat16,
                    device,
                )
            )
        stream.wait_stream(torch.cuda.current_stream(device))
        graph.replay()
        stream.synchronize()
        for out, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(out, reference, rtol=2e-2, atol=2e-2)
        for out, reference in zip(gathered_queries, expected_queries, strict=True):
            torch.testing.assert_close(
                out.view(torch.uint8),
                reference.view(torch.uint8),
                rtol=0,
                atol=0,
            )

    # One captured operation has odd capture-time parity. Adjacent A -> A
    # replays must advance the device-owned slot at execution time.
    odd_input = torch.empty(
        1,
        TOTAL_HEADS // world_size,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    odd_output = torch.empty(
        1,
        TOTAL_HEADS,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    with torch.cuda.stream(stream):
        channel.all_gather_heads(odd_input, odd_output)
    stream.synchronize()
    dist.barrier()

    odd_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(odd_graph, stream=stream):
        channel.all_gather_heads(odd_input, odd_output)
    stream.synchronize()

    # Prime both slots after capture, then observe them read-only. A graph with
    # a host-baked slot changes the same word on both replays; execution-owned
    # parity changes alternate words.
    with torch.cuda.stream(stream):
        for value in (11.0, 12.0):
            odd_input.fill_(value)
            channel.all_gather_heads(odd_input, odd_output)
    snapshots = [_local_staging_words(channel, stream)]
    for value in (1.0, 2.0):
        with torch.cuda.stream(stream):
            odd_input.fill_(value)
            odd_graph.replay()
        snapshots.append(_local_staging_words(channel, stream))
        torch.testing.assert_close(
            odd_output, torch.full_like(odd_output, value), rtol=0, atol=0
        )

    changed_slots = [
        {
            slot
            for slot, (before, after) in enumerate(
                zip(snapshots[index], snapshots[index + 1], strict=True)
            )
            if before != after
        }
        for index in range(2)
    ]
    assert all(len(changed) == 1 for changed in changed_slots)
    assert changed_slots[0] != changed_slots[1]


def _check_semantic_capture_warmup(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Match vLLM's eager DCP warmup inside a graph-owner scope."""
    stream = torch.cuda.Stream(device=device)
    local_query = _rank_query(
        900,
        rank,
        world_size,
        1,
        torch.bfloat16,
        device,
    )
    gathered = torch.empty(
        1,
        TOTAL_HEADS,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    expected = torch.cat(
        [
            _rank_query(
                900,
                source,
                world_size,
                1,
                torch.bfloat16,
                device,
            )
            for source in range(world_size)
        ],
        dim=1,
    )
    pool.prepare_channels(("graph:warmup",))

    graph = torch.cuda.CUDAGraph()
    with (
        torch.cuda.stream(stream),
        pool.capture(stream, channel_id="graph:warmup") as graph_channel,
    ):
        warmup_channel = pool.for_stream(stream, channel_id="eager:dcp")
        assert warmup_channel is graph_channel
        warmup_channel.all_gather_heads(local_query, gathered)
        stream.synchronize()
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)

        with torch.cuda.graph(graph, stream=stream):
            graph_channel.all_gather_heads(local_query, gathered)

    # The eager DCP channel was already bound to vLLM's default stream. Its
    # original mapping must be usable again after the graph-owner scope exits.
    eager_channel = pool.for_stream(channel_id="eager:dcp")
    assert eager_channel is not graph_channel
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)


def _check_teardown_retry(
    pool: PCIeDCPA2APool,
    rank: int,
    device: torch.device,
) -> None:
    """Inject one local unmap failure and prove every rank retries coherently."""

    assert pool._all_channels
    channel = pool._all_channels[0]
    assert channel._ipc is not None
    original_close = channel._ipc.cudaIpcCloseMemHandle
    injected = False

    if rank == 0:

        def fail_once(ptr):
            nonlocal injected
            if not injected:
                injected = True
                raise RuntimeError("injected DCP IPC unmap failure")
            return original_close(ptr)

        channel._ipc.cudaIpcCloseMemHandle = fail_once

    failed = False
    try:
        pool.close()
    except RuntimeError as exc:
        failed = "IPC unmap" in str(exc)
    verdict = torch.tensor(int(failed), device=device, dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    assert int(verdict.item()) == 1
    assert not pool._closed

    if rank == 0:
        channel._ipc.cudaIpcCloseMemHandle = original_close
    pool.close()
    assert pool._closed


def _check_queued_mixed_grid_graph(
    pool: PCIeDCPA2APool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> None:
    """Stress slot retirement across odd, mixed-grid graph replays.

    The graph intentionally contains 2-block gather, 16-block LSE, then
    2-block gather.  Its odd transport-node count flips the starting slot on
    every replay.  Replays are queued without a rank barrier or device sync,
    and rank zero pauses after its first enqueue so peer hosts can enqueue the
    next replay before rank zero enqueues its matching launch.
    """

    batch = MAX_BATCH
    local_heads = TOTAL_HEADS // world_size
    query_a = torch.empty(
        1,
        local_heads,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    query_b = torch.empty_like(query_a)
    gathered_a = torch.empty(
        1,
        TOTAL_HEADS,
        QUERY_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    gathered_b = torch.empty_like(gathered_a)
    partial_output = torch.empty(
        batch,
        TOTAL_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    partial_lse = torch.empty(
        batch,
        TOTAL_HEADS,
        dtype=torch.float32,
        device=device,
    )
    reduced = torch.empty(
        batch,
        local_heads,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    with pool.capture(channel_id="graph:mixed-grid") as channel, torch.cuda.graph(
        graph
    ):
        channel.all_gather_heads(query_a, gathered_a)
        channel.lse_reduce_scatter(partial_output, partial_lse, reduced)
        channel.all_gather_heads(query_b, gathered_b)

    replay_steps = (4100, 4200, 4300)
    local_payloads = []
    local_lses = []
    local_queries_a = []
    local_queries_b = []
    for step in replay_steps:
        payload, lse = _rank_inputs(
            step,
            rank,
            batch,
            torch.bfloat16,
            device,
        )
        local_payloads.append(payload)
        local_lses.append(lse)
        local_queries_a.append(
            _rank_query(
                step + 100,
                rank,
                world_size,
                1,
                torch.bfloat16,
                device,
            )
        )
        local_queries_b.append(
            _rank_query(
                step + 200,
                rank,
                world_size,
                1,
                torch.bfloat16,
                device,
            )
        )

    reduced_snapshots = [torch.empty_like(reduced) for _ in replay_steps]
    gathered_a_snapshots = [torch.empty_like(gathered_a) for _ in replay_steps]
    gathered_b_snapshots = [torch.empty_like(gathered_b) for _ in replay_steps]
    torch.cuda.synchronize(device)
    for replay_index in range(len(replay_steps)):
        partial_output.copy_(local_payloads[replay_index])
        partial_lse.copy_(local_lses[replay_index])
        query_a.copy_(local_queries_a[replay_index])
        query_b.copy_(local_queries_b[replay_index])
        graph.replay()
        reduced_snapshots[replay_index].copy_(reduced)
        gathered_a_snapshots[replay_index].copy_(gathered_a)
        gathered_b_snapshots[replay_index].copy_(gathered_b)
        if replay_index == 0 and rank == 0:
            time.sleep(0.02)
    torch.cuda.synchronize(device)
    dist.barrier()

    for replay_index, step in enumerate(replay_steps):
        expected_reduced = _reference(
            step,
            rank,
            world_size,
            batch,
            torch.bfloat16,
            device,
        )
        expected_a = torch.cat(
            [
                _rank_query(
                    step + 100,
                    source,
                    world_size,
                    1,
                    torch.bfloat16,
                    device,
                )
                for source in range(world_size)
            ],
            dim=1,
        )
        expected_b = torch.cat(
            [
                _rank_query(
                    step + 200,
                    source,
                    world_size,
                    1,
                    torch.bfloat16,
                    device,
                )
                for source in range(world_size)
            ],
            dim=1,
        )
        torch.testing.assert_close(
            reduced_snapshots[replay_index],
            expected_reduced,
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            gathered_a_snapshots[replay_index], expected_a, rtol=0, atol=0
        )
        torch.testing.assert_close(
            gathered_b_snapshots[replay_index], expected_b, rtol=0, atol=0
        )


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeDCPA2APool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_batch_size=MAX_BATCH,
        total_heads=TOTAL_HEADS,
        head_dim=HEAD_DIM,
        query_head_dim=QUERY_HEAD_DIM,
        max_concurrent_channels=2,
    )
    closed = False
    try:
        pool.prepare_channels(("eager:dcp", "graph"))
        if rank == 0:
            print("A2A GPU gate: eager", flush=True)
        _check_eager(pool, rank, world_size, device)
        dist.barrier()
        _check_eager_adjacency(pool, rank, world_size, device)
        dist.barrier()
        _check_semantic_capture_warmup(pool, rank, world_size, device)
        dist.barrier()
        if rank == 0:
            print("A2A GPU gate: graph replay", flush=True)
        _check_graph(pool, rank, world_size, device)
        dist.barrier()
        if rank == 0:
            print("A2A GPU gate: queued mixed-grid skew", flush=True)
        _check_queued_mixed_grid_graph(pool, rank, world_size, device)
        if rank == 0:
            print("A2A GPU gate: complete", flush=True)
        torch.cuda.synchronize(device)
        if os.getenv("B12X_PCIE_DCP_TEST_TEARDOWN_RETRY", "0") == "1":
            _check_teardown_retry(pool, rank, device)
            closed = True
    finally:
        if not closed:
            pool.close()
        dist.destroy_process_group()


def _residency_rejection_worker(rank: int, world_size: int, port: int) -> None:
    if rank != 0:
        # Simulate one constrained/MIG-like rank in an otherwise full device
        # group; every peer must reject before any IPC allocation.
        os.environ.pop("B12X_PCIE_TEST_VISIBLE_SM_COUNT", None)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    original_cuda_rt = pcie_dcp_a2a.CudaRTLibrary

    def unexpected_cuda_rt():
        raise AssertionError("CUDA IPC allocation started before residency rejection")

    pcie_dcp_a2a.CudaRTLibrary = unexpected_cuda_rt
    try:
        with pytest.raises(RuntimeError, match="requires at least 64 visible SMs"):
            PCIeDCPA2A.from_process_group(
                process_group=dist.group.WORLD,
                device=device,
                max_batch_size=MAX_BATCH,
                total_heads=TOTAL_HEADS,
                head_dim=HEAD_DIM,
                query_head_dim=QUERY_HEAD_DIM,
            )
        dist.barrier()
    finally:
        pcie_dcp_a2a.CudaRTLibrary = original_cuda_rt
        dist.destroy_process_group()


def _preflight_rejection_worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    original_cuda_rt = pcie_dcp_a2a.CudaRTLibrary
    original_allocate = PCIeOneshotAllReduce._allocate_shared_buffer

    class FakeExt:
        @staticmethod
        def meta_size():
            return 256

    def fail_cuda_rt():
        raise RuntimeError("injected rank-zero pre-allocation failure")

    def unexpected_allocate(*args, **kwargs):
        raise AssertionError(
            "CUDA IPC allocation started after a peer preflight failure"
        )

    if rank == 0:
        pcie_dcp_a2a.CudaRTLibrary = fail_cuda_rt
    PCIeOneshotAllReduce._allocate_shared_buffer = staticmethod(unexpected_allocate)
    try:
        with pytest.raises(RuntimeError, match="pre-allocation setup"):
            PCIeDCPA2A.from_process_group(
                process_group=dist.group.WORLD,
                device=device,
                max_batch_size=MAX_BATCH,
                total_heads=TOTAL_HEADS,
                head_dim=HEAD_DIM,
                query_head_dim=QUERY_HEAD_DIM,
                ext_module=FakeExt(),
            )
        dist.barrier()
    finally:
        pcie_dcp_a2a.CudaRTLibrary = original_cuda_rt
        PCIeOneshotAllReduce._allocate_shared_buffer = original_allocate
        dist.destroy_process_group()


def test_pcie_dcp_a2a_eager_and_cuda_graph_correctness():
    if (
        os.getenv("B12X_PCIE_DCP_TEST_EXPECT_RESIDENCY_REJECTION") == "1"
        or os.getenv("B12X_PCIE_DCP_TEST_EXPECT_PREFLIGHT_REJECTION") == "1"
    ):
        pytest.skip("running the reduced-SM residency rejection gate")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = int(os.getenv("B12X_PCIE_DCP_A2A_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 8, 16):
        pytest.skip("PCIe DCP A2A supports world sizes 2, 4, 8, and 16")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


def test_pcie_dcp_a2a_rejects_reduced_sm_slice_before_ipc_allocation():
    if os.getenv("B12X_PCIE_DCP_TEST_EXPECT_RESIDENCY_REJECTION") != "1":
        pytest.skip("set the reduced-SM residency rejection gate to run this test")
    visible_sms = int(os.getenv("B12X_PCIE_TEST_VISIBLE_SM_COUNT", "0") or "0")
    assert 0 < visible_sms < pcie_dcp_a2a.DCP_A2A_REQUIRED_SMS, (
        "set B12X_PCIE_TEST_VISIBLE_SM_COUNT below "
        f"{pcie_dcp_a2a.DCP_A2A_REQUIRED_SMS} to exercise the residency gate"
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = int(os.getenv("B12X_PCIE_DCP_A2A_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 8, 16):
        pytest.skip("PCIe DCP A2A supports world sizes 2, 4, 8, and 16")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _residency_rejection_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


def test_pcie_dcp_a2a_coordinates_one_rank_preflight_failure_before_ipc():
    if os.getenv("B12X_PCIE_DCP_TEST_EXPECT_PREFLIGHT_REJECTION") != "1":
        pytest.skip("set the one-rank preflight rejection gate to run this test")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    world_size = int(os.getenv("B12X_PCIE_DCP_A2A_WORLD_SIZE", "2"))
    if world_size not in (2, 4, 8):
        pytest.skip("PCIe DCP A2A supports world sizes 2, 4, and 8")
    if torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need {world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _preflight_rejection_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )
