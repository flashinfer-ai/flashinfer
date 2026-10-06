from __future__ import annotations

import ctypes
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from b12x._lib.runtime_control import kernel_resolution_guard

from b12x.comm.pcie.pcie_oneshot import (
    TP2_PLAIN_REMOTE_PUSH_MAX_BYTES,
    PCIeOneshotAllReducePool,
    _tp2_plain_remote_push_enabled,
)
from b12x.comm.pcie._oneshot_cute import (
    _PLAIN_GRAPH_ARRIVED_OFFSET,
    _PLAIN_GRAPH_EPOCH_OFFSET,
)


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ONESHOT_TORTURE") != "1",
    reason="set B12X_RUN_PCIE_ONESHOT_TORTURE=1 to run PCIe oneshot CUDA torture tests",
)

TORTURE_EAGER_ITERS = int(os.getenv("B12X_PCIE_ONESHOT_TORTURE_EAGER_ITERS", "256"))
TORTURE_GRAPH_REPLAYS = int(os.getenv("B12X_PCIE_ONESHOT_TORTURE_GRAPH_REPLAYS", "256"))
TORTURE_MULTISTREAM_ITERS = int(
    os.getenv("B12X_PCIE_ONESHOT_TORTURE_MULTISTREAM_ITERS", "256")
)
TORTURE_PAYLOAD_ITERS = int(os.getenv("B12X_PCIE_ONESHOT_TORTURE_PAYLOAD_ITERS", "64"))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _assert_constant(tensor: torch.Tensor, value: float) -> None:
    expected = torch.full_like(tensor, value)
    torch.testing.assert_close(tensor, expected, rtol=1e-2, atol=1e-2)


def _rank_sum(world_size: int) -> int:
    return world_size * (world_size - 1) // 2


def _tp2_plain_remote_push_active(world_size: int) -> bool:
    return world_size == 2 and _tp2_plain_remote_push_enabled()


def _local_eager_words(
    channel, stream: torch.cuda.Stream, *, offset: int = 0
) -> tuple[int, int]:
    assert channel._ipc is not None
    assert channel._eager_ptrs is not None
    words = (ctypes.c_uint64(), ctypes.c_uint64())
    for word, slot_ptrs in zip(words, channel._eager_ptrs, strict=True):
        channel._ipc.cudaMemcpyAsync(
            ctypes.addressof(word),
            slot_ptrs[channel.rank] + offset,
            ctypes.sizeof(word),
            int(stream.cuda_stream),
        )
    stream.synchronize()
    return words[0].value, words[1].value


def _copy_host_to_device(
    channel, destination: int, source: ctypes.Array, stream: torch.cuda.Stream
) -> None:
    channel._ipc.cudaMemcpyAsync(
        destination,
        ctypes.addressof(source),
        ctypes.sizeof(source),
        int(stream.cuda_stream),
    )


def _read_local_u32(channel, source: int, stream: torch.cuda.Stream) -> int:
    value = ctypes.c_uint32()
    channel._ipc.cudaMemcpyAsync(
        ctypes.addressof(value),
        source,
        ctypes.sizeof(value),
        int(stream.cuda_stream),
    )
    stream.synchronize()
    return value.value


def _run_eager(
    pool: PCIeOneshotAllReducePool, device: torch.device, rank: int, world_size: int
) -> None:
    remote_push = _tp2_plain_remote_push_active(world_size)
    dtypes = (
        (torch.float16, torch.bfloat16)
        if remote_push
        else (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )
    )
    shapes = (
        tuple((rows, 4096) for rows in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512))
        if remote_push
        else ((8,), (256,), (4096,), (32768,))
    )
    rank_sum = _rank_sum(world_size)

    for dtype in dtypes:
        for shape in shapes:
            inp = torch.empty(shape, device=device, dtype=dtype)
            out = torch.empty_like(inp)
            for iteration in range(TORTURE_EAGER_ITERS):
                base = float((iteration % 64) * 3)
                inp.fill_(base + rank)
                pool.all_reduce(inp, out=out, channel_id="eager:default")
                torch.cuda.synchronize(device)
                _assert_constant(out, world_size * base + rank_sum)


def _run_graph_scratch_reuse(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    stream = torch.cuda.Stream(device=device)
    rank_sum = _rank_sum(world_size)
    layers = 17
    shape = (6, 4096) if _tp2_plain_remote_push_active(world_size) else (4096,)
    dtype = torch.bfloat16
    sources = [torch.empty(shape, device=device, dtype=dtype) for _ in range(layers)]
    scratch = torch.empty(shape, device=device, dtype=dtype)
    outs = [torch.empty(shape, device=device, dtype=dtype) for _ in range(layers)]

    def fill_sources(iteration: int) -> None:
        for layer, source in enumerate(sources):
            source.fill_(float((iteration % 32) * 5 + layer) + rank)

    fill_sources(0)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with pool.capture(stream, channel_id="graph:torture") as graph_channel:
        with torch.cuda.stream(stream):
            graph_channel.prepare_graph_all_reduce(scratch)
        with torch.cuda.graph(graph, stream=stream):
            for layer in range(layers):
                scratch.copy_(sources[layer])
                graph_channel.all_reduce(scratch, out=outs[layer])
    stream.synchronize()

    for iteration in range(TORTURE_GRAPH_REPLAYS):
        with torch.cuda.stream(stream):
            fill_sources(iteration)
            graph.replay()
        stream.synchronize()
        for layer, out in enumerate(outs):
            base = float((iteration % 32) * 5 + layer)
            _assert_constant(out, world_size * base + rank_sum)

    if _tp2_plain_remote_push_active(world_size):
        # The six-row input validates repeated generation publication and
        # alternating graph-slot reuse across multiple collectives per replay.
        return

    # A graph with an odd number of collectives must swap which slot receives
    # the final layer on every replay. Both slots are overwritten by this
    # 17-layer graph, so merely counting changed slots cannot identify the
    # selected parity: the final two layer markers must exchange positions.
    snapshots = [_local_eager_words(graph_channel, stream)]
    expected_words = [
        tuple(int(source.view(torch.uint64)[0].item()) for source in sources[-2:])
    ]
    for iteration in (101, 102):
        with torch.cuda.stream(stream):
            fill_sources(iteration)
            graph.replay()
        stream.synchronize()
        snapshots.append(_local_eager_words(graph_channel, stream))
        expected_words.append(
            tuple(int(source.view(torch.uint64)[0].item()) for source in sources[-2:])
        )
        for layer, out in enumerate(outs):
            base = float((iteration % 32) * 5 + layer)
            _assert_constant(out, world_size * base + rank_sum)

    final_slots = []
    for snapshot, expected in zip(snapshots, expected_words, strict=True):
        assert set(snapshot) == set(expected), (
            f"staging slots {snapshot} do not contain the final layer markers "
            f"{expected}"
        )
        final_slots.append(snapshot.index(expected[-1]))
    assert all(
        final_slots[index] != final_slots[index + 1]
        for index in range(len(final_slots) - 1)
    ), f"odd-collective graph did not alternate final staging slots: {final_slots}"


def _run_multistream(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    pool.for_stream(stream_a, channel_id="eager:a")
    pool.for_stream(stream_b, channel_id="eager:b")
    rank_sum = _rank_sum(world_size)

    remote_push = _tp2_plain_remote_push_active(world_size)
    shape_a = (1, 4096) if remote_push else (2048,)
    shape_b = (2, 4096) if remote_push else (2048,)
    inp_a = torch.empty(shape_a, device=device, dtype=torch.float16)
    out_a = torch.empty_like(inp_a)
    inp_b = torch.empty(shape_b, device=device, dtype=torch.bfloat16)
    out_b = torch.empty_like(inp_b)

    for iteration in range(TORTURE_MULTISTREAM_ITERS):
        base_a = float(iteration % 64)
        base_b = float(100 + (iteration % 64) * 2)
        with torch.cuda.stream(stream_a):
            inp_a.fill_(base_a + rank)
            pool.all_reduce(inp_a, out=out_a, channel_id="eager:a")
        with torch.cuda.stream(stream_b):
            inp_b.fill_(base_b + rank)
            pool.all_reduce(inp_b, out=out_b, channel_id="eager:b")
        stream_a.synchronize()
        stream_b.synchronize()
        _assert_constant(out_a, world_size * base_a + rank_sum)
        _assert_constant(out_b, world_size * base_b + rank_sum)


def _run_tp2_mixed_protocol_reuse(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Alternate pull, fused, and peer-push protocols on one eager channel."""

    if not _tp2_plain_remote_push_active(world_size):
        return
    channel_id = "eager:default"
    channel = pool.for_stream(channel_id=channel_id)
    rank_sum = _rank_sum(world_size)

    pull_input = torch.full((1, 6144), 11.0 + rank, device=device, dtype=torch.bfloat16)
    pull_output = pool.all_reduce(pull_input, channel_id=channel_id)
    torch.cuda.synchronize(device)
    _assert_constant(pull_output, world_size * 11.0 + rank_sum)

    fp32_input = torch.full((2048,), 13.0 + rank, device=device, dtype=torch.float32)
    fp32_output = pool.all_reduce(fp32_input, channel_id=channel_id)
    torch.cuda.synchronize(device)
    _assert_constant(fp32_output, world_size * 13.0 + rank_sum)

    push_input = torch.full((1, 4096), 17.0 + rank, device=device, dtype=torch.bfloat16)
    push_output = pool.all_reduce(push_input, channel_id=channel_id)
    torch.cuda.synchronize(device)
    _assert_constant(push_output, world_size * 17.0 + rank_sum)

    residual = torch.full_like(push_input, 0.25)
    weight = torch.ones(4096, device=device, dtype=torch.bfloat16)
    pool.all_reduce_fused_add_rms_norm(
        push_input,
        residual,
        weight,
        1e-6,
        channel_id=channel_id,
    )
    torch.cuda.synchronize(device)

    push_input.fill_(23.0 + rank)
    push_output = pool.all_reduce(push_input, channel_id=channel_id)
    torch.cuda.synchronize(device)
    _assert_constant(push_output, world_size * 23.0 + rank_sum)

    state = channel._ext._state(channel._ptr)
    assert state.plain_remote_push_region_packs > 0
    offset = state.plain_remote_push_region_packs * 16
    assert _local_eager_words(
        channel, torch.cuda.current_stream(device), offset=offset
    ) == (0, 0)


def _run_tp2_payload_patterns(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Exercise peer-push publication with zeros and lane-varying payloads."""

    if not _tp2_plain_remote_push_active(world_size):
        return
    elements = 4 * 4096
    lane = torch.arange(elements, device=device, dtype=torch.float32)
    output = torch.empty((4, 4096), device=device, dtype=torch.bfloat16)
    for iteration in range(TORTURE_PAYLOAD_ITERS):
        patterns = []
        for source_rank in range(world_size):
            pattern = (((lane * (source_rank + 3) + iteration * 11) % 257) - 128) / 16
            pattern[(lane.to(torch.int64) + iteration + source_rank) % 19 == 0] = 0.0
            pattern[(lane.to(torch.int64) + iteration + source_rank) % 23 == 0] = -0.0
            patterns.append(pattern.to(torch.bfloat16).view(4, 4096))
        inp = patterns[rank]
        expected = patterns[0] + patterns[1]
        pool.all_reduce(inp, out=output, channel_id="eager:default")
        torch.cuda.synchronize(device)
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


def _run_tp2_graph_payload_patterns(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Validate every qualified graph geometry with arbitrary payload bits."""

    if not _tp2_plain_remote_push_active(world_size):
        return
    tensor_specs = tuple(
        (rows, dtype)
        for dtype in (torch.bfloat16, torch.float16)
        for rows in (1, 2, 4, 6, 8, 16, 24, 32, 512)
    )
    lanes = [
        torch.arange(rows * 4096, device=device, dtype=torch.float32)
        for rows, _ in tensor_specs
    ]
    sources = [
        torch.empty((rows, 4096), device=device, dtype=dtype)
        for rows, dtype in tensor_specs
    ]
    scratches = [torch.empty_like(source) for source in sources]
    outputs = [torch.empty_like(source) for source in sources]
    stream = torch.cuda.Stream(device=device)

    graph = torch.cuda.CUDAGraph()
    for source in sources:
        source.zero_()
    with pool.capture(stream, channel_id="graph:payload") as graph_channel:
        with torch.cuda.stream(stream):
            for scratch in scratches:
                graph_channel.prepare_graph_all_reduce(scratch)
        with torch.cuda.graph(graph, stream=stream):
            for source, scratch, output in zip(
                sources, scratches, outputs, strict=True
            ):
                scratch.copy_(source)
                graph_channel.all_reduce(scratch, out=output)
    stream.synchronize()

    for iteration in range(TORTURE_PAYLOAD_ITERS):
        expected_outputs = []
        for shape_index, (lane, source) in enumerate(zip(lanes, sources, strict=True)):
            patterns = []
            for source_rank in range(world_size):
                pattern = (
                    ((lane * (source_rank + 3) + iteration * 11 + shape_index) % 257)
                    - 128
                ) / 16
                lane_index = lane.to(torch.int64)
                pattern[
                    (lane_index + iteration + source_rank + shape_index) % 19 == 0
                ] = 0.0
                pattern[
                    (lane_index + iteration + source_rank + shape_index) % 23 == 0
                ] = -0.0
                pattern[0] = 0.0
                pattern[1] = -0.0
                patterns.append(pattern.to(source.dtype).view_as(source))
            source.copy_(patterns[rank])
            expected_outputs.append(patterns[0] + patterns[1])
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        for output, expected in zip(outputs, expected_outputs, strict=True):
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


def _run_tp2_graph_weak_contiguous_storage(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Validate graph peer-push over a dense, non-contiguous logical view."""

    source = torch.empty((4096, 4), device=device, dtype=torch.bfloat16).transpose(0, 1)
    scratch = torch.empty_like(source)
    output = torch.empty_like(source)
    assert not source.is_contiguous()
    assert source.untyped_storage().nbytes() == source.numel() * source.element_size()

    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    with pool.capture(stream, channel_id="graph:weak-contiguous") as graph_channel:
        with torch.cuda.stream(stream):
            graph_channel.prepare_graph_all_reduce(scratch)
        with torch.cuda.graph(graph, stream=stream):
            scratch.copy_(source)
            graph_channel.all_reduce(scratch, out=output)
    stream.synchronize()

    lane = torch.arange(source.numel(), device=device, dtype=torch.float32).view_as(
        source
    )
    for iteration in range(4):
        patterns = tuple(
            (((lane + 7 * source_rank + 3 * iteration) % 127) - 63)
            .div(8)
            .to(source.dtype)
            for source_rank in range(world_size)
        )
        source.copy_(patterns[rank])
        expected = patterns[0] + patterns[1]
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        assert torch.equal(output, expected)


def _run_tp2_graph_generation_wrap(
    pool: PCIeOneshotAllReducePool,
    device: torch.device,
    rank: int,
    world_size: int,
) -> None:
    """Replay the peer-push kernel across uint32 generation wraparound."""

    if not _tp2_plain_remote_push_active(world_size):
        return
    source = torch.empty((4, 4096), device=device, dtype=torch.bfloat16)
    scratch = torch.empty_like(source)
    output = torch.empty_like(source)
    stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()

    with pool.capture(stream, channel_id="graph:wrap") as graph_channel:
        with torch.cuda.stream(stream):
            graph_channel.prepare_graph_all_reduce(scratch)
        with torch.cuda.graph(graph, stream=stream):
            scratch.copy_(source)
            graph_channel.all_reduce(scratch, out=output)
    stream.synchronize()

    backend_state = graph_channel._ext._state(graph_channel._ptr)
    start_generation = 0xFFFFFFFC
    selected_slot = (start_generation + backend_state.slot_bias) & 1
    size_packs = source.numel() * source.element_size() // 16
    record_words = size_packs * 8
    record_offset = backend_state.plain_remote_push_region_packs * 16

    for slot in range(2):
        # A slot is reused every two generations. Before generation g, its
        # retained marker is g; the other slot most recently published g + 1.
        retained_generation = (
            start_generation if slot == selected_slot else start_generation + 1
        ) & 0xFFFFFFFF
        records = (ctypes.c_uint32 * record_words)()
        for record in range(size_packs):
            base = record * 8
            for lane in (1, 3, 5, 7):
                records[base + lane] = retained_generation
        _copy_host_to_device(
            graph_channel,
            graph_channel._eager_ptrs[slot][rank] + record_offset,
            records,
            stream,
        )

    control = (ctypes.c_uint32 * 2)(start_generation, 0)
    assert _PLAIN_GRAPH_ARRIVED_OFFSET == _PLAIN_GRAPH_EPOCH_OFFSET + 4
    _copy_host_to_device(
        graph_channel,
        graph_channel.signal_ptrs[rank] + _PLAIN_GRAPH_EPOCH_OFFSET,
        control,
        stream,
    )
    stream.synchronize()
    dist.barrier()

    lane = torch.arange(source.numel(), device=device, dtype=torch.float32)
    for iteration in range(4):
        patterns = []
        for source_rank in range(world_size):
            pattern = (((lane + 3 * source_rank + 5 * iteration) % 127) - 63) / 8
            pattern[0] = 0.0
            pattern[1] = -0.0
            patterns.append(pattern.to(source.dtype).view_as(source))
        source.copy_(patterns[rank])
        expected = patterns[0] + patterns[1]
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))

    assert (
        _read_local_u32(
            graph_channel,
            graph_channel.signal_ptrs[rank] + _PLAIN_GRAPH_EPOCH_OFFSET,
            stream,
        )
        == 2
    )


def _graph_payload_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["B12X_PCIE_TP2_PLAIN_REMOTE_PUSH"] = "1"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=TP2_PLAIN_REMOTE_PUSH_MAX_BYTES,
        max_concurrent_channels=2,
    )
    try:
        pool.prepare_channels(("graph:payload", "graph:weak-contiguous", "graph:wrap"))
        _run_tp2_graph_payload_patterns(pool, device, rank, world_size)
        dist.barrier()
        _run_tp2_graph_weak_contiguous_storage(pool, device, rank, world_size)
        dist.barrier()
        _run_tp2_graph_generation_wrap(pool, device, rank, world_size)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=(
            TP2_PLAIN_REMOTE_PUSH_MAX_BYTES
            if _tp2_plain_remote_push_active(world_size)
            else 1 << 20
        ),
        max_concurrent_channels=2,
    )
    try:
        pool.prepare_channels(
            (
                "eager:default",
                "eager:a",
                "eager:b",
                "graph:torture",
            )
        )
        _run_eager(pool, device, rank, world_size)
        dist.barrier()
        _run_tp2_mixed_protocol_reuse(pool, device, rank, world_size)
        dist.barrier()
        _run_tp2_payload_patterns(pool, device, rank, world_size)
        dist.barrier()
        _run_graph_scratch_reuse(pool, device, rank, world_size)
        dist.barrier()
        _run_multistream(pool, device, rank, world_size)
        torch.cuda.synchronize(device)
    finally:
        pool.close()
        dist.destroy_process_group()


def test_pcie_oneshot_eager_graph_and_multistream_torture():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    available = torch.cuda.device_count()
    requested = int(os.getenv("B12X_PCIE_ONESHOT_TORTURE_WORLD_SIZE", "2"))
    if requested not in (2, 4, 6, 8, 10):
        pytest.skip("PCIe oneshot only supports world sizes 2, 4, 6, 8, and 10")
    if available < requested:
        pytest.skip(f"need {requested} CUDA devices, found {available}")
    mp.spawn(_worker, args=(requested, _free_port()), nprocs=requested, join=True)


def test_tp2_graph_peer_push_preserves_payload_bits(monkeypatch: pytest.MonkeyPatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("TP2 graph peer-push requires two CUDA devices")
    monkeypatch.setenv("B12X_PCIE_TP2_PLAIN_REMOTE_PUSH", "1")
    mp.spawn(_graph_payload_worker, args=(2, _free_port()), nprocs=2, join=True)


def _shape_growth_wrap_worker(
    rank: int, world_size: int, port: int, dtype_name: str, retained_tag: int,
) -> None:
    import b12x

    os.environ["B12X_PCIE_TP2_PLAIN_REMOTE_PUSH"] = "1"
    os.environ["B12X_PCIE_TP2_REMOTE_PUSH"] = "1" if retained_tag else "0"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}",
        rank=rank, world_size=world_size,
    )
    capacity_bytes = 4 * 4096 * 2
    pool = PCIeOneshotAllReducePool(
        rank=rank, world_size=world_size, device=device,
        exchange_group=dist.group.WORLD, eager_buffer_bytes=capacity_bytes,
        max_size=capacity_bytes, rank_data_bytes=capacity_bytes, single_channel=True,
    )
    stream = torch.cuda.Stream(device=device)
    graphs, inputs, outputs = [], [], []
    try:
        for rows in (1, 4):
            source = torch.full(
                (rows, 4096), float(rank + 1), device=device,
                dtype=getattr(torch, dtype_name),
            )
            output = torch.empty_like(source)
            stream.wait_stream(torch.cuda.current_stream(device))
            with pool.capture(stream) as channel:
                with torch.cuda.stream(stream):
                    channel.prepare_graph_all_reduce(source)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    channel.all_reduce(source, out=output)
            stream.synchronize()
            graphs.append(graph)
            inputs.append(source)
            outputs.append(output)

        state = channel._ext._state(channel._ptr)
        records = (ctypes.c_uint32 * (capacity_bytes // 16 * 8))()
        for record in range(capacity_bytes // 16):
            for lane in (1, 3, 5, 7):
                records[record * 8 + lane] = retained_tag
        for slot in range(2):
            _copy_host_to_device(
                channel,
                channel._eager_ptrs[slot][rank] + state.plain_remote_push_region_packs * 16,
                records, stream,
            )
        control = (ctypes.c_uint32 * 2)(0xFFFFFFFC, 0)
        _copy_host_to_device(
            channel, channel.signal_ptrs[rank] + _PLAIN_GRAPH_EPOCH_OFFSET,
            control, stream,
        )
        stream.synchronize()
        dist.barrier()
        with kernel_resolution_guard('TP2 shape growth across epoch rollover'):
            with torch.cuda.stream(stream):
                graphs[0].replay()
                graphs[0].replay()
            stream.synchronize()
            assert torch.all(outputs[0] == 3)
            assert _read_local_u32(
                channel, channel.signal_ptrs[rank] + _PLAIN_GRAPH_EPOCH_OFFSET, stream,
            ) == 0
            dist.barrier()
            address = outputs[1].data_ptr()
            allocated = torch.cuda.memory_allocated(device)
            with torch.cuda.stream(stream):
                # Force the receiver to observe stale tail records before its
                # peer publishes the larger shape after rollover.
                if rank == 1:
                    torch.cuda._sleep(20_000_000)
                graphs[1].replay()
            stream.synchronize()
            assert outputs[1].data_ptr() == address
            assert torch.cuda.memory_allocated(device) == allocated
            mismatches = int((outputs[1] != 3).sum().item())
            results = [None] * world_size
            dist.all_gather_object(results, mismatches)
            assert results == [0, 0], results
    finally:
        for graph in graphs:
            graph.reset()
        pool.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("retained_tag", [0, 2])
def test_tp2_graph_rollover_clears_unused_shape_capacity(dtype_name, retained_tag):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("TP2 graph peer-push requires two CUDA devices")
    mp.spawn(
        _shape_growth_wrap_worker,
        args=(2, _free_port(), dtype_name, retained_tag), nprocs=2, join=True,
    )
