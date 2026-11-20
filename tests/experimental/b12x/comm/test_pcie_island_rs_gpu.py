from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_allreduce import (
    ISLAND_RS_CROSSOVER_ELEMENTS,
    ISLAND_RS_MAX_BYTES,
    ISLAND_RS_PREFERRED_ALIGNMENT_ELEMENTS,
    PCIeAllReduce,
)
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from b12x.comm.pcie._island_rs_cute import HEADER_BYTES, island_rs_peers


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ISLAND_RS_TEST") != "1",
    reason="set B12X_RUN_PCIE_ISLAND_RS_TEST=1 to run the TP16 GPU test",
)

WORLD_SIZE = 16
SHAPES = (2, 6, 7_168, 14_336, 14_338, 28_672, 57_344, 81_918, 81_920)


def _expected_pci_bus_islands() -> tuple[tuple[int, ...], ...]:
    raw = os.getenv("B12X_PCIE_ISLAND_RS_EXPECTED_PCI_BUS_ISLANDS")
    if raw is None:
        raise RuntimeError(
            "B12X_PCIE_ISLAND_RS_EXPECTED_PCI_BUS_ISLANDS must describe four "
            "rank-ordered groups of four hexadecimal PCI bus IDs"
        )
    islands = tuple(
        tuple(int(bus.strip(), 0) for bus in island.split(","))
        for island in raw.split("|")
    )
    if len(islands) != 4 or any(len(island) != 4 for island in islands):
        raise ValueError(
            "B12X_PCIE_ISLAND_RS_EXPECTED_PCI_BUS_ISLANDS must contain four "
            "groups of four PCI bus IDs"
        )
    return islands


def _allocator_replay_state(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    keys = (
        "allocation.all.allocated",
        "allocated_bytes.all.allocated",
        "reserved_bytes.all.allocated",
        "segment.all.allocated",
    )
    return {key: int(stats[key]) for key in keys}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reference(inp: torch.Tensor, world_size: int) -> torch.Tensor:
    gathered = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(gathered, inp)
    return torch.stack([value.float() for value in gathered]).sum(dim=0)


def _assert_result(
    actual: torch.Tensor,
    expected: torch.Tensor,
    inp: torch.Tensor,
    input_before: torch.Tensor,
) -> None:
    assert torch.equal(inp, input_before)
    rank_zero = actual.clone()
    dist.broadcast(rank_zero, src=0)
    assert torch.equal(actual, rank_zero)
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=0.02,
        atol=0.125,
    )


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    bootstrap = torch.zeros(max(SHAPES), dtype=torch.bfloat16, device=device)
    dist.all_reduce(bootstrap)
    torch.cuda.synchronize(device)

    dcp_pool = None
    runtime = None
    try:
        dcp_pool = PCIeDCPA2APool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_batch_size=1,
            total_heads=96,
            head_dim=512,
            query_head_dim=576,
        )
        runtime = PCIeAllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=device,
            max_size=ISLAND_RS_MAX_BYTES,
        )
        assert runtime.algorithm == "hierarchical"
        assert runtime._algorithm_override == "island_rs"
        assert runtime._island_rs is not None
        assert runtime._island_rs.mapped_peer_count == 6
        assert runtime._island_rs.mapped_peers == island_rs_peers(rank, world_size)
        quarter_capacity = runtime._island_rs.quarter_capacity
        vector_bytes = quarter_capacity * 4 * 4
        expected_slab_bytes = (
            (runtime._island_rs.final_offset + 2 * vector_bytes + 255) // 256 * 256
        )
        assert runtime._island_rs.stage_offset == (HEADER_BYTES + 255) // 256 * 256
        assert runtime._island_rs.slab_bytes == expected_slab_bytes

        inputs: list[torch.Tensor] = []
        references: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        inputs_before: list[torch.Tensor] = []
        for shape in SHAPES:
            generator = torch.Generator(device=device).manual_seed(
                0x1A51_4D + rank * 17 + shape
            )
            inp = torch.randn(
                shape,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            inputs.append(inp)
            references.append(_reference(inp, world_size))
            outputs.append(torch.empty_like(inp))
            inputs_before.append(inp.clone())

        for inp, out, expected, before in zip(
            inputs,
            outputs,
            references,
            inputs_before,
            strict=True,
        ):
            use_island = (
                inp.numel() > ISLAND_RS_CROSSOVER_ELEMENTS
                and inp.numel() % ISLAND_RS_PREFERRED_ALIGNMENT_ELEMENTS == 0
            )
            assert runtime._use_island_rs(inp) is use_island
            actual = runtime.all_reduce(inp, out=out)
            assert actual is out
            torch.cuda.synchronize(device)
            _assert_result(actual, expected, inp, before)

        # Exercise an unaligned quarter boundary without a host synchronization
        # between generations. The two device-selected workspace slots must
        # prevent a faster rank from overwriting a slower peer's input.
        stress_index = SHAPES.index(14_338)
        for _ in range(32):
            runtime.all_reduce(
                inputs[stress_index],
                out=outputs[stress_index],
            )
        torch.cuda.synchronize(device)
        _assert_result(
            outputs[stress_index],
            references[stress_index],
            inputs[stress_index],
            inputs_before[stress_index],
        )

        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=device)
        input_ptrs = tuple(tensor.data_ptr() for tensor in inputs)
        output_ptrs = tuple(tensor.data_ptr() for tensor in outputs)
        slab_ptrs = tuple(runtime._island_rs._slab_ptrs)
        with (
            runtime.capture(stream=stream, channel_id="target"),
            torch.cuda.graph(graph, stream=stream),
        ):
            for inp, out in zip(inputs, outputs, strict=True):
                runtime.all_reduce(
                    inp,
                    out=out,
                    stream=stream,
                    channel_id="target",
                )
        graph_pool = graph.pool()
        for out in outputs:
            out.fill_(float("nan"))
        torch.cuda.synchronize(device)
        allocator_before = _allocator_replay_state(device)
        for _ in range(100):
            graph.replay()
        torch.cuda.synchronize(device)
        allocator_after = _allocator_replay_state(device)
        assert allocator_after == allocator_before
        assert tuple(tensor.data_ptr() for tensor in inputs) == input_ptrs
        assert tuple(tensor.data_ptr() for tensor in outputs) == output_ptrs
        assert tuple(runtime._island_rs._slab_ptrs) == slab_ptrs
        assert graph.pool() == graph_pool
        assert runtime._island_rs.slab_bytes == expected_slab_bytes
        for inp, out, expected, before in zip(
            inputs,
            outputs,
            references,
            inputs_before,
            strict=True,
        ):
            _assert_result(out, expected, inp, before)

        implicit_index = SHAPES.index(28_672)
        implicit_graph = torch.cuda.CUDAGraph()
        implicit_stream = torch.cuda.Stream(device=device)
        with runtime.capture(
            stream=implicit_stream,
            channel_id="target-implicit-output",
        ) as captured:
            assert captured is runtime
            with torch.cuda.graph(implicit_graph, stream=implicit_stream):
                implicit_output = captured.all_reduce(
                    inputs[implicit_index],
                    stream=implicit_stream,
                    channel_id="target-implicit-output",
                )
        implicit_output_ptr = implicit_output.data_ptr()
        implicit_graph_pool = implicit_graph.pool()
        implicit_output.fill_(float("nan"))
        torch.cuda.synchronize(device)
        implicit_allocator_before = _allocator_replay_state(device)
        for _ in range(100):
            implicit_graph.replay()
        torch.cuda.synchronize(device)
        implicit_allocator_after = _allocator_replay_state(device)
        assert implicit_allocator_after == implicit_allocator_before
        assert implicit_output.data_ptr() == implicit_output_ptr
        assert implicit_graph.pool() == implicit_graph_pool
        _assert_result(
            implicit_output,
            references[implicit_index],
            inputs[implicit_index],
            inputs_before[implicit_index],
        )
    finally:
        if runtime is not None:
            runtime.close()
        if dcp_pool is not None:
            dcp_pool.close()
        dist.destroy_process_group()


def test_tp16_island_reduce_scatter_eager_and_graph_correctness() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    assert torch.cuda.device_count() == WORLD_SIZE
    expected_islands = _expected_pci_bus_islands()
    actual_bus_ids = tuple(
        int(torch.cuda.get_device_properties(index).pci_bus_id)
        for index in range(WORLD_SIZE)
    )
    assert actual_bus_ids == tuple(bus for island in expected_islands for bus in island)
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _free_port()),
        nprocs=WORLD_SIZE,
        join=True,
    )
