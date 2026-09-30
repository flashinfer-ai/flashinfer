from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuda.bindings import runtime as cudart

from b12x.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    owner_stage_reference,
)


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_TOPK_TEST") != "1",
    reason="set B12X_RUN_PCIE_DCP_TOPK_TEST=1 to run GPU tests",
)

MAX_ROWS = 64
TOPK = 2048


def _cuda_graph_kernel_count(graph: torch.cuda.CUDAGraph) -> int:
    graph_handle = graph.raw_cuda_graph()
    result, _, num_nodes = cudart.cudaGraphGetNodes(graph_handle)
    assert result == cudart.cudaError_t.cudaSuccess
    result, nodes, returned_nodes = cudart.cudaGraphGetNodes(
        graph_handle,
        num_nodes,
    )
    assert result == cudart.cudaError_t.cudaSuccess
    assert returned_nodes == num_nodes
    kernel_type = cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    kernel_count = 0
    for node in nodes[:num_nodes]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        assert result == cudart.cudaError_t.cudaSuccess
        kernel_count += node_type == kernel_type
    return kernel_count


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _inputs(
    step: int, global_rank: int, rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK)
    indices = (base + global_rank * 1_000_000 + step * 10_000).to(device)
    score_bits = (
        torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK)
        + 0x3E800000
        + global_rank * 4096
        + step * 64
    )
    return indices, score_bits.to(device).view(torch.float32)


def _dcp_groups(tp_world_size: int, dcp_world_size: int):
    return [
        dist.new_group(
            list(range(start, start + dcp_world_size)),
            backend="nccl",
        )
        for start in range(0, tp_world_size, dcp_world_size)
    ]


def _worker(
    rank: int,
    tp_world_size: int,
    dcp_world_size: int,
    port: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=tp_world_size,
    )
    groups = _dcp_groups(tp_world_size, dcp_world_size)
    dcp_partition = rank // dcp_world_size
    dcp_rank = rank % dcp_world_size
    dcp_group = groups[dcp_partition]
    dcp_global_ranks = list(
        range(
            dcp_partition * dcp_world_size,
            (dcp_partition + 1) * dcp_world_size,
        )
    )

    max_rows = (MAX_ROWS // dcp_world_size) * dcp_world_size
    test_rows = sorted(
        {
            dcp_world_size,
            2 * dcp_world_size,
            4 * dcp_world_size,
            max_rows,
        }
    )
    owner = PCIeDCPTopKOwnerExchange.from_process_group(
        process_group=dcp_group,
        device=device,
        max_rows=max_rows,
        topk=TOPK,
    )
    graph_owner = PCIeDCPTopKOwnerExchange.from_process_group(
        process_group=dcp_group,
        device=device,
        max_rows=max_rows,
        topk=TOPK,
    )
    try:
        eager_output_ptrs = []
        for step, rows in enumerate(test_rows):
            local_indices, local_scores = _inputs(step, rank, rows, device)
            wrong_device = (rank + 1) % tp_world_size
            if step == 0:
                torch.cuda.set_device(wrong_device)
            candidate_indices, candidate_scores = owner.stage_candidates(
                local_indices, local_scores
            )
            if step == 0:
                assert torch.cuda.current_device() == wrong_device
                torch.cuda.set_device(rank)
            eager_output_ptrs.append(candidate_indices.data_ptr())
            torch.cuda.synchronize(device)

            rank_inputs = [
                _inputs(step, source, rows, device) for source in dcp_global_ranks
            ]
            expected_indices, expected_scores = owner_stage_reference(
                torch.stack([item[0] for item in rank_inputs]),
                torch.stack([item[1] for item in rank_inputs]),
                dcp_rank,
            )
            if not torch.equal(candidate_indices, expected_indices):
                mismatch = (candidate_indices != expected_indices).nonzero()
                details = [
                    (
                        tuple(int(value) for value in index.tolist()),
                        int(candidate_indices[tuple(index)].item()),
                        int(expected_indices[tuple(index)].item()),
                    )
                    for index in mismatch[:16]
                ]
                raise AssertionError(
                    f"rank={rank} step={step} rows={rows} index mismatches={details}"
                )
            torch.testing.assert_close(
                candidate_indices, expected_indices, rtol=0, atol=0
            )
            assert torch.equal(
                candidate_scores.view(torch.int32),
                expected_scores.view(torch.int32),
            )
        assert eager_output_ptrs[0] != eager_output_ptrs[1]
        assert eager_output_ptrs[0] == eager_output_ptrs[2]
        assert eager_output_ptrs[1] == eager_output_ptrs[3]
        graph_rows = max_rows
        graph_indices, graph_scores = _inputs(10, rank, graph_rows, device)
        graph_owner_rows = graph_rows // dcp_world_size
        graph_candidate_width = dcp_world_size * TOPK
        consumed_indices = torch.empty(
            (graph_owner_rows, graph_candidate_width),
            dtype=torch.int32,
            device=device,
        )
        consumed_scores = torch.empty(
            (graph_owner_rows, graph_candidate_width),
            dtype=torch.float32,
            device=device,
        )
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        dist.barrier()
        with graph_owner.capture(), torch.cuda.graph(graph):
            graph_candidate_indices, graph_candidate_scores = (
                graph_owner.stage_candidates(graph_indices, graph_scores)
            )
            # Force one owner per DCP group to remain in its consumer while
            # faster peers are ready to replay and write its staging slab.
            if dcp_rank == 0:
                torch.cuda._sleep(20_000_000)
            consumed_indices.copy_(graph_candidate_indices)
            consumed_scores.copy_(graph_candidate_scores)

        # The transport remains one kernel node. Rank zero's deliberate skew
        # contributes the only additional kernel; the two consumers are memcpy
        # nodes. The fixed-slot safety barrier is part of the transport kernel.
        assert _cuda_graph_kernel_count(graph) == 1 + int(dcp_rank == 0)
        assert graph_owner._graph_slot is not None
        graph_views = graph_owner._candidate_views[graph_owner._graph_slot]
        assert (
            graph_candidate_indices.data_ptr()
            == graph_views[0].data_ptr()
        )
        assert (
            graph_candidate_scores.data_ptr()
            == graph_views[1].data_ptr()
        )
        assert graph_candidate_indices.data_ptr() == graph_owner._staging_ptrs[
            graph_owner._graph_slot
        ][dcp_rank]

        # Queue several replays without host/device synchronization. Every
        # replay writes the capture-stable slab, then consumes it on the same
        # stream. The next replay's in-kernel peer barrier prevents any rank
        # from overwriting it until all prior consumers have retired.
        replay_steps = tuple(range(20, 27))
        replay_inputs = [
            _inputs(replay_step, rank, graph_rows, device)
            for replay_step in replay_steps
        ]
        replayed_indices = [torch.empty_like(consumed_indices) for _ in replay_steps]
        replayed_scores = [torch.empty_like(consumed_scores) for _ in replay_steps]
        torch.cuda.synchronize(device)
        for replay_idx in range(len(replay_steps)):
            next_indices, next_scores = replay_inputs[replay_idx]
            graph_indices.copy_(next_indices)
            graph_scores.copy_(next_scores)
            graph.replay()
            replayed_indices[replay_idx].copy_(consumed_indices)
            replayed_scores[replay_idx].copy_(consumed_scores)
        torch.cuda.synchronize(device)

        for replay_idx, replay_step in enumerate(replay_steps):
            expected_rank_inputs = [
                _inputs(replay_step, source, graph_rows, device)
                for source in dcp_global_ranks
            ]
            expected_indices, expected_scores = owner_stage_reference(
                torch.stack([item[0] for item in expected_rank_inputs]),
                torch.stack([item[1] for item in expected_rank_inputs]),
                dcp_rank,
            )
            torch.testing.assert_close(
                replayed_indices[replay_idx], expected_indices, rtol=0, atol=0
            )
            assert torch.equal(
                replayed_scores[replay_idx].view(torch.int32),
                expected_scores.view(torch.int32),
            )
        dist.barrier()
        torch.cuda.synchronize(device)
    finally:
        graph_owner.close_coordinated()
        owner.close_coordinated()
        dist.destroy_process_group()


def test_pcie_dcp_topk_exact_owner_exchange():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp_world_size = int(os.getenv("B12X_PCIE_DCP_TOPK_TP", "8"))
    dcp_world_size = int(os.getenv("B12X_PCIE_DCP_TOPK_DCP", "4"))
    if (
        tp_world_size not in (2, 3, 4, 6, 8)
        or dcp_world_size not in (2, 3, 4, 6, 8)
        or tp_world_size % dcp_world_size != 0
    ):
        pytest.skip("unsupported TP/DCP top-k geometry")
    if torch.cuda.device_count() < tp_world_size:
        pytest.skip(f"need {tp_world_size} CUDA devices")
    mp.spawn(
        _worker,
        args=(tp_world_size, dcp_world_size, _free_port()),
        nprocs=tp_world_size,
        join=True,
    )
