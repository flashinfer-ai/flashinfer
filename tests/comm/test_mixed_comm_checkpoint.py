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

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import flashinfer.comm.mixed_comm as mixed_comm
from flashinfer.comm.mixed_comm import (
    MixedCommHandler,
    MixedCommMode,
    MixedCommOp,
    run_mixed_comm,
)
from flashinfer.comm.mnnvl import TorchDistBackend
from flashinfer.utils import get_compute_capability


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _addresses(handler: MixedCommHandler) -> tuple:
    return (
        tuple(int(ptr) for ptr in handler.uc_ptr_list),
        tuple(int(ptr) for ptr in handler.mc_ptr_dict.values() if ptr is not None),
        int(handler.uc_buffer_dict["raw"]),
    )


def _set_input(
    op: MixedCommOp,
    input_: torch.Tensor,
    rank: int,
    world_size: int,
    offset: int,
) -> torch.Tensor:
    if op == MixedCommOp.ALLGATHER:
        input_.fill_(rank + offset)
        return torch.cat(
            [
                torch.full_like(input_, peer_rank + offset)
                for peer_rank in range(world_size)
            ]
        )

    rows_per_rank = input_.shape[0] // world_size
    row_values = torch.arange(
        input_.shape[0], dtype=input_.dtype, device=input_.device
    ).view(-1, 1)
    input_.copy_(row_values + rank + offset)
    expected = row_values * world_size + sum(range(world_size)) + offset * world_size
    return expected[rank * rows_per_rank : (rank + 1) * rows_per_rank].expand(
        -1, input_.shape[1]
    )


def _run_checkpoint_worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )

    def _make_handler() -> MixedCommHandler:
        return MixedCommHandler(
            world_rank=rank,
            world_size=world_size,
            local_rank=rank,
            local_size=world_size,
            inter_rank=0,
            inter_size=1,
            local_tp_size=1,
            local_dp_size=world_size,
            inter_tp_size=1,
            inter_dp_size=1,
            dtype=torch.float16,
            device=torch.device("cuda", rank),
            use_autotune=False,
        )

    handler = _make_handler()
    initial_addresses = _addresses(handler)
    op_modes = tuple(
        (op, mode)
        for op in (MixedCommOp.ALLGATHER, MixedCommOp.REDUCESCATTER)
        for mode in (
            MixedCommMode.FUSED_OPT_WAITS_UC,
            MixedCommMode.FUSED_OPT_WAITS_MC,
        )
    )
    graphs = {}
    inputs = {}
    outputs = {}

    for op, mode in op_modes:
        input_rows = 4 if op == MixedCommOp.ALLGATHER else 4 * world_size
        input_ = torch.empty((input_rows, 64), dtype=torch.float16, device="cuda")
        output_rows = (
            input_rows * world_size
            if op == MixedCommOp.ALLGATHER
            else input_rows // world_size
        )
        output = torch.empty((output_rows, 64), dtype=torch.float16, device="cuda")
        expected = _set_input(op, input_, rank, world_size, offset=1)
        run_mixed_comm(op, handler, input_, output, mode)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            run_mixed_comm(op, handler, input_, output, mode)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        key = (op, mode)
        graphs[key] = graph
        inputs[key] = input_
        outputs[key] = output

    handler.checkpoint_prepare()
    handler.checkpoint_prepare()
    assert not handler._vm_mapped
    assert _addresses(handler) == initial_addresses

    fresh_group = dist.new_group()
    fresh_backend = TorchDistBackend(fresh_group)
    handler.checkpoint_restore(fresh_backend)
    handler.checkpoint_restore(fresh_backend)
    assert handler._vm_mapped
    assert handler._vmm_comm_backend is fresh_backend
    assert _addresses(handler) == initial_addresses

    for key in op_modes:
        op, _mode = key
        expected = _set_input(op, inputs[key], rank, world_size, offset=4)
        graphs[key].replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[key], expected, rtol=0, atol=0)

    handler.checkpoint_prepare()
    handler.checkpoint_restore(fresh_backend)
    handler.shutdown()
    assert not handler.is_running
    assert not handler._vm_mapped

    detached_handler = _make_handler()
    detached_handler.checkpoint_prepare()
    detached_handler.checkpoint_prepare()
    detached_handler.shutdown()
    assert not detached_handler.is_running
    assert not detached_handler._vm_mapped

    dist.destroy_process_group(fresh_group)
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="checkpointable MixedComm requires two CUDA devices",
)
def test_graph_replay_after_mixed_comm_checkpoint_restore() -> None:
    try:
        multicast_attribute = (
            mixed_comm.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
        )
        multicast_supported = all(
            mixed_comm.checkCudaErrors(
                mixed_comm.cuda.cuDeviceGetAttribute(multicast_attribute, device)
            )
            for device in range(2)
        )
        peer_access_supported = all(
            src == dst or torch.cuda.can_device_access_peer(src, dst)
            for src in range(2)
            for dst in range(2)
        )
    except (AttributeError, RuntimeError):
        multicast_supported = False
        peer_access_supported = False
    if not multicast_supported or not peer_access_supported:
        pytest.skip("CUDA multicast and peer-access topology are required")

    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]
    if not run_mixed_comm.is_compute_capability_supported(compute_capability_number):
        pytest.skip(f"run_mixed_comm is not supported on sm{compute_capability_number}")

    mp.spawn(
        _run_checkpoint_worker,
        args=(2, _free_port()),
        nprocs=2,
        join=True,
    )
