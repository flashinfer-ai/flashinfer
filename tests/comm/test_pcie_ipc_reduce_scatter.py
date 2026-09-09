# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-GPU correctness tests for PCIe IPC ReduceScatter."""

import os
from contextlib import suppress

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.pcie_ipc_rs_policy import (
    PcieIpcReduceScatterLaunchConfig,
    PcieIpcReduceScatterVariant,
)
from tests.comm.test_pcie_ipc_all_reduce import (
    _init_process_group,
    multi_process_parallel,
)


_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_TOLERANCE = {
    torch.bfloat16: (2e-2, 2e-2),
    torch.float16: (5e-3, 5e-3),
    torch.float32: (1e-5, 1e-5),
}


def _assert_close_collectively(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float = 0,
    atol: float = 0,
) -> None:
    passed = torch.tensor(
        [int(torch.allclose(actual, expected, rtol=rtol, atol=atol))],
        dtype=torch.int32,
        device=actual.device,
    )
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if not passed.item():
        if torch.allclose(actual, expected, rtol=rtol, atol=atol):
            pytest.fail("collective result mismatch on another rank")
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _reference(inp: torch.Tensor, world_size: int) -> torch.Tensor:
    out = torch.empty(
        (inp.shape[0] // world_size, inp.shape[1]),
        dtype=inp.dtype,
        device=inp.device,
    )
    dist.reduce_scatter_tensor(out, inp)
    return out


def _fp32_reference(inp: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    reduced = inp.float()
    dist.all_reduce(reduced)
    local_rows = inp.shape[0] // world_size
    return reduced.narrow(0, rank * local_rows, local_rows).to(inp.dtype)


def _configs(world_size: int):
    if world_size in (2, 4):
        return [
            PcieIpcReduceScatterLaunchConfig(
                1, 128, PcieIpcReduceScatterVariant.FLAT_CYCLIC
            ),
            PcieIpcReduceScatterLaunchConfig(
                1, 128, PcieIpcReduceScatterVariant.FLAT_ONE_PACK
            ),
        ]
    return [
        PcieIpcReduceScatterLaunchConfig(
            1, 128, PcieIpcReduceScatterVariant.TOPOLOGY_CYCLIC
        ),
        PcieIpcReduceScatterLaunchConfig(
            1, 128, PcieIpcReduceScatterVariant.TOPOLOGY_ONE_PACK
        ),
    ]


def _remove_cache(path: str, rank: int) -> None:
    if rank == 0:
        with suppress(FileNotFoundError):
            os.unlink(path)
    dist.barrier()


def _worker(world_size: int, rank: int, port: int) -> None:
    _init_process_group(world_size, rank, port)
    device = torch.device(f"cuda:{rank}")
    hidden = 1024
    try:
        for dtype in _DTYPES:
            cache = f"/tmp/flashinfer_pcie_rs_{port}_{dtype}.json"
            _remove_cache(cache, rank)
            workspace = comm.PcieIpcReduceScatterWorkspace(
                dist.group.WORLD,
                max_numel=2 * hidden,
                dtype=dtype,
                tune_batches=(1,),
                tune_cache=cache,
            )
            try:
                if world_size == 8 and os.getenv(
                    "FLASHINFER_TEST_PCIE_IPC_ORDERED_4PLUS4"
                ):
                    assert workspace.ordered_4plus4, workspace.ordered_4plus4_reason
                if world_size == 8 and not workspace.ordered_4plus4:
                    assert not workspace.supports(
                        torch.empty((world_size, hidden), dtype=dtype, device=device)
                    )
                    return
                inp = (
                    torch.arange(world_size * hidden, dtype=torch.int32, device=device)
                    .remainder_(8)
                    .view(world_size, hidden)
                    .to(dtype)
                )
                inp.add_(rank)
                reference = _reference(inp, world_size)

                assert workspace.supports(inp)
                _assert_close_collectively(workspace.reduce_scatter(inp), reference)
                for config in _configs(world_size):
                    out = torch.empty_like(reference)
                    assert workspace.reduce_scatter(inp, out=out, config=config) is out
                    _assert_close_collectively(out, reference)

                pack_elements = 16 // inp.element_size()
                for shard_elements in (pack_elements, 129 * pack_elements):
                    edge_input = (
                        torch.arange(
                            world_size * shard_elements,
                            dtype=torch.int32,
                            device=device,
                        )
                        .remainder_(8)
                        .view(world_size, shard_elements)
                        .to(dtype)
                    )
                    edge_input.add_(rank)
                    edge_reference = _reference(edge_input, world_size)
                    for config in _configs(world_size):
                        _assert_close_collectively(
                            workspace.reduce_scatter(edge_input, config=config),
                            edge_reference,
                        )

                generator = torch.Generator(device=device).manual_seed(2026 + rank)
                floating = torch.randn(
                    (world_size * 2, hidden),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                rtol, atol = _TOLERANCE[dtype]
                _assert_close_collectively(
                    workspace.reduce_scatter(floating),
                    _fp32_reference(floating, world_size, rank),
                    rtol=rtol,
                    atol=atol,
                )

                if world_size == 2 and dtype is torch.bfloat16:
                    out = torch.empty_like(reference)
                    workspace.reduce_scatter(inp, out=out)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        workspace.reduce_scatter(inp, out=out)
                    inp.fill_(rank + 5)
                    reference = _reference(inp, world_size)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    _assert_close_collectively(out, reference)

                if world_size == 2 and dtype is torch.float32:
                    tuned = workspace.tune([hidden], warmup=1, repeat=2)
                    assert (hidden, 1) in tuned
                    assert workspace.tuned_launch_config(inp) == tuned[(hidden, 1)]
            finally:
                workspace.destroy()
            _remove_cache(cache, rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_reduce_scatter(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _worker, timeout_s=900)
