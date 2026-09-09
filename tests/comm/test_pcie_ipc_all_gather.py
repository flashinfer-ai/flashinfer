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

"""Multi-GPU correctness tests for PCIe IPC AllGather."""

import os
from contextlib import suppress

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.pcie_ipc_ag_policy import (
    PcieIpcAllGatherLaunchConfig,
    PcieIpcAllGatherVariant,
)
from tests.comm.test_pcie_ipc_all_reduce import (
    _init_process_group,
    multi_process_parallel,
)


_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


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
        (inp.shape[0] * world_size, inp.shape[1]),
        dtype=inp.dtype,
        device=inp.device,
    )
    dist.all_gather_into_tensor(out, inp)
    return out


def _configs(world_size: int, ordered_4plus4: bool):
    configs = [
        PcieIpcAllGatherLaunchConfig(1, 128, PcieIpcAllGatherVariant.RECURSIVE_DOUBLING)
    ]
    if world_size == 4:
        configs.append(
            PcieIpcAllGatherLaunchConfig(1, 128, PcieIpcAllGatherVariant.FLAT_PUSH)
        )
    if world_size == 8 and ordered_4plus4:
        configs.append(
            PcieIpcAllGatherLaunchConfig(1, 32, PcieIpcAllGatherVariant.COPY_ENGINE)
        )
    return configs


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
            cache = f"/tmp/flashinfer_pcie_ag_{port}_{dtype}.json"
            _remove_cache(cache, rank)
            workspace = comm.PcieIpcAllGatherWorkspace(
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
                inp = (
                    torch.arange(hidden, dtype=torch.int32, device=device)
                    .view(1, hidden)
                    .to(dtype)
                )
                inp.add_(rank)
                reference = _reference(inp, world_size)

                assert workspace.supports(inp)
                _assert_close_collectively(workspace.all_gather(inp), reference)
                for config in _configs(world_size, workspace.ordered_4plus4):
                    out = torch.empty_like(reference)
                    assert workspace.all_gather(inp, out=out, config=config) is out
                    _assert_close_collectively(out, reference)
                    if config.variant == PcieIpcAllGatherVariant.COPY_ENGINE:
                        inp.add_(3)
                        reference = _reference(inp, world_size)
                        workspace.all_gather(inp, out=out, config=config)
                        _assert_close_collectively(out, reference)

                pack_elements = 16 // inp.element_size()
                for shard_elements in (pack_elements, 129 * pack_elements):
                    edge_input = (
                        torch.arange(shard_elements, dtype=torch.int32, device=device)
                        .view(1, shard_elements)
                        .to(dtype)
                    )
                    edge_input.add_(rank)
                    edge_reference = _reference(edge_input, world_size)
                    for config in _configs(world_size, workspace.ordered_4plus4):
                        _assert_close_collectively(
                            workspace.all_gather(edge_input, config=config),
                            edge_reference,
                        )

                if world_size == 2 and dtype is torch.bfloat16:
                    out = torch.empty_like(reference)
                    workspace.all_gather(inp, out=out)
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        workspace.all_gather(inp, out=out)
                    inp.fill_(rank + 7)
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

            if world_size == 2 and dtype is torch.float32:
                reloaded = comm.PcieIpcAllGatherWorkspace(
                    dist.group.WORLD,
                    max_numel=2 * hidden,
                    dtype=dtype,
                    tune_batches=(1,),
                    tune_cache=cache,
                )
                try:
                    assert reloaded.tuned_launch_config(inp) == tuned[(hidden, 1)]
                    _assert_close_collectively(
                        reloaded.all_gather(inp), _reference(inp, world_size)
                    )
                finally:
                    reloaded.destroy()
            _remove_cache(cache, rank)

        if world_size == 2:
            with pytest.raises(ValueError, match="dtype"):
                comm.PcieIpcAllGatherWorkspace(
                    dist.group.WORLD, max_numel=hidden, dtype=torch.float64
                )
            with pytest.raises(ValueError, match="multiple of 4"):
                comm.PcieIpcAllGatherWorkspace(
                    dist.group.WORLD, max_numel=6, dtype=torch.float32
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_all_gather(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _worker, timeout_s=900)
