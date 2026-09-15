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
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm._pcie_ipc_ag_rs_module import get_pcie_ipc_ag_rs_module
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
    reduced = inp.to(torch.float32, copy=True)
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


def _queued_epoch_stress(
    workspace: comm.PcieIpcReduceScatterWorkspace,
    world_size: int,
    rank: int,
    device: torch.device,
    hidden: int,
) -> None:
    """Queue mixed variants/grids eagerly and in changed-input graph replays."""
    queued = []
    variants = tuple(config.variant for config in _configs(world_size))
    for step in range(10):
        rows = 1 + step % 2
        inp = (
            torch.arange(
                world_size * rows * hidden,
                dtype=torch.int32,
                device=device,
            )
            .remainder_(8)
            .view(world_size * rows, hidden)
            .to(workspace.dtype)
        )
        inp.add_(rank + step)
        reference = _reference(inp, world_size)
        out = torch.empty_like(reference)
        config = PcieIpcReduceScatterLaunchConfig(
            1 if (step // len(variants)) % 2 == 0 else 3,
            64,
            variants[step % len(variants)],
        )
        queued.append((inp, out, reference, config))

    # Complete all NCCL references first.  The ten custom calls below must stay
    # back-to-back so their epoch, parity slot, and changing grid state overlap.
    torch.cuda.synchronize(device)
    for inp, out, _, config in queued:
        workspace.reduce_scatter(inp, out=out, config=config)
    torch.cuda.synchronize(device)
    _assert_close_collectively(
        torch.cat([out.flatten() for _, out, _, _ in queued]),
        torch.cat([reference.flatten() for _, _, reference, _ in queued]),
    )

    # Build references before capture/replay so no NCCL operation orders the
    # custom launches. Keep a separate output snapshot for every queued replay.
    expected = [
        [_reference(inp + replay, world_size) for inp, _, _, _ in queued]
        for replay in range(1, 5)
    ]
    observed = [[torch.empty_like(out) for _, out, _, _ in queued] for _ in expected]
    torch.cuda.synchronize(device)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for inp, out, _, config in queued:
            out.fill_(float("nan"))
            workspace.reduce_scatter(inp, out=out, config=config)
    for snapshots in observed:
        for inp, _, _, _ in queued:
            inp.add_(1)
        graph.replay()
        for snapshot, (_, out, _, _) in zip(snapshots, queued, strict=True):
            snapshot.copy_(out)
    torch.cuda.synchronize(device)
    _assert_close_collectively(
        torch.cat([out.flatten() for replay in observed for out in replay]),
        torch.cat([out.flatten() for replay in expected for out in replay]),
    )


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
                max_blocks=3,
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
                floating_reference = _fp32_reference(floating, world_size, rank)
                _assert_close_collectively(
                    workspace.reduce_scatter(floating),
                    floating_reference,
                    rtol=rtol,
                    atol=atol,
                )
                for config in _configs(world_size):
                    _assert_close_collectively(
                        workspace.reduce_scatter(floating, config=config),
                        floating_reference,
                        rtol=rtol,
                        atol=atol,
                    )

                if world_size in (4, 8) and dtype is torch.bfloat16:
                    _queued_epoch_stress(workspace, world_size, rank, device, hidden)

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

                if dtype is torch.float32:
                    state = workspace._tuning_state
                    with patch.object(
                        state, "_latency_us", wraps=state._latency_us
                    ) as measured:
                        tuned = workspace.tune([hidden], warmup=1, repeat=2)
                    # A variant reaches timing only after real GPU correctness
                    # screening. max_blocks=3 keeps this smoke search bounded.
                    assert {
                        call.args[2].variant for call in measured.call_args_list
                    } == {config.variant for config in _configs(world_size)}
                    assert (hidden, 1) in tuned
                    assert workspace.tuned_launch_config(inp) == tuned[(hidden, 1)]
            finally:
                workspace.destroy()
            _remove_cache(cache, rank)

        if world_size == 2:
            with pytest.raises(RuntimeError, match="workspace layout overflows"):
                get_pcie_ipc_ag_rs_module().reduce_scatter_workspace_size(
                    2, 1 << 62, 4, 64
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_reduce_scatter(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _worker, timeout_s=900)
