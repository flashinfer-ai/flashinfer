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

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm import (
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
    variants = (
        (
            PcieIpcReduceScatterVariant.FLAT_CYCLIC,
            PcieIpcReduceScatterVariant.FLAT_ONE_PACK,
        )
        if world_size in (2, 4)
        else (
            PcieIpcReduceScatterVariant.TOPOLOGY_CYCLIC,
            PcieIpcReduceScatterVariant.TOPOLOGY_ONE_PACK,
        )
    )
    return [PcieIpcReduceScatterLaunchConfig(1, 128, variant) for variant in variants]


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
    for step in range(4):
        rows = 1 + step % 2
        inp = torch.full(
            (world_size * rows, hidden),
            rank + step,
            dtype=workspace.dtype,
            device=device,
        )
        reference = _reference(inp, world_size)
        out = torch.empty_like(reference)
        config = PcieIpcReduceScatterLaunchConfig(
            1 if (step // len(variants)) % 2 == 0 else 3,
            64,
            variants[step % len(variants)],
        )
        queued.append((inp, out, reference, config))

    # Keep NCCL outside the back-to-back custom calls to avoid masking stale
    # workspace state when the variant, grid, or input changes.
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
        for replay in range(1, 4)
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


def _worker(world_size: int, rank: int, port: int, cache_path: str) -> None:
    _init_process_group(world_size, rank, port)
    device = torch.device(f"cuda:{rank}")
    hidden = 1024
    try:
        for dtype in _DTYPES:
            with comm.PcieIpcReduceScatterWorkspace(
                dist.group.WORLD,
                max_numel=2 * hidden,
                dtype=dtype,
                max_blocks=3,
                tune_cache=cache_path,
            ) as workspace:
                if world_size == 8 and os.getenv(
                    "FLASHINFER_TEST_PCIE_IPC_ORDERED_4PLUS4"
                ):
                    assert workspace.ordered_4plus4, workspace.ordered_4plus4_reason
                if world_size == 8 and not workspace.ordered_4plus4:
                    assert not workspace.supports(
                        torch.empty((world_size, hidden), dtype=dtype, device=device)
                    )
                    return
                pack_elements = 16 // torch.empty((), dtype=dtype).element_size()
                for rows, width in (
                    (1, pack_elements),
                    (1, 129 * pack_elements),
                    (1, hidden),
                    (2, hidden),
                ):
                    inp = (
                        torch.arange(
                            world_size * rows * width,
                            dtype=torch.int32,
                            device=device,
                        )
                        .remainder_(8)
                        .view(world_size * rows, width)
                        .to(dtype)
                    )
                    inp.add_(rank)
                    reference = _reference(inp, world_size)
                    _assert_close_collectively(workspace.reduce_scatter(inp), reference)
                    out = torch.empty_like(reference)
                    for config in _configs(world_size):
                        assert (
                            workspace.reduce_scatter(inp, out=out, config=config) is out
                        )
                        _assert_close_collectively(out, reference)

                generator = torch.Generator(device=device).manual_seed(2026 + rank)
                floating = torch.randn(
                    (world_size * 2, hidden),
                    dtype=dtype,
                    device=device,
                    generator=generator,
                )
                rtol, atol = _TOLERANCE[dtype]
                floating_reference = _fp32_reference(floating, world_size, rank)
                for config in (None, *_configs(world_size)):
                    _assert_close_collectively(
                        workspace.reduce_scatter(floating, config=config),
                        floating_reference,
                        rtol=rtol,
                        atol=atol,
                    )

                if dtype is torch.bfloat16:
                    _queued_epoch_stress(workspace, world_size, rank, device, hidden)

    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_reduce_scatter(world_size: int, tmp_path) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(
        world_size, _worker, args=(str(tmp_path / "tuning.json"),), timeout_s=900
    )
