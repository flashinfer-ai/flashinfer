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

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm import (
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
) -> None:
    passed = torch.tensor(
        [int(torch.equal(actual, expected))],
        dtype=torch.int32,
        device=actual.device,
    )
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if not passed.item():
        if torch.equal(actual, expected):
            pytest.fail("collective result mismatch on another rank")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


def _queued_epoch_stress(
    workspace: comm.PcieIpcAllGatherWorkspace,
    world_size: int,
    rank: int,
    device: torch.device,
    hidden: int,
) -> None:
    """Queue mixed variants/grids eagerly and in changed-input graph replays."""
    queued = []
    variants = tuple(
        config.variant for config in _configs(world_size, workspace.ordered_4plus4)
    )
    for step in range(4):
        rows = 1 + step % 2
        inp = (
            torch.arange(rows * hidden, dtype=torch.int32, device=device)
            .remainder_(13)
            .view(rows, hidden)
            .to(workspace.dtype)
        )
        inp.add_(rank * 3 + step)
        reference = _reference(inp, world_size)
        out = torch.empty_like(reference)
        variant = variants[step % len(variants)]
        copy_engine = variant == PcieIpcAllGatherVariant.COPY_ENGINE
        config = PcieIpcAllGatherLaunchConfig(
            1 if copy_engine or (step // len(variants)) % 2 == 0 else 3,
            32 if copy_engine else 64,
            variant,
        )
        queued.append((inp, out, reference, config))

    # Complete all NCCL references first. The custom calls below must stay
    # back-to-back so their epoch, parity slot, and changing grid state overlap.
    torch.cuda.synchronize(device)
    for inp, out, _, config in queued:
        workspace.all_gather(inp, out=out, config=config)
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
            workspace.all_gather(inp, out=out, config=config)
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
            with comm.PcieIpcAllGatherWorkspace(
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
                pack_elements = 16 // torch.empty((), dtype=dtype).element_size()
                # One pack, a non-block-aligned tail, and a normal shard.
                for shard_elements in (pack_elements, 129 * pack_elements, hidden):
                    inp = (
                        torch.arange(shard_elements, dtype=torch.int32, device=device)
                        .view(1, shard_elements)
                        .to(dtype)
                    )
                    inp.add_(rank)
                    reference = _reference(inp, world_size)
                    _assert_close_collectively(workspace.all_gather(inp), reference)
                    for config in _configs(world_size, workspace.ordered_4plus4):
                        out = torch.empty_like(reference)
                        assert workspace.all_gather(inp, out=out, config=config) is out
                        _assert_close_collectively(out, reference)

                if dtype is torch.bfloat16:
                    _queued_epoch_stress(workspace, world_size, rank, device, hidden)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_all_gather(world_size: int, tmp_path) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(
        world_size, _worker, args=(str(tmp_path / "tuning.json"),), timeout_s=900
    )
