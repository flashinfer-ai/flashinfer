"""Distributed correctness of the NVFP4-transport all-reduce (world size 2 one-shot, world size 4 two-shot)."""

from __future__ import annotations

import multiprocessing as mp
import socket
import time

import pytest
import torch
import torch.distributed as dist

from flashinfer.comm.fp4_allreduce import (
    Fp4AllReduceFusedNormOutput,
    Fp4AllReduceFusedNormSpec,
    Fp4AllReduceWorkspace,
    fp4_allreduce,
    fp4_allreduce_nvfp4,
    fp4_allreduce_rmsnorm,
    window_plan,
)
from flashinfer.jit.fp4_allreduce import CAPACITY, HIDDEN, RECORD_BYTES
from tests.comm import fp4_allreduce_reference as codec

ROWS = {2: (64, 1000, 1001, 4096), 4: (64, 1000, 4096)}
SUPPORTED_CAPABILITIES = {(10, 0), (10, 3), (10, 7)}
PROCESS_JOIN_TIMEOUT_S = 2400.0  # the first run compiles up to 24 generated programs
PROCESS_CLEANUP_TIMEOUT_S = 30.0
SEED = 1000


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _gather(tensor: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    peers = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(peers, tensor.contiguous(), group=group)
    return peers


def _check(out: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if out.shape != expected.shape or out.dtype != expected.dtype:
        raise AssertionError(
            f"{label}: got {tuple(out.shape)} {out.dtype}, expected {tuple(expected.shape)} {expected.dtype}"
        )
    if not torch.equal(out, expected):
        differing = int((out != expected).sum().item())
        raise AssertionError(
            f"{label}: {differing} of {out.numel()} elements differ from the codec reference"
        )


def _check_identical_across_ranks(
    out: torch.Tensor, group: dist.ProcessGroup, label: str
) -> None:
    for peer_rank, peer in enumerate(_gather(out, group)):
        if not torch.equal(peer, out):
            raise AssertionError(
                f"{label}: rank {dist.get_rank(group)} and rank {peer_rank} disagree"
            )


def _expect(exc_type: type, fn, label: str) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001 - the test reports the wrong type
        raise AssertionError(
            f"{label}: raised {type(exc).__name__} instead of {exc_type.__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}: no {exc_type.__name__} was raised")


def _bf16_cases(workspace, world_size, rank, device, group):
    for launch_with_pdl in (False, True):
        for rows in ROWS[world_size]:
            x = codec.generate_rank_input(rows, rank, SEED + 17 * rows, device)
            expected = codec.reference_allreduce(_gather(x, group))
            out = fp4_allreduce(x, workspace, launch_with_pdl=launch_with_pdl)
            torch.cuda.synchronize(device)
            label = f"bf16 rows={rows} pdl={launch_with_pdl}"
            _check(out, expected, label)
            _check_identical_across_ranks(out, group, label)
            assert workspace.kernel_name("bf16", rows).startswith(
                f"kernel_fp4_allreduce_tp{world_size}_"
            )


def _back_to_back(workspace, world_size, rank, device, group):
    """Four calls without host synchronisation: the three-slot rotation wraps around once."""
    rows = ROWS[world_size][-1]
    inputs = [
        codec.generate_rank_input(rows, rank, SEED + 7777 * call, device)
        for call in range(4)
    ]
    expected = [codec.reference_allreduce(_gather(x, group)) for x in inputs]
    outputs = [
        torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device) for _ in inputs
    ]
    for x, out in zip(inputs, outputs, strict=True):
        fp4_allreduce(x, workspace, out=out)
    torch.cuda.synchronize(device)
    for call, (out, ref) in enumerate(zip(outputs, expected, strict=True)):
        _check(out, ref, f"back-to-back call {call}")


def _alternating_rows(workspace, world_size, rank, device, group):
    """Changing row counts between calls: the in-kernel reset must cover the previous call's ranges."""
    sequence = (
        ROWS[world_size][-1],
        ROWS[world_size][0],
        ROWS[world_size][-1],
        ROWS[world_size][1],
    )
    for call, rows in enumerate(sequence):
        x = codec.generate_rank_input(rows, rank, SEED + 31 * call, device)
        expected = codec.reference_allreduce(_gather(x, group))
        out = fp4_allreduce(x, workspace)
        torch.cuda.synchronize(device)
        _check(out, expected, f"alternating rows call {call} rows={rows}")


def _nvfp4_records(workspace, world_size, rank, device, group):
    for rows in ROWS[world_size][1:]:
        x = codec.generate_rank_input(rows, rank, SEED + 101 * rows, device)
        scale = codec.tensor_global_scale(x)
        records = codec.encode_nvfp4_records(x, scale)
        scales = [
            float(value.item())
            for value in _gather(
                torch.tensor([scale], dtype=torch.float32, device=device), group
            )
        ]
        expected = codec.reference_allreduce_nvfp4(_gather(records, group), scales)
        out = fp4_allreduce_nvfp4(
            records, torch.tensor(scales, dtype=torch.float32, device=device), workspace
        )
        torch.cuda.synchronize(device)
        label = f"nvfp4 rows={rows}"
        _check(out, expected, label)
        _check_identical_across_ranks(out, group, label)


def _graph_replay(workspace, world_size, rank, device, group):
    rows = ROWS[world_size][1]
    x_static = codec.generate_rank_input(rows, rank, SEED + 4242, device)
    out_static = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
    expected = codec.reference_allreduce(_gather(x_static, group))
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        fp4_allreduce(x_static, workspace, out=out_static)
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for _ in range(3):
            fp4_allreduce(x_static, workspace, out=out_static)
    for replay in range(3):
        out_static.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
        _check(out_static, expected, f"graph replay {replay}")
    # Eager calls after the replays continue the same slot rotation.
    out = fp4_allreduce(x_static, workspace)
    torch.cuda.synchronize(device)
    _check(out, expected, "eager call after graph replays")
    del graph


def _negative_control(workspace, world_size, rank, device, group):
    """A perturbed partial on the last rank must change every rank's result."""
    rows = ROWS[world_size][1]
    x = codec.generate_rank_input(rows, rank, SEED + 99, device)
    expected = codec.reference_allreduce(_gather(x, group))
    perturbed = x.clone()
    if rank == world_size - 1:
        perturbed[rows // 2, HIDDEN // 2] = perturbed[rows // 2, HIDDEN // 2] + 1000.0
    out = fp4_allreduce(perturbed, workspace)
    torch.cuda.synchronize(device)
    if torch.equal(out, expected):
        raise AssertionError(
            "the perturbed partial did not change the all-reduce result"
        )


def _validation(workspace, world_size, device):
    bf16 = dict(dtype=torch.bfloat16, device=device)
    _expect(
        ValueError,
        lambda: fp4_allreduce(
            torch.zeros((64, HIDDEN), dtype=torch.float32, device=device), workspace
        ),
        "float32 input",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(torch.zeros((64, HIDDEN - 1), **bf16), workspace),
        "hidden size",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(torch.zeros((CAPACITY + 1, HIDDEN), **bf16), workspace),
        "rows above capacity",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(torch.zeros((64, 2 * HIDDEN), **bf16)[:, ::2], workspace),
        "non-contiguous input",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(torch.zeros((0, HIDDEN), **bf16), workspace),
        "zero rows",
    )
    if world_size == 4:
        _expect(
            ValueError,
            lambda: fp4_allreduce(torch.zeros((6, HIDDEN), **bf16), workspace),
            "rows not a multiple of 4",
        )
    x = torch.zeros((64, HIDDEN), **bf16)
    _expect(
        ValueError,
        lambda: fp4_allreduce(
            x,
            workspace,
            out=torch.zeros((64, HIDDEN), dtype=torch.float32, device=device),
        ),
        "float32 out",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce(x, workspace, out=torch.zeros((65, HIDDEN), **bf16)),
        "out rows",
    )
    scales = torch.ones((world_size,), dtype=torch.float32, device=device)
    _expect(
        ValueError,
        lambda: fp4_allreduce_nvfp4(
            torch.zeros((64, RECORD_BYTES - 1), dtype=torch.uint8, device=device),
            scales,
            workspace,
        ),
        "record width",
    )
    _expect(
        ValueError,
        lambda: fp4_allreduce_nvfp4(
            torch.zeros((64, RECORD_BYTES), dtype=torch.uint8, device=device),
            scales[:-1],
            workspace,
        ),
        "scale count",
    )
    _expect(TypeError, lambda: fp4_allreduce(x, object()), "workspace type")
    spec = Fp4AllReduceFusedNormSpec(
        weight=torch.ones((HIDDEN,), **bf16),
        eps=1e-6,
        output=Fp4AllReduceFusedNormOutput.NVFP4,
    )
    _expect(
        NotImplementedError,
        lambda: fp4_allreduce_rmsnorm(x, x, spec, workspace),
        "fused norm entry point",
    )


def _check_workspace(workspace, world_size, device):
    plan = window_plan(world_size)
    assert workspace.world_size == world_size
    assert workspace.capacity == CAPACITY
    assert workspace.table.numel() == plan.table_entries
    assert workspace.table.dtype == torch.int64
    assert workspace.ctrl.numel() == plan.ctrl_words
    assert workspace.ctrl.dtype == torch.int32
    assert (
        workspace.grid == torch.cuda.get_device_properties(device).multi_processor_count
    )
    assert all(int(pointer) != 0 for pointer in workspace.table.tolist())
    for rows in ROWS[world_size]:
        assert 0 <= workspace.variant(rows) < 3


def _worker(world_size: int, rank: int, port: int) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD
    workspace = Fp4AllReduceWorkspace(group, device=device)
    try:
        _check_workspace(workspace, world_size, device)
        _bf16_cases(workspace, world_size, rank, device, group)
        _back_to_back(workspace, world_size, rank, device, group)
        _alternating_rows(workspace, world_size, rank, device, group)
        _nvfp4_records(workspace, world_size, rank, device, group)
        _graph_replay(workspace, world_size, rank, device, group)
        _negative_control(workspace, world_size, rank, device, group)
        _validation(workspace, world_size, device)
    finally:
        dist.barrier(group=group)
        workspace.destroy()
        workspace.destroy()  # a second destroy is a no-op
    _expect(
        RuntimeError,
        lambda: fp4_allreduce(
            torch.zeros((64, HIDDEN), dtype=torch.bfloat16, device=device), workspace
        ),
        "call after destroy",
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_fp4_allreduce(world_size: int) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"test requires {world_size} CUDA devices")
    capabilities = {
        torch.cuda.get_device_capability(index) for index in range(world_size)
    }
    if len(capabilities) != 1 or not capabilities.issubset(SUPPORTED_CAPABILITIES):
        pytest.skip("test requires one homogeneous SM100, SM103 or SM107 node")

    port = _open_port()
    context = mp.get_context("spawn")
    processes = [
        context.Process(target=_worker, args=(world_size, rank, port))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + PROCESS_JOIN_TIMEOUT_S
    try:
        for rank, process in enumerate(processes):
            process.join(timeout=max(1.0, deadline - time.monotonic()))
            if process.is_alive():
                raise AssertionError(
                    f"rank {rank} did not finish within {PROCESS_JOIN_TIMEOUT_S}s"
                )
            assert process.exitcode == 0, f"rank {rank} exited with {process.exitcode}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=PROCESS_CLEANUP_TIMEOUT_S)
            if process.is_alive():
                process.kill()
                process.join(timeout=PROCESS_CLEANUP_TIMEOUT_S)
