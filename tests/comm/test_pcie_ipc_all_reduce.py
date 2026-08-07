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

import multiprocessing as mp
import socket
import time
from typing import Any

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.pcie_ipc_policy import IpcLaunchConfig
from flashinfer.comm.pcie_ipc_topology import PCIE_IPC_PROFILES

# (world_size, hidden, batch, blocks, threads, stream_mode, ring_push)
#
# Between them these cases select every kernel the header can dispatch to.
# Most mirror what the tuning table picks on a switch-free PCIe machine; the
# two marked "coverage" are configurations the table does not choose there but
# which remain reachable, so they are exercised rather than left untested.
_JOIN_TIMEOUT_S = 600

_CASES = [
    (2, 2048, 1, 32, 64, False, False),  # tp2, unstaged
    (2, 2048, 8, 96, 64, True, False),  # tp2, staged
    (2, 2048, 128, 16, 128, False, False),
    (4, 4096, 1, 1, 128, True, False),  # rsag push
    (4, 4096, 8, 1, 256, True, True),  # rsag ring push
    (4, 4096, 128, 4, 256, True, True),
    (4, 4096, 16, 8, 256, False, False),  # coverage: one-shot push
    (8, 6144, 1, 32, 128, False, False),  # topo pack (blocks % 4 == 0)
    (8, 6144, 8, 1, 256, True, True),  # topo ring push
    (8, 6144, 128, 2, 256, True, True),
    (8, 6144, 16, 8, 256, True, False),  # coverage: topo block (blocks % 4 == 0)
]


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]


def _init_process_group(world_size: int, rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )


def multi_process_parallel(
    world_size: int,
    target: Any,
    args: tuple = (),
    timeout_s: float = _JOIN_TIMEOUT_S,
) -> None:
    """Run ``target`` on ``world_size`` ranks, or fail with a bounded wait.

    ``timeout_s`` is lowered by the negative-control tests, which *expect* the
    deadlock and would otherwise pay the full timeout to observe it.
    """
    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=target, args=(world_size, rank, port) + args, name=f"Worker-{rank}"
        )
        p.start()
        procs.append(p)
    # A bounded join is essential here: the failure mode this suite is built to
    # catch is a spin-wait deadlock, and an unbounded join would hang CI rather
    # than report it.
    #
    # Every exit runs the reaper, not just the timeout one. When one rank dies
    # the others are left spinning inside a collective their peer will never
    # reach, so returning without them would leak processes that hold a GPU
    # context for the rest of the session -- and the next test in the file
    # would then fail for reasons that have nothing to do with it.
    deadline = time.monotonic() + timeout_s
    try:
        for rank, p in enumerate(procs):
            p.join(timeout=max(1.0, deadline - time.monotonic()))
            if p.is_alive():
                raise AssertionError(
                    f"rank {rank} did not finish within {timeout_s}s; "
                    "the collective most likely deadlocked"
                )
            assert p.exitcode == 0, f"rank {rank} failed with exit code {p.exitcode}"
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(timeout=30.0)


def _correctness_worker(
    world_size: int, rank: int, port: int, dtype: torch.dtype
) -> None:
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        cases = [c for c in _CASES if c[0] == world_size]
        max_numel = max(h * b for _, h, b, *_ in cases)

        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=max_numel, dtype=dtype, max_blocks=128
        )
        # Correctness must hold under either tuning table, so assert only that
        # a valid profile was resolved; which one is a property of the machine.
        assert ws.profile in PCIE_IPC_PROFILES, ws.profile_reason

        for _, hidden, batch, blocks, threads, stream_mode, ring_push in cases:
            shape = (batch, hidden)
            # Small integers so the reduction is exact in every dtype and the
            # comparison can use a zero tolerance; the kernel sums in a
            # different order than NCCL, which would otherwise show up.
            inp = torch.randint(0, 16, shape, dtype=torch.int32, device=device).to(
                dtype
            )
            # Snapshot before any call, and build the reference from the
            # snapshot: cloning after the fact would follow the input if the
            # kernel mutated it, hiding exactly what we mean to check.
            inp_before = inp.clone()
            ref = inp_before.clone()
            dist.all_reduce(ref, group=group)

            # Every case is a shape the tuning table covers, so the policy path
            # must accept it, and its result must be correct -- not merely the
            # right shape, or a mis-selected kernel would go unnoticed.
            assert ws.supports(inp)
            torch.testing.assert_close(ws.all_reduce(inp), ref, rtol=0, atol=0)

            # The explicit config then pins the kernel this case exists for.
            out = ws.all_reduce(
                inp,
                config=IpcLaunchConfig(blocks, threads, stream_mode, ring_push),
            )
            torch.testing.assert_close(out, ref, rtol=0, atol=0)

            # The sentinel rewrite must happen on a register copy, never in the
            # caller's buffer.
            assert torch.equal(inp, inp_before)
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


def _unsupported_shape_worker(world_size: int, rank: int, port: int) -> None:
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=8192, dtype=torch.bfloat16
        )

        # A shape the table does not cover (hidden 2048 needs 2 ranks, and
        # this group has 2, so use an uncovered hidden instead).
        assert not ws.supports(
            torch.empty(4, 4096, dtype=torch.bfloat16, device=device)
        )
        # Larger than the workspace.
        assert not ws.supports(torch.empty(16384, dtype=torch.bfloat16, device=device))
        # Element size does not match the workspace.
        assert not ws.supports(torch.empty(1024, dtype=torch.float32, device=device))
        # Byte size not a multiple of 16.
        assert not ws.supports(torch.empty(4, dtype=torch.bfloat16, device=device))
        # Non-contiguous.
        assert not ws.supports(
            torch.empty(64, 64, dtype=torch.bfloat16, device=device).t()
        )

        # A device mismatch is a caller bug, so it raises instead of reporting
        # "unsupported". Reporting unsupported would send this rank to another
        # backend while its peers stay here, and the group would hang.
        peer = torch.device(f"cuda:{(rank + 1) % world_size}")
        for wrong in (torch.device("cpu"), peer):
            with pytest.raises(ValueError, match="workspace was built on"):
                ws.supports(torch.empty(1024, dtype=torch.bfloat16, device=wrong))
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


def _cuda_graph_worker(world_size: int, rank: int, port: int) -> None:
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        hidden = {2: 2048, 4: 4096, 8: 6144}[world_size]
        batch = 8
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=hidden * batch, dtype=torch.bfloat16
        )
        inp = torch.randint(
            0, 16, (batch, hidden), dtype=torch.int32, device=device
        ).to(torch.bfloat16)
        out = torch.empty_like(inp)

        # Warm up outside the graph. The protocol state lives in device memory
        # and is read at kernel entry, so replays pick up whatever the previous
        # launch left; nothing is baked into the captured node.
        for _ in range(3):
            ws.all_reduce(inp, out=out)
        torch.cuda.synchronize()
        dist.barrier(group=group)

        ref = inp.clone()
        dist.all_reduce(ref, group=group)

        # Capture an odd count as well as an even one. Protocol state that
        # alternates per call returns to its starting parity after an even
        # capture, so only an odd capture exercises a replay that begins on the
        # opposite parity from the one before it.
        for captured in (3, 4):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(captured):
                    ws.all_reduce(inp, out=out)
            for _ in range(16):
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, ref, rtol=0, atol=0)
            dist.barrier(group=group)
            del graph
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


def _shape_change_worker(world_size: int, rank: int, port: int) -> None:
    """Interleave shapes with no intervening synchronisation.

    The epoch double buffer lets a rank start the next collective before its
    peer has drained the previous one, so a shape change is exactly when a
    staging offset derived from the current payload rather than from the
    workspace capacity would corrupt.

    Consecutive calls must carry *different* payloads. A reuse-too-early bug
    overwrites a scratch slot with another call's copy of the same reduction;
    if every call reduces the same tensor the two values are bit-identical and
    the corruption is invisible no matter how the calls are interleaved.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        hidden = {2: 2048, 4: 4096, 8: 6144}[world_size]
        batches = [1, 16, 1, 96, 1]
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=hidden * max(batches), dtype=torch.bfloat16
        )
        # Four payloads per shape, cycled, so a call never repeats the data of
        # the call two before it -- the distance at which the scratch half is
        # reused.
        variants = 4
        inputs = {
            b: [
                torch.randint(0, 16, (b, hidden), dtype=torch.int32, device=device).to(
                    torch.bfloat16
                )
                for _ in range(variants)
            ]
            for b in batches
        }
        refs = {}
        for b, xs in inputs.items():
            refs[b] = []
            for x in xs:
                r = x.clone()
                dist.all_reduce(r, group=group)
                refs[b].append(r)
        dist.barrier(group=group)

        outs = []
        for i in range(40):
            for b in batches:
                v = i % variants
                outs.append((b, v, ws.all_reduce(inputs[b][v])))
        torch.cuda.synchronize()
        for b, v, o in outs:
            torch.testing.assert_close(o, refs[b][v], rtol=0, atol=0)
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_cuda_graph(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _cuda_graph_worker)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_pcie_ipc_shape_change(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _shape_change_worker)


# Batches chosen against the rootcplx TP4 table, where 8 -> 1 block and 9, 10,
# 12 -> 2 blocks. Each pair therefore grows the grid; 12 is the control.
_GRID_CHANGE_BATCHES = [8, 9, 8, 10, 8, 12, 9]


def _grid_change_worker(world_size: int, rank: int, port: int) -> None:
    """Change the grid size between calls, which used to be an instant hang.

    The epoch used to be per block: each block flipped its own parity on exit,
    so the parity recorded how many times *that block* had run. Grow the grid
    and a block running for the first time reads the initial 0 and picks the
    half the previous call just used, with no slack at all.

    The sequence ends with ``12, 9`` deliberately. Those two calls have the
    same grid, so nothing about *them* is unusual -- but the earlier growth has
    already desynchronised the block ranges, and under the old scheme they hung
    anyway. One grid change was enough to arm it permanently, which is why
    testing only at the boundary would not be enough.

    Deterministic: no stall, no instrumentation, no race to lose. The failure
    was a hang rather than a wrong answer, because in this kernel the racing
    write lands on the address the victim polls next.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        hidden = 4096
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=hidden * max(_GRID_CHANGE_BATCHES),
            dtype=torch.bfloat16,
            profile="rootcplx",  # the batch->blocks mapping above is this table's
        )
        # Distinct payloads: a repeated one makes a mis-addressed write store a
        # bit-identical value, hiding any corruption that does not also hang.
        inputs = [
            torch.randint(1, 16, (b, hidden), dtype=torch.int32, device=device).to(
                torch.bfloat16
            )
            + i
            for i, b in enumerate(_GRID_CHANGE_BATCHES)
        ]
        refs = []
        for x in inputs:
            r = x.clone()
            dist.all_reduce(r, group=group)
            refs.append(r)
        dist.barrier(group=group)

        # Nothing drains the queue between calls -- that is the point.
        outs = [ws.all_reduce(x) for x in inputs]
        for out, ref, batch in zip(outs, refs, _GRID_CHANGE_BATCHES, strict=True):
            torch.testing.assert_close(
                out, ref, rtol=0, atol=0, msg=f"batch {batch} in the grid-change run"
            )
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [4])
def test_pcie_ipc_grid_change(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _grid_change_worker)


def _single_block_epoch_worker(world_size: int, rank: int, port: int) -> None:
    """A one-block launch must leave the epoch exactly where a larger one would.

    ``gridDim.x == 1`` skips the arrival atomic -- the sole CTA is trivially the
    last arrival -- and flips the epoch directly. That is only sound if the two
    paths leave identical state, which is invisible from outside except through
    what the *next* call reads. Alternating the two therefore checks it: if the
    fast path left the epoch unflipped, or flipped twice, a following call would
    land on the half its predecessor just used.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        hidden = 4096
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=hidden * 8, dtype=torch.bfloat16, profile="rootcplx"
        )
        one = IpcLaunchConfig(1, 256, True, True)
        many = IpcLaunchConfig(4, 256, True, True)
        payloads = [
            torch.randint(1, 16, (8, hidden), dtype=torch.int32, device=device).to(
                torch.bfloat16
            )
            + i
            for i in range(4)
        ]
        refs = []
        for x in payloads:
            r = x.clone()
            dist.all_reduce(r, group=group)
            refs.append(r)
        dist.barrier(group=group)

        # Every ordering of the two paths, back to back, nothing draining.
        outs = []
        for cfg_a, cfg_b in ((one, many), (many, one), (one, one), (many, many)):
            for cfg, x in zip((cfg_a, cfg_b), payloads[:2], strict=True):
                outs.append((ws.all_reduce(x, config=cfg), cfg))
        for i, (out, cfg) in enumerate(outs):
            torch.testing.assert_close(
                out, refs[i % 2], rtol=0, atol=0, msg=f"call {i} at blocks={cfg.blocks}"
            )
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [4])
def test_pcie_ipc_single_block_epoch_matches(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _single_block_epoch_worker)


def _switchpair_region_worker(world_size: int, rank: int, port: int) -> None:
    """Alternate the two scratch regions under the switch-paired table.

    That table is the only one whose batch ranges put the pack kernel and a
    block kernel next to each other, so it is the only way to exercise the two
    call-level epoch counters against each other -- and the only way to catch a
    state pointer bound to the wrong region, which would otherwise show up on no
    machine we test on.

    Forcing the profile rather than probing is the point: the machine this runs
    on resolves to rootcplx, where these batches never cross regions.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        hidden = 4096
        # 96 -> rsag_push (block region), 1 -> push_oneshot (pack region).
        batches = [96, 1, 96, 1, 1, 96]
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group,
            max_numel=hidden * max(batches),
            dtype=torch.bfloat16,
            profile="pcieswitch",
        )
        seen_regions = set()
        payloads, refs = [], []
        for i, b in enumerate(batches):
            cfg = ws.launch_config(
                torch.empty(b, hidden, dtype=torch.bfloat16, device=device)
            )
            assert cfg is not None, f"batch {b} unexpectedly untuned"
            seen_regions.add(cfg.stream_mode)
            x = (
                torch.randint(1, 16, (b, hidden), dtype=torch.int32, device=device).to(
                    torch.bfloat16
                )
                + i
            )
            r = x.clone()
            dist.all_reduce(r, group=group)
            payloads.append(x)
            refs.append(r)
        # If the table stopped straddling the region boundary here, the test
        # would still pass while checking nothing.
        assert seen_regions == {True, False}, "batches no longer cross both regions"
        dist.barrier(group=group)

        for x, ref, b in zip(payloads, refs, batches, strict=True):
            torch.testing.assert_close(
                ws.all_reduce(x), ref, rtol=0, atol=0, msg=f"batch {b}"
            )
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [4])
def test_pcie_ipc_switchpair_region_alternation(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _switchpair_region_worker)


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pcie_ipc_all_reduce(world_size: int, dtype: torch.dtype) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"world_size {world_size} exceeds the {torch.cuda.device_count()} available GPUs"
        )
    multi_process_parallel(world_size, _correctness_worker, (dtype,))


def _second_stream_worker(world_size: int, rank: int, port: int) -> None:
    """A workspace is bound to one stream; the second one must be rejected.

    The epoch and arrival counters that drive the scratch double buffer are
    advanced by the kernels and only well defined if the calls sharing the
    workspace are totally ordered. Two streams do not give that.
    """
    ws = None
    group = None
    try:
        _init_process_group(world_size, rank, port)
        group = dist.group.WORLD
        device = torch.device(f"cuda:{rank}")
        hidden = {2: 2048, 4: 4096, 8: 6144}[world_size]
        ws = comm.PcieIpcAllReduceWorkspace(
            group=group, max_numel=hidden * 8, dtype=torch.bfloat16
        )
        inp = torch.randn(8, hidden, dtype=torch.bfloat16, device=device)

        ws.all_reduce(inp)  # binds to the current stream
        ws.all_reduce(inp)  # same stream, still fine

        other = torch.cuda.Stream(device=device)
        with (
            torch.cuda.stream(other),
            pytest.raises(RuntimeError, match="already bound to"),
        ):
            ws.all_reduce(inp)
        torch.cuda.synchronize(device)

        # And the escape hatch works: once the caller has ordered the two
        # streams, rebind_stream() lets the move through and the results stay
        # correct. Rejecting the second stream would be no use if the only way
        # past it were to build another workspace.
        ref = inp.clone()
        dist.all_reduce(ref, group=group)
        ws.rebind_stream()
        with torch.cuda.stream(other):
            torch.testing.assert_close(ws.all_reduce(inp), ref, rtol=0, atol=0)
            torch.testing.assert_close(ws.all_reduce(inp), ref, rtol=0, atol=0)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)
    finally:
        if ws is not None:
            ws.destroy()
        if group is not None:
            dist.destroy_process_group(group)


@pytest.mark.parametrize("world_size", [2])
def test_pcie_ipc_second_stream_rejected(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _second_stream_worker)


@pytest.mark.parametrize("world_size", [2])
def test_pcie_ipc_unsupported_shapes(world_size: int) -> None:
    if world_size > torch.cuda.device_count():
        pytest.skip("not enough GPUs")
    multi_process_parallel(world_size, _unsupported_shape_worker)
